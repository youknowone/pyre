//! Flow graph construction context.
//!
//! RPython basis: `rpython/flowspace/flowcontext.py`.
//!
//! This file ports the core control-flow machinery used by
//! `framestate.rs`: translator-level `FlowSignal`s, interpreter
//! block-stack entries, recorder/replayer helpers, and the
//! `FlowContext` state machine that snapshots and restores
//! `FrameState`s while building graphs.
//!
//! ## Deviations from upstream (parity rule #1)
//!
//! * **Dispatch shape.** Upstream uses `@FlowContext.opcode('LOAD_FAST')`
//!   decorator-based per-opcode methods (`flowcontext.py:279` onward).
//!   Rust has no runtime class decorator, so opcode dispatch collapses
//!   into a single `match instruction { … }` inside
//!   `FlowContext::handle_bytecode`. The arms remain in upstream
//!   opcode-method order where the source opcode name is stable across
//!   PyPy-2.7 and CPython 3.14.
//!
//! * **CPython 3.14 opcode fusion.** Roadmap decision D4 called for
//!   `flowcontext_py314.rs` to host the 3.14-only handlers (`PushNull`,
//!   `Resume`, `Cache`, `LoadCommonConstant`, `LoadSmallInt`,
//!   `LoadFast{Borrow,Check,AndClear,LoadFast,BorrowLoadFastBorrow}`,
//!   `BinaryOp`, `PopJumpForwardIf*`, `JumpBackward`, `Copy`, `Swap`,
//!   `Send`, `ReturnGenerator`, `Instrumented*`). F3.4-F3.7 landed them
//!   inline in this file because the decoder returns a single
//!   `Instruction` enum and splitting requires a follow-up refactor to
//!   share the match arm or hand off a `(dispatch, kind)` pair. PYRE-ONLY
//!   arms carry a `// PYRE-ONLY` marker comment next to the arm; the
//!   planned file split is tracked as a Phase 3 follow-up.
//!
//! * **`FlowContextError` enum.** Upstream uses `raise FlowingError` /
//!   `raise StopFlowing` / `raise Raise(…)` / `raise BytecodeCorruption`
//!   to unwind. Rust has no structural raise; each upstream raise is
//!   converted to `Err(FlowContextError::<variant>)` so callers can
//!   pattern-match on the raise kind. Semantics match upstream's
//!   flow-unwind; name is Rust-local.
//!
//! * **`PendingBlock` enum.** Upstream's `ctx.pendingblocks` stores both
//!   `SpamBlock` and `EggBlock` instances in one list via Python's duck
//!   typing. Rust closes the union as
//!   `enum PendingBlock { Spam(SpamBlock), Egg(EggBlock) }`.

use std::collections::{HashMap, VecDeque};

use pyre_interpreter::bytecode::oparg::VarNums;

use crate::argument::{CallShape, CallSpec};
use crate::bytecode::{
    BinaryOperator, BytecodeCorruption, ComparisonOperator, ConstantData, ExceptionTableEntry,
    HostCode, Instruction, MakeFunctionFlag,
};
use crate::framestate::{FrameState, StackElem};
use crate::model::{
    Block, BlockRef, BlockRefExt, BuiltinFunction, BuiltinObject, ConstValue, Constant,
    ExceptionClass, FSException, FunctionGraph, GraphFunc, Hlvalue, Link, SpaceOperation, Variable,
    c_last_exception,
};
use crate::specialcase::{builtin_function, lookup_builtin};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlowingError {
    pub message: String,
}

impl FlowingError {
    pub fn new(message: impl Into<String>) -> Self {
        FlowingError {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for FlowingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FlowingError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StopFlowing;

impl std::fmt::Display for StopFlowing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StopFlowing")
    }
}

impl std::error::Error for StopFlowing {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlowSignal {
    Return { w_value: Hlvalue },
    Raise { w_exc: FSException },
    RaiseImplicit { w_exc: FSException },
    Break,
    Continue { jump_to: i64 },
}

impl FlowSignal {
    pub fn args(&self) -> Vec<Hlvalue> {
        match self {
            FlowSignal::Return { w_value } => vec![w_value.clone()],
            FlowSignal::Raise { w_exc } | FlowSignal::RaiseImplicit { w_exc } => {
                vec![w_exc.w_type.clone(), w_exc.w_value.clone()]
            }
            FlowSignal::Break => Vec::new(),
            FlowSignal::Continue { jump_to } => {
                vec![Hlvalue::Constant(Constant::new(ConstValue::Int(*jump_to)))]
            }
        }
    }

    pub fn rebuild_with_args(tag: FlowSignalTag, args: Vec<Hlvalue>) -> FlowSignal {
        match tag {
            FlowSignalTag::Return => {
                assert_eq!(args.len(), 1, "Return.rebuild takes 1 arg");
                FlowSignal::Return {
                    w_value: args.into_iter().next().unwrap(),
                }
            }
            FlowSignalTag::Raise => {
                assert_eq!(args.len(), 2, "Raise.rebuild takes 2 args");
                let mut it = args.into_iter();
                FlowSignal::Raise {
                    w_exc: FSException::new(it.next().unwrap(), it.next().unwrap()),
                }
            }
            FlowSignalTag::RaiseImplicit => {
                assert_eq!(args.len(), 2, "RaiseImplicit.rebuild takes 2 args");
                let mut it = args.into_iter();
                FlowSignal::RaiseImplicit {
                    w_exc: FSException::new(it.next().unwrap(), it.next().unwrap()),
                }
            }
            FlowSignalTag::Break => {
                assert!(args.is_empty(), "Break.rebuild takes 0 args");
                FlowSignal::Break
            }
            FlowSignalTag::Continue => {
                assert_eq!(args.len(), 1, "Continue.rebuild takes 1 arg");
                let arg = args.into_iter().next().unwrap();
                let jump_to = match arg {
                    Hlvalue::Constant(Constant {
                        value: ConstValue::Int(n),
                        ..
                    }) => n,
                    other => panic!("Continue.rebuild expects Constant(Int), got {other:?}"),
                };
                FlowSignal::Continue { jump_to }
            }
        }
    }

    pub fn tag(&self) -> FlowSignalTag {
        match self {
            FlowSignal::Return { .. } => FlowSignalTag::Return,
            FlowSignal::Raise { .. } => FlowSignalTag::Raise,
            FlowSignal::RaiseImplicit { .. } => FlowSignalTag::RaiseImplicit,
            FlowSignal::Break => FlowSignalTag::Break,
            FlowSignal::Continue { .. } => FlowSignalTag::Continue,
        }
    }

    pub fn nomoreblocks(&self, ctx: &mut FlowContext) -> Result<i64, FlowContextError> {
        match self {
            FlowSignal::Return { w_value } => {
                let link = Link::new(
                    vec![w_value.clone()],
                    Some(ctx.graph.returnblock.clone()),
                    None,
                )
                .into_ref();
                ctx.current_block().closeblock(vec![link]);
                Err(FlowContextError::StopFlowing)
            }
            FlowSignal::Raise { w_exc } => {
                if matches!(
                    w_exc.w_type,
                    Hlvalue::Constant(Constant {
                        value: ConstValue::ExceptionClass(ref class),
                        ..
                    }) if class.name == "ImportError"
                ) {
                    let message = exception_message(&w_exc.w_value);
                    return Err(FlowContextError::Flowing(FlowingError::new(format!(
                        "ImportError is raised in RPython: {message}"
                    ))));
                }
                let link = Link::new(
                    vec![w_exc.w_type.clone(), w_exc.w_value.clone()],
                    Some(ctx.graph.exceptblock.clone()),
                    None,
                )
                .into_ref();
                ctx.current_block().closeblock(vec![link]);
                Err(FlowContextError::StopFlowing)
            }
            FlowSignal::RaiseImplicit { w_exc } => {
                let exc_cls = exception_class_name(&w_exc.w_type).unwrap_or("Exception");
                let message = format!("implicit {exc_cls} shouldn't occur");
                let link = Link::new(
                    vec![
                        exception_class_value("AssertionError"),
                        exception_instance_value("AssertionError", Some(message)),
                    ],
                    Some(ctx.graph.exceptblock.clone()),
                    None,
                )
                .into_ref();
                ctx.current_block().closeblock(vec![link]);
                Err(FlowContextError::StopFlowing)
            }
            _ => Err(FlowContextError::BytecodeCorruption(
                BytecodeCorruption::new("misplaced bytecode - should not return"),
            )),
        }
    }
}

fn exception_class_value(name: &str) -> Hlvalue {
    Hlvalue::Constant(Constant::new(ConstValue::ExceptionClass(
        ExceptionClass::builtin(name),
    )))
}

fn exception_instance_value(name: &str, message: Option<String>) -> Hlvalue {
    Hlvalue::Constant(Constant::new(ConstValue::ExceptionInstance {
        class: ExceptionClass::builtin(name),
        message,
    }))
}

fn exception_message(w_exc_value: &Hlvalue) -> String {
    match w_exc_value {
        Hlvalue::Constant(Constant {
            value:
                ConstValue::ExceptionInstance {
                    message: Some(message),
                    ..
                },
            ..
        }) => message.clone(),
        Hlvalue::Constant(Constant {
            value: ConstValue::Str(message),
            ..
        }) => message.clone(),
        _ => "<not a constant message>".to_owned(),
    }
}

fn exception_class_name(w_value: &Hlvalue) -> Option<&str> {
    match w_value {
        Hlvalue::Constant(Constant {
            value: ConstValue::ExceptionClass(class),
            ..
        }) => Some(class.name.as_str()),
        Hlvalue::Constant(Constant {
            value: ConstValue::ExceptionInstance { class, .. },
            ..
        }) => Some(class.name.as_str()),
        _ => None,
    }
}

fn exception_class_from_hlvalue(w_value: &Hlvalue) -> Option<ExceptionClass> {
    match w_value {
        Hlvalue::Constant(Constant {
            value: ConstValue::ExceptionClass(class),
            ..
        }) => Some(class.clone()),
        Hlvalue::Constant(Constant {
            value: ConstValue::ExceptionInstance { class, .. },
            ..
        }) => Some(class.clone()),
        _ => None,
    }
}

fn empty_globals_constant() -> Constant {
    Constant::new(ConstValue::Dict(HashMap::new()))
}

fn null_value() -> Hlvalue {
    Hlvalue::Constant(Constant::new(ConstValue::Placeholder))
}

fn is_null_value(value: &Hlvalue) -> bool {
    matches!(
        value,
        Hlvalue::Constant(Constant {
            value: ConstValue::Placeholder,
            ..
        })
    )
}

fn exception_restore_token(previous: Option<FSException>) -> StackElem {
    match previous {
        Some(w_exc) => StackElem::Signal(FlowSignal::Raise { w_exc }),
        None => StackElem::Value(Hlvalue::Constant(Constant::new(ConstValue::None))),
    }
}

fn exception_from_hlvalue(value: &Hlvalue) -> Option<FSException> {
    match value {
        Hlvalue::Constant(Constant {
            value: ConstValue::ExceptionInstance { class, message },
            ..
        }) => Some(FSException::new(
            Hlvalue::Constant(Constant::new(ConstValue::ExceptionClass(class.clone()))),
            Hlvalue::Constant(Constant::new(ConstValue::ExceptionInstance {
                class: class.clone(),
                message: message.clone(),
            })),
        )),
        Hlvalue::Constant(Constant {
            value: ConstValue::ExceptionClass(class),
            ..
        }) => Some(FSException::new(
            Hlvalue::Constant(Constant::new(ConstValue::ExceptionClass(class.clone()))),
            Hlvalue::Constant(Constant::new(ConstValue::ExceptionInstance {
                class: class.clone(),
                message: None,
            })),
        )),
        _ => None,
    }
}

fn load_global_name_index(oparg: u32) -> usize {
    (oparg >> 1) as usize
}

fn load_attr_name_index(oparg: u32) -> usize {
    (oparg >> 1) as usize
}

fn build_call_shape_constant(shape: &CallShape) -> Hlvalue {
    Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![
        ConstValue::Int(shape.shape_cnt as i64),
        ConstValue::Tuple(
            shape
                .shape_keys
                .iter()
                .cloned()
                .map(ConstValue::Str)
                .collect(),
        ),
        ConstValue::Bool(shape.shape_star),
    ])))
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FlowSignalTag {
    Return,
    Raise,
    RaiseImplicit,
    Break,
    Continue,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FrameBlockKind {
    Loop,
    Except,
    Iter,
    Finally,
    With,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FrameBlock {
    pub handlerposition: i64,
    pub stackdepth: usize,
    pub kind: FrameBlockKind,
}

impl FrameBlock {
    pub fn new(handlerposition: i64, stackdepth: usize, kind: FrameBlockKind) -> Self {
        FrameBlock {
            handlerposition,
            stackdepth,
            kind,
        }
    }

    pub fn cleanupstack(&self, ctx: &mut FlowContext) {
        ctx.dropvaluesuntil(self.stackdepth);
    }

    pub fn handles(&self, signal: &FlowSignal) -> bool {
        match self.kind {
            FrameBlockKind::Loop => {
                matches!(signal, FlowSignal::Break | FlowSignal::Continue { .. })
            }
            FrameBlockKind::Except | FrameBlockKind::Iter => {
                matches!(
                    signal,
                    FlowSignal::Raise { .. } | FlowSignal::RaiseImplicit { .. }
                )
            }
            FrameBlockKind::Finally | FrameBlockKind::With => true,
        }
    }

    pub fn handle(
        &self,
        ctx: &mut FlowContext,
        signal: FlowSignal,
    ) -> Result<i64, FlowContextError> {
        match self.kind {
            FrameBlockKind::Loop => match signal {
                FlowSignal::Continue { jump_to } => {
                    ctx.blockstack.push(self.clone());
                    Ok(jump_to)
                }
                FlowSignal::Break => {
                    self.cleanupstack(ctx);
                    Ok(self.handlerposition)
                }
                _ => Err(FlowContextError::Flowing(FlowingError::new(
                    "loop block received unsupported signal",
                ))),
            },
            FrameBlockKind::Except => {
                self.cleanupstack(ctx);
                let w_exc = match &signal {
                    FlowSignal::Raise { w_exc } | FlowSignal::RaiseImplicit { w_exc } => {
                        w_exc.clone()
                    }
                    _ => {
                        return Err(FlowContextError::Flowing(FlowingError::new(
                            "except block received unsupported signal",
                        )));
                    }
                };
                ctx.pushsignal(signal);
                ctx.pushvalue(StackElem::Value(w_exc.w_value.clone()));
                ctx.pushvalue(StackElem::Value(w_exc.w_type.clone()));
                ctx.last_exception = Some(w_exc);
                Ok(self.handlerposition)
            }
            FrameBlockKind::Iter => {
                let w_exc = match signal {
                    FlowSignal::Raise { w_exc } | FlowSignal::RaiseImplicit { w_exc } => w_exc,
                    _ => {
                        return Err(FlowContextError::Flowing(FlowingError::new(
                            "iter block received unsupported signal",
                        )));
                    }
                };
                if ctx.exception_match(&w_exc.w_type, &exception_class_value("StopIteration"))? {
                    ctx.popvalue();
                    Ok(self.handlerposition)
                } else {
                    ctx.unroll(FlowSignal::Raise { w_exc })
                }
            }
            FrameBlockKind::Finally | FrameBlockKind::With => {
                self.cleanupstack(ctx);
                ctx.pushsignal(signal);
                Ok(self.handlerposition)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpamBlock {
    pub block: BlockRef,
    pub framestate: FrameState,
    pub dead: bool,
}

impl SpamBlock {
    pub fn new(framestate: FrameState) -> Self {
        let inputargs = framestate
            .getvariables()
            .into_iter()
            .map(Hlvalue::Variable)
            .collect();
        SpamBlock {
            block: Block::shared(inputargs),
            framestate,
            dead: false,
        }
    }

    pub fn from_block(block: BlockRef, framestate: FrameState) -> Self {
        SpamBlock {
            block,
            framestate,
            dead: false,
        }
    }

    pub fn make_recorder(&self) -> Recorder {
        Recorder::Block(BlockRecorder::new(
            self.block.clone(),
            self.framestate.clone(),
        ))
    }
}

/// RPython `flowcontext.py:47-78` — `class EggBlock(Block)`.
///
/// Deviation from upstream (parity rule #1):
///
/// * Upstream exposes `EggBlock.framestate` and `EggBlock.dead` as
///   `@property` methods that walk `prevblock` back to the ancestor
///   `SpamBlock`. The Rust port caches them as owned fields because
///   `EggBlock` instances are always constructed from a
///   `BlockRecorder` whose `entry_state` is already the ancestor's
///   framestate, so the walked value equals the cached field at
///   construction time. Cache coherence is preserved because
///   `fixeggblocks` (upstream `flowcontext.py:80-83`) only *deletes*
///   `SpamBlock.framestate` after graph finalization — it never
///   mutates it — and Rust finalization runs after all `EggBlock`s are
///   created.
#[derive(Clone, Debug)]
pub struct EggBlock {
    pub block: BlockRef,
    pub prevblock: BlockRef,
    /// RPython `EggBlock.booloutcome`. Upstream stores a Python duck-
    /// typed value here: `guessbool` assigns a `bool`
    /// (`flowcontext.py:111-113`), while `guessexception` assigns
    /// `None` or an exception class Constant
    /// (`flowcontext.py:139-141`). Rust closes the union by storing
    /// the value as an `Option<Hlvalue>` — `None` mirrors Python
    /// `None`, `Some(Constant(Bool(_)))` covers the `guessbool` case,
    /// and `Some(Constant(ExceptionClass(_)))` covers the
    /// `guessexception` class case.
    pub booloutcome: Option<Hlvalue>,
    pub last_exception: Option<Hlvalue>,
    /// Mirrors `EggBlock.framestate` @property's walked result. See
    /// type-level comment for cache-coherence argument.
    pub framestate: FrameState,
    /// Mirrors `EggBlock.dead` @property's walked result.
    pub dead: bool,
}

impl EggBlock {
    pub fn new(
        inputargs: Vec<Hlvalue>,
        prevblock: BlockRef,
        booloutcome: Option<Hlvalue>,
        framestate: FrameState,
        dead: bool,
    ) -> Self {
        EggBlock {
            block: Block::shared(inputargs),
            prevblock,
            booloutcome,
            last_exception: None,
            framestate,
            dead,
        }
    }

    pub fn extravars(&mut self, last_exception: Option<Hlvalue>) {
        self.last_exception = last_exception;
    }

    pub fn make_recorder(&self) -> Recorder {
        Recorder::Replay(Replayer::new(
            self.prevblock.clone(),
            self.booloutcome.clone(),
            Recorder::Block(BlockRecorder::new(
                self.block.clone(),
                self.framestate.clone(),
            )),
            self.last_exception.clone(),
            self.block.borrow().inputargs.clone(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct BlockRecorder {
    pub crnt_block: BlockRef,
    pub final_state: Option<FrameState>,
    pub entry_state: FrameState,
}

impl BlockRecorder {
    pub fn new(block: BlockRef, entry_state: FrameState) -> Self {
        BlockRecorder {
            crnt_block: block,
            final_state: None,
            entry_state,
        }
    }

    pub fn append(&mut self, operation: SpaceOperation) {
        self.crnt_block.borrow_mut().operations.push(operation);
    }

    pub fn guessbool(
        &mut self,
        ctx: &mut FlowContext,
        w_condition: Hlvalue,
    ) -> Result<bool, FlowContextError> {
        let mut links = Vec::new();
        for case in [false, true] {
            let egg = EggBlock::new(
                Vec::new(),
                self.crnt_block.clone(),
                Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(case)))),
                self.entry_state.clone(),
                false,
            );
            ctx.pendingblocks.push_back(PendingBlock::Egg(egg.clone()));
            links.push(
                Link::new(
                    Vec::new(),
                    Some(egg.block.clone()),
                    Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(case)))),
                )
                .into_ref(),
            );
        }
        self.crnt_block.borrow_mut().exitswitch = Some(w_condition);
        self.crnt_block.closeblock(links);
        Err(FlowContextError::StopFlowing)
    }

    pub fn guessexception(
        &mut self,
        ctx: &mut FlowContext,
        cases: &[Hlvalue],
    ) -> Result<(), FlowContextError> {
        let mut links = Vec::new();
        for case in std::iter::once(None).chain(cases.iter().cloned().map(Some)) {
            let (vars, inputargs, last_exception) = if let Some(case_value) = case.clone() {
                let last_exc = if matches!(
                    case_value,
                    Hlvalue::Constant(Constant {
                        value: ConstValue::ExceptionClass(ref class),
                        ..
                    }) if class.name == "Exception"
                ) {
                    Hlvalue::Variable(Variable::named("last_exception"))
                } else {
                    case_value.clone()
                };
                let last_exc_value = Hlvalue::Variable(Variable::named("last_exc_value"));
                (
                    vec![last_exc.clone(), last_exc_value.clone()],
                    vec![
                        Hlvalue::Variable(Variable::new()),
                        Hlvalue::Variable(Variable::new()),
                    ],
                    Some(last_exc),
                )
            } else {
                (Vec::new(), Vec::new(), None)
            };
            let mut egg = EggBlock::new(
                inputargs,
                self.crnt_block.clone(),
                case.clone(),
                self.entry_state.clone(),
                false,
            );
            egg.extravars(last_exception.clone());
            ctx.pendingblocks.push_back(PendingBlock::Egg(egg.clone()));
            let mut link = Link::new(vars, Some(egg.block.clone()), case.clone());
            if let Some(last_exc) = last_exception {
                link.extravars(
                    Some(last_exc),
                    Some(Hlvalue::Variable(Variable::named("last_exc_value"))),
                );
            }
            links.push(link.into_ref());
        }
        self.crnt_block.borrow_mut().exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        self.crnt_block.closeblock(links);
        Err(FlowContextError::StopFlowing)
    }
}

/// RPython `flowcontext.py:152-178` — `class Replayer(Recorder)`.
#[derive(Clone, Debug)]
pub struct Replayer {
    pub crnt_block: BlockRef,
    pub listtoreplay: Vec<SpaceOperation>,
    /// RPython `Replayer.booloutcome`. Same duck-typed union as
    /// `EggBlock.booloutcome`: `bool` for the `guessbool` case,
    /// `None` / exception class for `guessexception`.
    pub booloutcome: Option<Hlvalue>,
    pub nextreplayer: Box<Recorder>,
    pub index: usize,
    pub last_exception: Option<Hlvalue>,
    pub inputargs: Vec<Hlvalue>,
    pub final_state: Option<FrameState>,
}

impl Replayer {
    pub fn new(
        block: BlockRef,
        booloutcome: Option<Hlvalue>,
        nextreplayer: Recorder,
        last_exception: Option<Hlvalue>,
        inputargs: Vec<Hlvalue>,
    ) -> Self {
        let listtoreplay = block.borrow().operations.clone();
        Replayer {
            crnt_block: block,
            listtoreplay,
            booloutcome,
            nextreplayer: Box::new(nextreplayer),
            index: 0,
            last_exception,
            inputargs,
            final_state: None,
        }
    }

    pub fn append(&mut self, operation: SpaceOperation) {
        let expected = &self.listtoreplay[self.index];
        let mut operation = operation;
        operation.result = expected.result.clone();
        assert_eq!(operation, *expected, "operation replay diverged");
        self.index += 1;
    }

    pub fn guessbool(
        &mut self,
        ctx: &mut FlowContext,
        _w_condition: Hlvalue,
    ) -> Result<bool, FlowContextError> {
        assert_eq!(self.index, self.listtoreplay.len());
        ctx.recorder = Some(*self.nextreplayer.clone());
        match &self.booloutcome {
            Some(Hlvalue::Constant(Constant {
                value: ConstValue::Bool(outcome),
                ..
            })) => Ok(*outcome),
            _ => Err(FlowContextError::Flowing(FlowingError::new(
                "replayer bool outcome mismatch",
            ))),
        }
    }

    pub fn guessexception(&mut self, ctx: &mut FlowContext) -> Result<(), FlowContextError> {
        assert_eq!(self.index, self.listtoreplay.len());
        ctx.recorder = Some(*self.nextreplayer.clone());
        if self.booloutcome.is_some() {
            let w_exc_cls = self
                .last_exception
                .clone()
                .unwrap_or_else(|| self.inputargs[self.inputargs.len() - 2].clone());
            let w_exc_value = self.inputargs[self.inputargs.len() - 1].clone();
            return Err(FlowContextError::Signal(FlowSignal::RaiseImplicit {
                w_exc: FSException::new(w_exc_cls, w_exc_value),
            }));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Recorder {
    Block(BlockRecorder),
    Replay(Replayer),
}

impl Recorder {
    pub fn current_block(&self) -> BlockRef {
        match self {
            Recorder::Block(rec) => rec.crnt_block.clone(),
            Recorder::Replay(rep) => rep.crnt_block.clone(),
        }
    }

    pub fn final_state(&self) -> Option<FrameState> {
        match self {
            Recorder::Block(rec) => rec.final_state.clone(),
            Recorder::Replay(rep) => rep.final_state.clone(),
        }
    }

    pub fn set_final_state(&mut self, state: FrameState) {
        match self {
            Recorder::Block(rec) => rec.final_state = Some(state),
            Recorder::Replay(rep) => rep.final_state = Some(state),
        }
    }
}

#[derive(Clone, Debug)]
pub enum PendingBlock {
    Spam(SpamBlock),
    Egg(EggBlock),
}

impl PendingBlock {
    pub fn block(&self) -> BlockRef {
        match self {
            PendingBlock::Spam(block) => block.block.clone(),
            PendingBlock::Egg(block) => block.block.clone(),
        }
    }

    pub fn framestate(&self) -> FrameState {
        match self {
            PendingBlock::Spam(block) => block.framestate.clone(),
            PendingBlock::Egg(block) => block.framestate.clone(),
        }
    }

    pub fn dead(&self) -> bool {
        match self {
            PendingBlock::Spam(block) => block.dead,
            PendingBlock::Egg(block) => block.dead,
        }
    }

    pub fn make_recorder(&self) -> Recorder {
        match self {
            PendingBlock::Spam(block) => block.make_recorder(),
            PendingBlock::Egg(block) => block.make_recorder(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlowContextError {
    StopFlowing,
    BytecodeCorruption(BytecodeCorruption),
    Flowing(FlowingError),
    Signal(FlowSignal),
}

impl From<BytecodeCorruption> for FlowContextError {
    fn from(value: BytecodeCorruption) -> Self {
        FlowContextError::BytecodeCorruption(value)
    }
}

pub struct FlowContext {
    pub graph: FunctionGraph,
    pub pycode: HostCode,
    pub w_globals: Constant,
    pub blockstack: Vec<FrameBlock>,
    pub closure: Vec<Constant>,
    pub f_lineno: u32,
    pub last_offset: i64,
    pub nlocals: usize,
    pub locals_w: Vec<Option<Hlvalue>>,
    pub stack: Vec<StackElem>,
    pub last_exception: Option<FSException>,
    pub joinpoints: HashMap<i64, Vec<SpamBlock>>,
    pub pendingblocks: VecDeque<PendingBlock>,
    pub recorder: Option<Recorder>,
    pending_exception_restore: Option<Option<FSException>>,
}

impl FlowContext {
    pub fn new(mut graph: FunctionGraph, code: HostCode) -> Self {
        if graph.func.is_none() {
            let mut func = GraphFunc::new(code.co_name.clone(), empty_globals_constant());
            func.filename = Some(code.co_filename.clone());
            func.firstlineno = Some(code.co_firstlineno);
            func.code = Some(Box::new(code.clone()));
            graph.func = Some(func);
        }
        let w_globals = graph
            .func
            .as_ref()
            .map(|func| func.globals.clone())
            .unwrap_or_else(empty_globals_constant);
        let closure = graph
            .func
            .as_ref()
            .map(|func| func.closure.clone())
            .unwrap_or_default();
        let f_lineno = code.co_firstlineno;
        let nlocals = code.co_nlocals as usize;
        FlowContext {
            graph,
            pycode: code,
            w_globals,
            blockstack: Vec::new(),
            closure,
            f_lineno,
            last_offset: 0,
            nlocals,
            locals_w: vec![None; nlocals],
            stack: Vec::new(),
            last_exception: None,
            joinpoints: HashMap::new(),
            pendingblocks: VecDeque::new(),
            recorder: None,
            pending_exception_restore: None,
        }
    }

    pub fn stackdepth(&self) -> usize {
        self.stack.len()
    }

    pub fn pushvalue(&mut self, w_object: StackElem) {
        self.stack.push(w_object);
    }

    pub fn pushsignal(&mut self, signal: FlowSignal) {
        self.stack.push(StackElem::Signal(signal));
    }

    pub fn peekvalue(&self) -> StackElem {
        self.stack.last().cloned().expect("value stack underflow")
    }

    pub fn peekvalue_at(&self, delta: usize) -> StackElem {
        self.stack
            .get(
                self.stack
                    .len()
                    .checked_sub(delta + 1)
                    .expect("value stack underflow"),
            )
            .cloned()
            .expect("value stack underflow")
    }

    pub fn settopvalue(&mut self, w_object: StackElem) {
        *self.stack.last_mut().expect("value stack underflow") = w_object;
    }

    pub fn popvalue(&mut self) -> StackElem {
        self.stack.pop().expect("value stack underflow")
    }

    pub fn popvalues(&mut self, n: usize) -> Vec<StackElem> {
        if n == 0 {
            return Vec::new();
        }
        let split = self.stack.len() - n;
        self.stack.split_off(split)
    }

    pub fn dropvaluesuntil(&mut self, finaldepth: usize) {
        self.stack.truncate(finaldepth);
    }

    pub fn getstate(&self, next_offset: i64) -> FrameState {
        FrameState::new(
            self.locals_w.clone(),
            self.stack.clone(),
            self.last_exception.clone(),
            self.blockstack.clone(),
            next_offset,
        )
    }

    pub fn setstate(&mut self, state: &FrameState) {
        self.locals_w = state.locals_w.clone();
        self.stack = state.stack.clone();
        self.last_exception = state.last_exception.clone();
        self.blockstack = state.blocklist.clone();
        self.normalize_raise_signals();
    }

    pub fn normalize_raise_signals(&mut self) {
        for cell in &mut self.stack {
            if let StackElem::Signal(FlowSignal::RaiseImplicit { w_exc }) = cell {
                *cell = StackElem::Signal(FlowSignal::Raise {
                    w_exc: w_exc.clone(),
                });
            }
        }
    }

    pub fn current_block(&self) -> BlockRef {
        self.recorder
            .as_ref()
            .expect("recorder not installed")
            .current_block()
    }

    pub fn guessbool(&mut self, w_condition: Hlvalue) -> Result<bool, FlowContextError> {
        if let Hlvalue::Constant(constant) = &w_condition {
            if let Some(value) = constant.value.truthy() {
                return Ok(value);
            }
            return Err(FlowContextError::Flowing(FlowingError::new(format!(
                "cannot guessbool({w_condition})"
            ))));
        }
        let recorder = self.recorder.take().expect("recorder not installed");
        let result = match recorder.clone() {
            Recorder::Block(mut rec) => rec.guessbool(self, w_condition),
            Recorder::Replay(mut rep) => rep.guessbool(self, w_condition),
        };
        if self.recorder.is_none() {
            self.recorder = Some(recorder);
        }
        result
    }

    pub fn maybe_merge(&mut self) -> Result<(), FlowContextError> {
        if let Some(recorder) = &self.recorder {
            if let Some(final_state) = recorder.final_state() {
                self.mergeblock(recorder.current_block(), final_state);
                return Err(FlowContextError::StopFlowing);
            }
        }
        Ok(())
    }

    pub fn record(&mut self, mut spaceop: SpaceOperation) -> Result<(), FlowContextError> {
        spaceop.offset = self.last_offset;
        match self.recorder.as_mut().expect("recorder not installed") {
            Recorder::Block(rec) => rec.append(spaceop),
            Recorder::Replay(rep) => rep.append(spaceop),
        }
        Ok(())
    }

    fn appcall(
        &mut self,
        func: BuiltinFunction,
        args_w: Vec<Hlvalue>,
    ) -> Result<Hlvalue, FlowContextError> {
        let mut args = vec![Hlvalue::Constant(Constant::new(builtin_function(func)))];
        args.extend(args_w);
        self.record_maybe_raise_op("direct_call", args, Self::common_exception_cases())
    }

    fn newfunction(
        &self,
        w_code: Hlvalue,
        defaults_w: Vec<Hlvalue>,
    ) -> Result<Hlvalue, FlowContextError> {
        let mut defaults = Vec::with_capacity(defaults_w.len());
        for default in defaults_w {
            match default {
                Hlvalue::Constant(constant) => defaults.push(constant),
                _ => {
                    return Err(FlowContextError::Flowing(FlowingError::new(
                        "Dynamically created function must have constant default values.",
                    )));
                }
            }
        }
        let code = match w_code {
            Hlvalue::Constant(Constant {
                value: ConstValue::Code(code),
                ..
            }) => *code,
            other => {
                return Err(FlowContextError::Flowing(FlowingError::new(format!(
                    "MAKE_FUNCTION expected code constant, got {other:?}"
                ))));
            }
        };
        Ok(Hlvalue::Constant(Constant::new(ConstValue::Function(
            Box::new(GraphFunc::from_host_code(
                code,
                self.w_globals.clone(),
                defaults,
            )),
        ))))
    }

    fn handle_print_function(
        &mut self,
        arguments: Vec<Hlvalue>,
        keywords: HashMap<String, Hlvalue>,
        w_star: Option<Hlvalue>,
    ) -> Result<Hlvalue, FlowContextError> {
        if w_star.is_some() {
            return Err(FlowContextError::Flowing(FlowingError::new(
                "print function with *args is not RPython",
            )));
        }
        let bad_kwarg = keywords.keys().find(|key| key.as_str() != "end").cloned();
        if let Some(keyword) = bad_kwarg {
            return Err(FlowContextError::Flowing(FlowingError::new(format!(
                "print function with {keyword}= is not RPython"
            ))));
        }
        for w_arg in arguments {
            let w_s = self.record_pure_op("str", vec![w_arg])?;
            let _ = self.appcall(BuiltinFunction::RpythonPrintItem, vec![w_s])?;
        }
        if let Some(w_end) = keywords.get("end") {
            let _ = self.appcall(BuiltinFunction::RpythonPrintEnd, vec![w_end.clone()])?;
        } else {
            let _ = self.appcall(BuiltinFunction::RpythonPrintNewline, Vec::new())?;
        }
        Ok(Hlvalue::Constant(Constant::new(ConstValue::None)))
    }

    fn unpack_sequence(
        &mut self,
        w_iterable: Hlvalue,
        expected_length: usize,
    ) -> Result<Vec<Hlvalue>, FlowContextError> {
        if let Hlvalue::Constant(Constant { value, .. }) = &w_iterable {
            if let Some(items) = value.sequence_items() {
                if items.len() != expected_length {
                    return Err(FlowContextError::Signal(FlowSignal::Raise {
                        w_exc: FSException::new(
                            exception_class_value("ValueError"),
                            exception_instance_value("ValueError", None),
                        ),
                    }));
                }
                return Ok(items
                    .iter()
                    .cloned()
                    .map(|item| Hlvalue::Constant(Constant::new(item)))
                    .collect());
            }
        }
        let w_len = self.record_maybe_raise_op(
            "len",
            vec![w_iterable.clone()],
            Self::common_exception_cases(),
        )?;
        let w_correct = self.record_pure_op(
            "eq",
            vec![
                w_len,
                Hlvalue::Constant(Constant::new(ConstValue::Int(expected_length as i64))),
            ],
        )?;
        let w_correct_bool = self.bool_operand(w_correct)?;
        if !self.guessbool(w_correct_bool)? {
            return Err(FlowContextError::Signal(FlowSignal::Raise {
                w_exc: FSException::new(
                    exception_class_value("ValueError"),
                    exception_instance_value("ValueError", None),
                ),
            }));
        }
        let mut items = Vec::with_capacity(expected_length);
        for index in 0..expected_length {
            items.push(self.record_pure_op(
                "getitem",
                vec![
                    w_iterable.clone(),
                    Hlvalue::Constant(Constant::new(ConstValue::Int(index as i64))),
                ],
            )?);
        }
        Ok(items)
    }

    fn call_function(
        &mut self,
        w_function: Hlvalue,
        arguments: Vec<Hlvalue>,
        keywords: HashMap<String, Hlvalue>,
        w_star: Option<Hlvalue>,
        w_starstar: Option<Hlvalue>,
    ) -> Result<Hlvalue, FlowContextError> {
        if w_starstar.is_some() {
            return Err(FlowContextError::Flowing(FlowingError::new(
                "Dict-unpacking is not RPython",
            )));
        }
        if matches!(
            &w_function,
            Hlvalue::Constant(Constant {
                value: ConstValue::Builtin(BuiltinObject::Function(BuiltinFunction::Print)),
                ..
            })
        ) {
            return self.handle_print_function(arguments, keywords, w_star);
        }
        let args = CallSpec::new(arguments, Some(keywords), w_star);
        if args.keywords.is_empty() && !matches!(args.w_stararg, Some(Hlvalue::Variable(_))) {
            let mut flat = vec![w_function.clone()];
            flat.extend(args.as_list());
            let opname = if matches!(w_function, Hlvalue::Constant(_)) {
                "direct_call"
            } else {
                "indirect_call"
            };
            self.record_maybe_raise_op(opname, flat, Self::common_exception_cases())
        } else {
            let (shape, mut args_w) = args.flatten();
            let mut flat = vec![w_function.clone(), build_call_shape_constant(&shape)];
            flat.append(&mut args_w);
            self.record_maybe_raise_op("call_args", flat, Self::common_exception_cases())
        }
    }

    fn pop_hlvalue(&mut self) -> Result<Hlvalue, FlowContextError> {
        match self.popvalue() {
            StackElem::Value(value) => Ok(value),
            StackElem::Signal(signal) => Err(FlowContextError::Signal(signal)),
        }
    }

    fn pop_hlvalues(&mut self, n: usize) -> Result<Vec<Hlvalue>, FlowContextError> {
        self.popvalues(n)
            .into_iter()
            .map(|item| match item {
                StackElem::Value(value) => Ok(value),
                StackElem::Signal(signal) => Err(FlowContextError::Signal(signal)),
            })
            .collect()
    }

    pub fn getlocalvarname(&self, index: usize) -> &str {
        &self.pycode.co_varnames[index]
    }

    pub fn getconstant_w(&self, index: usize) -> Result<Hlvalue, FlowContextError> {
        let constant = self.pycode.consts.get(index).ok_or_else(|| {
            FlowContextError::BytecodeCorruption(BytecodeCorruption::new(format!(
                "constant index {index} out of range"
            )))
        })?;
        Ok(Hlvalue::Constant(Constant::new(Self::host_const_value(
            constant,
        )?)))
    }

    pub fn getname_u(&self, index: usize) -> Result<&str, FlowContextError> {
        self.pycode
            .names
            .get(index)
            .map(String::as_str)
            .ok_or_else(|| {
                FlowContextError::BytecodeCorruption(BytecodeCorruption::new(format!(
                    "name index {index} out of range"
                )))
            })
    }

    pub fn getname_w(&self, index: usize) -> Result<Hlvalue, FlowContextError> {
        Ok(Hlvalue::Constant(Constant::new(ConstValue::Str(
            self.getname_u(index)?.to_owned(),
        ))))
    }

    fn get_keyword_names(&self, w_names: Hlvalue) -> Result<Vec<String>, FlowContextError> {
        match w_names {
            Hlvalue::Constant(Constant {
                value: ConstValue::Tuple(items),
                ..
            }) => items
                .into_iter()
                .map(|item| match item {
                    ConstValue::Str(name) => Ok(name),
                    other => Err(FlowContextError::Flowing(FlowingError::new(format!(
                        "CALL_KW expected tuple[str], got {other:?}"
                    )))),
                })
                .collect(),
            other => Err(FlowContextError::Flowing(FlowingError::new(format!(
                "CALL_KW expected tuple constant, got {other:?}"
            )))),
        }
    }

    fn host_const_value(constant: &ConstantData) -> Result<ConstValue, FlowContextError> {
        match constant {
            ConstantData::Boolean { value } => Ok(ConstValue::Bool(*value)),
            ConstantData::Integer { value } => {
                let value = i64::try_from(value.clone()).map_err(|_| {
                    FlowContextError::Flowing(FlowingError::new(format!(
                        "integer constant does not fit in i64: {value}"
                    )))
                })?;
                Ok(ConstValue::Int(value))
            }
            ConstantData::None => Ok(ConstValue::None),
            ConstantData::Str { value } => Ok(ConstValue::Str(value.to_string())),
            ConstantData::Tuple { elements } => {
                let mut out = Vec::with_capacity(elements.len());
                for element in elements {
                    out.push(Self::host_const_value(element)?);
                }
                Ok(ConstValue::Tuple(out))
            }
            ConstantData::Code { code } => {
                Ok(ConstValue::Code(Box::new(HostCode::from_code(code))))
            }
            other => Err(FlowContextError::Flowing(FlowingError::new(format!(
                "unsupported host constant in flowspace: {other:?}"
            )))),
        }
    }

    fn find_global(&self, varname: &str) -> Result<Hlvalue, FlowContextError> {
        if let Some(globals) = self.w_globals.value.dict_items() {
            if let Some(value) = globals.get(varname) {
                return Ok(Hlvalue::Constant(Constant::new(value.clone())));
            }
        } else {
            return Err(FlowContextError::Flowing(FlowingError::new(
                "function globals must be a dict constant",
            )));
        }
        lookup_builtin(varname)
            .map(|value| Hlvalue::Constant(Constant::new(value)))
            .ok_or_else(|| {
                FlowContextError::Flowing(FlowingError::new(format!(
                    "global name '{varname}' is not defined"
                )))
            })
    }

    fn record_pure_op(
        &mut self,
        opname: impl Into<String>,
        args: Vec<Hlvalue>,
    ) -> Result<Hlvalue, FlowContextError> {
        self.maybe_merge()?;
        let result = Hlvalue::Variable(Variable::new());
        self.record(SpaceOperation::new(opname, args, result.clone()))?;
        Ok(result)
    }

    fn record_maybe_raise_op(
        &mut self,
        opname: impl Into<String>,
        args: Vec<Hlvalue>,
        exceptions: Vec<Hlvalue>,
    ) -> Result<Hlvalue, FlowContextError> {
        self.maybe_merge()?;
        let result = Hlvalue::Variable(Variable::new());
        self.record(SpaceOperation::new(opname, args, result.clone()))?;
        self.guessexception(&exceptions)?;
        Ok(result)
    }

    fn record_side_effect_op(
        &mut self,
        opname: impl Into<String>,
        args: Vec<Hlvalue>,
        exceptions: Vec<Hlvalue>,
    ) -> Result<(), FlowContextError> {
        let _ = self.record_maybe_raise_op(opname, args, exceptions)?;
        Ok(())
    }

    fn bool_operand(&mut self, value: Hlvalue) -> Result<Hlvalue, FlowContextError> {
        match value {
            Hlvalue::Constant(constant) => Ok(Hlvalue::Constant(Constant::new(ConstValue::Bool(
                constant.value.truthy().ok_or_else(|| {
                    FlowContextError::Flowing(FlowingError::new(
                        "cannot determine truthiness of constant",
                    ))
                })?,
            )))),
            other => self.record_pure_op("bool", vec![other]),
        }
    }

    fn unsupported_rpython<T>(&self, message: impl Into<String>) -> Result<T, FlowContextError> {
        Err(FlowContextError::Flowing(FlowingError::new(message)))
    }

    fn current_exception_table_handler(&self) -> Option<ExceptionTableEntry> {
        if self.last_offset < 0 {
            return None;
        }
        self.pycode.find_exception_handler(self.last_offset as u32)
    }

    fn dispatch_exception_to_handler(
        &mut self,
        signal: FlowSignal,
        entry: ExceptionTableEntry,
    ) -> Result<i64, FlowContextError> {
        let w_exc = match &signal {
            FlowSignal::Raise { w_exc } | FlowSignal::RaiseImplicit { w_exc } => w_exc.clone(),
            _ => {
                return Err(FlowContextError::BytecodeCorruption(
                    BytecodeCorruption::new("non-exception signal routed to exception table"),
                ));
            }
        };
        let previous = self.last_exception.clone();
        self.dropvaluesuntil(entry.depth as usize);
        if entry.push_lasti {
            self.pushvalue(StackElem::Value(Hlvalue::Constant(Constant::new(
                ConstValue::Int(self.last_offset),
            ))));
        }
        let (_, target_instruction, _) = self.pycode.read(entry.target)?;
        let starts_with_push_exc =
            matches!(target_instruction.deoptimize(), Instruction::PushExcInfo);
        if starts_with_push_exc {
            self.pushvalue(StackElem::Value(w_exc.w_value.clone()));
        } else {
            self.pushsignal(signal);
        }
        self.last_exception = Some(w_exc);
        self.pending_exception_restore = if starts_with_push_exc {
            Some(previous)
        } else {
            None
        };
        Ok(entry.target as i64)
    }

    fn current_stack_exception(&self) -> Option<FSException> {
        match self.stack.last() {
            Some(StackElem::Value(value)) => exception_from_hlvalue(value),
            Some(StackElem::Signal(FlowSignal::Raise { w_exc }))
            | Some(StackElem::Signal(FlowSignal::RaiseImplicit { w_exc })) => Some(w_exc.clone()),
            _ => None,
        }
    }

    fn current_exception_type(&self) -> Result<Hlvalue, FlowContextError> {
        if let Some(w_exc) = &self.last_exception {
            return Ok(w_exc.w_type.clone());
        }
        self.current_stack_exception()
            .map(|w_exc| w_exc.w_type)
            .ok_or_else(|| {
                FlowContextError::Flowing(FlowingError::new(
                    "CHECK_EXC_MATCH without an active exception",
                ))
            })
    }

    fn restore_exception_from_token(&mut self, token: StackElem) -> Result<(), FlowContextError> {
        self.last_exception = match token {
            StackElem::Signal(FlowSignal::Raise { w_exc })
            | StackElem::Signal(FlowSignal::RaiseImplicit { w_exc }) => Some(w_exc),
            StackElem::Value(Hlvalue::Constant(Constant {
                value: ConstValue::None | ConstValue::Placeholder,
                ..
            })) => None,
            other => {
                return Err(FlowContextError::Flowing(FlowingError::new(format!(
                    "POP_EXCEPT expected saved exception token, got {other:?}"
                ))));
            }
        };
        Ok(())
    }

    fn common_exception_cases() -> Vec<Hlvalue> {
        vec![exception_class_value("Exception")]
    }

    fn stop_iteration_cases() -> Vec<Hlvalue> {
        vec![exception_class_value("StopIteration")]
    }

    fn store_fast(&mut self, varindex: usize, w_newvalue: Hlvalue) {
        self.locals_w[varindex] = Some(w_newvalue.clone());
        if let Hlvalue::Variable(mut variable) = w_newvalue {
            variable.rename(self.getlocalvarname(varindex));
            self.locals_w[varindex] = Some(Hlvalue::Variable(variable));
        }
    }

    pub fn has_exc_handler(&self) -> bool {
        self.blockstack
            .iter()
            .any(|block| matches!(block.kind, FrameBlockKind::Except | FrameBlockKind::Finally))
            || self.current_exception_table_handler().is_some()
    }

    pub fn guessexception(&mut self, cases: &[Hlvalue]) -> Result<(), FlowContextError> {
        if cases.is_empty() || !self.has_exc_handler() {
            return Ok(());
        }
        let recorder = self.recorder.take().expect("recorder not installed");
        let result = match recorder.clone() {
            Recorder::Block(mut rec) => rec.guessexception(self, cases),
            Recorder::Replay(mut rep) => rep.guessexception(self),
        };
        if self.recorder.is_none() {
            self.recorder = Some(recorder);
        }
        result
    }

    pub fn mergeblock(&mut self, currentblock: BlockRef, currentstate: FrameState) {
        let next_offset = currentstate.next_offset;
        let candidate_len = self.joinpoints.get(&next_offset).map_or(0, Vec::len);
        for idx in 0..candidate_len {
            let maybe_newstate = {
                let block = &self.joinpoints.get(&next_offset).unwrap()[idx];
                block.framestate.union(&currentstate)
            };
            if let Some(newstate) = maybe_newstate {
                let matches_existing = {
                    let block = &self.joinpoints.get(&next_offset).unwrap()[idx];
                    newstate.matches(&block.framestate)
                };
                if matches_existing {
                    let target = self.joinpoints.get(&next_offset).unwrap()[idx]
                        .block
                        .clone();
                    let outputargs = currentstate.getoutputargs(&newstate);
                    currentblock.closeblock(vec![
                        Link::new_mergeable(outputargs, Some(target), None).into_ref(),
                    ]);
                    return;
                }
                let replacement = SpamBlock::new(newstate.clone());
                let old_block = self.joinpoints.get(&next_offset).unwrap()[idx]
                    .block
                    .clone();
                let old_framestate = self.joinpoints.get(&next_offset).unwrap()[idx]
                    .framestate
                    .clone();
                let outputargs = currentstate.getoutputargs(&newstate);
                currentblock.closeblock(vec![
                    Link::new_mergeable(outputargs, Some(replacement.block.clone()), None)
                        .into_ref(),
                ]);
                old_block.borrow_mut().operations.clear();
                old_block.borrow_mut().exitswitch = None;
                let old_outputargs = old_framestate.getoutputargs(&newstate);
                old_block.recloseblock(vec![
                    Link::new_mergeable(old_outputargs, Some(replacement.block.clone()), None)
                        .into_ref(),
                ]);
                self.pendingblocks
                    .push_back(PendingBlock::Spam(replacement.clone()));
                self.joinpoints.get_mut(&next_offset).unwrap()[idx] = replacement;
                return;
            }
        }
        let newblock = self.make_next_block(currentblock, currentstate);
        self.joinpoints
            .entry(next_offset)
            .or_default()
            .insert(0, newblock);
    }

    pub fn make_next_block(&mut self, block: BlockRef, state: FrameState) -> SpamBlock {
        let newstate = state.copy();
        let newblock = SpamBlock::new(newstate);
        let outputargs = state.getoutputargs(&newblock.framestate);
        block.closeblock(vec![
            Link::new_mergeable(outputargs, Some(newblock.block.clone()), None).into_ref(),
        ]);
        self.pendingblocks
            .push_back(PendingBlock::Spam(newblock.clone()));
        newblock
    }

    fn initial_pending_block(&self) -> PendingBlock {
        let mut locals = vec![None; self.nlocals];
        let args = self.graph.startblock.borrow().inputargs.clone();
        for (index, arg) in args
            .into_iter()
            .enumerate()
            .take(self.pycode.formalargcount())
        {
            locals[index] = Some(arg);
        }
        PendingBlock::Spam(SpamBlock::from_block(
            self.graph.startblock.clone(),
            FrameState::new(locals, Vec::new(), None, Vec::new(), 0),
        ))
    }

    pub fn build_flow(&mut self) -> Result<(), FlowContextError> {
        self.pendingblocks.clear();
        self.pendingblocks.push_back(self.initial_pending_block());
        while let Some(block) = self.pendingblocks.pop_front() {
            if !block.dead() {
                self.record_block(block)?;
            }
        }
        Ok(())
    }

    pub fn record_block(&mut self, block: PendingBlock) -> Result<(), FlowContextError> {
        self.setstate(&block.framestate());
        self.pending_exception_restore = None;
        let mut next_offset = block.framestate().next_offset;
        self.recorder = Some(block.make_recorder());
        let result = loop {
            match self.handle_bytecode(next_offset) {
                Ok(next) => {
                    next_offset = next;
                    let state = self.getstate(next_offset);
                    if let Some(recorder) = self.recorder.as_mut() {
                        recorder.set_final_state(state);
                    }
                }
                Err(FlowContextError::StopFlowing) => break Ok(()),
                Err(err) => break Err(err),
            }
        };
        self.recorder = None;
        self.pending_exception_restore = None;
        result
    }

    pub fn exception_match(
        &mut self,
        w_exc_type: &Hlvalue,
        w_check_class: &Hlvalue,
    ) -> Result<bool, FlowContextError> {
        let check_value = match w_check_class {
            Hlvalue::Constant(constant) => &constant.value,
            _ => {
                return Err(FlowContextError::Flowing(FlowingError::new(
                    "Non-constant except guard.",
                )));
            }
        };
        match check_value {
            ConstValue::ExceptionClass(class) => {
                if class.name == "StackOverflow" {
                    return self
                        .exception_match(w_exc_type, &exception_class_value("_StackOverflow"));
                }
                if class.is_subclass_of_name("AssertionError")
                    || class.is_subclass_of_name("NotImplementedError")
                {
                    return Err(FlowContextError::Flowing(FlowingError::new(format!(
                        "Catching NotImplementedError, AssertionError, or a subclass is not valid in RPython ({class:?})"
                    ))));
                }
                if let Some(actual) = exception_class_from_hlvalue(w_exc_type) {
                    return Ok(actual.is_subclass_of(class));
                }
                let w_match = self
                    .record_pure_op("issubtype", vec![w_exc_type.clone(), w_check_class.clone()])?;
                self.guessbool(w_match)
            }
            ConstValue::Tuple(elements) => {
                for klass in elements {
                    let w_klass = Hlvalue::Constant(Constant::new(klass.clone()));
                    if self.exception_match(w_exc_type, &w_klass)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            _ => Err(FlowContextError::Flowing(FlowingError::new(
                "Non-constant except guard.",
            ))),
        }
    }

    pub fn unroll(&mut self, signal: FlowSignal) -> Result<i64, FlowContextError> {
        while let Some(block) = self.blockstack.pop() {
            if block.handles(&signal) {
                return block.handle(self, signal);
            }
            block.cleanupstack(self);
        }
        if matches!(
            signal,
            FlowSignal::Raise { .. } | FlowSignal::RaiseImplicit { .. }
        ) {
            if let Some(entry) = self.current_exception_table_handler() {
                return self.dispatch_exception_to_handler(signal, entry);
            }
        }
        signal.nomoreblocks(self)
    }

    pub fn handle_bytecode(&mut self, next_offset: i64) -> Result<i64, FlowContextError> {
        self.last_offset = next_offset;
        let (decoded_next_offset, instruction, oparg) = self.pycode.read(next_offset as u32)?;
        let instruction = instruction.deoptimize();
        let step: Result<Option<i64>, FlowContextError> = (|| {
            match instruction {
                Instruction::Resume { .. }
                | Instruction::Nop
                | Instruction::NotTaken
                | Instruction::ReturnGenerator
                | Instruction::Cache
                | Instruction::SetupAnnotations => Ok(None),
                Instruction::PushNull => {
                    self.pushvalue(StackElem::Value(null_value()));
                    Ok(None)
                }
                Instruction::LoadBuildClass => {
                    self.unsupported_rpython("defining classes inside functions is not RPython")
                }
                Instruction::LoadLocals => self.unsupported_rpython("locals() is not RPython"),
                Instruction::LoadConst { .. } => {
                    self.pushvalue(StackElem::Value(self.getconstant_w(oparg as usize)?));
                    Ok(None)
                }
                Instruction::LoadCommonConstant { .. } => {
                    let value = match pyre_interpreter::bytecode::oparg::CommonConstant::try_from(
                        oparg,
                    )
                    .map_err(|_| {
                        FlowContextError::BytecodeCorruption(BytecodeCorruption::new(format!(
                            "invalid LOAD_COMMON_CONSTANT oparg {oparg}"
                        )))
                    })? {
                        pyre_interpreter::bytecode::oparg::CommonConstant::AssertionError => {
                            exception_class_value("AssertionError")
                        }
                        pyre_interpreter::bytecode::oparg::CommonConstant::NotImplementedError => {
                            exception_class_value("NotImplementedError")
                        }
                        pyre_interpreter::bytecode::oparg::CommonConstant::BuiltinTuple => {
                            self.find_global("tuple")?
                        }
                        pyre_interpreter::bytecode::oparg::CommonConstant::BuiltinAll => {
                            self.find_global("all")?
                        }
                        pyre_interpreter::bytecode::oparg::CommonConstant::BuiltinAny => {
                            self.find_global("any")?
                        }
                        pyre_interpreter::bytecode::oparg::CommonConstant::BuiltinList => {
                            self.find_global("list")?
                        }
                        pyre_interpreter::bytecode::oparg::CommonConstant::BuiltinSet => {
                            self.find_global("set")?
                        }
                    };
                    self.pushvalue(StackElem::Value(value));
                    Ok(None)
                }
                Instruction::LoadSmallInt { .. } => {
                    self.pushvalue(StackElem::Value(Hlvalue::Constant(Constant::new(
                        ConstValue::Int(oparg as i64),
                    ))));
                    Ok(None)
                }
                Instruction::LoadFast { .. }
                | Instruction::LoadFastBorrow { .. }
                | Instruction::LoadFastCheck { .. } => {
                    let value = self.locals_w[oparg as usize].clone().ok_or_else(|| {
                        FlowContextError::Flowing(FlowingError::new(
                            "Local variable referenced before assignment",
                        ))
                    })?;
                    self.pushvalue(StackElem::Value(value));
                    Ok(None)
                }
                Instruction::LoadFastAndClear { .. } => {
                    let value = self.locals_w[oparg as usize].clone().ok_or_else(|| {
                        FlowContextError::Flowing(FlowingError::new(
                            "Local variable referenced before assignment",
                        ))
                    })?;
                    self.locals_w[oparg as usize] = None;
                    self.pushvalue(StackElem::Value(value));
                    Ok(None)
                }
                Instruction::LoadFastLoadFast { var_nums: _ }
                | Instruction::LoadFastBorrowLoadFastBorrow { var_nums: _ } => {
                    let (left, right) = VarNums::from_u32(oparg).indexes();
                    for index in [left.as_usize(), right.as_usize()] {
                        let value = self.locals_w[index].clone().ok_or_else(|| {
                            FlowContextError::Flowing(FlowingError::new(
                                "Local variable referenced before assignment",
                            ))
                        })?;
                        self.pushvalue(StackElem::Value(value));
                    }
                    Ok(None)
                }
                Instruction::StoreFast { .. } => {
                    let value = self.pop_hlvalue()?;
                    self.store_fast(oparg as usize, value);
                    Ok(None)
                }
                Instruction::StoreFastLoadFast { var_nums: _ } => {
                    let (store_idx, load_idx) = VarNums::from_u32(oparg).indexes();
                    let value = self.pop_hlvalue()?;
                    self.store_fast(store_idx.as_usize(), value);
                    let value = self.locals_w[load_idx.as_usize()].clone().ok_or_else(|| {
                        FlowContextError::Flowing(FlowingError::new(
                            "Local variable referenced before assignment",
                        ))
                    })?;
                    self.pushvalue(StackElem::Value(value));
                    Ok(None)
                }
                Instruction::StoreFastStoreFast { var_nums: _ } => {
                    let (left_idx, right_idx) = VarNums::from_u32(oparg).indexes();
                    let left = self.pop_hlvalue()?;
                    let right = self.pop_hlvalue()?;
                    self.store_fast(left_idx.as_usize(), left);
                    self.store_fast(right_idx.as_usize(), right);
                    Ok(None)
                }
                Instruction::LoadGlobal { .. } | Instruction::LoadName { .. } => {
                    let (name_index, push_null) = match instruction {
                        Instruction::LoadGlobal { .. } => {
                            (load_global_name_index(oparg), (oparg & 1) != 0)
                        }
                        _ => (oparg as usize, false),
                    };
                    if push_null {
                        self.pushvalue(StackElem::Value(null_value()));
                    }
                    let value = self.find_global(self.getname_u(name_index)?)?;
                    self.pushvalue(StackElem::Value(value));
                    Ok(None)
                }
                Instruction::StoreGlobal { .. } | Instruction::StoreName { .. } => {
                    let varname = self.getname_u(oparg as usize)?;
                    Err(FlowContextError::Flowing(FlowingError::new(format!(
                        "Attempting to modify global variable  {varname:?}."
                    ))))
                }
                Instruction::DeleteGlobal { .. } | Instruction::DeleteName { .. } => {
                    let varname = self.getname_u(oparg as usize)?;
                    self.unsupported_rpython(format!(
                        "Attempting to modify global variable  {varname:?}."
                    ))
                }
                Instruction::LoadAttr { .. } => {
                    let w_obj = self.pop_hlvalue()?;
                    let w_name = self.getname_w(load_attr_name_index(oparg))?;
                    let w_value = self.record_maybe_raise_op(
                        "getattr",
                        vec![w_obj, w_name],
                        Self::common_exception_cases(),
                    )?;
                    if (oparg & 1) != 0 {
                        self.pushvalue(StackElem::Value(null_value()));
                    }
                    self.pushvalue(StackElem::Value(w_value));
                    Ok(None)
                }
                Instruction::StoreAttr { .. } => {
                    let w_name = self.getname_w(oparg as usize)?;
                    let w_obj = self.pop_hlvalue()?;
                    let w_newvalue = self.pop_hlvalue()?;
                    self.record_side_effect_op(
                        "setattr",
                        vec![w_obj, w_name, w_newvalue],
                        Self::common_exception_cases(),
                    )?;
                    Ok(None)
                }
                Instruction::DeleteAttr { .. } => {
                    let w_name = self.getname_w(oparg as usize)?;
                    let w_obj = self.pop_hlvalue()?;
                    self.record_side_effect_op(
                        "delattr",
                        vec![w_obj, w_name],
                        Self::common_exception_cases(),
                    )?;
                    Ok(None)
                }
                Instruction::LoadDeref { .. } => {
                    let cell = self.closure.get(oparg as usize).ok_or_else(|| {
                        FlowContextError::Flowing(FlowingError::new(format!(
                            "Undefined closure variable index {oparg}"
                        )))
                    })?;
                    self.pushvalue(StackElem::Value(Hlvalue::Constant(cell.clone())));
                    Ok(None)
                }
                Instruction::StoreDeref { .. }
                | Instruction::DeleteDeref { .. }
                | Instruction::MakeCell { .. }
                | Instruction::CopyFreeVars { .. }
                | Instruction::LoadFromDictOrDeref { .. } => {
                    self.unsupported_rpython("closure cell mutation is not RPython")
                }
                Instruction::LoadFromDictOrGlobals { .. } => {
                    let value = self.find_global(self.getname_u(oparg as usize)?)?;
                    self.pushvalue(StackElem::Value(value));
                    Ok(None)
                }
                Instruction::PopTop => {
                    self.popvalue();
                    Ok(None)
                }
                Instruction::Copy { .. } => {
                    let copied = self.peekvalue_at((oparg as usize).saturating_sub(1));
                    self.pushvalue(copied);
                    Ok(None)
                }
                Instruction::Swap { .. } => {
                    let top_index = self.stack.len() - 1;
                    let other_index = self
                        .stack
                        .len()
                        .checked_sub(oparg as usize)
                        .expect("value stack underflow");
                    self.stack.swap(top_index, other_index);
                    Ok(None)
                }
                Instruction::ToBool => {
                    let value = self.pop_hlvalue()?;
                    let result = self.bool_operand(value)?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::UnaryNegative => {
                    let value = self.pop_hlvalue()?;
                    let result = self.record_pure_op("neg", vec![value])?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::CallIntrinsic1 { .. } => {
                    let value = self.pop_hlvalue()?;
                    let intrinsic =
                        pyre_interpreter::bytecode::oparg::IntrinsicFunction1::try_from(oparg)
                            .map_err(|_| {
                                FlowContextError::BytecodeCorruption(BytecodeCorruption::new(
                                    format!("invalid CALL_INTRINSIC_1 func {oparg}"),
                                ))
                            })?;
                    let result = match intrinsic {
                        pyre_interpreter::bytecode::oparg::IntrinsicFunction1::UnaryPositive => {
                            self.record_pure_op("pos", vec![value])?
                        }
                        pyre_interpreter::bytecode::oparg::IntrinsicFunction1::ListToTuple => {
                            match value {
                                Hlvalue::Constant(Constant {
                                    value: ConstValue::List(items),
                                    ..
                                }) => Hlvalue::Constant(Constant::new(ConstValue::Tuple(items))),
                                other => self.record_pure_op("list_to_tuple", vec![other])?,
                            }
                        }
                        other => {
                            return self.unsupported_rpython(format!(
                                "CALL_INTRINSIC_1 {other:?} is not RPython"
                            ));
                        }
                    };
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::CallIntrinsic2 { .. } => {
                    let right = self.pop_hlvalue()?;
                    let left = self.pop_hlvalue()?;
                    let intrinsic =
                        pyre_interpreter::bytecode::oparg::IntrinsicFunction2::try_from(oparg)
                            .map_err(|_| {
                                FlowContextError::BytecodeCorruption(BytecodeCorruption::new(
                                    format!("invalid CALL_INTRINSIC_2 func {oparg}"),
                                ))
                            })?;
                    match intrinsic {
                    pyre_interpreter::bytecode::oparg::IntrinsicFunction2::SetTypeparamDefault => {
                        let result =
                            self.record_pure_op("set_typeparam_default", vec![left, right])?;
                        self.pushvalue(StackElem::Value(result));
                        Ok(None)
                    }
                    other => self.unsupported_rpython(format!(
                        "CALL_INTRINSIC_2 {other:?} is not RPython"
                    )),
                }
                }
                Instruction::UnaryInvert => {
                    let value = self.pop_hlvalue()?;
                    let result = self.record_pure_op("invert", vec![value])?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::UnaryNot => {
                    let value = self.pop_hlvalue()?;
                    let result = match value {
                        Hlvalue::Constant(constant) => Hlvalue::Constant(Constant::new(
                            ConstValue::Bool(!constant.value.truthy().ok_or_else(|| {
                                FlowContextError::Flowing(FlowingError::new(
                                    "cannot determine truthiness of constant",
                                ))
                            })?),
                        )),
                        other => self.record_pure_op("not_", vec![other])?,
                    };
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::BinaryOp { op: _ } => {
                    let right = self.pop_hlvalue()?;
                    let left = self.pop_hlvalue()?;
                    let result = self.record_maybe_raise_op(
                        binary_opname(
                            BinaryOperator::try_from(oparg).expect("invalid BinaryOp oparg"),
                        ),
                        vec![left, right],
                        Self::common_exception_cases(),
                    )?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::CompareOp { opname: _ } => {
                    let right = self.pop_hlvalue()?;
                    let left = self.pop_hlvalue()?;
                    let result = self.record_pure_op(
                        comparison_opname(
                            ComparisonOperator::try_from(oparg).expect("invalid CompareOp oparg"),
                        ),
                        vec![left, right],
                    )?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::ContainsOp { invert: _ } => {
                    let container = self.pop_hlvalue()?;
                    let item = self.pop_hlvalue()?;
                    let opname = if matches!(
                        pyre_interpreter::bytecode::Invert::try_from(oparg)
                            .expect("invalid ContainsOp invert"),
                        pyre_interpreter::bytecode::Invert::Yes
                    ) {
                        "not_contains"
                    } else {
                        "contains"
                    };
                    let result = self.record_pure_op(opname, vec![item, container])?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::IsOp { invert: _ } => {
                    let right = self.pop_hlvalue()?;
                    let left = self.pop_hlvalue()?;
                    let result = self.record_pure_op("is_", vec![left, right])?;
                    let invert =
                        pyre_interpreter::bytecode::Invert::try_from(oparg).map_err(|_| {
                            FlowContextError::BytecodeCorruption(BytecodeCorruption::new(format!(
                                "invalid IS_OP invert {oparg}"
                            )))
                        })?;
                    let result = if matches!(invert, pyre_interpreter::bytecode::Invert::Yes) {
                        self.record_pure_op("not_", vec![result])?
                    } else {
                        result
                    };
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::BinarySlice => {
                    let w_end = self.pop_hlvalue()?;
                    let w_start = self.pop_hlvalue()?;
                    let w_obj = self.pop_hlvalue()?;
                    let result = self.record_maybe_raise_op(
                        "getslice",
                        vec![w_obj, w_start, w_end],
                        Self::common_exception_cases(),
                    )?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::BuildTuple { .. } => {
                    let items = self.pop_hlvalues(oparg as usize)?;
                    let result = if items
                        .iter()
                        .all(|item| matches!(item, Hlvalue::Constant(_)))
                    {
                        let elements = items
                            .into_iter()
                            .map(|item| match item {
                                Hlvalue::Constant(constant) => constant.value,
                                Hlvalue::Variable(_) => unreachable!(),
                            })
                            .collect();
                        Hlvalue::Constant(Constant::new(ConstValue::Tuple(elements)))
                    } else {
                        self.record_pure_op("newtuple", items)?
                    };
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::BuildString { .. } => {
                    let items = self.pop_hlvalues(oparg as usize)?;
                    let result = if items.iter().all(|item| {
                        matches!(
                            item,
                            Hlvalue::Constant(Constant {
                                value: ConstValue::Str(_),
                                ..
                            })
                        )
                    }) {
                        let mut out = String::new();
                        for item in items {
                            let Hlvalue::Constant(Constant {
                                value: ConstValue::Str(value),
                                ..
                            }) = item
                            else {
                                unreachable!();
                            };
                            out.push_str(&value);
                        }
                        Hlvalue::Constant(Constant::new(ConstValue::Str(out)))
                    } else {
                        self.record_pure_op("buildstr", items)?
                    };
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::BuildList { .. } => {
                    let items = self.pop_hlvalues(oparg as usize)?;
                    let result = if items
                        .iter()
                        .all(|item| matches!(item, Hlvalue::Constant(_)))
                    {
                        let elements = items
                            .into_iter()
                            .map(|item| match item {
                                Hlvalue::Constant(constant) => constant.value,
                                Hlvalue::Variable(_) => unreachable!(),
                            })
                            .collect();
                        Hlvalue::Constant(Constant::new(ConstValue::List(elements)))
                    } else {
                        self.record_pure_op("newlist", items)?
                    };
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::BuildSet { .. } => {
                    let items = self.pop_hlvalues(oparg as usize)?;
                    let result = self.record_pure_op("newset", items)?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::BuildMap { .. } => {
                    let items = self.pop_hlvalues((oparg as usize) * 2)?;
                    let result = self.record_pure_op("newdict", items)?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::BuildSlice { .. } => {
                    let w_step =
                        match pyre_interpreter::bytecode::BuildSliceArgCount::try_from(oparg)
                            .map_err(|_| {
                                FlowContextError::BytecodeCorruption(BytecodeCorruption::new(
                                    format!("invalid BUILD_SLICE argc {oparg}"),
                                ))
                            })? {
                            pyre_interpreter::bytecode::BuildSliceArgCount::Two => {
                                Hlvalue::Constant(Constant::new(ConstValue::None))
                            }
                            pyre_interpreter::bytecode::BuildSliceArgCount::Three => {
                                self.pop_hlvalue()?
                            }
                        };
                    let w_end = self.pop_hlvalue()?;
                    let w_start = self.pop_hlvalue()?;
                    let w_slice = self.record_pure_op("newslice", vec![w_start, w_end, w_step])?;
                    self.pushvalue(StackElem::Value(w_slice));
                    Ok(None)
                }
                Instruction::UnpackSequence { .. } => {
                    let sequence = self.pop_hlvalue()?;
                    let items = self.unpack_sequence(sequence, oparg as usize)?;
                    for item in items.into_iter().rev() {
                        self.pushvalue(StackElem::Value(item));
                    }
                    Ok(None)
                }
                Instruction::UnpackEx { .. } => {
                    self.unsupported_rpython("extended iterable unpacking is not RPython")
                }
                Instruction::GetIter => {
                    let iterable = self.pop_hlvalue()?;
                    let iterator = self.record_maybe_raise_op(
                        "iter",
                        vec![iterable],
                        Self::common_exception_cases(),
                    )?;
                    self.pushvalue(StackElem::Value(iterator));
                    Ok(None)
                }
                Instruction::ForIter { .. } => {
                    let w_iterator = self.pop_hlvalue()?;
                    self.pushvalue(StackElem::Value(w_iterator.clone()));
                    self.blockstack.push(FrameBlock::new(
                        oparg as i64,
                        self.stackdepth(),
                        FrameBlockKind::Iter,
                    ));
                    let w_nextitem = self.record_maybe_raise_op(
                        "next",
                        vec![w_iterator],
                        Self::stop_iteration_cases(),
                    )?;
                    self.blockstack.pop();
                    self.pushvalue(StackElem::Value(w_nextitem));
                    Ok(None)
                }
                Instruction::EndFor | Instruction::PopIter => {
                    self.popvalue();
                    Ok(None)
                }
                Instruction::JumpForward { .. }
                | Instruction::JumpBackward { .. }
                | Instruction::JumpBackwardNoInterrupt { .. } => Ok(Some(oparg as i64)),
                Instruction::PopJumpIfFalse { .. } => {
                    let value = self.pop_hlvalue()?;
                    let w_bool = self.bool_operand(value)?;
                    if !self.guessbool(w_bool)? {
                        Ok(Some(oparg as i64))
                    } else {
                        Ok(None)
                    }
                }
                Instruction::PopJumpIfTrue { .. } => {
                    let value = self.pop_hlvalue()?;
                    let w_bool = self.bool_operand(value)?;
                    if self.guessbool(w_bool)? {
                        Ok(Some(oparg as i64))
                    } else {
                        Ok(None)
                    }
                }
                Instruction::PopJumpIfNone { .. } => {
                    let value = self.pop_hlvalue()?;
                    let is_none = matches!(
                        value,
                        Hlvalue::Constant(Constant {
                            value: ConstValue::None,
                            ..
                        })
                    );
                    if is_none {
                        Ok(Some(oparg as i64))
                    } else {
                        Ok(None)
                    }
                }
                Instruction::PopJumpIfNotNone { .. } => {
                    let value = self.pop_hlvalue()?;
                    let is_none = matches!(
                        value,
                        Hlvalue::Constant(Constant {
                            value: ConstValue::None,
                            ..
                        })
                    );
                    if !is_none {
                        Ok(Some(oparg as i64))
                    } else {
                        Ok(None)
                    }
                }
                Instruction::ImportName { .. } => {
                    let w_fromlist = self.pop_hlvalue()?;
                    let w_level = self.pop_hlvalue()?;
                    let w_modulename = self.getname_w(oparg as usize)?;
                    let w_module = self.record_maybe_raise_op(
                        "import_name",
                        vec![
                            w_modulename,
                            Hlvalue::Constant(self.w_globals.clone()),
                            w_fromlist,
                            w_level,
                        ],
                        vec![exception_class_value("ImportError")],
                    )?;
                    self.pushvalue(StackElem::Value(w_module));
                    Ok(None)
                }
                Instruction::CallFunctionEx => {
                    let w_kwargs_or_null = self.pop_hlvalue()?;
                    let w_starargs = self.pop_hlvalue()?;
                    let w_self_or_null = self.pop_hlvalue()?;
                    let w_function = self.pop_hlvalue()?;
                    let mut arguments = Vec::new();
                    if !is_null_value(&w_self_or_null) {
                        arguments.push(w_self_or_null);
                    }
                    let result = self.call_function(
                        w_function,
                        arguments,
                        HashMap::new(),
                        Some(w_starargs),
                        (!matches!(
                            w_kwargs_or_null,
                            Hlvalue::Constant(Constant {
                                value: ConstValue::None | ConstValue::Placeholder,
                                ..
                            })
                        ))
                        .then_some(w_kwargs_or_null),
                    )?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::ImportFrom { .. } => {
                    let w_name = self.getname_w(oparg as usize)?;
                    let w_module = match self.peekvalue() {
                        StackElem::Value(value) => value,
                        StackElem::Signal(signal) => return Err(FlowContextError::Signal(signal)),
                    };
                    let w_value = self.record_maybe_raise_op(
                        "getattr",
                        vec![w_module, w_name],
                        vec![exception_class_value("ImportError")],
                    )?;
                    self.pushvalue(StackElem::Value(w_value));
                    Ok(None)
                }
                Instruction::Call { .. } => {
                    let mut arguments = self.pop_hlvalues(oparg as usize)?;
                    let w_function = self.pop_hlvalue()?;
                    let w_self_or_null = self.pop_hlvalue()?;
                    if !is_null_value(&w_self_or_null) {
                        arguments.insert(0, w_self_or_null);
                    }
                    let result =
                        self.call_function(w_function, arguments, HashMap::new(), None, None)?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::CallKw { .. } => {
                    let w_keyword_names = self.pop_hlvalue()?;
                    let keyword_names = self.get_keyword_names(w_keyword_names)?;
                    let total_values = self.pop_hlvalues(oparg as usize)?;
                    if total_values.len() < keyword_names.len() {
                        return Err(FlowContextError::BytecodeCorruption(
                            BytecodeCorruption::new(
                                "CALL_KW has more keyword names than argument values",
                            ),
                        ));
                    }
                    let split = total_values.len() - keyword_names.len();
                    let mut positional = total_values[..split].to_vec();
                    let keyword_values = total_values[split..].to_vec();
                    let mut keywords = HashMap::new();
                    for (name, value) in keyword_names.into_iter().zip(keyword_values.into_iter()) {
                        keywords.insert(name, value);
                    }
                    let w_function = self.pop_hlvalue()?;
                    let w_self_or_null = self.pop_hlvalue()?;
                    if !is_null_value(&w_self_or_null) {
                        positional.insert(0, w_self_or_null);
                    }
                    let result =
                        self.call_function(w_function, positional, keywords, None, None)?;
                    self.pushvalue(StackElem::Value(result));
                    Ok(None)
                }
                Instruction::RaiseVarargs { .. } => {
                    let w_exc = match pyre_interpreter::bytecode::oparg::RaiseKind::try_from(oparg)
                        .map_err(|_| {
                            FlowContextError::BytecodeCorruption(BytecodeCorruption::new(format!(
                                "invalid RAISE_VARARGS kind {oparg}"
                            )))
                        })? {
                        pyre_interpreter::bytecode::oparg::RaiseKind::BareRaise => {
                            self.last_exception.clone().unwrap_or_else(|| {
                                FSException::new(
                                    exception_class_value("TypeError"),
                                    exception_instance_value(
                                        "TypeError",
                                        Some("raise: no active exception to re-raise".to_owned()),
                                    ),
                                )
                            })
                        }
                        pyre_interpreter::bytecode::oparg::RaiseKind::Raise => {
                            let raised = self.pop_hlvalue()?;
                            match raised {
                                Hlvalue::Constant(Constant {
                                    value: ConstValue::ExceptionInstance { class, message },
                                    ..
                                }) => FSException::new(
                                    Hlvalue::Constant(Constant::new(ConstValue::ExceptionClass(
                                        class.clone(),
                                    ))),
                                    Hlvalue::Constant(Constant::new(
                                        ConstValue::ExceptionInstance { class, message },
                                    )),
                                ),
                                Hlvalue::Constant(Constant {
                                    value: ConstValue::ExceptionClass(class),
                                    ..
                                }) => FSException::new(
                                    Hlvalue::Constant(Constant::new(ConstValue::ExceptionClass(
                                        class.clone(),
                                    ))),
                                    Hlvalue::Constant(Constant::new(
                                        ConstValue::ExceptionInstance {
                                            class,
                                            message: None,
                                        },
                                    )),
                                ),
                                other => {
                                    FSException::new(exception_class_value("Exception"), other)
                                }
                            }
                        }
                        pyre_interpreter::bytecode::oparg::RaiseKind::RaiseCause
                        | pyre_interpreter::bytecode::oparg::RaiseKind::ReraiseFromStack => {
                            let _cause = self.pop_hlvalue()?;
                            let raised = self.pop_hlvalue()?;
                            match raised {
                                Hlvalue::Constant(Constant {
                                    value: ConstValue::ExceptionInstance { class, message },
                                    ..
                                }) => FSException::new(
                                    Hlvalue::Constant(Constant::new(ConstValue::ExceptionClass(
                                        class.clone(),
                                    ))),
                                    Hlvalue::Constant(Constant::new(
                                        ConstValue::ExceptionInstance { class, message },
                                    )),
                                ),
                                Hlvalue::Constant(Constant {
                                    value: ConstValue::ExceptionClass(class),
                                    ..
                                }) => FSException::new(
                                    Hlvalue::Constant(Constant::new(ConstValue::ExceptionClass(
                                        class.clone(),
                                    ))),
                                    Hlvalue::Constant(Constant::new(
                                        ConstValue::ExceptionInstance {
                                            class,
                                            message: None,
                                        },
                                    )),
                                ),
                                other => {
                                    FSException::new(exception_class_value("Exception"), other)
                                }
                            }
                        }
                    };
                    Err(FlowContextError::Signal(FlowSignal::Raise { w_exc }))
                }
                Instruction::CheckExcMatch => {
                    let w_check_class = self.pop_hlvalue()?;
                    let w_exc_type = self.current_exception_type()?;
                    let matched = self.exception_match(&w_exc_type, &w_check_class)?;
                    self.pushvalue(StackElem::Value(Hlvalue::Constant(Constant::new(
                        ConstValue::Bool(matched),
                    ))));
                    Ok(None)
                }
                Instruction::DeleteFast { .. } => {
                    if self.locals_w[oparg as usize].is_none() {
                        return Err(FlowContextError::Flowing(FlowingError::new(format!(
                            "local variable '{}' referenced before assignment",
                            self.getlocalvarname(oparg as usize)
                        ))));
                    }
                    self.locals_w[oparg as usize] = None;
                    Ok(None)
                }
                Instruction::StoreSubscr => {
                    let w_subscr = self.pop_hlvalue()?;
                    let w_obj = self.pop_hlvalue()?;
                    let w_newvalue = self.pop_hlvalue()?;
                    self.record_side_effect_op(
                        "setitem",
                        vec![w_obj, w_subscr, w_newvalue],
                        Self::common_exception_cases(),
                    )?;
                    Ok(None)
                }
                Instruction::DeleteSubscr => {
                    let w_subscr = self.pop_hlvalue()?;
                    let w_obj = self.pop_hlvalue()?;
                    self.record_side_effect_op(
                        "delitem",
                        vec![w_obj, w_subscr],
                        Self::common_exception_cases(),
                    )?;
                    Ok(None)
                }
                Instruction::StoreSlice => {
                    let w_end = self.pop_hlvalue()?;
                    let w_start = self.pop_hlvalue()?;
                    let w_obj = self.pop_hlvalue()?;
                    let w_newvalue = self.pop_hlvalue()?;
                    self.record_side_effect_op(
                        "setslice",
                        vec![w_obj, w_start, w_end, w_newvalue],
                        Self::common_exception_cases(),
                    )?;
                    Ok(None)
                }
                Instruction::ListAppend { .. } => {
                    let w_value = self.pop_hlvalue()?;
                    let w_list = match self.peekvalue_at((oparg as usize).saturating_sub(1)) {
                        StackElem::Value(value) => value,
                        StackElem::Signal(signal) => return Err(FlowContextError::Signal(signal)),
                    };
                    self.record_side_effect_op(
                        "list_append",
                        vec![w_list, w_value],
                        Self::common_exception_cases(),
                    )?;
                    Ok(None)
                }
                Instruction::ListExtend { .. } => {
                    let w_iterable = self.pop_hlvalue()?;
                    let w_list = match self.peekvalue_at((oparg as usize).saturating_sub(1)) {
                        StackElem::Value(value) => value,
                        StackElem::Signal(signal) => return Err(FlowContextError::Signal(signal)),
                    };
                    self.record_side_effect_op(
                        "list_extend",
                        vec![w_list, w_iterable],
                        Self::common_exception_cases(),
                    )?;
                    Ok(None)
                }
                Instruction::SetAdd { .. } => {
                    let w_value = self.pop_hlvalue()?;
                    let w_set = match self.peekvalue_at((oparg as usize).saturating_sub(1)) {
                        StackElem::Value(value) => value,
                        StackElem::Signal(signal) => return Err(FlowContextError::Signal(signal)),
                    };
                    self.record_side_effect_op(
                        "set_add",
                        vec![w_set, w_value],
                        Self::common_exception_cases(),
                    )?;
                    Ok(None)
                }
                Instruction::SetUpdate { .. } => {
                    let w_other = self.pop_hlvalue()?;
                    let w_set = match self.peekvalue_at((oparg as usize).saturating_sub(1)) {
                        StackElem::Value(value) => value,
                        StackElem::Signal(signal) => return Err(FlowContextError::Signal(signal)),
                    };
                    self.record_side_effect_op(
                        "set_update",
                        vec![w_set, w_other],
                        Self::common_exception_cases(),
                    )?;
                    Ok(None)
                }
                Instruction::MapAdd { .. } => {
                    let w_value = self.pop_hlvalue()?;
                    let w_key = self.pop_hlvalue()?;
                    let w_dict = match self.peekvalue_at((oparg as usize).saturating_sub(1)) {
                        StackElem::Value(value) => value,
                        StackElem::Signal(signal) => return Err(FlowContextError::Signal(signal)),
                    };
                    self.record_side_effect_op(
                        "map_add",
                        vec![w_dict, w_key, w_value],
                        Self::common_exception_cases(),
                    )?;
                    Ok(None)
                }
                Instruction::DictUpdate { .. } | Instruction::DictMerge { .. } => {
                    let w_other = self.pop_hlvalue()?;
                    let w_dict = match self.peekvalue_at((oparg as usize).saturating_sub(1)) {
                        StackElem::Value(value) => value,
                        StackElem::Signal(signal) => return Err(FlowContextError::Signal(signal)),
                    };
                    let opname = if matches!(instruction, Instruction::DictMerge { .. }) {
                        "dict_merge"
                    } else {
                        "dict_update"
                    };
                    self.record_side_effect_op(
                        opname,
                        vec![w_dict, w_other],
                        Self::common_exception_cases(),
                    )?;
                    Ok(None)
                }
                Instruction::BuildTemplate
                | Instruction::BuildInterpolation { .. }
                | Instruction::FormatSimple
                | Instruction::FormatWithSpec
                | Instruction::ConvertValue { .. } => {
                    self.unsupported_rpython("f-strings and template strings are not RPython")
                }
                Instruction::MatchKeys
                | Instruction::MatchMapping
                | Instruction::MatchSequence
                | Instruction::MatchClass { .. } => {
                    self.unsupported_rpython("structural pattern matching is not RPython")
                }
                Instruction::GetAIter
                | Instruction::GetANext
                | Instruction::GetAwaitable { .. }
                | Instruction::EndAsyncFor
                | Instruction::EndSend
                | Instruction::Send { .. }
                | Instruction::CleanupThrow => {
                    self.unsupported_rpython("async iteration is not RPython")
                }
                Instruction::PushExcInfo => {
                    let w_exc_value = self.pop_hlvalue()?;
                    let restore_token = exception_restore_token(
                        self.pending_exception_restore.take().unwrap_or(None),
                    );
                    self.pushvalue(restore_token);
                    self.pushvalue(StackElem::Value(w_exc_value));
                    Ok(None)
                }
                Instruction::PopExcept => {
                    let token = self.popvalue();
                    self.restore_exception_from_token(token)?;
                    Ok(None)
                }
                Instruction::Reraise { .. } => {
                    let w_exc = self
                        .current_stack_exception()
                        .or_else(|| self.last_exception.clone())
                        .ok_or_else(|| {
                            FlowContextError::Flowing(FlowingError::new(
                                "RERAISE without an active exception",
                            ))
                        })?;
                    Err(FlowContextError::Signal(FlowSignal::Raise { w_exc }))
                }
                Instruction::WithExceptStart => {
                    if self.stack.len() < 5 {
                        return Err(FlowContextError::Flowing(FlowingError::new(
                            "WITH_EXCEPT_START expects __exit__, self-or-null, and exception state",
                        )));
                    }
                    let w_exitfunc = match self.peekvalue_at(3) {
                        StackElem::Value(value) => value,
                        StackElem::Signal(signal) => return Err(FlowContextError::Signal(signal)),
                    };
                    let w_self_or_null = match self.peekvalue_at(4) {
                        StackElem::Value(value) => value,
                        StackElem::Signal(signal) => return Err(FlowContextError::Signal(signal)),
                    };
                    let w_exc_value = self
                        .last_exception
                        .as_ref()
                        .map(|w_exc| w_exc.w_value.clone())
                        .ok_or_else(|| {
                            FlowContextError::Flowing(FlowingError::new(
                                "WITH_EXCEPT_START without an active exception",
                            ))
                        })?;
                    let mut args = vec![w_exitfunc.clone()];
                    if !is_null_value(&w_self_or_null) {
                        args.push(w_self_or_null);
                    }
                    args.push(w_exc_value.clone());
                    args.push(w_exc_value);
                    args.push(Hlvalue::Constant(Constant::new(ConstValue::None)));
                    let opname = if matches!(w_exitfunc, Hlvalue::Constant(_)) {
                        "direct_call"
                    } else {
                        "indirect_call"
                    };
                    let _ =
                        self.record_maybe_raise_op(opname, args, Self::common_exception_cases())?;
                    // RPython WITH_CLEANUP cannot suppress exceptions; keep the
                    // 3.14 host bytecode on the re-raise path.
                    self.pushvalue(StackElem::Value(Hlvalue::Constant(Constant::new(
                        ConstValue::Bool(false),
                    ))));
                    Ok(None)
                }
                Instruction::LoadSpecial { method } => {
                    let name = match method.get(oparg.into()) {
                        pyre_interpreter::bytecode::oparg::SpecialMethod::Enter => "__enter__",
                        pyre_interpreter::bytecode::oparg::SpecialMethod::Exit => "__exit__",
                        pyre_interpreter::bytecode::oparg::SpecialMethod::AEnter
                        | pyre_interpreter::bytecode::oparg::SpecialMethod::AExit => {
                            return self.unsupported_rpython("async with is not RPython");
                        }
                    };
                    let w_obj = self.pop_hlvalue()?;
                    let w_name = Hlvalue::Constant(Constant::new(ConstValue::Str(name.to_owned())));
                    let w_method = self.record_maybe_raise_op(
                        "getattr",
                        vec![w_obj, w_name],
                        Self::common_exception_cases(),
                    )?;
                    self.pushvalue(StackElem::Value(null_value()));
                    self.pushvalue(StackElem::Value(w_method));
                    Ok(None)
                }
                Instruction::LoadSuperAttr { .. } => {
                    self.unsupported_rpython("special method lookup is not RPython")
                }
                Instruction::MakeFunction => {
                    let w_codeobj = self.pop_hlvalue()?;
                    let function = self.newfunction(w_codeobj, Vec::new())?;
                    self.pushvalue(StackElem::Value(function));
                    Ok(None)
                }
                Instruction::SetFunctionAttribute { .. } => {
                    let flag = MakeFunctionFlag::try_from(oparg).map_err(|_| {
                        FlowContextError::BytecodeCorruption(BytecodeCorruption::new(format!(
                            "invalid SET_FUNCTION_ATTRIBUTE flag {oparg}"
                        )))
                    })?;
                    let function = self.pop_hlvalue()?;
                    let attr = self.pop_hlvalue()?;
                    let mut graph_func = match function {
                        Hlvalue::Constant(Constant {
                            value: ConstValue::Function(func),
                            ..
                        }) => *func,
                        other => {
                            return Err(FlowContextError::Flowing(FlowingError::new(format!(
                                "SET_FUNCTION_ATTRIBUTE expected function constant, got {other:?}"
                            ))));
                        }
                    };
                    match flag {
                        MakeFunctionFlag::Defaults => {
                            let defaults = match attr {
                                Hlvalue::Constant(Constant {
                                    value: ConstValue::Tuple(items),
                                    ..
                                })
                                | Hlvalue::Constant(Constant {
                                    value: ConstValue::List(items),
                                    ..
                                }) => items.into_iter().map(Constant::new).collect(),
                                other => {
                                    return Err(FlowContextError::Flowing(FlowingError::new(
                                        format!(
                                            "SET_FUNCTION_ATTRIBUTE Defaults expected constant tuple/list, got {other:?}"
                                        ),
                                    )));
                                }
                            };
                            graph_func.defaults = defaults;
                        }
                        MakeFunctionFlag::Closure => {
                            return self.unsupported_rpython(
                                "RPython functions cannot create closures inside functions",
                            );
                        }
                        MakeFunctionFlag::KwOnlyDefaults
                        | MakeFunctionFlag::Annotations
                        | MakeFunctionFlag::TypeParams
                        | MakeFunctionFlag::Annotate => {
                            return self.unsupported_rpython(format!(
                                "SET_FUNCTION_ATTRIBUTE {flag:?} is not RPython"
                            ));
                        }
                    }
                    self.pushvalue(StackElem::Value(Hlvalue::Constant(Constant::new(
                        ConstValue::Function(Box::new(graph_func)),
                    ))));
                    Ok(None)
                }
                Instruction::YieldValue { .. } => {
                    let w_result = self.pop_hlvalue()?;
                    let _ = self.record_pure_op("yield", vec![w_result])?;
                    self.pushvalue(StackElem::Value(Hlvalue::Constant(Constant::new(
                        ConstValue::None,
                    ))));
                    Ok(None)
                }
                Instruction::ReturnValue => {
                    let value = self.pop_hlvalue()?;
                    Err(FlowContextError::Signal(FlowSignal::Return {
                        w_value: value,
                    }))
                }
                other => Err(FlowContextError::Flowing(FlowingError::new(format!(
                    "opcode handler not implemented yet: {other:?}"
                )))),
            }
        })();
        match step {
            Ok(Some(offset)) => Ok(offset),
            Ok(None) => Ok(decoded_next_offset as i64),
            Err(FlowContextError::Signal(signal)) => self.unroll(signal),
            Err(err) => Err(err),
        }
    }
}

fn binary_opname(op: BinaryOperator) -> &'static str {
    match op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => "add",
        BinaryOperator::And | BinaryOperator::InplaceAnd => "and_",
        BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => "floordiv",
        BinaryOperator::Lshift | BinaryOperator::InplaceLshift => "lshift",
        BinaryOperator::MatrixMultiply | BinaryOperator::InplaceMatrixMultiply => "matmul",
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => "mul",
        BinaryOperator::Or | BinaryOperator::InplaceOr => "or_",
        BinaryOperator::Power | BinaryOperator::InplacePower => "pow",
        BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => "mod",
        BinaryOperator::Rshift | BinaryOperator::InplaceRshift => "rshift",
        BinaryOperator::Subscr => "getitem",
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => "sub",
        BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide => "truediv",
        BinaryOperator::Xor | BinaryOperator::InplaceXor => "xor",
    }
}

fn comparison_opname(op: ComparisonOperator) -> &'static str {
    match op {
        ComparisonOperator::Less => "lt",
        ComparisonOperator::LessOrEqual => "le",
        ComparisonOperator::Equal => "eq",
        ComparisonOperator::NotEqual => "ne",
        ComparisonOperator::Greater => "gt",
        ComparisonOperator::GreaterOrEqual => "ge",
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use pyre_interpreter::compile::{self, CodeObject, Mode};
    use std::rc::Rc;

    fn var() -> Hlvalue {
        Hlvalue::Variable(Variable::new())
    }

    fn iconst(n: i64) -> Hlvalue {
        Hlvalue::Constant(Constant::new(ConstValue::Int(n)))
    }

    fn compile_function_body(src: &str) -> CodeObject {
        let module = compile::compile_source(src, Mode::Exec).expect("compile should succeed");
        module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("source should contain a function body")
    }

    fn flow_context(src: &str) -> FlowContext {
        let host = HostCode::from_code(&compile_function_body(src));
        let inputargs = (0..host.formalargcount())
            .map(|_| Hlvalue::Variable(Variable::new()))
            .collect();
        let startblock = Block::shared(inputargs);
        let graph = FunctionGraph::new(host.co_name.clone(), startblock.clone());
        let mut ctx = FlowContext::new(graph, host);
        ctx.recorder = Some(Recorder::Block(BlockRecorder::new(
            startblock,
            FrameState::new(vec![None; ctx.nlocals], Vec::new(), None, Vec::new(), 0),
        )));
        ctx
    }

    fn flow_context_with_globals(src: &str, globals: HashMap<String, ConstValue>) -> FlowContext {
        let host = HostCode::from_code(&compile_function_body(src));
        let inputargs = (0..host.formalargcount())
            .map(|_| Hlvalue::Variable(Variable::new()))
            .collect();
        let startblock = Block::shared(inputargs);
        let mut graph = FunctionGraph::new(host.co_name.clone(), startblock.clone());
        let mut func = GraphFunc::new(
            host.co_name.clone(),
            Constant::new(ConstValue::Dict(globals)),
        );
        func.code = Some(Box::new(host.clone()));
        func.filename = Some(host.co_filename.clone());
        func.firstlineno = Some(host.co_firstlineno);
        graph.func = Some(func);
        let mut ctx = FlowContext::new(graph, host);
        ctx.recorder = Some(Recorder::Block(BlockRecorder::new(
            startblock,
            FrameState::new(vec![None; ctx.nlocals], Vec::new(), None, Vec::new(), 0),
        )));
        ctx
    }

    #[test]
    fn return_args_and_rebuild_roundtrip() {
        let v = var();
        let sig = FlowSignal::Return { w_value: v.clone() };
        assert_eq!(sig.args(), vec![v.clone()]);
        let rebuilt = FlowSignal::rebuild_with_args(FlowSignalTag::Return, sig.args());
        assert_eq!(rebuilt, sig);
    }

    #[test]
    fn raise_args_and_rebuild_roundtrip() {
        let t = iconst(1);
        let v = iconst(2);
        let sig = FlowSignal::Raise {
            w_exc: FSException::new(t.clone(), v.clone()),
        };
        assert_eq!(sig.args(), vec![t, v]);
        let rebuilt = FlowSignal::rebuild_with_args(FlowSignalTag::Raise, sig.args());
        assert_eq!(rebuilt, sig);
    }

    #[test]
    fn block_handles_loop_control_signals() {
        let block = FrameBlock::new(11, 0, FrameBlockKind::Loop);
        assert!(block.handles(&FlowSignal::Break));
        assert!(block.handles(&FlowSignal::Continue { jump_to: 3 }));
        assert!(!block.handles(&FlowSignal::Return { w_value: iconst(1) }));
    }

    #[test]
    fn continue_wraps_jump_offset() {
        let sig = FlowSignal::Continue { jump_to: 42 };
        assert_eq!(sig.args(), vec![iconst(42)]);
        let rebuilt = FlowSignal::rebuild_with_args(FlowSignalTag::Continue, sig.args());
        assert_eq!(rebuilt, sig);
    }

    #[test]
    fn raise_implicit_nomoreblocks_rewrites_to_assertionerror() {
        let mut ctx = flow_context("def f():\n    return 1\n");
        let signal = FlowSignal::RaiseImplicit {
            w_exc: FSException::new(
                exception_class_value("ValueError"),
                exception_instance_value("ValueError", Some("x".to_owned())),
            ),
        };
        assert_eq!(
            signal.nomoreblocks(&mut ctx),
            Err(FlowContextError::StopFlowing)
        );
        let exits = ctx.graph.startblock.borrow().exits.clone();
        assert_eq!(exits.len(), 1);
        let link = exits[0].borrow();
        assert_eq!(link.args[0], Some(exception_class_value("AssertionError")));
        assert_eq!(
            link.args[1],
            Some(exception_instance_value(
                "AssertionError",
                Some("implicit ValueError shouldn't occur".to_owned())
            ))
        );
    }

    #[test]
    fn raise_nomoreblocks_importerror_aborts_flow() {
        let mut ctx = flow_context("def f():\n    return 1\n");
        let signal = FlowSignal::Raise {
            w_exc: FSException::new(
                exception_class_value("ImportError"),
                exception_instance_value("ImportError", Some("boom".to_owned())),
            ),
        };
        let err = signal.nomoreblocks(&mut ctx).unwrap_err();
        match err {
            FlowContextError::Flowing(flowing) => {
                assert_eq!(flowing.message, "ImportError is raised in RPython: boom");
            }
            other => panic!("expected FlowingError, got {other:?}"),
        }
        assert!(ctx.graph.startblock.borrow().exits.is_empty());
    }

    #[test]
    fn guessbool_uses_python_truthiness_for_constants() {
        let mut ctx = flow_context("def f():\n    return 1\n");
        assert!(ctx.guessbool(iconst(1)).unwrap());
        assert!(!ctx.guessbool(iconst(0)).unwrap());
        assert!(
            ctx.guessbool(Hlvalue::Constant(Constant::new(ConstValue::Str(
                "x".to_owned()
            ))))
            .unwrap()
        );
        assert!(
            !ctx.guessbool(Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![]))))
                .unwrap()
        );
    }

    #[test]
    fn find_global_reads_function_globals_before_builtins() {
        let mut globals = HashMap::new();
        globals.insert("sentinel".to_owned(), ConstValue::Int(42));
        let ctx = flow_context_with_globals("def f():\n    return sentinel\n", globals);
        assert_eq!(ctx.find_global("sentinel").unwrap(), iconst(42));
        assert!(matches!(
            ctx.find_global("print").unwrap(),
            Hlvalue::Constant(Constant {
                value: ConstValue::Builtin(BuiltinObject::Function(BuiltinFunction::Print)),
                ..
            })
        ));
    }

    #[test]
    fn merge_links_preserve_undefined_locals_in_link_args() {
        let target = Block::shared(vec![Hlvalue::Variable(Variable::new())]);
        let link = Link::new_mergeable(vec![None], Some(target), None);
        assert_eq!(link.args, vec![None]);
    }

    #[test]
    fn exception_match_handles_subclasses_and_tuples() {
        let mut ctx = flow_context("def f():\n    return 1\n");
        assert!(
            ctx.exception_match(
                &exception_class_value("ValueError"),
                &exception_class_value("Exception")
            )
            .unwrap()
        );
        assert!(
            ctx.exception_match(
                &exception_class_value("ValueError"),
                &Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![
                    ConstValue::ExceptionClass(ExceptionClass::builtin("ImportError")),
                    ConstValue::ExceptionClass(ExceptionClass::builtin("ValueError")),
                ]))),
            )
            .unwrap()
        );
        let err = ctx
            .exception_match(
                &exception_class_value("ValueError"),
                &exception_class_value("AssertionError"),
            )
            .unwrap_err();
        match err {
            FlowContextError::Flowing(flowing) => {
                assert!(
                    flowing
                        .message
                        .contains("Catching NotImplementedError, AssertionError")
                );
            }
            other => panic!("expected FlowingError, got {other:?}"),
        }
    }

    #[test]
    fn handle_bytecode_records_simple_addition() {
        let mut ctx = flow_context("def f(x, y):\n    return x + y\n");
        let args = ctx.graph.startblock.borrow().inputargs.clone();
        ctx.locals_w[0] = Some(args[0].clone());
        ctx.locals_w[1] = Some(args[1].clone());
        let mut offset = 0i64;
        loop {
            match ctx.handle_bytecode(offset) {
                Ok(next) => offset = next,
                Err(FlowContextError::StopFlowing) => break,
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
        let startblock = ctx.graph.startblock.borrow();
        assert_eq!(startblock.operations.len(), 1);
        assert_eq!(startblock.operations[0].opname, "add");
        assert_eq!(startblock.exits.len(), 1);
    }

    #[test]
    fn handle_bytecode_rejects_store_global() {
        let mut ctx = flow_context("def f():\n    global g\n    g = 1\n");
        let mut offset = 0i64;
        loop {
            match ctx.handle_bytecode(offset) {
                Ok(next) => offset = next,
                Err(FlowContextError::Flowing(flowing)) => {
                    assert!(
                        flowing
                            .message
                            .contains("Attempting to modify global variable")
                    );
                    break;
                }
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn handle_bytecode_lowers_print_via_rpython_print_helpers() {
        let mut ctx = flow_context("def f(x):\n    return print(x, end='\\n')\n");
        let args = ctx.graph.startblock.borrow().inputargs.clone();
        ctx.locals_w[0] = Some(args[0].clone());
        let mut offset = 0i64;
        loop {
            match ctx.handle_bytecode(offset) {
                Ok(next) => offset = next,
                Err(FlowContextError::StopFlowing) => break,
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
        let startblock = ctx.graph.startblock.borrow();
        assert!(
            startblock.operations.iter().any(|op| op.opname == "str"),
            "expected print lowering to stringify arguments, got {:?}",
            startblock.operations
        );
        assert!(
            startblock
                .operations
                .iter()
                .filter(|op| op.opname == "direct_call")
                .count()
                >= 2,
            "expected print lowering to rpython_print_* direct calls, got {:?}",
            startblock.operations
        );
        assert_eq!(startblock.exits.len(), 1);
    }

    #[test]
    fn unpack_sequence_raises_valueerror_on_length_mismatch() {
        let mut ctx = flow_context("def f():\n    return 1\n");
        let err = ctx
            .unpack_sequence(
                Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![ConstValue::Int(1)]))),
                2,
            )
            .unwrap_err();
        match err {
            FlowContextError::Signal(FlowSignal::Raise { w_exc }) => {
                assert_eq!(w_exc.w_type, exception_class_value("ValueError"));
            }
            other => panic!("expected Raise(ValueError), got {other:?}"),
        }
    }

    #[test]
    fn build_flow_constructs_branch_graph() {
        let mut ctx = flow_context(
            "def f(x):\n    if x:\n        y = 1\n    else:\n        y = 2\n    return y\n",
        );
        ctx.recorder = None;
        ctx.build_flow().expect("build_flow should succeed");
        let blocks = ctx.graph.iterblocks();
        assert!(
            blocks.len() >= 4,
            "expected start, then, else, and return-adjacent blocks; got {}",
            blocks.len()
        );
        assert!(
            blocks.iter().any(|block| block.borrow().exits.len() == 2),
            "expected a boolean split block, got {:?}",
            blocks
                .iter()
                .map(|block| block.borrow().exits.len())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn build_flow_preserves_unpack_sequence_order() {
        let mut ctx = flow_context("def f():\n    a, b = (1, 2)\n    return a, b\n");
        ctx.recorder = None;
        ctx.build_flow().expect("build_flow should succeed");
        let return_links: Vec<_> = ctx
            .graph
            .iterlinks()
            .into_iter()
            .filter(|link| {
                link.borrow()
                    .target
                    .as_ref()
                    .is_some_and(|target| Rc::ptr_eq(target, &ctx.graph.returnblock))
            })
            .collect();
        assert_eq!(return_links.len(), 1);
        let link = return_links[0].borrow();
        assert_eq!(
            link.args[0],
            Some(Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![
                ConstValue::Int(1),
                ConstValue::Int(2),
            ]))))
        );
    }

    #[test]
    fn handle_bytecode_rejects_nested_class_definition() {
        let mut ctx = flow_context("def f():\n    class C:\n        pass\n    return C\n");
        let mut offset = 0i64;
        loop {
            match ctx.handle_bytecode(offset) {
                Ok(next) => offset = next,
                Err(FlowContextError::Flowing(flowing)) => {
                    assert!(
                        flowing
                            .message
                            .contains("defining classes inside functions")
                    );
                    break;
                }
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn handle_bytecode_supports_call_function_ex_with_starargs() {
        let mut ctx = flow_context("def f(g, xs):\n    return g(*xs)\n");
        let args = ctx.graph.startblock.borrow().inputargs.clone();
        ctx.locals_w[0] = Some(args[0].clone());
        ctx.locals_w[1] = Some(args[1].clone());
        let mut offset = 0i64;
        loop {
            match ctx.handle_bytecode(offset) {
                Ok(next) => offset = next,
                Err(FlowContextError::StopFlowing) => break,
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
        let startblock = ctx.graph.startblock.borrow();
        assert!(
            startblock
                .operations
                .iter()
                .any(|op| op.opname == "indirect_call"
                    || op.opname == "direct_call"
                    || op.opname == "call_args"),
            "expected CALL_FUNCTION_EX lowering to a call op, got {:?}",
            startblock.operations
        );
    }

    #[test]
    fn build_flow_handles_for_iter_loop() {
        let mut ctx = flow_context(
            "def f(xs):\n    total = 0\n    for x in xs:\n        total = total + x\n    return total\n",
        );
        ctx.recorder = None;
        ctx.build_flow().expect("build_flow should succeed");
        let ops: Vec<String> = ctx
            .graph
            .iterblockops()
            .into_iter()
            .map(|(_, op)| op.opname)
            .collect();
        assert!(
            ops.iter().any(|op| op == "iter"),
            "expected iter op in built graph: {ops:?}"
        );
        assert!(
            ops.iter().any(|op| op == "next"),
            "expected next op in built graph: {ops:?}"
        );
        assert!(
            ops.iter().any(|op| op == "add"),
            "expected loop body add op in built graph: {ops:?}"
        );
    }

    #[test]
    fn build_flow_handles_try_except_via_exception_table() {
        let mut ctx = flow_context(
            "def f(x):\n    try:\n        return 1 / x\n    except Exception:\n        return 0\n",
        );
        ctx.recorder = None;
        ctx.build_flow().expect("build_flow should succeed");
        let ops: Vec<String> = ctx
            .graph
            .iterblockops()
            .into_iter()
            .map(|(_, op)| op.opname)
            .collect();
        assert!(
            ops.iter().any(|op| op == "truediv"),
            "expected protected operation in graph: {ops:?}"
        );
        assert!(
            ctx.graph
                .iterblocks()
                .iter()
                .any(|block| block.borrow().canraise()),
            "expected an exception split block in graph"
        );
        assert!(
            ctx.graph.iterlinks().iter().any(|link| {
                link.borrow()
                    .exitcase
                    .as_ref()
                    .is_some_and(|case| *case == exception_class_value("Exception"))
            }),
            "expected an exception handler edge, got {:?}",
            ctx.graph
                .iterlinks()
                .iter()
                .map(|link| link.borrow().args.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn build_flow_models_make_function_with_constant_defaults() {
        let mut ctx = flow_context("def f():\n    def g(x=1):\n        return x\n    return g\n");
        ctx.recorder = None;
        ctx.build_flow().expect("build_flow should succeed");
        let return_links: Vec<_> = ctx
            .graph
            .iterlinks()
            .into_iter()
            .filter(|link| {
                link.borrow()
                    .target
                    .as_ref()
                    .is_some_and(|target| Rc::ptr_eq(target, &ctx.graph.returnblock))
            })
            .collect();
        assert_eq!(return_links.len(), 1);
        let link = return_links[0].borrow();
        match link.args[0].as_ref() {
            Some(Hlvalue::Constant(Constant {
                value: ConstValue::Function(func),
                ..
            })) => {
                assert_eq!(func.name, "g");
                assert_eq!(func.defaults, vec![Constant::new(ConstValue::Int(1))]);
                assert!(func.code.is_some(), "function constant must carry __code__");
            }
            other => panic!("expected function constant return, got {other:?}"),
        }
    }

    #[test]
    fn build_flow_handles_try_finally_via_exception_table() {
        let mut ctx = flow_context(
            "def f(x):\n    try:\n        y = 1 / x\n    finally:\n        y = -x\n    return y\n",
        );
        ctx.recorder = None;
        ctx.build_flow().expect("build_flow should succeed");
        let ops: Vec<String> = ctx
            .graph
            .iterblockops()
            .into_iter()
            .map(|(_, op)| op.opname)
            .collect();
        assert!(
            ops.iter().any(|op| op == "truediv"),
            "expected protected operation in graph: {ops:?}"
        );
        assert!(
            ops.iter().any(|op| op == "neg"),
            "expected finally-body operation in graph: {ops:?}"
        );
    }

    #[test]
    fn build_flow_handles_with_statement() {
        let mut ctx = flow_context("def f(cm):\n    with cm as x:\n        return x\n");
        ctx.recorder = None;
        ctx.build_flow().expect("build_flow should succeed");
        let ops: Vec<String> = ctx
            .graph
            .iterblockops()
            .into_iter()
            .map(|(_, op)| op.opname)
            .collect();
        assert!(
            ops.iter().filter(|op| op.as_str() == "getattr").count() >= 2,
            "expected __enter__/__exit__ lookups in graph: {ops:?}"
        );
        assert!(
            ops.iter()
                .filter(|op| op.as_str() == "direct_call" || op.as_str() == "indirect_call")
                .count()
                >= 2,
            "expected __enter__/__exit__ calls in graph: {ops:?}"
        );
    }

    #[test]
    fn build_flow_handles_with_statement_exception_path() {
        let mut ctx = flow_context(
            "def f(cm):\n    try:\n        with cm:\n            1 / 0\n    except ZeroDivisionError:\n        return 5\n",
        );
        ctx.recorder = None;
        ctx.build_flow().expect("build_flow should succeed");
        let ops: Vec<String> = ctx
            .graph
            .iterblockops()
            .into_iter()
            .map(|(_, op)| op.opname)
            .collect();
        assert!(
            ops.iter().any(|op| op == "truediv"),
            "expected protected operation in graph: {ops:?}"
        );
        assert!(
            ops.iter().filter(|op| op.as_str() == "getattr").count() >= 2,
            "expected with-special lookups in graph: {ops:?}"
        );
        assert!(
            ops.iter()
                .filter(|op| op.as_str() == "direct_call" || op.as_str() == "indirect_call")
                .count()
                >= 2,
            "expected __enter__/__exit__ calls in graph: {ops:?}"
        );
    }
}
