/// Python bytecode → JitCode compiler.
///
/// RPython: rpython/jit/codewriter/codewriter.py
///
/// RPython's CodeWriter transforms flow graphs → JitCode via a 4-step pipeline:
///   1. jtransform  — rewrite operations
///   2. regalloc    — assign registers
///   3. flatten     — CFG → linear SSARepr
///   4. assemble    — SSARepr → JitCode bytecode
///
/// For pyre, Python bytecodes are already linearized and register-allocated
/// (fast locals = registers, value stack = runtime stack). Steps 1-3 collapse
/// into a single bytecode-to-bytecode translation.
///
/// The Assembler (majit JitCodeBuilder) is RPython's assembler.py equivalent.
use std::cell::{OnceCell, RefCell, UnsafeCell};
use std::collections::HashMap;
use std::rc::Rc;

use majit_ir::OpCode;
use majit_metainterp::jitcode::{JitCode, JitCodeBuilder};
use pyre_jit_trace::{PyJitCode, PyJitCodeMetadata};

use super::assembler::Assembler;
use super::ssa_emitter::SSAReprEmitter;
use pyre_interpreter::bytecode::{CodeFlags, CodeObject, Instruction, OpArgState};
use pyre_interpreter::runtime_ops::{binary_op_tag, compare_op_tag};

use super::flatten::{
    CallFlavor, DescrOperand, GraphFlattener, Insn, Kind, ListOfKind, Operand, Register, ResKind,
    SSARepr, TLabel,
};

// ---------------------------------------------------------------------------
// RPython: codewriter/flatten.py KINDS = ['int', 'ref', 'float']
// ---------------------------------------------------------------------------

/// Python `var_num` → flat index into the `locals_cells_stack_w`
/// virtualizable array.
///
/// PyFrame lays out locals, cells, and the value stack in a single
/// vector; `var_num` from `LOAD_FAST`/`STORE_FAST` is already a direct
/// offset into that vector (no indirection).
///
/// jtransform.py:1877 do_fixed_list_getitem / :1898 do_fixed_list_setitem
/// rewrite portal local accesses into `getarrayitem_vable_r` /
/// `setarrayitem_vable_r` against the virtualizable array.
#[inline]
fn local_to_vable_slot(var_num: usize) -> usize {
    var_num
}

#[inline]
fn entry_arg_slots(code: &CodeObject) -> usize {
    let mut argcount = code.arg_count as usize + code.kwonlyarg_count as usize;
    if code.flags.contains(CodeFlags::VARARGS) {
        argcount += 1;
    }
    if code.flags.contains(CodeFlags::VARKEYWORDS) {
        argcount += 1;
    }
    argcount
}

fn entry_inputargs(code: &CodeObject) -> Vec<super::flow::FlowValue> {
    (0..entry_arg_slots(code))
        .map(|idx| {
            super::flow::Variable::new(super::flow::VariableId(idx as u32), Kind::Ref).into()
        })
        .collect()
}

fn portal_graph_inputvars(code: &CodeObject) -> (super::flow::Variable, super::flow::Variable) {
    let base = entry_arg_slots(code) as u32;
    (
        super::flow::Variable::new(super::flow::VariableId(base), Kind::Ref),
        super::flow::Variable::new(super::flow::VariableId(base + 1), Kind::Ref),
    )
}

fn graph_entry_inputargs(code: &CodeObject, portal_inputs: bool) -> Vec<super::flow::FlowValue> {
    let mut inputargs = entry_inputargs(code);
    if portal_inputs {
        let (frame, ec) = portal_graph_inputvars(code);
        inputargs.push(frame.into());
        inputargs.push(ec.into());
    }
    inputargs
}

fn portal_graph_inputvars_from_startblock(
    graph: &super::flow::FunctionGraph,
) -> (super::flow::Variable, super::flow::Variable) {
    let startblock = graph.startblock.borrow();
    let len = startblock.inputargs.len();
    assert!(
        len >= 2,
        "portal graph startblock missing frame/ec inputargs"
    );
    let frame = match &startblock.inputargs[len - 2] {
        super::flow::FlowValue::Variable(variable) => *variable,
        other => panic!("portal graph frame inputarg must be Variable, got {other:?}"),
    };
    let ec = match &startblock.inputargs[len - 1] {
        super::flow::FlowValue::Variable(variable) => *variable,
        other => panic!("portal graph ec inputarg must be Variable, got {other:?}"),
    };
    (frame, ec)
}

fn flow_value_kind(value: &super::flow::FlowValue) -> Kind {
    match value {
        super::flow::FlowValue::Variable(variable) => variable
            .kind
            .expect("flow graph variable missing kind in jit_merge_point arg"),
        super::flow::FlowValue::Constant(constant) => constant
            .kind
            .expect("flow graph constant missing kind in jit_merge_point arg"),
    }
}

fn make_three_flow_lists(values: &[super::flow::FlowValue]) -> Vec<super::flow::SpaceOperationArg> {
    let mut ints = Vec::new();
    let mut refs = Vec::new();
    let mut floats = Vec::new();
    for value in values {
        match flow_value_kind(value) {
            Kind::Int => ints.push(value.clone()),
            Kind::Ref => refs.push(value.clone()),
            Kind::Float => floats.push(value.clone()),
        }
    }
    vec![
        super::flow::FlowListOfKind::new(Kind::Int, ints).into(),
        super::flow::FlowListOfKind::new(Kind::Ref, refs).into(),
        super::flow::FlowListOfKind::new(Kind::Float, floats).into(),
    ]
}

fn portal_jit_merge_point_graph_args(
    graph: &super::flow::FunctionGraph,
    next_instr: usize,
    w_code: *const (),
) -> Vec<super::flow::SpaceOperationArg> {
    let (frame, ec) = portal_graph_inputvars_from_startblock(graph);
    let greens = vec![
        super::flow::Constant::signed(next_instr as i64).into(),
        super::flow::Constant::signed(0).into(),
        super::flow::Constant::opaque(format!("pycode@{w_code:p}"), Some(Kind::Ref)).into(),
    ];
    let reds = vec![frame.into(), ec.into()];
    let mut args = vec![super::flow::Constant::signed(0).into()];
    args.extend(make_three_flow_lists(&greens));
    args.extend(make_three_flow_lists(&reds));
    args
}

fn frame_blocks_for_offset(code: &CodeObject, next_offset: usize) -> Vec<FrameBlock> {
    if next_offset >= code.instructions.len() {
        return Vec::new();
    }

    pyre_interpreter::bytecode::decode_exception_table(&code.exceptiontable)
        .into_iter()
        .filter(|entry| next_offset >= entry.start as usize && next_offset < entry.end as usize)
        .map(|entry| FrameBlock {
            start_offset: entry.start as usize,
            end_offset: entry.end as usize,
            handler_offset: entry.target as usize,
            stack_depth: entry.depth as u16,
            push_lasti: entry.push_lasti,
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrameBlock {
    start_offset: usize,
    end_offset: usize,
    handler_offset: usize,
    stack_depth: u16,
    push_lasti: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrameState {
    /// `rpython/flowspace/framestate.py:20` `locals_w`.
    locals_w: Vec<Option<super::flow::FlowValue>>,
    /// `framestate.py:21` `stack`.
    stack: Vec<super::flow::FlowValue>,
    /// `framestate.py:22` `last_exception`.
    last_exception: Option<(super::flow::FlowValue, super::flow::FlowValue)>,
    /// `framestate.py:23` `blocklist`.
    blocklist: Vec<FrameBlock>,
    /// `framestate.py:24` `next_offset`.
    next_offset: usize,
    /// Graph-level portal red slots: the `(frame, ec)` Variables that
    /// flow through every block of a portal graph.  Populated on the
    /// entry FrameState (via `entry_frame_state(code, portal_inputs=
    /// true)`) and propagated through block transitions unchanged —
    /// portal Variables carry graph-level identity, not per-block SSA
    /// names, so `copy()` passes them through without freshening and
    /// `union()` requires both sides to agree.  Mirrors the red
    /// carry-through in `rpython/jit/metainterp/warmspot.py` where the
    /// jitdriver_sd.reds list names `(jitframe, ec)` that the portal
    /// interpreter function threads through every iteration of the
    /// loop.  Participates in `mergeable()` after the last-exception
    /// pair so backedge `Link.args` produced by `getoutputargs()` stay
    /// aligned with the portal `startblock.inputargs` appended by
    /// `graph_entry_inputargs(code, portal_inputs=true)`.
    portal_extras: Option<(super::flow::FlowValue, super::flow::FlowValue)>,
}

impl FrameState {
    fn new(
        locals_w: Vec<Option<super::flow::FlowValue>>,
        stack: Vec<super::flow::FlowValue>,
        last_exception: Option<(super::flow::FlowValue, super::flow::FlowValue)>,
        blocklist: Vec<FrameBlock>,
        next_offset: usize,
    ) -> Self {
        Self {
            locals_w,
            stack,
            last_exception,
            blocklist,
            next_offset,
            portal_extras: None,
        }
    }

    /// Seed the graph-level portal `(frame, ec)` pair on this state.
    /// Called from `entry_frame_state(code, portal_inputs=true)` for
    /// the startblock state of portal graphs; every state derived from
    /// that entry state via `copy()` or `union()` preserves the same
    /// pair.
    fn with_portal_extras(
        mut self,
        extras: (super::flow::FlowValue, super::flow::FlowValue),
    ) -> Self {
        self.portal_extras = Some(extras);
        self
    }

    fn mergeable(&self) -> Vec<Option<super::flow::FlowValue>> {
        let mut data = self.locals_w.clone();
        data.extend(self.stack.iter().cloned().map(Some));
        if let Some((w_type, w_value)) = &self.last_exception {
            data.push(Some(w_type.clone()));
            data.push(Some(w_value.clone()));
        } else {
            data.push(Some(super::flow::Constant::none().into()));
            data.push(Some(super::flow::Constant::none().into()));
        }
        if let Some((frame, ec)) = &self.portal_extras {
            data.push(Some(frame.clone()));
            data.push(Some(ec.clone()));
        }
        data
    }

    /// Step 6A slice S1 infrastructure: return the `mergeable()` position
    /// at which a given Variable appears, or `None` if it is not present.
    ///
    /// `framestate.py:38-43` `mergeable` concatenates `locals_w + stack +
    /// last_exc pair`; the i-th position is a stable per-FrameState slot
    /// identity that `Link.args` / `target.inputargs` correspondence is
    /// built on (see `getoutputargs` above — `link.args[j]` and
    /// `target.inputargs[j]` are both the j-th entry of their respective
    /// mergeable lists filtered for Variables).  Subsequent slices (S2)
    /// translate this mergeable index to the concrete SSARepr register
    /// slot by folding in `nlocals` / `ncells` / `stack_base`.  S3 uses
    /// the pair (mergeable index of `link.args[j]` in source state,
    /// mergeable index of `target.inputargs[j]` in target state) to
    /// drive `coalesce_by_links()`, the CFG-level replacement for pyre's
    /// current SSARepr `*_copy` scanner (`regalloc.rs::coalesce_variables`).
    ///
    /// Match identity is by `VariableId` (Python object identity in
    /// RPython); constants and other FlowValue shapes are ignored.
    fn mergeable_index_of(&self, var: &super::flow::Variable) -> Option<usize> {
        self.mergeable().iter().position(
            |value| matches!(value, Some(super::flow::FlowValue::Variable(v)) if v.id == var.id),
        )
    }

    /// Step 6A slice S2 infrastructure: translate a `mergeable()` index
    /// (S1) into the SSARepr register slot that the walker emits for
    /// that FrameState position.
    ///
    /// Pyre's register layout packs fast locals and the operand stack
    /// contiguously as `[locals 0..nlocals][stack nlocals..nlocals+
    /// max_stackdepth]` (see `RegisterLayout::compute`: `stack_base =
    /// nlocals as u16`).  `FrameState.locals_w.len() == nlocals` and
    /// `FrameState.stack` is indexed from `0` at the bottom of the
    /// operand stack, so `mergeable[0..locals_w.len() + stack.len())`
    /// maps identity to the register slot.
    ///
    /// The final two `mergeable()` entries carry the `last_exception`
    /// pair (`framestate.py:23` `last_exception`) — these come from
    /// exception-edge wiring (`rpython/flowspace/flowcontext.py:1259`)
    /// rather than a regular FrameState slot, so they have no register
    /// and the function returns `None`.
    ///
    /// Cell / free variables (`pyframe::ncells`) live in the absolute
    /// virtualizable array between locals and stack, but pyre's
    /// register layout does NOT reserve register slots for them — see
    /// `RegisterLayout::stack_base_absolute = nlocals + ncells` (the
    /// runtime offset) vs `stack_base = nlocals` (the register-space
    /// offset).  Consumers that need the absolute PyFrame slot for a
    /// virtualizable access compute it separately.
    fn mergeable_index_to_slot(&self, merge_idx: usize) -> Option<u16> {
        let regular_len = self.locals_w.len() + self.stack.len();
        if merge_idx < regular_len {
            Some(merge_idx as u16)
        } else {
            None
        }
    }

    /// Convenience composition of S1 + S2: resolve a Variable to its
    /// SSARepr register slot in one call.  Returns `None` if the
    /// Variable does not appear in this FrameState or appears only in
    /// the `last_exception` pair.
    fn variable_slot(&self, var: &super::flow::Variable) -> Option<u16> {
        self.mergeable_index_of(var)
            .and_then(|idx| self.mergeable_index_to_slot(idx))
    }

    fn copy<F>(&self, fresh_variable: &mut F) -> Self
    where
        F: FnMut(Option<Kind>) -> super::flow::Variable,
    {
        Self {
            locals_w: self
                .locals_w
                .iter()
                .map(|value| copy_optional_flow_value(value.as_ref(), fresh_variable))
                .collect(),
            stack: self
                .stack
                .iter()
                .map(|value| copy_flow_value(value, fresh_variable))
                .collect(),
            last_exception: self.last_exception.as_ref().map(|(w_type, w_value)| {
                (
                    copy_flow_value(w_type, fresh_variable),
                    copy_flow_value(w_value, fresh_variable),
                )
            }),
            blocklist: self.blocklist.clone(),
            next_offset: self.next_offset,
            // Portal extras are graph-level identity — same Variables
            // across every FrameState in the graph.  Do not freshen.
            portal_extras: self.portal_extras.clone(),
        }
    }

    fn getvariables(&self) -> Vec<super::flow::FlowValue> {
        self.mergeable()
            .into_iter()
            .flatten()
            .filter(|value| matches!(value, super::flow::FlowValue::Variable(_)))
            .collect()
    }

    fn matches(&self, other: &Self) -> bool {
        assert_eq!(self.blocklist, other.blocklist);
        assert_eq!(self.next_offset, other.next_offset);
        let mergeable = self.mergeable();
        let other_mergeable = other.mergeable();
        if mergeable.len() != other_mergeable.len() {
            return false;
        }
        for (left, right) in mergeable.iter().zip(other_mergeable.iter()) {
            if left == right {
                continue;
            }
            if matches!(
                (left, right),
                (
                    Some(super::flow::FlowValue::Variable(_)),
                    Some(super::flow::FlowValue::Variable(_))
                )
            ) {
                continue;
            }
            return false;
        }
        true
    }

    fn union<F>(&self, other: &Self, fresh_variable: &mut F) -> Option<Self>
    where
        F: FnMut(Option<Kind>) -> super::flow::Variable,
    {
        if self.next_offset != other.next_offset
            || self.locals_w.len() != other.locals_w.len()
            || self.stack.len() != other.stack.len()
        {
            return None;
        }

        let locals_w = self
            .locals_w
            .iter()
            .zip(other.locals_w.iter())
            .map(|(left, right)| union_optional_flow_value(left, right, fresh_variable))
            .collect();
        let stack = self
            .stack
            .iter()
            .zip(other.stack.iter())
            .map(|(left, right)| union_flow_value(left, right, fresh_variable))
            .collect::<Option<Vec<_>>>()?;
        let last_exception = match (&self.last_exception, &other.last_exception) {
            (None, None) => None,
            (Some((left_type, left_value)), Some((right_type, right_value))) => Some((
                union_flow_value(left_type, right_type, fresh_variable)?,
                union_flow_value(left_value, right_value, fresh_variable)?,
            )),
            (Some((left_type, left_value)), None) => Some((
                union_flow_value(
                    left_type,
                    &super::flow::Constant::none().into(),
                    fresh_variable,
                )?,
                union_flow_value(
                    left_value,
                    &super::flow::Constant::none().into(),
                    fresh_variable,
                )?,
            )),
            (None, Some((right_type, right_value))) => Some((
                union_flow_value(
                    &super::flow::Constant::none().into(),
                    right_type,
                    fresh_variable,
                )?,
                union_flow_value(
                    &super::flow::Constant::none().into(),
                    right_value,
                    fresh_variable,
                )?,
            )),
        };
        // Portal extras carry graph-level identity; if the two sides
        // are both portal-seeded they must reference the same Variables,
        // otherwise the graph is malformed.  Non-portal graphs never
        // populate them.
        let portal_extras = match (&self.portal_extras, &other.portal_extras) {
            (None, None) => None,
            (Some(left), Some(right)) => {
                if left == right {
                    Some(left.clone())
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        let mut merged = Self::new(
            locals_w,
            stack,
            last_exception,
            self.blocklist.clone(),
            self.next_offset,
        );
        merged.portal_extras = portal_extras;
        Some(merged)
    }

    fn getoutputargs(&self, targetstate: &Self) -> Vec<super::flow::FlowValue> {
        self.getoutputargs_with_positions(targetstate).0
    }

    fn getoutputargs_with_positions(
        &self,
        targetstate: &Self,
    ) -> (
        Vec<super::flow::FlowValue>,
        Vec<super::flow::LinkArgPosition>,
    ) {
        let mergeable = self.mergeable();
        let mut result = Vec::new();
        let mut positions = Vec::new();
        for (index, target_value) in targetstate.mergeable().iter().enumerate() {
            if matches!(target_value, Some(super::flow::FlowValue::Variable(_))) {
                result.push(
                    mergeable[index]
                        .clone()
                        .expect("target variable must correspond to a mergeable source value"),
                );
                positions.push(super::flow::LinkArgPosition {
                    source_mergeable_index: Some(index),
                    target_mergeable_index: Some(index),
                });
            }
        }
        (result, positions)
    }
}

fn copy_optional_flow_value<F>(
    value: Option<&super::flow::FlowValue>,
    fresh_variable: &mut F,
) -> Option<super::flow::FlowValue>
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    value.map(|value| copy_flow_value(value, fresh_variable))
}

fn copy_flow_value<F>(
    value: &super::flow::FlowValue,
    fresh_variable: &mut F,
) -> super::flow::FlowValue
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    match value {
        super::flow::FlowValue::Variable(variable) => fresh_variable(variable.kind).into(),
        super::flow::FlowValue::Constant(constant) => constant.clone().into(),
    }
}

fn union_optional_flow_value<F>(
    left: &Option<super::flow::FlowValue>,
    right: &Option<super::flow::FlowValue>,
    fresh_variable: &mut F,
) -> Option<super::flow::FlowValue>
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    match (left, right) {
        (Some(left), Some(right)) => union_flow_value(left, right, fresh_variable),
        (None, _) | (_, None) => None,
    }
}

fn union_flow_value<F>(
    left: &super::flow::FlowValue,
    right: &super::flow::FlowValue,
    fresh_variable: &mut F,
) -> Option<super::flow::FlowValue>
where
    F: FnMut(Option<Kind>) -> super::flow::Variable,
{
    if left == right {
        return Some(left.clone());
    }
    match (left, right) {
        (super::flow::FlowValue::Variable(left), super::flow::FlowValue::Variable(right)) => {
            Some(fresh_variable(union_kind(left.kind, right.kind)).into())
        }
        (
            super::flow::FlowValue::Variable(variable),
            super::flow::FlowValue::Constant(constant),
        )
        | (
            super::flow::FlowValue::Constant(constant),
            super::flow::FlowValue::Variable(variable),
        ) => Some(fresh_variable(union_kind(variable.kind, constant.kind)).into()),
        (super::flow::FlowValue::Constant(left), super::flow::FlowValue::Constant(right)) => {
            Some(fresh_variable(union_kind(left.kind, right.kind)).into())
        }
    }
}

fn union_kind(left: Option<Kind>, right: Option<Kind>) -> Option<Kind> {
    if left == right { left } else { None }
}

fn entry_frame_state(code: &CodeObject, portal_inputs: bool) -> FrameState {
    let inputargs = entry_inputargs(code);
    let mut locals_w = vec![None; code.varnames.len()];
    for (index, value) in inputargs.into_iter().enumerate() {
        if index < locals_w.len() {
            locals_w[index] = Some(value);
        }
    }
    let state = FrameState::new(
        locals_w,
        Vec::new(),
        None,
        frame_blocks_for_offset(code, 0),
        0,
    );
    if portal_inputs {
        let (frame, ec) = portal_graph_inputvars(code);
        state.with_portal_extras((frame.into(), ec.into()))
    } else {
        state
    }
}

#[derive(Debug)]
struct SpamBlock {
    /// `flowcontext.py:40` underlying `Block`.
    block: super::flow::BlockRef,
    /// `flowcontext.py:40` `block.framestate`.
    framestate: Option<FrameState>,
    /// `flowcontext.py:41` `block.dead`.
    dead: bool,
}

#[derive(Debug, Clone)]
struct SpamBlockRef(Rc<RefCell<SpamBlock>>);

impl SpamBlockRef {
    fn new(block: super::flow::BlockRef, framestate: Option<FrameState>) -> Self {
        Self(Rc::new(RefCell::new(SpamBlock {
            block,
            framestate,
            dead: false,
        })))
    }

    fn block(&self) -> super::flow::BlockRef {
        self.0.borrow().block.clone()
    }

    fn framestate(&self) -> Option<FrameState> {
        self.0.borrow().framestate.clone()
    }

    fn set_framestate(&self, framestate: FrameState) {
        self.0.borrow_mut().framestate = Some(framestate);
    }

    fn mark_dead(&self) {
        self.0.borrow_mut().dead = true;
    }

    fn dead(&self) -> bool {
        self.0.borrow().dead
    }
}

impl PartialEq for SpamBlockRef {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SpamBlockRef {}

fn fresh_variable_for_state(
    graph: &mut super::flow::FunctionGraph,
    kind: Option<Kind>,
) -> super::flow::Variable {
    match kind {
        Some(kind) => graph.fresh_variable(kind),
        None => graph.fresh_untyped_variable(),
    }
}

fn append_exit(block: &super::flow::BlockRef, link: super::flow::LinkRef) {
    link.borrow_mut().prevblock = Some(block.downgrade());
    block.borrow_mut().exits.push(link);
}

/// Step 6A slice S4a: atomically append `link` to `block.exits` and
/// snapshot `source_state` into `link_exit_states` so later passes
/// (`collect_link_slot_pairs`) can resolve the source-side register
/// slots at this link.
///
/// RPython parity: there is no direct counterpart — RPython's
/// `coalesce_variables` runs inline with Variable-keyed UnionFind
/// over `graph.iterblocks()`, so no per-link state capture is
/// needed.  pyre's regalloc runs after-the-fact on a u16-indexed
/// SSARepr (`regalloc.rs` docstring, lines 26-36 PRE-EXISTING-
/// ADAPTATION), so the collector needs the source FrameState to
/// translate Variables back to slots.  The snapshot is the minimal
/// bridging data — one FrameState per link, cloned at emission time
/// (the walker discards its `currentstate` after the terminator
/// finishes so a clone is the only way to preserve it).
fn append_exit_with_state(
    block: &super::flow::BlockRef,
    link: super::flow::LinkRef,
    source_state: &FrameState,
    link_exit_states: &mut HashMap<super::flow::LinkRef, FrameState>,
) {
    link_exit_states.insert(link.clone(), source_state.clone());
    append_exit(block, link);
}

fn output_link(
    source_state: &FrameState,
    target_state: &FrameState,
    target: super::flow::BlockRef,
) -> super::flow::LinkRef {
    let (outputargs, arg_positions) = source_state.getoutputargs_with_positions(target_state);
    super::flow::Link::new(outputargs, Some(target), None)
        .with_arg_positions(arg_positions)
        .into_ref()
}

/// Build the `[w_type, w_value]` argument list for a Link targeting
/// `graph.exceptblock`.  Mirrors `flatten.py:161-162` —
/// `assert link.last_exception is not None; assert link.last_exc_value
/// is not None`.  Callers must have seeded `source_state.last_exception`
/// before emitting the link.
fn exceptblock_link_args(source_state: &FrameState) -> Vec<super::flow::FlowValue> {
    match &source_state.last_exception {
        Some((w_type, w_value)) => vec![w_type.clone(), w_value.clone()],
        None => panic!(
            "exceptblock edge requires materialized exception pair \
             (flatten.py:161-162 make_exception_link parity)"
        ),
    }
}

/// Allocate the fresh `(exc_type, exc_value)` Variable pair that
/// represents an exception edge's payload at the graph level.
///
/// PRE-EXISTING-ADAPTATION vs `rpython/flowspace/flowcontext.py:1250-
/// 1261 Raise.nomoreblocks`: RPython's flow analysis sees the Python
/// source form `raise SomeError("msg")` and builds an
/// `OperationException(w_type=Constant(SomeError), w_value=...)` from
/// which `Raise.nomoreblocks` projects `[w_exc.w_type, w_exc.w_value]`
/// as real trace-level values into the exception Link.  Pyre's tracer
/// is one level lower: the stack carries a SINGLE Ref value
/// (`obj_tmp0`, the exception instance), and the exception type is
/// extracted at runtime inside the `raise` opcode's backend
/// implementation (`ssa_emitter.rs emit_raise` + blackhole handler).
/// There is no graph-level Variable that stands for "the type of the
/// raised value" because pyre's graph emission is driven by bytecode,
/// not by `raise`-statement source.  Synthesizing fresh Variables here
/// matches `flowcontext.py:133-143 guessexception` — the same
/// mechanism upstream itself uses on implicit exception edges, where
/// type/value are also not statically knowable.
///
/// The fresh pair is carried on the Link as BOTH `link.args` AND
/// `link.extravars` (see `exception_edge_extravars`), so the upstream
/// `flatten.py:163-164 make_exception_link` check `link.args ==
/// [link.last_exception, link.last_exc_value]` matches and the
/// pass-through `raise` / `reraise` emission path fires.  The payload
/// is structurally synthetic at the graph layer and becomes concrete
/// only when the backend `raise`/`reraise` opcode populates the
/// JitFrame's exception slots from `obj_tmp0` at runtime.
fn exception_edge_vars(
    graph: &mut super::flow::FunctionGraph,
) -> (super::flow::Variable, super::flow::Variable) {
    (
        graph.fresh_untyped_variable(),
        graph.fresh_untyped_variable(),
    )
}

fn exception_landing_state(
    graph: &mut super::flow::FunctionGraph,
    source_state: &FrameState,
) -> FrameState {
    let (exc_type, exc_value) = exception_edge_vars(graph);
    let mut state = source_state.clone();
    state.last_exception = Some((exc_type.into(), exc_value.into()));
    state
}

/// `flowcontext.py:635-636` computes `w_type = op.type(w_value).eval(self)`
/// before `Raise.nomoreblocks` projects the explicit raise edge to
/// `[w_exc.w_type, w_exc.w_value]`.
///
/// pyre's production bytecode still emits a single `raise/r` opcode and
/// derives the exception type at runtime, but the shadow graph can still
/// mirror the upstream shape exactly: record a graph-level `type`
/// operation whose result becomes `link.args[0]`, and carry the actual
/// raised value as `link.args[1]`.
fn explicit_raise_exception_pair(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    raised_value: super::flow::FlowValue,
    offset: i64,
) -> (super::flow::FlowValue, super::flow::FlowValue) {
    let exc_type = graph.fresh_variable(Kind::Ref);
    record_graph_op(
        block,
        "type",
        vec![raised_value.clone().into()],
        Some(exc_type.into()),
        offset,
    );
    (exc_type.into(), raised_value)
}

fn explicit_raise_state(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    source_state: &FrameState,
    raised_value: super::flow::FlowValue,
    offset: i64,
) -> FrameState {
    let mut state = source_state.clone();
    state.last_exception = Some(explicit_raise_exception_pair(
        graph,
        block,
        raised_value,
        offset,
    ));
    state
}

/// Extract the `(etype, evalue)` Variable pair from the edge state
/// produced by `explicit_raise_state` / `exception_landing_state`.
/// Mirrors the pattern used by `flowcontext.py:141-143` where the
/// freshly-created `last_exc` / `last_exc_value` Variables are both
/// placed into `link.args` AND attached to `link.extravars(...)`
/// (`model.py:127-129 Link.extravars`) — so downstream passes that
/// check `link.args == [link.last_exception, link.last_exc_value]`
/// (`flatten.py:163-164 make_exception_link`) can identify the edge
/// as a pass-through of the exception pair.
fn exception_edge_extravars(
    edge_state: &FrameState,
) -> (super::flow::Variable, super::flow::Variable) {
    let (w_type, w_value) = edge_state
        .last_exception
        .as_ref()
        .expect("exception edge state missing last_exception pair");
    let as_variable = |value: &super::flow::FlowValue| match value {
        super::flow::FlowValue::Variable(v) => *v,
        super::flow::FlowValue::Constant(_) => panic!(
            "exception edge last_exception carries Constant; extravars \
             expects Variables (flowcontext.py:130-134 guessexception)"
        ),
    };
    (as_variable(w_type), as_variable(w_value))
}

fn update_catch_landing_state(
    graph: &mut super::flow::FunctionGraph,
    target: &SpamBlockRef,
    source_state: &FrameState,
) {
    let candidate = exception_landing_state(graph, source_state);
    if let Some(existing) = target.framestate() {
        let mut fresh = |kind| fresh_variable_for_state(graph, kind);
        if let Some(merged) = existing.union(&candidate, &mut fresh) {
            target.set_framestate(merged);
        }
    } else {
        target.set_framestate(candidate);
    }
}

fn handler_entry_state_from_catch_site(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    landing_state: &FrameState,
    site: &ExceptionCatchSite,
) -> FrameState {
    let mut state = landing_state.clone();
    sync_stack_state(graph, &mut state, site.stack_depth);
    if site.push_lasti {
        state.stack.push(fresh_ref_value(graph));
    }
    let exc_value = state
        .last_exception
        .as_ref()
        .map(|(_w_type, w_value)| w_value.clone())
        .unwrap_or_else(|| fresh_ref_value(graph));
    state.stack.push(exc_value);
    state.next_offset = site.handler_py_pc;
    state.blocklist = frame_blocks_for_offset(code, site.handler_py_pc);
    state
}

fn handler_entry_state_from_catch_sites(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    catch_sites: &[ExceptionCatchSite],
    catch_landing_blocks: &HashMap<u16, SpamBlockRef>,
    handler_py_pc: usize,
) -> Option<FrameState> {
    let mut merged: Option<FrameState> = None;
    for site in catch_sites {
        if site.handler_py_pc != handler_py_pc {
            continue;
        }
        let landing_state = catch_landing_blocks
            .get(&site.landing_label)
            .and_then(|block| block.framestate())?;
        let candidate = handler_entry_state_from_catch_site(code, graph, &landing_state, site);
        merged = Some(match merged {
            None => candidate,
            Some(existing) => {
                let mut fresh = |kind| fresh_variable_for_state(graph, kind);
                existing.union(&candidate, &mut fresh).unwrap_or(candidate)
            }
        });
    }
    merged
}

fn initialize_spam_block(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    target: &SpamBlockRef,
    source_state: &FrameState,
    next_offset: usize,
) -> FrameState {
    if let Some(state) = target.framestate() {
        return state;
    }

    let mut fresh = |kind| fresh_variable_for_state(graph, kind);
    let mut target_state = source_state.copy(&mut fresh);
    target_state.blocklist = frame_blocks_for_offset(code, next_offset);
    target_state.next_offset = next_offset;
    target.block().borrow_mut().inputargs = target_state.getvariables();
    target.set_framestate(target_state.clone());
    target_state
}

fn make_next_block(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    currentblock: &SpamBlockRef,
    currentstate: &FrameState,
    next_offset: usize,
    link_exit_states: &mut HashMap<super::flow::LinkRef, FrameState>,
) -> SpamBlockRef {
    let mut fresh = |kind| fresh_variable_for_state(graph, kind);
    let mut newstate = currentstate.copy(&mut fresh);
    newstate.blocklist = frame_blocks_for_offset(code, next_offset);
    newstate.next_offset = next_offset;
    let newblock = SpamBlockRef::new(graph.new_block(Vec::new()), Some(newstate.clone()));
    newblock.block().borrow_mut().inputargs = newstate.getvariables();
    append_exit_with_state(
        &currentblock.block(),
        output_link(currentstate, &newstate, newblock.block()),
        currentstate,
        link_exit_states,
    );
    newblock
}

fn mergeblock(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    joinpoints: &mut HashMap<usize, Vec<SpamBlockRef>>,
    pc_blocks: &mut [Option<SpamBlockRef>],
    currentblock: &SpamBlockRef,
    currentstate: &FrameState,
    next_offset: usize,
    link_exit_states: &mut HashMap<super::flow::LinkRef, FrameState>,
) -> SpamBlockRef {
    let candidates = joinpoints.entry(next_offset).or_default();
    for index in 0..candidates.len() {
        let block = candidates[index].clone();
        let block_state = block
            .framestate()
            .expect("joinpoint candidate must carry a FrameState");
        let mut fresh = |kind| fresh_variable_for_state(graph, kind);
        let Some(mut newstate) = block_state.union(currentstate, &mut fresh) else {
            continue;
        };
        if newstate.matches(&block_state) {
            append_exit_with_state(
                &currentblock.block(),
                output_link(currentstate, &newstate, block.block()),
                currentstate,
                link_exit_states,
            );
            pc_blocks[next_offset] = Some(block.clone());
            return block;
        }

        for (name, value) in code.varnames.iter().zip(newstate.locals_w.iter_mut()) {
            if let Some(super::flow::FlowValue::Variable(variable)) = value.as_mut() {
                variable.rename(name);
            }
        }
        let newblock = SpamBlockRef::new(graph.new_block(Vec::new()), Some(newstate.clone()));
        newblock.block().borrow_mut().inputargs = newstate.getvariables();
        append_exit_with_state(
            &currentblock.block(),
            output_link(currentstate, &newstate, newblock.block()),
            currentstate,
            link_exit_states,
        );

        block.mark_dead();
        block.block().borrow_mut().operations.clear();
        block.block().borrow_mut().exitswitch = None;
        // Supersede link: the dead candidate block's EXIT state is
        // `block_state` (its stored FrameState — no mid-block ops
        // were ever walked through it, since it was a joinpoint
        // candidate not yet visited by the dispatch loop).  The new
        // merged block absorbs its outgoing edge; snapshot
        // `block_state` as the source EXIT so later passes see the
        // correct Variables.
        let supersede_link = output_link(&block_state, &newstate, newblock.block());
        link_exit_states.insert(supersede_link.clone(), block_state.clone());
        block.block().recloseblock(vec![supersede_link]);

        candidates.remove(index);
        candidates.insert(0, newblock.clone());
        pc_blocks[next_offset] = Some(newblock.clone());
        return newblock;
    }

    let newblock = make_next_block(
        code,
        graph,
        currentblock,
        currentstate,
        next_offset,
        link_exit_states,
    );
    candidates.insert(0, newblock.clone());
    pc_blocks[next_offset] = Some(newblock.clone());
    newblock
}

fn ensure_pc_block(
    code: &CodeObject,
    graph: &mut super::flow::FunctionGraph,
    joinpoints: &mut HashMap<usize, Vec<SpamBlockRef>>,
    pc_blocks: &mut [Option<SpamBlockRef>],
    current_state: &FrameState,
    py_pc: usize,
) -> SpamBlockRef {
    if let Some(block) = pc_blocks[py_pc].clone() {
        return block;
    }
    if let Some(block) = joinpoints
        .get(&py_pc)
        .and_then(|blocks| blocks.first())
        .cloned()
    {
        pc_blocks[py_pc] = Some(block.clone());
        return block;
    }

    let block = SpamBlockRef::new(graph.new_block(Vec::new()), None);
    initialize_spam_block(code, graph, &block, current_state, py_pc);
    joinpoints
        .entry(py_pc)
        .or_default()
        .insert(0, block.clone());
    pc_blocks[py_pc] = Some(block.clone());
    block
}

fn fresh_ref_value(graph: &mut super::flow::FunctionGraph) -> super::flow::FlowValue {
    graph.fresh_variable(Kind::Ref).into()
}

/// Step 6 transitional dual-write.  `rpython/jit/codewriter/codewriter.py:44-67`
/// runs `perform_register_allocation(graph) → flatten_graph(graph) →
/// compute_liveness(ssarepr) → assemble(ssarepr)`.  Upstream has **one**
/// IR stream — the flow graph — which `flatten_graph` lowers into an
/// `SSARepr`.
///
/// Pyre historically emitted `SSARepr` directly from the trace recorder
/// and skipped the flow-graph stage.  Step 6A reintroduces the graph
/// (so CFG-level `regalloc.py:79-96 coalesce_variables` can run), but the
/// SSARepr emission has not yet been replaced with a `flatten_graph` pass
/// (Task #214).  Until it is, each opcode handler must populate both
/// streams — the SSARepr byte stream that backend/blackhole consume, and
/// the graph that `FlowGraphRegAllocator` consumes.
///
/// Delete this helper once Task #214 lands and the SSARepr stream is
/// generated from the graph by `flatten_graph`.
fn record_graph_op(
    block: &super::flow::BlockRef,
    opname: impl Into<String>,
    args: Vec<super::flow::SpaceOperationArg>,
    result: Option<super::flow::FlowValue>,
    offset: i64,
) -> super::flow::SpaceOperation {
    let op = super::flow::SpaceOperation::new(opname, args, result, offset);
    super::flow::push_op(block, op.clone());
    op
}

fn sync_stack_state(graph: &mut super::flow::FunctionGraph, state: &mut FrameState, depth: u16) {
    while state.stack.len() > depth as usize {
        state.stack.pop();
    }
    while state.stack.len() < depth as usize {
        state.stack.push(fresh_ref_value(graph));
    }
}

fn new_shadow_graph_with_portal_inputs(
    code: &CodeObject,
    portal_inputs: bool,
) -> super::flow::FunctionGraph {
    let start_inputargs = graph_entry_inputargs(code, portal_inputs);
    let return_var = Some(super::flow::Variable::new(
        super::flow::VariableId(start_inputargs.len() as u32),
        Kind::Ref,
    ));
    super::flow::FunctionGraph::new(
        code.obj_name.to_string(),
        super::flow::Block::shared(start_inputargs),
        return_var,
    )
}

fn new_shadow_graph(code: &CodeObject) -> super::flow::FunctionGraph {
    new_shadow_graph_with_portal_inputs(code, false)
}

fn attach_catch_exception_edge(
    graph: &mut super::flow::FunctionGraph,
    block: &super::flow::BlockRef,
    target: &SpamBlockRef,
    source_state: &FrameState,
    link_exit_states: &mut HashMap<super::flow::LinkRef, FrameState>,
) -> super::flow::LinkRef {
    let mut link = super::flow::Link::new(Vec::new(), Some(target.block()), None);
    let mut block_mut = block.borrow_mut();
    block_mut.exitswitch = Some(super::flow::ExitSwitch::Value(
        super::flow::c_last_exception().into(),
    ));
    drop(block_mut);
    let (exc_type, exc_value) = exception_edge_vars(graph);
    link.extravars(Some(exc_type), Some(exc_value));
    update_catch_landing_state(graph, target, source_state);
    let link = link.into_ref();
    append_exit_with_state(block, link.clone(), source_state, link_exit_states);
    link
}

/// Step 6A slice S3b: collect `BlockRef → FrameState` entries from the
/// walker's in-flight block catalogues.  Pure function, no side effects.
///
/// The walker maintains three `SpamBlockRef` containers:
///   - `pc_blocks[py_pc]`           — joinpoint / current block per Python PC.
///   - `joinpoints[py_pc]`          — merged / superseded candidates.
///   - `catch_landing_blocks[label]` — pre-allocated catch-landing entries.
///
/// Catch-landing `SpamBlockRef`s are constructed with `framestate =
/// None` (`SpamBlockRef::new(..., None)`), so they are naturally
/// skipped here.  Same for `FunctionGraph::returnblock` /
/// `exceptblock` — those are canonical blocks that never flow through
/// a `SpamBlockRef`.
///
/// Later entries overwrite earlier ones when the same `BlockRef` is
/// seen more than once (e.g. a joinpoint that is also the current
/// `pc_blocks[pc]`).  Because `mergeblock` discards a dead block and
/// installs its successor in `pc_blocks`, the last-write-wins policy
/// yields the freshest live `FrameState` per block.
///
/// Consumer: S4 will feed this map plus the graph into
/// `collect_link_slot_pairs` to produce per-link coalesce pairs in
/// the production walker path.
fn collect_block_states(
    pc_blocks: &[Option<SpamBlockRef>],
    joinpoints: &HashMap<usize, Vec<SpamBlockRef>>,
    catch_landing_blocks: &HashMap<u16, SpamBlockRef>,
) -> HashMap<super::flow::BlockRef, FrameState> {
    let mut map = HashMap::new();
    let mut absorb = |entry: &SpamBlockRef| {
        if let Some(state) = entry.framestate() {
            map.insert(entry.block(), state);
        }
    };
    for entry in pc_blocks.iter().flatten() {
        absorb(entry);
    }
    for candidates in joinpoints.values() {
        for entry in candidates {
            absorb(entry);
        }
    }
    for entry in catch_landing_blocks.values() {
        absorb(entry);
    }
    map
}

/// Step 6A slice S3 (S3c revision): CFG-level collection of
/// `(source_slot, target_slot)` coalesce pairs.  Pure function, no
/// side effects.
///
/// Walks `graph.iterblocks()` → each block's exits.  For each Link:
///   1. Source state = `link_exit_states[link]` — the walker's
///      `currentstate` snapshot captured at terminator emission time
///      (`flowcontext.py:1237,1268-1280`).  This is the source
///      block's EXIT state, not its ENTRY state, because fresh
///      Variables produced by mid-block operations live in
///      `currentstate.locals_w` / `currentstate.stack` but never in
///      the source block's stored ENTRY FrameState.
///   2. Target state = `block_entry_states[link.target]` — the target
///      block's ENTRY FrameState set up by `mergeblock` /
///      `initialize_spam_block` (its mergeable positions correspond
///      directly to `target.inputargs`).
///   3. Links with no source EXIT entry or no target ENTRY entry
///      (catch landings, `returnblock`, `exceptblock`) contribute no
///      pairs.
///   4. For each `link.args[j]` with preserved
///      `Link.arg_positions[j]`:
///        - `source_mergeable_index` comes from the source
///          `FrameState.getoutputargs()` walk at edge-construction
///          time; `target_mergeable_index` records the target-side
///          mergeable entry that produced `target.inputargs[j]`.
///        - Skip non-Variable source args, matching
///          `regalloc.py:99-101` `if isinstance(v, Variable)`.
///        - Resolve source / target slots independently with
///          `FrameState::mergeable_index_to_slot(...)`; either side
///          may return `None` for the `last_exception` pair.
///        - Push `(source_slot, target_slot)`.  For ordinary
///          jump/merge edges these are usually equal because
///          `framestate.py:getoutputargs()` uses the same mergeable
///          index on both sides, but pyre now reads the recorded
///          per-link positions instead of re-deriving them from whole
///          FrameState scans.
///
/// Upstream reference: `rpython/tool/algo/regalloc.py:79-96`
/// `RegAllocator.coalesce_variables` iterates `graph.iterblocks()` →
/// `block.exits` → `zip(link.args, link.target.inputargs)` and unions
/// each Variable pair via `_try_coalesce`.  RPython has no FrameState
/// indirection — Variables carry their own UnionFind identity.
/// pyre's regalloc is u16-register-keyed (PRE-EXISTING-ADAPTATION;
/// see `regalloc.rs:26-36`), so this helper projects Variables back
/// onto slots through the per-link mergeable positions preserved when
/// the Link was created.
///
/// Why positional, not Variable-keyed: pyre's walker can reuse one
/// Variable across multiple mergeable positions simultaneously — e.g.
/// `LoadFast` at `codewriter.rs:2413-2414` pushes the local's own
/// Variable onto the stack, so that Variable lives at slot `x` (in
/// `locals_w`) AND at slot `stack_base + depth` (in `stack`) in the
/// same FrameState.  A Variable → single slot map would be ambiguous;
/// the per-link mergeable indices preserved from
/// `FrameState::getoutputargs_with_positions` keep the exact source /
/// target positions.
fn collect_link_slot_pairs(
    graph: &super::flow::FunctionGraph,
    block_entry_states: &HashMap<super::flow::BlockRef, FrameState>,
    link_exit_states: &HashMap<super::flow::LinkRef, FrameState>,
) -> Vec<(u16, u16)> {
    let mut pairs = Vec::new();
    for block in graph.iterblocks() {
        let block_borrow = block.borrow();
        for link in &block_borrow.exits {
            let Some(source_state) = link_exit_states.get(link) else {
                continue;
            };
            let link_borrow = link.borrow();
            let Some(target) = link_borrow.target.clone() else {
                continue;
            };
            let Some(target_state) = block_entry_states.get(&target) else {
                continue;
            };
            for (arg, positions) in link_borrow
                .args
                .iter()
                .zip(link_borrow.arg_positions.iter())
            {
                let Some(super::flow::FlowValue::Variable(_)) = arg.as_ref() else {
                    continue;
                };
                let Some(source_idx) = positions.source_mergeable_index else {
                    continue;
                };
                let Some(target_idx) = positions.target_mergeable_index else {
                    continue;
                };
                let Some(source_slot) = source_state.mergeable_index_to_slot(source_idx) else {
                    continue;
                };
                let Some(target_slot) = target_state.mergeable_index_to_slot(target_idx) else {
                    continue;
                };
                pairs.push((source_slot, target_slot));
            }
        }
    }
    pairs
}

// `PyJitCode` and `PyJitCodeMetadata` live in `pyre_jit_trace::pyjitcode`
// so both the codewriter (here) and the trace/blackhole runtime can hold
// the same `Arc<PyJitCode>` instances.

#[derive(Clone, Copy)]
struct ExceptionCatchSite {
    landing_label: u16,
    handler_py_pc: usize,
    stack_depth: u16,
    push_lasti: bool,
    lasti_py_pc: usize,
}

/// RPython: per-graph output of `perform_register_allocation` over the
/// three register kinds (codewriter.py:46-48). pyre's regalloc is
/// trivial — fast locals occupy the bottom of the ref register file
/// and the value stack stacks above them — so the "allocation" reduces
/// to a handful of constant offsets derived from `code.varnames` /
/// `code.max_stackdepth`. `RegisterLayout::compute` runs the same
/// arithmetic the inline section of `transform_graph_to_jitcode` used
/// to do directly; its only purpose is to give the layout a name and
/// pull the calculation out of the 1400-line dispatch loop.
#[derive(Clone, Copy, Debug)]
struct RegisterLayout {
    /// `code.varnames.len()` — number of fast locals.
    nlocals: usize,
    /// pyre-only: number of cell + free vars (`pyframe::ncells`).
    ncells: usize,
    /// Absolute index where the operand stack begins in
    /// `PyFrame.locals_cells_stack_w` — `nlocals + ncells`.
    stack_base_absolute: usize,
    /// Compile-time depth bound from `code.max_stackdepth` (clamped to ≥ 1).
    max_stackdepth: usize,
    /// Ref register index where the operand stack begins
    /// (`stack_base = nlocals` since locals occupy the first registers).
    stack_base: u16,
    /// Scratch ref register #0 — sits at `nlocals + max_stackdepth`.
    obj_tmp0: u16,
    /// Scratch ref register #1.
    obj_tmp1: u16,
    /// First ref register reserved for portal red-arg shuffling.
    arg_regs_start: u16,
    /// Ref register holding the `NULL` sentinel value.
    null_ref_reg: u16,
    /// `interp_jit.py:64` portal red `frame` register.
    portal_frame_reg: u16,
    /// `interp_jit.py:64` portal red `ec` register.
    portal_ec_reg: u16,
    /// Scratch int register #0.
    int_tmp0: u16,
    /// Scratch int register #1.
    int_tmp1: u16,
    /// Int register holding the current opcode value during dispatch.
    op_code_reg: u16,
}

impl RegisterLayout {
    /// Pure arithmetic over `code` — no allocation, no side effects.
    /// Mirrors the constant block at the top of
    /// `transform_graph_to_jitcode`.
    fn compute(code: &CodeObject) -> Self {
        let nlocals = code.varnames.len();
        let ncells = pyre_interpreter::pyframe::ncells(code);
        let stack_base_absolute = nlocals + ncells;
        let max_stackdepth = code.max_stackdepth.max(1) as usize;
        let stack_base = nlocals as u16;
        let obj_tmp0 = (nlocals + max_stackdepth) as u16;
        let obj_tmp1 = (nlocals + max_stackdepth + 1) as u16;
        let arg_regs_start = (nlocals + max_stackdepth + 2) as u16;
        let null_ref_reg = (nlocals + max_stackdepth + 10) as u16;
        let portal_frame_reg = null_ref_reg + 1;
        let portal_ec_reg = null_ref_reg + 2;
        Self {
            nlocals,
            ncells,
            stack_base_absolute,
            max_stackdepth,
            stack_base,
            obj_tmp0,
            obj_tmp1,
            arg_regs_start,
            null_ref_reg,
            portal_frame_reg,
            portal_ec_reg,
            int_tmp0: 0,
            int_tmp1: 1,
            op_code_reg: 2,
        }
    }
}

/// Indices returned by `assembler.add_fn_ptr` for every blackhole
/// helper fn pointer the dispatch loop references. Mirrors the slot
/// shape of RPython's `_callinfo_for_oopspec`-derived index table —
/// the helpers are interned in a fixed order so the dispatch handlers
/// can capture the indices once and reuse them across emit sites.
///
/// PRE-EXISTING-ADAPTATION: the order matches the historical
/// inline sequence (`call_fn`, then the per-opcode helpers, then the
/// per-arity `call_fn_n`). Changing the order would shift every
/// `assembler.add_fn_ptr` index — RPython's `assembler.see_raw_object`
/// path has the same constraint.
#[derive(Clone, Copy, Debug)]
struct FnPtrIndices {
    call_fn: u16,
    load_global_fn: u16,
    compare_fn: u16,
    binary_op_fn: u16,
    box_int_fn: u16,
    truth_fn: u16,
    load_const_fn: u16,
    store_subscr_fn: u16,
    build_list_fn: u16,
    normalize_raise_varargs_fn: u16,
    call_fn_0: u16,
    call_fn_2: u16,
    call_fn_3: u16,
    call_fn_4: u16,
    call_fn_5: u16,
    call_fn_6: u16,
    call_fn_7: u16,
    call_fn_8: u16,
    get_current_exception_fn: u16,
    set_current_exception_fn: u16,
}

/// Register every blackhole helper fn pointer with the assembler in
/// the canonical order. Returns the per-helper index table used by
/// the dispatch loop.
fn register_helper_fn_pointers(
    assembler: &mut SSAReprEmitter,
    cpu: &super::cpu::Cpu,
) -> FnPtrIndices {
    // RPython: CallControl manages fn addresses; assembler.finished()
    // writes them into callinfocollection. pyre adds them inline so
    // each handler can capture the index it needs.
    let call_fn = assembler.add_fn_ptr(cpu.call_fn as *const ());
    let load_global_fn = assembler.add_fn_ptr(cpu.load_global_fn as *const ());
    let compare_fn = assembler.add_fn_ptr(cpu.compare_fn as *const ());
    let binary_op_fn = assembler.add_fn_ptr(cpu.binary_op_fn as *const ());
    let box_int_fn = assembler.add_fn_ptr(cpu.box_int_fn as *const ());
    let truth_fn = assembler.add_fn_ptr(cpu.truth_fn as *const ());
    let load_const_fn = assembler.add_fn_ptr(cpu.load_const_fn as *const ());
    let store_subscr_fn = assembler.add_fn_ptr(cpu.store_subscr_fn as *const ());
    let build_list_fn = assembler.add_fn_ptr(cpu.build_list_fn as *const ());
    let normalize_raise_varargs_fn =
        assembler.add_fn_ptr(cpu.normalize_raise_varargs_fn as *const ());
    // Per-arity call helpers (appended AFTER existing fn_ptrs to preserve indices).
    let call_fn_0 = assembler.add_fn_ptr(cpu.call_fn_0 as *const ());
    let call_fn_2 = assembler.add_fn_ptr(cpu.call_fn_2 as *const ());
    let call_fn_3 = assembler.add_fn_ptr(cpu.call_fn_3 as *const ());
    let call_fn_4 = assembler.add_fn_ptr(cpu.call_fn_4 as *const ());
    let call_fn_5 = assembler.add_fn_ptr(cpu.call_fn_5 as *const ());
    let call_fn_6 = assembler.add_fn_ptr(cpu.call_fn_6 as *const ());
    let call_fn_7 = assembler.add_fn_ptr(cpu.call_fn_7 as *const ());
    let call_fn_8 = assembler.add_fn_ptr(cpu.call_fn_8 as *const ());
    let get_current_exception_fn = assembler.add_fn_ptr(cpu.get_current_exception_fn as *const ());
    let set_current_exception_fn = assembler.add_fn_ptr(cpu.set_current_exception_fn as *const ());
    FnPtrIndices {
        call_fn,
        load_global_fn,
        compare_fn,
        binary_op_fn,
        box_int_fn,
        truth_fn,
        load_const_fn,
        store_subscr_fn,
        build_list_fn,
        normalize_raise_varargs_fn,
        call_fn_0,
        call_fn_2,
        call_fn_3,
        call_fn_4,
        call_fn_5,
        call_fn_6,
        call_fn_7,
        call_fn_8,
        get_current_exception_fn,
        set_current_exception_fn,
    }
}

/// RPython: `liveness.py:19-80` `compute_liveness(ssarepr)` —
/// backward dataflow over the populated `SSARepr` that fills each
/// `-live-` marker with the set of registers alive across it.
///
/// The dataflow runs on the post-regalloc `SSARepr` via the upstream
/// `liveness::compute_liveness`, including `remove_repeated_live`.
/// pyre's `PcAnchor(py_pc)` markers survive that rewrite unchanged, so
/// the follow-up filter rescans anchor-delimited ranges in the FINAL
/// `SSARepr` to find each Python PC's `-live-` marker instead of caching
/// pre-rewrite insn indices.
///
/// After the dataflow, pyre applies an in-place post-filter to each
/// `-live-` marker so the args carry only the subset of registers
/// the runtime contract is currently wired to consume (see rules
/// below). `get_liveness_info` at assemble time
/// (`assembler.rs:422-437`) partitions by kind and emits the
/// `all_liveness` byte stream. Both the tracer's
/// `get_list_of_active_boxes` and the blackhole's bridge-resume
/// `consume_one_section` read `all_liveness` via `LivenessIterator`,
/// so the single post-rename `-live-` marker is the sole source.
///
/// Post-filter rules (pyre-only adaptation on top of SSA output):
///   - Only Ref-kind registers: pyre Int/Float regs are scratch
///     tmps (int_tmp0/1, op_code_reg) or constant-as-reg encodings
///     inside `jit_merge_point`; they never carry a Python box
///     across py_pc boundaries. Leaving live_i / live_f empty keeps
///     the tracer / blackhole (which index box arrays by raw
///     register index at `trace_opcode.rs:229-263` /
///     `call_jit.rs:965-982`) from pulling a Ref box through an
///     Int/Float slot.
///   - Only indices inside the Python-frame range: locals
///     `0..nlocals` or in-depth stack slots
///     `stack_base..stack_base+depth`. Helper Ref regs above that
///     range (obj_tmp0/1, arg_regs_start, null_ref_reg,
///     portal_{frame,ec}_reg) are correctly dead across py_pcs and
///     would desynchronise box layout if leaked.
///   - Every in-depth stack slot `stack_base + d` for
///     `d in 0..depth_at_pc[py_pc]` is force-added even if SSA
///     didn't mark it alive, because pyre's runtime invariant at
///     every `-live-` marker is "all stack slots up to the current
///     depth hold a live box".
///
/// Unreachable PCs still get emptied in place via the bytecode
/// `LiveVars` analysis. The direct-dispatch walker emits one
/// `PcAnchor`/`-live-` pair per Python PC, including dead bytecodes
/// that never execute, whereas upstream RPython only flattens
/// reachable flow-graph blocks.
fn pc_anchor_positions(ssarepr: &super::flatten::SSARepr, num_pcs: usize) -> Vec<usize> {
    let mut positions = vec![usize::MAX; num_pcs];
    for (insn_idx, insn) in ssarepr.insns.iter().enumerate() {
        if let Insn::PcAnchor(py_pc) = insn {
            assert!(
                *py_pc < num_pcs,
                "pc_anchor_positions: py_pc {py_pc} out of range {num_pcs}"
            );
            assert_eq!(
                positions[*py_pc],
                usize::MAX,
                "pc_anchor_positions: duplicate PcAnchor for py_pc {py_pc}"
            );
            positions[*py_pc] = insn_idx;
        }
    }
    for (py_pc, &insn_idx) in positions.iter().enumerate() {
        assert_ne!(
            insn_idx,
            usize::MAX,
            "pc_anchor_positions: missing PcAnchor for py_pc {py_pc}"
        );
    }
    positions
}

fn live_marker_indices_by_pc(ssarepr: &super::flatten::SSARepr, num_pcs: usize) -> Vec<usize> {
    let mut anchors: Vec<(usize, usize)> = Vec::with_capacity(num_pcs);
    for (insn_idx, insn) in ssarepr.insns.iter().enumerate() {
        if let Insn::PcAnchor(py_pc) = insn {
            anchors.push((insn_idx, *py_pc));
        }
    }
    assert_eq!(
        anchors.len(),
        num_pcs,
        "live_marker_indices_by_pc: expected {num_pcs} PcAnchors, found {}",
        anchors.len()
    );
    let mut live_indices = vec![usize::MAX; num_pcs];
    for (anchor_pos, (anchor_idx, py_pc)) in anchors.iter().enumerate() {
        let end = anchors
            .get(anchor_pos + 1)
            .map(|(next_idx, _)| *next_idx)
            .unwrap_or(ssarepr.insns.len());
        let mut live_idx: Option<usize> = None;
        for insn_idx in (anchor_idx + 1)..end {
            if ssarepr.insns[insn_idx].is_live() {
                assert!(
                    live_idx.is_none(),
                    "live_marker_indices_by_pc: multiple -live- markers for py_pc {} in range {}..{}",
                    py_pc,
                    anchor_idx + 1,
                    end
                );
                live_idx = Some(insn_idx);
            }
        }
        live_indices[*py_pc] = live_idx.unwrap_or_else(|| {
            panic!(
                "live_marker_indices_by_pc: missing -live- marker for py_pc {} in range {}..{}",
                py_pc,
                anchor_idx + 1,
                end
            )
        });
    }
    live_indices
}

fn filter_liveness_in_place(
    ssarepr: &mut super::flatten::SSARepr,
    code: &CodeObject,
    nlocals: usize,
    stack_base: u16,
    depth_at_pc: &[u16],
) {
    use super::flatten::{Insn, Kind as SsaKind, Operand as SsaOperand};
    super::liveness::compute_liveness(ssarepr);
    let live_vars = pyre_jit_trace::state::liveness_for(code as *const _);
    let live_markers = live_marker_indices_by_pc(ssarepr, code.instructions.len());
    for (py_pc, insn_idx) in live_markers.into_iter().enumerate() {
        let existing = match ssarepr.insns.get_mut(insn_idx) {
            Some(insn) if insn.is_live() => insn.live_args_mut().unwrap(),
            Some(other) => panic!(
                "filter_liveness_in_place: expected -live- marker at index {insn_idx}, got {other:?}"
            ),
            None => panic!(
                "filter_liveness_in_place: insn index {insn_idx} out of range (len {})",
                ssarepr.insns.len()
            ),
        };
        // Preserve non-Register operands (TLabel) exactly as RPython
        // `liveness.py:52` keeps them alongside the `alive` set.
        let mut non_register: Vec<SsaOperand> = Vec::new();
        for op in existing.iter() {
            if !matches!(op, SsaOperand::Register(_)) {
                non_register.push(op.clone());
            }
        }

        if !live_vars.is_reachable(py_pc) {
            existing.clear();
            existing.extend(non_register);
            continue;
        }

        let depth = depth_at_pc[py_pc];
        let stack_limit = stack_base as usize + depth as usize;
        let mut seen_r: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        let mut seen_i: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        let mut seen_f: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        let mut live_r: Vec<u16> = Vec::new();
        let mut live_i: Vec<u16> = Vec::new();
        let mut live_f: Vec<u16> = Vec::new();
        // `liveness.py:67-75` `compute_liveness` adds every Register read
        // to the alive set without a kind filter. Preserve Int/Float
        // registers as-is; only Ref registers carry pyre's stack-base /
        // in-range adaptation (PRE-EXISTING-ADAPTATION, see the vable
        // locals/stack decoding contract below).
        for op in existing.iter() {
            let SsaOperand::Register(reg) = op else {
                continue;
            };
            match reg.kind {
                SsaKind::Ref => {
                    let idx = reg.index as usize;
                    let in_locals = idx < nlocals;
                    let in_stack = idx >= stack_base as usize && idx < stack_limit;
                    if (in_locals || in_stack) && seen_r.insert(reg.index) {
                        live_r.push(reg.index);
                    }
                }
                SsaKind::Int => {
                    if seen_i.insert(reg.index) {
                        live_i.push(reg.index);
                    }
                }
                SsaKind::Float => {
                    if seen_f.insert(reg.index) {
                        live_f.push(reg.index);
                    }
                }
                _ => {}
            }
        }
        for d in 0..depth {
            let idx = stack_base + d;
            if seen_r.insert(idx) {
                live_r.push(idx);
            }
        }
        let lv_live: std::collections::BTreeSet<u16> = {
            let mut s: std::collections::BTreeSet<u16> = (0..nlocals)
                .filter(|&idx| live_vars.is_local_live(py_pc, idx))
                .map(|idx| idx as u16)
                .collect();
            for d in 0..depth {
                s.insert(stack_base + d);
            }
            s
        };
        live_r.retain(|idx| lv_live.contains(idx));

        existing.clear();
        for &idx in &live_i {
            existing.push(SsaOperand::Register(super::flatten::Register::new(
                SsaKind::Int,
                idx,
            )));
        }
        for &idx in &live_r {
            existing.push(SsaOperand::Register(super::flatten::Register::new(
                SsaKind::Ref,
                idx,
            )));
        }
        for &idx in &live_f {
            existing.push(SsaOperand::Register(super::flatten::Register::new(
                SsaKind::Float,
                idx,
            )));
        }
        existing.extend(non_register);
    }
}

/// Decode `code.exceptiontable` into the structures the dispatch loop
/// consumes:
/// - `catch_for_pc[py_pc]` — `Some(landing_label)` for every PC that
///   falls inside an exception range, mapping to the landing label
///   the dispatch loop will branch to on raise.
/// - `catch_sites` — one entry per active range, holding the handler
///   PC, the saved stack depth, and the `push_lasti` flag. The
///   dispatch loop emits a landing block per entry at the end.
/// - `handler_depth_at[handler_pc]` — the stack depth Python sets on
///   exception-handler entry (`entry.depth + 1` plus another `+1`
///   when `push_lasti`); used by the dispatch loop to fix
///   `current_depth` at the handler's first instruction.
///
/// PRE-EXISTING-ADAPTATION: RPython has no analog because RPython
/// flow graphs already carry exception-handling links; pyre's input
/// is raw CPython bytecode + the packed exception table, so this
/// preprocessing step is pyre-specific.
fn decode_exception_catch_sites(
    assembler: &mut SSAReprEmitter,
    code: &CodeObject,
    num_instrs: usize,
) -> (
    Vec<Option<u16>>,
    Vec<ExceptionCatchSite>,
    std::collections::HashMap<usize, u16>,
) {
    let exception_entries =
        pyre_interpreter::bytecode::decode_exception_table(&code.exceptiontable);
    let mut catch_for_pc: Vec<Option<u16>> = vec![None; num_instrs];
    let mut catch_sites: Vec<ExceptionCatchSite> = Vec::new();
    for py_pc in 0..num_instrs {
        let Some(entry) = exception_entries
            .iter()
            .find(|entry| py_pc >= entry.start as usize && py_pc < entry.end as usize)
        else {
            continue;
        };
        let handler_py_pc = entry.target as usize;
        if handler_py_pc >= num_instrs {
            continue;
        }
        let landing_label = assembler.new_label();
        catch_for_pc[py_pc] = Some(landing_label);
        catch_sites.push(ExceptionCatchSite {
            landing_label,
            handler_py_pc,
            stack_depth: entry.depth,
            push_lasti: entry.push_lasti,
            lasti_py_pc: py_pc,
        });
    }
    let handler_depth_at: std::collections::HashMap<usize, u16> = exception_entries
        .iter()
        .map(|e| {
            let extra = if e.push_lasti { 1u16 } else { 0 };
            (e.target as usize, e.depth as u16 + extra + 1)
        })
        .collect();
    (catch_for_pc, catch_sites, handler_depth_at)
}

// Note: the legacy `liveness_regs_to_u8_sorted` helper that returned
// `Option<Vec<u8>>` to flag the 256-register cap is gone. The cap is
// now enforced by `majit_translate::liveness::encode_liveness`'s
// `assert!(char_ < 256)` (RPython `liveness.py:147-166` parity), and
// the post-pass register allocator
// (`super::regalloc::allocate_registers`) compresses the indices so
// the cap fires only on pathological functions whose `nlocals` alone
// exceeds 256 — the same condition that crashes the RPython
// translator.

/// Entry point for residual calls. Pushes the upstream-canonical
/// `residual_call_{kinds}_{reskind}` Insn into the walker-local
/// `ssarepr`; `Assembler::assemble`'s `dispatch_residual_call` arm
/// reconstructs the runtime `call_*_typed` path from the SSA operand
/// shape (jtransform.py:414-435 `rewrite_call` parity).
fn emit_residual_call(
    ssarepr: &mut SSARepr,
    _graph: &mut super::flow::FunctionGraph,
    current_block: &super::flow::BlockRef,
    flavor: CallFlavor,
    fn_idx: u16,
    call_args: &[majit_metainterp::jitcode::JitCallArg],
    reskind: ResKind,
    dst: Option<u16>,
) {
    emit_residual_call_shape(
        ssarepr,
        _graph,
        current_block,
        flavor,
        fn_idx,
        call_args,
        reskind,
        dst,
    );
}

/// Upstream-shape Insn builder for the walker-local `ssarepr`. See
/// `emit_residual_call` for the dual-emit policy.
///
/// `rpython/jit/codewriter/jtransform.py:414-435 rewrite_call` emits
///
/// ```text
/// SpaceOperation('%s_%s_%s' % (namebase, kinds, reskind),
///                [fn, ListOfKind('int',   args_i),
///                     ListOfKind('ref',   args_r),   # only if 'r' in kinds
///                     ListOfKind('float', args_f),   # only if 'f' in kinds
///                     calldescr],
///                result)
/// ```
///
/// where `kinds` is the smallest cover of actually-present arg kinds
/// (`'r'` if only ref args, `'ir'` if int+ref, `'irf'` if floats are
/// present or the result itself is float) and `reskind` ∈ {`i`, `r`,
/// `f`, `v`}. `namebase = 'residual_call'` for the direct-call helper
/// (`jtransform.py:460-471 handle_residual_call`).
///
/// This helper is the external-`SSARepr` side of the Phase 3b dual
/// emission: every call site pairs a direct `assembler.call_*_typed(...)`
/// (pushing a pyre-only `call_*` Insn into the runtime-consumed
/// `SSAReprEmitter::ssarepr`) with one call here, which pushes the
/// upstream-canonical shape into the walker-local `ssarepr` that
/// Phase 3c will wire up as the authoritative `Assembler::assemble`
/// input. Upstream stores `calldescr` (an `AbstractDescr` carrying
/// `EffectInfo`) in the trailing slot; pyre threads a
/// `DescrOperand::CallFlavor` there so Phase 3c's assembler dispatch
/// can recover the (flavor, reskind) pair the runtime path resolves
/// statically today.
fn emit_residual_call_shape(
    ssarepr: &mut SSARepr,
    _graph: &mut super::flow::FunctionGraph,
    _current_block: &super::flow::BlockRef,
    flavor: CallFlavor,
    fn_idx: u16,
    // Arguments in the C function's parameter order. `JitCallArg` carries
    // its own kind tag so pyre's runtime builder can interleave kinds on
    // the machine-code call. The SSARepr side projects this list into
    // kind-separated sublists to match upstream shape — upstream itself
    // loses per-C-parameter order in the SSARepr and relies on
    // `calldescr` at `bh_call_*` time to reconstruct it.
    call_args: &[majit_metainterp::jitcode::JitCallArg],
    reskind: ResKind,
    dst: Option<u16>,
) {
    use majit_metainterp::jitcode::JitArgKind;

    // `rpython/jit/codewriter/jtransform.py:437-445 make_three_lists` —
    // project parameter-ordered `call_args` into per-kind sublists for the
    // SSARepr shape. The original per-parameter kind sequence is preserved
    // in `arg_kinds` below so the assembler dispatch can reassemble the
    // flat `&[JitCallArg]` list pyre's builder expects.
    let mut args_i: Vec<u16> = Vec::new();
    let mut args_r: Vec<u16> = Vec::new();
    let mut args_f: Vec<u16> = Vec::new();
    let mut arg_kinds: Vec<Kind> = Vec::with_capacity(call_args.len());
    for arg in call_args {
        match arg.kind {
            JitArgKind::Int => {
                args_i.push(arg.reg);
                arg_kinds.push(Kind::Int);
            }
            JitArgKind::Ref => {
                args_r.push(arg.reg);
                arg_kinds.push(Kind::Ref);
            }
            JitArgKind::Float => {
                args_f.push(arg.reg);
                arg_kinds.push(Kind::Float);
            }
        }
    }

    // `rewrite_call` kinds selection (jtransform.py:423-426).
    let kinds: &str = if !args_f.is_empty() || reskind == ResKind::Float {
        "irf"
    } else if !args_i.is_empty() {
        "ir"
    } else {
        "r"
    };
    let reskind_ch = reskind.as_char();
    let opname = format!("residual_call_{kinds}_{reskind_ch}");

    // SSARepr arg list: [Const(fn), ListI?, ListR, ListF?, Descr(flavor)].
    let mut args: Vec<Operand> = Vec::with_capacity(5);
    args.push(Operand::ConstInt(fn_idx as i64));
    let reg_list = |kind: Kind, regs: &[u16]| {
        Operand::ListOfKind(ListOfKind::new(
            kind,
            regs.iter().map(|&r| Operand::reg(kind, r)).collect(),
        ))
    };
    if kinds.contains('i') {
        args.push(reg_list(Kind::Int, &args_i));
    }
    if kinds.contains('r') {
        args.push(reg_list(Kind::Ref, &args_r));
    }
    if kinds.contains('f') {
        args.push(reg_list(Kind::Float, &args_f));
    }
    args.push(Operand::descr(DescrOperand::CallDescrStub(
        super::flatten::CallDescrStub { flavor, arg_kinds },
    )));

    let insn = match (reskind.to_kind(), dst) {
        (Some(kind), Some(d)) => Insn::op_with_result(opname, args, Register::new(kind, d)),
        (None, None) => Insn::op(opname, args),
        (Some(_), None) => panic!("residual_call with non-void reskind requires dst"),
        (None, Some(_)) => panic!("residual_call with void reskind must not have dst"),
    };
    ssarepr.insns.push(insn.clone());
}

// ---------------------------------------------------------------------------
// RPython: codewriter/codewriter.py — class CodeWriter
// ---------------------------------------------------------------------------

/// Compiles Python CodeObjects into JitCode for blackhole execution.
///
/// RPython: `rpython/jit/codewriter/codewriter.py::CodeWriter`.
/// `codewriter.py:20-23` stores `self.assembler = Assembler()` and
/// `self.callcontrol = CallControl(cpu, jitdrivers_sd)` once on the
/// CodeWriter and reuses them across every `transform_graph_to_jitcode`
/// call so `all_liveness` / `num_liveness_ops` and the `jitcodes` dict
/// accumulate over the whole translator session.
///
/// pyre mirrors that ownership via a per-thread singleton: the process
/// holds a single `CodeWriter` instance (one per thread) reachable via
/// [`CodeWriter::instance`], matching `warmspot.py:245`
/// `codewriter = CodeWriter(cpu, [jd])`. The owned `Assembler` lives on
/// a `RefCell<Assembler>` field so `transform_graph_to_jitcode` can
/// still mutate it under the immutable-by-default singleton borrow.
pub struct CodeWriter {
    /// `codewriter.py:22` `self.assembler = Assembler()`.
    ///
    /// Single Assembler instance shared across every `transform_graph_to_jitcode`
    /// call on this CodeWriter. `all_liveness` / `all_liveness_positions` /
    /// `num_liveness_ops` accumulate here just like the upstream object.
    assembler: RefCell<Assembler>,
    /// RPython: `self.callcontrol = CallControl(cpu, jitdrivers_sd)`
    /// (codewriter.py:23). Owned in a `UnsafeCell` so `&CodeWriter` can
    /// mint `&mut CallControl` through [`Self::callcontrol`] — matches
    /// the legacy `JITCODE_CACHE` interior-mutability contract.
    callcontrol: UnsafeCell<super::call::CallControl>,
}

impl CodeWriter {
    /// RPython: `CodeWriter.__init__(cpu, jitdrivers_sd)` (codewriter.py:20-23).
    ///
    /// Phase A: the cpu helpers are fixed module-level functions in
    /// `crate::call_jit`; Phase D.2 wired `callcontrol` as a field so
    /// `writer.callcontrol()` matches `self.callcontrol` in upstream.
    pub fn new() -> Self {
        // codewriter.py:21-23 `self.cpu = cpu; self.assembler = Assembler();
        //   self.callcontrol = CallControl(cpu, jitdrivers_sd)`.
        // pyre owns the single `Cpu` on `CallControl`; `CodeWriter::cpu()`
        // returns a borrow back out so the upstream attribute access
        // pattern (`self.cpu`) still works.
        let cpu = super::cpu::Cpu::new();
        // Register the trace-side `jitcode_for` compile callback the
        // first time any `CodeWriter` is constructed in this process.
        // This is the lazy analog of jtransform's eager call to
        // `cc.callcontrol.get_jitcode(callee_graph)` (call.py:155):
        // when the tracer references a callee's `JitCode`, the same
        // `make_jitcodes` pipeline (codewriter.py:74-89) runs for that
        // one entry so `CallControl.find_jitcode` and
        // `MetaInterpStaticData.jitcodes` agree before the blackhole
        // resume looks anything up (resume.py:1338).
        static INIT_COMPILE_CALLBACK: std::sync::Once = std::sync::Once::new();
        INIT_COMPILE_CALLBACK.call_once(|| {
            pyre_jit_trace::set_compile_jitcode_fn(compile_jitcode_via_w_code);
        });
        Self {
            assembler: RefCell::new(Assembler::new()),
            callcontrol: UnsafeCell::new(super::call::CallControl::new(cpu, Vec::new())),
        }
    }

    /// `codewriter.py:21` `self.cpu = cpu`.
    ///
    /// Convenience accessor — pyre owns the single `Cpu` on
    /// `CallControl` (call.py:27 `self.cpu = cpu`); upstream both
    /// attributes point at the same object.
    pub fn cpu(&self) -> &super::cpu::Cpu {
        &self.callcontrol().cpu
    }

    /// RPython: `CodeWriter.setup_vrefinfo(self, vrefinfo)`
    /// (codewriter.py:91-94).
    ///
    /// ```python
    /// def setup_vrefinfo(self, vrefinfo):
    ///     # must be called at most once
    ///     assert self.callcontrol.virtualref_info is None
    ///     self.callcontrol.virtualref_info = vrefinfo
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: pyre has no `virtualref` machinery
    /// (no `@jit.virtual_ref`, no `vref_info` lookup); the slot is
    /// preserved so future warmspot wiring can call through with the
    /// same name.
    pub fn setup_vrefinfo(&self, vrefinfo: ()) {
        // codewriter.py:93 `assert self.callcontrol.virtualref_info is None`.
        assert!(self.callcontrol().virtualref_info.is_none());
        // codewriter.py:94 `self.callcontrol.virtualref_info = vrefinfo`.
        self.callcontrol().virtualref_info = Some(vrefinfo);
    }

    /// RPython: `CodeWriter.setup_jitdriver(self, jitdriver_sd)`
    /// (codewriter.py:96-99).
    ///
    /// ```python
    /// def setup_jitdriver(self, jitdriver_sd):
    ///     # Must be called once per jitdriver.
    ///     self.callcontrol.jitdrivers_sd.append(jitdriver_sd)
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: RPython appends unconditionally because
    /// each `@jit_callback` decoration calls `setup_jitdriver` exactly
    /// once at translation time. pyre's portal discovery is lazy and
    /// fires on every JIT entry, so the same `portal_graph` would be
    /// pushed repeatedly without the `find` guard below — `jitdrivers_sd`
    /// would grow linearly with JIT entries instead of staying bounded
    /// by the number of unique portals. The dedup updates the existing
    /// jd's `merge_point_pc` so the refinement hint propagates into
    /// the next `grab_initial_jitcodes` pass.
    pub fn setup_jitdriver(&self, jitdriver_sd: super::call::JitDriverStaticData) {
        let jitdriver_sd = jitdriver_sd.canonicalized();
        let cc = self.callcontrol();
        if let Some(existing) = cc
            .jitdrivers_sd
            .iter_mut()
            .find(|j| j.portal_graph == jitdriver_sd.portal_graph)
        {
            if jitdriver_sd.merge_point_pc.is_some() {
                existing.merge_point_pc = jitdriver_sd.merge_point_pc;
            }
            return;
        }
        // codewriter.py:99 `self.callcontrol.jitdrivers_sd.append(jitdriver_sd)`.
        cc.jitdrivers_sd.push(jitdriver_sd);
    }

    /// RPython: `self.callcontrol` (codewriter.py:23).
    ///
    /// Returns a mutable reference to the owned `CallControl`. Safe under
    /// the same invariant as the legacy `JITCODE_CACHE` thread_local: the
    /// caller must not re-enter `callcontrol()` while the returned borrow
    /// is live.
    #[allow(clippy::mut_from_ref)]
    pub fn callcontrol(&self) -> &mut super::call::CallControl {
        // SAFETY: `CodeWriter` is only accessed via `instance()` which
        // returns a thread-local reference; all callers execute on the
        // owning thread.
        unsafe { &mut *self.callcontrol.get() }
    }

    /// Access the process-wide single `CodeWriter` — analog of the
    /// single `codewriter` owned by `warmspot.py:245-281` for the
    /// lifetime of the JIT.
    ///
    /// Implemented as a per-thread singleton: pyre's JIT currently runs
    /// one interpreter per thread and function pointers in `Self` are
    /// `Sync`, so a thread-local provides the RPython "one CodeWriter
    /// per warmspot" invariant without a global lock.
    pub fn instance() -> &'static CodeWriter {
        thread_local! {
            static INSTANCE: CodeWriter = CodeWriter::new();
        }
        INSTANCE.with(|cw| unsafe { &*(cw as *const CodeWriter) })
    }

    /// Transform a Python CodeObject into a JitCode.
    ///
    /// RPython: CodeWriter.transform_graph_to_jitcode(graph, jitcode, verbose, index)
    ///
    /// Python bytecodes serve as the "graph". Since they are already linear
    /// and register-allocated, jtransform/regalloc/flatten are identity
    /// transforms. We go directly to assembly.
    pub fn transform_graph_to_jitcode(
        &self,
        code: &CodeObject,
        w_code: *const (),
        merge_point_pc: Option<usize>,
    ) -> PyJitCode {
        // RPython codewriter.py:46-48 `regallocs[kind] = perform_register_allocation(graph, kind)`.
        // pyre's regalloc is trivial — fast locals occupy the bottom of
        // the ref register file and the operand stack stacks above
        // them — so the "allocation" reduces to a `RegisterLayout`
        // computed from `code.varnames` / `code.max_stackdepth`.
        let layout = RegisterLayout::compute(code);
        let RegisterLayout {
            nlocals,
            ncells: _,
            stack_base_absolute,
            max_stackdepth,
            stack_base,
            obj_tmp0,
            obj_tmp1,
            arg_regs_start,
            null_ref_reg,
            portal_frame_reg,
            portal_ec_reg,
            int_tmp0,
            int_tmp1,
            op_code_reg,
        } = layout;
        // jtransform.py: virtualizable field indices for getfield_vable_*
        // interp_jit.py:25-31 / virtualizable_spec.rs parity:
        //   0=last_instr, 1=code, 2=vsd, 3=debugdata, 4=lastblock, 5=namespace
        const VABLE_CODE_FIELD_IDX: u16 = 1;
        const VABLE_VALUESTACKDEPTH_FIELD_IDX: u16 = 2;
        const VABLE_NAMESPACE_FIELD_IDX: u16 = 5;

        // regalloc.py: compile-time stack depth counter — tracks which
        // stack register (stack_base + depth) is the current TOS.
        let mut current_depth: u16 = 0;

        // RPython: self.assembler = Assembler() + JitCode(graph.name, ...)
        // (rpython/jit/codewriter/jitcode.py:14-15 takes name as the first
        // __init__ argument; majit's JitCodeBuilder::set_name mirrors that).
        let mut assembler = SSAReprEmitter::new();
        assembler.set_name(code.obj_name.to_string());
        // B6 Phase 3b scaffolding: grow an `SSARepr` alongside the direct
        // `JitCodeBuilder` calls. Currently only a handful of handlers
        // (`ref_return` below) dual-emit an `Insn::Op`; the remaining
        // bytecode handlers still route through the builder only. When
        // every handler has been converted, `ssarepr` becomes the
        // authoritative input to `jit::assembler::Assembler::assemble`
        // (Phase 3c switchover) and the direct builder calls disappear.
        // See `pyre/pyre-jit/src/jit/B6_CODEWRITER_PIPELINE_PLAN.md`.
        let mut ssarepr = SSARepr::new(code.obj_name.to_string());

        // RPython regalloc.py: keep kind-separated register files.
        assembler.ensure_r_regs(portal_ec_reg + 1);
        assembler.ensure_i_regs(op_code_reg + 1);

        // Register helper fn pointers in the canonical order; the
        // returned struct names every index so the dispatch handlers
        // below can reference them by field instead of an opaque local.
        let FnPtrIndices {
            call_fn: call_fn_idx,
            load_global_fn: load_global_fn_idx,
            compare_fn: compare_fn_idx,
            binary_op_fn: binary_op_fn_idx,
            box_int_fn: box_int_fn_idx,
            truth_fn: truth_fn_idx,
            load_const_fn: load_const_fn_idx,
            store_subscr_fn: store_subscr_fn_idx,
            build_list_fn: build_list_fn_idx,
            normalize_raise_varargs_fn: normalize_raise_varargs_fn_idx,
            call_fn_0: call_fn_0_idx,
            call_fn_2: call_fn_2_idx,
            call_fn_3: call_fn_3_idx,
            call_fn_4: call_fn_4_idx,
            call_fn_5: call_fn_5_idx,
            call_fn_6: call_fn_6_idx,
            call_fn_7: call_fn_7_idx,
            call_fn_8: call_fn_8_idx,
            get_current_exception_fn: get_current_exception_fn_idx,
            set_current_exception_fn: set_current_exception_fn_idx,
        } = register_helper_fn_pointers(&mut assembler, self.cpu());

        // RPython flatten.py: pre-create labels for each block.
        // Python bytecodes are linear, so each instruction index gets a label.
        let num_instrs = code.instructions.len();
        let mut labels: Vec<u16> = Vec::with_capacity(num_instrs);
        for _ in 0..num_instrs {
            labels.push(assembler.new_label());
        }

        let (catch_for_pc, catch_sites, handler_depth_at) =
            decode_exception_catch_sites(&mut assembler, code, num_instrs);

        // Step 6.1 Phase 2a: shadow `FunctionGraph` alongside `ssarepr`.
        //
        // RPython's flow space keeps `framestate` on each `SpamBlock`
        // (`flowcontext.py:38-44`) and derives `Link.args ↔
        // target.inputargs` from `FrameState.getoutputargs()`. Pyre's
        // walker is still single-pass over Python bytecode, but the
        // shadow graph now carries the same per-block `FrameState`
        // object instead of a topology-only `BlockRef`.
        //
        // Portal graphs (whose bytecode contains a `jit_merge_point`
        // marker, see `merge_point_pc`) carry two extra red inputs —
        // `(frame, ec)` — appended to both `startblock.inputargs` via
        // `graph_entry_inputargs(code, portal_inputs=true)` AND to
        // `FrameState` via `entry_frame_state(code, portal_inputs=
        // true)`.  `FrameState.portal_extras` carries those Variables
        // through every block transition so `getoutputargs()` on any
        // backedge produces link args aligned with the appended
        // startblock slots.  Non-portal graphs populate neither side
        // and behave exactly as before.
        let mut graph = new_shadow_graph_with_portal_inputs(code, merge_point_pc.is_some());
        let mut pc_blocks: Vec<Option<SpamBlockRef>> = vec![None; num_instrs];
        let mut joinpoints: HashMap<usize, Vec<SpamBlockRef>> = HashMap::new();
        // Step 6A slice S4a: snapshot the walker's `currentstate` at
        // every terminator emission so `collect_link_slot_pairs` can
        // translate link-arg Variables to SSARepr register slots via
        // the positional walk.  RPython does not need this map because
        // `regalloc.py:79-96` unions Variables directly via UnionFind;
        // pyre's u16-keyed regalloc (regalloc.rs:26-36 PRE-EXISTING-
        // ADAPTATION) reads the source state per-link to project back
        // onto slots.  Keyed on `LinkRef` (Rc-pointer identity).
        let mut link_exit_states: HashMap<super::flow::LinkRef, FrameState> = HashMap::new();
        let start_state = entry_frame_state(code, merge_point_pc.is_some());
        if num_instrs > 0 {
            let start_block =
                SpamBlockRef::new(graph.startblock.clone(), Some(start_state.clone()));
            pc_blocks[0] = Some(start_block.clone());
            joinpoints.insert(0, vec![start_block]);
        }
        let mut catch_landing_blocks: HashMap<u16, SpamBlockRef> =
            HashMap::with_capacity(catch_sites.len());
        for site in &catch_sites {
            catch_landing_blocks.insert(
                site.landing_label,
                SpamBlockRef::new(graph.new_block(Vec::new()), None),
            );
        }
        // The walker emits into `current_block`; `emit_mark_label_pc!` and
        // `emit_mark_label_catch_landing!` reassign it as the walker enters
        // each block. Initialised to the first PC block so the PcAnchor /
        // live_placeholder / jit_merge_point emissions that precede the
        // first `emit_mark_label_pc!` belong to it.
        let mut current_block: SpamBlockRef = pc_blocks
            .first()
            .and_then(|block| block.clone())
            .unwrap_or_else(|| {
                SpamBlockRef::new(graph.startblock.clone(), Some(start_state.clone()))
            });
        let mut current_state = current_block
            .framestate()
            .unwrap_or_else(|| start_state.clone());
        // Tracks whether the current block still needs an implicit
        // fallthrough `Link` on the next `emit_mark_label_pc!`. Reset
        // to `true` at every block entry; terminator macros that fully
        // close the block (`emit_goto!` / `emit_ref_return!` /
        // `emit_raise!` / `emit_reraise!` / `emit_abort_permanent!`)
        // clear it. Terminators that leave fallthrough open —
        // `emit_goto_if_not!`, `emit_goto_if_not_int_is_zero!`,
        // `emit_catch_exception!` — keep it set. Mirrors RPython
        // `flatten.py:240-267` where a conditional / exception exit
        // always coexists with the straight-line successor on
        // `Block.exits`.
        let mut needs_fallthrough: bool = true;

        // interp_jit.py:118 `pypyjitdriver.can_enter_jit(...)` is called in
        // `jump_absolute` (`jumpto < next_instr` branch), i.e. at each
        // Python backward jump.  jtransform.py:1714-1723
        // `handle_jit_marker__can_enter_jit = handle_jit_marker__loop_header`
        // lowers each one to a `loop_header` jitcode op.  Pyre has no
        // `jump_absolute` Python wrapper — the equivalent is to pre-scan
        // `JumpBackward` opcodes and record their targets; each target PC
        // becomes a `loop_header` site.
        let loop_header_pcs = find_loop_header_pcs(code);

        // RPython: flatten_graph() walks blocks and emits instruction tuples.
        // RPython: assembler.assemble(ssarepr, jitcode, num_regs) emits bytecodes.
        // For pyre, we combine both steps: walk Python bytecodes and emit
        // JitCode bytecodes directly.
        let mut arg_state = OpArgState::default();
        // liveness.py parity: record stack depth at each Python PC for
        // precise liveness generation. Stack registers stack_base..stack_base+depth
        // are live at each PC.
        let mut depth_at_pc: Vec<u16> = vec![0; num_instrs];
        // codewriter.py:37 `portal_jd = self.callcontrol.jitdriver_sd_from_portal_graph(graph)`
        // — RPython looks up portal-ness in the registry that
        // `setup_jitdriver` populates. pyre matches that: a code is a
        // portal iff it is in `CallControl.jitdrivers_sd`. The portal
        // path (`register_portal_jitdriver`) registers before the drain
        // runs `transform_graph_to_jitcode`, so the lookup sees the
        // entry; the callee path (`compile_jitcode_for_callee`) never
        // touches `jitdrivers_sd`, so the lookup returns `None`. This
        // replaces the older "any backedge → portal" heuristic.
        let is_portal = self
            .callcontrol()
            .jitdriver_sd_from_portal_graph(code as *const CodeObject)
            .is_some();
        // RPython parity: every backward jump goes through dispatch() →
        // jit_merge_point(). The blackhole's bhimpl_jit_merge_point raises
        // ContinueRunningNormally at the bottommost level. Ideally all
        // loop headers should emit BC_JIT_MERGE_POINT, but the
        // CRN→interpreter→JIT-reentry cycle crashes in JIT compiled code
        // because blackhole-modified frame locals can contain values
        // incompatible with the compiled trace's assumptions. Until the
        // full RPython CRN→portal_ptr restart is implemented, only the
        // first loop header is a merge point.
        // RPython jtransform.py:1690: jit_merge_point only in the portal
        // graph. merge_point_pc is the trace entry PC (from bound_reached).
        // Other loop headers use loop_header (no-op in the blackhole).
        let merge_point_pc = if is_portal {
            merge_point_pc.or_else(|| loop_header_pcs.iter().copied().min())
        } else {
            // Callee — no jit_merge_point emit. RPython's jtransform.py:1690
            // `jit_merge_point only in the portal graph` is the matching
            // statement.
            None
        };

        // pyframe.py:379-417 pushvalue/popvalue_maybe_none parity:
        // Each push/pop writes self.valuestackdepth = depth ± 1.
        // jtransform.py:923-928 lowers this to setfield_vable_i.
        // This macro emits the equivalent BC_SETFIELD_VABLE_I after
        // every current_depth mutation so the frame's valuestackdepth
        // stays in sync at every guard/call point — matching RPython's
        // per-push/per-pop semantics.
        macro_rules! emit_vsd {
            ($depth:expr) => {
                if is_portal {
                    emit_load_const_i!(
                        ssarepr,
                        int_tmp0,
                        (stack_base_absolute + $depth as usize) as i64
                    );
                    emit_vable_setfield_int!(ssarepr, VABLE_VALUESTACKDEPTH_FIELD_IDX, int_tmp0);
                }
            };
        }

        // PRE-EXISTING-ADAPTATION: the `BC_ABORT_PERMANENT` runtime
        // bytecode does not appear in `rpython/jit/codewriter/` or
        // `rpython/jit/metainterp/`. RPython refuses to build jitcode for
        // bytecodes it cannot translate (the translator surfaces the
        // failure at build time); pyre must always produce runnable
        // jitcode because bytecode translation is lazy at runtime. We
        // therefore keep the runtime-side adaptation (assembler emits
        // `BC_ABORT_PERMANENT` so the blackhole interpreter falls back to
        // CPython evaluation) but never surface the pyre-only opname into
        // the RPython-parity SSARepr layer — `flatten.py:106` uses plain
        // `Label` for loop headers and `assembler.py:159` does not encode
        // unsupported bytecodes as named opnames.

        // B6 Phase 3b dual emission for `last_exc_value`. RPython parity:
        // `flatten.py:347` `self.emitline("last_exc_value", "->",
        // self.getcolor(w))` — `assembler.py:220` turns it into
        // `last_exc_value/>r`. pyre emits this once per catch site to
        // load the thread-local exception into the handler's input
        // register.
        macro_rules! emit_last_exc_value {
            ($ssarepr:expr, $dst:expr) => {{
                let dst = $dst;
                let insn = Insn::op_with_result(
                    "last_exc_value",
                    Vec::new(),
                    Register::new(Kind::Ref, dst),
                );
                $ssarepr.insns.push(insn.clone());
            }};
        }

        // PRE-EXISTING-ADAPTATION: the `BC_JUMP_TARGET` runtime opcode
        // does not appear in `rpython/jit/codewriter/`. RPython marks
        // loop-header block entries with a plain `Insn::Label` and lets
        // the blackhole's dispatch loop recognise them via the label
        // position; pyre emits a dedicated `BC_JUMP_TARGET` opcode so the
        // runtime inner-loop can cheaply identify back-edge targets
        // without consulting a label table. The runtime-side adaptation
        // stays (assembler dispatch at `assembler.rs:367-372`) but the
        // pyre-only opname is not surfaced into the RPython-parity
        // SSARepr layer — `flatten.py:106` uses plain `Label` for loop
        // headers.

        // B6 Phase 3b dual emission for `int_copy` / `ref_copy` /
        // `float_copy` with a Constant source. RPython parity:
        // `flatten.py:333` `self.emitline('%s_copy' % kind, v, "->", w)`
        // — `v` is resolved via `getcolor(v)` which returns either a
        // `Register` or an unchanged `Constant` (see `flatten.py:382-384`).
        // The `assembler.py:140-222` dispatch handles both: the Register
        // source emits an `int_copy/i>i` entry, and the Constant source
        // emits an `int_copy/c>i` entry (argcode `'c'` for a compact
        // Constant — `>` is the result marker per
        // `assembler.py:210-212`). pyre's legacy `load_const_{i,r,f}_value`
        // emits the same runtime bytes under pyre-only `load_const_*`
        // opnames; the SSARepr now carries the RPython-parity `*_copy`
        // name with a ConstInt/ConstRef/ConstFloat source operand.
        // Per-opname integer / float primitives — `int_add`, `int_sub`,
        // `int_mul`, `int_{floordiv,mod,and,or,xor,lshift,rshift}`,
        // `int_{eq,ne,lt,le,gt,ge}`, `int_neg`, `int_invert`,
        // `uint_{rshift,mul_high,lt,le,gt,ge}`, `float_{add,sub,mul,
        // truediv,neg,abs}` — flow through canonical RPython opnames
        // and the matching `record_binop_*` / `record_unary_*` arms in
        // `assembler.rs`. The build-time pyre codewriter currently
        // emits `BINARY_OP` / `COMPARE_OP` via polymorphic residual
        // calls because pyre can't prove static operand types from the
        // bytecode alone; the canonical per-opname handlers handle
        // emissions that come from #[jit_interp]-lowered macros.

        // Call family intentionally has NO dual-emit.
        //
        // `rpython/jit/codewriter/jtransform.py:414-435` `rewrite_call()`
        // emits `residual_call_{kinds}_{reskind}` with
        // `[fnptr_constant, ListOfKind(int)?, ListOfKind(ref),
        //   ListOfKind(float)?, calldescr]`. pyre's runtime ABI uses
        // a caller-order Register list plus a u16 helper-table index and
        // encodes `may_force` in the target bytecode — none of which
        // fit into RPython's SSA tuple shape. Reviewer guidance:
        //   "codewriter는 원본 RPython tuple 을 만들고, pyre 적응은
        //    assembler 가 해야 한다."
        // Rather than baking the pyre shape into the SSA (which would
        // ossify a pyre-only SSA vocabulary), we keep the call handlers
        // on the direct builder path until assembler.rs grows exact
        // `residual_call_*` dispatch that can reconstruct the pyre
        // caller-order list. See `B6_CODEWRITER_PIPELINE_PLAN.md`.

        macro_rules! emit_load_const_i {
            ($ssarepr:expr, $dst:expr, $value:expr $(,)?) => {{
                let dst = $dst;
                let value: i64 = $value;
                let insn = Insn::op_with_result(
                    "int_copy",
                    vec![Operand::ConstInt(value)],
                    Register::new(Kind::Int, dst),
                );
                $ssarepr.insns.push(insn.clone());
            }};
        }

        // B6 Phase 3b dual emission for `ref_return`. Every site that used
        // to invoke `assembler.ref_return(src)` now also appends an
        // `Insn::Op { opname: "ref_return", args: [Register(Ref, src)] }`
        // to the SSARepr so the future `Assembler::assemble` path
        // (`assembler.rs::dispatch_op:374`) can reproduce the same byte at
        // the Phase 3c switchover. The direct builder call stays until the
        // switchover runs so the emitted JitCode remains bit-identical.
        // RPython parity: `rpython/jit/codewriter/jtransform.py` emits
        // `op_ref_return(v)` via `rewrite_op_jit_return` for the portal
        // return path; `assembler.py:221` turns that into the `ref_return/r`
        // bytecode key.
        macro_rules! emit_ref_return {
            ($ssarepr:expr, $src:expr, $retval:expr) => {{
                let src = $src;
                let retval = $retval;
                let insn = Insn::op("ref_return", vec![Operand::reg(Kind::Ref, src)]);
                $ssarepr.insns.push(insn.clone());
                // `rpython/jit/codewriter/flatten.py:144-146`: terminators
                // emit `('---',)` so the backward liveness pass clears its
                // alive set.
                $ssarepr.insns.push(Insn::Unreachable);
                // Step 6.1 Phase 2c: attach the return edge to
                // `graph.returnblock` (`model.py:18`). The return value
                // now comes from the symbolic `FrameState` stack,
                // matching `flatten.py:130-139` `make_return(args)`.
                let link =
                    super::flow::Link::new(vec![retval], Some(graph.returnblock.clone()), None)
                        .into_ref();
                // Step 6A slice S4a: snapshot the EXIT FrameState.
                append_exit_with_state(
                    &current_block.block(),
                    link,
                    &current_state,
                    &mut link_exit_states,
                );
                needs_fallthrough = false;
            }};
        }

        // B6 Phase 3b dual emission for `goto`. RPython parity:
        // `flatten.py:161` `self.emitline('goto', TLabel(link.target))` —
        // `assembler.py:220` turns the op into `goto/L`. Pyre labels are
        // integer indices into `labels[]`, one per Python PC; the
        // `TLabel` carries the synthetic name `pc{target_py_pc}` so the
        // Phase 3c dispatch (`assembler.rs::dispatch_op:345`) can resolve
        // it against `builder_label`.
        macro_rules! emit_goto {
            ($ssarepr:expr, $target_py_pc:expr) => {{
                let target_py_pc = $target_py_pc;
                let insn = Insn::op(
                    "goto",
                    vec![Operand::TLabel(TLabel::new(format!("pc{}", target_py_pc)))],
                );
                $ssarepr.insns.push(insn.clone());
                // `rpython/jit/codewriter/flatten.py:111-112`: an
                // unconditional goto implicitly ends a block so the
                // liveness pass (`liveness.py:68-69`) can reset the alive
                // set.
                $ssarepr.insns.push(Insn::Unreachable);
                // Step 6.1 Phase 2b: attach a single unconditional
                // `Link` from the current block to the target PC's
                // block. `flatten.py:161` `self.emitline('goto',
                // TLabel(link.target))` is the serialised view of the
                // same edge.
                mergeblock(
                    code,
                    &mut graph,
                    &mut joinpoints,
                    &mut pc_blocks,
                    &current_block,
                    &{
                        let mut branch_state = current_state.clone();
                        branch_state.next_offset = target_py_pc;
                        branch_state.blocklist = frame_blocks_for_offset(code, target_py_pc);
                        branch_state
                    },
                    target_py_pc,
                    &mut link_exit_states,
                );
                needs_fallthrough = false;
            }};
        }

        // B6 Phase 3b dual emission for `abort_permanent`. The opname is
        // a pyre-only runtime construct (`BC_ABORT_PERMANENT`) with no
        // counterpart in `rpython/jit/codewriter/*` or
        // `rpython/jit/metainterp/*` — pyre uses it to short-circuit the
        // translation of unsupported bytecode handlers and permanent
        // guard-fail edges, which upstream sidesteps via
        // `rpython/jit/metainterp/policy.py`-driven whitelisting. Because
        // the opname *already* surfaces at the runtime layer, Phase 3c's
        // single-SSARepr requirement forces it through the walker-local
        // `ssarepr` too; the alternative — a hybrid "some ops go through
        // SSARepr, some don't" dispatch — defeats the purpose of the
        // switchover. `dispatch_op` in `assembler.rs:510` already routes
        // `"abort_permanent"` to the builder, so the external push is
        // an exact mirror of the pre-existing internal behavior.
        macro_rules! emit_abort_permanent {
            ($ssarepr:expr) => {{
                let insn = Insn::op("abort_permanent", Vec::new());
                $ssarepr.insns.push(insn.clone());
                // pyre-only dead-end: the block has no successor in
                // the shadow graph. Leaving `needs_fallthrough = false`
                // blocks the auto-fallthrough at the next
                // `emit_mark_label_pc!`.
                needs_fallthrough = false;
            }};
        }

        // B6 Phase 3b dual emission for `raise`. RPython parity:
        // `flatten.py` emits `self.emitline("raise", self.getcolor(args[1]))`
        // inside the exception-link handler; `assembler.py:220` turns it
        // into `raise/r`. pyre's single `emit_raise(exc_reg)` call site
        // (RAISE_VARARGS with argc >= 1) corresponds to the same edge.
        macro_rules! emit_raise {
            ($ssarepr:expr, $src:expr, $evalue:expr, $offset:expr) => {{
                let src = $src;
                let evalue_fv: super::flow::FlowValue = $evalue;
                let offset = $offset;
                let insn = Insn::op("raise", vec![Operand::reg(Kind::Ref, src)]);
                $ssarepr.insns.push(insn.clone());
                // `flowcontext.py:1246-1261 Raise.nomoreblocks` shape:
                //   link = Link([w_exc.w_type, w_exc.w_value],
                //               ctx.graph.exceptblock)
                // `w_exc.w_value` is the actual trace-level FlowValue
                // of the raised exception instance; `w_exc.w_type`
                // upstream is a statically-known Constant because flow
                // analysis sees the `raise SomeError(...)` source form.
                //
                // pyre still emits a single runtime `raise/r`, but the
                // shadow graph can mirror `flowcontext.py:635-636`
                // exactly by recording `w_type = type(w_value)` and
                // routing that result through the explicit raise edge.
                // Like upstream `Raise.nomoreblocks`, this edge does
                // NOT use `link.extravars`.
                let edge_state = explicit_raise_state(
                    &mut graph,
                    &current_block.block(),
                    &current_state,
                    evalue_fv,
                    offset,
                );
                let link = super::flow::Link::new(
                    exceptblock_link_args(&edge_state),
                    Some(graph.exceptblock.clone()),
                    None,
                );
                let link = link.into_ref();
                append_exit_with_state(
                    &current_block.block(),
                    link,
                    &edge_state,
                    &mut link_exit_states,
                );
                needs_fallthrough = false;
            }};
        }

        // B6 Phase 3b dual emission for `reraise`. RPython parity:
        // `flatten.py` emits the zero-arg `self.emitline("reraise")` for
        // the re-raise edge; `assembler.py:220` turns it into
        // `reraise/`. pyre emits this for RAISE_VARARGS with argc == 0.
        macro_rules! emit_reraise {
            ($ssarepr:expr) => {{
                let insn = Insn::op("reraise", Vec::new());
                $ssarepr.insns.push(insn.clone());
                // Step 6.1 Phase 2c: same edge as `emit_raise!` — the
                // re-raise opname shares the `Block.exits` topology
                // (`flatten.py` emits the two as alternative codings
                // of the same exception exit).
                //
                // `reraise` preserves the current handler exception in
                // `FrameState.last_exception` (framestate.py:22).
                // Upstream `rpython/jit/codewriter/flatten.py:161-162`
                // `make_exception_link` asserts
                //     assert link.last_exception is not None
                //     assert link.last_exc_value is not None
                // before emitting `reraise`, so reaching this macro
                // with `current_state.last_exception == None` is a
                // structural bug in the caller rather than a normal
                // path. Fail loudly instead of quietly constructing
                // a sentinel-filled exit link.
                let (etype, evalue) = exception_edge_extravars(&current_state);
                let mut link = super::flow::Link::new(
                    exceptblock_link_args(&current_state),
                    Some(graph.exceptblock.clone()),
                    None,
                );
                // `flowcontext.py:141-143` `guessexception` / `model.py:
                // 127-129 Link.extravars`: pass the exception pair as
                // both `link.args` and `link.extravars` so the
                // downstream `flatten.py:163-174 make_exception_link`
                // check `link.args == [link.last_exception,
                // link.last_exc_value]` matches and emits `reraise`.
                link.extravars(Some(etype), Some(evalue));
                let link = link.into_ref();
                // Step 6A slice S4a: snapshot the EXIT state (same
                // reasoning as `emit_raise!`).
                append_exit_with_state(
                    &current_block.block(),
                    link,
                    &current_state,
                    &mut link_exit_states,
                );
                needs_fallthrough = false;
            }};
        }

        // B6 Phase 3b dual emission for `catch_exception`. RPython parity:
        // `flatten.py` emits `self.emitline('catch_exception',
        // TLabel(block.exits[0]))` when a block has an exception edge;
        // `assembler.py:220` turns it into `catch_exception/L`. pyre
        // emits this after each Python PC that has an exception handler.
        // The catch landing block lives after the main loop
        // (`mark_label(site.landing_label)`), so the `TLabel` carries
        // `catch_landing_{landing_label}` — distinct from the
        // `pc{py_pc}` naming used for PC-indexed labels.
        macro_rules! emit_catch_exception {
            ($ssarepr:expr, $catch_label:expr) => {{
                let catch_label = $catch_label;
                let insn = Insn::op(
                    "catch_exception",
                    vec![Operand::TLabel(TLabel::new(format!(
                        "catch_landing_{}",
                        catch_label
                    )))],
                );
                $ssarepr.insns.push(insn.clone());
                // Step 6.1 Phase 2b: attach the exception edge to the
                // current PC's block. In RPython this is the
                // `Constant(last_exception)` exit added by
                // `flatten.py` when the block `canraise`; the matching
                // normal-control-flow Link (fallthrough / goto) is
                // added by its own emit macro so the two edges coexist
                // on `Block.exits`.
                attach_catch_exception_edge(
                    &mut graph,
                    &current_block.block(),
                    &catch_landing_blocks[&catch_label],
                    &current_state,
                    &mut link_exit_states,
                );
            }};
        }

        // B6 Phase 3b dual emission for block `Label`. RPython parity:
        // `flatten.py:180` `self.emitline(Label(block))` marks block
        // entry; `assembler.py:157-158` records the label position in
        // `self.label_positions`. pyre marks a label at every Python PC
        // (`mark_label(labels[py_pc])`) and at each catch landing
        // block's entry. The two naming schemes (`pc{py_pc}` vs
        // `catch_landing_{u16}`) match the TLabel schemes used by
        // `emit_goto!` and `emit_catch_exception!`.
        macro_rules! emit_mark_label_pc {
            ($ssarepr:expr, $py_pc:expr) => {{
                let py_pc = $py_pc;
                $ssarepr
                    .insns
                    .push(Insn::Label(super::flatten::Label::new(format!(
                        "pc{}",
                        py_pc
                    ))));
                // Step 6.1 Phase 2d: if the previous block still needs
                // a fallthrough edge, attach one before switching
                // `current_block`. `new_block != current_block` skips
                // the self-link on the very first PC (where both refs
                // are `pc_blocks[0]`).
                let new_block = if needs_fallthrough {
                    mergeblock(
                        code,
                        &mut graph,
                        &mut joinpoints,
                        &mut pc_blocks,
                        &current_block,
                        &current_state,
                        py_pc,
                        &mut link_exit_states,
                    )
                } else {
                    ensure_pc_block(
                        code,
                        &mut graph,
                        &mut joinpoints,
                        &mut pc_blocks,
                        &current_state,
                        py_pc,
                    )
                };
                current_block = new_block;
                current_state = current_block
                    .framestate()
                    .expect("block state should exist at label");
                needs_fallthrough = true;
            }};
        }
        macro_rules! emit_mark_label_catch_landing {
            ($ssarepr:expr, $landing_label:expr) => {{
                let landing_label = $landing_label;
                $ssarepr
                    .insns
                    .push(Insn::Label(super::flatten::Label::new(format!(
                        "catch_landing_{}",
                        landing_label
                    ))));
                // Step 6.1 Phase 2a: switch the shadow graph's
                // `current_block` into the pre-allocated catch-landing
                // block. Matches `flatten.py:180` `Label(block)` being the
                // block-entry marker in RPython. Catch landings are
                // reached via `catch_exception` edges rather than
                // fallthrough, so no implicit Link is inserted here —
                // reset `needs_fallthrough` for the landing block's
                // own emission sequence.
                current_block = catch_landing_blocks[&landing_label].clone();
                if let Some(state) = current_block.framestate() {
                    current_state = state;
                }
                needs_fallthrough = true;
            }};
        }

        // B6 Phase 3b dual emission for `-live-`. RPython parity:
        // `flatten.py` inserts `self.emitline('-live-')` at every block
        // entry and at every point with a live resume set; `assembler.py:146-158`
        // allocates an `all_liveness` slot and encodes the live-register
        // triple. pyre emits the same placeholder here, then fills it
        // after regalloc by rescanning the final SSARepr's PcAnchor ranges.
        macro_rules! emit_live_placeholder {
            ($ssarepr:expr) => {{
                $ssarepr.insns.push(Insn::live(Vec::new()));
            }};
        }

        // flatten.py:240-260 boolean exitswitch emission. When the bool is a
        // plain variable (truth_fn result), flatten emits `goto_if_not <v> L`
        // (alias of bhimpl_goto_if_not_int_is_true per blackhole.py:913).
        // PopJumpIfTrue inverts the polarity via jtransform.py:1212
        // `_rewrite_equality` + flatten.py:247 specialisation
        // `goto_if_not_int_is_zero <v> L` (blackhole.py:916-920).
        macro_rules! emit_goto_if_not {
            ($ssarepr:expr, $cond:expr, $py_pc:expr) => {{
                let cond = $cond;
                let py_pc = $py_pc;
                let insn = Insn::op(
                    "goto_if_not",
                    vec![
                        Operand::reg(Kind::Int, cond),
                        Operand::TLabel(TLabel::new(format!("pc{}", py_pc))),
                    ],
                );
                $ssarepr.insns.push(insn.clone());
                // Step 6.1 Phase 2b: attach the conditional-False edge
                // to `pc_blocks[py_pc]`. RPython `flatten.py:240-267`
                // records both the False target and the fallthrough on
                // the block's `exits`; the fallthrough link is added
                // implicitly at the next `emit_mark_label_pc!` in a
                // follow-up slice when pyre's walker learns to insert
                // fallthrough Links for non-terminating blocks.
                mergeblock(
                    code,
                    &mut graph,
                    &mut joinpoints,
                    &mut pc_blocks,
                    &current_block,
                    &{
                        let mut branch_state = current_state.clone();
                        branch_state.next_offset = py_pc;
                        branch_state.blocklist = frame_blocks_for_offset(code, py_pc);
                        branch_state
                    },
                    py_pc,
                    &mut link_exit_states,
                );
            }};
        }
        macro_rules! emit_goto_if_not_int_is_zero {
            ($ssarepr:expr, $cond:expr, $py_pc:expr) => {{
                let cond = $cond;
                let py_pc = $py_pc;
                let insn = Insn::op(
                    "goto_if_not_int_is_zero",
                    vec![
                        Operand::reg(Kind::Int, cond),
                        Operand::TLabel(TLabel::new(format!("pc{}", py_pc))),
                    ],
                );
                $ssarepr.insns.push(insn.clone());
                // Step 6.1 Phase 2b: same as `emit_goto_if_not!` — the
                // specialised `int_is_zero` form is the pyre-port of
                // `flatten.py:247` `goto_if_not_int_is_zero`; Link
                // shape is identical.
                mergeblock(
                    code,
                    &mut graph,
                    &mut joinpoints,
                    &mut pc_blocks,
                    &current_block,
                    &{
                        let mut branch_state = current_state.clone();
                        branch_state.next_offset = py_pc;
                        branch_state.blocklist = frame_blocks_for_offset(code, py_pc);
                        branch_state
                    },
                    py_pc,
                    &mut link_exit_states,
                );
            }};
        }

        // B6 Phase 3b dual emission for the `*_vable_*` field/array
        // accessors. RPython parity: `jtransform.py:844`
        // `SpaceOperation('getfield_vable_%s' % kind, ...)` and
        // `jtransform.py:923` `SpaceOperation('setfield_vable_%s' % kind,
        // ...)` use the FULL kind name (`int` / `ref` / `float`);
        // `jtransform.py` `SpaceOperation('getarrayitem_vable_%s' %
        // kind[0], ...)` / `SpaceOperation('setarrayitem_vable_%s' %
        // kind[0], ...)` use the SHORT form (`i` / `r` / `f`).
        //
        // The runtime dispatch arms in `assembler.rs:486-524` were
        // already written against short-form field names; the SSA
        // emission here uses the RPython-parity full-form names and
        // relies on the Phase 3c dispatch aliases (added alongside)
        // to route the full-form opnames to the same builder methods.
        macro_rules! emit_vable_getfield_ref {
            ($ssarepr:expr, $dst:expr, $field_idx:expr) => {{
                let dst = $dst;
                let field_idx = $field_idx;
                let insn = Insn::op_with_result(
                    "getfield_vable_ref",
                    vec![Operand::ConstInt(field_idx as i64)],
                    Register::new(Kind::Ref, dst),
                );
                $ssarepr.insns.push(insn.clone());
            }};
        }
        macro_rules! emit_vable_setfield_int {
            ($ssarepr:expr, $field_idx:expr, $src:expr) => {{
                let field_idx = $field_idx;
                let src = $src;
                let insn = Insn::op(
                    "setfield_vable_int",
                    vec![
                        Operand::ConstInt(field_idx as i64),
                        Operand::reg(Kind::Int, src),
                    ],
                );
                $ssarepr.insns.push(insn.clone());
            }};
        }
        macro_rules! emit_vable_getarrayitem_ref {
            ($ssarepr:expr, $dst:expr, $field_idx:expr, $index:expr) => {{
                let dst = $dst;
                let field_idx = $field_idx;
                let index = $index;
                let insn = Insn::op_with_result(
                    "getarrayitem_vable_r",
                    vec![
                        Operand::ConstInt(field_idx as i64),
                        Operand::reg(Kind::Int, index),
                    ],
                    Register::new(Kind::Ref, dst),
                );
                $ssarepr.insns.push(insn.clone());
            }};
        }
        macro_rules! emit_vable_setarrayitem_ref {
            ($ssarepr:expr, $field_idx:expr, $index:expr, $src:expr) => {{
                let field_idx = $field_idx;
                let index = $index;
                let src = $src;
                let insn = Insn::op(
                    "setarrayitem_vable_r",
                    vec![
                        Operand::ConstInt(field_idx as i64),
                        Operand::reg(Kind::Int, index),
                        Operand::reg(Kind::Ref, src),
                    ],
                );
                $ssarepr.insns.push(insn.clone());
            }};
        }

        // RPython parity: `flatten.py:333`
        // `self.emitline('%s_copy' % kind, v, "->", w)` emits the
        // register-to-register move as `ref_copy` when `kind == 'ref'`;
        // `assembler.py:220` turns it into the bytecode key
        // `ref_copy/r>r`. The SSARepr arg list follows the upstream
        // `(src, '->', dst)` shape via `op_with_result`.
        macro_rules! emit_ref_copy {
            ($ssarepr:expr, $dst:expr, $src:expr) => {{
                let dst = $dst;
                let src = $src;
                let insn = Insn::op_with_result(
                    "ref_copy",
                    vec![Operand::reg(Kind::Ref, src)],
                    Register::new(Kind::Ref, dst),
                );
                $ssarepr.insns.push(insn.clone());
            }};
        }

        // `flatten.py:333-334` parity for `ref_copy` with a ConstRef source.
        // Used when opcode semantics push a real `None`, not the internal
        // CALL `NULL` sentinel.
        macro_rules! emit_ref_const_copy {
            ($ssarepr:expr, $dst:expr, $value:expr) => {{
                let dst = $dst;
                let value = $value;
                let insn = Insn::op_with_result(
                    "ref_copy",
                    vec![Operand::ConstRef(value)],
                    Register::new(Kind::Ref, dst),
                );
                $ssarepr.insns.push(insn.clone());
            }};
        }

        // pyframe.py:378-381 `pushvalue` lowers to
        // `setarrayitem_vable_r(locals_cells_stack_w, depth, w_object)`
        // + `setfield_vable_i(valuestackdepth, depth + 1)` via
        // jtransform.py:1898 `do_fixed_list_setitem` +
        // jtransform.py:920-928. RPython's optimizer folds the
        // per-push `setarrayitem_vable_r` via OptVirtualize so that the
        // compiled trace pays only the final force-vable cost. pyre's
        // OptVirtualize does not yet fold these at the same grain
        // (benchmark: enabling per-push mirror regresses nbody from
        // ~1.9s to ~22s). Emit only the SSA-stack `ref_copy` +
        // `setfield_vable_i` here until that fold lands; the parity
        // gap is the pyre side's optimizer, not this lowering.
        macro_rules! emit_pushvalue_ref {
            ($ssarepr:expr, $depth:ident, $src:expr) => {{
                let src_reg = $src;
                emit_ref_copy!($ssarepr, stack_base + $depth, src_reg);
                $depth += 1;
                emit_vsd!($depth);
            }};
        }

        // pyframe.py:411-417 `popvalue_maybe_none` lowers to
        // `setarrayitem_vable_r(locals_cells_stack_w, depth, None)` +
        // `setfield_vable_i(valuestackdepth, depth)`. The per-pop
        // null-clear mirror is folded by RPython OptVirtualize; pyre's
        // optimizer does not yet do that (same overhead pattern as
        // `emit_pushvalue_ref!`). Emit only the SSA-stack pop + depth
        // write here; the popped value stays available in the returned
        // SSA register for downstream uses.
        macro_rules! emit_popvalue_ref {
            ($ssarepr:expr, $depth:ident) => {{
                $depth -= 1;
                let popped_reg = stack_base + $depth;
                emit_vsd!($depth);
                popped_reg
            }};
        }

        // pyopcode.py:500-507 LOAD_FAST + pyframe.py:378-381 pushvalue.
        // Portal case lowers the local read to `getarrayitem_vable_r`
        // (jtransform.py:1877 `do_fixed_list_getitem`). The subsequent
        // pushvalue's `setarrayitem_vable_r` mirror would be required
        // per jtransform.py:1898, but RPython's OptVirtualize folds it
        // at trace time. pyre's optimizer does not — emitting the
        // mirror regresses nbody ~12x (see `emit_pushvalue_ref!`
        // comment). Emit the vable read + stack-slot ref_copy only; the
        // push mirror is deferred to the OptVirtualize port.
        macro_rules! emit_load_fast_ref {
            ($ssarepr:expr, $depth:ident, $reg:expr) => {{
                let reg = $reg;
                if is_portal {
                    emit_load_const_i!(
                        $ssarepr,
                        int_tmp0,
                        local_to_vable_slot(reg as usize) as i64
                    );
                    emit_vable_getarrayitem_ref!($ssarepr, stack_base + $depth, 0_u16, int_tmp0);
                    let loaded = current_state
                        .locals_w
                        .get(reg as usize)
                        .and_then(|value| value.clone())
                        .unwrap_or_else(|| fresh_ref_value(&mut graph));
                    current_state.stack.push(loaded);
                    $depth += 1;
                    emit_vsd!($depth);
                } else {
                    let loaded = current_state
                        .locals_w
                        .get(reg as usize)
                        .and_then(|value| value.clone())
                        .unwrap_or_else(|| fresh_ref_value(&mut graph));
                    current_state.stack.push(loaded);
                    emit_pushvalue_ref!($ssarepr, $depth, reg);
                }
            }};
        }

        // B6 Phase 3b dual emission for `jit_merge_point`. RPython parity:
        // `jtransform.py:rewrite_op_jit_merge_point` emits
        // `SpaceOperation('jit_merge_point', args, None)` with
        // ListOfKind-tagged greens/reds; `assembler.py:220` turns it into
        // `jit_merge_point/IRR` (int-list, ref-list, ref-list). pyre
        // emits this at the portal loop header.
        macro_rules! emit_jit_merge_point {
            ($ssarepr:expr, $greens_i:expr, $greens_r:expr, $reds_r:expr) => {{
                let greens_i: &[u8] = $greens_i;
                let greens_r: &[u8] = $greens_r;
                let reds_r: &[u8] = $reds_r;
                let to_list = |kind: Kind, regs: &[u8]| {
                    ListOfKind::new(
                        kind,
                        regs.iter()
                            .map(|&r| Operand::Register(Register::new(kind, r as u16)))
                            .collect(),
                    )
                };
                let insn = Insn::op(
                    "jit_merge_point",
                    vec![
                        Operand::ListOfKind(to_list(Kind::Int, greens_i)),
                        Operand::ListOfKind(to_list(Kind::Ref, greens_r)),
                        Operand::ListOfKind(to_list(Kind::Ref, reds_r)),
                    ],
                );
                $ssarepr.insns.push(insn.clone());
            }};
        }

        // Materialize the dedicated NULL sentinel register once so
        // bytecodes that need a literal null ref (`push_null`,
        // `RAISE_VARARGS argc=1`, call-site NULL sentinels) do not read
        // an uninitialized temp.
        emit_ref_const_copy!(ssarepr, null_ref_reg, pyre_object::PY_NULL as i64);

        for py_pc in 0..num_instrs {
            // Exception handler entry: Python resets stack depth to the
            // handler's specified depth and arrives only from
            // `catch_exception` edges, not from sequential fallthrough.
            if handler_depth_at.contains_key(&py_pc) {
                if let Some(handler_state) = handler_entry_state_from_catch_sites(
                    code,
                    &mut graph,
                    &catch_sites,
                    &catch_landing_blocks,
                    py_pc,
                ) {
                    current_depth = handler_state.stack.len() as u16;
                    current_state = handler_state;
                    needs_fallthrough = false;
                } else if let Some(&handler_depth) = handler_depth_at.get(&py_pc) {
                    current_depth = handler_depth;
                }
            }
            // RPython flatten.py: Label(block) at block entry
            emit_mark_label_pc!(ssarepr, py_pc);
            // pyre PRE-EXISTING-ADAPTATION (see `Insn::PcAnchor`
            // docstring in `flatten.rs`): emit a stable anchor at every
            // Python PC start so the post-compute_liveness /
            // post-remove_repeated_live SSARepr position is recoverable.
            ssarepr.insns.push(Insn::PcAnchor(py_pc));
            depth_at_pc[py_pc] = current_depth;
            emit_live_placeholder!(ssarepr);

            if loop_header_pcs.contains(&py_pc) {
                if merge_point_pc == Some(py_pc) {
                    // interp_jit.py:64 portal contract:
                    //   greens = ['next_instr', 'is_being_profiled', 'pycode']
                    //   reds = ['frame', 'ec']
                    //
                    // Graph side: record the upstream-matched 7-arg
                    // SpaceOperation per
                    // `jtransform.py:1690-1712 handle_jit_marker__jit_merge_point`.
                    // This parallels the SSARepr emission below so the
                    // graph layer carries the full `[jd_index, 3 green
                    // ListOfKinds, 3 red ListOfKinds]` shape that
                    // upstream regalloc + flatten consume.
                    //
                    // PRE-EXISTING-ADAPTATION: SSARepr/byte side emits
                    // pyre's native 3-list `(greens_i, greens_r, reds_r)`
                    // shape via `emit_portal_jit_merge_point`.  The
                    // assembler/blackhole/backend decoders (`assembler.rs:
                    // 681`) read that compressed shape AND the upstream-
                    // orthodox 7-arg shape (`assembler.rs` lowers the
                    // 7-arg form to the same 3-list byte protocol the
                    // runtime builder consumes).  pycode is carried as
                    // an `Opaque(Ref)` Constant at the graph layer so
                    // `Constant` Eq/Hash stays Clone + PartialEq-deriveable
                    // (raw pointers cannot hash cleanly); the
                    // `lower_constant` callback below recovers the
                    // original `w_code` pointer from its closure capture
                    // and routes it through the runtime constant pool.
                    let w_code_i64 = w_code as i64;
                    let (frame_var, ec_var) = portal_graph_inputvars(code);
                    let graph_args =
                        portal_jit_merge_point_graph_args(&graph, py_pc, w_code as *const ());
                    let graph_op = record_graph_op(
                        &current_block.block(),
                        "jit_merge_point",
                        graph_args,
                        None,
                        py_pc as i64,
                    );
                    GraphFlattener::new_with_constant_lowering(
                        &mut ssarepr,
                        |v: super::flow::Variable| {
                            if v.id == frame_var.id {
                                Register::new(Kind::Ref, portal_frame_reg)
                            } else if v.id == ec_var.id {
                                Register::new(Kind::Ref, portal_ec_reg)
                            } else {
                                panic!(
                                    "portal jit_merge_point: unexpected graph Variable {v:?} \
                                     (only portal frame/ec expected)"
                                )
                            }
                        },
                        |c: &super::flow::Constant| match (&c.value, c.kind) {
                            (super::flow::ConstantValue::Signed(value), Some(Kind::Int)) => {
                                Operand::ConstInt(*value)
                            }
                            (super::flow::ConstantValue::Opaque(_), Some(Kind::Ref)) => {
                                // pycode ref — real pointer recovered
                                // from closure capture above.  The
                                // assembler's `expect_list_regs_or_pool`
                                // routes `Operand::ConstRef` through
                                // `builder.add_const_r`.
                                Operand::ConstRef(w_code_i64)
                            }
                            other => {
                                panic!("portal jit_merge_point: unexpected Constant {other:?}")
                            }
                        },
                    )
                    .emit_space_operation(&graph_op);
                } else {
                    // pyre has a single jitdriver (PyPyJitDriver), index 0.
                    let loop_header_op = record_graph_op(
                        &current_block.block(),
                        "loop_header",
                        vec![super::flow::Constant::signed(0).into()],
                        None,
                        py_pc as i64,
                    );
                    GraphFlattener::new(&mut ssarepr, |_variable| {
                        unreachable!("loop_header graph op does not carry Variables")
                    })
                    .emit_space_operation(&loop_header_op);
                }
            }

            let code_unit = code.instructions[py_pc];
            let (instruction, op_arg) = arg_state.get(code_unit);

            // pyframe.py:379-417 pushvalue/popvalue_maybe_none parity:
            // RPython's push/pop each write `self.valuestackdepth = depth +/- 1`.
            // On the JIT, these map to per-push `setfield_vable_i`. pyre's
            // codewriter stores stack values in typed registers rather than
            // the `locals_cells_stack_w` array, so we cannot emit a vable
            // setitem for each push. As the coarsest RPython-compatible
            // approximation we flush `valuestackdepth` once at opcode entry,
            // reflecting the pre-opcode stack depth — which is what the
            // interpreter (eval.rs:92 `target_depth = frame.nlocals() +
            // frame.ncells() + entry.depth`) uses when an exception handler
            // unwinds the frame.
            //
            // RPython interp_jit.py keeps `next_instr` as a green portal
            // argument and updates `last_instr` in the interpreter loop; it
            // does not lower a per-bytecode virtualizable write here. pyre's
            // portal entry / guard-resume paths already restore
            // `frame.next_instr`, and the interpreter updates `last_instr`
            // once execution returns there. Emitting `py_pc + 1` here only
            // grows the int constant pool linearly with function size and
            // trips assembler.py's 256-entry cap.
            // pyframe.py:379-417: valuestackdepth is written per-push/per-pop
            // via setfield_vable_i (jtransform.py:923-928), NOT once at opcode
            // entry. The per-push/per-pop emit_vsd! calls below mirror that.
            // (The old single-entry flush is removed.)

            // RPython jtransform.py: rewrite_operation() dispatches per opname.
            // Each match arm is the pyre equivalent of rewrite_op_*.
            match instruction {
                Instruction::Resume { .. }
                | Instruction::Nop
                | Instruction::Cache
                | Instruction::NotTaken
                | Instruction::ExtendedArg => {
                    // RPython: no-op operations produce no jitcode output
                }

                // jtransform.py:1877 do_fixed_list_getitem vable case:
                // portal locals are virtualizable array items — emit
                // vable_getarrayitem_ref so the optimizer folds the read
                // against virtualizable_boxes and the blackhole pulls the
                // live frame value into stack_base+current_depth on
                // resume. Non-portal frames keep ref_copy (no virtualizable
                // in scope).
                Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    emit_load_fast_ref!(ssarepr, current_depth, reg);
                }

                // jtransform.py:1898 do_fixed_list_setitem vable case:
                // Portal frames treat `locals_cells_stack_w` as the sole
                // storage for locals — setarrayitem_vable_r writes from
                // the value-stack slot directly, so no register-per-local
                // shadow exists. Non-portal frames keep ref_copy (no vable
                // in scope).
                Instruction::StoreFast { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    let stored_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    let stored = current_state
                        .stack
                        .pop()
                        .unwrap_or_else(|| fresh_ref_value(&mut graph));
                    if is_portal {
                        emit_load_const_i!(
                            ssarepr,
                            int_tmp0,
                            local_to_vable_slot(reg as usize) as i64
                        );
                        emit_vable_setarrayitem_ref!(ssarepr, 0_u16, int_tmp0, stored_reg);
                    }
                    // D.vable Phase 1 (2026-04-23): mirror the stored value
                    // into reg_N so super-inst consumers
                    // (LoadFastLoadFast / LoadFastBorrowLoadFastBorrow)
                    // which read reg_N directly see the post-STORE_FAST
                    // value. Portal frames previously skipped this step,
                    // leaving reg_N stale after vable_setarrayitem_ref.
                    // Establishing the reg==vable invariant here is the
                    // foundation for the eventual LFLF vable flip — see
                    // memo super_inst_candidate1_probe_scope_2026_04_23.
                    emit_ref_copy!(ssarepr, reg, stored_reg);
                    if let Some(slot) = current_state.locals_w.get_mut(reg as usize) {
                        *slot = Some(stored);
                    }
                }

                Instruction::LoadSmallInt { i } => {
                    let val = i.get(op_arg) as u32 as i64;
                    emit_load_const_i!(ssarepr, int_tmp0, val);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::Plain,
                        box_int_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    current_state
                        .stack
                        .push(super::flow::Constant::signed(val).into());
                    emit_pushvalue_ref!(ssarepr, current_depth, obj_tmp0);
                }

                Instruction::LoadConst { consti } => {
                    let idx = consti.get(op_arg).as_usize();
                    // jtransform.py: getfield_vable_r for pycode (field 1)
                    emit_vable_getfield_ref!(ssarepr, obj_tmp0, VABLE_CODE_FIELD_IDX);
                    emit_load_const_i!(ssarepr, int_tmp0, idx as i64);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::Plain,
                        load_const_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    let value = match code.constants.get(idx) {
                        Some(pyre_interpreter::bytecode::ConstantData::None) => {
                            super::flow::Constant::none().into()
                        }
                        Some(pyre_interpreter::bytecode::ConstantData::Boolean { value }) => {
                            super::flow::Constant::bool(*value).into()
                        }
                        Some(pyre_interpreter::bytecode::ConstantData::Integer { value }) => {
                            use num_traits::ToPrimitive;
                            value
                                .to_i64()
                                .map(|value| super::flow::Constant::signed(value).into())
                                .unwrap_or_else(|| fresh_ref_value(&mut graph))
                        }
                        _ => fresh_ref_value(&mut graph),
                    };
                    current_state.stack.push(value);
                    emit_pushvalue_ref!(ssarepr, current_depth, obj_tmp0);
                }

                // CPython 3.13 super-instructions LOAD_FAST_LOAD_FAST /
                // LOAD_FAST_BORROW_LOAD_FAST_BORROW decompose to two plain
                // LOAD_FAST reads. Portal parity with plain LoadFast would
                // route both halves through vable_getarrayitem_ref, but
                // flipping this arm (attempted 2026-04-19 after P1 Step 1
                // + P3 seed helper + unroll reserve_pos refactor
                // `66d1f7212d`) still breaks spectral_norm/nbody/fannkuch
                // on both backends. The heap-vs-symbolic-state gap
                // persists — the compiled loop's vable reads still see
                // stale slots that the bridge / blackhole resume path
                // has not re-synchronized. Keep ref_copy here until the
                // liveness pipeline rework (Priority 4) lands; the full
                // `flatten → compute_liveness(ssarepr) → assemble`
                // sequence should close the gap by ensuring the vable
                // mirror is write-back-consistent before each compiled
                // entry.
                Instruction::LoadFastBorrowLoadFastBorrow { var_nums }
                | Instruction::LoadFastLoadFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let reg_a = u32::from(pair.idx_1()) as u16;
                    let reg_b = u32::from(pair.idx_2()) as u16;
                    emit_ref_copy!(ssarepr, stack_base + current_depth, reg_a);
                    current_state.stack.push(
                        current_state
                            .locals_w
                            .get(reg_a as usize)
                            .and_then(|value| value.clone())
                            .unwrap_or_else(|| fresh_ref_value(&mut graph)),
                    );
                    current_depth += 1;
                    emit_vsd!(current_depth);
                    emit_ref_copy!(ssarepr, stack_base + current_depth, reg_b);
                    current_state.stack.push(
                        current_state
                            .locals_w
                            .get(reg_b as usize)
                            .and_then(|value| value.clone())
                            .unwrap_or_else(|| fresh_ref_value(&mut graph)),
                    );
                    current_depth += 1;
                    emit_vsd!(current_depth);
                }

                // Super-instruction STORE_FAST; LOAD_FAST: pop TOS into
                // idx_1 (store), then push idx_2 (load). Net depth 0.
                // Portal: store via setarrayitem_vable_r, load via
                // getarrayitem_vable_r. Non-portal: ref_copy for both halves.
                Instruction::StoreFastLoadFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let store_reg = u32::from(pair.idx_1()) as u16;
                    let load_reg = u32::from(pair.idx_2()) as u16;
                    let stored_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    let stored = current_state
                        .stack
                        .pop()
                        .unwrap_or_else(|| fresh_ref_value(&mut graph));
                    if is_portal {
                        // STORE_FAST half: popvalue + locals_cells_stack_w[local_slot] = w
                        emit_load_const_i!(
                            ssarepr,
                            int_tmp0,
                            local_to_vable_slot(store_reg as usize) as i64
                        );
                        emit_vable_setarrayitem_ref!(ssarepr, 0_u16, int_tmp0, stored_reg);
                        // D.vable Phase 1: mirror STORE half into reg_N for
                        // LFLF consumers. See Instruction::StoreFast arm.
                        emit_ref_copy!(ssarepr, store_reg, stored_reg);
                        // LOAD_FAST half: read local, then pyframe.py:378-381
                        // pushvalue parity — mirror to the value-stack slot.
                        emit_load_const_i!(
                            ssarepr,
                            int_tmp0,
                            local_to_vable_slot(load_reg as usize) as i64
                        );
                        emit_vable_getarrayitem_ref!(
                            ssarepr,
                            stack_base + current_depth,
                            0_u16,
                            int_tmp0
                        );
                        emit_load_const_i!(
                            ssarepr,
                            int_tmp0,
                            (stack_base_absolute + current_depth as usize) as i64
                        );
                        emit_vable_setarrayitem_ref!(
                            ssarepr,
                            0_u16,
                            int_tmp0,
                            stack_base + current_depth
                        );
                        if let Some(slot) = current_state.locals_w.get_mut(store_reg as usize) {
                            *slot = Some(stored);
                        }
                        let loaded = current_state
                            .locals_w
                            .get(load_reg as usize)
                            .and_then(|value| value.clone())
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        current_state.stack.push(loaded);
                        current_depth += 1;
                        emit_vsd!(current_depth);
                    } else {
                        emit_ref_copy!(ssarepr, store_reg, stack_base + current_depth);
                        if let Some(slot) = current_state.locals_w.get_mut(store_reg as usize) {
                            *slot = Some(stored);
                        }
                        let loaded = current_state
                            .locals_w
                            .get(load_reg as usize)
                            .and_then(|value| value.clone())
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        current_state.stack.push(loaded);
                        emit_pushvalue_ref!(ssarepr, current_depth, load_reg);
                    }
                }

                // STORE_SUBSCR: stack [value, obj, key] → obj[key] = value
                Instruction::StoreSubscr => {
                    let key_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    emit_ref_copy!(ssarepr, obj_tmp1, key_reg);
                    let obj_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    emit_ref_copy!(ssarepr, obj_tmp0, obj_reg);
                    let value_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    emit_ref_copy!(ssarepr, arg_regs_start, value_reg);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::MayForce,
                        store_subscr_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::reference(arg_regs_start),
                        ],
                        ResKind::Void,
                        None,
                    );
                }

                Instruction::PopTop => {
                    let _ = emit_popvalue_ref!(ssarepr, current_depth);
                    // regalloc.py: discard = just decrement depth, no bytecode
                }

                Instruction::PushNull => {
                    emit_pushvalue_ref!(ssarepr, current_depth, null_ref_reg);
                }

                // jtransform.py: rewrite_op_int_add etc.
                //
                // Call reads stack slots DIRECTLY rather than copying through
                // obj_tmp0/obj_tmp1 temps. This keeps the call's argument
                // registers inside the trace-tracked range (`registers_r`
                // + `symbolic_stack`), so guards fired across the op (e.g.
                // `GUARD_NOT_FORCED_2` after a helper call) capture the
                // lhs/rhs values in fail_args. See
                // `memory/pyre_trace_temp_reg_tracking_gap_2026_04_19.md`.
                Instruction::BinaryOp { op } => {
                    let op_val = binary_op_tag(op.get(op_arg))
                        .expect("unsupported binary op tag in jitcode lowering")
                        as u32;
                    // Pop rhs (blackhole will see vsd reflect this pop).
                    let rhs_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    // Pop lhs.
                    let lhs_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    emit_load_const_i!(ssarepr, op_code_reg, op_val as i64);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::MayForce,
                        binary_op_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(lhs_reg),
                            majit_metainterp::jitcode::JitCallArg::reference(rhs_reg),
                            majit_metainterp::jitcode::JitCallArg::int(op_code_reg),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    emit_pushvalue_ref!(ssarepr, current_depth, obj_tmp0);
                }

                // jtransform.py: rewrite_op_int_lt, optimize_goto_if_not
                Instruction::CompareOp { opname } => {
                    // Same stack-direct pattern as BinaryOp — see its comment.
                    let op_val = compare_op_tag(opname.get(op_arg)) as u32;
                    let rhs_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    let lhs_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    emit_load_const_i!(ssarepr, op_code_reg, op_val as i64);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::MayForce,
                        compare_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(lhs_reg),
                            majit_metainterp::jitcode::JitCallArg::reference(rhs_reg),
                            majit_metainterp::jitcode::JitCallArg::int(op_code_reg),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    emit_pushvalue_ref!(ssarepr, current_depth, obj_tmp0);
                }

                // flatten.py:240-260 + blackhole.py:865-869. truth_fn returns
                // a bool-as-int; emit plain `goto_if_not <bool> L` — the
                // unfused form flatten.py takes when the exitswitch is a
                // plain variable (not a tuple of a foldable comparison op).
                // bhimpl_goto_if_not takes the target when `a == 0`.
                Instruction::PopJumpIfFalse { delta } => {
                    let target_py_pc = jump_target_forward(
                        code,
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    let value_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    emit_ref_copy!(ssarepr, obj_tmp0, value_reg);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::Plain,
                        truth_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0)],
                        ResKind::Int,
                        Some(int_tmp0),
                    );
                    if target_py_pc < num_instrs {
                        emit_goto_if_not!(ssarepr, int_tmp0, target_py_pc);
                    }
                }

                // jtransform.py:1212 `_rewrite_equality` rewrites
                // `int_eq(x, 0)` → `int_is_zero(x)`; flatten.py:247
                // specialises into `goto_if_not_int_is_zero <v> L`
                // (blackhole.py:916-920). Polarity is inverted vs
                // PopJumpIfFalse: target taken iff the truth-fn result
                // is non-zero (truthy).
                Instruction::PopJumpIfTrue { delta } => {
                    let target_py_pc = jump_target_forward(
                        code,
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    let value_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    emit_ref_copy!(ssarepr, obj_tmp0, value_reg);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::Plain,
                        truth_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0)],
                        ResKind::Int,
                        Some(int_tmp0),
                    );
                    if target_py_pc < num_instrs {
                        emit_goto_if_not_int_is_zero!(ssarepr, int_tmp0, target_py_pc);
                    }
                }

                // RPython flatten.py: goto Label
                Instruction::JumpForward { delta } => {
                    let target_py_pc = jump_target_forward(
                        code,
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    if target_py_pc < num_instrs {
                        emit_goto!(ssarepr, target_py_pc);
                    }
                }

                instr @ Instruction::JumpBackward { .. } => {
                    if let Some(target_py_pc) = backward_jump_target(code, py_pc, instr, op_arg) {
                        if target_py_pc < num_instrs {
                            emit_goto!(ssarepr, target_py_pc);
                        }
                    }
                }

                // flatten.py: int_return / ref_return
                Instruction::ReturnValue => {
                    let retval_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    let retval = current_state
                        .stack
                        .pop()
                        .unwrap_or_else(|| fresh_ref_value(&mut graph));
                    emit_ref_copy!(ssarepr, obj_tmp0, retval_reg);
                    emit_ref_return!(ssarepr, obj_tmp0, retval);
                }

                // RPython jtransform.py: rewrite_op_direct_call (residual)
                Instruction::LoadGlobal { namei } => {
                    let raw_namei = namei.get(op_arg) as usize as i64;
                    // jtransform.py: getfield_vable_r for w_globals (field 3)
                    // and pycode (field 1) — namespace for lookup, code for names.
                    emit_vable_getfield_ref!(ssarepr, obj_tmp0, VABLE_NAMESPACE_FIELD_IDX);
                    emit_vable_getfield_ref!(ssarepr, obj_tmp1, VABLE_CODE_FIELD_IDX);
                    emit_load_const_i!(ssarepr, int_tmp0, raw_namei);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::MayForce,
                        load_global_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    // LOAD_GLOBAL with (namei >> 1) & 1: push NULL first
                    if raw_namei & 1 != 0 {
                        emit_pushvalue_ref!(ssarepr, current_depth, null_ref_reg);
                    }
                    emit_pushvalue_ref!(ssarepr, current_depth, obj_tmp0);
                }

                // RPython jtransform.py: rewrite_op_direct_call →
                // call_may_force / residual_call
                //
                // RPython blackhole.py: bhimpl_recursive_call_i calls
                // portal_runner directly, bypassing JIT entry.
                // Here we pop args and callable from the stack into
                // registers, then call the helper with explicit args.
                //
                // shared_opcode.rs:56 opcode_call parity:
                // Stack layout before CALL(argc):
                //   [callable, null_or_self, arg0, ..., arg(argc-1)]
                // Pop in reverse: args, null_or_self, callable.
                Instruction::Call { argc } => {
                    let nargs = argc.get(op_arg) as usize;
                    for i in (0..nargs).rev() {
                        let arg_reg = emit_popvalue_ref!(ssarepr, current_depth);
                        emit_ref_copy!(ssarepr, arg_regs_start + i as u16, arg_reg);
                    }
                    let callable_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    emit_ref_copy!(ssarepr, obj_tmp1, callable_reg);
                    let _ = emit_popvalue_ref!(ssarepr, current_depth); // NULL (discard)

                    // RPython: bhimpl_recursive_call_i(jdindex, greens, reds)
                    // call_fn(callable, arg0, ...) → result
                    // Parent frame accessed via BH_VABLE_PTR thread-local.
                    let mut call_args =
                        vec![majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1)];
                    for i in 0..nargs {
                        call_args.push(majit_metainterp::jitcode::JitCallArg::reference(
                            arg_regs_start + i as u16,
                        ));
                    }
                    // Select the correct arity-specific call helper.
                    // RPython blackhole.py: call_int_function transmutes
                    // to the correct arity. Each nargs needs a matching
                    // extern "C" fn with that many i64 parameters.
                    // nargs > 8 → abort_permanent (no matching helper).
                    if nargs > 8 {
                        emit_abort_permanent!(ssarepr);
                    } else {
                        let fn_idx = match nargs {
                            0 => call_fn_0_idx,
                            1 => call_fn_idx,
                            2 => call_fn_2_idx,
                            3 => call_fn_3_idx,
                            4 => call_fn_4_idx,
                            5 => call_fn_5_idx,
                            6 => call_fn_6_idx,
                            7 => call_fn_7_idx,
                            _ => call_fn_8_idx,
                        };
                        emit_residual_call(
                            &mut ssarepr,
                            &mut graph,
                            &current_block.block(),
                            CallFlavor::MayForce,
                            fn_idx,
                            &call_args,
                            ResKind::Ref,
                            Some(obj_tmp0),
                        );
                    }
                    emit_pushvalue_ref!(ssarepr, current_depth, obj_tmp0);
                }

                // Python 3.13: ToBool converts TOS to bool before branch.
                // No-op in JitCode: the value is already truthy/falsy and
                // the following PopJumpIfFalse guards on it.
                Instruction::ToBool => {}

                // RPython bhimpl_int_neg: -obj via binary_op(0, obj, NB_SUBTRACT)
                Instruction::UnaryNegative => {
                    let value_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    emit_ref_copy!(ssarepr, obj_tmp0, value_reg);
                    emit_load_const_i!(ssarepr, int_tmp0, 0);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::MayForce,
                        box_int_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                        ResKind::Ref,
                        Some(obj_tmp1),
                    );
                    emit_load_const_i!(
                        ssarepr,
                        int_tmp0,
                        binary_op_tag(pyre_interpreter::bytecode::BinaryOperator::Subtract)
                            .expect("subtract must have a jit binary-op tag"),
                    );
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::MayForce,
                        binary_op_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    emit_pushvalue_ref!(ssarepr, current_depth, obj_tmp0);
                }

                // JumpBackwardNoInterrupt reuses `backward_jump_target`:
                // the encoding differs from JumpBackward (no skip_caches
                // on the next-PC base) but the helper routes each variant
                // to its correct arithmetic so pre-scan and emit stay in
                // lockstep.  interp_jit.py:103 + jtransform.py:1714.
                instr @ Instruction::JumpBackwardNoInterrupt { .. } => {
                    if let Some(target_py_pc) = backward_jump_target(code, py_pc, instr, op_arg) {
                        if target_py_pc < num_instrs {
                            emit_goto!(ssarepr, target_py_pc);
                        }
                    }
                }

                // RPython bhimpl_newlist: build list from N items on stack.
                Instruction::BuildList { count } => {
                    let argc = count.get(op_arg) as usize;
                    for i in (0..argc.min(2)).rev() {
                        let item_reg = emit_popvalue_ref!(ssarepr, current_depth);
                        emit_ref_copy!(ssarepr, arg_regs_start + i as u16, item_reg);
                    }
                    // Discard extra items beyond 2 (helper supports 0-2).
                    for _ in 2..argc {
                        let _ = emit_popvalue_ref!(ssarepr, current_depth);
                    }
                    // build_list_fn(argc, item0, item1) → list
                    emit_load_const_i!(ssarepr, int_tmp0, argc as i64);
                    let item0 = if argc >= 1 {
                        majit_metainterp::jitcode::JitCallArg::reference(arg_regs_start)
                    } else {
                        majit_metainterp::jitcode::JitCallArg::int(int_tmp0) // dummy
                    };
                    let item1 = if argc >= 2 {
                        majit_metainterp::jitcode::JitCallArg::reference(arg_regs_start + 1)
                    } else {
                        majit_metainterp::jitcode::JitCallArg::int(int_tmp0) // dummy
                    };
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::MayForce,
                        build_list_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                            item0,
                            item1,
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    emit_pushvalue_ref!(ssarepr, current_depth, obj_tmp0);
                }

                // Exception handling: residual calls to frame helpers.
                // RPython blackhole.py handles exceptions via dedicated
                // bhimpl_* functions. In pyre, we delegate to the frame's
                // exception machinery via call_fn.
                Instruction::RaiseVarargs { argc } => {
                    let n = argc.get(op_arg) as i64;
                    if n >= 1 {
                        // Pop the cause operand (argc=2 only) before the
                        // exception — RAISE_VARARGS stack order is
                        // [exc, cause] with cause on top (eval.rs:1091).
                        // The cause FlowValue is discarded: the
                        // exceptblock link in the shadow graph carries
                        // the exception value, not the cause.
                        let cause = if n >= 2 {
                            let _cause_fv = current_state
                                .stack
                                .pop()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph));
                            let cause_reg = emit_popvalue_ref!(ssarepr, current_depth);
                            emit_ref_copy!(ssarepr, obj_tmp1, cause_reg);
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1)
                        } else {
                            majit_metainterp::jitcode::JitCallArg::reference(null_ref_reg)
                        };
                        // Drop the original exception operand from the
                        // shadow stack. The residual helper below may
                        // normalize `raise SomeExcClass` into a fresh
                        // exception instance, so the exception edge must
                        // carry a NEW FlowValue representing the helper
                        // result, not the pre-normalization operand.
                        let _ = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        // pyframe.py:411-417 popvalue_maybe_none parity:
                        // clears locals_cells_stack_w[depth] on pop in
                        // portal frames so vable mirror stays in sync.
                        let exc_reg = emit_popvalue_ref!(ssarepr, current_depth);
                        emit_ref_copy!(ssarepr, obj_tmp0, exc_reg);
                        // pyopcode.py:711 / eval.rs:1073,1112 parity:
                        // residual call normalizes class / builtin-
                        // callable exceptions to instances and attaches
                        // `__cause__` before the raise edge.
                        emit_residual_call(
                            &mut ssarepr,
                            &mut graph,
                            &current_block.block(),
                            CallFlavor::Plain,
                            normalize_raise_varargs_fn_idx,
                            &[
                                majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                                cause,
                            ],
                            ResKind::Ref,
                            Some(obj_tmp0),
                        );
                        let normalized_exc_fv = fresh_ref_value(&mut graph);
                        emit_raise!(ssarepr, obj_tmp0, normalized_exc_fv, py_pc as i64);
                    } else {
                        // reraise: re-raise exception_last_value
                        emit_reraise!(ssarepr);
                    }
                }

                Instruction::PushExcInfo => {
                    // eval.rs:1220-1229 / pyopcode.py:786 parity:
                    //   exc  = pop()
                    //   prev = CURRENT_EXCEPTION
                    //   CURRENT_EXCEPTION = exc
                    //   push(prev)
                    //   push(exc)
                    //
                    // The JIT emits two residual helper calls so the
                    // traced code reads and writes the same per-thread
                    // slot as the interpreter. `push(prev)` used to be
                    // `push(None)`, which breaks nested exception state
                    // (pyopcode.py:786 saves the previous sys_exc_info
                    // so `POP_EXCEPT` can restore it).
                    let exc_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    let exc_value = current_state
                        .stack
                        .pop()
                        .unwrap_or_else(|| fresh_ref_value(&mut graph));
                    emit_ref_copy!(ssarepr, obj_tmp0, exc_reg);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::Plain,
                        get_current_exception_fn_idx,
                        &[],
                        ResKind::Ref,
                        Some(obj_tmp1),
                    );
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::Plain,
                        set_current_exception_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0)],
                        ResKind::Void,
                        None,
                    );
                    let prev_value = fresh_ref_value(&mut graph);
                    current_state.stack.push(prev_value);
                    emit_pushvalue_ref!(ssarepr, current_depth, obj_tmp1);
                    current_state.stack.push(exc_value);
                    emit_pushvalue_ref!(ssarepr, current_depth, obj_tmp0);
                }

                Instruction::CheckExcMatch => {
                    let match_type_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    let _ = current_state.stack.pop();
                    emit_ref_copy!(ssarepr, obj_tmp1, match_type_reg); // match type
                    emit_ref_copy!(ssarepr, obj_tmp0, stack_base + current_depth - 1); // exception
                    // isinstance check via compare_fn(exc, type, ISINSTANCE_OP)
                    emit_load_const_i!(ssarepr, int_tmp0, 10); // isinstance op
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::MayForce,
                        compare_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    current_state.stack.push(fresh_ref_value(&mut graph));
                    emit_pushvalue_ref!(ssarepr, current_depth, obj_tmp0);
                }

                Instruction::PopExcept => {
                    // eval.rs:1243-1249 / pyopcode.py:778 parity:
                    //   prev = pop()
                    //   CURRENT_EXCEPTION = prev
                    //
                    // Previously the arm just popped and left TLS stale,
                    // which silently broke nested `except` blocks: after
                    // `POP_EXCEPT` the outer handler's exception must be
                    // reinstated as the "current" one so a bare `raise`
                    // re-propagates it.
                    let prev_reg = emit_popvalue_ref!(ssarepr, current_depth);
                    let _ = current_state.stack.pop();
                    emit_ref_copy!(ssarepr, obj_tmp0, prev_reg);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut graph,
                        &current_block.block(),
                        CallFlavor::Plain,
                        set_current_exception_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0)],
                        ResKind::Void,
                        None,
                    );
                    current_state.last_exception = None;
                }

                Instruction::Reraise { .. } => {
                    // Exception path: abort_permanent.
                    emit_abort_permanent!(ssarepr);
                }

                Instruction::Copy { i } => {
                    let d = i.get(op_arg) as usize;
                    if d == 1 {
                        current_state.stack.push(
                            current_state
                                .stack
                                .last()
                                .cloned()
                                .unwrap_or_else(|| fresh_ref_value(&mut graph)),
                        );
                        emit_pushvalue_ref!(ssarepr, current_depth, stack_base + current_depth - 1);
                    } else {
                        // COPY(d>1): exception handler pattern only.
                        // Use abort_permanent (BC_ABORT_PERMANENT=14) so it
                        // doesn't trigger the has_abort(BC_ABORT=13) check.
                        emit_abort_permanent!(ssarepr);
                    }
                }

                // Stack-effect-aware abort_permanent for unsupported ops.
                // current_depth must track interpreter parity so that
                // subsequent CALL handlers don't underflow.
                Instruction::LoadName { .. } => {
                    // PyPy assemble.py: LOAD_NAME has stack effect +1.
                    emit_abort_permanent!(ssarepr);
                    current_depth += 1;
                }
                Instruction::StoreName { .. } => {
                    // PyPy assemble.py: STORE_NAME has stack effect -1.
                    emit_abort_permanent!(ssarepr);
                    current_depth = current_depth.saturating_sub(1);
                }
                Instruction::MakeFunction { .. } => {
                    // Module-level only: abort_permanent (won't block blackhole).
                    emit_abort_permanent!(ssarepr);
                }
                Instruction::LoadAttr { namei } => {
                    // PyPy assemble.py gives LOAD_ATTR a net-0 stack effect.
                    // pyre's CPython-3.13 method form pushes an extra
                    // null/self sentinel, so keep current_depth in sync.
                    let attr = namei.get(op_arg);
                    emit_abort_permanent!(ssarepr);
                    if attr.is_method() {
                        current_depth += 1;
                    }
                }

                // CPython 3.13 superinstruction: STORE_FAST_STORE_FAST.
                // jtransform.py:1898 — each local write → setarrayitem_vable_r
                // in portal, ref_copy in non-portal. Mirrors plain StoreFast.
                Instruction::StoreFastStoreFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let reg_a = u32::from(pair.idx_1()) as u16;
                    let reg_b = u32::from(pair.idx_2()) as u16;
                    for reg in [reg_a, reg_b] {
                        let stored_reg = emit_popvalue_ref!(ssarepr, current_depth);
                        let stored = current_state
                            .stack
                            .pop()
                            .unwrap_or_else(|| fresh_ref_value(&mut graph));
                        if is_portal {
                            emit_load_const_i!(
                                ssarepr,
                                int_tmp0,
                                local_to_vable_slot(reg as usize) as i64,
                            );
                            emit_vable_setarrayitem_ref!(ssarepr, 0_u16, int_tmp0, stored_reg);
                        }
                        // D.vable Phase 1: mirror each store into reg_N for
                        // LFLF consumers. See Instruction::StoreFast arm.
                        emit_ref_copy!(ssarepr, reg, stored_reg);
                        if let Some(slot) = current_state.locals_w.get_mut(reg as usize) {
                            *slot = Some(stored);
                        }
                    }
                }

                // CPython 3.13 UNPACK_SEQUENCE: pop 1 (seq), push `count`.
                // Emit abort_permanent (no getitem helper yet) but
                // adjust current_depth so subsequent instructions don't
                // underflow.
                Instruction::UnpackSequence { count } => {
                    let n = count.get(op_arg) as u16;
                    emit_abort_permanent!(ssarepr);
                    // Stack effect: pop 1 + push n = net (n - 1)
                    if current_depth > 0 {
                        current_depth -= 1;
                        emit_vsd!(current_depth);
                    }
                    current_depth += n;
                }

                // CPython 3.13 iterator protocol — emit abort_permanent
                // with correct depth tracking so subsequent instructions
                // don't underflow.
                Instruction::GetIter => {
                    // pop iterable, push iterator: net 0
                    emit_abort_permanent!(ssarepr);
                }

                Instruction::ForIter { .. } => {
                    // push next item: net +1
                    emit_abort_permanent!(ssarepr);
                    current_depth += 1;
                    emit_vsd!(current_depth);
                }

                Instruction::EndFor => {
                    // pop iterator + last value: net -2
                    emit_abort_permanent!(ssarepr);
                    current_depth = current_depth.saturating_sub(2);
                    // No emit_vsd: after abort_permanent, depth is
                    // simulation-only for subsequent compile-time tracking.
                }

                Instruction::PopIter => {
                    // pop iterator: net -1
                    current_depth = current_depth.saturating_sub(1);
                    emit_vsd!(current_depth);
                }

                // Unsupported instruction: abort_permanent.
                // BC_ABORT_PERMANENT(14) so has_abort_opcode doesn't
                // false-positive on functions with only module-level paths.
                _other => {
                    emit_abort_permanent!(ssarepr);
                }
            }
            sync_stack_state(&mut graph, &mut current_state, current_depth);
            current_state.next_offset = py_pc + 1;
            current_state.blocklist = frame_blocks_for_offset(code, current_state.next_offset);
            if let Some(catch_label) = catch_for_pc[py_pc] {
                emit_catch_exception!(ssarepr, catch_label);
            }
        }

        // RPython flatten.py parity: every code path ends with an explicit
        // return/raise/goto/unreachable. No end-of-code sentinel needed —
        // falling off the end is unreachable if all bytecodes are covered.

        // pyre-only PyJitCode.has_abort: a "this jitcode cannot be
        // blackhole-dispatched, pipe straight to the interpreter" flag.
        // RPython has no such flag (rpython/jit/codewriter/jitcode.py:14
        // — no abort tracking on JitCode). Upstream's `Assembler.abort()`
        // (assembler.py:177-181, bhimpl_abort) emits BC_ABORT so the
        // blackhole raises SwitchToBlackhole(ABORT_ESCAPE) at runtime;
        // `abort_permanent()` is a different pyre-only bytecode we emit
        // for genuinely unsupported Python opcodes, and its execution
        // path already raises/aborts correctly from the blackhole. We
        // keep has_abort narrowly scoped to `abort()` emissions (matches
        // the JitCodeBuilder flag shape) so the flag's meaning doesn't
        // drift into "assembler overflow" or "abort_permanent present"
        // — both of which the assembler/blackhole already handle without
        // a front-end gate.
        let mut has_abort = assembler.has_abort_flag();

        for site in catch_sites {
            emit_mark_label_catch_landing!(ssarepr, site.landing_label);
            // eval.rs:150-168 handle_exception parity:
            // the handler edge enters with the protected prefix of the
            // value stack preserved, then `push_lasti` (if any), then the
            // exception value. `emit_goto!(handler_py_pc)` snapshots
            // `current_state`, so mirror the same stack shape here before
            // linking the landing block to the handler block.
            sync_stack_state(&mut graph, &mut current_state, site.stack_depth);
            if site.push_lasti {
                current_state.stack.push(fresh_ref_value(&mut graph));
            }
            let exc_value = current_state
                .last_exception
                .as_ref()
                .map(|(_w_type, w_value)| w_value.clone())
                .unwrap_or_else(|| fresh_ref_value(&mut graph));
            current_state.stack.push(exc_value);
            // pyframe.py:503-510 + eval.rs:155-158 `dropvaluesuntil` parity:
            //
            //     while frame.valuestackdepth > target_depth:
            //         frame.pop()          # locals_cells_stack_w[d] = None
            //
            // Python 3.11+ exception-table dispatch pops each value-stack
            // slot above the handler's declared depth and clears it to
            // `None` before pushing lasti / the exception value. Without
            // this step the vable array keeps stale refs at the popped
            // slots, which GC tracing and blackhole resume will read back.
            //
            // The raising PC is `site.lasti_py_pc`; its entry depth
            // (`depth_at_pc[site.lasti_py_pc]`) is the upper bound on
            // runtime valuestackdepth at any guard-firing point within
            // that PC's emitted IR, because every sub-op's guard runs
            // after its `emit_vsd!` and the peak depth within a pc equals
            // `depth_at_pc[pc]` for all raise-capable opcodes (BINARY_OP,
            // CALL, etc. enter with their args already on the stack).
            let raising_depth = depth_at_pc
                .get(site.lasti_py_pc)
                .copied()
                .unwrap_or(site.stack_depth);
            if is_portal {
                let mut unwind_depth = raising_depth;
                while unwind_depth > site.stack_depth {
                    unwind_depth -= 1;
                    emit_load_const_i!(
                        ssarepr,
                        int_tmp0,
                        (stack_base_absolute + unwind_depth as usize) as i64,
                    );
                    emit_vable_setarrayitem_ref!(ssarepr, 0_u16, int_tmp0, null_ref_reg);
                }
            }
            // pyframe.py:378-387 `pushvalue` semantics — each push writes
            // `locals_cells_stack_w[depth]` AND bumps `valuestackdepth`.
            // jtransform.py:1898 `do_fixed_list_setitem` lowers the array
            // write to `setarrayitem_vable_r`; jtransform.py:920-928
            // lowers the `valuestackdepth` write to `setfield_vable_i`.
            // Without this mirror, the handler's first opcode (and any
            // compiled-trace re-entry via ContinueRunningNormally) reads
            // stale vable state because only the SSA stack slot was
            // populated.
            let mut exc_slot = stack_base + site.stack_depth;
            let mut depth: u16 = site.stack_depth;
            if site.push_lasti {
                emit_load_const_i!(ssarepr, int_tmp0, site.lasti_py_pc as i64);
                emit_residual_call(
                    &mut ssarepr,
                    &mut graph,
                    &current_block.block(),
                    CallFlavor::Plain,
                    box_int_fn_idx,
                    &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                    ResKind::Ref,
                    Some(obj_tmp0),
                );
                emit_ref_copy!(ssarepr, exc_slot, obj_tmp0);
                if is_portal {
                    emit_load_const_i!(
                        ssarepr,
                        int_tmp0,
                        (stack_base_absolute + depth as usize) as i64,
                    );
                    emit_vable_setarrayitem_ref!(ssarepr, 0_u16, int_tmp0, exc_slot);
                }
                depth += 1;
                emit_vsd!(depth);
                exc_slot += 1;
            }
            emit_last_exc_value!(ssarepr, exc_slot);
            if is_portal {
                emit_load_const_i!(
                    ssarepr,
                    int_tmp0,
                    (stack_base_absolute + depth as usize) as i64,
                );
                emit_vable_setarrayitem_ref!(ssarepr, 0_u16, int_tmp0, exc_slot);
            }
            depth += 1;
            emit_vsd!(depth);
            emit_goto!(ssarepr, site.handler_py_pc);
        }

        // codewriter.py:45-47 `for kind in KINDS:
        //   regallocs[kind] = perform_register_allocation(graph, kind)`
        //
        // RPython runs regalloc on the CFG before flatten emits the
        // SSARepr (`codewriter.py:44` vs `:53-56`). Regalloc uses
        // `block.operations` + `link.args` for interference; `-live-`
        // markers don't exist yet. pyre dispatches directly to the
        // SSARepr — at regalloc time the `-live-` markers are present
        // but still hold empty args (`filter_liveness_in_place` runs
        // post-rename), so the allocator's generic `Insn::Op` walk is
        // a no-op on them, matching the upstream pre-liveness ordering.
        let max_stack_depth_observed = depth_at_pc.iter().copied().max().unwrap_or(0);
        let inputs = super::regalloc::ExternalInputs {
            portal_frame_reg,
            portal_ec_reg,
            // Portal frames carry a virtualizable + ec red argument
            // pair (interp_jit.py:64-69). Non-portal callees pass red
            // args via the call assembler edge; the dispatch loop
            // does not pre-load them into Ref registers.
            portal_inputs: portal_frame_reg != u16::MAX,
            stack_base,
            max_stack_depth: max_stack_depth_observed,
        };
        // `flow.rs` now models `Block.operations` as upstream
        // `SpaceOperation`, not flattened `Insn`. The direct-dispatch
        // walker still emits only SSA/flatten-level data, so the shadow
        // graph remains topology-only until a pre-regalloc Variable
        // environment is introduced.

        // Step 6A slice S4b: compute CFG-level link coalesce pairs
        // (`regalloc.py:79-96` projected onto pyre's u16 slot space)
        // and feed them into `allocate_registers` alongside the
        // existing SSARepr `*_copy` scanner.  Consumers (this call):
        //   - `collect_block_states(pc_blocks, joinpoints,
        //      catch_landing_blocks)` → target ENTRY FrameStates per
        //     BlockRef.
        //   - `link_exit_states` — populated by the walker at every
        //     `append_exit_with_state` site (S4a).
        //   - `collect_link_slot_pairs(graph, block_entry_states,
        //      link_exit_states)` → `(src_slot, dst_slot)` pairs.
        //
        // In pyre's positional-aligned architecture the pairs are
        // always `(slot, slot)` with `slot == slot`, so
        // `try_coalesce` is a runtime no-op — but wiring the call
        // preserves the exact iteration shape of
        // `regalloc.py:79-96`.  Intra-block `*_copy` coalescing stays
        // in `RegAllocator::coalesce_variables` (orthogonal source).
        let block_entry_states =
            collect_block_states(&pc_blocks, &joinpoints, &catch_landing_blocks);
        let cfg_coalesce_pairs =
            collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        let alloc_result = super::regalloc::allocate_registers(
            &ssarepr,
            code.varnames.len(),
            inputs,
            &cfg_coalesce_pairs,
        );
        super::regalloc::apply_rename(&mut ssarepr, &alloc_result.rename);

        // codewriter.py:55-56 parity: `compute_liveness(ssarepr)` runs
        // AFTER regalloc + flatten, so the live-register indices the
        // pass writes into each `-live-` marker are already the
        // post-rename colors. pyre's `filter_liveness_in_place` keeps
        // that ordering and then applies the pyre-only runtime-contract
        // filter (Ref-only, in-range, stack-complete, LiveVars-intersected).
        filter_liveness_in_place(&mut ssarepr, code, nlocals, stack_base, &depth_at_pc);
        // Runtime entry/liveness lookups expect the byte offset of the
        // surviving `-live-` marker for each Python PC
        // (`jitcode.get_live_vars_info` first checks `code[pc] ==
        // op_live`). `remove_repeated_live` may move that marker away
        // from the zero-byte `PcAnchor`, so record the final per-PC
        // live-marker positions here instead of the anchor indices.
        let pc_map = live_marker_indices_by_pc(&ssarepr, num_instrs);

        // codewriter.py:62-67 num_regs[kind] = max(coloring)+1
        // (or 0 if coloring is empty). Pass through to the Assembler
        // step so `JitCode.num_regs_*` reflect the post-regalloc
        // ceiling rather than the pre-regalloc PyFrame-slot range.
        let num_regs = super::assembler::NumRegs {
            int: alloc_result
                .num_regs
                .get(&super::flatten::Kind::Int)
                .copied()
                .unwrap_or(0),
            ref_: alloc_result
                .num_regs
                .get(&super::flatten::Kind::Ref)
                .copied()
                .unwrap_or(0),
            float: alloc_result
                .num_regs
                .get(&super::flatten::Kind::Float)
                .copied()
                .unwrap_or(0),
        };

        // codewriter.py:67-72 step 4 — assemble the SSARepr into an
        // owned JitCode, translate pc_map insn indices to byte offsets,
        // and stamp the per-graph metadata. See `Self::finalize_jitcode`.
        self.finalize_jitcode(
            assembler,
            ssarepr,
            code,
            w_code,
            pc_map,
            depth_at_pc,
            portal_frame_reg,
            portal_ec_reg,
            has_abort,
            merge_point_pc,
            num_regs,
        )
    }

    /// RPython: `codewriter.py:62-72` step 4 — produce the
    /// owned `JitCode` from the populated `SSARepr` and stamp the
    /// per-graph metadata.
    ///
    /// ```python
    /// num_regs = {kind: ... for kind in KINDS}
    /// self.assembler.assemble(ssarepr, jitcode, num_regs)
    /// jitcode.index = index
    /// ```
    ///
    /// pyre's combined step:
    ///   - `SSAReprEmitter::finish_with_positions` runs the
    ///     `assembler.py:assemble` analog through the shared
    ///     `self.assembler`, returning the owned `JitCode` plus the
    ///     translated `pc_map` byte offsets.
    ///   - jitdriver_sd / calldescr / fnaddr are stamped onto the
    ///     `JitCode` (call.py:148, call.py:174-187, call.py:167).
    ///   - `PyJitCodeMetadata` is bundled with the ref-count-stable
    ///     `Arc<JitCode>` plus the pyre-only `has_abort` /
    ///     `merge_point_pc` fields into the returned `PyJitCode`.
    fn finalize_jitcode(
        &self,
        assembler: SSAReprEmitter,
        ssarepr: SSARepr,
        code: &CodeObject,
        w_code: *const (),
        pc_map: Vec<usize>,
        depth_at_pc: Vec<u16>,
        portal_frame_reg: u16,
        portal_ec_reg: u16,
        has_abort: bool,
        merge_point_pc: Option<usize>,
        num_regs: super::assembler::NumRegs,
    ) -> PyJitCode {
        // pc_map[py_pc] currently holds SSARepr insn indices (returned by
        // SSAReprEmitter::current_pos()). Translate them to JitCode byte
        // offsets via ssarepr.insns_pos, populated during
        // Assembler::assemble (assembler.py:41-44). Runtime readers
        // (get_live_vars_info, resume dispatch) expect byte offsets.
        //
        // `codewriter.py:67` `self.assembler.assemble(ssarepr, jitcode, num_regs)`
        // parity: borrow the CodeWriter's single Assembler so
        // `all_liveness` / `num_liveness_ops` continue to accumulate
        // across every jitcode compiled on this thread.
        let (mut jitcode, pc_map_bytes) = {
            let mut asm = self.assembler.borrow_mut();
            assembler.finish_with_positions_from(&mut *asm, ssarepr, &pc_map, num_regs)
        };

        // call.py:148 `jd.mainjitcode.jitdriver_sd = jd` is assigned by
        // `assign_portal_jitdriver_indices` after the drain, where the
        // jdindex is known from this code's position in
        // `CallControl.jitdrivers_sd`. RPython's `JitCode` constructor
        // (jitcode.py:18) leaves `jitdriver_sd = None` for non-portals;
        // `grab_initial_jitcodes` is the single source of truth for
        // the portal assignment, so we leave it `None` here too rather
        // than guessing an index from a local `is_portal` flag (which
        // would collide once a second portal lands or a non-portal
        // slips in first).
        jitcode.jitdriver_sd = None;
        // call.py:167 `(fnaddr, calldescr) = self.get_jitcode_calldescr(graph)`.
        // The values are constant across CodeObjects in pyre — see
        // [`super::call::CallControl::get_jitcode_calldescr`] for the
        // PRE-EXISTING-ADAPTATION rationale.
        let (fnaddr, calldescr) = self
            .callcontrol()
            .get_jitcode_calldescr(code as *const CodeObject);
        jitcode.fnaddr = fnaddr;
        jitcode.calldescr = calldescr;
        // Per-code stack base in `locals_cells_stack_w`. RPython's JitCode
        // does not carry PyFrame layout data; keep it in PyJitCodeMetadata
        // and attach it to BlackholeInterpreter setup when pyre needs it.
        let frame_stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);

        let metadata = PyJitCodeMetadata {
            pc_map: pc_map_bytes,
            depth_at_py_pc: depth_at_pc,
            portal_frame_reg,
            portal_ec_reg,
            stack_base: frame_stack_base,
        };

        PyJitCode {
            jitcode: std::sync::Arc::new(jitcode),
            metadata,
            code_ptr: code as *const CodeObject,
            w_code,
            has_abort,
            merge_point_pc,
        }
    }

    /// RPython: `CodeWriter.make_jitcodes(verbose)` (codewriter.py:74-89).
    ///
    /// ```python
    /// def make_jitcodes(self, verbose=False):
    ///     log.info("making JitCodes...")
    ///     self.callcontrol.grab_initial_jitcodes()
    ///     count = 0
    ///     all_jitcodes = []
    ///     for graph, jitcode in self.callcontrol.enum_pending_graphs():
    ///         self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
    ///         all_jitcodes.append(jitcode)
    ///         count += 1
    ///         if not count % 500:
    ///             log.info("Produced %d jitcodes" % count)
    ///     self.assembler.finished(self.callcontrol.callinfocollection)
    ///     log.info("There are %d JitCode instances." % count)
    ///     log.info("There are %d -live- ops. Size of liveness is %s bytes" % (
    ///         self.assembler.num_liveness_ops, self.assembler.all_liveness_length))
    ///     return all_jitcodes
    /// ```
    ///
    /// Each freshly-compiled `PyJitCode` is `Arc`-wrapped before being
    /// inserted into `CallControl.jitcodes`; the next trace-side
    /// `state::jitcode_for(w_code)` callback hands the same `Arc`
    /// to `MetaInterpStaticData.jitcodes`, so both stores reference
    /// one allocation — the Rust analog of RPython's two stores
    /// referencing the same Python `JitCode` via refcount semantics.
    ///
    /// `grab_initial_jitcodes` reads its seed list from
    /// [`super::call::CallControl::jitdrivers_sd`]; callers register
    /// portals with [`Self::setup_jitdriver`] before invoking this
    /// method (matching codewriter.py:74 — `setup_jitdriver` followed
    /// by `make_jitcodes` is the upstream order).
    pub fn make_jitcodes(&self) -> Vec<*const PyJitCode> {
        // codewriter.py:75 `log.info("making JitCodes...")` — pyre has no
        // codewriter.py log channel, intentionally elided.

        // codewriter.py:76 `self.callcontrol.grab_initial_jitcodes()`.
        self.callcontrol().grab_initial_jitcodes();
        // codewriter.py:79-84 drain + per-jitcode assemble.
        let all_jitcodes = self.drain_unfinished_graphs();
        // call.py:148 `jd.mainjitcode.jitdriver_sd = jd` — assign
        // jdindex to each portal's populated `PyJitCode` AFTER the
        // drain so we use the actual position in
        // `CallControl.jitdrivers_sd` instead of a hardcoded `Some(0)`.
        self.assign_portal_jitdriver_indices();
        // codewriter.py:86-88 final log lines — elided.
        // codewriter.py:89 `return all_jitcodes`.
        all_jitcodes
    }

    /// Drain `CallControl.unfinished_graphs` — the body shared between
    /// [`Self::make_jitcodes`] (portal entry) and
    /// [`compile_jitcode_for_callee`] (trace-side callee path).
    ///
    /// RPython's `make_jitcodes` (codewriter.py:79-85) drains the queue
    /// once and then calls `assembler.finished()`. Both pyre adapters
    /// run the same drain so each batch ends with `assembler.finished()`
    /// and the matching `setup_indirectcalltargets(asm.indirectcalltargets)`
    /// publish, matching `codewriter.py:85` plus `pyjitpl.py:2262`.
    pub(crate) fn drain_unfinished_graphs(&self) -> Vec<*const PyJitCode> {
        let mut all_jitcodes: Vec<*const PyJitCode> = Vec::new();
        // codewriter.py:79 `for graph, jitcode in enum_pending_graphs():`.
        loop {
            let popped = self.callcontrol().enum_pending_graphs();
            let Some(code_ptr) = popped else {
                break;
            };
            let (w_code, merge_point_pc) = self
                .callcontrol()
                .queued_graph_inputs(code_ptr)
                .expect("queued graph must still have a cached skeleton");
            // codewriter.py:80 `self.transform_graph_to_jitcode(graph,
            //                     jitcode, verbose, len(all_jitcodes))`.
            //
            // PRE-EXISTING-ADAPTATION: `transform_graph_to_jitcode`
            // still returns a fresh `PyJitCode`, but `publish_jitcode`
            // now mutates the cached skeleton in place whenever the
            // slot's `Arc` is still unique. That keeps pyre closer to
            // RPython's "same JitCode object is filled later" flow; the
            // fallback for already-shared slots is still to replace the
            // `Arc`.
            let pyjitcode =
                self.transform_graph_to_jitcode(unsafe { &*code_ptr }, w_code, merge_point_pc);
            let key = code_ptr as usize;
            let raw_ptr = self.callcontrol().publish_jitcode(key, pyjitcode);
            // codewriter.py:81 `all_jitcodes.append(jitcode)`.
            all_jitcodes.push(raw_ptr);
        }
        // codewriter.py:85 `self.assembler.finished(self.callcontrol.callinfocollection)`.
        self.assembler
            .borrow_mut()
            .finished(&self.callcontrol().callinfocollection);
        self.publish_indirectcalltargets();
        all_jitcodes
    }

    /// `pyjitpl.py:2262`
    /// `self.setup_indirectcalltargets(asm.indirectcalltargets)`.
    ///
    /// RPython wires the codewriter's accumulated assembler set into
    /// `MetaInterpStaticData` during `finish_setup(codewriter, optimizer)`.
    /// pyre publishes the same accumulated set after each drain batch so the
    /// trace-side staticdata stays aligned with the writer's current
    /// `Assembler.indirectcalltargets`.
    fn publish_indirectcalltargets(&self) {
        let targets = self.assembler.borrow().indirectcalltargets_vec();
        pyre_jit_trace::state::setup_indirectcalltargets(targets);
    }

    /// call.py:148 `jd.mainjitcode.jitdriver_sd = jd` — propagate the
    /// jdindex from `CallControl.jitdrivers_sd` onto each portal's
    /// inner `JitCode.jitdriver_sd`. RPython mutates the same Python
    /// `JitCode` object the dict references; pyre uses
    /// `Arc::get_mut` on the freshly-installed `Arc<PyJitCode>` (and
    /// the inner `Arc<JitCode>`) for the same effect. Idempotent — a
    /// later `make_jitcodes` call may find the `Arc` already cloned
    /// into `MetaInterpStaticData.jitcodes` and skip the assignment;
    /// the previously-set value still holds.
    fn assign_portal_jitdriver_indices(&self) {
        let cc = self.callcontrol();
        // Snapshot the (key, jdindex) pairs first so the borrow on
        // `cc.jitdrivers_sd` is released before we mutate `cc.jitcodes`.
        let assignments: Vec<(usize, usize)> = cc
            .jitdrivers_sd
            .iter()
            .enumerate()
            .map(|(idx, jd)| (super::call::CallControl::jitcode_key(jd.portal_graph), idx))
            .collect();
        for (key, idx) in assignments {
            if let Some(arc) = cc.jitcodes.get_mut(&key) {
                if let Some(pyjitcode) = std::sync::Arc::get_mut(arc) {
                    if let Some(inner) = std::sync::Arc::get_mut(&mut pyjitcode.jitcode) {
                        inner.jitdriver_sd = Some(idx);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Jump target calculation (RPython: flatten.py link following)
// ---------------------------------------------------------------------------

/// Forward jump target: skip_caches(next_instr) + delta.
/// Must match pyre-interpreter/pyopcode.rs:jump_target_forward.
fn jump_target_forward(
    code: &CodeObject,
    num_instrs: usize,
    next_instr: usize,
    delta: usize,
) -> usize {
    let target = skip_caches(code, next_instr) + delta;
    target.min(num_instrs)
}

/// Single-source-of-truth backward-jump target calculation used by both
/// the loop-header pre-scan (pypy/module/pypyjit/interp_jit.py:103) and
/// the emitter (jtransform.py:1714 `handle_jit_marker__loop_header`).
///
/// Returns the target PC for `JumpBackward` (with `skip_caches` on the
/// next-PC base) and `JumpBackwardNoInterrupt` (direct `py_pc + 1 - delta`
/// arithmetic to match the interpreter's dispatch in pyopcode.rs).
/// Returns `None` for any non-backward-jump opcode.
fn backward_jump_target(
    code: &CodeObject,
    py_pc: usize,
    instr: Instruction,
    op_arg: pyre_interpreter::OpArg,
) -> Option<usize> {
    match instr {
        Instruction::JumpBackward { delta } => {
            Some(skip_caches(code, py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
        }
        Instruction::JumpBackwardNoInterrupt { delta } => {
            Some((py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
        }
        _ => None,
    }
}

/// Match pyre-interpreter/pyopcode.rs:skip_caches.
fn skip_caches(code: &CodeObject, mut pos: usize) -> usize {
    let mut state = OpArgState::default();
    while pos < code.instructions.len() {
        let (instruction, _) = state.get(code.instructions[pos]);
        if matches!(instruction, Instruction::Cache) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

// ---------------------------------------------------------------------------
// JitCode cache — RPython: `CallControl.get_jitcode` (call.py:155-172).
// The cache + `unfinished_graphs` queue live on `super::call::CallControl`;
// `CallControl::get_jitcode` is the canonical entry point.
// ---------------------------------------------------------------------------

/// Portal entry path: `setup_jitdriver` followed by `make_jitcodes` —
/// the warmspot order at codewriter.py:74-99. RPython runs this once
/// per `@jit_callback` decoration; pyre's portal discovery is lazy,
/// so this adapter fires per JIT entry. `setup_jitdriver` dedups by
/// `portal_graph` so `jitdrivers_sd` stays bounded by the number of
/// unique portals (see [`CodeWriter::setup_jitdriver`] for the
/// PRE-EXISTING-ADAPTATION rationale).
///
/// `make_jitcodes` is then the canonical RPython no-arg call: it
/// pulls its seed list from `CallControl.jitdrivers_sd` and runs
/// `grab_initial_jitcodes` → drain → `assembler.finished()` →
/// `assign_portal_jitdriver_indices`. The final stages short-circuit
/// on the second-and-later registration of any given portal because
/// `get_jitcode` skips already-built entries.
///
/// **Use this only for true portals.** Trace-side callee compiles go
/// through [`compile_jitcode_for_callee`] so they never touch
/// `jitdrivers_sd` — see `feedback_setup_jitdriver_portal_only`.
pub fn register_portal_jitdriver(
    code: &pyre_interpreter::CodeObject,
    w_code: *const (),
    merge_point_pc: Option<usize>,
) {
    let writer = CodeWriter::instance();
    // codewriter.py:96-99 `setup_jitdriver(jd)` — register the
    // portal so `grab_initial_jitcodes` finds it.
    writer.setup_jitdriver(super::call::JitDriverStaticData {
        portal_graph: code as *const pyre_interpreter::CodeObject,
        w_code,
        merge_point_pc,
    });
    // codewriter.py:74 `make_jitcodes()` — drain everything pending.
    writer.make_jitcodes();
}

/// Callee compile path: `CallControl.get_jitcode(graph)` followed by
/// the shared drain — the analog of jtransform's
/// `cc.callcontrol.get_jitcode(callee_graph)` (call.py:155-172) that
/// inserts the callee into `CallControl.jitcodes` and lets the
/// surrounding `make_jitcodes` drain transform it. RPython never
/// touches `jitdrivers_sd` here; pyre matches that by going through
/// `get_jitcode` + the shared drain helper directly, *not* through
/// `setup_jitdriver`.
pub fn compile_jitcode_for_callee(code: &pyre_interpreter::CodeObject, w_code: *const ()) {
    let writer = CodeWriter::instance();
    // call.py:155 `get_jitcode(graph)` — insert skeleton + queue.
    let _ = writer.callcontrol().get_jitcode(code, w_code, None);
    // codewriter.py:79-85 drain + assembler.finished() for the
    // newly-queued entry. No portal-jitdriver assignment because
    // this code is a callee, not a portal.
    writer.drain_unfinished_graphs();
}

/// Trace-side hook registered with `pyre_jit_trace::set_compile_jitcode_fn`
/// from `CodeWriter::new()`. Resolves the `w_code` (a `PyObjectRef`
/// wrapping a `CodeObject`) and routes it through
/// [`compile_jitcode_for_callee`] so the lazy compile pipeline runs
/// for one entry without polluting `jitdrivers_sd`.
///
/// RPython parity: jtransform's call to `cc.callcontrol.get_jitcode(callee)`
/// (call.py:155-172) inserts the callee into `CallControl.jitcodes`
/// and queues it on `unfinished_graphs`; the surrounding
/// `make_jitcodes` drain (codewriter.py:79-85) then transforms the
/// queued graph and pipes the result through `assembler.finished()`.
/// Pyre fires this callback per trace-side `state::jitcode_for(w_code)`
/// so the same transitive closure the RPython warmspot builds eagerly
/// is built lazily here, one callee at a time. The callback explicitly
/// must NOT call `setup_jitdriver` — see
/// `feedback_setup_jitdriver_portal_only`.
fn compile_jitcode_via_w_code(w_code: *const ()) -> Option<std::sync::Arc<PyJitCode>> {
    if w_code.is_null() {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_code.is_null() {
        return None;
    }
    let code = unsafe { &*raw_code };
    if let Some(existing) = CodeWriter::instance()
        .callcontrol()
        .find_compiled_jitcode_arc(code as *const _)
    {
        return Some(existing);
    }
    compile_jitcode_for_callee(code, w_code);
    // Hand the populated `Arc` back to the trace-side caller so the
    // SD entry stores the same allocation as `CallControl.jitcodes`.
    CodeWriter::instance()
        .callcontrol()
        .find_compiled_jitcode_arc(code as *const _)
}

/// Scan `code` for JUMP_BACKWARD targets — the PCs where
/// `transform_graph_to_jitcode` would emit `BC_JUMP_TARGET` and where
/// `jit_merge_point` is evaluated.
///
/// RPython parity: corresponds to `jtransform.py:1714-1718`
/// `handle_jit_marker__loop_header`, which walks the flow graph looking
/// for `jit_marker('loop_header', ...)` operations. pyre's "graph" is
/// raw Python bytecode, so the equivalent scan looks for
/// `JUMP_BACKWARD` instructions and resolves their target PCs.
///
/// Shared between `transform_graph_to_jitcode` (which uses the set to
/// decide emit locations) and `is_portal` (which tests emptiness to
/// classify the CodeObject without triggering a compile).
pub fn find_loop_header_pcs(
    code: &pyre_interpreter::CodeObject,
) -> std::collections::HashSet<usize> {
    let num_instrs = code.instructions.len();
    let mut loop_header_pcs: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut scan_state = OpArgState::default();
    for scan_pc in 0..num_instrs {
        let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
        if let Some(target) = backward_jump_target(code, scan_pc, scan_instr, scan_arg) {
            if target < num_instrs {
                loop_header_pcs.insert(target);
            }
        }
    }
    loop_header_pcs
}

// `liveness_regs_to_u8_sorted` tests removed alongside the helper.
// The 256-register cap is now enforced inside `encode_liveness` and
// covered by `majit_translate::liveness::encode_liveness*` tests.

#[cfg(test)]
mod tests {
    use super::*;
    use super::{
        FrameState, SpamBlockRef, attach_catch_exception_edge, collect_block_states,
        collect_link_slot_pairs, entry_arg_slots, entry_frame_state, entry_inputargs, mergeblock,
        new_shadow_graph,
    };
    use crate::jit::assembler::ArcByPtr;
    use crate::jit::flatten::{Insn, Kind, Label as FlatLabel, Operand, Register, SSARepr};
    use crate::jit::flow::{
        Block, Constant, ExitSwitch, FlowValue, FunctionGraph, Link, LinkArgPosition, LinkRef,
        SpaceOperationArg, Variable, VariableId, c_last_exception,
    };
    use pyre_interpreter::bytecode::{CodeObject, ConstantData};
    use pyre_interpreter::compile_exec;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn make_runtime_jitcode_with_fnaddr(fnaddr: usize) -> Arc<JitCode> {
        let mut jitcode = JitCodeBuilder::default().finish();
        jitcode.fnaddr = fnaddr as i64;
        Arc::new(jitcode)
    }

    fn first_nested_function_code(source: &str) -> CodeObject {
        let module = compile_exec(source).expect("compile failed");
        module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("expected nested function code object")
    }

    fn fresh_variable_factory(start: u32) -> impl FnMut(Option<Kind>) -> Variable {
        let mut next_id = start;
        move |kind| {
            let variable = Variable {
                id: VariableId(next_id),
                kind,
            };
            next_id += 1;
            variable
        }
    }

    fn sample_framestate() -> FrameState {
        FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::none().into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        )
    }

    #[test]
    fn exceptblock_link_args_uses_framestate_exception_pair() {
        let exc_type = Variable::new(VariableId(10), Kind::Ref);
        let exc_value = Variable::new(VariableId(11), Kind::Ref);
        let state = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            Some((exc_type.into(), exc_value.into())),
            Vec::new(),
            0,
        );

        assert_eq!(
            exceptblock_link_args(&state),
            vec![exc_type.into(), exc_value.into()],
        );
    }

    #[test]
    #[should_panic(expected = "exceptblock edge requires materialized exception pair")]
    fn exceptblock_link_args_rejects_missing_exception_pair() {
        let state = sample_framestate();
        let _ = exceptblock_link_args(&state);
    }

    #[test]
    fn explicit_raise_state_records_type_of_raised_value() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let block = graph.startblock.clone();
        let handler_exc_type = Variable::new(VariableId(20), Kind::Ref);
        let handler_exc_value = Variable::new(VariableId(21), Kind::Ref);
        let raised_value = Variable::new(VariableId(22), Kind::Ref);
        let state = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            Some((handler_exc_type.into(), handler_exc_value.into())),
            Vec::new(),
            0,
        );

        let raised = explicit_raise_state(&mut graph, &block, &state, raised_value.into(), 123);
        let Some((FlowValue::Variable(new_type), FlowValue::Variable(new_value))) =
            raised.last_exception
        else {
            panic!("explicit raise should materialize fresh exception vars");
        };
        assert_ne!(new_type.id, handler_exc_type.id);
        assert_eq!(new_value.id, raised_value.id);

        let block_borrow = block.borrow();
        let Some(op) = block_borrow.operations.last() else {
            panic!("explicit raise should record a graph operation");
        };
        assert_eq!(op.opname, "type");
        assert_eq!(op.offset, 123);
        assert_eq!(op.args, vec![SpaceOperationArg::from(raised_value)]);
        assert_eq!(op.result, Some(new_type.into()));
    }

    fn identity_arg_positions(count: usize) -> Vec<LinkArgPosition> {
        (0..count)
            .map(|index| LinkArgPosition {
                source_mergeable_index: Some(index),
                target_mergeable_index: Some(index),
            })
            .collect()
    }

    /// Step 6A slice S1 regression: `FrameState::mergeable_index_of` locates
    /// a Variable by its `VariableId` across locals / stack / last-exc
    /// positions and returns `None` for non-existent ids or non-Variable
    /// FlowValues.  Mirrors `framestate.py:38-43` `mergeable()` layout.
    #[test]
    fn mergeable_index_of_finds_variables_across_locals_stack_and_last_exc() {
        let v_local = Variable::new(VariableId(0), Kind::Ref);
        let v_stack = Variable::new(VariableId(1), Kind::Int);
        let v_exc_type = Variable::new(VariableId(2), Kind::Int);
        let v_exc_value = Variable::new(VariableId(3), Kind::Ref);
        let state = FrameState::new(
            vec![Some(v_local.into()), Some(Constant::none().into())],
            vec![v_stack.into()],
            Some((v_exc_type.into(), v_exc_value.into())),
            Vec::new(),
            0,
        );

        // Local at mergeable[0]; Constant at [1] has no Variable id.
        assert_eq!(state.mergeable_index_of(&v_local), Some(0));
        // Stack pushed after locals_w: len(locals_w) == 2, so stack[0] is at [2].
        assert_eq!(state.mergeable_index_of(&v_stack), Some(2));
        // last_exception pair sits at the end.
        assert_eq!(state.mergeable_index_of(&v_exc_type), Some(3));
        assert_eq!(state.mergeable_index_of(&v_exc_value), Some(4));
        // Unknown VariableId returns None.
        let v_absent = Variable::new(VariableId(99), Kind::Ref);
        assert_eq!(state.mergeable_index_of(&v_absent), None);
    }

    /// Step 6A slice S2 regression: `FrameState::mergeable_index_to_slot`
    /// is identity in the regular `[0, locals_w.len() + stack.len())`
    /// range and returns `None` for the `last_exception` pair.
    #[test]
    fn mergeable_index_to_slot_is_identity_in_regular_range() {
        let v_local = Variable::new(VariableId(0), Kind::Ref);
        let v_stack = Variable::new(VariableId(1), Kind::Int);
        let v_exc_type = Variable::new(VariableId(2), Kind::Int);
        let v_exc_value = Variable::new(VariableId(3), Kind::Ref);
        let state = FrameState::new(
            vec![Some(v_local.into()), Some(Constant::none().into())],
            vec![v_stack.into()],
            Some((v_exc_type.into(), v_exc_value.into())),
            Vec::new(),
            0,
        );

        // Regular range: identity. locals_w.len() + stack.len() = 2 + 1 = 3.
        assert_eq!(state.mergeable_index_to_slot(0), Some(0));
        assert_eq!(state.mergeable_index_to_slot(1), Some(1));
        assert_eq!(state.mergeable_index_to_slot(2), Some(2));
        // last_exception pair at mergeable[3..5) has no register slot.
        assert_eq!(state.mergeable_index_to_slot(3), None);
        assert_eq!(state.mergeable_index_to_slot(4), None);
        // Anything beyond mergeable: also None.
        assert_eq!(state.mergeable_index_to_slot(100), None);
    }

    /// Step 6A slice S2 regression: `variable_slot` composes S1 + S2 so
    /// a Variable resolves directly to its register slot.  last_exc
    /// Variables resolve to `None` even though they DO appear in
    /// `mergeable()`.
    #[test]
    fn variable_slot_resolves_locals_and_stack_but_not_last_exc() {
        let v_local = Variable::new(VariableId(0), Kind::Ref);
        let v_stack = Variable::new(VariableId(1), Kind::Int);
        let v_exc_type = Variable::new(VariableId(2), Kind::Int);
        let v_exc_value = Variable::new(VariableId(3), Kind::Ref);
        let state = FrameState::new(
            vec![Some(v_local.into())],
            vec![v_stack.into()],
            Some((v_exc_type.into(), v_exc_value.into())),
            Vec::new(),
            0,
        );

        // Local at mergeable[0] → slot 0.  Stack at mergeable[1] → slot 1.
        assert_eq!(state.variable_slot(&v_local), Some(0));
        assert_eq!(state.variable_slot(&v_stack), Some(1));
        // last_exception variables: present in mergeable but no slot.
        assert_eq!(state.variable_slot(&v_exc_type), None);
        assert_eq!(state.variable_slot(&v_exc_value), None);
        // Absent variable: None.
        let v_absent = Variable::new(VariableId(99), Kind::Ref);
        assert_eq!(state.variable_slot(&v_absent), None);
    }

    /// Helper: build a `link_exit_states` map from `(LinkRef,
    /// FrameState)` pairs.  Production walker will populate this by
    /// cloning `currentstate` at each `append_exit` call
    /// (`flowcontext.py:1237,1268-1280`).
    fn link_exit_states_from(pairs: Vec<(LinkRef, FrameState)>) -> HashMap<LinkRef, FrameState> {
        let mut map = HashMap::new();
        for (link, state) in pairs {
            map.insert(link, state);
        }
        map
    }

    /// Step 6A slice S3 regression: `collect_link_slot_pairs` emits a
    /// trivially-equal slot pair at every mergeable position where
    /// both source (EXIT state) and target (ENTRY state) hold a
    /// Variable.  The pairs are positional by `getoutputargs`
    /// construction (`codewriter.rs:333-346`); see S3c docstring.
    #[test]
    fn collect_link_slot_pairs_emits_positional_pairs_for_variable_links() {
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let start_arg2 = Variable::new(VariableId(1), Kind::Ref);
        let mid_arg = Variable::new(VariableId(2), Kind::Ref);
        let mid_arg2 = Variable::new(VariableId(3), Kind::Ref);
        let mut graph = FunctionGraph::new(
            "coalesce",
            Block::shared(vec![start_arg.into(), start_arg2.into()]),
            None,
        );
        let mid = graph.new_block(vec![mid_arg.into(), mid_arg2.into()]);
        let link = Link::new(
            vec![start_arg.into(), start_arg2.into()],
            Some(mid.clone()),
            None,
        )
        .with_arg_positions(identity_arg_positions(2))
        .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        let start_state = FrameState::new(
            vec![Some(start_arg.into()), Some(start_arg2.into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mid_state = FrameState::new(
            vec![Some(mid_arg.into()), Some(mid_arg2.into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(graph.startblock.clone(), start_state.clone());
        block_entry_states.insert(mid.clone(), mid_state);
        let link_exit_states = link_exit_states_from(vec![(link, start_state)]);

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);
    }

    /// Step 6A slice S3 regression: Constant link args do not
    /// contribute a pair (source mergeable at that position is a
    /// Constant, not a Variable).  Mirrors `flatten.py:355-363`
    /// `flatten_list` + `regalloc.py:99-101` `if isinstance(v,
    /// Variable)` — Constants pass through unchanged.
    #[test]
    fn collect_link_slot_pairs_skips_constant_link_args() {
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let next_arg = Variable::new(VariableId(1), Kind::Ref);
        let mut graph =
            FunctionGraph::new("with_const", Block::shared(vec![start_arg.into()]), None);
        let next = graph.new_block(vec![next_arg.into()]);
        let link = Link::new(vec![Constant::signed(42).into()], Some(next.clone()), None)
            .with_arg_positions(identity_arg_positions(1))
            .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        // Source EXIT state has a Constant at position 0 (matching
        // the Constant-carrying link arg) — e.g. a parameter with a
        // default Constant flowing through a branch.  Target ENTRY
        // state still has a Variable.
        let start_exit = FrameState::new(
            vec![Some(Constant::signed(42).into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let next_state =
            FrameState::new(vec![Some(next_arg.into())], Vec::new(), None, Vec::new(), 0);
        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(
            graph.startblock.clone(),
            FrameState::new(
                vec![Some(start_arg.into())],
                Vec::new(),
                None,
                Vec::new(),
                0,
            ),
        );
        block_entry_states.insert(next.clone(), next_state);
        let link_exit_states = link_exit_states_from(vec![(link, start_exit)]);

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert!(
            pairs.is_empty(),
            "constant link args contribute no coalesce pairs"
        );
    }

    /// Step 6A slice S3 regression: a Link whose target has no
    /// attached FrameState (catch landings, returnblock, exceptblock)
    /// contributes no pairs.  Covers the
    /// `block_entry_states.get(&target)` early-exit branch.
    #[test]
    fn collect_link_slot_pairs_skips_missing_target_framestate() {
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let next_arg = Variable::new(VariableId(1), Kind::Ref);
        let mut graph = FunctionGraph::new(
            "missing_target",
            Block::shared(vec![start_arg.into()]),
            None,
        );
        let next = graph.new_block(vec![next_arg.into()]);
        let link = Link::new(vec![start_arg.into()], Some(next.clone()), None)
            .with_arg_positions(identity_arg_positions(1))
            .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        let start_state = FrameState::new(
            vec![Some(start_arg.into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(graph.startblock.clone(), start_state.clone());
        // Deliberately do NOT insert `next` — mimics catch landing block.
        let link_exit_states = link_exit_states_from(vec![(link, start_state)]);

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert!(pairs.is_empty());
    }

    /// Step 6A slice S3c regression: a Link whose source EXIT state
    /// replaced the ENTRY-state Variable with a freshly-allocated
    /// mid-block Variable still emits the correct slot pair.
    /// Previously the helper consulted only the source block's ENTRY
    /// state and missed the fresh Variable.  S3c supplies the source
    /// state via `link_exit_states`; the positional walk ignores
    /// identity and looks only at whether each mergeable position is
    /// a Variable on both sides.
    ///
    /// Scenario:
    ///  - Source ENTRY locals_w = [v_entry] at mergeable position 0.
    ///  - Walker STORE_FAST overwrites locals_w[0] with v_exit; at
    ///    terminator time currentstate.locals_w[0] == v_exit.
    ///  - Link.args = [v_exit].  Target ENTRY locals_w = [v_target].
    ///
    /// Expected: one (0, 0) coalesce pair via link_exit_states[link].
    /// See Task #222.
    #[test]
    fn collect_link_slot_pairs_finds_variable_via_link_exit_state() {
        let v_entry = Variable::new(VariableId(0), Kind::Ref);
        let v_exit = Variable::new(VariableId(1), Kind::Ref);
        let v_target = Variable::new(VariableId(2), Kind::Ref);
        let mut graph = FunctionGraph::new("exit_state", Block::shared(vec![v_entry.into()]), None);
        let target = graph.new_block(vec![v_target.into()]);
        let link = Link::new(vec![v_exit.into()], Some(target.clone()), None)
            .with_arg_positions(identity_arg_positions(1))
            .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        let start_entry =
            FrameState::new(vec![Some(v_entry.into())], Vec::new(), None, Vec::new(), 0);
        let start_exit =
            FrameState::new(vec![Some(v_exit.into())], Vec::new(), None, Vec::new(), 0);
        let target_entry =
            FrameState::new(vec![Some(v_target.into())], Vec::new(), None, Vec::new(), 0);

        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(graph.startblock.clone(), start_entry);
        block_entry_states.insert(target.clone(), target_entry);
        let link_exit_states = link_exit_states_from(vec![(link, start_exit)]);

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert_eq!(
            pairs,
            vec![(0, 0)],
            "EXIT-state Variable must not prevent pair emission",
        );
    }

    /// Step 6A slice S3c regression: a Link with no
    /// `link_exit_states` entry contributes no pairs.  Production
    /// walker MUST populate the EXIT snapshot for every link it
    /// emits; a missing entry (un-wired path or test that skipped it)
    /// skips rather than panicking to keep the helper robust during
    /// staged integration.
    #[test]
    fn collect_link_slot_pairs_skips_links_without_exit_state() {
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let next_arg = Variable::new(VariableId(1), Kind::Ref);
        let mut graph = FunctionGraph::new(
            "missing_exit_state",
            Block::shared(vec![start_arg.into()]),
            None,
        );
        let next = graph.new_block(vec![next_arg.into()]);
        let link = Link::new(vec![start_arg.into()], Some(next.clone()), None)
            .with_arg_positions(identity_arg_positions(1))
            .into_ref();
        graph.startblock.closeblock(vec![link]);

        let start_state = FrameState::new(
            vec![Some(start_arg.into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let next_state =
            FrameState::new(vec![Some(next_arg.into())], Vec::new(), None, Vec::new(), 0);
        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(graph.startblock.clone(), start_state);
        block_entry_states.insert(next.clone(), next_state);
        // Deliberately empty: no source EXIT state available.
        let link_exit_states = HashMap::new();

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert!(pairs.is_empty());
    }

    /// Step 6A slice S3c regression: `LoadFast`-style aliasing where
    /// the same Variable lives at two mergeable positions
    /// simultaneously (`codewriter.rs:2413-2414` pushes the local's
    /// own Variable onto the stack).  The positional walk must emit
    /// one pair per mergeable position, not one pair per Variable,
    /// so both (0, 0) for the local slot and (1, 1) for the stack
    /// slot fire.  Proves the helper is not vulnerable to
    /// Variable-collision ambiguity.
    #[test]
    fn collect_link_slot_pairs_handles_variable_aliased_across_slots() {
        let v_local = Variable::new(VariableId(0), Kind::Ref);
        let v_next_local = Variable::new(VariableId(1), Kind::Ref);
        let v_next_stack = Variable::new(VariableId(2), Kind::Ref);
        let mut graph = FunctionGraph::new("aliased", Block::shared(vec![v_local.into()]), None);
        // target inputargs == mergeable Variables in locals_w + stack
        let next = graph.new_block(vec![v_next_local.into(), v_next_stack.into()]);
        // Link carries v_local twice — once for locals_w[0], once for stack[0].
        let link = Link::new(
            vec![v_local.into(), v_local.into()],
            Some(next.clone()),
            None,
        )
        .with_arg_positions(identity_arg_positions(2))
        .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        // Source EXIT state: locals_w[0] AND stack[0] both hold v_local.
        let start_exit = FrameState::new(
            vec![Some(v_local.into())],
            vec![v_local.into()],
            None,
            Vec::new(),
            0,
        );
        let next_entry = FrameState::new(
            vec![Some(v_next_local.into())],
            vec![v_next_stack.into()],
            None,
            Vec::new(),
            0,
        );
        let mut block_entry_states = HashMap::new();
        block_entry_states.insert(
            graph.startblock.clone(),
            FrameState::new(
                vec![Some(v_local.into())],
                vec![v_local.into()],
                None,
                Vec::new(),
                0,
            ),
        );
        block_entry_states.insert(next.clone(), next_entry);
        let link_exit_states = link_exit_states_from(vec![(link, start_exit)]);

        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert_eq!(
            pairs,
            vec![(0, 0), (1, 1)],
            "positional walk must emit one pair per mergeable slot, not per Variable",
        );
    }

    /// Step 6A slice S3b regression: `collect_block_states` absorbs
    /// the walker's SpamBlockRef containers, skipping entries whose
    /// FrameState is `None` (catch landings), deduplicating blocks
    /// that appear in multiple containers.
    #[test]
    fn collect_block_states_walks_all_walker_containers() {
        let mut graph = FunctionGraph::new("s3b", Block::shared(Vec::new()), None);
        let block_a = graph.new_block(Vec::new());
        let block_b = graph.new_block(Vec::new());
        let block_landing = graph.new_block(Vec::new());

        let state_a = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let state_b = FrameState::new(
            vec![Some(Variable::new(VariableId(1), Kind::Ref).into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );

        let a_ref = SpamBlockRef::new(block_a.clone(), Some(state_a.clone()));
        let b_ref = SpamBlockRef::new(block_b.clone(), Some(state_b.clone()));
        let landing_ref = SpamBlockRef::new(block_landing.clone(), None);

        let pc_blocks: Vec<Option<SpamBlockRef>> =
            vec![Some(a_ref.clone()), None, Some(b_ref.clone())];
        let mut joinpoints: HashMap<usize, Vec<SpamBlockRef>> = HashMap::new();
        // Same block_a appears also under joinpoints — last-write-wins
        // still yields state_a, not a corrupt empty.
        joinpoints.insert(0, vec![a_ref.clone()]);
        joinpoints.insert(2, vec![b_ref.clone()]);
        let mut catch_landing_blocks: HashMap<u16, SpamBlockRef> = HashMap::new();
        // Catch landings have framestate = None and MUST be skipped.
        catch_landing_blocks.insert(7, landing_ref);

        let map = collect_block_states(&pc_blocks, &joinpoints, &catch_landing_blocks);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&block_a), Some(&state_a));
        assert_eq!(map.get(&block_b), Some(&state_b));
        assert!(
            !map.contains_key(&block_landing),
            "catch-landing block with None framestate must not appear in the map"
        );
    }

    /// Step 6A slice S3b + S3 end-to-end: when the
    /// `block_entry_states` map is built from the walker helpers
    /// (`collect_block_states`), `collect_link_slot_pairs` still
    /// yields the same positional pair as the hand-built variant.
    /// S3c revision: caller also supplies a `link_exit_states` map —
    /// here populated with the source block's ENTRY state because the
    /// fabricated graph has no mid-block ops.
    #[test]
    fn collect_block_states_feeds_collect_link_slot_pairs() {
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let next_arg = Variable::new(VariableId(1), Kind::Ref);
        let mut graph = FunctionGraph::new("s3b_e2e", Block::shared(vec![start_arg.into()]), None);
        let next = graph.new_block(vec![next_arg.into()]);
        let link = Link::new(vec![start_arg.into()], Some(next.clone()), None)
            .with_arg_positions(identity_arg_positions(1))
            .into_ref();
        graph.startblock.closeblock(vec![link.clone()]);

        let start_state = FrameState::new(
            vec![Some(start_arg.into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let next_state =
            FrameState::new(vec![Some(next_arg.into())], Vec::new(), None, Vec::new(), 0);

        let pc_blocks = vec![
            Some(SpamBlockRef::new(
                graph.startblock.clone(),
                Some(start_state.clone()),
            )),
            Some(SpamBlockRef::new(next.clone(), Some(next_state.clone()))),
        ];
        let joinpoints = HashMap::new();
        let catch_landing_blocks = HashMap::new();

        let block_entry_states =
            collect_block_states(&pc_blocks, &joinpoints, &catch_landing_blocks);
        let link_exit_states = link_exit_states_from(vec![(link, start_state)]);
        let pairs = collect_link_slot_pairs(&graph, &block_entry_states, &link_exit_states);
        assert_eq!(pairs, vec![(0, 0)]);
    }

    #[test]
    fn pc_anchor_and_live_marker_rescan_follow_final_ssarepr_order() {
        let mut ssarepr = SSARepr::new("t");
        ssarepr.insns.push(Insn::PcAnchor(0));
        ssarepr
            .insns
            .push(Insn::live(vec![Operand::Register(Register::new(
                Kind::Ref,
                0,
            ))]));
        // `remove_repeated_live` rewrites `-live-, Label` into
        // `Label, -live-`; the anchor scan must use the final insn order,
        // not the pre-rewrite placeholder index.
        ssarepr.insns.push(Insn::Label(FlatLabel::new("pc1")));
        ssarepr.insns.push(Insn::PcAnchor(1));
        ssarepr
            .insns
            .push(Insn::live(vec![Operand::Register(Register::new(
                Kind::Ref,
                1,
            ))]));

        crate::jit::liveness::remove_repeated_live(&mut ssarepr);

        assert_eq!(pc_anchor_positions(&ssarepr, 2), vec![0, 3]);
        assert_eq!(live_marker_indices_by_pc(&ssarepr, 2), vec![2, 4]);
    }

    #[test]
    fn publish_indirectcalltargets_updates_trace_staticdata() {
        let writer = CodeWriter::new();
        let j100 = make_runtime_jitcode_with_fnaddr(0x100);
        let j200 = make_runtime_jitcode_with_fnaddr(0x200);

        {
            let mut assembler = writer.assembler.borrow_mut();
            assembler
                .indirectcalltargets
                .insert(ArcByPtr::new(j100.clone()));
            assembler
                .indirectcalltargets
                .insert(ArcByPtr::new(j200.clone()));
        }

        writer.publish_indirectcalltargets();

        let hit_100 = pyre_jit_trace::state::bytecode_for_address(0x100)
            .expect("fnaddr 0x100 must be published to trace staticdata");
        let hit_200 = pyre_jit_trace::state::bytecode_for_address(0x200)
            .expect("fnaddr 0x200 must be published to trace staticdata");
        assert!(Arc::ptr_eq(&hit_100, &j100));
        assert!(Arc::ptr_eq(&hit_200, &j200));
        assert!(pyre_jit_trace::state::bytecode_for_address(0x300).is_none());
    }

    #[test]
    fn get_jitcode_queues_canonical_raw_graph_only() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let code_ref = unsafe { &*raw_code };

        let _ = writer
            .callcontrol()
            .get_jitcode(code_ref, w_code as *const (), Some(11));

        let queued = writer
            .callcontrol()
            .enum_pending_graphs()
            .expect("fresh jitcode must queue one graph");
        let (queued_w_code, queued_merge_point_pc) = writer
            .callcontrol()
            .queued_graph_inputs(raw_code)
            .expect("queued graph must still have a cached skeleton");

        assert_eq!(queued, raw_code);
        assert_eq!(queued_w_code, w_code as *const ());
        assert_eq!(queued_merge_point_pc, Some(11));
    }

    #[test]
    fn drain_unfinished_graphs_preserves_unique_pyjitcode_identity() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let code_ref = unsafe { &*raw_code };
        let key = raw_code as usize;

        let _ = writer
            .callcontrol()
            .get_jitcode(code_ref, w_code as *const (), None);
        let skeleton_ptr = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("skeleton jitcode must be cached");
            Arc::as_ptr(slot)
        };

        let all_jitcodes = writer.drain_unfinished_graphs();
        let populated_ptr = {
            let slot = writer
                .callcontrol()
                .jitcodes
                .get(&key)
                .expect("populated jitcode must remain cached");
            Arc::as_ptr(slot)
        };

        assert_eq!(all_jitcodes, vec![populated_ptr]);
        assert_eq!(
            populated_ptr, skeleton_ptr,
            "unique skeleton Arc should be filled in place"
        );
        let pyjit = writer.callcontrol().find_jitcode(raw_code).unwrap();
        assert!(
            !pyjit.metadata.pc_map.is_empty(),
            "drain must populate bytecode metadata on the existing entry"
        );
        assert_eq!(pyjit.w_code, w_code as *const ());
    }

    #[test]
    fn entry_arg_slots_counts_kwonly_varargs_and_varkeywords() {
        let code =
            first_nested_function_code("def f(a, b, *args, c, d, **kw):\n    return a + b\n");

        assert_eq!(entry_arg_slots(&code), 6);
    }

    #[test]
    fn new_shadow_graph_uses_entry_inputargs_as_startblock_shape() {
        let code =
            first_nested_function_code("def f(a, b, *args, c, d, **kw):\n    return a + b\n");

        let expected_inputargs = entry_inputargs(&code);
        let graph = new_shadow_graph(&code);
        let startblock = graph.startblock.borrow();
        let returnblock = graph.returnblock.borrow();

        assert_eq!(graph.name, "f");
        assert_eq!(startblock.inputargs, expected_inputargs);
        assert_eq!(startblock.inputargs.len(), 6);
        for (idx, value) in startblock.inputargs.iter().enumerate() {
            match value {
                FlowValue::Variable(variable) => {
                    assert_eq!(variable.id, VariableId(idx as u32));
                    assert_eq!(variable.kind, Some(Kind::Ref));
                }
                other => panic!("expected variable inputarg, got {other:?}"),
            }
        }

        assert_eq!(returnblock.inputargs.len(), 1);
        match &returnblock.inputargs[0] {
            FlowValue::Variable(variable) => {
                assert_eq!(variable.id, VariableId(6));
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected variable return arg, got {other:?}"),
        }
    }

    #[test]
    fn graph_entry_inputargs_append_portal_frame_and_ec() {
        let code = first_nested_function_code(
            "def f(a):\n    while a:\n        a = a - 1\n    return a\n",
        );

        let inputargs = graph_entry_inputargs(&code, true);
        let arg_slots = entry_arg_slots(&code);
        assert_eq!(inputargs.len(), arg_slots + 2);
        match &inputargs[arg_slots] {
            FlowValue::Variable(variable) => {
                assert_eq!(*variable, portal_graph_inputvars(&code).0);
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected portal frame variable, got {other:?}"),
        }
        match &inputargs[arg_slots + 1] {
            FlowValue::Variable(variable) => {
                assert_eq!(*variable, portal_graph_inputvars(&code).1);
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected portal ec variable, got {other:?}"),
        }
    }

    #[test]
    fn portal_shadow_graph_reserves_return_var_after_frame_and_ec() {
        let code = first_nested_function_code(
            "def f(a):\n    while a:\n        a = a - 1\n    return a\n",
        );

        let graph = new_shadow_graph_with_portal_inputs(&code, true);
        let startblock = graph.startblock.borrow();
        let returnblock = graph.returnblock.borrow();

        assert_eq!(startblock.inputargs, graph_entry_inputargs(&code, true));
        match &returnblock.inputargs[0] {
            FlowValue::Variable(variable) => {
                assert_eq!(
                    variable.id,
                    VariableId(graph_entry_inputargs(&code, true).len() as u32)
                );
                assert_eq!(variable.kind, Some(Kind::Ref));
            }
            other => panic!("expected variable return arg, got {other:?}"),
        }
    }

    #[test]
    fn portal_jit_merge_point_graph_args_match_upstream_shape() {
        let code = first_nested_function_code(
            "def f(a):\n    while a:\n        a = a - 1\n    return a\n",
        );
        let w_code = pyre_interpreter::box_code_constant(&code);
        let graph = new_shadow_graph_with_portal_inputs(&code, true);
        let args = portal_jit_merge_point_graph_args(&graph, 17, w_code as *const ());

        assert_eq!(args.len(), 7);
        match &args[0] {
            SpaceOperationArg::Value(FlowValue::Constant(constant)) => {
                assert_eq!(constant, &Constant::signed(0));
            }
            other => panic!("expected jdindex constant, got {other:?}"),
        }
        match &args[1] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Int);
                assert_eq!(
                    list.content,
                    vec![Constant::signed(17).into(), Constant::signed(0).into()]
                );
            }
            other => panic!("expected greens int list, got {other:?}"),
        }
        match &args[2] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Ref);
                assert_eq!(list.content.len(), 1);
                match &list.content[0] {
                    FlowValue::Constant(constant) => {
                        assert_eq!(constant.kind, Some(Kind::Ref));
                    }
                    other => panic!("expected pycode ref constant, got {other:?}"),
                }
            }
            other => panic!("expected greens ref list, got {other:?}"),
        }
        match &args[3] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Float);
                assert!(list.content.is_empty());
            }
            other => panic!("expected empty greens float list, got {other:?}"),
        }
        match &args[4] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Int);
                assert!(list.content.is_empty());
            }
            other => panic!("expected empty reds int list, got {other:?}"),
        }
        match &args[5] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Ref);
                assert_eq!(
                    list.content,
                    vec![
                        portal_graph_inputvars(&code).0.into(),
                        portal_graph_inputvars(&code).1.into(),
                    ]
                );
            }
            other => panic!("expected reds ref list, got {other:?}"),
        }
        match &args[6] {
            SpaceOperationArg::ListOfKind(list) => {
                assert_eq!(list.kind, Kind::Float);
                assert!(list.content.is_empty());
            }
            other => panic!("expected empty reds float list, got {other:?}"),
        }
    }

    #[test]
    fn entry_frame_state_matches_pygraph_locals_shape() {
        let code =
            first_nested_function_code("def f(a, b, *args, c, d, **kw):\n    return a + b\n");
        let state = entry_frame_state(&code, false);

        assert_eq!(state.locals_w.len(), code.varnames.len());
        assert_eq!(state.getvariables(), entry_inputargs(&code));
        assert!(state.stack.is_empty());
        assert!(state.last_exception.is_none());
    }

    #[test]
    fn frame_blocks_follow_exception_table_ranges() {
        let code = first_nested_function_code(
            "def f(a):\n    try:\n        return a\n    except Exception:\n        return 0\n",
        );
        let entries = pyre_interpreter::bytecode::decode_exception_table(&code.exceptiontable);
        assert!(!entries.is_empty());

        let first = &entries[0];
        let blocks = frame_blocks_for_offset(&code, first.start as usize);

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].start_offset, first.start as usize);
        assert_eq!(blocks[0].end_offset, first.end as usize);
        assert_eq!(blocks[0].handler_offset, first.target as usize);
        assert_eq!(blocks[0].stack_depth, first.depth as u16);
        assert_eq!(blocks[0].push_lasti, first.push_lasti);
    }

    #[test]
    fn framestate_copy_refreshes_variables() {
        let state = sample_framestate();
        let mut fresh = fresh_variable_factory(10);
        let copied = state.copy(&mut fresh);

        assert!(state.matches(&copied));
        assert_ne!(state, copied);
        assert_eq!(copied.locals_w[1], Some(Constant::none().into()));
    }

    #[test]
    #[should_panic]
    fn framestate_matches_asserts_on_different_next_offset() {
        let left = sample_framestate();
        let right = FrameState::new(
            left.locals_w.clone(),
            left.stack.clone(),
            left.last_exception.clone(),
            left.blocklist.clone(),
            1,
        );

        let _ = left.matches(&right);
    }

    #[test]
    fn framestate_union_generalizes_different_constants() {
        let state1 = sample_framestate();
        let state2 = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::signed(42).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mut fresh = fresh_variable_factory(20);
        let union = state1
            .union(&state2, &mut fresh)
            .expect("union should succeed");

        match union.locals_w[1].as_ref() {
            Some(FlowValue::Variable(variable)) => assert_eq!(variable.id, VariableId(20)),
            other => panic!("expected generalized variable, got {other:?}"),
        }
    }

    #[test]
    fn framestate_union_matches_more_general_variable_state() {
        let state1 = sample_framestate();
        let state2 = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Variable::new(VariableId(5), Kind::Ref).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let mut fresh = fresh_variable_factory(20);
        let union = state1
            .union(&state2, &mut fresh)
            .expect("union should succeed");

        assert!(union.matches(&state2));
    }

    #[test]
    fn framestate_getoutputargs_follows_target_variables() {
        let state1 = sample_framestate();
        let state2 = FrameState::new(
            vec![
                Some(Variable::new(VariableId(1), Kind::Ref).into()),
                Some(Variable::new(VariableId(2), Kind::Ref).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );

        assert_eq!(
            state1.getoutputargs(&state2),
            vec![
                Variable::new(VariableId(0), Kind::Ref).into(),
                Constant::none().into(),
            ]
        );
        assert_eq!(
            state1.getoutputargs_with_positions(&state2).1,
            identity_arg_positions(2),
        );
    }

    #[test]
    fn callcontrol_compiled_lookup_ignores_skeleton_pyjitcode() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let code_ref = unsafe { &*raw_code };

        let _ = writer
            .callcontrol()
            .get_jitcode(code_ref, w_code as *const (), None);
        assert!(writer.callcontrol().find_jitcode_arc(raw_code).is_some());
        assert!(
            writer
                .callcontrol()
                .find_compiled_jitcode_arc(raw_code)
                .is_none(),
            "fresh shells must not be treated as populated jitcodes"
        );

        writer.drain_unfinished_graphs();
        assert!(
            writer
                .callcontrol()
                .find_compiled_jitcode_arc(raw_code)
                .is_some(),
            "drained jitcodes must become visible through the compiled-only lookup"
        );
    }

    #[test]
    fn compile_jitcode_via_w_code_reuses_existing_compiled_arc() {
        let writer = CodeWriter::instance();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let code_ptr = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };
        let code_ref = unsafe { &*code_ptr };

        compile_jitcode_for_callee(code_ref, w_code as *const ());
        let cached = writer
            .callcontrol()
            .find_compiled_jitcode_arc(code_ptr)
            .expect("callee compile must cache a populated jitcode");

        let returned = compile_jitcode_via_w_code(w_code as *const ())
            .expect("callback should reuse existing populated jitcode");
        assert!(
            Arc::ptr_eq(&cached, &returned),
            "trace-side callback should return the already-cached populated Arc"
        );
    }

    #[test]
    fn attach_catch_exception_edge_marks_block_as_canraise() {
        let code = first_nested_function_code("def f():\n    return 1\n");
        let mut graph = new_shadow_graph(&code);
        let catch_block = graph.new_block(Vec::new());
        let catch_ref = SpamBlockRef::new(catch_block.clone(), None);
        let mut link_exit_states: HashMap<LinkRef, FrameState> = HashMap::new();
        let source_state = FrameState::new(Vec::new(), Vec::new(), None, Vec::new(), 0);
        let startblock_ref = graph.startblock.clone();

        let link = attach_catch_exception_edge(
            &mut graph,
            &startblock_ref,
            &catch_ref,
            &source_state,
            &mut link_exit_states,
        );
        let startblock = graph.startblock.borrow();

        assert_eq!(
            startblock.exitswitch,
            Some(ExitSwitch::Value(c_last_exception().into()))
        );
        assert_eq!(startblock.exits.len(), 1);
        assert_eq!(startblock.exits[0], link);

        let link_borrow = startblock.exits[0].borrow();
        assert_eq!(link_borrow.target, Some(catch_block));
        assert_eq!(link_borrow.exitcase, None);
        assert!(link_borrow.last_exception.is_some());
        assert!(link_borrow.last_exc_value.is_some());
    }

    #[test]
    fn attach_catch_exception_edge_materializes_exception_state_and_extravars() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let catch_block = graph.new_block(Vec::new());
        let catch_ref = SpamBlockRef::new(catch_block.clone(), None);
        let mut link_exit_states: HashMap<LinkRef, FrameState> = HashMap::new();
        let source_state = FrameState::new(
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())],
            Vec::new(),
            None,
            Vec::new(),
            0,
        );
        let startblock_ref = graph.startblock.clone();

        let link = attach_catch_exception_edge(
            &mut graph,
            &startblock_ref,
            &catch_ref,
            &source_state,
            &mut link_exit_states,
        );

        let link_borrow = link.borrow();
        assert!(link_borrow.last_exception.is_some());
        assert!(link_borrow.last_exc_value.is_some());
        drop(link_borrow);

        let catch_state = catch_ref
            .framestate()
            .expect("catch landing should acquire a FrameState");
        assert!(catch_state.last_exception.is_some());
        assert_eq!(link_exit_states.get(&link), Some(&source_state));
    }

    #[test]
    fn mergeblock_reuses_matching_joinpoint() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let current_state = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::none().into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            1,
        );
        let current_block =
            SpamBlockRef::new(graph.startblock.clone(), Some(current_state.clone()));
        let target_state = FrameState::new(
            current_state.locals_w.clone(),
            current_state.stack.clone(),
            current_state.last_exception.clone(),
            Vec::new(),
            1,
        );
        let target_block = SpamBlockRef::new(
            graph.new_block(target_state.getvariables()),
            Some(target_state),
        );
        let mut joinpoints: HashMap<usize, Vec<SpamBlockRef>> = HashMap::new();
        let mut pc_blocks = vec![None; code.instructions.len().max(2)];
        joinpoints.insert(1, vec![target_block.clone()]);
        let mut link_exit_states: HashMap<LinkRef, FrameState> = HashMap::new();

        let merged = mergeblock(
            &code,
            &mut graph,
            &mut joinpoints,
            &mut pc_blocks,
            &current_block,
            &current_state,
            1,
            &mut link_exit_states,
        );

        assert_eq!(merged, target_block);
        assert_eq!(pc_blocks[1], Some(target_block.clone()));
        let exits = current_block.block().borrow().exits.clone();
        assert_eq!(exits.len(), 1);
        let link = exits[0].borrow();
        assert_eq!(link.target, Some(target_block.block()));
        assert_eq!(
            link.args,
            vec![Some(Variable::new(VariableId(0), Kind::Ref).into())]
        );
    }

    #[test]
    fn mergeblock_generalizes_existing_joinpoint() {
        let code = first_nested_function_code("def f(a):\n    return a\n");
        let mut graph = new_shadow_graph(&code);
        let source_state = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::signed(7).into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            2,
        );
        let current_block = SpamBlockRef::new(graph.startblock.clone(), Some(source_state.clone()));
        let existing_state = FrameState::new(
            vec![
                Some(Variable::new(VariableId(0), Kind::Ref).into()),
                Some(Constant::none().into()),
            ],
            Vec::new(),
            None,
            Vec::new(),
            2,
        );
        let existing_block = SpamBlockRef::new(
            graph.new_block(existing_state.getvariables()),
            Some(existing_state),
        );
        let mut joinpoints: HashMap<usize, Vec<SpamBlockRef>> = HashMap::new();
        let mut pc_blocks = vec![None; code.instructions.len().max(3)];
        joinpoints.insert(2, vec![existing_block.clone()]);
        let mut link_exit_states: HashMap<LinkRef, FrameState> = HashMap::new();

        let merged = mergeblock(
            &code,
            &mut graph,
            &mut joinpoints,
            &mut pc_blocks,
            &current_block,
            &source_state,
            2,
            &mut link_exit_states,
        );

        assert_ne!(merged, existing_block);
        assert!(existing_block.dead());
        assert_eq!(pc_blocks[2], Some(merged.clone()));
        let merged_state = merged
            .framestate()
            .expect("merged block should keep framestate");
        match merged_state.locals_w[1].as_ref() {
            Some(FlowValue::Variable(_)) => {}
            other => panic!("expected generalized variable, got {other:?}"),
        }
        match merged_state.locals_w[0].as_ref() {
            Some(FlowValue::Variable(variable)) => assert!(variable.name().starts_with("a_")),
            other => panic!("expected renamed local variable, got {other:?}"),
        }
        let existing_ref = existing_block.block();
        let existing_borrow = existing_ref.borrow();
        assert_eq!(existing_borrow.exits.len(), 1);
        let forwarded = existing_borrow.exits[0].borrow();
        assert_eq!(forwarded.target, Some(merged.block()));
    }

    #[test]
    fn setup_jitdriver_dedups_by_runtime_code_identity() {
        let writer = CodeWriter::new();
        let code = pyre_interpreter::compile_exec("x = 1\n").expect("source must compile");
        let w_code = pyre_interpreter::box_code_constant(&code);
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code) as *const pyre_interpreter::CodeObject
        };

        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: &code as *const _,
            w_code: w_code as *const (),
            merge_point_pc: None,
        });
        writer.setup_jitdriver(crate::jit::call::JitDriverStaticData {
            portal_graph: raw_code,
            w_code: w_code as *const (),
            merge_point_pc: Some(7),
        });

        let cc = writer.callcontrol();
        assert_eq!(cc.jitdrivers_sd.len(), 1);
        assert_eq!(cc.jitdrivers_sd[0].portal_graph, raw_code);
        assert_eq!(cc.jitdrivers_sd[0].merge_point_pc, Some(7));
        assert_eq!(cc.jitdriver_sd_from_portal_graph(raw_code), Some(0));
    }
}
