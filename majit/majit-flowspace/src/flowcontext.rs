//! Flow-context skeleton — the types `framestate.rs` needs before
//! Phase 3 F3.4 lands the full flow-context engine.
//!
//! RPython basis: `rpython/flowspace/flowcontext.py:1203-1390`.
//!
//! This file intentionally ships as a narrow skeleton for F3.1: only
//! the `FlowSignal` abstract-exception enum and the `FrameBlock` block
//! type, both consumed by `framestate::FrameState`.
//!
//! The upstream file is 1,404 LOC; the rest of it (`FlowContext`,
//! `SpamBlock`, `EggBlock`, `Recorder`, the full block-handler
//! hierarchy, opcode handlers) lands across F3.4 – F3.6 per the
//! roadmap.
//!
//! ## Deviation from upstream (parity rule #1)
//!
//! RPython's `FlowSignal` is an **abstract Python exception class**
//! whose concrete subclasses (`Return`, `Raise`, `RaiseImplicit`,
//! `Break`, `Continue`) carry per-variant state. Python's duck typing
//! lets `isinstance(x, FlowSignal)` accept any subclass; each subclass
//! owns its own `__init__`, `args` property, and `rebuild` factory.
//!
//! Rust has no subclass hierarchy; we close this as a single `enum
//! FlowSignal` with one variant per upstream subclass. The `args()`
//! method mirrors the `args` property across subclasses; the `rebuild`
//! classmethod family collapses into a single `rebuild_with_args`
//! associated function that takes a tag + fresh args.
//!
//! Similarly, `FrameBlock` is an abstract class with concrete
//! subclasses (`LoopBlock`, `ExceptBlock`, `FinallyBlock`,
//! `WithBlock`) that land in F3.5. F3.1 only needs the common shape
//! so `framestate::FrameState.blocklist` typechecks; we expose an
//! opaque `FrameBlock` struct holding the two fields every subclass
//! shares (`handlerposition`, `stackdepth`) plus a `kind` tag that
//! F3.5 fills in.

use crate::model::{ConstValue, Constant, FSException, Hlvalue};

#[cfg(test)]
use crate::model::Variable;

/// Abstract base for translator-level control-flow signals.
///
/// RPython basis: `flowcontext.py:1203-1314`. Concrete variants mirror
/// upstream subclasses in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FlowSignal {
    /// RPython `class Return(FlowSignal)` — argument is the wrapped
    /// return value.
    Return { w_value: Hlvalue },
    /// RPython `class Raise(FlowSignal)` — application-level exception.
    Raise { w_exc: FSException },
    /// RPython `class RaiseImplicit(Raise)` — exception raised
    /// implicitly.
    RaiseImplicit { w_exc: FSException },
    /// RPython `class Break(FlowSignal)` — singleton, no args.
    Break,
    /// RPython `class Continue(FlowSignal)` — bytecode position of the
    /// enclosing loop's start.
    Continue { jump_to: i64 },
}

impl FlowSignal {
    /// RPython `FlowSignal.args` property (dispatched across
    /// subclasses).
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

    /// RPython `FlowSignal.rebuild(*args)` dispatched across
    /// subclasses. Reconstructs the signal variant with fresh args
    /// after `_copy` / `union` remaps them.
    ///
    /// Panics if `args.len()` does not match the variant's arity, or
    /// if `Continue` receives a non-integer constant (matching
    /// upstream `Continue.rebuild` which calls `w_jump_to.value` and
    /// would raise `AttributeError` on a non-Constant).
    pub fn rebuild_with_args(tag: FlowSignalTag, args: Vec<Hlvalue>) -> FlowSignal {
        match tag {
            FlowSignalTag::Return => {
                assert_eq!(args.len(), 1, "Return.rebuild takes 1 arg");
                let mut it = args.into_iter();
                FlowSignal::Return {
                    w_value: it.next().unwrap(),
                }
            }
            FlowSignalTag::Raise => {
                assert_eq!(args.len(), 2, "Raise.rebuild takes 2 args");
                let mut it = args.into_iter();
                let w_type = it.next().unwrap();
                let w_value = it.next().unwrap();
                FlowSignal::Raise {
                    w_exc: FSException::new(w_type, w_value),
                }
            }
            FlowSignalTag::RaiseImplicit => {
                assert_eq!(args.len(), 2, "RaiseImplicit.rebuild takes 2 args");
                let mut it = args.into_iter();
                let w_type = it.next().unwrap();
                let w_value = it.next().unwrap();
                FlowSignal::RaiseImplicit {
                    w_exc: FSException::new(w_type, w_value),
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

    /// RPython-style variant tag used by `rebuild_with_args` and the
    /// `framestate.union()` typecheck ("type(w1) is not type(w2)").
    pub fn tag(&self) -> FlowSignalTag {
        match self {
            FlowSignal::Return { .. } => FlowSignalTag::Return,
            FlowSignal::Raise { .. } => FlowSignalTag::Raise,
            FlowSignal::RaiseImplicit { .. } => FlowSignalTag::RaiseImplicit,
            FlowSignal::Break => FlowSignalTag::Break,
            FlowSignal::Continue { .. } => FlowSignalTag::Continue,
        }
    }
}

/// Variant discriminator for `FlowSignal`. Corresponds to
/// `type(signal)` in upstream's `union()` isinstance / type-equality
/// dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FlowSignalTag {
    Return,
    Raise,
    RaiseImplicit,
    Break,
    Continue,
}

/// A block on the interpreter block stack (loop / except / finally /
/// with). `FrameState.blocklist` is a stack of these.
///
/// RPython basis: `flowcontext.py:1316-1339` — `class FrameBlock`.
/// The concrete subclasses (LoopBlock, ExceptBlock, FinallyBlock,
/// WithBlock) land in F3.5; F3.1 only needs the common shape for
/// `FrameState.blocklist` equality and copy.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FrameBlock {
    /// RPython `FrameBlock.handlerposition` — bytecode offset of the
    /// block's handler.
    pub handlerposition: i64,
    /// RPython `FrameBlock.stackdepth` — stack depth at SETUP_*.
    pub stackdepth: usize,
    /// F3.5 enriches this with a concrete subclass tag (Loop /
    /// Except / Finally / With). Left opaque for F3.1.
    pub kind: FrameBlockKind,
}

impl FrameBlock {
    /// RPython `FrameBlock.__init__(ctx, handlerposition)` — F3.1
    /// stub. Real construction through `ctx.stackdepth` lives in F3.5.
    pub fn new(handlerposition: i64, stackdepth: usize, kind: FrameBlockKind) -> Self {
        FrameBlock {
            handlerposition,
            stackdepth,
            kind,
        }
    }
}

/// Opaque tag for `FrameBlock`'s upstream subclass. F3.5 populates
/// the real variants (`LoopBlock`, `ExceptBlock`, `FinallyBlock`,
/// `WithBlock`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum FrameBlockKind {
    /// Placeholder — F3.1 has no way to construct a real
    /// subclass-tagged block since no opcode handlers exist yet.
    Unresolved,
}

#[cfg(test)]
mod test {
    use super::*;

    // RPython basis: no upstream test for FlowSignal in isolation —
    // `test_flowcontext.py` exercises it via FlowContext. These smoke
    // tests pin the shapes `framestate.rs` relies on.

    fn var() -> Hlvalue {
        Hlvalue::Variable(Variable::new())
    }

    fn iconst(n: i64) -> Hlvalue {
        Hlvalue::Constant(Constant::new(ConstValue::Int(n)))
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
    fn break_is_argless() {
        let sig = FlowSignal::Break;
        assert!(sig.args().is_empty());
        let rebuilt = FlowSignal::rebuild_with_args(FlowSignalTag::Break, Vec::new());
        assert_eq!(rebuilt, sig);
    }

    #[test]
    fn continue_wraps_jump_offset() {
        let sig = FlowSignal::Continue { jump_to: 42 };
        assert_eq!(sig.args(), vec![iconst(42)]);
        let rebuilt = FlowSignal::rebuild_with_args(FlowSignalTag::Continue, sig.args());
        assert_eq!(rebuilt, sig);
    }

    #[test]
    fn tag_discriminates_variants() {
        assert_eq!(
            FlowSignal::Return { w_value: var() }.tag(),
            FlowSignalTag::Return
        );
        assert_eq!(FlowSignal::Break.tag(), FlowSignalTag::Break);
    }
}
