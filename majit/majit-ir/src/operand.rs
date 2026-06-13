//! `Operand` — the operand-union successor to [`BoxRef`] for `Op.args` /
//! `Op.fail_args` (#9 / S-11).
//!
//! `resoperation.py:281` `N_aryOp._args` stores operands as the
//! `AbstractValue` objects themselves — a result op, an input arg, or a
//! constant — with no integer-position indirection. `Operand` is the Rust
//! shape of that: a strong-ref union carrying the producer directly, so
//! operand identity is `Rc::ptr_eq` and forwarding reads straight off the
//! carried producer's `_forwarded` slot — with no `find_producer_op`
//! position→producer registry, no `Op::box_cache` memoization, and no
//! `BoxRef::from_opref` position fabrication.
//!
//! Strong `Rc` (not the `Weak` of [`Forwarded`](crate::box_ref::Forwarded)):
//! operands must keep their producers alive. The trace already holds the 1st
//! strong ref in `Trace.ops: Vec<OpRc>` (#103); an operand `Rc<Op>` is a 2nd
//! strong ref on the acyclic SSA use-before-def DAG (operands reference
//! predecessors only), so no `Rc` cycle can form.
//!
//! This module is the #9 foundation. `Op.args` still carries [`BoxRef`] until
//! the storage flip; [`Operand::to_boxref`] and the `from_bound_*`
//! constructors let the two representations coexist during the migration.

use crate::box_ref::BoxRef;
use crate::resoperation::{OpRc, OpRef};
use crate::value::{Const, GcRef, InputArgRc, Type, Value};
use std::rc::Rc;

/// An operand stored in `Op.args` / `Op.fail_args`.
///
/// Mirror of `OpRef`'s four logical cases, but carrying the producer by
/// strong `Rc` instead of a flat position: `Op` ⇆ `OpRef::*Op`, `InputArg`
/// ⇆ `OpRef::InputArg*`, `Const` ⇆ the inline `OpRef::Const*`, and `None` ⇆
/// `OpRef::None` (an absent `fail_args` slot).
#[derive(Clone, Debug)]
pub enum Operand {
    /// Absent slot — the mirror of `OpRef::None`.
    None,
    /// A result-op producer (`resoperation.py` `AbstractResOp`).
    Op(OpRc),
    /// An input-arg producer (`resoperation.py` `AbstractInputArg`).
    InputArg(InputArgRc),
    /// A constant (`history.py:227/268/314` `ConstInt`/`ConstFloat`/
    /// `ConstPtr`). Upstream `Const` is a heap object — `_args[i]` returns
    /// the stored object, `==` is Python `is`, and value equality is the
    /// explicit `same_constant` (history.py:211) — so the operand carries
    /// the const box by `Rc` (object identity across reads of one slot,
    /// `Cell`-backed GC walk of an inline `ConstPtr`), NOT a bare value.
    /// The carrier is the const-kind [`BoxRef`] for the migration window;
    /// it type-narrows to a dedicated const-box `Rc` when the `BoxRef`
    /// wrapper is deleted (#9 endgame).
    Const(BoxRef),
    /// MIGRATION-ONLY catch-all (`#9` operand-union flip): a not-yet-converted
    /// [`BoxRef`] for the operands `from_boxref` cannot yet lower to a pure
    /// `Op`/`InputArg`/`Const` — position-only operands minted by
    /// `BoxRef::from_opref` (no live producer to bind). Holding the original
    /// `BoxRef` makes the storage flip byte-identical: `to_boxref` clones the
    /// same `Rc<Box>` back, preserving box identity and GC-walkability
    /// verbatim. Ground down to zero (then deleted) as later slices bind
    /// these.
    Box(BoxRef),
}

impl Operand {
    /// Wrap a bound op as `Operand::Op` (`Rc::clone`, cheap). The successor
    /// to [`BoxRef::from_bound_op`] — no `box_cache` memoization, the `Rc`
    /// itself IS the stable identity.
    pub fn from_bound_op(op: &OpRc) -> Operand {
        Operand::Op(Rc::clone(op))
    }

    /// Wrap a bound input arg as `Operand::InputArg` (`Rc::clone`). Successor
    /// to [`BoxRef::from_bound_inputarg`].
    pub fn from_bound_inputarg(ia: &InputArgRc) -> Operand {
        Operand::InputArg(Rc::clone(ia))
    }

    /// A constant operand — mints a fresh const box (`history.py:227`
    /// `ConstInt(value)` object construction; identity starts here and is
    /// shared by every read of the slot).
    pub fn const_(value: Const) -> Operand {
        Operand::Const(BoxRef::new_const(value.to_value()))
    }

    /// The absent-slot sentinel.
    pub fn none() -> Operand {
        Operand::None
    }

    /// Flat-`OpRef` view for the OpRef-keyed side tables, `op.pos`
    /// comparisons, and backend/gc encoding (`box_ref.rs:494` parity). This
    /// is the PERMANENT handoff boundary where the optimizer's operand
    /// identity converts to the backend's `OpRef` encoding; it is
    /// re-expressed, never retired. An `Op` reads its (post-compaction)
    /// position straight off `op.pos`; a `Const*` maps to the matching inline
    /// `OpRef` (`history.py:227/268/314`).
    pub fn to_opref(&self) -> OpRef {
        match self {
            Operand::None => OpRef::NONE,
            Operand::Op(op) => op.pos.get(),
            Operand::InputArg(ia) => OpRef::input_arg_typed(ia.index, ia.tp),
            // Re-encodes from the box's live `Cell` value, so a GC-moved
            // `ConstPtr` reads back at its post-move address.
            Operand::Const(b) => b.to_opref(),
            Operand::Box(b) => b.to_opref(),
        }
    }

    /// `resoperation.py:233 _pos` accessor: the pool index for `Op` /
    /// `InputArg`; `Const` / `None` have no canonical position.
    pub fn position(&self) -> Option<u32> {
        match self {
            Operand::Op(op) => Some(op.pos.get().raw()),
            Operand::InputArg(ia) => Some(ia.index),
            Operand::Box(b) => b.position(),
            Operand::Const(_) | Operand::None => None,
        }
    }

    /// The operand's `Type` (`Int` / `Float` / `Ref` / `Void`).
    pub fn type_(&self) -> Type {
        match self {
            Operand::Op(op) => op.pos.get().ty().unwrap_or(Type::Void),
            Operand::InputArg(ia) => ia.tp,
            Operand::Const(b) => b.type_(),
            Operand::Box(b) => b.type_(),
            Operand::None => Type::Void,
        }
    }

    /// The inline constant value (`history.py:233` `Const.getint` family),
    /// `None` for non-`Const`.
    pub fn const_value(&self) -> Option<Value> {
        match self {
            Operand::Const(b) => b.const_value(),
            Operand::Box(b) => b.const_value(),
            _ => None,
        }
    }

    /// Raw `ConstInt` value with no `IntBound` synthesis (`box_ref.rs:480`
    /// parity).
    pub fn const_int(&self) -> Option<i64> {
        match self {
            Operand::Const(b) => b.const_int(),
            Operand::Box(b) => b.const_int(),
            _ => None,
        }
    }

    /// `resoperation.py:47 is_constant`.
    pub fn is_constant(&self) -> bool {
        match self {
            Operand::Const(_) => true,
            Operand::Box(b) => b.is_constant(),
            _ => false,
        }
    }

    pub fn is_inputarg(&self) -> bool {
        match self {
            Operand::InputArg(_) => true,
            Operand::Box(b) => b.is_inputarg(),
            _ => false,
        }
    }

    pub fn is_resop(&self) -> bool {
        match self {
            Operand::Op(_) => true,
            Operand::Box(b) => b.is_resop(),
            _ => false,
        }
    }

    /// True for the absent-slot sentinel — the mirror of `OpRef::is_none`.
    pub fn is_none(&self) -> bool {
        match self {
            Operand::None => true,
            Operand::Box(b) => b.is_none(),
            _ => false,
        }
    }

    /// `resoperation.py:38 AbstractValue.same_box`: pointer identity
    /// (`Rc::ptr_eq`) for `Op` / `InputArg`, value comparison for `Const`
    /// (`history.py:211 same_constant`), and the `None` sentinel matches only
    /// itself. Routed through [`BoxRef::same_box`] (the canonical predicate) so
    /// the migration `Box` variant compares uniformly against pure variants —
    /// the `from_bound_*` view is memoized, so two operands holding the same op
    /// still resolve to the same `Rc<Box>` and stay `ptr_eq`.
    pub fn same_box(&self, other: &Operand) -> bool {
        self.to_boxref().same_box(&other.to_boxref())
    }

    /// Faithful [`BoxRef`] view of this operand, for the migration window
    /// while `Op.args` still carries `BoxRef`. `Op` / `InputArg` route
    /// through the memoizing `from_bound_*` (so the view round-trips to the
    /// SAME `Rc<Box>`); `Const` mints an inline const box; `None` is the
    /// sentinel. The inverse of the `from_bound_*` constructors modulo the
    /// `BoxRef` wrapper.
    pub fn to_boxref(&self) -> BoxRef {
        match self {
            Operand::None => BoxRef::none(),
            Operand::Op(op) => BoxRef::from_bound_op(op),
            Operand::InputArg(ia) => BoxRef::from_bound_inputarg(ia),
            // The SAME `Rc<Box>` back — `_args[i]` returns the stored
            // Const object, never a fresh equal-valued one.
            Operand::Const(b) => b.clone(),
            Operand::Box(b) => b.clone(),
        }
    }

    /// Classify a [`BoxRef`] into an [`Operand`] for storage.
    ///
    /// A genuinely-bound box sheds to its live-tracking producer `Rc`
    /// (`Operand::Op` / `Operand::InputArg`) — the operand IS the producer
    /// (`resoperation.py` `N_aryOp._args` holds the `AbstractResOp` /
    /// `AbstractInputArg` directly). Its `to_opref` then reads the producer's
    /// live `op.pos`, so renumbering the producer auto-propagates without a
    /// snapshot rewrite. The two position-remap passes
    /// (`optimizer.rs` `new_operations` / `exported_short_boxes`) mutate
    /// `op.pos` and must therefore SKIP bound operands
    /// ([`Operand::is_bound`]) and rewrite only position-only snapshots —
    /// otherwise a bound operand reading the already-remapped live pos would
    /// double-remap.
    ///
    /// A position-only box (no bound handle — e.g. `from_opref`) and a Const
    /// box (value-typed, GC-walked in place through its `Cell`) carry no
    /// producer to track and stay `Operand::Box`, whose `to_boxref` returns
    /// the same `Rc<Box>` (identity, `Cell`-backed GC walk, frozen position
    /// snapshot all preserved).
    pub fn from_boxref(b: &BoxRef) -> Operand {
        if b.is_none() {
            return Operand::None;
        }
        // Shed a genuinely-bound box to its live-tracking producer `Rc`: the
        // operand IS the producer (resoperation.py `N_aryOp._args` holds the
        // `AbstractResOp`/`AbstractInputArg` directly), so its position
        // auto-tracks the producer's `op.pos` and its forwarding resolves
        // through the canonical `Op`/`InputArg`. The strong `Rc` keeps the
        // producer alive (acyclic on the SSA use-before-def DAG). A
        // position-only box (no bound handle) and a Const box (value-typed,
        // GC-walked in place via its `Cell`) have no producer to carry and
        // stay `Operand::Box`.
        if let Some(op) = b.bound_op() {
            return Operand::Op(op);
        }
        if let Some(ia) = b.bound_inputarg() {
            return Operand::InputArg(ia);
        }
        // A Const box lowers to the terminal `Operand::Const` carrying the
        // same `Rc<Box>` (history.py Const object identity + `Cell`-backed
        // GC walk preserved). Only position-only boxes remain `Operand::Box`.
        if b.is_constant() {
            return Operand::Const(b.clone());
        }
        // Only position-only boxes remain `Operand::Box`. The optimizer
        // grind (#9-④) reduced production mints to the S9 cross-phase
        // residual: export-boundary `get_box_replacement` fallbacks (which
        // carry their own producerless tripwire) and unmapped Phase-1
        // short-preamble jump args. Both are producer-less by construction
        // in the current context.
        Operand::Box(b.clone())
    }

    /// True only for the live-tracking bound variants (`Op` / `InputArg`),
    /// whose `to_opref()` reads the producer's CURRENT `op.pos`. Excludes the
    /// frozen `Operand::Box` snapshot even when it wraps a ResOp / InputArg
    /// box — unlike [`Operand::is_resop`] / [`Operand::is_inputarg`], which
    /// fold the snapshot case in. The position-remap passes use this to skip
    /// operands that auto-track a renumbered producer (no snapshot rewrite
    /// needed); only position-only `Operand::Box` operands carry a stale
    /// position the remap table must rewrite.
    pub fn is_bound(&self) -> bool {
        matches!(self, Operand::Op(_) | Operand::InputArg(_))
    }

    /// GC walk over any inline `ConstPtr` reachable from this operand
    /// (`resoperation.py` `walk_const_ptr_refs`). Const / position-only
    /// operands are held `Cell`-backed in their box, so their `GcRef`
    /// updates in place; pure `Op` / `InputArg` carry no inline const (their
    /// own `value` slot is walked at the producer).
    pub fn walk_const_ptr_refs(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        match self {
            Operand::Const(b) | Operand::Box(b) => b.walk_const_ptr_refs(visitor),
            Operand::None | Operand::Op(_) | Operand::InputArg(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resoperation::{Op, OpCode};
    use crate::value::{Const, InputArg, Type, Value};

    fn op_at(pos: u32, tp: Type) -> OpRc {
        let op = Rc::new(Op::new(OpCode::SameAsI, &[]));
        op.pos.set(OpRef::op_typed(pos, tp));
        op
    }

    #[test]
    fn to_opref_round_trips_each_variant() {
        let op = op_at(3, Type::Int);
        assert_eq!(
            Operand::from_bound_op(&op).to_opref(),
            OpRef::op_typed(3, Type::Int)
        );

        let ia = Rc::new(InputArg::from_type(Type::Ref, 2));
        assert_eq!(
            Operand::from_bound_inputarg(&ia).to_opref(),
            OpRef::input_arg_typed(2, Type::Ref),
        );

        assert_eq!(
            Operand::const_(Const::Int(7)).to_opref(),
            OpRef::const_int(7)
        );
        assert_eq!(Operand::none().to_opref(), OpRef::NONE);
    }

    #[test]
    fn accessors_match_variant() {
        let op = op_at(5, Type::Int);
        let o_op = Operand::from_bound_op(&op);
        assert!(o_op.is_resop());
        assert_eq!(o_op.position(), Some(5));
        assert_eq!(o_op.type_(), Type::Int);
        assert_eq!(o_op.const_value(), None);

        let ia = Rc::new(InputArg::from_type(Type::Float, 1));
        let o_ia = Operand::from_bound_inputarg(&ia);
        assert!(o_ia.is_inputarg());
        assert_eq!(o_ia.position(), Some(1));
        assert_eq!(o_ia.type_(), Type::Float);

        let o_c = Operand::const_(Const::Int(9));
        assert!(o_c.is_constant());
        assert_eq!(o_c.position(), None);
        assert_eq!(o_c.type_(), Type::Int);
        assert_eq!(o_c.const_value(), Some(Value::Int(9)));
        assert_eq!(o_c.const_int(), Some(9));

        let o_n = Operand::none();
        assert!(o_n.is_none());
        assert_eq!(o_n.position(), None);
        assert_eq!(o_n.type_(), Type::Void);
    }

    #[test]
    fn same_box_is_pointer_identity_for_producers_value_for_const() {
        let op = op_at(0, Type::Int);
        // Same Rc -> same box.
        assert!(Operand::from_bound_op(&op).same_box(&Operand::from_bound_op(&op)));
        // Distinct ops at the same position -> distinct boxes.
        let op_other = op_at(0, Type::Int);
        assert!(!Operand::from_bound_op(&op).same_box(&Operand::from_bound_op(&op_other)));

        // Equal-valued constants -> same box (value identity).
        assert!(Operand::const_(Const::Int(4)).same_box(&Operand::const_(Const::Int(4))));
        assert!(!Operand::const_(Const::Int(4)).same_box(&Operand::const_(Const::Int(5))));

        // None matches only None.
        assert!(Operand::none().same_box(&Operand::none()));
        assert!(!Operand::none().same_box(&Operand::const_(Const::Int(0))));
    }

    #[test]
    fn to_boxref_preserves_identity_and_value() {
        let op = op_at(8, Type::Int);
        let via_operand = Operand::from_bound_op(&op).to_boxref();
        // The BoxRef bridge round-trips to the SAME memoized Rc<Box> as a
        // direct from_bound_op (box_cache identity).
        assert_eq!(via_operand.as_ptr(), BoxRef::from_bound_op(&op).as_ptr());
        assert_eq!(via_operand.to_opref(), OpRef::op_typed(8, Type::Int));

        let c = Operand::const_(Const::Int(11)).to_boxref();
        assert_eq!(c.const_int(), Some(11));

        assert!(Operand::none().to_boxref().is_none());
    }

    #[test]
    fn from_boxref_sheds_bound_keeps_const_and_position_only() {
        // Bound op -> Operand::Op (live-tracking). is_resop stays true;
        // is_bound is true. to_boxref re-resolves through the memoized
        // from_bound_op, so the canonical box is ptr-equal to the original.
        let op = op_at(8, Type::Int);
        let bound = BoxRef::from_bound_op(&op);
        let o = Operand::from_boxref(&bound);
        assert!(matches!(o, Operand::Op(_)));
        assert!(o.is_resop());
        assert!(o.is_bound());
        assert_eq!(o.to_boxref().as_ptr(), bound.as_ptr());

        // Bound input arg -> Operand::InputArg (live-tracking, bound).
        let ia = Rc::new(InputArg::from_type(Type::Ref, 2));
        let bia = BoxRef::from_bound_inputarg(&ia);
        let o = Operand::from_boxref(&bia);
        assert!(matches!(o, Operand::InputArg(_)));
        assert!(o.is_inputarg());
        assert!(o.is_bound());
        assert_eq!(o.to_boxref().as_ptr(), bia.as_ptr());

        // Const -> Operand::Const carrying the SAME const box (history.py
        // Const object identity; Cell-backed GC walk); NOT bound.
        let cbox = BoxRef::new_const(Value::Int(11));
        let o = Operand::from_boxref(&cbox);
        assert!(matches!(o, Operand::Const(_)));
        assert!(o.is_constant());
        assert!(!o.is_bound());
        assert_eq!(o.const_int(), Some(11));
        assert_eq!(o.to_boxref().as_ptr(), cbox.as_ptr());

        // Position-only box (from_opref, no live producer) -> kept as Box
        // (no bound handle to shed onto); NOT bound, frozen position survives.
        let pos_only = BoxRef::from_opref(OpRef::op_typed(4, Type::Int));
        let o = Operand::from_boxref(&pos_only);
        assert!(matches!(o, Operand::Box(_)));
        assert!(!o.is_bound());
        assert_eq!(o.position(), Some(4));
        assert_eq!(o.to_boxref().as_ptr(), pos_only.as_ptr());

        // None sentinel -> Operand::None.
        assert!(matches!(
            Operand::from_boxref(&BoxRef::none()),
            Operand::None
        ));
    }
}
