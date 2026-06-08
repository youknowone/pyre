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
    /// An inline constant (`history.py` `Const*`); value identity, never
    /// `Rc::ptr_eq`.
    Const(Const),
    /// MIGRATION-ONLY catch-all (`#9` operand-union flip): a not-yet-converted
    /// [`BoxRef`] for the operands `from_boxref` cannot yet lower to a pure
    /// `Op`/`InputArg`/`Const` — const operands (kept `Cell`-backed for the GC
    /// walk) and position-only operands minted by `BoxRef::from_opref` (no live
    /// producer to bind). Holding the original `BoxRef` makes the storage flip
    /// byte-identical: `to_boxref` clones the same `Rc<Box>` back, preserving
    /// box identity and GC-walkability verbatim. Ground down to zero
    /// (then deleted) as later slices bind/inline these.
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

    /// An inline-constant operand.
    pub fn const_(value: Const) -> Operand {
        Operand::Const(value)
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
            Operand::Const(c) => match *c {
                Const::Int(v) => OpRef::const_int(v),
                Const::Float(v) => OpRef::const_float(v),
                Const::Ref(v) => OpRef::const_ptr(v),
            },
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
            Operand::Const(c) => c.get_type(),
            Operand::Box(b) => b.type_(),
            Operand::None => Type::Void,
        }
    }

    /// The inline constant value (`history.py:233` `Const.getint` family),
    /// `None` for non-`Const`.
    pub fn const_value(&self) -> Option<Value> {
        match self {
            Operand::Const(c) => Some(c.to_value()),
            Operand::Box(b) => b.const_value(),
            _ => None,
        }
    }

    /// Raw `ConstInt` value with no `IntBound` synthesis (`box_ref.rs:480`
    /// parity).
    pub fn const_int(&self) -> Option<i64> {
        match self {
            Operand::Const(Const::Int(v)) => Some(*v),
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
            Operand::Const(c) => BoxRef::new_const(c.to_value()),
            Operand::Box(b) => b.clone(),
        }
    }

    /// Classify a [`BoxRef`] into an [`Operand`] for the storage flip.
    ///
    /// This slice is strictly behavior-preserving, so it sheds only the
    /// identity-free `None` sentinel; every value-carrying box is kept verbatim
    /// as `Operand::Box`, whose `to_boxref` returns the same `Rc<Box>` —
    /// preserving box identity, the `Cell`-backed GC walk, AND the box's
    /// snapshot `position` field. The last point matters: a bound `ResOp`
    /// box's `to_opref` reads its own frozen `BoxKind::ResOp.position` Cell,
    /// NOT the producer's live `op.pos`, and the optimizer's position-remap
    /// (`optimizer.rs`) deliberately mutates `op.pos` before reading operand
    /// positions — so shedding a bound `ResOp` to a live-tracking
    /// `Operand::Op` here would double-remap. Shedding `Box` → `Op`/`InputArg`/
    /// `Const` (and adapting those remap consumers) is a later slice.
    pub fn from_boxref(b: &BoxRef) -> Operand {
        if b.is_none() {
            return Operand::None;
        }
        Operand::Box(b.clone())
    }

    /// GC walk over any inline `ConstPtr` reachable from this operand
    /// (`resoperation.py` `walk_const_ptr_refs`). Const / position-only
    /// operands are held `Cell`-backed via `Operand::Box`, so their `GcRef`
    /// updates in place; pure `Op` / `InputArg` carry no inline const (their
    /// own `value` slot is walked at the producer); pure `Operand::Const`
    /// is value-typed with no in-place slot and is not produced by
    /// `from_boxref` during the migration window.
    pub fn walk_const_ptr_refs(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        if let Operand::Box(b) = self {
            b.walk_const_ptr_refs(visitor);
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
        assert_eq!(Operand::from_bound_op(&op).to_opref(), OpRef::op_typed(3, Type::Int));

        let ia = Rc::new(InputArg::from_type(Type::Ref, 2));
        assert_eq!(
            Operand::from_bound_inputarg(&ia).to_opref(),
            OpRef::input_arg_typed(2, Type::Ref),
        );

        assert_eq!(Operand::const_(Const::Int(7)).to_opref(), OpRef::const_int(7));
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
    fn from_boxref_is_byte_identical_keeps_every_value_box() {
        // Behavior-preserving storage flip: every value-carrying box is kept
        // verbatim as Operand::Box, returning the identical Rc<Box> on read so
        // identity, type, position, and const value all round-trip exactly.

        // Bound op -> kept as Box (its frozen `position` snapshot is preserved,
        // not replaced by the producer's live op.pos).
        let op = op_at(8, Type::Int);
        let bound = BoxRef::from_bound_op(&op);
        let o = Operand::from_boxref(&bound);
        assert!(matches!(o, Operand::Box(_)));
        assert!(o.is_resop());
        assert_eq!(o.to_boxref().as_ptr(), bound.as_ptr());

        // Bound input arg -> kept as Box.
        let ia = Rc::new(InputArg::from_type(Type::Ref, 2));
        let bia = BoxRef::from_bound_inputarg(&ia);
        let o = Operand::from_boxref(&bia);
        assert!(matches!(o, Operand::Box(_)));
        assert!(o.is_inputarg());
        assert_eq!(o.to_boxref().as_ptr(), bia.as_ptr());

        // Const -> kept as Box (Cell-backed), round-trips to the same box.
        let cbox = BoxRef::new_const(Value::Int(11));
        let o = Operand::from_boxref(&cbox);
        assert!(matches!(o, Operand::Box(_)));
        assert!(o.is_constant());
        assert_eq!(o.const_int(), Some(11));
        assert_eq!(o.to_boxref().as_ptr(), cbox.as_ptr());

        // Position-only box (from_opref, no live producer) -> kept as Box,
        // returns the identical Rc<Box> on read.
        let pos_only = BoxRef::from_opref(OpRef::op_typed(4, Type::Int));
        let o = Operand::from_boxref(&pos_only);
        assert!(matches!(o, Operand::Box(_)));
        assert_eq!(o.position(), Some(4));
        assert_eq!(o.to_boxref().as_ptr(), pos_only.as_ptr());

        // None sentinel -> Operand::None (identity-free, byte-identical).
        assert!(matches!(Operand::from_boxref(&BoxRef::none()), Operand::None));
    }
}
