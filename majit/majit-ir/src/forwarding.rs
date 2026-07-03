//! `_forwarded` slot mirror of RPython's `AbstractResOpOrInputArg`.
//!
//! Direct port of the shared forwarding slot from
//! `rpython/jit/metainterp/resoperation.py AbstractResOpOrInputArg`,
//! carried on `Op` / `InputArg` themselves (`resoperation.py`).
//! The optimizer holds producer identities as [`crate::operand::Operand`];
//! this module hosts the [`Forwarded`] slot variant, the [`ForwardingHost`]
//! trait that exposes the `get_forwarded` / `set_forwarded_*` /
//! `ptr_info` / `int_bound` projections, and the borrow-guard wrappers those
//! projections hand back.
//!
//! Hosted in `majit-ir` so the slot can carry `Weak<Op>` / `Weak<InputArg>`
//! without a `majit-metainterp -> majit-ir` circular dep.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use crate::intbound::IntBound;
use crate::op_info::OpInfo;
use crate::ptr_info::PtrInfo;
use crate::resoperation::Op;
use crate::value::{Const, InputArg};
use crate::{OpRef, Type};

/// Variant of the `_forwarded` slot.
///
/// `Const` is an `AbstractValue` subclass too (`history.py
/// ConstInt`), so forwarding to a constant is its own shape: `Const`
/// is a value-typed `Copy` payload with no `_forwarded` slot of its
/// own, unlike `ResOp`/`InputArg`. Keeping it as a separate variant
/// retires a dedicated const-as-chain-target carrier.
#[derive(Clone, Debug)]
pub enum Forwarded {
    None,

    /// `resoperation.py AbstractResOp` forwarding — direct
    /// `Weak<Op>` reference. The chain walker upgrades the `Weak` into a
    /// producer-bound `Operand::Op` (via `Operand::from_bound_op`) and
    /// continues from there. A dropped `Weak` terminates the chain at the predecessor
    /// (PyPy never observes a dropped target — RPython keeps the
    /// underlying object alive through the trace `operations` list).
    Op(Weak<Op>),

    /// `resoperation.py AbstractInputArg` forwarding — direct
    /// `Weak<InputArg>` reference. Same chain-walk semantics as `Op`
    /// (producer-bound `Operand::from_bound_inputarg` materialization). RPython
    /// uses this for inputarg→inputarg redirects in bridge import and
    /// retrace remap (compile.py / unroll.py).
    InputArg(Weak<InputArg>),

    /// `history.py ConstInt` / `ConstFloat` / `ConstPtr`
    /// — forwarding terminates here; the constant value is carried
    /// inline. Chain walkers stop on this variant (`not_const=true`
    /// returns the pre-Const box; `not_const=false` materializes a
    /// terminal const-bearing operand).
    ///
    /// PyPy has no analog (callers hold the Python `Const` object
    /// directly); `box_to_opref` reconstructs the inline-Const OpRef
    /// from the payload value (history.py/268/314).
    Const(Const),

    /// `optimizeopt/info.py AbstractInfo (is_info_class = True)` family —
    /// `PtrInfo`, `IntBound`, `FloatConstInfo`, `EmptyInfo`, etc.
    Info(OpInfo),
    // No `VectorInfo` variant here yet — PRE-EXISTING-ADAPTATION, not parity.
    // RPython attaches vectorizer scratch to the op itself:
    // `op.set_forwarded(VectorizationInfo(op))` (`vector.py
    // setup_vectorization`, read back by `schedule.py forwarded_vecinfo`),
    // and re-propagates it across its SINGLE clone path `copy_resop`
    // (`vector.py`), which COPIES the already-resolved struct — INT_SIGNEXT's
    // arg1 bytesize is resolved once at setup time (`resoperation.py`) and
    // never recomputed on clone. So the INT_SIGNEXT dynamic-arg concern argues FOR
    // attach-and-copy, not against it. pyre instead keys the scratch in the
    // OpRef-keyed `VecScheduleState.vecinfo_cache` (optimizeopt/schedule.rs)
    // because it has no `copy_resop` analog: `Op::clone` resets `forwarded` to
    // `None` (resoperation.rs) and `DependencyGraph::build` clones ops by value
    // into `Node`s (optimizeopt/dependency.rs) where RPython shares the op
    // reference, so a `_forwarded`-borne vecinfo would clone-drop. Convergence
    // path: add a `Forwarded::VectorInfo` variant + a `copy_resop`-equivalent that
    // re-attaches it at every vectorizer clone site (`DependencyGraph::build` and
    // the unroll paths), keeping the const resolver only at the single setup-time
    // INT_SIGNEXT stamp. That touches the shared `_forwarded` core (GC-adjacent)
    // and the vectorizer is off by default, so it needs x86_64 + vectorizer-on
    // validation before landing. `Op.vecinfo` (resoperation.rs) is the SEPARATE
    // permanent `resoperation.py` VecOp datatype/bytesize/signed/count
    // store and stays.
}

/// `resoperation.py AbstractResOpOrInputArg` — the shared `_forwarded`
/// host. Both `Op` (`AbstractResOp`, resoperation.py) and `InputArg`
/// (`AbstractInputArg`, resoperation.py) carry a
/// `forwarded: RefCell<Forwarded>` slot and inherit `get_forwarded` /
/// `set_forwarded` from this base class; the Rust mirror is a shared trait
/// whose one required method exposes that slot. The `ptr_info` / `int_bound`
/// readers project the `Forwarded::Info` payload (the
/// `optimizer.py getptrinfo` / `getintbound` reads of
/// `box.get_forwarded()`), re-homed here so production can read forwarding
/// state straight off a producer identity.
///
/// `Operand`'s same-named methods route through its carried `Op` /
/// `InputArg` to these impls — the canonical forwarding logic lives on the
/// bound `Op` / `InputArg`.
pub trait ForwardingHost {
    /// The canonical `_forwarded` slot (`resoperation.py`).
    fn forwarded_cell(&self) -> &RefCell<Forwarded>;

    /// Pointer-identity probes backing the `resoperation.py
    /// assert forwarded_to is not self` self-cycle guard. A different
    /// concrete type can never be `self`, so the cross-type default is
    /// `false`; each host overrides only its own-type probe.
    fn is_same_op(&self, _op: &crate::resoperation::OpRc) -> bool {
        false
    }
    fn is_same_inputarg(&self, _ia: &crate::value::InputArgRc) -> bool {
        false
    }

    /// `resoperation.py get_forwarded` — clone the slot.
    fn get_forwarded(&self) -> Forwarded {
        self.forwarded_cell().borrow().clone()
    }

    /// `resoperation.py self._forwarded = forwarded_to` — the slot write
    /// shared by every typed setter. Prefer the typed `set_forwarded_*`,
    /// which carry the self-cycle assert.
    fn store_forwarded(&self, value: Forwarded) {
        *self.forwarded_cell().borrow_mut() = value;
    }

    /// `optimizer.py op.set_forwarded(newop)` — Op target.
    fn set_forwarded_op(&self, target: &crate::resoperation::OpRc) {
        assert!(
            !self.is_same_op(target),
            "set_forwarded_op on the same Op creates a one-node chain cycle"
        );
        self.store_forwarded(Forwarded::Op(Rc::downgrade(target)));
    }

    /// `compile.py` / `unroll.py` InputArg→InputArg redirect.
    fn set_forwarded_inputarg(&self, target: &crate::value::InputArgRc) {
        assert!(
            !self.is_same_inputarg(target),
            "set_forwarded_inputarg on the same InputArg creates a one-node \
             chain cycle"
        );
        self.store_forwarded(Forwarded::InputArg(Rc::downgrade(target)));
    }

    /// `optimizer.py make_constant(box, constbox)` — terminate the chain
    /// in an inline constant value.
    fn set_forwarded_const(&self, value: Const) {
        self.store_forwarded(Forwarded::Const(value));
    }

    /// `resoperation.py set_forwarded(forwarded_to)` — Info target.
    fn set_forwarded_info(&self, info: OpInfo) {
        self.store_forwarded(Forwarded::Info(info));
    }

    /// `_forwarded = None` (optimizer state reset).
    fn clear_forwarded(&self) {
        self.store_forwarded(Forwarded::None);
    }

    /// `optimizer.py getptrinfo` — project a `Forwarded::Info(Ptr)`
    /// into a shared borrow guard. Other states yield `None`. Does not walk
    /// the chain; the caller advances to the terminal identity first.
    fn ptr_info(&self) -> Option<PtrInfoBorrow> {
        match self.get_forwarded() {
            Forwarded::Info(OpInfo::Ptr(rc)) => Some(PtrInfoBorrow::new(rc)),
            _ => None,
        }
    }

    /// Live `Rc<RefCell<PtrInfo>>` handle (for `Rc::ptr_eq` identity / handoff).
    fn ptr_info_handle(&self) -> Option<Rc<std::cell::RefCell<PtrInfo>>> {
        match self.get_forwarded() {
            Forwarded::Info(OpInfo::Ptr(rc)) => Some(rc),
            _ => None,
        }
    }

    /// Mutable counterpart of `ptr_info`.
    fn ptr_info_mut(&self) -> Option<PtrInfoBorrowMut> {
        match self.get_forwarded() {
            Forwarded::Info(OpInfo::Ptr(rc)) => Some(PtrInfoBorrowMut::new(rc)),
            _ => None,
        }
    }

    /// `optimizer.py getintbound`.
    fn int_bound(&self) -> Option<IntBoundBorrow> {
        match self.get_forwarded() {
            Forwarded::Info(OpInfo::IntBound(rc)) => Some(IntBoundBorrow::new(rc)),
            _ => None,
        }
    }

    /// Live `Rc<RefCell<IntBound>>` handle.
    fn int_bound_handle(&self) -> Option<Rc<std::cell::RefCell<IntBound>>> {
        match self.get_forwarded() {
            Forwarded::Info(OpInfo::IntBound(rc)) => Some(rc),
            _ => None,
        }
    }

    /// Mutable counterpart of `int_bound`.
    fn int_bound_mut(&self) -> Option<IntBoundBorrowMut> {
        match self.get_forwarded() {
            Forwarded::Info(OpInfo::IntBound(rc)) => Some(IntBoundBorrowMut::new(rc)),
            _ => None,
        }
    }
}

impl ForwardingHost for Op {
    fn forwarded_cell(&self) -> &RefCell<Forwarded> {
        &self.forwarded
    }
    fn is_same_op(&self, op: &crate::resoperation::OpRc) -> bool {
        std::ptr::eq(self, Rc::as_ptr(op))
    }
}

impl ForwardingHost for InputArg {
    fn forwarded_cell(&self) -> &RefCell<Forwarded> {
        &self.forwarded
    }
    fn is_same_inputarg(&self, ia: &crate::value::InputArgRc) -> bool {
        std::ptr::eq(self, Rc::as_ptr(ia))
    }
}

pub struct PtrInfoBorrow {
    inner: std::cell::Ref<'static, PtrInfo>,
    _rc: Rc<std::cell::RefCell<PtrInfo>>,
}

impl PtrInfoBorrow {
    pub(crate) fn new(rc: Rc<std::cell::RefCell<PtrInfo>>) -> Self {
        // SAFETY: see struct doc — _rc keeps the RefCell allocation
        // alive for at least as long as Self exists.
        let r: std::cell::Ref<'_, PtrInfo> = rc.borrow();
        let r: std::cell::Ref<'static, PtrInfo> = unsafe { std::mem::transmute(r) };
        Self { inner: r, _rc: rc }
    }
}

impl std::ops::Deref for PtrInfoBorrow {
    type Target = PtrInfo;
    fn deref(&self) -> &PtrInfo {
        &self.inner
    }
}

impl std::fmt::Debug for PtrInfoBorrow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&*self.inner, f)
    }
}

/// Mutable counterpart of `PtrInfoBorrow`.  Holds the inner `RefCell`
/// exclusive borrow; conflicts with concurrent `PtrInfoBorrow` /
/// `PtrInfoBorrowMut` on the same handle panic at runtime per
/// `RefCell` semantics.
pub struct PtrInfoBorrowMut {
    inner: std::cell::RefMut<'static, PtrInfo>,
    _rc: Rc<std::cell::RefCell<PtrInfo>>,
}

impl PtrInfoBorrowMut {
    pub(crate) fn new(rc: Rc<std::cell::RefCell<PtrInfo>>) -> Self {
        // SAFETY: see `PtrInfoBorrow::new`.
        let r: std::cell::RefMut<'_, PtrInfo> = rc.borrow_mut();
        let r: std::cell::RefMut<'static, PtrInfo> = unsafe { std::mem::transmute(r) };
        Self { inner: r, _rc: rc }
    }
}

impl std::ops::Deref for PtrInfoBorrowMut {
    type Target = PtrInfo;
    fn deref(&self) -> &PtrInfo {
        &self.inner
    }
}

impl std::ops::DerefMut for PtrInfoBorrowMut {
    fn deref_mut(&mut self) -> &mut PtrInfo {
        &mut self.inner
    }
}

/// Owning borrow guard for `int_bound()`.  Same shape as
/// `PtrInfoBorrow` but parameterised on `IntBound`.
pub struct IntBoundBorrow {
    inner: std::cell::Ref<'static, IntBound>,
    _rc: Rc<std::cell::RefCell<IntBound>>,
}

impl IntBoundBorrow {
    pub(crate) fn new(rc: Rc<std::cell::RefCell<IntBound>>) -> Self {
        let r: std::cell::Ref<'_, IntBound> = rc.borrow();
        let r: std::cell::Ref<'static, IntBound> = unsafe { std::mem::transmute(r) };
        Self { inner: r, _rc: rc }
    }
}

impl std::ops::Deref for IntBoundBorrow {
    type Target = IntBound;
    fn deref(&self) -> &IntBound {
        &self.inner
    }
}

impl std::fmt::Debug for IntBoundBorrow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&*self.inner, f)
    }
}

pub struct IntBoundBorrowMut {
    inner: std::cell::RefMut<'static, IntBound>,
    _rc: Rc<std::cell::RefCell<IntBound>>,
}

impl IntBoundBorrowMut {
    pub(crate) fn new(rc: Rc<std::cell::RefCell<IntBound>>) -> Self {
        let r: std::cell::RefMut<'_, IntBound> = rc.borrow_mut();
        let r: std::cell::RefMut<'static, IntBound> = unsafe { std::mem::transmute(r) };
        Self { inner: r, _rc: rc }
    }
}

impl std::ops::Deref for IntBoundBorrowMut {
    type Target = IntBound;
    fn deref(&self) -> &IntBound {
        &self.inner
    }
}

impl std::ops::DerefMut for IntBoundBorrowMut {
    fn deref_mut(&mut self) -> &mut IntBound {
        &mut self.inner
    }
}

/// Turn an `OpRef` into a **bound** [`Operand`](crate::operand::Operand) for
/// op-argument / fail-arg fixtures: `None` / `Const` shed inline, an
/// `InputArg` / `ResOp` position binds to a freshly-minted synthetic producer
/// (`Operand::Op` / `Operand::InputArg`) carrying the same `pos`. The returned
/// operand holds a strong `Rc`, so the synthetic producer stays alive as long
/// as the operand is stored. Used behind the per-crate `as rb` import in the
/// backend / gc / jit-trace test suites.
#[cfg(feature = "test-support")]
pub fn bound_operand_from_opref(a: OpRef) -> crate::operand::Operand {
    crate::operand::Operand::bound_from_opref(a)
}

/// Shared helpers for building **bound** operands from majit-ir test modules
/// (`resoperation.rs`, ...). Production binds every `AbstractResOp` /
/// `AbstractInputArg` to its `Op` / `InputArg` identity, so tests that seed op
/// operands directly must do the same.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{OpRef, Type};
    use crate::operand::Operand;
    use crate::resoperation::{Op, OpCode};

    /// A self-rooting bound `Operand::Op` at `position`: the returned operand
    /// holds a strong `Rc` to the synthetic `SameAs*` / `Jump` producer, so it
    /// keeps that producer alive on its own and `to_opref()`s to
    /// `(type, position)`.
    pub(crate) fn bound_resop_operand(tp: Type, position: u32) -> Operand {
        let opcode = match tp {
            Type::Int => OpCode::SameAsI,
            Type::Float => OpCode::SameAsF,
            Type::Ref => OpCode::SameAsR,
            Type::Void => OpCode::Jump,
        };
        let op = std::rc::Rc::new(Op::new(opcode, &[]));
        op.pos.set(OpRef::op_typed(position, tp));
        Operand::from_bound_op(&op)
    }
}
