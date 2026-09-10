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
//! Hosted in `majit-ir` so the slot can carry `Rc<Op>` / `Rc<InputArg>`
//! without a `majit-metainterp -> majit-ir` circular dep.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[cfg(feature = "test-support")]
use crate::OpRef;
use crate::intbound::IntBound;
use crate::op_info::OpInfo;
use crate::ptr_info::PtrInfo;
use crate::resoperation::Op;
use crate::value::{Const, InputArg, Value};

/// Variant of the `_forwarded` slot.
///
/// `Const` is an `AbstractValue` subclass too (`history.py
/// ConstInt`), so forwarding to a constant is its own shape: `Const`
/// is a value-typed `Copy` payload with no `_forwarded` slot of its
/// own, unlike `ResOp`/`InputArg`. Keeping it as a separate variant
/// retires a dedicated const-as-chain-target carrier.
#[derive(Clone)]
pub enum Forwarded {
    None,

    /// `resoperation.py AbstractResOp` forwarding — direct
    /// `OpRc` reference. The chain walker steps straight into a
    /// producer-bound `Operand::Op` and continues from there.
    ///
    /// The slot OWNS its target, because `_forwarded` is an ordinary
    /// attribute assignment in RPython and the target is additionally
    /// reachable from the trace `operations` list, so a forwarding target
    /// is never collected while a chain step still names it. Holding a
    /// `Weak` here instead made that reachability a pyre-side invariant
    /// spread over `resop_refs` / `phase1_emit_ops` / `new_operations`, and
    /// a target held by none of them died mid-optimization: the walker then
    /// stopped one hop early and handed its caller an operand that was still
    /// forwarding.
    Op(crate::resoperation::OpRc),

    /// `resoperation.py AbstractInputArg` forwarding — direct
    /// `InputArgRc` reference. Same chain-walk and same ownership as `Op`.
    /// RPython uses this for inputarg→inputarg redirects in bridge import
    /// and retrace remap (compile.py / unroll.py).
    InputArg(Rc<InputArg>),

    /// `history.py ConstInt` / `ConstFloat` / `ConstPtr` object assigned to
    /// `_forwarded`. Forwarding terminates here; chain walkers clone this
    /// handle, so every read returns the SAME Const identity instead of
    /// allocating a fresh wrapper from an inline value. The `Cell` permits
    /// the explicit GC walker to forward `ConstPtr` in place.
    Const(Rc<Cell<Value>>),

    /// `Operand::SmallInt` twin — a freshly-minted `ConstInt` whose value
    /// fits in i32. Same identity-token encoding, no process allocator.
    SmallConst(u64),

    /// `Operand::SmallWide` twin — `ConstFloat` / `ConstPtr` / out-of-i32
    /// `ConstInt`. Same unique-token slab index, no `Rc<Cell<Value>>`.
    SmallWide(u64),

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

impl Forwarded {
    /// Mint the `_forwarded` Const object `optimizer.py make_constant` stores.
    pub fn from_const_value(value: Value) -> Self {
        if let Value::Int(v) = value
            && let Some(enc) = crate::operand::fresh_small_int(v)
        {
            return Forwarded::SmallConst(enc);
        }
        Forwarded::SmallWide(crate::operand::fresh_wide(value))
    }

    pub fn is_const(&self) -> bool {
        matches!(
            self,
            Forwarded::Const(_) | Forwarded::SmallConst(_) | Forwarded::SmallWide(_)
        )
    }

    pub fn const_value(&self) -> Option<Value> {
        match self {
            Forwarded::Const(c) => Some(c.get()),
            Forwarded::SmallConst(enc) => Some(Value::Int(crate::operand::small_int_value(*enc))),
            Forwarded::SmallWide(id) => Some(crate::operand::wide_value(*id)),
            _ => None,
        }
    }

    /// Non-null `ConstPtr` payload, if this slot is a const-ref terminal.
    pub fn const_ref(&self) -> Option<crate::value::GcRef> {
        match self.const_value() {
            Some(Value::Ref(gcref)) if !gcref.is_null() => Some(gcref),
            _ => None,
        }
    }

    /// Forward an inline `ConstPtr` in place (`walk_const_ptr_refs`).
    pub fn walk_const_ptr_refs(&self, visitor: &mut dyn FnMut(&mut crate::value::GcRef)) {
        match self {
            Forwarded::Const(cell) => {
                let mut v = cell.get();
                if let Value::Ref(gcref) = &mut v {
                    visitor(gcref);
                    cell.set(v);
                }
            }
            Forwarded::SmallWide(id) => {
                let cell = crate::operand::wide_slot(*id as u32);
                let mut v = cell.get();
                if let Value::Ref(gcref) = &mut v {
                    visitor(gcref);
                    cell.set(v);
                }
            }
            _ => {}
        }
    }

    /// Overwrite a const-ref terminal's `GcRef` after a collection.
    pub fn refresh_const_ref(&self, updated: crate::value::GcRef) {
        match self {
            Forwarded::Const(cell) if matches!(cell.get(), Value::Ref(_)) => {
                cell.set(Value::Ref(updated));
            }
            Forwarded::SmallWide(id) => {
                let cell = crate::operand::wide_slot(*id as u32);
                if matches!(cell.get(), Value::Ref(_)) {
                    cell.set(Value::Ref(updated));
                }
            }
            _ => {}
        }
    }
}

impl std::fmt::Debug for Forwarded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Compact by design: `Info(OpInfo)` carries the abstract-value graph
        // (nested ops / virtual fields whose own `forwarded` slots reference
        // back through `Info` edges), so a derived Debug recurses
        // without bound and overflows the stack when a `forwarded`-bearing
        // `Op` / `InputArg` is printed. Render the variant shape only.
        match self {
            Forwarded::None => f.write_str("None"),
            Forwarded::Op(_) => f.write_str("Op(..)"),
            Forwarded::InputArg(_) => f.write_str("InputArg(..)"),
            Forwarded::Const(c) => write!(f, "Const({:?})", c.get()),
            Forwarded::SmallConst(enc) => {
                write!(f, "SmallConst({})", crate::operand::small_int_value(*enc))
            }
            Forwarded::SmallWide(id) => {
                write!(f, "SmallWide({:?})", crate::operand::wide_value(*id))
            }
            Forwarded::Info(_) => f.write_str("Info(..)"),
        }
    }
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
    /// `resoperation.py get_forwarded` — clone the slot.
    fn get_forwarded(&self) -> Forwarded;

    /// `resoperation.py self._forwarded = forwarded_to` — the slot write
    /// shared by every typed setter. Prefer the typed `set_forwarded_*`,
    /// which carry the self-cycle assert.
    fn store_forwarded(&self, value: Forwarded);

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

    /// `optimizer.py op.set_forwarded(newop)` — Op target.
    fn set_forwarded_op(&self, target: &crate::resoperation::OpRc) {
        assert!(
            !self.is_same_op(target),
            "set_forwarded_op on the same Op creates a one-node chain cycle"
        );
        self.store_forwarded(Forwarded::Op(target.clone()));
    }

    /// `compile.py` / `unroll.py` InputArg→InputArg redirect.
    fn set_forwarded_inputarg(&self, target: &crate::value::InputArgRc) {
        assert!(
            !self.is_same_inputarg(target),
            "set_forwarded_inputarg on the same InputArg creates a one-node \
             chain cycle"
        );
        self.store_forwarded(Forwarded::InputArg(Rc::clone(target)));
    }

    /// `optimizer.py make_constant(box, constbox)` — terminate the chain
    /// in an inline constant value.
    fn set_forwarded_const(&self, value: Const) {
        // `optimizer.py make_constant` stores the Const object itself in the
        // box's `_forwarded` slot. Mint that object once at the write, not on
        // every `get_box_replacement` read. Small ConstInts use the same
        // inline identity encoding as `Operand::SmallInt`.
        self.store_forwarded(Forwarded::from_const_value(value.to_value()));
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
    fn get_forwarded(&self) -> Forwarded {
        self.forwarded().borrow()
    }
    fn store_forwarded(&self, value: Forwarded) {
        self.forwarded().set(value);
    }
    fn is_same_op(&self, op: &crate::resoperation::OpRc) -> bool {
        std::ptr::eq(self, crate::resoperation::OpRc::as_ptr(op))
    }
}

impl ForwardingHost for InputArg {
    fn get_forwarded(&self) -> Forwarded {
        self.forwarded.borrow()
    }
    fn store_forwarded(&self, value: Forwarded) {
        self.forwarded.set(value);
    }
    fn is_same_inputarg(&self, ia: &crate::value::InputArgRc) -> bool {
        std::ptr::eq(self, Rc::as_ptr(ia))
    }
}

/// Owning borrow guard: a shared `Ref<T>` transmuted to `'static`, kept
/// sound by holding the source `Rc<RefCell<T>>` alongside it.
///
/// SAFETY invariant (centralised here for all guard types): `_rc` keeps
/// the `RefCell` allocation alive for at least as long as `Self` exists,
/// so the `'static` `Ref` never dangles. Struct fields drop in
/// declaration order, so `inner` (the borrow) is released before `_rc`
/// drops the allocation.
pub struct BorrowGuard<T: 'static> {
    inner: std::cell::Ref<'static, T>,
    _rc: Rc<std::cell::RefCell<T>>,
}

impl<T> BorrowGuard<T> {
    pub(crate) fn new(rc: Rc<std::cell::RefCell<T>>) -> Self {
        // SAFETY: see the type-level invariant above.
        let r: std::cell::Ref<'_, T> = rc.borrow();
        let r: std::cell::Ref<'static, T> = unsafe { std::mem::transmute(r) };
        Self { inner: r, _rc: rc }
    }
}

impl<T> std::ops::Deref for BorrowGuard<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for BorrowGuard<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&*self.inner, f)
    }
}

/// Mutable counterpart of [`BorrowGuard`]. Holds the inner `RefCell`
/// exclusive borrow; a concurrent shared or exclusive borrow on the same
/// handle panics at runtime per `RefCell` semantics. Same `'static`
/// transmute + drop-order invariant as [`BorrowGuard`].
pub struct BorrowGuardMut<T: 'static> {
    inner: std::cell::RefMut<'static, T>,
    _rc: Rc<std::cell::RefCell<T>>,
}

impl<T> BorrowGuardMut<T> {
    pub(crate) fn new(rc: Rc<std::cell::RefCell<T>>) -> Self {
        // SAFETY: see the [`BorrowGuard`] type-level invariant.
        let r: std::cell::RefMut<'_, T> = rc.borrow_mut();
        let r: std::cell::RefMut<'static, T> = unsafe { std::mem::transmute(r) };
        Self { inner: r, _rc: rc }
    }
}

impl<T> std::ops::Deref for BorrowGuardMut<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<T> std::ops::DerefMut for BorrowGuardMut<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

/// Owning shared borrow guard for `ptr_info()`.
pub type PtrInfoBorrow = BorrowGuard<PtrInfo>;
/// Owning exclusive borrow guard for `ptr_info_mut()`.
pub type PtrInfoBorrowMut = BorrowGuardMut<PtrInfo>;
/// Owning shared borrow guard for `int_bound()`.
pub type IntBoundBorrow = BorrowGuard<IntBound>;
/// Owning exclusive borrow guard for `int_bound_mut()`.
pub type IntBoundBorrowMut = BorrowGuardMut<IntBound>;

/// Packed `_forwarded` word. Low 3 bits are the tag; an 8-aligned Rc
/// pointer uses tag 0 (`None` is the zero word). SmallConst keeps the
/// i32 payload and identity token in the remaining 61 bits so the slot
/// on every `Op` is 8 B and `Rc<Op>` leaves the 144-byte class.
const FW_TAG: u64 = 0b111;
const FW_OP: u64 = 0;
const FW_INPUTARG: u64 = 1;
const FW_CONST: u64 = 2;
const FW_SMALL_CONST: u64 = 3;
const FW_SMALL_WIDE: u64 = 4;
const FW_INFO_PTR: u64 = 5;
const FW_INFO_BOUND: u64 = 6;
const FW_INFO_OTHER: u64 = 7;
/// Heap pointers are 48-bit. Bits 56-61 of an `IntBound` word may carry
/// a small `_resint` stamp so optimizer restamp does not mint ThinStamp
/// around forwarded-only.
const FWD_PTR_MASK: u64 = (1 << 48) - 1;
const FWD_STAMP_SHIFT: u64 = 56;
const FWD_STAMP_MASK: u64 = 0x3f;

#[inline]
fn fwd_ptr(w: u64) -> u64 {
    w & FWD_PTR_MASK & !FW_TAG
}

/// SmallConst identity lives in bits 35-63. Bits 56-61 may carry a
/// small stamp, so a packed stamp keeps the id in bits 35-55 (21 bits).
const SMALL_CONST_ID_STAMP_BITS: u64 = 21;

#[inline]
pub(crate) fn fwd_stamp(w: u64) -> u32 {
    match w & FW_TAG {
        FW_INFO_BOUND | FW_SMALL_CONST => ((w >> FWD_STAMP_SHIFT) & FWD_STAMP_MASK) as u32,
        _ => 0,
    }
}

#[inline]
pub(crate) fn try_pack_fwd_stamp(packed: u64, stamp: u32) -> Option<u64> {
    if stamp >= 64 {
        return None;
    }
    match packed & FW_TAG {
        FW_INFO_BOUND => Some((packed & FWD_PTR_MASK) | ((stamp as u64) << FWD_STAMP_SHIFT)),
        FW_SMALL_CONST if packed >> 35 < (1 << SMALL_CONST_ID_STAMP_BITS) => {
            let body = packed & ((1u64 << FWD_STAMP_SHIFT) - 1);
            Some(body | ((stamp as u64) << FWD_STAMP_SHIFT))
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn strip_fwd_stamp(w: u64) -> u64 {
    match w & FW_TAG {
        FW_OP | FW_INPUTARG | FW_CONST | FW_INFO_PTR | FW_INFO_BOUND | FW_INFO_OTHER => {
            w & FWD_PTR_MASK
        }
        FW_SMALL_CONST => w & ((1u64 << FWD_STAMP_SHIFT) - 1),
        _ => w,
    }
}

pub(crate) fn pack_forwarded(v: Forwarded) -> u64 {
    match v {
        Forwarded::None => 0,
        Forwarded::Op(rc) => {
            let p = crate::resoperation::OpRc::into_raw(rc) as u64;
            debug_assert_eq!(p & FW_TAG, 0);
            p
        }
        Forwarded::InputArg(rc) => {
            let p = Rc::into_raw(rc) as u64;
            debug_assert_eq!(p & FW_TAG, 0);
            p | FW_INPUTARG
        }
        Forwarded::Const(rc) => {
            let p = Rc::into_raw(rc) as u64;
            debug_assert_eq!(p & FW_TAG, 0);
            p | FW_CONST
        }
        Forwarded::SmallConst(enc) => {
            let id = enc >> 32;
            let val = enc as u32 as u64;
            debug_assert!(id < (1 << 29), "SmallConst identity exceeds 29 bits");
            FW_SMALL_CONST | (val << 3) | (id << 35)
        }
        Forwarded::SmallWide(id) => {
            debug_assert!(id < (1 << 61));
            FW_SMALL_WIDE | (id << 3)
        }
        Forwarded::Info(info) => pack_info(info),
    }
}

fn pack_info(info: OpInfo) -> u64 {
    match info {
        OpInfo::Ptr(rc) => {
            let p = Rc::into_raw(rc) as u64;
            debug_assert_eq!(p & FW_TAG, 0);
            p | FW_INFO_PTR
        }
        OpInfo::IntBound(rc) => {
            let p = Rc::into_raw(rc) as u64;
            debug_assert_eq!(p & FW_TAG, 0);
            p | FW_INFO_BOUND
        }
        other => {
            let p = Box::into_raw(Box::new(other)) as u64;
            debug_assert_eq!(p & FW_TAG, 0);
            p | FW_INFO_OTHER
        }
    }
}

pub(crate) fn unpack_forwarded(w: u64) -> Forwarded {
    if w == 0 {
        return Forwarded::None;
    }
    match w & FW_TAG {
        FW_OP => {
            let rc = unsafe { crate::resoperation::OpRc::from_raw(fwd_ptr(w) as *const Op) };
            let out = Forwarded::Op(rc.clone());
            std::mem::forget(rc);
            out
        }
        FW_INPUTARG => {
            let rc = unsafe { Rc::from_raw(fwd_ptr(w) as *const InputArg) };
            let out = Forwarded::InputArg(Rc::clone(&rc));
            std::mem::forget(rc);
            out
        }
        FW_CONST => {
            let rc = unsafe { Rc::from_raw(fwd_ptr(w) as *const Cell<Value>) };
            let out = Forwarded::Const(Rc::clone(&rc));
            std::mem::forget(rc);
            out
        }
        FW_SMALL_CONST => {
            let val = ((w >> 3) as u32) as u64;
            let stamp = (w >> FWD_STAMP_SHIFT) & FWD_STAMP_MASK;
            let id = if stamp == 0 {
                w >> 35
            } else {
                (w >> 35) & ((1 << SMALL_CONST_ID_STAMP_BITS) - 1)
            };
            Forwarded::SmallConst((id << 32) | val)
        }
        FW_SMALL_WIDE => Forwarded::SmallWide(w >> 3),
        FW_INFO_PTR => {
            let rc = unsafe { Rc::from_raw(fwd_ptr(w) as *const RefCell<PtrInfo>) };
            let out = Forwarded::Info(OpInfo::Ptr(Rc::clone(&rc)));
            std::mem::forget(rc);
            out
        }
        FW_INFO_BOUND => {
            let rc = unsafe { Rc::from_raw(fwd_ptr(w) as *const RefCell<IntBound>) };
            let out = Forwarded::Info(OpInfo::IntBound(Rc::clone(&rc)));
            std::mem::forget(rc);
            out
        }
        FW_INFO_OTHER => {
            let boxed = unsafe { Box::from_raw(fwd_ptr(w) as *mut OpInfo) };
            let out = Forwarded::Info((*boxed).clone());
            std::mem::forget(boxed);
            out
        }
        _ => Forwarded::None,
    }
}

pub(crate) fn drop_packed_forwarded(w: u64) {
    if w == 0 {
        return;
    }
    match w & FW_TAG {
        FW_OP => drop(unsafe { crate::resoperation::OpRc::from_raw(fwd_ptr(w) as *const Op) }),
        FW_INPUTARG => drop(unsafe { Rc::from_raw(fwd_ptr(w) as *const InputArg) }),
        FW_CONST => drop(unsafe { Rc::from_raw(fwd_ptr(w) as *const Cell<Value>) }),
        FW_INFO_PTR => drop(unsafe { Rc::from_raw(fwd_ptr(w) as *const RefCell<PtrInfo>) }),
        FW_INFO_BOUND => drop(unsafe { Rc::from_raw(fwd_ptr(w) as *const RefCell<IntBound>) }),
        FW_INFO_OTHER => drop(unsafe { Box::from_raw(fwd_ptr(w) as *mut OpInfo) }),
        _ => {}
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
    use crate::operand::Operand;
    use crate::resoperation::{Op, OpCode};
    use crate::{OpRef, Type};

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
        let op = crate::resoperation::OpRc::new(Op::new(opcode, &[]));
        op.pos().set(OpRef::op_typed(position, tp));
        Operand::from_bound_op(&op)
    }
}
