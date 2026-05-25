//! Rust mirror of RPython's `AbstractValue` object identity.
//!
//! Direct port of the Python object identity hierarchy formed by
//! `rpython/jit/metainterp/resoperation.py:29 AbstractValue` together with
//! `AbstractResOpOrInputArg` / `AbstractResOp` / `AbstractInputArg` /
//! `Const*` (`history.py:182`), expressed as `Rc<Box>`.
//!
//! Hosted in `majit-ir` so the `forwarded: RefCell<Forwarded>` slot
//! eventually lifted onto `Op` / `InputArg` can carry the type without a
//! `majit-metainterp → majit-ir` circular dep. The `BoxPool` side-table
//! that addresses these objects by `OpRef` index stays in metainterp.
//!
//! # Design decisions
//!
//! - The `forwarded` slot is a `RefCell<Forwarded>`. `Cell` is not used
//!   because `Forwarded` carries `OpInfo` / `BoxRef`, neither of which is
//!   `Copy`. Helpers terminate the borrow scope immediately after reading.
//! - `BoxRef`'s `Eq` / `Hash` use `Rc::ptr_eq` / `Rc::as_ptr` — equivalent
//!   to RPython's use of object identity as a dict key.
//! - When `Forwarded::Box(BoxRef)` carries a BoxRef whose kind is
//!   `BoxKind::Const(...)`, that mirrors RPython's
//!   `box.set_forwarded(constbox)`. We do not introduce a separate `Const`
//!   variant: RPython stores everything in a single `_forwarded` slot.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use crate::intbound::IntBound;
use crate::op_info::OpInfo;
use crate::ptr_info::PtrInfo;
use crate::resoperation::{Op, VectorizationInfo};
use crate::value::{Const, InputArg};
use crate::{OpRef, Type, Value};

/// `AbstractValue` mirror — unified representation of RPython's
/// op/inputarg/const objects.
pub struct Box {
    /// `resoperation.py:233-243 AbstractResOpOrInputArg._forwarded`.
    ///
    /// Const boxes also carry the slot, but in RPython `Const` is not a
    /// subclass of `AbstractResOpOrInputArg`, so its `_forwarded` is
    /// always `None`. Rust unifies the layout into a single struct
    /// shape while preserving the same invariant.
    pub forwarded: RefCell<Forwarded>,

    /// `resoperation.py:260 type` (`'i'` / `'r'` / `'f'` / `'v'`).
    /// Absorbs the frontend semantic portion that majit currently spreads
    /// across `value_types` / `inputarg_types` / `constant_types`.
    pub type_: Type,

    /// Rust enum mirror of RPython's subclass hierarchy.
    pub kind: BoxKind,

    /// `history.py:803-807 IntFrontendOp(pos, intval)` /
    /// `FloatFrontendOp(pos, floatval)` /
    /// `RefFrontendOp(pos, gcref)` parity — the intrinsic
    /// per-position concrete-value carrier that PyPy attaches to every
    /// operation-result Box at execute-time.  Replaces the previous
    /// `TraceCtx::opref_concrete: HashMap<u32, Value>` side-table.
    /// Const boxes ignore this slot; their value lives in
    /// `BoxKind::Const { value, .. }` instead.
    pub value: Cell<Option<Value>>,

    /// Dual-write backref to the `Op` this Box stands in for.
    /// `Weak` to avoid an Rc cycle (Op.forwarded may carry a `BoxRef`
    /// holding an `Rc<Box>` whose `op_handle` could otherwise loop back
    /// through the trace). Empty for `BoxKind::InputArg` / `Const`
    /// (those have their own backref strategy or no Op counterpart);
    /// for `BoxKind::ResOp` it is filled by `BoxRef::bind_op` at the
    /// recorder→TreeLoop handoff. When `Some`, `set_forwarded_box` /
    /// `set_forwarded_info` / `clear_forwarded` also write through to
    /// `op.forwarded`, establishing the invariant
    /// `Box.forwarded == op.forwarded`.
    pub op_handle: RefCell<Option<Weak<Op>>>,

    /// Parity: backref to the `InputArg` this Box stands in
    /// for, mirroring `op_handle` for the `BoxKind::InputArg` variant.
    /// Empty for `BoxKind::ResOp` / `Const`; filled by
    /// `BoxRef::bind_inputarg` at the recorder→TreeLoop handoff so the
    /// authoritative `_forwarded` slot lives on the `InputArg` itself
    /// (RPython `resoperation.py:700 AbstractInputArg._forwarded`).
    /// Without this, `InputArg.forwarded` would remain a dead field and
    /// readers migrating off `BoxRef.forwarded` would see stale `None`.
    pub inputarg_handle: RefCell<Option<Weak<InputArg>>>,
}

/// Enum mirror of the PyPy class hierarchy.
pub enum BoxKind {
    /// `resoperation.py:250 AbstractResOp` — the operation object itself
    /// is the result identity. `position` is a pyre-only field with no
    /// RPython counterpart on `AbstractResOpOrInputArg` (which carries
    /// only `_forwarded`); it stores the index where the box lands in the
    /// pool so the chain walker can reconstruct an `OpRef::op_typed(pos,
    /// type)` when advancing through `Forwarded::Box(target)`. Set by
    /// `opencoder.py Trace.record()` (and the recorder's `record_op_*`
    /// family in pyre) at construction; updated by `optimizer.rs:2783`
    /// const-pool compaction via `BoxRef::set_position`. RPython has no
    /// equivalent on `AbstractResOp` itself — it threads
    /// `position_and_flags` through `FrontendOp` (resoperation.py:233).
    ResOp { position: std::cell::Cell<u32> },

    /// `resoperation.py:699 AbstractInputArg`.
    /// `position` mirrors `AbstractInputArg.position`
    /// (resoperation.py:699).
    InputArg { position: u32 },

    /// `history.py:220 ConstInt` / `:261 ConstFloat` / `:307 ConstPtr`.
    /// `const_index` is a pyre-only field carrying the
    /// `OpRef::Const{Int,Float,Ptr}.const_index()` so the chain walker
    /// can reconstruct a constant-namespace OpRef when advancing past
    /// `Forwarded::Box(const_box)` written by `replace_op(_, const_target)`.
    /// `None` for `BoxRef::new_const(value)` (no index in scope —
    /// `make_constant` used to pick this path before slice 18 split it
    /// off; remaining `None` callers are test fixtures).
    Const {
        value: Value,
        const_index: Option<u32>,
    },
}

/// Variant of the `_forwarded` slot.
///
/// `Const` is an `AbstractValue` subclass too (`history.py:220
/// ConstInt`), so forwarding to a constant is its own shape: `Const`
/// is a value-typed `Copy` payload with no `_forwarded` slot of its
/// own, unlike `ResOp`/`InputArg`. Keeping it as a separate variant
/// retires the dedicated `BoxKind::Const`-as-chain-target carrier.
#[derive(Clone, Debug)]
pub enum Forwarded {
    None,

    /// Forwarding to another `AbstractResOpOrInputArg`. Const targets
    /// route through [`Forwarded::Const`] instead; `ResOp` targets route
    /// through [`Forwarded::Op`] once C.5 retires this variant.
    Box(BoxRef),

    /// `resoperation.py:250 AbstractResOp` forwarding — direct
    /// `Weak<Op>` reference, no `BoxRef`/`BoxKind::ResOp` carrier.
    /// Chain walker upgrades the `Weak`, wraps the `Op` in a transient
    /// `BoxRef` (via `BoxRef::from_bound_op`), and continues from
    /// there. A dropped `Weak` terminates the chain at the predecessor
    /// (PyPy never observes a dropped target — RPython keeps the
    /// underlying object alive through the trace `operations` list).
    Op(Weak<Op>),

    /// `resoperation.py:699 AbstractInputArg` forwarding — direct
    /// `Weak<InputArg>` reference, no `BoxRef`/`BoxKind::InputArg`
    /// carrier. Same chain-walk semantics as `Op` (transient BoxRef
    /// materialization via `BoxRef::from_bound_inputarg`). RPython
    /// uses this for inputarg→inputarg redirects in bridge import and
    /// retrace remap (compile.py:478 / unroll.py:497).
    InputArg(Weak<InputArg>),

    /// `history.py:220 ConstInt` / `:261 ConstFloat` / `:307 ConstPtr`
    /// — forwarding terminates here; the constant value is carried
    /// inline. Chain walkers stop on this variant (`not_const=true`
    /// returns the pre-Const box; `not_const=false` materializes a
    /// terminal const-bearing `BoxRef` for legacy callers).
    ///
    /// The trailing `Option<u32>` is the pyre-only `const_pool`
    /// constant-namespace index that lets `box_to_opref` reconstruct
    /// an `OpRef::Const{Int,Float,Ptr}(idx)` from a chain-walked-to-
    /// Const terminal. PyPy has no analog (callers hold the Python
    /// `Const` object directly). Sidecar to be retired alongside the
    /// broader OpRef-reverse-lookup machinery; tagged `None` for
    /// indexless seed_constant plantings (`optimizer.py:432` body
    /// arm) and test plantings.
    Const(Const, Option<u32>),

    /// `optimizeopt/info.py:17 AbstractInfo (is_info_class = True)` family —
    /// `PtrInfo`, `IntBound`, `FloatConstInfo`, `EmptyInfo`, etc.
    Info(OpInfo),

    /// `resoperation.py:156 VectorizationInfo(AbstractValue)` written by
    /// `schedule.py:20-28 forwarded_vecinfo`.  Separate from `Info`
    /// because vectorizer metadata is not an `OpInfo` semantic class —
    /// it's per-op codegen state.
    VectorInfo(VectorizationInfo),
}

/// `Rc<Box>` newtype.
///
/// `Eq` / `Hash` are pointer identity for ALL variants — mirrors PyPy's
/// `is`-based default `__eq__` on `AbstractValue` (covers `ResOp` /
/// `InputArg` / `Const`). PyPy's value-based comparison for constants is
/// the explicit `same_box` / `same_constant` method (`history.py:204`),
/// not `__eq__`. Callers that need value comparison on constants compare
/// `const_value()` outputs directly.
pub struct BoxRef(Rc<Box>);

impl BoxRef {
    /// Raw pointer to the underlying allocation. Identity-equality probe
    /// used by tests; not part of the public PyPy parity surface.
    pub fn as_ptr(&self) -> *const Box {
        Rc::as_ptr(&self.0)
    }

    /// New `AbstractResOp` Box.
    ///
    /// `position` is a pyre-only field with no RPython counterpart on
    /// `AbstractResOpOrInputArg` (which carries only `_forwarded`).
    /// Stores the index this box occupies in the pool (== raw value of
    /// the matching OpRef). Construction-time assignment matches PyPy's
    /// `Trace.record()`.
    pub fn new_resop(type_: Type, position: u32) -> Self {
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::ResOp {
                position: std::cell::Cell::new(position),
            },
            value: Cell::new(None),
            op_handle: RefCell::new(None),
            inputarg_handle: RefCell::new(None),
        }))
    }

    /// Transient `AbstractResOp` Box wrapping an already-bound
    /// `OpRc`. Used by the chain walker to materialize a `BoxRef`
    /// terminal from a `Forwarded::Op(Weak<Op>)` payload without
    /// going through the pool; the new box does not live in
    /// `BoxPool`, but its `bound_op` immediately answers with the
    /// same `Rc<Op>` and its `_forwarded` slot reads via the bound op
    /// per `get_forwarded`. Type and position are mirrored from
    /// `op.pos.get()`.
    pub fn from_bound_op(op: &crate::resoperation::OpRc) -> Self {
        let opref = op.pos.get();
        let type_ = opref.ty().unwrap_or(Type::Void);
        let position = opref.raw();
        Self(Rc::new(Box {
            // `Box.forwarded` is the legacy mirror — the bound op's
            // own slot is canonical so this stays None; `get_forwarded`
            // returns op.forwarded via the `bound_op()` fastpath.
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::ResOp {
                position: std::cell::Cell::new(position),
            },
            value: Cell::new(None),
            op_handle: RefCell::new(Some(Rc::downgrade(op))),
            inputarg_handle: RefCell::new(None),
        }))
    }

    /// Transient `AbstractInputArg` Box wrapping an already-bound
    /// `InputArgRc`. Mirror of `from_bound_op` for the chain walker's
    /// `Forwarded::InputArg` terminal materialization.
    pub fn from_bound_inputarg(ia: &crate::value::InputArgRc) -> Self {
        let type_ = ia.tp;
        let position = ia.index;
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::InputArg { position },
            value: Cell::new(None),
            op_handle: RefCell::new(None),
            inputarg_handle: RefCell::new(Some(Rc::downgrade(ia))),
        }))
    }

    /// New `AbstractInputArg` Box.
    pub fn new_inputarg(type_: Type, position: u32) -> Self {
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::InputArg { position },
            value: Cell::new(None),
            op_handle: RefCell::new(None),
            inputarg_handle: RefCell::new(None),
        }))
    }

    /// New `Const*` Box. `type_` is inferred from `value`.
    /// No `const_index` — used by callers without a const-namespace
    /// OpRef in scope (test fixtures, default constructors).
    pub fn new_const(value: Value) -> Self {
        let type_ = value.get_type();
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::Const {
                value,
                const_index: None,
            },
            value: Cell::new(None),
            op_handle: RefCell::new(None),
            inputarg_handle: RefCell::new(None),
        }))
    }

    /// New `Const*` Box carrying a `const_index`. Used by
    /// `replace_op(_, const_target)` so the chain walker can reconstruct
    /// `OpRef::Const{Int,Float,Ptr}(const_index)` when advancing past
    /// `Forwarded::Box(const_box)`.
    pub fn new_const_with_index(value: Value, const_index: u32) -> Self {
        let type_ = value.get_type();
        Self(Rc::new(Box {
            forwarded: RefCell::new(Forwarded::None),
            type_,
            kind: BoxKind::Const {
                value,
                const_index: Some(const_index),
            },
            value: Cell::new(None),
            op_handle: RefCell::new(None),
            inputarg_handle: RefCell::new(None),
        }))
    }

    /// Bind this Box to its corresponding `Op` so subsequent
    /// `set_forwarded_*` / `clear_forwarded` calls dual-write through to
    /// `op.forwarded`. Stores a `Weak<Op>` to avoid an Rc cycle. Called by
    /// `TreeLoop::with_box_pool` at the recorder→TreeLoop handoff. Panics
    /// if called on a non-ResOp Box.
    ///
    /// Late-binding carry-over: at bind time the `Box`'s forwarded state
    /// becomes the source of truth for the bound `Op`. Copy `Box.forwarded`
    /// into `op.forwarded` unconditionally — including when the box is
    /// `Forwarded::None` — so any stale forwarding the `OpRc` happened to
    /// carry (e.g. from a clone path) is overwritten and post-bind
    /// `get_forwarded` reads exactly what the writer set.
    pub fn bind_op(&self, op: &crate::resoperation::OpRc) {
        assert!(
            matches!(&self.0.kind, BoxKind::ResOp { .. }),
            "BoxRef::bind_op only valid for ResOp boxes"
        );
        // Read the canonical effective forwarded slot, not the in-Box
        // mirror. `get_forwarded` consults the bound op/inputarg first,
        // so a `bind → set_forwarded → rebind` sequence carries the
        // freshest state across the rebind even if a writer ever bypasses
        // `BoxRef::set_forwarded_*` and updates `op.forwarded` directly.
        let carry = self.get_forwarded();
        *op.forwarded.borrow_mut() = carry;
        *self.0.op_handle.borrow_mut() = Some(Rc::downgrade(op));
    }

    /// Upgrade the bound `Weak<Op>` into a strong `OpRc`.
    /// Returns `None` for unbound boxes (InputArg / Const / lazy-allocated)
    /// or if the bound `Op` was dropped. Callers migrating away from
    /// `box.get_forwarded()` use this to reach `op.forwarded.borrow()` for
    /// readers and `op.forwarded.borrow_mut()` for writers.
    pub fn bound_op(&self) -> Option<crate::resoperation::OpRc> {
        self.0.op_handle.borrow().as_ref().and_then(|w| w.upgrade())
    }

    /// InputArg counterpart of `bind_op`. Stores a
    /// `Weak<InputArg>` so subsequent `set_forwarded_*` / `clear_forwarded`
    /// route through `inputarg.forwarded` (`resoperation.py:700
    /// AbstractInputArg._forwarded`). Panics if called on a non-InputArg
    /// box. Late-binding carry-over: `Box.forwarded` is copied into
    /// `inputarg.forwarded` unconditionally so any forwarding written
    /// before bind survives the handoff and post-bind readers see what
    /// was set.
    pub fn bind_inputarg(&self, ia: &crate::value::InputArgRc) {
        assert!(
            matches!(&self.0.kind, BoxKind::InputArg { .. }),
            "BoxRef::bind_inputarg only valid for InputArg boxes"
        );
        // Same canonical-slot rule as `bind_op`: `get_forwarded` reads via
        // the bound handle first so the rebind sees the freshest state.
        let carry = self.get_forwarded();
        *ia.forwarded.borrow_mut() = carry;
        *self.0.inputarg_handle.borrow_mut() = Some(Rc::downgrade(ia));
    }

    /// Upgrade the bound `Weak<InputArg>` into a strong
    /// `InputArgRc`. Returns `None` for unbound or non-InputArg boxes and
    /// for dropped `InputArg`s. Symmetric to `bound_op`.
    pub fn bound_inputarg(&self) -> Option<crate::value::InputArgRc> {
        self.0
            .inputarg_handle
            .borrow()
            .as_ref()
            .and_then(|w| w.upgrade())
    }

    /// Extract the `const_index` field for chain-walker reconstruction.
    /// Returns `None` for non-Const boxes and for Consts created via
    /// `new_const` (no index in scope).
    pub fn const_index(&self) -> Option<u32> {
        match &self.0.kind {
            BoxKind::Const { const_index, .. } => *const_index,
            _ => None,
        }
    }

    pub fn type_(&self) -> Type {
        self.0.type_
    }

    /// `resoperation.py:47 is_constant`.
    pub fn is_constant(&self) -> bool {
        matches!(self.0.kind, BoxKind::Const { .. })
    }

    pub fn is_inputarg(&self) -> bool {
        matches!(self.0.kind, BoxKind::InputArg { .. })
    }

    pub fn is_resop(&self) -> bool {
        matches!(self.0.kind, BoxKind::ResOp { .. })
    }

    /// `resoperation.py:233 AbstractResOpOrInputArg._pos` accessor.
    ///
    /// Returns the index where this box resides in the pool for
    /// `ResOp` / `InputArg` (which are the two PyPy classes that own
    /// `_pos`). `Const` has no canonical position and returns `None`.
    pub fn position(&self) -> Option<u32> {
        match &self.0.kind {
            BoxKind::ResOp { position } => Some(position.get()),
            BoxKind::InputArg { position } => Some(*position),
            BoxKind::Const { .. } => None,
        }
    }

    /// Update the ResOp position field. Used by `optimizer.rs:2783`
    /// const-pool compaction to keep `BoxKind::ResOp { position }` aligned
    /// with the new dense op-position range so the chain walker's BoxRef
    /// reconstruction (`OpRef::op_typed(target.position(), tp)`) returns
    /// post-compact positions.
    ///
    /// No-op for `InputArg` / `Const` (their positions are not subject
    /// to compaction).
    pub fn set_position(&self, new_pos: u32) {
        if let BoxKind::ResOp { position } = &self.0.kind {
            position.set(new_pos);
        }
    }

    /// Extract the constant value. Mirrors `history.py:233 ConstInt.getint`
    /// and the equivalent accessors on the other Const subclasses.
    pub fn const_value(&self) -> Option<Value> {
        match self.0.kind {
            BoxKind::Const { value, .. } => Some(value),
            _ => None,
        }
    }

    /// `resoperation.py:50 get_forwarded`. Returns a clone of the
    /// current slot. For bound ResOp boxes the read is routed
    /// through `op.forwarded`; extends this to InputArg via
    /// `inputarg.forwarded`. Unbound boxes (Const / lazy-allocated) fall
    /// back to `Box.forwarded`. The clone is cheap — every `Forwarded`
    /// payload is an `Rc`/`Copy` handle.
    pub fn get_forwarded(&self) -> Forwarded {
        if let Some(op) = self.bound_op() {
            return op.forwarded.borrow().clone();
        }
        if let Some(ia) = self.bound_inputarg() {
            return ia.forwarded.borrow().clone();
        }
        self.0.forwarded.borrow().clone()
    }

    /// `resoperation.py:53 set_forwarded(forwarded_to)` — Box variant.
    pub fn set_forwarded_box(&self, target: BoxRef) {
        // `assert forwarded_to is not self` (resoperation.py:241).
        // Always-on assert so a release build can't accept a one-node
        // forwarding cycle that would make `get_box_replacement()` spin.
        // After `bind_op` / `bind_inputarg` two distinct `Rc<Box>`
        // wrappers can share the same canonical bound `OpRc`/`InputArgRc`;
        // the `Rc::ptr_eq` on `self.0` alone misses that case, so also
        // compare the bound handles when both sides carry one.
        assert!(!Rc::ptr_eq(&self.0, &target.0));
        if let (Some(self_op), Some(target_op)) = (self.bound_op(), target.bound_op()) {
            assert!(
                !Rc::ptr_eq(&self_op, &target_op),
                "set_forwarded_box on a BoxRef that wraps the same bound \
                 Op as the target creates a one-node chain cycle"
            );
        }
        if let (Some(self_ia), Some(target_ia)) = (self.bound_inputarg(), target.bound_inputarg()) {
            assert!(
                !Rc::ptr_eq(&self_ia, &target_ia),
                "set_forwarded_box on a BoxRef that wraps the same bound \
                 InputArg as the target creates a one-node chain cycle"
            );
        }
        // RPython AbstractValue invariant: `Const` is not a subclass of
        // `AbstractResOpOrInputArg` (history.py:182), so `set_forwarded`
        // is undefined on Const objects (resoperation.py:50 default
        // raises). The Rust port unifies the layout into a single struct
        // shape; this assertion preserves the invariant. PyPy raises
        // unconditionally, so the check is always-on (not `debug_assert!`).
        assert!(
            !matches!(self.0.kind, BoxKind::Const { .. }),
            "set_forwarded_box on Const violates RPython AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        let next = Forwarded::Box(target);
        self.write_forwarded(next);
    }

    /// `optimizer.py:394 op.set_forwarded(newop)` — InputArg variant.
    /// Targets an `AbstractInputArg` identity directly via
    /// `Weak<InputArg>`. Mirror of `set_forwarded_op` for the InputArg
    /// chain-step case (compile.py:478, unroll.py:497).
    pub fn set_forwarded_inputarg(&self, target: &crate::value::InputArgRc) {
        // `resoperation.py:241 assert forwarded_to is not self` —
        // compare against this BoxRef's bound InputArg so a chain step
        // targeting the box's own identity panics instead of spinning
        // the walker.
        if let Some(self_ia) = self.bound_inputarg() {
            assert!(
                !Rc::ptr_eq(&self_ia, target),
                "set_forwarded_inputarg on the same InputArg creates a \
                 one-node chain cycle"
            );
        }
        assert!(
            !matches!(self.0.kind, BoxKind::Const { .. }),
            "set_forwarded_inputarg on Const violates RPython AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        self.write_forwarded(Forwarded::InputArg(Rc::downgrade(target)));
    }

    /// `optimizer.py:394 op.set_forwarded(newop)` — Op variant.
    /// Targets an `AbstractResOp` identity directly via `Weak<Op>`,
    /// retiring the `BoxKind::ResOp`-as-chain-target carrier. The
    /// caller passes the canonical `OpRc` (typically a `TreeLoop.ops`
    /// entry) — chain walkers upgrade the `Weak` and continue from
    /// there.
    pub fn set_forwarded_op(&self, target: &crate::resoperation::OpRc) {
        // `resoperation.py:241 assert forwarded_to is not self` —
        // compare against this BoxRef's bound Op so a chain step
        // targeting the box's own identity panics instead of spinning
        // the walker.
        if let Some(self_op) = self.bound_op() {
            assert!(
                !Rc::ptr_eq(&self_op, target),
                "set_forwarded_op on the same Op creates a one-node chain \
                 cycle"
            );
        }
        assert!(
            !matches!(self.0.kind, BoxKind::Const { .. }),
            "set_forwarded_op on Const violates RPython AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        self.write_forwarded(Forwarded::Op(Rc::downgrade(target)));
    }

    /// `optimizer.py:432 make_constant(box, constbox)` — Const variant.
    /// `Const` is an `AbstractValue` subclass (`history.py:220`), so PyPy
    /// `box.set_forwarded(constbox)` is well-typed; here it terminates
    /// the chain in a value-typed payload rather than allocating a
    /// `BoxKind::Const` carrier. `const_index` is the pyre-side sidecar
    /// for `box_to_opref` OpRef reconstruction; pass `None` when the
    /// const has no const-namespace OpRef (PyPy parity site).
    pub fn set_forwarded_const(&self, value: Const, const_index: Option<u32>) {
        // Same Const-as-source invariant as the other set_forwarded_*
        // variants — Const has no `_forwarded` slot per PyPy.
        assert!(
            !matches!(self.0.kind, BoxKind::Const { .. }),
            "set_forwarded_const on Const violates RPython AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        self.write_forwarded(Forwarded::Const(value, const_index));
    }

    /// `resoperation.py:53 set_forwarded(forwarded_to)` — Info variant.
    pub fn set_forwarded_info(&self, info: OpInfo) {
        // PyPy `AbstractValue.set_forwarded` raises unconditionally on
        // `Const`; mirror that with an always-on assert (not
        // `debug_assert!`) so release builds preserve the invariant.
        assert!(
            !matches!(self.0.kind, BoxKind::Const { .. }),
            "set_forwarded_info on Const violates RPython AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        let next = Forwarded::Info(info);
        self.write_forwarded(next);
    }

    /// `schedule.py:20-28 forwarded_vecinfo` — set the per-op vectorizer
    /// metadata into the `_forwarded` slot via the dedicated
    /// `VectorInfo` variant.
    pub fn set_forwarded_vector_info(&self, info: VectorizationInfo) {
        assert!(
            !matches!(self.0.kind, BoxKind::Const { .. }),
            "set_forwarded_vector_info on Const violates RPython AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        *self.0.forwarded.borrow_mut() = Forwarded::VectorInfo(info);
    }

    /// `schedule.py:30-36 forwarded_vecinfo` reader — return a clone of the
    /// `VectorizationInfo` currently in the `_forwarded` slot, or `None`
    /// if the slot does not hold `VectorInfo`.
    pub fn vector_info(&self) -> Option<VectorizationInfo> {
        match &*self.0.forwarded.borrow() {
            Forwarded::VectorInfo(info) => Some(info.clone()),
            _ => None,
        }
    }

    /// `history.py:803-807 IntFrontendOp(pos, intval).intval` parity —
    /// read the per-Box intrinsic value carrier.  Const boxes return
    /// their fixed `BoxKind::Const::value`.
    pub fn get_value(&self) -> Option<Value> {
        match &self.0.kind {
            BoxKind::Const { value, .. } => Some(*value),
            _ => self.0.value.get(),
        }
    }

    /// `history.py:803-807` construction-time field assignment analog.
    /// Const boxes are immutable and panic.
    pub fn set_value(&self, value: Value) {
        match &self.0.kind {
            BoxKind::Const { .. } => {
                panic!("BoxRef::set_value: Const value is immutable (BoxKind::Const)");
            }
            _ => self.0.value.set(Some(value)),
        }
    }

    /// `_forwarded = None` (used during transition / phase reset).
    pub fn clear_forwarded(&self) {
        // Const has no _forwarded slot to reset; clearing is a no-op for
        // Const but should not be called on it. Allow clear (idempotent
        // None) for transitional safety while migration progresses.
        if matches!(self.0.kind, BoxKind::Const { .. }) {
            return;
        }
        self.write_forwarded(Forwarded::None);
    }

    /// Dual-write to `op.forwarded` / `inputarg.forwarded` (the canonical
    /// `_forwarded` host, resoperation.py:233-242 / :700) AND
    /// `Box.forwarded`. Snapshot consumers (`compile_retrace` partial
    /// trace import, test fixtures) clone the `BoxRef` by value and may
    /// outlive the originating `OpRc` / `InputArgRc`; once the `Weak`
    /// upgrade fails, `get_forwarded` falls back to `Box.forwarded`, so
    /// that slot must stay current. The cost is one extra `RefCell` write
    /// per `set_forwarded_*` — `Forwarded` is `Clone`, payloads are
    /// `Rc`/`Copy` handles.
    fn write_forwarded(&self, value: Forwarded) {
        if let Some(weak) = self.0.op_handle.borrow().as_ref() {
            if let Some(op) = weak.upgrade() {
                *op.forwarded.borrow_mut() = value.clone();
            }
        }
        if let Some(weak) = self.0.inputarg_handle.borrow().as_ref() {
            if let Some(ia) = weak.upgrade() {
                *ia.forwarded.borrow_mut() = value.clone();
            }
        }
        *self.0.forwarded.borrow_mut() = value;
    }

    /// `resoperation.py:57-68 get_box_replacement(not_const=False)`.
    ///
    /// Walk the `_forwarded` chain, returning the box one step before the
    /// chain hits `None`, `Info`, or (`not_const=true && next.is_constant()`).
    pub fn get_box_replacement(&self, not_const: bool) -> BoxRef {
        let mut cur = self.clone();
        loop {
            match cur.get_forwarded() {
                Forwarded::None | Forwarded::Info(_) | Forwarded::VectorInfo(_) => return cur,
                Forwarded::Box(b) => {
                    if not_const && b.is_constant() {
                        return cur;
                    }
                    cur = b;
                }
                Forwarded::Op(weak) => {
                    let Some(op_rc) = weak.upgrade() else {
                        // Dropped target: PyPy has no analog (Python
                        // GC keeps targets alive through `operations`).
                        // Terminate the chain at `cur` to avoid a
                        // dangling read.
                        return cur;
                    };
                    cur = BoxRef::from_bound_op(&op_rc);
                }
                Forwarded::InputArg(weak) => {
                    let Some(ia_rc) = weak.upgrade() else {
                        return cur;
                    };
                    cur = BoxRef::from_bound_inputarg(&ia_rc);
                }
                Forwarded::Const(c, idx) => {
                    if not_const {
                        return cur;
                    }
                    // Materialize a terminal Const-bearing BoxRef so
                    // legacy callers that expect `.const_value()` /
                    // `BoxKind::Const` on the walker output keep
                    // working until the BoxRef return type itself is
                    // retired. Index sidecar round-trips so
                    // `box_to_opref` can rebuild
                    // `OpRef::Const{Int,Float,Ptr}(idx)`.
                    return match idx {
                        Some(i) => BoxRef::new_const_with_index(c.to_value(), i),
                        None => BoxRef::new_const(c.to_value()),
                    };
                }
            }
        }
    }

    /// `optimizer.py:99-113 getptrinfo` BoxRef-native reader.
    ///
    /// When `_forwarded` is `Info(OpInfo::Ptr(rc))`, returns the inner
    /// `PtrInfo` as a `PtrInfoBorrow` guard. The guard owns an `Rc` clone
    /// of the live `Rc<RefCell<PtrInfo>>` and holds a shared `RefCell`
    /// borrow into it. All other states (`None`, `Box(_)`, other
    /// `OpInfo` variants) return `None`.
    ///
    /// Object identity: two boxes whose `_forwarded` slots carry clones
    /// of the same `Rc` see in-place mutations through each other, just
    /// like RPython `_forwarded` Python object identity.
    ///
    /// Does not walk the chain — the caller is responsible for advancing
    /// to the terminal BoxRef (e.g. via
    /// `OptContext::get_box_replacement_box`) before calling. This mirrors
    /// reading `box.get_forwarded()` directly in RPython.
    pub fn ptr_info(&self) -> Option<PtrInfoBorrow> {
        let rc = match self.get_forwarded() {
            Forwarded::Info(OpInfo::Ptr(rc)) => rc,
            _ => return None,
        };
        Some(PtrInfoBorrow::new(rc))
    }

    /// Live `Rc<RefCell<PtrInfo>>` handle. Use when callers need to
    /// retain identity (e.g. `Rc::ptr_eq`-based `same_info`) or pass the
    /// handle elsewhere without the borrow guard.
    pub fn ptr_info_handle(&self) -> Option<Rc<std::cell::RefCell<PtrInfo>>> {
        match self.get_forwarded() {
            Forwarded::Info(OpInfo::Ptr(rc)) => Some(rc),
            _ => None,
        }
    }

    /// `optimizer.py:99-113 getintbound` BoxRef-native reader.
    ///
    /// When `_forwarded` is `Info(OpInfo::IntBound(rc))`, returns an
    /// `IntBoundBorrow` guard around the live `Rc<RefCell<IntBound>>`.
    /// Other states return `None`.  Same caller-walks-the-chain contract
    /// as `ptr_info`.
    pub fn int_bound(&self) -> Option<IntBoundBorrow> {
        let rc = match self.get_forwarded() {
            Forwarded::Info(OpInfo::IntBound(rc)) => rc,
            _ => return None,
        };
        Some(IntBoundBorrow::new(rc))
    }

    /// Live `Rc<RefCell<IntBound>>` handle.
    pub fn int_bound_handle(&self) -> Option<Rc<std::cell::RefCell<IntBound>>> {
        match self.get_forwarded() {
            Forwarded::Info(OpInfo::IntBound(rc)) => Some(rc),
            _ => None,
        }
    }

    /// Mutable counterpart of `ptr_info`.
    ///
    /// PyPy's `optimizer.py:99-113` mutates the `PtrInfo` returned from
    /// `box.get_forwarded()` in place — Python objects are reference types,
    /// so any `info.<method>(...)` call on the returned object mutates the
    /// `_forwarded` slot's contents directly. The Rust mirror exposes that
    /// through a `PtrInfoBorrowMut` guard that exclusively borrows the
    /// inner `Rc<RefCell<PtrInfo>>`.
    ///
    /// Holds the inner `RefCell` borrow for the lifetime of the returned
    /// guard; callers must drop the guard before any other access to the
    /// same handle.  The outer `forwarded` `RefCell` is released as soon
    /// as the `Rc` clone is captured, so other consumers can still take
    /// non-conflicting borrows of `self.0.forwarded`.
    pub fn ptr_info_mut(&self) -> Option<PtrInfoBorrowMut> {
        let rc = match self.get_forwarded() {
            Forwarded::Info(OpInfo::Ptr(rc)) => rc,
            _ => return None,
        };
        Some(PtrInfoBorrowMut::new(rc))
    }

    /// Mutable counterpart of `int_bound`. Same contract as `ptr_info_mut`.
    pub fn int_bound_mut(&self) -> Option<IntBoundBorrowMut> {
        let rc = match self.get_forwarded() {
            Forwarded::Info(OpInfo::IntBound(rc)) => rc,
            _ => return None,
        };
        Some(IntBoundBorrowMut::new(rc))
    }
}

impl Clone for BoxRef {
    fn clone(&self) -> Self {
        Self(Rc::clone(&self.0))
    }
}

/// Owning borrow guard for `BoxRef::ptr_info()`.
///
/// Carries an `Rc` clone of the live `Rc<RefCell<PtrInfo>>` together
/// with a shared `RefCell` borrow into it.  `Deref<Target = PtrInfo>`
/// gives ergonomic read-only access.
///
/// SAFETY: The inner `Ref<'static, PtrInfo>` is constructed by widening
/// a `Ref` whose true lifetime is bounded by `_rc` (the `Rc` clone we
/// own).  Field declaration order ensures `inner` drops before `_rc`,
/// so the `RefCell::release` runs while the allocation is still alive.
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

/// Owning borrow guard for `BoxRef::int_bound()`.  Same shape as
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

impl PartialEq for BoxRef {
    fn eq(&self, other: &Self) -> bool {
        // PyPy `__eq__` on `AbstractValue` defaults to Python `is` for all
        // box subclasses (ResOp / InputArg / Const). Value equality on
        // constants is the explicit `same_constant` method, not `==`.
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for BoxRef {}

impl std::hash::Hash for BoxRef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.0) as usize).hash(state);
    }
}

impl std::fmt::Debug for BoxRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.0.kind {
            BoxKind::ResOp { .. } => "ResOp",
            BoxKind::InputArg { .. } => "InputArg",
            BoxKind::Const { .. } => "Const",
        };
        write!(
            f,
            "BoxRef@{:p}({:?},{})",
            Rc::as_ptr(&self.0),
            self.0.type_,
            kind
        )
    }
}

// Quiet `OpRef` import warning if no other item references it directly.
// `OpRef` is part of the public Box / BoxRef API surface even when not
// named in this module's signatures.
const _: fn() = || {
    let _ = std::mem::size_of::<OpRef>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intbound::IntBound;
    use crate::op_info::OpInfo;
    use crate::ptr_info::PtrInfo;

    #[test]
    fn box_ref_identity_is_pointer_equality() {
        let a = BoxRef::new_resop(Type::Int, 0);
        let cloned = a.clone();
        let other = BoxRef::new_resop(Type::Int, 1);
        assert_eq!(a, cloned);
        assert_ne!(a, other);
    }

    #[test]
    fn forwarded_chain_walk_returns_terminal() {
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 1);
        let c = BoxRef::new_resop(Type::Int, 2);
        a.set_forwarded_box(b.clone());
        b.set_forwarded_box(c.clone());
        assert_eq!(a.get_box_replacement(false), c);
        assert_eq!(b.get_box_replacement(false), c);
        assert_eq!(c.get_box_replacement(false), c);
    }

    #[test]
    fn forwarded_chain_stops_at_info() {
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 1);
        a.set_forwarded_box(b.clone());
        b.set_forwarded_info(OpInfo::Unknown);
        assert_eq!(a.get_box_replacement(false), b);
    }

    #[test]
    fn forwarded_chain_not_const_stops_before_const() {
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 1);
        let c = BoxRef::new_const(Value::Int(42));
        a.set_forwarded_box(b.clone());
        b.set_forwarded_box(c.clone());
        assert_eq!(a.get_box_replacement(true), b);
        assert_eq!(a.get_box_replacement(false), c);
    }

    #[test]
    fn const_box_kind_and_type() {
        let i = BoxRef::new_const(Value::Int(7));
        assert!(i.is_constant());
        assert_eq!(i.const_value(), Some(Value::Int(7)));
        assert_eq!(i.type_(), Type::Int);

        let f = BoxRef::new_const(Value::Float(1.5));
        assert!(f.is_constant());
        assert_eq!(f.type_(), Type::Float);
    }

    /// PyPy `__eq__` defaults to `is` on `AbstractValue` for ALL box
    /// subclasses (ResOp / InputArg / Const). Value comparison on constants
    /// is the explicit `same_box` / `same_constant` method
    /// (`history.py:204`), not `==`. `BoxRef::eq` therefore stays pointer
    /// identity (`Rc::ptr_eq`) across every variant.
    #[test]
    fn boxref_eq_is_pointer_identity_for_every_variant() {
        use crate::vec_set::VecSet;

        let a = BoxRef::new_const(Value::Int(42));
        let b = BoxRef::new_const(Value::Int(42));
        assert_ne!(a.as_ptr(), b.as_ptr());
        assert_ne!(a, b);
        assert_eq!(a, a.clone());

        let mut set: VecSet<BoxRef> = VecSet::new();
        set.insert(a.clone());
        assert!(set.contains(&a));
        assert!(!set.contains(&b));

        let r1 = BoxRef::new_resop(Type::Int, 0);
        let r2 = BoxRef::new_resop(Type::Int, 0);
        assert_ne!(r1, r2);
        assert_eq!(r1, r1.clone());
    }

    #[test]
    fn inputarg_position_preserved() {
        let arg = BoxRef::new_inputarg(Type::Ref, 3);
        assert!(arg.is_inputarg());
        assert_eq!(arg.position(), Some(3));
        assert_eq!(arg.type_(), Type::Ref);
    }

    #[test]
    fn clear_forwarded_resets_slot() {
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_box(b.clone());
        a.clear_forwarded();
        assert_eq!(a.get_box_replacement(false), a);
        assert!(matches!(a.get_forwarded(), Forwarded::None));
    }

    #[test]
    fn boxref_used_as_assoc_key() {
        use crate::vec_assoc::VecAssoc;
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 0);
        let mut m: VecAssoc<BoxRef, i32> = VecAssoc::new();
        m.insert(a.clone(), 1);
        m.insert(b.clone(), 2);
        assert_eq!(m.get(&a), Some(&1));
        assert_eq!(m.get(&b), Some(&2));
        assert_eq!(m.get(&a.clone()), Some(&1));
    }

    #[test]
    fn resop_position_round_trips() {
        let r = BoxRef::new_resop(Type::Int, 42);
        assert_eq!(r.position(), Some(42));
    }

    #[test]
    fn position_returns_inputarg_position() {
        let arg = BoxRef::new_inputarg(Type::Ref, 7);
        assert_eq!(arg.position(), Some(7));
    }

    #[test]
    fn position_is_none_for_const() {
        let c = BoxRef::new_const(Value::Int(5));
        assert_eq!(c.position(), None);
    }

    #[test]
    #[should_panic]
    fn set_forwarded_to_self_panics() {
        let a = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_box(a.clone());
    }

    #[test]
    fn ptr_info_returns_inner_when_forwarded_is_ptr_info() {
        let a = BoxRef::new_resop(Type::Ref, 0);
        a.set_forwarded_info(OpInfo::ptr(PtrInfo::nonnull()));
        let pi = a.ptr_info().expect("ptr_info should return Some");
        assert!(pi.is_nonnull());
    }

    #[test]
    fn ptr_info_returns_none_for_unset_box() {
        let a = BoxRef::new_resop(Type::Ref, 0);
        assert!(a.ptr_info().is_none());
    }

    #[test]
    fn ptr_info_returns_none_for_box_forwarded() {
        let a = BoxRef::new_resop(Type::Ref, 0);
        let b = BoxRef::new_resop(Type::Ref, 0);
        a.set_forwarded_box(b.clone());
        assert!(a.ptr_info().is_none());
    }

    #[test]
    fn ptr_info_returns_none_for_intbound_forwarded() {
        let a = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(7)));
        assert!(a.ptr_info().is_none());
    }

    #[test]
    fn int_bound_returns_inner_when_forwarded_is_intbound() {
        let a = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(42)));
        let ib = a.int_bound().expect("int_bound should return Some");
        assert!(ib.is_constant());
        assert_eq!(ib.get_constant_int(), 42);
    }

    #[test]
    fn int_bound_returns_none_for_unset_box() {
        let a = BoxRef::new_resop(Type::Int, 0);
        assert!(a.int_bound().is_none());
    }

    #[test]
    fn int_bound_returns_none_for_box_forwarded() {
        let a = BoxRef::new_resop(Type::Int, 0);
        let b = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_box(b.clone());
        assert!(a.int_bound().is_none());
    }

    #[test]
    fn int_bound_returns_none_for_ptrinfo_forwarded() {
        let a = BoxRef::new_resop(Type::Ref, 0);
        a.set_forwarded_info(OpInfo::ptr(PtrInfo::nonnull()));
        assert!(a.int_bound().is_none());
    }

    #[test]
    fn ptr_info_mut_mutates_inner_in_place() {
        let a = BoxRef::new_resop(Type::Ref, 0);
        a.set_forwarded_info(OpInfo::ptr(PtrInfo::nonnull()));
        {
            let mut pi = a.ptr_info_mut().expect("ptr_info_mut should return Some");
            pi.set_last_guard_pos(7);
        }
        let pi = a
            .ptr_info()
            .expect("ptr_info should return Some after mutation");
        assert_eq!(pi.last_guard_pos(), Some(7));
    }

    #[test]
    fn ptr_info_mut_returns_none_for_intbound_forwarded() {
        let a = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(7)));
        assert!(a.ptr_info_mut().is_none());
    }

    #[test]
    fn int_bound_mut_mutates_inner_in_place() {
        let a = BoxRef::new_resop(Type::Int, 0);
        a.set_forwarded_info(OpInfo::int_bound(IntBound::unbounded()));
        {
            let mut ib = a.int_bound_mut().expect("int_bound_mut should return Some");
            let _ = ib.make_eq_const(99);
        }
        let ib = a
            .int_bound()
            .expect("int_bound should return Some after mutation");
        assert!(ib.is_constant());
        assert_eq!(ib.get_constant_int(), 99);
    }

    #[test]
    fn int_bound_mut_returns_none_for_ptrinfo_forwarded() {
        let a = BoxRef::new_resop(Type::Ref, 0);
        a.set_forwarded_info(OpInfo::ptr(PtrInfo::nonnull()));
        assert!(a.int_bound_mut().is_none());
    }

    /// After `bind_op`, `set_forwarded_*` dual-writes through
    /// to `op.forwarded`, so a reader on `op.forwarded` sees the same
    /// state as `box.get_forwarded()`.
    #[test]
    fn bind_op_makes_set_forwarded_dual_write_to_op() {
        use crate::resoperation::{Op, OpCode};
        let op = std::rc::Rc::new(Op::new(OpCode::IntAdd, &[]));
        let b = BoxRef::new_resop(Type::Int, 0);
        b.bind_op(&op);

        // Initially both are Forwarded::None.
        assert!(matches!(b.get_forwarded(), Forwarded::None));
        assert!(matches!(*op.forwarded.borrow(), Forwarded::None));

        // set_forwarded_info → both slots updated.
        b.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(42)));
        assert!(matches!(b.get_forwarded(), Forwarded::Info(_)));
        assert!(matches!(*op.forwarded.borrow(), Forwarded::Info(_)));

        // clear_forwarded → both slots reset.
        b.clear_forwarded();
        assert!(matches!(b.get_forwarded(), Forwarded::None));
        assert!(matches!(*op.forwarded.borrow(), Forwarded::None));

        // set_forwarded_box → both slots carry the target.
        let target = BoxRef::new_resop(Type::Int, 1);
        b.set_forwarded_box(target.clone());
        match (&b.get_forwarded(), &*op.forwarded.borrow()) {
            (Forwarded::Box(box_target), Forwarded::Box(op_target)) => {
                assert_eq!(box_target, op_target);
            }
            _ => panic!("expected Forwarded::Box on both slots"),
        }
    }

    /// `bind_op` stores `Weak<Op>`; once the `Rc<Op>` is dropped, the
    /// dual-write becomes a no-op (Weak::upgrade returns None) without
    /// panicking. Safety net.
    #[test]
    fn dropped_op_makes_dual_write_a_noop() {
        use crate::resoperation::{Op, OpCode};
        let b = BoxRef::new_resop(Type::Int, 0);
        {
            let op = std::rc::Rc::new(Op::new(OpCode::IntAdd, &[]));
            b.bind_op(&op);
            // op drops here.
        }
        // Dual-write target is gone; set_forwarded should not panic.
        b.set_forwarded_info(OpInfo::Unknown);
        assert!(matches!(b.get_forwarded(), Forwarded::Info(_)));
    }

    /// CodeRabbit mod.rs:2204 scenario: `ensure_box` materializes an
    /// unbound ResOp placeholder; subsequent code writes Info / IntBound
    /// / PtrInfo onto it through `BoxRef::set_forwarded_*` (which lands
    /// in `Box.forwarded` for unbound boxes). When the producer is
    /// emitted and `bind_op` runs, the pre-emit state must reach
    /// `op.forwarded` so reads routed through the bound op see it.
    #[test]
    fn bind_op_carries_pre_emit_forwarded_to_new_op() {
        use crate::resoperation::{Op, OpCode};
        let placeholder = BoxRef::new_resop(Type::Int, 5);
        placeholder.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(13)));

        let producer = std::rc::Rc::new(Op::new(OpCode::IntAdd, &[]));
        placeholder.bind_op(&producer);

        // The pre-emit Info forwarding is reachable on the bound op.
        match &*producer.forwarded.borrow() {
            Forwarded::Info(_) => {}
            other => panic!("bind_op dropped the pre-emit forwarding: {other:?}"),
        }
        // And via the BoxRef accessor too.
        assert!(matches!(placeholder.get_forwarded(), Forwarded::Info(_)));
    }

    /// `bind_inputarg` panics on a non-InputArg box (same contract as
    /// `bind_op`'s ResOp-only check).
    #[test]
    #[should_panic(expected = "bind_inputarg only valid for InputArg boxes")]
    fn bind_inputarg_on_resop_panics() {
        use crate::value::InputArg;
        let b = BoxRef::new_resop(Type::Int, 0);
        let ia = InputArg::new_int_rc(0);
        b.bind_inputarg(&ia);
    }

    /// `bind_inputarg` carries pre-bind `Box.forwarded` state into
    /// `inputarg.forwarded`, mirroring `bind_op`'s carry-over.
    #[test]
    fn bind_inputarg_carries_pre_bind_forwarded_to_inputarg() {
        use crate::value::InputArg;
        let b = BoxRef::new_inputarg(Type::Int, 3);
        b.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(7)));

        let ia = InputArg::new_int_rc(3);
        b.bind_inputarg(&ia);

        // The pre-bind Info forwarding survives on the InputArg slot.
        assert!(matches!(*ia.forwarded.borrow(), Forwarded::Info(_)));
        // bound_inputarg upgrades the Weak.
        assert!(b.bound_inputarg().is_some());
    }

    /// `bound_inputarg` returns `None` for unbound boxes (no Weak set)
    /// and for non-InputArg variants (the field is reserved for InputArg).
    #[test]
    fn bound_inputarg_none_for_unbound_and_wrong_kind() {
        let ia_box = BoxRef::new_inputarg(Type::Int, 0);
        assert!(ia_box.bound_inputarg().is_none());

        let resop_box = BoxRef::new_resop(Type::Int, 0);
        assert!(resop_box.bound_inputarg().is_none());
    }

    /// After `bind_inputarg`, `set_forwarded_*` writes through to
    /// `inputarg.forwarded` and `get_forwarded` reads from it — same
    /// dual-write invariant established for ResOp.
    #[test]
    fn bind_inputarg_makes_set_forwarded_dual_write_to_inputarg() {
        use crate::value::InputArg;
        let b = BoxRef::new_inputarg(Type::Int, 0);
        let ia = InputArg::new_int_rc(0);
        b.bind_inputarg(&ia);

        // Initial state.
        assert!(matches!(b.get_forwarded(), Forwarded::None));
        assert!(matches!(*ia.forwarded.borrow(), Forwarded::None));

        // Info write: inputarg slot reflects it; get_forwarded reads it back.
        b.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(99)));
        assert!(matches!(b.get_forwarded(), Forwarded::Info(_)));
        assert!(matches!(*ia.forwarded.borrow(), Forwarded::Info(_)));

        // Box write should also reach the InputArg slot.
        let target = BoxRef::new_resop(Type::Int, 5);
        b.set_forwarded_box(target.clone());
        match (&b.get_forwarded(), &*ia.forwarded.borrow()) {
            (Forwarded::Box(bt), Forwarded::Box(iat)) => assert_eq!(bt, iat),
            _ => panic!("expected Forwarded::Box on both slots"),
        }

        // Clear reaches the InputArg slot.
        b.clear_forwarded();
        assert!(matches!(b.get_forwarded(), Forwarded::None));
        assert!(matches!(*ia.forwarded.borrow(), Forwarded::None));
    }

    /// If the bound `InputArgRc` is dropped, `write_forwarded` falls back
    /// to `Box.forwarded` instead of panicking — symmetric to
    /// `dropped_op_makes_dual_write_a_noop`.
    #[test]
    fn dropped_inputarg_makes_write_fall_back_to_box() {
        use crate::value::InputArg;
        let b = BoxRef::new_inputarg(Type::Int, 0);
        {
            let ia = InputArg::new_int_rc(0);
            b.bind_inputarg(&ia);
            // ia drops here.
        }
        b.set_forwarded_info(OpInfo::Unknown);
        assert!(matches!(b.get_forwarded(), Forwarded::Info(_)));
    }

    /// Cross-pass snapshots (e.g. test fixtures cloning a BoxRef whose
    /// originating `OpRc` then drops) outlive the `Weak<Op>` referent.
    /// Writes done while the Op is alive must also land in `Box.forwarded`
    /// so the snapshot keeps the latest forwarding once `Weak::upgrade`
    /// fails.
    #[test]
    fn write_forwarded_mirrors_to_box_so_snapshot_survives_op_drop() {
        use crate::resoperation::{Op, OpCode};
        let b = BoxRef::new_resop(Type::Int, 0);
        let snapshot = b.clone();
        {
            let op = std::rc::Rc::new(Op::new(OpCode::IntAdd, &[]));
            b.bind_op(&op);
            b.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(7)));
            // op drops at the end of this block.
        }
        // The snapshot still sees the Info forwarding because the writer
        // mirrored to Box.forwarded alongside op.forwarded.
        assert!(matches!(snapshot.get_forwarded(), Forwarded::Info(_)));
    }

    /// Reviewer scenario: `bind A → set_forwarded(X) → bind B`. PyPy's
    /// single `_forwarded` slot model preserves X across the rebind
    /// (resoperation.py:233-240). The rust port must do the same — `bind_op`
    /// reads via `get_forwarded()` so it carries X into B even if a
    /// future writer ever updates `A.forwarded` without going through
    /// `BoxRef::set_forwarded_*`.
    #[test]
    fn rebind_carries_forwarded_state_to_new_op() {
        use crate::resoperation::{Op, OpCode};
        let b = BoxRef::new_resop(Type::Int, 0);
        let op_a = std::rc::Rc::new(Op::new(OpCode::IntAdd, &[]));
        b.bind_op(&op_a);
        b.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(42)));

        // Simulate a direct write that bypasses BoxRef so the in-Box
        // mirror could plausibly diverge in a future migration step.
        *op_a.forwarded.borrow_mut() = Forwarded::Info(OpInfo::Unknown);

        let op_b = std::rc::Rc::new(Op::new(OpCode::IntAdd, &[]));
        b.bind_op(&op_b);

        // op_b must carry the canonical effective forwarding (the post-
        // direct-write state on op_a), not the stale Box mirror.
        assert!(matches!(*op_b.forwarded.borrow(), Forwarded::Info(_)));
        match &*op_b.forwarded.borrow() {
            Forwarded::Info(OpInfo::Unknown) => {}
            other => panic!("rebind dropped the latest forwarding: {other:?}"),
        }
    }

    /// Same invariant for InputArg rebinding.
    #[test]
    fn rebind_inputarg_carries_forwarded_state_to_new_ia() {
        use crate::value::InputArg;
        let b = BoxRef::new_inputarg(Type::Int, 0);
        let ia_a = InputArg::new_int_rc(0);
        b.bind_inputarg(&ia_a);
        b.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(11)));

        *ia_a.forwarded.borrow_mut() = Forwarded::Info(OpInfo::Unknown);

        let ia_b = InputArg::new_int_rc(0);
        b.bind_inputarg(&ia_b);

        match &*ia_b.forwarded.borrow() {
            Forwarded::Info(OpInfo::Unknown) => {}
            other => panic!("inputarg rebind dropped the latest forwarding: {other:?}"),
        }
    }

    /// Same invariant for InputArg-bound boxes.
    #[test]
    fn write_forwarded_mirrors_to_box_so_snapshot_survives_inputarg_drop() {
        use crate::value::InputArg;
        let b = BoxRef::new_inputarg(Type::Int, 0);
        let snapshot = b.clone();
        {
            let ia = InputArg::new_int_rc(0);
            b.bind_inputarg(&ia);
            b.set_forwarded_info(OpInfo::Unknown);
            // ia drops here.
        }
        assert!(matches!(snapshot.get_forwarded(), Forwarded::Info(_)));
    }
}
