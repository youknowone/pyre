//! Rust mirror of RPython's `AbstractValue` object identity.
//!
//! Direct port of the Python object identity hierarchy formed by
//! `rpython/jit/metainterp/resoperation.py:29 AbstractValue` together with
//! `AbstractResOpOrInputArg` / `AbstractResOp` / `AbstractInputArg` /
//! `Const*` (`history.py:182`), expressed as `Rc<Box>`.
//!
//! Hosted in `majit-ir` so `BoxRef` can carry `Weak<Op>` / `Weak<InputArg>`
//! without a `majit-metainterp → majit-ir` circular dep. The canonical
//! `_forwarded` slot lives on `Op` / `InputArg` themselves
//! (`resoperation.py:233` / `:700`); `BoxRef` is a thin wrapper that
//! routes `get_forwarded` / `set_forwarded_*` through the bound handle.
//!
//! # Design decisions
//!
//! - The canonical `forwarded` slot is a `RefCell<Forwarded>` on
//!   `Op` / `InputArg`. `Cell` is not used because `Forwarded` carries
//!   `OpInfo` / `BoxRef`, neither of which is `Copy`. Helpers terminate
//!   the borrow scope immediately after reading.
//! - `BoxRef`'s `Eq` / `Hash` use `Rc::ptr_eq` / `Rc::as_ptr` — equivalent
//!   to RPython's use of object identity as a dict key.
//! - When `Forwarded::Box(BoxRef)` carries a BoxRef whose kind is
//!   `BoxKind::Const(...)`, that mirrors RPython's
//!   `box.set_forwarded(constbox)`. Constants are split into
//!   `Forwarded::Const(Const)` separately so chain walkers terminate on
//!   the inline value without needing a `BoxKind::Const` carrier.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use crate::intbound::IntBound;
use crate::op_info::OpInfo;
use crate::ptr_info::PtrInfo;
use crate::resoperation::Op;
use crate::value::{Const, InputArg};
use crate::{GcRef, OpRef, Type, Value};

/// `AbstractValue` mirror — unified representation of RPython's
/// op/inputarg/const objects.
pub struct Box {
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

    /// Canonical `_forwarded` host backref for `BoxKind::ResOp`
    /// (`resoperation.py:233 AbstractResOpOrInputArg._forwarded` —
    /// the slot lives on the `Op` itself, not on this wrapper).
    /// `Weak` to avoid an `Rc` cycle (`Op.forwarded` may carry a
    /// `BoxRef` holding an `Rc<Box>` whose `op_handle` could
    /// otherwise loop back through the trace). Empty for
    /// `BoxKind::InputArg` / `Const`; filled at construction
    /// (`BoxRef::from_bound_op`) or by `BoxRef::bind_op` during emit's
    /// `bound_is_synthetic` rebind path (`mod.rs::emit`).
    /// `BoxRef::get_forwarded` / `set_forwarded_*` route through this
    /// handle exclusively for ResOp boxes — there is no `Box`-side mirror
    /// to consult.
    pub op_handle: RefCell<Option<Weak<Op>>>,

    /// Canonical `_forwarded` host backref for `BoxKind::InputArg`
    /// (`resoperation.py:700 AbstractInputArg._forwarded`). Empty for
    /// `BoxKind::ResOp` / `Const`; filled by `BoxRef::bind_inputarg`
    /// at the recorder→TreeLoop handoff and by
    /// `OptContext::ensure_inputarg_bindings` /
    /// `bind_input_resops` for per-iter TraceIterator pools.
    /// `set_forwarded_*` route through this handle exclusively for
    /// InputArg boxes.
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
    /// The per-type mixin slot (`resoperation.py:566 IntOp._resint`,
    /// `resoperation.py:612 RefOp._resref`,
    /// `resoperation.py:582 FloatOp._resfloat`) lives on the outer
    /// `Box.value: Cell<Option<Value>>` field — single storage
    /// matching RPython's single `_resint`/`_resref`/`_resfloat`.
    ResOp { position: std::cell::Cell<u32> },

    /// `resoperation.py:699 AbstractInputArg`.
    /// `position` mirrors `AbstractInputArg.position`
    /// (resoperation.py:699).
    /// Mixin value slot lives on outer `Box.value`.
    InputArg { position: u32 },

    /// `history.py:220 ConstInt` / `:261 ConstFloat` / `:307 ConstPtr`.
    /// The Const carries its `value` directly (history.py:227/268/314); the
    /// chain walker reconstructs the inline-Const OpRef from it.
    ///
    /// `Cell` so the GC root walker can forward an inline `Value::Ref`
    /// `GcRef` in place via a get/set cycle (`walk_const_ptr_refs`) without
    /// a `&mut Box`. The wrapped `Value` is otherwise immutable.
    Const { value: Cell<Value> },

    /// Absent-reference sentinel — the `BoxRef` mirror of `OpRef::None`.
    /// Fills a `fail_args` hole in-place (failargs hold `None` entries for
    /// slots that were never written), preserving arity. Carries no value,
    /// position, or meaningful `type_` (`Type::Void`); every accessor
    /// treats it as empty.
    None,
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
    /// terminal const-bearing `BoxRef`).
    ///
    /// PyPy has no analog (callers hold the Python `Const` object
    /// directly); `box_to_opref` reconstructs the inline-Const OpRef
    /// from the payload value (history.py:227/268/314).
    Const(Const),

    /// `optimizeopt/info.py:17 AbstractInfo (is_info_class = True)` family —
    /// `PtrInfo`, `IntBound`, `FloatConstInfo`, `EmptyInfo`, etc.
    Info(OpInfo),
    // There is deliberately NO `VectorInfo` variant. Vectorizer scheduling
    // scratch (`schedule.py:20-28 forwarded_vecinfo`) lives in the pos/OpRef-
    // keyed `VecScheduleState.vecinfo_cache` (optimizeopt/schedule.rs), not in
    // a `_forwarded` slot, on purpose: `Op::clone` resets `forwarded` to
    // `None` (fresh identity) while preserving `pos`, and the scheduler reads
    // vecinfo off cloned ops, so a `_forwarded`-borne scratch clone-drops and
    // silently miscomputes INT_SIGNEXT (its dynamic arg1 bytesize resolves
    // only at setup time through the const resolver that the bare
    // `vectorization_info_for_op` reader does not hold). Loop `InputArg`
    // operands also own no per-op vecinfo slot, so an OpRef-keyed cache is
    // required regardless. `Op.vecinfo` (resoperation.rs) is the SEPARATE
    // permanent `resoperation.py:511-518` VecOp datatype/bytesize/signed/count
    // store and stays. Reviews recurringly flag the cache as a `_forwarded`
    // parity regression — it is the clone-stable successor to the retired
    // opref-keyed box_pool, not a regression; do not restore.
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
            type_,
            kind: BoxKind::ResOp {
                position: std::cell::Cell::new(position),
            },
            value: Cell::new(None),
            op_handle: RefCell::new(None),
            inputarg_handle: RefCell::new(None),
        }))
    }

    /// `AbstractResOp` Box wrapping an already-bound `OpRc`. Used by the
    /// chain walker to materialize a `BoxRef` terminal from a
    /// `Forwarded::Op(Weak<Op>)` payload and by `resolve_to_boxref`; its
    /// `bound_op` answers with the same `Rc<Op>` and its `_forwarded`
    /// slot reads via the bound op per `get_forwarded`.
    ///
    /// The wrapper is MEMOIZED on the op (`Op::box_cache`): the op object
    /// IS its own `AbstractValue`, so every call for the same `Rc<Op>`
    /// returns the SAME `Rc<Box>`, giving stable pointer identity
    /// (`Rc::ptr_eq` == `self is other`). The cached box's `position`
    /// (`BoxKind::ResOp`) is refreshed from the current `op.pos` on every
    /// call, since `op.pos` is mutable (recorder `op.pos.set`, unroll
    /// resume-retarget, const-pool compaction). `type_` is not refreshed —
    /// `Op.type_` is immutable and `op.pos`'s variant kind never changes
    /// type for a given op.
    pub fn from_bound_op(op: &crate::resoperation::OpRc) -> Self {
        {
            let cache = op.box_cache.borrow();
            if let Some(cached) = cache.as_ref() {
                cached.set_position(op.pos.get().raw());
                return cached.clone();
            }
        }
        let opref = op.pos.get();
        let type_ = opref.ty().unwrap_or(Type::Void);
        let position = opref.raw();
        let boxref = Self(Rc::new(Box {
            type_,
            kind: BoxKind::ResOp {
                position: std::cell::Cell::new(position),
            },
            value: Cell::new(None),
            op_handle: RefCell::new(Some(Rc::downgrade(op))),
            inputarg_handle: RefCell::new(None),
        }));
        *op.box_cache.borrow_mut() = Some(boxref.clone());
        boxref
    }

    /// `AbstractInputArg` Box wrapping an already-bound `InputArgRc`.
    /// Mirror of `from_bound_op` for the chain walker's
    /// `Forwarded::InputArg` terminal materialization and for
    /// `resolve_to_boxref`. MEMOIZED on the input arg
    /// (`InputArg::box_cache`) so every call for the same `Rc<InputArg>`
    /// returns the SAME `Rc<Box>`. No position refresh: `InputArg.index`
    /// is immutable (`resoperation.py:699 AbstractInputArg.position`).
    pub fn from_bound_inputarg(ia: &crate::value::InputArgRc) -> Self {
        {
            let cache = ia.box_cache.borrow();
            if let Some(cached) = cache.as_ref() {
                return cached.clone();
            }
        }
        let type_ = ia.tp;
        let position = ia.index;
        let boxref = Self(Rc::new(Box {
            type_,
            kind: BoxKind::InputArg { position },
            value: Cell::new(None),
            op_handle: RefCell::new(None),
            inputarg_handle: RefCell::new(Some(Rc::downgrade(ia))),
        }));
        *ia.box_cache.borrow_mut() = Some(boxref.clone());
        boxref
    }

    /// New `AbstractInputArg` Box.
    pub fn new_inputarg(type_: Type, position: u32) -> Self {
        Self(Rc::new(Box {
            type_,
            kind: BoxKind::InputArg { position },
            value: Cell::new(None),
            op_handle: RefCell::new(None),
            inputarg_handle: RefCell::new(None),
        }))
    }

    /// New `Const*` Box. `type_` is inferred from `value`.
    pub fn new_const(value: Value) -> Self {
        let type_ = value.get_type();
        Self(Rc::new(Box {
            type_,
            kind: BoxKind::Const {
                value: Cell::new(value),
            },
            value: Cell::new(None),
            op_handle: RefCell::new(None),
            inputarg_handle: RefCell::new(None),
        }))
    }

    /// Absent-reference sentinel — the `BoxRef` mirror of `OpRef::None`.
    /// Fills a `fail_args` hole in-place (failargs hold `None` entries for
    /// slots that were never written), preserving arity. `type_` is
    /// `Void`; carries no value, position, or forwarding.
    pub fn none() -> Self {
        Self(Rc::new(Box {
            type_: Type::Void,
            kind: BoxKind::None,
            value: Cell::new(None),
            op_handle: RefCell::new(None),
            inputarg_handle: RefCell::new(None),
        }))
    }

    /// Bind this Box to its corresponding `Op` so subsequent
    /// `set_forwarded_*` / `clear_forwarded` calls write through to
    /// `op.forwarded` (the canonical host; there is no Box-side mirror).
    /// Stores a `Weak<Op>` to avoid an Rc cycle. Called when re-binding a
    /// box to its producer (`ensure_box`'s synthetic-mint path, test
    /// fixtures). Panics if called on a non-ResOp Box.
    ///
    /// Late-binding carry-over: when the box is *already bound* (a rebind:
    /// `bind → set_forwarded → rebind`, e.g. `emit()` re-pointing a
    /// synthetic placeholder at its real producer), its effective forwarded
    /// state (`self.get_forwarded()`) is transferred into the new `Op` host.
    /// A freshly minted, still-unbound box has no forwarded state of its own
    /// — S-0.C removed the Box-side mirror, so `get_forwarded()` on an
    /// unbound box is always `Forwarded::None`. Carrying that `None` would
    /// *clobber* an already-populated canonical host's authoritative
    /// `_forwarded` (the bug that `box_pool` memoization used to mask by
    /// returning the same bound box on a repeat `ensure_box`). So the
    /// carry-over fires only when `self` is bound; binding a fresh box leaves
    /// the host's `_forwarded` intact.
    pub fn bind_op(&self, op: &crate::resoperation::OpRc) {
        assert!(
            matches!(&self.0.kind, BoxKind::ResOp { .. }),
            "BoxRef::bind_op only valid for ResOp boxes"
        );
        if self.bound_op().is_some() {
            let carry = self.get_forwarded();
            *op.forwarded.borrow_mut() = carry;
        }
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
    /// box. Late-binding carry-over follows the same rule as `bind_op`: it
    /// fires only when `self` is already bound (a rebind that carries real
    /// state), never when binding a freshly minted box — carrying an unbound
    /// box's `Forwarded::None` would clobber the canonical host's
    /// authoritative `_forwarded`.
    pub fn bind_inputarg(&self, ia: &crate::value::InputArgRc) {
        assert!(
            matches!(&self.0.kind, BoxKind::InputArg { .. }),
            "BoxRef::bind_inputarg only valid for InputArg boxes"
        );
        if self.bound_inputarg().is_some() {
            let carry = self.get_forwarded();
            *ia.forwarded.borrow_mut() = carry;
        }
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

    /// Return the source OpRef the Const Box was constructed from.
    /// Returns `None` for non-Const boxes. The Const carries its value
    /// directly (history.py:227/268/314), so reconstruction is the inline
    /// variant.
    pub fn source_opref(&self) -> Option<OpRef> {
        match &self.0.kind {
            BoxKind::Const { value } => {
                let opref = match value.get() {
                    Value::Int(v) => OpRef::const_int(v),
                    Value::Float(v) => OpRef::const_float(v),
                    Value::Ref(v) => OpRef::const_ptr(v),
                    Value::Void => OpRef::None,
                };
                Some(opref)
            }
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

    /// True for the `none()` sentinel — the `BoxRef` mirror of
    /// `OpRef::is_none` (an absent `fail_args` slot).
    pub fn is_none(&self) -> bool {
        matches!(self.0.kind, BoxKind::None)
    }

    /// `resoperation.py:233 AbstractResOpOrInputArg._pos` accessor.
    ///
    /// Returns the index where this box resides in the pool for
    /// `ResOp` / `InputArg` (which are the two PyPy classes that own
    /// `_pos`). `Const` has no canonical position and returns `None`.
    pub fn position(&self) -> Option<u32> {
        match &self.0.kind {
            BoxKind::ResOp { position, .. } => Some(position.get()),
            BoxKind::InputArg { position, .. } => Some(*position),
            BoxKind::Const { .. } | BoxKind::None => None,
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
        if let BoxKind::ResOp { position, .. } = &self.0.kind {
            position.set(new_pos);
        }
    }

    /// Extract the constant value. Mirrors `history.py:233 ConstInt.getint`
    /// and the equivalent accessors on the other Const subclasses.
    pub fn const_value(&self) -> Option<Value> {
        match &self.0.kind {
            BoxKind::Const { value, .. } => Some(value.get()),
            _ => None,
        }
    }

    /// `isinstance(box, ConstInt)` + `box.getint()` (history.py:233) — the
    /// raw Const-int accessor with NO IntBound synthesis. Distinct from the
    /// optimizer's `get_constant_int_box`, which also synthesizes ConstInt
    /// from a constant IntBound (optimizer.py:383-386).
    pub fn const_int(&self) -> Option<i64> {
        match self.const_value() {
            Some(Value::Int(i)) => Some(i),
            _ => None,
        }
    }

    /// Bridge: reconstruct the flat `OpRef` view of this box for the
    /// OpRef-keyed side tables and `op.pos` comparisons that the optimizer,
    /// GC rewriter, and backends still maintain while `Op.args` carry
    /// `BoxRef`. A `Const*` box maps to the matching inline-const `OpRef`
    /// (history.py:227/268/314); an InputArg / ResOp box maps to its typed
    /// position. Inverse of [`BoxRef::from_opref`]; the two round-trip
    /// (`from_opref(b.to_opref()) ≡ b` modulo identity).
    pub fn to_opref(&self) -> OpRef {
        if self.is_none() {
            return OpRef::NONE;
        }
        match self.const_value() {
            Some(Value::Int(v)) => return OpRef::const_int(v),
            Some(Value::Float(v)) => return OpRef::const_float(v),
            Some(Value::Ref(v)) => return OpRef::const_ptr(v),
            Some(Value::Void) => return OpRef::NONE,
            None => {}
        }
        let pos = self
            .position()
            .expect("non-const box must carry a position");
        let ty = self.type_();
        if self.is_inputarg() {
            OpRef::input_arg_typed(pos, ty)
        } else {
            OpRef::op_typed(pos, ty)
        }
    }

    /// Bridge inverse of [`BoxRef::to_opref`]: materialize a `BoxRef` view of
    /// an `OpRef` held in a position-keyed side table. ResOp positions become
    /// a position-carrying `new_resop` box (no live op handle — identity by
    /// position, sufficient for the OpRef-keyed consumers that bridge back via
    /// `to_opref`).
    pub fn from_opref(r: OpRef) -> BoxRef {
        if r.is_none() {
            return BoxRef::none();
        }
        if r.is_constant() {
            return match r {
                OpRef::ConstInt(v) => BoxRef::new_const(Value::Int(v)),
                OpRef::ConstFloat(v) => BoxRef::new_const(Value::Float(v)),
                OpRef::ConstPtr(v) => BoxRef::new_const(Value::Ref(v)),
                _ => unreachable!("is_constant but not a Const variant"),
            };
        }
        let ty = r.ty().unwrap_or(Type::Void);
        let pos = r.raw();
        match r {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                BoxRef::new_inputarg(ty, pos)
            }
            _ => BoxRef::new_resop(ty, pos),
        }
    }

    /// GC root walk over an inline `Value::Ref` carried by a `Const` box —
    /// the `BoxRef` mirror of the per-arg `OpRef::ConstPtr` forwarding in
    /// `Op::walk_const_ptr_refs_mut`. The collector forwards in place; the
    /// `Cell` get/set cycle writes the moved address back so the box no
    /// longer holds the stale (pre-move) pointer. No-op for non-`Const`
    /// boxes and for non-`Ref` constants.
    pub fn walk_const_ptr_refs(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        if let BoxKind::Const { value } = &self.0.kind {
            let mut v = value.get();
            if let Value::Ref(gcref) = &mut v {
                visitor(gcref);
                value.set(v);
            }
        }
    }

    /// `resoperation.py:38 AbstractResOpOrInputArg.same_box`: `self is other`
    /// for ResOp / InputArg; `history.py:211 Const.same_box` delegates to
    /// `same_constant` (value comparison, `history.py:251/292/338`).
    ///
    /// `from_bound_op` / `from_bound_inputarg` memoize exactly one `BoxRef`
    /// wrapper per op / inputarg (`Op::box_cache` / `InputArg::box_cache`),
    /// so every resolution of the same ResOp / InputArg shares one `Rc<Box>`
    /// and the `Rc::ptr_eq` of `==` is a faithful `self is other` probe.
    /// Const boxes are minted fresh per resolution (no op to memoize on), so
    /// their identity is compared by value via the `same_constant` arm.
    pub fn same_box(&self, other: &BoxRef) -> bool {
        if self == other {
            return true;
        }
        // Two `none()` sentinels denote the same absent reference
        // (`OpRef::None`; `None is None`).
        if self.is_none() || other.is_none() {
            return self.is_none() && other.is_none();
        }
        match (self.const_value(), other.const_value()) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    /// `resoperation.py:50 get_forwarded`. Returns a clone of the
    /// canonical `_forwarded` slot. For bound ResOp boxes the read is
    /// routed through `op.forwarded` (`resoperation.py:233`); for
    /// bound InputArg boxes through `inputarg.forwarded`
    /// (`resoperation.py:700`). Const boxes and unbound non-Const
    /// boxes return `Forwarded::None` (RPython `Const._forwarded` is
    /// permanently `None`, and unbound non-Consts are an invariant
    /// violation that `write_forwarded`'s precondition assert
    /// catches on the writer side). The clone is cheap — every
    /// `Forwarded` payload is an `Rc`/`Copy` handle.
    pub fn get_forwarded(&self) -> Forwarded {
        if let Some(op) = self.bound_op() {
            return op.forwarded.borrow().clone();
        }
        if let Some(ia) = self.bound_inputarg() {
            return ia.forwarded.borrow().clone();
        }
        Forwarded::None
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
    /// `BoxKind::Const` carrier.
    pub fn set_forwarded_const(&self, value: Const) {
        // Same Const-as-source invariant as the other set_forwarded_*
        // variants — Const has no `_forwarded` slot per PyPy.
        assert!(
            !matches!(self.0.kind, BoxKind::Const { .. }),
            "set_forwarded_const on Const violates RPython AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        self.write_forwarded(Forwarded::Const(value));
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

    /// `history.py:803 IntFrontendOp(pos, intval)` parity — read the
    /// per-Box intrinsic value carrier.  `None` when the mixin slot
    /// has not been stamped (Pyre allocates BoxRefs before the tracer
    /// computes the concrete value; RPython constructs each
    /// `*FrontendOp(pos, value)` with the value already in hand).
    /// Const boxes always return `Some`.
    pub fn get_value(&self) -> Option<Value> {
        match &self.0.kind {
            BoxKind::Const { value, .. } => Some(value.get()),
            _ => {
                // The concrete value's canonical host is the bound
                // `Op`/`InputArg` (`resoperation.py:566 IntOp._resint`).
                // Prefer it; fall back to the transitional `Box.value`
                // slot only for boxes not yet bound to their object
                // (recorder pool entries).
                if let Some(op) = self.bound_op() {
                    if let Some(v) = op.get_value() {
                        return Some(v);
                    }
                } else if let Some(ia) = self.bound_inputarg() {
                    if let Some(v) = ia.get_value() {
                        return Some(v);
                    }
                }
                self.0.value.get()
            }
        }
    }

    /// Generic mixin slot write.  Asserts type consistency as the Rust
    /// equivalent of RPython's class-hierarchy guarantee (`setint` only
    /// on `IntOp`, etc.).  All producer sites must type-gate before
    /// calling: `close_loop_args_at` via `collect_kind(opref, cv)`,
    /// bridge-entry via `heap_value_for(tp, raw)`, optimizer
    /// `setup_optimizations` via Phase 2 remap type filter.
    pub fn set_value(&self, v: Value) {
        match &self.0.kind {
            BoxKind::Const { .. } => {
                panic!("Const boxes do not inherit IntOp/RefOp/FloatOp value slots")
            }
            _ => {
                assert_eq!(
                    self.type_(),
                    v.get_type(),
                    "BoxRef::set_value type mismatch: box {:?}, value {:?}",
                    self.type_(),
                    v.get_type()
                );
                // Stamp the concrete value on the canonical host (the bound
                // `Op`/`InputArg`, `resoperation.py:566 IntOp._resint`).
                // Dual-write the transitional `Box.value` slot so boxes not
                // yet bound to their object keep working.
                if let Some(op) = self.bound_op() {
                    op.set_value(v);
                } else if let Some(ia) = self.bound_inputarg() {
                    ia.set_value(v);
                }
                self.0.value.set(Some(v));
            }
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

    /// Write to the canonical `_forwarded` host — `op.forwarded`
    /// (`resoperation.py:233-242`) for ResOp boxes,
    /// `inputarg.forwarded` (`resoperation.py:700`) for InputArg
    /// boxes. `BoxKind::ResOp` and `BoxKind::InputArg` are mutually
    /// exclusive so at most one branch fires per call.
    ///
    /// Asserts that at least one handle is bound and upgradable; an
    /// unbound or dropped-target write would silently lose data
    /// since there is no `Box`-side mirror to catch it. Production
    /// pre-binds every chain-walker-reachable slot via
    /// `OptContext::ensure_inputarg_bindings` and `bind_input_resops`.
    fn write_forwarded(&self, value: Forwarded) {
        if let Some(op) = self.bound_op() {
            *op.forwarded.borrow_mut() = value;
            return;
        }
        if let Some(ia) = self.bound_inputarg() {
            *ia.forwarded.borrow_mut() = value;
            return;
        }
        panic!(
            "BoxRef::write_forwarded on unbound BoxRef — bind the box to its \
             Op/InputArg before writing forwarded (box identity precondition)"
        );
    }

    /// `resoperation.py:57-68 get_box_replacement(not_const=False)`.
    ///
    /// Walk the `_forwarded` chain, returning the box one step before the
    /// chain hits `None`, `Info`, or (`not_const=true && next.is_constant()`).
    pub fn get_box_replacement(&self, not_const: bool) -> BoxRef {
        let mut cur = self.clone();
        loop {
            match cur.get_forwarded() {
                Forwarded::None | Forwarded::Info(_) => return cur,
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
                Forwarded::Const(c) => {
                    if not_const {
                        return cur;
                    }
                    // Materialize a terminal Const-bearing BoxRef so
                    // callers that expect `.const_value()` / `BoxKind::Const`
                    // on the walker output keep working until the BoxRef
                    // return type itself is retired.
                    return BoxRef::new_const(c.to_value());
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
    /// same handle.  The canonical `forwarded` `RefCell` (on the bound
    /// `Op` / `InputArg`) is released as soon as the `Rc` clone is
    /// captured, so other consumers can still take non-conflicting
    /// borrows of it.
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
            BoxKind::None => "None",
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
    use crate::resoperation::{Op, OpCode, OpRc};
    use crate::value::{InputArg, InputArgRc};

    /// Bind a fresh ResOp BoxRef to a synthetic `SameAs*`/`Jump` OpRc so
    /// `set_forwarded_*` writes land on `op.forwarded` (the canonical
    /// `_forwarded` host per resoperation.py:233). The OpRc must outlive
    /// every write through the returned BoxRef.
    fn bound_resop(tp: Type, position: u32) -> (BoxRef, OpRc) {
        let b = BoxRef::new_resop(tp, position);
        let opcode = match tp {
            Type::Int => OpCode::SameAsI,
            Type::Float => OpCode::SameAsF,
            Type::Ref => OpCode::SameAsR,
            Type::Void => OpCode::Jump,
        };
        let op = std::rc::Rc::new(Op::new(opcode, &[]));
        op.pos.set(OpRef::op_typed(position, tp));
        b.bind_op(&op);
        (b, op)
    }

    /// InputArg counterpart of `bound_resop` — binds a fresh `BoxRef::new_inputarg`
    /// to a fresh `InputArgRc` so writes land on `inputarg.forwarded`
    /// (resoperation.py:700).
    fn bound_inputarg(tp: Type, index: u32) -> (BoxRef, InputArgRc) {
        let b = BoxRef::new_inputarg(tp, index);
        let ia = std::rc::Rc::new(InputArg::from_type(tp, index));
        b.bind_inputarg(&ia);
        (b, ia)
    }

    #[test]
    fn box_ref_identity_is_pointer_equality() {
        let a = BoxRef::new_resop(Type::Int, 0);
        let cloned = a.clone();
        let other = BoxRef::new_resop(Type::Int, 1);
        assert_eq!(a, cloned);
        assert_ne!(a, other);
    }

    /// Goal D identity stabilization: `from_bound_op` memoizes the box
    /// wrapper on the op (`Op::box_cache`), so two resolutions of the SAME
    /// `Rc<Op>` yield the SAME `Rc<Box>` (`Rc::ptr_eq` == `self is other`),
    /// while a distinct (cloned) op gets its own. The cached box's position
    /// is refreshed from the (mutable) `op.pos` on every call.
    #[test]
    fn from_bound_op_memoizes_identity_per_op() {
        let op = std::rc::Rc::new(Op::new(OpCode::SameAsI, &[]));
        op.pos.set(OpRef::op_typed(3, Type::Int));

        let a = BoxRef::from_bound_op(&op);
        let b = BoxRef::from_bound_op(&op);
        // Same op -> same wrapper identity (pointer equality).
        assert_eq!(a.as_ptr(), b.as_ptr());
        assert_eq!(a, b);

        // A fresh-identity clone (box_cache reset to None) gets its own wrapper.
        let op2 = std::rc::Rc::new((*op).clone());
        let c = BoxRef::from_bound_op(&op2);
        assert_ne!(a.as_ptr(), c.as_ptr());
        assert_ne!(a, c);

        // Position refresh: mutating op.pos is reflected on the next resolve,
        // and through the already-held reference (shared Rc<Box>).
        op.pos.set(OpRef::op_typed(9, Type::Int));
        let d = BoxRef::from_bound_op(&op);
        assert_eq!(a.as_ptr(), d.as_ptr());
        assert_eq!(d.position(), Some(9));
        assert_eq!(a.position(), Some(9));
    }

    /// `from_bound_inputarg` memoizes identically on `InputArg::box_cache`
    /// (no position refresh — `InputArg.index` is immutable).
    #[test]
    fn from_bound_inputarg_memoizes_identity_per_inputarg() {
        let ia = std::rc::Rc::new(InputArg::from_type(Type::Ref, 2));
        let a = BoxRef::from_bound_inputarg(&ia);
        let b = BoxRef::from_bound_inputarg(&ia);
        assert_eq!(a.as_ptr(), b.as_ptr());
        assert_eq!(a, b);

        // A distinct InputArg (even with equal tp/index) gets its own wrapper.
        let ia2 = std::rc::Rc::new(InputArg::from_type(Type::Ref, 2));
        let c = BoxRef::from_bound_inputarg(&ia2);
        assert_ne!(a.as_ptr(), c.as_ptr());
        assert_ne!(a, c);
    }

    /// After identity memoization (slice 1), `BoxRef::same_box` is object
    /// identity (`Rc::ptr_eq`) for ResOp / InputArg plus `same_constant`
    /// (value comparison) for Const — resoperation.py:38 / history.py:211.
    /// Two memoized resolutions of the same op/inputarg are `same_box`;
    /// DISTINCT op/inputarg identities are NOT (even at the same position —
    /// distinct objects are distinct boxes); equal-valued Const boxes are
    /// `same_box` by value; Const vs non-Const is not.
    #[test]
    fn same_box_identity_and_same_constant() {
        let op = std::rc::Rc::new(Op::new(OpCode::SameAsI, &[]));
        op.pos.set(OpRef::op_typed(4, Type::Int));
        let a = BoxRef::from_bound_op(&op);
        let b = BoxRef::from_bound_op(&op);
        assert!(a.same_box(&b)); // same op -> shared Rc

        // A distinct op even at the SAME position/type is NOT same_box.
        let op2 = std::rc::Rc::new((*op).clone());
        let c = BoxRef::from_bound_op(&op2);
        assert!(!a.same_box(&c));

        let ia = std::rc::Rc::new(InputArg::from_type(Type::Ref, 1));
        let ja = BoxRef::from_bound_inputarg(&ia);
        let jb = BoxRef::from_bound_inputarg(&ia);
        assert!(ja.same_box(&jb)); // same inputarg -> shared Rc
        let ia2 = std::rc::Rc::new(InputArg::from_type(Type::Ref, 1));
        assert!(!ja.same_box(&BoxRef::from_bound_inputarg(&ia2)));

        let k7a = BoxRef::new_const(Value::Int(7));
        let k7b = BoxRef::new_const(Value::Int(7));
        assert!(k7a.same_box(&k7b)); // same_constant by value
        assert!(!k7a.same_box(&BoxRef::new_const(Value::Int(8))));
        assert!(!k7a.same_box(&a)); // Const vs ResOp
    }

    #[test]
    fn forwarded_chain_walk_returns_terminal() {
        let (a, _ao) = bound_resop(Type::Int, 0);
        let (b, _bo) = bound_resop(Type::Int, 1);
        let (c, _co) = bound_resop(Type::Int, 2);
        a.set_forwarded_box(b.clone());
        b.set_forwarded_box(c.clone());
        assert_eq!(a.get_box_replacement(false), c);
        assert_eq!(b.get_box_replacement(false), c);
        assert_eq!(c.get_box_replacement(false), c);
    }

    #[test]
    fn forwarded_chain_stops_at_info() {
        let (a, _ao) = bound_resop(Type::Int, 0);
        let (b, _bo) = bound_resop(Type::Int, 1);
        a.set_forwarded_box(b.clone());
        b.set_forwarded_info(OpInfo::Unknown);
        assert_eq!(a.get_box_replacement(false), b);
    }

    #[test]
    fn forwarded_chain_not_const_stops_before_const() {
        let (a, _ao) = bound_resop(Type::Int, 0);
        let (b, _bo) = bound_resop(Type::Int, 1);
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
        let (a, _ao) = bound_resop(Type::Int, 0);
        let (b, _bo) = bound_resop(Type::Int, 0);
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
        let (a, _ao) = bound_resop(Type::Ref, 0);
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
        let (a, _ao) = bound_resop(Type::Ref, 0);
        let (b, _bo) = bound_resop(Type::Ref, 0);
        a.set_forwarded_box(b.clone());
        assert!(a.ptr_info().is_none());
    }

    #[test]
    fn ptr_info_returns_none_for_intbound_forwarded() {
        let (a, _ao) = bound_resop(Type::Int, 0);
        a.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(7)));
        assert!(a.ptr_info().is_none());
    }

    #[test]
    fn int_bound_returns_inner_when_forwarded_is_intbound() {
        let (a, _ao) = bound_resop(Type::Int, 0);
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
        let (a, _ao) = bound_resop(Type::Int, 0);
        let (b, _bo) = bound_resop(Type::Int, 0);
        a.set_forwarded_box(b.clone());
        assert!(a.int_bound().is_none());
    }

    #[test]
    fn int_bound_returns_none_for_ptrinfo_forwarded() {
        let (a, _ao) = bound_resop(Type::Ref, 0);
        a.set_forwarded_info(OpInfo::ptr(PtrInfo::nonnull()));
        assert!(a.int_bound().is_none());
    }

    #[test]
    fn ptr_info_mut_mutates_inner_in_place() {
        let (a, _ao) = bound_resop(Type::Ref, 0);
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
        let (a, _ao) = bound_resop(Type::Int, 0);
        a.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(7)));
        assert!(a.ptr_info_mut().is_none());
    }

    #[test]
    fn int_bound_mut_mutates_inner_in_place() {
        let (a, _ao) = bound_resop(Type::Int, 0);
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
        let (a, _ao) = bound_resop(Type::Ref, 0);
        a.set_forwarded_info(OpInfo::ptr(PtrInfo::nonnull()));
        assert!(a.int_bound_mut().is_none());
    }

    /// After `bind_op`, `set_forwarded_*` writes through to
    /// `op.forwarded` (the single canonical host), so a reader on
    /// `op.forwarded` sees the same state as `box.get_forwarded()`.
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
    /// single-canonical-host invariant established for ResOp.
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
}
