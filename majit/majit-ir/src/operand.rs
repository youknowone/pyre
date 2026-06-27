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

use crate::box_ref::{
    BoxRef, Forwarded, ForwardingHost, IntBoundBorrow, IntBoundBorrowMut, PtrInfoBorrow,
    PtrInfoBorrowMut,
};
use crate::intbound::IntBound;
use crate::op_info::OpInfo;
use crate::ptr_info::PtrInfo;
use crate::resoperation::{OpRc, OpRef};
use crate::value::{Const, GcRef, InputArgRc, Type, Value};
use std::cell::{Cell, RefCell};
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
    /// `ConstPtr`). The value lives in an `Rc<Cell<Value>>`: the `Cell` lets
    /// the GC root walker forward an inline `ConstPtr` `GcRef` in place
    /// through a shared `&self` borrow of `Op.args` (`walk_const_ptr_refs`
    /// get/visit/set cycle), and the `Rc` gives the const an object identity
    /// — `==` is `Rc::ptr_eq` (resoperation.py:29-39 `AbstractValue` keys by
    /// `is`), so two distinct `const_` mints compare unequal while a clone
    /// shares the same const object (`getarglist_copy` reuses the same
    /// `Const`). Value equality is the opt-in `same_constant` (history.py:211),
    /// surfaced as [`same_box`](Self::same_box). This is the same shared-cell
    /// in-place-forward contract the const-kind `BoxKind::Const { value:
    /// Cell<Value> }` `Rc<Box>` carrier provided. The forwarding visitor is
    /// idempotent on an already-forwarded object (collector.rs:1133), so a
    /// const cell reachable from two slots forwards safely.
    Const(Rc<Cell<Value>>),
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
        Operand::Const(Rc::new(Cell::new(value.to_value())))
    }

    /// A constant operand straight from a [`Value`] — the successor to
    /// `Operand::from_boxref(&BoxRef::new_const(value))`, minting the same
    /// fresh `Rc<Cell<Value>>` const identity without the BoxRef round-trip.
    pub fn const_from_value(value: Value) -> Operand {
        Operand::Const(Rc::new(Cell::new(value)))
    }

    /// The absent-slot sentinel.
    pub fn none() -> Operand {
        Operand::None
    }

    /// Build an operand from a flat `OpRef`, for the producer-resolution sites
    /// that pick between a bound producer and the absent/const cases off a
    /// position ref. `None` and the three `Const*` variants carry their value
    /// inline (mirror of [`BoxRef::from_opref`]'s non-position arms); a
    /// position-only ref (a `*Op` / `InputArg*` with no producer `Rc`) has no
    /// `Operand` representation under the #9 union and panics, the same
    /// invariant tripwire as [`Operand::from_boxref`]. Callers route bound
    /// positions through [`from_bound_op`](Self::from_bound_op) /
    /// [`from_bound_inputarg`](Self::from_bound_inputarg) and reach here only on
    /// `None` / `Const`.
    pub fn from_opref(r: OpRef) -> Operand {
        match r {
            OpRef::None => Operand::None,
            OpRef::ConstInt(v) => Operand::Const(Rc::new(Cell::new(Value::Int(v)))),
            OpRef::ConstFloat(v) => Operand::Const(Rc::new(Cell::new(Value::Float(v)))),
            OpRef::ConstPtr(v) => Operand::Const(Rc::new(Cell::new(Value::Ref(v)))),
            _ => panic!(
                "from_opref: position-only ref {r:?} has no producer to bind — \
                 every operand source must carry a bound producer or a const (#9)"
            ),
        }
    }

    /// Bind `r` to a producer-carrying operand whose [`to_opref`](Self::to_opref)
    /// equals `r` — the binding sibling of [`from_opref`](Self::from_opref). A
    /// `None`/`Const` ref sheds inline (identical to `from_opref`); a ResOp /
    /// InputArg position — which `from_opref` cannot represent and panics on —
    /// binds to a freshly-minted synthetic producer (`SameAs*` / `InputArg`)
    /// carrying the same `pos`. The returned `Operand::Op` / `Operand::InputArg`
    /// holds a strong `Rc`, so the synthetic producer stays alive for exactly as
    /// long as the operand is stored — no external root table (unlike the
    /// `Weak`-backed [`bound_box_from_opref`](crate::box_ref::bound_box_from_opref)).
    /// The vector optimizer's guard-strengthening / accumulation stitching uses
    /// this where its producer buffers hold `Op` values (not `OpRc`), so no real
    /// producer `Rc` is reachable to bind — `guard.py` emits fresh boxes,
    /// `renamer.py` carries box objects.
    pub fn bound_from_opref(r: OpRef) -> Operand {
        use crate::resoperation::Op;
        use crate::resoperation::OpCode;
        use crate::value::InputArg;
        if r.is_none() || r.is_constant() {
            return Operand::from_opref(r);
        }
        let ty = r.ty().unwrap_or(Type::Void);
        match r {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                let ia: InputArgRc = Rc::new(InputArg::from_type(ty, r.raw()));
                Operand::from_bound_inputarg(&ia)
            }
            _ => {
                let opcode = match ty {
                    Type::Int => OpCode::SameAsI,
                    Type::Float => OpCode::SameAsF,
                    Type::Ref => OpCode::SameAsR,
                    Type::Void => OpCode::Jump,
                };
                let op: OpRc = Rc::new(Op::new(opcode, &[]));
                op.pos.set(r);
                Operand::from_bound_op(&op)
            }
        }
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
            // Re-encodes from the live `Cell` value, so a GC-moved `ConstPtr`
            // reads back at its post-move address (box_ref.rs:510-514 parity).
            Operand::Const(cell) => match cell.get() {
                Value::Int(v) => OpRef::const_int(v),
                Value::Float(v) => OpRef::const_float(v),
                Value::Ref(v) => OpRef::const_ptr(v),
                Value::Void => OpRef::NONE,
            },
        }
    }

    /// `resoperation.py:233 _pos` accessor: the pool index for `Op` /
    /// `InputArg`; `Const` / `None` have no canonical position.
    pub fn position(&self) -> Option<u32> {
        match self {
            Operand::Op(op) => Some(op.pos.get().raw()),
            Operand::InputArg(ia) => Some(ia.index),
            Operand::Const(_) | Operand::None => None,
        }
    }

    /// The operand's `Type` (`Int` / `Float` / `Ref` / `Void`).
    pub fn type_(&self) -> Type {
        match self {
            Operand::Op(op) => op.pos.get().ty().unwrap_or(Type::Void),
            Operand::InputArg(ia) => ia.tp,
            Operand::Const(cell) => cell.get().get_type(),
            Operand::None => Type::Void,
        }
    }

    /// The inline constant value (`history.py:233` `Const.getint` family),
    /// `None` for non-`Const`.
    pub fn const_value(&self) -> Option<Value> {
        match self {
            Operand::Const(cell) => Some(cell.get()),
            _ => None,
        }
    }

    /// `history.py:803 IntFrontendOp(pos, intval)` parity — read the
    /// concrete intrinsic value off this operand. `Const` reads its inline
    /// cell; a bound `Op` / `InputArg` reads the producer's value carrier
    /// (`resoperation.py:566 IntOp._resint`); `None` carries no value.
    /// Mirror of [`BoxRef::get_value`].
    pub fn get_value(&self) -> Option<Value> {
        match self {
            Operand::Const(cell) => Some(cell.get()),
            Operand::Op(op) => op.get_value(),
            Operand::InputArg(ia) => ia.get_value(),
            Operand::None => None,
        }
    }

    /// Raw `ConstInt` value with no `IntBound` synthesis (`box_ref.rs:480`
    /// parity).
    pub fn const_int(&self) -> Option<i64> {
        match self {
            Operand::Const(cell) => match cell.get() {
                Value::Int(v) => Some(v),
                _ => None,
            },
            _ => None,
        }
    }

    /// `resoperation.py:47 is_constant`.
    pub fn is_constant(&self) -> bool {
        matches!(self, Operand::Const(_))
    }

    pub fn is_inputarg(&self) -> bool {
        matches!(self, Operand::InputArg(_))
    }

    pub fn is_resop(&self) -> bool {
        matches!(self, Operand::Op(_))
    }

    /// True for the absent-slot sentinel — the mirror of `OpRef::is_none`.
    pub fn is_none(&self) -> bool {
        matches!(self, Operand::None)
    }

    /// `resoperation.py:38 AbstractValue.same_box`: pointer identity
    /// (`Rc::ptr_eq`) for `Op` / `InputArg`, value comparison for `Const`
    /// (`history.py:211 Const.same_box` delegates to `same_constant`), and the
    /// `None` sentinel matches only itself. Native dispatch on the operand
    /// union: two operands carrying the same producer `Rc` are `ptr_eq`; two
    /// `Const` operands compare by value (`Value`'s `==` is bit-exact, so
    /// `0.0 != -0.0` and `NaN == NaN` — `history.py:251/292/338`); cross-kind
    /// is never the same box. Unlike `==` (uniform `Rc::ptr_eq`, so two equal
    /// fresh `Const`s differ), `same_box` is the value-aware predicate callers
    /// opt into exactly where RPython spells out `same_box(...)`. Equivalent to
    /// the former [`BoxRef::same_box`] round-trip (`from_bound_*` memoizes one
    /// wrapper per producer, so its `Rc::ptr_eq` short-circuit and this
    /// producer-`Rc` `ptr_eq` agree), without re-minting a `Const` box.
    pub fn same_box(&self, other: &Operand) -> bool {
        match (self, other) {
            (Operand::Op(a), Operand::Op(b)) => Rc::ptr_eq(a, b),
            (Operand::InputArg(a), Operand::InputArg(b)) => Rc::ptr_eq(a, b),
            (Operand::Const(a), Operand::Const(b)) => a.get() == b.get(),
            (Operand::None, Operand::None) => true,
            _ => false,
        }
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
            // Re-mint a const box from the live cell value (migration-window
            // bridge). Const boxes are fresh-per-resolution and never
            // `ptr_eq`-deduped, so a new identity is faithful.
            Operand::Const(cell) => BoxRef::new_const(cell.get()),
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
    /// A Const box lowers to `Operand::Const`, whose value is read out into a
    /// fresh `Rc<Cell<Value>>` (the `Cell`-backed in-place GC walk is
    /// preserved; the fresh `Rc` is a new const identity, since the source
    /// `BoxRef` and the operand carrier are distinct `Rc` types). A
    /// position-only box (no
    /// bound handle — `BoxRef::from_opref` of a non-const ResOp/InputArg
    /// position) has no `Operand` to lower to and is a contract violation: by
    /// #9 every operand source binds its producer (`from_bound_op` /
    /// `from_bound_inputarg`) or carries a const. The drain to zero was proven
    /// across the lib corpus and the bench suite (`MAJIT_DIAG_OPERAND_BOX`),
    /// so this case panics rather than fabricating an untracked operand.
    pub fn from_boxref(b: &BoxRef) -> Operand {
        if b.is_none() {
            return Operand::None;
        }
        // Shed a genuinely-bound box to its live-tracking producer `Rc`: the
        // operand IS the producer (resoperation.py `N_aryOp._args` holds the
        // `AbstractResOp`/`AbstractInputArg` directly), so its position
        // auto-tracks the producer's `op.pos` and its forwarding resolves
        // through the canonical `Op`/`InputArg`. The strong `Rc` keeps the
        // producer alive (acyclic on the SSA use-before-def DAG).
        if let Some(op) = b.bound_op() {
            return Operand::Op(op);
        }
        if let Some(ia) = b.bound_inputarg() {
            return Operand::InputArg(ia);
        }
        // A Const box lowers to the terminal `Operand::Const`, reading its
        // value into a fresh `Rc<Cell<Value>>` (the inline-`ConstPtr` GC walk
        // is preserved; the fresh `Rc` is a new const identity).
        if b.is_constant() {
            return Operand::Const(Rc::new(Cell::new(
                b.const_value()
                    .expect("is_constant box carries a const value"),
            )));
        }
        // A position-only box (no producer Rc, non-const) reaches here only if
        // some operand source skipped binding its producer — an invariant
        // violation under the #9 operand-union model where `_args[i]` is
        // always a producer or a constant. Bind it at its producer
        // (`from_bound_op` / `from_bound_inputarg`) instead of routing an
        // unbound position through here.
        panic!(
            "from_boxref: position-only box {:?} has no producer to bind — \
             every operand source must carry a bound producer or a const (#9)",
            b.to_opref()
        )
    }

    /// `resoperation.py:58-70 get_box_replacement(not_const=False)`.
    ///
    /// Walk the `_forwarded` chain from this operand, returning the operand
    /// one step before the chain hits `None`, an `Info` instance, or (when
    /// `not_const`) a constant. Only `Op` / `InputArg` carry a `_forwarded`
    /// slot (`AbstractResOpOrInputArg`); `Const` / `None` are terminal
    /// (`resoperation.py:62 while isinstance(op, AbstractResOpOrInputArg)`).
    /// This is the canonical walker; [`BoxRef::get_box_replacement`]
    /// delegates here.
    pub fn get_box_replacement(&self, not_const: bool) -> Operand {
        let mut cur = self.clone();
        loop {
            // Only a bound producer has a forwarded slot to read.
            let forwarded = match &cur {
                Operand::Op(op) => op.get_forwarded(),
                Operand::InputArg(ia) => ia.get_forwarded(),
                Operand::Const(_) | Operand::None => return cur,
            };
            match forwarded {
                Forwarded::None | Forwarded::Info(_) => return cur,
                Forwarded::Op(weak) => {
                    let Some(op_rc) = weak.upgrade() else {
                        // Dropped target: terminate at `cur` (PyPy keeps
                        // targets alive through the `operations` list).
                        return cur;
                    };
                    cur = Operand::Op(op_rc);
                }
                Forwarded::InputArg(weak) => {
                    let Some(ia_rc) = weak.upgrade() else {
                        return cur;
                    };
                    cur = Operand::InputArg(ia_rc);
                }
                Forwarded::Const(c) => {
                    if not_const {
                        return cur;
                    }
                    // Materialize a terminal const operand so callers can
                    // read `.const_value()` from the walker output.
                    return Operand::const_(c);
                }
            }
        }
    }

    /// The bound producer `Op` (`Operand::Op` arm), or `None` for
    /// `InputArg` / `Const` / `None`. The operand IS the producer `Rc` — no
    /// `Weak` upgrade and no `box_cache` (successor to [`BoxRef::bound_op`]).
    pub fn bound_op(&self) -> Option<OpRc> {
        match self {
            Operand::Op(op) => Some(Rc::clone(op)),
            _ => None,
        }
    }

    /// The bound `InputArg` (`Operand::InputArg` arm); `None` otherwise.
    /// Successor to [`BoxRef::bound_inputarg`].
    pub fn bound_inputarg(&self) -> Option<InputArgRc> {
        match self {
            Operand::InputArg(ia) => Some(Rc::clone(ia)),
            _ => None,
        }
    }

    /// Route a forwarding read to the carried `_forwarded` host
    /// ([`ForwardingHost`]): the bound `Op` / `InputArg`. `Const` / `None`
    /// have no `_forwarded` slot and take the default (mirror of
    /// `BoxRef::read_forwarding_host`).
    fn read_forwarding_host<R>(&self, default: R, f: impl FnOnce(&dyn ForwardingHost) -> R) -> R {
        match self {
            Operand::Op(op) => f(&**op),
            Operand::InputArg(ia) => f(&**ia),
            Operand::Const(_) | Operand::None => default,
        }
    }

    /// Route a forwarding write to the carried `_forwarded` host. `Const` is
    /// rejected by the caller's assert first; `None` has no slot and panics
    /// (mirror of `BoxRef::with_forwarding_host`).
    fn with_forwarding_host(&self, what: &str, f: impl FnOnce(&dyn ForwardingHost)) {
        match self {
            Operand::Op(op) => f(&**op),
            Operand::InputArg(ia) => f(&**ia),
            Operand::Const(_) | Operand::None => panic!(
                "Operand::{what} on a non-producer operand — only a bound \
                 Op/InputArg carries a _forwarded slot (box identity precondition)"
            ),
        }
    }

    /// `resoperation.py:237 get_forwarded`. Clone of the canonical
    /// `_forwarded` slot routed through the carried `Op` / `InputArg`; `Const`
    /// and `None` return `Forwarded::None`. Successor to
    /// [`BoxRef::get_forwarded`].
    pub fn get_forwarded(&self) -> Forwarded {
        self.read_forwarding_host(Forwarded::None, |h| h.get_forwarded())
    }

    /// `optimizer.py:394 op.set_forwarded(newop)` — `Op` target. Routes to
    /// [`ForwardingHost::set_forwarded_op`], which carries the
    /// `resoperation.py:241` self-cycle assert. Const has no `_forwarded`
    /// slot (`AbstractValue` invariant).
    pub fn set_forwarded_op(&self, target: &OpRc) {
        assert!(
            !self.is_constant(),
            "set_forwarded_op on Const violates the AbstractValue invariant \
             (Const has no _forwarded slot)"
        );
        self.with_forwarding_host("set_forwarded_op", |h| h.set_forwarded_op(target));
    }

    /// `optimizer.py:394 op.set_forwarded(newop)` — `InputArg` target
    /// (compile.py:478, unroll.py:497). Routes to
    /// [`ForwardingHost::set_forwarded_inputarg`].
    pub fn set_forwarded_inputarg(&self, target: &InputArgRc) {
        assert!(
            !self.is_constant(),
            "set_forwarded_inputarg on Const violates the AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        self.with_forwarding_host("set_forwarded_inputarg", |h| {
            h.set_forwarded_inputarg(target)
        });
    }

    /// `optimizer.py:432 make_constant(box, constbox)` — terminates the chain
    /// in a value-typed payload. Routes to
    /// [`ForwardingHost::set_forwarded_const`].
    pub fn set_forwarded_const(&self, value: Const) {
        assert!(
            !self.is_constant(),
            "set_forwarded_const on Const violates the AbstractValue \
             invariant (Const has no _forwarded slot)"
        );
        self.with_forwarding_host("set_forwarded_const", |h| h.set_forwarded_const(value));
    }

    /// `resoperation.py:53 set_forwarded(forwarded_to)` — `Info` target.
    /// Routes to [`ForwardingHost::set_forwarded_info`].
    pub fn set_forwarded_info(&self, info: OpInfo) {
        assert!(
            !self.is_constant(),
            "set_forwarded_info on Const violates the AbstractValue invariant \
             (Const has no _forwarded slot)"
        );
        self.with_forwarding_host("set_forwarded_info", |h| h.set_forwarded_info(info));
    }

    /// `_forwarded = None`. No-op on `Const` (no slot); routes to
    /// [`ForwardingHost::clear_forwarded`] on a bound producer.
    pub fn clear_forwarded(&self) {
        if self.is_constant() {
            return;
        }
        self.with_forwarding_host("clear_forwarded", |h| h.clear_forwarded());
    }

    /// `optimizer.py:99-113 getptrinfo` reader: the inner `PtrInfo` when
    /// `_forwarded` is `Info(OpInfo::Ptr(_))`, else `None`. Does not walk the
    /// chain (mirror of [`BoxRef::ptr_info`]).
    pub fn ptr_info(&self) -> Option<PtrInfoBorrow> {
        self.read_forwarding_host(None, |h| h.ptr_info())
    }

    /// Live `Rc<RefCell<PtrInfo>>` handle for identity-preserving callers
    /// (`Rc::ptr_eq`-based `same_info`). Mirror of [`BoxRef::ptr_info_handle`].
    pub fn ptr_info_handle(&self) -> Option<Rc<RefCell<PtrInfo>>> {
        self.read_forwarding_host(None, |h| h.ptr_info_handle())
    }

    /// Mutable `PtrInfo` guard for in-place mutation through the shared `Rc`.
    /// Mirror of [`BoxRef::ptr_info_mut`].
    pub fn ptr_info_mut(&self) -> Option<PtrInfoBorrowMut> {
        self.read_forwarding_host(None, |h| h.ptr_info_mut())
    }

    /// `optimizer.py:99-113 getintbound` reader: the inner `IntBound` when
    /// `_forwarded` is `Info(OpInfo::IntBound(_))`, else `None`. Mirror of
    /// [`BoxRef::int_bound`].
    pub fn int_bound(&self) -> Option<IntBoundBorrow> {
        self.read_forwarding_host(None, |h| h.int_bound())
    }

    /// Live `Rc<RefCell<IntBound>>` handle. Mirror of
    /// [`BoxRef::int_bound_handle`].
    pub fn int_bound_handle(&self) -> Option<Rc<RefCell<IntBound>>> {
        self.read_forwarding_host(None, |h| h.int_bound_handle())
    }

    /// Mutable `IntBound` guard for in-place mutation. Mirror of
    /// [`BoxRef::int_bound_mut`].
    pub fn int_bound_mut(&self) -> Option<IntBoundBorrowMut> {
        self.read_forwarding_host(None, |h| h.int_bound_mut())
    }

    /// True for the live-tracking producer variants (`Op` / `InputArg`),
    /// whose `to_opref()` reads the producer's CURRENT `op.pos`. The
    /// position-remap passes use this to skip operands that auto-track a
    /// renumbered producer (no snapshot rewrite needed); `Const` / `None`
    /// carry no position to remap.
    pub fn is_bound(&self) -> bool {
        matches!(self, Operand::Op(_) | Operand::InputArg(_))
    }

    /// GC walk over any inline `ConstPtr` reachable from this operand
    /// (`resoperation.py` `walk_const_ptr_refs`). A `Const` operand is held
    /// `Cell`-backed in its box, so its `GcRef` updates in place; pure `Op` /
    /// `InputArg` carry no inline const (their own `value` slot is walked at
    /// the producer).
    pub fn walk_const_ptr_refs(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        match self {
            // Forward an inline `ConstPtr` `GcRef` in place through the cell's
            // get/visit/set cycle (box_ref.rs:561-568 parity) — no `&mut self`
            // needed, so `Op.args` GC walks keep their shared `borrow()`.
            Operand::Const(cell) => {
                let mut v = cell.get();
                if let Value::Ref(gcref) = &mut v {
                    visitor(gcref);
                    cell.set(v);
                }
            }
            Operand::None | Operand::Op(_) | Operand::InputArg(_) => {}
        }
    }
}

impl PartialEq for Operand {
    /// Object identity, mirroring [`BoxRef`]'s pure `Rc::ptr_eq`
    /// (`box_ref.rs:1050`): `AbstractValue` defines no `__eq__`
    /// (`resoperation.py:29-39`), so every plain box-keyed dict keys by `is`.
    /// `Op` / `InputArg` / `Const` each carry an `Rc`, so `==` is `ptr_eq` on
    /// that producer/const handle; two `none()` sentinels match (Python's
    /// singleton `None`). Equal-valued constants minted separately are NOT
    /// equal here — value equality is the opt-in [`same_box`](Self::same_box)
    /// (`history.py:211`), never `==`, so a `same_box`-deduping table must
    /// build an explicit value-keyed map, not key on `Operand`.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Operand::None, Operand::None) => true,
            (Operand::Op(a), Operand::Op(b)) => Rc::ptr_eq(a, b),
            (Operand::InputArg(a), Operand::InputArg(b)) => Rc::ptr_eq(a, b),
            (Operand::Const(a), Operand::Const(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for Operand {}

impl std::hash::Hash for Operand {
    /// Identity hashing consistent with [`eq`](Self::eq) — the
    /// `compute_identity_hash` default (`resoperation.py:33-35`). A
    /// per-variant tag keeps cross-variant collisions from aliasing, and the
    /// `Rc` address is the identity for `Op` / `InputArg` / `Const`.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Operand::None => 0u8.hash(state),
            Operand::Op(op) => {
                1u8.hash(state);
                (Rc::as_ptr(op) as *const () as usize).hash(state);
            }
            Operand::InputArg(ia) => {
                2u8.hash(state);
                (Rc::as_ptr(ia) as *const () as usize).hash(state);
            }
            Operand::Const(cell) => {
                3u8.hash(state);
                (Rc::as_ptr(cell) as usize).hash(state);
            }
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

    /// Native same_box edge cases the round-trip version also met: the
    /// InputArg `Rc::ptr_eq` arm, the float bit-exact Const compare (hazard 3:
    /// `0.0 != -0.0`, `NaN == NaN`), and cross-kind always-false.
    #[test]
    fn same_box_inputarg_float_and_cross_kind() {
        let ia = Rc::new(InputArg::from_type(Type::Int, 0));
        assert!(Operand::from_bound_inputarg(&ia).same_box(&Operand::from_bound_inputarg(&ia)));
        let ia_other = Rc::new(InputArg::from_type(Type::Int, 0));
        assert!(
            !Operand::from_bound_inputarg(&ia).same_box(&Operand::from_bound_inputarg(&ia_other))
        );

        // Float Const compares bit-exact (Value::eq is to_bits-based).
        assert!(Operand::const_(Const::Float(1.5)).same_box(&Operand::const_(Const::Float(1.5))));
        assert!(!Operand::const_(Const::Float(0.0)).same_box(&Operand::const_(Const::Float(-0.0))));
        assert!(
            Operand::const_(Const::Float(f64::NAN))
                .same_box(&Operand::const_(Const::Float(f64::NAN)))
        );

        // Cross-kind is never the same box.
        let op = op_at(0, Type::Int);
        assert!(!Operand::from_bound_op(&op).same_box(&Operand::from_bound_inputarg(&ia)));
        assert!(!Operand::from_bound_op(&op).same_box(&Operand::const_(Const::Int(0))));
        assert!(!Operand::from_bound_inputarg(&ia).same_box(&Operand::none()));
    }

    /// `from_opref` builds the absent / inline-const arms natively (mirror of
    /// `BoxRef::from_opref`'s non-position cases); a position-only ref has no
    /// operand representation and panics (#9 invariant tripwire).
    #[test]
    fn from_opref_none_and_const_arms() {
        assert!(matches!(Operand::from_opref(OpRef::None), Operand::None));
        assert_eq!(
            Operand::from_opref(OpRef::ConstInt(7)).const_value(),
            Some(Value::Int(7))
        );
        assert_eq!(
            Operand::from_opref(OpRef::ConstFloat(1.5)).const_value(),
            Some(Value::Float(1.5))
        );
    }

    #[test]
    #[should_panic(expected = "position-only")]
    fn from_opref_position_only_panics() {
        let _ = Operand::from_opref(OpRef::IntOp(3));
    }

    #[test]
    fn to_boxref_preserves_identity_and_value() {
        let op = op_at(8, Type::Int);
        let via_operand = Operand::from_bound_op(&op).to_boxref();
        // The BoxRef bridge round-trips to a box equal (by bound-producer
        // identity) to a direct from_bound_op, though each is a distinct Rc.
        assert_eq!(via_operand, BoxRef::from_bound_op(&op));
        assert_eq!(via_operand.to_opref(), OpRef::op_typed(8, Type::Int));

        let c = Operand::const_(Const::Int(11)).to_boxref();
        assert_eq!(c.const_int(), Some(11));

        assert!(Operand::none().to_boxref().is_none());
    }

    #[test]
    fn from_boxref_sheds_bound_keeps_const_and_position_only() {
        // Bound op -> Operand::Op (live-tracking). is_resop stays true;
        // is_bound is true. to_boxref re-resolves through from_bound_op, so
        // the canonical box is equal (by bound producer) to the original.
        let op = op_at(8, Type::Int);
        let bound = BoxRef::from_bound_op(&op);
        let o = Operand::from_boxref(&bound);
        assert!(matches!(o, Operand::Op(_)));
        assert!(o.is_resop());
        assert!(o.is_bound());
        assert_eq!(o.to_boxref(), bound);

        // Bound input arg -> Operand::InputArg (live-tracking, bound).
        let ia = Rc::new(InputArg::from_type(Type::Ref, 2));
        let bia = BoxRef::from_bound_inputarg(&ia);
        let o = Operand::from_boxref(&bia);
        assert!(matches!(o, Operand::InputArg(_)));
        assert!(o.is_inputarg());
        assert!(o.is_bound());
        assert_eq!(o.to_boxref(), bia);

        // Const -> Operand::Const carrying the const VALUE in a fresh
        // Rc<Cell<Value>> (Cell-backed GC walk); NOT bound. to_boxref
        // re-mints, so it no longer ptr-aliases the source box.
        let cbox = BoxRef::new_const(Value::Int(11));
        let o = Operand::from_boxref(&cbox);
        assert!(matches!(o, Operand::Const(_)));
        assert!(o.is_constant());
        assert!(!o.is_bound());
        assert_eq!(o.const_int(), Some(11));
        assert_eq!(o.to_boxref().const_int(), Some(11));

        // None sentinel -> Operand::None.
        assert!(matches!(
            Operand::from_boxref(&BoxRef::none()),
            Operand::None
        ));
    }

    /// A position-only box (`from_opref`, no live producer, non-const) has no
    /// `Operand` to lower to under the #9 operand-union model — `from_boxref`
    /// panics rather than fabricating an untracked operand.
    #[test]
    #[should_panic(expected = "has no producer to bind")]
    fn from_boxref_panics_on_position_only_box() {
        let pos_only = BoxRef::from_opref(OpRef::op_typed(4, Type::Int));
        let _ = Operand::from_boxref(&pos_only);
    }

    /// `Eq` is object identity (`Rc::ptr_eq`), the `BoxRef`-key behaviour the
    /// re-keyed side tables depend on: same `Rc` is equal, a fresh mint is
    /// not — including for constants (value equality is `same_box`, never
    /// `==`). A clone shares the `Rc`, so it stays equal and `HashSet`-stable.
    #[test]
    fn eq_and_hash_are_object_identity() {
        use std::collections::HashSet;

        let op = op_at(0, Type::Int);
        // Same producer Rc -> equal; a clone shares the Rc -> equal.
        let a = Operand::from_bound_op(&op);
        assert_eq!(a, a.clone());
        assert_eq!(Operand::from_bound_op(&op), Operand::from_bound_op(&op));
        // Distinct ops at the same position -> distinct identity.
        let op_other = op_at(0, Type::Int);
        assert_ne!(
            Operand::from_bound_op(&op),
            Operand::from_bound_op(&op_other)
        );

        // Equal-valued constants minted separately are NOT `==` (distinct Rc),
        // even though they are `same_box`-equal.
        let c1 = Operand::const_(Const::Int(4));
        let c2 = Operand::const_(Const::Int(4));
        assert_ne!(c1, c2);
        assert!(c1.same_box(&c2));
        // A clone shares the const Rc -> equal.
        assert_eq!(c1, c1.clone());

        // None is a singleton; cross-variant never matches.
        assert_eq!(Operand::none(), Operand::none());
        assert_ne!(Operand::none(), Operand::const_(Const::Int(0)));

        // Hash agrees with Eq: a clone resolves the same bucket/membership.
        let mut set = HashSet::new();
        set.insert(Operand::from_bound_op(&op));
        assert!(set.contains(&Operand::from_bound_op(&op)));
        assert!(!set.contains(&Operand::from_bound_op(&op_other)));
        set.insert(c1.clone());
        assert!(set.contains(&c1));
        assert!(!set.contains(&c2));
    }

    /// Forwarding read/write/clear routes through the carried `Op` host,
    /// mirroring `BoxRef::{get,set,clear}_forwarded` without the wrapper.
    #[test]
    fn forwarding_get_set_clear_on_op() {
        let a = Operand::from_bound_op(&op_at(0, Type::Int));
        let b = op_at(1, Type::Int);
        assert!(matches!(a.get_forwarded(), Forwarded::None));
        a.set_forwarded_op(&b);
        match a.get_forwarded() {
            Forwarded::Op(w) => assert!(Rc::ptr_eq(&w.upgrade().unwrap(), &b)),
            other => panic!("expected Forwarded::Op, got {other:?}"),
        }
        // The walker follows a -> b to the terminal.
        match a.get_box_replacement(false) {
            Operand::Op(op) => assert!(Rc::ptr_eq(&op, &b)),
            other => panic!("expected Operand::Op(b), got {other:?}"),
        }
        a.clear_forwarded();
        assert!(matches!(a.get_forwarded(), Forwarded::None));
    }

    /// `bound_op` / `bound_inputarg` expose the carried producer `Rc` for the
    /// matching arm and `None` everywhere else.
    #[test]
    fn bound_op_and_bound_inputarg_arms() {
        let op = op_at(2, Type::Int);
        let o_op = Operand::from_bound_op(&op);
        assert!(o_op.bound_op().is_some_and(|o| Rc::ptr_eq(&o, &op)));
        assert!(o_op.bound_inputarg().is_none());

        let ia = Rc::new(InputArg::from_type(Type::Ref, 1));
        let o_ia = Operand::from_bound_inputarg(&ia);
        assert!(o_ia.bound_inputarg().is_some_and(|i| Rc::ptr_eq(&i, &ia)));
        assert!(o_ia.bound_op().is_none());

        let o_c = Operand::const_(Const::Int(3));
        assert!(o_c.bound_op().is_none() && o_c.bound_inputarg().is_none());
        assert!(Operand::none().bound_op().is_none());
    }

    /// `resoperation.py:241` self-cycle assert fires straight off the carried
    /// producer (the production-direct write path).
    #[test]
    #[should_panic(expected = "one-node chain cycle")]
    fn set_forwarded_op_to_self_panics() {
        let op = op_at(0, Type::Int);
        Operand::from_bound_op(&op).set_forwarded_op(&op);
    }

    /// Const has no `_forwarded` slot — a forwarding write is rejected before
    /// it can silently lose data (`AbstractValue` invariant).
    #[test]
    #[should_panic(expected = "AbstractValue invariant")]
    fn set_forwarded_on_const_panics() {
        let op = op_at(0, Type::Int);
        Operand::const_(Const::Int(0)).set_forwarded_op(&op);
    }

    /// The `None` sentinel carries no host, so a forwarding write panics
    /// rather than no-op away the write.
    #[test]
    #[should_panic(expected = "non-producer operand")]
    fn set_forwarded_on_none_panics() {
        let op = op_at(0, Type::Int);
        Operand::none().set_forwarded_op(&op);
    }

    /// `ptr_info` / `int_bound` read the inner `OpInfo` payload off the
    /// carried host; the `_mut` guard mutates it in place through the shared
    /// `Rc`. Const / unset operands read `None`.
    #[test]
    fn ptr_info_and_int_bound_readers() {
        use crate::intbound::IntBound;
        use crate::op_info::OpInfo;
        use crate::ptr_info::PtrInfo;

        let a = Operand::from_bound_op(&op_at(0, Type::Ref));
        assert!(a.ptr_info().is_none() && a.int_bound().is_none());
        a.set_forwarded_info(OpInfo::ptr(PtrInfo::nonnull()));
        assert!(a.ptr_info().expect("ptr_info Some").is_nonnull());
        assert!(a.int_bound().is_none());

        let b = Operand::from_bound_op(&op_at(1, Type::Int));
        b.set_forwarded_info(OpInfo::int_bound(IntBound::from_constant(42)));
        let ib = b.int_bound().expect("int_bound Some");
        assert!(ib.is_constant());
        assert_eq!(ib.get_constant_int(), 42);

        // Const has no _forwarded slot -> readers return None (no panic).
        assert!(Operand::const_(Const::Int(0)).ptr_info().is_none());
        assert!(Operand::none().int_bound().is_none());
    }
}
