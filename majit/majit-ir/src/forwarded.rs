//! `_forwarded` slot for op / inputarg objects.
//!
//! Direct port of `rpython/jit/metainterp/resoperation.py:233-242`
//! (`AbstractResOpOrInputArg._forwarded`). The slot holds `None`, another
//! `AbstractResOpOrInputArg`, a `Const`, or an `AbstractInfo`.
//!
//! ### Where the trait lives
//!
//! In PyPy `AbstractInfo` is defined at
//! `rpython/jit/metainterp/optimizeopt/info.py:17` and inherits from
//! `AbstractValue` (`resoperation.py:29`). Pyre keeps `Op` /
//! `InputArg` in `majit-ir`; for the slot to live inline on those
//! structs the marker trait has to be visible from `majit-ir` too,
//! so we declare it here and let `OpInfo` in
//! `majit-metainterp/src/optimizeopt/info.rs` `impl AbstractInfo for OpInfo`.
//! The trait location divergence is bookkeeping only — the runtime
//! semantics match upstream.

use std::any::Any;
use std::rc::Rc;

use crate::resoperation::OpRef;

/// `resoperation.py:29` `AbstractValue` — base of the hierarchy whose
/// instances can be stored in `_forwarded`.
///
/// Used only as a polymorphic carrier for the `Forwarded::Info` variant
/// today; the trait may be widened in later slices when `Op` /
/// `InputArg` / `Const` start participating in the same dyn-trait
/// container.
pub trait AbstractValue: std::fmt::Debug + Any {
    fn as_any(&self) -> &dyn Any;

    /// `resoperation.py:31` `is_info_class = False`. Overridden by
    /// `AbstractInfo` subclasses.
    fn is_info_class(&self) -> bool {
        false
    }

    /// `resoperation.py:47` `is_constant`. Overridden by `Const`.
    fn is_constant(&self) -> bool {
        false
    }
}

/// `optimizeopt/info.py:17` `AbstractInfo(AbstractValue)`.
///
/// Marker trait implemented by every analysis-info type (`IntBound`,
/// `PtrInfo`, `FloatConstInfo`, virtual info subclasses, …). The trait
/// adds no required methods because `AbstractInfo` upstream is itself
/// just a tagged base (`_attrs_ = ()`, `is_info_class = True`).
pub trait AbstractInfo: AbstractValue {}

/// `resoperation.py:235` `_forwarded` slot.
///
/// ```text
/// _forwarded = None # either another resop or OptInfo
/// ```
///
/// Pyre projects the polymorphic Python slot into a typed enum:
///
/// - [`Forwarded::None`] — initial state.
/// - [`Forwarded::OpRef`] — forward to another `AbstractResOpOrInputArg`.
///   The forwarded box is identified by its `OpRef` position; a later
///   slice will retype this to hold an `Rc<Op>` / `Rc<InputArg>`
///   directly once `Vec<Op>` storage moves to `Vec<Rc<Op>>`.
/// - [`Forwarded::Const`] — forward to a `Const` (`optimizer.py:413`
///   `make_constant`).
/// - [`Forwarded::Info`] — attach analysis info (`info.py:17`).
#[derive(Clone, Debug, Default)]
pub enum Forwarded {
    /// `_forwarded = None`.
    #[default]
    None,
    /// `_forwarded = another AbstractResOpOrInputArg`.
    ///
    /// Slice 1A carries the target by `OpRef`; slice 1B will replace
    /// this with `Op(Rc<Op>)` / `InputArg(Rc<InputArg>)` once trace
    /// storage moves to `Vec<Rc<…>>`.
    OpRef(OpRef),
    /// `_forwarded = constbox` (`optimizer.py:413 make_constant`).
    ///
    /// PyPy stores a `Const` instance whose object identity IS the box.
    /// Pyre encodes the same identity as a typed `OpRef::Const*` variant
    /// from the constant pool — `OpRef::ConstInt(idx)` / `ConstFloat(idx)`
    /// / `ConstPtr(idx)` (history.py:220/261/307). Storing the typed
    /// OpRef here lets `get_box_replacement` advance into the Const and
    /// return the const-OpRef just as RPython returns the Const box.
    Const(OpRef),
    /// `_forwarded = AbstractInfo` (`info.py:17`).
    Info(Rc<dyn AbstractInfo>),
}

impl Forwarded {
    /// True when the slot holds `None`.
    pub fn is_none(&self) -> bool {
        matches!(self, Forwarded::None)
    }

    /// True for `Forwarded::Info(_)` — corresponds to
    /// `next_op.is_info_class` (`resoperation.py:64`).
    pub fn is_info(&self) -> bool {
        matches!(self, Forwarded::Info(_))
    }

    /// True for `Forwarded::Const(_)` — corresponds to
    /// `next_op.is_constant()` on the Const branch
    /// (`resoperation.py:65`).
    pub fn is_constant(&self) -> bool {
        matches!(self, Forwarded::Const(_))
    }
}

/// `resoperation.py:233-242 AbstractResOpOrInputArg` reader/writer
/// surface, expressed as a trait so `Op` and `InputArg` share the
/// implementation.
pub trait AbstractResOpOrInputArg {
    /// Underlying `RefCell<Forwarded>` slot. The methods below operate
    /// through this slot; implementors only need to return it.
    fn forwarded_slot(&self) -> &std::cell::RefCell<Forwarded>;

    /// `resoperation.py:237` `get_forwarded`.
    fn get_forwarded(&self) -> std::cell::Ref<'_, Forwarded> {
        self.forwarded_slot().borrow()
    }

    /// `resoperation.py:240` `set_forwarded(forwarded_to)`. Replaces
    /// the slot wholesale.
    fn set_forwarded(&self, forwarded_to: Forwarded) {
        *self.forwarded_slot().borrow_mut() = forwarded_to;
    }

    /// `resoperation.py:240` `set_forwarded` Box-target form.
    /// Convenience for `set_forwarded(Forwarded::OpRef(target))`.
    fn set_forwarded_opref(&self, target: OpRef) {
        self.set_forwarded(Forwarded::OpRef(target));
    }

    /// `optimizer.py:413` `make_constant` — replace with constbox.
    /// Takes the typed `OpRef::Const*` (history.py:220/261/307) that
    /// identifies the pooled constant; RPython equivalent uses the
    /// `Const` box object itself as the identity.
    fn set_forwarded_const(&self, const_opref: OpRef) {
        debug_assert!(
            const_opref.is_constant(),
            "set_forwarded_const requires a typed OpRef::Const* variant, got {const_opref:?}",
        );
        self.set_forwarded(Forwarded::Const(const_opref));
    }

    /// `optimizer.py:393` setting an `AbstractInfo` instance.
    fn set_forwarded_info(&self, info: Rc<dyn AbstractInfo>) {
        self.set_forwarded(Forwarded::Info(info));
    }
}

impl AbstractResOpOrInputArg for crate::value::InputArg {
    fn forwarded_slot(&self) -> &std::cell::RefCell<Forwarded> {
        &self.forwarded
    }
}

impl AbstractResOpOrInputArg for crate::resoperation::Op {
    fn forwarded_slot(&self) -> &std::cell::RefCell<Forwarded> {
        &self.forwarded
    }
}

/// `resoperation.py:57-68` `AbstractValue.get_box_replacement(op,
/// not_const=False)` — chain-walk through `_forwarded`.
///
/// ```text
/// while isinstance(op, AbstractResOpOrInputArg):
///     next_op = op._forwarded
///     if (next_op is None or next_op.is_info_class or
///         (not_const and next_op.is_constant())):
///         return op
///     op = next_op
/// return op
/// ```
///
/// `trace_ops` / `inputargs` resolve `Forwarded::OpRef(target)` to the
/// concrete next `Op` / `InputArg`. `num_inputargs` is the index split
/// (inputargs occupy the first `num_inputargs` positions; ops occupy
/// `num_inputargs..`). Returns the terminal `OpRef` (which may equal
/// the input if the chain is empty or stops immediately).
pub fn get_box_replacement(
    start: OpRef,
    trace_ops: &[crate::resoperation::Op],
    inputargs: &[crate::value::InputArg],
    num_inputargs: u32,
    not_const: bool,
) -> OpRef {
    let mut cur = start;
    loop {
        // `Const*` variants of OpRef and `OpRef::NONE` are not
        // `AbstractResOpOrInputArg` in PyPy — the while-loop exits
        // immediately for them.
        let Some(slot) = forwarded_slot_at(cur, trace_ops, inputargs, num_inputargs) else {
            return cur;
        };
        // RPython `resoperation.py:58-64 get_box_replacement`: stop when
        // `_forwarded` is None, an Info, or (when `not_const=True`) a
        // constant; otherwise advance to `_forwarded`.  A Const advance
        // exits on the next iteration because `forwarded_slot_at` returns
        // None for Const variants (Const isn't `AbstractResOpOrInputArg`).
        let step = match &*slot.borrow() {
            Forwarded::None => StepOut::Stop,
            Forwarded::Info(_) => StepOut::Stop,
            Forwarded::Const(_) if not_const => StepOut::Stop,
            Forwarded::Const(target) => StepOut::Advance(*target),
            Forwarded::OpRef(target) => StepOut::Advance(*target),
        };
        match step {
            StepOut::Stop => return cur,
            StepOut::Advance(next) => cur = next,
        }
    }
}

enum StepOut {
    Stop,
    Advance(OpRef),
}

/// Look up the `_forwarded` slot for an OpRef. Returns None for
/// non-AbstractResOpOrInputArg refs (constants, NONE, out-of-bounds).
fn forwarded_slot_at<'a>(
    opref: OpRef,
    trace_ops: &'a [crate::resoperation::Op],
    inputargs: &'a [crate::value::InputArg],
    num_inputargs: u32,
) -> Option<&'a std::cell::RefCell<Forwarded>> {
    use crate::resoperation::OpRef as O;
    match opref {
        O::ConstInt(_) | O::ConstFloat(_) | O::ConstPtr(_) => None,
        O::None => None,
        O::TempVar(_) => None,
        O::InputArgInt(i) | O::InputArgFloat(i) | O::InputArgRef(i) => {
            inputargs.get(i as usize).map(|ia| ia.forwarded_slot())
        }
        O::IntOp(p) | O::FloatOp(p) | O::RefOp(p) | O::VoidOp(p) => {
            let idx = p.checked_sub(num_inputargs)?;
            trace_ops.get(idx as usize).map(|op| op.forwarded_slot())
        }
    }
}
