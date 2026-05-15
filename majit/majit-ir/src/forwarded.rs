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
use crate::value::Const;

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
    /// PyPy stores a `Const` instance with its own object identity;
    /// pyre's `Const` is `Copy`, so the value is embedded directly.
    Const(Const),
    /// `_forwarded = AbstractInfo` (`info.py:17`).
    Info(Rc<dyn AbstractInfo>),
}

impl Forwarded {
    /// True when the slot holds `None`.
    pub fn is_none(&self) -> bool {
        matches!(self, Forwarded::None)
    }
}
