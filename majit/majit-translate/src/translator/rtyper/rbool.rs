//! RPython `rpython/rtyper/rbool.py` — `BoolRepr(IntegerRepr)` +
//! `bool_repr` singleton.
//!
//! ## Parity scope of this commit
//!
//! Like `rfloat.rs` / `rint.rs`, this commit stops at the Repr-shape
//! boundary: `BoolRepr` struct, `bool_repr()` singleton, and the
//! `SomeBool` arm in [`super::rmodel::rtyper_makerepr`]. `rtype_*`
//! methods + the pairtype conversion blocks (`BoolRepr ↔ FloatRepr`,
//! `BoolRepr ↔ IntegerRepr`) are reserved for Cascade 2.
//!
//! | upstream line | pyre mirror |
//! |---|---|
//! | `class BoolRepr(IntegerRepr)` `rbool.py:10-20` | [`BoolRepr`] |
//! | `BoolRepr.lowleveltype = Bool` `rbool.py:11` | `BoolRepr::lowleveltype` returns [`LowLevelType::Bool`] |
//! | `BoolRepr.as_int = signed_repr` `rbool.py:13-15` | [`BoolRepr::as_int`] |
//! | `BoolRepr.convert_const` `rbool.py:17-20` | [`BoolRepr::convert_const`] |
//! | `bool_repr = BoolRepr()` `rbool.py:36` | [`bool_repr`] singleton |
//! | `__extend__(annmodel.SomeBool)` `rbool.py:39-44` | [`super::rmodel::rtyper_makerepr`] `SomeBool` arm |
//! | `rtype_bool` / `rtype_int` / `rtype_float` `rbool.py:22-34` | deferred (Cascade 2 — need `hop.inputargs`) |
//! | `pairtype(BoolRepr, FloatRepr)` `rbool.py:49-54` | deferred (Cascade 2 — pairtype dispatch) |
//! | `pairtype(FloatRepr, BoolRepr)` `rbool.py:56-61` | deferred |
//! | `pairtype(BoolRepr, IntegerRepr)` `rbool.py:63-74` | deferred |
//! | `pairtype(IntegerRepr, BoolRepr)` `rbool.py:76-84` | deferred |

use std::sync::{Arc, OnceLock};

use crate::flowspace::model::{ConstValue, Constant};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rint::{IntegerRepr, signed_repr};
use crate::translator::rtyper::rmodel::{Repr, ReprState};

// ____________________________________________________________
// RPython `class BoolRepr(IntegerRepr)` (rbool.py:10-34).

/// RPython `class BoolRepr(IntegerRepr)` (`rbool.py:10-34`).
///
/// ```python
/// class BoolRepr(IntegerRepr):
///     lowleveltype = Bool
///     # NB. no 'opprefix' here.  Use 'as_int' systematically.
///     def __init__(self):
///         from rpython.rtyper.rint import signed_repr
///         self.as_int = signed_repr
///
///     def convert_const(self, value):
///         if not isinstance(value, bool):
///             raise TyperError("not a bool: %r" % (value,))
///         return value
///     ...
/// ```
///
/// Upstream inherits from `IntegerRepr`. Rust `BoolRepr` implements
/// [`Repr`] directly but routes `as_int` to [`signed_repr`] exactly
/// as upstream does — this lets downstream arithmetic rtype_* methods
/// dispatch through `r_bool.as_int` (an IntegerRepr Arc) when cascade
/// step 2 lands.
#[derive(Debug)]
pub struct BoolRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl BoolRepr {
    /// RPython `BoolRepr.__init__` (`rbool.py:13-15`). No opprefix —
    /// any arithmetic goes through `self.as_int = signed_repr`.
    pub fn new() -> Self {
        BoolRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Bool,
        }
    }

    /// RPython `BoolRepr.as_int = signed_repr` (`rbool.py:14-15`).
    ///
    /// Upstream stores a reference to the module-level `signed_repr`
    /// instance during `__init__`. Pyre returns the [`signed_repr`]
    /// singleton Arc so identity comparison (`is signed_repr`)
    /// survives as `Arc::ptr_eq`.
    pub fn as_int(&self) -> Arc<IntegerRepr> {
        signed_repr()
    }
}

impl Default for BoolRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for BoolRepr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "BoolRepr"
    }

    /// RPython `BoolRepr.convert_const` (`rbool.py:17-20`).
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if not isinstance(value, bool):
    ///         raise TyperError("not a bool: %r" % (value,))
    ///     return value
    /// ```
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        match value {
            ConstValue::Bool(_) => Ok(Constant::with_concretetype(
                value.clone(),
                self.lltype.clone(),
            )),
            other => Err(TyperError::message(format!("not a bool: {other:?}"))),
        }
    }
}

/// Singleton — `bool_repr = BoolRepr()` (`rbool.py:36`).
///
/// Upstream module-level instance referenced from `rtype_bool`
/// dispatch in rint.py / rfloat.py (as `rpython.rtyper.rbool.bool_repr`).
pub fn bool_repr() -> Arc<BoolRepr> {
    static REPR: OnceLock<Arc<BoolRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(BoolRepr::new())).clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::rmodel::Setupstate;

    #[test]
    fn bool_repr_is_singleton_with_bool_lowleveltype() {
        // rbool.py:11 + :36 module-level singleton.
        let a = bool_repr();
        let b = bool_repr();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.lowleveltype(), &LowLevelType::Bool);
        assert_eq!(a.class_name(), "BoolRepr");
    }

    #[test]
    fn as_int_returns_signed_repr_singleton() {
        // rbool.py:13-15 — `self.as_int = signed_repr`. Must return
        // the same Arc that rint.rs' signed_repr() cache holds.
        let r = bool_repr();
        let asi = r.as_int();
        assert!(Arc::ptr_eq(&asi, &signed_repr()));
        assert_eq!(asi.lowleveltype(), &LowLevelType::Signed);
    }

    #[test]
    fn convert_const_accepts_bool_only() {
        // rbool.py:17-20.
        let r = bool_repr();
        let ok = r.convert_const(&ConstValue::Bool(true)).unwrap();
        assert_eq!(ok.concretetype.as_ref(), Some(&LowLevelType::Bool));

        // Int is rejected — bool-specifically, unlike upstream
        // FloatRepr (which accepts "can be bool too" per rfloat.py:15)
        // and IntegerRepr (which accepts bool via the Number/Bool
        // branch).
        let err = r.convert_const(&ConstValue::Int(0)).unwrap_err();
        assert!(err.to_string().contains("not a bool"));

        let err = r.convert_const(&ConstValue::Float(0)).unwrap_err();
        assert!(err.to_string().contains("not a bool"));
    }

    #[test]
    fn convert_const_rejects_placeholder_sentinel() {
        let r = bool_repr();
        let err = r.convert_const(&ConstValue::Placeholder).unwrap_err();
        assert!(err.to_string().contains("not a bool"));
    }

    #[test]
    fn bool_repr_setup_state_matches_base_repr() {
        // Repr state machine applies to BoolRepr identically.
        let r = BoolRepr::new();
        assert_eq!(r.state().get(), Setupstate::NotInitialized);
        r.setup().unwrap();
        assert_eq!(r.state().get(), Setupstate::Finished);
    }
}
