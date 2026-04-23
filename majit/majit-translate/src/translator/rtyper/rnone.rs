//! RPython `rpython/rtyper/rnone.py` — `NoneRepr` + `none_repr` singleton.
//!
//! ## Parity scope of this commit
//!
//! This commit lands the Repr-shape boundary: `NoneRepr` struct,
//! `none_repr()` singleton, `ll_none_hash` helper, and the `SomeNone`
//! arm in [`super::rmodel::rtyper_makerepr`]. The pairtype conversion
//! blocks (`pairtype(Repr, NoneRepr).convert_from_to` +
//! `pairtype(NoneRepr, Repr).convert_from_to` + `rtype_is_` dispatch)
//! are reserved for Cascade 2c — they need `hop.genop` / `inputconst`
//! / `LowLevelOpList` infrastructure which arrives alongside the
//! pairtype double-dispatch port.
//!
//! | upstream line | pyre mirror |
//! |---|---|
//! | `class NoneRepr(Repr)` `rnone.py:10-31` | [`NoneRepr`] |
//! | `NoneRepr.lowleveltype = Void` `rnone.py:11` | `NoneRepr::lowleveltype` returns [`LowLevelType::Void`] |
//! | `NoneRepr.get_ll_eq_function` `rnone.py:22-23` | [`NoneRepr::get_ll_eq_function`] |
//! | `NoneRepr.get_ll_hash_function` `rnone.py:25-26` | [`NoneRepr::get_ll_hash_function`] |
//! | `NoneRepr.get_ll_fasthash_function` `rnone.py:28` | aliased to `get_ll_hash_function` |
//! | `none_repr = NoneRepr()` `rnone.py:33` | [`none_repr`] singleton |
//! | `ll_none_hash(_)` `rnone.py:42-43` | [`ll_none_hash`] |
//! | `__extend__(SomeNone).rtyper_makerepr` `rnone.py:35-37` | [`super::rmodel::rtyper_makerepr`] `SomeNone` arm |
//! | `__extend__(SomeNone).rtyper_makekey` `rnone.py:39-40` | [`super::rmodel::rtyper_makekey`] `SomeNone` arm |
//! | `rtype_bool` / `none_call` / `ll_str` `rnone.py:13-20` | deferred (Cascade 2 — need `hop.inputconst`) |
//! | `pairtype(Repr, NoneRepr)` `rnone.py:46-54` | deferred (Cascade 2c — pairtype dispatch) |
//! | `pairtype(NoneRepr, Repr)` `rnone.py:56-64` | deferred |
//! | `rtype_is_None` `rnone.py:66-84` | deferred |

use std::sync::{Arc, OnceLock};

use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{LlHelper, Repr, ReprState};

// ____________________________________________________________
// RPython `class NoneRepr(Repr)` (rnone.py:10-31).

/// RPython `class NoneRepr(Repr)` (`rnone.py:10-31`).
///
/// ```python
/// class NoneRepr(Repr):
///     lowleveltype = Void
///
///     def rtype_bool(self, hop):
///         return Constant(False, Bool)
///
///     def none_call(self, hop):
///         raise TyperError("attempt to call constant None")
///
///     def ll_str(self, none):
///         return llstr("None")
///
///     def get_ll_eq_function(self):
///         return None
///
///     def get_ll_hash_function(self):
///         return ll_none_hash
///
///     get_ll_fasthash_function = get_ll_hash_function
///
///     rtype_simple_call = none_call
///     rtype_call_args = none_call
/// ```
///
/// Upstream `lowleveltype = Void` means the default
/// `Repr.convert_const` (which only checks
/// `lowleveltype._contains_value(value)`) accepts arbitrary constants,
/// since `Void._contains_value` is unconditionally true. Pyre inherits
/// this behavior through the default [`Repr::convert_const`]
/// implementation — no override is needed for parity.
#[derive(Debug)]
pub struct NoneRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl NoneRepr {
    /// Upstream `NoneRepr()` no-arg constructor — all fields come from
    /// class attributes (`rnone.py:11`).
    pub fn new() -> Self {
        NoneRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Void,
        }
    }

    /// RPython `NoneRepr.get_ll_eq_function` (`rnone.py:22-23`).
    ///
    /// ```python
    /// def get_ll_eq_function(self):
    ///     return None
    /// ```
    ///
    /// Upstream returns `None` to signal "no custom eq helper" — pyre
    /// mirrors with `Option<()>` matching the `FloatRepr` convention.
    pub fn get_ll_eq_function(&self) -> Option<()> {
        None
    }

    /// RPython `NoneRepr.get_ll_hash_function` (`rnone.py:25-26`).
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     return ll_none_hash
    /// ```
    ///
    /// Upstream returns the `ll_none_hash` function pointer; pyre
    /// returns a typed helper descriptor carrying the same identity.
    pub fn get_ll_hash_function(&self) -> LlHelper {
        LlHelper::HashNone
    }

    /// RPython `NoneRepr.get_ll_fasthash_function = get_ll_hash_function`
    /// (`rnone.py:28`).
    pub fn get_ll_fasthash_function(&self) -> LlHelper {
        self.get_ll_hash_function()
    }
}

impl Default for NoneRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for NoneRepr {
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
        "NoneRepr"
    }
}

/// RPython `none_repr = NoneRepr()` (`rnone.py:33`) — module-level
/// singleton referenced from `__extend__(SomeNone).rtyper_makerepr`
/// (`rnone.py:36-37`). Pyre returns an [`Arc<NoneRepr>`] clone of the
/// cached singleton so the pointer identity upstream relies on
/// (`r is none_repr`) survives as `Arc::ptr_eq`.
pub fn none_repr() -> Arc<NoneRepr> {
    static REPR: OnceLock<Arc<NoneRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(NoneRepr::new())).clone()
}

/// RPython `ll_none_hash(_)` (`rnone.py:42-43`).
///
/// ```python
/// def ll_none_hash(_):
///     return 0
/// ```
pub fn ll_none_hash(_: ()) -> i64 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::rmodel::Setupstate;

    #[test]
    fn none_repr_is_singleton_with_void_lowleveltype() {
        // rnone.py:11 + :33 module-level singleton.
        let a = none_repr();
        let b = none_repr();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.lowleveltype(), &LowLevelType::Void);
        assert_eq!(a.class_name(), "NoneRepr");
    }

    #[test]
    fn get_ll_eq_function_returns_none_like_upstream() {
        // rnone.py:22-23.
        let r = none_repr();
        assert!(r.get_ll_eq_function().is_none());
    }

    #[test]
    fn get_ll_hash_function_returns_ll_none_hash_handle() {
        // rnone.py:25-26 `return ll_none_hash`.
        let r = none_repr();
        assert_eq!(r.get_ll_hash_function(), LlHelper::HashNone);
    }

    #[test]
    fn get_ll_fasthash_function_aliased_to_hash_function() {
        // rnone.py:28 `get_ll_fasthash_function = get_ll_hash_function`.
        let r = none_repr();
        assert_eq!(r.get_ll_fasthash_function(), r.get_ll_hash_function());
    }

    #[test]
    fn ll_none_hash_always_returns_zero() {
        // rnone.py:42-43.
        assert_eq!(ll_none_hash(()), 0);
    }

    #[test]
    fn none_repr_setup_state_matches_base_repr() {
        // Repr state machine applies to NoneRepr identically.
        let r = NoneRepr::new();
        assert_eq!(r.state().get(), Setupstate::NotInitialized);
        r.setup().unwrap();
        assert_eq!(r.state().get(), Setupstate::Finished);
    }
}
