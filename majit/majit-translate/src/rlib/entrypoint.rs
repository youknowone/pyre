//! Port of `rpython/rlib/entrypoint.py`.
//!
//! Currently carries only [`export_symbol`] — the other entrypoint
//! helpers (`jit_entrypoint`, `entrypoint_lowlevel`,
//! `entrypoint_highlevel`, `secondary_entrypoints`) live on top of
//! lltype / rtyper machinery that is not ported locally and are
//! added here as their dependencies land.
//!
//! Upstream line-for-line:
//!
//! ```python
//! def export_symbol(func):
//!     func.exported_symbol = True
//!     return func
//! ```
//!
//! (`rpython/rlib/entrypoint.py:10-12`.)

use crate::flowspace::model::{GraphFunc, HostObject};

/// Port of upstream `rpython/rlib/entrypoint.py:10-12 export_symbol`.
///
/// Upstream sets `func.exported_symbol = True` in-place on a Python
/// function object and returns the same reference. The Rust port
/// cannot mutate the `GraphFunc` stored inside an
/// [`HostObject::UserFunction`] in place — `HostObject` wraps its
/// `GraphFunc` in an `Arc<HostObjectInner>` keyed on pointer identity
/// — so the flag is applied by rebuilding the wrapper around a
/// mutated `GraphFunc` clone. Callers that compare `HostObject`
/// equality across a single `export_symbol` hop are already
/// upstream-incompatible (the flag is by contract a permanent
/// mutation), so the identity break matches upstream semantics
/// rather than the operational contract.
///
/// Non-function host objects are returned unchanged: upstream
/// `export_symbol` only ever runs on Python function objects (called
/// from `interactive.py:18 self.entry_point = export_symbol(entry_point)`
/// right after the caller selects a function as the translation
/// entry point), so any other input passing through this helper is
/// outside the upstream contract. Silently passing through keeps the
/// port permissive without obscuring that callers *should* be handing
/// in a `UserFunction`.
pub fn export_symbol(func: HostObject) -> HostObject {
    match func.user_function() {
        Some(existing) => {
            // Upstream `func.exported_symbol = True; return func`. The
            // attribute write is the observable side effect; the
            // returned reference is the same object. We approximate
            // by producing an HostObject carrying a GraphFunc whose
            // flag is True.
            let mut flagged: GraphFunc = existing.clone();
            flagged.exported_symbol = true;
            HostObject::new_user_function(flagged)
        }
        None => func,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, Constant};

    fn sample_user_function(name: &str) -> HostObject {
        // Minimum viable GraphFunc for `HostObject::new_user_function`
        // — upstream `GraphFunc.new(name, globals)` at
        // `model.rs:3068-3088` defaults every non-essential slot. No
        // code object is needed because the `exported_symbol` flag
        // lives on `GraphFunc` itself, not on `HostCode`.
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let gf = GraphFunc::new(name, globals);
        HostObject::new_user_function(gf)
    }

    #[test]
    fn export_symbol_sets_flag_on_graphfunc() {
        // Upstream `entrypoint.py:11 func.exported_symbol = True`.
        let func = sample_user_function("demo");
        assert!(
            !func.user_function().expect("user function").exported_symbol,
            "fresh GraphFunc defaults to exported_symbol=false per model.rs:3087"
        );

        let flagged = export_symbol(func);
        assert!(
            flagged
                .user_function()
                .expect("user function")
                .exported_symbol,
            "export_symbol must flip exported_symbol to true"
        );
    }

    #[test]
    fn export_symbol_preserves_qualname_and_user_fn_identity_semantics() {
        // `new_user_function` derives qualname from `GraphFunc.name`, so
        // the post-flag HostObject must still round-trip the same
        // qualname / `is_user_function()` invariants the caller at
        // `interactive.py:18` relies on downstream when storing the
        // result as `self.entry_point` and later feeding it to
        // `buildflowgraph` / `_prebuilt_graphs`.
        let func = sample_user_function("pkg.demo");
        let flagged = export_symbol(func.clone());
        assert!(flagged.is_user_function());
        assert_eq!(flagged.qualname(), "pkg.demo");

        // Per the module doc: the resulting HostObject is a *new*
        // wrapper, not the input object. Upstream Python identity
        // parity isn't achievable without interior mutability, so the
        // contract is "flag observable on the returned value", not
        // "returned value is `is`-identical to the input".
        assert!(!func.user_function().expect("").exported_symbol);
        assert!(flagged.user_function().expect("").exported_symbol);
    }

    #[test]
    fn export_symbol_is_idempotent() {
        // Calling `export_symbol` twice must not change the observable
        // flag state — upstream's `True -> True` assignment is a no-op.
        let once = export_symbol(sample_user_function("demo"));
        let twice = export_symbol(once);
        assert!(twice.user_function().expect("").exported_symbol);
    }

    #[test]
    fn export_symbol_passes_through_non_user_function() {
        // The module doc pins this: non-function inputs are outside
        // upstream's contract, but the helper must not panic. Class
        // objects (`HostObject::new_class`) are the most visible
        // non-function kind — route through and confirm they come
        // back unchanged.
        let cls = HostObject::new_class("pkg.SomeType", vec![]);
        let after = export_symbol(cls.clone());
        assert_eq!(after, cls, "non-function inputs pass through unchanged");
    }
}
