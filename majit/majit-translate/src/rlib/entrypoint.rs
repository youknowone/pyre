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

use std::sync::atomic::Ordering;

use crate::flowspace::model::HostObject;

/// Port of upstream `rpython/rlib/entrypoint.py:10-12 export_symbol`.
///
/// Upstream `func.exported_symbol = True; return func` is a Python
/// in-place attribute assignment — the function object's identity
/// does not change, and every other reference to the same callable
/// observes the flipped flag (Python object identity).
///
/// The Rust port mirrors this with `GraphFunc.exported_symbol:
/// Arc<AtomicBool>` (see `model.rs:GraphFunc::exported_symbol`).
/// `Clone for GraphFunc` Arc-clones the AtomicBool so every clone of
/// a single GraphFunc shares the same flag cell — `store(true)` from
/// any reference is observable on every other, matching upstream's
/// "same Python object" semantics.
///
/// Returns the input HostObject unchanged in identity (`Arc::ptr_eq`
/// holds): callers can rely on `self.entry_point ==
/// export_symbol(entry_point)` and on `_prebuilt_graphs[entry_point]`
/// remaining keyed on the same instance.
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
    if let Some(gf) = func.user_function() {
        // Upstream `func.exported_symbol = True`. The Arc<AtomicBool>
        // is shared with every other clone of this GraphFunc — both
        // the HostObject's wrapped Box<GraphFunc> and any
        // adapter-built `pygraph.func` / `pygraph.graph.func` clones
        // observe the flip immediately.
        gf.exported_symbol.store(true, Ordering::Relaxed);
    }
    // Upstream `return func` — same object identity in PyPy. The
    // Rust port returns the same HostObject (Arc-cloned on the way
    // through the function boundary, but `Arc::ptr_eq` is preserved).
    func
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, Constant, GraphFunc};

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
            !func
                .user_function()
                .expect("user function")
                .exported_symbol
                .load(Ordering::Relaxed),
            "fresh GraphFunc defaults to exported_symbol=false per model.rs:3087"
        );

        let flagged = export_symbol(func);
        assert!(
            flagged
                .user_function()
                .expect("user function")
                .exported_symbol
                .load(Ordering::Relaxed),
            "export_symbol must flip exported_symbol to true"
        );
    }

    #[test]
    fn export_symbol_preserves_host_object_identity() {
        // Upstream parity: `func.exported_symbol = True; return func`
        // returns the SAME Python object. In Rust, `Arc::ptr_eq` on
        // the input vs the output must hold (fixed by routing the
        // flag through `Arc<AtomicBool>` so `export_symbol` no
        // longer rebuilds the HostObject wrapper).
        let func = sample_user_function("pkg.demo");
        let flagged = export_symbol(func.clone());
        assert!(flagged.is_user_function());
        assert_eq!(flagged.qualname(), "pkg.demo");
        assert_eq!(
            flagged, func,
            "export_symbol must preserve HostObject identity (Arc::ptr_eq holds)"
        );

        // Both references observe the flag flip — Arc<AtomicBool>
        // shares the cell across every GraphFunc clone.
        assert!(
            func.user_function()
                .expect("")
                .exported_symbol
                .load(Ordering::Relaxed),
            "input HostObject sees the flip (shared Arc<AtomicBool>)"
        );
        assert!(
            flagged
                .user_function()
                .expect("")
                .exported_symbol
                .load(Ordering::Relaxed),
        );
    }

    #[test]
    fn export_symbol_propagates_through_independent_graphfunc_clones() {
        // The constructor at `interactive.rs` builds a `pygraph.func`
        // clone of the GraphFunc separately from the HostObject's
        // wrapped one. Upstream Python's in-place mutation flips the
        // flag on every reference to the same function object;
        // `Arc<AtomicBool>` mirrors that — even an independently
        // cloned GraphFunc observes the flip.
        let host = sample_user_function("pkg.demo");
        let detached_clone = host.user_function().expect("uf").clone();
        assert!(!detached_clone.exported_symbol.load(Ordering::Relaxed));

        export_symbol(host);
        assert!(
            detached_clone.exported_symbol.load(Ordering::Relaxed),
            "Arc<AtomicBool> must propagate the flip across detached clones"
        );
    }

    #[test]
    fn export_symbol_is_idempotent() {
        // Calling `export_symbol` twice must not change the observable
        // flag state — upstream's `True -> True` assignment is a no-op.
        let once = export_symbol(sample_user_function("demo"));
        let twice = export_symbol(once);
        assert!(
            twice
                .user_function()
                .expect("")
                .exported_symbol
                .load(Ordering::Relaxed)
        );
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
