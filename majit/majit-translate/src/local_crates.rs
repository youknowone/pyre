//! Local-crate alias roots for symbolic `CallPath` resolution.
//!
//! RPython needs nothing like this: `Bookkeeper.getdesc`
//! (`bookkeeper.py`) keys `FunctionDesc`s by function-object
//! identity, so a callable has one identity regardless of the import
//! spelling. pyre resolves symbolic paths extracted from LLBC, where a
//! cross-crate callsite spells the callee with its crate name
//! (`somecrate::io::output_flush`) while the graph registers under
//! module-relative spellings — so every *local* (LLBC-extracted) crate
//! name must be an alias root on both the registration side
//! (`free_function_alias_paths`) and the canonical-dedup side
//! (`populate_call_registry_from_call_graphs`).
//!
//! Seeded from the loaded LLBC set's `crate_name()`s by
//! `build_semantic_program_via_active_frontend`. Consumers that construct
//! programs without the active frontend must register their own crate roots.

use std::cell::RefCell;

thread_local! {
    /// Per-pipeline-invocation local-crate alias roots, seeded once at the
    /// top of `build_semantic_program_via_active_frontend` and read back at
    /// the alias / dedup / tie-break sites during the SAME invocation.
    ///
    /// Thread-local, not a process-global `RwLock`: a translate pipeline
    /// runs start-to-finish on one thread (there is no `par_iter` inside it),
    /// and `generated::all_jitcodes` already scopes the whole per-thread
    /// pipeline registry with a `thread_local!` `OnceCell` to preserve
    /// RPython's single-thread annotator invariant. A shared `RwLock` let a
    /// second pipeline on another thread (parallel `cargo test`, or any
    /// future parallel translate) overwrite this run's roots between its own
    /// seed and read, flaking alias resolution. The roots belong to one
    /// invocation, so scoping them to the invocation's thread is exact — the
    /// same TLS adaptation of a PyPy GIL-singleton as
    /// `jitdriver.rs::BACK_EDGE_BH_BUILDER`.
    static REGISTERED: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Replace this thread's registered local-crate set with one pipeline
/// invocation's LLBC crate names. A later invocation on the same thread
/// overwrites (per-invocation semantics, like the `STRUCT_ORIGIN_REGISTRY`
/// re-seed).
pub(crate) fn register_local_crate_roots(names: impl IntoIterator<Item = String>) {
    REGISTERED.with(|registered| *registered.borrow_mut() = names.into_iter().collect());
}

/// Registered local crate names for the current pipeline invocation.
pub(crate) fn local_crate_roots() -> Vec<String> {
    REGISTERED.with(|registered| registered.borrow().clone())
}

pub(crate) fn is_local_crate_root(seg: &str) -> bool {
    REGISTERED.with(|registered| registered.borrow().iter().any(|r| r == seg))
}

/// Run one stand-alone, single-LLBC lowering with that artefact's crate root
/// present in the same invocation-local alias set used by the whole-program
/// frontend.
///
/// `build_semantic_program_via_active_frontend` seeds all loaded crates once.
/// The public `lower_fun_decl*` entry points bypass that driver, so without a
/// scoped seed their transparent constructors retain the defining crate while
/// the layout metadata derived from the same LLBC uses crate-relative names.
/// Preserve any surrounding multi-crate invocation and restore it even when
/// lowering unwinds.
pub(crate) fn with_local_crate_root<R>(root: &str, f: impl FnOnce() -> R) -> R {
    struct Restore(Vec<String>);

    impl Drop for Restore {
        fn drop(&mut self) {
            register_local_crate_roots(std::mem::take(&mut self.0));
        }
    }

    let previous = local_crate_roots();
    if previous.iter().any(|registered| registered == root) {
        return f();
    }
    let mut scoped = previous.clone();
    scoped.push(root.to_string());
    register_local_crate_roots(scoped);
    let _restore = Restore(previous);
    f()
}
