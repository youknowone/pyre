//! Local-crate alias roots for symbolic `CallPath` resolution.
//!
//! RPython needs nothing like this: `Bookkeeper.getdesc`
//! (`bookkeeper.py:353-409`) keys `FunctionDesc`s by function-object
//! identity, so a callable has one identity regardless of the import
//! spelling. pyre resolves symbolic paths extracted from LLBC, where a
//! cross-crate callsite spells the callee with its crate name
//! (`aheui_runtime::io::output_flush`) while the graph registers under
//! module-relative spellings — so every *local* (LLBC-extracted) crate
//! name must be an alias root on both the registration side
//! (`free_function_alias_paths`) and the canonical-dedup side
//! (`populate_call_registry_from_call_graphs`).
//!
//! Seeded from the loaded LLBC set's `crate_name()`s by
//! `build_semantic_program_via_active_frontend`; the pyre trio is always
//! included so fixtures that build programs without the active frontend
//! (and the pyre production set itself) keep their spellings unchanged.

use std::cell::RefCell;

const DEFAULT_ROOTS: [&str; 3] = ["pyre_interpreter", "pyre_object", "pyre_jit"];

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

/// Registered local crate names plus the always-included pyre trio,
/// registered names first.
pub(crate) fn local_crate_roots() -> Vec<String> {
    let mut roots: Vec<String> = REGISTERED.with(|registered| registered.borrow().clone());
    for default in DEFAULT_ROOTS {
        if !roots.iter().any(|r| r == default) {
            roots.push(default.to_string());
        }
    }
    roots
}

pub(crate) fn is_local_crate_root(seg: &str) -> bool {
    DEFAULT_ROOTS.contains(&seg)
        || REGISTERED.with(|registered| registered.borrow().iter().any(|r| r == seg))
}
