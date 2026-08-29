//! The `_virtualizable_` class declaration, as a registered set of roots.
//!
//! RPython puts it on the class: `_virtualizable_ = ['x', 'y[*]']` is read
//! back through `classdesc.get_param('_virtualizable_')`, and
//! `rlib/jit.py`'s `hint` entry consults exactly that before it mints
//! `access_directly` on a `SomeInstance` — a value whose class does not
//! declare it has the flags deleted instead.
//!
//! Pyre's interpreter is hand-written Rust and no struct carries that
//! parameter, so the declaration is supplied out of band by the consumer, the
//! same way the codewriter's `GraphTransformConfig::vable_fields` is
//! (`rvirtualizable.rs` records why). This module is where the front end
//! reads it back, so the minter's class test can run before there is a
//! `ClassDesc` to ask.
//!
//! Upstream the declaration is per-CLASS Bookkeeper state, so it is neither
//! thread- nor invocation-scoped: `get_param` asks the class every time.
//! Scoping it to an invocation is a pyre deviation forced by having no
//! `ClassDesc` at this point, and the way it is kept honest is that the
//! pipeline re-seeds this registry from its own `AnalyzeConfig` on every run
//! (`lib.rs analyze_pipeline_from_module_paths`), deriving the roots from the
//! `owner_root` the same config already puts on every
//! `VirtualizableFieldDescriptor`. One declaration channel, refreshed per
//! invocation — the shape `local_crates` has.

use std::cell::RefCell;

thread_local! {
    /// Per-pipeline-invocation `_virtualizable_` roots, seeded by the
    /// consumer before it builds a program and read back by
    /// `front::semantic::propagate_access_directly` during the SAME
    /// invocation.
    ///
    /// Thread-local for the reason `local_crates.rs` spells out: a translate
    /// pipeline runs start-to-finish on one thread, and a process-global
    /// would let a parallel `cargo test` pipeline overwrite this run's
    /// declaration between its own seed and read.
    static REGISTERED: RefCell<std::collections::HashSet<String>> =
        RefCell::new(std::collections::HashSet::new());
}

/// Replace this thread's `_virtualizable_` root set. A later invocation on
/// the same thread overwrites, matching the per-invocation semantics of
/// `local_crates::register_local_crate_roots`.
///
/// The production caller is the pipeline itself, which derives the set from
/// the `AnalyzeConfig` it was handed. Public for a consumer that builds a
/// program without going through `analyze_pipeline_from_module_paths`; such a
/// consumer owns the ordering, since the read happens during the build.
pub fn register_virtualizable_roots(roots: impl IntoIterator<Item = String>) {
    REGISTERED.with(|registered| *registered.borrow_mut() = roots.into_iter().collect());
}

/// The registered roots for the current pipeline invocation. Empty when the
/// consumer declared none, which makes the minter's class test fail closed —
/// upstream's erasing branch.
pub(crate) fn virtualizable_roots() -> std::collections::HashSet<String> {
    REGISTERED.with(|registered| registered.borrow().clone())
}
