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
pub fn register_virtualizable_roots(roots: impl IntoIterator<Item = String>) {
    REGISTERED.with(|registered| *registered.borrow_mut() = roots.into_iter().collect());
}

/// The registered roots for the current pipeline invocation. Empty when the
/// consumer declared none, which makes the minter's class test fail closed —
/// upstream's erasing branch.
pub(crate) fn virtualizable_roots() -> std::collections::HashSet<String> {
    REGISTERED.with(|registered| registered.borrow().clone())
}
