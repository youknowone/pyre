//! `rpython/rlib/nonconst.py` — a constant the annotator must not read as one.
//!
//! `NonConstant(x)` is `x` at run time and a *non-constant* annotation at
//! translation time: `EntryNonConstant.compute_result_annotation` returns
//! `not_const(s_arg)`, and `specialize_call` returns the argument unchanged.
//! Wrapping a literal in it is how a helper whose body is a fixed value keeps
//! its callers' branches alive — without it the annotator folds the caller's
//! `if helper(...)` and the branch is gone before the rtyper sees it.
//!
//! The translator resolves [`non_constant`] as a registered external, not as a
//! lifted graph, so the body below is only the untranslated residual.

/// `rlib/nonconst.py NonConstant`.
pub fn non_constant<T>(value: T) -> T {
    value
}
