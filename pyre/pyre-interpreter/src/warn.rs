//! `space.warn` — interpreter-level warnings.

/// `space.warn(space.newtext(msg), space.w_DeprecationWarning)`
pub fn warn_deprecation(msg: &str) -> Result<(), crate::PyError> {
    warn_category(msg, "DeprecationWarning", 2)
}

/// `space.warn(w_msg, w_warningcls, stacklevel)` — hands the message to
/// `_warnings.do_warn` with `stacklevel - 1`, so the filters, the module
/// `__warningregistry__` and `warnings.catch_warnings(record=True)` all
/// observe the event whether or not the `warnings` wrapper is imported.
pub fn warn_category(
    msg: &str,
    category_name: &str,
    stacklevel: i64,
) -> Result<(), crate::PyError> {
    // Upstream reaches the filters and the once-registry through
    // `space.fromcache(State)`, which exists from space construction and which
    // app code cannot reach.  pyre keeps them on the `_warnings` module, so a
    // warning raised before that module is installed — or after the name has
    // been rebound — has nowhere to be matched.  Report it unfiltered rather
    // than turn a warning into an error out of the operator that issued it.
    let Some(category) = crate::builtins::lookup_exc_class(category_name)
        .filter(|_| crate::module::_warnings::state_is_readable())
    else {
        warn(msg, category_name);
        return Ok(());
    };
    crate::module::_warnings::do_warn(
        pyre_object::w_str_new(msg),
        category,
        stacklevel - 1,
        pyre_object::PY_NULL,
        &[],
    )
}

/// `do_warn_explicit`'s stderr format without the location prefix, for
/// reporting before the warnings machinery is usable.
pub fn warn(msg: &str, category: &str) {
    crate::host_seam::emit_stderr(format!("{category}: {msg}\n").as_bytes());
}
