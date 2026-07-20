/// baseobjspace.py:2087-2093 space.warn(w_msg, w_warningcls, stacklevel)
///
/// _warnings/interp_warnings.py:263-290 do_warn → do_warn_explicit.
///
/// Routes through Python's `warnings.warn()` when available, otherwise
/// falls back to stderr in the standard format.
/// Currently uses stderr-only path; Python-level routing requires the
/// _warnings C-extension module (not yet ported).

/// baseobjspace.py:2087: space.warn(space.newtext(msg), space.w_DeprecationWarning)
pub fn warn_deprecation(msg: &str) -> Result<(), crate::PyError> {
    if let (Some(warnings), Some(category)) = (
        crate::importing::get_sys_module("warnings"),
        crate::builtins::lookup_exc_class("DeprecationWarning"),
    ) {
        if let Ok(warn_fn) = crate::baseobjspace::getattr_str(warnings, "warn") {
            crate::call::call_function_impl_result(
                warn_fn,
                &[
                    pyre_object::w_str_new(msg),
                    category,
                    pyre_object::w_int_new(2),
                ],
            )
            .map(|_| ())?;
            return Ok(());
        }
    }
    warn(msg, "DeprecationWarning");
    Ok(())
}

/// baseobjspace.py:2087-2093
///
/// _warnings/interp_warnings.py:263: do_warn(space, w_message, w_category, stacklevel-1)
/// do_warn_explicit formats: "{filename}:{lineno}: {category}: {message}"
pub fn warn(msg: &str, category: &str) {
    crate::host_seam::emit_stderr(format!("{category}: {msg}\n").as_bytes());
}
