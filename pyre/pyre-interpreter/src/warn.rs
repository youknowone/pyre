/// baseobjspace.py:2087-2093 space.warn(w_msg, w_warningcls, stacklevel)
///
/// _warnings/interp_warnings.py:263-290 do_warn → do_warn_explicit.
///
/// Routes through Python's `warnings.warn()` when available, otherwise
/// falls back to stderr in the standard format.
/// Currently uses stderr-only path; Python-level routing requires the
/// _warnings C-extension module (not yet ported).

/// baseobjspace.py:2087: space.warn(space.newtext(msg), space.w_DeprecationWarning)
pub fn warn_deprecation(msg: &str) {
    warn(msg, "DeprecationWarning");
}

/// baseobjspace.py:2087-2093
///
/// _warnings/interp_warnings.py:263: do_warn(space, w_message, w_category, stacklevel-1)
/// do_warn_explicit formats: "{filename}:{lineno}: {category}: {message}"
pub fn warn(msg: &str, category: &str) {
    eprintln!("{category}: {msg}");
}
