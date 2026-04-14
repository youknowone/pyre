/// baseobjspace.py:2087 space.warn(w_msg, w_warningcls, stacklevel)
///
/// Minimal warning infrastructure — prints to stderr.
/// Full _warnings module integration is future work.

/// Issue a DeprecationWarning.
///
/// baseobjspace.py:2087: space.warn(space.newtext(msg), space.w_DeprecationWarning)
pub fn warn_deprecation(msg: &str) {
    eprintln!("DeprecationWarning: {msg}");
}
