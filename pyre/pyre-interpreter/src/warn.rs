//! `space.warn` — interpreter-level warnings.

use pyre_object::PyObjectRef;

/// A `space.newtext(...)` applied to an interpreter-level string literal.
///
/// RPython builds the string for a literal once, at translation time, and
/// `ll_strhash` memoizes its digest in that prebuilt, so `space.newtext` on a
/// constant costs neither a copy nor a rehash however often it runs.  pyre
/// reaches the same place with a cell per literal: `w_str_new` is immortal
/// (`malloc_typed` header over a `malloc_raw` buffer), so the box is never
/// swept and never relocated and needs no root.
pub struct PrebuiltText(std::sync::OnceLock<usize>);

impl PrebuiltText {
    pub const fn new() -> Self {
        Self(std::sync::OnceLock::new())
    }

    /// The wrapped constant.  `text` must be the same literal on every call.
    pub fn get(&self, text: &str) -> PyObjectRef {
        *self.0.get_or_init(|| pyre_object::w_str_new(text) as usize) as PyObjectRef
    }
}

impl Default for PrebuiltText {
    fn default() -> Self {
        Self::new()
    }
}

/// `space.warn(space.newtext(msg), space.w_DeprecationWarning)`
pub fn warn_deprecation(msg: &str) -> Result<(), crate::PyError> {
    warn_category(msg, "DeprecationWarning", 2)
}

/// `space.warn(w_msg, w_warningcls, stacklevel)` for a message that is built
/// per call.
pub fn warn_category(
    msg: &str,
    category_name: &str,
    stacklevel: i64,
) -> Result<(), crate::PyError> {
    warn_category_w(pyre_object::w_str_new(msg), category_name, stacklevel)
}

/// `space.warn(w_msg, w_warningcls, stacklevel)` — hands the message to
/// `_warnings.do_warn` with `stacklevel - 1`, so the filters, the module
/// `__warningregistry__` and `warnings.catch_warnings(record=True)` all
/// observe the event whether or not the `warnings` wrapper is imported.
pub fn warn_category_w(
    w_msg: PyObjectRef,
    category_name: &str,
    stacklevel: i64,
) -> Result<(), crate::PyError> {
    // Upstream reaches the filters and the once-registry through
    // `space.fromcache(State)`, which exists from space construction and which
    // app code cannot reach.  pyre's equivalent is installed with the
    // `_warnings` module, so a warning raised before that point has nowhere to
    // be matched.  Report it unfiltered rather than turn a warning into an
    // error out of the operator that issued it.
    let Some(category) = crate::builtins::lookup_exc_class(category_name)
        .filter(|_| crate::module::_warnings::state_is_readable())
    else {
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_msg);
        let text = crate::baseobjspace::text_w(w_msg)
            .map(|text| text.to_string())
            .unwrap_or_default();
        warn(&text, category_name);
        return Ok(());
    };
    crate::module::_warnings::do_warn(w_msg, category, stacklevel - 1, pyre_object::PY_NULL, &[])
}

/// `do_warn_explicit`'s stderr format without the location prefix, for
/// reporting before the warnings machinery is usable.
pub fn warn(msg: &str, category: &str) {
    crate::host_seam::emit_stderr(format!("{category}: {msg}\n").as_bytes());
}
