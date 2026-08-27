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

/// `PyErr_ResourceWarning(source, stacklevel, format, ...)` -- the same warning
/// carrying the object it is about.
///
/// `_py_warnings._formatwarnmsg_impl` reads `msg.source`: with it the message
/// gains the object's allocation traceback, or the "Enable tracemalloc to get
/// the object allocation traceback" line when tracing is off.  A warning
/// raised with no source gets neither line, so every finalizer that reports an
/// unclosed resource passes itself.
pub fn warn_category_source(
    msg: &str,
    category_name: &str,
    stacklevel: i64,
    source: PyObjectRef,
) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let source_slot = crate::module::_warnings::pin_root_slot(source);
    let msg_slot = crate::module::_warnings::pin_root_slot(pyre_object::w_str_new(msg));
    warn_category_w_source(
        pyre_object::gc_roots::shadow_stack_get(msg_slot),
        category_name,
        stacklevel,
        pyre_object::gc_roots::shadow_stack_get(source_slot),
    )
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
    warn_category_w_source(w_msg, category_name, stacklevel, pyre_object::PY_NULL)
}

fn warn_category_w_source(
    w_msg: PyObjectRef,
    category_name: &str,
    stacklevel: i64,
    source: PyObjectRef,
) -> Result<(), crate::PyError> {
    // Upstream reaches the filters and the once-registry through
    // `space.fromcache(State)`, which exists from space construction and which
    // app code cannot reach.  pyre's equivalent is installed with the
    // `_warnings` module, so a warning raised before that point has nowhere to
    // be matched.  Report it unfiltered rather than turn a warning into an
    // error out of the operator that issued it.
    let _roots = pyre_object::gc_roots::push_roots();
    let msg_slot = crate::module::_warnings::pin_root_slot(w_msg);
    let source_slot = crate::module::_warnings::pin_root_slot(source);
    // A startup that never imports `_warnings` would leave the State absent
    // for the whole run, so the fallback below would answer every warning the
    // process ever raises.  Install it at the first warning that could use it
    // instead, once the category the caller named exists — before that the
    // exception hierarchy is not up and the fallback is the right answer
    // anyway.  `install_state` may allocate, so re-read both operands after it.
    if !crate::module::_warnings::state_is_readable()
        && crate::builtins::lookup_exc_class(category_name).is_some()
    {
        crate::module::_warnings::install_state();
    }
    let w_msg = pyre_object::gc_roots::shadow_stack_get(msg_slot);
    let source = pyre_object::gc_roots::shadow_stack_get(source_slot);
    let Some(category) = crate::builtins::lookup_exc_class(category_name)
        .filter(|_| crate::module::_warnings::state_is_readable())
    else {
        let text = crate::baseobjspace::text_w(w_msg)
            .map(|text| text.to_string())
            .unwrap_or_default();
        warn(&text, category_name);
        return Ok(());
    };
    crate::module::_warnings::do_warn(w_msg, category, stacklevel - 1, source, &[])
}

/// `do_warn_explicit`'s stderr format without the location prefix, for
/// reporting before the warnings machinery is usable.
pub fn warn(msg: &str, category: &str) {
    crate::host_seam::emit_stderr(format!("{category}: {msg}\n").as_bytes());
}
