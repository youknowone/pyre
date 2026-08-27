//! _template module — the t-string runtime objects (Template, Interpolation)
//! that `string.templatelib` exposes.  CPython implements these in C
//! (Objects/templateobject.c, Objects/interpolationobject.c); here they are
//! app-level Python the BUILD_TEMPLATE / BUILD_INTERPOLATION opcodes construct
//! through `_build_template` / `_build_interpolation`.

crate::py_module! {
    "_template",
    appleveldefs: {
        "_template_app.py" => [
            "Template", "Interpolation",
            "_build_template", "_build_interpolation",
        ],
    },
    extra_init: |ns| {
        // `Template` and `Interpolation` are final: the C runtime types lack
        // `Py_TPFLAGS_BASETYPE`, so `class Sub(Template)` raises TypeError.
        // These app-level classes retain object's TypeDef.  Suppress the
        // CPython BASETYPE projection per type; mutating the shared TypeDef
        // would make object itself unacceptable as a base.
        for name in ["Template", "Interpolation"] {
            if let Some(t) = crate::module_ns_get(ns, name) {
                // CPython 3.14 exposes both as immutable non-heap types.
                // Keep the PyPy app-level owner internally and project only
                // the caller-visible static/immutable type capabilities.
                crate::typedef::mark_cpython_static_extension_type(t);
                unsafe { pyre_object::w_type_suppress_cpython_basetype(t) };
            }
        }
    },
}
