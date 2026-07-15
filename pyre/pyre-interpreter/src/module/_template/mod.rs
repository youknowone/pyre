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
            "_build_template", "_build_interpolation", "_reconstruct",
        ],
    },
    extra_init: |ns| {
        // `Template` and `Interpolation` are final: the C runtime types lack
        // `Py_TPFLAGS_BASETYPE`, so `class Sub(Template)` raises TypeError.
        // App-level classes default to `acceptable_as_base_class=true`, so
        // flip it off here to reject subclassing.
        for name in ["Template", "Interpolation"] {
            if let Some(t) = crate::module_ns_get(ns, name) {
                unsafe { pyre_object::w_type_set_acceptable_as_base_class(t, false) };
            }
        }
    },
}
