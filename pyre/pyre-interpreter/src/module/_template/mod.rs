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
}
