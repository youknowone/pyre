//! builtins module — PyPy: pypy/module/__builtin__/
//!
//! `import builtins` exposes the same names as the default builtins
//! namespace; we seed the module dict via `install_default_builtins`.

crate::py_module! {
    "builtins",
    interpleveldefs: {},
    extra_init: |ns| {
        crate::install_default_builtins(ns);
    }
}
