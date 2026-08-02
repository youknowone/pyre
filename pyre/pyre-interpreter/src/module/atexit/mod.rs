//! atexit module — PyPy: `pypy/module/atexit/`.
//!
//! PyPy keeps the callback list and all five operations at app level in
//! `app_atexit.py`; preserve that ownership and storage shape here.

crate::py_module! {
    "atexit",
    appleveldefs: {
        "app_atexit.py" => [
            "register",
            "unregister",
            "_clear",
            "_run_exitfuncs",
            "_ncallbacks",
        ],
    },
}
