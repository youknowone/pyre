//! atexit module — PyPy: `pypy/module/atexit/`.
//!
//! PyPy keeps the callback list and all five operations at app level in
//! `app_atexit.py`; preserve that ownership and storage shape here.

crate::py_module! {
    "atexit",
    extra_init: |ns| {
        // `app_atexit.py` anchors the callback list on the interpreter's own
        // `sys`, so it needs that module object rather than whatever the
        // program left under the name.  A program is free to bind
        // `sys.modules["sys"] = None`, the sentinel that stops everything
        // downstream from importing it, and the body below runs when `atexit`
        // is first minted -- which for a program that never imported it is
        // during finalization, where an `ImportError` has nowhere to go.  Seed
        // the module the way `_PySys_GetOptionalAttr` reads
        // `PyInterpreterState.sysdict` instead of importing the name.
        let Some(w_sys) = crate::importing::get_interpreter_sys_module() else {
            panic!("appleveldef `app_atexit.py`: no sys module to anchor the callbacks on");
        };
        crate::importing::appleveldef_install_seeded(
            ns,
            include_str!("app_atexit.py"),
            "app_atexit.py",
            "atexit",
            &[
                "register",
                "unregister",
                "_clear",
                "_run_exitfuncs",
                "_ncallbacks",
            ],
            &[("_sys", w_sys)],
        )?;
    },
}
