//! `_minimal_curses` — `pypy/module/_minimal_curses/moduledef.py`.
//!
//! The package table is this file. `setupterm`, `tigetstr` and `tparm`
//! are the `interp_curses` function pointers `interpleveldefs` names.
//! `error` is `app_curses.py` `class error(Exception)`.

mod fficurses;
mod interp_curses;

pyre_interpreter::py_module! {
    "_minimal_curses",
    exceptions: {
        // app_curses.py `class error(Exception)`.
        "error" => pyre_interpreter::builtins::lookup_exc_class("Exception")
            .expect("Exception installed"),
    },
    extra_init: |ns| {
        let mut ns = ns;
        for &(name, value) in fficurses::CURSES_INTS {
            let stored = pyre_object::with_roots!(ns => pyre_object::w_int_new(value));
            pyre_interpreter::module_ns_store(ns, name, stored);
        }
        fn install(
            mut ns: pyre_object::PyObjectRef,
            name: &'static str,
            func: pyre_interpreter::BuiltinCodeFn,
            arity: u16,
            sig: Option<pyre_interpreter::Signature>,
        ) -> pyre_object::PyObjectRef {
            let value = pyre_object::with_roots!(ns => pyre_interpreter::gateway::with_module(
                "_minimal_curses",
                pyre_interpreter::make_module_builtin_function_with_arity_and_maybe_sig(
                    name, func, arity, sig,
                ),
            ));
            pyre_interpreter::module_ns_store(ns, name, value);
            ns
        }
        ns = install(
            ns,
            "setupterm",
            interp_curses::setupterm,
            interp_curses::setupterm_pyre_arity(),
            interp_curses::setupterm_pyre_sig(),
        );
        ns = install(
            ns,
            "tigetstr",
            interp_curses::tigetstr,
            interp_curses::tigetstr_pyre_arity(),
            interp_curses::tigetstr_pyre_sig(),
        );
        ns = install(
            ns,
            "tparm",
            interp_curses::tparm,
            pyre_interpreter::HOPELESS,
            Some(interp_curses::tparm_sig()),
        );
        let _ = ns;
    },
}
