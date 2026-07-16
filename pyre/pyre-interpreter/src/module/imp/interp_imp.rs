//! _imp implementation — PyPy: pypy/module/imp/interp_imp.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::importing::BUILTIN_MODULES;

/// _imp stub — PyPy: pypy/module/imp/
///
/// Minimal subset required by importlib._bootstrap to decide which loader
/// handles a name. We report every name we know about as a builtin so
/// pyre's own registrations remain authoritative.
pub fn register_module(ns: pyre_object::PyObjectRef) {
    crate::module_ns_store(
        ns,
        "is_builtin",
        crate::make_builtin_function_with_arity(
            "is_builtin",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                let name = unsafe {
                    if pyre_object::is_str(args[0]) {
                        pyre_object::w_str_get_value(args[0])
                    } else {
                        return Ok(pyre_object::w_int_new(0));
                    }
                };
                let is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(name));
                Ok(pyre_object::w_int_new(if is_builtin { 1 } else { 0 }))
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "is_frozen",
        crate::make_builtin_function_with_arity(
            "is_frozen",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "is_frozen_package",
        crate::make_builtin_function_with_arity(
            "is_frozen_package",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "_frozen_module_names",
        crate::make_builtin_function("_frozen_module_names", |_| {
            Ok(pyre_object::w_list_new(Vec::new()))
        }),
    );
    // `_imp.find_frozen(name)` — `FrozenImporter.find_spec` calls this and
    // treats None as "not a frozen module". Pyre has no frozen modules, so
    // every name resolves to None and the import falls through to the next
    // finder on `sys.meta_path`.
    crate::module_ns_store(
        ns,
        "find_frozen",
        crate::make_builtin_function_with_arity(
            "find_frozen",
            |_| Ok(pyre_object::w_none()),
            1,
        ),
    );
    // `_imp._override_frozen_modules_for_tests(value)` — the CPython test
    // harness (`test.support.import_helper`) toggles frozen-module
    // overriding.  Pyre has no frozen modules, so accept and ignore.
    crate::module_ns_store(
        ns,
        "_override_frozen_modules_for_tests",
        crate::make_builtin_function("_override_frozen_modules_for_tests", |_| {
            Ok(pyre_object::w_none())
        }),
    );
    // `_imp.get_frozen_object(name, data=None)` — pyre has no frozen modules,
    // so every name is unknown and `set_frozen_error(FROZEN_NOT_FOUND)` raises
    // `ImportError("No such frozen object named %R")`.
    crate::module_ns_store(
        ns,
        "get_frozen_object",
        crate::make_builtin_function_with_arity(
            "get_frozen_object",
            |args| {
                let Some(&name) = args.first() else {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::TypeError,
                        "get_frozen_object expected at least 1 argument, got 0".to_string(),
                    ));
                };
                let name_repr = unsafe { crate::display::py_repr(name)? };
                Err(crate::PyError::new(
                    crate::PyErrorKind::ImportError,
                    format!("No such frozen object named {name_repr}"),
                ))
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "create_builtin",
        crate::make_builtin_function_with_arity(
            "create_builtin",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                Ok(args[0])
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "exec_builtin",
        crate::make_builtin_function_with_arity(
            "exec_builtin",
            |_| Ok(pyre_object::w_int_new(0)),
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "exec_dynamic",
        crate::make_builtin_function_with_arity(
            "exec_dynamic",
            |_| Ok(pyre_object::w_int_new(0)),
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "acquire_lock",
        crate::make_builtin_function_with_arity("acquire_lock", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::module_ns_store(
        ns,
        "release_lock",
        crate::make_builtin_function_with_arity("release_lock", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::module_ns_store(
        ns,
        "lock_held",
        crate::make_builtin_function_with_arity(
            "lock_held",
            |_| Ok(pyre_object::w_bool_from(false)),
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "_fix_co_filename",
        crate::make_builtin_function_with_arity(
            "_fix_co_filename",
            |_| Ok(pyre_object::w_none()),
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "extension_suffixes",
        crate::make_builtin_function_with_arity(
            "extension_suffixes",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "source_hash",
        crate::make_builtin_function_with_arity(
            "source_hash",
            |_| Ok(pyre_object::w_int_new(0)),
            2,
        ),
    );
    crate::module_ns_store(
        ns,
        "check_hash_based_pycs",
        pyre_object::w_str_new("default"),
    );
    crate::module_ns_store(ns, "pyc_magic_number_token", pyre_object::w_int_new(3495));
}
