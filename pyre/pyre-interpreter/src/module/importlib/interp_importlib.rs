//! importlib submodule stubs — PyPy: Lib/importlib/* (app-level).
//!
//! Verbatim move of the four init_importlib_* blocks previously in
//! importing.rs.  Each one is renamed to register_<suffix> to match
//! the moduledef entry points.

use crate::DictStorage;

/// importlib stub — PyPy: pypy/module/importlib/
/// Avoid loading the real importlib.__init__ since it drags in
/// _bootstrap and _bootstrap_external.
pub fn register_pkg(ns: &mut DictStorage) {
    // importlib.import_module(name, package=None) — return an imported
    // module by name. PyPy: Lib/importlib/__init__.py import_module →
    // _bootstrap._gcd_import. We defer to the interpreter's importhook
    // since it handles both builtins and source modules.
    crate::dict_storage_store(
        ns,
        "import_module",
        crate::make_builtin_function("import_module", |args| {
            let name = args.first().copied().unwrap_or(pyre_object::w_none());
            unsafe {
                if !pyre_object::is_str(name) {
                    return Err(crate::PyError::type_error(
                        "import_module: name must be str",
                    ));
                }
                let name_str = pyre_object::w_str_get_value(name).to_string();
                crate::importing::importhook(
                    &name_str,
                    pyre_object::w_none(),
                    pyre_object::w_list_new(vec![pyre_object::w_str_new("*")]),
                    0,
                    std::ptr::null(),
                )
            }
        }),
    );
    crate::dict_storage_store(
        ns,
        "invalidate_caches",
        crate::make_builtin_function_with_arity(
            "invalidate_caches",
            |_| Ok(pyre_object::w_none()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "reload",
        crate::make_builtin_function_with_arity(
            "reload",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    );
    // Mark as a package so dotted imports treat it as such.
    crate::dict_storage_store(ns, "__path__", pyre_object::w_list_new(vec![]));
}

/// importlib.util stub — minimal subset.
pub fn register_util(ns: &mut DictStorage) {
    // `importlib.util.spec_from_file_location(name, location, *, ...)`,
    // `module_from_spec(spec)`, and `find_spec(name, package=None)` —
    // accept any positional/keyword shape so the stubs do not reject
    // legitimate call signatures via the arity gate.
    crate::dict_storage_store(
        ns,
        "spec_from_file_location",
        crate::make_builtin_function("spec_from_file_location", |_| Ok(pyre_object::w_none())),
    );
    crate::dict_storage_store(
        ns,
        "module_from_spec",
        crate::make_builtin_function("module_from_spec", |_| Ok(pyre_object::w_none())),
    );
    crate::dict_storage_store(
        ns,
        "find_spec",
        crate::make_builtin_function("find_spec", |_| Ok(pyre_object::w_none())),
    );
    crate::dict_storage_store(
        ns,
        "resolve_name",
        crate::make_builtin_function("resolve_name", |args| {
            Ok(args.first().copied().unwrap_or(pyre_object::w_str_new("")))
        }),
    );
    crate::dict_storage_store(ns, "MAGIC_NUMBER", pyre_object::w_int_new(0));
}

/// importlib.abc stub — abstract base classes.
pub fn register_abc(ns: &mut DictStorage) {
    for name in [
        "Loader",
        "Finder",
        "MetaPathFinder",
        "PathEntryFinder",
        "ResourceLoader",
        "InspectLoader",
        "ExecutionLoader",
        "FileLoader",
        "SourceLoader",
    ] {
        crate::dict_storage_store(ns, name, crate::typedef::w_object());
    }
}

/// importlib.machinery stub — provides the names inspect.py references.
/// PyPy ships the real importlib; we shortcut it with a stub so pyre does
/// not have to execute _bootstrap_external.
pub fn register_machinery(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "SOURCE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".py")]),
    );
    crate::dict_storage_store(
        ns,
        "BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::dict_storage_store(
        ns,
        "EXTENSION_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".so")]),
    );
    crate::dict_storage_store(
        ns,
        "DEBUG_BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::dict_storage_store(
        ns,
        "OPTIMIZED_BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::dict_storage_store(
        ns,
        "all_suffixes",
        crate::make_builtin_function_with_arity(
            "all_suffixes",
            |_| {
                Ok(pyre_object::w_list_new(vec![
                    pyre_object::w_str_new(".py"),
                    pyre_object::w_str_new(".pyc"),
                    pyre_object::w_str_new(".so"),
                ]))
            },
            0,
        ),
    );
    crate::dict_storage_store(ns, "ModuleSpec", crate::typedef::w_object());
    crate::dict_storage_store(ns, "BuiltinImporter", crate::typedef::w_object());
    crate::dict_storage_store(ns, "FrozenImporter", crate::typedef::w_object());
    crate::dict_storage_store(ns, "PathFinder", crate::typedef::w_object());
    crate::dict_storage_store(ns, "FileFinder", crate::typedef::w_object());
    crate::dict_storage_store(ns, "SourceFileLoader", crate::typedef::w_object());
    crate::dict_storage_store(ns, "SourcelessFileLoader", crate::typedef::w_object());
    crate::dict_storage_store(ns, "ExtensionFileLoader", crate::typedef::w_object());
    crate::dict_storage_store(ns, "AppleFrameworkLoader", crate::typedef::w_object());
    crate::dict_storage_store(ns, "NamespaceLoader", crate::typedef::w_object());
    crate::dict_storage_store(ns, "WindowsRegistryFinder", crate::typedef::w_object());
}
