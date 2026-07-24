//! _imp implementation — PyPy: pypy/module/imp/interp_imp.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::importing::BUILTIN_MODULES;
use std::sync::atomic::{AtomicI64, Ordering};

struct FrozenModule {
    name: &'static str,
    origname: Option<&'static str>,
    is_package: bool,
    source: FrozenSource,
}

enum FrozenSource {
    Stdlib(&'static str),
    Literal(&'static str),
}

static FROZEN_MODULES: &[FrozenModule] = &[
    FrozenModule {
        name: "_frozen_importlib",
        origname: Some("importlib._bootstrap"),
        is_package: false,
        source: FrozenSource::Stdlib("importlib/_bootstrap.py"),
    },
    FrozenModule {
        name: "_frozen_importlib_external",
        origname: Some("importlib._bootstrap_external"),
        is_package: false,
        source: FrozenSource::Stdlib("importlib/_bootstrap_external.py"),
    },
    FrozenModule {
        name: "zipimport",
        origname: Some("zipimport"),
        is_package: false,
        source: FrozenSource::Stdlib("zipimport.py"),
    },
    FrozenModule {
        name: "__hello__",
        origname: Some("__hello__"),
        is_package: false,
        source: FrozenSource::Stdlib("__hello__.py"),
    },
    FrozenModule {
        name: "__hello_alias__",
        origname: Some("__hello__"),
        is_package: false,
        source: FrozenSource::Stdlib("__hello__.py"),
    },
    FrozenModule {
        name: "__phello_alias__",
        origname: Some("__hello__"),
        is_package: true,
        source: FrozenSource::Stdlib("__hello__.py"),
    },
    FrozenModule {
        name: "__phello_alias__.spam",
        origname: Some("__hello__"),
        is_package: false,
        source: FrozenSource::Stdlib("__hello__.py"),
    },
    FrozenModule {
        name: "__phello__",
        origname: Some("__phello__"),
        is_package: true,
        source: FrozenSource::Stdlib("__phello__/__init__.py"),
    },
    FrozenModule {
        name: "__phello__.__init__",
        origname: Some("<__phello__"),
        is_package: false,
        source: FrozenSource::Stdlib("__phello__/__init__.py"),
    },
    FrozenModule {
        name: "__phello__.ham",
        origname: Some("__phello__.ham"),
        is_package: true,
        source: FrozenSource::Stdlib("__phello__/ham/__init__.py"),
    },
    FrozenModule {
        name: "__phello__.ham.__init__",
        origname: Some("<__phello__.ham"),
        is_package: false,
        source: FrozenSource::Stdlib("__phello__/ham/__init__.py"),
    },
    FrozenModule {
        name: "__phello__.ham.eggs",
        origname: Some("__phello__.ham.eggs"),
        is_package: false,
        source: FrozenSource::Stdlib("__phello__/ham/eggs.py"),
    },
    FrozenModule {
        name: "__phello__.spam",
        origname: Some("__phello__.spam"),
        is_package: false,
        source: FrozenSource::Stdlib("__phello__/spam.py"),
    },
    FrozenModule {
        name: "__hello_only__",
        origname: None,
        is_package: false,
        source: FrozenSource::Literal("initialized = True\n"),
    },
];

static FROZEN_OVERRIDE: AtomicI64 = AtomicI64::new(0);

fn frozen_module(name: &str) -> Option<&'static FrozenModule> {
    FROZEN_MODULES.iter().find(|entry| entry.name == name)
}

fn is_bootstrap_frozen(name: &str) -> bool {
    matches!(
        name,
        "_frozen_importlib" | "_frozen_importlib_external" | "zipimport"
    )
}

fn frozen_module_served(entry: &FrozenModule) -> bool {
    let mode = FROZEN_OVERRIDE.load(Ordering::Relaxed);
    // `_override_frozen_modules_for_tests`: 0 is the default (the normal
    // frozen table is enabled), a positive value forces frozen modules on,
    // and a negative value disables the non-essential ones, keeping only the
    // essential bootstrap set frozen.
    mode >= 0 || is_bootstrap_frozen(entry.name)
}

fn served_frozen_module(name: &str) -> Option<&'static FrozenModule> {
    frozen_module(name).filter(|entry| frozen_module_served(entry))
}

fn frozen_name(args: &[pyre_object::PyObjectRef], function: &str) -> Result<String, crate::PyError> {
    let Some(&name) = args.first() else {
        return Err(crate::PyError::type_error(format!(
            "{function} expected at least 1 argument, got 0"
        )));
    };
    if !unsafe { pyre_object::is_str(name) } {
        return Err(crate::PyError::type_error(format!(
            "{function}() argument 1 must be str"
        )));
    }
    Ok(unsafe { pyre_object::w_str_get_value(name) }.to_owned())
}

fn missing_frozen_error(name: &str) -> crate::PyError {
    crate::PyError::import_error_name_path(
        format!("No such frozen object named {name:?}"),
        pyre_object::w_str_new(name),
        pyre_object::w_none(),
    )
}

fn frozen_source(entry: &FrozenModule) -> Result<(String, String), crate::PyError> {
    match entry.source {
        FrozenSource::Literal(source) => Ok((source.to_owned(), "frozen_only".to_owned())),
        FrozenSource::Stdlib(relative) => {
            #[cfg(feature = "host_env")]
            {
                let stdlib = crate::importing::detect_stdlib_path().ok_or_else(|| {
                    crate::PyError::new(
                        crate::PyErrorKind::ImportError,
                        format!("cannot resolve source for frozen module {:?}", entry.name),
                    )
                })?;
                let path = stdlib.join(relative);
                let source = crate::importing::read_source_to_string(&path).map_err(|error| {
                    crate::PyError::new(
                        crate::PyErrorKind::ImportError,
                        format!("cannot read '{}': {error}", path.display()),
                    )
                })?;
                let code_name = entry
                    .origname
                    .unwrap_or(entry.name)
                    .strip_prefix('<')
                    .unwrap_or(entry.origname.unwrap_or(entry.name));
                Ok((source, code_name.to_owned()))
            }
            #[cfg(not(feature = "host_env"))]
            {
                let _ = relative;
                Err(crate::PyError::new(
                    crate::PyErrorKind::ImportError,
                    format!("cannot resolve source for frozen module {:?}", entry.name),
                ))
            }
        }
    }
}

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
                // `interp_imp.is_builtin`: 0 = not a builtin, 1 = a builtin
                // not yet imported, -1 = a builtin already in sys.modules and
                // thus not re-initializable (sys/builtins and any other
                // already-imported builtin).
                let is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(name));
                let result = if !is_builtin {
                    0
                } else if crate::importing::check_sys_modules(name).is_some() {
                    -1
                } else {
                    1
                };
                Ok(pyre_object::w_int_new(result))
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "is_frozen",
        crate::make_builtin_function_with_arity(
            "is_frozen",
            |args| {
                let name = frozen_name(args, "is_frozen")?;
                Ok(pyre_object::w_bool_from(
                    served_frozen_module(&name).is_some(),
                ))
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "is_frozen_package",
        crate::make_builtin_function_with_arity(
            "is_frozen_package",
            |args| {
                let name = frozen_name(args, "is_frozen_package")?;
                let entry =
                    served_frozen_module(&name).ok_or_else(|| missing_frozen_error(&name))?;
                Ok(pyre_object::w_bool_from(entry.is_package))
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "init_frozen",
        crate::make_builtin_function_with_arity(
            "init_frozen",
            // interp_imp.py:74 — frozen modules are served through the meta
            // path, never re-initialized by this legacy entry point.
            |args| {
                let _ = frozen_name(args, "init_frozen")?;
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "_frozen_module_names",
        crate::make_builtin_function("_frozen_module_names", |_| {
            Ok(pyre_object::w_list_new(
                FROZEN_MODULES
                    .iter()
                    .filter(|entry| frozen_module_served(entry))
                    .map(|entry| pyre_object::w_str_new(entry.name))
                    .collect(),
            ))
        }),
    );
    crate::module_ns_store(
        ns,
        "find_frozen",
        crate::make_builtin_function_with_arity(
            "find_frozen",
            |args| {
                let name = frozen_name(args, "find_frozen")?;
                let Some(entry) = served_frozen_module(&name) else {
                    return Ok(pyre_object::w_none());
                };
                let origname = entry
                    .origname
                    .map(pyre_object::w_str_new)
                    .unwrap_or_else(pyre_object::w_none);
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_none(),
                    pyre_object::w_bool_from(entry.is_package),
                    origname,
                ]))
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "_override_frozen_modules_for_tests",
        crate::make_builtin_function("_override_frozen_modules_for_tests", |args| {
            let Some(&value) = args.first() else {
                return Err(crate::PyError::type_error(
                    "_override_frozen_modules_for_tests expected at least 1 argument, got 0",
                ));
            };
            let value = crate::baseobjspace::gateway_int_w(value)?;
            FROZEN_OVERRIDE.store(value, Ordering::Relaxed);
            Ok(pyre_object::w_none())
        }),
    );
    crate::module_ns_store(
        ns,
        "get_frozen_object",
        crate::make_builtin_function_with_arity(
            "get_frozen_object",
            |args| {
                let name = frozen_name(args, "get_frozen_object")?;
                let entry =
                    served_frozen_module(&name).ok_or_else(|| missing_frozen_error(&name))?;
                let (source, code_name) = frozen_source(entry)?;
                let filename = format!("<frozen {code_name}>");
                let code = crate::compile::compile_source_with_filename(
                    &source,
                    crate::compile::Mode::Exec,
                    &filename,
                )
                .map_err(|error| crate::builtins::compile_err_to_syntax_error(error, &source))?;
                Ok(crate::w_code_new(
                    Box::into_raw(Box::new(code)) as *const ()
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
                // `BuiltinImporter.create_module` passes the spec and expects the
                // module back; `_load` then binds it in sys.modules. A name
                // already imported keeps its module, so the machinery and a plain
                // `import X` agree on one object.
                let Some(&spec) = args.first() else {
                    return Err(crate::PyError::type_error(
                        "create_builtin() missing required argument 'spec'",
                    ));
                };
                let w_name = crate::baseobjspace::getattr_str(spec, "name")?;
                if !unsafe { pyre_object::is_str(w_name) } {
                    return Err(crate::PyError::type_error("spec.name must be a string"));
                }
                let name = unsafe { pyre_object::w_str_get_value(w_name) }.to_string();
                if let Some(module) = crate::importing::get_sys_module(&name) {
                    return Ok(module);
                }
                crate::importing::create_builtin_module(
                    &name,
                    crate::call::getexecutioncontext(),
                )?
                .ok_or_else(|| {
                    crate::PyError::new(
                        crate::PyErrorKind::ImportError,
                        format!("no built-in module named {name}"),
                    )
                })
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
            |args| {
                // `interp_imp.py source_hash`: siphash-2-4 of the source
                // bytes keyed by the pyc magic (k0=magic, k1=0), serialized
                // low-byte-first — the 8-byte hash field of hash-based pycs
                // (`_code_to_hash_pyc` asserts `len(source_hash) == 8`).
                use std::hash::Hasher;
                let magic = crate::baseobjspace::int_w(args[0])? as u64;
                let content = if unsafe { pyre_object::bytesobject::is_bytes_like(args[1]) } {
                    unsafe { pyre_object::bytesobject::bytes_like_data(args[1]) }.to_vec()
                } else if let Some(src) = crate::typedef::buffer_as_bytes_like(args[1])? {
                    unsafe { pyre_object::bytesobject::bytes_like_data(src) }.to_vec()
                } else {
                    return Err(crate::PyError::type_error(
                        "source_hash() argument 2 must be a bytes-like object",
                    ));
                };
                let mut hasher = siphasher::sip::SipHasher24::new_with_keys(magic, 0);
                hasher.write(&content);
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(
                    &hasher.finish().to_le_bytes(),
                ))
            },
            2,
        ),
    );
    crate::module_ns_store(
        ns,
        "check_hash_based_pycs",
        pyre_object::w_str_new("default"),
    );
    // `MAGIC_NUMBER = _imp.pyc_magic_number_token.to_bytes(4, 'little')`
    // (_bootstrap_external.py) — low half is the 3.14 magic 3627, high half
    // the `\r\n` marker so the number breaks when read as text.  Cache
    // files are already segregated by `sys.implementation.cache_tag`.
    crate::module_ns_store(
        ns,
        "pyc_magic_number_token",
        pyre_object::w_int_new(0x0A0D_0E2B),
    );
}
