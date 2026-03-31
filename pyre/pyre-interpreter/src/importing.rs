//! Module importing — PyPy equivalent: pypy/module/imp/importing.py
//!
//! Implements the import machinery:
//! - `importhook()` — main entry point (called by IMPORT_NAME opcode)
//! - `find_module()` — locate a .py file on sys.path
//! - `load_source_module()` — compile and execute a .py file
//! - `check_sys_modules()` — consult the module cache
//! - `import_all_from()` — IMPORT_STAR handler

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::{CodeObject, Mode, compile_source_with_filename};
use crate::{PyExecutionContext, PyNamespace, namespace_load, namespace_store};
use pyre_object::*;

use crate::eval::eval_frame_plain;
use crate::pyframe::PyFrame;

// ── sys.modules cache ────────────────────────────────────────────────
// PyPy equivalent: space.sys.get('modules') — a dict mapping module names
// to module objects. We use a thread-local HashMap<String, PyObjectRef>.

thread_local! {
    static SYS_MODULES: RefCell<HashMap<String, PyObjectRef>> = RefCell::new(HashMap::new());
    /// sys.path equivalent — list of directories to search for modules.
    static SYS_PATH: RefCell<Vec<PathBuf>> = RefCell::new(Vec::new());
    /// Builtin modules registry — PyPy equivalent: space.builtin_modules
    ///
    /// Maps module name → initializer function that populates a PyNamespace.
    /// Each builtin module is lazily created on first import.
    static BUILTIN_MODULES: RefCell<HashMap<&'static str, fn(&mut PyNamespace)>> =
        RefCell::new(HashMap::new());
}

// ── builtin module registry ──────────────────────────────────────────
// PyPy equivalent: space.builtin_modules dict + MixedModule.interpleveldefs

/// Register a builtin module initializer.
///
/// PyPy equivalent: Module.install() → space.builtin_modules[name] = mod
pub fn register_builtin_module(name: &'static str, init: fn(&mut PyNamespace)) {
    BUILTIN_MODULES.with(|m| {
        m.borrow_mut().insert(name, init);
    });
}

/// Install all standard builtin modules.
///
/// PyPy equivalent: baseobjspace.py `make_builtins()` +
/// `install_mixedmodule()` for each module in objspace.usemodules.
pub fn install_builtin_modules() {
    register_builtin_module("math", crate::module::math::moduledef::init);
    register_builtin_module("cmath", crate::module::math::cmath_moduledef::init);
    register_builtin_module("time", crate::module::time::moduledef::init);
    register_builtin_module("sys", crate::module::sys::moduledef::init);
    register_builtin_module("operator", crate::module::operator::moduledef::init);
    register_builtin_module("_operator", crate::module::operator::moduledef::init);
    register_builtin_module("builtins", crate::module::__builtin__::moduledef::init);
    register_builtin_module("_io", crate::module::_io::moduledef::init);
    register_builtin_module("_sre", crate::module::_sre::moduledef::init);

    // Minimal C-extension stubs required for stdlib import chains.
    // PyPy: these are all implemented as mixed modules under pypy/module/.
    register_builtin_module("_weakref", init_weakref);
    register_builtin_module("_abc", init_abc);
    register_builtin_module("_functools", init_functools);
    register_builtin_module("_thread", init_thread);
    register_builtin_module("itertools", init_itertools);
    register_builtin_module("_contextvars", init_contextvars);
    for name in &[
        "_signal",
        "_string",
        "_stat",
        "_codecs",
        "_locale",
        "_warnings",
        "_imp",
        "_collections",
        "copyreg",
        "_heapq",
        "_opcode",
        "_tokenize",
        "_typing",
        "_bisect",
        "errno",
        "atexit",
        "_struct",
        "binascii",
        "_hashlib",
        "_sha2",
        "_md5",
        "_sha1",
        "_sha3",
        "_blake2",
        "_random",
        "_decimal",
        "_pickle",
        "_datetime",
        "_json",
        "_csv",
        "marshal",
        "posix",
        "fcntl",
        "grp",
        "pwd",
        "select",
        "_socket",
        "_tracemalloc",
    ] {
        register_builtin_module(name, empty_module_init);
    }
}

/// Empty module initializer for C-extension stubs.
fn empty_module_init(_ns: &mut PyNamespace) {}

/// _collections_abc stub — provides _check_methods for io.py etc.
/// The real _collections_abc.py requires ABCMeta (metaclass support).
fn init_collections_abc(ns: &mut PyNamespace) {
    // _check_methods(C, *methods) — check if class C has all methods
    crate::namespace_store(
        ns,
        "_check_methods",
        crate::builtin_code_new("_check_methods", |args| {
            // args[0] = C (class), args[1..] = method names
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(true));
            }
            let cls = args[0];
            for &method_name in &args[1..] {
                let name = unsafe { pyre_object::w_str_get_value(method_name) };
                // Check if method exists in class MRO
                let found = if unsafe { pyre_object::is_type(cls) } {
                    unsafe { crate::baseobjspace::lookup_in_type(cls, name) }.is_some()
                } else {
                    false
                };
                if !found {
                    return Ok(pyre_object::w_not_implemented());
                }
            }
            Ok(pyre_object::w_bool_from(true))
        }),
    );
    // Stub ABC classes
    crate::namespace_store(ns, "Hashable", crate::typedef::w_object());
    crate::namespace_store(ns, "Awaitable", crate::typedef::w_object());
    crate::namespace_store(ns, "Coroutine", crate::typedef::w_object());
    crate::namespace_store(ns, "Iterator", crate::typedef::w_object());
    crate::namespace_store(ns, "Generator", crate::typedef::w_object());
    crate::namespace_store(ns, "Iterable", crate::typedef::w_object());
    crate::namespace_store(ns, "Callable", crate::typedef::w_object());
    crate::namespace_store(ns, "Sized", crate::typedef::w_object());
    crate::namespace_store(ns, "Container", crate::typedef::w_object());
    crate::namespace_store(ns, "Collection", crate::typedef::w_object());
    crate::namespace_store(ns, "Sequence", crate::typedef::w_object());
    crate::namespace_store(ns, "MutableSequence", crate::typedef::w_object());
    crate::namespace_store(ns, "Mapping", crate::typedef::w_object());
    crate::namespace_store(ns, "MutableMapping", crate::typedef::w_object());
    crate::namespace_store(ns, "Set", crate::typedef::w_object());
    crate::namespace_store(ns, "MutableSet", crate::typedef::w_object());
    crate::namespace_store(ns, "ByteString", crate::typedef::w_object());
    crate::namespace_store(ns, "Buffer", crate::typedef::w_object());
    crate::namespace_store(ns, "Reversible", crate::typedef::w_object());
    crate::namespace_store(ns, "MappingView", crate::typedef::w_object());
    crate::namespace_store(ns, "KeysView", crate::typedef::w_object());
    crate::namespace_store(ns, "ItemsView", crate::typedef::w_object());
    crate::namespace_store(ns, "ValuesView", crate::typedef::w_object());
    crate::namespace_store(ns, "AsyncIterator", crate::typedef::w_object());
    crate::namespace_store(ns, "AsyncGenerator", crate::typedef::w_object());
    crate::namespace_store(ns, "AsyncIterable", crate::typedef::w_object());
}

/// itertools stub
fn init_itertools(ns: &mut PyNamespace) {
    // chain(*iterables) → flat iterator
    crate::namespace_store(
        ns,
        "chain",
        crate::builtin_code_new("chain", |args| {
            let mut items = Vec::new();
            for &arg in args {
                items.extend(crate::builtins::collect_iterable(arg)?);
            }
            let n = items.len();
            let list = pyre_object::w_list_new(items);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // starmap stub
    crate::namespace_store(
        ns,
        "starmap",
        crate::builtin_code_new("starmap", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    // count(start=0, step=1)
    crate::namespace_store(
        ns,
        "count",
        crate::builtin_code_new("count", |_| Ok(pyre_object::w_none())),
    );
    // repeat
    crate::namespace_store(
        ns,
        "repeat",
        crate::builtin_code_new("repeat", |_| Ok(pyre_object::w_none())),
    );
    // islice
    crate::namespace_store(
        ns,
        "islice",
        crate::builtin_code_new("islice", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    // groupby
    crate::namespace_store(
        ns,
        "groupby",
        crate::builtin_code_new("groupby", |_| Ok(pyre_object::w_none())),
    );
}

/// _contextvars stub
fn init_contextvars(ns: &mut PyNamespace) {
    // ContextVar(name, *, default=_MISSING) — context variable
    crate::namespace_store(
        ns,
        "ContextVar",
        crate::builtin_code_new("ContextVar", |args| {
            // Return stub object with get/set methods
            let obj = pyre_object::w_instance_new(crate::typedef::w_object());
            if !args.is_empty() {
                let _ = crate::baseobjspace::setattr(obj, "name", args[0]);
            }
            // get() returns default or raises LookupError
            let _ = crate::baseobjspace::setattr(
                obj,
                "get",
                crate::builtin_code_new("get", |args| {
                    // Return default if provided
                    if args.len() > 1 {
                        Ok(args[1])
                    } else {
                        Ok(pyre_object::w_none())
                    }
                }),
            );
            let _ = crate::baseobjspace::setattr(
                obj,
                "set",
                crate::builtin_code_new("set", |_| Ok(pyre_object::w_none())),
            );
            Ok(obj)
        }),
    );
    crate::namespace_store(
        ns,
        "Context",
        crate::builtin_code_new("Context", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "Token",
        crate::builtin_code_new("Token", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "copy_context",
        crate::builtin_code_new("copy_context", |_| Ok(pyre_object::w_none())),
    );
}

/// _weakref stub — PyPy: pypy/module/_weakref/
fn init_weakref(ns: &mut PyNamespace) {
    // ref(obj[, callback]) → weakref. Stub: returns obj itself (no GC).
    crate::namespace_store(
        ns,
        "ref",
        crate::builtin_code_new("ref", |args| {
            Ok(if args.is_empty() {
                pyre_object::w_none()
            } else {
                args[0]
            })
        }),
    );
    crate::namespace_store(
        ns,
        "proxy",
        crate::builtin_code_new("proxy", |args| {
            Ok(if args.is_empty() {
                pyre_object::w_none()
            } else {
                args[0]
            })
        }),
    );
    crate::namespace_store(
        ns,
        "getweakrefcount",
        crate::builtin_code_new("getweakrefcount", |_| Ok(pyre_object::w_int_new(0))),
    );
    crate::namespace_store(
        ns,
        "getweakrefs",
        crate::builtin_code_new("getweakrefs", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
}

/// _abc stub — PyPy: pypy/module/_abc/
fn init_abc(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "get_cache_token",
        crate::builtin_code_new("get_cache_token", |_| Ok(pyre_object::w_int_new(0))),
    );
    crate::namespace_store(
        ns,
        "_abc_init",
        crate::builtin_code_new("_abc_init", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "_abc_register",
        crate::builtin_code_new("_abc_register", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "_abc_instancecheck",
        crate::builtin_code_new("_abc_instancecheck", |_args| {
            Ok(pyre_object::w_bool_from(false))
        }),
    );
    crate::namespace_store(
        ns,
        "_abc_subclasscheck",
        crate::builtin_code_new("_abc_subclasscheck", |_args| {
            Ok(pyre_object::w_bool_from(false))
        }),
    );
    crate::namespace_store(
        ns,
        "_get_dump",
        crate::builtin_code_new("_get_dump", |_| Ok(pyre_object::w_tuple_new(vec![]))),
    );
    crate::namespace_store(
        ns,
        "_reset_registry",
        crate::builtin_code_new("_reset_registry", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "_reset_caches",
        crate::builtin_code_new("_reset_caches", |_| Ok(pyre_object::w_none())),
    );
}

/// _functools stub
fn init_functools(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "reduce",
        crate::builtin_code_new("reduce", |_| {
            Err(crate::PyError::type_error("reduce not implemented"))
        }),
    );
    crate::namespace_store(
        ns,
        "cmp_to_key",
        crate::builtin_code_new("cmp_to_key", |_| {
            Err(crate::PyError::type_error("cmp_to_key not implemented"))
        }),
    );
}

/// _thread stub
fn init_thread(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "allocate_lock",
        crate::builtin_code_new("allocate_lock", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "LockType",
        crate::builtin_code_new("LockType", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "RLock",
        crate::builtin_code_new("RLock", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "get_ident",
        crate::builtin_code_new("get_ident", |_| Ok(pyre_object::w_int_new(1))),
    );
    crate::namespace_store(
        ns,
        "_count",
        crate::builtin_code_new("_count", |_| Ok(pyre_object::w_int_new(1))),
    );
}

/// Try to load a builtin module by name.
///
/// PyPy equivalent: find_module() → C_BUILTIN path →
/// getbuiltinmodule() → Module.__init__ + startup()
fn load_builtin_module(name: &str) -> Option<PyObjectRef> {
    let init_fn = BUILTIN_MODULES.with(|m| m.borrow().get(name).copied())?;

    let mut namespace = Box::new(PyNamespace::new());
    namespace.fix_ptr();

    // Set __name__ (PyPy: Module.__init__ sets __name__)
    let name_obj = pyre_object::w_str_new(name);
    namespace_store(&mut namespace, "__name__", name_obj);

    // Run module-specific initializer (PyPy: interpleveldefs)
    init_fn(&mut namespace);

    let ns_ptr = Box::into_raw(namespace);
    let module = w_module_new(name, ns_ptr as *mut u8);
    Some(module)
}

/// Initialize sys.path with the directory containing the main script.
///
/// PyPy equivalent: sys.path is populated at startup with the script
/// directory, then PYTHONPATH entries, then the stdlib.
pub fn init_sys_path(script_dir: &Path) {
    // Register builtin modules (PyPy: make_builtins / setup_builtin_modules)
    install_builtin_modules();

    SYS_PATH.with(|p| {
        let mut path = p.borrow_mut();
        path.clear();
        // Script directory first (PyPy: first entry in sys.path)
        path.push(script_dir.to_path_buf());
        // Current working directory as fallback
        if let Ok(cwd) = std::env::current_dir() {
            if cwd != script_dir {
                path.push(cwd);
            }
        }
        // CPython stdlib path is detected lazily on first stdlib import
        // to avoid spawning python3 subprocess on every startup.
        // See find_module() → ensure_stdlib_path().
    });
}

/// Detect CPython stdlib path via `python3 -c "import sysconfig; ..."`.
///
/// PyPy equivalent: initpath.py scans for lib-python/X.Y at startup.
fn detect_stdlib_path() -> Option<PathBuf> {
    // Try PYRE_STDLIB env var first
    if let Ok(p) = std::env::var("PYRE_STDLIB") {
        let path = PathBuf::from(p);
        if path.is_dir() {
            return Some(path);
        }
    }
    // Auto-detect via python3
    let output = std::process::Command::new("python3")
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['stdlib'])",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?;
    let path = PathBuf::from(s.trim());
    if path.is_dir() { Some(path) } else { None }
}

/// Add a directory to sys.path.
pub fn add_sys_path(dir: &Path) {
    SYS_PATH.with(|p| {
        let mut path = p.borrow_mut();
        let pb = dir.to_path_buf();
        if !path.contains(&pb) {
            path.push(pb);
        }
    });
}

// ── check_sys_modules ────────────────────────────────────────────────
// PyPy equivalent: importing.py `check_sys_modules(space, w_modulename)`

fn check_sys_modules(name: &str) -> Option<PyObjectRef> {
    SYS_MODULES.with(|m| m.borrow().get(name).copied())
}

fn set_sys_module(name: &str, module: PyObjectRef) {
    SYS_MODULES.with(|m| {
        m.borrow_mut().insert(name.to_string(), module);
    });
}

// ── find_module ──────────────────────────────────────────────────────
// PyPy equivalent: importing.py `find_module()`
// Searches sys.path for `<partname>.py` or `<partname>/__init__.py` (package).

#[derive(Debug)]
enum FindInfo {
    /// A .py source file was found.
    SourceFile { pathname: PathBuf },
    /// A package directory with __init__.py was found.
    Package { dirpath: PathBuf },
    /// A builtin (Rust-implemented) module was found.
    /// PyPy equivalent: C_BUILTIN modtype in find_module()
    Builtin,
}

fn find_module(partname: &str) -> Option<FindInfo> {
    // Check builtin modules first (PyPy: space.builtin_modules check in find_module)
    let is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(partname));
    if is_builtin {
        return Some(FindInfo::Builtin);
    }

    // Try sys.path first
    if let Some(info) = find_in_sys_path(partname) {
        return Some(info);
    }

    // Lazy stdlib detection — only on first miss (avoid python3 spawn at startup)
    ensure_stdlib_path();
    return find_in_sys_path(partname);
}

/// Detect and add CPython stdlib to sys.path (once).
fn ensure_stdlib_path() {
    thread_local! {
        static DONE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    DONE.with(|d| {
        if d.get() {
            return;
        }
        d.set(true);
        if let Some(stdlib) = detect_stdlib_path() {
            add_sys_path(&stdlib);
        }
    });
}

fn find_in_sys_path(partname: &str) -> Option<FindInfo> {
    SYS_PATH.with(|p| {
        let path = p.borrow();
        for dir in path.iter() {
            // Check for package: <dir>/<partname>/__init__.py
            let pkg_dir = dir.join(partname);
            let init_file = pkg_dir.join("__init__.py");
            if init_file.is_file() {
                return Some(FindInfo::Package { dirpath: pkg_dir });
            }

            // Check for source file: <dir>/<partname>.py
            let source_file = dir.join(format!("{partname}.py"));
            if source_file.is_file() {
                return Some(FindInfo::SourceFile {
                    pathname: source_file,
                });
            }
        }
        None
    })
}

// ── parse_source_module ──────────────────────────────────────────────
// PyPy equivalent: importing.py `parse_source_module(space, pathname, source)`

fn parse_source_module(pathname: &str, source: &str) -> Result<CodeObject, String> {
    compile_source_with_filename(source, Mode::Exec, pathname)
}

// ── exec_code_module ─────────────────────────────────────────────────
// PyPy equivalent: importing.py `exec_code_module(space, w_mod, code_w, ...)`
//
// Execute a code object in the module's namespace dict.

fn exec_code_module(
    code: CodeObject,
    namespace: *mut PyNamespace,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    let code_ptr = Box::into_raw(Box::new(code));
    let mut frame = PyFrame::new_with_namespace(code_ptr, execution_context, namespace);
    eval_frame_plain(&mut frame)
}

// ── load_source_module ───────────────────────────────────────────────
// PyPy equivalent: importing.py `load_source_module()`
//
// Parse + execute a .py source file, producing a module object.

fn load_source_module(
    modulename: &str,
    pathname: &Path,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    let source = std::fs::read_to_string(pathname).map_err(|e| {
        crate::PyError::new(
            crate::PyErrorKind::ImportError,
            format!("cannot read '{}': {e}", pathname.display()),
        )
    })?;

    let pathname_str = pathname.to_string_lossy();
    let code = parse_source_module(&pathname_str, &source).map_err(|e| {
        crate::PyError::new(
            crate::PyErrorKind::ImportError,
            format!("cannot compile '{}': {e}", pathname.display()),
        )
    })?;

    // Create a fresh namespace for the module, seeded with builtins.
    // PyPy equivalent: Module.__init__ creates w_dict = space.newdict()
    // then exec_code_module sets __builtins__ and runs code in w_dict.
    let ctx = unsafe { &*execution_context };
    let mut namespace = Box::new(ctx.fresh_namespace());
    namespace.fix_ptr();

    // Set __name__ in the module namespace (PyPy: Module.__init__ sets __name__)
    let name_obj = pyre_object::w_str_new(modulename);
    crate::namespace_store(&mut namespace, "__name__", name_obj);

    // Set __file__ (PyPy: _prepare_module sets __file__)
    let file_obj = pyre_object::w_str_new(&pathname_str);
    crate::namespace_store(&mut namespace, "__file__", file_obj);

    // Set __package__ — PyPy: _prepare_module sets __package__
    // For "a.b.c" → __package__ = "a.b"; for "a" → __package__ = "a"
    let pkg = if let Some(dot) = modulename.rfind('.') {
        &modulename[..dot]
    } else {
        modulename
    };
    crate::namespace_store(&mut namespace, "__package__", pyre_object::w_str_new(pkg));

    let ns_ptr = Box::into_raw(namespace);

    // Create the module object BEFORE execution and register in sys.modules.
    // PyPy: load_source_module → set_sys_modules BEFORE exec_code_module.
    // This prevents infinite recursion on circular imports.
    let module = w_module_new(modulename, ns_ptr as *mut u8);
    set_sys_module(modulename, module);

    exec_code_module(code, ns_ptr, execution_context)?;

    Ok(module)
}

// ── load_package ─────────────────────────────────────────────────────
// PyPy equivalent: load_module with PKG_DIRECTORY modtype

fn load_package(
    modulename: &str,
    dirpath: &Path,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // Add package directory to sys.path BEFORE executing __init__.py,
    // so that relative sub-imports within the package can find siblings.
    // PyPy: sets __path__ on module before exec.
    add_sys_path(dirpath);

    let init_path = dirpath.join("__init__.py");
    let module = load_source_module(modulename, &init_path, execution_context)?;

    // Set __path__ and __package__ on the module namespace
    let ns_ptr = unsafe { w_module_get_dict_ptr(module) } as *mut PyNamespace;
    let path_str = pyre_object::w_str_new(&dirpath.to_string_lossy());
    let path_list = pyre_object::w_list_new(vec![path_str]);
    unsafe {
        crate::namespace_store(&mut *ns_ptr, "__path__", path_list);
        crate::namespace_store(
            &mut *ns_ptr,
            "__package__",
            pyre_object::w_str_new(modulename),
        );
    }

    Ok(module)
}

// ── load_part ────────────────────────────────────────────────────────
// PyPy equivalent: importing.py `load_part()`

fn load_part(
    modulename: &str,
    partname: &str,
    execution_context: *const PyExecutionContext,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    // Check sys.modules cache first
    if let Some(cached) = check_sys_modules(modulename) {
        return Ok(Some(cached));
    }

    // Find the module on disk
    let find_info = find_module(partname);
    let Some(info) = find_info else {
        return Ok(None);
    };

    let module = match info {
        FindInfo::SourceFile { pathname } => {
            load_source_module(modulename, &pathname, execution_context)?
        }
        FindInfo::Package { dirpath } => load_package(modulename, &dirpath, execution_context)?,
        FindInfo::Builtin => {
            // PyPy: getbuiltinmodule() path
            let m = load_builtin_module(partname).ok_or_else(|| crate::PyError {
                kind: crate::PyErrorKind::ImportError,
                message: format!("builtin module '{modulename}' failed to initialize"),
                exc_object: std::ptr::null_mut(),
            })?;
            // Store builtin modules in cache immediately
            set_sys_module(modulename, m);
            m
        }
    };

    Ok(Some(module))
}

// ── _absolute_import ─────────────────────────────────────────────────
// PyPy equivalent: importing.py `_absolute_import()`

fn absolute_import(
    modulename: &str,
    w_fromlist: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    let parts: Vec<&str> = modulename.split('.').collect();
    let mut first: Option<PyObjectRef> = None;
    let mut prefix = Vec::new();

    for (level, &part) in parts.iter().enumerate() {
        prefix.push(part);
        let full_name = prefix.join(".");
        let w_mod = load_part(&full_name, part, execution_context)?;
        let Some(module) = w_mod else {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ImportError,
                format!("No module named '{modulename}'"),
            ));
        };
        if level == 0 {
            first = Some(module);
        }
    }

    // PyPy: if w_fromlist is not None, return the leaf module.
    // Otherwise, return the first (top-level) module.
    if !w_fromlist.is_null() && !unsafe { is_none(w_fromlist) } {
        // `from X.Y import Z` → return the leaf module (Y)
        if let Some(cached) = check_sys_modules(modulename) {
            return Ok(cached);
        }
    }

    // `import X.Y` → return the top-level module (X)
    first.ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::ImportError,
            format!("No module named '{modulename}'"),
        )
    })
}

// ── importhook ───────────────────────────────────────────────────────
// PyPy equivalent: importing.py `importhook()`
//
// Main entry point called by the IMPORT_NAME opcode.
// Stack: [level, fromlist] → [module]

pub fn importhook(
    name: &str,
    w_globals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    if name.is_empty() && level < 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "Empty module name",
        ));
    }

    if level > 0 {
        return relative_import(name, w_globals, w_fromlist, level, execution_context);
    }

    absolute_import(name, w_fromlist, execution_context)
}

/// Relative import: `from .foo import bar` (level=1), `from ..foo import bar` (level=2).
///
/// PyPy: importing.py `_relative_import()`.
/// Resolves the package base from __package__ or __name__ in w_globals,
/// strips `level - 1` trailing components, then does absolute import.
fn relative_import(
    name: &str,
    w_globals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // Get the package name from the calling module's globals.
    // PyPy: pkgname = globals.get('__package__') or globals.get('__name__')
    let package = resolve_package_name(w_globals);
    let package = package.ok_or_else(|| crate::PyError {
        kind: crate::PyErrorKind::ImportError,
        message: "attempted relative import with no known parent package".to_string(),
        exc_object: std::ptr::null_mut(),
    })?;

    // Strip (level - 1) trailing components from package
    // PyPy: for dotted name "a.b.c" with level=2, strip "c" → "a.b", then strip "b" → "a"
    let mut parts: Vec<&str> = package.split('.').collect();
    let strips = (level - 1) as usize;
    if strips >= parts.len() {
        return Err(crate::PyError {
            kind: crate::PyErrorKind::ImportError,
            message: format!(
                "attempted relative import beyond top-level package (package='{package}', level={level})"
            ),
            exc_object: std::ptr::null_mut(),
        });
    }
    for _ in 0..strips {
        parts.pop();
    }
    let base = parts.join(".");

    // Build the fully-qualified module name
    let fqn = if name.is_empty() {
        base.clone()
    } else {
        format!("{base}.{name}")
    };

    absolute_import(&fqn, w_fromlist, execution_context)
}

/// Extract the package name from the calling module's globals namespace.
///
/// PyPy: importing.py — checks __package__ first, falls back to __name__,
/// strips the last component if __name__ has dots (module in a package).
fn resolve_package_name(w_globals: PyObjectRef) -> Option<String> {
    if w_globals.is_null() {
        return None;
    }
    let ns = w_globals as *const crate::PyNamespace;
    let ns = unsafe { &*ns };

    // Try __package__ first (PyPy: space.finditem_str(w_globals, '__package__'))
    if let Some(&pkg) = ns.get("__package__") {
        if !pkg.is_null() && unsafe { pyre_object::is_str(pkg) } {
            let s = unsafe { pyre_object::w_str_get_value(pkg) };
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }

    // Fallback: __name__ (for modules inside packages)
    if let Some(&name_obj) = ns.get("__name__") {
        if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
            let name = unsafe { pyre_object::w_str_get_value(name_obj) };
            // If the module has a __path__, it's a package — use __name__ as-is
            if ns.get("__path__").is_some() {
                return Some(name.to_string());
            }
            // Otherwise strip the last component (module name within package)
            if let Some(dot) = name.rfind('.') {
                return Some(name[..dot].to_string());
            }
        }
    }

    None
}

// ── import_from ──────────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py `IMPORT_FROM`
//
// Get an attribute from the module on TOS. Like `space.getattr(w_module, w_name)`.

pub fn import_from(module: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    // First try the module's namespace dict (PyPy: space.getattr → w_dict lookup)
    if unsafe { is_module(module) } {
        let ns_ptr = unsafe { w_module_get_dict_ptr(module) } as *mut PyNamespace;
        if !ns_ptr.is_null() {
            let ns = unsafe { &*ns_ptr };
            if let Ok(value) = namespace_load(ns, name) {
                return Ok(value);
            }
        }
    }

    // Fallback: try getattr (for non-module objects or attrs set via setattr)
    match crate::baseobjspace::getattr(module, name) {
        Ok(value) => Ok(value),
        Err(_) => Err(crate::PyError::new(
            crate::PyErrorKind::ImportError,
            format!("cannot import name '{name}'"),
        )),
    }
}

// ── import_all_from ──────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py `import_all_from(module, into_locals)`
//
// Merge all public names from a module into the current namespace.
// If __all__ exists, use it; otherwise copy all names not starting with '_'.

pub fn import_all_from(module: PyObjectRef, into_namespace: *mut PyNamespace) {
    if !unsafe { is_module(module) } {
        return;
    }

    let ns_ptr = unsafe { w_module_get_dict_ptr(module) } as *mut PyNamespace;
    if ns_ptr.is_null() {
        return;
    }

    let src_ns = unsafe { &*ns_ptr };
    let dst_ns = unsafe { &mut *into_namespace };

    // Check for __all__ (PyPy: try module.__all__)
    let has_all = src_ns.get("__all__").is_some();

    if has_all {
        // TODO: iterate __all__ list and copy named entries.
        // For now, fall through to copying all non-underscore names.
    }

    // Copy all names not starting with '_' (PyPy: skip_leading_underscores)
    for name in src_ns.keys() {
        if name.starts_with('_') {
            continue;
        }
        if let Some(&value) = src_ns.get(name) {
            if !value.is_null() {
                namespace_store(dst_ns, name, value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sys_modules_cache() {
        let sentinel = w_none();
        set_sys_module("test_cached", sentinel);
        let cached = check_sys_modules("test_cached");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), sentinel);
    }

    #[test]
    fn test_find_module_nonexistent() {
        // Should not find a module that doesn't exist
        let result = find_module("__nonexistent_pyre_test_module__");
        assert!(result.is_none());
    }
}
