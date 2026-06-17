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
#[cfg(feature = "host_env")]
use std::path::{Path, PathBuf};

use crate::{CodeObject, Mode, compile_source_with_filename};
use crate::{DictStorage, PyExecutionContext, dict_storage_store};
use pyre_object::*;

/// Module-local re-export of the host-OS surface.  Routes through
/// `rustpython_host_env` when the `host_env` feature is enabled; when
/// disabled the same names fall back to `std::*` shims so call sites
/// stay uniform.
#[cfg(feature = "host_env")]
pub(crate) mod host {
    #[cfg(not(target_arch = "wasm32"))]
    pub use rustpython_host_env::fs;
    pub use rustpython_host_env::os;
}
#[cfg(not(feature = "host_env"))]
pub(crate) mod host {
    pub mod fs {
        pub use std::fs::{metadata, read, read_dir, read_to_string, symlink_metadata};
    }
    pub mod os {
        pub fn current_dir() -> std::io::Result<std::path::PathBuf> {
            std::env::current_dir()
        }
        pub fn var(key: &str) -> Result<String, std::env::VarError> {
            std::env::var(key)
        }
        pub fn vars_os() -> std::env::VarsOs {
            std::env::vars_os()
        }
        pub fn process_id() -> u32 {
            std::process::id()
        }
        pub fn isatty(fd: i32) -> bool {
            unsafe { libc::isatty(fd) != 0 }
        }
        pub fn rename(
            from: impl AsRef<std::path::Path>,
            to: impl AsRef<std::path::Path>,
        ) -> std::io::Result<()> {
            std::fs::rename(from, to)
        }
        pub fn urandom(size: usize) -> std::io::Result<Vec<u8>> {
            use std::io::Read;
            let mut f = std::fs::File::open("/dev/urandom")?;
            let mut buf = vec![0u8; size];
            f.read_exact(&mut buf)?;
            Ok(buf)
        }
    }
}
#[cfg(not(target_arch = "wasm32"))]
use host::fs as host_fs;
use host::os as host_os;

// ── sys.modules cache ────────────────────────────────────────────────
// PyPy equivalent: space.sys.get('modules') — a dict mapping module names
// to module objects. We use a thread-local HashMap<String, PyObjectRef>.

thread_local! {
    static SYS_MODULES: RefCell<HashMap<String, PyObjectRef>> = RefCell::new(HashMap::new());
    /// The Python-visible `sys.modules` dict. Kept in sync with SYS_MODULES
    /// so that `sys.modules['name']` lookups work from Python code.
    static SYS_MODULES_DICT: std::cell::Cell<PyObjectRef> = const { std::cell::Cell::new(pyre_object::PY_NULL) };
    /// sys.path equivalent — list of directories to search for modules.
    #[cfg(feature = "host_env")]
    static SYS_PATH: RefCell<Vec<PathBuf>> = RefCell::new(Vec::new());
    /// Builtin modules registry — PyPy equivalent: space.builtin_modules
    ///
    /// Maps module name → initializer function that populates a DictStorage.
    /// Each builtin module is lazily created on first import.
    pub(crate) static BUILTIN_MODULES: RefCell<HashMap<&'static str, fn(&mut DictStorage)>> =
        RefCell::new(HashMap::new());
}

// ── builtin module registry ──────────────────────────────────────────
// PyPy equivalent: space.builtin_modules dict + MixedModule.interpleveldefs
//
// Lazy loading (MixedModule.buildloaders / getdictvalue,
// `mixedmodule.py:84-193`): PyPy defers two things — (1) creating a
// module's contents until the module is first imported, and (2)
// evaluating each interpleveldef/appleveldef until the corresponding
// attribute is first accessed.  Pyre achieves (1) directly: this
// registry stores `name → init` and `load_builtin_module` runs `init`
// on demand at first import, never at interpreter startup.  (2) has no
// counterpart and is deliberately not ported: pyre's interpleveldefs are
// compile-time `const` / function-pointer expressions (not interp-eval
// strings), so there is nothing expensive to defer per attribute — a
// per-attribute loader table would be a side-table with no upstream
// basis.  The import-triggered `init` IS the buildloaders equivalent.

/// Register a builtin module initializer.
///
/// PyPy equivalent: Module.install() → space.builtin_modules[name] = mod
pub fn register_builtin_module(name: &'static str, init: fn(&mut DictStorage)) {
    BUILTIN_MODULES.with(|m| {
        m.borrow_mut().insert(name, init);
    });
}

/// Install all standard builtin modules.
///
/// Mirrors PyPy's `baseobjspace.make_builtins()` +
/// `install_mixedmodule()` walk of `objspace.usemodules`.  The
/// `pyre_install_module!` arms below give a per-line declarative shape:
///
/// * `name(module)`               — register `crate::module::module::init` under `"name"` (alias arm).
/// * `module`                     — `name` defaults to the module identifier.
/// * `name => path`               — explicit init function path.
///
/// This is an explicit hand-maintained list by design — the upstream
/// equivalent (`pypy/config/pypyoption.py` `essential_modules` /
/// `default_modules` / `working_modules`) is likewise an explicit set of
/// string literals with platform conditionals, not filesystem discovery.
/// Automatic discovery is intentionally not done: it could not express
/// the alias arms (`"_operator"` → `operator`), explicit-path arms
/// (`importlib.machinery` → a non-default init fn), or the
/// `#[cfg(unix)]` gating that `resource` / `fcntl` / `syslog` require.
pub fn install_builtin_modules() {
    macro_rules! pyre_install_module {
        // `module` — `register_builtin_module("module", crate::module::module::init)`.
        ($mod:ident) => {
            register_builtin_module(stringify!($mod), crate::module::$mod::init);
        };
        // `name(module)` — re-register `module::init` under a different name.
        ($name:literal ( $mod:ident )) => {
            register_builtin_module($name, crate::module::$mod::init);
        };
        // `name => path::to::fn` — explicit init fn.
        ($name:literal => $path:path) => {
            register_builtin_module($name, $path);
        };
    }

    // Core pyre modules backed by `interpleveldefs` tables.
    pyre_install_module!(math);
    pyre_install_module!(cmath);
    #[cfg(not(target_arch = "wasm32"))]
    pyre_install_module!(time);
    pyre_install_module!(sys);
    pyre_install_module!(operator);
    pyre_install_module!("_operator"(operator));
    pyre_install_module!("builtins"(__builtin__));
    pyre_install_module!(_io);
    pyre_install_module!(_sre);

    // C-extension stubs required for stdlib import chains
    // (PyPy: pypy/module/* mixed modules).
    pyre_install_module!(_weakref);
    pyre_install_module!(_abc);
    pyre_install_module!(_functools);
    pyre_install_module!(_thread);
    pyre_install_module!(itertools);
    pyre_install_module!(_contextvars);
    pyre_install_module!(_codecs);
    #[cfg(not(target_arch = "wasm32"))]
    pyre_install_module!(posix);
    pyre_install_module!(errno);
    pyre_install_module!(_collections);
    pyre_install_module!(_ast);
    pyre_install_module!(_opcode);
    pyre_install_module!(_imp);

    // importlib package — four submodules backed by distinct init fns.
    pyre_install_module!(
        "importlib.machinery" =>
        crate::module::importlib::interp_importlib::register_machinery
    );
    pyre_install_module!(
        "importlib" =>
        crate::module::importlib::interp_importlib::register_pkg
    );
    pyre_install_module!(
        "importlib.util" =>
        crate::module::importlib::interp_importlib::register_util
    );
    pyre_install_module!(
        "importlib.abc" =>
        crate::module::importlib::interp_importlib::register_abc
    );

    // __pypy__ package + builders submodule — the PyPy-only surface
    // pickle.py imports (identity_dict + builders.BytesBuilder).
    pyre_install_module!("__pypy__" => crate::module::__pypy__::init);
    pyre_install_module!("__pypy__.builders" => crate::module::__pypy__::builders::init);

    #[cfg(not(target_arch = "wasm32"))]
    pyre_install_module!(_signal);
    pyre_install_module!(atexit);
    #[cfg(not(target_arch = "wasm32"))]
    pyre_install_module!(pwd);
    #[cfg(not(target_arch = "wasm32"))]
    pyre_install_module!(grp);
    #[cfg(unix)]
    pyre_install_module!(resource);
    #[cfg(unix)]
    pyre_install_module!(fcntl);
    #[cfg(unix)]
    pyre_install_module!(syslog);
    pyre_install_module!(select);
    pyre_install_module!(termios);
    pyre_install_module!(_socket);
    #[cfg(not(target_arch = "wasm32"))]
    pyre_install_module!(mmap);
    #[cfg(not(target_arch = "wasm32"))]
    pyre_install_module!(faulthandler);
    pyre_install_module!(_ctypes);
    #[cfg(not(target_arch = "wasm32"))]
    pyre_install_module!(_posixshmem);
    pyre_install_module!(_multiprocessing);
    pyre_install_module!(_locale);
    pyre_install_module!(_random);
    pyre_install_module!(_pickle);
    pyre_install_module!(_struct);
    pyre_install_module!(gc);
    pyre_install_module!(unicodedata);

    // `_sysconfigdata_{abiflags}_{platform}_{multiarch}` is a generated
    // Python module containing `build_time_vars = {...}` that sysconfig
    // imports from `_init_posix`.  Empty dict suffices.
    // PyPy: `pypy/tool/build_cffi_imports.py` creates the same file.
    for name in &[
        "_sysconfigdata__darwin_",
        "_sysconfigdata__linux_",
        "_sysconfigdata__linux_x86_64-linux-gnu",
        "_sysconfigdata__linux_aarch64-linux-gnu",
    ] {
        register_builtin_module(name, init_sysconfigdata_empty);
    }

    // Empty C-extension stubs — `_opcode_metadata.py` etc. exist in the
    // real stdlib and are loaded from disk, but their builtin shims here
    // simply succeed at `import X`.
    for name in &[
        "_string",
        "_warnings",
        "_heapq",
        "_tokenize",
        "_typing",
        "_bisect",
        "binascii",
        "_hashlib",
        "_sha2",
        "_md5",
        "_sha1",
        "_sha3",
        "_blake2",
        "_decimal",
        "_datetime",
        "_json",
        "_csv",
        "marshal",
        "_tracemalloc",
        "_stat",
        "_asyncio",
        "_queue",
        "_zoneinfo",
        "array",
        "zlib",
    ] {
        register_builtin_module(name, empty_module_init);
    }
}

/// Empty module initializer for C-extension stubs.
fn empty_module_init(_ns: &mut DictStorage) {}

/// `_sysconfigdata_*` stub — sysconfig imports this generated module to
/// read the CPython build variables. We expose a minimal `build_time_vars`
/// dict that lets sysconfig initialize without crashing.
fn init_sysconfigdata_empty(ns: &mut DictStorage) {
    let vars = pyre_object::w_dict_new();
    // A few keys are load-bearing — sysconfig.get_config_vars() populates
    // them, but an import-time crash hits on 'Py_GIL_DISABLED' and
    // similar. Leave the dict empty; .get('X') returns None for unknown
    // keys which every caller already handles.
    crate::dict_storage_store(ns, "build_time_vars", vars);
}

/// Try to load a builtin module by name.
///
/// PyPy equivalent: `find_module()` → C_BUILTIN path →
/// `getbuiltinmodule()` → `Module.__init__` + `startup()`.
///
/// PyPy `pypy/objspace/std/dictmultiobject.py:60-69` allocates a
/// `W_ModuleDictObject` for every module via
/// `allocate_and_init_instance(module=True)`.  Pyre mirrors that here
/// by running the legacy `init_fn(&mut DictStorage)` against a
/// temporary `DictStorage`, then folding the populated entries into a
/// fresh `W_ModuleDictObject` whose `ModuleDictStrategy` (from
/// `celldict.py:28`) is the post-Phase-5 canonical store.  The
/// temporary storage drops at function exit; the module's `w_dict`
/// is the `W_ModuleDictObject`.
fn load_builtin_module(name: &str) -> Option<PyObjectRef> {
    let init_fn = BUILTIN_MODULES.with(|m| m.borrow().get(name).copied())?;

    let mut namespace = DictStorage::new();
    namespace.fix_ptr();

    // Set __name__ (PyPy: Module.__init__ sets __name__)
    let name_obj = pyre_object::w_str_new(name);
    dict_storage_store(&mut namespace, "__name__", name_obj);

    // Run module-specific initializer (PyPy: interpleveldefs)
    init_fn(&mut namespace);

    // Fold the legacy DictStorage population into the upstream
    // `W_ModuleDictObject` carrier.  `init_fn` continues to take
    // `&mut DictStorage` so the ~20 builtin moduledef.rs init
    // functions remain untouched in this slice; the storage drops at
    // function exit and the W_ModuleDictObject owns the live state.
    let w_dict = pyre_object::w_module_dict_new();
    for (key, &value) in namespace.entries() {
        if !value.is_null() {
            // MixedModule parity: interp-level builtin functions carry the
            // module name as `__module__`, so `pickle` can save them by
            // reference (`save_global`) without guessing via `whichmodule`.
            unsafe { crate::function::builtin_function_set_module(value, name_obj) };
            unsafe { pyre_object::w_dict_setitem_str(w_dict, key, value) };
        }
    }
    let module = pyre_object::w_module_new_aliasing_dict(name, std::ptr::null_mut(), w_dict);
    // `pypy/interpreter/baseobjspace.py:647` installs the self
    // reference `space.builtin.w_dict['__builtins__'] = space.builtin`
    // so user code can reach the builtins module through
    // `import builtins; builtins.__builtins__`.  The pyre split
    // between EC.builtins_module (used by LOAD_GLOBAL fallback) and
    // the import-time module (returned here) is a known pre-existing
    // adaptation; install the self-reference on the imported flavour
    // so `import builtins; builtins.__builtins__ is builtins` holds
    // for user code regardless of the split.
    if name == "builtins" {
        unsafe { pyre_object::w_dict_setitem_str(w_dict, "__builtins__", module) };
    }
    Some(module)
}

/// Initialize sys.path with the directory containing the main script.
///
/// PyPy equivalent: sys.path is populated at startup with the script
/// directory, then PYTHONPATH entries, then the stdlib.
#[cfg(feature = "host_env")]
pub fn init_sys_path(script_dir: &Path) {
    // Register builtin modules (PyPy: make_builtins / setup_builtin_modules)
    install_builtin_modules();

    SYS_PATH.with(|p| {
        let mut path = p.borrow_mut();
        path.clear();
        // Script directory first (PyPy: first entry in sys.path)
        path.push(script_dir.to_path_buf());
        // Current working directory as fallback
        if let Ok(cwd) = host_os::current_dir() {
            if cwd != script_dir {
                path.push(cwd);
            }
        }
        // CPython stdlib path is detected lazily on first stdlib import
        // to avoid spawning python3 subprocess on every startup.
        // See find_module() → ensure_stdlib_path().
    });
}

/// Locate the vendored stdlib (`lib-python/3`) by walking up the running
/// executable's ancestor directories.
///
/// PyPy equivalent: initpath.py walks up from the executable to a
/// directory containing `lib-python/X.Y`.
#[cfg(feature = "host_env")]
fn find_intree_stdlib() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let mut dir = exe.parent();
    while let Some(d) = dir {
        let candidate = d.join("lib-python").join("3");
        if candidate.is_dir() {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

/// Resolve the stdlib directory to add to `sys.path`.
///
/// Order: the `PYRE_STDLIB` override, then the vendored `lib-python/3`
/// next to the executable, then a host `python3`'s stdlib as a last
/// resort. The vendored copy matches the `_sre` MAGIC pyre links; a host
/// stdlib only works when its `re`/`_sre` MAGIC agrees.
///
/// PyPy equivalent: initpath.py scans for lib-python/X.Y at startup.
#[cfg(feature = "host_env")]
fn detect_stdlib_path() -> Option<PathBuf> {
    // Explicit override.
    if let Ok(p) = host_os::var("PYRE_STDLIB") {
        let path = PathBuf::from(p);
        if path.is_dir() {
            return Some(path);
        }
    }
    // Vendored in-tree stdlib, located relative to the executable.
    if let Some(path) = find_intree_stdlib() {
        return Some(path);
    }
    // Last resort: borrow a host CPython's stdlib.
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
#[cfg(feature = "host_env")]
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
    // Consult the Python-visible sys.modules dict first so that user code
    // writing `sys.modules['foo'] = mod` is immediately visible to imports.
    // PyPy: importing.py check_sys_modules reads space.sys.get('modules').
    let key = pyre_object::w_str_new(name);
    let dict = SYS_MODULES_DICT.with(|d| d.get());
    if !dict.is_null() {
        if let Some(m) = unsafe { pyre_object::w_dict_lookup(dict, key) } {
            if !m.is_null() && !unsafe { pyre_object::is_none(m) } {
                return Some(m);
            }
        }
    }
    SYS_MODULES.with(|m| m.borrow().get(name).copied())
}

/// Look up a loaded module by name in `sys.modules` (Python-visible dict
/// first, then the interpreter cache). Mirrors `check_sys_modules`.
pub fn get_sys_module(name: &str) -> Option<PyObjectRef> {
    check_sys_modules(name)
}

/// The Python-visible `sys.modules` dict, or `PY_NULL` before it is
/// installed. Used by callers that need to iterate every loaded module
/// (e.g. pickle's `whichmodule` scan).
pub fn sys_modules_dict() -> PyObjectRef {
    SYS_MODULES_DICT.with(|d| d.get())
}

pub fn set_sys_module(name: &str, module: PyObjectRef) {
    SYS_MODULES.with(|m| {
        m.borrow_mut().insert(name.to_string(), module);
    });
    // Keep the Python-visible sys.modules dict in sync.
    SYS_MODULES_DICT.with(|d| {
        let dict = d.get();
        if !dict.is_null() {
            unsafe {
                pyre_object::w_dict_store(dict, pyre_object::w_str_new(name), module);
            }
        }
    });
}

/// GC root walk over every loaded module's dict storage.
///
/// Modules (`malloc_typed`) and their `W_ModuleDictObject`s are
/// Box-immortal, so the collector cannot reach a module dict's
/// authoritative `dstorage` / `object_storage` / cell registry
/// transitively (a Box-immortal object is never relocated and never
/// has its custom trace fired).  A movable value bound at module scope
/// — e.g. `gc.collect` reached through `gc.__dict__`, or any
/// module-level list / instance — would otherwise be read back stale
/// after a collection relocates it.  Treat each loaded module's dict as
/// a pinned root source so those slots stay forwarded.  This complements
/// the per-frame `w_globals_obj` walk in `eval::walk_pyframe_roots`,
/// which additionally covers `exec`/`eval` globals dicts that are not
/// registered in `sys.modules`.
///
/// # Safety
/// `visitor` must tolerate being called on every movable module-dict
/// value slot reachable here.
pub unsafe fn walk_module_dicts_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    SYS_MODULES.with(|m| {
        for &module in m.borrow().values() {
            if module.is_null() || !unsafe { pyre_object::is_module(module) } {
                continue;
            }
            unsafe {
                let w_dict = pyre_object::w_module_get_w_dict(module);
                pyre_object::dictmultiobject::w_module_dict_walk_gc_cells(w_dict, visitor);
            }
        }
    });
}

/// Forward the `sys.modules` dict pointer cached in `SYS_MODULES_DICT`.
///
/// The same dict object is also reachable as `sys.__dict__["modules"]`
/// (forwarded by [`walk_module_dicts_gc`]), but this fast-path cell holds an
/// independent raw copy.  `w_dict_new` allocates the dict in the movable
/// nursery, so a collection relocates it and leaves this cell pointing at the
/// vacated (reclaimed) slot; the next `check_sys_modules` would then run
/// `w_dict_lookup` against dead memory.  Forward the cell in place so it
/// tracks the relocation, mirroring the EC-slot forwarding in
/// `eval::walk_pyframe_roots`.
///
/// # Safety
/// `visitor` must tolerate a non-nursery or already-forwarded pointer.
pub unsafe fn walk_sys_modules_dict_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    SYS_MODULES_DICT.with(|d| {
        let mut dict = d.get();
        if dict.is_null() {
            return;
        }
        visitor(&mut dict);
        d.set(dict);
    });
}

/// Set the Python-visible sys.modules dict reference. Called during sys
/// module initialization so subsequent set_sys_module calls keep it in sync.
/// Also copies all previously cached modules into the dict.
/// Set sys.argv from a list of strings.
/// Must be called after the first `import sys` has run (e.g. after
/// `run_source` compiles the module-level code).
pub fn set_sys_argv(args: &[String]) {
    let items: Vec<pyre_object::PyObjectRef> =
        args.iter().map(|s| pyre_object::w_str_new(s)).collect();
    let argv = pyre_object::w_list_new(items);
    SYS_ARGV_PENDING.with(|p| p.set(argv));
}

thread_local! {
    static SYS_ARGV_PENDING: std::cell::Cell<pyre_object::PyObjectRef> =
        const { std::cell::Cell::new(pyre_object::PY_NULL) };
}

/// Called from sys module init to pick up any pending argv.
pub fn take_pending_sys_argv() -> pyre_object::PyObjectRef {
    SYS_ARGV_PENDING.with(|p| {
        let v = p.get();
        p.set(pyre_object::PY_NULL);
        v
    })
}

pub fn set_sys_modules_dict(dict: PyObjectRef) {
    SYS_MODULES_DICT.with(|d| d.set(dict));
    // Populate with all modules already in the cache.
    SYS_MODULES.with(|m| {
        for (name, &module) in m.borrow().iter() {
            unsafe {
                pyre_object::w_dict_store(dict, pyre_object::w_str_new(name), module);
            }
        }
    });
}

// ── find_module ──────────────────────────────────────────────────────
// PyPy equivalent: importing.py `find_module()`
// Searches sys.path for `<partname>.py` or `<partname>/__init__.py` (package).

#[derive(Debug)]
enum FindInfo {
    /// A .py source file was found.
    #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
    SourceFile { pathname: PathBuf },
    /// A package directory with __init__.py was found.
    #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
    Package { dirpath: PathBuf },
    /// A builtin (Rust-implemented) module was found.
    /// PyPy equivalent: C_BUILTIN modtype in find_module()
    Builtin,
}

#[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
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

#[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
fn find_module(partname: &str) -> Option<FindInfo> {
    let is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(partname));
    if is_builtin {
        return Some(FindInfo::Builtin);
    }
    None
}

/// Detect and add CPython stdlib to sys.path (once).
#[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
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

#[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
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
// PyPy equivalent: importing.py `exec_code_module(space, w_mod, code_w,
//                                  pathname, cpathname, write_paths=True)`
//
// Mirrors `pypy/module/imp/importing.py:269-300` line-by-line:
//   w_dict = space.getattr(w_mod, '__dict__')                       # ns
//   space.call_method(w_dict, 'setdefault',
//                     '__builtins__', space.builtin)
//   if write_paths:
//       space.setitem(w_dict, '__file__', w_pathname)
//       space.setitem(w_dict, '__cached__', w_cpathname)
//       _fix_up_module(d, name, pathname, cpathname)               # appexec
//   code_w.exec_code(space, w_dict, w_dict)
//
// `pathname` is `None` for callers that do not have a filesystem path
// (REPL `__main__`, builtin module bootstrap), matching PyPy's
// `write_paths=False` shape.  `cpathname` is `None` when no `.pyc` cache
// is available (pyre has no .pyc cache today, so all reachable callers
// pass `None` here — kept as a parameter so the signature mirrors PyPy
// instead of erasing the field).

fn exec_code_module(
    code: CodeObject,
    namespace: *mut DictStorage,
    execution_context: *const PyExecutionContext,
    pathname: Option<&str>,
    cpathname: Option<&str>,
) -> Result<PyObjectRef, crate::PyError> {
    // importing.py:272-274 — setdefault('__builtins__', space.builtin).
    // `fresh_dict_storage` already seeds `__builtins__` for module-shape
    // namespaces; the explicit setdefault here mirrors PyPy's defensive
    // call so callers that hand in a pre-built storage (future
    // `_imp.exec_dynamic`-style entry) still inherit the builtins
    // pointer with no surprises.
    {
        let ns = unsafe { &mut *namespace };
        if crate::dict_storage_get(ns, "__builtins__").is_none() {
            let ctx = unsafe { &*execution_context };
            let w_builtin = ctx.get_builtin();
            if !w_builtin.is_null() {
                crate::dict_storage_store(ns, "__builtins__", w_builtin);
            }
        }
    }
    // importing.py:275-298 write_paths block.  Pyre callers always pass
    // `Some(pathname)` for source-file imports and `None` for the
    // `write_paths=False` shape (REPL, builtin bootstrap).
    if let Some(p) = pathname {
        let ns = unsafe { &mut *namespace };
        // importing.py:284 setitem('__file__', w_pathname).
        let w_pathname = pyre_object::w_str_new(p);
        crate::dict_storage_store(ns, "__file__", w_pathname);
        // importing.py:285 setitem('__cached__', w_cpathname).  PyPy
        // surfaces `space.w_None` when `cpathname is None`, i.e. the
        // import was not satisfied from a `.pyc`.  Pyre has no .pyc
        // path today so reachable callers still hit the None arm.
        let w_cpathname = match cpathname {
            Some(c) => pyre_object::w_str_new(c),
            None => pyre_object::w_none(),
        };
        crate::dict_storage_store(ns, "__cached__", w_cpathname);
        // importing.py:286-298 — `_fix_up_module(d, name, pathname,
        // cpathname)`.  PyPy's `_fix_up_module`
        // (`lib-python/3/importlib/_bootstrap_external.py:1728`) sets
        // `__spec__`/`__loader__`/`__file__`/`__cached__` from the
        // app-level `SourceFileLoader` + `spec_from_file_location`
        // helpers.  Pyre lacks the importlib bootstrap machinery
        // (`SourceFileLoader`, `ModuleSpec`, `spec_from_file_location`
        // are not yet ported), so as a TODO we seed
        // `__loader__`/`__spec__` with `None` only when missing —
        // matching PyPy's `if not loader / if not spec` guards
        // (_bootstrap_external.py:1732, 1739).  When the importlib
        // app-level layer lands, the `None` arms will collapse onto the
        // mechanical PyPy port.
        if crate::dict_storage_get(ns, "__loader__").is_none() {
            crate::dict_storage_store(ns, "__loader__", pyre_object::w_none());
        }
        if crate::dict_storage_get(ns, "__spec__").is_none() {
            crate::dict_storage_store(ns, "__spec__", pyre_object::w_none());
        }
    }
    let code_ptr = Box::into_raw(Box::new(code));
    let w_code = crate::w_code_new(code_ptr as *const ());
    // importing.py:300 code_w.exec_code(space, w_dict, w_dict) → eval.py:31-33
    // Code.exec_code → space.createframe(...) + frame.run().  Surface
    // initialize_frame_scopes' freevar/closure mismatch (TypeError /
    // ValueError per pyframe.py:242-253) as PyError so the importer
    // reports it instead of panicking.  Route through run() so the
    // GENERATOR / COROUTINE / ASYNC_GENERATOR dispatch in
    // pyframe.py:268-273 holds for the import path too.
    let mut frame = crate::createframe(w_code as *const (), namespace, execution_context, None)?;
    frame.run()
}

// ── appleveldef_install ──────────────────────────────────────────────
// PyPy equivalent: `pypy/interpreter/mixedmodule.py:135 MixedModule.get`
// resolves an `appleveldefs` entry by lazily executing the sibling
// `app_*.py` file into a per-mixedmodule namespace and reading the
// named attribute.  Pyre's macro form bundles all entries from one app
// file into a single install call; the source is included at
// compile time via `include_str!` so no filesystem read happens at
// module-init time.

/// Execute `source` (a Python module) into a fresh namespace and copy
/// each binding in `names` into the caller's module dict `ns`.
///
/// `filename` is used as the source path for tracebacks / co_filename
/// only.  The intermediate namespace is intentionally leaked: every
/// function defined in `source` retains it as its `__globals__`, so the
/// box must outlive the bound names — which, for module-init artifacts,
/// is "forever".
pub fn appleveldef_install(ns: &mut DictStorage, source: &str, filename: &str, names: &[&str]) {
    let code = compile_source_with_filename(source, Mode::Exec, filename)
        .unwrap_or_else(|e| panic!("appleveldef `{filename}`: compile failed — {e}"));
    let ctx = crate::call::getexecutioncontext();
    if ctx.is_null() {
        panic!("appleveldef `{filename}`: no execution context at module init");
    }
    let mut app_ns = Box::new(unsafe { (*ctx).fresh_dict_storage() });
    app_ns.fix_ptr();
    let app_ns_ptr: *mut DictStorage = Box::leak(app_ns);
    let code_ptr = Box::into_raw(Box::new(code));
    let w_code = crate::w_code_new(code_ptr as *const ());
    let mut frame = crate::createframe(w_code as *const (), app_ns_ptr, ctx, None)
        .unwrap_or_else(|e| panic!("appleveldef `{filename}`: createframe — {e:?}"));
    if let Err(e) = frame.run() {
        panic!("appleveldef `{filename}`: exec — {e:?}");
    }
    let app_ns_ref = unsafe { &*app_ns_ptr };
    for &name in names {
        match crate::dict_storage_get(app_ns_ref, name) {
            Some(val) => crate::dict_storage_store(ns, name, val),
            None => panic!("appleveldef `{filename}`: name `{name}` not bound by source"),
        }
    }
}

// ── load_source_module ───────────────────────────────────────────────
// PyPy equivalent: importing.py `load_source_module()`
//
// Parse + execute a .py source file, producing a module object.

#[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
fn load_source_module(
    modulename: &str,
    pathname: &Path,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    let source = host_fs::read_to_string(pathname).map_err(|e| {
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
    let mut namespace = Box::new(ctx.fresh_dict_storage());
    namespace.fix_ptr();

    // PyPy `interpreter/module.py:Module.__init__` seeds `__name__` on
    // the module's w_dict.  `w_module_new(modulename, ns_ptr)` below
    // does that via `w_dict_setitem_str("__name__", ...)` which the
    // storage proxy mirrors back into `namespace`, so an explicit
    // dict_storage_store here would be redundant.
    //
    // `__file__`/`__cached__` setting moved into `exec_code_module`
    // (`importing.py:284-285`) so the per-module attribute seeding
    // mirrors the PyPy call order.
    //
    // `__package__` is set by PyPy `interp_import._prepare_module`
    // (`pypy/module/imp/interp_import.py`); pyre has no `_prepare_module`
    // yet, so we still seed it here as a TODO until
    // the prepare-module path is ported.
    let pkg = if let Some(dot) = modulename.rfind('.') {
        &modulename[..dot]
    } else {
        modulename
    };
    crate::dict_storage_store(&mut namespace, "__package__", pyre_object::w_str_new(pkg));

    let ns_ptr = Box::into_raw(namespace);

    // Create the module object BEFORE execution and register in sys.modules.
    // PyPy: load_source_module → set_sys_modules BEFORE exec_code_module.
    // This prevents infinite recursion on circular imports.
    //
    // `dict_storage_to_dict(ns_ptr)` now constructs a W_ModuleDictObject
    // (PyPy `dictmultiobject.py:60-69 allocate_and_init_instance(
    // module=True)` shape) with `dict_storage_proxy = ns_ptr` and
    // registers it as `DictStorage.mirror_target`, so `module.w_dict`,
    // `function.__globals__`, and `globals()` all converge on the same
    // W_ModuleDictObject identity.  Forward writes via the module dict
    // fan out to the DictStorage; back-mirror updates the strategy
    // storage in step — the frame-side `*mut DictStorage` carrier
    // stays valid until `PyFrame.w_globals` migrates to
    // `PyObjectRef`.  The simpler builtin module loader path (no
    // frame globals dependency) already uses `W_ModuleDictObject`.
    let canonical = crate::baseobjspace::dict_storage_to_dict(ns_ptr);
    let module = pyre_object::w_module_new_aliasing_dict(modulename, ns_ptr as *mut u8, canonical);
    set_sys_module(modulename, module);

    // PyPy `importing.py:300` passes `pathname`/`cpathname` to
    // `exec_code_module`; pyre has no .pyc cache today so cpathname is
    // always None, matching the PyPy `cpathname is None` arm at line
    // 282-283.
    exec_code_module(code, ns_ptr, execution_context, Some(&pathname_str), None)?;

    // Module-level code may have rewritten `sys.modules[name]` (the
    // `decimal` → `_pydecimal` pattern, or PyPy's `_cffi_backend` style
    // late rewiring). Honour that — PyPy: interp_import.importhook
    // reads sys.modules again after exec_code_module via importcache.
    if let Some(replaced) = check_sys_modules(modulename) {
        if !std::ptr::eq(replaced, module) {
            return Ok(replaced);
        }
    }

    Ok(module)
}

// ── load_package ─────────────────────────────────────────────────────
// PyPy equivalent: load_module with PKG_DIRECTORY modtype

#[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
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

    // Set __path__ and __package__ on the module namespace via
    // `module.w_dict` so storage-backed and dict-subclass-backed Modules
    // both observe the writes (`pypy/module/__builtin__/moduledef.py:102-103
    // Module(space, None, w_builtin)`).  When the dict is storage-backed
    // the proxy store hook propagates the entry into the underlying
    // DictStorage; when it's a subclass instance the write lands in the
    // entries Vec where the subclass's `__init__` placed any seeded keys.
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    let path_str = pyre_object::w_str_new(&dirpath.to_string_lossy());
    let path_list = pyre_object::w_list_new(vec![path_str]);
    unsafe {
        if !w_dict.is_null() && pyre_object::is_dict(w_dict) {
            pyre_object::dictmultiobject::w_dict_setitem_str(w_dict, "__path__", path_list);
            pyre_object::dictmultiobject::w_dict_setitem_str(
                w_dict,
                "__package__",
                pyre_object::w_str_new(modulename),
            );
        }
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

    // Try a full-name builtin match first so dotted stubs like
    // `importlib.machinery` can override the filesystem search.
    // PyPy: interp_import.importhook consults sys.builtin_module_names by
    // the fully-qualified name.
    let full_is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(modulename));
    if full_is_builtin {
        // `pypy/interpreter/module.py:18 Module.__init__` keeps a single
        // `Module` per imported module name; `space.builtin` IS the
        // module returned by `import builtins`.  Pyre's
        // `ExecutionContext::get_builtin()` lazily caches the Module
        // wrapping `self.builtins_module` — route the "builtins" case
        // through it so identity equality holds against `space.builtin`.
        let m = if modulename == "builtins" && !execution_context.is_null() {
            unsafe { (*execution_context).get_builtin() }
        } else {
            load_builtin_module(modulename).ok_or_else(|| {
                crate::PyError::new(
                    crate::PyErrorKind::ImportError,
                    format!("builtin module '{modulename}' failed to initialize"),
                )
            })?
        };
        set_sys_module(modulename, m);
        return Ok(Some(m));
    }

    // Find the module on disk
    let find_info = find_module(partname);
    let Some(info) = find_info else {
        return Ok(None);
    };

    let module = match info {
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        FindInfo::SourceFile { pathname } => {
            match load_source_module(modulename, &pathname, execution_context) {
                Ok(m) => m,
                Err(e) => {
                    return Err(e);
                }
            }
        }
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        FindInfo::Package { dirpath } => load_package(modulename, &dirpath, execution_context)?,
        FindInfo::Builtin => {
            // Same builtins-identity path as the full_is_builtin branch
            // above: route `import builtins` through `EC.get_builtin()`
            // so `import builtins is space.builtin` holds.
            let m = if partname == "builtins" && !execution_context.is_null() {
                unsafe { (*execution_context).get_builtin() }
            } else {
                load_builtin_module(partname).ok_or_else(|| {
                    crate::PyError::new(
                        crate::PyErrorKind::ImportError,
                        format!("builtin module '{modulename}' failed to initialize"),
                    )
                })?
            };
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
    let mut parent: Option<PyObjectRef> = None;
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
        // _bootstrap._find_and_load (_bootstrap.py:1346-1352): bind the
        // submodule as an attribute of its parent package so `import a.b`
        // makes `a.b` reachable. Only an AttributeError is swallowed (with an
        // ImportWarning); any other exception propagates.
        if let Some(parent_mod) = parent {
            if let Err(err) = crate::setattr_str(parent_mod, part, module) {
                if err.kind != crate::PyErrorKind::AttributeError {
                    return Err(err);
                }
                let parent_name = parts[..level].join(".");
                crate::warn::warn(
                    &format!(
                        "Cannot set an attribute on '{parent_name}' for child module '{part}'"
                    ),
                    "ImportWarning",
                );
            }
        }
        if level == 0 {
            first = Some(module);
        }
        parent = Some(module);
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
    let package = resolve_package_name(w_globals)?.ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::ImportError,
            "attempted relative import with no known parent package",
        )
    })?;

    // Strip (level - 1) trailing components from package
    // PyPy: for dotted name "a.b.c" with level=2, strip "c" → "a.b", then strip "b" → "a"
    let mut parts: Vec<&str> = package.split('.').collect();
    let strips = (level - 1) as usize;
    if strips >= parts.len() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ImportError,
            format!(
                "attempted relative import beyond top-level package (package='{package}', level={level})"
            ),
        ));
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
fn resolve_package_name(w_globals: PyObjectRef) -> Result<Option<String>, crate::PyError> {
    if w_globals.is_null() {
        return Ok(None);
    }

    // `space.finditem_str` (baseobjspace.py:870-878) maps only KeyError to a
    // missing entry; any other `__getitem__` error (a dict-subclass globals
    // raising) must propagate.  `?` re-raises it; `if let Some(..)` consumes
    // the present case.
    // Try __package__ first (PyPy: space.finditem_str(w_globals, '__package__'))
    if let Some(pkg) = crate::baseobjspace::finditem_str(w_globals, "__package__")? {
        if !pkg.is_null() && unsafe { pyre_object::is_str(pkg) } {
            let s = unsafe { pyre_object::w_str_get_value(pkg) };
            if !s.is_empty() {
                return Ok(Some(s.to_string()));
            }
        }
    }

    // Fallback: __name__ (for modules inside packages)
    if let Some(name_obj) = crate::baseobjspace::finditem_str(w_globals, "__name__")? {
        if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
            let name = unsafe { pyre_object::w_str_get_value(name_obj) };
            // If the module has a __path__, it's a package — use __name__ as-is
            if crate::baseobjspace::finditem_str(w_globals, "__path__")?.is_some() {
                return Ok(Some(name.to_string()));
            }
            // Otherwise strip the last component (module name within package)
            if let Some(dot) = name.rfind('.') {
                return Ok(Some(name[..dot].to_string()));
            }
        }
    }

    Ok(None)
}

// ── import_from ──────────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py `IMPORT_FROM`
//
// Get an attribute from the module on TOS. Like `space.getattr(w_module, w_name)`.

pub fn import_from(
    module: PyObjectRef,
    name: &str,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // First try the module's namespace dict (PyPy: space.getattr → w_dict lookup).
    // Routed through `w_module.w_dict` so dict-subclass-backed Modules
    // (`pypy/module/__builtin__/moduledef.py:102-103`) honour their
    // `__getitem__` overrides via the same lookup path.
    if unsafe { is_module(module) } {
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        if !w_dict.is_null() && unsafe { pyre_object::is_dict(w_dict) } {
            if let Some(value) = unsafe { pyre_object::w_dict_getitem_str(w_dict, name) } {
                return Ok(value);
            }
        }
    }

    // Fallback: try getattr (for non-module objects or attrs set via setattr)
    if let Ok(value) = crate::baseobjspace::getattr_str(module, name) {
        return Ok(value);
    }

    // PyPy: pyopcode.py _import_from — try importing as a submodule.
    // Build fullname = module.__name__ + "." + name and import it.
    // Same `w_dict` routing as the first lookup so dict-subclass-backed
    // Modules' submodule fallback honours overridden `__getitem__`.
    if unsafe { is_module(module) } {
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        if !w_dict.is_null() && unsafe { pyre_object::is_dict(w_dict) } {
            if let Some(modname_obj) =
                unsafe { pyre_object::w_dict_getitem_str(w_dict, "__name__") }
            {
                if !modname_obj.is_null() && unsafe { pyre_object::is_str(modname_obj) } {
                    let modname = unsafe { pyre_object::w_str_get_value(modname_obj) };
                    let fullname = format!("{modname}.{name}");
                    if importhook(
                        &fullname,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0,
                        execution_context,
                    )
                    .is_ok()
                    {
                        // importhook returns the top-level module when
                        // fromlist is empty. Retrieve the actual leaf
                        // module from sys.modules.
                        if let Some(submod) = check_sys_modules(&fullname) {
                            unsafe {
                                pyre_object::dictmultiobject::w_dict_setitem_str(
                                    w_dict, name, submod,
                                );
                            }
                            return Ok(submod);
                        }
                    }
                }
            }
        }
    }

    Err(crate::PyError::new(
        crate::PyErrorKind::ImportError,
        format!("cannot import name '{name}'"),
    ))
}

// ── import_all_from ──────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py:2221-2258 `import_all_from(module,
// into_locals)` (applevel function called by IMPORT_STAR).

fn type_name_for_err(w_obj: PyObjectRef) -> String {
    unsafe {
        match crate::typedef::r#type(w_obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => (*(*w_obj).ob_type).name.to_string(),
        }
    }
}

/// pypy/interpreter/pyopcode.py:2221-2258 `import_all_from` — applevel
/// driver.  Iterates `for name in all:` lazily via `space.iter` /
/// `space.next`, applies the per-name str check + leading-underscore
/// filter, and invokes `write` once per accepted name.  Used by the
/// `*mut DictStorage` and generic-mapping wrappers below.
///
/// ```python
/// try:
///     all = module.__all__
/// except AttributeError:
///     try:
///         dict = module.__dict__
///     except AttributeError:
///         raise ImportError("from-import-* object has no __dict__ "
///                           "and no __all__")
///     all = dict.keys()
///     skip_leading_underscores = True
/// else:
///     skip_leading_underscores = False
///
/// module_name = module.__name__
/// if not isinstance(module_name, str):
///     raise TypeError("module __name__ must be a string, not %s",
///                     type(module_name).__name__)
///
/// for name in all:
///     if not isinstance(name, str):
///         ...  # raise TypeError ("Item in <m>.__all__ ..." or
///              #                  "Key in <m>.__dict__ ...")
///     if skip_leading_underscores and name and name[0] == '_':
///         continue
///     into_locals[name] = getattr(module, name)
/// ```
fn import_all_from_each<F>(module: PyObjectRef, mut write: F) -> Result<(), crate::PyError>
where
    F: FnMut(&str, PyObjectRef) -> Result<(), crate::PyError>,
{
    let (w_iterable, skip_leading_underscores) =
        match crate::baseobjspace::getattr_str(module, "__all__") {
            Ok(w_all) => (w_all, false),
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
                // pyopcode.py:2225-2230 — `dict = module.__dict__; all = dict.keys()`.
                // `space.getattr(module, '__dict__')` so any object exposing
                // `__dict__` (Module, class, instance with `__dict__`,
                // bytes-keyed proxies, ...) participates.
                match crate::baseobjspace::getattr_str(module, "__dict__") {
                    Ok(w_dict) => {
                        let w_keys_method = crate::baseobjspace::getattr_str(w_dict, "keys")?;
                        // pyopcode.py:2230 `all = dict.keys()` — pyre's
                        // `call_function` stashes errors as PY_NULL; use
                        // `call_and_check` so a misbehaving `keys()` (or
                        // `__getattr__`-installed override) raises here
                        // rather than handing a bogus iterable to
                        // `space.iter` below.
                        let w_keys = crate::builtins::call_and_check(w_keys_method, &[])?;
                        (w_keys, true)
                    }
                    Err(e2) if e2.kind == crate::PyErrorKind::AttributeError => {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::ImportError,
                            "from-import-* object has no __dict__ and no __all__".to_string(),
                        ));
                    }
                    Err(e2) => return Err(e2),
                }
            }
            Err(e) => return Err(e),
        };

    // pyopcode.py:2235-2237 — `module_name = module.__name__` with str check.
    let module_name_w = crate::baseobjspace::getattr_str(module, "__name__")?;
    if !unsafe { is_str(module_name_w) } {
        return Err(crate::PyError::type_error(format!(
            "module __name__ must be a string, not {}",
            type_name_for_err(module_name_w),
        )));
    }
    let module_name = unsafe { pyre_object::w_str_get_value(module_name_w) }.to_string();

    // pyopcode.py:2239 — `for name in all:` lazy iteration.
    let w_iter = crate::baseobjspace::iter(w_iterable)?;
    loop {
        let w_name = match crate::baseobjspace::next(w_iter) {
            Ok(v) => v,
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        };
        // pyopcode.py:2240-2255 — per-name str check.
        if !unsafe { is_str(w_name) } {
            let (container, accessor) = if skip_leading_underscores {
                ("__dict__", "Key")
            } else {
                ("__all__", "Item")
            };
            return Err(crate::PyError::type_error(format!(
                "{accessor} in {module_name}.{container} must be str, not {}",
                type_name_for_err(w_name),
            )));
        }
        let name = unsafe { pyre_object::w_str_get_value(w_name) }.to_string();
        // pyopcode.py:2256-2257 — leading-underscore filter (only for
        // the `__dict__.keys()` fallback).
        if skip_leading_underscores && name.starts_with('_') {
            continue;
        }
        // pyopcode.py:2258 — `into_locals[name] = getattr(module, name)`.
        let value = crate::baseobjspace::getattr_str(module, &name)?;
        write(&name, value)?;
    }
    Ok(())
}

/// pypy/interpreter/pyopcode.py:2221-2258 `import_all_from` —
/// `*mut DictStorage` (dict locals fast path) target variant.
pub fn import_all_from(
    module: PyObjectRef,
    into_namespace: *mut DictStorage,
) -> Result<(), crate::PyError> {
    let dst_ns = unsafe { &mut *into_namespace };
    import_all_from_each(module, |name, value| {
        dict_storage_store(dst_ns, name, value);
        Ok(())
    })
}

/// pypy/interpreter/pyopcode.py:2221-2258 `import_all_from` — generic
/// mapping (`PyObjectRef`) target variant.  Errors from `__setitem__`
/// propagate (CPython behaviour: a misbehaving mapping surfaces its
/// TypeError / KeyError to the caller).
pub fn import_all_from_w(
    module: PyObjectRef,
    into_locals: PyObjectRef,
) -> Result<(), crate::PyError> {
    import_all_from_each(module, |name, value| {
        crate::baseobjspace::setitem(into_locals, unsafe { pyre_object::w_str_new(name) }, value)?;
        Ok(())
    })
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
