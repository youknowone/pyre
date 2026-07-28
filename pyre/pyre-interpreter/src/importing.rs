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
use std::path::PathBuf;
use std::sync::{
    LazyLock, Mutex,
    atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
};
// `Path` is used only by the host_env source/package loaders; keep it gated
// so an host_env-off build does not warn on an unused import. `PathBuf`
// appears in the host_env-independent module-search surface
// (`SYS_PATH`, `find_module`, `parent_package_path`, `load_part`) and must
// stay in scope unconditionally.
#[cfg(feature = "host_env")]
use std::path::Path;

use crate::PyExecutionContext;
use crate::{CodeObject, Mode, PyFrame, compile_source_with_filename};
use pyre_object::*;
use rustpython_wtf8::Wtf8Buf;

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
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
use host::fs as host_fs;
use host::os as host_os;

// ── SourceProvider: the host-agnostic byte source for module loading ──
// PyPy/CPython read module source from a filesystem.  pyre routes the three
// FS touchpoints the import machinery actually exercises — the package/module
// `is_file` probes in `find_in_dirs` and the `read_to_string` in
// `load_source_module` — through one object, so the SAME import resolution
// runs over a real kernel FS (native, and the wasmtime runner via host
// imports) or an in-memory VFS (the browser/web build, populated from an
// embedded stdlib bundle).  The import machinery never branches per host;
// only the installed provider differs.
#[cfg(feature = "host_env")]
pub trait SourceProvider {
    /// True when `path` names a readable regular file.
    fn is_file(&self, path: &Path) -> bool;
    /// True when `path` names a directory.
    fn is_dir(&self, path: &Path) -> bool;
    /// Read the whole file at `path` as UTF-8 source.
    fn read_to_string(&self, path: &Path) -> std::io::Result<String>;
}

#[cfg(feature = "host_env")]
thread_local! {
    static SOURCE_PROVIDER: RefCell<Option<std::rc::Rc<dyn SourceProvider>>> =
        const { RefCell::new(None) };
}

/// Install the byte source the import machinery reads through.  The wasm
/// bootstrap installs a host-import-backed or in-memory-VFS provider before
/// the first import; native/pyrex leaves it unset and the default kernel-FS
/// provider answers every probe.
#[cfg(feature = "host_env")]
pub fn install_source_provider(provider: std::rc::Rc<dyn SourceProvider>) {
    SOURCE_PROVIDER.with(|p| *p.borrow_mut() = Some(provider));
}

/// Run `f` against the installed provider, lazily defaulting to the platform's
/// kernel-FS provider when none was installed.  The `Rc` is cloned out before
/// `f` runs so the thread-local borrow is not held across the call (the import
/// path is re-entrant).
#[cfg(feature = "host_env")]
fn with_source_provider<R>(f: impl FnOnce(&dyn SourceProvider) -> R) -> R {
    let provider = SOURCE_PROVIDER.with(|p| {
        let mut slot = p.borrow_mut();
        if slot.is_none() {
            *slot = Some(default_source_provider());
        }
        slot.clone().unwrap()
    });
    f(&*provider)
}

/// Read a source file through the installed [`SourceProvider`] — the
/// seam-mediated VFS under sandbox, the host FS otherwise. Traceback rendering
/// uses this so it honours the same jail as the import machinery instead of
/// reaching `std::fs` for a guest-controlled path.
#[cfg(feature = "host_env")]
pub fn read_source_to_string(path: &Path) -> std::io::Result<String> {
    with_source_provider(|p| p.read_to_string(path))
}

#[cfg(all(
    feature = "host_env",
    not(target_arch = "wasm32"),
    not(feature = "sandbox")
))]
fn default_source_provider() -> std::rc::Rc<dyn SourceProvider> {
    std::rc::Rc::new(HostFsProvider)
}

#[cfg(all(feature = "host_env", not(target_arch = "wasm32"), feature = "sandbox"))]
fn default_source_provider() -> std::rc::Rc<dyn SourceProvider> {
    std::rc::Rc::new(SeamSourceProvider)
}

#[cfg(all(feature = "host_env", target_arch = "wasm32"))]
fn default_source_provider() -> std::rc::Rc<dyn SourceProvider> {
    std::rc::Rc::new(NullSourceProvider)
}

/// Kernel-filesystem provider — the default on native and the wasmtime
/// runner's real-FS path.  `is_file`/`is_dir` go straight to `std::fs::
/// metadata` via the `Path` methods (matching the historical `find_in_dirs`
/// probes); reads route through the host_env `fs` shim.
#[cfg(all(
    feature = "host_env",
    not(target_arch = "wasm32"),
    not(feature = "sandbox")
))]
struct HostFsProvider;

#[cfg(all(
    feature = "host_env",
    not(target_arch = "wasm32"),
    not(feature = "sandbox")
))]
impl SourceProvider for HostFsProvider {
    fn is_file(&self, path: &Path) -> bool {
        path.is_file()
    }
    fn is_dir(&self, path: &Path) -> bool {
        path.is_dir()
    }
    fn read_to_string(&self, path: &Path) -> std::io::Result<String> {
        host_fs::read_to_string(path)
    }
}

/// Sandbox provider: every import probe and source read round-trips through the
/// host_seam trampoline to the trusted controller, which enforces the virtual
/// filesystem policy (read-only, path-jailed). Replaces `HostFsProvider` so the
/// import machinery cannot `std::fs` its way to an attacker-chosen host path
/// (e.g. `sys.path.append('/etc'); __import__('shadow')`).
#[cfg(all(feature = "host_env", feature = "sandbox"))]
struct SeamSourceProvider;

#[cfg(all(feature = "host_env", feature = "sandbox"))]
impl SeamSourceProvider {
    fn stat_mode(path: &Path) -> Option<u32> {
        use std::os::unix::ffi::OsStrExt;
        crate::host_seam::ops::stat(path.as_os_str().as_bytes())
            .ok()
            .map(|s| s.mode)
    }
}

#[cfg(all(feature = "host_env", feature = "sandbox"))]
impl SourceProvider for SeamSourceProvider {
    fn is_file(&self, path: &Path) -> bool {
        Self::stat_mode(path).is_some_and(|m| m & libc::S_IFMT as u32 == libc::S_IFREG as u32)
    }
    fn is_dir(&self, path: &Path) -> bool {
        Self::stat_mode(path).is_some_and(|m| m & libc::S_IFMT as u32 == libc::S_IFDIR as u32)
    }
    fn read_to_string(&self, path: &Path) -> std::io::Result<String> {
        use std::os::unix::ffi::OsStrExt;
        fn to_io(e: crate::host_seam::SeamError) -> std::io::Error {
            match e {
                crate::host_seam::SeamError::Os(errno) => std::io::Error::from_raw_os_error(errno),
                _ => std::io::Error::other("sandbox source read failed"),
            }
        }
        let bytes = path.as_os_str().as_bytes();
        let fd = crate::host_seam::ops::open(bytes, libc::O_RDONLY, 0).map_err(to_io)?;
        let mut data = Vec::new();
        loop {
            match crate::host_seam::ops::read(fd, 65536) {
                Ok(chunk) if chunk.is_empty() => break,
                Ok(chunk) => data.extend_from_slice(&chunk),
                Err(e) => {
                    let _ = crate::host_seam::ops::close(fd);
                    return Err(to_io(e));
                }
            }
        }
        let _ = crate::host_seam::ops::close(fd);
        String::from_utf8(data)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "source not utf-8"))
    }
}

/// Default provider on wasm before the bootstrap installs a real one: resolves
/// nothing, preserving the historical "builtins only" behaviour.
#[cfg(all(feature = "host_env", target_arch = "wasm32"))]
struct NullSourceProvider;

#[cfg(all(feature = "host_env", target_arch = "wasm32"))]
impl SourceProvider for NullSourceProvider {
    fn is_file(&self, _path: &Path) -> bool {
        false
    }
    fn is_dir(&self, _path: &Path) -> bool {
        false
    }
    fn read_to_string(&self, path: &Path) -> std::io::Result<String> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no source provider installed: {}", path.display()),
        ))
    }
}

// ── embedded-stdlib VFS (wasm_vfs) ───────────────────────────────────
// The browser/web wasm target has no filesystem, so the pure-Python stdlib
// closure that `import re` needs is compiled into the binary (see build.rs)
// and served from this in-memory map.  Keys are `mount.join(<relpath>)`, so the
// SAME `find_in_dirs` probes (`<dir>/re/__init__.py`, `<dir>/enum.py`, …) that
// hit a real FS on native resolve here once `mount` is on sys.path.
#[cfg(feature = "wasm_vfs")]
pub static VFS_BLOB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stdlib_vfs.lz4"));

#[cfg(feature = "wasm_vfs")]
enum VfsEntry {
    File(std::rc::Rc<str>),
    Dir,
}

#[cfg(feature = "wasm_vfs")]
struct VfsProvider {
    map: HashMap<PathBuf, VfsEntry>,
}

#[cfg(feature = "wasm_vfs")]
impl VfsProvider {
    /// Decompress and parse the build-time blob into a `mount`-rooted map.
    /// Each embedded file becomes a `File` entry at `mount/<relpath>`, plus a
    /// synthetic `Dir` entry for every ancestor directory (so `is_dir` answers
    /// for `re/`, `collections/`, and the mount itself).
    fn from_blob(blob: &[u8], mount: &Path) -> Self {
        let raw = lz4_flex::block::decompress_size_prepended(blob)
            .expect("wasm_vfs: corrupt embedded stdlib blob");
        let mut map: HashMap<PathBuf, VfsEntry> = HashMap::new();
        map.insert(mount.to_path_buf(), VfsEntry::Dir);

        let mut pos = 0usize;
        let read_u32 = |raw: &[u8], pos: &mut usize| -> usize {
            let n = u32::from_le_bytes(raw[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;
            n
        };
        let count = read_u32(&raw, &mut pos);
        for _ in 0..count {
            let name_len = read_u32(&raw, &mut pos);
            let name = std::str::from_utf8(&raw[pos..pos + name_len])
                .expect("wasm_vfs: non-utf8 module name")
                .to_owned();
            pos += name_len;
            let src_len = read_u32(&raw, &mut pos);
            let src = std::str::from_utf8(&raw[pos..pos + src_len])
                .expect("wasm_vfs: non-utf8 module source")
                .to_owned();
            pos += src_len;

            let full = mount.join(&name);
            // Register every ancestor directory under `mount` as a Dir.
            let mut ancestor = full.parent();
            while let Some(dir) = ancestor {
                if dir == mount || !dir.starts_with(mount) {
                    break;
                }
                map.entry(dir.to_path_buf()).or_insert(VfsEntry::Dir);
                ancestor = dir.parent();
            }
            map.insert(full, VfsEntry::File(std::rc::Rc::from(src.as_str())));
        }
        VfsProvider { map }
    }
}

#[cfg(feature = "wasm_vfs")]
impl SourceProvider for VfsProvider {
    fn is_file(&self, path: &Path) -> bool {
        matches!(self.map.get(path), Some(VfsEntry::File(_)))
    }
    fn is_dir(&self, path: &Path) -> bool {
        matches!(self.map.get(path), Some(VfsEntry::Dir))
    }
    fn read_to_string(&self, path: &Path) -> std::io::Result<String> {
        match self.map.get(path) {
            Some(VfsEntry::File(src)) => Ok(src.to_string()),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("not in embedded stdlib: {}", path.display()),
            )),
        }
    }
}

/// Mount the embedded stdlib closure at `mount`, add `mount` to `sys.path`, and
/// install the VFS as the import source.  Called by the web wasm bootstrap
/// before the first import.
#[cfg(feature = "wasm_vfs")]
pub fn mount_embedded_stdlib(mount: &Path) {
    let provider = VfsProvider::from_blob(VFS_BLOB, mount);
    add_sys_path(mount);
    install_source_provider(std::rc::Rc::new(provider));
}

// ── sys.modules cache ────────────────────────────────────────────────
// PyPy equivalent: `space.sys.get('modules')`, `space.sys.path`, and
// `space.builtin_modules` are object-space/process state, shared by every
// ExecutionContext.  Raw GC references use the established process-global
// `usize` representation; the mutex serializes semantic access and keeps
// foreign STW root walks well-defined.
static SYS_MODULES: LazyLock<Mutex<HashMap<String, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static SYS_MODULES_DICT: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "host_env")]
static SYS_PATH: LazyLock<Mutex<Vec<PathBuf>>> = LazyLock::new(|| Mutex::new(Vec::new()));
/// The directory prepended to `sys.path` at startup (`config->sys_path_0`):
/// the script's directory, or the cwd for `-c` / `-m`.  Captured once by
/// `init_sys_path`; read by the shadowing check, which must not see later
/// `sys.path` mutations.  PyPy records this on interpreter/import state, not
/// on an OS thread: import shadowing decisions made by free-threaded workers
/// must observe the startup path captured by the launcher.
static SYS_PATH_0: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));
/// The literal `sys.path[0]` entry staged by `init_sys_path` and prepended
/// by `add_sys_path_0` once `site` has run (`pymain_sys_path_add_path0`):
/// `""` for `-c` / stdin / the REPL, the cwd for `-m`, the script's
/// directory for a script.  `None` under `-P`, and taken on first insert so
/// the `-i` REPL-after-script path does not prepend it twice.  Process-global
/// for the same reason as `SYS_PATH_0`: the launcher stages it and whichever
/// thread runs the insert must observe that staging.
static SYS_PATH_0_PENDING: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));
pub(crate) static BUILTIN_MODULES: LazyLock<Mutex<HashMap<&'static str, BuiltinModuleDef>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

thread_local! {
    static IMPORT_ROOT_AREA: ImportRootArea = ImportRootArea {
        argv_pending: SYS_ARGV_PENDING.with(|p| p as *const _),
    };
}

#[derive(Clone, Copy)]
pub(crate) struct BuiltinModuleDef {
    init: fn(PyObjectRef),
    startup: Option<fn(PyObjectRef, *const PyExecutionContext) -> Result<(), crate::PyError>>,
}

struct ImportRootArea {
    /// The pending `sys.argv` list is reachable only from this cell between
    /// `set_sys_argv` and `take_pending_sys_argv`.
    argv_pending: *const std::cell::Cell<PyObjectRef>,
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
pub fn register_builtin_module(name: &'static str, init: fn(PyObjectRef)) {
    BUILTIN_MODULES.lock().unwrap().insert(
        name,
        BuiltinModuleDef {
            init,
            startup: None,
        },
    );
}

/// Register a MixedModule initializer plus its `Module.startup` hook.
/// The hook runs after the new module is visible in `sys.modules`, matching
/// `getbuiltinmodule()` and allowing startup imports without a module cycle.
pub fn register_builtin_module_with_startup(
    name: &'static str,
    init: fn(PyObjectRef),
    startup: fn(PyObjectRef, *const PyExecutionContext) -> Result<(), crate::PyError>,
) {
    BUILTIN_MODULES.lock().unwrap().insert(
        name,
        BuiltinModuleDef {
            init,
            startup: Some(startup),
        },
    );
}

/// The registered builtin module names, for `sys.builtin_module_names`.
///
/// PyPy equivalent: `pypy/module/sys/state.py get_builtin_module_names`,
/// which likewise reads `space.builtin_modules` rather than a second list.
/// Reading the registry keeps the advertised set equal to the set `import`
/// can satisfy under every `cfg` combination, which a parallel list cannot:
/// the modules gated on `unix` / `wasm32` / `sandbox` differ per build.
/// Dotted keys (`importlib.machinery`, `__pypy__.builders`) are registry
/// entries for submodules, not top-level modules, so they are left out.
pub fn builtin_module_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = BUILTIN_MODULES
        .lock()
        .unwrap()
        .keys()
        .copied()
        .filter(|name| !name.contains('.'))
        .collect();
    names.sort_unstable();
    names
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
    // `moduledef.py:5 applevel_name = '_operator'` — the interp-level table
    // is reachable only as `_operator`; `import operator` resolves to
    // `operator.py`, whose `from _operator import *` drops the underscore
    // names the table also carries.
    pyre_install_module!("_operator"(operator));
    pyre_install_module!("builtins"(__builtin__));
    pyre_install_module!(_io);
    pyre_install_module!(_sre);

    // C-extension stubs required for stdlib import chains
    // (PyPy: pypy/module/* mixed modules).
    pyre_install_module!(_weakref);
    pyre_install_module!(_warnings);
    // `sys.platform == "win32"` sends shutil (and so tempfile) through the
    // `_winapi` import even though the Windows build installs `posix`.
    #[cfg(windows)]
    pyre_install_module!(_winapi);
    // `importlib._bootstrap_external` eagerly `import winreg`s on win32; the
    // module must exist for the import machinery (and `import site`) to start.
    #[cfg(windows)]
    pyre_install_module!(winreg);
    // `subprocess` picks its Windows implementation from the presence of
    // `msvcrt`; `getpass` reads the console through it.
    #[cfg(all(windows, feature = "host_env"))]
    pyre_install_module!(msvcrt);
    pyre_install_module!(_abc);
    pyre_install_module!(_functools);
    pyre_install_module!("_thread"(thread));
    pyre_install_module!(itertools);
    pyre_install_module!(_contextvars);
    pyre_install_module!(_codecs);
    // moduledef.py: `applevel_name = os.name` installs the one posix module
    // under `os.name` — `"posix"` on a POSIX host, `"nt"` on Windows, where a
    // module literally named `posix` does not exist. os.py picks `os.name` and
    // the `path` module (posixpath vs ntpath) from which of the two names is in
    // `sys.builtin_module_names`.
    #[cfg(all(not(target_arch = "wasm32"), not(windows)))]
    pyre_install_module!(posix);
    #[cfg(windows)]
    pyre_install_module!("nt"(posix));
    pyre_install_module!(errno);
    pyre_install_module!(_collections);
    pyre_install_module!(_ast);
    pyre_install_module!(_opcode);
    pyre_install_module!("_imp"(imp));

    // importlib package and its submodules load their real source from disk:
    // the package `__init__.py` binds `__import__`/`import_module`/… from the
    // frozen `_bootstrap`, and `machinery`/`abc`/`util` re-export the real
    // finders/loaders/spec classes out of `_frozen_importlib{,_external}`. The
    // frozen bootstrap modules already carry the full surface, so a native stub
    // would only inject placeholder `object` classes that shadow them.

    // __pypy__ package + builders submodule — the PyPy-only surface
    // pickle.py imports (identity_dict + builders.BytesBuilder).
    pyre_install_module!("__pypy__" => crate::module::__pypy__::init);
    pyre_install_module!("__pypy__.builders" => crate::module::__pypy__::builders::init);

    // pypyjit — runtime JIT-parameter control (`set_param`).
    pyre_install_module!("pypyjit" => crate::module::pypyjit::init);

    pyre_install_module!(atexit);
    // faulthandler installs host signal handlers and writes tracebacks to a raw
    // fd, neither of which is mediated; like the other host-access modules below
    // the sandbox interpreter omits it (PyPy keeps it out of default_modules
    // under translation.sandbox).
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    pyre_install_module!(faulthandler);

    // Host-access modules — network (`_socket`), arbitrary FFI (`_ctypes`),
    // subprocess/`fork`+`exec` (`_posixsubprocess`), shared memory
    // (`_multiprocessing`/`_posixshmem`), system log, fd/tty control
    // (`fcntl`/`termios`/`select`/`resource`), real signals, and the host
    // user/group databases (`pwd`/`grp`).  None belong to the mediated
    // ll_os/ll_time surface, so the sandbox interpreter omits them entirely:
    // `import _socket` then raises ModuleNotFoundError, as in a build whose
    // syscall code is absent.
    #[cfg(not(feature = "sandbox"))]
    {
        #[cfg(not(target_arch = "wasm32"))]
        pyre_install_module!("_signal"(signal));
        // Only a POSIX host has the user/group databases these read; the
        // platforms without them have no `pwd`/`grp` module at all, and the
        // callers depend on that: `posixpath.expanduser`, `pathlib` and
        // `tarfile` all reach for the module inside `try/except ImportError`
        // and take a fallback when it is missing.
        #[cfg(unix)]
        pyre_install_module!(pwd);
        #[cfg(unix)]
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
        pyre_install_module!(_ctypes);
        #[cfg(not(target_arch = "wasm32"))]
        pyre_install_module!(_posixshmem);
        pyre_install_module!(_posixsubprocess);
        pyre_install_module!(_multiprocessing);
    }
    pyre_install_module!(_locale);
    pyre_install_module!(_random);
    pyre_install_module!(_pickle);
    pyre_install_module!("_struct"(r#struct));
    pyre_install_module!(binascii);
    pyre_install_module!(marshal);
    pyre_install_module!(zlib);
    pyre_install_module!(_typing);
    pyre_install_module!(_template);
    pyre_install_module!(_hashlib);
    pyre_install_module!(_blake2);
    pyre_install_module!(gc);
    pyre_install_module!(unicodedata);
    pyre_install_module!(pyexpat);

    // `_sysconfigdata_{abiflags}_{platform}_{multiarch}` is a generated
    // Python module containing `build_time_vars = {...}` that sysconfig
    // imports from `_init_posix`.  Empty dict suffices.
    // PyPy: `lib_pypy/pypy_tools/build_cffi_imports.py` creates the same file.
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
    //
    // Modules whose stdlib wrapper does `import X` + attribute access or
    // `from X import *` are deliberately NOT stubbed here: an empty stub
    // makes the `import` succeed and the later access raise AttributeError
    // (or silently bind nothing), which the wrapper's `try/except
    // ImportError` cannot recover from.  Leaving them unregistered lets the
    // pure-Python fallback take over: `_datetime` -> `_pydatetime`,
    // `_decimal` -> `_pydecimal`, `_asyncio` -> pure-Python asyncio.
    for name in &["_heapq", "_bisect", "_stat", "_queue", "_zoneinfo"] {
        register_builtin_module(name, empty_module_init);
    }
    register_builtin_module_with_startup(
        "array",
        crate::module::array::init_array_module,
        crate::module::array::startup_array_module,
    );
    register_builtin_module("_csv", crate::module::_csv::init);
    register_builtin_module("_tokenize", crate::module::_tokenize::init);
    register_builtin_module("_scproxy", init_scproxy);
    register_builtin_module("_string", init_string_module);
    register_builtin_module("_tracemalloc", init_tracemalloc);
    register_builtin_module("_sysconfig", init_sysconfig_stub);
}

fn require_string_module_str(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(&arg) = args.first() else {
        return Err(crate::PyError::type_error("expected str, got object"));
    };
    if !unsafe { pyre_object::is_str(arg) } {
        return Err(crate::PyError::type_error(format!(
            "expected str, got {}",
            crate::type_methods::arg_type_name(arg)
        )));
    }
    Ok(arg)
}

fn init_string_module(ns: PyObjectRef) {
    crate::module_ns_store(
        ns,
        "formatter_parser",
        crate::make_builtin_function("formatter_parser", |args| {
            use rustpython_common::format::{FormatPart, FormatString, FromTemplate};

            let arg = require_string_module_str(args)?;
            let body = unsafe { pyre_object::w_str_get_wtf8(arg) };
            let parsed = FormatString::from_str(body)
                .map_err(|_| crate::PyError::value_error("bad format string"))?;

            let mut tuples = Vec::new();
            let mut pending: Option<Wtf8Buf> = None;
            for part in parsed.format_parts {
                match part {
                    FormatPart::Literal(text) => pending = Some(text),
                    FormatPart::Field {
                        field_name,
                        conversion_spec,
                        format_spec,
                    } => {
                        let literal = pending.take().unwrap_or_default();
                        let conversion = match conversion_spec {
                            Some(c) => pyre_object::w_str_new(&c.to_char_lossy().to_string()),
                            None => pyre_object::w_none(),
                        };
                        tuples.push(pyre_object::w_tuple_new(vec![
                            pyre_object::w_str_from_wtf8(literal),
                            pyre_object::w_str_from_wtf8(field_name),
                            pyre_object::w_str_from_wtf8(format_spec),
                            conversion,
                        ]));
                    }
                }
            }
            if let Some(text) = pending {
                tuples.push(pyre_object::w_tuple_new(vec![
                    pyre_object::w_str_from_wtf8(text),
                    pyre_object::w_none(),
                    pyre_object::w_none(),
                    pyre_object::w_none(),
                ]));
            }
            Ok(pyre_object::w_list_new(tuples))
        }),
    );
    crate::module_ns_store(
        ns,
        "formatter_field_name_split",
        crate::make_builtin_function("formatter_field_name_split", |args| {
            use rustpython_common::format::{FieldName, FieldNamePart, FieldType};

            let arg = require_string_module_str(args)?;
            let body = unsafe { pyre_object::w_str_get_wtf8(arg) };
            let FieldName { field_type, parts } = FieldName::parse(body)
                .map_err(|_| crate::PyError::value_error("bad field name"))?;

            let first = match field_type {
                FieldType::Auto => pyre_object::w_str_new(""),
                FieldType::Index(n) => pyre_object::w_int_new(n as i64),
                FieldType::Keyword(s) => pyre_object::w_str_from_wtf8(s),
            };
            let rest = parts
                .into_iter()
                .map(|part| match part {
                    FieldNamePart::Attribute(s) => pyre_object::w_tuple_new(vec![
                        pyre_object::w_bool_from(true),
                        pyre_object::w_str_from_wtf8(s),
                    ]),
                    FieldNamePart::Index(n) => pyre_object::w_tuple_new(vec![
                        pyre_object::w_bool_from(false),
                        pyre_object::w_int_new(n as i64),
                    ]),
                    FieldNamePart::StringIndex(s) => pyre_object::w_tuple_new(vec![
                        pyre_object::w_bool_from(false),
                        pyre_object::w_str_from_wtf8(s),
                    ]),
                })
                .collect();
            Ok(pyre_object::w_tuple_new(vec![
                first,
                pyre_object::w_list_new(rest),
            ]))
        }),
    );
}

/// `_sysconfig` stub — exposes `config_vars()` returning an empty dict. On
/// POSIX `sysconfig` only consults this for the build variables that pyre does
/// not generate; importing it is enough to satisfy `test_sysconfig`.
fn init_sysconfig_stub(ns: PyObjectRef) {
    crate::module_ns_store(
        ns,
        "config_vars",
        crate::make_builtin_function("config_vars", |_| Ok(pyre_object::w_dict_new())),
    );
}

/// `_tracemalloc` stub — allocation tracking is not implemented, so the
/// tracing primitives are neutral no-ops that let `tracemalloc` import and
/// report an inactive tracer.
fn init_tracemalloc(ns: PyObjectRef) {
    crate::module_ns_store(
        ns,
        "start",
        crate::make_builtin_function("start", |_| Ok(pyre_object::w_none())),
    );
    crate::module_ns_store(
        ns,
        "stop",
        crate::make_builtin_function("stop", |_| Ok(pyre_object::w_none())),
    );
    crate::module_ns_store(
        ns,
        "clear_traces",
        crate::make_builtin_function("clear_traces", |_| Ok(pyre_object::w_none())),
    );
    crate::module_ns_store(
        ns,
        "reset_peak",
        crate::make_builtin_function("reset_peak", |_| Ok(pyre_object::w_none())),
    );
    crate::module_ns_store(
        ns,
        "is_tracing",
        crate::make_builtin_function("is_tracing", |_| Ok(pyre_object::w_bool_from(false))),
    );
    crate::module_ns_store(
        ns,
        "get_traceback_limit",
        crate::make_builtin_function("get_traceback_limit", |_| Ok(pyre_object::w_int_new(1))),
    );
    crate::module_ns_store(
        ns,
        "get_tracemalloc_memory",
        crate::make_builtin_function("get_tracemalloc_memory", |_| Ok(pyre_object::w_int_new(0))),
    );
    crate::module_ns_store(
        ns,
        "get_traced_memory",
        crate::make_builtin_function("get_traced_memory", |_| {
            Ok(pyre_object::w_tuple_new(vec![
                pyre_object::w_int_new(0),
                pyre_object::w_int_new(0),
            ]))
        }),
    );
    crate::module_ns_store(
        ns,
        "_get_traces",
        crate::make_builtin_function("_get_traces", |_| Ok(pyre_object::w_list_new(Vec::new()))),
    );
    crate::module_ns_store(
        ns,
        "_get_object_traceback",
        crate::make_builtin_function("_get_object_traceback", |_| Ok(pyre_object::w_none())),
    );
}

/// `_scproxy` — the macOS SystemConfiguration proxy probe that
/// `urllib.request.getproxies_macosx_sysconf` / `proxy_bypass_macosx_sysconf`
/// import.  Report "no system proxy configured" so the import succeeds and
/// proxy resolution yields an empty mapping.
fn init_scproxy(ns: PyObjectRef) {
    crate::module_ns_store(
        ns,
        "_get_proxies",
        crate::make_builtin_function("_get_proxies", |_| Ok(pyre_object::w_dict_new())),
    );
    crate::module_ns_store(
        ns,
        "_get_proxy_settings",
        crate::make_builtin_function("_get_proxy_settings", |_| {
            let d = pyre_object::w_dict_new();
            unsafe {
                pyre_object::w_dict_store(
                    d,
                    pyre_object::w_str_new("exclude_simple"),
                    pyre_object::w_bool_from(false),
                );
                pyre_object::w_dict_store(
                    d,
                    pyre_object::w_str_new("exceptions"),
                    pyre_object::w_list_new(Vec::new()),
                );
            }
            Ok(d)
        }),
    );
}

/// Empty module initializer for C-extension stubs.
fn empty_module_init(_ns: PyObjectRef) {}

/// `_sysconfigdata_*` stub — sysconfig imports this generated module to
/// read the CPython build variables. We expose a minimal `build_time_vars`
/// dict that lets sysconfig initialize without crashing.
fn init_sysconfigdata_empty(ns: PyObjectRef) {
    let vars = pyre_object::w_dict_new();
    // A few keys are load-bearing — sysconfig.get_config_vars() populates
    // them, but an import-time crash hits on 'Py_GIL_DISABLED' and
    // similar. Leave the dict empty; .get('X') returns None for unknown
    // keys which every caller already handles.
    crate::module_ns_store(ns, "build_time_vars", vars);
}

/// Try to load a builtin module by name.
///
/// PyPy equivalent: `find_module()` → C_BUILTIN path →
/// `getbuiltinmodule()` → `Module.__init__` + `startup()`.
///
/// PyPy `pypy/objspace/std/dictmultiobject.py:60-69` allocates a
/// `W_ModuleDictObject` for every module via
/// `allocate_and_init_instance(module=True)`. Pyre mirrors that here:
/// the initializer writes directly into a rooted, non-moving module dict.
pub(crate) fn load_builtin_module(name: &str) -> Option<PyObjectRef> {
    let module_def = BUILTIN_MODULES.lock().unwrap().get(name).copied()?;
    let w_dict = pyre_object::dictmultiobject::w_module_dict_new();
    let _roots = pyre_object::gc_roots::push_roots();
    let save_point = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_dict);
    let name_obj = pyre_object::w_str_new(name);
    pyre_object::gc_roots::pin_root(name_obj);
    // Set __name__ (PyPy: Module.__init__ sets __name__)
    crate::module_ns_store(
        w_dict,
        "__name__",
        pyre_object::gc_roots::shadow_stack_get(save_point + 1),
    );
    // Run module-specific initializer (PyPy: interpleveldefs)
    (module_def.init)(w_dict);
    // MixedModule parity: interp-level builtin functions carry the module
    // name as `__module__`, so `pickle` can save them by reference
    // (`save_global`) without guessing via `whichmodule`. Snapshot owned
    // keys, then reload each movable value and the rooted name afresh.
    let keys: Vec<String> = unsafe { pyre_object::dictmultiobject::w_dict_str_entries(w_dict) }
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    for key in &keys {
        if let Some(value) =
            unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(w_dict, key) }
        {
            unsafe {
                // MixedModule._load_lazily: every function directly in a
                // mixed-module is a non-descriptor `BuiltinFunction` (no
                // `__get__`), so storing it on a user class does not bind `self`.
                crate::function::demote_module_function_to_builtin(value);
                crate::function::builtin_function_set_module(
                    value,
                    pyre_object::gc_roots::shadow_stack_get(save_point + 1),
                );
            }
        }
    }
    let module = pyre_object::w_module_new_aliasing_dict(name, w_dict);
    // function.py:797-815 BuiltinFunction.w_moduleobj — MixedModule binds
    // every interp-level function to the live defining module object.
    for key in &keys {
        if let Some(value) =
            unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(w_dict, key) }
        {
            unsafe { crate::function::builtin_function_set_module_obj(value, module) };
        }
    }
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
        crate::module_ns_store(w_dict, "__builtins__", module);
    }
    Some(module)
}

/// The builtin-module half of `load_part`: build the module, bind it in
/// `sys.modules`, then run its `startup` hook. `_imp.create_builtin` performs
/// the same three steps a native `import` does, so the two entry points leave
/// a builtin module in the same state.
pub(crate) fn create_builtin_module(
    name: &str,
    execution_context: *const PyExecutionContext,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    // `import builtins` must resolve to `space.builtin`, the one Module every
    // frame uses for its LOAD_GLOBAL fallback. Historically a fresh
    // `load_builtin_module` reran `install_default_builtins`, minted a second
    // exception hierarchy, and overwrote the name→class registry. The
    // process-global get-or-mint registry now prevents that identity
    // clobbering even on another fresh-dictionary path, while this guard still
    // preserves the builtins Module identity. `load_part` routes the name this
    // way; the `_imp.create_builtin` entry point must too.
    if name == "builtins" && !execution_context.is_null() {
        let module = unsafe { (*execution_context).get_builtin() };
        set_sys_module(name, module);
        return Ok(Some(module));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let Some(module) = load_builtin_module(name) else {
        return Ok(None);
    };
    pyre_object::gc_roots::pin_root(module);
    set_sys_module(name, pyre_object::gc_roots::shadow_stack_get(module_slot));
    let module = pyre_object::gc_roots::shadow_stack_get(module_slot);
    startup_builtin_module(name, module, execution_context)?;
    Ok(Some(pyre_object::gc_roots::shadow_stack_get(module_slot)))
}

/// Set a builtin module's `__spec__`/`__loader__`/`__package__` from the
/// app-level `BuiltinImporter`, matching `BuiltinImporter.exec_module` →
/// `_init_module_attrs`. Reachable only once `importlib._bootstrap` is wired;
/// the handful of builtins imported before that are fixed up in bulk by
/// `_bootstrap._setup`'s sys.modules walk, so a no-op here is correct then.
#[cfg(feature = "host_env")]
fn set_builtin_module_spec(name: &str, module: PyObjectRef) -> Result<(), crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let Some(bootstrap) = get_sys_module("importlib._bootstrap") else {
        return Ok(());
    };

    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    pin_root(module);
    let boot_slot = shadow_stack_len();
    pin_root(bootstrap);

    // Best-effort throughout: a builtin imported while `_bootstrap` is still
    // executing sees a partially-initialised module whose `BuiltinImporter` /
    // `_init_module_attrs` are not defined yet. Skip rather than break the
    // import — `_bootstrap._setup` fixes up any builtin missed here.
    let Ok(importer) =
        crate::baseobjspace::getattr_str(shadow_stack_get(boot_slot), "BuiltinImporter")
    else {
        return Ok(());
    };
    let importer_slot = shadow_stack_len();
    pin_root(importer);
    let Ok(find_spec) =
        crate::baseobjspace::getattr_str(shadow_stack_get(importer_slot), "find_spec")
    else {
        return Ok(());
    };
    let find_spec_slot = shadow_stack_len();
    pin_root(find_spec);
    let w_name = pyre_object::w_str_new(name);
    let name_slot = shadow_stack_len();
    pin_root(w_name);
    let Ok(spec) = crate::call::call_function_impl_result(
        shadow_stack_get(find_spec_slot),
        &[shadow_stack_get(name_slot)],
    ) else {
        return Ok(());
    };
    if unsafe { pyre_object::is_none(spec) } {
        return Ok(());
    }
    let spec_slot = shadow_stack_len();
    pin_root(spec);

    // _init_module_attrs(spec, module)
    let Ok(init) =
        crate::baseobjspace::getattr_str(shadow_stack_get(boot_slot), "_init_module_attrs")
    else {
        return Ok(());
    };
    let init_slot = shadow_stack_len();
    pin_root(init);
    let _ = crate::call::call_function_impl_result(
        shadow_stack_get(init_slot),
        &[shadow_stack_get(spec_slot), shadow_stack_get(mod_slot)],
    );
    Ok(())
}

/// Off-`host_env` builds have no app-level importlib to source specs from.
#[cfg(not(feature = "host_env"))]
fn set_builtin_module_spec(_name: &str, _module: PyObjectRef) -> Result<(), crate::PyError> {
    Ok(())
}

/// Set a source module's `__spec__`/`__loader__`/`__file__`/`__cached__` from
/// the app-level `_bootstrap_external._fix_up_module` — the helper
/// `PyImport_ExecCodeModuleObject` calls. Returns `false` when the importlib
/// bootstrap is not wired yet, so the caller can seed `None` instead.
///
/// `ns` (the module dict) is written in place; the caller keeps it pinned.
#[cfg(feature = "host_env")]
fn fix_up_source_module_spec(
    ns: PyObjectRef,
    pathname: &str,
    cpathname: Option<&str>,
) -> Result<bool, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let Some(ext) = get_sys_module("importlib._bootstrap_external") else {
        return Ok(false);
    };
    let Some(w_name) = (unsafe { pyre_object::w_dict_getitem_str(ns, "__name__") }) else {
        return Ok(false);
    };

    let _roots = push_roots();
    let ns_slot = shadow_stack_len();
    pin_root(ns);
    let name_slot = shadow_stack_len();
    pin_root(w_name);
    let ext_slot = shadow_stack_len();
    pin_root(ext);
    let w_path = pyre_object::w_str_new(pathname);
    let path_slot = shadow_stack_len();
    pin_root(w_path);
    let w_cpath = match cpathname {
        Some(c) => pyre_object::w_str_new(c),
        None => pyre_object::w_none(),
    };
    let cpath_slot = shadow_stack_len();
    pin_root(w_cpath);

    // Best-effort: `_bootstrap_external` itself is a source module, and its
    // spec is fixed up (during partial-init) before its body defines
    // `_fix_up_module` — the getattr then raises. Fall back to `None` seeding
    // for that (and any other partially-initialised) case rather than break
    // the import; the module's spec is corrected by later imports / `_setup`.
    let Ok(fix) = crate::baseobjspace::getattr_str(shadow_stack_get(ext_slot), "_fix_up_module")
    else {
        return Ok(false);
    };
    let fix_slot = shadow_stack_len();
    pin_root(fix);
    // An error raised by `_fix_up_module` itself propagates — the appexec
    // at importing.py:293-298 does not shield the call.
    crate::call::call_function_impl_result(
        shadow_stack_get(fix_slot),
        &[
            shadow_stack_get(ns_slot),
            shadow_stack_get(name_slot),
            shadow_stack_get(path_slot),
            shadow_stack_get(cpath_slot),
        ],
    )?;
    Ok(true)
}

/// Off-`host_env` builds have no app-level importlib to source specs from.
#[cfg(not(feature = "host_env"))]
fn fix_up_source_module_spec(
    _ns: PyObjectRef,
    _pathname: &str,
    _cpathname: Option<&str>,
) -> Result<bool, crate::PyError> {
    Ok(false)
}

fn startup_builtin_module(
    name: &str,
    module: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<(), crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    pin_root(module);

    let startup = BUILTIN_MODULES
        .lock()
        .unwrap()
        .get(name)
        .and_then(|d| d.startup);
    if let Some(startup) = startup {
        startup(shadow_stack_get(mod_slot), execution_context)?;
    }
    set_builtin_module_spec(name, shadow_stack_get(mod_slot))?;
    Ok(())
}

/// Initialize sys.path with the directory containing the main script.
///
/// PyPy equivalent: sys.path is populated at startup with the script
/// directory, then PYTHONPATH entries, then the stdlib.
#[cfg(feature = "host_env")]
/// The canonical absolute form of a startup `sys.path[0]` directory, for the
/// shadowing check to compare against absolute module origins.  Under the
/// sandbox the path is left as given (a virtual path the controller mediates);
/// `canonicalize` would issue raw host syscalls past the seccomp lockdown.
fn canonical_startup_dir(dir: &Path) -> String {
    #[cfg(not(feature = "sandbox"))]
    if let Ok(abs) = std::path::absolute(dir) {
        return abs
            .canonicalize()
            .unwrap_or(abs)
            .to_string_lossy()
            .into_owned();
    }
    dir.to_string_lossy().into_owned()
}

/// `script_dir` is the shadowing-check anchor (`config->sys_path_0`'s
/// directory); `path0` is the literal entry `add_sys_path_0` later prepends to
/// `sys.path` — `""` for `-c` / stdin / the REPL, the cwd for `-m`, the
/// script's directory for a script.
pub fn init_sys_path(script_dir: &Path, path0: &str) {
    // Register builtin modules (PyPy: make_builtins / setup_builtin_modules)
    install_builtin_modules();

    // Record the startup `sys.path[0]` for the shadowing check before any user
    // code can mutate `sys.path`.  Module origins are absolute canonical paths,
    // so store the canonical absolute form here for the directory comparison to
    // hold for a relative script argument or a symlinked working directory.
    *SYS_PATH_0.lock().unwrap() = Some(canonical_startup_dir(script_dir));

    // `pymain_run_python` prepends `sys.path[0]` only after `site` has run, so
    // `site.removeduppaths()` never absolutizes the `-c` / REPL empty entry into
    // the cwd.  Stage the entry here; `add_sys_path_0` performs the insert.
    // `-P` (safe_path) suppresses it entirely.
    *SYS_PATH_0_PENDING.lock().unwrap() = (!safe_path_flag()).then(|| path0.to_string());

    {
        let mut path = SYS_PATH.lock().unwrap();
        path.clear();
        // PYTHONPATH entries head the seed and precede the stdlib
        // (pathconfig.c), split on the platform path-list separator.  Honoured
        // regardless of the safe-path flag, which only suppresses the
        // `sys.path[0]` entry, but skipped under `-E` / `-I` (ignore_environment),
        // which ignore every `PYTHON*` variable. The sandbox interpreter takes
        // its search path from the controller, so it does not read the host
        // environment here.
        #[cfg(not(feature = "sandbox"))]
        if !ignore_environment_flag() {
            if let Ok(pythonpath) = host_os::var("PYTHONPATH") {
                let sep = if cfg!(windows) { ';' } else { ':' };
                // Empty components are preserved — an empty `sys.path` entry
                // denotes the current directory (app_main.setup_and_fix_paths
                // extends with the raw split).
                path.extend(pythonpath.split(sep).map(PathBuf::from));
            }
        }
        // The stdlib entry is appended when the `sys` module is created —
        // `create_sys_path_list` forces `ensure_stdlib_path` before flushing
        // this seed into `sys.path`.
    }
}

/// Locate the vendored stdlib (`lib-python/3`) by walking up the running
/// executable's ancestor directories.
///
/// PyPy equivalent: initpath.py walks up from the executable to a
/// directory containing `lib-python/X.Y`.
#[cfg(all(feature = "host_env", not(feature = "sandbox")))]
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
pub(crate) fn detect_stdlib_path() -> Option<PathBuf> {
    // Under sandbox the controller provisions the stdlib mount via
    // `PYRE_STDLIB`; trust it verbatim — the seam-backed SourceProvider
    // mediates every subsequent read — and never read `current_exe` or spawn
    // a `python3` subprocess (both escape the controller).
    #[cfg(feature = "sandbox")]
    {
        // Read through the env seam so the lookup reaches the controller's
        // virtual environment (the child's real env was cleared at spawn); the
        // controller seeds PYRE_STDLIB to the `--lib` mount at `/bin/lib`.
        use std::os::unix::ffi::OsStrExt;
        return crate::host_seam::ops::getenv(b"PYRE_STDLIB")
            .ok()
            .flatten()
            .map(|bytes| PathBuf::from(std::ffi::OsStr::from_bytes(&bytes)));
    }
    #[cfg(not(feature = "sandbox"))]
    {
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
}

/// Add a directory to `sys.path`.
///
/// Before the `sys` module exists this stages the entry in the native
/// `SYS_PATH` seed, which is flushed into `sys.path` when `sys` is created.
/// Once `sys` exists the Python list is authoritative, so the entry is appended
/// to it in place (deduplicated) and the spent seed is left untouched. A
/// missing or non-list `sys.path` (e.g. after `del sys.path`) is respected.
#[cfg(feature = "host_env")]
pub fn add_sys_path(dir: &Path) {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let entry = dir.to_string_lossy();
    if get_sys_module("sys").is_none() {
        {
            let mut path = SYS_PATH.lock().unwrap();
            let pb = dir.to_path_buf();
            if !path.contains(&pb) {
                path.push(pb);
            }
        }
        return;
    }
    // Pin the new entry before any further allocation (`get_sys_module` and the
    // dict lookup allocate) can relocate it. After the list is fetched below the
    // path is allocation-free until the pinned entry reaches `w_list_append`.
    let _roots = push_roots();
    let slot = shadow_stack_len();
    pin_root(pyre_object::w_str_new(entry.as_ref()));
    let Some(sys_mod) = get_sys_module("sys") else {
        return;
    };
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(sys_mod) };
    if w_dict.is_null() {
        return;
    }
    let Some(w_path) = (unsafe { pyre_object::w_dict_getitem_str(w_dict, "path") }) else {
        return;
    };
    if !unsafe { pyre_object::is_list(w_path) } {
        return;
    }
    let n = unsafe { pyre_object::listobject::w_list_len(w_path) };
    for i in 0..n {
        if let Some(item) = unsafe { pyre_object::listobject::w_list_getitem(w_path, i as i64) } {
            if unsafe { pyre_object::is_str(item) }
                && unsafe { pyre_object::w_str_get_value(item) } == entry.as_ref()
            {
                return;
            }
        }
    }
    unsafe { pyre_object::listobject::w_list_append(w_path, shadow_stack_get(slot)) };
}

/// `pymain_sys_path_add_path0` — prepend the startup `sys.path[0]` entry staged
/// by `init_sys_path`.  Called after `site` has run so `removeduppaths` cannot
/// absolutize an empty entry into the cwd, and inserted unconditionally (no
/// dedup).  The entry is taken, so a second call is a no-op.
#[cfg(feature = "host_env")]
pub fn add_sys_path_0() {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let Some(entry) = SYS_PATH_0_PENDING.lock().unwrap().take() else {
        return;
    };
    // No `sys` yet (an embedder that never imports `site`): stage at the front
    // of the seed instead, which `create_sys_path_list` flushes in order.
    if get_sys_module("sys").is_none() {
        SYS_PATH.lock().unwrap().insert(0, PathBuf::from(&entry));
        return;
    }
    // Pin the new entry before any further allocation (`get_sys_module` and the
    // dict lookup allocate) can relocate it — `add_sys_path` parity.
    let _roots = push_roots();
    let slot = shadow_stack_len();
    pin_root(pyre_object::w_str_new(&entry));
    let Some(sys_mod) = get_sys_module("sys") else {
        return;
    };
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(sys_mod) };
    if w_dict.is_null() {
        return;
    }
    let Some(w_path) = (unsafe { pyre_object::w_dict_getitem_str(w_dict, "path") }) else {
        return;
    };
    if !unsafe { pyre_object::is_list(w_path) } {
        return;
    }
    unsafe { pyre_object::listobject::w_list_insert(w_path, 0, shadow_stack_get(slot)) };
}

// ── check_sys_modules ────────────────────────────────────────────────
// PyPy equivalent: importing.py `check_sys_modules(space, w_modulename)`

/// Reads the process-owned `SYS_MODULES` registry (and the runtime-stamped
/// `sys.modules` dict through `sys_modules_dict`), neither a build-time
/// constant, so the JIT residualizes the call rather than folding a stale
/// `sys.modules` snapshot (`@dont_look_inside`, the `sys_modules_dict` /
/// `lookup_exc_class` shape).  The `Option<PyObjectRef>` return fits one word
/// and the `&str` argument matches `lookup_exc_class`.
#[majit_macros::dont_look_inside]
pub(crate) fn check_sys_modules(name: &str) -> Option<PyObjectRef> {
    // Consult the Python-visible sys.modules dict first so that user code
    // writing `sys.modules['foo'] = mod` is immediately visible to imports.
    // PyPy: importing.py check_sys_modules reads space.sys.get('modules').
    let dict = sys_modules_dict();
    if !dict.is_null() {
        if let Some(m) = unsafe { pyre_object::w_dict_getitem_str(dict, name) } {
            if !m.is_null() && !unsafe { pyre_object::is_none(m) } {
                return Some(m);
            }
        }
    }
    sys_modules_registry_get(name)
}

/// Look `name` up in `SYS_MODULES`, the process-owned name→module registry.
///
/// The single read seam every traced reader of `SYS_MODULES` goes through.
/// Entries are stamped at runtime by `set_sys_module` / `remove_sys_module`,
/// so the map is not a build-time constant and the JIT residualizes the read
/// instead of tracing into it (`@dont_look_inside`, the `sys_modules_dict`
/// shape).  The mutating and GC-walking users keep the raw static.
///
/// The read is poison-tolerant, following the `get_interpreter_sys_module`
/// spelling this replaces: a poisoned mutex means another thread panicked
/// while holding it, and a `HashMap<String, usize>` has no invariant a panic
/// mid-`insert`/`remove` could leave broken.
#[majit_macros::dont_look_inside]
pub(crate) fn sys_modules_registry_get(name: &str) -> Option<PyObjectRef> {
    SYS_MODULES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(name)
        .copied()
        .map(|module| module as PyObjectRef)
}

/// Whether `sys.modules[name]` is bound to `None`, the sentinel that marks
/// a name as blocked.
///
/// `_bootstrap._find_and_load` treats it as a cached "this import must not
/// succeed" and raises instead of searching, which is how code disables an
/// import for everything downstream of it.  `check_sys_modules` cannot
/// report it — `None` is not a module it can hand back, so it falls through
/// to the search — hence the separate lookup.
fn sys_modules_blocks(name: &str) -> bool {
    // Build the key before reading the dict, the order `check_sys_modules`
    // uses: reading the thread-local last keeps the borrow off the stack
    // across the allocation.
    let key = pyre_object::w_str_new(name);
    let dict = sys_modules_dict();
    if dict.is_null() {
        return false;
    }
    match unsafe { pyre_object::w_dict_lookup(dict, key) } {
        Some(m) => !m.is_null() && unsafe { pyre_object::is_none(m) },
        None => false,
    }
}

/// Look up a loaded module by name in `sys.modules` (Python-visible dict
/// first, then the interpreter cache). Mirrors `check_sys_modules`.
pub fn get_sys_module(name: &str) -> Option<PyObjectRef> {
    check_sys_modules(name)
}

/// Return the interpreter-owned `sys` module, bypassing the Python-visible
/// `sys.modules` mapping.
///
/// CPython's `_PySys_GetOptionalAttr*` reads `PyInterpreterState.sysdict`
/// directly, and PyPy reads `space.sys`; neither follows a replacement
/// `sys.modules["sys"]`.  `SYS_MODULES` is pyre's process/interpreter-owned
/// module registry and is independently walked as a GC root, so its original
/// `sys` entry is the corresponding owner.
pub fn get_interpreter_sys_module() -> Option<PyObjectRef> {
    sys_modules_registry_get("sys")
}

/// The Python-visible `sys.modules` dict, or `PY_NULL` before it is
/// installed. Used by callers that need to iterate every loaded module
/// (e.g. pickle's `whichmodule` scan), and the single read seam every
/// traced reader of `SYS_MODULES_DICT` goes through.
///
/// The pointer is stamped at runtime by `set_sys_modules_dict`, so it is not
/// a build-time constant and the JIT residualizes the read instead of
/// tracing into it (`@dont_look_inside`, the `gc_interp::enabled` shape).
/// The `-> PyObjectRef` return fits a single word and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn sys_modules_dict() -> PyObjectRef {
    SYS_MODULES_DICT.load(Ordering::Acquire) as PyObjectRef
}

pub fn set_sys_module(name: &str, module: PyObjectRef) {
    // A new module joins the `walk_module_dicts_gc` root set; its dict
    // may hold young values — rescan on the next minor collection.
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    SYS_MODULES
        .lock()
        .unwrap()
        .insert(name.to_string(), module as usize);
    // Keep the Python-visible sys.modules dict in sync.
    let dict = sys_modules_dict();
    if !dict.is_null() {
        unsafe {
            pyre_object::w_dict_store(dict, pyre_object::w_str_new(name), module);
        }
    }
}

/// Remove a (partially initialised) module from `sys.modules`.
///
/// `importlib._bootstrap._load` deletes the module it pre-registered when
/// `exec_module` raises, so a retried import re-executes the body rather
/// than handing back a half-built module.  Without this a failed
/// `import ssl` (missing `_ssl`) leaves a broken `ssl` shell behind, and
/// the next `import ssl` succeeds with no `SSLWantReadError`, etc.
pub fn remove_sys_module(name: &str) {
    SYS_MODULES.lock().unwrap().remove(name);
    let dict = sys_modules_dict();
    if !dict.is_null() {
        unsafe {
            pyre_object::w_dict_delitem_str(dict, name);
        }
    }
}

/// GC root walk over every loaded module's dict storage.
///
/// Modules (`malloc_typed`) are Box-immortal, while their non-moving
/// `W_ModuleDictObject`s are GC-managed. Visit each `Module.w_dict` field
/// first so the header is marked and its custom trace can reach the
/// authoritative `dstorage` / `object_storage` / cell registry. A movable
/// value bound at module scope
/// — e.g. `gc.collect` reached through `gc.__dict__`, or any
/// module-level list / instance — would otherwise be read back stale
/// after a collection relocates it.  Treat each loaded module's dict as
/// a pinned root source so those slots stay forwarded.  This complements
/// the per-frame `w_globals` walk in `eval::walk_pyframe_roots`,
/// which additionally covers `exec`/`eval` globals dicts that are not
/// registered in `sys.modules`.
///
/// # Safety
/// `visitor` must tolerate being called on every movable module-dict
/// value slot reachable here.
pub unsafe fn walk_module_dicts_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    {
        for &module in SYS_MODULES.lock().unwrap().values() {
            let module = module as PyObjectRef;
            if module.is_null() || !unsafe { pyre_object::is_module(module) } {
                continue;
            }
            unsafe {
                let module = &mut *(module as *mut pyre_object::module::Module);
                visitor(&mut module.w_dict);
                let w_dict = module.w_dict;
                pyre_object::dictmultiobject::w_module_dict_walk_gc_cells(w_dict, visitor);
            }
        }
    }
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
    let mut dict = SYS_MODULES_DICT.load(Ordering::Acquire) as PyObjectRef;
    if dict.is_null() {
        return;
    }
    visitor(&mut dict);
    SYS_MODULES_DICT.store(dict as usize, Ordering::Release);
}

pub(crate) fn capture_import_root_area() -> *const () {
    IMPORT_ROOT_AREA.with(|area| area as *const _ as *const ())
}

/// # Safety
/// `data` must come from [`capture_import_root_area`], and the owning thread
/// must be quiesced.
pub(crate) unsafe fn walk_import_roots_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut PyObjectRef),
) {
    let area = unsafe { &*(data as *const ImportRootArea) };
    let argv_pending = unsafe { &*area.argv_pending };
    let mut argv = argv_pending.get();
    if !argv.is_null() {
        visitor(&mut argv);
        argv_pending.set(argv);
    }
}

/// Walk the import state owned by PyPy's process/interpreter object space.
///
/// Unlike `SYS_ARGV_PENDING`, these slots are not thread-local.  Keeping this
/// walk separate lets the collector's `collect_nonstack_roots` parity rescan
/// them exactly once at the end of incremental marking.
pub(crate) unsafe fn walk_process_import_roots(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    // `space.sys.modules` is process-owned in PyPy.  STW has quiesced every
    // mutator, so the process-global cache cannot be semantically mutated
    // while this walk holds its native lock.
    for &module in SYS_MODULES.lock().unwrap().values() {
        let module = module as PyObjectRef;
        if module.is_null() || !unsafe { pyre_object::is_module(module) } {
            continue;
        }
        unsafe {
            let module = &mut *(module as *mut pyre_object::module::Module);
            visitor(&mut module.w_dict);
            let w_dict = module.w_dict;
            pyre_object::dictmultiobject::w_module_dict_walk_gc_cells(w_dict, visitor);
        }
    }
    let mut dict = SYS_MODULES_DICT.load(Ordering::Acquire) as PyObjectRef;
    if !dict.is_null() {
        visitor(&mut dict);
        SYS_MODULES_DICT.store(dict as usize, Ordering::Release);
    }
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

// The launcher flags belong to the interpreter, not to whichever thread
// happened to parse the command line: `space.sys.get_flag(...)` reads them
// off the one `space`.  Keeping them process-global means a reader on any
// thread — a codec lookup, `sys.flags`, an embedder calling in — observes
// the values the launcher recorded instead of the per-thread default.
static SYS_NO_SITE: AtomicBool = AtomicBool::new(false);
static SYS_QUIET: AtomicBool = AtomicBool::new(false);
static SYS_INSPECT: AtomicBool = AtomicBool::new(false);
static SYS_NO_USER_SITE: AtomicBool = AtomicBool::new(false);
static SYS_IGNORE_ENVIRONMENT: AtomicBool = AtomicBool::new(false);
static SYS_ISOLATED: AtomicBool = AtomicBool::new(false);
static SYS_DEV_MODE: AtomicBool = AtomicBool::new(false);
static SYS_UTF8_MODE: AtomicI64 = AtomicI64::new(0);
static SYS_SAFE_PATH: AtomicBool = AtomicBool::new(false);
static SYS_OPTIMIZE: AtomicI64 = AtomicI64::new(0);
static SYS_BYTES_WARNING: AtomicI64 = AtomicI64::new(0);
static SYS_DONT_WRITE_BYTECODE: AtomicBool = AtomicBool::new(false);
static SYS_UNBUFFERED: AtomicBool = AtomicBool::new(false);
// pypy/interpreter/app_main.py keeps the raw `-X` strings in
// `options['_xoptions']` (a list) until sys initialization builds the public
// dict.  Preserve that owner/storage shape rather than introducing a map here.
static SYS_XOPTIONS: LazyLock<Mutex<Vec<String>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static SYS_WARNOPTIONS: LazyLock<Mutex<Vec<String>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static SYS_ORIG_ARGV: LazyLock<Mutex<Vec<String>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static SYS_STDIO_ENCODING: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));

/// Record whether the launcher was given `-S` (no `site` import), so the
/// `sys.flags.no_site` field built during sys module init reflects it. Set
/// before the first `import sys`.
pub fn set_no_site(no_site: bool) {
    SYS_NO_SITE.store(no_site, Ordering::Relaxed);
}

/// Read the `-S` flag for `sys.flags.no_site`.
pub fn no_site_flag() -> bool {
    SYS_NO_SITE.load(Ordering::Relaxed)
}

/// Record the command-line flags consumed by `app_main.py` before `sys` is
/// initialized.  The codec paths also read `dev_mode` directly, matching
/// `space.sys.get_flag('dev_mode')` in `unicodeobject.py`.
#[allow(clippy::too_many_arguments)]
pub fn set_runtime_flags(
    quiet: bool,
    inspect: bool,
    no_user_site: bool,
    ignore_environment: bool,
    isolated: bool,
    dev_mode: bool,
    utf8_mode: i64,
    safe_path: bool,
    optimize: i64,
    bytes_warning: i64,
    dont_write_bytecode: bool,
    unbuffered: bool,
    xoptions: Vec<String>,
    warnoptions: Vec<String>,
    stdio_encoding: Option<String>,
) {
    SYS_QUIET.store(quiet, Ordering::Relaxed);
    SYS_INSPECT.store(inspect, Ordering::Relaxed);
    SYS_NO_USER_SITE.store(no_user_site, Ordering::Relaxed);
    SYS_IGNORE_ENVIRONMENT.store(ignore_environment, Ordering::Relaxed);
    SYS_ISOLATED.store(isolated, Ordering::Relaxed);
    SYS_DEV_MODE.store(dev_mode, Ordering::Relaxed);
    SYS_UTF8_MODE.store(utf8_mode, Ordering::Relaxed);
    SYS_SAFE_PATH.store(safe_path, Ordering::Relaxed);
    SYS_OPTIMIZE.store(optimize, Ordering::Relaxed);
    SYS_BYTES_WARNING.store(bytes_warning, Ordering::Relaxed);
    SYS_DONT_WRITE_BYTECODE.store(dont_write_bytecode, Ordering::Relaxed);
    SYS_UNBUFFERED.store(unbuffered, Ordering::Relaxed);
    *SYS_XOPTIONS.lock().unwrap() = xoptions;
    *SYS_WARNOPTIONS.lock().unwrap() = warnoptions;
    *SYS_STDIO_ENCODING.lock().unwrap() = stdio_encoding;
}

/// Raw `-X` values recorded by the launcher, in command-line order.
pub fn xoptions() -> Vec<String> {
    SYS_XOPTIONS.lock().unwrap().clone()
}

pub fn bytes_warning_flag() -> i64 {
    SYS_BYTES_WARNING.load(Ordering::Relaxed)
}

pub fn unbuffered_flag() -> bool {
    SYS_UNBUFFERED.load(Ordering::Relaxed)
}

pub fn warnoptions() -> Vec<String> {
    SYS_WARNOPTIONS.lock().unwrap().clone()
}

pub fn stdio_encoding() -> Option<String> {
    SYS_STDIO_ENCODING.lock().unwrap().clone()
}

pub fn set_sys_orig_argv(argv: Vec<String>) {
    *SYS_ORIG_ARGV.lock().unwrap() = argv;
}

pub fn sys_orig_argv() -> Vec<String> {
    SYS_ORIG_ARGV.lock().unwrap().clone()
}

pub fn quiet_flag() -> bool {
    SYS_QUIET.load(Ordering::Relaxed)
}

/// `-i`, which `make_flags` reports as both `inspect` and `interactive`.
/// `PYTHONINSPECT` would set only the former, but the launcher does not
/// read it, so `-i` is the whole of this flag's input.
pub fn inspect_flag() -> bool {
    SYS_INSPECT.load(Ordering::Relaxed)
}

pub fn no_user_site_flag() -> bool {
    SYS_NO_USER_SITE.load(Ordering::Relaxed)
}

pub fn ignore_environment_flag() -> bool {
    SYS_IGNORE_ENVIRONMENT.load(Ordering::Relaxed)
}

pub fn isolated_flag() -> bool {
    SYS_ISOLATED.load(Ordering::Relaxed)
}

pub fn dev_mode_flag() -> bool {
    SYS_DEV_MODE.load(Ordering::Relaxed)
}

pub fn utf8_mode_flag() -> i64 {
    SYS_UTF8_MODE.load(Ordering::Relaxed)
}

/// `#[dont_look_inside]`: reads the runtime-mutable `SYS_SAFE_PATH` global
/// (`-P` / `PYTHONSAFEPATH`), so the tracer residualises the read rather than
/// folding a build-time constant — the runtime-mutable-global accessor pattern.
#[majit_macros::dont_look_inside]
pub fn safe_path_flag() -> bool {
    SYS_SAFE_PATH.load(Ordering::Relaxed)
}

/// `-O` / `-OO` / PYTHONOPTIMIZE level for the default compile `optimize`
/// (stripped asserts, `__debug__` = False, and at level 2 discarded
/// docstrings), clamped into the compiler's byte-wide field.  Levels above 2
/// behave as 2.
pub fn optimize_flag() -> u8 {
    SYS_OPTIMIZE
        .load(Ordering::Relaxed)
        .clamp(0, i64::from(u8::MAX)) as u8
}

/// The raw optimization level for `sys.flags.optimize`, mirroring a large
/// `PYTHONOPTIMIZE` value verbatim; `optimize_flag` clamps this for the
/// compiler.
pub fn optimize_level() -> i64 {
    SYS_OPTIMIZE.load(Ordering::Relaxed)
}

/// `-B` / PYTHONDONTWRITEBYTECODE, driving `sys.flags.dont_write_bytecode` and
/// `sys.dont_write_bytecode`.
pub fn dont_write_bytecode_flag() -> bool {
    SYS_DONT_WRITE_BYTECODE.load(Ordering::Relaxed)
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
    // The fast-path cell walked by `walk_sys_modules_dict_gc` now holds
    // a possibly-young dict; rescan on the next minor collection.
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    SYS_MODULES_DICT.store(dict as usize, Ordering::Release);
    // Populate with all modules already in the cache.
    for (name, &module) in SYS_MODULES.lock().unwrap().iter() {
        unsafe {
            pyre_object::w_dict_store(dict, pyre_object::w_str_new(name), module as PyObjectRef);
        }
    }
}

// ── find_module ──────────────────────────────────────────────────────
// PyPy equivalent: importing.py `find_module()`
// Searches sys.path for `<partname>.py` or `<partname>/__init__.py` (package).

#[derive(Debug)]
enum FindInfo {
    /// A .py source file was found.
    #[cfg(feature = "host_env")]
    SourceFile { pathname: PathBuf },
    /// A package directory with __init__.py was found.
    #[cfg(feature = "host_env")]
    Package { dirpath: PathBuf },
    /// PEP 420 namespace package: one or more matching directories that carry
    /// no `__init__.py`. The portions become the package's `__path__`.
    #[cfg(feature = "host_env")]
    Namespace { dirs: Vec<PathBuf> },
    /// A builtin (Rust-implemented) module was found.
    /// PyPy equivalent: C_BUILTIN modtype in find_module()
    Builtin,
}

#[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
fn find_module(partname: &str, parent_dirs: Option<&[PathBuf]>) -> Option<FindInfo> {
    // Submodule import: search ONLY the parent package's `__path__`, never
    // sys.path or builtins by leaf name.  `_bootstrap._find_and_load` resolves
    // `pkg.sub` against `pkg.__path__`; routing through sys.path lets a
    // same-leaf module from an unrelated package on sys.path shadow it (e.g.
    // `concurrent.futures` resolving to `asyncio/futures.py`).
    if let Some(dirs) = parent_dirs {
        return find_in_dirs(partname, dirs);
    }

    // Check builtin modules first (PyPy: space.builtin_modules check in find_module)
    let is_builtin = BUILTIN_MODULES.lock().unwrap().contains_key(partname);
    if is_builtin {
        return Some(FindInfo::Builtin);
    }

    // Try sys.path first
    if let Some(info) = find_in_sys_path(partname) {
        return Some(info);
    }

    // Fallback stdlib detection. `create_sys_path_list` already forces this at
    // sys-module creation, so the `DONE` guard normally makes this a no-op; it
    // stays as a defensive retry for any miss reached before sys exists.
    ensure_stdlib_path();
    return find_in_sys_path(partname);
}

// wasm has no current_exe / python3 spawn, so there is no `ensure_stdlib_path`
// lazy detection; `sys.path` is seeded by the wasm bootstrap (the embedded-VFS
// mount point, or the host stdlib root for the runner). Otherwise this mirrors
// the native `find_module`: submodule imports search the parent package's
// `__path__`, top-level names check builtins then sys.path — all FS probes go
// through the installed `SourceProvider`.
#[cfg(all(feature = "host_env", target_arch = "wasm32"))]
fn find_module(partname: &str, parent_dirs: Option<&[PathBuf]>) -> Option<FindInfo> {
    if let Some(dirs) = parent_dirs {
        return find_in_dirs(partname, dirs);
    }
    let is_builtin = BUILTIN_MODULES.lock().unwrap().contains_key(partname);
    if is_builtin {
        return Some(FindInfo::Builtin);
    }
    find_in_sys_path(partname)
}

#[cfg(not(feature = "host_env"))]
fn find_module(partname: &str, _parent_dirs: Option<&[PathBuf]>) -> Option<FindInfo> {
    let is_builtin = BUILTIN_MODULES.lock().unwrap().contains_key(partname);
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

#[cfg(feature = "host_env")]
fn find_in_dirs(partname: &str, dirs: &[PathBuf]) -> Option<FindInfo> {
    let mut namespace_dirs: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        // Check for package: <dir>/<partname>/__init__.py
        let pkg_dir = dir.join(partname);
        let init_file = pkg_dir.join("__init__.py");
        if with_source_provider(|p| p.is_file(&init_file)) {
            return Some(FindInfo::Package { dirpath: pkg_dir });
        }

        // Check for source file: <dir>/<partname>.py
        let source_file = dir.join(format!("{partname}.py"));
        if with_source_provider(|p| p.is_file(&source_file)) {
            return Some(FindInfo::SourceFile {
                pathname: source_file,
            });
        }

        // PEP 420: a matching directory without `__init__.py` is a namespace
        // portion. Record it and keep scanning — a regular module or package
        // in a later directory still wins; only if no concrete match is found
        // do the recorded portions form a namespace package.
        if with_source_provider(|p| p.is_dir(&pkg_dir)) {
            namespace_dirs.push(pkg_dir);
        }
    }
    if !namespace_dirs.is_empty() {
        return Some(FindInfo::Namespace {
            dirs: namespace_dirs,
        });
    }
    None
}

/// Read the live Python `sys.path` list as filesystem directories, or an
/// empty vec before `sys` / its `path` list exists (the pre-`sync` bootstrap
/// window). Only `str` entries are collected — path hooks (zipimporter keys
/// and the like) are not resolvable by the native file search.
#[cfg(feature = "host_env")]
fn python_sys_path_dirs() -> Option<Vec<PathBuf>> {
    // `None` means the `sys` module does not exist yet (the pre-`sys` bootstrap
    // window) — the caller falls back to the native seed. Once `sys` exists the
    // Python list is authoritative even when empty: a missing / non-list / empty
    // `sys.path` searches nothing, so `del sys.path` and `sys.path.clear()` break
    // imports exactly as they do under CPython, rather than resurrecting the seed.
    let sys_mod = get_sys_module("sys")?;
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(sys_mod) };
    if w_dict.is_null() {
        return None;
    }
    // Copy the entries up front so no Python borrow is held across the
    // filesystem probes in `find_in_dirs` (which never invoke user code).
    let mut dirs = Vec::new();
    if let Some(w_path) = unsafe { pyre_object::w_dict_getitem_str(w_dict, "path") } {
        if unsafe { pyre_object::is_list(w_path) } {
            let n = unsafe { pyre_object::listobject::w_list_len(w_path) };
            dirs.reserve(n);
            for i in 0..n {
                if let Some(item) =
                    unsafe { pyre_object::listobject::w_list_getitem(w_path, i as i64) }
                {
                    // Non-str entries are skipped: pyre's only path hook is the
                    // native filesystem probe, and CPython also skips an entry no
                    // hook accepts.
                    if unsafe { pyre_object::is_str(item) } {
                        dirs.push(PathBuf::from(unsafe { pyre_object::w_str_get_value(item) }));
                    }
                }
            }
        }
    }
    Some(dirs)
}

/// Search the import path for a top-level `partname`.
///
/// The live Python `sys.path` list is authoritative — user code (and PYTHONPATH)
/// mutate it and imports must honor it, the same precedence `check_sys_modules`
/// gives the Python `sys.modules` dict. The native `SYS_PATH` is only the write
/// side of a pre-`sys` staging seed (flushed into `sys.path` at sys-module
/// creation) and is consulted here solely in that pre-`sys` window.
#[cfg(feature = "host_env")]
fn find_in_sys_path(partname: &str) -> Option<FindInfo> {
    match python_sys_path_dirs() {
        Some(dirs) => {
            let found = find_in_dirs(partname, &dirs);
            // Windows: pyre still registers the `posix` builtin (never `nt`),
            // so `os.path` is posixpath and `site.removeduppaths()` rewrites
            // every drive-letter `sys.path` entry into `<cwd>/D:\...` garbage
            // at startup. Until the `nt` registration lands, a live-list miss
            // falls back to the native seed so the stdlib stays importable.
            #[cfg(windows)]
            let found = found.or_else(|| {
                let path = SYS_PATH.lock().unwrap();
                find_in_dirs(partname, &path)
            });
            found
        }
        None => {
            let path = SYS_PATH.lock().unwrap();
            find_in_dirs(partname, &path)
        }
    }
}

/// Build the initial Python `sys.path` list from the native `SYS_PATH` seed.
/// Called once at sys-module creation so the Python list is populated the
/// instant `sys` exists; from then on the Python list is authoritative and the
/// seed is spent.
///
/// Stdlib detection is forced here (off-wasm) so the vendored stdlib is on
/// `sys.path` before any user code — including `python -S` / `-S -P` runs that
/// never import `site` — reads it, matching the unconditional detection the
/// removed `sync_python_sys_path` performed. `sys` is not yet in `sys.modules`
/// during its own creation, so `ensure_stdlib_path`'s `add_sys_path` stages the
/// stdlib in the seed, and the flush below picks it up.
#[cfg(feature = "host_env")]
pub(crate) fn create_sys_path_list() -> PyObjectRef {
    #[cfg(not(target_arch = "wasm32"))]
    ensure_stdlib_path();
    let items: Vec<PyObjectRef> = SYS_PATH
        .lock()
        .unwrap()
        .iter()
        .map(|d| pyre_object::w_str_new(&d.to_string_lossy()))
        .collect();
    pyre_object::w_list_new(items)
}

/// Off-`host_env` builds have no native seed; `sys.path` starts empty.
#[cfg(not(feature = "host_env"))]
pub(crate) fn create_sys_path_list() -> PyObjectRef {
    pyre_object::w_list_new(vec![])
}

/// Extract a package module's `__path__` as filesystem directories.
///
/// Returns `None` when `__path__` is absent or is not a plain list of `str`.
/// `absolute_import` rejects the absent case up front ("'<parent>' is not a
/// package"), so a `None` reaching the caller means a package whose `__path__`
/// yielded no usable directories.
fn parent_package_path(parent: PyObjectRef) -> Option<Vec<PathBuf>> {
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(parent) };
    if w_dict.is_null() || !unsafe { pyre_object::is_dict(w_dict) } {
        return None;
    }
    let path_obj = unsafe { pyre_object::w_dict_getitem_str(w_dict, "__path__") }?;
    if path_obj.is_null() || !unsafe { pyre_object::is_list(path_obj) } {
        return None;
    }
    let n = unsafe { pyre_object::listobject::w_list_len(path_obj) };
    let mut dirs = Vec::with_capacity(n);
    for i in 0..n {
        if let Some(item) = unsafe { pyre_object::listobject::w_list_getitem(path_obj, i as i64) } {
            if unsafe { pyre_object::is_str(item) } {
                dirs.push(PathBuf::from(unsafe { pyre_object::w_str_get_value(item) }));
            }
        }
    }
    Some(dirs)
}

// ── parse_source_module ──────────────────────────────────────────────
// PyPy equivalent: importing.py `parse_source_module(space, pathname, source)`

fn parse_source_module(pathname: &str, source: &str) -> Result<CodeObject, String> {
    compile_source_with_filename(source, Mode::Exec, pathname).map_err(|e| e.to_string())
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
    w_globals: pyre_object::PyObjectRef,
    execution_context: *const PyExecutionContext,
    pathname: Option<&str>,
    cpathname: Option<&str>,
) -> Result<PyObjectRef, crate::PyError> {
    // importing.py:272-274 — setdefault('__builtins__', space.builtin).
    // `fresh_module_globals` already seeds `__builtins__` for module-shape
    // namespaces; the explicit setdefault here mirrors PyPy's defensive
    // call so callers that hand in a pre-built globals object (future
    // `_imp.exec_dynamic`-style entry) still inherit the builtins
    // pointer with no surprises.
    if unsafe { pyre_object::w_dict_getitem_str(w_globals, "__builtins__") }.is_none() {
        let ctx = unsafe { &*execution_context };
        let w_builtin = ctx.get_builtin();
        if !w_builtin.is_null() {
            unsafe {
                pyre_object::w_dict_setitem_str(w_globals, "__builtins__", w_builtin);
            }
        }
    }
    // importing.py:275-298 write_paths block.  Pyre callers always pass
    // `Some(pathname)` for source-file imports and `None` for the
    // `write_paths=False` shape (REPL, builtin bootstrap).
    if let Some(p) = pathname {
        // importing.py:284 setitem('__file__', w_pathname).
        let w_pathname = pyre_object::w_str_new(p);
        unsafe {
            pyre_object::w_dict_setitem_str(w_globals, "__file__", w_pathname);
        }
        // importing.py:285 setitem('__cached__', w_cpathname).  PyPy
        // surfaces `space.w_None` when `cpathname is None`, i.e. the
        // import was not satisfied from a `.pyc`.  Pyre has no .pyc
        // path today so reachable callers still hit the None arm.
        let w_cpathname = match cpathname {
            Some(c) => pyre_object::w_str_new(c),
            None => pyre_object::w_none(),
        };
        unsafe {
            pyre_object::w_dict_setitem_str(w_globals, "__cached__", w_cpathname);
        }
        // importing.py:286-298 — `_fix_up_module(d, name, pathname,
        // cpathname)` sets `__spec__`/`__loader__`/`__file__`/`__cached__`
        // from the app-level `SourceFileLoader` + `spec_from_file_location`
        // helpers.  Reachable only once the importlib bootstrap is wired;
        // before that, seed `__loader__`/`__spec__` with `None` only when
        // missing — the `if not loader / if not spec` guards at
        // `_bootstrap_external.py:_fix_up_module`.
        if !fix_up_source_module_spec(w_globals, p, cpathname)? {
            if unsafe { pyre_object::w_dict_getitem_str(w_globals, "__loader__") }.is_none() {
                unsafe {
                    pyre_object::w_dict_setitem_str(w_globals, "__loader__", pyre_object::w_none());
                }
            }
            if unsafe { pyre_object::w_dict_getitem_str(w_globals, "__spec__") }.is_none() {
                unsafe {
                    pyre_object::w_dict_setitem_str(w_globals, "__spec__", pyre_object::w_none());
                }
            }
        }
    }
    let code_ptr = Box::into_raw(Box::new(code));
    let w_code = crate::w_code_new(code_ptr as *const ());
    // importing.py:300 code_w.exec_code(space, w_dict, w_dict) → eval.py:31-33
    // Code.exec_code → space.createframe(...) + frame.run().  Surface
    // initialize_frame_scopes' freevar/closure mismatch (TypeError /
    // ValueError per pyframe.py:242-253) as PyError so the importer
    // reports it instead of panicking.  Route through run_with_jit so the
    // GENERATOR / COROUTINE / ASYNC_GENERATOR dispatch in
    // pyframe.py:268-273 holds for the import path too, and so an imported
    // module's top-level hot loop reaches the JIT portal.
    let mut frame =
        crate::pyframe::createframe_obj(w_code as *const (), w_globals, execution_context, None)?;
    frame.run_with_jit()
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
/// only.  Every function defined in `source` retains the intermediate
/// namespace as its `__globals__`; once copied into the caller's module,
/// those functions keep the namespace transitively reachable.
#[doc(hidden)]
pub trait AppleveldefNamespace {
    fn store(&mut self, name: &str, value: PyObjectRef);
}

impl AppleveldefNamespace for PyObjectRef {
    fn store(&mut self, name: &str, value: PyObjectRef) {
        crate::module_ns_store(*self, name, value);
    }
}

pub fn appleveldef_install(
    ns: impl AppleveldefNamespace,
    source: &str,
    filename: &str,
    names: &[&str],
) {
    appleveldef_install_seeded(ns, source, filename, names, &[]);
}

/// [`appleveldef_install`] with `seed` bound into the app namespace before the
/// source runs.
///
/// `mixedmodule.py:135 MixedModule.get` executes the sibling app file lazily, on
/// the first access to a name it defines, so by then the module is importable
/// and the file can reach its own interp-level types through a plain
/// `import _io` (`app_io.py:1`).  pyre installs app files eagerly from the
/// module initializer, before the module object exists, so a name the source
/// needs from its own module is bound up front instead.
pub fn appleveldef_install_seeded(
    mut ns: impl AppleveldefNamespace,
    source: &str,
    filename: &str,
    names: &[&str],
    seed: &[(&str, PyObjectRef)],
) {
    let code = compile_source_with_filename(source, Mode::Exec, filename)
        .unwrap_or_else(|e| panic!("appleveldef `{filename}`: compile failed — {e}"));
    let ctx = crate::call::getexecutioncontext();
    if ctx.is_null() {
        panic!("appleveldef `{filename}`: no execution context at module init");
    }
    let w_app_globals = unsafe { (*ctx).fresh_module_globals() };
    let _root = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_app_globals);
    for &(name, value) in seed {
        unsafe { pyre_object::w_dict_setitem_str(w_app_globals, name, value) };
    }
    let code_ptr = Box::into_raw(Box::new(code));
    let w_code = crate::w_code_new(code_ptr as *const ());
    let mut frame = crate::pyframe::createframe_obj(w_code as *const (), w_app_globals, ctx, None)
        .unwrap_or_else(|e| panic!("appleveldef `{filename}`: createframe — {e:?}"));
    if let Err(e) = frame.run_with_jit() {
        panic!("appleveldef `{filename}`: exec — {e:?}");
    }
    for &name in names {
        match unsafe { pyre_object::w_dict_getitem_str(w_app_globals, name) } {
            Some(val) => ns.store(name, val),
            None => panic!("appleveldef `{filename}`: name `{name}` not bound by source"),
        }
    }
}

// ── load_source_module ───────────────────────────────────────────────
// PyPy equivalent: importing.py `load_source_module()`
//
// Parse + execute a .py source file, producing a module object.

#[cfg(feature = "host_env")]
fn load_source_module(
    modulename: &str,
    pathname: &Path,
    package_dir: Option<&Path>,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    let source = with_source_provider(|p| p.read_to_string(pathname)).map_err(|e| {
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
    let w_globals = ctx.fresh_module_globals();
    let _root = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_globals);

    // PyPy `interpreter/module.py:Module.__init__` seeds `__name__` on
    // the module's w_dict.  `w_module_new_aliasing_dict` below does that
    // via `w_dict_setitem_str("__name__", ...)`, so an explicit store
    // here would be redundant.
    //
    // `__file__`/`__cached__` setting moved into `exec_code_module`
    // (`importing.py:284-285`) so the per-module attribute seeding
    // mirrors the PyPy call order.
    //
    // `__package__` is set by PyPy `interp_imp._prepare_module`
    // (`pypy/module/imp/interp_imp.py`); pyre has no `_prepare_module`
    // yet, so we still seed it here as a TODO until
    // the prepare-module path is ported.
    // A package's `__init__.py` is its own `__package__`; a plain module's
    // `__package__` is its containing package.
    let pkg = if package_dir.is_some() {
        modulename
    } else if let Some(dot) = modulename.rfind('.') {
        &modulename[..dot]
    } else {
        modulename
    };
    unsafe {
        pyre_object::w_dict_setitem_str(w_globals, "__package__", pyre_object::w_str_new(pkg));
    }

    // Seed `__path__` BEFORE executing the package body so relative imports
    // inside `__init__.py` (`from .sub import *`) resolve against the package
    // directory.  `_bootstrap` sets `__path__` on the module before
    // `exec_module`; setting it afterwards lets those imports fall through to
    // sys.path and pick up a same-leaf module from an unrelated package.
    if let Some(dir) = package_dir {
        let path_str = pyre_object::w_str_new(&dir.to_string_lossy());
        unsafe {
            pyre_object::w_dict_setitem_str(
                w_globals,
                "__path__",
                pyre_object::w_list_new(vec![path_str]),
            );
        }
    }

    // Create the module object BEFORE execution and register in sys.modules.
    // PyPy: load_source_module → set_sys_modules BEFORE exec_code_module.
    // This prevents infinite recursion on circular imports.
    let canonical = w_globals;
    let module = pyre_object::w_module_new_aliasing_dict(modulename, canonical);
    set_sys_module(modulename, module);

    // `_frozen_importlib`'s install() (moduledef.py:17-49) executes the two
    // bootstrap sources under their frozen names, so classes defined in them
    // capture `__module__` = the frozen name; importlib/__init__.py then renames
    // the modules to `importlib._bootstrap{,_external}` afterward.  Mirror that:
    // exec under the frozen `__name__`, restore it after (before install, whose
    // `sys.modules[__name__]` lookups need the real name).
    let frozen_exec_name = match modulename {
        "importlib._bootstrap" => Some("_frozen_importlib"),
        "importlib._bootstrap_external" => Some("_frozen_importlib_external"),
        _ => None,
    };
    if let Some(frozen) = frozen_exec_name {
        unsafe {
            pyre_object::w_dict_setitem_str(w_globals, "__name__", pyre_object::w_str_new(frozen));
        }
    }

    // PyPy `importing.py:300` passes `pathname`/`cpathname` to
    // `exec_code_module`; pyre has no .pyc cache today so cpathname is
    // always None, matching the PyPy `cpathname is None` arm at line
    // 282-283.
    //
    // On exec failure drop the pre-registered module from sys.modules
    // (`_bootstrap._load`) so a retried import re-runs the body instead of
    // observing a half-built module.
    if let Err(e) = exec_code_module(
        code,
        w_globals,
        execution_context,
        Some(&pathname_str),
        None,
    ) {
        remove_sys_module(modulename);
        return Err(e);
    }

    // Restore the public name now that class bodies have captured the frozen
    // `__module__`; `module.__name__` resolves from this dict entry.
    if frozen_exec_name.is_some() {
        unsafe {
            pyre_object::w_dict_setitem_str(
                w_globals,
                "__name__",
                pyre_object::w_str_new(modulename),
            );
        }
    }

    // Module-level code may have rewritten `sys.modules[name]` (the
    // `decimal` → `_pydecimal` pattern, or PyPy's `_cffi_backend` style
    // late rewiring). Honour that — PyPy: interp_import.importhook
    // reads sys.modules again after exec_code_module via importcache.
    if let Some(replaced) = check_sys_modules(modulename) {
        if !std::ptr::eq(replaced, module) {
            return Ok(replaced);
        }
    }

    // `_bootstrap` has no `sys` or `_imp` of its own until it is handed them,
    // so every entry point through it raises NameError until this runs. Read
    // the module back out of sys.modules rather than reusing the local: the
    // body just executed arbitrary code, and a collection in there relocates
    // a young module while only the dict entry is updated.
    if modulename == "importlib._bootstrap" {
        if let Some(loaded) = check_sys_modules(modulename) {
            if let Err(e) = install_importlib_bootstrap(loaded, execution_context) {
                // Unwind the partial install: `dunder_import` routes through
                // `_bootstrap.__import__` whenever `importlib._bootstrap` is
                // in `sys.modules`, and a half-installed bootstrap (module
                // registered, PathFinder missing — e.g. `_bootstrap_external`
                // needs the `nt` builtin on Windows) would then answer every
                // import with no file finder installed. Dropping the entries
                // keeps the native importer authoritative, the minimal-
                // importer role the boot sequence already documents.
                remove_sys_module(modulename);
                remove_sys_module("_frozen_importlib");
                remove_sys_module("_frozen_importlib_external");
                return Err(e);
            }
        }
    }

    Ok(module)
}

/// Wire a freshly executed `importlib._bootstrap` into this interpreter.
///
/// PyPy does the same from `_frozen_importlib`'s `install()`: `_install`
/// binds `sys` / `_imp` into the module globals and appends BuiltinImporter
/// and FrozenImporter to `sys.meta_path`, then
/// `_install_external_importers` imports `_bootstrap_external` — reached
/// through the `_frozen_importlib_external` alias `absolute_import` already
/// maps — and appends PathFinder.
///
/// PyPy runs it while building the space; doing it as the module finishes
/// loading keeps both bootstrap files off the startup path, since nothing
/// reads `sys.meta_path` before something has imported the machinery that
/// fills it.
#[cfg(feature = "host_env")]
fn install_importlib_bootstrap(
    module: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<(), crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let module_slot = shadow_stack_len();
    pin_root(module);

    let w_sys = absolute_import("sys", pyre_object::PY_NULL, execution_context)?;
    let sys_slot = shadow_stack_len();
    pin_root(w_sys);

    let w_imp = absolute_import("_imp", pyre_object::PY_NULL, execution_context)?;
    let imp_slot = shadow_stack_len();
    pin_root(w_imp);

    let install = crate::baseobjspace::getattr_str(shadow_stack_get(module_slot), "_install")?;
    let install_slot = shadow_stack_len();
    pin_root(install);
    let args = [shadow_stack_get(sys_slot), shadow_stack_get(imp_slot)];
    crate::call::call_function_impl_result(shadow_stack_get(install_slot), &args)?;

    let install_external = crate::baseobjspace::getattr_str(
        shadow_stack_get(module_slot),
        "_install_external_importers",
    )?;
    crate::call::call_function_impl_result(install_external, &[])?;

    if let Some(external) = check_sys_modules("_frozen_importlib_external") {
        set_frozen_alias_metadata(
            external,
            "_frozen_importlib_external",
            shadow_stack_get(module_slot),
        )?;
    }

    // `_install_external_importers` imports `_frozen_importlib_external`,
    // which aliases that name onto the loaded submodule; the bootstrap module
    // itself is only reached under its submodule name, so it never picks up
    // the matching alias. Register it once both installs have succeeded, so a
    // body that raised leaves no alias behind.
    set_sys_module("_frozen_importlib", shadow_stack_get(module_slot));
    set_frozen_alias_metadata(
        shadow_stack_get(module_slot),
        "_frozen_importlib",
        shadow_stack_get(module_slot),
    )?;

    // `sys.path_hooks.insert(0, zipimporter)` (zipimport moduledef startup /
    // pylifecycle.c init after the external importers) so zip archives on
    // `sys.path` are importable. `zipimport` is served from the frozen table
    // and its body imports `_frozen_importlib`, hence after the alias above.
    // A failed import leaves the hook out — the tolerant `# can't import
    // zipimport` path — rather than failing the whole bootstrap.
    if let Ok(w_zipimport) = absolute_import("zipimport", pyre_object::PY_NULL, execution_context) {
        let zipimport_slot = shadow_stack_len();
        pin_root(w_zipimport);
        let w_zipimporter =
            crate::baseobjspace::getattr_str(shadow_stack_get(zipimport_slot), "zipimporter")?;
        let zipimporter_slot = shadow_stack_len();
        pin_root(w_zipimporter);
        let w_path_hooks =
            crate::baseobjspace::getattr_str(shadow_stack_get(sys_slot), "path_hooks")?;
        unsafe {
            pyre_object::w_list_insert(w_path_hooks, 0, shadow_stack_get(zipimporter_slot));
        }
    }
    Ok(())
}

#[cfg(feature = "host_env")]
fn set_frozen_alias_metadata(
    module: PyObjectRef,
    name: &str,
    bootstrap: PyObjectRef,
) -> Result<(), crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let module_slot = shadow_stack_len();
    pin_root(module);
    let bootstrap_slot = shadow_stack_len();
    pin_root(bootstrap);

    let loader =
        crate::baseobjspace::getattr_str(shadow_stack_get(bootstrap_slot), "FrozenImporter")?;
    let loader_slot = shadow_stack_len();
    pin_root(loader);
    let find_spec = crate::baseobjspace::getattr_str(shadow_stack_get(loader_slot), "find_spec")?;
    let find_spec_slot = shadow_stack_len();
    pin_root(find_spec);
    let w_name = pyre_object::w_str_new(name);
    let name_slot = shadow_stack_len();
    pin_root(w_name);
    let spec = crate::call::call_function_impl_result(
        shadow_stack_get(find_spec_slot),
        &[shadow_stack_get(name_slot)],
    )?;
    let spec_slot = shadow_stack_len();
    pin_root(spec);

    crate::baseobjspace::setattr_str(
        shadow_stack_get(module_slot),
        "__loader__",
        shadow_stack_get(loader_slot),
    )?;
    crate::baseobjspace::setattr_str(
        shadow_stack_get(module_slot),
        "__spec__",
        shadow_stack_get(spec_slot),
    )?;
    Ok(())
}

// ── load_package ─────────────────────────────────────────────────────
// PyPy equivalent: load_module with PKG_DIRECTORY modtype

#[cfg(feature = "host_env")]
fn load_package(
    modulename: &str,
    dirpath: &Path,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // `__path__` / `__package__` are seeded in `load_source_module` BEFORE
    // the body runs (relative imports in `__init__.py` need them in place),
    // and `__init__.py` may legitimately rewrite `__path__` (namespace
    // packages via `pkgutil.extend_path`), so they are not re-stamped here.
    let init_path = dirpath.join("__init__.py");
    load_source_module(modulename, &init_path, Some(dirpath), execution_context)
}

// ── load_namespace_package ───────────────────────────────────────────
// PEP 420: a package directory (or set of directories) with no `__init__.py`.

#[cfg(feature = "host_env")]
fn load_namespace_package(
    modulename: &str,
    dirs: &[PathBuf],
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // A namespace package has no source to read or execute: it is a module
    // carrying `__path__` (the portions) and `__package__`, but no `__file__`.
    // Submodule imports resolve against `__path__` exactly as for a regular
    // package.
    let ctx = unsafe { &*execution_context };
    let w_globals = ctx.fresh_module_globals();
    let _root = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_globals);

    unsafe {
        pyre_object::w_dict_setitem_str(
            w_globals,
            "__package__",
            pyre_object::w_str_new(modulename),
        );
    }

    let path_items: Vec<PyObjectRef> = dirs
        .iter()
        .map(|d| pyre_object::w_str_new(&d.to_string_lossy()))
        .collect();
    unsafe {
        pyre_object::w_dict_setitem_str(w_globals, "__path__", pyre_object::w_list_new(path_items));
    }

    let module = pyre_object::w_module_new_aliasing_dict(modulename, w_globals);
    set_sys_module(modulename, module);
    Ok(module)
}

// ── load_part ────────────────────────────────────────────────────────
// PyPy equivalent: importing.py `load_part()`

fn load_part(
    modulename: &str,
    partname: &str,
    parent_dirs: Option<&[PathBuf]>,
    execution_context: *const PyExecutionContext,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    // A blocked name is answered before any search, so it also applies to a
    // name that was never importable to begin with.
    if sys_modules_blocks(modulename) {
        return Err(crate::PyError::module_not_found_with_name(
            format!("import of {modulename} halted; None in sys.modules"),
            modulename,
        ));
    }

    // Check sys.modules cache first
    if let Some(cached) = check_sys_modules(modulename) {
        return Ok(Some(cached));
    }

    // Try a full-name builtin match first so dotted stubs like
    // `importlib.machinery` can override the filesystem search.
    // PyPy: interp_import.importhook consults sys.builtin_module_names by
    // the fully-qualified name.
    let full_is_builtin = BUILTIN_MODULES.lock().unwrap().contains_key(modulename);
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
        startup_builtin_module(modulename, m, execution_context)?;
        // `startup_builtin_module` runs app-level spec construction that can
        // collect and relocate `m`; re-read the live pointer from sys.modules.
        let m = check_sys_modules(modulename).unwrap_or(m);
        return Ok(Some(m));
    }

    // Find the module on disk
    let find_info = find_module(partname, parent_dirs);
    let Some(info) = find_info else {
        return Ok(None);
    };

    let module = match info {
        #[cfg(feature = "host_env")]
        FindInfo::SourceFile { pathname } => {
            match load_source_module(modulename, &pathname, None, execution_context) {
                Ok(m) => m,
                Err(e) => {
                    return Err(e);
                }
            }
        }
        #[cfg(feature = "host_env")]
        FindInfo::Package { dirpath } => load_package(modulename, &dirpath, execution_context)?,
        #[cfg(feature = "host_env")]
        FindInfo::Namespace { dirs } => {
            load_namespace_package(modulename, &dirs, execution_context)?
        }
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
            startup_builtin_module(partname, m, execution_context)?;
            // `startup_builtin_module` may collect and relocate `m`; re-read
            // the live pointer from sys.modules.
            check_sys_modules(modulename).unwrap_or(m)
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
    // The frozen importlib bootstrap modules live on disk as the
    // `importlib._bootstrap{,_external}` submodules. A direct
    // `import _frozen_importlib` / `_frozen_importlib_external` (zipimport,
    // the runpy diagnostics) loads the corresponding submodule and, only once
    // it has been fully imported, aliases it under the frozen name. Registering
    // the alias after a successful import (rather than when the module is
    // pre-registered) means a body that raises during execution does not leave
    // a stale alias behind. The recursive call terminates: the submodule name
    // does not match.
    let frozen_target = match modulename {
        "_frozen_importlib" => Some("importlib._bootstrap"),
        "_frozen_importlib_external" => Some("importlib._bootstrap_external"),
        _ => None,
    };
    if let Some(target) = frozen_target {
        if let Some(cached) = check_sys_modules(modulename) {
            return Ok(cached);
        }
        absolute_import(target, pyre_object::PY_NULL, execution_context)?;
        if let Some(leaf) = check_sys_modules(target) {
            set_sys_module(modulename, leaf);
            return Ok(leaf);
        }
    }

    let parts: Vec<&str> = modulename.split('.').collect();
    let mut first: Option<PyObjectRef> = None;
    let mut parent: Option<PyObjectRef> = None;
    let mut prefix = Vec::new();

    for (level, &part) in parts.iter().enumerate() {
        prefix.push(part);
        let full_name = prefix.join(".");
        // A submodule is resolved against its parent package's `__path__`;
        // top-level names (level 0, no parent) search sys.path.
        //
        // `_bootstrap._find_and_load_unlocked` fixes the order: a sys.modules
        // hit for the full name wins first (`load_part` answers both the cached
        // entry and the blocked-name sentinel), and only then must the parent be
        // a package. Presence of `__path__` is the whole test — a PEP 420
        // `_NamespacePath` (or any other non-list value) still marks a package
        // even though `parent_package_path` cannot turn it into directories.
        // Without the check a dotted name whose parent is a plain module falls
        // back to the top-level search and can resolve to a same-leaf builtin.
        let parent_dirs = match parent {
            None => None,
            Some(_)
                if check_sys_modules(&full_name).is_some() || sys_modules_blocks(&full_name) =>
            {
                None
            }
            Some(parent_mod) => {
                if crate::baseobjspace::findattr_result(parent_mod, "__path__")?.is_none() {
                    let parent_name = parts[..level].join(".");
                    return Err(crate::PyError::module_not_found_with_name(
                        format!("No module named '{full_name}'; '{parent_name}' is not a package"),
                        &full_name,
                    ));
                }
                parent_package_path(parent_mod)
            }
        };
        let w_mod = load_part(&full_name, part, parent_dirs.as_deref(), execution_context)?;
        let Some(module) = w_mod else {
            // _bootstrap.py:1335 raises for the prefix that actually failed
            // (`name=name`): `import a.b.c` with `a.b` missing reports `a.b`.
            return Err(crate::PyError::module_not_found_with_name(
                format!("No module named '{full_name}'"),
                &full_name,
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
        crate::PyError::module_not_found_with_name(
            format!("No module named '{modulename}'"),
            modulename,
        )
    })
}

// ── IMPORT_NAME ──────────────────────────────────────────────────────

/// PyPy equivalent: pyopcode.py `IMPORT_NAME`.
pub fn import_name(
    frame: &mut PyFrame,
    name: &str,
    w_fromlist: PyObjectRef,
    w_flag: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let w_builtin = frame.get_builtin();
    let w_import = if !w_builtin.is_null() && unsafe { is_module(w_builtin) } {
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        if w_dict.is_null() {
            None
        } else {
            crate::baseobjspace::finditem_str(w_dict, "__import__")?
        }
    } else {
        None
    }
    .ok_or_else(|| crate::PyError::new(crate::PyErrorKind::ImportError, "__import__ not found"))?;

    let w_locals = match frame.getdebug() {
        Some(d) if !d.w_locals.is_null() => d.w_locals,
        _ => pyre_object::w_none(),
    };
    let w_globals = frame.get_w_globals();
    let w_modulename = pyre_object::w_str_new(name);

    crate::call::call_callable(
        frame,
        w_import,
        &[w_modulename, w_globals, w_locals, w_fromlist, w_flag],
    )
}

// ── __import__ ───────────────────────────────────────────────────────
// PyPy equivalent: _frozen_importlib/interp_import.py `interp___import__`

/// `_gcd_import` fast path: the already-imported module for `name`, after
/// waiting for another thread to finish initialising it.
///
/// PyPy's `interp_import.py:_gcd_import` performs the same `sys.modules` /
/// `__spec__._initializing` inspection.  CPython 3.14's
/// `import_ensure_initialized` supplies the missing concurrent-import tail:
/// call `_lock_unlock_module`, whose deadlock handler deliberately accepts a
/// partially initialised module for a concurrent circular import.  Re-read
/// `sys.modules` afterwards, as `PyImport_ImportModuleLevelObject` does, so an
/// import that failed and removed or replaced its module is retried by the
/// slow path instead of returning a stale object.
///
/// `None` means the slow path must run — a missing `sys.modules` entry, a
/// missing `__spec__`, or an entry removed/replaced while we waited.  A
/// `__spec__` without `_initializing` counts as initialised (a builtin module).
fn gcd_import_fast(name: &str) -> Result<Option<PyObjectRef>, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    // A `None` sentinel blocks the name; `check_sys_modules` skips it and
    // would fall back to the interpreter cache, resurrecting a builtin the
    // sentinel is meant to block.  Give up so the slow path raises
    // `import of {name} halted; None in sys.modules`.
    if sys_modules_blocks(name) {
        return Ok(None);
    }
    let Some(w_module) = check_sys_modules(name) else {
        return Ok(None);
    };
    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    pin_root(w_module);
    let Some(w_spec) =
        crate::baseobjspace::findattr_result(shadow_stack_get(mod_slot), "__spec__")?
    else {
        return Ok(None);
    };
    let spec_slot = shadow_stack_len();
    pin_root(w_spec);
    if let Some(w_initializing) =
        crate::baseobjspace::findattr_result(shadow_stack_get(spec_slot), "_initializing")?
    {
        if crate::baseobjspace::is_true(w_initializing)? {
            let Some(w_bootstrap) = get_sys_module("importlib._bootstrap") else {
                return Ok(None);
            };
            let bootstrap_slot = shadow_stack_len();
            pin_root(w_bootstrap);
            let Some(w_lock_unlock) = crate::baseobjspace::findattr_result(
                shadow_stack_get(bootstrap_slot),
                "_lock_unlock_module",
            )?
            else {
                return Ok(None);
            };
            let lock_unlock_slot = shadow_stack_len();
            pin_root(w_lock_unlock);
            let name_slot = shadow_stack_len();
            pin_root(pyre_object::w_str_new(name));
            crate::call::call_function_impl_result(
                shadow_stack_get(lock_unlock_slot),
                &[shadow_stack_get(name_slot)],
            )?;

            let Some(w_current) = check_sys_modules(name) else {
                return Ok(None);
            };
            if w_current != shadow_stack_get(mod_slot) {
                return Ok(None);
            }
        }
    }
    Ok(Some(shadow_stack_get(mod_slot)))
}

/// `interp_import.py:98` — `e.remove_traceback_module_frames('<frozen
/// importlib._bootstrap>', '<frozen importlib._bootstrap_external>', ...)`:
/// drop the leading traceback entries that belong to the importlib bootstrap
/// so an import error does not expose its internal `__import__` /
/// `_find_and_load` machinery. pyre runs the bootstrap from the on-disk
/// `importlib/_bootstrap{,_external}.py` sources, so match those filenames as
/// well as the frozen pseudo-names. Only leading (outermost, contiguous)
/// bootstrap frames are removed; a user frame stops the walk, keeping real
/// application frames intact.
fn strip_bootstrap_traceback_frames(mut err: crate::PyError) -> crate::PyError {
    use pyre_object::interp_exceptions::{w_exception_get_traceback, w_exception_set_traceback};

    fn is_bootstrap_filename(path: &str) -> bool {
        let norm = path.replace('\\', "/");
        norm.ends_with("importlib/_bootstrap.py")
            || norm.ends_with("importlib/_bootstrap_external.py")
            || norm == "<frozen importlib._bootstrap>"
            || norm == "<frozen importlib._bootstrap_external>"
    }

    let exc = err.to_exc_object();
    if exc.is_null() {
        return err;
    }
    unsafe {
        let mut tb = w_exception_get_traceback(exc);
        while !tb.is_null() && !is_none(tb) {
            let w_code = crate::pytraceback::w_pytraceback_get_w_code(tb);
            let is_bootstrap = !w_code.is_null()
                && crate::pycode::code_get_field(w_code, "co_filename")
                    .ok()
                    .filter(|f| pyre_object::is_str(*f))
                    .is_some_and(|f| is_bootstrap_filename(&pyre_object::w_str_get_value(f)));
            if !is_bootstrap {
                break;
            }
            tb = crate::pytraceback::w_pytraceback_get_w_next(tb);
        }
        w_exception_set_traceback(exc, tb);
    }
    err
}

/// `builtins.__import__` — `interp___import__`: a fast path answering
/// absolute imports from initialised `sys.modules` entries, the app-level
/// `_bootstrap.__import__` (the full `sys.meta_path` / `sys.path_hooks`
/// protocol) otherwise.  While the importlib bootstrap is not installed —
/// during startup, or when no stdlib is reachable — the native `importhook`
/// stands in, the role of PyPy's minimal `importing.py` importer.
pub fn dunder_import(
    name: &str,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    // Captured before any Python can run below (`is_true` may call a
    // `__bool__`); the raw argument pointers are stale after that.
    let fromlist_missing = w_fromlist.is_null() || unsafe { is_none(w_fromlist) };

    let _roots = push_roots();
    let globals_slot = shadow_stack_len();
    pin_root(if w_globals.is_null() {
        pyre_object::w_none()
    } else {
        w_globals
    });
    let locals_slot = shadow_stack_len();
    pin_root(if w_locals.is_null() {
        pyre_object::w_none()
    } else {
        w_locals
    });
    let fromlist_slot = shadow_stack_len();
    pin_root(if w_fromlist.is_null() {
        pyre_object::w_none()
    } else {
        w_fromlist
    });

    if level == 0 {
        // Fast path only for absolute imports (interp_import.py).
        // A package with a fromlist needs `_handle_fromlist`, which the
        // slow path runs.
        let have_fromlist =
            !fromlist_missing && crate::baseobjspace::is_true(shadow_stack_get(fromlist_slot))?;
        if let Some(w_mod) = gcd_import_fast(name)? {
            let mod_slot = shadow_stack_len();
            pin_root(w_mod);
            if !have_fromlist {
                match name.find('.') {
                    None => return Ok(shadow_stack_get(mod_slot)),
                    Some(dot) => {
                        // `import a.b` returns `a`; give up when the
                        // top-level ancestor is not initialised yet.
                        if let Some(w_top) = gcd_import_fast(&name[..dot])? {
                            return Ok(w_top);
                        }
                    }
                }
            } else if crate::baseobjspace::findattr_result(shadow_stack_get(mod_slot), "__path__")?
                .is_none()
            {
                return Ok(shadow_stack_get(mod_slot));
            }
        }
    }

    // The frozen bootstrap aliases stay on the native importer:
    // `_install_external_importers` imports `_frozen_importlib_external`
    // while installing PathFinder, so no finder can serve it yet —
    // `absolute_import` maps the alias onto the on-disk bootstrap sources.
    if matches!(name, "_frozen_importlib" | "_frozen_importlib_external") {
        return importhook(
            name,
            if w_globals.is_null() {
                pyre_object::PY_NULL
            } else {
                shadow_stack_get(globals_slot)
            },
            if w_fromlist.is_null() {
                pyre_object::PY_NULL
            } else {
                shadow_stack_get(fromlist_slot)
            },
            level,
            execution_context,
        );
    }

    // Slow path: the app-level `_bootstrap.__import__`.
    if let Some(w_bootstrap) = get_sys_module("importlib._bootstrap") {
        let bootstrap_slot = shadow_stack_len();
        pin_root(w_bootstrap);
        if let Some(w_import) =
            crate::baseobjspace::findattr_result(shadow_stack_get(bootstrap_slot), "__import__")?
        {
            let import_slot = shadow_stack_len();
            pin_root(w_import);
            // `PyImport_ImportModuleLevelObject`: a relative import with an
            // empty fromlist hands back the head package named by the *resolved
            // absolute name*, and the imported module itself when `name` carries
            // no dot.  `_bootstrap.__import__` reaches the same slice through
            // `module.__name__`, so a `sys.modules` entry that is not a module
            // raises AttributeError there instead of the KeyError the
            // interpreter entry point reports.
            if level > 0
                && !name.is_empty()
                && !(!fromlist_missing
                    && crate::baseobjspace::is_true(shadow_stack_get(fromlist_slot))?)
            {
                return relative_import_head(
                    shadow_stack_get(bootstrap_slot),
                    name,
                    shadow_stack_get(globals_slot),
                    level,
                );
            }
            let w_name = pyre_object::w_str_new(name);
            return call_bootstrap_import(
                shadow_stack_get(import_slot),
                w_name,
                shadow_stack_get(globals_slot),
                shadow_stack_get(locals_slot),
                if fromlist_missing {
                    pyre_object::PY_NULL
                } else {
                    shadow_stack_get(fromlist_slot)
                },
                level,
            );
        }
    }
    importhook(
        name,
        if w_globals.is_null() {
            pyre_object::PY_NULL
        } else {
            shadow_stack_get(globals_slot)
        },
        if w_fromlist.is_null() {
            pyre_object::PY_NULL
        } else {
            shadow_stack_get(fromlist_slot)
        },
        level,
        execution_context,
    )
}

/// Invoke the app-level `_bootstrap.__import__`.  A null `w_fromlist` means the
/// argument was omitted, which reaches `__import__` as its `()` default rather
/// than as `None` (interp_import.py `WrappedDefault(())`).
fn call_bootstrap_import(
    w_import: PyObjectRef,
    w_name: PyObjectRef,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let import_slot = shadow_stack_len();
    pin_root(w_import);
    let name_slot = shadow_stack_len();
    pin_root(w_name);
    let globals_slot = shadow_stack_len();
    pin_root(w_globals);
    let locals_slot = shadow_stack_len();
    pin_root(w_locals);
    let fromlist_slot = shadow_stack_len();
    pin_root(if w_fromlist.is_null() {
        pyre_object::w_tuple_new(vec![])
    } else {
        w_fromlist
    });
    let level_slot = shadow_stack_len();
    pin_root(pyre_object::w_int_new(level));
    crate::call::call_function_impl_result(
        shadow_stack_get(import_slot),
        &[
            shadow_stack_get(name_slot),
            shadow_stack_get(globals_slot),
            shadow_stack_get(locals_slot),
            shadow_stack_get(fromlist_slot),
            shadow_stack_get(level_slot),
        ],
    )
    .map_err(strip_bootstrap_traceback_frames)
}

/// `__import__` for a name with no `&str` spelling.
///
/// Every native lookup — `sys.modules`, the builtin registry, the frozen table
/// and the path search — is `&str`-keyed and can hold no such name, so the
/// app-level `_bootstrap.__import__` runs it with the name object as given,
/// which keeps the dotted-name, `level` and `fromlist` handling intact.
pub fn dunder_import_name_obj(
    w_name: PyObjectRef,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let name_slot = shadow_stack_len();
    pin_root(w_name);
    let globals_slot = shadow_stack_len();
    pin_root(if w_globals.is_null() {
        pyre_object::w_none()
    } else {
        w_globals
    });
    let locals_slot = shadow_stack_len();
    pin_root(if w_locals.is_null() {
        pyre_object::w_none()
    } else {
        w_locals
    });
    let fromlist_slot = shadow_stack_len();
    let w_fromlist = if w_fromlist.is_null() {
        pyre_object::w_none()
    } else {
        w_fromlist
    };
    pin_root(w_fromlist);

    let bootstrap = if let Some(w_bootstrap) = get_sys_module("importlib._bootstrap") {
        let bootstrap_slot = shadow_stack_len();
        pin_root(w_bootstrap);
        crate::baseobjspace::findattr_result(shadow_stack_get(bootstrap_slot), "__import__")?
    } else {
        None
    };
    let Some(w_import) = bootstrap else {
        // The bootstrap is not importable yet, and no native lookup can serve
        // this name, so it is reported missing.
        let repr = crate::display::format_wtf8_repr(unsafe {
            pyre_object::w_str_get_wtf8(shadow_stack_get(name_slot))
        });
        return Err(crate::PyError::module_not_found_with_name_obj(
            format!("No module named {repr}"),
            shadow_stack_get(name_slot),
        ));
    };
    call_bootstrap_import(
        w_import,
        shadow_stack_get(name_slot),
        shadow_stack_get(globals_slot),
        shadow_stack_get(locals_slot),
        shadow_stack_get(fromlist_slot),
        level,
    )
}

/// `PyImport_ImportModuleLevelObject`, the `level > 0` / empty-fromlist tail:
/// import `name` relative to `globals`, then return the head package named by
/// the absolute name the resolution produced.
///
/// `_bootstrap.__import__` spells the same slice as
/// `sys.modules[module.__name__[:len(module.__name__) - cut_off]]`, which
/// requires the loaded object to carry `__name__`.  A `sys.modules` entry that
/// is not a module has none, so the interpreter entry point slices the name it
/// resolved itself and reports a missing head as a KeyError.
fn relative_import_head(
    w_bootstrap: PyObjectRef,
    name: &str,
    w_globals: PyObjectRef,
    level: i64,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let bootstrap_slot = shadow_stack_len();
    pin_root(w_bootstrap);
    // `_bootstrap.__import__` — `globals_ = globals if globals is not None else {}`.
    let globals_slot = shadow_stack_len();
    pin_root(if w_globals.is_null() || unsafe { is_none(w_globals) } {
        pyre_object::w_dict_new()
    } else {
        w_globals
    });
    let name_slot = shadow_stack_len();
    pin_root(pyre_object::w_str_new(name));
    let level_slot = shadow_stack_len();
    pin_root(pyre_object::w_int_new(level));

    // `_bootstrap.__import__` — `package = _calc___package__(globals_)`.
    let w_calc =
        crate::baseobjspace::getattr_str(shadow_stack_get(bootstrap_slot), "_calc___package__")?;
    let calc_slot = shadow_stack_len();
    pin_root(w_calc);
    let w_package = crate::call::call_function_impl_result(
        shadow_stack_get(calc_slot),
        &[shadow_stack_get(globals_slot)],
    )
    .map_err(strip_bootstrap_traceback_frames)?;
    let package_slot = shadow_stack_len();
    pin_root(w_package);

    // `_bootstrap.__import__` — `module = _gcd_import(name, package, level)`.
    // `_sanity_check` / `_resolve_name` validate `package` and `level` in there,
    // so running it before the slice below keeps their diagnostics first.
    let w_gcd = crate::baseobjspace::getattr_str(shadow_stack_get(bootstrap_slot), "_gcd_import")?;
    let gcd_slot = shadow_stack_len();
    pin_root(w_gcd);
    let w_module = crate::call::call_function_impl_result(
        shadow_stack_get(gcd_slot),
        &[
            shadow_stack_get(name_slot),
            shadow_stack_get(package_slot),
            shadow_stack_get(level_slot),
        ],
    )
    .map_err(strip_bootstrap_traceback_frames)?;
    let module_slot = shadow_stack_len();
    pin_root(w_module);

    // No dot in `name`: the imported module is already the head.
    let Some(dot) = name.find('.') else {
        return Ok(shadow_stack_get(module_slot));
    };

    // `_resolve_name(name, package, level)` is the absolute name just imported;
    // it ends with `name`, so trimming from `name`'s first dot leaves the head.
    let w_resolve =
        crate::baseobjspace::getattr_str(shadow_stack_get(bootstrap_slot), "_resolve_name")?;
    let resolve_slot = shadow_stack_len();
    pin_root(w_resolve);
    let w_abs_name = crate::call::call_function_impl_result(
        shadow_stack_get(resolve_slot),
        &[
            shadow_stack_get(name_slot),
            shadow_stack_get(package_slot),
            shadow_stack_get(level_slot),
        ],
    )
    .map_err(strip_bootstrap_traceback_frames)?;
    let abs_slot = shadow_stack_len();
    pin_root(w_abs_name);
    // Owned: the slicing below outlives the allocations that follow it.
    let abs_name = crate::baseobjspace::utf8_w(shadow_stack_get(abs_slot))?.to_string();
    let cut_off = name.len() - dot;
    let head = abs_name
        .len()
        .checked_sub(cut_off)
        .and_then(|end| abs_name.get(..end))
        .unwrap_or(abs_name.as_str());

    let head_slot = shadow_stack_len();
    pin_root(pyre_object::w_str_new(head));
    // The raw `sys.modules` entry, `None` sentinel included: only a *missing*
    // key is the KeyError.
    let w_modules = sys_modules_dict();
    let found = if w_modules.is_null() {
        check_sys_modules(head)
    } else {
        unsafe {
            pyre_object::dictmultiobject::w_dict_lookup(w_modules, shadow_stack_get(head_slot))
        }
        .filter(|w| !w.is_null())
    };
    if let Some(w_found) = found {
        return Ok(w_found);
    }
    let head_repr = unsafe { crate::display::py_repr(shadow_stack_get(head_slot)) }?;
    Err(crate::PyError::key_error(format!(
        "{head_repr} not in sys.modules as expected"
    )))
}

// ── importhook ───────────────────────────────────────────────────────
// PyPy equivalent: importing.py `importhook()`

pub fn importhook(
    name: &str,
    w_globals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // CPython 3.14 import.c / PyPy's import-level validation: negative
    // levels are rejected independently of the module name.  An empty name
    // is valid only for an actual relative import (`from . import x`).
    if level < 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "level must be >= 0",
        ));
    }
    if name.is_empty() && level == 0 {
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
    // importlib._bootstrap._resolve_name: an empty fallback package (the
    // `__main__` case) has no parent to anchor even a level-1 import.
    if package.is_empty() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ImportError,
            "attempted relative import with no known parent package",
        ));
    }

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
    // Python 3.14 importlib._bootstrap._calc___package__: an explicit
    // non-None __package__ wins; otherwise __spec__.parent is authoritative.
    let package = crate::baseobjspace::finditem_str(w_globals, "__package__")?;
    let spec = crate::baseobjspace::finditem_str(w_globals, "__spec__")?;
    if let Some(pkg) = package {
        if !unsafe { pyre_object::is_none(pkg) } {
            if !unsafe { pyre_object::is_str(pkg) } {
                return Err(crate::PyError::type_error(
                    "__package__ not set to a string",
                ));
            }
            return Ok(Some(
                unsafe { pyre_object::w_str_get_value(pkg) }.to_string(),
            ));
        }
    }
    if let Some(spec) = spec {
        if !unsafe { pyre_object::is_none(spec) } {
            let parent = crate::baseobjspace::getattr_str(spec, "parent")?;
            if !unsafe { pyre_object::is_str(parent) } {
                return Err(crate::PyError::type_error(
                    "__spec__.parent is not a string",
                ));
            }
            return Ok(Some(
                unsafe { pyre_object::w_str_get_value(parent) }.to_string(),
            ));
        }
    }

    // _calc___package__ emits ImportWarning before the legacy __name__ /
    // __path__ fallback.  Route it through warnings.warn so assertWarns and
    // user warning filters see the event.
    crate::warn::warn_category(
        "can't resolve package from __spec__ or __package__, falling back on __name__ and __path__",
        "ImportWarning",
        2,
    )?;

    // Fallback: __name__ (for modules inside packages)
    if let Some(name_obj) = crate::baseobjspace::finditem_str(w_globals, "__name__")? {
        if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
            let name = unsafe { pyre_object::w_str_get_value(name_obj) };
            // If the module has a __path__, it's a package — use __name__ as-is
            if crate::baseobjspace::finditem_str(w_globals, "__path__")?.is_some() {
                return Ok(Some(name.to_string()));
            }
            // Otherwise `rpartition('.')[0]` is also the empty string for a
            // top-level module such as __main__.
            return Ok(Some(
                name.rfind('.').map_or("", |dot| &name[..dot]).to_string(),
            ));
        }
    }

    Ok(None)
}

// ── import_from ──────────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py `IMPORT_FROM`
//
// Get an attribute from the module on TOS. Like `space.getattr(w_module, w_name)`.

/// `importing.py:430 get_spec` — `space.getattr(w_module, '__spec__')`,
/// returning None when the module carries no `__spec__`.  Only a missing
/// attribute is suppressed; any other lookup error propagates.
pub(crate) fn get_spec(module: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    match crate::baseobjspace::getattr_str(module, "__spec__") {
        Ok(v) => Ok(v),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => Ok(pyre_object::w_none()),
        Err(e) => Err(e),
    }
}

/// `importing.py:438 is_spec_initializing` — a spec whose `_initializing`
/// flag is truthy marks a module still executing, the circular-import signal.
/// A missing `_initializing` reads as not initializing; any other lookup error,
/// and the truth test itself, propagate.
pub(crate) fn is_spec_initializing(w_spec: PyObjectRef) -> Result<bool, crate::PyError> {
    if unsafe { pyre_object::is_none(w_spec) } {
        return Ok(false);
    }
    let w_initializing = match crate::baseobjspace::getattr_str(w_spec, "_initializing") {
        Ok(v) => v,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => return Ok(false),
        Err(e) => return Err(e),
    };
    crate::baseobjspace::is_true(w_initializing)
}

/// `importing.py:452 is_spec_uninitialized_submodule` — the name appears in the
/// spec's `_uninitialized_submodules` list.  A missing attribute reads as not a
/// submodule; any other lookup error, and the containment test, propagate.
pub(crate) fn is_spec_uninitialized_submodule(
    w_spec: PyObjectRef,
    name: &str,
) -> Result<bool, crate::PyError> {
    if unsafe { pyre_object::is_none(w_spec) } {
        return Ok(false);
    }
    let w_value = match crate::baseobjspace::getattr_str(w_spec, "_uninitialized_submodules") {
        Ok(v) => v,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => return Ok(false),
        Err(e) => return Err(e),
    };
    crate::baseobjspace::contains(w_value, pyre_object::w_str_new(name))
}

/// `_PyModuleSpec_GetFileOrigin` — the spec's file origin: its `origin` string
/// when `has_location` is truthy and `origin` is a string, otherwise None.  A
/// missing `has_location` / `origin`, or a falsey `has_location`, yields None;
/// other lookup errors and the truth test propagate.
pub(crate) fn spec_file_origin(w_spec: PyObjectRef) -> Result<Option<PyObjectRef>, crate::PyError> {
    if unsafe { pyre_object::is_none(w_spec) } {
        return Ok(None);
    }
    // `has_location` is a property whose getter runs Python and allocates, so
    // pin the spec and read it back before the `origin` lookup.
    let _scope = pyre_object::gc_roots::push_roots();
    let spec_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_spec);
    let w_has_location = match crate::baseobjspace::getattr_str(w_spec, "has_location") {
        Ok(v) => v,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => return Ok(None),
        Err(e) => return Err(e),
    };
    if !crate::baseobjspace::is_true(w_has_location)? {
        return Ok(None);
    }
    let w_spec = pyre_object::gc_roots::shadow_stack_get(spec_slot);
    let w_origin = match crate::baseobjspace::getattr_str(w_spec, "origin") {
        Ok(v) => v,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => return Ok(None),
        Err(e) => return Err(e),
    };
    if !unsafe { pyre_object::is_str(w_origin) } {
        return Ok(None);
    }
    Ok(Some(w_origin))
}

/// `_PyModule_IsPossiblyShadowing` — whether a module loaded from `origin` could
/// be shadowing a same-named module later on the search path.  True when `-P`
/// is off and the file's directory equals the startup `sys.path[0]`; a package
/// `__init__.py` compares its parent directory instead.
pub(crate) fn is_possibly_shadowing(origin: &str) -> bool {
    if safe_path_flag() {
        return false;
    }
    let Some(sys_path_0) = SYS_PATH_0.lock().unwrap().clone() else {
        return false;
    };
    let sep = std::path::MAIN_SEPARATOR;
    // root = os.path.dirname(origin.removesuffix(os.sep + "__init__.py"))
    let mut root = origin.to_string();
    let Some(idx) = root.rfind(sep) else {
        return false;
    };
    if root[idx + 1..] == *"__init__.py" {
        root.truncate(idx);
        let Some(idx2) = root.rfind(sep) else {
            return false;
        };
        root.truncate(idx2);
    } else {
        root.truncate(idx);
    }
    root == sys_path_0
}

/// The shadowing classification for a module: its spec file origin (a path
/// string, when it has one), whether it is possibly shadowing a same-named
/// search-path module, and whether that shadowed module is a standard-library
/// one.  `w_name` is the module's `__name__` object, tested against
/// `sys.stdlib_module_names`.
pub(crate) fn module_shadow_info(
    w_spec: PyObjectRef,
    w_name: PyObjectRef,
) -> Result<(Option<String>, bool, bool), crate::PyError> {
    let origin = match spec_file_origin(w_spec)? {
        Some(o) => unsafe { pyre_object::w_str_get_value(o) }.to_string(),
        None => return Ok((None, false, false)),
    };
    if !is_possibly_shadowing(&origin) {
        return Ok((Some(origin), false, false));
    }
    // Shadowing a same-named search-path module; a standard-library name gets
    // the stronger hint.  `sys.stdlib_module_names` may be replaced or deleted
    // by user code — only a real set participates, and its membership test
    // (which hashes `__name__`) propagates.
    let mut shadowing_stdlib = false;
    if let Some(sys_mod) = get_sys_module("sys") {
        let _scope = pyre_object::gc_roots::push_roots();
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(w_name);
        let w_names = match crate::baseobjspace::getattr_str(sys_mod, "stdlib_module_names") {
            Ok(v) => v,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => pyre_object::w_none(),
            Err(e) => return Err(e),
        };
        if unsafe { pyre_object::is_set_or_frozenset(w_names) } {
            let w_name = pyre_object::gc_roots::shadow_stack_get(name_slot);
            shadowing_stdlib = crate::baseobjspace::contains(w_names, w_name)?;
        }
    }
    Ok((Some(origin), true, shadowing_stdlib))
}

pub fn import_from(
    module: PyObjectRef,
    name: &str,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // pyopcode.py:1127 import_from — first `space.getattr(w_module, w_name)`,
    // which honours the module attribute protocol (`__getattribute__` /
    // `__getattr__`).  Only an AttributeError falls through to the submodule
    // import below; any other error propagates.
    match crate::baseobjspace::getattr_str(module, name) {
        Ok(value) => return Ok(value),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
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
                    match importhook(
                        &fullname,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0,
                        execution_context,
                    ) {
                        Ok(_) => {
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
                        Err(e) => {
                            // A ModuleNotFoundError naming `fullname` itself
                            // means `name` is simply not a submodule, so fall
                            // through to the attribute-style "cannot import
                            // name".  Any other failure is a transitive import
                            // error inside the submodule and must propagate
                            // rather than be masked (`_handle_fromlist`).
                            let absent_submodule = e.kind
                                == crate::PyErrorKind::ModuleNotFoundError
                                && e.message.contains(&format!("'{fullname}'"));
                            if !absent_submodule {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }
    }

    // import_from Issue #17636 — a submodule already bound in sys.modules under
    // `<__name__>.<name>` is returned even when the parent object rejected the
    // attribute (e.g. a non-module stand-in with restrictive slots that
    // `_handle_fromlist` could not setattr onto). Read `__name__` off the object
    // rather than requiring it to be a module.
    if let Ok(w_name) = crate::baseobjspace::getattr_str(module, "__name__") {
        if unsafe { pyre_object::is_str(w_name) } {
            let modname = unsafe { pyre_object::w_str_get_value(w_name) };
            let fullname = format!("{modname}.{name}");
            if let Some(submod) = check_sys_modules(&fullname) {
                return Ok(submod);
            }
        }
    }

    // The name is neither a real attribute nor an importable submodule.
    // `error.py:new_import_error` — raise `ImportError(msg, name=pkgname,
    // path=pkgpath)`: `pkgname` is the package `__name__` (default "<unknown
    // module name>"), `pkgpath` its `__file__` (`get_path`, default "unknown
    // location").  pyopcode.py:1152 resolves the name via
    // `space.getattr(w_module, '__name__')` and importing.py:460-470 `get_path`
    // via `space.getattr(w_module, '__file__')`, so a descriptor- or
    // `__getattr__`-supplied value and a non-module `from` target are honored;
    // a missing / None path takes the default.
    let _roots = pyre_object::gc_roots::push_roots();
    let pkgname_slot = pyre_object::gc_roots::shadow_stack_len();
    let w_pkgname = match crate::baseobjspace::getattr_str(module, "__name__") {
        Ok(v) if unsafe { pyre_object::is_str(v) } => v,
        _ => pyre_object::w_str_new("<unknown module name>"),
    };
    // The `__file__` lookup below can run a descriptor or `__getattr__` and so
    // collect; the name string is young and movable, so pin it and read it
    // back from the slot afterwards.
    pyre_object::gc_roots::pin_root(w_pkgname);
    // pypy/module/imp/importing.py:460 get_path — a non-str `__file__`
    // (including None) reports the location as unknown.
    let w_pkgpath = match crate::baseobjspace::getattr_str(module, "__file__") {
        Ok(v) if unsafe { pyre_object::is_str(v) } => v,
        Ok(_) => pyre_object::w_str_new("unknown location"),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
            pyre_object::w_str_new("unknown location")
        }
        Err(e) => return Err(e),
    };
    // Pin the path string across the spec lookups below (which allocate) too.
    let pkgpath_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_pkgpath);
    let w_pkgname = pyre_object::gc_roots::shadow_stack_get(pkgname_slot);
    // Own the name so the spec lookups below (which allocate) cannot dangle it.
    let pkgname = unsafe { pyre_object::w_str_get_value(w_pkgname) }.to_string();
    // Classify the failure through `__spec__`: a same-named file shadowing a
    // search-path module is flagged first, then a module still executing
    // reports the circular-import cause, then an unset submodule slot.
    let w_spec = get_spec(module)?;
    let spec_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_spec);
    let w_pkgname = pyre_object::gc_roots::shadow_stack_get(pkgname_slot);
    let (origin, is_shadowing, is_shadowing_stdlib) = module_shadow_info(w_spec, w_pkgname)?;
    let w_spec = pyre_object::gc_roots::shadow_stack_get(spec_slot);
    let initializing = is_spec_initializing(w_spec)?;
    let w_spec = pyre_object::gc_roots::shadow_stack_get(spec_slot);
    let uninit_submodule = !initializing && is_spec_uninitialized_submodule(w_spec, &pkgname)?;
    let pkgpath =
        crate::baseobjspace::utf8_w(pyre_object::gc_roots::shadow_stack_get(pkgpath_slot))?;
    let origin = origin.as_deref().unwrap_or("");
    let msg = if is_shadowing_stdlib {
        format!(
            "cannot import name '{name}' from '{pkgname}' (consider renaming \
             '{origin}' since it has the same name as the standard library \
             module named '{pkgname}' and prevents importing that standard \
             library module)"
        )
    } else if initializing {
        if is_shadowing {
            format!(
                "cannot import name '{name}' from '{pkgname}' (consider renaming \
                 '{origin}' if it has the same name as a library you intended \
                 to import)"
            )
        } else {
            format!(
                "cannot import name '{name}' from partially initialized module \
                 '{pkgname}' (most likely due to a circular import) ({pkgpath})"
            )
        }
    } else if uninit_submodule {
        format!(
            "cannot access submodule '{pkgname}' of module '{name}' \
             (most likely due to a circular import)"
        )
    } else {
        format!("cannot import name '{name}' from '{pkgname}' ({pkgpath})")
    };
    let w_pkgname = pyre_object::gc_roots::shadow_stack_get(pkgname_slot);
    let w_pkgpath = pyre_object::gc_roots::shadow_stack_get(pkgpath_slot);
    Err(crate::PyError::import_error_name_path(
        msg, w_pkgname, w_pkgpath,
    ))
}

// ── import_all_from ──────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py:2221-2258 `import_all_from(module,
// into_locals)` (applevel function called by IMPORT_STAR).

fn type_name_for_err(w_obj: PyObjectRef) -> String {
    unsafe {
        match crate::typedef::r#type(w_obj) {
            Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
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

/// pypy/interpreter/pyopcode.py:2221-2258 `import_all_from` — applies each
/// public name to the locals mapping object via `space.setitem`.  Errors from
/// `__setitem__` propagate (a misbehaving mapping surfaces its TypeError /
/// KeyError to the caller).
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
    fn importhook_rejects_invalid_absolute_name_and_level() {
        crate::test_hooks::install_hash_hook();
        let empty = importhook("", PY_NULL, PY_NULL, 0, std::ptr::null()).unwrap_err();
        assert_eq!(empty.kind, crate::PyErrorKind::ValueError);
        assert_eq!(empty.message, "Empty module name");

        let negative = importhook("sys", PY_NULL, PY_NULL, -1, std::ptr::null()).unwrap_err();
        assert_eq!(negative.kind, crate::PyErrorKind::ValueError);
        assert_eq!(negative.message, "level must be >= 0");

        let globals = pyre_object::w_dict_new();
        unsafe {
            pyre_object::w_dict_setitem_str(globals, "__package__", pyre_object::w_str_new(""));
        }
        let no_parent = importhook("", globals, PY_NULL, 1, std::ptr::null()).unwrap_err();
        assert_eq!(no_parent.kind, crate::PyErrorKind::ImportError);
        assert_eq!(
            no_parent.message,
            "attempted relative import with no known parent package"
        );
    }

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
        let result = find_module("__nonexistent_pyre_test_module__", None);
        assert!(result.is_none());
    }

    #[cfg(feature = "wasm_vfs")]
    #[test]
    fn test_embedded_vfs_round_trips() {
        let mount = Path::new("/stdlib");
        let vfs = VfsProvider::from_blob(VFS_BLOB, mount);

        // `re` is a package: its `__init__.py` is a file and `re/` is a dir.
        assert!(vfs.is_file(&mount.join("re/__init__.py")));
        assert!(vfs.is_dir(&mount.join("re")));
        assert!(vfs.is_dir(mount));

        // A top-level module the closure pulls in.
        assert!(vfs.is_file(&mount.join("enum.py")));

        // Source is readable and non-empty; misses report NotFound.
        let src = vfs.read_to_string(&mount.join("re/__init__.py")).unwrap();
        assert!(src.contains("def compile"));
        assert!(vfs.read_to_string(&mount.join("re/_nope.py")).is_err());
        assert!(!vfs.is_file(&mount.join("re/_nope.py")));
    }
}
