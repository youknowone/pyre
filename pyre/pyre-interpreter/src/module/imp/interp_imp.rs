//! _imp implementation — PyPy: pypy/module/imp/interp_imp.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::importing::BUILTIN_MODULES;
use rustpython_wtf8::{Wtf8, Wtf8Buf};
use std::ffi::CString;
use std::sync::atomic::{AtomicI64, AtomicPtr, Ordering};
use std::sync::OnceLock;

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

/// The four-field prefix of CPython's public `struct _frozen` ABI consumed by
/// `ctypes.POINTER(...).in_dll(pythonapi, "_PyImport_Frozen...")`.
///
/// Pyre's canonical frozen-module owner above stores source rather than
/// marshalled CPython bytecode.  The ABI projection therefore exposes the
/// source bytes as its non-empty payload while preserving the observable
/// name/order/package table.  Import execution continues to go through
/// `frozen_code`, so this compatibility view cannot become a second semantic
/// owner of the modules.
#[repr(C)]
struct FrozenAbiEntry {
    name: *const std::ffi::c_char,
    code: *const u8,
    size: i32,
    is_package: i32,
}

fn build_frozen_abi_table(entries: &[FrozenModule]) -> usize {
    let mut table = Vec::with_capacity(entries.len() + 1);
    for entry in entries {
        let name = CString::new(entry.name)
            .expect("frozen module names contain no NUL bytes")
            .into_raw();
        let source = frozen_source(entry)
            .map(|(source, _)| source.into_bytes())
            .unwrap_or_else(|_| b"# frozen source unavailable\n".to_vec());
        let source = Box::leak(source.into_boxed_slice());
        let size = i32::try_from(source.len()).unwrap_or(i32::MAX);
        table.push(FrozenAbiEntry {
            name,
            code: source.as_ptr(),
            size,
            is_package: i32::from(entry.is_package),
        });
    }
    table.push(FrozenAbiEntry {
        name: std::ptr::null(),
        code: std::ptr::null(),
        size: 0,
        is_package: 0,
    });

    let table = Box::leak(table.into_boxed_slice());
    Box::leak(Box::new(table.as_ptr() as usize)) as *mut usize as usize
}

/// Address of the stable pointer variable exported by CPython for each frozen
/// table.  The split matches CPython's Bootstrap/Stdlib/Test ABI while the
/// concatenated order remains exactly `FROZEN_MODULES`, the list returned by
/// `_imp._frozen_module_names()`.
pub(crate) fn frozen_abi_pointer_variable(name: &str) -> Option<usize> {
    static BOOTSTRAP: OnceLock<usize> = OnceLock::new();
    static STDLIB: OnceLock<usize> = OnceLock::new();
    static TEST: OnceLock<usize> = OnceLock::new();

    match name {
        "_PyImport_FrozenBootstrap" => {
            Some(*BOOTSTRAP.get_or_init(|| build_frozen_abi_table(&FROZEN_MODULES[..3])))
        }
        "_PyImport_FrozenStdlib" => {
            Some(*STDLIB.get_or_init(|| build_frozen_abi_table(&FROZEN_MODULES[3..3])))
        }
        "_PyImport_FrozenTest" => {
            Some(*TEST.get_or_init(|| build_frozen_abi_table(&FROZEN_MODULES[3..])))
        }
        _ => None,
    }
}

static FROZEN_OVERRIDE: AtomicI64 = AtomicI64::new(0);

/// `importing.py:159 ImportRLock` — the interpreter's reentrant import lock.
///
/// Upstream mutates the three fields with the GIL held (`importing.py:175`
/// "this function runs with the GIL acquired so there is no race condition in
/// the creation of the lock"); pyre has no GIL, so `lock` is published with a
/// compare-exchange and `lockowner` is stored only after the real lock has
/// been taken.
struct ImportRLock {
    /// `importing.py:162 self.lock` — what `space.allocate_lock()` returned,
    /// allocated on the first acquire.  Null is upstream's `None`, and the
    /// distinction is observable: `release_lock` is silent while it is null.
    lock: AtomicPtr<crate::baseobjspace::Lock>,
    /// `importing.py:163 self.lockowner` — the owning thread, held as the
    /// ident rather than as the context object because only its identity is
    /// ever read.  Zero is `None`.
    lockowner: AtomicI64,
    /// `importing.py:164 self.lockcounter` — recursion depth.  Only the owning
    /// thread mutates it.
    lockcounter: AtomicI64,
}

/// `importing.py:168 me = self.space.getexecutioncontext()  # used as thread
/// ident`.
///
/// The comment states what the value is for, and that is what is ported: the
/// ident comes from `rthread.get_ident`, the same source `_thread.get_ident`
/// reads.  Taking the execution context instead would not survive the trip —
/// `space.getexecutioncontext()` returns the per-thread context
/// `threadlocals.get_ec()` creates once and caches for the thread's whole life
/// (`baseobjspace.py:741`), whereas pyre's accessor reads a slot the launcher
/// re-seeds with a fresh context per phase, so a lock taken while running a
/// script would be owned by a stranger once the REPL starts.
fn thread_ident() -> i64 {
    match crate::module::thread::current_ident() {
        // Zero is the `None` encoding, so it cannot also name a thread; fall
        // back to the same single-threaded sentinel `_thread.get_ident` uses.
        0 => 1,
        ident => ident,
    }
}

impl ImportRLock {
    const fn new() -> Self {
        Self {
            lock: AtomicPtr::new(std::ptr::null_mut()),
            lockowner: AtomicI64::new(0),
            lockcounter: AtomicI64::new(0),
        }
    }

    /// `importing.py:167 lock_held_by_someone_else`.  No caller upstream
    /// either; its `true` branch is additionally unreachable while pyre runs a
    /// single Python thread.
    #[allow(dead_code)]
    fn lock_held_by_someone_else(&self) -> bool {
        let me = thread_ident();
        let owner = self.lockowner.load(Ordering::Acquire);
        owner != 0 && owner != me
    }

    /// `importing.py:171 lock_held_by_anyone` — owner-agnostic, which is what
    /// makes `_imp.lock_held()` true when read from a non-owning thread.
    fn lock_held_by_anyone(&self) -> bool {
        self.lockowner.load(Ordering::Acquire) != 0
    }

    /// `importing.py:174 acquire_lock`.
    fn acquire_lock(&self) {
        // importing.py:177-181 — allocate on first use.  A losing racer drops
        // its own box and uses the winner's.
        if self.lock.load(Ordering::Acquire).is_null() {
            let fresh = crate::baseobjspace::allocate_lock();
            if self
                .lock
                .compare_exchange(
                    std::ptr::null_mut(),
                    fresh,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                drop(unsafe { Box::from_raw(fresh) });
            }
        }
        let me = thread_ident();
        // importing.py:183-189
        if self.lockowner.load(Ordering::Acquire) != me {
            let lock = unsafe { &*self.lock.load(Ordering::Acquire) };
            lock.acquire(true);
            debug_assert_eq!(self.lockowner.load(Ordering::Acquire), 0);
            debug_assert_eq!(self.lockcounter.load(Ordering::Relaxed), 0);
            self.lockowner.store(me, Ordering::Release);
        }
        // importing.py:190
        self.lockcounter.fetch_add(1, Ordering::Relaxed);
    }

    /// `importing.py:192 release_lock(silent_after_fork)`.
    fn release_lock(&self, silent_after_fork: bool) -> Result<(), crate::PyError> {
        let me = thread_ident();
        let owner = self.lockowner.load(Ordering::Acquire);
        if owner != me {
            // importing.py:195-198 — a fork() with the import lock held leaves
            // the child owner-less.  The `is None` conjunct is what keeps a
            // release by a *foreign* owner an error even here.
            if owner == 0 && silent_after_fork {
                return Ok(());
            }
            // importing.py:199-200 `if self.lock is None: # CannotHaveLock
            // occurred` is dropped for the same reason as its partner in
            // `acquire_lock`: it is the downstream half of a condition pyre
            // cannot occupy.  Porting it would make the branch live off a
            // different cause — an import path that never took the lock —
            // and then the very first unbalanced release of a process would
            // return silently instead of raising.  CONVERGENCE: have the
            // native importer bracket module execution with the lock the way
            // `importing.py:91 importhook` brackets its own, after which the
            // lock is allocated before user code runs and the branch stops
            // being observable either way.
            // importing.py:201-203
            return Err(crate::PyError::runtime_error(
                "not holding the import lock",
            ));
        }
        debug_assert!(self.lockcounter.load(Ordering::Relaxed) > 0);
        // importing.py:204-207 — clear the owner BEFORE releasing, so the
        // waiter that wakes up finds the state `acquire_lock` asserts.
        if self.lockcounter.fetch_sub(1, Ordering::Relaxed) - 1 == 0 {
            self.lockowner.store(0, Ordering::Release);
            unsafe { &*self.lock.load(Ordering::Acquire) }.release();
        }
        Ok(())
    }

    /// `importing.py:209 reinit_lock` — run in a fork child so it does not
    /// share the parent's lock.  Branches on the depth, not on ownership:
    /// a depth above one means the fork happened underneath an import, whose
    /// `before` hook already acquired.
    ///
    /// Registered upstream as the `child` fork hook
    /// (`pypy/module/imp/moduledef.py:45`, driven by
    /// `pypy/module/posix/interp_posix.py:1560`) and never exposed to Python.
    fn reinit_lock(&self) {
        if self.lockcounter.load(Ordering::Relaxed) > 1 {
            // importing.py:216-224 — the old lock object is abandoned rather
            // than freed: a foreign thread could be mid-acquire on it.
            let fresh = crate::baseobjspace::allocate_lock();
            self.lock.store(fresh, Ordering::Release);
            let me = thread_ident();
            unsafe { &*fresh }.acquire(true);
            self.lockowner.store(me, Ordering::Release);
            self.lockcounter.fetch_sub(1, Ordering::Relaxed);
        } else {
            self.lock.store(std::ptr::null_mut(), Ordering::Release);
            self.lockowner.store(0, Ordering::Release);
            self.lockcounter.store(0, Ordering::Relaxed);
        }
    }
}

/// `importing.py:228 getimportlock(space)` = `space.fromcache(ImportRLock)` —
/// one instance per interpreter, shared by every thread.  That sharing is what
/// makes `lockowner` meaningful, so this is a global and never a
/// `thread_local!`.
///
/// A static rather than a real per-space cache entry because
/// `ObjSpace::fromcache` re-runs `build` instead of memoizing it, and pyre runs
/// one space per process — the same reason `sys.modules` and the builtin-module
/// table are globals.  A second space in one process would need all of them
/// moved together, not this one alone.
static IMPORT_LOCK: ImportRLock = ImportRLock::new();

fn getimportlock() -> &'static ImportRLock {
    &IMPORT_LOCK
}

// The `space.config.objspace.usemodules.thread` gate on the four gateways
// below (`interp_imp.py:140`) is constant-true here: pyre's build always
// registers `_thread`, so the gateways stay ungated.

/// `interp_imp.py:139 lock_held`.
fn lock_held() -> bool {
    getimportlock().lock_held_by_anyone()
}

/// `interp_imp.py:146 acquire_lock`.
fn acquire_lock() {
    getimportlock().acquire_lock();
}

/// `interp_imp.py:150 release_lock` — `silent_after_fork=False`, which is why
/// an unbalanced `_imp.release_lock()` from Python raises.
fn release_lock() -> Result<(), crate::PyError> {
    getimportlock().release_lock(false)
}

/// `interp_imp.py:153 reinit_lock`.  Deliberately absent from the `_imp`
/// namespace: `moduledef.py:26` exposes only `lock_held`, `acquire_lock` and
/// `release_lock`.
fn reinit_lock() {
    getimportlock().reinit_lock();
}

/// `pypy/module/imp/moduledef.py:45-47` — the import module registers this
/// exact before/parent/child trio with the process fork-hook lists.  Keep the
/// gateways private to the interpreter: `_imp` exposes only the three public
/// lock operations above.
pub(crate) fn before_fork() {
    acquire_lock();
}

pub(crate) fn after_fork_parent() -> Result<(), crate::PyError> {
    // moduledef.py registers `interp_imp.release_lock`, whose gateway passes
    // `silent_after_fork=False`; only native importhook cleanup uses `True`.
    getimportlock().release_lock(false)
}

pub(crate) fn after_fork_child() {
    reinit_lock();
}

fn frozen_module(name: &Wtf8) -> Option<&'static FrozenModule> {
    FROZEN_MODULES
        .iter()
        .find(|entry| name == Wtf8::new(entry.name))
}

fn is_bootstrap_frozen(name: &str) -> bool {
    matches!(
        name,
        "_frozen_importlib" | "_frozen_importlib_external"
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

fn served_frozen_module(name: &Wtf8) -> Option<&'static FrozenModule> {
    frozen_module(name).filter(|entry| frozen_module_served(entry))
}

/// The module-name argument, kept in WTF-8.
///
/// `find_frozen` reads the name with `PyUnicode_AsUTF8` and treats one that
/// does not encode as `FROZEN_BAD_NAME`, clearing the error rather than
/// propagating it and reporting the status exactly as `FROZEN_NOT_FOUND` is
/// reported.  The frozen table is keyed by `&'static str`, so such a name
/// matches nothing; carrying it as WTF-8 keeps the code points for the `%R` in
/// that report instead of demanding a `&str` view the buffer cannot give.
fn frozen_name(
    args: &[pyre_object::PyObjectRef],
    function: &str,
) -> Result<Wtf8Buf, crate::PyError> {
    let Some(&name) = args.first() else {
        return Err(crate::PyError::type_error(format!(
            "{function} expected at least 1 argument, got 0"
        )));
    };
    if !unsafe { pyre_object::is_str(name) } {
        return Err(crate::PyError::type_error(format!(
            "{function}() argument 1 must be str, not {}",
            bad_argument_type_name(name)
        )));
    }
    Ok(unsafe { pyre_object::w_str_get_wtf8(name) }.to_owned())
}

/// `set_frozen_error` — both frozen diagnostics carry the module name as
/// `.name` with no `.path`, and render it the way `%R` does.
fn frozen_error(message: String, name: &Wtf8) -> crate::PyError {
    crate::PyError::import_error_name_path(
        message,
        pyre_object::w_str_from_wtf8(name.to_owned()),
        pyre_object::w_none(),
    )
}

/// `%R` on the module name: quote selection and escaping identical to
/// `repr(str)`, which Rust's `{:?}` does not reproduce (it always
/// double-quotes).
fn frozen_name_repr(name: &Wtf8) -> String {
    crate::display::format_wtf8_repr(name)
}

fn missing_frozen_error(name: &Wtf8) -> crate::PyError {
    frozen_error(
        format!("No such frozen object named {}", frozen_name_repr(name)),
        name,
    )
}

/// `set_frozen_error(FROZEN_INVALID)` — the frozen data was supplied by the
/// caller but does not unmarshal.
fn invalid_frozen_error(name: &Wtf8) -> crate::PyError {
    frozen_error(
        format!("Frozen object named {} is invalid", frozen_name_repr(name)),
        name,
    )
}

/// The type name `_PyArg_BadArgument` prints for a rejected argument: `None`
/// rather than `NoneType`.
fn bad_argument_type_name(obj: pyre_object::PyObjectRef) -> String {
    if unsafe { pyre_object::is_none(obj) } {
        return "None".to_string();
    }
    type_name(obj)
}

/// The receiver's type name, for argument-type error messages.
fn type_name(obj: pyre_object::PyObjectRef) -> String {
    match crate::typedef::r#type(obj) {
        Some(tp) => unsafe { pyre_object::w_type_get_name(tp.as_ptr()) }.to_string(),
        None => "object".to_string(),
    }
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

/// The module name as the extension-module loader spells it.
///
/// `_Py_ext_module_loader_info_init` encodes the name to ASCII because it has
/// to build the `PyInit_<name>` symbol from it, so a name outside ASCII is
/// rejected by the codec before the builtin registry is ever consulted.
fn ascii_module_name(w_name: pyre_object::PyObjectRef) -> Result<String, crate::PyError> {
    // Read straight off the buffer: reaching the name through `encode` would
    // run a `str` subclass's override, which decides which builtin is loaded.
    let name = unsafe { pyre_object::w_str_get_wtf8(w_name) };
    if let Some(pos) = name.code_points().position(|cp| cp.to_u32() > 127) {
        return Err(crate::typedef::unicode_encode_error(
            "ascii",
            w_name,
            pos,
            pos + 1,
            "ordinal not in range(128)",
        ));
    }
    Ok(name.to_string())
}

/// The code object a frozen table entry stands for.  A real frozen module ships
/// pre-marshalled bytes; pyre keeps the source and compiles it on demand, so
/// both `get_frozen_object` and the `withdata` arm of `find_frozen` come
/// through here and observe the same object.
fn frozen_code(entry: &FrozenModule) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let (source, code_name) = frozen_source(entry)?;
    // `_frozen_importlib._cached_compile`: recompiling the frozen sources (the
    // ~116 KB importlib bootstrap) on every startup is a large recurring cost,
    // so reload a marshalled code object from a source-validated cache when one
    // is present and recompile only on a miss.  A `Literal` source is trivial to
    // recompile and has no stdlib file backing, so only stdlib sources are cached.
    let cache_key = matches!(entry.source, FrozenSource::Stdlib(_)).then_some(entry.name);
    if let Some(key) = cache_key
        && let Some(code) = frozen_cache_load(key, &source)
    {
        return Ok(code);
    }
    let filename = format!("<frozen {code_name}>");
    let code = crate::compile::compile_source_with_filename(
        &source,
        crate::compile::Mode::Exec,
        &filename,
    )
    .map_err(|error| crate::builtins::compile_err_to_syntax_error(error, &source))?;
    let w_code = crate::w_code_new(Box::into_raw(Box::new(code)) as *const ());
    if let Some(key) = cache_key {
        // `frozen_cache_store` marshals `w_code`, which can allocate and collect;
        // keep the freshly boxed code reachable across that call.
        let _root = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_code);
        frozen_cache_store(key, &source, w_code);
    }
    Ok(w_code)
}

/// Bytecode/marshal version token stamped into the frozen cache header and
/// exposed as `_imp.pyc_magic_number_token`: the low half is the 3.14 magic
/// 3627, the high half the `\r\n` marker so the number breaks when read as
/// text.  A cache written by an interpreter with a different token is rejected.
pub(crate) const PYC_MAGIC_NUMBER_TOKEN: u32 = 0x0A0D_0E2B;

/// Per-process identity shared by every frozen-cache access: the stdlib
/// `__pycache__` directory, the executable name (segregates backends), and the
/// executable's modification time in ns (invalidates the cache across
/// rebuilds).  Computed once so `current_exe`, the `stat`, and
/// `detect_stdlib_path` each run a single time rather than once per lookup.
///
/// The cache is compiled out under `sandbox`: it needs `current_exe` (which the
/// host seam has no equivalent for) and writes into the stdlib `__pycache__`
/// (host FS mutation the seam forbids), so under sandbox the stubs below apply
/// and the bootstrap sources simply recompile each startup.
#[cfg(all(feature = "host_env", not(feature = "sandbox")))]
struct FrozenCacheBase {
    dir: std::path::PathBuf,
    exe_name: String,
    binary_mtime: u64,
}

#[cfg(all(feature = "host_env", not(feature = "sandbox")))]
fn frozen_cache_base() -> Option<&'static FrozenCacheBase> {
    use std::sync::OnceLock;
    static BASE: OnceLock<Option<FrozenCacheBase>> = OnceLock::new();
    BASE.get_or_init(|| {
        let exe = std::env::current_exe().ok()?;
        let exe_name = exe.file_name()?.to_str()?.to_owned();
        let modified = std::fs::metadata(&exe).ok()?.modified().ok()?;
        let binary_mtime =
            modified.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos() as u64;
        let dir = crate::importing::detect_stdlib_path()?.join("__pycache__");
        Some(FrozenCacheBase { dir, exe_name, binary_mtime })
    })
    .as_ref()
}

/// Cache file for a source module's marshalled code, colocated with the stdlib
/// bytecode cache.  `cache_key` names the entry (its frozen name, or the dotted
/// module name for the native bootstrap import).  The executable name segregates
/// backends and the optimize level segregates `-O`/`-OO` bytecode, so those
/// variants live in distinct files instead of overwriting one another.
#[cfg(all(feature = "host_env", not(feature = "sandbox")))]
pub(crate) fn frozen_cache_path(cache_key: &str) -> Option<std::path::PathBuf> {
    let base = frozen_cache_base()?;
    let optimize = crate::importing::optimize_flag();
    let stem = cache_key.replace(['.', '/', '\\'], "_");
    Some(base.dir.join(format!(
        "frozen.{stem}.{}.opt-{optimize}.marshalcache",
        base.exe_name
    )))
}

/// `_cached_compile` load half: return the cached marshalled code object when
/// the recorded version token, binary mtime, and full source all match.  Any
/// mismatch or I/O error yields `None`, so the caller recompiles.
#[cfg(all(feature = "host_env", not(feature = "sandbox")))]
pub(crate) fn frozen_cache_load(cache_key: &str, source: &str) -> Option<pyre_object::PyObjectRef> {
    let path = frozen_cache_path(cache_key)?;
    let mtime = frozen_cache_base()?.binary_mtime;
    let bytes = std::fs::read(&path).ok()?;
    // Header: [u32 magic][u64 binary_mtime][u64 source_len][source bytes][marshalled code].
    let magic = u32::from_le_bytes(bytes.get(0..4)?.try_into().ok()?);
    if magic != PYC_MAGIC_NUMBER_TOKEN {
        return None;
    }
    let stored_mtime = u64::from_le_bytes(bytes.get(4..12)?.try_into().ok()?);
    if stored_mtime != mtime {
        return None;
    }
    let src_len = u64::from_le_bytes(bytes.get(12..20)?.try_into().ok()?) as usize;
    let src_end = 20usize.checked_add(src_len)?;
    if bytes.get(20..src_end)? != source.as_bytes() {
        return None;
    }
    let code = crate::module::marshal::loads_bytes(bytes.get(src_end..)?).ok()?;
    unsafe { crate::is_code(code) }.then_some(code)
}

/// `_cached_compile` store half (best effort): write the marshalled code with
/// the validating header.  I/O failures (e.g. a read-only stdlib) are ignored —
/// the next startup simply recompiles.
///
/// `-B` / PYTHONDONTWRITEBYTECODE does **not** suppress this, even though the
/// file lands in `__pycache__`.  A frozen module is marshalled into the binary
/// at build time and never recompiled; pyre keeps the source and caches the
/// compile instead, so this file stands in for that build step rather than for
/// an import-time `.pyc`.  Letting `-B` reach it would make the flag force a
/// recompile of the ~116 KB bootstrap on every startup, which nothing else
/// does.
#[cfg(all(feature = "host_env", not(feature = "sandbox")))]
pub(crate) fn frozen_cache_store(cache_key: &str, source: &str, code: pyre_object::PyObjectRef) {
    let Some(path) = frozen_cache_path(cache_key) else {
        return;
    };
    let Some(base) = frozen_cache_base() else {
        return;
    };
    let Ok(marshalled) = crate::module::marshal::dumps_bytes(code) else {
        return;
    };
    let mut buf = Vec::with_capacity(20 + source.len() + marshalled.len());
    buf.extend_from_slice(&PYC_MAGIC_NUMBER_TOKEN.to_le_bytes());
    buf.extend_from_slice(&base.binary_mtime.to_le_bytes());
    buf.extend_from_slice(&(source.len() as u64).to_le_bytes());
    buf.extend_from_slice(source.as_bytes());
    buf.extend_from_slice(&marshalled);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&path, &buf);
}

#[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
pub(crate) fn frozen_cache_load(_cache_key: &str, _source: &str) -> Option<pyre_object::PyObjectRef> {
    None
}

#[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
pub(crate) fn frozen_cache_store(_cache_key: &str, _source: &str, _code: pyre_object::PyObjectRef) {}

/// The `data` element of a `withdata=True` `find_frozen` result: a read-only
/// `memoryview` over the frozen bytes, so `marshal.loads(bytes(data))`
/// reconstructs the same code object `get_frozen_object` returns.
fn frozen_data(entry: &FrozenModule) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let code = frozen_code(entry)?;
    let bytes = crate::module::marshal::dumps_bytes(code)?;
    let w_bytes = pyre_object::bytesobject::w_bytes_from_bytes(&bytes);
    let _roots = pyre_object::gc_roots::push_roots();
    let bytes_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_bytes);
    let mv_type = crate::typedef::gettypeobject(&pyre_object::memoryview::MEMORYVIEW_TYPE);
    crate::module::_pickle::call_fn(
        mv_type,
        &[pyre_object::gc_roots::shadow_stack_get(bytes_slot)],
    )
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
                // The name is compared against every `_inittab` entry with
                // `_PyUnicode_EqualToASCIIString`, which never decodes;
                // `BUILTIN_MODULES` is `&'static str`-keyed and cannot hold a
                // lone surrogate either, so such a name is not a builtin.
                let name = unsafe {
                    if pyre_object::is_str(args[0]) {
                        match pyre_object::w_str_get_value_opt(args[0]) {
                            Some(name) => name,
                            None => return Ok(pyre_object::w_int_new(0)),
                        }
                    } else {
                        return Ok(pyre_object::w_int_new(0));
                    }
                };
                // `import.c is_builtin`: 0 = not a builtin, -1 = an inittab
                // entry whose `initfunc` is NULL and so cannot be
                // (re)initialized, 1 = every other builtin.  `sys` and
                // `builtins` are the two NULL-initfunc entries.
                // `interp_imp.py is_builtin` instead answers -1 for *any*
                // builtin already in `sys.modules`, which makes an ordinary
                // imported builtin such as `time` report -1.
                let is_builtin = BUILTIN_MODULES.lock().unwrap().contains_key(name);
                let result = if !is_builtin {
                    0
                } else if matches!(name, "sys" | "builtins") {
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
            // `import.c _imp_init_frozen_impl` — run the frozen module's code
            // in a fresh namespace registered under its name and hand the
            // module back, or None when the name is not frozen.  A name
            // already in sys.modules keeps its module.
            // `interp_imp.py:74 init_frozen` instead always answers None,
            // leaving frozen modules to the meta path.
            |args| {
                let name = frozen_name(args, "init_frozen")?;
                let Some(entry) = served_frozen_module(&name) else {
                    return Ok(pyre_object::w_none());
                };
                // A frozen name is ASCII by construction (the table's keys),
                // so the lossy view is the name itself.
                let name = name.to_string_lossy().into_owned();
                if let Some(module) = crate::importing::get_sys_module(&name) {
                    return Ok(module);
                }
                let code = frozen_code(entry)?;
                let _roots = pyre_object::gc_roots::push_roots();
                let code_slot = pyre_object::gc_roots::shadow_stack_len();
                pyre_object::gc_roots::pin_root(code);
                let ec = crate::call::getexecutioncontext();
                let w_globals = unsafe { &*ec }.fresh_module_globals();
                let globals_slot = pyre_object::gc_roots::shadow_stack_len();
                pyre_object::gc_roots::pin_root(w_globals);
                // The name string is allocated before the store so the mapping
                // it writes into is read after that allocation, not before it.
                let w_name = pyre_object::w_str_new(&name);
                unsafe {
                    pyre_object::w_dict_setitem_str(
                        pyre_object::gc_roots::shadow_stack_get(globals_slot),
                        "__name__",
                        w_name,
                    );
                };
                // `PyImport_ImportFrozenModuleObject` leaves the source-table
                // name for `FrozenImporter._fix_up_module`, which consumes it
                // when constructing loader_state.  This is load-bearing when
                // the 3.14 tests import a fresh source copy of importlib and
                // its `_setup` repairs already-loaded frozen modules.
                let w_origname = pyre_object::w_str_new(entry.origname.unwrap_or(entry.name));
                unsafe {
                    pyre_object::w_dict_setitem_str(
                        pyre_object::gc_roots::shadow_stack_get(globals_slot),
                        "__origname__",
                        w_origname,
                    );
                }
                // `PyImport_ImportFrozenModuleObject`: a frozen *package* gets
                // `__path__` set to the empty list before its code runs, which
                // is what makes `import __phello__.spam` resolve through it
                // instead of reporting that `__phello__` is not a package.
                if entry.is_package {
                    let w_path = pyre_object::w_list_new(Vec::new());
                    unsafe {
                        pyre_object::w_dict_setitem_str(
                            pyre_object::gc_roots::shadow_stack_get(globals_slot),
                            "__path__",
                            w_path,
                        );
                    };
                }
                let module_slot = pyre_object::gc_roots::shadow_stack_len();
                pyre_object::gc_roots::pin_root(pyre_object::w_module_new_aliasing_dict(
                    &name,
                    pyre_object::gc_roots::shadow_stack_get(globals_slot),
                ));
                // `set_sys_module` inserts into `sys.modules` and so allocates:
                // publish the module from its rooted slot rather than from a
                // pointer captured before the insert.
                crate::importing::set_sys_module(
                    &name,
                    pyre_object::gc_roots::shadow_stack_get(module_slot),
                );
                if let Err(error) = crate::builtins::builtin_exec(&[
                    pyre_object::gc_roots::shadow_stack_get(code_slot),
                    pyre_object::gc_roots::shadow_stack_get(globals_slot),
                ]) {
                    crate::importing::remove_sys_module(&name);
                    return Err(error);
                }
                Ok(pyre_object::gc_roots::shadow_stack_get(module_slot))
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
        // `withdata` is keyword-only, so no call shape fills every parameter
        // positionally and there is no fixed natural arity to fast-path on.
        crate::make_builtin_function("find_frozen", |args| {
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(kwargs, &["withdata"], "find_frozen")?;
            if positional.len() != 1 {
                return Err(crate::PyError::type_error(format!(
                    "find_frozen() takes exactly 1 positional argument ({} given)",
                    positional.len()
                )));
            }
            let name = frozen_name(positional, "find_frozen")?;
            // `withdata: bool(accept={int})` — any object, read for truth.
            let withdata = match crate::builtins::kwarg_get(kwargs, "withdata") {
                Some(value) => crate::baseobjspace::is_true(value)?,
                None => false,
            };
            let Some(entry) = served_frozen_module(&name) else {
                return Ok(pyre_object::w_none());
            };
            let _roots = pyre_object::gc_roots::push_roots();
            let data_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(if withdata {
                frozen_data(entry)?
            } else {
                pyre_object::w_none()
            });
            let origname_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(
                entry
                    .origname
                    .map(pyre_object::w_str_new)
                    .unwrap_or_else(pyre_object::w_none),
            );
            Ok(pyre_object::w_tuple_new(vec![
                pyre_object::gc_roots::shadow_stack_get(data_slot),
                pyre_object::w_bool_from(entry.is_package),
                pyre_object::gc_roots::shadow_stack_get(origname_slot),
            ]))
        }),
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
        "_override_multi_interp_extensions_check",
        // Overrides `PyInterpreterConfig.check_multi_interp_extensions` for a
        // subinterpreter; the main interpreter is refused outright.  pyre runs
        // one interpreter, so every call takes the refusing arm — the override
        // has no state to keep.  The int conversion runs first, so a non-int
        // argument still reports the argument error rather than this one.
        crate::make_builtin_function_with_arity(
            "_override_multi_interp_extensions_check",
            |args| {
                crate::baseobjspace::gateway_int_w(args[0])?;
                Err(crate::PyError::runtime_error(
                    "_imp._override_multi_interp_extensions_check() cannot be used \
                     in the main interpreter",
                ))
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "get_frozen_object",
        // `data` is optional, so there is no fixed natural arity to fast-path
        // on; registering one would declare a call shape the closure does not
        // actually require.
        crate::make_builtin_function("get_frozen_object", |args| {
                let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
                if crate::builtins::has_real_kwargs(kwargs) {
                    return Err(crate::PyError::type_error(
                        "_imp.get_frozen_object() takes no keyword arguments",
                    ));
                }
                if positional.len() > 2 {
                    return Err(crate::PyError::type_error(format!(
                        "get_frozen_object expected at most 2 arguments, got {}",
                        positional.len()
                    )));
                }
                let name = frozen_name(positional, "get_frozen_object")?;
                // `_imp_get_frozen_object_impl`: a `data` buffer stands in for
                // the frozen table entry, so those bytes are unmarshalled
                // directly and a stream that does not decode reports the object
                // as invalid rather than as missing.
                let data = positional
                    .get(1)
                    .copied()
                    .filter(|&data| !unsafe { pyre_object::is_none(data) });
                if let Some(data) = data {
                    let Some(buffer) = crate::typedef::buffer_as_bytes_like(data)? else {
                        return Err(crate::PyError::type_error(format!(
                            "get_frozen_object() argument 2 must be bytes, not {}",
                            bad_argument_type_name(data)
                        )));
                    };
                    // Owned before the unmarshal allocates: the buffer may be a
                    // freshly minted bytes that nothing roots.
                    let bytes =
                        unsafe { pyre_object::bytesobject::bytes_like_data(buffer) }.to_vec();
                    let code = crate::module::marshal::loads_bytes(&bytes)
                        .map_err(|_| invalid_frozen_error(&name))?;
                    if !unsafe { crate::is_code(code) } {
                        return Err(crate::PyError::type_error(format!(
                            "frozen object {} is not a code object",
                            frozen_name_repr(&name)
                        )));
                    }
                    return Ok(code);
                }
                let entry =
                    served_frozen_module(&name).ok_or_else(|| missing_frozen_error(&name))?;
                frozen_code(entry)
            }),
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
                let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
                if crate::builtins::has_real_kwargs(kwargs) {
                    return Err(crate::PyError::type_error(
                        "_imp.create_builtin() takes no keyword arguments",
                    ));
                }
                if positional.len() != 1 {
                    return Err(crate::PyError::type_error(format!(
                        "_imp.create_builtin() takes exactly one argument ({} given)",
                        positional.len()
                    )));
                }
                let spec = positional[0];
                let w_name = crate::baseobjspace::getattr_str(spec, "name")?;
                if !unsafe { pyre_object::is_str(w_name) } {
                    return Err(crate::PyError::type_error(format!(
                        "name must be string, not {}",
                        type_name(w_name)
                    )));
                }
                let name = ascii_module_name(w_name)?;
                if name.as_bytes().contains(&0) {
                    return Err(crate::PyError::value_error("embedded null character"));
                }
                if let Some(module) = crate::importing::get_sys_module(&name) {
                    return Ok(module);
                }
                // A name that is spelled correctly but names no builtin is
                // reported by returning None, leaving the diagnostic to
                // `BuiltinImporter.create_module`, which has already screened
                // the name against `sys.builtin_module_names`.
                Ok(
                    crate::importing::create_builtin_module(
                        &name,
                        crate::call::getexecutioncontext(),
                    )?
                    .unwrap_or_else(pyre_object::w_none),
                )
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
            |_args| {
                #[cfg(all(
                    feature = "cpyext",
                    not(feature = "sandbox"),
                    any(target_os = "macos", target_os = "linux")
                ))]
                {
                    crate::cpyext::exec_dynamic(_args[0])
                }
                #[cfg(not(all(
                    feature = "cpyext",
                    not(feature = "sandbox"),
                    any(target_os = "macos", target_os = "linux")
                )))]
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::module_ns_store(
        ns,
        "create_dynamic",
        // interp_imp.py:49 create_dynamic. Without the `cpyext` feature this is
        // the `has_so_extension() == False` branch, which is the default build:
        // the spec's `name` and `origin` are read and rejected for an embedded
        // null before reporting the unsupported load, matching
        // `_imp_create_dynamic_impl`. Raising ImportError (rather than being
        // absent) matches the meta-path `hasattr` probe while still letting
        // `except ImportError` fall back to a pure-Python module.
        crate::make_builtin_function_with_signature(
            "create_dynamic",
            |args| {
                // `file` is bound and dropped: the loader opens the library by
                // path, so the already-open stream has no use here. Declaring
                // it is what keeps a two-argument call from being rejected.
                let spec = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                if spec.is_null() {
                    return Err(crate::PyError::type_error(
                        "create_dynamic() missing required argument 'spec'",
                    ));
                }
                #[cfg(all(
                    feature = "cpyext",
                    not(feature = "sandbox"),
                    any(target_os = "macos", target_os = "linux")
                ))]
                {
                    crate::cpyext::create_dynamic(spec)
                }
                #[cfg(not(all(
                    feature = "cpyext",
                    not(feature = "sandbox"),
                    any(target_os = "macos", target_os = "linux")
                )))]
                {
                    crate::baseobjspace::text0_wtf8_w(crate::baseobjspace::getattr_str(
                        spec, "name",
                    )?)?;
                    crate::baseobjspace::text0_wtf8_w(crate::baseobjspace::getattr_str(
                        spec, "origin",
                    )?)?;
                    Err(crate::PyError::new(
                        crate::PyErrorKind::ImportError,
                        "Not implemented".to_string(),
                    ))
                }
            },
            crate::Signature::new(vec!["spec", "file"], None, None, 0, 0),
        ),
    );
    crate::module_ns_store(
        ns,
        "acquire_lock",
        crate::make_builtin_function_with_arity(
            "acquire_lock",
            |_| {
                acquire_lock();
                Ok(pyre_object::w_none())
            },
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "release_lock",
        crate::make_builtin_function_with_arity(
            "release_lock",
            |_| {
                release_lock()?;
                Ok(pyre_object::w_none())
            },
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "lock_held",
        crate::make_builtin_function_with_arity(
            "lock_held",
            |_| Ok(pyre_object::w_bool_from(lock_held())),
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "_fix_co_filename",
        // interp_imp.py:157 fix_co_filename(code, pathname).
        crate::make_builtin_function_with_arity(
            "_fix_co_filename",
            |args| {
                if !unsafe { crate::is_code(args[0]) } {
                    return Err(crate::PyError::type_error(format!(
                        "_fix_co_filename() argument 1 must be code, not {}",
                        type_name(args[0])
                    )));
                }
                if !unsafe { pyre_object::is_str(args[1]) } {
                    return Err(crate::PyError::type_error(format!(
                        "_fix_co_filename() argument 2 must be str, not {}",
                        type_name(args[1])
                    )));
                }
                // `interp_imp.py:157 pathname='fsencode'`: preserve the raw
                // filesystem spelling before storing it on the code object.
                let newname = crate::gateway::fsencode_bytes_w(args[1])?;
                unsafe { crate::pycode::fix_co_filename(args[0], &newname) };
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::module_ns_store(
        ns,
        "extension_suffixes",
        crate::make_builtin_function_with_arity(
            "extension_suffixes",
            |_| {
                #[cfg(all(
                    feature = "cpyext",
                    not(feature = "sandbox"),
                    any(target_os = "macos", target_os = "linux")
                ))]
                {
                    Ok(pyre_object::w_list_new(vec![pyre_object::w_str_new(
                        crate::cpyext::extension_suffix(),
                    )]))
                }
                #[cfg(not(all(
                    feature = "cpyext",
                    not(feature = "sandbox"),
                    any(target_os = "macos", target_os = "linux")
                )))]
                {
                    Ok(pyre_object::w_list_new(vec![]))
                }
            },
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "get_tag",
        // PyPy `interp_imp.py:get_tag`: the cache tag for .pyc files.  Keep
        // this identical to sys.implementation.cache_tag.
        crate::make_builtin_function_with_arity(
            "get_tag",
            |_| Ok(pyre_object::w_str_new("pyre314")),
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "source_hash",
        crate::make_builtin_function_with_arity(
            "source_hash",
            |args| {
                // `_imp_source_hash_impl` hashes the source bytes with
                // `_Py_KeyedHash`, which is siphash-1-3 keyed by the pyc magic
                // (k0=magic, k1=0) and serialized low-byte-first — the 8-byte
                // hash field of hash-based pycs (`_code_to_hash_pyc` asserts
                // `len(source_hash) == 8`).  The pyc header this fills already
                // carries the 3.14 magic number, so the digest has to agree
                // with the one that format specifies.
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
                let mut hasher = siphasher::sip::SipHasher13::new_with_keys(magic, 0);
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
    // (_bootstrap_external.py).  Cache files are already segregated by
    // `sys.implementation.cache_tag`.
    crate::module_ns_store(
        ns,
        "pyc_magic_number_token",
        pyre_object::w_int_new(i64::from(PYC_MAGIC_NUMBER_TOKEN)),
    );
}
