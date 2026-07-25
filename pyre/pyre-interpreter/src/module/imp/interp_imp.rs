//! _imp implementation — PyPy: pypy/module/imp/interp_imp.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::importing::BUILTIN_MODULES;
use std::sync::atomic::{AtomicI64, AtomicPtr, Ordering};

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
    /// pyre has no fork-hook dispatch, so nothing calls this yet.
    /// CONVERGENCE: port `add_fork_hook`/`run_fork_hooks` into the posix
    /// module and register the whole trio — `before`→`acquire_lock`,
    /// `parent`→`release_lock`, `child`→`reinit_lock`.  Wiring the child hook
    /// alone would make the depth test pick the wrong branch.
    #[allow(dead_code)]
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
#[allow(dead_code)]
fn reinit_lock() {
    getimportlock().reinit_lock();
}

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
            "{function}() argument 1 must be str, not {}",
            bad_argument_type_name(name)
        )));
    }
    Ok(unsafe { pyre_object::w_str_get_value(name) }.to_owned())
}

/// `set_frozen_error` — both frozen diagnostics carry the module name as
/// `.name` with no `.path`, and render it the way `%R` does.
fn frozen_error(message: String, name: &str) -> crate::PyError {
    crate::PyError::import_error_name_path(
        message,
        pyre_object::w_str_new(name),
        pyre_object::w_none(),
    )
}

/// `%R` on the module name: quote selection and escaping identical to
/// `repr(str)`, which Rust's `{:?}` does not reproduce (it always
/// double-quotes).
fn frozen_name_repr(name: &str) -> String {
    let w_name = pyre_object::w_str_new(name);
    crate::display::format_wtf8_repr(unsafe { pyre_object::w_str_get_wtf8(w_name) })
}

fn missing_frozen_error(name: &str) -> crate::PyError {
    frozen_error(
        format!("No such frozen object named {}", frozen_name_repr(name)),
        name,
    )
}

/// `set_frozen_error(FROZEN_INVALID)` — the frozen data was supplied by the
/// caller but does not unmarshal.
fn invalid_frozen_error(name: &str) -> crate::PyError {
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
        "create_dynamic",
        // interp_imp.py:49 create_dynamic — no C-extension support, the
        // `has_so_extension() == False` branch. The spec's `name` and `origin`
        // are read and rejected for an embedded null before reporting the
        // unsupported load, matching `_imp_create_dynamic_impl`. Raising
        // ImportError (rather than being absent) matches the meta-path
        // `hasattr` probe while still letting `except ImportError` fall back to
        // a pure-Python module.
        crate::make_builtin_function_with_arity(
            "create_dynamic",
            |args| {
                let Some(&spec) = args.first() else {
                    return Err(crate::PyError::type_error(
                        "create_dynamic() missing required argument 'spec'",
                    ));
                };
                crate::baseobjspace::text0_w(crate::baseobjspace::getattr_str(spec, "name")?)?;
                crate::baseobjspace::text0_w(crate::baseobjspace::getattr_str(spec, "origin")?)?;
                Err(crate::PyError::new(
                    crate::PyErrorKind::ImportError,
                    "Not implemented".to_string(),
                ))
            },
            1,
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
                let newname = unsafe { pyre_object::w_str_get_value(args[1]) }.to_owned();
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
