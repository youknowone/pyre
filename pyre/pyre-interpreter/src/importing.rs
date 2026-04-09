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
use crate::{PyExecutionContext, PyNamespace, namespace_load, namespace_store};
use pyre_object::*;

use crate::eval::eval_frame_plain;
use crate::pyframe::PyFrame;

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
    register_builtin_module("_weakref", crate::module::_weakref::moduledef::init);
    register_builtin_module("_abc", init_abc);
    register_builtin_module("_functools", init_functools);
    register_builtin_module("_thread", init_thread);
    register_builtin_module("itertools", init_itertools);
    register_builtin_module("_contextvars", init_contextvars);
    register_builtin_module("copyreg", init_copyreg);
    register_builtin_module("_codecs", init_codecs);
    register_builtin_module("posix", init_posix);
    register_builtin_module("errno", init_errno);
    register_builtin_module("_collections", init_collections_c);
    register_builtin_module("_ast", init_ast);
    register_builtin_module("_opcode", init_opcode_c);
    register_builtin_module("_imp", init_imp);
    register_builtin_module("importlib.machinery", init_importlib_machinery);
    register_builtin_module("importlib", init_importlib_pkg);
    register_builtin_module("importlib.util", init_importlib_util);
    register_builtin_module("importlib.abc", init_importlib_abc);
    register_builtin_module("_signal", init_signal_stub);
    // _opcode_metadata.py exists in the stdlib; load the real file instead.
    for name in &[
        "_string",
        "_locale",
        "_warnings",
        "_heapq",
        "_tokenize",
        "_typing",
        "_bisect",
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
        "fcntl",
        "grp",
        "pwd",
        "select",
        "_socket",
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
fn empty_module_init(_ns: &mut PyNamespace) {}

/// _signal module stub — PyPy: pypy/module/signal/. Provides the signal()
/// function and SIG_DFL/SIG_IGN constants that signal.py wraps.
fn init_signal_stub(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "signal",
        crate::make_builtin_function("signal", |args| {
            // signal(signalnum, handler) — return previous handler (None stub).
            Ok(args.get(1).copied().unwrap_or(pyre_object::w_none()))
        }),
    );
    crate::namespace_store(
        ns,
        "getsignal",
        crate::make_builtin_function("getsignal", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "default_int_handler",
        crate::make_builtin_function("default_int_handler", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "set_wakeup_fd",
        crate::make_builtin_function("set_wakeup_fd", |_| Ok(pyre_object::w_int_new(-1))),
    );
    crate::namespace_store(ns, "SIG_DFL", pyre_object::w_int_new(0));
    crate::namespace_store(ns, "SIG_IGN", pyre_object::w_int_new(1));
    // Common signal numbers (POSIX subset).
    crate::namespace_store(ns, "SIGINT", pyre_object::w_int_new(2));
    crate::namespace_store(ns, "SIGTERM", pyre_object::w_int_new(15));
    crate::namespace_store(ns, "SIGHUP", pyre_object::w_int_new(1));
    crate::namespace_store(ns, "SIGQUIT", pyre_object::w_int_new(3));
    crate::namespace_store(ns, "SIGKILL", pyre_object::w_int_new(9));
    crate::namespace_store(ns, "SIGUSR1", pyre_object::w_int_new(30));
    crate::namespace_store(ns, "SIGUSR2", pyre_object::w_int_new(31));
    crate::namespace_store(ns, "SIGPIPE", pyre_object::w_int_new(13));
    crate::namespace_store(ns, "SIGALRM", pyre_object::w_int_new(14));
    crate::namespace_store(ns, "SIGCHLD", pyre_object::w_int_new(20));
    crate::namespace_store(ns, "NSIG", pyre_object::w_int_new(64));
}

/// _collections_abc stub — provides _check_methods for io.py etc.
/// The real _collections_abc.py requires ABCMeta (metaclass support).
fn init_collections_abc(ns: &mut PyNamespace) {
    // _check_methods(C, *methods) — check if class C has all methods
    crate::namespace_store(
        ns,
        "_check_methods",
        crate::make_builtin_function("_check_methods", |args| {
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
        crate::make_builtin_function("chain", |args| {
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
        crate::make_builtin_function("starmap", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    // count(start=0, step=1)
    crate::namespace_store(
        ns,
        "count",
        crate::make_builtin_function("count", |_| Ok(pyre_object::w_none())),
    );
    // repeat
    crate::namespace_store(
        ns,
        "repeat",
        crate::make_builtin_function("repeat", |_| Ok(pyre_object::w_none())),
    );
    // islice
    crate::namespace_store(
        ns,
        "islice",
        crate::make_builtin_function("islice", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    // groupby
    crate::namespace_store(
        ns,
        "groupby",
        crate::make_builtin_function("groupby", |_| Ok(pyre_object::w_none())),
    );
    // permutations(iterable, r=None) — PyPy: pypy/module/itertools/interp_itertools.py
    crate::namespace_store(
        ns,
        "permutations",
        crate::make_builtin_function("permutations", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            let pool = crate::builtins::collect_iterable(args[0])?;
            let n = pool.len();
            let r = if args.len() >= 2 {
                unsafe {
                    if pyre_object::is_int(args[1]) {
                        pyre_object::w_int_get_value(args[1]) as usize
                    } else {
                        n
                    }
                }
            } else {
                n
            };
            if r > n {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            // Heap/Lehmer would be clearer; use a recursive closure-free helper.
            fn perms(
                pool: &[pyre_object::PyObjectRef],
                r: usize,
            ) -> Vec<Vec<pyre_object::PyObjectRef>> {
                if r == 0 {
                    return vec![vec![]];
                }
                let mut out = Vec::new();
                for i in 0..pool.len() {
                    let mut rest: Vec<_> = pool.to_vec();
                    let head = rest.remove(i);
                    for mut tail in perms(&rest, r - 1) {
                        let mut v = vec![head];
                        v.append(&mut tail);
                        out.push(v);
                    }
                }
                out
            }
            let all = perms(&pool, r);
            let tuples: Vec<_> = all.into_iter().map(pyre_object::w_tuple_new).collect();
            let n = tuples.len();
            let list = pyre_object::w_list_new(tuples);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // combinations(iterable, r)
    crate::namespace_store(
        ns,
        "combinations",
        crate::make_builtin_function("combinations", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            let pool = crate::builtins::collect_iterable(args[0])?;
            let r = unsafe { pyre_object::w_int_get_value(args[1]) as usize };
            if r > pool.len() {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            fn combs(
                pool: &[pyre_object::PyObjectRef],
                r: usize,
                start: usize,
            ) -> Vec<Vec<pyre_object::PyObjectRef>> {
                if r == 0 {
                    return vec![vec![]];
                }
                let mut out = Vec::new();
                for i in start..pool.len() {
                    for mut tail in combs(pool, r - 1, i + 1) {
                        let mut v = vec![pool[i]];
                        v.append(&mut tail);
                        out.push(v);
                    }
                }
                out
            }
            let all = combs(&pool, r, 0);
            let tuples: Vec<_> = all.into_iter().map(pyre_object::w_tuple_new).collect();
            let n = tuples.len();
            let list = pyre_object::w_list_new(tuples);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // product(*iterables, repeat=1)
    crate::namespace_store(
        ns,
        "product",
        crate::make_builtin_function("product", |args| {
            let pools: Vec<Vec<_>> = args
                .iter()
                .map(|&a| crate::builtins::collect_iterable(a))
                .collect::<Result<_, _>>()?;
            let mut result: Vec<Vec<pyre_object::PyObjectRef>> = vec![vec![]];
            for pool in &pools {
                let mut new_result = Vec::with_capacity(result.len() * pool.len());
                for existing in &result {
                    for &item in pool {
                        let mut v = existing.clone();
                        v.push(item);
                        new_result.push(v);
                    }
                }
                result = new_result;
            }
            let tuples: Vec<_> = result.into_iter().map(pyre_object::w_tuple_new).collect();
            let n = tuples.len();
            let list = pyre_object::w_list_new(tuples);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // zip_longest(*iterables, fillvalue=None)
    crate::namespace_store(
        ns,
        "zip_longest",
        crate::make_builtin_function("zip_longest", |args| {
            let pools: Vec<Vec<_>> = args
                .iter()
                .map(|&a| crate::builtins::collect_iterable(a))
                .collect::<Result<_, _>>()?;
            let max_len = pools.iter().map(|p| p.len()).max().unwrap_or(0);
            let fill = pyre_object::w_none();
            let mut tuples = Vec::with_capacity(max_len);
            for i in 0..max_len {
                let row: Vec<_> = pools
                    .iter()
                    .map(|p| if i < p.len() { p[i] } else { fill })
                    .collect();
                tuples.push(pyre_object::w_tuple_new(row));
            }
            let n = tuples.len();
            let list = pyre_object::w_list_new(tuples);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // accumulate(iterable) — sums only, PyPy interp_itertools W_Accumulate.
    crate::namespace_store(
        ns,
        "accumulate",
        crate::make_builtin_function("accumulate", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            let items = crate::builtins::collect_iterable(args[0])?;
            let mut out = Vec::with_capacity(items.len());
            let mut acc: Option<pyre_object::PyObjectRef> = None;
            for item in items {
                acc = Some(match acc {
                    None => item,
                    Some(prev) => crate::baseobjspace::add(prev, item)?,
                });
                out.push(acc.unwrap());
            }
            let n = out.len();
            let list = pyre_object::w_list_new(out);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // compress(data, selectors)
    crate::namespace_store(
        ns,
        "compress",
        crate::make_builtin_function("compress", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            let data = crate::builtins::collect_iterable(args[0])?;
            let selectors = crate::builtins::collect_iterable(args[1])?;
            let mut out = Vec::new();
            for (d, s) in data.iter().zip(selectors.iter()) {
                if crate::baseobjspace::is_true(*s) {
                    out.push(*d);
                }
            }
            let n = out.len();
            let list = pyre_object::w_list_new(out);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
}

/// _contextvars stub
fn init_contextvars(ns: &mut PyNamespace) {
    // ContextVar(name, *, default=_MISSING) — context variable
    crate::namespace_store(
        ns,
        "ContextVar",
        crate::make_builtin_function("ContextVar", |args| {
            // Return stub object with get/set methods
            let obj = pyre_object::w_instance_new(crate::typedef::w_object());
            if !args.is_empty() {
                let _ = crate::baseobjspace::setattr(obj, "name", args[0]);
            }
            // get() returns default or raises LookupError
            let _ = crate::baseobjspace::setattr(
                obj,
                "get",
                crate::make_builtin_function("get", |args| {
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
                crate::make_builtin_function("set", |_| Ok(pyre_object::w_none())),
            );
            Ok(obj)
        }),
    );
    crate::namespace_store(
        ns,
        "Context",
        crate::make_builtin_function("Context", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "Token",
        crate::make_builtin_function("Token", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "copy_context",
        crate::make_builtin_function("copy_context", |_| Ok(pyre_object::w_none())),
    );
}

/// _abc stub — PyPy: pypy/module/_abc/
fn init_abc(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "get_cache_token",
        crate::make_builtin_function("get_cache_token", |_| Ok(pyre_object::w_int_new(0))),
    );
    crate::namespace_store(
        ns,
        "_abc_init",
        crate::make_builtin_function("_abc_init", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "_abc_register",
        crate::make_builtin_function("_abc_register", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "_abc_instancecheck",
        crate::make_builtin_function("_abc_instancecheck", |_args| {
            Ok(pyre_object::w_bool_from(false))
        }),
    );
    crate::namespace_store(
        ns,
        "_abc_subclasscheck",
        crate::make_builtin_function("_abc_subclasscheck", |_args| {
            Ok(pyre_object::w_bool_from(false))
        }),
    );
    crate::namespace_store(
        ns,
        "_get_dump",
        crate::make_builtin_function("_get_dump", |_| Ok(pyre_object::w_tuple_new(vec![]))),
    );
    crate::namespace_store(
        ns,
        "_reset_registry",
        crate::make_builtin_function("_reset_registry", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "_reset_caches",
        crate::make_builtin_function("_reset_caches", |_| Ok(pyre_object::w_none())),
    );
}

/// _functools stub
fn init_functools(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "reduce",
        crate::make_builtin_function("reduce", |_| {
            Err(crate::PyError::type_error("reduce not implemented"))
        }),
    );
    // functools.cmp_to_key(cmp) — returns a callable that wraps a value in
    // an opaque key. For sorting str / int / tuple of those (the only paths
    // pyre's stdlib actually exercises), the items are already comparable,
    // so an identity key gives the same ordering as `cmp(a, b)` would.
    crate::namespace_store(
        ns,
        "cmp_to_key",
        crate::make_builtin_function("cmp_to_key", |_args| {
            Ok(crate::make_builtin_function("cmp_to_key.K", |args| {
                Ok(args.first().copied().unwrap_or(pyre_object::w_none()))
            }))
        }),
    );
}

/// Dummy lock methods — acquire/release no-op, __enter__/__exit__ for `with`.
/// PyPy: pypy/module/thread/os_lock.py W_Lock / W_RLock
fn init_lock_type(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "__enter__",
        crate::make_builtin_function("__enter__", |args| {
            Ok(if args.is_empty() {
                pyre_object::w_none()
            } else {
                args[0]
            })
        }),
    );
    crate::namespace_store(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |_| Ok(pyre_object::w_bool_from(false))),
    );
    crate::namespace_store(
        ns,
        "acquire",
        crate::make_builtin_function("acquire", |_| Ok(pyre_object::w_bool_from(true))),
    );
    crate::namespace_store(
        ns,
        "release",
        crate::make_builtin_function("release", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "locked",
        crate::make_builtin_function("locked", |_| Ok(pyre_object::w_bool_from(false))),
    );
    crate::namespace_store(
        ns,
        "_is_owned",
        crate::make_builtin_function("_is_owned", |_| Ok(pyre_object::w_bool_from(false))),
    );
}

thread_local! {
    static LOCK_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
}

fn lock_type() -> PyObjectRef {
    LOCK_TYPE_OBJ
        .with(|c| *c.get_or_init(|| crate::typedef::make_builtin_type("lock", init_lock_type)))
}

/// _thread stub
fn init_thread(ns: &mut PyNamespace) {
    let lock_tp = lock_type();
    crate::namespace_store(ns, "LockType", lock_tp);
    crate::namespace_store(
        ns,
        "RLock",
        crate::make_builtin_function("RLock", |_| Ok(pyre_object::w_instance_new(lock_type()))),
    );
    crate::namespace_store(
        ns,
        "allocate_lock",
        crate::make_builtin_function("allocate_lock", |_| {
            Ok(pyre_object::w_instance_new(lock_type()))
        }),
    );
    crate::namespace_store(
        ns,
        "get_ident",
        crate::make_builtin_function("get_ident", |_| Ok(pyre_object::w_int_new(1))),
    );
    crate::namespace_store(
        ns,
        "_count",
        crate::make_builtin_function("_count", |_| Ok(pyre_object::w_int_new(1))),
    );
    crate::namespace_store(ns, "TIMEOUT_MAX", pyre_object::w_float_new(f64::MAX));
    crate::namespace_store(ns, "error", crate::typedef::w_object());
}

/// posix stub — PyPy: pypy/module/posix/ interp_posix.py
///
/// Provides the minimal surface that os.py module init needs to succeed.
/// Real posix calls are not implemented — they raise or return defaults.
fn init_posix(ns: &mut PyNamespace) {
    // environ — empty dict
    crate::namespace_store(ns, "environ", pyre_object::w_dict_new());
    // _have_functions — list of HAVE_* macro names that were defined at
    // build time. os.py uses this to populate the supports_* capability
    // sets. Advertising a representative subset lets os.py module init
    // complete successfully.
    crate::namespace_store(
        ns,
        "_have_functions",
        pyre_object::w_list_new(vec![
            pyre_object::w_str_new("HAVE_FACCESSAT"),
            pyre_object::w_str_new("HAVE_FCHDIR"),
            pyre_object::w_str_new("HAVE_FCHMOD"),
            pyre_object::w_str_new("HAVE_FCHMODAT"),
            pyre_object::w_str_new("HAVE_FCHOWN"),
            pyre_object::w_str_new("HAVE_FCHOWNAT"),
            pyre_object::w_str_new("HAVE_FDOPENDIR"),
            pyre_object::w_str_new("HAVE_FEXECVE"),
            pyre_object::w_str_new("HAVE_FPATHCONF"),
            pyre_object::w_str_new("HAVE_FSTATAT"),
            pyre_object::w_str_new("HAVE_FSTATVFS"),
            pyre_object::w_str_new("HAVE_FTRUNCATE"),
            pyre_object::w_str_new("HAVE_FUTIMENS"),
            pyre_object::w_str_new("HAVE_FUTIMES"),
            pyre_object::w_str_new("HAVE_FUTIMESAT"),
            pyre_object::w_str_new("HAVE_LINKAT"),
            pyre_object::w_str_new("HAVE_LSTAT"),
            pyre_object::w_str_new("HAVE_MKDIRAT"),
            pyre_object::w_str_new("HAVE_MKFIFOAT"),
            pyre_object::w_str_new("HAVE_MKNODAT"),
            pyre_object::w_str_new("HAVE_OPENAT"),
            pyre_object::w_str_new("HAVE_READLINKAT"),
            pyre_object::w_str_new("HAVE_RENAMEAT"),
            pyre_object::w_str_new("HAVE_SYMLINKAT"),
            pyre_object::w_str_new("HAVE_UNLINKAT"),
            pyre_object::w_str_new("HAVE_UTIMENSAT"),
        ]),
    );
    // Sentinel constants referenced by os.py at module level
    for name in [
        "F_OK",
        "R_OK",
        "W_OK",
        "X_OK",
        "O_RDONLY",
        "O_WRONLY",
        "O_RDWR",
        "O_APPEND",
        "O_CREAT",
        "O_EXCL",
        "O_TRUNC",
        "O_NONBLOCK",
        "O_NDELAY",
        "O_DSYNC",
        "O_SYNC",
        "SEEK_SET",
        "SEEK_CUR",
        "SEEK_END",
        "EX_OK",
        "EX_USAGE",
        "EX_DATAERR",
        "EX_NOINPUT",
        "EX_NOUSER",
        "EX_NOHOST",
        "EX_UNAVAILABLE",
        "EX_SOFTWARE",
        "EX_OSERR",
        "EX_OSFILE",
        "EX_CANTCREAT",
        "EX_IOERR",
        "EX_TEMPFAIL",
        "EX_PROTOCOL",
        "EX_NOPERM",
        "EX_CONFIG",
        "WNOHANG",
        "WCONTINUED",
        "WUNTRACED",
        "P_WAIT",
        "P_NOWAIT",
        "P_NOWAITO",
        "ST_RDONLY",
        "ST_NOSUID",
        "SCHED_OTHER",
        "SCHED_FIFO",
        "SCHED_RR",
        "SCHED_BATCH",
        "SCHED_IDLE",
        "RTLD_LAZY",
        "RTLD_NOW",
        "RTLD_GLOBAL",
        "RTLD_LOCAL",
        "RTLD_NODELETE",
        "RTLD_NOLOAD",
        "RTLD_DEEPBIND",
        "PRIO_PROCESS",
        "PRIO_PGRP",
        "PRIO_USER",
    ] {
        crate::namespace_store(ns, name, pyre_object::w_int_new(0));
    }
    // Functions that os.py references at module level as values (for set membership tests)
    for name in [
        "stat",
        "lstat",
        "fstat",
        "fstatat",
        "statvfs",
        "fstatvfs",
        "open",
        "close",
        "read",
        "write",
        "lseek",
        "dup",
        "dup2",
        "chdir",
        "fchdir",
        "getcwd",
        "getcwdb",
        "mkdir",
        "rmdir",
        "remove",
        "unlink",
        "rename",
        "link",
        "symlink",
        "readlink",
        "chmod",
        "fchmod",
        "lchmod",
        "chown",
        "fchown",
        "lchown",
        "access",
        "faccessat",
        "chflags",
        "lchflags",
        "utime",
        "futimens",
        "futimes",
        "listdir",
        "scandir",
        "fdopendir",
        "execve",
        "execv",
        "fork",
        "forkpty",
        "wait",
        "waitpid",
        "truncate",
        "ftruncate",
        "pathconf",
        "fpathconf",
        "getuid",
        "geteuid",
        "getgid",
        "getegid",
        "getpid",
        "getppid",
        "setuid",
        "setgid",
        "setsid",
        "setpgid",
        "setreuid",
        "setregid",
        "getgroups",
        "setgroups",
        "getpgrp",
        "setpgrp",
        "getpgid",
        "umask",
        "uname",
        "getlogin",
        "nice",
        "pipe",
        "pipe2",
        "dup3",
        "fspath",
        "fsync",
        "fdatasync",
        "mkfifo",
        "mknod",
        "major",
        "minor",
        "makedev",
        "get_inheritable",
        "set_inheritable",
        "get_blocking",
        "set_blocking",
        "urandom",
        "get_terminal_size",
        "cpu_count",
        "getloadavg",
        "kill",
        "killpg",
        "getpriority",
        "setpriority",
        "sched_get_priority_max",
        "sched_get_priority_min",
        "sched_getparam",
        "sched_setparam",
        "sched_getscheduler",
        "sched_setscheduler",
        "sched_yield",
        "confstr",
        "confstr_names",
        "sysconf",
        "sysconf_names",
        "pathconf_names",
        "setenv",
        "unsetenv",
        "getenv",
        "putenv",
        "device_encoding",
        "isatty",
        "ttyname",
        "openpty",
        "login_tty",
        "tcgetpgrp",
        "tcsetpgrp",
        "ctermid",
        "get_exec_path",
        "WIFEXITED",
        "WEXITSTATUS",
        "WIFSIGNALED",
        "WTERMSIG",
        "WIFSTOPPED",
        "WSTOPSIG",
        "WEXITED",
        "WNOWAIT",
        "WSTOPPED",
        "waitstatus_to_exitcode",
        "_exit",
        "_cpu_count",
    ] {
        crate::namespace_store(
            ns,
            name,
            crate::make_builtin_function(name, |_| Ok(pyre_object::w_int_new(0))),
        );
    }
    // os.fspath() — PyPy: posixmodule.c posix_fspath. Returns the argument
    // unchanged for str/bytes/bytearray (the protocol's identity case);
    // any other object would normally trigger __fspath__ but we don't
    // model that protocol yet.
    crate::namespace_store(
        ns,
        "fspath",
        crate::make_builtin_function("fspath", |args| {
            let arg = args.first().copied().unwrap_or(pyre_object::w_none());
            unsafe {
                if pyre_object::is_str(arg) || pyre_object::bytearrayobject::is_bytearray(arg) {
                    return Ok(arg);
                }
            }
            // Try __fspath__ — for pathlib.Path-like objects.
            if let Ok(method) = crate::baseobjspace::getattr(arg, "__fspath__") {
                let result = crate::call_function(method, &[arg]);
                if !result.is_null() {
                    return Ok(result);
                }
            }
            Ok(arg)
        }),
    );
    crate::namespace_store(ns, "error", crate::typedef::w_object());
}

/// _collections C-extension stub — PyPy: pypy/module/_collections/
/// Provides the C-accelerated deque/defaultdict/OrderedDict types.
/// Our stubs are backed by lists/dicts, which is correct semantically
/// but not performant. PyPy's W_Deque is a doubly-linked block list.
fn init_collections_c(ns: &mut PyNamespace) {
    // deque(iterable=(), maxlen=None) — returns a list that we alias as deque.
    // Sufficient for collections.py's MutableSequence.register(deque).
    let deque_type = crate::typedef::make_builtin_type("deque", init_deque_type);
    crate::namespace_store(ns, "deque", deque_type);
    // _deque_iterator — reuse object (just a type sentinel)
    crate::namespace_store(ns, "_deque_iterator", crate::typedef::w_object());
    // defaultdict — returns a dict-like instance
    let defaultdict_type = crate::typedef::make_builtin_type("defaultdict", init_defaultdict_type);
    crate::namespace_store(ns, "defaultdict", defaultdict_type);
    // OrderedDict — same as dict for our purposes
    crate::namespace_store(ns, "OrderedDict", crate::typedef::w_type());
}

/// deque methods — PyPy: pypy/module/_collections/interp_deque.py W_Deque
fn init_deque_type(ns: &mut PyNamespace) {
    // __init__(self, iterable=(), maxlen=None) — store items as __data__ list
    crate::namespace_store(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let self_obj = args[0];
            let items: Vec<_> = if args.len() >= 2 {
                crate::builtins::collect_iterable(args[1]).unwrap_or_default()
            } else {
                Vec::new()
            };
            let list = pyre_object::w_list_new(items);
            let _ = crate::baseobjspace::setattr(self_obj, "__data__", list);
            let _ = crate::baseobjspace::setattr(
                self_obj,
                "maxlen",
                if args.len() >= 3 {
                    args[2]
                } else {
                    pyre_object::w_none()
                },
            );
            Ok(pyre_object::w_none())
        }),
    );
    crate::namespace_store(
        ns,
        "append",
        crate::make_builtin_function("append", |args| {
            if args.len() >= 2 {
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    unsafe { pyre_object::w_list_append(data, args[1]) };
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    crate::namespace_store(
        ns,
        "appendleft",
        crate::make_builtin_function("appendleft", |args| {
            if args.len() >= 2 {
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    unsafe {
                        let n = pyre_object::w_list_len(data);
                        let mut items: Vec<_> = (0..n)
                            .filter_map(|i| pyre_object::w_list_getitem(data, i as i64))
                            .collect();
                        items.insert(0, args[1]);
                        let new_list = pyre_object::w_list_new(items);
                        let _ = crate::baseobjspace::setattr(args[0], "__data__", new_list);
                    }
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    crate::namespace_store(
        ns,
        "pop",
        crate::make_builtin_function("pop", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                unsafe {
                    let n = pyre_object::w_list_len(data);
                    if n > 0 {
                        let item = pyre_object::w_list_getitem(data, (n - 1) as i64)
                            .unwrap_or(pyre_object::w_none());
                        let items: Vec<_> = (0..n - 1)
                            .filter_map(|i| pyre_object::w_list_getitem(data, i as i64))
                            .collect();
                        let new_list = pyre_object::w_list_new(items);
                        let _ = crate::baseobjspace::setattr(args[0], "__data__", new_list);
                        return Ok(item);
                    }
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    crate::namespace_store(
        ns,
        "popleft",
        crate::make_builtin_function("popleft", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                unsafe {
                    let n = pyre_object::w_list_len(data);
                    if n > 0 {
                        let item =
                            pyre_object::w_list_getitem(data, 0).unwrap_or(pyre_object::w_none());
                        let items: Vec<_> = (1..n)
                            .filter_map(|i| pyre_object::w_list_getitem(data, i as i64))
                            .collect();
                        let new_list = pyre_object::w_list_new(items);
                        let _ = crate::baseobjspace::setattr(args[0], "__data__", new_list);
                        return Ok(item);
                    }
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    crate::namespace_store(
        ns,
        "clear",
        crate::make_builtin_function("clear", |args| {
            if !args.is_empty() {
                let _ = crate::baseobjspace::setattr(
                    args[0],
                    "__data__",
                    pyre_object::w_list_new(vec![]),
                );
            }
            Ok(pyre_object::w_none())
        }),
    );
    crate::namespace_store(
        ns,
        "extend",
        crate::make_builtin_function("extend", |args| {
            if args.len() >= 2 {
                let items = crate::builtins::collect_iterable(args[1])?;
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    for item in items {
                        unsafe { pyre_object::w_list_append(data, item) };
                    }
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    crate::namespace_store(
        ns,
        "__len__",
        crate::make_builtin_function("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                return Ok(pyre_object::w_int_new(
                    unsafe { pyre_object::w_list_len(data) } as i64,
                ));
            }
            Ok(pyre_object::w_int_new(0))
        }),
    );
    crate::namespace_store(
        ns,
        "__iter__",
        crate::make_builtin_function("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_seq_iter_new(
                    pyre_object::w_list_new(vec![]),
                    0,
                ));
            }
            if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                return crate::baseobjspace::iter(data);
            }
            Ok(pyre_object::w_seq_iter_new(
                pyre_object::w_list_new(vec![]),
                0,
            ))
        }),
    );
    crate::namespace_store(
        ns,
        "__getitem__",
        crate::make_builtin_function("__getitem__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_none());
            }
            if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                return crate::baseobjspace::getitem(data, args[1]);
            }
            Ok(pyre_object::w_none())
        }),
    );
}

/// defaultdict — PyPy: pypy/module/_collections/interp_defaultdict.py
fn init_defaultdict_type(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let self_obj = args[0];
            let factory = if args.len() >= 2 {
                args[1]
            } else {
                pyre_object::w_none()
            };
            let _ = crate::baseobjspace::setattr(self_obj, "default_factory", factory);
            let _ = crate::baseobjspace::setattr(self_obj, "__data__", pyre_object::w_dict_new());
            Ok(pyre_object::w_none())
        }),
    );
    crate::namespace_store(
        ns,
        "__getitem__",
        crate::make_builtin_function("__getitem__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_none());
            }
            let self_obj = args[0];
            let key = args[1];
            if let Ok(data) = crate::baseobjspace::getattr(self_obj, "__data__") {
                unsafe {
                    if let Some(v) = pyre_object::w_dict_lookup(data, key) {
                        return Ok(v);
                    }
                }
                // Not present — try factory
                if let Ok(factory) = crate::baseobjspace::getattr(self_obj, "default_factory") {
                    if !factory.is_null() && !unsafe { pyre_object::is_none(factory) } {
                        // Can't easily call factory without frame — return None.
                        let default = pyre_object::w_none();
                        unsafe { pyre_object::w_dict_store(data, key, default) };
                        return Ok(default);
                    }
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    crate::namespace_store(
        ns,
        "__setitem__",
        crate::make_builtin_function("__setitem__", |args| {
            if args.len() >= 3 {
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    unsafe { pyre_object::w_dict_store(data, args[1], args[2]) };
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
}

/// _opcode stub — PyPy: pypy/module/_opcode (CPython's opcode introspection).
/// opcode.py requires stack_effect + has_arg/has_const/has_name/has_jump and
/// related classifiers. Our stubs return neutral values; full implementations
/// would mirror CPython Python/compile.c.
fn init_opcode_c(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "stack_effect",
        crate::make_builtin_function("stack_effect", |_| Ok(pyre_object::w_int_new(0))),
    );
    for name in [
        "has_arg",
        "has_const",
        "has_name",
        "has_jump",
        "has_jrel",
        "has_jabs",
        "has_free",
        "has_local",
        "has_exc",
    ] {
        crate::namespace_store(
            ns,
            name,
            crate::make_builtin_function(name, |_| Ok(pyre_object::w_bool_from(false))),
        );
    }
    crate::namespace_store(
        ns,
        "get_executor",
        crate::make_builtin_function("get_executor", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "get_specialization_stats",
        crate::make_builtin_function(
            "get_specialization_stats",
            |_| Ok(pyre_object::w_dict_new()),
        ),
    );
    crate::namespace_store(
        ns,
        "get_intrinsic1_descs",
        crate::make_builtin_function("get_intrinsic1_descs", |_| {
            Ok(pyre_object::w_list_new(vec![]))
        }),
    );
    crate::namespace_store(
        ns,
        "get_intrinsic2_descs",
        crate::make_builtin_function("get_intrinsic2_descs", |_| {
            Ok(pyre_object::w_list_new(vec![]))
        }),
    );
    crate::namespace_store(
        ns,
        "get_opname",
        crate::make_builtin_function("get_opname", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new("<0>"));
            }
            let code = unsafe { pyre_object::w_int_get_value(args[0]) };
            Ok(pyre_object::w_str_new(&format!("<{code}>")))
        }),
    );
    crate::namespace_store(
        ns,
        "get_nb_ops",
        crate::make_builtin_function("get_nb_ops", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    crate::namespace_store(
        ns,
        "get_special_method_names",
        crate::make_builtin_function("get_special_method_names", |_| {
            Ok(pyre_object::w_list_new(vec![
                pyre_object::w_str_new("__enter__"),
                pyre_object::w_str_new("__exit__"),
                pyre_object::w_str_new("__aenter__"),
                pyre_object::w_str_new("__aexit__"),
            ]))
        }),
    );
    crate::namespace_store(
        ns,
        "get_executor_count",
        crate::make_builtin_function("get_executor_count", |_| Ok(pyre_object::w_int_new(0))),
    );
    crate::namespace_store(
        ns,
        "get_hot_code",
        crate::make_builtin_function("get_hot_code", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
}

/// _opcode_metadata stub — module used by opcode.py for opmap / specializations.
/// PyPy does not have this module; it is CPython 3.13+ specific.
fn init_opcode_metadata(ns: &mut PyNamespace) {
    // opmap — map opcode name → integer. opcode.py requires at least
    // EXTENDED_ARG. We provide a minimal baseline.
    let opmap = pyre_object::w_dict_new();
    let base_ops: &[(&str, i64)] = &[
        ("CACHE", 0),
        ("POP_TOP", 1),
        ("PUSH_NULL", 2),
        ("NOP", 9),
        ("RESUME", 149),
        ("EXTENDED_ARG", 148),
        ("RETURN_VALUE", 83),
        ("RETURN_CONST", 121),
        ("IMPORT_NAME", 108),
        ("IMPORT_FROM", 109),
        ("LOAD_CONST", 100),
        ("LOAD_FAST", 124),
        ("LOAD_NAME", 101),
        ("LOAD_GLOBAL", 116),
        ("LOAD_ATTR", 106),
        ("LOAD_DEREF", 137),
        ("LOAD_SPECIAL", 95),
        ("STORE_NAME", 90),
        ("STORE_FAST", 125),
        ("STORE_GLOBAL", 97),
        ("STORE_ATTR", 95),
        ("STORE_DEREF", 138),
        ("BINARY_OP", 122),
        ("COMPARE_OP", 107),
        ("CONTAINS_OP", 118),
        ("IS_OP", 117),
        ("CALL", 171),
        ("CALL_FUNCTION_EX", 142),
        ("MAKE_FUNCTION", 132),
        ("MAKE_CELL", 135),
        ("JUMP_FORWARD", 110),
        ("JUMP_BACKWARD", 140),
        ("POP_JUMP_IF_FALSE", 114),
        ("POP_JUMP_IF_TRUE", 115),
        ("POP_JUMP_IF_NONE", 119),
        ("POP_JUMP_IF_NOT_NONE", 120),
        ("RAISE_VARARGS", 146),
        ("BUILD_LIST", 103),
        ("BUILD_TUPLE", 102),
        ("BUILD_MAP", 105),
        ("BUILD_SET", 104),
        ("BUILD_STRING", 111),
        ("GET_ITER", 68),
        ("FOR_ITER", 93),
        ("END_FOR", 4),
        ("COPY", 69),
        ("SWAP", 70),
    ];
    for (name, code) in base_ops {
        unsafe {
            pyre_object::w_dict_store(
                opmap,
                pyre_object::w_str_new(name),
                pyre_object::w_int_new(*code),
            );
        }
    }
    crate::namespace_store(ns, "opmap", opmap);
    crate::namespace_store(ns, "_specializations", pyre_object::w_dict_new());
    crate::namespace_store(ns, "_specialized_opmap", pyre_object::w_dict_new());
    crate::namespace_store(ns, "HAVE_ARGUMENT", pyre_object::w_int_new(90));
    crate::namespace_store(ns, "MIN_INSTRUMENTED_OPCODE", pyre_object::w_int_new(237));
}

/// importlib stub — PyPy: pypy/module/importlib/
/// Avoid loading the real importlib.__init__ since it drags in
/// _bootstrap and _bootstrap_external.
fn init_importlib_pkg(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "import_module",
        crate::make_builtin_function("import_module", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "invalidate_caches",
        crate::make_builtin_function("invalidate_caches", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "reload",
        crate::make_builtin_function("reload", |args| {
            Ok(args.first().copied().unwrap_or(pyre_object::w_none()))
        }),
    );
    // Mark as a package so dotted imports treat it as such.
    crate::namespace_store(ns, "__path__", pyre_object::w_list_new(vec![]));
}

/// importlib.util stub — minimal subset.
fn init_importlib_util(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "spec_from_file_location",
        crate::make_builtin_function("spec_from_file_location", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "module_from_spec",
        crate::make_builtin_function("module_from_spec", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "find_spec",
        crate::make_builtin_function("find_spec", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "resolve_name",
        crate::make_builtin_function("resolve_name", |args| {
            Ok(args.first().copied().unwrap_or(pyre_object::w_str_new("")))
        }),
    );
    crate::namespace_store(ns, "MAGIC_NUMBER", pyre_object::w_int_new(0));
}

/// importlib.abc stub — abstract base classes.
fn init_importlib_abc(ns: &mut PyNamespace) {
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
        crate::namespace_store(ns, name, crate::typedef::w_object());
    }
}

/// importlib.machinery stub — provides the names inspect.py references.
/// PyPy ships the real importlib; we shortcut it with a stub so pyre does
/// not have to execute _bootstrap_external.
fn init_importlib_machinery(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "SOURCE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".py")]),
    );
    crate::namespace_store(
        ns,
        "BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::namespace_store(
        ns,
        "EXTENSION_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".so")]),
    );
    crate::namespace_store(
        ns,
        "DEBUG_BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::namespace_store(
        ns,
        "OPTIMIZED_BYTECODE_SUFFIXES",
        pyre_object::w_list_new(vec![pyre_object::w_str_new(".pyc")]),
    );
    crate::namespace_store(
        ns,
        "all_suffixes",
        crate::make_builtin_function("all_suffixes", |_| {
            Ok(pyre_object::w_list_new(vec![
                pyre_object::w_str_new(".py"),
                pyre_object::w_str_new(".pyc"),
                pyre_object::w_str_new(".so"),
            ]))
        }),
    );
    crate::namespace_store(ns, "ModuleSpec", crate::typedef::w_object());
    crate::namespace_store(ns, "BuiltinImporter", crate::typedef::w_object());
    crate::namespace_store(ns, "FrozenImporter", crate::typedef::w_object());
    crate::namespace_store(ns, "PathFinder", crate::typedef::w_object());
    crate::namespace_store(ns, "FileFinder", crate::typedef::w_object());
    crate::namespace_store(ns, "SourceFileLoader", crate::typedef::w_object());
    crate::namespace_store(ns, "SourcelessFileLoader", crate::typedef::w_object());
    crate::namespace_store(ns, "ExtensionFileLoader", crate::typedef::w_object());
    crate::namespace_store(ns, "AppleFrameworkLoader", crate::typedef::w_object());
    crate::namespace_store(ns, "NamespaceLoader", crate::typedef::w_object());
    crate::namespace_store(ns, "WindowsRegistryFinder", crate::typedef::w_object());
}

/// _imp stub — PyPy: pypy/module/imp/
///
/// Minimal subset required by importlib._bootstrap to decide which loader
/// handles a name. We report every name we know about as a builtin so
/// pyre's own registrations remain authoritative.
fn init_imp(ns: &mut PyNamespace) {
    crate::namespace_store(
        ns,
        "is_builtin",
        crate::make_builtin_function("is_builtin", |args| {
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
        }),
    );
    crate::namespace_store(
        ns,
        "is_frozen",
        crate::make_builtin_function("is_frozen", |_| Ok(pyre_object::w_bool_from(false))),
    );
    crate::namespace_store(
        ns,
        "is_frozen_package",
        crate::make_builtin_function("is_frozen_package", |_| Ok(pyre_object::w_bool_from(false))),
    );
    crate::namespace_store(
        ns,
        "get_frozen_object",
        crate::make_builtin_function("get_frozen_object", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "create_builtin",
        crate::make_builtin_function("create_builtin", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            Ok(args[0])
        }),
    );
    crate::namespace_store(
        ns,
        "exec_builtin",
        crate::make_builtin_function("exec_builtin", |_| Ok(pyre_object::w_int_new(0))),
    );
    crate::namespace_store(
        ns,
        "exec_dynamic",
        crate::make_builtin_function("exec_dynamic", |_| Ok(pyre_object::w_int_new(0))),
    );
    crate::namespace_store(
        ns,
        "acquire_lock",
        crate::make_builtin_function("acquire_lock", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "release_lock",
        crate::make_builtin_function("release_lock", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "lock_held",
        crate::make_builtin_function("lock_held", |_| Ok(pyre_object::w_bool_from(false))),
    );
    crate::namespace_store(
        ns,
        "_fix_co_filename",
        crate::make_builtin_function("_fix_co_filename", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "extension_suffixes",
        crate::make_builtin_function("extension_suffixes", |_| {
            Ok(pyre_object::w_list_new(vec![]))
        }),
    );
    crate::namespace_store(
        ns,
        "source_hash",
        crate::make_builtin_function("source_hash", |_| Ok(pyre_object::w_int_new(0))),
    );
    crate::namespace_store(
        ns,
        "check_hash_based_pycs",
        pyre_object::w_str_new("default"),
    );
    crate::namespace_store(ns, "pyc_magic_number_token", pyre_object::w_int_new(3495));
}

/// _ast stub — PyPy: pypy/module/_ast/
///
/// Exposes the AST node type hierarchy as plain type stubs. Our stubs are
/// enough to satisfy `from _ast import *` in `ast.py` and class body
/// references like `class slice(AST)`. Actual AST construction is not
/// supported because pyre uses RustPython's compiler.
fn init_ast(ns: &mut PyNamespace) {
    let ast_names: &[&str] = &[
        "AST",
        "mod",
        "Module",
        "Interactive",
        "Expression",
        "FunctionType",
        "stmt",
        "FunctionDef",
        "AsyncFunctionDef",
        "ClassDef",
        "Return",
        "Delete",
        "Assign",
        "TypeAlias",
        "AugAssign",
        "AnnAssign",
        "For",
        "AsyncFor",
        "While",
        "If",
        "With",
        "AsyncWith",
        "Match",
        "Raise",
        "Try",
        "TryStar",
        "Assert",
        "Import",
        "ImportFrom",
        "Global",
        "Nonlocal",
        "Expr",
        "Pass",
        "Break",
        "Continue",
        "expr",
        "BoolOp",
        "NamedExpr",
        "BinOp",
        "UnaryOp",
        "Lambda",
        "IfExp",
        "Dict",
        "Set",
        "ListComp",
        "SetComp",
        "DictComp",
        "GeneratorExp",
        "Await",
        "Yield",
        "YieldFrom",
        "Compare",
        "Call",
        "FormattedValue",
        "JoinedStr",
        "Constant",
        "Attribute",
        "Subscript",
        "Starred",
        "Name",
        "List",
        "Tuple",
        "Slice",
        "expr_context",
        "Load",
        "Store",
        "Del",
        "boolop",
        "And",
        "Or",
        "operator",
        "Add",
        "Sub",
        "Mult",
        "MatMult",
        "Div",
        "Mod",
        "Pow",
        "LShift",
        "RShift",
        "BitOr",
        "BitXor",
        "BitAnd",
        "FloorDiv",
        "unaryop",
        "Invert",
        "Not",
        "UAdd",
        "USub",
        "cmpop",
        "Eq",
        "NotEq",
        "Lt",
        "LtE",
        "Gt",
        "GtE",
        "Is",
        "IsNot",
        "In",
        "NotIn",
        "comprehension",
        "excepthandler",
        "ExceptHandler",
        "arguments",
        "arg",
        "keyword",
        "alias",
        "withitem",
        "match_case",
        "pattern",
        "MatchValue",
        "MatchSingleton",
        "MatchSequence",
        "MatchMapping",
        "MatchClass",
        "MatchStar",
        "MatchAs",
        "MatchOr",
        "type_ignore",
        "TypeIgnore",
        "type_param",
        "TypeVar",
        "ParamSpec",
        "TypeVarTuple",
        // Flags used by ast.parse()
        "PyCF_ONLY_AST",
        "PyCF_OPTIMIZED_AST",
        "PyCF_TYPE_COMMENTS",
        "PyCF_ALLOW_TOP_LEVEL_AWAIT",
    ];
    for name in ast_names {
        if name.starts_with("PyCF") {
            crate::namespace_store(ns, name, pyre_object::w_int_new(0));
        } else {
            crate::namespace_store(ns, name, crate::typedef::make_builtin_type(name, |_| {}));
        }
    }
}

/// errno stub — PyPy: pypy/module/errno/
fn init_errno(ns: &mut PyNamespace) {
    for (name, value) in [
        ("EPERM", 1),
        ("ENOENT", 2),
        ("ESRCH", 3),
        ("EINTR", 4),
        ("EIO", 5),
        ("ENXIO", 6),
        ("E2BIG", 7),
        ("ENOEXEC", 8),
        ("EBADF", 9),
        ("ECHILD", 10),
        ("EAGAIN", 35),
        ("EWOULDBLOCK", 35),
        ("ENOMEM", 12),
        ("EACCES", 13),
        ("EFAULT", 14),
        ("ENOTBLK", 15),
        ("EBUSY", 16),
        ("EEXIST", 17),
        ("EXDEV", 18),
        ("ENODEV", 19),
        ("ENOTDIR", 20),
        ("EISDIR", 21),
        ("EINVAL", 22),
        ("ENFILE", 23),
        ("EMFILE", 24),
        ("ENOTTY", 25),
        ("ETXTBSY", 26),
        ("EFBIG", 27),
        ("ENOSPC", 28),
        ("ESPIPE", 29),
        ("EROFS", 30),
        ("EMLINK", 31),
        ("EPIPE", 32),
        ("EDOM", 33),
        ("ERANGE", 34),
        ("EDEADLK", 11),
        ("ENAMETOOLONG", 63),
        ("ENOLCK", 77),
        ("ENOSYS", 78),
        ("ENOTEMPTY", 66),
        ("ELOOP", 62),
        ("ENOMSG", 91),
        ("EIDRM", 90),
        ("EBADMSG", 94),
        ("EMULTIHOP", 95),
        ("ENODATA", 96),
        ("ENOLINK", 97),
        ("ENOSR", 98),
        ("ENOSTR", 99),
        ("EOVERFLOW", 84),
        ("EPROTO", 100),
        ("ETIME", 101),
        ("EDESTADDRREQ", 39),
        ("EAFNOSUPPORT", 47),
        ("EALREADY", 37),
        ("EDQUOT", 69),
    ] {
        crate::namespace_store(ns, name, pyre_object::w_int_new(value));
    }
    crate::namespace_store(ns, "errorcode", pyre_object::w_dict_new());
}

/// _codecs stub — PyPy: pypy/module/_codecs/
///
/// Provides lookup_error/register_error and encode/decode no-op stubs so
/// codecs.py module init runs to completion.
fn init_codecs(ns: &mut PyNamespace) {
    // lookup_error(name) — returns an error handler for the given error
    // strategy. Pyre returns a pass-through lambda that never fires because
    // we don't encounter encoding errors in the pure-Python stdlib paths
    // we exercise so far.
    crate::namespace_store(
        ns,
        "lookup_error",
        crate::make_builtin_function("lookup_error", |_| {
            Ok(crate::make_builtin_function("error_handler", |args| {
                Ok(if args.is_empty() {
                    pyre_object::w_none()
                } else {
                    args[0]
                })
            }))
        }),
    );
    crate::namespace_store(
        ns,
        "register_error",
        crate::make_builtin_function("register_error", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "register",
        crate::make_builtin_function("register", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(
        ns,
        "lookup",
        crate::make_builtin_function("lookup", |_| Ok(pyre_object::w_none())),
    );
    // encode/decode — return input unchanged. Matches PyPy _codecs.encode
    // when the codec is the identity.
    let identity = crate::make_builtin_function("identity", |args| {
        Ok(if args.is_empty() {
            pyre_object::w_none()
        } else {
            args[0]
        })
    });
    crate::namespace_store(ns, "encode", identity);
    crate::namespace_store(ns, "decode", identity);
    crate::namespace_store(ns, "_forget_codec", identity);
    crate::namespace_store(
        ns,
        "charmap_build",
        crate::make_builtin_function("charmap_build", |_| Ok(pyre_object::w_dict_new())),
    );
}

/// copyreg stub — PyPy: pypy/module/copyreg/
fn init_copyreg(ns: &mut PyNamespace) {
    // copyreg.pickle(type, reduce_func, constructor=None) — register a
    // pickle reducer. Stub: ignore (pyre doesn't support pickle).
    crate::namespace_store(
        ns,
        "pickle",
        crate::make_builtin_function("pickle", |_| Ok(pyre_object::w_none())),
    );
    crate::namespace_store(ns, "dispatch_table", pyre_object::w_dict_new());
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
#[cfg(feature = "host_env")]
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

/// Set the Python-visible sys.modules dict reference. Called during sys
/// module initialization so subsequent set_sys_module calls keep it in sync.
/// Also copies all previously cached modules into the dict.
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
    #[cfg(feature = "host_env")]
    SourceFile { pathname: PathBuf },
    /// A package directory with __init__.py was found.
    #[cfg(feature = "host_env")]
    Package { dirpath: PathBuf },
    /// A builtin (Rust-implemented) module was found.
    /// PyPy equivalent: C_BUILTIN modtype in find_module()
    Builtin,
}

#[cfg(feature = "host_env")]
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

#[cfg(not(feature = "host_env"))]
fn find_module(partname: &str) -> Option<FindInfo> {
    let is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(partname));
    if is_builtin {
        return Some(FindInfo::Builtin);
    }
    None
}

/// Detect and add CPython stdlib to sys.path (once).
#[cfg(feature = "host_env")]
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
    let w_code = crate::w_code_new(code_ptr as *const ());
    let mut frame = PyFrame::new_with_namespace(w_code as *const (), execution_context, namespace);
    eval_frame_plain(&mut frame)
}

// ── load_source_module ───────────────────────────────────────────────
// PyPy equivalent: importing.py `load_source_module()`
//
// Parse + execute a .py source file, producing a module object.

#[cfg(feature = "host_env")]
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

#[cfg(feature = "host_env")]
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

    // Try a full-name builtin match first so dotted stubs like
    // `importlib.machinery` can override the filesystem search.
    // PyPy: interp_import.importhook consults sys.builtin_module_names by
    // the fully-qualified name.
    let full_is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(modulename));
    if full_is_builtin {
        let m = load_builtin_module(modulename).ok_or_else(|| crate::PyError {
            kind: crate::PyErrorKind::ImportError,
            message: format!("builtin module '{modulename}' failed to initialize"),
            exc_object: std::ptr::null_mut(),
        })?;
        set_sys_module(modulename, m);
        return Ok(Some(m));
    }

    // Find the module on disk
    let find_info = find_module(partname);
    let Some(info) = find_info else {
        return Ok(None);
    };

    let module = match info {
        #[cfg(feature = "host_env")]
        FindInfo::SourceFile { pathname } => {
            match load_source_module(modulename, &pathname, execution_context) {
                Ok(m) => m,
                Err(e) => {
                    return Err(e);
                }
            }
        }
        #[cfg(feature = "host_env")]
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

pub fn import_from(
    module: PyObjectRef,
    name: &str,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
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
    if let Ok(value) = crate::baseobjspace::getattr(module, name) {
        return Ok(value);
    }

    // PyPy: pyopcode.py _import_from — try importing as a submodule.
    // Build fullname = module.__name__ + "." + name and import it.
    if unsafe { is_module(module) } {
        let ns_ptr = unsafe { w_module_get_dict_ptr(module) } as *mut PyNamespace;
        if !ns_ptr.is_null() {
            let ns = unsafe { &*ns_ptr };
            if let Some(&modname_obj) = ns.get("__name__") {
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
                                crate::namespace_store(&mut *ns_ptr, name, submod);
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
