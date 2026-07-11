//! `sys` module VM helpers.
//!
//! PyPy equivalent: `pypy/module/sys/vm.py`.

use crate::{DictStorage, dict_storage_store, make_builtin_function_with_arity};
use pyre_object::*;
use std::sync::OnceLock;

/// Shared stub type for `sys._getframe`, `sys.flags`, `sys.stdout` and other
/// module-level sys attributes that expose CPython-looking attribute bags.
///
/// `typedef::w_object()` (plain `object`) cannot store instance attributes —
/// its type flag `hasdict` is false, matching CPython where `object()`
/// instances reject `__setattr__` unless their subclass explicitly opts in.
/// PyPy's `sys` module exposes these as dedicated W_Root types with their
/// own typedefs, not as bare `object` instances. The Rust port mirrors that
/// by installing a single `sys.namespace` type with `__dict__` in its
/// typedef slots so every stub instance supports `setattr`.
fn sys_namespace_type() -> PyObjectRef {
    static TYPE: OnceLock<usize> = OnceLock::new();
    let raw = *TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("sys.namespace", |ns| {
            crate::dict_storage_store(
                ns,
                "__init__",
                crate::make_builtin_function("__init__", sys_namespace_init),
            );
        });
        // The stubs want a per-instance mapdict store; a `__dict__`
        // rawdict key would instead claim the typedef manages the dict
        // (typedef.py:40) and suppress the mapdict one
        // (typeobject.py:253-257), so flip `hasdict` directly — the
        // `create_dict_slot` flag flip (typeobject.py:1222-1226).
        unsafe { w_type_set_hasdict(tp, true) };
        tp as usize
    });
    raw as PyObjectRef
}

fn sys_namespace_init(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let Some(&self_obj) = positional.first() else {
        return Err(crate::PyError::type_error(
            "__init__() missing 1 required positional argument: 'self'",
        ));
    };
    if positional.len() > 1 {
        return Err(crate::PyError::type_error(
            "types.SimpleNamespace() takes no positional arguments",
        ));
    }
    if let Some(dict) = kwargs {
        unsafe {
            for (key, value) in pyre_object::w_dict_items(dict) {
                if pyre_object::is_str(key)
                    && pyre_object::w_str_get_wtf8(key).as_str() == Ok("__pyre_kw__")
                {
                    continue;
                }
                crate::baseobjspace::setattr(self_obj, key, value)?;
            }
        }
    }
    Ok(w_none())
}

/// Allocate a fresh stub instance whose type supports `setattr`. Used for
/// all the CPython-style attribute bags surfaced by the sys module.
fn make_sys_namespace_instance() -> PyObjectRef {
    w_instance_new(sys_namespace_type())
}


/// `pypy/module/sys/vm.py:217 space.getexecutioncontext()` access for
/// `sys.gettrace`/`settrace`/`getprofile`/`setprofile`.
///
/// Pyre's `crate::call::getexecutioncontext` returns the TLS-cached
/// active context (set on eval-loop entry); see the helper's doc for
/// the staleness gap relative to PyPy's `space.getexecutioncontext()`
/// which always queries the thread state.
fn current_execution_context() -> *mut crate::PyExecutionContext {
    crate::call::getexecutioncontext() as *mut crate::PyExecutionContext
}

fn sys_gettrace_impl(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    if ec.is_null() {
        return Ok(w_none());
    }
    let w_trace = unsafe { (*ec).gettrace() };
    Ok(if w_trace.is_null() { w_none() } else { w_trace })
}

fn sys_settrace_impl(args: &[PyObjectRef]) -> crate::PyResult {
    // pypy/module/sys/vm.py:217 `def settrace(space, w_func)` — w_func is
    // a required positional. Calling `sys.settrace()` with no args raises
    // TypeError at the gateway layer in PyPy; reproduce that here.
    let w_func = *args.first().ok_or_else(|| {
        crate::PyError::type_error("settrace() missing 1 required positional argument: 'function'")
    })?;
    let ec = current_execution_context();
    if !ec.is_null() {
        unsafe { (*ec).settrace(w_func) };
    }
    Ok(w_none())
}

fn sys_getprofile_impl(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    if ec.is_null() {
        return Ok(w_none());
    }
    let w_profile = unsafe { (*ec).getprofile() };
    Ok(if w_profile.is_null() {
        w_none()
    } else {
        w_profile
    })
}

fn sys_setprofile_impl(args: &[PyObjectRef]) -> crate::PyResult {
    // pypy/module/sys/vm.py:227 `def setprofile(space, w_func)` — w_func
    // is a required positional. Calling `sys.setprofile()` with no args
    // raises TypeError at the gateway layer in PyPy.
    let w_func = *args.first().ok_or_else(|| {
        crate::PyError::type_error(
            "setprofile() missing 1 required positional argument: 'function'",
        )
    })?;
    let ec = current_execution_context();
    if !ec.is_null() {
        // executioncontext.py:317-318 ValueError("Cannot call setllprofile
        // with real None") propagates via setprofile -> setllprofile.
        unsafe { (*ec).setprofile(w_func)? };
    }
    Ok(w_none())
}

fn sys_unraisablehook(args: &[PyObjectRef]) -> crate::PyResult {
    let Some(&w_hookargs) = args.first() else {
        return Err(crate::PyError::type_error(
            "unraisablehook() missing 1 required positional argument",
        ));
    };
    let w_type = crate::baseobjspace::getattr_str(w_hookargs, "exc_type")?;
    let w_value = crate::baseobjspace::getattr_str(w_hookargs, "exc_value")?;
    let w_tb = crate::baseobjspace::getattr_str(w_hookargs, "exc_traceback")?;
    let w_err_msg = crate::baseobjspace::getattr_str(w_hookargs, "err_msg")?;
    let err_msg = if unsafe { pyre_object::is_none(w_err_msg) } {
        String::new()
    } else if unsafe { pyre_object::is_str(w_err_msg) } {
        unsafe { pyre_object::w_str_get_value(w_err_msg) }.to_string()
    } else {
        unsafe { crate::display::py_str(w_err_msg)? }
    };
    let w_object = crate::baseobjspace::getattr_str(w_hookargs, "object")?;
    crate::PyError::write_unraisable_default(
        w_none(),
        w_type,
        w_value,
        w_tb,
        &err_msg,
        w_object,
        "",
    );
    Ok(w_none())
}

/// pypy/module/sys/vm.py `exc_info_direct` — return the active exception
/// as a `(type, value, traceback)` tuple.
///
/// Used by both the regular `sys.exc_info` builtin and the JIT direct path
/// in `function.funccall_valuestack` (function.py:146-150). Splitting it
/// out lets the JIT bypass invoke the same logic without going through the
/// builtin call dispatch.
pub fn exc_info_direct() -> PyObjectRef {
    let exc = crate::eval::get_current_exception();
    unsafe {
        if exc.is_null() || pyre_object::is_none(exc) || !pyre_object::is_exception(exc) {
            w_tuple_new(vec![w_none(), w_none(), w_none()])
        } else {
            // `pypy/module/sys/vm.py exc_info_direct` returns
            // `(type, value, traceback)` where `type` is
            // `space.exception_getclass(value)` — the specific
            // subclass W_TypeObject (e.g. `ZeroDivisionError`), not
            // the generic `Exception` stub set in
            // `w_exception_new`.  Pyre routes the per-`ExcKind`
            // lookup through `typedef::r#type` (`typedef.rs:186-197`)
            // which `exception_getclass` delegates to, so go through
            // that instead of dereferencing the raw `w_class` slot
            // (which still points at the constructor-time
            // `EXCEPTION_TYPE` stub).
            let exc_type = crate::baseobjspace::exception_getclass(exc);
            let exc_type = if exc_type.is_null() {
                w_none()
            } else {
                exc_type
            };
            // The third tuple slot mirrors
            // `space.exception_gettraceback(operror)`
            // (`error.py:140-145`).  Pyre stores the chain on the
            // typed `w_traceback` slot of `W_BaseException`
            // (`interp_exceptions.rs:303`); surface it directly here.
            let tb = pyre_object::interp_exceptions::w_exception_get_traceback(exc);
            let w_tb = if tb.is_null() { w_none() } else { tb };
            w_tuple_new(vec![exc_type, exc, w_tb])
        }
    }
}

pub fn register_module(ns: &mut DictStorage) {
    dict_storage_store(ns, "maxsize", w_int_new(i64::MAX));
    dict_storage_store(ns, "maxunicode", w_int_new(0x10FFFF));
    // Format matches `platform._sys_version`'s CPython parser:
    // `version (buildinfo) [compiler]`.
    dict_storage_store(ns, "version", w_str_new("3.14.6 (pyre 0.0.1) [Rust]"));
    dict_storage_store(
        ns,
        "platform",
        w_str_new(if cfg!(target_os = "macos") {
            "darwin"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "windows") {
            "win32"
        } else {
            "unknown"
        }),
    );
    dict_storage_store(
        ns,
        "byteorder",
        w_str_new(if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        }),
    );
    // sys.version_info — structseq(major, minor, micro, releaselevel,
    // serial); a tuple subclass so `>= (3, 14)` / `[0]` and `.major` both work.
    {
        let version_info_type = crate::_structseq::make_struct_seq(
            "sys.version_info",
            &["major", "minor", "micro", "releaselevel", "serial"],
        );
        let vi = crate::_structseq::new_instance(
            version_info_type,
            vec![
                w_int_new(3),
                w_int_new(14),
                w_int_new(6),
                w_str_new("final"),
                w_int_new(0),
            ],
        );
        dict_storage_store(ns, "version_info", vi);
    }
    // sys.modules — live dict synced with the import cache.
    let modules_dict = w_dict_new();
    crate::importing::set_sys_modules_dict(modules_dict);
    dict_storage_store(ns, "modules", modules_dict);
    // sys.path — empty list placeholder
    dict_storage_store(ns, "path", w_list_new(vec![]));
    // sys.stdout/stderr/stdin — `_io.TextIOWrapper`-typed file-like objects.
    // Real CPython wires these through io.TextIOWrapper around the std fds;
    // pyre exposes objects of the same type with the minimum surface so
    // anything that writes status (unittest, traceback, warnings) keeps
    // working.  `sys.__stdout__ is sys.stdout` (a single object each).
    let stdout = make_std_stream("<stdout>", 1);
    let stderr = make_std_stream("<stderr>", 2);
    let stdin = make_std_stream("<stdin>", 0);
    dict_storage_store(ns, "stdout", stdout);
    dict_storage_store(ns, "stderr", stderr);
    dict_storage_store(ns, "stdin", stdin);
    dict_storage_store(ns, "__stdout__", stdout);
    dict_storage_store(ns, "__stderr__", stderr);
    dict_storage_store(ns, "__stdin__", stdin);
    // `pypy/module/sys/vm.py:30 _getframe` walks the
    // `space.getexecutioncontext().gettopframe_nohidden()` chain,
    // following `f_back` `depth` times.  PyPy returns the frame
    // object directly so `frame.f_globals is module.__dict__` /
    // `frame.f_globals is globals()` (callee's scope) holds.  Pyre
    // mirrors the depth walk through `CURRENT_FRAME` + `f_back`,
    // populating the stub frame's attributes from the resolved
    // PyFrame.  `f_globals` / `f_locals` flow through
    // `dict_storage_to_dict` (canonical W_DictObject) so the
    // `is module.__dict__` invariant survives sys._getframe access.
    dict_storage_store(
        ns,
        "_getframe",
        crate::make_builtin_function("_getframe", |args| {
            // `pypy/module/sys/vm.py:28-39 _getframe`:
            //   @unwrap_spec(depth=int) def _getframe(space, depth=0)
            // `unwrap_spec` enforces a single optional int argument, so
            // any extra positional arg must surface as TypeError before
            // the depth walk runs.
            if args.len() > 1 {
                return Err(crate::PyError::type_error(format!(
                    "_getframe expected at most 1 argument, got {}",
                    args.len()
                )));
            }
            let depth_signed = if args.is_empty() {
                0i64
            } else if unsafe { pyre_object::is_int(args[0]) } {
                unsafe { pyre_object::w_int_get_value(args[0]) }
            } else {
                return Err(crate::PyError::type_error(
                    "_getframe(): argument must be an int",
                ));
            };
            // `vm.py:37-38 if depth < 0: raise ... "frame index must not
            // be negative"` — the message string differs from the
            // exhausted-stack case below.
            if depth_signed < 0 {
                return Err(crate::PyError::value_error(
                    "frame index must not be negative",
                ));
            }
            // `vm.py:44-54 getframe`: start from
            // `ec.gettopframe_nohidden()` and walk
            // `ec.getnextframe_nohidden(f)` `depth` times, so
            // `hidden_applevel` gateway / bridge frames are skipped
            // (matching `f_back`).  The `f is None` guard runs at the
            // *start* of every iteration including the first, so a
            // missing top frame raises rather than fabricating a stub.
            let ec = current_execution_context();
            let mut current = if ec.is_null() {
                std::ptr::null_mut()
            } else {
                unsafe { (*ec).gettopframe_nohidden() }
            };
            let mut remaining = depth_signed as usize;
            loop {
                if current.is_null() {
                    return Err(crate::PyError::value_error("call stack is not deep enough"));
                }
                if remaining == 0 {
                    break;
                }
                remaining -= 1;
                current =
                    crate::executioncontext::ExecutionContext::getnextframe_nohidden(current);
            }
            // `pyframe.py:767 f_back = GetSetProperty(PyFrame.fget_f_back)`.
            // Return the live `PyFrame` itself as the user-visible `frame`
            // object (`FRAME_TYPE` typedef); `f_back` chains lazily through
            // the getset.  Mark it escaped so the JIT keeps the frame
            // materialised for the exposed reference (pyframe.py:176
            // `mark_as_escaped`).
            unsafe { (*current).mark_as_escaped() };
            Ok(current as pyre_object::PyObjectRef)
        }),
    );
    // sys.exc_info() → (type, value, traceback)
    //
    // Tuple construction is shared with `exc_info_direct` (the JIT fast-path
    // entry registered below), so the regular call path and the JIT bypass
    // observe the same value.
    let exc_info_fn = make_builtin_function_with_arity("exc_info", |_| Ok(exc_info_direct()), 0);
    dict_storage_store(ns, "exc_info", exc_info_fn);
    // baseobjspace.py: register `space._code_of_sys_exc_info` so
    // `function.funccall_valuestack` can take the JIT direct path
    // (function.py:146-150). The builtin code pointer lives on the
    // `BuiltinCode` object backing `exc_info_fn`; `getcode` returns it.
    let exc_info_code = unsafe { crate::getcode(exc_info_fn) };
    crate::function::register_sys_exc_info_path(exc_info_code, exc_info_direct);
    // sys.flags — pypy/module/sys/app.py:99-119 `class sysflags` with
    // `__metaclass__ = structseqtype`. PyPy exposes it as a structseq
    // (immutable tuple subclass with named fields). pyre does not have
    // structseq yet, so we approximate the orthodox behavior with a
    // dedicated type whose attributes live in the TYPE's class
    // namespace rather than the instance dict. Read access via the
    // descriptor protocol still works (`sys.flags.optimize`); writes
    // fall through to `setdictvalue → raiseattrerror` because the type
    // has no `__dict__` slot, matching the read-only contract:
    //
    //     >>> sys.flags.optimize = 3
    //     AttributeError: 'sys.flags' object has no attribute 'optimize'
    //
    // The exact exception type differs from PyPy
    // (`pypy/module/sys/test/test_sysmodule.py:148` expects TypeError)
    // because pyre lacks the structseq tp_setattro slot. The full
    // structseq port is tracked separately.
    {
        let flags_type = crate::typedef::make_builtin_type("sys.flags", |fns| {
            dict_storage_store(fns, "debug", w_int_new(0));
            dict_storage_store(fns, "inspect", w_int_new(0));
            dict_storage_store(fns, "interactive", w_int_new(0));
            dict_storage_store(fns, "optimize", w_int_new(0));
            dict_storage_store(fns, "dont_write_bytecode", w_int_new(0));
            dict_storage_store(
                fns,
                "no_user_site",
                w_int_new(i64::from(crate::importing::no_user_site_flag())),
            );
            // `-S` (skip `import site`) is recorded by the launcher.
            dict_storage_store(
                fns,
                "no_site",
                w_int_new(i64::from(crate::importing::no_site_flag())),
            );
            dict_storage_store(
                fns,
                "ignore_environment",
                w_int_new(i64::from(crate::importing::ignore_environment_flag())),
            );
            dict_storage_store(fns, "verbose", w_int_new(0));
            dict_storage_store(fns, "bytes_warning", w_int_new(0));
            dict_storage_store(fns, "quiet", w_int_new(0));
            dict_storage_store(fns, "hash_randomization", w_int_new(0));
            dict_storage_store(
                fns,
                "isolated",
                w_int_new(i64::from(crate::importing::isolated_flag())),
            );
            dict_storage_store(
                fns,
                "dev_mode",
                w_bool_from(crate::importing::dev_mode_flag()),
            );
            dict_storage_store(
                fns,
                "utf8_mode",
                w_int_new(crate::importing::utf8_mode_flag()),
            );
            dict_storage_store(fns, "warn_default_encoding", w_int_new(0));
            dict_storage_store(
                fns,
                "safe_path",
                w_bool_from(crate::importing::safe_path_flag()),
            );
            dict_storage_store(fns, "int_max_str_digits", w_int_new(4300));
            dict_storage_store(fns, "context_aware_warnings", w_bool_from(false));
            dict_storage_store(fns, "thread_inherit_context", w_int_new(0));
        });
        let flags = w_instance_new(flags_type);
        dict_storage_store(ns, "flags", flags);
    }
    // sys.getdefaultencoding
    dict_storage_store(
        ns,
        "getdefaultencoding",
        make_builtin_function_with_arity("getdefaultencoding", |_| Ok(w_str_new("utf-8")), 0),
    );
    // sys.getrecursionlimit / setrecursionlimit — pypy/module/sys/vm.py:45.
    // The runtime stack budget lives in `crate::stack_check`; both
    // helpers route through it so the interpreter, JIT prologue probe,
    // and blackhole resume see a consistent recursion budget.
    dict_storage_store(
        ns,
        "getrecursionlimit",
        make_builtin_function_with_arity(
            "getrecursionlimit",
            |args| {
                // pypy/module/sys/vm.py:72 — no arguments.
                if !args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "getrecursionlimit() takes no arguments",
                    ));
                }
                Ok(w_int_new(crate::stack_check::get_recursion_limit() as i64))
            },
            0,
        ),
    );
    dict_storage_store(
        ns,
        "setrecursionlimit",
        make_builtin_function_with_arity(
            "setrecursionlimit",
            |args| {
                // pypy/module/sys/vm.py:63 `@unwrap_spec(new_limit="c_int")`
                // — exactly one positional argument, coerced through
                // baseobjspace.c_int_w (gateway_int_w + 32-bit range
                // check). `c_int_w` accepts int subclasses and any object
                // implementing `__int__`, rejects floats, and surfaces
                // out-of-range values as OverflowError.
                if args.len() != 1 {
                    return Err(crate::PyError::type_error(
                        "setrecursionlimit() takes exactly one argument",
                    ));
                }
                let new_limit = crate::baseobjspace::c_int_w(args[0])?;
                crate::stack_check::set_recursion_limit(new_limit)?;
                Ok(w_none())
            },
            1,
        ),
    );
    // sys.intern
    dict_storage_store(
        ns,
        "intern",
        make_builtin_function_with_arity(
            "intern",
            |args| {
                Ok(if args.is_empty() {
                    w_str_new("")
                } else {
                    args[0]
                })
            },
            1,
        ),
    );
    // sys.implementation — structseq-like namespace with name, version, ...
    {
        let impl_obj = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue(impl_obj, "name", w_str_new("pyre"));
        crate::baseobjspace::setdictvalue(
            impl_obj,
            "version",
            w_tuple_new(vec![
                w_int_new(3),
                w_int_new(14),
                w_int_new(6),
                w_str_new("final"),
                w_int_new(0),
            ]),
        );
        crate::baseobjspace::setdictvalue(impl_obj, "hexversion", w_int_new(0x030e06f0));
        crate::baseobjspace::setdictvalue(impl_obj, "cache_tag", w_str_new("pyre-3.14"));
        crate::baseobjspace::setdictvalue(impl_obj, "_multiarch", w_str_new(""));
        dict_storage_store(ns, "implementation", impl_obj);
    }
    // sys.hash_info — structseq with width/modulus/... fields.
    // PyPy: pypy/module/sys/system.py hash_info.
    {
        let hash_info = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue(hash_info, "width", w_int_new(64));
        crate::baseobjspace::setdictvalue(hash_info, "modulus", w_int_new((1i64 << 61) - 1));
        crate::baseobjspace::setdictvalue(hash_info, "inf", w_int_new(314159));
        crate::baseobjspace::setdictvalue(hash_info, "nan", w_int_new(0));
        crate::baseobjspace::setdictvalue(hash_info, "imag", w_int_new(1000003));
        crate::baseobjspace::setdictvalue(hash_info, "algorithm", w_str_new("siphash13"));
        crate::baseobjspace::setdictvalue(hash_info, "hash_bits", w_int_new(64));
        crate::baseobjspace::setdictvalue(hash_info, "seed_bits", w_int_new(128));
        crate::baseobjspace::setdictvalue(hash_info, "cutoff", w_int_new(0));
        dict_storage_store(ns, "hash_info", hash_info);
    }
    // sys.float_info — structseq with IEEE 754 double metadata.
    // PyPy: pypy/module/sys/system.py float_info.
    {
        let fi = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue(fi, "max", w_float_new(f64::MAX));
        crate::baseobjspace::setdictvalue(fi, "max_exp", w_int_new(1024));
        crate::baseobjspace::setdictvalue(fi, "max_10_exp", w_int_new(308));
        crate::baseobjspace::setdictvalue(fi, "min", w_float_new(f64::MIN_POSITIVE));
        crate::baseobjspace::setdictvalue(fi, "min_exp", w_int_new(-1021));
        crate::baseobjspace::setdictvalue(fi, "min_10_exp", w_int_new(-307));
        crate::baseobjspace::setdictvalue(fi, "dig", w_int_new(15));
        crate::baseobjspace::setdictvalue(fi, "mant_dig", w_int_new(53));
        crate::baseobjspace::setdictvalue(fi, "epsilon", w_float_new(f64::EPSILON));
        crate::baseobjspace::setdictvalue(fi, "radix", w_int_new(2));
        crate::baseobjspace::setdictvalue(fi, "rounds", w_int_new(1));
        dict_storage_store(ns, "float_info", fi);
    }
    // sysmodule.c — `sys.float_repr_style` is "short" wherever float repr
    // uses David Gay's shortest-round-trip algorithm (always, here).
    dict_storage_store(ns, "float_repr_style", w_str_new("short"));
    // sys.thread_info — structseq(name, lock, version).
    {
        let ti = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue(ti, "name", w_str_new("pthread"));
        crate::baseobjspace::setdictvalue(ti, "lock", w_str_new("semaphore"));
        crate::baseobjspace::setdictvalue(ti, "version", w_none());
        dict_storage_store(ns, "thread_info", ti);
    }
    // sys.int_info — structseq with int implementation details.
    {
        let ii = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue(ii, "bits_per_digit", w_int_new(30));
        crate::baseobjspace::setdictvalue(ii, "sizeof_digit", w_int_new(4));
        crate::baseobjspace::setdictvalue(ii, "default_max_str_digits", w_int_new(4300));
        crate::baseobjspace::setdictvalue(ii, "str_digits_check_threshold", w_int_new(640));
        dict_storage_store(ns, "int_info", ii);
    }
    dict_storage_store(ns, "hexversion", w_int_new(0x030e06f0));
    // sys.executable — absolute path to the running interpreter so that
    // subprocess spawns via `sys.executable` resolve.
    #[cfg(not(feature = "sandbox"))]
    let executable = std::env::current_exe()
        .ok()
        .and_then(|p| p.to_str().map(str::to_owned))
        .unwrap_or_else(|| "pyre".to_owned());
    // Under sandbox a fixed placeholder: current_exe() leaks the host binary
    // path (and username), and subprocess spawning is unavailable anyway.
    #[cfg(feature = "sandbox")]
    let executable = "/bin/pyre".to_owned();
    dict_storage_store(ns, "executable", w_str_new(&executable));
    // sys.prefix / exec_prefix
    dict_storage_store(ns, "prefix", w_str_new(""));
    dict_storage_store(ns, "exec_prefix", w_str_new(""));
    dict_storage_store(ns, "base_prefix", w_str_new(""));
    dict_storage_store(ns, "base_exec_prefix", w_str_new(""));
    // sys._framework — macOS framework name (empty string on non-framework builds)
    dict_storage_store(ns, "_framework", w_str_new(""));
    // sys._jit — namespace with is_enabled/is_available methods.
    // Python 3.14+ introduced sys._jit for CPython tier-2 JIT support checks.
    {
        let jit = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue(
            jit,
            "is_enabled",
            make_builtin_function_with_arity("is_enabled", |_| Ok(w_bool_from(false)), 0),
        );
        crate::baseobjspace::setdictvalue(
            jit,
            "is_available",
            make_builtin_function_with_arity("is_available", |_| Ok(w_bool_from(false)), 0),
        );
        dict_storage_store(ns, "_jit", jit);
    }
    // sys.monitoring — PEP 669 low-impact monitoring API. The runtime hooks
    // are stubbed (no events ever fire), but the namespace, tool-id
    // constants, sentinels, and `events` flags are present so importing
    // modules (bdb/pdb/cProfile/coverage tooling) succeed and can build
    // their tracer objects.
    {
        let mon = make_sys_namespace_instance();
        // Tool-id constants (Python/instrumentation.c).
        for (name, id) in [
            ("DEBUGGER_ID", 0),
            ("COVERAGE_ID", 1),
            ("PROFILER_ID", 2),
            ("OPTIMIZER_ID", 5),
        ] {
            crate::baseobjspace::setdictvalue(mon, name, w_int_new(id));
        }
        // DISABLE / MISSING sentinels — distinct singleton objects compared
        // by identity (`callback() == DISABLE`, `assertIs(x, MISSING)`).
        crate::baseobjspace::setdictvalue(mon, "DISABLE", make_sys_namespace_instance());
        crate::baseobjspace::setdictvalue(mon, "MISSING", make_sys_namespace_instance());
        // events namespace — `1 << event_id` flags that OR together.
        {
            let events = make_sys_namespace_instance();
            crate::baseobjspace::setdictvalue(events, "NO_EVENTS", w_int_new(0));
            for (i, name) in [
                "PY_START",
                "PY_RESUME",
                "PY_RETURN",
                "PY_YIELD",
                "CALL",
                "LINE",
                "INSTRUCTION",
                "JUMP",
                "BRANCH_LEFT",
                "BRANCH_RIGHT",
                "STOP_ITERATION",
                "RAISE",
                "EXCEPTION_HANDLED",
                "PY_UNWIND",
                "PY_THROW",
                "RERAISE",
                "C_RETURN",
                "C_RAISE",
            ]
            .iter()
            .enumerate()
            {
                crate::baseobjspace::setdictvalue(events, name, w_int_new(1i64 << i));
            }
            // BRANCH retained as an alias of BRANCH_LEFT for callers predating
            // the 3.14 left/right split.
            crate::baseobjspace::setdictvalue(events, "BRANCH", w_int_new(1i64 << 8));
            crate::baseobjspace::setdictvalue(mon, "events", events);
        }
        // Runtime hooks — no-op stubs returning sensible defaults.
        let store_fn = |obj, name: &'static str, f: crate::gateway::BuiltinCodeFn, arity: u16| {
            crate::baseobjspace::setdictvalue(
                obj,
                name,
                make_builtin_function_with_arity(name, f, arity),
            );
        };
        store_fn(mon, "use_tool_id", |_| Ok(w_none()), 2);
        store_fn(mon, "free_tool_id", |_| Ok(w_none()), 1);
        store_fn(mon, "clear_tool_id", |_| Ok(w_none()), 1);
        store_fn(mon, "get_tool", |_| Ok(w_none()), 1);
        store_fn(mon, "register_callback", |_| Ok(w_none()), 3);
        store_fn(mon, "set_events", |_| Ok(w_none()), 2);
        store_fn(mon, "get_events", |_| Ok(w_int_new(0)), 1);
        store_fn(mon, "set_local_events", |_| Ok(w_none()), 3);
        store_fn(mon, "get_local_events", |_| Ok(w_int_new(0)), 2);
        store_fn(mon, "restart_events", |_| Ok(w_none()), 0);
        dict_storage_store(ns, "monitoring", mon);
    }
    // sys.platlibdir — typically "lib" on POSIX; used by sysconfig to
    // construct install paths.
    dict_storage_store(ns, "platlibdir", w_str_new("lib"));
    // `sys/app.py:114-126 exit(exitcode=None)` — raise SystemExit(exitcode),
    // de-tupelizing a tuple argument so `exit((a, b))` becomes
    // `SystemExit(a, b)` (the extra de-tupelizing normalize_exception does
    // for `raise SystemExit, exitcode`).  A bare `exit()` defaults exitcode
    // to None, so the instance carries `code = None` / `args = (None,)`.
    // Interpreting the code (None → 0, int() coercion,
    // print-non-integral-and-exit-1) is the launcher's job
    // (`app_main.py:114-129 handle_sys_exit`).
    dict_storage_store(
        ns,
        "exit",
        crate::make_builtin_function("exit", |args| {
            // `exit(exitcode=None)` — resolve the single optional argument
            // like the app-level signature: strip the `__pyre_kw__` trailer,
            // reject unknown keywords, reproduce the normal function-call
            // arity diagnostics, and reject a positional/`exitcode=`
            // duplicate.
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(kwargs, &["exitcode"], "exit")?;
            if positional.len() > 1 {
                return Err(crate::PyError::type_error(format!(
                    "exit() takes from 0 to 1 positional arguments but {} were given",
                    positional.len()
                )));
            }
            let kw_exitcode = crate::builtins::kwarg_get(kwargs, "exitcode");
            if !positional.is_empty() && kw_exitcode.is_some() {
                return Err(crate::PyError::type_error(
                    "exit() got multiple values for argument 'exitcode'",
                ));
            }
            let exitcode = positional
                .first()
                .copied()
                .or(kw_exitcode)
                .unwrap_or_else(w_none);
            let cls = crate::builtins::lookup_exc_class("SystemExit")
                .ok_or_else(|| crate::PyError::runtime_error("SystemExit class missing"))?;
            let ctor_args = if unsafe { is_tuple(exitcode) } {
                unsafe { w_tuple_items_copy_as_vec(exitcode) }
            } else {
                vec![exitcode]
            };
            let exc = crate::call::call_function_impl_result(cls, &ctor_args)?;
            Err(unsafe { crate::PyError::from_exc_object(exc) })
        }),
    );
    // sys.abiflags
    dict_storage_store(ns, "abiflags", w_str_new(""));
    // sys.argv — pick up pending argv from set_sys_argv if available.
    let pending = crate::importing::take_pending_sys_argv();
    let argv = if pending.is_null() {
        w_list_new(vec![])
    } else {
        pending
    };
    dict_storage_store(ns, "argv", argv);
    // sys.warnoptions
    dict_storage_store(ns, "warnoptions", w_list_new(vec![]));
    // sys.builtin_module_names — tuple of names of modules compiled into
    // the interpreter. PyPy: pypy/module/sys/state.py get_builtin_module_names.
    // Pyre: include all stub/native built-ins from importing.rs. The host-access
    // modules importing.rs omits under `sandbox` are gated out here too, so the
    // advertised set matches what is actually importable.
    #[allow(unused_mut)]
    let mut builtin_names = vec![
        "__pypy__", "_abc", "_bisect", "_blake2", "_codecs", "_collections",
        "_collections_abc", "_contextvars", "_csv", "_datetime", "_decimal",
        "_functools", "_hashlib", "_heapq", "_imp", "_io", "_json", "_locale",
        "_md5", "_opcode", "_operator", "_pickle", "_random", "_sha1", "_sha2",
        "_sha3", "_signal", "_socket", "_sre", "_stat", "_string", "_struct",
        "_thread", "_tokenize", "_tracemalloc", "_typing", "_warnings", "_weakref",
        "atexit", "binascii", "builtins", "errno", "fcntl", "grp", "itertools",
        "marshal", "math", "cmath", "operator", "posix", "pwd", "select", "sys",
        "time",
    ];
    // Host-access modules registered only in non-sandbox builds (importing.rs);
    // under `sandbox` they are omitted, so drop them from the advertised set
    // in place — the surrounding order is left untouched.
    #[cfg(feature = "sandbox")]
    builtin_names.retain(|n| !matches!(*n, "_signal" | "_socket" | "fcntl" | "grp" | "pwd" | "select"));
    dict_storage_store(
        ns,
        "builtin_module_names",
        w_tuple_new(builtin_names.into_iter().map(w_str_new).collect()),
    );
    // sys.stdlib_module_names — frozenset of stdlib module names, read by
    // `traceback.TracebackException` (`wrong_name in sys.stdlib_module_names`)
    // to offer "did you forget to import" hints.  Seeded from the
    // compiled-in builtin module names; the full pure-Python stdlib set is
    // not enumerated yet, so a name absent here simply yields no hint
    // rather than a crash.
    dict_storage_store(
        ns,
        "stdlib_module_names",
        pyre_object::setobject::w_frozenset_from_items(&[
            w_str_new("sys"),
            w_str_new("builtins"),
            w_str_new("_thread"),
            w_str_new("time"),
            w_str_new("errno"),
            w_str_new("_io"),
            w_str_new("marshal"),
            w_str_new("_imp"),
            w_str_new("gc"),
            w_str_new("_warnings"),
            w_str_new("_string"),
            w_str_new("_codecs"),
            w_str_new("_weakref"),
            w_str_new("_operator"),
            w_str_new("_collections"),
            w_str_new("_functools"),
            w_str_new("itertools"),
            w_str_new("atexit"),
        ]),
    );
    // sys.exception() — the value half of `sys.exc_info()`: the exception
    // instance currently being handled, or None outside an `except` block.
    dict_storage_store(
        ns,
        "exception",
        make_builtin_function_with_arity(
            "exception",
            |_| {
                let exc = crate::eval::get_current_exception();
                Ok(unsafe {
                    if exc.is_null() || !pyre_object::is_exception(exc) {
                        w_none()
                    } else {
                        exc
                    }
                })
            },
            0,
        ),
    );
    // sys.exc_clear — no-op
    dict_storage_store(
        ns,
        "exc_clear",
        make_builtin_function_with_arity("exc_clear", |_| Ok(w_none()), 0),
    );
    // sys.is_remote_debug_enabled() — no remote-debug interface is wired,
    // so always False.
    dict_storage_store(
        ns,
        "is_remote_debug_enabled",
        make_builtin_function_with_arity(
            "is_remote_debug_enabled",
            |_| Ok(pyre_object::w_bool_from(false)),
            0,
        ),
    );
    // sys.copyright — informational string consumed by `site` and `test`.
    dict_storage_store(
        ns,
        "copyright",
        w_str_new("Copyright (c) 2001-2024 Python Software Foundation.\nAll Rights Reserved."),
    );
    // sys.getsizeof(obj[, default]) — PyPy vm.py returns the supplied default
    // for untracked objects.  str additionally exposes its PEP 393-compatible
    // `__sizeof__`, needed by the shared CPython test_str overflow check.
    dict_storage_store(
        ns,
        "getsizeof",
        make_builtin_function_with_arity(
            "getsizeof",
            |args| {
                if unsafe { pyre_object::is_str(args[0]) } {
                    let method = crate::baseobjspace::getattr_str(args[0], "__sizeof__")?;
                    return crate::call::call_function_impl_result(method, &[]);
                }
                match args.get(1).copied() {
                    Some(w_default) => Ok(w_default),
                    None => Err(crate::PyError::type_error(
                        "getsizeof(object, default) -> int: object size is not tracked; supply a default",
                    )),
                }
            },
            1,
        ),
    );
    // sys.gettrace / settrace
    dict_storage_store(
        ns,
        "gettrace",
        make_builtin_function_with_arity("gettrace", sys_gettrace_impl, 0),
    );
    dict_storage_store(
        ns,
        "settrace",
        make_builtin_function_with_arity("settrace", sys_settrace_impl, 1),
    );
    // sys.getprofile / setprofile
    dict_storage_store(
        ns,
        "getprofile",
        make_builtin_function_with_arity("getprofile", sys_getprofile_impl, 0),
    );
    dict_storage_store(
        ns,
        "setprofile",
        make_builtin_function_with_arity("setprofile", sys_setprofile_impl, 1),
    );
    // sys.getfilesystemencoding
    dict_storage_store(
        ns,
        "getfilesystemencoding",
        make_builtin_function_with_arity("getfilesystemencoding", |_| Ok(w_str_new("utf-8")), 0),
    );
    dict_storage_store(
        ns,
        "getfilesystemencodeerrors",
        make_builtin_function_with_arity(
            "getfilesystemencodeerrors",
            |_| Ok(w_str_new("surrogateescape")),
            0,
        ),
    );
    // sys.audit — no-op
    dict_storage_store(
        ns,
        "audit",
        crate::make_builtin_function("audit", |_| Ok(w_none())),
    );
    // sys._clear_type_descriptors(cls) — clears a type's cached descriptors
    // before `dataclasses._add_slots` rebuilds the class with __slots__.  The
    // rebuilt class is constructed fresh from the original's dict, so dropping
    // the descriptors on the soon-discarded original is a no-op here.
    dict_storage_store(
        ns,
        "_clear_type_descriptors",
        crate::make_builtin_function("_clear_type_descriptors", |_| Ok(w_none())),
    );
    // sys.is_finalizing
    dict_storage_store(
        ns,
        "is_finalizing",
        make_builtin_function_with_arity("is_finalizing", |_| Ok(w_bool_from(false)), 0),
    );
    // sys.displayhook / excepthook. `__displayhook__` keeps the original so
    // code (e.g. doctest) can save and restore the hook.
    dict_storage_store(
        ns,
        "displayhook",
        make_builtin_function_with_arity("displayhook", crate::builtins::sys_displayhook, 1),
    );
    dict_storage_store(
        ns,
        "__displayhook__",
        make_builtin_function_with_arity("displayhook", crate::builtins::sys_displayhook, 1),
    );
    dict_storage_store(
        ns,
        "excepthook",
        make_builtin_function_with_arity("excepthook", |_| Ok(w_none()), 3),
    );
    // sys.unraisablehook(unraisable) — handles exceptions raised where they
    // cannot propagate (e.g. __del__).  Stored alongside the read-only
    // `__unraisablehook__` original so code can save and restore it.
    let unraisablehook_fn =
        make_builtin_function_with_arity("unraisablehook", sys_unraisablehook, 1);
    dict_storage_store(ns, "unraisablehook", unraisablehook_fn);
    dict_storage_store(ns, "__unraisablehook__", unraisablehook_fn);
    // sys.path_hooks / path_importer_cache
    dict_storage_store(ns, "path_hooks", w_list_new(vec![]));
    dict_storage_store(ns, "path_importer_cache", w_dict_new());
    // sys.meta_path — empty
    dict_storage_store(ns, "meta_path", w_list_new(vec![]));
    // sys.dont_write_bytecode
    dict_storage_store(ns, "dont_write_bytecode", w_bool_from(true));
    // sys.pycache_prefix — None unless -X pycache_prefix / PYTHONPYCACHEPREFIX.
    // `importlib._bootstrap_external.cache_from_source` reads it to compute the
    // bytecode path before `dont_write_bytecode` is consulted.
    dict_storage_store(ns, "pycache_prefix", w_none());
    // sys.addaudithook
    dict_storage_store(
        ns,
        "addaudithook",
        make_builtin_function_with_arity("addaudithook", |_| Ok(w_none()), 1),
    );
}

/// Construct a stdio object whose type is `_io.TextIOWrapper` (so
/// `isinstance(sys.stdout, io.TextIOWrapper)` holds), exposing `write`,
/// `flush`, `isatty`, `fileno`, `reconfigure`, and `name`.  `fd` is the
/// descriptor it reports: 0 (stdin) / 1 (stdout) / 2 (stderr).  PyPy wires a
/// real W_File-backed `TextIOWrapper`; pyre routes writes through Rust's
/// stdout/stderr (the same sink as `print`) so output ordering is preserved,
/// storing the read/write surface as instance attributes.
fn make_std_stream(name: &'static str, fd: i32) -> PyObjectRef {
    let writable = fd != 0;
    let to_stderr = fd == 2;
    let stream = pyre_object::w_instance_new(crate::builtins::text_io_wrapper_type());
    crate::baseobjspace::setdictvalue(stream, "name", w_str_new(name));
    crate::baseobjspace::setdictvalue(stream, "encoding", w_str_new("utf-8"));
    // `pylifecycle.c init_set_builtins_open`/`init_sys_streams`: stderr uses the
    // `backslashreplace` handler so traceback printing never fails on a lone
    // surrogate; stdout/stdin default to `strict`.
    crate::baseobjspace::setdictvalue(
        stream,
        "errors",
        w_str_new(if to_stderr { "backslashreplace" } else { "strict" }),
    );
    crate::baseobjspace::setdictvalue(
        stream,
        "mode",
        w_str_new(if writable { "w" } else { "r" }),
    );
    crate::baseobjspace::setdictvalue(stream, "closed", w_bool_from(false));
    crate::baseobjspace::setdictvalue(stream, "buffer", w_none());
    // Instance-stored builtin methods do not get `self` prepended (see
    // pyopcode load_method dispatch), so the first arg may be the string
    // directly. Pick whichever element is a real str.
    fn pick_str(args: &[PyObjectRef]) -> Option<PyObjectRef> {
        for &a in args {
            if !a.is_null() && unsafe { is_str(a) } {
                return Some(a);
            }
        }
        None
    }
    // Encode through `encode_object` with the stream's error handler so a lone
    // surrogate is routed there (stdout `strict` → UnicodeEncodeError; stderr
    // `backslashreplace` → escaped) instead of panicking in `w_str_get_value`.
    let write_fn = if to_stderr {
        crate::make_builtin_function("write", |args| {
            if let Some(s_obj) = pick_str(args) {
                let bytes = crate::type_methods::encode_object(s_obj, "utf-8", "backslashreplace")?;
                // Under sandbox fd 1 is the marshalling pipe, so a raw write
                // would corrupt the protocol: route through ll_os_write(2,…)
                // and let the controller relay it to its own stderr.
                #[cfg(not(feature = "sandbox"))]
                {
                    use std::io::Write;
                    let _ = std::io::stderr().write_all(&bytes);
                }
                #[cfg(feature = "sandbox")]
                crate::host_seam::ops::write(2, &bytes)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                return Ok(w_int_new(unsafe { w_str_len(s_obj) } as i64));
            }
            Ok(w_int_new(0))
        })
    } else {
        crate::make_builtin_function("write", |args| {
            if let Some(s_obj) = pick_str(args) {
                let bytes = crate::type_methods::encode_object(s_obj, "utf-8", "strict")?;
                #[cfg(not(feature = "sandbox"))]
                {
                    use std::io::Write;
                    let _ = std::io::stdout().write_all(&bytes);
                }
                #[cfg(feature = "sandbox")]
                crate::host_seam::ops::write(1, &bytes)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                return Ok(w_int_new(unsafe { w_str_len(s_obj) } as i64));
            }
            Ok(w_int_new(0))
        })
    };
    crate::baseobjspace::setdictvalue(stream, "write", write_fn);
    crate::baseobjspace::setdictvalue(
        stream,
        "flush",
        crate::make_builtin_function("flush", |_| {
            // The sandbox path writes unbuffered ll_os_write requests, so there
            // is nothing to flush (and the real fds are the marshalling pipe).
            #[cfg(not(feature = "sandbox"))]
            {
                use std::io::Write;
                let _ = std::io::stdout().flush();
                let _ = std::io::stderr().flush();
            }
            Ok(w_none())
        }),
    );
    crate::baseobjspace::setdictvalue(
        stream,
        "isatty",
        crate::make_builtin_function("isatty", |_| Ok(w_bool_from(false))),
    );
    // `TextIOWrapper.reconfigure(*, encoding=None, errors=None, ...)` only
    // adjusts codec/newline policy; pyre's streams are fixed UTF-8, so accept
    // and ignore the request.
    crate::baseobjspace::setdictvalue(
        stream,
        "reconfigure",
        crate::make_builtin_function("reconfigure", |_| Ok(w_none())),
    );
    // `BuiltinCodeFn` is a bare `fn` pointer (no captures), so select a
    // constant-returning function per descriptor rather than closing over `fd`.
    let fileno_fn = match fd {
        0 => crate::make_builtin_function("fileno", |_| Ok(w_int_new(0))),
        2 => crate::make_builtin_function("fileno", |_| Ok(w_int_new(2))),
        _ => crate::make_builtin_function("fileno", |_| Ok(w_int_new(1))),
    };
    crate::baseobjspace::setdictvalue(stream, "fileno", fileno_fn);
    let (writable_fn, readable_fn) = if writable {
        (
            crate::make_builtin_function("writable", |_| Ok(w_bool_from(true))),
            crate::make_builtin_function("readable", |_| Ok(w_bool_from(false))),
        )
    } else {
        (
            crate::make_builtin_function("writable", |_| Ok(w_bool_from(false))),
            crate::make_builtin_function("readable", |_| Ok(w_bool_from(true))),
        )
    };
    crate::baseobjspace::setdictvalue(stream, "writable", writable_fn);
    crate::baseobjspace::setdictvalue(stream, "readable", readable_fn);
    stream
}
