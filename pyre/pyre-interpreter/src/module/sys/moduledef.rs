//! sys module definition.
//!
//! PyPy equivalent: pypy/module/sys/

use crate::{PyNamespace, namespace_store};
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
            // typedef.py:34 — presence of `__dict__` in the init namespace
            // is what flips `hasdict` on the resulting type (see
            // typedef.rs:541/554). The value itself is never read; PyPy
            // stores a `GetSetProperty` there, but the Rust port only needs
            // the key to light up the per-instance attribute store.
            namespace_store(ns, "__dict__", w_none());
        });
        tp as usize
    });
    raw as PyObjectRef
}

/// Allocate a fresh stub instance whose type supports `setattr`. Used for
/// all the CPython-style attribute bags surfaced by the sys module.
fn make_sys_namespace_instance() -> PyObjectRef {
    w_instance_new(sys_namespace_type())
}

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "maxsize", w_int_new(i64::MAX));
    namespace_store(ns, "maxunicode", w_int_new(0x10FFFF));
    namespace_store(ns, "version", w_str_new("3.13.0 (pyre 0.0.1)"));
    namespace_store(
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
    namespace_store(
        ns,
        "byteorder",
        w_str_new(if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        }),
    );
    namespace_store(
        ns,
        "version_info",
        w_tuple_new(vec![
            w_int_new(3),
            w_int_new(13),
            w_int_new(0),
            w_str_new("final"),
            w_int_new(0),
        ]),
    );
    // sys.modules — live dict synced with the import cache.
    let modules_dict = w_dict_new();
    crate::importing::set_sys_modules_dict(modules_dict);
    namespace_store(ns, "modules", modules_dict);
    // sys.path — empty list placeholder
    namespace_store(ns, "path", w_list_new(vec![]));
    // sys.stdout/stderr/stdin — stub file-like objects.  Real CPython
    // wires these through io.TextIOWrapper around sys.__stdout__; pyre
    // exposes a tiny object with the bare minimum surface so anything
    // that writes status (unittest, traceback, warnings) keeps working.
    namespace_store(ns, "stdout", make_std_stream("<stdout>", false));
    namespace_store(ns, "stderr", make_std_stream("<stderr>", true));
    namespace_store(ns, "stdin", make_std_stream("<stdin>", false));
    namespace_store(ns, "__stdout__", make_std_stream("<stdout>", false));
    namespace_store(ns, "__stderr__", make_std_stream("<stderr>", true));
    namespace_store(ns, "__stdin__", make_std_stream("<stdin>", false));
    // sys._getframe — returns a stub frame object with f_locals/f_globals
    namespace_store(
        ns,
        "_getframe",
        crate::make_builtin_function("_getframe", |_| {
            let frame = make_sys_namespace_instance();
            let _ = crate::baseobjspace::setattr(frame, "f_locals", w_dict_new());
            let _ = crate::baseobjspace::setattr(frame, "f_globals", w_dict_new());
            let _ = crate::baseobjspace::setattr(frame, "f_code", w_none());
            let _ = crate::baseobjspace::setattr(frame, "f_back", w_none());
            let _ = crate::baseobjspace::setattr(frame, "f_lineno", w_int_new(0));
            Ok(frame)
        }),
    );
    // sys.exc_info — stub
    namespace_store(
        ns,
        "exc_info",
        crate::make_builtin_function("exc_info", |_| {
            Ok(w_tuple_new(vec![w_none(), w_none(), w_none()]))
        }),
    );
    // sys.flags — stub object with named attributes
    {
        let flags = make_sys_namespace_instance();
        let _ = crate::baseobjspace::setattr(flags, "debug", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "inspect", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "interactive", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "optimize", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "dont_write_bytecode", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "no_user_site", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "no_site", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "ignore_environment", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "verbose", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "bytes_warning", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "quiet", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "hash_randomization", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "isolated", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "dev_mode", w_bool_from(false));
        let _ = crate::baseobjspace::setattr(flags, "utf8_mode", w_int_new(1));
        let _ = crate::baseobjspace::setattr(flags, "warn_default_encoding", w_int_new(0));
        let _ = crate::baseobjspace::setattr(flags, "safe_path", w_bool_from(false));
        let _ = crate::baseobjspace::setattr(flags, "int_max_str_digits", w_int_new(4300));
        let _ = crate::baseobjspace::setattr(flags, "context_aware_warnings", w_bool_from(false));
        namespace_store(ns, "flags", flags);
    }
    // sys.getdefaultencoding
    namespace_store(
        ns,
        "getdefaultencoding",
        crate::make_builtin_function("getdefaultencoding", |_| Ok(w_str_new("utf-8"))),
    );
    // sys.getrecursionlimit / setrecursionlimit
    namespace_store(
        ns,
        "getrecursionlimit",
        crate::make_builtin_function("getrecursionlimit", |_| Ok(w_int_new(1000))),
    );
    namespace_store(
        ns,
        "setrecursionlimit",
        crate::make_builtin_function("setrecursionlimit", |_| Ok(w_none())),
    );
    // sys.intern
    namespace_store(
        ns,
        "intern",
        crate::make_builtin_function("intern", |args| {
            Ok(if args.is_empty() {
                w_str_new("")
            } else {
                args[0]
            })
        }),
    );
    // sys.implementation
    namespace_store(ns, "implementation", w_none());
    // sys.hash_info
    namespace_store(ns, "hash_info", w_none());
    // sys.float_info
    namespace_store(ns, "float_info", w_none());
    // sys.int_info
    namespace_store(ns, "int_info", w_none());
    // sys.executable
    namespace_store(ns, "executable", w_str_new("pyre"));
    // sys.prefix / exec_prefix
    namespace_store(ns, "prefix", w_str_new(""));
    namespace_store(ns, "exec_prefix", w_str_new(""));
    namespace_store(ns, "base_prefix", w_str_new(""));
    namespace_store(ns, "base_exec_prefix", w_str_new(""));
    // sys.argv
    namespace_store(ns, "argv", w_list_new(vec![]));
    // sys.warnoptions
    namespace_store(ns, "warnoptions", w_list_new(vec![]));
    // sys.builtin_module_names — tuple of names of modules compiled into
    // the interpreter. PyPy: pypy/module/sys/state.py get_builtin_module_names.
    // Pyre: include all stub/native built-ins from importing.rs.
    namespace_store(
        ns,
        "builtin_module_names",
        w_tuple_new(vec![
            w_str_new("_abc"),
            w_str_new("_bisect"),
            w_str_new("_blake2"),
            w_str_new("_codecs"),
            w_str_new("_collections"),
            w_str_new("_collections_abc"),
            w_str_new("_contextvars"),
            w_str_new("_csv"),
            w_str_new("_datetime"),
            w_str_new("_decimal"),
            w_str_new("_functools"),
            w_str_new("_hashlib"),
            w_str_new("_heapq"),
            w_str_new("_imp"),
            w_str_new("_io"),
            w_str_new("_json"),
            w_str_new("_locale"),
            w_str_new("_md5"),
            w_str_new("_opcode"),
            w_str_new("_operator"),
            w_str_new("_pickle"),
            w_str_new("_random"),
            w_str_new("_sha1"),
            w_str_new("_sha2"),
            w_str_new("_sha3"),
            w_str_new("_signal"),
            w_str_new("_socket"),
            w_str_new("_sre"),
            w_str_new("_stat"),
            w_str_new("_string"),
            w_str_new("_struct"),
            w_str_new("_thread"),
            w_str_new("_tokenize"),
            w_str_new("_tracemalloc"),
            w_str_new("_typing"),
            w_str_new("_warnings"),
            w_str_new("_weakref"),
            w_str_new("atexit"),
            w_str_new("binascii"),
            w_str_new("builtins"),
            w_str_new("copyreg"),
            w_str_new("errno"),
            w_str_new("fcntl"),
            w_str_new("grp"),
            w_str_new("itertools"),
            w_str_new("marshal"),
            w_str_new("math"),
            w_str_new("cmath"),
            w_str_new("operator"),
            w_str_new("posix"),
            w_str_new("pwd"),
            w_str_new("select"),
            w_str_new("sys"),
            w_str_new("time"),
        ]),
    );
    // sys.exception — returns the currently handled exception or None
    namespace_store(
        ns,
        "exception",
        crate::make_builtin_function("exception", |_| Ok(w_none())),
    );
    // sys.exc_clear — no-op
    namespace_store(
        ns,
        "exc_clear",
        crate::make_builtin_function("exc_clear", |_| Ok(w_none())),
    );
    // sys.gettrace / settrace — no-op
    namespace_store(
        ns,
        "gettrace",
        crate::make_builtin_function("gettrace", |_| Ok(w_none())),
    );
    namespace_store(
        ns,
        "settrace",
        crate::make_builtin_function("settrace", |_| Ok(w_none())),
    );
    // sys.getprofile / setprofile — no-op
    namespace_store(
        ns,
        "getprofile",
        crate::make_builtin_function("getprofile", |_| Ok(w_none())),
    );
    namespace_store(
        ns,
        "setprofile",
        crate::make_builtin_function("setprofile", |_| Ok(w_none())),
    );
    // sys.getfilesystemencoding
    namespace_store(
        ns,
        "getfilesystemencoding",
        crate::make_builtin_function("getfilesystemencoding", |_| Ok(w_str_new("utf-8"))),
    );
    namespace_store(
        ns,
        "getfilesystemencodeerrors",
        crate::make_builtin_function("getfilesystemencodeerrors", |_| {
            Ok(w_str_new("surrogateescape"))
        }),
    );
    // sys.audit — no-op
    namespace_store(
        ns,
        "audit",
        crate::make_builtin_function("audit", |_| Ok(w_none())),
    );
    // sys.is_finalizing
    namespace_store(
        ns,
        "is_finalizing",
        crate::make_builtin_function("is_finalizing", |_| Ok(w_bool_from(false))),
    );
    // sys.displayhook / excepthook
    namespace_store(
        ns,
        "displayhook",
        crate::make_builtin_function("displayhook", |_| Ok(w_none())),
    );
    namespace_store(
        ns,
        "excepthook",
        crate::make_builtin_function("excepthook", |_| Ok(w_none())),
    );
    // sys.path_hooks / path_importer_cache
    namespace_store(ns, "path_hooks", w_list_new(vec![]));
    namespace_store(ns, "path_importer_cache", w_dict_new());
    // sys.meta_path — empty
    namespace_store(ns, "meta_path", w_list_new(vec![]));
    // sys.dont_write_bytecode
    namespace_store(ns, "dont_write_bytecode", w_bool_from(true));
    // sys.addaudithook
    namespace_store(
        ns,
        "addaudithook",
        crate::make_builtin_function("addaudithook", |_| Ok(w_none())),
    );
}

/// Construct a stub stdio object exposing `write`, `flush`, `isatty`,
/// `fileno`, and `name`.  PyPy uses real W_File-backed objects via the io
/// module; pyre routes writes through Rust's stdout/stderr directly.
fn make_std_stream(name: &'static str, is_stderr: bool) -> PyObjectRef {
    let stream = make_sys_namespace_instance();
    let _ = crate::baseobjspace::setattr(stream, "name", w_str_new(name));
    let _ = crate::baseobjspace::setattr(stream, "encoding", w_str_new("utf-8"));
    let _ =
        crate::baseobjspace::setattr(stream, "mode", w_str_new(if is_stderr { "w" } else { "r" }));
    let _ = crate::baseobjspace::setattr(stream, "closed", w_bool_from(false));
    let _ = crate::baseobjspace::setattr(stream, "buffer", w_none());
    // ATTR_TABLE-stored builtin methods do not get `self` prepended (see
    // pyopcode load_method dispatch), so the first arg may be the string
    // directly. Pick whichever element is a real str.
    fn pick_str(args: &[PyObjectRef]) -> Option<&str> {
        for &a in args {
            if !a.is_null() && unsafe { is_str(a) } {
                return Some(unsafe { w_str_get_value(a) });
            }
        }
        None
    }
    let write_fn = if is_stderr {
        crate::make_builtin_function("write", |args| {
            use std::io::Write;
            if let Some(text) = pick_str(args) {
                let _ = std::io::stderr().write_all(text.as_bytes());
                return Ok(w_int_new(text.len() as i64));
            }
            Ok(w_int_new(0))
        })
    } else {
        crate::make_builtin_function("write", |args| {
            use std::io::Write;
            if let Some(text) = pick_str(args) {
                let _ = std::io::stdout().write_all(text.as_bytes());
                return Ok(w_int_new(text.len() as i64));
            }
            Ok(w_int_new(0))
        })
    };
    let _ = crate::baseobjspace::setattr(stream, "write", write_fn);
    let _ = crate::baseobjspace::setattr(
        stream,
        "flush",
        crate::make_builtin_function("flush", |_| {
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            Ok(w_none())
        }),
    );
    let _ = crate::baseobjspace::setattr(
        stream,
        "isatty",
        crate::make_builtin_function("isatty", |_| Ok(w_bool_from(false))),
    );
    let fileno_fn = if is_stderr {
        crate::make_builtin_function("fileno", |_| Ok(w_int_new(2)))
    } else {
        crate::make_builtin_function("fileno", |_| Ok(w_int_new(1)))
    };
    let _ = crate::baseobjspace::setattr(stream, "fileno", fileno_fn);
    let _ = crate::baseobjspace::setattr(
        stream,
        "writable",
        crate::make_builtin_function("writable", |_| Ok(w_bool_from(true))),
    );
    let _ = crate::baseobjspace::setattr(
        stream,
        "readable",
        crate::make_builtin_function("readable", |_| Ok(w_bool_from(false))),
    );
    stream
}
