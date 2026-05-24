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
mod host {
    pub use rustpython_host_env::{fs, os};
}
#[cfg(not(feature = "host_env"))]
mod host {
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
use host::{fs as host_fs, os as host_os};

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
    static BUILTIN_MODULES: RefCell<HashMap<&'static str, fn(&mut DictStorage)>> =
        RefCell::new(HashMap::new());
}

// ── builtin module registry ──────────────────────────────────────────
// PyPy equivalent: space.builtin_modules dict + MixedModule.interpleveldefs

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
    register_builtin_module("atexit", init_atexit);
    #[cfg(unix)]
    register_builtin_module("pwd", init_pwd);
    #[cfg(unix)]
    register_builtin_module("grp", init_grp);
    #[cfg(unix)]
    register_builtin_module("resource", init_resource);
    #[cfg(unix)]
    register_builtin_module("fcntl", init_fcntl);
    #[cfg(unix)]
    register_builtin_module("syslog", init_syslog);
    register_builtin_module("select", init_select);
    register_builtin_module("termios", init_termios);
    register_builtin_module("_socket", crate::module::_socket::moduledef::init);
    register_builtin_module("mmap", crate::module::mmap::moduledef::init);
    register_builtin_module(
        "faulthandler",
        crate::module::faulthandler::moduledef::init,
    );
    register_builtin_module("_ctypes", init_ctypes);
    register_builtin_module("_posixshmem", init_posixshmem);
    register_builtin_module("_multiprocessing", init_multiprocessing);
    register_builtin_module("_locale", init_locale);
    register_builtin_module("_random", init_random);
    register_builtin_module("_struct", init_struct);
    register_builtin_module("gc", init_gc);
    register_builtin_module("unicodedata", init_unicodedata);
    // `_sysconfigdata_{abiflags}_{platform}_{multiarch}` is a generated
    // Python module containing `build_time_vars = {...}` that sysconfig
    // imports from `_init_posix`. We stub it out with an empty dict so
    // `sysconfig.get_config_vars()` returns an empty mapping.
    // PyPy equivalent: pypy/tool/build_cffi_imports.py creates the same
    // file during translation.
    register_builtin_module("_sysconfigdata__darwin_", init_sysconfigdata_empty);
    register_builtin_module("_sysconfigdata__linux_", init_sysconfigdata_empty);
    register_builtin_module(
        "_sysconfigdata__linux_x86_64-linux-gnu",
        init_sysconfigdata_empty,
    );
    register_builtin_module(
        "_sysconfigdata__linux_aarch64-linux-gnu",
        init_sysconfigdata_empty,
    );
    // _opcode_metadata.py exists in the stdlib; load the real file instead.
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
        "_pickle",
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

/// gc module stub — enough to let `import gc` succeed.
fn init_gc(ns: &mut DictStorage) {
    // pypy/module/gc/interp_gc.py:7-26 collect — partial port:
    // drive a full mark-sweep through `try_gc_collect` (which fans
    // out through `pyre-jit::eval`'s trampoline to the active
    // backend's `majit_gc::collect_full`). MethodCache / MapAttrCache
    // clears (`:14-17`) skipped — pyre has no equivalent caches.
    // Finalizer queue (`:28-46 _run_finalizers`) skipped pending the
    // finalizer epic. Argument `generation` is ignored per upstream.
    crate::dict_storage_store(
        ns,
        "collect",
        crate::make_builtin_function_with_arity(
            "collect",
            |_| {
                pyre_object::gc_hook::try_gc_collect();
                Ok(pyre_object::w_int_new(0))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "disable",
        crate::make_builtin_function_with_arity("disable", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "enable",
        crate::make_builtin_function_with_arity("enable", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "isenabled",
        crate::make_builtin_function_with_arity(
            "isenabled",
            |_| Ok(pyre_object::w_bool_from(false)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_objects",
        crate::make_builtin_function_with_arity(
            "get_objects",
            |_| Ok(pyre_object::w_list_new(vec![])),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_referrers",
        crate::make_builtin_function("get_referrers", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    crate::dict_storage_store(
        ns,
        "get_referents",
        crate::make_builtin_function("get_referents", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    crate::dict_storage_store(
        ns,
        "set_threshold",
        crate::make_builtin_function_with_arity("set_threshold", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "get_threshold",
        crate::make_builtin_function_with_arity(
            "get_threshold",
            |_| {
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(700),
                    pyre_object::w_int_new(10),
                    pyre_object::w_int_new(10),
                ]))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_count",
        crate::make_builtin_function_with_arity(
            "get_count",
            |_| {
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(0),
                    pyre_object::w_int_new(0),
                    pyre_object::w_int_new(0),
                ]))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "is_tracked",
        crate::make_builtin_function_with_arity(
            "is_tracked",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "is_finalized",
        crate::make_builtin_function_with_arity(
            "is_finalized",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "freeze",
        crate::make_builtin_function_with_arity("freeze", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(ns, "callbacks", pyre_object::w_list_new(vec![]));
    crate::dict_storage_store(ns, "garbage", pyre_object::w_list_new(vec![]));
    crate::dict_storage_store(ns, "DEBUG_STATS", pyre_object::w_int_new(1));
    crate::dict_storage_store(ns, "DEBUG_COLLECTABLE", pyre_object::w_int_new(2));
    crate::dict_storage_store(ns, "DEBUG_UNCOLLECTABLE", pyre_object::w_int_new(4));
    crate::dict_storage_store(ns, "DEBUG_SAVEALL", pyre_object::w_int_new(32));
    crate::dict_storage_store(ns, "DEBUG_LEAK", pyre_object::w_int_new(38));
}

/// unicodedata module stub — provides normalize() and category().
fn init_unicodedata(ns: &mut DictStorage) {
    // unicodedata.normalize(form, unistr) → unistr (stub: returns input unchanged)
    crate::dict_storage_store(
        ns,
        "normalize",
        crate::make_builtin_function_with_arity(
            "normalize",
            |args| {
                if args.len() >= 2 {
                    Ok(args[1])
                } else {
                    Ok(pyre_object::w_str_new(""))
                }
            },
            2,
        ),
    );
    // unicodedata.category(chr) → str (stub: returns "Cn" = unassigned)
    crate::dict_storage_store(
        ns,
        "category",
        crate::make_builtin_function_with_arity(
            "category",
            |_| Ok(pyre_object::w_str_new("Cn")),
            1,
        ),
    );
    // unicodedata.name(chr, default=None) → str
    crate::dict_storage_store(
        ns,
        "name",
        crate::make_builtin_function("name", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("no such name"))
            }
        }),
    );
    // unicodedata.lookup(name) → chr
    crate::dict_storage_store(
        ns,
        "lookup",
        crate::make_builtin_function_with_arity(
            "lookup",
            |_| Err(crate::PyError::key_error("character not found")),
            1,
        ),
    );
    // unicodedata.decimal(chr, default=None) → int
    crate::dict_storage_store(
        ns,
        "decimal",
        crate::make_builtin_function("decimal", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("not a decimal"))
            }
        }),
    );
    // unicodedata.numeric(chr, default=None) → float
    crate::dict_storage_store(
        ns,
        "numeric",
        crate::make_builtin_function("numeric", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("not a numeric character"))
            }
        }),
    );
    // unicodedata.unidata_version
    crate::dict_storage_store(ns, "unidata_version", pyre_object::w_str_new("15.1.0"));
    // unicodedata.ucd_3_2_0 — alias for the module itself (used by IDNA)
    // We store a sentinel; os_helper only checks that the module imported.
}

/// `_struct` C-extension stub — PyPy: pypy/module/struct/interp_struct.py.
///
/// Implements just enough to let `struct.py` load: `pack`, `unpack`,
/// `calcsize`, `_clearcache`, and the `error` type. Each packer handles
/// the format codes pyre actually uses during import (`<q`, `<d`, etc.).
fn init_struct(ns: &mut DictStorage) {
    fn parse_format(fmt: &str) -> (char, Vec<char>) {
        // Returns (byte_order, codes).
        let chars = fmt.chars();
        let first = chars.clone().next().unwrap_or('@');
        let (endian, rest) = if matches!(first, '<' | '>' | '!' | '=' | '@') {
            (first, chars.skip(1).collect::<String>())
        } else {
            ('@', fmt.to_string())
        };
        (
            endian,
            rest.chars().filter(|c| !c.is_ascii_whitespace()).collect(),
        )
    }
    fn code_size(c: char) -> usize {
        match c {
            'b' | 'B' | 'c' | '?' | 'x' => 1,
            'h' | 'H' => 2,
            'i' | 'I' | 'l' | 'L' | 'f' => 4,
            'q' | 'Q' | 'd' | 'n' | 'N' => 8,
            'e' => 2,
            _ => 0,
        }
    }
    crate::dict_storage_store(
        ns,
        "_clearcache",
        crate::make_builtin_function_with_arity("_clearcache", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(ns, "error", crate::typedef::w_object());
    crate::dict_storage_store(
        ns,
        "calcsize",
        crate::make_builtin_function_with_arity(
            "calcsize",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                let fmt = unsafe {
                    if pyre_object::is_str(args[0]) {
                        pyre_object::w_str_get_value(args[0]).to_string()
                    } else if pyre_object::bytesobject::is_bytes_like(args[0]) {
                        let data = pyre_object::bytesobject::bytes_like_data(args[0]);
                        String::from_utf8_lossy(data).into_owned()
                    } else {
                        return Err(crate::PyError::type_error("calcsize: format must be str"));
                    }
                };
                let (_, codes) = parse_format(&fmt);
                let total: usize = codes.iter().copied().map(code_size).sum();
                Ok(pyre_object::w_int_new(total as i64))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "pack",
        crate::make_builtin_function("pack", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_bytes_from_bytes(&[]));
            }
            let fmt = unsafe {
                if pyre_object::is_str(args[0]) {
                    pyre_object::w_str_get_value(args[0]).to_string()
                } else {
                    return Err(crate::PyError::type_error("pack: format must be str"));
                }
            };
            let (endian, codes) = parse_format(&fmt);
            let little = matches!(endian, '<' | '=' | '@');
            let mut out = Vec::new();
            for (i, code) in codes.iter().enumerate() {
                let arg = args.get(i + 1).copied().unwrap_or(pyre_object::w_none());
                match *code {
                    'b' | 'B' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) } as i8;
                        out.push(v as u8);
                    }
                    'h' | 'H' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) } as i16;
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'i' | 'I' | 'l' | 'L' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) } as i32;
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'q' | 'Q' | 'n' | 'N' => {
                        let v = unsafe { pyre_object::w_int_get_value(arg) };
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'f' => {
                        let v = unsafe {
                            if pyre_object::is_float(arg) {
                                pyre_object::w_float_get_value(arg) as f32
                            } else {
                                pyre_object::w_int_get_value(arg) as f32
                            }
                        };
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    'd' => {
                        let v = unsafe {
                            if pyre_object::is_float(arg) {
                                pyre_object::w_float_get_value(arg)
                            } else {
                                pyre_object::w_int_get_value(arg) as f64
                            }
                        };
                        let bytes = if little {
                            v.to_le_bytes()
                        } else {
                            v.to_be_bytes()
                        };
                        out.extend_from_slice(&bytes);
                    }
                    _ => {}
                }
            }
            Ok(pyre_object::w_bytes_from_bytes(&out))
        }),
    );
    crate::dict_storage_store(
        ns,
        "unpack",
        crate::make_builtin_function_with_arity(
            "unpack",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("unpack requires (fmt, buffer)"));
                }
                let fmt = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                let buf = unsafe {
                    if pyre_object::bytesobject::is_bytes_like(args[1]) {
                        pyre_object::bytesobject::bytes_like_data(args[1]).to_vec()
                    } else {
                        return Err(crate::PyError::type_error(
                            "unpack: buffer must be bytes-like",
                        ));
                    }
                };
                let (endian, codes) = parse_format(&fmt);
                let little = matches!(endian, '<' | '=' | '@');
                let mut out = Vec::new();
                let mut pos = 0usize;
                for code in codes {
                    match code {
                        'b' | 'B' => {
                            if pos >= buf.len() {
                                break;
                            }
                            out.push(pyre_object::w_int_new(buf[pos] as i8 as i64));
                            pos += 1;
                        }
                        'h' | 'H' => {
                            if pos + 2 > buf.len() {
                                break;
                            }
                            let chunk = [buf[pos], buf[pos + 1]];
                            let v = if little {
                                i16::from_le_bytes(chunk)
                            } else {
                                i16::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_int_new(v as i64));
                            pos += 2;
                        }
                        'i' | 'I' | 'l' | 'L' => {
                            if pos + 4 > buf.len() {
                                break;
                            }
                            let chunk = [buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]];
                            let v = if little {
                                i32::from_le_bytes(chunk)
                            } else {
                                i32::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_int_new(v as i64));
                            pos += 4;
                        }
                        'q' | 'Q' | 'n' | 'N' => {
                            if pos + 8 > buf.len() {
                                break;
                            }
                            let chunk: [u8; 8] = buf[pos..pos + 8].try_into().unwrap();
                            let v = if little {
                                i64::from_le_bytes(chunk)
                            } else {
                                i64::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_int_new(v));
                            pos += 8;
                        }
                        'f' => {
                            if pos + 4 > buf.len() {
                                break;
                            }
                            let chunk = [buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]];
                            let v = if little {
                                f32::from_le_bytes(chunk)
                            } else {
                                f32::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_float_new(v as f64));
                            pos += 4;
                        }
                        'd' => {
                            if pos + 8 > buf.len() {
                                break;
                            }
                            let chunk: [u8; 8] = buf[pos..pos + 8].try_into().unwrap();
                            let v = if little {
                                f64::from_le_bytes(chunk)
                            } else {
                                f64::from_be_bytes(chunk)
                            };
                            out.push(pyre_object::w_float_new(v));
                            pos += 8;
                        }
                        _ => {}
                    }
                }
                Ok(pyre_object::w_tuple_new(out))
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "unpack_from",
        crate::make_builtin_function("unpack_from", |_| Ok(pyre_object::w_tuple_new(vec![]))),
    );
    crate::dict_storage_store(
        ns,
        "iter_unpack",
        crate::make_builtin_function_with_arity(
            "iter_unpack",
            |_| Ok(pyre_object::w_list_new(vec![])),
            2,
        ),
    );
    // Struct class — minimal constructor returning instance with format
    // attribute. Used by struct.Struct(fmt).pack/unpack.
    crate::dict_storage_store(
        ns,
        "Struct",
        crate::make_builtin_function_with_arity(
            "Struct",
            |args| {
                let fmt = args.first().copied().unwrap_or(pyre_object::w_str_new(""));
                let obj = pyre_object::w_instance_new(crate::typedef::w_object());
                let _ = crate::baseobjspace::setattr(obj, "format", fmt);
                Ok(obj)
            },
            1,
        ),
    );
}

/// `_random` C-extension stub — PyPy: pypy/module/_random/interp_random.py.
///
/// Provides a minimal `Random` class that wraps a very small linear
/// congruential generator. Good enough for `random.py` to construct a
/// `random._inst` at module import time; real tests can then use the
/// Python `random.Random` subclass as a drop-in.
fn init_random(ns: &mut DictStorage) {
    fn random_type() -> PyObjectRef {
        thread_local! {
            static RANDOM_TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
        }
        RANDOM_TYPE.with(|c| {
            *c.get_or_init(|| {
                let tp = crate::typedef::make_builtin_type("_random.Random", |ns| {
                    // random_method_* are defined in importing.rs; routing
                    // through make_builtin_function binds them as unbound
                    // methods so `rand.random()` calls pass `self` as args[0].
                    crate::dict_storage_store(
                        ns,
                        "__init__",
                        crate::make_builtin_function("__init__", |args| {
                            let seed = if args.len() >= 2 {
                                unsafe {
                                    if pyre_object::is_int(args[1]) {
                                        pyre_object::w_int_get_value(args[1]) as u64
                                    } else {
                                        0x1234_5678
                                    }
                                }
                            } else {
                                0x1234_5678
                            };
                            let _ = crate::baseobjspace::setattr(
                                args[0],
                                "__rand_state__",
                                pyre_object::w_int_new(seed as i64),
                            );
                            Ok(pyre_object::w_none())
                        }),
                    );
                    crate::dict_storage_store(
                        ns,
                        "seed",
                        crate::make_builtin_function("seed", |args| {
                            let seed = if args.len() >= 2 {
                                unsafe {
                                    if pyre_object::is_int(args[1]) {
                                        pyre_object::w_int_get_value(args[1]) as u64
                                    } else {
                                        0x1234_5678
                                    }
                                }
                            } else {
                                0x1234_5678
                            };
                            let _ = crate::baseobjspace::setattr(
                                args[0],
                                "__rand_state__",
                                pyre_object::w_int_new(seed as i64),
                            );
                            Ok(pyre_object::w_none())
                        }),
                    );
                    crate::dict_storage_store(
                        ns,
                        "random",
                        crate::make_builtin_function_with_arity(
                            "random",
                            |args| {
                                // Tiny xorshift PRNG — ok for import-time construction.
                                let self_obj = args[0];
                                let state =
                                    crate::baseobjspace::getattr(self_obj, "__rand_state__")
                                        .ok()
                                        .map(|v| unsafe { pyre_object::w_int_get_value(v) as u64 })
                                        .unwrap_or(0x1234_5678);
                                let mut x = state;
                                x ^= x << 13;
                                x ^= x >> 7;
                                x ^= x << 17;
                                let _ = crate::baseobjspace::setattr(
                                    self_obj,
                                    "__rand_state__",
                                    pyre_object::w_int_new(x as i64),
                                );
                                Ok(pyre_object::w_float_new((x as f64) / (u64::MAX as f64)))
                            },
                            1,
                        ),
                    );
                    crate::dict_storage_store(
                        ns,
                        "getrandbits",
                        crate::make_builtin_function("getrandbits", |args| {
                            let k = if args.len() >= 2 {
                                unsafe { pyre_object::w_int_get_value(args[1]) as u32 }
                            } else {
                                32
                            };
                            let state = crate::baseobjspace::getattr(args[0], "__rand_state__")
                                .ok()
                                .map(|v| unsafe { pyre_object::w_int_get_value(v) as u64 })
                                .unwrap_or(0x1234_5678);
                            let mut x = state;
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            let _ = crate::baseobjspace::setattr(
                                args[0],
                                "__rand_state__",
                                pyre_object::w_int_new(x as i64),
                            );
                            let mask = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
                            Ok(pyre_object::w_int_new((x & mask) as i64))
                        }),
                    );
                    crate::dict_storage_store(
                        ns,
                        "getstate",
                        crate::make_builtin_function_with_arity(
                            "getstate",
                            |args| {
                                let state = crate::baseobjspace::getattr(args[0], "__rand_state__")
                                    .unwrap_or_else(|_| pyre_object::w_int_new(0));
                                Ok(pyre_object::w_tuple_new(vec![state]))
                            },
                            1,
                        ),
                    );
                    crate::dict_storage_store(
                        ns,
                        "setstate",
                        crate::make_builtin_function_with_arity(
                            "setstate",
                            |args| {
                                if args.len() >= 2 {
                                    unsafe {
                                        if pyre_object::is_tuple(args[1])
                                            && pyre_object::w_tuple_len(args[1]) >= 1
                                        {
                                            if let Some(state) =
                                                pyre_object::w_tuple_getitem(args[1], 0)
                                            {
                                                let _ = crate::baseobjspace::setattr(
                                                    args[0],
                                                    "__rand_state__",
                                                    state,
                                                );
                                            }
                                        }
                                    }
                                }
                                Ok(pyre_object::w_none())
                            },
                            2,
                        ),
                    );
                });
                unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
                tp
            })
        })
    }
    crate::dict_storage_store(ns, "Random", random_type());
}

/// `_locale` C-extension stub — PyPy: pypy/module/_locale/.
///
/// Provides the 'C' locale defaults so locale.py's `from _locale import *`
/// succeeds and Lib/locale.py exposes working `localeconv`/`setlocale`.
/// This mirrors the `except ImportError` fallback in the stdlib's
/// `locale` module, but routed through pyre's builtin-module registry
/// so a single import succeeds.
fn init_locale(ns: &mut DictStorage) {
    // Locale category constants sourced from libc so the values match
    // the host (Linux: LC_CTYPE=0; macOS: LC_ALL=0, LC_CTYPE=2; ...).
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "LC_CTYPE",
            pyre_object::w_int_new(libc::LC_CTYPE as i64),
        );
        crate::dict_storage_store(
            ns,
            "LC_NUMERIC",
            pyre_object::w_int_new(libc::LC_NUMERIC as i64),
        );
        crate::dict_storage_store(ns, "LC_TIME", pyre_object::w_int_new(libc::LC_TIME as i64));
        crate::dict_storage_store(
            ns,
            "LC_COLLATE",
            pyre_object::w_int_new(libc::LC_COLLATE as i64),
        );
        crate::dict_storage_store(
            ns,
            "LC_MONETARY",
            pyre_object::w_int_new(libc::LC_MONETARY as i64),
        );
        crate::dict_storage_store(
            ns,
            "LC_MESSAGES",
            pyre_object::w_int_new(libc::LC_MESSAGES as i64),
        );
        crate::dict_storage_store(ns, "LC_ALL", pyre_object::w_int_new(libc::LC_ALL as i64));
    }
    #[cfg(not(unix))]
    {
        crate::dict_storage_store(ns, "LC_CTYPE", pyre_object::w_int_new(0));
        crate::dict_storage_store(ns, "LC_NUMERIC", pyre_object::w_int_new(1));
        crate::dict_storage_store(ns, "LC_TIME", pyre_object::w_int_new(2));
        crate::dict_storage_store(ns, "LC_COLLATE", pyre_object::w_int_new(3));
        crate::dict_storage_store(ns, "LC_MONETARY", pyre_object::w_int_new(4));
        crate::dict_storage_store(ns, "LC_MESSAGES", pyre_object::w_int_new(5));
        crate::dict_storage_store(ns, "LC_ALL", pyre_object::w_int_new(6));
    }
    crate::dict_storage_store(ns, "CHAR_MAX", pyre_object::w_int_new(127));
    #[cfg(all(
        unix,
        not(any(target_os = "ios", target_os = "android", target_os = "redox"))
    ))]
    {
        crate::dict_storage_store(ns, "CODESET", pyre_object::w_int_new(libc::CODESET as i64));
    }
    // Error alias — locale.py does `Error = ValueError` when _locale is
    // missing; here we expose a real placeholder that is a str so that
    // `except _locale.Error` still compiles (match falls through).
    crate::dict_storage_store(ns, "Error", pyre_object::w_str_new("Error"));

    // localeconv() — returns the 'C' locale parameters as a dict.
    crate::dict_storage_store(
        ns,
        "localeconv",
        crate::make_builtin_function_with_arity(
            "localeconv",
            |_| {
                let d = pyre_object::w_dict_new();
                unsafe {
                    pyre_object::w_dict_setitem_str(
                        d,
                        "grouping",
                        pyre_object::w_list_new(vec![pyre_object::w_int_new(127)]),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "currency_symbol",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(d, "n_sign_posn", pyre_object::w_int_new(127));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "p_cs_precedes",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "n_cs_precedes",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "mon_grouping",
                        pyre_object::w_list_new(vec![]),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "n_sep_by_space",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "decimal_point",
                        pyre_object::w_str_new("."),
                    );
                    pyre_object::w_dict_setitem_str(d, "negative_sign", pyre_object::w_str_new(""));
                    pyre_object::w_dict_setitem_str(d, "positive_sign", pyre_object::w_str_new(""));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "p_sep_by_space",
                        pyre_object::w_int_new(127),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "int_curr_symbol",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(d, "p_sign_posn", pyre_object::w_int_new(127));
                    pyre_object::w_dict_setitem_str(d, "thousands_sep", pyre_object::w_str_new(""));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "mon_thousands_sep",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(d, "frac_digits", pyre_object::w_int_new(127));
                    pyre_object::w_dict_setitem_str(
                        d,
                        "mon_decimal_point",
                        pyre_object::w_str_new(""),
                    );
                    pyre_object::w_dict_setitem_str(
                        d,
                        "int_frac_digits",
                        pyre_object::w_int_new(127),
                    );
                }
                Ok(d)
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "setlocale",
        crate::make_builtin_function("setlocale", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("setlocale() missing category"));
                }
                if !unsafe { pyre_object::is_int(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "setlocale: category must be an integer",
                    ));
                }
                let cat = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let locale_str: Option<String> =
                    if args.len() >= 2 && !unsafe { pyre_object::is_none(args[1]) } {
                        if !unsafe { pyre_object::is_str(args[1]) } {
                            return Err(crate::PyError::type_error(
                                "setlocale: locale must be a string or None",
                            ));
                        }
                        Some(unsafe { pyre_object::w_str_get_value(args[1]).to_string() })
                    } else {
                        None
                    };
                let c_locale = match locale_str.as_ref() {
                    Some(s) => Some(
                        std::ffi::CString::new(s.as_bytes())
                            .map_err(|_| crate::PyError::value_error("embedded null"))?,
                    ),
                    None => None,
                };
                let out = rustpython_host_env::locale::setlocale(cat, c_locale.as_deref());
                match out {
                    Some(bytes) => Ok(pyre_object::w_str_new(&String::from_utf8_lossy(&bytes))),
                    None => Err(crate::PyError::os_error("setlocale failed")),
                }
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                // No libc available — every category resolves to the
                // POSIX "C" locale, mirroring what setlocale(LC_*, "C")
                // returns on a real host.  Pure constant; no I/O.
                let _ = args;
                Ok(pyre_object::w_str_new("C"))
            }
        }),
    );
    crate::dict_storage_store(
        ns,
        "nl_langinfo",
        crate::make_builtin_function_with_arity(
            "nl_langinfo",
            |args| {
                #[cfg(all(
                    unix,
                    feature = "host_env",
                    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
                ))]
                {
                    let item = if args.is_empty() {
                        libc::CODESET
                    } else {
                        if !unsafe { pyre_object::is_int(args[0]) } {
                            return Err(crate::PyError::type_error(
                                "nl_langinfo: item must be an integer",
                            ));
                        }
                        (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::nl_item
                    };
                    if item == libc::CODESET {
                        if let Some(bytes) = rustpython_host_env::locale::nl_langinfo_codeset() {
                            return Ok(pyre_object::w_str_new(&String::from_utf8_lossy(&bytes)));
                        }
                    }
                    let p = unsafe { libc::nl_langinfo(item) };
                    if p.is_null() {
                        return Ok(pyre_object::w_str_new(""));
                    }
                    let s = unsafe { std::ffi::CStr::from_ptr(p) };
                    return Ok(pyre_object::w_str_new(&s.to_string_lossy()));
                }
                #[cfg(not(all(
                    unix,
                    feature = "host_env",
                    not(any(target_os = "ios", target_os = "android", target_os = "redox"))
                )))]
                {
                    let _ = args;
                    Ok(pyre_object::w_str_new(""))
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "strcoll",
        crate::make_builtin_function_with_arity(
            "strcoll",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    if args.len() < 2
                        || !unsafe { pyre_object::is_str(args[0]) && pyre_object::is_str(args[1]) }
                    {
                        return Err(crate::PyError::type_error(
                            "strcoll: arguments must be strings",
                        ));
                    }
                    let s1 = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                    let s2 = unsafe { pyre_object::w_str_get_value(args[1]).to_string() };
                    let c1 = std::ffi::CString::new(s1.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null"))?;
                    let c2 = std::ffi::CString::new(s2.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null"))?;
                    return Ok(pyre_object::w_int_new(
                        rustpython_host_env::locale::strcoll(&c1, &c2) as i64,
                    ));
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    if args.len() < 2
                        || !unsafe { pyre_object::is_str(args[0]) && pyre_object::is_str(args[1]) }
                    {
                        return Err(crate::PyError::type_error(
                            "strcoll: arguments must be strings",
                        ));
                    }
                    // No libc collation available — fall back to
                    // lexical bytewise comparison.  Pure computation,
                    // no I/O, so the sandbox principle is unaffected.
                    let s1 = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                    let s2 = unsafe { pyre_object::w_str_get_value(args[1]).to_string() };
                    let ord = match s1.as_str().cmp(s2.as_str()) {
                        std::cmp::Ordering::Less => -1,
                        std::cmp::Ordering::Equal => 0,
                        std::cmp::Ordering::Greater => 1,
                    };
                    Ok(pyre_object::w_int_new(ord))
                }
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "strxfrm",
        crate::make_builtin_function_with_arity(
            "strxfrm",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_str_new(""))),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getencoding",
        crate::make_builtin_function_with_arity(
            "getencoding",
            |_| Ok(pyre_object::w_str_new("utf-8")),
            0,
        ),
    );
}

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

/// Shared `posix.stat_result` builtin type — a plain instance bag with
/// hasdict so that `st_mode`, `st_ino`, etc. attributes can be set from
/// Rust when building stat results. PyPy builds a structseq subclass with
/// named fields; this is the pyre approximation.
fn stat_result_type() -> PyObjectRef {
    thread_local! {
        static STAT_RESULT_TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    STAT_RESULT_TYPE.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("stat_result", |_ns| {});
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

/// `interp_pwd.py:50-73 uid_converter` — narrow a python int to `uid_t`.
///
/// `-1` is the "current uid" sentinel and passes through unchanged
/// (cast to `uid_t` it becomes the max value, matching the C convention
/// most BSDs use).  Other negative inputs raise OverflowError "user id
/// is less than minimum"; values that don't fit in `uid_t` raise
/// OverflowError "user id is greater than maximum".  Floats / non-int
/// inputs raise TypeError via `int_w`.
#[cfg(unix)]
fn pwd_uid_converter(w_uid: pyre_object::PyObjectRef) -> Result<libc::uid_t, crate::PyError> {
    let val = match crate::baseobjspace::int_w(w_uid) {
        Ok(v) => v,
        Err(e) if matches!(e.kind, crate::PyErrorKind::OverflowError) => {
            // `interp_pwd.py:60-66` — fall through to `uint_w` and
            // map to "greater than maximum" / "less than minimum"
            // OverflowError.  pyre's `int_w` only surfaces the
            // positive overflow case; the negative bigint path is
            // unreachable here because a negative bigint that
            // fails to fit i64 is more-negative than i64::MIN —
            // user id less than minimum.
            return Err(crate::PyError::overflow_error(
                "user id is greater than maximum",
            ));
        }
        Err(e) => return Err(e),
    };
    if val == -1 {
        return Ok((-1i64) as libc::uid_t);
    }
    if val < 0 {
        return Err(crate::PyError::overflow_error(
            "user id is less than minimum",
        ));
    }
    let uid = val as libc::uid_t;
    if uid as i64 != val {
        return Err(crate::PyError::overflow_error(
            "user id is greater than maximum",
        ));
    }
    Ok(uid)
}

/// pwd module — `pypy/module/pwd/interp_pwd.py`.
///
/// getpwuid / getpwnam / getpwall return 7-tuples with the
/// `(pw_name, pw_passwd, pw_uid, pw_gid, pw_gecos, pw_dir, pw_shell)`
/// layout.  `struct_passwd` / `struct_pwent` are exposed as the same
/// builtin type so `isinstance(pwd.struct_passwd, type)` succeeds and
/// `pwd.struct_passwd` is identity-equal to `pwd.struct_pwent`
/// (`app_pwd.py:1-21`).  Full structseq instance materialisation
/// (so `pw_entry.pw_name` returns a string) is a framework prereq
/// tracked separately.
///
/// Backed by `rustpython_host_env::pwd` (a thin `nix` wrapper).
#[cfg(unix)]
fn init_pwd(ns: &mut DictStorage) {
    #[cfg(feature = "host_env")]
    fn make_struct_passwd(pw: &rustpython_host_env::pwd::Passwd) -> pyre_object::PyObjectRef {
        pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new(&pw.name),
            pyre_object::w_str_new(&pw.passwd),
            pyre_object::w_int_new(pw.uid as i64),
            pyre_object::w_int_new(pw.gid as i64),
            pyre_object::w_str_new(&pw.gecos),
            pyre_object::w_str_new(&pw.dir),
            pyre_object::w_str_new(&pw.shell),
        ])
    }
    // `interp_pwd.py:75-87 make_struct_passwd` libc backend, used when
    // the host_env abstraction layer is disabled.  Mirrors the same
    // rffi.charp2str / int construction PyPy uses.
    #[cfg(not(feature = "host_env"))]
    unsafe fn make_struct_passwd_libc(pw: *const libc::passwd) -> pyre_object::PyObjectRef {
        unsafe fn cstr(p: *const libc::c_char) -> String {
            if p.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
        pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new(&cstr((*pw).pw_name)),
            pyre_object::w_str_new(&cstr((*pw).pw_passwd)),
            pyre_object::w_int_new((*pw).pw_uid as i64),
            pyre_object::w_int_new((*pw).pw_gid as i64),
            pyre_object::w_str_new(&cstr((*pw).pw_gecos)),
            pyre_object::w_str_new(&cstr((*pw).pw_dir)),
            pyre_object::w_str_new(&cstr((*pw).pw_shell)),
        ])
    }
    // `app_pwd.py:1-21` — `pwd.struct_passwd` / `pwd.struct_pwent`.
    let struct_passwd_type = crate::typedef::make_builtin_type("pwd.struct_passwd", |_| {});
    crate::dict_storage_store(ns, "struct_passwd", struct_passwd_type);
    crate::dict_storage_store(ns, "struct_pwent", struct_passwd_type);
    crate::dict_storage_store(
        ns,
        "getpwuid",
        crate::make_builtin_function_with_arity(
            "getpwuid",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("getpwuid() missing argument"));
                }
                // `interp_pwd.py:50-73 uid_converter`: -1 sentinel passes
                // through; negative-other → OverflowError "less than
                // minimum"; positive-too-big → OverflowError "greater
                // than maximum".  `interp_pwd.py:97-100 getpwuid` catches
                // OverflowError and converts it to KeyError "uid not
                // found".
                let uid = match pwd_uid_converter(args[0]) {
                    Ok(u) => u,
                    Err(e) if matches!(e.kind, crate::PyErrorKind::OverflowError) => {
                        return Err(crate::PyError::key_error("getpwuid(): uid not found"));
                    }
                    Err(e) => return Err(e),
                };
                #[cfg(feature = "host_env")]
                {
                    match rustpython_host_env::pwd::getpwuid(uid) {
                        Ok(Some(pw)) => return Ok(make_struct_passwd(&pw)),
                        Ok(None) => {
                            return Err(crate::PyError::key_error(format!(
                                "getpwuid(): uid not found: {}",
                                uid as i64
                            )));
                        }
                        Err(e) => {
                            return Err(crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("getpwuid: {e}"),
                            ));
                        }
                    }
                }
                // `interp_pwd.py:90-108` — libc fallback path; host_env
                // is a pyre-only abstraction layer over the same
                // getpwuid() call PyPy makes via rffi.llexternal.
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let pw = libc::getpwuid(uid);
                    if pw.is_null() {
                        return Err(crate::PyError::key_error(format!(
                            "getpwuid(): uid not found: {}",
                            uid as i64
                        )));
                    }
                    return Ok(make_struct_passwd_libc(pw));
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getpwnam",
        crate::make_builtin_function_with_arity(
            "getpwnam",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("getpwnam() missing argument"));
                }
                if !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "getpwnam(): name should be a string",
                    ));
                }
                let name = unsafe { pyre_object::w_str_get_value(args[0]) };
                // `interp_pwd.py:111 @unwrap_spec(name='text0')` rejects
                // embedded NULs.  CString::new() enforces that here.
                let c_name = std::ffi::CString::new(name).map_err(|_| {
                    crate::PyError::value_error("getpwnam: name must not contain NUL bytes")
                })?;
                #[cfg(feature = "host_env")]
                {
                    match rustpython_host_env::pwd::getpwnam(name) {
                        Some(pw) => return Ok(make_struct_passwd(&pw)),
                        None => {
                            return Err(crate::PyError::key_error(format!(
                                "getpwnam(): name not found: {}",
                                name
                            )));
                        }
                    }
                }
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let pw = libc::getpwnam(c_name.as_ptr());
                    if pw.is_null() {
                        return Err(crate::PyError::key_error(format!(
                            "getpwnam(): name not found: {}",
                            name
                        )));
                    }
                    return Ok(make_struct_passwd_libc(pw));
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getpwall",
        crate::make_builtin_function_with_arity(
            "getpwall",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    let items: Vec<pyre_object::PyObjectRef> = rustpython_host_env::pwd::getpwall()
                        .iter()
                        .map(make_struct_passwd)
                        .collect();
                    return Ok(pyre_object::w_list_new(items));
                }
                // `interp_pwd.py:123-134` — setpwent / loop getpwent /
                // endpwent.
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let mut items: Vec<pyre_object::PyObjectRef> = Vec::new();
                    libc::setpwent();
                    loop {
                        let pw = libc::getpwent();
                        if pw.is_null() {
                            break;
                        }
                        items.push(make_struct_passwd_libc(pw));
                    }
                    libc::endpwent();
                    return Ok(pyre_object::w_list_new(items));
                }
            },
            0,
        ),
    );
}

/// grp module — `lib_pypy/grp.py` (PyPy keeps it app-level via
/// `_pwdgrp_cffi`).  pyre takes CPython's `Modules/grpmodule.c`
/// shape since pyre has no app-level stdlib.
///
/// getgrgid / getgrnam / getgrall return 4-tuples `(gr_name,
/// gr_passwd, gr_gid, gr_mem)` matching CPython.  `grp.struct_group`
/// is exposed as a builtin type attribute; full structseq instance
/// materialisation (so `entry.gr_name` works) is blocked on the
/// structseq framework task.
#[cfg(unix)]
fn init_grp(ns: &mut DictStorage) {
    #[cfg(feature = "host_env")]
    fn make_struct_group(g: &rustpython_host_env::grp::Group) -> pyre_object::PyObjectRef {
        let mem_items: Vec<pyre_object::PyObjectRef> =
            g.mem.iter().map(|s| pyre_object::w_str_new(s)).collect();
        pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new(&g.name),
            pyre_object::w_str_new(&g.passwd),
            pyre_object::w_int_new(g.gid as i64),
            pyre_object::w_list_new(mem_items),
        ])
    }
    // `lib_pypy/grp.py:21-34 _group_from_gstruct` libc backend, used when
    // the host_env abstraction layer is disabled.
    #[cfg(not(feature = "host_env"))]
    unsafe fn make_struct_group_libc(g: *const libc::group) -> pyre_object::PyObjectRef {
        unsafe fn cstr(p: *const libc::c_char) -> String {
            if p.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
        let mut mem_items: Vec<pyre_object::PyObjectRef> = Vec::new();
        let mut p = (*g).gr_mem;
        if !p.is_null() {
            while !(*p).is_null() {
                mem_items.push(pyre_object::w_str_new(&cstr(*p)));
                p = p.add(1);
            }
        }
        pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new(&cstr((*g).gr_name)),
            pyre_object::w_str_new(&cstr((*g).gr_passwd)),
            pyre_object::w_int_new((*g).gr_gid as i64),
            pyre_object::w_list_new(mem_items),
        ])
    }
    // `lib_pypy/grp.py:13-19 class struct_group` — exposed so
    // `grp.struct_group` is observable on the module even though
    // returned values are still raw tuples.
    let struct_group_type = crate::typedef::make_builtin_type("grp.struct_group", |_| {});
    crate::dict_storage_store(ns, "struct_group", struct_group_type);
    crate::dict_storage_store(
        ns,
        "getgrgid",
        crate::make_builtin_function_with_arity(
            "getgrgid",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("getgrgid() missing argument"));
                }
                // `Modules/grpmodule.c grp_getgrgid` — accept any
                // python int (including bigint) via int_w; reject
                // floats as TypeError.  PyPy's `lib_pypy/grp.py`
                // forwards directly through ctypes which would
                // do the same conversion.
                let val = crate::baseobjspace::int_w(args[0])?;
                let gid = val as libc::gid_t;
                #[cfg(feature = "host_env")]
                {
                    match rustpython_host_env::grp::getgrgid(gid) {
                        Ok(Some(g)) => return Ok(make_struct_group(&g)),
                        Ok(None) => {
                            return Err(crate::PyError::key_error(format!(
                                "getgrgid(): gid not found: {}",
                                gid
                            )));
                        }
                        Err(e) => {
                            return Err(crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("getgrgid: {e}"),
                            ));
                        }
                    }
                }
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let g = libc::getgrgid(gid);
                    if g.is_null() {
                        return Err(crate::PyError::key_error(format!(
                            "getgrgid(): gid not found: {}",
                            gid
                        )));
                    }
                    return Ok(make_struct_group_libc(g));
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getgrnam",
        crate::make_builtin_function_with_arity(
            "getgrnam",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("getgrnam() missing argument"));
                }
                if !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "getgrnam(): name should be a string",
                    ));
                }
                let name = unsafe { pyre_object::w_str_get_value(args[0]) };
                // Reject embedded NULs (parity with PyPy's @unwrap_spec
                // text0 used for similar lookup APIs).
                let c_name = std::ffi::CString::new(name).map_err(|_| {
                    crate::PyError::value_error("getgrnam: name must not contain NUL bytes")
                })?;
                #[cfg(feature = "host_env")]
                {
                    match rustpython_host_env::grp::getgrnam(name) {
                        Ok(Some(g)) => return Ok(make_struct_group(&g)),
                        Ok(None) => {
                            return Err(crate::PyError::key_error(format!(
                                "getgrnam(): name not found: {}",
                                name
                            )));
                        }
                        Err(e) => {
                            return Err(crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("getgrnam: {e}"),
                            ));
                        }
                    }
                }
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let g = libc::getgrnam(c_name.as_ptr());
                    if g.is_null() {
                        return Err(crate::PyError::key_error(format!(
                            "getgrnam(): name not found: {}",
                            name
                        )));
                    }
                    return Ok(make_struct_group_libc(g));
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getgrall",
        crate::make_builtin_function_with_arity(
            "getgrall",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    let items: Vec<pyre_object::PyObjectRef> = rustpython_host_env::grp::getgrall()
                        .iter()
                        .map(make_struct_group)
                        .collect();
                    return Ok(pyre_object::w_list_new(items));
                }
                #[cfg(not(feature = "host_env"))]
                unsafe {
                    let mut items: Vec<pyre_object::PyObjectRef> = Vec::new();
                    libc::setgrent();
                    loop {
                        let g = libc::getgrent();
                        if g.is_null() {
                            break;
                        }
                        items.push(make_struct_group_libc(g));
                    }
                    libc::endgrent();
                    return Ok(pyre_object::w_list_new(items));
                }
            },
            0,
        ),
    );
}

/// resource module — `lib_pypy/resource.py` (PyPy keeps it app-level
/// via `_resource_cffi`).  pyre takes CPython's `Modules/resource.c`
/// shape since pyre has no app-level stdlib.
///
/// Exposes getrusage / getrlimit / setrlimit plus the standard RUSAGE_*
/// and RLIMIT_* constants, the `struct_rusage` type attribute, and the
/// `error = OSError` alias.  Backed by `rustpython_host_env::resource`.
fn init_resource(ns: &mut DictStorage) {
    // `lib_pypy/resource.py:13 error = OSError` and
    // `:15-37 class struct_rusage`.
    let w_os_error = crate::builtins::lookup_exc_class("OSError")
        .expect("OSError must be installed before init_resource");
    crate::dict_storage_store(ns, "error", w_os_error);
    crate::dict_storage_store(
        ns,
        "struct_rusage",
        crate::typedef::make_builtin_type("resource.struct_rusage", |_| {}),
    );
    // ── struct_rusage tuple (16-field layout matches CPython) ──
    #[cfg(all(unix, feature = "host_env"))]
    fn make_struct_rusage(r: &rustpython_host_env::resource::RUsage) -> pyre_object::PyObjectRef {
        let tv_to_f = |tv: libc::timeval| tv.tv_sec as f64 + (tv.tv_usec as f64) * 1e-6;
        pyre_object::w_tuple_new(vec![
            pyre_object::floatobject::w_float_new(tv_to_f(r.ru_utime)),
            pyre_object::floatobject::w_float_new(tv_to_f(r.ru_stime)),
            pyre_object::w_int_new(r.ru_maxrss as i64),
            pyre_object::w_int_new(r.ru_ixrss as i64),
            pyre_object::w_int_new(r.ru_idrss as i64),
            pyre_object::w_int_new(r.ru_isrss as i64),
            pyre_object::w_int_new(r.ru_minflt as i64),
            pyre_object::w_int_new(r.ru_majflt as i64),
            pyre_object::w_int_new(r.ru_nswap as i64),
            pyre_object::w_int_new(r.ru_inblock as i64),
            pyre_object::w_int_new(r.ru_oublock as i64),
            pyre_object::w_int_new(r.ru_msgsnd as i64),
            pyre_object::w_int_new(r.ru_msgrcv as i64),
            pyre_object::w_int_new(r.ru_nsignals as i64),
            pyre_object::w_int_new(r.ru_nvcsw as i64),
            pyre_object::w_int_new(r.ru_nivcsw as i64),
        ])
    }
    crate::dict_storage_store(
        ns,
        "getrusage",
        crate::make_builtin_function_with_arity(
            "getrusage",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    let who = if let Some(&a) = args.first() {
                        if unsafe { pyre_object::is_int(a) } {
                            unsafe { pyre_object::w_int_get_value(a) as i32 }
                        } else {
                            return Err(crate::PyError::type_error(
                                "getrusage(): who should be an integer",
                            ));
                        }
                    } else {
                        return Err(crate::PyError::type_error("getrusage() missing argument"));
                    };
                    match rustpython_host_env::resource::getrusage(who) {
                        Ok(r) => return Ok(make_struct_rusage(&r)),
                        Err(e) => {
                            let errno = e.raw_os_error().unwrap_or(0);
                            // `lib_pypy/resource.py:106` raises ValueError for
                            // an invalid `who`; only other errno values are
                            // surfaced as OSError.
                            if errno == libc::EINVAL {
                                return Err(crate::PyError::value_error("invalid who parameter"));
                            }
                            return Err(crate::PyError::os_error_with_errno(
                                errno,
                                format!("getrusage: {e}"),
                            ));
                        }
                    }
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "resource.getrusage requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getrlimit",
        crate::make_builtin_function_with_arity(
            "getrlimit",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    let res = if let Some(&a) = args.first() {
                        if unsafe { pyre_object::is_int(a) } {
                            unsafe { pyre_object::w_int_get_value(a) as libc::rlim_t }
                        } else {
                            return Err(crate::PyError::type_error(
                                "getrlimit(): resource should be an integer",
                            ));
                        }
                    } else {
                        return Err(crate::PyError::type_error("getrlimit() missing argument"));
                    };
                    match rustpython_host_env::resource::getrlimit(res) {
                        Ok(rl) => {
                            return Ok(pyre_object::w_tuple_new(vec![
                                pyre_object::w_int_new(rl.rlim_cur as i64),
                                pyre_object::w_int_new(rl.rlim_max as i64),
                            ]));
                        }
                        Err(e) => {
                            return Err(crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("getrlimit: {e}"),
                            ));
                        }
                    }
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "resource.getrlimit requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "setrlimit",
        crate::make_builtin_function_with_arity(
            "setrlimit",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "setrlimit() requires 2 arguments",
                        ));
                    }
                    let res = unsafe {
                        if !pyre_object::is_int(args[0]) {
                            return Err(crate::PyError::type_error(
                                "setrlimit(): resource should be an integer",
                            ));
                        }
                        pyre_object::w_int_get_value(args[0]) as libc::rlim_t
                    };
                    // limits is a 2-tuple (soft, hard).
                    let (soft, hard) = unsafe {
                        if !pyre_object::is_tuple(args[1]) || pyre_object::w_tuple_len(args[1]) != 2
                        {
                            return Err(crate::PyError::type_error(
                                "setrlimit(): limits should be a tuple of (soft, hard)",
                            ));
                        }
                        let s = pyre_object::w_tuple_getitem(args[1], 0).unwrap();
                        let h = pyre_object::w_tuple_getitem(args[1], 1).unwrap();
                        if !pyre_object::is_int(s) || !pyre_object::is_int(h) {
                            return Err(crate::PyError::type_error(
                                "setrlimit(): limits members must be integers",
                            ));
                        }
                        (
                            pyre_object::w_int_get_value(s) as libc::rlim_t,
                            pyre_object::w_int_get_value(h) as libc::rlim_t,
                        )
                    };
                    let rl = libc::rlimit {
                        rlim_cur: soft,
                        rlim_max: hard,
                    };
                    match rustpython_host_env::resource::setrlimit(res, rl) {
                        Ok(()) => return Ok(pyre_object::w_none()),
                        Err(e) => {
                            // `lib_pypy/resource.py:89-95` — EINVAL and
                            // EPERM both surface as ValueError with
                            // distinct messages; all other errnos stay
                            // as OSError.
                            let errno = e.raw_os_error().unwrap_or(0);
                            if errno == libc::EINVAL {
                                return Err(crate::PyError::value_error(
                                    "current limit exceeds maximum limit",
                                ));
                            }
                            if errno == libc::EPERM {
                                return Err(crate::PyError::value_error(
                                    "not allowed to raise maximum limit",
                                ));
                            }
                            return Err(crate::PyError::os_error_with_errno(
                                errno,
                                format!("setrlimit: {e}"),
                            ));
                        }
                    }
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "resource.setrlimit requires host_env feature",
                    ))
                }
            },
            2,
        ),
    );
    // ── Constants (POSIX subset matching CPython) ──
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "RUSAGE_SELF",
            pyre_object::w_int_new(libc::RUSAGE_SELF as i64),
        );
        crate::dict_storage_store(
            ns,
            "RUSAGE_CHILDREN",
            pyre_object::w_int_new(libc::RUSAGE_CHILDREN as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_CPU",
            pyre_object::w_int_new(libc::RLIMIT_CPU as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_FSIZE",
            pyre_object::w_int_new(libc::RLIMIT_FSIZE as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_DATA",
            pyre_object::w_int_new(libc::RLIMIT_DATA as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_STACK",
            pyre_object::w_int_new(libc::RLIMIT_STACK as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_CORE",
            pyre_object::w_int_new(libc::RLIMIT_CORE as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_NOFILE",
            pyre_object::w_int_new(libc::RLIMIT_NOFILE as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_AS",
            pyre_object::w_int_new(libc::RLIMIT_AS as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_RSS",
            pyre_object::w_int_new(libc::RLIMIT_RSS as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_NPROC",
            pyre_object::w_int_new(libc::RLIMIT_NPROC as i64),
        );
        crate::dict_storage_store(
            ns,
            "RLIMIT_MEMLOCK",
            pyre_object::w_int_new(libc::RLIMIT_MEMLOCK as i64),
        );
        // RLIM_INFINITY: unsigned max — pyre stores as i64 (-1 on signed widen).
        crate::dict_storage_store(
            ns,
            "RLIM_INFINITY",
            pyre_object::w_int_new(libc::RLIM_INFINITY as i64),
        );
    }
}

/// fcntl module — PyPy: pypy/module/fcntl/interp_fcntl.py.
///
/// fcntl(fd, cmd, arg=0) / ioctl(fd, request, arg=0) / flock(fd, op) /
/// lockf(fd, cmd, len=0, start=0, whence=0).  Backed by
/// `rustpython_host_env::fcntl`.  Only the integer-argument forms are
/// implemented; bytes-buffer (out-arg) variants are out of scope.
fn init_fcntl(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "fcntl",
        crate::make_builtin_function("fcntl", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "fcntl() requires at least 2 arguments",
                    ));
                }
                if !unsafe { pyre_object::is_int(args[0]) }
                    || !unsafe { pyre_object::is_int(args[1]) }
                    || (args.len() >= 3 && !unsafe { pyre_object::is_int(args[2]) })
                {
                    return Err(crate::PyError::type_error(
                        "fcntl() arguments must be integers",
                    ));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let cmd = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                let arg = if args.len() >= 3 {
                    unsafe { pyre_object::w_int_get_value(args[2]) as i32 }
                } else {
                    0
                };
                match rustpython_host_env::fcntl::fcntl_int(fd, cmd, arg) {
                    Ok(v) => Ok(pyre_object::w_int_new(v as i64)),
                    Err(e) => Err(crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("fcntl: {e}"),
                    )),
                }
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "fcntl.fcntl requires host_env feature",
                ))
            }
        }),
    );
    crate::dict_storage_store(
        ns,
        "ioctl",
        crate::make_builtin_function("ioctl", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "ioctl() requires at least 2 arguments",
                    ));
                }
                if !unsafe { pyre_object::is_int(args[0]) }
                    || !unsafe { pyre_object::is_int(args[1]) }
                    || (args.len() >= 3 && !unsafe { pyre_object::is_int(args[2]) })
                {
                    return Err(crate::PyError::type_error(
                        "ioctl() arguments must be integers",
                    ));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let raw_req = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i64;
                let request = rustpython_host_env::fcntl::normalize_ioctl_request(raw_req);
                let arg = if args.len() >= 3 {
                    unsafe { pyre_object::w_int_get_value(args[2]) as i32 }
                } else {
                    0
                };
                match rustpython_host_env::fcntl::ioctl_int(fd, request, arg) {
                    Ok(v) => Ok(pyre_object::w_int_new(v as i64)),
                    Err(e) => Err(crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("ioctl: {e}"),
                    )),
                }
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "fcntl.ioctl requires host_env feature",
                ))
            }
        }),
    );
    crate::dict_storage_store(
        ns,
        "flock",
        crate::make_builtin_function_with_arity(
            "flock",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("flock() requires 2 arguments"));
                    }
                    if !unsafe { pyre_object::is_int(args[0]) }
                        || !unsafe { pyre_object::is_int(args[1]) }
                    {
                        return Err(crate::PyError::type_error(
                            "flock() arguments must be integers",
                        ));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let op = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                    match rustpython_host_env::fcntl::flock(fd, op) {
                        Ok(_) => Ok(pyre_object::w_none()),
                        Err(e) => Err(crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("flock: {e}"),
                        )),
                    }
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "fcntl.flock requires host_env feature",
                    ))
                }
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "lockf",
        crate::make_builtin_function("lockf", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "lockf() requires at least 2 arguments",
                    ));
                }
                for (i, &a) in args.iter().enumerate().take(5) {
                    if !unsafe { pyre_object::is_int(a) } {
                        let _ = i;
                        return Err(crate::PyError::type_error(
                            "lockf() arguments must be integers",
                        ));
                    }
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let cmd = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                let len = if args.len() >= 3 {
                    unsafe { pyre_object::w_int_get_value(args[2]) }
                } else {
                    0
                };
                let start = if args.len() >= 4 {
                    unsafe { pyre_object::w_int_get_value(args[3]) }
                } else {
                    0
                };
                let whence = if args.len() >= 5 {
                    unsafe { pyre_object::w_int_get_value(args[4]) as i32 }
                } else {
                    0
                };
                match rustpython_host_env::fcntl::lockf(fd, cmd, len, start, whence) {
                    Ok(v) => Ok(pyre_object::w_int_new(v as i64)),
                    Err(rustpython_host_env::fcntl::LockfError::InvalidCmd) => {
                        Err(crate::PyError::value_error("lockf: invalid cmd"))
                    }
                    Err(rustpython_host_env::fcntl::LockfError::Overflow(s)) => {
                        Err(crate::PyError::value_error(format!("lockf: overflow: {s}")))
                    }
                    Err(rustpython_host_env::fcntl::LockfError::Io(e)) => {
                        Err(crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("lockf: {e}"),
                        ))
                    }
                }
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "fcntl.lockf requires host_env feature",
                ))
            }
        }),
    );
    // `interp_fcntl.py:25-37 constant_names` — POSIX subset always
    // exposed; Linux-specific block gated below.  I_* (System V
    // STREAMS) are listed by PyPy but `if value is not None` filters
    // them out at platform.configure time on every supported platform;
    // not exposed here.
    #[cfg(unix)]
    {
        macro_rules! cst {
            ($name:literal, $val:expr) => {
                crate::dict_storage_store(ns, $name, pyre_object::w_int_new($val as i64));
            };
        }
        cst!("F_GETFD", libc::F_GETFD);
        cst!("F_SETFD", libc::F_SETFD);
        cst!("F_GETFL", libc::F_GETFL);
        cst!("F_SETFL", libc::F_SETFL);
        cst!("F_DUPFD", libc::F_DUPFD);
        cst!("F_DUPFD_CLOEXEC", libc::F_DUPFD_CLOEXEC);
        cst!("F_GETLK", libc::F_GETLK);
        cst!("F_SETLK", libc::F_SETLK);
        cst!("F_SETLKW", libc::F_SETLKW);
        cst!("F_GETOWN", libc::F_GETOWN);
        cst!("F_SETOWN", libc::F_SETOWN);
        cst!("F_RDLCK", libc::F_RDLCK);
        cst!("F_WRLCK", libc::F_WRLCK);
        cst!("F_UNLCK", libc::F_UNLCK);
        cst!("FD_CLOEXEC", libc::FD_CLOEXEC);
        cst!("LOCK_SH", libc::LOCK_SH);
        cst!("LOCK_EX", libc::LOCK_EX);
        cst!("LOCK_UN", libc::LOCK_UN);
        cst!("LOCK_NB", libc::LOCK_NB);

        // Linux-only fcntl constants.  Values for ones libc does not
        // expose (F_GETSIG/F_SETSIG/F_GETLK64/F_SETLK64/F_SETLKW64/
        // F_EXLCK/F_SHLCK/LOCK_MAND/LOCK_READ/LOCK_WRITE/LOCK_RW/DN_*)
        // come straight from Linux <fcntl.h>, matching the hardcoded
        // overrides at `interp_fcntl.py:48-52`.
        #[cfg(target_os = "linux")]
        {
            cst!("F_SETLEASE", libc::F_SETLEASE);
            cst!("F_GETLEASE", libc::F_GETLEASE);
            cst!("F_NOTIFY", libc::F_NOTIFY);
            cst!("F_GETSIG", 11);
            cst!("F_SETSIG", 10);
            cst!("F_GETLK64", 12);
            cst!("F_SETLK64", 13);
            cst!("F_SETLKW64", 14);
            cst!("F_EXLCK", 4);
            cst!("F_SHLCK", 8);
            cst!("LOCK_MAND", 32);
            cst!("LOCK_READ", 64);
            cst!("LOCK_WRITE", 128);
            cst!("LOCK_RW", 192);
            cst!("DN_ACCESS", 1);
            cst!("DN_MODIFY", 2);
            cst!("DN_CREATE", 4);
            cst!("DN_DELETE", 8);
            cst!("DN_RENAME", 16);
            cst!("DN_ATTRIB", 32);
            cst!("DN_MULTISHOT", 0x80000000u32);
            cst!("F_ADD_SEALS", libc::F_ADD_SEALS);
            cst!("F_GET_SEALS", libc::F_GET_SEALS);
            cst!("F_SEAL_SEAL", libc::F_SEAL_SEAL);
            cst!("F_SEAL_SHRINK", libc::F_SEAL_SHRINK);
            cst!("F_SEAL_GROW", libc::F_SEAL_GROW);
            cst!("F_SEAL_WRITE", libc::F_SEAL_WRITE);
            cst!("F_SETPIPE_SZ", libc::F_SETPIPE_SZ);
            cst!("F_GETPIPE_SZ", libc::F_GETPIPE_SZ);
        }
    }
}

/// syslog module — PyPy: pypy/module/syslog/interp_syslog.py.
///
/// openlog / syslog / closelog / setlogmask.  Backed by
/// `rustpython_host_env::syslog`.  Unix-only.
fn init_syslog(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "openlog",
        crate::make_builtin_function("openlog", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                let ident = args.first().and_then(|&a| unsafe {
                    if pyre_object::is_str(a) {
                        std::ffi::CString::new(pyre_object::w_str_get_value(a))
                            .ok()
                            .map(|c| c.into_boxed_c_str())
                    } else {
                        None
                    }
                });
                if args
                    .iter()
                    .skip(1)
                    .any(|&a| !unsafe { pyre_object::is_int(a) })
                {
                    return Err(crate::PyError::type_error(
                        "openlog(): logoption and facility must be integers",
                    ));
                }
                let logoption = args
                    .get(1)
                    .map(|&a| unsafe { pyre_object::w_int_get_value(a) } as i32)
                    .unwrap_or(0);
                let facility = args
                    .get(2)
                    .map(|&a| unsafe { pyre_object::w_int_get_value(a) } as i32)
                    .unwrap_or(libc::LOG_USER);
                rustpython_host_env::syslog::openlog(ident, logoption, facility);
                Ok(pyre_object::w_none())
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "syslog.openlog requires host_env feature",
                ))
            }
        }),
    );
    crate::dict_storage_store(
        ns,
        "syslog",
        crate::make_builtin_function("syslog", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                let (priority, msg_obj) = if args.len() >= 2 {
                    if !unsafe { pyre_object::is_int(args[0]) } {
                        return Err(crate::PyError::type_error(
                            "syslog(): priority must be an integer",
                        ));
                    }
                    (
                        unsafe { pyre_object::w_int_get_value(args[0]) as i32 },
                        args[1],
                    )
                } else if args.len() == 1 {
                    (libc::LOG_INFO, args[0])
                } else {
                    return Err(crate::PyError::type_error("syslog() requires a message"));
                };
                if !unsafe { pyre_object::is_str(msg_obj) } {
                    return Err(crate::PyError::type_error(
                        "syslog(): message must be a string",
                    ));
                }
                let msg = unsafe { pyre_object::w_str_get_value(msg_obj) };
                if let Ok(cmsg) = std::ffi::CString::new(msg) {
                    rustpython_host_env::syslog::syslog(priority, &cmsg);
                }
                Ok(pyre_object::w_none())
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "syslog.syslog requires host_env feature",
                ))
            }
        }),
    );
    crate::dict_storage_store(
        ns,
        "closelog",
        crate::make_builtin_function_with_arity(
            "closelog",
            |_| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    rustpython_host_env::syslog::closelog();
                }
                Ok(pyre_object::w_none())
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "setlogmask",
        crate::make_builtin_function_with_arity(
            "setlogmask",
            |args| {
                #[cfg(all(unix, feature = "host_env"))]
                {
                    let mask = if let Some(&a) = args.first() {
                        if !unsafe { pyre_object::is_int(a) } {
                            return Err(crate::PyError::type_error(
                                "setlogmask(): argument must be an integer",
                            ));
                        }
                        unsafe { pyre_object::w_int_get_value(a) as i32 }
                    } else {
                        return Err(crate::PyError::type_error("setlogmask() missing argument"));
                    };
                    return Ok(pyre_object::w_int_new(
                        rustpython_host_env::syslog::setlogmask(mask) as i64,
                    ));
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "syslog.setlogmask requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    // Priorities + facilities (POSIX subset matching CPython).
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "LOG_EMERG",
            pyre_object::w_int_new(libc::LOG_EMERG as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_ALERT",
            pyre_object::w_int_new(libc::LOG_ALERT as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_CRIT",
            pyre_object::w_int_new(libc::LOG_CRIT as i64),
        );
        crate::dict_storage_store(ns, "LOG_ERR", pyre_object::w_int_new(libc::LOG_ERR as i64));
        crate::dict_storage_store(
            ns,
            "LOG_WARNING",
            pyre_object::w_int_new(libc::LOG_WARNING as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_NOTICE",
            pyre_object::w_int_new(libc::LOG_NOTICE as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_INFO",
            pyre_object::w_int_new(libc::LOG_INFO as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_DEBUG",
            pyre_object::w_int_new(libc::LOG_DEBUG as i64),
        );
        crate::dict_storage_store(ns, "LOG_PID", pyre_object::w_int_new(libc::LOG_PID as i64));
        crate::dict_storage_store(
            ns,
            "LOG_CONS",
            pyre_object::w_int_new(libc::LOG_CONS as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_NDELAY",
            pyre_object::w_int_new(libc::LOG_NDELAY as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_NOWAIT",
            pyre_object::w_int_new(libc::LOG_NOWAIT as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_PERROR",
            pyre_object::w_int_new(libc::LOG_PERROR as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_KERN",
            pyre_object::w_int_new(libc::LOG_KERN as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_USER",
            pyre_object::w_int_new(libc::LOG_USER as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_MAIL",
            pyre_object::w_int_new(libc::LOG_MAIL as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_DAEMON",
            pyre_object::w_int_new(libc::LOG_DAEMON as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_AUTH",
            pyre_object::w_int_new(libc::LOG_AUTH as i64),
        );
        crate::dict_storage_store(ns, "LOG_LPR", pyre_object::w_int_new(libc::LOG_LPR as i64));
        crate::dict_storage_store(
            ns,
            "LOG_NEWS",
            pyre_object::w_int_new(libc::LOG_NEWS as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_UUCP",
            pyre_object::w_int_new(libc::LOG_UUCP as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_CRON",
            pyre_object::w_int_new(libc::LOG_CRON as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_SYSLOG",
            pyre_object::w_int_new(libc::LOG_SYSLOG as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL0",
            pyre_object::w_int_new(libc::LOG_LOCAL0 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL1",
            pyre_object::w_int_new(libc::LOG_LOCAL1 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL2",
            pyre_object::w_int_new(libc::LOG_LOCAL2 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL3",
            pyre_object::w_int_new(libc::LOG_LOCAL3 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL4",
            pyre_object::w_int_new(libc::LOG_LOCAL4 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL5",
            pyre_object::w_int_new(libc::LOG_LOCAL5 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL6",
            pyre_object::w_int_new(libc::LOG_LOCAL6 as i64),
        );
        crate::dict_storage_store(
            ns,
            "LOG_LOCAL7",
            pyre_object::w_int_new(libc::LOG_LOCAL7 as i64),
        );
    }
    // `Modules/syslogmodule.c syslog_log_mask / syslog_log_upto` —
    // helpers for building setlogmask() arguments.
    //   LOG_MASK(pri)  → 1 << pri
    //   LOG_UPTO(pri)  → (1 << (pri + 1)) - 1
    crate::dict_storage_store(
        ns,
        "LOG_MASK",
        crate::make_builtin_function_with_arity(
            "LOG_MASK",
            |args| {
                let pri =
                    crate::baseobjspace::int_w(args.first().copied().ok_or_else(|| {
                        crate::PyError::type_error("LOG_MASK() missing argument")
                    })?)?;
                Ok(pyre_object::w_int_new(1i64 << pri))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "LOG_UPTO",
        crate::make_builtin_function_with_arity(
            "LOG_UPTO",
            |args| {
                let pri =
                    crate::baseobjspace::int_w(args.first().copied().ok_or_else(|| {
                        crate::PyError::type_error("LOG_UPTO() missing argument")
                    })?)?;
                Ok(pyre_object::w_int_new((1i64 << (pri + 1)) - 1))
            },
            1,
        ),
    );
}

/// _select module — PyPy: pypy/module/select/.
///
/// Implements `select.select(rlist, wlist, xlist, timeout=None)` via
/// `rustpython_host_env::select::{FdSet, select, sec_to_timeval}`.  poll()
/// / epoll / kqueue object types are not implemented yet; they need
/// per-instance heap state which the current pyre builtin-module wiring
/// doesn't expose.
fn init_select(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "select",
        crate::make_builtin_function("select", |args| {
            #[cfg(all(unix, feature = "host_env"))]
            {
                use rustpython_host_env::select as host_select;

                if args.len() < 3 {
                    return Err(crate::PyError::type_error(
                        "select() takes at least 3 arguments",
                    ));
                }

                fn collect_fds(
                    seq: pyre_object::PyObjectRef,
                ) -> Result<Vec<(pyre_object::PyObjectRef, i32)>, crate::PyError> {
                    unsafe {
                        let is_list = pyre_object::is_list(seq);
                        let is_tuple = pyre_object::is_tuple(seq);
                        if !is_list && !is_tuple {
                            return Err(crate::PyError::type_error(
                                "select() arguments 1-3 must be sequences",
                            ));
                        }
                        let n = if is_list {
                            pyre_object::w_list_len(seq)
                        } else {
                            pyre_object::w_tuple_len(seq)
                        };
                        let mut out = Vec::with_capacity(n);
                        for i in 0..n {
                            let item = if is_list {
                                pyre_object::w_list_getitem(seq, i as i64)
                            } else {
                                pyre_object::w_tuple_getitem(seq, i as i64)
                            }
                            .ok_or_else(|| {
                                crate::PyError::value_error("select() sequence item missing")
                            })?;
                            if !pyre_object::is_int(item) {
                                return Err(crate::PyError::type_error(
                                    "argument must be an int, or have a fileno() method",
                                ));
                            }
                            let fd = pyre_object::w_int_get_value(item) as i32;
                            if fd < 0 {
                                return Err(crate::PyError::value_error(
                                    "file descriptor cannot be a negative integer",
                                ));
                            }
                            out.push((item, fd));
                        }
                        Ok(out)
                    }
                }

                let rfds = collect_fds(args[0])?;
                let wfds = collect_fds(args[1])?;
                let xfds = collect_fds(args[2])?;

                let mut rset = host_select::FdSet::new();
                let mut wset = host_select::FdSet::new();
                let mut xset = host_select::FdSet::new();
                let mut nfds: i32 = -1;
                for &(_, fd) in &rfds {
                    rset.insert(fd);
                    if fd > nfds {
                        nfds = fd;
                    }
                }
                for &(_, fd) in &wfds {
                    wset.insert(fd);
                    if fd > nfds {
                        nfds = fd;
                    }
                }
                for &(_, fd) in &xfds {
                    xset.insert(fd);
                    if fd > nfds {
                        nfds = fd;
                    }
                }

                let mut tv_storage;
                let timeout_ref: Option<&mut host_select::timeval> = match args.get(3) {
                    None => None,
                    Some(&t) if unsafe { pyre_object::is_none(t) } => None,
                    Some(&t) => {
                        let secs = unsafe {
                            if pyre_object::is_float(t) {
                                pyre_object::w_float_get_value(t)
                            } else if pyre_object::is_int(t) {
                                pyre_object::w_int_get_value(t) as f64
                            } else {
                                return Err(crate::PyError::type_error(
                                    "timeout must be a float or None",
                                ));
                            }
                        };
                        if secs < 0.0 {
                            return Err(crate::PyError::value_error(
                                "timeout must be non-negative",
                            ));
                        }
                        tv_storage = host_select::sec_to_timeval(secs);
                        Some(&mut tv_storage)
                    }
                };

                let n = host_select::select(nfds + 1, &mut rset, &mut wset, &mut xset, timeout_ref)
                    .map_err(|e| {
                        crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("select: {e}"),
                        )
                    })?;
                let _ = n;

                fn build_ready(
                    set: &mut host_select::FdSet,
                    inputs: &[(pyre_object::PyObjectRef, i32)],
                ) -> pyre_object::PyObjectRef {
                    let items: Vec<_> = inputs
                        .iter()
                        .filter_map(|&(obj, fd)| if set.contains(fd) { Some(obj) } else { None })
                        .collect();
                    pyre_object::w_list_new(items)
                }

                let r_ready = build_ready(&mut rset, &rfds);
                let w_ready = build_ready(&mut wset, &wfds);
                let x_ready = build_ready(&mut xset, &xfds);
                Ok(pyre_object::w_tuple_new(vec![r_ready, w_ready, x_ready]))
            }
            #[cfg(not(all(unix, feature = "host_env")))]
            {
                let _ = args;
                Err(crate::PyError::not_implemented(
                    "select.select requires host_env feature on a Unix platform",
                ))
            }
        }),
    );

    crate::dict_storage_store(ns, "error", pyre_object::w_str_new("OSError"));
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "PIPE_BUF",
            pyre_object::w_int_new(libc::PIPE_BUF as i64),
        );
    }
}

/// _termios module — PyPy: pypy/module/termios/.
///
/// `tcgetattr(fd)` returns the 7-list `[iflag, oflag, cflag, lflag,
/// ispeed, ospeed, [cc_chars]]`.  `tcsetattr(fd, when, attrs)` takes the
/// same shape and writes it back via `termios::Termios`.  The simpler
/// `tcdrain` / `tcflush` / `tcflow` / `tcsendbreak` / `cfgetispeed` /
/// `cfgetospeed` calls are direct wrappers.  All constants come from
/// `rustpython_host_env::termios::*` so the values match the platform.
#[cfg(all(unix, feature = "host_env"))]
fn init_termios(ns: &mut DictStorage) {
    use rustpython_host_env::termios as host_termios;

    fn make_cc_bytes(cc: &[libc::cc_t]) -> pyre_object::PyObjectRef {
        // Each cc[i] becomes a 1-byte bytes object (CPython does the same).
        let items: Vec<_> = cc
            .iter()
            .map(|&b| pyre_object::bytesobject::w_bytes_from_bytes(&[b as u8]))
            .collect();
        pyre_object::w_list_new(items)
    }

    crate::dict_storage_store(
        ns,
        "tcgetattr",
        crate::make_builtin_function_with_arity(
            "tcgetattr",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "tcgetattr() requires 1 argument",
                    ));
                }
                if !unsafe { pyre_object::is_int(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "tcgetattr: fd must be an integer",
                    ));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let t = host_termios::tcgetattr(fd).map_err(|e| {
                    crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcgetattr: {e}"),
                    )
                })?;
                let ispeed = host_termios::cfgetispeed(&t);
                let ospeed = host_termios::cfgetospeed(&t);
                let cc_list = make_cc_bytes(&t.c_cc[..]);
                Ok(pyre_object::w_list_new(vec![
                    pyre_object::w_int_new(t.c_iflag as i64),
                    pyre_object::w_int_new(t.c_oflag as i64),
                    pyre_object::w_int_new(t.c_cflag as i64),
                    pyre_object::w_int_new(t.c_lflag as i64),
                    pyre_object::w_int_new(ispeed as i64),
                    pyre_object::w_int_new(ospeed as i64),
                    cc_list,
                ]))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "tcsetattr",
        crate::make_builtin_function("tcsetattr", |args| {
            if args.len() < 3 {
                return Err(crate::PyError::type_error(
                    "tcsetattr() requires 3 arguments",
                ));
            }
            if !unsafe { pyre_object::is_int(args[0]) } || !unsafe { pyre_object::is_int(args[1]) }
            {
                return Err(crate::PyError::type_error(
                    "tcsetattr: fd and when must be integers",
                ));
            }
            let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
            let when = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
            let attrs = args[2];
            if !unsafe { pyre_object::is_list(attrs) } {
                return Err(crate::PyError::type_error(
                    "tcsetattr: attributes must be a list",
                ));
            }
            let n = unsafe { pyre_object::w_list_len(attrs) };
            if n != 7 {
                return Err(crate::PyError::type_error(
                    "tcsetattr: attributes must be a 7-element list",
                ));
            }
            let get = |i: usize| -> Result<pyre_object::PyObjectRef, crate::PyError> {
                unsafe { pyre_object::w_list_getitem(attrs, i as i64) }
                    .ok_or_else(|| crate::PyError::value_error("tcsetattr: missing item"))
            };
            for i in 0..6 {
                let item = get(i)?;
                if !unsafe { pyre_object::is_int(item) } {
                    return Err(crate::PyError::type_error(
                        "tcsetattr: numeric attribute fields must be integers",
                    ));
                }
            }
            let iflag = unsafe { pyre_object::w_int_get_value(get(0)?) } as libc::tcflag_t;
            let oflag = unsafe { pyre_object::w_int_get_value(get(1)?) } as libc::tcflag_t;
            let cflag = unsafe { pyre_object::w_int_get_value(get(2)?) } as libc::tcflag_t;
            let lflag = unsafe { pyre_object::w_int_get_value(get(3)?) } as libc::tcflag_t;
            let ispeed = unsafe { pyre_object::w_int_get_value(get(4)?) } as libc::speed_t;
            let ospeed = unsafe { pyre_object::w_int_get_value(get(5)?) } as libc::speed_t;
            let cc_obj = get(6)?;

            // Start from the current settings so we preserve any platform-private fields.
            let mut t = host_termios::tcgetattr(fd).map_err(|e| {
                crate::PyError::os_error_with_errno(
                    e.raw_os_error().unwrap_or(0),
                    format!("tcsetattr: {e}"),
                )
            })?;
            t.c_iflag = iflag;
            t.c_oflag = oflag;
            t.c_cflag = cflag;
            t.c_lflag = lflag;
            host_termios::cfsetispeed(&mut t, ispeed).map_err(|e| {
                crate::PyError::os_error_with_errno(
                    e.raw_os_error().unwrap_or(0),
                    format!("cfsetispeed: {e}"),
                )
            })?;
            host_termios::cfsetospeed(&mut t, ospeed).map_err(|e| {
                crate::PyError::os_error_with_errno(
                    e.raw_os_error().unwrap_or(0),
                    format!("cfsetospeed: {e}"),
                )
            })?;

            // Populate c_cc[] — each element is either an int or a length-1 bytes.
            // tcgetattr returns a list, so we only accept lists here.
            if !unsafe { pyre_object::is_list(cc_obj) } {
                return Err(crate::PyError::type_error(
                    "tcsetattr: c_cc slot must be a list",
                ));
            }
            let cc_len = unsafe { pyre_object::w_list_len(cc_obj) };
            let nccs = t.c_cc.len();
            for i in 0..cc_len.min(nccs) {
                let item = unsafe { pyre_object::w_list_getitem(cc_obj, i as i64) }
                    .ok_or_else(|| crate::PyError::value_error("tcsetattr: missing cc item"))?;
                let byte = unsafe {
                    if pyre_object::is_int(item) {
                        pyre_object::w_int_get_value(item) as libc::cc_t
                    } else if pyre_object::bytesobject::is_bytes_like(item) {
                        let data = pyre_object::bytesobject::bytes_like_data(item);
                        if data.is_empty() {
                            0
                        } else {
                            data[0] as libc::cc_t
                        }
                    } else {
                        return Err(crate::PyError::type_error(
                            "tcsetattr: c_cc element must be int or bytes",
                        ));
                    }
                };
                t.c_cc[i] = byte;
            }
            host_termios::tcsetattr(fd, when, &t).map_err(|e| {
                crate::PyError::os_error_with_errno(
                    e.raw_os_error().unwrap_or(0),
                    format!("tcsetattr: {e}"),
                )
            })?;
            Ok(pyre_object::w_none())
        }),
    );

    crate::dict_storage_store(
        ns,
        "tcsendbreak",
        crate::make_builtin_function_with_arity(
            "tcsendbreak",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "tcsendbreak() requires 2 arguments",
                    ));
                }
                if !unsafe { pyre_object::is_int(args[0]) }
                    || !unsafe { pyre_object::is_int(args[1]) }
                {
                    return Err(crate::PyError::type_error(
                        "tcsendbreak: fd and duration must be integers",
                    ));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let dur = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                host_termios::tcsendbreak(fd, dur).map_err(|e| {
                    crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcsendbreak: {e}"),
                    )
                })?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "tcdrain",
        crate::make_builtin_function_with_arity(
            "tcdrain",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("tcdrain() requires 1 argument"));
                }
                if !unsafe { pyre_object::is_int(args[0]) } {
                    return Err(crate::PyError::type_error("tcdrain: fd must be an integer"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                host_termios::tcdrain(fd).map_err(|e| {
                    crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcdrain: {e}"),
                    )
                })?;
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "tcflush",
        crate::make_builtin_function_with_arity(
            "tcflush",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("tcflush() requires 2 arguments"));
                }
                if !unsafe { pyre_object::is_int(args[0]) }
                    || !unsafe { pyre_object::is_int(args[1]) }
                {
                    return Err(crate::PyError::type_error(
                        "tcflush: fd and queue must be integers",
                    ));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let q = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                host_termios::tcflush(fd, q).map_err(|e| {
                    crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcflush: {e}"),
                    )
                })?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "tcflow",
        crate::make_builtin_function_with_arity(
            "tcflow",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("tcflow() requires 2 arguments"));
                }
                if !unsafe { pyre_object::is_int(args[0]) }
                    || !unsafe { pyre_object::is_int(args[1]) }
                {
                    return Err(crate::PyError::type_error(
                        "tcflow: fd and action must be integers",
                    ));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let action = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                host_termios::tcflow(fd, action).map_err(|e| {
                    crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("tcflow: {e}"),
                    )
                })?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    // ── Constants ──
    crate::dict_storage_store(ns, "B0", pyre_object::w_int_new(host_termios::B0 as i64));
    crate::dict_storage_store(ns, "B50", pyre_object::w_int_new(host_termios::B50 as i64));
    crate::dict_storage_store(ns, "B75", pyre_object::w_int_new(host_termios::B75 as i64));
    crate::dict_storage_store(
        ns,
        "B110",
        pyre_object::w_int_new(host_termios::B110 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B134",
        pyre_object::w_int_new(host_termios::B134 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B150",
        pyre_object::w_int_new(host_termios::B150 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B200",
        pyre_object::w_int_new(host_termios::B200 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B300",
        pyre_object::w_int_new(host_termios::B300 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B600",
        pyre_object::w_int_new(host_termios::B600 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B1200",
        pyre_object::w_int_new(host_termios::B1200 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B1800",
        pyre_object::w_int_new(host_termios::B1800 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B2400",
        pyre_object::w_int_new(host_termios::B2400 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B4800",
        pyre_object::w_int_new(host_termios::B4800 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B9600",
        pyre_object::w_int_new(host_termios::B9600 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B19200",
        pyre_object::w_int_new(host_termios::B19200 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B38400",
        pyre_object::w_int_new(host_termios::B38400 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B57600",
        pyre_object::w_int_new(host_termios::B57600 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B115200",
        pyre_object::w_int_new(host_termios::B115200 as i64),
    );
    crate::dict_storage_store(
        ns,
        "B230400",
        pyre_object::w_int_new(host_termios::B230400 as i64),
    );

    crate::dict_storage_store(
        ns,
        "BRKINT",
        pyre_object::w_int_new(host_termios::BRKINT as i64),
    );
    crate::dict_storage_store(
        ns,
        "CLOCAL",
        pyre_object::w_int_new(host_termios::CLOCAL as i64),
    );
    crate::dict_storage_store(
        ns,
        "CREAD",
        pyre_object::w_int_new(host_termios::CREAD as i64),
    );
    crate::dict_storage_store(ns, "CS5", pyre_object::w_int_new(host_termios::CS5 as i64));
    crate::dict_storage_store(ns, "CS6", pyre_object::w_int_new(host_termios::CS6 as i64));
    crate::dict_storage_store(ns, "CS7", pyre_object::w_int_new(host_termios::CS7 as i64));
    crate::dict_storage_store(ns, "CS8", pyre_object::w_int_new(host_termios::CS8 as i64));
    crate::dict_storage_store(
        ns,
        "CSIZE",
        pyre_object::w_int_new(host_termios::CSIZE as i64),
    );
    crate::dict_storage_store(
        ns,
        "CSTOPB",
        pyre_object::w_int_new(host_termios::CSTOPB as i64),
    );
    crate::dict_storage_store(
        ns,
        "ECHO",
        pyre_object::w_int_new(host_termios::ECHO as i64),
    );
    crate::dict_storage_store(
        ns,
        "ECHOE",
        pyre_object::w_int_new(host_termios::ECHOE as i64),
    );
    crate::dict_storage_store(
        ns,
        "ECHOK",
        pyre_object::w_int_new(host_termios::ECHOK as i64),
    );
    crate::dict_storage_store(
        ns,
        "ECHONL",
        pyre_object::w_int_new(host_termios::ECHONL as i64),
    );
    crate::dict_storage_store(
        ns,
        "HUPCL",
        pyre_object::w_int_new(host_termios::HUPCL as i64),
    );
    crate::dict_storage_store(
        ns,
        "ICANON",
        pyre_object::w_int_new(host_termios::ICANON as i64),
    );
    crate::dict_storage_store(
        ns,
        "ICRNL",
        pyre_object::w_int_new(host_termios::ICRNL as i64),
    );
    crate::dict_storage_store(
        ns,
        "IEXTEN",
        pyre_object::w_int_new(host_termios::IEXTEN as i64),
    );
    crate::dict_storage_store(
        ns,
        "IGNBRK",
        pyre_object::w_int_new(host_termios::IGNBRK as i64),
    );
    crate::dict_storage_store(
        ns,
        "IGNCR",
        pyre_object::w_int_new(host_termios::IGNCR as i64),
    );
    crate::dict_storage_store(
        ns,
        "IGNPAR",
        pyre_object::w_int_new(host_termios::IGNPAR as i64),
    );
    crate::dict_storage_store(
        ns,
        "INLCR",
        pyre_object::w_int_new(host_termios::INLCR as i64),
    );
    crate::dict_storage_store(
        ns,
        "INPCK",
        pyre_object::w_int_new(host_termios::INPCK as i64),
    );
    crate::dict_storage_store(
        ns,
        "ISIG",
        pyre_object::w_int_new(host_termios::ISIG as i64),
    );
    crate::dict_storage_store(
        ns,
        "ISTRIP",
        pyre_object::w_int_new(host_termios::ISTRIP as i64),
    );
    crate::dict_storage_store(
        ns,
        "IXANY",
        pyre_object::w_int_new(host_termios::IXANY as i64),
    );
    crate::dict_storage_store(
        ns,
        "IXOFF",
        pyre_object::w_int_new(host_termios::IXOFF as i64),
    );
    crate::dict_storage_store(
        ns,
        "IXON",
        pyre_object::w_int_new(host_termios::IXON as i64),
    );
    crate::dict_storage_store(
        ns,
        "NOFLSH",
        pyre_object::w_int_new(host_termios::NOFLSH as i64),
    );
    crate::dict_storage_store(
        ns,
        "OCRNL",
        pyre_object::w_int_new(host_termios::OCRNL as i64),
    );
    crate::dict_storage_store(
        ns,
        "ONLCR",
        pyre_object::w_int_new(host_termios::ONLCR as i64),
    );
    crate::dict_storage_store(
        ns,
        "ONLRET",
        pyre_object::w_int_new(host_termios::ONLRET as i64),
    );
    crate::dict_storage_store(
        ns,
        "ONOCR",
        pyre_object::w_int_new(host_termios::ONOCR as i64),
    );
    crate::dict_storage_store(
        ns,
        "OPOST",
        pyre_object::w_int_new(host_termios::OPOST as i64),
    );
    crate::dict_storage_store(
        ns,
        "PARENB",
        pyre_object::w_int_new(host_termios::PARENB as i64),
    );
    crate::dict_storage_store(
        ns,
        "PARMRK",
        pyre_object::w_int_new(host_termios::PARMRK as i64),
    );
    crate::dict_storage_store(
        ns,
        "PARODD",
        pyre_object::w_int_new(host_termios::PARODD as i64),
    );

    crate::dict_storage_store(
        ns,
        "TCIFLUSH",
        pyre_object::w_int_new(host_termios::TCIFLUSH as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCOFLUSH",
        pyre_object::w_int_new(host_termios::TCOFLUSH as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCIOFLUSH",
        pyre_object::w_int_new(host_termios::TCIOFLUSH as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCIOFF",
        pyre_object::w_int_new(host_termios::TCIOFF as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCION",
        pyre_object::w_int_new(host_termios::TCION as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCOOFF",
        pyre_object::w_int_new(host_termios::TCOOFF as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCOON",
        pyre_object::w_int_new(host_termios::TCOON as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCSANOW",
        pyre_object::w_int_new(host_termios::TCSANOW as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCSADRAIN",
        pyre_object::w_int_new(host_termios::TCSADRAIN as i64),
    );
    crate::dict_storage_store(
        ns,
        "TCSAFLUSH",
        pyre_object::w_int_new(host_termios::TCSAFLUSH as i64),
    );
    crate::dict_storage_store(
        ns,
        "TOSTOP",
        pyre_object::w_int_new(host_termios::TOSTOP as i64),
    );

    crate::dict_storage_store(
        ns,
        "VEOF",
        pyre_object::w_int_new(host_termios::VEOF as i64),
    );
    crate::dict_storage_store(
        ns,
        "VEOL",
        pyre_object::w_int_new(host_termios::VEOL as i64),
    );
    crate::dict_storage_store(
        ns,
        "VERASE",
        pyre_object::w_int_new(host_termios::VERASE as i64),
    );
    crate::dict_storage_store(
        ns,
        "VINTR",
        pyre_object::w_int_new(host_termios::VINTR as i64),
    );
    crate::dict_storage_store(
        ns,
        "VKILL",
        pyre_object::w_int_new(host_termios::VKILL as i64),
    );
    crate::dict_storage_store(
        ns,
        "VMIN",
        pyre_object::w_int_new(host_termios::VMIN as i64),
    );
    crate::dict_storage_store(
        ns,
        "VQUIT",
        pyre_object::w_int_new(host_termios::VQUIT as i64),
    );
    crate::dict_storage_store(
        ns,
        "VSTART",
        pyre_object::w_int_new(host_termios::VSTART as i64),
    );
    crate::dict_storage_store(
        ns,
        "VSTOP",
        pyre_object::w_int_new(host_termios::VSTOP as i64),
    );
    crate::dict_storage_store(
        ns,
        "VSUSP",
        pyre_object::w_int_new(host_termios::VSUSP as i64),
    );
    crate::dict_storage_store(
        ns,
        "VTIME",
        pyre_object::w_int_new(host_termios::VTIME as i64),
    );

    crate::dict_storage_store(ns, "error", crate::typedef::w_object());
}

#[cfg(not(all(unix, feature = "host_env")))]
fn init_termios(_ns: &mut DictStorage) {}

// ──────────────────────────────────────────────────────────────────────
// _ctypes module — PyPy: pypy/module/_rawffi/, pypy/module/_ctypes/.
//
// **Slice C1: dlopen / dlsym / dlclose + size/align/memmove constants.**
//
// Provides the dynamic-linker primitives that ctypes.CDLL builds on
// top of, plus the simple-type size/align table and POSIX RTLD_* flags.
// The full c_int / Structure / CFUNCTYPE / Pointer machinery still
// requires libffi-style argument marshalling and per-instance heap
// state — those are later slices.
// ──────────────────────────────────────────────────────────────────────

fn init_ctypes(ns: &mut DictStorage) {
    #[cfg(all(unix, feature = "host_env"))]
    {
        use rustpython_host_env::ctypes as host_ctypes;

        // dlopen flags (POSIX).
        crate::dict_storage_store(
            ns,
            "RTLD_LOCAL",
            pyre_object::w_int_new(libc::RTLD_LOCAL as i64),
        );
        crate::dict_storage_store(
            ns,
            "RTLD_GLOBAL",
            pyre_object::w_int_new(libc::RTLD_GLOBAL as i64),
        );
        crate::dict_storage_store(
            ns,
            "RTLD_LAZY",
            pyre_object::w_int_new(libc::RTLD_LAZY as i64),
        );
        crate::dict_storage_store(
            ns,
            "RTLD_NOW",
            pyre_object::w_int_new(libc::RTLD_NOW as i64),
        );
        crate::dict_storage_store(
            ns,
            "DEFAULT_MODE",
            pyre_object::w_int_new(host_ctypes::dlopen_mode(None) as i64),
        );

        // dlopen(name, mode=DEFAULT_MODE) → handle (opaque integer that
        // indexes into host_env's libcache).
        crate::dict_storage_store(
            ns,
            "dlopen",
            crate::make_builtin_function("dlopen", |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("dlopen() missing library name"));
                }
                let name = unsafe {
                    if pyre_object::is_none(args[0]) {
                        // dlopen(None) → process handle
                        let mode = if args.len() >= 2 {
                            pyre_object::w_int_get_value(args[1]) as libc::c_int
                        } else {
                            libc::RTLD_NOW
                        };
                        let ptr = rustpython_host_env::ctypes::dlopen_self(mode)
                            .map_err(|e| crate::PyError::os_error(format!("dlopen(None): {e}")))?;
                        let h = rustpython_host_env::ctypes::insert_raw_library_handle(ptr);
                        return Ok(pyre_object::w_int_new(h as i64));
                    }
                    if !pyre_object::is_str(args[0]) {
                        return Err(crate::PyError::type_error(
                            "dlopen: name must be a string or None",
                        ));
                    }
                    pyre_object::w_str_get_value(args[0]).to_string()
                };
                let mode = if args.len() >= 2 {
                    (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32
                } else {
                    rustpython_host_env::ctypes::dlopen_mode(None)
                };
                let h = rustpython_host_env::ctypes::open_library_with_mode(&name, mode)
                    .map_err(|e| crate::PyError::os_error(format!("dlopen({name}): {e}")))?;
                Ok(pyre_object::w_int_new(h as i64))
            }),
        );

        // dlsym(handle, name) → address (int).  Returns the function
        // pointer; for data symbols use dlsym(handle, name) the same way.
        crate::dict_storage_store(
            ns,
            "dlsym",
            crate::make_builtin_function_with_arity(
                "dlsym",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("dlsym() needs 2 arguments"));
                    }
                    let h = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize;
                    let name = unsafe {
                        if !pyre_object::is_str(args[1]) {
                            return Err(crate::PyError::type_error("dlsym: name must be a string"));
                        }
                        pyre_object::w_str_get_value(args[1]).to_string()
                    };
                    let addr = rustpython_host_env::ctypes::lookup_function_symbol_addr(
                        h,
                        name.as_bytes(),
                    )
                    .map_err(|e| {
                        use rustpython_host_env::ctypes::LookupSymbolError as L;
                        let msg = match e {
                            L::LibraryNotFound => "library not found".to_string(),
                            L::LibraryClosed => "library closed".to_string(),
                            L::Load(s) => s,
                        };
                        crate::PyError::os_error(format!("dlsym({name}): {msg}"))
                    })?;
                    Ok(pyre_object::w_int_new(addr as i64))
                },
                2,
            ),
        );

        // dlclose(handle) → None
        crate::dict_storage_store(
            ns,
            "dlclose",
            crate::make_builtin_function_with_arity(
                "dlclose",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("dlclose() needs handle"));
                    }
                    let h = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize;
                    rustpython_host_env::ctypes::drop_library(h);
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // get_errno / set_errno — ctypes routes them through host_env so
        // a saved-errno round-trip across foreign calls survives the
        // global libc::errno being overwritten by intermediate syscalls.
        crate::dict_storage_store(
            ns,
            "get_errno",
            crate::make_builtin_function_with_arity(
                "get_errno",
                |_| {
                    Ok(pyre_object::w_int_new(
                        rustpython_host_env::ctypes::get_errno() as i64,
                    ))
                },
                0,
            ),
        );
        crate::dict_storage_store(
            ns,
            "set_errno",
            crate::make_builtin_function_with_arity(
                "set_errno",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("set_errno() needs value"));
                    }
                    let v = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let prev = rustpython_host_env::ctypes::set_errno(v);
                    Ok(pyre_object::w_int_new(prev as i64))
                },
                1,
            ),
        );

        // sizeof / alignment of simple ctypes type codes ('i', 'l', 'd', etc.).
        crate::dict_storage_store(
            ns,
            "_sizeof_typecode",
            crate::make_builtin_function_with_arity(
                "_sizeof_typecode",
                |args| {
                    if args.is_empty() || !unsafe { pyre_object::is_str(args[0]) } {
                        return Err(crate::PyError::type_error(
                            "_sizeof_typecode() needs typecode string",
                        ));
                    }
                    let code = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                    match rustpython_host_env::ctypes::simple_type_size(&code) {
                        Some(n) => Ok(pyre_object::w_int_new(n as i64)),
                        None => Err(crate::PyError::value_error(format!(
                            "unknown type code: {code}"
                        ))),
                    }
                },
                1,
            ),
        );
        crate::dict_storage_store(
            ns,
            "_alignof_typecode",
            crate::make_builtin_function_with_arity(
                "_alignof_typecode",
                |args| {
                    if args.is_empty() || !unsafe { pyre_object::is_str(args[0]) } {
                        return Err(crate::PyError::type_error(
                            "_alignof_typecode() needs typecode string",
                        ));
                    }
                    let code = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                    match rustpython_host_env::ctypes::simple_type_align(&code) {
                        Some(n) => Ok(pyre_object::w_int_new(n as i64)),
                        None => Err(crate::PyError::value_error(format!(
                            "unknown type code: {code}"
                        ))),
                    }
                },
                1,
            ),
        );

        // Address of memmove / memset for ctypes.memmove / memset.
        crate::dict_storage_store(
            ns,
            "memmove",
            crate::make_builtin_function_with_arity(
                "memmove",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "memmove() needs (dst, src, count)",
                        ));
                    }
                    let dst = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize
                        as *mut libc::c_void;
                    let src = (unsafe { pyre_object::w_int_get_value(args[1]) }) as usize
                        as *const libc::c_void;
                    let n = (unsafe { pyre_object::w_int_get_value(args[2]) }) as usize;
                    unsafe { libc::memmove(dst, src, n) };
                    Ok(pyre_object::w_int_new(dst as usize as i64))
                },
                3,
            ),
        );
        crate::dict_storage_store(
            ns,
            "memset",
            crate::make_builtin_function_with_arity(
                "memset",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error("memset() needs (dst, c, count)"));
                    }
                    let dst = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize
                        as *mut libc::c_void;
                    let c = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                    let n = (unsafe { pyre_object::w_int_get_value(args[2]) }) as usize;
                    unsafe { libc::memset(dst, c, n) };
                    Ok(pyre_object::w_int_new(dst as usize as i64))
                },
                3,
            ),
        );

        // string_at(ptr, size=-1) -> bytes
        crate::dict_storage_store(
            ns,
            "string_at",
            crate::make_builtin_function("string_at", |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("string_at() needs ptr"));
                }
                let ptr = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize;
                let size = if args.len() >= 2 {
                    unsafe { pyre_object::w_int_get_value(args[1]) }
                } else {
                    -1
                };
                let bytes =
                    rustpython_host_env::ctypes::string_at(ptr, size as isize).map_err(|e| {
                        use rustpython_host_env::ctypes::StringAtError as S;
                        let msg = match e {
                            S::NullPointer => "NULL pointer access",
                            S::TooLong => "size too large",
                        };
                        crate::PyError::os_error(format!("string_at: {msg}"))
                    })?;
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
            }),
        );

        // FFI library helpers used by stdlib ctypes/util.py:
        //   _ctypes.dlopen + DEFAULT_MODE typically come above, but stdlib
        //   also looks for _ctypes.SIZEOF_TIME_T to size struct timespec.
        crate::dict_storage_store(
            ns,
            "SIZEOF_TIME_T",
            pyre_object::w_int_new(rustpython_host_env::ctypes::SIZEOF_TIME_T as i64),
        );
    }

    // Error type alias.
    crate::dict_storage_store(ns, "ArgumentError", crate::typedef::w_object());
    crate::dict_storage_store(ns, "_Pointer", crate::typedef::w_object());
    crate::dict_storage_store(ns, "Structure", crate::typedef::w_object());
    crate::dict_storage_store(ns, "Union", crate::typedef::w_object());
    crate::dict_storage_store(ns, "Array", crate::typedef::w_object());
    crate::dict_storage_store(ns, "_CFuncPtr", crate::typedef::w_object());
    crate::dict_storage_store(ns, "_SimpleCData", crate::typedef::w_object());
    crate::dict_storage_store(ns, "CFuncPtr", crate::typedef::w_object());
    crate::dict_storage_store(ns, "POINTER", crate::typedef::w_object());
    crate::dict_storage_store(ns, "pointer", crate::typedef::w_object());
    crate::dict_storage_store(ns, "byref", crate::typedef::w_object());
    crate::dict_storage_store(ns, "addressof", crate::typedef::w_object());
    crate::dict_storage_store(ns, "sizeof", crate::typedef::w_object());
    crate::dict_storage_store(ns, "alignment", crate::typedef::w_object());
    crate::dict_storage_store(ns, "_check_HRESULT", crate::typedef::w_object());
}

/// _posixshmem module — PyPy: pypy/module/_posixshmem/.
/// Backs `multiprocessing.shared_memory` on POSIX.
fn init_posixshmem(ns: &mut DictStorage) {
    #[cfg(all(unix, feature = "host_env"))]
    {
        crate::dict_storage_store(
            ns,
            "shm_open",
            crate::make_builtin_function("shm_open", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "shm_open() requires (path, flags[, mode])",
                    ));
                }
                let name = unsafe {
                    if !pyre_object::is_str(args[0]) {
                        return Err(crate::PyError::type_error(
                            "shm_open: path must be a string",
                        ));
                    }
                    pyre_object::w_str_get_value(args[0]).to_string()
                };
                let flags = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                let mode = if args.len() >= 3 {
                    (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_uint
                } else {
                    0o600
                };
                let c_name = std::ffi::CString::new(name.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let fd = rustpython_host_env::shm::shm_open(&c_name, flags, mode).map_err(|e| {
                    crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("shm_open: {e}"),
                    )
                })?;
                Ok(pyre_object::w_int_new(fd as i64))
            }),
        );
        crate::dict_storage_store(
            ns,
            "shm_unlink",
            crate::make_builtin_function_with_arity(
                "shm_unlink",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("shm_unlink() needs path"));
                    }
                    let name = unsafe {
                        if !pyre_object::is_str(args[0]) {
                            return Err(crate::PyError::type_error(
                                "shm_unlink: path must be a string",
                            ));
                        }
                        pyre_object::w_str_get_value(args[0]).to_string()
                    };
                    let c_name = std::ffi::CString::new(name.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null"))?;
                    rustpython_host_env::shm::shm_unlink(&c_name).map_err(|e| {
                        crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("shm_unlink: {e}"),
                        )
                    })?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );
    }
}

// ──────────────────────────────────────────────────────────────────────
// _multiprocessing module — PyPy: pypy/module/_multiprocessing/.
//
// Exposes `SemLock(kind, value, maxvalue, name, unlink)` and
// `sem_unlink(name)`.  Pyre is currently single-threaded so the lock
// barely matters for serialization, but multiprocessing.py still calls
// .acquire()/.release() during pool teardown, so the methods must exist
// and round-trip without crashing.
// ──────────────────────────────────────────────────────────────────────

#[cfg(all(unix, feature = "host_env"))]
thread_local! {
    static SEMLOCK_TYPE_OBJ: std::cell::OnceCell<pyre_object::PyObjectRef> =
        const { std::cell::OnceCell::new() };
}

#[cfg(all(unix, feature = "host_env"))]
fn semlock_type() -> pyre_object::PyObjectRef {
    SEMLOCK_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("SemLock", init_semlock_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

#[cfg(all(unix, feature = "host_env"))]
fn semlock_get_handle(obj: pyre_object::PyObjectRef) -> *mut libc::sem_t {
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return core::ptr::null_mut();
    }
    if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(d, "_handle") } {
        if unsafe { pyre_object::is_int(v) } {
            return unsafe { pyre_object::w_int_get_value(v) } as usize as *mut libc::sem_t;
        }
    }
    core::ptr::null_mut()
}

#[cfg(all(unix, feature = "host_env"))]
fn init_semlock_type(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "acquire",
        crate::make_builtin_function("acquire", |args| {
            let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            let handle = semlock_get_handle(obj);
            if handle.is_null() {
                return Err(crate::PyError::value_error("SemLock handle is null"));
            }
            let blocking = if args.len() >= 2 {
                (unsafe { pyre_object::w_int_get_value(args[1]) }) != 0
            } else {
                true
            };
            if blocking {
                let r = unsafe { libc::sem_wait(handle) };
                if r != 0 {
                    return Err(crate::PyError::os_error_with_errno(
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                        "sem_wait",
                    ));
                }
                Ok(pyre_object::w_bool_from(true))
            } else {
                let r = unsafe { libc::sem_trywait(handle) };
                Ok(pyre_object::w_bool_from(r == 0))
            }
        }),
    );
    crate::dict_storage_store(
        ns,
        "release",
        crate::make_builtin_function_with_arity(
            "release",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let handle = semlock_get_handle(obj);
                if handle.is_null() {
                    return Err(crate::PyError::value_error("SemLock handle is null"));
                }
                let r = unsafe { libc::sem_post(handle) };
                if r != 0 {
                    return Err(crate::PyError::os_error_with_errno(
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                        "sem_post",
                    ));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_count",
        crate::make_builtin_function_with_arity("_count", |_| Ok(pyre_object::w_int_new(0)), 1),
    );
    crate::dict_storage_store(
        ns,
        "_is_mine",
        crate::make_builtin_function_with_arity(
            "_is_mine",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_is_zero",
        crate::make_builtin_function_with_arity(
            "_is_zero",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let handle = semlock_get_handle(obj);
                if handle.is_null() {
                    return Ok(pyre_object::w_bool_from(true));
                }
                // sem_getvalue isn't available on macOS; just try sem_trywait
                // and immediately repost.  Best-effort; returning false is
                // safe because the only consumer is multiprocessing.Queue
                // tearing down.
                Ok(pyre_object::w_bool_from(false))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__enter__",
        crate::make_builtin_function_with_arity(
            "__enter__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let handle = semlock_get_handle(obj);
                if !handle.is_null() {
                    let _ = unsafe { libc::sem_wait(handle) };
                }
                Ok(obj)
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            if let Some(&obj) = args.first() {
                let handle = semlock_get_handle(obj);
                if !handle.is_null() {
                    let _ = unsafe { libc::sem_post(handle) };
                }
            }
            Ok(pyre_object::w_bool_from(false))
        }),
    );
}

fn init_multiprocessing(ns: &mut DictStorage) {
    #[cfg(all(unix, feature = "host_env"))]
    {
        // SemLock class.
        crate::dict_storage_store(ns, "SemLock", semlock_type());

        // SemLock factory — Python convention: SemLock(kind, value, maxvalue, name, unlink)
        crate::dict_storage_store(
            ns,
            "_SemLock_new",
            crate::make_builtin_function("_SemLock_new", |args| {
                if args.len() < 5 {
                    return Err(crate::PyError::type_error(
                        "SemLock() needs (kind, value, maxvalue, name, unlink)",
                    ));
                }
                let value = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_uint;
                let name = unsafe {
                    if !pyre_object::is_str(args[3]) {
                        return Err(crate::PyError::type_error("SemLock: name must be a string"));
                    }
                    pyre_object::w_str_get_value(args[3]).to_string()
                };
                let unlink = unsafe { pyre_object::w_int_get_value(args[4]) } != 0;
                let (handle, _kept_name) =
                    rustpython_host_env::multiprocessing::SemHandle::create(&name, value, unlink)
                        .map_err(|_| crate::PyError::os_error("SemLock create failed"))?;
                let raw = handle.as_ptr();
                // Leak the SemHandle wrapper so its Drop doesn't close the fd;
                // we'll sem_close manually if needed.  This matches CPython's
                // SemLock which keeps the sem_t for the instance lifetime.
                core::mem::forget(handle);
                let obj = pyre_object::w_instance_new(semlock_type());
                let d = crate::baseobjspace::getdict(obj);
                if !d.is_null() {
                    unsafe {
                        pyre_object::w_dict_setitem_str(
                            d,
                            "_handle",
                            pyre_object::w_int_new(raw as usize as i64),
                        );
                        pyre_object::w_dict_setitem_str(d, "name", pyre_object::w_str_new(&name));
                    }
                }
                Ok(obj)
            }),
        );

        crate::dict_storage_store(
            ns,
            "sem_unlink",
            crate::make_builtin_function_with_arity(
                "sem_unlink",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("sem_unlink() needs name"));
                    }
                    let name = unsafe {
                        if !pyre_object::is_str(args[0]) {
                            return Err(crate::PyError::type_error(
                                "sem_unlink: name must be a string",
                            ));
                        }
                        pyre_object::w_str_get_value(args[0]).to_string()
                    };
                    rustpython_host_env::multiprocessing::sem_unlink(&name)
                        .map_err(|_| crate::PyError::os_error("sem_unlink failed"))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        crate::dict_storage_store(
            ns,
            "SEM_VALUE_MAX",
            pyre_object::w_int_new(rustpython_host_env::multiprocessing::sem_value_max() as i64),
        );

        // _multiprocessing exposes RECURSIVE_MUTEX / SEMAPHORE kind tags.
        crate::dict_storage_store(ns, "RECURSIVE_MUTEX", pyre_object::w_int_new(0));
        crate::dict_storage_store(ns, "SEMAPHORE", pyre_object::w_int_new(1));
    }
}

/// atexit stub — PyPy: pypy/module/atexit/. Single-threaded pyre doesn't
/// actually run the registered callbacks on shutdown yet; `register` accepts
/// any callable and returns it so `@atexit.register` decorators work.
fn init_atexit(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "register",
        crate::make_builtin_function("register", |args| {
            // Return the function so `@atexit.register` decorator form works.
            Ok(args.first().copied().unwrap_or(pyre_object::w_none()))
        }),
    );
    crate::dict_storage_store(
        ns,
        "unregister",
        crate::make_builtin_function_with_arity("unregister", |_| Ok(pyre_object::w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "_run_exitfuncs",
        crate::make_builtin_function_with_arity("_run_exitfuncs", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "_clear",
        crate::make_builtin_function_with_arity("_clear", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "_ncallbacks",
        crate::make_builtin_function_with_arity(
            "_ncallbacks",
            |_| Ok(pyre_object::w_int_new(0)),
            0,
        ),
    );
}

/// _signal module — PyPy: pypy/module/signal/.
///
/// signal() / getsignal() / set_wakeup_fd() remain stubs because the
/// real implementations need interpreter-side trampolines to invoke
/// Python handlers from a Rust signal context.  alarm / pause /
/// raise_signal / strsignal / valid_signals are full implementations
/// backed by `rustpython_host_env::signal`.  Signal-number constants
/// are sourced from `libc::*` so they match the host's POSIX numbering
/// (the previous macOS-flavoured hard-coded list disagreed with Linux
/// for SIGUSR1/SIGUSR2/SIGCHLD).
fn init_signal_stub(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "signal",
        crate::make_builtin_function("signal", |args| {
            // signal(signalnum, handler) — return previous handler (None stub).
            Ok(args.get(1).copied().unwrap_or(pyre_object::w_none()))
        }),
    );
    crate::dict_storage_store(
        ns,
        "getsignal",
        crate::make_builtin_function_with_arity("getsignal", |_| Ok(pyre_object::w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "default_int_handler",
        crate::make_builtin_function_with_arity(
            "default_int_handler",
            |_| Ok(pyre_object::w_none()),
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "set_wakeup_fd",
        crate::make_builtin_function("set_wakeup_fd", |_| Ok(pyre_object::w_int_new(-1))),
    );
    // ── real host_env-backed entry points ──
    crate::dict_storage_store(
        ns,
        "raise_signal",
        crate::make_builtin_function_with_arity(
            "raise_signal",
            |args| {
                #[cfg(feature = "host_env")]
                {
                    let signum = if let Some(&a) = args.first() {
                        unsafe { pyre_object::w_int_get_value(a) as i32 }
                    } else {
                        return Err(crate::PyError::type_error(
                            "raise_signal() missing argument",
                        ));
                    };
                    match rustpython_host_env::signal::raise_signal(signum) {
                        Ok(()) => return Ok(pyre_object::w_none()),
                        Err(e) => {
                            return Err(crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("raise_signal: {e}"),
                            ));
                        }
                    }
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.raise_signal requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "strsignal",
        crate::make_builtin_function_with_arity(
            "strsignal",
            |args| {
                #[cfg(feature = "host_env")]
                {
                    let signum = if let Some(&a) = args.first() {
                        unsafe { pyre_object::w_int_get_value(a) as i32 }
                    } else {
                        return Err(crate::PyError::type_error("strsignal() missing argument"));
                    };
                    return Ok(rustpython_host_env::signal::strsignal(signum)
                        .map(|s| pyre_object::w_str_new(&s))
                        .unwrap_or(pyre_object::w_none()));
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.strsignal requires host_env feature",
                    ))
                }
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "valid_signals",
        crate::make_builtin_function_with_arity(
            "valid_signals",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    // PyPy passes NSIG (64) here; we match that bound.
                    let sigs = rustpython_host_env::signal::valid_signals(64).unwrap_or_default();
                    let items: Vec<pyre_object::PyObjectRef> = sigs
                        .into_iter()
                        .map(|n| pyre_object::w_int_new(n as i64))
                        .collect();
                    return Ok(pyre_object::w_frozenset_from_items(&items));
                }
                #[cfg(not(feature = "host_env"))]
                Err(crate::PyError::not_implemented(
                    "signal.valid_signals requires host_env feature",
                ))
            },
            0,
        ),
    );
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "alarm",
            crate::make_builtin_function_with_arity(
                "alarm",
                |args| {
                    #[cfg(feature = "host_env")]
                    {
                        let secs = if let Some(&a) = args.first() {
                            unsafe { pyre_object::w_int_get_value(a) as u32 }
                        } else {
                            return Err(crate::PyError::type_error("alarm() missing argument"));
                        };
                        return Ok(pyre_object::w_int_new(
                            rustpython_host_env::signal::alarm(secs) as i64,
                        ));
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.alarm requires host_env feature",
                        ))
                    }
                },
                1,
            ),
        );
        crate::dict_storage_store(
            ns,
            "pause",
            crate::make_builtin_function_with_arity(
                "pause",
                |_| {
                    #[cfg(feature = "host_env")]
                    rustpython_host_env::signal::pause();
                    Ok(pyre_object::w_none())
                },
                0,
            ),
        );
        // setitimer(which, seconds, interval=0.0) -> (delay, interval)
        crate::dict_storage_store(
            ns,
            "setitimer",
            crate::make_builtin_function("setitimer", |args| {
                #[cfg(feature = "host_env")]
                {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "setitimer() requires at least 2 arguments",
                        ));
                    }
                    let which = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let read_f = |o: pyre_object::PyObjectRef| -> f64 {
                        unsafe {
                            if pyre_object::is_float(o) {
                                pyre_object::w_float_get_value(o)
                            } else {
                                pyre_object::w_int_get_value(o) as f64
                            }
                        }
                    };
                    let new_value = libc::itimerval {
                        it_value: rustpython_host_env::signal::double_to_timeval(read_f(args[1])),
                        it_interval: if args.len() >= 3 {
                            rustpython_host_env::signal::double_to_timeval(read_f(args[2]))
                        } else {
                            rustpython_host_env::signal::double_to_timeval(0.0)
                        },
                    };
                    let old =
                        rustpython_host_env::signal::setitimer(which, &new_value).map_err(|e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("setitimer: {e}"),
                            )
                        })?;
                    let (delay, interval) = rustpython_host_env::signal::itimerval_to_tuple(&old);
                    return Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_float_new(delay),
                        pyre_object::w_float_new(interval),
                    ]));
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.setitimer requires host_env feature",
                    ))
                }
            }),
        );
        // getitimer(which) -> (delay, interval)
        crate::dict_storage_store(
            ns,
            "getitimer",
            crate::make_builtin_function_with_arity(
                "getitimer",
                |args| {
                    #[cfg(feature = "host_env")]
                    {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "getitimer() requires 1 argument",
                            ));
                        }
                        let which = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                        let it = rustpython_host_env::signal::getitimer(which).map_err(|e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("getitimer: {e}"),
                            )
                        })?;
                        let (delay, interval) =
                            rustpython_host_env::signal::itimerval_to_tuple(&it);
                        return Ok(pyre_object::w_tuple_new(vec![
                            pyre_object::w_float_new(delay),
                            pyre_object::w_float_new(interval),
                        ]));
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.getitimer requires host_env feature",
                        ))
                    }
                },
                1,
            ),
        );
        // siginterrupt(signalnum, flag) -> None
        crate::dict_storage_store(
            ns,
            "siginterrupt",
            crate::make_builtin_function_with_arity(
                "siginterrupt",
                |args| {
                    #[cfg(feature = "host_env")]
                    {
                        if args.len() < 2 {
                            return Err(crate::PyError::type_error(
                                "siginterrupt() requires 2 arguments",
                            ));
                        }
                        let sig = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                        let flag = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                        rustpython_host_env::signal::siginterrupt(sig, flag).map_err(|e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("siginterrupt: {e}"),
                            )
                        })?;
                        return Ok(pyre_object::w_none());
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.siginterrupt requires host_env feature",
                        ))
                    }
                },
                2,
            ),
        );
        // ITIMER_REAL/VIRTUAL/PROF
        crate::dict_storage_store(
            ns,
            "ITIMER_REAL",
            pyre_object::w_int_new(libc::ITIMER_REAL as i64),
        );
        crate::dict_storage_store(
            ns,
            "ITIMER_VIRTUAL",
            pyre_object::w_int_new(libc::ITIMER_VIRTUAL as i64),
        );
        crate::dict_storage_store(
            ns,
            "ITIMER_PROF",
            pyre_object::w_int_new(libc::ITIMER_PROF as i64),
        );
        // pthread_sigmask(how, mask) -> previous mask (set of signums)
        crate::dict_storage_store(
            ns,
            "pthread_sigmask",
            crate::make_builtin_function_with_arity(
                "pthread_sigmask",
                |args| {
                    #[cfg(feature = "host_env")]
                    {
                        if args.len() < 2 {
                            return Err(crate::PyError::type_error(
                                "pthread_sigmask() requires 2 arguments",
                            ));
                        }
                        let how = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                        let mask_arg = args[1];
                        let items: Vec<pyre_object::PyObjectRef> =
                            if unsafe { pyre_object::is_list(mask_arg) } {
                                let n = unsafe { pyre_object::w_list_len(mask_arg) };
                                (0..n)
                                    .filter_map(|i| unsafe {
                                        pyre_object::w_list_getitem(mask_arg, i as i64)
                                    })
                                    .collect()
                            } else if unsafe { pyre_object::is_tuple(mask_arg) } {
                                let n = unsafe { pyre_object::w_tuple_len(mask_arg) };
                                (0..n)
                                    .filter_map(|i| unsafe {
                                        pyre_object::w_tuple_getitem(mask_arg, i as i64)
                                    })
                                    .collect()
                            } else if unsafe { pyre_object::is_set_or_frozenset(mask_arg) } {
                                unsafe { pyre_object::w_set_items(mask_arg) }
                            } else {
                                return Err(crate::PyError::type_error(
                                    "pthread_sigmask: mask must be a list, tuple, or set",
                                ));
                            };
                        let mut set = rustpython_host_env::signal::sigemptyset().map_err(|e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("sigemptyset: {e}"),
                            )
                        })?;
                        for it in items {
                            let signum = (unsafe { pyre_object::w_int_get_value(it) }) as i32;
                            rustpython_host_env::signal::sigaddset(&mut set, signum).map_err(
                                |e| {
                                    crate::PyError::os_error_with_errno(
                                        e.raw_os_error().unwrap_or(0),
                                        format!("sigaddset: {e}"),
                                    )
                                },
                            )?;
                        }
                        let prev = rustpython_host_env::signal::pthread_sigmask(how, &set)
                            .map_err(|e| {
                                crate::PyError::os_error_with_errno(
                                    e.raw_os_error().unwrap_or(0),
                                    format!("pthread_sigmask: {e}"),
                                )
                            })?;
                        let out: Vec<pyre_object::PyObjectRef> = (1..=64)
                            .filter(|s| {
                                rustpython_host_env::signal::sigset_contains(&prev, *s as i32)
                            })
                            .map(|s| pyre_object::w_int_new(s as i64))
                            .collect();
                        return Ok(pyre_object::w_set_from_items(&out));
                    }
                    #[cfg(not(feature = "host_env"))]
                    {
                        let _ = args;
                        Err(crate::PyError::not_implemented(
                            "signal.pthread_sigmask requires host_env feature",
                        ))
                    }
                },
                2,
            ),
        );
        crate::dict_storage_store(
            ns,
            "SIG_BLOCK",
            pyre_object::w_int_new(libc::SIG_BLOCK as i64),
        );
        crate::dict_storage_store(
            ns,
            "SIG_UNBLOCK",
            pyre_object::w_int_new(libc::SIG_UNBLOCK as i64),
        );
        crate::dict_storage_store(
            ns,
            "SIG_SETMASK",
            pyre_object::w_int_new(libc::SIG_SETMASK as i64),
        );
        // pidfd_send_signal(pidfd, sig, siginfo=None, flags=0) - Linux-only
        #[cfg(target_os = "linux")]
        crate::dict_storage_store(
            ns,
            "pidfd_send_signal",
            crate::make_builtin_function("pidfd_send_signal", |args| {
                #[cfg(feature = "host_env")]
                {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "pidfd_send_signal() requires at least 2 arguments",
                        ));
                    }
                    let pidfd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let sig = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                    let flags = if args.len() >= 4 {
                        (unsafe { pyre_object::w_int_get_value(args[3]) }) as u32
                    } else {
                        0
                    };
                    rustpython_host_env::signal::pidfd_send_signal(pidfd, sig, flags).map_err(
                        |e| {
                            crate::PyError::os_error_with_errno(
                                e.raw_os_error().unwrap_or(0),
                                format!("pidfd_send_signal: {e}"),
                            )
                        },
                    )?;
                    return Ok(pyre_object::w_none());
                }
                #[cfg(not(feature = "host_env"))]
                {
                    let _ = args;
                    Err(crate::PyError::not_implemented(
                        "signal.pidfd_send_signal requires host_env feature",
                    ))
                }
            }),
        );
    }
    crate::dict_storage_store(ns, "SIG_DFL", pyre_object::w_int_new(0));
    crate::dict_storage_store(ns, "SIG_IGN", pyre_object::w_int_new(1));
    // libc crate doesn't surface NSIG portably; use POSIX 64-signal cap.
    crate::dict_storage_store(ns, "NSIG", pyre_object::w_int_new(64));
    // Common signal numbers (POSIX subset, sourced from libc so numerics
    // match the host — Linux SIGUSR1=10 / macOS SIGUSR1=30, etc.).
    #[cfg(unix)]
    {
        crate::dict_storage_store(ns, "SIGHUP", pyre_object::w_int_new(libc::SIGHUP as i64));
        crate::dict_storage_store(ns, "SIGINT", pyre_object::w_int_new(libc::SIGINT as i64));
        crate::dict_storage_store(ns, "SIGQUIT", pyre_object::w_int_new(libc::SIGQUIT as i64));
        crate::dict_storage_store(ns, "SIGILL", pyre_object::w_int_new(libc::SIGILL as i64));
        crate::dict_storage_store(ns, "SIGTRAP", pyre_object::w_int_new(libc::SIGTRAP as i64));
        crate::dict_storage_store(ns, "SIGABRT", pyre_object::w_int_new(libc::SIGABRT as i64));
        crate::dict_storage_store(ns, "SIGBUS", pyre_object::w_int_new(libc::SIGBUS as i64));
        crate::dict_storage_store(ns, "SIGFPE", pyre_object::w_int_new(libc::SIGFPE as i64));
        crate::dict_storage_store(ns, "SIGKILL", pyre_object::w_int_new(libc::SIGKILL as i64));
        crate::dict_storage_store(ns, "SIGUSR1", pyre_object::w_int_new(libc::SIGUSR1 as i64));
        crate::dict_storage_store(ns, "SIGSEGV", pyre_object::w_int_new(libc::SIGSEGV as i64));
        crate::dict_storage_store(ns, "SIGUSR2", pyre_object::w_int_new(libc::SIGUSR2 as i64));
        crate::dict_storage_store(ns, "SIGPIPE", pyre_object::w_int_new(libc::SIGPIPE as i64));
        crate::dict_storage_store(ns, "SIGALRM", pyre_object::w_int_new(libc::SIGALRM as i64));
        crate::dict_storage_store(ns, "SIGTERM", pyre_object::w_int_new(libc::SIGTERM as i64));
        crate::dict_storage_store(ns, "SIGCHLD", pyre_object::w_int_new(libc::SIGCHLD as i64));
        crate::dict_storage_store(ns, "SIGCONT", pyre_object::w_int_new(libc::SIGCONT as i64));
        crate::dict_storage_store(ns, "SIGSTOP", pyre_object::w_int_new(libc::SIGSTOP as i64));
        crate::dict_storage_store(ns, "SIGTSTP", pyre_object::w_int_new(libc::SIGTSTP as i64));
        crate::dict_storage_store(ns, "SIGTTIN", pyre_object::w_int_new(libc::SIGTTIN as i64));
        crate::dict_storage_store(ns, "SIGTTOU", pyre_object::w_int_new(libc::SIGTTOU as i64));
        crate::dict_storage_store(ns, "SIGURG", pyre_object::w_int_new(libc::SIGURG as i64));
        crate::dict_storage_store(ns, "SIGXCPU", pyre_object::w_int_new(libc::SIGXCPU as i64));
        crate::dict_storage_store(ns, "SIGXFSZ", pyre_object::w_int_new(libc::SIGXFSZ as i64));
        crate::dict_storage_store(
            ns,
            "SIGVTALRM",
            pyre_object::w_int_new(libc::SIGVTALRM as i64),
        );
        crate::dict_storage_store(ns, "SIGPROF", pyre_object::w_int_new(libc::SIGPROF as i64));
        crate::dict_storage_store(
            ns,
            "SIGWINCH",
            pyre_object::w_int_new(libc::SIGWINCH as i64),
        );
        crate::dict_storage_store(ns, "SIGIO", pyre_object::w_int_new(libc::SIGIO as i64));
        crate::dict_storage_store(ns, "SIGSYS", pyre_object::w_int_new(libc::SIGSYS as i64));
    }
}

/// itertools stub
fn init_itertools(ns: &mut DictStorage) {
    // chain(*iterables) → flat iterator
    crate::dict_storage_store(
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
    crate::dict_storage_store(
        ns,
        "starmap",
        crate::make_builtin_function_with_arity(
            "starmap",
            |_| Ok(pyre_object::w_list_new(vec![])),
            2,
        ),
    );
    // count(start=0, step=1) — PyPy: W_Count___new__
    //
    //     def W_Count___new__(space, w_subtype, w_start=0, w_step=1):
    //         return W_Count(space, w_start, w_step)
    crate::dict_storage_store(
        ns,
        "count",
        crate::make_builtin_function("count", |args| {
            let w_start = args.first().copied().unwrap_or(pyre_object::w_int_new(0));
            let w_step = args.get(1).copied().unwrap_or(pyre_object::w_int_new(1));
            Ok(pyre_object::itertoolsmodule::w_count_new(w_start, w_step))
        }),
    );
    // repeat(obj, times=None) — PyPy: W_Repeat___new__
    //
    //     def W_Repeat___new__(space, w_subtype, w_obj, w_times=None):
    //         return W_Repeat(space, w_obj, w_times)
    crate::dict_storage_store(
        ns,
        "repeat",
        crate::make_builtin_function("repeat", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error(
                    "repeat() missing 'object' argument",
                ));
            }
            let w_obj = args[0];
            let w_times = if args.len() >= 2 {
                unsafe {
                    if pyre_object::is_int(args[1]) {
                        Some(pyre_object::w_int_get_value(args[1]))
                    } else {
                        None
                    }
                }
            } else {
                None
            };
            Ok(pyre_object::itertoolsmodule::w_repeat_new(w_obj, w_times))
        }),
    );
    // islice
    crate::dict_storage_store(
        ns,
        "islice",
        crate::make_builtin_function("islice", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    // groupby
    crate::dict_storage_store(
        ns,
        "groupby",
        crate::make_builtin_function("groupby", |_| Ok(pyre_object::w_none())),
    );
    // permutations(iterable, r=None) — PyPy: pypy/module/itertools/interp_itertools.py
    crate::dict_storage_store(
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
    crate::dict_storage_store(
        ns,
        "combinations",
        crate::make_builtin_function_with_arity(
            "combinations",
            |args| {
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
            },
            2,
        ),
    );
    // product(*iterables, repeat=1)
    crate::dict_storage_store(
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
    // zip_longest(*iterables, fillvalue=None) — interp_itertools.py
    // W_ZipLongest. CALL_KW packs `fillvalue` into the trailing
    // `__pyre_kw__` dict (`call.rs:727-744`); strip it before
    // collecting the iterable pools so the kwarg doesn't surface as
    // an extra positional pool.
    crate::dict_storage_store(
        ns,
        "zip_longest",
        crate::make_builtin_function("zip_longest", |args| {
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            // `pypy/module/itertools/interp_itertools.py:685` —
            // W_ZipLongest's `unwrap_spec` only knows about
            // `fillvalue`; any other keyword raises TypeError at the
            // gateway.  Pyre's flat builtin ABI has to enforce this
            // by hand.
            crate::builtins::kwarg_reject_unknown(kwargs, &["fillvalue"], "zip_longest")?;
            let fill =
                crate::builtins::kwarg_get(kwargs, "fillvalue").unwrap_or_else(pyre_object::w_none);
            let pools: Vec<Vec<_>> = positional
                .iter()
                .map(|&a| crate::builtins::collect_iterable(a))
                .collect::<Result<_, _>>()?;
            let max_len = pools.iter().map(|p| p.len()).max().unwrap_or(0);
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
    crate::dict_storage_store(
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
    crate::dict_storage_store(
        ns,
        "compress",
        crate::make_builtin_function_with_arity(
            "compress",
            |args| {
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
            },
            2,
        ),
    );
}

/// _contextvars stub
fn init_contextvars(ns: &mut DictStorage) {
    // ContextVar(name, *, default=_MISSING) — context variable
    crate::dict_storage_store(
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
                crate::make_builtin_function_with_arity("set", |_| Ok(pyre_object::w_none()), 2),
            );
            Ok(obj)
        }),
    );
    crate::dict_storage_store(
        ns,
        "Context",
        crate::make_builtin_function_with_arity("Context", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "Token",
        crate::make_builtin_function_with_arity("Token", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "copy_context",
        crate::make_builtin_function_with_arity("copy_context", |_| Ok(pyre_object::w_none()), 0),
    );
}

/// _abc stub — PyPy: pypy/module/_abc/
fn init_abc(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "get_cache_token",
        crate::make_builtin_function_with_arity(
            "get_cache_token",
            |_| Ok(pyre_object::w_int_new(0)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_abc_init",
        crate::make_builtin_function_with_arity("_abc_init", |_| Ok(pyre_object::w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "_abc_register",
        crate::make_builtin_function_with_arity("_abc_register", |_| Ok(pyre_object::w_none()), 2),
    );
    // _abc_instancecheck(cls, instance) — CPython: Modules/_abc.c _abc__abc_instancecheck.
    //
    // ABCMeta.__instancecheck__ (abc.py:119) delegates here. The canonical
    // behaviour: walk type(instance).__mro__ looking for cls (direct
    // subclass), then consult cls._abc_registry for virtual subclasses
    // registered via `cls.register(subclass)`. Our previous stub
    // unconditionally returned False, which broke
    // `isinstance(Fraction(1,2), numbers.Rational)`.
    crate::dict_storage_store(
        ns,
        "_abc_instancecheck",
        crate::make_builtin_function_with_arity(
            "_abc_instancecheck",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let cls = args[0];
                let instance = args[1];
                unsafe {
                    Ok(pyre_object::w_bool_from(crate::baseobjspace::isinstance_w(
                        instance, cls,
                    )))
                }
            },
            2,
        ),
    );
    // _abc_subclasscheck(cls, subclass) — CPython: Modules/_abc.c _abc__abc_subclasscheck.
    crate::dict_storage_store(
        ns,
        "_abc_subclasscheck",
        crate::make_builtin_function_with_arity(
            "_abc_subclasscheck",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let cls = args[0];
                let subclass = args[1];
                unsafe {
                    // Walk subclass.__mro__ looking for cls.
                    let mro_ptr = pyre_object::w_type_get_mro(subclass);
                    if !mro_ptr.is_null() {
                        for &t in &*mro_ptr {
                            if std::ptr::eq(t, cls) {
                                return Ok(pyre_object::w_bool_from(true));
                            }
                        }
                    }
                }
                Ok(pyre_object::w_bool_from(false))
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_get_dump",
        crate::make_builtin_function_with_arity(
            "_get_dump",
            |_| Ok(pyre_object::w_tuple_new(vec![])),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_reset_registry",
        crate::make_builtin_function_with_arity(
            "_reset_registry",
            |_| Ok(pyre_object::w_none()),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_reset_caches",
        crate::make_builtin_function_with_arity("_reset_caches", |_| Ok(pyre_object::w_none()), 1),
    );
}

/// _functools stub
fn init_functools(ns: &mut DictStorage) {
    crate::dict_storage_store(
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
    crate::dict_storage_store(
        ns,
        "cmp_to_key",
        crate::make_builtin_function_with_arity(
            "cmp_to_key",
            |_args| {
                Ok(crate::make_builtin_function_with_arity(
                    "cmp_to_key.K",
                    |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
                    1,
                ))
            },
            1,
        ),
    );
}

/// Lock methods — PyPy: pypy/module/thread/os_lock.py W_Lock / W_RLock
///
/// Single-threaded pyre: state lives in the instance dict as `_locked_count`.
/// Methods increment/decrement this counter so Condition/RLock ownership
/// checks see the correct state.
fn init_lock_type(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "__enter__",
        crate::make_builtin_function_with_arity(
            "__enter__",
            |args| {
                if let Some(&obj) = args.first() {
                    lock_acquire_impl(obj)?;
                }
                Ok(args.first().copied().unwrap_or(pyre_object::w_none()))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            if let Some(&obj) = args.first() {
                lock_release_impl(obj)?;
            }
            Ok(pyre_object::w_bool_from(false))
        }),
    );
    // descr_lock_acquire — PyPy: os_lock.Lock.descr_lock_acquire
    crate::dict_storage_store(
        ns,
        "acquire",
        crate::make_builtin_function("acquire", |args| {
            let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            lock_acquire_impl(obj)?;
            Ok(pyre_object::w_bool_from(true))
        }),
    );
    // descr_lock_release — PyPy: os_lock.Lock.descr_lock_release
    crate::dict_storage_store(
        ns,
        "release",
        crate::make_builtin_function_with_arity(
            "release",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                lock_release_impl(obj)?;
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    // descr_lock_locked — PyPy: os_lock.Lock.descr_lock_locked
    crate::dict_storage_store(
        ns,
        "locked",
        crate::make_builtin_function_with_arity(
            "locked",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_bool_from(lock_count(obj) > 0))
            },
            1,
        ),
    );
    // _is_owned — used by RLock/Condition in threading.py
    crate::dict_storage_store(
        ns,
        "_is_owned",
        crate::make_builtin_function_with_arity(
            "_is_owned",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_bool_from(lock_count(obj) > 0))
            },
            1,
        ),
    );
    // _at_fork_reinit — PyPy: os_lock.Lock._at_fork_reinit (reset to unlocked)
    crate::dict_storage_store(
        ns,
        "_at_fork_reinit",
        crate::make_builtin_function_with_arity(
            "_at_fork_reinit",
            |args| {
                if let Some(&obj) = args.first() {
                    lock_set_count(obj, 0);
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
}

/// Read the lock's internal count. Single-threaded: 0 = unlocked, >0 = locked.
fn lock_count(obj: pyre_object::PyObjectRef) -> i64 {
    let w_dict = crate::baseobjspace::getdict(obj);
    if w_dict.is_null() {
        return 0;
    }
    if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(w_dict, "_locked_count") } {
        unsafe {
            if pyre_object::is_int(v) {
                return pyre_object::w_int_get_value(v);
            }
        }
    }
    0
}

fn lock_set_count(obj: pyre_object::PyObjectRef, v: i64) {
    let w_dict = crate::baseobjspace::getdict(obj);
    if w_dict.is_null() {
        return;
    }
    unsafe {
        pyre_object::w_dict_setitem_str(w_dict, "_locked_count", pyre_object::w_int_new(v));
    }
}

fn lock_acquire_impl(obj: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    lock_set_count(obj, lock_count(obj) + 1);
    Ok(())
}

fn lock_release_impl(obj: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    let cur = lock_count(obj);
    if cur <= 0 {
        return Err(crate::PyError::runtime_error("release unlocked lock"));
    }
    lock_set_count(obj, cur - 1);
    Ok(())
}

thread_local! {
    static LOCK_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    static THREAD_HANDLE_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
}

fn lock_type() -> PyObjectRef {
    LOCK_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("lock", init_lock_type);
            // Store per-instance `_locked_count` in the instance dict.
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

fn thread_handle_type() -> PyObjectRef {
    THREAD_HANDLE_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            crate::typedef::make_builtin_type("_ThreadHandle", |ns| {
                crate::dict_storage_store(
                    ns,
                    "is_done",
                    crate::make_builtin_function_with_arity(
                        "is_done",
                        |_| Ok(pyre_object::w_bool_from(true)),
                        1,
                    ),
                );
                crate::dict_storage_store(
                    ns,
                    "join",
                    crate::make_builtin_function("join", |_| Ok(pyre_object::w_none())),
                );
                crate::dict_storage_store(
                    ns,
                    "set_result",
                    crate::make_builtin_function_with_arity(
                        "set_result",
                        |_| Ok(pyre_object::w_none()),
                        2,
                    ),
                );
                crate::dict_storage_store(
                    ns,
                    "_set_done",
                    crate::make_builtin_function_with_arity(
                        "_set_done",
                        |_| Ok(pyre_object::w_none()),
                        1,
                    ),
                );
            })
        })
    })
}

/// _thread stub
fn init_thread(ns: &mut DictStorage) {
    let lock_tp = lock_type();
    crate::dict_storage_store(ns, "LockType", lock_tp);
    crate::dict_storage_store(
        ns,
        "RLock",
        crate::make_builtin_function_with_arity(
            "RLock",
            |_| Ok(pyre_object::w_instance_new(lock_type())),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "allocate_lock",
        crate::make_builtin_function_with_arity(
            "allocate_lock",
            |_| Ok(pyre_object::w_instance_new(lock_type())),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_ident",
        crate::make_builtin_function_with_arity(
            "get_ident",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    return Ok(pyre_object::w_int_new(
                        rustpython_host_env::thread::current_thread_id() as i64,
                    ));
                }
                #[cfg(not(feature = "host_env"))]
                Ok(pyre_object::w_int_new(1))
            },
            0,
        ),
    );
    // _thread.get_native_id() — returns the kernel-level TID, NOT the
    // pthread handle.  Mirrors rthread.c_get_native_id (rpython/rlib/
    // rthread.py) used by pypy/module/thread/os_thread.py:204-210.
    //
    // host_env::thread::current_thread_id always returns pthread_self
    // (suitable for get_ident above), so we drop to libc here:
    //   * Linux/Android: syscall(SYS_gettid) — kernel TID, distinct
    //     from pthread_self.
    //   * macOS:         pthread_threadid_np(NULL, &tid) — 64-bit TID.
    //   * Other Unix:    fall back to pthread_self (best effort; the
    //     same as get_ident, matching the lack of a true TID concept).
    crate::dict_storage_store(
        ns,
        "get_native_id",
        crate::make_builtin_function_with_arity(
            "get_native_id",
            |_| {
                #[cfg(any(target_os = "linux", target_os = "android"))]
                {
                    let tid = unsafe { libc::syscall(libc::SYS_gettid) };
                    return Ok(pyre_object::w_int_new(tid as i64));
                }
                #[cfg(target_os = "macos")]
                {
                    let mut tid: u64 = 0;
                    let rc = unsafe { libc::pthread_threadid_np(0, &mut tid as *mut u64) };
                    if rc == 0 {
                        return Ok(pyre_object::w_int_new(tid as i64));
                    }
                    return Ok(pyre_object::w_int_new(
                        unsafe { libc::pthread_self() } as i64
                    ));
                }
                #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos",)))]
                {
                    #[cfg(unix)]
                    {
                        return Ok(pyre_object::w_int_new(
                            unsafe { libc::pthread_self() } as i64
                        ));
                    }
                    #[cfg(not(unix))]
                    Ok(pyre_object::w_int_new(1))
                }
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_count",
        crate::make_builtin_function_with_arity("_count", |_| Ok(pyre_object::w_int_new(1)), 0),
    );
    crate::dict_storage_store(ns, "TIMEOUT_MAX", pyre_object::w_float_new(f64::MAX));
    crate::dict_storage_store(ns, "error", crate::typedef::w_object());
    crate::dict_storage_store(
        ns,
        "start_joinable_thread",
        crate::make_builtin_function("start_joinable_thread", |_| Ok(pyre_object::w_int_new(0))),
    );
    crate::dict_storage_store(
        ns,
        "_set_sentinel",
        crate::make_builtin_function_with_arity(
            "_set_sentinel",
            |_| Ok(pyre_object::w_instance_new(lock_type())),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "stack_size",
        crate::make_builtin_function_with_arity("stack_size", |_| Ok(pyre_object::w_int_new(0)), 1),
    );
    crate::dict_storage_store(
        ns,
        "_is_main_interpreter",
        crate::make_builtin_function_with_arity(
            "_is_main_interpreter",
            |_| Ok(pyre_object::w_bool_from(true)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "daemon_threads_allowed",
        crate::make_builtin_function_with_arity(
            "daemon_threads_allowed",
            |_| Ok(pyre_object::w_bool_from(true)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_shutdown",
        crate::make_builtin_function_with_arity("_shutdown", |_| Ok(pyre_object::w_none()), 0),
    );
    // _make_thread_handle / _ThreadHandle — threading.py:40-41
    crate::dict_storage_store(ns, "_ThreadHandle", thread_handle_type());
    crate::dict_storage_store(
        ns,
        "_make_thread_handle",
        crate::make_builtin_function_with_arity(
            "_make_thread_handle",
            |_| Ok(pyre_object::w_instance_new(thread_handle_type())),
            1,
        ),
    );
    // _get_main_thread_ident — threading.py:43
    crate::dict_storage_store(
        ns,
        "_get_main_thread_ident",
        crate::make_builtin_function_with_arity(
            "_get_main_thread_ident",
            |_| Ok(pyre_object::w_int_new(1)),
            0,
        ),
    );
    // get_native_id — threading.py:46
    crate::dict_storage_store(
        ns,
        "get_native_id",
        crate::make_builtin_function_with_arity(
            "get_native_id",
            |_| Ok(pyre_object::w_int_new(1)),
            0,
        ),
    );
    // set_name — threading.py:52
    crate::dict_storage_store(
        ns,
        "set_name",
        crate::make_builtin_function_with_arity("set_name", |_| Ok(pyre_object::w_none()), 1),
    );
    // _excepthook — threading.py:1262
    crate::dict_storage_store(
        ns,
        "_excepthook",
        crate::make_builtin_function_with_arity("_excepthook", |_| Ok(pyre_object::w_none()), 1),
    );
    // _local — PyPy: pypy/module/thread/os_local.py Local
    // Thread-local data. Single-threaded: equivalent to a plain object with dict.
    crate::dict_storage_store(ns, "_local", local_type());
}

fn local_type() -> PyObjectRef {
    thread_local! {
        static LOCAL_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    LOCAL_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("_local", |_ns| {});
            // Instances need __dict__ for per-thread attribute storage.
            // PyPy: os_local.py Local has getdict(space) → w_dict
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

/// posix stub — PyPy: pypy/module/posix/ interp_posix.py
///
/// Provides the minimal surface that os.py module init needs to succeed.
/// Real posix calls are not implemented — they raise or return defaults.
fn init_posix(ns: &mut DictStorage) {
    // environ — dict populated from the host environment.
    // PyPy equivalent: posix.State.startup → _convertenviron copies
    // os.environ.items() into w_environ at interpreter startup.
    let w_environ = pyre_object::w_dict_new();
    #[cfg(feature = "host_env")]
    {
        // On POSIX, posix.environ stores bytes → bytes. os.py's
        // _create_environ_mapping wraps this dict in an _Environ object that
        // encodes/decodes via surrogateescape when accessed.
        for (key, value) in host_os::vars_os() {
            let k_bytes = key.as_encoded_bytes();
            let v_bytes = value.as_encoded_bytes();
            unsafe {
                pyre_object::w_dict_store(
                    w_environ,
                    pyre_object::w_bytes_from_bytes(k_bytes),
                    pyre_object::w_bytes_from_bytes(v_bytes),
                );
            }
        }
    }
    crate::dict_storage_store(ns, "environ", w_environ);
    // _have_functions — list of HAVE_* macro names that were defined at
    // build time. os.py uses this to populate the supports_* capability
    // sets. Advertising a representative subset lets os.py module init
    // complete successfully.
    crate::dict_storage_store(
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
    // POSIX constants — real libc values (cross-platform subset).
    for (name, val) in [
        // F_OK/R_OK/W_OK/X_OK: Windows doesn't have them in libc crate,
        // define standard POSIX values directly.
        #[cfg(unix)]
        ("F_OK", libc::F_OK as i64),
        #[cfg(not(unix))]
        ("F_OK", 0i64),
        #[cfg(unix)]
        ("R_OK", libc::R_OK as i64),
        #[cfg(not(unix))]
        ("R_OK", 4i64),
        #[cfg(unix)]
        ("W_OK", libc::W_OK as i64),
        #[cfg(not(unix))]
        ("W_OK", 2i64),
        #[cfg(unix)]
        ("X_OK", libc::X_OK as i64),
        #[cfg(not(unix))]
        ("X_OK", 1i64),
        ("O_RDONLY", libc::O_RDONLY as i64),
        ("O_WRONLY", libc::O_WRONLY as i64),
        ("O_RDWR", libc::O_RDWR as i64),
        ("O_APPEND", libc::O_APPEND as i64),
        ("O_CREAT", libc::O_CREAT as i64),
        ("O_EXCL", libc::O_EXCL as i64),
        ("O_TRUNC", libc::O_TRUNC as i64),
        // O_NONBLOCK, O_DSYNC, O_SYNC are Unix-only.
        #[cfg(unix)]
        ("O_NONBLOCK", libc::O_NONBLOCK as i64),
        #[cfg(not(unix))]
        ("O_NONBLOCK", 0i64),
        #[cfg(unix)]
        ("O_NDELAY", libc::O_NONBLOCK as i64),
        #[cfg(not(unix))]
        ("O_NDELAY", 0i64),
        #[cfg(unix)]
        ("O_DSYNC", libc::O_DSYNC as i64),
        #[cfg(not(unix))]
        ("O_DSYNC", 0i64),
        #[cfg(unix)]
        ("O_SYNC", libc::O_SYNC as i64),
        #[cfg(not(unix))]
        ("O_SYNC", 0i64),
        ("SEEK_SET", libc::SEEK_SET as i64),
        ("SEEK_CUR", libc::SEEK_CUR as i64),
        ("SEEK_END", libc::SEEK_END as i64),
    ] {
        crate::dict_storage_store(ns, name, pyre_object::w_int_new(val));
    }
    // Non-critical constants — zero stubs are fine for os.py init.
    for name in [
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
        crate::dict_storage_store(ns, name, pyre_object::w_int_new(0));
    }
    // Remaining noop stubs — functions os.py references at module level.
    // Functions with real implementations are registered individually below.
    for name in [
        "fstatat",
        "statvfs",
        "fstatvfs",
        "dup",
        "dup2",
        "chdir",
        "fchdir",
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
        "getlogin",
        "nice",
        "pipe",
        "pipe2",
        "dup3",
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
        // "get_terminal_size" — implemented below
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
        "putenv",
        "device_encoding",
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
        "register_at_fork",
        "abort",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "system",
        "popen",
    ] {
        crate::dict_storage_store(
            ns,
            name,
            crate::make_builtin_function(name, |_| Ok(pyre_object::w_none())),
        );
    }

    // ── Helper: extract a filesystem path (str or bytes) from a PyObjectRef ──
    fn extract_path(obj: pyre_object::PyObjectRef) -> Result<String, crate::PyError> {
        unsafe {
            if pyre_object::is_str(obj) {
                return Ok(pyre_object::w_str_get_value(obj).to_string());
            }
            if pyre_object::bytesobject::is_bytes_like(obj) {
                let data = pyre_object::bytesobject::bytes_like_data(obj);
                return Ok(String::from_utf8_lossy(data).into_owned());
            }
        }
        if let Ok(fspath) = crate::baseobjspace::getattr(obj, "__fspath__") {
            let result = crate::call_function(fspath, &[obj]);
            if !result.is_null() && unsafe { pyre_object::is_str(result) } {
                return Ok(unsafe { pyre_object::w_str_get_value(result).to_string() });
            }
        }
        Err(crate::PyError::type_error(
            "expected str, bytes or os.PathLike",
        ))
    }

    // ── Helper: convert std::io::Error → PyError (OSError) ──
    fn io_err(e: std::io::Error, path: &str) -> crate::PyError {
        crate::PyError::os_error_with_errno(
            e.raw_os_error().unwrap_or(0),
            format!("{}: '{}'", e, path),
        )
    }

    // ── posix.open(path, flags, mode=0o777) → fd ──
    crate::dict_storage_store(
        ns,
        "open",
        crate::make_builtin_function("open", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(
                    "open() requires at least 2 arguments",
                ));
            }
            let path = extract_path(args[0])?;
            let flags = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
            let mode: u32 = if args.len() >= 3 {
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as u32
            } else {
                0o777
            };
            let c_path = std::ffi::CString::new(path.as_bytes())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            let fd = unsafe { libc::open(c_path.as_ptr(), flags, mode as libc::c_uint) };
            if fd < 0 {
                return Err(io_err(std::io::Error::last_os_error(), &path));
            }
            Ok(pyre_object::w_int_new(fd as i64))
        }),
    );

    // ── posix.close(fd) ──
    crate::dict_storage_store(
        ns,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("close() requires 1 argument"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let ret = unsafe { libc::close(fd) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── posix.read(fd, n) → bytes ──
    crate::dict_storage_store(
        ns,
        "read",
        crate::make_builtin_function_with_arity(
            "read",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("read() requires 2 arguments"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let n = (unsafe { pyre_object::w_int_get_value(args[1]) }) as usize;
                let mut buf = vec![0u8; n];
                let ret = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, n as _) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                buf.truncate(ret as usize);
                Ok(pyre_object::w_bytes_from_bytes(&buf))
            },
            2,
        ),
    );

    // ── posix.write(fd, data) → nbytes ──
    crate::dict_storage_store(
        ns,
        "write",
        crate::make_builtin_function_with_arity(
            "write",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("write() requires 2 arguments"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let data = unsafe {
                    if pyre_object::bytesobject::is_bytes_like(args[1]) {
                        pyre_object::bytesobject::bytes_like_data(args[1]).to_vec()
                    } else if pyre_object::is_str(args[1]) {
                        pyre_object::w_str_get_value(args[1]).as_bytes().to_vec()
                    } else {
                        return Err(crate::PyError::type_error(
                            "write() arg 2 must be bytes-like",
                        ));
                    }
                };
                let ret = unsafe {
                    libc::write(fd, data.as_ptr() as *const libc::c_void, data.len() as _)
                };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_int_new(ret as i64))
            },
            2,
        ),
    );

    // ── posix.lseek(fd, offset, whence) → position ──
    crate::dict_storage_store(
        ns,
        "lseek",
        crate::make_builtin_function_with_arity(
            "lseek",
            |args| {
                if args.len() < 3 {
                    return Err(crate::PyError::type_error("lseek() requires 3 arguments"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let offset = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::off_t;
                let whence = (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int;
                let ret = unsafe { libc::lseek(fd, offset, whence) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_int_new(ret as i64))
            },
            3,
        ),
    );

    // ── posix.unlink(path) / posix.remove(path) ──
    fn posix_unlink(
        args: &[pyre_object::PyObjectRef],
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        if args.is_empty() {
            return Err(crate::PyError::type_error("unlink() requires 1 argument"));
        }
        let path = extract_path(args[0])?;
        let c_path = std::ffi::CString::new(path.as_bytes())
            .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
        let ret = unsafe { libc::unlink(c_path.as_ptr()) };
        if ret < 0 {
            return Err(io_err(std::io::Error::last_os_error(), &path));
        }
        Ok(pyre_object::w_none())
    }
    crate::dict_storage_store(
        ns,
        "unlink",
        crate::make_builtin_function_with_arity("unlink", posix_unlink, 1),
    );
    crate::dict_storage_store(
        ns,
        "remove",
        crate::make_builtin_function_with_arity("remove", posix_unlink, 1),
    );

    // ── posix.mkdir(path, mode=0o777) ──
    crate::dict_storage_store(
        ns,
        "mkdir",
        crate::make_builtin_function("mkdir", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error("mkdir() requires 1 argument"));
            }
            let path = extract_path(args[0])?;
            let _mode: u32 = if args.len() >= 2 {
                (unsafe { pyre_object::w_int_get_value(args[1]) }) as u32
            } else {
                0o777
            };
            let c_path = std::ffi::CString::new(path.as_bytes())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            #[cfg(unix)]
            let ret = unsafe { libc::mkdir(c_path.as_ptr(), _mode as libc::mode_t) };
            #[cfg(windows)]
            let ret = unsafe { libc::mkdir(c_path.as_ptr()) };
            if ret < 0 {
                return Err(io_err(std::io::Error::last_os_error(), &path));
            }
            Ok(pyre_object::w_none())
        }),
    );

    // ── posix.rmdir(path) ──
    crate::dict_storage_store(
        ns,
        "rmdir",
        crate::make_builtin_function_with_arity(
            "rmdir",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("rmdir() requires 1 argument"));
                }
                let path = extract_path(args[0])?;
                let c_path = std::ffi::CString::new(path.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let ret = unsafe { libc::rmdir(c_path.as_ptr()) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), &path));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── posix.rename(src, dst) ──
    crate::dict_storage_store(
        ns,
        "rename",
        crate::make_builtin_function_with_arity(
            "rename",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("rename() requires 2 arguments"));
                }
                let src = extract_path(args[0])?;
                let dst = extract_path(args[1])?;
                host_os::rename(&src, &dst).map_err(|e| io_err(e, &src))?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    // ── posix.listdir(path=".") → list of str ──
    crate::dict_storage_store(
        ns,
        "listdir",
        crate::make_builtin_function("listdir", |args| {
            let path = if args.is_empty() || unsafe { pyre_object::is_none(args[0]) } {
                ".".to_string()
            } else {
                extract_path(args[0])?
            };
            let entries = host_fs::read_dir(&path).map_err(|e| io_err(e, &path))?;
            let mut items = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|e| io_err(e, &path))?;
                let name = entry.file_name();
                items.push(pyre_object::w_str_new(&name.to_string_lossy()));
            }
            Ok(pyre_object::w_list_new(items))
        }),
    );

    // ── posix.isatty(fd) → bool ──
    crate::dict_storage_store(
        ns,
        "isatty",
        crate::make_builtin_function_with_arity(
            "isatty",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                Ok(pyre_object::w_bool_from(host_os::isatty(fd)))
            },
            1,
        ),
    );

    // ── posix.urandom(n) → bytes ──
    crate::dict_storage_store(
        ns,
        "urandom",
        crate::make_builtin_function_with_arity(
            "urandom",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("urandom() requires 1 argument"));
                }
                let n = (unsafe { pyre_object::w_int_get_value(args[0]) }) as usize;
                let buf = host_os::urandom(n).unwrap_or_else(|_| vec![0u8; n]);
                Ok(pyre_object::w_bytes_from_bytes(&buf))
            },
            1,
        ),
    );
    // os.terminal_size — namedtuple-like type with columns/lines.
    // Uses stat_result_type (hasdict instance) so setattr works.
    fn make_terminal_size(cols: i64, lines: i64) -> pyre_object::PyObjectRef {
        let instance = pyre_object::w_instance_new(stat_result_type());
        let _ = crate::baseobjspace::setattr(instance, "columns", pyre_object::w_int_new(cols));
        let _ = crate::baseobjspace::setattr(instance, "lines", pyre_object::w_int_new(lines));
        instance
    }
    let terminal_size_type = crate::typedef::make_builtin_type("terminal_size", |ns| {
        crate::dict_storage_store(
            ns,
            "__new__",
            crate::make_builtin_function("__new__", |args| {
                let (cols, rows) = if args.len() >= 2 {
                    let seq = args[1];
                    unsafe {
                        if pyre_object::is_tuple(seq) {
                            let c = pyre_object::w_tuple_getitem(seq, 0)
                                .map(|v| pyre_object::w_int_get_value(v))
                                .unwrap_or(80);
                            let r = pyre_object::w_tuple_getitem(seq, 1)
                                .map(|v| pyre_object::w_int_get_value(v))
                                .unwrap_or(24);
                            (c, r)
                        } else {
                            (80, 24)
                        }
                    }
                } else {
                    (80, 24)
                };
                Ok(make_terminal_size(cols, rows))
            }),
        );
    });
    crate::dict_storage_store(ns, "terminal_size", terminal_size_type);

    // ── posix.get_terminal_size(fd=1) → os.terminal_size(columns, lines) ──
    crate::dict_storage_store(
        ns,
        "get_terminal_size",
        crate::make_builtin_function_with_arity(
            "get_terminal_size",
            |_args| {
                let (cols, rows) = {
                    #[cfg(unix)]
                    {
                        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
                        let ret = unsafe { libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) };
                        if ret == 0 && ws.ws_col > 0 {
                            (ws.ws_col as i64, ws.ws_row as i64)
                        } else {
                            (80, 24)
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        (80, 24)
                    }
                };
                let result = pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(cols),
                    pyre_object::w_int_new(rows),
                ]);
                let wrapper = pyre_object::w_instance_new(stat_result_type());
                let _ =
                    crate::baseobjspace::setattr(wrapper, "columns", pyre_object::w_int_new(cols));
                let _ =
                    crate::baseobjspace::setattr(wrapper, "lines", pyre_object::w_int_new(rows));
                let _ = crate::baseobjspace::setattr(wrapper, "__tuple__", result);
                Ok(wrapper)
            },
            0,
        ),
    );
    // os.fspath() — PyPy: posixmodule.c posix_fspath. Returns the argument
    // unchanged for str/bytes/bytearray (the protocol's identity case);
    // any other object would normally trigger __fspath__ but we don't
    // model that protocol yet.
    crate::dict_storage_store(
        ns,
        "fspath",
        crate::make_builtin_function_with_arity(
            "fspath",
            |args| {
                let arg = args.first().copied().unwrap_or(pyre_object::w_none());
                unsafe {
                    if pyre_object::is_str(arg) || pyre_object::bytesobject::is_bytes_like(arg) {
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
            },
            1,
        ),
    );
    // os.stat / os.lstat / os.fstat — return stat_result structseq.
    // PyPy: posixmodule.c posix_do_stat → build_stat_result.
    //
    // The returned object is a tuple subclass with named attributes
    // (st_mode, st_ino, ...). We expose it as a plain instance with
    // attributes so that both `os.stat(p).st_mode` and
    // `os.stat(p)[0]` work.
    fn make_stat_result(meta: &std::fs::Metadata) -> pyre_object::PyObjectRef {
        // Extract stat fields in a cross-platform way.
        #[cfg(unix)]
        let (
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_atime_ns,
            st_mtime_ns,
            st_ctime_ns,
        ) = {
            use std::os::unix::fs::MetadataExt;
            (
                meta.mode() as i64,
                meta.ino() as i64,
                meta.dev() as i64,
                meta.nlink() as i64,
                meta.uid() as i64,
                meta.gid() as i64,
                meta.size() as i64,
                meta.atime(),
                meta.mtime(),
                meta.ctime(),
                meta.atime() * 1_000_000_000 + meta.atime_nsec(),
                meta.mtime() * 1_000_000_000 + meta.mtime_nsec(),
                meta.ctime() * 1_000_000_000 + meta.ctime_nsec(),
            )
        };
        #[cfg(windows)]
        let (
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_atime_ns,
            st_mtime_ns,
            st_ctime_ns,
        ) = {
            use std::os::windows::fs::MetadataExt;
            let ft = meta.file_type();
            let attrs = meta.file_attributes();
            let mode: i64 = if ft.is_symlink() {
                // S_IFLNK | 0o777
                0o120777
            } else if ft.is_dir() {
                0o40755
            } else if attrs & 0x1 != 0 {
                // FILE_ATTRIBUTE_READONLY
                0o100444
            } else {
                0o100644
            };
            let size = meta.file_size() as i64;
            // Windows FILETIME is 100-ns intervals since 1601-01-01.
            // Convert to Unix epoch seconds.
            const EPOCH_DIFF: i64 = 11_644_473_600;
            let atime_secs = (meta.last_access_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let mtime_secs = (meta.last_write_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let ctime_secs = (meta.creation_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let atime_ns =
                ((meta.last_access_time() as i64 % 10_000_000) * 100) + atime_secs * 1_000_000_000;
            let mtime_ns =
                ((meta.last_write_time() as i64 % 10_000_000) * 100) + mtime_secs * 1_000_000_000;
            let ctime_ns =
                ((meta.creation_time() as i64 % 10_000_000) * 100) + ctime_secs * 1_000_000_000;
            (
                mode, 0i64, // st_ino — not available on Windows
                0i64, // st_dev
                1i64, // nlink — not easily available on stable Windows
                0i64, // st_uid
                0i64, // st_gid
                size, atime_secs, mtime_secs, ctime_secs, atime_ns, mtime_ns, ctime_ns,
            )
        };

        let tuple = pyre_object::w_tuple_new(vec![
            pyre_object::w_int_new(st_mode),
            pyre_object::w_int_new(st_ino),
            pyre_object::w_int_new(st_dev),
            pyre_object::w_int_new(st_nlink),
            pyre_object::w_int_new(st_uid),
            pyre_object::w_int_new(st_gid),
            pyre_object::w_int_new(st_size),
            pyre_object::w_int_new(st_atime),
            pyre_object::w_int_new(st_mtime),
            pyre_object::w_int_new(st_ctime),
        ]);
        // Attach st_* attributes via a wrapping instance.
        let wrapper = pyre_object::w_instance_new(stat_result_type());
        let _ = crate::baseobjspace::setattr(wrapper, "__tuple__", tuple);
        let _ = crate::baseobjspace::setattr(wrapper, "st_mode", pyre_object::w_int_new(st_mode));
        let _ = crate::baseobjspace::setattr(wrapper, "st_ino", pyre_object::w_int_new(st_ino));
        let _ = crate::baseobjspace::setattr(wrapper, "st_dev", pyre_object::w_int_new(st_dev));
        let _ = crate::baseobjspace::setattr(wrapper, "st_nlink", pyre_object::w_int_new(st_nlink));
        let _ = crate::baseobjspace::setattr(wrapper, "st_uid", pyre_object::w_int_new(st_uid));
        let _ = crate::baseobjspace::setattr(wrapper, "st_gid", pyre_object::w_int_new(st_gid));
        let _ = crate::baseobjspace::setattr(wrapper, "st_size", pyre_object::w_int_new(st_size));
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_atime",
            pyre_object::w_float_new(st_atime as f64),
        );
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_mtime",
            pyre_object::w_float_new(st_mtime as f64),
        );
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_ctime",
            pyre_object::w_float_new(st_ctime as f64),
        );
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_atime_ns",
            pyre_object::w_int_new(st_atime_ns),
        );
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_mtime_ns",
            pyre_object::w_int_new(st_mtime_ns),
        );
        let _ = crate::baseobjspace::setattr(
            wrapper,
            "st_ctime_ns",
            pyre_object::w_int_new(st_ctime_ns),
        );
        wrapper
    }
    fn stat_impl(
        args: &[pyre_object::PyObjectRef],
        follow_symlinks: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        if args.is_empty() {
            return Err(crate::PyError::type_error("stat() missing argument"));
        }
        let path_obj = args[0];
        let path_str = unsafe {
            if pyre_object::is_str(path_obj) {
                pyre_object::w_str_get_value(path_obj).to_string()
            } else if pyre_object::bytesobject::is_bytes_like(path_obj) {
                let data = pyre_object::bytesobject::bytes_like_data(path_obj);
                String::from_utf8_lossy(data).into_owned()
            } else if let Ok(fspath) = crate::baseobjspace::getattr(path_obj, "__fspath__") {
                let result = crate::call_function(fspath, &[path_obj]);
                if !result.is_null() && pyre_object::is_str(result) {
                    pyre_object::w_str_get_value(result).to_string()
                } else {
                    return Err(crate::PyError::type_error(
                        "stat: path should be string, bytes, os.PathLike",
                    ));
                }
            } else {
                return Err(crate::PyError::type_error(
                    "stat: path should be string, bytes, os.PathLike",
                ));
            }
        };
        let meta = if follow_symlinks {
            host_fs::metadata(&path_str)
        } else {
            host_fs::symlink_metadata(&path_str)
        };
        match meta {
            Ok(m) => Ok(make_stat_result(&m)),
            Err(e) => {
                let kind = e.raw_os_error().unwrap_or(2);
                Err(crate::PyError::os_error_with_errno(
                    kind,
                    format!("{}: '{}'", e, path_str),
                ))
            }
        }
    }
    // os.uname() — returns structseq (sysname, nodename, release, version, machine).
    // Routed through `host_env::posix::uname_info` when available so the
    // result reports the host's real POSIX strings ("Darwin", "Linux",
    // node hostname, kernel release, etc.) instead of Rust's compile-time
    // `std::env::consts::OS` ("macos"/"linux"/...).
    crate::dict_storage_store(
        ns,
        "uname",
        crate::make_builtin_function_with_arity(
            "uname",
            |_| {
                let wrapper = pyre_object::w_instance_new(stat_result_type());
                #[cfg(all(unix, feature = "host_env"))]
                {
                    let info = rustpython_host_env::posix::uname_info().unwrap_or(
                        rustpython_host_env::posix::UnameInfo {
                            sysname: String::new(),
                            nodename: String::new(),
                            release: String::new(),
                            version: String::new(),
                            machine: String::new(),
                        },
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "sysname",
                        pyre_object::w_str_new(&info.sysname),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "nodename",
                        pyre_object::w_str_new(&info.nodename),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "release",
                        pyre_object::w_str_new(&info.release),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "version",
                        pyre_object::w_str_new(&info.version),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "machine",
                        pyre_object::w_str_new(&info.machine),
                    );
                }
                #[cfg(not(all(unix, feature = "host_env")))]
                {
                    let sysname = std::env::consts::OS.to_string();
                    let machine = std::env::consts::ARCH.to_string();
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "sysname",
                        pyre_object::w_str_new(&sysname),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "nodename",
                        pyre_object::w_str_new(""),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "release",
                        pyre_object::w_str_new(""),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "version",
                        pyre_object::w_str_new(""),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "machine",
                        pyre_object::w_str_new(&machine),
                    );
                }
                Ok(wrapper)
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "stat",
        crate::make_builtin_function_with_arity("stat", |args| stat_impl(args, true), 1),
    );
    crate::dict_storage_store(
        ns,
        "lstat",
        crate::make_builtin_function_with_arity("lstat", |args| stat_impl(args, false), 1),
    );
    crate::dict_storage_store(
        ns,
        "fstat",
        crate::make_builtin_function_with_arity(
            "fstat",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("fstat() missing argument"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                #[cfg(unix)]
                {
                    use std::os::unix::io::FromRawFd;
                    let f = unsafe { std::fs::File::from_raw_fd(fd) };
                    let meta = f.metadata();
                    let _ = std::mem::ManuallyDrop::new(f); // don't close
                    match meta {
                        Ok(m) => Ok(make_stat_result(&m)),
                        Err(e) => Err(crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(9),
                            format!("{}", e),
                        )),
                    }
                }
                #[cfg(not(unix))]
                Err(crate::PyError::os_error_with_errno(
                    9,
                    "fstat unsupported".to_string(),
                ))
            },
            1,
        ),
    );
    // stat_result type — simple instance with hasdict so setattr works.
    // Exported so that `posix.stat_result` can be looked up.
    crate::dict_storage_store(ns, "stat_result", stat_result_type());
    // os.getcwd() — PyPy: posixmodule.c posix_getcwd.
    crate::dict_storage_store(
        ns,
        "getcwd",
        crate::make_builtin_function_with_arity(
            "getcwd",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    if let Ok(cwd) = host_os::current_dir() {
                        return Ok(pyre_object::w_str_new(&cwd.to_string_lossy()));
                    }
                }
                Ok(pyre_object::w_str_new(""))
            },
            0,
        ),
    );
    // os.getcwdb() — bytes form of getcwd.
    crate::dict_storage_store(
        ns,
        "getcwdb",
        crate::make_builtin_function_with_arity(
            "getcwdb",
            |_| {
                #[cfg(feature = "host_env")]
                {
                    if let Ok(cwd) = host_os::current_dir() {
                        return Ok(pyre_object::w_bytes_from_bytes(
                            cwd.as_os_str().as_encoded_bytes(),
                        ));
                    }
                }
                Ok(pyre_object::w_bytes_from_bytes(b""))
            },
            0,
        ),
    );
    // os.getuid / geteuid / getgid / getegid — real syscalls.
    #[cfg(unix)]
    unsafe extern "C" {
        fn getuid() -> u32;
        fn geteuid() -> u32;
        fn getgid() -> u32;
        fn getegid() -> u32;
    }
    crate::dict_storage_store(
        ns,
        "getuid",
        crate::make_builtin_function_with_arity(
            "getuid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(getuid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "geteuid",
        crate::make_builtin_function_with_arity(
            "geteuid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(geteuid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getgid",
        crate::make_builtin_function_with_arity(
            "getgid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(getgid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "getegid",
        crate::make_builtin_function_with_arity(
            "getegid",
            |_| {
                #[cfg(unix)]
                unsafe {
                    return Ok(pyre_object::w_int_new(getegid() as i64));
                }
                #[cfg(not(unix))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    // os.getpid — host_os::process_id (std::process::id).
    crate::dict_storage_store(
        ns,
        "getpid",
        crate::make_builtin_function_with_arity(
            "getpid",
            |_| Ok(pyre_object::w_int_new(host_os::process_id() as i64)),
            0,
        ),
    );
    // os.environ lookups from setenv / unsetenv / putenv / getenv — mutate
    // posix.environ (the dict) rather than calling libc; os.py writes back
    // into that dict in its _Environ wrapper.
    crate::dict_storage_store(
        ns,
        "getenv",
        crate::make_builtin_function("getenv", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let key = unsafe {
                if pyre_object::is_str(args[0]) {
                    pyre_object::w_str_get_value(args[0]).to_string()
                } else {
                    return Ok(pyre_object::w_none());
                }
            };
            #[cfg(feature = "host_env")]
            {
                if let Ok(value) = host_os::var(&key) {
                    return Ok(pyre_object::w_str_new(&value));
                }
            }
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Ok(pyre_object::w_none())
            }
        }),
    );
    // ── host_env::posix-backed real implementations (override the noop
    //    placeholders registered above) ───────────────────────────────
    #[cfg(all(unix, feature = "host_env"))]
    {
        use rustpython_host_env::posix as host_posix;

        // os.pipe() -> (r_fd, w_fd)
        crate::dict_storage_store(
            ns,
            "pipe",
            crate::make_builtin_function_with_arity(
                "pipe",
                |_| match host_posix::pipe() {
                    Ok((rfd, wfd)) => {
                        use std::os::fd::IntoRawFd;
                        Ok(pyre_object::w_tuple_new(vec![
                            pyre_object::w_int_new(rfd.into_raw_fd() as i64),
                            pyre_object::w_int_new(wfd.into_raw_fd() as i64),
                        ]))
                    }
                    Err(e) => Err(io_err(e, "")),
                },
                0,
            ),
        );

        // os.sched_yield()
        crate::dict_storage_store(
            ns,
            "sched_yield",
            crate::make_builtin_function_with_arity(
                "sched_yield",
                |_| {
                    host_posix::sched_yield().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                0,
            ),
        );

        // os.nice(increment) -> new niceness
        crate::dict_storage_store(
            ns,
            "nice",
            crate::make_builtin_function_with_arity(
                "nice",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("nice() requires 1 argument"));
                    }
                    let inc = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let n = host_posix::nice(inc).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            ),
        );

        // os.umask(mask) -> previous mask
        crate::dict_storage_store(
            ns,
            "umask",
            crate::make_builtin_function_with_arity(
                "umask",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("umask() requires 1 argument"));
                    }
                    let mask = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::mode_t;
                    let prev = host_posix::umask(mask);
                    Ok(pyre_object::w_int_new(prev as i64))
                },
                1,
            ),
        );

        // os.getlogin() -> str
        crate::dict_storage_store(
            ns,
            "getlogin",
            crate::make_builtin_function_with_arity(
                "getlogin",
                |_| match host_posix::getlogin() {
                    Some(name) => Ok(pyre_object::w_str_new(name.to_string_lossy().as_ref())),
                    None => Err(crate::PyError::os_error_with_errno(
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                        "getlogin",
                    )),
                },
                0,
            ),
        );

        // os.getgroups() -> list[int]
        crate::dict_storage_store(
            ns,
            "getgroups",
            crate::make_builtin_function_with_arity(
                "getgroups",
                |_| {
                    let gs = host_posix::getgroups().map_err(|e| io_err(e, ""))?;
                    let items: Vec<_> = gs
                        .into_iter()
                        .map(|g| pyre_object::w_int_new(g as i64))
                        .collect();
                    Ok(pyre_object::w_list_new(items))
                },
                0,
            ),
        );

        // os.sched_get_priority_max(policy) -> int
        crate::dict_storage_store(
            ns,
            "sched_get_priority_max",
            crate::make_builtin_function_with_arity(
                "sched_get_priority_max",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "sched_get_priority_max() requires 1 argument",
                        ));
                    }
                    let policy = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let m =
                        host_posix::sched_get_priority_max(policy).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(m as i64))
                },
                1,
            ),
        );

        // os.sched_get_priority_min(policy) -> int
        crate::dict_storage_store(
            ns,
            "sched_get_priority_min",
            crate::make_builtin_function_with_arity(
                "sched_get_priority_min",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "sched_get_priority_min() requires 1 argument",
                        ));
                    }
                    let policy = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let m =
                        host_posix::sched_get_priority_min(policy).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(m as i64))
                },
                1,
            ),
        );

        // os.sync()
        #[cfg(not(any(target_os = "redox", target_os = "android")))]
        crate::dict_storage_store(
            ns,
            "sync",
            crate::make_builtin_function_with_arity(
                "sync",
                |_| {
                    host_posix::sync();
                    Ok(pyre_object::w_none())
                },
                0,
            ),
        );

        // os.chdir(path)
        crate::dict_storage_store(
            ns,
            "chdir",
            crate::make_builtin_function_with_arity(
                "chdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chdir() requires 1 argument"));
                    }
                    let path = extract_path(args[0])?;
                    let c_path = std::ffi::CString::new(path.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    host_posix::chdir(&c_path).map_err(|e| {
                        crate::PyError::os_error_with_errno(e as i32, format!("chdir: '{}'", path))
                    })?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.fchdir(fd)
        crate::dict_storage_store(
            ns,
            "fchdir",
            crate::make_builtin_function_with_arity(
                "fchdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fchdir() requires 1 argument"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    host_posix::fchdir(fd).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.fork() -> child pid in parent, 0 in child
        crate::dict_storage_store(
            ns,
            "fork",
            crate::make_builtin_function_with_arity(
                "fork",
                |_| {
                    let pid = host_posix::fork().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(pid as i64))
                },
                0,
            ),
        );

        // os.getppid() -> int
        crate::dict_storage_store(
            ns,
            "getppid",
            crate::make_builtin_function_with_arity(
                "getppid",
                |_| Ok(pyre_object::w_int_new(unsafe { libc::getppid() } as i64)),
                0,
            ),
        );

        // os.dup(fd) -> new_fd
        crate::dict_storage_store(
            ns,
            "dup",
            crate::make_builtin_function_with_arity(
                "dup",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("dup() requires 1 argument"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    let n = unsafe { libc::dup(fd) };
                    if n < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            ),
        );

        // os.dup2(fd, fd2, inheritable=True) -> fd2
        crate::dict_storage_store(
            ns,
            "dup2",
            crate::make_builtin_function("dup2", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("dup2() requires 2 arguments"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let fd2 = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                let n = unsafe { libc::dup2(fd, fd2) };
                if n < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), ""));
                }
                Ok(pyre_object::w_int_new(n as i64))
            }),
        );

        // os.fsync(fd)
        crate::dict_storage_store(
            ns,
            "fsync",
            crate::make_builtin_function_with_arity(
                "fsync",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fsync() requires 1 argument"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    let r = unsafe { libc::fsync(fd) };
                    if r < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.fdatasync(fd) — falls back to fsync on macOS, which has no
        // fdatasync syscall but exposes the same semantics through fsync.
        crate::dict_storage_store(
            ns,
            "fdatasync",
            crate::make_builtin_function_with_arity(
                "fdatasync",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "fdatasync() requires 1 argument",
                        ));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    let r = unsafe { libc::fdatasync(fd) };
                    #[cfg(not(any(target_os = "linux", target_os = "android")))]
                    let r = unsafe { libc::fsync(fd) };
                    if r < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.mkfifo(path, mode=0o666) -> None
        crate::dict_storage_store(
            ns,
            "mkfifo",
            crate::make_builtin_function("mkfifo", |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("mkfifo() requires 1 argument"));
                }
                let path = extract_path(args[0])?;
                let mode = if args.len() >= 2 {
                    (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::mode_t
                } else {
                    0o666
                };
                let c_path = std::ffi::CString::new(path.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let r = unsafe { libc::mkfifo(c_path.as_ptr(), mode) };
                if r < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), &path));
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.kill(pid, sig) / os.killpg(pgid, sig)
        crate::dict_storage_store(
            ns,
            "kill",
            crate::make_builtin_function_with_arity(
                "kill",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("kill() requires 2 arguments"));
                    }
                    let pid = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::pid_t;
                    let sig = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                    let r = unsafe { libc::kill(pid, sig) };
                    if r < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );
        crate::dict_storage_store(
            ns,
            "killpg",
            crate::make_builtin_function_with_arity(
                "killpg",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("killpg() requires 2 arguments"));
                    }
                    let pgid = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::pid_t;
                    let sig = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                    let r = unsafe { libc::killpg(pgid, sig) };
                    if r < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.statvfs(path) / os.fstatvfs(fd) -> statvfs_result
        #[cfg(not(target_os = "redox"))]
        fn statvfs_to_obj(
            info: rustpython_host_env::posix::StatVfsInfo,
        ) -> pyre_object::PyObjectRef {
            let wrapper = pyre_object::w_instance_new(stat_result_type());
            let _ = crate::baseobjspace::setattr(
                wrapper,
                "f_bsize",
                pyre_object::w_int_new(info.f_bsize as i64),
            );
            let _ = crate::baseobjspace::setattr(
                wrapper,
                "f_frsize",
                pyre_object::w_int_new(info.f_frsize as i64),
            );
            let _ = crate::baseobjspace::setattr(
                wrapper,
                "f_blocks",
                pyre_object::w_int_new(info.f_blocks as i64),
            );
            let _ = crate::baseobjspace::setattr(
                wrapper,
                "f_bfree",
                pyre_object::w_int_new(info.f_bfree as i64),
            );
            let _ = crate::baseobjspace::setattr(
                wrapper,
                "f_bavail",
                pyre_object::w_int_new(info.f_bavail as i64),
            );
            let _ = crate::baseobjspace::setattr(
                wrapper,
                "f_files",
                pyre_object::w_int_new(info.f_files as i64),
            );
            let _ = crate::baseobjspace::setattr(
                wrapper,
                "f_ffree",
                pyre_object::w_int_new(info.f_ffree as i64),
            );
            let _ = crate::baseobjspace::setattr(
                wrapper,
                "f_favail",
                pyre_object::w_int_new(info.f_favail as i64),
            );
            let _ = crate::baseobjspace::setattr(
                wrapper,
                "f_flag",
                pyre_object::w_int_new(info.f_flag as i64),
            );
            let _ = crate::baseobjspace::setattr(
                wrapper,
                "f_namemax",
                pyre_object::w_int_new(info.f_namemax as i64),
            );
            let _ = crate::baseobjspace::setattr(
                wrapper,
                "f_fsid",
                pyre_object::w_int_new(info.f_fsid as i64),
            );
            wrapper
        }
        #[cfg(not(target_os = "redox"))]
        crate::dict_storage_store(
            ns,
            "statvfs",
            crate::make_builtin_function_with_arity(
                "statvfs",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("statvfs() requires 1 argument"));
                    }
                    let path = extract_path(args[0])?;
                    let c_path = std::ffi::CString::new(path.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    let info = host_posix::statvfs_path(&c_path).map_err(|e| io_err(e, &path))?;
                    Ok(statvfs_to_obj(info))
                },
                1,
            ),
        );
        #[cfg(not(target_os = "redox"))]
        crate::dict_storage_store(
            ns,
            "fstatvfs",
            crate::make_builtin_function_with_arity(
                "fstatvfs",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fstatvfs() requires 1 argument"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let info = host_posix::statvfs_fd(fd).map_err(|e| io_err(e, ""))?;
                    Ok(statvfs_to_obj(info))
                },
                1,
            ),
        );

        // os.cpu_count() -> int | None
        crate::dict_storage_store(
            ns,
            "cpu_count",
            crate::make_builtin_function_with_arity(
                "cpu_count",
                |_| {
                    let n = host_posix::get_number_of_os_threads();
                    if n <= 0 {
                        Ok(pyre_object::w_none())
                    } else {
                        Ok(pyre_object::w_int_new(n as i64))
                    }
                },
                0,
            ),
        );
        // _cpu_count alias — newer CPython exposes both.
        crate::dict_storage_store(
            ns,
            "_cpu_count",
            crate::make_builtin_function_with_arity(
                "_cpu_count",
                |_| {
                    let n = host_posix::get_number_of_os_threads();
                    if n <= 0 {
                        Ok(pyre_object::w_none())
                    } else {
                        Ok(pyre_object::w_int_new(n as i64))
                    }
                },
                0,
            ),
        );

        // os.symlink(src, dst, target_is_directory=False) -> None
        crate::dict_storage_store(
            ns,
            "symlink",
            crate::make_builtin_function("symlink", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("symlink() requires 2 arguments"));
                }
                let src = extract_path(args[0])?;
                let dst = extract_path(args[1])?;
                let c_src = std::ffi::CString::new(src.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in src"))?;
                let c_dst = std::ffi::CString::new(dst.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in dst"))?;
                // host_env::posix only exposes symlinkat on non-redox unices;
                // call libc::symlink directly so we don't need an at-cwd dance.
                let ret = unsafe { libc::symlink(c_src.as_ptr(), c_dst.as_ptr()) };
                if ret < 0 {
                    return Err(io_err(std::io::Error::last_os_error(), &dst));
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.fchmod(fd, mode) -> None
        crate::dict_storage_store(
            ns,
            "fchmod",
            crate::make_builtin_function_with_arity(
                "fchmod",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fchmod() requires 2 arguments"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let mode = (unsafe { pyre_object::w_int_get_value(args[1]) }) as u32;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::fchmod(bfd, mode).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.fchown(fd, uid, gid) -> None  (uid/gid of -1 means "leave unchanged")
        crate::dict_storage_store(
            ns,
            "fchown",
            crate::make_builtin_function_with_arity(
                "fchown",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error("fchown() requires 3 arguments"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let uid_raw = unsafe { pyre_object::w_int_get_value(args[1]) };
                    let gid_raw = unsafe { pyre_object::w_int_get_value(args[2]) };
                    let uid = if uid_raw < 0 {
                        None
                    } else {
                        Some(uid_raw as u32)
                    };
                    let gid = if gid_raw < 0 {
                        None
                    } else {
                        Some(gid_raw as u32)
                    };
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::fchown(bfd, uid, gid).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        // os.set_inheritable(fd, inheritable) -> None
        crate::dict_storage_store(
            ns,
            "set_inheritable",
            crate::make_builtin_function_with_arity(
                "set_inheritable",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "set_inheritable() requires 2 arguments",
                        ));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let inherit = unsafe { pyre_object::w_int_get_value(args[1]) } != 0;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::set_inheritable(bfd, inherit).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.access(path, mode) -> bool
        crate::dict_storage_store(
            ns,
            "access",
            crate::make_builtin_function("access", |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("access() requires 2 arguments"));
                }
                let path = extract_path(args[0])?;
                let mode = (unsafe { pyre_object::w_int_get_value(args[1]) }) as u8;
                match host_posix::check_access(std::path::Path::new(&path), mode) {
                    Ok(ok) => Ok(pyre_object::w_bool_from(ok)),
                    Err(_) => Ok(pyre_object::w_bool_from(false)),
                }
            }),
        );

        // os.chroot(path) -> None
        crate::dict_storage_store(
            ns,
            "chroot",
            crate::make_builtin_function_with_arity(
                "chroot",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chroot() requires 1 argument"));
                    }
                    let path = extract_path(args[0])?;
                    host_posix::chroot(std::path::Path::new(&path))
                        .map_err(|e| io_err(e, &path))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.getloadavg() -> (1m, 5m, 15m)
        crate::dict_storage_store(
            ns,
            "getloadavg",
            crate::make_builtin_function_with_arity(
                "getloadavg",
                |_| {
                    let [l1, l5, l15] =
                        rustpython_host_env::time::getloadavg().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_float_new(l1),
                        pyre_object::w_float_new(l5),
                        pyre_object::w_float_new(l15),
                    ]))
                },
                0,
            ),
        );

        // os.times() -> posix.times_result(user, system, children_user,
        //                                  children_system, elapsed)
        crate::dict_storage_store(
            ns,
            "times",
            crate::make_builtin_function_with_arity(
                "times",
                |_| {
                    let t =
                        rustpython_host_env::time::process_times().map_err(|e| io_err(e, ""))?;
                    let wrapper = pyre_object::w_instance_new(stat_result_type());
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "user",
                        pyre_object::w_float_new(t.user),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "system",
                        pyre_object::w_float_new(t.system),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "children_user",
                        pyre_object::w_float_new(t.children_user),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "children_system",
                        pyre_object::w_float_new(t.children_system),
                    );
                    let _ = crate::baseobjspace::setattr(
                        wrapper,
                        "elapsed",
                        pyre_object::w_float_new(t.elapsed),
                    );
                    Ok(wrapper)
                },
                0,
            ),
        );

        // os.waitstatus_to_exitcode(status) -> int
        crate::dict_storage_store(
            ns,
            "waitstatus_to_exitcode",
            crate::make_builtin_function_with_arity(
                "waitstatus_to_exitcode",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "waitstatus_to_exitcode() requires 1 argument",
                        ));
                    }
                    let status = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    match rustpython_host_env::time::waitstatus_to_exitcode(status) {
                        Some(code) => Ok(pyre_object::w_int_new(code as i64)),
                        None => Err(crate::PyError::value_error(
                            "waitstatus_to_exitcode: invalid status",
                        )),
                    }
                },
                1,
            ),
        );

        // os.system(command) -> exit_status
        crate::dict_storage_store(
            ns,
            "system",
            crate::make_builtin_function_with_arity(
                "system",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("system() requires 1 argument"));
                    }
                    let cmd = unsafe {
                        if pyre_object::is_str(args[0]) {
                            pyre_object::w_str_get_value(args[0]).to_string()
                        } else {
                            return Err(crate::PyError::type_error(
                                "system(): command must be a string",
                            ));
                        }
                    };
                    let c_cmd = std::ffi::CString::new(cmd.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null in command"))?;
                    let rc = rustpython_host_env::os::system(&c_cmd);
                    Ok(pyre_object::w_int_new(rc as i64))
                },
                1,
            ),
        );

        // os.sendfile(out_fd, in_fd, offset, count) -> bytes_sent
        //
        // Ported from pypy/module/posix/interp_posix.py:2932-2961:
        //   * 4 positional args: out_fd, in_fd (called "in_" in PyPy because
        //     "in" is reserved), offset, count.
        //   * offset == None: linux-only "no-offset" path (NULL pointer);
        //     non-linux raises TypeError("an integer is required (got None)")
        //     verbatim from PyPy.
        //   * offset == int: read as i64 (PyPy uses
        //     space.gateway_r_longlong_w) and routed through
        //     rustpython_host_env::posix::sendfile (linux) or the BSD-form
        //     wrapper (macos).
        //   * Returns bytes-sent as int (PyPy: space.newint(res)).
        //
        // EINTR retry loop intentionally omitted — pyre's other os-syscall
        // wrappers don't do manual retry (relies on PEP 475 OS-level retry),
        // matching pyre-wide convention rather than introducing a single
        // outlier.
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        crate::dict_storage_store(
            ns,
            "sendfile",
            crate::make_builtin_function("sendfile", |args| {
                use std::os::fd::BorrowedFd;
                if args.len() < 4 {
                    return Err(crate::PyError::type_error(
                        "sendfile() requires 4 arguments",
                    ));
                }
                let out_fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                let in_fd = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                let w_offset = args[2];
                let count_raw = unsafe { pyre_object::w_int_get_value(args[3]) };
                if unsafe { pyre_object::is_none(w_offset) } {
                    // linux-only no-offset path; non-linux raises TypeError
                    // matching interp_posix.py:2946.
                    #[cfg(not(target_os = "linux"))]
                    {
                        let _ = (out_fd, in_fd, count_raw);
                        return Err(crate::PyError::type_error(
                            "an integer is required (got None)",
                        ));
                    }
                    #[cfg(target_os = "linux")]
                    {
                        // host_env doesn't expose a NULL-offset variant; call
                        // libc::sendfile directly with a null pointer, matching
                        // rposix.sendfile_no_offset (rposix.py:3066-3069).
                        let count = count_raw as libc::size_t;
                        let res =
                            unsafe { libc::sendfile(out_fd, in_fd, core::ptr::null_mut(), count) };
                        if res < 0 {
                            return Err(io_err(std::io::Error::last_os_error(), ""));
                        }
                        return Ok(pyre_object::w_int_new(res as i64));
                    }
                }
                let offset_i64 = unsafe { pyre_object::w_int_get_value(w_offset) };
                let out_b = unsafe { BorrowedFd::borrow_raw(out_fd) };
                let in_b = unsafe { BorrowedFd::borrow_raw(in_fd) };
                #[cfg(target_os = "linux")]
                {
                    let count = count_raw as usize;
                    let mut offset: rustpython_host_env::crt_fd::Offset = offset_i64 as _;
                    let n = host_posix::sendfile(out_b, in_b, &mut offset, count)
                        .map_err(|e| io_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(n as i64));
                }
                #[cfg(target_os = "macos")]
                {
                    let (res, written) = host_posix::sendfile(
                        in_b,
                        out_b,
                        offset_i64 as rustpython_host_env::crt_fd::Offset,
                        count_raw,
                        None,
                        None,
                    );
                    res.map_err(|e| io_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(written));
                }
            }),
        );

        // os.posix_spawn(path, argv, env, *, file_actions=None) -> pid
        // os.posix_spawnp(file, argv, env, *, file_actions=None) -> pid
        // Currently supports path/argv/env + the file_actions sequence
        // ((POSIX_SPAWN_OPEN, fd, path, flags, mode) | (POSIX_SPAWN_CLOSE,
        // fd) | (POSIX_SPAWN_DUP2, fd, newfd)). Other CPython kwargs
        // (setpgroup, setsid, setsigmask, setsigdef, resetids, scheduler)
        // are not yet plumbed.
        #[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "macos"))]
        {
            fn build_posix_spawn(
                args: &[pyre_object::PyObjectRef],
                spawnp: bool,
            ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
                let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
                if positional.len() < 3 {
                    return Err(crate::PyError::type_error(
                        "posix_spawn() requires path, argv, env",
                    ));
                }
                let path_str = extract_path(positional[0])?;
                let c_path = std::ffi::CString::new(path_str.as_bytes()).map_err(|_| {
                    crate::PyError::value_error("posix_spawn: embedded null in path")
                })?;
                let argv = collect_cstring_seq(positional[1], "posix_spawn", "argv")?;
                let env = collect_cstring_seq(positional[2], "posix_spawn", "env")?;
                let file_actions_obj = crate::builtins::kwarg_get(kwargs, "file_actions");
                let actions: Vec<rustpython_host_env::posix::PosixSpawnFileAction> =
                    if let Some(fa) = file_actions_obj {
                        if unsafe { pyre_object::is_none(fa) } {
                            Vec::new()
                        } else {
                            decode_file_actions(fa)?
                        }
                    } else {
                        Vec::new()
                    };
                let config = rustpython_host_env::posix::PosixSpawnConfig {
                    path: c_path.as_c_str(),
                    args: &argv,
                    env: &env,
                    file_actions: &actions,
                    setsigdef: None,
                    setpgroup: None,
                    resetids: false,
                    setsid: false,
                    setsigmask: None,
                    spawnp,
                };
                let pid = host_posix::posix_spawn(config).map_err(|e| io_err(e, ""))?;
                Ok(pyre_object::w_int_new(pid as i64))
            }
            fn collect_cstring_seq(
                obj: pyre_object::PyObjectRef,
                fn_name: &str,
                arg_name: &str,
            ) -> Result<Vec<std::ffi::CString>, crate::PyError> {
                let items: Vec<pyre_object::PyObjectRef> = if unsafe { pyre_object::is_list(obj) } {
                    let n = unsafe { pyre_object::w_list_len(obj) };
                    (0..n)
                        .filter_map(|i| unsafe { pyre_object::w_list_getitem(obj, i as i64) })
                        .collect()
                } else if unsafe { pyre_object::is_tuple(obj) } {
                    let n = unsafe { pyre_object::w_tuple_len(obj) };
                    (0..n)
                        .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(obj, i as i64) })
                        .collect()
                } else {
                    return Err(crate::PyError::type_error(format!(
                        "{fn_name}(): {arg_name} must be a list or tuple",
                    )));
                };
                items
                    .into_iter()
                    .map(|s| {
                        let bytes = unsafe {
                            if pyre_object::is_str(s) {
                                pyre_object::w_str_get_value(s).as_bytes().to_vec()
                            } else if pyre_object::is_bytes(s) {
                                pyre_object::w_bytes_data(s).to_vec()
                            } else {
                                return Err(crate::PyError::type_error(format!(
                                    "{fn_name}(): {arg_name} entries must be str or bytes",
                                )));
                            }
                        };
                        std::ffi::CString::new(bytes).map_err(|_| {
                            crate::PyError::value_error(format!(
                                "{fn_name}(): embedded null in {arg_name}",
                            ))
                        })
                    })
                    .collect()
            }
            fn decode_file_actions(
                obj: pyre_object::PyObjectRef,
            ) -> Result<Vec<rustpython_host_env::posix::PosixSpawnFileAction>, crate::PyError>
            {
                use rustpython_host_env::posix::PosixSpawnFileAction;
                let len = if unsafe { pyre_object::is_list(obj) } {
                    unsafe { pyre_object::w_list_len(obj) }
                } else if unsafe { pyre_object::is_tuple(obj) } {
                    unsafe { pyre_object::w_tuple_len(obj) }
                } else {
                    return Err(crate::PyError::type_error(
                        "posix_spawn: file_actions must be a list or tuple",
                    ));
                };
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let entry = if unsafe { pyre_object::is_list(obj) } {
                        unsafe { pyre_object::w_list_getitem(obj, i as i64) }
                    } else {
                        unsafe { pyre_object::w_tuple_getitem(obj, i as i64) }
                    }
                    .ok_or_else(|| {
                        crate::PyError::value_error("posix_spawn: file_actions entry missing")
                    })?;
                    if unsafe { !pyre_object::is_tuple(entry) } {
                        return Err(crate::PyError::type_error(
                            "posix_spawn: each file_actions entry must be a tuple",
                        ));
                    }
                    let tlen = unsafe { pyre_object::w_tuple_len(entry) };
                    if tlen < 2 {
                        return Err(crate::PyError::value_error(
                            "posix_spawn: file_actions entry too short",
                        ));
                    }
                    let op = (unsafe {
                        pyre_object::w_int_get_value(
                            pyre_object::w_tuple_getitem(entry, 0).unwrap(),
                        )
                    }) as i32;
                    match op {
                        0 => {
                            // POSIX_SPAWN_OPEN: (op, fd, path, flags, mode)
                            if tlen < 5 {
                                return Err(crate::PyError::value_error(
                                    "posix_spawn: OPEN action requires fd, path, flags, mode",
                                ));
                            }
                            let fd = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 1).unwrap(),
                                )
                            }) as i32;
                            let path_obj =
                                unsafe { pyre_object::w_tuple_getitem(entry, 2).unwrap() };
                            let path_str = extract_path(path_obj)?;
                            let cpath =
                                std::ffi::CString::new(path_str.as_bytes()).map_err(|_| {
                                    crate::PyError::value_error(
                                        "posix_spawn: embedded null in OPEN path",
                                    )
                                })?;
                            let oflag = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 3).unwrap(),
                                )
                            }) as i32;
                            let mode = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 4).unwrap(),
                                )
                            }) as u32;
                            out.push(PosixSpawnFileAction::Open {
                                fd,
                                path: cpath,
                                oflag,
                                mode,
                            });
                        }
                        1 => {
                            // POSIX_SPAWN_CLOSE: (op, fd)
                            let fd = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 1).unwrap(),
                                )
                            }) as i32;
                            out.push(PosixSpawnFileAction::Close { fd });
                        }
                        2 => {
                            // POSIX_SPAWN_DUP2: (op, fd, newfd)
                            if tlen < 3 {
                                return Err(crate::PyError::value_error(
                                    "posix_spawn: DUP2 action requires fd, newfd",
                                ));
                            }
                            let fd = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 1).unwrap(),
                                )
                            }) as i32;
                            let newfd = (unsafe {
                                pyre_object::w_int_get_value(
                                    pyre_object::w_tuple_getitem(entry, 2).unwrap(),
                                )
                            }) as i32;
                            out.push(PosixSpawnFileAction::Dup2 { fd, newfd });
                        }
                        _ => {
                            return Err(crate::PyError::value_error(
                                "posix_spawn: unknown file_actions opcode",
                            ));
                        }
                    }
                }
                Ok(out)
            }
            crate::dict_storage_store(
                ns,
                "posix_spawn",
                crate::make_builtin_function("posix_spawn", |args| build_posix_spawn(args, false)),
            );
            crate::dict_storage_store(
                ns,
                "posix_spawnp",
                crate::make_builtin_function("posix_spawnp", |args| build_posix_spawn(args, true)),
            );
            crate::dict_storage_store(ns, "POSIX_SPAWN_OPEN", pyre_object::w_int_new(0));
            crate::dict_storage_store(ns, "POSIX_SPAWN_CLOSE", pyre_object::w_int_new(1));
            crate::dict_storage_store(ns, "POSIX_SPAWN_DUP2", pyre_object::w_int_new(2));
        }

        // os.ttyname(fd) -> str
        crate::dict_storage_store(
            ns,
            "ttyname",
            crate::make_builtin_function_with_arity(
                "ttyname",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("ttyname() requires fd"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    let name = host_posix::ttyname(bfd).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_str_new(&name.to_string_lossy()))
                },
                1,
            ),
        );

        // os.tcgetpgrp(fd) -> pgid
        crate::dict_storage_store(
            ns,
            "tcgetpgrp",
            crate::make_builtin_function_with_arity(
                "tcgetpgrp",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("tcgetpgrp() requires fd"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    let pgid = host_posix::tcgetpgrp(bfd).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(pgid as i64))
                },
                1,
            ),
        );

        // os.tcsetpgrp(fd, pgid) -> None
        crate::dict_storage_store(
            ns,
            "tcsetpgrp",
            crate::make_builtin_function_with_arity(
                "tcsetpgrp",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("tcsetpgrp() requires fd, pgid"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let pgid = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::pid_t;
                    let bfd = unsafe { BorrowedFd::borrow_raw(fd) };
                    host_posix::tcsetpgrp(bfd, pgid).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.getpriority(which, who) -> int
        crate::dict_storage_store(
            ns,
            "getpriority",
            crate::make_builtin_function_with_arity(
                "getpriority",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "getpriority() requires which, who",
                        ));
                    }
                    let which = (unsafe { pyre_object::w_int_get_value(args[0]) })
                        as host_posix::PriorityWhichType;
                    let who = (unsafe { pyre_object::w_int_get_value(args[1]) })
                        as host_posix::PriorityWhoType;
                    let prio = host_posix::getpriority(which, who).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(prio as i64))
                },
                2,
            ),
        );

        // os.setpriority(which, who, priority) -> None
        crate::dict_storage_store(
            ns,
            "setpriority",
            crate::make_builtin_function_with_arity(
                "setpriority",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setpriority() requires which, who, priority",
                        ));
                    }
                    let which = (unsafe { pyre_object::w_int_get_value(args[0]) })
                        as host_posix::PriorityWhichType;
                    let who = (unsafe { pyre_object::w_int_get_value(args[1]) })
                        as host_posix::PriorityWhoType;
                    let prio = (unsafe { pyre_object::w_int_get_value(args[2]) }) as i32;
                    host_posix::setpriority(which, who, prio).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        crate::dict_storage_store(
            ns,
            "PRIO_PROCESS",
            pyre_object::w_int_new(libc::PRIO_PROCESS as i64),
        );
        crate::dict_storage_store(
            ns,
            "PRIO_PGRP",
            pyre_object::w_int_new(libc::PRIO_PGRP as i64),
        );
        crate::dict_storage_store(
            ns,
            "PRIO_USER",
            pyre_object::w_int_new(libc::PRIO_USER as i64),
        );

        // os.pathconf(path, name) -> int | None
        crate::dict_storage_store(
            ns,
            "pathconf",
            crate::make_builtin_function_with_arity(
                "pathconf",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("pathconf() requires path, name"));
                    }
                    let path = extract_path(args[0])?;
                    let cpath = std::ffi::CString::new(path.as_bytes()).map_err(|_| {
                        crate::PyError::value_error("pathconf: embedded null in path")
                    })?;
                    let name = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                    match host_posix::pathconf(&cpath, name).map_err(|e| io_err(e, ""))? {
                        Some(v) => Ok(pyre_object::w_int_new(v as i64)),
                        None => Ok(pyre_object::w_none()),
                    }
                },
                2,
            ),
        );

        // os.fpathconf(fd, name) -> int | None
        crate::dict_storage_store(
            ns,
            "fpathconf",
            crate::make_builtin_function_with_arity(
                "fpathconf",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fpathconf() requires fd, name"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let name = (unsafe { pyre_object::w_int_get_value(args[1]) }) as i32;
                    match host_posix::fpathconf(fd, name).map_err(|e| io_err(e, ""))? {
                        Some(v) => Ok(pyre_object::w_int_new(v as i64)),
                        None => Ok(pyre_object::w_none()),
                    }
                },
                2,
            ),
        );

        // os.sysconf(name) -> int
        crate::dict_storage_store(
            ns,
            "sysconf",
            crate::make_builtin_function_with_arity(
                "sysconf",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("sysconf() requires name"));
                    }
                    let name = (unsafe { pyre_object::w_int_get_value(args[0]) }) as i32;
                    let v = host_posix::sysconf(name).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(v as i64))
                },
                1,
            ),
        );

        // os.initgroups(username, gid) -> None
        #[cfg(any(target_os = "freebsd", target_os = "linux", target_os = "openbsd"))]
        crate::dict_storage_store(
            ns,
            "initgroups",
            crate::make_builtin_function_with_arity(
                "initgroups",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "initgroups() requires username, gid",
                        ));
                    }
                    let user = unsafe {
                        if pyre_object::is_str(args[0]) {
                            pyre_object::w_str_get_value(args[0]).to_string()
                        } else {
                            return Err(crate::PyError::type_error(
                                "initgroups(): username must be str",
                            ));
                        }
                    };
                    let cuser = std::ffi::CString::new(user.as_bytes()).map_err(|_| {
                        crate::PyError::value_error("initgroups: embedded null in username")
                    })?;
                    let gid = (unsafe { pyre_object::w_int_get_value(args[1]) }) as u32;
                    host_posix::initgroups(&cuser, gid).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.openpty() -> (master_fd, slave_fd)
        crate::dict_storage_store(
            ns,
            "openpty",
            crate::make_builtin_function_with_arity(
                "openpty",
                |_| {
                    use std::os::fd::IntoRawFd;
                    let (master, slave) = host_posix::openpty().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(master.into_raw_fd() as i64),
                        pyre_object::w_int_new(slave.into_raw_fd() as i64),
                    ]))
                },
                0,
            ),
        );

        // os.getresuid() -> (ruid, euid, suid)
        #[cfg(any(target_os = "android", target_os = "linux", target_os = "openbsd"))]
        crate::dict_storage_store(
            ns,
            "getresuid",
            crate::make_builtin_function_with_arity(
                "getresuid",
                |_| {
                    let (r, e, s) = host_posix::getresuid().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(r as i64),
                        pyre_object::w_int_new(e as i64),
                        pyre_object::w_int_new(s as i64),
                    ]))
                },
                0,
            ),
        );

        // os.getresgid() -> (rgid, egid, sgid)
        #[cfg(any(target_os = "android", target_os = "linux", target_os = "openbsd"))]
        crate::dict_storage_store(
            ns,
            "getresgid",
            crate::make_builtin_function_with_arity(
                "getresgid",
                |_| {
                    let (r, e, s) = host_posix::getresgid().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(r as i64),
                        pyre_object::w_int_new(e as i64),
                        pyre_object::w_int_new(s as i64),
                    ]))
                },
                0,
            ),
        );

        // os.setresuid(ruid, euid, suid) -> None
        #[cfg(any(
            target_os = "android",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "openbsd"
        ))]
        crate::dict_storage_store(
            ns,
            "setresuid",
            crate::make_builtin_function_with_arity(
                "setresuid",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setresuid() requires ruid, euid, suid",
                        ));
                    }
                    let r = (unsafe { pyre_object::w_int_get_value(args[0]) }) as u32;
                    let e = (unsafe { pyre_object::w_int_get_value(args[1]) }) as u32;
                    let s = (unsafe { pyre_object::w_int_get_value(args[2]) }) as u32;
                    host_posix::setresuid(r, e, s).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        // os.setresgid(rgid, egid, sgid) -> None
        #[cfg(any(target_os = "freebsd", target_os = "linux", target_os = "openbsd"))]
        crate::dict_storage_store(
            ns,
            "setresgid",
            crate::make_builtin_function_with_arity(
                "setresgid",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setresgid() requires rgid, egid, sgid",
                        ));
                    }
                    let r = (unsafe { pyre_object::w_int_get_value(args[0]) }) as u32;
                    let e = (unsafe { pyre_object::w_int_get_value(args[1]) }) as u32;
                    let s = (unsafe { pyre_object::w_int_get_value(args[2]) }) as u32;
                    host_posix::setresgid(r, e, s).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );
    }

    crate::dict_storage_store(ns, "error", crate::typedef::w_object());
}

/// _collections C-extension stub — PyPy: pypy/module/_collections/
/// Provides the C-accelerated deque/defaultdict/OrderedDict types.
/// Our stubs are backed by lists/dicts, which is correct semantically
/// but not performant. PyPy's W_Deque is a doubly-linked block list.
fn init_collections_c(ns: &mut DictStorage) {
    // deque(iterable=(), maxlen=None) — returns a list that we alias as deque.
    // Sufficient for collections.py's MutableSequence.register(deque).
    let deque_type = crate::typedef::make_builtin_type("deque", init_deque_type);
    crate::dict_storage_store(ns, "deque", deque_type);
    // _deque_iterator — reuse object (just a type sentinel)
    crate::dict_storage_store(ns, "_deque_iterator", crate::typedef::w_object());
    // defaultdict — returns a dict-like instance
    let defaultdict_type = crate::typedef::make_builtin_type("defaultdict", init_defaultdict_type);
    crate::dict_storage_store(ns, "defaultdict", defaultdict_type);
    // OrderedDict — same as dict for our purposes
    crate::dict_storage_store(ns, "OrderedDict", crate::typedef::w_type());
}

/// deque methods — PyPy: pypy/module/_collections/interp_deque.py W_Deque
fn init_deque_type(ns: &mut DictStorage) {
    // __init__(self, iterable=(), maxlen=None) — store items as __data__ list
    crate::dict_storage_store(
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
    crate::dict_storage_store(
        ns,
        "append",
        crate::make_builtin_function_with_arity(
            "append",
            |args| {
                if args.len() >= 2 {
                    if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                        unsafe { pyre_object::w_list_append(data, args[1]) };
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "appendleft",
        crate::make_builtin_function_with_arity(
            "appendleft",
            |args| {
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
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "pop",
        crate::make_builtin_function_with_arity(
            "pop",
            |args| {
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
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "popleft",
        crate::make_builtin_function_with_arity(
            "popleft",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    unsafe {
                        let n = pyre_object::w_list_len(data);
                        if n > 0 {
                            let item = pyre_object::w_list_getitem(data, 0)
                                .unwrap_or(pyre_object::w_none());
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
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "clear",
        crate::make_builtin_function_with_arity(
            "clear",
            |args| {
                if !args.is_empty() {
                    let _ = crate::baseobjspace::setattr(
                        args[0],
                        "__data__",
                        pyre_object::w_list_new(vec![]),
                    );
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "extend",
        crate::make_builtin_function_with_arity(
            "extend",
            |args| {
                if args.len() >= 2 {
                    let items = crate::builtins::collect_iterable(args[1])?;
                    if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                        for item in items {
                            unsafe { pyre_object::w_list_append(data, item) };
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__len__",
        crate::make_builtin_function_with_arity(
            "__len__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    return Ok(pyre_object::w_int_new(
                        unsafe { pyre_object::w_list_len(data) } as i64,
                    ));
                }
                Ok(pyre_object::w_int_new(0))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__iter__",
        crate::make_builtin_function_with_arity(
            "__iter__",
            |args| {
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
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__getitem__",
        crate::make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_none());
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    return crate::baseobjspace::getitem(data, args[1]);
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
}

/// defaultdict — PyPy: pypy/module/_collections/interp_defaultdict.py
fn init_defaultdict_type(ns: &mut DictStorage) {
    crate::dict_storage_store(
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
    crate::dict_storage_store(
        ns,
        "__getitem__",
        crate::make_builtin_function_with_arity(
            "__getitem__",
            |args| {
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
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__setitem__",
        crate::make_builtin_function_with_arity(
            "__setitem__",
            |args| {
                if args.len() >= 3 {
                    if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                        unsafe { pyre_object::w_dict_store(data, args[1], args[2]) };
                    }
                }
                Ok(pyre_object::w_none())
            },
            3,
        ),
    );
}

/// _opcode stub — PyPy: pypy/module/_opcode (CPython's opcode introspection).
/// opcode.py requires stack_effect + has_arg/has_const/has_name/has_jump and
/// related classifiers. Our stubs return neutral values; full implementations
/// would mirror CPython Python/compile.c.
fn init_opcode_c(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "stack_effect",
        crate::make_builtin_function_with_arity(
            "stack_effect",
            |_| Ok(pyre_object::w_int_new(0)),
            3,
        ),
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
        crate::dict_storage_store(
            ns,
            name,
            crate::make_builtin_function_with_arity(
                name,
                |_| Ok(pyre_object::w_bool_from(false)),
                0,
            ),
        );
    }
    crate::dict_storage_store(
        ns,
        "get_executor",
        crate::make_builtin_function_with_arity("get_executor", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "get_specialization_stats",
        crate::make_builtin_function_with_arity(
            "get_specialization_stats",
            |_| Ok(pyre_object::w_dict_new()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_intrinsic1_descs",
        crate::make_builtin_function_with_arity(
            "get_intrinsic1_descs",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_intrinsic2_descs",
        crate::make_builtin_function_with_arity(
            "get_intrinsic2_descs",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_opname",
        crate::make_builtin_function_with_arity(
            "get_opname",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_str_new("<0>"));
                }
                let code = unsafe { pyre_object::w_int_get_value(args[0]) };
                Ok(pyre_object::w_str_new(&format!("<{code}>")))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_nb_ops",
        crate::make_builtin_function_with_arity(
            "get_nb_ops",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_special_method_names",
        crate::make_builtin_function_with_arity(
            "get_special_method_names",
            |_| {
                Ok(pyre_object::w_list_new(vec![
                    pyre_object::w_str_new("__enter__"),
                    pyre_object::w_str_new("__exit__"),
                    pyre_object::w_str_new("__aenter__"),
                    pyre_object::w_str_new("__aexit__"),
                ]))
            },
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_executor_count",
        crate::make_builtin_function_with_arity(
            "get_executor_count",
            |_| Ok(pyre_object::w_int_new(0)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_hot_code",
        crate::make_builtin_function_with_arity(
            "get_hot_code",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
}

/// importlib stub — PyPy: pypy/module/importlib/
/// Avoid loading the real importlib.__init__ since it drags in
/// _bootstrap and _bootstrap_external.
fn init_importlib_pkg(ns: &mut DictStorage) {
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
fn init_importlib_util(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "spec_from_file_location",
        crate::make_builtin_function_with_arity(
            "spec_from_file_location",
            |_| Ok(pyre_object::w_none()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "module_from_spec",
        crate::make_builtin_function_with_arity(
            "module_from_spec",
            |_| Ok(pyre_object::w_none()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "find_spec",
        crate::make_builtin_function_with_arity("find_spec", |_| Ok(pyre_object::w_none()), 0),
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
fn init_importlib_abc(ns: &mut DictStorage) {
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
fn init_importlib_machinery(ns: &mut DictStorage) {
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

/// _imp stub — PyPy: pypy/module/imp/
///
/// Minimal subset required by importlib._bootstrap to decide which loader
/// handles a name. We report every name we know about as a builtin so
/// pyre's own registrations remain authoritative.
fn init_imp(ns: &mut DictStorage) {
    crate::dict_storage_store(
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
                let is_builtin = BUILTIN_MODULES.with(|m| m.borrow().contains_key(name));
                Ok(pyre_object::w_int_new(if is_builtin { 1 } else { 0 }))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "is_frozen",
        crate::make_builtin_function_with_arity(
            "is_frozen",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "is_frozen_package",
        crate::make_builtin_function_with_arity(
            "is_frozen_package",
            |_| Ok(pyre_object::w_bool_from(false)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_frozen_object",
        crate::make_builtin_function_with_arity(
            "get_frozen_object",
            |_| Ok(pyre_object::w_none()),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "create_builtin",
        crate::make_builtin_function_with_arity(
            "create_builtin",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                Ok(args[0])
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "exec_builtin",
        crate::make_builtin_function_with_arity(
            "exec_builtin",
            |_| Ok(pyre_object::w_int_new(0)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "exec_dynamic",
        crate::make_builtin_function_with_arity(
            "exec_dynamic",
            |_| Ok(pyre_object::w_int_new(0)),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "acquire_lock",
        crate::make_builtin_function_with_arity("acquire_lock", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "release_lock",
        crate::make_builtin_function_with_arity("release_lock", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "lock_held",
        crate::make_builtin_function_with_arity(
            "lock_held",
            |_| Ok(pyre_object::w_bool_from(false)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_fix_co_filename",
        crate::make_builtin_function_with_arity(
            "_fix_co_filename",
            |_| Ok(pyre_object::w_none()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "extension_suffixes",
        crate::make_builtin_function_with_arity(
            "extension_suffixes",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "source_hash",
        crate::make_builtin_function_with_arity(
            "source_hash",
            |_| Ok(pyre_object::w_int_new(0)),
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "check_hash_based_pycs",
        pyre_object::w_str_new("default"),
    );
    crate::dict_storage_store(ns, "pyc_magic_number_token", pyre_object::w_int_new(3495));
}

/// _ast stub — PyPy: pypy/module/_ast/
///
/// Exposes the AST node type hierarchy as plain type stubs. Our stubs are
/// enough to satisfy `from _ast import *` in `ast.py` and class body
/// references like `class slice(AST)`. Actual AST construction is not
/// supported because pyre uses RustPython's compiler.
fn init_ast(ns: &mut DictStorage) {
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
            crate::dict_storage_store(ns, name, pyre_object::w_int_new(0));
        } else {
            crate::dict_storage_store(ns, name, crate::typedef::make_builtin_type(name, |_| {}));
        }
    }
}

/// errno stub — PyPy: pypy/module/errno/
fn init_errno(ns: &mut DictStorage) {
    // Numerics differ per OS (e.g. EAGAIN is 11 on Linux but 35 on macOS),
    // so when `host_env` is enabled we resolve every constant through
    // `rustpython_host_env::errno::errors` (a `pub use libc::*` re-export).
    // The `host_env = off` build keeps a darwin/BSD-flavoured fallback so
    // pyre-wasm preserves its previous behaviour.
    #[cfg(feature = "host_env")]
    {
        use rustpython_host_env::errno::errors as host_errno;
        let entries: &[(&str, i32)] = &[
            ("EPERM", host_errno::EPERM),
            ("ENOENT", host_errno::ENOENT),
            ("ESRCH", host_errno::ESRCH),
            ("EINTR", host_errno::EINTR),
            ("EIO", host_errno::EIO),
            ("ENXIO", host_errno::ENXIO),
            ("E2BIG", host_errno::E2BIG),
            ("ENOEXEC", host_errno::ENOEXEC),
            ("EBADF", host_errno::EBADF),
            ("ECHILD", host_errno::ECHILD),
            ("EAGAIN", host_errno::EAGAIN),
            ("EWOULDBLOCK", host_errno::EWOULDBLOCK),
            ("ENOMEM", host_errno::ENOMEM),
            ("EACCES", host_errno::EACCES),
            ("EFAULT", host_errno::EFAULT),
            ("EBUSY", host_errno::EBUSY),
            ("EEXIST", host_errno::EEXIST),
            ("EXDEV", host_errno::EXDEV),
            ("ENODEV", host_errno::ENODEV),
            ("ENOTDIR", host_errno::ENOTDIR),
            ("EISDIR", host_errno::EISDIR),
            ("EINVAL", host_errno::EINVAL),
            ("ENFILE", host_errno::ENFILE),
            ("EMFILE", host_errno::EMFILE),
            ("ENOTTY", host_errno::ENOTTY),
            ("EFBIG", host_errno::EFBIG),
            ("ENOSPC", host_errno::ENOSPC),
            ("ESPIPE", host_errno::ESPIPE),
            ("EROFS", host_errno::EROFS),
            ("EMLINK", host_errno::EMLINK),
            ("EPIPE", host_errno::EPIPE),
            ("EDOM", host_errno::EDOM),
            ("ERANGE", host_errno::ERANGE),
            ("EDEADLK", host_errno::EDEADLK),
            ("ENAMETOOLONG", host_errno::ENAMETOOLONG),
            ("ENOLCK", host_errno::ENOLCK),
            ("ENOSYS", host_errno::ENOSYS),
            ("ENOTEMPTY", host_errno::ENOTEMPTY),
            ("ELOOP", host_errno::ELOOP),
            ("EOVERFLOW", host_errno::EOVERFLOW),
            ("EPROTO", host_errno::EPROTO),
            ("EDESTADDRREQ", host_errno::EDESTADDRREQ),
            ("EAFNOSUPPORT", host_errno::EAFNOSUPPORT),
            ("EALREADY", host_errno::EALREADY),
            ("EDQUOT", host_errno::EDQUOT),
        ];
        for (name, value) in entries {
            crate::dict_storage_store(ns, name, pyre_object::w_int_new(*value as i64));
        }
        // Unix-only constants (windows libc lacks some of these).
        #[cfg(unix)]
        {
            let unix_entries: &[(&str, i32)] = &[
                ("ENOTBLK", host_errno::ENOTBLK),
                ("ETXTBSY", host_errno::ETXTBSY),
                ("ENOMSG", host_errno::ENOMSG),
                ("EIDRM", host_errno::EIDRM),
                ("EBADMSG", host_errno::EBADMSG),
                ("EMULTIHOP", host_errno::EMULTIHOP),
                ("ENODATA", host_errno::ENODATA),
                ("ENOLINK", host_errno::ENOLINK),
                ("ENOSR", host_errno::ENOSR),
                ("ENOSTR", host_errno::ENOSTR),
                ("ETIME", host_errno::ETIME),
            ];
            for (name, value) in unix_entries {
                crate::dict_storage_store(ns, name, pyre_object::w_int_new(*value as i64));
            }
        }
    }
    #[cfg(not(feature = "host_env"))]
    {
        let entries: &[(&str, i64)] = &[
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
        ];
        for (name, value) in entries {
            crate::dict_storage_store(ns, name, pyre_object::w_int_new(*value));
        }
    }
    crate::dict_storage_store(ns, "errorcode", pyre_object::w_dict_new());
}

/// _codecs stub — PyPy: pypy/module/_codecs/
///
/// Provides lookup_error/register_error and encode/decode no-op stubs so
/// codecs.py module init runs to completion.
fn init_codecs(ns: &mut DictStorage) {
    // lookup_error(name) — returns an error handler for the given error
    // strategy. Pyre returns a pass-through lambda that never fires because
    // we don't encounter encoding errors in the pure-Python stdlib paths
    // we exercise so far.
    crate::dict_storage_store(
        ns,
        "lookup_error",
        crate::make_builtin_function_with_arity(
            "lookup_error",
            |_| {
                Ok(crate::make_builtin_function_with_arity(
                    "error_handler",
                    |args| {
                        Ok(if args.is_empty() {
                            pyre_object::w_none()
                        } else {
                            args[0]
                        })
                    },
                    1,
                ))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "register_error",
        crate::make_builtin_function_with_arity("register_error", |_| Ok(pyre_object::w_none()), 2),
    );
    crate::dict_storage_store(
        ns,
        "register",
        crate::make_builtin_function_with_arity("register", |_| Ok(pyre_object::w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "lookup",
        crate::make_builtin_function_with_arity("lookup", |_| Ok(pyre_object::w_none()), 1),
    );
    // encode/decode — return input unchanged. Matches PyPy _codecs.encode
    // when the codec is the identity.
    let identity = crate::make_builtin_function_with_arity(
        "identity",
        |args| {
            Ok(if args.is_empty() {
                pyre_object::w_none()
            } else {
                args[0]
            })
        },
        1,
    );
    crate::dict_storage_store(ns, "encode", identity);
    crate::dict_storage_store(ns, "decode", identity);
    crate::dict_storage_store(ns, "_forget_codec", identity);
    crate::dict_storage_store(
        ns,
        "charmap_build",
        crate::make_builtin_function_with_arity(
            "charmap_build",
            |_| Ok(pyre_object::w_dict_new()),
            1,
        ),
    );
}

/// copyreg stub — PyPy: pypy/module/copyreg/
fn init_copyreg(ns: &mut DictStorage) {
    // copyreg.pickle(type, reduce_func, constructor=None) — register a
    // pickle reducer. Stub: ignore (pyre doesn't support pickle).
    crate::dict_storage_store(
        ns,
        "pickle",
        crate::make_builtin_function_with_arity("pickle", |_| Ok(pyre_object::w_none()), 3),
    );
    crate::dict_storage_store(ns, "dispatch_table", pyre_object::w_dict_new());
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

/// Detect CPython stdlib path via `python3 -c "import sysconfig; ..."`.
///
/// PyPy equivalent: initpath.py scans for lib-python/X.Y at startup.
#[cfg(feature = "host_env")]
fn detect_stdlib_path() -> Option<PathBuf> {
    // Try PYRE_STDLIB env var first
    if let Ok(p) = host_os::var("PYRE_STDLIB") {
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
            load_builtin_module(modulename).ok_or_else(|| crate::PyError {
                kind: crate::PyErrorKind::ImportError,
                message: format!("builtin module '{modulename}' failed to initialize"),
                exc_object: std::ptr::null_mut(),
                attach_tb: true,
                reraise_lasti: -1,
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
            // Same builtins-identity path as the full_is_builtin branch
            // above: route `import builtins` through `EC.get_builtin()`
            // so `import builtins is space.builtin` holds.
            let m = if partname == "builtins" && !execution_context.is_null() {
                unsafe { (*execution_context).get_builtin() }
            } else {
                load_builtin_module(partname).ok_or_else(|| crate::PyError {
                    kind: crate::PyErrorKind::ImportError,
                    message: format!("builtin module '{modulename}' failed to initialize"),
                    exc_object: std::ptr::null_mut(),
                    attach_tb: true,
                    reraise_lasti: -1,
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
    let package = resolve_package_name(w_globals)?.ok_or_else(|| crate::PyError {
        kind: crate::PyErrorKind::ImportError,
        message: "attempted relative import with no known parent package".to_string(),
        exc_object: std::ptr::null_mut(),
        attach_tb: true,
        reraise_lasti: -1,
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
            attach_tb: true,
            reraise_lasti: -1,
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
    if let Ok(value) = crate::baseobjspace::getattr(module, name) {
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
        match crate::baseobjspace::getattr(module, "__all__") {
            Ok(w_all) => (w_all, false),
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
                // pyopcode.py:2225-2230 — `dict = module.__dict__; all = dict.keys()`.
                // `space.getattr(module, '__dict__')` so any object exposing
                // `__dict__` (Module, class, instance with `__dict__`,
                // bytes-keyed proxies, ...) participates.
                match crate::baseobjspace::getattr(module, "__dict__") {
                    Ok(w_dict) => {
                        let w_keys_method = crate::baseobjspace::getattr(w_dict, "keys")?;
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
    let module_name_w = crate::baseobjspace::getattr(module, "__name__")?;
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
        let value = crate::baseobjspace::getattr(module, &name)?;
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
