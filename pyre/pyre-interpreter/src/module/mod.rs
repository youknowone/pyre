//! Builtin module registry — PyPy equivalent: pypy/module/
//!
//! Each subdirectory corresponds to a PyPy module package
//! (e.g. `math/` ↔ `pypy/module/math/`).

pub mod __builtin__;
#[allow(non_snake_case)]
pub mod __pypy__;
#[allow(non_snake_case)]
pub mod _abc;
#[allow(non_snake_case)]
pub mod _ast;
#[allow(non_snake_case)]
pub mod _bisect;
#[allow(non_snake_case)]
pub mod _blake2;
#[allow(non_snake_case)]
pub mod _bz2;
#[allow(non_snake_case)]
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
pub mod _cffi_backend;
#[allow(non_snake_case)]
pub mod _codecs;
#[allow(non_snake_case)]
pub mod _codecs_cn;
#[allow(non_snake_case)]
pub mod _codecs_hk;
#[allow(non_snake_case)]
pub mod _codecs_iso2022;
#[allow(non_snake_case)]
pub mod _codecs_jp;
#[allow(non_snake_case)]
pub mod _codecs_kr;
#[allow(non_snake_case)]
pub mod _codecs_tw;
#[allow(non_snake_case)]
pub mod _collections;
#[allow(non_snake_case)]
pub mod _contextvars;
#[allow(non_snake_case)]
pub mod _csv;
#[allow(non_snake_case)]
#[cfg(not(feature = "sandbox"))]
pub mod _ctypes;
#[allow(non_snake_case)]
pub mod _functools;
#[allow(non_snake_case)]
pub mod _hashlib;
#[allow(non_snake_case)]
pub mod _heapq;
#[allow(non_snake_case)]
pub mod _immutables_map;
pub mod _io;
#[allow(non_snake_case)]
pub mod _json;
#[allow(non_snake_case)]
pub mod _locale;
#[allow(non_snake_case)]
pub mod _lsprof;
#[allow(non_snake_case)]
pub mod _lzma;
#[allow(non_snake_case)]
pub mod _multibytecodec;
#[allow(non_snake_case)]
#[cfg(not(feature = "sandbox"))]
pub mod _multiprocessing;
#[allow(non_snake_case)]
pub mod _opcode;
#[allow(non_snake_case)]
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub mod _overlapped;
#[allow(non_snake_case)]
pub mod _pickle;
#[allow(non_snake_case)]
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod _posixshmem;
#[allow(non_snake_case)]
#[cfg(not(feature = "sandbox"))]
pub mod _posixsubprocess;
#[allow(non_snake_case)]
pub mod _pypy_generic_alias;
#[allow(non_snake_case)]
pub mod _queue;
#[allow(non_snake_case)]
pub mod _random;
#[allow(non_snake_case)]
#[cfg(not(feature = "sandbox"))]
pub mod _socket;
pub mod _sre;
#[allow(non_snake_case)]
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod _ssl;
#[allow(non_snake_case)]
pub mod _stat;
#[allow(non_snake_case)]
pub mod _statistics;
pub mod _suggestions;
#[allow(non_snake_case)]
pub mod _symtable;
#[allow(non_snake_case)]
pub mod _template;
pub mod _tokenize;
#[allow(non_snake_case)]
pub mod _types;
#[allow(non_snake_case)]
pub mod _typing;
// `uuid.py` reaches a MAC-derived node only through this module; a build
// without it answers `getnode()` from `os.urandom`.
#[cfg(all(windows, not(feature = "sandbox")))]
#[allow(non_snake_case)]
pub mod _uuid;
pub mod _warnings;
pub mod _weakref;
#[allow(non_snake_case)]
#[cfg(windows)]
pub mod _winapi;
#[cfg(all(windows, not(feature = "sandbox")))]
pub mod _wmi;
pub mod array;
pub mod atexit;
pub mod binascii;
pub mod cmath;
pub mod errno;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod faulthandler;
// All four callables inside are gated on `all(unix, feature = "host_env")` with
// a NotImplementedError on the other arm, so a build without it would offer a
// module of constants and four functions that cannot run.  `mailbox`,
// `subprocess` and `pathlib._os` each import it inside `try/except ImportError`
// and take a fallback when it is missing; an import that succeeds would take
// that fallback away without supplying anything to replace it, so the module is
// left out instead.
#[cfg(all(not(feature = "sandbox"), feature = "host_env"))]
pub mod fcntl;
pub mod gc;
#[cfg(all(unix, not(feature = "sandbox")))]
pub mod grp;
#[allow(non_snake_case)]
pub mod imp;
pub mod importlib;
pub mod itertools;
pub mod marshal;
pub mod math;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod mmap;
#[cfg(all(windows, feature = "host_env"))]
pub mod msvcrt;
pub mod operator;
pub mod posix;
#[cfg(all(unix, not(feature = "sandbox")))]
pub mod pwd;
pub mod pyexpat;
pub mod pypyjit;
// All three callables are gated on `all(unix, feature = "host_env")` the way
// `fcntl`'s are, and it is left out for the same reason.  Its readers are test
// modules rather than the stdlib proper, and they ask the same question:
// `test.support`, `test_os`, `test_subprocess` and `test_selectors` all import
// it inside `try/except ImportError`.
#[cfg(all(not(feature = "sandbox"), feature = "host_env"))]
pub mod resource;
#[cfg(not(feature = "sandbox"))]
pub mod select;
#[allow(non_snake_case)]
pub mod signal;
#[allow(non_snake_case)]
pub mod r#struct;
pub mod sys;
#[cfg(not(feature = "sandbox"))]
pub mod syslog;
#[cfg(not(feature = "sandbox"))]
pub mod termios;
#[allow(non_snake_case)]
pub mod thread;
pub mod time;
pub mod unicodedata;
#[cfg(windows)]
pub mod winreg;
#[allow(non_snake_case)]
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub mod winsound;
pub mod zlib;
