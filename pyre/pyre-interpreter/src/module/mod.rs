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
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
pub mod _cffi_backend;
#[allow(non_snake_case)]
pub mod _codecs;
#[allow(non_snake_case)]
pub mod _collections;
#[allow(non_snake_case)]
pub mod _contextvars;
#[allow(non_snake_case)]
#[cfg(not(feature = "sandbox"))]
pub mod _ctypes;
#[allow(non_snake_case)]
pub mod _functools;
#[allow(non_snake_case)]
pub mod _hashlib;
pub mod _io;
#[allow(non_snake_case)]
pub mod _locale;
#[allow(non_snake_case)]
pub mod _pickle;
#[allow(non_snake_case)]
pub mod _pypy_generic_alias;
#[allow(non_snake_case)]
pub mod _random;
pub mod _sre;
#[allow(non_snake_case)]
pub mod _stat;
#[allow(non_snake_case)]
pub mod _symtable;
#[allow(non_snake_case)]
pub mod _types;
#[allow(non_snake_case)]
pub mod _typing;
pub mod _warnings;
pub mod _weakref;
#[allow(non_snake_case)]
#[cfg(windows)]
pub mod _winapi;
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub mod _wmi;
pub mod array;
pub mod atexit;
pub mod errno;
pub mod gc;
#[allow(non_snake_case)]
pub mod imp;
pub mod importlib;
pub mod itertools;
pub mod marshal;
#[cfg(all(windows, feature = "host_env"))]
pub mod msvcrt;
pub mod operator;
pub mod posix;
pub mod pypyjit;
#[allow(non_snake_case)]
pub mod signal;
#[allow(non_snake_case)]
pub mod r#struct;
pub mod sys;
#[allow(non_snake_case)]
pub mod thread;
pub mod time;
#[cfg(windows)]
pub mod winreg;
#[allow(non_snake_case)]
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub mod winsound;
