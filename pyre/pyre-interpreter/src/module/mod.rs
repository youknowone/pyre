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
pub mod _codecs;
#[allow(non_snake_case)]
#[cfg(not(target_arch = "wasm32"))]
pub mod _codecs_jp;
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
#[cfg(not(target_arch = "wasm32"))]
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
#[allow(non_snake_case)]
pub mod _symtable;
#[allow(non_snake_case)]
pub mod _template;
pub mod _tokenize;
#[allow(non_snake_case)]
pub mod _types;
#[allow(non_snake_case)]
pub mod _typing;
pub mod _warnings;
pub mod _weakref;
#[allow(non_snake_case)]
#[cfg(windows)]
pub mod _winapi;
pub mod array;
pub mod atexit;
pub mod binascii;
pub mod cmath;
pub mod errno;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod faulthandler;
#[cfg(not(feature = "sandbox"))]
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
#[cfg(not(target_arch = "wasm32"))]
pub mod posix;
#[cfg(all(unix, not(feature = "sandbox")))]
pub mod pwd;
pub mod pyexpat;
pub mod pypyjit;
#[cfg(not(feature = "sandbox"))]
pub mod resource;
#[cfg(not(feature = "sandbox"))]
pub mod select;
#[allow(non_snake_case)]
#[cfg(not(target_arch = "wasm32"))]
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
