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
pub mod _codecs;
#[allow(non_snake_case)]
pub mod _collections;
#[allow(non_snake_case)]
pub mod _contextvars;
#[allow(non_snake_case)]
pub mod _ctypes;
#[allow(non_snake_case)]
pub mod _functools;
#[allow(non_snake_case)]
pub mod _imp;
pub mod _io;
#[allow(non_snake_case)]
pub mod _locale;
#[allow(non_snake_case)]
pub mod _multiprocessing;
#[allow(non_snake_case)]
pub mod _opcode;
#[allow(non_snake_case)]
pub mod _pickle;
#[allow(non_snake_case)]
#[cfg(not(target_arch = "wasm32"))]
pub mod _posixshmem;
#[allow(non_snake_case)]
pub mod _random;
#[allow(non_snake_case)]
#[cfg(not(target_arch = "wasm32"))]
pub mod _signal;
#[allow(non_snake_case)]
pub mod _socket;
pub mod _sre;
#[allow(non_snake_case)]
pub mod _struct;
#[allow(non_snake_case)]
pub mod _thread;
pub mod _weakref;
pub mod atexit;
pub mod cmath;
pub mod errno;
#[cfg(not(target_arch = "wasm32"))]
pub mod faulthandler;
pub mod fcntl;
pub mod gc;
#[cfg(not(target_arch = "wasm32"))]
pub mod grp;
pub mod importlib;
pub mod itertools;
pub mod math;
#[cfg(not(target_arch = "wasm32"))]
pub mod mmap;
pub mod operator;
#[cfg(not(target_arch = "wasm32"))]
pub mod posix;
#[cfg(not(target_arch = "wasm32"))]
pub mod pwd;
pub mod resource;
pub mod select;
pub mod sys;
pub mod syslog;
pub mod termios;
#[cfg(not(target_arch = "wasm32"))]
pub mod time;
pub mod unicodedata;
