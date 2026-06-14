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
pub mod _posixshmem;
#[allow(non_snake_case)]
pub mod _random;
#[allow(non_snake_case)]
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
pub mod faulthandler;
pub mod fcntl;
pub mod gc;
pub mod grp;
pub mod importlib;
pub mod itertools;
pub mod math;
pub mod mmap;
pub mod operator;
pub mod posix;
pub mod pwd;
pub mod resource;
pub mod select;
pub mod sys;
pub mod syslog;
pub mod termios;
pub mod time;
pub mod unicodedata;
