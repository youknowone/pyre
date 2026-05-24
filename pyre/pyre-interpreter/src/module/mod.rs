//! Builtin module registry — PyPy equivalent: pypy/module/
//!
//! Each subdirectory corresponds to a PyPy module package
//! (e.g. `math/` ↔ `pypy/module/math/`).

pub mod __builtin__;
pub mod _io;
#[allow(non_snake_case)]
pub mod _ctypes;
#[allow(non_snake_case)]
pub mod _multiprocessing;
#[allow(non_snake_case)]
pub mod _posixshmem;
#[allow(non_snake_case)]
pub mod _socket;
pub mod _sre;
pub mod _weakref;
pub mod faulthandler;
pub mod math;
pub mod mmap;
pub mod operator;
pub mod sys;
pub mod time;
