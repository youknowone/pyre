//! Builtin module registry — PyPy equivalent: pypy/module/
//!
//! Each subdirectory corresponds to a PyPy module package
//! (e.g. `math/` ↔ `pypy/module/math/`).

pub mod __builtin__;
#[allow(non_snake_case)]
pub mod __pypy__;
#[allow(non_snake_case)]
pub mod _ast;
#[allow(non_snake_case)]
pub mod _codecs;
#[allow(non_snake_case)]
pub mod _collections;
#[allow(non_snake_case)]
pub mod _contextvars;
pub mod _io;
#[allow(non_snake_case)]
pub mod _locale;
#[allow(non_snake_case)]
pub mod _pickle;
#[allow(non_snake_case)]
pub mod _random;
pub mod _sre;
#[allow(non_snake_case)]
pub mod _types;
pub mod _warnings;
pub mod _weakref;
pub mod array;
pub mod atexit;
pub mod gc;
#[allow(non_snake_case)]
pub mod imp;
pub mod importlib;
pub mod itertools;
pub mod marshal;
pub mod operator;
pub mod posix;
#[allow(non_snake_case)]
pub mod signal;
#[allow(non_snake_case)]
pub mod r#struct;
pub mod sys;
#[allow(non_snake_case)]
pub mod thread;
pub mod time;
