//! Builtin module registry — PyPy equivalent: pypy/module/
//!
//! Each subdirectory corresponds to a PyPy module package
//! (e.g. `math/` ↔ `pypy/module/math/`).

pub mod __builtin__;
pub mod __pypy__;
pub mod _abc;
pub mod _ast;
pub mod _codecs;
pub mod _collections;
pub mod _contextvars;
pub mod _functools;
pub mod _immutables_map;
pub mod _io;
pub mod _locale;
pub mod _opcode;
pub mod _pickle;
pub mod _random;
pub mod _sre;
pub mod _stat;
pub mod _suggestions;
pub mod _symtable;
pub mod _template;
pub mod _tokenize;
pub mod _types;
pub mod _typing;
pub mod _warnings;
pub mod _weakref;
pub mod array;
pub mod atexit;
pub mod cmath;
pub mod errno;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod faulthandler;
pub mod gc;
pub mod imp;
pub mod importlib;
pub mod itertools;
pub mod marshal;
pub mod math;
pub mod operator;
pub mod posix;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod pwd;
pub mod pypyjit;
pub mod signal;
pub mod r#struct;
pub mod sys;
pub mod thread;
pub mod time;
