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
pub mod _functools;
#[allow(non_snake_case)]
pub mod _immutables_map;
pub mod _io;
#[allow(non_snake_case)]
pub mod _locale;
#[allow(non_snake_case)]
pub mod _opcode;
#[allow(non_snake_case)]
pub mod _pickle;
#[allow(non_snake_case)]
pub mod _random;
pub mod _sre;
#[allow(non_snake_case)]
pub mod _stat;
#[allow(non_snake_case)]
pub mod _suggestions;
#[allow(non_snake_case)]
pub mod _symtable;
pub mod _template;
#[allow(non_snake_case)]
pub mod _tokenize;
#[allow(non_snake_case)]
pub mod _types;
#[allow(non_snake_case)]
pub mod _typing;
pub mod _warnings;
pub mod _weakref;
pub mod array;
pub mod atexit;
pub mod errno;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod faulthandler;
pub mod gc;
#[allow(non_snake_case)]
pub mod imp;
pub mod importlib;
pub mod itertools;
pub mod marshal;
pub mod operator;
pub mod posix;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod pwd;
pub mod pypyjit;
#[allow(non_snake_case)]
pub mod signal;
#[allow(non_snake_case)]
pub mod r#struct;
pub mod sys;
#[allow(non_snake_case)]
pub mod thread;
pub mod time;
