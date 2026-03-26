//! Builtin module registry — PyPy equivalent: pypy/module/
//!
//! Each subdirectory corresponds to a PyPy module package
//! (e.g. `math/` ↔ `pypy/module/math/`).

pub mod builtins_mod;
pub mod io_mod;
pub mod math;
pub mod operator;
pub mod sys;
pub mod time;
