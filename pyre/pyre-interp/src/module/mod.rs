//! Builtin module registry — PyPy equivalent: pypy/module/
//!
//! Each subdirectory corresponds to a PyPy module package
//! (e.g. `math/` ↔ `pypy/module/math/`).

pub mod math;
pub mod sys;
pub mod time;
