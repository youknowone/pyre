//! Fixed trace helpers for `MIFrame` — hand-maintained Rust (not translator
//! output), the trace-time analogs of PyPy's hand-written RPython. Split by
//! PyPy's file boundaries:
//!   - `dispatch_tables` → `resoperation.py` / `lloperation.py`
//!   - `concrete`        → `executor.py` + `floatobject.py` / `intobject.py`
//!   - `typed_trace`     → `pyjitpl.py` `opimpl_*` + `listobject.py` strategies
//!
//! The box/unbox/binop recording primitives are interpreter-agnostic and live
//! in `majit_metainterp::box_trace`; they are re-exported below so call sites
//! keep using `crate::<name>`.

mod concrete;
mod dispatch_tables;
mod typed_trace;

pub use concrete::*;
pub use dispatch_tables::*;
pub use majit_metainterp::box_trace::*;
pub use typed_trace::*;
