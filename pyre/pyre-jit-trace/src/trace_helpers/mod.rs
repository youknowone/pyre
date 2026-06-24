//! Fixed trace helpers for `MIFrame` — hand-maintained Rust (not translator
//! output), the trace-time analogs of PyPy's hand-written RPython. Split by
//! PyPy's file boundaries:
//!   - `dispatch_tables` → `resoperation.py` / `lloperation.py`
//!   - `concrete`        → `executor.py` + `floatobject.py` / `intobject.py`
//!   - `trace_primitives`→ `pyjitpl.py` `_record_helper` / `history` boxing
//!   - `typed_trace`     → `pyjitpl.py` `opimpl_*` + `listobject.py` strategies

mod dispatch_tables;
mod concrete;
mod trace_primitives;
mod typed_trace;

pub use concrete::*;
pub use dispatch_tables::*;
pub use trace_primitives::*;
pub use typed_trace::*;
