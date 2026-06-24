//! RPython parity module for `rpython/jit/metainterp/optimizeopt/rawbuffer.py`.
//!
//! The storage lives in `majit_ir` because `PtrInfo` owns raw-buffer state and
//! is shared across metainterp optimizer crates. This module preserves the
//! upstream `optimizeopt.rawbuffer` import path.

pub use majit_ir::rawbuffer::{InvalidRawOperation, InvalidRawRead, InvalidRawWrite, RawBuffer};
