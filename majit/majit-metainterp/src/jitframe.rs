//! JitFrame — shim re-exporting the canonical definition from
//! `majit_backend::jitframe`. The type lives in the `majit-backend`
//! crate so that all backends (cranelift, dynasm) and metainterp can
//! share a single upstream-aligned implementation of
//! `rpython/jit/backend/llsupport/jitframe.py`.

pub use majit_backend::jitframe::*;
