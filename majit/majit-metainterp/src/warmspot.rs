//! RPython parity module for `rpython/jit/metainterp/warmspot.py`.
//!
//! PyPy keeps the translation-time warmspot bootstrap in one Python module.
//! Pyre's Rust port splits that lifecycle across the static metainterp data,
//! jitdriver metadata, warmstate, compile helpers, and the `pyre-jit` portal
//! boundary. This module is the parity namespace that re-exports those pieces
//! under the upstream module name without introducing a second implementation.

pub use crate::jitdriver::{
    DeclarativeJitDriver, JitDriver, JitDriverStaticData, TraceContinuationSuspendGuard,
};
pub use crate::memmgr::MemoryManager;
pub use crate::pyjitpl::{
    BackEdgeAction, CompileOutcome, DoneWithThisFrame, JitHooks, JitStats, MetaInterp,
    MetaInterpGlobalData, MetaInterpStaticData,
};
pub use crate::warmstate::{
    BaseJitCell, BaseJitCellState, CellJitState, HotResult, WarmEnterState, jc_flags,
};
