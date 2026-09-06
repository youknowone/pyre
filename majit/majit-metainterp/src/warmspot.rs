//! RPython parity module for `rpython/jit/metainterp/warmspot.py`.
//!
//! PyPy keeps the translation-time warmspot bootstrap in one Python module.
//! Pyre's Rust port splits that lifecycle across the static metainterp data,
//! jitdriver metadata, warmstate, compile helpers, and the `pyre-jit` portal
//! boundary. This module is the parity namespace that re-exports those pieces
//! under the upstream module name without introducing a second implementation.
//!
//! `WarmRunnerDesc.apply_jit` graph rewrites have no mutable interpreter-graph
//! stage here (design.md §3.7 A1). Their source-level ports:
//! - `split_graph_and_record_jitdriver` — `register_configured_jitdrivers`
//!   (autoreds run `autodetect_jit_markers_redvars` first)
//! - `rewrite_jit_merge_point` — `_unpackiterable_unknown_length` returns
//!   `unpack_portal_runner`; jd0 is `eval_loop_jit` / `ll_portal_runner_shim`
//! - `rewrite_can_enter_jits` — `can_enter_jit` / `unpack_merge_point` bodies
//! - `rewrite_set_param_and_get_stats` — `set_jit_param` hook
//! - `rewrite_force_virtual` — `force_pyframe_vref`
//! - `rewrite_force_quasi_immutable` — `jtransform` + `do_force_quasi_immutable`
//! - `rewrite_jitcell_accesses` — `WarmEnterState` methods
//! - `make_driverhook_graphs` — `get_unique_id` / `get_printable_location`
//! - `inline_inlineable_portals` / `prejit_optimizations` / `add_finish` /
//!   `create_jit_entry_points` — no `@jitdriver.inline` sites, no backendopt
//!   pass over interpreter graphs, no translated finish callback

pub use crate::jitdriver::{
    DeclarativeJitDriver, JitDriver, JitDriverStaticData, TraceContinuationSuspendGuard,
};
pub use crate::memmgr::MemoryManager;
pub use crate::pyjitpl::{
    BackEdgeAction, CompileOutcome, DoneWithThisFrame, JitHooks, JitStats, MetaInterp,
    MetaInterpGlobalData, MetaInterpStaticData,
};
pub use crate::warmstate::{
    BaseJitCell, BaseJitCellState, CellJitState, HotResult, JcFlags, WarmEnterState,
};
