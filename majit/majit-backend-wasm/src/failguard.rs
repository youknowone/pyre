/// Guard failure descriptors and frame data for the wasm backend.
///
/// Simplified from CraneliftFailDescr — no bridge data, GC maps, or force tokens.
use std::cell::RefCell;
use std::sync::Arc;

use majit_ir::{Descr, DescrRef, FailDescr, Type};

/// Wasm-backend guard failure descriptor.
#[derive(Debug)]
pub struct WasmFailDescr {
    pub fail_index: u32,
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,
    /// `history.py:125 id(descr)` parity — when the optimizer
    /// (`store_final_boxes_in_guard` / `make_and_attach_done_descrs`)
    /// stamps a metainterp `ResumeGuardDescr` / `DoneWithThisFrame*` /
    /// `ExitFrameWithExceptionDescrRef` / `PropagateExceptionDescr` on
    /// `op.descr`, we keep it here so `get_latest_descr_arc` returns the
    /// canonical metainterp Arc (matching dynasm/cranelift).  `None`
    /// for synthetic backend-only descrs (`compile_bridge` placeholders,
    /// test scaffolds).
    pub meta_descr: Option<DescrRef>,
}

impl Descr for WasmFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }

    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for WasmFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }

    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }

    fn is_finish(&self) -> bool {
        self.is_finish
    }

    fn trace_id(&self) -> u64 {
        self.trace_id
    }
}

/// Wasm-backend dead frame data.
///
/// Stored inside `DeadFrame.data` after `execute_token` returns.
pub struct WasmFrameData {
    pub raw_values: Vec<i64>,
    pub fail_descr: Arc<WasmFailDescr>,
    /// Pending exception value captured by `execute_token` after the trace
    /// exited through a GuardNoException / GuardException (0 = none), surfaced
    /// via `grab_exc_value`.
    pub exc_value: i64,
}

/// Compiled wasm loop metadata, stored in `JitCellToken.compiled`.
pub struct CompiledWasmLoop {
    pub trace_id: u64,
    pub input_types: Vec<Type>,
    pub func_handle: u32,
    /// Guard/finish exit descriptors, indexed by the `fail_index` written into
    /// `frame[0]`. `compile_bridge` appends its bridge's descrs here (past the
    /// loop's own `[0, num_guards)` range) so `execute_token` resolves loop and
    /// chained-bridge exits through one array. `RefCell` because the append
    /// happens through the shared `&JitCellToken` the bridge attaches to; the
    /// wasm host is single-threaded so no cross-thread access occurs.
    pub fail_descrs: RefCell<Vec<Arc<WasmFailDescr>>>,
    pub num_inputs: usize,
    pub max_output_slots: usize,
    /// Number of Ref-typed values given a home slot in the frame's Ref-home
    /// region (`codegen::HOME_SLOT_BASE`). `execute_token` sizes the host
    /// frame to include this region and registers each home slot as a GC root.
    pub num_ref_homes: usize,
    /// Base address (shared linear memory) of this loop's per-guard bridge-slot
    /// cell array — one i32 per `fail_index`, `0` = no bridge. The trace's
    /// epilogue reads `cells[fail_index]` and `compile_bridge` writes a bridge's
    /// table slot here. `0` when the trace has no in-module dispatch (native, or
    /// a guardless / straight-line trace).
    pub bridge_cells_base: u32,
    /// Number of cells in the `bridge_cells_base` array = this loop's own guard
    /// count at compile time. A bridge attaches only to one of these original
    /// guards (`source_fail_index < num_guard_cells`); descrs appended past this
    /// range belong to already-chained bridges and have no cell of their own.
    pub num_guard_cells: usize,
    /// True when this is a peeled loop — there is real work (a preamble = the
    /// unrolled first iteration) before the loop's `LABEL`. A loop-closing
    /// bridge re-enters through the loop's table slot (the function entry), so
    /// for a peeled loop it re-runs the preamble against mid-loop state instead
    /// of resuming at the `LABEL`, never advancing the induction variable — an
    /// infinite loop. `compile_bridge` declines a loop-closing bridge into such
    /// a loop so the guard falls back to blackhole resume instead of livelocking.
    pub has_preamble: bool,
}
