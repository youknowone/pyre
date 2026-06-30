/// Guard failure descriptors and frame data for the wasm backend.
///
/// Simplified from CraneliftFailDescr — no bridge data, GC maps, or force tokens.
use std::cell::{Cell, RefCell};
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
    /// True when this is a peeled loop (`codegen::is_resumable_peeled`) — there
    /// is real work (a preamble = the unrolled first iteration) before the last
    /// `LABEL`, single- or multi-label. Such a loop carries the resume-at-LABEL
    /// preamble-skip dispatch and resumes at the LAST label (where the `loop`
    /// is). A loop-closing bridge re-enters through the loop's table slot (the
    /// function entry); for a peeled loop, re-running the preamble against
    /// mid-loop state would never advance the induction variable — an infinite
    /// loop. `compile_bridge` therefore declines a loop-closing bridge into a
    /// peeled loop UNLESS it resumes at that last label: always so for a
    /// single-label source (`is_single_label_peeled`), and for a multi-label
    /// source only when the bridge's JUMP targets `last_label_block_id`.
    pub has_preamble: bool,
    /// True when this peeled loop has exactly one `LABEL`
    /// (`codegen::is_single_label_peeled`). A loop-closing bridge into such a
    /// loop always targets that sole label, so `compile_bridge` accepts it
    /// without recovering the JUMP's target ordinal.
    pub is_single_label_peeled: bool,
    /// `label_block_id` of the LAST `LABEL` (= label_count − 1), the one carrying
    /// the wasm `loop`. A loop-closing bridge into a multi-label source is
    /// accepted only when its JUMP descr's recovered `label_block_id` equals this
    /// — i.e. the bridge resumes at the label the resume dispatch lands on.
    pub last_label_block_id: u32,
    /// Argument count of the LAST `LABEL`. The accept-condition declines a bridge
    /// whose closing JUMP arity differs from this, since the resume loader reads
    /// exactly this many positional frame slots (an arity mismatch would resume
    /// with stale/missing induction values).
    pub last_label_num_args: usize,
    /// `(source_fail_index, start, count)` ranges into `fail_descrs` for each
    /// chained bridge `compile_bridge` appended (lib.rs extend site). Lets
    /// `compiled_bridge_fail_descr_layouts` / `store_bridge_guard_hashes` map a
    /// source guard back to its bridge's appended descr slice — the wasm analog
    /// of dynasm's `lookup_bridge_addr` (runner.rs). Recorded in lockstep with
    /// the `extend`, inside the same `borrow_mut` critical section.
    pub bridge_descr_ranges: RefCell<Vec<(u32, usize, usize)>>,
    /// Owns this loop's per-guard bridge-slot cell array so it is freed on
    /// `Drop`; `bridge_cells_base` aliases its heap address (stable across the
    /// struct move). `None` when the trace has no in-module dispatch.
    pub _bridge_cells_owner: Option<Box<[u32]>>,
    /// Owns the cell arrays of every bridge chained onto this loop. A bridge
    /// module lives as long as the source loop it attaches to, so its cells are
    /// freed when this loop drops. Appended by `compile_bridge`.
    pub _bridge_owned_cells: RefCell<Vec<Box<[u32]>>>,
    /// Max `num_ref_homes` over the self-recursive `CallAssemblerR` bridges
    /// (`PYRE_WASM_CA`) chained onto this loop, or 0 when there are none. Such a
    /// bridge runs in the host entry frame `F0` for the outermost call, so
    /// `execute_token` must size `F0` (and register its GC roots) for the LARGER
    /// of the loop's own homes and this — the bridge's home writes would
    /// otherwise overflow a loop-sized `F0`. Set by `compile_bridge` when it
    /// accepts a CA bridge; `Cell` because the source token is shared (`&`) and
    /// the wasm host is single-threaded.
    pub ca_bridge_ref_homes: Cell<usize>,
}
