/// Guard failure descriptors and frame data for the wasm backend.
///
/// Simplified from CraneliftFailDescr — no bridge data, GC maps, or force tokens.
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
    pub fail_descrs: Vec<Arc<WasmFailDescr>>,
    pub num_inputs: usize,
    pub max_output_slots: usize,
}
