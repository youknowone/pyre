/// Guard failure descriptors and frame data for the wasm backend.
///
/// Simplified from CraneliftFailDescr — no bridge data, GC maps, or force tokens.
use std::sync::Arc;

use majit_ir::{Descr, FailDescr, Type};

/// Wasm-backend guard failure descriptor.
#[derive(Debug)]
pub struct WasmFailDescr {
    pub fail_index: u32,
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,
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

    /// FINISH carries its one result in `fail_arg_types[0]` (or
    /// nothing for void). compile.py:626-656 parity.
    fn finish_result_type(&self) -> Type {
        self.fail_arg_types.first().copied().unwrap_or(Type::Void)
    }

    fn trace_id(&self) -> u64 {
        self.trace_id
    }

    fn handle_fail(&self, ctx: &mut dyn majit_ir::HandleFailContext) -> majit_ir::HandleFailResult {
        // finish → compile.py:626-656 `_DoneWithThisFrameDescr.handle_fail`;
        // else  → compile.py:701-717 `AbstractResumeGuardDescr.handle_fail`.
        majit_ir::dispatch_handle_fail(self, ctx)
    }
}

/// Wasm-backend dead frame data.
///
/// Stored inside `DeadFrame.data` after `execute_token` returns.
pub struct WasmFrameData {
    pub raw_values: Vec<i64>,
    pub fail_descr: Arc<WasmFailDescr>,
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
