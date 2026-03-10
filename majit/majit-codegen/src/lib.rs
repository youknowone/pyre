/// Backend abstraction trait for JIT code generation.
///
/// Translated from rpython/jit/backend/model.py (AbstractCPU).
/// The Backend trait is the contract between the JIT frontend (tracing + optimization)
/// and the code generation backend (Cranelift, etc.).
use majit_ir::{FailDescr, InputArg, Op, Type, Value};

/// Result of compiling a loop or bridge.
#[derive(Debug)]
pub struct AsmInfo {
    /// Start address of the generated code.
    pub code_addr: usize,
    /// Size of the generated code in bytes.
    pub code_size: usize,
}

/// Token identifying a compiled loop. Bridges are attached to this.
pub struct LoopToken {
    /// Unique number for this token.
    pub number: u64,
    /// Types of the input arguments.
    pub inputarg_types: Vec<Type>,
    /// Backend-specific compiled data.
    pub compiled: Option<Box<dyn std::any::Any + Send>>,
}

impl LoopToken {
    pub fn new(number: u64) -> Self {
        LoopToken {
            number,
            inputarg_types: Vec::new(),
            compiled: None,
        }
    }
}

impl std::fmt::Debug for LoopToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopToken")
            .field("number", &self.number)
            .finish()
    }
}

/// A "dead frame" — the state after JIT execution finishes or hits a guard.
///
/// The backend stores register/stack values here so the frontend can read them.
pub struct DeadFrame {
    /// Backend-specific frame data.
    pub data: Box<dyn std::any::Any + Send>,
}

/// The backend trait — implemented by Cranelift (or other code generators).
///
/// Mirrors rpython/jit/backend/model.py AbstractCPU.
pub trait Backend: Send {
    /// Compile a loop trace into native code.
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        token: &mut LoopToken,
    ) -> Result<AsmInfo, BackendError>;

    /// Compile a bridge (side exit path) and attach it to the loop.
    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        original_token: &LoopToken,
    ) -> Result<AsmInfo, BackendError>;

    /// Execute compiled code starting at the given token.
    fn execute_token(&self, token: &LoopToken, args: &[Value]) -> DeadFrame;

    /// Read the FailDescr from the last guard failure.
    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr;

    /// Read an integer value from a dead frame at the given index.
    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64;

    /// Read a float value from a dead frame.
    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64;

    /// Read a GC reference value from a dead frame.
    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> majit_ir::GcRef;

    /// Invalidate a compiled loop (e.g., due to GUARD_NOT_INVALIDATED).
    fn invalidate_loop(&self, token: &LoopToken);

    /// Redirect calls from one loop token to another (for CALL_ASSEMBLER).
    fn redirect_call_assembler(&self, _old: &LoopToken, _new: &LoopToken) {
        // Default: no-op
    }

    /// Free resources associated with a compiled loop.
    fn free_loop(&mut self, _token: &LoopToken) {
        // Default: no-op
    }
}

/// Errors from the backend.
#[derive(Debug)]
pub enum BackendError {
    /// Compilation failed.
    CompilationFailed(String),
    /// Unsupported operation.
    Unsupported(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::CompilationFailed(s) => write!(f, "compilation failed: {s}"),
            BackendError::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}

impl std::error::Error for BackendError {}
