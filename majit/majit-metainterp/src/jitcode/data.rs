/// JitCode data container — RPython jitcode.py parity.
///
/// In RPython, JitCode is a pure data container (167 lines) holding:
/// - name, code (bytecode), constants_i/r/f, num_regs_i/r/f
/// - fnaddr, calldescr, jitdriver_sd
///
/// In majit, JitCode carries the same data plus:
/// - opcodes pool (majit IR opcodes referenced by bytecode)
/// - sub_jitcodes (inline call targets)
/// - fn_ptrs (residual call targets)
/// - assembler_targets (CALL_ASSEMBLER loop tokens)
use majit_ir::OpCode;

/// GC liveness metadata at a specific bytecode PC.
#[derive(Clone, Debug, Default)]
pub struct LivenessInfo {
    pub pc: u16,
    pub live_i_regs: Vec<u16>,
}

/// Serialized interpreter step description.
///
/// RPython jitcode.py: `class JitCode(AbstractDescr)`
#[derive(Clone, Debug, Default)]
pub struct JitCode {
    // ── RPython jitcode.py fields ──

    /// RPython: `self.name` — symbolic name for debugging.
    pub name: String,
    /// RPython: `self.code` — encoded bytecode stream.
    pub code: Vec<u8>,
    /// Number of registers by kind: [int, ref, float].
    /// RPython: `self.c_num_regs_i/r/f`
    pub num_regs: [u16; 3],
    /// RPython: `self.constants_i` — integer constant pool.
    pub constants_i: Vec<i64>,
    /// RPython: `self.constants_r` — reference constant pool.
    pub constants_r: Vec<u64>,
    /// RPython: `self.constants_f` — float constant pool.
    pub constants_f: Vec<f64>,

    // ── majit extensions ──

    /// Liveness metadata for GC / deopt expansion.
    pub liveness: Vec<LivenessInfo>,
    /// Pool of majit IR opcodes referenced from the bytecode stream.
    pub opcodes: Vec<OpCode>,
    /// Sub-JitCodes for `inline_call` targets (compound methods).
    pub sub_jitcodes: Vec<JitCode>,
    /// Function pointers for `residual_call` targets.
    pub fn_ptrs: Vec<JitCallTarget>,
    /// CALL_ASSEMBLER targets keyed by loop token number.
    pub(crate) assembler_targets: Vec<JitCallAssemblerTarget>,
}

impl JitCode {
    /// RPython: `JitCode.num_regs_i()`
    pub fn num_regs_i(&self) -> u16 {
        self.num_regs[0]
    }

    /// RPython: `JitCode.num_regs_r()`
    pub fn num_regs_r(&self) -> u16 {
        self.num_regs[1]
    }

    /// RPython: `JitCode.num_regs_f()`
    pub fn num_regs_f(&self) -> u16 {
        self.num_regs[2]
    }

    /// RPython: `JitCode.num_regs_and_consts_i()`
    pub fn num_regs_and_consts_i(&self) -> usize {
        self.num_regs[0] as usize + self.constants_i.len()
    }

    /// RPython: `JitCode.num_regs_and_consts_r()`
    pub fn num_regs_and_consts_r(&self) -> usize {
        self.num_regs[1] as usize + self.constants_r.len()
    }

    /// RPython: `JitCode.num_regs_and_consts_f()`
    pub fn num_regs_and_consts_f(&self) -> usize {
        self.num_regs[2] as usize + self.constants_f.len()
    }

    /// RPython: `JitCode.follow_jump(position)` — follow a label at position.
    pub fn follow_jump(&self, position: usize) -> usize {
        if position < 2 {
            return 0;
        }
        let pos = position - 2;
        if pos + 1 >= self.code.len() {
            return 0;
        }
        let label = (self.code[pos] as usize) | ((self.code[pos + 1] as usize) << 8);
        label
    }

    /// RPython: `JitCode.dump()` — debug representation.
    pub fn dump(&self) -> String {
        format!(
            "<JitCode '{}': {} bytes, {} int regs, {} consts>",
            self.name,
            self.code.len(),
            self.num_regs[0],
            self.constants_i.len()
        )
    }

    /// RPython: `JitCode.__repr__()`
    pub fn repr(&self) -> String {
        format!("<JitCode '{}'>", self.name)
    }
}

/// Function pointer pair for residual calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCallTarget {
    pub trace_ptr: *const (),
    pub concrete_ptr: *const (),
}

impl JitCallTarget {
    pub fn new(trace_ptr: *const (), concrete_ptr: *const ()) -> Self {
        Self {
            trace_ptr,
            concrete_ptr,
        }
    }
}

unsafe impl Send for JitCallTarget {}
unsafe impl Sync for JitCallTarget {}

/// CALL_ASSEMBLER target: loop token + concrete function pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct JitCallAssemblerTarget {
    pub(crate) token_number: u64,
    pub(crate) concrete_ptr: *const (),
}

impl JitCallAssemblerTarget {
    pub(crate) fn new(token_number: u64, concrete_ptr: *const ()) -> Self {
        Self {
            token_number,
            concrete_ptr,
        }
    }
}

unsafe impl Send for JitCallAssemblerTarget {}
unsafe impl Sync for JitCallAssemblerTarget {}
