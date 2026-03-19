mod codewriter;
mod frame;
mod machine;

pub use codewriter::JitCodeBuilder;
pub use machine::*;

use majit_ir::OpCode;

const BC_LOAD_CONST_I: u8 = 1;
const BC_POP_I: u8 = 2;
const BC_PEEK_I: u8 = 3;
const BC_PUSH_I: u8 = 4;
const BC_POP_DISCARD: u8 = 5;
const BC_DUP_STACK: u8 = 6;
const BC_SWAP_STACK: u8 = 7;
const BC_RECORD_BINOP_I: u8 = 8;
const BC_RECORD_UNARY_I: u8 = 9;
const BC_REQUIRE_STACK: u8 = 10;
const BC_BRANCH_ZERO: u8 = 11;
const BC_JUMP_TARGET: u8 = 12;
const BC_ABORT: u8 = 13;
const BC_ABORT_PERMANENT: u8 = 14;
const BC_BRANCH_REG_ZERO: u8 = 15;
const BC_JUMP: u8 = 16;
const BC_INLINE_CALL: u8 = 17;
const BC_RESIDUAL_CALL_VOID: u8 = 18;
const BC_SET_SELECTED: u8 = 19;
const BC_PUSH_TO: u8 = 20;
const BC_MOVE_I: u8 = 21;
const BC_CALL_INT: u8 = 22;
const BC_CALL_PURE_INT: u8 = 23;
// Ref-typed bytecodes
const BC_LOAD_CONST_R: u8 = 24;
const BC_POP_R: u8 = 25;
const BC_PUSH_R: u8 = 26;
const BC_MOVE_R: u8 = 27;
const BC_CALL_REF: u8 = 28;
const BC_CALL_PURE_REF: u8 = 29;
// Float-typed bytecodes
const BC_LOAD_CONST_F: u8 = 30;
const BC_POP_F: u8 = 31;
const BC_PUSH_F: u8 = 32;
const BC_MOVE_F: u8 = 33;
const BC_CALL_FLOAT: u8 = 34;
const BC_CALL_PURE_FLOAT: u8 = 35;
const BC_RECORD_BINOP_F: u8 = 36;
const BC_RECORD_UNARY_F: u8 = 37;
const BC_CALL_MAY_FORCE_INT: u8 = 38;
const BC_CALL_MAY_FORCE_REF: u8 = 39;
const BC_CALL_MAY_FORCE_FLOAT: u8 = 40;
const BC_CALL_MAY_FORCE_VOID: u8 = 41;
const BC_CALL_RELEASE_GIL_INT: u8 = 42;
const BC_CALL_RELEASE_GIL_REF: u8 = 43;
const BC_CALL_RELEASE_GIL_FLOAT: u8 = 44;
const BC_CALL_RELEASE_GIL_VOID: u8 = 45;
const BC_CALL_LOOPINVARIANT_INT: u8 = 46;
const BC_CALL_LOOPINVARIANT_REF: u8 = 47;
const BC_CALL_LOOPINVARIANT_FLOAT: u8 = 48;
const BC_CALL_LOOPINVARIANT_VOID: u8 = 49;
const BC_CALL_ASSEMBLER_INT: u8 = 50;
const BC_CALL_ASSEMBLER_REF: u8 = 51;
const BC_CALL_ASSEMBLER_FLOAT: u8 = 52;
const BC_CALL_ASSEMBLER_VOID: u8 = 53;
const BC_COPY_FROM_BOTTOM: u8 = 54;
const BC_STORE_DOWN: u8 = 55;
const BC_LOAD_STATE_FIELD: u8 = 56;
const BC_STORE_STATE_FIELD: u8 = 57;
const BC_LOAD_STATE_ARRAY: u8 = 58;
const BC_STORE_STATE_ARRAY: u8 = 59;
const BC_LOAD_STATE_VARRAY: u8 = 60;
const BC_STORE_STATE_VARRAY: u8 = 61;
const BC_GETFIELD_VABLE_I: u8 = 62;
const BC_GETFIELD_VABLE_R: u8 = 63;
const BC_GETFIELD_VABLE_F: u8 = 64;
const BC_SETFIELD_VABLE_I: u8 = 65;
const BC_SETFIELD_VABLE_R: u8 = 66;
const BC_SETFIELD_VABLE_F: u8 = 67;
const BC_GETARRAYITEM_VABLE_I: u8 = 68;
const BC_GETARRAYITEM_VABLE_R: u8 = 69;
const BC_GETARRAYITEM_VABLE_F: u8 = 70;
const BC_SETARRAYITEM_VABLE_I: u8 = 71;
const BC_SETARRAYITEM_VABLE_R: u8 = 72;
const BC_SETARRAYITEM_VABLE_F: u8 = 73;
const BC_ARRAYLEN_VABLE: u8 = 74;
const BC_HINT_FORCE_VIRTUALIZABLE: u8 = 75;
const MAX_HOST_CALL_ARITY: usize = 16;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LivenessInfo {
    pub pc: u16,
    pub live_i_regs: Vec<u16>,
}

/// Serialized interpreter step description, mirroring RPython's `JitCode`
/// at a much smaller subset for the current proc-macro tracing surface.
#[derive(Clone, Debug, Default)]
pub struct JitCode {
    /// Encoded bytecode stream.
    pub code: Vec<u8>,
    /// Number of registers by kind: int/ref/float.
    pub num_regs: [u16; 3],
    /// Integer constant pool.
    pub constants_i: Vec<i64>,
    /// Liveness metadata for GC / deopt expansion.
    pub liveness: Vec<LivenessInfo>,
    /// Pool of majit IR opcodes referenced from the bytecode stream.
    pub opcodes: Vec<OpCode>,
    /// Sub-JitCodes for `inline_call` targets (compound methods).
    pub sub_jitcodes: Vec<JitCode>,
    /// Function pointers for `residual_call` targets (I/O shims, external calls).
    pub fn_ptrs: Vec<JitCallTarget>,
    /// CALL_ASSEMBLER targets keyed by loop token number plus a concrete hook.
    assembler_targets: Vec<JitCallAssemblerTarget>,
}

// -- RPython jitcode.py parity methods --

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

    /// RPython: `JitCode.follow_jump(position)` -- follow a label at position.
    pub fn follow_jump(&self, position: usize) -> usize {
        if position < 2 || position - 2 + 1 >= self.code.len() {
            return 0;
        }
        let pos = position - 2;
        (self.code[pos] as usize) | ((self.code[pos + 1] as usize) << 8)
    }

    /// RPython: `JitCode.dump()`
    pub fn dump(&self) -> String {
        format!(
            "<JitCode: {} bytes, {} int regs, {} consts>",
            self.code.len(),
            self.num_regs[0],
            self.constants_i.len()
        )
    }
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct JitCallAssemblerTarget {
    token_number: u64,
    concrete_ptr: *const (),
}

impl JitCallAssemblerTarget {
    fn new(token_number: u64, concrete_ptr: *const ()) -> Self {
        Self {
            token_number,
            concrete_ptr,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitArgKind {
    Int = 0,
    Ref = 1,
    Float = 2,
}

impl JitArgKind {
    fn encode(self) -> u8 {
        self as u8
    }

    fn decode(byte: u8) -> Self {
        match byte {
            0 => Self::Int,
            1 => Self::Ref,
            2 => Self::Float,
            other => panic!("unknown jitcode arg kind {other}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCallArg {
    pub kind: JitArgKind,
    pub reg: u16,
}

impl JitCallArg {
    pub fn int(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Int,
            reg,
        }
    }

    pub fn reference(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Ref,
            reg,
        }
    }

    pub fn float(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Float,
            reg,
        }
    }
}

// SAFETY: call targets are function pointers to immutable code.
unsafe impl Send for JitCode {}
unsafe impl Sync for JitCode {}

// MIFrame and MIFrameStack are in frame.rs (RPython pyjitpl.py parity).
pub use frame::{MIFrame, MIFrameStack};

pub(crate) fn read_u8(code: &[u8], cursor: &mut usize) -> u8 {
    let value = *code.get(*cursor).expect("truncated jitcode");
    *cursor += 1;
    value
}

pub(crate) fn read_u16(code: &[u8], cursor: &mut usize) -> u16 {
    let lo = *code.get(*cursor).expect("truncated jitcode");
    let hi = *code.get(*cursor + 1).expect("truncated jitcode");
    *cursor += 2;
    u16::from_le_bytes([lo, hi])
}
