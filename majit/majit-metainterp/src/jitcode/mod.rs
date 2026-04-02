mod codewriter;
mod frame;
pub(crate) mod machine;

pub use codewriter::JitCodeBuilder;
pub use machine::*;

use majit_ir::OpCode;

pub(crate) const BC_LOAD_CONST_I: u8 = 1;
pub(crate) const BC_POP_I: u8 = 2;
pub(crate) const BC_PEEK_I: u8 = 3;
pub(crate) const BC_PUSH_I: u8 = 4;
pub(crate) const BC_POP_DISCARD: u8 = 5;
pub(crate) const BC_DUP_STACK: u8 = 6;
pub(crate) const BC_SWAP_STACK: u8 = 7;
pub(crate) const BC_RECORD_BINOP_I: u8 = 8;
pub(crate) const BC_RECORD_UNARY_I: u8 = 9;
pub(crate) const BC_REQUIRE_STACK: u8 = 10;
pub(crate) const BC_BRANCH_ZERO: u8 = 11;
pub(crate) const BC_JUMP_TARGET: u8 = 12;
pub(crate) const BC_ABORT: u8 = 13;
pub(crate) const BC_ABORT_PERMANENT: u8 = 14;
pub(crate) const BC_BRANCH_REG_ZERO: u8 = 15;
pub(crate) const BC_JUMP: u8 = 16;
pub(crate) const BC_INLINE_CALL: u8 = 17;
pub(crate) const BC_RESIDUAL_CALL_VOID: u8 = 18;
pub(crate) const BC_SET_SELECTED: u8 = 19;
pub(crate) const BC_PUSH_TO: u8 = 20;
pub(crate) const BC_MOVE_I: u8 = 21;
pub(crate) const BC_CALL_INT: u8 = 22;
pub(crate) const BC_CALL_PURE_INT: u8 = 23;
// Ref-typed bytecodes
pub(crate) const BC_LOAD_CONST_R: u8 = 24;
pub(crate) const BC_POP_R: u8 = 25;
pub(crate) const BC_PUSH_R: u8 = 26;
pub(crate) const BC_MOVE_R: u8 = 27;
pub(crate) const BC_CALL_REF: u8 = 28;
pub(crate) const BC_CALL_PURE_REF: u8 = 29;
// Float-typed bytecodes
pub(crate) const BC_LOAD_CONST_F: u8 = 30;
pub(crate) const BC_POP_F: u8 = 31;
pub(crate) const BC_PUSH_F: u8 = 32;
pub(crate) const BC_MOVE_F: u8 = 33;
pub(crate) const BC_CALL_FLOAT: u8 = 34;
pub(crate) const BC_CALL_PURE_FLOAT: u8 = 35;
pub(crate) const BC_RECORD_BINOP_F: u8 = 36;
pub(crate) const BC_RECORD_UNARY_F: u8 = 37;
pub(crate) const BC_CALL_MAY_FORCE_INT: u8 = 38;
pub(crate) const BC_CALL_MAY_FORCE_REF: u8 = 39;
pub(crate) const BC_CALL_MAY_FORCE_FLOAT: u8 = 40;
pub(crate) const BC_CALL_MAY_FORCE_VOID: u8 = 41;
pub(crate) const BC_CALL_RELEASE_GIL_INT: u8 = 42;
pub(crate) const BC_CALL_RELEASE_GIL_REF: u8 = 43;
pub(crate) const BC_CALL_RELEASE_GIL_FLOAT: u8 = 44;
pub(crate) const BC_CALL_RELEASE_GIL_VOID: u8 = 45;
pub(crate) const BC_CALL_LOOPINVARIANT_INT: u8 = 46;
pub(crate) const BC_CALL_LOOPINVARIANT_REF: u8 = 47;
pub(crate) const BC_CALL_LOOPINVARIANT_FLOAT: u8 = 48;
pub(crate) const BC_CALL_LOOPINVARIANT_VOID: u8 = 49;
pub(crate) const BC_CALL_ASSEMBLER_INT: u8 = 50;
pub(crate) const BC_CALL_ASSEMBLER_REF: u8 = 51;
pub(crate) const BC_CALL_ASSEMBLER_FLOAT: u8 = 52;
pub(crate) const BC_CALL_ASSEMBLER_VOID: u8 = 53;
pub(crate) const BC_COPY_FROM_BOTTOM: u8 = 54;
pub(crate) const BC_STORE_DOWN: u8 = 55;
pub(crate) const BC_LOAD_STATE_FIELD: u8 = 56;
pub(crate) const BC_STORE_STATE_FIELD: u8 = 57;
pub(crate) const BC_LOAD_STATE_ARRAY: u8 = 58;
pub(crate) const BC_STORE_STATE_ARRAY: u8 = 59;
pub(crate) const BC_LOAD_STATE_VARRAY: u8 = 60;
pub(crate) const BC_STORE_STATE_VARRAY: u8 = 61;
pub(crate) const BC_GETFIELD_VABLE_I: u8 = 62;
pub(crate) const BC_GETFIELD_VABLE_R: u8 = 63;
pub(crate) const BC_GETFIELD_VABLE_F: u8 = 64;
pub(crate) const BC_SETFIELD_VABLE_I: u8 = 65;
pub(crate) const BC_SETFIELD_VABLE_R: u8 = 66;
pub(crate) const BC_SETFIELD_VABLE_F: u8 = 67;
pub(crate) const BC_GETARRAYITEM_VABLE_I: u8 = 68;
pub(crate) const BC_GETARRAYITEM_VABLE_R: u8 = 69;
pub(crate) const BC_GETARRAYITEM_VABLE_F: u8 = 70;
pub(crate) const BC_SETARRAYITEM_VABLE_I: u8 = 71;
pub(crate) const BC_SETARRAYITEM_VABLE_R: u8 = 72;
pub(crate) const BC_SETARRAYITEM_VABLE_F: u8 = 73;
pub(crate) const BC_ARRAYLEN_VABLE: u8 = 74;
pub(crate) const BC_HINT_FORCE_VIRTUALIZABLE: u8 = 75;
/// RPython bhimpl_ref_return: callee returns a ref value.
pub(crate) const BC_REF_RETURN: u8 = 76;
/// blackhole.py bhimpl_raise: raise an exception from a ref register.
pub(crate) const BC_RAISE: u8 = 77;
/// blackhole.py bhimpl_reraise: re-raise exception_last_value.
pub(crate) const BC_RERAISE: u8 = 78;
pub(crate) const MAX_HOST_CALL_ARITY: usize = 16;

/// GC liveness metadata at a specific bytecode PC.
///
/// RPython liveness.py: `[len_i][len_r][len_f][bitset_i][bitset_r][bitset_f]`
/// Tracks which registers of each type (int/ref/float) are live at a given PC.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LivenessInfo {
    pub pc: u16,
    /// Live integer register indices at this PC.
    pub live_i_regs: Vec<u16>,
    /// Live reference register indices at this PC.
    pub live_r_regs: Vec<u16>,
    /// Live float register indices at this PC.
    pub live_f_regs: Vec<u16>,
}

impl LivenessInfo {
    /// jitcode.py:146-167 enumerate_vars parity.
    /// Iterates live registers in typed order: int, then ref, then float.
    /// Both encoder (get_list_of_active_boxes) and decoder (consume_one_section)
    /// MUST use this same iteration order.
    pub fn enumerate_vars(
        &self,
        mut callback_i: impl FnMut(u16),
        mut callback_r: impl FnMut(u16),
        mut callback_f: impl FnMut(u16),
    ) {
        for &idx in &self.live_i_regs {
            callback_i(idx);
        }
        for &idx in &self.live_r_regs {
            callback_r(idx);
        }
        for &idx in &self.live_f_regs {
            callback_f(idx);
        }
    }

    /// Total number of live registers across all typed banks.
    pub fn total_live(&self) -> usize {
        self.live_i_regs.len() + self.live_r_regs.len() + self.live_f_regs.len()
    }
}

/// Serialized interpreter step description, mirroring RPython's `JitCode`
/// at a much smaller subset for the current proc-macro tracing surface.
#[derive(Clone, Debug, Default)]
pub struct JitCode {
    /// Encoded bytecode stream.
    pub code: Vec<u8>,
    /// Number of registers by kind: int/ref/float.
    pub num_regs: [u16; 3],
    /// RPython: `jitcode.constants_i` — integer constant pool.
    pub constants_i: Vec<i64>,
    /// RPython: `jitcode.constants_r` — reference constant pool.
    pub constants_r: Vec<i64>,
    /// RPython: `jitcode.constants_f` — float constant pool (bits as i64).
    pub constants_f: Vec<i64>,
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
    /// RPython: `jitcode.jitdriver_sd is not None` — true if this jitcode
    /// is a portal (entry point for a jit driver). Used by
    /// `_handle_jitexception` to find the portal level in the BH chain.
    pub is_portal: bool,
    /// blackhole.py handle_exception_in_frame: exception handler table.
    /// Pre-computed from Python's code.exceptiontable during compilation.
    pub exception_handlers: Vec<JitExceptionHandler>,
    /// Reverse PC map: sorted (jitcode_pc, py_pc) pairs for binary search.
    /// Used by handle_exception_in_frame to determine faulting Python PC (lasti).
    pub jit_to_py_pc: Vec<(usize, usize)>,
    /// Forward PC map: py_pc → jitcode_pc. Used by get_list_of_active_boxes
    /// to look up LivenessInfo at the current Python bytecode position.
    /// RPython: pc → offset into all_liveness (embedded in jitcode bytecodes).
    pub py_to_jit_pc: Vec<usize>,
    /// True if BC_ABORT was emitted (not BC_ABORT_PERMANENT).
    /// Set by JitCodeBuilder::abort(), not by raw byte scan.
    pub has_abort: bool,
}

/// blackhole.py catch_exception: pre-computed exception handler for a JitCode PC range.
#[derive(Clone, Debug)]
pub struct JitExceptionHandler {
    /// Start JitCode PC (inclusive).
    pub jit_start: usize,
    /// End JitCode PC (exclusive).
    pub jit_end: usize,
    /// Handler target JitCode PC.
    pub jit_target: usize,
    /// Stack depth at handler entry (runtime stack items to keep).
    pub stack_depth: u16,
    /// Whether to push lasti (Python PC) before exception object.
    pub push_lasti: bool,
    /// Raw Python PC for lasti (boxed at runtime via box_int_fn).
    pub lasti_value: i64,
    /// fn_ptr index for box_int_fn (to box lasti at dispatch time).
    pub box_int_fn_idx: u16,
}

// -- RPython jitcode.py parity methods --

impl JitCode {
    /// blackhole.py:396 handle_exception_in_frame: find exception handler
    /// for the given JitCode PC. Returns the first matching handler.
    pub fn find_exception_handler(&self, jitcode_pc: usize) -> Option<&JitExceptionHandler> {
        self.exception_handlers
            .iter()
            .find(|h| jitcode_pc >= h.jit_start && jitcode_pc < h.jit_end)
    }

    /// Reverse lookup: JitCode PC → Python PC.
    /// Binary search in jit_to_py_pc for the largest jit_pc <= target.
    pub fn jit_pc_to_py_pc(&self, jit_pc: usize) -> i64 {
        match self
            .jit_to_py_pc
            .binary_search_by_key(&jit_pc, |&(jp, _)| jp)
        {
            Ok(idx) => self.jit_to_py_pc[idx].1 as i64,
            Err(idx) if idx > 0 => self.jit_to_py_pc[idx - 1].1 as i64,
            _ => 0,
        }
    }

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

    pub(crate) fn decode(byte: u8) -> Self {
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
