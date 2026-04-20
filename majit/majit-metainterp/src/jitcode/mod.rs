mod assembler;

pub use assembler::JitCodeBuilder;

use majit_ir::OpCode;

pub(crate) const BC_LOAD_CONST_I: u8 = 1;
pub(crate) const BC_RECORD_BINOP_I: u8 = 8;
pub(crate) const BC_RECORD_UNARY_I: u8 = 9;
pub(crate) const BC_LOOP_HEADER: u8 = 12;
pub(crate) const BC_ABORT: u8 = 13;
pub(crate) const BC_ABORT_PERMANENT: u8 = 14;
pub(crate) const BC_BRANCH_REG_ZERO: u8 = 15;
pub(crate) const BC_JUMP: u8 = 16;
pub(crate) const BC_INLINE_CALL: u8 = 17;
pub(crate) const BC_RESIDUAL_CALL_VOID: u8 = 18;
pub(crate) const BC_MOVE_I: u8 = 21;
pub(crate) const BC_CALL_INT: u8 = 22;
pub(crate) const BC_CALL_PURE_INT: u8 = 23;
// Ref-typed bytecodes
pub(crate) const BC_LOAD_CONST_R: u8 = 24;
pub(crate) const BC_MOVE_R: u8 = 27;
pub(crate) const BC_CALL_REF: u8 = 28;
pub(crate) const BC_CALL_PURE_REF: u8 = 29;
// Float-typed bytecodes
pub(crate) const BC_LOAD_CONST_F: u8 = 30;
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
// RPython jtransform.py:1685 — conditional_call_ir_v
pub(crate) const BC_COND_CALL_VOID: u8 = 79;
// RPython jtransform.py:1687 — conditional_call_value_ir_i / conditional_call_value_ir_r
pub(crate) const BC_COND_CALL_VALUE_INT: u8 = 80;
pub(crate) const BC_COND_CALL_VALUE_REF: u8 = 81;
// RPython jtransform.py:292 — record_known_result_i_ir_v / record_known_result_r_ir_v
pub(crate) const BC_RECORD_KNOWN_RESULT_INT: u8 = 82;
pub(crate) const BC_RECORD_KNOWN_RESULT_REF: u8 = 83;
/// pyjitpl.py opimpl_int_guard_value: promote int to constant via GUARD_VALUE.
pub(crate) const BC_INT_GUARD_VALUE: u8 = 84;
/// pyjitpl.py opimpl_ref_guard_value: promote ref to constant via GUARD_VALUE.
pub(crate) const BC_REF_GUARD_VALUE: u8 = 85;
/// pyjitpl.py opimpl_float_guard_value: promote float to constant via GUARD_VALUE.
pub(crate) const BC_FLOAT_GUARD_VALUE: u8 = 86;
/// blackhole.py:1066 bhimpl_jit_merge_point: portal merge point marker.
pub(crate) const BC_JIT_MERGE_POINT: u8 = 87;
pub const BC_LIVE: u8 = 88;
pub(crate) const BC_CATCH_EXCEPTION: u8 = 89;
pub(crate) const BC_LAST_EXC_VALUE: u8 = 90;
/// blackhole.py bhimpl_rvmprof_code: rvmprof enter/leave marker.
pub(crate) const BC_RVMPROF_CODE: u8 = 91;

// RPython jtransform.py:196 `optimize_goto_if_not` fuses
// `v = int_lt(x, y); exitswitch = v` into
// `exitswitch = ('int_lt', x, y)`, emitted by flatten.py:247-250 as
// the jitcode op `goto_if_not_int_lt`. blackhole.py:864-944 consumes
// the fused form with dedicated bhimpls.
//
// majit currently reserves one `BC_GOTO_IF_NOT_*` per RPython opname
// variant; the 'c' short-const argcode (assembler.py:312 `USE_C_FORM`)
// is not yet supported in the pyre JitCodeBuilder so only the canonical
// `iiL` / `ffL` / `rrL` forms get a BC_* allocation here.
pub(crate) const BC_GOTO_IF_NOT_INT_LT: u8 = 92;
pub(crate) const BC_GOTO_IF_NOT_INT_LE: u8 = 93;
pub(crate) const BC_GOTO_IF_NOT_INT_EQ: u8 = 94;
pub(crate) const BC_GOTO_IF_NOT_INT_NE: u8 = 95;
pub(crate) const BC_GOTO_IF_NOT_INT_GT: u8 = 96;
pub(crate) const BC_GOTO_IF_NOT_INT_GE: u8 = 97;
pub(crate) const BC_GOTO_IF_NOT_FLOAT_LT: u8 = 98;
pub(crate) const BC_GOTO_IF_NOT_FLOAT_LE: u8 = 99;
pub(crate) const BC_GOTO_IF_NOT_FLOAT_EQ: u8 = 100;
pub(crate) const BC_GOTO_IF_NOT_FLOAT_NE: u8 = 101;
pub(crate) const BC_GOTO_IF_NOT_FLOAT_GT: u8 = 102;
pub(crate) const BC_GOTO_IF_NOT_FLOAT_GE: u8 = 103;
pub(crate) const BC_GOTO_IF_NOT_PTR_EQ: u8 = 104;
pub(crate) const BC_GOTO_IF_NOT_PTR_NE: u8 = 105;
// blackhole.py:916-920 `bhimpl_goto_if_not_int_is_zero(a, target, pc)`:
// take target iff `a != 0`. jtransform.py:1212 `_rewrite_equality`
// folds `int_eq(x, 0)` into `int_is_zero(x)`; flatten.py:247 then
// specialises the bool exitswitch into `goto_if_not_int_is_zero/iL`.
pub(crate) const BC_GOTO_IF_NOT_INT_IS_ZERO: u8 = 106;

pub(crate) const MAX_HOST_CALL_ARITY: usize = 16;

/// Fixed majit blackhole opcode-name table.
///
/// RPython's `Assembler.insns` is a dense dict grown by
/// `Assembler.write_insn()` in emission order. majit's current runtime
/// `JitCodeBuilder` still emits fixed `BC_*` numbers, so this helper is an
/// adapter table rather than a line-by-line port of `assembler.py`.
/// Downstream consumers use it only for `insns.get('...', -1)`-style opcode
/// cache fields and for wiring handlers against majit's fixed bytecodes.
///
/// The `argcodes` alphabet follows `assembler.py:162-196`:
///   `i` int reg, `r` ref reg, `f` float reg, `c` short-const int,
///   `I/R/F` constant-pool int/ref/float, `L` label, `d` descr,
///   `N` `ListOfKind` (mixed-kind literal list).
pub fn wellknown_bh_insns() -> std::collections::HashMap<&'static str, u8> {
    let mut m = std::collections::HashMap::new();

    // pyjitpl.py:2236-2243 — fields `setup_insns` probes explicitly.
    m.insert("live/", BC_LIVE);
    m.insert("catch_exception/L", BC_CATCH_EXCEPTION);
    m.insert("rvmprof_code/ii", BC_RVMPROF_CODE);
    m.insert("ref_return/r", BC_REF_RETURN);
    // NOTE: majit currently emits only `ref_return` because pyre's portal
    // result type is REF; `int_return`, `float_return`, `void_return`, and
    // `goto/L` have no corresponding BC_* constants yet. `setup_insns`
    // therefore observes `u8::MAX` for those missing RPython opcodes, matching
    // the `insns.get('...', -1)` fallback but not the full RPython opcode set.

    // Control flow / structural markers that actually emit.
    m.insert("jump/L", BC_JUMP);
    // loop_header takes a single int constant operand (the jitdriver index).
    // RPython jtransform.py:1714-1718 handle_jit_marker__loop_header emits
    // SpaceOperation('loop_header', [c_index], None); blackhole.py:1063
    // bhimpl_loop_header(jdindex) is @arguments("i").
    m.insert("loop_header/i", BC_LOOP_HEADER);
    m.insert("raise/r", BC_RAISE);
    m.insert("reraise/", BC_RERAISE);
    m.insert("last_exc_value/", BC_LAST_EXC_VALUE);
    m.insert("jit_merge_point/", BC_JIT_MERGE_POINT);

    // jtransform.py:196 / flatten.py:247 — fused `goto_if_not_<op>_<type>`.
    // Argcodes follow assembler.py:162-196: two registers + label.
    m.insert("goto_if_not_int_lt/iiL", BC_GOTO_IF_NOT_INT_LT);
    m.insert("goto_if_not_int_le/iiL", BC_GOTO_IF_NOT_INT_LE);
    m.insert("goto_if_not_int_eq/iiL", BC_GOTO_IF_NOT_INT_EQ);
    m.insert("goto_if_not_int_ne/iiL", BC_GOTO_IF_NOT_INT_NE);
    m.insert("goto_if_not_int_gt/iiL", BC_GOTO_IF_NOT_INT_GT);
    m.insert("goto_if_not_int_ge/iiL", BC_GOTO_IF_NOT_INT_GE);
    m.insert("goto_if_not_float_lt/ffL", BC_GOTO_IF_NOT_FLOAT_LT);
    m.insert("goto_if_not_float_le/ffL", BC_GOTO_IF_NOT_FLOAT_LE);
    m.insert("goto_if_not_float_eq/ffL", BC_GOTO_IF_NOT_FLOAT_EQ);
    m.insert("goto_if_not_float_ne/ffL", BC_GOTO_IF_NOT_FLOAT_NE);
    m.insert("goto_if_not_float_gt/ffL", BC_GOTO_IF_NOT_FLOAT_GT);
    m.insert("goto_if_not_float_ge/ffL", BC_GOTO_IF_NOT_FLOAT_GE);
    m.insert("goto_if_not_ptr_eq/rrL", BC_GOTO_IF_NOT_PTR_EQ);
    m.insert("goto_if_not_ptr_ne/rrL", BC_GOTO_IF_NOT_PTR_NE);
    m.insert("goto_if_not_int_is_zero/iL", BC_GOTO_IF_NOT_INT_IS_ZERO);

    m
}

/// GC liveness metadata at a specific bytecode PC.
///
/// RPython liveness.py: `[len_i][len_r][len_f][bitset_i][bitset_r][bitset_f]`.
/// Tracks which registers of each type (int/ref/float) are live at a given PC.
///
/// TODO: pyre currently keeps this per-entry form alongside the packed
/// Temporary pyre-side liveness shape used before the codewriter emits
/// RPython `-live-` opcodes directly. Canonical JitCode does not store this.
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
    /// Total number of live registers across all typed banks.
    pub fn total_live(&self) -> usize {
        self.live_i_regs.len() + self.live_r_regs.len() + self.live_f_regs.len()
    }
}

/// Re-export of the canonical `enumerate_vars` function so existing
/// metainterp callers can keep using `crate::jitcode::enumerate_vars`.
///
/// RPython places this function in `rpython/jit/codewriter/jitcode.py`,
/// not in metainterp. majit follows the same module placement: the
/// definition lives in `majit_translate::jitcode::enumerate_vars`.
pub use majit_translate::jitcode::enumerate_vars;

/// Serialized interpreter step description, mirroring RPython's `JitCode`
/// at a much smaller subset for the current proc-macro tracing surface.
///
/// Field names follow `rpython/jit/codewriter/jitcode.py` (`c_num_regs_i`
/// etc). RPython packs each count into `chr(int)` for a 0..=255 range;
/// pyre needs u16 because some Python codes legitimately exceed 255
/// registers per kind (see pyre/pyre-jit/src/jit/codewriter.rs
/// `liveness_regs_to_u8_sorted` for the companion fallback). The name
/// matches RPython even though the representation is wider.
#[derive(Clone, Debug, Default)]
pub struct JitCode {
    // rpython/jit/codewriter/jitcode.py:14-43 canonical fields.
    /// RPython `jitcode.py:15` `self.name = name` — symbolic name for
    /// debugging/logging. RPython takes this as the first `__init__`
    /// argument; pyre's runtime JitCodeBuilder currently leaves it
    /// empty until each compile site wires it through.
    pub name: String,
    /// Encoded bytecode stream.
    pub code: Vec<u8>,
    /// RPython `jitcode.py:37` `self.c_num_regs_i = chr(num_regs_i)`.
    pub c_num_regs_i: u16,
    /// RPython `jitcode.py:38` `self.c_num_regs_r = chr(num_regs_r)`.
    pub c_num_regs_r: u16,
    /// RPython `jitcode.py:39` `self.c_num_regs_f = chr(num_regs_f)`.
    pub c_num_regs_f: u16,
    /// RPython: `jitcode.constants_i` — integer constant pool.
    pub constants_i: Vec<i64>,
    /// RPython: `jitcode.constants_r` — reference constant pool.
    pub constants_r: Vec<i64>,
    /// RPython: `jitcode.constants_f` — float constant pool (bits as i64).
    pub constants_f: Vec<i64>,
    /// jitcode.py:18 `jitdriver_sd` — index into jitdrivers_sd array.
    /// `Some(index)` for portal jitcodes, `None` for helpers.
    /// Set by call.py:148 grab_initial_jitcodes parity:
    /// `jd.mainjitcode.jitdriver_sd = jd`.
    /// Used by `_handle_jitexception` to find the portal level in the BH chain.
    pub jitdriver_sd: Option<usize>,
    /// RPython `jitcode.py:16` `self.fnaddr` — function address for `bh_call_*`.
    /// Set by warmspot.py/call.py from `getfunctionptr(graph)`.
    pub fnaddr: i64,
    /// RPython `jitcode.py:17` `self.calldescr` — calling convention descriptor.
    /// Set by `call.py:get_jitcode_calldescr(graph)` from the function's type.
    pub calldescr: majit_translate::jitcode::BhCallDescr,

    // majit bytecode adapter extension.
    // These fields have no RPython JitCode counterpart. RPython's
    // bytecode operand stream references external dispatch tables
    // maintained by the blackhole builder / metainterp_sd; majit's
    // runtime JitCodeBuilder inlines the equivalent pools directly
    // on each JitCode for straight-line dispatch. Future parity work
    // is to lift these back out to a separate per-adapter structure.
    /// Pool of majit IR opcodes referenced from the bytecode stream.
    pub opcodes: Vec<OpCode>,
    /// Sub-JitCodes for `inline_call` targets (compound methods).
    pub sub_jitcodes: Vec<std::sync::Arc<JitCode>>,
    /// Function pointers for `residual_call` targets (I/O shims, external calls).
    pub fn_ptrs: Vec<JitCallTarget>,
    /// CALL_ASSEMBLER targets keyed by loop token number plus a concrete hook.
    assembler_targets: Vec<JitCallAssemblerTarget>,
}

// -- RPython jitcode.py parity methods --

impl JitCode {
    /// RPython `jitcode.py:47-48` `def num_regs_i(self): return ord(self.c_num_regs_i)`.
    pub fn num_regs_i(&self) -> u16 {
        self.c_num_regs_i
    }

    /// RPython `jitcode.py:50-51` `def num_regs_r(self): return ord(self.c_num_regs_r)`.
    pub fn num_regs_r(&self) -> u16 {
        self.c_num_regs_r
    }

    /// RPython `jitcode.py:53-54` `def num_regs_f(self): return ord(self.c_num_regs_f)`.
    pub fn num_regs_f(&self) -> u16 {
        self.c_num_regs_f
    }

    /// Transitional CALL_ASSEMBLER target lookup for the hardcoded
    /// JitCodeBuilder bytecode. RPython stores the callee loop token in
    /// descriptor data; this keeps the private compat table encapsulated.
    pub(crate) fn call_assembler_target(&self, index: usize) -> (u64, *const ()) {
        let target = self.assembler_targets[index];
        (target.token_number, target.concrete_ptr)
    }

    /// RPython `jitcode.py:56-57` `def num_regs_and_consts_i(self): return ord(self.c_num_regs_i) + len(self.constants_i)`.
    pub fn num_regs_and_consts_i(&self) -> usize {
        self.c_num_regs_i as usize + self.constants_i.len()
    }

    /// RPython `jitcode.py:59-60` `def num_regs_and_consts_r(self): return ord(self.c_num_regs_r) + len(self.constants_r)`.
    pub fn num_regs_and_consts_r(&self) -> usize {
        self.c_num_regs_r as usize + self.constants_r.len()
    }

    /// RPython `jitcode.py:62-63` `def num_regs_and_consts_f(self): return ord(self.c_num_regs_f) + len(self.constants_f)`.
    pub fn num_regs_and_consts_f(&self) -> usize {
        self.c_num_regs_f as usize + self.constants_f.len()
    }

    /// RPython jitcode.py:82-93 `get_live_vars_info(self, pc, op_live)`.
    pub fn get_live_vars_info(&self, pc: usize, op_live: u8) -> usize {
        let mut pc = pc;
        if self.code.get(pc).copied() != Some(op_live) {
            let step = majit_translate::liveness::OFFSET_SIZE + 1;
            if pc < step {
                self._missing_liveness(pc);
            }
            pc -= step;
            if self.code.get(pc).copied() != Some(op_live) {
                self._missing_liveness(pc);
            }
        }
        majit_translate::liveness::decode_offset(&self.code, pc + 1)
    }

    /// RPython jitcode.py:95-100 `_missing_liveness(self, pc)`.
    pub fn _missing_liveness(&self, pc: usize) -> ! {
        panic!("missing liveness[{}] in JitCode", pc)
    }

    /// RPython: `JitCode.follow_jump(position)` -- follow a label at position.
    pub fn follow_jump(&self, position: usize) -> usize {
        if position < 2 || position - 2 + 1 >= self.code.len() {
            return 0;
        }
        let pos = position - 2;
        (self.code[pos] as usize) | ((self.code[pos + 1] as usize) << 8)
    }

    /// RPython `jitcode.py:114-119` `def dump(self)`.
    pub fn dump(&self) -> String {
        format!(
            "<JitCode {:?}: {} bytes, {} int regs, {} consts>",
            self.name,
            self.code.len(),
            self.c_num_regs_i,
            self.constants_i.len()
        )
    }
}

// RPython `jitcode.py:121-122` `def __repr__(self): return '<JitCode %r>' % self.name`.
impl std::fmt::Display for JitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<JitCode {:?}>", self.name)
    }
}

/// Convert a build-time `majit_translate::jitcode::JitCode` (emitted by
/// `build.rs` and serialized to `opcode_jitcodes.bin`) into a runtime
/// `JitCode` usable by `BlackholeInterpreter::dispatch_one`.
///
/// Field-width and representation differences that this conversion
/// papers over (RPython has one `JitCode` struct, so there is no upstream
/// counterpart for the width mismatch — it exists only because pyre's
/// build-time serialization uses narrower fields to save space):
///
///   - `c_num_regs_{i,r,f}`: build-time `u8`, runtime `u16`.
///   - `constants_r`: build-time `Vec<u64>`, runtime `Vec<i64>` (same
///     bit pattern; GCREF pointer reinterpretation).
///   - `constants_f`: build-time `Vec<f64>`, runtime `Vec<i64>`
///     (runtime stores `longlong.FLOATSTORAGE` bits per RPython
///     `jitcode.py:34`; `f64::to_bits` yields the same raw pattern).
///
/// Runtime-only fields (`opcodes`, `sub_jitcodes`, `fn_ptrs`,
/// `assembler_targets`) default-initialize to empty — they are populated
/// by the runtime codewriter (`pyre-jit/src/jit/`) for function-level
/// jitcodes, not by `build.rs`. Phase D-2's shadow dispatch is
/// intentionally limited to opcode arms whose decomposed bytecode does
/// not reach `inline_call` / `residual_call` until these pools are
/// wired too.
///
/// Panics if the build-time jitcode body has not been set
/// (`set_body()` not called). Build artifacts that were fully assembled
/// by `codewriter.drain_pending_graphs` always satisfy this.
impl From<&majit_translate::jitcode::JitCode> for JitCode {
    fn from(bt: &majit_translate::jitcode::JitCode) -> Self {
        let body = bt.body();
        Self {
            name: bt.name.clone(),
            code: body.code.clone(),
            c_num_regs_i: body.c_num_regs_i as u16,
            c_num_regs_r: body.c_num_regs_r as u16,
            c_num_regs_f: body.c_num_regs_f as u16,
            constants_i: body.constants_i.clone(),
            constants_r: body.constants_r.iter().map(|&u| u as i64).collect(),
            constants_f: body
                .constants_f
                .iter()
                .map(|&f| f.to_bits() as i64)
                .collect(),
            jitdriver_sd: bt.jitdriver_sd.get().copied(),
            fnaddr: bt.fnaddr,
            calldescr: body.calldescr.clone(),
            opcodes: Vec::new(),
            sub_jitcodes: Vec::new(),
            fn_ptrs: Vec::new(),
            assembler_targets: Vec::new(),
        }
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

    /// Map a [`majit_ir::Type`] to its `JitArgKind`.  RPython encodes
    /// the same mapping inline in `_build_allboxes` per
    /// `pyjitpl.py:1969-1989` (`history.INT`/`history.REF`/`history.FLOAT`
    /// chars + `'S'` single-float / `'L'` long-long aliases).  Pyre's
    /// `Type::Void` has no JitArgKind because void calls carry no
    /// argbox.
    pub fn from_type(ty: majit_ir::Type) -> Option<Self> {
        match ty {
            majit_ir::Type::Int => Some(Self::Int),
            majit_ir::Type::Ref => Some(Self::Ref),
            majit_ir::Type::Float => Some(Self::Float),
            majit_ir::Type::Void => None,
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

#[cfg(test)]
mod tests {
    use super::*;
    use majit_translate::jitcode::{JitCode as BuildJitCode, JitCodeBody as BuildJitCodeBody};

    #[test]
    fn from_build_jitcode_shares_code_and_widens_reg_counts() {
        // Build-time: u8 register counts, Vec<u64> ref constants,
        // Vec<f64> float constants. Runtime needs u16 / Vec<i64>.
        let body = BuildJitCodeBody {
            code: vec![0xAA, 0xBB, 0xCC],
            c_num_regs_i: 17,
            c_num_regs_r: 5,
            c_num_regs_f: 2,
            constants_i: vec![42, -1],
            constants_r: vec![0x_FFFF_FFFF_FFFF_0000_u64],
            constants_f: vec![1.5_f64, -0.25_f64],
            ..Default::default()
        };
        let bt = BuildJitCode::new("test/jc");
        bt.set_body(body);

        let rt: JitCode = (&bt).into();

        assert_eq!(rt.name, "test/jc");
        assert_eq!(rt.code, vec![0xAA, 0xBB, 0xCC]);
        assert_eq!(rt.c_num_regs_i, 17);
        assert_eq!(rt.c_num_regs_r, 5);
        assert_eq!(rt.c_num_regs_f, 2);
        assert_eq!(rt.constants_i, vec![42, -1]);
        // u64 → i64 bit-reinterpret: the top bit flips to signed.
        assert_eq!(rt.constants_r, vec![0xFFFF_FFFF_FFFF_0000_u64 as i64]);
        // f64 → bits → i64: round-trip via f64::to_bits.
        assert_eq!(rt.constants_f[0], f64::to_bits(1.5_f64) as i64);
        assert_eq!(rt.constants_f[1], f64::to_bits(-0.25_f64) as i64);
        // Runtime-only fields default-init empty.
        assert!(rt.opcodes.is_empty());
        assert!(rt.sub_jitcodes.is_empty());
        assert!(rt.fn_ptrs.is_empty());
        assert!(rt.assembler_targets.is_empty());
    }

    #[test]
    fn from_build_jitcode_preserves_jitdriver_and_fnaddr() {
        let body = BuildJitCodeBody::default();
        let mut bt = BuildJitCode::new("portal/jc");
        bt.fnaddr = 0x1234;
        bt.set_body(body);
        bt.set_jitdriver_sd(3);

        let rt: JitCode = (&bt).into();
        assert_eq!(rt.jitdriver_sd, Some(3));
        assert_eq!(rt.fnaddr, 0x1234);
    }

    #[test]
    fn setposition_accepts_converted_jitcode_and_sizes_register_files() {
        // Slice 2: end-to-end proof that a build-time JitCode converted via
        // `From<&BuildJitCode>` is directly usable as input to
        // `BlackholeInterpreter::setposition`. This is the integration point
        // the Phase D-2 blackhole record-on-execute path needs in place
        // before any bhimpl dispatch is attempted.
        //
        // RPython: `blackhole.py:312 setposition` allocates `num_regs_i +
        // len(constants_i)` slots per register file and copies each constant
        // into the tail portion of the file. We verify both — the array
        // sizes and the copied-in constants.
        use crate::blackhole::BlackholeInterpreter;

        let body = BuildJitCodeBody {
            code: vec![BC_LIVE, 0x00, 0x00], // live/ with 2-byte offset
            c_num_regs_i: 4,
            c_num_regs_r: 2,
            c_num_regs_f: 1,
            constants_i: vec![100, 200, 300],
            constants_r: vec![0xAABB_CCDD_EEFF_0011_u64, 0x2233_4455_6677_8899_u64],
            constants_f: vec![1.25_f64],
            ..Default::default()
        };
        let bt = BuildJitCode::new("slice2/test");
        bt.set_body(body);

        let rt_jc = std::sync::Arc::new(JitCode::from(&bt));
        let mut bh = BlackholeInterpreter::new();
        bh.setposition(rt_jc.clone(), 0);

        // num_regs_and_consts_i = 4 + 3 = 7; constants occupy [4..7].
        assert_eq!(bh.registers_i.len(), 7);
        assert_eq!(&bh.registers_i[4..7], &[100, 200, 300]);
        // Working regs remain zero-initialised.
        assert_eq!(&bh.registers_i[0..4], &[0, 0, 0, 0]);

        // Refs: u64 bit pattern reinterpreted as i64 by the conversion.
        assert_eq!(bh.registers_r.len(), 4); // 2 regs + 2 constants
        assert_eq!(bh.registers_r[2], 0xAABB_CCDD_EEFF_0011_u64 as i64);
        assert_eq!(bh.registers_r[3], 0x2233_4455_6677_8899_u64 as i64);

        // Floats: f64 bits reinterpreted; round-trip through f64::to_bits
        // must match what BlackholeInterpreter sees.
        assert_eq!(bh.registers_f.len(), 2);
        assert_eq!(bh.registers_f[1], f64::to_bits(1.25_f64) as i64);

        assert_eq!(bh.position, 0);
        assert_eq!(&bh.jitcode.code, &[BC_LIVE, 0x00, 0x00]);
    }
}
