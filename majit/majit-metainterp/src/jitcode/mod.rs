mod assembler;

pub use assembler::JitCodeBuilder;
pub use majit_translate::jitcode::{
    BhCallDescr as CanonicalBhCallDescr, BhDescr as CanonicalBhDescr, JitCode as CanonicalJitCode,
};

pub(crate) const BC_LOOP_HEADER: u8 = 12;
pub(crate) const BC_ABORT: u8 = 13;
pub(crate) const BC_ABORT_PERMANENT: u8 = 14;
/// RPython `blackhole.py:913` aliases `bhimpl_goto_if_not_int_is_true`
/// to `bhimpl_goto_if_not`, whose body takes the branch iff the int
/// register is zero/false (`goto_if_not_int_is_true/iL`).
pub(crate) const BC_GOTO_IF_NOT_INT_IS_TRUE: u8 = 15;
pub(crate) const BC_JUMP: u8 = 16;
pub(crate) const BC_INLINE_CALL: u8 = 17;
pub(crate) const BC_RESIDUAL_CALL_VOID: u8 = 18;
pub(crate) const BC_MOVE_I: u8 = 21;
pub(crate) const BC_CALL_INT: u8 = 22;
pub(crate) const BC_CALL_PURE_INT: u8 = 23;
// Ref-typed bytecodes
pub(crate) const BC_MOVE_R: u8 = 27;
pub(crate) const BC_CALL_REF: u8 = 28;
pub(crate) const BC_CALL_PURE_REF: u8 = 29;
// Float-typed bytecodes
pub(crate) const BC_MOVE_F: u8 = 33;
pub(crate) const BC_CALL_FLOAT: u8 = 34;
pub(crate) const BC_CALL_PURE_FLOAT: u8 = 35;
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
/// RPython blackhole.py:987 `last_exception/>i`.
pub(crate) const BC_LAST_EXCEPTION: u8 = 129;
/// RPython blackhole.py:976-985 `goto_if_exception_mismatch/iL`.
pub(crate) const BC_GOTO_IF_EXCEPTION_MISMATCH: u8 = 130;
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

// blackhole.py:661-679 bhimpl_int_push / bhimpl_ref_push /
// bhimpl_float_push and matching pops — one-slot scratch for the
// cycle-break path emitted by flatten.py:326-332 `insert_renamings`.
pub(crate) const BC_INT_PUSH: u8 = 107;
pub(crate) const BC_REF_PUSH: u8 = 108;
pub(crate) const BC_FLOAT_PUSH: u8 = 109;
pub(crate) const BC_INT_POP: u8 = 110;
pub(crate) const BC_REF_POP: u8 = 111;
pub(crate) const BC_FLOAT_POP: u8 = 112;

pub(crate) const BC_INT_ADD: u8 = 113;
pub(crate) const BC_INT_SUB: u8 = 114;
pub(crate) const BC_INT_MUL: u8 = 115;
pub(crate) const BC_INT_FLOORDIV: u8 = 116;
pub(crate) const BC_INT_MOD: u8 = 117;
pub(crate) const BC_INT_AND: u8 = 118;
pub(crate) const BC_INT_OR: u8 = 119;
pub(crate) const BC_INT_XOR: u8 = 120;
pub(crate) const BC_INT_LSHIFT: u8 = 121;
pub(crate) const BC_INT_RSHIFT: u8 = 122;
pub(crate) const BC_INT_EQ: u8 = 123;
pub(crate) const BC_INT_NE: u8 = 124;
pub(crate) const BC_INT_LT: u8 = 125;
pub(crate) const BC_INT_LE: u8 = 126;
pub(crate) const BC_INT_GT: u8 = 127;
pub(crate) const BC_INT_GE: u8 = 128;
pub(crate) const BC_INT_NEG: u8 = 132;
pub(crate) const BC_FLOAT_ADD: u8 = 133;
pub(crate) const BC_FLOAT_SUB: u8 = 134;
pub(crate) const BC_FLOAT_MUL: u8 = 135;
pub(crate) const BC_FLOAT_TRUEDIV: u8 = 136;
pub(crate) const BC_FLOAT_NEG: u8 = 139;
pub(crate) const BC_FLOAT_ABS: u8 = 140;
pub(crate) const BC_INT_INVERT: u8 = 141;
pub(crate) const BC_UINT_RSHIFT: u8 = 142;
pub(crate) const BC_UINT_MUL_HIGH: u8 = 143;
pub(crate) const BC_UINT_LT: u8 = 144;
pub(crate) const BC_UINT_LE: u8 = 145;
pub(crate) const BC_UINT_GT: u8 = 146;
pub(crate) const BC_UINT_GE: u8 = 147;
// Ref/nullity primitives — RPython `blackhole.py:584-610`
// `bhimpl_{ptr_eq,ptr_ne,ptr_iszero,ptr_nonzero,instance_ptr_eq,instance_ptr_ne}`.
pub(crate) const BC_PTR_EQ: u8 = 151;
pub(crate) const BC_PTR_NE: u8 = 152;
pub(crate) const BC_INSTANCE_PTR_EQ: u8 = 153;
pub(crate) const BC_INSTANCE_PTR_NE: u8 = 154;
pub(crate) const BC_PTR_ISZERO: u8 = 155;
pub(crate) const BC_PTR_NONZERO: u8 = 156;
// Unary ptr nullity exitswitch specialisations — `blackhole.py:937-944`
// `bhimpl_goto_if_not_ptr_{iszero,nonzero}`.
pub(crate) const BC_GOTO_IF_NOT_PTR_ISZERO: u8 = 157;
pub(crate) const BC_GOTO_IF_NOT_PTR_NONZERO: u8 = 158;
// Typed return opcodes — RPython `blackhole.py:841-862`
// `bhimpl_int_return`, `bhimpl_float_return`, `bhimpl_void_return`.
// pyre's portal return is REF (see BC_REF_RETURN) but the insns map
// still needs every upstream return flavour so
// `pyjitpl.py:2240-2243` `setup_insns` fields do not fall back to
// `u8::MAX` sentinels.
pub(crate) const BC_INT_RETURN: u8 = 148;
pub(crate) const BC_FLOAT_RETURN: u8 = 149;
pub(crate) const BC_VOID_RETURN: u8 = 150;

pub(crate) const MAX_HOST_CALL_ARITY: usize = 16;

/// Lookup a bytecode opcode by its `opname/argcodes` key.
///
/// RPython `assembler.py:220-222`:
/// ```text
/// key = opname + '/' + ''.join(argcodes)
/// num = self.insns.setdefault(key, len(self.insns))
/// self.code[startposition] = chr(num)
/// ```
///
/// majit currently pre-populates the dict from `wellknown_bh_insns` so
/// numbers match the hardcoded `BC_*` constants consumed by the
/// blackhole dispatch. Once dispatch becomes table-driven the
/// pre-population will drop and numbers will be allocated in emission
/// order exactly like RPython.
///
/// Panics if `key` is not registered — mirrors the `assert 0 <= num <=
/// 0xFF` behaviour RPython relies on at the assembler layer.
pub(crate) fn insn_byte(key: &str) -> u8 {
    use std::sync::OnceLock;
    static TABLE: OnceLock<std::collections::HashMap<&'static str, u8>> = OnceLock::new();
    let table = TABLE.get_or_init(wellknown_bh_insns);
    *table
        .get(key)
        .unwrap_or_else(|| panic!("insn_byte: unregistered insns key {key:?}"))
}

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
    // pyjitpl.py:2240-2243 typed return accessors:
    //   op_int_return / op_ref_return / op_float_return / op_void_return
    // pyre's portal result type is REF so `ref_return/r` is the only
    // one produced by the current emitter, but the other three must be
    // registered so `setup_insns` does not fall back to `u8::MAX` for
    // them.
    m.insert("int_return/i", BC_INT_RETURN);
    m.insert("ref_return/r", BC_REF_RETURN);
    m.insert("float_return/f", BC_FLOAT_RETURN);
    m.insert("void_return/", BC_VOID_RETURN);

    // pyre-only tracer termination markers. RPython's tracing bails out via
    // exceptions (`SwitchToBlackhole`, `ContinueRunningNormally`), but pyre's
    // borrow-checker forbids unwinding through the trace loop, so the
    // JitCodeMachine signals abort via explicit opcodes instead. No RPython
    // counterpart; keys kept pyre-local (no `>`-return marker).
    m.insert("abort/", BC_ABORT);
    m.insert("abort_permanent/", BC_ABORT_PERMANENT);

    // pyre-only virtualizable state-field / state-array / state-varray
    // machine. No RPython counterpart — RPython expresses frame-local
    // field access via the canonical `{get,set}field_vable_*` family
    // that routes through an explicit vable pointer register. pyre's
    // SSA codewriter instead models the virtualizable as a set of
    // "state slots" that are loaded/stored through dedicated jitcode
    // opcodes before the vable pool is plumbed through; the emit sites
    // are in `majit-macros/src/jit_interp/jitcode_lower.rs`.
    //
    // Argcodes: `d` = state-slot index (emitted as u16), `i` = int
    // register (u16). Array variants carry an extra `i` for the index
    // register before the destination/source slot.
    m.insert("load_state_field/di", BC_LOAD_STATE_FIELD);
    m.insert("store_state_field/di", BC_STORE_STATE_FIELD);
    m.insert("load_state_array/dii", BC_LOAD_STATE_ARRAY);
    m.insert("store_state_array/dii", BC_STORE_STATE_ARRAY);
    m.insert("load_state_varray/dii", BC_LOAD_STATE_VARRAY);
    m.insert("store_state_varray/dii", BC_STORE_STATE_VARRAY);

    // Control flow / structural markers that actually emit.
    // pyjitpl.py:2237 `op_goto = insns.get('goto/L', -1)` and
    // blackhole.py:950 `bhimpl_goto(target): return target` — the
    // canonical key is `goto/L`.
    m.insert("goto/L", BC_JUMP);
    // loop_header takes a single int constant operand (the jitdriver index).
    // RPython jtransform.py:1714-1718 handle_jit_marker__loop_header emits
    // SpaceOperation('loop_header', [c_index], None); blackhole.py:1063
    // bhimpl_loop_header(jdindex) is @arguments("i").
    m.insert("loop_header/i", BC_LOOP_HEADER);
    m.insert("raise/r", BC_RAISE);
    m.insert("reraise/", BC_RERAISE);
    // blackhole.py:987 `@arguments("self", returns="i") bhimpl_last_exception`
    // yields canonical key `last_exception/>i`.
    m.insert("last_exception/>i", BC_LAST_EXCEPTION);
    m.insert(
        "goto_if_exception_mismatch/iL",
        BC_GOTO_IF_EXCEPTION_MISMATCH,
    );
    // flatten.py:347 emits `last_exc_value, '->', reg`, so
    // assembler.py grows the canonical key `last_exc_value/>r`.
    m.insert("last_exc_value/>r", BC_LAST_EXC_VALUE);
    // pyre codewriter emits the portal subset of
    // `jit_merge_point`: green-int list, green-ref list, red-ref list.
    // The canonical SSA key is therefore `jit_merge_point/IRR`.
    m.insert("jit_merge_point/IRR", BC_JIT_MERGE_POINT);
    // jtransform.py:292-313 / 1672-1688 conditional/known-result family
    // intentionally omitted. The helper-side `BC_COND_CALL_*` /
    // `BC_RECORD_KNOWN_RESULT_*` adapters encode argc + per-arg kind tags
    // in a flat payload, which is not line-by-line compatible with the
    // canonical `iiIRd` / `riIRd>r` argcode layout. The translator-owned
    // codewriter pipeline emits the real canonical keys when it actually
    // assembles those operations.
    // blackhole.py:1278-1319 inline-call family intentionally omitted.
    // The helper-side `BC_INLINE_CALL` adapter in majit-metainterp uses a
    // typed arg + caller-destination payload that is not line-by-line compatible
    // with canonical `inline_call_*` argcodes. The real RPython-shape
    // `inline_call_*` keys come from the translator/codewriter pipeline
    // when they are actually emitted; pre-registering them here would make
    // `wellknown_bh_insns()` claim a bytecode contract this runtime does
    // not truthfully expose.

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
    m.insert("goto_if_not_ptr_iszero/rL", BC_GOTO_IF_NOT_PTR_ISZERO);
    m.insert("goto_if_not_ptr_nonzero/rL", BC_GOTO_IF_NOT_PTR_NONZERO);
    m.insert("goto_if_not_int_is_zero/iL", BC_GOTO_IF_NOT_INT_IS_ZERO);

    // flatten.py:326-332 `insert_renamings` cycle-break push/pop pairs.
    // Argcodes follow assembler.py:162-196 / blackhole.py:661-679:
    // push takes one register source (`i`/`r`/`f`), pop writes one
    // register destination (`>i`/`>r`/`>f`).
    m.insert("int_push/i", BC_INT_PUSH);
    m.insert("ref_push/r", BC_REF_PUSH);
    m.insert("float_push/f", BC_FLOAT_PUSH);
    m.insert("int_pop/>i", BC_INT_POP);
    m.insert("ref_pop/>r", BC_REF_POP);
    m.insert("float_pop/>f", BC_FLOAT_POP);

    m.insert("int_add/ii>i", BC_INT_ADD);
    m.insert("int_sub/ii>i", BC_INT_SUB);
    m.insert("int_mul/ii>i", BC_INT_MUL);
    m.insert("int_floordiv/ii>i", BC_INT_FLOORDIV);
    m.insert("int_mod/ii>i", BC_INT_MOD);
    m.insert("int_and/ii>i", BC_INT_AND);
    m.insert("int_or/ii>i", BC_INT_OR);
    m.insert("int_xor/ii>i", BC_INT_XOR);
    m.insert("int_lshift/ii>i", BC_INT_LSHIFT);
    m.insert("int_rshift/ii>i", BC_INT_RSHIFT);
    m.insert("int_eq/ii>i", BC_INT_EQ);
    m.insert("int_ne/ii>i", BC_INT_NE);
    m.insert("int_lt/ii>i", BC_INT_LT);
    m.insert("int_le/ii>i", BC_INT_LE);
    m.insert("int_gt/ii>i", BC_INT_GT);
    m.insert("int_ge/ii>i", BC_INT_GE);
    m.insert("int_neg/i>i", BC_INT_NEG);
    m.insert("int_invert/i>i", BC_INT_INVERT);
    m.insert("uint_rshift/ii>i", BC_UINT_RSHIFT);
    m.insert("uint_mul_high/ii>i", BC_UINT_MUL_HIGH);
    m.insert("uint_lt/ii>i", BC_UINT_LT);
    m.insert("uint_le/ii>i", BC_UINT_LE);
    m.insert("uint_gt/ii>i", BC_UINT_GT);
    m.insert("uint_ge/ii>i", BC_UINT_GE);
    // Ref/nullity primitives — `blackhole.py:584-610`.
    m.insert("ptr_eq/rr>i", BC_PTR_EQ);
    m.insert("ptr_ne/rr>i", BC_PTR_NE);
    m.insert("instance_ptr_eq/rr>i", BC_INSTANCE_PTR_EQ);
    m.insert("instance_ptr_ne/rr>i", BC_INSTANCE_PTR_NE);
    m.insert("ptr_iszero/r>i", BC_PTR_ISZERO);
    m.insert("ptr_nonzero/r>i", BC_PTR_NONZERO);
    // Per-opname float primitives — `blackhole.py:696-723`
    // `bhimpl_float_{add,sub,mul,truediv,neg,abs}`.
    m.insert("float_add/ff>f", BC_FLOAT_ADD);
    m.insert("float_sub/ff>f", BC_FLOAT_SUB);
    m.insert("float_mul/ff>f", BC_FLOAT_MUL);
    m.insert("float_truediv/ff>f", BC_FLOAT_TRUEDIV);
    m.insert("float_neg/f>f", BC_FLOAT_NEG);
    m.insert("float_abs/f>f", BC_FLOAT_ABS);

    // Typed register copy — `blackhole.py:638-646`
    // `bhimpl_{int,ref,float}_copy`. `@arguments("i"|"r"|"f",
    // returns="i"|"r"|"f")` yields canonical keys
    // `{int,ref,float}_copy/X>X`. pyre's `move_{i,r,f}` emitters route
    // through these bytes; flatten.py:326-332 `insert_renamings` is the
    // main RPython producer of `int_copy` ops (cycle-break renamings),
    // which pyre's super-inst expansion also re-uses.
    m.insert("int_copy/i>i", BC_MOVE_I);
    m.insert("ref_copy/r>r", BC_MOVE_R);
    m.insert("float_copy/f>f", BC_MOVE_F);

    // Guard-value promotions — `blackhole.py:648-656`
    // `bhimpl_{int,ref,float}_guard_value`. Body is a no-op on the
    // blackhole side; `pyjitpl.py:1512-1515`
    // `opimpl_{int,ref,float}_guard_value` = `_opimpl_guard_value`
    // emits GUARD_VALUE during tracing to promote the operand.
    m.insert("int_guard_value/i", BC_INT_GUARD_VALUE);
    m.insert("ref_guard_value/r", BC_REF_GUARD_VALUE);
    m.insert("float_guard_value/f", BC_FLOAT_GUARD_VALUE);

    // Truthy-exitswitch branch — `flatten.py:245` emits the canonical
    // `goto_if_not/iL`; `blackhole.py:913`
    // `bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not` adds the
    // specialised alias. Both keys map to the same fixed runtime byte.
    m.insert("goto_if_not/iL", BC_GOTO_IF_NOT_INT_IS_TRUE);
    m.insert("goto_if_not_int_is_true/iL", BC_GOTO_IF_NOT_INT_IS_TRUE);

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

/// Runtime descriptor entry — heterogeneous pool element indexed by
/// `j` / `d` argcodes at dispatch time. Equivalent of RPython
/// `self.descrs[idx]` where each entry is an instance of one of the
/// `AbstractDescr` subclasses (`FieldDescr`, `ArrayDescr`, `JitCode`,
/// ...). RPython uses `isinstance(value, JitCode)` to discriminate at
/// runtime; pyre encodes the same discrimination in the enum tag
/// because `majit_translate::jitcode::BhDescr` (build-time) cannot
/// carry an `Arc<majit_metainterp::jitcode::JitCode>` without a
/// layering inversion between the two crates.
#[derive(Clone, Debug)]
pub enum RuntimeBhDescr {
    /// Target JitCode for a `j` argcode (`BC_INLINE_CALL`). RPython:
    /// `blackhole.py:150-157` — `argtype == 'j' → descrs[idx]` asserted
    /// `isinstance(value, JitCode)`. pyre's runtime carries the `Arc`
    /// inline because the runtime-generated JitCodes have no global
    /// allocation index into `metainterp_sd.all_jitcodes`.
    JitCode(std::sync::Arc<JitCode>),
    /// Target function for `BC_CALL_*` / `BC_RESIDUAL_CALL_*`.
    /// RPython `blackhole.py:1225-1256` reads the function address
    /// from an int register (`i` argcode) and the calling convention
    /// from a descr (`d` argcode); pyre keeps the two together in
    /// `JitCallTarget` because the runtime emitter wires trace-side
    /// and blackhole-side function pointers in a single indirection
    /// slot. Once pyre emits the function address via an int register
    /// this variant can split into the RPython-shaped pair.
    Call(JitCallTarget),
    /// Compiled-assembler target for `BC_CALL_ASSEMBLER_*`. The
    /// `token_number` identifies a `CompiledLoopToken` (RPython
    /// `compile.py CompiledLoopToken.number`) that the tracer hands to
    /// `ctx.call_assembler_*_typed` so the metainterp can chain this
    /// trace into an already-compiled one.
    AssemblerToken(JitCallAssemblerTarget),
}

impl RuntimeBhDescr {
    /// RPython parity: `isinstance(value, JitCode)` assertion at
    /// `blackhole.py:156`. Returns the callee JitCode for `BC_INLINE_CALL`.
    pub fn as_jitcode(&self) -> Option<&std::sync::Arc<JitCode>> {
        match self {
            Self::JitCode(arc) => Some(arc),
            _ => None,
        }
    }

    /// Extract the `Call` target for `BC_CALL_*` / `BC_RESIDUAL_CALL_*`.
    pub fn as_call(&self) -> Option<&JitCallTarget> {
        match self {
            Self::Call(target) => Some(target),
            _ => None,
        }
    }

    /// Extract the assembler-call target for `BC_CALL_ASSEMBLER_*`.
    pub(crate) fn as_assembler_token(&self) -> Option<&JitCallAssemblerTarget> {
        match self {
            Self::AssemblerToken(target) => Some(target),
            _ => None,
        }
    }
}

/// Runtime adapter state — pyre-only descriptor pool not present on
/// canonical RPython `JitCode`. RPython keeps descriptors in a single
/// shared `BlackholeInterpBuilder.descrs` list keyed by the `d`/`j`
/// argcodes (blackhole.py:59, 154); pyre's runtime jitcodes are
/// generated per-Python-frame and cannot index into that shared pool
/// because they lack a global allocation index, so the list lives
/// per-JitCode here.
///
/// Phase I2 Step 2 moved the four pyre-only adapter fields
/// (`sub_jitcodes`, `fn_ptrs`, `assembler_targets`, `opcodes`) off the
/// canonical `JitCode` into this container. Step 3 collapsed three of
/// those pools (`sub_jitcodes` → `RuntimeBhDescr::JitCode`, `fn_ptrs` →
/// `RuntimeBhDescr::Call`, `assembler_targets` →
/// `RuntimeBhDescr::AssemblerToken`) into the unified `descrs` list.
/// Slice 3c dissolved the remaining `opcodes` pool by splitting
/// `BC_RECORD_BINOP_*` / `BC_RECORD_UNARY_*` into per-opname `BC_*`
/// bytes (`BC_INT_ADD`, `BC_FLOAT_MUL`, …), so the struct now holds
/// only `descrs`, mirroring `BlackholeInterpBuilder.descrs` /
/// `BlackholeInterpreter.descrs`
/// (`blackhole.py:103, 288`).
#[derive(Clone, Debug, Default)]
pub struct JitCodeExecState {
    /// Descriptor pool — indexed by the 2-byte `j`/`d` argcode operand.
    /// Naming mirrors RPython `BlackholeInterpBuilder.descrs`
    /// (blackhole.py:103) / `BlackholeInterpreter.descrs`
    /// (blackhole.py:288). Build-time pyre routes the shared pool via
    /// `BlackholeInterpBuilder.descrs`; runtime pyre keeps the
    /// equivalent list per-JitCode because each Python-frame JitCode is
    /// generated on demand. Heterogeneous entries discriminated by the
    /// `RuntimeBhDescr` variant tag:
    /// - `JitCode(Arc<JitCode>)` for `j` argcode (`BC_INLINE_CALL`).
    /// - `Call(JitCallTarget)` for `d` argcode (`BC_CALL_*` /
    ///   `BC_RESIDUAL_CALL_*`) — pyre bundles trace + concrete fn
    ///   pointers in one slot; RPython keeps the function address in
    ///   an int register and only the calling convention in the descr.
    pub descrs: Vec<RuntimeBhDescr>,
}

/// Serialized interpreter step description, mirroring RPython's `JitCode`
/// at a much smaller subset for the current proc-macro tracing surface.
///
/// Field names follow `rpython/jit/codewriter/jitcode.py` (`c_num_regs_i`
/// etc). RPython packs each count into `chr(int)` for a 0..=255 range;
/// pyre needs u16 because some Python codes legitimately exceed 255
/// registers per kind (see pyre/pyre-jit/src/jit/codewriter.rs
/// `liveness_regs_to_u8_sorted` for the companion fallback). The name
/// matches RPython even though the representation is wider.
///
/// PRE-EXISTING-ADAPTATION: `index` uses `AtomicI64` interior mutability
/// because pyre's runtime registration path (`state::jitcode_for`) needs
/// to back-stamp the canonical `metainterp_sd.jitcodes` position onto
/// the inner `JitCode` after the surrounding `Arc<PyJitCode>` is already
/// shared (refcount > 1).  RPython mutates the `JitCode` Python object
/// in place via `jitcode.index = index` (codewriter.py:68); the atomic
/// field gives Rust the same in-place semantics under shared ownership
/// without forcing a deep clone or restructuring the `Arc` plumbing.
#[derive(Debug, Default)]
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
    /// RPython `codewriter.py:68` `jitcode.index = index` — the
    /// zero-based index of this jitcode in `metainterp_sd.jitcodes`.
    /// Snapshot records encode this value as the caller's `jitcode`
    /// identity (`opencoder.py:777 frame.jitcode.index`). Defaults to
    /// `0` until the owning codewriter/setup path assigns it; the -1
    /// sentinel for "empty snapshot" is handled on the snapshot side
    /// (see `create_empty_top_snapshot`).
    ///
    /// `AtomicI64` so `state::jitcode_for` can back-stamp the wrapper's
    /// allocated position onto an already-`Arc`-shared inner JitCode.
    pub index: std::sync::atomic::AtomicI64,

    // majit bytecode adapter extension.
    // These fields have no RPython JitCode counterpart. RPython's
    // bytecode operand stream references external dispatch tables
    // maintained by the blackhole builder / metainterp_sd; majit's
    // runtime JitCodeBuilder inlines the equivalent pools directly
    // on each JitCode for straight-line dispatch. Future parity work
    // is to lift these back out to a separate per-adapter structure.
    /// Runtime adapter pools not present on canonical `JitCode`
    /// (Phase I2 Step 2). Holds `opcodes`, `fn_ptrs`, `sub_jitcodes`,
    /// `assembler_targets` ordered by access frequency (hottest first)
    /// so the tight-loop benchmarks keep their cache locality.
    pub exec: JitCodeExecState,
}

impl Clone for JitCode {
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering;
        Self {
            name: self.name.clone(),
            code: self.code.clone(),
            c_num_regs_i: self.c_num_regs_i,
            c_num_regs_r: self.c_num_regs_r,
            c_num_regs_f: self.c_num_regs_f,
            constants_i: self.constants_i.clone(),
            constants_r: self.constants_r.clone(),
            constants_f: self.constants_f.clone(),
            jitdriver_sd: self.jitdriver_sd,
            fnaddr: self.fnaddr,
            calldescr: self.calldescr.clone(),
            index: std::sync::atomic::AtomicI64::new(self.index.load(Ordering::Relaxed)),
            exec: self.exec.clone(),
        }
    }
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
    /// descriptor data threaded through the shared `descrs` pool; pyre
    /// mirrors the shape via the `AssemblerToken` variant.
    pub(crate) fn call_assembler_target(&self, index: usize) -> (u64, *const ()) {
        let target = self
            .exec
            .descrs
            .get(index)
            .and_then(RuntimeBhDescr::as_assembler_token)
            .unwrap_or_else(|| {
                panic!("BC_CALL_ASSEMBLER_*: descrs[{index}] is not an AssemblerToken entry",)
            });
        (target.token_number, target.concrete_ptr)
    }

    /// Resolve `BC_CALL_*` / `BC_RESIDUAL_CALL_*` function-target
    /// descr. Mirrors RPython `blackhole.py:1225-1256` where the
    /// calling-convention descr travels through `descrs[idx]`; pyre
    /// additionally bundles the trace and concrete fn pointers in
    /// the `Call` variant because the call encoding pre-dates the
    /// RPython-orthodox register-fed function address.
    pub fn call_target(&self, index: usize) -> &JitCallTarget {
        match self.exec.descrs.get(index) {
            Some(RuntimeBhDescr::Call(target)) => target,
            other => {
                panic!("BC_CALL_*/RESIDUAL_CALL_*: descrs[{index}] is not a Call entry: {other:?}",)
            }
        }
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

    /// Inspect the trailing typed return opcode of a helper jitcode.
    ///
    /// RPython does not thread inline-helper result metadata through a
    /// side-table; callers recover the return type from the callee bytecode
    /// itself. `#[jit_inline]` helpers now terminate in the same typed
    /// `*_return` opcode, so consumers should read that trailing instruction
    /// instead of depending on an out-of-band `(return_reg, return_kind)` pair.
    pub fn trailing_return_info(&self) -> Option<(JitArgKind, u16)> {
        if self.code.last().copied() == Some(BC_VOID_RETURN) || self.code.len() < 3 {
            return None;
        }
        let opcode_pos = self.code.len() - 3;
        let opcode = self.code[opcode_pos];
        let src = u16::from_le_bytes([self.code[opcode_pos + 1], self.code[opcode_pos + 2]]);
        match opcode {
            BC_INT_RETURN => Some((JitArgKind::Int, src)),
            BC_REF_RETURN => Some((JitArgKind::Ref, src)),
            BC_FLOAT_RETURN => Some((JitArgKind::Float, src)),
            _ => None,
        }
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
    fn wellknown_bh_insns_stays_canonical_and_avoids_false_call_family_keys() {
        let insns = wellknown_bh_insns();
        assert!(
            !insns.contains_key("jump/L"),
            "wellknown_bh_insns must keep the canonical goto/L spelling",
        );
        assert!(
            !insns.contains_key("conditional_call_ir_v/iiIRd"),
            "helper-side BC_COND_CALL_VOID must not masquerade as \
             canonical conditional_call_ir_v/iiIRd",
        );
        assert!(
            !insns.contains_key("conditional_call_value_ir_r/riIRd>r"),
            "helper-side BC_COND_CALL_VALUE_REF must not masquerade as \
             canonical conditional_call_value_ir_r/riIRd>r",
        );
        assert!(
            !insns.contains_key("record_known_result_r_ir_v/riIRd"),
            "helper-side BC_RECORD_KNOWN_RESULT_REF must not masquerade as \
             canonical record_known_result_r_ir_v/riIRd",
        );
        assert!(
            !insns.contains_key("inline_call_ir_r/dIR>r"),
            "helper-side BC_INLINE_CALL adapter must not masquerade as \
             canonical inline_call_ir_r/dIR>r",
        );
        assert!(
            !insns.contains_key("inline_call_irf_f/dIRF>f"),
            "helper-side BC_INLINE_CALL adapter must not masquerade as \
             canonical inline_call_irf_f/dIRF>f",
        );
        // blackhole.py:1446,1485 `bhimpl_{get,set}field_vable_*` expect
        // the canonical key shape `{get,set}field_vable_x/{rd,rxd}`
        // with an explicit vable register in the leading `r` argcode.
        // pyre's helper-side `BC_*FIELD_VABLE_*` adapters emit the
        // descr + typed value operand only (the vable pointer is kept
        // in the `BH_VABLE_PTR` thread-local), so they cannot be wired
        // to the canonical bhimpl without first growing the vable
        // register argcode.
        assert!(
            !insns.contains_key("getfield_vable_i/rd>i"),
            "helper-side BC_GETFIELD_VABLE_I must not masquerade as \
             canonical getfield_vable_i/rd>i",
        );
        assert!(
            !insns.contains_key("getfield_vable_r/rd>r"),
            "helper-side BC_GETFIELD_VABLE_R must not masquerade as \
             canonical getfield_vable_r/rd>r",
        );
        assert!(
            !insns.contains_key("getfield_vable_f/rd>f"),
            "helper-side BC_GETFIELD_VABLE_F must not masquerade as \
             canonical getfield_vable_f/rd>f",
        );
        assert!(
            !insns.contains_key("setfield_vable_i/rid"),
            "helper-side BC_SETFIELD_VABLE_I must not masquerade as \
             canonical setfield_vable_i/rid",
        );
        assert!(
            !insns.contains_key("setfield_vable_r/rrd"),
            "helper-side BC_SETFIELD_VABLE_R must not masquerade as \
             canonical setfield_vable_r/rrd",
        );
        assert!(
            !insns.contains_key("setfield_vable_f/rfd"),
            "helper-side BC_SETFIELD_VABLE_F must not masquerade as \
             canonical setfield_vable_f/rfd",
        );
        // blackhole.py:1374,1390 `bhimpl_{get,set}arrayitem_vable_*`
        // canonical keys are `{get,set}arrayitem_vable_x/{ridd>x,rixdd}`
        // — vable register + index + (optional value) + fielddescr + arraydescr.
        // pyre's helper-side adapters fuse fielddescr+arraydescr into a
        // single `array_idx` descr and drop the vable register.
        assert!(
            !insns.contains_key("getarrayitem_vable_i/ridd>i"),
            "helper-side BC_GETARRAYITEM_VABLE_I must not masquerade as \
             canonical getarrayitem_vable_i/ridd>i",
        );
        assert!(
            !insns.contains_key("setarrayitem_vable_i/riidd"),
            "helper-side BC_SETARRAYITEM_VABLE_I must not masquerade as \
             canonical setarrayitem_vable_i/riidd",
        );
        // blackhole.py:1406 `bhimpl_arraylen_vable` canonical key
        // `arraylen_vable/rdd>i`; pyre fuses descrs and drops vable reg.
        assert!(
            !insns.contains_key("arraylen_vable/rdd>i"),
            "helper-side BC_ARRAYLEN_VABLE must not masquerade as \
             canonical arraylen_vable/rdd>i",
        );
        // blackhole.py:1547 `bhimpl_hint_force_virtualizable(r)`
        // canonical key `hint_force_virtualizable/r`. pyre's adapter
        // reads the vable from `BH_VABLE_PTR` instead.
        assert!(
            !insns.contains_key("hint_force_virtualizable/r"),
            "helper-side BC_HINT_FORCE_VIRTUALIZABLE must not masquerade \
             as canonical hint_force_virtualizable/r",
        );
    }

    #[test]
    fn canonical_build_jitcode_sizes_blackhole_register_files_without_conversion() {
        // Extract the upstream-common part of blackhole.py:312 setposition
        // (register sizing + constant copy) and apply it directly to the
        // canonical codewriter JitCode. Dispatch still needs the runtime
        // adapter JitCode for exec.* pools, but the register-file setup no
        // longer needs a build→runtime conversion just to match RPython's
        // `num_regs_* + len(constants_*)` logic.
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

        let mut bh = BlackholeInterpreter::new();
        bh.prepare_registers_for_canonical_jitcode(&bt, 0);

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
        assert!(bh.jitcode.code.is_empty());
    }
}
