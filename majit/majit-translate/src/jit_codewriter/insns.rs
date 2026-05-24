//! Canonical opcode-name table for jitcode bytecode dispatch.
//!
//! RPython parity: `rpython/jit/codewriter/assembler.py:21,221`
//! `Assembler.insns` lives in the same module as `Assembler`. Pyre's
//! mechanical mirror of `rpython/jit/codewriter/` is this directory
//! (`majit/majit-translate/src/jit_codewriter/`); the canonical home
//! for `wellknown_bh_insns` + `BC_*` constants + `insn_byte` is
//! therefore here, not in `majit-metainterp`.
//!
//! See `epic_e_task86_canonical_home_design_2026_05_04.md` for the
//! full design rationale + 5-slice mechanical-move plan. Slice #86c
//! moved the `BC_*` constants from `majit-metainterp::jitcode` to
//! this module; subsequent slices migrate `wellknown_bh_insns` /
//! `insn_byte` (#86d), import paths (#86e), classification doc
//! comments (#86f), and remove the back-compat re-exports (#88).
//!
//! PRE-EXISTING-ADAPTATION: pyre's runtime jitcode builder uses
//! fixed `BC_*` byte values (compile-time stable) instead of
//! RPython's dynamic `setdefault(key, len(self.insns))` allocation.
//! Compile-time stability is required because pyre serialises
//! `opcode_jitcodes.bin` at build time and the runtime decoder
//! reads those exact bytes. RPython runs the assembler at startup
//! and never serialises across process boundaries, so dynamic
//! allocation is fine there. Codex review Correction 1
//! (`epic_e_codex_review_2026_05_04.md`) flags that this byte-
//! stability adaptation should be documented at the table site —
//! see slice #86f.

use majit_ir::VecAssoc;

// Canonical RPython keys filling free low bytes (`blackhole.py:1149-1525`).
// One byte per `(opname, argcodes)` per `assembler.py:221
// setdefault(key, len(self.insns))` — alias-shared bytes would violate the
// 1:1 invariant that `pyjitpl.py:2230 setup_insns` enforces.
pub const BC_CAST_FLOAT_TO_INT: u8 = 0;
pub const BC_ARRAYLEN_GC: u8 = 1;
// (slot 2 free — `BC_ARRAYLEN_VABLE` already pinned at byte 74
//  in the vable group.)
pub const BC_GETINTERIORFIELD_GC_I: u8 = 3;
pub const BC_GETINTERIORFIELD_GC_R: u8 = 4;
pub const BC_GETINTERIORFIELD_GC_F: u8 = 5;
pub const BC_GETLISTITEM_GC_I: u8 = 6;
pub const BC_GETLISTITEM_GC_R: u8 = 7;
pub const BC_GETLISTITEM_GC_F: u8 = 8;
pub const BC_NEWLIST: u8 = 9;
pub const BC_NEWLIST_CLEAR: u8 = 10;
pub const BC_NEWLIST_HINT: u8 = 11;

pub const BC_LOOP_HEADER: u8 = 12;
pub const BC_ABORT: u8 = 13;
pub const BC_ABORT_PERMANENT: u8 = 14;
/// RPython `blackhole.py:962` `bhimpl_unreachable()` raises
/// `AssertionError("unreachable")`. Distinct from `BC_ABORT_PERMANENT`
/// which permits the interpreter to take over via
/// `DispatchError::RaiseException`.
pub const BC_UNREACHABLE: u8 = 19;
/// RPython `blackhole.py:913` `bhimpl_goto_if_not_int_is_true =
/// bhimpl_goto_if_not` aliases the handler.  `assembler.py:221
/// setdefault()` still allocates a distinct byte per `(opname,
/// argcodes)` key — `goto_if_not_int_is_true/iL` gets its own byte
/// here (alias); the canonical `goto_if_not/iL` lives at
/// [`BC_GOTO_IF_NOT`] = 18.  Both bytes route to the same handler.
pub const BC_GOTO_IF_NOT_INT_IS_TRUE: u8 = 15;
pub const BC_JUMP: u8 = 16;
pub const BC_INLINE_CALL: u8 = 17;
/// RPython `flatten.py:245` `opname = 'goto_if_not'` — the canonical
/// boolean-exitswitch branch.  `blackhole.py:865 bhimpl_goto_if_not`
/// is the handler shared with the
/// [`BC_GOTO_IF_NOT_INT_IS_TRUE`] alias per `blackhole.py:913
/// bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not`.  Slot 18
/// (formerly `BC_RESIDUAL_CALL_VOID`, freed by Slice 1c of
/// `pyre-call-family-canonical-migration.md`) now houses this
/// canonical key so each `(opname, argcodes)` per
/// `assembler.py:221 setdefault()` keeps its own byte.
pub const BC_GOTO_IF_NOT: u8 = 18;
pub const BC_MOVE_I: u8 = 21;
// Canonical RPython keys taking previously-freed slots 22-26 and 28-31
// (`blackhole.py:1149-1525` + `blackhole.py:1257-1296` conditional/known
// result + `blackhole.py:621-630` record_known_result + raw/indexed
// loads + bounds check).  Each `(opname, argcodes)` gets a distinct
// byte per `assembler.py:221 setdefault()` semantics.
pub const BC_RAW_LOAD_I: u8 = 20;
pub const BC_RAW_LOAD_F: u8 = 22;
pub const BC_GC_LOAD_INDEXED_I: u8 = 23;
pub const BC_GC_LOAD_INDEXED_F: u8 = 24;
pub const BC_CHECK_NEG_INDEX: u8 = 25;
pub const BC_CONDITIONAL_CALL_IR_V: u8 = 26;
pub const BC_CONDITIONAL_CALL_VALUE_IR_I: u8 = 28;
pub const BC_CONDITIONAL_CALL_VALUE_IR_R: u8 = 29;
pub const BC_RECORD_KNOWN_RESULT_I_IR_V: u8 = 30;
pub const BC_RECORD_KNOWN_RESULT_R_IR_V: u8 = 31;
// Ref-typed bytecodes
pub const BC_MOVE_R: u8 = 27;
// Float-typed bytecodes
pub const BC_MOVE_F: u8 = 33;
// slot 34 (formerly BC_CALL_FLOAT) freed by Slice 4 Phase B.4 — see
// the BC_CALL_REF (slot 28) note above for the canonical replacement.
// slot 35 (formerly BC_CALL_PURE_FLOAT) freed by Parity #14 Slice C.5
// — see slot 23 above.
// slots 38..=40 (formerly BC_CALL_MAY_FORCE_{INT,REF,FLOAT}) freed by
// Slice 4 Phase B.4 — the may_force policy now rides on
// `EffectInfo.extraeffect = EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` carried
// by the canonical `BC_RESIDUAL_CALL_{R,IR,IRF}_{I,R,F}` family
// (`effectinfo.py:201`).
// slot 41 (formerly BC_CALL_MAY_FORCE_VOID) freed by Slice 1c — same
// EffectInfo-carries-policy rationale as above; see slots 38..=40.
// slot 42 (formerly BC_CALL_RELEASE_GIL_INT) freed by Slice 4 Phase B.4
// — release-GIL policy rides on
// `EffectInfo.call_release_gil_target` (`effectinfo.py:255
// is_call_release_gil`).
// BC_CALL_RELEASE_GIL_REF (slot 43) intentionally absent:
// resoperation.py:1243-1244 (`# no such thing`) excludes
// CALL_RELEASE_GIL_R from the upstream opcode table.
// slot 44 (formerly BC_CALL_RELEASE_GIL_FLOAT) freed by Slice 4 Phase
// B.4 — see slot 42 above.
// slot 45 (formerly BC_CALL_RELEASE_GIL_VOID) freed by Slice 1c — see
// slot 42 above.
// slots 46..=48 (formerly BC_CALL_LOOPINVARIANT_{INT,REF,FLOAT}) freed
// by Slice 4 Phase B.4 — loop-invariant policy rides on
// `EffectInfo.extraeffect = EF_LOOPINVARIANT` (`effectinfo.py:202`).
// slot 49 (formerly BC_CALL_LOOPINVARIANT_VOID) freed by Slice 1c — see
// slots 46..=48 above.
pub const BC_CALL_ASSEMBLER_INT: u8 = 50;
pub const BC_CALL_ASSEMBLER_REF: u8 = 51;
pub const BC_CALL_ASSEMBLER_FLOAT: u8 = 52;
pub const BC_CALL_ASSEMBLER_VOID: u8 = 53;
pub const BC_LOAD_STATE_FIELD: u8 = 56;
pub const BC_STORE_STATE_FIELD: u8 = 57;
pub const BC_LOAD_STATE_ARRAY: u8 = 58;
pub const BC_STORE_STATE_ARRAY: u8 = 59;
pub const BC_LOAD_STATE_VARRAY: u8 = 60;
pub const BC_STORE_STATE_VARRAY: u8 = 61;
pub const BC_GETFIELD_VABLE_I: u8 = 62;
pub const BC_GETFIELD_VABLE_R: u8 = 63;
pub const BC_GETFIELD_VABLE_F: u8 = 64;
pub const BC_SETFIELD_VABLE_I: u8 = 65;
pub const BC_SETFIELD_VABLE_R: u8 = 66;
pub const BC_SETFIELD_VABLE_F: u8 = 67;
pub const BC_GETARRAYITEM_VABLE_I: u8 = 68;
pub const BC_GETARRAYITEM_VABLE_R: u8 = 69;
pub const BC_GETARRAYITEM_VABLE_F: u8 = 70;
pub const BC_SETARRAYITEM_VABLE_I: u8 = 71;
pub const BC_SETARRAYITEM_VABLE_R: u8 = 72;
pub const BC_SETARRAYITEM_VABLE_F: u8 = 73;
pub const BC_ARRAYLEN_VABLE: u8 = 74;
pub const BC_HINT_FORCE_VIRTUALIZABLE: u8 = 75;
/// RPython bhimpl_ref_return: callee returns a ref value.
pub const BC_REF_RETURN: u8 = 76;
/// blackhole.py bhimpl_raise: raise an exception from a ref register.
pub const BC_RAISE: u8 = 77;
/// blackhole.py bhimpl_reraise: re-raise exception_last_value.
pub const BC_RERAISE: u8 = 78;
// RPython jtransform.py:1685 — conditional_call_ir_v
pub const BC_COND_CALL_VOID: u8 = 79;
// RPython jtransform.py:1687 — conditional_call_value_ir_i / conditional_call_value_ir_r
pub const BC_COND_CALL_VALUE_INT: u8 = 80;
pub const BC_COND_CALL_VALUE_REF: u8 = 81;
// RPython jtransform.py:292 — record_known_result_i_ir_v / record_known_result_r_ir_v
pub const BC_RECORD_KNOWN_RESULT_INT: u8 = 82;
pub const BC_RECORD_KNOWN_RESULT_REF: u8 = 83;
/// pyjitpl.py opimpl_int_guard_value: promote int to constant via GUARD_VALUE.
pub const BC_INT_GUARD_VALUE: u8 = 84;
/// pyjitpl.py opimpl_ref_guard_value: promote ref to constant via GUARD_VALUE.
pub const BC_REF_GUARD_VALUE: u8 = 85;
/// pyjitpl.py opimpl_float_guard_value: promote float to constant via GUARD_VALUE.
pub const BC_FLOAT_GUARD_VALUE: u8 = 86;
/// blackhole.py:1066 bhimpl_jit_merge_point: portal merge point marker.
/// `iIRFIRF` form: jdindex byte is a `registers_i` pool slot index
/// (assembler.py:106-107 emit_const default path when `allow_short=False`
/// or value > 127).
pub const BC_JIT_MERGE_POINT: u8 = 87;
/// `cIRFIRF` form of `bhimpl_jit_merge_point` selected by
/// assembler.py:312 `USE_C_FORM` membership when jdindex fits in
/// `signed i8` (`-128..=127`). The jdindex byte is the raw signed
/// value (assembler.py:99-107 emit_const short branch +
/// blackhole.py:121-123 `argcode == 'c'` handler reads `signedord`).
pub const BC_JIT_MERGE_POINT_C: u8 = 131;
pub const BC_LIVE: u8 = 88;
pub const BC_CATCH_EXCEPTION: u8 = 89;
pub const BC_LAST_EXC_VALUE: u8 = 90;
/// RPython blackhole.py:987 `last_exception/>i`.
pub const BC_LAST_EXCEPTION: u8 = 129;
/// RPython blackhole.py:976-985 `goto_if_exception_mismatch/iL`.
pub const BC_GOTO_IF_EXCEPTION_MISMATCH: u8 = 130;
/// blackhole.py bhimpl_rvmprof_code: rvmprof enter/leave marker.
pub const BC_RVMPROF_CODE: u8 = 91;

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
pub const BC_GOTO_IF_NOT_INT_LT: u8 = 92;
pub const BC_GOTO_IF_NOT_INT_LE: u8 = 93;
pub const BC_GOTO_IF_NOT_INT_EQ: u8 = 94;
pub const BC_GOTO_IF_NOT_INT_NE: u8 = 95;
pub const BC_GOTO_IF_NOT_INT_GT: u8 = 96;
pub const BC_GOTO_IF_NOT_INT_GE: u8 = 97;
pub const BC_GOTO_IF_NOT_FLOAT_LT: u8 = 98;
pub const BC_GOTO_IF_NOT_FLOAT_LE: u8 = 99;
pub const BC_GOTO_IF_NOT_FLOAT_EQ: u8 = 100;
pub const BC_GOTO_IF_NOT_FLOAT_NE: u8 = 101;
pub const BC_GOTO_IF_NOT_FLOAT_GT: u8 = 102;
pub const BC_GOTO_IF_NOT_FLOAT_GE: u8 = 103;
pub const BC_GOTO_IF_NOT_PTR_EQ: u8 = 104;
pub const BC_GOTO_IF_NOT_PTR_NE: u8 = 105;
// blackhole.py:916-920 `bhimpl_goto_if_not_int_is_zero(a, target, pc)`:
// take target iff `a != 0`. jtransform.py:1212 `_rewrite_equality`
// folds `int_eq(x, 0)` into `int_is_zero(x)`; flatten.py:247 then
// specialises the bool exitswitch into `goto_if_not_int_is_zero/iL`.
pub const BC_GOTO_IF_NOT_INT_IS_ZERO: u8 = 106;

// blackhole.py:661-679 bhimpl_int_push / bhimpl_ref_push /
// bhimpl_float_push and matching pops — one-slot scratch for the
// cycle-break path emitted by flatten.py:326-332 `insert_renamings`.
pub const BC_INT_PUSH: u8 = 107;
pub const BC_REF_PUSH: u8 = 108;
pub const BC_FLOAT_PUSH: u8 = 109;
pub const BC_INT_POP: u8 = 110;
pub const BC_REF_POP: u8 = 111;
pub const BC_FLOAT_POP: u8 = 112;

pub const BC_INT_ADD: u8 = 113;
pub const BC_INT_SUB: u8 = 114;
pub const BC_INT_MUL: u8 = 115;
// 116, 117 — reserved (no opcode for `int_floordiv` / `int_mod`).
// `jtransform.py:576-577` rewrites both via `_do_builtin_call` to
// `direct_call(ll_int_py_div)` / `direct_call(ll_int_py_mod)` before
// jitcode emission; the bytecode dispatch path therefore has no
// `BC_INT_FLOORDIV` / `BC_INT_MOD` opcode.  Pyre's residual call at
// `majit-translate/src/codegen.rs::generated_binary_int_value`
// emits `CallI(ll_int_py_div, ...)` / `CallI(ll_int_py_mod, ...)`
// for the same effect.  Slot numbers stay reserved so opcode
// renumbering stays stable across this gap.
pub const BC_INT_AND: u8 = 118;
pub const BC_INT_OR: u8 = 119;
pub const BC_INT_XOR: u8 = 120;
pub const BC_INT_LSHIFT: u8 = 121;
pub const BC_INT_RSHIFT: u8 = 122;
pub const BC_INT_EQ: u8 = 123;
pub const BC_INT_NE: u8 = 124;
pub const BC_INT_LT: u8 = 125;
pub const BC_INT_LE: u8 = 126;
pub const BC_INT_GT: u8 = 127;
pub const BC_INT_GE: u8 = 128;
pub const BC_INT_NEG: u8 = 132;
pub const BC_FLOAT_ADD: u8 = 133;
pub const BC_FLOAT_SUB: u8 = 134;
pub const BC_FLOAT_MUL: u8 = 135;
pub const BC_FLOAT_TRUEDIV: u8 = 136;
pub const BC_FLOAT_NEG: u8 = 139;
pub const BC_FLOAT_ABS: u8 = 140;
pub const BC_INT_INVERT: u8 = 141;
pub const BC_UINT_RSHIFT: u8 = 142;
pub const BC_UINT_MUL_HIGH: u8 = 143;
pub const BC_UINT_LT: u8 = 144;
pub const BC_UINT_LE: u8 = 145;
pub const BC_UINT_GT: u8 = 146;
pub const BC_UINT_GE: u8 = 147;
// Ref/nullity primitives — RPython `blackhole.py:584-610`
// `bhimpl_{ptr_eq,ptr_ne,ptr_iszero,ptr_nonzero,instance_ptr_eq,instance_ptr_ne}`.
pub const BC_PTR_EQ: u8 = 151;
pub const BC_PTR_NE: u8 = 152;
pub const BC_INSTANCE_PTR_EQ: u8 = 153;
pub const BC_INSTANCE_PTR_NE: u8 = 154;
pub const BC_PTR_ISZERO: u8 = 155;
pub const BC_PTR_NONZERO: u8 = 156;
// Unary ptr nullity exitswitch specialisations — `blackhole.py:937-944`
// `bhimpl_goto_if_not_ptr_{iszero,nonzero}`.
pub const BC_GOTO_IF_NOT_PTR_ISZERO: u8 = 157;
pub const BC_GOTO_IF_NOT_PTR_NONZERO: u8 = 158;
// canonical residual_call_*_v opcodes — RPython `blackhole.py:1240-1255`
// `bhimpl_residual_call_{r,ir,irf}_v`. Distinct opcodes per argcode shape so
// `setup_insns` (`blackhole.rs:3241`) keeps its 1:1 opcode→key invariant.
// Slice 0 of `pyre-call-family-canonical-migration.md` reserves these slots
// in the 159-255 free range; emit-site migration lives in Slice 1.
pub const BC_RESIDUAL_CALL_R_V: u8 = 159;
pub const BC_RESIDUAL_CALL_IR_V: u8 = 160;
pub const BC_RESIDUAL_CALL_IRF_V: u8 = 161;
// canonical residual_call_*_i / *_r / *_f opcodes — RPython
// `blackhole.py:1208-1239 bhimpl_residual_call_{r,ir,irf}_{i,r,f}`.
// One opcode per (arg-shape, return-kind) pair so `setup_insns`
// keeps its 1:1 invariant.
pub const BC_RESIDUAL_CALL_R_I: u8 = 162;
pub const BC_RESIDUAL_CALL_IR_I: u8 = 163;
pub const BC_RESIDUAL_CALL_IRF_I: u8 = 164;
pub const BC_RESIDUAL_CALL_R_R: u8 = 165;
pub const BC_RESIDUAL_CALL_IR_R: u8 = 166;
pub const BC_RESIDUAL_CALL_IRF_R: u8 = 167;
pub const BC_RESIDUAL_CALL_IRF_F: u8 = 168;
// Typed return opcodes — RPython `blackhole.py:841-862`
// `bhimpl_int_return`, `bhimpl_float_return`, `bhimpl_void_return`.
// pyre's portal return is REF (see BC_REF_RETURN) but the insns map
// still needs every upstream return flavour so
// `pyjitpl.py:2240-2243` `setup_insns` fields do not fall back to
// `u8::MAX` sentinels.
pub const BC_INT_RETURN: u8 = 148;
pub const BC_FLOAT_RETURN: u8 = 149;
pub const BC_VOID_RETURN: u8 = 150;

// `getfield_gc_*` / `setfield_gc_*` — RPython `blackhole.py:1432-1481`
// `bhimpl_getfield_gc_{i,r,f}` + `bhimpl_setfield_gc_{i,r,f}` —
// descr-keyed instance field load/store on the GC heap.  Pyre emits
// these at build time from the assembler so the bytes are pinned
// canonical to keep the byte → opname map stable across build/runtime.
pub const BC_GETFIELD_GC_I: u8 = 169;
pub const BC_GETFIELD_GC_R: u8 = 170;
pub const BC_GETFIELD_GC_F: u8 = 171;
pub const BC_SETFIELD_GC_I: u8 = 172;
pub const BC_SETFIELD_GC_R: u8 = 173;

// `getarrayitem_gc_*` / `setarrayitem_gc_*` — RPython
// `blackhole.py:1330-1422`.  `bhimpl_getarrayitem_gc_{i,r,f}` only
// defines the `rid` (ref + int_idx + descr) argcode shape.  Pyre also
// emits a `rrd` (ref + ref_idx + descr) shape from
// `OpKind::ArrayRead { index_ty: Ref }` (`blackhole.rs:6938`); that
// shape has no upstream counterpart and lives in
// `pyre_extension_insns()` — not here.
pub const BC_GETARRAYITEM_GC_R_RID: u8 = 174;
pub const BC_SETARRAYITEM_GC_I: u8 = 176;
pub const BC_SETARRAYITEM_GC_R: u8 = 177;
pub const BC_SETARRAYITEM_GC_F: u8 = 178;
pub const BC_GETARRAYITEM_GC_I: u8 = 197;
pub const BC_GETARRAYITEM_GC_F: u8 = 198;

// `float_*` comparison ops — RPython `blackhole.py:721-746`
// `bhimpl_float_{lt,le,eq,ne,gt,ge}` — float comparisons returning int.
pub const BC_FLOAT_EQ: u8 = 179;
pub const BC_FLOAT_NE: u8 = 180;
pub const BC_FLOAT_LT: u8 = 181;
pub const BC_FLOAT_GT: u8 = 182;

// `cast_*_to_*` conversion ops — RPython `bhimpl_cast_ptr_to_int`
// (`blackhole.py:603-606`), `bhimpl_cast_int_to_ptr`
// (`blackhole.py:608-610`), `bhimpl_cast_int_to_float`
// (`blackhole.py:811-816`).
pub const BC_CAST_INT_TO_FLOAT: u8 = 183;
pub const BC_CAST_INT_TO_PTR: u8 = 184;
pub const BC_CAST_PTR_TO_INT: u8 = 185;

// `switch/id` — RPython `blackhole.py:954-960` `bhimpl_switch` —
// table-of-cases dispatch keyed by an int register + a descr selecting
// the case table.
pub const BC_SWITCH: u8 = 186;

// `inline_call_{r,ir,irf}_{i,r,v,f}` — RPython `blackhole.py:1279-1320`
// `bhimpl_inline_call_*` — inlined helper-jitcode dispatch.  The argcode
// suffix encodes which kind families the helper takes (R/IR/IRF) and
// what kind it returns (i/r/v/f).
pub const BC_INLINE_CALL_R_I: u8 = 187;
pub const BC_INLINE_CALL_R_R: u8 = 188;
pub const BC_INLINE_CALL_R_V: u8 = 189;
pub const BC_INLINE_CALL_IR_I: u8 = 190;
pub const BC_INLINE_CALL_IR_R: u8 = 191;
pub const BC_INLINE_CALL_IR_V: u8 = 192;
pub const BC_INLINE_CALL_IRF_F: u8 = 193;
pub const BC_INLINE_CALL_IRF_R: u8 = 194;
pub const BC_INLINE_CALL_IRF_I: u8 = 202;
pub const BC_INLINE_CALL_IRF_V: u8 = 203;

// Float `<=` / `>=` comparisons — RPython `blackhole.py:726-749`
// `bhimpl_float_{le,ge}` (le @726, ge @746).  `BC_FLOAT_LE` /
// `BC_FLOAT_GE` join the existing `BC_FLOAT_{EQ,NE,LT,GT}` quartet at
// 179-182.
pub const BC_FLOAT_LE: u8 = 204;
pub const BC_FLOAT_GE: u8 = 205;

// `setfield_gc_f/rfd` — RPython `blackhole.py:1481`
// `bhimpl_setfield_gc_f`.  Float-result sibling of
// `bhimpl_setfield_gc_{i,r}` which already pin `BC_SETFIELD_GC_{I,R}`
// at 172-173; deferred until now because pyre's float-field stores
// only landed alongside the float comparison fix.
pub const BC_SETFIELD_GC_F: u8 = 206;

// `getfield_gc_*_pure/rd>X` and `getarrayitem_gc_*_pure/rid>X` —
// RPython `blackhole.py:1339-1341, 1441-1443` aliases
// `bhimpl_get{field,arrayitem}_gc_{i,r,f}_pure =
// bhimpl_get{field,arrayitem}_gc_{i,r,f}`.  The Python aliases share
// a function body but each `(opname, argcodes)` key still gets its
// own opcode byte through `assembler.py:220
// self.insns.setdefault(key, len(self.insns))`.  The walker dispatch
// at `pyre/pyre-jit-trace/src/jitcode_dispatch.rs` routes the `_pure`
// keys to the same handler as the non-pure form, but the byte must
// be distinct or `pyjitpl.py:2230 setup_insns`'s
// `assert opcode_implementations[value] is None` and pyre's
// `jitcode_runtime.rs:272` duplicate-byte assert would fire.
pub const BC_GETFIELD_GC_I_PURE: u8 = 207;
pub const BC_GETFIELD_GC_R_PURE: u8 = 208;
pub const BC_GETFIELD_GC_F_PURE: u8 = 209;
pub const BC_GETARRAYITEM_GC_I_PURE: u8 = 210;
pub const BC_GETARRAYITEM_GC_R_PURE: u8 = 211;
pub const BC_GETARRAYITEM_GC_F_PURE: u8 = 212;

// pyre-only `abort/>r` — Ref-result variant of `abort/` (BC_ABORT = 13)
// emitted by `Assembler::encode_op`'s default branch when an `OpKind::
// Abort { result_kind: Ref }` reaches the assembler.  Lives in
// `pyre_extension_insns()` alongside `abort/` / `abort_permanent/`.
pub const BC_ABORT_RESULT_R: u8 = 195;

// pyre-only `vtable_method_ptr/rd>i` — emitted by `OpKind::
// VtableMethodPtr` (assembler.rs:2762).  RPython has no counterpart
// because Python dispatch goes through `cpu.bh_call_*` resolved at
// runtime; pyre's Rust port hits this only when a `dyn Trait` method
// pointer must be reified into the bytecode stream (backend-epic
// adaptation, see `blackhole.rs:8462-8474`).
pub const BC_VTABLE_METHOD_PTR: u8 = 196;

// pyre-only `getarrayitem_gc_r/rrd>r` — Ref-indexed array read.  RPython
// `blackhole.py:1333` only defines the `rid` (int-indexed) shape;
// pyre's `OpKind::ArrayRead { index_ty: Ref, .. }` lowering
// (`blackhole.rs:6938`) emits this `rrd` shape so the assembler can
// encode the typed-Ref index without an int-coerce.  Quarantined into
// `pyre_extension_insns()` until a porting pass lifts the ref-index
// case to upstream's `cpu.bh_call_*`-resolved dispatch shape.
pub const BC_GETARRAYITEM_GC_R_RRD: u8 = 175;

// `record_quasiimmut_field/rdd` — RPython `blackhole.py:1538-1545`
// `bhimpl_record_quasiimmut_field`.  Records that a quasi-immutable
// struct field has been read so the optimizer can install a
// `QUASIIMMUT_FIELD` guard before the corresponding `getfield_gc_*`.
// Argcodes: `rdd` (struct ref + fielddescr + mutatefielddescr).
pub const BC_RECORD_QUASIIMMUT_FIELD: u8 = 199;

// `assert_not_none/r` — RPython `pyjitpl.py:385-391 opimpl_assert_not_none`
// + `blackhole.py:613 bhimpl_assert_not_none`.  Records that `box` is
// known non-null so subsequent nullity-aware sites
// (`_establish_nullity`, KnownClass guards) can skip their own checks.
// Argcodes: `r` (struct ref).
pub const BC_ASSERT_NOT_NONE: u8 = 200;

// `record_exact_class/ri` — RPython `pyjitpl.py:393-410
// opimpl_record_exact_class` + `blackhole.py:616 bhimpl_record_exact_class`.
// Records that `box`'s class is exactly `cls` (vtable ref constant) so
// subsequent class-aware sites can skip the GuardClass.  Argcodes:
// `ri` (struct ref + class pointer as int), matching
// `blackhole.py:616 @arguments("r", "i")`.
pub const BC_RECORD_EXACT_CLASS: u8 = 201;

pub const MAX_HOST_CALL_ARITY: usize = 16;

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
pub fn insn_byte(key: &str) -> u8 {
    insn_byte_opt(key).unwrap_or_else(|| panic!("insn_byte: unregistered insns key {key:?}"))
}

/// Non-panicking lookup against the merged
/// `wellknown_bh_insns()` + `pyre_extension_insns()` table.  Returns
/// `Some(byte)` for canonical/extension keys (build/runtime-stable
/// `BC_*`) and `None` for translator-only keys.  `Assembler::get_opnum`
/// uses the `None` return to fall through to the
/// `assembler.py:221 setdefault(key, len(self.insns))` dynamic-byte
/// allocator, adjusted for pyre's fixed-byte table by scanning from
/// zero and skipping reserved bytes.  A `None` here is not a
/// fail-loud condition — it just means the key flows through the
/// dynamic allocator instead of a reserved byte.
///
/// Build-time `JitCodeBuilder::write_insn` callers use the panicking
/// [`insn_byte`] variant (every key the BH dispatcher knows about must
/// be registered here); the translator-pipeline assembler consults
/// this `_opt` variant so unregistered translator-only keys can take
/// the dynamic-allocation path without tripping the runtime decoder.
pub fn insn_byte_opt(key: &str) -> Option<u8> {
    use std::sync::OnceLock;
    static TABLE: OnceLock<VecAssoc<&'static str, u8>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        // Build-time consumers (`JitCodeBuilder::write_insn`) lookup any
        // canonical opname (RPython parity) AND any pyre-only Rust-
        // adaptation opname (`abort/`, `abort_permanent/`). Merging both
        // tables keeps every legal key resolvable from a single entry
        // point while the source of truth remains split — canonical
        // opnames live in `wellknown_bh_insns()`, pyre-only quarantine
        // lives in `pyre_extension_insns()`.
        let mut merged = wellknown_bh_insns();
        for (k, v) in pyre_extension_insns() {
            assert!(
                !merged.contains_key(k),
                "insn_byte: pyre extension key {k:?} collides with \
                 wellknown_bh_insns; the two tables must be disjoint",
            );
            merged.insert(k, v);
        }
        merged
    });
    table.get(key).copied()
}

/// Highest byte value reserved across `wellknown_bh_insns()` ∪
/// `pyre_extension_insns()`.  This is a diagnostic for pyre's fixed
/// `BC_*` range; translator-only opcode allocation must not use this
/// as a start offset because gaps below the high-water remain legal
/// dynamic bytes.
pub fn canonical_byte_high_water() -> u8 {
    use std::sync::OnceLock;
    static HIGH_WATER: OnceLock<u8> = OnceLock::new();
    *HIGH_WATER.get_or_init(|| {
        let mut max = 0u8;
        for (_, v) in wellknown_bh_insns() {
            if v > max {
                max = v;
            }
        }
        for (_, v) in pyre_extension_insns() {
            if v > max {
                max = v;
            }
        }
        max
    })
}

/// True when `byte` is pinned by either canonical RPython-mirror keys
/// (`wellknown_bh_insns`) or pyre-only extension keys
/// (`pyre_extension_insns`).  Multiple keys may intentionally share a
/// byte because some RPython blackhole handlers are aliases; this helper
/// only answers whether the byte is reserved, not whether the table is
/// one-to-one.
pub fn is_reserved_opcode_byte(byte: u8) -> bool {
    use std::sync::OnceLock;
    static RESERVED: OnceLock<[bool; 256]> = OnceLock::new();
    let reserved = RESERVED.get_or_init(|| {
        let mut reserved = [false; 256];
        for (_, value) in wellknown_bh_insns() {
            reserved[value as usize] = true;
        }
        for (_, value) in pyre_extension_insns() {
            reserved[value as usize] = true;
        }
        reserved
    });
    reserved[byte as usize]
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
pub fn wellknown_bh_insns() -> VecAssoc<&'static str, u8> {
    let mut m = VecAssoc::new();

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

    // `abort/` and `abort_permanent/` are pyre-only Rust adaptations and
    // live in `pyre_extension_insns()` (Task #94c) — their byte values
    // are still `BC_ABORT` / `BC_ABORT_PERMANENT`, but they are kept out
    // of the canonical-mirror table to keep `wellknown_bh_insns()` a
    // strict subset of RPython's `Assembler.insns`. See the comment on
    // `pyre_extension_insns` for the borrow-checker rationale.

    // RPython blackhole.py:962 `bhimpl_unreachable()` raises
    // `AssertionError("unreachable")`. Distinct opcode from
    // `abort_permanent/` so the interpreter fallback path cannot be taken.
    m.insert("unreachable/", BC_UNREACHABLE);

    // The 6 `*_state_*` keys were quarantined into
    // `pyre_extension_insns()` (Task #94b' / #94c). They model the
    // proc-macro-generated JIT-machine-state addressing scheme and have
    // no RPython counterpart — see the doc-comment on
    // `pyre_extension_insns` for the proc-macro/runtime-bridge
    // rationale.

    // Virtualizable operations — RPython canonical argcode shapes from
    // blackhole.py:1374-1409 and :1446-1495.  The legacy helper-side
    // `JitCodeBuilder::vable_*` methods still emit their old compact
    // payloads for the state-field macro path; Pyre's SSA assembler uses
    // the `*_with_base` methods that match these keys byte-for-byte.
    m.insert("getfield_vable_i/rd>i", BC_GETFIELD_VABLE_I);
    m.insert("getfield_vable_r/rd>r", BC_GETFIELD_VABLE_R);
    m.insert("getfield_vable_f/rd>f", BC_GETFIELD_VABLE_F);
    m.insert("setfield_vable_i/rid", BC_SETFIELD_VABLE_I);
    m.insert("setfield_vable_r/rrd", BC_SETFIELD_VABLE_R);
    m.insert("setfield_vable_f/rfd", BC_SETFIELD_VABLE_F);
    m.insert("getarrayitem_vable_i/ridd>i", BC_GETARRAYITEM_VABLE_I);
    m.insert("getarrayitem_vable_r/ridd>r", BC_GETARRAYITEM_VABLE_R);
    m.insert("getarrayitem_vable_f/ridd>f", BC_GETARRAYITEM_VABLE_F);
    m.insert("setarrayitem_vable_i/riidd", BC_SETARRAYITEM_VABLE_I);
    m.insert("setarrayitem_vable_r/rirdd", BC_SETARRAYITEM_VABLE_R);
    m.insert("setarrayitem_vable_f/rifdd", BC_SETARRAYITEM_VABLE_F);
    m.insert("arraylen_vable/rdd>i", BC_ARRAYLEN_VABLE);
    m.insert("hint_force_virtualizable/r", BC_HINT_FORCE_VIRTUALIZABLE);

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
    // assembler.py:163,181-220 builds the canonical key from the full
    // 7-arg shape (`jdindex + I/R/F + I/R/F`). assembler.py:312 places
    // `jit_merge_point` in `USE_C_FORM`, so the jdindex argcode is `c`
    // when the value fits in a signed byte and `i` (constants-pool
    // slot index) otherwise. Both forms reach the same
    // `bhimpl_jit_merge_point` (blackhole.py:1066) because the
    // `@arguments("i", ...)` decoder dispatches on the runtime argcode
    // (blackhole.py:113-123 `argtype == 'i'` branch).
    m.insert("jit_merge_point/cIRFIRF", BC_JIT_MERGE_POINT_C);
    m.insert("jit_merge_point/iIRFIRF", BC_JIT_MERGE_POINT);
    // RPython `blackhole.py:1240-1255` `bhimpl_residual_call_{r,ir,irf}_v`.
    // Slice 1c of `pyre-call-family-canonical-migration.md` retired the
    // legacy `BC_RESIDUAL_CALL_VOID` (=18) byte layout in favour of the
    // canonical `iRd / iIRd / iIRFd` argcode triple; the freed slot is
    // documented at the const-table site above.
    m.insert("residual_call_r_v/iRd", BC_RESIDUAL_CALL_R_V);
    m.insert("residual_call_ir_v/iIRd", BC_RESIDUAL_CALL_IR_V);
    m.insert("residual_call_irf_v/iIRFd", BC_RESIDUAL_CALL_IRF_V);
    m.insert("residual_call_r_i/iRd>i", BC_RESIDUAL_CALL_R_I);
    m.insert("residual_call_ir_i/iIRd>i", BC_RESIDUAL_CALL_IR_I);
    m.insert("residual_call_irf_i/iIRFd>i", BC_RESIDUAL_CALL_IRF_I);
    m.insert("residual_call_r_r/iRd>r", BC_RESIDUAL_CALL_R_R);
    m.insert("residual_call_ir_r/iIRd>r", BC_RESIDUAL_CALL_IR_R);
    m.insert("residual_call_irf_r/iIRFd>r", BC_RESIDUAL_CALL_IRF_R);
    m.insert("residual_call_irf_f/iIRFd>f", BC_RESIDUAL_CALL_IRF_F);
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
    // not truthfully expose.  The pyre-only nested-bytecode shape is
    // registered separately as `inline_call_pyre_nested/P` in
    // `pyre_extension_insns()`.

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
    // `int_floordiv/ii>i` / `int_mod/ii>i` intentionally absent —
    // `jtransform.py:576-577` rewrites via `_do_builtin_call`, so the
    // SSA-name → bytecode table never matches these.  See the
    // `BC_INT_AND` constants block above for the parity rationale.
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
    // `goto_if_not/iL`; `blackhole.py:913
    // bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not` aliases the
    // handler on the alias key `goto_if_not_int_is_true/iL`.
    // `assembler.py:221 setdefault()` allocates one byte per
    // `(opname, argcodes)`: the canonical key gets [`BC_GOTO_IF_NOT`]
    // and the alias key gets [`BC_GOTO_IF_NOT_INT_IS_TRUE`].  Both
    // bytes route to the same handler at runtime.
    m.insert("goto_if_not/iL", BC_GOTO_IF_NOT);
    m.insert("goto_if_not_int_is_true/iL", BC_GOTO_IF_NOT_INT_IS_TRUE);

    // GC heap field load/store — `blackhole.py:1432-1481` `bhimpl_
    // {get,set}field_gc_{i,r,f}`.  Argcodes: `rd` (struct ref + descr)
    // for gets, `r{i,r,f}d` (struct ref + value reg + descr) for sets.
    m.insert("getfield_gc_i/rd>i", BC_GETFIELD_GC_I);
    m.insert("getfield_gc_r/rd>r", BC_GETFIELD_GC_R);
    m.insert("getfield_gc_f/rd>f", BC_GETFIELD_GC_F);
    m.insert("setfield_gc_i/rid", BC_SETFIELD_GC_I);
    m.insert("setfield_gc_r/rrd", BC_SETFIELD_GC_R);
    m.insert("setfield_gc_f/rfd", BC_SETFIELD_GC_F);
    // RPython `blackhole.py:1441-1443` aliases `bhimpl_getfield_gc_{i,r,f}_pure
    // = bhimpl_getfield_gc_{i,r,f}` — pure-getter shape on quasi-immutable
    // descrs.  Each `(opname, argcodes)` key gets its own byte per
    // `assembler.py:220 setdefault(key, len(self.insns))`; `pyjitpl.py:2230
    // setup_insns` and pyre's `jitcode_runtime.rs:272` both assert no
    // duplicate bytes.  Walker dispatch routes `_pure` to the same handler.
    m.insert("getfield_gc_i_pure/rd>i", BC_GETFIELD_GC_I_PURE);
    m.insert("getfield_gc_r_pure/rd>r", BC_GETFIELD_GC_R_PURE);
    m.insert("getfield_gc_f_pure/rd>f", BC_GETFIELD_GC_F_PURE);

    // GC heap array element load/store — `blackhole.py:1330-1422`.
    // `bhimpl_getarrayitem_gc_{i,r,f}` only register the canonical
    // `rid` (ref + int_idx + descr) shape.  Pyre's pyre-only `rrd` (ref
    // + ref_idx + descr) shape lives in `pyre_extension_insns()`.
    // Setters are 4-byte payloads `ri{i,r,f}d` per
    // `bhimpl_setarrayitem_gc_{i,r,f}(cpu, array, index, newvalue,
    // arraydescr)` (`blackhole.py:1351-1359`): array (Ref), index (Int),
    // newvalue ({Int,Ref,Float}), arraydescr.
    m.insert("getarrayitem_gc_r/rid>r", BC_GETARRAYITEM_GC_R_RID);
    m.insert("setarrayitem_gc_i/riid", BC_SETARRAYITEM_GC_I);
    m.insert("setarrayitem_gc_r/rird", BC_SETARRAYITEM_GC_R);
    m.insert("setarrayitem_gc_f/rifd", BC_SETARRAYITEM_GC_F);
    m.insert("getarrayitem_gc_i/rid>i", BC_GETARRAYITEM_GC_I);
    m.insert("getarrayitem_gc_f/rid>f", BC_GETARRAYITEM_GC_F);
    // RPython `blackhole.py:1339-1341` aliases
    // `bhimpl_getarrayitem_gc_{i,r,f}_pure = bhimpl_getarrayitem_gc_{i,r,f}`
    // — pure-getter shape on immutable-array descrs.  Each
    // `(opname, argcodes)` key gets its own byte per `assembler.py:220
    // setdefault(key, len(self.insns))`; walker dispatch routes
    // `_pure` to the same handler as the non-pure form.
    m.insert("getarrayitem_gc_i_pure/rid>i", BC_GETARRAYITEM_GC_I_PURE);
    m.insert("getarrayitem_gc_r_pure/rid>r", BC_GETARRAYITEM_GC_R_PURE);
    m.insert("getarrayitem_gc_f_pure/rid>f", BC_GETARRAYITEM_GC_F_PURE);

    // Array length — `blackhole.py:1370` `bhimpl_arraylen_gc`
    // (`@arguments("cpu", "r", "d", returns="i")`).
    // `bhimpl_arraylen_vable` (`blackhole.py:1406`,
    // `@arguments("cpu", "r", "d", "d", returns="i")`) is pinned with
    // its vable siblings above at [`BC_ARRAYLEN_VABLE`] = 74.
    m.insert("arraylen_gc/rd>i", BC_ARRAYLEN_GC);

    // Interior-field load — `blackhole.py:1412-1418`
    // `bhimpl_getinteriorfield_gc_{i,r,f}`
    // (`@arguments("cpu", "r", "i", "d", returns="i"|"r"|"f")`).
    m.insert("getinteriorfield_gc_i/rid>i", BC_GETINTERIORFIELD_GC_I);
    m.insert("getinteriorfield_gc_r/rid>r", BC_GETINTERIORFIELD_GC_R);
    m.insert("getinteriorfield_gc_f/rid>f", BC_GETINTERIORFIELD_GC_F);

    // List item load — `blackhole.py:1196-1207`
    // `bhimpl_getlistitem_gc_{i,r,f}`
    // (`@arguments("cpu", "r", "i", "d", "d", returns="i"|"r"|"f")`).
    m.insert("getlistitem_gc_i/ridd>i", BC_GETLISTITEM_GC_I);
    m.insert("getlistitem_gc_r/ridd>r", BC_GETLISTITEM_GC_R);
    m.insert("getlistitem_gc_f/ridd>f", BC_GETLISTITEM_GC_F);

    // Quasi-immutable record — `blackhole.py:1538-1545`
    // `bhimpl_record_quasiimmut_field`.  Argcodes `rdd`: struct ref +
    // field descr + mutate descr (no result).
    m.insert("record_quasiimmut_field/rdd", BC_RECORD_QUASIIMMUT_FIELD);

    // Heapcache hint markers — `pyjitpl.py:385-410`.
    // `assert_not_none/r` records that `box` is non-null;
    // `record_exact_class/ri` records that `box`'s class equals the
    // `cls` constant (vtable pointer carried through the int bank).
    // `record_exact_class` is a no-op at
    // blackhole resume; `assert_not_none` checks the concrete ref just
    // like `blackhole.py:613`.  The trace-time dispatcher routes them through
    // `TraceCtx::trace_assert_not_none` /
    // `TraceCtx::trace_record_exact_class`, which gate on
    // `heap_cache.is_nullity_known` / `is_class_known` and bump
    // `HEAPCACHED_OPS` on cache hit per `pyjitpl.py:387-388 / 396-397`.
    // The `/ri` suffix matches `blackhole.py:616`; tracing reboxes a
    // constant int class pointer as a ConstPtr before recording.
    m.insert("assert_not_none/r", BC_ASSERT_NOT_NONE);
    m.insert("record_exact_class/ri", BC_RECORD_EXACT_CLASS);

    // Float comparisons — `blackhole.py:721-746`
    // `bhimpl_float_{lt,le,eq,ne,gt,ge}` — float pair → int (0/1).
    m.insert("float_eq/ff>i", BC_FLOAT_EQ);
    m.insert("float_ne/ff>i", BC_FLOAT_NE);
    m.insert("float_lt/ff>i", BC_FLOAT_LT);
    m.insert("float_le/ff>i", BC_FLOAT_LE);
    m.insert("float_gt/ff>i", BC_FLOAT_GT);
    m.insert("float_ge/ff>i", BC_FLOAT_GE);

    // Cross-kind casts — `bhimpl_cast_ptr_to_int` (`blackhole.py:603-606`),
    // `bhimpl_cast_int_to_ptr` (`blackhole.py:608-610`),
    // `bhimpl_cast_int_to_float` (`blackhole.py:811-816`),
    // `bhimpl_cast_float_to_int` (`blackhole.py:801-808`).
    m.insert("cast_int_to_float/i>f", BC_CAST_INT_TO_FLOAT);
    m.insert("cast_int_to_ptr/i>r", BC_CAST_INT_TO_PTR);
    m.insert("cast_ptr_to_int/r>i", BC_CAST_PTR_TO_INT);
    m.insert("cast_float_to_int/f>i", BC_CAST_FLOAT_TO_INT);

    // Switch dispatch — `blackhole.py:954-960` `bhimpl_switch`.
    // Argcodes `id`: int discriminator + descr selecting the case table.
    m.insert("switch/id", BC_SWITCH);

    // Inlined helper-jitcode dispatch — `blackhole.py:1279-1320`
    // `bhimpl_inline_call_*`.  Argcodes encode (a) which kind families
    // the helper takes after the descr — `R` (ref-only), `IR`
    // (int+ref), `IRF` (int+ref+float) — and (b) the helper's return
    // kind — `>i`/`>r`/`>f` for typed returns, no suffix for void.  The
    // descr `d` slot resolves to the helper's `JitCode` at runtime.
    m.insert("inline_call_r_i/dR>i", BC_INLINE_CALL_R_I);
    m.insert("inline_call_r_r/dR>r", BC_INLINE_CALL_R_R);
    m.insert("inline_call_r_v/dR", BC_INLINE_CALL_R_V);
    m.insert("inline_call_ir_i/dIR>i", BC_INLINE_CALL_IR_I);
    m.insert("inline_call_ir_r/dIR>r", BC_INLINE_CALL_IR_R);
    m.insert("inline_call_ir_v/dIR", BC_INLINE_CALL_IR_V);
    m.insert("inline_call_irf_i/dIRF>i", BC_INLINE_CALL_IRF_I);
    m.insert("inline_call_irf_r/dIRF>r", BC_INLINE_CALL_IRF_R);
    m.insert("inline_call_irf_f/dIRF>f", BC_INLINE_CALL_IRF_F);
    m.insert("inline_call_irf_v/dIRF", BC_INLINE_CALL_IRF_V);

    // List construction — `blackhole.py:1161-1194`
    // `bhimpl_newlist{,_clear,_hint}`
    // (`@arguments("cpu", "i", "d", "d", "d", "d", returns="r")`):
    // length + 4 descrs (struct + length + items + array) → ref result.
    m.insert("newlist/idddd>r", BC_NEWLIST);
    m.insert("newlist_clear/idddd>r", BC_NEWLIST_CLEAR);
    m.insert("newlist_hint/idddd>r", BC_NEWLIST_HINT);

    // Raw memory load — `blackhole.py:1512-1518`
    // `bhimpl_raw_load_{i,f}` (`@arguments("cpu", "i", "i", "d",
    // returns="i"|"f")`): addr + offset + arraydescr.
    m.insert("raw_load_i/iid>i", BC_RAW_LOAD_I);
    m.insert("raw_load_f/iid>f", BC_RAW_LOAD_F);

    // GC indexed load — `blackhole.py:1519-1525`
    // `bhimpl_gc_load_indexed_{i,f}` (`@arguments("cpu", "r", "i", "i",
    // "i", "i", returns="i"|"f")`): base ref + index + scale + base_ofs +
    // bytes.
    m.insert("gc_load_indexed_i/riiii>i", BC_GC_LOAD_INDEXED_I);
    m.insert("gc_load_indexed_f/riiii>f", BC_GC_LOAD_INDEXED_F);

    // Negative-index normalisation — `blackhole.py:1149-1158`
    // `bhimpl_check_neg_index` (`@arguments("cpu", "r", "i", "d",
    // returns="i")`): array ref + index + arraydescr → wrapped index.
    m.insert("check_neg_index/rid>i", BC_CHECK_NEG_INDEX);

    // Conditional residual call — `blackhole.py:1258-1296`
    // `bhimpl_conditional_call_ir_v` (`@arguments("cpu", "i", "i", "I",
    // "R", "d")`) is fired iff the int `condition` is non-zero;
    // `bhimpl_conditional_call_value_ir_{i,r}` returns the `value`
    // operand when the helper isn't fired so the result is always
    // typed.
    m.insert("conditional_call_ir_v/iiIRd", BC_CONDITIONAL_CALL_IR_V);
    m.insert(
        "conditional_call_value_ir_i/iiIRd>i",
        BC_CONDITIONAL_CALL_VALUE_IR_I,
    );
    m.insert(
        "conditional_call_value_ir_r/riIRd>r",
        BC_CONDITIONAL_CALL_VALUE_IR_R,
    );

    // Record-known-result — `blackhole.py:621-630`
    // `bhimpl_record_known_result_{i,r}_ir_v`
    // (`@arguments("cpu", "i"|"r", "i", "I", "R", "d")`): tells the
    // tracer that the helper's result equals the pre-supplied operand,
    // so the call is elided on the trace.
    m.insert(
        "record_known_result_i_ir_v/iiIRd",
        BC_RECORD_KNOWN_RESULT_I_IR_V,
    );
    m.insert(
        "record_known_result_r_ir_v/riIRd",
        BC_RECORD_KNOWN_RESULT_R_IR_V,
    );

    m
}

/// Pyre-only opcodes that have NO RPython counterpart and arise from
/// Rust language adaptations permitted by `CLAUDE.md` ("Rust language
/// adaptations are permitted ONLY when unavoidable AND minimal" / "the
/// proc-macro/runtime bridge").
///
/// Two adaptation sources land here:
///
/// 1. **Borrow-checker tracing-abort signal** (`abort/`,
///    `abort_permanent/`).  RPython's tracing dispatch (`pyjitpl.py`)
///    bails out via Python exceptions — `SwitchToBlackhole`,
///    `ContinueRunningNormally`, `AssertionError("unreachable")` — and
///    the GC/exception machinery rebuilds the dispatch state on the
///    bailout side. Pyre cannot unwind through the trace loop because
///    the recording state (`PyreSym`, `TraceCtx`, per-kind register
///    banks, symbolic stack) is held by `&mut` references that must
///    remain sound on the unwind path; even `panic::catch_unwind` cannot
///    safely resume because the borrow checker sees the references as
///    still-live. Pyre's `JitCodeMachine` therefore emits explicit
///    `abort/` / `abort_permanent/` opcodes that the dispatch loop
///    converts into `DispatchError::AbortTrace` /
///    `DispatchError::AbortPermanentTrace` `Result` returns.
///
/// 2. **Proc-macro JIT-machine state addressing** (the 6 `*_state_*`
///    keys).  Pyre uses Rust proc-macros (`#[jit_interp]`,
///    `#[jit_driver]`, `state_fields = { ... }`) to generate the entire
///    JIT machine from a single function definition. RPython's
///    metaprogramming is runtime / annotator-driven and has no
///    proc-macro counterpart. The generated machine carries a
///    `JitCodeSym` whose state-field slots are accessed by flat slot
///    index (`d` argcode) without an explicit virtualizable-pointer
///    register — the canonical `setfield_vable_*` shape (`r` vable-ptr
///    + `d` FieldDescr) does not apply because the entire machine IS
///    the vable, accessed via the implicit `self` of the proc-macro-
///    generated handler functions.  Migration to canonical `*_vable_*`
///    is feasible (see `epic_e_task94b_prereq_audit_2026_05_04.md`) but
///    requires materializing `state` as a vable-ptr register +
///    synthesizing FieldDescr/ArrayDescr/LenDescr objects for every
///    state slot — a 4-6 session proc-macro refactor with non-obvious
///    failure modes. Per CLAUDE.md, the proc-macro bridge is itself
///    a permitted Rust adaptation, so quarantining the 6 keys is the
///    orthodox shape: keep `wellknown_bh_insns()` strictly canonical,
///    keep the proc-macro state addressing here.
///
/// All extension keys retain their fixed `BC_*` byte values in the same
/// number-space as the canonical opcodes; only the catalogue is split
/// from `wellknown_bh_insns()`.  `insn_byte` merges this table with
/// `wellknown_bh_insns()` at OnceLock init time, so build-time
/// `write_insn(...)` callers resolve transparently. Runtime dispatch
/// reads `BC_*` constants directly and is unaffected.
pub fn pyre_extension_insns() -> VecAssoc<&'static str, u8> {
    let mut m = VecAssoc::new();
    // Borrow-checker abort signals.
    m.insert("abort/", BC_ABORT);
    m.insert("abort_permanent/", BC_ABORT_PERMANENT);
    // Proc-macro JIT-machine state addressing — emit sites are in
    // `majit-macros/src/jit_interp/jitcode_lower.rs`. Argcodes:
    //   `d` = state-slot index (u16); `i` = int register (u16).
    //   Array variants carry an extra `i` for the index register
    //   before the destination/source slot.
    m.insert("load_state_field/di", BC_LOAD_STATE_FIELD);
    m.insert("store_state_field/di", BC_STORE_STATE_FIELD);
    m.insert("load_state_array/dii", BC_LOAD_STATE_ARRAY);
    m.insert("store_state_array/dii", BC_STORE_STATE_ARRAY);
    m.insert("load_state_varray/dii", BC_LOAD_STATE_VARRAY);
    m.insert("store_state_varray/dii", BC_STORE_STATE_VARRAY);
    // PRE-EXISTING-ADAPTATION: pyre nested-bytecode `inline_call`.
    //
    // RPython's canonical `inline_call_*` keys (`/dIRF>i`, `/dIR>r`, …)
    // dispatch through a real C-ABI `fnaddr` stored on `BhDescr::JitCode`.
    // Pyre does not compile inlined helpers into separate native
    // functions — guard-failure resume must re-interpret the helper as
    // nested bytecode.  Byte 17 (`BC_INLINE_CALL`) is reused for this
    // pyre-only handler (`handler_inline_call_pyre_nested`).  The `P`
    // pseudo-argcode is opaque from the canonical RPython argcodes
    // alphabet: payload is `sub_idx u16 + num_args u16 + num_args ×
    // (kind u8, src u16, dst u16) + 3 × (return slot u16; u16::MAX = None)`.
    // Generic walkers must consult `decode_op_at` (which knows `P`) for
    // length, not the canonical argcodes table.
    m.insert("inline_call_pyre_nested/P", BC_INLINE_CALL);
    // PRE-EXISTING-ADAPTATION: pyre `call_assembler_*` adapters.
    //
    // `JitCodeBuilder::call_assembler_{int,ref,float,void}_like`
    // (`assembler.rs:3370,3429,3451,3489`) emits a pyre-only flat
    // payload: typed `[target_idx u16, dst u16, num_args u16,
    // (kind u8, reg u16) × num_args]`; void omits `dst`.  RPython has
    // no `bhimpl_call_assembler_*`; pyre re-interprets the recorded
    // operation by direct-calling `target.concrete_ptr` via the
    // shared `call_int_function` / `call_void_function` helpers.
    // `P` pseudo-argcode mirrors `inline_call_pyre_nested`'s opaque
    // pyre-payload classification.
    m.insert("call_assembler_int_pyre/P", BC_CALL_ASSEMBLER_INT);
    m.insert("call_assembler_ref_pyre/P", BC_CALL_ASSEMBLER_REF);
    m.insert("call_assembler_float_pyre/P", BC_CALL_ASSEMBLER_FLOAT);
    m.insert("call_assembler_void_pyre/P", BC_CALL_ASSEMBLER_VOID);
    // PRE-EXISTING-ADAPTATION: pyre `cond_call` / `record_known_result`
    // adapters.
    //
    // `JitCodeBuilder::call_cond_like` / `call_cond_value_like`
    // (`assembler.rs:2642,2660`) emit a pyre-only flat payload that
    // does not match canonical `iiIRd` / `riIRd>r` argcodes.  The
    // `_pyre/P` handlers split semantically:
    //
    //   * `cond_call_*_pyre` (`blackhole.rs:9965` onward) execute the
    //     conditional call directly, mirroring upstream
    //     `bhimpl_conditional_call_ir_v` /
    //     `bhimpl_conditional_call_value_ir_{i,r}`
    //     (`blackhole.py:1257-1276`).
    //   * `record_known_result_*_pyre` (`blackhole.rs:10068` onward)
    //     are no-ops that skip the operand bytes, mirroring the
    //     `pass`-bodied `bhimpl_record_known_result_{i,r}_ir_v`
    //     (`blackhole.py:621-628`).
    //
    // Producers: `majit-macros/src/jit_interp/jitcode_lower.rs:2166-2458`,
    // `pyre/pyre-jit/src/jit/assembler.rs:1181`.
    m.insert("cond_call_void_pyre/P", BC_COND_CALL_VOID);
    m.insert("cond_call_value_int_pyre/P", BC_COND_CALL_VALUE_INT);
    m.insert("cond_call_value_ref_pyre/P", BC_COND_CALL_VALUE_REF);
    m.insert("record_known_result_int_pyre/P", BC_RECORD_KNOWN_RESULT_INT);
    m.insert("record_known_result_ref_pyre/P", BC_RECORD_KNOWN_RESULT_REF);
    // pyre-only `abort/>r` — Ref-result variant of `abort/`.  Emitted by
    // `Assembler::encode_op`'s default branch when an `OpKind::Abort` with
    // `result_kind: Ref` reaches the assembler.  The blackhole-side
    // handler `handler_abort_result_marker_r` (`blackhole.rs:6042`) is a
    // no-op that advances past the trailing destination register byte;
    // the abort signal proper goes through `abort/` (`BC_ABORT = 13`).
    m.insert("abort/>r", BC_ABORT_RESULT_R);
    // pyre-only `vtable_method_ptr/rd>i` — emitted by
    // `OpKind::VtableMethodPtr` (`assembler.rs:2762`).  RPython's dispatch
    // resolves dyn-method addresses through `cpu.bh_call_*` at runtime
    // and never needs to reify a method pointer into the bytecode stream;
    // pyre's Rust port hits this only on `dyn Trait` calls, where the
    // backend epic must look up the vtable slot itself.
    m.insert("vtable_method_ptr/rd>i", BC_VTABLE_METHOD_PTR);
    // pyre-only `getarrayitem_gc_r/rrd>r` — Ref-indexed GC array read.
    // RPython `blackhole.py:1333 bhimpl_getarrayitem_gc_r` only
    // registers the canonical `rid` (int-indexed) shape.  Pyre's
    // `OpKind::ArrayRead { index_ty: Ref, .. }` lowering
    // (`blackhole.rs:6938`) emits this `rrd` shape so the assembler can
    // encode the typed-Ref index without an int-coerce; converging
    // requires lifting the ref-index case to upstream's
    // `cpu.bh_call_*`-resolved dispatch shape.
    m.insert("getarrayitem_gc_r/rrd>r", BC_GETARRAYITEM_GC_R_RRD);
    m
}
