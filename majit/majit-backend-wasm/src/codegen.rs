/// IR → wasm bytecode compilation.
///
/// Generates a wasm module from majit IR ops using `wasm-encoder`.
/// Generated function signature: `(param $frame_ptr i32) (result i32)`
///
/// Frame layout in shared linear memory:
///   offset 0:       fail_index (i64)
///   offset 8:       slot[0] (i64)
///   offset 16:      slot[1] (i64)
///   ...
///   CALL_AREA_OFS:  func_ptr (i64)   — used by jit_call trampoline
///   CALL_AREA_OFS+8: num_args (i64)
///   CALL_AREA_OFS+16: arg[0] (i64)
///   CALL_AREA_OFS+24: arg[1] (i64)
///   ...
///   CALL_RESULT_OFS: result (i64)    — written by host after call
use std::collections::HashMap;

use majit_backend::BackendError;
use majit_gc::header::{GcHeader, TYPE_ID_MASK};
use majit_ir::{InputArg, Op, OpCode, OpRef, Type};
use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, InstructionSink, MemArg, MemoryType, Module, TypeSection, ValType,
};

/// Frame slot byte offset: slot[i] is at frame_ptr + 8 + i * 8.
const FRAME_SLOT_BASE: u64 = 8;
const SLOT_SIZE: u64 = 8;

/// Scratch i64 locals reserved past the value locals for `emit_umulhi`
/// (al, ah, bl, bh, mid1).
const UMULHI_SCRATCH: u32 = 5;

/// Call area layout (fixed offsets from frame_ptr).
const CALL_RESULT_OFS: u64 = 2000;
const CALL_FUNC_OFS: u64 = 2008;
const CALL_NARGS_OFS: u64 = 2016;
const CALL_ARGS_OFS: u64 = 2024;

/// Minimum frame allocation size in bytes to accommodate the call area.
pub const MIN_FRAME_BYTES: usize = 2024 + 16 * 8; // 16 max call args

fn mem64(offset: u64) -> MemArg {
    MemArg {
        offset,
        align: 3,
        memory_index: 0,
    }
}

fn memarg(offset: u64, align: u32) -> MemArg {
    MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

/// Emit a width-correct integer load. The element address (i32) must be on
/// the stack; the result is an i64, sign- or zero-extended from `size`
/// bytes. Word-sized fields are 4 bytes on wasm32 (`isize`/`usize`/pointer),
/// 8 bytes on 64-bit; reading a fixed 8 bytes here would fold in the next
/// field's bytes on wasm32.
fn emit_sized_int_load(sink: &mut InstructionSink<'_>, offset: u64, size: usize, signed: bool) {
    match (size, signed) {
        (8, _) => {
            sink.i64_load(mem64(offset));
        }
        (4, true) => {
            sink.i32_load(memarg(offset, 2));
            sink.i64_extend_i32_s();
        }
        (4, false) => {
            sink.i32_load(memarg(offset, 2));
            sink.i64_extend_i32_u();
        }
        (2, true) => {
            sink.i32_load16_s(memarg(offset, 1));
            sink.i64_extend_i32_s();
        }
        (2, false) => {
            sink.i32_load16_u(memarg(offset, 1));
            sink.i64_extend_i32_u();
        }
        (1, true) => {
            sink.i32_load8_s(memarg(offset, 0));
            sink.i64_extend_i32_s();
        }
        (1, false) => {
            sink.i32_load8_u(memarg(offset, 0));
            sink.i64_extend_i32_u();
        }
        _ => {
            sink.i64_load(mem64(offset));
        }
    }
}

/// Emit a width-correct integer store. The stack must hold
/// `[addr_i32, value_i64]`; the low `size` bytes of the value are stored.
/// A fixed 8-byte store would clobber the adjacent field/item (or run past
/// the array end) for word-sized fields and pointer array items on wasm32.
fn emit_sized_int_store(sink: &mut InstructionSink<'_>, offset: u64, size: usize) {
    match size {
        8 => {
            sink.i64_store(mem64(offset));
        }
        4 => {
            sink.i32_wrap_i64();
            sink.i32_store(memarg(offset, 2));
        }
        2 => {
            sink.i32_wrap_i64();
            sink.i32_store16(memarg(offset, 1));
        }
        1 => {
            sink.i32_wrap_i64();
            sink.i32_store8(memarg(offset, 0));
        }
        _ => {
            sink.i64_store(mem64(offset));
        }
    }
}

/// `(field_size, is_signed)` from an op's FieldDescr; defaults to word-sized
/// signed when the descr is absent.
fn field_size_sign_from_descr(op: &Op) -> (usize, bool) {
    let descr = op.getdescr();
    if let Some(fd) = descr.as_ref().and_then(|d| d.as_field_descr()) {
        return (fd.field_size(), fd.is_field_signed());
    }
    (std::mem::size_of::<usize>(), true)
}

/// `(item_size, is_signed)` from an op's ArrayDescr; defaults to 8-byte
/// signed when the descr is absent.
fn array_item_size_sign_from_descr(op: &Op) -> (usize, bool) {
    op.with_array_descr(|ad| (ad.item_size(), ad.is_item_signed()))
        .unwrap_or((8, true))
}

/// llsupport/gc.py:563 GcLLDescr_framework
///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
/// Looks up the materialized table populated by the runner from the
/// active gc_ll_descr. RPython resolves the same value via
/// `cpu.gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr`.
fn lookup_typeid_from_classptr(table: &HashMap<i64, u32>, classptr: usize) -> Option<u32> {
    table.get(&(classptr as i64)).copied()
}

/// Information about a guard exit collected during pre-scan.
pub struct GuardExit {
    pub fail_index: u32,
    pub fail_arg_refs: Vec<OpRef>,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,
    /// `op.descr` snapshot — passed through to `WasmFailDescr.meta_descr`
    /// so `get_latest_descr_arc` can return the canonical metainterp Arc
    /// (parity with dynasm/cranelift's `meta_descr` forwarding).
    pub meta_descr: Option<majit_ir::DescrRef>,
}

/// Pre-fetched GC-type-guard metadata for the wasm codegen.
///
/// RPython's `genop_guard_guard_*` methods call into
/// `self.cpu.gc_ll_descr` at codegen time to obtain the TYPE_INFO
/// table base, the `infobits` offset / byte mask, the subclassrange
/// field offset, and the `(subclassrange_min, subclassrange_max)`
/// bounds for the constant expected-class pointer. The wasm backend
/// has no direct handle on a `GcAllocator` at this layer, so the
/// caller (`WasmBackend::compile_loop`) pre-fetches each of those
/// values and bundles them here.
///
/// Parity references:
///  * `llsupport/gc.py:162` / `gc.py:318` — `supports_guard_gc_type`
///  * `llsupport/gc.py:592` — `get_translated_info_for_typeinfo`
///  * `llsupport/gc.py:619` — `get_translated_info_for_guard_is_object`
///  * `x86/assembler.py:1951` — `cpu.subclassrange_min_offset`
///  * `x86/assembler.py:1971-1974` — constant-time
///    `(vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max)`
///
/// The default sets `supports_guard_gc_type = false`, matching
/// `AbstractCPU.supports_guard_gc_type` in `backend/model.py:21`; the
/// codegen arms assert this flag before reading any other field.
#[derive(Default)]
pub struct GuardGcTypeInfo {
    pub supports_guard_gc_type: bool,
    /// `get_translated_info_for_typeinfo()` = (base, shift, sizeof_ti).
    pub base_type_info: usize,
    pub shift_by: u8,
    pub sizeof_ti: usize,
    /// `get_translated_info_for_guard_is_object()`
    ///     = (infobits_offset, T_IS_RPYTHON_INSTANCE_BYTE).
    pub infobits_offset: usize,
    pub is_object_flag: u8,
    /// `cpu.subclassrange_min_offset` (x86/assembler.py:1951).
    pub subclassrange_min_offset: usize,
    /// `(vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max)`
    /// looked up by constant classptr. Empty when
    /// `supports_guard_gc_type == false`.
    pub subclass_ranges: HashMap<i64, (i64, i64)>,
}

/// Check if any op in the trace is a CALL variant.
fn has_call_ops(ops: &[Op]) -> bool {
    // Allocation ops (`New*`, `Newstr`/`Newunicode`) also reach the host via
    // the `jit_call` trampoline, so the import must be present for them too.
    ops.iter().any(|op| {
        op.opcode.is_call()
            || matches!(
                op.opcode,
                OpCode::New
                    | OpCode::NewWithVtable
                    | OpCode::NewArray
                    | OpCode::NewArrayClear
                    | OpCode::Newstr
                    | OpCode::Newunicode
            )
    })
}

fn collect_guards_and_vars(inputargs: &[InputArg], ops: &[Op]) -> (Vec<GuardExit>, u32) {
    let mut guards = Vec::new();
    let mut max_var: u32 = 0;

    for ia in inputargs {
        if ia.index + 1 > max_var {
            max_var = ia.index + 1;
        }
    }

    let mut fail_index = 0u32;
    for op in ops {
        if op.pos.get() != OpRef::NONE
            && !op.pos.get().is_constant()
            && op.pos.get().raw() + 1 > max_var
        {
            max_var = op.pos.get().raw() + 1;
        }
        if op.opcode == OpCode::Label {
            for a in op.getarglist().iter() {
                let a = a.to_opref();
                if a != OpRef::NONE && !a.is_constant() && a.raw() + 1 > max_var {
                    max_var = a.raw() + 1;
                }
            }
        }

        if op.opcode.is_guard() || op.opcode == OpCode::Finish {
            let fail_args: Vec<OpRef> = op
                .getfailargs()
                .map(|fa| fa.iter().map(|a| a.to_opref()).collect())
                .unwrap_or_else(|| op.getarglist().iter().map(|a| a.to_opref()).collect());
            let fail_arg_types = op
                .get_fail_arg_types()
                .unwrap_or_else(|| fail_args.iter().map(|_| Type::Int).collect());

            guards.push(GuardExit {
                fail_index,
                fail_arg_refs: fail_args,
                fail_arg_types,
                is_finish: op.opcode == OpCode::Finish,
                meta_descr: op.getdescr(),
            });
            fail_index += 1;
        }
    }

    (guards, max_var)
}

/// Build a wasm module from majit IR.
pub fn build_wasm_module(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &majit_ir::VecAssoc<u32, i64>,
    vtable_offset: Option<usize>,
    classptr_to_typeid: &HashMap<i64, u32>,
    guard_gc_type_info: &GuardGcTypeInfo,
    alloc_fn_ptr: i64,
    alloc_array_fn_ptr: i64,
) -> Result<(Vec<u8>, Vec<GuardExit>), BackendError> {
    let (guards, num_vars) = collect_guards_and_vars(inputargs, ops);
    let needs_call = has_call_ops(ops);

    let mut module = Module::new();

    // Type section
    let mut types = TypeSection::new();
    // Type 0: trace function (param i32) -> (result i32)
    types.ty().function(vec![ValType::I32], vec![ValType::I32]);
    if needs_call {
        // Type 1: jit_call trampoline (param i32) -> ()
        types.ty().function(vec![ValType::I32], vec![]);
    }
    module.section(&types);

    // Import section
    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "memory",
        MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        },
    );
    if needs_call {
        // Import jit_call trampoline as function index 0
        imports.import("env", "jit_call", EntityType::Function(1));
    }
    module.section(&imports);

    // Function section
    let mut functions = FunctionSection::new();
    functions.function(0); // type 0
    module.section(&functions);

    // Export section: trace function index depends on whether we imported jit_call
    let trace_func_idx = if needs_call { 1 } else { 0 };
    let mut exports = ExportSection::new();
    exports.export("trace", ExportKind::Func, trace_func_idx);
    module.section(&exports);

    // Code section
    let mut codes = CodeSection::new();
    let jit_call_idx = if needs_call { Some(0u32) } else { None };
    let func = build_function(
        inputargs,
        ops,
        constants,
        num_vars,
        jit_call_idx,
        vtable_offset,
        classptr_to_typeid,
        guard_gc_type_info,
        alloc_fn_ptr,
        alloc_array_fn_ptr,
    )?;
    codes.function(&func);
    module.section(&codes);

    Ok((module.finish(), guards))
}

fn build_function(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &majit_ir::VecAssoc<u32, i64>,
    num_vars: u32,
    jit_call_idx: Option<u32>,
    vtable_offset: Option<usize>,
    classptr_to_typeid: &HashMap<i64, u32>,
    guard_gc_type_info: &GuardGcTypeInfo,
    alloc_fn_ptr: i64,
    alloc_array_fn_ptr: i64,
) -> Result<Function, BackendError> {
    // Value locals occupy `1 ..= num_vars`; reserve `UMULHI_SCRATCH` extra i64
    // locals past them (`num_vars+1 ..= num_vars+UMULHI_SCRATCH`) as scratch for
    // the `UintMulHigh` 32-bit-split expansion (`emit_umulhi`).
    let mut func = Function::new(vec![(num_vars + UMULHI_SCRATCH, ValType::I64)]);
    let mut sink = func.instructions();

    // Load inputs from frame into locals
    for ia in inputargs {
        let local_idx = 1 + ia.index;
        let offset = FRAME_SLOT_BASE + ia.index as u64 * SLOT_SIZE;
        sink.local_get(0)
            .i64_load(mem64(offset))
            .local_set(local_idx);
    }

    // A peeled loop arrives as `[preamble..][LABEL][body..][JUMP]`: the
    // preamble runs once on entry, the LABEL is the loop-back target, and
    // JUMP branches back to it. Emit the `loop` at the LABEL (not the top)
    // so the preamble is not re-executed every iteration; wrap everything in
    // a `block` so guard exits `br` out to the function epilogue. Use the
    // LAST label (a peeled trace may carry an outer entry label plus the
    // inner loop header).
    let loop_label_idx = ops.iter().rposition(|op| op.opcode == OpCode::Label);
    let has_loop = loop_label_idx.is_some();
    if has_loop {
        sink.block(BlockType::Empty);
    }

    let mut guard_idx = 0u32;
    let mut in_loop_body = false;

    for (op_idx, op) in ops.iter().enumerate() {
        if Some(op_idx) == loop_label_idx {
            sink.loop_(BlockType::Empty);
            in_loop_body = true;
        }
        // Depth (from statement level) of the enclosing `block` that guard
        // exits `br` to: preamble = 0, loop body = 1 (the `loop` sits in
        // between). `None` for straight-line traces (no block emitted).
        let block_exit_depth = match (has_loop, in_loop_body) {
            (false, _) => None,
            (true, false) => Some(0u32),
            (true, true) => Some(1u32),
        };
        match op.opcode {
            OpCode::Label => {}

            OpCode::Jump => {
                // The jump rebinds the loop's label args to the jump args — a
                // parallel move. A jump arg may read a target local that another
                // pair overwrites (e.g. the swap `x, y = y, x` → x<-y, y<-x), so
                // resolving-then-storing each pair in turn would feed a clobbered
                // value to a later read. Do all reads first (push every resolved
                // jump arg onto the operand stack), then all writes (pop into the
                // targets in reverse, the stack being LIFO).
                let label_args = find_label_args(ops);
                let jump_args = op.getarglist();
                let n = jump_args.len().min(label_args.len());
                for jump_arg in jump_args.iter().take(n) {
                    emit_resolve(&mut sink, constants, jump_arg.to_opref());
                }
                for i in (0..n).rev() {
                    sink.local_set(1 + label_args[i].raw());
                }
                sink.br(0);
            }

            OpCode::Finish => {
                emit_guard_exit(&mut sink, constants, guard_idx, op);
                if let Some(d) = block_exit_depth {
                    sink.br(d);
                }
                guard_idx += 1;
            }

            // ── Guards ──
            OpCode::GuardTrue => {
                emit_guard_true(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardFalse => {
                emit_guard_false(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardValue => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                sink.i64_ne();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardNonnull => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i64_eqz();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardIsnull => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i64_const(0);
                sink.i64_ne();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardClass | OpCode::GuardNonnullClass => {
                // x86/assembler.py:1880-1891 _cmp_guard_class:
                //   offset = self.cpu.vtable_offset
                //   if offset is not None: CMP(mem(loc_ptr, offset), classptr)
                //   else:
                //       assert isinstance(loc_classptr, ImmedLoc)
                //       expected_typeid = gc_ll_descr.
                //           get_typeid_from_classptr_if_gcremovetypeptr(...)
                //       _cmp_guard_gc_type(loc_ptr, ImmedLoc(expected_typeid))
                if let Some(off_usize) = vtable_offset {
                    let off = off_usize as u64;
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64(); // struct ptr (i64) → i32 address
                    // The typeptr (`ob_type`) is a pointer-width field: 4
                    // bytes on wasm32. Reading it as i64 would fold in the
                    // following field's bytes and never match the class
                    // immediate. Load 4 bytes and zero-extend.
                    sink.i32_load(MemArg {
                        offset: off,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                    sink.i64_ne();
                } else {
                    // gcremovetypeptr fallback (assembler.py:1893-1901):
                    //   on x86_64 the typeid is a 32-bit value at offset 0.
                    let class_arg = op.arg(1).to_opref();
                    // history.py:227 — inline-Const carries its class pointer directly.
                    let classptr = class_arg.const_int_value().expect(
                        "_cmp_guard_class: gcremovetypeptr requires \
                             loc_classptr to be a ConstInt immediate \
                             (aarch64/regalloc.py:829 op.getarg(1).getint())",
                    );
                    let expected_typeid =
                        lookup_typeid_from_classptr(classptr_to_typeid, classptr as usize).expect(
                            "GuardClass: vtable_offset is None but the wasm \
                                 backend has no gc_ll_descr.\
                                 get_typeid_from_classptr_if_gcremovetypeptr",
                        );
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i32_const(expected_typeid as i32);
                    sink.i32_ne();
                }
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardNoOverflow => {
                // RPython: 0 args — overflow flag implicit from preceding ovf op.
                // Wasm MVP doesn't detect overflow, so always passes.
                guard_idx += 1;
            }
            OpCode::GuardOverflow => {
                // Always fails (no overflow detected in wasm MVP).
                emit_guard_exit(&mut sink, constants, guard_idx, op);
                match block_exit_depth {
                    Some(d) => {
                        sink.br(d);
                    }
                    // Straight-line: return directly so the following ops and
                    // the terminal Finish do not overwrite this exit.
                    None => {
                        sink.local_get(0);
                        sink.return_();
                    }
                }
                guard_idx += 1;
            }
            // Guards that always pass in wasm MVP (no force-token /
            // invalidation tracking yet).
            OpCode::GuardNotInvalidated | OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
                guard_idx += 1;
            }
            OpCode::GuardNoException => {
                // x86/assembler.py:1799-1801 generate_guard_no_exception:
                // `CMP(pos_exception, imm0)` — fail the guard when a pending
                // exception is present, keyed on the exception TYPE slot
                // (pos_exception), the same slot GuardException reads and the
                // one llgraph's `last_exception is not None` tests. The slot
                // lives in the host's shared linear memory; load it by absolute
                // address (the trace imports env.memory).
                sink.i32_const(crate::jit_exc_type_addr() as i32);
                sink.i64_load(mem64(0));
                sink.i64_const(0);
                sink.i64_ne();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardException => {
                // x86/assembler.py:1808-1815 genop_guard_guard_exception:
                //   load pos_exception; CMP expected; guard on equal; then
                //   _store_and_reset_exception: resloc = pos_exc_value;
                //   pos_exception = 0; pos_exc_value = 0.
                let exc_type_addr = crate::jit_exc_type_addr() as i32;
                let exc_value_addr = crate::jit_exc_value_addr() as i32;
                sink.i32_const(exc_type_addr);
                sink.i64_load(mem64(0));
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i64_ne();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
                // Success path: capture the caught exception into the result
                // var, then clear both slots.
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    sink.i32_const(exc_value_addr);
                    sink.i64_load(mem64(0));
                    sink.local_set(1 + vi);
                }
                sink.i32_const(exc_type_addr);
                sink.i64_const(0);
                sink.i64_store(mem64(0));
                sink.i32_const(exc_value_addr);
                sink.i64_const(0);
                sink.i64_store(mem64(0));
            }

            // ── Integer arithmetic ──
            OpCode::IntAdd => emit_binop(&mut sink, constants, op, BinOp::I64Add),
            OpCode::IntSub => emit_binop(&mut sink, constants, op, BinOp::I64Sub),
            OpCode::IntMul => emit_binop(&mut sink, constants, op, BinOp::I64Mul),
            OpCode::IntFloorDiv => emit_binop(&mut sink, constants, op, BinOp::I64DivS),
            OpCode::IntMod => emit_binop(&mut sink, constants, op, BinOp::I64RemS),
            OpCode::IntAnd => emit_binop(&mut sink, constants, op, BinOp::I64And),
            OpCode::IntOr => emit_binop(&mut sink, constants, op, BinOp::I64Or),
            OpCode::IntXor => emit_binop(&mut sink, constants, op, BinOp::I64Xor),
            OpCode::IntLshift => emit_binop(&mut sink, constants, op, BinOp::I64Shl),
            OpCode::IntRshift => emit_binop(&mut sink, constants, op, BinOp::I64ShrS),
            OpCode::UintRshift => emit_binop(&mut sink, constants, op, BinOp::I64ShrU),
            // High 64 bits of the unsigned 64×64→128 product. The optimizer
            // emits this for division/modulo-by-constant strength reduction;
            // wasm has no mul-high instruction, so expand via 32-bit split.
            OpCode::UintMulHigh => emit_umulhi(&mut sink, constants, op, num_vars),

            // Overflow variants: compute result + overflow flag
            OpCode::IntAddOvf => emit_ovf_binop(&mut sink, constants, op, BinOp::I64Add),
            OpCode::IntSubOvf => emit_ovf_binop(&mut sink, constants, op, BinOp::I64Sub),
            OpCode::IntMulOvf => emit_ovf_binop(&mut sink, constants, op, BinOp::I64Mul),

            // ── Integer comparisons (signed) ──
            OpCode::IntLt => emit_cmp(&mut sink, constants, op, CmpOp::I64LtS),
            OpCode::IntLe => emit_cmp(&mut sink, constants, op, CmpOp::I64LeS),
            OpCode::IntEq => emit_cmp(&mut sink, constants, op, CmpOp::I64Eq),
            OpCode::IntNe => emit_cmp(&mut sink, constants, op, CmpOp::I64Ne),
            OpCode::IntGt => emit_cmp(&mut sink, constants, op, CmpOp::I64GtS),
            OpCode::IntGe => emit_cmp(&mut sink, constants, op, CmpOp::I64GeS),

            // ── Integer comparisons (unsigned) ──
            OpCode::UintLt => emit_cmp(&mut sink, constants, op, CmpOp::I64LtU),
            OpCode::UintLe => emit_cmp(&mut sink, constants, op, CmpOp::I64LeU),
            OpCode::UintGt => emit_cmp(&mut sink, constants, op, CmpOp::I64GtU),
            OpCode::UintGe => emit_cmp(&mut sink, constants, op, CmpOp::I64GeU),

            // ── Pointer comparisons ──
            OpCode::PtrEq | OpCode::InstancePtrEq => {
                emit_cmp(&mut sink, constants, op, CmpOp::I64Eq);
            }
            OpCode::PtrNe | OpCode::InstancePtrNe => {
                emit_cmp(&mut sink, constants, op, CmpOp::I64Ne);
            }

            // ── Unary ops ──
            OpCode::IntNeg => emit_unary_vi(
                &mut sink,
                constants,
                op,
                |s| {
                    s.i64_const(0);
                },
                |s| {
                    s.i64_sub();
                },
            ),
            OpCode::IntInvert => emit_unary_vi(
                &mut sink,
                constants,
                op,
                |s| {
                    s.i64_const(-1);
                },
                |s| {
                    s.i64_xor();
                },
            ),
            OpCode::IntIsTrue => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_const(0);
                    sink.i64_ne();
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::IntIsZero => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_eqz();
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }

            // ── Extended integer ops ──
            OpCode::IntSignext => {
                // int_signext(val, num_bytes): sign-extend from num_bytes width
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    // num_bytes is arg(1), typically a constant
                    let arg1 = op.arg(1).to_opref();
                    let num_bytes = if arg1.is_constant() {
                        arg1.inline_const_bits()
                            .or_else(|| constants.get(&arg1.raw()).copied())
                            .unwrap_or(8)
                    } else {
                        8 // default to no-op
                    };
                    let shift = 64 - num_bytes * 8;
                    if shift > 0 && shift < 64 {
                        sink.i64_const(shift);
                        sink.i64_shl();
                        sink.i64_const(shift);
                        sink.i64_shr_s();
                    }
                    sink.local_set(1 + vi);
                }
            }
            OpCode::IntForceGeZero => {
                // max(val, 0)
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    // if val < 0, use 0; else use val
                    // Wasm: local.tee + i64.const 0 + local.get + i64.lt_s + select
                    let tmp_local = 1 + vi; // reuse result local as temp
                    sink.local_tee(tmp_local);
                    sink.i64_const(0);
                    sink.local_get(tmp_local);
                    sink.i64_const(0);
                    sink.i64_lt_s();
                    sink.select();
                    sink.local_set(1 + vi);
                }
            }

            // ── Float comparisons ──
            OpCode::FloatLt => emit_float_cmp(&mut sink, constants, op, FloatCmp::Lt),
            OpCode::FloatLe => emit_float_cmp(&mut sink, constants, op, FloatCmp::Le),
            OpCode::FloatEq => emit_float_cmp(&mut sink, constants, op, FloatCmp::Eq),
            OpCode::FloatNe => emit_float_cmp(&mut sink, constants, op, FloatCmp::Ne),
            OpCode::FloatGt => emit_float_cmp(&mut sink, constants, op, FloatCmp::Gt),
            OpCode::FloatGe => emit_float_cmp(&mut sink, constants, op, FloatCmp::Ge),

            // ── Float floor/mod ──
            OpCode::FloatFloorDiv => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                    sink.f64_reinterpret_i64();
                    sink.f64_div();
                    sink.f64_floor();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }

            // ── Float/Int conversions ──
            OpCode::CastFloatToInt => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    sink.i64_trunc_sat_f64_s();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::CastIntToFloat => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_convert_i64_s();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::ConvertFloatBytesToLonglong | OpCode::ConvertLonglongBytesToFloat => {
                // These are bitcast (no-op on the i64 representation)
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.local_set(1 + vi);
                }
            }

            // ── Pointer/Int conversions ──
            OpCode::CastPtrToInt | OpCode::CastIntToPtr | OpCode::CastOpaquePtr => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.local_set(1 + vi);
                }
            }

            // ── SameAs (forwarding) ──
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.local_set(1 + vi);
                }
            }

            // ── Field access (direct memory operations) ──
            OpCode::GetfieldGcI | OpCode::GetfieldGcPureI | OpCode::GetfieldRawI => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // struct ptr (i64)
                    sink.i32_wrap_i64(); // convert to i32 address
                    let field_offset = field_offset_from_descr(op);
                    let (size, signed) = field_size_sign_from_descr(op);
                    emit_sized_int_load(&mut sink, field_offset, size, signed);
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetfieldGcR | OpCode::GetfieldGcPureR | OpCode::GetfieldRawR => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    // Load as i32 (pointer on wasm32) and extend to i64
                    sink.i32_load(MemArg {
                        offset: field_offset,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // struct ptr
                sink.i32_wrap_i64();
                let field_offset = field_offset_from_descr(op);
                emit_resolve(&mut sink, constants, op.arg(1).to_opref()); // value
                let (size, _signed) = field_size_sign_from_descr(op);
                emit_sized_int_store(&mut sink, field_offset, size);
            }

            // ── Float field access ──
            OpCode::GetfieldGcF | OpCode::GetfieldGcPureF | OpCode::GetfieldRawF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    sink.f64_load(MemArg {
                        offset: field_offset,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }

            // ── Array access ──
            OpCode::ArraylenGc => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // array ptr
                    sink.i32_wrap_i64();
                    let len_offset = array_len_offset_from_descr(op);
                    sink.i64_load(mem64(len_offset));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcPureI | OpCode::GetarrayitemRawI => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // addr = base + base_size + index * item_size
                    emit_array_addr(&mut sink, constants, op);
                    let (item_size, signed) = array_item_size_sign_from_descr(op);
                    emit_sized_int_load(&mut sink, 0, item_size, signed);
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcPureR | OpCode::GetarrayitemRawR => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_array_addr(&mut sink, constants, op);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetarrayitemGcF | OpCode::GetarrayitemGcPureF | OpCode::GetarrayitemRawF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // A Float item is 8 bytes; load it as f64 and carry its bit
                    // pattern in the i64 value slot (the IntArray/value-slot ABI
                    // is i64).
                    emit_array_addr(&mut sink, constants, op);
                    sink.f64_load(MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                emit_array_addr(&mut sink, constants, op);
                emit_resolve(&mut sink, constants, op.arg(2).to_opref()); // value
                // A Ref item is pointer-width (4 bytes on wasm32). Storing a
                // fixed 8 bytes would clobber the next item, or run past the
                // array end on the last item and corrupt the heap. A Float
                // item is 8 bytes, so its bit pattern stores via the i64 path.
                let (item_size, _signed) = array_item_size_sign_from_descr(op);
                emit_sized_int_store(&mut sink, 0, item_size);
            }

            // ── Interior field access ──
            OpCode::GetinteriorfieldGcI | OpCode::GetinteriorfieldGcR => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // getinteriorfield(array, index, offset)
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // array ptr
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    // Simplified: use field_offset directly (RPython computes base+index*itemsize+offset)
                    let (size, signed) = field_size_sign_from_descr(op);
                    emit_sized_int_load(&mut sink, field_offset, size, signed);
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GetinteriorfieldGcF => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // array ptr
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    sink.f64_load(MemArg {
                        offset: field_offset,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::SetinteriorfieldGc => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i32_wrap_i64();
                let field_offset = field_offset_from_descr(op);
                emit_resolve(&mut sink, constants, op.arg(2).to_opref()); // value
                let (size, _signed) = field_size_sign_from_descr(op);
                emit_sized_int_store(&mut sink, field_offset, size);
            }

            // ── String/Unicode ops (direct memory access) ──
            OpCode::Strlen | OpCode::Unicodelen => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    // Length at offset 8 (after ob_type pointer on wasm32)
                    sink.i64_load(mem64(8));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::Strgetitem | OpCode::Unicodegetitem => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // str[index]: base + header_size + index
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref()); // index
                    sink.i32_wrap_i64();
                    sink.i32_add();
                    // String data starts after header (assume 16 bytes: ob_type + length)
                    sink.i32_load8_u(MemArg {
                        offset: 16,
                        align: 0,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }

            // ── GC memory ops ──
            OpCode::GcLoadI => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let offset = field_offset_from_descr(op);
                    sink.i64_load(mem64(offset));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GcLoadR => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let offset = field_offset_from_descr(op);
                    sink.i32_load(MemArg {
                        offset,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i64_extend_i32_u();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::GcStore => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i32_wrap_i64();
                let offset = field_offset_from_descr(op);
                emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                sink.i64_store(mem64(offset));
            }

            // ── Raw memory access ──
            OpCode::RawLoadI => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // ptr
                    sink.i32_wrap_i64();
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref()); // offset
                    sink.i32_wrap_i64();
                    sink.i32_add();
                    sink.i64_load(mem64(0));
                    sink.local_set(1 + vi);
                }
            }
            OpCode::RawStore => {
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i32_wrap_i64();
                emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                sink.i32_wrap_i64();
                sink.i32_add();
                emit_resolve(&mut sink, constants, op.arg(2).to_opref());
                sink.i64_store(mem64(0));
            }

            // ── Exception handling ──
            OpCode::SaveException | OpCode::SaveExcClass | OpCode::RestoreException => {
                // No-op in wasm MVP — exception state is managed by the host.
            }

            // ── Conditional calls ──
            OpCode::CondCallN | OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                // GC write barriers and conditional void calls — no-op in wasm.
            }

            // x86/assembler.py:1919-1922 genop_guard_guard_gc_type:
            // GUARD_GC_TYPE: args[0] = object ref, args[1] = expected
            // type_id. The majit runtime stores the typeid in the GC
            // header word placed immediately before the object payload
            // (`majit_gc::header::GcHeader::tid_and_flags`, lower 32
            // bits). The cranelift backend lowers the same op this way
            // (compiler.rs GuardGcType branch). This is NOT the RPython
            // gcremovetypeptr layout — pyre's GC keeps the typeid in the
            // header, not at `obj[0]`.
            OpCode::GuardGcType => {
                let _ = classptr_to_typeid; // typeid is already an immediate
                if op.num_args() >= 2 {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    // header address = obj - GcHeader::SIZE
                    sink.i64_const(GcHeader::SIZE as i64);
                    sink.i64_sub();
                    sink.i32_wrap_i64();
                    // Load 8-byte header word (tid_and_flags)
                    sink.i64_load(mem64(0));
                    // Mask lower TYPE_ID_BITS to extract the type id
                    sink.i64_const(TYPE_ID_MASK as i64);
                    sink.i64_and();
                    // Compare against expected_typeid (arg1 — already an
                    // i64 in the constant pool or a frame slot).
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                    sink.i64_ne();
                    emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                }
                guard_idx += 1;
            }
            // x86/assembler.py:1924-1943 genop_guard_guard_is_object.
            //     assert self.cpu.supports_guard_gc_type
            //     [loc_object, loc_typeid] = locs
            //     if IS_X86_32:
            //         self.mc.MOVZX16(loc_typeid, mem(loc_object, 0))
            //     else:
            //         self.mc.MOV32(loc_typeid, mem(loc_object, 0))
            //     base_type_info, shift_by, sizeof_ti = (
            //         self.cpu.gc_ll_descr
            //             .get_translated_info_for_typeinfo())
            //     infobits_offset, IS_OBJECT_FLAG = (
            //         self.cpu.gc_ll_descr
            //             .get_translated_info_for_guard_is_object())
            //     loc_infobits = addr_add(imm(base_type_info),
            //                             loc_typeid,
            //                             scale=shift_by,
            //                             offset=infobits_offset)
            //     self.mc.TEST8(loc_infobits, imm(IS_OBJECT_FLAG))
            //     self.guard_success_cc = rx86.Conditions['NZ']
            //     self.implement_guard(guard_token)
            OpCode::GuardIsObject => {
                // assembler.py:1925 assert self.cpu.supports_guard_gc_type
                assert!(
                    guard_gc_type_info.supports_guard_gc_type,
                    "x86/assembler.py:1925: assert self.cpu.\
                     supports_guard_gc_type (GcAllocator has not \
                     installed a TYPE_INFO layout)"
                );
                // assembler.py:1931-1932 MOV32 loc_typeid, mem(loc_object, 0).
                // majit's GC header sits at obj - GcHeader::SIZE; the
                // typeid occupies the lower TYPE_ID_BITS of that word.
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i64_const(GcHeader::SIZE as i64);
                sink.i64_sub();
                sink.i32_wrap_i64();
                sink.i64_load(mem64(0));
                sink.i64_const(TYPE_ID_MASK as i64);
                sink.i64_and();
                // Stack: [..., loc_typeid]

                // assembler.py:1938-1939 addr_add(imm(base_type_info),
                //     loc_typeid, scale=shift_by, offset=infobits_offset)
                if guard_gc_type_info.shift_by > 0 {
                    sink.i64_const(guard_gc_type_info.shift_by as i64);
                    sink.i64_shl();
                }
                sink.i64_const(guard_gc_type_info.base_type_info as i64);
                sink.i64_add();
                sink.i64_const(guard_gc_type_info.infobits_offset as i64);
                sink.i64_add();
                sink.i32_wrap_i64();
                // Stack: [..., loc_infobits(i32 addr)]

                // assembler.py:1940 TEST8 [loc_infobits], IS_OBJECT_FLAG
                sink.i32_load8_u(MemArg {
                    offset: 0,
                    align: 0,
                    memory_index: 0,
                });
                sink.i32_const(guard_gc_type_info.is_object_flag as i32);
                sink.i32_and();
                // assembler.py:1942 guard_success_cc = Conditions['NZ']:
                // guard passes when byte & flag != 0; fail when == 0.
                sink.i32_eqz();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            // x86/assembler.py:1945-1980 genop_guard_guard_subclass.
            //     assert self.cpu.supports_guard_gc_type
            //     [loc_object, loc_check_against_class, loc_tmp] = locs
            //     offset = self.cpu.vtable_offset
            //     offset2 = self.cpu.subclassrange_min_offset
            //     if offset is not None:
            //         self.mc.MOV_rm(loc_tmp, (loc_object, offset))
            //         self.mc.MOV_rm(loc_tmp, (loc_tmp, offset2))
            //     else:
            //         self.mc.MOV32(loc_tmp, mem(loc_object, 0))
            //         base_type_info, shift_by, sizeof_ti = (
            //             gc_ll_descr.get_translated_info_for_typeinfo())
            //         self.mc.MOV(loc_tmp, addr_add(
            //             imm(base_type_info), loc_tmp,
            //             scale=shift_by,
            //             offset=sizeof_ti + offset2))
            //     vtable_ptr = loc_check_against_class.getint()
            //     vtable_ptr = rffi.cast(rclass.CLASSTYPE, vtable_ptr)
            //     check_min = vtable_ptr.subclassrange_min
            //     check_max = vtable_ptr.subclassrange_max
            //     self.mc.SUB_ri(loc_tmp, check_min)
            //     self.mc.CMP_ri(loc_tmp, check_max - check_min)
            //     self.guard_success_cc = Conditions['B']
            //     self.implement_guard(guard_token)
            OpCode::GuardSubclass => {
                // assembler.py:1946 assert self.cpu.supports_guard_gc_type
                assert!(
                    guard_gc_type_info.supports_guard_gc_type,
                    "x86/assembler.py:1946: assert self.cpu.\
                     supports_guard_gc_type (GcAllocator has not \
                     installed a TYPE_INFO / rclass.CLASSTYPE layout)"
                );

                // assembler.py:1971 vtable_ptr = loc_check_against_class
                //   .getint(): the bounds are resolved at codegen time,
                //   so arg1 must be an immediate class pointer.
                let class_arg = op.arg(1).to_opref();
                // history.py:227 — inline-Const carries its class pointer directly.
                let loc_check_against_class = class_arg.const_int_value().unwrap_or_else(|| {
                    panic!(
                        "x86/assembler.py:1971 vtable_ptr = \
                             loc_check_against_class.getint(): \
                             GUARD_SUBCLASS requires arg1 to be a \
                             ConstInt immediate class pointer"
                    )
                });
                // assembler.py:1973-1974: vtable_ptr.subclassrange_{min,max}
                let (check_min, check_max) = guard_gc_type_info
                    .subclass_ranges
                    .get(&loc_check_against_class)
                    .copied()
                    .unwrap_or_else(|| {
                        panic!(
                            "x86/assembler.py:1973-1974 vtable_ptr.\
                             subclassrange_min/max: GcAllocator has no \
                             rclass.CLASSTYPE entry for classptr {:#x}",
                            loc_check_against_class
                        )
                    });

                // assembler.py:1950-1951 offset / offset2.
                let offset2 = guard_gc_type_info.subclassrange_min_offset;
                if let Some(vtable_off) = vtable_offset {
                    // assembler.py:1953-1956
                    //     MOV_rm(loc_tmp, (loc_object, offset))
                    //     MOV_rm(loc_tmp, (loc_tmp, offset2))
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(vtable_off as u64));
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(offset2 as u64));
                } else {
                    // assembler.py:1957-1969 gcremovetypeptr path.
                    //     MOV32 loc_tmp, mem(loc_object, 0)
                    //     base_type_info, shift_by, sizeof_ti = ...
                    //     MOV loc_tmp, [base_type_info
                    //         + (loc_tmp << shift_by)
                    //         + sizeof_ti + offset2]
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.i64_const(GcHeader::SIZE as i64);
                    sink.i64_sub();
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(0));
                    sink.i64_const(TYPE_ID_MASK as i64);
                    sink.i64_and();
                    if guard_gc_type_info.shift_by > 0 {
                        sink.i64_const(guard_gc_type_info.shift_by as i64);
                        sink.i64_shl();
                    }
                    sink.i64_const(guard_gc_type_info.base_type_info as i64);
                    sink.i64_add();
                    sink.i64_const((guard_gc_type_info.sizeof_ti + offset2) as i64);
                    sink.i64_add();
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(0));
                }
                // Stack: [..., loc_tmp (i64)]

                // assembler.py:1976-1978 unsigned comparison:
                //     (loc_tmp - check_min) <u (check_max - check_min)
                sink.i64_const(check_min);
                sink.i64_sub();
                sink.i64_const(check_max - check_min);
                // assembler.py:1979 guard_success_cc = Conditions['B']:
                // guard passes when sub <u limit; fail when sub >=u limit.
                sink.i64_ge_u();
                emit_guard_if_exit(&mut sink, constants, guard_idx, op, block_exit_depth);
                guard_idx += 1;
            }
            OpCode::GuardFutureCondition | OpCode::GuardAlwaysFails => {
                // GuardAlwaysFails always exits.
                emit_guard_exit(&mut sink, constants, guard_idx, op);
                if let Some(d) = block_exit_depth {
                    sink.br(d);
                }
                guard_idx += 1;
            }

            // ── Quasi-immutable / record / assert ──
            OpCode::QuasiimmutField
            | OpCode::RecordExactClass
            | OpCode::RecordExactValueI
            | OpCode::RecordExactValueR
            | OpCode::AssertNotNone => {
                // Metadata-only ops, no codegen needed.
            }

            // ── Allocation via trampoline ──
            OpCode::Newstr | OpCode::Newunicode => {
                // These may appear in traces that materialize strings.
                // Use CALL trampoline if available, otherwise skip.
                if let Some(jit_call) = jit_call_idx {
                    let vi = op.pos.get().raw();
                    sink.local_get(0);
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref()); // length
                    sink.i64_store(mem64(CALL_ARGS_OFS));
                    sink.local_get(0);
                    sink.i64_const(0); // func_ptr = 0 signals "newstr" to host
                    sink.i64_store(mem64(CALL_FUNC_OFS));
                    sink.local_get(0);
                    sink.i64_const(1);
                    sink.i64_store(mem64(CALL_NARGS_OFS));
                    sink.local_get(0);
                    sink.call(jit_call);
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_get(0);
                        sink.i64_load(mem64(CALL_RESULT_OFS));
                        sink.local_set(1 + vi);
                    }
                }
            }

            // ── String content copy ──
            OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                // Bulk memory copy — use CALL trampoline or skip
            }

            // ── Misc ops ──
            OpCode::NurseryPtrIncrement => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                    sink.i64_add();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::CheckMemoryError => {
                // After allocation: check if result is null
                // No-op in wasm (allocations don't fail the same way)
            }
            OpCode::ZeroArray => {
                // Zero-initialize array region — skip for MVP
            }
            OpCode::LoadFromGcTable => {
                // `assembler.py:1545` `genop_load_from_gc_table`: this op is
                // produced only by the GC rewrite's `remove_constptr`
                // (`rewrite.py:1100`), whose arg is a `ConstInt(index)` into
                // a per-loop `GcTable` whose base is baked absolute. The
                // wasm backend does not run the GC rewrite and has no
                // host-address gc_table model (linear memory), so this op
                // never reaches here. Panic loudly rather than emit the old
                // SAME_AS pass-through, which after the rewrite flip would
                // load the raw index in place of the reference constant.
                panic!(
                    "wasm backend: LoadFromGcTable is unsupported (no gc_table model); \
                     the GC rewrite must not run for wasm"
                );
            }

            // ── CALL operations (via trampoline) ──
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallN
            | OpCode::CallF
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceN
            | OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerN
            | OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilN
            | OpCode::CondCallValueI
            | OpCode::CondCallValueR
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantN
            | OpCode::CallLoopinvariantF
            | OpCode::CallPureF
            | OpCode::CallMayForceF
            | OpCode::CallAssemblerF
            | OpCode::CallReleaseGilF => {
                let vi = op.pos.get().raw();
                let jit_call = jit_call_idx.expect("CALL op present but jit_call not imported");

                // args[0] = func_ptr, args[1..] = call arguments
                let func_ptr_ref = op.arg(0).to_opref();
                let call_args = &op.getarglist()[1..];

                // Store func_ptr to call area
                sink.local_get(0);
                emit_resolve(&mut sink, constants, func_ptr_ref);
                sink.i64_store(mem64(CALL_FUNC_OFS));

                // Store num_args
                sink.local_get(0);
                sink.i64_const(call_args.len() as i64);
                sink.i64_store(mem64(CALL_NARGS_OFS));

                // Store each arg
                for (i, arg) in call_args.iter().enumerate() {
                    sink.local_get(0);
                    emit_resolve(&mut sink, constants, arg.to_opref());
                    sink.i64_store(mem64(CALL_ARGS_OFS + i as u64 * SLOT_SIZE));
                }

                // Call trampoline
                sink.local_get(0);
                sink.call(jit_call);

                // Read result (for non-void calls)
                let is_void = matches!(
                    op.opcode,
                    OpCode::CallN
                        | OpCode::CallPureN
                        | OpCode::CallMayForceN
                        | OpCode::CallAssemblerN
                        | OpCode::CallReleaseGilN
                        | OpCode::CallLoopinvariantN
                );
                if !OpRef::raw_is_constant(vi) && !is_void {
                    sink.local_get(0);
                    sink.i64_load(mem64(CALL_RESULT_OFS));
                    sink.local_set(1 + vi);
                }
            }

            // ── Allocation (via trampoline — treated as CALL) ──
            // llmodel.py:775-790 bh_new* parity: a `New*` survives
            // optimization whenever the allocated object escapes the trace
            // (e.g. reboxed result stored into a namespace). The trace cannot
            // allocate inline (the GC is host-side), so route through the
            // `jit_call` trampoline to the `wasm_jit_alloc` helper, then write
            // the vtable / length fields with pointer-width (i32) stores.
            OpCode::New | OpCode::NewWithVtable => {
                let jit_call = jit_call_idx.expect("New op present but jit_call not imported");
                let vi = op.pos.get().raw();
                // llmodel.py:778-782: size, type_id, vtable from the size descr.
                let descr = op.getdescr();
                let sd = descr.as_ref().and_then(|d| d.as_size_descr());
                let (size, type_id, vtable) = sd.map_or((16i64, 0i64, 0usize), |sd| {
                    (sd.size() as i64, sd.type_id() as i64, sd.vtable())
                });

                // func_ptr = wasm_jit_alloc
                sink.local_get(0);
                sink.i64_const(alloc_fn_ptr);
                sink.i64_store(mem64(CALL_FUNC_OFS));
                // num_args = 2
                sink.local_get(0);
                sink.i64_const(2);
                sink.i64_store(mem64(CALL_NARGS_OFS));
                // arg0 = type_id
                sink.local_get(0);
                sink.i64_const(type_id);
                sink.i64_store(mem64(CALL_ARGS_OFS));
                // arg1 = size
                sink.local_get(0);
                sink.i64_const(size);
                sink.i64_store(mem64(CALL_ARGS_OFS + SLOT_SIZE));
                // call trampoline
                sink.local_get(0);
                sink.call(jit_call);

                if !OpRef::raw_is_constant(vi) {
                    // result pointer
                    sink.local_get(0);
                    sink.i64_load(mem64(CALL_RESULT_OFS));
                    sink.local_set(1 + vi);

                    // llmodel.py:779-781 write_int_at_mem(res, vtable_offset,
                    // WORD, vtable). The `ob_type` field is pointer-width: 4
                    // bytes on wasm32 (GuardClass reads it as i32), so store
                    // the low 32 bits to avoid clobbering the next field.
                    let write_vtable = op.opcode == OpCode::NewWithVtable
                        && vtable != 0
                        && vtable_offset.is_some();
                    if write_vtable {
                        let vt_off = vtable_offset.unwrap() as u64;
                        sink.local_get(1 + vi);
                        sink.i32_wrap_i64();
                        sink.i32_const(vtable as i32);
                        sink.i32_store(MemArg {
                            offset: vt_off,
                            align: 2,
                            memory_index: 0,
                        });
                    }
                }
            }
            OpCode::NewArray | OpCode::NewArrayClear => {
                let jit_call = jit_call_idx.expect("NewArray op present but jit_call not imported");
                let vi = op.pos.get().raw();
                let descr = op.getdescr();
                let ad = descr.as_ref().and_then(|d| d.as_array_descr());
                let (base_size, item_size) = ad.map_or((16i64, 8i64), |ad| {
                    (ad.base_size() as i64, ad.item_size() as i64)
                });
                let len_offset = ad
                    .and_then(|ad| ad.len_descr())
                    .map_or(0i64, |ld| ld.offset() as i64);
                let type_id = ad.map_or(0i64, |ad| ad.type_id() as i64);

                // func_ptr = wasm_jit_alloc_array
                sink.local_get(0);
                sink.i64_const(alloc_array_fn_ptr);
                sink.i64_store(mem64(CALL_FUNC_OFS));
                // num_args = 5
                sink.local_get(0);
                sink.i64_const(5);
                sink.i64_store(mem64(CALL_NARGS_OFS));
                // arg0 = type_id
                sink.local_get(0);
                sink.i64_const(type_id);
                sink.i64_store(mem64(CALL_ARGS_OFS));
                // arg1 = base_size
                sink.local_get(0);
                sink.i64_const(base_size);
                sink.i64_store(mem64(CALL_ARGS_OFS + SLOT_SIZE));
                // arg2 = item_size
                sink.local_get(0);
                sink.i64_const(item_size);
                sink.i64_store(mem64(CALL_ARGS_OFS + 2 * SLOT_SIZE));
                // arg3 = length (op.arg(0))
                sink.local_get(0);
                emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                sink.i64_store(mem64(CALL_ARGS_OFS + 3 * SLOT_SIZE));
                // arg4 = len_offset
                sink.local_get(0);
                sink.i64_const(len_offset);
                sink.i64_store(mem64(CALL_ARGS_OFS + 4 * SLOT_SIZE));
                // call trampoline
                sink.local_get(0);
                sink.call(jit_call);

                if !OpRef::raw_is_constant(vi) {
                    sink.local_get(0);
                    sink.i64_load(mem64(CALL_RESULT_OFS));
                    sink.local_set(1 + vi);
                }
            }

            // ── Misc ──
            OpCode::ForceToken => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    sink.i64_const(0); // sentinel force token
                    sink.local_set(1 + vi);
                }
            }

            // Float operations
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // Values stored as i64 (bitcast from f64)
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    emit_resolve(&mut sink, constants, op.arg(1).to_opref());
                    sink.f64_reinterpret_i64();
                    match op.opcode {
                        OpCode::FloatAdd => {
                            sink.f64_add();
                        }
                        OpCode::FloatSub => {
                            sink.f64_sub();
                        }
                        OpCode::FloatMul => {
                            sink.f64_mul();
                        }
                        OpCode::FloatTrueDiv => {
                            sink.f64_div();
                        }
                        _ => unreachable!(),
                    }
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::FloatNeg => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    sink.f64_neg();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }
            OpCode::FloatAbs => {
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    sink.f64_abs();
                    sink.i64_reinterpret_f64();
                    sink.local_set(1 + vi);
                }
            }

            // Debug / metadata / no-op
            OpCode::DebugMergePoint
            | OpCode::IncrementDebugCounter
            | OpCode::EnterPortalFrame
            | OpCode::LeavePortalFrame
            | OpCode::VirtualRefFinish
            | OpCode::ForceSpill
            | OpCode::Keepalive => {}

            _ => {
                // An opcode with no codegen arm. If it produces a value (a
                // result local that later ops read), silently skipping it
                // leaves a stale slot and yields wrong results, so decline the
                // whole trace and let the metainterp fall back to the
                // interpreter (correct, unaccelerated). Side-effect-free
                // metadata opcodes that produce no value are enumerated above.
                let vi = op.pos.get().raw();
                if !OpRef::raw_is_constant(vi) {
                    return Err(BackendError::Unsupported(format!(
                        "wasm codegen: unhandled value-producing opcode {:?}",
                        op.opcode
                    )));
                }
            }
        }
    }

    if has_loop {
        sink.end(); // end loop
        sink.end(); // end block
    }

    sink.local_get(0);
    sink.end(); // end function

    Ok(func)
}

// ── Helpers ──

fn find_label_args(ops: &[Op]) -> Vec<OpRef> {
    // The JUMP branches back to the loop-header label, which is the LAST
    // label in a peeled trace (an outer entry label may precede it). The
    // `loop` is emitted at that same label in `build_function`.
    for op in ops.iter().rev() {
        if op.opcode == OpCode::Label {
            return op.getarglist().iter().map(|a| a.to_opref()).collect();
        }
    }
    Vec::new()
}

fn emit_resolve(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    opref: OpRef,
) {
    if opref.is_constant() {
        // history.py:227/268/314 — inline-Const variants carry value inline.
        let val = opref
            .inline_const_bits()
            .unwrap_or_else(|| constants.get(&opref.raw()).copied().unwrap_or(0));
        sink.i64_const(val);
    } else {
        sink.local_get(1 + opref.raw());
    }
}

/// Extract field offset from op's descr (FieldDescr).
fn field_offset_from_descr(op: &Op) -> u64 {
    let __descr_arc_descr = op.getdescr();
    if let Some(ref descr) = __descr_arc_descr.as_ref() {
        if let Some(fd) = descr.as_field_descr() {
            return fd.offset() as u64;
        }
    }
    0
}

/// Extract array length offset from descr.
fn array_len_offset_from_descr(_op: &Op) -> u64 {
    // RPython arrays store length before the data.
    // On wasm32, the length is typically at a fixed offset.
    8 // default: length at offset 8 (after ob_type)
}

/// Compute array element address: base + base_size + index * item_size.
/// Leaves i32 address on the wasm stack.
fn emit_array_addr(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    op: &Op,
) {
    let (base_size, item_size) = op
        .with_array_descr(|ad| (ad.base_size() as u64, ad.item_size() as u64))
        .unwrap_or((16, 8));
    emit_resolve(sink, constants, op.arg(0).to_opref()); // array ptr
    sink.i32_wrap_i64();
    // base + base_size + index * item_size
    emit_resolve(sink, constants, op.arg(1).to_opref()); // index
    sink.i32_wrap_i64();
    sink.i32_const(item_size as i32);
    sink.i32_mul();
    sink.i32_add();
    sink.i32_const(base_size as i32);
    sink.i32_add();
}

// ── Guard emission helpers ──

fn emit_guard_true(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: Option<u32>,
) {
    emit_resolve(sink, constants, op.arg(0).to_opref());
    sink.i64_eqz();
    emit_guard_if_exit(sink, constants, guard_idx, op, block_exit_depth);
}

fn emit_guard_false(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: Option<u32>,
) {
    emit_resolve(sink, constants, op.arg(0).to_opref());
    sink.i64_const(0);
    sink.i64_ne();
    emit_guard_if_exit(sink, constants, guard_idx, op, block_exit_depth);
}

/// Common guard exit: condition is on stack (i32), emit if + exit.
///
/// `block_exit_depth` is the statement-level depth of the enclosing exit
/// `block` (preamble = 0, loop body = 1); the `+ 1` accounts for the `if`
/// this opens. `None` for straight-line traces with no exit block.
fn emit_guard_if_exit(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: Option<u32>,
) {
    sink.if_(BlockType::Empty);
    emit_guard_exit(sink, constants, guard_idx, op);
    match block_exit_depth {
        // Loop traces: `br` out of this `if` and the enclosing exit `block`
        // (the `+ 1` accounts for the `if`) to the function epilogue.
        Some(d) => {
            sink.br(d + 1);
        }
        // Straight-line traces have no enclosing block, so fall-through would
        // reach the terminal Finish and overwrite frame[0] with its
        // fail_index, discarding this guard's exit. Return the frame pointer
        // directly (the epilogue's value) to hand control to the metainterp.
        None => {
            sink.local_get(0);
            sink.return_();
        }
    }
    sink.end();
}

fn emit_guard_exit(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    guard_idx: u32,
    op: &Op,
) {
    let fail_args: Vec<OpRef> = op
        .getfailargs()
        .map(|fa| fa.iter().map(|a| a.to_opref()).collect())
        .unwrap_or_else(|| op.getarglist().iter().map(|a| a.to_opref()).collect());

    for (i, &arg_ref) in fail_args.iter().enumerate() {
        let offset = FRAME_SLOT_BASE + i as u64 * SLOT_SIZE;
        sink.local_get(0);
        emit_resolve(sink, constants, arg_ref);
        sink.i64_store(mem64(offset));
    }

    sink.local_get(0);
    sink.i64_const(guard_idx as i64);
    sink.i64_store(mem64(0));
}

// ── Binary ops ──

enum BinOp {
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64RemS,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
}

fn apply_binop(sink: &mut InstructionSink<'_>, op: BinOp) {
    match op {
        BinOp::I64Add => {
            sink.i64_add();
        }
        BinOp::I64Sub => {
            sink.i64_sub();
        }
        BinOp::I64Mul => {
            sink.i64_mul();
        }
        BinOp::I64DivS => {
            sink.i64_div_s();
        }
        BinOp::I64RemS => {
            sink.i64_rem_s();
        }
        BinOp::I64And => {
            sink.i64_and();
        }
        BinOp::I64Or => {
            sink.i64_or();
        }
        BinOp::I64Xor => {
            sink.i64_xor();
        }
        BinOp::I64Shl => {
            sink.i64_shl();
        }
        BinOp::I64ShrS => {
            sink.i64_shr_s();
        }
        BinOp::I64ShrU => {
            sink.i64_shr_u();
        }
    }
}

fn emit_binop(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    op: &Op,
    binop: BinOp,
) {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    emit_resolve(sink, constants, op.arg(0).to_opref());
    emit_resolve(sink, constants, op.arg(1).to_opref());
    apply_binop(sink, binop);
    sink.local_set(1 + vi);
}

/// `UintMulHigh`: high 64 bits of the unsigned 64×64→128 product. Wasm has
/// only `i64.mul` (low 64 bits), so compute via the classic 32-bit split:
/// a = ah·2³²+al, b = bh·2³²+bl, with carry-safe intermediates
///   mid1 = ah·bl + (al·bl >> 32)
///   high = ah·bh + (mid1 >> 32) + ((al·bh + (mid1 & 0xFFFFFFFF)) >> 32)
/// Uses the five scratch locals reserved at `num_vars+1 ..= num_vars+5`.
fn emit_umulhi(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    op: &Op,
    num_vars: u32,
) {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    const MASK32: i64 = 0xFFFF_FFFF;
    let al = num_vars + 1;
    let ah = num_vars + 2;
    let bl = num_vars + 3;
    let bh = num_vars + 4;
    let mid1 = num_vars + 5;

    // al = a & 0xFFFFFFFF
    emit_resolve(sink, constants, op.arg(0).to_opref());
    sink.i64_const(MASK32);
    sink.i64_and();
    sink.local_set(al);
    // ah = a >>u 32
    emit_resolve(sink, constants, op.arg(0).to_opref());
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.local_set(ah);
    // bl = b & 0xFFFFFFFF
    emit_resolve(sink, constants, op.arg(1).to_opref());
    sink.i64_const(MASK32);
    sink.i64_and();
    sink.local_set(bl);
    // bh = b >>u 32
    emit_resolve(sink, constants, op.arg(1).to_opref());
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.local_set(bh);

    // mid1 = ah*bl + ((al*bl) >>u 32)
    sink.local_get(al);
    sink.local_get(bl);
    sink.i64_mul();
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.local_get(ah);
    sink.local_get(bl);
    sink.i64_mul();
    sink.i64_add();
    sink.local_set(mid1);

    // high = ah*bh + (mid1 >>u 32) + ((al*bh + (mid1 & MASK32)) >>u 32)
    sink.local_get(ah);
    sink.local_get(bh);
    sink.i64_mul();
    sink.local_get(mid1);
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.i64_add();
    sink.local_get(al);
    sink.local_get(bh);
    sink.i64_mul();
    sink.local_get(mid1);
    sink.i64_const(MASK32);
    sink.i64_and();
    sink.i64_add();
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.i64_add();

    sink.local_set(1 + vi);
}

/// Overflow binary op: stores result in pos, overflow flag convention.
/// The overflow flag is not stored separately — GuardNoOverflow/GuardOverflow
/// is handled by checking after the fact (simplified for wasm MVP).
fn emit_ovf_binop(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    op: &Op,
    binop: BinOp,
) {
    // For wasm MVP, just compute the result without overflow detection.
    // GuardNoOverflow/GuardOverflow are treated as always-pass.
    emit_binop(sink, constants, op, binop);
}

// ── Comparison ops ──

enum CmpOp {
    I64LtS,
    I64LeS,
    I64Eq,
    I64Ne,
    I64GtS,
    I64GeS,
    I64LtU,
    I64LeU,
    I64GtU,
    I64GeU,
}

fn apply_cmp(sink: &mut InstructionSink<'_>, op: CmpOp) {
    match op {
        CmpOp::I64LtS => {
            sink.i64_lt_s();
        }
        CmpOp::I64LeS => {
            sink.i64_le_s();
        }
        CmpOp::I64Eq => {
            sink.i64_eq();
        }
        CmpOp::I64Ne => {
            sink.i64_ne();
        }
        CmpOp::I64GtS => {
            sink.i64_gt_s();
        }
        CmpOp::I64GeS => {
            sink.i64_ge_s();
        }
        CmpOp::I64LtU => {
            sink.i64_lt_u();
        }
        CmpOp::I64LeU => {
            sink.i64_le_u();
        }
        CmpOp::I64GtU => {
            sink.i64_gt_u();
        }
        CmpOp::I64GeU => {
            sink.i64_ge_u();
        }
    }
}

// ── Float comparison helper ──

enum FloatCmp {
    Lt,
    Le,
    Eq,
    Ne,
    Gt,
    Ge,
}

fn emit_float_cmp(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    op: &Op,
    cmp: FloatCmp,
) {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    emit_resolve(sink, constants, op.arg(0).to_opref());
    sink.f64_reinterpret_i64();
    emit_resolve(sink, constants, op.arg(1).to_opref());
    sink.f64_reinterpret_i64();
    match cmp {
        FloatCmp::Lt => {
            sink.f64_lt();
        }
        FloatCmp::Le => {
            sink.f64_le();
        }
        FloatCmp::Eq => {
            sink.f64_eq();
        }
        FloatCmp::Ne => {
            sink.f64_ne();
        }
        FloatCmp::Gt => {
            sink.f64_gt();
        }
        FloatCmp::Ge => {
            sink.f64_ge();
        }
    }
    sink.i64_extend_i32_u();
    sink.local_set(1 + vi);
}

fn emit_cmp(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    op: &Op,
    cmpop: CmpOp,
) {
    let vi = op.pos.get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    emit_resolve(sink, constants, op.arg(0).to_opref());
    emit_resolve(sink, constants, op.arg(1).to_opref());
    apply_cmp(sink, cmpop);
    sink.i64_extend_i32_u();
    sink.local_set(1 + vi);
}

// ── Unary op helper ──

fn emit_unary_vi(
    sink: &mut InstructionSink<'_>,
    constants: &majit_ir::VecAssoc<u32, i64>,
    op: &Op,
    prefix: impl FnOnce(&mut InstructionSink<'_>),
    suffix: impl FnOnce(&mut InstructionSink<'_>),
) {
    let vi = op.pos.get().raw();
    if !OpRef::raw_is_constant(vi) {
        prefix(sink);
        emit_resolve(sink, constants, op.arg(0).to_opref());
        suffix(sink);
        sink.local_set(1 + vi);
    }
}
