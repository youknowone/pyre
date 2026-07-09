/// aarch64/assembler.py: AssemblerARM64 — aarch64 JIT code generation backend.
///
/// Generates machine code from IR operations via dynasm-rs.
/// RPython: AssemblerARM64(ResOpAssembler) in aarch64/assembler.py.
///
/// Key methods:
///   assemble_loop — assembler.py:501
///   assemble_bridge — assembler.py:623
///   _assemble — assembler.py:779 (walk ops + emit code)
///   patch_jump_for_descr — assembler.py:965
///   redirect_call_assembler — assembler.py:1138
use indexmap::IndexMap;
use majit_ir::IndexMapExt;
use std::sync::Arc;

// aarch64/assembler.py parity: aarch64-only backend.
use dynasmrt::aarch64::Assembler;
use dynasmrt::{AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi, ExecutableBuffer, dynasm};

use majit_backend::BackendError;
use majit_ir::{FailDescr, InputArg, Op, OpCode, OpRef, OpTypeIndex, TargetArgLoc, Type};

use crate::arch::*;
use crate::codebuf;
use crate::gcmap::{allocate_gcmap, gcmap_set_bit};
use crate::jitframe::{
    FIRST_ITEM_OFFSET, JF_DESCR_OFS, JF_FORCE_DESCR_OFS, JF_FRAME_OFS, JF_GCMAP_OFS,
    JF_GUARD_EXC_OFS,
};
use crate::regalloc::{RegAlloc, RegAllocOp};
use crate::regloc::{Loc, RegLoc};
use crate::runner::GuardGcTypeInfo;

const AARCH64_GEN_REGS: [crate::regloc::RegLoc; 18] = crate::aarch64::registers::ALL_REGS;

const AARCH64_FLOAT_REGS: [crate::regloc::RegLoc; 8] = crate::aarch64::registers::ALL_VFP_REGS;

/// Resolved argument: either a frame slot (frame-pointer-relative offset) or a constant.
enum ResolvedArg {
    /// Frame-pointer-relative byte offset: [rbp + offset] on x64, [x29, #offset] on aarch64.
    Slot(i32),
    /// Immediate constant value.
    Const(i64),
}

/// Pointer-identity key for `target_tokens_currently_compiling`. PyPy
/// x86/assembler.py:93 keys it by the descr Python object itself; we use
/// the underlying allocation address of the `Arc<dyn Descr>` so two
/// distinct TargetToken descriptors are never confused.
fn loop_target_id(op: &Op) -> Option<usize> {
    op.getdescr().as_ref().map(majit_ir::descr_identity)
}

fn target_argloc_from_loc(loc: Loc) -> TargetArgLoc {
    match loc {
        Loc::Reg(r) => TargetArgLoc::Reg {
            regnum: r.value,
            is_xmm: r.is_xmm,
        },
        Loc::Ebp(e) => TargetArgLoc::Ebp {
            ebp_offset: e.value,
            is_float: e.is_float,
        },
        Loc::Frame(f) => TargetArgLoc::Frame {
            position: f.position,
            ebp_offset: f.ebp_loc.value,
            is_float: f.ebp_loc.is_float,
        },
        Loc::Immed(i) => TargetArgLoc::Immed {
            value: i.value,
            is_float: i.is_float,
        },
        Loc::Addr(a) => TargetArgLoc::Addr {
            base: a.base,
            index: a.index,
            scale: a.scale,
            offset: a.offset,
        },
    }
}

fn loc_from_target_argloc(loc: &TargetArgLoc) -> Loc {
    match *loc {
        TargetArgLoc::Reg { regnum, is_xmm } => {
            Loc::Reg(crate::regloc::RegLoc::new(regnum, is_xmm))
        }
        TargetArgLoc::Ebp {
            ebp_offset,
            is_float,
        } => Loc::Ebp(crate::regloc::RawEbpLoc {
            value: ebp_offset,
            is_float,
        }),
        TargetArgLoc::Frame {
            position,
            ebp_offset,
            is_float,
        } => Loc::Frame(crate::regloc::FrameLoc::new(position, ebp_offset, is_float)),
        TargetArgLoc::Immed { value, is_float } => {
            Loc::Immed(crate::regloc::ImmedLoc { value, is_float })
        }
        TargetArgLoc::Addr {
            base,
            index,
            scale,
            offset,
        } => Loc::Addr(crate::regloc::AddressLoc {
            base,
            index,
            scale,
            offset,
        }),
    }
}

fn all_gen_regs() -> &'static [crate::regloc::RegLoc] {
    &AARCH64_GEN_REGS
}

fn all_float_regs() -> &'static [crate::regloc::RegLoc] {
    &AARCH64_FLOAT_REGS
}

fn core_reg_position(reg: crate::regloc::RegLoc) -> Option<usize> {
    all_gen_regs()
        .iter()
        .position(|candidate| *candidate == reg)
}

fn float_reg_position(reg: crate::regloc::RegLoc) -> Option<usize> {
    all_float_regs()
        .iter()
        .position(|candidate| *candidate == reg)
        .map(|idx| all_gen_regs().len() + idx)
}

fn reg_position_in_jitframe(reg: crate::regloc::RegLoc) -> Option<usize> {
    if reg.is_xmm {
        float_reg_position(reg)
    } else {
        core_reg_position(reg)
    }
}

// ── Abstract condition codes ──
// Architecture-independent CC values used throughout the assembler.
// Converted to arch-specific encoding at emission time.
const CC_O: u8 = 0;
const CC_NO: u8 = 1;
const CC_B: u8 = 2; // unsigned <
const CC_AE: u8 = 3; // unsigned >=
const CC_E: u8 = 4; // ==
const CC_NE: u8 = 5; // !=
const CC_BE: u8 = 6; // unsigned <=
const CC_A: u8 = 7; // unsigned >
const CC_S: u8 = 8;
const CC_NS: u8 = 9;
const CC_L: u8 = 10; // signed <
const CC_GE: u8 = 11; // signed >=
const CC_LE: u8 = 12; // signed <=
const CC_G: u8 = 13; // signed >

/// Invert a condition code.
fn invert_cc(cc: u8) -> u8 {
    match cc {
        CC_O => CC_NO,
        CC_NO => CC_O,
        CC_B => CC_AE,
        CC_AE => CC_B,
        CC_E => CC_NE,
        CC_NE => CC_E,
        CC_BE => CC_A,
        CC_A => CC_BE,
        CC_S => CC_NS,
        CC_NS => CC_S,
        CC_L => CC_GE,
        CC_GE => CC_L,
        CC_LE => CC_G,
        CC_G => CC_LE,
        _ => CC_E, // fallback
    }
}

/// assembler.py:47 AssemblerARM64.
/// In Rust, this is a transient builder — created per compilation,
/// not a long-lived object like RPython's.
///
/// Borrows the trace's `inputargs` and `operations` for its lifetime so
/// `OpRef → Type` resolves through `op.type_` / `inputarg.tp` directly
/// (RPython `box.type` parity); no `value_types: HashMap` side-table.
pub struct AssemblerARM64<'a> {
    /// The dynasm assembler (rx86.py + codebuf.py combined).
    pub(crate) mc: Assembler,
    /// assembler.py:83 pending_guard_tokens — guards awaiting recovery stubs.
    pending_guard_tokens: Vec<GuardToken>,
    /// GC bitmap to push before the current collecting call (e.g.,
    /// CallMallocNursery slow path). Set by the `RegAllocOp::Perform`
    /// emit path when `gcmap: Some(..)` is carried, cleared after.
    pending_malloc_nursery_gcmap: Option<usize>,
    /// Frame depth (in WORD units) for the current trace.
    frame_depth: usize,
    /// assembler.py:94 setup() `self.frame_depth_to_patch = []` — buffer
    /// offsets of the `gen_load_int` depth placeholders emitted by
    /// `emit_check_frame_depth`, rewritten by `patch_stack_checks` once the
    /// final `frame_depth` is known.
    frame_depth_to_patch: Vec<usize>,
    /// assembler.py:1167-1171 `_assemble`: the frame depth of a cross-loop
    /// JUMP target (its `target_frame_depth`), or 0 when the trace has no
    /// external JUMP.  The closing `br` enters the target loop's body, which
    /// may use deeper frame slots than this trace; `_assemble` grows
    /// `frame_depth` to fit so a bridge's prologue `_check_frame_depth`
    /// reallocs the in-flight JITFRAME large enough before the `br`.
    jump_target_frame_depth: usize,
    /// Fail descriptors built during assembly.
    fail_descrs: Vec<std::sync::Arc<majit_ir::FailDescrCell>>,
    /// trace_id for this compilation.
    trace_id: u64,
    /// header_pc (green_key) for this compilation.
    header_pc: u64,
    /// Input argument types.
    input_types: Vec<Type>,
    /// assembler.py:641 rebuild_faillocs_from_descr parity:
    /// bridge input locations recovered from the source guard descr.
    bridge_input_locs: Option<Vec<Loc>>,

    // ── State tracking for code generation ──
    /// Maps OpRef → jitframe slot index. `IndexMap` (O(1) get/insert), not the
    /// Vec-backed `IndexMap`: `resolve_opref` reads this per emitted op during
    /// codegen and the sync loop inserts one entry per live var, so a Vec-`get`
    /// is O(n) and makes `_assemble` O(n^2) on large traces — `get_index_of`
    /// inlines into `_assemble` and dominated aheui's logo compile. The box→
    /// location map is a dict in the reference assembler; insertion order is
    /// preserved (no semantic change, only the lookup cost).
    opref_to_slot: indexmap::IndexMap<OpRef, usize>,
    /// Trace inputargs — borrowed for `opref_type` lookups.
    inputargs: &'a [InputArg],
    /// Trace operations — borrowed for `opref_type` lookups (reads
    /// `op.type_` directly, RPython `box.type` parity).
    operations: &'a [Op],
    /// `inputarg_pos[arg.index] = idx in inputargs`, sentinel
    /// [`OpTypeIndex::NO_POS`] for unset slots. Mirrors
    /// `OpTypeIndex::inputarg_pos`.
    inputarg_pos: Vec<u32>,
    /// `op_pos[op.pos.raw()] = idx in operations`, sentinel
    /// [`OpTypeIndex::NO_POS`] for unset slots and Void/None ops.
    /// Mirrors `OpTypeIndex::op_pos`.
    op_pos: Vec<u32>,
    /// Constants: OpRef index (>= 10000) → typed `Const` value. The box
    /// variant carries its own type (`Const::get_type`), so no separate
    /// constant-type map is needed.
    constants: majit_ir::ConstMap<majit_ir::Const>,
    /// Next available frame slot index.
    next_slot: usize,
    /// Condition code from the most recent CMP/TEST instruction,
    /// consumed by a following GUARD_TRUE/GUARD_FALSE.
    /// Stores an abstract condition code (CC_* constants).
    guard_success_cc: Option<u8>,
    /// x86/assembler.py:93 target_tokens_currently_compiling parity.
    /// Keyed by descriptor pointer identity (PyPy uses Python `is`).
    target_tokens_currently_compiling: IndexMap<usize, DynamicLabel>,
    compiled_target_tokens: Vec<majit_ir::DescrRef>,
    /// llmodel.py:64-69 self.vtable_offset — typeptr field byte offset.
    /// `None` corresponds to RPython's gcremovetypeptr config.
    vtable_offset: Option<usize>,
    /// llsupport/gc.py:563 vtable→typeid table, materialized by the runner
    /// via gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr. Used by
    /// the gcremovetypeptr branch of `_cmp_guard_class`.
    classptr_to_typeid: IndexMap<i64, u32>,
    /// TYPE_INFO / CLASSTYPE constants for `GUARD_IS_OBJECT` and
    /// `GUARD_SUBCLASS`, fetched by the runner from the active gc_ll_descr.
    guard_gc_type_info: Option<GuardGcTypeInfo>,
    /// Constant classptr → `(subclassrange_min, subclassrange_max)`, matching
    /// `loc_check_against_class.getint()` field reads in
    /// `aarch64/opassembler.py:695-698`.
    classptr_to_subclass_range: IndexMap<i64, (i64, i64)>,
    /// Dynamic label at the function entry for self-recursive CALL_ASSEMBLER.
    self_entry_label: Option<DynamicLabel>,
    /// Leaked pointer holding the resolved entry address for self-recursive
    /// CALL_ASSEMBLER via the execute trampoline. Written after finalization.
    self_entry_addr_ptr: *mut usize,
    /// assembler.py:320 descr._ll_function_addr parity:
    /// Maps call_target_token → compiled code address for CALL_ASSEMBLER.
    /// Populated by the runner before compilation, from registered loop targets.
    call_assembler_targets: IndexMap<u64, usize>,
    /// opassembler.py:1177 _finish_gcmap.
    finish_gcmap: Option<*mut usize>,
    /// opassembler.py:1215 gcmap_for_finish.
    gcmap_for_finish: *mut usize,
    /// assembler.py:2207 _store_force_index parity:
    /// Pre-allocated fail descr for the next GUARD_NOT_FORCED, created
    /// at CALL_ASSEMBLER emission time so we can store its pointer to
    /// jf_force_descr before the call. Consumed by the subsequent
    /// GUARD_NOT_FORCED guard emission.
    pending_force_descr: Option<majit_ir::DescrRef>,
    /// Pre-wrapped `FailDescrCell` for the same pending guard.  Codegen
    /// bakes the cell's thin pointer into `JF_FORCE_DESCR_OFS` so that
    /// `force_token_to_dead_frame` (cranelift/compiler.rs:2660) can
    /// recover the descr via `recover_fail_descr_cell` without the
    /// fat-pointer mismatch a bare `Arc<dyn Descr>` ptr would cause.
    /// The same cell is consumed by `append_guard_token_with_faillocs`
    /// so jf_force_descr and jf_descr resolve to the same identity.
    pending_force_cell: Option<std::sync::Arc<majit_ir::FailDescrCell>>,
    /// `compile.py:665-674` + `pyjitpl.py:2283`: construction-time
    /// snapshot of the six descr pointers attached to the owning cpu
    /// instance.  Retained for constructor signature stability across
    /// runner / call_assembler callsites — emission helpers
    /// (`done_with_this_frame_descr_ptr_for_type`,
    /// `exit_frame_with_exception_descr_ref_ptr`,
    /// `propagate_exception_descr_ptr`) read live from `cpu_handle`
    /// instead so the raw ptr baked into `JF_DESCR_OFS` and the Arc
    /// stamped into `meta_descr` come from the same snapshot.
    #[allow(dead_code)]
    attached_descrs: crate::guard::AttachedDescrPtrs,
    /// `Arc` clone of the owning cpu's attachment handle.  Its heap
    /// address is baked into the CALL_ASSEMBLER helper call site as a
    /// compile-time immediate (`Arc::as_ptr`) and the `Arc` is moved
    /// into the resulting `CompiledCode` so the pointee outlives any
    /// subsequent `DynasmBackend` drop — same role as RPython's
    /// `self.cpu` attribute-access after whole-program translation,
    /// where the `cpu` object's identity is guaranteed by Python.
    cpu_handle: crate::guard::CpuDescrHandle,
    /// `assembler.py:1545` `genop_load_from_gc_table`: base address of
    /// this loop's per-loop `GcTable` slot array. Baked as a 64-bit
    /// immediate by the `LoadFromGcTable` genop; 0 when the trace
    /// references no reference constants. Set before `assemble_loop` /
    /// `assemble_bridge` via [`set_gc_table_base`](Self::set_gc_table_base).
    gc_table_base: usize,
}

/// assembler.py GuardToken — represents a pending guard needing
/// a recovery stub to be written after the main loop body.
struct GuardToken {
    /// Dynamic label that the guard's Jcc jumps to — bound in
    /// write_pending_failure_recoveries to the recovery stub.
    fail_label: DynamicLabel,
    /// The fail descriptor cell for this guard.  `Arc::as_ptr(&fail_descr)`
    /// is the thin pointer baked into `jf_descr`; the same cell instance is
    /// stored on `Asm::fail_descrs` so registration on the owning CLT keeps
    /// it alive while the recovery stub references its address.
    fail_descr: std::sync::Arc<majit_ir::FailDescrCell>,
    /// Constants to store in frame during recovery.
    /// Each entry: (frame_slot_index, constant_value).
    const_stores: Vec<(usize, i64)>,
    /// opassembler.py:515 GuardToken.gcmap.
    gcmap: *mut usize,
    /// llsupport/assembler.py:40-44 must_save_exception: true for
    /// GUARD_EXCEPTION / GUARD_NO_EXCEPTION / GUARD_NOT_FORCED.  Selects the
    /// exc=True failure-recovery variant that stages pos_exc_value into
    /// jf_guard_exc (store_info_on_descr:236) so grab_exc_value can read it.
    must_save_exception: bool,
}

/// Compiled output from assemble_loop/assemble_bridge.
pub struct CompiledCode {
    /// Executable memory buffer (keeps code alive).
    pub buffer: ExecutableBuffer,
    /// Entry point offset within the buffer.
    pub entry_offset: AssemblyOffset,
    /// Fail descriptors for guards + FINISH ops.
    /// Frozen after compile — `Box<[T]>` reflects RPython's no-mutation
    /// contract (compile.py:183-203 record_loop_or_bridge). Position
    /// equals `descr.fail_index` by an invariant asserted at conversion
    /// from the in-progress `AssemblerARM64.fail_descrs` Vec.
    pub fail_descrs: Box<[std::sync::Arc<majit_ir::FailDescrCell>]>,
    /// Input argument types.
    pub input_types: Vec<Type>,
    /// `compile.py:665-674` parity: `Arc` clone of the owning cpu's
    /// attachment handle.  Keeps the heap pointee alive for the whole
    /// lifetime of this compiled trace so the `cpu_handle` immediate
    /// baked into the CALL_ASSEMBLER helper call site never dangles,
    /// even if the emitting `DynasmBackend` is dropped first.
    pub cpu_attachments: crate::guard::CpuDescrHandle,
    /// trace_id.
    pub trace_id: u64,
    /// header_pc (green_key).
    pub header_pc: u64,
    /// Frame depth (number of jitframe slots used).
    /// AtomicUsize for redirect_call_assembler's update_frame_info
    /// parity: may be updated through &CompiledCode (shared ref).
    pub frame_depth: std::sync::atomic::AtomicUsize,
    /// `None` for root loops; bridges set `(source_trace_id, source_fail_index_per_trace)`.
    pub source_guard: Option<(u64, u32)>,
}

#[allow(dead_code)]
impl<'a> AssemblerARM64<'a> {
    /// rpython/jit/metainterp/history.py:220 `box.type` parity.
    /// Single source of truth: `op.type_` for ops, `inputarg.tp` for
    /// inputargs, the `Const` variant tag for constants.
    #[inline]
    fn opref_type(&self, opref: OpRef) -> Option<Type> {
        self.opref_type_at(opref, None)
    }

    #[inline]
    fn opref_type_at(&self, opref: OpRef, at_op_index: Option<usize>) -> Option<Type> {
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        match at_op_index {
            Some(at) => type_index.opref_type_at(opref, at),
            None => type_index.opref_type(opref),
        }
    }

    /// assembler.py:54 __init__
    pub(crate) fn new(
        trace_id: u64,
        header_pc: u64,
        constants: majit_ir::ConstMap<majit_ir::Const>,
        vtable_offset: Option<usize>,
        classptr_to_typeid: IndexMap<i64, u32>,
        guard_gc_type_info: Option<GuardGcTypeInfo>,
        classptr_to_subclass_range: IndexMap<i64, (i64, i64)>,
        attached_descrs: crate::guard::AttachedDescrPtrs,
        cpu_handle: crate::guard::CpuDescrHandle,
        inputargs: &'a [InputArg],
        operations: &'a [Op],
    ) -> Self {
        let inputarg_pos = OpTypeIndex::<Op>::build_inputarg_pos(inputargs);
        let op_pos = OpTypeIndex::build_op_pos(operations);
        AssemblerARM64 {
            mc: Assembler::new().unwrap(),
            pending_guard_tokens: Vec::new(),
            pending_malloc_nursery_gcmap: None,
            frame_depth: JITFRAME_FIXED_SIZE,
            frame_depth_to_patch: Vec::new(),
            jump_target_frame_depth: 0,
            fail_descrs: Vec::new(),
            trace_id,
            header_pc,
            input_types: Vec::new(),
            bridge_input_locs: None,
            opref_to_slot: indexmap::IndexMap::new(),
            inputargs,
            operations,
            inputarg_pos,
            op_pos,
            constants,
            next_slot: 0,
            guard_success_cc: None,
            target_tokens_currently_compiling: IndexMap::new(),
            compiled_target_tokens: Vec::new(),
            vtable_offset,
            classptr_to_typeid,
            guard_gc_type_info,
            classptr_to_subclass_range,
            self_entry_label: None,
            self_entry_addr_ptr: Box::into_raw(Box::new(0usize)),
            call_assembler_targets: IndexMap::new(),
            finish_gcmap: None,
            gcmap_for_finish: {
                let gcmap = allocate_gcmap(1, JITFRAME_FIXED_SIZE);
                gcmap_set_bit(gcmap, 0);
                gcmap
            },
            pending_force_descr: None,
            pending_force_cell: None,
            attached_descrs,
            cpu_handle,
            gc_table_base: 0,
        }
    }

    /// `assembler.py:793-824` parity: hand the per-loop `GcTable`'s base
    /// address to the assembler before emission, so `LoadFromGcTable`
    /// genops bake it as the slot-array base immediate. 0 leaves the
    /// trace with no reference constants (no `LoadFromGcTable` emitted).
    pub(crate) fn set_gc_table_base(&mut self, base: usize) {
        self.gc_table_base = base;
    }

    /// `compile.py:665` parity: heap-pinned address of `self.cpu`'s
    /// attachment handle, derived from the Arc clone.  Baked into the
    /// CALL_ASSEMBLER helper call site.
    fn cpu_handle_ptr(&self) -> i64 {
        Arc::as_ptr(&self.cpu_handle) as *const () as i64
    }

    /// `compile.py:665-674` parity: attach the six metainterp descrs on
    /// the emission side.  Mirrors `self.cpu.done_with_this_frame_descr_*`
    /// reads in `rpython/jit/backend/aarch64/assembler.py`.  Reads from
    /// the live `cpu_handle` snapshot so the raw pointer baked into
    /// `JF_DESCR_OFS` and the Arc returned by
    /// `done_with_this_frame_descr_arc_for_type` resolve to the same
    /// metainterp singleton.
    fn done_with_this_frame_descr_ptr_for_type(&self, tp: Type) -> i64 {
        self.cpu_handle
            .read()
            .unwrap()
            .descr_ptrs()
            .done_with_this_frame_descr_ptr_for_type(tp) as i64
    }

    /// `compile.py:665-674` `make_and_attach_done_descrs` Arc lookup —
    /// returns the metainterp `DoneWithThisFrameDescr*` Arc the
    /// optimizer attached for the given result type.  Used to stamp
    /// `meta_descr` on backend FINISH descrs so trait forwarding routes
    /// `is_finish` / `fail_arg_types` through the metainterp class
    /// hierarchy (`compile.py:624 final_descr=True`).
    fn done_with_this_frame_descr_arc_for_type(&self, tp: Type) -> Option<majit_ir::DescrRef> {
        let attachments = self.cpu_handle.read().unwrap();
        match tp {
            Type::Void => attachments.done_with_this_frame_descr_void.clone(),
            Type::Int => attachments.done_with_this_frame_descr_int.clone(),
            Type::Ref => attachments.done_with_this_frame_descr_ref.clone(),
            Type::Float => attachments.done_with_this_frame_descr_float.clone(),
        }
    }

    /// `compile.py:658` parity: `self.cpu.exit_frame_with_exception_descr_ref`.
    fn exit_frame_with_exception_descr_ref_ptr(&self) -> i64 {
        self.cpu_handle
            .read()
            .unwrap()
            .descr_ptrs()
            .exit_frame_with_exception_descr_ref as i64
    }

    /// `pyjitpl.py:2283` parity: `self.cpu.propagate_exception_descr`.
    /// Stamped into `jf_descr` by the inline propagate path emitted at
    /// `OpCode::CheckMemoryError` (opassembler.py:258
    /// `emit_op_check_memory_error` → `propagate_memoryerror_if_reg_is_null`).
    fn propagate_exception_descr_ptr(&self) -> i64 {
        self.cpu_handle
            .read()
            .unwrap()
            .descr_ptrs()
            .propagate_exception_descr as i64
    }

    // ----------------------------------------------------------------
    // Helper methods
    // ----------------------------------------------------------------

    /// Emit the inline propagate-MemoryError sequence: if `reg_x` is
    /// NULL, route through the `propagate_exception_descr` exit
    /// (`_build_propagate_exception_path`, assembler.py:559-577 —
    /// inlined per call site because pyre's dynasm backend doesn't
    /// emit a separate trampoline on aarch64).
    ///
    /// Used by:
    ///   * `OpCode::CheckMemoryError` (after the four CALL_R malloc
    ///     helpers in `gen_call_malloc_gc`).
    ///   * Each inline malloc-nursery slowpath site, where the helper
    ///     (`dynasm_nursery_slowpath` / `_varsize` / `_jitframe`)
    ///     returns x0 = 0 on real host OOM (calloc failure).  Before
    ///     this hook the OOM null was stored straight into a typed
    ///     Ref slot, corrupting subsequent generated stores.
    ///
    /// Skips emission when `propagate_exception_descr` is unattached
    /// (unit tests that bypass `MetaInterp::finish_setup`).
    fn emit_propagate_memory_error_if_null(&mut self, reg_x: u8) {
        let propagate_descr = self.propagate_exception_descr_ptr();
        if propagate_descr == 0 {
            return;
        }
        let skip = self.mc.new_dynamic_label();
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        // x0 is freely clobberable on the propagate path: _call_footer
        // (assembler.py:574) overwrites x0 with fp before returning,
        // so any live value in x0 was already dead the moment we
        // branched into the path.
        dynasm!(self.mc ; .arch aarch64
            ; cbnz X(reg_x), =>skip
        );
        // assembler.py:509-512 — load pos_exc_value into x0,
        // then clear pos_exc_value.
        self.emit_mov_imm64(16, exc_value_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x0, [x16]
            ; str xzr, [x16]
        );
        // assembler.py:535-536 — clear pos_exception.
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64
            ; str xzr, [x16]
            // assembler.py:565 — store x0 → jf_guard_exc.
            ; str x0, [x29, JF_GUARD_EXC_OFS as u32]
        );
        // assembler.py:572-573 — store propagate_descr → jf_descr.
        self.emit_mov_imm64(0, propagate_descr);
        dynasm!(self.mc ; .arch aarch64 ; str x0, [x29, JF_DESCR_OFS as u32]);
        self._call_footer();
        dynasm!(self.mc ; .arch aarch64 ; =>skip);
    }

    /// Frame-pointer-relative byte offset for a given slot index.
    /// Slots are absolute jf_frame indices, including the fixed
    /// JITFRAME-managed prefix. FIRST_ITEM_OFFSET accounts for the object
    /// header and array-length word that precede jf_frame[0].
    fn slot_offset(slot: usize) -> i32 {
        FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32
    }

    /// Resolve an OpRef to either a frame slot offset or an immediate constant.
    /// Cranelift resolve_opref parity: check constants map FIRST
    /// (regardless of CONST_BIT), then fall back to slot mapping.
    fn resolve_opref(&self, opref: OpRef) -> ResolvedArg {
        // Op results take precedence over constants (Cranelift parity).
        if let Some(&slot) = self.opref_to_slot.get(&opref) {
            return ResolvedArg::Slot(Self::slot_offset(slot));
        }
        // history.py:227/268/314 — inline-Const variants carry value inline.
        if let Some(val) = opref
            .inline_const_bits()
            .or_else(|| self.constants.get(&opref.raw()).map(|c| c.as_raw_i64()))
        {
            return ResolvedArg::Const(val);
        }
        // history.py:227/268/314 — Const always carries a value, so a
        // constant OpRef with no resolvable value is an invariant break,
        // not a `#0`.
        if opref.is_constant() {
            panic!(
                "resolve_opref: legacy constant {opref:?} missing from constants pool — \
                 Const always carries a value (history.py:227/268/314)"
            );
        }
        // regalloc.py:102 `FrameManager.loc(must_exist=True)` raises KeyError
        // for a non-constant box that is neither register- nor frame-resident:
        // a used box is always slot-mapped or constant. Silently materializing
        // an unmapped box as `#0` would hide a wrong value — the same hazard
        // that moved genop_call_assembler onto regalloc arglocs. Fail loud.
        panic!(
            "resolve_opref: unmapped non-constant OpRef {opref:?} — every used \
             box must be slot-mapped or constant (regalloc.py:102 loc must_exist)"
        );
    }

    /// Allocate a frame slot for an OpRef and return the ABSOLUTE jitframe
    /// slot index (already offset by JITFRAME_FIXED_SIZE so slot_offset()
    /// yields the user-area byte offset directly).
    ///
    /// Reuses the existing slot if the OpRef has already been allocated.
    fn allocate_slot(&mut self, opref: OpRef) -> usize {
        if let Some(&existing) = self.opref_to_slot.get(&opref) {
            return existing;
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        if self.next_slot + 1 > self.frame_depth {
            self.frame_depth = self.next_slot + 1;
        }
        self.opref_to_slot.insert(opref, slot);
        slot
    }

    // ── Location-aware code emission (RPython regalloc parity) ──
    // assembler.py regalloc_mov: move value between any two locations.

    /// assembler.py:1145 regalloc_mov(from_loc, to_loc).
    /// Emit a move between any two locations: reg↔reg, reg↔frame, imm→reg, imm→frame.
    pub(crate) fn regalloc_mov(&mut self, src: &Loc, dst: &Loc) {
        if majit_ir::debug::have_debug_prints() {
            majit_ir::debug::log_one("jit-backend", &format!("remap-mov: {src:?} -> {dst:?}"));
        }
        match (src, dst) {
            (Loc::Reg(s), Loc::Reg(d)) if s == d => {}
            (Loc::Reg(s), Loc::Reg(d)) => {
                if s.is_xmm && d.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), D(s.value));
                } else if !s.is_xmm && !d.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; mov X(d.value), X(s.value));
                } else if s.is_xmm && !d.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; fmov X(d.value), D(s.value));
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), X(s.value));
                }
            }
            (Loc::Reg(s), Loc::Frame(f)) => {
                let ofs = f.ebp_loc.value;
                if s.is_xmm {
                    self.emit_str_fp_d(s.value, ofs);
                } else {
                    self.emit_str_fp(s.value, ofs);
                }
            }
            (Loc::Frame(f), Loc::Reg(d)) => {
                let ofs = f.ebp_loc.value;
                if d.is_xmm {
                    self.emit_ldr_fp_d(d.value, ofs);
                } else {
                    self.emit_ldr_fp(d.value, ofs);
                }
            }
            (Loc::Immed(i), Loc::Reg(d)) => {
                if d.is_xmm {
                    self.emit_mov_imm64(16, i.value); // x16 = scratch
                    dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), X(16));
                } else {
                    self.emit_mov_imm64(d.value as u32, i.value);
                }
            }
            (Loc::Immed(i), Loc::Frame(f)) => {
                let ofs = f.ebp_loc.value;
                self.emit_mov_imm64(16, i.value);
                self.emit_str_fp(16, ofs);
            }
            (Loc::Frame(f1), Loc::Frame(f2)) if f1.position == f2.position => {}
            (Loc::Frame(f1), Loc::Frame(f2)) => {
                let o1 = f1.ebp_loc.value;
                let o2 = f2.ebp_loc.value;
                self.emit_ldr_fp(16, o1);
                self.emit_str_fp(16, o2);
            }
            _ => {}
        }
    }

    fn loc_as_key(loc: &Loc) -> i32 {
        match loc {
            Loc::Reg(r) if r.is_xmm => 0x2000 + i32::from(r.value),
            Loc::Reg(r) => 0x1000 + i32::from(r.value),
            Loc::Frame(f) => f.ebp_loc.value,
            Loc::Ebp(e) => e.value,
            Loc::Immed(_) => i32::MIN,
            Loc::Addr(a) => a.offset,
        }
    }

    fn loc_width(loc: &Loc) -> usize {
        match loc {
            Loc::Reg(r) => r.get_width(),
            Loc::Frame(f) => f.ebp_loc.get_width(),
            Loc::Ebp(e) => e.get_width(),
            _ => WORD,
        }
    }

    fn regalloc_push(&mut self, loc: &Loc) {
        match loc {
            Loc::Reg(r) if r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; str D(r.value), [sp, #-16]!);
            }
            Loc::Reg(r) => {
                dynasm!(self.mc ; .arch aarch64 ; str X(r.value), [sp, #-16]!);
            }
            Loc::Frame(f) if f.ebp_loc.is_float => {
                self.emit_ldr_fp_d(15, f.ebp_loc.value);
                dynasm!(self.mc ; .arch aarch64 ; str D(15), [sp, #-16]!);
            }
            Loc::Frame(f) => {
                self.emit_ldr_fp(16, f.ebp_loc.value);
                dynasm!(self.mc ; .arch aarch64 ; str x16, [sp, #-16]!);
            }
            _ => {}
        }
    }

    fn regalloc_pop(&mut self, loc: &Loc) {
        match loc {
            Loc::Reg(r) if r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; ldr D(r.value), [sp], #16);
            }
            Loc::Reg(r) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr X(r.value), [sp], #16);
            }
            Loc::Frame(f) if f.ebp_loc.is_float => {
                dynasm!(self.mc ; .arch aarch64 ; ldr D(15), [sp], #16);
                self.emit_str_fp_d(15, f.ebp_loc.value);
            }
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [sp], #16);
                self.emit_str_fp(16, f.ebp_loc.value);
            }
            _ => {}
        }
    }

    fn remap_frame_layout(&mut self, src_locations: &[Loc], dst_locations: &[Loc], tmpreg: Loc) {
        let mut pending_dests = dst_locations.len() as i32;
        let mut srccount: IndexMap<i32, i32> = IndexMap::new();
        for dst in dst_locations {
            srccount.insert(Self::loc_as_key(dst), 0);
        }
        for i in 0..dst_locations.len() {
            let src = src_locations[i];
            if src.is_immed() {
                continue;
            }
            let key = Self::loc_as_key(&src);
            if let Some(cnt) = srccount.get_mut(&key) {
                if key == Self::loc_as_key(&dst_locations[i]) {
                    *cnt = -(dst_locations.len() as i32) - 1;
                    pending_dests -= 1;
                } else {
                    *cnt += 1;
                }
            }
        }

        while pending_dests > 0 {
            let mut progress = false;
            for i in 0..dst_locations.len() {
                let dst = dst_locations[i];
                let key = Self::loc_as_key(&dst);
                if srccount.get(&key).copied().unwrap_or(-1) == 0 {
                    srccount.insert(key, -1);
                    pending_dests -= 1;
                    let src = src_locations[i];
                    if !src.is_immed() {
                        let src_key = Self::loc_as_key(&src);
                        if let Some(cnt) = srccount.get_mut(&src_key) {
                            *cnt -= 1;
                        }
                    }
                    if dst.is_stack() && src.is_stack() {
                        self.regalloc_mov(&src, &tmpreg);
                        self.regalloc_mov(&tmpreg, &dst);
                    } else {
                        self.regalloc_mov(&src, &dst);
                    }
                    progress = true;
                }
            }
            if !progress {
                let mut sources: IndexMap<i32, Loc> = IndexMap::new();
                for i in 0..dst_locations.len() {
                    sources.insert(Self::loc_as_key(&dst_locations[i]), src_locations[i]);
                }
                for dst in dst_locations {
                    let originalkey = Self::loc_as_key(dst);
                    if srccount.get(&originalkey).copied().unwrap_or(-1) >= 0 {
                        self.regalloc_push(dst);
                        let mut cur_dst = *dst;
                        loop {
                            let key = Self::loc_as_key(&cur_dst);
                            srccount.insert(key, -1);
                            pending_dests -= 1;
                            let src = sources[&key];
                            if Self::loc_as_key(&src) == originalkey {
                                break;
                            }
                            if cur_dst.is_stack() && src.is_stack() {
                                self.regalloc_mov(&src, &tmpreg);
                                self.regalloc_mov(&tmpreg, &cur_dst);
                            } else {
                                self.regalloc_mov(&src, &cur_dst);
                            }
                            cur_dst = src;
                        }
                        self.regalloc_pop(&cur_dst);
                    }
                }
            }
        }
    }

    fn remap_frame_layout_mixed(
        &mut self,
        src_locations1: &[Loc],
        dst_locations1: &[Loc],
        tmpreg1: Loc,
        src_locations2: &[Loc],
        dst_locations2: &[Loc],
        tmpreg2: Loc,
    ) {
        let mut extrapushes = Vec::new();
        let mut dst_keys = IndexMap::new();
        for loc in dst_locations1 {
            dst_keys.insert(Self::loc_as_key(loc), ());
        }
        let mut src_locations2red = Vec::new();
        let mut dst_locations2red = Vec::new();
        for i in 0..src_locations2.len() {
            let loc = src_locations2[i];
            let dstloc = dst_locations2[i];
            if loc.is_stack() {
                let key = Self::loc_as_key(&loc);
                if dst_keys.contains_key(&key)
                    || (Self::loc_width(&loc) > WORD && dst_keys.contains_key(&(key + WORD as i32)))
                {
                    self.regalloc_push(&loc);
                    extrapushes.push(dstloc);
                    continue;
                }
            }
            src_locations2red.push(loc);
            dst_locations2red.push(dstloc);
        }
        self.remap_frame_layout(src_locations1, dst_locations1, tmpreg1);
        self.remap_frame_layout(&src_locations2red, &dst_locations2red, tmpreg2);
        while let Some(loc) = extrapushes.pop() {
            self.regalloc_pop(&loc);
        }
    }

    /// Load a (lhs, src) pair for a 3-operand binop into registers,
    /// returning the register numbers to use. Uses x17 for lhs scratch
    /// and x16 for src scratch when the loc is Frame/Immed.
    fn load_3op_into_scratch(&mut self, lhs: &Loc, src: &Loc) -> (u8, u8) {
        let lhs_reg = self.load_loc_to_reg(lhs, 17);
        let src_reg = self.load_loc_to_reg(src, 16);
        (lhs_reg, src_reg)
    }

    /// Load a single operand `loc` into a register, returning its number.
    /// Register locs are used in place; Frame/Immed locs are materialised
    /// into the caller-supplied `scratch` register (x16/x17).
    fn load_loc_to_reg(&mut self, loc: &Loc, scratch: u8) -> u8 {
        match loc {
            Loc::Reg(r) => r.value,
            Loc::Frame(f) => {
                self.emit_ldr_fp(scratch, f.ebp_loc.value);
                scratch
            }
            Loc::Immed(i) => {
                self.emit_mov_imm64(scratch as u32, i.value);
                scratch
            }
            // The regalloc only routes Reg/Frame/Immed operands into the 3-op /
            // overflow arith paths; an Ebp/Addr loc here means a broken regalloc
            // contract. Panic instead of emitting arith on a stale scratch reg.
            other => panic!("load_loc_to_reg expected Reg/Frame/Immed, got {other:?}"),
        }
    }

    /// If `loc` is a `ConstInt` that fits the AArch64 `#imm12` add/sub
    /// immediate range, return it for the immediate instruction form.
    /// Range mirrors `check_imm_box` (`0 <= v < DEFAULT_IMM_SIZE` = 4096).
    fn addsub_imm12(loc: &Loc) -> Option<i64> {
        match loc {
            Loc::Immed(i) if (0..4096).contains(&i.value) => Some(i.value),
            _ => None,
        }
    }

    /// Emit a 3-operand integer binop: `OP dst, lhs, src`.
    ///
    /// aarch64/opassembler.py parity: `ADD/SUB/AND/ORR/EOR Rd, Rn, Rm` —
    /// the hardware supports distinct Rd and Rn, unlike x86's 2-operand
    /// `SUB dst, dst, src`. Handles reg/frame/immediate forms for both
    /// operands, routing scratch via x16/x17 as needed.
    fn emit_binop_3op(&mut self, opcode: OpCode, dst_reg: u8, lhs: &Loc, src: &Loc) {
        // Fold a small non-negative ConstInt rhs into the `add/sub Rd,Rn,#imm12`
        // immediate form instead of materialising it into a scratch register.
        // The regalloc (`consider_int_ri_j2` / `consider_int_sub_j2`) only keeps
        // add/sub rhs as `Loc::Immed`; the range mirrors `check_imm_box`
        // (`0 <= v < DEFAULT_IMM_SIZE` = 4096, the AArch64 `#imm12` range).
        if matches!(
            opcode,
            OpCode::IntAdd | OpCode::IntSub | OpCode::NurseryPtrIncrement
        ) {
            if let Some(im) = Self::addsub_imm12(src) {
                let l = self.load_loc_to_reg(lhs, 17);
                let d = dst_reg;
                // Rd/Rn take the SP-form register family for the `#imm12`
                // encoding; `XSP(r)` encodes identically to `X(r)` for the
                // non-SP registers the allocator hands out here.
                match opcode {
                    OpCode::IntSub => {
                        dynasm!(self.mc ; .arch aarch64 ; sub XSP(d), XSP(l), im as u32)
                    }
                    _ => dynasm!(self.mc ; .arch aarch64 ; add XSP(d), XSP(l), im as u32),
                }
                return;
            }
        }
        let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
        let d = dst_reg;
        let l = lhs_reg as u8;
        let s = src_reg as u8;
        match opcode {
            OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                dynasm!(self.mc ; .arch aarch64 ; add X(d), X(l), X(s));
            }
            OpCode::IntSub | OpCode::IntSubOvf => {
                dynasm!(self.mc ; .arch aarch64 ; sub X(d), X(l), X(s));
            }
            OpCode::IntMul | OpCode::IntMulOvf => {
                dynasm!(self.mc ; .arch aarch64 ; mul X(d), X(l), X(s));
            }
            OpCode::IntAnd => {
                dynasm!(self.mc ; .arch aarch64 ; and X(d), X(l), X(s));
            }
            OpCode::IntOr => {
                dynasm!(self.mc ; .arch aarch64 ; orr X(d), X(l), X(s));
            }
            OpCode::IntXor => {
                dynasm!(self.mc ; .arch aarch64 ; eor X(d), X(l), X(s));
            }
            _ => {}
        }
    }

    /// Emit: ADD/SUB/AND/OR/XOR reg, loc
    fn emit_binop_reg_loc(&mut self, opcode: OpCode, dst_reg: u8, src: &Loc) {
        // aarch64: load src to x16 scratch if not in register
        let src_reg = match src {
            Loc::Reg(s) => s.value,
            Loc::Frame(f) => {
                self.emit_ldr_fp(16, f.ebp_loc.value);
                16
            }
            Loc::Immed(i) => {
                self.emit_mov_imm64(16, i.value);
                16
            }
            _ => return,
        };
        let d = dst_reg;
        let s = src_reg as u8;
        match opcode {
            OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                dynasm!(self.mc ; .arch aarch64 ; add X(d), X(d), X(s));
            }
            OpCode::IntSub | OpCode::IntSubOvf => {
                dynasm!(self.mc ; .arch aarch64 ; sub X(d), X(d), X(s));
            }
            OpCode::IntMul | OpCode::IntMulOvf => {
                dynasm!(self.mc ; .arch aarch64 ; mul X(d), X(d), X(s));
            }
            OpCode::IntAnd => {
                dynasm!(self.mc ; .arch aarch64 ; and X(d), X(d), X(s));
            }
            OpCode::IntOr => {
                dynasm!(self.mc ; .arch aarch64 ; orr X(d), X(d), X(s));
            }
            OpCode::IntXor => {
                dynasm!(self.mc ; .arch aarch64 ; eor X(d), X(d), X(s));
            }
            _ => {}
        }
        return;
    }

    /// Emit: CMP loc0, loc1
    fn emit_cmp_loc_loc(&mut self, loc0: &Loc, loc1: &Loc) {
        // Load loc0 into x16 if needed, loc1 into x17 if needed
        let r0 = match loc0 {
            Loc::Reg(r) => r.value,
            Loc::Frame(f) => {
                self.emit_ldr_fp(16, f.ebp_loc.value);
                16
            }
            Loc::Immed(i) => {
                self.emit_mov_imm64(16, i.value);
                16
            }
            _ => return,
        };
        let r1 = match loc1 {
            Loc::Reg(s) => s.value,
            Loc::Frame(f) => {
                self.emit_ldr_fp(17, f.ebp_loc.value);
                17
            }
            Loc::Immed(i) => {
                self.emit_mov_imm64(17, i.value);
                17
            }
            _ => return,
        };
        dynasm!(self.mc ; .arch aarch64 ; cmp X(r0 as u8), X(r1 as u8));
        return;
    }

    /// Emit: TEST loc, loc (for guard_true/guard_false)
    fn emit_test_loc(&mut self, loc: &Loc) {
        let r = match loc {
            Loc::Reg(r) => r.value,
            Loc::Frame(f) => {
                self.emit_ldr_fp(16, f.ebp_loc.value);
                16
            }
            Loc::Immed(i) => {
                self.emit_mov_imm64(16, i.value);
                16
            }
            _ => return,
        };
        dynasm!(self.mc ; .arch aarch64 ; tst X(r as u8), X(r as u8));
        return;
    }

    /// Maximum unsigned immediate for `ldr/str X, [base, #imm]` (64-bit).
    /// AArch64 scaled unsigned offset: 12-bit field × 8 = 0..32760.
    const MAX_LDR_STR_UIMM: i32 = 32760;

    /// Emit `str Xsrc, [x29, #offset]`, using x16 as scratch when the
    /// offset exceeds the unsigned-immediate range of the instruction.
    fn emit_str_fp(&mut self, src: u8, offset: i32) {
        if offset >= 0 && offset <= Self::MAX_LDR_STR_UIMM {
            dynasm!(self.mc ; .arch aarch64
                ; str X(src), [x29, offset as u32]
            );
        } else {
            // Large offset: add x16, x29, #offset; str Xsrc, [x16]
            self.emit_mov_imm64(16, offset as i64);
            dynasm!(self.mc ; .arch aarch64
                ; add x16, x29, x16
                ; str X(src), [x16]
            );
        }
    }

    /// Emit `str Dsrc, [x29, #offset]` (float/double), using x16 as scratch
    /// when the offset exceeds the unsigned-immediate range.
    fn emit_str_fp_d(&mut self, src: u8, offset: i32) {
        if offset >= 0 && offset <= Self::MAX_LDR_STR_UIMM {
            dynasm!(self.mc ; .arch aarch64
                ; str D(src), [x29, offset as u32]
            );
        } else {
            self.emit_mov_imm64(16, offset as i64);
            dynasm!(self.mc ; .arch aarch64
                ; add x16, x29, x16
                ; str D(src), [x16]
            );
        }
    }

    /// Emit `ldr Ddst, [x29, #offset]` (float/double), using x16 as scratch
    /// when the offset exceeds the unsigned-immediate range.
    fn emit_ldr_fp_d(&mut self, dst: u8, offset: i32) {
        if offset >= 0 && offset <= Self::MAX_LDR_STR_UIMM {
            dynasm!(self.mc ; .arch aarch64
                ; ldr D(dst), [x29, offset as u32]
            );
        } else {
            self.emit_mov_imm64(16, offset as i64);
            dynasm!(self.mc ; .arch aarch64
                ; add x16, x29, x16
                ; ldr D(dst), [x16]
            );
        }
    }

    /// Emit `ldr Xdst, [x29, #offset]`, using x16 as scratch when the
    /// offset exceeds the unsigned-immediate range of the instruction.
    fn emit_ldr_fp(&mut self, dst: u8, offset: i32) {
        if offset >= 0 && offset <= Self::MAX_LDR_STR_UIMM {
            dynasm!(self.mc ; .arch aarch64
                ; ldr X(dst), [x29, offset as u32]
            );
        } else {
            // Large offset: add x16, x29, #offset; ldr Xdst, [x16]
            self.emit_mov_imm64(16, offset as i64);
            dynasm!(self.mc ; .arch aarch64
                ; add x16, x29, x16
                ; ldr X(dst), [x16]
            );
        }
    }

    /// Emit: load the value of `opref` into RAX (x64) / X0 (aarch64).
    fn load_arg_to_rax(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                self.emit_ldr_fp(0, offset);
            }
            ResolvedArg::Const(val) => {
                self.emit_mov_imm64(0, val);
            }
        }
    }

    /// Emit: load a regalloc Loc into RAX (x64) / X0 (aarch64).
    /// Unlike load_arg_to_rax, this uses the regalloc-determined location
    /// instead of resolve_opref(), so register-carried values are preserved.
    fn emit_load_to_rax(&mut self, loc: Loc) {
        let rax = Loc::Reg(crate::regloc::RegLoc {
            value: 0,
            is_xmm: false,
        });
        match loc {
            Loc::Reg(r) if r.value == 0 && !r.is_xmm => {
                // already in rax/x0
            }
            Loc::Immed(imm) => {
                self.emit_mov_imm64(0, imm.value);
            }
            _ => self.regalloc_mov(&loc, &rax),
        }
    }

    /// Emit: load the value of `opref` into RCX (x64) / X1 (aarch64).
    fn load_arg_to_rcx(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                self.emit_ldr_fp(1, offset);
            }
            ResolvedArg::Const(val) => {
                self.emit_mov_imm64(1, val);
            }
        }
    }

    /// Emit: store RAX/X0 to the frame slot for `result_opref`.
    /// Allocates a new slot if needed.
    fn store_rax_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        self.emit_str_fp(0, offset);
    }

    // ----------------------------------------------------------------
    // assembler.py:543 _call_header — function prologue
    // ----------------------------------------------------------------

    fn setup_input_state(&mut self, inputargs: &[InputArg]) {
        // opref_to_slot stores ABSOLUTE jitframe slot indices so that
        // slot_offset(slot) returns the correct byte offset directly.
        // User position `p` maps to absolute slot `p + JITFRAME_FIXED_SIZE`.
        if let Some(ref input_locs) = self.bridge_input_locs {
            let mut max_abs_slot = JITFRAME_FIXED_SIZE;
            for (ia, loc) in inputargs.iter().zip(input_locs.iter()) {
                if let Loc::Frame(floc) = loc {
                    let abs_slot = JITFRAME_FIXED_SIZE + floc.position;
                    self.opref_to_slot.insert(ia.opref(), abs_slot);
                    if abs_slot + 1 > max_abs_slot {
                        max_abs_slot = abs_slot + 1;
                    }
                }
            }
            self.next_slot = max_abs_slot;
        } else {
            for (i, ia) in inputargs.iter().enumerate() {
                self.opref_to_slot
                    .insert(ia.opref(), JITFRAME_FIXED_SIZE + i);
            }
            self.next_slot = JITFRAME_FIXED_SIZE + inputargs.len();
        }
    }

    /// Emit the function prologue.
    /// x64: System V AMD64 ABI — first arg (jf_ptr) in RDI.
    /// aarch64: AAPCS64 — first arg (jf_ptr) in X0.
    ///
    /// Save slots: 48 bytes total (fp/lr at [sp,#0], x19/x20 at
    /// [sp,#16], x21/x22 at [sp,#32]).  Mirrors aarch64/assembler.py:1117-1122
    /// which iterates `r.callee_saved_registers = [x19, x20, x21, x22]`.
    ///
    /// aarch64/assembler.py:1092-1114 `_call_header_with_stack_check`:
    /// inline SP probe right after the frame pointer is captured,
    /// before `gen_shadowstack_header`. On overflow, return the
    /// incoming jf_ptr in x0 without pushing to the shadow stack so
    /// the JIT glue drains the overflow flag on the way back to the
    /// interpreter.
    ///
    /// Inline probe (aarch64/assembler.py:1099-1114 parity):
    /// ```text
    ///   gen_load_int x30, endaddr       ; load endaddr
    ///   LDR  x30, [x30]                  ; x30 = end
    ///   gen_load_int x16, lengthaddr
    ///   LDR  x16, [x16]                  ; x16 = length
    ///   MOV  x17, sp                     ; x17 = current sp (sp can't be
    ///                                    ;   the source of SUB_rr directly)
    ///   SUB  x30, x30, x17               ; x30 = ofs = end - sp
    ///   CMP  x30, x16
    ///   B.LS continue                    ; fast path: ofs <= length
    ///   MOV  x0, sp                      ; arg0 = current sp
    ///   gen_load_int x17, slowpath
    ///   BLR  x17                         ; call pyre_stack_too_big_slowpath
    ///   CBZ  w0, continue                ; slowpath: 0 = OK
    ///   ; fallthrough = real overflow → return x29 as jf_ptr
    /// ```
    fn _call_header(&mut self, inputargs: &[InputArg]) {
        dynasm!(self.mc ; .arch aarch64
            ; stp x29, x30, [sp, #-48]!
            ; stp x19, x20, [sp, #16]   // save callee-saved regs
            ; stp x21, x22, [sp, #32]   // save callee-saved regs
            ; mov x29, x0
        );
        let propagate_descr = self.propagate_exception_descr_ptr();
        if propagate_descr != 0 {
            if let Some(addrs) = crate::stack_check_addresses() {
                let continue_label = self.mc.new_dynamic_label();
                let exc_value_addr = crate::jit_exc_value_addr() as i64;
                let exc_type_addr = crate::jit_exc_type_addr() as i64;
                // Fast path: load end, subtract sp, compare with length.
                self.emit_mov_imm64(30, addrs.end_adr as i64); // lr holds end addr
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x30, [x30]                  // x30 = *end_adr
                );
                self.emit_mov_imm64(16, addrs.length_adr as i64); // ip0 holds length addr
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x16, [x16]                  // x16 = *length_adr
                    ; mov x17, sp                     // x17 = sp (can't SUB sp directly)
                    ; sub x30, x30, x17               // x30 = ofs = end - sp
                    ; cmp x30, x16
                    ; b.ls =>continue_label           // fast path OK: ofs <= length
                    // Slow path: call pyre_stack_too_big_slowpath(sp).
                    ; mov x0, sp
                );
                self.emit_mov_imm64(17, addrs.slowpath_addr as i64);
                dynasm!(self.mc ; .arch aarch64
                    ; blr x17
                    ; cbz w0, =>continue_label        // slowpath says OK
                );
                // aarch64/assembler.py `_build_stack_check_slowpath` jumps
                // into the same propagate-exception path as x86: move
                // pos_exc_value into jf_guard_exc, clear pos_exception, stamp
                // propagate_exception_descr, then return the incoming jf_ptr.
                self.emit_mov_imm64(16, exc_value_addr);
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x0, [x16]
                    ; str xzr, [x16]
                    ; str x0, [x29, JF_GUARD_EXC_OFS as u32]
                );
                self.emit_mov_imm64(16, exc_type_addr);
                dynasm!(self.mc ; .arch aarch64
                    ; str xzr, [x16]
                );
                self.emit_mov_imm64(0, propagate_descr);
                dynasm!(self.mc ; .arch aarch64
                    ; str x0, [x29, JF_DESCR_OFS as u32]
                    // Overflow fallthrough: return x29 as jf_ptr.
                    ; mov x0, x29
                    ; ldp x19, x20, [sp, #16]
                    ; ldp x21, x22, [sp, #32]
                    ; ldp x29, x30, [sp], #48
                    ; ret
                    ; =>continue_label
                );
            }
        }
        // When addresses are not registered (tests / early startup), no
        // stack check is emitted — aarch64/assembler.py:1094-1095 parity.
        self.gen_shadowstack_header();
        self.setup_input_state(inputargs);
    }

    // ----------------------------------------------------------------
    // assembler.py:2153 _call_footer — function epilogue
    // ----------------------------------------------------------------

    /// Emit the function epilogue: return jf_ptr in RAX/X0.
    fn _call_footer(&mut self) {
        self.gen_footer_shadowstack();
        dynasm!(self.mc ; .arch aarch64
            ; mov x0, x29
            ; ldp x19, x20, [sp, #16]   // restore callee-saved regs
            ; ldp x21, x22, [sp, #32]   // restore callee-saved regs
            ; ldp x29, x30, [sp], #48
            ; ret
        );
    }

    /// aarch64/assembler.py:1422 `gen_shadowstack_header` parity:
    ///
    /// ```python
    ///   rst = gcrootmap.get_root_stack_top_addr()
    ///   mc.gen_load_int(r.ip1.value, rst)
    ///   self.load_reg(mc, r.x8, r.ip1)   # x8 = *rst = root_stack_top
    ///   mc.gen_load_int(r.ip0.value, 1)
    ///   self.store_reg(mc, r.ip0, r.x8)  # x8[0] = 1 (is_minor marker)
    ///   self.store_reg(mc, r.fp, r.x8, WORD) # x8[WORD] = fp (jf_ptr)
    ///   mc.ADD_ri(r.x8.value, r.x8.value, 2 * WORD)
    ///   self.store_reg(mc, r.x8, r.ip1)  # *rst = x8
    /// ```
    ///
    /// Pushes two words onto the jf shadow stack: the `is_minor` marker
    /// (1) and the current jitframe pointer. Done inline on every JIT
    /// function entry so the collector can walk jf roots without ever
    /// calling into Rust.
    fn gen_shadowstack_header(&mut self) {
        let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        // Register assignment mirrors assembler.py:1426-1436 literally:
        // r.ip1 = x17 holds the rst address; r.ip0 = x16 holds the
        // `1` is_minor marker; r.x8 = x8 holds the loaded top.
        self.emit_mov_imm64(17, rst);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x8, [x17]             // x8 = *rst = root_stack_top
            ; mov x16, 1                 // is_minor marker
            ; str x16, [x8]             // [x8] = 1
            ; str x29, [x8, 8]          // [x8 + WORD] = fp (jf_ptr)
            ; add x8, x8, 16            // x8 += 2*WORD
            ; str x8, [x17]             // *rst = x8
        );
    }

    /// aarch64/assembler.py:1438 `gen_footer_shadowstack` parity:
    ///
    /// ```python
    ///   rst = gcrootmap.get_root_stack_top_addr()
    ///   mc.gen_load_int(r.ip0.value, rst)
    ///   self.load_reg(mc, r.ip1, r.ip0)  # ip1 = *rst = top
    ///   mc.SUB_ri(r.ip1.value, r.ip1.value, 2 * WORD)
    ///   self.store_reg(mc, r.ip1, r.ip0) # *rst = ip1
    /// ```
    fn gen_footer_shadowstack(&mut self) {
        let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        self.emit_mov_imm64(16, rst);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x17, [x16]            // x17 = *rst = top
            ; sub x17, x17, 16          // top -= 2*WORD
            ; str x17, [x16]            // *rst = top
        );
    }

    /// assembler.py:993 push_gcmap.
    fn push_gcmap(&mut self, gcmap: *mut usize) {
        let gcmap_ptr = gcmap as i64;
        self.emit_mov_imm64(16, gcmap_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str x16, [x29, JF_GCMAP_OFS as u32]
        );
    }

    /// assembler.py:1000 pop_gcmap.
    fn pop_gcmap(&mut self) {
        dynasm!(self.mc ; .arch aarch64
            ; str xzr, [x29, JF_GCMAP_OFS as u32]
        );
    }

    /// aarch64/assembler.py:927 `_check_frame_depth` +
    /// `build_frame_realloc_slowpath` (assembler.py:434) — emit a dynamic
    /// JITFRAME depth check at bridge entry.  If the in-flight frame is
    /// shorter than the bridge's required depth, fall into an inlined
    /// realloc slowpath that grows the frame via `dynasm_realloc_frame`.
    ///
    /// Mirrors the x86 port: the upstream backends `BL` a shared
    /// `_frame_realloc_slowpath` stub, but pyre inlines its body so the
    /// gcmap (a compile-time pointer) and the depth (a patchable
    /// immediate) need not be marshalled through the stack.
    ///
    /// The depth immediate is materialized by `emit_mov_imm64_fixed4` as a
    /// fixed 4-instruction `movz/movk` block; `frame_depth_to_patch` records
    /// each block offset (CMP operand + slowpath ARG1) and `patch_stack_checks`
    /// rewrites all four words at finalize once `frame_depth` is final.
    /// Unlike upstream's variable-length `gen_load_int` (which reserves
    /// `get_max_size_of_gen_load_int()` NOPs), this block is always four
    /// words, so no NOP padding is needed.  (The general `emit_mov_imm64`
    /// is variable length and would let the patch overrun the block.)
    fn emit_check_frame_depth(&mut self, gcmap: *mut usize) {
        let frame_len_ofs = (JF_FRAME_OFS + crate::jitframe::LENGTHOFS) as u32;
        let placeholder: i64 = 0xffffff;

        // assembler.py:935 — LDR ip0, [fp, lenofs] — current frame length.
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x29, frame_len_ofs]
        );
        // assembler.py:936-941 — gen_load_int ip1, expected_size (patched).
        let cmp_imm_ofs = self.mc.offset().0;
        self.emit_mov_imm64_fixed4(17, placeholder);
        self.frame_depth_to_patch.push(cmp_imm_ofs);

        // assembler.py:942-944 — CMP ip0, ip1; B.GE skip (fast path:
        // frame length >= required depth → no realloc).
        let continue_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64
            ; cmp x16, x17
            ; b.ge =>continue_label
        );

        // ── inlined build_frame_realloc_slowpath (assembler.py:434-493) ──
        // a) store all registers in the jitframe.
        self.push_all_regs_to_jitframe(&[], true);
        // c) store the gcmap in the jitframe (push_gcmap bakes it as imm).
        self.push_gcmap(gcmap);

        // assembler.py:461 _store_and_reset_exception(None, x19, on_frame=True):
        // jf_guard_exc ← *pos_exc_value (survives the realloc frame copy);
        // x19 ← *pos_exception (callee-saved across the C call); then clear
        // both globals so the helper sees no leftover state.  ip0/ip1 (x16/x17)
        // are scratch and x19 is saved/restored by push/pop_all_regs.
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        self.emit_mov_imm64(16, exc_value_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]                            // x16 = *pos_exc_value
            ; str x16, [x29, JF_GUARD_EXC_OFS as u32]   // jf_guard_exc = excval
        );
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x19, [x16]                            // x19 = *pos_exception
            ; str xzr, [x16]                            // *pos_exception = 0
        );
        self.emit_mov_imm64(16, exc_value_addr);
        dynasm!(self.mc ; .arch aarch64
            ; str xzr, [x16]                            // *pos_exc_value = 0
        );

        // assembler.py:458 MOV x0 = fp (arg0 = old frame).
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x29);
        // assembler.py:447-449 arg1 = expected_size (depth), patched imm.
        let arg1_imm_ofs = self.mc.offset().0;
        self.emit_mov_imm64_fixed4(1, placeholder);
        self.frame_depth_to_patch.push(arg1_imm_ofs);

        // assembler.py:467 BL realloc_frame.  pyre bakes the C-ABI wrapper
        // address as an immediate and calls through ip1.  `blr` clobbers lr
        // (x30), which is safe here: a bridge reuses the originating loop's
        // stack frame and reloads lr from it at `_call_footer`, treating lr
        // as call-clobbered scratch throughout the body.
        let helper_addr = crate::runner::dynasm_realloc_frame as *const () as i64;
        self.emit_mov_imm64(17, helper_addr);
        dynasm!(self.mc ; .arch aarch64
            ; blr x17
            ; mov x29, x0                               // fp = new jitframe
        );

        // assembler.py:473 _restore_exception(None, x19):
        // *pos_exc_value ← jf_guard_exc (copied into the new frame);
        // *pos_exception ← x19 (preserved across the C call).
        self.emit_mov_imm64(16, exc_value_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x17, [x29, JF_GUARD_EXC_OFS as u32]   // x17 = jf_guard_exc
            ; str x17, [x16]                            // *pos_exc_value = x17
            ; str xzr, [x29, JF_GUARD_EXC_OFS as u32]   // reset jf_guard_exc
        );
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64
            ; str x19, [x16]                            // *pos_exception = x19
        );

        // assembler.py:476-480 — update the shadow-stack top entry so the
        // GC visitor finds the post-realloc frame on the next minor
        // collection.  x0 was clobbered by _restore_exception, so write fp.
        let rst_addr = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        self.emit_mov_imm64(16, rst_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]            // x16 = *rst = root_stack_top
            ; sub x16, x16, 8          // x16 = top - WORD
            ; str x29, [x16]           // [top - WORD] = fp (new frame)
        );

        // assembler.py:483-487 — reset jf_gcmap to 0 + restore registers.
        self.pop_gcmap();
        self.pop_all_regs_from_jitframe(&[], true);

        dynasm!(self.mc ; .arch aarch64 ; =>continue_label);
    }

    /// aarch64/assembler.py:898 `patch_stack_checks` — rewrite each recorded
    /// `gen_load_int` depth placeholder with the final absolute frame depth
    /// (already includes `JITFRAME_FIXED_SIZE`).  No-op when no check was
    /// emitted (loops).  Takes the patch list by slice so the caller can
    /// drive it after `self.mc` has been consumed by `finalize()`.
    fn patch_stack_checks(framedepth: usize, rawstart: usize, offsets: &[usize]) {
        for &ofs in offsets {
            Self::patch_frame_depth(rawstart + ofs, framedepth);
        }
    }

    /// aarch64/assembler.py:948 `_patch_frame_depth` — overwrite the
    /// 4-instruction `movz/movk` block emitted by `emit_mov_imm64` with the
    /// finalised frame depth.  Unlike x86's flat 32-bit immediate, the
    /// aarch64 depth is materialised as four instruction words, so all four
    /// are regenerated.  The destination register is recovered from the
    /// original block (the CMP site targets x17, the ARG1 site x1) and
    /// preserved.  The patched region needs an explicit icache flush since
    /// `finalize()` made the buffer executable before this rewrite.
    fn patch_frame_depth(adr: usize, allocated_depth: usize) {
        let rd = unsafe { (adr as *const u32).read() } & 0x1F;
        let words = Self::encode_mov_imm64_words(rd, allocated_depth as i64);
        codebuf::with_writable(adr as *mut u8, 16, || {
            let p = adr as *mut u32;
            for (i, w) in words.iter().enumerate() {
                unsafe { p.add(i).write_unaligned(*w) };
            }
        });
        flush_icache(adr as *const u8, 16);
    }

    /// RPython `AbstractCallBuilder.emit`: CALL_ASSEMBLER is a collecting
    /// call, so the caller jitframe must publish the regalloc gcmap before
    /// jumping into the callee and clear it only after reloading a possibly
    /// moved frame pointer.
    fn push_pending_call_gcmap(&mut self) -> bool {
        if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.push_gcmap(gcmap as *mut usize);
            true
        } else {
            false
        }
    }

    fn pop_pending_call_gcmap_after_collect(&mut self, pushed: bool) {
        self.reload_frame_if_necessary();
        if pushed {
            self.pop_gcmap();
        }
    }

    /// aarch64/assembler.py:967 `_reload_frame_if_necessary` parity:
    ///
    /// ```python
    ///   rst = gcrootmap.get_root_stack_top_addr()
    ///   mc.gen_load_int(r.ip0.value, rst)
    ///   self.load_reg(mc, r.ip0, r.ip0)       # ip0 = *rst = root_stack_top
    ///   mc.SUB_ri(r.ip0.value, r.ip0.value, WORD)
    ///   mc.LDR_ri(r.fp.value, r.ip0.value, 0) # fp = *(top - WORD) = jf_ptr
    /// ```
    ///
    /// After a collecting helper call the GC may have moved the
    /// jitframe; the visitor rewrites the shadow-stack top entry in
    /// place during copy, so the current pointer lives at
    /// `*(root_stack_top - WORD)`. Reload fp/x29 from there.
    fn reload_frame_if_necessary(&mut self) {
        let rst_addr = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        self.emit_mov_imm64(16, rst_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]            // ip0 = *rst_addr = root_stack_top
            ; sub x16, x16, 8           // ip0 -= WORD
            ; ldr x29, [x16]            // fp = *(top - WORD) = jf_ptr
        );
        // aarch64/assembler.py:972-976 `_reload_frame_if_necessary`:
        // after a collecting helper call, re-apply the non-array write
        // barrier fast path on the current jitframe (`is_frame=True`).
        let loc_base = crate::aarch64::registers::FP;
        self.emit_write_barrier_fastpath_for_base(loc_base, false, None);
    }

    /// aarch64/assembler.py:254 `_push_all_regs_to_jitframe` parity.
    fn push_all_regs_to_jitframe(
        &mut self,
        ignored_regs: &[crate::regloc::RegLoc],
        withfloats: bool,
    ) {
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as u32;
        for (i, reg) in all_gen_regs().iter().enumerate() {
            if ignored_regs.contains(reg) {
                continue;
            }
            let ofs = base_ofs + (i as u32 * WORD as u32);
            dynasm!(self.mc ; .arch aarch64 ; str X(reg.value), [x29, ofs]);
        }
        if withfloats {
            let float_base = base_ofs + (all_gen_regs().len() as u32 * WORD as u32);
            for reg in all_float_regs().iter() {
                let ofs = float_base + (reg.value as u32 * WORD as u32);
                dynasm!(self.mc ; .arch aarch64 ; str D(reg.value), [x29, ofs]);
            }
        }
    }

    /// aarch64/assembler.py:283 `_pop_all_regs_from_jitframe` parity.
    fn pop_all_regs_from_jitframe(
        &mut self,
        ignored_regs: &[crate::regloc::RegLoc],
        withfloats: bool,
    ) {
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as u32;
        for (i, reg) in all_gen_regs().iter().enumerate() {
            if ignored_regs.contains(reg) {
                continue;
            }
            let ofs = base_ofs + (i as u32 * WORD as u32);
            dynasm!(self.mc ; .arch aarch64 ; ldr X(reg.value), [x29, ofs]);
        }
        if withfloats {
            let float_base = base_ofs + (all_gen_regs().len() as u32 * WORD as u32);
            for reg in all_float_regs().iter() {
                let ofs = float_base + (reg.value as u32 * WORD as u32);
                dynasm!(self.mc ; .arch aarch64 ; ldr D(reg.value), [x29, ofs]);
            }
        }
    }

    fn guard_gcmap_from_faillocs(
        &self,
        fail_arg_types: &[Type],
        faillocs: &[Option<Loc>],
    ) -> *mut usize {
        let frame_depth = self.frame_depth.saturating_sub(JITFRAME_FIXED_SIZE);
        let gcmap = allocate_gcmap(frame_depth, JITFRAME_FIXED_SIZE);
        for (tp, loc) in fail_arg_types.iter().zip(faillocs.iter()) {
            if *tp != Type::Ref {
                continue;
            }
            match loc {
                Some(Loc::Reg(r)) => {
                    if let Some(position) = reg_position_in_jitframe(*r) {
                        gcmap_set_bit(gcmap, position);
                    }
                }
                Some(Loc::Frame(f)) => {
                    gcmap_set_bit(gcmap, f.position + JITFRAME_FIXED_SIZE);
                }
                _ => {}
            }
        }
        gcmap
    }

    fn gcmap_from_fail_arg_locs(
        &self,
        fail_arg_types: &[Type],
        fail_arg_locs: &[Option<usize>],
    ) -> *mut usize {
        let frame_depth = self.frame_depth.saturating_sub(JITFRAME_FIXED_SIZE);
        let gcmap = allocate_gcmap(frame_depth, JITFRAME_FIXED_SIZE);
        for (tp, loc) in fail_arg_types.iter().zip(fail_arg_locs.iter()) {
            if *tp == Type::Ref {
                if let Some(position) = loc {
                    gcmap_set_bit(gcmap, *position);
                }
            }
        }
        gcmap
    }

    // ----------------------------------------------------------------
    // assembler.py:501 assemble_loop
    // ----------------------------------------------------------------

    /// assembler.py:501 assemble_loop: compile a loop trace.
    ///
    /// Returns compiled code with fail descriptors and entry point.
    pub fn assemble_loop(mut self) -> Result<CompiledCode, BackendError> {
        self.input_types = self.inputargs.iter().map(|ia| ia.tp).collect();

        // assembler.py:537 prepare_loop — set up regalloc
        // For now, simplified: all args in frame slots

        // assembler.py:547 _assemble — generate code for all ops
        // Create a dynamic label at the entry point for self-recursive
        // CALL_ASSEMBLER (redirect_call_assembler parity).
        let entry_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; =>entry_label);
        self.self_entry_label = Some(entry_label);
        let entry = self.mc.offset();
        self._assemble(true)?;

        // regalloc sets fail_arg_locs in append_guard_token_with_faillocs.
        // No allocate_unmapped_fail_arg_slots needed.

        // assembler.py:553 write_pending_failure_recoveries
        let stub_offsets = self.write_pending_failure_recoveries();

        // assembler.py:556 materialize_loop — finalize to executable memory
        let buffer = self
            .mc
            .finalize()
            .map_err(|_| BackendError::CompilationFailed("dynasm finalize failed".to_string()))?;

        // assembler.py:849 patch_pending_failure_recoveries
        let rawstart = codebuf::buffer_ptr(&buffer) as usize;
        Self::patch_pending_failure_recoveries(rawstart, &stub_offsets);

        // assembler.py:556 patch_stack_checks — rewrite any
        // `_check_frame_depth` depth placeholders with the loop's final
        // absolute frame depth.  No-op for loops (the list is empty).
        Self::patch_stack_checks(self.frame_depth, rawstart, &self.frame_depth_to_patch);

        // Write resolved entry address for self-recursive CALL_ASSEMBLER
        // trampoline. The JIT code loads from this pointer at runtime.
        unsafe { *self.self_entry_addr_ptr = rawstart + entry.0 };

        for descr in &self.compiled_target_tokens {
            if let Some(loop_descr) = descr.as_loop_target_descr() {
                // assembler.py:1167-1171 reads the target loop's
                // `frame_info.jfi_frame_depth` at a cross-loop JUMP; publish
                // this loop's full frame depth so a later trace's JUMP can
                // size its frame for this target.  Carries the depth grown by
                // `_assemble` for this loop's own onward JUMP.
                //
                // Store the companion `target_frame_depth` BEFORE the
                // `ll_loop_code` gate (both Release stores): a reader
                // Acquire-loads `ll_loop_code` and only reads
                // `target_frame_depth` once the gate is non-zero, so the depth
                // must become visible first — otherwise the reader pairs the
                // new code pointer with a stale 0 depth and bypasses the
                // frame-capacity check (descr.rs set_dispatch_target ordering
                // contract).  Dynasm ignores `label_block_id` (descr.rs:1236 —
                // it bakes the LABEL address straight into `ll_loop_code`), so
                // that companion is not published here.
                loop_descr.set_target_frame_depth(self.frame_depth);
                loop_descr.set_ll_loop_code(loop_descr.ll_loop_code() + rawstart);
            }
        }

        // Position is the canonical fail_index identity (matching
        // `llsupport/assembler.py`'s `_allgcrefs` index — PyPy does not
        // carry per-emission `fail_index` on the descr itself).  Codegen
        // increments the `fail_index` counter in lockstep with
        // `fail_descrs.push`, so the contract is structural rather than
        // descr-internal.  The earlier per-descr assertion was a pyre
        // Deviation removed: singleton FINISH
        // descrs (`compile.py:623-662`) answer the trait-default `0`
        // for `fail_index_per_trace()` regardless of their Vec position.
        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: self.fail_descrs.into_boxed_slice(),
            input_types: self.input_types,
            cpu_attachments: self.cpu_handle,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
            source_guard: None,
        })
    }

    /// assembler.py:320 descr._ll_function_addr parity: store
    /// call_target_token → code_addr mappings for CALL_ASSEMBLER.
    pub fn set_call_assembler_targets(&mut self, targets: IndexMap<u64, usize>) {
        self.call_assembler_targets = targets;
    }

    /// llsupport/assembler.py:201 rebuild_faillocs_from_descr — reconstruct
    /// the locations of bridge inputargs from the guard's recovery layout.
    ///
    /// patch_jump_for_descr overwrites the recovery stub with a direct
    /// jump to the bridge, so the register-save subroutine never runs.
    /// The bridge sees live registers exactly as they were at guard time.
    /// Return Reg locs for register positions, matching RPython.
    pub fn rebuild_faillocs_from_descr(
        descr: &dyn majit_ir::FailDescr,
        inputargs: &[InputArg],
    ) -> Vec<Loc> {
        let mut locs = Vec::new();
        let gpr_regs = all_gen_regs();
        let float_regs = all_float_regs();
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        let mut input_i = 0usize;
        for &pos in descr.rd_locs() {
            if pos == 0xFFFF {
                continue;
            }
            let pos = pos as usize;
            if pos < gpr_regs.len() {
                // llsupport/assembler.py:211 — GPR: return register location
                locs.push(Loc::Reg(gpr_regs[pos]));
            } else if pos < gpr_regs.len() + float_regs.len() {
                // llsupport/assembler.py:213 — FPR: return float register
                locs.push(Loc::Reg(float_regs[pos - gpr_regs.len()]));
            } else {
                // llsupport/assembler.py:217 — frame slot
                let slot = pos - JITFRAME_FIXED_SIZE;
                let tp = inputargs.get(input_i).map(|ia| ia.tp).unwrap_or(Type::Int);
                locs.push(Loc::Frame(crate::regloc::FrameLoc::new(
                    slot,
                    crate::regalloc::get_ebp_ofs(base_ofs, slot),
                    tp == Type::Float,
                )));
            }
            input_i += 1;
        }
        locs
    }

    /// assembler.py:623 assemble_bridge: compile a bridge trace.
    pub fn assemble_bridge(
        mut self,
        fail_descr: &dyn FailDescr,
        arglocs: &[Loc],
    ) -> Result<CompiledCode, BackendError> {
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] assemble_bridge: input_types={:?} arglocs={:?}",
                self.inputargs.iter().map(|ia| ia.tp).collect::<Vec<_>>(),
                arglocs
            );
        }
        self.input_types = self.inputargs.iter().map(|ia| ia.tp).collect();
        self.bridge_input_locs = if arglocs.is_empty() {
            None
        } else {
            Some(arglocs.to_vec())
        };

        // assembler.py:641 prepare_bridge
        let entry = self.mc.offset();
        self._assemble(false)?;
        let stub_offsets = self.write_pending_failure_recoveries();

        let buffer = self
            .mc
            .finalize()
            .map_err(|_| BackendError::CompilationFailed("dynasm finalize failed".to_string()))?;

        let rawstart = codebuf::buffer_ptr(&buffer) as usize;
        Self::patch_pending_failure_recoveries(rawstart, &stub_offsets);

        // assembler.py:556 patch_stack_checks — rewrite the bridge's
        // `_check_frame_depth` depth placeholder with the final absolute
        // frame depth (max of loop/bridge), now known post-finalize.
        Self::patch_stack_checks(self.frame_depth, rawstart, &self.frame_depth_to_patch);

        if crate::majit_dump_enabled() {
            let code = unsafe { std::slice::from_raw_parts(rawstart as *const u8, buffer.len()) };
            eprintln!(
                "[dynasm] BRIDGE CODE DUMP ({} bytes at {:#x}, entry +{:?}):",
                code.len(),
                rawstart,
                entry
            );
            for (i, chunk) in code.chunks(4).enumerate() {
                let word = u32::from_le_bytes([
                    chunk.first().copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                ]);
                eprint!("{:08x} ", word);
                if (i + 1) % 8 == 0 {
                    eprintln!();
                }
            }
            eprintln!();
        }

        // Position is the canonical fail_index identity (matching
        // `llsupport/assembler.py`'s `_allgcrefs` index — PyPy does not
        // carry per-emission `fail_index` on the descr itself).  Codegen
        // increments the `fail_index` counter in lockstep with
        // `fail_descrs.push`, so the contract is structural rather than
        // descr-internal.  The earlier per-descr assertion was a pyre
        // Deviation removed: singleton FINISH
        // descrs (`compile.py:623-662`) answer the trait-default `0`
        // for `fail_index_per_trace()` regardless of their Vec position.
        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: self.fail_descrs.into_boxed_slice(),
            input_types: self.input_types,
            cpu_attachments: self.cpu_handle,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
            source_guard: Some((fail_descr.trace_id(), fail_descr.fail_index_per_trace())),
        })
    }

    /// assembler.py:779 _assemble — walk operations and emit code.
    ///
    /// Uses the register allocator (regalloc.rs) to assign registers/frame
    /// locations, then emits code using those locations. This replaces the
    /// old frame-slot model where every value went through [rbp+offset].
    fn _assemble(&mut self, emit_prologue: bool) -> Result<(), BackendError> {
        let inputargs: &'a [InputArg] = self.inputargs;
        let ops: &'a [Op] = self.operations;
        if emit_prologue {
            self._call_header(inputargs);
        } else {
            self.setup_input_state(inputargs);
        }
        // input_slot_depth is an ABSOLUTE jitframe slot count (already
        // offset by JITFRAME_FIXED_SIZE via setup_input_state).
        let input_slot_depth = self.next_slot;

        // ── Run register allocator ──
        // assembler.py:537 prepare_loop / assembler.py:638 prepare_bridge
        if std::env::var_os("MAJIT_J2PLAN_LOG").is_some() {
            let plan = crate::j2plan::TracePlan::build(inputargs, ops);
            // Independent debug toggle — not gated by MAJIT_LOG.
            eprintln!("[dynasm:j2plan] {}", plan.summary());
        }

        // RegAlloc keeps the raw `i64` value map; project it from the
        // typed pool at this boundary (each Const carries its own type).
        let ra_constants: indexmap::IndexMap<u32, i64> = self
            .constants
            .iter()
            .map(|(&k, c)| (k, c.as_raw_i64()))
            .collect();
        let mut ra = RegAlloc::new(ra_constants, inputargs, ops);
        if let Some(ref arglocs) = self.bridge_input_locs {
            ra.prepare_bridge(arglocs);
        } else {
            ra.prepare_loop();
        }
        // assembler.py:647 — bridges emit `_check_frame_depth` between
        // `prepare_bridge` and the body so the JIT can grow the in-flight
        // JITFRAME if the bridge's frame_depth exceeds the loop's
        // allocation.  Loops skip this (assembler.py:544 uses
        // `_check_frame_depth_debug`, a no-op outside DEBUG_FRAME_DEPTH).
        if self.bridge_input_locs.is_some() {
            let gcmap = ra.get_gcmap(&[], false);
            self.emit_check_frame_depth(gcmap);
        }
        // assembler.py:374 walk_operations — get allocation decisions.
        let ra_ops = ra.walk_operations();
        // ra.get_final_frame_depth() returns a USER-position count; convert
        // to absolute by adding JITFRAME_FIXED_SIZE before comparing.
        let frame_slot_depth =
            input_slot_depth.max(JITFRAME_FIXED_SIZE + ra.get_final_frame_depth());
        self.frame_depth = self.frame_depth.max(frame_slot_depth);

        // Sync regalloc frame positions to opref_to_slot for backward
        // compatibility with genop_call/genop_call_assembler which still
        // use resolve_opref. When regalloc spills a value to a frame slot,
        // that slot's position must be visible to resolve_opref.
        // opref_to_slot stores ABSOLUTE jitframe slots (user position +
        // JITFRAME_FIXED_SIZE) so slot_offset(slot) gives the correct byte
        // offset without further adjustment.
        for iarg in inputargs {
            self.opref_to_slot
                .insert(iarg.opref(), JITFRAME_FIXED_SIZE + iarg.index as usize);
        }
        // Also sync any frame allocations from regalloc's FrameManager.
        for (&opref, lifetime) in ra.longevity.lifetimes_iter() {
            if let Some(floc) = lifetime.current_frame_loc {
                self.opref_to_slot
                    .insert(opref, JITFRAME_FIXED_SIZE + floc.position);
            }
        }
        // frame_slot_depth is already absolute (see calculation above).
        self.next_slot = frame_slot_depth;

        let mut fail_index = 0u32;

        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] _assemble: {} ops → {} ra_ops, frame_depth={}",
                ops.len(),
                ra_ops.len(),
                self.frame_depth
            );
        }

        // ── Emit code from regalloc decisions ──
        for ra_op in &ra_ops {
            match ra_op {
                RegAllocOp::Skip => {
                    // Dead operation — skip.
                    continue;
                }
                RegAllocOp::Move { src, dst } => {
                    if majit_ir::debug::have_debug_prints() {
                        majit_ir::debug::log_one(
                            "jit-backend",
                            &format!("move: {src:?} → {dst:?}"),
                        );
                    }
                    self.regalloc_mov(src, dst);
                    continue;
                }
                RegAllocOp::Perform {
                    op_index,
                    arglocs,
                    result_loc,
                    gcmap,
                } => {
                    let op = &ops[*op_index];
                    if crate::majit_log_enabled() {
                        let al: Vec<String> = arglocs.iter().map(|l| format!("{:?}", l)).collect();
                        eprintln!(
                            "[dynasm] emit[{}]: {:?} args=[{}] result={:?}",
                            op_index,
                            op.opcode,
                            al.join(", "),
                            result_loc
                        );
                    }
                    self.pending_malloc_nursery_gcmap = *gcmap;
                    self.regalloc_perform(
                        op,
                        *op_index,
                        arglocs,
                        result_loc.as_ref(),
                        fail_index,
                        ops,
                    );
                    self.pending_malloc_nursery_gcmap = None;
                }
                RegAllocOp::PerformGuard {
                    op_index,
                    arglocs,
                    result_loc,
                    faillocs,
                } => {
                    let op = &ops[*op_index];
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[dynasm] guard[{}]: {:?} args=[{}] faillocs={}",
                            op_index,
                            op.opcode,
                            arglocs
                                .iter()
                                .map(|l| format!("{:?}", l))
                                .collect::<Vec<_>>()
                                .join(", "),
                            faillocs.len()
                        );
                    }
                    self.regalloc_perform_guard(
                        op,
                        *op_index,
                        arglocs,
                        result_loc.as_ref(),
                        faillocs,
                        fail_index,
                    );
                    fail_index += 1;
                }
                RegAllocOp::PerformDiscard { op_index, arglocs } => {
                    let op = &ops[*op_index];
                    if crate::majit_log_enabled() {
                        let al: Vec<String> = arglocs.iter().map(|l| format!("{:?}", l)).collect();
                        eprintln!(
                            "[dynasm] discard[{}]: {:?} args=[{}]",
                            op_index,
                            op.opcode,
                            al.join(", ")
                        );
                    }
                    self.regalloc_perform(op, *op_index, arglocs, None, fail_index, ops);
                    if op.opcode.is_guard() || op.opcode == OpCode::Finish {
                        fail_index += 1;
                    }
                }
            }
        }

        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] _assemble done: pending_guard_tokens={} fail_index={}",
                self.pending_guard_tokens.len(),
                fail_index
            );
        }

        // assembler.py:1167-1171 `_assemble`: grow the frame to fit a
        // cross-loop JUMP target.  The closing `br` jumps into the target
        // loop's body, which can use deeper frame slots than this trace; for
        // a bridge the prologue `_check_frame_depth` then reallocs the live
        // JITFRAME to this grown depth on entry.  `target_frame_depth` is
        // full-width (includes JITFRAME_FIXED_SIZE), matching `frame_depth`,
        // so no adjustment is needed.  Zero (no external JUMP) is a no-op.
        self.frame_depth = self.frame_depth.max(self.jump_target_frame_depth);

        Ok(())
    }

    /// assembler.py:326 regalloc_perform — emit code for a non-guard op.
    /// Called from the regalloc dispatch loop with pre-computed locations.
    fn regalloc_perform(
        &mut self,
        op: &Op,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<&Loc>,
        fail_index: u32,
        ops: &[Op],
    ) {
        match op.opcode {
            OpCode::IntAddOvf => {
                // RPython aarch64/opassembler.py int_add_impl parity —
                // 3-operand `ADDS Rd, Rn, Rm` (s=1 sets flags for GUARD_NO_OVERFLOW).
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    if let Some(im) = Self::addsub_imm12(src) {
                        let l = self.load_loc_to_reg(lhs, 17);
                        dynasm!(self.mc ; .arch aarch64 ; adds X(dst.value), XSP(l), im as u32);
                    } else {
                        let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
                        dynasm!(self.mc ; .arch aarch64
                            ; adds X(dst.value), X(lhs_reg as u8), X(src_reg as u8));
                    }
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            OpCode::IntSubOvf => {
                // aarch64/opassembler.py int_sub_impl: `SUBS Rd, Rn, Rm`.
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    if let Some(im) = Self::addsub_imm12(src) {
                        let l = self.load_loc_to_reg(lhs, 17);
                        dynasm!(self.mc ; .arch aarch64 ; subs X(dst.value), XSP(l), im as u32);
                    } else {
                        let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
                        dynasm!(self.mc ; .arch aarch64
                            ; subs X(dst.value), X(lhs_reg as u8), X(src_reg as u8));
                    }
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            OpCode::IntMulOvf => {
                // aarch64/opassembler.py emit_comp_op_int_mul_ovf: smulh+mul+asr+cmp
                // against the 64-bit sign-extended high half.
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
                    dynasm!(self.mc ; .arch aarch64
                        ; smulh x15, X(lhs_reg as u8), X(src_reg as u8)
                        ; mul X(dst.value), X(lhs_reg as u8), X(src_reg as u8)
                        ; asr x14, X(dst.value), 63
                        ; cmp x15, x14
                    );
                    self.guard_success_cc = Some(CC_E);
                }
            }
            // ── Integer binary (aarch64 3-operand form) ──
            // RPython parity: aarch64/opassembler.py `int_sub_impl` emits
            // `SUB res, l0, l1` with three distinct locations — NOT the
            // x86 `SUB dst, dst, src` 2-operand pattern. The regalloc may
            // pick `res != arglocs[0]` (see `consider_int_sub` LEA path),
            // so we must read arglocs[0] as the true LHS.
            OpCode::IntAdd
            | OpCode::IntSub
            | OpCode::IntMul
            | OpCode::IntAnd
            | OpCode::IntOr
            | OpCode::IntXor
            | OpCode::NurseryPtrIncrement => {
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    self.emit_binop_3op(op.opcode, dst.value, lhs, src);
                }
            }
            // ── Unary integer ──
            OpCode::IntNeg => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let src_reg = self.load_loc_to_reg(src, 17);
                    dynasm!(self.mc ; .arch aarch64 ; neg X(dst.value), X(src_reg as u8));
                }
            }
            OpCode::IntInvert => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let src_reg = self.load_loc_to_reg(src, 17);
                    dynasm!(self.mc ; .arch aarch64 ; mvn X(dst.value), X(src_reg as u8));
                }
            }
            // ── Shifts ──
            OpCode::IntLshift | OpCode::IntRshift | OpCode::UintRshift => {
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(shift_loc)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    // aarch64: shifts are true 3-operand forms (`lsl/asr/lsr
                    // Rd, Rn, Rm`). The regalloc may pick `result_loc !=
                    // arglocs[0]`, so we must read the real LHS from arg0,
                    // not from the destination register.
                    let lhs_reg = self.load_loc_to_reg(lhs, 17);
                    let sr = self.load_loc_to_reg(shift_loc, 16);
                    match op.opcode {
                        OpCode::IntLshift => {
                            dynasm!(self.mc ; .arch aarch64 ; lsl X(dst.value), X(lhs_reg as u8), X(sr as u8));
                        }
                        OpCode::IntRshift => {
                            dynasm!(self.mc ; .arch aarch64 ; asr X(dst.value), X(lhs_reg as u8), X(sr as u8));
                        }
                        OpCode::UintRshift => {
                            dynasm!(self.mc ; .arch aarch64 ; lsr X(dst.value), X(lhs_reg as u8), X(sr as u8));
                        }
                        _ => {}
                    }
                }
            }
            // ── Integer comparisons ──
            OpCode::IntLt
            | OpCode::IntLe
            | OpCode::IntGt
            | OpCode::IntGe
            | OpCode::IntEq
            | OpCode::IntNe
            | OpCode::UintLt
            | OpCode::UintLe
            | OpCode::UintGt
            | OpCode::UintGe
            | OpCode::PtrEq
            | OpCode::PtrNe
            | OpCode::InstancePtrEq
            | OpCode::InstancePtrNe => {
                if arglocs.len() >= 2 {
                    self.emit_cmp_loc_loc(&arglocs[0], &arglocs[1]);
                }
                if let Some(Loc::Reg(r)) = result_loc {
                    let cc = Self::opcode_to_cc(op.opcode);
                    self.emit_setcc(cc, r.value);
                }
            }
            OpCode::IntIsTrue => {
                if let (Some(src), Some(Loc::Reg(r))) = (arglocs.first(), result_loc) {
                    self.emit_test_loc(src);
                    self.emit_setcc(CC_NE, r.value);
                }
            }
            OpCode::IntIsZero => {
                if let (Some(src), Some(Loc::Reg(r))) = (arglocs.first(), result_loc) {
                    self.emit_test_loc(src);
                    self.emit_setcc(CC_E, r.value);
                }
            }
            OpCode::UintMulHigh => {
                if let Some(Loc::Reg(dst)) = result_loc {
                    if arglocs.len() >= 2 {
                        let lhs = match arglocs[0] {
                            Loc::Reg(r) => r.value,
                            _ => {
                                self.regalloc_mov(
                                    &arglocs[0],
                                    &Loc::Reg(crate::regloc::RegLoc::new(16, false)),
                                );
                                16
                            }
                        };
                        let rhs = match arglocs[1] {
                            Loc::Reg(r) => r.value,
                            _ => {
                                self.regalloc_mov(
                                    &arglocs[1],
                                    &Loc::Reg(crate::regloc::RegLoc::new(17, false)),
                                );
                                17
                            }
                        };
                        dynasm!(self.mc ; .arch aarch64 ; umulh X(dst.value), X(lhs), X(rhs));
                    }
                }
            }
            OpCode::IntForceGeZero => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let src_reg = self.load_loc_to_reg(src, 17);
                    dynasm!(self.mc ; .arch aarch64
                        ; cmp X(src_reg as u8), xzr
                        ; csel X(dst.value), X(src_reg as u8), xzr, ge
                    );
                }
            }
            OpCode::IntSignext => {
                // arglocs = [argloc, numbytes_loc], result_loc = separate reg
                if let (Some(src), Some(Loc::Reg(r))) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, &Loc::Reg(*r));
                    // signext handled by assembler based on numbytes
                }
            }
            // ── Float binary ──
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                if let (Some(Loc::Reg(dst)), Some(lhs_loc), Some(rhs_loc)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    // aarch64 float ops are true 3-operand forms
                    // (`fadd/fsub/fmul/fdiv Dd, Dn, Dm`). The regalloc may
                    // choose `result_loc != arglocs[0]`, so we must read the
                    // actual lhs from arg0 rather than assuming it already
                    // lives in the destination register.
                    let lhs_reg = match lhs_loc {
                        Loc::Reg(r) => *r,
                        _ => {
                            let scratch = crate::regloc::RegLoc::new(15, true);
                            self.regalloc_mov(lhs_loc, &Loc::Reg(scratch));
                            scratch
                        }
                    };
                    let src_reg = if let Loc::Reg(s) = rhs_loc {
                        *s
                    } else {
                        // Immed or Frame — load to scratch XMM (d14/xmm14)
                        let scratch = crate::regloc::RegLoc::new(14, true);
                        self.regalloc_mov(rhs_loc, &Loc::Reg(scratch));
                        scratch
                    };
                    match op.opcode {
                        OpCode::FloatAdd => {
                            dynasm!(self.mc ; .arch aarch64 ; fadd D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        OpCode::FloatSub => {
                            dynasm!(self.mc ; .arch aarch64 ; fsub D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        OpCode::FloatMul => {
                            dynasm!(self.mc ; .arch aarch64 ; fmul D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        OpCode::FloatTrueDiv => {
                            dynasm!(self.mc ; .arch aarch64 ; fdiv D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        _ => {}
                    }
                }
            }
            OpCode::FloatNeg | OpCode::FloatAbs => {
                if let (Some(Loc::Reg(dst)), Some(src_loc)) = (result_loc, arglocs.first()) {
                    // aarch64 fneg/fabs are 2-operand `fneg/fabs Dd, Dn`.
                    // When regalloc chooses `dst != arg0`, reading `Dd` as
                    // the source consumes a stale destination register
                    // instead of the real input, so we must read from
                    // `arglocs[0]` like the float-binop path above.
                    let src_reg = match src_loc {
                        Loc::Reg(r) => *r,
                        _ => {
                            let scratch = crate::regloc::RegLoc::new(15, true);
                            self.regalloc_mov(src_loc, &Loc::Reg(scratch));
                            scratch
                        }
                    };
                    match op.opcode {
                        OpCode::FloatNeg => {
                            dynasm!(self.mc ; .arch aarch64 ; fneg D(dst.value), D(src_reg.value));
                        }
                        OpCode::FloatAbs => {
                            dynasm!(self.mc ; .arch aarch64 ; fabs D(dst.value), D(src_reg.value));
                        }
                        _ => {}
                    }
                }
            }
            // ── Float comparisons ──
            OpCode::FloatLt
            | OpCode::FloatLe
            | OpCode::FloatEq
            | OpCode::FloatNe
            | OpCode::FloatGt
            | OpCode::FloatGe => {
                if let (Some(a_loc), Some(b_loc)) = (arglocs.first(), arglocs.get(1)) {
                    let b_reg = match b_loc {
                        Loc::Reg(b) => Some(*b),
                        _ => None,
                    };
                    let a = if let Loc::Reg(a) = a_loc {
                        *a
                    } else {
                        let scratch = if b_reg.map_or(false, |b| b.value == 15 && b.is_xmm) {
                            crate::regloc::RegLoc::new(14, true)
                        } else {
                            crate::regloc::RegLoc::new(15, true)
                        };
                        self.regalloc_mov(a_loc, &Loc::Reg(scratch));
                        scratch
                    };
                    let b = if let Loc::Reg(b) = b_loc {
                        *b
                    } else {
                        let scratch = if a.value == 15 && a.is_xmm {
                            crate::regloc::RegLoc::new(14, true)
                        } else {
                            crate::regloc::RegLoc::new(15, true)
                        };
                        self.regalloc_mov(b_loc, &Loc::Reg(scratch));
                        scratch
                    };
                    dynasm!(self.mc ; .arch aarch64 ; fcmp D(a.value), D(b.value));
                    if let Some(Loc::Reg(r)) = result_loc {
                        let cc = Self::float_opcode_to_cc(op.opcode);
                        self.emit_setcc(cc, r.value);
                    }
                }
            }
            // ── Casts ──
            OpCode::CastIntToFloat => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let sr = match src {
                        Loc::Reg(s) => s.value,
                        _ => {
                            self.regalloc_mov(
                                src,
                                &Loc::Reg(crate::regloc::RegLoc::new(16, false)),
                            );
                            16
                        }
                    };
                    dynasm!(self.mc ; .arch aarch64 ; scvtf D(dst.value), X(sr));
                }
            }
            OpCode::CastFloatToInt => {
                if let (Some(Loc::Reg(src)), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    dynasm!(self.mc ; .arch aarch64 ; fcvtzs X(dst.value), D(src.value));
                }
            }
            // ── Same-as / identity ──
            OpCode::SameAsI
            | OpCode::SameAsR
            | OpCode::SameAsF
            | OpCode::CastOpaquePtr
            | OpCode::VirtualRefR
            | OpCode::ConvertFloatBytesToLonglong
            | OpCode::ConvertLonglongBytesToFloat => {
                if let (Some(src), Some(dst)) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, dst);
                }
            }
            // `assembler.py:1545` `genop_load_from_gc_table`: load the
            // reference constant at `gc_table_base + index*WORD`. The
            // index arrives as an immediate (the `ConstInt(index)` arg
            // produced by `remove_constptr`); the table base is baked
            // absolute (x86-32 `MOV_rj` model, `assembler.py:1551-1552`)
            // because dynasm has no code-buffer-start reservation seam.
            // The slot value is GC-forwarded in place by the gc_table
            // root walker, so each load observes the relocated object.
            OpCode::LoadFromGcTable => {
                let (Some(Loc::Immed(idx)), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc)
                else {
                    panic!(
                        "LoadFromGcTable expects [Immed(index)] and a register result, \
                         got arglocs={arglocs:?} result={result_loc:?}"
                    );
                };
                debug_assert_ne!(
                    self.gc_table_base, 0,
                    "LoadFromGcTable emitted without a GcTable base"
                );
                let slot_addr = self.gc_table_base + (idx.value as usize) * WORD;
                self.emit_mov_imm64(dst.value as u32, slot_addr as i64);
                dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [X(dst.value)]);
            }
            // `opassembler.py:269-270 emit_op_cast_ptr_to_int =
            // _genop_same_as` / `emit_op_cast_int_to_ptr = _genop_same_as`.
            // PyPy's aarch64 backend treats both casts as plain `mov` —
            // the AddressAsInt low-bit tag is a `blackhole.py:603-610`
            // interpreter-side invariant, not a backend codegen step.
            // See x86 sibling for the full rationale.
            OpCode::CastPtrToInt | OpCode::CastIntToPtr => {
                if let (Some(src), Some(dst)) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, dst);
                }
            }
            // ── Memory loads: getfield pattern ──
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldGcPureI
            | OpCode::GetfieldGcPureR
            | OpCode::GetfieldGcPureF
            | OpCode::GetfieldRawI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldRawF
            | OpCode::ArraylenGc
            | OpCode::Strlen
            | OpCode::Unicodelen => {
                if let (Some(Loc::Reg(base)), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let ofs = op.with_field_descr(|fd| fd.offset() as i32).unwrap_or(0);
                    let field_size = op.with_field_descr(|fd| fd.field_size()).unwrap_or(8);
                    if dst.is_xmm {
                        dynasm!(self.mc ; .arch aarch64 ; ldr D(dst.value), [X(base.value), ofs as u32]);
                    } else {
                        match field_size {
                            1 => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrb W(dst.value), [X(base.value), ofs as u32]);
                            }
                            2 => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrh W(dst.value), [X(base.value), ofs as u32]);
                            }
                            4 => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsw X(dst.value), [X(base.value), ofs as u32]);
                            }
                            _ => {
                                dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [X(base.value), ofs as u32]);
                            }
                        }
                    }
                }
            }
            // ── Memory loads: getarrayitem pattern ──
            OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF
            | OpCode::GetarrayitemGcPureI
            | OpCode::GetarrayitemGcPureR
            | OpCode::GetarrayitemGcPureF
            | OpCode::GetarrayitemRawI
            | OpCode::GetarrayitemRawR
            | OpCode::GetarrayitemRawF => {
                if let (Some(base_loc), Some(index_loc), Some(Loc::Reg(dst))) =
                    (arglocs.first(), arglocs.get(1), result_loc)
                {
                    let (base_size, item_size, signed) = op
                        .with_array_descr(|ad| {
                            (
                                ad.base_size() as i32,
                                ad.item_size() as i32,
                                op.opcode.result_type() == Type::Int && ad.is_item_signed(),
                            )
                        })
                        .unwrap_or((0, 8, false));
                    let index_reg = match index_loc {
                        Loc::Reg(r) => r.value,
                        _ => {
                            self.regalloc_mov(
                                index_loc,
                                &Loc::Reg(crate::regloc::RegLoc::new(16, false)),
                            );
                            16
                        }
                    };
                    if item_size != 1 {
                        self.emit_mov_imm64(17, item_size as i64);
                        dynasm!(self.mc ; .arch aarch64 ; mul x16, X(index_reg), x17);
                    } else if index_reg != 16 {
                        dynasm!(self.mc ; .arch aarch64 ; mov x16, X(index_reg));
                    }
                    if base_size != 0 {
                        dynasm!(self.mc ; .arch aarch64 ; add x16, x16, base_size as u32);
                    }
                    // Resolve the base into a register. A Loc::Immed base
                    // (constant array pointer from the green-pc inline dispatch
                    // reading `program[const]`) previously failed the Loc::Reg
                    // pattern, so the address computation and load were skipped
                    // and dst kept a stale value (mirrors the GcLoad
                    // immediate-base handling below). x17 is dead here: the
                    // item_size multiply above is its only other use.
                    let base_reg = match base_loc {
                        Loc::Reg(r) => r.value,
                        _ => {
                            self.regalloc_mov(
                                base_loc,
                                &Loc::Reg(crate::regloc::RegLoc::new(17, false)),
                            );
                            17
                        }
                    };
                    dynasm!(self.mc ; .arch aarch64 ; add x16, X(base_reg), x16);
                    if dst.is_xmm {
                        dynasm!(self.mc ; .arch aarch64 ; ldr D(dst.value), [x16]);
                    } else {
                        match item_size {
                            1 if signed => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsb X(dst.value), [x16])
                            }
                            1 => dynasm!(self.mc ; .arch aarch64 ; ldrb W(dst.value), [x16]),
                            2 if signed => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsh X(dst.value), [x16])
                            }
                            2 => dynasm!(self.mc ; .arch aarch64 ; ldrh W(dst.value), [x16]),
                            4 if signed => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsw X(dst.value), [x16])
                            }
                            4 => dynasm!(self.mc ; .arch aarch64 ; ldr W(dst.value), [x16]),
                            _ => dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [x16]),
                        }
                    }
                }
            }
            // ── Memory stores: setarrayitem pattern ──
            OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                if let (Some(Loc::Reg(base)), Some(index_loc), Some(value_loc)) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let (base_size, item_size) = op
                        .with_array_descr(|ad| (ad.base_size() as i32, ad.item_size() as i32))
                        .unwrap_or((0, 8));
                    let index_reg = 16u8;
                    let value_reg = 17u8;

                    self.regalloc_mov(
                        index_loc,
                        &Loc::Reg(crate::regloc::RegLoc::new(index_reg, false)),
                    );
                    if item_size != 1 {
                        self.emit_mov_imm64(value_reg.into(), item_size as i64);
                        dynasm!(self.mc ; .arch aarch64 ; mul X(index_reg), X(index_reg), X(value_reg));
                    }
                    if base_size != 0 {
                        self.emit_mov_imm64(value_reg.into(), base_size as i64);
                        dynasm!(self.mc ; .arch aarch64 ; add X(index_reg), X(index_reg), X(value_reg));
                    }
                    dynasm!(self.mc ; .arch aarch64 ; add X(index_reg), X(base.value), X(index_reg));

                    match value_loc {
                        Loc::Reg(val) if val.is_xmm => {
                            dynasm!(self.mc ; .arch aarch64 ; str D(val.value), [X(index_reg)]);
                        }
                        Loc::Reg(val) => match item_size {
                            1 => {
                                dynasm!(self.mc ; .arch aarch64 ; strb W(val.value), [X(index_reg)])
                            }
                            2 => {
                                dynasm!(self.mc ; .arch aarch64 ; strh W(val.value), [X(index_reg)])
                            }
                            4 => {
                                dynasm!(self.mc ; .arch aarch64 ; str W(val.value), [X(index_reg)])
                            }
                            _ => {
                                dynasm!(self.mc ; .arch aarch64 ; str X(val.value), [X(index_reg)])
                            }
                        },
                        _ => {
                            self.regalloc_mov(
                                value_loc,
                                &Loc::Reg(crate::regloc::RegLoc::new(value_reg, false)),
                            );
                            match item_size {
                                1 => {
                                    dynasm!(self.mc ; .arch aarch64 ; strb W(value_reg), [X(index_reg)])
                                }
                                2 => {
                                    dynasm!(self.mc ; .arch aarch64 ; strh W(value_reg), [X(index_reg)])
                                }
                                4 => {
                                    dynasm!(self.mc ; .arch aarch64 ; str W(value_reg), [X(index_reg)])
                                }
                                _ => {
                                    dynasm!(self.mc ; .arch aarch64 ; str X(value_reg), [X(index_reg)])
                                }
                            }
                        }
                    }
                }
            }
            // ── Memory stores: opassembler.rs emit_op_setfield_regalloc ──
            OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                if let (Some(Loc::Reg(base)), Some(val_loc)) = (arglocs.first(), arglocs.get(1)) {
                    let ofs = op.with_field_descr(|fd| fd.offset() as i32).unwrap_or(0);
                    let field_size = op.with_field_descr(|fd| fd.field_size()).unwrap_or(8);
                    self.emit_op_setfield_regalloc(base, val_loc, ofs, field_size);
                } else {
                    self.genop_discard_setfield(op);
                }
            }
            // ── aarch64/opassembler.py:370 _emit_op_gc_load ──
            // arglocs = [base_loc, ofs_loc, res_loc, imm(nsize)]
            OpCode::GcLoadI
            | OpCode::GcLoadR
            | OpCode::GcLoadF
            | OpCode::RawLoadI
            | OpCode::RawLoadF => {
                if let Some(ofs_loc) = arglocs.get(1) {
                    // aarch64/opassembler.py:371-374
                    let dst = match arglocs.get(2) {
                        Some(Loc::Reg(r)) => r,
                        _ => match result_loc {
                            Some(Loc::Reg(r)) => r,
                            _ => return,
                        },
                    };
                    let nsize = match arglocs.get(3) {
                        Some(Loc::Immed(i)) => i.value,
                        _ => op
                            .with_array_descr(|ad| {
                                let s = ad.item_size() as i64;
                                if ad.is_item_signed() { -s } else { s }
                            })
                            .unwrap_or(8),
                    };
                    match arglocs.first() {
                        Some(Loc::Reg(base)) => {
                            self.emit_op_gcload_regalloc(base, ofs_loc, dst, nsize);
                        }
                        Some(Loc::Immed(base_i)) => {
                            self.emit_mov_imm64(16, base_i.value);
                            let base = RegLoc {
                                value: 16,
                                is_xmm: false,
                            };
                            self.emit_op_gcload_regalloc(&base, ofs_loc, dst, nsize);
                        }
                        _ => {}
                    }
                }
            }
            // ── aarch64/opassembler.py:365 emit_op_gc_store ──
            // `value_loc, base_loc, ofs_loc, size_loc = arglocs`.
            // regalloc.rs:3042 `consider_gc_store` (mirroring RPython
            // aarch64/regalloc.py:520 prepare_op_gc_store) always emits a
            // 4-tuple `[Reg|Immed(value), Reg(base), Reg|Immed(ofs),
            // Immed(size)]`; value_loc may be Immed when the source is a
            // Const (llsupport/regalloc.py:625 `return_constant`), and
            // the emitter moves it to x16 before the store. Everything
            // else is tight and panics now so regalloc regressions can't
            // hide behind silent defaults.
            OpCode::GcStore | OpCode::RawStore => {
                let (value_loc, base_loc, ofs_loc, size_loc) = match arglocs {
                    [v, b, o, s] => (v, b, o, s),
                    _ => panic!(
                        "GcStore arglocs must be [value, base, ofs, size] (got {} locs)",
                        arglocs.len(),
                    ),
                };
                let val_reg = match value_loc {
                    Loc::Reg(r) => *r,
                    Loc::Immed(i) => {
                        self.emit_mov_imm64(16, i.value);
                        crate::regloc::RegLoc::new(16, false)
                    }
                    other => {
                        panic!("GcStore value_loc must be Loc::Reg or Loc::Immed, got {other:?}",)
                    }
                };
                let base = match base_loc {
                    Loc::Reg(r) => r,
                    other => panic!(
                        "GcStore base_loc must be Loc::Reg (regalloc contract), got {other:?}",
                    ),
                };
                // aarch64/opassembler.py:367: scale = get_scale(size_loc.value)
                let size = match size_loc {
                    Loc::Immed(i) => i.value.unsigned_abs() as usize,
                    other => panic!(
                        "GcStore size_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                if crate::majit_log_enabled() {
                    if let Loc::Immed(i) = ofs_loc {
                        let input0_ofs = Self::slot_offset(JITFRAME_FIXED_SIZE);
                        let input1_ofs = Self::slot_offset(JITFRAME_FIXED_SIZE + 1);
                        if i.value as i32 == input0_ofs || i.value as i32 == input1_ofs {
                            eprintln!(
                                "[dynasm][gcstore-input] ofs={} value_loc={:?} base_loc={:?} size={}",
                                i.value, value_loc, base_loc, size
                            );
                        }
                    }
                }
                // aarch64/opassembler.py:368: self._write_to_mem(value_loc, base_loc, ofs_loc, scale)
                self.emit_op_gcstore_regalloc(base, ofs_loc, &val_reg, size);
            }
            // ── aarch64/opassembler.py:396-412 _emit_op_gc_load_indexed ──
            // arglocs = [res_loc, base_loc, index_loc, imm(nsize), imm(ofs)]
            // per `consider_gc_load_indexed` in regalloc.rs:2981-3014.
            OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
                if let (Some(Loc::Reg(res)), Some(Loc::Reg(base)), Some(Loc::Reg(index))) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let nsize = match arglocs.get(3) {
                        Some(Loc::Immed(i)) => i.value,
                        _ => op
                            .with_array_descr(|ad| {
                                let s = ad.item_size() as i64;
                                if ad.is_item_signed() { -s } else { s }
                            })
                            .unwrap_or(8),
                    };
                    // aarch64/opassembler.py:402 keeps `ofs_loc.value` at
                    // the Signed word width; `_try_use_older_box` (rewrite.
                    // py:383-387) may fold large displacements into the
                    // offset immediate so the emitter handles the full
                    // range via the check_imm_arg / materialize fallback.
                    let ofs = match arglocs.get(4) {
                        Some(Loc::Immed(i)) => i.value,
                        _ => 0,
                    };
                    self.emit_op_gcload_indexed_regalloc(base, index, res, ofs, nsize);
                }
            }
            // ── aarch64/opassembler.py:381 emit_op_gc_store_indexed ──
            // arglocs = [value_loc, base_loc, index_loc, imm(size), imm(ofs)]
            OpCode::GcStoreIndexed => {
                if let (Some(value_loc), Some(Loc::Reg(base)), Some(Loc::Reg(index))) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let size = match arglocs.get(3) {
                        Some(Loc::Immed(i)) => i.value.unsigned_abs() as usize,
                        _ => 8,
                    };
                    // aarch64/opassembler.py:385-392 keeps `ofs_loc.value`
                    // at the Signed word width; large displacements folded
                    // in via `_try_use_older_box` must not be truncated.
                    let ofs = match arglocs.get(4) {
                        Some(Loc::Immed(i)) => i.value,
                        _ => 0,
                    };
                    // aarch64/opassembler.py:385-392: combine ofs into ip0 = index + ofs.
                    // check_imm_arg() in regalloc.py:161 allows 0..4095; for any
                    // value outside that (including negatives) RPython falls back
                    // to load(ip0, ofs_loc) + ADD_rr. ip0 = x16 (reserved scratch).
                    let combined_index = if ofs != 0 {
                        if (0..4096).contains(&ofs) {
                            dynasm!(self.mc ; .arch aarch64
                                ; mov x16, X(index.value)
                                ; add x16, x16, ofs as u32);
                        } else {
                            self.emit_mov_imm64(16, ofs);
                            dynasm!(self.mc ; .arch aarch64
                                ; add x16, x16, X(index.value));
                        }
                        crate::regloc::RegLoc::new(16, false)
                    } else {
                        *index
                    };
                    let val_reg = match value_loc {
                        Loc::Reg(r) => *r,
                        Loc::Immed(i) => {
                            self.emit_mov_imm64(17, i.value);
                            crate::regloc::RegLoc::new(17, false)
                        }
                        _ => crate::regloc::RegLoc::new(17, false),
                    };
                    // aarch64/opassembler.py:393-394:
                    //   scale = get_scale(size_loc.value)
                    //   self._write_to_mem(value_loc, base_loc, index_loc, scale)
                    self.emit_op_gcstore_regalloc(base, &Loc::Reg(combined_index), &val_reg, size);
                }
            }
            // ── Control flow ──
            OpCode::Jump => {
                let descr_arc = op.getdescr();
                let jump_descr = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr());
                let target_arglocs = jump_descr
                    .map(|descr| {
                        descr
                            .target_arglocs()
                            .into_iter()
                            .map(|loc| loc_from_target_argloc(&loc))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[dynasm] JUMP remap: src={:?} dst={:?}",
                        arglocs, target_arglocs
                    );
                }
                let mut src_locations1 = Vec::new();
                let mut dst_locations1 = Vec::new();
                let mut src_locations2 = Vec::new();
                let mut dst_locations2 = Vec::new();
                // x86/regalloc.py:1287: assert len(arglocs) == jump_op.numargs()
                let remap_count = if target_arglocs.is_empty() {
                    arglocs.len()
                } else {
                    assert_eq!(
                        arglocs.len(),
                        target_arglocs.len(),
                        "JUMP args ({}) must match target LABEL args ({})",
                        arglocs.len(),
                        target_arglocs.len(),
                    );
                    target_arglocs.len()
                };
                for (i, src_loc) in arglocs[..remap_count].iter().enumerate() {
                    let dst_loc = if i < target_arglocs.len() {
                        target_arglocs[i]
                    } else {
                        let dst_ofs = crate::regalloc::get_ebp_ofs(0, i);
                        Loc::Frame(crate::regloc::FrameLoc::new(i, dst_ofs, false))
                    };
                    let arg_tp = op
                        .getarglist()
                        .get(i)
                        .and_then(|arg| self.opref_type_at(arg.to_opref(), Some(op_index)))
                        .unwrap_or(Type::Int);
                    if arg_tp == Type::Float {
                        src_locations2.push(*src_loc);
                        dst_locations2.push(dst_loc);
                    } else {
                        src_locations1.push(*src_loc);
                        dst_locations1.push(dst_loc);
                    }
                }
                let tmpreg1 = Loc::Reg(crate::regloc::RegLoc::new(16, false));
                let tmpreg2 = Loc::Reg(crate::regloc::RegLoc::new(15, true));
                self.remap_frame_layout_mixed(
                    &src_locations1,
                    &dst_locations1,
                    tmpreg1,
                    &src_locations2,
                    &dst_locations2,
                    tmpreg2,
                );
                // assembler.py:2456-2462 closing_jump
                if let Some(label) = loop_target_id(op)
                    .and_then(|k| self.target_tokens_currently_compiling.get(&k).copied())
                {
                    dynasm!(self.mc ; .arch aarch64 ; b =>label);
                } else if let Some(target) = jump_descr.map(|descr| descr.ll_loop_code()) {
                    // External JUMP: direct branch to target loop code.
                    // assembler.py:2461 mc.JMP(imm(target))
                    // Use x16 (IP0 scratch) to avoid clobbering remap'd regs.
                    // assembler.py:1167-1171 `_assemble`: record the target
                    // loop's frame depth so this trace's frame grows to fit it.
                    if let Some(descr) = jump_descr {
                        self.jump_target_frame_depth =
                            self.jump_target_frame_depth.max(descr.target_frame_depth());
                    }
                    self.emit_mov_imm64(16, target as i64);
                    dynasm!(self.mc ; .arch aarch64 ; br x16);
                }
            }
            OpCode::Finish => {
                // RPython: genop_finish stores result at jf_frame[0] (base_ofs),
                // writes descr ptr to jf_descr, then calls _call_footer.
                // arglocs[0] = result location (if any)
                let fail_arg_types = self.infer_fail_arg_types(op, Some(op_index));
                let result_type = if fail_arg_types.is_empty() {
                    Type::Void
                } else {
                    fail_arg_types[0]
                };
                // `compile.py:658` ExitFrameWithExceptionDescrRef identity:
                // route to the metainterp `exit_frame_with_exception_descr_ref`
                // when the FINISH was emitted for
                // `pyjitpl.py:3238 compile_exit_frame_with_exception`.
                let is_exit_exc = op
                    .with_fail_descr(|fd| fd.is_exit_frame_with_exception())
                    .unwrap_or(false);
                let global_descr_ptr = if is_exit_exc {
                    self.exit_frame_with_exception_descr_ref_ptr()
                } else {
                    self.done_with_this_frame_descr_ptr_for_type(result_type)
                };
                // FINISH op exit (DoneWithThisFrame* / ExitFrameWithExceptionDescr).
                // `compile.py:185` skips these — not a `ResumeDescr`.
                // Singleton-direct push: machine code bakes the singleton
                // ptr into jf_descr, and runner.rs::find_descr_by_ptr
                // short-circuits FINISH/Exit/Propagate to the cpu-attached
                // singleton before consulting the registry (see x86
                // counterpart for full rationale).
                let descr: majit_ir::DescrRef = if is_exit_exc {
                    self.cpu_handle
                        .read()
                        .unwrap()
                        .exit_frame_with_exception_descr_ref
                        .clone()
                } else {
                    self.done_with_this_frame_descr_arc_for_type(result_type)
                }
                .expect(
                    "FINISH emission requires cpu-attached singleton — \
                     call `attach_default_test_descrs` or use `MetaInterp::new`",
                );

                // Store result to jf_frame[0]
                if let Some(result) = arglocs.first() {
                    let slot0 = Loc::Frame(crate::regloc::FrameLoc::new(
                        0,
                        crate::jitframe::FIRST_ITEM_OFFSET as i32,
                        result_type == Type::Float,
                    ));
                    self.regalloc_mov(result, &slot0);
                }

                // Store descr ptr to jf_descr.
                self.emit_mov_imm64(0, global_descr_ptr);
                dynasm!(self.mc ; .arch aarch64 ; str x0, [x29, JF_DESCR_OFS as u32]);

                if result_type == Type::Ref {
                    if let Some(gcmap) = self.finish_gcmap {
                        gcmap_set_bit(gcmap, 0);
                        self.push_gcmap(gcmap);
                    } else {
                        self.push_gcmap(self.gcmap_for_finish);
                    }
                } else if let Some(gcmap) = self.finish_gcmap {
                    self.push_gcmap(gcmap);
                } else {
                    self.pop_gcmap();
                }

                self._call_footer();
                // Singleton: jf_descr bakes the cpu-attached `global_descr_ptr`,
                // not the cell pointer.  `handle_fail_done_with_this_frame`
                // and `handle_fail_exit_frame_with_exception` match by
                // ptr-equality on the singleton, so the cell only carries
                // the keep-alive identity for `clt.asmmemmgr_gcreftracers`.
                self.fail_descrs
                    .push(majit_ir::FailDescrCell::wrap(descr.clone()));
            }
            OpCode::Label => {
                let label = self.mc.new_dynamic_label();
                let descr_arc = op.getdescr();
                let label_descr = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr());
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[dynasm] LABEL: new DynamicLabel({:?}) arglocs={:?}",
                        label, arglocs
                    );
                }
                dynasm!(self.mc ; =>label);
                if let Some(descr) = label_descr {
                    let stored_arglocs = arglocs
                        .iter()
                        .copied()
                        .map(target_argloc_from_loc)
                        .collect::<Vec<_>>();
                    if majit_ir::debug::have_debug_prints() {
                        majit_ir::debug::log_one(
                            "jit-backend",
                            &format!("LABEL target_arglocs={stored_arglocs:?}"),
                        );
                    }
                    descr.set_target_arglocs(stored_arglocs);
                    descr.set_ll_loop_code(self.mc.offset().0);
                    if let Some(id) = loop_target_id(op) {
                        self.target_tokens_currently_compiling.insert(id, label);
                    }
                    if let Some(descr_ref) = op.getdescr() {
                        self.compiled_target_tokens.push(descr_ref.clone());
                    }
                }
            }
            // ── Calls ──
            // RPython: regalloc consider_call does before_call (save caller-saved
            // regs), collects arglocs, calls after_call for result. The assembler
            // receives arglocs = [func_addr_or_descr_info..., arg_locs...] and
            // result_loc = register for return value.
            //
            // For now, flush all register-resident values to their frame slots
            // before the call, use the existing frame-slot genop_call, then
            // mark the result in the allocated register.
            OpCode::CallI
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureF
            | OpCode::CallPureN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN
            | OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilF
            | OpCode::CallReleaseGilN => {
                self.genop_call_with_arglocs(op, arglocs);
            }
            OpCode::CallR => {
                let is_nursery_alloc = op.with_call_descr(|cd| cd.get_extra_info().pyre_helper)
                    == Some(majit_ir::PyreHelperKind::NurseryAlloc);
                if is_nursery_alloc {
                    self.genop_nursery_alloc_inline(op, arglocs);
                } else {
                    self.genop_call_with_arglocs(op, arglocs);
                }
            }
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                // assembler.py:2207 _store_force_index parity:
                // store next GUARD_NOT_FORCED's descr ptr to jf_force_descr
                // BEFORE the call, so forcing code knows which guard to resume.
                self._store_force_index_if_next_guard(ops, op_index, fail_index);
                self.genop_call_assembler(op, arglocs);
            }
            OpCode::CondCallN => self.genop_discard_cond_call(op, arglocs),
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                self.genop_cond_call_value(op);
            }
            // ── Allocation (raw, when GC rewriter is not active) ──
            OpCode::New => self.genop_new(op),
            OpCode::NewWithVtable => self.genop_new_with_vtable(op),
            OpCode::NewArray | OpCode::NewArrayClear => self.genop_new_array(op),
            OpCode::Newstr => self.genop_newstr(op),
            OpCode::Newunicode => self.genop_newunicode(op),
            // ── Allocation (rewritten by GC rewriter) ──
            // aarch64/regalloc.py:958 + assembler.py:682 malloc_cond parity
            OpCode::CallMallocNursery => {
                self.genop_call_malloc_nursery(op);
                if let Some(Loc::Reg(r)) = result_loc {
                    if r.value != 0 {
                        let rv = r.value;
                        dynasm!(self.mc ; .arch aarch64 ; mov X(rv), x0);
                    }
                }
                if !op.pos.get().is_none() {
                    self.store_rax_to_result(op.pos.get());
                }
            }
            // aarch64/assembler.py:715 malloc_cond_varsize_frame
            OpCode::CallMallocNurseryVarsizeFrame => {
                let sizeloc = match arglocs.first() {
                    Some(Loc::Reg(sizeloc)) => *sizeloc,
                    _ => panic!("CallMallocNurseryVarsizeFrame expects register size arg"),
                };
                let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();
                let slow_path = self.mc.new_dynamic_label();
                let done = self.mc.new_dynamic_label();

                if nf_addr == 0 || nt_addr == 0 {
                    if sizeloc.value != 0 {
                        dynasm!(self.mc ; .arch aarch64 ; mov x0, X(sizeloc.value));
                    }
                    dynasm!(self.mc ; .arch aarch64 ; b =>slow_path);
                } else {
                    let size_reg = if sizeloc == crate::aarch64::registers::X0 {
                        dynasm!(self.mc ; .arch aarch64 ; mov x1, x0);
                        crate::aarch64::registers::X1
                    } else {
                        sizeloc
                    };
                    let gc_header_size = majit_gc::header::GcHeader::SIZE as u32;
                    self.emit_mov_imm64(0, nf_addr as i64);
                    dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x0]);
                    dynasm!(self.mc ; .arch aarch64 ; add x1, x0, X(size_reg.value));
                    dynasm!(self.mc ; .arch aarch64 ; add x1, x1, gc_header_size);
                    self.emit_mov_imm64(16, nt_addr as i64);
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x16, [x16]
                        ; cmp x1, x16
                        ; b.hi =>slow_path
                    );
                    self.emit_mov_imm64(16, nf_addr as i64);
                    dynasm!(self.mc ; .arch aarch64
                        ; str x1, [x16]
                        ; str xzr, [x0]
                        ; add x0, x0, gc_header_size
                        ; b =>done
                    );
                }

                dynasm!(self.mc ; .arch aarch64 ; =>slow_path);
                self.push_all_regs_to_jitframe(
                    &[crate::aarch64::registers::X0, crate::aarch64::registers::X1],
                    true,
                );
                // aarch64/assembler.py:716-734 malloc_cond_varsize_frame:
                // compute size into X0 BEFORE storing gcmap, because the
                // gcmap store must NOT clobber the slowpath's size arg.
                // Upstream uses IP1 for the gcmap pointer (gen_load_int_full
                // r.ip1.value, ...) so the size arg in X0 stays intact.
                if nf_addr != 0 && nt_addr != 0 {
                    dynasm!(self.mc ; .arch aarch64 ; sub x0, x1, x0);
                    let gc_header_size = majit_gc::header::GcHeader::SIZE as u32;
                    dynasm!(self.mc ; .arch aarch64 ; sub x0, x0, gc_header_size);
                } else if sizeloc.value != 0 {
                    dynasm!(self.mc ; .arch aarch64 ; mov x0, X(sizeloc.value));
                }
                if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
                    // Store gcmap into jf[jf_gcmap_ofs] via a scratch reg
                    // that is NOT X0 (size arg) or X1 (original nursery_free).
                    // Using x16 (IP0) matches _reload_frame_if_necessary and
                    // stays out of the argument register path.
                    self.emit_mov_imm64(16, gcmap as i64);
                    dynasm!(self.mc ; .arch aarch64
                        ; str x16, [x29, crate::jitframe::JF_GCMAP_OFS as u32]
                    );
                } else {
                    let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
                    dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                }
                self.emit_mov_imm64(
                    16,
                    crate::runner::dynasm_nursery_slowpath_jitframe as *const () as i64,
                );
                dynasm!(self.mc ; .arch aarch64 ; blr x16);
                self.reload_frame_if_necessary();
                let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
                dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                self.pop_all_regs_from_jitframe(
                    &[crate::aarch64::registers::X0, crate::aarch64::registers::X1],
                    true,
                );
                // `dynasm_nursery_slowpath_jitframe` falls back to
                // `libc::calloc`, which returns NULL on real host OOM.
                // Route through the propagate path before the fast/slow
                // paths join so the null frame pointer never reaches the
                // CALL_ASSEMBLER store sequence that follows.
                self.emit_propagate_memory_error_if_null(0);
                dynasm!(self.mc ; .arch aarch64 ; =>done);
                if !op.pos.get().is_none() {
                    self.store_rax_to_result(op.pos.get());
                }
            }
            // aarch64/assembler.py:738 malloc_cond_varsize
            // arglocs = [lengthloc, imm(itemsize), imm(kind)]
            OpCode::CallMallocNurseryVarsize => {
                let base_size = op.with_array_descr(|ad| ad.base_size()).unwrap_or(16) as i64;
                let itemsize = match arglocs.get(1) {
                    Some(Loc::Immed(i)) => i.value,
                    _ => 8,
                };
                // _build_malloc_slowpath(kind='var') parity:
                // x0 = base_size, x1 = item_size, x2 = length
                self.emit_mov_imm64(0, base_size);
                self.emit_mov_imm64(1, itemsize);
                match arglocs.first() {
                    Some(Loc::Reg(len_r)) => {
                        dynasm!(self.mc ; .arch aarch64 ; mov x2, X(len_r.value));
                    }
                    Some(Loc::Immed(len_i)) => {
                        self.emit_mov_imm64(2, len_i.value);
                    }
                    _ => {
                        self.emit_mov_imm64(2, 0);
                    }
                }
                // push_gcmap
                let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
                dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                self.emit_mov_imm64(
                    3,
                    crate::runner::dynasm_nursery_slowpath_varsize as *const () as i64,
                );
                self.emit_malloc_slowpath_helper_call(3);
                self.reload_frame_if_necessary();
                // pop_gcmap
                dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                // `dynasm_nursery_slowpath_varsize` returns x0 = 0 on
                // real host OOM (calloc failure preserved as NULL per
                // runner.rs).  Route through the propagate path before
                // the result store so the null pointer never reaches
                // subsequent typed stores.
                self.emit_propagate_memory_error_if_null(0);
                if !op.pos.get().is_none() {
                    self.store_rax_to_result(op.pos.get());
                }
            }
            // aarch64/opassembler.py:258 `emit_op_check_memory_error` →
            // assembler.py:342-346 `propagate_memoryerror_if_reg_is_null`
            // — emit `CBNZ reg, skip` and inline the propagate path
            // (`_build_propagate_exception_path`, assembler.py:559-577)
            // so a NULL return from a malloc helper raises a MemoryError
            // via `self.cpu.propagate_exception_descr`.
            //
            // Upstream materializes `propagate_exception_path` once per
            // backend instance and per-CHECK_MEMORY_ERROR branches to
            // it.  Pyre's dynasm doesn't have that out-of-line trampoline
            // infrastructure yet, so the path is inlined per occurrence
            // — equivalent semantics, slightly more code per site.
            // CHECK_MEMORY_ERROR is rare (only after the four CALL_R
            // malloc helpers in `gen_call_malloc_gc`), so the size
            // overhead is negligible.
            //
            // Sequence per assembler.py:559-577:
            //   1. _store_and_reset_exception(mc, r.x0)
            //      — load pos_exc_value → x0, clear pos_exc_value and
            //      pos_exception (assembler.py:501-536).
            //   2. str x0, [fp, jf_guard_exc]
            //      — transfer the saved value into the deadframe so
            //      `cpu.grab_exc_value(deadframe)` can read it back in
            //      `PropagateExceptionDescr.handle_fail` (compile.py:1095).
            //   3. gen_load_int x0, propagate_exception_descr
            //   4. str x0, [fp, jf_descr]
            //   5. mov x0, fp + gen_func_epilog
            OpCode::CheckMemoryError => {
                let reg = match arglocs.first() {
                    Some(Loc::Reg(r)) if !r.is_xmm => r.value,
                    _ => panic!("CheckMemoryError arglocs[0] must be a non-xmm register"),
                };
                self.emit_propagate_memory_error_if_null(reg);
            }
            // aarch64/opassembler.py:912 _write_barrier_fastpath parity.
            OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                self.emit_write_barrier_fastpath(op, &arglocs);
            }
            // ── Misc ──
            OpCode::ForceToken => {
                if let Some(Loc::Reg(r)) = result_loc {
                    dynasm!(self.mc ; .arch aarch64 ; mov X(r.value), x29);
                }
            }
            OpCode::SaveException => self.genop_save_exception(op),
            OpCode::SaveExcClass => self.genop_save_exc_class(op),
            // Guards never reach the non-guard regalloc dispatch — they
            // are emitted exclusively from `regalloc_perform_guard` via
            // the `RegAllocOp::PerformWithGuard` arm.
            _ if op.opcode.is_guard() => unreachable!(
                "regalloc_perform reached with guard {:?}; guards must \
                 route through regalloc_perform_guard",
                op.opcode
            ),
            // ── No-ops ──
            _ => {}
        }
    }

    /// assembler.py:329 regalloc_perform_guard — emit guard with faillocs.
    fn regalloc_perform_guard(
        &mut self,
        op: &Op,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<&Loc>,
        faillocs: &[Option<Loc>],
        fail_index: u32,
    ) {
        match op.opcode {
            OpCode::GuardTrue | OpCode::VecGuardTrue | OpCode::GuardNonnull => {
                // arglocs[0] = condition location
                if let Some(loc) = arglocs.first() {
                    self.emit_test_loc(loc);
                    self.guard_success_cc = Some(CC_NE);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardFalse | OpCode::VecGuardFalse | OpCode::GuardIsnull => {
                if let Some(loc) = arglocs.first() {
                    self.emit_test_loc(loc);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardValue => {
                if arglocs.len() >= 2 {
                    self.emit_cmp_loc_loc(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardClass => {
                if arglocs.len() >= 2 {
                    self._cmp_guard_class(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardGcType => {
                if arglocs.len() >= 2 {
                    self._cmp_guard_gc_type(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardIsObject => {
                if let Some(obj_loc) = arglocs.first() {
                    self.emit_guard_is_object(obj_loc);
                    self.guard_success_cc = Some(CC_NE);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardSubclass => {
                if arglocs.len() >= 2 {
                    self.emit_guard_subclass(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_B);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardException => {
                if let Some(expected_loc) = arglocs.first() {
                    self.emit_guard_exception(expected_loc);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
                self.emit_store_and_reset_exception(result_loc);
            }
            OpCode::GuardNonnullClass => {
                if arglocs.len() >= 2 {
                    self.emit_test_loc(&arglocs[0]);
                    let fail_label = self.emit_guard_jcc(CC_E);
                    self._cmp_guard_class(&arglocs[0], &arglocs[1]);
                    self.emit_jcc_to_label(CC_NE, fail_label);
                    self.append_guard_token_with_faillocs(
                        op, op_index, fail_index, fail_label, faillocs,
                    );
                }
            }
            OpCode::GuardNoException => {
                self.emit_guard_no_exception_check();
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardNoOverflow | OpCode::GuardOverflow => {
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
                dynasm!(self.mc ; .arch aarch64 ; ldr X(16), [x29, JF_DESCR_OFS as u32] ; cmp X(16), xzr);
                self.guard_success_cc = Some(CC_E);
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardNotInvalidated => {
                self.implement_guard_nojump_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardEvalBreaker => {
                // Back-edge eval-breaker poll: load the async-action ticker
                // cell and deopt when it is negative (a pending signal /
                // async action). `0` = no ticker published (signal handling
                // not installed) → inert guard, no runtime check.
                let ticker_addr = majit_ir::eval_breaker::ticker_addr();
                if ticker_addr == 0 {
                    self.implement_guard_nojump_with_faillocs(op, op_index, fail_index, faillocs);
                } else {
                    self.emit_mov_imm64(16, ticker_addr as i64);
                    dynasm!(self.mc ; .arch aarch64 ; ldr X(16), [x16] ; cmp X(16), xzr);
                    self.guard_success_cc = Some(CC_GE);
                    self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
                }
            }
            OpCode::GuardAlwaysFails => {
                self.implement_guard_always_fails_with_faillocs(op, op_index, fail_index, faillocs);
            }
            _ => {
                self.implement_guard_nojump_with_faillocs(op, op_index, fail_index, faillocs);
            }
        }
    }

    /// Helper: guard class comparison
    fn _cmp_guard_class(&mut self, obj_loc: &Loc, class_loc: &Loc) {
        if let Loc::Reg(obj) = obj_loc {
            if let Some(vtable_offset) = self.vtable_offset {
                let ofs = vtable_offset as u32;
                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [X(obj.value), ofs]);
                self.regalloc_mov(class_loc, &Loc::Reg(crate::regloc::RegLoc::new(17, false)));
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
            } else if let Loc::Immed(i) = class_loc {
                let expected_typeid = self
                    .lookup_typeid_from_classptr(i.value as usize)
                    .expect("GuardClass: missing typeid for classptr");
                self._cmp_guard_gc_type(
                    &Loc::Reg(*obj),
                    &Loc::Immed(crate::regloc::ImmedLoc::new(expected_typeid as i64)),
                );
            }
        }
    }

    fn require_guard_gc_type_info(&self, guard_name: &'static str) -> GuardGcTypeInfo {
        self.guard_gc_type_info.unwrap_or_else(|| {
            panic!(
                "{} requires cpu.supports_guard_gc_type and a TYPE_INFO layout",
                guard_name
            )
        })
    }

    fn lookup_subclass_range(&self, classptr: usize) -> Option<(i64, i64)> {
        self.classptr_to_subclass_range
            .get(&(classptr as i64))
            .copied()
    }

    fn emit_load_gc_typeid_into_x16(&mut self, obj_reg: u8) {
        let tid_ofs = -(majit_gc::header::GcHeader::SIZE as i64);
        self.emit_mov_imm64(16, tid_ofs);
        dynasm!(self.mc ; .arch aarch64
            ; add x16, X(obj_reg), x16
            ; ldr w16, [x16]
        );
    }

    /// aarch64 analogue of x86/assembler.py:1893-1901
    /// `_cmp_guard_gc_type`, adjusted for majit's `obj - GcHeader::SIZE`
    /// header layout.
    fn _cmp_guard_gc_type(&mut self, obj_loc: &Loc, expected_typeid_loc: &Loc) {
        let Loc::Reg(obj) = obj_loc else {
            return;
        };
        self.emit_load_gc_typeid_into_x16(obj.value);
        match expected_typeid_loc {
            Loc::Reg(expected) => {
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, X(expected.value));
            }
            Loc::Frame(frame) => {
                self.emit_ldr_fp(17, frame.ebp_loc.value);
                dynasm!(self.mc ; .arch aarch64
                    ; cmp x16, x17
                );
            }
            Loc::Immed(expected) => {
                self.emit_mov_imm64(17, expected.value);
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
            }
            _ => {}
        }
    }

    /// aarch64/opassembler.py:650-663 `emit_op_guard_is_object`.
    fn emit_guard_is_object(&mut self, obj_loc: &Loc) {
        let info = self.require_guard_gc_type_info("GUARD_IS_OBJECT");
        let Loc::Reg(obj) = obj_loc else {
            return;
        };
        self.emit_load_gc_typeid_into_x16(obj.value);
        if info.shift_by > 0 {
            dynasm!(self.mc ; .arch aarch64 ; lsl x16, x16, info.shift_by as u32);
        }
        self.emit_mov_imm64(17, (info.base_type_info + info.infobits_offset) as i64);
        let flag = info.is_object_flag as u32;
        dynasm!(self.mc ; .arch aarch64
            ; add x16, x16, x17
            ; ldrb w16, [x16]
            ; mov w17, flag
            ; tst w16, w17
        );
    }

    /// aarch64/opassembler.py:665-706 `emit_op_guard_subclass`.
    fn emit_guard_subclass(&mut self, obj_loc: &Loc, class_loc: &Loc) {
        let info = self.require_guard_gc_type_info("GUARD_SUBCLASS");
        let (Loc::Reg(obj), Loc::Immed(classptr)) = (obj_loc, class_loc) else {
            panic!(
                "GUARD_SUBCLASS expects [Reg object, Immed classptr] \
                 like aarch64/opassembler.py:667"
            );
        };
        let (check_min, check_max) = self
            .lookup_subclass_range(classptr.value as usize)
            .unwrap_or_else(|| {
                panic!(
                    "GUARD_SUBCLASS missing subclassrange_min/max for classptr {:#x}",
                    classptr.value
                )
            });
        if let Some(vtable_offset) = self.vtable_offset {
            let offset = vtable_offset as u32;
            let offset2 = info.subclassrange_min_offset as u32;
            dynasm!(self.mc ; .arch aarch64
                ; ldr x16, [X(obj.value), offset]
                ; ldr x16, [x16, offset2]
            );
        } else {
            self.emit_load_gc_typeid_into_x16(obj.value);
            if info.shift_by > 0 {
                dynasm!(self.mc ; .arch aarch64 ; lsl x16, x16, info.shift_by as u32);
            }
            self.emit_mov_imm64(
                17,
                (info.base_type_info + info.sizeof_ti + info.subclassrange_min_offset) as i64,
            );
            dynasm!(self.mc ; .arch aarch64
                ; add x16, x16, x17
                ; ldr x16, [x16]
            );
        }
        self.emit_mov_imm64(17, check_min);
        dynasm!(self.mc ; .arch aarch64 ; sub x16, x16, x17);
        self.emit_mov_imm64(17, check_max - check_min);
        dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
    }

    /// aarch64/opassembler.py:711-718 `emit_op_guard_exception`.
    fn emit_guard_exception(&mut self, expected_loc: &Loc) {
        self.emit_mov_imm64(16, crate::jit_exc_type_addr() as i64);
        dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x16]);
        match expected_loc {
            Loc::Reg(expected) => {
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, X(expected.value));
            }
            Loc::Frame(frame) => {
                self.emit_ldr_fp(17, frame.ebp_loc.value);
                dynasm!(self.mc ; .arch aarch64
                    ; cmp x16, x17
                );
            }
            Loc::Immed(expected) => {
                self.emit_mov_imm64(17, expected.value);
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
            }
            _ => {}
        }
    }

    /// `_store_and_reset_exception`: result = pos_exc_value; clear both
    /// pos_exception and pos_exc_value on the success fallthrough.
    fn emit_store_and_reset_exception(&mut self, result_loc: Option<&Loc>) {
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        self.emit_mov_imm64(16, exc_value_addr);
        if let Some(loc) = result_loc {
            match loc {
                Loc::Reg(dst) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [x16]);
                }
                Loc::Frame(frame) => {
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x17, [x16]
                    );
                    self.emit_str_fp(17, frame.ebp_loc.value);
                }
                _ => {}
            }
        }
        dynasm!(self.mc ; .arch aarch64 ; str xzr, [x16]);
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64 ; str xzr, [x16]);
    }

    /// aarch64/opassembler.py:720-724 `emit_op_guard_no_exception`.
    fn emit_cmp_no_exception(&mut self) {
        self.emit_mov_imm64(16, crate::jit_exc_type_addr() as i64);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]
            ; cmp x16, xzr
        );
    }

    /// Emit SETcc into a register (zero-extend to 64-bit).
    fn emit_setcc(&mut self, cc: u8, dst_reg: u8) {
        match cc {
            CC_E => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), eq);
            }
            CC_NE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ne);
            }
            CC_L => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), lt);
            }
            CC_GE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ge);
            }
            CC_LE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), le);
            }
            CC_G => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), gt);
            }
            CC_B => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), lo);
            }
            CC_AE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), hs);
            }
            CC_BE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ls);
            }
            CC_A => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), hi);
            }
            CC_S => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), mi);
            }
            CC_NS => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), pl);
            }
            CC_O => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), vs);
            }
            CC_NO => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), vc);
            }
            _ => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), eq);
            }
        }
    }

    /// Map an integer comparison OpCode to a condition code.
    fn opcode_to_cc(opcode: OpCode) -> u8 {
        match opcode {
            OpCode::IntLt => CC_L,
            OpCode::IntLe => CC_LE,
            OpCode::IntGt => CC_G,
            OpCode::IntGe => CC_GE,
            OpCode::IntEq | OpCode::PtrEq | OpCode::InstancePtrEq => CC_E,
            OpCode::IntNe | OpCode::PtrNe | OpCode::InstancePtrNe => CC_NE,
            OpCode::UintLt => CC_B,
            OpCode::UintLe => CC_BE,
            OpCode::UintGt => CC_A,
            OpCode::UintGe => CC_AE,
            _ => CC_E,
        }
    }

    /// Map a float comparison OpCode to a condition code (after ucomisd).
    fn float_opcode_to_cc(opcode: OpCode) -> u8 {
        match opcode {
            OpCode::FloatLt => CC_B,  // ucomisd: below = less than
            OpCode::FloatLe => CC_BE, // below or equal
            OpCode::FloatGt => CC_A,  // above
            OpCode::FloatGe => CC_AE, // above or equal
            OpCode::FloatEq => CC_E,  // equal
            OpCode::FloatNe => CC_NE, // not equal
            _ => CC_E,
        }
    }

    /// Guard with faillocs — emit conditional jump and store faillocs on descr.
    fn implement_guard_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        faillocs: &[Option<Loc>],
    ) {
        let cc = self
            .guard_success_cc
            .take()
            .expect("implement_guard_with_faillocs: guard_success_cc not set");
        let fail_cc = invert_cc(cc);
        let fail_label = self.emit_guard_jcc(fail_cc);
        self.append_guard_token_with_faillocs(op, op_index, fail_index, fail_label, faillocs);
    }

    /// Guard no-jump with faillocs.
    fn implement_guard_nojump_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        faillocs: &[Option<Loc>],
    ) {
        let fail_label = self.mc.new_dynamic_label();
        self.append_guard_token_with_faillocs(op, op_index, fail_index, fail_label, faillocs);
    }

    fn implement_guard_always_fails_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        faillocs: &[Option<Loc>],
    ) {
        let fail_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; b =>fail_label);
        self.append_guard_token_with_faillocs(op, op_index, fail_index, fail_label, faillocs);
    }

    /// Append guard token with regalloc faillocs instead of opref_to_slot snapshot.
    fn append_guard_token_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        fail_label: DynamicLabel,
        faillocs: &[Option<Loc>],
    ) {
        let fail_arg_types = self.infer_fail_arg_types(op, Some(op_index));
        // assembler.py:2207 _store_force_index parity:
        // If a CALL_ASSEMBLER already pre-allocated this guard's descr
        // (stored in pending_force_descr), reuse it — same Arc, same ptr
        // that was written to jf_force_descr.
        // Stamp the per-trace fail_index and trace_id onto the metainterp
        // ResumeGuardDescr (`op.descr`).  See x86 counterpart for rationale:
        // routing the writes through the trait at codegen time lets readers
        // consume the canonical metainterp identity before
        // `build_guard_metadata` (`compile.rs:232`) re-stamps.
        if let Some(d) = op.getdescr() {
            if d.is_resume_guard() || d.is_resume_guard_copied() {
                if let Some(fd) = d.as_fail_descr() {
                    fd.set_fail_index_per_trace(fail_index);
                    fd.set_trace_id(self.trace_id);
                }
            }
        }
        let descr: majit_ir::DescrRef = if let Some(pre) = self.pending_force_descr.take() {
            pre
        } else if let Some(d) = op.getdescr() {
            // Guard exit — `compile.py:185` ResumeGuardDescr family.
            // Use the metainterp `AbstractFailDescr` Arc from `op.descr`
            // directly; per-trace fail_index / trace_id were stamped
            // above.  Refresh the descr's `fail_arg_types` slot when the
            // inferred list disagrees: `store_final_boxes_in_guard` is
            // supposed to keep it in sync, but test scaffolds and earlier
            // optimizer paths can leave it empty.  The GC map below
            // (`guard_gcmap_from_faillocs(descr_fd.fail_arg_types(), ...)`)
            // reads back through the descr, so a stale empty list would
            // under-report `Type::Ref` slots and miss live roots.
            if let Some(fd) = d.as_fail_descr() {
                if fd.fail_arg_types() != fail_arg_types.as_slice() {
                    fd.set_fail_arg_types(fail_arg_types);
                }
            }
            d
        } else {
            // Test scaffold: tests synthesise guard ops without op.descr.
            // Mint a fresh metainterp ResumeGuardDescr to carry the
            // codegen-time identity (fail_index / trace_id / fail_arg_types).
            let fresh = majit_backend::make_resume_guard_descr_typed(fail_arg_types);
            if let Some(fd) = fresh.as_fail_descr() {
                fd.set_fail_index_per_trace(fail_index);
                fd.set_trace_id(self.trace_id);
            }
            fresh
        };
        let descr_fd = descr.as_fail_descr().expect("guard descr is FailDescr");
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] guard-token: fail_index={} op_index={} opcode={:?} fail_args={:?} fail_arg_types={:?} faillocs={:?}",
                fail_index,
                op_index,
                op.opcode,
                op.getfailargs(),
                descr_fd.fail_arg_types(),
                faillocs
            );
        }

        // `llsupport/assembler.py:248-276 store_info_on_descr` parity.
        // PyPy encodes each fail-arg location as a USHORT in `rd_locs`;
        // pyre allocates a const-store slot for `Loc::Immed` and writes
        // the slot into rd_locs so the deopt path reads it via PyPy's
        // stack-position decode (`llmodel.py:422-424`).
        let mut const_stores: Vec<(usize, i64)> = Vec::new();
        let gpr_regs = all_gen_regs();
        let float_regs = all_float_regs();
        let rd_locs: Vec<u16> = faillocs
            .iter()
            .map(|fl| match fl {
                None => 0xFFFF,
                Some(Loc::Frame(f)) => (f.position + JITFRAME_FIXED_SIZE) as u16,
                Some(Loc::Reg(r)) if r.is_xmm => {
                    (gpr_regs.len()
                        + float_regs
                            .iter()
                            .position(|reg| *reg == *r)
                            .expect("rd_locs: float register not in float_regs"))
                        as u16
                }
                Some(Loc::Reg(r)) => gpr_regs
                    .iter()
                    .position(|reg| *reg == *r)
                    .expect("rd_locs: register not in gen_regs")
                    as u16,
                Some(Loc::Immed(i)) => {
                    let slot = self.frame_depth;
                    self.frame_depth += 1;
                    const_stores.push((slot, i.value));
                    slot as u16
                }
                Some(Loc::Ebp(_)) | Some(Loc::Addr(_)) => 0xFFFF,
            })
            .collect();
        // Stamp source_op_index directly on the meta descr (UnsafeCell slot
        // owned by ResumeGuardDescr / ResumeGuardCopiedDescr per
        // resume_guard_descr.rs:166); `layout_for_fail_descr` reads it back
        // via `fd.source_op_index()` so no side-table is needed.
        // Recovery layouts are owned by the metainterp's
        // `StoredExitLayout.recovery_layout` (populated by
        // `patch_guard_recovery_layouts_for_trace`) per
        // `resume.py:450-488`; the backend keeps no parallel cache.
        if descr_fd.is_resume_guard() || descr_fd.is_resume_guard_copied() {
            descr_fd.set_source_op_index(op_index);
        }
        // `llsupport/assembler.py:279 guardtok.faildescr.rd_locs = positions`
        // — write through the trait accessor so the metainterp
        // `AbstractFailDescr` (`history.py:132 _attrs_`) receives the
        // canonical copy when present.  Must follow the `meta_descr`
        // stamp above for the forward to reach the meta side.
        descr_fd.set_rd_locs(rd_locs);
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] guard-token-slots: fail_index={} rd_locs={:?}",
                fail_index,
                descr_fd.rd_locs()
            );
        }
        let gcmap = self.guard_gcmap_from_faillocs(descr_fd.fail_arg_types(), faillocs);

        // Reuse the cell pre-allocated by `_store_force_index_if_next_guard`
        // when this guard is paired with a CALL_ASSEMBLER's force-store —
        // jf_force_descr and jf_descr then resolve to the same cell, and
        // `fail_descrs[fail_index]` carries exactly one entry per guard.
        let cell = self
            .pending_force_cell
            .take()
            .unwrap_or_else(|| majit_ir::FailDescrCell::wrap(descr.clone()));
        self.pending_guard_tokens.push(GuardToken {
            fail_label,
            fail_descr: cell.clone(),
            const_stores,
            gcmap,
            must_save_exception: matches!(
                op.opcode,
                OpCode::GuardException | OpCode::GuardNoException | OpCode::GuardNotForced
            ),
        });
        if op.opcode == OpCode::GuardNotForced2 {
            self.finish_gcmap = Some(gcmap);
        }
        self.fail_descrs.push(cell);
    }

    // ----------------------------------------------------------------
    // assembler.py:652 write_pending_failure_recoveries
    // ----------------------------------------------------------------

    /// assembler.py:982 generate_quick_failure.
    ///
    /// RPython parity: the quick-failure stub saves managed registers into the
    /// fixed jitframe prefix before publishing jf_descr and returning.
    fn generate_quick_failure(
        &mut self,
        guard_token: GuardToken,
        save_regs_label: DynamicLabel,
    ) -> (std::sync::Arc<majit_ir::FailDescrCell>, usize) {
        let stub_start = self.mc.offset();

        let fail_label = guard_token.fail_label;
        if majit_ir::debug::have_debug_prints() {
            majit_ir::debug::log_one(
                "jit-backend",
                &format!("recovery stub: binding {fail_label:?}"),
            );
        }
        dynasm!(self.mc ; .arch aarch64 ; =>fail_label);

        dynasm!(self.mc ; .arch aarch64 ; bl =>save_regs_label);

        // llsupport/assembler.py:236 store_info_on_descr — must_save_exception
        // guards run the exc=True failure-recovery variant: stage pos_exc_value
        // into jf_guard_exc and clear both globals so grab_exc_value reads the
        // value off the deadframe (assembler.py:316-329 _build_failure_recovery).
        // x16/x17 are scratch (ip0/ip1); all managed regs were saved by the
        // bl above, so clobbering them here is safe.
        if guard_token.must_save_exception {
            let exc_value_addr = crate::jit_exc_value_addr() as i64;
            let exc_type_addr = crate::jit_exc_type_addr() as i64;
            self.emit_mov_imm64(16, exc_value_addr);
            dynasm!(self.mc ; .arch aarch64
                ; ldr x17, [x16]                            // x17 = *pos_exc_value
                ; str x17, [x29, JF_GUARD_EXC_OFS as u32]   // jf_guard_exc = excval
                ; str xzr, [x16]                            // *pos_exc_value = 0
            );
            self.emit_mov_imm64(16, exc_type_addr);
            dynasm!(self.mc ; .arch aarch64
                ; str xzr, [x16]                            // *pos_exception = 0
            );
        }

        let descr_ptr = Arc::as_ptr(&guard_token.fail_descr) as *const () as i64;
        self.emit_mov_imm64(0, descr_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str x0, [x29, JF_DESCR_OFS as u32]
        );
        self.push_gcmap(guard_token.gcmap);

        for &(slot, val) in &guard_token.const_stores {
            let ofs = Self::slot_offset(slot);
            self.emit_mov_imm64(16, val);
            self.emit_str_fp(16, ofs);
        }

        self._call_footer();
        (guard_token.fail_descr, stub_start.0)
    }

    /// assembler.py:1005 write_pending_failure_recoveries.
    /// Returns recovery stub offsets for post-finalize address fixup.
    fn write_pending_failure_recoveries(
        &mut self,
    ) -> Vec<(std::sync::Arc<majit_ir::FailDescrCell>, usize)> {
        // Emit a shared _push_all_regs_to_frame routine once, then let each
        // generate_quick_failure() stub call it.
        let save_regs_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; =>save_regs_label);
        let gprs = all_gen_regs();
        for &reg in gprs.iter() {
            let save_slot = core_reg_position(reg).expect("managed aarch64 GPR");
            let ofs = Self::slot_offset(save_slot) as u32;
            dynasm!(self.mc ; .arch aarch64 ; str X(reg.value), [x29, ofs]);
        }
        let fprs = all_float_regs();
        for &reg in fprs.iter() {
            let save_slot = float_reg_position(reg).expect("managed aarch64 VFP");
            let ofs = Self::slot_offset(save_slot) as u32;
            dynasm!(self.mc ; .arch aarch64 ; str D(reg.value), [x29, ofs]);
        }
        dynasm!(self.mc ; .arch aarch64 ; ret);

        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] write_pending_failure_recoveries: {} tokens",
                self.pending_guard_tokens.len()
            );
        }
        let mut stub_offsets = Vec::new();
        for guard_token in std::mem::take(&mut self.pending_guard_tokens) {
            stub_offsets.push(self.generate_quick_failure(guard_token, save_regs_label));
        }
        if majit_ir::debug::have_debug_prints() {
            majit_ir::debug::log_one(
                "jit-backend",
                &format!("write_pending done: {} stubs", stub_offsets.len()),
            );
        }
        stub_offsets
    }

    /// assembler.py:849 patch_pending_failure_recoveries — convert
    /// buffer-relative offsets to absolute addresses after finalize.
    fn patch_pending_failure_recoveries(
        rawstart: usize,
        stub_offsets: &[(std::sync::Arc<majit_ir::FailDescrCell>, usize)],
    ) {
        for (cell, stub_offset) in stub_offsets {
            let abs_addr = rawstart + stub_offset;
            if let Some(fd) = cell.as_fail_descr() {
                fd.set_adr_jump_offset(abs_addr);
            }
        }
    }

    // ----------------------------------------------------------------
    // assembler.py:965-987 patch_jump_for_descr
    // ----------------------------------------------------------------

    /// Overwrite the code at `at` with a branch to `target`.
    ///
    /// `link` selects the branch form to match the upstream method this
    /// stands in for: assembler.py patch_trace overwrites a guard stub with
    /// `BL` (branch-with-link → bridge), while redirect_call_assembler
    /// overwrites a loop entry with a plain `B`. A direct `B`/`BL imm26`
    /// reaches ±128 MB; once the code arena spans more than that, a bridge /
    /// retraced loop can land farther from the originating guard stub / loop
    /// entry, so the scaled 26-bit displacement would silently truncate and
    /// the branch would land on garbage. When the displacement does not fit,
    /// fall back to materializing the absolute target in ip0 and branching
    /// through it (`BR`/`BLR x16`) — the indirect form codebuilder.py emits
    /// for an out-of-range `B`/`BL`. ip0/x16 is call-clobbered scratch at
    /// every redirect site (the bridge prologue's `_check_frame_depth`
    /// reloads it before use).
    ///
    /// # Safety
    /// `at` must point to at least 20 writable bytes of now-dead code:
    /// both callers overwrite the head of a recovery stub / loop prologue
    /// that is longer than the emitted sequence.
    unsafe fn write_redirect_branch(at: usize, target: usize, link: bool) {
        let offset = target as isize - at as isize;
        assert!(
            at & 0b11 == 0 && target & 0b11 == 0,
            "AArch64 redirect branch endpoints must be 4-byte aligned: at={at:#x}, target={target:#x}"
        );
        // B/BL imm26: signed 26-bit, scaled by 4 → ±128 MB reach.
        const B_REACH: isize = 1 << 27;
        if (-B_REACH..B_REACH).contains(&offset) {
            let imm26 = ((offset >> 2) & 0x03FF_FFFF) as u32;
            // 0x14000000 = B imm26, 0x94000000 = BL imm26.
            let opc = if link { 0x9400_0000 } else { 0x1400_0000 };
            let insn = opc | imm26;
            codebuf::with_writable(at as *mut u8, 4, || {
                unsafe { (at as *mut u32).write(insn) };
            });
            flush_icache(at as *const u8, 4);
        } else {
            // MOVZ/MOVK x16, target; BR/BLR x16 — 5 words. The four MOV words
            // come from the shared encoder so the veneer stays byte-identical
            // to emit_mov_imm64 (pinned by frame_depth_patch_words_match_emit_mov_imm64).
            let mov = Self::encode_mov_imm64_words(16, target as i64);
            // 0xD61F0000 = BR Xn, 0xD63F0000 = BLR Xn (Rn in bits 9:5).
            let br = if link { 0xD63F_0000 } else { 0xD61F_0000 };
            let words: [u32; 5] = [mov[0], mov[1], mov[2], mov[3], br | (16 << 5)];
            codebuf::with_writable(at as *mut u8, words.len() * 4, || {
                let p = at as *mut u32;
                for (i, w) in words.iter().enumerate() {
                    unsafe { p.add(i).write(*w) };
                }
            });
            flush_icache(at as *const u8, words.len() * 4);
        }
    }

    /// assembler.py:965 patch_jump_for_descr: redirect a guard to a
    /// bridge by overwriting the recovery stub with a JMP to bridge.
    ///
    /// `adr_jump_offset` is the absolute address of the recovery stub
    /// (set by patch_pending_failure_recoveries). We overwrite the
    /// stub with "MOV r11, bridge_addr; JMP r11" (x64) or "BL imm26"
    /// (aarch64), matching rpython/jit/backend/aarch64/assembler.py
    /// patch_trace().
    pub fn patch_jump_for_descr(descr: &dyn majit_ir::FailDescr, adr_new_target: usize) {
        let stub_addr = descr.adr_jump_offset();
        assert!(stub_addr != 0, "guard already patched");

        // patch_trace uses BL (branch-with-link) into the bridge.
        unsafe { Self::write_redirect_branch(stub_addr, adr_new_target, true) };

        // Verify patch was applied correctly
        if crate::majit_log_enabled() {
            let word = unsafe { (stub_addr as *const u32).read() };
            eprintln!(
                "[patch-verify] stub_addr={:#x} first_word={:#010x} target={:#x}",
                stub_addr, word, adr_new_target
            );
        }

        // assembler.py:987
        descr.set_adr_jump_offset(0); // "patched"
    }

    /// assembler.py:1138 redirect_call_assembler: patch old loop entry
    /// to JMP to new loop after retrace.
    pub fn redirect_call_assembler(old_addr: *const u8, new_addr: *const u8) {
        // redirect_call_assembler uses a plain B (no link) to the new loop.
        unsafe { Self::write_redirect_branch(old_addr as usize, new_addr as usize, false) };
    }

    // ----------------------------------------------------------------
    // genop_* — integer arithmetic
    // ----------------------------------------------------------------

    /// INT_ADD: result = arg0 + arg1
    fn genop_int_add(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, x1
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_SUB: result = arg0 - arg1
    fn genop_int_sub(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; sub x0, x0, x1
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_MUL: result = arg0 * arg1
    fn genop_int_mul(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; mul x0, x0, x1
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_AND: result = arg0 & arg1
    fn genop_int_and(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; and x0, x0, x1
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_OR: result = arg0 | arg1
    fn genop_int_or(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; orr x0, x0, x1
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_XOR: result = arg0 ^ arg1
    fn genop_int_xor(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; eor x0, x0, x1
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_NEG: result = -arg0
    fn genop_int_neg(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; neg x0, x0
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_INVERT: result = ~arg0
    fn genop_int_invert(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; mvn x0, x0
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_LSHIFT: result = arg0 << arg1
    fn genop_int_lshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; lsl x0, x0, x1
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_RSHIFT: result = arg0 >> arg1 (arithmetic/signed)
    fn genop_int_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; asr x0, x0, x1
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// UINT_RSHIFT: result = arg0 >> arg1 (logical/unsigned)
    fn genop_uint_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; lsr x0, x0, x1
        );
        self.store_rax_to_result(op.pos.get());
    }

    // ----------------------------------------------------------------
    // genop_* — overflow arithmetic (assembler.py:1413-1425)
    // ----------------------------------------------------------------

    /// assembler.py:1856 genop_int_add_ovf — delegates to genop_int_add,
    /// then sets guard_success_cc = 'NO'. On x86, ADD always sets OF.
    fn genop_int_add_ovf(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; adds x0, x0, x1);
        self.store_rax_to_result(op.pos.get());
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py:1860 genop_int_sub_ovf.
    fn genop_int_sub_ovf(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; subs x0, x0, x1);
        self.store_rax_to_result(op.pos.get());
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py:1864 genop_int_mul_ovf.
    fn genop_int_mul_ovf(&mut self, op: &Op) {
        // aarch64/opassembler.py multiplies, computes SMULH, and then
        // compares the high word against the sign-extension of the low
        // word. regalloc.py's prepare_op_guard_no_overflow then uses
        // EQ for INT_MUL_OVF specifically.
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; mul x2, x0, x1
            ; smulh x3, x0, x1
            ; asr x4, x2, 63
            ; cmp x3, x4
            ; mov x0, x2
        );
        self.store_rax_to_result(op.pos.get());
        self.guard_success_cc = Some(CC_E);
    }

    // ----------------------------------------------------------------
    // genop_* — comparisons
    // ----------------------------------------------------------------

    /// INT_LT/LE/GT/GE/EQ/NE/UINT_*: CMP arg0, arg1 then store CC.
    /// If the next op is a guard, guard_success_cc is set and consumed.
    /// Otherwise, materialize the boolean result via SETcc/CSET.
    fn genop_int_cmp(&mut self, op: &Op) {
        let cc = Self::opcode_to_cc(op.opcode);

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; cmp x0, x1
        );

        // Store the CC for a following guard to consume.
        self.guard_success_cc = Some(cc);

        // Also materialize the boolean result for non-guard consumers.
        if !op.pos.get().is_none() {
            self.emit_setcc_to_result(cc, op.pos.get());
        }
    }

    /// Emit SETcc/CSET to materialize a boolean result.
    /// x64: SETcc AL; MOVZX EAX, AL
    /// aarch64: CSET X0, cc
    fn emit_setcc_to_result(&mut self, cc: u8, result_opref: OpRef) {
        // CSET Xd, cc — sets Xd to 1 if condition is true, 0 otherwise.
        // Note: CSET Xd, cc is an alias for CSINC Xd, XZR, XZR, invert(cc).
        match cc {
            CC_L => dynasm!(self.mc ; .arch aarch64 ; cset x0, lt),
            CC_LE => dynasm!(self.mc ; .arch aarch64 ; cset x0, le),
            CC_G => dynasm!(self.mc ; .arch aarch64 ; cset x0, gt),
            CC_GE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ge),
            CC_E => dynasm!(self.mc ; .arch aarch64 ; cset x0, eq),
            CC_NE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ne),
            CC_B => dynasm!(self.mc ; .arch aarch64 ; cset x0, lo),
            CC_BE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ls),
            CC_A => dynasm!(self.mc ; .arch aarch64 ; cset x0, hi),
            CC_AE => dynasm!(self.mc ; .arch aarch64 ; cset x0, hs),
            CC_O => dynasm!(self.mc ; .arch aarch64 ; cset x0, vs),
            CC_NO => dynasm!(self.mc ; .arch aarch64 ; cset x0, vc),
            _ => dynasm!(self.mc ; .arch aarch64 ; cset x0, eq),
        }
        self.store_rax_to_result(result_opref);
    }

    /// INT_IS_TRUE: result = (arg0 != 0)
    fn genop_int_is_true(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; cmp x0, 0
        );
        self.guard_success_cc = Some(CC_NE);
        if !op.pos.get().is_none() {
            self.emit_setcc_to_result(CC_NE, op.pos.get());
        }
    }

    /// INT_IS_ZERO: result = (arg0 == 0)
    fn genop_int_is_zero(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; cmp x0, 0
        );
        self.guard_success_cc = Some(CC_E);
        if !op.pos.get().is_none() {
            self.emit_setcc_to_result(CC_E, op.pos.get());
        }
    }

    // ----------------------------------------------------------------
    // genop_* — guards
    // ----------------------------------------------------------------

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Looks up the materialized table populated by the runner from
    /// the active gc_ll_descr. RPython resolves the same value via
    /// `cpu.gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr`.
    fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        self.classptr_to_typeid.get(&(classptr as i64)).copied()
    }

    fn emit_guard_jcc(&mut self, fail_cc: u8) -> DynamicLabel {
        let fail_label = self.mc.new_dynamic_label();
        self.emit_bcond_to_label(fail_cc, fail_label);
        fail_label
    }

    /// Emit a conditional branch to `label` using the long-branch pattern:
    /// `b.<inv_cc> skip; b =>label; skip:` so that the displacement of the
    /// unconditional `b` is 26-bit / ±128MB instead of the 19-bit / ±1MB
    /// of `b.cond`. The extra inversion+skip adds one instruction per
    /// guard, but avoids `ImpossibleRelocation` on large traces (logo's
    /// 70000-op trace generates >1MB of machine code).
    fn emit_bcond_to_label(&mut self, cc: u8, label: DynamicLabel) {
        let skip = self.mc.new_dynamic_label();
        // Invert: branch over the unconditional `b` when the guard succeeds
        match cc {
            CC_L => dynasm!(self.mc ; .arch aarch64 ; b.ge =>skip),
            CC_LE => dynasm!(self.mc ; .arch aarch64 ; b.gt =>skip),
            CC_G => dynasm!(self.mc ; .arch aarch64 ; b.le =>skip),
            CC_GE => dynasm!(self.mc ; .arch aarch64 ; b.lt =>skip),
            CC_E => dynasm!(self.mc ; .arch aarch64 ; b.ne =>skip),
            CC_NE => dynasm!(self.mc ; .arch aarch64 ; b.eq =>skip),
            CC_B => dynasm!(self.mc ; .arch aarch64 ; b.hs =>skip),
            CC_BE => dynasm!(self.mc ; .arch aarch64 ; b.hi =>skip),
            CC_A => dynasm!(self.mc ; .arch aarch64 ; b.ls =>skip),
            CC_AE => dynasm!(self.mc ; .arch aarch64 ; b.lo =>skip),
            CC_O => dynasm!(self.mc ; .arch aarch64 ; b.vc =>skip),
            CC_NO => dynasm!(self.mc ; .arch aarch64 ; b.vs =>skip),
            CC_S => dynasm!(self.mc ; .arch aarch64 ; b.pl =>skip),
            CC_NS => dynasm!(self.mc ; .arch aarch64 ; b.mi =>skip),
            _ => dynasm!(self.mc ; .arch aarch64 ; b.ne =>skip),
        }
        // Unconditional branch to the failure target (26-bit, ±128MB)
        dynasm!(self.mc ; .arch aarch64 ; b =>label);
        dynasm!(self.mc ; .arch aarch64 ; =>skip);
    }

    fn emit_jcc_to_label(&mut self, fail_cc: u8, fail_label: DynamicLabel) {
        self.emit_bcond_to_label(fail_cc, fail_label);
    }

    /// Infer fail_arg_types from `op.type_` (via `opref_type`) or
    /// `op.fail_arg_types`.
    fn infer_fail_arg_types(&self, op: &Op, op_index: Option<usize>) -> Vec<Type> {
        if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
            if let Some(descr_types) = op.with_fail_descr(|fd| fd.fail_arg_types().to_vec()) {
                if !descr_types.is_empty() {
                    return descr_types;
                }
            }
        }
        let descr_arc = op.getdescr();
        if let Some(fd) = descr_arc.as_ref().and_then(|d| d.as_fail_descr()) {
            // op.descr is a ResumeGuardDescr carrying post-numbering
            // fail_arg_types installed by
            // store_final_boxes_in_guard (optimizeopt/mod.rs:3393-3404).
            // Prefer the descr for guards too; fall through to
            // op.fail_arg_types only for sharing-path guards
            // (optimizeopt/mod.rs:3068-3088) where op.descr=None.
            let dt = fd.fail_arg_types();
            let expected_len = op.getfailargs().map(|fa| fa.len()).unwrap_or(0);
            if dt.len() == expected_len && !dt.is_empty() {
                return dt.to_vec();
            }
        }
        if let Some(ts) = op.get_fail_arg_types() {
            let expected_len = if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
                op.num_args()
            } else {
                op.getfailargs().map(|fa| fa.len()).unwrap_or(0)
            };
            if ts.len() == expected_len {
                ts.to_vec()
            } else if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
                op.getarglist()
                    .iter()
                    .map(|opref| {
                        self.opref_type_at(opref.to_opref(), op_index)
                            .unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                 op_index={:?} (Finish/Jump arg): RPython box.type is fixed at \
                                 construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                    })
                    .collect()
            } else if let Some(fa) = op.getfailargs() {
                fa.iter()
                    .map(|opref| {
                        if opref.is_none() {
                            // resume.py:411-417 parity: TAGCONST/TAGVIRTUAL
                            // slots are kept as OpRef::NONE in fail_args
                            // (PyPy filters them out; pyre keeps positional).
                            // `Type::Void` is the "hole" sentinel — value
                            // comes from the resume snapshot, not the
                            // deadframe, so downstream consumers
                            // (`gc_ref_slots`, `guard_gcmap_from_faillocs`,
                            // `typed_outputs` reconstruction) must skip
                            // these slots. Earlier code used `Type::Ref`
                            // which silently leaked a NULL `GcRef` into
                            // `gc_ref_slots` and the shadow stack.
                            Type::Void
                        } else {
                            self.opref_type_at(opref.to_opref(), op_index).unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                     op_index={:?} (fail_arg): RPython box.type is fixed at \
                                     construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            }
        } else if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
            // Finish/Jump carry no failargs; their result kind comes from
            // the argument boxes, whose types are fixed at construction
            // (resoperation.py:719/727/739).  When neither a fail descr nor
            // a preset fail_arg_types list supplies them, infer from the
            // arglist so the FINISH's done_with_this_frame_descr kind
            // matches the caller's CALL_ASSEMBLER result kind (a Void
            // mismatch routes every return through the assembler helper
            // instead of the result-loading fast path).
            op.getarglist()
                .iter()
                .map(|opref| {
                    self.opref_type_at(opref.to_opref(), op_index)
                        .unwrap_or_else(|| {
                            panic!(
                                "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                 op_index={:?} (Finish/Jump arg): RPython box.type is fixed at \
                                 construction (resoperation.py:719/727/739)",
                                opref, op_index
                            )
                        })
                })
                .collect()
        } else if let Some(fa) = op.getfailargs() {
            fa.iter()
                .map(|opref| {
                    if opref.is_none() {
                        // resume.py:411-417 parity: see comment above —
                        // Type::Void is the "hole" sentinel.
                        Type::Void
                    } else {
                        self.opref_type_at(opref.to_opref(), op_index)
                            .unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                 op_index={:?} (fail_arg): RPython box.type is fixed at \
                                 construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// assembler.py:1794 generate_guard_no_exception:
    /// `LDR loc, [loc]; CMP loc, #0` with success on zero.
    fn emit_guard_no_exception_check(&mut self) {
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]
            ; cmp x16, xzr
        );
        self.guard_success_cc = Some(CC_E);
    }

    /// assembler.py:2207-2222 _store_force_index: before a call that may force,
    /// store the next GUARD_NOT_FORCED's fail descr ptr to jf_force_descr,
    /// and zero jf_descr so GUARD_NOT_FORCED's CMP [jf_descr], 0 starts clean.
    fn _store_force_index_if_next_guard(&mut self, ops: &[Op], op_idx: usize, fail_index: u32) {
        // assembler.py:2224-2226 _find_nearby_operation(+1)
        let next_idx = op_idx + 1;
        if next_idx >= ops.len() {
            return;
        }
        let next_op = &ops[next_idx];
        if next_op.opcode != OpCode::GuardNotForced && next_op.opcode != OpCode::GuardNotForced2 {
            return;
        }
        // Pre-allocate the fail descr for the next GUARD_NOT_FORCED.
        // The full metadata (faillocs, rd_numb, etc.) will be filled in
        // when the guard is actually emitted in append_guard_token_with_faillocs.
        let fail_arg_types = self.infer_fail_arg_types(next_op, Some(next_idx));
        // Pre-allocated GuardNotForced descr — ResumeGuardDescr family.
        // Stamp the metainterp `AbstractFailDescr` Arc from `next_op.descr`
        // here so `append_guard_token_with_faillocs` does not need a second
        // pass through `unsafe { Arc::as_ptr as *mut }`.
        if let Some(d) = next_op.getdescr() {
            if d.is_resume_guard() || d.is_resume_guard_copied() {
                if let Some(fd) = d.as_fail_descr() {
                    fd.set_fail_index_per_trace(fail_index);
                    fd.set_trace_id(self.trace_id);
                }
            }
        }
        let descr: majit_ir::DescrRef = if let Some(d) = next_op.getdescr() {
            // Same staleness guard as the main guard-emission path: keep
            // descriptor `fail_arg_types` in sync with the inferred list
            // so downstream GC-map / rd_locs readers see the right Ref
            // slots.
            if let Some(fd) = d.as_fail_descr() {
                if fd.fail_arg_types() != fail_arg_types.as_slice() {
                    fd.set_fail_arg_types(fail_arg_types);
                }
            }
            d
        } else {
            let fresh = majit_backend::make_resume_guard_descr_typed(fail_arg_types);
            if let Some(fd) = fresh.as_fail_descr() {
                fd.set_fail_index_per_trace(fail_index);
                fd.set_trace_id(self.trace_id);
            }
            fresh
        };
        // `force_token_to_dead_frame` (cranelift/compiler.rs:2660)
        // recovers `jf_force_descr` via `recover_fail_descr_cell`, which
        // requires a `FailDescrCell` thin pointer.  Bake the cell pointer
        // here (not the bare `Arc<dyn Descr>` fat-pointer data half) and
        // hand the cell off to `append_guard_token_with_faillocs` so the
        // inline guard-exit path bakes the same identity into jf_descr.
        let cell = majit_ir::FailDescrCell::wrap(descr.clone());
        let descr_ptr = Arc::as_ptr(&cell) as *const () as i64;
        self.pending_force_descr = Some(descr);
        self.pending_force_cell = Some(cell);

        // x86/assembler.py:2210-2222: store descr to jf_force_descr,
        // zero jf_descr.
        self.emit_mov_imm64(16, descr_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str X(16), [x29, JF_FORCE_DESCR_OFS as u32]
            ; str xzr, [x29, JF_DESCR_OFS as u32]
        );
    }

    // ----------------------------------------------------------------
    // genop_* — control flow
    // ----------------------------------------------------------------

    /// LABEL: define the back-edge target for JUMP.
    ///
    /// RPython: LABEL does NOT emit code. The regalloc establishes
    /// the slot mapping. JUMP handles slot remapping.
    ///
    /// In our frame-slot model: preamble values may be in non-canonical
    /// slots. We emit copies from old→canonical slots BEFORE the
    /// LABEL binding, so they execute only on first entry from the
    /// preamble. JUMP writes directly to canonical slots and jumps
    /// to the LABEL, skipping the copies.
    fn genop_label(&mut self, op: &Op) {
        // Emit preamble→canonical copies BEFORE the label.
        // Two-pass push/pop: safely handles slot overlaps.
        let n_label = op.num_args();
        // Pass 1: push source values
        for i in 0..n_label {
            let arg_ref = op.arg(i).to_opref();
            if arg_ref.is_none() {
                let dst = Self::slot_offset(i);
                self.emit_ldr_fp(0, dst);
                dynasm!(self.mc ; .arch aarch64 ; str x0, [sp, #-16]!);
            } else if arg_ref.is_constant() {
                let val = arg_ref.inline_const_bits().unwrap_or_else(|| {
                    self.constants
                        .get(&arg_ref.raw())
                        .map(|c| c.as_raw_i64())
                        .unwrap_or(0)
                });
                self.emit_mov_imm64(0, val);
                dynasm!(self.mc ; .arch aarch64 ; str x0, [sp, #-16]!);
            } else if let Some(&old_slot) = self.opref_to_slot.get(&arg_ref) {
                let src = Self::slot_offset(old_slot);
                self.emit_ldr_fp(0, src);
                dynasm!(self.mc ; .arch aarch64 ; str x0, [sp, #-16]!);
            } else {
                dynasm!(self.mc ; .arch aarch64 ; str xzr, [sp, #-16]!);
            }
        }
        // Pass 2: pop in reverse into canonical slots
        for i in (0..n_label).rev() {
            let dst = Self::slot_offset(i);
            dynasm!(self.mc ; .arch aarch64 ; ldr x0, [sp], #16);
            self.emit_str_fp(0, dst);
        }

        // Bind the LABEL — JUMP targets here (after the copies).
        let label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; =>label);
        let descr_arc = op.getdescr();
        if let Some(descr) = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr()) {
            descr.set_ll_loop_code(self.mc.offset().0);
            if let Some(id) = loop_target_id(op) {
                self.target_tokens_currently_compiling.insert(id, label);
            }
            if let Some(descr_ref) = op.getdescr() {
                self.compiled_target_tokens.push(descr_ref.clone());
            }
        }

        // Remap: Label's arg[i] → canonical slot i
        for (i, arg_ref) in op.getarglist().iter().enumerate() {
            if !arg_ref.is_none() {
                self.opref_to_slot.insert(arg_ref.to_opref(), i);
            }
        }
        self.next_slot = self.next_slot.max(op.num_args());
    }

    /// jump.py:66 _move: emit a single slot-to-slot or const-to-slot move.
    fn emit_slot_move(&mut self, src: i32, dst: i32, is_const: bool, val: i64) {
        if is_const {
            self.emit_mov_imm64(0, val);
            self.emit_str_fp(0, dst);
        } else if src != dst {
            self.emit_ldr_fp(0, src);
            self.emit_str_fp(0, dst);
        }
    }

    /// JUMP: unconditional branch to the loop label.
    /// jump.py:1 remap_frame_layout parity: parallel move algorithm
    /// to handle cyclic slot dependencies.
    fn genop_jump(&mut self, op: &Op) {
        // Build src→dst move list.
        // Each entry: (src_offset_or_const, dst_offset, is_const, const_val)
        let n = op.num_args();
        let mut moves: Vec<(i32, i32, bool, i64)> = Vec::with_capacity(n);
        for (i, arg_ref) in op.getarglist().iter().enumerate() {
            let dst = Self::slot_offset(i);
            match self.resolve_opref(arg_ref.to_opref()) {
                ResolvedArg::Slot(src) => moves.push((src, dst, false, 0)),
                ResolvedArg::Const(val) => moves.push((0, dst, true, val)),
            }
        }

        // jump.py:1-64 remap_frame_layout: topological order with
        // cycle breaking via push/pop.
        // srccount[dst] = number of times dst appears as a src
        let mut srccount: IndexMap<i32, i32> = IndexMap::new();
        for m in &moves {
            srccount.entry_or_default(m.1); // ensure dst exists
        }
        let mut pending = n as i32;
        for (i, m) in moves.iter().enumerate() {
            if m.2 {
                continue;
            } // constant → no src dependency
            let src = m.0;
            if let Some(cnt) = srccount.get_mut(&src) {
                if src == moves[i].1 {
                    // self-move: skip
                    *cnt = -(n as i32) - 1;
                    pending -= 1;
                } else {
                    *cnt += 1;
                }
            }
        }

        while pending > 0 {
            let mut progress = false;
            for i in 0..n {
                let dst = moves[i].1;
                if srccount.get(&dst).copied().unwrap_or(-1) == 0 {
                    *srccount.get_mut(&dst).unwrap() = -1; // done
                    pending -= 1;
                    if !moves[i].2 {
                        let src = moves[i].0;
                        if let Some(cnt) = srccount.get_mut(&src) {
                            *cnt -= 1;
                        }
                    }
                    self.emit_slot_move(moves[i].0, dst, moves[i].2, moves[i].3);
                    progress = true;
                }
            }
            if !progress {
                // Cycle: use push/pop to break it.
                for i in 0..n {
                    let dst = moves[i].1;
                    if srccount.get(&dst).copied().unwrap_or(-1) >= 0 {
                        // Push first dst in the cycle
                        self.emit_ldr_fp(0, dst);
                        dynasm!(self.mc ; .arch aarch64
                            ; str x0, [sp, #-16]!
                        );
                        // Walk the cycle
                        let mut cur = i;
                        loop {
                            let cd = moves[cur].1;
                            *srccount.get_mut(&cd).unwrap() = -1;
                            pending -= 1;
                            // Find the move whose dst is this src
                            let src = moves[cur].0;
                            let next = moves.iter().position(|m| m.1 == src);
                            if let Some(ni) = next {
                                if srccount.get(&moves[ni].1).copied().unwrap_or(-1) < 0 {
                                    // End of cycle: pop into this slot
                                    dynasm!(self.mc ; .arch aarch64
                                        ; ldr x0, [sp], #16
                                    );
                                    self.emit_str_fp(0, cd);
                                    break;
                                }
                                self.emit_slot_move(src, cd, false, 0);
                                cur = ni;
                            } else {
                                // No cycle found — emit move and break
                                self.emit_slot_move(moves[cur].0, cd, moves[cur].2, moves[cur].3);
                                break;
                            }
                        }
                    }
                }
            }
        }

        let descr_arc = op.getdescr();
        let jump_descr = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr());
        if let Some(label) =
            loop_target_id(op).and_then(|k| self.target_tokens_currently_compiling.get(&k).copied())
        {
            // Same-buffer jump (loop body)
            dynasm!(self.mc ; .arch aarch64 ; b =>label);
        } else if let Some(target) = jump_descr.map(|descr| descr.ll_loop_code()) {
            // assembler.py closing_jump parity: bridge jumps back to
            // the original loop's LABEL via absolute address.
            // assembler.py:1167-1171 `_assemble`: record the target loop's
            // frame depth so this trace's frame grows to fit it.
            if let Some(descr) = jump_descr {
                self.jump_target_frame_depth =
                    self.jump_target_frame_depth.max(descr.target_frame_depth());
            }
            self.emit_mov_imm64(0, target as i64);
            dynasm!(self.mc ; .arch aarch64 ; br x0);
        }
    }

    /// FINISH: store result (if any), store descr ptr, return jf_ptr.
    fn genop_finish(&mut self, op: &Op, _fail_index: u32) {
        // compiler.rs:9667-9681 parity: trust explicit FINISH types only when
        // they match the actual result arity; otherwise infer from the op args.
        let finish_refs: Vec<OpRef> = op.getarglist().iter().map(|a| a.to_opref()).collect();
        let fail_arg_types = if let Some(explicit) = op.get_fail_arg_types() {
            if explicit.len() == finish_refs.len() {
                explicit.to_vec()
            } else {
                finish_refs
                    .iter()
                    .map(|opref| self.opref_type_at(*opref, None).unwrap_or(Type::Int))
                    .collect()
            }
        } else {
            finish_refs
                .iter()
                .map(|opref| self.opref_type_at(*opref, None).unwrap_or(Type::Int))
                .collect()
        };
        // compile.py:618-669 parity: use type-specific global singleton.
        // FINISH op exit (DoneWithThisFrame*) — `compile.py:185` skips these.
        // Finish ops write the type-appropriate singleton pointer to jf_descr
        // so CALL_ASSEMBLER's fast path CMP matches the correct variant.
        let result_type = if fail_arg_types.is_empty() {
            Type::Void
        } else {
            fail_arg_types[0]
        };
        let global_descr_ptr = self.done_with_this_frame_descr_ptr_for_type(result_type);
        // Singleton-direct push (see OpCode::Finish above for rationale).
        let descr: majit_ir::DescrRef = self
            .done_with_this_frame_descr_arc_for_type(result_type)
            .expect(
                "genop_finish requires cpu-attached singleton — \
                 call `attach_default_test_descrs` or use `MetaInterp::new`",
            );

        // If there's a result argument, store it to jf_frame[0].
        // assembler.py:2291-2303 parity: float results use xmm0/MOVSD.
        if op.num_args() > 0 {
            let arg0 = op.arg(0).to_opref();
            let slot0_offset = Self::slot_offset(0);
            if result_type == Type::Float {
                // Float: load to xmm0, store via MOVSD
                self.load_arg_to_rax(arg0); // loads raw bits
                dynasm!(self.mc ; .arch aarch64
                    ; str x0, [x29, slot0_offset as u32]
                );
            } else {
                self.load_arg_to_rax(arg0);
                dynasm!(self.mc ; .arch aarch64
                    ; str x0, [x29, slot0_offset as u32]
                );
            }
        }

        // Store descr pointer at jf_ptr[0] (jf_descr slot).
        // compile.py:665-674 parity: use global singleton pointer.
        let descr_ptr = global_descr_ptr;
        self.emit_mov_imm64(0, descr_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str x0, [x29, JF_DESCR_OFS as u32]
        );

        if result_type == Type::Ref {
            if let Some(gcmap) = self.finish_gcmap {
                gcmap_set_bit(gcmap, 0);
                self.push_gcmap(gcmap);
            } else {
                self.push_gcmap(self.gcmap_for_finish);
            }
        } else if let Some(gcmap) = self.finish_gcmap {
            self.push_gcmap(gcmap);
        } else {
            self.pop_gcmap();
        }

        // Emit epilogue (return jf_ptr).
        self._call_footer();

        // Singleton: jf_descr bakes the cpu-attached `global_descr_ptr`,
        // not the cell pointer (see OpCode::Finish comment above).
        self.fail_descrs
            .push(majit_ir::FailDescrCell::wrap(descr.clone()));
    }

    // ----------------------------------------------------------------
    // genop_* — type conversions
    // ----------------------------------------------------------------

    /// SAME_AS: result = arg0 (copy value)
    /// SAME_AS: result = arg0 (identity).
    /// regalloc.py parity: no code emitted — just alias the slot.
    fn genop_same_as(&mut self, op: &Op) {
        let arg = op.arg(0).to_opref();
        if let Some(&slot) = self.opref_to_slot.get(&arg) {
            self.opref_to_slot.insert(op.pos.get(), slot);
        } else {
            self.load_arg_to_rax(arg);
            self.store_rax_to_result(op.pos.get());
        }
    }

    // ----------------------------------------------------------------
    // Float helpers
    // ----------------------------------------------------------------

    /// Load a float value from `opref` into XMM0 (x64) / D0 (aarch64).
    /// Float values are stored as bit-cast i64 in frame slots.
    fn load_float_arg_to_d0(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                self.emit_ldr_fp_d(0, offset);
            }
            ResolvedArg::Const(val) => {
                // Load constant via integer register, then move to float register.
                self.emit_mov_imm64(0, val);
                dynasm!(self.mc ; .arch aarch64
                    ; fmov d0, x0
                );
            }
        }
    }

    /// Load a float value from `opref` into XMM1 (x64) / D1 (aarch64).
    fn load_float_arg_to_d1(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                self.emit_ldr_fp_d(1, offset);
            }
            ResolvedArg::Const(val) => {
                self.emit_mov_imm64(1, val);
                dynasm!(self.mc ; .arch aarch64
                    ; fmov d1, x1
                );
            }
        }
    }

    /// Store XMM0 (x64) / D0 (aarch64) to the frame slot for `result_opref`.
    fn store_d0_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        self.emit_str_fp_d(0, offset);
    }

    // ----------------------------------------------------------------
    // genop_* — float arithmetic
    // x86/assembler.py:1648 genop_float_add etc.
    // aarch64/assembler.py float equivalents
    // ----------------------------------------------------------------

    /// FLOAT_ADD: result = arg0 + arg1
    fn genop_float_add(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fadd d0, d0, d1
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// FLOAT_SUB: result = arg0 - arg1
    fn genop_float_sub(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fsub d0, d0, d1
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// FLOAT_MUL: result = arg0 * arg1
    fn genop_float_mul(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fmul d0, d0, d1
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// FLOAT_TRUEDIV: result = arg0 / arg1
    fn genop_float_truediv(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fdiv d0, d0, d1
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// FLOAT_NEG: result = -arg0
    /// x64: XOR with sign-bit mask (0x8000000000000000).
    /// aarch64: FNEG d0, d0.
    fn genop_float_neg(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fneg d0, d0
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// CAST_INT_TO_FLOAT: result = (f64)arg0
    fn genop_cast_int_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; scvtf d0, x0
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// CAST_FLOAT_TO_INT: result = (i64)arg0 (truncation)
    fn genop_cast_float_to_int(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fcvtzs x0, d0
        );
        self.store_rax_to_result(op.pos.get());
    }

    // ----------------------------------------------------------------
    // genop_* — memory operations
    // x86/assembler.py:1747 genop_getfield_gc etc.
    // ----------------------------------------------------------------

    /// Extract the byte offset from an op's FieldDescr.
    /// Returns 0 if no field descriptor is present.
    fn field_offset_from_descr(op: &Op) -> i32 {
        op.with_field_descr(|fd| fd.offset() as i32).unwrap_or(0)
    }

    /// Extract the field size from an op's FieldDescr.
    /// Returns 8 (WORD) if no field descriptor is present.
    fn field_size_from_descr(op: &Op) -> usize {
        op.with_field_descr(|fd| fd.field_size()).unwrap_or(8)
    }

    /// GETFIELD_GC_*: result = [arg0 + offset]
    /// The offset comes from the op's FieldDescr.
    fn genop_getfield(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        // Load the object pointer from arg0.
        self.load_arg_to_rax(op.arg(0).to_opref());

        // Load the field value at [rax + offset] into rax/x0.

        match size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; ldrb w0, [x0, offset as u32]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; ldrh w0, [x0, offset as u32]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; ldr w0, [x0, offset as u32]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; ldr x0, [x0, offset as u32]
            ),
        }

        self.store_rax_to_result(op.pos.get());
    }

    /// SETFIELD_GC: [arg0 + offset] = arg1
    fn genop_discard_setfield(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        // Load object pointer into rax/x0 and value into rcx/x1.
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());

        match size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; strb w1, [x0, offset as u32]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; strh w1, [x0, offset as u32]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; str w1, [x0, offset as u32]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; str x1, [x0, offset as u32]
            ),
        }
    }

    /// GETARRAYITEM_GC_*: result = array[index]
    /// arg0 = array pointer, arg1 = index.
    /// The base_size and item_size come from the op's ArrayDescr.
    fn genop_getarrayitem(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .with_array_descr(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((8, 8));

        // Load array pointer and index.
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());

        // Compute address: rax = rax + base_size + rcx * item_size

        // x1 = x1 * item_size; x0 = x0 + base_size + x1
        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64); // x2 = item_size
            dynasm!(self.mc ; .arch aarch64
                ; mul x1, x1, x2
            );
        }
        if base_size != 0 {
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, base_size as u32
            );
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, x1
        );
        match item_size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; ldrb w0, [x0]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; ldrh w0, [x0]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; ldr w0, [x0]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; ldr x0, [x0]
            ),
        }

        self.store_rax_to_result(op.pos.get());
    }

    /// SETARRAYITEM_GC: array[index] = value
    /// arg0 = array pointer, arg1 = index, arg2 = value.
    fn genop_discard_setarrayitem(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .with_array_descr(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((8, 8));

        // Load array pointer.
        self.load_arg_to_rax(op.arg(0).to_opref());
        // Load index.
        self.load_arg_to_rcx(op.arg(1).to_opref());

        // Compute element address: rax = rax + base_size + rcx * item_size
        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64);
            dynasm!(self.mc ; .arch aarch64
                ; mul x1, x1, x2
            );
        }
        if base_size != 0 {
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, base_size as u32
            );
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, x1
        );

        // Now load value from arg2 and store it.
        // We need a third register: use rcx/x1 again for the value
        // (the address is in rax/x0).
        // Save rax/x0 (element address) before loading value.

        // Save x0 (element address) in x2, load value into x1.
        dynasm!(self.mc ; .arch aarch64
            ; mov x2, x0
        );
        self.load_arg_to_rcx(op.arg(2).to_opref()); // loads into x1
        match item_size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; strb w1, [x2]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; strh w1, [x2]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; str w1, [x2]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; str x1, [x2]
            ),
        }
    }

    /// ARRAYLEN_GC: result = array.length
    /// The length field location comes from the ArrayDescr's len_descr().
    fn genop_arraylen(&mut self, op: &Op) {
        let len_offset = op
            .with_array_descr(|ad| ad.len_descr().map(|ld| ld.offset() as i32))
            .flatten()
            .unwrap_or(0); // Default: length at offset 0 in array header

        // Load array pointer.
        self.load_arg_to_rax(op.arg(0).to_opref());

        // Load length from [array + len_offset].
        dynasm!(self.mc ; .arch aarch64
            ; ldr x0, [x0, len_offset as u32]
        );

        self.store_rax_to_result(op.pos.get());
    }

    // ----------------------------------------------------------------
    // genop_* — calls
    // x86/assembler.py:2230 _genop_call
    // ----------------------------------------------------------------

    fn argloc_imm(arglocs: &[Loc], index: usize) -> i64 {
        match arglocs.get(index) {
            Some(Loc::Immed(i)) => i.value,
            _ => 0,
        }
    }

    /// Emit a function call. `func_arg` is the index of the function
    /// pointer arg; call arguments start at `func_arg + 1`.
    fn emit_call(&mut self, op: &Op, func_arg: usize) {
        let arg_count = op.num_args();

        dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);

        for i in (func_arg + 1)..arg_count.min(func_arg + 7) {
            let arg = op.arg(i).to_opref();
            let abi_idx = i - func_arg - 1;
            match self.resolve_opref(arg) {
                ResolvedArg::Slot(offset) => {
                    let reg = abi_idx as u8;
                    self.emit_ldr_fp(reg, offset);
                }
                ResolvedArg::Const(val) => {
                    let reg = abi_idx as u32;
                    self.emit_mov_imm64(reg, val);
                }
            }
        }

        match self.resolve_opref(op.arg(func_arg).to_opref()) {
            ResolvedArg::Slot(offset) => {
                self.emit_ldr_fp(8, offset);
                dynasm!(self.mc ; .arch aarch64 ; blr x8);
            }
            ResolvedArg::Const(val) => {
                self.emit_mov_imm64(8, val);
                dynasm!(self.mc ; .arch aarch64 ; blr x8);
            }
        }

        dynasm!(self.mc ; .arch aarch64 ; ldp x29, x30, [sp], #16);
    }

    /// aarch64/opassembler.py:1036 _emit_call + aarch64/callbuilder.py:21-67
    /// prepare_arguments.
    ///
    /// Register-to-ABI-reg shuffles must go through remap_frame_layout to
    /// handle cycles: with naïve sequential `mov x0,x2; mov x1,x0` the
    /// second move reads the already-clobbered x0. RPython routes the
    /// non-float arg plan through remap_frame_layout(asm, src, dst, ip0)
    /// and preserves the fnloc across the remap via ip1 when it is a
    /// register.
    fn emit_call_from_arglocs(&mut self, arglocs: &[Loc], func_index: usize) {
        let arg_count = arglocs.len();

        dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);

        // aarch64/callbuilder.py:54-60 — if fnloc is a core reg or stack,
        // save it to ip1 (x17) so later remap can clobber its source reg.
        let fnloc = arglocs.get(func_index).copied();
        let fnloc_in_ip1 = match fnloc {
            Some(Loc::Reg(r)) if !r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; mov x17, X(r.value));
                true
            }
            Some(Loc::Frame(f)) if !f.ebp_loc.is_float => {
                self.emit_ldr_fp(17, f.ebp_loc.value);
                true
            }
            _ => false,
        };

        // aarch64/callbuilder.py:29-41 — build arg plan.  Core and float
        // arguments have independent x0..x7 / d0..d7 banks; overflow from
        // either bank is written to the outgoing stack area in source order.
        let mut non_float_src: Vec<Loc> = Vec::new();
        let mut non_float_dst: Vec<Loc> = Vec::new();
        let mut float_src: Vec<Loc> = Vec::new();
        let mut float_dst: Vec<Loc> = Vec::new();
        let mut immed_args: Vec<(u8, i64, bool)> = Vec::new(); // (abi_idx, value, is_float)
        let mut stack_args: Vec<Loc> = Vec::new();
        let mut next_core = 0u8;
        let mut next_float = 0u8;

        for &arg in &arglocs[(func_index + 1)..arg_count] {
            let is_float = match arg {
                Loc::Frame(f) => f.ebp_loc.is_float,
                Loc::Reg(r) => r.is_xmm,
                Loc::Immed(im) => im.is_float,
                _ => false,
            };
            let abi_idx = if is_float {
                if next_float == 8 {
                    stack_args.push(arg);
                    continue;
                }
                let idx = next_float;
                next_float += 1;
                idx
            } else {
                if next_core == 8 {
                    stack_args.push(arg);
                    continue;
                }
                let idx = next_core;
                next_core += 1;
                idx
            };
            match arg {
                Loc::Frame(f) if f.ebp_loc.is_float => {
                    float_src.push(arg);
                    float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, true)));
                }
                Loc::Frame(_) => {
                    non_float_src.push(arg);
                    non_float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, false)));
                }
                Loc::Reg(r) if r.is_xmm => {
                    float_src.push(arg);
                    float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, true)));
                }
                Loc::Reg(_) => {
                    non_float_src.push(arg);
                    non_float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, false)));
                }
                Loc::Immed(im) => {
                    immed_args.push((abi_idx, im.value, im.is_float));
                }
                _ => {}
            }
        }

        // aarch64/callbuilder.py:42-50: reserve a 16-byte-aligned outgoing
        // stack area and copy every register-bank overflow argument before
        // remapping the register arguments (whose moves may clobber sources).
        let stack_bytes = (stack_args.len() * WORD + 15) & !15;
        if stack_bytes != 0 {
            dynasm!(self.mc ; .arch aarch64 ; sub sp, sp, stack_bytes as u32);
            for (slot, arg) in stack_args.into_iter().enumerate() {
                let offset = (slot * WORD) as u32;
                match arg {
                    Loc::Reg(r) if r.is_xmm => {
                        dynasm!(self.mc ; .arch aarch64 ; str D(r.value), [sp, offset]);
                    }
                    Loc::Reg(r) => {
                        dynasm!(self.mc ; .arch aarch64 ; str X(r.value), [sp, offset]);
                    }
                    Loc::Frame(f) if f.ebp_loc.is_float => {
                        self.emit_ldr_fp_d(15, f.ebp_loc.value);
                        dynasm!(self.mc ; .arch aarch64 ; str d15, [sp, offset]);
                    }
                    Loc::Frame(f) => {
                        self.emit_ldr_fp(16, f.ebp_loc.value);
                        dynasm!(self.mc ; .arch aarch64 ; str x16, [sp, offset]);
                    }
                    Loc::Immed(im) => {
                        self.emit_mov_imm64(16, im.value);
                        dynasm!(self.mc ; .arch aarch64 ; str x16, [sp, offset]);
                    }
                    other => panic!("unsupported AArch64 stack call argument {other:?}"),
                }
            }
        }

        // aarch64/callbuilder.py:62-65 — remap non-float then float args.
        let tmp_nf = Loc::Reg(crate::regloc::RegLoc::new(16, false)); // x16 (ip0)
        let tmp_fp = Loc::Reg(crate::regloc::RegLoc::new(15, true)); // d15
        self.remap_frame_layout(&non_float_src, &non_float_dst, tmp_nf);
        self.remap_frame_layout(&float_src, &float_dst, tmp_fp);

        // Immediate args after remap (each targets a distinct ABI reg).
        for (abi_idx, val, is_float) in immed_args {
            if is_float {
                self.emit_mov_imm64(15, val);
                dynasm!(self.mc ; .arch aarch64 ; fmov D(abi_idx), X(15));
            } else {
                self.emit_mov_imm64(abi_idx as u32, val);
            }
        }

        // aarch64/callbuilder.py:67 — restore fnloc from ip1 / emit call.
        if fnloc_in_ip1 {
            dynasm!(self.mc ; .arch aarch64 ; blr x17);
        } else {
            match fnloc {
                Some(Loc::Immed(i)) => {
                    let val = i.value;
                    self.emit_mov_imm64(8, val);
                    dynasm!(self.mc ; .arch aarch64 ; blr x8);
                }
                _ => {}
            }
        }

        if stack_bytes != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add sp, sp, stack_bytes as u32);
        }
        dynasm!(self.mc ; .arch aarch64 ; ldp x29, x30, [sp], #16);
    }

    fn ensure_call_result_bit_extension(&mut self, arglocs: &[Loc]) {
        let size = Self::argloc_imm(arglocs, 1) as usize;
        let signed = Self::argloc_imm(arglocs, 2) != 0;
        if size >= WORD {
            return;
        }

        match size {
            4 => {
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 32 ; asr x0, x0, 32);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 32 ; lsr x0, x0, 32);
                }
            }
            2 => {
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 48 ; asr x0, x0, 48);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; and x0, x0, 0xFFFF);
                }
            }
            1 => {
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 56 ; asr x0, x0, 56);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; and x0, x0, 0xFF);
                }
            }
            _ => {}
        }
    }

    /// assembler.py:2176 _genop_call — internal call implementation.
    fn _genop_call(&mut self, op: &Op) {
        self.emit_call(op, 0);
    }

    fn _genop_call_with_arglocs(&mut self, op: &Op, arglocs: &[Loc]) {
        let func_index = 3 + usize::from(op.opcode.is_call_release_gil());
        self.emit_call_from_arglocs(arglocs, func_index);
        if op.opcode.result_type() == Type::Int {
            self.ensure_call_result_bit_extension(arglocs);
        }
    }

    /// assembler.py:2169-2174 _genop_real_call.
    /// genop_call_i = genop_call_r = genop_call_f = genop_call_n
    fn genop_call(&mut self, op: &Op) {
        self._genop_call(op);
        if !op.pos.get().is_none() {
            if op.opcode.result_type() == Type::Float {
                self.store_d0_to_result(op.pos.get());
            } else {
                self.store_rax_to_result(op.pos.get());
            }
        }
    }

    fn genop_call_with_arglocs(&mut self, op: &Op, arglocs: &[Loc]) {
        let can_collect = op
            .with_call_descr(|descr| descr.get_extra_info().check_can_collect())
            .unwrap_or(false);
        if can_collect {
            if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
                self.push_gcmap(gcmap as *mut usize);
            }
        }
        self._genop_call_with_arglocs(op, arglocs);
        if can_collect {
            self.reload_frame_if_necessary();
            self.pop_gcmap();
        }
        if !op.pos.get().is_none() {
            if op.opcode.result_type() == Type::Float {
                self.store_d0_to_result(op.pos.get());
            } else {
                self.store_rax_to_result(op.pos.get());
            }
        }
    }

    /// Inline aheui's headerless `jit_alloc_node(value, next)` nursery bump.
    ///
    /// The fast path only advances `nursery_free` and initializes the
    /// 16-byte Node payload, so it cannot collect and emits no gcmap. The
    /// slow path is the ordinary residual call wrapper, which may collect
    /// inside `jit_alloc_node` / `Nursery::alloc` and passes `next` (the old
    /// head) as the keep-root. This is sound because the op is still a call:
    /// the optimizer's residual-call emission fences pending head/size
    /// setfields before this allocation, matching the storage collector's
    /// root-currentness requirement.
    fn genop_nursery_alloc_inline(&mut self, op: &Op, arglocs: &[Loc]) {
        const NURSERY_ALLOC_NODE_SIZE: u32 = 16; // aheui headerless Node = 16B (value@0,next@8)

        let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();
        if nf_addr == 0 || nt_addr == 0 {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        }

        let func_index = 3 + usize::from(op.opcode.is_call_release_gil());
        let (Some(&value_loc), Some(&next_loc)) =
            (arglocs.get(func_index + 1), arglocs.get(func_index + 2))
        else {
            self.genop_call_with_arglocs(op, arglocs);
            return;
        };

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();

        // Fast path uses ONLY reserved IP regs (x14/x15/x16/x17); never touches a
        // regalloc-managed register, so the slow path's original arglocs stay
        // intact. x16=nf_addr->base, x17=base, x14=newf(temp), x15=nt_addr->top.
        self.emit_mov_imm64(16, nf_addr as i64); // x16 = &nursery_free
        dynasm!(self.mc ; .arch aarch64
            ; ldr x17, [x16]                        // x17 = base = *nursery_free
            ; add x14, x17, NURSERY_ALLOC_NODE_SIZE // x14 = newf = base + 16
        );
        self.emit_mov_imm64(15, nt_addr as i64); // x15 = &nursery_top
        dynasm!(self.mc ; .arch aarch64
            ; ldr x15, [x15]                        // x15 = top = *nursery_top
            ; cmp x14, x15                          // newf vs top
            ; b.hi =>slow_path                      // newf > top -> exhausted
            ; str x14, [x16]                        // *nursery_free = newf
        );
        // Init node fields; value/next loaded from ORIGINAL arglocs (no pre-clobber).
        // load_loc_to_reg returns a managed reg untouched, or loads Frame/Immed into
        // the IP scratch we pass (x15 and x14 are free here: top consumed, newf stored).
        let vreg = self.load_loc_to_reg(&value_loc, 15);
        dynasm!(self.mc ; .arch aarch64 ; str X(vreg), [x17]); // value @ base+0
        let nreg = self.load_loc_to_reg(&next_loc, 14);
        dynasm!(self.mc ; .arch aarch64 ; str X(nreg), [x17, 8]); // next @ base+8
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x17); // result = base
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
        dynasm!(self.mc ; .arch aarch64 ; b =>done);

        dynasm!(self.mc ; .arch aarch64 ; =>slow_path);
        self.genop_call_with_arglocs(op, arglocs);
        dynasm!(self.mc ; .arch aarch64 ; =>done);
    }

    /// assembler.py:295-360 call_assembler: invoke a compiled JIT loop.
    ///
    /// RPython fast path (assembler.py:295-360):
    ///   1. _call_assembler_emit_call — call the target trace
    ///   2. _call_assembler_check_descr — CMP jf_descr == done_with_this_frame_descr
    ///   3. Path A (slow): call assembler_helper
    ///   4. Path B (fast): MOV result, [frame + ofs]
    ///   5. join paths
    ///
    /// RPython allocates the callee jitframe via malloc_jitframe (heap).
    /// We use malloc for parity: the callee's frame-slot model needs
    /// frame_depth slots (not just num_args), and heap allocation avoids
    /// stack overflow on deep recursion.
    /// assembler.py:295-360 call_assembler parity.
    /// Uses regalloc-provided arglocs to load callee arguments instead of
    /// resolve_opref(), which drops register-carried values to Const(0).
    fn genop_call_assembler(&mut self, op: &Op, arglocs: &[Loc]) {
        // handle_call_assembler (rewrite.py:665-695) always pre-builds the
        // callee jitframe — storing every inputarg, and for a virtualizable
        // passing the forced vable object as the second arg — so the backend
        // only loads arglocs[0] (the rewritten frame) and invokes the target.
        // There is no genop-side vable "expansion": materialising vable fields
        // into callee frame slots here clobbered the inputargs the rewrite had
        // already stored (double-materialisation).
        {
            let frame_loc = arglocs
                .first()
                .copied()
                .expect("call_assembler missing rewritten jitframe arg");
            let vable_loc = arglocs.get(1).copied();
            // aarch64/regalloc.py:661-664 routes CALL_ASSEMBLER through
            // `_call(..., gc_level=2)`, which spills all managed registers.
            // x19 is already saved by the JIT prologue, so use it as scratch
            // here without an extra call-site stack save.
            dynasm!(self.mc ; .arch aarch64 ; mov x19, x29);
            self.emit_load_to_rax(frame_loc);

            let target_addr: Option<usize> = op
                .with_call_descr(|cd| cd.call_target_token())
                .flatten()
                .and_then(|token| self.call_assembler_targets.get(&token).copied())
                .filter(|&addr| addr != 0);
            let is_resolved = target_addr.is_some() || self.self_entry_label.is_some();
            let result_type = op.opcode.result_type();
            let done_descr_ptr = self.done_with_this_frame_descr_ptr_for_type(result_type);
            let helper_addr = crate::call_assembler_helper_addr() as i64;
            let green_key = self.header_pc as i64;

            if !is_resolved {
                let force_addr = crate::call_assembler_force_fn_addr() as i64;
                dynasm!(self.mc ; .arch aarch64
                    ; mov x29, x19
                );
                if force_addr != 0 {
                    if let Some(vloc) = vable_loc {
                        self.emit_load_to_rax(vloc);
                    } else {
                        dynasm!(self.mc ; .arch aarch64
                            ; ldr x0, [x0, FIRST_ITEM_OFFSET as u32]
                        );
                    }
                    self.emit_mov_imm64(2, force_addr);
                    let pushed_gcmap = self.push_pending_call_gcmap();
                    dynasm!(self.mc ; .arch aarch64
                        ; blr x2
                    );
                    self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
                } else {
                    dynasm!(self.mc ; .arch aarch64
                        ; mov x0, xzr
                    );
                }
                if !op.pos.get().is_none() {
                    self.store_rax_to_result(op.pos.get());
                }
                return;
            }

            // llsupport/assembler.py:320 +
            // aarch64/opassembler.py:1110-1115 `_call_assembler_emit_call`:
            // call the target assembler entry directly with the callee
            // jitframe in x0.  The extra Rust execute trampoline was a
            // pre-existing adaptation; it is not part of the RPython fast
            // path and is too expensive for recursive CALL_ASSEMBLER.
            let pushed_gcmap = self.push_pending_call_gcmap();
            if let Some(addr) = target_addr {
                let addr = addr as i64;
                self.emit_mov_imm64(1, addr);
                dynasm!(self.mc ; .arch aarch64 ; blr x1);
            } else if let Some(entry_label) = self.self_entry_label {
                dynasm!(self.mc ; .arch aarch64 ; bl =>entry_label);
            }
            dynasm!(self.mc ; .arch aarch64
                ; mov x29, x19
            );
            self.pop_pending_call_gcmap_after_collect(pushed_gcmap);

            let fast_path = self.mc.new_dynamic_label();
            let merge = self.mc.new_dynamic_label();
            self.emit_mov_imm64(2, done_descr_ptr);
            dynasm!(self.mc ; .arch aarch64
                ; ldr x1, [x0, JF_DESCR_OFS as u32]
                ; cmp x1, x2
                ; b.eq =>fast_path
            );
            {
                // `compile.py:665` parity: helper signature is
                // `(cpu_handle, callee_jf_ptr, green_key)`. Pass cpu_ptr as
                // arg0 so the trampoline can resolve the attached
                // `done_with_this_frame_descr_*` /
                // `exit_frame_with_exception_descr_ref` identities. blr
                // through x3 so x0/x1/x2 stay live as the helper args.
                let cpu_ptr = self.cpu_handle_ptr();
                self.emit_mov_imm64(3, helper_addr);
                dynasm!(self.mc ; .arch aarch64
                    ; mov x1, x0                    // arg1 = callee_jf_ptr
                );
                self.emit_mov_imm64(2, green_key); // arg2 = green_key
                self.emit_mov_imm64(0, cpu_ptr); // arg0 = cpu_handle
                let pushed_gcmap = self.push_pending_call_gcmap();
                dynasm!(self.mc ; .arch aarch64
                    ; blr x3                        // x0 = helper result
                );
                self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
                dynasm!(self.mc ; .arch aarch64 ; b =>merge);
            }
            {
                dynasm!(self.mc ; .arch aarch64
                    ; =>fast_path
                );
                if result_type == Type::Float {
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr d0, [x0, FIRST_ITEM_OFFSET as u32]
                        ; fmov x0, d0
                    );
                } else {
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x0, [x0, FIRST_ITEM_OFFSET as u32]
                    );
                }
                dynasm!(self.mc ; .arch aarch64
                    ; =>merge
                );
            }
            // RPython `BaseAssembler.call_assembler` joins helper/fast paths
            // here without another frame reload.  The collecting target/helper
            // calls above already reload in `Aarch64CallBuilder.pop_gcmap`.
            if !op.pos.get().is_none() {
                self.store_rax_to_result(op.pos.get());
            }
        }
    }

    // ----------------------------------------------------------------
    // genop_* — allocation
    // x86/assembler.py:2338 genop_new etc.
    // These require GC runtime support. Emit trap for now.
    // ----------------------------------------------------------------

    /// NEW: allocate a fixed-size object. Requires GC runtime.
    /// Emits a trap (UD2/BRK) until GC nursery allocation is wired.
    /// Zero-initialize `size` bytes at address in x0 (preserved).
    /// Avoids clobbering callee-saved registers (x19-x28).
    fn inline_memzero(&mut self, size: i64) {
        let mut ofs = 0i32;
        while ofs + 16 <= size as i32 {
            dynasm!(self.mc ; .arch aarch64 ; stp xzr, xzr, [x0, ofs]);
            ofs += 16;
        }
        if ofs + 8 <= size as i32 {
            let u = ofs as u32;
            dynasm!(self.mc ; .arch aarch64 ; str xzr, [x0, u]);
            ofs += 8;
        }
        if ofs + 4 <= size as i32 {
            let u = ofs as u32;
            dynasm!(self.mc ; .arch aarch64 ; str wzr, [x0, u]);
        }
    }

    /// aarch64/opassembler.py:912 _write_barrier_fastpath parity.
    fn emit_write_barrier_fastpath(&mut self, op: &Op, arglocs: &[Loc]) {
        let loc_base = match arglocs.first() {
            Some(Loc::Reg(r)) => *r,
            _ => return,
        };
        let is_array = op.opcode == majit_ir::OpCode::CondCallGcWbArray;
        let loc_index = match arglocs.get(1) {
            Some(Loc::Reg(r)) => Some(*r),
            _ => None,
        };
        self.emit_write_barrier_fastpath_for_base(loc_base, is_array, loc_index);
    }

    fn emit_write_barrier_fastpath_for_base(
        &mut self,
        loc_base: crate::regloc::RegLoc,
        is_array: bool,
        loc_index: Option<crate::regloc::RegLoc>,
    ) {
        let wb = match crate::runner::with_dynasm_active_gc(|gc| gc.get_write_barrier_descr()) {
            Some(Some(wb)) => wb,
            _ => return,
        };
        let card_marking = is_array && wb.jit_wb_cards_set != 0;

        // opassembler.py:922-929: build mask
        let mut mask = wb.jit_wb_if_flag_singlebyte as i64;
        if card_marking {
            mask |= wb.jit_wb_cards_set_singlebyte as i64;
        }
        mask &= 0xFF;

        // opassembler.py:934: LDRB ip0, [base, wb_byteofs]
        let byteofs = wb.jit_wb_if_flag_byteofs;
        self.emit_ldrb_signed_offset(&loc_base, byteofs);
        // opassembler.py:936-937: TST ip0, mask
        dynasm!(self.mc ; .arch aarch64
            ; mov w17, mask as u32
            ; tst w16, w17
        );

        // opassembler.py:938-939: BEQ done (flag not set → skip)
        let done = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; b.eq =>done);

        if card_marking {
            // opassembler.py:943-949: test GCFLAG_CARDS_SET
            let cards_mask = (wb.jit_wb_cards_set_singlebyte as u8) as u32;
            dynasm!(self.mc ; .arch aarch64
                ; mov w17, cards_mask
                ; tst w16, w17
            );
            let card_mark = self.mc.new_dynamic_label();
            dynasm!(self.mc ; .arch aarch64 ; b.ne =>card_mark);

            // opassembler.py:953-976: array-specific helper call
            self.emit_wb_helper_call(
                loc_base,
                crate::runner::dynasm_write_barrier_from_array as *const () as i64,
            );

            // opassembler.py:982-987: re-check CARDS_SET after helper
            self.emit_ldrb_signed_offset(&loc_base, byteofs);
            dynasm!(self.mc ; .arch aarch64
                ; mov w17, cards_mask
                ; tst w16, w17
                ; b.eq =>done
            );

            // opassembler.py:996-1015: card marking inline
            dynasm!(self.mc ; .arch aarch64 ; =>card_mark);
            if let Some(loc_index) = loc_index {
                let shift = 3 + wb.jit_wb_card_page_shift;
                dynasm!(self.mc ; .arch aarch64
                    ; lsr x16, X(loc_index.value), shift
                    ; mvn x30, x16
                    ; lsr x16, X(loc_index.value), wb.jit_wb_card_page_shift
                    ; and x17, x16, 7
                    ; mov x16, 1
                    ; lsl x17, x16, x17
                    ; sub x30, x30, majit_gc::header::GcHeader::SIZE as u32
                    ; ldrb w16, [X(loc_base.value), x30]
                    ; orr w16, w16, w17
                    ; strb w16, [X(loc_base.value), x30]
                );
            }
        } else {
            // opassembler.py:968-976: non-array slow path
            self.emit_wb_helper_call(
                loc_base,
                crate::runner::dynasm_write_barrier as *const () as i64,
            );
        }

        dynasm!(self.mc ; .arch aarch64 ; =>done);
    }

    /// Load byte at [base + signed_byteofs] into w16 (ip0).
    fn emit_ldrb_signed_offset(&mut self, base: &crate::regloc::RegLoc, byteofs: i32) {
        if byteofs >= 0 && byteofs < 4096 {
            let ofs = byteofs as u32;
            dynasm!(self.mc ; .arch aarch64
                ; ldrb W(16), [X(base.value), ofs]
            );
        } else if byteofs >= -256 && byteofs < 0 {
            dynasm!(self.mc ; .arch aarch64
                ; ldurb W(16), [X(base.value), byteofs]
            );
        } else {
            // Large negative offset: compute address in x16, then load
            self.emit_mov_imm64(16, byteofs as i64);
            dynasm!(self.mc ; .arch aarch64
                ; add x16, X(base.value), x16
                ; ldrb W(16), [x16]
            );
        }
    }

    /// _push_all_regs_to_jitframe(also_push_vfp=True) parity: save x0-x15
    /// (GPR) + d0-d15 (VFP) on the C stack. Total 256 bytes (16-aligned).
    /// The pop counterpart is [`emit_pop_all_volatile_regs`].
    fn emit_push_all_volatile_regs(&mut self) {
        dynasm!(self.mc ; .arch aarch64
            ; stp x0, x1, [sp, -256]!
            ; stp x2, x3, [sp, 16]
            ; stp x4, x5, [sp, 32]
            ; stp x6, x7, [sp, 48]
            ; stp x8, x9, [sp, 64]
            ; stp x10, x11, [sp, 80]
            ; stp x12, x13, [sp, 96]
            ; stp x14, x15, [sp, 112]
            ; stp d0, d1, [sp, 128]
            ; stp d2, d3, [sp, 144]
            ; stp d4, d5, [sp, 160]
            ; stp d6, d7, [sp, 176]
            ; stp d8, d9, [sp, 192]
            ; stp d10, d11, [sp, 208]
            ; stp d12, d13, [sp, 224]
            ; stp d14, d15, [sp, 240]
        );
    }

    /// _pop_all_regs_from_jitframe parity: restore VFP then GPR saved by
    /// [`emit_push_all_volatile_regs`].
    fn emit_pop_all_volatile_regs(&mut self) {
        dynasm!(self.mc ; .arch aarch64
            ; ldp d14, d15, [sp, 240]
            ; ldp d12, d13, [sp, 224]
            ; ldp d10, d11, [sp, 208]
            ; ldp d8, d9, [sp, 192]
            ; ldp d6, d7, [sp, 176]
            ; ldp d4, d5, [sp, 160]
            ; ldp d2, d3, [sp, 144]
            ; ldp d0, d1, [sp, 128]
            ; ldp x14, x15, [sp, 112]
            ; ldp x12, x13, [sp, 96]
            ; ldp x10, x11, [sp, 80]
            ; ldp x8, x9, [sp, 64]
            ; ldp x6, x7, [sp, 48]
            ; ldp x4, x5, [sp, 32]
            ; ldp x2, x3, [sp, 16]
            ; ldp x0, x1, [sp], 256
        );
    }

    /// _build_wb_slowpath parity: save all GPR + VFP regs, call helper, restore.
    /// RPython: _push_all_regs_to_jitframe(also_push_vpf=True) + BL + _pop_all
    /// opassembler.py:956-960: helper_num variant depends on live VFP bindings.
    fn emit_wb_helper_call(&mut self, loc_base: crate::regloc::RegLoc, helper: i64) {
        self.emit_push_all_volatile_regs();
        if loc_base.value != 0 {
            dynasm!(self.mc ; .arch aarch64 ; mov x0, X(loc_base.value));
        }
        self.emit_mov_imm64(2, helper);
        dynasm!(self.mc ; .arch aarch64 ; blr x2);
        self.emit_pop_all_volatile_regs();
    }

    /// `_build_malloc_slowpath()` parity: preserve live state across the
    /// helper call. RPython's JIT-generated slowpath explicitly preserves all
    /// registers apart from the result convention registers; our Rust helper
    /// call must do the same at the call site.
    ///
    /// The nursery slowpath helpers communicate through `x0` (result) and may
    /// clobber `x1`, so we preserve `x2..x15` and `d0..d15` here.
    fn emit_malloc_slowpath_helper_call(&mut self, helper_reg: u8) {
        dynasm!(self.mc ; .arch aarch64
            ; stp x2, x3, [sp, -240]!
            ; stp x4, x5, [sp, 16]
            ; stp x6, x7, [sp, 32]
            ; stp x8, x9, [sp, 48]
            ; stp x10, x11, [sp, 64]
            ; stp x12, x13, [sp, 80]
            ; stp x14, x15, [sp, 96]
            ; stp d0, d1, [sp, 112]
            ; stp d2, d3, [sp, 128]
            ; stp d4, d5, [sp, 144]
            ; stp d6, d7, [sp, 160]
            ; stp d8, d9, [sp, 176]
            ; stp d10, d11, [sp, 192]
            ; stp d12, d13, [sp, 208]
            ; stp d14, d15, [sp, 224]
            ; blr X(helper_reg)
            ; ldp d14, d15, [sp, 224]
            ; ldp d12, d13, [sp, 208]
            ; ldp d10, d11, [sp, 192]
            ; ldp d8, d9, [sp, 176]
            ; ldp d6, d7, [sp, 160]
            ; ldp d4, d5, [sp, 144]
            ; ldp d2, d3, [sp, 128]
            ; ldp d0, d1, [sp, 112]
            ; ldp x14, x15, [sp, 96]
            ; ldp x12, x13, [sp, 80]
            ; ldp x10, x11, [sp, 64]
            ; ldp x8, x9, [sp, 48]
            ; ldp x6, x7, [sp, 32]
            ; ldp x4, x5, [sp, 16]
            ; ldp x2, x3, [sp], 240
        );
    }

    /// aarch64/assembler.py:682 malloc_cond parity.
    ///
    /// Inline nursery bump allocation. total_size (from op.arg(0))
    /// includes GcHeader. Result in x0 = object pointer (after header).
    fn genop_call_malloc_nursery(&mut self, op: &Op) {
        let size_ref = op.arg(0).to_opref();
        // history.py:227 ConstInt.value carried inline — prefer the inline
        // payload before falling through to the legacy pool / raw u32.
        let total_size = size_ref.inline_const_bits().unwrap_or_else(|| {
            self.constants
                .get(&size_ref.raw())
                .map(|c| c.as_raw_i64())
                .unwrap_or(size_ref.raw() as i64)
        });
        let gc_hdr = majit_gc::header::GcHeader::SIZE as i64;
        // gc.py:525-531 — read nursery slot addresses from the active GC
        // descriptor (cpu.gc_ll_descr.get_nursery_free_addr() parity), not
        // from a process-global singleton.
        let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();
        if nf_addr == 0 || nt_addr == 0 {
            self.emit_mov_imm64(0, total_size);
            self.emit_mov_imm64(
                2,
                crate::runner::dynasm_nursery_slowpath as *const () as i64,
            );
            self.emit_malloc_slowpath_helper_call(2);
            self.reload_frame_if_necessary();
            return;
        }

        // x0 = nursery_free, x1 = new_free, x2 = scratch, x3 = nursery_top
        self.emit_mov_imm64(2, nf_addr as i64);
        dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x2]);

        if total_size < 4096 {
            let ts = total_size as u32;
            dynasm!(self.mc ; .arch aarch64 ; add x1, x0, ts);
        } else {
            self.emit_mov_imm64(1, total_size);
            dynasm!(self.mc ; .arch aarch64 ; add x1, x0, x1);
        }

        self.emit_mov_imm64(3, nt_addr as i64);
        dynasm!(self.mc ; .arch aarch64 ; ldr x3, [x3]);
        dynasm!(self.mc ; .arch aarch64 ; cmp x1, x3);

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; b.hi =>slow_path);

        // Fast path: bump nursery_free, zero header, return payload ptr
        self.emit_mov_imm64(2, nf_addr as i64);
        dynasm!(self.mc ; .arch aarch64
            ; str x1, [x2]       // *nursery_free = new_free
            ; str xzr, [x0]      // zero GcHeader
        );
        let hs = gc_hdr as u32;
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, hs     // x0 = payload pointer (after header)
            ; b =>done
        );

        // Slow path: aarch64/assembler.py:605-676 `_build_malloc_slowpath` parity.
        //
        // Before calling `dynasm_nursery_slowpath` (which may trigger a
        // minor collection) we must spill every managed register holding a
        // live Ref into its canonical jitframe slot. `gcmap` identifies
        // those slots by the same register-index table that
        // `get_gcmap` writes bits from (`core_reg_index`:
        //   x0..x13 → slots 0..13, x19 → 14, x20 → 15, x21 → 16, x22 → 17).
        //
        // Without this spill the GC walks `jf_frame` slots 0..15 (per
        // gcmap bits), finds garbage, and returns with live Refs in CPU
        // registers left pointing into the pre-collection nursery. A
        // subsequent guard then captures those stale pointers into its
        // deadframe and the blackhole decoder crashes when reading the
        // freed object (observed in fib_recursive: `bh_binary_op_fn` gets
        // an int object with `ob_type=NULL` after minor GC).
        //
        // Matches RPython's `_push_all_regs_to_jitframe(mc, [r.x0, r.x1], True)`
        // before the `BL` and `_pop_all_regs_from_jitframe(mc, [r.x0, r.x1], ...)` after.
        // x0 is the total_size argument / return pointer; x1 is an internal
        // temp that regalloc already guarantees doesn't hold a live Ref at
        // the malloc site (see `MALLOC_NURSERY_CLOBBER`).
        dynasm!(self.mc ; .arch aarch64 ; =>slow_path);
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        // Save x2..x13, x19..x22 to their slots (x0/x1 excluded per ignored_regs).
        dynasm!(self.mc ; .arch aarch64
            ; stp x2, x3, [x29, base_ofs + 2 * 8]
            ; stp x4, x5, [x29, base_ofs + 4 * 8]
            ; stp x6, x7, [x29, base_ofs + 6 * 8]
            ; stp x8, x9, [x29, base_ofs + 8 * 8]
            ; stp x10, x11, [x29, base_ofs + 10 * 8]
            ; stp x12, x13, [x29, base_ofs + 12 * 8]
            ; stp x19, x20, [x29, base_ofs + 14 * 8]
            ; stp x21, x22, [x29, base_ofs + 16 * 8]
        );
        // assembler.py:649-650: store gcmap to jf_gcmap so the collector
        // can trace live Refs pinned to frame slots during the slow-path
        // allocator call. The gcmap was captured at regalloc time in
        // `perform_with_gcmap` and threaded via RegAllocOp::Perform.gcmap.
        if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.push_gcmap(gcmap as *mut usize);
        } else {
            let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
            dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
        }
        self.emit_mov_imm64(0, total_size);
        self.emit_mov_imm64(
            2,
            crate::runner::dynasm_nursery_slowpath as *const () as i64,
        );
        self.emit_malloc_slowpath_helper_call(2);
        self.reload_frame_if_necessary();
        // pop_gcmap: clear jf_gcmap after collecting call
        let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
        dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
        // Restore x2..x13, x19..x22 from jitframe slots (GC may have
        // updated the stored pointers). x0 keeps the allocated payload ptr.
        let base_ofs_r = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        dynasm!(self.mc ; .arch aarch64
            ; ldp x2, x3, [x29, base_ofs_r + 2 * 8]
            ; ldp x4, x5, [x29, base_ofs_r + 4 * 8]
            ; ldp x6, x7, [x29, base_ofs_r + 6 * 8]
            ; ldp x8, x9, [x29, base_ofs_r + 8 * 8]
            ; ldp x10, x11, [x29, base_ofs_r + 10 * 8]
            ; ldp x12, x13, [x29, base_ofs_r + 12 * 8]
            ; ldp x19, x20, [x29, base_ofs_r + 14 * 8]
            ; ldp x21, x22, [x29, base_ofs_r + 16 * 8]
        );
        // `dynasm_nursery_slowpath` returns x0 = 0 on real host OOM
        // (calloc failure preserved as NULL per runner.rs).  Route
        // through the propagate path before the fast-path join so the
        // null pointer never reaches subsequent typed stores.
        self.emit_propagate_memory_error_if_null(0);

        dynasm!(self.mc ; .arch aarch64 ; =>done);
    }

    /// aarch64/assembler.py:682 malloc_cond parity.
    /// Inline nursery bump allocation for NEW.
    fn new_alloc_fn_addr() -> i64 {
        if crate::runner::new_via_gc_enabled() {
            crate::runner::dynasm_new_alloc as *const () as i64
        } else {
            libc::malloc as *const () as i64
        }
    }

    fn genop_new(&mut self, op: &Op) {
        let obj_size = op.with_size_descr(|sd| sd.size()).unwrap_or(16) as i64;
        self.emit_mov_imm64(0, obj_size);
        self.emit_mov_imm64(2, Self::new_alloc_fn_addr());
        dynasm!(self.mc ; .arch aarch64 ; blr x2);
        self.inline_memzero(obj_size);
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    fn genop_new_with_vtable(&mut self, op: &Op) {
        let obj_size = op.with_size_descr(|sd| sd.size()).unwrap_or(16) as i64;
        let vtable = op.with_size_descr(|sd| sd.vtable()).unwrap_or(0) as i64;
        self.emit_mov_imm64(0, obj_size);
        self.emit_mov_imm64(2, Self::new_alloc_fn_addr());
        dynasm!(self.mc ; .arch aarch64 ; blr x2);
        self.inline_memzero(obj_size);
        if vtable != 0 {
            self.emit_mov_imm64(1, vtable);
            dynasm!(self.mc ; .arch aarch64 ; str x1, [x0]);
        }
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    /// NEW_ARRAY / NEW_ARRAY_CLEAR: allocate an array.
    fn genop_new_array(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .with_array_descr(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((8, 8));
        self.genop_alloc_varsize(op, base_size, item_size);
    }

    // ----------------------------------------------------------------
    // genop_* — misc
    // ----------------------------------------------------------------

    /// FORCE_TOKEN: return the jitframe pointer itself.
    /// x86/assembler.py genop_force_token: mov resloc, ebp
    fn genop_force_token(&mut self, op: &Op) {
        // The frame pointer is the force token.
        dynasm!(self.mc ; .arch aarch64
            ; mov x0, x29
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// STRLEN / UNICODELEN: result = string.length
    /// Load the length field from the string/unicode object header.
    /// arg0 = string pointer. The length is at a fixed offset in the
    /// RPython string representation. For RPython strings, the length
    /// is typically at offset 8 (after the GC header / hash field).
    fn genop_strlen(&mut self, op: &Op) {
        // rewrite.py:273-282 parity — the length field offset comes from
        // `get_array_token(rstr.STR/UNICODE, ...)` → `ArrayDescr.lendescr`
        // in the JIT descr model.  Upstream emits `GC_LOAD_I` reading
        // `ArrayDescr.lendescr.offset`; the direct path mirrors that.
        let offset = op
            .with_array_descr(|ad| ad.len_descr().map(|ld| ld.offset() as i32))
            .flatten()
            .unwrap_or_else(|| Self::field_offset_from_descr(op));
        self.load_arg_to_rax(op.arg(0).to_opref());

        dynasm!(self.mc ; .arch aarch64
            ; ldr x0, [x0, offset as u32]
        );

        self.store_rax_to_result(op.pos.get());
    }

    /// STRGETITEM / UNICODEGETITEM: result = string[index]
    /// arg0 = string pointer, arg1 = index.
    /// Address = base + (basesize - extra_null) + index * itemsize, per
    /// `rewrite.py:295-306` — STR has `extra_item_after_alloc=1` so the
    /// token basesize overshoots the first char by 1.
    fn genop_strgetitem(&mut self, op: &Op) {
        let (mut base_size, item_size) = op
            .with_array_descr(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((17, 1)); // rstr.STR token defaults (basesize=17, itemsize=1)
        if op.opcode == OpCode::Strgetitem {
            debug_assert_eq!(item_size, 1, "STRGETITEM itemsize must be 1");
            base_size -= 1; // rewrite.py:299 — skip the extra null character
        }

        self.load_arg_to_rax(op.arg(0).to_opref()); // string pointer
        self.load_arg_to_rcx(op.arg(1).to_opref()); // index

        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64);
            dynasm!(self.mc ; .arch aarch64
                ; mul x1, x1, x2
            );
        }
        if base_size != 0 {
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, base_size as u32
            );
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, x1
        );
        match item_size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; ldrb w0, [x0]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; ldrh w0, [x0]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; ldr w0, [x0]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; ldr x0, [x0]
            ),
        }

        self.store_rax_to_result(op.pos.get());
    }

    // ================================================================
    // assembler.py:1817 genop_save_exc_class / genop_save_exception
    // ================================================================

    /// assembler.py:1817 genop_save_exc_class — stub: returns 0.
    fn genop_save_exc_class(&mut self, op: &Op) {
        dynasm!(self.mc ; .arch aarch64 ; mov x0, 0);
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    /// assembler.py:1827 genop_save_exception — stub: returns 0.
    fn genop_save_exception(&mut self, op: &Op) {
        dynasm!(self.mc ; .arch aarch64 ; mov x0, 0);
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    // ================================================================
    // genop_* — extended integer arithmetic
    // ================================================================

    /// INT_FLOORDIV: result = arg0 / arg1 (signed)
    fn genop_int_floordiv(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; sdiv x0, x0, x1
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_MOD: result = arg0 % arg1 (signed)
    fn genop_int_mod(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; sdiv x2, x0, x1
            ; msub x0, x2, x1, x0
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// UINT_MUL_HIGH: upper 64 bits of unsigned multiply
    fn genop_uint_mul_high(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; umulh x0, x0, x1
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_SIGNEXT: sign-extend from num_bytes width to 64 bits.
    fn genop_int_signext(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        let num_bytes = match self.resolve_opref(op.arg(1).to_opref()) {
            ResolvedArg::Const(v) => v,
            _ => 8,
        };
        let shift = 64 - num_bytes * 8;
        if shift > 0 && shift < 64 {
            let sh32 = shift as u32;
            dynasm!(self.mc ; .arch aarch64
                ; lsl x0, x0, sh32
                ; asr x0, x0, sh32
            );
        }
        self.store_rax_to_result(op.pos.get());
    }

    // ================================================================
    // genop_* — extended float operations
    // ================================================================

    /// FLOAT_ABS: result = |arg0|
    fn genop_float_abs(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fabs d0, d0
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// FLOAT_LT/LE/EQ/NE/GT/GE: float comparison.
    /// For lt/le, swap operands so JA/JAE handles NaN correctly.
    fn genop_float_cmp(&mut self, op: &Op) {
        let swap = matches!(op.opcode, OpCode::FloatLt | OpCode::FloatLe);
        if swap {
            self.load_float_arg_to_d0(op.arg(1).to_opref());
            self.load_float_arg_to_d1(op.arg(0).to_opref());
        } else {
            self.load_float_arg_to_d0(op.arg(0).to_opref());
            self.load_float_arg_to_d1(op.arg(1).to_opref());
        }

        dynasm!(self.mc ; .arch aarch64 ; fcmp d0, d1);
        match op.opcode {
            OpCode::FloatLt | OpCode::FloatGt => {
                dynasm!(self.mc ; .arch aarch64 ; cset x0, gt);
            }
            OpCode::FloatLe | OpCode::FloatGe => {
                dynasm!(self.mc ; .arch aarch64 ; cset x0, ge);
            }
            OpCode::FloatEq => {
                dynasm!(self.mc ; .arch aarch64 ; cset x0, eq);
            }
            OpCode::FloatNe => {
                dynasm!(self.mc ; .arch aarch64 ; cset x0, ne);
            }
            _ => {
                dynasm!(self.mc ; .arch aarch64 ; cset x0, eq);
            }
        }
        dynasm!(self.mc ; .arch aarch64 ; cmp x0, 0);
        self.guard_success_cc = Some(CC_NE);
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    /// CAST_FLOAT_TO_SINGLEFLOAT: f64 → f32 (bits in lower 32 of i64)
    fn genop_cast_float_to_singlefloat(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fcvt s0, d0
            ; fmov w0, s0
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// CAST_SINGLEFLOAT_TO_FLOAT: f32 (bits in lower 32) → f64
    fn genop_cast_singlefloat_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch aarch64
            ; fmov s0, w0
            ; fcvt d0, s0
        );
        self.store_d0_to_result(op.pos.get());
    }

    // ================================================================
    // genop_* — GC memory operations
    // ================================================================

    /// Emit a sized load from [rax]/[x0]. Positive size = zero-extend,
    /// negative = sign-extend.
    fn emit_load_from_rax_sized(&mut self, itemsize: i32) {
        let abs_size = itemsize.unsigned_abs() as usize;
        let signed = itemsize < 0;
        match (abs_size, signed) {
            (1, true) => dynasm!(self.mc ; .arch aarch64 ; ldrsb x0, [x0]),
            (2, true) => dynasm!(self.mc ; .arch aarch64 ; ldrsh x0, [x0]),
            (4, true) => dynasm!(self.mc ; .arch aarch64 ; ldrsw x0, [x0]),
            (1, false) => dynasm!(self.mc ; .arch aarch64 ; ldrb w0, [x0]),
            (2, false) => dynasm!(self.mc ; .arch aarch64 ; ldrh w0, [x0]),
            (4, false) => dynasm!(self.mc ; .arch aarch64 ; ldr w0, [x0]),
            _ => dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x0]),
        }
    }

    /// Emit a sized store of rcx/x1 to [rax]/[x0].
    fn emit_store_to_rax_sized(&mut self, size: usize) {
        match size {
            1 => dynasm!(self.mc ; .arch aarch64 ; strb w1, [x0]),
            2 => dynasm!(self.mc ; .arch aarch64 ; strh w1, [x0]),
            4 => dynasm!(self.mc ; .arch aarch64 ; str w1, [x0]),
            _ => dynasm!(self.mc ; .arch aarch64 ; str x1, [x0]),
        }
    }

    /// Resolve an OpRef that is expected to be a compile-time constant.
    fn resolve_const_or(&self, opref: OpRef, default: i64) -> i64 {
        match self.resolve_opref(opref) {
            ResolvedArg::Const(v) => v,
            _ => default,
        }
    }

    /// GC_LOAD_I/R/F: load from base + offset with given itemsize.
    /// arg(0) = base, arg(1) = offset, arg(2) = itemsize.
    fn genop_gc_load(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);

        let itemsize = self.resolve_const_or(op.arg(2).to_opref(), 8) as i32;
        self.emit_load_from_rax_sized(itemsize);
        self.store_rax_to_result(op.pos.get());
    }

    /// GC_LOAD_INDEXED_I/R/F: load from base + base_offset + index * scale.
    /// arg(0)=base, arg(1)=index, arg(2)=scale, arg(3)=base_offset, arg(4)=itemsize.
    fn genop_gc_load_indexed(&mut self, op: &Op) {
        let scale = self.resolve_const_or(op.arg(2).to_opref(), 1) as i32;
        let base_offset = self.resolve_const_or(op.arg(3).to_opref(), 0) as i32;
        let itemsize = self.resolve_const_or(op.arg(4).to_opref(), 8) as i32;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());

        if scale != 1 {
            self.emit_mov_imm64(2, scale as i64);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        if base_offset != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, base_offset as u32);
        }

        self.emit_load_from_rax_sized(itemsize);
        self.store_rax_to_result(op.pos.get());
    }

    /// GC_STORE: store value to base + offset.
    /// 4-arg form: arg(0)=base, arg(1)=offset, arg(2)=value, arg(3)=itemsize.
    fn genop_discard_gc_store(&mut self, op: &Op) {
        if op.num_args() < 4 {
            return; // 3-arg GC rewrite form — skip for now
        }
        let itemsize = self
            .resolve_const_or(op.arg(3).to_opref(), 8)
            .unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        self.emit_store_to_rax_sized(itemsize);
    }

    /// GC_STORE_INDEXED: store to base + base_offset + index * scale.
    /// arg(0)=base, arg(1)=index, arg(2)=value, arg(3)=scale,
    /// arg(4)=base_offset, arg(5)=itemsize.
    fn genop_discard_gc_store_indexed(&mut self, op: &Op) {
        let scale = self.resolve_const_or(op.arg(3).to_opref(), 1) as i32;
        let base_offset = self.resolve_const_or(op.arg(4).to_opref(), 0) as i32;
        let itemsize = self
            .resolve_const_or(op.arg(5).to_opref(), 8)
            .unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        if scale != 1 {
            self.emit_mov_imm64(2, scale as i64);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        if base_offset != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, base_offset as u32);
        }
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        self.emit_store_to_rax_sized(itemsize);
    }

    /// RAW_LOAD_I/F: load from base + offset using descriptor.
    fn genop_raw_load(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);

        self.emit_load_from_rax_sized(size as i32);
        let _ = offset; // offset is in the descriptor, not used for raw_load
        self.store_rax_to_result(op.pos.get());
    }

    /// RAW_STORE: store value to base + offset using descriptor.
    fn genop_discard_raw_store(&mut self, op: &Op) {
        let size = Self::field_size_from_descr(op);

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        self.emit_store_to_rax_sized(size);
    }

    // ================================================================
    // genop_* — interior field operations
    // ================================================================

    /// GETINTERIORFIELD_GC_I/R/F: load field from array-of-structs element.
    fn genop_getinteriorfield(&mut self, op: &Op) {
        let (base_size, item_size, field_offset, field_size) = op
            .with_interior_field_descr(|id| {
                let ad = id.array_descr();
                let fd = id.field_descr();
                (
                    ad.base_size() as i32,
                    ad.item_size() as i32,
                    fd.offset() as i32,
                    fd.field_size(),
                )
            })
            .unwrap_or((8, 8, 0, 8));

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        let total_offset = base_size + field_offset;

        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        if total_offset != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, total_offset as u32);
        }
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);

        self.emit_load_from_rax_sized(field_size as i32);
        self.store_rax_to_result(op.pos.get());
    }

    /// SETINTERIORFIELD_GC/RAW: write field in array-of-structs element.
    fn genop_discard_setinteriorfield(&mut self, op: &Op) {
        let (base_size, item_size, field_offset, field_size) = op
            .with_interior_field_descr(|id| {
                let ad = id.array_descr();
                let fd = id.field_descr();
                (
                    ad.base_size() as i32,
                    ad.item_size() as i32,
                    fd.offset() as i32,
                    fd.field_size(),
                )
            })
            .unwrap_or((8, 8, 0, 8));

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        let total_offset = base_size + field_offset;

        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        if total_offset != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, total_offset as u32);
        }
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);

        self.emit_store_to_rax_sized(field_size);
    }

    // ================================================================
    // genop_* — call variants
    // ================================================================

    /// COND_CALL_N: if arg(0) != 0, call function at arg(1).
    ///
    /// arglocs come from the regalloc plan (`consider_discard_nargs_j2`):
    /// `[condition, fn, call_args...]`. aarch64 has no cc-fusion (the compare
    /// result is always materialised into a register), so the condition can
    /// be register-resident — `resolve_opref` only handles slots/constants and
    /// would fail on a register. Use the regalloc locations directly:
    /// `emit_load_to_rax` carries register/slot/immediate, and
    /// `emit_call_from_arglocs` (func_index 1) loads the call args from their
    /// regalloc locations rather than re-resolving the op operands.
    /// COND_CALL: if arg(0) != 0, call arg(1)(arg(2..)); discard result.
    ///
    /// `consider_discard_nargs_j2` emits no `before_call`, so the regalloc
    /// does NOT spill caller-saved registers across this op. On the taken
    /// path the callee clobbers x0..x13 + d0..d7, which would destroy any
    /// value live across the cond_call (e.g. a residual-call result stored
    /// into a frame after the cond_call). Save and restore all volatile
    /// registers around the call — `_build_cond_call_slowpath(callee_only=
    /// False)` parity, inlined like the WB slowpath.
    fn genop_discard_cond_call(&mut self, op: &Op, arglocs: &[Loc]) {
        let _ = op;
        // Test the condition in a scratch register (ip0/x16), not an
        // allocatable register, so the test never clobbers a live value
        // before it is saved.
        self.emit_load_loc_to_ip0(arglocs[0]);
        let skip_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; cbz x16, =>skip_label);

        self.emit_push_all_volatile_regs();
        self.emit_call_from_arglocs(arglocs, 1);
        self.emit_pop_all_volatile_regs();

        dynasm!(self.mc ; .arch aarch64 ; =>skip_label);
    }

    /// Load `loc` into the ip0 scratch register (x16) without touching any
    /// allocatable register.
    fn emit_load_loc_to_ip0(&mut self, loc: Loc) {
        match loc {
            Loc::Reg(r) if !r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; mov x16, X(r.value));
            }
            Loc::Immed(imm) => {
                self.emit_mov_imm64(16, imm.value);
            }
            Loc::Frame(f) if !f.ebp_loc.is_float => {
                self.emit_ldr_fp(16, f.ebp_loc.value);
            }
            _ => {
                self.emit_load_to_rax(loc);
                dynasm!(self.mc ; .arch aarch64 ; mov x16, x0);
            }
        }
    }

    /// COND_CALL_VALUE_I/R: if arg(0) == 0, call function; else result = arg(0).
    fn genop_cond_call_value(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        let skip_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; cbnz x0, =>skip_label);

        self.emit_call(op, 1);

        dynasm!(self.mc ; .arch aarch64 ; =>skip_label);

        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    // ================================================================
    // genop_* — string/array operations
    // ================================================================

    /// STRSETITEM / UNICODESETITEM: string[index] = value.
    /// Address = base + (basesize - extra_null) + index * itemsize, per
    /// `rewrite.py:307-318` — STR has `extra_item_after_alloc=1` so the
    /// token basesize overshoots the first char by 1; UNICODE does not.
    fn genop_discard_strsetitem(&mut self, op: &Op) {
        let (mut base_size, item_size) = op
            .with_array_descr(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((17, 1));
        if op.opcode == OpCode::Strsetitem {
            debug_assert_eq!(item_size, 1, "STRSETITEM itemsize must be 1");
            base_size -= 1; // rewrite.py:311 — skip the extra null character
        }

        self.load_arg_to_rax(op.arg(0).to_opref()); // string
        self.load_arg_to_rcx(op.arg(1).to_opref()); // index
        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, base_size as u32
            ; add x0, x0, x1
        );
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        self.emit_store_to_rax_sized(item_size as usize);
    }

    /// COPYSTRCONTENT / COPYUNICODECONTENT: copy substring.
    /// arg(0)=src, arg(1)=dst, arg(2)=src_start, arg(3)=dst_start, arg(4)=length.
    ///
    /// Fallback emitter for the no-rewriter path.  When the GC rewriter is
    /// present (production), `rewrite.py:1045-1080
    /// rewrite_copy_str_content` replaces this op with
    /// LOAD_EFFECTIVE_ADDRESS × 2 + CALL_N(memcpy, …) before assembly, so
    /// this function is never reached.  Kept for tests that run the
    /// backend directly on un-rewritten ops.  Mirrors upstream's basesize
    /// handling (`rewrite.py:1049-1053`).
    fn genop_discard_copystrcontent(&mut self, op: &Op) {
        let (mut base_size, item_size) = op
            .with_array_descr(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((16, 1));
        // rewrite.py:1049-1053 `rewrite_copy_str_content` — COPYSTRCONTENT
        // uses `str_descr.basesize - 1` to skip the `extra_item_after_alloc`
        // null terminator carried by `rstr.STR.chars` (`rstr.py:1226-1228`).
        // COPYUNICODECONTENT's unicode_descr has no extra item.  Mirrors the
        // same correction `strgetsetitem_token` applies for STR{GET,SET}ITEM.
        if op.opcode == OpCode::Copystrcontent {
            debug_assert_eq!(item_size, 1, "COPYSTRCONTENT itemsize must be 1");
            base_size -= 1;
        }

        // Compute byte_count = length * item_size
        self.load_arg_to_rax(op.arg(4).to_opref());
        if item_size != 1 {
            self.emit_mov_imm64(1, item_size);
            dynasm!(self.mc ; .arch aarch64 ; mul x0, x0, x1);
        }
        dynasm!(self.mc ; .arch aarch64 ; str x0, [sp, #-16]!); // byte_count

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(2).to_opref());
        if item_size != 1 {
            self.emit_mov_imm64(2, item_size);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, base_size as u32
            ; add x0, x0, x1
            ; str x0, [sp, #-16]!  // src_addr
        );

        self.load_arg_to_rax(op.arg(1).to_opref());
        self.load_arg_to_rcx(op.arg(3).to_opref());
        if item_size != 1 {
            self.emit_mov_imm64(2, item_size);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, base_size as u32
            ; add x0, x0, x1
        );

        let memmove_ptr = libc::memmove as *const () as i64;
        dynasm!(self.mc ; .arch aarch64
            ; ldr x1, [sp], #16  // src
            ; ldr x2, [sp], #16  // count
        );
        // x0 = dst already
        dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);
        self.emit_mov_imm64(3, memmove_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; blr x3
            ; ldp x29, x30, [sp], #16
        );
    }

    /// NEWSTR: allocate a byte string of given length.
    /// `base_size` / `item_size` come from the injected ArrayDescr
    /// (`builtin_string_array_descr` in `runner.rs`), which encodes
    /// `get_array_token(rstr.STR, ...)` — basesize includes the +1
    /// extra_item_after_alloc null terminator.
    fn genop_newstr(&mut self, op: &Op) {
        let (base_size, item_size) = Self::array_token_from_descr(op, 16, 1);
        self.genop_alloc_varsize(op, base_size, item_size);
    }

    /// NEWUNICODE: allocate a unicode string (4-byte chars).
    /// Basesize = 16 (no extra_item_after_alloc), itemsize = 4.
    fn genop_newunicode(&mut self, op: &Op) {
        let (base_size, item_size) = Self::array_token_from_descr(op, 16, 4);
        self.genop_alloc_varsize(op, base_size, item_size);
    }

    /// Read `(base_size, item_size)` from the injected ArrayDescr.
    /// Fallback used only when the descr is missing (should never happen
    /// for NEWSTR/NEWUNICODE after `inject_builtin_string_descrs`).
    fn array_token_from_descr(op: &Op, fallback_base: i64, fallback_item: i64) -> (i64, i64) {
        op.with_array_descr(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((fallback_base, fallback_item))
    }

    /// Shared implementation for NEWSTR / NEWUNICODE / NEW_ARRAY.
    /// Allocates base_size + length * item_size bytes, zero-fills,
    /// and writes length to the header.
    fn genop_alloc_varsize(&mut self, op: &Op, base_size: i64, item_size: i64) {
        // arg(0) = length
        self.load_arg_to_rax(op.arg(0).to_opref());
        let malloc_ptr = libc::malloc as *const () as i64;
        let memset_ptr = libc::memset as *const () as i64;

        dynasm!(self.mc ; .arch aarch64
            ; str x0, [sp, #-16]!               // save length
        );
        if item_size != 1 {
            self.emit_mov_imm64(1, item_size);
            dynasm!(self.mc ; .arch aarch64 ; mul x0, x0, x1);
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, base_size as u32
            ; str x0, [sp, #-16]!               // save total_size
        );
        dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);
        self.emit_mov_imm64(8, malloc_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; blr x8
            ; ldp x29, x30, [sp], #16
            ; ldr x2, [sp], #16                 // total_size
            ; str x0, [sp, #-16]!               // save ptr on stack
            ; mov x1, 0                          // val = 0
            ; stp x29, x30, [sp, #-16]!
        );
        self.emit_mov_imm64(8, memset_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; blr x8
            ; ldp x29, x30, [sp], #16
            ; ldr x0, [sp], #16                  // restore ptr from stack
            ; ldr x1, [sp], #16                  // length
            ; str x1, [x0, 8]                    // store length at offset 8
        );

        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    /// ZERO_ARRAY: zero a range in an array.
    /// arg(0)=base, arg(1)=start, arg(2)=size, arg(3)=scale_start, arg(4)=scale_size.
    fn genop_discard_zero_array(&mut self, op: &Op) {
        let (base_size, _) = op
            .with_array_descr(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((8, 8));

        let scale_start = self.resolve_const_or(op.arg(3).to_opref(), 1);
        let scale_size = self.resolve_const_or(op.arg(4).to_opref(), 1);
        let memset_ptr = libc::memset as *const () as i64;

        // byte_offset = base_size + start * scale_start
        // byte_length = size * scale_size
        self.load_arg_to_rax(op.arg(0).to_opref()); // base
        self.load_arg_to_rcx(op.arg(1).to_opref()); // start

        if scale_start != 1 {
            self.emit_mov_imm64(2, scale_start);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, base_size as u32
            ; add x0, x0, x1
            ; str x0, [sp, #-16]!               // save dest
        );
        self.load_arg_to_rax(op.arg(2).to_opref());
        if scale_size != 1 {
            self.emit_mov_imm64(1, scale_size);
            dynasm!(self.mc ; .arch aarch64 ; mul x0, x0, x1);
        }
        dynasm!(self.mc ; .arch aarch64
            ; mov x2, x0                         // byte_length
            ; ldr x0, [sp], #16                  // dest
            ; mov x1, 0
            ; stp x29, x30, [sp, #-16]!
        );
        self.emit_mov_imm64(8, memset_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; blr x8
            ; ldp x29, x30, [sp], #16
        );
    }

    // ================================================================
    // genop_* — address computation
    // ================================================================

    /// LOAD_EFFECTIVE_ADDRESS: result = base + (index << shift) + baseofs.
    /// resoperation.py:1052-1054 — `[v_gcptr, v_index, c_baseofs, c_shift]`.
    /// arg(0)=base, arg(1)=index, arg(2)=baseofs, arg(3)=shift.
    fn genop_load_effective_address(&mut self, op: &Op) {
        let baseofs = self.resolve_const_or(op.arg(2).to_opref(), 0) as i32;
        let shift = self.resolve_const_or(op.arg(3).to_opref(), 0) as i32;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());

        if shift != 0 {
            // add x0, x0, x1, lsl #shift fuses << and + in one insn;
            // opencode via a scratch shift to keep store_rax_to_result in rax.
            dynasm!(self.mc ; .arch aarch64 ; lsl x1, x1, shift as u32);
        }
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        if baseofs != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, baseofs as u32);
        }

        self.store_rax_to_result(op.pos.get());
    }
}

/// Flush icache — aarch64 only.
fn flush_icache(addr: *const u8, len: usize) {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sys_icache_invalidate(start: *mut u8, size: usize);
        }
        unsafe { sys_icache_invalidate(addr as *mut u8, len) };
    }
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn __clear_cache(start: *mut u8, end: *mut u8);
        }
        unsafe { __clear_cache(addr as *mut u8, (addr as *mut u8).add(len)) };
    }
}
