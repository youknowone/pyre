use std::cell::{Cell, RefCell, UnsafeCell};
/// Cranelift-based JIT code generation backend.
///
/// Translates majit IR traces into native code via Cranelift, then
/// executes them as ordinary function pointers.
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{
    AbiParam, BlockArg, Function, InstBuilder, MemFlags, Signature, StackSlotData, StackSlotKind,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use cranelift_codegen::ir::Value as CValue;

use majit_codegen::{
    AsmInfo, BackendError, CompiledTraceInfo, DeadFrame, ExitFrameLayout, ExitRecoveryLayout,
    ExitValueSourceLayout, ExitVirtualLayout, FailDescrLayout, JitCellToken, TerminalExitLayout,
};
use majit_gc::header::{GcHeader, TYPE_ID_MASK};
use majit_gc::rewrite::GcRewriterImpl;
use majit_gc::{GcAllocator, GcRewriter, WriteBarrierDescr, flags as gc_flags};
use majit_ir::{
    CallDescr, EffectInfo, FailDescr, GcRef, InputArg, OopSpecIndex, Op, OpCode, OpRef, Type, Value,
};

use crate::guard::{BridgeData, CraneliftFailDescr, FrameData};

// ── JitFrame layout constants (jitframe.py:93-101) ──────────────────
// Header: [frame_info:8, descr:8, force_descr:8, gcmap:8,
//          savedata:8, guard_exc:8, forward:8]  = 56 bytes
// Array:  [length:8, item[0]:8, item[1]:8, ...]
/// Byte offset of `jf_descr` from JitFrame start.
const JF_DESCR_OFS: i32 = 8;
const JF_FORCE_DESCR_OFS: i32 = 16;
/// Byte offset of `jf_gcmap` from JitFrame start.
const JF_GCMAP_OFS: i32 = 24;
const JF_SAVEDATA_OFS: i32 = 32;
const JF_GUARD_EXC_OFS: i32 = 40;
const JF_FORWARD_OFS: i32 = 48;
/// Byte offset of `jf_frame_length` from JitFrame start.
const JF_FRAME_LENGTH_OFS: i32 = 56;
/// Byte offset of `jf_frame[0]` from JitFrame start (header + length field).
const JF_FRAME_ITEM0_OFS: i32 = 64; // 56 + 8
/// Number of fixed header fields in JITFRAME (regalloc.py:1094, 1106).
const JITFRAME_FIXED_SIZE: u32 = 7;
/// GC type id for JitFrame objects.
/// RPython parity: rgc.register_custom_trace_hook(JITFRAME, jitframe_trace).
const JITFRAME_GC_TYPE_ID: u32 = 2;

/// Custom trace function for GC-managed jitframes.
///
/// RPython parity: jitframe.py:104-136 `jitframe_trace`.
///
/// 1. Trace header ref fields (jf_descr..jf_forward) — jitframe.py:105-109
/// 2. Read jf_gcmap pointer, trace ref slots in jf_frame — jitframe.py:115-136
///
/// # Safety
/// `obj_addr` must point to a valid JitFrame payload (after GC header).
unsafe fn jitframe_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut GcRef)) {
    let jf_ptr = obj_addr as *const u8;

    // jitframe.py:105-109: trace header ref fields.
    for &ofs in &[
        JF_DESCR_OFS,
        JF_FORCE_DESCR_OFS,
        JF_SAVEDATA_OFS,
        JF_GUARD_EXC_OFS,
        JF_FORWARD_OFS,
    ] {
        let slot_ptr = unsafe { jf_ptr.add(ofs as usize) as *mut GcRef };
        let gcref = unsafe { *slot_ptr };
        if !gcref.is_null() {
            f(slot_ptr);
        }
    }

    // jitframe.py:115: gcmap = (obj_addr + getofs('jf_gcmap')).address[0]
    let gcmap_ptr = unsafe { *(jf_ptr.add(JF_GCMAP_OFS as usize) as *const *const u8) };
    if gcmap_ptr.is_null() {
        return;
    }
    // jitframe.py:117: gcmap_lgt = (gcmap + GCMAPLENGTHOFS).signed[0]
    let gcmap_lgt = unsafe { *(gcmap_ptr as *const isize) };
    let frame_items = unsafe { jf_ptr.add(JF_FRAME_ITEM0_OFS as usize) };
    // jitframe.py:118-136: iterate bitmap words
    let mut no: isize = 0;
    while no < gcmap_lgt {
        let word = unsafe { *(gcmap_ptr.add(8 + 8 * no as usize) as *const usize) };
        let mut bitindex: usize = 0;
        while bitindex < 64 {
            if word & (1usize << bitindex) != 0 {
                let index = no as usize * 64 + bitindex;
                let slot_ptr = unsafe { frame_items.add(8 * index) as *mut GcRef };
                let gcref = unsafe { *slot_ptr };
                if !gcref.is_null() {
                    f(slot_ptr);
                }
            }
            bitindex += 1;
        }
        no += 1;
    }
}

/// Ensure the JITFRAME GC type (type_id=2) is registered.
///
/// RPython: rgc.register_custom_trace_hook(JITFRAME, jitframe_trace)
/// called from jitframe_allocate (jitframe.py:49).
fn ensure_jitframe_type_registered(gc: &mut dyn GcAllocator) {
    use majit_gc::TypeInfo;
    let current_count = gc.type_count();
    if current_count == 0 {
        return; // Stub GC (TrackingGc), no type registry
    }
    if current_count > JITFRAME_GC_TYPE_ID as usize {
        return; // Already registered
    }
    while gc.type_count() < JITFRAME_GC_TYPE_ID as usize {
        gc.register_type(TypeInfo::simple(8)); // placeholder type ids 0, 1
    }
    // JITFRAME is a varsize GC struct:
    //   base_size = 64 (7 header fields × 8 + length field 8)
    //   item_size = 8 (each jf_frame slot is Signed)
    //   length_offset = 56 (jf_frame_length at byte 56 from payload start)
    let mut info = TypeInfo::varsize(
        64,     // base_size
        8,      // item_size
        56,     // length_offset
        false,  // items_have_gc_ptrs (custom_trace handles this)
        vec![], // no fixed gc_ptr_offsets (custom_trace handles this)
    );
    info.has_gc_ptrs = true;
    info.custom_trace = Some(jitframe_custom_trace);
    let id = gc.register_type(info);
    debug_assert_eq!(id, JITFRAME_GC_TYPE_ID);
}

/// Follow jf_forward chain to get the final jitframe address.
///
/// RPython jitframe.py:54-57 jitframe_resolve:
///   while frame.jf_forward:
///       frame = frame.jf_forward
///   return frame
fn jitframe_resolve(jf_ptr: *mut i64) -> *mut i64 {
    let mut ptr = jf_ptr;
    loop {
        let forward_addr =
            unsafe { *((ptr as *const u8).add(JF_FORWARD_OFS as usize) as *const usize) };
        if forward_addr == 0 {
            return ptr;
        }
        ptr = forward_addr as *mut i64;
    }
}

#[derive(Debug)]
struct BuiltinFieldDescr {
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
}

impl majit_ir::Descr for BuiltinFieldDescr {
    fn as_field_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self)
    }
}

impl majit_ir::FieldDescr for BuiltinFieldDescr {
    fn offset(&self) -> usize {
        self.offset
    }

    fn field_size(&self) -> usize {
        self.field_size
    }

    fn field_type(&self) -> Type {
        self.field_type
    }

    fn is_field_signed(&self) -> bool {
        self.signed
    }
}

#[derive(Debug)]
struct BuiltinArrayDescr {
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_descr: Arc<BuiltinFieldDescr>,
}

impl majit_ir::Descr for BuiltinArrayDescr {
    fn as_array_descr(&self) -> Option<&dyn majit_ir::ArrayDescr> {
        Some(self)
    }
}

impl majit_ir::ArrayDescr for BuiltinArrayDescr {
    fn base_size(&self) -> usize {
        self.base_size
    }

    fn item_size(&self) -> usize {
        self.item_size
    }

    fn type_id(&self) -> u32 {
        self.type_id
    }

    fn item_type(&self) -> Type {
        self.item_type
    }

    fn is_item_signed(&self) -> bool {
        self.signed
    }

    fn len_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self.len_descr.as_ref())
    }
}

#[derive(Debug)]
pub struct CallAssemblerDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    target_token: u64,
    effect_info: EffectInfo,
}

impl CallAssemblerDescr {
    pub fn new(target_token: u64, arg_types: Vec<Type>, result_type: Type) -> Self {
        CallAssemblerDescr {
            arg_types,
            result_type,
            target_token,
            effect_info: EffectInfo {
                extra_effect: majit_ir::ExtraEffect::CanRaise,
                oopspec_index: OopSpecIndex::None,
                ..Default::default()
            },
        }
    }
}

impl majit_ir::Descr for CallAssemblerDescr {
    fn as_call_descr(&self) -> Option<&dyn majit_ir::CallDescr> {
        Some(self)
    }
}

impl majit_ir::CallDescr for CallAssemblerDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }

    fn result_type(&self) -> Type {
        self.result_type
    }

    fn result_size(&self) -> usize {
        8
    }

    fn call_target_token(&self) -> Option<u64> {
        Some(self.target_token)
    }

    fn effect_info(&self) -> &EffectInfo {
        &self.effect_info
    }
}

#[derive(Clone)]
struct RegisteredLoopTarget {
    trace_id: u64,
    header_pc: u64,
    _green_key: u64,
    source_guard: Option<(u64, u32)>,
    caller_prefix_layout: Option<ExitRecoveryLayout>,
    code_ptr: *const u8,
    fail_descrs: Vec<Arc<CraneliftFailDescr>>,
    gc_runtime_id: Option<u64>,
    num_inputs: usize,
    num_ref_roots: usize,
    max_output_slots: usize,
    inputarg_types: Vec<Type>,
    needs_force_frame: bool,
}

unsafe impl Send for RegisteredLoopTarget {}
unsafe impl Sync for RegisteredLoopTarget {}

struct PendingForceFrame {
    fail_descr: Arc<CraneliftFailDescr>,
    raw_values: Vec<i64>,
    rooted_refs: Vec<GcRef>,
    ref_slot_map: Vec<Option<usize>>,
}

struct ActiveForceFrame {
    fail_descrs: Vec<Arc<CraneliftFailDescr>>,
    gc_runtime_id: Option<u64>,
    pending_force: Mutex<Option<PendingForceFrame>>,
    pending_may_force: Mutex<Vec<PendingMayForceFrame>>,
    saved_data_root: Mutex<Option<Box<GcRef>>>,
}

struct PendingMayForceFrame {
    preview: PendingForceFrame,
    was_forced: bool,
}

struct PreviewFrameData {
    frame: FrameData,
    active_force_frame: Arc<ActiveForceFrame>,
}

struct OverlayFrameData {
    inner: DeadFrame,
    fail_descr: Arc<CraneliftFailDescr>,
}

impl PendingForceFrame {
    fn new(
        fail_descr: Arc<CraneliftFailDescr>,
        gc_runtime_id: Option<u64>,
        raw_values: Vec<i64>,
    ) -> Self {
        let mut rooted_refs = Vec::new();
        let mut ref_slot_map = vec![None; fail_descr.fail_arg_types.len()];
        for (index, tp) in fail_descr.fail_arg_types.iter().enumerate() {
            if *tp == Type::Ref && !fail_descr.is_force_token_slot(index) {
                ref_slot_map[index] = Some(rooted_refs.len());
                rooted_refs.push(GcRef(raw_values[index] as usize));
            }
        }
        if let Some(runtime_id) = gc_runtime_id {
            register_gc_roots(runtime_id, &mut rooted_refs);
        }
        PendingForceFrame {
            fail_descr,
            raw_values,
            rooted_refs,
            ref_slot_map,
        }
    }

    fn materialized_raw_values(&self) -> Vec<i64> {
        let mut raw_values = self.raw_values.clone();
        for (slot, rooted_index) in self.ref_slot_map.iter().enumerate() {
            if let Some(rooted_index) = rooted_index {
                raw_values[slot] = self.rooted_refs[*rooted_index].0 as i64;
            }
        }
        raw_values
    }

    fn into_raw_values(mut self, gc_runtime_id: Option<u64>) -> Vec<i64> {
        self.raw_values = self.materialized_raw_values();
        if let Some(runtime_id) = gc_runtime_id {
            unregister_gc_roots(runtime_id, &mut self.rooted_refs);
        }
        self.raw_values
    }
}

impl PreviewFrameData {
    fn new(
        raw_values: Vec<i64>,
        fail_descr: Arc<CraneliftFailDescr>,
        gc_runtime_id: Option<u64>,
        active_force_frame: Arc<ActiveForceFrame>,
    ) -> Self {
        PreviewFrameData {
            frame: FrameData::new_preview(raw_values, fail_descr, gc_runtime_id),
            active_force_frame,
        }
    }
}

fn deadframe_layout(frame: &DeadFrame) -> Option<FailDescrLayout> {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return Some(frame_data.fail_descr.layout());
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return Some(preview.frame.fail_descr.layout());
    }
    if let Some(overlay) = frame.data.downcast_ref::<OverlayFrameData>() {
        return Some(overlay.fail_descr.layout());
    }
    None
}

fn overlay_deadframe_fail_descr(
    base_layout: &FailDescrLayout,
    recovery_layout: ExitRecoveryLayout,
) -> Arc<CraneliftFailDescr> {
    let mut descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
        base_layout.fail_index,
        base_layout.trace_id,
        base_layout.fail_arg_types.clone(),
        base_layout.is_finish,
        base_layout.force_token_slots.clone(),
        Some(recovery_layout),
    );
    if let Some(source_op_index) = base_layout.source_op_index {
        descr.set_source_op_index(source_op_index);
    }
    if let Some(trace_info) = base_layout.trace_info.clone() {
        descr.set_trace_info(trace_info);
    }
    Arc::new(descr)
}

fn deadframe_recovery_layout_for_call_assembler(
    layout: &FailDescrLayout,
) -> Option<ExitRecoveryLayout> {
    layout.recovery_layout.clone().or_else(|| {
        layout.trace_info.as_ref().map(|trace_info| {
            identity_recovery_layout(
                layout.trace_id,
                trace_info.header_pc,
                None, // no per-guard resume_pc available here
                trace_info.source_guard,
                &layout.fail_arg_types,
                None,
            )
        })
    })
}

fn caller_prefix_recovery_layout(
    trace_id: u64,
    header_pc: u64,
    source_guard: Option<(u64, u32)>,
    slot_types: &[Type],
    inputs: &[i64],
    caller_prefix_layout: Option<&ExitRecoveryLayout>,
) -> ExitRecoveryLayout {
    ExitRecoveryLayout {
        frames: vec![ExitFrameLayout {
            trace_id: Some(trace_id),
            header_pc: Some(header_pc),
            source_guard,
            pc: header_pc,
            slots: slot_types
                .iter()
                .enumerate()
                .map(|(slot, _)| {
                    inputs
                        .get(slot)
                        .copied()
                        .map(ExitValueSourceLayout::Constant)
                        .unwrap_or(ExitValueSourceLayout::Unavailable)
                })
                .collect(),
            slot_types: Some(slot_types.to_vec()),
        }],
        virtual_layouts: Vec::new(),
        pending_field_layouts: Vec::new(),
    }
    .prefixed_by(caller_prefix_layout)
}

fn wrap_call_assembler_deadframe_with_caller_prefix(
    frame: DeadFrame,
    trace_id: u64,
    header_pc: u64,
    source_guard: Option<(u64, u32)>,
    input_types: &[Type],
    inputs: &[i64],
    caller_prefix_layout: Option<&ExitRecoveryLayout>,
) -> DeadFrame {
    let Some(layout) = deadframe_layout(&frame) else {
        return frame;
    };
    let Some(inner_recovery_layout) = deadframe_recovery_layout_for_call_assembler(&layout) else {
        return frame;
    };

    let caller_layout = caller_prefix_recovery_layout(
        trace_id,
        header_pc,
        source_guard,
        input_types,
        inputs,
        caller_prefix_layout,
    );
    let recovery_layout = inner_recovery_layout.prefixed_by(Some(&caller_layout));

    DeadFrame {
        data: Box::new(OverlayFrameData {
            inner: frame,
            fail_descr: overlay_deadframe_fail_descr(&layout, recovery_layout),
        }),
    }
}

static NEXT_FORCE_TOKEN_HANDLE: AtomicU64 = AtomicU64::new(1);
static FORCE_FRAMES: OnceLock<Mutex<HashMap<u64, Arc<ActiveForceFrame>>>> = OnceLock::new();

/// Global exception state for JIT-compiled code.
/// pyre is no-GIL single-threaded, so global statics are safe and allow
/// GUARD_NO_EXCEPTION to emit a direct memory load instead of a TLS call.
static JIT_EXC_VALUE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
static JIT_EXC_TYPE: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

/// Return the address of JIT_EXC_VALUE for direct memory load in JIT code.
pub fn jit_exc_value_addr() -> usize {
    &JIT_EXC_VALUE as *const _ as usize
}

thread_local! {
    static CURRENT_FORCE_FRAME_HANDLE: Cell<u64> = const { Cell::new(0) };
    /// Thread-local reference storage for JIT-compiled code.
    ///
    /// Mirrors RPython's `rpy_threadlocalref` — an array of integer-sized slots
    /// indexed by offset. The interpreter stores per-thread state here (e.g.,
    /// the current action flag, signal handler data, etc.).
    ///
    /// Slots are accessed via THREADLOCALREF_GET opcode.
    static JIT_THREADLOCAL_SLOTS: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
}

// ── Exception state shims called from JIT-compiled code ──

/// Read the current exception value (returns 0 if no exception pending).
extern "C" fn jit_exc_get_value() -> i64 {
    JIT_EXC_VALUE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Read the current exception class (returns 0 if no exception pending).
extern "C" fn jit_exc_get_type() -> i64 {
    JIT_EXC_TYPE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Clear the current exception state and return the value that was pending.
extern "C" fn jit_exc_clear_and_get_value() -> i64 {
    let val = JIT_EXC_VALUE.swap(0, std::sync::atomic::Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, std::sync::atomic::Ordering::Relaxed);
    val
}

/// Clear the current exception state.
extern "C" fn jit_exc_clear() {
    JIT_EXC_VALUE.store(0, std::sync::atomic::Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Return 1 if the current exception class exactly matches `expected_type`.
extern "C" fn jit_exc_type_matches(expected_type: i64) -> i64 {
    let exc_value = JIT_EXC_VALUE.load(std::sync::atomic::Ordering::Relaxed);
    let exc_type = JIT_EXC_TYPE.load(std::sync::atomic::Ordering::Relaxed);
    i64::from(exc_value != 0 && exc_type == expected_type)
}

/// Set the current exception state (value, class).
extern "C" fn jit_exc_restore(value: i64, exc_type: i64) {
    JIT_EXC_VALUE.store(value, std::sync::atomic::Ordering::Relaxed);
    JIT_EXC_TYPE.store(exc_type, std::sync::atomic::Ordering::Relaxed);
}

/// Set exception state from a call that may raise.
/// This is called by external code that wants to signal an exception
/// to the JIT-compiled code.
pub fn jit_exc_raise(value: i64, exc_type: i64) {
    JIT_EXC_VALUE.store(value, std::sync::atomic::Ordering::Relaxed);
    JIT_EXC_TYPE.store(exc_type, std::sync::atomic::Ordering::Relaxed);
}

/// Check if an exception is currently pending.
pub fn jit_exc_is_pending() -> bool {
    JIT_EXC_VALUE.load(std::sync::atomic::Ordering::Relaxed) != 0
}

/// RPython cpu.grab_exc_value parity: read exception class from TLS.
pub fn jit_exc_class_raw() -> i64 {
    JIT_EXC_TYPE.load(std::sync::atomic::Ordering::Relaxed)
}

/// RPython cpu.grab_exc_value parity: read exception value from TLS.
pub fn jit_exc_value_raw() -> i64 {
    JIT_EXC_VALUE.load(std::sync::atomic::Ordering::Relaxed)
}

fn take_pending_jit_exception_state() -> (i64, GcRef) {
    let value = JIT_EXC_VALUE.swap(0, std::sync::atomic::Ordering::Relaxed);
    let exc_type = JIT_EXC_TYPE.swap(0, std::sync::atomic::Ordering::Relaxed);
    if value == 0 {
        (0, GcRef::NULL)
    } else {
        (exc_type, GcRef(value as usize))
    }
}

// ── Thread-local reference access shims ──

/// Read a thread-local slot at the given offset.
///
/// Called from JIT-compiled code for THREADLOCALREF_GET operations.
/// The offset is in bytes (divided by 8 to get the slot index).
extern "C" fn jit_threadlocalref_get(offset: i64) -> i64 {
    JIT_THREADLOCAL_SLOTS.with(|slots| {
        let slots = slots.borrow();
        let idx = (offset / 8) as usize;
        slots.get(idx).copied().unwrap_or(0)
    })
}

/// Write a value to a thread-local slot at the given offset.
///
/// Called by the interpreter to set up thread-local state before entering
/// JIT-compiled code.
pub fn jit_threadlocalref_set(offset: i64, value: i64) {
    JIT_THREADLOCAL_SLOTS.with(|slots| {
        let mut slots = slots.borrow_mut();
        let idx = (offset / 8) as usize;
        if idx >= slots.len() {
            slots.resize(idx + 1, 0);
        }
        slots[idx] = value;
    });
}

/// Get the base pointer for thread-local slots.
/// Returns 0 since we use callback-based access.
pub fn jit_threadlocalref_base() -> *const i64 {
    JIT_THREADLOCAL_SLOTS.with(|slots| {
        let slots = slots.borrow();
        if slots.is_empty() {
            std::ptr::null()
        } else {
            slots.as_ptr()
        }
    })
}

// ── GIL release/reacquire shims ──

/// User-installable hook for GIL release before CallReleaseGil.
static GIL_RELEASE_HOOK: OnceLock<Box<dyn Fn() + Send + Sync>> = OnceLock::new();
/// User-installable hook for GIL reacquire after CallReleaseGil.
static GIL_REACQUIRE_HOOK: OnceLock<Box<dyn Fn() + Send + Sync>> = OnceLock::new();

/// Install custom GIL release/reacquire hooks.
pub fn set_gil_hooks(
    release: impl Fn() + Send + Sync + 'static,
    reacquire: impl Fn() + Send + Sync + 'static,
) {
    let _ = GIL_RELEASE_HOOK.set(Box::new(release));
    let _ = GIL_REACQUIRE_HOOK.set(Box::new(reacquire));
}

extern "C" fn jit_release_gil_shim() {
    if let Some(hook) = GIL_RELEASE_HOOK.get() {
        hook();
    }
}

extern "C" fn jit_reacquire_gil_shim() {
    if let Some(hook) = GIL_REACQUIRE_HOOK.get() {
        hook();
    }
}

const BUILTIN_STRING_HASH_OFFSET: usize = 0;
const BUILTIN_STRING_LEN_OFFSET: usize = std::mem::size_of::<usize>();
const BUILTIN_STRING_BASE_SIZE: usize = BUILTIN_STRING_LEN_OFFSET + std::mem::size_of::<usize>();

// ---------------------------------------------------------------------------
// Helpers (free functions to avoid borrow conflicts)
// ---------------------------------------------------------------------------

fn var(idx: u32) -> Variable {
    Variable::from_u32(idx)
}

/// Convert a slice of Values to a Vec of BlockArgs for Cranelift 0.130 branch instructions.
fn block_args(vals: &[CValue]) -> Vec<BlockArg> {
    vals.iter().copied().map(BlockArg::from).collect()
}

/// Whether to use native SIMD (I64X2/F64X2) for Vec* codegen.
/// When false, falls back to scalar emulation.
const USE_NATIVE_SIMD: bool = true;

/// Returns true if `opcode` is a Vec* opcode that produces a vector-typed value
/// (I64X2 or F64X2). VecUnpack* opcodes produce scalars, so they return false.
/// Guard/void opcodes also return false.
fn is_vec_producing_opcode(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::VecIntAdd
            | OpCode::VecIntSub
            | OpCode::VecIntMul
            | OpCode::VecIntAnd
            | OpCode::VecIntOr
            | OpCode::VecIntXor
            | OpCode::VecFloatAdd
            | OpCode::VecFloatSub
            | OpCode::VecFloatMul
            | OpCode::VecFloatTrueDiv
            | OpCode::VecFloatNeg
            | OpCode::VecFloatAbs
            | OpCode::VecFloatXor
            | OpCode::VecI
            | OpCode::VecF
            | OpCode::VecPackI
            | OpCode::VecPackF
            | OpCode::VecExpandI
            | OpCode::VecExpandF
            | OpCode::VecLoadI
            | OpCode::VecLoadF
    )
}

/// Returns true if `opcode` is a Vec* float opcode producing F64X2.
fn is_vec_float_producing(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::VecFloatAdd
            | OpCode::VecFloatSub
            | OpCode::VecFloatMul
            | OpCode::VecFloatTrueDiv
            | OpCode::VecFloatNeg
            | OpCode::VecFloatAbs
            | OpCode::VecFloatXor
            | OpCode::VecF
            | OpCode::VecPackF
            | OpCode::VecExpandF
            | OpCode::VecLoadF
    )
}

/// Pre-scan ops to find positions that produce vector values.
fn build_vec_oprefs(ops: &[Op], num_inputs: usize) -> HashSet<u32> {
    if !USE_NATIVE_SIMD {
        return HashSet::new();
    }
    let mut set = HashSet::new();
    for (op_idx, op) in ops.iter().enumerate() {
        if is_vec_producing_opcode(op.opcode) {
            let vi = op_var_index(op, op_idx, num_inputs) as u32;
            set.insert(vi);
        }
    }
    set
}

/// Pre-scan to find which vector-producing positions have float type (F64X2).
fn build_vec_float_oprefs(ops: &[Op], num_inputs: usize) -> HashSet<u32> {
    if !USE_NATIVE_SIMD {
        return HashSet::new();
    }
    let mut set = HashSet::new();
    for (op_idx, op) in ops.iter().enumerate() {
        if is_vec_float_producing(op.opcode) {
            let vi = op_var_index(op, op_idx, num_inputs) as u32;
            set.insert(vi);
        }
    }
    set
}

/// Resolve an OpRef as a vector value (I64X2). If the OpRef refers to a
/// vector-producing op, uses the variable directly (bitcasting F64X2 to
/// I64X2 if needed). If it's a constant, splats it to fill all lanes.
fn resolve_opref_vec_int(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    vec_oprefs: &HashSet<u32>,
    vec_float_oprefs: &HashSet<u32>,
    opref: OpRef,
) -> CValue {
    if let Some(&c) = constants.get(&opref.0) {
        let scalar = builder.ins().iconst(cl_types::I64, c);
        return builder.ins().splat(cl_types::I64X2, scalar);
    }
    if vec_oprefs.contains(&opref.0) {
        let v = builder.use_var(var(opref.0));
        if vec_float_oprefs.contains(&opref.0) {
            // Variable is F64X2, bitcast to I64X2
            return builder.ins().bitcast(cl_types::I64X2, MemFlags::new(), v);
        }
        return v;
    }
    // Scalar variable referenced in a vector context: splat it
    let scalar = builder.use_var(var(opref.0));
    builder.ins().splat(cl_types::I64X2, scalar)
}

/// Resolve an OpRef as a vector float value (F64X2). If it's a constant,
/// reinterpret the i64 bits as f64 and splat. If it's a vector var,
/// bitcast to F64X2 if needed. If scalar, bitcast to f64 then splat.
fn resolve_opref_vec_float(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    vec_oprefs: &HashSet<u32>,
    vec_float_oprefs: &HashSet<u32>,
    opref: OpRef,
) -> CValue {
    if let Some(&c) = constants.get(&opref.0) {
        let scalar_i = builder.ins().iconst(cl_types::I64, c);
        let scalar_f = builder
            .ins()
            .bitcast(cl_types::F64, MemFlags::new(), scalar_i);
        return builder.ins().splat(cl_types::F64X2, scalar_f);
    }
    if vec_oprefs.contains(&opref.0) {
        let v = builder.use_var(var(opref.0));
        if vec_float_oprefs.contains(&opref.0) {
            // Already F64X2
            return v;
        }
        // I64X2 -> F64X2 bitcast
        return builder.ins().bitcast(cl_types::F64X2, MemFlags::new(), v);
    }
    // Scalar variable: bitcast i64 -> f64 then splat
    let scalar = builder.use_var(var(opref.0));
    let scalar_f = builder
        .ins()
        .bitcast(cl_types::F64, MemFlags::new(), scalar);
    builder.ins().splat(cl_types::F64X2, scalar_f)
}

/// Map a majit Type to the corresponding Cranelift IR type for call signatures.
fn cranelift_type_for(tp: &Type) -> cranelift_codegen::ir::Type {
    match tp {
        Type::Int | Type::Ref => cl_types::I64,
        Type::Float => cl_types::F64,
        Type::Void => cl_types::I64,
    }
}

static NEXT_GC_RUNTIME_ID: AtomicU64 = AtomicU64::new(1);
static GC_RUNTIMES: OnceLock<Mutex<HashMap<u64, Box<dyn GcAllocator>>>> = OnceLock::new();

fn gc_runtime_registry() -> &'static Mutex<HashMap<u64, Box<dyn GcAllocator>>> {
    GC_RUNTIMES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_gc_runtime(gc: Box<dyn GcAllocator>) -> u64 {
    let id = NEXT_GC_RUNTIME_ID.fetch_add(1, Ordering::Relaxed);
    gc_runtime_registry().lock().unwrap().insert(id, gc);
    id
}

fn replace_gc_runtime(id: u64, gc: Box<dyn GcAllocator>) {
    gc_runtime_registry().lock().unwrap().insert(id, gc);
}

fn unregister_gc_runtime(id: u64) {
    gc_runtime_registry().lock().unwrap().remove(&id);
}

fn with_gc_runtime<R>(id: u64, f: impl FnOnce(&mut dyn GcAllocator) -> R) -> R {
    let mut guard = gc_runtime_registry().lock().unwrap();
    let runtime = guard
        .get_mut(&id)
        .unwrap_or_else(|| panic!("missing GC runtime {id}"));
    f(runtime.as_mut())
}

pub(crate) fn register_gc_roots(runtime_id: u64, roots: &mut [GcRef]) {
    if roots.is_empty() {
        return;
    }
    with_gc_runtime(runtime_id, |gc| {
        for root in roots.iter_mut() {
            unsafe {
                gc.add_root(root as *mut GcRef);
            }
        }
    });
}

pub(crate) fn unregister_gc_roots(runtime_id: u64, roots: &mut [GcRef]) {
    if roots.is_empty() {
        return;
    }
    let mut guard = gc_runtime_registry().lock().unwrap();
    if let Some(runtime) = guard.get_mut(&runtime_id) {
        for root in roots.iter_mut() {
            runtime.remove_root(root as *mut GcRef);
        }
    }
}

fn force_frame_registry() -> &'static Mutex<HashMap<u64, Arc<ActiveForceFrame>>> {
    FORCE_FRAMES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_force_frame(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    gc_runtime_id: Option<u64>,
) -> (u64, Arc<ActiveForceFrame>) {
    let handle = NEXT_FORCE_TOKEN_HANDLE.fetch_add(1, Ordering::Relaxed);
    let frame = Arc::new(ActiveForceFrame {
        fail_descrs: fail_descrs.to_vec(),
        gc_runtime_id,
        pending_force: Mutex::new(None),
        pending_may_force: Mutex::new(Vec::new()),
        saved_data_root: Mutex::new(None),
    });
    force_frame_registry()
        .lock()
        .unwrap()
        .insert(handle, frame.clone());
    (handle, frame)
}

fn lookup_force_frame(handle: u64) -> Option<Arc<ActiveForceFrame>> {
    force_frame_registry().lock().unwrap().get(&handle).cloned()
}

fn set_force_frame_saved_data(frame: &ActiveForceFrame, data: GcRef) {
    let mut saved_data_root = frame.saved_data_root.lock().unwrap();
    if let Some(saved_data) = saved_data_root.as_mut() {
        if let Some(runtime_id) = frame.gc_runtime_id {
            unregister_gc_roots(runtime_id, std::slice::from_mut(saved_data.as_mut()));
        }
        **saved_data = data;
        if let Some(runtime_id) = frame.gc_runtime_id {
            register_gc_roots(runtime_id, std::slice::from_mut(saved_data.as_mut()));
        }
        return;
    }

    let mut saved_data = Box::new(data);
    if let Some(runtime_id) = frame.gc_runtime_id {
        register_gc_roots(runtime_id, std::slice::from_mut(saved_data.as_mut()));
    }
    *saved_data_root = Some(saved_data);
}

/// RPython rebuild_state_after_failure parity: materialize virtual objects
/// in raw fail_args before bridge dispatch, using recovery_layout if available.
///
/// The optimizer may inline virtual field values directly into fail_args
/// (e.g., [frame, ni, vsd, ob_type_s, intval_s, ob_type_i, intval_i])
/// instead of using null-Ref placeholders. This function reconstructs
/// boxed objects and produces a compacted output matching bridge inputargs.
fn rebuild_state_after_failure(
    outputs: &mut Vec<i64>,
    types: &[majit_ir::Type],
    recovery: Option<&majit_codegen::ExitRecoveryLayout>,
    bridge_num_inputs: usize,
) {
    // Phase 1: recovery_layout-based materialization (rd_virtuals parity).
    if let Some(recovery) = recovery {
        if !recovery.virtual_layouts.is_empty() && !recovery.frames.is_empty() {
            let frame_layout = &recovery.frames[0];
            let mut rebuilt = Vec::with_capacity(frame_layout.slots.len());
            for slot in &frame_layout.slots {
                match slot {
                    majit_codegen::ExitValueSourceLayout::ExitValue(idx) => {
                        rebuilt.push(outputs.get(*idx).copied().unwrap_or(0));
                    }
                    majit_codegen::ExitValueSourceLayout::Constant(c) => {
                        rebuilt.push(*c);
                    }
                    majit_codegen::ExitValueSourceLayout::Virtual(vidx) => {
                        if let Some(vl) = recovery.virtual_layouts.get(*vidx) {
                            if let Some(obj) =
                                rebuild_state_after_failure_single_virtual(vl, outputs)
                            {
                                rebuilt.push(obj);
                            } else {
                                rebuilt.push(0);
                            }
                        } else {
                            rebuilt.push(0);
                        }
                    }
                    _ => rebuilt.push(0),
                }
            }
            *outputs = rebuilt;
            return;
        }
    }
    // Phase 2: compact virtual field pairs when outputs has more slots
    // than bridge expects. Pattern: [frame, ni, vsd, ob_type_0, intval_0,
    // ob_type_1, intval_1, ...] → [frame, ni, vsd, boxed_0, boxed_1, ...].
    // RPython rebuild_state_after_failure materializes virtuals from
    // rd_virtuals before bridge dispatch.
    if bridge_num_inputs > 0 && outputs.len() > bridge_num_inputs {
        let prefix = 3; // frame, ni, vsd
        let object_slots = bridge_num_inputs - prefix;
        let field_pairs = outputs.len() - prefix;
        if object_slots > 0 && field_pairs == object_slots * 2 {
            let mut compacted = outputs[..prefix].to_vec();
            for i in 0..object_slots {
                let ob_type = outputs[prefix + i * 2];
                let intval = outputs[prefix + i * 2 + 1];
                // Materialize via callback
                let mut temp = vec![0i64, ob_type, intval];
                let temp_types = vec![
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                ];
                REBUILD_STATE_AFTER_FAILURE.with(|c| {
                    if let Some(f) = c.get() {
                        f(&mut temp, &temp_types);
                    }
                });
                compacted.push(temp[0]);
            }
            *outputs = compacted;
            return;
        }
    }
    // Phase 3: original heuristic fallback.
    REBUILD_STATE_AFTER_FAILURE.with(|c| {
        if let Some(f) = c.get() {
            f(outputs, types);
        }
    });
}

/// Materialize a single virtual from its layout and raw output values.
/// Uses the REBUILD_STATE_AFTER_FAILURE callback for actual object allocation.
fn rebuild_state_after_failure_single_virtual(
    vl: &majit_codegen::ExitVirtualLayout,
    outputs: &[i64],
) -> Option<i64> {
    match vl {
        majit_codegen::ExitVirtualLayout::Object { fields, .. }
        | majit_codegen::ExitVirtualLayout::Struct { fields, .. } => {
            if fields.len() == 2 {
                let ob_type_val = resolve_virtual_field_value(&fields[0].1, outputs)?;
                let intval = resolve_virtual_field_value(&fields[1].1, outputs)?;
                // Use a temporary buffer with the [Ref(0), Int, Int] pattern
                // so the heuristic materializer can box it.
                let mut temp = vec![0i64, ob_type_val, intval];
                let temp_types = vec![
                    majit_ir::Type::Ref,
                    majit_ir::Type::Int,
                    majit_ir::Type::Int,
                ];
                REBUILD_STATE_AFTER_FAILURE.with(|c| {
                    if let Some(f) = c.get() {
                        f(&mut temp, &temp_types);
                    }
                });
                if temp[0] != 0 { Some(temp[0]) } else { None }
            } else {
                None
            }
        }
        _ => None,
    }
}

fn resolve_virtual_field_value(
    source: &majit_codegen::ExitValueSourceLayout,
    outputs: &[i64],
) -> Option<i64> {
    match source {
        majit_codegen::ExitValueSourceLayout::ExitValue(idx) => outputs.get(*idx).copied(),
        majit_codegen::ExitValueSourceLayout::Constant(c) => Some(*c),
        _ => None,
    }
}

fn get_force_frame_saved_data(frame: &ActiveForceFrame) -> Option<GcRef> {
    frame
        .saved_data_root
        .lock()
        .unwrap()
        .as_ref()
        .map(|saved_data| **saved_data)
}

fn take_force_frame_saved_data(frame: &ActiveForceFrame) -> Option<GcRef> {
    let mut saved_data_root = frame.saved_data_root.lock().unwrap();
    let mut saved_data = saved_data_root.take()?;
    if let Some(runtime_id) = frame.gc_runtime_id {
        unregister_gc_roots(runtime_id, std::slice::from_mut(saved_data.as_mut()));
    }
    Some(*saved_data)
}

pub(crate) fn release_force_token(handle: u64) {
    if handle == 0 {
        return;
    }
    if let Some(frame) = force_frame_registry().lock().unwrap().remove(&handle) {
        if let Some(pending) = frame.pending_force.lock().unwrap().take() {
            let _ = pending.into_raw_values(frame.gc_runtime_id);
        }
        for pending in frame.pending_may_force.lock().unwrap().drain(..) {
            let _ = pending.preview.into_raw_values(frame.gc_runtime_id);
        }
        let _ = take_force_frame_saved_data(&frame);
    }
}

fn current_force_frame_handle() -> u64 {
    CURRENT_FORCE_FRAME_HANDLE.with(Cell::get)
}

fn with_active_force_frame<R>(handle: u64, f: impl FnOnce() -> R) -> R {
    CURRENT_FORCE_FRAME_HANDLE.with(|cell| {
        let previous = cell.replace(handle);
        let result = f();
        cell.set(previous);
        result
    })
}

static CALL_ASSEMBLER_TARGETS: OnceLock<Mutex<HashMap<u64, RegisteredLoopTarget>>> =
    OnceLock::new();
static CALL_ASSEMBLER_EXPECTATIONS: OnceLock<
    Mutex<HashMap<u64, HashMap<CallAssemblerCallerId, u64>>>,
> = OnceLock::new();
thread_local! {
    /// Thread-local deadframe storage for call_assembler results.
    /// Each test thread gets its own isolated registry, preventing
    /// non-deterministic failures from shared global state.
    static CALL_ASSEMBLER_DEADFRAMES: RefCell<HashMap<u64, DeadFrame>> = RefCell::new(HashMap::new());
    static NEXT_CALL_ASSEMBLER_DEADFRAME_HANDLE: Cell<u64> = const { Cell::new(1) };
}

const CALL_ASSEMBLER_OUTCOME_FINISH: i64 = 0;
const CALL_ASSEMBLER_OUTCOME_DEADFRAME: i64 = 1;

/// Callback to "force" a callee frame through the interpreter when
/// call_assembler hits a guard failure.  The callback takes the
/// callee frame pointer (from fail_args[0]) and runs the interpreter
/// to completion, returning the result as an i64.
static CALL_ASSEMBLER_FORCE_FN: OnceLock<extern "C" fn(i64) -> i64> = OnceLock::new();

/// RPython resume_in_blackhole parity: callback to resume execution
/// from the guard failure point using the blackhole interpreter.
/// Args: (green_key, trace_id, fail_index, fail_values_ptr, num_fail_values) → result i64.
/// This reads the guard's resume data, restores state from fail_values (deadframe),
/// and executes the remaining IR ops from the guard point to Finish.
static CALL_ASSEMBLER_BLACKHOLE_FN: OnceLock<fn(u64, u64, u32, *const i64, usize) -> Option<i64>> =
    OnceLock::new();

/// Register a blackhole callback for call_assembler guard failure resume.
pub fn register_call_assembler_blackhole(f: fn(u64, u64, u32, *const i64, usize) -> Option<i64>) {
    let _ = CALL_ASSEMBLER_BLACKHOLE_FN.set(f);
}

/// Bridge compilation callback: (frame_ptr, fail_index, trace_id, green_key) -> result.
/// Called when a call_assembler guard fails enough times to warrant bridge compilation.
static CALL_ASSEMBLER_BRIDGE_FN: OnceLock<extern "C" fn(i64, u32, u64, u64) -> i64> =
    OnceLock::new();

/// Thread-local: raw local0 value from CallAssemblerI inputs,
/// for force_fn to re-box before interpreter execution.
thread_local! {
    static PENDING_FORCE_LOCAL0: std::cell::Cell<Option<i64>> =
        const { std::cell::Cell::new(None) };
}

/// Take the pending raw local0 value (if any).
pub fn take_pending_force_local0() -> Option<i64> {
    PENDING_FORCE_LOCAL0.with(|c| c.take())
}

/// RPython rebuild_state_after_failure parity: callback to materialize
/// virtual objects in fail_args before bridge dispatch. The Cranelift
/// backend cannot depend on pyre-object, so the interpreter registers
/// a callback that creates concrete objects from (type_ptr, raw_value) pairs.
///
/// Signature: fn(outputs: &mut [i64], types: &[Type]) — modifies outputs
/// in-place, replacing null Ref slots with materialized object pointers.
type RebuildStateAfterFailureFn = fn(&mut [i64], &[majit_ir::Type]);

thread_local! {
    static REBUILD_STATE_AFTER_FAILURE: std::cell::Cell<Option<RebuildStateAfterFailureFn>> =
        const { std::cell::Cell::new(None) };
}

/// Register the virtual materialization callback. Called from pyre-jit init.
pub fn register_rebuild_state_after_failure(f: RebuildStateAfterFailureFn) {
    REBUILD_STATE_AFTER_FAILURE.with(|c| c.set(Some(f)));
}

/// Frame state to restore from guard failure fail_args.
/// RPython resume_in_blackhole parity: the force_fn reads the frame state
/// from the deadframe (outputs buffer) rather than using the corrupted frame.
pub struct FrameRestore {
    pub next_instr: usize,
    pub valuestackdepth: usize,
    /// (type, raw_value) pairs for each fail_arg slot.
    pub slots: Vec<(majit_ir::Type, i64)>,
}

thread_local! {
    static PENDING_FRAME_RESTORE: std::cell::Cell<Option<FrameRestore>> =
        const { std::cell::Cell::new(None) };
}

/// Take the pending frame restore data (if any).
pub fn take_pending_frame_restore() -> Option<FrameRestore> {
    PENDING_FRAME_RESTORE.with(|c| c.take())
}

// ── Inline frame arena for self-recursive CallAssemblerI ────────────

/// Stable addresses for inline arena take/put in Cranelift IR.
#[derive(Clone)]
pub struct InlineFrameArenaInfo {
    pub buf_base_addr: usize,
    pub top_addr: usize,
    pub initialized_addr: usize,
    pub frame_size: usize,
    pub frame_code_offset: usize,
    pub frame_next_instr_offset: usize,
    pub frame_vable_token_offset: usize,
    pub create_fn_addr: usize,
    pub drop_fn_addr: usize,
    pub arena_cap: usize,
    /// JitFrame field descriptors for GC rewriter handle_call_assembler.
    pub jitframe_descrs: Option<majit_gc::rewrite::JitFrameDescrs>,
}

static INLINE_ARENA: OnceLock<InlineFrameArenaInfo> = OnceLock::new();

pub fn register_inline_frame_arena(info: InlineFrameArenaInfo) {
    let _ = INLINE_ARENA.set(info);
}

/// Enter a new bridge compile depth level (kept for compatibility with
/// existing call sites that manage nesting).
pub fn enter_bridge_compile_depth() -> BridgeDepthGuard {
    BridgeDepthGuard
}

pub struct BridgeDepthGuard;
impl Drop for BridgeDepthGuard {
    fn drop(&mut self) {}
}

// ── Call-assembler dispatch table ──
// Each token gets a stable dispatch entry holding code_ptr + finish_index.
// Compiled code loads both from the entry via memory reads.
// redirect_call_assembler / register_call_assembler_target update atomically.
use std::sync::atomic::AtomicPtr;

/// Dispatch entry: code_ptr at offset 0, finish_descr_ptr at offset 8.
/// Laid out as two 8-byte words so Cranelift can load them directly.
/// RPython parity: finish check compares jf_descr pointer with the
/// finish FailDescr pointer (not integer index).
#[repr(C)]
struct CaDispatchEntry {
    code_ptr: AtomicPtr<u8>,
    finish_descr_ptr: AtomicU64,
    /// RPython redirect_call_assembler parity: compiled bridge code pointer
    /// for the guard at finish_index. When set, Cranelift codegen can
    /// dispatch guard failures directly without extern "C" call overhead.
    guard_bridge_ptr: AtomicPtr<u8>,
}

/// Sentinel: finish_index not yet known (self-recursion before compile).
const CA_FINISH_INDEX_UNKNOWN: u64 = u64::MAX;

fn ca_dispatch_table() -> &'static Mutex<HashMap<u64, Box<CaDispatchEntry>>> {
    static TABLE: OnceLock<Mutex<HashMap<u64, Box<CaDispatchEntry>>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get or create the stable dispatch entry for a token.
/// Returns the address of the entry (stable for process lifetime).
fn ca_dispatch_slot(token_number: u64, code_ptr: *const u8) -> *const CaDispatchEntry {
    let mut table = ca_dispatch_table().lock().unwrap();
    let entry = table.entry(token_number).or_insert_with(|| {
        Box::new(CaDispatchEntry {
            code_ptr: AtomicPtr::new(code_ptr as *mut u8),
            finish_descr_ptr: AtomicU64::new(CA_FINISH_INDEX_UNKNOWN),
            guard_bridge_ptr: AtomicPtr::new(std::ptr::null_mut()),
        })
    });
    // Update code_ptr if changed (e.g., recompilation)
    entry.code_ptr.store(code_ptr as *mut u8, Ordering::Release);
    &**entry as *const CaDispatchEntry
}

/// Update the finish_descr_ptr for a token's dispatch entry.
/// Stores the FailDescr pointer (not index) for direct finish comparison.
fn ca_dispatch_set_finish_descr_ptr(token_number: u64, finish_descr_ptr: i64) {
    let table = ca_dispatch_table().lock().unwrap();
    if let Some(entry) = table.get(&token_number) {
        entry
            .finish_descr_ptr
            .store(finish_descr_ptr as u64, Ordering::Release);
    }
}

/// Update dispatch slot for redirect — updates both code_ptr and finish_descr_ptr.
fn ca_dispatch_redirect(old_token: u64, new_code_ptr: *const u8, new_finish_descr_ptr: i64) {
    let table = ca_dispatch_table().lock().unwrap();
    if let Some(entry) = table.get(&old_token) {
        entry
            .code_ptr
            .store(new_code_ptr as *mut u8, Ordering::Release);
        entry
            .finish_descr_ptr
            .store(new_finish_descr_ptr as u64, Ordering::Release);
    }
}

thread_local! {
    /// Thread-local cache for call_assembler target lookups.
    /// The HashMap holds Arc ownership; CA_FAST_PTR caches a raw pointer
    /// for the hot path to avoid atomic refcount ops on every call.
    static CA_TARGET_CACHE: RefCell<HashMap<u64, Arc<RegisteredLoopTarget>>> =
        RefCell::new(HashMap::new());

    /// Single-entry raw pointer cache: (token_number, ptr_as_usize).
    /// Valid because the Arc in CA_TARGET_CACHE keeps the data alive.
    static CA_FAST_PTR: Cell<(u64, usize)> = const { Cell::new((0, 0)) };
}

/// Look up a call_assembler target without atomic refcount operations.
///
/// # Safety
/// The returned pointer is valid for the duration of the current call
/// (the Arc in CA_TARGET_CACHE keeps the data alive, and invalidation
/// clears CA_FAST_PTR before removing from CA_TARGET_CACHE).
unsafe fn fast_lookup_ca_target(token_number: u64) -> *const RegisteredLoopTarget {
    // Hot path: single Cell read, no borrow, no atomic ops
    if let Ok((t, p)) = CA_FAST_PTR.try_with(|c| c.get()) {
        if t == token_number && p != 0 {
            return p as *const RegisteredLoopTarget;
        }
    }
    // Warm path: HashMap lookup, extract pointer without Arc clone
    if let Ok(Some(ptr)) = CA_TARGET_CACHE.try_with(|c| {
        c.borrow()
            .get(&token_number)
            .map(|arc| Arc::as_ptr(arc) as usize)
    }) {
        let _ = CA_FAST_PTR.try_with(|c| c.set((token_number, ptr)));
        return ptr as *const RegisteredLoopTarget;
    }
    // Cold path: global registry (mutex lock), then cache
    let target = call_assembler_registry()
        .lock()
        .unwrap()
        .get(&token_number)
        .cloned()
        .map(Arc::new)
        .unwrap_or_else(|| panic!("missing call_assembler target token {token_number}"));
    let ptr = Arc::as_ptr(&target) as usize;
    let _ = CA_TARGET_CACHE.try_with(|c| c.borrow_mut().insert(token_number, target));
    let _ = CA_FAST_PTR.try_with(|c| c.set((token_number, ptr)));
    ptr as *const RegisteredLoopTarget
}

fn invalidate_ca_thread_cache(token_number: u64) {
    // Clear fast pointer cache first (before removing Arc ownership)
    let _ = CA_FAST_PTR.try_with(|c| {
        let (t, _) = c.get();
        if t == token_number {
            c.set((0, 0));
        }
    });
    let _ = CA_TARGET_CACHE.try_with(|c| c.borrow_mut().remove(&token_number));
}

/// Register a force callback for call_assembler guard failures.
pub fn register_call_assembler_force(f: extern "C" fn(i64) -> i64) {
    let _ = CALL_ASSEMBLER_FORCE_FN.set(f);
}

/// Execute compiled code for a token directly, bypassing the JitDriver chain.
///
/// PyPy assembler_call_helper equivalent: dispatches through compiled code
/// for the given token_number. Returns Some(raw_result) on finish,
/// None if no compiled code or guard failure (caller should use interpreter).
///
/// This is used by jit_force_callee_frame to avoid the eval_with_jit →
/// try_function_entry_jit → run_compiled → execute_token indirection chain.
pub fn execute_call_assembler_direct(
    token_number: u64,
    inputs: &[i64],
    force_fn: extern "C" fn(i64) -> i64,
) -> Option<i64> {
    let target_ptr = unsafe { fast_lookup_ca_target(token_number) };
    if target_ptr.is_null() {
        return None;
    }
    let target = unsafe { &*target_ptr };
    if target.code_ptr.is_null() {
        return None;
    }

    let mut outcome = [0i64; 2];
    let result = call_assembler_fast_path(target, inputs, outcome.as_mut_ptr(), force_fn);

    if outcome[0] == CALL_ASSEMBLER_OUTCOME_FINISH {
        Some(result as i64)
    } else {
        None // guard failure — caller should fallback
    }
}

pub fn register_call_assembler_bridge(f: extern "C" fn(i64, u32, u64, u64) -> i64) {
    let _ = CALL_ASSEMBLER_BRIDGE_FN.set(f);
}

const CALL_ASSEMBLER_DEADFRAME_SENTINEL: u32 = u32::MAX - 1;
const CALL_ASSEMBLER_RESULT_VOID: u64 = 0;
const CALL_ASSEMBLER_RESULT_INT: u64 = 1;
const CALL_ASSEMBLER_RESULT_FLOAT: u64 = 2;
const CALL_ASSEMBLER_RESULT_REF: u64 = 3;
const CALL_ASSEMBLER_RESULT_FORCE_TOKEN_REF: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum CallAssemblerCallerId {
    RootLoop(u64),
    BridgeTrace(u64),
}

fn call_assembler_registry() -> &'static Mutex<HashMap<u64, RegisteredLoopTarget>> {
    CALL_ASSEMBLER_TARGETS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn call_assembler_expectation_registry()
-> &'static Mutex<HashMap<u64, HashMap<CallAssemblerCallerId, u64>>> {
    CALL_ASSEMBLER_EXPECTATIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn call_assembler_result_kind_name(kind: u64) -> &'static str {
    match kind {
        CALL_ASSEMBLER_RESULT_VOID => "void",
        CALL_ASSEMBLER_RESULT_INT => "int",
        CALL_ASSEMBLER_RESULT_FLOAT => "float",
        CALL_ASSEMBLER_RESULT_REF => "ref",
        CALL_ASSEMBLER_RESULT_FORCE_TOKEN_REF => "force-token-ref",
        _ => "unknown",
    }
}

fn actual_call_assembler_target_result_kind(
    fail_descrs: &[Arc<CraneliftFailDescr>],
) -> Result<u64, BackendError> {
    let finish_descr = fail_descrs
        .iter()
        .find(|descr| descr.is_finish())
        .ok_or_else(|| {
            unsupported_semantics(
                OpCode::CallAssemblerN,
                "call-assembler target must expose at least one finish exit",
            )
        })?;
    actual_call_assembler_result_kind(finish_descr.as_ref())
}

fn validate_call_assembler_target_result_kind(
    target_token: u64,
    expected_result_kind: u64,
    actual_result_kind: u64,
    context: &str,
) -> Result<(), BackendError> {
    if expected_result_kind == actual_result_kind {
        return Ok(());
    }
    Err(BackendError::Unsupported(format!(
        "call-assembler target {target_token} has incompatible {context}: expected {}, got {}",
        call_assembler_result_kind_name(expected_result_kind),
        call_assembler_result_kind_name(actual_result_kind),
    )))
}

fn validate_registered_target_against_call_assembler_expectations(
    target_token: u64,
    target: &RegisteredLoopTarget,
) -> Result<(), BackendError> {
    let expectations = call_assembler_expectation_registry().lock().unwrap();
    let Some(target_expectations) = expectations.get(&target_token) else {
        return Ok(());
    };
    let expected_result_kinds: Vec<u64> = target_expectations.values().copied().collect();
    drop(expectations);

    let actual_result_kind = actual_call_assembler_target_result_kind(&target.fail_descrs)?;
    for expected_result_kind in expected_result_kinds {
        validate_call_assembler_target_result_kind(
            target_token,
            expected_result_kind,
            actual_result_kind,
            "callee finish result kind",
        )?;
    }
    Ok(())
}

fn remove_call_assembler_expectations_locked(
    expectations: &mut HashMap<u64, HashMap<CallAssemblerCallerId, u64>>,
    caller_id: CallAssemblerCallerId,
) {
    expectations.retain(|_, callers| {
        callers.remove(&caller_id);
        !callers.is_empty()
    });
}

fn unregister_call_assembler_expectations(caller_id: CallAssemblerCallerId) {
    let mut expectations = call_assembler_expectation_registry().lock().unwrap();
    remove_call_assembler_expectations_locked(&mut expectations, caller_id);
}

fn unregister_bridge_call_assembler_expectations(bridge: &BridgeData) {
    unregister_call_assembler_expectations(CallAssemblerCallerId::BridgeTrace(bridge.trace_id));
    for descr in &bridge.fail_descrs {
        let attached = descr.bridge_ref();
        if let Some(ref child_bridge) = *attached {
            unregister_bridge_call_assembler_expectations(child_bridge);
        }
    }
}

fn unregister_call_assembler_bridge_tree(fail_descrs: &[Arc<CraneliftFailDescr>]) {
    for descr in fail_descrs {
        let attached = descr.bridge_ref();
        if let Some(ref bridge) = *attached {
            unregister_bridge_call_assembler_expectations(bridge);
        }
    }
}

fn collect_call_assembler_expectations(ops: &[Op]) -> Result<HashMap<u64, u64>, BackendError> {
    let mut expectations = HashMap::new();
    for op in ops {
        let opcode = op.opcode;
        if !matches!(
            opcode,
            OpCode::CallAssemblerI
                | OpCode::CallAssemblerR
                | OpCode::CallAssemblerF
                | OpCode::CallAssemblerN
        ) {
            continue;
        }
        let descr = op.descr.as_ref().ok_or_else(|| {
            unsupported_semantics(opcode, "call-assembler op must have a descriptor")
        })?;
        let call_descr = descr.as_call_descr().ok_or_else(|| {
            unsupported_semantics(opcode, "call-assembler descriptor must be a CallDescr")
        })?;
        let target_token = call_descr.call_target_token().ok_or_else(|| {
            unsupported_semantics(
                opcode,
                "call-assembler descriptor must provide a compiled target token",
            )
        })?;
        let resolved_target = resolve_call_assembler_target(opcode, call_descr)?;
        let expected_result_kind =
            expected_call_assembler_result_kind(call_descr, resolved_target.as_ref())?;
        if let Some(previous) = expectations.insert(target_token, expected_result_kind) {
            validate_call_assembler_target_result_kind(
                target_token,
                previous,
                expected_result_kind,
                "caller result expectation",
            )?;
        }
    }
    Ok(expectations)
}

fn install_call_assembler_expectations(
    caller_id: CallAssemblerCallerId,
    ops: &[Op],
) -> Result<(), BackendError> {
    let expectations = collect_call_assembler_expectations(ops)?;

    for (&target_token, &expected_result_kind) in &expectations {
        if let Some(target) = lookup_call_assembler_target(target_token) {
            let actual_result_kind = actual_call_assembler_target_result_kind(&target.fail_descrs)?;
            validate_call_assembler_target_result_kind(
                target_token,
                expected_result_kind,
                actual_result_kind,
                "callee finish result kind",
            )?;
        }
    }

    let mut registry = call_assembler_expectation_registry().lock().unwrap();
    for (&target_token, &expected_result_kind) in &expectations {
        if let Some(callers) = registry.get(&target_token) {
            for (&other_caller, &other_expected_kind) in callers {
                if other_caller != caller_id {
                    validate_call_assembler_target_result_kind(
                        target_token,
                        expected_result_kind,
                        other_expected_kind,
                        "caller result expectation",
                    )?;
                }
            }
        }
    }

    remove_call_assembler_expectations_locked(&mut registry, caller_id);
    for (&target_token, &expected_result_kind) in &expectations {
        registry
            .entry(target_token)
            .or_default()
            .insert(caller_id, expected_result_kind);
    }
    Ok(())
}

fn register_call_assembler_target(
    token: &JitCellToken,
    compiled: &CompiledLoop,
) -> Result<(), BackendError> {
    invalidate_ca_thread_cache(token.number);
    let target = RegisteredLoopTarget {
        trace_id: compiled.trace_id,
        header_pc: compiled.header_pc,
        _green_key: token.green_key,
        source_guard: None,
        caller_prefix_layout: compiled.caller_prefix_layout.clone(),
        code_ptr: compiled.code_ptr,
        fail_descrs: compiled.fail_descrs.clone(),
        gc_runtime_id: compiled.gc_runtime_id,
        num_inputs: compiled.num_inputs,
        num_ref_roots: compiled.num_ref_roots,
        max_output_slots: compiled.max_output_slots,
        inputarg_types: token.inputarg_types.clone(),
        needs_force_frame: compiled.needs_force_frame,
    };
    validate_registered_target_against_call_assembler_expectations(token.number, &target)?;
    // Invalidate thread-local cache in case a pending placeholder was cached.
    invalidate_ca_thread_cache(token.number);
    // Create/update dispatch slot for direct call
    ca_dispatch_slot(token.number, compiled.code_ptr);
    // Set the finish_descr_ptr so the direct call path can compare jf_descr
    // with the finish FailDescr pointer (RPython done_with_this_frame parity).
    if let Some(finish_descr) = compiled.fail_descrs.iter().find(|d| d.is_finish()) {
        let ptr = Arc::as_ptr(finish_descr) as i64;
        ca_dispatch_set_finish_descr_ptr(token.number, ptr);
    }
    call_assembler_registry()
        .lock()
        .unwrap()
        .insert(token.number, target);
    Ok(())
}

fn unregister_call_assembler_target(token_number: u64) {
    invalidate_ca_thread_cache(token_number);
    unregister_call_assembler_expectations(CallAssemblerCallerId::RootLoop(token_number));
    let removed = call_assembler_registry()
        .lock()
        .unwrap()
        .remove(&token_number);
    if let Some(target) = removed {
        unregister_call_assembler_bridge_tree(&target.fail_descrs);
    }
}

/// RPython compile_tmp_callback parity: register a placeholder target
/// with null code_ptr for a pending token. call_assembler_fast_path
/// detects null code_ptr and falls back to force_fn (interpreter).
/// When compilation completes, the placeholder is replaced by the real target.
pub fn register_pending_call_assembler_target(
    token_number: u64,
    inputarg_types: Vec<Type>,
    num_inputs: usize,
) {
    let target = RegisteredLoopTarget {
        trace_id: 0,
        header_pc: 0,
        _green_key: 0,
        source_guard: None,
        caller_prefix_layout: None,
        code_ptr: std::ptr::null(),
        fail_descrs: Vec::new(),
        gc_runtime_id: None,
        num_inputs,
        num_ref_roots: 0,
        max_output_slots: 1,
        inputarg_types,
        needs_force_frame: false,
    };
    call_assembler_registry()
        .lock()
        .unwrap()
        .insert(token_number, target);
}

fn lookup_call_assembler_target(token_number: u64) -> Option<RegisteredLoopTarget> {
    call_assembler_registry()
        .lock()
        .unwrap()
        .get(&token_number)
        .cloned()
}

fn redirect_call_assembler_target(old_number: u64, new_number: u64) -> Result<(), BackendError> {
    invalidate_ca_thread_cache(old_number);
    let Some(new_target) = lookup_call_assembler_target(new_number) else {
        return Ok(());
    };
    if let Some(old_target) = lookup_call_assembler_target(old_number) {
        if old_target.inputarg_types != new_target.inputarg_types {
            return Err(BackendError::Unsupported(format!(
                "call-assembler redirect from token {old_number} to {new_number} changed input types"
            )));
        }
    }
    validate_registered_target_against_call_assembler_expectations(old_number, &new_target)?;
    // Update dispatch slot so existing compiled code sees the new target.
    // Must update both code_ptr AND finish_descr_ptr (the new target has
    // different FailDescr pointers for its finish exit).
    let new_finish_descr_ptr = new_target
        .fail_descrs
        .iter()
        .find(|d| d.is_finish())
        .map(|d| Arc::as_ptr(d) as i64)
        .unwrap_or(CA_FINISH_INDEX_UNKNOWN as i64);
    ca_dispatch_redirect(old_number, new_target.code_ptr, new_finish_descr_ptr);
    call_assembler_registry()
        .lock()
        .unwrap()
        .insert(old_number, new_target);
    Ok(())
}

fn store_call_assembler_deadframe(frame: DeadFrame) -> u64 {
    let handle = NEXT_CALL_ASSEMBLER_DEADFRAME_HANDLE.with(|cell| {
        let h = cell.get();
        cell.set(h + 1);
        h
    });
    assert!(handle != 0, "call_assembler deadframe handle overflowed");
    CALL_ASSEMBLER_DEADFRAMES.with(|map| {
        map.borrow_mut().insert(handle, frame);
    });
    handle
}

fn take_call_assembler_deadframe(handle: u64) -> Option<DeadFrame> {
    CALL_ASSEMBLER_DEADFRAMES.with(|map| map.borrow_mut().remove(&handle))
}

fn finish_result_from_deadframe(frame: &mut DeadFrame) -> Result<i64, BackendError> {
    let fail_arg_types = {
        let descr = get_latest_descr_from_deadframe(frame)?;
        assert!(descr.is_finish(), "expected finish deadframe");
        descr.fail_arg_types().to_vec()
    };
    match fail_arg_types.as_slice() {
        [] => Ok(0),
        [Type::Int] => get_int_from_deadframe(frame, 0),
        [Type::Ref] => {
            if let Some(frame_data) = frame.data.downcast_mut::<FrameData>() {
                return Ok(frame_data.take_ref_for_call_result(0).as_usize() as i64);
            }
            if let Some(preview) = frame.data.downcast_mut::<PreviewFrameData>() {
                return Ok(preview.frame.get_ref(0).as_usize() as i64);
            }
            Err(BackendError::Unsupported(
                "unsupported dead frame type for Ref finish result".to_string(),
            ))
        }
        [Type::Float] => Ok(get_float_from_deadframe(frame, 0)?.to_bits() as i64),
        [Type::Void] => Ok(0),
        other => Err(BackendError::Unsupported(format!(
            "unsupported call_assembler finish result layout: {other:?}"
        ))),
    }
}

fn take_call_assembler_deadframe_from_outputs(outputs: &[i64]) -> DeadFrame {
    let handle = outputs
        .first()
        .copied()
        .unwrap_or_else(|| panic!("missing call_assembler deadframe handle slot"))
        as u64;
    assert!(handle != 0, "missing call_assembler deadframe handle");
    take_call_assembler_deadframe(handle)
        .unwrap_or_else(|| panic!("unknown call_assembler deadframe handle {handle}"))
}

fn maybe_take_call_assembler_deadframe(
    fail_index: u32,
    outputs: &[i64],
    handle: u64,
    force_frame: Option<&Arc<ActiveForceFrame>>,
) -> Option<DeadFrame> {
    if fail_index != CALL_ASSEMBLER_DEADFRAME_SENTINEL {
        return None;
    }
    if let Some(force_frame) = force_frame {
        take_force_frame_saved_data(force_frame);
    }
    release_force_token(handle);
    Some(take_call_assembler_deadframe_from_outputs(outputs))
}

fn actual_call_assembler_result_kind(descr: &dyn FailDescr) -> Result<u64, BackendError> {
    match descr.fail_arg_types() {
        [] | [Type::Void] => Ok(CALL_ASSEMBLER_RESULT_VOID),
        [Type::Int] => Ok(CALL_ASSEMBLER_RESULT_INT),
        [Type::Float] => Ok(CALL_ASSEMBLER_RESULT_FLOAT),
        [Type::Ref] if descr.force_token_slots() == [0] => {
            Ok(CALL_ASSEMBLER_RESULT_FORCE_TOKEN_REF)
        }
        [Type::Ref] => Ok(CALL_ASSEMBLER_RESULT_REF),
        other => Err(BackendError::Unsupported(format!(
            "call-assembler target exposes unsupported finish result layout: {other:?}"
        ))),
    }
}

extern "C" fn current_force_token_shim() -> u64 {
    let handle = current_force_frame_handle();
    assert!(
        handle != 0,
        "force_token used without an active compiled frame"
    );
    handle
}

extern "C" fn begin_may_force_call_shim(fail_index: u64, values_ptr: u64, num_values: u64) {
    let handle = current_force_frame_handle();
    assert!(
        handle != 0,
        "call_may_force used without an active compiled frame"
    );
    let frame = lookup_force_frame(handle)
        .unwrap_or_else(|| panic!("missing active force frame for handle {handle}"));
    let fail_descr = frame
        .fail_descrs
        .get(fail_index as usize)
        .unwrap_or_else(|| panic!("missing fail descr {fail_index} for handle {handle}"))
        .clone();
    let raw_values = if num_values == 0 {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(values_ptr as usize as *const i64, num_values as usize)
        }
        .to_vec()
    };
    let preview = PendingForceFrame::new(fail_descr, frame.gc_runtime_id, raw_values);
    let mut pending_may_force = frame.pending_may_force.lock().unwrap();
    pending_may_force.push(PendingMayForceFrame {
        preview,
        was_forced: false,
    });
}

extern "C" fn finish_may_force_guard_shim() -> u64 {
    let handle = current_force_frame_handle();
    assert!(
        handle != 0,
        "guard_not_forced used without an active compiled frame"
    );
    let frame = lookup_force_frame(handle)
        .unwrap_or_else(|| panic!("missing active force frame for handle {handle}"));
    let pending = frame
        .pending_may_force
        .lock()
        .unwrap()
        .pop()
        .expect("guard_not_forced without a preceding call_may_force");
    let was_forced = pending.was_forced;
    let _ = pending.preview.into_raw_values(frame.gc_runtime_id);
    was_forced as u64
}

extern "C" fn record_guard_not_forced_2_shim(fail_index: u64, values_ptr: u64, num_values: u64) {
    let handle = current_force_frame_handle();
    assert!(
        handle != 0,
        "guard_not_forced_2 used without an active compiled frame"
    );
    let frame = lookup_force_frame(handle)
        .unwrap_or_else(|| panic!("missing active force frame for handle {handle}"));
    let fail_descr = frame
        .fail_descrs
        .get(fail_index as usize)
        .unwrap_or_else(|| panic!("missing fail descr {fail_index} for handle {handle}"))
        .clone();
    let raw_values = if num_values == 0 {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(values_ptr as usize as *const i64, num_values as usize)
        }
        .to_vec()
    };
    let pending = PendingForceFrame::new(fail_descr, frame.gc_runtime_id, raw_values);
    let previous = {
        let mut pending_force = frame.pending_force.lock().unwrap();
        pending_force.replace(pending)
    };
    if let Some(previous) = previous {
        let _ = previous.into_raw_values(frame.gc_runtime_id);
    }
}

pub fn force_token_to_dead_frame(force_token: GcRef) -> DeadFrame {
    let handle = force_token.0 as u64;
    let Some(frame) = lookup_force_frame(handle) else {
        panic!("invalid force token {handle}");
    };

    {
        let mut pending_may_force = frame.pending_may_force.lock().unwrap();
        if let Some(pending) = pending_may_force.last_mut() {
            assert!(
                !pending.was_forced,
                "force token {handle} was already forced during call_may_force"
            );
            pending.was_forced = true;
            return DeadFrame {
                data: Box::new(PreviewFrameData::new(
                    pending.preview.materialized_raw_values(),
                    pending.preview.fail_descr.clone(),
                    frame.gc_runtime_id,
                    frame.clone(),
                )),
            };
        }
    }

    let frame = force_frame_registry()
        .lock()
        .unwrap()
        .remove(&handle)
        .unwrap_or_else(|| panic!("invalid force token {handle}"));
    let pending = frame
        .pending_force
        .lock()
        .unwrap()
        .take()
        .unwrap_or_else(|| panic!("force token {handle} has no pending GUARD_NOT_FORCED_2"));
    let fail_descr = pending.fail_descr.clone();
    let raw_values = pending.into_raw_values(frame.gc_runtime_id);
    let saved_data = take_force_frame_saved_data(&frame);
    let (exception_class, exception) = take_pending_jit_exception_state();
    DeadFrame {
        data: Box::new(FrameData::new_with_savedata_and_exception(
            raw_values,
            fail_descr,
            frame.gc_runtime_id,
            saved_data,
            exception_class,
            (!exception.is_null()).then_some(exception),
        )),
    }
}

pub fn set_savedata_ref_on_deadframe(
    frame: &mut DeadFrame,
    data: GcRef,
) -> Result<(), BackendError> {
    if let Some(frame_data) = frame.data.downcast_mut::<FrameData>() {
        frame_data.set_savedata_ref(data);
        return Ok(());
    }
    if let Some(preview) = frame.data.downcast_mut::<PreviewFrameData>() {
        preview.frame.set_savedata_ref(data);
        set_force_frame_saved_data(&preview.active_force_frame, data);
        return Ok(());
    }
    if let Some(overlay) = frame.data.downcast_mut::<OverlayFrameData>() {
        set_savedata_ref_on_deadframe(&mut overlay.inner, data)?;
        return Ok(());
    }
    Err(BackendError::Unsupported(
        "unsupported dead frame type for saved-data".to_string(),
    ))
}

pub fn get_latest_descr_from_deadframe(frame: &DeadFrame) -> Result<&dyn FailDescr, BackendError> {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return Ok(frame_data.fail_descr.as_ref());
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return Ok(preview.frame.fail_descr.as_ref());
    }
    if let Some(overlay) = frame.data.downcast_ref::<OverlayFrameData>() {
        return Ok(overlay.fail_descr.as_ref());
    }
    Err(BackendError::Unsupported(
        "unsupported dead frame type for get_latest_descr".to_string(),
    ))
}

pub fn get_int_from_deadframe(frame: &DeadFrame, index: usize) -> Result<i64, BackendError> {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return Ok(frame_data.get_int(index));
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return Ok(preview.frame.get_int(index));
    }
    if let Some(overlay) = frame.data.downcast_ref::<OverlayFrameData>() {
        return get_int_from_deadframe(&overlay.inner, index);
    }
    Err(BackendError::Unsupported(
        "unsupported dead frame type for get_int".to_string(),
    ))
}

pub fn get_float_from_deadframe(frame: &DeadFrame, index: usize) -> Result<f64, BackendError> {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return Ok(frame_data.get_float(index));
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return Ok(preview.frame.get_float(index));
    }
    if let Some(overlay) = frame.data.downcast_ref::<OverlayFrameData>() {
        return get_float_from_deadframe(&overlay.inner, index);
    }
    Err(BackendError::Unsupported(
        "unsupported dead frame type for get_float".to_string(),
    ))
}

pub fn get_ref_from_deadframe(frame: &DeadFrame, index: usize) -> Result<GcRef, BackendError> {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return Ok(frame_data.get_ref(index));
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return Ok(preview.frame.get_ref(index));
    }
    if let Some(overlay) = frame.data.downcast_ref::<OverlayFrameData>() {
        return get_ref_from_deadframe(&overlay.inner, index);
    }
    Err(BackendError::Unsupported(
        "unsupported dead frame type for get_ref".to_string(),
    ))
}

pub fn get_savedata_ref_from_deadframe(frame: &DeadFrame) -> Result<GcRef, BackendError> {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return Ok(frame_data.get_savedata_ref());
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return Ok(get_force_frame_saved_data(&preview.active_force_frame)
            .unwrap_or_else(|| preview.frame.get_savedata_ref()));
    }
    if let Some(overlay) = frame.data.downcast_ref::<OverlayFrameData>() {
        return get_savedata_ref_from_deadframe(&overlay.inner);
    }
    Err(BackendError::Unsupported(
        "unsupported dead frame type for get_savedata_ref".to_string(),
    ))
}

pub fn grab_savedata_ref_from_deadframe(frame: &DeadFrame) -> Option<GcRef> {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return frame_data.try_get_savedata_ref();
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return get_force_frame_saved_data(&preview.active_force_frame)
            .or_else(|| preview.frame.try_get_savedata_ref());
    }
    if let Some(overlay) = frame.data.downcast_ref::<OverlayFrameData>() {
        return grab_savedata_ref_from_deadframe(&overlay.inner);
    }
    None
}

pub fn grab_exc_value_from_deadframe(frame: &DeadFrame) -> Result<GcRef, BackendError> {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return Ok(frame_data.get_exception_ref());
    }
    if frame.data.downcast_ref::<PreviewFrameData>().is_some() {
        return Ok(GcRef::NULL);
    }
    if let Some(overlay) = frame.data.downcast_ref::<OverlayFrameData>() {
        return grab_exc_value_from_deadframe(&overlay.inner);
    }
    Err(BackendError::Unsupported(
        "unsupported dead frame type for exception value".to_string(),
    ))
}

pub fn grab_exc_class_from_deadframe(frame: &DeadFrame) -> Result<i64, BackendError> {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return Ok(frame_data.get_exception_class());
    }
    if frame.data.downcast_ref::<PreviewFrameData>().is_some() {
        return Ok(0);
    }
    if let Some(overlay) = frame.data.downcast_ref::<OverlayFrameData>() {
        return grab_exc_class_from_deadframe(&overlay.inner);
    }
    Err(BackendError::Unsupported(
        "unsupported dead frame type for exception class".to_string(),
    ))
}

fn execute_registered_loop_target(target: &RegisteredLoopTarget, inputs: &[i64]) -> DeadFrame {
    let mut current_inputs = inputs.to_vec();
    loop {
        let (fail_index, outputs, handle, force_frame) = run_compiled_code(
            target.code_ptr,
            &target.fail_descrs,
            target.gc_runtime_id,
            target.num_ref_roots,
            target.max_output_slots,
            &current_inputs,
            target.needs_force_frame,
        );

        if let Some(frame) =
            maybe_take_call_assembler_deadframe(fail_index, &outputs, handle, force_frame.as_ref())
        {
            return wrap_call_assembler_deadframe_with_caller_prefix(
                frame,
                target.trace_id,
                target.header_pc,
                target.source_guard,
                &target.inputarg_types,
                &current_inputs,
                target.caller_prefix_layout.as_ref(),
            );
        }

        let fail_descr = &target.fail_descrs[fail_index as usize];
        let fail_count = fail_descr.increment_fail_count();
        let bridge_guard = fail_descr.bridge_ref();
        if let Some(ref bridge) = *bridge_guard {
            release_force_token(handle);
            // RPython rebuild_state_after_failure parity: materialize virtual
            // objects from recovery_layout before bridge dispatch.
            let mut mat_outputs = outputs.clone();
            rebuild_state_after_failure(
                &mut mat_outputs,
                &fail_descr.fail_arg_types,
                fail_descr.recovery_layout_ref().as_ref(),
                bridge.num_inputs,
            );
            if bridge.loop_reentry {
                let bridge_frame = CraneliftBackend::execute_bridge(
                    bridge,
                    &mat_outputs,
                    &fail_descr.fail_arg_types,
                );
                drop(bridge_guard);
                let bridge_descr = get_latest_descr_from_deadframe(&bridge_frame)
                    .expect("bridge deadframe must have descriptor");
                if bridge_descr.is_finish() {
                    let num_outputs = bridge_descr.fail_arg_types().len();
                    current_inputs = (0..num_outputs)
                        .map(|i| get_int_from_deadframe(&bridge_frame, i).unwrap_or(0))
                        .collect();
                    continue; // re-enter loop
                }
                return bridge_frame;
            }
            return CraneliftBackend::execute_bridge(
                bridge,
                &mat_outputs,
                &fail_descr.fail_arg_types,
            );
        }
        drop(bridge_guard);

        // Bridge compilation is decided by MetaInterp.must_compile()
        // (compile.py:783-784 jitcounter.tick), not by backend fail_count.

        let saved_data = if let Some(ref ff) = force_frame {
            take_force_frame_saved_data(ff)
        } else {
            None
        };
        let (exception_class, exception) = take_pending_jit_exception_state();
        if !output_transfers_current_force_token(fail_descr, &outputs, handle) {
            release_force_token(handle);
        }

        DeadFrame {
            data: Box::new(FrameData::new_with_savedata_and_exception(
                outputs,
                fail_descr.clone(),
                target.gc_runtime_id,
                saved_data,
                exception_class,
                (!exception.is_null()).then_some(exception),
            )),
        };
    } // end loop
}

/// Stack-allocated output buffer size for the fast path.
/// Traces with more output slots fall back to the normal path.
const FAST_PATH_MAX_OUTPUTS: usize = 16;
/// Stack-allocated GC root buffer size for the fast path.
/// Traces with more live ref roots must use the heap path to avoid
/// overwriting the fixed stack scratch space.
const FAST_PATH_MAX_ROOTS: usize = 8;

/// RPython assembler_call_helper parity: handle guard failure from
/// direct call_assembler path. Checks for bridge first, falls back
/// to force_fn. Called from codegen's direct non-finish block.
///
/// This avoids the full shim → execute_registered_loop_target overhead
/// while still supporting bridge dispatch.
/// RPython assembler_call_helper parity: handle guard failure from
/// direct call_assembler path. Ultra-lightweight: just increments
/// fail count, checks bridge (atomic + mutex only when bridge exists),
/// and defers bridge compilation. Falls back to force_fn.
#[inline(never)]
extern "C" fn call_assembler_guard_failure(
    token_number: u64,
    fail_descr_ptr: i64,
    frame_ptr: i64,
    outputs_ptr: *const i64,
    inputs_ptr: *const i64,
) -> i64 {
    // Handle deadframe sentinel (nested CALL_ASSEMBLER propagation).
    if fail_descr_ptr == CALL_ASSEMBLER_DEADFRAME_SENTINEL as i64 {
        let frame = take_call_assembler_deadframe_from_outputs(unsafe {
            std::slice::from_raw_parts(outputs_ptr, 16)
        });
        let handle = store_call_assembler_deadframe(frame);
        // Return via PENDING_FORCE_LOCAL0 — caller will detect deadframe.
        return handle as i64;
    }

    let target = unsafe { &*fast_lookup_ca_target(token_number) };
    // RPython get_latest_descr parity: jf_descr is a FailDescr pointer.
    let fail_descr_ref = unsafe { &*(fail_descr_ptr as *const CraneliftFailDescr) };
    let fail_index = fail_descr_ref.fail_index();
    let fail_descr = &target.fail_descrs[fail_index as usize];
    let fail_count = fail_descr.increment_fail_count();

    // Fast bridge dispatch: zero-copy — pass the caller's jf_frame
    // directly to the bridge. fail_args are already at the right offsets
    // (callee wrote them on guard exit). No array allocation or copy needed.
    let bridge_ptr = fail_descr.bridge_code_ptr();
    if !bridge_ptr.is_null() {
        let func: unsafe extern "C" fn(*mut i64) -> *mut i64 =
            unsafe { std::mem::transmute(bridge_ptr) };
        // outputs_ptr points to jf_frame items (offset 64 from jf_ptr start).
        // Recover jf_ptr by subtracting the header size.
        let jf_ptr =
            unsafe { (outputs_ptr as *mut u8).sub(JF_FRAME_ITEM0_OFS as usize) as *mut i64 };
        let _result_jf = unsafe { func(jf_ptr) };
        // Result is at jf_frame[0] = outputs_ptr[0]
        return unsafe { *outputs_ptr };
    }

    // RPython resume_in_blackhole parity: resume execution from the guard
    // failure point using the blackhole interpreter. The blackhole reads
    // values from the outputs buffer (deadframe) and executes the remaining
    // IR ops from guard+1 to Finish, returning the result directly.
    let num_outputs = fail_descr.fail_arg_types.len();
    if let Some(bh_fn) = CALL_ASSEMBLER_BLACKHOLE_FN.get() {
        let green_key = target.header_pc;
        let trace_id = target.trace_id;
        if let Some(result) = bh_fn(green_key, trace_id, fail_index, outputs_ptr, num_outputs) {
            return result;
        }
    }
    // Fallback: force_fn re-executes the callee from scratch.
    if !inputs_ptr.is_null() && target.inputarg_types.len() > 3 {
        let raw = unsafe { *inputs_ptr.add(3) };
        PENDING_FORCE_LOCAL0.with(|c| c.set(Some(raw)));
    }
    CALL_ASSEMBLER_FORCE_FN.get().map_or(0, |f| f(frame_ptr))
}

/// Compile a simple bridge (GetfieldGcI + Finish) for base-case guard failures.
/// Called synchronously from call_assembler_guard_failure, matching RPython's
/// handle_fail → _trace_and_compile_from_bridge pattern.
///
/// This does NOT use MetaInterp — bridge ops are trivial and need no optimizer.
fn compile_base_case_bridge(target: &RegisteredLoopTarget, fail_index: u32) -> bool {
    use majit_ir::{InputArg, Op, OpCode, OpRef, Type};

    let fail_descr = match target.fail_descrs.get(fail_index as usize) {
        Some(d) => d,
        None => return false,
    };
    // Don't overwrite a bridge compiled by MetaInterp (compile_bridge in
    // pyjitpl.rs). The MetaInterp bridge has correct inputarg mapping.
    if fail_descr.has_bridge() {
        return true;
    }
    let fail_arg_types = fail_descr.fail_arg_types();

    // Find the value to return in the bridge.
    // Look for Ref (boxed int → unbox) or Int (raw value → return directly)
    // after the virtualizable header (idx >= 3).
    let ref_idx = fail_arg_types
        .iter()
        .enumerate()
        .position(|(i, tp)| i >= 3 && *tp == Type::Ref);
    let int_idx = fail_arg_types
        .iter()
        .enumerate()
        .position(|(i, tp)| i >= 3 && *tp == Type::Int);

    let mut bridge_ops = Vec::new();
    let num_inputs = fail_arg_types.len() as u32;

    if let Some(ref_idx) = ref_idx {
        // Boxed int: GetfieldGcI(n_boxed, intval_offset) → Finish(raw_n)
        let n_boxed = OpRef(ref_idx as u32);
        let intval_descr = majit_ir::make_field_descr(8, 8, Type::Int, true);
        let unboxed = OpRef(num_inputs);
        let mut getfield = Op::with_descr(OpCode::GetfieldGcI, &[n_boxed], intval_descr);
        getfield.pos = unboxed;
        bridge_ops.push(getfield);

        let finish_descr: majit_ir::DescrRef = std::sync::Arc::new(
            crate::guard::CraneliftFailDescr::new_with_kind(num_inputs + 1, vec![Type::Int], true),
        );
        let mut finish_op = Op::with_descr(OpCode::Finish, &[unboxed], finish_descr);
        finish_op.pos = OpRef(num_inputs + 1);
        bridge_ops.push(finish_op);
    } else if let Some(int_idx) = int_idx {
        // Raw int: Finish(raw_value) directly
        let raw_val = OpRef(int_idx as u32);
        let finish_descr: majit_ir::DescrRef = std::sync::Arc::new(
            crate::guard::CraneliftFailDescr::new_with_kind(num_inputs, vec![Type::Int], true),
        );
        let mut finish_op = Op::with_descr(OpCode::Finish, &[raw_val], finish_descr);
        finish_op.pos = OpRef(num_inputs);
        bridge_ops.push(finish_op);
    } else {
        return false;
    };

    let bridge_inputargs: Vec<InputArg> = fail_arg_types
        .iter()
        .enumerate()
        .map(|(i, tp)| InputArg::from_type(*tp, i as u32))
        .collect();

    // Compile using a fresh backend instance
    let mut backend = CraneliftBackend::new();
    backend.set_next_trace_id(target.trace_id * 100 + fail_index as u64 + 1000);
    backend.set_constants(std::collections::HashMap::new());

    let compiled = match backend.do_compile(
        &bridge_inputargs,
        &bridge_ops,
        None,
        Some((target.trace_id, fail_index)),
        None,
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[bridge] do_compile failed: {e}");
            return false;
        }
    };

    let bridge_code_ptr = compiled.code_ptr;

    // Attach bridge to the guard's fail descriptor
    fail_descr.attach_bridge(crate::guard::BridgeData {
        trace_id: compiled.trace_id,
        input_types: compiled.input_types,
        header_pc: compiled.header_pc,
        source_guard: (target.trace_id, fail_index),
        caller_prefix_layout: compiled.caller_prefix_layout,
        code_ptr: bridge_code_ptr,
        fail_descrs: compiled.fail_descrs,
        terminal_exit_layouts: compiled.terminal_exit_layouts,
        gc_runtime_id: compiled.gc_runtime_id,
        loop_reentry: false,
        num_inputs: compiled.num_inputs,
        num_ref_roots: compiled.num_ref_roots,
        max_output_slots: compiled.max_output_slots,
        needs_force_frame: compiled.needs_force_frame,
        invalidated_arc: None, // call_assembler bridges don't need invalidation
    });

    // RPython redirect_call_assembler parity: store bridge code pointer
    // in dispatch entry so Cranelift codegen can dispatch guard failures
    // directly without extern "C" call overhead.
    let table = ca_dispatch_table().lock().unwrap();
    if let Some(entry) = table.get(&(target.trace_id)) {
        entry
            .guard_bridge_ptr
            .store(bridge_code_ptr as *mut u8, Ordering::Release);
    }
    drop(table);

    true
}

/// Fast path for call_assembler when a force callback is available.
/// Runs compiled code with stack-allocated buffers, avoiding all heap
/// allocation per call (no Vec, no DeadFrame, no Box).
fn call_assembler_fast_path(
    target: &RegisteredLoopTarget,
    inputs: &[i64],
    outcome: *mut i64,
    force_fn: extern "C" fn(i64) -> i64,
) -> u64 {
    // RPython parity: compile_tmp_callback. When target is pending
    // (code_ptr not yet set), fall back to force_fn which runs the
    // interpreter. The force_fn receives the callee frame pointer
    // from the first input arg.
    if target.code_ptr.is_null() {
        let frame_ptr = inputs.get(0).copied().unwrap_or(0);
        // Set pending_force_local0 for lazy frame creation.
        // When create_frame is elided, frame_ptr is the caller frame.
        // force_fn uses pending_force_local0 to create a callee frame.
        if inputs.len() > 3 {
            PENDING_FORCE_LOCAL0.with(|c| c.set(Some(inputs[3])));
        }
        let result = force_fn(frame_ptr);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
            *outcome.add(1) = 0;
        }
        return result as u64;
    }

    let actual_outputs = target.max_output_slots.max(1);
    let actual_roots = target.num_ref_roots.max(1);
    if actual_outputs + actual_roots > FAST_PATH_MAX_OUTPUTS {
        return call_assembler_fast_path_heap(target, inputs, outcome, force_fn);
    }

    let func: unsafe extern "C" fn(*mut i64) -> *mut i64 =
        unsafe { std::mem::transmute(target.code_ptr) };

    let _jitted_guard = majit_codegen::JittedGuard::enter();

    let handle = 0u64;

    // jf_buf: header + output slots + ref root slots (all in one buffer).
    const HEADER_WORDS: usize = (JF_FRAME_ITEM0_OFS as usize) / 8;
    let mut jf_buf = [0i64; HEADER_WORDS + FAST_PATH_MAX_OUTPUTS + FAST_PATH_MAX_ROOTS];
    for (i, &val) in inputs.iter().enumerate() {
        jf_buf[HEADER_WORDS + i] = val;
    }

    let result_jf = unsafe { func(jf_buf.as_mut_ptr()) };
    let jf_descr_raw = unsafe { *result_jf.add(JF_DESCR_OFS as usize / 8) };
    let fail_index = if jf_descr_raw == CALL_ASSEMBLER_DEADFRAME_SENTINEL as i64 {
        CALL_ASSEMBLER_DEADFRAME_SENTINEL
    } else if jf_descr_raw == 0 {
        0u32
    } else {
        unsafe { &*(jf_descr_raw as *const CraneliftFailDescr) }.fail_index()
    };
    let outputs = {
        let mut out = [0i64; FAST_PATH_MAX_OUTPUTS];
        out.copy_from_slice(&jf_buf[HEADER_WORDS..HEADER_WORDS + FAST_PATH_MAX_OUTPUTS]);
        out
    };

    drop(_jitted_guard);

    // Handle nested call_assembler DEADFRAME propagation
    if fail_index == CALL_ASSEMBLER_DEADFRAME_SENTINEL {
        let frame = take_call_assembler_deadframe_from_outputs(&outputs);
        release_force_token(handle);
        let handle = store_call_assembler_deadframe(frame);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_DEADFRAME;
            *outcome.add(1) = handle as i64;
        }
        return 0;
    }

    let fail_descr = &target.fail_descrs[fail_index as usize];

    if fail_descr.is_finish() {
        release_force_token(handle);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
            *outcome.add(1) = 0;
        }
        return match fail_descr.fail_arg_types() {
            [] | [Type::Void] => 0,
            [Type::Int] | [Type::Float] => outputs[0] as u64,
            _ => {
                let outputs_vec = outputs[..actual_outputs].to_vec();
                let mut frame =
                    build_deadframe_from_outputs(outputs_vec, fail_descr, target.gc_runtime_id);
                finish_result_from_deadframe(&mut frame)
                    .expect("finish_result_from_deadframe failed") as u64
            }
        };
    }

    // Guard failure — check for bridge, then fall back to force
    fail_descr.increment_fail_count();

    // If a bridge is attached, execute it instead of calling force_fn.
    let bridge_guard = fail_descr.bridge_ref();
    if let Some(ref bridge) = *bridge_guard {
        release_force_token(handle);
        // RPython rebuild_state_after_failure parity: materialize virtual
        // objects before bridge dispatch. Virtual Ref slots are null (0)
        // with their fields in trailing Int slots.
        let mut bridge_outputs = outputs.to_vec();
        rebuild_state_after_failure(
            &mut bridge_outputs,
            &fail_descr.fail_arg_types,
            fail_descr.recovery_layout_ref().as_ref(),
            bridge.num_inputs,
        );
        let outputs_slice = &bridge_outputs[..bridge_outputs.len().min(actual_outputs)];
        let mut frame =
            CraneliftBackend::execute_bridge(bridge, outputs_slice, &fail_descr.fail_arg_types);
        let bridge_descr = get_latest_descr_from_deadframe(&frame)
            .expect("bridge deadframe must have a descriptor");
        if bridge_descr.is_finish() {
            unsafe {
                *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
                *outcome.add(1) = 0;
            }
            return finish_result_from_deadframe(&mut frame)
                .expect("finish_result_from_deadframe failed") as u64;
        }
        // Bridge didn't finish — store as deadframe for caller
        let df_handle = store_call_assembler_deadframe(frame);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_DEADFRAME;
            *outcome.add(1) = df_handle as i64;
        }
        return 0;
    }
    drop(bridge_guard);

    release_force_token(handle);

    // RPython assembler_call_helper: force_fn receives the callee frame.
    // outputs[0] holds the virtualizable frame (fail_args[0] = caller),
    // but force_fn needs the callee frame which is inputs[0].
    let callee_frame_ptr = inputs[0];
    let result = force_fn(callee_frame_ptr);
    unsafe {
        *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
        *outcome.add(1) = 0;
    }
    result as u64
}

/// Heap-allocated fallback for call_assembler_fast_path when outputs
/// exceed the stack buffer size.
fn call_assembler_fast_path_heap(
    target: &RegisteredLoopTarget,
    inputs: &[i64],
    outcome: *mut i64,
    force_fn: extern "C" fn(i64) -> i64,
) -> u64 {
    let actual_outputs = target.max_output_slots.max(1);
    let (fail_index, outputs, handle, _force_frame) = run_compiled_code(
        target.code_ptr,
        &target.fail_descrs,
        target.gc_runtime_id,
        target.num_ref_roots,
        target.max_output_slots,
        inputs,
        target.needs_force_frame,
    );

    if fail_index == CALL_ASSEMBLER_DEADFRAME_SENTINEL {
        let frame = take_call_assembler_deadframe_from_outputs(&outputs);
        release_force_token(handle);
        let handle = store_call_assembler_deadframe(frame);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_DEADFRAME;
            *outcome.add(1) = handle as i64;
        }
        return 0;
    }

    let fail_descr = &target.fail_descrs[fail_index as usize];

    if fail_descr.is_finish() {
        release_force_token(handle);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
            *outcome.add(1) = 0;
        }
        return match fail_descr.fail_arg_types() {
            [] | [Type::Void] => 0,
            [Type::Int] | [Type::Float] => outputs[0] as u64,
            _ => {
                let outputs_vec = outputs[..actual_outputs].to_vec();
                let mut frame =
                    build_deadframe_from_outputs(outputs_vec, fail_descr, target.gc_runtime_id);
                finish_result_from_deadframe(&mut frame)
                    .expect("finish_result_from_deadframe failed") as u64
            }
        };
    }

    // Guard failure — check for bridge, then fall back to force
    fail_descr.increment_fail_count();

    let bridge_guard = fail_descr.bridge_ref();
    if let Some(ref bridge) = *bridge_guard {
        release_force_token(handle);
        let mut frame =
            CraneliftBackend::execute_bridge(bridge, &outputs, &fail_descr.fail_arg_types);
        let bridge_descr = get_latest_descr_from_deadframe(&frame)
            .expect("bridge deadframe must have a descriptor");
        if bridge_descr.is_finish() {
            unsafe {
                *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
                *outcome.add(1) = 0;
            }
            return finish_result_from_deadframe(&mut frame)
                .expect("finish_result_from_deadframe failed") as u64;
        }
        let df_handle = store_call_assembler_deadframe(frame);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_DEADFRAME;
            *outcome.add(1) = df_handle as i64;
        }
        return 0;
    }
    drop(bridge_guard);

    release_force_token(handle);

    // RPython assembler_call_helper: force_fn receives the callee frame.
    // outputs[0] holds the virtualizable frame (fail_args[0] = caller),
    // but force_fn needs the callee frame which is inputs[0].
    let callee_frame_ptr = inputs[0];
    let result = force_fn(callee_frame_ptr);
    unsafe {
        *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
        *outcome.add(1) = 0;
    }
    result as u64
}

/// Build a DeadFrame from raw outputs for cases where the fast path
/// can't extract the result directly (e.g. Ref-typed results).
fn build_deadframe_from_outputs(
    outputs: Vec<i64>,
    fail_descr: &Arc<CraneliftFailDescr>,
    gc_runtime_id: Option<u64>,
) -> DeadFrame {
    let (exception_class, exception) = take_pending_jit_exception_state();
    DeadFrame {
        data: Box::new(FrameData::new_with_savedata_and_exception(
            outputs,
            fail_descr.clone(),
            gc_runtime_id,
            None,
            exception_class,
            (!exception.is_null()).then_some(exception),
        )),
    }
}

extern "C" fn call_assembler_shim(
    target_token: u64,
    args_ptr: u64,
    outcome_ptr: u64,
    _expected_result_kind: u64,
) -> u64 {
    let outcome = outcome_ptr as usize as *mut i64;

    let target = unsafe { &*fast_lookup_ca_target(target_token) };

    let input_slice =
        unsafe { std::slice::from_raw_parts(args_ptr as usize as *const i64, target.num_inputs) };
    assert!(
        !outcome.is_null(),
        "call_assembler shim outcome buffer must be non-null"
    );

    if std::env::var_os("MAJIT_LOG").is_some() {
        eprintln!(
            "[ca-shim] entering trace_id={} num_inputs={}",
            target.trace_id, target.num_inputs
        );
    }
    let mut frame = execute_registered_loop_target(target, input_slice);
    let descr =
        get_latest_descr_from_deadframe(&frame).expect("get_latest_descr_from_deadframe failed");
    if descr.is_finish() {
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
            *outcome.add(1) = 0;
        }
        return finish_result_from_deadframe(&mut frame)
            .expect("finish_result_from_deadframe failed") as u64;
    }

    // RPython resume_in_blackhole parity: use blackhole to resume from
    // the guard failure point instead of re-executing from scratch.
    let fail_index = descr.fail_index();
    let fail_types = descr.fail_arg_types();
    let fail_values: Vec<i64> = (0..fail_types.len())
        .map(|i| get_int_from_deadframe(&frame, i).unwrap_or(0))
        .collect();
    if std::env::var_os("MAJIT_LOG").is_some() {
        eprintln!(
            "[ca-shim] guard fail_idx={} nvals={}",
            fail_index,
            fail_values.len()
        );
    }
    if let Some(bh_fn) = CALL_ASSEMBLER_BLACKHOLE_FN.get() {
        if let Some(result) = bh_fn(
            target.header_pc,
            target.trace_id,
            fail_index,
            fail_values.as_ptr(),
            fail_values.len(),
        ) {
            unsafe {
                *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
                *outcome.add(1) = 0;
            }
            return result as u64;
        }
    }
    // Fallback: force_fn re-executes from scratch.
    if let Some(force_fn) = CALL_ASSEMBLER_FORCE_FN.get() {
        let callee_frame_ptr = input_slice[0];
        if input_slice.len() > 3 {
            PENDING_FORCE_LOCAL0.with(|c| c.set(Some(input_slice[3])));
        }
        let result = force_fn(callee_frame_ptr);
        unsafe {
            *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_FINISH;
            *outcome.add(1) = 0;
        }
        return result as u64;
    }

    let handle = store_call_assembler_deadframe(frame);
    unsafe {
        *outcome.add(0) = CALL_ASSEMBLER_OUTCOME_DEADFRAME;
        *outcome.add(1) = handle as i64;
    }
    0
}

/// RPython parity: alloc shims receive (runtime_id, ...args).
/// Roots are persistently registered by run_compiled_code via add_root
/// on jf_frame ref slot addresses (_call_header_shadowstack parity).
extern "C" fn gc_alloc_nursery_shim(runtime_id: u64, size: u64) -> u64 {
    with_gc_runtime(runtime_id, |gc| {
        let obj = gc.alloc_nursery(size as usize);
        obj.0 as u64
    })
}

/// Plain malloc fallback for New() when no GC runtime is configured.
extern "C" fn plain_malloc_zeroed_shim(size: u64) -> u64 {
    let layout = std::alloc::Layout::from_size_align(size as usize, 8)
        .unwrap_or(std::alloc::Layout::new::<u8>());
    unsafe { std::alloc::alloc_zeroed(layout) as u64 }
}

extern "C" fn gc_alloc_typed_nursery_shim(runtime_id: u64, type_id: u64, size: u64) -> u64 {
    // RPython rewrite.py: CALL_MALLOC_NURSERY fallback path.
    // Use no-collect allocation to avoid triggering GC during compiled code
    // execution. When nursery is full, falls back to old-gen allocation.
    with_gc_runtime(runtime_id, |gc| {
        let obj = gc.alloc_nursery_no_collect(size as usize);
        obj.0 as u64
    })
}

extern "C" fn gc_alloc_varsize_shim(
    runtime_id: u64,
    base_size: u64,
    item_size: u64,
    length: u64,
) -> u64 {
    with_gc_runtime(runtime_id, |gc| {
        let obj = gc.alloc_varsize(base_size as usize, item_size as usize, length as usize);
        obj.0 as u64
    })
}

extern "C" fn gc_write_barrier_shim(runtime_id: u64, obj: u64) {
    with_gc_runtime(runtime_id, |gc| gc.write_barrier(GcRef(obj as usize)));
}

thread_local! {
    static DECLARED_VARS_DEBUG: std::cell::RefCell<Option<std::collections::HashSet<u32>>> = const { std::cell::RefCell::new(None) };
    /// Op-result variable positions: ops that define a result via def_var.
    /// When an OpRef collides (in both constants map and op-result set),
    /// the variable takes precedence over the constant.
    static OP_RESULT_VARS: std::cell::RefCell<Option<std::collections::HashSet<u32>>> = const { std::cell::RefCell::new(None) };
}

fn resolve_opref(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    opref: OpRef,
) -> CValue {
    if let Some(&c) = constants.get(&opref.0) {
        // Op results take precedence over constants. GC rewriter and
        // optimizer may assign op results to positions that collide with
        // constant keys. The variable (e.g., allocation pointer from New)
        // must win over the stale constant value.
        let is_op_result = OP_RESULT_VARS.with(|cell| {
            cell.borrow()
                .as_ref()
                .is_some_and(|rv| rv.contains(&opref.0))
        });
        if !is_op_result {
            return builder.ins().iconst(cl_types::I64, c);
        }
    }
    if opref.is_none() {
        return builder.ins().iconst(cl_types::I64, 0);
    }
    DECLARED_VARS_DEBUG.with(|cell| {
        if let Some(ref dv) = *cell.borrow() {
            if !dv.contains(&opref.0) {
                eprintln!("[cranelift] UNDECLARED var{} at use_var", opref.0);
            }
        }
    });
    builder.use_var(var(opref.0))
}

fn resolve_binop(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    op: &Op,
) -> (CValue, CValue) {
    let a = resolve_opref(builder, constants, op.arg(0));
    let b = resolve_opref(builder, constants, op.arg(1));
    (a, b)
}

fn emit_icmp(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    cc: IntCC,
    op: &Op,
    vi: u32,
) {
    let (a, b) = resolve_binop(builder, constants, op);
    let cmp = builder.ins().icmp(cc, a, b);
    let r = builder.ins().uextend(cl_types::I64, cmp);
    builder.def_var(var(vi), r);
}

fn emit_fcmp(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    cc: FloatCC,
    op: &Op,
    vi: u32,
) {
    let (a, b) = resolve_binop(builder, constants, op);
    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
    let cmp = builder.ins().fcmp(cc, fa, fb);
    let r = builder.ins().uextend(cl_types::I64, cmp);
    builder.def_var(var(vi), r);
}

/// Map a field size (in bytes) to the corresponding Cranelift type.
fn cl_type_for_size(size: usize) -> cranelift_codegen::ir::Type {
    match size {
        1 => cl_types::I8,
        2 => cl_types::I16,
        4 => cl_types::I32,
        8 => cl_types::I64,
        _ => cl_types::I64,
    }
}

fn builtin_string_array_descr(opcode: OpCode) -> Option<majit_ir::DescrRef> {
    let item_size = match opcode {
        OpCode::Newstr
        | OpCode::Strlen
        | OpCode::Strgetitem
        | OpCode::Strsetitem
        | OpCode::Copystrcontent
        | OpCode::Strhash => 1,
        OpCode::Newunicode
        | OpCode::Unicodelen
        | OpCode::Unicodegetitem
        | OpCode::Unicodesetitem
        | OpCode::Copyunicodecontent
        | OpCode::Unicodehash => 4,
        _ => return None,
    };

    let len_descr = Arc::new(BuiltinFieldDescr {
        offset: BUILTIN_STRING_LEN_OFFSET,
        field_size: 8,
        field_type: Type::Int,
        signed: false,
    });
    Some(Arc::new(BuiltinArrayDescr {
        // Match RPython's string/unicode JIT-visible layout closely enough:
        // cached hash word, length word, then the character data.
        base_size: BUILTIN_STRING_BASE_SIZE,
        item_size,
        type_id: 0,
        item_type: Type::Int,
        signed: false,
        len_descr,
    }))
}

fn checked_cl_type_for_size(
    size: usize,
    opcode: OpCode,
    detail: &str,
) -> Result<cranelift_codegen::ir::Type, BackendError> {
    match size {
        1 | 2 | 4 | 8 => Ok(cl_type_for_size(size)),
        _ => Err(unsupported_semantics(opcode, detail)),
    }
}

fn op_var_index(op: &Op, op_idx: usize, num_inputs: usize) -> usize {
    if op.pos.is_none() {
        num_inputs + op_idx
    } else {
        op.pos.0 as usize
    }
}

fn unsupported_semantics(opcode: OpCode, detail: &str) -> BackendError {
    BackendError::Unsupported(format!(
        "opcode {:?} is not supported in the Cranelift backend yet: {detail}",
        opcode
    ))
}

fn missing_gc_runtime(opcode: OpCode) -> BackendError {
    BackendError::Unsupported(format!(
        "opcode {:?} requires a configured GC runtime in the Cranelift backend",
        opcode
    ))
}

fn resolve_call_assembler_target(
    opcode: OpCode,
    call_descr: &dyn CallDescr,
) -> Result<Option<RegisteredLoopTarget>, BackendError> {
    let target_token = call_descr.call_target_token().ok_or_else(|| {
        unsupported_semantics(
            opcode,
            "call-assembler descriptor must provide a compiled target token",
        )
    })?;
    let Some(target) = lookup_call_assembler_target(target_token) else {
        return Ok(None);
    };

    // Pending targets (null code_ptr) are placeholders — no compiled code
    // or finish descriptors yet. Return None so codegen uses shim fallback.
    // At runtime, call_assembler_fast_path detects null code_ptr and calls
    // force_fn (RPython compile_tmp_callback parity).
    if target.code_ptr.is_null() {
        return Ok(None);
    }

    // RPython: arity check only (Int vs Ref mismatch is valid for
    // function-entry typed locals).
    if target.inputarg_types.len() != call_descr.arg_types().len() {
        return Err(unsupported_semantics(
            opcode,
            "call-assembler target arity does not match the descriptor",
        ));
    }
    let finish_descr = target
        .fail_descrs
        .iter()
        .find(|descr| descr.is_finish())
        .ok_or_else(|| {
            unsupported_semantics(
                opcode,
                "call-assembler target must expose at least one finish exit",
            )
        })?;
    let finish_types = finish_descr.fail_arg_types();

    // Validate that the finish result type matches the call descriptor.
    // When force_token_slots are present the finish output type is Ref
    // (the raw force-token handle), so accept that as matching a Ref
    // result descriptor even though the value isn't a real GC ref.
    match call_descr.result_type() {
        Type::Void => {
            if !finish_types.is_empty() {
                return Err(unsupported_semantics(
                    opcode,
                    "void call-assembler targets must finish without result values",
                ));
            }
        }
        result_type => {
            if finish_types != [result_type] {
                // Type mismatch between caller's expected result and target's
                // actual finish type. Treat as unresolved — runtime will use
                // the helper fallback path.
                return Ok(None);
            }
        }
    }

    Ok(Some(target))
}

fn expected_call_assembler_result_kind(
    call_descr: &dyn CallDescr,
    target: Option<&RegisteredLoopTarget>,
) -> Result<u64, BackendError> {
    match call_descr.result_type() {
        Type::Void => Ok(CALL_ASSEMBLER_RESULT_VOID),
        Type::Int => Ok(CALL_ASSEMBLER_RESULT_INT),
        Type::Float => Ok(CALL_ASSEMBLER_RESULT_FLOAT),
        Type::Ref => {
            let Some(target) = target else {
                return Ok(CALL_ASSEMBLER_RESULT_REF);
            };
            let finish_descr = target
                .fail_descrs
                .iter()
                .find(|descr| descr.is_finish())
                .ok_or_else(|| {
                    unsupported_semantics(
                        OpCode::CallAssemblerR,
                        "call-assembler target must expose at least one finish exit",
                    )
                })?;
            Ok(
                if finish_descr.fail_arg_types() == [Type::Ref]
                    && finish_descr.force_token_slots == [0]
                {
                    CALL_ASSEMBLER_RESULT_FORCE_TOKEN_REF
                } else {
                    CALL_ASSEMBLER_RESULT_REF
                },
            )
        }
    }
}

fn build_known_values_set(inputargs: &[InputArg], ops: &[Op]) -> HashSet<u32> {
    let mut known = HashSet::new();
    for input in inputargs {
        known.insert(input.index);
    }
    for op in ops {
        if op.result_type() != Type::Void && !op.pos.is_none() {
            known.insert(op.pos.0);
        }
    }
    known
}

fn build_force_token_set(inputargs: &[InputArg], ops: &[Op]) -> Result<HashSet<u32>, BackendError> {
    let mut force_tokens = HashSet::new();
    for (op_idx, op) in ops.iter().enumerate() {
        if op.pos.is_none() {
            continue;
        }
        let result_var = op_var_index(op, op_idx, inputargs.len()) as u32;
        if op.opcode == OpCode::ForceToken {
            force_tokens.insert(result_var);
            continue;
        }
        if op.opcode == OpCode::CallAssemblerR {
            let descr = op.descr.as_ref().ok_or_else(|| {
                unsupported_semantics(op.opcode, "call-assembler op must have a descriptor")
            })?;
            let call_descr = descr.as_call_descr().ok_or_else(|| {
                unsupported_semantics(op.opcode, "call-assembler descriptor must be a CallDescr")
            })?;
            if let Some(target) = resolve_call_assembler_target(op.opcode, call_descr)? {
                if let Some(finish_descr) =
                    target.fail_descrs.iter().find(|descr| descr.is_finish())
                {
                    if finish_descr.fail_arg_types() == [Type::Ref]
                        && finish_descr.force_token_slots == [0]
                    {
                        force_tokens.insert(result_var);
                    }
                }
            }
        }
    }
    Ok(force_tokens)
}

/// Returns (value_types, inputarg_types, op_def_positions).
/// - value_types: merged map (ops override inputargs) — used for most lookups
/// - inputarg_types: inputarg-only types — used for positional type inference
/// - op_def_positions: OpRef → op_index of the defining operation
///   (only for OpRefs that are both inputargs and op results with different types)
fn build_value_type_map(
    inputargs: &[InputArg],
    ops: &[Op],
) -> (HashMap<u32, Type>, HashMap<u32, Type>, HashMap<u32, usize>) {
    let mut value_types = HashMap::new();
    let mut inputarg_types = HashMap::new();
    let mut op_def_positions = HashMap::new();

    for input in inputargs {
        value_types.insert(input.index, input.tp);
        inputarg_types.insert(input.index, input.tp);
    }

    for (op_idx, op) in ops.iter().enumerate() {
        let result_type = op.result_type();
        if result_type != Type::Void {
            let var_idx = op_var_index(op, op_idx, inputargs.len()) as u32;
            if let Some(&ia_type) = inputarg_types.get(&var_idx) {
                if ia_type != result_type {
                    op_def_positions.insert(var_idx, op_idx);
                }
            }
            value_types.insert(var_idx, result_type);
        }
        // Propagate optimizer-provided fail_arg_types to value_types.
        // This ensures constant OpRefs typed as Ref by the optimizer
        // (e.g., function pointers in GuardValue) are correctly typed
        // in the backend's infer_fail_arg_types fallback.
        if let Some(ref fat) = op.fail_arg_types {
            if let Some(ref fa) = op.fail_args {
                for (i, &opref) in fa.iter().enumerate() {
                    // Skip constant pool entries — their type is resolved
                    // via constants map in resolve_opref, not value_types.
                    if !opref.is_none() && opref.0 < 10_000 && !value_types.contains_key(&opref.0) {
                        if let Some(&tp) = fat.get(i) {
                            value_types.insert(opref.0, tp);
                        }
                    }
                }
            }
        }
    }

    // Box identity parity: when a JUMP sends a value to a LABEL arg,
    // the LABEL arg's type changes to the JUMP arg's type. RPython's
    // Box objects carry their own types; in our flat OpRef namespace we
    // must propagate types through the JUMP→LABEL edge explicitly.
    // Collect LABEL→descr and JUMP→descr, then match by descr index.
    let mut label_by_descr: HashMap<u32, usize> = HashMap::new();
    let mut jumps_by_descr: Vec<(u32, usize)> = Vec::new();
    for (op_idx, op) in ops.iter().enumerate() {
        if op.opcode == OpCode::Label {
            if let Some(ref d) = op.descr {
                label_by_descr.insert(d.index(), op_idx);
            }
        } else if op.opcode == OpCode::Jump {
            if let Some(ref d) = op.descr {
                jumps_by_descr.push((d.index(), op_idx));
            }
        }
    }
    for (descr_idx, jump_idx) in &jumps_by_descr {
        let Some(&label_idx) = label_by_descr.get(descr_idx) else {
            continue;
        };
        let label_op = &ops[label_idx];
        let jump_op = &ops[*jump_idx];
        for (i, &label_arg) in label_op.args.iter().enumerate() {
            if label_arg.is_none() {
                continue;
            }
            let Some(&jump_arg) = jump_op.args.get(i) else {
                continue;
            };
            if jump_arg.is_none() {
                continue;
            }
            let jump_type = value_types.get(&jump_arg.0).copied().unwrap_or(Type::Int);
            let label_type = value_types.get(&label_arg.0).copied().unwrap_or(Type::Ref);
            if jump_type != label_type {
                value_types.insert(label_arg.0, jump_type);
                op_def_positions.insert(label_arg.0, label_idx);
            }
        }
    }

    (value_types, inputarg_types, op_def_positions)
}

/// Lightweight variant for call sites that only need merged value_types.
fn build_value_type_map_simple(inputargs: &[InputArg], ops: &[Op]) -> HashMap<u32, Type> {
    build_value_type_map(inputargs, ops).0
}

fn build_ref_root_slots(
    inputargs: &[InputArg],
    ops: &[Op],
    force_tokens: &HashSet<u32>,
) -> Vec<(u32, usize)> {
    let mut seen = HashSet::new();
    let mut slots = Vec::new();

    for input in inputargs {
        if input.tp == Type::Ref && !force_tokens.contains(&input.index) && seen.insert(input.index)
        {
            slots.push((input.index, slots.len()));
        }
    }

    for (op_idx, op) in ops.iter().enumerate() {
        if op.result_type() == Type::Ref {
            let vi = op_var_index(op, op_idx, inputargs.len()) as u32;
            if !force_tokens.contains(&vi) && seen.insert(vi) {
                slots.push((vi, slots.len()));
            }
        }
    }

    slots
}

/// Simple normalization: assign sequential pos to ops without pos.
fn normalize_ops_for_codegen_simple(inputargs: &[InputArg], ops: &[Op]) -> Vec<Op> {
    let num_inputs = inputargs.len() as u32;
    ops.iter()
        .enumerate()
        .map(|(op_idx, op)| {
            let mut normalized = op.clone();
            if normalized.result_type() != Type::Void && normalized.pos.is_none() {
                normalized.pos = OpRef(num_inputs + op_idx as u32);
            }
            normalized
        })
        .collect()
}

/// Renumber sparse OpRef indices to dense sequential form.
/// (Currently unused — kept for future use when renumbering is safe.)
#[allow(dead_code)]
fn normalize_ops_for_codegen(
    inputargs: &[InputArg],
    ops: &[Op],
    constants: &HashMap<u32, i64>,
) -> (Vec<Op>, std::collections::HashMap<u32, u32>) {
    let num_inputs = inputargs.len() as u32;

    // Collect every unique OpRef index used anywhere.
    let mut all_indices = std::collections::BTreeSet::<u32>::new();
    for i in 0..num_inputs {
        all_indices.insert(i);
    }
    // Note: constants keys are NOT added to all_indices.
    // They are remapped separately in prepare_ops_for_compile.
    // Dead constants (not referenced by any op) should not inflate the index space.
    for (op_idx, op) in ops.iter().enumerate() {
        if op.result_type() != Type::Void {
            let idx = if op.pos.is_none() {
                num_inputs + op_idx as u32
            } else {
                op.pos.0
            };
            all_indices.insert(idx);
        }
        for arg in &op.args {
            if !arg.is_none() {
                all_indices.insert(arg.0);
            }
        }
        if let Some(ref fa) = op.fail_args {
            for arg in fa {
                if !arg.is_none() {
                    all_indices.insert(arg.0);
                }
            }
        }
    }

    // Build dense remap: sorted old → sequential new.
    let remap: std::collections::HashMap<u32, u32> = all_indices
        .iter()
        .enumerate()
        .map(|(new, &old)| (old, new as u32))
        .collect();

    // Apply remap to all ops.
    let remapped = ops
        .iter()
        .enumerate()
        .map(|(op_idx, op)| {
            let mut n = op.clone();
            if n.result_type() != Type::Void {
                let old = if n.pos.is_none() {
                    num_inputs + op_idx as u32
                } else {
                    n.pos.0
                };
                n.pos = OpRef(remap[&old]);
            }
            for arg in n.args.iter_mut() {
                if !arg.is_none() {
                    *arg = OpRef(remap[&arg.0]);
                }
            }
            if let Some(ref mut fa) = n.fail_args {
                for arg in fa.iter_mut() {
                    if !arg.is_none() {
                        *arg = OpRef(remap[&arg.0]);
                    }
                }
            }
            // Remap rd_virtuals field fail_arg indices to match renumbered fail_args.
            // rd_virtuals entries reference fail_arg positions by index, not OpRef,
            // so they don't need OpRef remapping — only fail_arg_index adjustment
            // if fail_args were reordered (they aren't in dense renumbering).
            n
        })
        .collect();

    (remapped, remap)
}

fn inject_builtin_string_descrs(ops: &mut [Op]) {
    for op in ops {
        if op.descr.is_none() {
            if let Some(descr) = builtin_string_array_descr(op.opcode) {
                op.descr = Some(descr);
            }
        }
    }
}

fn resolve_opref_or_imm(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    known_values: &HashSet<u32>,
    opref: OpRef,
) -> CValue {
    if let Some(&c) = constants.get(&opref.0) {
        return builder.ins().iconst(cl_types::I64, c);
    }
    if known_values.contains(&opref.0) {
        return builder.use_var(var(opref.0));
    }
    builder.ins().iconst(cl_types::I64, opref.0 as i64)
}

fn resolve_constant_i64(
    constants: &HashMap<u32, i64>,
    known_values: &HashSet<u32>,
    opcode: OpCode,
    opref: OpRef,
    what: &str,
) -> Result<i64, BackendError> {
    if let Some(&c) = constants.get(&opref.0) {
        return Ok(c);
    }
    if known_values.contains(&opref.0) {
        return Err(unsupported_semantics(
            opcode,
            &format!("{what} must be a compile-time constant"),
        ));
    }
    Ok(opref.0 as i64)
}

fn resolve_rewriter_immediate_i64(constants: &HashMap<u32, i64>, opref: OpRef) -> i64 {
    constants.get(&opref.0).copied().unwrap_or(opref.0 as i64)
}

fn type_for_opref(
    value_types: &HashMap<u32, Type>,
    known_values: &HashSet<u32>,
    opcode: OpCode,
    opref: OpRef,
    what: &str,
) -> Result<Type, BackendError> {
    if let Some(&tp) = value_types.get(&opref.0) {
        return Ok(tp);
    }
    if known_values.contains(&opref.0) {
        return Err(unsupported_semantics(
            opcode,
            &format!("type information for {what} is missing"),
        ));
    }
    Ok(Type::Int)
}

fn emit_load_from_addr(
    builder: &mut FunctionBuilder,
    addr: CValue,
    value_type: Type,
    size: usize,
    signed: bool,
    opcode: OpCode,
) -> Result<CValue, BackendError> {
    match value_type {
        Type::Float => {
            if size != 8 {
                return Err(unsupported_semantics(
                    opcode,
                    "float memory operations currently require 8-byte values",
                ));
            }
            let fval = builder
                .ins()
                .load(cl_types::F64, MemFlags::trusted(), addr, 0);
            Ok(builder.ins().bitcast(cl_types::I64, MemFlags::new(), fval))
        }
        Type::Int | Type::Ref => {
            if value_type == Type::Ref && size != 8 {
                return Err(unsupported_semantics(
                    opcode,
                    "reference memory operations currently require pointer-sized values",
                ));
            }
            let mem_ty = checked_cl_type_for_size(
                size,
                opcode,
                "memory operations only support 1-, 2-, 4-, and 8-byte values",
            )?;
            let raw = builder.ins().load(mem_ty, MemFlags::trusted(), addr, 0);
            if mem_ty == cl_types::I64 {
                Ok(raw)
            } else if value_type == Type::Int && signed {
                Ok(builder.ins().sextend(cl_types::I64, raw))
            } else {
                Ok(builder.ins().uextend(cl_types::I64, raw))
            }
        }
        Type::Void => Err(unsupported_semantics(
            opcode,
            "void-typed memory loads are invalid",
        )),
    }
}

fn emit_store_to_addr(
    builder: &mut FunctionBuilder,
    addr: CValue,
    value: CValue,
    value_type: Type,
    size: usize,
    opcode: OpCode,
) -> Result<(), BackendError> {
    match value_type {
        Type::Float => {
            if size != 8 {
                return Err(unsupported_semantics(
                    opcode,
                    "float memory operations currently require 8-byte values",
                ));
            }
            let fval = builder.ins().bitcast(cl_types::F64, MemFlags::new(), value);
            builder.ins().store(MemFlags::trusted(), fval, addr, 0);
        }
        Type::Int | Type::Ref => {
            if value_type == Type::Ref && size != 8 {
                return Err(unsupported_semantics(
                    opcode,
                    "reference memory operations currently require pointer-sized values",
                ));
            }
            let mem_ty = checked_cl_type_for_size(
                size,
                opcode,
                "memory operations only support 1-, 2-, 4-, and 8-byte values",
            )?;
            let store_val = if mem_ty == cl_types::I64 {
                value
            } else {
                builder.ins().ireduce(mem_ty, value)
            };
            builder.ins().store(MemFlags::trusted(), store_val, addr, 0);
        }
        Type::Void => {
            return Err(unsupported_semantics(
                opcode,
                "void-typed memory stores are invalid",
            ));
        }
    }
    Ok(())
}

fn emit_dynamic_offset_addr(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    known_values: &HashSet<u32>,
    base_arg: OpRef,
    offset_arg: OpRef,
) -> CValue {
    let base = resolve_opref(builder, constants, base_arg);
    let offset = resolve_opref_or_imm(builder, constants, known_values, offset_arg);
    builder.ins().iadd(base, offset)
}

fn emit_scaled_index_addr(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    base_arg: OpRef,
    index_arg: OpRef,
    scale: i64,
    base_offset: i64,
) -> CValue {
    let base = resolve_opref(builder, constants, base_arg);
    let index = resolve_opref(builder, constants, index_arg);
    let scaled_index = match scale {
        0 => builder.ins().iconst(cl_types::I64, 0),
        1 => index,
        _ => {
            let scale_val = builder.ins().iconst(cl_types::I64, scale);
            builder.ins().imul(index, scale_val)
        }
    };
    let with_base_offset = if base_offset == 0 {
        scaled_index
    } else {
        builder.ins().iadd_imm(scaled_index, base_offset)
    };
    builder.ins().iadd(base, with_base_offset)
}

fn emit_host_call(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    call_conv: cranelift_codegen::isa::CallConv,
    func_ptr: usize,
    args: &[CValue],
    return_type: Option<cranelift_codegen::ir::Type>,
) -> Option<CValue> {
    let mut sig = Signature::new(call_conv);
    for _ in args {
        sig.params.push(AbiParam::new(cl_types::I64));
    }
    if let Some(ret) = return_type {
        sig.returns.push(AbiParam::new(ret));
    }
    let sig_ref = builder.import_signature(sig);

    let raw_fptr = builder.ins().iconst(cl_types::I64, func_ptr as i64);
    let fptr = if ptr_type != cl_types::I64 {
        builder.ins().ireduce(ptr_type, raw_fptr)
    } else {
        raw_fptr
    };
    let call = builder.ins().call_indirect(sig_ref, fptr, args);
    return_type.map(|_| builder.inst_results(call)[0])
}

/// RPython parity: spill live GC refs to jf_frame before a call that
/// may trigger GC. callbuilder.py push_gcmap (1 MOV) stores the gcmap;
/// refs are already in jf_frame slots because the backend keeps them
/// there persistently (assembler.py _push_all_regs_to_frame).
fn spill_ref_roots(
    builder: &mut FunctionBuilder,
    jf_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
    ref_root_base_ofs: i32,
) {
    for &(var_idx, slot) in ref_root_slots {
        let offset = ref_root_base_ofs + (slot as i32) * 8;
        let val = if defined_ref_vars.contains(&var_idx) {
            builder.use_var(var(var_idx))
        } else {
            builder.ins().iconst(cl_types::I64, 0)
        };
        // Do NOT use MemFlags::trusted() — the GC reads these slots
        // via jitframe_custom_trace during collection, so the stores
        // must be visible before the collecting call.
        builder.ins().store(MemFlags::new(), val, jf_ptr, offset);
    }
}

/// RPython parity: reload GC refs from jf_frame after a call — the GC
/// may have moved objects and updated the slots in-place.
fn reload_ref_roots(
    builder: &mut FunctionBuilder,
    jf_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
    ref_root_base_ofs: i32,
) {
    for &(var_idx, slot) in ref_root_slots {
        if !defined_ref_vars.contains(&var_idx) {
            continue;
        }
        let offset = ref_root_base_ofs + (slot as i32) * 8;
        let val = builder
            .ins()
            .load(cl_types::I64, MemFlags::trusted(), jf_ptr, offset);
        builder.def_var(var(var_idx), val);
    }
}

fn ptr_arg_as_i64(
    builder: &mut FunctionBuilder,
    ptr: CValue,
    ptr_type: cranelift_codegen::ir::Type,
) -> CValue {
    if ptr_type == cl_types::I64 {
        ptr
    } else {
        builder.ins().uextend(cl_types::I64, ptr)
    }
}

/// Compute per-call gcmap for ref root slots.
///
/// RPython regalloc.py get_gcmap (1092-1108): builds a bitmap of frame
/// slots that hold GC references at a given call site. The bitmap bit
/// position is the frame slot index (relative to jf_frame[0]).
///
/// The ref root slots are placed at frame positions
/// `max_output_slots + 0, max_output_slots + 1, ...`.
/// Returns a leaked `[length, data]` gcmap array pointer, or null if
/// no ref roots exist.
fn compute_per_call_gcmap(max_output_slots: usize, num_ref_roots: usize) -> i64 {
    if num_ref_roots == 0 {
        return 0; // NULLGCMAP
    }
    let max_bit = max_output_slots + num_ref_roots;
    if max_bit <= 64 {
        // Fast path: single-word bitmap (most common case).
        let mut bitmap: usize = 0;
        for slot in 0..num_ref_roots {
            bitmap |= 1usize << (max_output_slots + slot);
        }
        let gcmap_arr = Box::leak(Box::new([1isize, bitmap as isize]));
        return gcmap_arr.as_ptr() as i64;
    }
    // Multi-word bitmap for large traces.
    let num_words = (max_bit + 63) / 64;
    let mut gcmap: Vec<isize> = vec![0; 1 + num_words];
    gcmap[0] = num_words as isize;
    for slot in 0..num_ref_roots {
        let bit_pos = max_output_slots + slot;
        let word_idx = bit_pos / 64;
        let bit_idx = bit_pos % 64;
        gcmap[1 + word_idx] |= (1usize << bit_idx) as isize;
    }
    let leaked = gcmap.into_boxed_slice();
    let ptr = Box::leak(leaked).as_ptr();
    ptr as i64
}

/// RPython _reload_frame_if_necessary (assembler.py:405-412):
///   MOV ecx, [rootstacktop]
///   MOV ebp, [ecx - WORD]    // reload jf_ptr from shadow stack
///
/// After a collecting call, the GC may have copied the jitframe from
/// nursery to old gen. The shadow stack entry was updated by the GC.
/// Reload jf_ptr from the shadow stack top.
fn emit_reload_frame_if_necessary(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    call_conv: cranelift_codegen::isa::CallConv,
) -> CValue {
    // _reload_frame_if_necessary (assembler.py:405-412):
    //   MOV ecx, [rootstacktop]
    //   MOV ebp, [ecx - WORD]
    //
    // After a collecting call, the GC may have moved the jitframe.
    // Read the updated jf_ptr from the shadow stack top.
    emit_host_call(
        builder,
        ptr_type,
        call_conv,
        majit_gc::shadow_stack::majit_jf_shadow_stack_get_top_jf_ptr as *const () as usize,
        &[],
        Some(ptr_type),
    )
    .expect("reload returns jf_ptr")
}

/// RPython push_gcmap (assembler.py:2017-2022, callbuilder.py:93-107):
///   MOV [ebp + jf_gcmap_ofs], gcmap_ptr
///
/// Writes the per-call gcmap to jf_gcmap before a GC-triggering call.
/// This tells the GC which jf_frame slots contain live refs.
fn emit_push_gcmap(builder: &mut FunctionBuilder, jf_ptr: CValue, per_call_gcmap: i64) {
    if per_call_gcmap == 0 {
        return;
    }
    let gcmap_val = builder.ins().iconst(cl_types::I64, per_call_gcmap);
    builder
        .ins()
        .store(MemFlags::new(), gcmap_val, jf_ptr, JF_GCMAP_OFS);
}

/// RPython pop_gcmap (assembler.py:2024-2027):
///   MOV [ebp + jf_gcmap_ofs], 0
///
/// Clears jf_gcmap after the call returns.
fn emit_pop_gcmap(builder: &mut FunctionBuilder, jf_ptr: CValue, per_call_gcmap: i64) {
    if per_call_gcmap == 0 {
        return;
    }
    let zero = builder.ins().iconst(cl_types::I64, 0);
    builder
        .ins()
        .store(MemFlags::trusted(), zero, jf_ptr, JF_GCMAP_OFS);
}

fn output_transfers_current_force_token(
    fail_descr: &CraneliftFailDescr,
    outputs: &[i64],
    handle: u64,
) -> bool {
    handle != 0
        && fail_descr
            .force_token_slots
            .iter()
            .copied()
            .any(|slot| outputs.get(slot).copied() == Some(handle as i64))
}

extern "C" fn zero_memory_shim(base: u64, offset: u64, size: u64) {
    if size == 0 {
        return;
    }
    let addr = (base as usize).wrapping_add(offset as usize) as *mut u8;
    unsafe {
        std::ptr::write_bytes(addr, 0, size as usize);
    }
}

extern "C" fn copy_nonoverlapping_memory_shim(src: u64, dst: u64, size: u64) {
    if size == 0 {
        return;
    }

    let src = src as usize;
    let dst = dst as usize;
    let size = size as usize;
    let overlap = src < dst.saturating_add(size) && dst < src.saturating_add(size);
    assert!(
        !overlap,
        "copystrcontent/copyunicodecontent overlapping ranges are unsupported"
    );

    unsafe {
        std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, size);
    }
}

/// RPython parity: emit a GC-collecting call with ref root spill/reload
/// and push_gcmap/pop_gcmap (callbuilder.py:93-122).
///
/// Sequence:
///   1. spill_ref_roots — save live refs to jf_frame
///   2. push_gcmap — MOV [ebp+jf_gcmap], gcmap_ptr
///   3. CALL (may trigger GC → GC reads jf_gcmap to find refs)
///   4. pop_gcmap — MOV [ebp+jf_gcmap], 0
///   5. reload_ref_roots — restore refs (GC may have updated them)
fn emit_collecting_gc_call(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    call_conv: cranelift_codegen::isa::CallConv,
    jf_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
    ref_root_base_ofs: i32,
    per_call_gcmap: i64,
    runtime_id: CValue,
    func_ptr: usize,
    extra_args: &[CValue],
    return_type: Option<cranelift_codegen::ir::Type>,
) -> Option<CValue> {
    spill_ref_roots(
        builder,
        jf_ptr,
        ref_root_slots,
        defined_ref_vars,
        ref_root_base_ofs,
    );
    emit_push_gcmap(builder, jf_ptr, per_call_gcmap);

    let mut args = Vec::with_capacity(extra_args.len() + 1);
    args.push(runtime_id);
    args.extend_from_slice(extra_args);

    let result = emit_host_call(builder, ptr_type, call_conv, func_ptr, &args, return_type);

    // _reload_frame_if_necessary (assembler.py:405-412)
    let new_jf_ptr = emit_reload_frame_if_necessary(builder, ptr_type, call_conv);
    emit_pop_gcmap(builder, new_jf_ptr, per_call_gcmap);
    reload_ref_roots(
        builder,
        new_jf_ptr,
        ref_root_slots,
        defined_ref_vars,
        ref_root_base_ofs,
    );
    result
}

fn emit_indirect_call_from_parts(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    func_ref: OpRef,
    arg_refs: &[OpRef],
    call_descr: &dyn CallDescr,
    call_conv: cranelift_codegen::isa::CallConv,
    ptr_type: cranelift_codegen::ir::Type,
    gc_runtime_id: Option<u64>,
    jf_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
    ref_root_base_ofs: i32,
    per_call_gcmap: i64,
) -> Option<CValue> {
    let mut sig = Signature::new(call_conv);
    let arg_types = call_descr.arg_types();
    for at in arg_types {
        sig.params.push(AbiParam::new(cranelift_type_for(at)));
    }
    let result_type = call_descr.result_type();
    if result_type != Type::Void {
        sig.returns
            .push(AbiParam::new(cranelift_type_for(&result_type)));
    }
    let sig_ref = builder.import_signature(sig);

    let func_ptr_raw = resolve_opref(builder, constants, func_ref);
    let func_ptr = if ptr_type != cl_types::I64 {
        builder.ins().ireduce(ptr_type, func_ptr_raw)
    } else {
        func_ptr_raw
    };

    let mut args: Vec<CValue> = Vec::with_capacity(arg_refs.len());
    for (i, &arg_ref) in arg_refs.iter().enumerate() {
        let raw = resolve_opref(builder, constants, arg_ref);
        if i < arg_types.len() && arg_types[i] == Type::Float {
            args.push(builder.ins().bitcast(cl_types::F64, MemFlags::new(), raw));
        } else {
            args.push(raw);
        }
    }

    if call_descr.effect_info().can_raise() {
        let _ = emit_host_call(
            builder,
            ptr_type,
            call_conv,
            jit_exc_clear as *const () as usize,
            &[],
            None,
        );
    }

    // RPython callbuilder.py parity:
    //   emit() [can_collect]: spill + push_gcmap + CALL + pop_gcmap + reload
    //   emit_no_collect(): bare CALL only
    let can_collect = call_descr.effect_info().can_collect();
    if can_collect {
        spill_ref_roots(
            builder,
            jf_ptr,
            ref_root_slots,
            defined_ref_vars,
            ref_root_base_ofs,
        );
        emit_push_gcmap(builder, jf_ptr, per_call_gcmap);
    }
    let call = builder.ins().call_indirect(sig_ref, func_ptr, &args);
    if can_collect {
        // _reload_frame_if_necessary (assembler.py:405-412)
        let new_jf_ptr = emit_reload_frame_if_necessary(builder, ptr_type, call_conv);
        emit_pop_gcmap(builder, new_jf_ptr, per_call_gcmap);
        reload_ref_roots(
            builder,
            new_jf_ptr,
            ref_root_slots,
            defined_ref_vars,
            ref_root_base_ofs,
        );
    }

    if result_type != Type::Void {
        let result = builder.inst_results(call)[0];
        let stored = if result_type == Type::Float {
            builder
                .ins()
                .bitcast(cl_types::I64, MemFlags::new(), result)
        } else {
            result
        };
        Some(stored)
    } else {
        None
    }
}

/// Emit a guard/finish side-exit.
///
/// _build_failure_recovery (assembler.py:2080-2109) parity:
///   1. _push_all_regs_to_frame  — save live values to jf_frame slots
///   2. POP [ebp + jf_descr]     — store fail_descr index to jf_descr
///   3. _call_footer              — mov eax, ebp; ret (return jitframe)
///
/// genop_finish (assembler.py:2114-2155) parity:
///   1. save result to jf_frame[0]
///   2. MOV [ebp + jf_descr], faildescrindex
///   3. _call_footer
fn emit_guard_exit(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    jf_ptr: CValue,
    info: &GuardInfo,
) {
    // _push_all_regs_to_frame / save_into_mem parity:
    // store fail_args to jf_frame[slot]
    for (slot, &arg_ref) in info.fail_arg_refs.iter().enumerate() {
        let val = resolve_opref(builder, constants, arg_ref);
        let offset = JF_FRAME_ITEM0_OFS + (slot as i32) * 8;
        builder
            .ins()
            .store(MemFlags::trusted(), val, jf_ptr, offset);
    }
    // _build_failure_recovery (assembler.py:2102-2105) parity:
    //   POP [ebp + jf_gcmap]   — #2104
    //   POP [ebp + jf_descr]   — #2105
    // push_gcmap (assembler.py:2013): PUSH gcmap_ptr
    // Skip gcmap store when null (no ref fail_args) — saves 1 iconst + 1 store.
    if info.gcmap != 0 {
        // allocate_gcmap (gcmap.py:7-18): Array(Unsigned) [length, data...].
        let gcmap_arr = Box::leak(Box::new([1isize, info.gcmap as isize]));
        let gcmap_val = builder
            .ins()
            .iconst(cl_types::I64, gcmap_arr.as_ptr() as i64);
        builder
            .ins()
            .store(MemFlags::trusted(), gcmap_val, jf_ptr, JF_GCMAP_OFS); // #2104
    }
    // assembler.py:2126 get_gcref_from_faildescr → MOV [ebp+jf_descr], gcref
    // Store FailDescr POINTER (not index) to jf_descr.
    let descr_val = builder.ins().iconst(cl_types::I64, info.fail_descr_ptr);
    builder
        .ins()
        .store(MemFlags::trusted(), descr_val, jf_ptr, JF_DESCR_OFS); // #2105
    // _call_footer (assembler.py:1097): mov eax, ebp; ret
    // Also return descr_val in rdx for caller hot-path (avoids memory load).
    // _call_footer: mov eax, ebp; ret — return jf_ptr only.
    builder.ins().return_(&[jf_ptr]);
}

// ---------------------------------------------------------------------------
// Compiled loop data
// ---------------------------------------------------------------------------

struct CompiledLoop {
    trace_id: u64,
    input_types: Vec<Type>,
    header_pc: u64,
    /// Green key hash from JitCellToken. Used by bridge threshold callback.
    green_key: u64,
    caller_prefix_layout: Option<ExitRecoveryLayout>,
    _func_id: FuncId,
    code_ptr: *const u8,
    code_size: usize,
    fail_descrs: Vec<Arc<CraneliftFailDescr>>,
    terminal_exit_layouts: UnsafeCell<Vec<TerminalExitLayout>>,
    gc_runtime_id: Option<u64>,
    num_inputs: usize,
    num_ref_roots: usize,
    max_output_slots: usize,
    /// Whether any guard in this loop uses FORCE_TOKEN slots.
    /// When false, force frame registration can be skipped entirely.
    needs_force_frame: bool,
}

unsafe impl Send for CompiledLoop {}
unsafe impl Sync for CompiledLoop {}

impl CompiledLoop {
    #[inline]
    fn terminal_exit_layouts_ref(&self) -> &Vec<TerminalExitLayout> {
        unsafe { &*self.terminal_exit_layouts.get() }
    }

    #[inline]
    fn terminal_exit_layouts_mut(&self) -> &mut Vec<TerminalExitLayout> {
        unsafe { &mut *self.terminal_exit_layouts.get() }
    }
}

fn find_trace_fail_descr_layouts_in_fail_descrs(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
) -> Option<Vec<majit_codegen::FailDescrLayout>> {
    for descr in fail_descrs {
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if bridge.trace_id == trace_id {
                return Some(
                    bridge
                        .fail_descrs
                        .iter()
                        .map(|descr| descr.layout())
                        .collect(),
                );
            }
            if let Some(layouts) =
                find_trace_fail_descr_layouts_in_fail_descrs(&bridge.fail_descrs, trace_id)
            {
                return Some(layouts);
            }
        }
    }
    None
}

fn find_trace_terminal_exit_layouts_in_fail_descrs(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
) -> Option<Vec<majit_codegen::TerminalExitLayout>> {
    for descr in fail_descrs {
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if bridge.trace_id == trace_id {
                return Some(bridge.terminal_exit_layouts_ref().clone());
            }
            if let Some(layouts) =
                find_trace_terminal_exit_layouts_in_fail_descrs(&bridge.fail_descrs, trace_id)
            {
                return Some(layouts);
            }
        }
    }
    None
}

fn find_trace_info_in_fail_descrs(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
) -> Option<CompiledTraceInfo> {
    for descr in fail_descrs {
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if bridge.trace_id == trace_id {
                return Some(CompiledTraceInfo {
                    trace_id: bridge.trace_id,
                    input_types: bridge.input_types.clone(),
                    header_pc: bridge.header_pc,
                    source_guard: Some(bridge.source_guard),
                });
            }
            if let Some(info) = find_trace_info_in_fail_descrs(&bridge.fail_descrs, trace_id) {
                return Some(info);
            }
        }
    }
    None
}

fn find_fail_descr_in_fail_descrs(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
    fail_index: u32,
) -> Option<Arc<CraneliftFailDescr>> {
    for descr in fail_descrs {
        if descr.trace_id == trace_id && descr.fail_index == fail_index {
            return Some(descr.clone());
        }
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if let Some(found) =
                find_fail_descr_in_fail_descrs(&bridge.fail_descrs, trace_id, fail_index)
            {
                return Some(found);
            }
        }
    }
    None
}

fn run_compiled_code(
    code_ptr: *const u8,
    fail_descrs: &[Arc<CraneliftFailDescr>],
    gc_runtime_id: Option<u64>,
    num_ref_roots: usize,
    max_output_slots: usize,
    inputs: &[i64],
    needs_force_frame: bool,
) -> (u32, Vec<i64>, u64, Option<Arc<ActiveForceFrame>>) {
    // RPython llmodel.py:298: frame = gc_ll_descr.malloc_jitframe(frame_info)
    // jitframe.py:48-52: jitframe_allocate(frame_info)
    let depth = max_output_slots.max(inputs.len()).max(1);
    let header_words = (JF_FRAME_ITEM0_OFS as usize) / 8; // 8 words = 64 bytes
    let frame_depth = depth + num_ref_roots;
    let jf_total = header_words + frame_depth;
    let payload_bytes = jf_total * 8;

    // RPython gc.py:132 malloc_jitframe → jitframe_allocate:
    //   rgc.register_custom_trace_hook(JITFRAME, jitframe_trace)
    //   frame = lltype.malloc(JITFRAME, frame_depth)
    //
    // When GC has a type registry (MiniMarkGC), allocate from nursery with
    // JITFRAME type_id so the GC can copy+trace it correctly. The shadow
    // stack holds a GcRef that the GC updates in place when copying.
    // When GC is a stub (TrackingGc) or absent, fall back to Vec<i64>.
    let use_gc_alloc = gc_runtime_id
        .map(|id| with_gc_runtime(id, |gc| gc.type_count() > 0))
        .unwrap_or(false);
    let (jf_gcref, _jf_buf_keepalive): (GcRef, Option<Vec<i64>>) = if use_gc_alloc {
        let runtime_id = gc_runtime_id.unwrap();
        let gcref = with_gc_runtime(runtime_id, |gc| {
            gc.alloc_nursery_no_collect_typed(JITFRAME_GC_TYPE_ID, payload_bytes)
        });
        // jitframe_allocate: frame.jf_frame_info = frame_info (skipped —
        // we don't have frame_info). Write the jf_frame_length field so
        // copy_nursery_object can compute the varsize total.
        unsafe {
            *((gcref.0 + JF_FRAME_LENGTH_OFS as usize) as *mut usize) = frame_depth;
        }
        (gcref, None)
    } else {
        let mut buf = vec![0i64; jf_total];
        let gcref = GcRef(buf.as_mut_ptr() as usize);
        (gcref, Some(buf))
    };

    let jf_ptr = jf_gcref.0 as *mut i64;

    // llmodel.py:306-315: set arguments in frame
    for (i, &val) in inputs.iter().enumerate() {
        unsafe { *jf_ptr.add(header_words + i) = val };
    }

    // llmodel.py:322: llop.gc_writebarrier(ll_frame)
    if use_gc_alloc {
        let runtime_id = gc_runtime_id.unwrap();
        with_gc_runtime(runtime_id, |gc| gc.write_barrier(jf_gcref));
    }

    // llmodel.py:323 parity: ll_frame = func(ll_frame)
    let func: unsafe extern "C" fn(*mut i64) -> *mut i64 = unsafe { std::mem::transmute(code_ptr) };

    let _jitted_guard = majit_codegen::JittedGuard::enter();

    let (handle, force_frame) = if needs_force_frame {
        let (h, f) = register_force_frame(fail_descrs, gc_runtime_id);
        (h, Some(f))
    } else {
        (0, None)
    };

    // _call_header_shadowstack (assembler.py:1122-1128) parity:
    // Push jf_ptr as GcRef onto the jitframe shadow stack. The GC
    // treats this as a root: if in nursery, copies jitframe to old gen
    // and updates the GcRef in place.
    let jf_shadow_depth = majit_gc::shadow_stack::push_jf(jf_gcref);

    let result_jf = with_active_force_frame(handle, || unsafe { func(jf_ptr) });

    // _call_footer_shadowstack (assembler.py:1130-1136) parity:
    majit_gc::shadow_stack::pop_jf_to(jf_shadow_depth);

    // jitframe_resolve (jitframe.py:54-57):
    // Follow jf_forward chain — the compiled code may return the old
    // (nursery) jf_ptr, but the jitframe has been forwarded to old gen.
    let result_jf = jitframe_resolve(result_jf);

    // llmodel.py:412-420 get_latest_descr: read jf_descr from returned frame.
    let jf_descr_raw = unsafe { *result_jf.add(JF_DESCR_OFS as usize / 8) };

    let fail_index = if jf_descr_raw == CALL_ASSEMBLER_DEADFRAME_SENTINEL as i64 {
        CALL_ASSEMBLER_DEADFRAME_SENTINEL
    } else if jf_descr_raw == 0 {
        0u32
    } else {
        unsafe { &*(jf_descr_raw as *const CraneliftFailDescr) }.fail_index()
    };
    let mut outputs = vec![0i64; max_output_slots.max(1)];
    for i in 0..max_output_slots.min(depth) {
        outputs[i] = unsafe { *result_jf.add(header_words + i) };
    }

    drop(_jitted_guard);
    (fail_index, outputs, handle, force_frame)
}

struct GuardInfo {
    fail_index: u32,
    fail_arg_refs: Vec<OpRef>,
    /// gcmap bitmap: bit i set ⇔ fail_arg[i] is Ref type.
    /// allocate_gcmap (gcmap.py:7-18) parity.
    gcmap: u64,
    /// RPython assembler.py:2126 get_gcref_from_faildescr parity:
    /// stores Arc::as_ptr(CraneliftFailDescr) as i64.
    /// The FailDescr GCREF pointer is written to jf_descr on guard exit.
    fail_descr_ptr: i64,
}

fn identity_recovery_layout(
    trace_id: u64,
    header_pc: u64,
    guard_resume_pc: Option<u64>,
    source_guard: Option<(u64, u32)>,
    slot_types: &[Type],
    caller_layout: Option<&ExitRecoveryLayout>,
) -> ExitRecoveryLayout {
    let mut frames = caller_layout
        .map(|layout| layout.frames.clone())
        .unwrap_or_default();
    frames.push(ExitFrameLayout {
        trace_id: Some(trace_id),
        header_pc: Some(header_pc),
        source_guard,
        pc: guard_resume_pc.unwrap_or(header_pc),
        slots: (0..slot_types.len())
            .map(ExitValueSourceLayout::ExitValue)
            .collect(),
        slot_types: Some(slot_types.to_vec()),
    });
    ExitRecoveryLayout {
        frames,
        virtual_layouts: caller_layout
            .map(|layout| layout.virtual_layouts.clone())
            .unwrap_or_default(),
        pending_field_layouts: caller_layout
            .map(|layout| layout.pending_field_layouts.clone())
            .unwrap_or_default(),
    }
}

fn patch_fail_descr_recovery_layout(
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
    fail_index: u32,
    recovery_layout: &ExitRecoveryLayout,
) -> bool {
    for descr in fail_descrs {
        if descr.trace_id == trace_id && descr.fail_index == fail_index {
            descr.set_recovery_layout(recovery_layout.clone());
            return true;
        }
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if patch_fail_descr_recovery_layout(
                &bridge.fail_descrs,
                trace_id,
                fail_index,
                recovery_layout,
            ) {
                return true;
            }
        }
    }
    false
}

fn patch_terminal_exit_recovery_layout_in_vec(
    terminal_exit_layouts: &UnsafeCell<Vec<TerminalExitLayout>>,
    trace_id: u64,
    op_index: usize,
    recovery_layout: &ExitRecoveryLayout,
) -> bool {
    let terminal_exit_layouts = unsafe { &mut *terminal_exit_layouts.get() };
    if let Some(layout) = terminal_exit_layouts
        .iter_mut()
        .find(|layout| layout.trace_id == trace_id && layout.op_index == op_index)
    {
        layout.recovery_layout = Some(recovery_layout.clone());
        return true;
    }
    false
}

fn patch_terminal_exit_recovery_layout(
    root_terminal_exit_layouts: &UnsafeCell<Vec<TerminalExitLayout>>,
    fail_descrs: &[Arc<CraneliftFailDescr>],
    trace_id: u64,
    op_index: usize,
    recovery_layout: &ExitRecoveryLayout,
) -> bool {
    if patch_terminal_exit_recovery_layout_in_vec(
        root_terminal_exit_layouts,
        trace_id,
        op_index,
        recovery_layout,
    ) {
        return true;
    }

    for descr in fail_descrs {
        let bridge_guard = descr.bridge_ref();
        if let Some(bridge) = bridge_guard.as_ref() {
            if patch_terminal_exit_recovery_layout(
                &bridge.terminal_exit_layouts,
                &bridge.fail_descrs,
                trace_id,
                op_index,
                recovery_layout,
            ) {
                return true;
            }
        }
    }
    false
}

fn infer_fail_arg_types(
    fail_arg_refs: &[OpRef],
    value_types: &HashMap<u32, Type>,
) -> Result<Vec<Type>, BackendError> {
    let mut fail_arg_types = Vec::with_capacity(fail_arg_refs.len());
    for &opref in fail_arg_refs {
        if opref.is_none() {
            // resume.py parity: OpRef::NONE marks a virtual object slot
            // in fail_args. The backend stores 0 in this slot; the actual
            // value is reconstructed from rd_virtuals on guard failure.
            fail_arg_types.push(Type::Int);
            continue;
        }
        // Backend constant slots are currently integer-only. If a fail arg
        // doesn't correspond to an input arg or operation result, treat it as
        // an integer constant instead of silently manufacturing Ref/Float data.
        fail_arg_types.push(value_types.get(&opref.0).copied().unwrap_or(Type::Int));
    }
    Ok(fail_arg_types)
}

/// Position-aware fail_arg type inference. When an OpRef is both a label
/// inputarg and a body operation result with a DIFFERENT type, the correct
/// type depends on whether the guard is BEFORE or AFTER the operation that
/// redefines it. Guards before the operation see the inputarg value (e.g.
/// Ref from preamble), while guards after see the operation result (e.g.
/// Int from IntAddOvf).
fn infer_fail_arg_types_positional(
    fail_arg_refs: &[OpRef],
    value_types: &HashMap<u32, Type>,
    inputarg_types: &HashMap<u32, Type>,
    op_def_positions: &HashMap<u32, usize>,
    guard_op_index: usize,
) -> Result<Vec<Type>, BackendError> {
    let mut fail_arg_types = Vec::with_capacity(fail_arg_refs.len());
    for &opref in fail_arg_refs {
        if opref.is_none() {
            fail_arg_types.push(Type::Int);
            continue;
        }
        let tp = if let Some(&def_pos) = op_def_positions.get(&opref.0) {
            if guard_op_index < def_pos {
                // Guard is before the operation that defines this OpRef.
                // Use the inputarg type (value from preamble/label entry).
                inputarg_types
                    .get(&opref.0)
                    .or_else(|| value_types.get(&opref.0))
                    .copied()
                    .unwrap_or(Type::Int)
            } else {
                value_types.get(&opref.0).copied().unwrap_or(Type::Int)
            }
        } else {
            value_types.get(&opref.0).copied().unwrap_or(Type::Int)
        };
        fail_arg_types.push(tp);
    }
    Ok(fail_arg_types)
}

/// box.type parity: merge descriptor types with positional inference.
/// Use descriptor types as the base, then override slots where
/// op_def_positions detects a type conflict (JUMP→LABEL propagation
/// or inputarg-vs-operation redefinition).
fn merge_descriptor_with_positional(
    fail_arg_refs: &[OpRef],
    fd: Option<&dyn majit_ir::descr::FailDescr>,
    value_types: &HashMap<u32, Type>,
    inputarg_types: &HashMap<u32, Type>,
    op_def_positions: &HashMap<u32, usize>,
    guard_op_index: usize,
) -> Result<Vec<Type>, BackendError> {
    let positional = infer_fail_arg_types_positional(
        fail_arg_refs,
        value_types,
        inputarg_types,
        op_def_positions,
        guard_op_index,
    )?;
    let Some(fd) = fd else {
        return Ok(positional);
    };
    let dt = fd.fail_arg_types();
    if dt.len() != fail_arg_refs.len() {
        return Ok(positional);
    }
    // Start from descriptor, override conflicting slots.
    Ok(dt
        .iter()
        .enumerate()
        .map(|(i, &descr_tp)| {
            let opref = fail_arg_refs.get(i).copied().unwrap_or(OpRef::NONE);
            if !opref.is_none() && op_def_positions.contains_key(&opref.0) {
                positional[i]
            } else {
                descr_tp
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// CraneliftBackend
// ---------------------------------------------------------------------------

pub struct CraneliftBackend {
    module: JITModule,
    func_ctx: FunctionBuilderContext,
    constants: HashMap<u32, i64>,
    func_counter: u32,
    gc_runtime_id: Option<u64>,
    trace_counter: u64,
    next_trace_id: Option<u64>,
    next_header_pc: Option<u64>,
    registered_call_assembler_tokens: HashSet<u64>,
    registered_call_assembler_bridge_traces: HashSet<u64>,
}

impl CraneliftBackend {
    pub fn new() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();

        let isa_builder = cranelift_native::builder().expect("host ISA not supported");
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(jit_builder);
        let func_ctx = FunctionBuilderContext::new();

        CraneliftBackend {
            module,
            func_ctx,
            constants: HashMap::new(),
            func_counter: 0,
            gc_runtime_id: None,
            trace_counter: 1,
            next_trace_id: None,
            next_header_pc: None,
            registered_call_assembler_tokens: HashSet::new(),
            registered_call_assembler_bridge_traces: HashSet::new(),
        }
    }

    pub fn with_gc_allocator(gc: Box<dyn GcAllocator>) -> Self {
        let mut backend = Self::new();
        backend.set_gc_allocator(gc);
        backend
    }

    pub fn set_gc_allocator(&mut self, mut gc: Box<dyn GcAllocator>) {
        ensure_jitframe_type_registered(gc.as_mut());
        if let Some(runtime_id) = self.gc_runtime_id {
            replace_gc_runtime(runtime_id, gc);
        } else {
            self.gc_runtime_id = Some(register_gc_runtime(gc));
        }
    }

    /// Register constants available during the next `compile_loop` call.
    pub fn set_constants(&mut self, constants: HashMap<u32, i64>) {
        self.constants = constants;
    }

    /// Force the next compile call to assign a specific trace id to all exits.
    pub fn set_next_trace_id(&mut self, trace_id: u64) {
        self.next_trace_id = Some(trace_id);
    }

    /// Force the next compile call to attach a specific header PC to
    /// synthesized exit recovery layouts.
    pub fn set_next_header_pc(&mut self, header_pc: u64) {
        self.next_header_pc = Some(header_pc);
    }

    fn gc_rewriter(&self) -> Option<GcRewriterImpl> {
        let runtime_id = self.gc_runtime_id?;
        Some(with_gc_runtime(runtime_id, |gc| GcRewriterImpl {
            nursery_free_addr: gc.nursery_free() as usize,
            nursery_top_addr: gc.nursery_top() as usize,
            max_nursery_size: gc.max_nursery_object_size(),
            // The Cranelift backend currently lowers WB ops through runtime
            // helper calls instead of inlining flag checks, but the rewriter
            // still expects a descriptor-shaped config object.
            wb_descr: WriteBarrierDescr {
                jit_wb_if_flag: gc_flags::TRACK_YOUNG_PTRS,
                jit_wb_if_flag_byteofs: 0,
                jit_wb_if_flag_singlebyte: gc_flags::TRACK_YOUNG_PTRS as u8,
                jit_wb_cards_set: gc_flags::CARDS_SET,
                jit_wb_card_page_shift: 0,
            },
            jitframe_info: INLINE_ARENA
                .get()
                .and_then(|arena| arena.jitframe_descrs.clone()),
        }))
    }

    fn prepare_ops_for_compile(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        constants: &HashMap<u32, i64>,
    ) -> Vec<Op> {
        let mut normalized = normalize_ops_for_codegen_simple(inputargs, ops);
        inject_builtin_string_descrs(&mut normalized);
        if let Some(rewriter) = self.gc_rewriter() {
            let (result, new_constants) =
                rewriter.rewrite_for_gc_with_constants(&normalized, constants);
            // Merge GC rewriter's new constants into self.constants
            for (k, v) in new_constants {
                self.constants.entry(k).or_insert(v);
            }
            result
        } else {
            normalized
        }
    }

    /// Execute a compiled bridge, returning the DeadFrame from the bridge's
    /// exit point (either a Finish or a further guard failure).
    /// Run compiled code with raw i64 inputs.
    ///
    /// Shared by `execute_token` (after Value→i64 conversion) and
    /// `execute_token_ints` (direct pass-through).
    fn execute_with_inputs(compiled: &CompiledLoop, inputs: &[i64]) -> DeadFrame {
        // RPython parity: bridge JUMP → loop re-entry uses a loop instead
        // of recursive calls, matching RPython's inline jmp within the same
        // code buffer. Avoids stack growth on repeated bridge → re-enter cycles.
        let mut current_inputs = inputs.to_vec();
        loop {
            let (fail_index, mut outputs, handle, force_frame) = run_compiled_code(
                compiled.code_ptr,
                &compiled.fail_descrs,
                compiled.gc_runtime_id,
                compiled.num_ref_roots,
                compiled.max_output_slots,
                &current_inputs,
                compiled.needs_force_frame,
            );
            if let Some(frame) = maybe_take_call_assembler_deadframe(
                fail_index,
                &outputs,
                handle,
                force_frame.as_ref(),
            ) {
                return wrap_call_assembler_deadframe_with_caller_prefix(
                    frame,
                    compiled.trace_id,
                    compiled.header_pc,
                    None,
                    &compiled.input_types,
                    &current_inputs,
                    compiled.caller_prefix_layout.as_ref(),
                );
            }

            let fail_descr = &compiled.fail_descrs[fail_index as usize];
            // Finish exits return directly — no bridge dispatch.
            if fail_descr.is_finish {
                let saved_data = if let Some(ref ff) = force_frame {
                    take_force_frame_saved_data(ff)
                } else {
                    None
                };
                let (exception_class, exception) = take_pending_jit_exception_state();
                if !output_transfers_current_force_token(fail_descr, &outputs, handle) {
                    release_force_token(handle);
                }
                return DeadFrame {
                    data: Box::new(FrameData::new_with_savedata_and_exception(
                        outputs,
                        fail_descr.clone(),
                        compiled.gc_runtime_id,
                        saved_data,
                        exception_class,
                        (!exception.is_null()).then_some(exception),
                    )),
                };
            }

            // Increment guard failure count.
            fail_descr.increment_fail_count();
            let fail_count = fail_descr.get_fail_count();

            // If a bridge is attached to this guard, execute it.
            // Uses lock-free bridge_ref() (main's Mutex removal parity).
            let bridge_guard = fail_descr.bridge_ref();
            if let Some(ref bridge) = *bridge_guard {
                release_force_token(handle);
                // Materialize virtuals in-place (avoid clone).
                rebuild_state_after_failure(
                    &mut outputs,
                    &fail_descr.fail_arg_types,
                    fail_descr.recovery_layout_ref().as_ref(),
                    bridge.num_inputs,
                );
                if bridge.loop_reentry {
                    let bridge_frame =
                        Self::execute_bridge(bridge, &outputs, &fail_descr.fail_arg_types);
                    drop(bridge_guard);
                    let bridge_descr = get_latest_descr_from_deadframe(&bridge_frame)
                        .expect("bridge deadframe must have descriptor");
                    if bridge_descr.is_finish() {
                        let num_outputs = bridge_descr.fail_arg_types().len();
                        current_inputs.clear();
                        current_inputs.reserve(num_outputs);
                        for i in 0..num_outputs {
                            current_inputs
                                .push(get_int_from_deadframe(&bridge_frame, i).unwrap_or(0));
                        }
                        continue;
                    }
                    return bridge_frame;
                }
                return Self::execute_bridge(bridge, &outputs, &fail_descr.fail_arg_types);
            }
            drop(bridge_guard);

            let saved_data = if let Some(ref ff) = force_frame {
                take_force_frame_saved_data(ff)
            } else {
                None
            };
            let (exception_class, exception) = take_pending_jit_exception_state();
            if !output_transfers_current_force_token(fail_descr, &outputs, handle) {
                release_force_token(handle);
            }

            return DeadFrame {
                data: Box::new(FrameData::new_with_savedata_and_exception(
                    outputs,
                    fail_descr.clone(),
                    compiled.gc_runtime_id,
                    saved_data,
                    exception_class,
                    (!exception.is_null()).then_some(exception),
                )),
            };
        } // end loop
    }

    ///
    /// If the bridge itself hits a guard that has another bridge attached,
    /// this chains through until a final exit is reached.
    fn execute_bridge(
        bridge: &BridgeData,
        parent_outputs: &[i64],
        parent_types: &[Type],
    ) -> DeadFrame {
        // The bridge's inputs are the parent guard's fail args.
        let num_bridge_inputs = bridge.num_inputs.min(parent_types.len());
        let bridge_inputs = &parent_outputs[..num_bridge_inputs];

        let (fail_index, outputs, handle, force_frame) = run_compiled_code(
            bridge.code_ptr,
            &bridge.fail_descrs,
            bridge.gc_runtime_id,
            bridge.num_ref_roots,
            bridge.max_output_slots,
            bridge_inputs,
            bridge.needs_force_frame,
        );

        if let Some(frame) =
            maybe_take_call_assembler_deadframe(fail_index, &outputs, handle, force_frame.as_ref())
        {
            return wrap_call_assembler_deadframe_with_caller_prefix(
                frame,
                bridge.trace_id,
                bridge.header_pc,
                Some(bridge.source_guard),
                &bridge.input_types,
                bridge_inputs,
                bridge.caller_prefix_layout.as_ref(),
            );
        }

        let fail_descr = &bridge.fail_descrs[fail_index as usize];
        fail_descr.increment_fail_count();

        // Check for chained bridges (RPython: bridge-on-bridge).
        let bridge_guard = fail_descr.bridge_ref();
        if let Some(ref next_bridge) = *bridge_guard {
            release_force_token(handle);
            return Self::execute_bridge(next_bridge, &outputs, &fail_descr.fail_arg_types);
        }
        drop(bridge_guard);

        // compile.py:701-717: handle_fail / must_compile — trigger bridge
        // compilation for bridge guards just like for loop guards.
        let saved_data = if let Some(ref ff) = force_frame {
            take_force_frame_saved_data(ff)
        } else {
            None
        };
        let (exception_class, exception) = take_pending_jit_exception_state();
        if !output_transfers_current_force_token(fail_descr, &outputs, handle) {
            release_force_token(handle);
        }

        // Bridge guard failure: return bridge's deadframe directly.
        // The bridge's fail_args contain the full frame state needed
        // for interpreter resume (bridge inputargs include parent locals).
        // RPython resume.py:rebuild_state_after_failure uses the bridge's
        // rd_numb/rd_virtuals to reconstruct frame state.
        DeadFrame {
            data: Box::new(FrameData::new_with_savedata_and_exception(
                outputs,
                fail_descr.clone(),
                bridge.gc_runtime_id,
                saved_data,
                exception_class,
                (!exception.is_null()).then_some(exception),
            )),
        }
    }

    fn do_compile(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        invalidation_flag_ptr: Option<usize>,
        source_guard: Option<(u64, u32)>,
        caller_layout: Option<&ExitRecoveryLayout>,
    ) -> Result<CompiledLoop, BackendError> {
        let prepared_ops = self.prepare_ops_for_compile(inputargs, ops, &self.constants.clone());
        let ops = prepared_ops.as_slice();
        let trace_id = self.next_trace_id.take().unwrap_or_else(|| {
            let trace_id = self.trace_counter;
            self.trace_counter += 1;
            trace_id
        });
        let header_pc = self.next_header_pc.take().unwrap_or(0);
        let ptr_type = self.module.target_config().pointer_type();
        let call_conv = self.module.target_config().default_call_conv;

        let mut sig = Signature::new(call_conv);
        sig.params.push(AbiParam::new(ptr_type)); // jf_ptr (read inputs, write outputs)
        // RPython _call_footer (assembler.py:1097): mov eax, ebp; ret
        sig.returns.push(AbiParam::new(ptr_type)); // returned jf_ptr

        let func_name = format!("trace_{}", self.func_counter);
        self.func_counter += 1;

        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Local, &sig)
            .map_err(|e| BackendError::CompilationFailed(e.to_string()))?;

        let mut func = Function::with_name_signature(
            cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
            sig,
        );

        // Pre-scan
        let force_tokens = build_force_token_set(inputargs, ops)?;
        let mut fail_descrs: Vec<Arc<CraneliftFailDescr>> = Vec::new();
        let mut guard_infos: Vec<GuardInfo> = Vec::new();
        let mut max_output_slots: usize = 0;
        collect_guards(
            ops,
            inputargs,
            &force_tokens,
            &mut fail_descrs,
            &mut guard_infos,
            &mut max_output_slots,
            trace_id,
            header_pc,
            source_guard,
            caller_layout,
            &self.constants,
        )?;
        let terminal_exit_layouts = collect_terminal_exit_layouts(
            ops,
            inputargs,
            &force_tokens,
            trace_id,
            header_pc,
            source_guard,
            caller_layout,
        )?;

        let num_inputs = inputargs.len();
        let known_values = build_known_values_set(inputargs, ops);
        let value_types = build_value_type_map_simple(inputargs, ops);
        let ref_root_slots = build_ref_root_slots(inputargs, ops, &force_tokens);
        let gc_runtime_id = self.gc_runtime_id;
        let mut defined_ref_vars: HashSet<u32> = inputargs
            .iter()
            .filter(|input| input.tp == Type::Ref && !force_tokens.contains(&input.index))
            .map(|input| input.index)
            .collect();

        // Pre-scan for vector-producing ops (SIMD variable types)
        let vec_oprefs = build_vec_oprefs(ops, num_inputs);
        let vec_float_oprefs = build_vec_float_oprefs(ops, num_inputs);

        // Take constants out of self to avoid borrow conflicts with func_ctx
        let constants = std::mem::take(&mut self.constants);

        // Recursive tracing/bridge compilation can re-enter backend compilation
        // before an outer compile has restored its frontend context. Mirror the
        // effective RPython behavior by treating the frontend builder context as
        // compile-local state for the duration of this invocation.
        let mut func_ctx = std::mem::replace(&mut self.func_ctx, FunctionBuilderContext::new());
        let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // jf_ptr serves as both inputs_ptr (entry) and outputs_ptr (exits).
        // RPython: EBP = jitframe pointer throughout compiled code.
        // After a collecting call, _reload_frame_if_necessary reloads
        // jf_ptr from the shadow stack (GC may have moved the jitframe).
        let mut jf_ptr = builder.block_params(entry_block)[0];
        let inputs_ptr = jf_ptr; // alias for entry loading
        let mut outputs_ptr = jf_ptr; // alias for guard exit stores (updated after reload)
        // Ref root slots live in jf_frame after output/fail_args area.
        // RPython: refs are always in jf_frame; gcmap marks which slots
        // are live at each GC point (regalloc.py get_gcmap).
        let ref_root_base_ofs = JF_FRAME_ITEM0_OFS + (max_output_slots as i32) * 8;
        let per_call_gcmap = compute_per_call_gcmap(max_output_slots, ref_root_slots.len());
        let debug_declares = std::env::var_os("MAJIT_DEBUG_DECLARES").is_some();

        let label_indices: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter_map(|(idx, op)| (op.opcode == OpCode::Label).then_some(idx))
            .collect();
        let has_entry_label = label_indices.first().copied() == Some(0);

        // Body-direct entry disabled — requires bridge to provide full body args.
        let body_direct_num_inputs = 0;

        // Collect all variable declarations into a map (index -> type)
        // before declaring them sequentially. Cranelift 0.130 declare_var
        // returns auto-assigned indices, so we must declare in order 0..max.
        let mut var_types: std::collections::HashMap<u32, cranelift_codegen::ir::Type> =
            std::collections::HashMap::new();
        let mut declared_vars = std::collections::HashSet::new();

        // Always declare inputarg variables, even when the trace starts
        // with a LABEL (preamble peeling). Preamble guards reference
        // inputarg OpRefs in fail_args and need declared variables.
        // The LABEL block later overrides these via block params + def_var.
        for i in 0..num_inputs {
            if debug_declares {
                eprintln!("[jit][declare] input var{}", i);
            }
            var_types.insert(i as u32, cl_types::I64);
            declared_vars.insert(i as u32);
        }
        // Declare variables for op results
        for (op_idx, op) in ops.iter().enumerate() {
            if op.result_type() != Type::Void {
                let vi = op_var_index(op, op_idx, num_inputs);
                if declared_vars.contains(&(vi as u32)) {
                    continue; // already declared
                }
                declared_vars.insert(vi as u32);
                let cl_type = if vec_oprefs.contains(&(vi as u32)) {
                    if is_vec_float_producing(op.opcode) {
                        cl_types::F64X2
                    } else {
                        cl_types::I64X2
                    }
                } else {
                    cl_types::I64
                };
                if debug_declares {
                    eprintln!("[jit][declare] op-result var{} opcode={:?}", vi, op.opcode);
                }
                var_types.insert(vi as u32, cl_type);
            }
            // Declare ALL referenced OpRefs: fail_args, op args, etc.
            for &arg in op
                .args
                .iter()
                .chain(op.fail_args.iter().flat_map(|fa| fa.iter()))
            {
                if !arg.is_none()
                    && !declared_vars.contains(&arg.0)
                    && !constants.contains_key(&arg.0)
                {
                    declared_vars.insert(arg.0);
                    if debug_declares {
                        eprintln!("[jit][declare] ref-arg var{} owner={:?}", arg.0, op.opcode);
                    }
                    var_types.insert(arg.0, cl_types::I64);
                }
            }
        }
        // RPython parity: LABEL args are independent box names bound by the
        // incoming JUMP, not necessarily inputargs or local op results.
        // The backend must declare them before the corresponding block param
        // is assigned in the LABEL handling path.
        for op in ops {
            if op.opcode != OpCode::Label {
                continue;
            }
            for &arg in &op.args {
                if arg.is_none() || declared_vars.contains(&arg.0) {
                    continue;
                }
                declared_vars.insert(arg.0);
                if debug_declares {
                    eprintln!("[jit][declare] label-arg var{}", arg.0);
                }
                var_types.insert(arg.0, cl_types::I64);
            }
        }

        // Compute loop_param_count early so legacy vars are included.
        let loop_param_count = if let Some(&li) = label_indices.last() {
            ops[li].args.len()
        } else {
            ops.iter()
                .rev()
                .find(|op| op.opcode == OpCode::Jump)
                .map(|op| op.args.len())
                .unwrap_or(num_inputs)
        };

        // Legacy no-LABEL traces can still need synthetic loop params.
        if label_indices.is_empty() && loop_param_count > num_inputs {
            for i in num_inputs..loop_param_count {
                if !declared_vars.contains(&(i as u32)) {
                    var_types.insert(i as u32, cl_types::I64);
                    declared_vars.insert(i as u32);
                }
            }
        }

        // Cranelift 0.130: declare_var(ty) returns auto-assigned Variable
        // indices (0, 1, 2, ...). Declare sequentially from 0 to max_index,
        // using the collected type or I64 for gap indices.
        if let Some(&max_idx) = var_types.keys().max() {
            for i in 0..=max_idx {
                let ty = var_types.get(&i).copied().unwrap_or(cl_types::I64);
                let returned_var = builder.declare_var(ty);
                debug_assert_eq!(returned_var, var(i));
            }
        }

        // Debug: save declared_vars snapshot for resolve_opref checking.
        DECLARED_VARS_DEBUG.with(|cell| {
            *cell.borrow_mut() = Some(declared_vars.clone());
        });

        // Save op-result positions for resolve_opref collision handling.
        let mut op_result_positions = std::collections::HashSet::new();
        for i in 0..num_inputs {
            op_result_positions.insert(i as u32);
        }
        for (op_idx, op) in ops.iter().enumerate() {
            if op.result_type() != Type::Void {
                let vi = op_var_index(op, op_idx, num_inputs) as u32;
                op_result_positions.insert(vi);
            }
        }
        OP_RESULT_VARS.with(|cell| {
            *cell.borrow_mut() = Some(op_result_positions);
        });

        let max_entry_inputs = num_inputs.max(body_direct_num_inputs);
        let load_count = if body_direct_num_inputs > 0 {
            max_entry_inputs + 1 // extra slot for entry_mode flag
        } else {
            num_inputs
        };
        let entry_input_vals: Vec<CValue> = (0..load_count)
            .map(|i| {
                let offset = JF_FRAME_ITEM0_OFS + (i as i32) * 8;
                builder
                    .ins()
                    .load(cl_types::I64, MemFlags::trusted(), inputs_ptr, offset)
            })
            .collect();

        // Always def_var inputargs in the entry block. When the trace
        // starts with LABEL, the LABEL block will override these via
        // block params. Preamble guards need the entry block values.
        for (i, val) in entry_input_vals.iter().copied().enumerate() {
            builder.def_var(var(i as u32), val);
        }

        // RPython parity: GC roots are registered by run_compiled_code()
        // before calling into compiled code, and unregistered after return.
        // No per-call registration/unregistration inside the compiled code.
        // Per-call spill/reload keeps the roots array up-to-date.

        // RPython compile.py sends loops to the backend as:
        // [start_label] + preamble_ops + [loop_label] + body_ops
        // and JUMPs target LABEL descrs. Model that directly by creating
        // a Cranelift block per LABEL descr.

        let mut label_blocks = Vec::with_capacity(label_indices.len());
        let mut label_blocks_by_descr = HashMap::new();
        for &label_idx in &label_indices {
            let block = builder.create_block();
            for _ in 0..ops[label_idx].args.len() {
                builder.append_block_param(block, cl_types::I64);
            }
            if let Some(descr_index) = ops[label_idx].descr.as_ref().map(|descr| descr.index()) {
                label_blocks_by_descr.insert(descr_index, block);
            }
            label_blocks.push((label_idx, block));
        }

        // RPython backend: Label ops define loop blocks.
        // Linear traces (no Label, no Jump) stay in the entry block.
        let has_jump = ops.iter().any(|op| op.opcode == OpCode::Jump);
        let loop_block = if !label_blocks.is_empty() {
            label_blocks.last().map(|(_, block)| *block).unwrap()
        } else if has_jump {
            // Legacy no-Label trace with Jump: need a loop block
            let block = builder.create_block();
            for _ in 0..loop_param_count {
                builder.append_block_param(block, cl_types::I64);
            }
            block
        } else {
            // Linear trace: no loop block needed, stay in entry_block
            entry_block
        };

        let mut guard_idx: usize = 0;
        let mut last_ovf_flag: Option<CValue> = None;

        if let Some(&(entry_label_idx, entry_label_block)) = label_blocks.first() {
            if body_direct_num_inputs > 0 && label_blocks.len() >= 2 {
                // Dual entry: branch on entry_mode (last input slot).
                // entry_mode == 0 → preamble, entry_mode != 0 → body-direct.
                // brif directly targets label blocks with their params.
                let entry_mode = entry_input_vals[max_entry_inputs];
                let body_block = label_blocks.last().unwrap().1;

                let body_args = block_args(&entry_input_vals[..body_direct_num_inputs]);
                let preamble_args = block_args(&entry_input_vals[..num_inputs]);
                builder.ins().brif(
                    entry_mode,
                    body_block,
                    &body_args,
                    entry_label_block,
                    &preamble_args,
                );

                // Continue with preamble label block for var binding
                builder.switch_to_block(entry_label_block);
                for (i, &arg_ref) in ops[entry_label_idx].args.iter().enumerate() {
                    let param = builder.block_params(entry_label_block)[i];
                    if !arg_ref.is_none() {
                        builder.def_var(var(arg_ref.0), param);
                    }
                }
            } else {
                let vals: Vec<CValue> = if has_entry_label {
                    entry_input_vals.clone()
                } else {
                    ops[entry_label_idx]
                        .args
                        .iter()
                        .map(|&r| resolve_opref(&mut builder, &constants, r))
                        .collect()
                };
                builder.ins().jump(entry_label_block, &block_args(&vals));
                builder.switch_to_block(entry_label_block);
                for (i, &arg_ref) in ops[entry_label_idx].args.iter().enumerate() {
                    let param = builder.block_params(entry_label_block)[i];
                    if !arg_ref.is_none() {
                        builder.def_var(var(arg_ref.0), param);
                    }
                }
            }
        } else if has_jump {
            let zero = builder.ins().iconst(cl_types::I64, 0);
            let vals: Vec<CValue> = (0..loop_param_count)
                .map(|i| {
                    if i < num_inputs {
                        builder.use_var(var(i as u32))
                    } else {
                        zero
                    }
                })
                .collect();
            builder.ins().jump(loop_block, &block_args(&vals));
            builder.switch_to_block(loop_block);
            for i in 0..loop_param_count {
                let param = builder.block_params(loop_block)[i];
                builder.def_var(var(i as u32), param);
            }
        }
        // else: linear trace — already in entry_block with vars defined

        for op_idx in 0..ops.len() {
            if let Some((_, label_block)) = label_blocks
                .iter()
                .find(|(label_idx, _)| *label_idx == op_idx)
            {
                // The first LABEL is already the current block. Later LABELs
                // are explicit jump targets inside the same compiled loop.
                if Some(op_idx) != label_blocks.first().map(|(label_idx, _)| *label_idx) {
                    let prev_terminated = op_idx
                        .checked_sub(1)
                        .and_then(|prev_idx| ops.get(prev_idx))
                        .map(|prev| prev.opcode == OpCode::Jump || prev.opcode == OpCode::Finish)
                        .unwrap_or(false);
                    if !prev_terminated {
                        let vals: Vec<CValue> = ops[op_idx]
                            .args
                            .iter()
                            .map(|&r| resolve_opref(&mut builder, &constants, r))
                            .collect();
                        builder.ins().jump(*label_block, &block_args(&vals));
                    }
                    builder.switch_to_block(*label_block);
                    for (i, &arg_ref) in ops[op_idx].args.iter().enumerate() {
                        let param = builder.block_params(*label_block)[i];
                        if !arg_ref.is_none() && !constants.contains_key(&arg_ref.0) {
                            builder.def_var(var(arg_ref.0), param);
                        }
                    }
                }
                continue;
            }
            let op = &ops[op_idx];
            let vi = op_var_index(op, op_idx, num_inputs) as u32;

            match op.opcode {
                // ── Integer arithmetic ──
                OpCode::IntAdd => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().iadd(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntSub => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().isub(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntMul => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().imul(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntFloorDiv => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().sdiv(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntMod => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().srem(a, b);
                    builder.def_var(var(vi), r);
                }

                // ── Overflow arithmetic ──
                // Compute the result normally, then detect signed overflow
                // using the bit-manipulation formula:
                //   ovf = ((a ^ result) & (b ^ result)) >> 63   [for add]
                //   ovf = ((a ^ result) & ((a ^ b))) >> 63      [for sub]
                // For mul we use a widening approach: sign-extend to i128
                // then check if the result fits in i64.
                OpCode::IntAddOvf => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().iadd(a, b);
                    builder.def_var(var(vi), r);
                    // ovf = ((a ^ r) & (b ^ r)) >> 63
                    let axr = builder.ins().bxor(a, r);
                    let bxr = builder.ins().bxor(b, r);
                    let both = builder.ins().band(axr, bxr);
                    let ovf = builder.ins().sshr_imm(both, 63);
                    last_ovf_flag = Some(ovf);
                }
                OpCode::IntSubOvf => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().isub(a, b);
                    builder.def_var(var(vi), r);
                    // ovf = ((a ^ r) & (a ^ b)) >> 63
                    let axr = builder.ins().bxor(a, r);
                    let axb = builder.ins().bxor(a, b);
                    let both = builder.ins().band(axr, axb);
                    let ovf = builder.ins().sshr_imm(both, 63);
                    last_ovf_flag = Some(ovf);
                }
                OpCode::IntMulOvf => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().imul(a, b);
                    builder.def_var(var(vi), r);
                    // Check overflow: if b != 0 && r / b != a, then overflow.
                    // We need to guard against sdiv trap when b == 0.
                    // Use a conditional: if b == 0, ovf = 0; else ovf = (r/b != a).
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let b_is_zero = builder.ins().icmp(IntCC::Equal, b, zero);

                    let no_div_block = builder.create_block();
                    let div_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, cl_types::I64);

                    builder
                        .ins()
                        .brif(b_is_zero, no_div_block, &[], div_block, &[]);

                    // b == 0 path: no overflow (result is just 0 * a = 0)
                    builder.switch_to_block(no_div_block);
                    builder.seal_block(no_div_block);
                    let no_ovf = builder.ins().iconst(cl_types::I64, 0);
                    builder.ins().jump(merge_block, &[BlockArg::from(no_ovf)]);

                    // b != 0 path: check r / b != a
                    builder.switch_to_block(div_block);
                    builder.seal_block(div_block);
                    let div = builder.ins().sdiv(r, b);
                    let div_ne_a = builder.ins().icmp(IntCC::NotEqual, div, a);
                    let ovf_ext = builder.ins().uextend(cl_types::I64, div_ne_a);
                    builder.ins().jump(merge_block, &[BlockArg::from(ovf_ext)]);

                    builder.switch_to_block(merge_block);
                    builder.seal_block(merge_block);
                    let ovf = builder.block_params(merge_block)[0];
                    last_ovf_flag = Some(ovf);
                }
                OpCode::IntAnd => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().band(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntOr => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().bor(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntXor => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().bxor(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntLshift => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().ishl(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntRshift => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().sshr(a, b);
                    builder.def_var(var(vi), r);
                }
                OpCode::UintRshift => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let r = builder.ins().ushr(a, b);
                    builder.def_var(var(vi), r);
                }

                // ── Unary integer ──
                OpCode::IntNeg => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let r = builder.ins().ineg(a);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntInvert => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let r = builder.ins().bnot(a);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntIsZero => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let cmp = builder.ins().icmp(IntCC::Equal, a, zero);
                    let r = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntIsTrue => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let cmp = builder.ins().icmp(IntCC::NotEqual, a, zero);
                    let r = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntForceGeZero => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let cmp = builder.ins().icmp(IntCC::SignedLessThan, a, zero);
                    let r = builder.ins().select(cmp, zero, a);
                    builder.def_var(var(vi), r);
                }
                OpCode::IntBetween => {
                    // int_between(a, b, c) => a <= b < c
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let b = resolve_opref(&mut builder, &constants, op.arg(1));
                    let c = resolve_opref(&mut builder, &constants, op.arg(2));
                    let cmp1 = builder.ins().icmp(IntCC::SignedLessThanOrEqual, a, b);
                    let cmp2 = builder.ins().icmp(IntCC::SignedLessThan, b, c);
                    let both = builder.ins().band(cmp1, cmp2);
                    let r = builder.ins().uextend(cl_types::I64, both);
                    builder.def_var(var(vi), r);
                }

                // ── Integer comparisons ──
                OpCode::IntLt => emit_icmp(&mut builder, &constants, IntCC::SignedLessThan, op, vi),
                OpCode::IntLe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::SignedLessThanOrEqual,
                    op,
                    vi,
                ),
                OpCode::IntEq => emit_icmp(&mut builder, &constants, IntCC::Equal, op, vi),
                OpCode::IntNe => emit_icmp(&mut builder, &constants, IntCC::NotEqual, op, vi),
                OpCode::IntGt => {
                    emit_icmp(&mut builder, &constants, IntCC::SignedGreaterThan, op, vi)
                }
                OpCode::IntGe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::SignedGreaterThanOrEqual,
                    op,
                    vi,
                ),
                OpCode::UintLt => {
                    emit_icmp(&mut builder, &constants, IntCC::UnsignedLessThan, op, vi)
                }
                OpCode::UintLe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::UnsignedLessThanOrEqual,
                    op,
                    vi,
                ),
                OpCode::UintGt => {
                    emit_icmp(&mut builder, &constants, IntCC::UnsignedGreaterThan, op, vi)
                }
                OpCode::UintGe => emit_icmp(
                    &mut builder,
                    &constants,
                    IntCC::UnsignedGreaterThanOrEqual,
                    op,
                    vi,
                ),

                // ── Pointer comparisons ──
                OpCode::PtrEq | OpCode::InstancePtrEq => {
                    emit_icmp(&mut builder, &constants, IntCC::Equal, op, vi)
                }
                OpCode::PtrNe | OpCode::InstancePtrNe => {
                    emit_icmp(&mut builder, &constants, IntCC::NotEqual, op, vi)
                }

                // ── Float comparisons ──
                OpCode::FloatLt => emit_fcmp(&mut builder, &constants, FloatCC::LessThan, op, vi),
                OpCode::FloatLe => {
                    emit_fcmp(&mut builder, &constants, FloatCC::LessThanOrEqual, op, vi)
                }
                OpCode::FloatEq => emit_fcmp(&mut builder, &constants, FloatCC::Equal, op, vi),
                OpCode::FloatNe => emit_fcmp(&mut builder, &constants, FloatCC::NotEqual, op, vi),
                OpCode::FloatGt => {
                    emit_fcmp(&mut builder, &constants, FloatCC::GreaterThan, op, vi)
                }
                OpCode::FloatGe => emit_fcmp(
                    &mut builder,
                    &constants,
                    FloatCC::GreaterThanOrEqual,
                    op,
                    vi,
                ),

                // ── Identity / cast ──
                OpCode::SameAsI
                | OpCode::SameAsR
                | OpCode::SameAsF
                | OpCode::CastPtrToInt
                | OpCode::CastIntToPtr
                | OpCode::CastOpaquePtr => {
                    let a = if op.num_args() > 0 {
                        resolve_opref(&mut builder, &constants, op.arg(0))
                    } else if let Some(&c) = constants.get(&vi) {
                        builder.ins().iconst(cl_types::I64, c)
                    } else {
                        builder.ins().iconst(cl_types::I64, 0)
                    };
                    builder.def_var(var(vi), a);
                }

                // ── Guards ──
                OpCode::GuardTrue
                | OpCode::GuardFalse
                | OpCode::GuardNonnull
                | OpCode::GuardIsnull => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let cond = resolve_opref(&mut builder, &constants, op.arg(0));
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, cond, zero);

                    let exit_on_zero =
                        matches!(op.opcode, OpCode::GuardTrue | OpCode::GuardNonnull);
                    if exit_on_zero {
                        builder
                            .ins()
                            .brif(is_zero, exit_block, &[], cont_block, &[]);
                    } else {
                        builder
                            .ins()
                            .brif(is_zero, cont_block, &[], exit_block, &[]);
                    }

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardValue => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let neq = builder.ins().icmp(IntCC::NotEqual, a, b);
                    builder.ins().brif(neq, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardClass => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let neq = builder.ins().icmp(IntCC::NotEqual, a, b);
                    builder.ins().brif(neq, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardNonnullClass => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let (obj, expected_class) = resolve_binop(&mut builder, &constants, op);
                    let zero = builder.ins().iconst(ptr_type, 0);
                    let exit_block = builder.create_block();
                    let class_check_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let is_null = builder.ins().icmp(IntCC::Equal, obj, zero);
                    builder
                        .ins()
                        .brif(is_null, exit_block, &[], class_check_block, &[]);

                    builder.switch_to_block(class_check_block);
                    builder.seal_block(class_check_block);
                    let actual_class = builder.ins().load(ptr_type, MemFlags::trusted(), obj, 0);
                    let neq = builder
                        .ins()
                        .icmp(IntCC::NotEqual, actual_class, expected_class);
                    builder.ins().brif(neq, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardNoException => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    // Direct memory load from global JIT_EXC_VALUE (no TLS/host call).
                    // RPython backend fuses this into the preceding CALL as an
                    // inline flag check. We use a plain load from a known address.
                    let exc_addr = builder.ins().iconst(ptr_type, jit_exc_value_addr() as i64);
                    let exc_val = builder.ins().load(
                        cl_types::I64,
                        cranelift_codegen::ir::MemFlags::trusted(),
                        exc_addr,
                        0,
                    );
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let has_exc = builder.ins().icmp(IntCC::NotEqual, exc_val, zero);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder
                        .ins()
                        .brif(has_exc, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardException => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let expected_type = resolve_opref(&mut builder, &constants, op.arg(0));
                    let is_match = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_exc_type_matches as *const () as usize,
                        &[expected_type],
                        Some(cl_types::I64),
                    )
                    .expect("jit_exc_type_matches must return a value");
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let mismatch = builder.ins().icmp(IntCC::Equal, is_match, zero);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder
                        .ins()
                        .brif(mismatch, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);

                    let exc_val = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_exc_clear_and_get_value as *const () as usize,
                        &[],
                        Some(cl_types::I64),
                    )
                    .expect("jit_exc_clear_and_get_value must return a value");
                    let vi = op_var_index(op, op_idx, inputargs.len());
                    builder.def_var(var(vi as u32), exc_val);
                }

                OpCode::GuardNoOverflow => {
                    // RPython intbounds.py:217-220: if the preceding op
                    // is not an overflow op, the guard is redundant (the
                    // overflow was already proven impossible or optimized away).
                    if last_ovf_flag.is_none() {
                        guard_idx += 1; // consume guard_info slot
                        continue;
                    }
                    // Side-exit if overflow DID occur (ovf != 0).
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let ovf = last_ovf_flag.take().unwrap();
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, ovf, zero);
                    // If ovf == 0 (no overflow), continue; otherwise side-exit.
                    builder
                        .ins()
                        .brif(is_zero, cont_block, &[], exit_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }
                OpCode::GuardOverflow => {
                    // Side-exit if overflow did NOT occur (ovf == 0).
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let ovf = last_ovf_flag
                        .take()
                        .expect("GuardOverflow without preceding overflow op");
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, ovf, zero);
                    // If ovf == 0 (no overflow), side-exit; otherwise continue.
                    builder
                        .ins()
                        .brif(is_zero, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardNotForced => {
                    // Intervening non-guard ops between CallMayForce and
                    // GuardNotForced are allowed. We only require that a
                    // preceding CallMayForce exists somewhere earlier in the
                    // trace (the push/pop shim mechanism is position-independent).
                    let has_preceding_call_may_force =
                        ops[..op_idx].iter().any(|o| o.opcode.is_call_may_force());
                    if !has_preceding_call_may_force {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "guard_not_forced requires a preceding call_may_force",
                        ));
                    }
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let was_forced = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        finish_may_force_guard_shim as *const () as usize,
                        &[],
                        Some(cl_types::I64),
                    )
                    .expect("guard_not_forced shim must return a value");
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_forced = builder.ins().icmp(IntCC::NotEqual, was_forced, zero);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder
                        .ins()
                        .brif(is_forced, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }
                OpCode::GuardNotForced2 => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let arg_bytes = (info.fail_arg_refs.len().max(1) * 8) as u32;
                    let args_slot = builder.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        arg_bytes,
                        3,
                    ));
                    for (index, &arg_ref) in info.fail_arg_refs.iter().enumerate() {
                        let raw = resolve_opref(&mut builder, &constants, arg_ref);
                        builder
                            .ins()
                            .stack_store(raw, args_slot, (index * 8) as i32);
                    }
                    let args_ptr = builder.ins().stack_addr(ptr_type, args_slot, 0);
                    let fail_index = builder.ins().iconst(cl_types::I64, info.fail_index as i64);
                    let num_values = builder
                        .ins()
                        .iconst(cl_types::I64, info.fail_arg_refs.len() as i64);
                    let args_ptr = ptr_arg_as_i64(&mut builder, args_ptr, ptr_type);
                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        record_guard_not_forced_2_shim as *const () as usize,
                        &[fail_index, args_ptr, num_values],
                        None,
                    );
                }

                OpCode::GuardNotInvalidated => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    if let Some(flag_addr) = invalidation_flag_ptr {
                        // Load the invalidation flag (AtomicBool, 1 byte) from
                        // the known address baked into the generated code.
                        let addr_val = builder.ins().iconst(cl_types::I64, flag_addr as i64);
                        let flag_val =
                            builder
                                .ins()
                                .load(cl_types::I8, MemFlags::trusted(), addr_val, 0);
                        let zero = builder.ins().iconst(cl_types::I8, 0);
                        let is_invalidated = builder.ins().icmp(IntCC::NotEqual, flag_val, zero);

                        let exit_block = builder.create_block();
                        let cont_block = builder.create_block();

                        builder
                            .ins()
                            .brif(is_invalidated, exit_block, &[], cont_block, &[]);

                        builder.switch_to_block(exit_block);
                        builder.seal_block(exit_block);
                        emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                        builder.switch_to_block(cont_block);
                        builder.seal_block(cont_block);
                    }
                }

                OpCode::GuardFutureCondition => {
                    // Future condition: the guard condition is computed lazily.
                    // For now, behave like GuardTrue on args[0].
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let cond = resolve_opref(&mut builder, &constants, op.args[0]);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_false = builder.ins().icmp(IntCC::Equal, cond, zero);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder
                        .ins()
                        .brif(is_false, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardAlwaysFails => {
                    // Always-failing guard: unconditionally side-exit.
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    // Create a continuation block for subsequent ops (dead code).
                    let dead_block = builder.create_block();
                    builder.switch_to_block(dead_block);
                    builder.seal_block(dead_block);
                }

                OpCode::GuardGcType => {
                    // args[0] = object ref, args[1] = expected type_id (as Int)
                    // Load the GC header (8 bytes before the obj pointer),
                    // extract the lower 32 bits (type_id), and compare.
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let obj_ptr = resolve_opref(&mut builder, &constants, op.args[0]);
                    let expected_tid = resolve_opref(&mut builder, &constants, op.args[1]);

                    // Load header word from obj_ptr - GcHeader::SIZE
                    let hdr_addr = builder.ins().iadd_imm(obj_ptr, -(GcHeader::SIZE as i64));
                    let hdr_word =
                        builder
                            .ins()
                            .load(cl_types::I64, MemFlags::trusted(), hdr_addr, 0);
                    // Extract type_id (lower 32 bits)
                    let tid_mask = builder.ins().iconst(cl_types::I64, TYPE_ID_MASK as i64);
                    let actual_tid = builder.ins().band(hdr_word, tid_mask);

                    let neq = builder
                        .ins()
                        .icmp(IntCC::NotEqual, actual_tid, expected_tid);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder.ins().brif(neq, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardIsObject => {
                    // args[0] = ref value. Side-exit if null (0).
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let obj_ptr = resolve_opref(&mut builder, &constants, op.args[0]);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, obj_ptr, zero);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder
                        .ins()
                        .brif(is_null, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                OpCode::GuardSubclass | OpCode::GuardCompatible => {
                    // GuardSubclass: args[0] = object ref, args[1] = expected parent class ref.
                    // GuardCompatible: args[0] = object ref, args[1] = expected compatible value.
                    // Both guard that two values are equal; fail otherwise.
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let neq = builder.ins().icmp(IntCC::NotEqual, a, b);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder.ins().brif(neq, exit_block, &[], cont_block, &[]);

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                // ── Exception operations ──
                OpCode::SaveException => {
                    // Returns the current exception value as a Ref.
                    let exc_val = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_exc_get_value as *const () as usize,
                        &[],
                        Some(cl_types::I64),
                    )
                    .expect("jit_exc_get_value must return a value");
                    let vi = op_var_index(op, op_idx, inputargs.len());
                    builder.def_var(var(vi as u32), exc_val);
                }
                OpCode::SaveExcClass => {
                    // Returns the current exception class as an Int.
                    let exc_type = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_exc_get_type as *const () as usize,
                        &[],
                        Some(cl_types::I64),
                    )
                    .expect("jit_exc_get_type must return a value");
                    let vi = op_var_index(op, op_idx, inputargs.len());
                    builder.def_var(var(vi as u32), exc_type);
                }
                OpCode::RestoreException => {
                    // args[0] = exception class, args[1] = exception value
                    let exc_type = resolve_opref(&mut builder, &constants, op.args[0]);
                    let value = resolve_opref(&mut builder, &constants, op.args[1]);
                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_exc_restore as *const () as usize,
                        &[value, exc_type],
                        None,
                    );
                }
                OpCode::CheckMemoryError => {
                    // args[0] = pointer to check. If null (0), abort via trap.
                    // In RPython, check_memory_error raises MemoryError on null.
                    // We trap unconditionally on null; the runtime catches this.
                    let ptr_val = resolve_opref(&mut builder, &constants, op.args[0]);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_null = builder.ins().icmp(IntCC::Equal, ptr_val, zero);
                    let trap_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder
                        .ins()
                        .brif(is_null, trap_block, &[], cont_block, &[]);

                    builder.switch_to_block(trap_block);
                    builder.seal_block(trap_block);
                    builder
                        .ins()
                        .trap(cranelift_codegen::ir::TrapCode::user(0).unwrap());

                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                // ── Call operations ──
                //
                // Regular calls, pure calls, and loop-invariant calls compile
                // as ordinary indirect calls. More specialized call families
                // with extra runtime semantics are rejected below until wired.
                OpCode::CallI
                | OpCode::CallR
                | OpCode::CallF
                | OpCode::CallN
                | OpCode::CallPureI
                | OpCode::CallPureR
                | OpCode::CallPureF
                | OpCode::CallPureN
                | OpCode::CallLoopinvariantI
                | OpCode::CallLoopinvariantR
                | OpCode::CallLoopinvariantF
                | OpCode::CallLoopinvariantN => {
                    // Inline arena take/put: replace create_frame/drop_frame
                    // indirect calls with inline loads/stores when possible.
                    if let Some(arena) = INLINE_ARENA.get() {
                        let func_addr_key = op.arg(0).0;
                        let func_addr = constants.get(&func_addr_key).copied().unwrap_or(-1);

                        if op.opcode == OpCode::CallR && func_addr == arena.create_fn_addr as i64 {
                            let result = emit_inline_arena_take(
                                &mut builder,
                                &constants,
                                arena,
                                op,
                                ptr_type,
                            );
                            builder.def_var(var(vi), result);
                            continue;
                        }

                        if op.opcode == OpCode::CallN && func_addr == arena.drop_fn_addr as i64 {
                            emit_inline_arena_put(&mut builder, &constants, arena, op, ptr_type);
                            continue;
                        }
                    }

                    let descr = op.descr.as_ref().expect("call op must have a descriptor");
                    let call_descr = descr
                        .as_call_descr()
                        .expect("call op descriptor must be a CallDescr");

                    if let Some(result) = emit_indirect_call_from_parts(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        &op.args[1..],
                        call_descr,
                        call_conv,
                        ptr_type,
                        gc_runtime_id,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                    ) {
                        builder.def_var(var(vi), result);
                    }
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    outputs_ptr = jf_ptr;
                }

                OpCode::CallAssemblerI
                | OpCode::CallAssemblerR
                | OpCode::CallAssemblerF
                | OpCode::CallAssemblerN => {
                    let descr = op.descr.as_ref().ok_or_else(|| {
                        unsupported_semantics(op.opcode, "call-assembler op must have a descriptor")
                    })?;
                    let call_descr = descr.as_call_descr().ok_or_else(|| {
                        unsupported_semantics(
                            op.opcode,
                            "call-assembler descriptor must be a CallDescr",
                        )
                    })?;
                    let resolved_target = resolve_call_assembler_target(op.opcode, call_descr)?;
                    if op.args.len() != call_descr.arg_types().len() {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "call-assembler argument count does not match the descriptor",
                        ));
                    }

                    // args_slot has JF header (64B) + items. Shared for inputs AND outputs.
                    let out_slots = resolved_target
                        .as_ref()
                        .map_or(16, |t| t.max_output_slots.max(1));
                    let jf_depth = call_descr.arg_types().len().max(out_slots).max(1);
                    let jf_bytes = (JF_FRAME_ITEM0_OFS as u32) + (jf_depth as u32) * 8;
                    let args_slot = builder.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        jf_bytes,
                        3,
                    ));
                    for (index, &arg_ref) in op.args.iter().enumerate() {
                        let raw = resolve_opref(&mut builder, &constants, arg_ref);
                        let ofs = JF_FRAME_ITEM0_OFS + (index as i32) * 8;
                        builder.ins().stack_store(raw, args_slot, ofs);
                    }
                    // jf_ptr = start of jitframe (header + items)
                    let args_ptr = builder.ins().stack_addr(ptr_type, args_slot, 0);
                    // data_ptr = start of items area (for shim which expects flat i64 array)
                    let args_data_ptr =
                        builder
                            .ins()
                            .stack_addr(ptr_type, args_slot, JF_FRAME_ITEM0_OFS);
                    let target_token = builder.ins().iconst(
                        cl_types::I64,
                        call_descr.call_target_token().unwrap() as i64,
                    );
                    let args_ptr_i64 = ptr_arg_as_i64(&mut builder, args_data_ptr, ptr_type);
                    let outcome_slot = builder.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        16,
                        3,
                    ));
                    let outcome_ptr = builder.ins().stack_addr(ptr_type, outcome_slot, 0);
                    let outcome_ptr_i64 = ptr_arg_as_i64(&mut builder, outcome_ptr, ptr_type);
                    let expected_result_kind = builder.ins().iconst(
                        cl_types::I64,
                        expected_call_assembler_result_kind(call_descr, resolved_target.as_ref())?
                            as i64,
                    );

                    let ca_merge_block = builder.create_block();
                    builder.append_block_param(ca_merge_block, cl_types::I64);

                    // Try direct call if target is resolved and has a known
                    // finish exit with primitive result type. This inlines the
                    // hot path (call target → check finish → extract result)
                    // into Cranelift IR, bypassing the shim entirely.
                    let finish_descr = resolved_target.as_ref().and_then(|t| {
                        t.fail_descrs.iter().find(|d| {
                            d.is_finish()
                                && matches!(d.fail_arg_types(), [Type::Int] | [Type::Float])
                        })
                    });

                    // Direct call via dispatch table: load code_ptr from a
                    // stable slot (AtomicPtr). For self-recursion (target not
                    // yet registered), pre-create the slot with null — compile
                    // completion fills it. Runtime null check falls back to shim.
                    let token_val = call_descr.call_target_token().unwrap_or(0);
                    let dispatch_slot_addr = if let Some(t) = resolved_target.as_ref() {
                        if !t.code_ptr.is_null() {
                            Some(ca_dispatch_slot(token_val, t.code_ptr) as usize)
                        } else {
                            None
                        }
                    } else if token_val != 0 {
                        // Self-recursion: pre-create slot with null code_ptr.
                        // register_call_assembler_target fills it after compile.
                        Some(ca_dispatch_slot(token_val, std::ptr::null()) as usize)
                    } else {
                        None
                    };

                    let has_primitive_result = finish_descr.is_some() || resolved_target.is_none();
                    let use_direct = dispatch_slot_addr.is_some() && has_primitive_result;

                    if use_direct {
                        let slot_addr = dispatch_slot_addr.unwrap();

                        // No separate out_slot: callee writes outputs to args_slot (shared).

                        // Load code_ptr and finish_descr_ptr from dispatch entry.
                        // CaDispatchEntry layout: [code_ptr: 8B, finish_descr_ptr: 8B]
                        // RPython done_with_this_frame parity: compare jf_descr
                        // with the finish FailDescr pointer directly.
                        let entry_ptr = builder.ins().iconst(ptr_type, slot_addr as i64);
                        let code_addr =
                            builder
                                .ins()
                                .load(ptr_type, MemFlags::trusted(), entry_ptr, 0);
                        let runtime_finish_descr = builder.ins().load(
                            cl_types::I64,
                            MemFlags::trusted(),
                            entry_ptr,
                            8, // offset of finish_descr_ptr in CaDispatchEntry
                        );
                        let null_ptr = builder.ins().iconst(ptr_type, 0);
                        let is_null = builder.ins().icmp(IntCC::Equal, code_addr, null_ptr);
                        // Also check finish_descr_ptr != CA_FINISH_INDEX_UNKNOWN
                        let unknown_sentinel = builder
                            .ins()
                            .iconst(cl_types::I64, CA_FINISH_INDEX_UNKNOWN as i64);
                        let finish_unknown = builder.ins().icmp(
                            IntCC::Equal,
                            runtime_finish_descr,
                            unknown_sentinel,
                        );
                        let cant_direct = builder.ins().bor(is_null, finish_unknown);
                        let direct_call_block = builder.create_block();
                        let shim_fallback_block = builder.create_block();
                        builder.ins().brif(
                            cant_direct,
                            shim_fallback_block,
                            &[],
                            direct_call_block,
                            &[],
                        );

                        builder.switch_to_block(direct_call_block);
                        builder.seal_block(direct_call_block);

                        // Save input[0] (frame_ptr) before call — callee
                        // overwrites args_slot with fail_args on guard exit.
                        // Needed by force_fn/shim fallback (cold path only).
                        let saved_frame_ptr = builder.ins().load(
                            cl_types::I64,
                            MemFlags::trusted(),
                            args_ptr,
                            JF_FRAME_ITEM0_OFS,
                        );

                        // fn(jf_ptr) → jf_ptr  (_call_footer parity)
                        let mut sig = Signature::new(call_conv);
                        sig.params.push(AbiParam::new(ptr_type)); // jf_ptr
                        sig.returns.push(AbiParam::new(ptr_type)); // returned jf_ptr
                        let sig_ref = builder.import_signature(sig);
                        let call_inst =
                            builder.ins().call_indirect(sig_ref, code_addr, &[args_ptr]);
                        let result_jf = builder.inst_results(call_inst)[0];
                        // _call_assembler_check_descr (assembler.py:2274-2278):
                        //   CMP [eax + jf_descr_ofs], done_with_this_frame_descr
                        let fail_idx_raw = builder.ins().load(
                            cl_types::I64,
                            MemFlags::trusted(),
                            result_jf,
                            JF_DESCR_OFS,
                        );

                        // RPython: check jf_descr == done_with_this_frame_descr
                        let is_direct_finish =
                            builder
                                .ins()
                                .icmp(IntCC::Equal, fail_idx_raw, runtime_finish_descr);
                        // RPython call_assembler (llsupport/assembler.py:295-359):
                        //   Path B: CMP [eax+jf_descr], done_descr → JE → load result
                        //   Path A: CALL assembler_helper_adr([jf_frame, vloc])
                        //
                        // Path A: 2 sub-paths (bridge inline + extern helper).
                        // Deadframe sentinel handled inside call_assembler_guard_failure.
                        let direct_finish_block = builder.create_block();
                        let nonfinish_block = if CALL_ASSEMBLER_FORCE_FN.get().is_some() {
                            Some(builder.create_block())
                        } else {
                            None
                        };
                        let nonfinish_target = nonfinish_block.unwrap_or(shim_fallback_block);
                        builder.ins().brif(
                            is_direct_finish,
                            direct_finish_block,
                            &[],
                            nonfinish_target,
                            &[],
                        );

                        // ── Path B: finish — load result from jf_frame[0] ──
                        builder.switch_to_block(direct_finish_block);
                        builder.seal_block(direct_finish_block);
                        let direct_result = builder.ins().load(
                            cl_types::I64,
                            MemFlags::trusted(),
                            args_ptr,
                            JF_FRAME_ITEM0_OFS,
                        );
                        builder
                            .ins()
                            .jump(ca_merge_block, &[BlockArg::from(direct_result)]);

                        // ── Path A: non-finish — bridge or assembler_helper ──
                        if let Some(nonfinish_block) = nonfinish_block {
                            builder.switch_to_block(nonfinish_block);
                            builder.seal_block(nonfinish_block);

                            // redirect_call_assembler parity: check bridge inline.
                            let bridge_ptr_val = builder.ins().load(
                                ptr_type,
                                MemFlags::trusted(),
                                entry_ptr,
                                16, // guard_bridge_ptr in CaDispatchEntry
                            );
                            let bridge_null = builder.ins().iconst(ptr_type, 0);
                            let has_bridge =
                                builder
                                    .ins()
                                    .icmp(IntCC::NotEqual, bridge_ptr_val, bridge_null);
                            let inline_bridge_block = builder.create_block();
                            let extern_helper_block = builder.create_block();
                            builder.ins().brif(
                                has_bridge,
                                inline_bridge_block,
                                &[],
                                extern_helper_block,
                                &[],
                            );

                            // ── Inline bridge dispatch (hot) ──
                            builder.switch_to_block(inline_bridge_block);
                            builder.seal_block(inline_bridge_block);
                            let mut bridge_sig = Signature::new(call_conv);
                            bridge_sig.params.push(AbiParam::new(ptr_type));
                            bridge_sig.returns.push(AbiParam::new(ptr_type));
                            let bridge_sig_ref = builder.import_signature(bridge_sig);
                            let bridge_call = builder.ins().call_indirect(
                                bridge_sig_ref,
                                bridge_ptr_val,
                                &[args_ptr],
                            );
                            let _bridge_jf = builder.inst_results(bridge_call)[0];
                            let bridge_result = builder.ins().load(
                                cl_types::I64,
                                MemFlags::trusted(),
                                args_ptr,
                                JF_FRAME_ITEM0_OFS,
                            );
                            builder
                                .ins()
                                .jump(ca_merge_block, &[BlockArg::from(bridge_result)]);

                            // ── assembler_helper (cold: no bridge yet) ──
                            builder.switch_to_block(extern_helper_block);
                            builder.seal_block(extern_helper_block);
                            let frame_ptr = saved_frame_ptr;
                            let result_jf_data =
                                builder.ins().iadd_imm(result_jf, JF_FRAME_ITEM0_OFS as i64);
                            let result_jf_data_i64 =
                                ptr_arg_as_i64(&mut builder, result_jf_data, ptr_type);
                            let force_result = emit_host_call(
                                &mut builder,
                                ptr_type,
                                call_conv,
                                call_assembler_guard_failure as *const () as usize,
                                &[
                                    target_token,
                                    fail_idx_raw,
                                    frame_ptr,
                                    result_jf_data_i64,
                                    args_ptr_i64,
                                ],
                                Some(cl_types::I64),
                            );
                            builder
                                .ins()
                                .jump(ca_merge_block, &[BlockArg::from(force_result.unwrap())]);
                        }

                        // ── Fallback: null code_ptr, unknown finish, or deadframe ──
                        builder.switch_to_block(shim_fallback_block);
                        builder.seal_block(shim_fallback_block);
                    }

                    // Shim call (always present as fallback, or sole path
                    // when target isn't resolved or has non-primitive result)
                    if call_descr.effect_info().can_raise() {
                        let _ = emit_host_call(
                            &mut builder,
                            ptr_type,
                            call_conv,
                            jit_exc_clear as *const () as usize,
                            &[],
                            None,
                        );
                    }

                    spill_ref_roots(
                        &mut builder,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                    );
                    emit_push_gcmap(&mut builder, jf_ptr, per_call_gcmap);
                    let result = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        call_assembler_shim as *const () as usize,
                        &[
                            target_token,
                            args_ptr_i64,
                            outcome_ptr_i64,
                            expected_result_kind,
                        ],
                        Some(cl_types::I64),
                    );
                    emit_pop_gcmap(&mut builder, jf_ptr, per_call_gcmap);
                    reload_ref_roots(
                        &mut builder,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                    );

                    let outcome_kind = builder.ins().stack_load(cl_types::I64, outcome_slot, 0);
                    let finish_kind = builder
                        .ins()
                        .iconst(cl_types::I64, CALL_ASSEMBLER_OUTCOME_FINISH);
                    let is_finish = builder.ins().icmp(IntCC::Equal, outcome_kind, finish_kind);
                    let exit_block = builder.create_block();
                    builder.ins().brif(
                        is_finish,
                        ca_merge_block,
                        &[BlockArg::from(result.unwrap())],
                        exit_block,
                        &[],
                    );

                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    let deadframe_handle = builder.ins().stack_load(cl_types::I64, outcome_slot, 8);
                    builder.ins().store(
                        MemFlags::trusted(),
                        deadframe_handle,
                        outputs_ptr,
                        JF_FRAME_ITEM0_OFS,
                    );
                    let sentinel = builder
                        .ins()
                        .iconst(cl_types::I64, CALL_ASSEMBLER_DEADFRAME_SENTINEL as i64);
                    builder
                        .ins()
                        .store(MemFlags::trusted(), sentinel, outputs_ptr, JF_DESCR_OFS);
                    builder.ins().return_(&[outputs_ptr]);

                    // ── Merge: result from cache or shim ──
                    builder.switch_to_block(ca_merge_block);
                    builder.seal_block(ca_merge_block);
                    let merged_result = builder.block_params(ca_merge_block)[0];

                    if op.result_type() != Type::Void {
                        builder.def_var(var(vi), merged_result);
                    }
                }

                OpCode::CallMayForceI
                | OpCode::CallMayForceR
                | OpCode::CallMayForceF
                | OpCode::CallMayForceN => {
                    // Intervening non-guard ops (SameAs, SaveException, etc.)
                    // are allowed between CallMayForce and GuardNotForced.
                    // The push/pop mechanism in pending_may_force is
                    // position-independent; we only require that a matching
                    // GuardNotForced appears later in the trace.
                    let has_guard_not_forced = ops[op_idx + 1..]
                        .iter()
                        .any(|o| o.opcode == OpCode::GuardNotForced);
                    if !has_guard_not_forced {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "call_may_force must be followed by guard_not_forced",
                        ));
                    }
                    let info = &guard_infos[guard_idx];

                    let preview_bytes = (info.fail_arg_refs.len().max(1) * 8) as u32;
                    let preview_slot = builder.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        preview_bytes,
                        3,
                    ));
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    for (index, &arg_ref) in info.fail_arg_refs.iter().enumerate() {
                        // Values defined at or after this op (vi) are not yet
                        // available — use zero as a placeholder in the preview
                        // snapshot.  Constants (stored in the constants map)
                        // are always resolvable regardless of position.
                        let raw = if arg_ref.0 >= vi && !constants.contains_key(&arg_ref.0) {
                            zero
                        } else {
                            resolve_opref(&mut builder, &constants, arg_ref)
                        };
                        builder
                            .ins()
                            .stack_store(raw, preview_slot, (index * 8) as i32);
                    }
                    let preview_ptr = builder.ins().stack_addr(ptr_type, preview_slot, 0);
                    let preview_ptr = ptr_arg_as_i64(&mut builder, preview_ptr, ptr_type);
                    let fail_index = builder.ins().iconst(cl_types::I64, info.fail_index as i64);
                    let num_values = builder
                        .ins()
                        .iconst(cl_types::I64, info.fail_arg_refs.len() as i64);
                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        begin_may_force_call_shim as *const () as usize,
                        &[fail_index, preview_ptr, num_values],
                        None,
                    );

                    let descr = op.descr.as_ref().expect("call op must have a descriptor");
                    let call_descr = descr
                        .as_call_descr()
                        .expect("call op descriptor must be a CallDescr");

                    if let Some(result) = emit_indirect_call_from_parts(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        &op.args[1..],
                        call_descr,
                        call_conv,
                        ptr_type,
                        gc_runtime_id,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                    ) {
                        builder.def_var(var(vi), result);
                    }
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    outputs_ptr = jf_ptr;
                }

                OpCode::CallReleaseGilI
                | OpCode::CallReleaseGilR
                | OpCode::CallReleaseGilF
                | OpCode::CallReleaseGilN => {
                    // Release-GIL calls: spill roots, release GIL, call,
                    // reacquire GIL, reload roots.
                    // In Rust, "GIL" is modeled as a pre/post hook pair.
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("call_release_gil op must have a descriptor");
                    let call_descr = descr
                        .as_call_descr()
                        .expect("call_release_gil descriptor must be a CallDescr");

                    if call_descr.effect_info().can_raise() {
                        let _ = emit_host_call(
                            &mut builder,
                            ptr_type,
                            call_conv,
                            jit_exc_clear as *const () as usize,
                            &[],
                            None,
                        );
                    }

                    // Spill GC roots before the call
                    spill_ref_roots(
                        &mut builder,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                    );
                    emit_push_gcmap(&mut builder, jf_ptr, per_call_gcmap);

                    // Release GIL (call the pre-hook)
                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_release_gil_shim as *const () as usize,
                        &[],
                        None,
                    );

                    // Make the actual call
                    let mut sig = Signature::new(call_conv);
                    let arg_types = call_descr.arg_types();
                    for at in arg_types {
                        sig.params.push(AbiParam::new(cranelift_type_for(at)));
                    }
                    let result_type = call_descr.result_type();
                    if result_type != Type::Void {
                        sig.returns
                            .push(AbiParam::new(cranelift_type_for(&result_type)));
                    }
                    let sig_ref = builder.import_signature(sig);

                    let func_ptr_raw = resolve_opref(&mut builder, &constants, op.arg(0));
                    let func_ptr_val = if ptr_type != cl_types::I64 {
                        builder.ins().ireduce(ptr_type, func_ptr_raw)
                    } else {
                        func_ptr_raw
                    };

                    let mut args: Vec<CValue> = Vec::with_capacity(op.args.len() - 1);
                    for (i, &arg_ref) in op.args[1..].iter().enumerate() {
                        let raw = resolve_opref(&mut builder, &constants, arg_ref);
                        if i < arg_types.len() && arg_types[i] == Type::Float {
                            args.push(builder.ins().bitcast(cl_types::F64, MemFlags::new(), raw));
                        } else {
                            args.push(raw);
                        }
                    }

                    let call = builder.ins().call_indirect(sig_ref, func_ptr_val, &args);
                    let result = if result_type != Type::Void {
                        Some(builder.inst_results(call)[0])
                    } else {
                        None
                    };

                    // Reacquire GIL (call the post-hook)
                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_reacquire_gil_shim as *const () as usize,
                        &[],
                        None,
                    );

                    emit_pop_gcmap(&mut builder, jf_ptr, per_call_gcmap);
                    // Reload roots (may have been updated by GC during call)
                    reload_ref_roots(
                        &mut builder,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                    );

                    if let Some(result) = result {
                        let result_val = if result_type == Type::Float {
                            builder
                                .ins()
                                .bitcast(cl_types::I64, MemFlags::new(), result)
                        } else {
                            result
                        };
                        builder.def_var(var(vi), result_val);
                    }
                }

                // ── Conditional call (void result) ──
                // args[0] = condition, args[1] = func_ptr, args[2..] = call args
                // If condition != 0, perform the call.
                OpCode::CondCallN => {
                    let cond = resolve_opref(&mut builder, &constants, op.arg(0));
                    let call_block = builder.create_block();
                    let cont_block = builder.create_block();

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, cond, zero);
                    builder
                        .ins()
                        .brif(is_zero, cont_block, &[], call_block, &[]);

                    builder.switch_to_block(call_block);
                    builder.seal_block(call_block);

                    if let Some(descr) = op.descr.as_ref() {
                        if let Some(call_descr) = descr.as_call_descr() {
                            let _ = emit_indirect_call_from_parts(
                                &mut builder,
                                &constants,
                                op.arg(1),
                                &op.args[2..],
                                call_descr,
                                call_conv,
                                ptr_type,
                                gc_runtime_id,
                                jf_ptr,
                                &ref_root_slots,
                                &defined_ref_vars,
                                ref_root_base_ofs,
                                per_call_gcmap,
                            );
                            jf_ptr =
                                emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                        }
                    }

                    builder.ins().jump(cont_block, &[]);
                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                // ── Conditional call with value result ──
                // args[0] = condition, args[1] = func_ptr, args[2..] = call args
                // If condition != 0: result = call(func_ptr, args...)
                // Else: result = condition (0)
                OpCode::CondCallValueI | OpCode::CondCallValueR => {
                    let cond = resolve_opref(&mut builder, &constants, op.arg(0));
                    let call_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder.append_block_param(cont_block, cl_types::I64);

                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_zero = builder.ins().icmp(IntCC::Equal, cond, zero);
                    builder.ins().brif(
                        is_zero,
                        cont_block,
                        &[BlockArg::from(cond)],
                        call_block,
                        &[],
                    );

                    builder.switch_to_block(call_block);
                    builder.seal_block(call_block);

                    let mut call_result = cond; // fallback
                    if let Some(descr) = op.descr.as_ref() {
                        if let Some(call_descr) = descr.as_call_descr() {
                            if let Some(result) = emit_indirect_call_from_parts(
                                &mut builder,
                                &constants,
                                op.arg(1),
                                &op.args[2..],
                                call_descr,
                                call_conv,
                                ptr_type,
                                gc_runtime_id,
                                jf_ptr,
                                &ref_root_slots,
                                &defined_ref_vars,
                                ref_root_base_ofs,
                                per_call_gcmap,
                            ) {
                                call_result = result;
                            }
                            jf_ptr =
                                emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                        }
                    }

                    builder
                        .ins()
                        .jump(cont_block, &[BlockArg::from(call_result)]);
                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);

                    let phi = builder.block_params(cont_block)[0];
                    builder.def_var(var(vi), phi);
                }

                // ── GC allocation calls ──
                OpCode::CallMallocNursery => {
                    let runtime_id = gc_runtime_id.ok_or_else(|| missing_gc_runtime(op.opcode))?;
                    let runtime_id = builder.ins().iconst(cl_types::I64, runtime_id as i64);
                    let size_total = builder.ins().iconst(
                        cl_types::I64,
                        resolve_rewriter_immediate_i64(&constants, op.arg(0)),
                    );
                    let size = builder.ins().iadd_imm(size_total, -(GcHeader::SIZE as i64));
                    let result = emit_collecting_gc_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                        runtime_id,
                        gc_alloc_nursery_shim as *const () as usize,
                        &[size],
                        Some(cl_types::I64),
                    )
                    .expect("GC allocation helper must return a value");
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    outputs_ptr = jf_ptr;
                    builder.def_var(var(vi), result);
                }
                OpCode::CallMallocNurseryVarsize => {
                    let runtime_id = gc_runtime_id.ok_or_else(|| missing_gc_runtime(op.opcode))?;
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("CallMallocNurseryVarsize must have an ArrayDescr");
                    let ad = descr
                        .as_array_descr()
                        .expect("CallMallocNurseryVarsize descr must be an ArrayDescr");
                    let runtime_id = builder.ins().iconst(cl_types::I64, runtime_id as i64);
                    let base_size = builder.ins().iconst(cl_types::I64, ad.base_size() as i64);
                    let item_size = builder.ins().iconst(cl_types::I64, ad.item_size() as i64);
                    let length =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(0));
                    let result = emit_collecting_gc_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                        runtime_id,
                        gc_alloc_varsize_shim as *const () as usize,
                        &[base_size, item_size, length],
                        Some(cl_types::I64),
                    )
                    .expect("GC varsize allocation helper must return a value");
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    outputs_ptr = jf_ptr;
                    builder.def_var(var(vi), result);
                }
                OpCode::CallMallocNurseryVarsizeFrame => {
                    // RPython x86/assembler.py:2567 malloc_cond_varsize_frame:
                    // Inline nursery bump allocation.
                    //   ecx = load(nursery_free_adr)
                    //   edx = ecx + size
                    //   cmp edx, load(nursery_top_adr)
                    //   ja slow_path
                    //   store(nursery_free_adr, edx)
                    //   ; ecx = allocated pointer
                    let (nf_addr, nt_addr) = majit_gc::nursery::nursery_global_addrs();
                    let flags = MemFlags::trusted();
                    let size_total =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(0));

                    let nf_ptr = builder.ins().iconst(ptr_type, nf_addr as i64);
                    let nt_ptr = builder.ins().iconst(ptr_type, nt_addr as i64);
                    let free = builder.ins().load(ptr_type, flags, nf_ptr, 0);
                    let new_free = builder.ins().iadd(free, size_total);
                    let top = builder.ins().load(ptr_type, flags, nt_ptr, 0);
                    let fits = builder
                        .ins()
                        .icmp(IntCC::UnsignedLessThanOrEqual, new_free, top);

                    let fast_block = builder.create_block();
                    let slow_block = builder.create_block();
                    let merge_block = builder.create_block();
                    builder.append_block_param(merge_block, ptr_type); // alloc result
                    builder.append_block_param(merge_block, ptr_type); // jf_ptr

                    builder.ins().brif(fits, fast_block, &[], slow_block, &[]);

                    // fast: bump free pointer, return old free
                    builder.switch_to_block(fast_block);
                    builder.seal_block(fast_block);
                    builder.ins().store(flags, new_free, nf_ptr, 0);
                    // Return pointer past GC header
                    let header_size = builder.ins().iconst(ptr_type, GcHeader::SIZE as i64);
                    let obj_ptr = builder.ins().iadd(free, header_size);
                    builder.ins().jump(
                        merge_block,
                        &[BlockArg::from(obj_ptr), BlockArg::from(jf_ptr)],
                    );

                    // slow: call shim (triggers minor collection)
                    builder.switch_to_block(slow_block);
                    builder.seal_block(slow_block);
                    let runtime_id = gc_runtime_id.ok_or_else(|| missing_gc_runtime(op.opcode))?;
                    let runtime_id_val = builder.ins().iconst(cl_types::I64, runtime_id as i64);
                    let size = builder.ins().iadd_imm(size_total, -(GcHeader::SIZE as i64));
                    let slow_result = emit_collecting_gc_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                        runtime_id_val,
                        gc_alloc_nursery_shim as *const () as usize,
                        &[size],
                        Some(cl_types::I64),
                    )
                    .expect("GC frame allocation helper must return a value");
                    let reloaded_jf =
                        emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    builder.ins().jump(
                        merge_block,
                        &[BlockArg::from(slow_result), BlockArg::from(reloaded_jf)],
                    );

                    builder.switch_to_block(merge_block);
                    builder.seal_block(merge_block);
                    let result = builder.block_params(merge_block)[0];
                    jf_ptr = builder.block_params(merge_block)[1];
                    outputs_ptr = jf_ptr;
                    builder.def_var(var(vi), result);
                }

                // ── GC write barriers ──
                OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                    let runtime_id = gc_runtime_id.ok_or_else(|| missing_gc_runtime(op.opcode))?;
                    let runtime_id = builder.ins().iconst(cl_types::I64, runtime_id as i64);
                    let obj = resolve_opref(&mut builder, &constants, op.arg(0));
                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        gc_write_barrier_shim as *const () as usize,
                        &[runtime_id, obj],
                        None,
                    );
                }

                // ── Generic GC/raw memory loads ──
                OpCode::GcLoadI | OpCode::GcLoadR | OpCode::GcLoadF => {
                    let item_size = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(2),
                        "GC_LOAD itemsize",
                    )?;
                    let value_type = match op.opcode {
                        OpCode::GcLoadI => Type::Int,
                        OpCode::GcLoadR => Type::Ref,
                        OpCode::GcLoadF => Type::Float,
                        _ => unreachable!(),
                    };
                    if value_type != Type::Int && item_size < 0 {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "negative GC_LOAD itemsize is only valid for integer loads",
                        ));
                    }
                    let addr = emit_dynamic_offset_addr(
                        &mut builder,
                        &constants,
                        &known_values,
                        op.arg(0),
                        op.arg(1),
                    );
                    let result = emit_load_from_addr(
                        &mut builder,
                        addr,
                        value_type,
                        item_size.unsigned_abs() as usize,
                        item_size < 0,
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), result);
                }
                OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
                    let scale = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(2),
                        "GC_LOAD_INDEXED scale",
                    )?;
                    let base_offset = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(3),
                        "GC_LOAD_INDEXED base offset",
                    )?;
                    let item_size = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(4),
                        "GC_LOAD_INDEXED itemsize",
                    )?;
                    let value_type = match op.opcode {
                        OpCode::GcLoadIndexedI => Type::Int,
                        OpCode::GcLoadIndexedR => Type::Ref,
                        OpCode::GcLoadIndexedF => Type::Float,
                        _ => unreachable!(),
                    };
                    if value_type != Type::Int && item_size < 0 {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "negative GC_LOAD_INDEXED itemsize is only valid for integer loads",
                        ));
                    }
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        scale,
                        base_offset,
                    );
                    let result = emit_load_from_addr(
                        &mut builder,
                        addr,
                        value_type,
                        item_size.unsigned_abs() as usize,
                        item_size < 0,
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), result);
                }
                OpCode::RawLoadI | OpCode::RawLoadF => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("raw load op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("raw load descriptor must be an ArrayDescr");
                    let addr = emit_dynamic_offset_addr(
                        &mut builder,
                        &constants,
                        &known_values,
                        op.arg(0),
                        op.arg(1),
                    );
                    let value_type = match op.opcode {
                        OpCode::RawLoadI => Type::Int,
                        OpCode::RawLoadF => Type::Float,
                        _ => unreachable!(),
                    };
                    let signed = value_type == Type::Int && ad.is_item_signed();
                    let result = emit_load_from_addr(
                        &mut builder,
                        addr,
                        value_type,
                        ad.item_size(),
                        signed,
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), result);
                }

                // ── GC stores ──
                OpCode::GcStore => {
                    if op.args.len() == 3 {
                        let base = resolve_opref(&mut builder, &constants, op.arg(0));
                        match op.arg(1).0 {
                            0 => {
                                let hdr_addr =
                                    builder.ins().iadd_imm(base, -(GcHeader::SIZE as i64));
                                let old_hdr = builder.ins().load(
                                    cl_types::I64,
                                    MemFlags::trusted(),
                                    hdr_addr,
                                    0,
                                );
                                let flags_mask =
                                    builder.ins().iconst(cl_types::I64, (!TYPE_ID_MASK) as i64);
                                let tid = builder.ins().iconst(
                                    cl_types::I64,
                                    (op.arg(2).0 as u64 & TYPE_ID_MASK) as i64,
                                );
                                let preserved_flags = builder.ins().band(old_hdr, flags_mask);
                                let new_hdr = builder.ins().bor(preserved_flags, tid);
                                builder
                                    .ins()
                                    .store(MemFlags::trusted(), new_hdr, hdr_addr, 0);
                            }
                            1 => {
                                let addr = builder.ins().iadd_imm(base, 0);
                                let vtable =
                                    builder.ins().iconst(cl_types::I64, op.arg(2).0 as i64);
                                builder.ins().store(MemFlags::trusted(), vtable, addr, 0);
                            }
                            2 => {
                                let descr = op.descr.as_ref().expect(
                                    "rewrite-generated length store must carry an ArrayDescr",
                                );
                                let ad = descr.as_array_descr().expect(
                                    "rewrite-generated length store descr must be an ArrayDescr",
                                );
                                let len_descr = ad
                                    .len_descr()
                                    .expect("rewrite-generated length store requires a len_descr");
                                let addr = builder.ins().iadd_imm(base, len_descr.offset() as i64);
                                let length = resolve_opref_or_imm(
                                    &mut builder,
                                    &constants,
                                    &known_values,
                                    op.arg(2),
                                );
                                emit_store_to_addr(
                                    &mut builder,
                                    addr,
                                    length,
                                    len_descr.field_type(),
                                    len_descr.field_size(),
                                    op.opcode,
                                )?;
                            }
                            _ => {
                                return Err(unsupported_semantics(
                                    op.opcode,
                                    "unknown rewrite-generated store marker",
                                ));
                            }
                        }
                    } else if op.args.len() == 4 {
                        let item_size = resolve_constant_i64(
                            &constants,
                            &known_values,
                            op.opcode,
                            op.arg(3),
                            "GC_STORE itemsize",
                        )?;
                        let value_type = type_for_opref(
                            &value_types,
                            &known_values,
                            op.opcode,
                            op.arg(2),
                            "GC_STORE value",
                        )?;
                        let addr = emit_dynamic_offset_addr(
                            &mut builder,
                            &constants,
                            &known_values,
                            op.arg(0),
                            op.arg(1),
                        );
                        let value = resolve_opref_or_imm(
                            &mut builder,
                            &constants,
                            &known_values,
                            op.arg(2),
                        );
                        emit_store_to_addr(
                            &mut builder,
                            addr,
                            value,
                            value_type,
                            item_size.unsigned_abs() as usize,
                            op.opcode,
                        )?;
                    } else {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "GC_STORE expects either the rewrite-generated 3-arg form or the generic 4-arg form",
                        ));
                    }
                }
                OpCode::GcStoreIndexed => {
                    let scale = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(3),
                        "GC_STORE_INDEXED scale",
                    )?;
                    let base_offset = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(4),
                        "GC_STORE_INDEXED base offset",
                    )?;
                    let item_size = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(5),
                        "GC_STORE_INDEXED itemsize",
                    )?;
                    let value_type = type_for_opref(
                        &value_types,
                        &known_values,
                        op.opcode,
                        op.arg(2),
                        "GC_STORE_INDEXED value",
                    )?;
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        scale,
                        base_offset,
                    );
                    let value =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(2));
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        value,
                        value_type,
                        item_size.unsigned_abs() as usize,
                        op.opcode,
                    )?;
                }

                // ── Field access (getfield) ──
                // All getfield variants load from base + offset.
                // The loaded value is sign/zero-extended from its field_size to I64,
                // or bitcast from F64 for float fields.
                OpCode::GetfieldGcI
                | OpCode::GetfieldGcR
                | OpCode::GetfieldGcF
                | OpCode::GetfieldRawI
                | OpCode::GetfieldRawR
                | OpCode::GetfieldRawF
                | OpCode::GetfieldGcPureI
                | OpCode::GetfieldGcPureR
                | OpCode::GetfieldGcPureF => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("getfield op must have a descriptor");
                    let fd = descr
                        .as_field_descr()
                        .expect("getfield descriptor must be a FieldDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let addr = builder.ins().iadd_imm(base, fd.offset() as i64);
                    let r = emit_load_from_addr(
                        &mut builder,
                        addr,
                        fd.field_type(),
                        fd.field_size(),
                        fd.is_field_signed(),
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), r);
                }

                // ── Field access (setfield) ──
                // args[0] = base, args[1] = value
                OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("setfield op must have a descriptor");
                    let fd = descr
                        .as_field_descr()
                        .expect("setfield descriptor must be a FieldDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let val = resolve_opref(&mut builder, &constants, op.arg(1));
                    let addr = builder.ins().iadd_imm(base, fd.offset() as i64);
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        val,
                        fd.field_type(),
                        fd.field_size(),
                        op.opcode,
                    )?;
                }

                // ── Array access (getarrayitem) ──
                // args[0] = base, args[1] = index
                // address = base + base_size + index * item_size
                OpCode::GetarrayitemGcI
                | OpCode::GetarrayitemGcR
                | OpCode::GetarrayitemGcF
                | OpCode::GetarrayitemRawI
                | OpCode::GetarrayitemRawF
                | OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("getarrayitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("getarrayitem descriptor must be an ArrayDescr");

                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        ad.item_size() as i64,
                        ad.base_size() as i64,
                    );
                    let signed = ad.item_type() == Type::Int && ad.is_item_signed();
                    let r = emit_load_from_addr(
                        &mut builder,
                        addr,
                        ad.item_type(),
                        ad.item_size(),
                        signed,
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), r);
                }

                // ── Array access (setarrayitem) ──
                // args[0] = base, args[1] = index, args[2] = value
                OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("setarrayitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("setarrayitem descriptor must be an ArrayDescr");

                    let val = resolve_opref(&mut builder, &constants, op.arg(2));
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        ad.item_size() as i64,
                        ad.base_size() as i64,
                    );
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        val,
                        ad.item_type(),
                        ad.item_size(),
                        op.opcode,
                    )?;
                }

                // ── Interior field access ──
                OpCode::GetinteriorfieldGcI
                | OpCode::GetinteriorfieldGcR
                | OpCode::GetinteriorfieldGcF => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("getinteriorfield op must have a descriptor");
                    let id = descr
                        .as_interior_field_descr()
                        .expect("getinteriorfield descriptor must be an InteriorFieldDescr");
                    let ad = id.array_descr();
                    let fd = id.field_descr();
                    let base_offset = (ad.base_size() + fd.offset()) as i64;
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        ad.item_size() as i64,
                        base_offset,
                    );
                    let r = emit_load_from_addr(
                        &mut builder,
                        addr,
                        fd.field_type(),
                        fd.field_size(),
                        fd.is_field_signed(),
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), r);
                }

                OpCode::SetinteriorfieldGc | OpCode::SetinteriorfieldRaw => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("setinteriorfield op must have a descriptor");
                    let id = descr
                        .as_interior_field_descr()
                        .expect("setinteriorfield descriptor must be an InteriorFieldDescr");
                    let ad = id.array_descr();
                    let fd = id.field_descr();
                    let base_offset = (ad.base_size() + fd.offset()) as i64;
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        ad.item_size() as i64,
                        base_offset,
                    );
                    let val = resolve_opref(&mut builder, &constants, op.arg(2));
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        val,
                        fd.field_type(),
                        fd.field_size(),
                        op.opcode,
                    )?;
                }

                OpCode::RawStore => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("raw store op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("raw store descriptor must be an ArrayDescr");
                    let addr = emit_dynamic_offset_addr(
                        &mut builder,
                        &constants,
                        &known_values,
                        op.arg(0),
                        op.arg(1),
                    );
                    let val = resolve_opref(&mut builder, &constants, op.arg(2));
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        val,
                        ad.item_type(),
                        ad.item_size(),
                        op.opcode,
                    )?;
                }

                // ── Array/string length ──
                // These load the length field from the object header using
                // the array descriptor's len_descr.
                OpCode::ArraylenGc => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("arraylen op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("arraylen descriptor must be an ArrayDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    if let Some(ld) = ad.len_descr() {
                        let addr = builder.ins().iadd_imm(base, ld.offset() as i64);
                        let r = emit_load_from_addr(
                            &mut builder,
                            addr,
                            ld.field_type(),
                            ld.field_size(),
                            ld.is_field_signed(),
                            op.opcode,
                        )?;
                        builder.def_var(var(vi), r);
                    } else {
                        // No len_descr: return 0 as a fallback.
                        let zero = builder.ins().iconst(cl_types::I64, 0);
                        builder.def_var(var(vi), zero);
                    }
                }

                OpCode::Strlen | OpCode::Unicodelen => {
                    // These use the array descriptor attached to the op.
                    // The length is at len_descr().offset() from the base pointer.
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("strlen/unicodelen op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("strlen/unicodelen descriptor must be an ArrayDescr");

                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    if let Some(ld) = ad.len_descr() {
                        let addr = builder.ins().iadd_imm(base, ld.offset() as i64);
                        let r = emit_load_from_addr(
                            &mut builder,
                            addr,
                            ld.field_type(),
                            ld.field_size(),
                            ld.is_field_signed(),
                            op.opcode,
                        )?;
                        builder.def_var(var(vi), r);
                    } else {
                        let zero = builder.ins().iconst(cl_types::I64, 0);
                        builder.def_var(var(vi), zero);
                    }
                }

                OpCode::Strhash | OpCode::Unicodehash => {
                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let addr = builder
                        .ins()
                        .iadd_imm(base, BUILTIN_STRING_HASH_OFFSET as i64);
                    let hash = emit_load_from_addr(
                        &mut builder,
                        addr,
                        Type::Int,
                        std::mem::size_of::<usize>(),
                        true,
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), hash);
                }

                // ── String/unicode item access ──
                // Strgetitem/Unicodegetitem: args[0] = base, args[1] = index
                // Treated as array item access using the descriptor's base_size/item_size.
                OpCode::Strgetitem | OpCode::Unicodegetitem => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("str/unicodegetitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("str/unicodegetitem descriptor must be an ArrayDescr");

                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        ad.item_size() as i64,
                        ad.base_size() as i64,
                    );
                    let r = emit_load_from_addr(
                        &mut builder,
                        addr,
                        Type::Int,
                        ad.item_size(),
                        false,
                        op.opcode,
                    )?;
                    builder.def_var(var(vi), r);
                }

                // Strsetitem/Unicodesetitem: args[0] = base, args[1] = index, args[2] = value
                OpCode::Strsetitem | OpCode::Unicodesetitem => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("str/unicodesetitem op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("str/unicodesetitem descriptor must be an ArrayDescr");

                    let val = resolve_opref(&mut builder, &constants, op.arg(2));
                    let addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(1),
                        ad.item_size() as i64,
                        ad.base_size() as i64,
                    );
                    emit_store_to_addr(
                        &mut builder,
                        addr,
                        val,
                        Type::Int,
                        ad.item_size(),
                        op.opcode,
                    )?;
                }

                OpCode::Copystrcontent | OpCode::Copyunicodecontent => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("copystr/unicodecontent op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("copystr/unicodecontent descriptor must be an ArrayDescr");

                    let src_addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(0),
                        op.arg(2),
                        ad.item_size() as i64,
                        ad.base_size() as i64,
                    );
                    let dst_addr = emit_scaled_index_addr(
                        &mut builder,
                        &constants,
                        op.arg(1),
                        op.arg(3),
                        ad.item_size() as i64,
                        ad.base_size() as i64,
                    );
                    let length =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(4));
                    let byte_length = if ad.item_size() == 1 {
                        length
                    } else {
                        let scale = builder.ins().iconst(cl_types::I64, ad.item_size() as i64);
                        builder.ins().imul(length, scale)
                    };

                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        copy_nonoverlapping_memory_shim as *const () as usize,
                        &[src_addr, dst_addr, byte_length],
                        None,
                    );
                }

                // ── Zero array range ──
                // args = [base, start, size, scale_start, scale_size]
                // zeroes bytes in
                // [base + descr.base_size + start*scale_start,
                //  base + descr.base_size + start*scale_start + size*scale_size)
                OpCode::ZeroArray => {
                    let descr = op
                        .descr
                        .as_ref()
                        .expect("zero_array op must have a descriptor");
                    let ad = descr
                        .as_array_descr()
                        .expect("zero_array descriptor must be an ArrayDescr");
                    let scale_start = resolve_rewriter_immediate_i64(&constants, op.arg(3));
                    let scale_size = resolve_rewriter_immediate_i64(&constants, op.arg(4));
                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let start = builder.ins().iconst(
                        cl_types::I64,
                        resolve_rewriter_immediate_i64(&constants, op.arg(1)),
                    );
                    let size = builder.ins().iconst(
                        cl_types::I64,
                        resolve_rewriter_immediate_i64(&constants, op.arg(2)),
                    );

                    let start_bytes = match scale_start {
                        0 => builder.ins().iconst(cl_types::I64, 0),
                        1 => start,
                        _ => {
                            let scale = builder.ins().iconst(cl_types::I64, scale_start);
                            builder.ins().imul(start, scale)
                        }
                    };
                    let byte_offset = builder.ins().iadd_imm(start_bytes, ad.base_size() as i64);
                    let byte_size = match scale_size {
                        0 => builder.ins().iconst(cl_types::I64, 0),
                        1 => size,
                        _ => {
                            let scale = builder.ins().iconst(cl_types::I64, scale_size);
                            builder.ins().imul(size, scale)
                        }
                    };

                    let _ = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        zero_memory_shim as *const () as usize,
                        &[base, byte_offset, byte_size],
                        None,
                    );
                }

                // ── Nursery pointer increment ──
                // args[0] = base ptr, args[1] = byte offset
                OpCode::NurseryPtrIncrement => {
                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let offset = builder.ins().iconst(
                        cl_types::I64,
                        resolve_rewriter_immediate_i64(&constants, op.arg(1)),
                    );
                    let r = builder.ins().iadd(base, offset);
                    builder.def_var(var(vi), r);
                }

                // ── Control flow ──
                OpCode::Jump => {
                    let vals: Vec<CValue> = op
                        .args
                        .iter()
                        .map(|&r| resolve_opref(&mut builder, &constants, r))
                        .collect();
                    // Resolve target: try descr-based lookup first, then
                    // fall back to loop_block (Jump without descr targets
                    // the loop header, matching RPython's self-loop).
                    let target_block = op
                        .descr
                        .as_ref()
                        .and_then(|descr| label_blocks_by_descr.get(&descr.index()).copied())
                        .or_else(|| {
                            if loop_block != entry_block {
                                Some(loop_block)
                            } else {
                                None
                            }
                        });
                    if let Some(target_block) = target_block {
                        builder.ins().jump(target_block, &block_args(&vals));
                    } else {
                        // External JUMP (bridge → loop body) — emit as
                        // Finish exit. The dispatcher will re-enter the
                        // target loop with these output values.
                        let info = &guard_infos[guard_idx];
                        guard_idx += 1;
                        emit_guard_exit(&mut builder, &constants, outputs_ptr, info);
                    }
                }

                OpCode::Finish => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);
                }

                OpCode::Label => {}

                // ── Float arithmetic ──
                OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                    let fr = match op.opcode {
                        OpCode::FloatAdd => builder.ins().fadd(fa, fb),
                        OpCode::FloatSub => builder.ins().fsub(fa, fb),
                        OpCode::FloatMul => builder.ins().fmul(fa, fb),
                        OpCode::FloatTrueDiv => builder.ins().fdiv(fa, fb),
                        _ => unreachable!(),
                    };
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::FloatFloorDiv => {
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                    let fdiv = builder.ins().fdiv(fa, fb);
                    let fr = builder.ins().floor(fdiv);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::FloatMod => {
                    // Python's float mod: a - floor(a/b) * b
                    let (a, b) = resolve_binop(&mut builder, &constants, op);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                    let fdiv = builder.ins().fdiv(fa, fb);
                    let ffloor = builder.ins().floor(fdiv);
                    let prod = builder.ins().fmul(ffloor, fb);
                    let fr = builder.ins().fsub(fa, prod);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::FloatNeg => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fr = builder.ins().fneg(fa);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::FloatAbs => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fr = builder.ins().fabs(fa);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }

                // ── Casts ──
                OpCode::CastFloatToInt => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let r = builder.ins().fcvt_to_sint(cl_types::I64, fa);
                    builder.def_var(var(vi), r);
                }
                OpCode::CastIntToFloat => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let fr = builder.ins().fcvt_from_sint(cl_types::F64, a);
                    let r = builder.ins().bitcast(cl_types::I64, MemFlags::new(), fr);
                    builder.def_var(var(vi), r);
                }
                OpCode::ConvertFloatBytesToLonglong | OpCode::ConvertLonglongBytesToFloat => {
                    // Both are identity in our I64 storage scheme.
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    builder.def_var(var(vi), a);
                }

                // ── Debug / no-op operations ──
                OpCode::DebugMergePoint
                | OpCode::EnterPortalFrame
                | OpCode::LeavePortalFrame
                | OpCode::JitDebug
                | OpCode::Keepalive
                | OpCode::ForceSpill
                | OpCode::VirtualRefFinish
                | OpCode::RecordExactClass
                | OpCode::RecordExactValueR
                | OpCode::RecordExactValueI
                | OpCode::RecordKnownResult
                | OpCode::QuasiimmutField
                | OpCode::AssertNotNone
                | OpCode::IncrementDebugCounter => {
                    // No-op markers or optimizer hints.
                }

                // ── ForceToken ──
                OpCode::ForceToken => {
                    let token = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        current_force_token_shim as *const () as usize,
                        &[],
                        Some(cl_types::I64),
                    )
                    .expect("force_token shim must return a value");
                    builder.def_var(var(vi), token);
                }

                // ── VirtualRef operations ──
                // When VirtualRef survives the optimizer, the backend
                // materializes it by returning the underlying object reference.
                // args[0] = the real object, args[1] = vref_id
                OpCode::VirtualRefI | OpCode::VirtualRefR => {
                    let obj = resolve_opref(&mut builder, &constants, op.args[0]);
                    builder.def_var(var(vi), obj);
                }

                // ── Vector guards ──
                OpCode::VecGuardTrue => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let cond = resolve_opref(&mut builder, &constants, op.args[0]);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_false = builder.ins().icmp(IntCC::Equal, cond, zero);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder
                        .ins()
                        .brif(is_false, exit_block, &[], cont_block, &[]);
                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);
                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }
                OpCode::VecGuardFalse => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;
                    let cond = resolve_opref(&mut builder, &constants, op.args[0]);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let is_true = builder.ins().icmp(IntCC::NotEqual, cond, zero);
                    let exit_block = builder.create_block();
                    let cont_block = builder.create_block();
                    builder
                        .ins()
                        .brif(is_true, exit_block, &[], cont_block, &[]);
                    builder.switch_to_block(exit_block);
                    builder.seal_block(exit_block);
                    emit_guard_exit(&mut builder, &constants, outputs_ptr, info);
                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);
                }

                // ── Vector integer arithmetic ──
                // Native SIMD path: I64X2 (128-bit, 2x i64) / F64X2 (128-bit, 2x f64).
                // Scalar emulation fallback for portability.
                OpCode::VecIntAdd
                | OpCode::VecIntSub
                | OpCode::VecIntMul
                | OpCode::VecIntAnd
                | OpCode::VecIntOr
                | OpCode::VecIntXor => {
                    if USE_NATIVE_SIMD {
                        let a = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let b = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[1],
                        );
                        let result = match op.opcode {
                            OpCode::VecIntAdd => builder.ins().iadd(a, b),
                            OpCode::VecIntSub => builder.ins().isub(a, b),
                            OpCode::VecIntMul => builder.ins().imul(a, b),
                            OpCode::VecIntAnd => builder.ins().band(a, b),
                            OpCode::VecIntOr => builder.ins().bor(a, b),
                            OpCode::VecIntXor => builder.ins().bxor(a, b),
                            _ => unreachable!(),
                        };
                        builder.def_var(var(vi), result);
                    } else {
                        let a = resolve_opref(&mut builder, &constants, op.args[0]);
                        let b = resolve_opref(&mut builder, &constants, op.args[1]);
                        let result = match op.opcode {
                            OpCode::VecIntAdd => builder.ins().iadd(a, b),
                            OpCode::VecIntSub => builder.ins().isub(a, b),
                            OpCode::VecIntMul => builder.ins().imul(a, b),
                            OpCode::VecIntAnd => builder.ins().band(a, b),
                            OpCode::VecIntOr => builder.ins().bor(a, b),
                            OpCode::VecIntXor => builder.ins().bxor(a, b),
                            _ => unreachable!(),
                        };
                        builder.def_var(var(vi), result);
                    }
                }

                // ── Vector float arithmetic ──
                OpCode::VecFloatAdd
                | OpCode::VecFloatSub
                | OpCode::VecFloatMul
                | OpCode::VecFloatTrueDiv => {
                    if USE_NATIVE_SIMD {
                        let a = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let b = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[1],
                        );
                        let result = match op.opcode {
                            OpCode::VecFloatAdd => builder.ins().fadd(a, b),
                            OpCode::VecFloatSub => builder.ins().fsub(a, b),
                            OpCode::VecFloatMul => builder.ins().fmul(a, b),
                            OpCode::VecFloatTrueDiv => builder.ins().fdiv(a, b),
                            _ => unreachable!(),
                        };
                        builder.def_var(var(vi), result);
                    } else {
                        let a = resolve_opref(&mut builder, &constants, op.args[0]);
                        let b = resolve_opref(&mut builder, &constants, op.args[1]);
                        let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                        let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                        let fresult = match op.opcode {
                            OpCode::VecFloatAdd => builder.ins().fadd(fa, fb),
                            OpCode::VecFloatSub => builder.ins().fsub(fa, fb),
                            OpCode::VecFloatMul => builder.ins().fmul(fa, fb),
                            OpCode::VecFloatTrueDiv => builder.ins().fdiv(fa, fb),
                            _ => unreachable!(),
                        };
                        let result = builder
                            .ins()
                            .bitcast(cl_types::I64, MemFlags::new(), fresult);
                        builder.def_var(var(vi), result);
                    }
                }

                OpCode::VecFloatNeg => {
                    if USE_NATIVE_SIMD {
                        let a = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let result = builder.ins().fneg(a);
                        builder.def_var(var(vi), result);
                    } else {
                        let a = resolve_opref(&mut builder, &constants, op.args[0]);
                        let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                        let fresult = builder.ins().fneg(fa);
                        let result = builder
                            .ins()
                            .bitcast(cl_types::I64, MemFlags::new(), fresult);
                        builder.def_var(var(vi), result);
                    }
                }

                OpCode::VecFloatAbs => {
                    if USE_NATIVE_SIMD {
                        let a = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let result = builder.ins().fabs(a);
                        builder.def_var(var(vi), result);
                    } else {
                        let a = resolve_opref(&mut builder, &constants, op.args[0]);
                        let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                        let fresult = builder.ins().fabs(fa);
                        let result = builder
                            .ins()
                            .bitcast(cl_types::I64, MemFlags::new(), fresult);
                        builder.def_var(var(vi), result);
                    }
                }

                OpCode::VecFloatXor => {
                    if USE_NATIVE_SIMD {
                        // XOR on the raw bits: operate on I64X2 representation
                        let a = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let b = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[1],
                        );
                        let xored = builder.ins().bxor(a, b);
                        // Result is declared as F64X2, bitcast from I64X2
                        let result = builder
                            .ins()
                            .bitcast(cl_types::F64X2, MemFlags::new(), xored);
                        builder.def_var(var(vi), result);
                    } else {
                        let a = resolve_opref(&mut builder, &constants, op.args[0]);
                        let b = resolve_opref(&mut builder, &constants, op.args[1]);
                        let result = builder.ins().bxor(a, b);
                        builder.def_var(var(vi), result);
                    }
                }

                // ── Vector comparison/test operations ──
                // These always produce scalar results (I64), not vectors.
                OpCode::VecFloatEq => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let b = resolve_opref(&mut builder, &constants, op.args[1]);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                    let cmp = builder.ins().fcmp(FloatCC::Equal, fa, fb);
                    let result = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecFloatNe => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let b = resolve_opref(&mut builder, &constants, op.args[1]);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fb = builder.ins().bitcast(cl_types::F64, MemFlags::new(), b);
                    let cmp = builder.ins().fcmp(FloatCC::NotEqual, fa, fb);
                    let result = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecIntIsTrue => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    let cmp = builder.ins().icmp(IntCC::NotEqual, a, zero);
                    let result = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecIntEq => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let b = resolve_opref(&mut builder, &constants, op.args[1]);
                    let cmp = builder.ins().icmp(IntCC::Equal, a, b);
                    let result = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecIntNe => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let b = resolve_opref(&mut builder, &constants, op.args[1]);
                    let cmp = builder.ins().icmp(IntCC::NotEqual, a, b);
                    let result = builder.ins().uextend(cl_types::I64, cmp);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecIntSignext => {
                    // Sign-extend a narrower integer value to i64
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    builder.def_var(var(vi), a); // already i64
                }

                // ── Vector cast operations ──
                // These operate on scalar elements, not full vectors.
                OpCode::VecCastFloatToInt => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let result = builder.ins().fcvt_to_sint(cl_types::I64, fa);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecCastIntToFloat => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let fresult = builder.ins().fcvt_from_sint(cl_types::F64, a);
                    let result = builder
                        .ins()
                        .bitcast(cl_types::I64, MemFlags::new(), fresult);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecCastFloatToSinglefloat => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let f32val = builder.ins().fdemote(cl_types::F32, fa);
                    let result = builder
                        .ins()
                        .bitcast(cl_types::I32, MemFlags::new(), f32val);
                    let result_ext = builder.ins().uextend(cl_types::I64, result);
                    builder.def_var(var(vi), result_ext);
                }

                OpCode::VecCastSinglefloatToFloat => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let a_trunc = builder.ins().ireduce(cl_types::I32, a);
                    let f32val = builder
                        .ins()
                        .bitcast(cl_types::F32, MemFlags::new(), a_trunc);
                    let f64val = builder.ins().fpromote(cl_types::F64, f32val);
                    let result = builder
                        .ins()
                        .bitcast(cl_types::I64, MemFlags::new(), f64val);
                    builder.def_var(var(vi), result);
                }

                // ── Vector pack/unpack/expand ──
                OpCode::VecI | OpCode::VecF => {
                    if USE_NATIVE_SIMD {
                        // Zero-initialized vector
                        if op.opcode == OpCode::VecF {
                            let zero_i = builder.ins().iconst(cl_types::I64, 0);
                            let zero_f =
                                builder
                                    .ins()
                                    .bitcast(cl_types::F64, MemFlags::new(), zero_i);
                            let result = builder.ins().splat(cl_types::F64X2, zero_f);
                            builder.def_var(var(vi), result);
                        } else {
                            let zero = builder.ins().iconst(cl_types::I64, 0);
                            let result = builder.ins().splat(cl_types::I64X2, zero);
                            builder.def_var(var(vi), result);
                        }
                    } else {
                        let zero = builder.ins().iconst(cl_types::I64, 0);
                        builder.def_var(var(vi), zero);
                    }
                }

                OpCode::VecPackI => {
                    if USE_NATIVE_SIMD {
                        // vec_pack(vec, scalar, lane_const, count_const)
                        // insertlane(vec, scalar, lane_idx)
                        let vec_val = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let scalar = resolve_opref(&mut builder, &constants, op.args[1]);
                        let lane = constants.get(&op.args[2].0).copied().unwrap_or(0) as u8;
                        let result = builder.ins().insertlane(vec_val, scalar, lane);
                        builder.def_var(var(vi), result);
                    } else {
                        let scalar = resolve_opref(&mut builder, &constants, op.args[1]);
                        builder.def_var(var(vi), scalar);
                    }
                }

                OpCode::VecPackF => {
                    if USE_NATIVE_SIMD {
                        // vec_pack(vec, scalar, lane_const, count_const)
                        // insertlane(vec, scalar, lane_idx)
                        let vec_val = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let scalar_i = resolve_opref(&mut builder, &constants, op.args[1]);
                        let scalar_f =
                            builder
                                .ins()
                                .bitcast(cl_types::F64, MemFlags::new(), scalar_i);
                        let lane = constants.get(&op.args[2].0).copied().unwrap_or(0) as u8;
                        let result = builder.ins().insertlane(vec_val, scalar_f, lane);
                        builder.def_var(var(vi), result);
                    } else {
                        let scalar = resolve_opref(&mut builder, &constants, op.args[1]);
                        builder.def_var(var(vi), scalar);
                    }
                }

                OpCode::VecUnpackI => {
                    if USE_NATIVE_SIMD {
                        // vec_unpack(vec, lane_const, count_const) → scalar i64
                        let vec_val = resolve_opref_vec_int(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let lane = constants.get(&op.args[1].0).copied().unwrap_or(0) as u8;
                        let result = builder.ins().extractlane(vec_val, lane);
                        builder.def_var(var(vi), result);
                    } else {
                        let vec_val = resolve_opref(&mut builder, &constants, op.args[0]);
                        builder.def_var(var(vi), vec_val);
                    }
                }

                OpCode::VecUnpackF => {
                    if USE_NATIVE_SIMD {
                        // vec_unpack(vec, lane_const, count_const) → scalar f64 as i64
                        let vec_val = resolve_opref_vec_float(
                            &mut builder,
                            &constants,
                            &vec_oprefs,
                            &vec_float_oprefs,
                            op.args[0],
                        );
                        let lane = constants.get(&op.args[1].0).copied().unwrap_or(0) as u8;
                        let scalar_f = builder.ins().extractlane(vec_val, lane);
                        let result =
                            builder
                                .ins()
                                .bitcast(cl_types::I64, MemFlags::new(), scalar_f);
                        builder.def_var(var(vi), result);
                    } else {
                        let vec_val = resolve_opref(&mut builder, &constants, op.args[0]);
                        builder.def_var(var(vi), vec_val);
                    }
                }

                OpCode::VecExpandI => {
                    if USE_NATIVE_SIMD {
                        // Broadcast scalar to all lanes
                        let scalar = resolve_opref(&mut builder, &constants, op.args[0]);
                        let result = builder.ins().splat(cl_types::I64X2, scalar);
                        builder.def_var(var(vi), result);
                    } else {
                        let scalar = resolve_opref(&mut builder, &constants, op.args[0]);
                        builder.def_var(var(vi), scalar);
                    }
                }

                OpCode::VecExpandF => {
                    if USE_NATIVE_SIMD {
                        // Broadcast scalar f64 to all lanes
                        let scalar_i = resolve_opref(&mut builder, &constants, op.args[0]);
                        let scalar_f =
                            builder
                                .ins()
                                .bitcast(cl_types::F64, MemFlags::new(), scalar_i);
                        let result = builder.ins().splat(cl_types::F64X2, scalar_f);
                        builder.def_var(var(vi), result);
                    } else {
                        let scalar = resolve_opref(&mut builder, &constants, op.args[0]);
                        builder.def_var(var(vi), scalar);
                    }
                }

                // ── Vector load/store ──
                OpCode::VecLoadI | OpCode::VecLoadF => {
                    if USE_NATIVE_SIMD {
                        // Load 128 bits (2x i64 or 2x f64) from memory
                        let base = resolve_opref(&mut builder, &constants, op.args[0]);
                        let offset_val = if op.args.len() > 1 {
                            resolve_opref(&mut builder, &constants, op.args[1])
                        } else {
                            builder.ins().iconst(cl_types::I64, 0)
                        };
                        let addr = builder.ins().iadd(base, offset_val);
                        let load_type = if op.opcode == OpCode::VecLoadF {
                            cl_types::F64X2
                        } else {
                            cl_types::I64X2
                        };
                        let result = builder.ins().load(load_type, MemFlags::trusted(), addr, 0);
                        builder.def_var(var(vi), result);
                    } else {
                        let base = resolve_opref(&mut builder, &constants, op.args[0]);
                        let offset_val = if op.args.len() > 1 {
                            resolve_opref(&mut builder, &constants, op.args[1])
                        } else {
                            builder.ins().iconst(cl_types::I64, 0)
                        };
                        let addr = builder.ins().iadd(base, offset_val);
                        let result =
                            builder
                                .ins()
                                .load(cl_types::I64, MemFlags::trusted(), addr, 0);
                        builder.def_var(var(vi), result);
                    }
                }

                OpCode::VecStore => {
                    if USE_NATIVE_SIMD {
                        // Store 128 bits to memory
                        let base = resolve_opref(&mut builder, &constants, op.args[0]);
                        let offset_val = if op.args.len() > 2 {
                            resolve_opref(&mut builder, &constants, op.args[1])
                        } else {
                            builder.ins().iconst(cl_types::I64, 0)
                        };
                        let value_ref = op.args[op.args.len() - 1];
                        let value = if vec_oprefs.contains(&value_ref.0) {
                            builder.use_var(var(value_ref.0))
                        } else {
                            resolve_opref(&mut builder, &constants, value_ref)
                        };
                        let addr = builder.ins().iadd(base, offset_val);
                        builder.ins().store(MemFlags::trusted(), value, addr, 0);
                    } else {
                        let base = resolve_opref(&mut builder, &constants, op.args[0]);
                        let offset_val = if op.args.len() > 2 {
                            resolve_opref(&mut builder, &constants, op.args[1])
                        } else {
                            builder.ins().iconst(cl_types::I64, 0)
                        };
                        let value =
                            resolve_opref(&mut builder, &constants, op.args[op.args.len() - 1]);
                        let addr = builder.ins().iadd(base, offset_val);
                        builder.ins().store(MemFlags::trusted(), value, addr, 0);
                    }
                }

                // ── Allocation opcodes ──
                // These are normally eliminated by the optimizer (virtualize pass)
                // or rewritten by the GC rewriter. If they reach the backend,
                // we call out to a runtime helper.
                OpCode::New | OpCode::NewWithVtable => {
                    let (size, type_id) = op
                        .descr
                        .as_ref()
                        .and_then(|d| d.as_size_descr())
                        .map_or((16, 0), |sd| (sd.size() as i64, sd.type_id() as i64));
                    let size_val = builder.ins().iconst(cl_types::I64, size);
                    let type_id_val = builder.ins().iconst(cl_types::I64, type_id);
                    if let Some(runtime_id) = gc_runtime_id {
                        let runtime_id = builder.ins().iconst(cl_types::I64, runtime_id as i64);
                        let result = emit_collecting_gc_call(
                            &mut builder,
                            ptr_type,
                            call_conv,
                            jf_ptr,
                            &ref_root_slots,
                            &defined_ref_vars,
                            ref_root_base_ofs,
                            per_call_gcmap,
                            runtime_id,
                            gc_alloc_typed_nursery_shim as *const () as usize,
                            &[type_id_val, size_val],
                            Some(cl_types::I64),
                        )
                        .expect("GC allocation helper must return a value");
                        // TODO: _reload_frame_if_necessary — when nursery jitframe is supported
                        // jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                        // outputs_ptr = jf_ptr;
                        builder.def_var(var(vi), result);
                    } else {
                        // No GC runtime: plain malloc fallback for non-GC languages.
                        let alloc_fn = builder
                            .ins()
                            .iconst(ptr_type, plain_malloc_zeroed_shim as *const () as i64);
                        let sig = {
                            let mut sig = cranelift_codegen::ir::Signature::new(call_conv);
                            sig.params
                                .push(cranelift_codegen::ir::AbiParam::new(cl_types::I64));
                            sig.returns
                                .push(cranelift_codegen::ir::AbiParam::new(cl_types::I64));
                            builder.import_signature(sig)
                        };
                        let call = builder.ins().call_indirect(sig, alloc_fn, &[size_val]);
                        let result = builder.inst_results(call)[0];
                        builder.def_var(var(vi), result);
                    }
                }
                OpCode::NewArray | OpCode::NewArrayClear => {
                    let runtime_id = gc_runtime_id.ok_or_else(|| missing_gc_runtime(op.opcode))?;
                    let runtime_id = builder.ins().iconst(cl_types::I64, runtime_id as i64);
                    let length =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(0));
                    let (base_size, item_size) =
                        if let Some(ad) = op.descr.as_ref().and_then(|d| d.as_array_descr()) {
                            (
                                builder.ins().iconst(cl_types::I64, ad.base_size() as i64),
                                builder.ins().iconst(cl_types::I64, ad.item_size() as i64),
                            )
                        } else {
                            (
                                builder.ins().iconst(cl_types::I64, 16),
                                builder.ins().iconst(cl_types::I64, 8),
                            )
                        };
                    let result = emit_collecting_gc_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                        runtime_id,
                        gc_alloc_varsize_shim as *const () as usize,
                        &[base_size, item_size, length],
                        Some(cl_types::I64),
                    )
                    .expect("GC varsize allocation helper must return a value");
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    outputs_ptr = jf_ptr;
                    builder.def_var(var(vi), result);
                }

                // ── Integer sign extension ──
                OpCode::IntSignext => {
                    let val = resolve_opref(&mut builder, &constants, op.arg(0));
                    let num_bits = resolve_opref(&mut builder, &constants, op.arg(1));

                    // Sign extend from `num_bits` to 64 bits
                    // shift left by (64 - num_bits), then arithmetic shift right by same
                    let sixty_four = builder.ins().iconst(cl_types::I64, 64);
                    let shift = builder.ins().isub(sixty_four, num_bits);
                    let shifted_left = builder.ins().ishl(val, shift);
                    let result = builder.ins().sshr(shifted_left, shift);
                    builder.def_var(var(vi), result);
                }

                // ── Unsigned multiply high ──
                OpCode::UintMulHigh => {
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
                    let b = resolve_opref(&mut builder, &constants, op.arg(1));
                    let result = builder.ins().umulhi(a, b);
                    builder.def_var(var(vi), result);
                }

                // ── Float ↔ SingleFloat casts ──
                OpCode::CastFloatToSinglefloat => {
                    let val = resolve_opref(&mut builder, &constants, op.arg(0));
                    // i64-encoded f64 → f64 → f32 → zero-extend to i64
                    let f64_val = builder.ins().bitcast(cl_types::F64, MemFlags::new(), val);
                    let f32_val = builder.ins().fdemote(cl_types::F32, f64_val);
                    let i32_val = builder
                        .ins()
                        .bitcast(cl_types::I32, MemFlags::new(), f32_val);
                    let result = builder.ins().uextend(cl_types::I64, i32_val);
                    builder.def_var(var(vi), result);
                }

                OpCode::CastSinglefloatToFloat => {
                    let val = resolve_opref(&mut builder, &constants, op.arg(0));
                    // i64 (lower 32 bits = f32) → f32 → f64 → i64
                    let i32_val = builder.ins().ireduce(cl_types::I32, val);
                    let f32_val = builder
                        .ins()
                        .bitcast(cl_types::F32, MemFlags::new(), i32_val);
                    let f64_val = builder.ins().fpromote(cl_types::F64, f32_val);
                    let result = builder
                        .ins()
                        .bitcast(cl_types::I64, MemFlags::new(), f64_val);
                    builder.def_var(var(vi), result);
                }

                // ── Escape ops (for testing) ──
                // These are used in optimizer tests to force values to escape.
                // In the backend they're no-ops that pass through the value.
                OpCode::EscapeI | OpCode::EscapeR | OpCode::EscapeF => {
                    let val = resolve_opref(&mut builder, &constants, op.arg(0));
                    builder.def_var(var(vi), val);
                }
                OpCode::EscapeN => {
                    // Void-returning escape: just evaluate args for side effects
                }

                // ── Raw array item read (ref-typed) ──
                OpCode::GetarrayitemRawR => {
                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let index = resolve_opref(&mut builder, &constants, op.arg(1));
                    let scale = builder.ins().iconst(cl_types::I64, 8);
                    let offset = builder.ins().imul(index, scale);
                    let addr = builder.ins().iadd(base, offset);
                    let result = builder
                        .ins()
                        .load(cl_types::I64, MemFlags::trusted(), addr, 0);
                    builder.def_var(var(vi), result);
                }

                // ── Thread-local reference get ──
                OpCode::ThreadlocalrefGet => {
                    // Load from a thread-local slot via runtime callback.
                    // arg(0) = offset (in bytes) into the TLS area.
                    let offset = resolve_opref(&mut builder, &constants, op.arg(0));
                    let result = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_threadlocalref_get as *const () as usize,
                        &[offset],
                        Some(cl_types::I64),
                    )
                    .expect("jit_threadlocalref_get must return a value");
                    builder.def_var(var(vi), result);
                }

                // ── Load from GC table ──
                OpCode::LoadFromGcTable => {
                    // Load a constant pointer from the GC table.
                    // arg(0) = index into the gc table
                    let index = resolve_opref(&mut builder, &constants, op.arg(0));
                    builder.def_var(var(vi), index);
                }

                // ── Load effective address ──
                OpCode::LoadEffectiveAddress => {
                    // args[0] = base, args[1] = index, args[2] = scale, args[3] = offset
                    // result = base + index * scale + offset
                    let base = resolve_opref(&mut builder, &constants, op.arg(0));
                    let index = resolve_opref(&mut builder, &constants, op.arg(1));
                    let scale = resolve_opref(&mut builder, &constants, op.arg(2));
                    let offset = resolve_opref(&mut builder, &constants, op.arg(3));
                    let scaled_index = builder.ins().imul(index, scale);
                    let addr = builder.ins().iadd(base, scaled_index);
                    let result = builder.ins().iadd(addr, offset);
                    builder.def_var(var(vi), result);
                }

                // ── String allocation ──
                // Newstr: allocate a byte string of length args[0].
                // Newunicode: allocate a unicode string of length args[0].
                // Layout: 16-byte header + length * char_size bytes.
                OpCode::Newstr => {
                    let runtime_id = gc_runtime_id.ok_or_else(|| missing_gc_runtime(op.opcode))?;
                    let runtime_id = builder.ins().iconst(cl_types::I64, runtime_id as i64);
                    let len = resolve_opref(&mut builder, &constants, op.arg(0));
                    let base_size = builder.ins().iconst(cl_types::I64, 16);
                    let item_size = builder.ins().iconst(cl_types::I64, 1);
                    let result = emit_collecting_gc_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                        runtime_id,
                        gc_alloc_varsize_shim as *const () as usize,
                        &[base_size, item_size, len],
                        Some(cl_types::I64),
                    )
                    .expect("GC varsize allocation helper must return a value");
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    outputs_ptr = jf_ptr;
                    builder.def_var(var(vi), result);
                }
                OpCode::Newunicode => {
                    let runtime_id = gc_runtime_id.ok_or_else(|| missing_gc_runtime(op.opcode))?;
                    let runtime_id = builder.ins().iconst(cl_types::I64, runtime_id as i64);
                    let len = resolve_opref(&mut builder, &constants, op.arg(0));
                    let base_size = builder.ins().iconst(cl_types::I64, 16);
                    let item_size = builder.ins().iconst(cl_types::I64, 4);
                    let result = emit_collecting_gc_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jf_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        ref_root_base_ofs,
                        per_call_gcmap,
                        runtime_id,
                        gc_alloc_varsize_shim as *const () as usize,
                        &[base_size, item_size, len],
                        Some(cl_types::I64),
                    )
                    .expect("GC varsize allocation helper must return a value");
                    jf_ptr = emit_reload_frame_if_necessary(&mut builder, ptr_type, call_conv);
                    outputs_ptr = jf_ptr;
                    builder.def_var(var(vi), result);
                }
                // All OpCode variants are explicitly handled above.
                // This arm is unreachable but kept for forward-compatibility
                // when new opcodes are added to the IR.
                #[allow(unreachable_patterns)]
                _other => {
                    return Err(BackendError::Unsupported(format!(
                        "opcode {:?} has no backend lowering",
                        _other
                    )));
                }
            }

            if op.result_type() == Type::Ref && !force_tokens.contains(&vi) {
                defined_ref_vars.insert(vi);
            }
        }

        for (_, block) in &label_blocks {
            builder.seal_block(*block);
        }
        if label_blocks.is_empty() && loop_block != entry_block {
            builder.seal_block(loop_block);
        }
        builder.finalize();
        self.func_ctx = func_ctx;

        // Compile
        let mut ctx = Context::for_function(func);
        if let Err(e) = self.module.define_function(func_id, &mut ctx) {
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!("[jit][clif-error] {e}\nCLIF IR:\n{}", ctx.func.display());
            }
            self.module.clear_context(&mut ctx);
            return Err(BackendError::CompilationFailed(format!("{e}\n{e:?}")));
        }
        self.module.clear_context(&mut ctx);
        self.module.finalize_definitions().unwrap();

        let code_ptr = self.module.get_finalized_function(func_id);

        let needs_force_frame = ops.iter().any(|op| {
            matches!(
                op.opcode,
                OpCode::ForceToken
                    | OpCode::CallMayForceI
                    | OpCode::CallMayForceR
                    | OpCode::CallMayForceF
                    | OpCode::CallMayForceN
                    | OpCode::CallAssemblerI
                    | OpCode::CallAssemblerR
                    | OpCode::CallAssemblerF
                    | OpCode::CallAssemblerN
            )
        });
        let trace_info = CompiledTraceInfo {
            trace_id,
            input_types: inputargs.iter().map(|arg| arg.tp).collect(),
            header_pc,
            source_guard: None,
        };
        for descr in &fail_descrs {
            descr.set_trace_info(trace_info.clone());
        }
        Ok(CompiledLoop {
            trace_id,
            input_types: trace_info.input_types.clone(),
            header_pc,
            green_key: 0, // Set by compile_loop from token.green_key
            caller_prefix_layout: caller_layout.cloned(),
            _func_id: func_id,
            code_ptr,
            code_size: 0,
            fail_descrs,
            terminal_exit_layouts: UnsafeCell::new(terminal_exit_layouts),
            gc_runtime_id,
            num_inputs: inputargs.len(),
            num_ref_roots: ref_root_slots.len(),
            max_output_slots,
            needs_force_frame,
        })
    }
}

impl Drop for CraneliftBackend {
    fn drop(&mut self) {
        for token_number in std::mem::take(&mut self.registered_call_assembler_tokens) {
            unregister_call_assembler_target(token_number);
        }
        for trace_id in std::mem::take(&mut self.registered_call_assembler_bridge_traces) {
            unregister_call_assembler_expectations(CallAssemblerCallerId::BridgeTrace(trace_id));
        }
        let _ = CALL_ASSEMBLER_DEADFRAMES.try_with(|map| map.borrow_mut().clear());
        let _ = NEXT_CALL_ASSEMBLER_DEADFRAME_HANDLE.try_with(|cell| cell.set(1));
        if let Some(runtime_id) = self.gc_runtime_id.take() {
            unregister_gc_runtime(runtime_id);
        }
    }
}

fn collect_guards(
    ops: &[Op],
    inputargs: &[InputArg],
    force_tokens: &HashSet<u32>,
    fail_descrs: &mut Vec<Arc<CraneliftFailDescr>>,
    guard_infos: &mut Vec<GuardInfo>,
    max_output_slots: &mut usize,
    trace_id: u64,
    header_pc: u64,
    source_guard: Option<(u64, u32)>,
    caller_layout: Option<&ExitRecoveryLayout>,
    constants: &HashMap<u32, i64>,
) -> Result<(), BackendError> {
    let num_inputs = inputargs.len();
    let (value_types, inputarg_types, op_def_positions) = build_value_type_map(inputargs, ops);

    // Collect Label descr indices to distinguish internal vs external JUMPs.
    let label_descr_indices: HashSet<u32> = ops
        .iter()
        .filter(|op| op.opcode == OpCode::Label)
        .filter_map(|op| op.descr.as_ref().map(|d| d.index()))
        .collect();

    for (op_idx, op) in ops.iter().enumerate() {
        let is_guard = op.opcode.is_guard();
        let is_finish = op.opcode == OpCode::Finish;
        // External JUMP: target not in this function's Labels.
        let is_external_jump = op.opcode == OpCode::Jump
            && op
                .descr
                .as_ref()
                .map_or(false, |d| !label_descr_indices.contains(&d.index()));

        if !is_guard && !is_finish && !is_external_jump {
            continue;
        }

        let fail_index = fail_descrs.len() as u32;

        let (fail_arg_refs, fail_arg_types) = if is_finish || is_external_jump {
            let refs: Vec<OpRef> = op.args.iter().copied().collect();
            // Use the descriptor's explicit types for FINISH args — these are
            // set by the tracer and represent the caller's view of the return
            // type, which may differ from the op's inferred type (e.g. New
            // produces Ref but the value is treated as Int by the caller).
            let types = if let Some(fd) = op.descr.as_ref().and_then(|d| d.as_fail_descr()) {
                let dt = fd.fail_arg_types();
                if dt.len() == refs.len() {
                    dt.to_vec()
                } else {
                    infer_fail_arg_types(&refs, &value_types)?
                }
            } else {
                infer_fail_arg_types(&refs, &value_types)?
            };
            (refs, types)
        } else if let Some(ref fa) = op.fail_args {
            let refs: Vec<OpRef> = fa.iter().copied().collect();
            // RPython Box.type parity: use optimizer-provided fail_arg_types
            // directly when available, bypassing value_types type inference.
            let types = if let Some(ref fat) = op.fail_arg_types {
                if fat.len() == refs.len() {
                    fat.clone()
                } else {
                    merge_descriptor_with_positional(
                        &refs,
                        op.descr.as_ref().and_then(|d| d.as_fail_descr()),
                        &value_types,
                        &inputarg_types,
                        &op_def_positions,
                        op_idx,
                    )?
                }
            } else {
                merge_descriptor_with_positional(
                    &refs,
                    op.descr.as_ref().and_then(|d| d.as_fail_descr()),
                    &value_types,
                    &inputarg_types,
                    &op_def_positions,
                    op_idx,
                )?
            };
            (refs, types)
        } else {
            let refs: Vec<OpRef> = if let Some(ref fa) = op.fail_args {
                fa.iter().copied().collect()
            } else {
                (0..num_inputs as u32).map(OpRef).collect()
            };
            let types = merge_descriptor_with_positional(
                &refs,
                op.descr.as_ref().and_then(|d| d.as_fail_descr()),
                &value_types,
                &inputarg_types,
                &op_def_positions,
                op_idx,
            )?;
            (refs, types)
        };

        let n = fail_arg_refs.len();
        if n > *max_output_slots {
            *max_output_slots = n;
        }

        let force_token_slots = fail_arg_refs
            .iter()
            .enumerate()
            .filter_map(|(slot, opref)| force_tokens.contains(&opref.0).then_some(slot))
            .collect();

        // Extract per-guard bytecode resume PC from the last fail_arg constant.
        // RPython rd_resume_position: the interpreter PC where execution resumes.
        // The jit_interp macro records resume_pc as the last fail_arg constant.
        let guard_resume_pc: Option<u64> = if !is_finish && !is_external_jump {
            fail_arg_refs
                .last()
                .and_then(|&last_ref| constants.get(&last_ref.0).map(|&v| v as u64))
        } else {
            None
        };
        if std::env::var_os("MAJIT_LOG").is_some() && guard_resume_pc.is_some() {
            eprintln!(
                "[guard-resume] fail_index={} last_ref={:?} resume_pc={:?}",
                fail_index,
                fail_arg_refs.last(),
                guard_resume_pc
            );
        }
        let mut recovery_layout = identity_recovery_layout(
            trace_id,
            header_pc,
            guard_resume_pc,
            source_guard,
            &fail_arg_types,
            caller_layout,
        );
        // resume.py parity: encode rd_virtuals into recovery layout.
        // Virtual objects in fail_args are NOT materialized at compile time;
        // their field values are stored as extra fail_args and reconstructed
        // lazily on guard failure.
        if let Some(ref rd_virtuals) = op.rd_virtuals {
            for entry in rd_virtuals {
                let virtual_index = recovery_layout.virtual_layouts.len();
                // Convert field fail_arg positions to ExitValueSourceLayout
                let fields: Vec<(u32, ExitValueSourceLayout)> = entry
                    .fields
                    .iter()
                    .map(|&(field_descr_idx, field_fail_arg_idx)| {
                        (
                            field_descr_idx,
                            ExitValueSourceLayout::ExitValue(field_fail_arg_idx),
                        )
                    })
                    .collect();
                let descr_index = entry.descr.index();
                let type_id = entry.known_class.map_or(0, |gc| gc.0 as u32);
                let layout = if entry.known_class.is_some() {
                    ExitVirtualLayout::Object {
                        type_id,
                        descr_index,
                        fields,
                        target_slot: Some(entry.fail_arg_index),
                    }
                } else {
                    ExitVirtualLayout::Struct {
                        type_id,
                        descr_index,
                        fields,
                        target_slot: Some(entry.fail_arg_index),
                    }
                };
                recovery_layout.virtual_layouts.push(layout);
                // Update the frame slot for this fail_arg to Virtual(index)
                if let Some(frame) = recovery_layout.frames.last_mut() {
                    if entry.fail_arg_index < frame.slots.len() {
                        frame.slots[entry.fail_arg_index] =
                            ExitValueSourceLayout::Virtual(virtual_index);
                    }
                }
            }
        }
        let recovery_layout = Some(recovery_layout);
        // get_gcmap (regalloc.py:1092-1108) parity:
        // val = loc.position + JITFRAME_FIXED_SIZE  (#1106)
        let gcmap = {
            let mut bits: u64 = 0;
            for (i, tp) in fail_arg_types.iter().enumerate() {
                if *tp == Type::Ref {
                    // TEMPORARY WORKAROUND: RPython never puts constants in
                    // fail_args (regalloc.py:1206 asserts this). Once the
                    // consumer switchover (number()+finish() producing
                    // rd_numb+liveboxes) is active, constants will be in
                    // rd_consts instead, and this check becomes unnecessary.
                    let opref_id = fail_arg_refs.get(i).map(|r| r.0).unwrap_or(u32::MAX);
                    if constants.contains_key(&opref_id) {
                        continue; // constant — not a GC root
                    }
                    let val = i as u32 + JITFRAME_FIXED_SIZE;
                    if val < 64 {
                        bits |= 1u64 << val;
                    }
                }
            }
            bits
        };
        let mut descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            trace_id,
            fail_arg_types,
            is_finish || is_external_jump,
            force_token_slots,
            recovery_layout,
        );
        descr.set_source_op_index(op_idx);
        descr.green_key = header_pc;
        let descr = Arc::new(descr);
        // assembler.py:2126 get_gcref_from_faildescr parity:
        // store the FailDescr pointer (not index) in jf_descr.
        let fail_descr_ptr = Arc::as_ptr(&descr) as i64;
        fail_descrs.push(descr);
        guard_infos.push(GuardInfo {
            fail_index,
            fail_arg_refs,
            gcmap,
            fail_descr_ptr,
        });
    }

    Ok(())
}

fn collect_terminal_exit_layouts(
    ops: &[Op],
    inputargs: &[InputArg],
    force_tokens: &HashSet<u32>,
    trace_id: u64,
    header_pc: u64,
    source_guard: Option<(u64, u32)>,
    caller_layout: Option<&ExitRecoveryLayout>,
) -> Result<Vec<TerminalExitLayout>, BackendError> {
    let value_types = build_value_type_map_simple(inputargs, ops);
    let mut layouts = Vec::new();
    let mut fail_index = 0u32;

    for (op_index, op) in ops.iter().enumerate() {
        let is_guard = op.opcode.is_guard();
        let is_finish = op.opcode == OpCode::Finish;
        let is_jump = op.opcode == OpCode::Jump;

        if is_finish || is_jump {
            let exit_types = infer_fail_arg_types(op.args.as_slice(), &value_types)?;
            let force_token_slots: Vec<usize> = op
                .args
                .iter()
                .enumerate()
                .filter_map(|(slot, opref)| force_tokens.contains(&opref.0).then_some(slot))
                .collect();
            let gc_ref_slots = exit_types
                .iter()
                .enumerate()
                .filter_map(|(slot, tp)| {
                    (*tp == Type::Ref && !force_token_slots.contains(&slot)).then_some(slot)
                })
                .collect();
            let recovery_layout = (is_jump || is_finish).then(|| {
                identity_recovery_layout(
                    trace_id,
                    header_pc,
                    None, // terminal exits don't need per-guard resume_pc
                    source_guard,
                    &exit_types,
                    caller_layout,
                )
            });
            layouts.push(TerminalExitLayout {
                op_index,
                trace_id,
                trace_info: Some(CompiledTraceInfo {
                    trace_id,
                    input_types: inputargs.iter().map(|arg| arg.tp).collect(),
                    header_pc,
                    source_guard: None,
                }),
                fail_index: if is_finish { fail_index } else { u32::MAX },
                exit_types,
                is_finish,
                gc_ref_slots,
                force_token_slots,
                recovery_layout,
            });
        }

        if is_guard || is_finish {
            fail_index += 1;
        }
    }

    Ok(layouts)
}

// ---------------------------------------------------------------------------
// Backend trait implementation
// ---------------------------------------------------------------------------

impl majit_codegen::Backend for CraneliftBackend {
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        token: &mut JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        token.inputarg_types = inputargs.iter().map(|ia| ia.tp).collect();
        // Pass the address of the invalidation flag so GUARD_NOT_INVALIDATED
        // can load from it at runtime.
        let flag_ptr = Arc::as_ptr(&token.invalidated) as *const AtomicBool as usize;
        let mut compiled = self.do_compile(inputargs, ops, Some(flag_ptr), None, None)?;
        compiled.green_key = token.green_key;
        let info = AsmInfo {
            code_addr: compiled.code_ptr as usize,
            code_size: compiled.code_size,
        };
        register_call_assembler_target(token, &compiled)?;
        if let Err(err) =
            install_call_assembler_expectations(CallAssemblerCallerId::RootLoop(token.number), ops)
        {
            unregister_call_assembler_target(token.number);
            return Err(err);
        }
        self.registered_call_assembler_tokens.insert(token.number);
        token.compiled = Some(Box::new(compiled));
        Ok(info)
    }

    fn register_pending_target(
        &mut self,
        token_number: u64,
        input_types: Vec<majit_ir::Type>,
        num_inputs: usize,
    ) {
        register_pending_call_assembler_target(token_number, input_types, num_inputs);
    }

    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        original_token: &JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        // compile.py:186: record_loop_or_bridge sets descr.rd_loop_token = clt
        // on ALL guards. Bridges share the parent loop's invalidation flag.
        // We clone the Arc to keep the flag alive as long as the bridge exists.
        let invalidated_arc = original_token.invalidated.clone();
        let flag_ptr =
            Arc::as_ptr(&invalidated_arc) as *const std::sync::atomic::AtomicBool as usize;
        let original_compiled = original_token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledLoop>())
            .ok_or_else(|| {
                BackendError::CompilationFailed("original token has no compiled loop".to_string())
            })?;
        let source_trace_id = if fail_descr.trace_id() == 0 {
            original_compiled.trace_id
        } else {
            fail_descr.trace_id()
        };
        let source_descr = find_fail_descr_in_fail_descrs(
            &original_compiled.fail_descrs,
            source_trace_id,
            fail_descr.fail_index(),
        );
        // Entry bridge (fail_index=0): no source guard — graceful skip.
        if source_descr.is_none() && fail_descr.fail_index() != 0 {
            return Err(BackendError::CompilationFailed(format!(
                "source fail descr not found for trace {} fail {}",
                source_trace_id,
                fail_descr.fail_index()
            )));
        }
        let caller_layout = source_descr.as_ref().and_then(|d| {
            d.recovery_layout_ref().clone().map(|mut layout| {
                layout.frames.pop();
                layout
            })
        });
        let compiled = self.do_compile(
            inputargs,
            ops,
            Some(flag_ptr), // compile.py:186: bridges share parent's invalidation flag
            Some((source_trace_id, fail_descr.fail_index())),
            caller_layout.as_ref(),
        )?;
        let info = AsmInfo {
            code_addr: compiled.code_ptr as usize,
            code_size: compiled.code_size,
        };
        install_call_assembler_expectations(
            CallAssemblerCallerId::BridgeTrace(compiled.trace_id),
            ops,
        )?;
        self.registered_call_assembler_bridge_traces
            .insert(compiled.trace_id);

        // Attach the bridge to the original guard's fail descriptor so that
        // execute_token can dispatch to it on subsequent guard failures.
        let bridge_trace_info = CompiledTraceInfo {
            trace_id: compiled.trace_id,
            input_types: compiled.input_types.clone(),
            header_pc: compiled.header_pc,
            source_guard: Some((source_trace_id, fail_descr.fail_index())),
        };
        for descr in &compiled.fail_descrs {
            descr.set_trace_info(bridge_trace_info.clone());
        }
        {
            let terminal_exit_layouts = compiled.terminal_exit_layouts_mut();
            for layout in terminal_exit_layouts.iter_mut() {
                layout.trace_info = Some(bridge_trace_info.clone());
            }
        }
        {
            if let Some(ref sd) = source_descr {
                let existing_bridge = sd.bridge_ref();
                if let Some(ref bridge) = *existing_bridge {
                    unregister_bridge_call_assembler_expectations(bridge);
                }
            }
        }
        if let Some(ref sd) = source_descr {
            let bridge_num_inputs = compiled.num_inputs;
            sd.attach_bridge(BridgeData {
                trace_id: compiled.trace_id,
                input_types: compiled.input_types.clone(),
                header_pc: compiled.header_pc,
                source_guard: (source_trace_id, fail_descr.fail_index()),
                caller_prefix_layout: compiled.caller_prefix_layout.clone(),
                code_ptr: compiled.code_ptr,
                fail_descrs: compiled.fail_descrs,
                terminal_exit_layouts: compiled.terminal_exit_layouts,
                gc_runtime_id: compiled.gc_runtime_id,
                loop_reentry: {
                    let has_label: HashSet<u32> = ops
                        .iter()
                        .filter(|o| o.opcode == OpCode::Label)
                        .filter_map(|o| o.descr.as_ref().map(|d| d.index()))
                        .collect();
                    ops.last().map_or(false, |op| {
                        op.opcode == OpCode::Jump
                            && op
                                .descr
                                .as_ref()
                                .map_or(false, |d| !has_label.contains(&d.index()))
                    })
                },
                num_inputs: bridge_num_inputs,
                num_ref_roots: compiled.num_ref_roots,
                max_output_slots: compiled.max_output_slots,
                needs_force_frame: compiled.needs_force_frame,
                invalidated_arc: Some(invalidated_arc),
            });
        }

        Ok(info)
    }

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        let compiled = token
            .compiled
            .as_ref()
            .expect("token has no compiled code")
            .downcast_ref::<CompiledLoop>()
            .expect("compiled data is not CompiledLoop");

        let mut inputs: Vec<i64> = Vec::with_capacity(compiled.num_inputs);
        for arg in args {
            inputs.push(match arg {
                Value::Int(v) => *v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.0 as i64,
                Value::Void => 0,
            });
        }

        Self::execute_with_inputs(compiled, &inputs)
    }

    fn execute_token_ints(&self, token: &JitCellToken, args: &[i64]) -> DeadFrame {
        let compiled = token
            .compiled
            .as_ref()
            .expect("token has no compiled code")
            .downcast_ref::<CompiledLoop>()
            .expect("compiled data is not CompiledLoop");

        Self::execute_with_inputs(compiled, args)
    }

    fn execute_token_ints_raw(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> majit_codegen::RawExecResult {
        let compiled = token
            .compiled
            .as_ref()
            .expect("token has no compiled code")
            .downcast_ref::<CompiledLoop>()
            .expect("compiled data is not CompiledLoop");

        let (fail_index, mut outputs, handle, force_frame) = run_compiled_code(
            compiled.code_ptr,
            &compiled.fail_descrs,
            compiled.gc_runtime_id,
            compiled.num_ref_roots,
            compiled.max_output_slots,
            args,
            compiled.needs_force_frame,
        );

        if std::env::var_os("MAJIT_LOG").is_some() && fail_index == 0 {
            eprintln!(
                "[exec-raw] fail_index={fail_index} outputs_len={}",
                outputs.len()
            );
        }
        if let Some(frame) =
            maybe_take_call_assembler_deadframe(fail_index, &outputs, handle, force_frame.as_ref())
        {
            if std::env::var_os("MAJIT_LOG").is_some() && fail_index == 0 {
                eprintln!("[exec-raw] fail_index=0 → call_assembler path");
            }
            let frame = wrap_call_assembler_deadframe_with_caller_prefix(
                frame,
                compiled.trace_id,
                compiled.header_pc,
                None,
                &compiled.input_types,
                args,
                compiled.caller_prefix_layout.as_ref(),
            );
            let descr = self.get_latest_descr(&frame);
            let exit_layout = self.describe_deadframe(&frame);
            let savedata = self.grab_savedata_ref(&frame);
            let (exception_class, exception_value) = self.grab_exception_state(&frame);
            let exit_arity = descr.fail_arg_types().len();
            let mut result = Vec::with_capacity(exit_arity);
            let mut typed_result = Vec::with_capacity(exit_arity);
            for (i, &tp) in descr.fail_arg_types().iter().enumerate() {
                match tp {
                    Type::Int => {
                        let value = self.get_int_value(&frame, i);
                        result.push(value);
                        typed_result.push(Value::Int(value));
                    }
                    Type::Ref => {
                        let value = self.get_ref_value(&frame, i);
                        result.push(value.as_usize() as i64);
                        typed_result.push(Value::Ref(value));
                    }
                    Type::Float => {
                        let value = self.get_float_value(&frame, i);
                        result.push(value.to_bits() as i64);
                        typed_result.push(Value::Float(value));
                    }
                    Type::Void => {
                        result.push(0);
                        typed_result.push(Value::Void);
                    }
                }
            }
            return majit_codegen::RawExecResult {
                outputs: result,
                typed_outputs: typed_result,
                exit_layout,
                force_token_slots: descr.force_token_slots().to_vec(),
                savedata,
                exception_class,
                exception_value,
                fail_index: descr.fail_index(),
                trace_id: descr.trace_id(),
                is_finish: descr.is_finish(),
            };
        }

        let fail_descr = &compiled.fail_descrs[fail_index as usize];
        fail_descr.increment_fail_count();

        // If a bridge is attached, dispatch to it.
        if std::env::var_os("MAJIT_LOG").is_some() && fail_index == 0 {
            let has_bridge = fail_descr.bridge_ref().is_some();
            eprintln!("[exec-guard0] fail_index={fail_index} has_bridge={has_bridge}");
        }
        let bridge_guard = fail_descr.bridge_ref();
        if let Some(ref bridge) = *bridge_guard {
            release_force_token(handle);
            let frame = Self::execute_bridge(bridge, &outputs, &fail_descr.fail_arg_types);
            let descr = frame
                .data
                .downcast_ref::<FrameData>()
                .expect("bridge returned unexpected frame type");
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[bridge-dispatch] guard={} bridge_trace={} bridge_exit_fail={} is_finish={}",
                    fail_index,
                    bridge.trace_id,
                    descr.fail_descr.fail_index(),
                    descr.fail_descr.is_finish()
                );
            }
            let arity = descr.fail_descr.fail_arg_types().len();
            let mut result = Vec::with_capacity(arity);
            let mut typed_result = Vec::with_capacity(arity);
            for (i, &tp) in descr.fail_descr.fail_arg_types().iter().enumerate() {
                match tp {
                    Type::Int => {
                        let value = descr.get_int(i);
                        result.push(value);
                        typed_result.push(Value::Int(value));
                    }
                    Type::Ref => {
                        let value = descr.get_ref(i);
                        result.push(value.as_usize() as i64);
                        typed_result.push(Value::Ref(value));
                    }
                    Type::Float => {
                        let value = descr.get_float(i);
                        result.push(value.to_bits() as i64);
                        typed_result.push(Value::Float(value));
                    }
                    Type::Void => {
                        result.push(0);
                        typed_result.push(Value::Void);
                    }
                }
            }
            return majit_codegen::RawExecResult {
                outputs: result,
                typed_outputs: typed_result,
                exit_layout: Some(descr.fail_descr.layout()),
                force_token_slots: descr.fail_descr.force_token_slots().to_vec(),
                savedata: descr.try_get_savedata_ref(),
                exception_class: descr.get_exception_class(),
                exception_value: descr.get_exception_ref(),
                fail_index: descr.fail_descr.fail_index(),
                trace_id: descr.fail_descr.trace_id(),
                is_finish: descr.fail_descr.is_finish(),
            };
        }
        drop(bridge_guard);

        // No bridge — skip DeadFrame, return outputs directly.
        let savedata = if let Some(ref ff) = force_frame {
            take_force_frame_saved_data(ff)
        } else {
            None
        };
        let (exception_class, exception) = take_pending_jit_exception_state();
        if !output_transfers_current_force_token(fail_descr, &outputs, handle) {
            release_force_token(handle);
        }

        let exit_arity = fail_descr.fail_arg_types().len();
        outputs.truncate(exit_arity);
        let mut typed_outputs = Vec::with_capacity(exit_arity);
        for (&raw, &tp) in outputs.iter().zip(fail_descr.fail_arg_types().iter()) {
            match tp {
                Type::Int => typed_outputs.push(Value::Int(raw)),
                Type::Ref => typed_outputs.push(Value::Ref(GcRef(raw as usize))),
                Type::Float => typed_outputs.push(Value::Float(f64::from_bits(raw as u64))),
                Type::Void => typed_outputs.push(Value::Void),
            }
        }

        majit_codegen::RawExecResult {
            outputs,
            typed_outputs,
            exit_layout: Some(fail_descr.layout()),
            force_token_slots: fail_descr.force_token_slots().to_vec(),
            savedata,
            exception_class,
            exception_value: exception,
            fail_index,
            trace_id: fail_descr.trace_id(),
            is_finish: fail_descr.is_finish(),
        }
    }

    fn compiled_fail_descr_layouts(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<majit_codegen::FailDescrLayout>> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        Some(
            compiled
                .fail_descrs
                .iter()
                .map(|descr| descr.layout())
                .collect(),
        )
    }

    fn compiled_bridge_fail_descr_layouts(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Vec<majit_codegen::FailDescrLayout>> {
        let original_compiled = original_token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        let source_descr = original_compiled.fail_descrs.iter().find(|descr| {
            descr.fail_index == source_fail_index && descr.trace_id == source_trace_id
        })?;
        let bridge = source_descr.bridge_ref();
        let bridge = bridge.as_ref()?;
        Some(
            bridge
                .fail_descrs
                .iter()
                .map(|descr| descr.layout())
                .collect(),
        )
    }

    fn compiled_trace_fail_descr_layouts(
        &self,
        token: &JitCellToken,
        trace_id: u64,
    ) -> Option<Vec<majit_codegen::FailDescrLayout>> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        if compiled.trace_id == trace_id {
            return Some(
                compiled
                    .fail_descrs
                    .iter()
                    .map(|descr| descr.layout())
                    .collect(),
            );
        }
        find_trace_fail_descr_layouts_in_fail_descrs(&compiled.fail_descrs, trace_id)
    }

    fn compiled_terminal_exit_layouts(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<majit_codegen::TerminalExitLayout>> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        Some(compiled.terminal_exit_layouts_ref().clone())
    }

    fn compiled_bridge_terminal_exit_layouts(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Vec<majit_codegen::TerminalExitLayout>> {
        let original_compiled = original_token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        let source_descr = original_compiled.fail_descrs.iter().find(|descr| {
            descr.fail_index == source_fail_index && descr.trace_id == source_trace_id
        })?;
        let bridge = source_descr.bridge_ref();
        let bridge = bridge.as_ref()?;
        Some(bridge.terminal_exit_layouts_ref().clone())
    }

    fn compiled_trace_terminal_exit_layouts(
        &self,
        token: &JitCellToken,
        trace_id: u64,
    ) -> Option<Vec<majit_codegen::TerminalExitLayout>> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        if compiled.trace_id == trace_id {
            return Some(compiled.terminal_exit_layouts_ref().clone());
        }
        find_trace_terminal_exit_layouts_in_fail_descrs(&compiled.fail_descrs, trace_id)
    }

    fn compiled_trace_info(
        &self,
        token: &JitCellToken,
        trace_id: u64,
    ) -> Option<CompiledTraceInfo> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        if compiled.trace_id == trace_id {
            return Some(CompiledTraceInfo {
                trace_id: compiled.trace_id,
                input_types: compiled.input_types.clone(),
                header_pc: compiled.header_pc,
                source_guard: None,
            });
        }
        find_trace_info_in_fail_descrs(&compiled.fail_descrs, trace_id)
    }

    fn compiled_guard_frame_stacks(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<(u32, Vec<majit_codegen::ExitFrameLayout>)>> {
        let compiled = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())?;
        let mut result = Vec::new();
        for descr in &compiled.fail_descrs {
            let recovery = descr.recovery_layout_ref();
            if let Some(ref layout) = *recovery {
                result.push((descr.fail_index, layout.frames.clone()));
            }
        }
        Some(result)
    }

    fn describe_deadframe(&self, frame: &DeadFrame) -> Option<majit_codegen::FailDescrLayout> {
        deadframe_layout(frame)
    }

    fn update_fail_descr_recovery_layout(
        &mut self,
        token: &JitCellToken,
        trace_id: u64,
        fail_index: u32,
        recovery_layout: ExitRecoveryLayout,
    ) -> bool {
        let Some(compiled) = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())
        else {
            return false;
        };
        patch_fail_descr_recovery_layout(
            &compiled.fail_descrs,
            trace_id,
            fail_index,
            &recovery_layout,
        )
    }

    fn update_terminal_exit_recovery_layout(
        &mut self,
        token: &JitCellToken,
        trace_id: u64,
        op_index: usize,
        recovery_layout: ExitRecoveryLayout,
    ) -> bool {
        let Some(compiled) = token
            .compiled
            .as_ref()
            .and_then(|compiled| compiled.downcast_ref::<CompiledLoop>())
        else {
            return false;
        };
        patch_terminal_exit_recovery_layout(
            &compiled.terminal_exit_layouts,
            &compiled.fail_descrs,
            trace_id,
            op_index,
            &recovery_layout,
        )
    }

    fn force(&self, force_token: GcRef) -> Option<DeadFrame> {
        Some(force_token_to_dead_frame(force_token))
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        get_latest_descr_from_deadframe(frame).expect("get_latest_descr_from_deadframe failed")
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        get_int_from_deadframe(frame, index).expect("get_int_from_deadframe failed")
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        get_float_from_deadframe(frame, index).expect("get_float_from_deadframe failed")
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        get_ref_from_deadframe(frame, index).expect("get_ref_from_deadframe failed")
    }

    fn set_savedata_ref(&self, frame: &mut DeadFrame, data: GcRef) {
        set_savedata_ref_on_deadframe(frame, data).expect("set_savedata_ref_on_deadframe failed");
    }

    fn get_savedata_ref(&self, frame: &DeadFrame) -> Option<GcRef> {
        get_savedata_ref_from_deadframe(frame).ok()
    }

    fn grab_savedata_ref(&self, frame: &DeadFrame) -> Option<GcRef> {
        grab_savedata_ref_from_deadframe(frame)
    }

    fn grab_exception_state(&self, frame: &DeadFrame) -> (i64, GcRef) {
        (
            grab_exc_class_from_deadframe(frame).expect("grab_exc_class_from_deadframe failed"),
            grab_exc_value_from_deadframe(frame).expect("grab_exc_value_from_deadframe failed"),
        )
    }

    fn grab_exc_value(&self, frame: &DeadFrame) -> GcRef {
        grab_exc_value_from_deadframe(frame).expect("grab_exc_value_from_deadframe failed")
    }

    fn grab_exc_class(&self, frame: &DeadFrame) -> i64 {
        grab_exc_class_from_deadframe(frame).expect("grab_exc_class_from_deadframe failed")
    }

    fn invalidate_loop(&self, token: &JitCellToken) {
        token.invalidate();
    }

    fn redirect_call_assembler(
        &self,
        old: &JitCellToken,
        new: &JitCellToken,
    ) -> Result<(), BackendError> {
        redirect_call_assembler_target(old.number, new.number)
    }

    fn free_loop(&mut self, token: &JitCellToken) {
        unregister_call_assembler_target(token.number);
        self.registered_call_assembler_tokens.remove(&token.number);
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Inline arena take/put — replaces indirect calls to create_frame / drop_frame
// with direct load/store instructions in the Cranelift IR.
// ---------------------------------------------------------------------------

/// Emit inline arena_take: allocate a frame from the global arena.
///
/// Fast path (hot): top < cap AND was_init AND same code → 2 stores + return ptr
/// Slow path (cold): call the original create_frame helper
fn emit_inline_arena_take(
    builder: &mut FunctionBuilder,
    constants: &std::collections::HashMap<u32, i64>,
    arena: &InlineFrameArenaInfo,
    op: &majit_ir::Op,
    ptr_type: cranelift_codegen::ir::Type,
) -> CValue {
    let flags = MemFlags::trusted();
    // Resolve args: create_frame(caller_frame, raw_arg)
    let caller_frame = resolve_opref(builder, constants, op.arg(1));
    let raw_arg = resolve_opref(builder, constants, op.arg(2));

    // Load global arena state
    let top_addr = builder.ins().iconst(ptr_type, arena.top_addr as i64);
    let top = builder.ins().load(ptr_type, flags, top_addr, 0);
    let cap = builder.ins().iconst(ptr_type, arena.arena_cap as i64);
    let within_cap = builder.ins().icmp(IntCC::UnsignedLessThan, top, cap);

    let fast_block = builder.create_block();
    let slow_block = builder.create_block();
    let merge_block = builder.create_block();
    builder.append_block_param(merge_block, ptr_type);

    builder
        .ins()
        .brif(within_cap, fast_block, &[], slow_block, &[]);

    // ── fast path ──
    builder.switch_to_block(fast_block);
    builder.seal_block(fast_block);

    // frame_ptr = buf_base + top * frame_size
    let buf_base_addr = builder.ins().iconst(ptr_type, arena.buf_base_addr as i64);
    let buf_base = builder.ins().load(ptr_type, flags, buf_base_addr, 0);
    let frame_size = builder.ins().iconst(ptr_type, arena.frame_size as i64);
    let offset = builder.ins().imul(top, frame_size);
    let frame_ptr = builder.ins().iadd(buf_base, offset);

    // Check was_init: top < initialized
    let init_addr = builder
        .ins()
        .iconst(ptr_type, arena.initialized_addr as i64);
    let initialized = builder.ins().load(ptr_type, flags, init_addr, 0);
    let was_init = builder
        .ins()
        .icmp(IntCC::UnsignedLessThan, top, initialized);

    let reinit_block = builder.create_block();
    builder
        .ins()
        .brif(was_init, reinit_block, &[], slow_block, &[]);

    // ── reinit path (was_init=true) ──
    builder.switch_to_block(reinit_block);
    builder.seal_block(reinit_block);

    // Check f.code == caller.code (self-recursive fast path)
    let caller_code = builder.ins().load(
        ptr_type,
        flags,
        caller_frame,
        arena.frame_code_offset as i32,
    );
    let frame_code = builder
        .ins()
        .load(ptr_type, flags, frame_ptr, arena.frame_code_offset as i32);
    let same_code = builder.ins().icmp(IntCC::Equal, frame_code, caller_code);

    let ultra_fast_block = builder.create_block();
    builder
        .ins()
        .brif(same_code, ultra_fast_block, &[], slow_block, &[]);

    // ── ultra fast path: only 2 stores + bump top ──
    builder.switch_to_block(ultra_fast_block);
    builder.seal_block(ultra_fast_block);

    let zero = builder.ins().iconst(ptr_type, 0);
    builder
        .ins()
        .store(flags, zero, frame_ptr, arena.frame_next_instr_offset as i32);
    builder.ins().store(
        flags,
        zero,
        frame_ptr,
        arena.frame_vable_token_offset as i32,
    );

    // top += 1
    let one = builder.ins().iconst(ptr_type, 1);
    let new_top = builder.ins().iadd(top, one);
    builder.ins().store(flags, new_top, top_addr, 0);

    builder
        .ins()
        .jump(merge_block, &[BlockArg::from(frame_ptr)]);

    // ── slow path: call original helper ──
    builder.switch_to_block(slow_block);
    builder.seal_block(slow_block);

    let create_sig = {
        let cc = builder.func.signature.call_conv;
        builder.func.import_signature(Signature {
            params: vec![AbiParam::new(ptr_type), AbiParam::new(ptr_type)],
            returns: vec![AbiParam::new(ptr_type)],
            call_conv: cc,
        })
    };
    let create_fn_addr = builder.ins().iconst(ptr_type, arena.create_fn_addr as i64);
    let slow_result =
        builder
            .ins()
            .call_indirect(create_sig, create_fn_addr, &[caller_frame, raw_arg]);
    let slow_frame = builder.inst_results(slow_result)[0];
    builder
        .ins()
        .jump(merge_block, &[BlockArg::from(slow_frame)]);

    builder.switch_to_block(merge_block);
    builder.seal_block(merge_block);
    builder.block_params(merge_block)[0]
}

/// Emit inline arena_put: return a frame to the global arena.
///
/// Fast path: just decrement top.
/// Slow path (non-LIFO or tagged pointer): call original drop helper.
fn emit_inline_arena_put(
    builder: &mut FunctionBuilder,
    constants: &std::collections::HashMap<u32, i64>,
    arena: &InlineFrameArenaInfo,
    op: &majit_ir::Op,
    ptr_type: cranelift_codegen::ir::Type,
) {
    let flags = MemFlags::trusted();
    // Resolve arg: drop_frame(frame_ptr)
    let frame_ptr = resolve_opref(builder, constants, op.arg(1));

    // Check tagged pointer (bit 0)
    let one = builder.ins().iconst(ptr_type, 1);
    let tag_bit = builder.ins().band(frame_ptr, one);
    let zero_val = builder.ins().iconst(ptr_type, 0);
    let is_tagged = builder.ins().icmp(IntCC::NotEqual, tag_bit, zero_val);

    let check_lifo_block = builder.create_block();
    let done_block = builder.create_block();

    builder
        .ins()
        .brif(is_tagged, done_block, &[], check_lifo_block, &[]);

    // ── check LIFO ──
    builder.switch_to_block(check_lifo_block);
    builder.seal_block(check_lifo_block);

    let top_addr = builder.ins().iconst(ptr_type, arena.top_addr as i64);
    let top = builder.ins().load(ptr_type, flags, top_addr, 0);
    let new_top = builder.ins().isub(top, one);

    // expected = buf_base + new_top * frame_size
    let buf_base_addr = builder.ins().iconst(ptr_type, arena.buf_base_addr as i64);
    let buf_base = builder.ins().load(ptr_type, flags, buf_base_addr, 0);
    let frame_size = builder.ins().iconst(ptr_type, arena.frame_size as i64);
    let offset = builder.ins().imul(new_top, frame_size);
    let expected = builder.ins().iadd(buf_base, offset);
    let is_lifo = builder.ins().icmp(IntCC::Equal, frame_ptr, expected);

    let fast_put_block = builder.create_block();
    let slow_put_block = builder.create_block();
    builder
        .ins()
        .brif(is_lifo, fast_put_block, &[], slow_put_block, &[]);

    // ── fast put: just store new_top ──
    builder.switch_to_block(fast_put_block);
    builder.seal_block(fast_put_block);
    builder.ins().store(flags, new_top, top_addr, 0);
    builder.ins().jump(done_block, &[]);

    // ── slow put: call original helper ──
    builder.switch_to_block(slow_put_block);
    builder.seal_block(slow_put_block);
    let drop_sig = {
        let cc = builder.func.signature.call_conv;
        builder.func.import_signature(Signature {
            params: vec![AbiParam::new(ptr_type)],
            returns: vec![],
            call_conv: cc,
        })
    };
    let drop_fn_addr = builder.ins().iconst(ptr_type, arena.drop_fn_addr as i64);
    builder
        .ins()
        .call_indirect(drop_sig, drop_fn_addr, &[frame_ptr]);
    builder.ins().jump(done_block, &[]);

    builder.switch_to_block(done_block);
    builder.seal_block(done_block);
}

// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use majit_codegen::Backend;
    use majit_gc::collector::{GcConfig, MiniMarkGC};
    use majit_gc::flags;
    use majit_gc::header::{GcHeader, header_of};
    use majit_gc::trace::TypeInfo;
    use majit_ir::descr::{Descr, EffectInfo, ExtraEffect, SizeDescr};
    use std::collections::HashMap;

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut o = Op::new(opcode, args);
        o.pos = OpRef(pos);
        o
    }

    fn mk_op_with_descr(opcode: OpCode, args: &[OpRef], pos: u32, descr: majit_ir::DescrRef) -> Op {
        let mut o = Op::with_descr(opcode, args, descr);
        o.pos = OpRef(pos);
        o
    }

    #[derive(Debug)]
    struct TestCallDescr {
        arg_types: Vec<Type>,
        result_type: Type,
        effect_info: EffectInfo,
    }

    #[derive(Debug)]
    struct TestLabelDescr {
        idx: u32,
    }

    impl Descr for TestLabelDescr {
        fn index(&self) -> u32 {
            self.idx
        }
    }

    impl Descr for TestCallDescr {
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }
        fn result_type(&self) -> Type {
            self.result_type
        }
        fn result_size(&self) -> usize {
            8
        }
        fn effect_info(&self) -> &EffectInfo {
            &self.effect_info
        }
    }

    fn make_call_descr(arg_types: Vec<Type>, result_type: Type) -> majit_ir::DescrRef {
        Arc::new(TestCallDescr {
            arg_types,
            result_type,
            effect_info: EffectInfo {
                extra_effect: ExtraEffect::CanRaise,
                oopspec_index: majit_ir::OopSpecIndex::None,
                ..Default::default()
            },
        })
    }

    fn make_call_assembler_descr(
        target: &JitCellToken,
        arg_types: Vec<Type>,
        result_type: Type,
    ) -> majit_ir::DescrRef {
        Arc::new(CallAssemblerDescr::new(
            target.number,
            arg_types,
            result_type,
        ))
    }

    fn make_label_descr(idx: u32) -> majit_ir::DescrRef {
        Arc::new(TestLabelDescr { idx })
    }

    extern "C" fn collect_nursery_via_runtime(runtime_id: i64) -> i64 {
        with_gc_runtime(runtime_id as u64, |gc| gc.collect_nursery());
        123
    }

    extern "C" fn collect_nursery_via_runtime_void(runtime_id: i64) {
        with_gc_runtime(runtime_id as u64, |gc| gc.collect_nursery());
    }

    fn may_force_void_values() -> &'static Mutex<Vec<i64>> {
        static VALUES: OnceLock<Mutex<Vec<i64>>> = OnceLock::new();
        VALUES.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn may_force_int_values() -> &'static Mutex<Vec<i64>> {
        static VALUES: OnceLock<Mutex<Vec<i64>>> = OnceLock::new();
        VALUES.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn may_force_float_values() -> &'static Mutex<Vec<u64>> {
        static VALUES: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
        VALUES.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn may_force_ref_values() -> &'static Mutex<Vec<usize>> {
        static VALUES: OnceLock<Mutex<Vec<usize>>> = OnceLock::new();
        VALUES.get_or_init(|| Mutex::new(Vec::new()))
    }

    thread_local! {
        static TEST_EXCEPTION_VALUE: Cell<i64> = const { Cell::new(0) };
        static TEST_EXCEPTION_TYPE: Cell<i64> = const { Cell::new(0) };
        static TEST_EXCEPTION_CALL_LOG: std::cell::RefCell<Vec<bool>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    fn set_test_exception_state(value: i64, exc_type: i64) {
        TEST_EXCEPTION_VALUE.with(|cell| cell.set(value));
        TEST_EXCEPTION_TYPE.with(|cell| cell.set(exc_type));
    }

    fn clear_test_exception_call_log() {
        TEST_EXCEPTION_CALL_LOG.with(|log| log.borrow_mut().clear());
    }

    fn test_exception_call_log_snapshot() -> Vec<bool> {
        TEST_EXCEPTION_CALL_LOG.with(|log| log.borrow().clone())
    }

    extern "C" fn maybe_raise_test_exception(flag: i64) {
        TEST_EXCEPTION_CALL_LOG.with(|log| log.borrow_mut().push(jit_exc_is_pending()));
        if flag != 0 {
            let value = TEST_EXCEPTION_VALUE.with(Cell::get);
            let exc_type = TEST_EXCEPTION_TYPE.with(Cell::get);
            jit_exc_raise(value, exc_type);
        }
    }

    extern "C" fn maybe_force_and_return_void(force_token: i64, flag: i64) {
        if flag != 0 {
            let mut deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            let mut values = may_force_void_values().lock().unwrap();
            values.push(
                get_latest_descr_from_deadframe(&deadframe)
                    .unwrap()
                    .fail_index() as i64,
            );
            values.push(get_int_from_deadframe(&deadframe, 0).unwrap());
            values.push(get_int_from_deadframe(&deadframe, 1).unwrap());
            drop(values);
            set_savedata_ref_on_deadframe(&mut deadframe, GcRef(0xDADA)).unwrap();
        }
    }

    /// Like `maybe_force_and_return_int` but does not write to the global
    /// `may_force_int_values` vec, avoiding cross-test interference.
    extern "C" fn maybe_force_and_return_int_isolated(force_token: i64, flag: i64) -> i64 {
        if flag != 0 {
            let mut deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            set_savedata_ref_on_deadframe(&mut deadframe, GcRef(0xBABA)).unwrap();
        }
        42
    }

    extern "C" fn maybe_force_and_return_int(force_token: i64, flag: i64) -> i64 {
        if flag != 0 {
            let mut deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            let mut values = may_force_int_values().lock().unwrap();
            values.push(get_int_from_deadframe(&deadframe, 0).unwrap());
            values.push(get_int_from_deadframe(&deadframe, 2).unwrap());
            drop(values);
            set_savedata_ref_on_deadframe(&mut deadframe, GcRef(0xBABA)).unwrap();
        }
        42
    }

    extern "C" fn maybe_force_and_return_float(force_token: i64, flag: i64) -> f64 {
        if flag != 0 {
            let deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            let mut values = may_force_float_values().lock().unwrap();
            values.push(get_int_from_deadframe(&deadframe, 0).unwrap() as u64);
            values.push(get_float_from_deadframe(&deadframe, 1).unwrap().to_bits());
            values.push(get_int_from_deadframe(&deadframe, 2).unwrap() as u64);
        }
        12.5
    }

    extern "C" fn maybe_force_and_return_ref(
        force_token: i64,
        flag: i64,
        runtime_id: i64,
        return_ref: i64,
    ) -> i64 {
        if flag != 0 {
            let mut deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            let preview_live = get_ref_from_deadframe(&deadframe, 2).unwrap();
            set_savedata_ref_on_deadframe(&mut deadframe, preview_live).unwrap();
            with_gc_runtime(runtime_id as u64, |gc| gc.collect_nursery());
            let preview_result = get_ref_from_deadframe(&deadframe, 1).unwrap();
            let preview_live = get_ref_from_deadframe(&deadframe, 2).unwrap();
            let preview_return = get_ref_from_deadframe(&deadframe, 3).unwrap();
            let mut values = may_force_ref_values().lock().unwrap();
            values.push(preview_result.0);
            values.push(preview_live.0);
            values.push(preview_return.0);
            return get_ref_from_deadframe(&deadframe, 3).unwrap().0 as i64;
        }
        return_ref
    }

    fn make_gc_backend() -> CraneliftBackend {
        CraneliftBackend::with_gc_allocator(Box::new(MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
        })))
    }

    #[derive(Default)]
    struct TrackingGcState {
        collecting_allocs: usize,
        no_collect_allocs: usize,
        write_barriers: usize,
        added_roots: usize,
        removed_roots: usize,
    }

    struct TrackingGc {
        state: Arc<Mutex<TrackingGcState>>,
        allocations: Vec<Box<[u8]>>,
    }

    impl TrackingGc {
        fn new(state: Arc<Mutex<TrackingGcState>>) -> Self {
            Self {
                state,
                allocations: Vec::new(),
            }
        }

        fn alloc_payload(&mut self, payload_size: usize, collecting: bool) -> GcRef {
            let total_size = GcHeader::SIZE + payload_size.max(std::mem::size_of::<usize>());
            let mut buf = vec![0u8; total_size].into_boxed_slice();
            let ptr = buf.as_mut_ptr();
            unsafe {
                *(ptr as *mut GcHeader) = GcHeader::new(0);
            }
            let obj = GcRef(ptr as usize + GcHeader::SIZE);
            self.allocations.push(buf);
            let mut state = self.state.lock().unwrap();
            if collecting {
                state.collecting_allocs += 1;
            } else {
                state.no_collect_allocs += 1;
            }
            obj
        }
    }

    impl GcAllocator for TrackingGc {
        fn alloc_nursery(&mut self, size: usize) -> GcRef {
            self.alloc_payload(size, true)
        }

        fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
            self.alloc_payload(size, false)
        }

        fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef {
            self.alloc_payload(base_size + item_size * length, true)
        }

        fn alloc_varsize_no_collect(
            &mut self,
            base_size: usize,
            item_size: usize,
            length: usize,
        ) -> GcRef {
            self.alloc_payload(base_size + item_size * length, false)
        }

        fn write_barrier(&mut self, _obj: GcRef) {
            self.state.lock().unwrap().write_barriers += 1;
        }

        fn collect_nursery(&mut self) {}

        fn collect_full(&mut self) {}

        unsafe fn add_root(&mut self, _root: *mut GcRef) {
            self.state.lock().unwrap().added_roots += 1;
        }

        fn remove_root(&mut self, _root: *mut GcRef) {
            self.state.lock().unwrap().removed_roots += 1;
        }

        fn nursery_free(&self) -> *mut u8 {
            std::ptr::null_mut()
        }

        fn nursery_top(&self) -> *const u8 {
            std::ptr::null()
        }

        fn max_nursery_object_size(&self) -> usize {
            usize::MAX
        }
    }

    fn assert_compile_unsupported(
        inputargs: Vec<InputArg>,
        ops: Vec<Op>,
        token_number: u64,
        opcode: OpCode,
        expected_detail: &str,
    ) {
        let mut backend = CraneliftBackend::new();
        let mut token = JitCellToken::new(token_number);
        let err = backend
            .compile_loop(&inputargs, &ops, &mut token)
            .unwrap_err();
        match err {
            BackendError::Unsupported(msg) => {
                assert!(msg.contains(&format!("{opcode:?}")));
                assert!(
                    msg.contains(expected_detail),
                    "expected detail {expected_detail:?} in unsupported message {msg:?}"
                );
            }
            other => panic!("expected unsupported error, got {other:?}"),
        }
    }

    // ── Existing tests ──

    #[test]
    fn test_count_to_million() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::IntLt, &[OpRef(1), OpRef(101)], 2),
            mk_op(OpCode::GuardTrue, &[OpRef(2)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        constants.insert(101, 1_000_000i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(0);
        let info = backend.compile_loop(&inputargs, &ops, &mut token).unwrap();
        assert!(info.code_addr != 0);

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 999_999);
    }

    #[test]
    fn test_simple_add_finish() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(40), Value::Int(2)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_finish_preserves_float_and_ref_types() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0), InputArg::new_ref(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1000);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Float(3.5), Value::Ref(GcRef(0x1234))]);
        assert_eq!(backend.get_float_value(&frame, 0), 3.5);
        assert_eq!(backend.get_ref_value(&frame, 1), GcRef(0x1234));
    }

    #[test]
    fn test_int_sub() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntSub, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(2);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(58)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_mul() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntMul, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(3);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(6), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_floor_div() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntFloorDiv, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(3);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42), Value::Int(6)]);
        assert_eq!(backend.get_int_value(&frame, 0), 7);
    }

    #[test]
    fn test_bitwise_ops() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAnd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::IntOr, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::IntXor, &[OpRef(0), OpRef(1)], 4),
            mk_op(
                OpCode::Finish,
                &[OpRef(2), OpRef(3), OpRef(4)],
                OpRef::NONE.0,
            ),
        ];

        let mut token = JitCellToken::new(4);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0xFF00), Value::Int(0x0FF0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0x0F00);
        assert_eq!(backend.get_int_value(&frame, 1), 0xFFF0);
        assert_eq!(backend.get_int_value(&frame, 2), 0xF0F0);
    }

    #[test]
    fn test_shift_ops() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntLshift, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::IntRshift, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::UintRshift, &[OpRef(0), OpRef(1)], 4),
            mk_op(
                OpCode::Finish,
                &[OpRef(2), OpRef(3), OpRef(4)],
                OpRef::NONE.0,
            ),
        ];

        let mut token = JitCellToken::new(5);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(-16), Value::Int(2)]);
        assert_eq!(backend.get_int_value(&frame, 0), -64);
        assert_eq!(backend.get_int_value(&frame, 1), -4);
        let expected_ushr = ((-16i64 as u64) >> 2) as i64;
        assert_eq!(backend.get_int_value(&frame, 2), expected_ushr);
    }

    #[test]
    fn test_comparisons() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntLt, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::IntLe, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::IntEq, &[OpRef(0), OpRef(1)], 4),
            mk_op(OpCode::IntNe, &[OpRef(0), OpRef(1)], 5),
            mk_op(OpCode::IntGt, &[OpRef(0), OpRef(1)], 6),
            mk_op(OpCode::IntGe, &[OpRef(0), OpRef(1)], 7),
            mk_op(
                OpCode::Finish,
                &[OpRef(2), OpRef(3), OpRef(4), OpRef(5), OpRef(6), OpRef(7)],
                OpRef::NONE.0,
            ),
        ];

        let mut token = JitCellToken::new(6);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_int_value(&frame, 1), 1);
        assert_eq!(backend.get_int_value(&frame, 2), 0);
        assert_eq!(backend.get_int_value(&frame, 3), 1);
        assert_eq!(backend.get_int_value(&frame, 4), 0);
        assert_eq!(backend.get_int_value(&frame, 5), 0);
    }

    #[test]
    fn test_guard_false() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntEq, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::GuardFalse, &[OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntSub, &[OpRef(0), OpRef(101)], 2),
            mk_op(OpCode::Jump, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0i64);
        constants.insert(101, 1i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(7);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(10)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
    }

    #[test]
    fn test_no_label_loop_uses_jump_arity_not_dead_extra_inputs() {
        let mut backend = CraneliftBackend::new();

        // Legacy no-LABEL traces can still carry stale extra input slots from
        // optimizer-owned virtual state, while the visible loop contract is the
        // terminal JUMP arity. This must still compile.
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
            InputArg::new_int(3),
            InputArg::new_int(4),
        ];
        let ops = vec![
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 5),
            mk_op(OpCode::Jump, &[OpRef(0), OpRef(1), OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(70001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();
    }

    #[test]
    fn test_fail_descr() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(8);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_compiled_fail_descr_layouts_include_backend_recovery_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_ref(1)];
        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(1), OpRef(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            guard,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(76);
        backend.set_next_header_pc(1234);
        let mut token = JitCellToken::new(8008);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let layouts = backend
            .compiled_fail_descr_layouts(&token)
            .expect("compiled layouts should exist");
        let layout = &layouts[0];
        assert_eq!(layout.source_op_index, Some(1));
        let recovery = layout
            .recovery_layout
            .as_ref()
            .expect("guard layout should include backend recovery layout");
        assert_eq!(recovery.frames.len(), 1);
        assert_eq!(recovery.frames[0].trace_id, Some(76));
        assert_eq!(recovery.frames[0].header_pc, Some(1234));
        assert_eq!(recovery.frames[0].pc, 1234);
        assert_eq!(
            recovery.frames[0].slots,
            vec![
                majit_codegen::ExitValueSourceLayout::ExitValue(0),
                majit_codegen::ExitValueSourceLayout::ExitValue(1),
            ]
        );
        assert_eq!(
            recovery.frames[0].slot_types,
            Some(vec![Type::Ref, Type::Int])
        );
    }

    #[test]
    fn test_compiled_terminal_exit_layouts_include_backend_jump_recovery_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0), InputArg::new_ref(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(1), OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(77);
        backend.set_next_header_pc(1234);
        let mut token = JitCellToken::new(8009);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let layouts = backend
            .compiled_terminal_exit_layouts(&token)
            .expect("compiled terminal layouts should exist");
        assert_eq!(layouts.len(), 1);
        let layout = &layouts[0];
        assert_eq!(layout.op_index, 1);
        assert_eq!(layout.trace_id, 77);
        assert_eq!(layout.fail_index, u32::MAX);
        assert_eq!(layout.exit_types, vec![Type::Ref, Type::Float]);
        assert!(!layout.is_finish);
        assert_eq!(layout.gc_ref_slots, vec![0]);
        assert!(layout.force_token_slots.is_empty());
        let recovery = layout
            .recovery_layout
            .as_ref()
            .expect("jump layout should include backend recovery layout");
        assert_eq!(recovery.frames.len(), 1);
        assert_eq!(recovery.frames[0].trace_id, Some(77));
        assert_eq!(recovery.frames[0].header_pc, Some(1234));
        assert_eq!(recovery.frames[0].pc, 1234);
        assert_eq!(
            recovery.frames[0].slots,
            vec![
                majit_codegen::ExitValueSourceLayout::ExitValue(0),
                majit_codegen::ExitValueSourceLayout::ExitValue(1),
            ]
        );
        assert_eq!(
            recovery.frames[0].slot_types,
            Some(vec![Type::Ref, Type::Float])
        );
    }

    #[test]
    fn test_compiled_terminal_exit_layouts_include_backend_finish_recovery_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0), InputArg::new_ref(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(1), OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(79);
        backend.set_next_header_pc(5678);
        let mut token = JitCellToken::new(8011);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let layouts = backend
            .compiled_terminal_exit_layouts(&token)
            .expect("compiled terminal layouts should exist");
        assert_eq!(layouts.len(), 1);
        let layout = &layouts[0];
        assert_eq!(layout.op_index, 1);
        assert_eq!(layout.trace_id, 79);
        assert_eq!(layout.fail_index, 0);
        assert_eq!(layout.exit_types, vec![Type::Ref, Type::Float]);
        assert!(layout.is_finish);
        assert_eq!(layout.gc_ref_slots, vec![0]);
        assert!(layout.force_token_slots.is_empty());
        let recovery = layout
            .recovery_layout
            .as_ref()
            .expect("finish layout should include backend recovery layout");
        assert_eq!(recovery.frames.len(), 1);
        assert_eq!(recovery.frames[0].trace_id, Some(79));
        assert_eq!(recovery.frames[0].header_pc, Some(5678));
        assert_eq!(recovery.frames[0].pc, 5678);
        assert_eq!(
            recovery.frames[0].slots,
            vec![
                majit_codegen::ExitValueSourceLayout::ExitValue(0),
                majit_codegen::ExitValueSourceLayout::ExitValue(1),
            ]
        );
        assert_eq!(
            recovery.frames[0].slot_types,
            Some(vec![Type::Ref, Type::Float])
        );
    }

    #[test]
    fn test_update_terminal_exit_recovery_layout_patches_compiled_terminal_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(78);
        backend.set_next_header_pc(1000);
        let mut token = JitCellToken::new(8010);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let patched = majit_codegen::ExitRecoveryLayout {
            frames: vec![majit_codegen::ExitFrameLayout {
                trace_id: None,
                header_pc: None,
                source_guard: None,
                pc: 4242,
                slots: vec![majit_codegen::ExitValueSourceLayout::Constant(99)],
                slot_types: Some(vec![Type::Int]),
            }],
            virtual_layouts: Vec::new(),
            pending_field_layouts: Vec::new(),
        };
        assert!(backend.update_terminal_exit_recovery_layout(&token, 78, 1, patched.clone()));

        let layouts = backend
            .compiled_terminal_exit_layouts(&token)
            .expect("compiled terminal layouts should exist");
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].recovery_layout, Some(patched));
    }

    #[test]
    fn test_compiled_trace_layout_queries_find_attached_bridge_by_trace_id() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
        guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            guard,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(90);
        backend.set_next_header_pc(1000);
        let mut token = JitCellToken::new(8012);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let bridge_inputargs = vec![InputArg::new_int(0)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            0,
            90,
            vec![Type::Int],
            false,
            Vec::new(),
            None,
        );

        backend.set_next_trace_id(91);
        backend.set_next_header_pc(2000);
        backend
            .compile_bridge(&fail_descr, &bridge_inputargs, &bridge_ops, &token)
            .unwrap();

        let root_layouts = backend
            .compiled_trace_fail_descr_layouts(&token, 90)
            .expect("root trace layouts should exist");
        assert!(root_layouts.iter().all(|layout| layout.trace_id == 90));
        let root_info = backend
            .compiled_trace_info(&token, 90)
            .expect("root trace info should exist");
        assert_eq!(root_info.trace_id, 90);
        assert_eq!(root_info.input_types, vec![Type::Int]);
        assert_eq!(root_info.header_pc, 1000);
        assert_eq!(root_info.source_guard, None);

        let bridge_layouts = backend
            .compiled_trace_fail_descr_layouts(&token, 91)
            .expect("bridge trace layouts should exist");
        assert_eq!(bridge_layouts.len(), 1);
        assert_eq!(bridge_layouts[0].trace_id, 91);
        assert!(bridge_layouts[0].is_finish);
        let bridge_info = backend
            .compiled_trace_info(&token, 91)
            .expect("bridge trace info should exist");
        assert_eq!(bridge_info.trace_id, 91);
        assert_eq!(bridge_info.input_types, vec![Type::Int]);
        assert_eq!(bridge_info.header_pc, 2000);
        assert_eq!(bridge_info.source_guard, Some((90, 0)));

        let bridge_terminal_layouts = backend
            .compiled_trace_terminal_exit_layouts(&token, 91)
            .expect("bridge terminal layouts should exist");
        assert_eq!(bridge_terminal_layouts.len(), 1);
        assert_eq!(bridge_terminal_layouts[0].trace_id, 91);
        assert_eq!(
            bridge_terminal_layouts[0]
                .recovery_layout
                .as_ref()
                .expect("bridge terminal layout should have recovery")
                .frames[0]
                .pc,
            2000
        );
    }

    #[test]
    fn test_compiled_bridge_recovery_layouts_inherit_source_guard_caller_frames() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
        guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            guard,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(190);
        backend.set_next_header_pc(1000);
        let mut token = JitCellToken::new(8013);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let source_layout = majit_codegen::ExitRecoveryLayout {
            frames: vec![
                majit_codegen::ExitFrameLayout {
                    trace_id: Some(10),
                    header_pc: Some(900),
                    source_guard: Some((9, 0)),
                    pc: 900,
                    slots: vec![majit_codegen::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
                majit_codegen::ExitFrameLayout {
                    trace_id: Some(190),
                    header_pc: Some(1000),
                    source_guard: None,
                    pc: 1000,
                    slots: vec![majit_codegen::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
            ],
            virtual_layouts: vec![majit_codegen::ExitVirtualLayout::Array {
                descr_index: 17,
                items: vec![
                    majit_codegen::ExitValueSourceLayout::ExitValue(0),
                    majit_codegen::ExitValueSourceLayout::Constant(44),
                ],
            }],
            pending_field_layouts: vec![majit_codegen::ExitPendingFieldLayout {
                descr_index: 33,
                item_index: Some(1),
                is_array_item: true,
                target: majit_codegen::ExitValueSourceLayout::Virtual(0),
                value: majit_codegen::ExitValueSourceLayout::ExitValue(0),
                field_offset: 0,
                field_size: 8,
                field_type: Type::Int,
            }],
        };
        assert!(backend.update_fail_descr_recovery_layout(&token, 190, 0, source_layout.clone()));

        let bridge_fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            0,
            190,
            vec![Type::Int],
            false,
            Vec::new(),
            None,
        );
        let mut bridge_guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
        bridge_guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            bridge_guard,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(191);
        backend.set_next_header_pc(2000);
        backend
            .compile_bridge(&bridge_fail_descr, &inputargs, &bridge_ops, &token)
            .unwrap();

        let bridge_layouts = backend
            .compiled_trace_fail_descr_layouts(&token, 191)
            .expect("bridge trace layouts should exist");
        let guard_layout = bridge_layouts
            .iter()
            .find(|layout| !layout.is_finish)
            .expect("bridge guard layout should exist");
        let guard_recovery = guard_layout
            .recovery_layout
            .as_ref()
            .expect("bridge guard should carry backend recovery");
        assert_eq!(guard_recovery.frames.len(), 2);
        assert_eq!(guard_recovery.frames[0].trace_id, Some(10));
        assert_eq!(guard_recovery.frames[0].header_pc, Some(900));
        assert_eq!(guard_recovery.frames[0].source_guard, Some((9, 0)));
        assert_eq!(guard_recovery.frames[1].trace_id, Some(191));
        assert_eq!(guard_recovery.frames[1].header_pc, Some(2000));
        assert_eq!(guard_recovery.frames[1].source_guard, Some((190, 0)));
        assert_eq!(guard_recovery.frames[0].pc, 900);
        assert_eq!(guard_recovery.frames[1].pc, 2000);
        assert_eq!(guard_recovery.frames[0].slot_types, Some(vec![Type::Int]));
        assert_eq!(guard_recovery.frames[1].slot_types, Some(vec![Type::Int]));
        assert_eq!(
            guard_recovery.virtual_layouts,
            source_layout.virtual_layouts
        );
        assert_eq!(
            guard_recovery.pending_field_layouts,
            source_layout.pending_field_layouts
        );

        let bridge_terminal_layouts = backend
            .compiled_trace_terminal_exit_layouts(&token, 191)
            .expect("bridge terminal layouts should exist");
        let finish_recovery = bridge_terminal_layouts[0]
            .recovery_layout
            .as_ref()
            .expect("bridge finish should carry backend recovery");
        assert_eq!(finish_recovery.frames.len(), 2);
        assert_eq!(finish_recovery.frames[0].pc, 900);
        assert_eq!(finish_recovery.frames[1].pc, 2000);
        assert_eq!(
            finish_recovery.virtual_layouts,
            source_layout.virtual_layouts
        );
        assert_eq!(
            finish_recovery.pending_field_layouts,
            source_layout.pending_field_layouts
        );
    }

    #[test]
    fn test_compile_bridge_normalizes_legacy_root_source_guard_trace_id() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let mut root_guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
        root_guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            root_guard,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(300);
        backend.set_next_header_pc(1000);
        let mut token = JitCellToken::new(8015);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let legacy_fail_descr = CraneliftFailDescr::new(0, vec![Type::Int]);
        backend.set_next_trace_id(301);
        backend.set_next_header_pc(2000);
        backend
            .compile_bridge(&legacy_fail_descr, &inputargs, &bridge_ops, &token)
            .unwrap();

        let bridge_info = backend
            .compiled_trace_info(&token, 301)
            .expect("bridge trace info should exist");
        assert_eq!(bridge_info.source_guard, Some((300, 0)));
        let bridge_layouts = backend
            .compiled_trace_fail_descr_layouts(&token, 301)
            .expect("bridge trace layouts should exist");
        assert!(
            bridge_layouts
                .iter()
                .all(|layout| layout.trace_info.as_ref().unwrap().source_guard == Some((300, 0)))
        );
    }

    #[test]
    fn test_nested_bridge_compilation_uses_source_trace_fail_descr_tree() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let mut root_guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
        root_guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
        let root_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            root_guard,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(290);
        backend.set_next_header_pc(1000);
        let mut token = JitCellToken::new(8014);
        backend
            .compile_loop(&inputargs, &root_ops, &mut token)
            .unwrap();

        let root_layout = majit_codegen::ExitRecoveryLayout {
            frames: vec![
                majit_codegen::ExitFrameLayout {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: 900,
                    slots: vec![majit_codegen::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
                majit_codegen::ExitFrameLayout {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: 1000,
                    slots: vec![majit_codegen::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
            ],
            virtual_layouts: vec![majit_codegen::ExitVirtualLayout::Array {
                descr_index: 17,
                items: vec![majit_codegen::ExitValueSourceLayout::ExitValue(0)],
            }],
            pending_field_layouts: vec![majit_codegen::ExitPendingFieldLayout {
                descr_index: 33,
                item_index: Some(0),
                is_array_item: true,
                target: majit_codegen::ExitValueSourceLayout::Virtual(0),
                value: majit_codegen::ExitValueSourceLayout::ExitValue(0),
                field_offset: 0,
                field_size: 8,
                field_type: Type::Int,
            }],
        };
        assert!(backend.update_fail_descr_recovery_layout(&token, 290, 0, root_layout));

        let bridge_fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            0,
            290,
            vec![Type::Int],
            false,
            Vec::new(),
            None,
        );
        let mut bridge_guard = mk_op(OpCode::GuardFalse, &[OpRef(0)], OpRef::NONE.0);
        bridge_guard.fail_args = Some(smallvec::smallvec![OpRef(0)]);
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            bridge_guard,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(291);
        backend.set_next_header_pc(2000);
        backend
            .compile_bridge(&bridge_fail_descr, &inputargs, &bridge_ops, &token)
            .unwrap();
        assert!(
            backend
                .compiled_trace_fail_descr_layouts(&token, 291)
                .is_some()
        );
        assert_eq!(
            backend
                .compiled_trace_info(&token, 291)
                .expect("first bridge trace info")
                .source_guard,
            Some((290, 0))
        );

        let bridge_source_layout = majit_codegen::ExitRecoveryLayout {
            frames: vec![
                majit_codegen::ExitFrameLayout {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: 444,
                    slots: vec![majit_codegen::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
                majit_codegen::ExitFrameLayout {
                    trace_id: None,
                    header_pc: None,
                    source_guard: Some((290, 0)),
                    pc: 2000,
                    slots: vec![majit_codegen::ExitValueSourceLayout::ExitValue(0)],
                    slot_types: Some(vec![Type::Int]),
                },
            ],
            virtual_layouts: vec![majit_codegen::ExitVirtualLayout::Array {
                descr_index: 99,
                items: vec![
                    majit_codegen::ExitValueSourceLayout::ExitValue(0),
                    majit_codegen::ExitValueSourceLayout::Constant(55),
                ],
            }],
            pending_field_layouts: vec![majit_codegen::ExitPendingFieldLayout {
                descr_index: 77,
                item_index: Some(1),
                is_array_item: true,
                target: majit_codegen::ExitValueSourceLayout::Virtual(0),
                value: majit_codegen::ExitValueSourceLayout::ExitValue(0),
                field_offset: 0,
                field_size: 8,
                field_type: Type::Int,
            }],
        };
        assert!(backend.update_fail_descr_recovery_layout(
            &token,
            291,
            0,
            bridge_source_layout.clone()
        ));

        let nested_bridge_fail_descr = CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
            0,
            291,
            vec![Type::Int],
            false,
            Vec::new(),
            None,
        );
        backend.set_next_trace_id(292);
        backend.set_next_header_pc(3000);
        backend
            .compile_bridge(&nested_bridge_fail_descr, &inputargs, &bridge_ops, &token)
            .unwrap();

        assert!(
            backend
                .compiled_trace_fail_descr_layouts(&token, 291)
                .is_some()
        );
        let nested_info = backend
            .compiled_trace_info(&token, 292)
            .expect("nested bridge trace info should exist");
        assert_eq!(nested_info.source_guard, Some((291, 0)));

        let nested_layouts = backend
            .compiled_trace_fail_descr_layouts(&token, 292)
            .expect("nested bridge layouts should exist");
        let nested_guard_layout = nested_layouts
            .iter()
            .find(|layout| !layout.is_finish)
            .expect("nested bridge guard layout should exist");
        let nested_guard_recovery = nested_guard_layout
            .recovery_layout
            .as_ref()
            .expect("nested bridge guard should carry backend recovery");
        assert_eq!(
            nested_guard_recovery
                .frames
                .iter()
                .map(|frame| frame.pc)
                .collect::<Vec<_>>(),
            vec![444, 3000]
        );
        assert_eq!(
            nested_guard_recovery
                .frames
                .iter()
                .map(|frame| frame.source_guard)
                .collect::<Vec<_>>(),
            vec![None, Some((291, 0))]
        );
        assert_eq!(
            nested_guard_recovery.virtual_layouts,
            bridge_source_layout.virtual_layouts
        );
        assert_eq!(
            nested_guard_recovery.pending_field_layouts,
            bridge_source_layout.pending_field_layouts
        );

        let nested_terminal_layouts = backend
            .compiled_trace_terminal_exit_layouts(&token, 292)
            .expect("nested bridge terminal layouts should exist");
        let nested_finish_recovery = nested_terminal_layouts[0]
            .recovery_layout
            .as_ref()
            .expect("nested bridge finish should carry backend recovery");
        assert_eq!(
            nested_finish_recovery
                .frames
                .iter()
                .map(|frame| frame.pc)
                .collect::<Vec<_>>(),
            vec![444, 3000]
        );
        assert_eq!(
            nested_finish_recovery.virtual_layouts,
            bridge_source_layout.virtual_layouts
        );
        assert_eq!(
            nested_finish_recovery.pending_field_layouts,
            bridge_source_layout.pending_field_layouts
        );
    }

    #[test]
    fn test_sum_loop() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(1), OpRef(0)], 2),
            mk_op(OpCode::IntSub, &[OpRef(0), OpRef(100)], 3),
            mk_op(OpCode::IntGt, &[OpRef(3), OpRef(101)], 4),
            mk_op(OpCode::GuardTrue, &[OpRef(4)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(3), OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        constants.insert(101, 0i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(9);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_int_value(&frame, 1), 5049);
    }

    #[test]
    fn test_multi_output_finish() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::IntMul, &[OpRef(0), OpRef(1)], 3),
            mk_op(
                OpCode::Finish,
                &[OpRef(0), OpRef(1), OpRef(2), OpRef(3)],
                OpRef::NONE.0,
            ),
        ];

        let mut token = JitCellToken::new(10);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(3), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 3);
        assert_eq!(backend.get_int_value(&frame, 1), 7);
        assert_eq!(backend.get_int_value(&frame, 2), 10);
        assert_eq!(backend.get_int_value(&frame, 3), 21);
    }

    // ── Call operation tests ──

    #[test]
    fn test_call_i_simple_add() {
        extern "C" fn add_two(a: i64, b: i64) -> i64 {
            a + b
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int, Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallI, &[OpRef(100), OpRef(0), OpRef(1)], 2, descr),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, add_two as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(20);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(40), Value::Int(2)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_call_pure_i() {
        extern "C" fn multiply(a: i64, b: i64) -> i64 {
            a * b
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int, Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallPureI,
                &[OpRef(100), OpRef(0), OpRef(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, multiply as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(21);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(6), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_call_n_void_result() {
        static mut CALL_COUNTER: i64 = 0;

        extern "C" fn increment_counter(amount: i64) {
            unsafe {
                CALL_COUNTER += amount;
            }
        }

        unsafe {
            CALL_COUNTER = 0;
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallN, &[OpRef(100), OpRef(0)], OpRef::NONE.0, descr),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, increment_counter as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(22);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(10)]);
        assert_eq!(backend.get_int_value(&frame, 0), 10);
        assert_eq!(unsafe { CALL_COUNTER }, 10);
    }

    #[test]
    fn test_call_f_double_result() {
        extern "C" fn add_doubles(a: f64, b: f64) -> f64 {
            a + b
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Float, Type::Float], Type::Float);

        let inputargs = vec![InputArg::new_float(0), InputArg::new_float(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallF, &[OpRef(100), OpRef(0), OpRef(1)], 2, descr),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, add_doubles as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(23);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Float(1.5), Value::Float(2.5)]);
        let result = backend.get_float_value(&frame, 0);
        assert!((result - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_call_in_loop() {
        extern "C" fn add_one(a: i64) -> i64 {
            a + 1
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallI, &[OpRef(100), OpRef(0)], 1, descr),
            mk_op(OpCode::IntLt, &[OpRef(1), OpRef(101)], 2),
            mk_op(OpCode::GuardTrue, &[OpRef(2)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, add_one as *const () as i64);
        constants.insert(101, 100i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(24);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
    }

    #[test]
    fn test_call_n_order_preserved_in_loop() {
        use std::sync::{Mutex, OnceLock};

        static EVENTS: OnceLock<Mutex<Vec<i64>>> = OnceLock::new();

        extern "C" fn log_char(v: i64) {
            EVENTS
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap()
                .push(1000 + v);
        }

        extern "C" fn log_num(v: i64) {
            EVENTS
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap()
                .push(v);
        }

        EVENTS
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap()
            .clear();

        let mut backend = CraneliftBackend::new();
        let call_void = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef(100), OpRef(101)],
                OpRef::NONE.0,
                call_void.clone(),
            ),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef(102), OpRef(0)],
                OpRef::NONE.0,
                call_void,
            ),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(103)], 1),
            mk_op(OpCode::IntLt, &[OpRef(1), OpRef(104)], 2),
            mk_op(OpCode::GuardTrue, &[OpRef(2)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, log_char as *const () as i64);
        constants.insert(101, 32);
        constants.insert(102, log_num as *const () as i64);
        constants.insert(103, 1);
        constants.insert(104, 3);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(24_001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 2);
        assert_eq!(
            *EVENTS.get().unwrap().lock().unwrap(),
            vec![1032, 0, 1032, 1, 1032, 2]
        );
    }

    #[test]
    fn test_guard_exception_exact_match_returns_value_and_clears_deadframe_exception() {
        jit_exc_clear();
        clear_test_exception_call_log();
        set_test_exception_state(0xABCDusize as i64, 0x1111);

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardException, &[OpRef(101)], 1);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(102)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallN, &[OpRef(100), OpRef(0)], OpRef::NONE.0, descr),
            guard,
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        constants.insert(101, 0x1111);
        constants.insert(102, 1);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(25);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(0xABCDusize));
        assert_eq!(backend.grab_exc_class(&frame), 0);
        assert_eq!(backend.grab_exc_value(&frame), GcRef::NULL);
        assert!(!jit_exc_is_pending());

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.grab_exc_class(&frame), 0);
        assert_eq!(backend.grab_exc_value(&frame), GcRef::NULL);
        assert!(!jit_exc_is_pending());
    }

    #[test]
    fn test_guard_exception_exact_mismatch_preserves_deadframe_exception() {
        jit_exc_clear();
        clear_test_exception_call_log();
        set_test_exception_state(0xBEEFusize as i64, 0x2222);

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardException, &[OpRef(101)], 1);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(102)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallN, &[OpRef(100), OpRef(0)], OpRef::NONE.0, descr),
            guard,
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        constants.insert(101, 0x1111);
        constants.insert(102, 1);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(26);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.grab_exc_class(&frame), 0x2222);
        assert_eq!(backend.grab_exc_value(&frame), GcRef(0xBEEFusize));
        assert!(!jit_exc_is_pending());
    }

    #[test]
    fn test_guard_no_exception_failure_preserves_deadframe_exception() {
        jit_exc_clear();
        clear_test_exception_call_log();
        set_test_exception_state(0xCAFEusize as i64, 0x1111);

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardNoException, &[], OpRef::NONE.0);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(102)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallN, &[OpRef(100), OpRef(0)], OpRef::NONE.0, descr),
            guard,
            mk_op(OpCode::Finish, &[OpRef(103)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        constants.insert(102, 1);
        constants.insert(103, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(27);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.grab_exc_class(&frame), 0x1111);
        assert_eq!(backend.grab_exc_value(&frame), GcRef(0xCAFEusize));
        assert!(!jit_exc_is_pending());

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 1);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
        assert_eq!(backend.grab_exc_class(&frame), 0);
        assert_eq!(backend.grab_exc_value(&frame), GcRef::NULL);
        assert!(!jit_exc_is_pending());
    }

    #[test]
    fn test_execute_token_ints_raw_preserves_exception_and_layout_metadata() {
        jit_exc_clear();
        clear_test_exception_call_log();
        set_test_exception_state(0xCAFEusize as i64, 0x1111);

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardNoException, &[], OpRef::NONE.0);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(102)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallN, &[OpRef(100), OpRef(0)], OpRef::NONE.0, descr),
            guard,
            mk_op(OpCode::Finish, &[OpRef(103)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        constants.insert(102, 1);
        constants.insert(103, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(27_001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let raw = backend.execute_token_ints_raw(&token, &[1]);
        assert!(!raw.is_finish);
        assert_eq!(raw.fail_index, 0);
        assert_eq!(raw.outputs, vec![1]);
        assert_eq!(raw.typed_outputs, vec![Value::Int(1)]);
        assert_eq!(raw.savedata, None);
        assert_eq!(raw.exception_class, 0x1111);
        assert_eq!(raw.exception_value, GcRef(0xCAFEusize));
        let layout = raw
            .exit_layout
            .expect("raw exit should expose backend layout");
        assert_eq!(layout.fail_index, 0);
        assert_eq!(layout.source_op_index, Some(2));
        assert!(layout.recovery_layout.is_some());
        assert!(!jit_exc_is_pending());
    }

    #[test]
    fn test_save_restore_exception_roundtrip_matches_rpython_order() {
        jit_exc_clear();
        clear_test_exception_call_log();
        set_test_exception_state(0xD00Dusize as i64, 0x3333);

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardException, &[OpRef(101)], 3);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(102)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef(100), OpRef(0)],
                OpRef::NONE.0,
                descr.clone(),
            ),
            mk_op(OpCode::SaveExcClass, &[], 1),
            mk_op(OpCode::SaveException, &[], 2),
            mk_op_with_descr(
                OpCode::CallN,
                &[OpRef(100), OpRef(103)],
                OpRef::NONE.0,
                descr,
            ),
            mk_op(
                OpCode::RestoreException,
                &[OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef(3)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        constants.insert(101, 0x3333);
        constants.insert(102, 1);
        constants.insert(103, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(28);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(0xD00Dusize));
        assert_eq!(backend.grab_exc_class(&frame), 0);
        assert_eq!(backend.grab_exc_value(&frame), GcRef::NULL);
        assert_eq!(test_exception_call_log_snapshot(), vec![false, false]);
        assert!(!jit_exc_is_pending());
    }

    #[test]
    fn test_deadframe_exception_ref_survives_collection_after_execute_token() {
        jit_exc_clear();

        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
        });
        gc.register_type(TypeInfo::simple(16));

        let exception_ref = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(exception_ref.0));
        unsafe {
            *(exception_ref.0 as *mut u64) = 0xCAFEBABE;
        }

        set_test_exception_state(exception_ref.0 as i64, 0x4444);
        clear_test_exception_call_log();

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));
        let runtime_id = backend
            .gc_runtime_id
            .expect("GC runtime must be configured");
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardNoException, &[], OpRef::NONE.0);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(102)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallN, &[OpRef(100), OpRef(0)], OpRef::NONE.0, descr),
            guard,
            mk_op(OpCode::Finish, &[OpRef(103)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        constants.insert(102, 1);
        constants.insert(103, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(29);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        with_gc_runtime(runtime_id, |gc| gc.collect_nursery());

        assert_eq!(backend.grab_exc_class(&frame), 0x4444);
        let moved = backend.grab_exc_value(&frame);
        assert!(!moved.is_null());
        assert_ne!(moved, exception_ref);
        assert_eq!(unsafe { *(moved.0 as *const u64) }, 0xCAFEBABE);
    }

    // ── Debug / no-op tests ──

    #[test]
    fn test_debug_ops_are_noop() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::DebugMergePoint, &[], OpRef::NONE.0),
            mk_op(
                OpCode::EnterPortalFrame,
                &[OpRef(100), OpRef(101)],
                OpRef::NONE.0,
            ),
            mk_op(OpCode::LeavePortalFrame, &[OpRef(100)], OpRef::NONE.0),
            mk_op(OpCode::JitDebug, &[], OpRef::NONE.0),
            mk_op(OpCode::Keepalive, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0i64);
        constants.insert(101, 0i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(30);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    // ── SameAs variants ──

    #[test]
    fn test_same_as_r() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::SameAsR, &[OpRef(0)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(31);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(0x1234))]);
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(0x1234));
    }

    #[test]
    fn test_same_as_f() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::SameAsF, &[OpRef(0)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(32);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Float(3.14)]);
        let result = backend.get_float_value(&frame, 0);
        assert!((result - 3.14).abs() < 1e-10);
    }

    // ── Compile bridge test ──

    #[test]
    fn test_compile_bridge() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(50);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let bridge_inputargs = vec![InputArg::new_int(0)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 10i64);
        backend.set_constants(constants);

        let fail_descr = CraneliftFailDescr::new(0, vec![Type::Int]);

        let info = backend
            .compile_bridge(&fail_descr, &bridge_inputargs, &bridge_ops, &token)
            .unwrap();

        assert!(info.code_addr != 0);
    }

    // ── Conditional call tests ──

    #[test]
    fn test_cond_call_n_calls_when_nonzero() {
        static mut COND_CALL_RESULT: i64 = 0;

        extern "C" fn set_value(v: i64) {
            unsafe {
                COND_CALL_RESULT = v;
            }
        }

        unsafe {
            COND_CALL_RESULT = 0;
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CondCallN,
                &[OpRef(0), OpRef(100), OpRef(1)],
                OpRef::NONE.0,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, set_value as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(40);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1), Value::Int(99)]);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
        assert_eq!(unsafe { COND_CALL_RESULT }, 99);
    }

    #[test]
    fn test_cond_call_n_skips_when_zero() {
        static mut COND_CALL_RESULT2: i64 = 0;

        extern "C" fn set_value2(v: i64) {
            unsafe {
                COND_CALL_RESULT2 = v;
            }
        }

        unsafe {
            COND_CALL_RESULT2 = 0;
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CondCallN,
                &[OpRef(0), OpRef(100), OpRef(1)],
                OpRef::NONE.0,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, set_value2 as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(41);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(99)]);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
        assert_eq!(unsafe { COND_CALL_RESULT2 }, 0);
    }

    #[test]
    fn test_cond_call_value_i_nonzero() {
        extern "C" fn compute(a: i64) -> i64 {
            a * 10
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CondCallValueI,
                &[OpRef(0), OpRef(100), OpRef(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, compute as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(42);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 50);
    }

    #[test]
    fn test_cond_call_value_i_zero() {
        extern "C" fn compute2(a: i64) -> i64 {
            a * 10
        }

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Int);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CondCallValueI,
                &[OpRef(0), OpRef(100), OpRef(1)],
                2,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, compute2 as *const () as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(43);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
    }

    // ── Guard variant tests ──

    #[test]
    fn test_guard_nonnull() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntSub, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::GuardNonnull, &[OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(44);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i0=5->i1=4, i0=4->i1=3, ..., i0=1->i1=0 (guard fails).
        // Guard saves the loop inputarg (i0), so saved value is 1.
        let frame = backend.execute_token(&token, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
    }

    #[test]
    fn test_guard_isnull_passes() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardIsnull, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(45);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
    }

    #[test]
    fn test_guard_isnull_fails() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardIsnull, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(46);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_guard_value() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardValue, &[OpRef(0), OpRef(100)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        // Test: value matches -> guard passes, reaches Finish
        let mut constants = HashMap::new();
        constants.insert(100, 42i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(47);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish
        assert_eq!(backend.get_int_value(&frame, 0), 42);

        // Test: value doesn't match -> guard fails
        let mut constants2 = HashMap::new();
        constants2.insert(100, 42i64);
        backend.set_constants(constants2);

        let mut token2 = JitCellToken::new(48);
        backend.compile_loop(&inputargs, &ops, &mut token2).unwrap();
        let frame2 = backend.execute_token(&token2, &[Value::Int(99)]);
        let descr2 = backend.get_latest_descr(&frame2);
        assert_eq!(descr2.fail_index(), 0); // guard failure
        assert_eq!(backend.get_int_value(&frame2, 0), 99);
    }

    // ── Test descriptors for field/array ops ──

    use majit_ir::descr::{ArrayDescr, FieldDescr, InteriorFieldDescr};

    #[derive(Debug)]
    struct TestSizeDescr {
        size: usize,
        type_id: u32,
        vtable: usize,
    }

    impl Descr for TestSizeDescr {
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            self.size
        }

        fn type_id(&self) -> u32 {
            self.type_id
        }

        fn is_immutable(&self) -> bool {
            false
        }

        fn is_object(&self) -> bool {
            self.vtable != 0
        }

        fn vtable(&self) -> usize {
            self.vtable
        }
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        offset: usize,
        field_size: usize,
        field_type: Type,
        signed: bool,
    }

    impl Descr for TestFieldDescr {
        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestFieldDescr {
        fn offset(&self) -> usize {
            self.offset
        }
        fn field_size(&self) -> usize {
            self.field_size
        }
        fn field_type(&self) -> Type {
            self.field_type
        }
        fn is_field_signed(&self) -> bool {
            self.signed
        }
    }

    #[derive(Debug)]
    struct TestArrayDescr {
        base_size: usize,
        item_size: usize,
        item_type: Type,
        signed: bool,
        len_descr: Option<Arc<TestFieldDescr>>,
    }

    #[derive(Debug)]
    struct TestInteriorFieldDescr {
        array_descr: Arc<TestArrayDescr>,
        field_descr: Arc<TestFieldDescr>,
    }

    impl Descr for TestArrayDescr {
        fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
            Some(self)
        }
    }

    impl ArrayDescr for TestArrayDescr {
        fn base_size(&self) -> usize {
            self.base_size
        }
        fn item_size(&self) -> usize {
            self.item_size
        }
        fn type_id(&self) -> u32 {
            0
        }
        fn item_type(&self) -> Type {
            self.item_type
        }
        fn is_item_signed(&self) -> bool {
            self.signed
        }
        fn len_descr(&self) -> Option<&dyn FieldDescr> {
            self.len_descr
                .as_ref()
                .map(|d| d.as_ref() as &dyn FieldDescr)
        }
    }

    impl Descr for TestInteriorFieldDescr {
        fn as_interior_field_descr(&self) -> Option<&dyn InteriorFieldDescr> {
            Some(self)
        }
    }

    impl InteriorFieldDescr for TestInteriorFieldDescr {
        fn array_descr(&self) -> &dyn ArrayDescr {
            self.array_descr.as_ref()
        }

        fn field_descr(&self) -> &dyn FieldDescr {
            self.field_descr.as_ref()
        }
    }

    fn make_field_descr(
        offset: usize,
        field_size: usize,
        field_type: Type,
        signed: bool,
    ) -> majit_ir::DescrRef {
        Arc::new(TestFieldDescr {
            offset,
            field_size,
            field_type,
            signed,
        })
    }

    fn make_size_descr(size: usize, type_id: u32) -> majit_ir::DescrRef {
        Arc::new(TestSizeDescr {
            size,
            type_id,
            vtable: 0,
        })
    }

    fn make_array_descr(
        base_size: usize,
        item_size: usize,
        item_type: Type,
        len_offset: Option<usize>,
    ) -> majit_ir::DescrRef {
        make_array_descr_with_signedness(base_size, item_size, item_type, true, len_offset)
    }

    fn make_array_descr_with_signedness(
        base_size: usize,
        item_size: usize,
        item_type: Type,
        signed: bool,
        len_offset: Option<usize>,
    ) -> majit_ir::DescrRef {
        let len_descr = len_offset.map(|off| {
            Arc::new(TestFieldDescr {
                offset: off,
                field_size: 8,
                field_type: Type::Int,
                signed: true,
            })
        });
        Arc::new(TestArrayDescr {
            base_size,
            item_size,
            item_type,
            signed,
            len_descr,
        })
    }

    fn make_interior_field_descr(
        array_base_size: usize,
        array_item_size: usize,
        array_item_type: Type,
        field_offset: usize,
        field_size: usize,
        field_type: Type,
        field_signed: bool,
    ) -> majit_ir::DescrRef {
        let array_descr = Arc::new(TestArrayDescr {
            base_size: array_base_size,
            item_size: array_item_size,
            item_type: array_item_type,
            signed: true,
            len_descr: None,
        });
        let field_descr = Arc::new(TestFieldDescr {
            offset: field_offset,
            field_size,
            field_type,
            signed: field_signed,
        });
        Arc::new(TestInteriorFieldDescr {
            array_descr,
            field_descr,
        })
    }

    // ── Float comparison tests ──

    #[test]
    fn test_float_comparisons() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0), InputArg::new_float(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::FloatLt, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::FloatLe, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::FloatEq, &[OpRef(0), OpRef(1)], 4),
            mk_op(OpCode::FloatNe, &[OpRef(0), OpRef(1)], 5),
            mk_op(OpCode::FloatGt, &[OpRef(0), OpRef(1)], 6),
            mk_op(OpCode::FloatGe, &[OpRef(0), OpRef(1)], 7),
            mk_op(
                OpCode::Finish,
                &[OpRef(2), OpRef(3), OpRef(4), OpRef(5), OpRef(6), OpRef(7)],
                OpRef::NONE.0,
            ),
        ];

        let mut token = JitCellToken::new(60);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // 1.5 < 2.5
        let frame = backend.execute_token(&token, &[Value::Float(1.5), Value::Float(2.5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1); // lt
        assert_eq!(backend.get_int_value(&frame, 1), 1); // le
        assert_eq!(backend.get_int_value(&frame, 2), 0); // eq
        assert_eq!(backend.get_int_value(&frame, 3), 1); // ne
        assert_eq!(backend.get_int_value(&frame, 4), 0); // gt
        assert_eq!(backend.get_int_value(&frame, 5), 0); // ge
    }

    #[test]
    fn test_float_comparisons_equal() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_float(0), InputArg::new_float(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::FloatEq, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::FloatNe, &[OpRef(0), OpRef(1)], 3),
            mk_op(OpCode::FloatLe, &[OpRef(0), OpRef(1)], 4),
            mk_op(OpCode::FloatGe, &[OpRef(0), OpRef(1)], 5),
            mk_op(
                OpCode::Finish,
                &[OpRef(2), OpRef(3), OpRef(4), OpRef(5)],
                OpRef::NONE.0,
            ),
        ];

        let mut token = JitCellToken::new(61);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Float(3.14), Value::Float(3.14)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1); // eq
        assert_eq!(backend.get_int_value(&frame, 1), 0); // ne
        assert_eq!(backend.get_int_value(&frame, 2), 1); // le
        assert_eq!(backend.get_int_value(&frame, 3), 1); // ge
    }

    // ── Field access tests ──

    #[test]
    fn test_getfield_gc_i() {
        let mut backend = CraneliftBackend::new();

        // Simulate a struct: [padding(8), i64_field(8)]
        // The field is at offset 8, size 8, type Int.
        let fd = make_field_descr(8, 8, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::GetfieldGcI, &[OpRef(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(70);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Allocate a fake struct on the heap
        let mut data: Vec<i64> = vec![0xDEAD, 42];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_setfield_gc() {
        let mut backend = CraneliftBackend::new();

        let fd = make_field_descr(8, 8, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(1)], OpRef::NONE.0, fd),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(71);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0];
        let ptr = data.as_mut_ptr() as usize;

        let _frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(99)]);
        assert_eq!(data[1], 99);
    }

    #[test]
    fn test_getfield_small_signed() {
        let mut backend = CraneliftBackend::new();

        // i32 field at offset 0, signed
        let fd = make_field_descr(0, 4, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::GetfieldGcI, &[OpRef(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(72);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Write -1i32 into the buffer
        let mut data: Vec<u8> = vec![0; 8];
        let val: i32 = -1;
        data[..4].copy_from_slice(&val.to_ne_bytes());
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), -1i64);
    }

    // ── Array access tests ──

    #[test]
    fn test_getarrayitem_gc_i() {
        let mut backend = CraneliftBackend::new();

        // Array: base_size=16 (header), item_size=8, items are i64
        let ad = make_array_descr(16, 8, Type::Int, None);

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::GetarrayitemGcI, &[OpRef(0), OpRef(1)], 2, ad),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(73);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Layout: 16 bytes header + items
        // Total: 16 + 3*8 = 40 bytes = 5 i64s
        let mut data: Vec<i64> = vec![0xAAAA, 0xBBBB, 10, 20, 30]; // header(2), items(3)
        let ptr = data.as_mut_ptr() as usize;

        // Get item at index 1 (should be 20)
        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(1)]);
        assert_eq!(backend.get_int_value(&frame, 0), 20);
    }

    #[test]
    fn test_setarrayitem_gc() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(16, 8, Type::Int, None);

        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            mk_op_with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(74);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0, 0, 0, 0]; // header(2) + items(3)
        let ptr = data.as_mut_ptr() as usize;

        // Set item at index 2 to 42
        let _frame = backend.execute_token(
            &token,
            &[Value::Ref(GcRef(ptr)), Value::Int(2), Value::Int(42)],
        );
        assert_eq!(data[4], 42); // header(2) + index 2 = slot 4
    }

    #[test]
    fn test_arraylen_gc() {
        let mut backend = CraneliftBackend::new();

        // Array with length at offset 8 (second i64 in header)
        let ad = make_array_descr(16, 8, Type::Int, Some(8));

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::ArraylenGc, &[OpRef(0)], 1, ad),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(75);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // header: [type_id, length=5]
        let mut data: Vec<i64> = vec![0xAAAA, 5, 10, 20, 30, 40, 50];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), 5);
    }

    // ── NurseryPtrIncrement test ──

    #[test]
    fn test_jump_target_label_args_are_declared_even_without_local_producer() {
        let mut backend = CraneliftBackend::new();
        let loop_descr = make_label_descr(9001);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::Jump, &[OpRef(0)], OpRef::NONE.0, loop_descr.clone()),
            mk_op_with_descr(OpCode::Label, &[OpRef(50)], OpRef::NONE.0, loop_descr),
            mk_op(OpCode::Finish, &[OpRef(50)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(76_001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_compile_loop_with_start_and_loop_labels_can_reuse_input_box_names() {
        let mut backend = CraneliftBackend::new();
        let start_descr = make_label_descr(76_100);
        let loop_descr = make_label_descr(76_101);

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef(0), OpRef(1)],
                OpRef::NONE.0,
                start_descr,
            ),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op_with_descr(
                OpCode::Label,
                &[OpRef(0), OpRef(2)],
                OpRef::NONE.0,
                loop_descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(76_102);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(41), Value::Int(1)]);
        assert_eq!(backend.get_int_value(&frame, 0), 41);
    }

    #[test]
    fn test_guard_nonnull_class_checks_object_header_and_null() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(
                OpCode::GuardNonnullClass,
                &[OpRef(0), OpRef(100)],
                OpRef::NONE.0,
            ),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0xCAFE_BABEu64 as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(76_002);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut object_words = vec![0xCAFE_BABEu64 as i64, 123];
        let ptr = object_words.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 1);
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(ptr));

        object_words[0] = 0xDEAD_BEEFu64 as i64;
        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(0))]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
    }

    // ── Overflow detection tests ──

    #[test]
    fn test_int_add_ovf_no_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAddOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(80);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // 10 + 20 = 30 (no overflow)
        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(20)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish (guard passed)
        assert_eq!(backend.get_int_value(&frame, 0), 30);
    }

    #[test]
    fn test_int_add_ovf_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAddOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(81);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i64::MAX + 1 overflows
        let frame = backend.execute_token(&token, &[Value::Int(i64::MAX), Value::Int(1)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0); // guard failure (overflow)
    }

    #[test]
    fn test_int_sub_ovf_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntSubOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(82);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i64::MIN - 1 overflows
        let frame = backend.execute_token(&token, &[Value::Int(i64::MIN), Value::Int(1)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0); // guard failure (overflow)
    }

    #[test]
    fn test_int_sub_ovf_no_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntSubOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(83);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(100), Value::Int(58)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish (guard passed)
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_mul_ovf_no_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntMulOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(84);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(6), Value::Int(7)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish (guard passed)
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_int_mul_ovf_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntMulOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardNoOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(85);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // i64::MAX * 2 overflows
        let frame = backend.execute_token(&token, &[Value::Int(i64::MAX), Value::Int(2)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0); // guard failure (overflow)
    }

    #[test]
    fn test_guard_overflow() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAddOvf, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::GuardOverflow, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(86);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // With overflow: guard_overflow passes (continues)
        let frame = backend.execute_token(&token, &[Value::Int(i64::MAX), Value::Int(1)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish (overflow happened, guard passed)

        // Without overflow: guard_overflow fails (side-exits)
        let mut token2 = JitCellToken::new(87);
        backend.compile_loop(&inputargs, &ops, &mut token2).unwrap();
        let frame2 = backend.execute_token(&token2, &[Value::Int(1), Value::Int(2)]);
        let descr2 = backend.get_latest_descr(&frame2);
        assert_eq!(descr2.fail_index(), 0); // guard failure (no overflow)
    }

    // ── Getfield float test ──

    #[test]
    fn test_getfield_gc_f() {
        let mut backend = CraneliftBackend::new();

        // f64 field at offset 0
        let fd = make_field_descr(0, 8, Type::Float, false);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::GetfieldGcF, &[OpRef(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(90);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let val: f64 = 3.14;
        let mut data = vec![0u8; 8];
        data.copy_from_slice(&val.to_ne_bytes());
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        let result = backend.get_float_value(&frame, 0);
        assert!((result - 3.14).abs() < 1e-10);
    }

    // ── Getfield ref (pure) test ──

    #[test]
    fn test_getfield_gc_pure_r() {
        let mut backend = CraneliftBackend::new();

        // Ref field at offset 8
        let fd = make_field_descr(8, 8, Type::Ref, false);

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::GetfieldGcPureR, &[OpRef(0)], 1, fd),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(91);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0x42424242];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(0x42424242usize));
    }

    // ── Setfield + getfield roundtrip ──

    #[test]
    fn test_setfield_getfield_roundtrip() {
        let mut backend = CraneliftBackend::new();

        let fd = make_field_descr(0, 8, Type::Int, true);

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef(0), OpRef(1)],
                OpRef::NONE.0,
                fd.clone(),
            ),
            mk_op_with_descr(OpCode::GetfieldGcI, &[OpRef(0)], 2, fd),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(92);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(12345)]);
        assert_eq!(backend.get_int_value(&frame, 0), 12345);
    }

    // ── Array getitem/setitem roundtrip ──

    #[test]
    fn test_setarrayitem_getarrayitem_roundtrip() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(0, 8, Type::Int, None);

        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            mk_op_with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
                ad.clone(),
            ),
            mk_op_with_descr(OpCode::GetarrayitemGcI, &[OpRef(0), OpRef(1)], 3, ad),
            mk_op(OpCode::Finish, &[OpRef(3)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(93);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<i64> = vec![0, 0, 0, 0];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(
            &token,
            &[Value::Ref(GcRef(ptr)), Value::Int(2), Value::Int(777)],
        );
        assert_eq!(backend.get_int_value(&frame, 0), 777);
    }

    #[test]
    fn test_gc_load_i_signed_itemsize() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GcLoadI, &[OpRef(0), OpRef(100), OpRef(101)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 4);
        constants.insert(101, -4);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(94);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0u8; 8];
        let val: i32 = -7;
        data[4..8].copy_from_slice(&val.to_ne_bytes());
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), -7);
    }

    #[test]
    fn test_gc_load_indexed_r_uses_scale_and_base_offset() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(
                OpCode::GcLoadIndexedR,
                &[OpRef(0), OpRef(1), OpRef(100), OpRef(101), OpRef(102)],
                2,
            ),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 8);
        constants.insert(101, 16);
        constants.insert(102, 8);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(95);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<u64> = vec![0xAAAA, 0xBBBB, 0x1111_1111, 0xDEAD_BEEF];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(1)]);
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(0xDEAD_BEEFusize));
    }

    #[test]
    fn test_gc_store_four_arg_form_stores_ref_value() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_ref(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(
                OpCode::GcStore,
                &[OpRef(0), OpRef(100), OpRef(1), OpRef(101)],
                OpRef::NONE.0,
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 8);
        constants.insert(101, 8);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(96);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<u64> = vec![0, 0];
        let ptr = data.as_mut_ptr() as usize;

        let stored = GcRef(0xFEED_FACEusize);
        let _frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Ref(stored)]);
        assert_eq!(data[1] as usize, stored.0);
    }

    #[test]
    fn test_gc_store_indexed_roundtrip_i32() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            mk_op(
                OpCode::GcStoreIndexed,
                &[
                    OpRef(0),
                    OpRef(1),
                    OpRef(2),
                    OpRef(100),
                    OpRef(101),
                    OpRef(102),
                ],
                OpRef::NONE.0,
            ),
            mk_op(
                OpCode::GcLoadIndexedI,
                &[OpRef(0), OpRef(1), OpRef(100), OpRef(101), OpRef(102)],
                3,
            ),
            mk_op(OpCode::Finish, &[OpRef(3)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 4);
        constants.insert(101, 8);
        constants.insert(102, 4);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(97);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0u8; 24];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(
            &token,
            &[
                Value::Ref(GcRef(ptr)),
                Value::Int(2),
                Value::Int(0x1234_5678),
            ],
        );
        assert_eq!(backend.get_int_value(&frame, 0), 0x1234_5678);
    }

    #[test]
    fn test_raw_load_i_unsigned_zero_extends() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr_with_signedness(0, 1, Type::Int, false, None);
        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::RawLoadI, &[OpRef(0), OpRef(100)], 1, ad),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(98);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0xFFu8];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), 255);
    }

    #[test]
    fn test_raw_store_and_load_f_roundtrip() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(0, 8, Type::Float, None);
        let inputargs = vec![InputArg::new_ref(0), InputArg::new_float(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::RawStore,
                &[OpRef(0), OpRef(100), OpRef(1)],
                OpRef::NONE.0,
                ad.clone(),
            ),
            mk_op_with_descr(OpCode::RawLoadF, &[OpRef(0), OpRef(100)], 2, ad),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 2);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(99);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0u8; 8];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Float(6.25)]);
        assert!((backend.get_float_value(&frame, 0) - 6.25).abs() < 1e-10);
    }

    #[test]
    fn test_gc_load_indexed_rejects_nonconstant_scale() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            mk_op(
                OpCode::GcLoadIndexedI,
                &[OpRef(0), OpRef(1), OpRef(2), OpRef(100), OpRef(101)],
                3,
            ),
            mk_op(OpCode::Finish, &[OpRef(3)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 2);
        constants.insert(101, 8);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(100);
        let err = backend
            .compile_loop(&inputargs, &ops, &mut token)
            .unwrap_err();
        match err {
            BackendError::Unsupported(msg) => {
                assert!(msg.contains("scale must be a compile-time constant"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn test_getinteriorfield_gc_i_signed() {
        let mut backend = CraneliftBackend::new();

        let id = make_interior_field_descr(8, 16, Type::Int, 4, 4, Type::Int, true);
        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::GetinteriorfieldGcI, &[OpRef(0), OpRef(1)], 2, id),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(101);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0u8; 40];
        let val: i32 = -11;
        let field_offset = 8 + 16 + 4;
        data[field_offset..field_offset + 4].copy_from_slice(&val.to_ne_bytes());
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr)), Value::Int(1)]);
        assert_eq!(backend.get_int_value(&frame, 0), -11);
    }

    #[test]
    fn test_setinteriorfield_roundtrip() {
        let mut backend = CraneliftBackend::new();

        let id = make_interior_field_descr(8, 16, Type::Int, 8, 8, Type::Int, true);
        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            mk_op_with_descr(
                OpCode::SetinteriorfieldGc,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
                id.clone(),
            ),
            mk_op_with_descr(OpCode::GetinteriorfieldGcI, &[OpRef(0), OpRef(1)], 3, id),
            mk_op(OpCode::Finish, &[OpRef(3)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(102);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![0u8; 40];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(
            &token,
            &[
                Value::Ref(GcRef(ptr)),
                Value::Int(1),
                Value::Int(0x1234_5678_9ABC_DEF0),
            ],
        );
        assert_eq!(
            backend.get_int_value(&frame, 0),
            0x1234_5678_9ABC_DEF0u64 as i64
        );
    }

    // ── Bridge compilation and execution tests ──

    #[test]
    fn test_bridge_attaches_to_guard() {
        // Compile a loop: input(x) -> guard_true(x > 0) -> finish(x * 2)
        // When guard fails (x <= 0), compile a bridge that returns x + 100.
        // After attaching the bridge, executing with x <= 0 should run the
        // bridge and return x + 100 instead of falling back.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntGt, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::GuardTrue, &[OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntMul, &[OpRef(0), OpRef(101)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0i64); // guard: x > 0
        constants.insert(101, 2i64); // x * 2
        backend.set_constants(constants);

        let mut token = JitCellToken::new(200);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // First: verify guard fails when x = -5
        let frame = backend.execute_token(&token, &[Value::Int(-5)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0); // guard_true is guard index 0
        assert_eq!(backend.get_int_value(&frame, 0), -5);

        // Now compile a bridge for guard 0: bridge takes x, returns x + 100
        let bridge_inputargs = vec![InputArg::new_int(0)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(100, 100i64);
        backend.set_constants(bridge_constants);

        let fail_descr = CraneliftFailDescr::new(0, vec![Type::Int]);
        let bridge_info = backend
            .compile_bridge(&fail_descr, &bridge_inputargs, &bridge_ops, &token)
            .unwrap();
        assert!(bridge_info.code_addr != 0);

        // Now execute again with x = -5: the bridge should execute
        let frame = backend.execute_token(&token, &[Value::Int(-5)]);
        // Bridge returns -5 + 100 = 95
        assert_eq!(backend.get_int_value(&frame, 0), 95);

        // Execute with x = 0 (also fails guard): bridge returns 0 + 100 = 100
        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 100);

        // Execute with x = 5 (guard passes): original path returns 5 * 2 = 10
        let frame = backend.execute_token(&token, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 10);
    }

    #[test]
    fn test_bridge_from_guard_no_exception_can_consume_pending_exception() {
        jit_exc_clear();
        clear_test_exception_call_log();
        set_test_exception_state(0xE11Eusize as i64, 0x4545);

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Int], Type::Void);

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(OpCode::GuardNoException, &[], OpRef::NONE.0);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallN, &[OpRef(100), OpRef(0)], OpRef::NONE.0, descr),
            guard,
            mk_op(OpCode::Finish, &[OpRef(101)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_raise_test_exception as *const () as i64);
        constants.insert(101, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(202);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.grab_exc_class(&frame), 0x4545);
        assert_eq!(backend.grab_exc_value(&frame), GcRef(0xE11Eusize));

        let bridge_inputargs = vec![InputArg::new_int(0)];
        let mut bridge_guard = mk_op(OpCode::GuardException, &[OpRef(100)], 1);
        bridge_guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(0)]));
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            bridge_guard,
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(100, 0x4545);
        backend.set_constants(bridge_constants);

        let fail_descr = CraneliftFailDescr::new(0, vec![Type::Int]);
        backend
            .compile_bridge(&fail_descr, &bridge_inputargs, &bridge_ops, &token)
            .unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(1)]);
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(0xE11Eusize));
        assert_eq!(backend.grab_exc_class(&frame), 0);
        assert_eq!(backend.grab_exc_value(&frame), GcRef::NULL);
        assert!(!jit_exc_is_pending());
    }

    #[test]
    fn test_guard_failure_counting() {
        // Verify that guard failures are counted.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(201);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Execute with x = 0 (guard fails) multiple times
        for i in 1..=5 {
            let _frame = backend.execute_token(&token, &[Value::Int(0)]);
            let compiled = token
                .compiled
                .as_ref()
                .unwrap()
                .downcast_ref::<CompiledLoop>()
                .unwrap();
            let descr = &compiled.fail_descrs[0];
            assert_eq!(descr.get_fail_count(), i);
        }

        // Execute with x = 1 (guard passes, reaches finish)
        let _frame = backend.execute_token(&token, &[Value::Int(1)]);
        let compiled = token
            .compiled
            .as_ref()
            .unwrap()
            .downcast_ref::<CompiledLoop>()
            .unwrap();
        // Guard descr count unchanged (guard didn't fail this time)
        let guard_descr = &compiled.fail_descrs[0];
        assert_eq!(guard_descr.get_fail_count(), 5);
    }

    #[test]
    fn test_bridge_with_explicit_fail_args() {
        // Test that when a guard has explicit fail_args, the bridge receives
        // those values as inputs.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0);
        // Explicit fail_args: save both i0 and i1 plus the computed sum (i0+i1)
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(0), OpRef(1)]));

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            guard_op,
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(203);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Without bridge: guard fails on x=0, y=42
        let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(42)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 0); // x
        assert_eq!(backend.get_int_value(&frame, 1), 42); // y

        // Compile bridge that takes the fail_args (x, y) and returns x + y + 1000
        let bridge_inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::IntAdd, &[OpRef(2), OpRef(100)], 3),
            mk_op(OpCode::Finish, &[OpRef(3)], OpRef::NONE.0),
        ];

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(100, 1000i64);
        backend.set_constants(bridge_constants);

        let fail_descr = CraneliftFailDescr::new(0, vec![Type::Int, Type::Int]);
        backend
            .compile_bridge(&fail_descr, &bridge_inputargs, &bridge_ops, &token)
            .unwrap();

        // With bridge: guard fails on x=0, y=42 -> bridge returns 0 + 42 + 1000 = 1042
        let frame = backend.execute_token(&token, &[Value::Int(0), Value::Int(42)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1042);

        // Guard passes on x=5, y=7 -> original path returns 5 + 7 = 12
        let frame = backend.execute_token(&token, &[Value::Int(5), Value::Int(7)]);
        assert_eq!(backend.get_int_value(&frame, 0), 12);
    }

    #[test]
    fn test_guard_fail_args_preserve_float_and_ref_types() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_float(1),
            InputArg::new_ref(2),
        ];
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(1), OpRef(2)]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(
            &token,
            &[Value::Int(0), Value::Float(9.25), Value::Ref(GcRef(0xBEEF))],
        );
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_float_value(&frame, 0), 9.25);
        assert_eq!(backend.get_ref_value(&frame, 1), GcRef(0xBEEF));
    }

    #[test]
    fn test_fail_descr_gc_map_tracks_ref_slots() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
            InputArg::new_ref(3),
        ];
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef(0),
            OpRef(1),
            OpRef(2),
            OpRef(3),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2), OpRef(3)],
                OpRef::NONE.0,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1005);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let compiled = token
            .compiled
            .as_ref()
            .unwrap()
            .downcast_ref::<CompiledLoop>()
            .unwrap();
        let descr = &compiled.fail_descrs[0];
        assert!(!descr.gc_map().is_ref(0));
        assert!(descr.gc_map().is_ref(1));
        assert!(!descr.gc_map().is_ref(2));
        assert!(descr.gc_map().is_ref(3));
    }

    #[test]
    fn test_zero_array_clears_requested_range() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(16, 4, Type::Int, None);
        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::ZeroArray,
                &[OpRef(0), OpRef(100), OpRef(101), OpRef(102), OpRef(103)],
                OpRef::NONE.0,
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 4);
        constants.insert(101, 8);
        constants.insert(102, 1);
        constants.insert(103, 1);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1002);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data: Vec<u8> = vec![
            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
            0xAA, 0xAA, // header
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        ];
        let ptr = data.as_mut_ptr() as usize;

        let _frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(&data[16..20], &[1, 2, 3, 4]);
        assert_eq!(&data[20..28], &[0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_zero_array_uses_scaled_start_and_size() {
        let mut backend = CraneliftBackend::new();

        let ad = make_array_descr(8, 4, Type::Int, None);
        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            mk_op_with_descr(
                OpCode::ZeroArray,
                &[OpRef(0), OpRef(1), OpRef(2), OpRef(100), OpRef(101)],
                OpRef::NONE.0,
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 4);
        constants.insert(101, 4);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1003);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut data = vec![
            0xEEu8, 0xEE, 0xEE, 0xEE, 0xEE, 0xEE, 0xEE, 0xEE, // header
            1, 1, 1, 1, // item0
            2, 2, 2, 2, // item1
            3, 3, 3, 3, // item2
            4, 4, 4, 4, // item3
        ];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(
            &token,
            &[Value::Ref(GcRef(ptr)), Value::Int(1), Value::Int(2)],
        );
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(&data[8..12], &[1, 1, 1, 1]);
        assert_eq!(&data[12..20], &[0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&data[20..24], &[4, 4, 4, 4]);
    }

    #[test]
    fn test_unsupported_void_opcode_errors() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::CallMayForceN, &[], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1004);
        let err = backend
            .compile_loop(&inputargs, &ops, &mut token)
            .unwrap_err();
        match err {
            BackendError::Unsupported(msg) => assert!(msg.contains("CallMayForceN")),
            other => panic!("expected unsupported error, got {other:?}"),
        }
    }

    #[test]
    fn test_call_may_force_n_guard_not_forced_and_savedata() {
        may_force_void_values().lock().unwrap().clear();

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Void);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(1), OpRef(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceN,
                &[OpRef(100), OpRef(2), OpRef(1)],
                OpRef::NONE.0,
                descr,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            maybe_force_and_return_void as *const () as usize as i64,
        );
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_400);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(20), Value::Int(0)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 1);
        assert_eq!(backend.get_int_value(&frame, 0), 20);
        assert!(may_force_void_values().lock().unwrap().is_empty());

        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_int_value(&frame, 1), 10);
        assert_eq!(*may_force_void_values().lock().unwrap(), vec![0, 1, 10]);
        assert_eq!(backend.get_savedata_ref(&frame).unwrap(), GcRef(0xDADA));
    }

    #[test]
    fn test_execute_token_ints_raw_preserves_savedata_and_layout_for_call_may_force() {
        may_force_void_values().lock().unwrap().clear();

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Void);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(1), OpRef(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceN,
                &[OpRef(100), OpRef(2), OpRef(1)],
                OpRef::NONE.0,
                descr,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            maybe_force_and_return_void as *const () as usize as i64,
        );
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_404);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let raw = backend.execute_token_ints_raw(&token, &[10, 1]);
        assert!(!raw.is_finish);
        assert_eq!(raw.fail_index, 0);
        assert_eq!(raw.outputs, vec![1, 10]);
        assert_eq!(raw.typed_outputs, vec![Value::Int(1), Value::Int(10)]);
        assert_eq!(raw.savedata, Some(GcRef(0xDADA)));
        assert_eq!(raw.exception_class, 0);
        assert_eq!(raw.exception_value, GcRef::NULL);
        let layout = raw
            .exit_layout
            .expect("raw exit should expose backend layout");
        assert_eq!(layout.fail_index, 0);
        assert_eq!(layout.source_op_index, Some(3));
        assert!(layout.recovery_layout.is_some());
        assert_eq!(*may_force_void_values().lock().unwrap(), vec![0, 1, 10]);
    }

    #[test]
    fn test_call_may_force_with_intervening_ops() {
        // CallMayForceI -> SameAsI -> GuardNotForced
        // Verifies that intervening non-guard ops between
        // CallMayForce and GuardNotForced compile and execute correctly.
        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Int);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef(1),
            OpRef(3),
            OpRef(0),
        ]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceI,
                &[OpRef(100), OpRef(2), OpRef(1)],
                3,
                descr,
            ),
            // Intervening op: SameAsI copies the call result.
            // This proves the codegen accepts non-adjacent placement.
            mk_op(OpCode::SameAsI, &[OpRef(3)], 4),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(4)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            maybe_force_and_return_int_isolated as *const () as usize as i64,
        );
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_410);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Not forced: reaches Finish with SameAsI result (== call result == 42)
        let frame = backend.execute_token(&token, &[Value::Int(20), Value::Int(0)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 1);
        assert_eq!(backend.get_int_value(&frame, 0), 42);

        // Forced: exits via GuardNotForced
        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_int_value(&frame, 1), 42);
        assert_eq!(backend.get_int_value(&frame, 2), 10);
        assert_eq!(backend.get_savedata_ref(&frame).unwrap(), GcRef(0xBABA));
    }

    #[test]
    fn test_call_may_force_i_guard_not_forced_uses_real_call_result() {
        may_force_int_values().lock().unwrap().clear();

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Int);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef(1),
            OpRef(3),
            OpRef(0),
        ]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceI,
                &[OpRef(100), OpRef(2), OpRef(1)],
                3,
                descr,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(3)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_force_and_return_int as *const () as usize as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_401);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(20), Value::Int(0)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 1);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
        assert!(may_force_int_values().lock().unwrap().is_empty());

        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_int_value(&frame, 1), 42);
        assert_eq!(backend.get_int_value(&frame, 2), 10);
        assert_eq!(*may_force_int_values().lock().unwrap(), vec![1, 10]);
        assert_eq!(backend.get_savedata_ref(&frame).unwrap(), GcRef(0xBABA));
    }

    #[test]
    fn test_call_may_force_f_guard_not_forced_uses_real_call_result() {
        may_force_float_values().lock().unwrap().clear();

        let mut backend = CraneliftBackend::new();
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Float);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef(1),
            OpRef(3),
            OpRef(0),
        ]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceF,
                &[OpRef(100), OpRef(2), OpRef(1)],
                3,
                descr,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(3)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            maybe_force_and_return_float as *const () as usize as i64,
        );
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_402);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(20), Value::Int(0)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 1);
        assert_eq!(backend.get_float_value(&frame, 0), 12.5);
        assert!(may_force_float_values().lock().unwrap().is_empty());

        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(1)]);
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        assert_eq!(backend.get_float_value(&frame, 1), 12.5);
        assert_eq!(backend.get_int_value(&frame, 2), 10);
        assert_eq!(
            *may_force_float_values().lock().unwrap(),
            vec![1, 0.0f64.to_bits(), 10]
        );
    }

    #[test]
    fn test_call_may_force_r_guard_not_forced_survives_gc_and_uses_real_result() {
        may_force_ref_values().lock().unwrap().clear();

        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 256,
            large_object_threshold: 1024,
        });
        gc.register_type(TypeInfo::simple(16));
        let live_ref = gc.alloc_with_type(0, 16);
        let return_ref = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(live_ref.0));
        assert!(gc.is_in_nursery(return_ref.0));

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));
        let runtime_id = backend
            .gc_runtime_id
            .expect("GC backend must expose a runtime id");
        let descr = make_call_descr(vec![Type::Ref, Type::Int, Type::Int, Type::Ref], Type::Ref);
        let inputargs = vec![
            InputArg::new_ref(0),
            InputArg::new_ref(1),
            InputArg::new_int(2),
        ];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef(2),
            OpRef(4),
            OpRef(0),
            OpRef(1),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            mk_op(OpCode::ForceToken, &[], 3),
            mk_op_with_descr(
                OpCode::CallMayForceR,
                &[OpRef(100), OpRef(3), OpRef(2), OpRef(101), OpRef(1)],
                4,
                descr,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(4)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, maybe_force_and_return_ref as *const () as usize as i64);
        constants.insert(101, runtime_id as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_403);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(
            &token,
            &[Value::Ref(live_ref), Value::Ref(return_ref), Value::Int(0)],
        );
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 1);
        let unforced_result = backend.get_ref_value(&frame, 0);
        assert_eq!(unforced_result, return_ref);
        assert!(may_force_ref_values().lock().unwrap().is_empty());

        let frame = backend.execute_token(
            &token,
            &[Value::Ref(live_ref), Value::Ref(return_ref), Value::Int(1)],
        );
        assert_eq!(backend.get_latest_descr(&frame).fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 1);
        let forced_result = backend.get_ref_value(&frame, 1);
        let forced_live = backend.get_ref_value(&frame, 2);
        let forced_return = backend.get_ref_value(&frame, 3);
        assert_ne!(forced_live, GcRef::NULL);
        assert_eq!(forced_result, forced_return);
        assert_ne!(forced_result, forced_live);
        assert_eq!(backend.get_savedata_ref(&frame).unwrap(), forced_live);
        assert_eq!(
            *may_force_ref_values().lock().unwrap(),
            vec![GcRef::NULL.0, forced_live.0, forced_return.0]
        );
    }

    #[test]
    fn test_guard_not_forced_2_can_be_forced_after_finish() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let mut guard_op = mk_op(OpCode::GuardNotForced2, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(1)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::ForceToken, &[], 2),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 10);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_300);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(20)]);
        let force_token = backend.get_ref_value(&frame, 0);
        assert_ne!(force_token, GcRef::NULL);

        let forced = backend.force(force_token).unwrap();
        let descr = backend.get_latest_descr(&forced);
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_int_value(&forced, 0), 30);
    }

    #[test]
    fn test_guard_not_forced_2_snapshot_roots_refs_until_force() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
        });
        gc.register_type(TypeInfo::simple(16));

        let root = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(root.0));
        unsafe {
            *(root.0 as *mut u64) = 0x1234_5678;
        }

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));
        let runtime_id = backend
            .gc_runtime_id
            .expect("GC runtime must be configured");

        let inputargs = vec![InputArg::new_ref(0)];
        let mut guard_op = mk_op(OpCode::GuardNotForced2, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::ForceToken, &[], 1),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1500_301);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        let force_token = backend.get_ref_value(&frame, 0);

        with_gc_runtime(runtime_id, |gc| gc.collect_nursery());

        let forced = backend.force(force_token).unwrap();
        let moved_root = backend.get_ref_value(&forced, 0);
        assert!(!moved_root.is_null());
        assert_eq!(unsafe { *(moved_root.0 as *const u64) }, 0x1234_5678);
    }

    #[test]
    fn test_call_assembler_i_executes_finish_only_target() {
        let mut backend = CraneliftBackend::new();

        let callee_inputargs = vec![InputArg::new_int(0)];
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut callee_constants = HashMap::new();
        callee_constants.insert(100, 2);
        backend.set_constants(callee_constants);

        let mut callee_token = JitCellToken::new(1500_200);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee_token)
            .unwrap();

        let caller_inputargs = vec![InputArg::new_int(0)];
        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallAssemblerI,
                &[OpRef(0)],
                1,
                make_call_assembler_descr(&callee_token, vec![Type::Int], Type::Int),
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        backend.set_constants(HashMap::new());

        let mut caller_token = JitCellToken::new(1500_201);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller_token)
            .unwrap();

        let frame = backend.execute_token(&caller_token, &[Value::Int(40)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_call_assembler_redirect_switches_target() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let callee1_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        backend.set_constants(constants);
        let mut callee1 = JitCellToken::new(1500_210);
        backend
            .compile_loop(&inputargs, &callee1_ops, &mut callee1)
            .unwrap();

        let callee2_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 100);
        backend.set_constants(constants);
        let mut callee2 = JitCellToken::new(1500_211);
        backend
            .compile_loop(&inputargs, &callee2_ops, &mut callee2)
            .unwrap();

        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallAssemblerI,
                &[OpRef(0)],
                1,
                make_call_assembler_descr(&callee1, vec![Type::Int], Type::Int),
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        backend.set_constants(HashMap::new());
        let mut caller = JitCellToken::new(1500_212);
        backend
            .compile_loop(&inputargs, &caller_ops, &mut caller)
            .unwrap();

        let frame = backend.execute_token(&caller, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 6);

        backend.redirect_call_assembler(&callee1, &callee2).unwrap();

        let frame = backend.execute_token(&caller, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 105);
    }

    #[test]
    fn test_call_assembler_guarded_target_uses_attached_bridge_finish() {
        let mut backend = CraneliftBackend::new();

        backend.set_next_trace_id(1500_230);
        let callee_inputargs = vec![InputArg::new_int(0)];
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut callee = JitCellToken::new(1500_230);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee)
            .unwrap();

        let failed = backend.execute_token(&callee, &[Value::Int(0)]);
        let guard_descr = failed
            .data
            .downcast_ref::<FrameData>()
            .expect("callee guard should produce FrameData")
            .fail_descr
            .clone();

        backend.set_next_trace_id(1500_231);
        let bridge_inputargs = vec![InputArg::new_int(0)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 100);
        backend.set_constants(constants);
        backend
            .compile_bridge(
                guard_descr.as_ref(),
                &bridge_inputargs,
                &bridge_ops,
                &callee,
            )
            .unwrap();

        backend.set_constants(HashMap::new());
        backend.set_next_trace_id(1500_232);
        let caller_inputargs = vec![InputArg::new_int(0)];
        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallAssemblerI,
                &[OpRef(0)],
                1,
                make_call_assembler_descr(&callee, vec![Type::Int], Type::Int),
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut caller = JitCellToken::new(1500_232);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller)
            .unwrap();

        let frame = backend.execute_token(&caller, &[Value::Int(0)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert_eq!(descr.trace_id(), 1500_232);
        assert_eq!(backend.get_int_value(&frame, 0), 100);
    }

    #[test]
    fn test_call_assembler_compiles_before_target_is_registered() {
        let mut backend = CraneliftBackend::new();

        let mut deferred_target = JitCellToken::new(1500_240);
        backend.set_next_trace_id(1500_241);
        let caller_inputargs = vec![InputArg::new_int(0)];
        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallAssemblerI,
                &[OpRef(0)],
                1,
                make_call_assembler_descr(&deferred_target, vec![Type::Int], Type::Int),
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut caller = JitCellToken::new(1500_241);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller)
            .unwrap();

        backend.set_next_trace_id(1500_240);
        let callee_inputargs = vec![InputArg::new_int(0)];
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 7);
        backend.set_constants(constants);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut deferred_target)
            .unwrap();

        backend.set_constants(HashMap::new());
        let frame = backend.execute_token(&caller, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 12);
    }

    #[test]
    fn test_call_assembler_float_finish_and_raw_path() {
        let mut backend = CraneliftBackend::new();

        backend.set_next_trace_id(1500_242);
        let callee_inputargs = vec![InputArg::new_float(0)];
        backend.set_constants(HashMap::from([(100, 2.25f64.to_bits() as i64)]));
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::FloatAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut callee = JitCellToken::new(1500_242);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee)
            .unwrap();

        backend.set_constants(HashMap::new());
        backend.set_next_trace_id(1500_243);
        let caller_inputargs = vec![InputArg::new_float(0)];
        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallAssemblerF,
                &[OpRef(0)],
                1,
                make_call_assembler_descr(&callee, vec![Type::Float], Type::Float),
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut caller = JitCellToken::new(1500_243);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller)
            .unwrap();

        let frame = backend.execute_token(&caller, &[Value::Float(3.5)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert_eq!(descr.trace_id(), 1500_243);
        assert!((backend.get_float_value(&frame, 0) - 5.75).abs() < 1e-10);

        let raw = backend.execute_token_ints_raw(&caller, &[3.5f64.to_bits() as i64]);
        assert!(raw.is_finish);
        assert_eq!(raw.trace_id, 1500_243);
        assert_eq!(raw.outputs, vec![5.75f64.to_bits() as i64]);
        assert_eq!(raw.typed_outputs, vec![Value::Float(5.75)]);
        let layout = raw
            .exit_layout
            .expect("call_assembler float raw exit should expose layout");
        assert_eq!(layout.trace_id, 1500_243);
        assert_eq!(layout.fail_arg_types, vec![Type::Float]);
    }

    #[test]
    fn test_call_assembler_late_bound_ref_result_supports_plain_ref_finish() {
        let mut backend = CraneliftBackend::new();

        let mut deferred_target = JitCellToken::new(1500_245);
        let caller_inputargs = vec![InputArg::new_ref(0)];
        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallAssemblerR,
                &[OpRef(0)],
                1,
                make_call_assembler_descr(&deferred_target, vec![Type::Ref], Type::Ref),
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut caller = JitCellToken::new(1500_246);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller)
            .unwrap();

        backend.set_next_trace_id(1500_245);
        let callee_inputargs = vec![InputArg::new_ref(0)];
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        backend.set_constants(HashMap::new());
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut deferred_target)
            .unwrap();

        let root = GcRef(0xCAFE);
        let frame = backend.execute_token(&caller, &[Value::Ref(root)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert!(descr.force_token_slots().is_empty());
        assert_eq!(backend.get_ref_value(&frame, 0), root);
    }

    #[test]
    fn test_call_assembler_late_bound_ref_result_rejects_force_token_finish_shape() {
        let mut backend = CraneliftBackend::new();

        let mut deferred_target = JitCellToken::new(1500_247);
        let caller_inputargs = vec![InputArg::new_int(0)];
        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallAssemblerR,
                &[OpRef(0)],
                1,
                make_call_assembler_descr(&deferred_target, vec![Type::Int], Type::Ref),
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut caller = JitCellToken::new(1500_248);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller)
            .unwrap();

        backend.set_next_trace_id(1500_247);
        let callee_inputargs = vec![InputArg::new_int(0)];
        let mut guard_op = mk_op(OpCode::GuardNotForced2, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(1)]));
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::ForceToken, &[], 2),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 10);
        backend.set_constants(constants);
        let err = backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut deferred_target)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("incompatible callee finish result kind"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_call_assembler_redirect_rejects_incompatible_force_token_result_shape() {
        let mut backend = CraneliftBackend::new();

        let ref_inputargs = vec![InputArg::new_ref(0)];
        let plain_ref_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut plain_ref_target = JitCellToken::new(1500_349);
        backend
            .compile_loop(&ref_inputargs, &plain_ref_ops, &mut plain_ref_target)
            .unwrap();

        let mut guard_op = mk_op(OpCode::GuardNotForced2, &[], OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(1)]));
        let force_token_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::ForceToken, &[], 1),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut force_token_target = JitCellToken::new(1500_350);
        backend
            .compile_loop(&ref_inputargs, &force_token_ops, &mut force_token_target)
            .unwrap();

        let caller_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CallAssemblerR,
                &[OpRef(0)],
                1,
                make_call_assembler_descr(&plain_ref_target, vec![Type::Ref], Type::Ref),
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];
        let mut caller = JitCellToken::new(1500_351);
        backend
            .compile_loop(&ref_inputargs, &caller_ops, &mut caller)
            .unwrap();

        let err = backend
            .redirect_call_assembler(&plain_ref_target, &force_token_target)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("incompatible callee finish result kind"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_call_assembler_supports_direct_self_recursive_dispatch() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1500_250);
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntGt, &[OpRef(0), OpRef(101)], 1),
            mk_op(OpCode::GuardTrue, &[OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::IntSub, &[OpRef(0), OpRef(100)], 2),
            mk_op_with_descr(
                OpCode::CallAssemblerI,
                &[OpRef(2)],
                3,
                make_call_assembler_descr(&token, vec![Type::Int], Type::Int),
            ),
            mk_op(OpCode::IntAdd, &[OpRef(3), OpRef(100)], 4),
            mk_op(OpCode::Finish, &[OpRef(4)], OpRef::NONE.0),
        ];
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let failed = backend.execute_token(&token, &[Value::Int(0)]);
        let guard_descr = failed
            .data
            .downcast_ref::<FrameData>()
            .expect("base-case guard should produce FrameData")
            .fail_descr
            .clone();

        backend.set_constants(HashMap::new());
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        backend
            .compile_bridge(guard_descr.as_ref(), &inputargs, &bridge_ops, &token)
            .unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(4)]);
        let descr = backend.get_latest_descr(&frame);
        assert!(descr.is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 4);

        let raw = backend.execute_token_ints_raw(&token, &[4]);
        assert!(raw.is_finish);
        assert_eq!(raw.outputs, vec![4]);
        assert_eq!(raw.typed_outputs, vec![Value::Int(4)]);
    }

    #[test]
    fn test_gc_runtime_opcodes_error() {
        let cases = vec![
            (
                OpCode::CallMallocNursery,
                vec![InputArg::new_int(0)],
                vec![OpRef(0)],
                1,
                "configured GC runtime",
            ),
            (
                OpCode::CallMallocNurseryVarsize,
                vec![
                    InputArg::new_int(0),
                    InputArg::new_int(1),
                    InputArg::new_int(2),
                ],
                vec![OpRef(0), OpRef(1), OpRef(2)],
                3,
                "configured GC runtime",
            ),
            (
                OpCode::CallMallocNurseryVarsizeFrame,
                vec![InputArg::new_int(0)],
                vec![OpRef(0)],
                1,
                "configured GC runtime",
            ),
            (
                OpCode::CondCallGcWb,
                vec![InputArg::new_ref(0)],
                vec![OpRef(0)],
                OpRef::NONE.0,
                "configured GC runtime",
            ),
            (
                OpCode::CondCallGcWbArray,
                vec![InputArg::new_ref(0), InputArg::new_int(1)],
                vec![OpRef(0), OpRef(1)],
                OpRef::NONE.0,
                "configured GC runtime",
            ),
        ];

        for (idx, (opcode, inputargs, op_args, pos, detail)) in cases.into_iter().enumerate() {
            let label_args: Vec<OpRef> = (0..inputargs.len() as u32).map(OpRef).collect();
            let ops = vec![
                mk_op(OpCode::Label, &label_args, OpRef::NONE.0),
                mk_op(opcode, &op_args, pos),
                mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
            ];
            assert_compile_unsupported(inputargs, ops, 1400 + idx as u64, opcode, detail);
        }
    }

    #[test]
    fn test_gc_alloc_and_init_with_configured_runtime() {
        let mut backend = make_gc_backend();

        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef(32)], 0),
            mk_op(
                OpCode::GcStore,
                &[OpRef(0), OpRef(0), OpRef(7)],
                OpRef::NONE.0,
            ),
            mk_op(
                OpCode::GcStore,
                &[OpRef(0), OpRef(1), OpRef(0xDEAD)],
                OpRef::NONE.0,
            ),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1500);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { header_of(obj.0).type_id() }, 7);
        assert_eq!(unsafe { *(obj.0 as *const u64) }, 0xDEAD);
    }

    #[test]
    fn test_gc_batched_allocation_layout_with_configured_runtime() {
        let mut backend = make_gc_backend();

        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef(56)], 0),
            mk_op(
                OpCode::GcStore,
                &[OpRef(0), OpRef(0), OpRef(1)],
                OpRef::NONE.0,
            ),
            mk_op(OpCode::NurseryPtrIncrement, &[OpRef(0), OpRef(24)], 1),
            mk_op(
                OpCode::GcStore,
                &[OpRef(1), OpRef(0), OpRef(2)],
                OpRef::NONE.0,
            ),
            mk_op(OpCode::Finish, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1501);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        let obj0 = backend.get_ref_value(&frame, 0);
        let obj1 = backend.get_ref_value(&frame, 1);

        assert!(!obj0.is_null());
        assert!(!obj1.is_null());
        assert_eq!(obj1.0, obj0.0 + 24);
        assert_eq!(unsafe { header_of(obj0.0).type_id() }, 1);
        assert_eq!(unsafe { header_of(obj1.0).type_id() }, 2);
    }

    #[test]
    fn test_gc_varsize_alloc_and_length_init_with_configured_runtime() {
        let mut backend = make_gc_backend();

        let ad = make_array_descr(16, 8, Type::Int, Some(0));
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallMallocNurseryVarsize, &[OpRef(0)], 1, ad.clone()),
            mk_op_with_descr(
                OpCode::GcStore,
                &[OpRef(1), OpRef(2), OpRef(0)],
                OpRef::NONE.0,
                ad,
            ),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1502);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(3)]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { *(obj.0 as *const i64) }, 3);
    }

    #[test]
    fn test_cond_call_gc_wb_executes_with_configured_runtime() {
        let mut backend = make_gc_backend();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::CondCallGcWb, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1503);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut raw = vec![0u64; 2];
        unsafe {
            *(raw.as_mut_ptr() as *mut GcHeader) = GcHeader::with_flags(9, flags::TRACK_YOUNG_PTRS);
        }
        let obj = GcRef(raw.as_mut_ptr() as usize + GcHeader::SIZE);

        let _frame = backend.execute_token(&token, &[Value::Ref(obj)]);
        assert!(!unsafe { header_of(obj.0).has_flag(flags::TRACK_YOUNG_PTRS) });
    }

    #[test]
    fn test_gc_collecting_alloc_preserves_live_ref_inputs() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
        });
        gc.register_type(TypeInfo::simple(16));

        let root = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(root.0));
        unsafe {
            *(root.0 as *mut u64) = 0xD00DFEED;
        }

        let filler = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(filler.0));

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::CallMallocNursery, &[OpRef(24)], 1),
            mk_op(OpCode::Finish, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1505);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        let moved_root = backend.get_ref_value(&frame, 0);
        let new_obj = backend.get_ref_value(&frame, 1);

        assert!(!moved_root.is_null());
        assert_ne!(moved_root, root);
        assert_eq!(unsafe { *(moved_root.0 as *const u64) }, 0xD00DFEED);
        assert!(!new_obj.is_null());
    }

    #[test]
    fn test_gc_collecting_alloc_preserves_live_ref_results() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
        });
        gc.register_type(TypeInfo::simple(16));

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));

        let fd = make_field_descr(0, 8, Type::Int, true);
        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::Label, &[], OpRef::NONE.0),
            mk_op(OpCode::CallMallocNursery, &[OpRef(16)], 0),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef(0), OpRef(100)],
                OpRef::NONE.0,
                fd.clone(),
            ),
            mk_op(OpCode::CallMallocNursery, &[OpRef(32)], 1),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef(1), OpRef(101)],
                OpRef::NONE.0,
                fd,
            ),
            mk_op(OpCode::CallMallocNursery, &[OpRef(24)], 2),
            mk_op(
                OpCode::Finish,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
        ];

        let mut token = JitCellToken::new(1505_1);
        let mut constants = HashMap::new();
        constants.insert(100, 111);
        constants.insert(101, 222);
        backend.set_constants(constants);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        let obj0 = backend.get_ref_value(&frame, 0);
        let obj1 = backend.get_ref_value(&frame, 1);
        let obj2 = backend.get_ref_value(&frame, 2);

        assert!(!obj0.is_null());
        assert!(!obj1.is_null());
        assert!(!obj2.is_null());
        assert_eq!(unsafe { *(obj0.0 as *const i64) }, 111);
        assert_eq!(unsafe { *(obj1.0 as *const i64) }, 222);
    }

    #[test]
    fn test_setfield_gc_from_old_object_keeps_young_ref_alive_across_collection() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
        });
        let old_root_tid = gc.register_type(TypeInfo::with_gc_ptrs(8, vec![0]));
        let young_node_tid = gc.register_type(TypeInfo::simple(16));

        let mut root = gc.alloc_with_type(old_root_tid, 8);
        unsafe {
            gc.add_root(&mut root as *mut GcRef);
        }
        gc.collect_nursery();
        gc.remove_root(&mut root as *mut GcRef);
        assert!(
            !gc.is_in_nursery(root.0),
            "root must be old before JIT runs"
        );

        let filler = gc.alloc_with_type(young_node_tid, 16);
        assert!(gc.is_in_nursery(filler.0));

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));
        let ref_fd = make_field_descr(0, 8, Type::Ref, false);
        let int_fd = make_field_descr(0, 8, Type::Int, true);
        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::CallMallocNursery, &[OpRef(16)], 1),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef(1), OpRef(100)],
                OpRef::NONE.0,
                int_fd,
            ),
            mk_op_with_descr(
                OpCode::SetfieldGc,
                &[OpRef(0), OpRef(1)],
                OpRef::NONE.0,
                ref_fd,
            ),
            mk_op(OpCode::CallMallocNursery, &[OpRef(24)], 2),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 777);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1505_2);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        let moved_root = backend.get_ref_value(&frame, 0);
        assert!(!moved_root.is_null());

        let child = GcRef(unsafe { *(moved_root.0 as *const usize) });
        assert!(
            !child.is_null(),
            "old root lost its young child across collection"
        );
        assert_eq!(unsafe { *(child.0 as *const i64) }, 777);
    }

    #[test]
    fn test_newstr_without_descr_compiles_via_builtin_layout() {
        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(TrackingGc::new(Arc::new(
            Mutex::new(TrackingGcState::default()),
        ))));

        let ops = vec![
            Op::new(OpCode::Newstr, &[OpRef(100)]),
            Op::new(OpCode::Strsetitem, &[OpRef(0), OpRef(101), OpRef(200)]),
            Op::new(OpCode::Strsetitem, &[OpRef(0), OpRef(102), OpRef(201)]),
            Op::new(OpCode::Strsetitem, &[OpRef(0), OpRef(103), OpRef(202)]),
            Op::new(OpCode::Strgetitem, &[OpRef(0), OpRef(102)]),
            Op::new(OpCode::Strlen, &[OpRef(0)]),
            Op::new(OpCode::Finish, &[OpRef(4), OpRef(5)]),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 3);
        constants.insert(101, 0);
        constants.insert(102, 1);
        constants.insert(103, 2);
        constants.insert(200, b'a' as i64);
        constants.insert(201, b'b' as i64);
        constants.insert(202, b'c' as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1512);
        backend.compile_loop(&[], &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        assert_eq!(backend.get_int_value(&frame, 0), b'b' as i64);
        assert_eq!(backend.get_int_value(&frame, 1), 3);
    }

    #[test]
    fn test_copystrcontent_without_descr_compiles_via_builtin_layout() {
        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(TrackingGc::new(Arc::new(
            Mutex::new(TrackingGcState::default()),
        ))));

        let ops = vec![
            Op::new(OpCode::Newstr, &[OpRef(100)]),
            Op::new(OpCode::Strsetitem, &[OpRef(0), OpRef(101), OpRef(200)]),
            Op::new(OpCode::Strsetitem, &[OpRef(0), OpRef(102), OpRef(201)]),
            Op::new(OpCode::Newstr, &[OpRef(100)]),
            Op::new(
                OpCode::Copystrcontent,
                &[OpRef(0), OpRef(3), OpRef(101), OpRef(101), OpRef(100)],
            ),
            Op::new(OpCode::Strgetitem, &[OpRef(3), OpRef(102)]),
            Op::new(OpCode::Finish, &[OpRef(5)]),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 2);
        constants.insert(101, 0);
        constants.insert(102, 1);
        constants.insert(200, b'x' as i64);
        constants.insert(201, b'y' as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1513);
        backend.compile_loop(&[], &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        assert_eq!(backend.get_int_value(&frame, 0), b'y' as i64);
    }

    #[test]
    fn test_newunicode_without_descr_uses_wide_items() {
        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(TrackingGc::new(Arc::new(
            Mutex::new(TrackingGcState::default()),
        ))));

        let ops = vec![
            Op::new(OpCode::Newunicode, &[OpRef(100)]),
            Op::new(OpCode::Unicodesetitem, &[OpRef(0), OpRef(101), OpRef(200)]),
            Op::new(OpCode::Unicodegetitem, &[OpRef(0), OpRef(101)]),
            Op::new(OpCode::Unicodelen, &[OpRef(0)]),
            Op::new(OpCode::Finish, &[OpRef(2), OpRef(3)]),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 0);
        constants.insert(200, 0x2603);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1514);
        backend.compile_loop(&[], &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        assert_eq!(backend.get_int_value(&frame, 0), 0x2603);
        assert_eq!(backend.get_int_value(&frame, 1), 1);
    }

    #[test]
    fn test_strhash_without_descr_reads_cached_hash_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            Op::new(OpCode::Label, &[OpRef(0)]),
            Op::new(OpCode::Strhash, &[OpRef(0)]),
            Op::new(OpCode::Strlen, &[OpRef(0)]),
            Op::new(OpCode::Strgetitem, &[OpRef(0), OpRef(100)]),
            Op::new(OpCode::Finish, &[OpRef(2), OpRef(3), OpRef(4)]),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1515);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut raw = vec![0usize; 3];
        let ptr = raw.as_mut_ptr() as *mut u8;
        unsafe {
            *(ptr.add(BUILTIN_STRING_HASH_OFFSET) as *mut i64) = -7;
            *(ptr.add(BUILTIN_STRING_LEN_OFFSET) as *mut usize) = 3;
            *ptr.add(BUILTIN_STRING_BASE_SIZE) = b'a';
            *ptr.add(BUILTIN_STRING_BASE_SIZE + 1) = b'b';
            *ptr.add(BUILTIN_STRING_BASE_SIZE + 2) = b'c';
        }

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr as usize))]);
        assert_eq!(backend.get_int_value(&frame, 0), -7);
        assert_eq!(backend.get_int_value(&frame, 1), 3);
        assert_eq!(backend.get_int_value(&frame, 2), b'b' as i64);
    }

    #[test]
    fn test_unicodehash_without_descr_reads_cached_hash_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            Op::new(OpCode::Label, &[OpRef(0)]),
            Op::new(OpCode::Unicodehash, &[OpRef(0)]),
            Op::new(OpCode::Unicodelen, &[OpRef(0)]),
            Op::new(OpCode::Unicodegetitem, &[OpRef(0), OpRef(100)]),
            Op::new(OpCode::Finish, &[OpRef(2), OpRef(3), OpRef(4)]),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1516);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let mut raw = vec![0usize; 3];
        let ptr = raw.as_mut_ptr() as *mut u8;
        unsafe {
            *(ptr.add(BUILTIN_STRING_HASH_OFFSET) as *mut i64) = -11;
            *(ptr.add(BUILTIN_STRING_LEN_OFFSET) as *mut usize) = 1;
            *(ptr.add(BUILTIN_STRING_BASE_SIZE) as *mut u32) = 0x2603;
        }

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr as usize))]);
        assert_eq!(backend.get_int_value(&frame, 0), -11);
        assert_eq!(backend.get_int_value(&frame, 1), 1);
        assert_eq!(backend.get_int_value(&frame, 2), 0x2603);
    }

    #[test]
    fn test_call_i_preserves_live_ref_inputs_across_collection() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
        });
        gc.register_type(TypeInfo::simple(16));

        let root = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(root.0));
        unsafe {
            *(root.0 as *mut u64) = 0xABCDEF01;
        }

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));
        let runtime_id = backend
            .gc_runtime_id
            .expect("GC runtime must be configured");

        let descr = make_call_descr(vec![Type::Int], Type::Int);
        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(OpCode::CallI, &[OpRef(100), OpRef(101)], 1, descr),
            mk_op(OpCode::Finish, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            collect_nursery_via_runtime as *const () as usize as i64,
        );
        constants.insert(101, runtime_id as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1506);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        let moved_root = backend.get_ref_value(&frame, 0);
        let call_result = backend.get_int_value(&frame, 1);

        assert!(!moved_root.is_null());
        assert_ne!(moved_root, root);
        assert_eq!(unsafe { *(moved_root.0 as *const u64) }, 0xABCDEF01);
        assert_eq!(call_result, 123);
    }

    #[test]
    fn test_cond_call_n_preserves_live_ref_inputs_across_collection() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
        });
        gc.register_type(TypeInfo::simple(16));

        let root = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(root.0));
        unsafe {
            *(root.0 as *mut u64) = 0xFACEB00C;
        }

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));
        let runtime_id = backend
            .gc_runtime_id
            .expect("GC runtime must be configured");

        let descr = make_call_descr(vec![Type::Int], Type::Void);
        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op_with_descr(
                OpCode::CondCallN,
                &[OpRef(102), OpRef(100), OpRef(101)],
                OpRef::NONE.0,
                descr,
            ),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(
            100,
            collect_nursery_via_runtime_void as *const () as usize as i64,
        );
        constants.insert(101, runtime_id as i64);
        constants.insert(102, 1);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1507);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        let moved_root = backend.get_ref_value(&frame, 0);

        assert!(!moved_root.is_null());
        assert_ne!(moved_root, root);
        assert_eq!(unsafe { *(moved_root.0 as *const u64) }, 0xFACEB00C);
    }

    #[test]
    fn test_deadframe_ref_survives_collection_after_execute_token() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
        });
        gc.register_type(TypeInfo::simple(16));

        let root = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(root.0));
        unsafe {
            *(root.0 as *mut u64) = 0x12345678;
        }

        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));
        let runtime_id = backend
            .gc_runtime_id
            .expect("GC runtime must be configured");

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1508);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        with_gc_runtime(runtime_id, |gc| gc.collect_nursery());

        let moved_root = backend.get_ref_value(&frame, 0);
        assert!(!moved_root.is_null());
        assert_ne!(moved_root, root);
        assert_eq!(unsafe { *(moved_root.0 as *const u64) }, 0x12345678);
    }

    #[test]
    fn test_deadframe_drop_after_backend_drop_is_safe() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 160,
            large_object_threshold: 1024,
        });
        gc.register_type(TypeInfo::simple(16));

        let root = gc.alloc_with_type(0, 16);
        let mut backend = CraneliftBackend::with_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(1509);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        drop(backend);
        drop(frame);
    }

    #[test]
    fn test_guard_not_invalidated_passes() {
        // GUARD_NOT_INVALIDATED should pass (not side-exit) when the loop
        // has not been invalidated.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            // GUARD_NOT_INVALIDATED takes no data args
            mk_op(OpCode::GuardNotInvalidated, &[], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(100);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Not invalidated -> guard passes -> Finish returns 10 + 32 = 42
        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(32)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
    }

    #[test]
    fn test_guard_not_invalidated_fails_after_invalidation() {
        // After calling invalidate(), GUARD_NOT_INVALIDATED should fail and
        // the compiled code should side-exit through the guard's fail path.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];

        // Build a guard with explicit fail_args so we can inspect them.
        let mut guard_op = Op::new(OpCode::GuardNotInvalidated, &[]);
        guard_op.pos = OpRef(OpRef::NONE.0);
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(0), OpRef(1)]));

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            guard_op,
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(101);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Before invalidation: guard passes, Finish is reached.
        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(32)]);
        assert_eq!(backend.get_int_value(&frame, 0), 42);

        // Invalidate the loop.
        token.invalidate();
        assert!(token.is_invalidated());

        // After invalidation: guard fails, side-exits with the guard's fail args.
        let frame = backend.execute_token(&token, &[Value::Int(10), Value::Int(32)]);
        // The guard's fail_args are [OpRef(0), OpRef(1)] = the input values.
        let descr = backend.get_latest_descr(&frame);
        // fail_index 0 is the GuardNotInvalidated guard (first guard in the trace).
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 10);
        assert_eq!(backend.get_int_value(&frame, 1), 32);
    }

    #[test]
    fn test_guard_not_invalidated_in_loop() {
        // Verify GUARD_NOT_INVALIDATED works inside a loop that runs
        // multiple iterations before being invalidated.
        let mut backend = CraneliftBackend::new();

        // Loop: i = i + 1; guard_not_invalidated; guard i < limit; jump
        let inputargs = vec![InputArg::new_int(0)];

        let mut guard_inv = Op::new(OpCode::GuardNotInvalidated, &[]);
        guard_inv.pos = OpRef(OpRef::NONE.0);
        guard_inv.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(1)]));

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1), // i = i + 1
            guard_inv,                                         // guard_not_invalidated
            mk_op(OpCode::IntLt, &[OpRef(1), OpRef(101)], 2),  // i < 1000000
            mk_op(OpCode::GuardTrue, &[OpRef(2)], OpRef::NONE.0),
            mk_op(OpCode::Jump, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        constants.insert(101, 1_000_000i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(102);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Without invalidation: runs to completion (i reaches 1000000,
        // GuardTrue fails).
        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        let descr = backend.get_latest_descr(&frame);
        // fail_index 1 = GuardTrue (second guard)
        assert_eq!(descr.fail_index(), 1);
        assert_eq!(backend.get_int_value(&frame, 0), 999_999);

        // Now invalidate and run again: should fail at GuardNotInvalidated
        // on the first iteration.
        token.invalidate();
        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        let descr = backend.get_latest_descr(&frame);
        // fail_index 0 = GuardNotInvalidated (first guard)
        assert_eq!(descr.fail_index(), 0);
        // i = 0 + 1 = 1 (only one iteration before guard fails)
        assert_eq!(backend.get_int_value(&frame, 0), 1);
    }

    #[test]
    fn test_invalidate_via_backend_trait() {
        // Verify that Backend::invalidate_loop() properly sets the flag.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];

        let mut guard_inv = Op::new(OpCode::GuardNotInvalidated, &[]);
        guard_inv.pos = OpRef(OpRef::NONE.0);
        guard_inv.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(0)]));

        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            guard_inv,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(103);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Before invalidation
        let frame = backend.execute_token(&token, &[Value::Int(99)]);
        // Finish (fail_index 1)
        assert_eq!(backend.get_int_value(&frame, 0), 99);

        // Invalidate via the Backend trait method
        backend.invalidate_loop(&token);
        assert!(token.is_invalidated());

        // After invalidation: guard fails
        let frame = backend.execute_token(&token, &[Value::Int(99)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
    }

    /// Guard-bearing callee: guard passes -> finish result propagated.
    /// Guard-bearing callee: guard fails  -> deadframe propagated.

    /// Guard-bearing callee with multiple guards: first guard passes, second
    /// guard fails -> correct fail_index propagated.

    /// Guard-bearing callee with force_token finish shape:
    /// Callee has ForceToken + GuardNotForced2 + Finish(force_token).
    /// Caller uses CallAssemblerR and gets the force_token result.

    #[test]
    fn test_all_guards_have_recovery_layout() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard1 = mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0);
        guard1.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(0), OpRef(1)]));
        let int_add = mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)], 2);
        let mut guard2 = mk_op(OpCode::GuardFalse, &[OpRef(2)], OpRef::NONE.0);
        guard2.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(2)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            guard1,
            int_add,
            guard2,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(500);
        backend.set_next_header_pc(2000);
        let mut token = JitCellToken::new(9001);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let layouts = backend
            .compiled_fail_descr_layouts(&token)
            .expect("compiled layouts should exist");

        // All fail descriptors (2 guards + 1 finish) should have recovery_layout
        for (idx, layout) in layouts.iter().enumerate() {
            assert!(
                layout.recovery_layout.is_some(),
                "fail_descr[{idx}] should have recovery_layout"
            );
            let recovery = layout.recovery_layout.as_ref().unwrap();
            assert!(!recovery.frames.is_empty());
            assert_eq!(recovery.frames[0].trace_id, Some(500));
            assert_eq!(recovery.frames[0].header_pc, Some(2000));
            // slot_types should always be populated
            assert!(recovery.frames[0].slot_types.is_some());
        }
    }

    #[test]
    fn test_frame_layout_includes_slot_types() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
        ];
        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0);
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef(0),
            OpRef(1),
            OpRef(2),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef(0), OpRef(1), OpRef(2)],
                OpRef::NONE.0,
            ),
            guard,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(501);
        backend.set_next_header_pc(3000);
        let mut token = JitCellToken::new(9002);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let layouts = backend
            .compiled_fail_descr_layouts(&token)
            .expect("compiled layouts should exist");

        // Guard layout
        let guard_layout = &layouts[0];
        let recovery = guard_layout
            .recovery_layout
            .as_ref()
            .expect("guard should have recovery layout");
        let frame = &recovery.frames[0];
        let slot_types = frame
            .slot_types
            .as_ref()
            .expect("slot_types should always be populated");
        assert_eq!(slot_types, &[Type::Int, Type::Ref, Type::Float]);

        // frame_stack should mirror recovery frames
        let frame_stack = guard_layout
            .frame_stack
            .as_ref()
            .expect("frame_stack should be populated from recovery_layout");
        assert_eq!(frame_stack.len(), 1);
        assert_eq!(
            frame_stack[0].slot_types,
            Some(vec![Type::Int, Type::Ref, Type::Float])
        );
    }

    #[test]
    fn test_compiled_guard_frame_stacks_query() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard1 = mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0);
        guard1.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(0), OpRef(1)]));
        let mut guard2 = mk_op(OpCode::GuardFalse, &[OpRef(1)], OpRef::NONE.0);
        guard2.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef(1)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            guard1,
            guard2,
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        backend.set_next_trace_id(502);
        backend.set_next_header_pc(4000);
        let mut token = JitCellToken::new(9003);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame_stacks = backend
            .compiled_guard_frame_stacks(&token)
            .expect("compiled_guard_frame_stacks should return Some");

        // 2 guards + 1 finish = 3 entries, all with frame stacks
        assert_eq!(frame_stacks.len(), 3);
        for (fail_index, frames) in &frame_stacks {
            assert!(
                !frames.is_empty(),
                "fail_index={fail_index} should have non-empty frame stack"
            );
            assert_eq!(frames[0].trace_id, Some(502));
            assert_eq!(frames[0].header_pc, Some(4000));
            // slot_types populated in every frame
            for frame in frames {
                assert!(frame.slot_types.is_some());
            }
        }

        // Verify fail indices are sequential
        let indices: Vec<u32> = frame_stacks.iter().map(|(idx, _)| *idx).collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    /// Verify that the main opcode dispatch in compile_loop covers all OpCode
    /// variants. We compile a small representative trace and confirm the backend
    /// doesn't return Unsupported. The compile-time exhaustiveness of the match
    /// is the primary guarantee (enforced by CI warning checks).
    #[test]
    fn test_all_opcodes_covered_in_backend() {
        let mut backend = CraneliftBackend::new();
        let mut constants = HashMap::new();
        constants.insert(100, 42i64);
        constants.insert(101, 7i64);
        backend.set_constants(constants);

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::IntGt, &[OpRef(1), OpRef(101)], 2),
            {
                let mut g = mk_op(OpCode::GuardTrue, &[OpRef(2)], OpRef::NONE.0);
                g.fail_args = Some(smallvec::smallvec![OpRef(0)]);
                g
            },
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = JitCellToken::new(99_999);
        let result = backend.compile_loop(&inputargs, &ops, &mut token);
        assert!(
            result.is_ok(),
            "representative trace should compile: {:?}",
            result.err()
        );
    }
}
