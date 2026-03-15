use std::cell::{Cell, RefCell};
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
    AbiParam, Function, InstBuilder, MemFlags, Signature, StackSlotData, StackSlotKind,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use cranelift_codegen::ir::Value as CValue;

use majit_codegen::{AsmInfo, BackendError, DeadFrame, LoopToken};
use majit_gc::header::{GcHeader, TYPE_ID_MASK};
use majit_gc::rewrite::GcRewriterImpl;
use majit_gc::{GcAllocator, GcRewriter, WriteBarrierDescr, flags as gc_flags};
use majit_ir::{
    CallDescr, EffectInfo, FailDescr, GcRef, InputArg, OopSpecIndex, Op, OpCode, OpRef, Type, Value,
};

use crate::guard::{BridgeData, CraneliftFailDescr, FrameData};

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
    pending_may_force: Mutex<Option<PendingMayForceFrame>>,
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

static NEXT_FORCE_TOKEN_HANDLE: AtomicU64 = AtomicU64::new(1);
static FORCE_FRAMES: OnceLock<Mutex<HashMap<u64, Arc<ActiveForceFrame>>>> = OnceLock::new();

thread_local! {
    static CURRENT_FORCE_FRAME_HANDLE: Cell<u64> = const { Cell::new(0) };
    /// Thread-local exception state for JIT-compiled code.
    /// exc_value holds the current exception value (as a GcRef encoded in i64).
    /// exc_type holds the current exception class (as a GcRef encoded in i64).
    static JIT_EXC_VALUE: Cell<i64> = const { Cell::new(0) };
    static JIT_EXC_TYPE: Cell<i64> = const { Cell::new(0) };
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
    JIT_EXC_VALUE.with(|c| c.get())
}

/// Read the current exception class (returns 0 if no exception pending).
extern "C" fn jit_exc_get_type() -> i64 {
    JIT_EXC_TYPE.with(|c| c.get())
}

/// Clear the current exception state and return the value that was pending.
extern "C" fn jit_exc_clear_and_get_value() -> i64 {
    let val = JIT_EXC_VALUE.with(|c| c.replace(0));
    JIT_EXC_TYPE.with(|c| c.set(0));
    val
}

/// Clear the current exception state.
extern "C" fn jit_exc_clear() {
    JIT_EXC_VALUE.with(|c| c.set(0));
    JIT_EXC_TYPE.with(|c| c.set(0));
}

/// Return 1 if the current exception class exactly matches `expected_type`.
extern "C" fn jit_exc_type_matches(expected_type: i64) -> i64 {
    let exc_value = JIT_EXC_VALUE.with(|c| c.get());
    let exc_type = JIT_EXC_TYPE.with(|c| c.get());
    i64::from(exc_value != 0 && exc_type == expected_type)
}

/// Set the current exception state (value, class).
extern "C" fn jit_exc_restore(value: i64, exc_type: i64) {
    JIT_EXC_VALUE.with(|c| c.set(value));
    JIT_EXC_TYPE.with(|c| c.set(exc_type));
}

/// Set exception state from a call that may raise.
/// This is called by external code that wants to signal an exception
/// to the JIT-compiled code.
pub fn jit_exc_raise(value: i64, exc_type: i64) {
    JIT_EXC_VALUE.with(|c| c.set(value));
    JIT_EXC_TYPE.with(|c| c.set(exc_type));
}

/// Check if an exception is currently pending.
pub fn jit_exc_is_pending() -> bool {
    JIT_EXC_VALUE.with(|c| c.get()) != 0
}

fn take_pending_jit_exception_state() -> (i64, GcRef) {
    let value = JIT_EXC_VALUE.with(|c| c.replace(0));
    let exc_type = JIT_EXC_TYPE.with(|c| c.replace(0));
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

/// Allocation shim: allocate `size` bytes from the nursery (or fallback to malloc).
///
/// In a real GC integration this would use the bump-pointer allocator.
/// The current implementation uses a simple heap allocation as a placeholder.
extern "C" fn jit_malloc_nursery_shim(size: i64) -> i64 {
    let layout = std::alloc::Layout::from_size_align(size.max(8) as usize, 8)
        .unwrap_or(std::alloc::Layout::new::<u64>());
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    ptr as i64
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

fn with_registered_gc_roots<R>(
    runtime_id: u64,
    roots_ptr: u64,
    num_roots: u64,
    f: impl FnOnce(&mut dyn GcAllocator) -> R,
) -> R {
    with_gc_runtime(runtime_id, |gc| {
        let roots_ptr = roots_ptr as usize as *mut GcRef;
        if !roots_ptr.is_null() {
            for slot in 0..num_roots as usize {
                unsafe {
                    gc.add_root(roots_ptr.add(slot));
                }
            }
        }

        let result = f(gc);

        if !roots_ptr.is_null() {
            for slot in 0..num_roots as usize {
                unsafe {
                    gc.remove_root(roots_ptr.add(slot));
                }
            }
        }

        result
    })
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
        pending_may_force: Mutex::new(None),
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
        if let Some(pending) = frame.pending_may_force.lock().unwrap().take() {
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

fn call_assembler_registry() -> &'static Mutex<HashMap<u64, RegisteredLoopTarget>> {
    CALL_ASSEMBLER_TARGETS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn register_call_assembler_target(token: &LoopToken, compiled: &CompiledLoop) {
    call_assembler_registry().lock().unwrap().insert(
        token.number,
        RegisteredLoopTarget {
            code_ptr: compiled.code_ptr,
            fail_descrs: compiled.fail_descrs.clone(),
            gc_runtime_id: compiled.gc_runtime_id,
            num_inputs: compiled.num_inputs,
            num_ref_roots: compiled.num_ref_roots,
            max_output_slots: compiled.max_output_slots,
            inputarg_types: token.inputarg_types.clone(),
            needs_force_frame: compiled.needs_force_frame,
        },
    );
}

fn unregister_call_assembler_target(token_number: u64) {
    call_assembler_registry()
        .lock()
        .unwrap()
        .remove(&token_number);
}

fn lookup_call_assembler_target(token_number: u64) -> Option<RegisteredLoopTarget> {
    call_assembler_registry()
        .lock()
        .unwrap()
        .get(&token_number)
        .cloned()
}

fn redirect_call_assembler_target(old_number: u64, new_number: u64) {
    let Some(new_target) = lookup_call_assembler_target(new_number) else {
        return;
    };
    call_assembler_registry()
        .lock()
        .unwrap()
        .insert(old_number, new_target);
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
    assert!(
        pending_may_force.is_none(),
        "nested call_may_force in the same compiled frame is unsupported"
    );
    *pending_may_force = Some(PendingMayForceFrame {
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
        .take()
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
        if let Some(pending) = pending_may_force.as_mut() {
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

pub fn set_savedata_ref_on_deadframe(frame: &mut DeadFrame, data: GcRef) {
    if let Some(frame_data) = frame.data.downcast_mut::<FrameData>() {
        frame_data.set_savedata_ref(data);
        return;
    }
    if let Some(preview) = frame.data.downcast_mut::<PreviewFrameData>() {
        preview.frame.set_savedata_ref(data);
        set_force_frame_saved_data(&preview.active_force_frame, data);
        return;
    }
    panic!("unsupported dead frame type for saved-data");
}

pub fn get_latest_descr_from_deadframe(frame: &DeadFrame) -> &dyn FailDescr {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return frame_data.fail_descr.as_ref();
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return preview.frame.fail_descr.as_ref();
    }
    panic!("unsupported dead frame type");
}

pub fn get_int_from_deadframe(frame: &DeadFrame, index: usize) -> i64 {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return frame_data.get_int(index);
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return preview.frame.get_int(index);
    }
    panic!("unsupported dead frame type");
}

pub fn get_float_from_deadframe(frame: &DeadFrame, index: usize) -> f64 {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return frame_data.get_float(index);
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return preview.frame.get_float(index);
    }
    panic!("unsupported dead frame type");
}

pub fn get_ref_from_deadframe(frame: &DeadFrame, index: usize) -> GcRef {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return frame_data.get_ref(index);
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return preview.frame.get_ref(index);
    }
    panic!("unsupported dead frame type");
}

pub fn get_savedata_ref_from_deadframe(frame: &DeadFrame) -> GcRef {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return frame_data.get_savedata_ref();
    }
    if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
        return get_force_frame_saved_data(&preview.active_force_frame)
            .unwrap_or_else(|| preview.frame.get_savedata_ref());
    }
    panic!("unsupported dead frame type for saved-data");
}

pub fn grab_exc_value_from_deadframe(frame: &DeadFrame) -> GcRef {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return frame_data.get_exception_ref();
    }
    if frame.data.downcast_ref::<PreviewFrameData>().is_some() {
        return GcRef::NULL;
    }
    panic!("unsupported dead frame type for exception value");
}

pub fn grab_exc_class_from_deadframe(frame: &DeadFrame) -> i64 {
    if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
        return frame_data.get_exception_class();
    }
    if frame.data.downcast_ref::<PreviewFrameData>().is_some() {
        return 0;
    }
    panic!("unsupported dead frame type for exception class");
}

extern "C" fn call_assembler_shim(target_token: u64, args_ptr: u64) -> u64 {
    let target = lookup_call_assembler_target(target_token)
        .unwrap_or_else(|| panic!("missing call_assembler target token {target_token}"));
    assert_eq!(
        target.fail_descrs.len(),
        1,
        "call_assembler shim only supports finish-only callees"
    );
    assert!(
        target.fail_descrs[0].is_finish(),
        "call_assembler shim only supports finish-only callees"
    );

    let input_slice =
        unsafe { std::slice::from_raw_parts(args_ptr as usize as *const i64, target.num_inputs) };
    let (fail_index, outputs, handle, _) = run_compiled_code(
        target.code_ptr,
        &target.fail_descrs,
        target.gc_runtime_id,
        target.num_ref_roots,
        target.max_output_slots,
        input_slice,
        target.needs_force_frame,
    );
    assert_eq!(
        fail_index as usize, 0,
        "call_assembler shim received a non-finish exit from the callee"
    );
    if target.fail_descrs[0].force_token_slots.is_empty() {
        release_force_token(handle);
    }

    outputs[0] as u64
}

extern "C" fn gc_alloc_nursery_shim(
    runtime_id: u64,
    roots_ptr: u64,
    num_roots: u64,
    size: u64,
) -> u64 {
    with_registered_gc_roots(runtime_id, roots_ptr, num_roots, |gc| {
        let obj = gc.alloc_nursery(size as usize);
        // Fresh old-gen allocations need to be remembered before the rewriter
        // starts skipping write barriers on stores into the same object.
        gc.write_barrier(obj);
        obj.0 as u64
    })
}

extern "C" fn gc_alloc_varsize_shim(
    runtime_id: u64,
    roots_ptr: u64,
    num_roots: u64,
    base_size: u64,
    item_size: u64,
    length: u64,
) -> u64 {
    with_registered_gc_roots(runtime_id, roots_ptr, num_roots, |gc| {
        let obj = gc.alloc_varsize(base_size as usize, item_size as usize, length as usize);
        gc.write_barrier(obj);
        obj.0 as u64
    })
}

extern "C" fn gc_write_barrier_shim(runtime_id: u64, obj: u64) {
    with_gc_runtime(runtime_id, |gc| gc.write_barrier(GcRef(obj as usize)));
}

extern "C" fn gc_register_roots_shim(runtime_id: u64, roots_ptr: u64, num_roots: u64) {
    with_gc_runtime(runtime_id, |gc| {
        let roots_ptr = roots_ptr as usize as *mut GcRef;
        if roots_ptr.is_null() {
            return;
        }
        for slot in 0..num_roots as usize {
            unsafe {
                gc.add_root(roots_ptr.add(slot));
            }
        }
    });
}

extern "C" fn gc_unregister_roots_shim(runtime_id: u64, roots_ptr: u64, num_roots: u64) {
    with_gc_runtime(runtime_id, |gc| {
        let roots_ptr = roots_ptr as usize as *mut GcRef;
        if roots_ptr.is_null() {
            return;
        }
        for slot in 0..num_roots as usize {
            unsafe {
                gc.remove_root(roots_ptr.add(slot));
            }
        }
    });
}

fn resolve_opref(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    opref: OpRef,
) -> CValue {
    if let Some(&c) = constants.get(&opref.0) {
        return builder.ins().iconst(cl_types::I64, c);
    }
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
) -> Result<RegisteredLoopTarget, BackendError> {
    let target_token = call_descr.call_target_token().ok_or_else(|| {
        unsupported_semantics(
            opcode,
            "call-assembler descriptor must provide a compiled target token",
        )
    })?;
    let target = lookup_call_assembler_target(target_token).ok_or_else(|| {
        unsupported_semantics(
            opcode,
            "call-assembler target token is not compiled or not registered",
        )
    })?;

    if target.inputarg_types != call_descr.arg_types() {
        return Err(unsupported_semantics(
            opcode,
            "call-assembler target input types do not match the descriptor",
        ));
    }
    if target.fail_descrs.len() != 1 || !target.fail_descrs[0].is_finish() {
        return Err(unsupported_semantics(
            opcode,
            "call-assembler currently supports only finish-only callee loops",
        ));
    }
    if !target.fail_descrs[0].force_token_slots.is_empty() {
        return Err(unsupported_semantics(
            opcode,
            "call-assembler does not yet support callee finish values containing force tokens",
        ));
    }

    let finish_types = target.fail_descrs[0].fail_arg_types();
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
                return Err(unsupported_semantics(
                    opcode,
                    "call-assembler target finish result does not match the descriptor",
                ));
            }
        }
    }

    Ok(target)
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

fn build_force_token_set(inputargs: &[InputArg], ops: &[Op]) -> HashSet<u32> {
    let mut force_tokens = HashSet::new();
    for (op_idx, op) in ops.iter().enumerate() {
        if op.opcode == OpCode::ForceToken && !op.pos.is_none() {
            force_tokens.insert(op_var_index(op, op_idx, inputargs.len()) as u32);
        }
    }
    force_tokens
}

fn build_value_type_map(inputargs: &[InputArg], ops: &[Op]) -> HashMap<u32, Type> {
    let mut value_types = HashMap::new();

    for input in inputargs {
        value_types.insert(input.index, input.tp);
    }

    for (op_idx, op) in ops.iter().enumerate() {
        let result_type = op.result_type();
        if result_type != Type::Void {
            value_types.insert(
                op_var_index(op, op_idx, inputargs.len()) as u32,
                result_type,
            );
        }
    }

    value_types
}

fn build_ref_root_slots(inputargs: &[InputArg], ops: &[Op]) -> Vec<(u32, usize)> {
    let mut seen = HashSet::new();
    let mut slots = Vec::new();

    for input in inputargs {
        if input.tp == Type::Ref && seen.insert(input.index) {
            slots.push((input.index, slots.len()));
        }
    }

    for (op_idx, op) in ops.iter().enumerate() {
        if op.result_type() == Type::Ref {
            let vi = op_var_index(op, op_idx, inputargs.len()) as u32;
            if seen.insert(vi) {
                slots.push((vi, slots.len()));
            }
        }
    }

    slots
}

fn normalize_ops_for_codegen(inputargs: &[InputArg], ops: &[Op]) -> Vec<Op> {
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

fn spill_ref_roots(
    builder: &mut FunctionBuilder,
    roots_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
) {
    for &(var_idx, slot) in ref_root_slots {
        let addr = builder.ins().iadd_imm(roots_ptr, (slot * 8) as i64);
        let val = if defined_ref_vars.contains(&var_idx) {
            builder.use_var(var(var_idx))
        } else {
            builder.ins().iconst(cl_types::I64, 0)
        };
        builder.ins().store(MemFlags::trusted(), val, addr, 0);
    }
}

fn reload_ref_roots(
    builder: &mut FunctionBuilder,
    roots_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
) {
    for &(var_idx, slot) in ref_root_slots {
        if !defined_ref_vars.contains(&var_idx) {
            continue;
        }
        let addr = builder.ins().iadd_imm(roots_ptr, (slot * 8) as i64);
        let val = builder
            .ins()
            .load(cl_types::I64, MemFlags::trusted(), addr, 0);
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

fn emit_collecting_gc_call(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    call_conv: cranelift_codegen::isa::CallConv,
    roots_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
    runtime_id: CValue,
    func_ptr: usize,
    extra_args: &[CValue],
    return_type: Option<cranelift_codegen::ir::Type>,
) -> Option<CValue> {
    spill_ref_roots(builder, roots_ptr, ref_root_slots, defined_ref_vars);

    let mut args = Vec::with_capacity(extra_args.len() + 3);
    args.push(runtime_id);
    args.push(ptr_arg_as_i64(builder, roots_ptr, ptr_type));
    args.push(
        builder
            .ins()
            .iconst(cl_types::I64, ref_root_slots.len() as i64),
    );
    args.extend_from_slice(extra_args);

    let result = emit_host_call(builder, ptr_type, call_conv, func_ptr, &args, return_type);
    reload_ref_roots(builder, roots_ptr, ref_root_slots, defined_ref_vars);
    result
}

/// Emit an indirect call through a function pointer.
///
/// `op.args[0]` is the function address (as an integer/pointer).
/// `op.args[1..]` are the call arguments.
/// `call_descr` provides `arg_types()` and `result_type()`.
///
/// Float arguments are bitcast from I64 before the call, and float
/// results are bitcast back to I64 for variable storage.
fn emit_gc_root_registration(
    builder: &mut FunctionBuilder,
    ptr_type: cranelift_codegen::ir::Type,
    call_conv: cranelift_codegen::isa::CallConv,
    runtime_id: Option<u64>,
    roots_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    register: bool,
) {
    let Some(runtime_id) = runtime_id else {
        return;
    };
    if ref_root_slots.is_empty() {
        return;
    }

    let runtime_id = builder.ins().iconst(cl_types::I64, runtime_id as i64);
    let roots_ptr = ptr_arg_as_i64(builder, roots_ptr, ptr_type);
    let num_roots = builder
        .ins()
        .iconst(cl_types::I64, ref_root_slots.len() as i64);
    let func_ptr = if register {
        gc_register_roots_shim as *const () as usize
    } else {
        gc_unregister_roots_shim as *const () as usize
    };
    let _ = emit_host_call(
        builder,
        ptr_type,
        call_conv,
        func_ptr,
        &[runtime_id, roots_ptr, num_roots],
        None,
    );
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
    roots_ptr: CValue,
    ref_root_slots: &[(u32, usize)],
    defined_ref_vars: &HashSet<u32>,
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

    spill_ref_roots(builder, roots_ptr, ref_root_slots, defined_ref_vars);
    emit_gc_root_registration(
        builder,
        ptr_type,
        call_conv,
        gc_runtime_id,
        roots_ptr,
        ref_root_slots,
        true,
    );
    let call = builder.ins().call_indirect(sig_ref, func_ptr, &args);
    emit_gc_root_registration(
        builder,
        ptr_type,
        call_conv,
        gc_runtime_id,
        roots_ptr,
        ref_root_slots,
        false,
    );
    reload_ref_roots(builder, roots_ptr, ref_root_slots, defined_ref_vars);

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

/// Emit a guard side-exit: store fail args to outputs_ptr and return fail_index.
fn emit_guard_exit(
    builder: &mut FunctionBuilder,
    constants: &HashMap<u32, i64>,
    outputs_ptr: CValue,
    info: &GuardInfo,
) {
    for (slot, &arg_ref) in info.fail_arg_refs.iter().enumerate() {
        let val = resolve_opref(builder, constants, arg_ref);
        let offset = (slot as i32) * 8;
        let addr = builder.ins().iadd_imm(outputs_ptr, offset as i64);
        builder.ins().store(MemFlags::trusted(), val, addr, 0);
    }
    let idx_val = builder.ins().iconst(cl_types::I64, info.fail_index as i64);
    builder.ins().return_(&[idx_val]);
}

// ---------------------------------------------------------------------------
// Compiled loop data
// ---------------------------------------------------------------------------

struct CompiledLoop {
    _func_id: FuncId,
    code_ptr: *const u8,
    code_size: usize,
    fail_descrs: Vec<Arc<CraneliftFailDescr>>,
    gc_runtime_id: Option<u64>,
    num_inputs: usize,
    num_ref_roots: usize,
    max_output_slots: usize,
    /// Whether any guard in this loop uses FORCE_TOKEN slots.
    /// When false, force frame registration can be skipped entirely.
    needs_force_frame: bool,
}

unsafe impl Send for CompiledLoop {}

fn run_compiled_code(
    code_ptr: *const u8,
    fail_descrs: &[Arc<CraneliftFailDescr>],
    gc_runtime_id: Option<u64>,
    num_ref_roots: usize,
    max_output_slots: usize,
    inputs: &[i64],
    needs_force_frame: bool,
) -> (u32, Vec<i64>, u64, Option<Arc<ActiveForceFrame>>) {
    let mut outputs = vec![0i64; max_output_slots.max(1)];
    let mut roots = vec![GcRef::NULL; num_ref_roots];
    let func: unsafe extern "C" fn(*const i64, *mut i64, *mut i64) -> i64 =
        unsafe { std::mem::transmute(code_ptr) };

    let _jitted_guard = majit_codegen::JittedGuard::enter();

    let (handle, force_frame) = if needs_force_frame {
        let (h, f) = register_force_frame(fail_descrs, gc_runtime_id);
        (h, Some(f))
    } else {
        (0, None)
    };

    let fail_index = with_active_force_frame(handle, || unsafe {
        func(
            inputs.as_ptr(),
            outputs.as_mut_ptr(),
            roots.as_mut_ptr() as *mut i64,
        )
    }) as u32;
    drop(_jitted_guard);
    (fail_index, outputs, handle, force_frame)
}

struct GuardInfo {
    fail_index: u32,
    fail_arg_refs: Vec<OpRef>,
}

fn infer_fail_arg_types(
    fail_arg_refs: &[OpRef],
    value_types: &HashMap<u32, Type>,
) -> Result<Vec<Type>, BackendError> {
    let mut fail_arg_types = Vec::with_capacity(fail_arg_refs.len());
    for &opref in fail_arg_refs {
        if opref.is_none() {
            return Err(BackendError::Unsupported(
                "guard/finish fail args cannot contain OpRef::NONE".to_string(),
            ));
        }
        // Backend constant slots are currently integer-only. If a fail arg
        // doesn't correspond to an input arg or operation result, treat it as
        // an integer constant instead of silently manufacturing Ref/Float data.
        fail_arg_types.push(value_types.get(&opref.0).copied().unwrap_or(Type::Int));
    }
    Ok(fail_arg_types)
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
        }
    }

    pub fn with_gc_allocator(gc: Box<dyn GcAllocator>) -> Self {
        let mut backend = Self::new();
        backend.set_gc_allocator(gc);
        backend
    }

    pub fn set_gc_allocator(&mut self, gc: Box<dyn GcAllocator>) {
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
        }))
    }

    fn prepare_ops_for_compile(&self, inputargs: &[InputArg], ops: &[Op]) -> Vec<Op> {
        let mut normalized = normalize_ops_for_codegen(inputargs, ops);
        inject_builtin_string_descrs(&mut normalized);
        if let Some(rewriter) = self.gc_rewriter() {
            rewriter.rewrite_for_gc(&normalized)
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
        let (fail_index, outputs, handle, force_frame) = run_compiled_code(
            compiled.code_ptr,
            &compiled.fail_descrs,
            compiled.gc_runtime_id,
            compiled.num_ref_roots,
            compiled.max_output_slots,
            inputs,
            compiled.needs_force_frame,
        );

        let fail_descr = &compiled.fail_descrs[fail_index as usize];

        // Increment guard failure count.
        fail_descr.increment_fail_count();

        // If a bridge is attached to this guard, execute it.
        let bridge_guard = fail_descr.bridge.lock().unwrap();
        if let Some(ref bridge) = *bridge_guard {
            release_force_token(handle);
            return Self::execute_bridge(bridge, &outputs, &fail_descr.fail_arg_types);
        }
        drop(bridge_guard);

        let saved_data = if let Some(ref ff) = force_frame {
            take_force_frame_saved_data(ff)
        } else {
            None
        };
        let (exception_class, exception) = take_pending_jit_exception_state();
        if fail_descr.force_token_slots.is_empty() {
            release_force_token(handle);
        }

        DeadFrame {
            data: Box::new(FrameData::new_with_savedata_and_exception(
                outputs,
                fail_descr.clone(),
                compiled.gc_runtime_id,
                saved_data,
                exception_class,
                (!exception.is_null()).then_some(exception),
            )),
        }
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

        let fail_descr = &bridge.fail_descrs[fail_index as usize];
        fail_descr.increment_fail_count();

        // Check for chained bridges.
        let bridge_guard = fail_descr.bridge.lock().unwrap();
        if let Some(ref next_bridge) = *bridge_guard {
            release_force_token(handle);
            return Self::execute_bridge(next_bridge, &outputs, &fail_descr.fail_arg_types);
        }
        drop(bridge_guard);

        let saved_data = if let Some(ref ff) = force_frame {
            take_force_frame_saved_data(ff)
        } else {
            None
        };
        let (exception_class, exception) = take_pending_jit_exception_state();
        if fail_descr.force_token_slots.is_empty() {
            release_force_token(handle);
        }

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
    ) -> Result<CompiledLoop, BackendError> {
        let prepared_ops = self.prepare_ops_for_compile(inputargs, ops);
        let ops = prepared_ops.as_slice();
        let trace_id = self.next_trace_id.take().unwrap_or_else(|| {
            let trace_id = self.trace_counter;
            self.trace_counter += 1;
            trace_id
        });
        let ptr_type = self.module.target_config().pointer_type();
        let call_conv = self.module.target_config().default_call_conv;

        let mut sig = Signature::new(call_conv);
        sig.params.push(AbiParam::new(ptr_type));
        sig.params.push(AbiParam::new(ptr_type));
        sig.params.push(AbiParam::new(ptr_type));
        sig.returns.push(AbiParam::new(cl_types::I64));

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
        let mut fail_descrs: Vec<Arc<CraneliftFailDescr>> = Vec::new();
        let mut guard_infos: Vec<GuardInfo> = Vec::new();
        let mut max_output_slots: usize = 0;
        collect_guards(
            ops,
            inputargs,
            &mut fail_descrs,
            &mut guard_infos,
            &mut max_output_slots,
            trace_id,
        )?;

        let num_inputs = inputargs.len();
        let known_values = build_known_values_set(inputargs, ops);
        let value_types = build_value_type_map(inputargs, ops);
        let ref_root_slots = build_ref_root_slots(inputargs, ops);
        let gc_runtime_id = self.gc_runtime_id;
        let mut defined_ref_vars: HashSet<u32> = inputargs
            .iter()
            .filter(|input| input.tp == Type::Ref)
            .map(|input| input.index)
            .collect();

        // Take constants out of self to avoid borrow conflicts with func_ctx
        let constants = std::mem::take(&mut self.constants);

        let mut builder = FunctionBuilder::new(&mut func, &mut self.func_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let inputs_ptr = builder.block_params(entry_block)[0];
        let outputs_ptr = builder.block_params(entry_block)[1];
        let roots_ptr = builder.block_params(entry_block)[2];

        // Declare variables for inputs
        for i in 0..num_inputs {
            builder.declare_var(var(i as u32), cl_types::I64);
        }
        // Declare variables for op results
        for (op_idx, op) in ops.iter().enumerate() {
            if op.result_type() != Type::Void {
                let vi = op_var_index(op, op_idx, num_inputs);
                builder.declare_var(var(vi as u32), cl_types::I64);
            }
        }

        // Load inputs from input buffer
        for i in 0..num_inputs {
            let offset = (i as i32) * 8;
            let addr = builder.ins().iadd_imm(inputs_ptr, offset as i64);
            let val = builder
                .ins()
                .load(cl_types::I64, MemFlags::trusted(), addr, 0);
            builder.def_var(var(i as u32), val);
        }

        // Find LABEL
        let label_idx = ops.iter().position(|op| op.opcode == OpCode::Label);

        // Loop header block
        let loop_block = builder.create_block();
        for _ in 0..num_inputs {
            builder.append_block_param(loop_block, cl_types::I64);
        }

        // Jump entry -> loop
        {
            let vals: Vec<CValue> = (0..num_inputs)
                .map(|i| builder.use_var(var(i as u32)))
                .collect();
            builder.ins().jump(loop_block, &vals);
        }

        builder.switch_to_block(loop_block);
        for i in 0..num_inputs {
            let param = builder.block_params(loop_block)[i];
            builder.def_var(var(i as u32), param);
        }

        // Emit body
        let body_start = label_idx.map_or(0, |i| i + 1);
        let mut guard_idx: usize = 0;
        let mut last_ovf_flag: Option<CValue> = None;

        for op_idx in body_start..ops.len() {
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
                    builder.ins().jump(merge_block, &[no_ovf]);

                    // b != 0 path: check r / b != a
                    builder.switch_to_block(div_block);
                    builder.seal_block(div_block);
                    let div = builder.ins().sdiv(r, b);
                    let div_ne_a = builder.ins().icmp(IntCC::NotEqual, div, a);
                    let ovf_ext = builder.ins().uextend(cl_types::I64, div_ne_a);
                    builder.ins().jump(merge_block, &[ovf_ext]);

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
                    let a = resolve_opref(&mut builder, &constants, op.arg(0));
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

                OpCode::GuardClass | OpCode::GuardNonnullClass => {
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

                OpCode::GuardNoException => {
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    // Call jit_exc_get_value() → if non-zero, exception pending → exit
                    let exc_val = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_exc_get_value as *const () as usize,
                        &[],
                        Some(cl_types::I64),
                    )
                    .expect("jit_exc_get_value must return a value");
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
                    // Side-exit if overflow DID occur (ovf != 0).
                    let info = &guard_infos[guard_idx];
                    guard_idx += 1;

                    let ovf = last_ovf_flag
                        .take()
                        .expect("GuardNoOverflow without preceding overflow op");
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
                    if op_idx == 0 || !ops[op_idx - 1].opcode.is_call_may_force() {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "guard_not_forced currently requires an immediately preceding call_may_force",
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
                        roots_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                    ) {
                        builder.def_var(var(vi), result);
                    }
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
                    let target = resolve_call_assembler_target(op.opcode, call_descr)?;
                    if op.args.len() != call_descr.arg_types().len() {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "call-assembler argument count does not match the descriptor",
                        ));
                    }

                    let arg_bytes = (target.num_inputs.max(1) * 8) as u32;
                    let args_slot = builder.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        arg_bytes,
                        3,
                    ));
                    for (index, &arg_ref) in op.args.iter().enumerate() {
                        let raw = resolve_opref(&mut builder, &constants, arg_ref);
                        builder
                            .ins()
                            .stack_store(raw, args_slot, (index * 8) as i32);
                    }
                    let args_ptr = builder.ins().stack_addr(ptr_type, args_slot, 0);
                    let target_token = builder.ins().iconst(
                        cl_types::I64,
                        call_descr.call_target_token().unwrap() as i64,
                    );
                    let args_ptr_i64 = ptr_arg_as_i64(&mut builder, args_ptr, ptr_type);

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

                    spill_ref_roots(&mut builder, roots_ptr, &ref_root_slots, &defined_ref_vars);
                    emit_gc_root_registration(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        gc_runtime_id,
                        roots_ptr,
                        &ref_root_slots,
                        true,
                    );
                    let result = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        call_assembler_shim as *const () as usize,
                        &[target_token, args_ptr_i64],
                        Some(cl_types::I64),
                    );
                    emit_gc_root_registration(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        gc_runtime_id,
                        roots_ptr,
                        &ref_root_slots,
                        false,
                    );
                    reload_ref_roots(&mut builder, roots_ptr, &ref_root_slots, &defined_ref_vars);

                    if op.result_type() != Type::Void {
                        let result = result.expect("call_assembler shim must return a value");
                        builder.def_var(var(vi), result);
                    }
                }

                OpCode::CallMayForceI
                | OpCode::CallMayForceR
                | OpCode::CallMayForceF
                | OpCode::CallMayForceN => {
                    let next_op = ops.get(op_idx + 1).ok_or_else(|| {
                        unsupported_semantics(
                            op.opcode,
                            "call_may_force must be followed by guard_not_forced",
                        )
                    })?;
                    if next_op.opcode != OpCode::GuardNotForced {
                        return Err(unsupported_semantics(
                            op.opcode,
                            "call_may_force currently requires an immediately following guard_not_forced",
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
                        let raw = if arg_ref.0 == vi {
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
                        roots_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                    ) {
                        builder.def_var(var(vi), result);
                    }
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
                    spill_ref_roots(&mut builder, roots_ptr, &ref_root_slots, &defined_ref_vars);
                    emit_gc_root_registration(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        gc_runtime_id,
                        roots_ptr,
                        &ref_root_slots,
                        true,
                    );

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

                    // Unregister roots and reload
                    emit_gc_root_registration(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        gc_runtime_id,
                        roots_ptr,
                        &ref_root_slots,
                        false,
                    );
                    reload_ref_roots(&mut builder, roots_ptr, &ref_root_slots, &defined_ref_vars);

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
                                roots_ptr,
                                &ref_root_slots,
                                &defined_ref_vars,
                            );
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
                    builder
                        .ins()
                        .brif(is_zero, cont_block, &[cond], call_block, &[]);

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
                                roots_ptr,
                                &ref_root_slots,
                                &defined_ref_vars,
                            ) {
                                call_result = result;
                            }
                        }
                    }

                    builder.ins().jump(cont_block, &[call_result]);
                    builder.switch_to_block(cont_block);
                    builder.seal_block(cont_block);

                    let phi = builder.block_params(cont_block)[0];
                    builder.def_var(var(vi), phi);
                }

                // ── GC allocation calls ──
                OpCode::CallMallocNursery => {
                    let runtime_id = gc_runtime_id.ok_or_else(|| missing_gc_runtime(op.opcode))?;
                    let runtime_id = builder.ins().iconst(cl_types::I64, runtime_id as i64);
                    let size_total =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(0));
                    let size = builder.ins().iadd_imm(size_total, -(GcHeader::SIZE as i64));
                    let result = emit_collecting_gc_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        roots_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        runtime_id,
                        gc_alloc_nursery_shim as *const () as usize,
                        &[size],
                        Some(cl_types::I64),
                    )
                    .expect("GC allocation helper must return a value");
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
                        roots_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        runtime_id,
                        gc_alloc_varsize_shim as *const () as usize,
                        &[base_size, item_size, length],
                        Some(cl_types::I64),
                    )
                    .expect("GC varsize allocation helper must return a value");
                    builder.def_var(var(vi), result);
                }
                OpCode::CallMallocNurseryVarsizeFrame => {
                    let runtime_id = gc_runtime_id.ok_or_else(|| missing_gc_runtime(op.opcode))?;
                    let runtime_id = builder.ins().iconst(cl_types::I64, runtime_id as i64);
                    let size_total =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(0));
                    let size = builder.ins().iadd_imm(size_total, -(GcHeader::SIZE as i64));
                    let result = emit_collecting_gc_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        roots_ptr,
                        &ref_root_slots,
                        &defined_ref_vars,
                        runtime_id,
                        gc_alloc_nursery_shim as *const () as usize,
                        &[size],
                        Some(cl_types::I64),
                    )
                    .expect("GC frame allocation helper must return a value");
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
                    let scale_start = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(3),
                        "ZERO_ARRAY start scale",
                    )?;
                    let scale_size = resolve_constant_i64(
                        &constants,
                        &known_values,
                        op.opcode,
                        op.arg(4),
                        "ZERO_ARRAY size scale",
                    )?;
                    let base =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(0));
                    let start =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(1));
                    let size =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(2));

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
                    let offset =
                        resolve_opref_or_imm(&mut builder, &constants, &known_values, op.arg(1));
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
                    builder.ins().jump(loop_block, &vals);
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
                // Vectors are represented as pairs of i64 values stored
                // in two consecutive Cranelift variables.
                // We implement vector ops via scalar emulation for portability.
                // A future SIMD pass can lower these to native vector instructions.
                OpCode::VecIntAdd
                | OpCode::VecIntSub
                | OpCode::VecIntMul
                | OpCode::VecIntAnd
                | OpCode::VecIntOr
                | OpCode::VecIntXor => {
                    // Binary vector integer ops: emulated as scalar ops on the
                    // packed representation (pairs of i64).
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

                // ── Vector float arithmetic ──
                OpCode::VecFloatAdd
                | OpCode::VecFloatSub
                | OpCode::VecFloatMul
                | OpCode::VecFloatTrueDiv => {
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

                OpCode::VecFloatNeg => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fresult = builder.ins().fneg(fa);
                    let result = builder
                        .ins()
                        .bitcast(cl_types::I64, MemFlags::new(), fresult);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecFloatAbs => {
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let fa = builder.ins().bitcast(cl_types::F64, MemFlags::new(), a);
                    let fresult = builder.ins().fabs(fa);
                    let result = builder
                        .ins()
                        .bitcast(cl_types::I64, MemFlags::new(), fresult);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecFloatXor => {
                    // XOR on the raw bits of float values
                    let a = resolve_opref(&mut builder, &constants, op.args[0]);
                    let b = resolve_opref(&mut builder, &constants, op.args[1]);
                    let result = builder.ins().bxor(a, b);
                    builder.def_var(var(vi), result);
                }

                // ── Vector comparison/test operations ──
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
                // These operate on scalar values, packing/unpacking from vector positions.
                // In the scalar emulation model, vec == the scalar value itself.
                OpCode::VecI | OpCode::VecF => {
                    // Create an "uninitialized" vector → just zero
                    let zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(var(vi), zero);
                }

                OpCode::VecPackI | OpCode::VecPackF => {
                    // vec_pack(vec, scalar, index, count) → just use the scalar
                    let scalar = resolve_opref(&mut builder, &constants, op.args[1]);
                    builder.def_var(var(vi), scalar);
                }

                OpCode::VecUnpackI | OpCode::VecUnpackF => {
                    // vec_unpack(vec, index, count) → extract the scalar
                    let vec_val = resolve_opref(&mut builder, &constants, op.args[0]);
                    builder.def_var(var(vi), vec_val);
                }

                OpCode::VecExpandI | OpCode::VecExpandF => {
                    // vec_expand(scalar) → replicate scalar across all lanes
                    let scalar = resolve_opref(&mut builder, &constants, op.args[0]);
                    builder.def_var(var(vi), scalar);
                }

                // ── Vector load/store ──
                OpCode::VecLoadI | OpCode::VecLoadF => {
                    // Load from memory (like raw_load but for vector data)
                    let base = resolve_opref(&mut builder, &constants, op.args[0]);
                    let offset_val = if op.args.len() > 1 {
                        resolve_opref(&mut builder, &constants, op.args[1])
                    } else {
                        builder.ins().iconst(cl_types::I64, 0)
                    };
                    let addr = builder.ins().iadd(base, offset_val);
                    let result = builder
                        .ins()
                        .load(cl_types::I64, MemFlags::trusted(), addr, 0);
                    builder.def_var(var(vi), result);
                }

                OpCode::VecStore => {
                    // Store to memory
                    let base = resolve_opref(&mut builder, &constants, op.args[0]);
                    let offset_val = if op.args.len() > 2 {
                        resolve_opref(&mut builder, &constants, op.args[1])
                    } else {
                        builder.ins().iconst(cl_types::I64, 0)
                    };
                    let value = resolve_opref(&mut builder, &constants, op.args[op.args.len() - 1]);
                    let addr = builder.ins().iadd(base, offset_val);
                    builder.ins().store(MemFlags::trusted(), value, addr, 0);
                }

                // ── Allocation opcodes ──
                // These are normally eliminated by the optimizer (virtualize pass)
                // or rewritten by the GC rewriter. If they reach the backend,
                // we call out to a runtime helper.
                OpCode::New | OpCode::NewWithVtable | OpCode::NewArray | OpCode::NewArrayClear => {
                    // Call jit_malloc_nursery(size) → returns ptr
                    let size =
                        if op.opcode == OpCode::NewArray || op.opcode == OpCode::NewArrayClear {
                            // Array allocation: first arg is the length
                            let len = resolve_opref(&mut builder, &constants, op.arg(0));
                            // size = base_size + len * item_size
                            // For simplicity, assume 8-byte items + 16-byte header
                            let item_size = builder.ins().iconst(cl_types::I64, 8);
                            let items_total = builder.ins().imul(len, item_size);
                            builder.ins().iadd_imm(items_total, 16)
                        } else {
                            // Fixed-size object: use descriptor's size_of if available, else 16
                            builder.ins().iconst(cl_types::I64, 16)
                        };

                    let result = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_malloc_nursery_shim as *const () as usize,
                        &[size],
                        Some(cl_types::I64),
                    )
                    .expect("jit_malloc_nursery_shim must return a value");
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
                    let len = resolve_opref(&mut builder, &constants, op.arg(0));
                    // Byte string: 1 byte per char + 16-byte header
                    let size = builder.ins().iadd_imm(len, 16);
                    let result = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_malloc_nursery_shim as *const () as usize,
                        &[size],
                        Some(cl_types::I64),
                    )
                    .expect("jit_malloc_nursery_shim must return a value");
                    builder.def_var(var(vi), result);
                }
                OpCode::Newunicode => {
                    let len = resolve_opref(&mut builder, &constants, op.arg(0));
                    // Unicode string: 4 bytes per char + 16-byte header
                    let char_size = builder.ins().iconst(cl_types::I64, 4);
                    let chars_total = builder.ins().imul(len, char_size);
                    let size = builder.ins().iadd_imm(chars_total, 16);
                    let result = emit_host_call(
                        &mut builder,
                        ptr_type,
                        call_conv,
                        jit_malloc_nursery_shim as *const () as usize,
                        &[size],
                        Some(cl_types::I64),
                    )
                    .expect("jit_malloc_nursery_shim must return a value");
                    builder.def_var(var(vi), result);
                }

                other => {
                    return Err(BackendError::Unsupported(format!(
                        "opcode {:?} not yet implemented",
                        other
                    )));
                }
            }

            if op.result_type() == Type::Ref {
                defined_ref_vars.insert(vi);
            }
        }

        builder.seal_block(loop_block);
        builder.finalize();

        // Compile
        let mut ctx = Context::for_function(func);
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| BackendError::CompilationFailed(e.to_string()))?;
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
        Ok(CompiledLoop {
            _func_id: func_id,
            code_ptr,
            code_size: 0,
            fail_descrs,
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
        if let Some(runtime_id) = self.gc_runtime_id.take() {
            unregister_gc_runtime(runtime_id);
        }
    }
}

fn collect_guards(
    ops: &[Op],
    inputargs: &[InputArg],
    fail_descrs: &mut Vec<Arc<CraneliftFailDescr>>,
    guard_infos: &mut Vec<GuardInfo>,
    max_output_slots: &mut usize,
    trace_id: u64,
) -> Result<(), BackendError> {
    let num_inputs = inputargs.len();
    let value_types = build_value_type_map(inputargs, ops);
    let force_tokens = build_force_token_set(inputargs, ops);

    for op in ops {
        let is_guard = op.opcode.is_guard();
        let is_finish = op.opcode == OpCode::Finish;

        if !is_guard && !is_finish {
            continue;
        }

        let fail_index = fail_descrs.len() as u32;

        let (fail_arg_refs, fail_arg_types) = if is_finish {
            let refs: Vec<OpRef> = op.args.iter().copied().collect();
            let types = infer_fail_arg_types(&refs, &value_types)?;
            (refs, types)
        } else if let Some(ref fa) = op.fail_args {
            let refs: Vec<OpRef> = fa.iter().copied().collect();
            let types = infer_fail_arg_types(&refs, &value_types)?;
            (refs, types)
        } else {
            let refs: Vec<OpRef> = (0..num_inputs as u32).map(OpRef).collect();
            let types: Vec<Type> = inputargs.iter().map(|ia| ia.tp).collect();
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

        let descr = Arc::new(
            CraneliftFailDescr::new_with_trace_and_kind_and_force_tokens(
                fail_index,
                trace_id,
                fail_arg_types,
                is_finish,
                force_token_slots,
            ),
        );
        fail_descrs.push(descr);
        guard_infos.push(GuardInfo {
            fail_index,
            fail_arg_refs,
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Backend trait implementation
// ---------------------------------------------------------------------------

impl majit_codegen::Backend for CraneliftBackend {
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        token: &mut LoopToken,
    ) -> Result<AsmInfo, BackendError> {
        token.inputarg_types = inputargs.iter().map(|ia| ia.tp).collect();
        // Pass the address of the invalidation flag so GUARD_NOT_INVALIDATED
        // can load from it at runtime.
        let flag_ptr = Arc::as_ptr(&token.invalidated) as *const AtomicBool as usize;
        let compiled = self.do_compile(inputargs, ops, Some(flag_ptr))?;
        let info = AsmInfo {
            code_addr: compiled.code_ptr as usize,
            code_size: compiled.code_size,
        };
        register_call_assembler_target(token, &compiled);
        token.compiled = Some(Box::new(compiled));
        Ok(info)
    }

    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        original_token: &LoopToken,
    ) -> Result<AsmInfo, BackendError> {
        // Compile the bridge trace as a standalone function using the same
        // code generation path as compile_loop.
        // Bridges share the parent loop's invalidation flag.
        let flag_ptr = Arc::as_ptr(&original_token.invalidated) as *const AtomicBool as usize;
        let compiled = self.do_compile(inputargs, ops, Some(flag_ptr))?;
        let info = AsmInfo {
            code_addr: compiled.code_ptr as usize,
            code_size: compiled.code_size,
        };

        // Attach the bridge to the original guard's fail descriptor so that
        // execute_token can dispatch to it on subsequent guard failures.
        let original_compiled = original_token
            .compiled
            .as_ref()
            .and_then(|c| c.downcast_ref::<CompiledLoop>())
            .ok_or_else(|| {
                BackendError::CompilationFailed("original token has no compiled loop".to_string())
            })?;

        let fi = fail_descr.fail_index() as usize;
        if fi < original_compiled.fail_descrs.len() {
            let target_descr = &original_compiled.fail_descrs[fi];
            target_descr.attach_bridge(BridgeData {
                code_ptr: compiled.code_ptr,
                fail_descrs: compiled.fail_descrs,
                gc_runtime_id: compiled.gc_runtime_id,
                num_inputs: compiled.num_inputs,
                num_ref_roots: compiled.num_ref_roots,
                max_output_slots: compiled.max_output_slots,
                needs_force_frame: compiled.needs_force_frame,
            });
        }

        Ok(info)
    }

    fn execute_token(&self, token: &LoopToken, args: &[Value]) -> DeadFrame {
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

    fn execute_token_ints(&self, token: &LoopToken, args: &[i64]) -> DeadFrame {
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
        token: &LoopToken,
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

        let fail_descr = &compiled.fail_descrs[fail_index as usize];
        fail_descr.increment_fail_count();

        // If a bridge is attached, fall back to the full DeadFrame path.
        let bridge_guard = fail_descr.bridge.lock().unwrap();
        if let Some(ref bridge) = *bridge_guard {
            release_force_token(handle);
            let frame = Self::execute_bridge(bridge, &outputs, &fail_descr.fail_arg_types);
            let descr = frame
                .data
                .downcast_ref::<FrameData>()
                .expect("bridge returned unexpected frame type");
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
                fail_index: descr.fail_descr.fail_index(),
                trace_id: descr.fail_descr.trace_id(),
                is_finish: descr.fail_descr.is_finish(),
            };
        }
        drop(bridge_guard);

        // No bridge — skip DeadFrame, return outputs directly.
        if let Some(ref ff) = force_frame {
            take_force_frame_saved_data(ff);
        }
        take_pending_jit_exception_state();
        if fail_descr.force_token_slots.is_empty() {
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
            fail_index,
            trace_id: fail_descr.trace_id(),
            is_finish: fail_descr.is_finish(),
        }
    }

    fn compiled_fail_descr_layouts(
        &self,
        token: &LoopToken,
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
        original_token: &LoopToken,
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
        let bridge = source_descr.bridge.lock().unwrap();
        let bridge = bridge.as_ref()?;
        Some(
            bridge
                .fail_descrs
                .iter()
                .map(|descr| descr.layout())
                .collect(),
        )
    }

    fn force(&self, force_token: GcRef) -> DeadFrame {
        force_token_to_dead_frame(force_token)
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
            return frame_data.fail_descr.as_ref();
        }
        if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
            return preview.frame.fail_descr.as_ref();
        }
        panic!("unsupported dead frame type")
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
            return frame_data.get_int(index);
        }
        if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
            return preview.frame.get_int(index);
        }
        panic!("unsupported dead frame type")
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
            return frame_data.get_float(index);
        }
        if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
            return preview.frame.get_float(index);
        }
        panic!("unsupported dead frame type")
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        if let Some(frame_data) = frame.data.downcast_ref::<FrameData>() {
            return frame_data.get_ref(index);
        }
        if let Some(preview) = frame.data.downcast_ref::<PreviewFrameData>() {
            return preview.frame.get_ref(index);
        }
        panic!("unsupported dead frame type")
    }

    fn set_savedata_ref(&self, frame: &mut DeadFrame, data: GcRef) {
        set_savedata_ref_on_deadframe(frame, data);
    }

    fn get_savedata_ref(&self, frame: &DeadFrame) -> GcRef {
        get_savedata_ref_from_deadframe(frame)
    }

    fn grab_exc_value(&self, frame: &DeadFrame) -> GcRef {
        grab_exc_value_from_deadframe(frame)
    }

    fn grab_exc_class(&self, frame: &DeadFrame) -> i64 {
        grab_exc_class_from_deadframe(frame)
    }

    fn invalidate_loop(&self, token: &LoopToken) {
        token.invalidate();
    }

    fn redirect_call_assembler(&self, old: &LoopToken, new: &LoopToken) {
        redirect_call_assembler_target(old.number, new.number);
    }

    fn free_loop(&mut self, token: &LoopToken) {
        unregister_call_assembler_target(token.number);
    }
}

// ---------------------------------------------------------------------------
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
            &EffectInfo {
                extra_effect: ExtraEffect::CanRaise,
                oopspec_index: majit_ir::OopSpecIndex::None,
            }
        }
    }

    fn make_call_descr(arg_types: Vec<Type>, result_type: Type) -> majit_ir::DescrRef {
        Arc::new(TestCallDescr {
            arg_types,
            result_type,
        })
    }

    fn make_call_assembler_descr(
        target: &LoopToken,
        arg_types: Vec<Type>,
        result_type: Type,
    ) -> majit_ir::DescrRef {
        Arc::new(CallAssemblerDescr::new(
            target.number,
            arg_types,
            result_type,
        ))
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
            values.push(get_latest_descr_from_deadframe(&deadframe).fail_index() as i64);
            values.push(get_int_from_deadframe(&deadframe, 0));
            values.push(get_int_from_deadframe(&deadframe, 1));
            drop(values);
            set_savedata_ref_on_deadframe(&mut deadframe, GcRef(0xDADA));
        }
    }

    extern "C" fn maybe_force_and_return_int(force_token: i64, flag: i64) -> i64 {
        if flag != 0 {
            let mut deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            let mut values = may_force_int_values().lock().unwrap();
            values.push(get_int_from_deadframe(&deadframe, 0));
            values.push(get_int_from_deadframe(&deadframe, 2));
            drop(values);
            set_savedata_ref_on_deadframe(&mut deadframe, GcRef(0xBABA));
        }
        42
    }

    extern "C" fn maybe_force_and_return_float(force_token: i64, flag: i64) -> f64 {
        if flag != 0 {
            let deadframe = force_token_to_dead_frame(GcRef(force_token as usize));
            let mut values = may_force_float_values().lock().unwrap();
            values.push(get_int_from_deadframe(&deadframe, 0) as u64);
            values.push(get_float_from_deadframe(&deadframe, 1).to_bits());
            values.push(get_int_from_deadframe(&deadframe, 2) as u64);
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
            let preview_live = get_ref_from_deadframe(&deadframe, 2);
            set_savedata_ref_on_deadframe(&mut deadframe, preview_live);
            with_gc_runtime(runtime_id as u64, |gc| gc.collect_nursery());
            let preview_result = get_ref_from_deadframe(&deadframe, 1);
            let preview_live = get_ref_from_deadframe(&deadframe, 2);
            let preview_return = get_ref_from_deadframe(&deadframe, 3);
            let mut values = may_force_ref_values().lock().unwrap();
            values.push(preview_result.0);
            values.push(preview_live.0);
            values.push(preview_return.0);
            return get_ref_from_deadframe(&deadframe, 3).0 as i64;
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
        let mut token = LoopToken::new(token_number);
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

        let mut token = LoopToken::new(0);
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

        let mut token = LoopToken::new(1);
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

        let mut token = LoopToken::new(1000);
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

        let mut token = LoopToken::new(2);
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

        let mut token = LoopToken::new(3);
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

        let mut token = LoopToken::new(3);
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

        let mut token = LoopToken::new(4);
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

        let mut token = LoopToken::new(5);
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

        let mut token = LoopToken::new(6);
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

        let mut token = LoopToken::new(7);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(10)]);
        assert_eq!(backend.get_int_value(&frame, 0), 0);
    }

    #[test]
    fn test_fail_descr() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(8);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_int_value(&frame, 0), 42);
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

        let mut token = LoopToken::new(9);
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

        let mut token = LoopToken::new(10);
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

        let mut token = LoopToken::new(20);
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

        let mut token = LoopToken::new(21);
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

        let mut token = LoopToken::new(22);
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

        let mut token = LoopToken::new(23);
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

        let mut token = LoopToken::new(24);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(0)]);
        assert_eq!(backend.get_int_value(&frame, 0), 99);
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

        let mut token = LoopToken::new(25);
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

        let mut token = LoopToken::new(26);
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

        let mut token = LoopToken::new(27);
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

        let mut token = LoopToken::new(28);
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
            nursery_size: 48,
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

        let mut token = LoopToken::new(29);
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

        let mut token = LoopToken::new(30);
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

        let mut token = LoopToken::new(31);
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

        let mut token = LoopToken::new(32);
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

        let mut token = LoopToken::new(50);
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

        let mut token = LoopToken::new(40);
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

        let mut token = LoopToken::new(41);
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

        let mut token = LoopToken::new(42);
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

        let mut token = LoopToken::new(43);
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

        let mut token = LoopToken::new(44);
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

        let mut token = LoopToken::new(45);
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

        let mut token = LoopToken::new(46);
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

        let mut token = LoopToken::new(47);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(42)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish
        assert_eq!(backend.get_int_value(&frame, 0), 42);

        // Test: value doesn't match -> guard fails
        let mut constants2 = HashMap::new();
        constants2.insert(100, 42i64);
        backend.set_constants(constants2);

        let mut token2 = LoopToken::new(48);
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

        let mut token = LoopToken::new(60);
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

        let mut token = LoopToken::new(61);
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

        let mut token = LoopToken::new(70);
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

        let mut token = LoopToken::new(71);
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

        let mut token = LoopToken::new(72);
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

        let mut token = LoopToken::new(73);
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

        let mut token = LoopToken::new(74);
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

        let mut token = LoopToken::new(75);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // header: [type_id, length=5]
        let mut data: Vec<i64> = vec![0xAAAA, 5, 10, 20, 30, 40, 50];
        let ptr = data.as_mut_ptr() as usize;

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(ptr))]);
        assert_eq!(backend.get_int_value(&frame, 0), 5);
    }

    // ── NurseryPtrIncrement test ──

    #[test]
    fn test_nursery_ptr_increment() {
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::NurseryPtrIncrement, &[OpRef(0), OpRef(1)], 2),
            mk_op(OpCode::Finish, &[OpRef(2)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(76);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(0x1000)), Value::Int(0x100)]);
        assert_eq!(backend.get_ref_value(&frame, 0), GcRef(0x1100));
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

        let mut token = LoopToken::new(80);
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

        let mut token = LoopToken::new(81);
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

        let mut token = LoopToken::new(82);
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

        let mut token = LoopToken::new(83);
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

        let mut token = LoopToken::new(84);
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

        let mut token = LoopToken::new(85);
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

        let mut token = LoopToken::new(86);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // With overflow: guard_overflow passes (continues)
        let frame = backend.execute_token(&token, &[Value::Int(i64::MAX), Value::Int(1)]);
        let descr = backend.get_latest_descr(&frame);
        assert_eq!(descr.fail_index(), 1); // Finish (overflow happened, guard passed)

        // Without overflow: guard_overflow fails (side-exits)
        let mut token2 = LoopToken::new(87);
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

        let mut token = LoopToken::new(90);
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

        let mut token = LoopToken::new(91);
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

        let mut token = LoopToken::new(92);
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

        let mut token = LoopToken::new(93);
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

        let mut token = LoopToken::new(94);
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

        let mut token = LoopToken::new(95);
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

        let mut token = LoopToken::new(96);
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

        let mut token = LoopToken::new(97);
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

        let mut token = LoopToken::new(98);
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

        let mut token = LoopToken::new(99);
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

        let mut token = LoopToken::new(100);
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

        let mut token = LoopToken::new(101);
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

        let mut token = LoopToken::new(102);
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

        let mut token = LoopToken::new(200);
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

        let mut token = LoopToken::new(202);
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

        let mut token = LoopToken::new(201);
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
        // The finish descr also gets its count incremented (it's index 1)
        let compiled = token
            .compiled
            .as_ref()
            .unwrap()
            .downcast_ref::<CompiledLoop>()
            .unwrap();
        let guard_descr = &compiled.fail_descrs[0];
        assert_eq!(guard_descr.get_fail_count(), 5); // Still 5
        let finish_descr = &compiled.fail_descrs[1];
        assert_eq!(finish_descr.get_fail_count(), 1);
    }

    #[test]
    fn test_bridge_with_loop() {
        // Main loop: counts down from N, guard fails when counter reaches 0.
        // Bridge: takes the counter value and adds 1000, then finishes.
        let mut backend = CraneliftBackend::new();

        let inputargs = vec![InputArg::new_int(0)]; // i0 = counter
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntSub, &[OpRef(0), OpRef(100)], 1), // i1 = i0 - 1
            mk_op(OpCode::IntGt, &[OpRef(1), OpRef(101)], 2),  // i2 = i1 > 0
            mk_op(OpCode::GuardTrue, &[OpRef(2)], OpRef::NONE.0), // guard_true(i2)
            mk_op(OpCode::Jump, &[OpRef(1)], OpRef::NONE.0),   // jump(i1)
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 1i64);
        constants.insert(101, 0i64);
        backend.set_constants(constants);

        let mut token = LoopToken::new(202);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        // Without bridge: run with N=5, guard fails when counter=0
        // The guard saves the input arg (i0), which at the point of failure is 1
        // (i0=1, i1=0, guard fails)
        let frame = backend.execute_token(&token, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1);

        // Compile bridge: takes the counter and returns it + 1000
        let bridge_inputargs = vec![InputArg::new_int(0)];
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::IntAdd, &[OpRef(0), OpRef(100)], 1),
            mk_op(OpCode::Finish, &[OpRef(1)], OpRef::NONE.0),
        ];

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(100, 1000i64);
        backend.set_constants(bridge_constants);

        let fail_descr = CraneliftFailDescr::new(0, vec![Type::Int]);
        backend
            .compile_bridge(&fail_descr, &bridge_inputargs, &bridge_ops, &token)
            .unwrap();

        // With bridge: run with N=5, guard fails at counter=1, bridge runs
        let frame = backend.execute_token(&token, &[Value::Int(5)]);
        // Bridge gets i0=1 (the input arg saved at guard failure), returns 1 + 1000 = 1001
        assert_eq!(backend.get_int_value(&frame, 0), 1001);

        // Run with N=3
        let frame = backend.execute_token(&token, &[Value::Int(3)]);
        assert_eq!(backend.get_int_value(&frame, 0), 1001);
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

        let mut token = LoopToken::new(203);
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

        let mut token = LoopToken::new(1001);
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

        let mut token = LoopToken::new(1005);
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

        let mut token = LoopToken::new(1002);
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

        let mut token = LoopToken::new(1003);
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

        let mut token = LoopToken::new(1004);
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

        let mut token = LoopToken::new(1500_400);
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
        assert_eq!(backend.get_savedata_ref(&frame), GcRef(0xDADA));
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

        let mut token = LoopToken::new(1500_401);
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
        assert_eq!(backend.get_savedata_ref(&frame), GcRef(0xBABA));
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

        let mut token = LoopToken::new(1500_402);
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
            nursery_size: 64,
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

        let mut token = LoopToken::new(1500_403);
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
        assert_eq!(backend.get_savedata_ref(&frame), forced_live);
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

        let mut token = LoopToken::new(1500_300);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(20)]);
        let force_token = backend.get_ref_value(&frame, 0);
        assert_ne!(force_token, GcRef::NULL);

        let forced = backend.force(force_token);
        let descr = backend.get_latest_descr(&forced);
        assert_eq!(descr.fail_index(), 0);
        assert_eq!(backend.get_int_value(&forced, 0), 30);
    }

    #[test]
    fn test_guard_not_forced_2_snapshot_roots_refs_until_force() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 48,
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

        let mut token = LoopToken::new(1500_301);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(root)]);
        let force_token = backend.get_ref_value(&frame, 0);

        with_gc_runtime(runtime_id, |gc| gc.collect_nursery());

        let forced = backend.force(force_token);
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

        let mut callee_token = LoopToken::new(1500_200);
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

        let mut caller_token = LoopToken::new(1500_201);
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
        let mut callee1 = LoopToken::new(1500_210);
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
        let mut callee2 = LoopToken::new(1500_211);
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
        let mut caller = LoopToken::new(1500_212);
        backend
            .compile_loop(&inputargs, &caller_ops, &mut caller)
            .unwrap();

        let frame = backend.execute_token(&caller, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 6);

        backend.redirect_call_assembler(&callee1, &callee2);

        let frame = backend.execute_token(&caller, &[Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 105);
    }

    #[test]
    fn test_call_assembler_rejects_guarded_callee_target() {
        let mut backend = CraneliftBackend::new();

        let callee_inputargs = vec![InputArg::new_int(0)];
        let callee_ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::GuardTrue, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0)], OpRef::NONE.0),
        ];
        let mut callee = LoopToken::new(1500_220);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee)
            .unwrap();

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

        let mut caller = LoopToken::new(1500_221);
        let err = backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller)
            .unwrap_err();
        match err {
            BackendError::Unsupported(msg) => assert!(msg.contains("finish-only callee loops")),
            other => panic!("expected unsupported error, got {other:?}"),
        }
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

        let mut token = LoopToken::new(1500);
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

        let mut token = LoopToken::new(1501);
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

        let mut token = LoopToken::new(1502);
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

        let mut token = LoopToken::new(1503);
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
    fn test_gc_backend_uses_collecting_allocator_entrypoints_and_registers_roots() {
        let state = Arc::new(Mutex::new(TrackingGcState::default()));
        let mut backend =
            CraneliftBackend::with_gc_allocator(Box::new(TrackingGc::new(state.clone())));

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef(0)], OpRef::NONE.0),
            mk_op(OpCode::CallMallocNursery, &[OpRef(24)], 1),
            mk_op(
                OpCode::GcStore,
                &[OpRef(1), OpRef(0), OpRef(3)],
                OpRef::NONE.0,
            ),
            mk_op(OpCode::CondCallGcWb, &[OpRef(1)], OpRef::NONE.0),
            mk_op(OpCode::Finish, &[OpRef(0), OpRef(1)], OpRef::NONE.0),
        ];

        let mut token = LoopToken::new(1504);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(GcRef(0x1234))]);
        let input_root = backend.get_ref_value(&frame, 0);
        let obj = backend.get_ref_value(&frame, 1);
        assert_eq!(input_root, GcRef(0x1234));
        assert!(!obj.is_null());
        drop(frame);

        let state = state.lock().unwrap();
        assert_eq!(state.collecting_allocs, 1);
        assert_eq!(state.no_collect_allocs, 0);
        // Two shadow-root slots are registered around the collecting helper,
        // and two rooted refs are registered again while the returned
        // DeadFrame is alive.
        assert_eq!(state.added_roots, 4);
        assert_eq!(state.removed_roots, 4);
        // One from fresh old-gen/nursery safety in the alloc shim, one explicit WB op.
        assert_eq!(state.write_barriers, 2);
    }

    #[test]
    fn test_gc_collecting_alloc_preserves_live_ref_inputs() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 48,
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

        let mut token = LoopToken::new(1505);
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
    fn test_high_level_new_and_setfield_gc_auto_rewrite() {
        let state = Arc::new(Mutex::new(TrackingGcState::default()));
        let mut backend =
            CraneliftBackend::with_gc_allocator(Box::new(TrackingGc::new(state.clone())));

        let sd = make_size_descr(16, 11);
        let fd = make_field_descr(0, 8, Type::Ref, false);
        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            Op::new(OpCode::Label, &[OpRef(0)]),
            Op::with_descr(OpCode::New, &[], sd),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(2), OpRef(0)], fd),
            Op::new(OpCode::Finish, &[OpRef(2), OpRef(0)]),
        ];

        let mut token = LoopToken::new(1510);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let input_ref = GcRef(0x1234);
        let frame = backend.execute_token(&token, &[Value::Ref(input_ref)]);
        let obj = backend.get_ref_value(&frame, 0);
        let stored_input = backend.get_ref_value(&frame, 1);
        assert!(!obj.is_null());
        assert_eq!(stored_input, input_ref);
        assert_eq!(unsafe { *((obj.0) as *const usize) }, input_ref.0);
        drop(frame);

        let state = state.lock().unwrap();
        assert_eq!(state.collecting_allocs, 1);
        // Fresh allocation helper accounts for one barrier; the rewritten
        // SETFIELD_GC must not add another barrier for the same young object.
        assert_eq!(state.write_barriers, 1);
    }

    #[test]
    fn test_high_level_new_array_and_setarrayitem_gc_auto_rewrite() {
        let state = Arc::new(Mutex::new(TrackingGcState::default()));
        let mut backend =
            CraneliftBackend::with_gc_allocator(Box::new(TrackingGc::new(state.clone())));

        let ad = make_array_descr(8, 8, Type::Ref, Some(0));
        let inputargs = vec![InputArg::new_int(0), InputArg::new_ref(1)];
        let ops = vec![
            Op::new(OpCode::Label, &[OpRef(0), OpRef(1)]),
            Op::with_descr(OpCode::NewArray, &[OpRef(0)], ad.clone()),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(3), OpRef(100), OpRef(1)],
                ad.clone(),
            ),
            Op::with_descr(OpCode::ArraylenGc, &[OpRef(3)], ad),
            Op::new(OpCode::Finish, &[OpRef(3), OpRef(5)]),
        ];

        let mut constants = HashMap::new();
        constants.insert(100, 2);
        backend.set_constants(constants);

        let mut token = LoopToken::new(1511);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let input_ref = GcRef(0xABCD);
        let frame = backend.execute_token(&token, &[Value::Int(3), Value::Ref(input_ref)]);
        let arr = backend.get_ref_value(&frame, 0);
        let length = backend.get_int_value(&frame, 1);
        assert!(!arr.is_null());
        assert_eq!(length, 3);
        assert_eq!(unsafe { *((arr.0 + 24) as *const usize) }, input_ref.0);
        drop(frame);

        let state = state.lock().unwrap();
        assert_eq!(state.collecting_allocs, 1);
        // Varlen alloc helper accounts for one barrier, and fresh arrays still
        // need the rewritten array WB/card-mark path.
        assert_eq!(state.write_barriers, 2);
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

        let mut token = LoopToken::new(1512);
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

        let mut token = LoopToken::new(1513);
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

        let mut token = LoopToken::new(1514);
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

        let mut token = LoopToken::new(1515);
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

        let mut token = LoopToken::new(1516);
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
            nursery_size: 48,
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

        let mut token = LoopToken::new(1506);
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
            nursery_size: 48,
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

        let mut token = LoopToken::new(1507);
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
            nursery_size: 48,
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

        let mut token = LoopToken::new(1508);
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
            nursery_size: 48,
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

        let mut token = LoopToken::new(1509);
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

        let mut token = LoopToken::new(100);
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

        let mut token = LoopToken::new(101);
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

        let mut token = LoopToken::new(102);
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

        let mut token = LoopToken::new(103);
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
}
