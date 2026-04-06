/// Guard failure handling for the Cranelift backend.
///
/// When a guard fails at runtime, execution exits the JIT-compiled loop
/// and values stay in the JitFrame. The JitFrame GcRef is returned as
/// the deadframe (RPython llmodel.py parity).
///
/// Bridge support: when a guard fails frequently, a bridge trace can be
/// compiled and attached to the fail descriptor. On subsequent guard
/// failures, execution transfers to the bridge instead of returning to
/// the interpreter.
use crate::compiler::{register_gc_roots, unregister_gc_roots};
use majit_backend::{CompiledTraceInfo, ExitRecoveryLayout, FailDescrLayout, TerminalExitLayout};
use majit_gc::GcMap;
use majit_ir::{AccumVectorInfo, FailDescr, GcRef, Type};
use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Compiled bridge data attached to a guard's fail descriptor.
///
/// When a bridge is compiled, its code pointer and metadata are stored
/// here so `execute_token` can dispatch to the bridge on guard failure.
pub struct BridgeData {
    /// Compiled trace identifier for this bridge.
    pub trace_id: u64,
    /// Input types expected at the bridge header.
    pub input_types: Vec<Type>,
    /// Interpreter header pc associated with this bridge trace.
    pub header_pc: u64,
    /// Source guard this bridge is attached to.
    pub source_guard: (u64, u32),
    /// Recovery-layout caller prefix inherited from the source guard.
    pub caller_prefix_layout: Option<ExitRecoveryLayout>,
    /// Function pointer to the bridge's compiled code.
    /// Same calling convention as a compiled loop:
    ///   fn(inputs_ptr: *const i64, outputs_ptr: *mut i64, roots_ptr: *mut i64) -> i64
    pub code_ptr: *const u8,
    /// Fail descriptors within the bridge (guards + finish).
    pub fail_descrs: Vec<Arc<CraneliftFailDescr>>,
    /// GC runtime used by the compiled bridge, if any.
    pub gc_runtime_id: Option<u64>,
    /// Number of input arguments the bridge expects.
    /// Set to parent guard's fail_arg count (not optimizer-reduced count)
    /// so execute_bridge passes all parent outputs and indices align.
    pub num_inputs: usize,
    /// Number of shadow-root slots the bridge expects.
    pub num_ref_roots: usize,
    /// Maximum output slots for guard exits within the bridge.
    pub max_output_slots: usize,
    /// Static terminal-exit layouts within the bridge trace.
    /// Write-once during bridge compilation, read-only after.
    /// No lock needed — RPython ResumeGuardDescr has no lock (GIL).
    pub terminal_exit_layouts: UnsafeCell<Vec<TerminalExitLayout>>,
    /// When true, a bridge Finish with matching arity should re-enter
    /// the parent loop instead of returning to the interpreter.
    /// Set for bridges that reach the loop's merge_point.
    pub loop_reentry: bool,
    /// compile.py:186: record_loop_or_bridge sets descr.rd_loop_token = clt
    /// on ALL guards (loop and bridge). The bridge shares the parent loop's
    /// invalidation flag (AtomicBool). Holding an Arc clone keeps the flag
    /// alive as long as the bridge exists.
    pub invalidated_arc: Option<Arc<std::sync::atomic::AtomicBool>>,
}

unsafe impl Send for BridgeData {}
unsafe impl Sync for BridgeData {}

impl BridgeData {
    #[inline]
    pub fn terminal_exit_layouts_ref(&self) -> &Vec<TerminalExitLayout> {
        unsafe { &*self.terminal_exit_layouts.get() }
    }

    #[inline]
    pub fn terminal_exit_layouts_mut(&self) -> &mut Vec<TerminalExitLayout> {
        unsafe { &mut *self.terminal_exit_layouts.get() }
    }
}

impl std::fmt::Debug for BridgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BridgeData")
            .field("trace_id", &self.trace_id)
            .field("input_types", &self.input_types)
            .field("header_pc", &self.header_pc)
            .field("source_guard", &self.source_guard)
            .field("caller_prefix_layout", &self.caller_prefix_layout)
            .field("code_ptr", &self.code_ptr)
            .field("gc_runtime_id", &self.gc_runtime_id)
            .field("num_inputs", &self.num_inputs)
            .field("num_ref_roots", &self.num_ref_roots)
            .field("terminal_exit_layouts", unsafe {
                &*self.terminal_exit_layouts.get()
            })
            .finish()
    }
}

/// Concrete fail descriptor used by the Cranelift backend.
///
/// Carries the fail_index and the types of values that will be
/// saved in the DeadFrame on guard failure.
///
/// Also tracks guard failure count and an optional bridge that
/// should be executed instead of returning to the interpreter.
pub struct CraneliftFailDescr {
    pub fail_index: u32,
    pub source_op_index: Option<usize>,
    pub trace_id: u64,
    /// RPython resumedescr.original_greenkey parity: the green_key of
    /// the compiled loop this guard belongs to.
    pub green_key: u64,
    pub fail_arg_types: Vec<Type>,
    pub gc_map: GcMap,
    pub is_finish: bool,
    /// Bridge external JUMP → parent loop: the caller should re-enter
    /// the parent loop with these fail_arg values as new inputs.
    /// Set after bridge compilation — uses atomic for interior mutability.
    pub is_loop_reentry: std::sync::atomic::AtomicBool,
    pub force_token_slots: Vec<usize>,
    /// Write-once during compilation, read-only after.
    /// No lock — RPython ResumeGuardDescr has no lock (GIL).
    pub trace_info: UnsafeCell<Option<CompiledTraceInfo>>,
    /// Write-once during bridge compilation, read-only after.
    pub recovery_layout: UnsafeCell<Option<ExitRecoveryLayout>>,
    /// compile.py:688-692 ResumeGuardDescr.status:
    /// Stores jitcounter hash (from store_hash / fetch_next_hash).
    /// Used by must_compile() to tick the guard's counter slot.
    /// Assigned at compile time, read at guard failure time.
    pub status: std::sync::atomic::AtomicU64,
    /// Number of times this guard has failed (for bridge compilation heuristics).
    pub fail_count: AtomicU32,
    /// schedule.py:654-655 / history.py:143-147 — vector guard metadata
    /// copied from the frontend fail descriptor during lowering.
    pub vector_info: Vec<AccumVectorInfo>,
    /// Compiled bridge attached to this guard, if any.
    /// Write-once when bridge is compiled, read-only after.
    /// No lock — RPython compile.py attach_bridge has no lock (GIL).
    pub bridge: UnsafeCell<Option<BridgeData>>,
    /// Atomic cache of bridge code_ptr for lock-free dispatch.
    pub bridge_code_ptr_cache: std::sync::atomic::AtomicUsize,
    /// GC runtime that owns the compiled loop this guard belongs to.
    /// Used by force() to register the JitFrame as a GC root without
    /// relying on thread-local ACTIVE_GC_RUNTIME_ID.
    pub gc_runtime_id: Option<u64>,
}

impl std::fmt::Debug for CraneliftFailDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CraneliftFailDescr")
            .field("fail_index", &self.fail_index)
            .field("source_op_index", &self.source_op_index)
            .field("trace_id", &self.trace_id)
            .field("fail_arg_types", &self.fail_arg_types)
            .field("gc_map", &self.gc_map)
            .field("is_finish", &self.is_finish)
            .field("force_token_slots", &self.force_token_slots)
            .field("trace_info", unsafe { &*self.trace_info.get() })
            .field("recovery_layout", unsafe { &*self.recovery_layout.get() })
            .field("fail_count", &self.fail_count.load(Ordering::Relaxed))
            .field("vector_info", &self.vector_info)
            .field("has_bridge", &unsafe { &*self.bridge.get() }.is_some())
            .finish()
    }
}

// Safety: CraneliftFailDescr is accessed from a single thread (the JIT thread).
// UnsafeCell fields (bridge, trace_info, recovery_layout) are write-once during
// compilation and read-only thereafter. RPython's ResumeGuardDescr has no locks
// (GIL-protected). pyre is single-threaded (no-GIL, single thread).
unsafe impl Send for CraneliftFailDescr {}
unsafe impl Sync for CraneliftFailDescr {}

impl CraneliftFailDescr {
    fn gc_map_for_types(fail_arg_types: &[Type], force_token_slots: &[usize]) -> GcMap {
        let mut gc_map = GcMap::new();
        for (slot, tp) in fail_arg_types.iter().enumerate() {
            if *tp == Type::Ref && !force_token_slots.contains(&slot) {
                gc_map.set_ref(slot);
            }
        }
        gc_map
    }

    /// Create a new fail descriptor.
    pub fn new(fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            false,
            Vec::new(),
            None,
        )
    }

    pub fn new_with_kind(fail_index: u32, fail_arg_types: Vec<Type>, is_finish: bool) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            is_finish,
            Vec::new(),
            None,
        )
    }

    pub fn new_with_kind_and_force_tokens(
        fail_index: u32,
        fail_arg_types: Vec<Type>,
        is_finish: bool,
        force_token_slots: Vec<usize>,
    ) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            is_finish,
            force_token_slots,
            None,
        )
    }

    pub fn new_with_trace_and_kind_and_force_tokens(
        fail_index: u32,
        trace_id: u64,
        fail_arg_types: Vec<Type>,
        is_finish: bool,
        mut force_token_slots: Vec<usize>,
        recovery_layout: Option<ExitRecoveryLayout>,
    ) -> Self {
        force_token_slots.sort_unstable();
        force_token_slots.dedup();
        CraneliftFailDescr {
            fail_index,
            source_op_index: None,
            trace_id,
            green_key: 0,
            gc_map: Self::gc_map_for_types(&fail_arg_types, &force_token_slots),
            fail_arg_types,
            is_finish,
            is_loop_reentry: std::sync::atomic::AtomicBool::new(false),
            force_token_slots,
            trace_info: UnsafeCell::new(None),
            recovery_layout: UnsafeCell::new(recovery_layout),
            status: std::sync::atomic::AtomicU64::new(0),
            fail_count: AtomicU32::new(0),
            vector_info: Vec::new(),
            bridge: UnsafeCell::new(None),
            bridge_code_ptr_cache: std::sync::atomic::AtomicUsize::new(0),
            gc_runtime_id: None,
        }
    }

    // UnsafeCell accessor helpers — single-threaded, no lock needed.
    // RPython ResumeGuardDescr fields are plain attributes (GIL-protected).

    #[inline]
    pub fn bridge_ref(&self) -> &Option<BridgeData> {
        unsafe { &*self.bridge.get() }
    }

    #[inline]
    pub fn trace_info_ref(&self) -> &Option<CompiledTraceInfo> {
        unsafe { &*self.trace_info.get() }
    }

    #[inline]
    pub fn recovery_layout_ref(&self) -> &Option<ExitRecoveryLayout> {
        unsafe { &*self.recovery_layout.get() }
    }

    /// Increment the failure counter and return the new value.
    pub fn increment_fail_count(&self) -> u32 {
        self.fail_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Get the current failure count.
    pub fn get_fail_count(&self) -> u32 {
        self.fail_count.load(Ordering::Relaxed)
    }

    /// Whether a bridge has been attached to this guard.
    pub fn has_bridge(&self) -> bool {
        self.bridge_code_ptr_cache
            .load(std::sync::atomic::Ordering::Relaxed)
            != 0
    }

    /// Get bridge code_ptr without Mutex lock (atomic read).
    pub fn bridge_code_ptr(&self) -> *const u8 {
        self.bridge_code_ptr_cache
            .load(std::sync::atomic::Ordering::Relaxed) as *const u8
    }

    /// Attach a compiled bridge to this guard.
    pub fn attach_bridge(&self, bridge: BridgeData) {
        let code_ptr = bridge.code_ptr as usize;
        unsafe { *self.bridge.get() = Some(bridge) };
        self.bridge_code_ptr_cache
            .store(code_ptr, std::sync::atomic::Ordering::Release);
    }

    // compile.py:687-696 status encoding constants.
    pub const ST_BUSY_FLAG: u64 = 0x01;
    pub const ST_TYPE_MASK: u64 = 0x06;
    pub const ST_SHIFT: u32 = 3;
    pub const ST_SHIFT_MASK: u64 = !((1u64 << Self::ST_SHIFT) - 1); // -(1 << ST_SHIFT)
    pub const TY_NONE: u64 = 0x00;
    pub const TY_INT: u64 = 0x02;
    pub const TY_REF: u64 = 0x04;
    pub const TY_FLOAT: u64 = 0x06;

    /// compile.py:826-830 store_hash: assign a unique jitcounter hash.
    /// `self.status = hash & self.ST_SHIFT_MASK`
    pub fn store_hash(&self, hash: u64) {
        self.status.store(
            hash & Self::ST_SHIFT_MASK,
            std::sync::atomic::Ordering::Release,
        );
    }

    /// compile.py:741-745: read status for must_compile.
    pub fn get_status(&self) -> u64 {
        self.status.load(std::sync::atomic::Ordering::Acquire)
    }

    /// compile.py:786-788: start_compiling — set ST_BUSY_FLAG.
    pub fn start_compiling(&self) {
        self.status
            .fetch_or(Self::ST_BUSY_FLAG, std::sync::atomic::Ordering::AcqRel);
    }

    /// compile.py:790-795: done_compiling — clear ST_BUSY_FLAG.
    pub fn done_compiling(&self) {
        self.status
            .fetch_and(!Self::ST_BUSY_FLAG, std::sync::atomic::Ordering::AcqRel);
    }

    /// compile.py:750: check ST_BUSY_FLAG.
    pub fn is_compiling(&self) -> bool {
        self.status.load(std::sync::atomic::Ordering::Acquire) & Self::ST_BUSY_FLAG != 0
    }

    /// compile.py:813-824: make_a_counter_per_value — for GUARD_VALUE,
    /// encode the fail_arg index and type tag in status.
    /// `self.status = ty | (index << ST_SHIFT)`
    pub fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        let status = type_tag | ((index as u64) << Self::ST_SHIFT);
        self.status
            .store(status, std::sync::atomic::Ordering::Release);
    }

    /// Take the bridge data out of this fail descriptor, leaving None.
    pub fn take_bridge(&self) -> Option<BridgeData> {
        let bridge = unsafe { &mut *self.bridge.get() }.take();
        if bridge.is_some() {
            self.bridge_code_ptr_cache
                .store(0, std::sync::atomic::Ordering::Release);
        }
        bridge
    }

    pub fn set_recovery_layout(&self, recovery_layout: ExitRecoveryLayout) {
        unsafe { *self.recovery_layout.get() = Some(recovery_layout) };
    }

    pub fn set_source_op_index(&mut self, source_op_index: usize) {
        self.source_op_index = Some(source_op_index);
    }

    pub fn set_trace_info(&self, trace_info: CompiledTraceInfo) {
        unsafe { *self.trace_info.get() = Some(trace_info) };
    }

    pub fn gc_map(&self) -> &GcMap {
        &self.gc_map
    }

    pub fn is_finish(&self) -> bool {
        self.is_finish
    }

    pub fn is_force_token_slot(&self, slot: usize) -> bool {
        self.force_token_slots.binary_search(&slot).is_ok()
    }

    pub fn layout(&self) -> FailDescrLayout {
        let gc_ref_slots = self
            .fail_arg_types
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| self.gc_map.is_ref(slot).then_some(slot))
            .collect();
        let recovery = unsafe { &*self.recovery_layout.get() }.clone();
        let frame_stack = recovery.as_ref().map(|r| r.frames.clone());
        FailDescrLayout {
            fail_index: self.fail_index,
            source_op_index: self.source_op_index,
            trace_id: self.trace_id,
            trace_info: unsafe { &*self.trace_info.get() }.clone(),
            fail_arg_types: self.fail_arg_types.clone(),
            is_finish: self.is_finish,
            gc_ref_slots,
            force_token_slots: self.force_token_slots.clone(),
            recovery_layout: recovery,
            frame_stack,
        }
    }
}

impl majit_ir::Descr for CraneliftFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }

    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for CraneliftFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }

    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }

    fn is_finish(&self) -> bool {
        self.is_finish
    }

    fn trace_id(&self) -> u64 {
        self.trace_id
    }

    fn is_gc_ref_slot(&self, slot: usize) -> bool {
        self.gc_map.is_ref(slot)
    }

    fn force_token_slots(&self) -> &[usize] {
        &self.force_token_slots
    }

    fn vector_info(&self) -> Vec<AccumVectorInfo> {
        self.vector_info.clone()
    }

    fn get_status(&self) -> u64 {
        self.get_status()
    }

    fn start_compiling(&self) {
        self.start_compiling()
    }

    fn done_compiling(&self) {
        self.done_compiling()
    }

    fn is_compiling(&self) -> bool {
        self.is_compiling()
    }
}

// ── JitFrameDeadFrame (llmodel.py deadframe-as-jitframe parity) ─────

/// RPython llmodel.py parity: the deadframe IS the JitFrame.
///
/// In RPython, `execute_token` returns the JitFrame GCREF directly as
/// the deadframe. Values stay in `jf_frame[]` — no copying to `Vec<i64>`.
/// `get_int_value(deadframe, index)` reads directly from `jf_frame[index]`.
pub struct JitFrameDeadFrame {
    /// GcRef pointing to the heap-allocated JitFrame.
    pub jf_gcref: GcRef,
    /// The fail descriptor for this exit.
    pub fail_descr: Arc<CraneliftFailDescr>,
    /// GC runtime id for root cleanup on Drop.
    pub gc_runtime_id: Option<u64>,
    /// Keeps the frame memory alive for non-GC allocations.
    pub _heap_owner: Option<Vec<i64>>,
}

/// Byte offset from JitFrame start to jf_frame[0].
const JF_FRAME_ITEM0_BYTES: usize = 64;
/// Byte offset to jf_savedata field.
const JF_SAVEDATA_BYTES: usize = 32;
/// Byte offset to jf_guard_exc field.
const JF_GUARD_EXC_BYTES: usize = 40;

impl JitFrameDeadFrame {
    pub fn new(
        jf_gcref: GcRef,
        fail_descr: Arc<CraneliftFailDescr>,
        gc_runtime_id: Option<u64>,
        heap_owner: Option<Vec<i64>>,
    ) -> Self {
        let mut frame = JitFrameDeadFrame {
            jf_gcref,
            fail_descr,
            gc_runtime_id,
            _heap_owner: heap_owner,
        };
        if let Some(runtime_id) = gc_runtime_id {
            register_gc_roots(runtime_id, std::slice::from_mut(&mut frame.jf_gcref));
        }
        frame
    }

    #[inline]
    pub fn get_int(&self, index: usize) -> i64 {
        unsafe { *((self.jf_gcref.0 + JF_FRAME_ITEM0_BYTES + index * 8) as *const i64) }
    }

    #[inline]
    pub fn get_float(&self, index: usize) -> f64 {
        f64::from_bits(self.get_int(index) as u64)
    }

    #[inline]
    pub fn get_ref(&self, index: usize) -> GcRef {
        GcRef(self.get_int(index) as usize)
    }

    pub fn take_ref_for_call_result(&mut self, index: usize) -> GcRef {
        GcRef(self.get_int(index) as usize)
    }

    #[inline]
    pub fn get_savedata_ref(&self) -> GcRef {
        GcRef(unsafe { *((self.jf_gcref.0 + JF_SAVEDATA_BYTES) as *const usize) })
    }

    #[inline]
    pub fn try_get_savedata_ref(&self) -> Option<GcRef> {
        let r = self.get_savedata_ref();
        if r.is_null() { None } else { Some(r) }
    }

    #[inline]
    pub fn set_savedata_ref(&mut self, data: GcRef) {
        unsafe { *((self.jf_gcref.0 + JF_SAVEDATA_BYTES) as *mut usize) = data.0 };
    }

    #[inline]
    pub fn grab_exc_value(&self) -> GcRef {
        GcRef(unsafe { *((self.jf_gcref.0 + JF_GUARD_EXC_BYTES) as *const usize) })
    }
}

impl Drop for JitFrameDeadFrame {
    fn drop(&mut self) {
        if let Some(runtime_id) = self.gc_runtime_id {
            unregister_gc_roots(runtime_id, std::slice::from_mut(&mut self.jf_gcref));
        }
    }
}
