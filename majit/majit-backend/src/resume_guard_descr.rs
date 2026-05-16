//! `compile.py:840-940` `ResumeGuardDescr` family — per-guard descr
//! carrying both the optimizer's snapshot resume payload and the
//! backend's codegen identity.  Moved to `majit-backend` so backends
//! can instantiate it directly without depending on `majit-metainterp`.
//!
//! This is the unified-descr endpoint of the Phase C-1 cascade: with
//! `ResumeGuardDescr` reachable from backend codegen, the dynasm
//! per-emission wrapper `DynasmFailDescr` was retired (Slice 7-Tα7);
//! the cranelift counterpart still carries codegen-bound payload
//! pending Phase 7-Tβ.

use std::any::Any;
use std::cell::UnsafeCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use majit_ir::{
    AccumInfo, Const, Descr, DescrRef, FailDescr, GuardPendingFieldEntry, RdVirtualInfo, Type,
};

use crate::CompiledLoopToken;
use crate::ExitRecoveryLayout;
use crate::rd_payload::RdPayload;
use crate::resume_value::ResumeData;

// `compile.py:687-696 AbstractResumeGuardDescr` status-bit constants.
//
// Status packs three pieces in one `u64`:
//   - bit 0          : `ST_BUSY_FLAG` (set during retrace; clear once done).
//   - bits 1..3      : `ST_TYPE_MASK` — `TY_NONE` / `TY_INT` / `TY_REF` /
//                      `TY_FLOAT`, set by `make_a_counter_per_value` to
//                      distinguish guard_value-by-int / -by-ref / -by-float.
//   - bits 3..end    : jitcounter hash (when TY_NONE) or guard_value
//                      failarg index (when TY_INT/REF/FLOAT), accessed via
//                      `>> ST_SHIFT` with `STATUS_SHIFT_MASK`.
pub const STATUS_BUSY_FLAG: u64 = 0x01;
pub const STATUS_TYPE_MASK: u64 = 0x06;
pub const STATUS_SHIFT: u32 = 3;
pub const STATUS_SHIFT_MASK: u64 = !((1u64 << STATUS_SHIFT) - 1);
pub const STATUS_TY_NONE: u64 = 0x00;
pub const STATUS_TY_INT: u64 = 0x02;
pub const STATUS_TY_REF: u64 = 0x04;
pub const STATUS_TY_FLOAT: u64 = 0x06;

/// Global counter for unique fail_index allocation.
///
/// Mirrors RPython's ResumeGuardDescr numbering — each guard in every
/// compiled trace receives a unique fail_index so the backend can
/// report exactly which guard failed.
static NEXT_FAIL_INDEX: AtomicU32 = AtomicU32::new(1);

/// Reset the global fail_index counter (for testing).
pub fn reset_fail_index_counter() {
    NEXT_FAIL_INDEX.store(1, Ordering::SeqCst);
}

/// Allocate the next unique fail_index.
pub fn alloc_fail_index() -> u32 {
    NEXT_FAIL_INDEX.fetch_add(1, Ordering::SeqCst)
}

pub fn push_vector_info(head: &mut Option<Box<AccumInfo>>, mut info: AccumInfo) {
    info.prev = head.take();
    *head = Some(Box::new(info));
}

pub fn flatten_vector_info(head: Option<&AccumInfo>) -> Vec<AccumInfo> {
    let mut result = Vec::new();
    let mut current = head;
    while let Some(info) = current {
        result.push(info.clone());
        current = info.prev.as_deref();
    }
    result
}

/// `compile.py:869 self.rd_vector_info = other.rd_vector_info.clone()`
/// rebuild helper: takes the donor's flattened chain (head at index 0)
/// and assembles the equivalent linked-list head suitable for writing
/// through `vector_info: UnsafeCell<Option<Box<AccumInfo>>>`.
pub fn build_vector_info_chain(chain: Vec<AccumInfo>) -> Option<Box<AccumInfo>> {
    let mut current: Option<Box<AccumInfo>> = None;
    for mut info in chain.into_iter().rev() {
        info.prev = current;
        current = Some(Box::new(info));
    }
    current
}

/// Per-guard FailDescr that also carries resume data for deoptimization.
///
/// Mirrors RPython's ResumeGuardDescr with snapshot information.
/// When a guard fails, the backend uses the resume data to reconstruct
/// the interpreter state (virtual objects, frame variables, etc.).
#[derive(Debug)]
pub struct ResumeGuardDescr {
    pub fail_index: u32,
    /// `compile.py:869 store_final_boxes` mutates types in place; pyre
    /// uses `UnsafeCell` so identity is preserved across the optimizer.
    pub types: UnsafeCell<Vec<Type>>,
    /// Pyre keeps `resume_data` (the RPython-style ResumeValueSource
    /// payload used by `prepare_pendingfields` for the
    /// `PendingFieldInfo` path) on the descr alongside the RPython
    /// `_attrs_` rd_* slots — both representations co-exist while
    /// the runtime resume reader is being aligned with upstream.
    pub resume_data: ResumeData,
    /// `compile.py:855` `_attrs_ = ('rd_numb', 'rd_consts',
    /// 'rd_virtuals', 'rd_pendingfields', 'status')`.
    pub payload: RdPayload,
    /// RPython history.py:127 rd_vector_info — no Mutex needed, single-threaded.
    pub vector_info: UnsafeCell<Option<Box<AccumInfo>>>,
    /// `compile.py:186` `descr.rd_loop_token = clt` — owning
    /// `Arc<CompiledLoopToken>`.
    pub rd_loop_token_clt: UnsafeCell<Option<Arc<CompiledLoopToken>>>,
    /// `history.py:132` `AbstractFailDescr._attrs_ = ('adr_jump_offset',
    /// 'rd_locs', 'rd_loop_token', 'rd_vector_info')`.
    pub adr_jump_offset: UnsafeCell<usize>,
    /// `history.py:132` `AbstractFailDescr._attrs_` `rd_locs`.
    pub rd_locs: UnsafeCell<Vec<u16>>,
    /// `compile.py:683` `AbstractResumeGuardDescr._attrs_ = ('status',)`.
    pub status: AtomicU64,
    /// Pyre-only: identifier of the compiled trace that owns this guard.
    pub trace_id: AtomicU64,
    /// Pyre-only: per-trace `fail_index` assigned by `build_guard_metadata`.
    pub fail_index_per_trace: AtomicU32,
    /// Pyre-only cache: cranelift bakes recovery decisions at codegen and
    /// reads them back at deopt time through `recovery_layout_ref`.  Stored
    /// here (rather than on `CraneliftFailDescr`) so the meta Arc is the
    /// single source of truth, matching Phase A's `_attrs_` migration shape.
    /// Will be removed entirely once `rebuild_state_after_failure` is
    /// re-ported to drive `ResumeDataDirectReader` on-demand from `rd_numb`
    /// / `rd_consts` / `rd_virtuals` / `rd_pendingfields` (PyPy's
    /// `assembler.py::rebuild_locs_from_resumedata` shape — no pre-baked
    /// `ExitRecoveryLayout`).
    pub recovery_layout: UnsafeCell<Option<Arc<ExitRecoveryLayout>>>,
    /// Codegen-time trace-op index for the originating guard op
    /// (`pyjitpl._compile_one_block` parity — the live op object passed
    /// at compile time has an implicit index in `loop.operations`).
    /// Used at the backend→metainterp interop boundary
    /// (`FailDescrLayout::source_op_index`).  Migrated here from
    /// `CraneliftFailDescr::source_op_index_cell` (Slice 7-Tβ6) so the
    /// meta Arc is the single source of truth.  `None` for synthetic
    /// FINISH / external-JUMP descrs that have no associated trace op.
    pub source_op_index: UnsafeCell<Option<usize>>,
}

// Safety: single-threaded JIT (RPython GIL parity).
unsafe impl Send for ResumeGuardDescr {}
unsafe impl Sync for ResumeGuardDescr {}

impl Descr for ResumeGuardDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn is_resume_guard(&self) -> bool {
        true
    }
    /// compile.py:844-846: ResumeGuardDescr.clone()
    fn clone_descr(&self) -> Option<DescrRef> {
        Some(Arc::new(ResumeGuardDescr {
            fail_index: alloc_fail_index(),
            types: UnsafeCell::new(unsafe { (&*self.types.get()).clone() }),
            resume_data: self.resume_data.clone(),
            payload: self.payload.deep_clone(),
            vector_info: UnsafeCell::new(unsafe { (&*self.vector_info.get()).clone() }),
            // `compile.py:844-846` mints a default-attributes object;
            // the `_attrs_` slots reset to their initial values when this
            // fresh descr reaches backend codegen.
            adr_jump_offset: UnsafeCell::new(0),
            rd_locs: UnsafeCell::new(Vec::new()),
            status: AtomicU64::new(0),
            rd_loop_token_clt: UnsafeCell::new(None),
            trace_id: AtomicU64::new(0),
            fail_index_per_trace: AtomicU32::new(0),
            recovery_layout: UnsafeCell::new(None),
            source_op_index: UnsafeCell::new(None),
        }))
    }
}

impl FailDescr for ResumeGuardDescr {
    fn fail_index(&self) -> u32 {
        // `assembler.py:227 self.faildescr.index = i` — per-trace key
        // (the global `alloc_fail_index()` value lives in `self.fail_index`
        // and is exposed via `Descr::index()`).
        self.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn trace_id(&self) -> u64 {
        self.trace_id.load(Ordering::Relaxed)
    }
    fn set_trace_id(&self, trace_id: u64) {
        self.trace_id.store(trace_id, Ordering::Relaxed);
    }
    fn fail_index_per_trace(&self) -> u32 {
        self.fail_index_per_trace.load(Ordering::Relaxed)
    }
    fn set_fail_index_per_trace(&self, fail_index: u32) {
        self.fail_index_per_trace
            .store(fail_index, Ordering::Relaxed);
    }
    fn fail_arg_types(&self) -> &[Type] {
        unsafe { &*self.types.get() }
    }
    fn set_fail_arg_types(&self, types: Vec<Type>) {
        unsafe { *self.types.get() = types }
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.vector_info.get()).as_deref() })
    }
    fn replace_vector_info(&self, chain: Vec<AccumInfo>) {
        unsafe { *self.vector_info.get() = build_vector_info_chain(chain) }
    }

    fn rd_numb(&self) -> Option<&[u8]> {
        self.payload.rd_numb()
    }
    fn rd_numb_arc(&self) -> Option<Arc<[u8]>> {
        self.payload.rd_numb_arc()
    }
    fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        self.payload.set_rd_numb(value)
    }
    fn set_rd_numb_arc(&self, value: Option<Arc<[u8]>>) {
        self.payload.set_rd_numb_arc(value)
    }
    fn rd_consts(&self) -> Option<&[Const]> {
        self.payload.rd_consts()
    }
    fn rd_consts_arc(&self) -> Option<Arc<[Const]>> {
        self.payload.rd_consts_arc()
    }
    fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        self.payload.set_rd_consts(value)
    }
    fn set_rd_consts_arc(&self, value: Option<Arc<[Const]>>) {
        self.payload.set_rd_consts_arc(value)
    }
    fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        self.payload.rd_virtuals()
    }
    fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.payload.rd_virtuals_arc()
    }
    fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        self.payload.set_rd_virtuals(value)
    }
    fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        self.payload.set_rd_virtuals_arc(value)
    }
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        self.payload.rd_pendingfields()
    }
    fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.payload.rd_pendingfields_arc()
    }
    fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        self.payload.set_rd_pendingfields(value)
    }
    fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        self.payload.set_rd_pendingfields_arc(value)
    }
    fn adr_jump_offset(&self) -> usize {
        unsafe { *self.adr_jump_offset.get() }
    }
    fn set_adr_jump_offset(&self, offset: usize) {
        unsafe { *self.adr_jump_offset.get() = offset };
    }
    fn rd_locs(&self) -> &[u16] {
        unsafe { &*self.rd_locs.get() }
    }
    fn set_rd_locs(&self, locs: Vec<u16>) {
        unsafe { *self.rd_locs.get() = locs };
    }
    fn get_status(&self) -> u64 {
        self.status.load(Ordering::Acquire)
    }
    fn start_compiling(&self) {
        self.status.fetch_or(STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn done_compiling(&self) {
        self.status.fetch_and(!STATUS_BUSY_FLAG, Ordering::AcqRel);
    }
    fn store_hash(&self, hash: u64) {
        self.status
            .store(hash & STATUS_SHIFT_MASK, Ordering::Release);
    }
    fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        let value = type_tag | ((index as u64) << STATUS_SHIFT);
        self.status.store(value, Ordering::Release);
    }
    fn rd_loop_token_clt(&self) -> Option<&dyn Any> {
        let cell = unsafe { &*self.rd_loop_token_clt.get() };
        cell.as_ref().map(|arc| arc as &dyn Any)
    }
    fn set_rd_loop_token_clt(&self, clt: Arc<dyn Any + Send + Sync>) {
        let typed: Arc<CompiledLoopToken> = clt
            .downcast::<CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        unsafe { *self.rd_loop_token_clt.get() = Some(typed) };
    }
}

/// compile.py:840-843 `ResumeGuardDescr` parity: a fresh guard descr
/// carrying the post-numbering `fail_arg_types`.  `payload` initialized
/// empty; `store_final_boxes_in_guard` fills `rd_*` slots post-numbering.
pub fn make_resume_guard_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(ResumeGuardDescr {
        fail_index: alloc_fail_index(),
        types: UnsafeCell::new(types),
        resume_data: ResumeData {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: Vec::new(),
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        },
        payload: RdPayload::empty(),
        vector_info: UnsafeCell::new(None),
        adr_jump_offset: UnsafeCell::new(0),
        rd_locs: UnsafeCell::new(Vec::new()),
        status: AtomicU64::new(0),
        rd_loop_token_clt: UnsafeCell::new(None),
        trace_id: AtomicU64::new(0),
        fail_index_per_trace: AtomicU32::new(0),
        recovery_layout: UnsafeCell::new(None),
        source_op_index: UnsafeCell::new(None),
    })
}

impl ResumeGuardDescr {
    /// Read the cached recovery_layout (clone of the stored `Arc` contents).
    /// Returns `None` when codegen has not yet stamped a layout.  Pyre-only
    /// cache (no PyPy counterpart); the eventual orthodox port replaces
    /// callers with on-demand `ResumeDataDirectReader` decoding from `rd_*`.
    pub fn recovery_layout(&self) -> Option<ExitRecoveryLayout> {
        // Safety: single-threaded JIT (RPython GIL parity); same access
        // contract as the sibling `vector_info` / `rd_locs` UnsafeCells.
        unsafe {
            (*self.recovery_layout.get())
                .as_ref()
                .map(|a| (**a).clone())
        }
    }

    /// Write the cached recovery_layout.  Wraps the layout in an `Arc` to
    /// keep `recovery_layout()`'s clone-on-read path cheap when consumers
    /// only need a snapshot.
    pub fn set_recovery_layout(&self, layout: ExitRecoveryLayout) {
        // Safety: single-threaded JIT.
        unsafe { *self.recovery_layout.get() = Some(Arc::new(layout)) };
    }

    /// Read the codegen-time `source_op_index` (Slice 7-Tβ6).  `None`
    /// when codegen has not yet stamped one (synthetic descrs minted
    /// outside `_compile_one_block`).
    pub fn source_op_index(&self) -> Option<usize> {
        // Safety: single-threaded JIT.
        unsafe { *self.source_op_index.get() }
    }

    /// Write the codegen-time `source_op_index` (Slice 7-Tβ6).
    pub fn set_source_op_index(&self, source_op_index: usize) {
        // Safety: single-threaded JIT.
        unsafe { *self.source_op_index.get() = Some(source_op_index) };
    }
}
