//! `compile.py:840-940` `ResumeGuardDescr` family — per-guard descr
//! carrying both the optimizer's snapshot resume payload and the
//! backend's codegen identity.  Moved to `majit-backend` so backends
//! can instantiate it directly without depending on `majit-metainterp`.
//!
//! This is the unified-descr endpoint: with `ResumeGuardDescr`
//! reachable from backend codegen, the dynasm per-emission wrapper
//! `DynasmFailDescr` was retired; the cranelift counterpart was
//! also retired once codegen-bound payload was lifted here.
//!
//! # Concurrency invariant (audited 2026-05-19)
//!
//! Several slots — `trace_info` (`AtomicPtr<CompiledTraceInfo>`),
//! `bridge_dispatch_cell` (`AtomicPtr<()>`), and `bridge_code_ptr_cache`
//! / `bridge_body_ptr_cache` — are accessed through atomics so that
//! JIT-baked machine code can read them without a Mutex.  The
//! `Arc::into_raw` / `Arc::increment_strong_count` / `Arc::from_raw`
//! protocol used on the dispatch / trace_info cells has a textbook
//! `load → retain` window that is unsafe under truly concurrent
//! publishers or droppers.  The protocol relies on this invariant:
//!
//! - `set_trace_info`, `bridge_dispatch_swap`, and the corresponding
//!   readers all execute on pyre's single JIT thread (RPython GIL
//!   parity).  All call paths originate from `MetaInterp` / backend
//!   codegen, both serial.
//! - `Drop::drop` for `ResumeGuardDescr` runs when the last `Arc<dyn
//!   FailDescr>` is released; a reader inside `trace_info()` /
//!   `bridge_dispatch_load()` necessarily holds such an `Arc` for the
//!   borrow lifetime, so drop cannot interleave with the load → retain
//!   window.
//! - The only background thread spawned by the driver
//!   (`jitdriver.rs:762 invalidation_thread`) touches a
//!   `Mutex<QuasiImmut>` and never reaches into `ResumeGuardDescr`.
//!
//! These three facts together close the race CodeRabbit and Codex
//! flagged on PR #68 (Critical #6/#10/#13).  Any future change that
//! introduces multi-threaded descr publishing or compilation MUST
//! replace this protocol with a hazard-pointer / RCU scheme — atomics
//! alone do not suffice.

use std::any::Any;
use std::cell::UnsafeCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use majit_ir::{
    AccumInfo, Const, Descr, DescrRef, FailDescr, GuardPendingFieldEntry, RdVirtualInfo, Type,
};

use crate::CompiledLoopToken;
use crate::CompiledTraceInfo;
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
    /// Codegen-time trace-op index for the originating guard op
    /// (`pyjitpl._compile_one_block` parity — the live op object passed
    /// at compile time has an implicit index in `loop.operations`).
    /// Used at the backend→metainterp interop boundary
    /// (`FailDescrLayout::source_op_index`).  Migrated here from
    /// the meta Arc is the single source of truth.  `None` for synthetic
    /// FINISH / external-JUMP descrs that have no associated trace op.
    pub source_op_index: UnsafeCell<Option<usize>>,
    /// Force-token slot positions for runtime GC-root filtering.
    /// PyPy encodes the same information into the machine code's
    /// GC-map immediates (`assembler.py` handles force-token slot
    /// produce/consume inline); cranelift IR has no equivalent inline
    /// encoding so the vector lives on the descr.  Migrated here from
    /// the meta Arc is the single source of truth.  Sorted and deduped
    /// at write time so `is_force_token_slot` can use `binary_search`.
    pub force_token_slots: UnsafeCell<Vec<usize>>,
    /// `AbstractResumeGuardDescr.handle_fail` (`compile.py:701-717`)
    /// drives `must_compile` via `jitcounter.tick(status_hash)` in
    /// RPython.  Pyre keeps a raw per-descr counter:
    /// the cranelift dispatch hot path calls `increment_fail_count()`
    /// once per guard failure to drive the same threshold logic.
    /// Migrated here from `CraneliftFailDescr::fail_count` so the
    /// meta Arc is the single source of truth.
    pub fail_count: AtomicU32,
    /// Per-descr `CompiledTraceInfo` cell.  PyPy
    /// recovers the same state on demand from `cpu.asmmemmgr_blocks` +
    /// `compiled_loop_token`; cranelift parks the per-trace metadata
    /// (input types / header_pc / source_guard tuple) here so the
    /// deopt and CALL_ASSEMBLER overlay paths can read it without a
    /// per-trace table lookup.  Migrated here from
    /// `CraneliftFailDescr::trace_info_cell` so the meta Arc is the
    /// single source of truth.
    ///
    /// Null on construction.  Written via
    /// `Arc::into_raw(Arc::new(info))`; `Drop` reclaims the Arc.
    pub trace_info: AtomicPtr<CompiledTraceInfo>,
    /// Per-descr external-JUMP target cell (cranelift-only TODO).
    /// PyPy's `assembler.py:2456-2462 closing_jump`
    /// emits a raw inter-function JMP to `target_token._ll_loop_code`
    /// at codegen time, so no per-descr slot exists upstream.  Cranelift
    /// IR cannot emit raw inter-function JMPs, so cross-loop JUMP descrs
    /// park the target `DescrRef` here; the dispatcher reads it and re-
    /// enters the target loop via the registered
    /// `JitCellToken.number -> RegisteredLoopTarget` metadata.
    /// Membership (`OnceLock.get().is_some()`) is the canonical
    /// `is_external_jump` predicate.
    ///
    /// Write-once: set at codegen finalisation.  Migrated here from
    /// `CraneliftFailDescr::external_jump_target_cell` so the meta Arc
    /// is the single source of truth.
    pub external_jump_target: OnceLock<DescrRef>,
    /// Bridge code-pointer cache (cranelift-only TODO: PyPy patches
    /// guard JMP targets in place).  PyPy's `assembler.py:987 patch_jump_for_descr`
    /// rewrites the guard JMP target in place when a bridge is
    /// attached; cranelift cannot patch finalised code, so the
    /// JIT-baked dispatch loads the bridge code-pointer from this
    /// cell at runtime.  Migrated here from
    /// `CraneliftFailDescr::bridge_code_ptr_cache` so the meta Arc is
    /// the single source of truth.
    ///
    /// `Box` gives the `AtomicUsize` a heap-pinned address that
    /// survives `Arc::clone` of the meta descr; cranelift's
    /// `emit_attached_bridge_dispatch` (compiler.rs:5347) embeds this
    /// address as an immediate.  `0` = no bridge attached.
    pub bridge_code_ptr_cache: Box<AtomicUsize>,
    /// Bridge body-pointer cache.  Same shape as
    /// `bridge_code_ptr_cache`, but holds the bridge's `CallConv::Tail`
    /// body entry (not the host-ABI wrapper).  The in-code dispatch
    /// (`emit_attached_bridge_dispatch`) tail-calls this so a guard
    /// failure transfers into the bridge without leaving a return frame
    /// on the machine stack — the cranelift analogue of PyPy's
    /// `patch_jump_for_descr` raw JMP.  The wrapper in
    /// `bridge_code_ptr_cache` stays for the host-loop fallback dispatch.
    pub bridge_body_ptr_cache: Box<AtomicUsize>,
    /// Bridge dispatch cell (cranelift-only TODO: PyPy patches guard
    /// JMP targets in place).  PyPy's `assembler.py:987 patch_jump_for_descr`
    /// rewrites the guard JMP target in place; cranelift cannot patch
    /// finalised code, so the runtime guard-failure dispatch loads
    /// the published `Arc<BridgeData>` raw pointer from this cell
    /// (via `bridge_dispatch_load` below) and reconstructs the Arc
    /// via `Arc::increment_strong_count + Arc::from_raw`.
    ///
    /// Type-erased to `*mut ()` because `BridgeData` lives in
    /// `majit-backend-cranelift` (downstream crate) and majit-backend
    /// must not depend on it.  The matching cleanup is registered by
    /// the backend on first `bridge_dispatch_swap` so `Drop` can
    /// reclaim the published Arc without knowing its concrete type.
    ///
    /// The cell's address is not JIT-baked (runtime reads only), so a
    /// plain `AtomicPtr<()>` suffices — `Arc::new(self)` pins the
    /// surrounding `ResumeGuardDescr`, and the field address is then
    /// stable across `Arc::clone` calls.  Null on construction (no
    /// bridge attached).
    pub bridge_dispatch_cell: AtomicPtr<()>,
    /// Type-aware cleanup function the backend registers on first
    /// `bridge_dispatch_swap`.  `Drop` invokes this with the cell's
    /// final non-null payload so the published `Arc<BridgeData>` is
    /// reclaimed by the owning crate.  `OnceLock` so the registration
    /// is idempotent across re-attach.
    pub bridge_dispatch_drop_fn: OnceLock<unsafe fn(*mut ())>,
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
            source_op_index: UnsafeCell::new(None),
            force_token_slots: UnsafeCell::new(Vec::new()),
            fail_count: AtomicU32::new(0),
            trace_info: AtomicPtr::new(std::ptr::null_mut()),
            external_jump_target: OnceLock::new(),
            bridge_code_ptr_cache: Box::new(AtomicUsize::new(0)),
            bridge_body_ptr_cache: Box::new(AtomicUsize::new(0)),
            bridge_dispatch_cell: AtomicPtr::new(std::ptr::null_mut()),
            bridge_dispatch_drop_fn: OnceLock::new(),
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
    fn source_op_index(&self) -> Option<usize> {
        // Safety: single-threaded JIT.
        unsafe { *self.source_op_index.get() }
    }
    fn set_source_op_index(&self, source_op_index: usize) {
        // Safety: single-threaded JIT.
        unsafe { *self.source_op_index.get() = Some(source_op_index) };
    }
    fn force_token_slots(&self) -> Vec<usize> {
        // Safety: single-threaded JIT.
        unsafe { (&*self.force_token_slots.get()).clone() }
    }
    fn set_force_token_slots(&self, mut slots: Vec<usize>) {
        slots.sort_unstable();
        slots.dedup();
        // Safety: single-threaded JIT.
        unsafe { *self.force_token_slots.get() = slots };
    }
    fn fail_count(&self) -> u32 {
        self.fail_count.load(Ordering::Relaxed)
    }
    fn increment_fail_count(&self) -> u32 {
        self.fail_count.fetch_add(1, Ordering::Relaxed) + 1
    }
    fn trace_info_any(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        let ptr = self.trace_info.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            // Safety: stored via `Arc::into_raw(Arc::new(info))` in
            // `set_trace_info_any` / `set_trace_info`.  Bump the strong
            // count and reconstruct so the caller gets an owning Arc
            // without taking ownership from the cell.
            //
            // The `load` → `increment_strong_count` window is sound under
            // pyre's single-JIT-thread invariant (RPython GIL parity):
            // `set_trace_info_any` and `Drop::drop` only run on the JIT
            // compiler thread.  Reads happen either from the same thread
            // (codegen helpers like `fail_descr_trace_info`) or from JIT-
            // baked code executed under the same GIL-equivalent
            // serialization.  A re-publishing `set_trace_info_any` cannot
            // interleave between this load and the strong-count bump,
            // so the pointed-to `Arc` cannot be freed mid-protocol.  The
            // atomic primitive exists so JIT-baked machine code can
            // address the cell without a lock; cross-thread concurrency
            // is not part of the invariant.
            unsafe {
                Arc::increment_strong_count(ptr as *const CompiledTraceInfo);
                let arc = Arc::from_raw(ptr as *const CompiledTraceInfo);
                Some(arc as Arc<dyn Any + Send + Sync>)
            }
        }
    }
    fn set_trace_info_any(&self, info: Arc<dyn Any + Send + Sync>) {
        let typed: Arc<CompiledTraceInfo> = info
            .downcast::<CompiledTraceInfo>()
            .expect("set_trace_info_any expected Arc<CompiledTraceInfo>");
        let new_ptr = Arc::into_raw(typed) as *mut CompiledTraceInfo;
        let old_ptr = self.trace_info.swap(new_ptr, Ordering::AcqRel);
        if !old_ptr.is_null() {
            // Safety: prior swap produced this pointer via the same
            // `Arc::into_raw(Arc::new(...))` invariant.
            unsafe { drop(Arc::from_raw(old_ptr as *const CompiledTraceInfo)) };
        }
    }
    fn bridge_cache_addrs(&self) -> Option<(usize, usize)> {
        Some((
            self.bridge_code_ptr_cache.as_ref() as *const _ as usize,
            self.bridge_body_ptr_cache.as_ref() as *const _ as usize,
        ))
    }
    fn bridge_code_ptr(&self) -> usize {
        self.bridge_code_ptr_cache.load(Ordering::Acquire)
    }
    fn store_bridge_caches(&self, code_ptr: usize, body_ptr: usize) {
        self.bridge_body_ptr_cache
            .store(body_ptr, Ordering::Release);
        self.bridge_code_ptr_cache
            .store(code_ptr, Ordering::Release);
    }
    fn bridge_dispatch_load(&self) -> *mut () {
        self.bridge_dispatch_cell.load(Ordering::Acquire)
    }
    fn bridge_dispatch_swap(&self, new_ptr: *mut (), drop_fn: unsafe fn(*mut ())) -> *mut () {
        // Forward to the inherent method so the re-attach cleanup-fn
        // identity assertion lives in one place.
        ResumeGuardDescr::bridge_dispatch_swap(self, new_ptr, drop_fn)
    }

    /// `assembler.py:2456-2462 closing_jump` parity: external JUMP exits
    /// are routed through a synthesised `ResumeGuardDescr` whose
    /// `external_jump_target` slot carries the cross-loop TargetToken
    /// `DescrRef`.  Membership in the slot IS the external-JUMP
    /// predicate.
    fn is_external_jump(&self) -> bool {
        self.external_jump_target.get().is_some()
    }

    /// `history.py:470` `TargetToken._ll_loop_code` parity: when this
    /// descr is the synthesised cross-loop JUMP exit, surface the target
    /// `DescrRef` the dispatcher re-enters via.  `None` for regular
    /// guard descrs.
    fn target_descr(&self) -> Option<DescrRef> {
        self.external_jump_target.get().cloned()
    }

    /// Trait override of `set_external_jump_target` forwarding to the
    /// inherent method; reuses the same write-once
    /// semantics so trait dispatch on `&dyn FailDescr` lands here for
    /// the cross-loop JUMP target publish in `collect_guards`.
    fn set_external_jump_target(&self, target: DescrRef) {
        ResumeGuardDescr::set_external_jump_target(self, target);
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
        source_op_index: UnsafeCell::new(None),
        force_token_slots: UnsafeCell::new(Vec::new()),
        fail_count: AtomicU32::new(0),
        trace_info: AtomicPtr::new(std::ptr::null_mut()),
        external_jump_target: OnceLock::new(),
        bridge_code_ptr_cache: Box::new(AtomicUsize::new(0)),
        bridge_body_ptr_cache: Box::new(AtomicUsize::new(0)),
        bridge_dispatch_cell: AtomicPtr::new(std::ptr::null_mut()),
        bridge_dispatch_drop_fn: OnceLock::new(),
    })
}

impl ResumeGuardDescr {
    /// Read the codegen-time `source_op_index`.  `None`
    /// when codegen has not yet stamped one (synthetic descrs minted
    /// outside `_compile_one_block`).
    pub fn source_op_index(&self) -> Option<usize> {
        // Safety: single-threaded JIT.
        unsafe { *self.source_op_index.get() }
    }

    /// Write the codegen-time `source_op_index`.
    pub fn set_source_op_index(&self, source_op_index: usize) {
        // Safety: single-threaded JIT.
        unsafe { *self.source_op_index.get() = Some(source_op_index) };
    }

    /// Read the codegen-time `force_token_slots`.
    /// Returns `&[]` when codegen has not stamped any slots (the
    /// common case for guards that do not produce force tokens).
    pub fn force_token_slots(&self) -> &[usize] {
        // Safety: single-threaded JIT.
        unsafe { &*self.force_token_slots.get() }
    }

    /// Write the codegen-time `force_token_slots`.
    /// Sorts + dedups so the stored vector satisfies the
    /// `binary_search` invariant used by
    /// `CraneliftFailDescr::is_force_token_slot`.
    pub fn set_force_token_slots(&self, mut slots: Vec<usize>) {
        slots.sort_unstable();
        slots.dedup();
        // Safety: single-threaded JIT.
        unsafe { *self.force_token_slots.get() = slots };
    }

    /// Increment the per-descr `fail_count`.  Returns
    /// the post-increment value.  Mirrors PyPy's `jitcounter.tick`
    /// semantics: one increment per observed guard failure, drives
    /// `must_compile` threshold in `compile.py:701-717`.
    pub fn increment_fail_count(&self) -> u32 {
        self.fail_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Read the per-descr `fail_count`.
    pub fn get_fail_count(&self) -> u32 {
        self.fail_count.load(Ordering::Relaxed)
    }

    /// Publish the per-trace `CompiledTraceInfo` into the descr-local
    /// atomic cell.  Any previously published Arc is
    /// reclaimed by the swap.
    pub fn set_trace_info(&self, info: CompiledTraceInfo) {
        let new_ptr = Arc::into_raw(Arc::new(info)) as *mut CompiledTraceInfo;
        let old_ptr = self.trace_info.swap(new_ptr, Ordering::AcqRel);
        if !old_ptr.is_null() {
            // Safety: prior `set_trace_info` published this pointer;
            // reclaim ownership and drop.
            unsafe { drop(Arc::from_raw(old_ptr as *const CompiledTraceInfo)) };
        }
    }

    /// Read the per-trace `CompiledTraceInfo`.
    /// Returns an owned clone of the published value, or `None` when
    /// no trace info has been published.  Lock-free.
    ///
    /// The `load` → `increment_strong_count` window relies on pyre's
    /// single-JIT-thread invariant (RPython GIL parity): no concurrent
    /// `set_trace_info` / `Drop::drop` can interleave between the load
    /// and the strong-count bump because all publishers run on the JIT
    /// compiler thread and readers run under the same serialization.
    /// The `AtomicPtr` exists so JIT-baked machine code can read the
    /// cell without a mutex, not to support cross-thread publishing.
    pub fn trace_info(&self) -> Option<CompiledTraceInfo> {
        let ptr = self.trace_info.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            // Safety: `ptr` was produced by `Arc::into_raw(Arc::new(info))`
            // in `set_trace_info`; increment_strong_count + from_raw
            // yields an extra owning Arc the caller can deref + clone.
            // Single-thread invariant above prevents UAF.
            unsafe {
                Arc::increment_strong_count(ptr as *const CompiledTraceInfo);
                let arc = Arc::from_raw(ptr as *const CompiledTraceInfo);
                Some((*arc).clone())
            }
        }
    }

    /// Publish the external-JUMP target.  Write-once;
    /// panics if invoked twice on the same descr (mirrors PyPy's
    /// `assembler.py:2456-2462 closing_jump` codegen-time finality —
    /// the target is determined at trace emission and never revised).
    pub fn set_external_jump_target(&self, target: DescrRef) {
        self.external_jump_target
            .set(target)
            .expect("external_jump_target already published");
    }

    /// Read the external-JUMP target.  `None` for
    /// regular guard descrs (the common case); `Some` only for the
    /// cranelift-synthesised cross-loop JUMP descrs.
    pub fn external_jump_target(&self) -> Option<DescrRef> {
        self.external_jump_target.get().cloned()
    }

    /// Predicate (`is_external_jump` parity) — membership in the
    /// external-JUMP target cell.
    pub fn is_external_jump(&self) -> bool {
        self.external_jump_target.get().is_some()
    }

    /// Heap-pinned addresses of the two bridge-cache atomic cells
    /// suitable for baking into JIT machine code as
    /// immediates.  Returns `(code_ptr_addr, body_ptr_addr)`.
    pub fn bridge_cache_addrs(&self) -> (usize, usize) {
        (
            self.bridge_code_ptr_cache.as_ref() as *const _ as usize,
            self.bridge_body_ptr_cache.as_ref() as *const _ as usize,
        )
    }

    /// Atomically store the bridge code-pointer + frame-depth caches
    /// Called from cranelift `attach_bridge` after
    /// the bridge has been compiled.
    pub fn store_bridge_caches(&self, code_ptr: usize, body_ptr: usize) {
        self.bridge_body_ptr_cache
            .store(body_ptr, Ordering::Release);
        self.bridge_code_ptr_cache
            .store(code_ptr, Ordering::Release);
    }

    /// Read the cached bridge code-pointer.  `0` when
    /// no bridge is attached.
    pub fn bridge_code_ptr(&self) -> usize {
        self.bridge_code_ptr_cache.load(Ordering::Acquire)
    }

    /// Read the type-erased bridge dispatch cell.
    /// Returns the published raw pointer for the backend to
    /// reconstruct its concrete `Arc<BridgeData>` via
    /// `Arc::increment_strong_count + Arc::from_raw`.  Null when no
    /// bridge has been attached.
    pub fn bridge_dispatch_load(&self) -> *mut () {
        self.bridge_dispatch_cell.load(Ordering::Acquire)
    }

    /// Atomic-swap a new bridge dispatch payload into the cell and
    /// register the backend-supplied cleanup function.
    /// Returns the previous payload so the backend can reclaim its
    /// owned `Arc`.  The cleanup function is registered once
    /// (idempotent) and invoked by `Drop` on any payload still in the
    /// cell at descr teardown.
    pub fn bridge_dispatch_swap(&self, new_ptr: *mut (), drop_fn: unsafe fn(*mut ())) -> *mut () {
        // Re-attach must use the same cleanup function the first
        // attach registered: `Drop` reclaims the surviving payload via
        // the stored `drop_fn`, so a mismatched destructor on a later
        // re-attach would type-pun the payload and corrupt the
        // backend's owned Arc.  Compare function pointers by raw
        // address — `fn` is `Copy` and equality on raw fn-pointers is
        // well-defined for monomorphised functions baked at codegen.
        if let Some(existing) = self.bridge_dispatch_drop_fn.get() {
            assert_eq!(
                *existing as usize, drop_fn as usize,
                "bridge_dispatch_swap re-attach registered a different cleanup fn \
                 for the same descr — payload type-shape must be stable across \
                 re-attach (otherwise Drop would type-pun the survivor)",
            );
        } else {
            let _ = self.bridge_dispatch_drop_fn.set(drop_fn);
        }
        self.bridge_dispatch_cell.swap(new_ptr, Ordering::AcqRel)
    }
}

impl Drop for ResumeGuardDescr {
    fn drop(&mut self) {
        // Reclaim any published `Arc<CompiledTraceInfo>`
        // by swapping the cell to null and reconstructing the Arc so
        // its Drop runs.
        let ptr = self.trace_info.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !ptr.is_null() {
            // Safety: produced by `Arc::into_raw(Arc::new(info))` in
            // `set_trace_info`.
            unsafe { drop(Arc::from_raw(ptr as *const CompiledTraceInfo)) };
        }
        // Reclaim any published bridge dispatch payload
        // via the backend-registered cleanup function.  Swap-to-null
        // first so a concurrent reader either sees the still-live
        // pointer (and bumps the strong count) or null (and skips).
        let bridge_ptr = self
            .bridge_dispatch_cell
            .swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !bridge_ptr.is_null() {
            if let Some(drop_fn) = self.bridge_dispatch_drop_fn.get() {
                // Safety: `drop_fn` was registered via `bridge_dispatch_swap`
                // alongside the payload at `bridge_ptr`; the publisher's
                // safety contract is that the function reclaims a value of
                // the same shape the publisher published.
                unsafe { drop_fn(bridge_ptr) };
            }
            // else: payload published with no cleanup registered — a
            // backend bug.  Leaks rather than risks reading the wrong
            // concrete type.
        }
    }
}
