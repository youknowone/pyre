/// assembler.py ResumeGuardDescr parity: fail descriptor; the patchable
/// jump offset (`history.py:132 _attrs_` `adr_jump_offset`) lives on the
/// metainterp `ResumeGuardDescr` (`majit-metainterp/src/compile.rs`) and
/// is accessed here via `meta_resume_fd()` forwarding.
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};

use majit_backend::ExitRecoveryLayout;
use majit_ir::{Descr, DescrRef, FailDescr, Type};

/// Re-export the shared per-cpu descr attachment types so existing
/// `crate::guard::{AttachedDescrPtrs, CpuDescrAttachments, CpuDescrHandle}`
/// imports keep resolving while the canonical definitions live in
/// `majit-backend` (shared with cranelift).
///
/// `rpython/jit/backend/model.py AbstractCPU` — descr attachments are a
/// cross-backend base-class concern; `compile.py:665-674
/// make_and_attach_done_descrs` binds them on each `cpu` instance
/// regardless of backend.
pub use majit_backend::{AttachedDescrPtrs, CpuDescrAttachments, CpuDescrHandle};

/// assembler.py: ResumeGuardDescr concrete type for dynasm backend.
pub struct DynasmFailDescr {
    pub fail_index: u32,
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,
    /// `compile.py:185` `isinstance(descr, ResumeDescr)` parity at the
    /// runtime descr layer.  Set explicitly at construction site to
    /// reflect the upstream class hierarchy:
    ///   - `ResumeGuardDescr` family (`ResumeAtPositionDescr`,
    ///     `ResumeGuardForcedDescr`, `ResumeGuardExcDescr`,
    ///     `CompileLoopVersionDescr`) → true.
    ///   - `DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef` /
    ///     `PropagateExceptionDescr` (`compile.py:1092` —
    ///     `class PropagateExceptionDescr(AbstractFailDescr)`, NOT
    ///     a `ResumeDescr`) → false.
    /// Stored explicitly because `!is_finish` is NOT equivalent to
    /// `is_resume_guard` upstream — `PropagateExceptionDescr` is
    /// `final_descr=False` AND not a `ResumeDescr`, so the predicate
    /// must come from the producer, not be derived at the use site.
    /// Dynasm has no `is_external_jump` counterpart (raw cross-loop
    /// JMP at `assembler.py:2456-2462 closing_jump` produces no fail
    /// descr), so the field is the only producer signal for the
    /// non-`ResumeDescr` non-FINISH case.
    pub is_resume_guard: bool,
    /// compile.py:658-662 ExitFrameWithExceptionDescrRef parity.
    /// True when this FINISH was emitted via
    /// pyjitpl.py:3238-3245 compile_exit_frame_with_exception.
    pub is_exit_frame_with_exception: bool,

    /// regalloc parity: fail_locs — maps fail_args[i] to jitframe slot.
    /// None = virtual/unmapped (not in jitframe).
    pub fail_arg_locs: Vec<Option<usize>>,
    /// `history.py:132` `AbstractFailDescr._attrs_` `rd_locs` —
    /// transitional dual-storage slot.  Canonical storage is on
    /// `op.descr` accessed via `meta_descr`; this backend slot is a
    /// fallback for test paths that build guards without attaching a
    /// metainterp descr.  Session 7 removes this slot.
    pub rd_locs: UnsafeCell<Vec<u16>>,

    /// Trace op index of the guard that produced this exit.
    pub source_op_index: Option<usize>,

    /// Backend-origin recovery layout, built at compile time from fail_arg_types.
    pub recovery_layout: UnsafeCell<Option<ExitRecoveryLayout>>,

    /// compile.py:685 status: packs ST_BUSY_FLAG + type tag + hash.
    pub status: AtomicU64,

    /// `history.py:132` `AbstractFailDescr._attrs_` `adr_jump_offset`
    /// — transitional dual-storage slot.  The canonical storage is on
    /// `op.descr` (the metainterp `AbstractFailDescr` Arc) accessed via
    /// `meta_descr`; this backend slot is the fallback for test paths
    /// that build guard ops without attaching a descr (`mk_op(...)`
    /// without `mk_op_with_descr`).  Session 7 will remove this slot
    /// once `DynasmFailDescr` itself is collapsed into the meta side.
    pub adr_jump_offset: UnsafeCell<usize>,

    /// Bridge code pointer (if a bridge has been compiled for this guard).
    /// Unlike Cranelift, we don't need bridge data — the machine code is
    /// patched in place to jump directly to the bridge.
    pub bridge_addr: UnsafeCell<usize>,
    // fail_args_slots removed: bridge source_slots are derived from
    // fail_arg_locs via rebuild_faillocs_from_descr (assembler.py:201).
    /// `compile.py:186` `descr.rd_loop_token = clt` line-by-line port:
    /// the owning `Arc<CompiledLoopToken>` itself. Set by
    /// `record_loop_or_bridge` (compile.py:171-211 walker).  Together
    /// with `CompiledLoopToken.loop_token_wref` (compile.py:180-181)
    /// this gives readers a direct chain `descr.rd_loop_token_clt() ->
    /// clt -> upgrade -> Arc<JitCellToken>` matching RPython's
    /// `descr.rd_loop_token.loop_token_wref()` access.
    pub rd_loop_token_clt: UnsafeCell<Option<std::sync::Arc<majit_backend::CompiledLoopToken>>>,
    /// Back-pointer to the metainterp `AbstractFailDescr` Arc the
    /// optimizer stamped onto the originating guard op (`op.descr`).
    /// PyPy keeps a single descr object per guard (`history.py:121`);
    /// pyre's transitional split-descr stores this Arc as a back-pointer
    /// so backend accessors forward
    /// `rd_numb`/`rd_consts`/`rd_virtuals`/`rd_pendingfields`/
    /// `fail_arg_types`/`adr_jump_offset`/`rd_locs` to the metainterp
    /// `AbstractFailDescr` (`history.py:132 _attrs_`).  The final
    /// Unified-Descr endpoint collapses `DynasmFailDescr` into the
    /// metainterp descr.
    ///
    /// `None` for synthetic backend descrs minted by the runtime
    /// classifier (`runner.rs::find_descr_by_ptr` for FINISH /
    /// PropagateExceptionDescr / ExitFrameWithExceptionDescr exits) —
    /// those exits route through dedicated metainterp Done* descrs
    /// owned by `MetaInterpStaticData`, not via `op.descr`.
    pub meta_descr: Option<DescrRef>,
}

// Safety: single-threaded JIT (like RPython with GIL).
unsafe impl Send for DynasmFailDescr {}
unsafe impl Sync for DynasmFailDescr {}

impl DynasmFailDescr {
    // compile.py:687-696 status encoding constants.
    pub const ST_BUSY_FLAG: u64 = 0x01;
    pub const ST_TYPE_MASK: u64 = 0x06;
    pub const ST_SHIFT: u32 = 3;
    pub const ST_SHIFT_MASK: u64 = !((1u64 << Self::ST_SHIFT) - 1);
    pub const TY_NONE: u64 = 0x00;
    pub const TY_INT: u64 = 0x02;
    pub const TY_REF: u64 = 0x04;
    pub const TY_FLOAT: u64 = 0x06;

    pub fn new(
        fail_index: u32,
        trace_id: u64,
        fail_arg_types: Vec<Type>,
        is_finish: bool,
        is_resume_guard: bool,
    ) -> Self {
        DynasmFailDescr {
            fail_index,
            trace_id,
            fail_arg_types,
            is_finish,
            is_resume_guard,
            is_exit_frame_with_exception: false,
            fail_arg_locs: Vec::new(),
            rd_locs: UnsafeCell::new(Vec::new()),
            source_op_index: None,
            recovery_layout: UnsafeCell::new(None),
            status: AtomicU64::new(0),
            adr_jump_offset: UnsafeCell::new(0),
            bridge_addr: UnsafeCell::new(0),
            rd_loop_token_clt: UnsafeCell::new(None),
            meta_descr: None,
        }
    }

    /// `compile.py:186` write side: invoked by the post-compile walker
    /// once per ResumeDescr in the newly-compiled trace.  Stamps the
    /// owning `Arc<CompiledLoopToken>`.
    pub fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<majit_backend::CompiledLoopToken>) {
        unsafe { *self.rd_loop_token_clt.get() = Some(clt) };
    }

    /// `compile.py:186` reader for the clt-typed slot.
    pub fn rd_loop_token_clt(&self) -> Option<&std::sync::Arc<majit_backend::CompiledLoopToken>> {
        unsafe { (*self.rd_loop_token_clt.get()).as_ref() }
    }

    /// `compile.py:826-830` `store_hash`.  Prefers the metainterp
    /// `AbstractResumeGuardDescr` (`compile.py:683 _attrs_`) reached
    /// through `meta_descr`; falls back to the backend-local transitional
    /// slot for test paths.
    pub fn store_hash(&self, hash: u64) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.store_hash(hash);
            return;
        }
        self.status
            .store(hash & Self::ST_SHIFT_MASK, Ordering::Release);
    }

    /// `compile.py:741-745` `get_status`.
    pub fn get_status(&self) -> u64 {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            return meta_fd.get_status();
        }
        self.status.load(Ordering::Acquire)
    }

    /// `compile.py:786-788` `start_compiling`.
    pub fn start_compiling(&self) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.start_compiling();
            return;
        }
        self.status.fetch_or(Self::ST_BUSY_FLAG, Ordering::AcqRel);
    }

    /// `compile.py:790-795` `done_compiling`.
    pub fn done_compiling(&self) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.done_compiling();
            return;
        }
        self.status.fetch_and(!Self::ST_BUSY_FLAG, Ordering::AcqRel);
    }

    /// `compile.py:813-824` `make_a_counter_per_value`.
    pub fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.make_a_counter_per_value(index, type_tag);
            return;
        }
        let status = type_tag | ((index as u64) << Self::ST_SHIFT);
        self.status.store(status, Ordering::Release);
    }

    /// `assembler.py:966` — read `adr_jump_offset`.  Prefers the
    /// metainterp `AbstractFailDescr` (`history.py:132 _attrs_`)
    /// reached through `meta_descr`; falls back to the backend-local
    /// transitional slot for test paths that build guards without
    /// attaching a descr.
    pub fn adr_jump_offset(&self) -> usize {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            return meta_fd.adr_jump_offset();
        }
        unsafe { *self.adr_jump_offset.get() }
    }

    /// `assembler.py:987` — set `adr_jump_offset` (`0` means
    /// "patched").  Writes through to the metainterp side when present;
    /// otherwise stores on the backend-local transitional slot.
    pub fn set_adr_jump_offset(&self, offset: usize) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.set_adr_jump_offset(offset);
            return;
        }
        unsafe { *self.adr_jump_offset.get() = offset };
    }

    /// `llsupport/llmodel.py:424` `descr.rd_locs[index]` — read the
    /// per-fail-arg jitframe slot positions.  Prefers the metainterp
    /// `AbstractFailDescr` (`history.py:132 _attrs_`) reached through
    /// `meta_descr`; falls back to the backend-local transitional slot.
    pub fn rd_locs(&self) -> &[u16] {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            return meta_fd.rd_locs();
        }
        unsafe { &*self.rd_locs.get() }
    }

    /// `llsupport/assembler.py:279` `guardtok.faildescr.rd_locs =
    /// positions`.  Writes through to the metainterp side when present;
    /// otherwise stores on the backend-local transitional slot.
    pub fn set_rd_locs(&self, locs: Vec<u16>) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.set_rd_locs(locs);
            return;
        }
        unsafe { *self.rd_locs.get() = locs };
    }

    /// Check if a bridge has been patched for this guard.
    pub fn has_bridge(&self) -> bool {
        unsafe { *self.bridge_addr.get() != 0 }
    }

    /// Read the compiled bridge entry address.
    pub fn bridge_addr(&self) -> usize {
        unsafe { *self.bridge_addr.get() }
    }

    /// Set the bridge address after patching.
    pub fn set_bridge_addr(&self, addr: usize) {
        unsafe { *self.bridge_addr.get() = addr };
    }

    /// Read the recovery_layout.
    pub fn recovery_layout(&self) -> Option<ExitRecoveryLayout> {
        unsafe { &*self.recovery_layout.get() }.clone()
    }

    /// Set the recovery_layout.
    pub fn set_recovery_layout(&self, layout: ExitRecoveryLayout) {
        unsafe { *self.recovery_layout.get() = Some(layout) };
    }

    /// `compile.py:185` `isinstance(descr, ResumeDescr)` gate for
    /// back-pointer forwarding.  See cranelift counterpart
    /// (`majit-backend-cranelift/src/guard.rs::meta_resume_fd`) for
    /// the full rationale: only `ResumeDescr` family meta descrs are
    /// the canonical source for fields the optimizer stamps via
    /// `record_loop_or_bridge` (`trace_id`, `fail_arg_types`,
    /// `rd_numb`, `rd_consts`, `rd_virtuals`, `rd_pendingfields`).
    /// `DoneWithThisFrame*` (`compile.py:623`),
    /// `ExitFrameWithExceptionDescrRef` (`compile.py:658-662`), and
    /// `PropagateExceptionDescr` (`compile.py:1092`) are NOT
    /// `ResumeDescr` upstream, so this returns `None` for them and
    /// callers fall back to the backend-local field set at descr
    /// construction.
    #[inline]
    fn meta_resume_fd(&self) -> Option<&dyn FailDescr> {
        let d = self.meta_descr.as_ref()?;
        if d.is_resume_guard() || d.is_resume_guard_copied() {
            d.as_fail_descr()
        } else {
            None
        }
    }

    /// Build a FailDescrLayout for this descriptor (parity with CraneliftFailDescr::layout).
    pub fn layout(&self) -> majit_backend::FailDescrLayout {
        // resume.py:450-488 propagate rd_* so `compiled_exit_layout_from_backend`
        // can reach them after the frontend trace cache evicts the owning
        // `CompiledTrace` entry (pyjitpl/mod.rs:817-845).  Read through
        // `meta_resume_fd()` — gated on isinstance(descr, ResumeDescr)
        // per `record_loop_or_bridge` (compile.py:183-185).
        let meta_fd = self.meta_resume_fd();
        let fail_arg_types = <Self as FailDescr>::fail_arg_types(self);
        majit_backend::FailDescrLayout {
            fail_index: self.fail_index,
            fail_arg_types: fail_arg_types.to_vec(),
            is_finish: self.is_finish,
            trace_id: <Self as FailDescr>::trace_id(self),
            source_op_index: self.source_op_index,
            gc_ref_slots: fail_arg_types
                .iter()
                .enumerate()
                .filter_map(|(i, tp)| (*tp == Type::Ref).then_some(i))
                .collect(),
            force_token_slots: Vec::new(),
            frame_stack: None,
            recovery_layout: self.recovery_layout(),
            trace_info: None,
            rd_numb: meta_fd.and_then(|fd| fd.rd_numb()).map(|s| s.to_vec()),
            rd_consts: meta_fd.and_then(|fd| fd.rd_consts()).map(|s| s.to_vec()),
            rd_virtuals: meta_fd.and_then(|fd| fd.rd_virtuals()).map(|s| s.to_vec()),
            rd_pendingfields: meta_fd
                .and_then(|fd| fd.rd_pendingfields())
                .map(|s| s.to_vec()),
        }
    }
}

impl std::fmt::Debug for DynasmFailDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynasmFailDescr")
            .field("fail_index", &self.fail_index)
            .field("trace_id", &self.trace_id)
            .field("is_finish", &self.is_finish)
            .field("status", &self.get_status())
            .field("adr_jump_offset", &self.adr_jump_offset())
            .field("has_bridge", &self.has_bridge())
            .finish()
    }
}

/// `compile.py:665-674` `make_and_attach_done_descrs([self, cpu])` —
/// per-result-type `DoneWithThisFrame*` singleton attached by the
/// metainterp side at `pyjitpl.py:2222`.  The `Arc` lives on
/// `MetaInterpStaticData` and is re-published here via
/// `Backend::set_done_with_this_frame_descr_*` so the CALL_ASSEMBLER
/// fast path (`runner.rs::call_assembler_helper_trampoline`) can
/// compare `jf_descr` against `Arc::as_ptr` of the same `Arc` the
/// metainterp reads back in `handle_fail`.
///
/// `compile.py:665` `setattr(cpu, name, descr)` binds the descr to a
/// specific cpu instance; each `(metainterp_sd, cpu)` pair gets its own
/// attachment, and re-running `make_and_attach_done_descrs` overwrites.
/// Pyre keeps the attachments inside a heap-pinned
/// `Arc<RwLock<CpuDescrAttachments>>` on each `DynasmBackend` instance
/// (`DynasmBackend::descr_attachments`); emission reads them via
/// `attached_descr_ptrs()` and runtime consumers dereference a baked
/// `cpu_handle` pointer.  There is no ambient thread-local — classifier
/// results always identify which cpu they were resolved against.
impl Descr for DynasmFailDescr {
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }

    /// `compile.py:185` `isinstance(descr, ResumeDescr)` parity at the
    /// runtime descr layer.  Forward through the metainterp class
    /// hierarchy when meta_descr is set (covers all real production
    /// paths: `set_meta_descr` is called at every assembler-time
    /// guard/FINISH construction, so `meta_descr.is_resume_guard()`
    /// directly answers the upstream isinstance check).  Fallback to
    /// the explicit local field for synthetic descrs minted by the
    /// runtime classifier (`runner.rs::find_descr_by_ptr` for FINISH /
    /// `ExitFrameWithExceptionDescr` / `PropagateExceptionDescr`) and
    /// for unit-test descrs that bypass the meta_descr stamp.  The
    /// local field is set at construction matching the upstream
    /// class — `!is_finish` would over-include
    /// `PropagateExceptionDescr` (final_descr=False AND not
    /// ResumeDescr) so the explicit producer-set bool is required.
    fn is_resume_guard(&self) -> bool {
        match self.meta_descr.as_ref() {
            Some(d) => d.is_resume_guard() || d.is_resume_guard_copied(),
            None => self.is_resume_guard,
        }
    }
}

impl FailDescr for DynasmFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }

    fn fail_index_per_trace(&self) -> u32 {
        // The backend descr's structural `fail_index` IS the per-trace
        // key — `assembler.py:227 self.faildescr.index = i` is allocated
        // per-trace at backend compile time.  Only the metainterp side
        // distinguishes a global `fail_index` (alloc_fail_index counter)
        // from the per-trace key; the backend has only the per-trace
        // value.  Override the trait default (0) so that callers that
        // receive the backend descr through `bridge_source_descr`'s
        // fallback chain (mod.rs:7713) can still locate the source guard.
        self.fail_index
    }

    fn fail_arg_types(&self) -> &[Type] {
        // `compile.py:185 isinstance(descr, ResumeDescr)` gate forwarding
        // via `meta_resume_fd()`.  For ResumeDescr-family meta descrs the
        // metainterp stamp via `store_final_boxes_in_guard`
        // (compile.py:869) is the single source of truth; for FINISH /
        // `ExitFrameWithExceptionDescr` / `PropagateExceptionDescr` and
        // synthetic descrs the backend local field set at construction is
        // canonical.
        self.meta_resume_fd()
            .map_or(&*self.fail_arg_types, |fd| fd.fail_arg_types())
    }

    fn is_finish(&self) -> bool {
        // Class-hierarchy property — backend-local field is set at
        // construction matching the meta_descr's class (parallel to
        // cranelift's local field).
        self.is_finish
    }

    fn is_exit_frame_with_exception(&self) -> bool {
        // Class-hierarchy property — local field set at construction.
        self.is_exit_frame_with_exception
    }

    fn trace_id(&self) -> u64 {
        // Post-audit: gate forwarding on `meta_resume_fd()`.  PyPy's
        // `record_loop_or_bridge` (compile.py:183-185) stamps trace_id
        // only on `ResumeDescr` family members; non-ResumeDescr meta
        // descrs (`DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`
        // / `PropagateExceptionDescr`) do not override `trace_id()` so
        // they would return the trait default 0, masking the
        // backend-local construction-time value.  Fallback to
        // backend-local field when meta_descr is absent or
        // non-ResumeDescr.
        self.meta_resume_fd()
            .map_or(self.trace_id, |fd| fd.trace_id())
    }

    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        // `history.py:132` `AbstractFailDescr._attrs_` `rd_loop_token` —
        // prefer the metainterp-side slot when meta_descr is attached;
        // fall back to the backend-local transitional slot.
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            if let Some(any) = meta_fd.rd_loop_token_clt() {
                return Some(any);
            }
        }
        DynasmFailDescr::rd_loop_token_clt(self).map(|arc| arc as &dyn std::any::Any)
    }

    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        // `compile.py:186` `descr.rd_loop_token = clt` — write through
        // to the metainterp side when present; otherwise stamp the
        // backend-local transitional slot.
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.set_rd_loop_token_clt(clt);
            return;
        }
        let typed: std::sync::Arc<majit_backend::CompiledLoopToken> = clt
            .downcast::<majit_backend::CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        DynasmFailDescr::set_rd_loop_token_clt(self, typed);
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
        // `compile.py:750` — read through the same forwarding chain as
        // `get_status`, so the busy-flag observation tracks the canonical
        // metainterp slot when meta_descr is set.
        self.get_status() & Self::ST_BUSY_FLAG != 0
    }

    // resume.py:450-488 readers gated on `meta_resume_fd()` —
    // `isinstance(descr, ResumeDescr)` per `record_loop_or_bridge`.
    fn rd_numb(&self) -> Option<&[u8]> {
        self.meta_resume_fd().and_then(|fd| fd.rd_numb())
    }
    fn rd_consts(&self) -> Option<&[majit_ir::Const]> {
        self.meta_resume_fd().and_then(|fd| fd.rd_consts())
    }
    fn rd_virtuals(&self) -> Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]> {
        self.meta_resume_fd().and_then(|fd| fd.rd_virtuals())
    }
    fn rd_pendingfields(&self) -> Option<&[majit_ir::GuardPendingFieldEntry]> {
        self.meta_resume_fd().and_then(|fd| fd.rd_pendingfields())
    }
}
