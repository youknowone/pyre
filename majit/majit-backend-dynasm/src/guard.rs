/// assembler.py ResumeGuardDescr parity: fail descriptor; the patchable
/// jump offset (`history.py:132 _attrs_` `adr_jump_offset`) lives on the
/// metainterp `ResumeGuardDescr` (`majit-metainterp/src/compile.rs`) and
/// is accessed here via `meta_resume_fd()` forwarding.
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Backend-static side-table mapping a `DynasmFailDescr` Arc's
/// `Arc::as_ptr` address to the codegen-time `source_op_index`
/// (the index of the trace op that produced this exit).
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) does not
/// carry this slot — RPython's `assembler.py` never re-fetches the
/// op index after codegen.  Pyre keeps it because backend layouts
/// (`FailDescrLayout::source_op_index`) cross the backend→metainterp
/// boundary and the metainterp consumer needs to align deadframe
/// metadata with the trace it came from.  Sharing the same shape
/// as the cranelift counterpart (`majit-backend-cranelift/src/
/// guard.rs::SOURCE_OP_INDEX_TABLE`).
static SOURCE_OP_INDEX_TABLE: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();

fn source_op_index_table() -> &'static Mutex<HashMap<usize, usize>> {
    SOURCE_OP_INDEX_TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_source_op_index(descr_ptr: usize, op_index: usize) {
    source_op_index_table()
        .lock()
        .expect("SOURCE_OP_INDEX_TABLE mutex poisoned")
        .insert(descr_ptr, op_index);
}

pub fn lookup_source_op_index(descr_ptr: usize) -> Option<usize> {
    source_op_index_table()
        .lock()
        .expect("SOURCE_OP_INDEX_TABLE mutex poisoned")
        .get(&descr_ptr)
        .copied()
}

use majit_backend::ExitRecoveryLayout;
use majit_ir::{Descr, DescrRef, FailDescr, Type};

/// Backend-static descr-by-address registry, mirroring the per-backend
/// `DynasmBackend::fail_descr_registry` so the `call_assembler_helper_trampoline`
/// (which has no `&DynasmBackend` reachable from its `extern "C"` body)
/// can recover the descr identity without dereferencing the raw pointer
/// as `*const DynasmFailDescr`.  Stored as `Weak<dyn Descr>` so the
/// global table does not keep evicted descrs alive past their owning
/// `CompiledLoop`/`CompiledBridge`; the per-backend registry retains a
/// strong reference for the live set.
static FAIL_DESCR_REGISTRY_GLOBAL: OnceLock<Mutex<HashMap<usize, std::sync::Weak<dyn Descr>>>> =
    OnceLock::new();

fn fail_descr_registry_global() -> &'static Mutex<HashMap<usize, std::sync::Weak<dyn Descr>>> {
    FAIL_DESCR_REGISTRY_GLOBAL.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_fail_descr_global(descr_ptr: usize, descr: &DescrRef) {
    // `insert` (not `entry().or_insert_with`): after the original descr
    // at this address is dropped, the allocator may reuse the address
    // for a freshly registered descr.  `or_insert_with` would leave the
    // stale `Weak` in place and `lookup_fail_descr_global` would keep
    // upgrading to `None`, dropping live guard failures into the
    // fallback `handle_fail_dispatch == 0` path.
    fail_descr_registry_global()
        .lock()
        .expect("FAIL_DESCR_REGISTRY_GLOBAL mutex poisoned")
        .insert(descr_ptr, std::sync::Arc::downgrade(descr));
}

pub fn lookup_fail_descr_global(descr_ptr: usize) -> Option<DescrRef> {
    fail_descr_registry_global()
        .lock()
        .expect("FAIL_DESCR_REGISTRY_GLOBAL mutex poisoned")
        .get(&descr_ptr)
        .and_then(|w| w.upgrade())
}

// RECOVERY_LAYOUT_TABLE removed (Slice NN): `recovery_layout` is not in
// PyPy `AbstractFailDescr._attrs_` (`history.py:132`).  Upstream resume
// code (`resume.py:450-488`) decodes recovery on demand from
// `rd_numb / rd_consts / rd_virtuals / rd_pendingfields`.  Pyre's
// metainterp `StoredExitLayout.recovery_layout` (populated by
// `compile.rs::patch_backend_guard_recovery_layouts_for_trace` from
// the resume snapshot) is the canonical store; the backend no longer
// caches.  At deopt, `FailDescrLayout.recovery_layout` is `None` and
// `pyjitpl/mod.rs:6322` falls back to `trace_layout_ref.recovery_layout`
// which reads from the metainterp cache.

// FAIL_ARG_LOCS_TABLE removed (Slice MM): duplicates `rd_locs`
// (`history.py:132 _attrs_`).  All readers (`lib.rs:handle_fail_*` and
// `runner.rs::execute_token`) now decode `descr.rd_locs()` directly
// via `decode_rd_loc_slot` below, matching PyPy's
// `llmodel.py:422-424 _decode_pos`.

/// PyPy `llmodel.py:422-424 _decode_pos` parity.  Translate one
/// `rd_locs[index]` entry into the jitframe slot pyre's
/// `get_int_value_direct(jf, slot)` consumes.  Returns `None` for
/// 0xFFFF (unmapped — resume system handles via `rd_numb`
/// TAGCONST/TAGVIRTUAL encoding) or for out-of-range indices.
#[inline]
pub fn decode_rd_loc_slot(descr: &dyn FailDescr, index: usize) -> Option<usize> {
    let pos = *descr.rd_locs().get(index)?;
    if pos == 0xFFFF {
        None
    } else {
        Some(pos as usize)
    }
}

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
    // is_finish removed: `compile.py:624 final_descr=True` is a class
    // attribute on `_DoneWithThisFrameDescr`/`ExitFrameWithExceptionDescrRef`.
    // After Phase A, every DynasmFailDescr — production guard, FINISH,
    // GuardNotForced pre-allocation, synthetic find_descr_by_ptr exit —
    // carries meta_descr to the corresponding metainterp class
    // (ResumeGuardDescr family for guards, DoneWithThisFrame*/
    // ExitFrameWithExceptionDescrRef/PropagateExceptionDescr for finish-
    // ish exits), so the FailDescr trait method answers via meta_descr
    // forwarding instead of a backend-local mirror.
    // status removed: `compile.py:683 AbstractResumeGuardDescr._attrs_
    // = ('status',)` — only ResumeGuardDescr family carries this slot.
    // Done*/Exit/Propagate inherit AbstractFailDescr without status.
    // After Phase A every backend descr forwards through meta_descr to
    // the metainterp class, so the local AtomicU64 mirror is unused.
    // is_resume_guard removed: the predicate is answered by the
    // metainterp class hierarchy through `meta_descr` forwarding
    // (`compile.py:185` `isinstance(descr, ResumeDescr)` test against
    // the actual `ResumeGuardDescr` / `DoneWithThisFrame*` /
    // `PropagateExceptionDescr` Arc reached via `op.descr`).  Production
    // codegen sets `meta_descr` immediately after constructing the
    // backend descr; synthetic `find_descr_by_ptr` exits also carry
    // `meta_descr` to the attached metainterp Arc.  Backend-only test
    // harnesses no longer mint plain `DynasmFailDescr` for the
    // Done*/Exit attachments — `attach_default_test_descrs` uses the
    // class-distinct Done* types directly.
    // is_exit_frame_with_exception removed: `compile.py:658-662
    // ExitFrameWithExceptionDescrRef` is a class identity on the
    // metainterp side.  After the FINISH-meta_descr stamping commit
    // every backend descr that should answer true for this predicate
    // (compile_exit_frame_with_exception emission, find_descr_by_ptr
    // ExitFrame and Propagate synthetic exits) carries meta_descr to
    // the metainterp ExitFrameWithExceptionDescrRef Arc.  The Propagate
    // path stamps ExitFrameWithExceptionDescrRef rather than
    // PropagateExceptionDescr because `compile.py:1092 handle_fail`
    // raises `jitexc.ExitFrameWithExceptionRef` — the dispatcher sees
    // the resulting class identity, not the originating Propagate.

    // fail_arg_locs removed (Slice MM): the per-fail-arg slot positions
    // live in `rd_locs` (`history.py:132 _attrs_`) on the metainterp
    // `AbstractFailDescr`.  PyPy `llsupport/assembler.py:248-279
    // store_info_on_descr` writes them there; pyre matches.  Runtime
    // readers (`runner.rs::execute_token`, `lib.rs::handle_fail_*`)
    // decode `descr.rd_locs()[i]` via `decode_rd_loc_slot` —
    // PyPy `llmodel.py:422-424 _decode_pos` parity.
    // rd_locs removed (Session 5g-1, paired with adr_jump_offset
    // 5e-1): canonical storage is on the metainterp
    // `AbstractFailDescr` Arc via `meta_descr`.  Backend access goes
    // through `Self::rd_locs()` which forwards to the meta side.

    // source_op_index removed (Session 5i-cl parity): not in PyPy
    // `AbstractFailDescr._attrs_` (`history.py:132`).  The codegen-
    // time trace-op index lives in `SOURCE_OP_INDEX_TABLE` keyed on
    // the descr's inner address.
    /// Backend-origin recovery layout, built at compile time from fail_arg_types.
    // recovery_layout removed (Session 7): not in PyPy
    // `AbstractFailDescr._attrs_` (`history.py:132`).  The structured
    // layout lives in `RECOVERY_LAYOUT_TABLE` keyed on
    // `Arc::as_ptr(&descr)`.

    // adr_jump_offset removed (Session 5e-1): the canonical
    // `history.py:132 _attrs_` slot lives on the metainterp
    // `AbstractFailDescr` Arc reached via `meta_descr`.  The previous
    // backend-local fallback is unreachable in production codegen —
    // all guard ops carry a `meta_descr`, and the synthetic FINISH /
    // ExitFrameWithException / PropagateException descrs minted by
    // the runtime classifier never have their `adr_jump_offset`
    // accessed.
    // fail_args_slots removed: bridge source_slots are derived from
    // fail_arg_locs via rebuild_faillocs_from_descr (assembler.py:201).
    // bridge_addr removed (Session 5f): not in PyPy `AbstractFailDescr._attrs_`
    // (`history.py:132`).  Pyre's backend-internal bridge-entry lookup
    // moved to `DynasmBackend::bridge_addr_by_descr` side-table keyed on
    // the source descr's `Arc::as_ptr` address.
    // rd_loop_token_clt removed: `history.py:132 AbstractFailDescr._attrs_`
    // `rd_loop_token` lives on the metainterp Arc.  Only ResumeDescr
    // family descrs receive `record_loop_or_bridge`'s
    // `descr.rd_loop_token = clt` stamp (compile.py:183-186); pyre's
    // walker (runner.rs:1518-1524) gates the call on
    // `descr.is_resume_guard()` so the write always lands on the
    // metainterp ResumeGuardDescr through meta_descr forwarding.
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
    /// Always `Some` in production codegen paths.  Production guard
    /// descrs stamp `meta_descr = op.descr` immediately after
    /// construction (`x86/aarch64 assembler.rs::append_guard_token_with_faillocs`);
    /// FINISH descrs stamp the matching DoneWithThisFrameDescr* Arc
    /// (`done_with_this_frame_descr_arc_for_type`); synthetic
    /// `find_descr_by_ptr` exits stamp the cloned attached Arc from
    /// `descr_attachments`.  Only test scaffolding (lib.rs:780/804
    /// helper-trampoline tests) leaves it `None`.
    pub meta_descr: Option<DescrRef>,
}

// Safety: single-threaded JIT (like RPython with GIL).
unsafe impl Send for DynasmFailDescr {}
unsafe impl Sync for DynasmFailDescr {}

impl Drop for DynasmFailDescr {
    /// Backend-static side-table `SOURCE_OP_INDEX_TABLE` is keyed on
    /// the descr's inner address.
    /// Without cleanup the entry would outlive the descr and a future
    /// descr at the same reused address would observe stale state.
    /// Same lifecycle discipline as `CraneliftFailDescr` (see its
    /// `Drop` impl).
    fn drop(&mut self) {
        let ptr = self as *const Self as usize;
        source_op_index_table()
            .lock()
            .expect("SOURCE_OP_INDEX_TABLE mutex poisoned")
            .remove(&ptr);
        // recovery_layout no longer cached on backend (Slice NN).
        // `Self::drop` runs after the registry's strong ref is dropped,
        // so if any other holders exist they hold a strong ref via
        // `DescrRef` (the global mirror) — when the strong count reaches
        // zero the global entry is the last holder.  Remove ourselves
        // unconditionally to keep the global mirror in sync.
        fail_descr_registry_global()
            .lock()
            .expect("FAIL_DESCR_REGISTRY_GLOBAL mutex poisoned")
            .remove(&ptr);
    }
}

impl DynasmFailDescr {
    /// Production codegen sites stamp `meta_descr` to the metainterp
    /// `AbstractFailDescr` Arc immediately after construction; threading
    /// it through the constructor avoids any post-wrap mutation.
    pub fn with_meta(
        fail_index: u32,
        trace_id: u64,
        fail_arg_types: Vec<Type>,
        meta_descr: Option<DescrRef>,
    ) -> Self {
        DynasmFailDescr {
            fail_index,
            trace_id,
            fail_arg_types,
            meta_descr,
        }
    }

    // recovery_layout / set_recovery_layout removed (Slice NN): see
    // module-level comment.  Backend no longer caches; metainterp
    // `StoredExitLayout.recovery_layout` is the canonical store.

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
        layout_for_fail_descr(self, self as *const Self as *const () as usize)
    }
}

/// `compile.py:849` build a FailDescrLayout snapshot for any FailDescr
/// descr identity (post Phase C-1 cascade — `find_descr_by_ptr` returns
/// `DescrRef`, so the layout extractor takes the trait object plus the
/// data-pointer address used to key backend-static side-tables
/// (`SOURCE_OP_INDEX_TABLE`, `RECOVERY_LAYOUT_TABLE`)).
pub fn layout_for_fail_descr(
    fd: &dyn FailDescr,
    descr_addr: usize,
) -> majit_backend::FailDescrLayout {
    // resume.py:450-488 propagate rd_* so `compiled_exit_layout_from_backend`
    // can reach them after the frontend trace cache evicts the owning
    // `CompiledTrace` entry (pyjitpl/mod.rs:817-845).
    let fail_arg_types = fd.fail_arg_types();
    majit_backend::FailDescrLayout {
        fail_index: fd.fail_index_per_trace(),
        fail_arg_types: fail_arg_types.to_vec(),
        is_finish: fd.is_finish(),
        trace_id: fd.trace_id(),
        source_op_index: lookup_source_op_index(descr_addr),
        // Forward through `FailDescr::is_gc_ref_slot` / `force_token_slots`
        // so concrete descrs that override these (e.g. cranelift descrs
        // that suppress force-token producer slots from GC classification)
        // contribute correct metadata to the layout.
        gc_ref_slots: (0..fail_arg_types.len())
            .filter(|&i| fd.is_gc_ref_slot(i))
            .collect(),
        force_token_slots: fd.force_token_slots(),
        frame_stack: None,
        // Slice NN: backend no longer caches recovery_layout.  The
        // metainterp `pyjitpl/mod.rs:6322` falls back to
        // `trace_layout_ref.recovery_layout` (read from
        // `StoredExitLayout.recovery_layout`).
        recovery_layout: None,
        trace_info: None,
        rd_numb: fd.rd_numb().map(|s| s.to_vec()),
        rd_consts: fd.rd_consts().map(|s| s.to_vec()),
        rd_virtuals: fd.rd_virtuals().map(|s| s.to_vec()),
        rd_pendingfields: fd.rd_pendingfields().map(|s| s.to_vec()),
    }
}

impl std::fmt::Debug for DynasmFailDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynasmFailDescr")
            .field("fail_index", &self.fail_index)
            .field("trace_id", &self.trace_id)
            .field("is_finish", &<Self as FailDescr>::is_finish(self))
            .field("status", &self.get_status())
            .field("adr_jump_offset", &self.adr_jump_offset())
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
        // `compile.py:185` `isinstance(descr, ResumeDescr)` — answered by
        // forwarding to the metainterp class hierarchy via meta_descr.
        // Synthetic descrs without meta_descr (test scaffolding for the
        // helper trampoline) take the trait default false; their
        // ResumeDescr-classification predicate is irrelevant in those
        // paths.
        self.meta_descr
            .as_ref()
            .map_or(false, |d| d.is_resume_guard() || d.is_resume_guard_copied())
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
        // `compile.py:624` `_DoneWithThisFrameDescr` family carries
        // `final_descr = True`.  After Phase A every DynasmFailDescr
        // carries meta_descr (production codegen + synthetic
        // find_descr_by_ptr exits both stamp it), so the metainterp
        // class hierarchy answers the predicate.  Synthetic descrs
        // without meta_descr (test scaffolding for the helper trampoline)
        // take the trait default false; their FINISH-flavour predicate
        // is irrelevant in those paths.
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .map_or(false, |fd| fd.is_finish())
    }

    fn is_exit_frame_with_exception(&self) -> bool {
        // `compile.py:658-662 ExitFrameWithExceptionDescrRef`'s identity
        // lives on the metainterp Arc.  After the FINISH-meta_descr
        // stamping commit every production codegen + synthetic
        // find_descr_by_ptr exit carries meta_descr; the trait method
        // forwards through it.  Synthetic descrs without meta_descr
        // (test scaffolding) take the trait default false.
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .map_or(false, |fd| fd.is_exit_frame_with_exception())
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
        // forward through `meta_descr` to the metainterp ResumeGuardDescr
        // slot.  `record_loop_or_bridge` only stamps ResumeDescr family
        // descrs (compile.py:183-186), so meta_descr is always present
        // when this read fires; absent meta_descr indicates a synthetic
        // FINISH/Exit/Propagate descr that never visits the walker.
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .and_then(|fd| fd.rd_loop_token_clt())
    }

    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        // `compile.py:186` `descr.rd_loop_token = clt` — write through
        // to the metainterp ResumeGuardDescr.  Caller (runner.rs walker)
        // gates on `descr.is_resume_guard()` before invocation, so
        // meta_descr is always present here in production.
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.set_rd_loop_token_clt(clt);
        }
    }

    /// `compile.py:741-745` `get_status`.  `compile.py:683`
    /// `AbstractResumeGuardDescr._attrs_ = ('status',)` — only resume
    /// guard descrs carry status.  Forwards through `meta_descr`;
    /// non-ResumeGuardDescr targets (Done*/Exit/Propagate) return the
    /// trait default 0, matching the upstream class hierarchy.
    fn get_status(&self) -> u64 {
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .map_or(0, |fd| fd.get_status())
    }

    /// `compile.py:786-788` `start_compiling`.
    fn start_compiling(&self) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.start_compiling();
        }
    }

    /// `compile.py:790-795` `done_compiling`.
    fn done_compiling(&self) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.done_compiling();
        }
    }

    /// `compile.py:826-830` `store_hash`.
    fn store_hash(&self, hash: u64) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.store_hash(hash);
        }
    }

    /// `compile.py:813-824` `make_a_counter_per_value`.
    fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.make_a_counter_per_value(index, type_tag);
        }
    }

    /// `assembler.py:966` — read `adr_jump_offset`.  Forwarded to the
    /// metainterp `AbstractFailDescr` (`history.py:132 _attrs_`) via
    /// `meta_descr`.  Returns `0` for synthetic backend descrs without
    /// a metainterp counterpart.
    fn adr_jump_offset(&self) -> usize {
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .map_or(0, |fd| fd.adr_jump_offset())
    }

    /// `assembler.py:987` — set `adr_jump_offset` (`0` means "patched").
    fn set_adr_jump_offset(&self, offset: usize) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.set_adr_jump_offset(offset);
        }
    }

    /// `llsupport/llmodel.py:424` `descr.rd_locs[index]`.
    fn rd_locs(&self) -> &[u16] {
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .map_or(&[][..], |fd| fd.rd_locs())
    }

    /// `llsupport/assembler.py:279` `guardtok.faildescr.rd_locs = positions`.
    fn set_rd_locs(&self, locs: Vec<u16>) {
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.set_rd_locs(locs);
        }
    }

    fn is_compiling(&self) -> bool {
        // `compile.py:750` — read through the same forwarding chain as
        // `get_status`, so the busy-flag observation tracks the canonical
        // metainterp slot when meta_descr is set.
        self.get_status() & majit_backend::STATUS_BUSY_FLAG != 0
    }

    // rd_numb / rd_consts / rd_virtuals / rd_pendingfields: trait defaults
    // (`majit-ir/src/descr.rs:1144-1212`) return `None`.  After the GUARD
    // wrapper deletion (`dd9848f35b`), `DynasmFailDescr` only wraps FINISH
    // descrs (`_DoneWithThisFrameDescr` / `ExitFrameWithExceptionDescrRef`
    // singletons in `meta_descr`).  Those singletons are NOT `ResumeDescr`
    // per `compile.py:185`, so `meta_resume_fd()` always returns `None` for
    // them, making the prior forwarding overrides equivalent to the trait
    // defaults — dropped.
}
