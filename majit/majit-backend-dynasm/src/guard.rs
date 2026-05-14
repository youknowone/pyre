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

/// Backend-static side-table mapping a `DynasmFailDescr` Arc's
/// `Arc::as_ptr` address to its `ExitRecoveryLayout` (Session 7
/// pre-flight, mirrors cranelift `RECOVERY_LAYOUT_TABLE`).
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) does not
/// carry a `recovery_layout` slot.  Upstream resume code
/// (`resume.py:450-488`) decodes recovery on demand from the four
/// payload attributes (rd_numb / rd_consts / rd_virtuals /
/// rd_pendingfields).  Pyre keeps the structured layout because the
/// recovery path is shared across both backends via
/// `ExitRecoveryLayout` rather than re-decoding the resume tagged
/// numbering inline.
static RECOVERY_LAYOUT_TABLE: OnceLock<Mutex<HashMap<usize, ExitRecoveryLayout>>> = OnceLock::new();

fn recovery_layout_table() -> &'static Mutex<HashMap<usize, ExitRecoveryLayout>> {
    RECOVERY_LAYOUT_TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_recovery_layout(descr_ptr: usize, layout: ExitRecoveryLayout) {
    recovery_layout_table()
        .lock()
        .expect("RECOVERY_LAYOUT_TABLE mutex poisoned")
        .insert(descr_ptr, layout);
}

pub fn lookup_recovery_layout(descr_ptr: usize) -> Option<ExitRecoveryLayout> {
    recovery_layout_table()
        .lock()
        .expect("RECOVERY_LAYOUT_TABLE mutex poisoned")
        .get(&descr_ptr)
        .cloned()
}

/// Backend-static side-table that maps a `DynasmFailDescr` Arc's
/// `Arc::as_ptr` address to the regalloc-derived `fail_arg_locs`
/// (each fail-arg's absolute jitframe slot, or `None` for unmapped
/// virtual / dead opref).
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) does not
/// carry a `fail_arg_locs` slot; PyPy `assembler.py:286-298`
/// (`write_failure_recovery_description`) encodes the per-slot
/// positions directly into the recovery stub's machine code via
/// immediate operands, and `failure_recovery_func` reads them
/// back from the instruction stream at fail-time.  Pyre's dynasm
/// backend retains the equivalent regalloc output as a structured
/// `Vec<Option<usize>>` keyed off the descr identity so the runtime
/// helper (`handle_fail_resume_guard` in `lib.rs`) and the dead-frame
/// build path (`runner.rs::execute_token`) can read jitframe slots
/// without re-decoding machine code — equivalent semantics, different
/// encoding.
///
/// Static because the trampoline (`handle_fail_resume_guard`) is
/// reached from a JIT call site that has no backend reference, only
/// the descr address.  The `DynasmBackend::fail_descr_registry`
/// (also static-discipline via `Arc<Mutex<…>>`) has the same shape
/// for the same reason.
static FAIL_ARG_LOCS_TABLE: OnceLock<Mutex<HashMap<usize, Vec<Option<usize>>>>> = OnceLock::new();

fn fail_arg_locs_table() -> &'static Mutex<HashMap<usize, Vec<Option<usize>>>> {
    FAIL_ARG_LOCS_TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Codegen-time write: invoked from `append_guard_token_with_faillocs`
/// (x86 + aarch64) right after regalloc finalises the per-fail-arg
/// jitframe slot map.  Also invoked from `allocate_unmapped_fail_arg_slots`
/// when guard ops carry unmapped (virtual / dead) oprefs and need a
/// second-pass write.
pub fn register_fail_arg_locs(descr_ptr: usize, locs: Vec<Option<usize>>) {
    fail_arg_locs_table()
        .lock()
        .expect("FAIL_ARG_LOCS_TABLE mutex poisoned")
        .insert(descr_ptr, locs);
}

/// Runtime / dead-frame read.  Returns `None` when the descr was
/// constructed outside the assembler path (e.g. backend unit-tests
/// that build a `DynasmFailDescr` directly without going through
/// `Assembler386::append_guard_token_with_faillocs`).  Callers that
/// observe `None` fall back to identity slot indexing (`slot == i`),
/// matching the pre-table runner behaviour for synthetic test descrs.
pub fn lookup_fail_arg_locs(descr_ptr: usize) -> Option<Vec<Option<usize>>> {
    fail_arg_locs_table()
        .lock()
        .expect("FAIL_ARG_LOCS_TABLE mutex poisoned")
        .get(&descr_ptr)
        .cloned()
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

    // fail_arg_locs removed (Session 5h): not in PyPy
    // `AbstractFailDescr._attrs_` (`history.py:132`).  PyPy encodes the
    // per-fail-arg slot positions as immediate operands in the recovery
    // stub's machine code (`assembler.py:286-298
    // write_failure_recovery_description`); pyre's dynasm retains the
    // structured `Vec<Option<usize>>` in a backend-static side-table
    // (`fail_arg_locs_table()` in this module) keyed on
    // `Arc::as_ptr(&descr)`.  Semantically equivalent, encoded
    // differently.  Will be folded into machine-code embedding in
    // future work; for now the side-table lookup is the bridge.
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
    /// Backend-static side-tables (`FAIL_ARG_LOCS_TABLE`,
    /// `SOURCE_OP_INDEX_TABLE`) are keyed on the descr's inner address.
    /// Without cleanup the entry would outlive the descr and a future
    /// descr at the same reused address would observe stale state.
    /// Same lifecycle discipline as `CraneliftFailDescr` (see its
    /// `Drop` impl).
    fn drop(&mut self) {
        let ptr = self as *const Self as usize;
        fail_arg_locs_table()
            .lock()
            .expect("FAIL_ARG_LOCS_TABLE mutex poisoned")
            .remove(&ptr);
        source_op_index_table()
            .lock()
            .expect("SOURCE_OP_INDEX_TABLE mutex poisoned")
            .remove(&ptr);
        recovery_layout_table()
            .lock()
            .expect("RECOVERY_LAYOUT_TABLE mutex poisoned")
            .remove(&ptr);
    }
}

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
    ) -> Self {
        DynasmFailDescr {
            fail_index,
            trace_id,
            fail_arg_types,
            meta_descr: None,
        }
    }

    /// Read the recovery_layout from the backend-static side-table.
    pub fn recovery_layout(&self) -> Option<ExitRecoveryLayout> {
        lookup_recovery_layout(self as *const Self as usize)
    }

    /// Set the recovery_layout in the backend-static side-table.
    pub fn set_recovery_layout(&self, layout: ExitRecoveryLayout) {
        register_recovery_layout(self as *const Self as usize, layout);
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
        gc_ref_slots: fail_arg_types
            .iter()
            .enumerate()
            .filter_map(|(i, tp)| (*tp == Type::Ref).then_some(i))
            .collect(),
        force_token_slots: Vec::new(),
        frame_stack: None,
        recovery_layout: lookup_recovery_layout(descr_addr),
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
