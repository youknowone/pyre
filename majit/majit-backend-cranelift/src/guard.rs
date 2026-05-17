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
use majit_ir::{AccumInfo, Const, DescrRef, FailDescr, GcRef, Type};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::Arc;

// BRIDGE_CACHES_TABLE removed (Slice JJ → Slice 7-Tβ11): the
// `Box<AtomicUsize>` cells for bridge code_ptr / frame_depth now live
// on `ResumeGuardDescr` (meta side).  Box gives each cell a heap-
// pinned address that survives `Arc::clone` of the meta Arc; the JIT
// bakes those addresses into the machine code
// (`compiler.rs::emit_attached_bridge_dispatch`) so they must remain
// stable for the descr's lifetime.

// FORCE_TOKEN_SLOTS_TABLE removed (Slice II): write-once at codegen.
// Lives in `ResumeGuardDescr::force_token_slots` on the meta-side Arc
// (Slice 7-Tβ7), reached via `as_any` downcast on `meta_descr`.

/// Backend-static side-table mapping a `CraneliftFailDescr` Arc's
/// `Arc::as_ptr` address to its codegen-time `source_op_index`.
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) carries no
/// `source_op_index` slot — RPython's `assembler.py` does not need
/// to remember the trace-op index post codegen because the metainterp
/// `pyjitpl` driver carries the same identity via the live op object
/// passed to `_compile_one_block`.  Pyre's `FailDescrLayout` keeps
/// the index for the backend→metainterp interop boundary; storing it
/// off the descr keeps the descr struct aligned with PyPy.
// SOURCE_OP_INDEX_TABLE removed (Slice HH): write-once at codegen.
// Lives in `ResumeGuardDescr::source_op_index` on the meta-side Arc
// (Slice 7-Tβ6), reached via `as_any` downcast on `meta_descr`.

/// Backend-static side-table mapping a `CraneliftFailDescr` Arc's
/// `Arc::as_ptr` address to its `ExitRecoveryLayout`.
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) carries no
/// `recovery_layout` slot.  Upstream resume code (`resume.py:450-488`)
/// decodes recovery on demand from `rd_numb` / `rd_consts` /
/// `rd_virtuals` / `rd_pendingfields` — the four `_attrs_` payload
/// fields.  Pyre's cranelift retains the structured layout in a
/// side-table because Cranelift IR cannot decode the resume tagged-
/// numbering inline; it is materialised at codegen time and consumed
/// from the dispatch path.
// RECOVERY_LAYOUT_TABLE removed (Slice EE): not in PyPy
// `AbstractFailDescr._attrs_` (`history.py:132`).  Upstream resume code
// decodes recovery on demand from the four payload attributes
// (rd_numb / rd_consts / rd_virtuals / rd_pendingfields) in
// `resume.py:450-488`.  Cranelift retains the structured layout per-descr
// in an `AtomicPtr<ExitRecoveryLayout>` cell (same pattern as
// `bridge_dispatch_cell`, Slice CC): write-mostly-once via
// `Arc::into_raw(Arc::new(layout))`, read via
// `cell.load(Acquire) + Arc::increment_strong_count + Arc::from_raw`,
// reclaimed in `Drop`.

/// Backend-static side-table mapping a `CraneliftFailDescr` Arc's
/// `Arc::as_ptr` address to its compile-time `CompiledTraceInfo`.
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) carries no
/// `trace_info` slot — RPython recovers the same information from
/// `cpu.asmmemmgr_blocks` + `compiled_loop_token`.  Cranelift's
/// per-trace metadata (input types / header_pc / source_guard tuple)
/// is the equivalent state, parked here so the descr struct stays
/// aligned with PyPy's surface.
// TRACE_INFO_TABLE removed (Slice FF): same descr-local atomic cell
// pattern as `recovery_layout_cell` (Slice EE).  Per-trace
// `CompiledTraceInfo` lives in the `trace_info_cell` field on
// CraneliftFailDescr; PyPy recovers equivalent state from
// `cpu.asmmemmgr_blocks` + `compiled_loop_token`.

// EXTERNAL_JUMP_TARGETS removed (Slice GG → Slice 7-Tβ8).  Lives in
// `ResumeGuardDescr::external_jump_target: OnceLock<DescrRef>` on
// the meta-side Arc, reached via `as_any` downcast on `meta_descr`.
// PyPy emits a raw inter-function JMP at
// `assembler.py:2456-2462 closing_jump`; cranelift's dispatcher
// returns to the runtime and consults this cell.

// FAIL_COUNT_TABLE removed (Slice DD): the per-descr failure counter
// is the bridge-compilation threshold input
// (`AbstractResumeGuardDescr.handle_fail` in `compile.py:701-717`
// drives `must_compile` via `jitcounter.tick(status_hash)` in RPython).
// Pyre's cranelift keeps a raw per-descr `AtomicU32` counter; moving
// it from the backend-static `HashMap` mutex into a descr-local
// atomic field follows the `patch_jump_for_descr` pattern (Slice CC):
// the dispatch hot path (`compiler.rs:3065 fail_descr.increment_fail_count()`)
// now executes a single `fetch_add(Relaxed)` with no lock, no
// HashMap lookup, and no allocator.

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
    /// Frozen after compile — `Box<[T]>` reflects RPython's no-mutation
    /// contract (compile.py:183-203 record_loop_or_bridge). Position
    /// equals `descr.fail_index` by an invariant asserted at construction.
    pub fail_descrs: Box<[Arc<CraneliftFailDescr>]>,
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
    // source_op_index removed (Session 5i-cl): not in PyPy
    // `AbstractFailDescr._attrs_` (`history.py:132`).  The codegen-
    // time trace-op index lives in `SOURCE_OP_INDEX_TABLE` keyed on
    // `Arc::as_ptr(&descr)`.
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    // gc_map removed (Session 5i-cl): not in PyPy
    // `AbstractFailDescr._attrs_` (`history.py:132`).  Upstream
    // `assembler.py` parks the GC-map in `compiled_loop_token.gcmap`.
    // Cranelift retains the per-descr GcMap in `GC_MAP_TABLE` keyed
    // on `Arc::as_ptr(&descr)`.
    // is_finish removed: `compile.py:624 final_descr=True` is a class
    // attribute on `_DoneWithThisFrameDescr`/`ExitFrameWithExceptionDescrRef`.
    // After cranelift singletons carry meta_descr to the class-distinct
    // majit-backend types and codegen descrs carry meta_descr =
    // op.descr, every CraneliftFailDescr forwards is_finish through the
    // upstream class hierarchy.
    // is_exit_frame_with_exception removed: `compile.py:658-662
    // ExitFrameWithExceptionDescrRef` is a class identity on the
    // metainterp side.  After cranelift singletons +
    // EXIT_FRAME_WITH_EXCEPTION_DESCR_REF_CL carry meta_descr to the
    // class-distinct majit-backend ExitFrameWithExceptionDescrRef and
    // codegen descrs carry meta_descr=op.descr (or the propagate-into-
    // exit synthesis route through the singleton), every
    // CraneliftFailDescr forwards the predicate through meta_descr.
    /// history.py:470-499 TargetToken parity for cross-loop JUMP.
    /// True for external JUMP exits (JUMP whose target TargetToken lives in
    /// a different compiled function). assembler.py:2456-2462 closing_jump
    /// emits a raw JMP to `target_token._ll_loop_code`. Cranelift can't
    /// emit raw inter-function JMPs, so the exit returns to the dispatcher
    /// which reads `target_descr` and re-enters the target loop via the
    /// registered `JitCellToken.number -> RegisteredLoopTarget` metadata.
    /// Mutually exclusive with is_finish.
    // is_external_jump / target_descr removed (Session 5i-cl): neither
    // is in PyPy `AbstractFailDescr._attrs_` (`history.py:132`).  PyPy
    // emits a raw inter-function JMP at `assembler.py:2456-2462
    // closing_jump`; the cranelift backend's dispatcher-mediated
    // equivalent now consults the `EXTERNAL_JUMP_TARGETS` side-table
    // (keyed on `Arc::as_ptr(&descr)`).  Membership = external-JUMP
    // predicate; lookup value = target `DescrRef`.
    // force_token_slots removed (Session 5i-cl): not in PyPy
    // `AbstractFailDescr._attrs_` (`history.py:132`).  Upstream
    // `assembler.py` encodes the slot positions inline into the
    // machine-code GC-map immediates; cranelift parks the per-descr
    // vector in `FORCE_TOKEN_SLOTS_TABLE` (this module) since
    // Cranelift IR has no equivalent inline encoding.
    // trace_info was moved to `trace_info_cell` below (Slice FF):
    // not in PyPy `AbstractFailDescr._attrs_` (`history.py:132`).
    // RPython recovers the same information from
    // `cpu.asmmemmgr_blocks` + `compiled_loop_token`.
    // recovery_layout was moved to `recovery_layout_cell` below
    // (Slice EE): not in PyPy `AbstractFailDescr._attrs_`
    // (`history.py:132`).  Upstream resume code decodes recovery on
    // demand from the four payload attributes
    // (rd_numb / rd_consts / rd_virtuals / rd_pendingfields) in
    // `resume.py:450-488`.
    // status removed: `compile.py:683 AbstractResumeGuardDescr._attrs_
    // = ('status',)` — only ResumeGuardDescr family carries this slot.
    // Done*/Exit/Propagate inherit AbstractFailDescr without status.
    // After Phase A every backend descr forwards through meta_descr to
    // the metainterp class, so the local AtomicU64 mirror is unused.
    // fail_count removed (Session 5i-cl): not in PyPy
    // `AbstractFailDescr._attrs_` (`history.py:132`).  The per-descr
    // bridge-compilation threshold counter moved to
    // `FAIL_COUNT_TABLE` in this module, keyed on `Arc::as_ptr(&descr)`.
    // `history.py:132` `AbstractFailDescr._attrs_` `rd_vector_info` —
    // the canonical store lives on the metainterp `AbstractFailDescr`
    // (`majit-metainterp/src/compile.rs`), reached via `meta_descr`.
    // The previous backend-local `vector_info: Vec<AccumInfo>` slot was
    // dead — initialized empty at construction, never written.
    // bridge removed (Session 5i-cl): not in PyPy
    // `AbstractFailDescr._attrs_` (`history.py:132`).  Upstream
    // `compile.py:attach_bridge` patches the failing guard's
    // machine-code JMP to the bridge entry directly.  Cranelift parks
    // the per-descr `BridgeData` in `BRIDGE_TABLE` (this module).
    // bridge_code_ptr_cache / bridge_frame_depth_cache removed
    // (Session 5i-cl): not in PyPy `AbstractFailDescr._attrs_`
    // (`history.py:132`).  The two `AtomicUsize` cells now live as
    // heap-pinned `Box<AtomicUsize>` entries in `BRIDGE_CACHES_TABLE`
    // (this module) so the JIT-baked addresses (see
    // `emit_attached_bridge_dispatch` in `compiler.rs`) stay valid
    // after the descr is wrapped in `Arc::new`.
    // rd_loop_token_clt removed: `history.py:132 AbstractFailDescr._attrs_`
    // `rd_loop_token` lives on the metainterp Arc.  Only ResumeDescr
    // family descrs receive `record_loop_or_bridge`'s
    // `descr.rd_loop_token = clt` stamp (compile.py:183-186); pyre's
    // walker (compiler.rs:13421-13428) gates on
    // `descr.is_resume_guard()` so the stamp always lands on the
    // metainterp ResumeGuardDescr through meta_descr forwarding.
    /// Back-pointer to the metainterp `ResumeGuardDescr` Arc the
    /// optimizer stamped onto the originating guard op (`op.descr`).
    /// PyPy keeps a single descr object per guard (`history.py:121`);
    /// pyre's transitional split-descr stores this Arc as a back-pointer
    /// so backend accessors forward `rd_numb`/`rd_consts`/`rd_virtuals`/
    /// `rd_pendingfields`/`fail_arg_types`/`status`/`rd_loop_token`/
    /// `rd_vector_info` to the metainterp `AbstractFailDescr`
    /// (`history.py:132 _attrs_`).  The final Unified-Descr endpoint
    /// collapses `CraneliftFailDescr` into the metainterp descr.
    ///
    /// `None` for synthetic backend descrs minted by the runtime
    /// classifier (`compiler.rs::find_descr_by_ptr` for FINISH /
    /// PropagateExceptionDescr / ExitFrameWithExceptionDescr exits) —
    /// those exits route through dedicated metainterp Done* descrs
    /// owned by `MetaInterpStaticData`, not via `op.descr`.
    pub meta_descr: Option<DescrRef>,
    // bridge_dispatch_cell moved to ResumeGuardDescr (Slice 7-Tβ12):
    // type-erased as `AtomicPtr<()>` on the meta side with a backend-
    // registered cleanup function.  Cranelift's `bridge_ref` /
    // `attach_bridge` forward through the meta Arc; the cleanup
    // (`drop_bridge_payload` below) reconstructs and drops the
    // owning `Arc<BridgeData>` so the cross-crate type is reclaimed
    // by the owning crate.
    // fail_count moved to ResumeGuardDescr (Slice 7-Tβ9): the meta
    // Arc is the canonical home, reached via `as_any` downcast on
    // `meta_descr` (precedent: Slice 7-Tβ6 source_op_index, Slice 7-Tβ7
    // force_token_slots).  Cranelift accessors `increment_fail_count`
    // / `get_fail_count` now forward through that chain.  Singletons
    // and external-JUMP descrs (no ResumeGuardDescr meta) silently
    // no-op on increment, which is correct — singletons never carry
    // bridges, and external JUMPs do not drive bridge thresholds.
    // trace_info_cell moved to ResumeGuardDescr (Slice 7-Tβ10): the
    // meta Arc is the canonical home, reached via `as_any` downcast
    // on `meta_descr`.  Cranelift accessors `set_trace_info` /
    // `trace_info_ref` forward through that chain.
    // external_jump_target_cell moved to ResumeGuardDescr (Slice
    // 7-Tβ8): the meta Arc is the canonical home, reached via
    // `as_any` downcast on `meta_descr`.  Cranelift accessors
    // `set_external_jump_target` / `external_jump_target_ref` and
    // the `is_external_jump()` predicate now forward through that
    // chain.
    // source_op_index_cell moved to ResumeGuardDescr (Slice 7-Tβ6):
    // the meta Arc is the canonical home, reached via `as_any`
    // downcast on `meta_descr` (precedent: Slice OO-half for
    // recovery_layout).  Cranelift accessors `source_op_index_ref` /
    // `set_source_op_index` now forward through that chain.
    // force_token_slots_cell moved to ResumeGuardDescr (Slice 7-Tβ7):
    // the meta Arc is the canonical home, reached via `as_any`
    // downcast on `meta_descr` (precedent: Slice 7-Tβ6 for
    // source_op_index, Slice OO-half for recovery_layout).
    // Cranelift accessors `force_token_slots_view` /
    // `set_force_token_slots` now forward through that chain.
    // bridge_code_ptr_cache / bridge_frame_depth_cache moved to
    // ResumeGuardDescr (Slice 7-Tβ11): the meta Arc is the canonical
    // home, reached via `as_any` downcast on `meta_descr`.  Cranelift
    // accessors `bridge_cache_addrs` / `bridge_code_ptr` / `has_bridge`
    // / `attach_bridge` now forward through that chain.  All guards
    // that reach `emit_attached_bridge_dispatch` carry a
    // `ResumeGuardDescr` meta (real `op.descr` or the test-scaffold
    // synthesis at compiler.rs:12884), so the downcast always
    // succeeds.
}

impl Drop for CraneliftFailDescr {
    /// Backend-static side-tables (`EXTERNAL_JUMP_TARGETS`,
    /// `FAIL_COUNT_TABLE`, `FORCE_TOKEN_SLOTS_TABLE`,
    /// `BRIDGE_CACHES_TABLE`) are keyed on the descr's inner address.
    /// Without cleanup the entry would outlive the descr and the
    /// allocator may reuse the freed address for a future descr that
    /// would then observe stale state.
    ///
    /// `bridge_dispatch_cell` lives directly on the descr; reclaim
    /// the published `Arc<BridgeData>` by swapping the cell to null
    /// and reconstructing the Arc.  `BridgeData::fail_descrs` may
    /// hold `Arc<CraneliftFailDescr>` clones whose own `Drop` re-runs
    /// this path on the same thread; the swap-to-null sequence is
    /// reentrant (each descr touches only its own cell).
    fn drop(&mut self) {
        // external_jump_target moved to ResumeGuardDescr meta-side
        // slot (Slice 7-Tβ8); no backend-local cell to reclaim.
        // fail_count is descr-local (Slice DD): drops naturally with self.
        // trace_info_cell moved to ResumeGuardDescr meta-side slot
        // (Slice 7-Tβ10); reclaim is owned by ResumeGuardDescr::drop.
        // recovery_layout moved to ResumeGuardDescr meta-side slot
        // (Slice QQ-4); no backend-local cell to reclaim.
        // source_op_index moved to ResumeGuardDescr meta-side slot
        // (Slice 7-Tβ6); no backend-local cell to reclaim.
        // force_token_slots moved to ResumeGuardDescr meta-side slot
        // (Slice 7-Tβ7); no backend-local cell to reclaim.
        // bridge_dispatch_cell moved to ResumeGuardDescr meta-side
        // slot (Slice 7-Tβ12); reclaim is owned by
        // ResumeGuardDescr::drop via the cleanup function registered
        // in `attach_bridge`.
    }
}

/// Backend-registered cleanup for the type-erased
/// `ResumeGuardDescr::bridge_dispatch_cell` (Slice 7-Tβ12).  Invoked
/// by `ResumeGuardDescr::drop` on any payload still in the cell at
/// descr teardown; reconstructs the owning `Arc<BridgeData>` so its
/// `Drop` runs.
fn drop_bridge_payload(ptr: *mut ()) {
    if !ptr.is_null() {
        // Safety: produced by `Arc::into_raw(Arc::new(bridge))` in
        // `attach_bridge`; reclaim ownership and drop.
        unsafe { drop(Arc::from_raw(ptr as *const BridgeData)) };
    }
}

impl std::fmt::Debug for CraneliftFailDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CraneliftFailDescr")
            .field(
                "fail_index",
                &<Self as FailDescr>::fail_index_per_trace(self),
            )
            .field("source_op_index", &self.source_op_index_ref())
            .field("trace_id", &self.trace_id)
            .field("fail_arg_types", &self.fail_arg_types)
            .field("gc_map", &self.gc_map())
            .field("is_finish", &<Self as FailDescr>::is_finish(self))
            .field(
                "external_jump_target",
                &self.external_jump_target_ref().map(|d| d.repr()),
            )
            .field("force_token_slots", &self.force_token_slots_view())
            .field("trace_info", &self.trace_info_ref())
            .field("recovery_layout", &self.recovery_layout_ref())
            .field("fail_count", &self.get_fail_count())
            .field("has_bridge", &self.has_bridge())
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

    /// Caller responsibility after `Arc::new(descr)`:
    ///   - if `recovery_layout` was previously passed: invoke
    ///     `descr.set_recovery_layout(layout)` to publish the layout
    ///     into the descr-local atomic cell (Slice EE).
    ///   - if `force_token_slots` is non-empty: invoke
    ///     `descr.set_force_token_slots(...)` AFTER `meta_descr` is
    ///     assigned (Slice 7-Tβ7) so the write lands on the
    ///     `ResumeGuardDescr` meta-side slot.
    ///
    /// The `_is_finish` parameter is preserved for caller-site clarity
    /// during the transition; it is no longer stored on the descr —
    /// `compile.py:624 final_descr=True` is answered through meta_descr
    /// forwarding.
    pub fn new_with_trace_and_kind(
        fail_index: u32,
        trace_id: u64,
        fail_arg_types: Vec<Type>,
        _is_finish: bool,
    ) -> Self {
        CraneliftFailDescr {
            fail_index,
            trace_id,
            fail_arg_types,
            meta_descr: None,
        }
    }

    /// Construct a fail descriptor for an external JUMP exit.
    /// assembler.py:2456-2462 closing_jump parity: JUMP whose target
    /// TargetToken lives in a different compiled function. Cranelift can't
    /// emit raw inter-function JMPs, so the dispatcher receives this descr
    /// and re-enters the target loop via the registered target token.
    pub fn new_external_jump(fail_index: u32, trace_id: u64, fail_arg_types: Vec<Type>) -> Self {
        // Caller is expected to wrap the returned descr in `Arc::new(...)`
        // and immediately publish the external-JUMP target via
        // `descr.set_external_jump_target(target)`.  The constructor
        // cannot do this itself because the callsite needs to perform
        // additional in-place mutations (`set_source_op_index`,
        // `meta_descr`) before sealing the descr behind `Arc`.
        CraneliftFailDescr {
            fail_index,
            trace_id,
            fail_arg_types,
            meta_descr: None,
        }
    }

    // UnsafeCell accessor helpers — single-threaded, no lock needed.
    // RPython ResumeGuardDescr fields are plain attributes (GIL-protected).

    /// `assembler.py:987 patch_jump_for_descr` parity — read the
    /// meta-side type-erased dispatch cell (Slice 7-Tβ12).  PyPy's
    /// dispatch is a JMP rel32 whose target is patched in-place by
    /// `attach_bridge`; pyre reads the `Arc<BridgeData>` raw pointer
    /// the JIT thread wrote there with `Arc::into_raw(Arc::new(...))`,
    /// then bumps the strong count and reconstructs the `Arc`.  Lock-
    /// free and HashMap-free (mirrors `adr_jump_offset` semantics).
    /// Returns `None` when no `ResumeGuardDescr` meta is present
    /// (singletons / non-Resume meta) — those guards never carry
    /// bridges.
    #[inline]
    pub fn bridge_ref(&self) -> Option<Arc<BridgeData>> {
        let rgd = self.meta_resume_guard_descr()?;
        let ptr = rgd.bridge_dispatch_load();
        if ptr.is_null() {
            None
        } else {
            // Safety: `ptr` was produced by `Arc::into_raw(Arc::new(bridge))`
            // in `attach_bridge`; the cell only stores valid Arc raw
            // pointers (or null).  `increment_strong_count` followed by
            // `from_raw` produces an additional owning `Arc` without
            // taking the original.  Drop ordering: the descr's `Drop`
            // swaps the cell to null and reclaims the stored Arc only
            // after no further `bridge_ref` reader can observe the old
            // ptr (same release/acquire pairing as PyPy's GIL-protected
            // descr access).
            unsafe {
                Arc::increment_strong_count(ptr as *const BridgeData);
                Some(Arc::from_raw(ptr as *const BridgeData))
            }
        }
    }

    #[inline]
    /// Read the per-trace `CompiledTraceInfo` from the meta-side
    /// `ResumeGuardDescr::trace_info` slot (Slice 7-Tβ10).  Returns
    /// `None` when meta_descr is absent or is not a `ResumeGuardDescr`
    /// (synthetic FINISH / singleton descrs), or when no trace info
    /// has been published.
    pub fn trace_info_ref(&self) -> Option<CompiledTraceInfo> {
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_any())
            .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
            .and_then(|rgd| rgd.trace_info())
    }

    #[inline]
    /// Read the recovery_layout from the meta-side `ResumeGuardDescr`
    /// slot — single source of truth (Slice QQ-4: backend-local cell
    /// removed).  Synthetic descrs without a `ResumeGuardDescr`
    /// `meta_descr` (codegen-time FINISH `Done*` / external-JUMP
    /// `None`) return `None`; the recovery_layout walker handles
    /// `None` as the no-recovery path (no virtuals to materialise).
    pub fn recovery_layout_ref(&self) -> Option<ExitRecoveryLayout> {
        // `compile.py:849` `ResumeGuardCopiedDescr.get_resumestorage():
        // return prev`.  Chase `prev_descr` until we land on the donor
        // `ResumeGuardDescr` — otherwise copied descrs would always
        // return `None` since their `as_any` is the trait default.
        let mut current = self.meta_descr.as_ref().cloned()?;
        loop {
            if let Some(rgd) = current
                .as_any()
                .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
            {
                return rgd.recovery_layout();
            }
            match current.prev_descr() {
                Some(next) => current = next,
                None => return None,
            }
        }
    }

    /// Forward the failure counter increment to the meta-side
    /// `ResumeGuardDescr::fail_count` slot (Slice 7-Tβ9).  Returns
    /// the post-increment value; returns 0 when meta_descr is
    /// absent or is not a `ResumeGuardDescr` (synthetic FINISH /
    /// singleton descrs that never carry bridges).
    pub fn increment_fail_count(&self) -> u32 {
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_any())
            .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
            .map_or(0, |rgd| rgd.increment_fail_count())
    }

    /// Read the failure counter from the meta-side
    /// `ResumeGuardDescr::fail_count` slot (Slice 7-Tβ9).
    pub fn get_fail_count(&self) -> u32 {
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_any())
            .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
            .map_or(0, |rgd| rgd.get_fail_count())
    }

    /// Forward to the meta-side `ResumeGuardDescr::bridge_code_ptr`
    /// slot (Slice 7-Tβ11) — whether a bridge has been attached.
    /// Returns `false` when no `ResumeGuardDescr` meta is present
    /// (singletons, cross-loop JUMP test scaffolds outside the
    /// production codepath).
    pub fn has_bridge(&self) -> bool {
        self.meta_resume_guard_descr()
            .is_some_and(|rgd| rgd.bridge_code_ptr() != 0)
    }

    /// Forward to the meta-side `ResumeGuardDescr::bridge_code_ptr`
    /// slot (Slice 7-Tβ11).  Returns null when no `ResumeGuardDescr`
    /// meta is present.
    pub fn bridge_code_ptr(&self) -> *const u8 {
        self.meta_resume_guard_descr()
            .map_or(std::ptr::null(), |rgd| rgd.bridge_code_ptr() as *const u8)
    }

    /// Forward to the meta-side cell addresses (Slice 7-Tβ11),
    /// suitable for baking into JIT machine code as immediates.
    /// Returns `(code_ptr_addr, frame_depth_addr)`.  Panics when no
    /// `ResumeGuardDescr` meta is present — all guards that reach
    /// `emit_attached_bridge_dispatch` carry one (real `op.descr` or
    /// the test-scaffold synthesis at compiler.rs:12884).
    pub fn bridge_cache_addrs(&self) -> (usize, usize) {
        self.meta_resume_guard_descr()
            .expect(
                "bridge_cache_addrs requires a ResumeGuardDescr meta_descr; \
                 all bridgeable guards carry one (op.descr or synthesised at \
                 compiler.rs:12884)",
            )
            .bridge_cache_addrs()
    }

    /// `compile.py:attach_bridge` / `assembler.py:987 patch_jump_for_descr`
    /// parity — atomic-store the bridge `Arc` raw pointer into the
    /// meta-side dispatch cell (Slice 7-Tβ12), and publish
    /// `(code_ptr, frame_depth)` into the meta-side cache cells
    /// (Slice 7-Tβ11) that `emit_attached_bridge_dispatch` baked
    /// addresses for.  Registers the backend-side cleanup function
    /// (idempotent) so `ResumeGuardDescr::drop` can reclaim the
    /// published `Arc<BridgeData>` without knowing its concrete type.
    pub fn attach_bridge(&self, bridge: BridgeData) {
        let code_ptr = bridge.code_ptr as usize;
        let frame_depth = bridge
            .max_output_slots
            .max(bridge.num_inputs)
            .max(1)
            .saturating_add(bridge.num_ref_roots);
        let rgd = self.meta_resume_guard_descr().expect(
            "attach_bridge requires a ResumeGuardDescr meta_descr; \
             all bridgeable guards carry one (op.descr or synthesised at \
             compiler.rs:12884)",
        );
        // `Arc::into_raw(Arc::new(bridge))` publishes the bridge data
        // as a raw pointer the dispatch path can re-Arc via
        // `increment_strong_count + Arc::from_raw`.  Swap atomically so
        // a re-attach (unusual) reclaims the previous Arc.
        let new_ptr = Arc::into_raw(Arc::new(bridge)) as *mut () as *mut ();
        let old_ptr = rgd.bridge_dispatch_swap(new_ptr, drop_bridge_payload);
        if !old_ptr.is_null() {
            // Safety: prior `attach_bridge` published this pointer;
            // reclaim ownership and drop.
            unsafe { drop(Arc::from_raw(old_ptr as *const BridgeData)) };
        }
        rgd.store_bridge_caches(code_ptr, frame_depth);
    }

    /// Forward the external-JUMP target publish to the meta-side
    /// `ResumeGuardDescr::set_external_jump_target` slot (Slice 7-Tβ8).
    /// Panics when `meta_descr` is absent or is not a
    /// `ResumeGuardDescr` — the caller must have stamped a synthetic
    /// `ResumeGuardDescr` meta on cross-loop JUMP descrs before
    /// invoking this (see `_compile_one_block` in compiler.rs which
    /// uses `make_resume_guard_descr_typed` for the external-JUMP
    /// path post Slice 7-Tβ7).
    pub fn set_external_jump_target(&self, target: DescrRef) {
        let rgd = self
            .meta_descr
            .as_ref()
            .and_then(|d| d.as_any())
            .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
            .expect(
                "set_external_jump_target requires a ResumeGuardDescr meta_descr; \
                 cross-loop JUMP descrs synthesise one in _compile_one_block",
            );
        rgd.set_external_jump_target(target);
    }

    #[inline]
    /// Read the external-JUMP target from the meta-side
    /// `ResumeGuardDescr::external_jump_target` slot (Slice 7-Tβ8).
    /// Returns `None` for descrs without a `ResumeGuardDescr` meta
    /// AND for regular guard descrs (the common case).
    pub fn external_jump_target_ref(&self) -> Option<DescrRef> {
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_any())
            .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
            .and_then(|rgd| rgd.external_jump_target())
    }

    /// Write recovery_layout to the meta-side `ResumeGuardDescr` slot
    /// (Slice QQ-4).  Silently skips synthetic descrs without a
    /// `ResumeGuardDescr` `meta_descr` (codegen-time FINISH `Done*` /
    /// external-JUMP `None`) — those descrs never reach the
    /// recovery_layout readers in production (guard-failure deopt
    /// only); when they do (test introspection, bridge-attach source
    /// chase), `recovery_layout_ref()` returns `None` and the caller
    /// handles the no-recovery path.
    pub fn set_recovery_layout(&self, recovery_layout: ExitRecoveryLayout) {
        // Match `recovery_layout_ref`: chase `prev_descr` through any
        // `ResumeGuardCopiedDescr` chain to write into the donor's slot.
        let Some(mut current) = self.meta_descr.as_ref().cloned() else {
            return;
        };
        loop {
            if let Some(rgd) = current
                .as_any()
                .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
            {
                rgd.set_recovery_layout(recovery_layout);
                return;
            }
            match current.prev_descr() {
                Some(next) => current = next,
                None => return,
            }
        }
    }

    /// Write the codegen-time trace-op index through to the meta-side
    /// `ResumeGuardDescr` slot (Slice 7-Tβ6).  Silently skips synthetic
    /// descrs without a `ResumeGuardDescr` `meta_descr` — those descrs
    /// (FINISH `Done*` / external-JUMP TargetToken) have no associated
    /// trace op upstream either.
    pub fn set_source_op_index(&self, source_op_index: usize) {
        // Match `recovery_layout_ref` shape: chase `prev_descr` through
        // any `ResumeGuardCopiedDescr` chain to write into the donor.
        let Some(mut current) = self.meta_descr.as_ref().cloned() else {
            return;
        };
        loop {
            if let Some(rgd) = current
                .as_any()
                .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
            {
                rgd.set_source_op_index(source_op_index);
                return;
            }
            match current.prev_descr() {
                Some(next) => current = next,
                None => return,
            }
        }
    }

    #[inline]
    /// Read the codegen-time trace-op index from the meta-side
    /// `ResumeGuardDescr` slot (Slice 7-Tβ6).  Returns `None` for
    /// synthetic descrs without a `ResumeGuardDescr` meta or for
    /// descrs whose codegen never stamped the slot.
    pub fn source_op_index_ref(&self) -> Option<usize> {
        let mut current = self.meta_descr.as_ref().cloned()?;
        loop {
            if let Some(rgd) = current
                .as_any()
                .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
            {
                return rgd.source_op_index();
            }
            match current.prev_descr() {
                Some(next) => current = next,
                None => return None,
            }
        }
    }

    /// Forward the force-token slot list to the meta-side
    /// `ResumeGuardDescr` slot (Slice 7-Tβ7).  `ResumeGuardDescr::
    /// set_force_token_slots` sorts+dedups the vector so the stored
    /// list satisfies `binary_search` (used by `is_force_token_slot`).
    /// When the meta_descr is absent or is not a `ResumeGuardDescr`
    /// (synthetic FINISH / external-JUMP descrs whose meta is a
    /// non-resume class), the write is silently discarded — those
    /// descrs never produce force tokens.
    pub fn set_force_token_slots(&self, slots: Vec<usize>) {
        if let Some(rgd) = self
            .meta_descr
            .as_ref()
            .and_then(|d| d.as_any())
            .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
        {
            rgd.set_force_token_slots(slots);
        }
    }

    #[inline]
    /// Read the force-token slot list from the meta-side
    /// `ResumeGuardDescr` slot (Slice 7-Tβ7).  Returns `&[]` when
    /// meta_descr is absent or is not a `ResumeGuardDescr`.
    pub fn force_token_slots_view(&self) -> &[usize] {
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_any())
            .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
            .map_or(&[], |rgd| rgd.force_token_slots())
    }

    /// Forward the per-trace `CompiledTraceInfo` publish to the meta-
    /// side `ResumeGuardDescr::set_trace_info` slot (Slice 7-Tβ10).
    /// Callers are `compile_loop` (codegen finaliser) and
    /// `overlay_deadframe_fail_descr` (CALL_ASSEMBLER prefix overlay).
    /// Silently dropped when meta_descr is absent or is not a
    /// `ResumeGuardDescr` (synthetic singletons that never carry
    /// per-trace metadata).
    pub fn set_trace_info(self: &Arc<Self>, trace_info: CompiledTraceInfo) {
        if let Some(rgd) = self
            .meta_descr
            .as_ref()
            .and_then(|d| d.as_any())
            .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
        {
            rgd.set_trace_info(trace_info);
        }
    }

    /// Derive the `GcMap` on demand from `fail_arg_types` and the
    /// meta-side `ResumeGuardDescr::force_token_slots` slot (Slice
    /// 7-Tβ7).  Replaces the previous `pub gc_map: GcMap` field
    /// (Session 5i-cl); upstream `assembler.py:write_failure_recovery_description`
    /// parity recomputes equivalent bits inline at codegen time.
    pub fn gc_map(&self) -> GcMap {
        // Use the forwarded `fail_arg_types()` — when meta_descr carries
        // an optimizer-stamped `ResumeGuardDescr`, its types are the
        // canonical view that downstream GC root classification depends
        // on (it may differ from the construction-time backend list).
        Self::gc_map_for_types(
            <Self as FailDescr>::fail_arg_types(self),
            self.force_token_slots_view(),
        )
    }

    pub fn is_force_token_slot(&self, slot: usize) -> bool {
        // Vector stored in `ResumeGuardDescr::force_token_slots` is
        // sorted+deduped at set time, preserving the `binary_search`
        // invariant.
        self.force_token_slots_view().binary_search(&slot).is_ok()
    }

    /// `compile.py:185` `isinstance(descr, ResumeDescr)` gate for
    /// back-pointer forwarding.  Returns the metainterp `FailDescr`
    /// Arc only when the metainterp class hierarchy says it is a
    /// `ResumeDescr` family member (`is_resume_guard()` returns true
    /// for `ResumeGuardDescr`/`ResumeAtPositionDescr`/
    /// `ResumeGuardForcedDescr`/`ResumeGuardExcDescr`/
    /// `CompileLoopVersionDescr`; `is_resume_guard_copied()` returns
    /// true for the `ResumeGuardCopiedDescr` sibling that chases
    /// `prev`).
    ///
    /// `DoneWithThisFrame*` (`compile.py:623`),
    /// `ExitFrameWithExceptionDescrRef` (`compile.py:658-662`),
    /// `PropagateExceptionDescr` (`compile.py:1092`), and
    /// external-JUMP backend descrs are NOT `ResumeDescr` upstream —
    /// they inherit `AbstractFailDescr` directly — so the
    /// `record_loop_or_bridge` walker (`compile.py:183-185`) never
    /// stamps them.  Backend fields the optimizer ports stamp through
    /// `op.descr` (`trace_id`, `fail_arg_types`, `rd_numb`,
    /// `rd_consts`, `rd_virtuals`, `rd_pendingfields`) therefore
    /// cannot read them from a non-`ResumeDescr` `meta_descr`; doing
    /// so returns the trait defaults (e.g. `trace_id() -> 0`) rather
    /// than the construction-time backend-local values.  When this
    /// gate returns `None`, callers fall back to the backend-local
    /// field set at descr construction.
    #[inline]
    fn meta_resume_fd(&self) -> Option<&dyn FailDescr> {
        let d = self.meta_descr.as_ref()?;
        if d.is_resume_guard() || d.is_resume_guard_copied() {
            d.as_fail_descr()
        } else {
            None
        }
    }

    /// `as_any` downcast on `meta_descr` to recover the concrete
    /// `majit_backend::ResumeGuardDescr`.  Used by the cells migrated
    /// in Slice 7-Tβ6..11 whose accessor needs the concrete type (not
    /// the `FailDescr` trait surface) — e.g. `bridge_cache_addrs` /
    /// `store_bridge_caches` are inherent methods on
    /// `ResumeGuardDescr` rather than trait methods because their
    /// signature exposes the heap-pinned Box addresses.
    #[inline]
    fn meta_resume_guard_descr(&self) -> Option<&majit_backend::ResumeGuardDescr> {
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_any())
            .and_then(|a| a.downcast_ref::<majit_backend::ResumeGuardDescr>())
    }

    pub fn layout(&self) -> FailDescrLayout {
        // resume.py:450-488 propagate rd_* for post-eviction reconstruction.
        // Read through the metainterp ResumeGuardDescr Arc gated by
        // isinstance(descr, ResumeDescr) — single source of truth for
        // resume-guard descrs; falls back to backend-local fields
        // otherwise (synthetic FINISH / external-JUMP descrs and
        // Done*/ExitExc/PropagateException meta descrs that are not
        // ResumeDescr upstream).
        let meta_fd = self.meta_resume_fd();
        let fail_arg_types = <Self as FailDescr>::fail_arg_types(self);
        let gc_map_local = self.gc_map();
        let gc_ref_slots = fail_arg_types
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| gc_map_local.is_ref(slot).then_some(slot))
            .collect();
        let recovery = self.recovery_layout_ref();
        let frame_stack = recovery.as_ref().map(|r| r.frames.clone());
        FailDescrLayout {
            fail_index: self.fail_index,
            source_op_index: self.source_op_index_ref(),
            trace_id: <Self as FailDescr>::trace_id(self),
            trace_info: self.trace_info_ref(),
            fail_arg_types: fail_arg_types.to_vec(),
            is_finish: <Self as FailDescr>::is_finish(self),
            is_exception_exit: <Self as FailDescr>::is_exit_frame_with_exception(self),
            gc_ref_slots,
            force_token_slots: self.force_token_slots_view().to_vec(),
            recovery_layout: recovery,
            frame_stack,
            rd_numb: meta_fd.and_then(|fd| fd.rd_numb()).map(|s| s.to_vec()),
            rd_consts: meta_fd.and_then(|fd| fd.rd_consts()).map(|s| s.to_vec()),
            rd_virtuals: meta_fd.and_then(|fd| fd.rd_virtuals()).map(|s| s.to_vec()),
            rd_pendingfields: meta_fd
                .and_then(|fd| fd.rd_pendingfields())
                .map(|s| s.to_vec()),
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

    /// `cranelift_resumedata_deopt` (`pyre/pyre-jit/src/call_jit.rs:3837`)
    /// receives the backend `CraneliftFailDescr` Arc from
    /// `fail_descr_arc_from_addr` and needs to reach the metainterp
    /// `ResumeGuardDescr` for the `rd_*` payload.  Forward through
    /// `meta_descr` so the downstream `downcast_ref::<ResumeGuardDescr>()`
    /// resolves against the metainterp Arc rather than failing on the
    /// backend wrapper's trait default `None`.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        self.meta_descr.as_ref().and_then(|d| d.as_any())
    }

    /// `compile.py:185` `isinstance(descr, ResumeDescr)` parity. Backend
    /// `CraneliftFailDescr` plays the role of upstream's
    /// `ResumeGuardDescr` for guard-failure exits, of the
    /// `DoneWithThisFrame*` / `ExitFrameWithExceptionDescr` family for
    /// finish exits, and of `TargetToken` for external JUMP exits (the
    /// dispatcher-routed cross-loop JUMP path).  Only the first is a
    /// `ResumeDescr` in upstream; finish descrs and `TargetToken`s are
    /// distinct class hierarchies and `compile.py:185` skips them.
    ///
    /// `compile.py:185 isinstance(descr, ResumeDescr)` — forward through
    /// `meta_descr` (covering both `ResumeGuardDescr`-family and
    /// `ResumeGuardCopiedDescr` siblings) so non-`ResumeDescr` meta
    /// descrs (`DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef` /
    /// `PropagateExceptionDescr`) do not flip the backend's role
    /// reading.  `is_external_jump` short-circuits to false because
    /// cranelift's external-JUMP descrs are backend-only synthetic
    /// objects with no metainterp counterpart.
    fn is_resume_guard(&self) -> bool {
        // Slice 7-Tβ8: external-JUMP cell now lives on the meta-side
        // `ResumeGuardDescr`.  Synthetic cross-loop JUMP descrs carry
        // a `ResumeGuardDescr` meta solely to host the
        // `external_jump_target` slot — `is_resume_guard()` must
        // still return false for them so the backend's role reading
        // does not flip.
        if self.external_jump_target_ref().is_some() {
            return false;
        }
        // `compile.py:185` `isinstance(descr, ResumeDescr)` — answered by
        // forwarding to the metainterp class hierarchy via meta_descr.
        // After cranelift singletons + codegen all stamp meta_descr,
        // every production CraneliftFailDescr forwards correctly.
        // Synthetic test descrs without meta_descr take the trait
        // default false.
        self.meta_descr
            .as_ref()
            .map_or(false, |d| d.is_resume_guard() || d.is_resume_guard_copied())
    }
}

impl FailDescr for CraneliftFailDescr {
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
        // Forward through `meta_resume_fd()` (gated on `compile.py:185
        // isinstance(descr, ResumeDescr)`) so the optimizer's
        // `store_final_boxes_in_guard` (compile.py:869) stamp on the
        // metainterp side is the single source of truth for guard
        // descrs.  Fallback to backend-local field when meta_descr is
        // None (synthetic FINISH / ExitFrameWithExceptionDescr /
        // external-JUMP) OR when meta_descr is set to a non-ResumeDescr
        // (Done*/ExitExc/PropagateException — these carry their own
        // construction-time `fail_arg_types` on the metainterp side
        // which happens to coincide, but the canonical-source rule
        // applies only to ResumeDescr per `record_loop_or_bridge`).
        self.meta_resume_fd()
            .map_or(&*self.fail_arg_types, |fd| fd.fail_arg_types())
    }

    fn is_finish(&self) -> bool {
        // `compile.py:624` `_DoneWithThisFrameDescr` family carries
        // `final_descr = True`.  After cranelift LazyLock singletons +
        // production codegen + external JUMP all stamp meta_descr
        // (singletons via majit-backend class-distinct types, codegen
        // via op.descr), the trait method forwards via meta_descr to
        // the upstream class hierarchy.  Synthetic test descrs without
        // meta_descr take the trait default false.
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .map_or(false, |fd| fd.is_finish())
    }

    fn is_exit_frame_with_exception(&self) -> bool {
        // `compile.py:658-662 ExitFrameWithExceptionDescrRef`'s identity
        // lives on the metainterp Arc (`ExitFrameWithExceptionDescrRef`
        // in `majit-backend::finish_descrs`).  After cranelift
        // singletons + codegen all stamp meta_descr, every production
        // CraneliftFailDescr forwards through meta_descr.  Synthetic
        // test descrs without meta_descr take the trait default false.
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .map_or(false, |fd| fd.is_exit_frame_with_exception())
    }

    fn is_external_jump(&self) -> bool {
        // Backend-only flag, no metainterp counterpart — external-JUMP
        // descrs are cranelift-only synthesised exits for cross-loop
        // JUMP targets.  Slice 7-Tβ8 moved the cell to the meta-side
        // `ResumeGuardDescr::external_jump_target` slot reached via
        // `as_any` downcast on `meta_descr`; cell membership is still
        // the canonical predicate.
        self.external_jump_target_ref().is_some()
    }

    fn target_descr(&self) -> Option<DescrRef> {
        self.external_jump_target_ref()
    }

    fn trace_id(&self) -> u64 {
        // Post-audit: gate forwarding on `meta_resume_fd()`.  PyPy's
        // `record_loop_or_bridge` (compile.py:183-185) stamps trace_id
        // only on `ResumeDescr` family members; `DoneWithThisFrame*`
        // and `ExitFrameWithExceptionDescrRef` do not override
        // `trace_id()` upstream and would return the trait default 0
        // — masking the backend-local construction-time trace_id.
        // Fallback to backend-local field when meta_descr is absent
        // or non-ResumeDescr.
        self.meta_resume_fd()
            .map_or(self.trace_id, |fd| fd.trace_id())
    }

    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        // `history.py:132` `AbstractFailDescr._attrs_` `rd_loop_token` —
        // forward through `meta_descr` to the metainterp ResumeGuardDescr.
        // `record_loop_or_bridge` only stamps ResumeDescr family
        // (compile.py:183-186), so meta_descr is always present when
        // the read fires in production.
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .and_then(|fd| fd.rd_loop_token_clt())
    }

    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        // `compile.py:186` `descr.rd_loop_token = clt` — write through
        // to the metainterp ResumeGuardDescr.  Caller (compiler.rs walker)
        // gates on `descr.is_resume_guard()` before invocation, so
        // meta_descr is always present here in production.
        if let Some(meta_fd) = self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            meta_fd.set_rd_loop_token_clt(clt);
        }
    }

    fn is_gc_ref_slot(&self, slot: usize) -> bool {
        // gc_map is derived on demand from fail_arg_types +
        // force_token_slots (Session 5i-cl).  Match the inline
        // semantics of `gc_map_for_types`: slot is a GC ref iff its
        // type is Ref AND the slot is not a force-token producer.
        // Forward through `<Self as FailDescr>::fail_arg_types` so the
        // meta_descr override (set by `store_final_boxes_in_guard`)
        // drives classification.
        match <Self as FailDescr>::fail_arg_types(self).get(slot) {
            Some(Type::Ref) => !self.is_force_token_slot(slot),
            _ => false,
        }
    }

    fn force_token_slots(&self) -> Vec<usize> {
        self.force_token_slots_view().to_vec()
    }

    fn vector_info(&self) -> Vec<AccumInfo> {
        // `history.py:132` `AbstractFailDescr._attrs_` `rd_vector_info`
        // — the canonical store lives on the metainterp
        // `AbstractFailDescr`, reached via `meta_descr`.  Synthetic /
        // FINISH descrs without a `meta_descr` carry no vector info.
        self.meta_descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .map(|fd| fd.vector_info())
            .unwrap_or_default()
    }

    /// `compile.py:741-745` `get_status`.  Forwards through the
    /// metainterp `AbstractResumeGuardDescr` (`compile.py:683 _attrs_`
    /// `('status',)`) when `meta_descr` is set; falls back to the
    /// backend-local mirror for synthetic descrs minted outside the
    /// optimizer.
    fn get_status(&self) -> u64 {
        // `compile.py:683 AbstractResumeGuardDescr._attrs_ = ('status',)`
        // — only ResumeGuardDescr family carries this slot.  Forward
        // through meta_descr; non-ResumeGuardDescr targets take the
        // trait default 0, matching upstream.
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

    /// `compile.py:750` check `ST_BUSY_FLAG`.
    fn is_compiling(&self) -> bool {
        self.get_status() & majit_backend::STATUS_BUSY_FLAG != 0
    }

    // resume.py:450-488 readers gated on `meta_resume_fd()` —
    // `isinstance(descr, ResumeDescr)` per `record_loop_or_bridge`.
    // Non-ResumeDescr meta descrs return None for these by trait
    // default (compile.rs Done*/ExitExc/PropagateException don't
    // override the rd_* setters), but the gate keeps the rule
    // explicit so future readers don't accidentally pull None from a
    // FINISH meta descr when a real `Some` was expected.
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
    /// Original attached `jf_descr` identity for finish exits emitted by
    /// the metainterp (`DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`).
    pub latest_descr: Option<DescrRef>,
    /// True when `register_roots` has registered `jf_gcref` with the
    /// active cranelift GC, so `Drop` knows to remove it. Replaces the
    /// pre-removal `gc_runtime_id` field that paired registration with
    /// a per-trace runtime id; the active GC is now a single thread-local
    /// (`compiler.rs CRANELIFT_ACTIVE_GC`, mirroring `llmodel.py:58`).
    pub roots_registered: bool,
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
        latest_descr: Option<DescrRef>,
        heap_owner: Option<Vec<i64>>,
    ) -> Self {
        JitFrameDeadFrame {
            jf_gcref,
            fail_descr,
            latest_descr,
            roots_registered: false,
            _heap_owner: heap_owner,
        }
    }

    pub fn register_roots(&mut self) {
        self.roots_registered = register_gc_roots(std::slice::from_mut(&mut self.jf_gcref));
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
        if self.roots_registered {
            unregister_gc_roots(std::slice::from_mut(&mut self.jf_gcref));
        }
    }
}
