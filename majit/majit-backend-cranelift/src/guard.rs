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
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

// BRIDGE_CACHES_TABLE removed (Slice JJ): the per-descr
// `Box<AtomicUsize>` cells for bridge code_ptr / frame_depth now live
// on CraneliftFailDescr directly.  Box gives each cell a heap-pinned
// address that survives the descr being moved into `Arc::new(...)`;
// the JIT bakes those addresses into the machine code
// (`compiler.rs::emit_attached_bridge_dispatch`), so they must remain
// stable for the descr's lifetime.

/// Backend-static side-table mapping a `CraneliftFailDescr` Arc's
/// `Arc::as_ptr` address to its force-token slot vector.
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) carries no
/// `force_token_slots`; upstream `assembler.py` handles force-token
/// produce/consume as a codegen-time concern, with the slot positions
/// encoded into the machine code's GC-map immediates.  Cranelift IR
/// has no equivalent inline encoding, so pyre retains the per-descr
/// vector in this side-table for runtime GC-root filtering.  The
/// table is consulted by `FailDescr::force_token_slots()` and
/// `is_force_token_slot()`.
// FORCE_TOKEN_SLOTS_TABLE removed (Slice II): write-once at codegen
// (the slot positions are determined by the trace's force-token
// produce/consume pairs and do not change after the descr is sealed).
// Lives in `force_token_slots_cell: OnceLock<Vec<usize>>` on
// CraneliftFailDescr.  Empty vectors are still elided
// (`force_token_slots_view` returns `&[]` for unset cells).

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

/// Backend-static side-table mapping a `CraneliftFailDescr` Arc's
/// `Arc::as_ptr` address to its external-JUMP target `DescrRef`.
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) does not
/// carry `is_external_jump` / `target_descr` slots; upstream
/// `assembler.py:2456-2462 closing_jump` emits a raw inter-function
/// JMP to `target_token._ll_loop_code`.  Cranelift can't emit raw
/// inter-function JMPs, so the exit returns to the dispatcher which
/// reads the target descr to re-enter via the registered
/// `JitCellToken.number → RegisteredLoopTarget` metadata.  Pyre
/// keeps the per-descr target as a backend-static side-table entry
/// keyed on `Arc::as_ptr(&descr)`.  Membership in the table is the
/// canonical `is_external_jump` predicate.
// EXTERNAL_JUMP_TARGETS removed (Slice GG): write-once after
// construction (no in-place mutation after `set_external_jump_target`).
// Lives in `external_jump_target_cell: OnceLock<DescrRef>` on
// CraneliftFailDescr.  PyPy emits a raw inter-function JMP at
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
    /// `compile.py:attach_bridge` / `assembler.py:987 patch_jump_for_descr`
    /// parity — descr-local atomic cell holding the published
    /// `Arc<BridgeData>` raw pointer.  Equivalent to PyPy's
    /// `adr_jump_offset` (`history.py:132 _attrs_`): a stable raw
    /// memory cell whose contents are patched in-place at
    /// `attach_bridge` time and read lock-free by the guard-failure
    /// dispatch (PyPy: JMP rel32; pyre: `bridge_ref` atomic load +
    /// `Arc::increment_strong_count`).  `Box` gives the cell a
    /// heap-pinned address (descr is wrapped in `Arc::new` after
    /// construction; without Box the cell address would change).
    ///
    /// Null on construction (no bridge attached).  Holds a raw pointer
    /// from `Arc::into_raw(Arc::new(bridge_data))` after
    /// `attach_bridge`; `Drop` reclaims the Arc.
    pub bridge_dispatch_cell: Box<std::sync::atomic::AtomicPtr<BridgeData>>,
    /// `AbstractResumeGuardDescr.handle_fail` (`compile.py:701-717`)
    /// drives `must_compile` via `jitcounter.tick(status_hash)` in
    /// RPython.  Pyre's cranelift keeps a raw per-descr counter
    /// (`compiler.rs:3065 fail_descr.increment_fail_count()`).
    /// Moved here from `FAIL_COUNT_TABLE` (Slice DD) for the same
    /// reason as `bridge_dispatch_cell`: dispatch hot path was
    /// observing a Mutex+HashMap lookup per guard failure.
    pub fail_count: AtomicU32,
    /// Per-descr `CompiledTraceInfo` cell.  PyPy recovers the same
    /// state on demand from `cpu.asmmemmgr_blocks` +
    /// `compiled_loop_token`.  Cranelift parks the per-trace metadata
    /// (input types / header_pc / source_guard tuple) on the descr.
    ///
    /// Moved here from `TRACE_INFO_TABLE` (Slice FF) for the same
    /// reason as `recovery_layout_cell` (Slice EE): Mutex+HashMap
    /// lookup on the dispatch hot path.
    ///
    /// Null on construction.  Written via
    /// `Arc::into_raw(Arc::new(info))`; `Drop` reclaims the Arc.
    pub trace_info_cell: AtomicPtr<CompiledTraceInfo>,
    /// Per-descr external-JUMP target cell.  PyPy's
    /// `assembler.py:2456-2462 closing_jump` emits a raw inter-
    /// function JMP to `target_token._ll_loop_code`; cranelift cannot
    /// emit raw inter-function JMPs, so the exit returns to the
    /// dispatcher which reads the target descr here and re-enters via
    /// the registered `JitCellToken.number -> RegisteredLoopTarget`.
    ///
    /// Moved here from `EXTERNAL_JUMP_TARGETS` (Slice GG) for the same
    /// reason as `recovery_layout_cell` (Slice EE).  Membership in the
    /// cell (== `OnceLock.get().is_some()`) is the canonical
    /// `is_external_jump` predicate.
    ///
    /// Write-once: set by `CraneliftFailDescr::set_external_jump_target`
    /// at codegen finalisation.  `OnceLock` is the right primitive —
    /// the target is immutable for the descr's lifetime.
    pub external_jump_target_cell: OnceLock<DescrRef>,
    // source_op_index_cell moved to ResumeGuardDescr (Slice 7-Tβ6):
    // the meta Arc is the canonical home, reached via `as_any`
    // downcast on `meta_descr` (precedent: Slice OO-half for
    // recovery_layout).  Cranelift accessors `source_op_index_ref` /
    // `set_source_op_index` now forward through that chain.
    /// Force-token slot positions for runtime GC-root filtering.
    /// PyPy encodes the same information into the machine code's
    /// GC-map immediates (`assembler.py` handles force-token slot
    /// produce/consume inline); cranelift IR has no equivalent inline
    /// encoding so the vector lives on the descr.
    ///
    /// Moved here from `FORCE_TOKEN_SLOTS_TABLE` (Slice II).  Sorted
    /// and deduped by `set_force_token_slots` so
    /// `force_token_slots_view` satisfies the `binary_search`
    /// invariant used by `is_force_token_slot`.  Empty vectors are
    /// elided (the cell stays unset).
    pub force_token_slots_cell: OnceLock<Vec<usize>>,
    /// Bridge code-pointer cache.  JIT-baked into the dispatch path
    /// (`emit_attached_bridge_dispatch`).  `Box` gives the
    /// `AtomicUsize` a heap-pinned address that survives the descr
    /// being moved into `Arc::new(...)`.  `0` = no bridge attached.
    ///
    /// Moved here from `BRIDGE_CACHES_TABLE` (Slice JJ).
    pub bridge_code_ptr_cache: Box<AtomicUsize>,
    /// Bridge frame-depth cache.  Same shape as
    /// `bridge_code_ptr_cache`; baked into the dispatch path so the
    /// runtime can grow the JIT frame before re-entering the bridge.
    pub bridge_frame_depth_cache: Box<AtomicUsize>,
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
        // external_jump_target_cell is descr-local (Slice GG): drops
        // naturally with self.
        // fail_count is descr-local (Slice DD): drops naturally with self.
        // trace_info_cell is descr-local (Slice FF): reclaim the
        // published `Arc<CompiledTraceInfo>` by swapping the cell to
        // null and reconstructing the Arc.
        let info_ptr = self
            .trace_info_cell
            .swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !info_ptr.is_null() {
            // Safety: produced by `Arc::into_raw(Arc::new(info))` in
            // `set_trace_info`; reclaim ownership and drop.
            unsafe { drop(Arc::from_raw(info_ptr as *const CompiledTraceInfo)) };
        }
        // recovery_layout moved to ResumeGuardDescr meta-side slot
        // (Slice QQ-4); no backend-local cell to reclaim.
        // source_op_index moved to ResumeGuardDescr meta-side slot
        // (Slice 7-Tβ6); no backend-local cell to reclaim.
        // force_token_slots_cell is descr-local (Slice II): drops
        // naturally with self.
        // bridge_code_ptr_cache / bridge_frame_depth_cache are descr-
        // local Box<AtomicUsize> (Slice JJ): drop naturally with self.
        // Reclaim the published `Arc<BridgeData>` (if any) from the
        // descr-local dispatch cell.  Swap-to-null first so any
        // concurrent `bridge_ref` reader either sees the still-live
        // Arc (after `increment_strong_count`) or the null and skips.
        let bridge_ptr = self
            .bridge_dispatch_cell
            .swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !bridge_ptr.is_null() {
            // Safety: produced by `Arc::into_raw(Arc::new(bridge))` in
            // `attach_bridge`; reconstruct the owning Arc and drop it.
            unsafe { drop(Arc::from_raw(bridge_ptr as *const BridgeData)) };
        }
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
            .field(
                "has_bridge",
                &(!self.bridge_dispatch_cell.load(Ordering::Acquire).is_null()),
            )
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
    ///
    /// The `_is_finish` parameter is preserved for caller-site clarity
    /// during the transition; it is no longer stored on the descr —
    /// `compile.py:624 final_descr=True` is answered through meta_descr
    /// forwarding.
    pub fn new_with_trace_and_kind_and_force_tokens(
        fail_index: u32,
        trace_id: u64,
        fail_arg_types: Vec<Type>,
        _is_finish: bool,
        force_token_slots: Vec<usize>,
    ) -> Self {
        let descr = CraneliftFailDescr {
            fail_index,
            trace_id,
            fail_arg_types,
            meta_descr: None,
            bridge_dispatch_cell: Box::new(AtomicPtr::new(std::ptr::null_mut())),
            fail_count: AtomicU32::new(0),
            trace_info_cell: AtomicPtr::new(std::ptr::null_mut()),
            external_jump_target_cell: OnceLock::new(),
            force_token_slots_cell: OnceLock::new(),
            bridge_code_ptr_cache: Box::new(AtomicUsize::new(0)),
            bridge_frame_depth_cache: Box::new(AtomicUsize::new(0)),
        };
        descr.set_force_token_slots(force_token_slots);
        descr
    }

    /// Construct a fail descriptor for an external JUMP exit.
    /// assembler.py:2456-2462 closing_jump parity: JUMP whose target
    /// TargetToken lives in a different compiled function. Cranelift can't
    /// emit raw inter-function JMPs, so the dispatcher receives this descr
    /// and re-enters the target loop via the registered target token.
    pub fn new_external_jump(
        fail_index: u32,
        trace_id: u64,
        fail_arg_types: Vec<Type>,
        force_token_slots: Vec<usize>,
    ) -> Self {
        // Caller is expected to wrap the returned descr in `Arc::new(...)`
        // and immediately publish the external-JUMP target via
        // `descr.set_external_jump_target(target)`.  The constructor
        // cannot do this itself because the callsite needs to perform
        // additional in-place mutations (`set_source_op_index`,
        // `meta_descr`) before sealing the descr behind `Arc`.
        let descr = CraneliftFailDescr {
            fail_index,
            trace_id,
            fail_arg_types,
            meta_descr: None,
            bridge_dispatch_cell: Box::new(AtomicPtr::new(std::ptr::null_mut())),
            fail_count: AtomicU32::new(0),
            trace_info_cell: AtomicPtr::new(std::ptr::null_mut()),
            external_jump_target_cell: OnceLock::new(),
            force_token_slots_cell: OnceLock::new(),
            bridge_code_ptr_cache: Box::new(AtomicUsize::new(0)),
            bridge_frame_depth_cache: Box::new(AtomicUsize::new(0)),
        };
        descr.set_force_token_slots(force_token_slots);
        descr
    }

    // UnsafeCell accessor helpers — single-threaded, no lock needed.
    // RPython ResumeGuardDescr fields are plain attributes (GIL-protected).

    /// `assembler.py:987 patch_jump_for_descr` parity — read the
    /// descr-local atomic dispatch cell.  PyPy's dispatch is a JMP
    /// rel32 whose target is patched in-place by `attach_bridge`; pyre
    /// reads the `Arc<BridgeData>` raw pointer the JIT thread wrote
    /// there with `Arc::into_raw(Arc::new(...))`, then bumps the
    /// strong count and reconstructs the `Arc`.  Lock-free and
    /// HashMap-free (mirrors `adr_jump_offset` semantics).
    #[inline]
    pub fn bridge_ref(&self) -> Option<Arc<BridgeData>> {
        let ptr = self.bridge_dispatch_cell.load(Ordering::Acquire);
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
    /// Descr-local atomic read (Slice FF).  Returns the owned
    /// `CompiledTraceInfo` clone, or `None` when no trace info has been
    /// published for this descr.  Lock-free and HashMap-free.
    pub fn trace_info_ref(&self) -> Option<CompiledTraceInfo> {
        let ptr = self.trace_info_cell.load(Ordering::Acquire);
        if ptr.is_null() {
            None
        } else {
            // Safety: `ptr` was produced by
            // `Arc::into_raw(Arc::new(info))` in `set_trace_info`;
            // increment_strong_count + from_raw yields an extra owning
            // Arc the caller can deref + clone.
            unsafe {
                Arc::increment_strong_count(ptr as *const CompiledTraceInfo);
                let arc = Arc::from_raw(ptr as *const CompiledTraceInfo);
                Some((*arc).clone())
            }
        }
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

    /// Increment the failure counter and return the new value.
    /// Backed by the descr-local `fail_count: AtomicU32` field
    /// (Slice DD) — single relaxed `fetch_add` on the dispatch hot
    /// path (`compiler.rs:3065`).
    pub fn increment_fail_count(&self) -> u32 {
        self.fail_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Get the current failure count (Slice DD: descr-local atomic).
    pub fn get_fail_count(&self) -> u32 {
        self.fail_count.load(Ordering::Relaxed)
    }

    /// Descr-local atomic read (Slice JJ) — whether a bridge has been
    /// attached to this guard.
    pub fn has_bridge(&self) -> bool {
        self.bridge_code_ptr_cache.load(Ordering::Acquire) != 0
    }

    /// Descr-local atomic read (Slice JJ) — bridge code_ptr.
    pub fn bridge_code_ptr(&self) -> *const u8 {
        self.bridge_code_ptr_cache.load(Ordering::Acquire) as *const u8
    }

    /// Heap-pinned addresses of the two bridge-cache atomic cells,
    /// suitable for baking into JIT machine code as immediates.
    /// Returns `(code_ptr_addr, frame_depth_addr)`.
    pub fn bridge_cache_addrs(&self) -> (usize, usize) {
        (
            self.bridge_code_ptr_cache.as_ref() as *const _ as usize,
            self.bridge_frame_depth_cache.as_ref() as *const _ as usize,
        )
    }

    /// `compile.py:attach_bridge` / `assembler.py:987 patch_jump_for_descr`
    /// parity — atomic-store the bridge `Arc` raw pointer into the
    /// descr-local dispatch cell.  PyPy patches the JMP rel32; pyre
    /// patches the heap-pinned atomic cell.  Cell address is stable
    /// for the descr's lifetime (heap-pinned via `Box`), so the
    /// JIT-baked dispatch can read it lock-free.
    pub fn attach_bridge(&self, bridge: BridgeData) {
        let code_ptr = bridge.code_ptr as usize;
        let frame_depth = bridge
            .max_output_slots
            .max(bridge.num_inputs)
            .max(1)
            .saturating_add(bridge.num_ref_roots);
        // `Arc::into_raw(Arc::new(bridge))` publishes the bridge data
        // as a raw pointer the dispatch path can re-Arc via
        // `increment_strong_count + Arc::from_raw`.  Swap atomically so
        // a re-attach (unusual) reclaims the previous Arc.
        let new_ptr = Arc::into_raw(Arc::new(bridge)) as *mut BridgeData;
        let old_ptr = self.bridge_dispatch_cell.swap(new_ptr, Ordering::AcqRel);
        if !old_ptr.is_null() {
            // Safety: prior `attach_bridge` published this pointer;
            // reclaim ownership and drop.
            unsafe { drop(Arc::from_raw(old_ptr as *const BridgeData)) };
        }
        self.bridge_frame_depth_cache
            .store(frame_depth, Ordering::Release);
        self.bridge_code_ptr_cache
            .store(code_ptr, Ordering::Release);
    }

    /// Descr-local write-once cell (Slice GG).  Publishes the
    /// external-JUMP target descr into the `OnceLock`.  Replaces the
    /// previous backend-static `EXTERNAL_JUMP_TARGETS` insert.
    /// Idempotent on equal targets; panics on a conflicting re-set
    /// (caller bug — `closing_jump` resolves once at codegen).
    pub fn set_external_jump_target(&self, target: DescrRef) {
        self.external_jump_target_cell
            .set(target)
            .expect("external_jump_target_cell already published");
    }

    #[inline]
    /// Descr-local read (Slice GG).  Returns the published external-
    /// JUMP target, or `None` for descrs that are not external-JUMP
    /// exits.  Membership = the previous `is_external_jump: true`
    /// predicate.
    pub fn external_jump_target_ref(&self) -> Option<DescrRef> {
        self.external_jump_target_cell.get().cloned()
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

    /// Descr-local write-once cell (Slice II).  Sorts+dedups the
    /// vector so the stored slot list satisfies `binary_search`
    /// (used by `is_force_token_slot`).  Empty vectors are elided
    /// (cell stays unset; `force_token_slots_view` returns `&[]`).
    pub fn set_force_token_slots(&self, mut slots: Vec<usize>) {
        slots.sort_unstable();
        slots.dedup();
        if slots.is_empty() {
            return;
        }
        self.force_token_slots_cell
            .set(slots)
            .expect("force_token_slots_cell already published");
    }

    #[inline]
    /// Descr-local read (Slice II).  Returns the published slot list
    /// as a slice or `&[]` when no slots have been registered.
    pub fn force_token_slots_view(&self) -> &[usize] {
        self.force_token_slots_cell.get().map_or(&[], Vec::as_slice)
    }

    /// Descr-local atomic write (Slice FF).  Callers are `compile_loop`
    /// (codegen finaliser) and `overlay_deadframe_fail_descr`
    /// (CALL_ASSEMBLER prefix overlay).  Publishes the trace info via
    /// `Arc::into_raw(Arc::new(...))`; any previously published Arc is
    /// reclaimed by the swap.
    pub fn set_trace_info(self: &Arc<Self>, trace_info: CompiledTraceInfo) {
        let new_ptr = Arc::into_raw(Arc::new(trace_info)) as *mut CompiledTraceInfo;
        let old_ptr = self.trace_info_cell.swap(new_ptr, Ordering::AcqRel);
        if !old_ptr.is_null() {
            // Safety: prior `set_trace_info` published this pointer;
            // reclaim ownership and drop.
            unsafe { drop(Arc::from_raw(old_ptr as *const CompiledTraceInfo)) };
        }
    }

    /// Derive the `GcMap` on demand from `fail_arg_types` and the
    /// descr-local `force_token_slots_cell` (Slice II).  Replaces the
    /// previous `pub gc_map: GcMap` field (Session 5i-cl); upstream
    /// `assembler.py:write_failure_recovery_description` parity
    /// recomputes equivalent bits inline at codegen time.
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
        // Vector stored in `force_token_slots_cell` is sorted+deduped
        // at set time, preserving the `binary_search` invariant.
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
        if self.external_jump_target_cell.get().is_some() {
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
        // descrs are synthesized at the cranelift backend for
        // cross-loop JUMP targets and have meta_descr == None.  Slice
        // GG moved the per-descr target to `external_jump_target_cell`;
        // cell membership is the canonical predicate.
        self.external_jump_target_cell.get().is_some()
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
