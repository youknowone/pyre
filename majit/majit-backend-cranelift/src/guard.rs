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
//
// Per-descr backend state slots (source_op_index, recovery_layout,
// trace_info, external_jump_target, fail_count, bridge caches,
// bridge_dispatch_cell, force_token_slots) live on the metainterp
// `ResumeGuardDescr` (Slices 7-Tβ6..12) — see `majit-backend::
// resume_guard_descr`.  PyPy `AbstractFailDescr._attrs_` (history.py:132)
// carries none of these; Pyre's metainterp descr is the single source
// of truth for both the `_attrs_` set and the backend-only cells.
use crate::compiler::{register_gc_roots, unregister_gc_roots};
use majit_backend::{ExitRecoveryLayout, TerminalExitLayout};
use majit_ir::{DescrRef, GcRef, Type};
use std::cell::UnsafeCell;
use std::sync::Arc;

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
    pub fail_descrs: Box<[DescrRef]>,
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

// `CraneliftFailDescr` removed (Slice 7-Tβ14f).  PyPy's cranelift-
// equivalent `assembler.py` carries no per-emission descr wrapper —
// every guard op's `op.descr` is the same `AbstractFailDescr` Arc that
// the metainterp stamps via `compile.py:870 store_final_boxes_in_guard`,
// and every FINISH emission writes the cpu-attached class-distinct
// singleton (`finish_descrs.rs DoneWithThisFrameDescr*` /
// `ExitFrameWithExceptionDescrRef`) directly into the deadframe's
// `jf_descr` slot.  After Phases A/B/C of the Unified-Descr Port Epic
// the cranelift backend wrapper carried only forwarding shims, and the
// singleton-direct push (Slices 7-Tβ3 / 7-Tβ14a..e) eliminated the
// last constructor of the wrapper.  All readers reach the descr
// through `DescrRef` and dispatch trait methods on `&dyn FailDescr` /
// `&dyn Descr`.

/// Backend-registered cleanup for the type-erased
/// `ResumeGuardDescr::bridge_dispatch_cell` (Slice 7-Tβ12).  Invoked
/// by `ResumeGuardDescr::drop` on any payload still in the cell at
/// descr teardown; reconstructs the owning `Arc<BridgeData>` so its
/// `Drop` runs.
///
/// # Safety
/// `ptr` must be null or come from `Arc::into_raw` applied to an
/// `Arc<BridgeData>` that the caller hands ownership of to this
/// function.  Any other origin is undefined behavior.
pub(crate) unsafe fn drop_bridge_payload(ptr: *mut ()) {
    if !ptr.is_null() {
        unsafe { drop(Arc::from_raw(ptr as *const BridgeData)) };
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
    /// The fail descriptor for this exit.  Stored as `DescrRef`
    /// (`Arc<dyn Descr>`) so the deadframe carries the same Arc identity
    /// the metainterp stamps onto `op.descr` — matching PyPy's
    /// `frame.jf_descr = descr` (llmodel.py:270) line-by-line.  The
    /// backend wrapper `CraneliftFailDescr` is on its way out; this slot
    /// already accepts the upcast forwarders so the eventual deletion
    /// is a pure type substitution.
    pub fail_descr: DescrRef,
    /// Original attached `jf_descr` identity for finish exits emitted by
    /// the metainterp (`DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`).
    pub latest_descr: Option<DescrRef>,
    /// Slice X3-D side-channel: caller-prefix layout assembled from the
    /// `CALL_ASSEMBLER_CALLER_STACK` top at deadframe interception
    /// (`wrap_call_assembler_deadframe_with_caller_prefix`).  When `Some`,
    /// `compiler.rs::deadframe_layout` prefixes the descr's own recovery
    /// layout by this value before returning.  Replaces the old overlay
    /// descr synthesis (`overlay_deadframe_fail_descr` + overlay registry)
    /// — the deadframe's `fail_descr` now keeps the callee's own Arc
    /// identity rather than being swapped for a synthetic one.
    pub call_assembler_caller_layout: Option<ExitRecoveryLayout>,
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
        fail_descr: DescrRef,
        latest_descr: Option<DescrRef>,
        heap_owner: Option<Vec<i64>>,
    ) -> Self {
        JitFrameDeadFrame {
            jf_gcref,
            fail_descr,
            latest_descr,
            call_assembler_caller_layout: None,
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
