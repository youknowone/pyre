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
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

/// Backend-static side-table mapping a `CraneliftFailDescr` Arc's
/// `Arc::as_ptr` address to its attached `BridgeData` (if any).
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) carries no
/// `bridge` slot; upstream `compile.py:attach_bridge` patches the
/// failing guard's machine-code JMP to point at the bridge entry,
/// leaving the descr untouched.  Cranelift cannot patch finalised
/// code, so it parks the per-descr `BridgeData` here and the JIT
/// dispatch path reads it on guard failure (mediated by
/// `BRIDGE_CACHES_TABLE` for the lock-free atomic cells).
///
/// Held as `Arc<BridgeData>` so `bridge_ref` can hand out a cloned
/// reference without holding the table mutex across the read.
static BRIDGE_TABLE: OnceLock<Mutex<HashMap<usize, Arc<BridgeData>>>> = OnceLock::new();

fn bridge_table() -> &'static Mutex<HashMap<usize, Arc<BridgeData>>> {
    BRIDGE_TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_bridge(descr_ptr: usize, bridge: BridgeData) {
    bridge_table()
        .lock()
        .expect("BRIDGE_TABLE mutex poisoned")
        .insert(descr_ptr, Arc::new(bridge));
}

pub fn lookup_bridge(descr_ptr: usize) -> Option<Arc<BridgeData>> {
    bridge_table()
        .lock()
        .expect("BRIDGE_TABLE mutex poisoned")
        .get(&descr_ptr)
        .cloned()
}

/// Backend-static side-table mapping a `CraneliftFailDescr` Arc's
/// `Arc::as_ptr` address to its `Box<AtomicUsize>` bridge caches.
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) carries no
/// `bridge_code_ptr_cache` / `bridge_frame_depth_cache`; upstream
/// `assembler.py:987 patch_jump_for_descr` patches the failing
/// guard's jump in place to point at the bridge entry (so no
/// per-descr cache is ever loaded by the dispatch path).  Cranelift
/// cannot patch finalised code; instead the guard exit emits a
/// `load` from these atomic cells.  Each cell's address must be
/// stable for the descr's lifetime because the JIT bakes it into
/// the machine code (`compiler.rs::emit_attached_bridge_dispatch`).
/// Boxing the atomics gives them a heap-pinned address that survives
/// even after the descr struct is moved into its owning `Arc`.
///
/// Cells are dropped when the descr is dropped (`Drop for
/// CraneliftFailDescr`); the JIT code holding the baked address has
/// already been invalidated because the owning trace is evicted
/// before the descr's last `Arc` clone drops (`compile.py:185-203
/// record_loop_or_bridge` lifecycle).
struct BridgeCaches {
    code_ptr: Box<std::sync::atomic::AtomicUsize>,
    frame_depth: Box<std::sync::atomic::AtomicUsize>,
}

static BRIDGE_CACHES_TABLE: OnceLock<Mutex<HashMap<usize, BridgeCaches>>> = OnceLock::new();

fn bridge_caches_table() -> &'static Mutex<HashMap<usize, BridgeCaches>> {
    BRIDGE_CACHES_TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Lazily allocate the bridge cache cells for `descr_ptr` and return
/// stable pointers to the boxed atomics.  Returns
/// `(code_ptr_addr, frame_depth_addr)`.  Repeated calls for the same
/// descr return the same addresses — required because the JIT code
/// embeds them as immediates.
pub fn bridge_cache_addrs(descr_ptr: usize) -> (usize, usize) {
    let mut table = bridge_caches_table()
        .lock()
        .expect("BRIDGE_CACHES_TABLE mutex poisoned");
    let entry = table.entry(descr_ptr).or_insert_with(|| BridgeCaches {
        code_ptr: Box::new(std::sync::atomic::AtomicUsize::new(0)),
        frame_depth: Box::new(std::sync::atomic::AtomicUsize::new(0)),
    });
    (
        entry.code_ptr.as_ref() as *const _ as usize,
        entry.frame_depth.as_ref() as *const _ as usize,
    )
}

/// Load the current bridge code pointer cache value.  Returns `0` for
/// descrs with no entry (no bridge ever attached).
pub fn load_bridge_code_ptr(descr_ptr: usize) -> usize {
    bridge_caches_table()
        .lock()
        .expect("BRIDGE_CACHES_TABLE mutex poisoned")
        .get(&descr_ptr)
        .map(|c| c.code_ptr.load(std::sync::atomic::Ordering::Acquire))
        .unwrap_or(0)
}

/// Store the bridge code pointer + frame depth caches atomically.
/// Lazily allocates the boxed cells if they don't yet exist so the
/// JIT-baked addresses remain stable across the runtime's first
/// `bridge_cache_addrs` call.
pub fn store_bridge_caches(descr_ptr: usize, code_ptr: usize, frame_depth: usize) {
    let mut table = bridge_caches_table()
        .lock()
        .expect("BRIDGE_CACHES_TABLE mutex poisoned");
    let entry = table.entry(descr_ptr).or_insert_with(|| BridgeCaches {
        code_ptr: Box::new(std::sync::atomic::AtomicUsize::new(0)),
        frame_depth: Box::new(std::sync::atomic::AtomicUsize::new(0)),
    });
    entry
        .frame_depth
        .store(frame_depth, std::sync::atomic::Ordering::Release);
    entry
        .code_ptr
        .store(code_ptr, std::sync::atomic::Ordering::Release);
}

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
static FORCE_TOKEN_SLOTS_TABLE: OnceLock<Mutex<HashMap<usize, Vec<usize>>>> = OnceLock::new();

fn force_token_slots_table() -> &'static Mutex<HashMap<usize, Vec<usize>>> {
    FORCE_TOKEN_SLOTS_TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Codegen-time write.  Sorts+dedupes the vector internally so the
/// stored slot list satisfies the `binary_search` invariant used by
/// `is_force_token_slot`.  Empty vectors are skipped — `lookup_*`
/// returns an empty `Vec` when the descr has no entry.
pub fn register_force_token_slots(descr_ptr: usize, mut slots: Vec<usize>) {
    slots.sort_unstable();
    slots.dedup();
    if slots.is_empty() {
        return;
    }
    force_token_slots_table()
        .lock()
        .expect("FORCE_TOKEN_SLOTS_TABLE mutex poisoned")
        .insert(descr_ptr, slots);
}

pub fn lookup_force_token_slots(descr_ptr: usize) -> Vec<usize> {
    force_token_slots_table()
        .lock()
        .expect("FORCE_TOKEN_SLOTS_TABLE mutex poisoned")
        .get(&descr_ptr)
        .cloned()
        .unwrap_or_default()
}

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
static RECOVERY_LAYOUT_TABLE: OnceLock<Mutex<HashMap<usize, ExitRecoveryLayout>>> = OnceLock::new();

fn recovery_layout_table() -> &'static Mutex<HashMap<usize, ExitRecoveryLayout>> {
    RECOVERY_LAYOUT_TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Codegen-time write.  Callers wrap the descr in `Arc::new(...)` then
/// invoke `register_recovery_layout(Arc::as_ptr(&descr) as usize, layout)`.
pub fn register_recovery_layout(descr_ptr: usize, layout: ExitRecoveryLayout) {
    recovery_layout_table()
        .lock()
        .expect("RECOVERY_LAYOUT_TABLE mutex poisoned")
        .insert(descr_ptr, layout);
}

/// Layout / dispatch-time read.  Returns owned `Option<ExitRecoveryLayout>`
/// (cloned from the table) so callers can hold it past the lock.
pub fn lookup_recovery_layout(descr_ptr: usize) -> Option<ExitRecoveryLayout> {
    recovery_layout_table()
        .lock()
        .expect("RECOVERY_LAYOUT_TABLE mutex poisoned")
        .get(&descr_ptr)
        .cloned()
}

/// Backend-static side-table mapping a `CraneliftFailDescr` Arc's
/// `Arc::as_ptr` address to its compile-time `CompiledTraceInfo`.
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) carries no
/// `trace_info` slot — RPython recovers the same information from
/// `cpu.asmmemmgr_blocks` + `compiled_loop_token`.  Cranelift's
/// per-trace metadata (input types / header_pc / source_guard tuple)
/// is the equivalent state, parked here so the descr struct stays
/// aligned with PyPy's surface.
static TRACE_INFO_TABLE: OnceLock<Mutex<HashMap<usize, CompiledTraceInfo>>> = OnceLock::new();

fn trace_info_table() -> &'static Mutex<HashMap<usize, CompiledTraceInfo>> {
    TRACE_INFO_TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Codegen-time write: invoked from `compile_loop` /
/// `overlay_deadframe_fail_descr` once the descr Arc is materialised.
pub fn register_trace_info(descr_ptr: usize, info: CompiledTraceInfo) {
    trace_info_table()
        .lock()
        .expect("TRACE_INFO_TABLE mutex poisoned")
        .insert(descr_ptr, info);
}

/// Layout / dispatch-time read.  Returns `None` when the descr has no
/// associated trace info (synthetic descrs, descrs built outside the
/// `compile_loop` path).
pub fn lookup_trace_info(descr_ptr: usize) -> Option<CompiledTraceInfo> {
    trace_info_table()
        .lock()
        .expect("TRACE_INFO_TABLE mutex poisoned")
        .get(&descr_ptr)
        .cloned()
}

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
static EXTERNAL_JUMP_TARGETS: OnceLock<Mutex<HashMap<usize, DescrRef>>> = OnceLock::new();

fn external_jump_targets() -> &'static Mutex<HashMap<usize, DescrRef>> {
    EXTERNAL_JUMP_TARGETS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Codegen-time write: invoked from `CraneliftFailDescr::new_external_jump`
/// once the descr is wrapped in its owning `Arc`.  The Arc address is
/// stable for the descr's lifetime; the table entry remains until the
/// descr is dropped (matching the previous in-descr `target_descr:
/// Option<DescrRef>` semantics — no explicit cleanup).
pub fn register_external_jump_target(descr_ptr: usize, target: DescrRef) {
    external_jump_targets()
        .lock()
        .expect("EXTERNAL_JUMP_TARGETS mutex poisoned")
        .insert(descr_ptr, target);
}

/// Runtime read: returns the registered target descr, or `None` for
/// descrs that are not external JUMP exits.  Membership equates to
/// the previous `is_external_jump: true` predicate.
pub fn lookup_external_jump_target(descr_ptr: usize) -> Option<DescrRef> {
    external_jump_targets()
        .lock()
        .expect("EXTERNAL_JUMP_TARGETS mutex poisoned")
        .get(&descr_ptr)
        .cloned()
}

/// Backend-static side-table mapping a `CraneliftFailDescr` Arc's
/// `Arc::as_ptr` address to a `AtomicU32` failure counter.
///
/// PyPy's `AbstractFailDescr._attrs_` (`history.py:132`) does not
/// carry a `fail_count` slot; RPython's bridge-compilation threshold
/// is driven by the hashed `jitcounter.tick(status_hash)` slot
/// (`compile.py:783-784`).  Pyre's cranelift keeps a raw per-descr
/// counter as the bridge-decision input — equivalent intent,
/// different mechanism.  Moving it off the descr is the surface-
/// matching step toward eventually folding cranelift fail-count
/// decisions into the shared metainterp `status` slot.
static FAIL_COUNT_TABLE: OnceLock<Mutex<HashMap<usize, AtomicU32>>> = OnceLock::new();

fn fail_count_table() -> &'static Mutex<HashMap<usize, AtomicU32>> {
    FAIL_COUNT_TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Increment the per-descr failure counter, returning the new value.
/// Lazily inserts an `AtomicU32(0)` entry on first access — descrs
/// constructed without an explicit register call still observe a
/// monotonically increasing count, matching the previous in-descr
/// `AtomicU32::fetch_add` semantics.
pub fn increment_fail_count(descr_ptr: usize) -> u32 {
    let mut table = fail_count_table()
        .lock()
        .expect("FAIL_COUNT_TABLE mutex poisoned");
    let entry = table.entry(descr_ptr).or_insert_with(|| AtomicU32::new(0));
    entry.fetch_add(1, Ordering::Relaxed) + 1
}

/// Read the per-descr failure counter.  Returns `0` for descrs that
/// have never failed (no entry yet), matching the previous in-descr
/// initial value.
pub fn get_fail_count(descr_ptr: usize) -> u32 {
    fail_count_table()
        .lock()
        .expect("FAIL_COUNT_TABLE mutex poisoned")
        .get(&descr_ptr)
        .map(|c| c.load(Ordering::Relaxed))
        .unwrap_or(0)
}

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
    /// compile.py:658-662 ExitFrameWithExceptionDescrRef parity.
    /// True when this FINISH was emitted via
    /// pyjitpl.py:3238-3245 compile_exit_frame_with_exception.
    pub is_exit_frame_with_exception: bool,
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
    // trace_info removed (Session 5i-cl): not in PyPy
    // `AbstractFailDescr._attrs_` (`history.py:132`).  Cranelift's
    // per-trace `CompiledTraceInfo` now lives in `TRACE_INFO_TABLE`
    // keyed on `Arc::as_ptr(&descr)`; RPython recovers the same
    // information from `cpu.asmmemmgr_blocks`.
    // recovery_layout removed (Session 5i-cl): not in PyPy
    // `AbstractFailDescr._attrs_` (`history.py:132`).  Upstream
    // resume code decodes recovery on demand from the four payload
    // attributes (rd_numb / rd_consts / rd_virtuals / rd_pendingfields)
    // in `resume.py:450-488`.  Cranelift retains the structured
    // layout in `RECOVERY_LAYOUT_TABLE` (this module) keyed on
    // `Arc::as_ptr(&descr)`.
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
}

impl Drop for CraneliftFailDescr {
    /// Backend-static side-tables (`EXTERNAL_JUMP_TARGETS`,
    /// `FAIL_COUNT_TABLE`, `FORCE_TOKEN_SLOTS_TABLE`,
    /// `BRIDGE_CACHES_TABLE`, `BRIDGE_TABLE`) are keyed on the descr's
    /// inner address.  Without cleanup the entry would outlive the
    /// descr and the allocator may reuse the freed address for a
    /// future descr that would then observe stale state.
    ///
    /// Bridge cleanup must drop the removed `Arc<BridgeData>` AFTER
    /// the `BRIDGE_TABLE` lock is released — `BridgeData::fail_descrs`
    /// holds `Arc<CraneliftFailDescr>` clones whose own `Drop` will
    /// recursively re-acquire the same mutex on the same thread, and
    /// `std::sync::Mutex` is non-reentrant.
    fn drop(&mut self) {
        let ptr = self as *const Self as usize;
        external_jump_targets()
            .lock()
            .expect("EXTERNAL_JUMP_TARGETS mutex poisoned")
            .remove(&ptr);
        fail_count_table()
            .lock()
            .expect("FAIL_COUNT_TABLE mutex poisoned")
            .remove(&ptr);
        trace_info_table()
            .lock()
            .expect("TRACE_INFO_TABLE mutex poisoned")
            .remove(&ptr);
        recovery_layout_table()
            .lock()
            .expect("RECOVERY_LAYOUT_TABLE mutex poisoned")
            .remove(&ptr);
        source_op_index_table()
            .lock()
            .expect("SOURCE_OP_INDEX_TABLE mutex poisoned")
            .remove(&ptr);
        force_token_slots_table()
            .lock()
            .expect("FORCE_TOKEN_SLOTS_TABLE mutex poisoned")
            .remove(&ptr);
        bridge_caches_table()
            .lock()
            .expect("BRIDGE_CACHES_TABLE mutex poisoned")
            .remove(&ptr);
        // Take ownership of the bridge under a scoped lock, then drop
        // the Arc outside it to avoid the non-reentrant deadlock when
        // `BridgeData::fail_descrs`' inner Arcs cascade into descr
        // `Drop` calls that re-acquire `BRIDGE_TABLE`.
        let removed_bridge = {
            let mut guard = bridge_table()
                .lock()
                .expect("BRIDGE_TABLE mutex poisoned");
            guard.remove(&ptr)
        };
        drop(removed_bridge);
    }
}

impl std::fmt::Debug for CraneliftFailDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CraneliftFailDescr")
            .field("fail_index", &self.fail_index)
            .field(
                "source_op_index",
                &lookup_source_op_index(self as *const Self as usize),
            )
            .field("trace_id", &self.trace_id)
            .field("fail_arg_types", &self.fail_arg_types)
            .field("gc_map", &self.gc_map())
            .field("is_finish", &<Self as FailDescr>::is_finish(self))
            .field(
                "external_jump_target",
                &lookup_external_jump_target(self as *const Self as usize).map(|d| d.repr()),
            )
            .field(
                "force_token_slots",
                &lookup_force_token_slots(self as *const Self as usize),
            )
            .field(
                "trace_info",
                &lookup_trace_info(self as *const Self as usize),
            )
            .field(
                "recovery_layout",
                &lookup_recovery_layout(self as *const Self as usize),
            )
            .field(
                "fail_count",
                &get_fail_count(self as *const Self as usize),
            )
            .field(
                "has_bridge",
                &lookup_bridge(self as *const Self as usize).is_some(),
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

    /// Create a new fail descriptor.
    pub fn new(fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            false,
            Vec::new(),
        )
    }

    pub fn new_with_kind(fail_index: u32, fail_arg_types: Vec<Type>, _is_finish: bool) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            _is_finish,
            Vec::new(),
        )
    }

    pub fn new_with_kind_and_force_tokens(
        fail_index: u32,
        fail_arg_types: Vec<Type>,
        _is_finish: bool,
        force_token_slots: Vec<usize>,
    ) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            _is_finish,
            force_token_slots,
        )
    }

    /// Caller responsibility after `Arc::new(descr)`:
    ///   - if `recovery_layout` was previously passed: invoke
    ///     `descr.set_recovery_layout(layout)` to install the layout
    ///     into the backend-static `RECOVERY_LAYOUT_TABLE` (Session
    ///     5i-cl).  Constructor no longer accepts the layout
    ///     because the descr's `Arc::as_ptr` address (the table key)
    ///     is not knowable until wrapping completes.
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
        mut force_token_slots: Vec<usize>,
    ) -> Self {
        force_token_slots.sort_unstable();
        force_token_slots.dedup();
        CraneliftFailDescr {
            fail_index,
            trace_id,
            fail_arg_types,
            is_exit_frame_with_exception: false,
            meta_descr: None,
        }
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
        mut force_token_slots: Vec<usize>,
    ) -> Self {
        // Caller is expected to wrap the returned descr in `Arc::new(...)`
        // and immediately register the external-JUMP target via
        // `register_external_jump_target(Arc::as_ptr(&descr) as usize,
        // target_descr)`.  The constructor cannot do this itself because
        // the callsite needs to perform additional in-place mutations
        // (`set_source_op_index`, `is_exit_frame_with_exception`,
        // `meta_descr`) before sealing the descr behind `Arc`.
        force_token_slots.sort_unstable();
        force_token_slots.dedup();
        CraneliftFailDescr {
            fail_index,
            trace_id,
            fail_arg_types,
            is_exit_frame_with_exception: false,
            meta_descr: None,
        }
    }

    // UnsafeCell accessor helpers — single-threaded, no lock needed.
    // RPython ResumeGuardDescr fields are plain attributes (GIL-protected).

    /// Returns an owned `Option<Arc<BridgeData>>` (cloned from the
    /// backend-static `BRIDGE_TABLE`).  Cheap clone — `Arc` bump only.
    /// Signature changed from `&Option<BridgeData>` to
    /// `Option<Arc<BridgeData>>` in Session 5i-cl when the field moved
    /// off the descr struct (cannot hand out a borrow under the table
    /// lock).
    #[inline]
    pub fn bridge_ref(&self) -> Option<Arc<BridgeData>> {
        lookup_bridge(self as *const Self as usize)
    }

    #[inline]
    /// Backend-static side-table read (Session 5i-cl).  Returns the
    /// owned `CompiledTraceInfo` clone, or `None` when no trace info
    /// has been registered for this descr.
    pub fn trace_info_ref(&self) -> Option<CompiledTraceInfo> {
        lookup_trace_info(self as *const Self as usize)
    }

    #[inline]
    /// Backend-static side-table read (Session 5i-cl).  Returns an owned
    /// `Option<ExitRecoveryLayout>` (cloned from the table) so callers
    /// can hold it past the lock.  Was previously `&Option<…>` borrowed
    /// from `UnsafeCell`.
    pub fn recovery_layout_ref(&self) -> Option<ExitRecoveryLayout> {
        lookup_recovery_layout(self as *const Self as usize)
    }

    /// Increment the failure counter and return the new value.
    /// Backed by the `FAIL_COUNT_TABLE` side-table (Session 5i-cl) keyed
    /// on the descr's inner address — identical to `Arc::as_ptr(&arc)`
    /// for descrs constructed via `Arc::new(...)`.
    pub fn increment_fail_count(&self) -> u32 {
        increment_fail_count(self as *const Self as usize)
    }

    /// Get the current failure count.
    /// Backed by the `FAIL_COUNT_TABLE` side-table (Session 5i-cl).
    pub fn get_fail_count(&self) -> u32 {
        get_fail_count(self as *const Self as usize)
    }

    /// Whether a bridge has been attached to this guard.  Reads
    /// through the backend-static `BRIDGE_CACHES_TABLE` (Session 5i-cl).
    pub fn has_bridge(&self) -> bool {
        load_bridge_code_ptr(self as *const Self as usize) != 0
    }

    /// Get bridge code_ptr without Mutex lock (atomic read via boxed
    /// `AtomicUsize` in the side-table).
    pub fn bridge_code_ptr(&self) -> *const u8 {
        load_bridge_code_ptr(self as *const Self as usize) as *const u8
    }

    /// Attach a compiled bridge to this guard.  Writes the bridge
    /// caches through the backend-static `BRIDGE_CACHES_TABLE` so the
    /// JIT-embedded cell addresses stay stable.
    pub fn attach_bridge(&self, bridge: BridgeData) {
        let code_ptr = bridge.code_ptr as usize;
        let frame_depth = bridge
            .max_output_slots
            .max(bridge.num_inputs)
            .max(1)
            .saturating_add(bridge.num_ref_roots);
        register_bridge(self as *const Self as usize, bridge);
        store_bridge_caches(self as *const Self as usize, code_ptr, frame_depth);
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

    /// Backend-static side-table write (Session 5i-cl).
    pub fn set_recovery_layout(&self, recovery_layout: ExitRecoveryLayout) {
        register_recovery_layout(self as *const Self as usize, recovery_layout);
    }

    /// Backend-static side-table write (Session 5i-cl).  Takes
    /// `self: &Arc<Self>` because the table is keyed on
    /// `Arc::as_ptr(&descr)` — that is, the address of the heap-pinned
    /// inner `Self`.  Callers must wrap the descr in `Arc::new(...)`
    /// before invoking; writing with a stack-allocated `Self` key would
    /// leave a stale entry once the descr is moved into the Arc.
    pub fn set_source_op_index(self: &Arc<Self>, source_op_index: usize) {
        register_source_op_index(Arc::as_ptr(self) as usize, source_op_index);
    }

    /// Backend-static side-table write (Session 5i-cl).  Callers are
    /// `compile_loop` (codegen finaliser) and
    /// `overlay_deadframe_fail_descr` (CALL_ASSEMBLER prefix overlay).
    pub fn set_trace_info(self: &Arc<Self>, trace_info: CompiledTraceInfo) {
        register_trace_info(Arc::as_ptr(self) as usize, trace_info);
    }

    /// Derive the `GcMap` on demand from `fail_arg_types` and the
    /// side-table-stored `force_token_slots`.  Replaces the previous
    /// `pub gc_map: GcMap` field (Session 5i-cl); upstream
    /// `assembler.py:write_failure_recovery_description` parity
    /// recomputes equivalent bits inline at codegen time.
    pub fn gc_map(&self) -> GcMap {
        let force_token_slots = lookup_force_token_slots(self as *const Self as usize);
        Self::gc_map_for_types(&self.fail_arg_types, &force_token_slots)
    }

    pub fn is_force_token_slot(&self, slot: usize) -> bool {
        // Vector stored in `FORCE_TOKEN_SLOTS_TABLE` is sorted+deduped
        // at register time, preserving the `binary_search` invariant.
        lookup_force_token_slots(self as *const Self as usize)
            .binary_search(&slot)
            .is_ok()
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
        let recovery = lookup_recovery_layout(self as *const Self as usize);
        let frame_stack = recovery.as_ref().map(|r| r.frames.clone());
        FailDescrLayout {
            fail_index: self.fail_index,
            source_op_index: lookup_source_op_index(self as *const Self as usize),
            trace_id: <Self as FailDescr>::trace_id(self),
            trace_info: lookup_trace_info(self as *const Self as usize),
            fail_arg_types: fail_arg_types.to_vec(),
            is_finish: <Self as FailDescr>::is_finish(self),
            gc_ref_slots,
            force_token_slots: lookup_force_token_slots(self as *const Self as usize),
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
        if lookup_external_jump_target(self as *const Self as usize).is_some() {
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
        // lives on the metainterp Arc (see
        // `ExitFrameWithExceptionDescrRef` in `majit-metainterp/src/
        // compile.rs:2311-2358`).  Forward through `meta_descr` so
        // backend descrs constructed with `op.descr = Some(meta)` defer
        // to the metainterp class hierarchy.  Synthetic backend descrs
        // (runner classifier path, `meta_descr = None`) fall back to
        // the local mirror — still needed for backend-only descrs that
        // never visit the optimizer.
        match self.meta_descr.as_ref().and_then(|d| d.as_fail_descr()) {
            Some(fd) => fd.is_exit_frame_with_exception(),
            None => self.is_exit_frame_with_exception,
        }
    }

    fn is_external_jump(&self) -> bool {
        // Backend-only flag, no metainterp counterpart — external-JUMP
        // descrs are synthesized at the cranelift backend for
        // cross-loop JUMP targets and have meta_descr == None.  Session
        // 5i-cl moved the per-descr target to `EXTERNAL_JUMP_TARGETS`;
        // table membership is the canonical predicate.
        lookup_external_jump_target(self as *const Self as usize).is_some()
    }

    fn target_descr(&self) -> Option<DescrRef> {
        lookup_external_jump_target(self as *const Self as usize)
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
        match self.fail_arg_types.get(slot) {
            Some(Type::Ref) => {
                !lookup_force_token_slots(self as *const Self as usize).contains(&slot)
            }
            _ => false,
        }
    }

    fn force_token_slots(&self) -> Vec<usize> {
        lookup_force_token_slots(self as *const Self as usize)
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
        self.get_status() & Self::ST_BUSY_FLAG != 0
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
