/// Guard failure descriptors and frame data for the wasm backend.
///
/// Simplified from CraneliftFailDescr — no bridge data, GC maps, or force tokens.
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use majit_ir::{Descr, DescrRef, FailDescr, Type};

/// Wasm-backend guard failure descriptor.
#[derive(Debug)]
pub struct WasmFailDescr {
    pub fail_index: u32,
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,
    /// `history.py:125 id(descr)` parity — when the optimizer
    /// (`store_final_boxes_in_guard` / `make_and_attach_done_descrs`)
    /// stamps a metainterp `ResumeGuardDescr` / `DoneWithThisFrame*` /
    /// `ExitFrameWithExceptionDescrRef` / `PropagateExceptionDescr` on
    /// `op.descr`, we keep it here so `get_latest_descr_arc` returns the
    /// canonical metainterp Arc (matching dynasm/cranelift).  `None`
    /// for synthetic backend-only descrs (`compile_bridge` placeholders,
    /// test scaffolds).
    pub meta_descr: Option<DescrRef>,
}

impl Descr for WasmFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }

    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for WasmFailDescr {
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
}

/// Wasm-backend dead frame data.
///
/// Stored inside `DeadFrame.data` after `execute_token` returns.
pub struct WasmFrameData {
    pub raw_values: Vec<i64>,
    pub fail_descr: Arc<WasmFailDescr>,
    /// Pending exception value captured by `execute_token` after the trace
    /// exited through a GuardNoException / GuardException (0 = none), surfaced
    /// via `grab_exc_value`.
    pub exc_value: i64,
}

/// A resumable `LABEL` of a compiled loop, published in `LABEL_TARGETS` so a
/// loop-closing bridge can chain into ANY compiled loop's label in-module
/// (jump-to-existing-trace), not only its own source loop's. Keyed by the
/// label's loop-target descr identity (`Arc::as_ptr`), which the JUMP shares.
#[derive(Clone, Copy, Debug)]
pub struct LabelTarget {
    /// Table slot of the owning loop's compiled function.
    pub func_handle: u32,
    /// Resume dispatch key (`label ordinal + 1`) the bridge's JUMP writes.
    pub key: u32,
    /// The label's arg count — the resume loader reads exactly this many
    /// positional frame slots, so the JUMP arity must equal it.
    pub num_args: usize,
    /// Whether the label's args are the complete live set of the owning
    /// trace's remainder (`codegen::label_resume_safety`).
    pub resume_safe: bool,
    /// Whether this is the owning loop's LAST label (the loop header). A
    /// bridge landing here re-runs no segment code before the `loop`, so the
    /// livelock advance-check applies; earlier labels execute the peeled
    /// segment, which advances the state by itself.
    pub is_last_label: bool,
    /// Frozen frame geometry of the target token. A tail-call can only reuse
    /// a frame when its offsets agree exactly, not merely when its allocation
    /// is large enough.
    pub frame: crate::codegen::FrameGeometry,
}

/// Frozen metadata for entering a compiled loop from a `CALL_ASSEMBLER` arm.
/// The table slot and frame layout are published only after the loop module is
/// installed, so a caller can decline before baking an unresolved target.
#[derive(Clone, Debug)]
pub struct CallAssemblerTarget {
    /// Owning `JitCellToken` number. Used only by the dormant wasm regression
    /// hook to select one target deterministically.
    pub token_number: u64,
    pub func_handle: u32,
    pub input_types: Vec<Type>,
    pub callee_frame_bytes: u32,
    pub callee_gcmap_ptr: i64,
    pub loop_finish_fi: u32,
    pub compiled_ptr: u64,
    pub has_trampoline_calls: bool,
}

/// Compiled loop targets keyed by their `JitCellToken` number. Unlike label
/// targets, CALL_ASSEMBLER identifies its callee by that number directly.
pub static CALL_ASSEMBLER_TARGETS: std::sync::Mutex<
    Option<std::collections::HashMap<u64, CallAssemblerTarget>>,
> = std::sync::Mutex::new(None);

// ── CALL_ASSEMBLER dispatch table ──
//
// A trace module imports the guest's linear memory, so a boxed wasm-side
// allocation is addressable by every trace with an ordinary i32.load.  The
// box keeps that address stable while the map grows; the emitted code must
// never bake a table slot because redirects and the pending->real transition
// replace it after the caller module was compiled.
#[repr(C)]
pub struct WasmCaDispatchEntry {
    /// `__indirect_function_table` slot. Zero means pending/unavailable.
    pub func_handle: AtomicU32,
    /// Callee DoneWithThisFrame fail index. `u32::MAX` means not installed.
    pub loop_finish_fi: AtomicU32,
    /// `CompiledWasmLoop` address for the deopt helper, in wasm32 memory.
    pub compiled_ptr: AtomicU32,
}

pub const WASM_CA_DISPATCH_FUNC_HANDLE_OFS: u64 = 0;
pub const WASM_CA_DISPATCH_LOOP_FINISH_FI_OFS: u64 = 4;
pub const WASM_CA_DISPATCH_COMPILED_PTR_OFS: u64 = 8;
pub const WASM_CA_FINISH_FI_UNKNOWN: u32 = u32::MAX;

/// Stable, guest-memory dispatch entries, keyed by CALL_ASSEMBLER token.
/// `Box` is intentional: an emitted module bakes the entry address.
pub static WASM_CA_DISPATCH: std::sync::Mutex<
    Option<std::collections::HashMap<u64, Box<WasmCaDispatchEntry>>>,
> = std::sync::Mutex::new(None);

/// Return the stable guest-memory address for `number`, creating a pending
/// (zero-slot) entry when needed.
pub fn ca_dispatch_slot(number: u64) -> u32 {
    let mut table = WASM_CA_DISPATCH.lock().unwrap();
    let entry = table
        .get_or_insert_with(Default::default)
        .entry(number)
        .or_insert_with(|| {
            Box::new(WasmCaDispatchEntry {
                func_handle: AtomicU32::new(0),
                loop_finish_fi: AtomicU32::new(WASM_CA_FINISH_FI_UNKNOWN),
                compiled_ptr: AtomicU32::new(0),
            })
        });
    (&**entry as *const WasmCaDispatchEntry as usize) as u32
}

pub fn ca_dispatch_exists(number: u64) -> bool {
    WASM_CA_DISPATCH
        .lock()
        .unwrap()
        .as_ref()
        .is_some_and(|table| table.contains_key(&number))
}

/// Publish an installed loop after its module has acquired a shared-table
/// slot. Release stores pair with the runtime loads in emitted trace modules;
/// wasm execution cannot begin until this compile call returns.
pub fn ca_dispatch_publish(number: u64, func_handle: u32, loop_finish_fi: u32, compiled_ptr: u32) {
    let _ = ca_dispatch_slot(number);
    let table = WASM_CA_DISPATCH.lock().unwrap();
    let entry = table
        .as_ref()
        .and_then(|table| table.get(&number))
        .expect("CALL_ASSEMBLER dispatch entry disappeared while publishing");
    entry.compiled_ptr.store(compiled_ptr, Ordering::Release);
    entry
        .loop_finish_fi
        .store(loop_finish_fi, Ordering::Release);
    entry.func_handle.store(func_handle, Ordering::Release);
}

/// Redirect existing callers of `old_number` to the installed target.
pub fn ca_dispatch_redirect(
    old_number: u64,
    func_handle: u32,
    loop_finish_fi: u32,
    compiled_ptr: u32,
) {
    ca_dispatch_publish(old_number, func_handle, loop_finish_fi, compiled_ptr);
}

/// Remove every dispatch entry that still resolves to `compiled_ptr`.  This
/// also retracts redirects into a dropped replacement loop, while preserving
/// an old token whose entry has already been redirected elsewhere.
pub fn ca_dispatch_remove_compiled_ptr(compiled_ptr: u32) {
    if let Some(table) = WASM_CA_DISPATCH.lock().unwrap().as_mut() {
        table.retain(|_, entry| entry.compiled_ptr.load(Ordering::Acquire) != compiled_ptr);
    }
}

pub fn ca_dispatch_remove(number: u64) {
    if let Some(table) = WASM_CA_DISPATCH.lock().unwrap().as_mut() {
        table.remove(&number);
    }
}

pub fn call_assembler_target(number: u64) -> Option<CallAssemblerTarget> {
    CALL_ASSEMBLER_TARGETS
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|targets| targets.get(&number).cloned())
}

pub fn publish_call_assembler_target(number: u64, target: CallAssemblerTarget) {
    CALL_ASSEMBLER_TARGETS
        .lock()
        .unwrap()
        .get_or_insert_with(Default::default)
        .insert(number, target);
}

/// Register the frontend's compile_tmp_callback placeholder.  The geometry is
/// deliberately zero: Stage-1 admission continues to require a fully live
/// target, while the stable dispatch entry lets a later self-recursive compile
/// use this token without baking a transient table slot.
pub fn register_pending_call_assembler_target(number: u64, input_types: Vec<Type>) {
    ca_dispatch_remove(number);
    let _ = ca_dispatch_slot(number);
    publish_call_assembler_target(
        number,
        CallAssemblerTarget {
            token_number: number,
            func_handle: 0,
            input_types,
            callee_frame_bytes: 0,
            callee_gcmap_ptr: 0,
            loop_finish_fi: WASM_CA_FINISH_FI_UNKNOWN,
            compiled_ptr: 0,
            has_trampoline_calls: false,
        },
    );
}

/// Remove metadata and the dispatch entry for an invalidated token.
pub fn remove_call_assembler_target(number: u64) {
    if let Some(targets) = CALL_ASSEMBLER_TARGETS.lock().unwrap().as_mut() {
        targets.remove(&number);
    }
    ca_dispatch_remove(number);
}

/// Retract all metadata aliases which point at a dropped compiled loop.
pub fn remove_call_assembler_targets_for_compiled_ptr(compiled_ptr: u32) {
    if let Some(targets) = CALL_ASSEMBLER_TARGETS.lock().unwrap().as_mut() {
        targets.retain(|_, target| target.compiled_ptr as u32 != compiled_ptr);
    }
    ca_dispatch_remove_compiled_ptr(compiled_ptr);
}

/// Global `frame[0]` fail-index space.
///
/// Cross-trace chaining (`LABEL_TARGETS`) means the module that last wrote
/// `frame[0]` is not necessarily the loop `execute_token` entered: a bridge's
/// terminal JUMP may tail-call a SIBLING loop, whose guards then write THEIR
/// exit indices. Per-loop index spaces would make those writes ambiguous at
/// the host — resolving `frame[0]` against the entry loop's own `fail_descrs`
/// picks a wrong descr (wrong arg types/resume ⇒ type confusion). So every
/// compile (`compile_loop` and `compile_bridge`) allocates its exits from this
/// one global space: it passes the registry length as codegen's
/// `fail_index_base`, guards write `base + local` into `frame[0]`, and the
/// registered descrs land at exactly those registry positions — any `frame[0]`
/// then resolves here regardless of which chained module wrote it. The
/// per-guard bridge-cell epilogue keeps its local indexing by subtracting the
/// owning module's base (`codegen`'s cell lookup).
///
/// Entries are never removed: a dropped loop's modules are unreachable (its
/// label targets are retracted and its token is gone), so its entries are just
/// retained memory, bounded by the total number of compiled exits.
static FAIL_DESCR_REGISTRY: std::sync::Mutex<Option<Vec<Arc<WasmFailDescr>>>> =
    std::sync::Mutex::new(None);

/// The next free global fail index — pass as `fail_index_base` to
/// `codegen::build_wasm_module`, then register the built descrs with
/// `register_fail_descrs`. The wasm host is single-threaded, so no other
/// compile can interleave between the two calls.
pub fn fail_descr_base() -> u32 {
    FAIL_DESCR_REGISTRY
        .lock()
        .unwrap()
        .as_ref()
        .map_or(0, |v| v.len() as u32)
}

/// Append a compile's exit descrs to the global space. Each descr's
/// `fail_index` (already base-offset by `build_wasm_module`) must equal the
/// registry position it lands at.
pub fn register_fail_descrs(descrs: &[Arc<WasmFailDescr>]) {
    let mut reg = FAIL_DESCR_REGISTRY.lock().unwrap();
    let vec = reg.get_or_insert_with(Default::default);
    for d in descrs {
        debug_assert_eq!(
            d.fail_index as usize,
            vec.len(),
            "fail descr registered out of lockstep with its global fail_index"
        );
        vec.push(Arc::clone(d));
    }
}

/// Resolve a `frame[0]` value through the global fail-index space.
pub fn global_fail_descr(fail_index: u32) -> Option<Arc<WasmFailDescr>> {
    FAIL_DESCR_REGISTRY
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|v| v.get(fail_index as usize).cloned())
}

/// Global `label descr identity → LabelTarget` registry (see `LabelTarget`).
/// The wasm host is single-threaded; the `Mutex` is for `static` soundness
/// only. `compile_loop` inserts every resumable label of a peeled loop;
/// `CompiledWasmLoop::drop` removes its own entries (guarded by
/// `func_handle`, so a recompile that re-stamped the same descr keeps the
/// replacement's entry).
pub static LABEL_TARGETS: std::sync::Mutex<Option<std::collections::HashMap<usize, LabelTarget>>> =
    std::sync::Mutex::new(None);

/// Look up a label target by descr identity.
pub fn label_target(descr_id: usize) -> Option<LabelTarget> {
    LABEL_TARGETS
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&descr_id).copied())
}

/// Publish a label target (see `LABEL_TARGETS`).
pub fn publish_label_target(descr_id: usize, target: LabelTarget) {
    LABEL_TARGETS
        .lock()
        .unwrap()
        .get_or_insert_with(Default::default)
        .insert(descr_id, target);
}

/// Guard-dispatch metadata of a bridge chained onto a loop, kept on the
/// source loop's `CompiledWasmLoop.chained_trace_meta` keyed by the bridge's
/// backend `trace_id`. Lets `compile_bridge` chain a NESTED sub-bridge onto a
/// guard that lives inside an already-chained bridge: the failing guard's
/// meta descr carries `(trace_id, per-trace fail_index)`, and this record
/// supplies the owning bridge's cell array and livelock advance flags — the
/// same data `CompiledWasmLoop` holds for the loop's own guards.
pub struct ChainedTraceMeta {
    /// Base address of the bridge's per-guard bridge-slot cell array
    /// (`CompiledWasmLoop::bridge_cells_base` analog); `0` = no dispatch.
    pub cells_base: u32,
    /// Cell count = the bridge's own guard count.
    pub num_cells: usize,
    /// Per-guard, per-fail-arg induction-advance flags
    /// (`CompiledWasmLoop::guard_fail_arg_advanced` analog).
    pub guard_fail_arg_advanced: Vec<Vec<bool>>,
}

/// Compiled wasm loop metadata, stored in `JitCellToken.compiled`.
pub struct CompiledWasmLoop {
    /// Owning `JitCellToken` number, used to retract this loop's
    /// CALL_ASSEMBLER target metadata on drop.
    pub token_number: u64,
    pub trace_id: u64,
    pub input_types: Vec<Type>,
    pub func_handle: u32,
    /// This loop's own guard/finish exit descriptors (positions `[0,
    /// num_guard_cells)`, per-trace order), followed by the descr slices of
    /// every chained bridge `compile_bridge` appended (positional bookkeeping
    /// for `bridge_descr_ranges` — layouts and jitcounter hashes). `frame[0]`
    /// exit resolution does NOT index this vec: exit indices live in the
    /// GLOBAL fail-index space (`register_fail_descrs`), because a cross-trace
    /// chain can exit through a sibling loop's guard. `RefCell` because the
    /// append happens through the shared `&JitCellToken` the bridge attaches
    /// to; the wasm host is single-threaded so no cross-thread access occurs.
    pub fail_descrs: RefCell<Vec<Arc<WasmFailDescr>>>,
    pub num_inputs: usize,
    pub max_output_slots: usize,
    /// Number of Ref-typed values given a home slot in the frame's Ref-home
    /// region (`codegen::HOME_SLOT_BASE`). `execute_token` sizes the host
    /// frame to include this region and registers each home slot as a GC root.
    pub num_ref_homes: usize,
    /// Geometry frozen when this token was first compiled. Every bridge
    /// chained onto it is emitted against this exact layout.
    pub frame: crate::codegen::FrameGeometry,
    /// True when this loop or any successfully chained bridge uses the host
    /// residual-call trampoline. A CA callee frame is movable, but that
    /// trampoline retains the pre-call frame pointer, so `compile_bridge` must
    /// not enable the CA arm for this source token.
    pub has_trampoline_calls: Cell<bool>,
    /// Base address (shared linear memory) of this loop's per-guard bridge-slot
    /// cell array — one i32 per `fail_index`, `0` = no bridge. The trace's
    /// epilogue reads `cells[fail_index]` and `compile_bridge` writes a bridge's
    /// table slot here. `0` when the trace has no in-module dispatch (native, or
    /// a guardless / straight-line trace).
    pub bridge_cells_base: u32,
    /// Number of cells in the `bridge_cells_base` array = this loop's own guard
    /// count at compile time. A bridge attaches only to one of these original
    /// guards (`source_fail_index < num_guard_cells`); descrs appended past this
    /// range belong to already-chained bridges and have no cell of their own.
    pub num_guard_cells: usize,
    /// True when this is a peeled loop (`codegen::is_resumable_peeled`) — there
    /// is real work (a preamble = the unrolled first iteration) before the last
    /// `LABEL`, single- or multi-label. Such a loop carries the resume-at-LABEL
    /// entry `br_table` (key = label ordinal + 1) so a loop-closing bridge can
    /// re-enter at any of its labels. A loop-closing bridge re-enters through
    /// the loop's table slot (the function entry); for a peeled loop,
    /// re-running the preamble against mid-loop state would never advance the
    /// induction variable — an infinite loop. `compile_bridge` therefore
    /// declines a loop-closing bridge UNLESS its JUMP's
    /// target label resolves to a published, resumable `LabelTarget`.
    pub has_preamble: bool,
    /// Descr identity (`Arc::as_ptr`) of each `LABEL`, in ordinal order; `0`
    /// for a descr-less label. `compile_bridge` resolves a closing JUMP's
    /// target label by matching its descr identity against this list — a JUMP
    /// whose descr is not here targets ANOTHER trace's label (e.g. a sibling
    /// retrace specialization, whose start label carries the same stamped
    /// ordinal) and must not be chained into this loop.
    pub label_descrs: Vec<usize>,
    /// Per-guard (indexed by this loop's own `fail_index`), per-fail-arg:
    /// whether the value was produced by induction-advancing arithmetic after
    /// the loop-header label — fresh in the failing iteration. Consulted by
    /// `compile_bridge`'s livelock check: a loop-closing bridge that JUMPs
    /// such a fail arg verbatim still advances the chained cycle.
    pub guard_fail_arg_advanced: Vec<Vec<bool>>,
    /// `(source_trace_id, source_fail_index, start, count)` ranges into
    /// `fail_descrs` for each chained bridge `compile_bridge` appended (lib.rs
    /// extend site). Lets `compiled_bridge_fail_descr_layouts` /
    /// `store_bridge_guard_hashes` map a source guard back to its bridge's
    /// appended descr slice — the wasm analog of dynasm's
    /// `lookup_bridge_addr` (runner.rs). Keyed by BOTH the source guard's
    /// owning trace and its per-trace fail index: with nested chaining, the
    /// loop's guard `k` and a chained bridge's guard `k` are distinct sources.
    /// Recorded in lockstep with the `extend`, inside the same `borrow_mut`
    /// critical section.
    pub bridge_descr_ranges: RefCell<Vec<(u64, u32, usize, usize)>>,
    /// Guard-dispatch metadata of every bridge chained onto this loop, keyed
    /// by the bridge's backend `trace_id` (see [`ChainedTraceMeta`]). Lets a
    /// guard INSIDE a chained bridge chain its own nested sub-bridge.
    pub chained_trace_meta: RefCell<std::collections::HashMap<u64, ChainedTraceMeta>>,
    /// Owns this loop's per-guard bridge-slot cell array so it is freed on
    /// `Drop`; `bridge_cells_base` aliases its heap address (stable across the
    /// struct move). `None` when the trace has no in-module dispatch.
    pub _bridge_cells_owner: Option<Box<[u32]>>,
    /// Owns the cell arrays of every bridge chained onto this loop. A bridge
    /// module lives as long as the source loop it attaches to, so its cells are
    /// freed when this loop drops. Appended by `compile_bridge`.
    pub _bridge_owned_cells: RefCell<Vec<Box<[u32]>>>,
    /// Set when `compile_bridge` accepts a self-recursive `CallAssemblerR`
    /// bridge (`PYRE_WASM_CA`) for this loop. While set, `compile_bridge`
    /// declines chaining any FURTHER bridge into this recursion (the guard
    /// falls back to host round-trips): a chained bridge deopting inside the
    /// CA recursion trips a resume seam that reads a clobbered class — see
    /// the decline site for the failing suite shapes.
    pub ca_active: Cell<bool>,
    /// A guard reached through this loop as a wasm CALL_ASSEMBLER callee was
    /// structurally declined by `compile_bridge`.  Admission refuses this
    /// target, because entering it from compiled wasm would only blackhole.
    pub ca_terminal_declined: Cell<bool>,
    /// Compiled callers that baked this loop as their CALL_ASSEMBLER target.
    /// A terminal callee decline invalidates them for a no-CA retrace.
    pub ca_callers: RefCell<Vec<std::sync::Arc<std::sync::atomic::AtomicBool>>>,
}

impl CompiledWasmLoop {
    /// Incorporate the normal (non-CA unless this bridge is the candidate)
    /// codegen census for a bridge after it has been chained onto this token.
    /// Every earlier bridge remains reachable from a later CA recursion's
    /// guard exits, so its host trampoline use also rules out CA.
    pub fn record_chained_bridge_trampoline_calls(&self, bridge_has_trampoline_calls: bool) {
        self.has_trampoline_calls
            .set(self.has_trampoline_calls.get() || bridge_has_trampoline_calls);
    }
}

impl Drop for CompiledWasmLoop {
    fn drop(&mut self) {
        // Remove every token alias still targeting this module, including a
        // redirect source. A source redirected to a newer module survives an
        // old-loop drop because its dispatch `compiled_ptr` no longer matches.
        remove_call_assembler_targets_for_compiled_ptr(self as *const Self as usize as u32);
        // Retract this loop's published label targets so a later bridge
        // cannot chain into a dropped loop's stale table slot. Guarded by
        // `func_handle`: a recompile that re-stamped the same descr onto its
        // replacement loop has already overwritten the entry, which must
        // survive the old loop's drop.
        let mut reg = LABEL_TARGETS.lock().unwrap();
        if let Some(map) = reg.as_mut() {
            for &id in &self.label_descrs {
                if id != 0 {
                    if let Some(t) = map.get(&id) {
                        if t.func_handle == self.func_handle {
                            map.remove(&id);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_with_trampoline_census(has_trampoline_calls: bool) -> CompiledWasmLoop {
        CompiledWasmLoop {
            token_number: 0,
            trace_id: 0,
            input_types: Vec::new(),
            func_handle: 0,
            fail_descrs: RefCell::new(Vec::new()),
            num_inputs: 0,
            max_output_slots: 0,
            num_ref_homes: 0,
            frame: crate::codegen::FrameGeometry::fixed(),
            has_trampoline_calls: Cell::new(has_trampoline_calls),
            bridge_cells_base: 0,
            num_guard_cells: 0,
            has_preamble: false,
            label_descrs: Vec::new(),
            guard_fail_arg_advanced: Vec::new(),
            bridge_descr_ranges: RefCell::new(Vec::new()),
            chained_trace_meta: RefCell::new(std::collections::HashMap::new()),
            _bridge_cells_owner: None,
            _bridge_owned_cells: RefCell::new(Vec::new()),
            ca_active: Cell::new(false),
            ca_terminal_declined: Cell::new(false),
            ca_callers: RefCell::new(Vec::new()),
        }
    }

    #[test]
    fn chained_bridge_trampoline_census_is_orred_into_token() {
        let token = token_with_trampoline_census(false);
        token.record_chained_bridge_trampoline_calls(false);
        assert!(!token.has_trampoline_calls.get());

        token.record_chained_bridge_trampoline_calls(true);
        assert!(token.has_trampoline_calls.get());

        // A later clean bridge cannot erase an earlier chained bridge's
        // trampoline census before a CA bridge is considered.
        token.record_chained_bridge_trampoline_calls(false);
        assert!(token.has_trampoline_calls.get());
    }
}
