/// WebAssembly backend for majit.
///
/// Generates wasm bytecodes via wasm-encoder. On wasm32 targets,
/// instantiates the emitted trace modules through a host binding (see
/// `glue`): the `web` feature uses the browser `WebAssembly` API via
/// wasm-bindgen, the `host-import` feature uses plain wasm imports that a
/// native embedder (wasmi / wasmtime) supplies. On native targets,
/// compile_loop succeeds but execute_token requires a wasm host
/// (unreachable natively).
///
/// Which binding a build has is a feature, not a target OS: `host-import` is
/// satisfied by an embedder, and an embedder is exactly what a WASI command
/// runs under. So the split below is wasm32 against native, and a wasm32 build
/// with no binding selected keeps `glue`'s stubs rather than being routed to
/// the native arm.
pub mod codegen;
pub mod failguard;

#[cfg(target_arch = "wasm32")]
mod glue;

use parking_lot::Mutex;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};

/// Diagnostic-only `compile_bridge` outcome tallies, read out via the
/// `pyre_jit_bridge_diag` guest export (the runner prints them at
/// `PYRE_WASM_JIT_STATS` time). A static counter — NOT a host import, which
/// would shift the wasm function-index space and break the JIT's baked
/// `fn as usize` table indices. Index legend: 0 = compile_bridge entered,
/// 1 = declined CALL_ASSEMBLER, 2 = declined multi-label peeled,
/// 3 = declined not-a-direct-loop-guard, 4 = declined ref-home overflow,
/// 5 = bridge compiled (chained in-module), 6 = loop-closing shape seen,
/// 7 = source loop has a preamble. Sub-breakdown of the index-2 multi-label
/// decline (TEMP, for the resume-at-last-label measurement): 8 = JUMP descr
/// did not resolve (target_ord None), 9 = target_ord Some but != last label,
/// 10 = arity mismatch, 11 = loop-closing bridge advances no loop-carried value
/// (guard side-trace that would livelock the chained loop), 14 = accepted
/// CALL_ASSEMBLER trace, 15 = declined CA because a trace would use the host
/// call trampoline on a movable CA frame.  Index 16 records the dormant
/// forced-terminal-decline runtime regression hook. Sub-breakdown of the
/// index-8 unresolved-target decline: 17 = the terminal JUMP carries no descr
/// at all, 18 = the descr is present but `LABEL_TARGETS` holds no entry for it.
/// Publish-side counterpart, so an unresolved lookup can be told from a label
/// that was never offered: 19 = labels published off a peeled trace, 20 =
/// published off a non-peeled trace, 21 = a non-peeled trace's first label left
/// unpublished (no descr, or its arity is not the inputarg count), 22 = a
/// dropped loop retracted a published entry. 19-21 count loops and
/// LABEL-bearing bridges alike, since both go through the same publish step
/// (`x86/assembler.py fixup_target_tokens` runs on either path); a bridge
/// with no LABEL is not tallied by 21. `compile_loop`'s own outcome
/// split — every `Err` it returns is a `loops_aborted` bump in the metainterp,
/// and the reason string never reaches the host from inside the guest, so the
/// classification has to be a counter: 23 = compile_loop entered, 24 =
/// compile_loop returned a token, 25 = declined by
/// `wasm_unsupported_trace_reason` (the #62 loop-callee CALL_ASSEMBLER gap),
/// 26 = the wasm host rejected the emitted module (`func_handle == 0`).
/// Indices 2 and 4 double as compile_loop's other two declines.
///
/// 27-28 ask 12-13's question of EVERY accepted bridge rather than only the
/// CALL_ASSEMBLER ones: 27 = the source guard's dispatch cell was written, so
/// the loop epilogue now tail-calls this bridge in-module; 28 = it was not,
/// because the owning trace reserved no cell array, so the guard keeps
/// round-tripping to the host and the bridge is compiled but unreachable.
/// `BRIDGE_OK` (5) only says the backend accepted a bridge, which is strictly
/// weaker than "the guard can reach it"; without this split the two are
/// indistinguishable from outside the guest.
///
/// 29 = the cell written at 27 was ALREADY non-zero, i.e. this guard had a
/// reachable bridge and kept failing anyway. One of those is ordinary (a guard
/// re-bridged after its first bridge was outgrown); a count that tracks
/// `BRIDGE_OK` says the epilogue dispatch is not taking the cell at all and
/// every bridge after the first is dead weight.
/// 30 = a host-armed loop re-emission was attempted but failed;
/// 31 = it succeeded and the rebuilt module is installed in the loop's
/// original table slot. 31 is the only positive evidence that a re-emission
/// ran at all: a re-emission that silently never fires is indistinguishable
/// from one that fires and changes nothing.
/// 32 = a loop-closing bridge region was inlined; 33 = inlining declined
/// because the source guard belongs to an already chained trace; 34 = the
/// bridge is not loop-closing; 35 = the owner has no retained module inputs;
/// 36 = that guard already owns a region; 37 = the merged stream exceeds the
/// owner's frozen frame geometry; 38 = the bridge does not resume at the loop
/// header; 39 = the merged stream has no local loop LABEL for the wasm back
/// edge. 40-43 split a rejected inline trial into value-layout,
/// Ref-home-layout, missing-local-label, and other backend errors. 44 = a
/// bridge compiled with a parameter entry; 45 = parameter entry declined
/// because the source module has frame-only dispatch; 46 = parameter entry
/// declined because the source guard and bridge input arities disagree; 47 =
/// LABEL publication suppressed because the bridge entry has nonzero parameters.
/// 48 = an inline trial's LABEL-resume storage exceeds the frozen frame; 49 =
/// the region carries a CALL_ASSEMBLER the owner build emits no arm for; 50 =
/// the owner is already invalidated, so a merged region would inherit its set
/// flag instead of starting valid; 51 = a region retained for a deferred merge
/// whose closing JUMP names a LABEL published by another module, so it keeps
/// the cross-module tail call and merges only its entry side; 52 = the
/// region's source guard is in the peeled preamble, outside the `loop` its
/// block is opened in; 53 = eligible but no trip callback is published to
/// defer to; 54 = eligible, merge deferred until the bridge standing in for it
/// has been entered `INLINE_TRIP_THRESHOLD` times; 55 = that trip fired and the
/// merge was attempted; 56 = eligible but not deferrable, because the region
/// carries a `GUARD_NOT_INVALIDATED` whose dependencies would outlive the flag
/// it reads.
///
/// 57-63 split slot 1, which says only that some CALL_ASSEMBLER target did not
/// resolve and leaves the trace unsupported. Each answers one of the questions
/// `general_int_call_assembler_target` asks before it admits a target: 57 = the
/// operation is a CALL_ASSEMBLER whose result is neither Int nor Ref; 58 = it
/// carries no call descr, or no target token; 59 = an argument or result type
/// is outside Int/Ref; 60 = the target token is not in the registry at all;
/// 61 = a registered target's deferred module failed to materialize a func
/// handle; 62 = the registered geometry disagrees with the operation (input
/// types, zero callee frame bytes, absent gcmap, absent compiled loop); 63 =
/// the target compiled once and has since declined terminally. A decline that
/// falls in 60-63 names a target that exists, which is the half of slot 1 a
/// retrace could plausibly resolve; 57-59 name the operation itself.
///
/// 64 = a cross-module region declined the EAGER merge arm. That arm forgoes
/// the trip threshold to keep a quasi-immutable dependency attached to the
/// owner's flag, and pays an owner re-emission whether or not the region ever
/// runs hot; a region that saves only its entry crossing does not earn it.
///
/// 65 = the same eager arm declined an owner already too large to re-emit. The
/// arm cannot wait for the entry evidence the deferred arm waits for, so the
/// only thing it can read is what the re-emission will cost.
pub static BRIDGE_DIAG: [AtomicU64; BRIDGE_DIAG_LABELS.len()] =
    [const { AtomicU64::new(0) }; BRIDGE_DIAG_LABELS.len()];

/// Short key per [`BRIDGE_DIAG`] slot, in index order, spelling the legend
/// above as something a reader can join against.
///
/// This array is the slot count — [`BRIDGE_DIAG`] takes its length from it —
/// so a tally cannot be added without naming it. The wasm host mirrors these
/// keys positionally (it links no majit crate) and prints them under
/// `[jit-stats] bridge_diag`; without a declaration to compare against, a slot
/// bumped here and unnamed there is simply never reported.
///
/// A few slots are legend entries that no site bumps today (a decline that was
/// split finer, and the two `ml_*` sub-breakdowns). They keep their names so
/// the indices below them do not move.
pub const BRIDGE_DIAG_LABELS: &[&str] = &[
    "entered",
    "decl_callasm",
    "decl_multipeel",
    "decl_notdirect",
    "decl_refhome",
    "BRIDGE_OK",
    "loopclosing",
    "src_preamble",
    "ml_descr_none",
    "ml_unsafe_label",
    "ml_arity_mismatch",
    "decl_noadvance",
    "ca_cell_set",
    "ca_cells_zero",
    "accepted_ca",
    "decl_ca_trampoline",
    "forced_ca_terminal_decline",
    "ml_no_descr",
    "ml_unpublished",
    "pub_peeled",
    "pub_flat",
    "pub_flat_skipped",
    "label_retracted",
    "cl_entered",
    "cl_ok",
    "cl_decl_unsupported",
    "cl_decl_host_reject",
    "cell_set",
    "cell_missing",
    "cell_rebridge",
    "reemit_failed",
    "reemit_ok",
    "inline_ok",
    "inline_decl_not_direct",
    "inline_decl_not_loop_closing",
    "inline_decl_not_reemittable",
    "inline_decl_already_owned",
    "inline_decl_frame",
    "inline_decl_not_header",
    "inline_decl_no_loop_label",
    "inline_decl_value_layout",
    "inline_decl_ref_layout",
    "inline_decl_missing_label",
    "inline_decl_other",
    "bridge_param_ok",
    "bridge_param_decl_source_frame",
    "bridge_param_decl_arity",
    "bridge_param_label_suppressed",
    "inline_decl_label_resume_layout",
    "inline_decl_call_assembler",
    "inline_decl_owner_invalidated",
    "inline_foreign_jump",
    "inline_ok_outside_loop",
    "inline_decl_no_trip_helper",
    "inline_deferred",
    "inline_trip_fired",
    "inline_decl_defer_invalidation_guard",
    "ca_decl_opcode",
    "ca_decl_descr",
    "ca_decl_types",
    "ca_decl_unregistered",
    "ca_decl_materialize",
    "ca_decl_geometry",
    "ca_decl_terminal",
    "inline_decl_foreign_eager",
    "inline_decl_eager_too_large",
];

#[repr(u8)]
#[derive(Clone, Copy)]
pub(crate) enum FrameShortageKind {
    FrameValueSlots = 1,
    OrdinaryRefHomes = 2,
    LabelResumeRefSlots = 3,
    LabelResumeCaptureSlots = 4,
}

#[derive(Clone, Copy)]
pub(crate) struct FrameShortage {
    pub(crate) kind: FrameShortageKind,
    pub(crate) needed: usize,
    pub(crate) available: usize,
}

impl FrameShortage {
    pub(crate) const fn new(kind: FrameShortageKind, needed: usize, available: usize) -> Self {
        Self {
            kind,
            needed,
            available,
        }
    }
}

/// The first three inline geometry failures, packed as
/// `(kind: u8, needed: u24, available: u24)`. They expose a frozen-layout
/// shortage without changing the compile result.
static INLINE_GEOMETRY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static INLINE_GEOMETRY_COUNT: AtomicU64 = AtomicU64::new(0);
/// The first three reasons an inline-bridge install was refused, verbatim.
/// The names carry "trial" because they are a guest export the runner looks up
/// by string; the errors themselves come from the install itself, which is the
/// only build there is.
static INLINE_TRIAL_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());
/// Why each loop-closing bridge was refused a merge into its owner, capped so
/// a long run cannot grow the log without bound. `bridge_diag`'s counters say
/// how many declines each reason took; these records carry the keys that say
/// which ones matter — `(slot, key)` joins a record against the trace-entry
/// census, whose `entries` count is how often that crossing actually ran.
static INLINE_DECLINES: Mutex<Vec<String>> = Mutex::new(Vec::new());
const INLINE_DECLINE_LOG_CAP: usize = 64;

pub(crate) fn record_inline_geometry(kind: FrameShortageKind, needed: usize, available: usize) {
    const FIELD_MASK: u64 = (1 << 24) - 1;

    let index = INLINE_GEOMETRY_COUNT.fetch_add(1, Ordering::Relaxed) as usize;
    if let Some(slot) = INLINE_GEOMETRY.get(index) {
        slot.store(
            ((kind as u64) << 48)
                | ((needed as u64).min(FIELD_MASK) << 24)
                | (available as u64).min(FIELD_MASK),
            Ordering::Relaxed,
        );
    }
}

/// Read a packed `(kind, needed, available)` inline geometry failure.
pub fn inline_geometry_diag(index: usize) -> u64 {
    INLINE_GEOMETRY
        .get(index)
        .map_or(0, |slot| slot.load(Ordering::Relaxed))
}

/// Number of inline geometry failures, including records beyond the three
/// diagnostics retained in [`INLINE_GEOMETRY`].
pub fn inline_geometry_count() -> u64 {
    INLINE_GEOMETRY_COUNT.load(Ordering::Relaxed)
}

pub fn inline_trial_errors() -> String {
    INLINE_TRIAL_ERRORS.lock().join(" | ")
}

pub(crate) fn record_inline_decline(record: String) {
    let mut log = INLINE_DECLINES.lock();
    if log.len() < INLINE_DECLINE_LOG_CAP {
        log.push(record);
    }
}

pub fn inline_declines() -> String {
    INLINE_DECLINES.lock().join(" | ")
}

fn record_inline_trial_error(error: &BackendError) {
    let mut errors = INLINE_TRIAL_ERRORS.lock();
    if errors.len() < 3 {
        errors.push(error.to_string());
    }
}

/// Sort a refused inline install into the decline tallies the host prints.
/// `replace_module` rejecting the bytes, or a build with no host binding to
/// replace them through, is a re-emission outcome and stays on its own counter;
/// every other reason is the merged module declining to emit, which is what the
/// per-shortage buckets are for.
fn classify_inline_install_error(error: &BackendError) {
    let BackendError::Unsupported(reason) = error else {
        diag_bump(37);
        diag_bump(43);
        return;
    };
    if reason.contains("wasm host rejected the re-emitted trace module")
        || reason.contains("no host replacement binding")
    {
        diag_bump(30);
        return;
    }
    diag_bump(37);
    if reason.contains("frame value slots exceed frozen frame layout") {
        diag_bump(40);
    } else if reason.contains("ordinary ref homes") {
        diag_bump(41);
    } else if reason.contains("label resume layout") {
        diag_bump(48);
    } else if reason.contains("no CALL_ASSEMBLER arm for") {
        diag_bump(49);
    } else if reason.contains("inlined bridge stream has no local loop LABEL") {
        diag_bump(42);
    } else {
        diag_bump(43);
    }
}

static REEMIT_ENABLED: AtomicBool = AtomicBool::new(false);
static INLINE_BRIDGE_ENABLED: AtomicBool = AtomicBool::new(true);
/// Off for the loop-body half of the class. See `inline_nonheader_enable`.
static INLINE_NONHEADER_ENABLED: AtomicBool = AtomicBool::new(false);
static BRIDGE_PARAMS_ENABLED: AtomicBool = AtomicBool::new(true);
/// Entries a merge must earn per byte of the module it re-emits. See
/// `inline_trip_threshold_for`. Zero leaves `INLINE_TRIP_THRESHOLD` as the
/// whole rule.
static INLINE_TRIP_BYTES_FACTOR: AtomicU64 = AtomicU64::new(DEFAULT_INLINE_TRIP_BYTES_FACTOR);
/// Owner size at which the eager merge arm stops merging. See
/// `DEFAULT_INLINE_EAGER_MAX_BYTES`.
static INLINE_EAGER_MAX_BYTES: AtomicU32 = AtomicU32::new(DEFAULT_INLINE_EAGER_MAX_BYTES);
static TRACE_ENTRY_CENSUS_FORCED: AtomicBool = AtomicBool::new(false);

/// One compiled trace's guest-memory entry counters.  The generated module
/// updates `counts[key]` directly, so this owner must outlive every module
/// that bakes its base address.
struct TraceEntryCensus {
    trace_id: u64,
    counts: Box<[u64]>,
}

/// The census deliberately has no per-entry Rust callback: a module writes
/// this guest-memory storage itself.  The runner reads it only after Python
/// exits, when no trace is executing.
static TRACE_ENTRY_CENSUS: Mutex<Vec<TraceEntryCensus>> = Mutex::new(Vec::new());

/// Baked into an armed module. `trace_id` is the backend's monotonic trace id,
/// which stays attached to a loop when its module is re-emitted.
#[derive(Clone, Copy)]
pub struct TraceEntryCensusStorage {
    pub trace_id: u64,
    pub base: u32,
    pub key_count: u32,
}

/// Arm trace-entry instrumentation before the guest starts compiling traces.
/// `MAJIT_TRACE_ENTRY_CENSUS` selects the same facility wherever the guest has
/// an environment to read it from; a guest that has none is armed by its host
/// through this function instead.
pub fn trace_entry_census_enable() {
    TRACE_ENTRY_CENSUS_FORCED.store(true, Ordering::Relaxed);
}

fn trace_entry_census_enabled() -> bool {
    if TRACE_ENTRY_CENSUS_FORCED.load(Ordering::Relaxed) {
        return true;
    }
    // Read on every target. Whether a wasm guest has an environment is a
    // property of its embedder, not of the architecture: one launched as a
    // WASI command inherits the variables its host passes it, and one with no
    // environment reads an absent variable rather than failing to compile.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MAJIT_TRACE_ENTRY_CENSUS").is_some())
}

/// Allocate the one counter array that an armed physical trace module uses.
/// Re-emission clones the stored descriptor, preserving both the trace id and
/// the counters rather than assigning the replacement a second identity.
fn alloc_trace_entry_census(trace_id: u64, key_count: usize) -> Option<TraceEntryCensusStorage> {
    if !trace_entry_census_enabled() {
        return None;
    }
    #[cfg(target_arch = "wasm32")]
    {
        let mut counts = vec![0u64; key_count].into_boxed_slice();
        let base = counts.as_mut_ptr() as usize as u32;
        TRACE_ENTRY_CENSUS
            .lock()
            .push(TraceEntryCensus { trace_id, counts });
        Some(TraceEntryCensusStorage {
            trace_id,
            base,
            key_count: key_count as u32,
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (trace_id, key_count);
        None
    }
}

/// Greppable, stable host readout of the guest-written entry counters.
pub fn trace_entry_census_summary() -> String {
    let census = TRACE_ENTRY_CENSUS.lock();
    let mut total = 0u64;
    let mut report = String::new();
    for trace in census.iter() {
        for (key, count) in trace.counts.iter().enumerate() {
            // Trace modules update this memory directly, outside Rust's alias
            // analysis; volatile makes the post-run host read explicit.
            let count = unsafe { core::ptr::read_volatile(count) };
            if count != 0 {
                total = total.saturating_add(count);
                report.push_str(&format!(
                    "[trace-entry-census] trace_id={} key={key} entries={count}\n",
                    trace.trace_id
                ));
            }
        }
    }
    report.push_str(&format!("[trace-entry-census] total={total}\n"));
    report
}

/// Arm loop-module replacement from the host before guest execution starts.
pub fn reemit_enable() {
    REEMIT_ENABLED.store(true, Ordering::Relaxed);
}

fn reemit_enabled() -> bool {
    REEMIT_ENABLED.load(Ordering::Relaxed)
}

/// Disable loop-closing bridge inlining from the host before guest execution
/// starts. A bridge that closes back onto its owner's loop is merged into the
/// owner's module by default, so the guard reaching it becomes a branch inside
/// one module instead of a call out to another; this carries the host's
/// explicit opt-out into the backend.
pub fn inline_bridge_disable() {
    INLINE_BRIDGE_ENABLED.store(false, Ordering::Relaxed);
}

fn inline_bridge_enabled() -> bool {
    INLINE_BRIDGE_ENABLED.load(Ordering::Relaxed)
}

/// Admit a region whose closing JUMP names a resumable LABEL that is not the
/// loop header AND whose guard sits inside the loop body. Both halves of that
/// class re-enter the same way — `codegen` wraps the entry dispatch in a `loop`
/// the region branches back into, landing past the named label's resume loader
/// with the values already in locals — but they are placed differently and they
/// measure differently, so only this half is behind the flag.
///
/// A region attached to a PREAMBLE guard is admitted unconditionally: the
/// `loop` holding the body regions' blocks has not been entered there, so
/// `build_function` opens its blocks outside that loop and emits its body past
/// the loop's `end`. On `str_getitem_len_hot`, whose bytes and bytearray legs
/// fail a peeled-preamble GuardClass on every iteration, that removes 72.0M of
/// 120.0M cross-module crossings and takes exec from 0.985s to 0.716s — 0.73x,
/// min of 15 interleaved runs with each arm's startup floor subtracted.
/// `spectral_norm` measures 0.95x and `fannkuch` 0.98x on the same change.
///
/// ⛔ The loop-body half stays off, on wall time rather than on correctness.
/// Admitting it declines nothing on 81 corpus fixtures and removes 49.4M of
/// their 257.3M crossings — 19.2%, 41 fixtures moved, none the wrong way — and
/// buys 0.74x on `short_circuit_value_local_kept` and 0.67x on
/// `short_circuit_boxed_int_cross_fn`. It still costs 1.23x on `spectral_norm`,
/// which sheds 99.7% of its own crossings and gets slower anyway, because
/// admitting a region costs the owner a re-emission and taxes its fall-through
/// path 18 ops on every iteration that does NOT fail the guard. A preamble
/// guard's region is not on that fall-through, which is why the two halves
/// separate.
pub fn inline_nonheader_enable() {
    INLINE_NONHEADER_ENABLED.store(true, Ordering::Relaxed);
}

fn inline_nonheader_enabled() -> bool {
    INLINE_NONHEADER_ENABLED.load(Ordering::Relaxed)
}

/// Set the per-byte entry price a deferred merge must earn, from the host
/// before guest execution, in place of [`DEFAULT_INLINE_TRIP_BYTES_FACTOR`].
/// Zero leaves [`INLINE_TRIP_THRESHOLD`] as the whole rule.
pub fn set_inline_trip_bytes_factor(entries_per_byte: u64) {
    INLINE_TRIP_BYTES_FACTOR.store(entries_per_byte, Ordering::Relaxed);
}

/// The wasm loop `token` was last compiled as, when it has one. Both merge
/// arms price themselves off its `module_bytes`.
fn compiled_wasm_loop(token: &JitCellToken) -> Option<&CompiledWasmLoop> {
    token
        .compiled
        .get()
        .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
}

/// Set the owner size at which the eager merge arm declines, from the host
/// before guest execution, in place of [`DEFAULT_INLINE_EAGER_MAX_BYTES`].
pub fn set_inline_eager_max_bytes(max_bytes: u32) {
    INLINE_EAGER_MAX_BYTES.store(max_bytes, Ordering::Relaxed);
}

/// Entries the bridge standing in for a merge must be entered before the merge
/// is taken, for an owner whose last emission was `owner_module_bytes` long.
///
/// A merge re-emits the whole owner, so its cost scales with the owner's size
/// rather than the region's: cranelift charges about 0.65 ms per KB of module,
/// while a cross-module crossing the merge removes is about 2.5 ns. Those two
/// rates are what [`INLINE_TRIP_BYTES_FACTOR`] converts between; a bridge's
/// entry count is a floor on the crossings removed, so the price is a floor
/// too.
///
/// [`INLINE_TRIP_THRESHOLD`] stays as the lower bound, because a fixture whose
/// entire crossing budget is under a millisecond cannot pay back any rebuild.
///
/// Above the price at which a merge is refused outright sits a band where the
/// price only postpones a merge that is taken anyway, and every crossing in
/// that window is paid for nothing. The default sits below that band.
fn inline_trip_threshold_for(owner_module_bytes: u32) -> u64 {
    let priced = INLINE_TRIP_BYTES_FACTOR
        .load(Ordering::Relaxed)
        .saturating_mul(owner_module_bytes as u64);
    priced.max(INLINE_TRIP_THRESHOLD)
}

/// Disable guard-to-bridge value parameters from the host before guest
/// execution. By default, a generated guard keeps the ordinary frame recovery
/// state for the uncompiled case, then passes its live failure values directly
/// once a bridge table slot is present.
pub fn bridge_params_disable() {
    BRIDGE_PARAMS_ENABLED.store(false, Ordering::Relaxed);
}

fn bridge_params_enabled() -> bool {
    BRIDGE_PARAMS_ENABLED.load(Ordering::Relaxed)
}

/// Read a `BRIDGE_DIAG` tally (saturating index). Surfaced to the host through
/// the `pyre_jit_bridge_diag` export in the `pyre-wasm` crate.
pub fn bridge_diag(i: usize) -> u64 {
    BRIDGE_DIAG
        .get(i)
        .map(|c| c.load(Ordering::Relaxed))
        .unwrap_or(0)
}

/// Number of JIT trace entries made from the guest.
#[cfg(target_arch = "wasm32")]
pub fn jit_execute_count() -> u64 {
    glue::jit_execute_count()
}

/// Number of host modules materialized after the lazy-install gate.
#[cfg(target_arch = "wasm32")]
pub fn jit_compile_count() -> u64 {
    glue::jit_compile_count()
}

/// Number of materializations served by the byte-identical module cache.
#[cfg(target_arch = "wasm32")]
pub fn jit_compile_cache_hits() -> u64 {
    glue::jit_compile_cache_hits()
}

#[inline]
fn diag_bump(i: usize) {
    BRIDGE_DIAG[i].fetch_add(1, Ordering::Relaxed);
}

// A source token is compiled before a later guard may become a CA bridge.
// Freeze modest room for that bridge at first compilation; a later trace that
// exceeds either bound is declined rather than changing the live frame's
// offsets. The recursive-unroll fib CA bridge needs more than 64 Ref homes:
// declining it leaves the recursive return guard permanently blackholed and
// turns every later invocation into a host round-trip. Keep enough bounded
// per-token reserve for that bridge and the existing full-suite shapes.
// A CALL_ASSEMBLER target must retain enough frozen spill/home geometry for a
// later exit bridge.  nbody's callee bridge needs more Ref homes than the old
// 16-slot floor; declining it turns every CA invocation into a blackhole.  The
// larger fixed reserve keeps the bridge in compiled wasm and is still bounded
// per compiled token.
const FROZEN_CHAIN_VALUE_SLOTS: usize = 64;
const FROZEN_CHAIN_REF_HOMES: usize = 128;
const FROZEN_CHAIN_LABEL_REF_SLOTS: usize = 2;
/// Slots a frozen layout is rounded up to.
///
/// A chained bridge runs in its source token's frame and reaches its target's
/// label loader, so the two layouts have to agree offset for offset — the
/// chain is refused outright when they do not. Above the floors the two
/// numbers are each loop's own spill count, and sibling loops through the same
/// interpreter differ by a slot or two, which is enough to refuse a chain that
/// is otherwise exactly the shape the floors exist to keep compiled. Rounding
/// lands those siblings on one layout; the cost is the rounded-away slots,
/// bounded by this constant per compiled token.
const FROZEN_CHAIN_SLOT_GRANULARITY: usize = 16;

/// `n` rounded up to a whole number of [`FROZEN_CHAIN_SLOT_GRANULARITY`] slots.
fn frozen_slot_count(n: usize) -> usize {
    n.div_ceil(FROZEN_CHAIN_SLOT_GRANULARITY) * FROZEN_CHAIN_SLOT_GRANULARITY
}

/// Words a label-parameter entry accepts after `frame_ptr`. One value for
/// every such entry in the process: a loop-closing JUMP reaches its target
/// with `return_call_indirect` on the shared table, and that type-checks the
/// callee against the *calling* module's type index, so two modules cannot
/// each pick their own width.
pub const FROZEN_LABEL_PARAM_ARITY: usize = 16;

/// An op whose result advances loop-carried state. A value produced inside the
/// re-running region by arithmetic or by a heap load is fresh on each pass, so
/// a JUMP carrying it advances the loop. Copies (`SameAs*`), casts,
/// comparisons, and allocations do not.
fn advances_loop_state(opcode: majit_ir::OpCode) -> bool {
    use majit_ir::OpCode::*;
    matches!(
        opcode,
        IntAdd
            | IntSub
            | IntMul
            | UintMulHigh
            | IntFloorDiv
            | IntMod
            | IntAnd
            | IntOr
            | IntXor
            | IntRshift
            | IntLshift
            | UintRshift
            | IntSignext
            | FloatAdd
            | FloatSub
            | FloatMul
            | FloatTrueDiv
            | FloatFloorDiv
            | FloatMod
            | FloatNeg
            | FloatAbs
            | IntNeg
            | IntInvert
            | IntAddOvf
            | IntSubOvf
            | IntMulOvf
            | GetfieldGcR
            | GetfieldGcI
            | GetfieldGcF
            | GetfieldRawI
            | GetfieldRawR
            | GetfieldRawF
            | GetarrayitemGcR
            | GetarrayitemGcI
            | GetarrayitemGcF
            | GetarrayitemRawI
            | GetarrayitemRawR
            | GetarrayitemRawF
            | GcLoadI
            | GcLoadR
            | GcLoadF
            | GcLoadIndexedI
            | GcLoadIndexedR
            | GcLoadIndexedF
            | RawLoadI
            | RawLoadF
    )
}

/// Per-guard (per-trace order), per-fail-arg: whether the value was produced
/// by loop-state-advancing arithmetic or a heap load in the part of the trace
/// that re-runs on every pass — the ops after the loop-header (last) LABEL, or
/// the WHOLE trace when it has no LABEL (a bridge, or a Label-less recursion
/// loop, whose body runs in full each pass). Such a fail arg is fresh in the
/// failing iteration, so a loop-closing bridge that JUMPs it verbatim still
/// advances the chained loop⇄bridge cycle (`compile_bridge`'s livelock check).
fn guard_fail_args_advanced(
    ops: &[majit_ir::Op],
    guard_exits: &[codegen::GuardExit],
) -> Vec<Vec<bool>> {
    let start = ops
        .iter()
        .rposition(|op| op.opcode == majit_ir::OpCode::Label)
        .map_or(0, |p| p + 1);
    let advanced_ids: std::collections::HashSet<u32> = ops[start..]
        .iter()
        .filter(|op| advances_loop_state(op.opcode))
        .map(|op| op.pos.get())
        .filter(|r| *r != majit_ir::OpRef::NONE && !r.is_constant())
        .map(|r| r.raw())
        .collect();
    guard_exits
        .iter()
        .map(|g| {
            let mask =
                crate::codegen::live_fail_arg_mask(g.meta_descr.as_ref(), g.fail_arg_refs.len());
            g.fail_arg_refs
                .iter()
                .zip(mask)
                .filter(|(_, live)| *live)
                .map(|(r, _)| !r.is_constant() && advanced_ids.contains(&r.raw()))
                .collect()
        })
        .collect()
}

use failguard::{
    CallAssemblerTarget, ChainedTraceMeta, CompiledWasmLoop, LabelTarget, WasmFailDescr,
    WasmFrameData, ca_dispatch_publish, ca_dispatch_redirect, ca_dispatch_slot,
    call_assembler_target, global_fail_descr, label_target, publish_call_assembler_target,
    publish_label_target, register_fail_descrs, reserve_fail_descrs,
};
use majit_backend::{AsmInfo, BackendError, DeadFrame, JitCellToken};
use majit_gc::GcAllocator;
use majit_ir::{FailDescr, GcRef, InputArg, Op, OpRc, Value};

/// `x86/assembler.py fixup_target_tokens`, called from BOTH `assemble_loop`
/// (:612) and `assemble_bridge` (:706) — a LABEL assembled inside a bridge is a
/// jump target for later traces exactly like a loop's, and `compile_retrace`
/// (compile.py) reaches the backend through `send_bridge_to_backend`,
/// so a retrace IS a bridge that defines its own LABEL.
///
/// Returns `(label_descrs, published_descrs)`: the descr identity of every
/// LABEL in ordinal order, and the subset actually entered into
/// `LABEL_TARGETS`. `compile_loop` keeps the first for its own JUMP
/// resolution; `compile_bridge` hands the second to the source loop so its
/// `Drop` retracts them.
fn stamp_and_publish_label_targets(
    func_handle: u32,
    frame: codegen::FrameGeometry,
    inputargs: &[InputArg],
    ops: &[Op],
    bridge_entry_arity: Option<usize>,
) -> (Vec<usize>, Vec<usize>) {
    // Stamp each LABEL's loop-target descr with its ordinal (0, 1, 2, …) so a
    // loop-closing bridge can recover which label its terminal JUMP targets:
    // the JUMP and the LABEL share the descr by Arc identity, so the ordinal
    // written here is readable from the bridge's JUMP in `compile_bridge`.
    // Pure metadata — emits no wasm bytes, so the module shape is unchanged.
    // Skip a LABEL whose descr is not loop-target-backed (`set_label_block_id`
    // would panic on a non-`AtomicU32` slot).
    let mut label_block_id: u32 = 0;
    let mut label_descrs: Vec<usize> = Vec::new();
    for op in ops.iter() {
        if op.opcode != majit_ir::OpCode::Label {
            continue;
        }
        // Descr identity of each label, in ordinal order, so
        // `compile_bridge` can resolve which of THIS loop's labels a
        // closing JUMP targets by Arc identity (the JUMP and the LABEL
        // share the descr). The stamped `label_block_id` alone cannot: a
        // loop retraced into several specializations re-stamps a shared
        // descr, and every specialization's start label carries ordinal
        // 0 — a bridge targeting ANOTHER specialization's label would
        // otherwise be mis-chained into this one.
        label_descrs.push(
            op.getdescr()
                .map(|d| std::sync::Arc::as_ptr(&d) as *const () as usize)
                .unwrap_or(0),
        );
        if let Some(descr) = op.getdescr()
            && let Some(target) = descr.as_loop_target_descr()
        {
            target.set_label_block_id(label_block_id);
        }
        label_block_id += 1;
    }
    // Per-label resume metadata (ordinal order) for `compile_bridge`'s
    // accept condition: a loop-closing bridge may resume at ANY label via
    // the entry `br_table`, provided its JUMP arity matches that label's
    // arg count and the label's args are the complete live set of the
    // trace remainder.
    let label_num_args = codegen::label_arg_counts(ops);
    let label_resume_info = codegen::label_resume_info(inputargs, ops, frame);
    // The wide entry occupies the slot the host appended right after this
    // module's narrow one, so it exists only once a handle does. Where
    // `func_handle` is 0 — a native build, which has no host at all —
    // `func_handle + 1` would name slot 1, another trace's entry rather than
    // an absent one, so both fields share the 0-means-absent encoding.
    let wide_slot = if func_handle != 0
        && codegen::has_label_param_entry(inputargs, ops, frame, bridge_entry_arity)
    {
        func_handle + 1
    } else {
        0
    };
    let mut published_descrs = Vec::new();
    // A parameter entry with no fail values remains structurally `(i32) ->
    // i32`, so type-0 indirect calls may enter it. Only a nonzero parameter
    // entry is incompatible with published LABEL targets.
    let suppress_publication = matches!(bridge_entry_arity, Some(arity) if arity > 0);

    // Publish this loop's enterable labels so a loop-closing bridge from
    // ANY loop can chain into them in-module (jump-to-existing-trace). A
    // peeled loop's labels are each enterable through the entry br_table
    // (key = ordinal + 1). A non-peeled loop has no dispatch: only its
    // FIRST label is enterable — through the plain entry (key 0), whose
    // input loader reads `num_inputs` positional slots — and only when
    // the label's arity equals that (the standard loop shape, whose
    // first label's args ARE the inputargs).
    if codegen::is_resumable_peeled(ops) {
        // Only labels at or before the loop header have a resume loader
        // (`codegen::resumable_label_count`); the header is the last of
        // them, and a bridge landing there re-runs no advancing segment.
        let resumable = codegen::resumable_label_count(ops);
        let header = resumable.saturating_sub(1);
        for (j, &id) in label_descrs.iter().enumerate().take(resumable) {
            if id == 0 {
                continue;
            }
            if suppress_publication {
                diag_bump(47);
            } else {
                diag_bump(19);
                publish_label_target(
                    id,
                    LabelTarget {
                        func_handle,
                        wide_slot,
                        key: j as u32 + 1,
                        num_args: label_num_args[j],
                        resume_safe: label_resume_info[j].0,
                        requires_own_frame: label_resume_info[j].1,
                        is_last_label: j == header,
                        frame,
                    },
                );
                published_descrs.push(id);
            }
        }
    } else {
        // A LABEL with real work before it is not reachable through the plain
        // entry: key 0 runs the function from its first op, so a bridge chaining
        // there would re-run that work. Before the descr-strict dispatch every
        // such trace was `is_resumable_peeled` and never reached this branch; a
        // `jump_to_preamble` retrace (own LABEL, foreign closing JUMP) is not, so
        // state the assumption the `is_last_label: true` publication relies on.
        let first_label_at_entry = ops
            .iter()
            .position(|op| op.opcode == majit_ir::OpCode::Label)
            == Some(0);
        let publishable = first_label_at_entry
            && label_descrs.first().is_some_and(|&id| id != 0)
            && label_num_args.first() == Some(&inputargs.len());
        // Counter 21 answers "this trace HAS a first label and it was left
        // unpublished". A trace with no LABEL at all — every ordinary bridge —
        // has nothing to publish and nothing withheld, so it is not a tally.
        if !publishable && !label_descrs.is_empty() {
            diag_bump(21);
        }
        if publishable && suppress_publication {
            diag_bump(47);
        } else if publishable {
            let id = label_descrs[0];
            diag_bump(20);
            publish_label_target(
                id,
                LabelTarget {
                    func_handle,
                    wide_slot,
                    key: 0,
                    num_args: inputargs.len(),
                    resume_safe: true,
                    requires_own_frame: false,
                    // No real ops precede a non-peeled loop's header, so
                    // an entry re-run lands at the header without any
                    // advancing segment — the livelock check applies.
                    is_last_label: true,
                    frame,
                },
            );
            published_descrs.push(id);
        }
    }

    (label_descrs, published_descrs)
}

/// JIT exception state, mirroring the native backends' `JIT_EXC_VALUE` /
/// `JIT_EXC_TYPE` globals. A can-raise helper publishes the pending exception
/// here via `jit_exc_raise`; the compiled trace's `GuardNoException` /
/// `GuardException` read these slots by absolute address through the shared
/// linear memory (host and trace import the same `env.memory`) and fail the
/// guard accordingly. Single-slot per process, matching the single-threaded
/// dynasm/cranelift backends.
static JIT_EXC_VALUE: AtomicI64 = AtomicI64::new(0);
static JIT_EXC_TYPE: AtomicI64 = AtomicI64::new(0);

/// Residual-call scratch shared by emitted wasm and the host trampoline.
/// Trampoline use is strictly LIFO: the host materialises every argument
/// before invoking the callee, and the guest loads the result immediately on
/// return, so a nested guest trampoline call cannot observe an outer call's
/// live data.
static JIT_CALL_AREA: [AtomicI64; codegen::FrameGeometry::CALL_AREA_SLOTS] =
    [const { AtomicI64::new(0) }; codegen::FrameGeometry::CALL_AREA_SLOTS];

/// llmodel.py _store_exception parity: set JIT exception state.
/// `value` is a valid OBJECTPTR (or 0); the exception class is read from
/// `value.typeptr` (offset 0).
pub fn jit_exc_raise(value: i64) {
    let exc_type = if value == 0 {
        0
    } else {
        // `typeptr` is a machine pointer (32-bit on wasm32); read it at
        // pointer width and zero-extend, so the high bits stay clear and
        // `GuardException`'s type comparison matches the baked class pointer.
        unsafe { *(value as *const usize) as i64 }
    };
    JIT_EXC_VALUE.store(value, Ordering::Relaxed);
    JIT_EXC_TYPE.store(exc_type, Ordering::Relaxed);
}

/// grab_exc_value parity: read the pending exception value and clear both
/// slots. Called host-side after a trace returns through a guard exit.
pub fn jit_exc_take() -> i64 {
    let value = JIT_EXC_VALUE.swap(0, Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, Ordering::Relaxed);
    value
}

/// Non-destructive read of `JIT_EXC_VALUE` for the GC root walker (unlike
/// `jit_exc_take`, which swaps the cell to 0).
pub fn jit_exc_value_peek() -> i64 {
    JIT_EXC_VALUE.load(Ordering::Relaxed)
}

/// Clear both exception slots without reading the value.
pub fn jit_exc_clear() {
    JIT_EXC_VALUE.store(0, Ordering::Relaxed);
    JIT_EXC_TYPE.store(0, Ordering::Relaxed);
}

/// Address of `JIT_EXC_VALUE`, embedded as an immediate in JIT-emitted wasm
/// so the trace can load/store it over the shared linear memory
/// (`_store_and_reset_exception` parity).
pub fn jit_exc_value_addr() -> usize {
    &JIT_EXC_VALUE as *const _ as usize
}

/// Address of `JIT_EXC_TYPE`, embedded as an immediate in JIT-emitted wasm.
pub fn jit_exc_type_addr() -> usize {
    &JIT_EXC_TYPE as *const _ as usize
}

/// Address of `JIT_CALL_AREA`, embedded as an immediate in JIT-emitted wasm.
pub fn jit_call_area_addr() -> usize {
    &JIT_CALL_AREA as *const _ as usize
}

/// The per-thread GC box, and the accessors every trampoline reaches it through.
///
/// `gc.py:30` `GcLLDescription.__init__` holds `self.gcdescr` as a plain field
/// on the backend descriptor — there is no per-thread allocator upstream — so
/// this cell is scaffolding, not a ported structure. Only `install_gc_box`
/// fills it and only tests reach that; the production build goes through
/// `install_gc_standalone` and allocates from the `gc_sync` singleton.
///
/// Every accessor opens with `majit_gc::gc_box_installed()`, which without
/// `majit-gc/gc_box` is a constant `false` — so in a production build each one
/// folds to `None`, the thread-local becomes unreachable, and the trampolines
/// call `gc_sync` directly. The gate lives in `majit-gc` because a Cargo
/// feature is per-crate: this crate cannot `#[cfg]` on a feature of its
/// dependency, so the box is eliminated by the optimizer rather than by
/// conditional compilation. Mirrors `majit-backend-dynasm/src/runner.rs`'s
/// `gc_box`.
mod gc_box {
    use super::{GcAllocator, RefCell};

    thread_local! {
        /// llmodel.py self.gc_ll_descr — owned by the active wasm backend on
        /// this thread. Stored as a thread-local so the backend-agnostic
        /// `majit_gc::ActiveGcGuardHooks` shims can reach the live allocator
        /// without taking a wasm dependency. RPython's `cpu.gc_ll_descr`
        /// parity, single-slot per thread.
        static WASM_ACTIVE_GC: RefCell<Option<Box<dyn GcAllocator>>> =
            const { RefCell::new(None) };
        /// Read-only mirror of the box address: the interpreter-safepoint major
        /// holds the mutable borrow while extra-root walkers ask whether a slot
        /// is GC-managed, so that query routes through the raw pointer instead
        /// of taking a second borrow.
        static WASM_ACTIVE_GC_RAW: std::cell::Cell<Option<*mut dyn GcAllocator>> =
            const { std::cell::Cell::new(None) };
    }

    /// `&mut` access to this thread's GC box, for allocation, write barriers
    /// and collection. `None` means there is no box, and the caller runs its
    /// `gc_sync` path instead.
    pub(super) fn with_mut<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> Option<R> {
        if !majit_gc::gc_box_installed() {
            return None;
        }
        WASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            let raw: *mut dyn GcAllocator = guard.as_deref_mut()?;
            // SAFETY: `guard` holds the borrow for the whole `f` call and
            // these are non-reentrant top-level trampolines, so the reborrow
            // is exclusive and outlives `f`.
            Some(f(unsafe { &mut *raw }))
        })
    }

    /// Read-only access that tolerates being reached from inside a collection:
    /// when an in-progress mutation already holds the mutable borrow, read the
    /// same allocator through the raw mirror rather than taking a second one.
    pub(super) fn with_reentrant_ref<R>(f: impl FnOnce(&dyn GcAllocator) -> R) -> Option<R> {
        if !majit_gc::gc_box_installed() {
            return None;
        }
        WASM_ACTIVE_GC.with(|cell| match cell.try_borrow() {
            Ok(guard) => guard.as_deref().map(f),
            // SAFETY: the mirror is published and cleared under the same
            // borrow as the box itself, so a non-null value points at the
            // live allocator, and this query only reads it.
            Err(_) => WASM_ACTIVE_GC_RAW.with(|raw| raw.get().map(|p| f(unsafe { &*p }))),
        })
    }

    /// Whether this thread holds a box at all.
    pub(super) fn present() -> bool {
        majit_gc::gc_box_installed() && WASM_ACTIVE_GC.with(|cell| cell.borrow().is_some())
    }

    /// Store `gc` as this thread's box, publishing the raw mirror with it.
    pub(super) fn store(gc: Box<dyn majit_gc::GcAllocator>) {
        WASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            *guard = Some(gc);
            let raw = guard.as_deref_mut().map(|gc| gc as *mut dyn GcAllocator);
            WASM_ACTIVE_GC_RAW.with(|raw_cell| raw_cell.set(raw));
        });
    }
}

/// Read-only GC query for the guard hooks and codegen helpers. The box arm is
/// reentrancy-tolerant because these can fire during a collection's extra-root
/// walk, which is also why the singleton arm is the reentrant read.
fn with_wasm_active_gc<R>(f: impl Fn(&dyn GcAllocator) -> R) -> Option<R> {
    if let Some(r) = gc_box::with_reentrant_ref(&f) {
        return Some(r);
    }
    if majit_gc::gc_sync::is_initialized() {
        return Some(majit_gc::gc_sync::gc_query_reentrant(|gc| f(gc)));
    }
    None
}

/// `&mut` counterpart of `with_wasm_active_gc` for GC mutations
/// (allocation, write barriers, collection). Test box → box; production
/// (no box, `gc_sync` initialized) → `gc_sync::gc_op`; no GC at all →
/// `None` so callers keep their non-GC fallback. Top-level mutator/
/// blackhole trampolines, never inside a collection, so `gc_op` is correct.
fn with_wasm_active_gc_mut<R>(f: impl FnOnce(&mut dyn GcAllocator) -> R) -> Option<R> {
    if gc_box::present() {
        return gc_box::with_mut(f);
    }
    if majit_gc::gc_sync::is_initialized() {
        return Some(majit_gc::gc_sync::gc_op(|gc| f(gc)));
    }
    None
}

/// Register all backend-agnostic `majit_gc::set_active_*` hooks to the
/// wasm trampolines. Shared by `install_gc_box` (test path: also stores a
/// box in TLS) and `install_gc_standalone` (production: hooks only, no box
/// — the trampolines then route to the `gc_sync` singleton).
fn register_active_hooks(supports_guard_gc_type: bool) {
    majit_gc::set_active_gc_guard_hooks(majit_gc::ActiveGcGuardHooks {
        check_is_object: Some(wasm_check_is_object),
        is_tagged_immediate: Some(wasm_is_tagged_immediate),
        get_actual_typeid: Some(wasm_get_actual_typeid),
        subclass_range: Some(wasm_subclass_range),
        typeid_subclass_range: Some(wasm_typeid_subclass_range),
        typeid_is_object: Some(wasm_typeid_is_object),
        is_registered_type_id: Some(wasm_is_registered_type_id),
        can_move: None,
        supports_guard_gc_type,
    });
    majit_gc::set_active_alloc_nursery_typed(Some(wasm_alloc_nursery_typed));
    majit_gc::set_active_alloc_nursery_headerless_no_collect(Some(
        wasm_alloc_nursery_headerless_no_collect,
    ));
    majit_gc::set_active_alloc_nursery_typed_with_placement(Some(
        wasm_alloc_nursery_typed_with_placement,
    ));
    majit_gc::set_active_alloc_nursery_collecting_typed(Some(wasm_alloc_nursery_collecting_typed));
    majit_gc::set_active_alloc_nursery_collecting_typed_rooted(Some(
        wasm_alloc_nursery_collecting_typed_rooted,
    ));
    majit_gc::set_active_alloc_oldgen_typed(Some(wasm_alloc_oldgen_typed));
    majit_gc::set_active_root_hooks(Some(wasm_gc_add_root), Some(wasm_gc_remove_root));
    majit_gc::set_active_gc_owns_object(Some(wasm_gc_owns_object));
    majit_gc::set_active_gc_shrink_array(Some(wasm_gc_shrink_array));
    majit_gc::set_active_gc_varsize_layout(Some(wasm_gc_varsize_layout));
    majit_gc::set_active_gc_id_or_identityhash(Some(wasm_id_or_identityhash));
    majit_gc::set_active_write_barrier(Some(wasm_active_gc_write_barrier));
    majit_gc::set_active_write_barrier_before_move(Some(wasm_active_gc_write_barrier_before_move));
    majit_gc::set_active_get_objects(Some(wasm_get_objects));
    majit_gc::set_active_get_referents(Some(wasm_get_referents));
    majit_gc::set_active_is_tracked(Some(wasm_is_tracked));
    majit_gc::set_active_get_rpy_memory_usage(Some(wasm_get_rpy_memory_usage));
    majit_gc::set_active_get_rpy_type_index(Some(wasm_get_rpy_type_index));
    majit_gc::set_active_get_rpy_roots(Some(wasm_get_rpy_roots));
    majit_gc::set_active_get_rpy_referents(Some(wasm_get_rpy_referents));
    majit_gc::set_active_is_app_level_object(Some(wasm_is_app_level_object));
    majit_gc::set_active_dump_rpy_heap(Some(wasm_dump_rpy_heap));
    majit_gc::set_active_get_typeids_text(Some(wasm_get_typeids_text));
    majit_gc::set_active_get_typeids_list(Some(wasm_get_typeids_list));
    majit_gc::set_active_add_memory_pressure(Some(wasm_add_memory_pressure));
    majit_gc::set_active_total_memory_pressure(Some(wasm_total_memory_pressure));
    majit_gc::set_active_collect_generation(Some(wasm_collect_generation));
    majit_gc::set_active_collect_step(Some(wasm_collect_step));
    majit_gc::set_active_collect_oldgen(Some(wasm_collect_oldgen_nonmoving));
    majit_gc::set_active_heap_stats(Some(active_gc_heap_stats));
    majit_gc::set_active_gc_memory_stats(Some(active_gc_memory_stats));
    majit_gc::set_active_major_threshold_reached(Some(active_gc_major_threshold_reached));
    majit_gc::set_active_minor_collections_since_major(Some(
        active_gc_minor_collections_since_major,
    ));
    majit_gc::set_active_finalizer_hooks(
        Some(wasm_register_finalizer),
        Some(wasm_finalizer_next_dead),
    );
}

/// Store a GC allocator in the wasm backend thread-local and register
/// the `majit_gc::set_active_*` function-pointer hooks, without
/// requiring a `WasmBackend` instance.
/// Install a GC box into TLS and register all `set_active_*` hooks. Test
/// path only — `set_gc_allocator` hands ownership of a real allocator to
/// the backend thread. Production uses [`install_gc_standalone`], which
/// registers the same hooks WITHOUT a box so the trampolines fall through
/// to `gc_sync`.
fn install_gc_box(gc: Box<dyn majit_gc::GcAllocator>) {
    // Per-thread allocator: its nursery is not the singleton's, so the
    // process-wide published range can no longer answer `is_nursery_object`.
    majit_gc::disarm_published_nursery();
    majit_gc::note_gc_box_installed();
    let supports_guard_gc_type = gc.supports_guard_gc_type();
    gc_box::store(gc);
    register_active_hooks(supports_guard_gc_type);
}

/// Production path: register all `set_active_*` hooks WITHOUT storing a
/// box. `WASM_ACTIVE_GC` stays `None`, so every trampoline routes to the
/// process-global `gc_sync` singleton (the per-thread GC box is the
/// free-threading gap R4 removes).
pub fn install_gc_standalone() {
    majit_gc::gc_sync::gc_op(|gc| gc.freeze_types());
    let supports_guard_gc_type = majit_gc::gc_sync::gc_query(|gc| gc.supports_guard_gc_type());
    register_active_hooks(supports_guard_gc_type);
}

/// Diagnostic only: `(oldgen_total_bytes, nursery_used_bytes)` of the GC owned
/// by this thread's wasm backend, or `(0, 0)` if none is installed. Lets a host
/// runner split GC-retained memory from host-heap growth.
pub fn active_gc_heap_stats() -> (usize, usize) {
    with_wasm_active_gc(|gc| gc.heap_byte_stats()).unwrap_or((0, 0))
}

pub fn active_gc_memory_stats() -> majit_gc::GcMemoryStats {
    with_wasm_active_gc(|gc| gc.gc_memory_stats()).unwrap_or_default()
}

/// Whether the GC owned by this thread's wasm backend wants a major collection
/// (incminimark.py `threshold_reached`). Drives the interpreter GC
/// safepoint, which is on by default on wasm.
pub fn active_gc_major_threshold_reached() -> bool {
    with_wasm_active_gc(|gc| gc.major_threshold_reached()).unwrap_or(false)
}

/// Minor collections the active GC has run since its last major, or `0` when
/// none is installed.
pub fn active_gc_minor_collections_since_major() -> usize {
    with_wasm_active_gc(|gc| gc.minor_collections_since_major()).unwrap_or(0)
}

/// Diagnostic: `(minor_collections, major_collections)` of the active GC, or
/// `(0, 0)` when none is installed. Companion to [`active_gc_heap_stats`].
pub fn active_gc_collection_counts() -> (usize, usize) {
    with_wasm_active_gc(|gc| gc.collection_counts()).unwrap_or((0, 0))
}

/// Assemble the inline nursery-bump parameters for this trace's `New` /
/// `NewWithVtable` ops (rewrite.py malloc-fast-path eligibility over the
/// gc.py:525-531 nursery address surface), or `None` when no GC is active,
/// the `gc_stress` feature is compiled in (the fast path would bypass its
/// per-allocation stress collections), or no allocation op qualifies.
fn nursery_alloc_params(ops: &[Op]) -> Option<codegen::NurseryAllocParams> {
    if majit_gc::gc_stress_enabled() {
        return None;
    }
    let tids: std::collections::HashSet<u32> = ops
        .iter()
        .filter_map(|op| match op.opcode {
            majit_ir::OpCode::New | majit_ir::OpCode::NewWithVtable => {
                Some(op.getdescr()?.as_size_descr()?.type_id())
            }
            majit_ir::OpCode::NewArray | majit_ir::OpCode::NewArrayClear => {
                Some(op.getdescr()?.as_array_descr()?.type_id())
            }
            _ => None,
        })
        .collect();
    if tids.is_empty() {
        return None;
    }
    with_wasm_active_gc(|gc| {
        let free_addr = gc.nursery_free_addr();
        let top_addr = gc.nursery_top_addr();
        if free_addr == 0 || top_addr == 0 {
            return None;
        }
        let plain_tids: std::collections::HashSet<u32> = tids
            .iter()
            .copied()
            .filter(|&t| gc.type_alloc_is_plain(t))
            .collect();
        if plain_tids.is_empty() {
            return None;
        }
        Some(codegen::NurseryAllocParams {
            free_addr: free_addr as u32,
            top_addr: top_addr as u32,
            large_threshold: gc.max_nursery_object_size(),
            plain_tids,
        })
    })?
}

/// Assemble the direct CA arm's fixed-size nursery/frame parameters. This is
/// deliberately separate from ordinary `New*` eligibility: a CA frame needs
/// both the nursery words and the JitFrame shadow-stack top/limit cells.
/// Missing active GC (or gc_stress) leaves the pre-existing helper path intact.
fn ca_inline_params(frame_bytes: u32) -> Option<codegen::CaInlineParams> {
    if majit_gc::gc_stress_enabled() {
        return None;
    }
    let jitframe_tid = wasm_jitframe_tid();
    let depth = frame_bytes as usize / std::mem::size_of::<isize>();
    let total = ((majit_gc::header::GcHeader::SIZE
        + majit_backend::jitframe::JitFrame::alloc_size(depth))
    .max(majit_gc::header::GcHeader::MIN_NURSERY_OBJ_SIZE)
        + 7)
        & !7;
    with_wasm_active_gc(|gc| {
        assert_ne!(
            jitframe_tid, 0,
            "wasm CA inline frame path requires the registered JitFrame type id"
        );
        if total >= gc.max_nursery_object_size() || !gc.type_alloc_is_plain(jitframe_tid) {
            return None;
        }
        let nursery_free_addr = gc.nursery_free_addr();
        let nursery_top_addr = gc.nursery_top_addr();
        let jf_top_addr = majit_gc::shadow_stack::get_root_stack_top_addr();
        let jf_limit_addr = majit_gc::shadow_stack::get_root_stack_limit_addr();
        (nursery_free_addr != 0 && nursery_top_addr != 0 && jf_top_addr != 0 && jf_limit_addr != 0)
            .then_some(codegen::CaInlineParams {
                nursery_free_addr: nursery_free_addr as u32,
                nursery_top_addr: nursery_top_addr as u32,
                jf_top_addr: jf_top_addr as u32,
                jf_limit_addr: jf_limit_addr as u32,
                jitframe_tid,
            })
    })?
}

/// Whether the host entry runs a trace on a `JitFrame` it pushed onto the
/// jitframe shadow stack.
///
/// `execute_token` allocates that frame only once a `JitFrame` type id has been
/// registered; with none it runs the trace on a plain host buffer, which no
/// collection moves and which the shadow stack never describes. Every frame
/// reload a trace body emits answers out of that shadow stack, so an embedder
/// that registered no type id must get no reloads at all — a reload there would
/// replace the running frame pointer with whatever root happens to sit on top.
fn host_entry_frame_is_jitframe() -> bool {
    wasm_jitframe_tid() != 0
}

/// Address of the active jitframe shadow-stack top cell for ordinary trace
/// body reloads. This does not depend on nursery fast-path eligibility: the
/// reload is valid whenever a GC is active at compilation time *and* the host
/// entry runs its traces on a pushed `JitFrame`.
fn jf_top_addr() -> Option<u32> {
    if !host_entry_frame_is_jitframe() {
        return None;
    }
    with_wasm_active_gc(|_| majit_gc::shadow_stack::get_root_stack_top_addr())
        .and_then(|addr| u32::try_from(addr).ok())
        .filter(|&addr| addr != 0)
}

/// Table slot of the frame-reload helper, or `0` when the running frame is not
/// one the shadow stack describes. Zero reaches codegen as "this trace needs no
/// reload", which is what a frame the host never pushed — and never moves —
/// requires.
fn body_reload_fn_ptr() -> i64 {
    if !host_entry_frame_is_jitframe() {
        return 0;
    }
    wasm_jit_ca_reload_frame as *const () as usize as i64
}

/// `majit_gc::CollectGenerationFn` installed by `register_active_hooks`. Drives
/// `gc.collect(n)` (`interp_gc.py`) through the active GC. Without it
/// `majit_gc::collect_generation` has no hook to dispatch to and silently
/// returns,
/// so no major cycle ever runs on this backend and
/// `deal_with_objects_with_finalizers` — which lives inside the major — never
/// executes: no `__del__`, no generator `finally`, not even under an explicit
/// `gc.collect()`. Mirrors dynasm's `dynasm_collect_generation` and cranelift's
/// `collect_generation_via_active_runtime`.
fn wasm_collect_generation(generation: i64) {
    with_wasm_active_gc_mut(|gc| gc.collect_generation(generation));
}

fn wasm_collect_step() -> majit_gc::GcStepTransition {
    with_wasm_active_gc_mut(|gc| gc.collect_step()).unwrap_or(majit_gc::GcStepTransition {
        // `rgc.py:20-31`: SCANNING on both sides would never report completion.
        old_state: majit_gc::GcStepTransition::STATE_MARKING,
        new_state: majit_gc::GcStepTransition::STATE_SCANNING,
    })
}

/// `majit_gc::CollectOldgenFn` installed by `set_gc_allocator`. Drives the
/// interpreter-safepoint non-moving old-gen major (`gc_interp::safepoint`,
/// default-on on wasm) through the active GC. Needs mutable access, so it
/// routes via `with_wasm_active_gc_mut` (test box → box; production → the
/// `gc_sync` singleton). Mirrors dynasm's `dynasm_collect_oldgen_nonmoving`
/// and cranelift's `collect_oldgen_nonmoving_via_active_runtime`.
fn wasm_collect_oldgen_nonmoving() {
    with_wasm_active_gc_mut(|gc| gc.collect_oldgen_nonmoving());
}

fn wasm_get_objects(generation: i8, visitor: majit_gc::GetObjectsVisitorFn) {
    let mut visit = visitor;
    with_wasm_active_gc_mut(|gc| gc.get_objects(generation, &mut visit));
}

fn wasm_get_referents(obj: GcRef, visitor: majit_gc::GetObjectsVisitorFn) {
    let mut visit = visitor;
    with_wasm_active_gc_mut(|gc| gc.get_referents(obj, &mut visit));
}

fn wasm_is_tracked(obj: GcRef) -> bool {
    with_wasm_active_gc_mut(|gc| gc.is_tracked(obj)).unwrap_or(false)
}

fn wasm_get_rpy_memory_usage(obj: GcRef) -> Option<usize> {
    with_wasm_active_gc_mut(|gc| gc.get_rpy_memory_usage(obj)).flatten()
}

fn wasm_get_rpy_type_index(obj: GcRef) -> Option<usize> {
    with_wasm_active_gc_mut(|gc| gc.get_rpy_type_index(obj)).flatten()
}

fn wasm_get_rpy_roots(visitor: majit_gc::GetObjectsVisitorFn) -> bool {
    let mut visit = visitor;
    with_wasm_active_gc_mut(|gc| gc.get_rpy_roots(&mut visit)).unwrap_or(false)
}

fn wasm_get_rpy_referents(obj: GcRef, visitor: majit_gc::GetObjectsVisitorFn) -> bool {
    let mut visit = visitor;
    with_wasm_active_gc_mut(|gc| gc.get_rpy_referents(obj, &mut visit)).unwrap_or(false)
}

fn wasm_is_app_level_object(obj: GcRef) -> bool {
    with_wasm_active_gc_mut(|gc| gc.is_app_level_object(obj)).unwrap_or(false)
}

fn wasm_dump_rpy_heap(fd: i32) -> Result<bool, i32> {
    with_wasm_active_gc_mut(|gc| gc.dump_rpy_heap(fd)).unwrap_or(Ok(false))
}

fn wasm_get_typeids_text() -> Option<Vec<u8>> {
    with_wasm_active_gc(|gc| gc.get_typeids_text()).flatten()
}

fn wasm_get_typeids_list() -> Option<Vec<usize>> {
    with_wasm_active_gc(|gc| gc.get_typeids_list()).flatten()
}

fn wasm_add_memory_pressure(size: isize, object: GcRef) {
    with_wasm_active_gc_mut(|gc| gc.add_memory_pressure(size, object));
}

fn wasm_total_memory_pressure() -> isize {
    with_wasm_active_gc_mut(|gc| gc.total_memory_pressure()).unwrap_or(0)
}

/// `minimark.py id_or_identityhash` trampoline. The collector
/// records a move-stable hash in its side table before the object can be
/// relocated; the unhooked `majit_gc::gc_id_or_identityhash` fallback returns
/// the raw address instead, which changes under the object when a minor
/// collection moves it out of the nursery. Mirrors dynasm's
/// `dynasm_id_or_identityhash`.
fn wasm_id_or_identityhash(addr: usize) -> usize {
    with_wasm_active_gc_mut(|gc| gc.id_or_identityhash(addr)).unwrap_or(addr)
}

fn wasm_register_finalizer(fq_index: usize, obj: GcRef, trigger: majit_gc::FinalizerTriggerFn) {
    with_wasm_active_gc_mut(|gc| gc.register_finalizer(fq_index, obj, trigger));
}

fn wasm_finalizer_next_dead(fq_index: usize) -> Option<GcRef> {
    with_wasm_active_gc_mut(|gc| gc.finalizer_next_dead(fq_index)).flatten()
}

/// `majit_gc::CheckIsObjectFn` installed by `set_gc_allocator`.
/// Mirrors cranelift's `check_is_object_via_active_runtime`: dispatches
/// through the wasm-thread-local GC allocator.
fn wasm_check_is_object(gcref: GcRef) -> bool {
    with_wasm_active_gc(|gc| gc.check_is_object(gcref)).unwrap_or(false)
}

fn wasm_is_tagged_immediate(addr: usize) -> bool {
    with_wasm_active_gc(|gc| gc.is_tagged_immediate(addr)).unwrap_or(false)
}

fn wasm_get_actual_typeid(gcref: GcRef) -> Option<u32> {
    with_wasm_active_gc(|gc| gc.get_actual_typeid(gcref)).flatten()
}

fn wasm_subclass_range(classptr: usize) -> Option<(i64, i64)> {
    with_wasm_active_gc(|gc| gc.subclass_range(classptr)).flatten()
}

fn wasm_typeid_subclass_range(typeid: u32) -> Option<(i64, i64)> {
    with_wasm_active_gc(|gc| gc.typeid_subclass_range(typeid)).flatten()
}

fn wasm_typeid_is_object(typeid: u32) -> Option<bool> {
    with_wasm_active_gc(|gc| gc.typeid_is_object(typeid)).flatten()
}

fn wasm_is_registered_type_id(typeid: u32) -> bool {
    with_wasm_active_gc(|gc| (typeid as usize) < gc.type_count()).unwrap_or(false)
}

/// Host-side nursery allocation trampoline. Published via
/// `majit_gc::set_active_alloc_nursery_typed` so backend-agnostic
/// callers (pyre-object `w_int_new`, …) can route through the
/// wasm-owned GC.
fn wasm_alloc_nursery_typed(type_id: u32, size: usize) -> GcRef {
    // See cranelift/dynasm counterparts: host-side allocation must not
    // trigger collection because the caller holds a raw pointer that
    // is not a registered GC root.
    with_wasm_active_gc_mut(|gc| gc.try_alloc_nursery_no_collect_typed(type_id, size))
        .unwrap_or(GcRef(0))
}

/// `majit_gc::AllocNurseryHeaderlessNoCollectFn`. The metainterp's jitcode
/// tracer allocates a `NEW` on a `headerless` descr through here so the object
/// lands in the interpreter's own collected pool rather than the host heap,
/// where its collector could not see it. Returns `GcRef(0)` when no GC is
/// bound, leaving the caller on its own path.
fn wasm_alloc_nursery_headerless_no_collect(size: usize) -> GcRef {
    with_wasm_active_gc_mut(|gc| gc.alloc_nursery_headerless_no_collect(size)).unwrap_or(GcRef(0))
}

/// Placement-reporting companion of [`wasm_alloc_nursery_typed`].
///
/// # Safety
/// `needs_write_barrier` must remain a valid mutable `bool` slot until this
/// call returns.
unsafe fn wasm_alloc_nursery_typed_with_placement(
    type_id: u32,
    size: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    with_wasm_active_gc_mut(|gc| unsafe {
        gc.try_alloc_nursery_no_collect_typed_with_placement(type_id, size, needs_write_barrier)
    })
    .unwrap_or(GcRef(0))
}

/// Host-side collecting nursery allocation used by elidable bigint payload
/// helpers. This is the wasm twin of the dynasm/cranelift hooks: the active
/// backend must replace every process-global allocation hook as one unit so a
/// previously-installed native backend cannot receive wasm allocations.
fn wasm_alloc_nursery_collecting_typed(type_id: u32, size: usize) -> GcRef {
    with_wasm_active_gc_mut(|gc| gc.alloc_nursery_typed(type_id, size)).unwrap_or(GcRef(0))
}

/// Rooted collecting companion for a result whose GC child exists only in a
/// native Rust slot while the parent allocation may collect.
///
/// # Safety
/// `root` and `needs_write_barrier` must remain valid mutable slots until this
/// call returns.
unsafe fn wasm_alloc_nursery_collecting_typed_rooted(
    type_id: u32,
    size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef {
    with_wasm_active_gc_mut(|gc| unsafe {
        gc.alloc_nursery_collecting_typed_rooted(type_id, size, root, needs_write_barrier)
    })
    .unwrap_or(GcRef(0))
}

/// Host-side old-gen allocation trampoline. Stable
/// across minor/major collections — see dynasm counterpart.
fn wasm_alloc_oldgen_typed(type_id: u32, size: usize) -> GcRef {
    with_wasm_active_gc_mut(|gc| gc.alloc_oldgen_typed(type_id, size)).unwrap_or(GcRef(0))
}

/// Allocate the block a blackhole `bh_new*` descr describes, in the non-moving
/// old generation as the dynasm and cranelift runners do: resume
/// materialization keeps raw pointers to these across the forward blackhole
/// run, and a nursery block would be relocated out from under them at the next
/// minor collection.
fn wasm_bh_alloc(type_id: u32, payload_size: usize) -> i64 {
    let gc_ptr = if type_id != 0 {
        wasm_alloc_oldgen_typed(type_id, payload_size).0
    } else {
        0
    };
    if gc_ptr != 0 {
        // A blackhole-materialized object is born into the old generation, so the
        // collector reaches it only through the remembered set. It is not on the
        // frame chain `walk_pyframe_roots` walks, and the resume fills it through
        // stores that are barrier-free precisely because that chain is a root set
        // (`ExecutionContext::enter`), so nothing would put it there. A nursery
        // object needed none of this — being young was enough to have its fields
        // traced. Remember it at birth to restore that, before any field is
        // written; `TRACK_YOUNG_PTRS` is set by `finish_alloc_in_oldgen`, so this
        // is the ordinary barrier, not a new mechanism.
        //
        // The raw fallback below is deliberately outside it: that block is plain
        // malloc, not a collector-owned object, so it has no header to remember.
        majit_gc::gc_write_barrier(GcRef(gc_ptr));
        return gc_ptr as i64;
    }
    if type_id != 0 {
        // `GcLLDescr_framework._bh_malloc` returns NULL on failure so the
        // blackhole wrapper can raise `MemoryError`.  A raw fallback for a
        // typed descr drops the GC header and its tracing layout.
        return 0;
    }
    wasm_bh_alloc_raw(payload_size)
}

/// Non-GC descrs (`type_id == 0`, raw buffers) keep the plain zeroed malloc the
/// dynasm runner uses for the same descr shape.
fn wasm_bh_alloc_raw(size: usize) -> i64 {
    let Ok(layout) = std::alloc::Layout::from_size_align(size.max(1), 8) else {
        return 0;
    };
    unsafe { std::alloc::alloc_zeroed(layout) as i64 }
}

/// Allocate the struct a blackhole `bh_new` / `bh_new_with_vtable` describes,
/// mirroring the dynasm runner's `bh_alloc_struct`.
///
/// A headerless descr names a struct from the interpreter's own
/// `headerless_structs` pool, which carries no type word at `ref - 8`; a
/// header-writing allocator returns `base + GcHeader::SIZE` and puts a block
/// the interpreter owns onto the collector's lists.
///
/// `resolve_gc_tid` routes the serialized `path_hash` cache key back to the
/// dense GC tid (`gc.py:536-542`); a raw cache key read as a tid indexes past
/// the type table on the first collection that traces the block.
fn wasm_bh_alloc_struct(sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
    let size = sizedescr.as_size();
    if sizedescr.is_headerless() {
        let gc_ptr = wasm_alloc_nursery_headerless_no_collect(size).0;
        if gc_ptr != 0 {
            return gc_ptr as i64;
        }
        return wasm_bh_alloc_raw(size);
    }
    wasm_bh_alloc(sizedescr.resolve_gc_tid(), size)
}

/// JIT-trace allocation trampoline target for `New` / `NewWithVtable`.
///
/// A compiled trace cannot allocate directly (the GC lives behind the
/// `WASM_ACTIVE_GC` thread-local), so the `New` codegen routes through the
/// host `jit_call` trampoline, which resolves this function via the module's
/// `__indirect_function_table` (its address is taken in `compile_loop`, so it
/// lands in the table) and invokes it with `(type_id, size)`. Returns the new
/// object pointer, or 0 when no GC is installed. The `ob_type` field for
/// `NewWithVtable` is written inline by codegen at `vtable_offset`.
///
/// Unlike the general [`wasm_alloc_nursery_typed`] host hook (which must not
/// collect — its callers hold unrooted raw pointers), this JIT-trace path is
/// safe to collect: the trace registers every live Ref's frame home slot as a
/// GC root and reloads its locals from the (forwarded) homes after each
/// allocation. So it uses the *collecting* `alloc_nursery_typed`, which
/// triggers a minor collection on nursery-full instead of leaking to old-gen.
pub extern "C" fn wasm_jit_alloc(type_id: i64, size: i64) -> i64 {
    with_wasm_active_gc_mut(|gc| gc.alloc_nursery_typed(type_id as u32, size as usize).0 as i64)
        .unwrap_or(0)
}

/// JIT-trace variable-size allocation trampoline target for `NewArray` /
/// `NewArrayClear`. Allocates `length` items and writes the length field at
/// `len_offset`, mirroring [`WasmBackend::bh_new_array`].
pub extern "C" fn wasm_jit_alloc_array(
    type_id: i64,
    base_size: i64,
    item_size: i64,
    length: i64,
    len_offset: i64,
) -> i64 {
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    with_wasm_active_gc_mut(|gc| {
        let obj = gc.alloc_varsize_typed(
            type_id as u32,
            base_size as usize,
            item_size as usize,
            length,
        );
        if obj.is_null() {
            0
        } else {
            unsafe {
                *((obj.0 as *mut u8).add(len_offset as usize) as *mut usize) = length;
            }
            obj.0 as i64
        }
    })
    .unwrap_or(0)
}

/// Old-generation twin of [`wasm_jit_alloc`], selected by the `New` /
/// `NewWithVtable` codegen for a `non_moving` size descr. Same signature, so it
/// shares the call shape; only the generation differs.
///
/// A descr marked `non_moving` is one whose object is reached through a raw
/// pointer nothing forwards — the interpreter holds it across an allocation, or
/// another object's field stores it outside the collector's view. Placing such
/// an object in the movable nursery leaves those pointers aimed at the pre-move
/// copy. The native backends honour the flag in the GC rewrite pass
/// (`rewrite.rs` `handle_new`); wasm lowers `New` itself, so it must apply the
/// same policy here.
pub extern "C" fn wasm_jit_alloc_oldgen(type_id: i64, size: i64) -> i64 {
    with_wasm_active_gc_mut(|gc| gc.alloc_oldgen_typed(type_id as u32, size as usize).0 as i64)
        .unwrap_or(0)
}

/// Old-generation twin of [`wasm_jit_alloc_array`], selected by the `NewArray` /
/// `NewArrayClear` codegen for a `non_moving` array descr. Same signature and
/// the same length stamp; see [`wasm_jit_alloc_oldgen`] for why the generation
/// is part of the descr's contract.
pub extern "C" fn wasm_jit_alloc_array_oldgen(
    type_id: i64,
    base_size: i64,
    item_size: i64,
    length: i64,
    len_offset: i64,
) -> i64 {
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    let payload_size = base_size as usize + item_size as usize * length;
    with_wasm_active_gc_mut(|gc| {
        let obj = gc.alloc_oldgen_typed(type_id as u32, payload_size);
        if obj.is_null() {
            0
        } else {
            unsafe {
                *((obj.0 as *mut u8).add(len_offset as usize) as *mut usize) = length;
            }
            obj.0 as i64
        }
    })
    .unwrap_or(0)
}

/// Table indices of the four allocation trampolines, for the `New*` /
/// `NewArray*` codegen. Taking each address here is what keeps the function in
/// the module's `__indirect_function_table`, so a trace can `call_indirect` it.
fn alloc_helpers() -> codegen::AllocHelpers {
    codegen::AllocHelpers {
        new_fn_ptr: wasm_jit_alloc as *const () as usize as i64,
        new_array_fn_ptr: wasm_jit_alloc_array as *const () as usize as i64,
        new_oldgen_fn_ptr: wasm_jit_alloc_oldgen as *const () as usize as i64,
        new_array_oldgen_fn_ptr: wasm_jit_alloc_array_oldgen as *const () as usize as i64,
    }
}

/// JIT-trace write-barrier trampoline target for ref-storing `SetfieldGc` /
/// `SetarrayitemGc` / `SetinteriorfieldGc`. Routes through the host `jit_call`
/// trampoline; invokes the active GC's `write_barrier`, which adds an old
/// object that may now hold a young reference to the remembered set (and clears
/// TRACK_YOUNG_PTRS). A young base (no flag) or a null base is a no-op. wasm
/// skips the native GC rewrite pass, so the trace emits this barrier directly
/// instead of `COND_CALL_GC_WB`. Returns 0 — the store codegen ignores it.
pub extern "C" fn wasm_jit_write_barrier(obj: i64) -> i64 {
    with_wasm_active_gc_mut(|gc| gc.write_barrier(GcRef(obj as usize)));
    0
}

/// Self-recursive CALL_ASSEMBLER (`PYRE_WASM_CA`) callee-frame allocation
/// helper. Allocates the callee's execution frame as a young nursery
/// GC-managed `JitFrame`, mirroring rewrite.py's nursery frame allocation:
/// steady recursive frames die young, while only frames alive across a
/// collection are promoted. The frame is traced through the jitframe type id's
/// custom trace using its per-frame `jf_gcmap`, rooted by pushing it on the
/// jitframe shadow stack, and reloaded after the recursive call because a
/// nursery frame may move. Returns the frame base (codegen adds
/// `FIRST_ITEM_OFFSET` for the bespoke-layout frame pointer), or 0 on
/// allocation failure.
///
/// Each callee frame self-describes through its own per-frame gcmap, so
/// mixed-geometry frames from distinct CA bridges are each forwarded by their
/// own geometry — no shared coarse single-stride scan that mis-reads a larger
/// frame's interior as a smaller frame's slots.
pub extern "C" fn wasm_jit_ca_alloc_frame(frame_bytes: i64, gcmap_ptr: i64) -> i64 {
    use majit_backend::jitframe::JitFrame;
    assert!(frame_bytes >= 0);
    assert_eq!(frame_bytes as usize % std::mem::size_of::<isize>(), 0);
    let depth = frame_bytes as usize / std::mem::size_of::<isize>();
    // Collecting nursery allocation, matching rewrite.py's
    // `gen_malloc_nursery_varsize_frame`. The caller frame remains rooted at
    // the shadow-stack top during a collection; wasm reloads it from there
    // after this call, then this freshly allocated callee is pushed below its
    // own execution. Steady recursive frames die young; only frames that live
    // through a collection are promoted instead of inflating the old-gen major
    // collection threshold on every call.
    let jf_ref = with_wasm_active_gc_mut(|gc| {
        gc.alloc_nursery_typed(wasm_jitframe_tid(), JitFrame::alloc_size(depth))
    })
    .unwrap_or(GcRef(0));
    if jf_ref.0 == 0 {
        return 0;
    }
    let jf = jf_ref.0 as *mut JitFrame;
    unsafe {
        JitFrame::init(jf, std::ptr::null(), depth);
        (*jf).jf_gcmap = gcmap_ptr as *const u8;
    }
    majit_gc::shadow_stack::push_jf(jf_ref);
    jf_ref.0 as i64
}

/// Companion to [`wasm_jit_ca_alloc_frame`]: pop the top jitframe shadow-stack
/// entry on CA-arm exit. The CA recursion is strict LIFO — each level pushes
/// one frame before its `call_indirect` and pops after, and a deopt resume runs
/// on the host's own shadow stack — so removing the top entry releases exactly
/// this callee's frame.
pub extern "C" fn wasm_jit_ca_pop_frame(_frame_base: i64) -> i64 {
    majit_gc::shadow_stack::pop_jf_top();
    0
}

/// Reload the current CA callee frame pointer after a recursive call. The GC
/// may have moved the callee frame during the recursive call; `jf_top_ptr()`
/// reads the forwarded base from the jitframe shadow-stack slot. At this point
/// this recursion level's frame is the top — deeper levels have already popped.
/// Analog of `_reload_frame_if_necessary`; returns the ITEMS base held in the
/// CA arm's `ca_cfp_local`.
pub extern "C" fn wasm_jit_ca_reload_frame() -> i64 {
    majit_gc::shadow_stack::jf_top_ptr().0 as i64
        + majit_backend::jitframe::FIRST_ITEM_OFFSET as i64
}

/// Reload the CA caller's frame pointer after the callee-frame allocation.
/// The allocation occurs before the callee is pushed, so while the callee is
/// live the caller remains one entry below the shadow-stack top. Returns that
/// caller's ITEMS base for local 0.
pub extern "C" fn wasm_jit_ca_reload_caller_frame() -> i64 {
    majit_gc::shadow_stack::jf_under_top_ptr().0 as i64
        + majit_backend::jitframe::FIRST_ITEM_OFFSET as i64
}

/// Build the per-frame `jf_gcmap` for a CA callee frame: mark the input slots
/// (at `FRAME_SLOT_BASE`) and the home slots (at `HOME_SLOT_BASE`), in the
/// `JitFrame`'s Signed-granular item indexing (see [`build_home_gcmap`] for the
/// wasm32 layout). The collector's `is_nursery_object_start` gate skips any
/// marked slot that does not hold a live nursery object base, so a slot holding
/// a scalar or an already-promoted Ref is traced harmlessly.
///
/// Returned buffer is leaked by the caller (one per bridge) and lives for the
/// program's life.
fn build_callee_gcmap(
    input_types: &[majit_ir::Type],
    frame: codegen::FrameGeometry,
) -> Box<[usize]> {
    let sign = std::mem::size_of::<isize>();
    let bits_per_word = std::mem::size_of::<usize>() * 8;
    let mut indices: Vec<usize> = Vec::with_capacity(input_types.len() + frame.home_slots);
    for (i, &tp) in input_types.iter().enumerate() {
        if tp == majit_ir::Type::Ref {
            indices.push((codegen::FRAME_SLOT_BASE as usize + i * 8) / sign);
        }
    }
    for h in 0..frame.home_slots {
        indices.push((frame.home_slot_base as usize + h * 8) / sign);
    }
    let max_index = indices.iter().copied().max().unwrap_or(0);
    // `wasm_jit_ca_alloc_frame` sets `jf_frame` from `ca_frame_bytes`, not the
    // full geometry. Inputs and homes must therefore fit that actual item
    // allocation; fail/deopt outputs live in the low value slots and are
    // covered by the same bound.
    debug_assert!(
        max_index < frame.ca_frame_bytes as usize / sign,
        "CA gcmap exceeds the allocated JitFrame item area"
    );
    let num_words = max_index / bits_per_word + 1;
    let mut buf = vec![0usize; 1 + num_words];
    buf[0] = num_words;
    for index in indices {
        buf[1 + index / bits_per_word] |= 1usize << (index % bits_per_word);
    }
    buf.into_boxed_slice()
}

/// Host-side root-register trampoline.
///
/// # Safety
/// Caller must keep `slot` valid until [`wasm_gc_remove_root`] is
/// called with the same pointer.
pub(crate) unsafe fn wasm_gc_add_root(slot: *mut GcRef) {
    with_wasm_active_gc_mut(|gc| unsafe { gc.add_root(slot) });
}

/// Batched [`wasm_gc_add_root`] for one stack-shaped root bracket.
///
/// # Safety
/// Every slot must remain valid until removed with [`wasm_gc_remove_roots`].
pub(crate) unsafe fn wasm_gc_add_roots(slots: &[usize]) {
    if slots.is_empty() {
        return;
    }
    with_wasm_active_gc_mut(|gc| {
        for &slot in slots {
            unsafe { gc.add_root(slot as *mut GcRef) };
        }
    });
}

/// Companion to [`wasm_gc_add_root`].
pub(crate) fn wasm_gc_remove_root(slot: *mut GcRef) {
    with_wasm_active_gc_mut(|gc| gc.remove_root(slot));
}

/// Batched [`wasm_gc_remove_root`] for one stack-shaped root bracket.
pub(crate) fn wasm_gc_remove_roots(slots: impl Iterator<Item = usize>) {
    with_wasm_active_gc_mut(|gc| {
        for slot in slots {
            gc.remove_root(slot as *mut GcRef);
        }
    });
}

/// Host-side write-barrier trampoline for the interpreter (mapdict / list /
/// set / dict stores route through `majit_gc::gc_write_barrier`). Mirrors
/// `dynasm_gc_write_barrier`; without it every interpreter ref-store is a
/// silent no-op, so a collecting nursery loses old→young pointers.
fn wasm_active_gc_write_barrier_before_move(obj: GcRef) {
    with_wasm_active_gc_mut(|gc| gc.writebarrier_before_move(obj));
}

fn wasm_active_gc_write_barrier(obj: GcRef) {
    with_wasm_active_gc_mut(|gc| gc.write_barrier(obj));
}

/// Host-side `is_managed_heap_object` trampoline.
///
/// This query can fire reentrantly from an extra-root walker mid-collection
/// (the interpreter-safepoint major holds the box's mutable borrow while
/// asking whether a slot is GC-managed), so both arms are read-only.
fn wasm_gc_owns_object(addr: usize) -> bool {
    if let Some(r) = gc_box::with_reentrant_ref(|gc| gc.is_managed_heap_object(addr)) {
        return r;
    }
    majit_gc::gc_sync::is_initialized()
        && majit_gc::gc_sync::gc_query_reentrant(|g| g.is_managed_heap_object(addr))
}

/// `llop.shrink_array`.  This changes a GC-owned object's length word and must
/// therefore take the exclusive collector path; `gc_query_reentrant` is for
/// read-only queries made while a collection may already hold `&mut`.
fn wasm_gc_shrink_array(addr: usize, smaller_length: usize) -> bool {
    with_wasm_active_gc_mut(|gc| gc.shrink_array(addr, smaller_length)).unwrap_or(false)
}

fn wasm_gc_varsize_layout(addr: usize) -> Option<majit_gc::GcVarSizeLayout> {
    let obj = GcRef(addr);
    if let Some(r) = gc_box::with_reentrant_ref(|gc| gc.varsize_layout(obj)) {
        return r;
    }
    if majit_gc::gc_sync::is_initialized() {
        majit_gc::gc_sync::gc_query_reentrant(|g| g.varsize_layout(obj))
    } else {
        None
    }
}

pub struct WasmBackend {
    /// `rpython/jit/backend/model.py:28-29 self.tracker =
    /// CPUTotalTracker()` parity — per-instance `cpu.tracker`
    /// exposed via [`majit_backend::Backend::cpu_tracker`].
    cpu_tracker: std::sync::Arc<majit_backend::CpuTotalTracker>,
    /// `asmmemmgr.py` `AsmMemoryManager` parity — what
    /// `jit_hooks.stats_asmmemmgr_{allocated,used}` reads. The emitted trace is
    /// a wasm module handed to the host compiler, so there is no arena of ours
    /// to size: `allocated` and `used` are both the module's byte length, which
    /// is the figure `asmmemmgr.py` counts for a block a `materialize`
    /// handed out.
    asm_memory_stats: std::sync::Arc<majit_backend::AsmMemoryManagerStats>,
    /// Lifetime tokens for the blocks recorded above. The host keeps every
    /// instantiated module for as long as this backend can enter it, so the
    /// tokens are held for the backend's life and give `used` back with it.
    asm_memory_blocks: Vec<majit_backend::AsmMemoryBlock>,
    trace_counter: u64,
    /// Optimizer constant pool (constant-namespace OpRef → i64 value).
    constants: indexmap::IndexMap<u32, i64>,
    /// llmodel.py:64-69 self.vtable_offset.
    vtable_offset: Option<usize>,
}

/// GC type id of the `JitFrame`. The single registration authority is `eval.rs`
/// (the type is registered there alongside the rest of the heap types, before
/// `freeze_types`); it pushes the id here through `set_wasm_jitframe_tid`,
/// mirroring how it feeds `majit_backend_{cranelift,dynasm}::set_jitframe_gc_type_id`.
/// The orthodox (`PYRE_WASM_CA`) frame path allocates the host-entry frame as a
/// real GC-managed `JitFrame` of this type so the collector forwards its Ref item
/// slots through the `jf_gcmap` custom trace. 0 = not yet pushed (the orthodox
/// path stays disabled until then).
static WASM_JITFRAME_TID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Host entry point used by `eval.rs` to publish the registered `JitFrame` type
/// id (counterpart to `set_jitframe_gc_type_id` on the native backends).
pub fn set_wasm_jitframe_tid(id: u32) {
    WASM_JITFRAME_TID.store(id, std::sync::atomic::Ordering::Relaxed);
    majit_gc::bh_probe_ignore_tid(id);
}

// Only read on the wasm32 execute_token path and by CA callee-frame allocation.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
fn wasm_jitframe_tid() -> u32 {
    WASM_JITFRAME_TID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Build a `jf_gcmap` bitmap marking the surviving Ref-home region as the
/// frame's traced GC roots, in the `JitFrame`'s Signed-granular item indexing.
///
/// On wasm32 `isize` is 4 bytes, so `jf_frame` items are 4-byte Signed slots and
/// each 8-byte data slot spans two items — the orthodox PyPy 32-bit layout where
/// a one-word value (a `GcRef`) occupies a single item and a two-word value
/// (i64) occupies a pair. A Ref home written as an i64 keeps the guest pointer in
/// its LOW word (little-endian), at Signed item index `(HOME_SLOT_BASE + h *
/// 8) / sign`. `jitframe_trace` strides items by `sign` and forwards one word per
/// marked bit, so marking those indices exposes each home's `GcRef` (the high
/// word stays unmarked). Returns `[data_word_count, word0, ...]` in `usize`
/// words (GCMAP array layout: `gcmap[0]` = number of data words).
#[cfg_attr(
    not(target_arch = "wasm32"),
    expect(
        dead_code,
        reason = "native test builds compile the wasm backend without running wasm frame entry"
    )
)]
fn build_home_gcmap(frame: codegen::FrameGeometry) -> Box<[usize]> {
    let sign = std::mem::size_of::<isize>();
    let bits_per_word = std::mem::size_of::<usize>() * 8;
    if frame.home_slots == 0 {
        // One empty data word: a non-null jf_gcmap that traces nothing.
        return vec![1usize, 0usize].into_boxed_slice();
    }
    let last_index = (frame.home_slot_base as usize + (frame.home_slots - 1) * 8) / sign;
    let num_words = last_index / bits_per_word + 1;
    let mut buf = vec![0usize; 1 + num_words];
    buf[0] = num_words;
    for h in 0..frame.home_slots {
        let index = (frame.home_slot_base as usize + h * 8) / sign;
        buf[1 + index / bits_per_word] |= 1usize << (index % bits_per_word);
    }
    buf.into_boxed_slice()
}

/// `__indirect_function_table` slot of `call_jit::wasm_ca_resume_deopt`,
/// published by pyre-jit at boot (`init_jit_hooks`). When an in-guest
/// self-recursive CALL_ASSEMBLER callee leaves its trace through a guard with no
/// bridge — a deopt the in-guest fast path cannot finish — the CA arm
/// `call_indirect`s this slot to blackhole-resume that callee on the host (no
/// re-execution of its pre-guard work) and read back its result. `0` (unset)
/// makes `compile_bridge` decline the CA lift, since the arm would have no way
/// to complete a deopt. Stored as `u64` to reuse the imported atomics.
static CA_DEOPT_HELPER_SLOT: AtomicU64 = AtomicU64::new(0);
/// Dormant runtime-regression selector. The wasm runner writes this through a
/// guest export before executing a test program; zero keeps production runs
/// unchanged. `1` selects the first admitted target, otherwise the value is a
/// `JitCellToken` number.
static FORCE_CA_TERMINAL_DECLINE: AtomicU64 = AtomicU64::new(0);

/// `__indirect_function_table` index of the deferred-merge trip callback,
/// published from pyre-jit the way [`CA_DEOPT_HELPER_SLOT`] is. Zero keeps
/// every merge deferred forever, which is what a host without the callback
/// wants: the bridge stays out of line and correct.
static INLINE_TRIP_HELPER_SLOT: AtomicU64 = AtomicU64::new(0);

/// Entries into an out-of-line bridge before its merge into the owner is
/// installed.
///
/// Merging costs the owner a re-emission — a whole module rebuild, tens of ms
/// of compile time — and buys 7.78ns per cross-module crossing it removes.
/// Nothing at admission time predicts how many that will be: every bridge is
/// compiled at the same guard-failure threshold, and the crossings that decide
/// the answer all happen afterwards and stay in-guest. So the bridge counts its
/// own entries and asks for the merge here.
///
/// The count is calibrated against the two populations the corpus actually
/// holds, because a bridge's entry count is a floor on the crossings its merge
/// removes rather than an estimate of them — merging also brings the region's
/// own external jumps in-module, which on `short_circuit_value_kept_stack` made
/// three merges worth ten times their bridges' entries. Below this, a fixture's
/// ENTIRE crossing budget is under a millisecond and no rebuild can pay back:
/// `polymorphic_slot_retype` crosses 88.4k times in its whole run and
/// `short_circuit_side_effects` 37.7k, and eagerly merging cost them 1.09x
/// each. Above it, the hot bridges of `short_circuit_value_kept_stack` (106k to
/// 532k entries) are worth 0.82x, and `str_getitem_len_hot`'s single region is
/// worth 0.75x on 72.0M crossings removed.
const INLINE_TRIP_THRESHOLD: u64 = 100_000;

/// The per-byte half of the same price, in entries per byte of the module the
/// merge re-emits — see [`inline_trip_threshold_for`], which takes the larger
/// of the two.
///
/// A merge charges cranelift for the whole owner while the crossings it removes
/// answer only to the bridge, so a fixture whose owner is large enough loses on
/// a merge the entry count alone would have taken. At this value `fannkuch`
/// keeps 8 of its 10 merges and `nbody` 3 of its 4, dropping 30KB and 18KB of
/// emitted module, while the merges of the four fixtures that never lose one
/// are postponed by an amount too small to charge them anything. Twice this is
/// already inside the band where postponement dominates.
const DEFAULT_INLINE_TRIP_BYTES_FACTOR: u64 = 40;

/// Bytes of owner module above which the eager merge arm declines.
///
/// That arm merges before the compile returns, so a quasi-immutable fold's
/// dependencies attach to the owner's flag rather than to a temporary bridge's
/// — which is why it cannot wait for entry evidence the way
/// [`inline_trip_threshold_for`] does. What it can read is the re-emission it
/// is about to buy, and successive merges into one owner re-emit it whole each
/// time: four merges into one owner re-emit it four times, at every size it
/// passes through on the way.
///
/// The value is where the corpus stops paying for those re-emissions and has
/// not yet started losing the merges that earn theirs. Below it the fixtures
/// whose merge removes millions of crossings begin to lose it, and each one
/// costs several times what the re-emissions saved.
const DEFAULT_INLINE_EAGER_MAX_BYTES: u32 = 4096;

/// A merge that passed every inline check and is waiting on
/// [`INLINE_TRIP_THRESHOLD`] entries into the bridge compiled in its place.
struct PendingInline {
    /// The loop this region merges into.
    owner: Arc<JitCellToken>,
    region: codegen::InlinedBridge,
}

thread_local! {
    /// Deferred merges by id, the id being what the bridge module passes back.
    static PENDING_INLINES: RefCell<HashMap<i64, PendingInline>> =
        RefCell::new(HashMap::new());
    /// Source of the ids above.
    static NEXT_PENDING_INLINE_ID: std::cell::Cell<i64> = const { std::cell::Cell::new(1) };
}

thread_local! {
    /// Ids whose bridges have reached [`INLINE_TRIP_THRESHOLD`], waiting to be
    /// merged. The probe runs inside the bridge, so the host is between
    /// `run_compiled` and its return and already holds the driver mutably; the
    /// trip only appends here, and [`take_tripped_inlines`]'s caller installs
    /// the merge once the trace has returned.
    static TRIPPED_INLINES: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
}

/// Drops a registered [`PendingInline`] unless the bridge whose probe would
/// fire its callback actually got published.
///
/// Registration has to precede the module build — the probe is one of the
/// build's inputs — so a build or host rejection after it would otherwise leave
/// an entry nothing can ever reach, holding the owner's `Arc<JitCellToken>`,
/// the copied region and its pool for the life of the thread, once per
/// rejected attempt. The counter stays leaked either way; it is eight bytes,
/// and on this path no module was published to increment it.
struct PendingInlineGuard(Option<i64>);

impl PendingInlineGuard {
    /// The bridge is published, so the entry is the callback's to remove.
    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for PendingInlineGuard {
    fn drop(&mut self) {
        if let Some(pending_id) = self.0 {
            PENDING_INLINES.with(|pending| pending.borrow_mut().remove(&pending_id));
        }
    }
}

/// The merged-stream exit ordinal of a guard belonging to a region already
/// merged into this loop, or `None` when `trace_id` names no such region.
///
/// `InlinedBridge::source_fail_index` indexes the merged stream — the owner's
/// own ops followed by every region's, in attach order — whose exits are
/// numbered across the whole of it. A guard in the owner itself is already at
/// its own ordinal; a region's guards start past the owner's and past every
/// region attached before it. The answer stays valid however long a deferred
/// merge waits, because merging only ever appends.
fn merged_region_fail_index(
    inputs: &codegen::ModuleBuildInputs,
    trace_id: u64,
    region_fail_index: u32,
) -> Option<u32> {
    let mut ordinal = codegen::guard_exit_count(&inputs.inputargs, &inputs.ops);
    for region in &inputs.inlined_bridges {
        let count = codegen::guard_exit_count(&region.inputargs, &region.ops);
        if region.trace_id == trace_id {
            return ((region_fail_index as usize) < count)
                .then(|| (ordinal + region_fail_index as usize) as u32);
        }
        ordinal += count;
    }
    None
}

/// Note that a bridge has counted its way to the threshold. Called from
/// compiled code, so it touches nothing but the list above.
pub fn record_inline_trip(pending_id: i64) {
    TRIPPED_INLINES.with(|tripped| tripped.borrow_mut().push(pending_id));
}

/// Take the merges whose bridges have tripped since the last call, for a caller
/// with no compiled trace left on the stack.
pub fn take_tripped_inlines() -> Vec<i64> {
    TRIPPED_INLINES.with(|tripped| std::mem::take(&mut *tripped.borrow_mut()))
}

/// Record a deferred merge and describe the probe the bridge standing in for
/// it carries.
///
/// ⛔ The counter is leaked rather than owned by the entry below: the bridge
/// module increments it on every entry and outlives the merge, which takes its
/// entry out of the map. One `u64` per deferred merge, and the alternative is a
/// live module writing to freed memory.
fn register_pending_inline(
    owner: Arc<JitCellToken>,
    region: codegen::InlinedBridge,
    cells_base_ptr: u32,
    owner_module_bytes: u32,
) -> codegen::InlineTripProbe {
    let counter_addr = Box::leak(Box::new(0u64)) as *const u64 as usize as u32;
    let dispatch_cell_index = region.source_fail_index;
    let pending_id = NEXT_PENDING_INLINE_ID.with(|next| {
        let id = next.get();
        next.set(id + 1);
        id
    });
    PENDING_INLINES.with(|pending| {
        pending
            .borrow_mut()
            .insert(pending_id, PendingInline { owner, region })
    });
    codegen::InlineTripProbe {
        counter_addr,
        threshold: inline_trip_threshold_for(owner_module_bytes),
        trip_fn_ptr: inline_trip_helper_slot() as i64,
        pending_id,
        cells_base_ptr,
        dispatch_cell_index,
    }
}

/// Host entry point publishing [`INLINE_TRIP_HELPER_SLOT`] (called from
/// pyre-jit's `init_jit_hooks` with `wasm_jit_inline_trip as *const () as
/// usize`, which on wasm32 is the function's table index).
pub fn set_inline_trip_helper_slot(slot: u32) {
    INLINE_TRIP_HELPER_SLOT.store(slot as u64, Ordering::Relaxed);
}

/// Current trip-callback table slot (0 = unset).
fn inline_trip_helper_slot() -> u32 {
    INLINE_TRIP_HELPER_SLOT.load(Ordering::Relaxed) as u32
}

/// Host entry point publishing [`CA_DEOPT_HELPER_SLOT`] (called from pyre-jit's
/// `init_jit_hooks` with `wasm_ca_resume_deopt as *const () as usize`, which on
/// wasm32 is the function's table index).
pub fn set_ca_deopt_helper_slot(slot: u32) {
    CA_DEOPT_HELPER_SLOT.store(slot as u64, Ordering::Relaxed);
}

/// Current CA deopt-helper table slot (0 = unset).
pub fn ca_deopt_helper_slot() -> u32 {
    CA_DEOPT_HELPER_SLOT.load(Ordering::Relaxed) as u32
}

/// Publish the residual-call targets whose call descr describes their real
/// wasm ABI, so codegen may lower them to a typed in-module `call_indirect`
/// instead of the `jit_call` host trampoline.
///
/// This is an exact-function allow-list, not a signature inference, for the
/// same reason [`ca_deopt_helper_slot`]'s twin in `pyre-wasm`
/// (`direct_uniform_i64_call`) is one: a descr `Float` does not prove the wasm
/// parameter is `f64`, and an `Int` or `Ref` beside it may be a real `i32`
/// pointer -- `jit_bigint_to_f64_or_inf(&BigInt) -> f64` is published raw and
/// is genuinely `(i32) -> f64`. Emitting a guessed signature traps at the
/// `call_indirect`. A caller vouches for each address by naming the function.
///
/// Addresses are `fn as usize`, which on wasm32 is the table index.
pub fn set_faithful_residual_call_addrs(addrs: &[i64]) {
    FAITHFUL_RESIDUAL_CALL_ADDRS.with(|set| {
        let mut set = set.borrow_mut();
        set.clear();
        set.extend(addrs.iter().copied());
    });
}

thread_local! {
    /// Per thread, and that is the whole process wherever the set is consulted for
    /// real: every registration a shipped build makes is compiled only for wasm32,
    /// and a module instance there is one thread.
    ///
    /// Two properties keep a wider store from being owed anyway. A lookup that
    /// misses is answered by the reflecting host trampoline, so a registration on
    /// one thread and a question on another costs a round trip and never a wrong
    /// signature -- the direction that traps is a spurious *hit*, which per-thread
    /// storage cannot manufacture. And a caller replacing the whole list leaves
    /// every other thread's alone, so two of them may run at once.
    static FAITHFUL_RESIDUAL_CALL_ADDRS: RefCell<std::collections::HashSet<i64>> =
        RefCell::new(std::collections::HashSet::new());
}

/// Vouch for one more callee, leaving the rest of the set alone.
///
/// [`set_faithful_residual_call_addrs`] is the embedder's whole list, declared
/// once. This is for a producer that mints word-spelled call targets as it
/// goes: the address does not exist until the target is built, so it cannot be
/// in a list written ahead of time, and the producer that built it is the one
/// that knows how it is spelled.
pub fn vouch_residual_call_addr(addr: i64) {
    FAITHFUL_RESIDUAL_CALL_ADDRS.with(|set| {
        set.borrow_mut().insert(addr);
    });
}

/// Whether `addr` was vouched for by [`set_faithful_residual_call_addrs`] or
/// [`vouch_residual_call_addr`].
pub(crate) fn residual_call_descr_is_faithful(addr: i64) -> bool {
    FAITHFUL_RESIDUAL_CALL_ADDRS.with(|set| set.borrow().contains(&addr))
}

/// [`vouch_residual_call_addr`] for a target whose *result* is a word too, so
/// its whole wasm signature is the uniform `(i64…) -> i64`.
///
/// The parameter half is what a compiled call needs to know; this is the half
/// [`direct_word_abi_call`] needs, because a caller reaching a target through
/// a raw address has no descr telling it whether the callee returns anything.
/// A void target vouched here would be called as if it returned a word, which
/// is the same type error the vouching exists to avoid.
pub fn vouch_residual_call_addr_returning_word(addr: i64) {
    vouch_residual_call_addr(addr);
    WORD_RESULT_RESIDUAL_CALL_ADDRS.with(|set| {
        set.borrow_mut().insert(addr);
    });
}

thread_local! {
    /// Per thread for the reasons [`FAITHFUL_RESIDUAL_CALL_ADDRS`] is, and it is
    /// read on the thread that wrote it for one more: the producer minting a
    /// word-spelled target and the guest call reaching that target are the same
    /// thread by construction, since the target's address is a table index in the
    /// instance that minted it.
    static WORD_RESULT_RESIDUAL_CALL_ADDRS: RefCell<std::collections::HashSet<i64>> =
        RefCell::new(std::collections::HashSet::new());
}

/// Call a residual target from inside the guest, when its whole signature is
/// the uniform `(i64…) -> i64` and this backend was told so.
///
/// The recording and blackhole paths reach a target through a raw address and
/// no descr, and wasm32 `call_indirect` type-checks the callee: transmuting
/// that address to the signature the residual call carries traps unless the
/// callee really is spelled that way. So those paths hand the call to a host
/// that reflects the callee's declared type first — a guest→host→guest round
/// trip per call. For a target vouched by
/// [`vouch_residual_call_addr_returning_word`] the transmute is exactly right
/// and the round trip buys nothing.
///
/// `None` means the caller still owes the reflecting path: the address was not
/// vouched for, or its arity is past what a residual call can carry.
#[cfg(target_arch = "wasm32")]
pub fn direct_word_abi_call(func_ptr: usize, args: &[i64]) -> Option<i64> {
    if !WORD_RESULT_RESIDUAL_CALL_ADDRS.with(|set| set.borrow().contains(&(func_ptr as i64))) {
        return None;
    }
    // One arm per arity: the transmuted type has to name the parameters, and
    // wasm has no variadic call. `MAX_CALL_ARGS` bounds what a residual call
    // can carry, so an arity past the arms below cannot arrive here.
    macro_rules! arms {
        ($( [$($arg:ident),*] ),* $(,)?) => {
            match args {
                $(
                    [$($arg),*] => {
                        let f: extern "C" fn($(arms!(@i64 $arg)),*) -> i64 =
                            unsafe { std::mem::transmute(func_ptr) };
                        Some(f($(*$arg),*))
                    }
                )*
                _ => None,
            }
        };
        (@i64 $arg:ident) => { i64 };
    }
    arms![
        [],
        [a0],
        [a0, a1],
        [a0, a1, a2],
        [a0, a1, a2, a3],
        [a0, a1, a2, a3, a4],
        [a0, a1, a2, a3, a4, a5],
        [a0, a1, a2, a3, a4, a5, a6],
        [a0, a1, a2, a3, a4, a5, a6, a7],
    ]
}

/// Configure the dormant terminal-decline regression hook.
pub fn set_force_ca_terminal_decline(selector: u64) {
    FORCE_CA_TERMINAL_DECLINE.store(selector, Ordering::Relaxed);
}

/// A legacy pool-indexed const (`ConstInt(u32)` etc.) reached the wasm backend
/// without a value in the constants pool. `set_constants_pool` runs before
/// `assemble`, so every legitimate legacy const is already present; an arg
/// landing here means the optimizer producer failed to seed it. RPython
/// `ConstInt.value` (history.py) is always present, so never register a
/// placeholder `0` — that would emit the constant as zero. Panic at the parity
/// hole, matching the dynasm/cranelift backends.
fn missing_legacy_const(arg: majit_ir::OpRef) -> ! {
    panic!(
        "wasm collect_constants_from_ops: legacy pool-indexed const OpRef \
         (raw={}) is absent from the constants pool — the optimizer producer \
         must seed it (or mint an inline Const) instead of registering 0.",
        arg.raw()
    );
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl WasmBackend {
    pub fn new() -> Self {
        WasmBackend {
            cpu_tracker: std::sync::Arc::new(majit_backend::CpuTotalTracker::default()),
            asm_memory_stats: std::sync::Arc::new(majit_backend::AsmMemoryManagerStats::default()),
            asm_memory_blocks: Vec::new(),
            trace_counter: 0,
            constants: indexmap::IndexMap::new(),
            vtable_offset: None,
        }
    }

    /// Active vtable_offset for wasm codegen.
    pub fn vtable_offset(&self) -> Option<usize> {
        self.vtable_offset
    }

    // `set_constants_pool`, `set_next_trace_id`, and `set_next_header_pc`
    // are provided via the `Backend` trait impl below.

    /// llmodel.py:53-54: store gc_ll_descr on the cpu instance.
    ///
    /// Mirrors `CraneliftBackend::set_gc_allocator`: stores the box in
    /// the wasm thread-local seam and publishes the same five
    /// `ActiveGcGuardHooks` so the backend-agnostic optimizer /
    /// blackhole executor reach the live allocator without taking a
    /// wasm dependency.
    pub fn set_gc_allocator(&mut self, mut gc: Box<dyn majit_gc::GcAllocator>) {
        gc.freeze_types();
        install_gc_box(gc);
    }

    /// No-op: present for API parity with the dynasm backend so
    /// backend-agnostic consumers can call it uniformly. The wasm `New`
    /// allocation path is out of scope for the GC-routed-New opt-in.
    pub fn set_new_via_gc(&mut self, _enabled: bool) {}

    /// llmodel.py:64-69 self.vtable_offset configuration.
    pub fn set_vtable_offset(&mut self, offset: Option<usize>) {
        self.vtable_offset = offset;
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer to its registered GC type id via the
    /// installed gc_ll_descr.
    pub fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        with_wasm_active_gc(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr)).flatten()
    }

    /// Resolve the vtable integer carried by GuardClass /
    /// GuardNonnullClass / GuardSubclass `arg(1)`.
    ///
    /// RPython represents these class operands as `ConstInt` vtable
    /// addresses: `model.py cls_of_box()` returns
    /// `ConstInt(ptr2int(obj.typeptr))`, `virtualstate.py:748` builds
    /// `ConstInt(descr.get_vtable())`, and backends read
    /// `op.getarg(1).getint()` (aarch64/regalloc.py:829). Inline ConstInt
    /// carries the value directly (history.py `ConstInt.value`).
    fn const_class_vtable(&self, arg: majit_ir::OpRef) -> Option<i64> {
        arg.const_int_value()
    }

    /// Pre-compute classptr → expected_typeid pairs for every GuardClass /
    /// GuardNonnullClass operand seen in `ops`. wasm codegen runs without a
    /// borrow of `self`, so we materialize the resolver as a HashMap.
    /// Only GuardClass / GuardNonnullClass need this table — GuardGcType
    /// already carries an immediate typeid (assembler.py:1919-1922) and
    /// GUARD_IS_OBJECT / GUARD_SUBCLASS use a different lookup path.
    fn collect_classptr_typeid_table(&self, ops: &[Op]) -> HashMap<i64, u32> {
        let mut table = HashMap::new();
        if self.vtable_offset.is_some() {
            return table;
        }
        if with_wasm_active_gc(|_| ()).is_none() {
            return table;
        }
        for op in ops {
            if matches!(
                op.opcode,
                majit_ir::OpCode::GuardClass | majit_ir::OpCode::GuardNonnullClass
            ) && op.num_args() >= 2
                && let Some(classptr) = self.const_class_vtable(op.arg(1).to_opref())
                && let Some(tid) = self.lookup_typeid_from_classptr(classptr as usize)
            {
                table.insert(classptr, tid);
            }
        }
        table
    }

    /// Pre-fetch `GuardGcTypeInfo` from the installed `gc_ll_descr`.
    ///
    /// Mirrors the `self.cpu.gc_ll_descr.get_translated_info_*` /
    /// `cpu.subclassrange_min_offset` lookups that RPython's
    /// `genop_guard_guard_is_object` (x86/assembler.py) and
    /// `genop_guard_guard_subclass` (x86/assembler.py) do at
    /// codegen time. The returned struct is handed to
    /// `codegen::build_wasm_module`; the codegen arms assert
    /// `supports_guard_gc_type` before reading any other field.
    ///
    /// Also pre-computes `(subclassrange_min, subclassrange_max)` for
    /// every constant classptr argument of a `GuardSubclass` op
    /// (assembler.py:1971-1974 reads these bounds at codegen time).
    fn collect_guard_gc_type_info(&self, ops: &[Op]) -> codegen::GuardGcTypeInfo {
        with_wasm_active_gc(|gc| {
            let mut info = codegen::GuardGcTypeInfo {
                supports_guard_gc_type: gc.supports_guard_gc_type(),
                ..codegen::GuardGcTypeInfo::default()
            };
            if !info.supports_guard_gc_type {
                return info;
            }
            // assembler.py:1934-1937: gc_ll_descr lookups.
            let (base, shift, sizeof_ti) = gc.get_translated_info_for_typeinfo();
            info.base_type_info = base;
            info.shift_by = shift;
            info.sizeof_ti = sizeof_ti;
            let (infobits_off, is_object_flag) = gc.get_translated_info_for_guard_is_object();
            info.infobits_offset = infobits_off;
            info.is_object_flag = is_object_flag;
            // assembler.py:1951: cpu.subclassrange_min_offset.
            info.subclassrange_min_offset = gc.subclassrange_min_offset();
            // assembler.py:1971-1974: (subclassrange_min, subclassrange_max)
            // for every constant GuardSubclass arg1.
            for op in ops {
                if op.opcode == majit_ir::OpCode::GuardSubclass
                    && op.num_args() >= 2
                    && let Some(classptr) = self.const_class_vtable(op.arg(1).to_opref())
                    && let Some(range) = gc.subclass_range(classptr as usize)
                {
                    info.subclass_ranges.insert(classptr, range);
                }
            }
            info
        })
        .unwrap_or_default()
    }

    /// Pull every reference constant out of `ops` into a per-loop `GcTable`
    /// and replace it with a `LoadFromGcTable` of its slot
    /// (`majit_gc::rewrite::remove_ref_constants`, rewrite.py).
    ///
    /// A `GcRef` baked as a code immediate is invisible to the moving
    /// collector: the first minor collection that promotes the referenced
    /// object out of the nursery leaves the immediate pointing into nursery
    /// space that is later reused or zeroed by `reset`. The table slot is a
    /// GC root the collector forwards in place, so the emitted load always
    /// reads the object at its current address. Returns `None` for a trace
    /// with no reference constant, leaving the module byte-identical.
    fn intern_ref_constants(
        inputargs: &[InputArg],
        ops: Vec<Op>,
    ) -> (Vec<Op>, Option<Arc<majit_gc::GcTable>>) {
        let next_pos = codegen::next_value_pos(inputargs, &ops);
        let (ops, gcrefs) = majit_gc::rewrite::remove_ref_constants(&ops, next_pos);
        let table = (!gcrefs.is_empty()).then(|| majit_gc::GcTable::from_gcrefs(&gcrefs));
        (ops, table)
    }

    /// `x86/assembler.py` `gcreftracers.append(tracer)` — keep the
    /// per-loop table alive for as long as the compiled trace that bakes its
    /// base address. `LIVE_GC_TABLES` holds only a `Weak`, so this strong
    /// reference is what keeps the slots rooted and forwardable.
    fn register_gc_table(token: &JitCellToken, table: Arc<majit_gc::GcTable>) {
        if let Some(clt) = token.compiled_loop_token() {
            let tracer: Arc<dyn std::any::Any + Send + Sync> = table;
            clt.asmmemmgr_gcreftracers.lock().push(tracer);
        }
    }

    /// Validate that every constant OpRef appearing as an arg is resolvable.
    ///
    /// Inline-Const variants (`ConstInt`/`ConstFloat`/
    /// `ConstPtr`) carry `.value` on the OpRef itself (history.py:
    /// 227/268/314), so they need no `self.constants` side-table entry and
    /// are skipped. A legacy idx-keyed `ConstInt(u32)` / `ConstFloat(u32)` /
    /// `ConstPtr(u32)` must have been seeded by `set_constants_pool`; one that
    /// is missing is a producer gap and panics rather than defaulting to 0.
    fn collect_constants_from_ops(&mut self, ops: &[Op]) {
        for op in ops {
            for arg in op.getarglist().iter() {
                let arg = arg.to_opref();
                if arg.is_constant()
                    && arg.inline_const_bits().is_none()
                    && !self.constants.contains_key(&arg.raw())
                {
                    missing_legacy_const(arg);
                }
            }
            if let Some(fail_args) = op.getfailargs() {
                for arg in fail_args.iter() {
                    let arg = arg.to_opref();
                    if arg.is_constant()
                        && arg.inline_const_bits().is_none()
                        && !self.constants.contains_key(&arg.raw())
                    {
                        missing_legacy_const(arg);
                    }
                }
            }
        }
    }

    /// Merge a deferred region into its owner, for a bridge that has been
    /// entered [`INLINE_TRIP_THRESHOLD`] times.
    ///
    /// The trip itself only queued the id ([`record_inline_trip`]); this runs
    /// from the host once the trace has returned.
    ///
    /// A candidate that no longer qualifies — an invalidated owner, a loop that
    /// has since taken a region for the same guard — is dropped rather than
    /// retried: the bridge is already installed and correct, so the only thing
    /// lost is the merge.
    pub fn install_pending_inline(&mut self, pending_id: i64) {
        let Some(pending) = PENDING_INLINES.with(|p| p.borrow_mut().remove(&pending_id)) else {
            return;
        };
        diag_bump(55);
        let PendingInline { owner, region } = pending;
        let source_fail_index = region.source_fail_index;
        if !self.install_inline_region(&owner, region) {
            // The probe cleared the dispatch cell to get here. A merge that
            // does not install leaves the out-of-line bridge as the only route
            // to that guard, so put the cell back rather than leave the guard
            // bailing to the host for the rest of the run.
            Self::restore_dispatch_cell(&owner, source_fail_index);
        }
    }

    /// Re-point a guard's dispatch cell at the bridge `bridge_slots` still
    /// names for it.
    fn restore_dispatch_cell(owner: &JitCellToken, source_fail_index: u32) {
        let Some(source_loop) = owner
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
        else {
            return;
        };
        let cells_base = source_loop.bridge_cells_base.get();
        let Some(slot) = source_loop
            .bridge_slots
            .borrow()
            .get(&source_fail_index)
            .copied()
        else {
            return;
        };
        let _ = (cells_base, slot);
        #[cfg(target_arch = "wasm32")]
        if cells_base != 0 {
            let cell = (cells_base as usize + source_fail_index as usize * 4) as *mut u32;
            unsafe { core::ptr::write(cell, slot) };
        }
    }

    /// Rebuild `owner` with `region` merged into it. `false` leaves the owner
    /// exactly as it was, for a caller that still has an out-of-line bridge to
    /// fall back on.
    fn install_inline_region(
        &mut self,
        owner: &JitCellToken,
        mut region: codegen::InlinedBridge,
    ) -> bool {
        if owner.is_invalidated() {
            diag_bump(50);
            return false;
        }
        let Some(source_loop) = owner
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
        else {
            return false;
        };
        let Some(mut candidate) = source_loop.reemit.borrow().as_ref().cloned() else {
            diag_bump(35);
            return false;
        };
        let source_fail_index = region.source_fail_index;
        if candidate
            .inlined_bridges
            .iter()
            .any(|r| r.source_fail_index == source_fail_index)
        {
            diag_bump(36);
            return false;
        }
        // Re-decided here rather than carried: the candidate may have taken
        // more regions since, and the placement depends on them.
        region.outside_loop =
            codegen::source_guard_precedes_loop_label(&candidate.ops, source_fail_index)
                || candidate.inlined_bridges.iter().any(|r| r.outside_loop);
        if region.outside_loop {
            diag_bump(52);
        }
        candidate.inlined_bridges.push(region);
        let mut merged_ops = candidate.ops.clone();
        for region in &candidate.inlined_bridges {
            merged_ops.extend(region.ops.iter().cloned());
        }
        // `candidate.constants` keeps the pool of the compile that recorded
        // these ops -- the owner's. The pool is keyed by value position, and
        // the merge rebases every region off the owner's ids, so a region's
        // own entries reach the build through `InlinedBridge::constants` and
        // are replayed at the rebase offset. Assigning the merging bridge's
        // pool here instead would leave the owner's window described by
        // another trace's keys: an owner-only folded value loses its entry,
        // and one the bridge happens to number the same way is seeded with the
        // wrong bits.
        candidate.classptr_to_typeid = self.collect_classptr_typeid_table(&merged_ops);
        candidate.guard_gc_type_info = self.collect_guard_gc_type_info(&merged_ops);
        candidate.nursery = nursery_alloc_params(&merged_ops);
        // The merged region supersedes the bridge's own dispatch cell. Remove
        // it before reemit so the fresh array cannot replay a contradictory
        // slot — the bridge on the stack right now finishes its pass either
        // way, and nothing enters it again.
        let source_cells_base = source_loop.bridge_cells_base.get();
        let old_bridge_slot = source_loop
            .bridge_slots
            .borrow_mut()
            .remove(&source_fail_index);
        #[cfg(target_arch = "wasm32")]
        if source_cells_base != 0 {
            let cell = (source_cells_base as usize + source_fail_index as usize * 4) as *mut u32;
            unsafe { core::ptr::write(cell, 0) };
        }
        // Eligibility IS the emission: `reemit_loop` runs the same
        // `build_wasm_module` over the same candidate, and nothing it does
        // before that call mutates state a failure would have to unwind — it
        // reads the fail-index base and allocates a cell array that is dropped
        // on the error path. So install directly and let the build answer,
        // instead of asking it once as a trial and once for real.
        let old_inputs = source_loop.reemit.replace(Some(candidate));
        match self.reemit_loop(owner) {
            Ok(()) => {
                diag_bump(31);
                diag_bump(32);
                // The region runs from the owner's module, so its
                // `GUARD_NOT_INVALIDATED` reads the owner's root flag. Name that
                // as this compile's generation, or the quasi-immutable
                // dependencies registered afterwards attach to a flag the merged
                // code never loads and a mutated field leaves the fold in place.
                owner.record_bridge_invalidation_flag(owner.invalidation_flag());
                return true;
            }
            Err(error) => {
                source_loop.reemit.replace(old_inputs);
                if let Some(slot) = old_bridge_slot {
                    source_loop
                        .bridge_slots
                        .borrow_mut()
                        .insert(source_fail_index, slot);
                    #[cfg(target_arch = "wasm32")]
                    if source_cells_base != 0 {
                        let cell = (source_cells_base as usize + source_fail_index as usize * 4)
                            as *mut u32;
                        unsafe { core::ptr::write(cell, slot) };
                    }
                }
                record_inline_trial_error(&error);
                classify_inline_install_error(&error);
            }
        }
        let _ = source_cells_base;
        false
    }

    /// Rebuild a loop module and install it into its original shared-table
    /// slot. The retained inputs are post-intern, so this does not allocate a
    /// second GC reference table or change any reference-constant immediate.
    #[allow(unreachable_code, unused_variables)]
    pub fn reemit_loop(&mut self, token: &JitCellToken) -> Result<(), BackendError> {
        let compiled = token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
            .ok_or_else(|| {
                BackendError::Unsupported("wasm backend: no compiled loop to re-emit".into())
            })?;
        let Some(mut inputs) = compiled.reemit.borrow().as_ref().cloned() else {
            return Err(BackendError::Unsupported(
                "wasm backend: entry bridge is not re-emittable".into(),
            ));
        };
        let old_handle = compiled.eager_func_handle();
        if old_handle == 0 {
            return Err(BackendError::Unsupported(
                "wasm backend: unmaterialized loop is not re-emittable".into(),
            ));
        }

        // `guard_exit_count` walks the whole op list it is handed, so each
        // count is taken once here: the exit loop below needs the per-region
        // counts for every one of its exits, and re-deriving them there would
        // walk every appended region once per guard.
        let own_guard_count = codegen::guard_exit_count(&inputs.inputargs, &inputs.ops);
        let region_guard_counts: Vec<usize> = inputs
            .inlined_bridges
            .iter()
            .map(|region| codegen::guard_exit_count(&region.inputargs, &region.ops))
            .collect();
        let merged_guard_count = own_guard_count + region_guard_counts.iter().sum::<usize>();
        inputs.fail_index_base = reserve_fail_descrs(merged_guard_count);
        let (new_cells_base, new_cells_owner) = codegen::alloc_bridge_cells(merged_guard_count);
        inputs.bridge_cells_base = new_cells_base;
        let (wasm_bytes, guard_exits, _) = codegen::build_wasm_module(&inputs)?;
        let code_size = wasm_bytes.len();
        let descrs: Vec<Arc<WasmFailDescr>> = guard_exits
            .iter()
            .enumerate()
            .map(|(index, g)| {
                let mut region_start = own_guard_count;
                let trace_id = inputs
                    .inlined_bridges
                    .iter()
                    .zip(&region_guard_counts)
                    .find_map(|(region, &count)| {
                        let contains = (region_start..region_start + count).contains(&index);
                        region_start += count;
                        contains.then_some(region.trace_id)
                    })
                    .unwrap_or(compiled.trace_id);
                Arc::new(WasmFailDescr {
                    fail_index: g.fail_index,
                    trace_id,
                    fail_arg_types: g.fail_arg_types.clone(),
                    is_finish: g.is_finish,
                    meta_descr: g.meta_descr.clone(),
                })
            })
            .collect();

        #[cfg(target_arch = "wasm32")]
        if glue::replace_module(old_handle, &wasm_bytes) != old_handle {
            return Err(BackendError::Unsupported(
                "wasm host rejected the re-emitted trace module".into(),
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = wasm_bytes;
            return Err(BackendError::Unsupported(
                "wasm backend: no host replacement binding".into(),
            ));
        }

        // The host has accepted the replacement, so its newly encoded global
        // indices can now be made visible in the registry and local metadata.
        // Both instances remain resident, so account for the replacement block
        // in the same lifetime ledger as an ordinary compiled module.
        let block = self.asm_memory_stats.record_block(code_size, code_size);
        self.asm_memory_blocks.push(block);
        // Keep still-standalone bridge descriptors after the rebuilt merged
        // prefix. Adding regions grows that prefix, so every old positional
        // range moves by exactly the difference in guard-cell counts.
        let old_guard_count = compiled.num_guard_cells.get();
        let chained_descrs = compiled.fail_descrs.borrow()[old_guard_count..].to_vec();
        let mut replacement_descrs = descrs.clone();
        replacement_descrs.extend(chained_descrs);
        *compiled.fail_descrs.borrow_mut() = replacement_descrs;
        register_fail_descrs(&descrs);
        let guard_growth = guard_exits.len().saturating_sub(old_guard_count);
        if guard_growth != 0 {
            for (_, _, start, _) in compiled.bridge_descr_ranges.borrow_mut().iter_mut() {
                *start += guard_growth;
            }
        }
        #[cfg(target_arch = "wasm32")]
        if new_cells_base != 0 {
            for (&fail_index, &bridge_slot) in compiled.bridge_slots.borrow().iter() {
                let cell = (new_cells_base as usize + fail_index as usize * 4) as *mut u32;
                unsafe { core::ptr::write(cell, bridge_slot) };
            }
        }
        if let Some(owner) = new_cells_owner {
            compiled._bridge_owned_cells.borrow_mut().push(owner);
        }
        compiled.bridge_cells_base.set(new_cells_base);
        compiled.module_bytes.set(code_size as u32);
        compiled.num_guard_cells.set(guard_exits.len());
        {
            let mut metas = compiled.chained_trace_meta.borrow_mut();
            let mut offset = own_guard_count;
            for region in &inputs.inlined_bridges {
                let count = codegen::guard_exit_count(&region.inputargs, &region.ops);
                let exits = &guard_exits[offset..offset + count];
                // This region's guards are carved out of the array that was
                // just reallocated, so every bridge already chained onto one of
                // them has lost its dispatch entry. Unreplayed, that guard
                // deopts to the tracer on every failure and retraces a bridge
                // it can never reach.
                #[cfg(target_arch = "wasm32")]
                if new_cells_base != 0 {
                    for (&(trace_id, fail_index), &bridge_slot) in
                        compiled.chained_bridge_slots.borrow().iter()
                    {
                        if trace_id != region.trace_id || fail_index as usize >= count {
                            continue;
                        }
                        let cell = (new_cells_base as usize + (offset + fail_index as usize) * 4)
                            as *mut u32;
                        unsafe { core::ptr::write(cell, bridge_slot) };
                    }
                }
                metas.insert(
                    region.trace_id,
                    ChainedTraceMeta {
                        cells_base: new_cells_base + offset as u32 * 4,
                        num_cells: count,
                        guard_fail_arg_advanced: guard_fail_args_advanced(&region.ops, exits),
                        guard_fail_arg_counts: exits
                            .iter()
                            .map(|guard| {
                                crate::codegen::live_fail_arg_count(
                                    guard.meta_descr.as_ref(),
                                    guard.fail_arg_refs.len(),
                                )
                            })
                            .collect(),
                        bridge_param_dispatch: inputs.bridge_param_dispatch,
                    },
                );
                offset += count;
            }
        }
        *compiled.reemit.borrow_mut() = Some(inputs.clone());

        // LABEL targets bake only the stable table slot, so restamp them for
        // this build. CA dispatch additionally carries the new finish index.
        let _ = stamp_and_publish_label_targets(
            old_handle,
            compiled.frame,
            &inputs.inputargs,
            &inputs.ops,
            inputs.bridge_entry_arity,
        );
        if let Some(mut target) = call_assembler_target(token.number) {
            target.func_handle = old_handle;
            target.compiled_ptr = compiled as *const CompiledWasmLoop as usize as u64;
            ca_dispatch_publish(
                token.number,
                old_handle,
                target.compiled_ptr as u32,
                target.callee_frame_bytes,
                target.dispatch_key_ofs as u32,
                target.callee_gcmap_ptr,
            );
            publish_call_assembler_target(token.number, target);
        }
        Ok(())
    }
}

unsafe impl Send for WasmBackend {}

/// Stamp a position onto every non-Void-result op left unpositioned by the
/// optimizer, so no operand resolves to `OpRef::NONE` during codegen.
///
/// The optimizer's force path emits materialized allocation/store ops (e.g. a
/// virtualized list's `NewArray` backing block and its `SetfieldGc` /
/// `SetarrayitemGc` stores) with `Op::new`, and only assigns a position to ops
/// whose `result_type() != Void` — a Void-result store keeps `pos == NONE`.
/// A later op that consumes such a producer's result reads its `pos` through
/// `Operand::Op`, and an unpositioned producer yields `OpRef::NONE`
/// (`raw() == u32::MAX`), which `emit_resolve` would use to index `value_types`
/// out of bounds. The native backends normalize positions before codegen
/// (dynasm `prepare_ops_for_compile`, cranelift `normalize_ops_for_codegen_simple`);
/// the wasm backend does the same here.
fn normalize_ops_for_codegen(inputargs: &[InputArg], ops: &[OpRc]) -> Vec<Op> {
    let num_inputs = inputargs.len() as u32;
    ops.iter()
        .enumerate()
        .map(|(op_idx, op)| {
            let normalized = (**op).clone();
            let rt = normalized.result_type();
            if rt != majit_ir::Type::Void && normalized.pos.get().is_none() {
                normalized
                    .pos
                    .set(majit_ir::OpRef::op_typed(num_inputs + op_idx as u32, rt));
            }
            normalized
        })
        .collect()
}

/// Report why a trace cannot be compiled by the wasm backend, or `None` if it
/// can. Declined traces fall back to the interpreter (correct, unaccelerated)
/// instead of producing an invalid trace module. `allow_ca` (set when every
/// CALL_ASSEMBLER target is admitted) lifts the CALL_ASSEMBLER decline so the
/// CA arm (guest→guest `call_indirect`) lowers it instead.
///
/// A JUMP whose target token is not defined by a local LABEL — the cross-loop
/// terminal jump — is not judged here: it is lowered to
/// `return_call_indirect(external_jump_slot)` and both callers resolve that
/// slot through [`resolve_cross_loop_jump_target`].
fn wasm_unsupported_trace_reason(ops: &[Op], allow_ca: bool) -> Option<String> {
    for op in ops {
        if op.opcode.is_call_assembler() && !allow_ca {
            // CALL_ASSEMBLER inlines a loop-bearing callee by jumping into another
            // trace's compiled token. `general_int_call_assembler_target` resolves
            // that token to a published guest function the CA arm reaches with a
            // `call_indirect`, so reaching here means some target did not resolve
            // — an unpublished token, a signature outside Int/Ref, or a missing
            // deopt-helper slot — and there is nothing to call.
            return Some(format!(
                "wasm backend: {:?} (loop-callee inline)",
                op.opcode
            ));
        }
    }
    None
}

/// Whether a trace has a `JUMP` whose target token is not defined by one of
/// this compilation's LABELs, so codegen lowers it to a tail call into another
/// module rather than a `br`. This is the token test from
/// `x86/assembler.py:2463`. Testing only whether the trace had any LABEL was
/// wrong: a trace may define LABELs and still close onto a token from another
/// compilation, as a retrace attached as a bridge does.
fn has_cross_loop_terminal_jump(ops: &[Op]) -> bool {
    let has_jump = ops.iter().any(|op| op.opcode == majit_ir::OpCode::Jump);
    has_jump && codegen::find_loop_label_index(ops).is_none()
}

/// Resolve the re-entry target of a cross-loop terminal JUMP BY DESCR IDENTITY
/// through the `LABEL_TARGETS` registry — the JUMP and its target LABEL share
/// the loop-target descr Arc, and every compiled loop published its enterable
/// labels there. The stamped `label_block_id` ordinal is NOT identity: a
/// retraced loop has several sibling specializations whose start labels all
/// carry ordinal 0, and a trace legitimately closes into a SIBLING
/// (jump-to-existing-trace) — the registry resolves the owning module's table
/// slot and resume key, so the tail call chains into the RIGHT loop.
///
/// Decline (`None`, after tallying which question answered) when the target is
/// unpublished (descr stripped, or its loop declined/was dropped), the JUMP
/// arity differs from the label's arg count (the resume loader reads exactly
/// that many positional frame slots), or the label's args are not the complete
/// live set of the target trace's remainder (`resume_safe` — resuming there
/// would read a null local).
///
/// `source` is the frame a chained bridge already runs on, with the table slot
/// of the loop that owns it. `compile_bridge` supplies it: that bridge shares
/// the source token's frozen layout, so the target's geometry must agree with
/// it exactly, and a target whose backend capture slots were filled by its own
/// fall-through (`requires_own_frame`) is resumable only when it IS that source
/// loop. `compile_loop` passes `None` for an entry bridge, which owns its
/// module and its frame: it has no source slot, so `requires_own_frame` always
/// declines, and instead of matching a geometry it ADOPTS the target's (the
/// caller checks its own slots fit, then compiles against `t.frame`).
fn resolve_cross_loop_jump_target(
    ops: &[Op],
    source: Option<(u32, codegen::FrameGeometry)>,
) -> Option<LabelTarget> {
    let closing_jump = ops
        .iter()
        .rev()
        .find(|op| op.opcode == majit_ir::OpCode::Jump);
    let target_descr_id = closing_jump
        .and_then(|j| j.getdescr())
        .map(|d| std::sync::Arc::as_ptr(&d) as *const () as usize)
        .filter(|id| *id != 0);
    let target = target_descr_id.and_then(label_target);
    let arity = closing_jump.map_or(0, |j| j.getarglist().len());
    match target {
        // Descr stripped, or the target label was never published.
        None => {
            diag_bump(8);
            diag_bump(if target_descr_id.is_none() { 17 } else { 18 });
            None
        }
        Some(t) if arity != t.num_args => {
            diag_bump(10); // arity mismatch
            None
        }
        Some(t) if !t.resume_safe => {
            diag_bump(9); // label args not the full live set
            None
        }
        Some(t) if t.requires_own_frame && Some(t.func_handle) != source.map(|(slot, _)| slot) => {
            // The target's high capture homes were populated by its own
            // fall-through path, not by this sibling source loop.
            diag_bump(9);
            None
        }
        Some(t) if source.is_some_and(|(_, frame)| t.frame != frame) => {
            diag_bump(4); // target uses different frozen frame offsets
            None
        }
        Some(t) => Some(t),
    }
}

/// Resolve every distinct compiled target used by CALL_ASSEMBLER ops in this
/// trace.  PyPy's `compile_tmp_callback` always supplies a real compiled token
/// while the final loop is pending, so every target here must likewise be an
/// installed `CompiledWasmLoop`; there is no bodyless self-placeholder case.
fn general_int_call_assembler_target(ops: &[Op]) -> Option<Vec<(u64, CallAssemblerTarget)>> {
    let mut resolved = Vec::new();
    let mut saw_ca = false;
    for op in ops.iter().filter(|op| op.opcode.is_call_assembler()) {
        saw_ca = true;
        if !matches!(
            op.opcode,
            majit_ir::OpCode::CallAssemblerI | majit_ir::OpCode::CallAssemblerR
        ) {
            diag_bump(57);
            return None;
        }
        let Some(descr_ref) = op.getdescr() else {
            diag_bump(58);
            return None;
        };
        let Some(descr) = descr_ref.as_call_descr() else {
            diag_bump(58);
            return None;
        };
        let arg_types = descr.arg_types();
        if !arg_types
            .iter()
            .all(|&tp| matches!(tp, majit_ir::Type::Int | majit_ir::Type::Ref))
            || !matches!(
                descr.result_type(),
                majit_ir::Type::Int | majit_ir::Type::Ref
            )
        {
            diag_bump(59);
            return None;
        }
        let Some(target_token) = descr.call_target_token() else {
            diag_bump(58);
            return None;
        };
        let Some(mut registered) = call_assembler_target(target_token) else {
            diag_bump(60);
            return None;
        };
        // A straight-line function trace may have deferred host module
        // compilation. CALL_ASSEMBLER is its first real consumer, so
        // materialize it before publishing the stable dispatch entry.
        if registered.func_handle == 0 && registered.compiled_ptr != 0 {
            let Some(loop_) =
                (unsafe { (registered.compiled_ptr as *const CompiledWasmLoop).as_ref() })
            else {
                diag_bump(61);
                return None;
            };
            let Ok(handle) = loop_.materialize_func_handle() else {
                diag_bump(61);
                return None;
            };
            if handle == 0 {
                diag_bump(61);
                return None;
            }
            registered.func_handle = handle;
            ca_dispatch_publish(
                target_token,
                handle,
                registered.compiled_ptr as u32,
                registered.callee_frame_bytes,
                registered.dispatch_key_ofs as u32,
                registered.callee_gcmap_ptr,
            );
            publish_call_assembler_target(target_token, registered.clone());
        }
        if registered.input_types.as_slice() != arg_types {
            diag_bump(62);
            return None;
        }
        if registered.callee_frame_bytes == 0 {
            diag_bump(62);
            return None;
        }
        if registered.callee_gcmap_ptr == 0 {
            diag_bump(62);
            return None;
        }
        if registered.compiled_ptr == 0 {
            diag_bump(62);
            return None;
        }
        // A successfully compiled loop is retained by its token while it is
        // registered. It can subsequently become terminally declined, so read
        // the live state before baking every CA entry.
        let live = unsafe {
            (registered.compiled_ptr as *const CompiledWasmLoop)
                .as_ref()
                .is_some_and(|loop_| !loop_.ca_terminal_declined.get())
        };
        if !live {
            diag_bump(63);
            return None;
        }
        // The same target may occur in several operations; each operation was
        // validated above, while the codegen map needs one geometry per token.
        if !resolved
            .iter()
            .any(|(known_token, _)| *known_token == target_token)
        {
            resolved.push((target_token, registered));
        }
    }
    (saw_ca && !resolved.is_empty()).then_some(resolved)
}

fn bridge_int_call_assembler_target(ops: &[Op]) -> Option<Vec<(u64, CallAssemblerTarget)>> {
    general_int_call_assembler_target(ops)
}

fn ca_codegen_targets(
    targets: &[(u64, CallAssemblerTarget)],
) -> std::collections::HashMap<u64, codegen::CaTarget> {
    targets
        .iter()
        .map(|(token, _target)| {
            (
                *token,
                codegen::CaTarget {
                    dispatch_entry: ca_dispatch_slot(*token),
                },
            )
        })
        .collect()
}

fn ca_max_frame_bytes(targets: &[(u64, CallAssemblerTarget)]) -> u32 {
    targets
        .iter()
        .map(|(_, target)| target.callee_frame_bytes)
        .max()
        .expect("admitted CALL_ASSEMBLER targets must be non-empty")
}

fn mark_call_assembler_target_active(
    target: &CallAssemblerTarget,
    caller_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    // `caller_flag` is the invalidation flag the calling artifact's
    // `GUARD_NOT_INVALIDATED` reads — the token flag for a loop, the
    // bridge-generation flag for a bridge — so a terminal decline of the
    // callee invalidates exactly the artifact embedding the CA edge.
    // The target metadata is removed by `CompiledWasmLoop::drop`; compilation
    // is single-threaded, and callers only retain the pointer while the token
    // remains compiled. This is the same lifetime used by the deopt helper.
    let force_terminal_decline = unsafe {
        if let Some(loop_) = (target.compiled_ptr as *const CompiledWasmLoop).as_ref() {
            loop_.ca_active.set(true);
            {
                let mut callers = loop_.ca_callers.borrow_mut();
                if !callers
                    .iter()
                    .any(|known| std::sync::Arc::ptr_eq(known, &caller_flag))
                {
                    callers.push(caller_flag);
                }
            }

            // Runtime-regression hook for the terminal-decline CA path.  It
            // is dormant unless explicitly selected, and runs only after this
            // caller has already admitted and compiled a CA edge.  `1` selects
            // the first such target; a decimal JitCellToken number selects a
            // particular target.  The caller's invalidation bit still makes
            // this a bounded window, exactly like a real terminal bridge
            // decline.
            let selector = FORCE_CA_TERMINAL_DECLINE.load(Ordering::Relaxed);
            if selector != 0 && (selector == 1 || selector == target.token_number) {
                // One forced target per guest run. A real terminal decline
                // also transitions its target just once.
                FORCE_CA_TERMINAL_DECLINE.store(0, Ordering::Relaxed);
                true
            } else {
                false
            }
        } else {
            false
        }
    };
    if force_terminal_decline {
        // `mark_call_assembler_terminal_decline` reads `ca_callers`; release
        // the registration borrow above before invalidating those callers.
        mark_call_assembler_terminal_decline(target.compiled_ptr as usize);
        diag_bump(16);
    }
}

/// Move the movable-CA caller census from a redirected target to its
/// replacement. Existing callers retain the old dispatch entry, but terminal
/// decline of the replacement must still invalidate those callers.
fn transfer_call_assembler_target_activity(
    old_target: &CallAssemblerTarget,
    new_target: &CallAssemblerTarget,
) {
    unsafe {
        let Some(old_loop) = (old_target.compiled_ptr as *const CompiledWasmLoop).as_ref() else {
            return;
        };
        let Some(new_loop) = (new_target.compiled_ptr as *const CompiledWasmLoop).as_ref() else {
            return;
        };

        new_loop
            .ca_active
            .set(new_loop.ca_active.get() || old_loop.ca_active.get());
        let old_callers = old_loop.ca_callers.borrow().clone();
        let mut new_callers = new_loop.ca_callers.borrow_mut();
        for caller in old_callers {
            if !new_callers
                .iter()
                .any(|known| std::sync::Arc::ptr_eq(known, &caller))
            {
                new_callers.push(caller);
            }
        }
    }
}

/// Mark a CA target whose callee guard was structurally declined.  The host
/// deopt helper calls this only after the exact guard descriptor was marked
/// terminally declined; invalidating the callers forces a retrace whose
/// admission check above restores the plain call path.
pub fn mark_call_assembler_terminal_decline(compiled_ptr: usize) {
    unsafe {
        let Some(loop_) = (compiled_ptr as *const CompiledWasmLoop).as_ref() else {
            return;
        };
        if loop_.ca_terminal_declined.replace(true) {
            return;
        }
        for caller in loop_.ca_callers.borrow().iter() {
            caller.store(true, Ordering::Release);
        }
    }
}

/// Exit slots to decode out of a returned frame for `fail_descr`.
///
/// Its fail arguments, plus the GUARD_VALUE operand the exit spills one slot
/// past them when the guard does not carry it as a fail argument
/// (`codegen::counter_value_spill`). That trailing word is what
/// `resolve_guard_value_operand` reads back through `get_value_direct` for
/// `make_a_counter_per_value`; it is never a fail argument, so it stays out of
/// `fail_arg_types` and out of every typed exit decode.
fn exit_slot_count(fail_descr: &failguard::WasmFailDescr) -> usize {
    let fail_args = fail_descr.fail_arg_types.len();
    fail_descr
        .meta_descr
        .as_ref()
        .and_then(|d| d.as_fail_descr())
        .and_then(majit_backend::guard_value_counter_slot)
        .map_or(fail_args, |slot| fail_args.max(slot + 1))
}

/// Reconstruct a [`DeadFrame`] from a callee frame an in-guest `call_indirect`
/// already ran to a guard/finish exit (the self-recursive CALL_ASSEMBLER fast
/// path, `PYRE_WASM_CA`). This is the post-`glue::execute` tail of
/// [`WasmBackend::execute_token`] factored for a frame the host did not itself
/// enter: `frame[0]` holds the exit `fail_index`, `frame[1..]` the exit slots,
/// and the pending-exception cell is captured with `jit_exc_take` exactly as
/// `execute_token` does after a GuardNoException / GuardException exit.
/// `pyre-jit`'s `call_jit::wasm_ca_resume_deopt` calls this, then drives the
/// resulting `DeadFrame` through the same `get_latest_descr_arc` /
/// `get_*_value` / `grab_exc_value` Backend path the host's outermost deopt
/// handling uses, so the in-guest deopt completes identically.
///
/// `frame[0]` resolves through the GLOBAL fail-index space
/// (`failguard::global_fail_descr`) — the exit may belong to a bridge chained
/// past the source loop. `_compiled_ptr` (the source loop's metadata address,
/// baked into the CA arm) is kept in the trace ABI but no longer consulted.
pub fn dead_frame_from_ran_frame(_compiled_ptr: usize, frame_ptr: usize) -> DeadFrame {
    let frame = frame_ptr as *const i64;
    let exc_value = jit_exc_take();
    let fail_index = unsafe { *frame } as u32;
    let fail_descr =
        global_fail_descr(fail_index).expect("invalid fail_index from in-guest CA callee frame");
    let num_outputs = exit_slot_count(&fail_descr);
    let raw_values: Vec<i64> = (0..num_outputs)
        .map(|i| unsafe { *frame.add(1 + i) })
        .collect();
    DeadFrame::Boxed(WasmFrameData::boxed(raw_values, fail_descr, exc_value))
}

/// Reconstruct a [`DeadFrame`] for a frame a FORCE interrupted while its call
/// is still on the stack, from the coordinate `emit_force_bracket_before_call`
/// published into it: `frame[0]` the bracketing GUARD_NOT_FORCED's exit index,
/// `frame[1..]` that guard's fail arguments.
///
/// Twin of [`dead_frame_from_ran_frame`] with one difference: a force is not an
/// exit, so it must not consume the pending-exception cell. `jit_exc_take`
/// clears what it reads, and the frame this force interrupted goes on running
/// afterwards -- draining the cell here would lose an exception the trace has
/// not delivered yet.
/// The data region of the frame a force token names — the address the trace
/// itself carries in local 0.
fn forced_frame_items_base(force_token: GcRef) -> usize {
    force_token.0 + majit_backend::jitframe::FIRST_ITEM_OFFSET
}

fn dead_frame_from_forced_frame(frame_ptr: usize) -> DeadFrame {
    let frame = frame_ptr as *const i64;
    let fail_index = unsafe { *frame } as u32;
    let fail_descr =
        global_fail_descr(fail_index).expect("invalid fail_index from a forced wasm frame");
    let num_outputs = exit_slot_count(&fail_descr);
    let types = fail_descr.fail_arg_types.as_slice();
    let raw_values: Vec<i64> = (0..num_outputs)
        .map(|i| {
            let word = unsafe { *frame.add(1 + i) };
            // `emit_force_arm` publishes a Ref argument as `home_offset * 2 + 1`
            // so the value is read out of the traced home slot a collection
            // inside the bracketed call forwards, rather than out of an
            // untraced copy in the exit slot. A literal is even (Ref pointers
            // are 8-aligned; a null and a non-Ref argument are published as
            // themselves).
            if types.get(i) == Some(&majit_ir::Type::Ref) && word & 1 == 1 {
                unsafe { *((frame_ptr + (word >> 1) as usize) as *const i64) }
            } else {
                word
            }
        })
        .collect();
    DeadFrame::Boxed(WasmFrameData::boxed(raw_values, fail_descr, 0))
}

impl majit_backend::Backend for WasmBackend {
    /// `force(token)` where the token is what `FORCE_TOKEN` parked in the
    /// virtualizable: the running frame's `JitFrame`, whose data region starts
    /// `FIRST_ITEM_OFFSET` in.
    ///
    /// A zero token means no frame is holding the virtualizable, which is the
    /// interpreter-only state the default answers `None` for.
    fn force(&self, force_token: GcRef) -> Option<DeadFrame> {
        if force_token.0 == 0 {
            return None;
        }
        // runner.rs `force` on the native backends: assert the frame carries the
        // bracket its call published, then mark it so the GUARD_NOT_FORCED
        // waiting past that call deopts instead of running on.
        let items_base = forced_frame_items_base(force_token);
        let slot = items_base as *mut i64;
        let word = unsafe { *slot };
        assert_ne!(
            word & codegen::FORCE_ARMED_BIT,
            0,
            "force: wasm frame 0x{items_base:x} carries no force bracket",
        );
        unsafe { *slot = word | codegen::FORCE_TAKEN_BIT };
        Some(dead_frame_from_forced_frame(items_base))
    }

    fn is_force_token_armed(&self, force_token: GcRef) -> bool {
        force_token.0 != 0
            && unsafe { *(forced_frame_items_base(force_token) as *const i64) }
                & codegen::FORCE_ARMED_BIT
                != 0
    }

    fn supports_efficient_uint_mul_high(&self) -> bool {
        // WebAssembly has no high-half integer multiply.  The fallback in
        // codegen is multi-precision software, while i64.div/rem are native
        // Wasm operations.
        false
    }

    fn cpu_tracker(&self) -> &std::sync::Arc<majit_backend::CpuTotalTracker> {
        &self.cpu_tracker
    }

    fn assembler_memory_stats(&self) -> (usize, usize) {
        self.asm_memory_stats.get_stats()
    }

    fn backend_name(&self) -> &'static str {
        "wasm"
    }

    fn bridge_decline_is_terminal(&self) -> bool {
        // Every `compile_bridge` `Unsupported` return is a deterministic
        // structural decline — a function of the (ops, source-loop) shape that
        // re-tracing the same guard reproduces identically: CALL_ASSEMBLER /
        // cross-loop JUMP shape, missing source loop, loop-closing bridge into a
        // peeled preamble, non-direct loop guard, ref-home overflow, or the
        // codegen frame-slot / unhandled-opcode declines. So re-firing the guard
        // only rebuilds the same unsupported bridge; record it terminal.
        true
    }

    // ── Blackhole allocation (llmodel.py:775-790) ──
    //
    // The blackhole interpreter materializes virtuals (e.g. a virtualized
    // `W_IntObject` loop variable forced at loop exit) through these. Without
    // a real implementation `bhimpl_new*` returns 0 and the resumed frame
    // carries null operands. Mirrors `CraneliftBackend`'s overrides but routes
    // through the wasm thread-local GC; the old-generation allocator never
    // collects, so allocation inputs need no rooting here.

    /// llmodel.py bh_new(sizedescr).
    fn bh_new(&self, sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        wasm_bh_alloc_struct(sizedescr)
    }

    /// llmodel.py bh_new_with_vtable(sizedescr): allocate, then write
    /// the type pointer at `vtable_offset`.
    fn bh_new_with_vtable(&self, sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let vtable = sizedescr.get_vtable();
        let ptr = wasm_bh_alloc_struct(sizedescr);
        if ptr != 0
            && vtable != 0
            && let Some(vt_off) = self.vtable_offset
        {
            unsafe {
                *((ptr as *mut u8).add(vt_off) as *mut usize) = vtable;
            }
        }
        ptr
    }

    /// llmodel.py bh_new_array(length, arraydescr).
    fn bh_new_array(&self, length: i64, arraydescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let Ok(length) = usize::try_from(length) else {
            return 0;
        };
        let (base_size, itemsize, _sign) = arraydescr.unpack_arraydescr_size();
        let len_offset = arraydescr
            .array_len_offset()
            .expect("bh_new_array requires ArrayDescr.lendescr");
        let type_id = arraydescr.resolve_gc_tid();
        let Some(payload_size) = itemsize
            .checked_mul(length)
            .and_then(|items| base_size.checked_add(items))
        else {
            return 0;
        };
        let ptr = wasm_bh_alloc(type_id, payload_size);
        if ptr != 0 {
            unsafe {
                *((ptr as *mut u8).add(len_offset) as *mut usize) = length;
            }
        }
        ptr
    }

    /// llmodel.py bh_new_array_clear = bh_new_array (allocator zeroes).
    fn bh_new_array_clear(
        &self,
        length: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        self.bh_new_array(length, arraydescr)
    }

    /// `LLtypeMixin.bh_newstr` → `gc_ll_descr.gc_malloc_str`.
    fn bh_newstr(&self, length: i64) -> i64 {
        let Ok(length) = usize::try_from(length) else {
            return 0;
        };
        let base_size = 2 * std::mem::size_of::<usize>() + 1;
        let Some(payload_size) = base_size.checked_add(length) else {
            return 0;
        };
        let ptr = wasm_bh_alloc(majit_gc::lowlevel_str_type_id(), payload_size);
        if ptr != 0 {
            unsafe {
                *((ptr as *mut u8).add(std::mem::size_of::<usize>()) as *mut usize) = length;
            }
        }
        ptr
    }

    /// `LLtypeMixin.bh_newunicode` → `gc_ll_descr.gc_malloc_unicode`.
    fn bh_newunicode(&self, length: i64) -> i64 {
        let Ok(length) = usize::try_from(length) else {
            return 0;
        };
        let base_size = 2 * std::mem::size_of::<usize>();
        let Some(payload_size) = length
            .checked_mul(std::mem::size_of::<u32>())
            .and_then(|items| base_size.checked_add(items))
        else {
            return 0;
        };
        let ptr = wasm_bh_alloc(majit_gc::lowlevel_unicode_type_id(), payload_size);
        if ptr != 0 {
            unsafe {
                *((ptr as *mut u8).add(std::mem::size_of::<usize>()) as *mut usize) = length;
            }
        }
        ptr
    }

    /// llmodel.py bh_arraylen_gc: read the length prefix at
    /// `lendescr.offset`. Word-width (`*const usize`), matching the store
    /// `bh_new_array` makes at the same offset — a fixed 8-byte read would fold
    /// the first item into the high half on wasm32.
    ///
    /// Without this the trait stub answers `0` for every array length reached
    /// at trace time, so a spare-capacity test (`length < len(items)`) records
    /// its at-capacity arm on a list that has room. The compiled code reads the
    /// real length, so that guard then fails on nearly every iteration and the
    /// trace never stays in compiled code.
    fn bh_arraylen_gc(
        &self,
        array_ptr: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        let ofs = arraydescr
            .array_len_offset()
            .expect("bh_arraylen_gc requires ArrayDescr.lendescr");
        unsafe { *((array_ptr as *const u8).add(ofs) as *const usize) as i64 }
    }

    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[OpRc],
        token: &JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        diag_bump(23);
        // `x86/assembler.py:514` parity — bump
        // `cpu.tracker.total_compiled_loops` at the same point PyPy
        // creates the `CompiledLoopToken`.
        if let Some(clt) = token.compiled_loop_token() {
            majit_backend::record_compiled_loop_token(&self.cpu_tracker, &clt);
        }
        let ops_owned: Vec<Op> = normalize_ops_for_codegen(inputargs, ops);
        let (ops_owned, gc_table) = Self::intern_ref_constants(inputargs, ops_owned);
        let gc_table_base = gc_table.as_ref().map_or(0, |t| t.base_addr() as u32);
        let ops: &[Op] = &ops_owned;
        // Freeze this token's generated frame layout before CA resolution.  A
        // self-recursive CALL_ASSEMBLER reaches this point while its token is
        // still a pending placeholder, so this is the one authoritative frame
        // geometry for both the loop and each nursery-allocated self callee.
        let raw_frame_value_slots = codegen::frame_value_slots(inputargs, ops);
        let raw_num_ref_homes = codegen::count_ref_homes(inputargs, ops);
        let label_ref_slots =
            codegen::label_ref_capture_slots(inputargs, ops).max(FROZEN_CHAIN_LABEL_REF_SLOTS);
        // An entry bridge (`compile.py ResumeFromInterpDescr`) is sent
        // to the backend through `compile_loop` like any loop, but it is not one:
        // it has no LABEL of its own and ends in a JUMP into an
        // already-compiled loop. Resolve that loop the same way `compile_bridge`
        // resolves a loop-closing bridge's target, and ADOPT its frozen
        // geometry — `LabelTarget::frame` exists because a tail call reuses the
        // caller's frame, so the two layouts must agree offset for offset, not
        // merely in size. Compiling against the target's geometry makes them
        // agree by construction.
        let entry_bridge_target = if has_cross_loop_terminal_jump(ops) {
            let Some(target) = resolve_cross_loop_jump_target(ops, None) else {
                diag_bump(2); // declined: JUMP target not chainable
                return Err(BackendError::Unsupported(
                    "wasm backend: cross-loop terminal JUMP target is not a \
                     chainable published label"
                        .into(),
                ));
            };
            // Same fit test a chained bridge gets against its source frame: the
            // adopted layout must hold everything this trace spills.
            if raw_frame_value_slots > target.frame.value_slots
                || raw_num_ref_homes > target.frame.ordinary_home_slots()
            {
                diag_bump(4);
                return Err(BackendError::Unsupported(format!(
                    "wasm backend: entry bridge needs values={raw_frame_value_slots}, \
                     homes={raw_num_ref_homes}; target frozen layout has values={}, homes={}",
                    target.frame.value_slots,
                    target.frame.ordinary_home_slots(),
                )));
            }
            Some(target)
        } else {
            None
        };
        let frame = match entry_bridge_target {
            Some(target) => target.frame,
            None => codegen::FrameGeometry::compact(
                frozen_slot_count(raw_frame_value_slots.max(FROZEN_CHAIN_VALUE_SLOTS)),
                frozen_slot_count(raw_num_ref_homes.max(FROZEN_CHAIN_REF_HOMES)) + label_ref_slots,
                label_ref_slots,
            ),
        };
        // `x86/assembler.py::assemble_loop` installs the generated frame
        // depth on the token's `CompiledLoopToken.frame_info`.  CALL_ASSEMBLER
        // redirect later propagates the replacement depth through that exact
        // object (`CompiledLoopToken.update_frame_info`).
        if let Some(clt) = token.compiled_loop_token() {
            let baseofs = (majit_gc::header::GcHeader::SIZE
                + majit_backend::jitframe::FIRST_ITEM_OFFSET) as i64;
            let depth = frame.ca_frame_bytes as usize / std::mem::size_of::<isize>();
            clt.frame_info
                .lock()
                .update_frame_depth(baseofs, depth as i64);
        }
        // A general CALL_ASSEMBLER enters the real compiled token selected by
        // the descr.  While this loop is pending, PyPy puts a separately
        // compiled tmp callback in that cell; the bodyless pending-token
        // shortcut previously used by wasm is intentionally absent.
        let ca_targets = general_int_call_assembler_target(ops);
        let allow_ca = ca_deopt_helper_slot() != 0 && ca_targets.is_some();

        // Decline traces the wasm backend cannot compile correctly, so the
        // metainterp falls back to the interpreter (correct, if unaccelerated)
        // rather than installing a structurally-invalid trace module. For
        // CALL_ASSEMBLER that is the unresolved-target case only, judged by
        // `allow_ca` above; see `wasm_unsupported_trace_reason`.
        if let Some(reason) = wasm_unsupported_trace_reason(ops, allow_ca) {
            diag_bump(25);
            return Err(BackendError::Unsupported(reason));
        }
        if allow_ca {
            diag_bump(14); // accepted general CALL_ASSEMBLER loop
        }

        self.collect_constants_from_ops(ops);
        let trace_id = self.trace_counter;
        self.trace_counter += 1;
        let trace_entry_census =
            alloc_trace_entry_census(trace_id, codegen::entry_dispatch_key_count(ops));

        let typeid_table = self.collect_classptr_typeid_table(ops);
        let guard_gc_type_info = self.collect_guard_gc_type_info(ops);
        // Allocation helpers reached from a compiled trace through the host
        // `jit_call` trampoline. `fn as usize` is the `__indirect_function_table`
        // index on wasm32; taking it here keeps the function in the table.
        let alloc = alloc_helpers();
        let wb_fn_ptr = wasm_jit_write_barrier as *const () as usize as i64;
        // Exit indices come from the global fail-index space so a cross-trace
        // chain's `frame[0]` resolves regardless of which module wrote it
        // (`failguard::FAIL_DESCR_REGISTRY`).
        let guard_exit_count = codegen::guard_exit_count(inputargs, ops);
        let fail_index_base = reserve_fail_descrs(guard_exit_count);
        let (bridge_cells_base, bridge_cells_owner) = codegen::alloc_bridge_cells(guard_exit_count);
        let module_inputs = codegen::ModuleBuildInputs {
            inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
            // Keep these rewritten operations exactly as intern_ref_constants
            // produced them; their LoadFromGcTable immediates share this base.
            ops: ops_owned.clone(),
            inlined_bridges: Vec::new(),
            constants: self.constants.clone(),
            vtable_offset: self.vtable_offset,
            classptr_to_typeid: typeid_table,
            guard_gc_type_info,
            alloc,
            wb_fn_ptr,
            nursery: nursery_alloc_params(ops),
            invalidated_flag_addr: Arc::as_ptr(&token.invalidated) as usize as u32,
            gc_table_base,
            fail_index_base,
            bridge_cells_base,
            bridge_entry_arity: None,
            bridge_param_dispatch: bridge_params_enabled(),
            trace_entry_census,
            inline_trip: None,
            // A real loop's JUMP is a local back-edge `br`; an entry bridge
            // tail-calls its target loop and is deliberately not re-emittable.
            external_jump_slot: entry_bridge_target.map_or(0, |t| t.func_handle),
            external_jump_key: entry_bridge_target.map_or(0, |t| t.key),
            external_jump_wide_slot: entry_bridge_target.map_or(0, |t| t.wide_slot),
            frame,
            ca: ca_targets.as_ref().map_or_else(
                || codegen::CaParams {
                    ca_reload_fn_ptr: body_reload_fn_ptr(),
                    jf_top_addr: jf_top_addr(),
                    ..codegen::CaParams::default()
                },
                |targets| codegen::CaParams {
                    emit_ca: true,
                    targets: ca_codegen_targets(targets),
                    deopt_helper_slot: ca_deopt_helper_slot(),
                    ca_alloc_fn_ptr: wasm_jit_ca_alloc_frame as *const () as usize as i64,
                    ca_pop_fn_ptr: wasm_jit_ca_pop_frame as *const () as usize as i64,
                    ca_reload_fn_ptr: wasm_jit_ca_reload_frame as *const () as usize as i64,
                    ca_reload_caller_fn_ptr: wasm_jit_ca_reload_caller_frame as *const () as usize
                        as i64,
                    inline: ca_inline_params(ca_max_frame_bytes(targets)),
                    jf_top_addr: jf_top_addr(),
                },
            ),
        };
        let (wasm_bytes, guard_exits, num_ref_homes) = codegen::build_wasm_module(&module_inputs)?;

        // Build fail descriptors
        let fail_descrs: Vec<Arc<WasmFailDescr>> = guard_exits
            .iter()
            .map(|g| {
                Arc::new(WasmFailDescr {
                    fail_index: g.fail_index,
                    trace_id,
                    fail_arg_types: g.fail_arg_types.clone(),
                    is_finish: g.is_finish,
                    meta_descr: g.meta_descr.clone(),
                })
            })
            .collect();
        register_fail_descrs(&fail_descrs);
        if let Some(table) = gc_table {
            Self::register_gc_table(token, table);
        }

        // `runner.rs` / `compiler.rs` parity: the entry path reads
        // this to size the live-value list it hands `execute_token`
        // (`jitdriver.rs extend_compiled_live_values` →
        // `warmstate.py:188 cell.loop_token`). Leaving it unset makes a trace
        // whose inputargs outnumber the portal's live values be entered with
        // the short list, so every frame slot past it reads as a zero the
        // prologue then loads as a null Ref.
        token.set_inputarg_types(inputargs.iter().map(|ia| ia.tp).collect());

        let max_output_slots = guard_exits
            .iter()
            .map(|g| g.fail_arg_refs.len())
            .max()
            .unwrap_or(0)
            .max(inputargs.len());

        // Straight-line function-entry traces finish the current invocation
        // concretely.  Do not ask the host to compile their wasm module until
        // a later invocation actually enters the token: quasi-immutable
        // invalidation can retire such a token before it ever executes (the
        // module-global `except ... as e` stress case does exactly that).
        // Loop-bearing and CALL_ASSEMBLER traces stay eager because their
        // published label/CA targets need a live table slot immediately.
        let defer_host_compile = !ops.iter().any(|op| {
            op.opcode == majit_ir::OpCode::Label
                || op.opcode == majit_ir::OpCode::Jump
                || op.opcode.is_call_assembler()
        });
        // The encoded module length, which is what this target can measure:
        // the host runtime owns the compiled code, so there is no pyre-owned
        // executable mapping and no retained capacity to report. It is read
        // before any host compilation, and on native builds no host exists at
        // all, so it is a submitted-bytes figure rather than
        // `asmmemmgr.py:90`'s mapped arena.
        let code_size = wasm_bytes.len();

        // Instantiate via the host binding on wasm32, or store bytes for
        // testing on native (no wasm host available).
        #[cfg(target_arch = "wasm32")]
        let func_handle = if defer_host_compile {
            0
        } else {
            glue::compile_module_cached(&wasm_bytes)
        };
        #[cfg(not(target_arch = "wasm32"))]
        let func_handle = 0u32; // Placeholder — no wasm host available

        // `jit_compile_wasm` returns 0 when the host runtime rejects the emitted
        // module (e.g. a function body exceeding the parser's size limit — a
        // trace within the metainterp `trace_limit` can still overflow it once
        // the optimizer peels/unrolls the loop). Storing a token whose handle is
        // this dead sentinel would let `execute_token` dispatch table slot 0 (not
        // a trace), leaving `frame[0]` unwritten and resolving a wrong exit descr.
        // Decline the compile so the metainterp keeps the interpreter fallback —
        // a backend capability limit, reported like any other unsupported shape.
        #[cfg(target_arch = "wasm32")]
        if !defer_host_compile && func_handle == 0 {
            diag_bump(26);
            return Err(BackendError::Unsupported(
                "wasm host rejected the compiled trace module (oversized function body \
                 or invalid module)"
                    .to_string(),
            ));
        }

        // `asmmemmgr.py` counts the block a `materialize` handed out, which
        // here is the module the host has taken. Below the decline above, as
        // `compile_bridge` does: a rejected module was never instantiated, and
        // the token retained for it would charge its bytes for the backend's
        // whole life.
        let block = self.asm_memory_stats.record_block(code_size, code_size);
        self.asm_memory_blocks.push(block);

        // A peeled loop carries real work before its (last) LABEL — the
        // unrolled first iteration. codegen emits the `loop` at that LABEL, so
        // the preamble runs once on entry and is NOT part of the iterating body.
        // A loop-closing bridge that re-enters through `func_handle` would
        // re-run this preamble; record the shape so `compile_bridge` can decline
        // such a bridge (see `has_preamble` doc on the struct).
        // A peeled loop (the resume-at-LABEL wrapper's shape) — real work before
        // the last LABEL. Computed through the same predicate codegen's wrapper
        // gates on, so the recorded field and the emitted wrapper cannot drift.
        let has_preamble = codegen::is_resumable_peeled(ops);
        let (label_descrs, _) =
            stamp_and_publish_label_targets(func_handle, frame, inputargs, ops, None);
        // Per-guard, per-fail-arg induction-advance flags for
        // `compile_bridge`'s livelock check (see `guard_fail_args_advanced`).
        let guard_fail_arg_advanced = guard_fail_args_advanced(ops, &guard_exits);

        let compiled = CompiledWasmLoop {
            token_number: token.number,
            trace_id,
            input_types: inputargs.iter().map(|ia| ia.tp).collect(),
            func_handle: std::cell::Cell::new(func_handle),
            pending_wasm_bytes: std::cell::RefCell::new(defer_host_compile.then_some(wasm_bytes)),
            fail_descrs: std::cell::RefCell::new(fail_descrs),
            num_inputs: inputargs.len(),
            max_output_slots,
            num_ref_homes,
            frame,
            bridge_cells_base: std::cell::Cell::new(bridge_cells_base),
            module_bytes: std::cell::Cell::new(code_size as u32),
            num_guard_cells: std::cell::Cell::new(guard_exits.len()),
            has_preamble,
            label_descrs,
            guard_fail_arg_advanced,
            guard_fail_arg_counts: guard_exits
                .iter()
                .map(|guard| {
                    crate::codegen::live_fail_arg_count(
                        guard.meta_descr.as_ref(),
                        guard.fail_arg_refs.len(),
                    )
                })
                .collect(),
            bridge_param_dispatch: bridge_params_enabled(),
            bridge_descr_ranges: std::cell::RefCell::new(Vec::new()),
            chained_trace_meta: std::cell::RefCell::new(std::collections::HashMap::new()),
            _bridge_owned_cells: std::cell::RefCell::new(bridge_cells_owner.into_iter().collect()),
            bridge_slots: std::cell::RefCell::new(HashMap::new()),
            chained_bridge_slots: std::cell::RefCell::new(HashMap::new()),
            // Retaining the snapshot costs long-lived heap for the token's
            // whole lifetime, which moves when the collector next runs and so
            // moves which iteration a back edge's eval-breaker guard bails on.
            // Keep it only when a re-emission can actually consume it, so a run
            // with the switches off allocates exactly what it did before.
            reemit: std::cell::RefCell::new(
                (entry_bridge_target.is_none() && (reemit_enabled() || inline_bridge_enabled()))
                    .then_some(module_inputs),
            ),
            reemitted: std::cell::Cell::new(false),
            bridge_owned_label_targets: std::cell::RefCell::new(Vec::new()),
            ca_active: std::cell::Cell::new(false),
            ca_terminal_declined: std::cell::Cell::new(false),
            ca_callers: std::cell::RefCell::new(Vec::new()),
        };

        token.set_compiled(Box::new(compiled));
        let compiled = token
            .compiled
            .get()
            .and_then(|compiled| compiled.downcast_ref::<CompiledWasmLoop>())
            .expect("newly compiled wasm loop is missing");
        // For a pending self target this is the exact map already embedded in
        // the module's CA arm. Reuse it for the published metadata so the
        // loop and its self-callee have demonstrably identical geometry. A
        // non-self loop still owns a freshly built map for future callers.
        let callee_gcmap_ptr = ca_targets
            .as_ref()
            .and_then(|targets| {
                targets
                    .iter()
                    .find(|(target_token, _)| *target_token == token.number)
                    .map(|(_, target)| target.callee_gcmap_ptr)
            })
            .unwrap_or_else(|| {
                Box::leak(build_callee_gcmap(&compiled.input_types, compiled.frame)).as_ptr() as i64
            });
        // The module has now acquired its host-appended shared-table slot and
        // its finish index. Publish those mutable pieces before exposing the
        // immutable geometry metadata: previously compiled CALL_ASSEMBLER
        // modules load this stable entry at runtime.
        ca_dispatch_publish(
            token.number,
            compiled.eager_func_handle(),
            compiled as *const CompiledWasmLoop as usize as u32,
            compiled.frame.ca_frame_bytes,
            compiled.frame.dispatch_key_ofs as u32,
            callee_gcmap_ptr,
        );
        publish_call_assembler_target(
            token.number,
            CallAssemblerTarget {
                token_number: token.number,
                func_handle: compiled.eager_func_handle(),
                input_types: compiled.input_types.clone(),
                dispatch_key_ofs: compiled.frame.dispatch_key_ofs,
                callee_frame_bytes: compiled.frame.ca_frame_bytes,
                callee_gcmap_ptr,
                compiled_ptr: compiled as *const CompiledWasmLoop as usize as u64,
            },
        );
        if let Some(targets) = ca_targets.as_ref() {
            for (_, target) in targets {
                mark_call_assembler_target_active(target, token.invalidation_flag());
            }
        }

        diag_bump(24);
        Ok(AsmInfo {
            code_addr: 0,
            code_size,
        })
    }

    fn set_constants_pool(&mut self, constants: majit_ir::ConstMap<majit_ir::Const>) {
        self.constants.clear();
        for (&k, c) in constants.iter() {
            self.constants.insert(k, c.as_raw_i64());
        }
    }

    fn set_next_trace_id(&mut self, trace_id: u64) {
        self.trace_counter = trace_id;
    }

    // `make_and_attach_done_descrs` — the FINISH fast path
    // needs the singletons' identity, so this backend takes the attachment
    // instead of the trait's no-op default. Where a native backend publishes
    // `Arc::as_ptr` to its comparison sites, a wasm frame slot holds an exit
    // index rather than a pointer, so each singleton is bound to a reserved
    // index in the global exit space (`failguard::FINISH_EXIT_INDEX_*`).
    fn set_done_with_this_frame_descr_void(&mut self, descr: Arc<dyn majit_ir::Descr>) {
        failguard::attach_finish_descr(failguard::FINISH_EXIT_INDEX_VOID, descr);
    }

    fn set_done_with_this_frame_descr_int(&mut self, descr: Arc<dyn majit_ir::Descr>) {
        failguard::attach_finish_descr(failguard::FINISH_EXIT_INDEX_INT, descr);
    }

    fn set_done_with_this_frame_descr_ref(&mut self, descr: Arc<dyn majit_ir::Descr>) {
        failguard::attach_finish_descr(failguard::FINISH_EXIT_INDEX_REF, descr);
    }

    fn set_done_with_this_frame_descr_float(&mut self, descr: Arc<dyn majit_ir::Descr>) {
        failguard::attach_finish_descr(failguard::FINISH_EXIT_INDEX_FLOAT, descr);
    }

    fn set_exit_frame_with_exception_descr_ref(&mut self, descr: Arc<dyn majit_ir::Descr>) {
        failguard::attach_finish_descr(failguard::FINISH_EXIT_INDEX_EXC, descr);
    }

    // `set_next_header_pc` uses the trait default (no-op) — wasm does
    // not currently honour it.

    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[OpRc],
        original_token: &JitCellToken,
        _previous_tokens: &[std::sync::Arc<JitCellToken>],
        _caller_recovery_layout: Option<&majit_backend::ExitRecoveryLayout>,
    ) -> Result<AsmInfo, BackendError> {
        // A bridge is a fresh trace that continues from a source loop's guard
        // exit. Instead of returning that guard's index to the host and
        // round-tripping through the interpreter, the source loop's epilogue
        // `call_indirect`s the bridge in-module (see `codegen` epilogue). The
        // bridge runs in the SOURCE loop's reused frame: the guard spilled its
        // fail args positionally into `frame[1..]`. `build_function` reads the
        // positional slot `k`, independently of the bridge value id, so no
        // argument-recovery layout is needed — hence `caller_recovery_layout`
        // and `previous_tokens` are unused.
        let ops_owned: Vec<Op> = normalize_ops_for_codegen(inputargs, ops);
        // A bridge gets its own table, like `compile_loop`'s.
        let (ops_owned, gc_table) = Self::intern_ref_constants(inputargs, ops_owned);
        let gc_table_base = gc_table.as_ref().map_or(0, |t| t.base_addr() as u32);
        let ops: &[Op] = &ops_owned;
        diag_bump(0); // compile_bridge entered

        // is_loop=false: a bridge's terminal JUMP with no LABEL is a loop-closing
        // bridge whose re-entry target is plumbed via `external_jump_slot`.
        // Lift the CALL_ASSEMBLER decline when every callee target has frozen,
        // directly-enterable geometry; the CA arm lowers each operation to its
        // own in-module `call_indirect` target.
        // The CA arm must be able to complete a callee deopt; without the
        // registered `wasm_ca_resume_deopt` slot it could not, so decline the
        // lift (the host round-trip path still handles the CALL_ASSEMBLER).
        let ca_targets = bridge_int_call_assembler_target(ops);
        let ca_candidate = ca_deopt_helper_slot() != 0 && ca_targets.is_some();
        // The source guard this bridge attaches to. `fail_index` is its index in
        // the source loop's `fail_descrs` / cell array; `trace_id` identifies the
        // owning trace.
        let source_trace_id = fail_descr.trace_id();
        let source_fail_index = fail_descr.fail_index();

        // Scalars read from the source loop up front, so the immutable borrow of
        // `original_token` is released before the `&mut self` codegen calls.
        let (source_guard, source_func_handle, source_has_preamble, source_frame, is_direct) = {
            let source_loop = original_token
                .compiled
                .get()
                .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                .ok_or_else(|| {
                    BackendError::Unsupported(
                        "wasm backend: bridge source token has no compiled loop".into(),
                    )
                })?;
            // Resolve the failing guard's owning trace by the descr's
            // `trace_id`: the source loop itself, or one of the bridges
            // already chained onto it (`chained_trace_meta`) — a NESTED
            // sub-bridge source. Either way the resolution yields the owning
            // trace's guard-cell array, cell count, and the guard's
            // per-fail-arg advance flags. `None` = foreign trace (declined
            // below, diag 3).
            let is_direct = source_trace_id == source_loop.trace_id;
            let guard = if is_direct {
                Some((
                    source_loop.bridge_cells_base.get(),
                    source_loop.num_guard_cells.get(),
                    source_loop
                        .guard_fail_arg_advanced
                        .get(source_fail_index as usize)
                        .cloned()
                        .unwrap_or_default(),
                    source_loop
                        .guard_fail_arg_counts
                        .get(source_fail_index as usize)
                        .copied(),
                    source_loop.bridge_param_dispatch,
                ))
            } else {
                source_loop
                    .chained_trace_meta
                    .borrow()
                    .get(&source_trace_id)
                    .map(|m| {
                        (
                            m.cells_base,
                            m.num_cells,
                            m.guard_fail_arg_advanced
                                .get(source_fail_index as usize)
                                .cloned()
                                .unwrap_or_default(),
                            m.guard_fail_arg_counts
                                .get(source_fail_index as usize)
                                .copied(),
                            m.bridge_param_dispatch,
                        )
                    })
            };
            (
                guard,
                source_loop.materialize_func_handle()?,
                source_loop.has_preamble,
                source_loop.frame,
                is_direct,
            )
        };

        // The failing guard must belong to the source loop or to a bridge
        // already chained onto it, and its per-trace index must have a cell in
        // that trace's array. A foreign descr has no cell to flip; decline so
        // the metainterp keeps the correct interpreter fallback rather than
        // installing an unreachable bridge module.
        let Some((
            source_cells_base,
            source_num_cells,
            source_fail_arg_advanced,
            source_fail_arg_count,
            source_bridge_param_dispatch,
        )) = source_guard
        else {
            diag_bump(3); // declined: source guard's trace is not chained here
            return Err(BackendError::Unsupported(
                "wasm backend: bridge source guard is not a direct loop guard".into(),
            ));
        };
        if source_fail_index as usize >= source_num_cells {
            diag_bump(3);
            return Err(BackendError::Unsupported(
                "wasm backend: bridge source guard index has no dispatch cell".into(),
            ));
        }
        let bridge_entry_arity = if bridge_params_enabled() {
            if !source_bridge_param_dispatch {
                diag_bump(45);
                return Err(BackendError::Unsupported(
                    "wasm backend: source guard has no parameter bridge dispatch".into(),
                ));
            }
            if source_fail_arg_count != Some(inputargs.len()) {
                diag_bump(46);
                return Err(BackendError::Unsupported(
                    "wasm backend: guard and bridge input arities differ".into(),
                ));
            }
            Some(inputargs.len())
        } else {
            None
        };
        let allow_ca = ca_candidate;
        if let Some(reason) = wasm_unsupported_trace_reason(ops, allow_ca) {
            diag_bump(1); // declined: CALL_ASSEMBLER
            return Err(BackendError::Unsupported(reason));
        }
        if allow_ca {
            diag_bump(14); // accepted CALL_ASSEMBLER bridge
        }

        // A chained bridge executes in the source token's *same* frame. Its
        // offsets are frozen when that token is compiled, so accept it only if
        // its positional spill region and Ref-home region fit exactly within
        // that layout. Declining here preserves the normal blackhole fallback;
        // it is never safe to grow an already-allocated CA frame underneath a
        // later bridge.
        let bridge_value_slots = codegen::frame_value_slots(inputargs, ops);
        let bridge_ref_homes = codegen::count_ref_homes(inputargs, ops);
        if bridge_value_slots > source_frame.value_slots
            || bridge_ref_homes > source_frame.ordinary_home_slots()
        {
            diag_bump(4);
            return Err(BackendError::Unsupported(format!(
                "wasm backend: bridge frame needs values={bridge_value_slots}, homes={bridge_ref_homes}; \
                 source frozen layout has values={}, homes={}",
                source_frame.value_slots,
                source_frame.ordinary_home_slots(),
            )));
        }

        // A loop-closing bridge (terminal JUMP, no local LABEL) re-enters the
        // source loop through `source_func_handle` — the function entry. For a
        // peeled source loop, entering at the function entry re-runs the preamble
        // (the unrolled first iteration) against the bridge's mid-loop state
        // instead of resuming at the LABEL, so the induction variable never
        // advances: an infinite loop (the wasm chaining hang on nbody / fannkuch).
        //
        // A peeled loop carries the resume-at-LABEL dispatch: the loop-closing
        // JUMP arm sets the frame dispatch key to `target label ordinal + 1`,
        // so re-entering through `source_func_handle` `br_table`s to that
        // label's resume loader — chaining stays in-module. The bridge is
        // accepted when its JUMP's target label is recoverable from the descr,
        // the arities match, and the label's args are the complete live set of
        // the trace remainder (`label_resume_safe`); otherwise decline — the
        // guard then falls back to blackhole resume and
        // the guard descriptor's terminal bit stops the metainterp re-tracing it.
        // Non-peeled loops (entry == LABEL) re-enter correctly and keep
        // chaining.
        let bridge_is_loop_closing = has_cross_loop_terminal_jump(ops);
        if bridge_is_loop_closing {
            diag_bump(6); // loop-closing shape
        }
        if source_has_preamble {
            diag_bump(7); // source loop has preamble
        }
        let mut external_jump_key: u32 = 0;
        let mut external_jump_slot: u32 = source_func_handle;
        let mut external_jump_wide_slot: u32 = 0;
        let mut resumes_at_loop_header = false;
        if bridge_is_loop_closing {
            let target =
                resolve_cross_loop_jump_target(ops, Some((source_func_handle, source_frame)));
            if let Some(t) = target {
                external_jump_key = t.key;
                external_jump_slot = t.func_handle;
                external_jump_wide_slot = t.wide_slot;
                resumes_at_loop_header = t.is_last_label;
            }
            if target.is_none() {
                diag_bump(2); // declined: JUMP target not chainable
                return Err(BackendError::Unsupported(
                    "wasm backend: loop-closing bridge JUMP target is not a \
                     chainable published label"
                        .into(),
                ));
            }
        }

        // A loop-closing bridge carries the source loop's loop-carried state in
        // its terminal JUMP args and tail-calls the loop to iterate again. If no
        // JUMP arg is the result of a loop-state-advancing arithmetic or load op
        // — i.e. every loop-carried value is a verbatim input reload, a fresh
        // allocation, or a baked constant — the bridge re-presents byte-identical
        // induction/guard state on every pass, so the loop's exit guard never
        // flips and the loop⇄bridge cycle spins forever (a control-flow
        // livelock at constant stack depth and heap state). Such a bridge is a
        // guard side-trace that omits the loop body's advancing arithmetic; it
        // has no correct in-module resume, so decline it — the guard falls back
        // to blackhole resume and the guard descriptor's terminal bit stops the metainterp
        // re-tracing it. A genuinely advancing loop-closing bridge (an `i += 1`
        // counter feeding a JUMP arg) passes and keeps chaining.
        //
        // The check only concerns a bridge that lands directly AT the loop
        // header (the target's last label, or the entry of a non-peeled
        // loop): only then can the guard re-fail on byte-identical state. A
        // resume at an EARLIER label executes the segment between that label
        // and the header — the peeled iteration — which advances the state
        // before the loop re-runs, so no advance is required of the bridge
        // itself.
        if bridge_is_loop_closing && resumes_at_loop_header {
            // Bridge input position `k` reads frame slot `k`, where the source
            // guard spilled its k-th fail arg — so an `InputArg` JUMP arg is a
            // verbatim reload of source fail arg `k`. The advance for such an
            // arg may have happened in the SOURCE loop's body before the guard
            // (an `i += 1` preceding the failing branch): the source recorded
            // per-fail-arg whether the value was produced by a loop-state-
            // advancing op within the failing iteration
            // (`guard_fail_arg_advanced`), so consult that alongside the
            // in-bridge producers.
            let input_pos: std::collections::HashMap<u32, usize> = inputargs
                .iter()
                .enumerate()
                .map(|(k, ia)| (ia.index, k))
                .collect();
            let advances = ops
                .iter()
                .rev()
                .find(|op| op.opcode == majit_ir::OpCode::Jump)
                .is_some_and(|jump| {
                    jump.getarglist().iter().any(|arg| match arg {
                        majit_ir::operand::Operand::Op(producer) => {
                            advances_loop_state(producer.opcode)
                        }
                        majit_ir::operand::Operand::InputArg(ia) => {
                            input_pos.get(&ia.index).is_some_and(|&k| {
                                source_fail_arg_advanced.get(k).copied().unwrap_or(false)
                            })
                        }
                        _ => false,
                    })
                });
            // The JUMP hands input `k` back at position `k` only when it
            // re-presents the state the guard failed on unchanged. One that
            // reorders those inputs starts the next pass from a different state
            // vector, which is the same reasoning the heap carve-out below
            // uses: the shield refuses PROVABLY static bridges, and a permuted
            // state is not the byte-identical one the livelock argument rests
            // on. An arg that is not an input reload at all (a baked constant,
            // a fresh allocation) is static by itself and does not make the
            // JUMP a permutation.
            //
            // Reordering is not by itself enough, though. Read the arg list as
            // a map from JUMP position to the input position it reloads: the
            // state one pass produces is `s'[j] = s[source[j]]`, so a second
            // pass leaves it unchanged exactly when every position a source
            // names reloads ITSELF. `JUMP(input0, input0)` is the smallest
            // case — it moves slot 0 into slot 1 once and is a fixed point from
            // then on, so it re-presents byte-identical state and is refused
            // here as any verbatim reload is. A source chain that is not
            // stationary after one pass (a swap, a rotation) is admitted: its
            // orbit does have a finite period, but the shield is a static
            // approximation of the bridge alone — it does not model the loop
            // body that runs between two passes, which is where such a bridge's
            // advance actually comes from. Refusing one is not local to the
            // bridge either: the decline registers the guard in
            // `declined_bridge_guards`, which sends every later failure of it
            // to blackhole resume.
            let permutes_inputs = ops
                .iter()
                .rev()
                .find(|op| op.opcode == majit_ir::OpCode::Jump)
                .is_some_and(|jump| {
                    let sources: Vec<Option<usize>> = jump
                        .getarglist()
                        .iter()
                        .map(|arg| match arg {
                            majit_ir::operand::Operand::InputArg(ia) => {
                                input_pos.get(&ia.index).copied()
                            }
                            _ => None,
                        })
                        .collect();
                    sources.iter().any(|source| {
                        source.is_some_and(|k| sources.get(k).copied().flatten() != Some(k))
                    })
                });
            // Loop state carried on the HEAP (a permutation array flipped via
            // setarrayitem, an object field bumped via setfield, a residual
            // call's arbitrary effects) advances the cycle without any JUMP
            // arg showing inductive arithmetic. The shield only exists to
            // refuse PROVABLY static bridges, so any state-mutating op counts
            // as an advance.
            let mutates_heap = ops.iter().any(|op| {
                use majit_ir::OpCode::*;
                op.opcode.is_call()
                    || matches!(
                        op.opcode,
                        SetfieldGc
                            | SetfieldRaw
                            | SetarrayitemGc
                            | SetarrayitemRaw
                            | GcStore
                            | GcStoreIndexed
                            | RawStore
                            | Strsetitem
                            | Unicodesetitem
                    )
            });
            if !advances && !permutes_inputs && !mutates_heap {
                diag_bump(11); // declined: loop-closing bridge advances no loop-carried value
                return Err(BackendError::Unsupported(
                    "wasm backend: loop-closing bridge advances no loop-carried value \
                     (guard side-trace would livelock the chained loop)"
                        .into(),
                ));
            }
        }

        // A closing JUMP that names a LABEL of ANOTHER module cannot become a
        // `br`, so a region carrying one keeps the cross-module tail call its
        // out-of-line bridge made and only the ENTRY side is merged: the source
        // guard branches to the region's block with its values in locals
        // instead of storing them for a bridge call to read back.
        let region_external =
            (external_jump_slot != source_func_handle).then_some(codegen::ExternalJump {
                slot: external_jump_slot,
                key: external_jump_key,
            });
        // Set by the inline block below to the owner of a merge candidate whose
        // merge waits on `INLINE_TRIP_THRESHOLD` entries into this bridge.
        let mut defer_inline: Option<(Arc<JitCellToken>, u32)> = None;
        if inline_bridge_enabled() {
            // `model.py`: a bridge compiled after `invalidate_loop`
            // starts valid, and only a later invalidation activates its
            // GUARD_NOT_INVALIDATED (`runner_test.py test_guard_not_invalidated`
            // steps 3-4). A merged region reads the owner's root flag, which is
            // already set here, so it would be dead on arrival. Decline, and let
            // the out-of-line path mint the fresh flag that keeps the contract.
            // `(slot, key)` names the crossing this decline leaves in place,
            // which is what the trace-entry census counts.
            // By value: the id this compile will take, which the eager arm
            // below consumes before the last `decline` call is out of scope.
            let bridge_trace_id = self.trace_counter;
            // `source_fail_index` is the SOURCE TRACE's own exit ordinal, and
            // stays that everywhere else in this function: the dispatch cell,
            // `chained_bridge_slots` and `bridge_descr_ranges` are all keyed by
            // it, and the metainterp re-derives that key off the FailDescr. The
            // region handed to the owner is the one thing that indexes the
            // merged stream instead.
            let merged_source_fail_index = if is_direct {
                // The owner's own guards come first in the merged stream, so a
                // direct guard's two numberings coincide.
                Some(source_fail_index)
            } else {
                original_token
                    .compiled
                    .get()
                    .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                    .and_then(|loop_| {
                        let inputs = loop_.reemit.borrow();
                        inputs.as_ref().and_then(|inputs| {
                            merged_region_fail_index(inputs, source_trace_id, source_fail_index)
                        })
                    })
            };
            let decline = |reason: &str| {
                record_inline_decline(format!(
                    "bridge={bridge_trace_id} src={source_trace_id} fi={source_fail_index} \
                     slot={external_jump_slot} key={external_jump_key} reason={reason}"
                ));
            };
            if original_token.is_invalidated() {
                diag_bump(50);
                decline("owner_invalidated");
            } else if merged_source_fail_index.is_none() {
                // Still out of line: the guard belongs to a bridge module of
                // its own, so the owner's stream holds no exit for it. A guard
                // in a region already merged into the owner is a different
                // case — its code was emitted from the merged stream, so it is
                // physically in this module and reachable by `br`.
                diag_bump(33);
                decline("not_direct");
            } else if !bridge_is_loop_closing {
                diag_bump(34);
                decline("not_loop_closing");
            } else if let Some(candidate) = original_token
                .compiled
                .get()
                .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                .and_then(|loop_| loop_.reemit.borrow().as_ref().cloned())
            {
                // A guard in the peeled preamble cannot reach the loop-body
                // region blocks: the `loop` holding them has not been entered
                // there. `build_function` gives that class blocks of its own
                // outside the loop, and a body past the loop's `end` that only
                // the entry dispatch re-enters — so the header question below
                // is not asked of it.
                let source_in_preamble =
                    codegen::source_guard_precedes_loop_label(&candidate.ops, source_fail_index);
                if candidate
                    .inlined_bridges
                    .iter()
                    .any(|r| Some(r.source_fail_index) == merged_source_fail_index)
                {
                    diag_bump(36);
                    decline("already_owned");
                } else if !codegen::merged_stream_has_loop_label(&candidate) {
                    diag_bump(39);
                    decline("no_loop_label");
                } else if region_external.is_none()
                    && !resumes_at_loop_header
                    && !source_in_preamble
                    && !inline_nonheader_enabled()
                {
                    // Resuming at the header lets a region inside the `loop`
                    // `br` straight to it. Resuming at an earlier LABEL from
                    // there goes through the `loop`-wrapped dispatch, which is
                    // correct but opt-in (`inline_nonheader_enable`) until it is
                    // worth its re-emission.
                    diag_bump(38);
                    decline("not_header");
                } else if inline_trip_helper_slot() == 0 {
                    // Nothing to defer to: without the callback published the
                    // count could never be acted on, so the bridge stays out of
                    // line.
                    diag_bump(53);
                    decline("no_trip_helper");
                } else if let Some(owner) = original_token
                    .compiled_loop_token()
                    .and_then(|clt| clt.upgrade_loop_token())
                {
                    // Merging is append-only, so once one region takes the
                    // outside-the-loop placement every later one must too: its
                    // ops are the tail of the merged stream, and splicing an
                    // inside-loop region ahead of them would renumber the exits
                    // their sub-bridges' dispatch cells are keyed by. A
                    // loop-body guard can branch out to that placement, so this
                    // costs the later region its `br` to the header and not the
                    // merge.
                    let outside_loop = source_in_preamble
                        || candidate.inlined_bridges.iter().any(|r| r.outside_loop);
                    // The `is_none` arm above already declined, so this holds.
                    let Some(merged_fail_index) = merged_source_fail_index else {
                        return Err(BackendError::Unsupported(
                            "wasm backend: inline candidate without a merged-stream exit \
                             ordinal"
                                .into(),
                        ));
                    };
                    self.collect_constants_from_ops(ops);
                    let has_invalidation_guard = codegen::has_invalidation_guard(ops);
                    if has_invalidation_guard && outside_loop {
                        // A quasi-immutable fold's dependencies are registered
                        // once, against whatever flag the token names when this
                        // compile returns — the bridge's own, because deferring
                        // takes the out-of-line path below. The merge cannot
                        // move a registration that has already happened, so
                        // merged, the region would read the owner's root flag
                        // while its dependencies still hold the bridge's: a
                        // field mutated before the trip would be forgotten, and
                        // one mutated after would leave the fold in place. The
                        // eager invalidation arm below has no such window — it
                        // merges before this compile returns, so the flag it
                        // records is the one the dependencies then attach to.
                        diag_bump(56);
                        decline("defer_invalidation_guard");
                    } else if !has_invalidation_guard {
                        // Eligible, but not yet worth its owner re-emission:
                        // arm the bridge's entry counter and merge when it
                        // trips. This applies equally to a header-resuming
                        // region: INLINE_TRIP_THRESHOLD is calibrated from
                        // bridge entries, and eager header merging otherwise
                        // bypasses that cost decision entirely. Everything else
                        // about this compile is the ordinary out-of-line path
                        // below.
                        defer_inline = Some((owner, merged_fail_index));
                        diag_bump(54);
                        decline("deferred");
                    } else if region_external.is_some() {
                        // The eager arm below forgoes the trip threshold to
                        // keep a quasi-immutable dependency attached to the
                        // owner's flag, and pays an owner re-emission for it
                        // whether or not the region is ever hot. A cross-module
                        // region saves only its entry crossing — the closing
                        // tail call it keeps is the same one the out-of-line
                        // bridge made — so that trade goes the other way:
                        // taking it unmeasured cost `synth/gc_iterator_source_
                        // drop` 10% of its wall clock and `guard_failures`
                        // 1816 -> 5280, four eager merges' worth of restarted
                        // warmup on a workload too short to amortize one.
                        diag_bump(64);
                        decline("foreign_eager");
                    } else if compiled_wasm_loop(&owner).is_some_and(|loop_| {
                        loop_.module_bytes.get() > INLINE_EAGER_MAX_BYTES.load(Ordering::Relaxed)
                    }) {
                        // The re-emission this arm pays for is the whole owner,
                        // and it takes it without the entry evidence the
                        // deferred arm waits for. Past this size that trade is
                        // one the region cannot be shown to earn.
                        diag_bump(65);
                        decline("eager_too_large");
                    } else {
                        // Deferral would register this region's dependencies
                        // against its temporary bridge flag. A header region
                        // can instead merge before this compile returns, so
                        // its dependencies attach to the owner's flag from the
                        // outset. The outside-loop case was declined above.
                        let region = codegen::InlinedBridge {
                            source_fail_index: merged_fail_index,
                            external_jump: region_external.clone(),
                            outside_loop,
                            trace_id: self.trace_counter,
                            inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
                            ops: ops_owned.clone(),
                            gc_table_base,
                            constants: self.constants.clone(),
                        };
                        if self.install_inline_region(&owner, region) {
                            self.trace_counter += 1;
                            if let Some(table) = gc_table {
                                Self::register_gc_table(original_token, table);
                            }
                            // The region has no code of its own: it was
                            // installed by rebuilding the owner, so there is no
                            // address to report. `model.py compile_bridge`
                            // permits `None` here, and the consumers treat the
                            // result as debug data — `interp_resop.py:253-255`
                            // defaults `asmaddr`/`asmlen` to 0 when it is
                            // absent — so a zero-address artifact says exactly
                            // "installed, but not as a block of its own".
                            return Ok(AsmInfo {
                                code_addr: 0,
                                code_size: 0,
                            });
                        }
                    }
                } else {
                    // The merge outlives this call and has to keep the owner
                    // alive; a token with no compiled-loop token has no strong
                    // handle to take.
                    diag_bump(53);
                    decline("no_owner_handle");
                }
            } else {
                diag_bump(35);
                decline("not_reemittable");
            }
        }

        self.collect_constants_from_ops(ops);
        let trace_id = self.trace_counter;
        self.trace_counter += 1;
        let trace_entry_census =
            alloc_trace_entry_census(trace_id, codegen::entry_dispatch_key_count(ops));

        let typeid_table = self.collect_classptr_typeid_table(ops);
        let guard_gc_type_info = self.collect_guard_gc_type_info(ops);
        let alloc = alloc_helpers();
        let wb_fn_ptr = wasm_jit_write_barrier as *const () as usize as i64;

        // CALL_ASSEMBLER: the CA arm allocates a fresh callee using the target
        // token's frozen geometry. The earlier frame-fit decline guarantees a
        // movable callee cannot execute a trampoline-lowered op.
        let ca_params = if let Some(targets) = ca_targets.as_ref().filter(|_| allow_ca) {
            codegen::CaParams {
                emit_ca: true,
                // `compile_bridge`'s trampoline-decline floor above guarantees
                // no trampoline-lowered op executes on this movable CA callee
                // frame, so its tail call area is never touched.
                targets: ca_codegen_targets(targets),
                deopt_helper_slot: ca_deopt_helper_slot(),
                ca_alloc_fn_ptr: wasm_jit_ca_alloc_frame as *const () as usize as i64,
                ca_pop_fn_ptr: wasm_jit_ca_pop_frame as *const () as usize as i64,
                ca_reload_fn_ptr: wasm_jit_ca_reload_frame as *const () as usize as i64,
                ca_reload_caller_fn_ptr: wasm_jit_ca_reload_caller_frame as *const () as usize
                    as i64,
                // See compile_loop: one shared inline path must fit every
                // per-op callee frame in this trace.
                inline: ca_inline_params(ca_max_frame_bytes(targets)),
                jf_top_addr: jf_top_addr(),
            }
        } else {
            codegen::CaParams {
                ca_reload_fn_ptr: body_reload_fn_ptr(),
                jf_top_addr: jf_top_addr(),
                ..codegen::CaParams::default()
            }
        };

        // The region carries the trace id of the bridge standing in for it, so
        // the sub-bridges chained onto this bridge's guards
        // (`chained_bridge_slots`, keyed by that id) are replayed into the
        // merged region's cells when the owner is finally rebuilt.
        let inline_trip = defer_inline.map(|(owner, merged_fail_index)| {
            if region_external.is_some() {
                diag_bump(51);
            }
            let region = codegen::InlinedBridge {
                source_fail_index: merged_fail_index,
                external_jump: region_external.clone(),
                // Decided against the candidate as it stands when the merge
                // actually runs, which may have taken more regions by then.
                outside_loop: false,
                trace_id,
                inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
                ops: ops_owned.clone(),
                gc_table_base,
                constants: self.constants.clone(),
            };
            // The cell the owner's guard consults travels by address of the
            // field rather than by value: a later re-emission reallocates the
            // array and the probe has to reach the live one. The owner's size
            // travels by value, because it prices this merge alone.
            let (cells_base_ptr, owner_module_bytes) =
                compiled_wasm_loop(&owner).map_or((0, 0), |loop_| {
                    (
                        &loop_.bridge_cells_base as *const std::cell::Cell<u32> as usize as u32,
                        loop_.module_bytes.get(),
                    )
                });
            register_pending_inline(owner, region, cells_base_ptr, owner_module_bytes)
        });
        let pending_guard = PendingInlineGuard(inline_trip.map(|probe| probe.pending_id));

        // This bridge's exit indices come from the global fail-index space,
        // like every trace's (`failguard::FAIL_DESCR_REGISTRY`).
        let guard_exit_count = codegen::guard_exit_count(inputargs, ops);
        let base = reserve_fail_descrs(guard_exit_count);
        // `rpython/jit/backend/model.py:145`: a bridge compiled after an
        // invalidation starts valid; only a later invalidation may kill its
        // `GUARD_NOT_INVALIDATED` operations.
        let bridge_flag = original_token.mint_bridge_invalidation_flag();
        let (bridge_cells_base, bridge_cells_owner) = codegen::alloc_bridge_cells(guard_exit_count);
        let module_inputs = codegen::ModuleBuildInputs {
            inputargs: inputargs.iter().map(InputArg::fresh_value_copy).collect(),
            ops: ops_owned.clone(),
            inlined_bridges: Vec::new(),
            constants: self.constants.clone(),
            vtable_offset: self.vtable_offset,
            classptr_to_typeid: typeid_table,
            guard_gc_type_info,
            alloc,
            wb_fn_ptr,
            nursery: nursery_alloc_params(ops),
            invalidated_flag_addr: Arc::as_ptr(&bridge_flag) as usize as u32,
            gc_table_base,
            fail_index_base: base,
            bridge_cells_base,
            bridge_entry_arity,
            bridge_param_dispatch: bridge_params_enabled(),
            trace_entry_census,
            inline_trip,
            external_jump_slot,
            external_jump_key,
            external_jump_wide_slot,
            frame: source_frame,
            ca: ca_params,
        };
        let (wasm_bytes, guard_exits, _num_ref_homes) = codegen::build_wasm_module(&module_inputs)?;

        // Bridge exit descrs (fail_index already base-offset by build_wasm_module).
        let bridge_descrs: Vec<Arc<WasmFailDescr>> = guard_exits
            .iter()
            .map(|g| {
                Arc::new(WasmFailDescr {
                    fail_index: g.fail_index,
                    trace_id,
                    fail_arg_types: g.fail_arg_types.clone(),
                    is_finish: g.is_finish,
                    meta_descr: g.meta_descr.clone(),
                })
            })
            .collect();
        register_fail_descrs(&bridge_descrs);

        // Register the bridge module into the shared table, then publish its
        // descrs and flip the source guard's cell. Order matters: the descrs
        // must be resolvable (appended) before the cell makes the guard dispatch
        // into the bridge.
        #[cfg(target_arch = "wasm32")]
        let bridge_slot = glue::compile_module(&wasm_bytes);
        #[cfg(not(target_arch = "wasm32"))]
        let bridge_slot = 0u32;
        // A 0 handle means the host rejected the bridge module (see the
        // `compile_loop` decline). Flipping the source guard's cell to dispatch
        // into slot 0 would tail-call a non-trace; decline instead so the guard
        // keeps its host round-trip (correct, unaccelerated).
        #[cfg(target_arch = "wasm32")]
        if bridge_slot == 0 {
            return Err(BackendError::Unsupported(
                "wasm host rejected the compiled bridge module (oversized function body \
                 or invalid module)"
                    .to_string(),
            ));
        }
        // Past every path that can fail with no module published: from here the
        // probe exists and its callback owns the pending entry.
        pending_guard.disarm();
        // Only a bridge that survived the decline above gets its reference
        // constants rooted. The table is attached to the long-lived original
        // loop token, so rooting a rejected bridge's table would keep its
        // constants alive permanently, once per rejected attempt.
        if let Some(table) = gc_table {
            Self::register_gc_table(original_token, table);
        }
        diag_bump(5); // bridge compiled — chained in-module
        if bridge_entry_arity.is_some() {
            diag_bump(44); // bridge compiled with a parameter entry
        }

        // x86/assembler.py:706 publishes the target tokens defined by an
        // accepted bridge. `codegen::is_resumable_peeled` and
        // `codegen::resumable_label_count` both use `find_loop_label_index`. A
        // retrace closing onto its OWN new target token resolves the terminal
        // JUMP among this trace's LABELs and is peeled: codegen emitted the
        // resume `br_table`, and every resumable label is published at key
        // ordinal + 1. A `jump_to_preamble` retrace closes onto the ORIGINAL
        // loop's start descr, so it is not peeled and its first op is not a
        // LABEL; the
        // existing `first_label_at_entry` / arity guard correctly leaves that
        // label unpublished, because key 0 would re-run the work before it.
        let (_, published_label_descrs) = stamp_and_publish_label_targets(
            bridge_slot,
            source_frame,
            inputargs,
            ops,
            bridge_entry_arity,
        );

        {
            let source_loop = original_token
                .compiled
                .get()
                .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                .expect("source loop disappeared between borrows");
            // Append the bridge's exit descrs to the source loop's flat
            // `fail_descrs` and record the slice they occupy, keyed by the
            // source guard's `fail_index`. `compiled_bridge_fail_descr_layouts`
            // / `store_bridge_guard_hashes` use that range to stamp jitcounter
            // hashes onto these bridge-internal guards (compile.py:826-830
            // store_hash). `start` is captured inside the same `borrow_mut`
            // critical section as the `extend`, so the range stays in lockstep
            // with the vec.
            let count = bridge_descrs.len();
            {
                let mut descrs = source_loop.fail_descrs.borrow_mut();
                let start = descrs.len();
                descrs.extend(bridge_descrs);
                source_loop.bridge_descr_ranges.borrow_mut().push((
                    source_trace_id,
                    source_fail_index,
                    start,
                    count,
                ));
            }
            // Publish this bridge's own guard-dispatch metadata so a hot guard
            // INSIDE it can chain a nested sub-bridge (same resolution the
            // loop's own guards get, keyed by this bridge's trace_id).
            source_loop.chained_trace_meta.borrow_mut().insert(
                trace_id,
                ChainedTraceMeta {
                    cells_base: bridge_cells_base,
                    num_cells: guard_exits.len(),
                    guard_fail_arg_advanced: guard_fail_args_advanced(ops, &guard_exits),
                    guard_fail_arg_counts: guard_exits
                        .iter()
                        .map(|guard| {
                            crate::codegen::live_fail_arg_count(
                                guard.meta_descr.as_ref(),
                                guard.fail_arg_refs.len(),
                            )
                        })
                        .collect(),
                    bridge_param_dispatch: bridge_params_enabled(),
                },
            );
            // The bridge module lives as long as this source loop, so hand its
            // own cell array (if any) to the loop, freed when the loop drops.
            if let Some(owner) = bridge_cells_owner {
                source_loop._bridge_owned_cells.borrow_mut().push(owner);
            }
            source_loop.bridge_owned_label_targets.borrow_mut().extend(
                published_label_descrs
                    .into_iter()
                    .map(|descr_id| (descr_id, bridge_slot)),
            );
            if let Some(targets) = ca_targets.as_ref().filter(|_| allow_ca) {
                // Freeze this recursion to the CA mechanism: no further bridge
                // chains here (see the decline above the codegen call).
                for (_, target) in targets {
                    mark_call_assembler_target_active(target, bridge_flag.clone());
                }
            }
        }

        // CA dispatch diagnostics (guest `eprintln` is a no-op on wasm32, so
        // route through the BRIDGE_DIAG tallies the host surfaces): 12 = CA bridge
        // cell actually written (loop epilogue will tail into it); 13 = CA bridge
        // but the source loop reserved no bridge cells (cells_base 0) so the guard
        // never dispatches in-module — the recursion stays a host round-trip.
        if allow_ca {
            if source_cells_base != 0 && bridge_slot != 0 {
                diag_bump(12);
            } else {
                diag_bump(13);
            }
        }
        // The same question for every accepted bridge (slots 27/28): a bridge
        // whose source guard has no cell is compiled and then unreachable.
        if source_cells_base != 0 && bridge_slot != 0 {
            diag_bump(27);
        } else {
            diag_bump(28);
        }
        #[cfg(target_arch = "wasm32")]
        if source_cells_base != 0 && bridge_slot != 0 {
            let cell = (source_cells_base as usize + source_fail_index as usize * 4) as *mut u32;
            if unsafe { core::ptr::read(cell) } != 0 {
                diag_bump(29); // this guard already had a reachable bridge
            }
            unsafe {
                core::ptr::write(cell, bridge_slot);
            }
            // Retained module replacement and loop-closing bridge inlining
            // restore this cell after allocating a fresh dispatch array.
            if reemit_enabled() || inline_bridge_enabled() {
                if let Some(source_loop) = original_token
                    .compiled
                    .get()
                    .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                {
                    if is_direct {
                        source_loop
                            .bridge_slots
                            .borrow_mut()
                            .insert(source_fail_index, bridge_slot);
                    } else {
                        source_loop
                            .chained_bridge_slots
                            .borrow_mut()
                            .insert((source_trace_id, source_fail_index), bridge_slot);
                    }
                }
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = (source_cells_base, bridge_slot);

        let code_size = wasm_bytes.len();
        // `asmmemmgr.py:37`, as in `compile_loop` above: a bridge's module is a
        // block of its own.
        let block = self.asm_memory_stats.record_block(code_size, code_size);
        self.asm_memory_blocks.push(block);

        // The first bridge installation is the identity re-emission probe.
        // A failed probe leaves the old module installed and must not disrupt
        // the bridge that just became reachable.
        if is_direct && reemit_enabled() {
            let should_reemit = original_token
                .compiled
                .get()
                .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
                .is_some_and(|loop_| !loop_.reemitted.replace(true));
            if should_reemit {
                match self.reemit_loop(original_token) {
                    Ok(()) => diag_bump(31),
                    Err(_) => diag_bump(30),
                }
            }
        }

        Ok(AsmInfo {
            code_addr: 0,
            code_size,
        })
    }

    /// `compile.py` store_hash relies on a per-guard fail-descr layout
    /// to know which exits are real guards (vs FINISH) and to count them.
    /// `assign_guard_hashes` fetches one jitcounter hash per non-finish guard
    /// from this list, so without it no guard ever gets a hash, `must_compile`
    /// never fires, and a hot guard exit round-trips to the host forever instead
    /// of triggering a bridge. Build one layout per exit from the metainterp
    /// `ResumeGuardDescr` the optimizer stamped on the guard (`meta_descr`); the
    /// wasm backend keeps no machine-code recovery metadata (resume runs through
    /// the frontend `WasmFrameData` path), so the recovery / rd_* / gc-slot
    /// fields stay empty — `merge_backend_exit_layouts` keeps the frontend's own
    /// entry (`or_insert_with`) and only consumes `is_finish` + `source_op_index`.
    fn compiled_fail_descr_layouts(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())?;
        let trace_id = compiled.trace_id;
        let descrs = compiled.fail_descrs.borrow();
        let layouts = descrs
            .iter()
            .enumerate()
            .map(|(position, wfd)| {
                let meta = wfd.meta_descr.as_ref().and_then(|m| m.as_fail_descr());
                majit_backend::FailDescrLayout {
                    fail_index: position as u32,
                    source_op_index: meta.and_then(|fd| fd.source_op_index()),
                    trace_id,
                    trace_info: None,
                    fail_arg_types: wfd.fail_arg_types.clone(),
                    is_finish: wfd.is_finish,
                    is_exception_exit: meta
                        .map(|fd| fd.is_exit_frame_with_exception())
                        .unwrap_or(false),
                    recovery_layout: None,
                    frame_stack: None,
                    rd_numb: meta.and_then(|fd| fd.rd_numb().map(|s| s.to_vec())),
                    rd_consts: meta.and_then(|fd| fd.rd_consts().map(|s| s.to_vec())),
                    rd_virtuals: meta.and_then(|fd| fd.rd_virtuals().map(|s| s.to_vec())),
                    rd_pendingfields: meta.and_then(|fd| fd.rd_pendingfields().map(|s| s.to_vec())),
                }
            })
            .collect();
        Some(layouts)
    }

    /// `compile.py` store_hash: stamp the jitcounter hashes assigned by
    /// `assign_guard_hashes` onto each guard's metainterp `ResumeGuardDescr`
    /// (`meta_descr`) — the descr `must_compile_with_values` reads the status
    /// from. Same `ResumeDescr`-family + status-0 gate as the native backends.
    fn store_guard_hashes(&self, token: &JitCellToken, hashes: &[u64]) {
        let Some(compiled) = token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
        else {
            return;
        };
        let descrs = compiled.fail_descrs.borrow();
        for (i, &hash) in hashes.iter().enumerate() {
            let Some(wfd) = descrs.get(i) else { break };
            let Some(meta) = wfd.meta_descr.as_ref().and_then(|m| m.as_fail_descr()) else {
                continue;
            };
            if (meta.is_resume_guard() || meta.is_resume_guard_copied()) && meta.get_status() == 0 {
                meta.store_hash(hash);
            }
        }
    }

    /// `compile.py` store_hash for the guards INSIDE a compiled bridge.
    /// `compile_bridge` appends a bridge's exit descrs to the source loop's flat
    /// `fail_descrs` and records their `(source_fail_index, start, count)` slice
    /// in `bridge_descr_ranges`. Return one layout per descr in that slice so
    /// `assign_bridge_guard_hashes` stamps a jitcounter hash on each non-finish
    /// bridge guard — without it they stay status 0 and collide in jitcounter
    /// bucket 0. `fail_index` is the 0-based position within the bridge's own
    /// exit list (matching the bridge's frontend `exit_layouts` keying and the
    /// native backends' `compiled_bridge_fail_descr_layouts`); `trace_id` is the
    /// bridge's own id, stamped on each appended `WasmFailDescr`.
    fn compiled_bridge_fail_descr_layouts(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = original_token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())?;
        // The most recently chained bridge at this source guard (last range).
        let (start, count) = compiled
            .bridge_descr_ranges
            .borrow()
            .iter()
            .rev()
            .find(|r| r.0 == source_trace_id && r.1 == source_fail_index)
            .map(|&(_, _, start, count)| (start, count))?;
        let descrs = compiled.fail_descrs.borrow();
        let layouts = descrs
            .get(start..start + count)?
            .iter()
            .enumerate()
            .map(|(position, wfd)| {
                let meta = wfd.meta_descr.as_ref().and_then(|m| m.as_fail_descr());
                majit_backend::FailDescrLayout {
                    fail_index: position as u32,
                    source_op_index: meta.and_then(|fd| fd.source_op_index()),
                    trace_id: wfd.trace_id,
                    trace_info: None,
                    fail_arg_types: wfd.fail_arg_types.clone(),
                    is_finish: wfd.is_finish,
                    is_exception_exit: meta
                        .map(|fd| fd.is_exit_frame_with_exception())
                        .unwrap_or(false),
                    recovery_layout: None,
                    frame_stack: None,
                    rd_numb: meta.and_then(|fd| fd.rd_numb().map(|s| s.to_vec())),
                    rd_consts: meta.and_then(|fd| fd.rd_consts().map(|s| s.to_vec())),
                    rd_virtuals: meta.and_then(|fd| fd.rd_virtuals().map(|s| s.to_vec())),
                    rd_pendingfields: meta.and_then(|fd| fd.rd_pendingfields().map(|s| s.to_vec())),
                }
            })
            .collect();
        Some(layouts)
    }

    /// `compile.py` store_hash: stamp the hashes `assign_bridge_guard_hashes`
    /// assigned onto the metainterp `ResumeGuardDescr` of each guard inside the
    /// bridge attached at `source_fail_index`. Same `ResumeDescr`-family +
    /// status-0 gate as `store_guard_hashes`; iterates the same slice in the
    /// same order as `compiled_bridge_fail_descr_layouts` so the hash vector
    /// lines up positionally.
    fn store_bridge_guard_hashes(
        &self,
        token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
        hashes: &[u64],
    ) {
        let Some(compiled) = token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
        else {
            return;
        };
        let Some((start, _count)) = compiled
            .bridge_descr_ranges
            .borrow()
            .iter()
            .rev()
            .find(|r| r.0 == source_trace_id && r.1 == source_fail_index)
            .map(|&(_, _, start, count)| (start, count))
        else {
            return;
        };
        let descrs = compiled.fail_descrs.borrow();
        for (i, &hash) in hashes.iter().enumerate() {
            let Some(wfd) = descrs.get(start + i) else {
                break;
            };
            let Some(meta) = wfd.meta_descr.as_ref().and_then(|m| m.as_fail_descr()) else {
                continue;
            };
            if (meta.is_resume_guard() || meta.is_resume_guard_copied()) && meta.get_status() == 0 {
                meta.store_hash(hash);
            }
        }
    }

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        let compiled = token
            .compiled
            .get()
            .expect("no compiled code")
            .downcast_ref::<CompiledWasmLoop>()
            .expect("not CompiledWasmLoop");
        #[cfg(target_arch = "wasm32")]
        let func_handle = compiled
            .materialize_func_handle()
            .expect("wasm backend failed to materialize a runnable trace");

        // Host entry allocates the complete frozen geometry, including the tail
        // call area. Chained bridges share these exact offsets; only CA callee
        // frames use the smaller homes prefix (`ca_frame_bytes`).
        let frame_size = (compiled.frame.frame_bytes as usize).div_ceil(8);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (frame_size, args);
            panic!("wasm backend execute_token requires a wasm host");
        }
        #[cfg(target_arch = "wasm32")]
        {
            // The pending-exception cell is global, unlike the native
            // per-jitframe `jf_guard_exc`. A residual raise on a blackhole
            // resume path (publish_residual_call_exception) writes it outside
            // any trace and nothing clears it, so clear it before running this
            // trace; otherwise jit_exc_take below would surface a stale
            // exception from a previous frame's resume as this trace's.
            jit_exc_clear();

            // Orthodox frame path (PYRE_WASM_CA): run the trace on a real
            // GC-managed `JitFrame` so a collecting allocation forwards the live
            // Ref-home slots through the `jf_gcmap` custom trace, discovered via
            // the jitframe shadow stack — replacing the bespoke add_root-over-
            // homes scheme. The frame is old-gen (non-moving), so the frame
            // pointer held across `glue::execute` never dangles without a reload
            // protocol. The data region (fail_index at 0, inputs/outputs at
            // FRAME_SLOT_BASE, call area, dispatch key, Ref homes) lives in the
            // `jf_frame` items area; passing `jf + FIRST_ITEM_OFFSET` as the wasm
            // frame pointer keeps every local-0-relative codegen access
            // unchanged. (See `build_home_gcmap` for the wasm32 Signed-item
            // layout.)
            if wasm_jitframe_tid() != 0 {
                use majit_backend::jitframe::{FIRST_ITEM_OFFSET, JitFrame};
                let sign = std::mem::size_of::<isize>();
                // Data region (frame_size i64 slots) expressed in Signed items.
                let depth = frame_size * 8 / sign;
                let jf_ref =
                    wasm_alloc_oldgen_typed(wasm_jitframe_tid(), JitFrame::alloc_size(depth));
                assert!(jf_ref.0 != 0, "wasm JitFrame allocation failed");
                let jf = jf_ref.0 as *mut JitFrame;
                // `JitFrame::init` requires zero-filled storage, which the
                // native `calloc` entry (`runner.rs` `execute_token`) and the
                // wasm nursery reset (`nursery.rs` `reset`) both provide but
                // the old-gen arena does not — `ArenaCollection::malloc`
                // deliberately returns recycled bytes. `build_home_gcmap`
                // marks every Ref home of the frozen geometry, so a home the
                // trace has not defined yet when a collection lands must read
                // as null rather than as a stale word.
                unsafe {
                    std::ptr::write_bytes(jf as *mut u8, 0, JitFrame::alloc_size(depth));
                    JitFrame::init(jf, std::ptr::null(), depth);
                }

                // Per-loop gcmap over the surviving Ref-home region. Held in this
                // stack frame (jf_gcmap points at it) until the outputs are read
                // after the trace returns.
                let gcmap = build_home_gcmap(compiled.frame);
                unsafe { (*jf).jf_gcmap = gcmap.as_ptr() as *const u8 };

                let items_base = jf as usize + FIRST_ITEM_OFFSET;
                let fsb = codegen::FRAME_SLOT_BASE as usize;
                for (i, arg) in args.iter().enumerate() {
                    let v = match arg {
                        Value::Int(v) => *v,
                        Value::Float(v) => v.to_bits() as i64,
                        Value::Ref(r) => r.0 as i64,
                        Value::Void => 0,
                    };
                    unsafe { *((items_base + fsb + i * 8) as *mut i64) = v };
                }

                let saved = majit_gc::shadow_stack::push_jf(jf_ref);
                glue::execute(func_handle, items_base as u32);

                let exc_value = jit_exc_take();
                let fail_index = unsafe { *(items_base as *const i64) } as u32;
                // Global fail-index space: a cross-trace chain may exit through
                // a sibling loop's guard, so `frame[0]` never resolves against
                // this loop's own `fail_descrs`.
                let fail_descr =
                    global_fail_descr(fail_index).expect("invalid fail_index from compiled wasm");
                let num_outputs = exit_slot_count(&fail_descr);
                let raw_values: Vec<i64> = (0..num_outputs)
                    .map(|i| unsafe { *((items_base + fsb + i * 8) as *const i64) })
                    .collect();

                // Done reading the frame; release it from the jf shadow stack
                // (it becomes collectible) and free the gcmap. Null the frame's
                // jf_gcmap before dropping the gcmap Box: the old-gen frame can
                // outlive this call still marked VISITED (grayed while it was a
                // root at a major-cycle start), reachable through the major
                // gray stack or the remembered set. A later collection would
                // then custom-trace it and read the freed gcmap. A null gcmap
                // makes jitframe_trace forward nothing (jitframe.py),
                // which is correct here: the outputs were already read out.
                unsafe { (*jf).jf_gcmap = std::ptr::null() };
                majit_gc::shadow_stack::pop_jf_to(saved);
                drop(gcmap);

                return DeadFrame::Boxed(WasmFrameData::boxed(raw_values, fail_descr, exc_value));
            }

            // Host-buffer frame path, for an embedder that registered no
            // `JitFrame` type id: fail_index at item[0], inputs/outputs at
            // item[1 + i], surviving Ref homes rooted across the trace. A home
            // slot only ever holds null (entry init) or a valid GcRef
            // (store-on-def), so forwarding is safe. No collection moves this
            // buffer, which is what `host_entry_frame_is_jitframe` reports to
            // codegen so the body emits no frame reload. The release below is
            // straight-line and the wasm32 build is `panic=abort`, so
            // `glue::execute` cannot unwind and leak roots.
            //
            // The buffer carries a `JitFrame` header all the same, and is
            // published on the jitframe shadow stack for the span of the call.
            // Rooting the homes with the active GC reaches only *that*
            // collector. A frontend keeping a heap of its own reads the shadow
            // stack instead -- it is the one publication point a collector that
            // is not this one can consult -- so without the frame on it a home
            // holding one of that heap's objects is invisible: the collector
            // moves the object, the home keeps the address it had, and the
            // trace resumes on a pointer into free space. The two walks forward
            // the same slots, which a forwarding collector does idempotently:
            // the second visit reads an address the first already took out of
            // from-space.
            let sign = std::mem::size_of::<isize>();
            let depth = frame_size * 8 / sign;
            let alloc_size = majit_backend::jitframe::JitFrame::alloc_size(depth);
            // An `i64` element type for the alignment a `JitFrame` needs and
            // for the zero fill `JitFrame::init` requires.
            let mut backing = vec![0i64; alloc_size.div_ceil(8)];
            let jf = backing.as_mut_ptr() as *mut majit_backend::jitframe::JitFrame;
            unsafe { majit_backend::jitframe::JitFrame::init(jf, std::ptr::null(), depth) };
            // Held until the outputs are read, which is what `jf_gcmap` points
            // at for as long as the frame is on the shadow stack.
            let gcmap = build_home_gcmap(compiled.frame);
            unsafe { (*jf).jf_gcmap = gcmap.as_ptr() as *const u8 };
            let items = (jf as usize + majit_backend::jitframe::FIRST_ITEM_OFFSET) as *mut i64;
            for (i, arg) in args.iter().enumerate() {
                let v = match arg {
                    Value::Int(v) => *v,
                    Value::Float(v) => v.to_bits() as i64,
                    Value::Ref(r) => r.0 as i64,
                    Value::Void => 0,
                };
                unsafe { *items.add(1 + i) = v };
            }
            let home_base = compiled.frame.home_slot_base as usize / 8;
            for h in 0..compiled.frame.home_slots {
                let slot = unsafe { items.add(home_base + h) } as *mut GcRef;
                unsafe { wasm_gc_add_root(slot) };
            }
            majit_gc::shadow_stack::register_libc_jitframe(jf as usize);
            let saved = majit_gc::shadow_stack::push_jf(GcRef(jf as usize));
            {
                let _bh_phase = majit_gc::BhProbePhase::enter("compiled");
                glue::execute(func_handle, items as usize as u32);
            }
            majit_gc::shadow_stack::pop_jf_to(saved);
            majit_gc::shadow_stack::unregister_libc_jitframe(jf as usize);
            // Nothing reads the frame's interior through the gcmap any more,
            // and the gcmap is about to go out of scope.
            unsafe { (*jf).jf_gcmap = std::ptr::null() };
            for h in 0..compiled.frame.home_slots {
                let slot = unsafe { items.add(home_base + h) } as *mut GcRef;
                wasm_gc_remove_root(slot);
            }
            let exc_value = jit_exc_take();
            let fail_index = unsafe { *items } as u32;
            // Global fail-index space (see the CA-path resolution above).
            let fail_descr =
                global_fail_descr(fail_index).expect("invalid fail_index from compiled wasm");
            let num_outputs = exit_slot_count(&fail_descr);
            let raw_values: Vec<i64> = (0..num_outputs)
                .map(|i| unsafe { *items.add(1 + i) })
                .collect();
            drop(gcmap);
            drop(backing);
            DeadFrame::Boxed(WasmFrameData::boxed(raw_values, fail_descr, exc_value))
        }
    }

    fn execute_token_ints(&self, token: &JitCellToken, args: &[i64]) -> DeadFrame {
        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        self.execute_token(token, &values)
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        // The same selection as `get_latest_descr_arc` below: the metainterp
        // descr when the optimizer stamped one, since that is the object
        // carrying `is_exit_frame_with_exception` / `get_status` /
        // `rd_loop_token_clt`, and the backend descr only for synthetic exits.
        let data = frame
            .boxed_data()
            .and_then(|d| d.downcast_ref::<WasmFrameData>())
            .expect("not WasmFrameData");
        data.fail_descr
            .meta_descr
            .as_ref()
            .and_then(|meta| meta.as_fail_descr())
            .unwrap_or(data.fail_descr.as_ref())
    }

    fn get_latest_descr_arc(&self, frame: &DeadFrame) -> Arc<dyn majit_ir::Descr> {
        // `history.py:125` parity — when the optimizer stamped a
        // metainterp `ResumeGuardDescr` / `DoneWithThisFrame*` /
        // `ExitFrameWithExceptionDescrRef` / `PropagateExceptionDescr` on
        // `op.descr`, the wasm backend snapshotted it into
        // `WasmFailDescr.meta_descr`.  Forward through that Arc so
        // identity (`Arc::ptr_eq`) matches dynasm/cranelift; otherwise
        // fall back to the backend Arc upcast (synthetic backend-only
        // descrs).
        let data = frame
            .boxed_data()
            .and_then(|d| d.downcast_ref::<WasmFrameData>())
            .expect("not WasmFrameData");
        if let Some(meta) = data.fail_descr.meta_descr.as_ref() {
            return Arc::clone(meta);
        }
        Arc::clone(&data.fail_descr) as Arc<dyn majit_ir::Descr>
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        let data = frame
            .boxed_data()
            .and_then(|d| d.downcast_ref::<WasmFrameData>())
            .expect("not WasmFrameData");
        data.raw_values[index]
    }

    fn get_value_direct(&self, frame: &DeadFrame, slot: usize) -> i64 {
        // Wasm's slot space is the dense fail-value vector.
        self.get_int_value(frame, slot)
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        let data = frame
            .boxed_data()
            .and_then(|d| d.downcast_ref::<WasmFrameData>())
            .expect("not WasmFrameData");
        f64::from_bits(data.raw_values[index] as u64)
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        let data = frame
            .boxed_data()
            .and_then(|d| d.downcast_ref::<WasmFrameData>())
            .expect("not WasmFrameData");
        GcRef(data.raw_values[index] as usize)
    }

    /// llmodel.py grab_exc_value parity: the exception captured when the
    /// trace exited through a GuardNoException / GuardException.
    fn grab_exc_value(&self, frame: &DeadFrame) -> GcRef {
        let data = frame
            .boxed_data()
            .and_then(|d| d.downcast_ref::<WasmFrameData>())
            .expect("not WasmFrameData");
        GcRef(data.exc_value as usize)
    }

    fn clear_stored_exception(&self) {
        crate::jit_exc_clear();
    }

    fn invalidate_loop(&self, token: &JitCellToken) {
        // A validated wasm module's code is immutable, so
        // GUARD_NOT_INVALIDATED loads a live flag instead of having its
        // instruction bytes patched in place — the same shape the llgraph
        // backend uses (`llgraph/runner.py:375` sets `trace.invalid` across
        // `_llgraph_alltraces`). `model.py:145` covers the loop AND its
        // attached bridges, each of which reads its own generation flag, so
        // this must go through `invalidate` rather than store to the root
        // flag alone.
        token.invalidate();
    }

    fn redirect_call_assembler(
        &self,
        old: &JitCellToken,
        new: &JitCellToken,
    ) -> Result<(), BackendError> {
        let Some(old_target) = call_assembler_target(old.number) else {
            // Without the old metadata, no baked frame geometry is available
            // to prove an existing caller can enter the replacement safely.
            return Err(BackendError::Unsupported(format!(
                "call-assembler redirect from token {} has no preserved geometry",
                old.number
            )));
        };
        let Some(mut new_target) = call_assembler_target(new.number) else {
            return Err(BackendError::Unsupported(format!(
                "call-assembler redirect to token {} has no compiled target",
                new.number
            )));
        };
        if old_target.input_types != new_target.input_types {
            return Err(BackendError::Unsupported(format!(
                "call-assembler redirect from token {} to {} changed input types",
                old.number, new.number
            )));
        }
        if new_target.func_handle == 0 && new_target.compiled_ptr != 0 {
            let new_loop = unsafe {
                (new_target.compiled_ptr as *const CompiledWasmLoop)
                    .as_ref()
                    .expect("published CALL_ASSEMBLER target has a live compiled loop")
            };
            let handle = new_loop.materialize_func_handle()?;
            #[cfg(target_arch = "wasm32")]
            if handle == 0 {
                return Err(BackendError::Unsupported(format!(
                    "call-assembler redirect to token {} could not materialize wasm code",
                    new.number
                )));
            }
            new_target.func_handle = handle;
            ca_dispatch_publish(
                new.number,
                handle,
                new_target.compiled_ptr as u32,
                new_target.callee_frame_bytes,
                new_target.dispatch_key_ofs as u32,
                new_target.callee_gcmap_ptr,
            );
            publish_call_assembler_target(new.number, new_target.clone());
        }
        let movable_callee = new_target.callee_frame_bytes != 0
            && new_target.callee_gcmap_ptr != 0
            && new_target.compiled_ptr != 0
            && unsafe {
                (new_target.compiled_ptr as *const CompiledWasmLoop)
                    .as_ref()
                    .is_some_and(|loop_| !loop_.ca_terminal_declined.get())
            };
        if !movable_callee {
            return Err(BackendError::Unsupported(format!(
                "call-assembler redirect to token {} is not a movable-CA callee",
                new.number
            )));
        }

        // `x86/assembler.py::redirect_call_assembler` first calls
        // `newlooptoken.compiled_loop_token.update_frame_info(old, baseofs)`.
        // In particular, a real loop may need a deeper frame than the temporary
        // callback it replaces; rejecting that growth is not PyPy semantics.
        if let (Some(new_clt), Some(old_clt)) =
            (new.compiled_loop_token(), old.compiled_loop_token())
        {
            let baseofs = (majit_gc::header::GcHeader::SIZE
                + majit_backend::jitframe::FIRST_ITEM_OFFSET) as i64;
            let old_weak = std::sync::Arc::downgrade(&old_clt);
            new_clt.update_frame_info(&old_clt, old_weak, baseofs);
        }

        // Existing callers retain `old`'s stable entry.  Make both the
        // runtime dispatch values and the metadata lookup resolve to `new`.
        ca_dispatch_redirect(
            old.number,
            new_target.func_handle,
            new_target.compiled_ptr as u32,
            new_target.callee_frame_bytes,
            new_target.dispatch_key_ofs as u32,
            new_target.callee_gcmap_ptr,
        );
        transfer_call_assembler_target_activity(&old_target, &new_target);
        new_target.token_number = old.number;
        publish_call_assembler_target(old.number, new_target);
        Ok(())
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer through the installed gc_ll_descr.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        self.lookup_typeid_from_classptr(classptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_backend::{Backend, JitCellToken};
    use majit_gc::collector::MiniMarkGC;
    use majit_gc::trace::TypeInfo;
    use majit_ir::forwarding::bound_operand_from_opref as rb;

    #[test]
    fn typed_blackhole_allocation_never_falls_back_to_raw_memory() {
        // No active wasm GC is installed on this test thread.  A typed descr
        // therefore has no legal allocator and must report NULL to
        // blackhole.py `_get_method`; the previous raw fallback returned a
        // headerless block that the collector could neither identify nor
        // trace.
        assert_eq!(wasm_bh_alloc(1, 32), 0);
    }

    #[test]
    fn blackhole_varsize_rejects_negative_lengths_without_panicking() {
        let backend = WasmBackend::new();
        assert_eq!(backend.bh_newstr(-1), 0);
        assert_eq!(backend.bh_newunicode(-1), 0);
    }

    #[test]
    fn cross_loop_terminal_jump_uses_target_descr_identity() {
        let local_descr = majit_ir::make_loop_target_descr(1, false);
        let foreign_descr = majit_ir::make_loop_target_descr(2, false);
        let label = Op::new(majit_ir::OpCode::Label, &[]);
        label.setdescr(local_descr.clone());
        let jump = Op::new(majit_ir::OpCode::Jump, &[]);
        jump.setdescr(foreign_descr);
        let ops = vec![label, jump];

        assert!(has_cross_loop_terminal_jump(&ops));
        ops[1].setdescr(local_descr);
        assert!(!has_cross_loop_terminal_jump(&ops));
    }

    #[test]
    fn straightline_trace_defers_host_module_until_execution() {
        let _compile_guard = failguard::FAIL_DESCR_TEST_LOCK.lock();
        let mut backend = WasmBackend::new();
        let token = JitCellToken::new(1);
        let finish = Op::new(majit_ir::OpCode::Finish, &[]);
        finish.pos.set(majit_ir::OpRef::void_op(0));
        finish.set_fail_arg_types(Vec::new());
        finish.setfailargs(Vec::new().into());

        backend
            .compile_loop(&[], &[std::rc::Rc::new(finish)], &token)
            .expect("compile straight-line wasm trace");
        let compiled = token
            .compiled
            .get()
            .and_then(|c| c.downcast_ref::<CompiledWasmLoop>())
            .expect("compiled wasm metadata");

        assert_eq!(compiled.eager_func_handle(), 0);
        assert!(compiled.pending_wasm_bytes.borrow().is_some());

        // Retiring an unentered token must leave the module unmaterialized;
        // this is the exception/global-version invalidation-storm case.
        token.invalidate();
        assert_eq!(compiled.eager_func_handle(), 0);
        assert!(compiled.pending_wasm_bytes.borrow().is_some());
    }

    #[test]
    fn identical_call_assembler_publication_reuses_the_runtime_snapshot() {
        let _compile_guard = failguard::FAIL_DESCR_TEST_LOCK.lock();
        let token_number = 9_900_000;
        ca_dispatch_publish(token_number, 11, 22, 33, 44, 55);
        ca_dispatch_publish(token_number, 11, 22, 33, 44, 55);

        let table = failguard::WASM_CA_DISPATCH.lock();
        let entry = table
            .as_ref()
            .and_then(|table| table.get(&token_number))
            .expect("published dispatch entry");
        assert_eq!(entry.targets.lock().unwrap().len(), 1);
        drop(table);
        failguard::ca_dispatch_remove(token_number);
    }

    #[test]
    fn redirect_call_assembler_grows_tmp_callback_frame_info() {
        let _compile_guard = failguard::FAIL_DESCR_TEST_LOCK.lock();
        fn compile_with_depth(backend: &mut WasmBackend, token: &JitCellToken, value_count: u32) {
            let inputargs = vec![InputArg::new_int(0)];
            let mut previous = majit_ir::OpRef::input_arg_int(0);
            let mut ops = Vec::new();
            let mut values = Vec::new();
            for position in 1..=value_count {
                let op = Op::new(
                    majit_ir::OpCode::IntAdd,
                    &[rb(previous), rb(majit_ir::OpRef::const_int(1))],
                );
                op.pos.set(majit_ir::OpRef::int_op(position));
                previous = op.pos.get();
                values.push(previous);
                ops.push(std::rc::Rc::new(op));
            }
            if value_count > 1 {
                let guard = Op::new(
                    majit_ir::OpCode::GuardTrue,
                    &[rb(majit_ir::OpRef::const_int(1))],
                );
                guard.pos.set(majit_ir::OpRef::void_op(value_count + 1));
                guard.setfailargs(values.iter().copied().map(rb).collect::<Vec<_>>().into());
                guard.set_fail_arg_types(vec![majit_ir::Type::Int; values.len()]);
                ops.push(std::rc::Rc::new(guard));
            }
            let finish = Op::new(majit_ir::OpCode::Finish, &[rb(previous)]);
            finish.pos.set(majit_ir::OpRef::void_op(value_count + 2));
            finish.set_fail_arg_types(vec![majit_ir::Type::Int]);
            ops.push(std::rc::Rc::new(finish));
            backend
                .compile_loop(&inputargs, &ops, token)
                .expect("compile wasm redirect target");
        }

        let mut backend = WasmBackend::new();
        let tmp = JitCellToken::new(9_900_001);
        let real = JitCellToken::new(9_900_002);
        compile_with_depth(&mut backend, &tmp, 1);
        compile_with_depth(&mut backend, &real, 96);

        let tmp_clt = tmp.compiled_loop_token().expect("tmp callback CLT");
        let real_clt = real.compiled_loop_token().expect("real loop CLT");
        let tmp_depth = tmp_clt.frame_info.lock().jfi_frame_depth;
        let real_depth = real_clt.frame_info.lock().jfi_frame_depth;
        assert!(tmp_depth < real_depth);
        let tmp_target = call_assembler_target(tmp.number).expect("tmp callback metadata");
        let real_target = call_assembler_target(real.number).expect("real loop metadata");
        assert_ne!(tmp_target.dispatch_key_ofs, real_target.dispatch_key_ofs);

        backend
            .redirect_call_assembler(&tmp, &real)
            .expect("redirect tmp callback to deeper real loop");
        assert_eq!(tmp_clt.frame_info.lock().jfi_frame_depth, real_depth);

        let redirected = call_assembler_target(tmp.number).expect("redirected target metadata");
        let installed = call_assembler_target(real.number).expect("real target metadata");
        assert_eq!(redirected.callee_frame_bytes, installed.callee_frame_bytes);
        assert_eq!(redirected.dispatch_key_ofs, installed.dispatch_key_ofs);
        assert_eq!(redirected.callee_gcmap_ptr, installed.callee_gcmap_ptr);

        let table = failguard::WASM_CA_DISPATCH.lock();
        let entry = table
            .as_ref()
            .and_then(|table| table.get(&tmp.number))
            .expect("redirected dispatch entry");
        let targets = entry.targets.lock().unwrap();
        let target = targets.last().expect("published runtime target");
        assert_eq!(target.callee_frame_bytes, installed.callee_frame_bytes);
        assert_eq!(target.dispatch_key_ofs as u64, installed.dispatch_key_ofs);
        assert_eq!(target.callee_gcmap_ptr, installed.callee_gcmap_ptr);
    }

    /// llsupport/gc.py GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr
    /// Verify the wasm backend's gc_ll_descr round-trips a registered
    /// vtable→type_id mapping.
    #[test]
    fn test_backend_typeid_from_classptr_via_gc_ll_descr() {
        let mut gc = MiniMarkGC::new();
        let int_tid = gc.register_type(TypeInfo::simple(16));
        let int_vtable: usize = 0x3333_4400;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, int_vtable, int_tid);

        let mut backend = WasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let resolved = backend.get_typeid_from_classptr_if_gcremovetypeptr(int_vtable);
        assert_eq!(resolved, Some(int_tid));
        let unknown = backend.get_typeid_from_classptr_if_gcremovetypeptr(0xCAFE_F00D);
        assert_eq!(unknown, None);
    }

    /// Spike for the wasm-JITFRAME refactor: prove the shared
    /// `MiniMarkGC` forwards a JitFrame's interior Ref item through the
    /// `jf_gcmap` custom-trace when the frame is discovered via the jitframe
    /// shadow stack. This is the exact GC path the orthodox wasm loop would
    /// depend on — a non-moving old-gen JitFrame whose live Ref item slots are
    /// traced by `jf_gcmap` during a minor collection (`do_collect_nursery`
    /// Phase 1c → `trace_and_update_object` → `jitframe_custom_trace`). The
    /// wasm backend has none of the feeders yet; this confirms the collector
    /// side works so the feeders can be built.
    #[test]
    fn jitframe_oldgen_gcmap_minor_forwards_ref_item() {
        use majit_backend::jitframe::{
            FIRST_ITEM_OFFSET, JF_FRAME_OFS, JF_GCMAP_OFS, JitFrame, jitframe_type_info,
        };
        use majit_gc::GcAllocator;

        let mut gc = MiniMarkGC::new();
        let jf_tid = gc.register_type(jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));

        let depth = 2usize;
        // Non-moving old-gen JitFrame (jitframe_prefer_oldgen()).
        let frame = gc.alloc_oldgen_typed(jf_tid, JitFrame::alloc_size(depth));
        assert_ne!(frame.0, 0, "old-gen JitFrame alloc failed");
        let frame_ptr = frame.0 as *mut JitFrame;
        unsafe { JitFrame::init(frame_ptr, std::ptr::null(), depth) };

        // A fresh nursery object reachable ONLY through the frame's item slot 0.
        let young = gc.alloc_nursery_typed(payload_tid, 16);
        assert_ne!(young.0, 0, "nursery alloc failed");
        let young_before = young.0;
        unsafe {
            let item0 = (frame_ptr as *mut u8).add(FIRST_ITEM_OFFSET) as *mut usize;
            *item0 = young_before;
        }

        // Per-loop gcmap marking item slot 0 as a Ref: [data_word_count, bits].
        // jitframe_trace reads gcmap_lgt at +0, a data word at +GCMAPBASEOFS(8),
        // and maps bit i (of word 0) to jf_frame item i.
        let gcmap: [usize; 2] = [1, 0b1];
        unsafe {
            let gcmap_field = (frame_ptr as *mut u8).add(JF_GCMAP_OFS as usize) as *mut *const u8;
            *gcmap_field = gcmap.as_ptr() as *const u8;
        }

        // Discover the frame the orthodox way: push it on the jitframe shadow
        // stack so Phase 1c traces its interior via the gcmap.
        let saved = majit_gc::shadow_stack::push_jf(frame);
        gc.do_collect_nursery();
        majit_gc::shadow_stack::pop_jf_to(saved);

        // The young object must have been forwarded out of the nursery and the
        // item slot rewritten to its new address — proving the gcmap bit was
        // honored. An untraced slot would still hold young_before (now dangling).
        let item0_after =
            unsafe { *((frame_ptr as *const u8).add(FIRST_ITEM_OFFSET) as *const usize) };
        assert_ne!(item0_after, 0, "item0 cleared: frame interior not traced");
        assert_ne!(
            item0_after, young_before,
            "item0 not forwarded: gcmap bit was not honored by the collector"
        );
        assert!(
            gc.is_managed_heap_object(item0_after),
            "forwarded item0 is not a live managed object"
        );

        // The old-gen frame must NOT have moved: its length header stays intact
        // in place, so a wasm local holding frame_ptr would remain valid.
        let len_after = unsafe { *((frame_ptr as *const u8).add(JF_FRAME_OFS) as *const isize) };
        assert_eq!(len_after, depth as isize, "old-gen frame moved/corrupted");
    }
}
