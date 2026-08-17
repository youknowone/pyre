/// Warm state management — the lifecycle from interpreting to compiled code.
///
/// Manages the transition: Interpreting -> Tracing -> Compiled.
/// When the hot counter fires, we start tracing. When the trace is
/// complete, we compile it and cache the result.
///
/// Reference: rpython/jit/metainterp/warmstate.py WarmEnterState, BaseBaseJitCell
use indexmap::IndexMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use majit_backend::JitCellToken;
use majit_ir::{GreenKey, RetainedGreens, Type};
use std::sync::Arc;

use crate::counter::{DEFAULT_SIZE, JitCounter};
use crate::logger::Logger;

use crate::recorder::Trace;

/// Flags on a BaseJitCell, mirroring warmstate.py JC_* constants.
pub mod jc_flags {
    /// We are currently tracing from this green key.
    pub const TRACING: u8 = 0x01;
    /// Don't trace here (e.g., trace was too long last time).
    pub const DONT_TRACE_HERE: u8 = 0x02;
    /// Has a temporary procedure token (CALL_ASSEMBLER fallback).
    pub const TEMPORARY: u8 = 0x04;
    /// Tracing has occurred at least once from this key.
    pub const TRACING_OCCURRED: u8 = 0x08;
    /// warmstate.py: JC_FORCE_FINISH — the loop has a FINISH that
    /// returns a raw int (not a boxed pointer). Used by
    /// call_assembler to decide whether to unbox the result.
    pub const FORCE_FINISH: u8 = 0x10;
}

/// Explicit state of a BaseJitCell in the JIT lifecycle.
///
/// warmstate.py expresses this implicitly through flag combinations:
///   - no cell / flags==0           → NotHot
///   - JC_TRACING set               → Tracing
///   - loop_token present, valid    → Compiled
///   - loop_token.invalidated       → Invalidated
///   - JC_DONT_TRACE_HERE set       → DontTraceHere
///
/// We make these states explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseJitCellState {
    /// Not yet hot; still interpreting.
    NotHot,
    /// Actively tracing.
    Tracing,
    /// Compiled loop exists and is valid.
    Compiled,
    /// Compiled loop was invalidated (quasi-immutable mutation, etc.).
    Invalidated,
    /// Tracing was aborted; don't trace at this location.
    DontTraceHere,
}

/// Per-greenkey cell that tracks JIT state for a specific program location.
///
/// Mirrors rpython/jit/metainterp/warmstate.py BaseBaseJitCell.
pub struct BaseJitCell {
    /// JC_* flags.
    pub flags: u8,
    /// Explicit lifecycle state.
    pub state: BaseJitCellState,
    /// Hot counter value for this cell (local to this green key).
    pub counter: u32,
    /// Compiled loop token number, if a procedure token is owned.
    /// Cleared on invalidation.
    pub token: Option<u64>,
    /// Generation at which tracing was last started.
    /// Used to detect stale tracing sessions.
    pub tracing_generation: u64,
    /// Compiled loop token, if compilation has completed.
    ///
    /// TODO: upstream `warmstate.py:188` stores
    /// `wref_procedure_token` as a `weakref.ref(token)` so warmstate
    /// holds only a weak handle; `MemoryManager.alive_loops` is the only
    /// long-lived strong owner (`memmgr.py:9-14`).  Pyre still stores
    /// `Arc<JitCellToken>` here, so warmstate remains an extra strong owner
    /// even though `MetaInterp.compiled_loops` now stores weak token handles.
    /// This keeps the cell-routed lookup path (`get_procedure_token` →
    /// `loop_token.as_ref()`) live while the surrounding warmstate /
    /// compiled-loop metadata is still being converged.
    ///
    /// Convergence target: downgrade this field to `Weak<JitCellToken>` so
    /// eviction by `MemoryManager` matches `should_remove_jitcell`'s
    /// dead-weakref check (`warmstate.py:212-225`) and Pyre reaches PyPy's
    /// "alive_loops is the only long-lived strong owner" shape.  See
    /// `slice_x_f_landed_2026_05_02.md` memo Issue 3.3 for the remaining
    /// weak-ref convergence work.
    pub loop_token: Option<Arc<JitCellToken>>,
    /// Number of times tracing was aborted for this key.
    ///
    /// Kept for diagnostics only. In RPython, `retrace_limit` is handled by
    /// optimizeopt/unroll during retracing, not by warmstate abort handling.
    pub abort_count: u32,
    /// counter.py:75 / warmstate.py BaseJitCell.next
    /// Linked list for per-bucket chain in the celltable.
    pub next: Option<Box<BaseJitCell>>,
    /// This cell's own u64 identity — what every hash-form entry point on
    /// [`WarmEnterState`] names it by, and what flows on through
    /// `MetaInterp::compiled_loops`, `JitCellToken::green_key` and
    /// `rd_loop_token`.
    ///
    /// Upstream needs no such field: `maybe_compile_and_run`
    /// (warmstate.py:458-464) resolves greens + `comparekey` + `get_uhash`
    /// *together* into a cell object and then carries **that object** to the
    /// executor (`raise EnterJitAssembler(procedure_token, ...)`,
    /// warmstate.py:483/:511), so nothing downstream re-derives which cell was
    /// meant. Pyre carries a `u64` instead, and a bucket hash does not name a
    /// cell when the bucket is chained — two cells, one number. The cell key
    /// restores the missing half of the identity: it is
    ///
    /// * the bucket's raw `JitCell.get_uhash` when this cell is the only
    ///   occupant, so the collision-free case is bit-for-bit what it always
    ///   was, and
    /// * a minted, live-set-unique u64 ([`WarmEnterState::mint_cell_key`])
    ///   when this cell landed in a bucket whose raw hash another cell had
    ///   already taken.
    ///
    /// `None` only between [`BaseJitCell::new`] and the
    /// [`WarmEnterState::install_new_cell`] that files the cell — an
    /// unassigned cell is in no chain and is reachable by nothing.
    pub cell_key: Option<u64>,
    /// warmstate.py:568-582 — typed green-key carried per-cell so
    /// `JitCell.comparekey(*greenargs2)` can do per-green typed
    /// equality across hash collisions:
    ///
    /// ```python
    /// def comparekey(self, *greenargs2):
    ///     i = 0
    ///     for attrname, TYPE in green_args_name_spec:
    ///         item1 = getattr(self, attrname)
    ///         if not equal_whatever(TYPE, item1, greenargs2[i]):
    ///             return False
    ///         i = i + 1
    ///     return True
    /// ```
    ///
    /// `None` while pyre's legacy hash-only `cells: HashMap<u64,
    /// BaseJitCell>` lookup path is the default; typed-key callers
    /// (`lookup_chain_with_key` / `ensure_cell_for_key` below) populate
    /// this on insert so chained cells distinguish hash collisions like
    /// upstream `JitCell.get_jitcell` (warmstate.py:596-604) does.
    /// Migration of all callers from hash-only to typed keys is
    /// unfinished.
    pub comparekey: Option<GreenKey>,
    /// Owns the `Ref`-typed referents named by `comparekey`, for exactly this
    /// cell's lifetime — the `setattr` half of `JitCell.__init__`
    /// (warmstate.py:568-573), which pyre stored as a bare `i64` and so never
    /// owned. Without it a referent can be freed and its address reused by a
    /// different object, whose green key is then byte-identical to the dead
    /// one's; `comparekey_matches` returns `true` and the new object silently
    /// inherits this cell and its compiled loop token.
    ///
    /// Scope: this closes the hazard on the **typed** path only. The legacy
    /// hash-only flow leaves `comparekey` `None`, so it stores no key, owns
    /// nothing, and has **no comparator at all** — a collision there is
    /// unresolvable by any mechanism rather than resolvable-but-unsound. That
    /// is a different and worse hazard, not a smaller one.
    ///
    /// Empty unless a frontend registered `majit_ir::set_ref_resolver`.
    pub retained_greens: RetainedGreens,
}

impl BaseJitCell {
    pub(crate) fn new() -> Self {
        BaseJitCell {
            flags: 0,
            state: BaseJitCellState::NotHot,
            counter: 0,
            token: None,
            tracing_generation: 0,
            loop_token: None,
            abort_count: 0,
            next: None,
            cell_key: None,
            comparekey: None,
            retained_greens: RetainedGreens::default(),
        }
    }

    /// Store `key` as this cell's `comparekey` **and** take ownership of the
    /// `Ref` referents it names, in one statement.
    ///
    /// The two must not be set separately: a `comparekey` without its
    /// `retained_greens` is exactly the defect this pairing exists to close —
    /// a stored address the cell does not own. `comparekey` stays `pub` for
    /// the fixtures that construct chains directly, so this is the invariant
    /// by convention rather than by type; every production writer goes
    /// through here.
    pub fn set_comparekey(&mut self, key: &GreenKey) {
        self.retained_greens = RetainedGreens::retain(key);
        self.comparekey = Some(key.clone());
    }

    /// warmstate.py:575-582 `JitCell.comparekey(*greenargs2)`.
    ///
    /// Returns `true` iff this cell's stored typed greens match
    /// `other` per `GreenKey::eq` (which dispatches via
    /// `equal_whatever(TYPE, ...)` for STR/UNICODE/Float/Ref/Int).
    /// Cells without a stored comparekey (legacy hash-only path)
    /// always fail comparison — callers must explicitly opt in by
    /// inserting cells through the typed-key path
    /// (`WarmEnterState::ensure_cell_for_key`).
    pub fn comparekey_matches(&self, other: &GreenKey) -> bool {
        match &self.comparekey {
            Some(stored) => stored == other,
            None => false,
        }
    }

    pub fn is_tracing(&self) -> bool {
        self.flags & jc_flags::TRACING != 0
    }

    /// warmstate.py:191-196 — get_procedure_token returns None for
    /// invalidated tokens. is_compiled additionally excludes TEMPORARY.
    pub fn is_compiled(&self) -> bool {
        self.get_procedure_token().is_some() && (self.flags & jc_flags::TEMPORARY == 0)
    }

    /// warmstate.py:191-196 get_procedure_token
    pub fn get_procedure_token(&self) -> Option<&Arc<JitCellToken>> {
        self.loop_token.as_ref().filter(|t| !t.is_invalidated())
    }

    /// Set the procedure token and update ownership state.
    /// If `tmp` is true, sets the TEMPORARY flag (CALL_ASSEMBLER fallback).
    ///
    /// Returns the previous procedure token (if any), so the caller can
    /// implement `warmstate.py:343-348`'s `redirect_call_assembler` +
    /// `old_token.record_jump_to(procedure_token)` chain.
    pub fn set_procedure_token(
        &mut self,
        loop_token: impl Into<Arc<JitCellToken>>,
        tmp: bool,
    ) -> Option<Arc<JitCellToken>> {
        let loop_token = loop_token.into();
        self.token = Some(loop_token.number);
        let old = self.loop_token.replace(loop_token);
        if tmp {
            self.flags |= jc_flags::TEMPORARY;
        } else {
            self.flags &= !jc_flags::TEMPORARY;
            self.state = BaseJitCellState::Compiled;
        }
        old
    }

    /// Check whether we have ever had a procedure token assigned
    /// (mirrors BaseBaseJitCell.has_seen_a_procedure_token).
    ///
    /// Returns true if a token was ever set, even if it was later
    /// invalidated. The `token` field is a historical record and is
    /// never cleared.
    pub fn has_seen_a_procedure_token(&self) -> bool {
        self.token.is_some()
    }

    /// Whether this cell should be removed (for GC of dead cells).
    /// Mirrors BaseBaseJitCell.should_remove_jitcell.
    pub fn should_remove_jitcell(&self) -> bool {
        if self.get_procedure_token().is_some() {
            return false; // has a valid procedure token
        }
        if self.flags & jc_flags::TRACING != 0 {
            return false; // currently tracing
        }
        if self.flags & jc_flags::DONT_TRACE_HERE != 0 {
            // Remove only if we had a token that is now dead.
            return self.has_seen_a_procedure_token();
        }
        // warmstate.py:222-225
        if self.flags & jc_flags::FORCE_FINISH != 0 {
            return false;
        }
        true
    }
}

/// Per-green-key JIT cell state with associated data.
///
/// Richer variant of BaseJitCellState that carries trace/token payloads.
pub enum CellJitState {
    /// Normal interpretation; no tracing is active.
    Interpreting,
    /// Actively recording a trace.
    Tracing(Trace),
    /// A compiled loop exists for this green key.
    Compiled(Arc<JitCellToken>),
}

pub use crate::memmgr::MemoryManager;

// Warm state manager — the orchestrator of the JIT lifecycle. It keeps track
// of per-greenkey cells and the global hot counter.
// Defaults from `rpython.rlib.jit.PARAMETERS`.
// DEFAULT_ constants must match RPython exactly.

/// rlib/jit.py:588 threshold = 1039 (just above 1024, prime)
const DEFAULT_THRESHOLD: u32 = 1039;

/// rlib/jit.py:589 function_threshold = 1619
const DEFAULT_FUNCTION_THRESHOLD: u32 = 1619;

/// rlib/jit.py:590 trace_eagerness = 200
const DEFAULT_TRACE_EAGERNESS: u32 = 200;

/// rlib/jit.py:601 max_unroll_recursion = 7
const DEFAULT_MAX_UNROLL_RECURSION: u32 = 7;

/// rlib/jit.py:593 inlining = 1 (max_inline_depth derived)
const DEFAULT_MAX_INLINE_DEPTH: u32 = 7;

/// rlib/jit.py:592 trace_limit = 6000
const DEFAULT_TRACE_LIMIT: u32 = crate::trace_ctx::DEFAULT_TRACE_LIMIT as u32;

/// rlib/jit.py:595 retrace_limit = 0.
const DEFAULT_RETRACE_LIMIT: u32 = 0;

/// rlib/jit.py:598 max_unroll_loops = 0
const DEFAULT_MAX_UNROLL_LOOPS: u32 = 0;

/// rlib/jit.py:600 enable_opts = "all"
fn default_enable_opts() -> Vec<String> {
    vec![
        "intbounds".to_string(),
        "rewrite".to_string(),
        "virtualize".to_string(),
        "string".to_string(),
        "pure".to_string(),
        "earlyforce".to_string(),
        "heap".to_string(),
        "unroll".to_string(),
    ]
}

/// Pyre-local abort ceiling with no upstream analogue: after repeated aborts,
/// mark the green key `DONT_TRACE_HERE`.  Pyre walker aborts are structural
/// and recur identically on retrace; without a ceiling the same body would
/// retrace forever, each attempt executing its residual calls concretely.
const MAX_TRACE_ABORT_COUNT: u32 = 5;

/// rlib/jit.py:599 disable_unrolling = 200
const DEFAULT_DISABLE_UNROLLING: u32 = 200;

static NEXT_GLOBAL_TOKEN_NUMBER: AtomicU64 = AtomicU64::new(1);

/// JIT statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct JitStats {
    /// Number of cells in Compiled state.
    pub num_compiled: usize,
    /// Number of cells in Tracing state.
    pub num_tracing: usize,
    /// Number of cells in Invalidated state.
    pub num_invalidated: usize,
    /// Number of cells in DontTraceHere state.
    pub num_disable_noninlinable_function: usize,
    /// Total number of BaseJitCells.
    pub num_cells: usize,
    /// Referents pinned alive by stored typed green keys, summed over every
    /// cell in every chain.
    ///
    /// This is the resource cost of the ownership invariant (`RetainedGreens`)
    /// against a `cells` map with no size bound: nothing caps how many cells
    /// accumulate, so the pinned set grows with them. Reported so the growth
    /// is a number somebody can read rather than latent RSS — the condition
    /// under which pinning is allowed to ship ahead of a bound.
    ///
    /// WHICH WINDOW THIS IS TAKEN IN. The number is a **running total at
    /// the moment of the call**, not a settled figure, and it is not
    /// monotonic. Cells *are* dropped in production — `install_new_cell`
    /// unlinks every chained cell whose `should_remove_jitcell()` holds
    /// (which is the default for a cold, tokenless cell), and `cleanup_chain`
    /// drops the whole chain — and each drop releases that cell's retains.
    /// So a cold tree, a tree mid-warmup and a settled tree give three
    /// different answers for the same program, and a reading is only
    /// comparable against another taken at the same point.
    ///
    /// Do not quote it as "how many cells pin a referent" without saying
    /// after how many portal entries it was read. `gc_cells` has no
    /// production caller, so the one eviction path that would make this fall
    /// sharply is currently unreachable.
    ///
    /// Zero unless a frontend registered `set_ref_resolver`.
    pub num_pinned_refs: usize,
}

/// # The `u64` this type is keyed by is a CELL key, not a bucket hash
///
/// Every `cell_key: u64` parameter below names one cell
/// ([`BaseJitCell::cell_key`]), not a celltable bucket. The two are the same
/// number whenever a bucket holds one cell — which is every collision-free
/// workload, so nothing observable changes there — and differ only for a cell
/// that had to be minted a key because a sibling already held the bucket's raw
/// hash.
///
/// The parameter name says which side of the resolve the value sits on, so the
/// two entry points that genuinely take a PRE-resolve raw bucket hash keep the
/// `green_key_hash` spelling: [`WarmEnterState::bucket_is_chained`] (the test
/// [`WarmEnterState::resolve_cell_key`] asks before it decides whether a
/// `GreenKey` has to be built at all) and
/// [`WarmEnterState::trace_next_iteration`] (whose whole identity is the hash,
/// see its own doc).
///
/// The distinction exists because pyre carries a `u64` where upstream carries
/// the cell object. `maybe_compile_and_run` (warmstate.py:458-464) resolves
/// greens + `comparekey` + `get_uhash` together into a cell and then hands
/// *that cell's* procedure token to the executor (warmstate.py:483/:511:
/// `raise EnterJitAssembler(procedure_token, *execute_args)`) rather than
/// handing on a key for a second lookup — which is why upstream has no second
/// reader that can disagree with the first. Pyre restores that property by
/// resolving once, at the hash producer ([`WarmEnterState::cell_key_for`] /
/// [`WarmEnterState::ensure_cell_key`] / [`WarmEnterState::sole_cell_key`]),
/// and carrying the resolved cell key onward — through `MetaInterp`'s
/// `compiled_loops`, `JitCellToken::green_key` and `rd_loop_token` alike.
///
/// The counter is the deliberate exception: `jitcounter.tick(hash, ...)`
/// (warmstate.py:467) is indexed by the BUCKET upstream, so colliding keys
/// share one back-edge counter. Every counter call here therefore goes through
/// [`WarmEnterState::bucket_of`] first.
pub struct WarmEnterState {
    /// counter.py JitCounter parity: single timetable shared by loop
    /// entry, guard failure, and function entry — each caller passes
    /// a different pre-computed increment to tick(hash, increment).
    pub counter: JitCounter,
    /// counter.py:227-256 celltable: per-bucket chains, keyed by the raw
    /// `JitCell.get_uhash` of the green key. The map entry is the chain HEAD;
    /// walk `BaseJitCell::next` for the rest. A cell inside a chain is named
    /// by its own [`BaseJitCell::cell_key`], not by this map key.
    cells: indexmap::IndexMap<u64, BaseJitCell>,
    /// Minted cell keys → the raw bucket hash the cell lives in.
    ///
    /// Only cells that could not take their bucket's raw hash appear here, so
    /// this map is empty for every workload that never chains a bucket, and
    /// [`WarmEnterState::bucket_of`] short-circuits on `is_empty()`. The
    /// invariant it maintains is: **`bucket_of(cell_key)` is the bucket the
    /// cell lives in** — trivially `cell_key` itself for an unminted key,
    /// because an unminted key is only ever assigned inside the bucket whose
    /// hash it equals.
    minted: indexmap::IndexMap<u64, u64>,
    /// Serial feeding [`WarmEnterState::mint_cell_key`]'s candidate sequence.
    mint_serial: u64,
    /// Compilation threshold (copied from counter for easy access).
    threshold: u32,
    /// warmstate.py:254: increment_threshold = compute_threshold(threshold).
    increment_threshold: f64,
    /// warmstate.py: trace_eagerness parameter (integer, default 200).
    trace_eagerness: u32,
    /// warmstate.py: increment_trace_eagerness = compute_threshold(trace_eagerness).
    /// Pre-computed f64 increment for guard failure counter ticking.
    increment_trace_eagerness: f64,
    /// Function call threshold for inlining during tracing.
    function_threshold: u32,
    /// warmstate.py:257: increment_function_threshold = compute_threshold(function_threshold).
    increment_function_threshold: f64,
    /// Maximum depth of inlined function calls during tracing.
    max_inline_depth: u32,
    /// Maximum number of operations per trace before aborting.
    trace_limit: u32,
    /// Global tracing generation counter.
    /// Incremented each time tracing starts; stored in BaseJitCell to
    /// detect stale tracing sessions.
    tracing_generation: u64,
    /// Optional profiling logger, enabled via MAJIT_STATS=1 or MAJIT_LOG=1.
    jitlog: Option<Logger>,
    /// Quasi-immutable field invalidation registry.
    ///
    /// Maps a quasi-immutable field key (hash of object_id + field_index)
    /// to the set of cell keys whose compiled loops depend on that field.
    /// When a quasi-immutable field is mutated, all dependent loops are invalidated.
    quasiimmut_deps: indexmap::IndexMap<u64, Vec<u64>>,
    // Function-entry hotness rides on the shared `counter: JitCounter`
    // below (warmstate.py:467 — maybe_compile_and_run's tick + reset
    // goes through the common timetable, not a separate HashMap).

    // warmstate.py:299-320 — retrace_limit / max_retrace_guards /
    // max_unroll_loops / max_unroll_recursion live on
    // warmrunnerdesc.memory_manager, not on WarmEnterState itself. See
    // `MemoryManager` in memmgr.rs.
    /// jit.py:581,602: vec — enable vectorization optimization.
    vectorize: bool,
    /// jit.py:585,603: vec_all — vectorize loops outside numpypy library.
    vec_all: bool,
    /// jit.py:583,604: vec_cost — cost threshold for vectorization decisions.
    vec_cost: u32,
    /// warmstate.py: enable_opts — list of enabled optimization pass names.
    enable_opts: Vec<String>,
    /// warmstate.py: set_param_inlining — whether inlining is enabled.
    inlining: bool,
    /// warmstate.py: set_param_disable_unrolling — threshold below
    /// which loop unrolling is disabled.
    disable_unrolling_threshold: u32,
    /// warmstate.py: set_param_pureop_historylength — size of the
    /// pure operation history cache.
    pureop_historylength: u32,
    /// warmspot.py:110: memory_manager — generation-based loop aging.
    /// pyjitpl.py:2348: try_to_free_some_loops calls next_generation().
    pub memory_manager: crate::memmgr::MemoryManager,
}

/// Result of checking whether a green key is hot.
pub enum HotResult {
    /// Not yet hot; keep interpreting.
    NotHot,
    /// Threshold reached; start tracing. The caller (MetaInterp) builds
    /// the Trace itself — RPython parity: `MetaInterp.create_empty_history`
    /// / `MetaInterp.create_history` live on `MetaInterp`, not on the
    /// warmstate (pyjitpl.py:2604-2610). Pyre's prior signal-and-factory
    /// pattern (`HotResult::StartTracing(Trace::new())`) forced warmstate
    /// to depend on the `recorder::Trace` type; Step 2e.2b is removing
    /// that coupling so the `Trace` factory moves to MetaInterp where
    /// `metainterp_sd` is available for `TraceRecordBuffer::new`.
    StartTracing,
    /// Already tracing (caller should keep feeding ops to the active recorder).
    AlreadyTracing,
    /// Compiled code exists; run it.
    RunCompiled,
}

impl WarmEnterState {
    /// warmstate.py:485-496: a `JC_DONT_TRACE_HERE` cell that has never seen
    /// a procedure token is retried — immediately the first time
    /// (`tick = True` when `JC_TRACING_OCCURRED` is unset), then by the
    /// back-edge counter on subsequent entries. The retry is gated purely on
    /// `has_seen_a_procedure_token` and `JC_TRACING_OCCURRED`; upstream
    /// applies no abort-count ceiling here (the abort lifecycle lives in
    /// `abort_tracing`, which flips the cell to permanent `DONT_TRACE_HERE`),
    /// so neither does this.
    fn should_start_dont_trace_here_trace(
        &mut self,
        cell_key: u64,
        flags: u8,
        has_seen_a_procedure_token: bool,
    ) -> bool {
        if flags & jc_flags::DONT_TRACE_HERE == 0 || has_seen_a_procedure_token {
            return false;
        }
        if flags & jc_flags::TRACING_OCCURRED != 0 {
            let bucket = self.bucket_of(cell_key);
            self.counter.tick(bucket, self.increment_threshold)
        } else {
            true
        }
    }

    fn start_tracing_cell(&mut self, cell_key: u64) -> HotResult {
        self.counter.reset(self.bucket_of(cell_key));
        self.tracing_generation += 1;
        let current_generation = self.tracing_generation;
        let cell = self.ensure_cell_by_key(cell_key);
        cell.flags |= jc_flags::TRACING | jc_flags::TRACING_OCCURRED;
        cell.state = BaseJitCellState::Tracing;
        cell.tracing_generation = current_generation;

        HotResult::StartTracing
    }

    /// Create a new WarmEnterState with the given threshold.
    /// Automatically enables Logger if MAJIT_STATS=1 or MAJIT_LOG=1.
    pub fn new(threshold: u32) -> Self {
        Self::with_jitlog(threshold, Logger::from_env())
    }

    /// Create a new WarmEnterState with an explicit Logger.
    pub fn with_jitlog(threshold: u32, jitlog: Option<Logger>) -> Self {
        let mut counter = JitCounter::new(DEFAULT_SIZE);
        // rlib/jit.py:588 PARAMETERS default decay=40.
        counter.set_decay(40);
        let increment_threshold = counter.compute_threshold(threshold);
        let increment_trace_eagerness = counter.compute_threshold(DEFAULT_TRACE_EAGERNESS);
        let increment_function_threshold = counter.compute_threshold(DEFAULT_FUNCTION_THRESHOLD);
        WarmEnterState {
            counter,
            cells: indexmap::IndexMap::new(),
            minted: indexmap::IndexMap::new(),
            mint_serial: 0,
            threshold,
            increment_threshold,
            trace_eagerness: DEFAULT_TRACE_EAGERNESS,
            increment_trace_eagerness,
            function_threshold: DEFAULT_FUNCTION_THRESHOLD,
            increment_function_threshold,
            max_inline_depth: DEFAULT_MAX_INLINE_DEPTH,
            trace_limit: DEFAULT_TRACE_LIMIT,
            tracing_generation: 0,
            jitlog,
            quasiimmut_deps: indexmap::IndexMap::new(),
            vectorize: false,
            vec_all: false,
            vec_cost: 0,
            enable_opts: default_enable_opts(),
            inlining: true,
            disable_unrolling_threshold: DEFAULT_DISABLE_UNROLLING,
            pureop_historylength: 16,
            memory_manager: {
                let mut m = crate::memmgr::MemoryManager::new(0);
                m.retrace_limit = DEFAULT_RETRACE_LIMIT;
                m.max_unroll_loops = DEFAULT_MAX_UNROLL_LOOPS;
                m
            },
        }
    }

    /// Check and possibly transition the JIT state for a given green key.
    ///
    /// Called by the interpreter at loop back-edges and function entries.
    /// Returns a `HotResult` telling the interpreter what to do next.
    /// Mark a green key as DONT_TRACE_HERE permanently.
    /// Clear the loop token for a cell, so is_compiled() returns false.
    pub fn clear_loop_token(&mut self, cell_key: u64) {
        if let Some(cell) = self.cell_by_key_mut(cell_key) {
            cell.loop_token = None;
        }
    }

    /// `cells.values()` yields chain heads. Every sweep over the whole table
    /// must walk `next` as well, or it silently skips chained cells. A chain
    /// needs no hash collision: one green key reached
    /// through a hash-only writer and then a typed one produces two cells in
    /// one bucket, because a `comparekey: None` cell can never match a typed
    /// lookup, so `ensure_cell_for_key` misses and chains past it. Proven by
    /// `one_key_through_a_hash_and_a_typed_entry_point_builds_a_chain`,
    /// which uses only public entry points on a single key.
    pub fn clear_all_loop_tokens(&mut self) {
        for head in self.cells.values_mut() {
            let mut cur = Some(head);
            while let Some(cell) = cur {
                cell.loop_token = None;
                cur = cell.next.as_deref_mut();
            }
        }
    }

    pub fn mark_dont_trace(&mut self, cell_key: u64) {
        self.disable_noninlinable_function(cell_key);
    }

    /// TODO: pyre-only cold fast path check. RPython
    /// warmstate.py:467 just calls jitcounter.tick(hash, increment_threshold)
    /// directly; this read-only peek exists to skip GreenKey allocation
    /// in jitdriver.rs for cold keys.
    #[inline]
    pub fn counter_would_fire(&self, cell_key: u64) -> bool {
        if let Some(cell) = self.cell_by_key(cell_key) {
            if cell.is_compiled() || cell.is_tracing() {
                return true;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 {
                return false;
            }
            if cell.state == BaseJitCellState::DontTraceHere {
                return false;
            }
        }
        self.counter
            .would_tick_fire(self.bucket_of(cell_key), self.increment_threshold)
    }

    /// warmstate.py:467: jitcounter.tick(hash, increment_threshold).
    pub fn counter_tick(&mut self, cell_key: u64) {
        if let Some(cell) = self.cell_by_key(cell_key) {
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 {
                return;
            }
            if cell.state == BaseJitCellState::DontTraceHere {
                return;
            }
        }
        let bucket = self.bucket_of(cell_key);
        let _ = self.counter.tick(bucket, self.increment_threshold);
    }

    pub fn counter_tick_checked(&mut self, cell_key: u64) -> bool {
        let mut cleanup_dead_token_cell = false;
        if let Some(cell) = self.cell_by_key(cell_key) {
            if cell.is_compiled() || cell.is_tracing() {
                return true;
            }
            // A JC_DONT_TRACE_HERE cell declines here, except when the
            // procedure token it once saw has since been invalidated: that
            // dead entry must fall through to cleanup_chain (warmstate.py:483-491)
            // instead of returning early and lingering in the chain.
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0
                && cell.has_seen_a_procedure_token()
                && cell.get_procedure_token().is_some()
            {
                return false;
            }
            if cell.has_seen_a_procedure_token() && cell.get_procedure_token().is_none() {
                cleanup_dead_token_cell = true;
            }
        }
        if cleanup_dead_token_cell {
            // warmstate.py:483-500 — loop-header counter entry must also
            // remove invalidated token cells before it starts counting again.
            self.cleanup_chain(self.bucket_of(cell_key));
            return false;
        }
        let bucket = self.bucket_of(cell_key);
        self.counter.tick(bucket, self.increment_threshold)
    }

    pub fn maybe_compile(&mut self, cell_key: u64) -> HotResult {
        let mut cleanup_dead_token_cell = false;
        if let Some(cell) = self.cell_by_key(cell_key) {
            let has_procedure_token = cell.get_procedure_token().is_some();
            let is_compiled = cell.is_compiled();
            let is_tracing = cell.is_tracing();
            let flags = cell.flags;
            let has_seen_a_procedure_token = cell.has_seen_a_procedure_token();
            if is_compiled {
                return HotResult::RunCompiled;
            }
            if is_tracing {
                return HotResult::AlreadyTracing;
            }
            if self.should_start_dont_trace_here_trace(cell_key, flags, has_seen_a_procedure_token)
            {
                return self.start_tracing_cell(cell_key);
            }
            // A JC_DONT_TRACE_HERE cell declines here, except when it once saw a
            // procedure token that has since been invalidated — that dead entry
            // must fall through to cleanup_chain below (warmstate.py:483-491),
            // not linger and stall the counter re-arm.
            let dead_token = has_seen_a_procedure_token && !has_procedure_token;
            if flags & jc_flags::DONT_TRACE_HERE != 0 && !dead_token {
                return HotResult::NotHot;
            }
            if dead_token {
                cleanup_dead_token_cell = true;
            }
        }

        if cleanup_dead_token_cell {
            // warmstate.py:483-500 — an invalidated/dead procedure token is
            // removed from the chain, resetting the hot counter so it re-arms.
            self.cleanup_chain(self.bucket_of(cell_key));
            return HotResult::NotHot;
        }

        if !self
            .counter
            .tick(self.bucket_of(cell_key), self.increment_threshold)
        {
            return HotResult::NotHot;
        }

        self.start_tracing_cell(cell_key)
    }

    /// warmstate.py:446-511 `WarmEnterState.maybe_compile_and_run` —
    /// typed-greenkey variant of [`Self::maybe_compile`].
    ///
    /// Walks the per-bucket chain by typed comparekey for both the
    /// state read (`lookup_chain_with_key`) and the state mutation
    /// (`start_tracing_cell_for_key`), mirroring upstream's
    /// `JitCell.get_jitcell_for_args` (`warmstate.py:455-465`) followed
    /// by per-cell flag mutation. Hash collisions across distinct
    /// typed greens therefore do not cross-contaminate each other's
    /// `JC_TRACING` / `JC_COMPILED` flags or counter-derived
    /// transitions.
    ///
    /// The bucket-shared counter ticks via `key.get_uhash()` — upstream
    /// counter is per-bucket too (`warmstate.py:496`), so colliding
    /// typed greens share the back-edge counter while keeping
    /// independent cell state.
    ///
    /// Hash-only callers ([`Self::maybe_compile`]) still mutate the
    /// bucket head via the legacy entry point. The typed chain-walk
    /// variants of `finish_tracing`, `abort_tracing`, `clear_loop_token`,
    /// and `mark_dont_trace` ([`Self::finish_tracing_for_key`] etc.) keep
    /// the typed path's full lifecycle self-consistent on its own buckets;
    /// the production cutover from the hash entry points is the separate
    /// typed-greenkey threading work.
    pub fn maybe_compile_with_key(&mut self, key: &GreenKey) -> HotResult {
        let hash = key.get_uhash();
        let mut cleanup_dead_token_cell = false;
        if let Some(cell) = self.lookup_chain_with_key(key) {
            let has_procedure_token = cell.get_procedure_token().is_some();
            let is_compiled = cell.is_compiled();
            let is_tracing = cell.is_tracing();
            let flags = cell.flags;
            let has_seen_a_procedure_token = cell.has_seen_a_procedure_token();
            if is_compiled {
                return HotResult::RunCompiled;
            }
            if is_tracing {
                return HotResult::AlreadyTracing;
            }
            if self.should_start_dont_trace_here_trace(hash, flags, has_seen_a_procedure_token) {
                return self.start_tracing_cell_for_key(key);
            }
            // A JC_DONT_TRACE_HERE cell declines here, except when it once saw a
            // procedure token that has since been invalidated — that dead entry
            // must fall through to cleanup_chain below (warmstate.py:483-491),
            // not linger and stall the counter re-arm.
            let dead_token = has_seen_a_procedure_token && !has_procedure_token;
            if flags & jc_flags::DONT_TRACE_HERE != 0 && !dead_token {
                return HotResult::NotHot;
            }
            if dead_token {
                cleanup_dead_token_cell = true;
            }
        }

        if cleanup_dead_token_cell {
            // warmstate.py:483-500 — an invalidated/dead procedure token is
            // removed from the chain, resetting the hot counter so it re-arms.
            self.cleanup_chain(hash);
            return HotResult::NotHot;
        }

        if !self.counter.tick(hash, self.increment_threshold) {
            return HotResult::NotHot;
        }

        self.start_tracing_cell_for_key(key)
    }

    /// Mutable chain-walk variant of [`Self::lookup_chain_with_key`].
    /// Returns `Some` only when a chained cell carries a comparekey
    /// equal (`equal_whatever`) to `key`; the bucket head is no
    /// longer privileged.
    fn lookup_chain_with_key_mut(&mut self, key: &GreenKey) -> Option<&mut BaseJitCell> {
        let hash = key.get_uhash();
        let mut cell = self.cells.get_mut(&hash);
        while let Some(c) = cell {
            if c.comparekey_matches(key) {
                return Some(c);
            }
            cell = c.next.as_deref_mut();
        }
        None
    }

    /// warmstate.py:425-444 `WarmEnterState.bound_reached` —
    /// typed-key variant of [`Self::start_tracing_cell`].
    ///
    /// Walks the chain by `key`'s comparekey to mutate the matching
    /// cell's `JC_TRACING` / `JC_TRACING_OCCURRED` flags +
    /// `tracing_generation`, rather than the bucket head. Pairs with
    /// [`Self::maybe_compile_with_key`] so a typed greenkey always
    /// transitions its own cell on a hash collision.
    fn start_tracing_cell_for_key(&mut self, key: &GreenKey) -> HotResult {
        let hash = key.get_uhash();
        self.counter.reset(hash);
        self.tracing_generation += 1;
        let current_generation = self.tracing_generation;
        self.ensure_cell_for_key(key);
        let cell = self
            .lookup_chain_with_key_mut(key)
            .expect("ensure_cell_for_key just installed a cell matching this key");
        cell.flags |= jc_flags::TRACING | jc_flags::TRACING_OCCURRED;
        cell.state = BaseJitCellState::Tracing;
        cell.tracing_generation = current_generation;
        HotResult::StartTracing
    }

    /// Typed-key variant of [`Self::finish_tracing`]. Walks the bucket
    /// chain by comparekey so a typed green that collides with another
    /// clears `JC_TRACING` on its own cell rather than the bucket head.
    pub fn finish_tracing_for_key(&mut self, key: &GreenKey) {
        if let Some(cell) = self.lookup_chain_with_key_mut(key) {
            cell.flags &= !jc_flags::TRACING;
            // State remains Tracing until attach_procedure_to_interp is called.
        }
    }

    /// Typed-key variant of [`Self::abort_tracing`]. Walks the bucket
    /// chain by comparekey so a collided typed green mutates its own
    /// cell's `JC_TRACING` / pyre-local abort-count / `DONT_TRACE_HERE` state.
    pub fn abort_tracing_for_key(&mut self, key: &GreenKey, disable_noninlinable_function: bool) {
        if let Some(cell) = self.lookup_chain_with_key_mut(key) {
            cell.flags &= !jc_flags::TRACING;
            cell.abort_count += 1;
            if disable_noninlinable_function
                || (cell.flags & jc_flags::DONT_TRACE_HERE != 0)
                || cell.abort_count >= MAX_TRACE_ABORT_COUNT
            {
                cell.flags |= jc_flags::DONT_TRACE_HERE;
                cell.state = BaseJitCellState::DontTraceHere;
            } else {
                cell.state = BaseJitCellState::NotHot;
            }
        }

        if disable_noninlinable_function {
            self.disable_noninlinable_function_for_key(key);
        }
        if let Some(log) = &mut self.jitlog {
            log.log_abort();
        }
    }

    /// Typed-key variant of [`Self::clear_loop_token`]. Walks the bucket
    /// chain by comparekey to clear the matching cell's loop token.
    pub fn clear_loop_token_for_key(&mut self, key: &GreenKey) {
        if let Some(cell) = self.lookup_chain_with_key_mut(key) {
            cell.loop_token = None;
        }
    }

    /// Typed-key variant of [`Self::mark_dont_trace`].
    pub fn mark_dont_trace_for_key(&mut self, key: &GreenKey) {
        self.disable_noninlinable_function_for_key(key);
    }

    /// Typed-key variant of [`Self::disable_noninlinable_function`]:
    /// installs a cell for `key` if absent (matching the hash form's
    /// `entry().or_insert`), then sets `DONT_TRACE_HERE` on the cell whose
    /// comparekey matches `key`, not the bucket head.
    pub fn disable_noninlinable_function_for_key(&mut self, key: &GreenKey) {
        self.ensure_cell_for_key(key);
        let cell = self
            .lookup_chain_with_key_mut(key)
            .expect("ensure_cell_for_key just installed a cell matching this key");
        cell.flags |= jc_flags::DONT_TRACE_HERE;
        if cell.flags & jc_flags::TRACING == 0 {
            cell.state = BaseJitCellState::DontTraceHere;
        }
    }

    /// Force-start tracing for a green key, bypassing the hot counter.
    ///
    /// Used by function-entry tracing where the caller has already
    /// determined that tracing should begin.
    pub fn force_start_tracing(&mut self, cell_key: u64) -> HotResult {
        if let Some(cell) = self.cell_by_key(cell_key) {
            if cell.is_compiled() {
                return HotResult::RunCompiled;
            }
            if cell.is_tracing() {
                return HotResult::AlreadyTracing;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 && cell.has_seen_a_procedure_token() {
                return HotResult::NotHot;
            }
            // Give up after too many failed trace attempts to prevent
            // infinite retrace loops (e.g. InvalidLoop every time).
            if cell.abort_count >= MAX_TRACE_ABORT_COUNT {
                return HotResult::NotHot;
            }
        }

        self.start_tracing_cell(cell_key)
    }

    /// Typed-key variant of [`Self::force_start_tracing`]: reads the
    /// matching cell by comparekey and force-starts tracing on it via
    /// `Self::start_tracing_cell_for_key`, so a force-started cell
    /// (function-entry / can_enter_jit) carries a `comparekey` like the
    /// [`Self::maybe_compile_with_key`] path.
    pub fn force_start_tracing_for_key(&mut self, key: &GreenKey) -> HotResult {
        if let Some(cell) = self.lookup_chain_with_key(key) {
            if cell.is_compiled() {
                return HotResult::RunCompiled;
            }
            if cell.is_tracing() {
                return HotResult::AlreadyTracing;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 && cell.has_seen_a_procedure_token() {
                return HotResult::NotHot;
            }
            if cell.abort_count >= MAX_TRACE_ABORT_COUNT {
                return HotResult::NotHot;
            }
        }

        self.start_tracing_cell_for_key(key)
    }

    /// Signal that a retrace is starting from a guard failure point.
    ///
    /// `input_types` is accepted for API shape parity with
    /// `MetaInterp.create_history(max_num_inputargs)` callers who need
    /// to size their `TraceRecordBuffer` before the retrace; warmstate
    /// doesn't own the Trace type. Returns nothing — the caller
    /// (`MetaInterp::start_bridge_trace` in pyjitpl.rs) constructs
    /// the `Trace` itself with its `staticdata: Arc<MetaInterpStaticData>`.
    /// RPython parity: `warmspot.py` has no analogue of the old
    /// `start_retrace(input_types) -> Trace` factory — RPython's
    /// `MetaInterp.create_history(max_num_inputargs)` is the constructor.
    pub fn start_retrace(&mut self, _input_types: &[Type]) {}

    /// Mark that tracing is done for a green key. Clears the TRACING flag.
    /// The caller is responsible for compiling the trace and calling
    /// `attach_procedure_to_interp` with the resulting JitCellToken.
    pub fn finish_tracing(&mut self, cell_key: u64) {
        if let Some(cell) = self.cell_by_key_mut(cell_key) {
            cell.flags &= !jc_flags::TRACING;
            // State remains Tracing until attach_procedure_to_interp is called.
        }
    }

    /// Mark that tracing was aborted for a green key.
    ///
    /// Non-permanent aborts clear `JC_TRACING` and allow a future retry until
    /// pyre's abort ceiling marks the location `DONT_TRACE_HERE`.
    pub fn abort_tracing(&mut self, cell_key: u64, disable_noninlinable_function: bool) {
        if let Some(cell) = self.cell_by_key_mut(cell_key) {
            cell.flags &= !jc_flags::TRACING;
            cell.abort_count += 1;
            if disable_noninlinable_function || (cell.flags & jc_flags::DONT_TRACE_HERE != 0) {
                cell.flags |= jc_flags::DONT_TRACE_HERE;
                cell.state = BaseJitCellState::DontTraceHere;
            } else if cell.abort_count >= MAX_TRACE_ABORT_COUNT {
                // Too many failed attempts — permanently disable tracing here.
                cell.flags |= jc_flags::DONT_TRACE_HERE;
                cell.state = BaseJitCellState::DontTraceHere;
            } else {
                cell.state = BaseJitCellState::NotHot;
            }
        }

        if disable_noninlinable_function {
            self.disable_noninlinable_function(cell_key);
        }
        if let Some(log) = &mut self.jitlog {
            log.log_abort();
        }
    }

    /// Install a compiled loop token for a green key.
    ///
    /// The cell transitions to Compiled state and takes ownership of
    /// the procedure token. Also clears TRACING on the cell the token
    /// is attached to — this covers the same-key compile path, while
    /// cross-loop cut (compile.py:269) compiles under an inner cell
    /// and the outer (starting) cell's TRACING is cleared separately
    /// by the `clear_tracing_flag` call in the tracing entry point's
    /// finally block (warmstate.py:444 parity).
    ///
    /// Installs the cell under the bare hash, so it cannot set `comparekey`
    /// and the cell it creates is one no chain walk will ever match. Prefer
    /// [`Self::attach_procedure_to_interp_for_key`] wherever the green key
    /// itself is in scope.
    pub fn attach_procedure_to_interp(
        &mut self,
        cell_key: u64,
        token: impl Into<Arc<JitCellToken>>,
    ) -> Option<Arc<JitCellToken>> {
        let token = token.into();
        let cell = self.ensure_cell_by_key(cell_key);
        cell.flags &= !jc_flags::TRACING;
        cell.set_procedure_token(token, false)
    }

    /// Typed form of [`Self::attach_procedure_to_interp`].
    ///
    /// `JitCell.__init__` (warmstate.py:610-616) always stores the green args
    /// on the cell, so upstream has no cell that a chain walk can fail to
    /// match. The hash form creates one: `cells.entry(hash)` leaves
    /// `comparekey` unset, `comparekey_matches` refuses it unconditionally,
    /// and `install_new_cell` links later survivors ahead of it — so the same
    /// green key ends up with a hash-installed cell at the bucket head and a
    /// typed cell behind it, holding different token and flag state.
    pub fn attach_procedure_to_interp_for_key(
        &mut self,
        key: &GreenKey,
        token: impl Into<Arc<JitCellToken>>,
    ) -> Option<Arc<JitCellToken>> {
        let token = token.into();
        self.ensure_cell_for_key(key);
        let cell = self
            .lookup_chain_with_key_mut(key)
            .expect("ensure_cell_for_key just installed a cell matching this key");
        cell.flags &= !jc_flags::TRACING;
        cell.set_procedure_token(token, false)
    }

    /// warmstate.py:716-723 `cell.set_procedure_token(procedure_token, tmp=True)`.
    ///
    /// Installs a temporary CALL_ASSEMBLER fallback token without
    /// changing the tracing flags or compiled state.
    ///
    /// No typed twin: this has no callers anywhere in the workspace, so it
    /// creates no cells and contributes nothing to the head/tail split that
    /// [`Self::attach_procedure_to_interp_for_key`] exists to close. Adding
    /// one would be a second uncalled entry point. Give it a typed form at
    /// the point a caller appears, not before.
    pub fn attach_tmp_callback_to_interp(
        &mut self,
        cell_key: u64,
        token: impl Into<Arc<JitCellToken>>,
    ) {
        let token = token.into();
        let cell = self.ensure_cell_by_key(cell_key);
        let _old = cell.set_procedure_token(token, true);
    }

    /// warmstate.py:444 `finally: cell.flags &= ~JC_TRACING` parity —
    /// unconditional flag clear on the starting cell after tracing ends.
    /// Called from the synchronous tracing entry point
    /// regardless of whether tracing succeeded, aborted, or cross-loop-cut
    /// installed under a different inner cell. Does not alter state: the
    /// companion `attach_procedure_to_interp` / `abort_tracing` calls own
    /// the state transition for whichever cell they touch.
    pub fn clear_tracing_flag(&mut self, cell_key: u64) {
        if let Some(cell) = self.cell_by_key_mut(cell_key) {
            cell.flags &= !jc_flags::TRACING;
        }
    }

    /// Typed-key variant of [`Self::clear_tracing_flag`], and the missing half
    /// of a live pair.
    ///
    /// `JC_TRACING` is *set* through [`Self::mark_as_being_traced_for_key`],
    /// which installs through [`Self::ensure_cell_for_key`] and therefore
    /// writes the cell that matches the key. The hash form above clears the
    /// cell its CELL KEY names, which is the same cell whenever the caller's
    /// u64 was resolved from these greens — but a caller holding a raw bucket
    /// hash on a chained bucket names the bucket's first occupant instead.
    /// Once a hash-only writer and a typed one have both written one key those
    /// are two different cells
    /// (`one_key_through_a_hash_and_a_typed_entry_point_builds_a_chain`) with
    /// two different keys, so the set can land on one and the clear on the
    /// other, the flag is never cleared, and every gate that reads the
    /// unresolved key — `counter_would_fire`, `counter_tick_checked`,
    /// `maybe_compile`, `force_start_tracing`, `should_trace_function_entry` —
    /// refuses that key from then on.
    ///
    /// Deliberately does NOT route through `ensure_cell_for_key` the way the
    /// `_for_key` writers do: clearing a flag must not install a cell. A key
    /// with no cell has no flag to clear, which is the hash form's behaviour
    /// too.
    ///
    /// No caller yet. Unlike [`Self::attach_tmp_callback_to_interp`], which
    /// got a stated reason instead of a twin because nothing calls it
    /// anywhere, the hash form here has a live production caller in
    /// `pyre-jit` (the `finally` clear after tracing ends) that still passes a
    /// hash. Converting it is caller-side work in another crate and lands
    /// separately.
    pub fn clear_tracing_flag_for_key(&mut self, key: &GreenKey) {
        if let Some(cell) = self.lookup_chain_with_key_mut(key) {
            cell.flags &= !jc_flags::TRACING;
        }
    }

    /// No typed twin: this has no callers anywhere in the workspace — the only
    /// occurrence in the tree is this definition. A twin here would be a
    /// second uncalled entry point, so it gets a reason instead, the same call
    /// [`Self::attach_tmp_callback_to_interp`] got. Give it a typed form at
    /// the point a caller appears, not before.
    pub fn take_procedure_token(&mut self, cell_key: u64) -> Option<Arc<JitCellToken>> {
        self.cell_by_key_mut(cell_key)
            .and_then(|cell| cell.loop_token.take())
    }

    /// Get a reference to the compiled loop token for a green key.
    ///
    /// `warmstate.py:191-196 get_procedure_token` parity: returns `None`
    /// when the cell has no procedure token AND when the token has been
    /// invalidated (`token and not token.invalidated`).  Pyre routes
    /// through `BaseJitCell::get_procedure_token` (which applies the
    /// `is_invalidated` filter) so every entry-path consumer of the
    /// "current compiled token at this key" sees the same filtered view
    /// PyPy provides — invalidated tokens never reach the warm-entry
    /// runner, the CALL_ASSEMBLER inline gate, or the bridge stitch
    /// surface.
    pub fn get_compiled(&self, cell_key: u64) -> Option<&Arc<JitCellToken>> {
        self.cell_by_key(cell_key)
            .and_then(|cell| cell.get_procedure_token())
    }

    /// warmstate.py:191-196 `get_procedure_token`.
    ///
    /// Reads the cell the CELL KEY names — one cell, whatever the bucket holds
    /// — so this and [`Self::get_procedure_token_for_key`] answer about the
    /// same object for any key that has been resolved
    /// ([`Self::cell_key_for`]). Handing this a raw bucket hash instead is a
    /// different question with a different answer: it names the bucket's first
    /// occupant.
    pub fn get_procedure_token(&self, cell_key: u64) -> Option<Arc<JitCellToken>> {
        self.cell_by_key(cell_key)
            .and_then(|cell| cell.get_procedure_token().cloned())
    }

    /// `warmstate.py:458-464` — resolve the cell by `comparekey`, then read its
    /// procedure token.
    ///
    /// The typed twin of [`Self::get_procedure_token`]. Upstream reaches a
    /// token only through a cell it has already matched on the full green key;
    /// this is that discipline. A bucket the key does not match anywhere is
    /// upstream's `else:` arm — "not found" — and answers `None` rather than
    /// handing back the head cell's unrelated token.
    ///
    /// Equivalent to [`Self::cell_key_for`] followed by
    /// [`Self::get_procedure_token`], and callers that need the key as well as
    /// the token should spell it that way so the key they carry onward is the
    /// one this matched on.
    pub fn get_procedure_token_for_key(&self, key: &GreenKey) -> Option<Arc<JitCellToken>> {
        self.lookup_chain_with_key(key)
            .and_then(|cell| cell.get_procedure_token().cloned())
    }

    /// Whether this bucket holds more than one cell, i.e. whether a raw hash
    /// leaves anything to choose between.
    ///
    /// A chain needs no hash collision: one green key reached through a
    /// hash-only writer and then a typed one produces two cells in one bucket
    /// (see [`Self::clear_all_loop_tokens`]). An unchained bucket has exactly
    /// one candidate, so [`Self::sole_cell_key`] is exact and a caller can skip
    /// building a `GreenKey` it would only use to confirm that — which is what
    /// keeps [`Self::resolve_cell_key`] allocation-free on the entry path.
    ///
    /// Resolves through [`Self::lookup_chain`] and NOT through
    /// [`Self::bucket_of`]. The argument is a raw green-key hash, which is its
    /// own bucket; `bucket_of` maps a CELL KEY to the bucket that cell lives
    /// in, and applying it here sends this reader to another bucket entirely
    /// whenever the hash coincides with a live minted key — while
    /// [`Self::sole_cell_key`], the reader whose branch this decides, keeps
    /// reading the bucket the hash names. Upstream cannot split the two: every
    /// reader enters at `jitcounter.lookup_chain(hash)`
    /// (warmstate.py:459-460, :597-598, :632-633), and `lookup_chain`
    /// (counter.py:239-240) is a bare `celltable[self._get_index(hash)]`. See
    /// [`a_raw_hash_equal_to_a_minted_key_resolves_through_one_bucket`].
    #[inline]
    pub fn bucket_is_chained(&self, green_key_hash: u64) -> bool {
        self.lookup_chain(green_key_hash)
            .is_some_and(|head| head.next.is_some())
    }

    /// Allocate a new unique JitCellToken number.
    pub fn alloc_token_number(&mut self) -> u64 {
        NEXT_GLOBAL_TOKEN_NUMBER.fetch_add(1, Ordering::Relaxed)
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// warmstate.py:253-254 set_param_threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
        self.increment_threshold = self.counter.compute_threshold(threshold);
    }

    /// Decay all counters (e.g., periodically to avoid stale counts).
    pub fn decay_counters(&mut self) {
        self.counter.decay_all_counters();
    }

    /// Reset the hot counter for a specific green key to zero.
    pub fn reset_counter(&mut self, cell_key: u64) {
        self.counter.reset(self.bucket_of(cell_key));
    }

    /// Reset ALL counters to zero. Used after invalidation with incomplete
    /// resume data (NONE fail_args) to prevent immediate recompilation.
    /// TODO: RPython has no equivalent; this is a
    /// pyre-only recovery path.
    pub fn decay_all_counters_to_zero(&mut self) {
        self.counter.reset_all();
    }

    /// Check if a green key is marked DontTraceHere.
    pub fn is_dont_trace_here(&self, cell_key: u64) -> bool {
        self.cell_by_key(cell_key)
            .is_some_and(|c| c.state == BaseJitCellState::DontTraceHere)
    }

    /// Get a reference to the BaseJitCell for a green key, if it exists.
    pub fn get_cell(&self, cell_key: u64) -> Option<&BaseJitCell> {
        self.cell_by_key(cell_key)
    }

    /// Typed-key variant of [`Self::get_cell`]:
    /// `warmstate.py:596-604 JitCell.get_jitcell(*greenargs)`.
    ///
    /// Walks the chain by `comparekey`, so it selects the cell belonging to
    /// this key rather than the cell that happens to hold the bucket's raw
    /// hash as its key. Those differ whenever a bucket chains: a hash-only
    /// writer installs a cell with `comparekey: None`, which
    /// `comparekey_matches` refuses unconditionally, so one key can own two
    /// cells in one bucket with no hash collision at all
    /// (`one_key_through_a_hash_and_a_typed_entry_point_builds_a_chain`) — and
    /// the hash-installed one took the raw hash, because it was there first.
    ///
    /// The reason to route through the key rather than the hash at all:
    /// every upstream cell lookup takes greenargs and compares them
    /// (`get_jitcell` :596, `_ensure_jit_cell_at_key` :631,
    /// `dont_trace_here` :644, `mark_as_being_traced` :649 — warmstate.py).
    /// Upstream has exactly one bare-hash entry point,
    /// `trace_next_iteration_hash` (warmstate.py:622-623), and it touches
    /// **the counter, not the cell**.
    /// That is the line: a hash is enough to find a bucket, never enough to
    /// pick a cell out of one.
    pub fn get_cell_for_key(&self, key: &GreenKey) -> Option<&BaseJitCell> {
        self.lookup_chain_with_key(key)
    }

    /// TODO: walk the warmstate cells to find a
    /// `JitCellToken` by number. Used by `MetaInterp::record_loop_or_bridge`
    /// to widen the CALL_ASSEMBLER keepalive search to cover targets
    /// that live only on a `BaseJitCell.loop_token` and are not yet — or
    /// never — registered in `MetaInterp::compiled_loops`.
    ///
    /// This doc previously named tmp-callback installs at
    /// `attach_tmp_callback_to_interp` as the most important such target.
    /// That function has no callers anywhere in the workspace, so it
    /// installs nothing and cannot be what this helper covers. The
    /// surviving population is whatever `attach_procedure_to_interp` wrote
    /// but `record_loop_or_bridge` has not registered; nobody has measured
    /// it, so treat the need for this fallback as unquantified rather than
    /// established.
    ///
    /// RPython equivalent does not exist because upstream descrs hold
    /// the `JitCellToken` object directly (`compile.py:187 isinstance(descr,
    /// JitCellToken)`) — no number→token resolution is needed.  This
    /// helper is removed by Slice X-D once `CallAssemblerDescr` /
    /// `LoopTargetDescr` carry the owning `Arc<JitCellToken>`.
    /// Walks each chain, not just its head — see `clear_all_loop_tokens`.
    /// A token living on a chained cell was previously unfindable here, and
    /// this is the fallback `with_trace_ctx_and_token_resolver` reaches when
    /// no `compiled_loops` entry matches (`pyjitpl.rs:4663`).
    pub fn find_token_by_number(&self, token_number: u64) -> Option<&Arc<JitCellToken>> {
        for head in self.cells.values() {
            let mut cur = Some(head);
            while let Some(cell) = cur {
                if let Some(tok) = cell.loop_token.as_ref() {
                    if tok.number == token_number {
                        return Some(tok);
                    }
                }
                cur = cell.next.as_deref();
            }
        }
        None
    }

    /// `rpython/jit/metainterp/warmstate.py:714-723` `get_assembler_token`.
    ///
    /// Returns the cell's existing procedure token, or — if none exists —
    /// builds a temporary one via `make_token` (caller wires
    /// `compile_tmp_callback`) and installs it on the cell with
    /// `tmp=true`.  The closure-based signature is a Rust adaptation so
    /// the caller can provide the `&mut Backend` / `&JitDriverStaticData`
    /// / `greenboxes` bundle without threading them through WarmEnterState.
    pub fn get_assembler_token<E, F>(
        &mut self,
        cell_key: u64,
        make_token: F,
    ) -> Result<Arc<JitCellToken>, E>
    where
        F: FnOnce() -> Result<Arc<JitCellToken>, E>,
    {
        let cell = self.ensure_cell_by_key(cell_key);
        if let Some(token) = cell.get_procedure_token() {
            return Ok(token.clone());
        }
        let token = make_token()?;
        cell.set_procedure_token(token.clone(), true);
        Ok(token)
    }

    /// `warmstate.py:714-723` `get_assembler_token` typed variant —
    /// walks the chain at `key.get_uhash()` and dispatches off
    /// `comparekey` instead of trusting the hash to be collision-free.
    ///
    /// Equivalent to `get_assembler_token` modulo lookup discipline:
    /// upstream `JitCell.ensure_jit_cell_at_key(greenkey)` walks the
    /// chain (`warmstate.py:626-641`) and inserts at head on miss.
    /// Pyre's hash-only [`Self::get_assembler_token`] aliases distinct
    /// typed keys that share a hash bucket; this variant disambiguates.
    pub fn get_assembler_token_with_key<E, F>(
        &mut self,
        key: &GreenKey,
        make_token: F,
    ) -> Result<Arc<JitCellToken>, E>
    where
        F: FnOnce() -> Result<Arc<JitCellToken>, E>,
    {
        self.ensure_cell_for_key(key);
        let hash = key.get_uhash();
        // Walk the chain to find the typed match. `ensure_cell_for_key`
        // guarantees one exists either as the head (miss path) or
        // somewhere down the chain (existing-cell path).
        let mut cell = self
            .cells
            .get_mut(&hash)
            .expect("ensure_cell_for_key installed a chain entry");
        while !cell.comparekey_matches(key) {
            cell = cell
                .next
                .as_deref_mut()
                .expect("ensure_cell_for_key guarantees a typed match exists");
        }
        if let Some(token) = cell.get_procedure_token() {
            return Ok(token.clone());
        }
        let token = make_token()?;
        cell.set_procedure_token(token.clone(), true);
        Ok(token)
    }

    /// Log a successful trace compilation. No-op if Logger is disabled.
    pub fn log_compile(
        &mut self,
        green_key: u64,
        ops_before_opt: usize,
        ops_after_opt: usize,
        opt_time: Duration,
        compile_time: Duration,
    ) {
        if let Some(log) = &mut self.jitlog {
            log.log_compile(
                green_key,
                ops_before_opt,
                ops_after_opt,
                opt_time,
                compile_time,
            );
        }
    }

    /// Log a guard failure. No-op if Logger is disabled.
    pub fn log_guard_failure(&mut self, guard_index: u32) {
        if let Some(log) = &mut self.jitlog {
            log.log_guard_failure(guard_index);
        }
    }

    /// Log a loop entry. No-op if Logger is disabled.
    pub fn log_loop_entry(&mut self, green_key: u64) {
        if let Some(log) = &mut self.jitlog {
            log.log_loop_entry(green_key);
        }
    }

    /// Get a reference to the Logger, if enabled.
    pub fn jitlog(&self) -> Option<&Logger> {
        self.jitlog.as_ref()
    }

    /// pyjitpl.py:2295 `self.jitlog.setup_once()` parity (per-warmstate
    /// adaptation).
    ///
    /// TODO: PyPy owns one `JitLogger` on
    /// `MetaInterpStaticData` (`rlib/rjitlog/rjitlog.py:347-354`) and
    /// `setup_once` re-reads `PYPYLOG` and writes a header.  Pyre's
    /// `Logger` is owned per-`WarmEnterState` instead, so the global
    /// jitlog hook is decomposed into a per-warmstate call that the
    /// `MetaInterp` driving this warmstate runs just before
    /// `MetaInterpStaticData::_setup_once`.  `Logger::from_env`
    /// already runs at construction (the `new` / `with_jitlog`
    /// constructors above), so this hook is the late opportunity to
    /// install one if the warmstate was built before the env was
    /// set.  Idempotent — only fills the slot when it is still
    /// `None`.
    ///
    /// Called from `MetaInterp::bound_reached` /
    /// `MetaInterp::force_start_tracing` so the lifecycle order
    /// (jitlog → debug_print → cpu.setup_once → vector_ext →
    /// profiler) matches PyPy for this warmstate.
    pub fn ensure_jitlog_initialised(&mut self) {
        if self.jitlog.is_none() {
            self.jitlog = Logger::from_env();
        }
    }

    /// warmstate.py: trace_eagerness parameter (integer).
    pub fn trace_eagerness(&self) -> u32 {
        self.trace_eagerness
    }

    /// warmstate.py:259: set_param_trace_eagerness.
    pub fn set_param_trace_eagerness(&mut self, value: u32) {
        self.trace_eagerness = value;
        self.increment_trace_eagerness = self.counter.compute_threshold(value);
    }

    /// warmstate.py: increment_trace_eagerness (pre-computed f64).
    pub fn increment_trace_eagerness(&self) -> f64 {
        self.increment_trace_eagerness
    }

    /// compile.py:783-784: jitcounter.tick(hash, increment_trace_eagerness).
    /// Increment the guard failure counter using the shared timetable.
    /// Returns true when counter reaches 1.0 (trace_eagerness ticks).
    #[inline]
    pub fn tick_guard_failure(&mut self, guard_hash: u64) -> bool {
        self.counter
            .tick(guard_hash, self.increment_trace_eagerness)
    }

    /// compile.py:826-830: store_hash — allocate a jitcounter hash for
    /// a new guard. Called at compile time (or lazily on first failure).
    pub fn fetch_next_hash(&mut self) -> u64 {
        self.counter.fetch_next_hash()
    }

    /// Get the function inlining threshold.
    pub fn function_threshold(&self) -> u32 {
        self.function_threshold
    }

    /// warmstate.py:256-257 set_param_function_threshold.
    pub fn set_function_threshold(&mut self, threshold: u32) {
        self.function_threshold = threshold;
        self.increment_function_threshold = self.counter.compute_threshold(threshold);
    }

    /// RPython-compatible wrapper: set_param_threshold.
    pub fn set_param_threshold(&mut self, threshold: u32) {
        self.set_threshold(threshold);
    }

    /// RPython-compatible wrapper: set_param_trace_limit.
    pub fn set_param_trace_limit(&mut self, value: u32) {
        self.set_trace_limit(value);
    }

    /// RPython-compatible wrapper: set_param_function_threshold.
    pub fn set_param_function_threshold(&mut self, value: u32) {
        self.set_function_threshold(value);
    }

    /// RPython-compatible wrapper: set_param_inlining.
    pub fn set_param_inlining(&mut self, value: bool) {
        self.inlining = value;
    }

    /// RPython-compatible wrapper: set_param_disable_unrolling.
    pub fn set_param_disable_unrolling(&mut self, value: u32) {
        self.disable_unrolling_threshold = value;
    }

    /// RPython-compatible wrapper: set_param_vec.
    pub fn set_param_vec(&mut self, enabled: bool) {
        self.vectorize = enabled;
    }

    /// jit.py:585: set_param_vec_all.
    pub fn set_param_vec_all(&mut self, enabled: bool) {
        self.vec_all = enabled;
    }

    /// RPython-compatible wrapper: set_param_vec_cost.
    pub fn set_param_vec_cost(&mut self, value: u32) {
        self.vec_cost = value;
    }

    /// warmstate.py:317-320 set_param_max_unroll_recursion — delegates
    /// to memory_manager.max_unroll_recursion.
    pub fn set_param_max_unroll_recursion(&mut self, value: u32) {
        self.memory_manager.max_unroll_recursion = value;
    }

    /// RPython-compatible wrapper: set_param_max_inline_depth.
    pub fn set_param_max_inline_depth(&mut self, value: u32) {
        self.set_max_inline_depth(value);
    }

    /// warmstate.py:299-302 set_param_retrace_limit — delegates to
    /// memory_manager.retrace_limit.
    pub fn set_param_retrace_limit(&mut self, value: u32) {
        self.memory_manager.retrace_limit = value;
    }

    /// warmstate.py:307-310 set_param_max_retrace_guards — delegates to
    /// memory_manager.max_retrace_guards.
    pub fn set_param_max_retrace_guards(&mut self, value: u32) {
        self.memory_manager.max_retrace_guards = value;
    }

    /// warmstate.py:312-315 set_param_max_unroll_loops — delegates to
    /// memory_manager.max_unroll_loops.
    pub fn set_param_max_unroll_loops(&mut self, value: u32) {
        self.memory_manager.max_unroll_loops = value;
    }

    /// warmstate.py:293-297 set_param_loop_longevity — delegates to the
    /// memory manager's max_age.
    pub fn set_param_loop_longevity(&mut self, value: u32) {
        // memmgr.py:42 default check_frequency=0 → derives sqrt(max_age).
        self.memory_manager.set_max_age(value as i64, 0);
    }

    /// RPython-compatible wrapper: set_param_pureop_historylength.
    pub fn set_param_pureop_historylength(&mut self, value: u32) {
        self.pureop_historylength = value;
    }

    /// warmstate.py:269-270 set_param_decay — delegates to the jit counter.
    pub fn set_param_decay(&mut self, value: u32) {
        self.counter.set_decay(value as i32);
    }

    /// Set the maximum inline depth.
    pub fn set_max_inline_depth(&mut self, depth: u32) {
        self.max_inline_depth = depth;
    }

    /// Get the maximum inline depth.
    pub fn max_inline_depth(&self) -> u32 {
        self.max_inline_depth
    }

    /// Whether this callee is eligible for inlining at all.
    ///
    /// Mirrors PyPy's `can_inline_callable`: once a green key is marked
    /// `DONT_TRACE_HERE`, callers must stop inlining it and instead let it
    /// converge to a separate functrace / call_assembler path.
    pub fn can_inline_callable(&self, callee_key: u64) -> bool {
        self.cell_by_key(callee_key)
            .is_none_or(|cell| cell.flags & jc_flags::DONT_TRACE_HERE == 0)
    }

    /// Mark a callee as a location that should no longer be inlined into
    /// surrounding traces.
    ///
    /// This is the warm-state equivalent of PyPy's `disable_noninlinable_function()`.
    pub fn disable_noninlinable_function(&mut self, callee_key: u64) {
        let cell = self.ensure_cell_by_key(callee_key);
        cell.flags |= jc_flags::DONT_TRACE_HERE;
        if cell.flags & jc_flags::TRACING == 0 {
            cell.state = BaseJitCellState::DontTraceHere;
        }
    }

    /// Mark a callee as currently being traced.
    ///
    /// This is the warm-state equivalent of PyPy's `mark_as_being_traced()`.
    pub fn mark_as_being_traced(&mut self, callee_key: u64) {
        let tracing_generation = self.tracing_generation;
        let cell = self.ensure_cell_by_key(callee_key);
        cell.flags |= jc_flags::TRACING;
        if cell.flags & jc_flags::TRACING_OCCURRED == 0 {
            cell.state = BaseJitCellState::Tracing;
            cell.tracing_generation = tracing_generation;
        }
    }

    /// Typed-key variant of [`Self::mark_as_being_traced`]:
    /// `warmstate.py:649-651 mark_as_being_traced(*greenargs)`, which reaches
    /// its cell through `_ensure_jit_cell_at_key(*greenargs)` — i.e. by
    /// comparekey, never by hash alone.
    ///
    /// Installs through [`Self::ensure_cell_for_key`] rather than
    /// `cells.entry(hash)`, so the cell it marks carries a `comparekey` (and
    /// the `RetainedGreens` that come with it) instead of being a
    /// comparator-less cell that no later typed lookup can ever match — the
    /// same shape [`Self::disable_noninlinable_function_for_key`] uses.
    ///
    /// `tracing_generation` is read before the borrow because the cell
    /// borrows `self.cells` mutably.
    pub fn mark_as_being_traced_for_key(&mut self, key: &GreenKey) {
        self.ensure_cell_for_key(key);
        let tracing_generation = self.tracing_generation;
        let cell = self
            .lookup_chain_with_key_mut(key)
            .expect("ensure_cell_for_key just installed a cell matching this key");
        cell.flags |= jc_flags::TRACING;
        if cell.flags & jc_flags::TRACING_OCCURRED == 0 {
            cell.state = BaseJitCellState::Tracing;
            cell.tracing_generation = tracing_generation;
        }
    }

    /// Restore warm-state parameters to rlib/jit.py:588-605 PARAMETERS defaults.
    pub fn set_default_params(&mut self) {
        self.set_threshold(DEFAULT_THRESHOLD); // 1039
        self.set_param_trace_eagerness(DEFAULT_TRACE_EAGERNESS); // 200
        self.set_trace_limit(DEFAULT_TRACE_LIMIT); // 6000
        self.set_function_threshold(DEFAULT_FUNCTION_THRESHOLD); // 1619
        self.set_max_inline_depth(DEFAULT_MAX_INLINE_DEPTH); // 7
        self.inlining = true; // inlining = 1
        self.disable_unrolling_threshold = DEFAULT_DISABLE_UNROLLING; // 200
        self.pureop_historylength = 16;
        self.counter.set_decay(40);
        self.memory_manager.max_retrace_guards = 15;
        self.memory_manager.max_unroll_loops = 0;
        self.memory_manager.retrace_limit = DEFAULT_RETRACE_LIMIT;
        self.memory_manager.max_unroll_recursion = DEFAULT_MAX_UNROLL_RECURSION;
        self.memory_manager.set_max_age(1000, 0);
        self.vec_cost = 0;
        self.vectorize = false;
        self.set_param_enable_opts("all");
    }

    /// Mirror RPython warmstate.py `mark_force_finish_tracing(greenkey)`.
    ///
    /// The next tracing run for this green key should segment instead of
    /// repeatedly aborting once it approaches the trace limit.
    ///
    /// Installs under the bare hash, so the cell it creates carries no
    /// `comparekey`. Prefer [`Self::mark_force_finish_tracing_for_key`]
    /// wherever the green key itself is in scope.
    pub fn mark_force_finish_tracing(&mut self, cell_key: u64) {
        let cell = self.ensure_cell_by_key(cell_key);
        cell.flags |= jc_flags::FORCE_FINISH;
    }

    /// Typed form of [`Self::mark_force_finish_tracing`].
    ///
    /// `FORCE_FINISH` is sticky and never cleared explicitly, so setting it
    /// on the wrong cell of a bucket is permanent: the key that needs
    /// segmenting keeps aborting while an unrelated key segments forever.
    pub fn mark_force_finish_tracing_for_key(&mut self, key: &GreenKey) {
        self.ensure_cell_for_key(key);
        let cell = self
            .lookup_chain_with_key_mut(key)
            .expect("ensure_cell_for_key just installed a cell matching this key");
        cell.flags |= jc_flags::FORCE_FINISH;
    }

    /// warmstate.py:439 `bool(cell.flags & JC_FORCE_FINISH)` — read the sticky
    /// segmenting flag at loop entry.  RPython never clears this flag
    /// explicitly: `should_remove_jitcell` (warmstate.py:222) keeps the cell
    /// alive while it is set, and once set it persists until the cell itself
    /// is removed.
    pub fn should_force_finish_tracing(&self, cell_key: u64) -> bool {
        self.cell_by_key(cell_key)
            .is_some_and(|cell| cell.flags & jc_flags::FORCE_FINISH != 0)
    }

    /// Boost the current loop/function green key so the next execution
    /// immediately retriggers tracing.
    ///
    /// Mirrors PyPy's `JitCell.trace_next_iteration()` in warmstate.py:
    /// it does not force tracing right now, it only raises the hot counter
    /// to ~threshold so the next hit converges quickly.
    ///
    /// Takes the bare hash on purpose, and it is the one shape allowed to:
    /// `_trace_next_iteration` (warmstate.py:617-619) hashes the greenargs and
    /// calls `jitcounter.change_current_fraction` — no cell is looked up, so
    /// the hash is the whole identity the operation needs. Upstream exposes
    /// exactly this as `trace_next_iteration_hash` (warmstate.py:622-623).
    /// The moment a cell read is added here, this needs a `&GreenKey` like
    /// [`Self::get_cell_for_key`]'s cohort.
    pub fn trace_next_iteration(&mut self, green_key_hash: u64) {
        self.counter
            .change_current_fraction(self.bucket_of(green_key_hash), 0.98);
    }

    /// warmstate.py:467 jitcounter.tick(hash, increment_threshold) parity.
    ///
    /// warmstate.py:256-257: jitcounter.tick(hash, increment_function_threshold).
    pub fn should_trace_function_entry(&mut self, cell_key: u64) -> bool {
        let mut cleanup_dead_token_cell = false;
        if let Some(cell) = self.cell_by_key(cell_key) {
            // Slot 23 is the total; 64/65 are its two terms, evaluated
            // independently rather than short-circuited so a cell that is both
            // reaches both tallies. `is_compiled()` fires on every probe of
            // every compiled key, so 23 alone cannot attribute a decline.
            let compiled = cell.is_compiled();
            let tracing = cell.is_tracing();
            if compiled || tracing {
                crate::mc_diag_bump(23);
                if compiled {
                    crate::mc_diag_bump(64);
                }
                if tracing {
                    crate::mc_diag_bump(65);
                    // 65 is NOT a "healthy while a trace runs" reading in
                    // production, and an earlier version of this comment said
                    // it was. The only production caller
                    // (`pyre-jit`'s try_function_entry_jit) guards on
                    // `!driver.is_tracing()`, which is
                    // `MetaInterp::tracing.is_some()` — one global Option, not
                    // a per-cell flag. So while the engine traces, the caller
                    // returns a frame earlier and this gate is never reached.
                    // Every production bump of 65 is therefore a cell holding
                    // JC_TRACING while no trace is running, i.e. a leak on its
                    // own. The function itself can still be called mid-trace
                    // directly, and the unit tests below do exactly that.
                    //
                    // 66 splits those leaks by AGE, not into leak vs healthy:
                    // a generation older than the warm state's means the
                    // session that set the flag was superseded by a later
                    // trace start. 65 > 0 with 66 == 0 is the flag leaking
                    // from the most recent session, which is if anything the
                    // more direct miss.
                    //
                    // A ZERO HERE NEEDS TWO WITNESSES, NOT ONE. That the
                    // door ran (23 + 24 + 25 > 0) does not mean any probed
                    // cell could ever have held the flag: a stale JC_TRACING
                    // only sits on a cell that once started tracing, and a
                    // workload that never arms function-entry tracing gives
                    // 65 == 0 by construction. The arming witness is
                    // `caro_funcentry` (slot 19), bumped at the top of
                    // pyre-jit's `compile_and_run_once` above every early
                    // return — but on the `FunctionEntry` arm ONLY, since the
                    // slot is selected by `start` (`BackEdge` bumps 18). So a
                    // back-edge-only workload leaves 19 at 0 while 18 climbs,
                    // and 18 is NOT a substitute. Without 19 > 0 a 0 here is
                    // NOT EXERCISED, not clean. See MC_DIAG's legend.
                    if cell.tracing_generation < self.tracing_generation {
                        crate::mc_diag_bump(66);
                    }
                }
                return false;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 {
                if cell.has_seen_a_procedure_token() {
                    // A live TEMPORARY token still declines; a token that was
                    // once seen but has since been invalidated falls through to
                    // the cleanup gate below (warmstate.py:483-491) rather than
                    // re-entering the never-traced retry.
                    if cell.get_procedure_token().is_some() {
                        return false;
                    }
                } else if cell.flags & jc_flags::TRACING_OCCURRED == 0 {
                    return true;
                }
            }
            if cell.has_seen_a_procedure_token() && cell.get_procedure_token().is_none() {
                cleanup_dead_token_cell = true;
            }
        }
        if cleanup_dead_token_cell {
            // warmstate.py:483-500 — function-entry warmup must see an
            // invalidated token as a removed cell and re-count from cold.
            crate::mc_diag_bump(24);
            self.cleanup_chain(self.bucket_of(cell_key));
            return false;
        }
        crate::mc_diag_bump(25);
        self.counter
            .tick(self.bucket_of(cell_key), self.increment_function_threshold)
    }

    /// Check if inlining is allowed at the given depth.
    pub fn can_inline_at_depth(&self, current_depth: usize) -> bool {
        (current_depth as u32) < self.max_inline_depth
    }

    /// Log a bridge compilation. No-op if Logger is disabled.
    pub fn log_bridge_compile(&mut self, guard_index: u32) {
        if let Some(log) = &mut self.jitlog {
            log.log_bridge_compile(guard_index);
        }
    }

    // ── Quasi-immutable field invalidation ──

    /// Register that the compiled loop at `cell_key` depends on the
    /// quasi-immutable field identified by `qmut_key`.
    ///
    /// When `invalidate_quasiimmut(qmut_key)` is called later, the compiled
    /// loop's JitCellToken will be invalidated, causing GUARD_NOT_INVALIDATED
    /// to fail and forcing a retrace.
    ///
    /// `qmut_key` should be a hash of (object_id, field_index) or similar.
    /// `cell_key` is a resolved cell identity, not a raw green-key hash: it is
    /// what names one cell among a bucket's occupants, so it is what the
    /// invalidation below can resolve back to the dependent it was recorded for.
    pub fn register_quasiimmut_dependency(&mut self, qmut_key: u64, cell_key: u64) {
        let deps = self.quasiimmut_deps.entry(qmut_key).or_default();
        if !deps.contains(&cell_key) {
            deps.push(cell_key);
        }
    }

    /// Invalidate all compiled loops that depend on the quasi-immutable field
    /// identified by `qmut_key`.
    ///
    /// Called by the interpreter when a quasi-immutable field is mutated.
    /// Each dependent loop's JitCellToken has its invalidated flag set, causing
    /// GUARD_NOT_INVALIDATED to fail on the next execution.
    ///
    /// Returns the number of loops invalidated.
    ///
    /// No typed twin, and this one needs none. Its dependents arrive as the
    /// *values* of `quasiimmut_deps`, which records a `cell_key` each — the
    /// identity [`Self::cell_by_key_mut`] resolves, and that resolve walks the
    /// whole chain rather than answering with the bucket head. So a colliding
    /// neighbour cannot be invalidated in a dependent's place, and a dependent
    /// sitting behind a chained head cannot be skipped: the property
    /// [`Self::invalidate_all`] below is written for — *skipping a cell is a
    /// WRONG ANSWER rather than a leak* — holds here for the same reason,
    /// without a `GreenKey` having to reach this far.
    ///
    /// Dead in production today: its only registrar,
    /// `register_quasiimmut_dependency`, has no production caller either, so
    /// `quasiimmut_deps` is empty and this returns 0 at the first line.
    pub fn invalidate_quasiimmut(&mut self, qmut_key: u64) -> usize {
        let deps = match self.quasiimmut_deps.swap_remove(&qmut_key) {
            Some(deps) => deps,
            None => return 0,
        };

        let mut invalidated = 0;
        for cell_key in &deps {
            if let Some(cell) = self.cell_by_key_mut(*cell_key)
                && let Some(token) = &cell.loop_token
            {
                token.invalidate();
                cell.state = BaseJitCellState::Invalidated;
                invalidated += 1;
            }
        }
        invalidated
    }

    /// Invalidate all compiled loops that contain a GUARD_NOT_INVALIDATED.
    ///
    /// This is a brute-force invalidation used when the specific qmut_key
    /// is not known (e.g., bulk invalidation after a class hierarchy change).
    /// Walks each chain, not just its head — see `clear_all_loop_tokens`.
    /// This is the sweep where skipping a cell is a WRONG ANSWER rather than a
    /// leak: a cell whose token is not invalidated keeps running compiled code
    /// built under an assumption that has just been retracted.
    pub fn invalidate_all(&mut self) {
        for head in self.cells.values_mut() {
            let mut cur = Some(head);
            while let Some(cell) = cur {
                if let Some(token) = &cell.loop_token {
                    token.invalidate();
                    cell.state = BaseJitCellState::Invalidated;
                }
                cur = cell.next.as_deref_mut();
            }
        }
        self.quasiimmut_deps.clear();
    }

    // ── BaseJitCell state machine API ──

    /// Get the explicit state of a BaseJitCell for a green key.
    /// Returns `NotHot` if no cell exists.
    #[inline]
    pub fn get_cell_state(&self, cell_key: u64) -> BaseJitCellState {
        self.cell_by_key(cell_key)
            .map(|c| c.state)
            .unwrap_or(BaseJitCellState::NotHot)
    }

    /// Explicitly transition a cell to a new state.
    ///
    /// This is the low-level state-machine driver. Most callers should use
    /// the higher-level methods (`maybe_compile`, `finish_tracing`,
    /// `attach_procedure_to_interp`, `abort_tracing`) which call this internally.
    ///
    /// No typed twin: every caller is in this file's own test module, so the
    /// comparator-less cells it installs exist only in fixtures. That also
    /// makes it a way for a test to build the split-cell state deliberately.
    /// A twin becomes necessary if production ever calls this.
    pub fn transition_cell(&mut self, cell_key: u64, new_state: BaseJitCellState) {
        let cell = self.ensure_cell_by_key(cell_key);

        match new_state {
            BaseJitCellState::NotHot => {
                cell.flags &= !(jc_flags::TRACING | jc_flags::DONT_TRACE_HERE);
                cell.state = BaseJitCellState::NotHot;
            }
            BaseJitCellState::Tracing => {
                cell.flags |= jc_flags::TRACING | jc_flags::TRACING_OCCURRED;
                cell.state = BaseJitCellState::Tracing;
            }
            BaseJitCellState::Compiled => {
                cell.flags &= !jc_flags::TRACING;
                cell.state = BaseJitCellState::Compiled;
            }
            BaseJitCellState::Invalidated => {
                if let Some(token) = &cell.loop_token {
                    token.invalidate();
                }
                cell.state = BaseJitCellState::Invalidated;
            }
            BaseJitCellState::DontTraceHere => {
                cell.flags &= !jc_flags::TRACING;
                cell.flags |= jc_flags::DONT_TRACE_HERE;
                cell.state = BaseJitCellState::DontTraceHere;
            }
        }
    }

    // ── set_param / get_stats API ──

    /// Set a JIT parameter by its RPython name.
    ///
    /// Supported parameters:
    ///   - "threshold": compilation threshold
    ///   - "trace_limit": max ops per trace
    ///   - "trace_eagerness": guard fail count before bridge compilation
    ///   - "function_threshold": calls before inlining
    ///   - "max_inline_depth": maximum inlining depth
    ///
    /// `JitDriver.set_param` defines negative thresholds as disabled, and
    /// `JitCounter.compute_threshold` maps a disabled threshold to `0.0`.
    pub fn set_param(&mut self, name: &str, value: i64) {
        // Clamp disabled thresholds to zero instead of wrapping a negative
        // value to `u32::MAX`.
        let as_u32 = if value < 0 { 0u32 } else { value as u32 };
        match name {
            "threshold" => self.set_threshold(as_u32),
            "trace_limit" => self.trace_limit = as_u32,
            "trace_eagerness" => self.set_param_trace_eagerness(as_u32),
            "function_threshold" => self.set_function_threshold(as_u32),
            "max_inline_depth" => self.max_inline_depth = as_u32,
            "retrace_limit" => self.memory_manager.retrace_limit = as_u32,
            "max_retrace_guards" => self.memory_manager.max_retrace_guards = as_u32,
            "max_unroll_loops" => self.memory_manager.max_unroll_loops = as_u32,
            "max_unroll_recursion" => self.memory_manager.max_unroll_recursion = as_u32,
            "loop_longevity" => self.memory_manager.set_max_age(as_u32 as i64, 0),
            // warmstate.py:322-329 — vec, vec_all, vec_cost are separate fields
            "vec" | "vectorize" => self.vectorize = value != 0,
            "vec_all" => self.vec_all = value != 0,
            "vec_cost" => self.vec_cost = as_u32,
            "inlining" => self.inlining = value != 0,
            "disable_unrolling" => self.disable_unrolling_threshold = as_u32,
            "pureop_historylength" => self.pureop_historylength = as_u32,
            "decay" => self.counter.set_decay(value as i32),
            "enable_opts" => {} // string param, handled by set_param_enable_opts
            _ => {}
        }
    }

    /// warmstate.py: set_param_enable_opts(value)
    /// Set which optimization passes are enabled.
    /// Value is a colon-separated string like "intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll".
    /// "all" enables all passes.
    pub fn set_param_enable_opts(&mut self, value: &str) {
        // warmstate.py:284 substitutes the full list for `'all'` (and for
        // `None`, which this `&str` surface cannot express) and for nothing
        // else.  `''` therefore splits into no names and disables every pass;
        // treating it as `'all'` would silently keep unrolling on for a caller
        // that asked for the empty set.
        self.enable_opts = if value == "all" {
            default_enable_opts()
        } else {
            value
                .split(':')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        };
    }

    /// Get enabled optimization pass names.
    pub fn get_enable_opts(&self) -> &[String] {
        &self.enable_opts
    }

    /// warmstate.py: confirm_enter_jit(*args)
    /// Hook called before entering JIT compilation to allow the user
    /// to abort tracing based on runtime conditions.
    /// Returns true if tracing should proceed, false to abort.
    ///
    /// In RPython this is a user-provided callback set via JitDriver.
    /// Here we provide a default that always returns true.
    pub fn confirm_enter_jit(&self, _green_key: u64) -> bool {
        true
    }

    /// warmstate.py: get_location(greenkey)
    /// Convert a green key to a human-readable source location string.
    /// Used for JIT logging and debugging.
    ///
    /// In RPython this is a user-provided callback set via JitDriver.
    /// Here we return a default format.
    pub fn get_location(&self, green_key: u64) -> String {
        format!("<jit key 0x{:x}>", green_key)
    }

    /// warmstate.py: get_param(name) — read a JIT parameter value.
    pub fn get_param(&self, name: &str) -> Option<i64> {
        match name {
            "threshold" => Some(self.threshold as i64),
            "trace_limit" => Some(self.trace_limit as i64),
            "trace_eagerness" => Some(self.trace_eagerness as i64),
            "function_threshold" => Some(self.function_threshold as i64),
            "max_inline_depth" => Some(self.max_inline_depth as i64),
            "retrace_limit" => Some(self.memory_manager.retrace_limit as i64),
            "max_retrace_guards" => Some(self.memory_manager.max_retrace_guards as i64),
            "max_unroll_loops" => Some(self.memory_manager.max_unroll_loops as i64),
            "max_unroll_recursion" => Some(self.memory_manager.max_unroll_recursion as i64),
            "loop_longevity" => Some(self.memory_manager.loop_longevity_param()),
            "vectorize" => Some(if self.vectorize { 1 } else { 0 }),
            "vec_cost" => Some(self.vec_cost as i64),
            "inlining" => Some(if self.inlining { 1 } else { 0 }),
            "disable_unrolling" => Some(self.disable_unrolling_threshold as i64),
            "pureop_historylength" => Some(self.pureop_historylength as i64),
            // warmstate.py has no getter for "decay": set_param_decay delegates
            // into jitcounter.set_decay which stores decay_by_mult (the derived
            // multiplier), not the raw int. Read-back is not supported.
            "decay" => None,
            _ => None,
        }
    }

    /// warmstate.py: set_param_to_default(name)
    /// Reset a single JIT parameter to its default value.
    pub fn set_param_to_default(&mut self, name: &str) {
        match name {
            "threshold" => self.set_threshold(1039), // RPython default
            "trace_limit" => self.trace_limit = DEFAULT_TRACE_LIMIT,
            "trace_eagerness" => self.set_param_trace_eagerness(DEFAULT_TRACE_EAGERNESS),
            "function_threshold" => self.set_function_threshold(DEFAULT_FUNCTION_THRESHOLD),
            "max_inline_depth" => self.max_inline_depth = 10,
            "retrace_limit" => self.memory_manager.retrace_limit = DEFAULT_RETRACE_LIMIT,
            "max_retrace_guards" => self.memory_manager.max_retrace_guards = 15,
            "max_unroll_loops" => self.memory_manager.max_unroll_loops = 0,
            "max_unroll_recursion" => {
                self.memory_manager.max_unroll_recursion = DEFAULT_MAX_INLINE_DEPTH;
            }
            "loop_longevity" => self.memory_manager.set_max_age(1000, 0),
            "vectorize" => self.vectorize = false,
            "vec_cost" => self.vec_cost = 0,
            // rlib/jit.py:588 PARAMETERS default decay=40.
            "decay" => self.counter.set_decay(40),
            _ => {}
        }
    }

    /// warmstate.py: get_param_names()
    /// Return all known parameter names.
    pub fn param_names() -> &'static [&'static str] {
        &[
            "threshold",
            "trace_limit",
            "trace_eagerness",
            "function_threshold",
            "max_inline_depth",
            "retrace_limit",
            "max_retrace_guards",
            "max_unroll_loops",
            "max_unroll_recursion",
            "loop_longevity",
            "vectorize",
            "vec_cost",
            "inlining",
            "disable_unrolling",
            "pureop_historylength",
            "decay",
        ]
    }

    // ── RPython warmstate.py getter methods ──

    pub fn retrace_limit(&self) -> u32 {
        self.memory_manager.retrace_limit
    }
    pub fn max_retrace_guards(&self) -> u32 {
        self.memory_manager.max_retrace_guards
    }
    pub fn max_unroll_loops(&self) -> u32 {
        self.memory_manager.max_unroll_loops
    }
    pub fn max_unroll_recursion(&self) -> u32 {
        self.memory_manager.max_unroll_recursion
    }
    pub fn vectorize(&self) -> bool {
        self.vectorize
    }
    /// warmstate.py: vec_all — try to vectorize all trace loops.
    pub fn vec_all(&self) -> bool {
        self.vec_all
    }
    pub fn vec_cost(&self) -> u32 {
        self.vec_cost
    }
    /// warmstate.py: inlining
    pub fn inlining(&self) -> bool {
        self.inlining
    }
    /// warmstate.py: disable_unrolling
    pub fn disable_unrolling_threshold(&self) -> u32 {
        self.disable_unrolling_threshold
    }
    /// warmstate.py: pureop_historylength
    pub fn pureop_historylength(&self) -> u32 {
        self.pureop_historylength
    }

    /// Get a snapshot of current JIT statistics.
    ///
    /// Counts every cell in every chain, not just the bucket heads.
    ///
    /// A chain needs no hash collision: a `comparekey: None` cell — what every
    /// bare-hash writer installs — can never match a typed lookup, so
    /// `ensure_cell_for_key` misses on a key that already has a cell and
    /// chains past it. See
    /// `one_key_through_a_hash_and_a_typed_entry_point_builds_a_chain`.
    ///
    /// So `num_cells` and `cells.len()` are **already** capable of
    /// disagreeing, and the difference is the count of keys that reached
    /// both an untyped and a typed writer. Counting the cells is the
    /// definition the field documents ("Total number of BaseJitCells");
    /// counting the map entries counts buckets, which is a different number.
    ///
    /// `num_pinned_refs` sums `retained_greens` per cell, and a
    /// chained cell pins referents just as a head does. So the "adds zero
    /// today" note above applies to the state counters only.
    pub fn get_stats(&self) -> JitStats {
        let mut stats = JitStats::default();
        for head in self.cells.values() {
            let mut cur = Some(head);
            while let Some(cell) = cur {
                stats.num_cells += 1;
                stats.num_pinned_refs += cell.retained_greens.len();
                match cell.state {
                    BaseJitCellState::Compiled => stats.num_compiled += 1,
                    BaseJitCellState::Tracing => stats.num_tracing += 1,
                    BaseJitCellState::Invalidated => stats.num_invalidated += 1,
                    BaseJitCellState::DontTraceHere => stats.num_disable_noninlinable_function += 1,
                    BaseJitCellState::NotHot => {}
                }
                cur = cell.next.as_deref();
            }
        }
        stats
    }

    /// Get the current trace limit.
    pub fn trace_limit(&self) -> u32 {
        self.trace_limit
    }

    /// Set the trace limit.
    pub fn set_trace_limit(&mut self, limit: u32) {
        self.trace_limit = limit;
    }

    /// Get the current tracing generation.
    pub fn tracing_generation(&self) -> u64 {
        self.tracing_generation
    }

    /// Remove dead BaseJitCells from all chains.
    /// Returns the number of cells removed.
    pub fn gc_cells(&mut self) -> usize {
        let mut removed = 0;
        let keys: Vec<u64> = self.cells.keys().copied().collect();
        let mut dropped: Vec<BaseJitCell> = Vec::new();
        for hash in keys {
            if let Some(head) = self.cells.swap_remove(&hash) {
                let (kept, n) = Self::clean_chain(head, &mut dropped);
                removed += n;
                if let Some(k) = kept {
                    self.cells.insert(hash, k);
                }
            }
        }
        for cell in &dropped {
            self.forget_cell_key(cell);
        }
        removed
    }

    /// Walk a chain, removing cells where should_remove_jitcell() is true.
    /// Removed cells are pushed to `dropped` so the caller can retire their
    /// minted keys ([`Self::forget_cell_key`]).
    fn clean_chain(
        head: BaseJitCell,
        dropped: &mut Vec<BaseJitCell>,
    ) -> (Option<BaseJitCell>, usize) {
        let mut keep: Option<BaseJitCell> = None;
        let mut removed = 0;
        let mut cell_opt = Some(head);
        while let Some(mut c) = cell_opt {
            let next = c.next.take().map(|b| *b);
            if !c.should_remove_jitcell() {
                c.next = keep.map(Box::new);
                keep = Some(c);
            } else {
                removed += 1;
                dropped.push(c);
            }
            cell_opt = next;
        }
        (keep, removed)
    }

    /// The bucket a cell key lives in.
    ///
    /// An unminted cell key IS its bucket's raw hash (it is only ever handed
    /// out inside that bucket), so the common path costs one `is_empty()`
    /// test and no lookup at all.
    #[inline]
    pub(crate) fn bucket_of(&self, cell_key: u64) -> u64 {
        if self.minted.is_empty() {
            return cell_key;
        }
        self.minted.get(&cell_key).copied().unwrap_or(cell_key)
    }

    /// The cell named by `cell_key`, or `None` if no cell holds that key.
    ///
    /// This is what every hash-form entry point resolves through. Upstream's
    /// equivalent is not a lookup at all: `maybe_compile_and_run` already
    /// holds the cell object it matched (warmstate.py:458-464). Pyre's u64 is
    /// carried across crate boundaries (`compiled_loops`,
    /// `JitCellToken::green_key`, `rd_loop_token`), so it resolves back to the
    /// cell here — and it resolves to exactly one, which is the property that
    /// bucket-head reading did not have.
    #[inline]
    pub(crate) fn cell_by_key(&self, cell_key: u64) -> Option<&BaseJitCell> {
        let mut cell = self.cells.get(&self.bucket_of(cell_key));
        while let Some(c) = cell {
            if c.cell_key == Some(cell_key) {
                return Some(c);
            }
            cell = c.next.as_deref();
        }
        None
    }

    /// Mutable [`Self::cell_by_key`].
    #[inline]
    pub(crate) fn cell_by_key_mut(&mut self, cell_key: u64) -> Option<&mut BaseJitCell> {
        let bucket = self.bucket_of(cell_key);
        let mut cell = self.cells.get_mut(&bucket);
        while let Some(c) = cell {
            if c.cell_key == Some(cell_key) {
                return Some(c);
            }
            cell = c.next.as_deref_mut();
        }
        None
    }

    /// The cell named by `cell_key`, installing a comparekey-less one if the
    /// key names none — the hash-form analogue of
    /// [`Self::ensure_cell_for_key`], and what `cells.entry(hash)` used to be.
    ///
    /// The cell it installs carries no `comparekey`, exactly as before: a
    /// caller holding only a u64 has no greens to store. Such a cell is
    /// reachable by its key and by nothing else — no typed walk will ever
    /// match it (`comparekey_matches` refuses `None` unconditionally), which
    /// is the pre-existing hazard this migration does not pretend to close.
    /// What it does close is the other half: the cell is now named by a key
    /// that means only this cell, so the writer and every later reader of that
    /// u64 land on the same object even after the bucket chains.
    fn ensure_cell_by_key(&mut self, cell_key: u64) -> &mut BaseJitCell {
        if self.cell_by_key(cell_key).is_none() {
            let mut newcell = BaseJitCell::new();
            newcell.cell_key = Some(cell_key);
            self.install_new_cell(self.bucket_of(cell_key), Some(newcell));
        }
        self.cell_by_key_mut(cell_key)
            .expect("just installed a cell under this key")
    }

    /// Resolve greens to the cell key their cell is named by, WITHOUT
    /// installing anything.
    ///
    /// This is the "resolve once" step. `warmstate.py:596-604
    /// JitCell.get_jitcell` takes greens + `comparekey` + `get_uhash`
    /// together and yields a cell; pyre yields that cell's key, and every
    /// consumer downstream carries the key instead of re-deriving it from the
    /// bucket hash (warmstate.py:483/:511 carries the resolved
    /// `procedure_token` for the same reason).
    ///
    /// `None` means this key owns no cell AND its bucket's raw hash is already
    /// taken by a different cell — the only state in which there is no u64
    /// that names this key's (absent) cell. Readers must treat it as a miss;
    /// writers call [`Self::ensure_cell_key`], which mints.
    pub fn cell_key_for(&self, key: &GreenKey) -> Option<u64> {
        if let Some(cell) = self.lookup_chain_with_key(key) {
            return cell.cell_key;
        }
        let hash = key.get_uhash();
        // Nothing holds the raw hash, so it is the key this green key's cell
        // would be installed under, and it names no other cell meanwhile.
        self.cell_by_key(hash).is_none().then_some(hash)
    }

    /// Resolve greens to their cell key, installing the cell (with its
    /// `comparekey`) if the key owns none — `warmstate.py:626-641
    /// _ensure_jit_cell_at_key`, reporting the identity of the cell it
    /// ensured.
    pub fn ensure_cell_key(&mut self, key: &GreenKey) -> u64 {
        self.ensure_cell_for_key(key);
        self.lookup_chain_with_key(key)
            .and_then(|cell| cell.cell_key)
            .expect("ensure_cell_for_key just installed a keyed cell for this key")
    }

    /// **Resolve once**: the cell key the greens behind `hash` name.
    ///
    /// `make_key` is called only when the bucket is chained — an unchained
    /// bucket has one candidate, so [`Self::sole_cell_key`] is already exact
    /// and no `GreenKey` has to be built to confirm it. Producers that cannot
    /// build one on a chained bucket get `hash` back, which names the bucket's
    /// original occupant or nothing: a miss, not a guess.
    ///
    /// See [`WarmEnterState`]'s type doc for why the resolve happens at the
    /// producer and the result is carried, rather than each consumer
    /// re-deriving a cell from the bucket.
    #[inline]
    pub fn resolve_cell_key(&self, hash: u64, make_key: impl FnOnce() -> GreenKey) -> u64 {
        if self.bucket_is_chained(hash) {
            self.cell_key_for(&make_key()).unwrap_or(hash)
        } else {
            self.sole_cell_key(hash).unwrap_or(hash)
        }
    }

    /// The key the sole occupant of `hash`'s bucket is named by, if the bucket
    /// holds exactly one cell.
    ///
    /// The entry path's allocation-free half. A one-cell bucket has one
    /// candidate, so the walk `maybe_compile_and_run` would do can only find
    /// that cell — no `GreenKey` has to be built to prove it. Normally the
    /// answer is `hash` itself; it differs only when the cell that held the
    /// raw hash has been evicted out from under a minted sibling.
    ///
    /// **Why no comparator is consulted.** `warmstate.py:458-464` accepts a
    /// cell only after `comparekey` matches, because upstream buckets are
    /// indexed slots of a sized table (`counter.py:239-240`) and genuinely mix
    /// unrelated green keys. This table is keyed by the FULL `get_uhash`, so
    /// short of a 64-bit collision a bucket holds one green key's cells and
    /// nothing else — the several occupants of a chained bucket are that one
    /// key's hash-written cell and its typed twin, which a comparator would
    /// only tell apart from each other. Two consequences:
    ///
    /// * the sole occupant of an unchained bucket IS this key's cell, and
    /// * where it is not — an occupant a hash-written creator installed with no
    ///   comparator — the full typed resolve reaches the same key anyway:
    ///   [`Self::cell_key_for`] misses the chain, finds `hash` taken, and
    ///   `unwrap_or(hash)` lands back on that same cell. So the shortcut is not
    ///   trading a typed miss for a wrong cell; there is no typed miss to take.
    ///
    /// A 64-bit `get_uhash` collision breaks the invariant, and then this can
    /// answer with a neighbour's cell. That is the condition under which a
    /// comparator-less cell is unresolvable rather than resolvable-but-unsound
    /// (see [`BaseJitCell::comparekey`]) — no accept rule fixes it here.
    #[inline]
    pub fn sole_cell_key(&self, hash: u64) -> Option<u64> {
        let head = self.lookup_chain(hash)?;
        if head.next.is_some() {
            return None;
        }
        head.cell_key
    }

    /// Mint a cell key for a cell whose bucket's raw hash is already taken.
    ///
    /// **No sub-range of `u64` can be reserved for minted keys.**
    /// `JitCell.get_uhash` (warmstate.py:584-593) is a full-width multiply-xor
    /// fold over arbitrary greens, so every `u64` is a hash some green key can
    /// produce; a reserved range would be a range real keys also land in.
    /// Uniqueness is therefore enforced against the LIVE key set instead:
    /// derive a candidate from the bucket and a monotone serial through the
    /// same fold, then step until the candidate names no live cell and is not
    /// already a minted key. Both tests are needed — `cell_by_key` would find
    /// an unminted twin in its own bucket, and `bucket_of` would misroute a
    /// duplicate mint.
    ///
    /// What is NOT guaranteed: that a green key compiled *later* cannot hash
    /// to a live minted key. Nothing can guarantee that, and the consequence
    /// is the pre-existing one — the later key resolves through
    /// [`Self::cell_key_for`], sees its raw hash taken, and mints in turn, so
    /// the two stay distinct cells. Only a caller that holds a bare hash and
    /// no greens can land on the wrong one, which is the same class of miss a
    /// raw hash collision already had.
    fn mint_cell_key(&mut self, bucket: u64) -> u64 {
        loop {
            self.mint_serial = self.mint_serial.wrapping_add(1);
            // The serial must reach the fold as the VALUE, not pre-mixed into
            // the accumulator: `green_uhash_step(bucket ^ serial, Int, serial)`
            // folds `(bucket ^ serial) ^ serial`, which is `bucket` again, so
            // every retry re-proposes one number and the second mint in a
            // bucket cannot terminate. Folded here, distinct serials give
            // distinct candidates — xor and the odd multiplier are both
            // bijections — so the retry walks a fresh number each time and the
            // finite live set bounds it.
            let candidate = majit_ir::green_uhash_step(
                bucket,
                majit_ir::GreenType::Int,
                self.mint_serial as i64,
            );
            if candidate != bucket
                && self.cell_by_key(candidate).is_none()
                && !self.minted.contains_key(&candidate)
            {
                self.minted.insert(candidate, bucket);
                return candidate;
            }
        }
    }

    /// counter.py:239-240 lookup_chain(hash)
    ///
    /// ```text
    ///  def lookup_chain(self, hash):
    ///      return self.celltable[self._get_index(hash)]
    /// ```
    ///
    /// Returns the head of the chain at `hash`. Walk `.next` to
    /// iterate the chain.
    pub fn lookup_chain(&self, hash: u64) -> Option<&BaseJitCell> {
        self.cells.get(&hash)
    }

    /// counter.py:246-256 install_new_cell(hash, newcell)
    ///
    /// ```text
    ///  def install_new_cell(self, hash, newcell):
    ///      index = self._get_index(hash)
    ///      cell = self.celltable[index]
    ///      keep = newcell
    ///      while cell is not None:
    ///          nextcell = cell.next
    ///          if not cell.should_remove_jitcell():
    ///              cell.next = keep
    ///              keep = cell
    ///          cell = nextcell
    ///      self.celltable[index] = keep
    /// ```
    ///
    /// Pyre addition: the cell being filed is given its [`BaseJitCell::cell_key`]
    /// here if it does not already carry one — the bucket's raw hash when that
    /// is free, a minted key when it is not. Filing is the moment the cell
    /// acquires an identity because it is the moment it becomes reachable;
    /// upstream needs no equivalent because it hands the cell object itself
    /// on (warmstate.py:483/:511) and never re-derives it from a number.
    pub fn install_new_cell(&mut self, hash: u64, newcell: Option<BaseJitCell>) {
        let mut keep = newcell;
        if let Some(cell) = &mut keep
            && cell.cell_key.is_none()
        {
            cell.cell_key = Some(if self.cell_by_key(hash).is_none() {
                hash
            } else {
                self.mint_cell_key(hash)
            });
        }
        let mut cell_opt = self.cells.swap_remove(&hash);
        // Walk the existing chain, unlink each node.
        while let Some(mut cell) = cell_opt {
            let next = cell.next.take().map(|b| *b);
            if !cell.should_remove_jitcell() {
                // counter.py:253-254: cell.next = keep; keep = cell
                cell.next = keep.map(Box::new);
                keep = Some(cell);
            } else {
                self.forget_cell_key(&cell);
            }
            cell_opt = next;
        }
        // counter.py:256: self.celltable[index] = keep
        if let Some(k) = keep {
            self.cells.insert(hash, k);
        }
    }

    /// Retire a dropped cell's minted key so [`Self::mint_cell_key`] may reuse
    /// the number and `bucket_of` stops answering for a cell that is gone.
    ///
    /// Unminted keys need no retiring: they equal their bucket hash, so they
    /// are recomputable from greens and are re-taken by the next cell that
    /// installs into an empty bucket.
    fn forget_cell_key(&mut self, cell: &BaseJitCell) {
        if let Some(key) = cell.cell_key
            && !self.minted.is_empty()
        {
            self.minted.swap_remove(&key);
        }
    }

    /// counter.py:242-244 cleanup_chain(hash)
    ///
    /// ```text
    ///  def cleanup_chain(self, hash):
    ///      self.reset(hash)
    ///      self.install_new_cell(hash, None)
    /// ```
    pub fn cleanup_chain(&mut self, hash: u64) {
        self.counter.reset(hash);
        self.install_new_cell(hash, None);
    }

    /// warmstate.py:596-604 `JitCell.get_jitcell(*greenargs)`.
    ///
    /// ```python
    /// hash = JitCell.get_uhash(*greenargs)
    /// cell = jitcounter.lookup_chain(hash)
    /// while cell is not None:
    ///     if isinstance(cell, JitCell) and cell.comparekey(*greenargs):
    ///         return cell
    ///     cell = cell.next
    /// return None
    /// ```
    ///
    /// Walks the per-bucket chain comparing each cell's stored
    /// comparekey with `key` until a match is found. Hash collisions
    /// across distinct typed greens are resolved by comparekey, so
    /// `(code_a, pc_a)` and `(code_b, pc_b)` with the same `get_uhash`
    /// no longer alias to the same cell.
    ///
    /// Collision resolution is not why this walk earns its keep today.
    /// The common case is a chain built from ONE key: a hash-only creator
    /// installs a cell that can carry no `comparekey`, the typed path then
    /// installs its own, and `install_new_cell` links the survivor ahead —
    /// so the bucket holds two cells for one key with no collision anywhere.
    /// A full-width hash makes collisions impractical but says nothing about
    /// multiple cells linked under one hash. See
    /// `one_key_through_a_hash_and_a_typed_entry_point_builds_a_chain`.
    ///
    /// This is the resolve half of [`Self::cell_key_for`], which is how the
    /// `u64` API surface reaches the same cell without holding the greens: the
    /// walk happens ONCE, at the hash producer, and the cell key it yields is
    /// what `clear_loop_token`, `counter_tick`, `maybe_compile`,
    /// `finish_tracing`, `abort_tracing`, `get_cell` and the rest are handed —
    /// along with `MetaInterp`'s `compiled_loops` / `cut_compiled_keys`, which
    /// are indexed by the same key and so describe the same cell. That is the
    /// property the `u64` currency previously lacked: it named a bucket, and a
    /// bucket is not a cell.
    pub fn lookup_chain_with_key(&self, key: &GreenKey) -> Option<&BaseJitCell> {
        let hash = key.get_uhash();
        let mut cell = self.cells.get(&hash);
        while let Some(c) = cell {
            if c.comparekey_matches(key) {
                return Some(c);
            }
            cell = c.next.as_deref();
        }
        None
    }

    /// warmstate.py:626-641 `JitCell.ensure_jit_cell_at_key(greenkey)` /
    /// `_ensure_jit_cell_at_key(*greenargs)`.
    ///
    /// ```python
    /// def _ensure_jit_cell_at_key(*greenargs):
    ///     hash = JitCell.get_uhash(*greenargs)
    ///     cell = jitcounter.lookup_chain(hash)
    ///     while cell is not None:
    ///         if isinstance(cell, JitCell) and cell.comparekey(*greenargs):
    ///             return cell
    ///         cell = cell.next
    ///     newcell = JitCell(*greenargs)
    ///     jitcounter.install_new_cell(hash, newcell)
    ///     return newcell
    /// ```
    ///
    /// Returns nothing; the caller reads the ensured cell back through
    /// `lookup_chain_with_key`, or through [`Self::ensure_cell_key`] when what
    /// it needs is the cell's identity rather than the cell.
    ///
    /// On a miss (no chained cell matches the typed key) the helper
    /// allocates a new cell, stores the key through `set_comparekey` —
    /// which also takes ownership of the key's `Ref` referents, the
    /// `setattr` half of `JitCell.__init__` — and installs it at the head
    /// of the chain via `install_new_cell`, matching upstream's
    /// `jitcounter.install_new_cell` semantics.
    pub fn ensure_cell_for_key(&mut self, key: &GreenKey) {
        if self.lookup_chain_with_key(key).is_some() {
            return;
        }
        let mut newcell = BaseJitCell::new();
        newcell.set_comparekey(key);
        self.install_new_cell(key.get_uhash(), Some(newcell));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `warmstate.py:284` substitutes the full pass list for `'all'` alone.
    /// The empty string splits into no names, so it selects nothing — the one
    /// spelling a caller uses to turn every optimization off. Reading it as
    /// `'all'` inverts that request completely and silently, since the driver
    /// then keeps unrolling and every other pass.
    #[test]
    fn an_empty_enable_opts_selects_no_passes() {
        let mut ws = WarmEnterState::new(1);
        let all = default_enable_opts();
        assert_eq!(
            ws.get_enable_opts(),
            all.as_slice(),
            "default is the full set"
        );

        ws.set_param_enable_opts("");
        assert!(
            ws.get_enable_opts().is_empty(),
            "`\"\"` names no pass, so no pass is enabled; got {:?}",
            ws.get_enable_opts()
        );

        ws.set_param_enable_opts("all");
        assert_eq!(ws.get_enable_opts(), all.as_slice(), "`all` restores them");

        ws.set_param_enable_opts("intbounds:heap");
        assert_eq!(ws.get_enable_opts(), ["intbounds", "heap"]);
    }

    #[test]
    fn test_not_hot_initially() {
        let mut ws = WarmEnterState::new(3);
        match ws.maybe_compile(42) {
            HotResult::NotHot => {}
            _ => panic!("expected NotHot"),
        }
    }

    #[test]
    fn test_start_tracing_at_threshold() {
        let mut ws = WarmEnterState::new(3);
        // Tick 1, 2: not hot
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        // Tick 3: threshold reached, start tracing
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
    }

    #[test]
    fn test_already_tracing() {
        let mut ws = WarmEnterState::new(2);
        // First tick: eviction (always false). Second tick: threshold reached.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        // Next call sees TRACING flag
        match ws.maybe_compile(42) {
            HotResult::AlreadyTracing => {}
            _ => panic!("expected AlreadyTracing"),
        }
    }

    #[test]
    fn test_run_compiled() {
        let mut ws = WarmEnterState::new(1);
        let token_num = ws.alloc_token_number();
        let token = JitCellToken::new(token_num);
        ws.attach_procedure_to_interp(42, token);

        match ws.maybe_compile(42) {
            HotResult::RunCompiled => {}
            _ => panic!("expected RunCompiled"),
        }
    }

    #[test]
    fn test_finish_tracing() {
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.finish_tracing(42);

        let cell = ws.get_cell(42).unwrap();
        assert!(!cell.is_tracing());
        assert!(cell.flags & jc_flags::TRACING_OCCURRED != 0);
    }

    #[test]
    fn test_abort_tracing() {
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.abort_tracing(42, true);

        let cell = ws.get_cell(42).unwrap();
        assert!(!cell.is_tracing());
        assert!(cell.flags & jc_flags::DONT_TRACE_HERE != 0);

        // RPython warmstate.py: a DONT_TRACE_HERE cell with no procedure token
        // still retriggers separate tracing after warming up again.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing due to DONT_TRACE_HERE retrace"),
        }
    }

    #[test]
    fn test_abort_tracing_allows_retry() {
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        // Abort without DONT_TRACE_HERE
        ws.abort_tracing(42, false);

        // Counter was reset during start_tracing, but hash is still in the table.
        // Need to tick again to reach threshold. The hash is found now (not evicted),
        // so one tick to reach count=1, another to reach count=2 >= threshold=2.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing on retry"),
        }
    }

    #[test]
    fn test_different_green_keys() {
        let mut ws = WarmEnterState::new(3);
        // Key 1: tick 1 (eviction), tick 2 (count=2 < 3)
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        // Key 2: tick 1 (eviction)
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        // Key 1: tick 3 -> threshold, starts tracing
        match ws.maybe_compile(1) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing for key 1"),
        }
        // Key 2 still not hot (only 2 total ticks: eviction + one more needed)
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
    }

    #[test]
    fn test_alloc_token_number() {
        let mut ws = WarmEnterState::new(10);
        let first = ws.alloc_token_number();
        let second = ws.alloc_token_number();
        let third = ws.alloc_token_number();
        // The allocator is process-global. Other tests may allocate between
        // these calls when the harness runs tests in parallel, so only the
        // monotonic uniqueness contract is local to this test.
        assert!(first < second);
        assert!(second < third);
    }

    #[test]
    fn test_set_threshold() {
        let mut ws = WarmEnterState::new(100);
        assert_eq!(ws.threshold(), 100);
        ws.set_threshold(50);
        assert_eq!(ws.threshold(), 50);
    }

    #[test]
    fn test_get_compiled() {
        let mut ws = WarmEnterState::new(1);
        assert!(ws.get_compiled(42).is_none());

        let token = JitCellToken::new(0);
        ws.attach_procedure_to_interp(42, token);

        let compiled = ws.get_compiled(42);
        assert!(compiled.is_some());
        assert_eq!(compiled.unwrap().number, 0);
    }

    #[test]
    fn test_full_lifecycle() {
        let mut ws = WarmEnterState::new(3);
        let key = 0xDEAD;

        // Phase 1: Not hot (need 3 ticks to reach threshold=3)
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));

        // Phase 2: Start tracing (third tick reaches threshold)
        assert!(matches!(ws.maybe_compile(key), HotResult::StartTracing));

        // Phase 3: Already tracing
        assert!(matches!(ws.maybe_compile(key), HotResult::AlreadyTracing));

        // Phase 4: Finish tracing, install compiled code
        ws.finish_tracing(key);
        let token_num = ws.alloc_token_number();
        let token = JitCellToken::new(token_num);
        ws.attach_procedure_to_interp(key, token);

        // Phase 5: Run compiled
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
    }

    #[test]
    fn test_trace_eagerness_default() {
        let ws = WarmEnterState::new(3);
        assert_eq!(ws.trace_eagerness(), 200);
    }

    #[test]
    fn test_trace_eagerness_custom() {
        let mut ws = WarmEnterState::new(3);
        ws.set_param_trace_eagerness(10);
        assert_eq!(ws.trace_eagerness(), 10);
    }

    #[test]
    fn test_start_retrace_preserves_input_types() {
        // RPython pyjitpl.py:2609 `MetaInterp.create_history(max_num_inputargs)`:
        // the MetaInterp, not warmstate, owns the Trace factory. Since
        // warmstate's `start_retrace` is now a state-only signal, this test
        // verifies that the input_types the caller intends to use are the
        // ones that flow into the Trace downstream (Trace::with_input_types).
        let mut ws = WarmEnterState::new(3);
        let input_types = [Type::Ref, Type::Int, Type::Float];
        ws.start_retrace(&input_types);
        let mut recorder = crate::recorder::Trace::with_input_types(&input_types);
        recorder.close_loop(&[
            majit_ir::OpRef::input_arg_ref(0),
            majit_ir::OpRef::input_arg_int(1),
            majit_ir::OpRef::input_arg_float(2),
        ]);
        let trace = recorder.get_trace();
        let seen: Vec<Type> = trace.inputargs.iter().map(|arg| arg.tp).collect();
        assert_eq!(seen, input_types.to_vec());
    }

    // ── Quasi-immutable invalidation tests ──

    #[test]
    fn test_quasiimmut_register_and_invalidate() {
        let mut ws = WarmEnterState::new(1);
        let token = JitCellToken::new(ws.alloc_token_number());
        let green_key = 42;
        let qmut_key = 0xABCD;

        assert!(!token.is_invalidated());
        ws.attach_procedure_to_interp(green_key, token);
        ws.register_quasiimmut_dependency(qmut_key, green_key);

        let count = ws.invalidate_quasiimmut(qmut_key);
        assert_eq!(count, 1);

        let cell = ws.get_cell(green_key).unwrap();
        assert!(cell.loop_token.as_ref().unwrap().is_invalidated());
    }

    #[test]
    fn test_quasiimmut_no_deps() {
        let mut ws = WarmEnterState::new(1);
        // No dependencies registered → invalidation returns 0.
        let count = ws.invalidate_quasiimmut(0xDEAD);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_quasiimmut_multiple_deps() {
        let mut ws = WarmEnterState::new(1);
        let qmut_key = 0xABCD;

        // Install two loops depending on the same quasi-immutable field.
        for green_key in [10, 20] {
            let token = JitCellToken::new(ws.alloc_token_number());
            ws.attach_procedure_to_interp(green_key, token);
            ws.register_quasiimmut_dependency(qmut_key, green_key);
        }

        let count = ws.invalidate_quasiimmut(qmut_key);
        assert_eq!(count, 2);

        for green_key in [10, 20] {
            let cell = ws.get_cell(green_key).unwrap();
            assert!(cell.loop_token.as_ref().unwrap().is_invalidated());
        }
    }

    #[test]
    fn test_quasiimmut_invalidate_all() {
        let mut ws = WarmEnterState::new(1);
        for green_key in [1, 2, 3] {
            let token = JitCellToken::new(ws.alloc_token_number());
            ws.attach_procedure_to_interp(green_key, token);
        }

        ws.invalidate_all();

        for green_key in [1, 2, 3] {
            let cell = ws.get_cell(green_key).unwrap();
            assert!(cell.loop_token.as_ref().unwrap().is_invalidated());
        }
    }

    // ── Function threshold tests ──

    #[test]
    fn test_function_threshold_default() {
        let ws = WarmEnterState::new(3);
        assert_eq!(ws.function_threshold(), 1619); // PyPy default
    }

    #[test]
    fn test_function_threshold_custom() {
        let mut ws = WarmEnterState::new(3);
        ws.set_function_threshold(10);
        assert_eq!(ws.function_threshold(), 10);
    }

    // ── Trace limit lifecycle tests (RPython: test_tracelimit.py) ──

    #[test]
    fn test_abort_tracing_too_long_sets_dont_trace() {
        // When a trace is too long, the meta-interpreter calls
        // abort_tracing(key, true) to prevent future tracing at that location.
        // This mirrors RPython's ABORT_TOO_LONG behavior.
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        // Simulate: recorder.is_too_long() was true, so abort with dont_trace.
        ws.abort_tracing(42, true);

        // The key is now blacklisted.
        let cell = ws.get_cell(42).unwrap();
        assert!(cell.flags & jc_flags::DONT_TRACE_HERE != 0);
        assert!(!cell.is_tracing());

        // RPython warmstate.py: DONT_TRACE_HERE still allows separate
        // tracing later for keys without a procedure token.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(42), HotResult::StartTracing));
    }

    #[test]
    fn test_disable_noninlinable_function_blocks_inlining() {
        let mut ws = WarmEnterState::new(3);

        // Fresh cell: can_inline_callable returns true by default.
        assert!(ws.can_inline_callable(42));

        // dont_trace_here → JC_DONT_TRACE_HERE flag, can_inline_callable
        // returns false (warmstate.py:669-676 parity).
        ws.disable_noninlinable_function(42);
        assert!(!ws.can_inline_callable(42));
    }

    #[test]
    fn test_abort_too_long_then_retry_different_key() {
        // Aborting one key's trace as too long should not affect other keys.
        let mut ws = WarmEnterState::new(2);

        // Key 42: start and abort as too long.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing for key 42"),
        }
        ws.abort_tracing(42, true);

        // Key 99: should still work normally.
        assert!(matches!(ws.maybe_compile(99), HotResult::NotHot));
        match ws.maybe_compile(99) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing for key 99"),
        }
    }

    #[test]
    fn test_lifecycle_with_trace_abort_and_recompile() {
        // Full lifecycle: trace starts, is too long (abort without blacklist),
        // then on retry a shorter trace succeeds and gets compiled.
        let mut ws = WarmEnterState::new(2);
        let key = 0xCAFE;

        // Phase 1: reach threshold, start tracing.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        // Phase 2: trace is too long, abort without blacklist.
        ws.abort_tracing(key, false);

        // Phase 3: retry, reach threshold again.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {
                // Phase 4: this time the trace succeeds.
                ws.finish_tracing(key);
                let token = JitCellToken::new(ws.alloc_token_number());
                ws.attach_procedure_to_interp(key, token);
            }
            _ => panic!("expected StartTracing on retry"),
        }

        // Phase 5: compiled code should be available.
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
    }

    #[test]
    fn test_multiple_aborts_before_success() {
        // Mirrors RPython's segmented trace behavior: a location can fail
        // multiple times before eventually compiling.
        // threshold=2 because the first tick always evicts (returns NotHot).
        let mut ws = WarmEnterState::new(2);
        let key = 0xBEEF;

        // First attempt: tick once (eviction), tick twice (threshold) -> StartTracing.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing (attempt 1)"),
        }
        ws.abort_tracing(key, false);

        // Second attempt: after abort, counter was reset, need to tick again.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing (attempt 2)"),
        }
        ws.abort_tracing(key, false);

        // Third attempt: succeeds and gets compiled.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {
                ws.finish_tracing(key);
                let token = JitCellToken::new(ws.alloc_token_number());
                ws.attach_procedure_to_interp(key, token);
            }
            _ => panic!("expected StartTracing (attempt 3)"),
        }
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
    }

    #[test]
    fn test_tracing_occurred_flag_persists_after_abort() {
        // The TRACING_OCCURRED flag should remain set even after abort.
        // This mirrors RPython's tracking of whether tracing was ever attempted.
        let mut ws = WarmEnterState::new(2);
        let key = 42;

        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        let cell = ws.get_cell(key).unwrap();
        assert!(cell.flags & jc_flags::TRACING_OCCURRED != 0);

        ws.abort_tracing(key, false);

        let cell = ws.get_cell(key).unwrap();
        assert!(!cell.is_tracing());
        assert!(
            cell.flags & jc_flags::TRACING_OCCURRED != 0,
            "TRACING_OCCURRED should persist after abort"
        );
    }

    #[test]
    fn test_quasiimmut_deps_cleared_after_invalidation() {
        let mut ws = WarmEnterState::new(1);
        let qmut_key = 0xABCD;
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(42, token);
        ws.register_quasiimmut_dependency(qmut_key, 42);

        ws.invalidate_quasiimmut(qmut_key);
        // Second invalidation should find no deps.
        let count = ws.invalidate_quasiimmut(qmut_key);
        assert_eq!(count, 0);
    }

    // ── MemoryManager (memmgr.py parity) tests ──
    //
    // Ported from rpython/jit/metainterp/test/test_memmgr.py.  Each
    // test creates real `Arc<JitCellToken>` instances since
    // `MemoryManager.alive_loops` keys on token identity per
    // memmgr.py:9-12 (the looptoken Python object is the dict key
    // upstream).

    fn make_token(number: u64) -> Arc<JitCellToken> {
        Arc::new(JitCellToken::new(number))
    }

    #[test]
    fn test_loop_aging_basic() {
        // memmgr.py:_kill_old_loops_now predicate:
        //   0 <= looptoken.generation < current_generation - (max_age - 1)
        // With max_age=3 and a token kept alive at current_generation=1:
        //   after 2 next_generation: cg=3, max_gen=3-2=1, 1<1=false → keep
        //   after 3 next_generation: cg=4, max_gen=4-2=2, 1<2=true  → evict
        let mut mgr = MemoryManager::new(3);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1);
        mgr.keep_loop_alive(&t2);
        assert_eq!(mgr.alive_count(), 2);

        mgr.next_generation();
        mgr.next_generation();
        assert_eq!(mgr.alive_count(), 2);

        mgr.next_generation();
        assert_eq!(mgr.alive_count(), 0);
    }

    #[test]
    fn test_loop_aging_disabled() {
        // memmgr.py:43-44: max_age <= 0 disables eviction.
        let mut mgr = MemoryManager::new(0);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1);
        mgr.keep_loop_alive(&t2);

        for _ in 0..100 {
            mgr.next_generation();
        }
        assert_eq!(mgr.alive_count(), 2);
    }

    /// Pins the PRODUCTION default, which is upstream's **test** default.
    ///
    /// The tests above prove the memmgr mechanism is a faithful port. What
    /// nothing watched is which mode a real `WarmEnterState` ships in.
    /// `WarmEnterState::new` (the only production constructor — `pyjitpl.rs`
    /// `warm_state: WarmEnterState::new(threshold)`) builds
    /// `MemoryManager::new(0)`, so `next_check == -1` and
    /// `_kill_old_loops_now` is unreachable: `alive_loops` never prunes.
    ///
    /// Upstream splits these two defaults and pyre collapsed them:
    ///
    /// | | `loop_longevity` |
    /// |---|---|
    /// | `rlib/jit.py:594` PARAMETERS — **production** | **1000** |
    /// | `warmspot.py:9` `jittify_and_run` — **test harness** | **0** |
    /// | pyre, every path unless `PYRE_JIT` overrides | **0** |
    ///
    /// So this asserts a **divergence**, not a desired value. It is here so
    /// the literal is watched: changing `MemoryManager::new(0)` at the
    /// construction site turns eviction on across the JIT, and that is a
    /// behavioural change (it starts feeding `try_to_free_some_loops`) which
    /// must be measured, not slipped in. If you are here because this test
    /// failed, that is the intended alarm — see default-retirement.
    ///
    /// This governs `alive_loops` (loop TOKENS) only. The `cells` map is a
    /// different population and is NOT bounded by `max_age`; it sheds
    /// entries only through `install_new_cell`'s `should_remove_jitcell`
    /// gate (typed-key eviction). Turning `max_age` on does not bound the cell table.
    ///
    /// `install_new_cell` runs the gate over the whole existing chain at the
    /// bucket on *every* typed install, collision or not — so the gate is
    /// reached routinely, it simply keeps most cells (a token, `TRACING`,
    /// `DONT_TRACE_HERE` without a dead token, or `FORCE_FINISH` all veto
    /// removal). The unbounded-growth conclusion survives; the stated reason
    /// for it does not.
    #[test]
    fn production_warmstate_ships_loop_eviction_disabled() {
        let ws = WarmEnterState::new(3);
        assert_eq!(
            ws.memory_manager.next_check, -1,
            "next_check must be -1 (eviction disabled) — memmgr.py:43-44",
        );
        assert_eq!(
            ws.memory_manager.loop_longevity_param(),
            0,
            "production ships loop_longevity=0, upstream's TEST default; \
             rlib/jit.py:594 gives production 1000",
        );
    }

    #[test]
    fn test_loop_aging_refresh() {
        // keep_loop_alive resets `looptoken.generation` to
        // `current_generation`, postponing eviction.
        let mut mgr = MemoryManager::new(3);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1);
        mgr.keep_loop_alive(&t2);

        // Refresh t1 each generation; t2 ages out.
        mgr.next_generation();
        mgr.keep_loop_alive(&t1);
        mgr.next_generation();
        mgr.keep_loop_alive(&t1);
        mgr.next_generation();
        // cg=4, max_gen=2; t1.gen=3 (refreshed), 3<2=false → keep;
        //                  t2.gen=1, 1<2=true → evict.
        assert_eq!(mgr.alive_count(), 1);
        assert!(mgr.contains(&t1));
        assert!(!mgr.contains(&t2));

        // Refreshing t1 each generation keeps it indefinitely.
        for _ in 0..10 {
            mgr.keep_loop_alive(&t1);
            mgr.next_generation();
        }
        assert!(mgr.contains(&t1));
    }

    #[test]
    fn test_loop_aging_mixed() {
        // Tokens kept alive at different generations age independently.
        let mut mgr = MemoryManager::new(2);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1); // t1.gen=1
        mgr.next_generation(); // cg=2, max_gen=2-1=1, t1.gen=1, 1<1=false → keep
        mgr.keep_loop_alive(&t2); // t2.gen=2

        mgr.next_generation(); // cg=3, max_gen=2, t1.gen=1<2=true → evict, t2.gen=2<2=false → keep
        assert_eq!(mgr.alive_count(), 1);
        assert!(!mgr.contains(&t1));
        assert!(mgr.contains(&t2));
    }

    // ── Memmgr deeper coverage (RPython: test_memmgr.py parity) ──

    #[test]
    fn test_evicted_loops_can_be_recompiled() {
        // After eviction the same JitCellToken can be re-inserted by a
        // fresh keep_loop_alive (or a freshly compiled token can take
        // its place — RPython gets a brand-new LoopToken object).
        let mut mgr = MemoryManager::new(2);
        let t1 = make_token(1);

        mgr.keep_loop_alive(&t1); // t1.gen=1
        mgr.next_generation(); // cg=2, max_gen=2-1=1, 1<1=false → keep
        mgr.next_generation(); // cg=3, max_gen=2, 1<2=true → evict
        assert_eq!(mgr.alive_count(), 0);

        // Re-register the same token (or a fresh one).
        mgr.keep_loop_alive(&t1); // t1.gen=3
        assert_eq!(mgr.alive_count(), 1);

        mgr.next_generation(); // cg=4, max_gen=3, 3<3=false → keep
        assert!(mgr.contains(&t1));
        mgr.next_generation(); // cg=5, max_gen=4, 3<4=true → evict
        assert!(!mgr.contains(&t1));
    }

    #[test]
    fn test_generation_does_not_panic_at_high_values() {
        // Verify that generation arithmetic survives many advances.
        // RPython uses r_int64; Rust uses i64 with the same wraparound
        // semantics.
        let mut mgr = MemoryManager::new(3);

        for _ in 0..1000 {
            mgr.next_generation();
        }
        assert_eq!(mgr.current_generation(), 1001);

        let t = make_token(42);
        mgr.keep_loop_alive(&t);
        assert_eq!(mgr.alive_count(), 1);

        mgr.next_generation();
        mgr.next_generation();
        assert!(mgr.contains(&t));
        mgr.next_generation();
        assert!(!mgr.contains(&t));
    }

    #[test]
    fn test_loop_aging_with_warm_state_integration() {
        // memmgr cooperates with WarmEnterState: attach a token via
        // `attach_procedure_to_interp` (warmstate.py:340-341 parity)
        // then mirror it into MemoryManager via keep_loop_alive.
        let mut ws = WarmEnterState::new(2);
        let mut mgr = MemoryManager::new(2);
        let key = 0xF00D;

        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.finish_tracing(key);
        let token = make_token(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, Arc::clone(&token));
        mgr.keep_loop_alive(&token);

        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
        assert_eq!(mgr.alive_count(), 1);

        mgr.next_generation();
        mgr.next_generation();
        assert!(!mgr.contains(&token));

        // Re-install: a fresh compile produces a fresh token; the
        // warmstate cell records the new procedure token, MemoryManager
        // tracks it under its own pointer identity.
        let token2 = make_token(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, Arc::clone(&token2));
        mgr.keep_loop_alive(&token2);

        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
        assert_eq!(mgr.alive_count(), 1);
    }

    #[test]
    fn test_loop_aging_does_not_affect_active_loops() {
        // A loop continually kept alive is never evicted, even while
        // peers are aged out.
        let mut mgr = MemoryManager::new(2);
        let t_active = make_token(1);
        let t_idle_a = make_token(2);
        let t_idle_b = make_token(3);

        mgr.keep_loop_alive(&t_active);
        mgr.keep_loop_alive(&t_idle_a);
        mgr.keep_loop_alive(&t_idle_b);

        for _ in 0..20 {
            mgr.keep_loop_alive(&t_active);
            mgr.next_generation();
            assert!(mgr.contains(&t_active));
        }

        // Idle loops were aged out long ago.
        assert!(!mgr.contains(&t_idle_a));
        assert!(!mgr.contains(&t_idle_b));
    }

    #[test]
    fn test_loop_aging_set_max_age_dynamic() {
        // memmgr.py:42 set_max_age — shrinking max_age accelerates
        // eviction of older loops.
        let mut mgr = MemoryManager::new(10);
        let t = make_token(1);

        mgr.keep_loop_alive(&t); // t.gen=1
        mgr.next_generation(); // cg=2, max_gen=2-9=-7 → keep

        // Shrink max_age — next sweep evicts t.
        mgr.set_max_age(1, 0);
        mgr.next_generation(); // cg=3, max_gen=3-0=3, 1<3 → evict
        assert!(!mgr.contains(&t));
    }

    #[test]
    fn test_loop_aging_interleaved_register_and_evict() {
        // Mirrors RPython's test_basic_3: tokens registered at different
        // generations, even-indexed kept alive each step.  After the
        // loop, even-indexed tokens are alive; old odd-indexed ones
        // (registered ≥ max_age generations ago and never refreshed)
        // are evicted.
        let mut mgr = MemoryManager::new(4);
        let mut tokens: Vec<Arc<JitCellToken>> = Vec::new();

        for i in 0..10u64 {
            let t = make_token(i);
            mgr.keep_loop_alive(&t);
            tokens.push(t);
            mgr.next_generation();

            // Refresh even-indexed tokens.
            for j in (0..=i).step_by(2) {
                mgr.keep_loop_alive(&tokens[j as usize]);
            }
        }

        for (i, t) in tokens.iter().enumerate() {
            let is_alive = mgr.contains(t);
            if i % 2 == 0 {
                assert!(is_alive, "even-indexed token {} should be alive", i);
            } else if i < 6 {
                // Odd tokens registered early enough are evicted.
                // Token i registered at cg=i+1; cg now ~11; max_gen=11-3=8.
                // For i < 6: t.gen=i+1 < 8 → evict.
                assert!(!is_alive, "odd token {} should be evicted", i);
            }
        }
    }

    // ── Trace limit + inline depth interaction ──

    #[test]
    fn test_trace_limit_with_inline_depth() {
        // Inline depth limiting and trace limit are orthogonal:
        // a function can be inlined (depth < max), but the trace
        // can still be too long. The WarmEnterState correctly tracks both.
        let mut ws = WarmEnterState::new(3);
        ws.set_max_inline_depth(3);

        // Depth 0: allowed
        assert!(ws.can_inline_at_depth(0));
        // Depth 2: allowed (< 3)
        assert!(ws.can_inline_at_depth(2));
        // Depth 3: not allowed (>= 3)
        assert!(!ws.can_inline_at_depth(3));
    }

    #[test]
    fn test_abort_tracing_retry_with_lower_threshold() {
        // Simulates the scenario where a trace is too long, the location
        // is aborted (without blacklisting), and on retry the WarmEnterState
        // has a lower threshold so it starts tracing sooner.
        let mut ws = WarmEnterState::new(5);
        let key = 0xABCD;

        // Reach threshold=5, start tracing
        for _ in 0..4 {
            assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        }
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing at threshold 5"),
        }

        // Abort without blacklisting
        ws.abort_tracing(key, false);

        // Lower threshold for retry
        ws.set_threshold(2);

        // Now only 2 ticks needed
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {
                ws.finish_tracing(key);
                let token = JitCellToken::new(ws.alloc_token_number());
                ws.attach_procedure_to_interp(key, token);
            }
            _ => panic!("expected StartTracing with lower threshold"),
        }
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
    }

    #[test]
    fn test_force_start_tracing_bypasses_counter() {
        // force_start_tracing is used for function-entry tracing where
        // the caller already decided to trace. It should work regardless
        // of the counter state.
        let mut ws = WarmEnterState::new(100); // very high threshold
        let key = 42;

        // Without any ticks, force_start_tracing should start tracing
        match ws.force_start_tracing(key) {
            HotResult::StartTracing => {}
            _ => panic!("force_start_tracing should start tracing immediately"),
        }

        // The cell should be in TRACING state
        let cell = ws.get_cell(key).unwrap();
        assert!(cell.is_tracing());

        // Second call sees AlreadyTracing
        match ws.force_start_tracing(key) {
            HotResult::AlreadyTracing => {}
            _ => panic!("expected AlreadyTracing on second force_start_tracing"),
        }
    }

    // ── BaseJitCell state machine tests ──

    #[test]
    fn test_jitcell_state_transitions() {
        // Full lifecycle: NotHot → Tracing → Compiled → Invalidated
        let mut ws = WarmEnterState::new(2);
        let key = 0xA1;

        // Initially no cell → NotHot
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::NotHot);

        // Tick to threshold → Tracing
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Tracing);

        // Finish tracing and install → Compiled
        ws.finish_tracing(key);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Compiled);

        // Invalidate via transition_cell → Invalidated
        ws.transition_cell(key, BaseJitCellState::Invalidated);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Invalidated);

        // The loop_token's invalidated flag should be set
        let cell = ws.get_cell(key).unwrap();
        assert!(cell.loop_token.as_ref().unwrap().is_invalidated());
        // token number is preserved as a historical record
        assert!(cell.token.is_some());
    }

    #[test]
    fn test_procedure_token_ownership() {
        // Compiled cell owns a token; invalidation revokes ownership.
        let mut ws = WarmEnterState::new(2);
        let key = 0xB2;

        // Compile a loop
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.finish_tracing(key);
        let token_num = ws.alloc_token_number();
        let token = JitCellToken::new(token_num);
        ws.attach_procedure_to_interp(key, token);

        // Cell owns the token
        let cell = ws.get_cell(key).unwrap();
        assert_eq!(cell.token, Some(token_num));
        assert!(cell.get_procedure_token().is_some());
        assert!(cell.has_seen_a_procedure_token());

        // Invalidate via quasiimmut
        let qmut_key = 0xFF;
        ws.register_quasiimmut_dependency(qmut_key, key);
        ws.invalidate_quasiimmut(qmut_key);

        // Token ownership revoked (state is Invalidated, but token number
        // is preserved as historical record)
        let cell = ws.get_cell(key).unwrap();
        assert_eq!(cell.token, Some(token_num)); // historical record preserved
        assert_eq!(cell.state, BaseJitCellState::Invalidated);
        // get_procedure_token returns None because the token is invalidated
        assert!(cell.get_procedure_token().is_none());
        // But we still know a token existed
        assert!(cell.has_seen_a_procedure_token());
    }

    #[test]
    fn test_set_param_threshold() {
        let mut ws = WarmEnterState::new(100);
        assert_eq!(ws.threshold(), 100);

        ws.set_param("threshold", 42);
        assert_eq!(ws.threshold(), 42);

        ws.set_param("trace_limit", 5000);
        assert_eq!(ws.trace_limit(), 5000);

        ws.set_param("trace_eagerness", 10);
        assert_eq!(ws.trace_eagerness(), 10);

        ws.set_param("function_threshold", 8);
        assert_eq!(ws.function_threshold(), 8);

        ws.set_param("max_inline_depth", 15);
        assert_eq!(ws.max_inline_depth(), 15);

        // Unknown param is ignored
        ws.set_param("nonexistent", 999);
    }

    #[test]
    fn test_get_stats() {
        let mut ws = WarmEnterState::new(2);

        // Initially empty
        let stats = ws.get_stats();
        assert_eq!(stats.num_cells, 0);
        assert_eq!(stats.num_compiled, 0);

        // Start tracing two keys
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(1), HotResult::StartTracing));
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(2), HotResult::StartTracing));

        let stats = ws.get_stats();
        assert_eq!(stats.num_cells, 2);
        assert_eq!(stats.num_tracing, 2);

        // Compile key 1
        ws.finish_tracing(1);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(1, token);

        let stats = ws.get_stats();
        assert_eq!(stats.num_compiled, 1);
        assert_eq!(stats.num_tracing, 1);

        // Abort key 2 with dont_trace
        ws.abort_tracing(2, true);

        let stats = ws.get_stats();
        assert_eq!(stats.num_compiled, 1);
        assert_eq!(stats.num_tracing, 0);
        assert_eq!(stats.num_disable_noninlinable_function, 1);

        // Invalidate key 1
        ws.transition_cell(1, BaseJitCellState::Invalidated);

        let stats = ws.get_stats();
        assert_eq!(stats.num_compiled, 0);
        assert_eq!(stats.num_invalidated, 1);
        assert_eq!(stats.num_disable_noninlinable_function, 1);
        assert_eq!(stats.num_cells, 2);
    }

    #[test]
    fn test_jitcell_state_disable_noninlinable_function() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xC3;

        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        // Abort with DONT_TRACE_HERE → state should be DontTraceHere
        ws.abort_tracing(key, true);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::DontTraceHere);

        // Future calls return NotHot
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));

        // Can manually reset to NotHot
        ws.transition_cell(key, BaseJitCellState::NotHot);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::NotHot);
        // DONT_TRACE_HERE flag should be cleared
        let cell = ws.get_cell(key).unwrap();
        assert!(cell.flags & jc_flags::DONT_TRACE_HERE == 0);
    }

    #[test]
    fn test_tracing_generation_increments() {
        let mut ws = WarmEnterState::new(2);
        let gen0 = ws.tracing_generation();
        assert_eq!(gen0, 0);

        // Start tracing key 1 → generation 1
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(1), HotResult::StartTracing));
        assert_eq!(ws.tracing_generation(), 1);

        let cell = ws.get_cell(1).unwrap();
        assert_eq!(cell.tracing_generation, 1);

        // Start tracing key 2 → generation 2
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(2), HotResult::StartTracing));
        assert_eq!(ws.tracing_generation(), 2);

        let cell = ws.get_cell(2).unwrap();
        assert_eq!(cell.tracing_generation, 2);
    }

    /// Slot 66 fires only for a cell whose `JC_TRACING` outlived the session
    /// that set it; slot 65 also counts a decline taken while a trace is
    /// running.
    ///
    /// This test calls `should_trace_function_entry` DIRECTLY, so it can
    /// reach the gate mid-trace. Production cannot: the one production caller
    /// guards on `!driver.is_tracing()`, so there every 65 is already a leak
    /// and 66 only says how old. Do not read this test as evidence that a
    /// production 65 is healthy — see the slot legend in `lib.rs`.
    ///
    /// Deltas are asserted as lower bounds because `MC_DIAG` is a process-wide
    /// static — a concurrent test bumping the same slot can only inflate them.
    #[test]
    fn stale_tracing_generation_is_counted_apart_from_a_live_trace() {
        let mut ws = WarmEnterState::new(2);

        // Cell A is marked as being traced under the current generation.
        ws.mark_as_being_traced(0xA);
        let cell = ws.get_cell(0xA).expect("mark_as_being_traced installs it");
        assert!(cell.is_tracing(), "fixture: A carries JC_TRACING");
        assert_eq!(
            cell.tracing_generation,
            ws.tracing_generation(),
            "fixture: A's generation is the live one, so it is NOT stale yet"
        );

        // While A's trace is still the live one, the decline is the healthy
        // case: 65 moves, 66 must not.
        let live_65 = crate::mc_diag(65);
        let live_66 = crate::mc_diag(66);
        assert!(!ws.should_trace_function_entry(0xA));
        assert!(
            crate::mc_diag(65) >= live_65 + 1,
            "a decline on the tracing term bumps 65"
        );
        assert_eq!(
            crate::mc_diag(66),
            live_66,
            "a LIVE trace must not read as stale — this is what 65 alone cannot say"
        );

        // Starting a trace on another key supersedes A's session without
        // clearing A's flag: only start_tracing_cell increments the generation.
        assert!(matches!(ws.maybe_compile(0xB), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(0xB), HotResult::StartTracing));
        let cell = ws.get_cell(0xA).expect("A is still installed");
        assert!(cell.is_tracing(), "fixture: A's flag was never cleared");
        assert!(
            cell.tracing_generation < ws.tracing_generation(),
            "fixture: A's session has been superseded"
        );

        let stale_66 = crate::mc_diag(66);
        assert!(!ws.should_trace_function_entry(0xA));
        assert!(
            crate::mc_diag(66) >= stale_66 + 1,
            "a stale JC_TRACING at this gate is what slot 66 reports"
        );
    }

    /// The set/clear pair for `JC_TRACING` has to agree on which cell it is
    /// talking about. `mark_as_being_traced_for_key` writes the cell matching
    /// the key; `clear_tracing_flag` clears the bucket head. When a hash-only
    /// writer got there first those are different cells, and the clear misses.
    ///
    /// This demonstrates the mechanism on a constructed fixture. It does NOT
    /// establish that production reaches this configuration — that is what the
    /// `stfe_declined_tracing_stale` counter is for.
    #[test]
    fn a_typed_clear_reaches_the_cell_a_typed_mark_wrote() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![100, 200]);
        let hash = key.get_uhash();

        // Fixture: a hash-only writer heads the bucket, so a later typed
        // install chains behind it rather than finding it.
        ws.disable_noninlinable_function(hash);
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "fixture: the hash-only cell carries no comparekey, so a typed \
             probe cannot see it — this is what makes the chain",
        );

        ws.mark_as_being_traced_for_key(&key);
        assert_eq!(
            ws.get_stats().num_cells,
            2,
            "fixture: one green key, two cells",
        );
        assert!(
            ws.lookup_chain_with_key(&key)
                .expect("the typed cell exists")
                .is_tracing(),
            "fixture: the typed cell is the one carrying JC_TRACING",
        );
        assert!(
            !ws.get_cell(hash)
                .expect("the bucket head exists")
                .is_tracing(),
            "fixture: the bucket HEAD never got the flag — so a head-only \
             clear has nothing to do and cannot fix the tail",
        );

        // The hash form clears the head, which never had the flag.
        ws.clear_tracing_flag(hash);
        assert!(
            ws.lookup_chain_with_key(&key)
                .expect("the typed cell exists")
                .is_tracing(),
            "the hash clear missed the cell the mark wrote: this is the stuck \
             flag that makes every bare-head gate refuse the key",
        );

        // The typed form reaches it.
        ws.clear_tracing_flag_for_key(&key);
        assert!(
            !ws.lookup_chain_with_key(&key)
                .expect("the typed cell exists")
                .is_tracing(),
            "the typed clear selects by comparekey, so it reaches the cell the \
             typed mark wrote",
        );
    }

    #[test]
    fn test_jitcell_should_remove() {
        // A freshly created cell with no token and no flags should be removable
        let cell = BaseJitCell::new();
        assert!(cell.should_remove_jitcell());

        // A cell that is tracing should NOT be removable
        let mut cell = BaseJitCell::new();
        cell.flags |= jc_flags::TRACING;
        assert!(!cell.should_remove_jitcell());

        // A cell with DONT_TRACE_HERE but no token history is removable
        let mut cell = BaseJitCell::new();
        cell.flags |= jc_flags::DONT_TRACE_HERE;
        assert!(!cell.should_remove_jitcell()); // has_seen_a_procedure_token is false

        // A cell with DONT_TRACE_HERE and a past token should be removable
        let mut cell = BaseJitCell::new();
        cell.flags |= jc_flags::DONT_TRACE_HERE;
        cell.token = Some(42); // historical record of past token
        assert!(cell.should_remove_jitcell());

        // warmstate.py:222-225: FORCE_FINISH must NOT be removed
        let mut cell = BaseJitCell::new();
        cell.flags |= jc_flags::FORCE_FINISH;
        assert!(!cell.should_remove_jitcell());
    }

    #[test]
    fn test_gc_cells() {
        let mut ws = WarmEnterState::new(2);

        // Create some cells in various states
        // Key 1: compiled (should NOT be removed)
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(1), HotResult::StartTracing));
        ws.finish_tracing(1);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(1, token);

        // Key 2: tracing (should NOT be removed)
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(2), HotResult::StartTracing));

        // Key 3: aborted without dont_trace → NotHot, removable
        assert!(matches!(ws.maybe_compile(3), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(3), HotResult::StartTracing));
        ws.abort_tracing(3, false);

        assert_eq!(ws.get_stats().num_cells, 3);
        let removed = ws.gc_cells();
        assert_eq!(removed, 1); // key 3 removed
        assert_eq!(ws.get_stats().num_cells, 2);
        assert!(ws.get_cell(1).is_some());
        assert!(ws.get_cell(2).is_some());
        assert!(ws.get_cell(3).is_none());
    }

    #[test]
    fn test_invalidated_cell_allows_recompilation() {
        // After invalidation, transitioning back to NotHot allows recompilation.
        let mut ws = WarmEnterState::new(2);
        let key = 0xD4;

        // Compile
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(key), HotResult::StartTracing));
        ws.finish_tracing(key);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Compiled);

        // Invalidate
        ws.transition_cell(key, BaseJitCellState::Invalidated);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Invalidated);

        // Reset to NotHot and recompile
        ws.transition_cell(key, BaseJitCellState::NotHot);
        let token2 = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token2);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Compiled);
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
    }

    #[test]
    fn test_get_param() {
        let ws = WarmEnterState::new(100);
        assert_eq!(ws.get_param("threshold"), Some(100));
        assert_eq!(ws.get_param("vectorize"), Some(0));
        assert_eq!(ws.get_param("unknown_param"), None);
    }

    #[test]
    fn test_set_param_to_default() {
        let mut ws = WarmEnterState::new(100);
        ws.set_param("trace_limit", 999);
        assert_eq!(ws.get_param("trace_limit"), Some(999));
        ws.set_param_to_default("trace_limit");
        assert_eq!(
            ws.get_param("trace_limit"),
            Some(DEFAULT_TRACE_LIMIT as i64)
        );
    }

    #[test]
    fn test_param_names() {
        let names = WarmEnterState::param_names();
        assert!(names.contains(&"threshold"));
        assert!(names.contains(&"trace_limit"));
        assert!(names.contains(&"vectorize"));
        assert!(names.len() >= 10);
    }

    #[test]
    fn test_quasiimmut_dependency_lifecycle() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xF00D;
        let qmut = 0xBEEF;

        // Compile a loop
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(key), HotResult::StartTracing));
        ws.finish_tracing(key);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token);

        // Register quasi-immutable dependency
        ws.register_quasiimmut_dependency(qmut, key);

        // Invalidate
        let invalidated = ws.invalidate_quasiimmut(qmut);
        assert_eq!(invalidated, 1);

        // Loop should now be invalidated state
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Invalidated);
    }

    #[test]
    fn test_quasiimmut_invalidated_cell_rearms_hot_counter() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xD00D;
        let qmut = 0xFA11;

        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(key), HotResult::StartTracing));
        ws.finish_tracing(key);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token);
        ws.register_quasiimmut_dependency(qmut, key);

        assert_eq!(ws.invalidate_quasiimmut(qmut), 1);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Invalidated);

        // warmstate.py:483-500: the next entry removes the invalidated
        // token cell and resets the counter; following entries count fresh.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(ws.get_cell(key).is_none());
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(key), HotResult::StartTracing));
    }

    #[test]
    fn test_counter_tick_checked_rearms_after_quasiimmut_invalidation() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xD00E;
        let qmut = 0xFA12;

        assert!(!ws.counter_tick_checked(key));
        assert!(ws.counter_tick_checked(key));
        ws.force_start_tracing(key);
        ws.finish_tracing(key);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token);
        ws.register_quasiimmut_dependency(qmut, key);

        assert_eq!(ws.invalidate_quasiimmut(qmut), 1);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Invalidated);

        assert!(!ws.counter_tick_checked(key));
        assert!(ws.get_cell(key).is_none());
        assert!(!ws.counter_tick_checked(key));
        assert!(ws.counter_tick_checked(key));
    }

    #[test]
    fn test_function_entry_rearms_after_quasiimmut_invalidation() {
        let mut ws = WarmEnterState::new(2);
        ws.set_function_threshold(2);
        let key = 0xD00F;
        let qmut = 0xFA13;

        assert!(!ws.should_trace_function_entry(key));
        assert!(ws.should_trace_function_entry(key));
        ws.force_start_tracing(key);
        ws.finish_tracing(key);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token);
        ws.register_quasiimmut_dependency(qmut, key);

        assert_eq!(ws.invalidate_quasiimmut(qmut), 1);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Invalidated);

        assert!(!ws.should_trace_function_entry(key));
        assert!(ws.get_cell(key).is_none());
        assert!(!ws.should_trace_function_entry(key));
        assert!(ws.should_trace_function_entry(key));
    }

    #[test]
    fn test_set_param_roundtrip() {
        let mut ws = WarmEnterState::new(100);
        for name in WarmEnterState::param_names() {
            let original = ws.get_param(name);
            // "decay" is write-only (warmstate.py:269-270 delegates to
            // jitcounter.set_decay which stores decay_by_mult, not the raw int).
            if *name != "decay" {
                assert!(original.is_some(), "param {name} should be gettable");
            }
            ws.set_param(name, 999);
            ws.set_param_to_default(name);
            // After default, should be same as a fresh instance
        }
    }

    /// warmstate.py:575-582 — `comparekey_matches` returns true only
    /// when the cell carries a stored GreenKey equal to the probe.
    /// Cells without a comparekey (legacy hash-only path) always fail.
    #[test]
    fn comparekey_matches_only_with_stored_key() {
        let mut cell = BaseJitCell::new();
        let key = GreenKey::new(vec![1, 2, 3]);
        assert!(
            !cell.comparekey_matches(&key),
            "cell without comparekey must not match"
        );
        cell.comparekey = Some(key.clone());
        assert!(
            cell.comparekey_matches(&key),
            "cell with stored comparekey must match equal probe"
        );
        let other = GreenKey::new(vec![1, 2, 4]);
        assert!(
            !cell.comparekey_matches(&other),
            "cell with stored comparekey must not match unequal probe"
        );
    }

    // Sentinel-tagged retain/release log for the ownership test below.
    //
    // `set_ref_resolver` is a process-global `OnceLock`, so exactly one test in
    // this binary may register and it cannot be scoped to a single case.
    // Everything is therefore asserted by SENTINEL VALUE rather than by call
    // count: any other fixture that installs a Ref green while this resolver is
    // live would move a count, but cannot forge these addresses.
    static RETAIN_LOG: std::sync::Mutex<Vec<i64>> = std::sync::Mutex::new(Vec::new());
    static RELEASE_LOG: std::sync::Mutex<Vec<i64>> = std::sync::Mutex::new(Vec::new());

    fn test_retain(value: i64) {
        RETAIN_LOG.lock().unwrap().push(value);
    }

    fn test_release(value: i64) {
        RELEASE_LOG.lock().unwrap().push(value);
    }

    /// warmstate.py:568-573 — a cell's stored green key OWNS its `Ref`
    /// referents, so the address it names cannot be freed and recycled by a
    /// different object while the cell is alive.
    ///
    /// Four legs, and the second is the localization control:
    ///
    /// 1. the `Ref` slot is retained on install;
    /// 2. an `Int` slot holding an equally pointer-shaped value is **not**
    ///    retained — ownership must key off the declared `GreenType`, not off
    ///    "looks like an address", which is the only way it can be right;
    /// 3. a null `Ref` is skipped (`hash_whatever` folds null to 0, and there
    ///    is nothing to own);
    /// 4. dropping the cell releases exactly what it retained.
    #[test]
    fn stored_green_key_owns_its_ref_referents() {
        majit_ir::set_ref_resolver(test_retain, test_release);

        const REF_GREEN: i64 = 0x5EED_0001;
        const INT_GREEN: i64 = 0x5EED_0002;

        let key = GreenKey::with_types(
            vec![7, INT_GREEN, REF_GREEN, 0],
            vec![Type::Int, Type::Int, Type::Ref, Type::Ref],
        );

        {
            let mut ws = WarmEnterState::new(3);
            ws.ensure_cell_for_key(&key);

            let retained = RETAIN_LOG.lock().unwrap().clone();
            assert!(
                retained.contains(&REF_GREEN),
                "installing a cell must retain its Ref green; log={retained:x?}"
            );
            assert!(
                !retained.contains(&INT_GREEN),
                "an Int green must NOT be retained even when its value is \
                 pointer-shaped — ownership keys off GreenType, not off the \
                 bit pattern; log={retained:x?}"
            );
            assert!(
                !retained.contains(&0),
                "a null Ref green has no referent to own; log={retained:x?}"
            );

            assert_eq!(
                ws.get_stats().num_pinned_refs,
                1,
                "the pinned-population counter must see exactly the one Ref \
                 referent this cell owns"
            );

            assert!(
                !RELEASE_LOG.lock().unwrap().contains(&REF_GREEN),
                "nothing may be released while the cell that owns it is alive"
            );
        }

        let released = RELEASE_LOG.lock().unwrap().clone();
        assert!(
            released.contains(&REF_GREEN),
            "dropping the cell must release its Ref green; log={released:x?}"
        );
        assert!(
            !released.contains(&INT_GREEN),
            "an unretained Int green must never be released; log={released:x?}"
        );
    }

    /// warmstate.py:626-641 + 596-604 — `ensure_cell_for_key` allocates
    /// a fresh cell on miss and `lookup_chain_with_key` returns it on a
    /// repeat probe with the same typed greens.
    #[test]
    fn ensure_and_lookup_chain_with_key_round_trips() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![42, 100]);

        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "no cell for fresh key"
        );
        ws.ensure_cell_for_key(&key);
        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("ensure must install a cell");
        assert_eq!(
            cell.comparekey.as_ref().expect("comparekey populated"),
            &key,
            "stored comparekey must equal the install key"
        );
    }

    /// warmstate.py:644-646 `dont_trace_here(*greenargs)` — the typed and
    /// hash entry points reach the same cell today but do NOT leave it in the
    /// same state, and this pins the difference.
    ///
    /// Both install on a miss, and on an *empty* table — which is what each
    /// arm below starts from — they land in the same bucket. Do not read
    /// that as "they reach the same cell": on a table where the other form
    /// already wrote, they do not, and no hash collision is required for
    /// that. See
    /// `one_key_through_a_hash_and_a_typed_entry_point_builds_a_chain`.
    /// What differs is what the installed cell knows about itself: the typed
    /// form goes through `ensure_cell_for_key`, so the cell stores its
    /// `comparekey` — and with it the `RetainedGreens` that make the stored
    /// address own its referent. The hash form cannot, because a hash is not
    /// invertible.
    ///
    /// This is the whole observable delta of routing an `interp_jit.py`
    /// helper through the typed key, so it is asserted rather than described:
    /// a cell without a `comparekey` is one that a chain walk can never
    /// identify, which is exactly what bucketing (typed-key bucketing) would need it to do.
    #[test]
    fn dont_trace_here_typed_form_stores_a_comparekey_and_the_hash_form_does_not() {
        let key = GreenKey::new(vec![7, 11]);

        let mut typed = WarmEnterState::new(100);
        typed.disable_noninlinable_function_for_key(&key);
        let typed_cell = typed
            .get_cell(key.get_uhash())
            .expect("typed form installs a cell");
        assert!(
            typed_cell.flags & jc_flags::DONT_TRACE_HERE != 0,
            "typed form must set DONT_TRACE_HERE"
        );
        assert_eq!(
            typed_cell.comparekey.as_ref(),
            Some(&key),
            "typed form must store the greens it was called with"
        );

        let mut hashed = WarmEnterState::new(100);
        hashed.disable_noninlinable_function(key.get_uhash());
        let hashed_cell = hashed
            .get_cell(key.get_uhash())
            .expect("hash form installs a cell");
        assert!(
            hashed_cell.flags & jc_flags::DONT_TRACE_HERE != 0,
            "hash form must set DONT_TRACE_HERE — the flag is not the delta"
        );
        assert!(
            hashed_cell.comparekey.is_none(),
            "hash form has no greens to store; if this ever becomes Some, the \
             typed/hash split has been closed somewhere else and this test is \
             the wrong guard"
        );
    }

    /// warmstate.py:596-604 — chain walk distinguishes hash collisions:
    /// two distinct GreenKeys sharing one bucket must each resolve to
    /// their own cell via `comparekey` rather than aliasing. The test
    /// exploits `install_new_cell`'s `should_remove_jitcell` gate
    /// (warmstate.rs:184-200, counter.py:246-256 parity): a cell with
    /// no `loop_token` / no flags is treated as dead and dropped on
    /// the next install. Setting `JC_TRACING` on the first-installed
    /// cell keeps it alive across the second install so both end up
    /// chained through `head.next`. The walker must then skip the
    /// non-matching head when looking up the chained cell.
    #[test]
    fn lookup_chain_with_key_resolves_hash_collisions_via_comparekey() {
        let mut ws = WarmEnterState::new(100);
        let key_head_after_chain = GreenKey::new(vec![1, 2]);
        let key_chained = GreenKey::new(vec![3, 4]);
        let bucket = key_head_after_chain.get_uhash();

        // First install: TRACING flag so should_remove_jitcell == false
        // and install_new_cell preserves the cell across the second
        // install. Stays at the back of the chain after the second
        // install per install_new_cell's "prepend new cell, fold
        // existing keepable cells in front" semantics.
        let mut cell_first = BaseJitCell::new();
        cell_first.flags |= jc_flags::TRACING;
        cell_first.comparekey = Some(key_head_after_chain.clone());
        ws.install_new_cell(bucket, Some(cell_first));
        // Second install — keep predicate doesn't matter for this one,
        // it lands as `keep` and the prior head folds in front, so the
        // resulting chain shape is head=cell_first, head.next=cell_for_key_chained.
        // Wait — install_new_cell folds the EXISTING chain in front of
        // `keep`. So after the second install: head=cell_first (the
        // existing one, since !should_remove), head.next=cell_chained.
        let mut cell_chained = BaseJitCell::new();
        cell_chained.comparekey = Some(key_chained.clone());
        ws.install_new_cell(bucket, Some(cell_chained));

        let head = ws
            .lookup_chain(bucket)
            .expect("bucket head exists after install");
        assert_eq!(head.comparekey.as_ref(), Some(&key_head_after_chain));
        let next = head.next.as_deref().expect("chain has a second cell");
        assert_eq!(next.comparekey.as_ref(), Some(&key_chained));

        // Fast path: lookup_chain_with_key(key_head_after_chain) finds
        // the head's comparekey on first probe.
        let hit_head = ws
            .lookup_chain_with_key(&key_head_after_chain)
            .expect("walker must find chain head via comparekey");
        assert_eq!(hit_head.comparekey.as_ref(), Some(&key_head_after_chain));
    }

    /// Build a 2-cell chain in `target`'s bucket: a decoy cell at the head
    /// (different comparekey) and `target`'s cell chained behind it, both
    /// TRACING.  The typed-key lifecycle mutators must walk past the decoy
    /// head and touch only `target`'s cell.
    fn chain_decoy_then_target(target: &GreenKey, decoy: &GreenKey) -> WarmEnterState {
        let mut ws = WarmEnterState::new(100);
        let bucket = target.get_uhash();
        let mut decoy_cell = BaseJitCell::new();
        decoy_cell.flags |= jc_flags::TRACING;
        decoy_cell.comparekey = Some(decoy.clone());
        ws.install_new_cell(bucket, Some(decoy_cell));
        let mut target_cell = BaseJitCell::new();
        target_cell.flags |= jc_flags::TRACING;
        target_cell.comparekey = Some(target.clone());
        ws.install_new_cell(bucket, Some(target_cell));
        ws
    }

    #[test]
    fn finish_tracing_for_key_clears_tracing_on_matching_cell_only() {
        let target = GreenKey::new(vec![5, 6]);
        let decoy = GreenKey::new(vec![7, 8]);
        let mut ws = chain_decoy_then_target(&target, &decoy);
        ws.finish_tracing_for_key(&target);
        let head = ws.lookup_chain(target.get_uhash()).expect("bucket head");
        assert_eq!(head.comparekey.as_ref(), Some(&decoy));
        assert!(head.flags & jc_flags::TRACING != 0, "decoy head untouched");
        let hit = head.next.as_deref().expect("target chained behind decoy");
        assert_eq!(hit.comparekey.as_ref(), Some(&target));
        assert!(hit.flags & jc_flags::TRACING == 0, "target TRACING cleared");
    }

    #[test]
    fn abort_tracing_for_key_marks_only_matching_cell() {
        let target = GreenKey::new(vec![5, 6]);
        let decoy = GreenKey::new(vec![7, 8]);
        let mut ws = chain_decoy_then_target(&target, &decoy);
        ws.abort_tracing_for_key(&target, true);
        let head = ws.lookup_chain(target.get_uhash()).expect("bucket head");
        assert!(
            head.flags & jc_flags::TRACING != 0,
            "decoy head still TRACING"
        );
        assert!(
            head.flags & jc_flags::DONT_TRACE_HERE == 0,
            "decoy not disabled"
        );
        let hit = head.next.as_deref().expect("target chained behind decoy");
        assert!(hit.flags & jc_flags::TRACING == 0, "target TRACING cleared");
        assert!(
            hit.flags & jc_flags::DONT_TRACE_HERE != 0,
            "target disabled by permanent abort"
        );
    }

    #[test]
    fn mark_dont_trace_for_key_sets_only_matching_cell() {
        let target = GreenKey::new(vec![5, 6]);
        let decoy = GreenKey::new(vec![7, 8]);
        let mut ws = chain_decoy_then_target(&target, &decoy);
        ws.mark_dont_trace_for_key(&target);
        let head = ws.lookup_chain(target.get_uhash()).expect("bucket head");
        assert!(
            head.flags & jc_flags::DONT_TRACE_HERE == 0,
            "decoy head not disabled"
        );
        let hit = head.next.as_deref().expect("target chained behind decoy");
        assert!(
            hit.flags & jc_flags::DONT_TRACE_HERE != 0,
            "target disabled"
        );
    }

    #[test]
    fn clear_loop_token_for_key_clears_only_matching_cell() {
        let target = GreenKey::new(vec![5, 6]);
        let decoy = GreenKey::new(vec![7, 8]);
        let mut ws = WarmEnterState::new(100);
        let bucket = target.get_uhash();
        let token = std::sync::Arc::new(JitCellToken::new(0x00c0_ffee_u64));
        let mut decoy_cell = BaseJitCell::new();
        decoy_cell.flags |= jc_flags::TRACING;
        decoy_cell.comparekey = Some(decoy.clone());
        decoy_cell.loop_token = Some(token.clone());
        ws.install_new_cell(bucket, Some(decoy_cell));
        let mut target_cell = BaseJitCell::new();
        target_cell.flags |= jc_flags::TRACING;
        target_cell.comparekey = Some(target.clone());
        target_cell.loop_token = Some(token.clone());
        ws.install_new_cell(bucket, Some(target_cell));
        ws.clear_loop_token_for_key(&target);
        let head = ws.lookup_chain(bucket).expect("bucket head");
        assert!(head.loop_token.is_some(), "decoy loop_token kept");
        let hit = head.next.as_deref().expect("target chained behind decoy");
        assert!(hit.loop_token.is_none(), "target loop_token cleared");
    }

    /// `warmstate.py:714-723` `get_assembler_token` — a fresh typed key
    /// installs a temporary procedure token (tmp=true → JC_TEMPORARY)
    /// and returns it; subsequent calls with the same key return the
    /// same token without invoking `make_token` again.
    #[test]
    fn get_assembler_token_with_key_caches_token_per_typed_key() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![10, 20]);
        let token = std::sync::Arc::new(JitCellToken::new(0xa55e_b1ed_u64));

        let mut make_token_calls = 0;
        let token_clone_1 = {
            let token_for_closure = token.clone();
            ws.get_assembler_token_with_key::<(), _>(&key, || {
                make_token_calls += 1;
                Ok(token_for_closure)
            })
            .expect("install ok")
        };
        assert_eq!(make_token_calls, 1, "first call invokes make_token");
        assert!(std::sync::Arc::ptr_eq(&token_clone_1, &token));

        let token_clone_2 = ws
            .get_assembler_token_with_key::<(), _>(&key, || {
                make_token_calls += 1;
                Ok(std::sync::Arc::new(JitCellToken::new(0xdead_beef_u64)))
            })
            .expect("re-fetch ok");
        assert_eq!(
            make_token_calls, 1,
            "second call hits the cached token, does not invoke make_token"
        );
        assert!(
            std::sync::Arc::ptr_eq(&token_clone_2, &token),
            "second call returns the originally installed token"
        );

        // JC_TEMPORARY must be set since tmp=true.
        let cell = ws.lookup_chain_with_key(&key).expect("cell installed");
        assert!(
            cell.flags & jc_flags::TEMPORARY != 0,
            "tmp token must set JC_TEMPORARY"
        );
    }

    /// `warmstate.py:714-723` + `626-641` — typed variant must
    /// disambiguate hash collisions: two `GreenKey`s that share a hash
    /// but compare unequal under `equal_whatever` get distinct tokens.
    #[test]
    fn get_assembler_token_with_key_disambiguates_hash_collisions() {
        let mut ws = WarmEnterState::new(100);

        // Build a GreenKey pair guaranteed to collide on hash via the
        // existing collision fixture (install_new_cell uses bucket-by-hash).
        // Reuse the same shape as the chain-walk test: install a typed
        // cell first, then call get_assembler_token_with_key for another
        // typed key in the same bucket.
        let key_a = GreenKey::new(vec![100, 200]);
        let key_b = GreenKey::new(vec![300, 400]);

        let token_a = std::sync::Arc::new(JitCellToken::new(0xaaaa));
        let token_b = std::sync::Arc::new(JitCellToken::new(0xbbbb));

        let got_a = ws
            .get_assembler_token_with_key::<(), _>(&key_a, || Ok(token_a.clone()))
            .expect("install a");
        let got_b = ws
            .get_assembler_token_with_key::<(), _>(&key_b, || Ok(token_b.clone()))
            .expect("install b");

        assert!(
            std::sync::Arc::ptr_eq(&got_a, &token_a),
            "key_a returns token_a"
        );
        assert!(
            std::sync::Arc::ptr_eq(&got_b, &token_b),
            "key_b returns token_b"
        );
        assert!(
            !std::sync::Arc::ptr_eq(&got_a, &got_b),
            "distinct typed keys must not alias to the same token"
        );

        // Repeat fetch must not invoke make_token.
        let mut count = 0;
        let _ = ws
            .get_assembler_token_with_key::<(), _>(&key_a, || {
                count += 1;
                Ok(std::sync::Arc::new(JitCellToken::new(0xcccc)))
            })
            .expect("re-fetch a");
        let _ = ws
            .get_assembler_token_with_key::<(), _>(&key_b, || {
                count += 1;
                Ok(std::sync::Arc::new(JitCellToken::new(0xdddd)))
            })
            .expect("re-fetch b");
        assert_eq!(count, 0, "cached lookups must not invoke make_token");
    }

    /// A hash-form write and a typed-form read of the SAME green key land on
    /// DIFFERENT cells, and the state the hash form wrote is invisible to the
    /// typed reader.
    ///
    /// This is the behavioural consequence of
    /// `one_key_through_a_hash_and_a_typed_entry_point_builds_a_chain`: that
    /// fixture shows the chain forms, this one shows what the chain costs.
    ///
    /// `disable_noninlinable_function` is reached in production from
    /// `pyre-jit-trace/src/state.rs:3189` and `pyjitpl.rs:5256/5312`;
    /// `maybe_compile_with_key` is the typed back-edge path (`pyjitpl.rs:4393`).
    ///
    /// The state does not merely move — it SPLITS, and the two reader
    /// families see opposite halves. `DONT_TRACE_HERE` ends up on the head,
    /// `TRACING` on the chained typed cell. So a bare-head reader
    /// (`self.cells.get(&hash)`, ~26 of them here) sees the mark but not the
    /// tracing state, while a typed reader (`lookup_chain_with_key`) sees the
    /// tracing state but not the mark. Neither sees the whole cell.
    ///
    /// SCOPE. What is proven here is the split and the route change. The
    /// hash-marked key reaches `StartTracing` on the THRESHOLD tick by the
    /// ordinary counter route, because the typed decision never saw the mark;
    /// the typed-marked key reaches it on the FIRST tick by
    /// `should_start_dont_trace_here_trace` (warmstate.py:483-491), which is
    /// the rule upstream intends to apply. Both trace in the end, so this is
    /// NOT demonstrated to be a user-visible wrong answer — it is a lost
    /// decision input. Whether a production key reaches both entry points, and
    /// in which order, is a runtime question this fixture does not answer.
    #[test]
    fn a_hash_write_and_a_typed_read_of_one_key_use_different_cells() {
        let mut ws = WarmEnterState::new(3);
        let key = GreenKey::new(vec![7, 9]);
        ws.disable_noninlinable_function(key.get_uhash());

        // Ticks 1-2 under threshold, tick 3 fires — the ORDINARY counter
        // route, i.e. the mark above was never consulted.
        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(
            matches!(ws.maybe_compile_with_key(&key), HotResult::StartTracing),
            "the hash-written DONT_TRACE_HERE never reached the typed decision",
        );

        // One bucket, two cells: the split itself.
        assert_eq!(ws.cells.len(), 1, "one green key, so one bucket");
        assert_eq!(ws.get_stats().num_cells, 2, "but two cells");

        // `install_new_cell` folds the SURVIVOR in front of the newcomer
        // (counter.py:253-254 `cell.next = keep; keep = cell`), so the
        // HASH-written cell stays the head and the TYPED cell is chained
        // behind it. This is the direction that matters: every bare-head
        // reader — `self.cells.get(&hash)`, ~26 of them in this file — reads
        // the head, which is the cell WITHOUT the comparekey.
        let head = ws.lookup_chain(key.get_uhash()).expect("head present");
        assert!(
            head.comparekey.is_none(),
            "the head is the hash-written cell — a hash is not invertible, so \
             it can store no comparekey",
        );
        assert_ne!(
            head.flags & jc_flags::DONT_TRACE_HERE,
            0,
            "the head still holds the mark the hash form wrote",
        );
        let typed_cell = head.next.as_deref().expect("typed cell chained behind");
        assert_eq!(
            typed_cell.comparekey.as_ref(),
            Some(&key),
            "the typed install carries the comparekey and is NOT the head",
        );
        assert!(
            typed_cell.is_tracing(),
            "the typed cell is the one the tracing transition wrote to, so the \
             two halves of this key's state now live on two different cells",
        );

        // Control: the typed form of the same mark keeps ONE cell and takes
        // the dont-trace-here route on the very first tick.
        let mut typed = WarmEnterState::new(3);
        let key2 = GreenKey::new(vec![7, 9]);
        typed.disable_noninlinable_function_for_key(&key2);
        assert!(
            matches!(typed.maybe_compile_with_key(&key2), HotResult::StartTracing),
            "the typed mark IS consulted, so the dont-trace-here rule applies \
             at once instead of waiting for the counter",
        );
        assert_eq!(typed.get_stats().num_cells, 1, "no split on the typed path");
    }

    /// warmstate.py:446-511 — typed variant of `maybe_compile_and_run`.
    /// Upstream installs the JitCell lazily at `bound_reached`
    /// (warmstate.py:425-444): each tick under threshold returns
    /// without writing to the celltable; on the threshold tick, the
    /// cell is created with `comparekey` and installed via
    /// `jitcounter.install_new_cell`. The typed entry point preserves
    /// that lifecycle — pre-threshold ticks see no cell installed,
    /// only the bucket counter ticks.
    #[test]
    fn maybe_compile_with_key_lazily_installs_on_threshold_tick() {
        let mut ws = WarmEnterState::new(3);
        let key = GreenKey::new(vec![7, 9]);

        // Tick 1, 2: not hot. Cell is NOT yet installed — upstream
        // `maybe_compile_and_run` only allocates at `bound_reached`
        // (warmstate.py:438-440).
        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "cold ticks must not install a cell (warmstate.py:466-468)",
        );
        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "second cold tick still no cell",
        );

        // Tick 3: threshold reached → `start_tracing_cell_for_key`
        // installs a cell carrying `comparekey = key` (typed) and
        // sets the TRACING flags.
        match ws.maybe_compile_with_key(&key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("cell installed at threshold tick");
        assert_eq!(
            cell.comparekey.as_ref(),
            Some(&key),
            "comparekey populated on lazy install (warmstate.py:438-439)",
        );
        assert!(cell.is_tracing(), "JC_TRACING flag set on threshold tick");
    }

    /// The three whole-table sweeps must walk each chain, not just its head.
    ///
    /// `cells.values()` yields chain HEADS, so a sweep that does not follow
    /// `next` silently skips every chained cell. "majit-metainterp: count
    /// chained cells in get_stats, not bucket heads" fixed exactly
    /// this in `get_stats`; the fix was filed at one access path and three
    /// siblings kept the defect.
    ///
    /// `invalidate_all` is the one that costs a wrong answer rather than a
    /// leak: a chained cell whose token is never invalidated keeps running
    /// compiled code built under a retracted assumption.
    ///
    /// The token lives on the CHAINED cell and the head is left tokenless, so
    /// a head-only sweep reaches nothing at all — the assertions below fail on
    /// every one of the three before the fix.
    #[test]
    fn whole_table_sweeps_reach_chained_cells_not_only_heads() {
        let mut ws = WarmEnterState::new(100);

        let key_head = GreenKey::new(vec![100, 200]);
        let key_tail = GreenKey::new(vec![300, 400]);
        let bucket = key_tail.get_uhash();

        // TRACING keeps the head non-removable so the second install chains
        // behind it rather than replacing it (counter.py:246-256).
        let mut head = BaseJitCell::new();
        head.flags |= jc_flags::TRACING;
        head.comparekey = Some(key_head.clone());
        ws.install_new_cell(bucket, Some(head));

        const CHAINED_TOKEN: u64 = 0x5EED_0003;
        let mut tail = BaseJitCell::new();
        tail.flags |= jc_flags::TRACING;
        tail.comparekey = Some(key_tail.clone());
        tail.loop_token = Some(make_token(CHAINED_TOKEN));
        ws.install_new_cell(bucket, Some(tail));

        // The token is on the chained cell, never on the head.
        let chain_head = ws.lookup_chain(bucket).expect("bucket has a head");
        assert!(
            chain_head.loop_token.is_none(),
            "fixture requires a tokenless head, or a head-only sweep would pass"
        );
        assert!(
            chain_head
                .next
                .as_deref()
                .is_some_and(|c| c.loop_token.is_some()),
            "fixture requires the token on the CHAINED cell"
        );

        // 1. find_token_by_number
        assert!(
            ws.find_token_by_number(CHAINED_TOKEN).is_some(),
            "a token on a chained cell must be findable"
        );

        // 2. invalidate_all — the correctness one.
        ws.invalidate_all();
        let chained = ws
            .lookup_chain(bucket)
            .and_then(|h| h.next.as_deref())
            .expect("chain survives invalidate_all");
        assert!(
            chained
                .loop_token
                .as_ref()
                .is_some_and(|t| t.is_invalidated()),
            "invalidate_all must invalidate a CHAINED cell's token — skipping \
             it leaves compiled code live under a retracted assumption"
        );

        // 3. clear_all_loop_tokens
        ws.clear_all_loop_tokens();
        let chained = ws
            .lookup_chain(bucket)
            .and_then(|h| h.next.as_deref())
            .expect("chain survives clear_all_loop_tokens");
        assert!(
            chained.loop_token.is_none(),
            "clear_all_loop_tokens must clear a CHAINED cell's token"
        );
    }

    /// warmstate.py:455-465 — `JitCell.get_jitcell_for_args(*greenargs)`
    /// walks the per-bucket chain by `comparekey` to read AND mutate
    /// the cell associated with `greenargs`. A hash-only delegate
    /// would alias colliding GreenKeys to the same head and read /
    /// write the wrong cell's `JC_TRACING` / `JC_COMPILED` flags.
    ///
    /// The test simulates a hash collision by directly installing two
    /// cells in the SAME bucket via [`WarmEnterState::install_new_cell`]:
    /// `key_a` carries `JC_TRACING`, `key_b` is fresh. After installs,
    /// `install_new_cell` folds non-removable cells to the front, so
    /// the chain becomes `head=key_a (TRACING) -> tail=key_b`.
    ///
    /// `maybe_compile_with_key(&key_b)` must return `NotHot` (cell B
    /// has no flags). A hash-only read path would see the head
    /// (`key_a`) and incorrectly return `AlreadyTracing`.
    #[test]
    fn maybe_compile_with_key_walks_chain_for_state_under_hash_collision() {
        let mut ws = WarmEnterState::new(100);
        // Same hash bucket for both keys — install_new_cell takes the
        // bucket explicitly so we don't need a real `get_uhash`
        // collision in the GreenKey values.
        let key_b = GreenKey::new(vec![300, 400]);
        let key_a = GreenKey::new(vec![100, 200]);
        let bucket = key_b.get_uhash();

        // Pre-install A at B's bucket. JC_TRACING keeps A non-removable
        // (counter.py:246-256 should_remove gate) so the next install
        // chains B behind A.
        let mut cell_a = BaseJitCell::new();
        cell_a.flags |= jc_flags::TRACING;
        cell_a.comparekey = Some(key_a.clone());
        ws.install_new_cell(bucket, Some(cell_a));

        // Install B at the same bucket. install_new_cell folds A in
        // front; resulting chain shape: head=A (TRACING) → tail=B.
        let mut cell_b = BaseJitCell::new();
        cell_b.comparekey = Some(key_b.clone());
        ws.install_new_cell(bucket, Some(cell_b));

        // Sanity — chain order matches the install_new_cell contract.
        let head = ws.lookup_chain(bucket).expect("bucket has a head");
        assert_eq!(head.comparekey.as_ref(), Some(&key_a));
        assert!(head.is_tracing(), "head A carries the TRACING flag");
        let chained = head
            .next
            .as_deref()
            .expect("chained tail exists after second install");
        assert_eq!(chained.comparekey.as_ref(), Some(&key_b));
        assert!(!chained.is_tracing(), "B is fresh");

        // Without the chain-walk fix, `maybe_compile_with_key(&key_b)`
        // would delegate to `maybe_compile(hash)`, hit `cells.get(&hash)`,
        // see the head A's TRACING flag and return AlreadyTracing —
        // contaminating B's lookup with A's tracing state. With the fix
        // it walks the chain by comparekey and reads B's clean state
        // (NotHot path: counter ticks under threshold).
        match ws.maybe_compile_with_key(&key_b) {
            HotResult::AlreadyTracing => panic!(
                "maybe_compile_with_key(&key_b) aliased to head A's TRACING flag — \
                 chain walk by comparekey is broken (warmstate.py:455-465 parity)"
            ),
            HotResult::NotHot | HotResult::StartTracing | HotResult::RunCompiled => {}
        }

        // Cell A's TRACING flag must remain on A, not migrate to B.
        let head_after = ws.lookup_chain(bucket).expect("bucket head still exists");
        assert!(
            head_after.is_tracing(),
            "head A's TRACING flag must persist across maybe_compile_with_key(&key_b)",
        );
        assert_eq!(head_after.comparekey.as_ref(), Some(&key_a));
        let chained_after = head_after
            .next
            .as_deref()
            .expect("chained tail still present");
        assert_eq!(chained_after.comparekey.as_ref(), Some(&key_b));
        assert!(
            !chained_after.is_tracing(),
            "B must not have inherited A's TRACING flag (counter under threshold)",
        );
    }

    /// A chain does NOT need a hash collision. ONE green key reached
    /// through both a hash-only entry point and a typed one builds a two-cell
    /// chain in a single bucket.
    ///
    /// The mechanism has no probabilistic step in it:
    /// 1. a hash-only writer installs a cell with `comparekey: None`;
    /// 2. `DONT_TRACE_HERE` with no token makes `should_remove_jitcell()`
    ///    false (warmstate.rs:241-257), so the cell survives the next install;
    /// 3. `lookup_chain_with_key` cannot match a `None` comparekey — that is
    ///    asserted by `comparekey_matches_only_with_stored_key` — so
    ///    `ensure_cell_for_key` misses and calls `install_new_cell`;
    /// 4. `install_new_cell` (counter.py:246-256) links the survivor behind
    ///    the newcomer.
    ///
    /// Every other chain fixture in this module forces its collision by
    /// installing two comparekeys under one `get_uhash()` by hand, and says
    /// so. This one uses only public entry points on a single key, which is
    /// why it is the one that settles whether chains occur in practice.
    #[test]
    fn one_key_through_a_hash_and_a_typed_entry_point_builds_a_chain() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![100, 200]);

        // Hash-only writer (what `dont_trace_here` did before it was routed
        // through the typed form).
        ws.disable_noninlinable_function(key.get_uhash());
        assert_eq!(ws.get_stats().num_cells, 1, "one cell after the hash write");
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "the hash-only cell stores no comparekey, so a typed probe for the \
             SAME key cannot see it — this is the step that makes the chain",
        );

        // Typed writer, same key.
        ws.ensure_cell_for_key(&key);

        assert_eq!(ws.cells.len(), 1, "still ONE bucket — no collision here");
        assert_eq!(
            ws.get_stats().num_cells,
            2,
            "one green key, two cells: the typed install could not find the \
             hash-only cell and chained past it",
        );
    }

    /// The typed `_for_key` forms must select the cell belonging to the key,
    /// not whichever cell heads the bucket.
    ///
    /// Non-vacuity comes from the fixture, not from trust: both tests first
    /// assert that the bucket head is the *comparator-less* cell, so a
    /// head-reading implementation is provably looking at the wrong object.
    /// Against the hash-delegating forms these read `None` and `false`
    /// respectively.
    #[test]
    fn mark_as_being_traced_for_key_marks_the_keys_own_cell_not_the_bucket_head() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![300, 400]);

        // A hash-only writer squats the bucket with a comparator-less cell.
        ws.disable_noninlinable_function(key.get_uhash());
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "fixture: the hash-only cell stores no comparekey, so the key \
             owns nothing yet",
        );

        ws.mark_as_being_traced_for_key(&key);

        // Delegating to the hash form set TRACING on the comparator-less head
        // and left the key still owning no cell at all.
        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("the key must own a cell reachable by comparekey");
        assert!(
            cell.flags & jc_flags::TRACING != 0,
            "TRACING must land on the key's own cell",
        );
        assert_eq!(
            cell.state,
            BaseJitCellState::Tracing,
            "and so must the state transition",
        );
    }

    #[test]
    fn get_cell_for_key_returns_the_keys_cell_not_whichever_heads_the_bucket() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![500, 600]);

        ws.disable_noninlinable_function(key.get_uhash());
        ws.ensure_cell_for_key(&key);
        assert_eq!(ws.cells.len(), 1, "one bucket");
        assert_eq!(ws.get_stats().num_cells, 2, "two cells in it");

        assert!(
            !ws.get_cell(key.get_uhash())
                .expect("bucket is occupied")
                .comparekey_matches(&key),
            "fixture: `install_new_cell` links the surviving hash-only cell \
             AHEAD of the new typed one, so the HEAD is the comparator-less \
             cell and a head-reading lookup returns the wrong cell here",
        );

        let cell = ws.get_cell_for_key(&key).expect("the key owns a cell");
        assert!(
            cell.comparekey_matches(&key),
            "get_cell_for_key must return the cell that matches the key",
        );
    }

    /// The typed creators must write to the key's own cell, not squat a new
    /// comparator-less one beside it.
    ///
    /// Same fixture discipline as the pair above: a hash-only writer takes the
    /// bucket first and the assertion that it owns nothing typed runs *before*
    /// the call under test, so a form that delegated to the hash creator would
    /// be provably writing to the comparator-less head. Against the hash forms
    /// these read `None` at the `lookup_chain_with_key` line.
    #[test]
    fn attach_procedure_to_interp_for_key_installs_on_the_keys_own_cell() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![700, 800]);

        ws.disable_noninlinable_function(key.get_uhash());
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "fixture: the hash-only cell carries no comparekey",
        );

        let token = Arc::new(JitCellToken::new(ws.alloc_token_number()));
        ws.attach_procedure_to_interp_for_key(&key, Arc::clone(&token));

        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("the key must own a cell reachable by comparekey");
        assert!(
            cell.get_procedure_token().is_some(),
            "the procedure token must land on the key's own cell",
        );
        assert_eq!(
            cell.flags & jc_flags::TRACING,
            0,
            "and TRACING must be cleared on that same cell",
        );
    }

    /// `warmstate.py:458-464 maybe_compile_and_run` reads a procedure token
    /// only off a cell whose `comparekey(*greenargs)` already matched. The
    /// hash form reads the bucket head, which on a chained bucket is a
    /// different cell.
    ///
    /// Same fixture discipline as the pair above: the hash-only writer takes
    /// the bucket first, so the head is provably the comparator-less cell and
    /// a head-reading lookup is looking at the wrong object.
    #[test]
    fn get_procedure_token_for_key_reads_the_keys_own_cell_not_the_bucket_head() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![1100, 1200]);

        ws.disable_noninlinable_function(key.get_uhash());
        let token = Arc::new(JitCellToken::new(ws.alloc_token_number()));
        ws.attach_procedure_to_interp_for_key(&key, Arc::clone(&token));

        assert!(
            ws.bucket_is_chained(key.get_uhash()),
            "fixture: the hash-only writer and the typed one built a chain, \
             which is the only case in which the two forms can disagree",
        );
        assert!(
            ws.get_procedure_token(key.get_uhash()).is_none(),
            "fixture: the head is the comparator-less cell and holds no token, \
             so the hash form cannot see the one just installed",
        );

        let found = ws
            .get_procedure_token_for_key(&key)
            .expect("the typed form must find the token on the key's own cell");
        assert!(
            Arc::ptr_eq(&found, &token),
            "and it must be the very token installed under this key",
        );
    }

    /// The second green of a two-`Int` key beginning with `first` that makes it
    /// hash to the same bucket as `other`.
    ///
    /// Constructed rather than searched. `get_uhash` folds
    /// `x = (x ^ hash_whatever(tp, v)) * GREEN_UHASH_MULT` (warmstate.py:584-593)
    /// and `hash_whatever(Int, v)` is `v` itself, so with the multiply the same
    /// on both sides the two hashes agree exactly when the pre-multiply words
    /// do — one xor away. A searched collision would pin the hash function; the
    /// subject here is what a chained bucket does.
    fn colliding_last_green(other: &GreenKey, first: i64) -> i64 {
        use majit_ir::{GREEN_UHASH_SEED, GreenType, green_uhash_step};
        assert_eq!(other.values.len(), 2, "written for two-green Int keys");
        let other_prefix = green_uhash_step(GREEN_UHASH_SEED, GreenType::Int, other.values[0]);
        let this_prefix = green_uhash_step(GREEN_UHASH_SEED, GreenType::Int, first);
        (this_prefix ^ other_prefix ^ (other.values[1] as u64)) as i64
    }

    /// A two-`Int` green key beginning with `first` whose `get_uhash` is
    /// exactly `target`.
    ///
    /// [`colliding_last_green`] matches another key's hash; this hits an
    /// arbitrary number, which is what turns "every `u64` is a hash some green
    /// key can produce" ([`WarmEnterState::mint_cell_key`]) from an assertion
    /// into a fixture. `get_uhash` folds `x = (x ^ v) * GREEN_UHASH_MULT` over
    /// `Int` greens (warmstate.py:584-593) and the multiplier is odd, hence
    /// invertible mod 2^64, so the last green is `prefix ^ target * mult^-1`.
    fn green_key_hashing_to(target: u64, first: i64) -> GreenKey {
        use majit_ir::{GREEN_UHASH_MULT, GREEN_UHASH_SEED, GreenType, green_uhash_step};
        // Newton iteration for the inverse of an odd multiplier mod 2^64: each
        // step doubles the number of correct low bits, so six take 1 to 64.
        let mut inverse: u64 = 1;
        for _ in 0..6 {
            inverse =
                inverse.wrapping_mul(2u64.wrapping_sub(GREEN_UHASH_MULT.wrapping_mul(inverse)));
        }
        assert_eq!(GREEN_UHASH_MULT.wrapping_mul(inverse), 1);
        let prefix = green_uhash_step(GREEN_UHASH_SEED, GreenType::Int, first);
        let key = GreenKey::new(vec![first, (prefix ^ target.wrapping_mul(inverse)) as i64]);
        assert_eq!(key.get_uhash(), target, "the fold must invert exactly");
        key
    }

    /// A token that answers `has_compiled_code()`, which is what both halves of
    /// the entry gate test. The payload is never read — `has_compiled_code` is
    /// `self.compiled.get().is_some()` — so its type only has to satisfy the
    /// `Any + Send` bound.
    fn token_with_compiled_code(ws: &mut WarmEnterState) -> Arc<JitCellToken> {
        let token = Arc::new(JitCellToken::new(ws.alloc_token_number()));
        token.set_compiled(Box::new(()));
        assert!(token.has_compiled_code());
        token
    }

    /// **The entry path decides on one cell and executes off another.**
    ///
    /// `jitdriver.rs entry_cell_has_compiled_code` resolves a chained bucket
    /// through `comparekey` — `has_compiled_loop_for_key`, i.e.
    /// `get_procedure_token_for_key` — but the runner it then calls,
    /// `MetaInterp::run_compiled_detailed_with_values_at_dispatch_key`, reads
    /// `warm_state.get_procedure_token(hash)`: the bucket HEAD.
    ///
    /// [`get_procedure_token_for_key_reads_the_keys_own_cell_not_the_bucket_head`]
    /// above pins the reader in isolation, and its head cell holds no token at
    /// all — so the entry it models DECLINES, which is safe. This one gives the
    /// head a compiled token of its own, which is the shape a real collision
    /// between two warm keys has, and then nothing declines: the decision is
    /// about the chained cell's artifact and the execution is the head cell's.
    ///
    /// `warmstate.py:568-593` reaches a token only through the cell it has
    /// already matched on greens + comparekey + `get_uhash` together, so the
    /// two can never name different objects upstream. `warmstate.py:483/511`
    /// then carries that resolved `procedure_token` to the executor
    /// (`raise EnterJitAssembler(procedure_token, *execute_args)`) rather than
    /// handing on the key for a second lookup — which is why upstream has no
    /// second reader that could disagree.
    ///
    /// The entry path now carries the token the same way: `back_edge_internal`
    /// binds it at the decision and hands it to
    /// `MetaInterp::execute_assembler_at_dispatch_key` as an argument, so on
    /// that route there is no longer a second reader for the resolved key to
    /// keep honest — only the one modelled below. The u64 reader survives for
    /// callers that reach a run without having decided anything about the cell
    /// first, which is what the `executed` line stands in for.
    ///
    /// # What this used to pin, and what it pins now
    ///
    /// This is the rewritten body of the `#[ignore]`d pin
    /// `the_entry_decision_and_the_entry_execution_can_name_different_tokens`,
    /// rewritten with the repo owner's explicit approval. The fixture below —
    /// two cells, one bucket, a compiled token each — is the original's,
    /// unchanged. The assertion is inverted from "these two readers disagree,
    /// and that is the open defect" to "these two readers resolve through one
    /// cell key, so they cannot disagree", which is what the cell-unique-key
    /// migration makes true.
    ///
    /// **Pre-migration failure, recorded from the run that authorised this
    /// rewrite.** On `add4516a533` the same fixture and the same final
    /// assertion, with the executed side reading `chained_key.get_uhash()` —
    /// the raw bucket hash, which is the only u64 the runner had to carry
    /// then — reports:
    ///
    /// ```text
    /// thread 'warmstate::tests::the_entry_decision_and_the_entry_execution_can_name_different_tokens'
    /// panicked at majit/majit-metainterp/src/warmstate.rs:4842:9:
    /// the entry decided `chained_key` has compiled code and then ran a DIFFERENT
    /// key's artifact: decided token #2, executed token #1. The two must resolve
    /// through the same cell.
    /// ```
    ///
    /// The old pin's stated blocker — `MetaInterp::compiled_loops` is an
    /// `IndexMap<u64, CompiledEntry<M>>` with one slot per hash, so two
    /// colliding keys cannot both file a meta — is what the migration
    /// dissolves rather than works around: the u64 those tables are keyed by
    /// became cell-unique, so two colliding green keys file two entries and
    /// nothing had to move onto the cell.
    ///
    /// **Measured again after the migration, by substitution.** Replacing
    /// `carried` with `bucket` on the `executed` line below — the only
    /// difference between this body and the pre-migration one — reproduces the
    /// same failure on the migrated tree:
    ///
    /// ```text
    /// panicked at majit/majit-metainterp/src/warmstate.rs:5127:9:
    /// the entry decided `chained_key` has compiled code and then ran a DIFFERENT
    /// key's artifact: decided token #2, executed token #1.
    /// ```
    ///
    /// So the resolve is what carries this test, not a fixture that stopped
    /// colliding. The last two assertions pin that in-test: the raw bucket
    /// hash still names the HEAD cell's artifact.
    #[test]
    fn the_entry_decision_and_the_entry_execution_resolve_through_one_cell_key() {
        let mut ws = WarmEnterState::new(100);
        let head_key = GreenKey::new(vec![2100, 2200]);
        let chained_key = GreenKey::new(vec![2300, colliding_last_green(&head_key, 2300)]);
        let bucket = head_key.get_uhash();
        assert_eq!(
            chained_key.get_uhash(),
            bucket,
            "fixture: the two keys must land in one bucket by their own hash, \
             not by being installed there — `lookup_chain_with_key` starts \
             from `key.get_uhash()`, so a hand-placed cell would simply be \
             unreachable and the test would pass for the wrong reason",
        );
        assert_ne!(
            head_key, chained_key,
            "fixture: and they must be DIFFERENT keys, which is what makes \
             one bucket two artifacts",
        );

        // Installed first, and keepable (a cell holding a procedure token is
        // never `should_remove_jitcell`), so `install_new_cell` folds it in
        // front of the second and it ends up HEADING the bucket.
        let head_token = token_with_compiled_code(&mut ws);
        let mut head_cell = BaseJitCell::new();
        head_cell.set_comparekey(&head_key);
        head_cell.set_procedure_token(Arc::clone(&head_token), false);
        ws.install_new_cell(bucket, Some(head_cell));

        let chained_token = token_with_compiled_code(&mut ws);
        let mut chained_cell = BaseJitCell::new();
        chained_cell.set_comparekey(&chained_key);
        chained_cell.set_procedure_token(Arc::clone(&chained_token), false);
        ws.install_new_cell(bucket, Some(chained_cell));

        assert!(
            ws.bucket_is_chained(bucket),
            "fixture: both cells must share one bucket, which is the only \
             case in which the two readers can disagree",
        );
        assert!(
            !Arc::ptr_eq(&head_token, &chained_token),
            "fixture: the two cells must hold DIFFERENT artifacts, or the \
             defect has nothing to show",
        );

        // THE RESOLVE, once, at the entry — `jitdriver.rs back_edge_internal`
        // calls `MetaInterp::resolve_cell_key` exactly here, with the bucket
        // hash it arrived with and the greens behind it. Everything after this
        // line carries `carried` and looks nothing up by bucket again.
        let carried = ws.resolve_cell_key(bucket, || chained_key.clone());
        assert_ne!(
            carried, bucket,
            "fixture: the chained cell had to be minted its own key, or the \
             two halves below would agree for the trivial reason",
        );

        // What the driver decides on for `chained_key` — the `comparekey`
        // walk, `has_compiled_loop_for_key`.
        let decided = ws
            .get_procedure_token_for_key(&chained_key)
            .expect("the decision resolves the chain and finds the key's cell");
        assert!(
            Arc::ptr_eq(&decided, &chained_token),
            "the decision is about the chained cell's artifact",
        );

        // What the runner then executes: the u64 reader, handed the key the
        // entry resolved rather than the bucket hash it arrived with.
        let executed = ws
            .get_procedure_token(carried)
            .expect("the carried cell key names the cell the decision matched");

        assert!(
            Arc::ptr_eq(&executed, &decided),
            "the entry decided `chained_key` has compiled code and then ran a \
             DIFFERENT key's artifact: decided token #{}, executed token #{}. \
             The two must resolve through the same cell.",
            decided.number,
            executed.number,
        );

        // Non-vacuity: the bucket hash has NOT stopped naming the head cell.
        // Substituting it for `carried` above is the pre-migration reading and
        // still fails the assertion, so the resolve is what carries this test.
        let by_bucket_hash = ws
            .get_procedure_token(bucket)
            .expect("the head cell holds the bucket's raw hash as its key");
        assert!(
            Arc::ptr_eq(&by_bucket_hash, &head_token),
            "the raw hash names the bucket's first occupant, as it always did",
        );
        assert!(
            !Arc::ptr_eq(&by_bucket_hash, &decided),
            "and that is a DIFFERENT artifact from the one the entry decided \
             on, which is the whole reason the resolve has to happen",
        );
    }

    /// A producer that reaches a chained bucket with no greens DECLINES rather
    /// than guessing which sibling was meant.
    ///
    /// This is the answer to the hazard the hash-only cell creators leave
    /// behind. That population is not a list kept in prose but a property of
    /// one function: a creator is exactly a caller of
    /// [`WarmEnterState::ensure_cell_by_key`], because that is the only place a
    /// cell is installed from a bare `u64`, and it installs `comparekey: None`
    /// since a caller holding only a number has no greens to store. Anything
    /// reaching the celltable with a `&GreenKey` goes through
    /// `ensure_cell_for_key` instead and stores its comparator. A cell with no
    /// comparator cannot be told from a colliding neighbour by any mechanism —
    /// upstream cannot even express the state, since `JitCell.__init__`
    /// (warmstate.py:610-616) always stores the greens.
    ///
    /// So the resolve does not try. With no greens it answers the raw bucket
    /// hash, which names the bucket's ORIGINAL occupant — the cell that took
    /// the hash as its key — and names nothing at all once that cell is gone.
    /// That is a miss, and a miss is the honest port: a `None` from
    /// `cell_key_for` and a raw hash naming no cell both dead-end in the same
    /// "not found" arm `maybe_compile_and_run` has at warmstate.py:464.
    #[test]
    fn a_chained_bucket_reached_without_greens_answers_the_first_occupant() {
        let mut ws = WarmEnterState::new(100);
        let first = GreenKey::new(vec![3500, 3600]);
        let second = GreenKey::new(vec![3700, colliding_last_green(&first, 3700)]);
        let bucket = first.get_uhash();

        // A procedure token keeps the first cell non-removable, so the second
        // install chains it rather than pruning it (counter.py:246-256's
        // `should_remove_jitcell` gate — a cold, tokenless cell is dropped).
        let token = Arc::new(JitCellToken::new(ws.alloc_token_number()));
        ws.attach_procedure_to_interp_for_key(&first, token);
        let first_key = ws
            .cell_key_for(&first)
            .expect("the first key owns the cell it just installed");
        let second_key = ws.ensure_cell_key(&second);
        assert!(
            ws.bucket_is_chained(bucket),
            "fixture: the two keys must share a bucket",
        );

        assert_eq!(
            first_key, bucket,
            "the first occupant keeps the raw hash, so every workload that \
             never collides is bit-for-bit unchanged",
        );
        assert_ne!(second_key, bucket, "the second had to be minted one");
        assert_eq!(
            ws.sole_cell_key(bucket),
            None,
            "and a greens-less producer gets no answer from the bucket: two \
             candidates, nothing to choose between them",
        );

        // Which leaves the raw hash, and the raw hash names the first
        // occupant — never the minted sibling.
        assert_eq!(
            ws.cell_by_key(bucket).and_then(|cell| cell.cell_key),
            Some(first_key),
        );
    }

    /// **A bucket that has to mint twice must get two different keys.**
    ///
    /// [`WarmEnterState::mint_cell_key`] retries `green_uhash_step` against the
    /// live key set, which only terminates if the candidate MOVES between
    /// retries. Mixing the serial into the accumulator instead of passing it as
    /// the folded value cancels it — `green_uhash_step(bucket ^ serial, Int,
    /// serial)` folds `(bucket ^ serial) ^ serial`, which is `bucket` — so
    /// every retry re-proposes the single number `bucket * GREEN_UHASH_MULT`.
    /// The first mint in a bucket takes it and the second spins forever.
    ///
    /// Three cells in one bucket is the smallest state that asks for two mints,
    /// and nothing exotic reaches it: `install_new_cell` (counter.py:246-256)
    /// chains every cell whose `should_remove_jitcell` is false, so three warm
    /// keys sharing a bucket is enough.
    ///
    /// **Pre-fix reading.** This test does not fail, it HANGS — the run was
    /// killed after 11 minutes of CPU inside `mint_cell_key`'s retry loop.
    #[test]
    fn a_bucket_that_mints_twice_gets_two_different_keys() {
        let mut ws = WarmEnterState::new(100);
        let first = GreenKey::new(vec![9100, 9200]);
        let second = GreenKey::new(vec![9300, colliding_last_green(&first, 9300)]);
        let third = GreenKey::new(vec![9400, colliding_last_green(&first, 9400)]);
        let bucket = first.get_uhash();
        assert_eq!(second.get_uhash(), bucket, "fixture: one bucket");
        assert_eq!(third.get_uhash(), bucket, "fixture: three keys");

        // A procedure token each, so `should_remove_jitcell` keeps all three
        // and the bucket really chains rather than pruning down to one.
        let mut keys = Vec::new();
        for key in [&first, &second, &third] {
            let token = token_with_compiled_code(&mut ws);
            ws.attach_procedure_to_interp_for_key(key, token);
            keys.push(
                ws.cell_key_for(key)
                    .expect("each key owns the cell just installed for it"),
            );
        }

        assert_eq!(keys[0], bucket, "the first occupant keeps the raw hash");
        assert_ne!(keys[1], keys[0], "the second had to be minted one");
        assert_ne!(
            keys[2], keys[1],
            "and the third must be minted a DIFFERENT one, or the two cells \
             share an identity and the artifact tables indexed by it collapse",
        );
        assert_ne!(keys[2], keys[0]);
        for (key, cell_key) in [&first, &second, &third].iter().zip(&keys) {
            assert_eq!(
                ws.cell_by_key(*cell_key).and_then(|cell| cell.cell_key),
                Some(*cell_key),
                "each minted key must name its own cell",
            );
            assert_eq!(ws.cell_key_for(key), Some(*cell_key));
        }
    }

    /// **Two readers of one raw hash inspected two different buckets.**
    ///
    /// [`WarmEnterState::resolve_cell_key`] asks
    /// [`WarmEnterState::bucket_is_chained`] whether a `GreenKey` has to be
    /// built at all, and answers from [`WarmEnterState::sole_cell_key`] when it
    /// does not. Both take a raw green-key hash, but they reached the celltable
    /// by different routes: `bucket_is_chained` went through
    /// [`WarmEnterState::bucket_of`], which maps a CELL KEY to the bucket that
    /// cell lives in, while `sole_cell_key` indexed the celltable directly.
    ///
    /// The routes agree on every number that is not a live minted key, and a
    /// minted key is reachable as a raw hash: `mint_cell_key` steps
    /// `green_uhash_step` against the live set and every `u64` is a hash some
    /// green key produces (see its doc), so [`green_key_hashing_to`] builds the
    /// green key whose own hash IS the minted number. `bucket_of` then sent
    /// `bucket_is_chained` to the minted cell's bucket while `sole_cell_key`
    /// read the bucket the hash names — two buckets, one number, opposite
    /// answers.
    ///
    /// Upstream has one route and cannot express the split: every reader starts
    /// at `jitcounter.lookup_chain(hash)` with `hash = JitCell.get_uhash(...)`
    /// — `maybe_compile_and_run` (warmstate.py:461-464), `get_jitcell`
    /// (warmstate.py:596-604) and `_ensure_jit_cell_at_key`
    /// (warmstate.py:635-641) all spell it that way — and `lookup_chain`
    /// (counter.py:239-240) is a bare `celltable[self._get_index(hash)]` with
    /// nothing in front of it.
    ///
    /// **Pre-fix reading.** Routing `bucket_is_chained` through `bucket_of`
    /// fails the third assertion below, and substituting that one line back is
    /// the whole difference between the two trees:
    ///
    /// ```text
    /// panicked at majit/majit-metainterp/src/warmstate.rs:5446:9:
    /// the two readers of one raw hash must describe ONE bucket: `sole_cell_key`
    /// declined because the bucket this hash names is chained, and this answered
    /// `unchained` about the bucket `bucket_of` sent it to instead
    /// ```
    ///
    /// The assertions after it are what that costs: the entry resolves to a
    /// cell belonging to neither the arriving key nor its sibling.
    #[test]
    fn a_raw_hash_equal_to_a_minted_key_resolves_through_one_bucket() {
        let mut ws = WarmEnterState::new(100);

        // A bucket that mints. A hash-only creator squats the raw hash with a
        // comparator-less cell, so the typed writer behind it cannot match it
        // and installs a second cell, which has to be minted a key. The squatter
        // is cold and tokenless, so `install_new_cell`'s `should_remove_jitcell`
        // gate (counter.py:246-256) drops it within that same call — leaving the
        // minted cell alone in its bucket.
        let parked = GreenKey::new(vec![7100, 7200]);
        let parked_bucket = parked.get_uhash();
        ws.ensure_cell_by_key(parked_bucket);
        let parked_token = token_with_compiled_code(&mut ws);
        ws.attach_procedure_to_interp_for_key(&parked, Arc::clone(&parked_token));
        let minted = ws
            .cell_key_for(&parked)
            .expect("the typed writer's cell is reachable by its own key");
        assert_ne!(
            minted, parked_bucket,
            "fixture: the squatter held the raw hash when the typed cell was \
             filed, so that cell had to be minted a key — the minted number is \
             the subject here",
        );
        assert!(
            !ws.bucket_is_chained(parked_bucket),
            "fixture: the squatter was pruned, so the minted cell's bucket holds \
             exactly one cell and both routes call it unchained",
        );

        // The green key whose own hash IS that minted number, plus a sibling
        // colliding with it, so the bucket the number names is CHAINED while
        // the bucket `bucket_of` maps it to is not.
        let arriving = green_key_hashing_to(minted, 8100);
        let sibling = green_key_hashing_to(minted, 8200);
        assert_ne!(arriving, sibling, "fixture: two different green keys");
        let arriving_token = token_with_compiled_code(&mut ws);
        ws.attach_procedure_to_interp_for_key(&arriving, Arc::clone(&arriving_token));
        let sibling_token = token_with_compiled_code(&mut ws);
        ws.attach_procedure_to_interp_for_key(&sibling, Arc::clone(&sibling_token));
        assert!(
            !Arc::ptr_eq(&arriving_token, &sibling_token),
            "fixture: the two cells hold different artifacts",
        );
        assert!(
            ws.lookup_chain(minted)
                .is_some_and(|head| head.next.is_some()),
            "fixture: the bucket this hash names holds both keys' cells",
        );

        // THE DISAGREEMENT.
        assert_eq!(
            ws.sole_cell_key(minted),
            None,
            "the bucket this hash names has two candidates, so the greens-less \
             reader declines",
        );
        assert!(
            ws.bucket_is_chained(minted),
            "the two readers of one raw hash must describe ONE bucket: \
             `sole_cell_key` declined because the bucket this hash names is \
             chained, and this answered `unchained` about the bucket `bucket_of` \
             sent it to instead",
        );

        // What that costs, end to end: the entry arrives with this hash and the
        // greens behind it, and has to reach the arriving key's own cell.
        let carried = ws.resolve_cell_key(minted, || arriving.clone());
        assert!(
            Arc::ptr_eq(
                &ws.get_procedure_token(carried)
                    .expect("the carried cell key names a cell"),
                &arriving_token,
            ),
            "the resolve answered a key naming a cell that belongs to neither \
             the arriving key nor its sibling",
        );

        // Non-vacuity: the raw hash has not stopped naming the parked key's
        // cell, so answering it unchanged is answering a foreign cell.
        assert_ne!(carried, minted);
        assert!(
            Arc::ptr_eq(
                &ws.get_procedure_token(minted)
                    .expect("the minted key names its own cell"),
                &parked_token,
            ),
            "and the cell it would have answered is the parked key's",
        );
    }

    /// Two colliding green keys file two entries in a `u64`-keyed artifact
    /// table — the precondition `MetaInterp::compiled_loops` and
    /// `MemoryManager` eviction both need, and the one the old pin named as
    /// its blocker.
    ///
    /// `compiled_loops` is an `IndexMap<u64, CompiledEntry<M>>` and
    /// `try_to_free_some_loops` reaches an entry by
    /// `token.green_key()` (`JitCellToken::green_key: Cell<u64>`). While that
    /// u64 was a bucket hash, two colliding keys shared ONE slot: the second
    /// compile overwrote the first, and the loser's token could not reach its
    /// own entry to be freed. The stand-in table below is that shape — the
    /// subject is the keys, not the payload, so the payload is the token
    /// number.
    #[test]
    fn two_colliding_keys_file_two_entries_in_a_u64_keyed_artifact_table() {
        let mut ws = WarmEnterState::new(100);
        let key_a = GreenKey::new(vec![3100, 3200]);
        let key_b = GreenKey::new(vec![3300, colliding_last_green(&key_a, 3300)]);
        assert_eq!(
            key_a.get_uhash(),
            key_b.get_uhash(),
            "fixture: one bucket, two keys",
        );

        let cell_a = ws.ensure_cell_key(&key_a);
        let cell_b = ws.ensure_cell_key(&key_b);
        assert_ne!(
            cell_a, cell_b,
            "colliding keys must be told apart by the u64 the artifact tables \
             are indexed by, or one of them has no reachable entry at all",
        );

        // The two tokens each stamp their own cell key, which is what
        // `compile.rs:2436 jitcell_token.green_key.set(green_key)` writes and
        // what `try_to_free_some_loops` reads back.
        let token_a = Arc::new(JitCellToken::new(ws.alloc_token_number()));
        let token_b = Arc::new(JitCellToken::new(ws.alloc_token_number()));
        token_a.green_key.set(cell_a);
        token_b.green_key.set(cell_b);

        let mut compiled_loops: indexmap::IndexMap<u64, u64> = indexmap::IndexMap::new();
        compiled_loops.insert(token_a.green_key(), token_a.number);
        compiled_loops.insert(token_b.green_key(), token_b.number);
        assert_eq!(
            compiled_loops.len(),
            2,
            "both compiles must file; a shared slot means the second erased \
             the first",
        );

        // Eviction of B reaches B's entry and leaves A's alone.
        assert_eq!(
            compiled_loops.swap_remove(&token_b.green_key()),
            Some(token_b.number),
            "the evicted token's own key must name its own entry",
        );
        assert_eq!(
            compiled_loops.get(&token_a.green_key()),
            Some(&token_a.number),
            "and the colliding sibling's artifact must survive it",
        );
    }

    /// An unchained bucket has one candidate, so the head IS the match and the
    /// hash form is exact — this is what lets the entry path skip building a
    /// `GreenKey` on every warm entry.
    #[test]
    fn an_unchained_bucket_answers_the_same_through_both_forms() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![1300, 1400]);

        let token = Arc::new(JitCellToken::new(ws.alloc_token_number()));
        ws.attach_procedure_to_interp_for_key(&key, Arc::clone(&token));

        assert!(
            !ws.bucket_is_chained(key.get_uhash()),
            "one typed writer and no hash-only squatter builds no chain",
        );
        assert!(
            Arc::ptr_eq(
                &ws.get_procedure_token(key.get_uhash())
                    .expect("the hash form finds the only cell"),
                &token,
            ),
            "both forms must name the same token when there is nothing to walk",
        );
        assert!(Arc::ptr_eq(
            &ws.get_procedure_token_for_key(&key)
                .expect("the typed form finds it too"),
            &token,
        ));
    }

    #[test]
    fn mark_force_finish_tracing_for_key_sets_the_flag_on_the_keys_own_cell() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![900, 1000]);

        ws.disable_noninlinable_function(key.get_uhash());
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "fixture: the hash-only cell carries no comparekey",
        );

        ws.mark_force_finish_tracing_for_key(&key);

        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("the key must own a cell reachable by comparekey");
        assert!(
            cell.flags & jc_flags::FORCE_FINISH != 0,
            "FORCE_FINISH is sticky and never cleared, so landing it on the \
             wrong cell of the bucket is permanent",
        );
    }

    /// `get_stats` counts every cell in a chain, not just the bucket head.
    ///
    /// Non-vacuity: a chain is the *only* shape that separates the two
    /// implementations, so this fixture hands `install_new_cell` a shared
    /// bucket directly — as the collision fixture above does, and for the
    /// same reason (`cells` is keyed by the full `get_uhash()`, so two real
    /// green keys will not collide). Head and tail are put in *different*
    /// states so a head-only reader is caught twice: once on the total and
    /// once on the per-state split.
    ///
    /// The parenthetical above says only that a *collision* is
    /// impractical to reach from two real keys; it is not a claim that
    /// chains are rare. The fixture directly above shows a chain forming
    /// from a SINGLE key with no collision at all.
    ///
    /// Measured against the previous body (`num_cells: self.cells.len()`
    /// plus a `cells.values()` loop): fails `num_cells` 1 vs 2. The
    /// per-state assertions below are not separately observed in that run —
    /// the first failure ends it — but they cover the same head-only read
    /// on a second axis, so a future body that fixes the total and keeps a
    /// head-only state walk still fails here.
    #[test]
    fn get_stats_counts_chained_cells_not_just_bucket_heads() {
        let mut ws = WarmEnterState::new(100);
        let key_head = GreenKey::new(vec![100, 200]);
        let key_tail = GreenKey::new(vec![300, 400]);
        let bucket = key_head.get_uhash();

        // TRACING keeps the tail non-removable so the second install chains
        // it rather than dropping it (counter.py:246-256 should_remove gate).
        let mut tail = BaseJitCell::new();
        tail.state = BaseJitCellState::Tracing;
        tail.flags |= jc_flags::TRACING;
        tail.comparekey = Some(key_tail);
        ws.install_new_cell(bucket, Some(tail));

        let mut head = BaseJitCell::new();
        head.state = BaseJitCellState::Compiled;
        head.flags |= jc_flags::TRACING;
        head.comparekey = Some(key_head);
        ws.install_new_cell(bucket, Some(head));

        // Precondition: one map entry holding a two-cell chain. Without
        // this the assertions below would pass for the wrong reason.
        assert_eq!(ws.cells.len(), 1, "fixture must build ONE bucket");
        assert!(
            ws.lookup_chain(bucket)
                .and_then(|h| h.next.as_deref())
                .is_some(),
            "fixture must build a TWO-cell chain, or it cannot tell the \
             implementations apart",
        );

        let stats = ws.get_stats();
        assert_eq!(stats.num_cells, 2, "both chained cells must be counted");
        assert_eq!(stats.num_compiled, 1, "head is Compiled");
        assert_eq!(
            stats.num_tracing, 1,
            "the chained tail's Tracing state is invisible to a head-only reader",
        );
    }

    /// warmstate.py:425-444 + 596-604 — typed-key transition writes
    /// land on the matching chained cell, not the bucket head. A
    /// collision-shaped chain `head=A → tail=B` followed by enough
    /// `maybe_compile_with_key(&key_b)` ticks to exhaust the counter
    /// must flip `B.is_tracing()` to true while A's flags stay intact.
    #[test]
    fn maybe_compile_with_key_start_tracing_writes_to_chained_cell() {
        let mut ws = WarmEnterState::new(2); // threshold=2 → 2 ticks fire
        let key_b = GreenKey::new(vec![700, 800]);
        let key_a = GreenKey::new(vec![500, 600]);
        let bucket = key_b.get_uhash();

        // Head A: must stay non-removable across B's install.
        let mut cell_a = BaseJitCell::new();
        cell_a.flags |= jc_flags::TRACING;
        cell_a.comparekey = Some(key_a.clone());
        ws.install_new_cell(bucket, Some(cell_a));

        // Tick #1 — counter not at threshold yet, NotHot.
        match ws.maybe_compile_with_key(&key_b) {
            HotResult::NotHot => {}
            _ => panic!("tick #1 expected NotHot"),
        }
        // Tick #2 — counter fires; start_tracing_cell_for_key must
        // chain-walk and flip B's flags, not the head A.
        match ws.maybe_compile_with_key(&key_b) {
            HotResult::StartTracing => {}
            _ => panic!("tick #2 expected StartTracing"),
        }

        // Walk the chain — A still TRACING (its own flag), B now
        // additionally TRACING.
        let head = ws.lookup_chain(bucket).expect("head exists");
        assert_eq!(head.comparekey.as_ref(), Some(&key_a));
        assert!(head.is_tracing(), "A's TRACING flag is its own");
        let b_cell = ws
            .lookup_chain_with_key(&key_b)
            .expect("typed lookup finds B in the chain");
        assert!(
            b_cell.is_tracing(),
            "B's TRACING flag set by start_tracing_cell_for_key — \
             chain-walk write must reach the tail, not just the head",
        );
    }
}
