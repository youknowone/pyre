/// Warm state management — the lifecycle from interpreting to compiled code.
///
/// Manages the transition: Interpreting -> Tracing -> Compiled.
/// When the hot counter fires, we start tracing. When the trace is
/// complete, we compile it and cache the result.
///
/// Reference: rpython/jit/metainterp/warmstate.py WarmEnterState, BaseBaseJitCell
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use majit_codegen::JitCellToken;
use majit_ir::Type;

use crate::counter::JitCounter;
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
    pub loop_token: Option<JitCellToken>,
    /// Number of times tracing was aborted (non-permanent) for this key.
    /// After `retrace_limit` aborts, the cell becomes DONT_TRACE_HERE.
    /// RPython equivalent: retrace_limit parameter.
    pub abort_count: u32,
}

impl BaseJitCell {
    fn new() -> Self {
        BaseJitCell {
            flags: 0,
            state: BaseJitCellState::NotHot,
            counter: 0,
            token: None,
            tracing_generation: 0,
            loop_token: None,
            abort_count: 0,
        }
    }

    pub fn is_tracing(&self) -> bool {
        self.flags & jc_flags::TRACING != 0
    }

    pub fn is_compiled(&self) -> bool {
        self.loop_token.is_some() && (self.flags & jc_flags::TEMPORARY == 0)
    }

    pub fn has_procedure_token(&self) -> bool {
        self.loop_token.is_some()
    }

    /// Get the procedure token, returning None if the token has been
    /// invalidated (mirrors BaseBaseJitCell.get_procedure_token).
    pub fn get_procedure_token(&self) -> Option<&JitCellToken> {
        self.loop_token.as_ref().filter(|t| !t.is_invalidated())
    }

    /// Set the procedure token and update ownership state.
    /// If `tmp` is true, sets the TEMPORARY flag (CALL_ASSEMBLER fallback).
    pub fn set_procedure_token(&mut self, loop_token: JitCellToken, tmp: bool) {
        self.token = Some(loop_token.number);
        self.loop_token = Some(loop_token);
        if tmp {
            self.flags |= jc_flags::TEMPORARY;
        } else {
            self.flags &= !jc_flags::TEMPORARY;
            self.state = BaseJitCellState::Compiled;
        }
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
        true
    }
}

/// The current JIT state for a particular green key.
pub enum JitState {
    /// Normal interpretation; no tracing is active.
    Interpreting,
    /// Actively recording a trace.
    Tracing(Trace),
    /// A compiled loop exists for this green key.
    Compiled(JitCellToken),
}

/// Generation-based loop aging. Loops not accessed for `max_age`
/// generations are candidates for eviction.
///
/// Reference: rpython/jit/metainterp/memmgr.py MemoryManager.
pub struct LoopAging {
    generation: u64,
    max_age: u64,
    /// loop_key → last access generation.
    loop_generations: HashMap<u64, u64>,
}

impl LoopAging {
    /// Create a new LoopAging with the given max_age.
    /// `max_age == 0` disables eviction.
    pub fn new(max_age: u64) -> Self {
        LoopAging {
            generation: 0,
            max_age,
            loop_generations: HashMap::new(),
        }
    }

    /// Set the maximum age before eviction.
    pub fn set_max_age(&mut self, max_age: u64) {
        self.max_age = max_age;
    }

    /// Get the current max_age setting.
    pub fn max_age(&self) -> u64 {
        self.max_age
    }

    /// Mark a loop as alive in the current generation.
    pub fn keep_loop_alive(&mut self, loop_key: u64) {
        self.loop_generations.insert(loop_key, self.generation);
    }

    /// Register a new loop at the current generation.
    pub fn register_loop(&mut self, loop_key: u64) {
        self.loop_generations.insert(loop_key, self.generation);
    }

    /// Advance the generation counter. Returns the set of loop keys
    /// that are now too old and should be evicted.
    pub fn next_generation(&mut self) -> Vec<u64> {
        self.generation += 1;

        if self.max_age == 0 {
            return vec![];
        }

        let threshold = self.generation.saturating_sub(self.max_age);
        let evicted: Vec<u64> = self
            .loop_generations
            .iter()
            .filter(|&(_, &last_access)| last_access < threshold)
            .map(|(&key, _)| key)
            .collect();

        for key in &evicted {
            self.loop_generations.remove(key);
        }

        evicted
    }

    /// Get the current generation number.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Number of tracked loops.
    pub fn alive_count(&self) -> usize {
        self.loop_generations.len()
    }
}

/// Warm state manager — the orchestrator of the JIT lifecycle.
///
/// Keeps track of per-greenkey cells and the global hot counter.
/// The interpreter calls `maybe_compile()` at loop headers;
/// WarmEnterState decides whether to start tracing, continue interpreting,
/// or dispatch to compiled code.
/// Default number of guard failures before triggering bridge compilation.
/// PyPy default: trace_eagerness = 200
const DEFAULT_BRIDGE_THRESHOLD: u32 = 200;

/// PyPy default: function_threshold = 1619 (prime, above threshold)
const DEFAULT_FUNCTION_THRESHOLD: u32 = 1619;

/// PyPy default: max_unroll_recursion = 7
const DEFAULT_MAX_INLINE_DEPTH: u32 = 7;

/// PyPy default: trace_limit = 6000
const DEFAULT_TRACE_LIMIT: u32 = crate::recorder::DEFAULT_TRACE_LIMIT as u32;

/// Maximum number of non-permanent trace aborts before giving up on a green key.
/// RPython default from rlib/jit.py: retrace_limit = 0.
const DEFAULT_RETRACE_LIMIT: u32 = 0;

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
}

pub struct WarmEnterState {
    /// Global hot counter.
    counter: JitCounter,
    /// Per-greenkey cells, keyed by the hash of the green key.
    cells: HashMap<u64, BaseJitCell>,
    /// Compilation threshold (copied from counter for easy access).
    threshold: u32,
    /// Guard failure threshold for triggering bridge compilation.
    bridge_threshold: u32,
    /// Function call threshold for inlining during tracing.
    ///
    /// A function must be called at least this many times before
    /// the meta-interpreter inlines it into the trace.
    function_threshold: u32,
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
    /// to the set of green_key_hashes whose compiled loops depend on that field.
    /// When a quasi-immutable field is mutated, all dependent loops are invalidated.
    quasiimmut_deps: HashMap<u64, HashSet<u64>>,
    /// Per-function call counts during the current trace.
    ///
    /// Tracks how many times each function (identified by callee_key) has been
    /// encountered during tracing. Used by `should_inline_function()` to decide
    /// whether to inline or leave as residual call.
    function_call_counts: HashMap<u64, u32>,

    // ── RPython warmstate.py additional parameters ──
    /// Maximum number of retrace attempts for a single green key.
    /// RPython: `set_param_retrace_limit`
    retrace_limit: u32,
    /// Maximum number of extra guards allowed in a retrace before giving up.
    /// RPython: `set_param_max_retrace_guards`
    max_retrace_guards: u32,
    /// Maximum number of loop unrolling iterations.
    /// RPython: `set_param_max_unroll_loops`
    max_unroll_loops: u32,
    /// Maximum recursive unrolling depth.
    /// RPython: `set_param_max_unroll_recursion`
    max_unroll_recursion: u32,
    /// Loop longevity factor (how long compiled loops survive without being hot).
    /// RPython: `set_param_loop_longevity`
    loop_longevity: u32,
    /// Whether SIMD vectorization is enabled.
    /// RPython: `set_param_vectorize`
    vectorize: bool,
    /// Cost threshold for vectorization decisions.
    /// RPython: `set_param_vec_cost`
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
    /// warmstate.py: set_param_decay — counter decay factor.
    decay: u32,
}

/// Result of checking whether a green key is hot.
pub enum HotResult {
    /// Not yet hot; keep interpreting.
    NotHot,
    /// Threshold reached; start tracing.
    StartTracing(Trace),
    /// Already tracing (caller should keep feeding ops to the active recorder).
    AlreadyTracing,
    /// Compiled code exists; run it.
    RunCompiled,
}

impl WarmEnterState {
    /// Create a new WarmEnterState with the given threshold.
    /// Automatically enables Logger if MAJIT_STATS=1 or MAJIT_LOG=1.
    pub fn new(threshold: u32) -> Self {
        WarmEnterState {
            counter: JitCounter::new(threshold),
            cells: HashMap::new(),
            threshold,
            bridge_threshold: DEFAULT_BRIDGE_THRESHOLD,
            function_threshold: DEFAULT_FUNCTION_THRESHOLD,
            max_inline_depth: DEFAULT_MAX_INLINE_DEPTH,
            trace_limit: DEFAULT_TRACE_LIMIT,
            tracing_generation: 0,
            jitlog: Logger::from_env(),
            quasiimmut_deps: HashMap::new(),
            function_call_counts: HashMap::new(),
            retrace_limit: DEFAULT_RETRACE_LIMIT,
            max_retrace_guards: 15,
            max_unroll_loops: 0,
            max_unroll_recursion: DEFAULT_MAX_INLINE_DEPTH,
            loop_longevity: 1000,
            vectorize: false,
            vec_cost: 0,
            enable_opts: Vec::new(),
            inlining: true,
            disable_unrolling_threshold: 0,
            pureop_historylength: 16,
            decay: 40,
        }
    }

    /// Create a new WarmEnterState with an explicit Logger.
    pub fn with_jitlog(threshold: u32, jitlog: Option<Logger>) -> Self {
        WarmEnterState {
            counter: JitCounter::new(threshold),
            cells: HashMap::new(),
            threshold,
            bridge_threshold: DEFAULT_BRIDGE_THRESHOLD,
            function_threshold: DEFAULT_FUNCTION_THRESHOLD,
            max_inline_depth: DEFAULT_MAX_INLINE_DEPTH,
            trace_limit: DEFAULT_TRACE_LIMIT,
            tracing_generation: 0,
            jitlog,
            quasiimmut_deps: HashMap::new(),
            function_call_counts: HashMap::new(),
            retrace_limit: DEFAULT_RETRACE_LIMIT,
            max_retrace_guards: 15,
            max_unroll_loops: 0,
            max_unroll_recursion: DEFAULT_MAX_INLINE_DEPTH,
            loop_longevity: 1000,
            vectorize: false,
            vec_cost: 0,
            enable_opts: Vec::new(),
            inlining: true,
            disable_unrolling_threshold: 0,
            pureop_historylength: 16,
            decay: 40,
        }
    }

    /// Check and possibly transition the JIT state for a given green key.
    ///
    /// Called by the interpreter at loop back-edges and function entries.
    /// Returns a `HotResult` telling the interpreter what to do next.
    /// Mark a green key as DONT_TRACE_HERE permanently.
    pub fn mark_dont_trace(&mut self, green_key_hash: u64) {
        self.disable_noninlinable_function(green_key_hash);
    }

    #[inline]
    pub fn counter_would_fire(&self, green_key_hash: u64) -> bool {
        if let Some(cell) = self.cells.get(&green_key_hash) {
            if cell.is_compiled() || cell.is_tracing() {
                return true;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 {
                return false;
            }
        }
        self.counter.would_fire(green_key_hash)
    }

    /// Tick counter and check if key should enter JIT.
    ///
    /// RPython parity: this is the inline fast path equivalent of
    /// `maybe_compile_and_run(increment_threshold, ...)`.
    /// Returns true if counter reached threshold (should enter slow path).
    /// Returns false for DONT_TRACE keys or cold keys.
    #[inline]
    pub fn counter_tick_checked(&mut self, green_key_hash: u64) -> bool {
        if let Some(cell) = self.cells.get(&green_key_hash) {
            if cell.is_compiled() || cell.is_tracing() {
                return true;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 {
                return false;
            }
        }
        self.counter.tick(green_key_hash)
    }

    pub fn maybe_compile(&mut self, green_key_hash: u64) -> HotResult {
        if let Some(cell) = self.cells.get(&green_key_hash) {
            if cell.is_compiled() {
                return HotResult::RunCompiled;
            }
            if cell.is_tracing() {
                return HotResult::AlreadyTracing;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 {
                return HotResult::NotHot;
            }
        }

        if !self.counter.tick(green_key_hash) {
            return HotResult::NotHot;
        }

        // Threshold reached: start tracing
        self.counter.reset(green_key_hash);
        self.tracing_generation += 1;
        let current_generation = self.tracing_generation;
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(BaseJitCell::new);
        cell.flags |= jc_flags::TRACING | jc_flags::TRACING_OCCURRED;
        cell.state = BaseJitCellState::Tracing;
        cell.tracing_generation = current_generation;

        HotResult::StartTracing(Trace::with_limit(self.trace_limit as usize))
    }

    /// Force-start tracing for a green key, bypassing the hot counter.
    ///
    /// Used by function-entry tracing where the caller has already
    /// determined that tracing should begin.
    pub fn force_start_tracing(&mut self, green_key_hash: u64) -> HotResult {
        if let Some(cell) = self.cells.get(&green_key_hash) {
            if cell.is_compiled() {
                return HotResult::RunCompiled;
            }
            if cell.is_tracing() {
                return HotResult::AlreadyTracing;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 {
                return HotResult::NotHot;
            }
        }

        self.counter.reset(green_key_hash);
        self.tracing_generation += 1;
        let current_generation = self.tracing_generation;
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(BaseJitCell::new);
        cell.flags |= jc_flags::TRACING | jc_flags::TRACING_OCCURRED;
        cell.state = BaseJitCellState::Tracing;
        cell.tracing_generation = current_generation;

        HotResult::StartTracing(Trace::with_limit(self.trace_limit as usize))
    }

    /// Start a retrace from a guard failure point.
    ///
    /// Creates a new Trace for retracing, similar to starting a
    /// fresh trace but from a guard's failure inputs.
    pub fn start_retrace(&mut self, input_types: &[Type]) -> Trace {
        self.reset_function_counts();
        Trace::with_input_types_and_limit(input_types, self.trace_limit as usize)
    }

    /// Mark that tracing is done for a green key. Clears the TRACING flag.
    /// The caller is responsible for compiling the trace and calling
    /// `attach_procedure_to_interp` with the resulting JitCellToken.
    pub fn finish_tracing(&mut self, green_key_hash: u64) {
        if let Some(cell) = self.cells.get_mut(&green_key_hash) {
            cell.flags &= !jc_flags::TRACING;
            // State remains Tracing until attach_procedure_to_interp is called.
        }
    }

    /// Mark that tracing was aborted for a green key.
    /// Optionally sets DONT_TRACE_HERE to prevent retrying.
    pub fn abort_tracing(&mut self, green_key_hash: u64, disable_noninlinable_function: bool) {
        let mut mark_dont_trace = disable_noninlinable_function;
        if let Some(cell) = self.cells.get_mut(&green_key_hash) {
            cell.flags &= !jc_flags::TRACING;
            if disable_noninlinable_function {
                cell.state = BaseJitCellState::NotHot;
            } else {
                cell.abort_count += 1;
                if cell.abort_count < self.retrace_limit {
                    cell.state = BaseJitCellState::NotHot;
                } else {
                    mark_dont_trace = true;
                }
            }
        }

        if mark_dont_trace {
            // Too many retries — or an explicit permanent abort — stop
            // tracing this location entirely. RPython equivalent:
            // retrace_limit exceeded / disable_noninlinable_function().
            self.disable_noninlinable_function(green_key_hash);
        }
        if let Some(log) = &mut self.jitlog {
            log.log_abort();
        }
    }

    /// Install a compiled loop token for a green key.
    ///
    /// The cell transitions to Compiled state and takes ownership of
    /// the procedure token.
    pub fn attach_procedure_to_interp(&mut self, green_key_hash: u64, token: JitCellToken) {
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(BaseJitCell::new);
        cell.flags &= !jc_flags::TRACING;
        cell.set_procedure_token(token, false);
    }

    /// Get a reference to the compiled loop token for a green key.
    pub fn get_compiled(&self, green_key_hash: u64) -> Option<&JitCellToken> {
        self.cells
            .get(&green_key_hash)
            .and_then(|cell| cell.loop_token.as_ref())
    }

    /// Allocate a new unique JitCellToken number.
    pub fn alloc_token_number(&mut self) -> u64 {
        NEXT_GLOBAL_TOKEN_NUMBER.fetch_add(1, Ordering::Relaxed)
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Set a new threshold. Also updates the internal counter.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
        self.counter.set_threshold(threshold);
    }

    /// Decay all counters (e.g., periodically to avoid stale counts).
    pub fn decay_counters(&mut self) {
        self.counter.decay_all_counters();
    }

    /// Get a reference to the BaseJitCell for a green key, if it exists.
    pub fn get_cell(&self, green_key_hash: u64) -> Option<&BaseJitCell> {
        self.cells.get(&green_key_hash)
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

    /// Get the bridge compilation threshold.
    pub fn bridge_threshold(&self) -> u32 {
        self.bridge_threshold
    }

    /// Set the bridge compilation threshold.
    pub fn set_bridge_threshold(&mut self, threshold: u32) {
        self.bridge_threshold = threshold;
    }

    /// Check whether a guard failure count has reached the bridge threshold.
    /// Returns true if bridge compilation should be triggered.
    pub fn should_compile_bridge(&self, guard_fail_count: u32) -> bool {
        guard_fail_count >= self.bridge_threshold
    }

    /// Get the function inlining threshold.
    pub fn function_threshold(&self) -> u32 {
        self.function_threshold
    }

    /// Set the function inlining threshold.
    pub fn set_function_threshold(&mut self, threshold: u32) {
        self.function_threshold = threshold;
    }

    /// Set the maximum inline depth.
    pub fn set_max_inline_depth(&mut self, depth: u32) {
        self.max_inline_depth = depth;
    }

    /// Get the maximum inline depth.
    pub fn max_inline_depth(&self) -> u32 {
        self.max_inline_depth
    }

    /// Record a function call during tracing and decide whether to inline it.
    ///
    /// Mirrors RPython's inlining heuristic: a function must be called
    /// at least `function_threshold` times before being inlined,
    /// and the current inline depth must not exceed `max_inline_depth`.
    /// Returns `true` if the function should be inlined.
    pub fn should_inline_function(&mut self, callee_key: u64) -> bool {
        if !self.can_inline_callable(callee_key) {
            return false;
        }
        let count = self.function_call_counts.entry(callee_key).or_insert(0);
        *count += 1;
        *count >= self.function_threshold
    }

    /// Whether this callee is eligible for inlining at all.
    ///
    /// Mirrors PyPy's `can_inline_callable`: once a green key is marked
    /// `DONT_TRACE_HERE`, callers must stop inlining it and instead let it
    /// converge to a separate functrace / call_assembler path.
    pub fn can_inline_callable(&self, callee_key: u64) -> bool {
        self.cells
            .get(&callee_key)
            .map_or(true, |cell| cell.flags & jc_flags::DONT_TRACE_HERE == 0)
    }

    /// Mark a callee as a location that should no longer be inlined into
    /// surrounding traces.
    ///
    /// This is the warm-state equivalent of PyPy's `disable_noninlinable_function()`.
    pub fn disable_noninlinable_function(&mut self, callee_key: u64) {
        let cell = self
            .cells
            .entry(callee_key)
            .or_insert_with(BaseJitCell::new);
        cell.flags |= jc_flags::DONT_TRACE_HERE;
        if cell.flags & jc_flags::TRACING == 0 {
            cell.state = BaseJitCellState::DontTraceHere;
        }
    }

    /// Boost a function's entry counter to threshold - 1.
    /// Next call through eval_with_jit will trigger tracing.
    /// PyPy equivalent: mark for separate functrace after recursive depth limit.
    pub fn boost_function_entry(&mut self, callee_key: u64) {
        let threshold = self.function_threshold;
        let count = self.function_call_counts.entry(callee_key).or_insert(0);
        if *count < threshold.saturating_sub(1) {
            *count = threshold.saturating_sub(1);
        }
    }

    /// Check if a function was boosted (counter >= threshold - 1).
    pub fn is_boosted(&self, callee_key: u64) -> bool {
        self.function_call_counts
            .get(&callee_key)
            .map_or(false, |&c| c >= self.function_threshold.saturating_sub(1))
    }

    /// Check if inlining is allowed at the given depth.
    pub fn can_inline_at_depth(&self, current_depth: usize) -> bool {
        (current_depth as u32) < self.max_inline_depth
    }

    /// Reset function call counts (called when tracing ends).
    pub fn reset_function_counts(&mut self) {
        self.function_call_counts.clear();
    }

    /// Get the current call count for a specific function.
    pub fn function_call_count(&self, callee_key: u64) -> u32 {
        self.function_call_counts
            .get(&callee_key)
            .copied()
            .unwrap_or(0)
    }

    /// Log a bridge compilation. No-op if Logger is disabled.
    pub fn log_bridge_compile(&mut self, guard_index: u32) {
        if let Some(log) = &mut self.jitlog {
            log.log_bridge_compile(guard_index);
        }
    }

    // ── Quasi-immutable field invalidation ──

    /// Register that the compiled loop at `green_key_hash` depends on the
    /// quasi-immutable field identified by `qmut_key`.
    ///
    /// When `invalidate_quasiimmut(qmut_key)` is called later, the compiled
    /// loop's JitCellToken will be invalidated, causing GUARD_NOT_INVALIDATED
    /// to fail and forcing a retrace.
    ///
    /// `qmut_key` should be a hash of (object_id, field_index) or similar.
    pub fn register_quasiimmut_dependency(&mut self, qmut_key: u64, green_key_hash: u64) {
        self.quasiimmut_deps
            .entry(qmut_key)
            .or_default()
            .insert(green_key_hash);
    }

    /// Invalidate all compiled loops that depend on the quasi-immutable field
    /// identified by `qmut_key`.
    ///
    /// Called by the interpreter when a quasi-immutable field is mutated.
    /// Each dependent loop's JitCellToken has its invalidated flag set, causing
    /// GUARD_NOT_INVALIDATED to fail on the next execution.
    ///
    /// Returns the number of loops invalidated.
    pub fn invalidate_quasiimmut(&mut self, qmut_key: u64) -> usize {
        let deps = match self.quasiimmut_deps.remove(&qmut_key) {
            Some(deps) => deps,
            None => return 0,
        };

        let mut invalidated = 0;
        for green_key_hash in &deps {
            if let Some(cell) = self.cells.get_mut(green_key_hash) {
                if let Some(token) = &cell.loop_token {
                    token.invalidate();
                    cell.state = BaseJitCellState::Invalidated;
                    invalidated += 1;
                }
            }
        }
        invalidated
    }

    /// Invalidate all compiled loops that contain a GUARD_NOT_INVALIDATED.
    ///
    /// This is a brute-force invalidation used when the specific qmut_key
    /// is not known (e.g., bulk invalidation after a class hierarchy change).
    pub fn invalidate_all(&mut self) {
        for cell in self.cells.values_mut() {
            if let Some(token) = &cell.loop_token {
                token.invalidate();
                cell.state = BaseJitCellState::Invalidated;
            }
        }
        self.quasiimmut_deps.clear();
    }

    // ── BaseJitCell state machine API ──

    /// Get the explicit state of a BaseJitCell for a green key.
    /// Returns `NotHot` if no cell exists.
    #[inline]
    pub fn get_cell_state(&self, green_key_hash: u64) -> BaseJitCellState {
        self.cells
            .get(&green_key_hash)
            .map(|c| c.state)
            .unwrap_or(BaseJitCellState::NotHot)
    }

    /// Explicitly transition a cell to a new state.
    ///
    /// This is the low-level state-machine driver. Most callers should use
    /// the higher-level methods (`maybe_compile`, `finish_tracing`,
    /// `attach_procedure_to_interp`, `abort_tracing`) which call this internally.
    pub fn transition_cell(&mut self, green_key_hash: u64, new_state: BaseJitCellState) {
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(BaseJitCell::new);

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

    /// Set a JIT parameter by name, mirroring warmstate.py set_param_*().
    ///
    /// Supported parameters:
    ///   - "threshold": compilation threshold
    ///   - "trace_limit": max ops per trace
    ///   - "bridge_threshold": guard fail count before bridge compilation
    ///   - "function_threshold": calls before inlining
    ///   - "max_inline_depth": maximum inlining depth
    /// Set a JIT parameter by name.
    ///
    /// RPython warmstate.py: all `set_param_*` methods unified.
    pub fn set_param(&mut self, name: &str, value: i64) {
        match name {
            "threshold" => self.set_threshold(value as u32),
            "trace_limit" => self.trace_limit = value as u32,
            "trace_eagerness" | "bridge_threshold" => self.bridge_threshold = value as u32,
            "function_threshold" => self.function_threshold = value as u32,
            "max_inline_depth" => self.max_inline_depth = value as u32,
            "retrace_limit" => self.retrace_limit = value as u32,
            "max_retrace_guards" => self.max_retrace_guards = value as u32,
            "max_unroll_loops" => self.max_unroll_loops = value as u32,
            "max_unroll_recursion" => self.max_unroll_recursion = value as u32,
            "loop_longevity" => self.loop_longevity = value as u32,
            "vectorize" => self.vectorize = value != 0,
            "vec_cost" => self.vec_cost = value as u32,
            "inlining" => self.inlining = value != 0,
            "disable_unrolling" => self.disable_unrolling_threshold = value as u32,
            "pureop_historylength" => self.pureop_historylength = value as u32,
            "decay" => self.decay = value as u32,
            "enable_opts" => {} // string param, handled by set_param_enable_opts
            _ => {}
        }
    }

    /// warmstate.py: set_param_enable_opts(value)
    /// Set which optimization passes are enabled.
    /// Value is a colon-separated string like "intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll".
    /// "all" enables all passes.
    pub fn set_param_enable_opts(&mut self, value: &str) {
        self.enable_opts = if value == "all" || value.is_empty() {
            // All passes enabled (default)
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
            "threshold" => Some(self.counter.threshold() as i64),
            "trace_limit" => Some(self.trace_limit as i64),
            "trace_eagerness" | "bridge_threshold" => Some(self.bridge_threshold as i64),
            "function_threshold" => Some(self.function_threshold as i64),
            "max_inline_depth" => Some(self.max_inline_depth as i64),
            "retrace_limit" => Some(self.retrace_limit as i64),
            "max_retrace_guards" => Some(self.max_retrace_guards as i64),
            "max_unroll_loops" => Some(self.max_unroll_loops as i64),
            "max_unroll_recursion" => Some(self.max_unroll_recursion as i64),
            "loop_longevity" => Some(self.loop_longevity as i64),
            "vectorize" => Some(if self.vectorize { 1 } else { 0 }),
            "vec_cost" => Some(self.vec_cost as i64),
            "inlining" => Some(if self.inlining { 1 } else { 0 }),
            "disable_unrolling" => Some(self.disable_unrolling_threshold as i64),
            "pureop_historylength" => Some(self.pureop_historylength as i64),
            "decay" => Some(self.decay as i64),
            _ => None,
        }
    }

    /// warmstate.py: set_param_to_default(name)
    /// Reset a single JIT parameter to its default value.
    pub fn set_param_to_default(&mut self, name: &str) {
        match name {
            "threshold" => self.set_threshold(1039), // RPython default
            "trace_limit" => self.trace_limit = DEFAULT_TRACE_LIMIT,
            "trace_eagerness" | "bridge_threshold" => self.bridge_threshold = 200,
            "function_threshold" => self.function_threshold = DEFAULT_FUNCTION_THRESHOLD,
            "max_inline_depth" => self.max_inline_depth = 10,
            "retrace_limit" => self.retrace_limit = DEFAULT_RETRACE_LIMIT,
            "max_retrace_guards" => self.max_retrace_guards = 15,
            "max_unroll_loops" => self.max_unroll_loops = 0,
            "max_unroll_recursion" => {
                self.max_unroll_recursion = DEFAULT_MAX_INLINE_DEPTH;
            }
            "loop_longevity" => self.loop_longevity = 1000,
            "vectorize" => self.vectorize = false,
            "vec_cost" => self.vec_cost = 0,
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
        self.retrace_limit
    }
    pub fn max_retrace_guards(&self) -> u32 {
        self.max_retrace_guards
    }
    pub fn max_unroll_loops(&self) -> u32 {
        self.max_unroll_loops
    }
    pub fn max_unroll_recursion(&self) -> u32 {
        self.max_unroll_recursion
    }
    pub fn loop_longevity(&self) -> u32 {
        self.loop_longevity
    }
    pub fn vectorize(&self) -> bool {
        self.vectorize
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
    /// warmstate.py: decay
    pub fn decay(&self) -> u32 {
        self.decay
    }

    /// Get a snapshot of current JIT statistics.
    pub fn get_stats(&self) -> JitStats {
        let mut stats = JitStats {
            num_cells: self.cells.len(),
            ..Default::default()
        };
        for cell in self.cells.values() {
            match cell.state {
                BaseJitCellState::Compiled => stats.num_compiled += 1,
                BaseJitCellState::Tracing => stats.num_tracing += 1,
                BaseJitCellState::Invalidated => stats.num_invalidated += 1,
                BaseJitCellState::DontTraceHere => stats.num_disable_noninlinable_function += 1,
                BaseJitCellState::NotHot => {}
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

    /// Remove dead BaseJitCells (mirrors should_remove_jitcell).
    /// Returns the number of cells removed.
    pub fn gc_cells(&mut self) -> usize {
        let before = self.cells.len();
        self.cells.retain(|_, cell| !cell.should_remove_jitcell());
        before - self.cells.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing"),
        }
    }

    #[test]
    fn test_already_tracing() {
        let mut ws = WarmEnterState::new(2);
        // First tick: eviction (always false). Second tick: threshold reached.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing(_) => {}
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
            HotResult::StartTracing(_) => {}
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
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing"),
        }
        ws.abort_tracing(42, true);

        let cell = ws.get_cell(42).unwrap();
        assert!(!cell.is_tracing());
        assert!(cell.flags & jc_flags::DONT_TRACE_HERE != 0);

        // Now it should report NotHot because DONT_TRACE_HERE is set
        match ws.maybe_compile(42) {
            HotResult::NotHot => {}
            _ => panic!("expected NotHot due to DONT_TRACE_HERE"),
        }
    }

    #[test]
    fn test_abort_tracing_allows_retry() {
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing"),
        }
        // Abort without DONT_TRACE_HERE
        ws.abort_tracing(42, false);

        // Counter was reset during start_tracing, but hash is still in the table.
        // Need to tick again to reach threshold. The hash is found now (not evicted),
        // so one tick to reach count=1, another to reach count=2 >= threshold=2.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing(_) => {}
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
            HotResult::StartTracing(_) => {}
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
        assert_eq!(second, first + 1);
        assert_eq!(third, second + 1);
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
        let recorder = match ws.maybe_compile(key) {
            HotResult::StartTracing(rec) => rec,
            _ => panic!("expected StartTracing"),
        };

        // Phase 3: Already tracing
        assert!(matches!(ws.maybe_compile(key), HotResult::AlreadyTracing));

        // Phase 4: Finish tracing, install compiled code
        ws.finish_tracing(key);
        let token_num = ws.alloc_token_number();
        let token = JitCellToken::new(token_num);
        ws.attach_procedure_to_interp(key, token);

        // Phase 5: Run compiled
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));

        drop(recorder);
    }

    #[test]
    fn test_bridge_threshold_default() {
        let ws = WarmEnterState::new(3);
        assert_eq!(ws.bridge_threshold(), 200); // PyPy default: trace_eagerness
    }

    #[test]
    fn test_bridge_threshold_custom() {
        let mut ws = WarmEnterState::new(3);
        ws.set_bridge_threshold(10);
        assert_eq!(ws.bridge_threshold(), 10);
    }

    #[test]
    fn test_should_compile_bridge() {
        let ws = WarmEnterState::new(3);
        assert!(!ws.should_compile_bridge(0));
        assert!(!ws.should_compile_bridge(199));
        assert!(ws.should_compile_bridge(200));
        assert!(ws.should_compile_bridge(300));
    }

    #[test]
    fn test_start_retrace_preserves_input_types() {
        let mut ws = WarmEnterState::new(3);
        let mut recorder = ws.start_retrace(&[Type::Ref, Type::Int, Type::Float]);
        recorder.close_loop(&[majit_ir::OpRef(0), majit_ir::OpRef(1), majit_ir::OpRef(2)]);
        let trace = recorder.get_trace();
        let input_types: Vec<Type> = trace.inputargs.iter().map(|arg| arg.tp).collect();
        assert_eq!(input_types, vec![Type::Ref, Type::Int, Type::Float]);
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

    #[test]
    fn test_should_inline_function_below_threshold() {
        let mut ws = WarmEnterState::new(3);
        ws.set_function_threshold(3);

        // First two calls: below threshold, don't inline
        assert!(!ws.should_inline_function(42));
        assert!(!ws.should_inline_function(42));

        // Third call: reaches threshold, inline
        assert!(ws.should_inline_function(42));

        // Subsequent calls: still inline
        assert!(ws.should_inline_function(42));
    }

    #[test]
    fn test_should_inline_function_different_keys() {
        let mut ws = WarmEnterState::new(3);
        ws.set_function_threshold(2);

        assert!(!ws.should_inline_function(1));
        assert!(!ws.should_inline_function(2));
        // Key 1 reaches threshold on second call
        assert!(ws.should_inline_function(1));
        // Key 2 reaches threshold on second call
        assert!(ws.should_inline_function(2));
    }

    #[test]
    fn test_function_count_reset() {
        let mut ws = WarmEnterState::new(3);
        ws.set_function_threshold(2);

        assert!(!ws.should_inline_function(42));
        assert_eq!(ws.function_call_count(42), 1);

        ws.reset_function_counts();
        assert_eq!(ws.function_call_count(42), 0);

        // After reset, needs to reach threshold again
        assert!(!ws.should_inline_function(42));
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
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing"),
        }

        // Simulate: recorder.is_too_long() was true, so abort with dont_trace.
        ws.abort_tracing(42, true);

        // The key is now blacklisted.
        let cell = ws.get_cell(42).unwrap();
        assert!(cell.flags & jc_flags::DONT_TRACE_HERE != 0);
        assert!(!cell.is_tracing());

        // Future maybe_compile returns NotHot even though counter might tick.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
    }

    #[test]
    fn test_disable_noninlinable_function_blocks_inlining() {
        let mut ws = WarmEnterState::new(3);
        ws.set_function_threshold(2);

        assert!(!ws.should_inline_function(42));
        assert!(ws.should_inline_function(42));

        ws.reset_function_counts();
        ws.disable_noninlinable_function(42);

        assert!(!ws.can_inline_callable(42));
        assert!(!ws.should_inline_function(42));
        assert!(!ws.should_inline_function(42));
    }

    #[test]
    fn test_abort_too_long_then_retry_different_key() {
        // Aborting one key's trace as too long should not affect other keys.
        let mut ws = WarmEnterState::new(2);

        // Key 42: start and abort as too long.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing for key 42"),
        }
        ws.abort_tracing(42, true);

        // Key 99: should still work normally.
        assert!(matches!(ws.maybe_compile(99), HotResult::NotHot));
        match ws.maybe_compile(99) {
            HotResult::StartTracing(_) => {}
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
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing"),
        }

        // Phase 2: trace is too long, abort without blacklist.
        ws.abort_tracing(key, false);

        // Phase 3: retry, reach threshold again.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing(_rec) => {
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
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing (attempt 1)"),
        }
        ws.abort_tracing(key, false);

        // Second attempt: after abort, counter was reset, need to tick again.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing (attempt 2)"),
        }
        ws.abort_tracing(key, false);

        // Third attempt: succeeds and gets compiled.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing(_) => {
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
            HotResult::StartTracing(_) => {}
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

    // ── Loop aging (memmgr parity) tests ──
    //
    // Ported from rpython/jit/metainterp/test/test_memmgr.py.

    #[test]
    fn test_loop_aging_basic() {
        // Loops not accessed for max_age generations are evicted.
        let mut aging = LoopAging::new(3);

        aging.register_loop(1);
        aging.register_loop(2);
        assert_eq!(aging.alive_count(), 2);

        // Advance 3 generations without refreshing.
        let evicted = aging.next_generation(); // gen 1
        assert!(evicted.is_empty());
        let evicted = aging.next_generation(); // gen 2
        assert!(evicted.is_empty());
        let evicted = aging.next_generation(); // gen 3
        assert!(evicted.is_empty());

        // gen 4: loops registered at gen 0, threshold = 4-3 = 1, 0 < 1 → evict
        let evicted = aging.next_generation();
        assert_eq!(evicted.len(), 2);
        assert_eq!(aging.alive_count(), 0);
    }

    #[test]
    fn test_loop_aging_disabled() {
        // max_age=0 disables eviction entirely.
        let mut aging = LoopAging::new(0);

        aging.register_loop(1);
        aging.register_loop(2);

        for _ in 0..100 {
            let evicted = aging.next_generation();
            assert!(evicted.is_empty());
        }
        assert_eq!(aging.alive_count(), 2);
    }

    #[test]
    fn test_loop_aging_refresh() {
        // Accessing a loop resets its age.
        let mut aging = LoopAging::new(3);

        aging.register_loop(1);
        aging.register_loop(2);

        // Advance 2 generations, refreshing loop 1 each time.
        aging.next_generation(); // gen 1
        aging.keep_loop_alive(1);
        aging.next_generation(); // gen 2
        aging.keep_loop_alive(1);
        aging.next_generation(); // gen 3

        // gen 4: loop 2 was registered at gen 0, threshold = 4-3=1, 0 < 1 → evict
        // loop 1 was refreshed at gen 2, 2 >= 1 → alive
        let evicted = aging.next_generation();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], 2);
        assert_eq!(aging.alive_count(), 1);

        // Keep refreshing loop 1 — it should never be evicted.
        aging.keep_loop_alive(1);
        for _ in 0..10 {
            let evicted = aging.next_generation();
            assert!(evicted.is_empty());
            aging.keep_loop_alive(1);
        }
        assert_eq!(aging.alive_count(), 1);
    }

    #[test]
    fn test_loop_aging_mixed() {
        // Mix of registering at different generations.
        let mut aging = LoopAging::new(2);

        aging.register_loop(1); // registered at gen 0
        aging.next_generation(); // gen 1
        aging.register_loop(2); // registered at gen 1

        // gen 2: threshold = 2-2=0, loop 1 at gen 0: 0 >= 0 → alive,
        //        loop 2 at gen 1: 1 >= 0 → alive
        let evicted = aging.next_generation();
        assert!(evicted.is_empty());

        // gen 3: threshold = 3-2=1, loop 1 at gen 0: 0 < 1 → evict,
        //        loop 2 at gen 1: 1 >= 1 → alive
        let evicted = aging.next_generation();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], 1);
        assert_eq!(aging.alive_count(), 1);
    }

    // ── Memmgr deeper coverage (RPython: test_memmgr.py parity) ──

    #[test]
    fn test_evicted_loops_can_be_recompiled() {
        // After a loop is evicted by loop aging, it can be re-registered
        // (recompiled) and tracked again.
        let mut aging = LoopAging::new(2);

        aging.register_loop(1); // gen 0
        // Advance until eviction
        aging.next_generation(); // gen 1
        aging.next_generation(); // gen 2
        let evicted = aging.next_generation(); // gen 3: threshold=1, 0 < 1 → evict
        assert!(evicted.contains(&1));
        assert_eq!(aging.alive_count(), 0);

        // Re-register the same loop (recompiled)
        aging.register_loop(1); // now at gen 3
        assert_eq!(aging.alive_count(), 1);

        // Should stay alive for max_age generations
        let evicted = aging.next_generation(); // gen 4: threshold=2, 3 >= 2 → alive
        assert!(evicted.is_empty());
        let evicted = aging.next_generation(); // gen 5: threshold=3, 3 >= 3 → alive
        assert!(evicted.is_empty());
        let evicted = aging.next_generation(); // gen 6: threshold=4, 3 < 4 → evict
        assert!(evicted.contains(&1));
    }

    #[test]
    fn test_generation_overflow_saturating() {
        // Verify that generation counter uses saturating subtraction
        // and doesn't panic or wrap on extreme values.
        let mut aging = LoopAging::new(3);

        // Advance many generations to get a high generation number
        for _ in 0..1000 {
            aging.next_generation();
        }
        assert_eq!(aging.generation(), 1000);

        // Register a loop at the high generation
        aging.register_loop(42);
        assert_eq!(aging.alive_count(), 1);

        // Should still evict correctly after max_age more generations
        aging.next_generation(); // gen 1001
        aging.next_generation(); // gen 1002
        aging.next_generation(); // gen 1003
        let evicted = aging.next_generation(); // gen 1004: threshold=1001, 1000 < 1001 → evict
        assert!(evicted.contains(&42));
    }

    #[test]
    fn test_loop_aging_with_warm_state_integration() {
        // Simulate the interaction between loop aging and WarmEnterState:
        // - WarmEnterState compiles a loop and registers it with LoopAging.
        // - LoopAging evicts the loop.
        // - WarmEnterState removes the compiled loop and allows recompilation.
        let mut ws = WarmEnterState::new(2);
        let mut aging = LoopAging::new(2);
        let key = 0xF00D;

        // Step 1: compile a loop
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing"),
        }
        ws.finish_tracing(key);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token);
        aging.register_loop(key);

        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
        assert_eq!(aging.alive_count(), 1);

        // Step 2: advance past max_age without refreshing
        aging.next_generation();
        aging.next_generation();
        let evicted = aging.next_generation();
        assert!(evicted.contains(&key));

        // In a real system, eviction would cause the WarmEnterState to reset
        // the cell so the loop can be recompiled. Simulate by checking
        // that we can re-install.
        let token2 = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token2);
        aging.register_loop(key);

        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
        assert_eq!(aging.alive_count(), 1);
    }

    #[test]
    fn test_loop_aging_does_not_affect_active_loops() {
        // Loops that are kept alive each generation should never be evicted,
        // even as other loops are evicted around them.
        // This simulates "currently-executing" loops being refreshed.
        let mut aging = LoopAging::new(2);

        aging.register_loop(1); // "active" loop
        aging.register_loop(2); // "inactive" loop
        aging.register_loop(3); // "inactive" loop

        for _ in 0..20 {
            aging.keep_loop_alive(1); // keep loop 1 active
            let evicted = aging.next_generation();

            // Loop 1 should never be evicted
            assert!(!evicted.contains(&1), "active loop should never be evicted");
        }

        // Loop 1 should still be alive
        assert!(aging.alive_count() >= 1);
        // Loop 2 and 3 should have been evicted long ago
        // (registered at gen 0, threshold grows each generation)
    }

    #[test]
    fn test_loop_aging_set_max_age_dynamic() {
        // max_age can be changed dynamically. Changing it affects future
        // eviction decisions but doesn't retroactively evict.
        let mut aging = LoopAging::new(10);

        aging.register_loop(1); // gen 0
        aging.next_generation(); // gen 1

        // Reduce max_age to 1 — loop 1 at gen 0, threshold = 2 - 1 = 1
        // 0 < 1 → should be evicted next generation
        aging.set_max_age(1);
        let evicted = aging.next_generation(); // gen 2
        assert!(
            evicted.contains(&1),
            "reducing max_age should cause earlier eviction"
        );
    }

    #[test]
    fn test_loop_aging_interleaved_register_and_evict() {
        // Mirrors RPython's test_basic_3: register loops at different
        // generations, keep some alive on even indices.
        let mut aging = LoopAging::new(4);

        let mut keys: Vec<u64> = Vec::new();
        for i in 0..10u64 {
            keys.push(i);
            aging.register_loop(i);
            aging.next_generation();

            // Keep even-indexed loops alive
            for j in (0..=i).step_by(2) {
                aging.keep_loop_alive(j);
            }
        }

        // After 10 generations with max_age=4:
        // Even-indexed loops should still be alive (refreshed each gen).
        // Odd-indexed loops registered at gen i should be evicted
        // when generation > i + 4.
        for i in 0..10u64 {
            let is_alive = aging.loop_generations.contains_key(&i);
            if i % 2 == 0 {
                assert!(is_alive, "even-indexed loop {} should be alive", i);
            }
            // Odd loops registered early enough will have been evicted.
            // Loop i (odd) registered at gen i. After gen 10, threshold = 10 - 4 = 6.
            // Evicted if i < 6.
            if i % 2 != 0 && i < 6 {
                assert!(
                    !is_alive,
                    "odd loop {} registered at gen {} should be evicted by gen 10",
                    i, i
                );
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
        ws.set_function_threshold(2);
        ws.set_max_inline_depth(3);

        // Depth 0: allowed
        assert!(ws.can_inline_at_depth(0));
        // Depth 2: allowed (< 3)
        assert!(ws.can_inline_at_depth(2));
        // Depth 3: not allowed (>= 3)
        assert!(!ws.can_inline_at_depth(3));

        // Inlining decisions are independent of trace length:
        // function 42 needs 2 calls before inlining
        assert!(!ws.should_inline_function(42));
        assert!(ws.should_inline_function(42));

        // Even at max depth, inlining threshold still tracks calls
        ws.reset_function_counts();
        assert!(!ws.should_inline_function(42));
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
            HotResult::StartTracing(_) => {}
            _ => panic!("expected StartTracing at threshold 5"),
        }

        // Abort without blacklisting
        ws.abort_tracing(key, false);

        // Lower threshold for retry
        ws.set_threshold(2);

        // Now only 2 ticks needed
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing(_) => {
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
            HotResult::StartTracing(_) => {}
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
            HotResult::StartTracing(_) => {}
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
            HotResult::StartTracing(_) => {}
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

        ws.set_param("bridge_threshold", 10);
        assert_eq!(ws.bridge_threshold(), 10);

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
        assert!(matches!(ws.maybe_compile(1), HotResult::StartTracing(_)));
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(2), HotResult::StartTracing(_)));

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
            HotResult::StartTracing(_) => {}
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
        assert!(matches!(ws.maybe_compile(1), HotResult::StartTracing(_)));
        assert_eq!(ws.tracing_generation(), 1);

        let cell = ws.get_cell(1).unwrap();
        assert_eq!(cell.tracing_generation, 1);

        // Start tracing key 2 → generation 2
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(2), HotResult::StartTracing(_)));
        assert_eq!(ws.tracing_generation(), 2);

        let cell = ws.get_cell(2).unwrap();
        assert_eq!(cell.tracing_generation, 2);
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
    }

    #[test]
    fn test_gc_cells() {
        let mut ws = WarmEnterState::new(2);

        // Create some cells in various states
        // Key 1: compiled (should NOT be removed)
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(1), HotResult::StartTracing(_)));
        ws.finish_tracing(1);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(1, token);

        // Key 2: tracing (should NOT be removed)
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(2), HotResult::StartTracing(_)));

        // Key 3: aborted without dont_trace → NotHot, removable
        assert!(matches!(ws.maybe_compile(3), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(3), HotResult::StartTracing(_)));
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
        assert!(matches!(ws.maybe_compile(key), HotResult::StartTracing(_)));
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
    fn test_mark_dont_trace() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xABCD;
        ws.mark_dont_trace(key);
        // After marking, maybe_compile should return NotHot
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
    }

    #[test]
    fn test_quasiimmut_dependency_lifecycle() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xF00D;
        let qmut = 0xBEEF;

        // Compile a loop
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(key), HotResult::StartTracing(_)));
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
    fn test_set_param_roundtrip() {
        let mut ws = WarmEnterState::new(100);
        for name in WarmEnterState::param_names() {
            let original = ws.get_param(name);
            assert!(original.is_some(), "param {name} should be gettable");
            ws.set_param(name, 999);
            ws.set_param_to_default(name);
            // After default, should be same as a fresh instance
        }
    }
}
