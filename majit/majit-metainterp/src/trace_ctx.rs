//! No direct RPython equivalent — unified trace recording context
//! (RPython splits this across MetaInterp.history, compile.py, and Trace).

use majit_ir::{DescrRef, GreenKey, OpCode, OpRef, Type, Value};
use majit_trace::heapcache::HeapCache;
use majit_trace::recorder::{Trace, TracePosition};

use majit_backend::JitCellToken;

use crate::call_descr::{
    make_call_assembler_descr, make_call_assembler_descr_with_vable, make_call_descr,
    make_call_may_force_descr,
};
use crate::constant_pool::ConstantPool;
use crate::fail_descr::{make_fail_descr, make_fail_descr_typed};
use crate::jitdriver::JitDriverStaticData;
use crate::virtualizable::VirtualizableInfo;

/// Inverse of `heap_value_for`: encode a typed `Value` into the raw i64
/// bit-pattern that `VirtualizableInfo::write_field`/`write_array_item`
/// interpret per field/item type.
pub(crate) fn value_to_raw_bits(value: Value) -> i64 {
    match value {
        Value::Int(v) => v,
        Value::Float(f) => f.to_bits() as i64,
        Value::Ref(r) => r.as_usize() as i64,
        Value::Void => 0,
    }
}

/// Tracing context: wraps Trace + ConstantPool with convenience API.
///
/// The interpreter uses this during trace recording to:
/// - Record IR operations
/// - Manage constants (with deduplication)
/// - Record guards (with auto-generated FailDescr)
/// - Record function calls (with auto-generated CallDescr)
pub struct TraceCtx {
    pub(crate) recorder: Trace,
    pub(crate) green_key: u64,
    root_green_key: u64,
    pub(crate) constants: ConstantPool,
    /// Stack of inlined function frames (callee green_keys).
    inline_frames: Vec<u64>,
    /// Start positions for currently active inlined trace-through frames.
    ///
    /// This mirrors the subset of PyPy's `portal_trace_positions` that we
    /// need for `find_biggest_function()`: active inlined callees and the
    /// trace length at which each one started tracing.
    inline_trace_positions: Vec<(u64, usize)>,
    /// Structured green key values (if provided by the interpreter).
    green_key_values: Option<GreenKey>,
    /// Declarative driver layout metadata, if provided by the interpreter.
    driver_descriptor: Option<JitDriverStaticData>,
    /// Standard virtualizable boxes -- OpRefs for each static field + array element.
    /// When set, vable_getfield/setfield access these instead of emitting heap ops.
    /// Layout: [field_0, ..., field_N, arr_0[0], ..., arr_0[M], ..., vable_ref]
    ///
    /// The last element (`boxes[-1]`) is the standard virtualizable identity
    /// (RPython parity: `virtualizable_boxes[-1]`). Used by gen_store_back_in_vable
    /// to distinguish standard vs nonstandard virtualizable.
    virtualizable_boxes: Option<Vec<OpRef>>,
    /// Concrete shadow of `virtualizable_boxes`. Same layout, each slot carries
    /// the current runtime `Value` (RPython Box ≡ OpRef + concrete value).
    /// Seeded from `original_boxes` in `initialize_virtualizable` and kept in
    /// sync on every standard vable write (`vable_setfield`,
    /// `vable_setarrayitem_indexed`, `store_local_value` mirror).
    virtualizable_values: Option<Vec<Value>>,
    /// VirtualizableInfo for the standard virtualizable (if any).
    virtualizable_info: Option<VirtualizableInfo>,
    /// Lengths of each virtualizable array field, needed for flat index computation.
    virtualizable_array_lengths: Option<Vec<usize>>,
    /// Live virtualizable heap pointer (pyjitpl.py:3446 write_boxes target).
    /// Mirrored from `MetaInterp::vable_ptr` at trace/bridge-entry.  Used by
    /// `synchronize_virtualizable` to write `virtualizable_values` back to
    /// the live PyFrame after every standard vable setfield / setarrayitem
    /// (virtualizable.py:101 write_boxes parity). `None` disables the
    /// write — unit-test or init-before-run path.
    virtualizable_heap_ptr: Option<*const u8>,
    /// Header PC at which this trace started (0 = function entry).
    pub header_pc: usize,
    /// When a cross-loop cut occurs (trace closes at inner loop header),
    /// the green key for the inner loop. Used to register an alias
    /// so can_enter_jit at the inner back-edge finds the outer key's entry.
    pub cut_inner_green_key: Option<u64>,
    /// Pending OpRef replacements from inline callee returns.
    /// Applied when the trace is finalized (close_loop/compile).
    replacements: Vec<(OpRef, OpRef)>,
    /// pyjitpl.py:3030 current_merge_points — loop headers visited during
    /// tracing with their trace positions. First visit records the key +
    /// position; second visit closes the loop.
    current_merge_points: Vec<MergePoint>,
    /// pyjitpl.py:2979 reached_loop_header parity: callback to check
    /// has_compiled_targets(ptoken) for a given green key. Bridge traces
    /// skip loop headers without compiled targets. Live lookup (not snapshot)
    /// matches RPython's get_procedure_token(greenboxes) + has_compiled_targets.
    pub has_compiled_targets_fn: Option<Box<dyn Fn(u64) -> bool>>,
    /// pyjitpl.py:2398: tracing-time heap cache.
    /// Tracks field/array values, allocations, escape status, and class/nullity
    /// knowledge during tracing to avoid recording redundant operations.
    heap_cache: HeapCache,
    /// pyjitpl.py:2411 force_finish_trace: when True, trace is segmented
    /// at 80% of limit via _create_segmented_trace_and_blackhole.
    force_finish: bool,
    /// pyjitpl.py:2594 frame.pc: last bytecode pc passed to trace_fn.
    /// Used by force_finish_trace segmenting to record the guard-point pc.
    pub last_traced_pc: usize,
    /// GC-safe constant OpRefs for each initial inputarg at trace start.
    /// Each entry is a ConstantPool-allocated OpRef whose value is kept
    /// up-to-date by the shadow stack (Ref constants survive GC moves).
    /// Used by cut_trace_from to remap escaped original inputargs to
    /// existing pool constants, avoiding both stale pointers and
    /// entry-contract mismatches.
    pub initial_inputarg_consts: Vec<majit_ir::OpRef>,
    /// pyjitpl.py:1087 parity: quasi-immutable field read needs a
    /// GUARD_NOT_INVALIDATED with full snapshot at the field read's orgpc.
    /// Stores Some(orgpc) when pending.
    pending_guard_not_invalidated_pc: Option<usize>,
    /// pyjitpl.py:2394 `MetaInterp.forced_virtualizable` parity. Tracks the
    /// vbox handed to `gen_store_back_in_vable` so the second
    /// `opimpl_hint_force_virtualizable` of the same trace can be skipped.
    /// RPython resets this in `MetaInterp.__init__`; pyre keeps it on
    /// TraceCtx because TraceCtx is freshly created per trace and the
    /// MetaInterp is reused across traces.
    forced_virtualizable: Option<OpRef>,
    /// pyjitpl.py:2397: call_pure_results — maps constant argument tuples
    /// to their concrete result values, recorded during tracing.
    /// Passed to the optimizer for cross-iteration CALL_PURE folding.
    call_pure_results: std::collections::HashMap<Vec<Value>, Value>,
}

/// pyjitpl.py:2989 — a visited loop header with its trace position.
///
/// RPython stores `(original_boxes, start)` where `original_boxes` is the
/// full list of green+red args at the first visit, and `start` is a 5-tuple
/// trace position. majit stores the equivalent as OpRef vectors + TracePosition.
#[derive(Clone, Debug)]
pub struct MergePoint {
    /// Green key of the loop header.
    pub green_key: u64,
    /// Trace position when this loop header was first visited.
    pub position: TracePosition,
    /// pyjitpl.py:2989: original_boxes — live variable OpRefs at the first
    /// visit to this loop header. Used by compile_loop/compile_retrace as
    /// the inputargs for trace cutting.
    pub original_boxes: Vec<OpRef>,
    /// Types of original_boxes. RPython Box carries type implicitly;
    /// majit stores types alongside OpRefs for compile_retrace parity.
    pub original_box_types: Vec<Type>,
    /// Bytecode PC of this loop header. Used by cut_trace_from to update
    /// meta when the trace closes at a different loop.
    pub header_pc: usize,
}

impl TraceCtx {
    /// pyjitpl.py:2991 — check if a loop header was already visited.
    pub fn has_merge_point(&self, key: u64) -> bool {
        self.current_merge_points
            .iter()
            .any(|mp| mp.green_key == key)
    }

    /// pyjitpl.py:3029-3030 — record a loop header visit with position
    /// and live variable snapshot.
    ///
    /// RPython allows multiple merge points with the same green key
    /// (representing different loop iterations or inlining depths).
    /// Always appends; has_merge_point checks if any match exists.
    pub fn add_merge_point(
        &mut self,
        key: u64,
        live_args: Vec<OpRef>,
        live_arg_types: Vec<Type>,
        header_pc: usize,
    ) {
        let position = self.recorder.get_position();
        self.current_merge_points.push(MergePoint {
            green_key: key,
            position,
            original_boxes: live_args,
            original_box_types: live_arg_types,
            header_pc,
        });
    }

    /// pyjitpl.py:2908 — bridge traces start with empty merge points.
    pub fn clear_merge_points(&mut self) {
        self.current_merge_points.clear();
    }

    /// pyjitpl.py:2801 / 2803 / 2818 / 7985 — `current_merge_points[0]`
    /// is the outermost loop header's greenkey.  Used by
    /// `blackhole_if_trace_too_long` / `prepare_trace_segmenting` /
    /// `aborted_tracing` to distinguish "tracing a loop body" from
    /// "tracing a bridge" (empty merge-points list).
    pub fn current_merge_points_first_greenkey(&self) -> Option<u64> {
        self.current_merge_points.first().map(|mp| mp.green_key)
    }

    /// pyjitpl.py:2988: find merge point by key, searching in reverse
    /// order (most recent first, matching RPython's range(len-1, -1, -1)).
    pub fn get_merge_point(&self, key: u64) -> Option<&MergePoint> {
        self.current_merge_points
            .iter()
            .rev()
            .find(|mp| mp.green_key == key)
    }

    /// pyjitpl.py:2994 same_greenkey + header identity: check if a specific
    /// loop header (key, header_pc) was already visited.
    pub fn has_merge_point_at(&self, key: u64, header_pc: usize) -> bool {
        self.current_merge_points
            .iter()
            .any(|mp| mp.green_key == key && mp.header_pc == header_pc)
    }

    /// pyjitpl.py:2988 + header identity: find merge point by (key, header_pc),
    /// searching in reverse order (most recent first).
    pub fn get_merge_point_at(&self, key: u64, header_pc: usize) -> Option<&MergePoint> {
        self.current_merge_points
            .iter()
            .rev()
            .find(|mp| mp.green_key == key && mp.header_pc == header_pc)
    }

    /// history.py: get_trace_position — current recorder position.
    pub fn get_trace_position(&self) -> TracePosition {
        self.recorder.get_position()
    }

    /// history.py: cut — restore recorder to a saved position.
    pub fn cut_trace(&mut self, pos: TracePosition) {
        self.recorder.cut(pos);
    }

    /// pyjitpl.py:2398: access the tracing-time heap cache.
    pub fn heap_cache(&self) -> &HeapCache {
        &self.heap_cache
    }

    /// Mutable access to the tracing-time heap cache.
    pub fn heap_cache_mut(&mut self) -> &mut HeapCache {
        &mut self.heap_cache
    }

    /// pyjitpl.py:1087 parity: check if a quasi-immut guard is pending.
    pub fn pending_guard_not_invalidated_pc(&self) -> Option<usize> {
        self.pending_guard_not_invalidated_pc
    }

    /// Set pending quasi-immut guard with the field read's orgpc.
    pub fn set_pending_guard_not_invalidated(&mut self, pc: Option<usize>) {
        self.pending_guard_not_invalidated_pc = pc;
    }

    /// pyjitpl.py:2951, 2418: reset heap cache at loop header / retrace.
    pub fn reset_heap_cache(&mut self) {
        self.heap_cache.reset();
    }

    /// heapcache.py: EF_RANDOM_EFFECTS — invalidate ALL caches including
    /// unescaped objects. Used for operations with completely unknown effects.
    pub fn invalidate_all_heap_caches(&mut self) {
        self.heap_cache.invalidate_all_caches();
    }

    /// pyjitpl.py:1776-1780: jit.isvirtual(obj) — check if an object
    /// is likely virtual (allocated during this trace and not escaped).
    pub fn is_likely_virtual(&self, obj: OpRef) -> bool {
        self.heap_cache.is_likely_virtual(obj)
    }

    /// pyjitpl.py:1805-1806: record VIRTUAL_REF(box, cindex).
    /// `cindex` = ConstInt(len(virtualref_boxes) // 2) — pair index.
    /// The optimizer can later eliminate the vref if the object stays virtual.
    pub fn virtual_ref(&mut self, obj: OpRef, cindex: OpRef) -> OpRef {
        let result = self.recorder.record_op(OpCode::VirtualRefR, &[obj, cindex]);
        // pyjitpl.py:1807: heapcache.new(resbox)
        self.heap_cache.new_object(result);
        result
    }

    /// Create a standalone TraceCtx for testing or external use.
    pub fn for_test(num_inputs: usize) -> Self {
        let mut recorder = Trace::new();
        for _ in 0..num_inputs {
            recorder.record_input_arg(majit_ir::Type::Int);
        }
        Self::new(recorder, 0)
    }

    /// Create a TraceCtx for tests whose input args have mixed types.
    /// Analog of RPython `MetaInterp.create_empty_loop()` +
    /// `inputargs = [Box(tp) for tp in types]`.
    pub fn for_test_types(types: &[majit_ir::Type]) -> Self {
        let mut recorder = Trace::new();
        for &tp in types {
            recorder.record_input_arg(tp);
        }
        Self::new(recorder, 0)
    }

    /// Take the recorder out of this context (consumes self).
    pub fn into_recorder(self) -> Trace {
        self.recorder
    }

    pub(crate) fn new(recorder: Trace, green_key: u64) -> Self {
        let initial_position = recorder.get_position();
        let initial_boxes: Vec<OpRef> = (0..recorder.num_inputargs())
            .map(|i| OpRef(i as u32))
            .collect();
        let initial_types: Vec<Type> = recorder.inputarg_types().to_vec();
        TraceCtx {
            recorder,
            green_key,
            root_green_key: green_key,
            constants: ConstantPool::new(),
            inline_frames: Vec::new(),
            inline_trace_positions: Vec::new(),
            green_key_values: None,
            driver_descriptor: None,
            virtualizable_boxes: None,
            virtualizable_values: None,
            virtualizable_info: None,
            virtualizable_array_lengths: None,
            virtualizable_heap_ptr: None,
            header_pc: 0,
            cut_inner_green_key: None,
            replacements: Vec::new(),
            current_merge_points: vec![MergePoint {
                green_key,
                position: initial_position,
                original_box_types: initial_types,
                original_boxes: initial_boxes.clone(),
                header_pc: 0,
            }],
            heap_cache: HeapCache::new(),
            force_finish: false,
            last_traced_pc: 0,
            initial_inputarg_consts: vec![],
            pending_guard_not_invalidated_pc: None,
            forced_virtualizable: None,
            has_compiled_targets_fn: None,
            call_pure_results: std::collections::HashMap::new(),
        }
    }

    /// Create a TraceCtx with a structured green key.
    pub(crate) fn with_green_key(
        recorder: Trace,
        green_key: u64,
        green_key_values: GreenKey,
    ) -> Self {
        let initial_position = recorder.get_position();
        let initial_boxes: Vec<OpRef> = (0..recorder.num_inputargs())
            .map(|i| OpRef(i as u32))
            .collect();
        // RPython pyjitpl.py:2878: initial merge point types come from
        // live_arg_boxes which carry actual types (INT/REF/FLOAT).
        let initial_input_types = recorder.inputarg_types();
        TraceCtx {
            recorder,
            green_key,
            root_green_key: green_key,
            constants: ConstantPool::new(),
            inline_frames: Vec::new(),
            inline_trace_positions: Vec::new(),
            green_key_values: Some(green_key_values),
            driver_descriptor: None,
            virtualizable_boxes: None,
            virtualizable_values: None,
            virtualizable_info: None,
            virtualizable_array_lengths: None,
            virtualizable_heap_ptr: None,
            header_pc: 0,
            cut_inner_green_key: None,
            replacements: Vec::new(),
            current_merge_points: vec![MergePoint {
                green_key,
                position: initial_position,
                original_box_types: initial_input_types,
                original_boxes: initial_boxes.clone(),
                header_pc: 0,
            }],
            heap_cache: HeapCache::new(),
            force_finish: false,
            last_traced_pc: 0,
            initial_inputarg_consts: vec![],
            pending_guard_not_invalidated_pc: None,
            forced_virtualizable: None,
            has_compiled_targets_fn: None,
            call_pure_results: std::collections::HashMap::new(),
        }
    }

    /// Get the current inlining depth.
    pub fn inline_depth(&self) -> usize {
        self.inline_frames.len()
    }

    pub fn inline_trace_depth(&self) -> usize {
        self.inline_trace_positions.len()
    }

    /// Update the green key for this trace.
    ///
    /// RPython pyjitpl.py reached_loop_header(): when func-entry tracing
    /// hits a back-edge, the loop must be registered under the back-edge's
    /// green key, not the function-entry key.
    pub fn set_green_key(&mut self, key: u64) {
        self.green_key = key;
    }

    /// Check if `key` matches the current trace's green_key or any
    /// inlined frame's key. Used for self-recursion detection.
    pub fn is_tracing_key(&self, key: u64) -> bool {
        self.green_key == key || self.inline_frames.contains(&key)
    }

    /// Count how many times `key` appears in the frame stack (recursive depth).
    pub fn recursive_depth(&self, key: u64) -> usize {
        let root = if self.green_key == key { 1 } else { 0 };
        root + self.inline_frames.iter().filter(|&&k| k == key).count()
    }

    /// Register a deferred OpRef replacement. When the trace is finalized,
    /// all ops referencing `old` in their args will use `new` instead.
    pub fn replace_op(&mut self, old: OpRef, new: OpRef) {
        self.replacements.push((old, new));
    }

    /// pyjitpl.py:3499 `replace_box(oldbox, newbox)` parity.
    ///
    /// In RPython this walks every place where the old Box might appear
    /// (frame stacks, virtualref_boxes, virtualizable_boxes, heap caches)
    /// and replaces it with the new Box. The pyre equivalent applies the
    /// same logic to the side-channel `virtualizable_boxes` so that
    /// `metainterp.virtualizable_boxes[-1]` (the standard vable identity)
    /// stays in sync after `replace_op` calls.
    ///
    /// Trace ops, virtualref_boxes, and heap cache state are handled by
    /// `apply_replacements` (deferred batch) and the optimizer's separate
    /// `forwarded` chain. This method covers only the
    /// `virtualizable_boxes` walk that has no other home.
    pub fn replace_box_in_virtualizable_boxes(&mut self, old: OpRef, new: OpRef) {
        // pyjitpl.py:3506-3511 parity:
        //     if (jitdriver_sd.virtualizable_info is not None or
        //         jitdriver_sd.greenfield_info is not None):
        //         boxes = self.virtualizable_boxes
        //         for i in range(len(boxes)):
        //             if boxes[i] is oldbox:
        //                 boxes[i] = newbox
        if let Some(boxes) = self.virtualizable_boxes.as_mut() {
            for slot in boxes.iter_mut() {
                if *slot == old {
                    *slot = new;
                }
            }
        }
    }

    /// pyjitpl.py:3499-3512 `MetaInterp.replace_box(oldbox, newbox)` —
    /// trace-context portion.
    ///
    /// ```text
    ///  def replace_box(self, oldbox, newbox):
    ///      for frame in self.framestack:
    ///          frame.replace_active_box_in_frame(oldbox, newbox)
    ///      boxes = self.virtualref_boxes
    ///      for i in range(len(boxes)):
    ///          if boxes[i] is oldbox:
    ///              boxes[i] = newbox
    ///      if (self.jitdriver_sd.virtualizable_info is not None or
    ///          self.jitdriver_sd.greenfield_info is not None):
    ///          boxes = self.virtualizable_boxes
    ///          for i in range(len(boxes)):
    ///              if boxes[i] is oldbox:
    ///                  boxes[i] = newbox
    ///      self.heapcache.replace_box(oldbox, newbox)
    /// ```
    ///
    /// pyre splits `MetaInterp.replace_box` across two layers:
    ///
    ///   * `TraceCtx::replace_box` (this method) handles the eager
    ///     virtualizable_boxes + heap_cache walks AND queues the
    ///     deferred recorder rewrite via `replace_op`. This is what
    ///     `is_nonstandard_virtualizable` Step 4 calls because the
    ///     framestack walk is not reachable from inside TraceCtx.
    ///
    ///   * `MetaInterp::replace_box` (in pyjitpl.rs) is a thin
    ///     wrapper that adds the `virtualref_boxes` walk on top of
    ///     this TraceCtx call. It is the structural mirror of the
    ///     full RPython entry point.
    ///
    /// The framestack walk
    /// (`for frame in self.framestack: frame.replace_active_box_in_frame(...)`)
    /// is missing because pyre's frame state lives in PyreSym
    /// (in pyre-jit-trace) and is not reachable from MetaInterp.
    /// The deferred recorder rewrite (queued via `replace_op` and
    /// flushed in `apply_replacements`) substitutes for the
    /// per-frame walk by rewriting all already-emitted op args
    /// once at trace finalization.
    pub fn replace_box(&mut self, oldbox: OpRef, newbox: OpRef) {
        // pyjitpl.py:3506-3511 virtualizable_boxes walk.
        if let Some(boxes) = self.virtualizable_boxes.as_mut() {
            for slot in boxes.iter_mut() {
                if *slot == oldbox {
                    *slot = newbox;
                }
            }
        }
        // pyjitpl.py:3512 self.heapcache.replace_box(oldbox, newbox).
        self.heap_cache.replace_box(oldbox, newbox);
        // Queue the deferred recorder rewrite (pyre-only — RPython's
        // framestack walk has no equivalent in pyre's MetaInterp;
        // the queued rewrite is applied at finalization in
        // `apply_replacements`).
        self.replace_op(oldbox, newbox);
    }

    /// Apply all pending replacements to the trace ops.
    ///
    /// Flushes the `replace_op` queue: for each `(old, new)` pair,
    /// re-runs the eager walks (so duplicates from optimizer-level
    /// `replace_op` calls that did NOT go through `TraceCtx::replace_box`
    /// also see the heapcache / virtualizable_boxes update) and rewrites
    /// the recorder's op args + fail_args.
    pub fn apply_replacements(&mut self) {
        if self.replacements.is_empty() {
            return;
        }
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] apply_replacements: {} entries",
                self.replacements.len()
            );
            for (old, new) in &self.replacements {
                eprintln!("  {:?} → {:?}", old, new);
            }
        }
        // pyjitpl.py:3506-3511 virtualizable_boxes walk for queued entries.
        for (old, new) in &self.replacements {
            if let Some(boxes) = self.virtualizable_boxes.as_mut() {
                for slot in boxes.iter_mut() {
                    if *slot == *old {
                        *slot = *new;
                    }
                }
            }
        }
        // pyjitpl.py:3512 heapcache walk for queued entries.
        for (old, new) in &self.replacements {
            self.heap_cache.replace_box(*old, *new);
        }
        let replacements: std::collections::HashMap<OpRef, OpRef> =
            self.replacements.drain(..).collect();
        self.recorder.apply_replacements(&replacements);
    }

    /// Push an inline frame (entering a callee).
    /// Returns false if the max inline depth has been exceeded.
    pub(crate) fn push_inline_frame(&mut self, callee_key: u64, max_depth: u32) -> bool {
        if (self.inline_frames.len() as u32) >= max_depth {
            return false;
        }
        self.inline_frames.push(callee_key);
        true
    }

    /// Pop an inline frame (returning from a callee).
    pub(crate) fn pop_inline_frame(&mut self) {
        self.inline_frames.pop();
    }

    pub fn push_inline_trace_position(&mut self, green_key: u64) {
        self.inline_trace_positions
            .push((green_key, self.recorder.num_ops()));
    }

    pub fn pop_inline_trace_position(&mut self) {
        self.inline_trace_positions.pop();
    }

    pub fn truncate_inline_trace_positions(&mut self, depth: usize) {
        self.inline_trace_positions.truncate(depth);
    }

    /// pyjitpl.py:3514 find_biggest_function
    pub fn find_biggest_function(&self) -> Option<u64> {
        let current_pos = self.recorder.num_ops();
        self.inline_trace_positions
            .iter()
            .map(|&(green_key, start_pos)| (green_key, current_pos.saturating_sub(start_pos)))
            .max_by_key(|&(_, size)| size)
            .map(|(green_key, _)| green_key)
    }

    /// Get or create a constant OpRef for a given i64 value.
    pub fn const_int(&mut self, value: i64) -> OpRef {
        self.constants.get_or_insert(value)
    }

    /// Get or create a Ref-typed constant OpRef.
    /// executor.py:544 constant_from_op(op) parity: get typed Value for OpRef.
    pub fn constants_get_value(&self, opref: OpRef) -> Option<Value> {
        self.constants.get_value(opref)
    }

    /// RPython parity: Ref constants preserve their type so guard
    /// fail_args are correctly typed during guard failure recovery.
    pub fn const_ref(&mut self, value: i64) -> OpRef {
        self.constants
            .get_or_insert_typed(value, majit_ir::Type::Ref)
    }

    /// history.py:361 CONST_NULL = ConstPtr(ConstPtr.value).
    /// Ref-typed null pointer constant.
    pub fn const_null(&mut self) -> OpRef {
        self.const_ref(0)
    }

    /// Get or create a Float-typed constant OpRef.
    pub fn const_float(&mut self, value: i64) -> OpRef {
        self.constants
            .get_or_insert_typed(value, majit_ir::Type::Float)
    }

    /// Mark an existing constant OpRef with a specific type.
    /// Used when const_int() was called but the value is actually a Ref pointer
    /// (e.g., ob_type field). This preserves Cranelift's Int treatment while
    /// recording the true type for resume data.
    pub fn mark_const_type(&mut self, opref: OpRef, tp: majit_ir::Type) {
        self.constants.mark_type(opref, tp);
    }

    /// Return the type of a constant OpRef, if recorded.
    ///
    /// `numbering_type_overrides` takes priority over `constant_type`:
    /// `mark_type(opref, Ref)` exists specifically to retype a raw-pointer
    /// `ConstInt` as `Ref` for resume-data encoding, mirroring RPython's
    /// `getrawptrinfo` ConstInt→Ref retag at numbering time. The intrinsic
    /// `Box.type = 'i'` stays in `constant_type` but the snapshot consumer
    /// needs the override to produce the correct `TAGCONSTPTR`. Without
    /// checking the override first, `get_or_insert`'s `Type::Int` entry
    /// masks every `mark_type(_, Ref)` and Ref pointers get serialized as
    /// integer bits.
    pub fn const_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        self.constants
            .numbering_type_overrides()
            .get(&opref.0)
            .copied()
            .or_else(|| self.constants.constant_type(opref))
    }

    /// Return the concrete value for a constant OpRef, if it is a pooled constant.
    pub fn const_value(&self, opref: OpRef) -> Option<i64> {
        self.constants.as_ref().get(&opref.0).copied()
    }

    /// Root an Int-typed constant on the GC shadow stack.
    /// Keeps the constant's type as Int (optimizer sees Value::Int),
    /// but prevents GC from freeing the referenced object.
    pub fn root_const_for_gc(&mut self, opref: OpRef) {
        self.constants.root_int_as_ref(opref);
    }

    /// Constant-fold a pure field read on a constant object pointer.
    /// If `obj` is a constant and `descr` is immutable, reads the field
    /// at runtime and returns the value as a constant OpRef.
    pub fn try_const_fold_pure_field(
        &mut self,
        obj: OpRef,
        descr: &dyn majit_ir::Descr,
    ) -> Option<OpRef> {
        if !descr.is_always_pure() {
            return None;
        }
        let obj_ptr = self.const_value(obj)? as usize;
        if obj_ptr == 0 {
            return None;
        }
        let fd = descr.as_field_descr()?;
        let offset = fd.offset();
        let field_size = fd.field_size();
        let value = unsafe {
            let base = obj_ptr as *const u8;
            match field_size {
                8 => *(base.add(offset) as *const i64),
                4 if fd.is_field_signed() => *(base.add(offset) as *const i32) as i64,
                4 => *(base.add(offset) as *const u32) as i64,
                _ => return None,
            }
        };
        Some(self.const_int(value))
    }

    /// Record a regular IR operation.
    pub fn record_op(&mut self, opcode: OpCode, args: &[OpRef]) -> OpRef {
        self.recorder.record_op(opcode, args)
    }

    /// Record an operation with a descriptor (e.g., calls).
    pub fn record_op_with_descr(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        self.recorder.record_op_with_descr(opcode, args, descr)
    }

    /// Record a guard with auto-generated FailDescr.
    ///
    /// `num_live` is the number of live integer values (for the FailDescr).
    /// opencoder.py:819 parity: capture a snapshot of the interpreter
    /// frame state. Returns a snapshot_id for use as rd_resume_position.
    pub fn capture_resumedata(&mut self, snapshot: majit_trace::recorder::Snapshot) -> i32 {
        self.recorder.capture_resumedata(snapshot)
    }

    /// Set rd_resume_position on the last recorded guard.
    pub fn set_last_guard_resume_position(&mut self, snapshot_id: i32) {
        self.recorder.set_last_op_resume_position(snapshot_id);
    }

    /// Look up a constant value by its OpRef (>= 10_000).
    pub fn constant_value(&self, opref: OpRef) -> Option<i64> {
        self.constants.as_ref().get(&opref.0).copied()
    }

    pub fn record_guard(&mut self, opcode: OpCode, args: &[OpRef], num_live: usize) -> OpRef {
        let descr = make_fail_descr(num_live);
        self.recorder.record_guard(opcode, args, descr)
    }

    /// Record a guard with explicit fail_args.
    pub fn record_guard_with_fail_args(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        num_live: usize,
        fail_args: &[OpRef],
    ) -> OpRef {
        let descr = make_fail_descr(num_live);
        self.recorder
            .record_guard_with_fail_args(opcode, args, descr, fail_args)
    }

    /// Record a guard with explicit typed fail_args.
    pub fn record_guard_typed_with_fail_args(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        fail_arg_types: Vec<Type>,
        fail_args: &[OpRef],
    ) -> OpRef {
        let descr = make_fail_descr_typed(fail_arg_types);
        self.recorder
            .record_guard_with_fail_args(opcode, args, descr, fail_args)
    }

    /// Record a void-returning function call (CallN).
    ///
    /// Automatically registers the function pointer as a constant and
    /// creates a CallDescr. The interpreter doesn't need to manage
    /// function pointer constants or CallDescr implementations.
    pub fn call_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_void_typed(func_ptr, args, &arg_types);
    }

    /// Record an integer-returning function call (CallI).
    ///
    /// Same convenience as `call_void` but returns an OpRef for the result.
    pub fn call_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a FINISH op with a single result value.
    /// pyjitpl.py:1637 history.record1(rop.FINISH, ..., descr=token)
    pub fn record_finish(&mut self, result: OpRef, _tp: Type) {
        self.recorder.record_op(OpCode::Finish, &[result]);
    }

    /// Whether the trace has exceeded the maximum allowed length.
    pub fn is_too_long(&self) -> bool {
        self.recorder.is_too_long()
    }

    /// Current trace limit (for diagnostics).
    pub fn trace_limit(&self) -> usize {
        self.recorder.trace_limit()
    }

    /// pyjitpl.py:1618 force_finish_trace flag.
    pub fn force_finish_trace(&self) -> bool {
        self.force_finish
    }

    /// Set force_finish_trace flag.
    pub fn set_force_finish(&mut self, val: bool) {
        self.force_finish = val;
    }

    /// Get the result type of an OpRef from the recorded trace.
    /// RPython parity: boxes carry their own type. Here we check
    /// inputargs, constant pool (constant_type + numbering overrides),
    /// and recorded ops to determine the type.
    ///
    /// resoperation.py Box.type parity: a Box's type is always one of
    /// `'i'` / `'r'` / `'f'`. Void is NOT a valid Box type — only
    /// value-producing ops have Boxes. pyre's recorder assigns `pos`
    /// to every op (including void ops like SetfieldGc and guards),
    /// so a stale lookup of a void op's pos would otherwise return
    /// `Type::Void`. Filter that case out and return `None` so callers
    /// fall back to a safe default rather than letting Void leak into
    /// `snapshot_box_types` / `livebox_types` / `fail_arg_types`.
    pub fn get_opref_type(&self, opref: OpRef) -> Option<Type> {
        if (opref.0 as usize) < self.recorder.num_inputargs() {
            return Some(self.recorder.inputarg_types()[opref.0 as usize]);
        }
        // ConstantPool: check constant_type first, then numbering
        // type overrides (mark_type for resume-data-only Ref constants).
        if opref.is_constant() {
            if let Some(tp) = self.constants.constant_type(opref) {
                return Some(tp);
            }
            if let Some(&tp) = self.constants.numbering_type_overrides().get(&opref.0) {
                return Some(tp);
            }
        }
        self.recorder
            .get_op_by_pos(opref)
            .map(|op| op.result_type())
            .filter(|tp| *tp != Type::Void)
    }

    /// The green key hash (loop header PC) for this trace.
    pub fn green_key(&self) -> u64 {
        self.green_key
    }

    /// Root portal merge-point green key for this trace.
    ///
    /// Mirrors RPython's `current_merge_points[0]`: this stays anchored to the
    /// original loop/portal merge point even if `green_key` is later retargeted
    /// to a reached loop header during tracing.
    pub fn root_green_key(&self) -> u64 {
        self.root_green_key
    }

    /// Number of input arguments to the current trace.
    pub fn num_inputs(&self) -> usize {
        self.recorder.num_inputargs()
    }

    /// Input argument types in loop-header order.
    pub fn inputarg_types(&self) -> Vec<Type> {
        self.recorder.inputarg_types()
    }

    /// Number of traced operations recorded so far.
    pub fn num_ops(&self) -> usize {
        self.recorder.num_ops()
    }

    /// The structured green key values, if provided.
    pub fn green_key_values(&self) -> Option<&GreenKey> {
        self.green_key_values.as_ref()
    }

    /// Set the structured green key values.
    pub fn set_green_key_values(&mut self, values: GreenKey) {
        self.green_key_values = Some(values);
    }

    /// The declarative JitDriver descriptor, if provided.
    pub fn driver_descriptor(&self) -> Option<&JitDriverStaticData> {
        self.driver_descriptor.as_ref()
    }

    /// Attach declarative JitDriver metadata to the active trace.
    pub fn set_driver_descriptor(&mut self, descriptor: JitDriverStaticData) {
        self.driver_descriptor = Some(descriptor);
    }

    /// Record a promote: emit GuardValue to specialize on a runtime value.
    ///
    /// In RPython this is `jit.promote(x)` — it records a `GUARD_VALUE`
    /// that asserts the runtime value equals the constant captured during
    /// tracing. After the guard, the optimizer treats the value as constant.
    ///
    /// `opref` is the traced value, `runtime_value` is the current concrete
    /// value seen at trace time.
    pub fn promote_int(&mut self, opref: OpRef, runtime_value: i64, num_live: usize) -> OpRef {
        let const_ref = self.const_int(runtime_value);
        self.record_guard(OpCode::GuardValue, &[opref, const_ref], num_live);
        const_ref
    }

    /// Record a ref-typed promote (GUARD_VALUE for GC references).
    pub fn promote_ref(&mut self, opref: OpRef, runtime_value: i64, num_live: usize) -> OpRef {
        let const_ref = self.const_ref(runtime_value);
        self.record_guard(OpCode::GuardValue, &[opref, const_ref], num_live);
        const_ref
    }

    /// Record a float-typed promote (GUARD_VALUE for floats).
    ///
    /// pyjitpl.py:1515 opimpl_float_guard_value = _opimpl_guard_value
    pub fn promote_float(&mut self, opref: OpRef, runtime_value: i64, num_live: usize) -> OpRef {
        let const_ref = self.const_float(runtime_value);
        self.record_guard(OpCode::GuardValue, &[opref, const_ref], num_live);
        const_ref
    }

    /// Record a call to an elidable (pure) function.
    ///
    /// In RPython, `@jit.elidable` marks a function whose result depends
    /// only on its arguments and has no side effects. The optimizer can
    /// constant-fold calls where all args are constants, or CSE identical calls.
    ///
    /// This records a CALL_PURE_I (or CALL_PURE_R/CALL_PURE_N) which the
    /// optimizer's pure pass can eliminate.
    pub fn call_elidable_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_elidable_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a void-returning call to a may-force function (e.g., one that
    /// may trigger GC or exceptions).
    ///
    /// In RPython this is `call_may_force` — a call that may force virtualizable
    /// frames or raise exceptions. Must be followed by `GUARD_NOT_FORCED`.
    pub fn call_may_force_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_int_typed(func_ptr, args, &arg_types)
    }

    /// Initialize standard virtualizable boxes from input args.
    /// Called at trace start when a virtualizable is registered.
    ///
    /// `input_oprefs` / `input_values` contain one (OpRef, Value) pair per
    /// static field + array element in the same flat layout as
    /// `VirtualizableInfo::get_index_in_array`. `vable_ref` / `vable_ref_value`
    /// are the OpRef and concrete of the virtualizable object (frame pointer).
    /// Boxes layout: [field0, ..., fieldN, arr[0], ..., arr[M], vable_ref]
    /// where `boxes[-1]` is the standard virtualizable identity (RPython parity).
    pub fn init_virtualizable_boxes(
        &mut self,
        info: &VirtualizableInfo,
        vable_ref: OpRef,
        vable_ref_value: Value,
        input_oprefs: &[OpRef],
        input_values: &[Value],
        array_lengths: &[usize],
    ) {
        let mut boxes = input_oprefs.to_vec();
        boxes.push(vable_ref); // RPython: virtualizable_boxes[-1] = vable identity
        self.virtualizable_boxes = Some(boxes);
        if input_values.is_empty() {
            // Caller has no live concrete values (e.g. bridge-entry rebuild
            // helper in pyre-jit-trace::state::seed_virtualizable_boxes).
            // Disable the concrete shadow; `virtualizable_entry_at` will
            // return None and readers fall back to the zero placeholder,
            // same as the pre-concrete-shadow state.
            self.virtualizable_values = None;
        } else {
            assert_eq!(
                input_oprefs.len(),
                input_values.len(),
                "init_virtualizable_boxes: OpRef and Value slices must match",
            );
            let mut values = input_values.to_vec();
            values.push(vable_ref_value);
            self.virtualizable_values = Some(values);
        }
        self.virtualizable_info = Some(info.clone());
        self.virtualizable_array_lengths = Some(array_lengths.to_vec());
    }

    /// Collect the current virtualizable boxes (for close_loop / finish).
    /// Returns `None` if no standard virtualizable is active.
    pub fn collect_virtualizable_boxes(&self) -> Option<Vec<OpRef>> {
        self.virtualizable_boxes.clone()
    }

    // (synchronize_virtualizable helper follows)

    /// Mirror of `MetaInterp::vable_ptr` used by `synchronize_virtualizable`.
    /// Callers set this at trace/bridge-entry so writes to
    /// `virtualizable_values` can propagate to the live PyFrame without
    /// routing back through MetaInterp (pyjitpl.py:3446 write_boxes target).
    pub fn set_virtualizable_heap_ptr(&mut self, ptr: *const u8) {
        self.virtualizable_heap_ptr = if ptr.is_null() { None } else { Some(ptr) };
    }

    /// pyjitpl.py:3446-3450 `synchronize_virtualizable()`.
    ///
    /// Writes the concrete half of `virtualizable_boxes` (the
    /// `virtualizable_values` shadow) back to the live virtualizable via
    /// `VirtualizableInfo::write_all_boxes`. The trailing identity slot
    /// (`virtualizable_boxes[-1]`) is excluded — RPython's `write_boxes`
    /// stops at `self.num_arrays + self.static_fields.len()` and leaves the
    /// identity untouched. No-op when the heap pointer, `virtualizable_info`,
    /// or `virtualizable_values` is unavailable.
    pub(crate) fn synchronize_virtualizable(&self) {
        let Some(heap_ptr) = self.virtualizable_heap_ptr else {
            return;
        };
        let Some(info) = self.virtualizable_info.as_ref() else {
            return;
        };
        let Some(values) = self.virtualizable_values.as_ref() else {
            return;
        };
        let Some(lengths) = self.virtualizable_array_lengths.as_ref() else {
            return;
        };
        let static_count = info.num_static_extra_boxes;
        if values.len() < static_count {
            return;
        }
        let mut static_bits: Vec<i64> = Vec::with_capacity(static_count);
        for v in &values[..static_count] {
            static_bits.push(value_to_raw_bits(*v));
        }
        let mut array_bits: Vec<Vec<i64>> = Vec::with_capacity(lengths.len());
        let mut cursor = static_count;
        for &len in lengths {
            if cursor + len > values.len() {
                return;
            }
            let mut items: Vec<i64> = Vec::with_capacity(len);
            for v in &values[cursor..cursor + len] {
                items.push(value_to_raw_bits(*v));
            }
            array_bits.push(items);
            cursor += len;
        }
        // Safety: `heap_ptr` is cached at trace/bridge entry from
        // `MetaInterp::vable_ptr`, which the JitState pins for the trace
        // session's lifetime. `write_all_boxes` uses typed offsets derived
        // from the same VirtualizableInfo used at the matching heap read.
        unsafe {
            info.write_all_boxes(heap_ptr as *mut u8, &static_bits, &array_bits);
        }
    }

    /// Read a standard virtualizable box by flat index.
    ///
    /// The last slot is the standard virtualizable identity itself
    /// (`virtualizable_boxes[-1]` in RPython terms).
    pub fn virtualizable_box_at(&self, index: usize) -> Option<OpRef> {
        self.virtualizable_boxes
            .as_ref()
            .and_then(|boxes| boxes.get(index).copied())
    }

    /// Read a standard virtualizable slot as (OpRef, concrete Value) — RPython
    /// `virtualizable_boxes[index]` parity: a Box carries both the traced
    /// reference and its concrete value. Callers that need to seed a register
    /// with both halves of the Box (e.g. `BC_GETARRAYITEM_VABLE_R` →
    /// `set_ref_reg`) MUST use this instead of `virtualizable_box_at`.
    pub fn virtualizable_entry_at(&self, index: usize) -> Option<(OpRef, Value)> {
        let boxes = self.virtualizable_boxes.as_ref()?;
        let values = self.virtualizable_values.as_ref()?;
        let opref = *boxes.get(index)?;
        let value = *values.get(index)?;
        Some((opref, value))
    }

    /// Declared majit_ir::Type for a flat virtualizable slot.
    ///
    /// Mirrors the layout used by `initialize_virtualizable`: the first
    /// `num_static_extra_boxes` slots take their types from
    /// `VirtualizableInfo.static_fields[i].field_type`, subsequent array
    /// slots take `array_fields[a].item_type`, and the trailing identity
    /// slot (`virtualizable_boxes[-1]`) is always `Ref`.  Returns `None`
    /// when no VirtualizableInfo is registered or the index falls outside
    /// the active layout.
    pub fn virtualizable_slot_type(&self, flat_idx: usize) -> Option<Type> {
        let info = self.virtualizable_info.as_ref()?;
        let lengths = self.virtualizable_array_lengths.as_deref().unwrap_or(&[]);
        let total_array: usize = lengths.iter().sum();
        let static_count = info.num_static_extra_boxes;
        if flat_idx < static_count {
            return Some(info.static_fields[flat_idx].field_type);
        }
        let array_local_idx = flat_idx - static_count;
        if array_local_idx < total_array {
            let mut remaining = array_local_idx;
            for (a, &len) in lengths.iter().enumerate() {
                if remaining < len {
                    return Some(info.array_fields[a].item_type);
                }
                remaining -= len;
            }
        }
        if flat_idx == static_count + total_array {
            // virtualizable_boxes[-1] — the identity slot.
            return Some(Type::Ref);
        }
        None
    }

    /// Update a standard virtualizable box (OpRef) by flat index.
    ///
    /// Used by SameAs dedup / `replace_box` walks — SSA-rename operations that
    /// do NOT change the concrete value carried by the slot. For updates that
    /// also change concrete (vable set{field,arrayitem}), use
    /// `set_virtualizable_entry_at`.
    pub fn set_virtualizable_box_at(&mut self, index: usize, value: OpRef) -> bool {
        if let Some(boxes) = &mut self.virtualizable_boxes {
            if let Some(slot) = boxes.get_mut(index) {
                *slot = value;
                return true;
            }
        }
        false
    }

    /// Update both halves of a standard virtualizable slot (OpRef + concrete).
    ///
    /// pyjitpl.py:1237 parity:
    ///
    /// ```text
    ///     self.metainterp.virtualizable_boxes[index] = valuebox
    ///     self.metainterp.synchronize_virtualizable()
    /// ```
    ///
    /// Writes the entire Box (SSA identity + concrete value) atomically so
    /// the (OpRef, concrete) pair never diverges.  Callers must ensure
    /// `value.get_type()` matches the slot's declared type
    /// (`virtualizable_slot_type(index)`); RPython guarantees this at the
    /// source level by emitting `NEW_W_INT` / `NEW_W_FLOAT` before any
    /// STORE into a Ref-typed `locals_cells_stack_w` slot
    /// (pypy/interpreter/pyframe.py:84 `list[W_Object]`).  Pyre's codewriter
    /// does not yet mirror that boxing at STORE_FAST → vable (Phase 4-5 of
    /// the portal-locals lowering plan); until it does, non-Phase-D paths
    /// like `pyre::trace_opcode::store_local_value` may write a pyre-unboxed
    /// `Value::Int`/`Value::Float` into a Ref slot and a later
    /// `BC_GETARRAYITEM_VABLE_R` read will decode 0 via `value_as_ref_bits`.
    /// That null is a pyre-upstream parity gap, not a shadow bug — the
    /// shadow faithfully reflects the caller's Box.
    pub fn set_virtualizable_entry_at(&mut self, index: usize, opref: OpRef, value: Value) -> bool {
        let have_boxes = self
            .virtualizable_boxes
            .as_mut()
            .and_then(|boxes| boxes.get_mut(index).map(|slot| *slot = opref))
            .is_some();
        let have_values = self
            .virtualizable_values
            .as_mut()
            .and_then(|vals| vals.get_mut(index).map(|slot| *slot = value))
            .is_some();
        have_boxes && have_values
    }

    /// Return the standard virtualizable identity (`virtualizable_boxes[-1]`).
    pub fn standard_virtualizable_box(&self) -> Option<OpRef> {
        self.virtualizable_boxes
            .as_ref()
            .and_then(|boxes| boxes.last().copied())
    }

    /// Whether standard virtualizable boxes are active.
    pub fn has_virtualizable_boxes(&self) -> bool {
        self.virtualizable_boxes.is_some()
    }

    /// Drop the tracing-time virtualizable_boxes mirror.
    ///
    /// Used at bridge entry: `init_symbolic` seeds the cache with OpRefs
    /// derived from the *parent* loop's `vable_array_base`, but the
    /// bridge owns a fresh inputarg stream (its own `OpRef(0..N)` bound
    /// to parent-guard fail_args). Keeping the parent seed makes
    /// subsequent `vable_getarrayitem_*` / `vable_setarrayitem_*` reads
    /// return stale parent-loop OpRefs; clearing forces the vable path
    /// to fall through to the raw `GetarrayitemGc` / `SetarrayitemGc`
    /// (`ctx.has_virtualizable_boxes() == false` branch) until the
    /// bridge itself reseeds via resume data — matching
    /// rpython/jit/metainterp/pyjitpl.py:3400-3430 where the
    /// `virtualizable_boxes` are rebuilt from the guard's resume data
    /// before the bridge replays any vable op.
    pub fn clear_virtualizable_boxes(&mut self) {
        self.virtualizable_boxes = None;
    }

    /// Set virtualizable_boxes with VirtualizableInfo and array lengths.
    /// Used by bridge tracing where the boxes are reconstructed from
    /// resume data (pyjitpl.py:3400 rebuild_state_after_failure parity).
    ///
    /// `values` carries the concrete shadow that parallels `boxes`. Callers
    /// must pass the matching live values recovered from the guard's fail
    /// args; an empty `values` slice disables the concrete shadow for the
    /// duration of the bridge (only safe when the bridge does not execute
    /// any `BC_GET*_VABLE_*` opcodes that feed `set_*_reg`).
    pub fn set_virtualizable_boxes_with_info(
        &mut self,
        boxes: Vec<OpRef>,
        values: Vec<Value>,
        info: &VirtualizableInfo,
        array_lengths: &[usize],
    ) {
        if !values.is_empty() {
            assert_eq!(
                boxes.len(),
                values.len(),
                "set_virtualizable_boxes_with_info: boxes/values length mismatch",
            );
            self.virtualizable_values = Some(values);
        } else {
            self.virtualizable_values = None;
        }
        self.virtualizable_boxes = Some(boxes);
        self.virtualizable_info = Some(info.clone());
        self.virtualizable_array_lengths = Some(array_lengths.to_vec());
    }

    /// Canonical virtualizable metadata for the active standard virtualizable.
    pub fn virtualizable_info(&self) -> Option<&VirtualizableInfo> {
        self.virtualizable_info.as_ref()
    }

    /// Cached array lengths for the active standard virtualizable.
    pub fn virtualizable_array_lengths(&self) -> Option<&[usize]> {
        self.virtualizable_array_lengths.as_deref()
    }

    /// pyjitpl.py:2394 `forced_virtualizable` accessor.
    pub fn forced_virtualizable(&self) -> Option<OpRef> {
        self.forced_virtualizable
    }

    /// pyjitpl.py:1126-1127 / 3478 `forced_virtualizable` mutator.
    pub fn set_forced_virtualizable(&mut self, value: Option<OpRef>) {
        self.forced_virtualizable = value;
    }

    // ── hint API consumption (RPython annotator/codewriter equivalent) ──

    /// Consume `hint(frame, access_directly=True)` during tracing.
    ///
    /// RPython's annotator generates JitCode that bypasses heap ops for
    /// virtualizable fields. In majit, this initializes the standard
    /// virtualizable boxes model so that subsequent vable_getfield/setfield
    /// calls access boxes directly instead of emitting heap ops.
    ///
    /// Must be called after `init_virtualizable_boxes`.
    /// Returns `true` if standard access is now active.
    pub fn hint_access_directly(&self) -> bool {
        self.virtualizable_boxes.is_some()
    }

    /// Consume `hint(frame, fresh_virtualizable=True)` during tracing.
    ///
    /// Marks that the virtualizable was freshly allocated, so its token is
    /// guaranteed to be TOKEN_NONE. The tracer skips token-check preamble.
    /// No IR is emitted; this is a tracing-time optimization.
    pub fn hint_fresh_virtualizable(&mut self, _vable_opref: OpRef) {
        // No IR needed — the token is already NONE for fresh objects.
        // This hint prevents the tracer from emitting unnecessary
        // GuardValue(token, 0) at loop entry for freshly created frames.
    }

    /// pyjitpl.py:3222-3236 `MetaInterp.store_token_in_vable()`.
    ///
    /// ```text
    /// def store_token_in_vable(self):
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is None:
    ///         return
    ///     vbox = self.virtualizable_boxes[-1]
    ///     if vbox is self.forced_virtualizable:
    ///         return # we already forced it by hand
    ///     # in case the force_token has not been recorded, record it here
    ///     # to make sure we know the virtualizable can be broken. However,
    ///     # the contents of the virtualizable should be generally correct
    ///     force_token = self.history.record0(rop.FORCE_TOKEN,
    ///                                        lltype.nullptr(llmemory.GCREF.TO))
    ///     self.history.record2(rop.SETFIELD_GC, vbox, force_token,
    ///                          None, descr=vinfo.vable_token_descr)
    ///     self.generate_guard(rop.GUARD_NOT_FORCED_2)
    /// ```
    pub fn store_token_in_vable_setfield(&mut self) -> bool {
        let info = match self.virtualizable_info.clone() {
            Some(info) => info,
            None => return false,
        };
        let vbox = match self.standard_virtualizable_box() {
            Some(b) => b,
            None => return false,
        };
        if self.forced_virtualizable == Some(vbox) {
            return false;
        }
        let force_token = self.recorder.record_op(OpCode::ForceToken, &[]);
        let token_descr = info.token_field_descr();
        self.vable_setfield_descr(vbox, force_token, token_descr);
        // pyjitpl.py:3236 self.generate_guard(rop.GUARD_NOT_FORCED_2)
        // is recorded by the caller via the proper guard generation
        // path (`MIFrame::generate_guard` in the pyre frontend) so the
        // guard captures fresh resumedata at the current framestack
        // position, matching RPython's gen_store_back_in_vable.
        true
    }

    /// pyjitpl.py:3465-3497 `MetaInterp.gen_store_back_in_vable(box)`.
    ///
    /// ```text
    /// def gen_store_back_in_vable(self, box):
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is not None:
    ///         # xxx only write back the fields really modified
    ///         vbox = self.virtualizable_boxes[-1]
    ///         if vbox is not box:
    ///             # ignore the hint on non-standard virtualizable
    ///             # specifically, ignore it on a virtual
    ///             return
    ///         if self.forced_virtualizable is not None:
    ///             # this can happen only in strange cases, but we don't care
    ///             # it was already forced
    ///             return
    ///         self.forced_virtualizable = vbox
    ///         ...emit SETFIELD_GC for each static field...
    ///         ...emit SETARRAYITEM_GC for each array item...
    ///         ...emit final SETFIELD_GC(vbox, NULL, vable_token_descr)...
    /// ```
    pub fn gen_store_back_in_vable(&mut self, vable_opref: OpRef) {
        let (info, boxes, lengths) = match (
            self.virtualizable_info.clone(),
            self.virtualizable_boxes.clone(),
            self.virtualizable_array_lengths.clone(),
        ) {
            (Some(info), Some(boxes), Some(lengths)) => (info, boxes, lengths),
            _ => return,
        };

        // pyjitpl.py:3469 vbox = self.virtualizable_boxes[-1]
        // pyjitpl.py:3470-3473 if vbox is not box: return  (ignore nonstandard)
        if boxes.last().copied() != Some(vable_opref) {
            return;
        }

        // pyjitpl.py:3474-3477 if forced_virtualizable is not None: return
        if self.forced_virtualizable.is_some() {
            return;
        }
        // pyjitpl.py:3478 self.forced_virtualizable = vbox
        self.forced_virtualizable = Some(vable_opref);

        for field_index in 0..info.static_fields.len() {
            if let Some(&value) = boxes.get(field_index) {
                let descr = info.static_field_descr(field_index);
                self.vable_setfield_descr(vable_opref, value, descr);
            }
        }

        let mut flat_box_index = info.static_fields.len();
        for array_index in 0..info.array_fields.len() {
            let len = lengths.get(array_index).copied().unwrap_or(0);
            let field_descr = info.array_pointer_field_descr(array_index);
            let array_descr = info.array_item_descr(array_index);
            let array_ref = self.vable_getfield_ref_descr(vable_opref, field_descr);
            for item_index in 0..len {
                if let Some(&value) = boxes.get(flat_box_index) {
                    let index = self.const_int(item_index as i64);
                    self.vable_setarrayitem_descr(array_ref, index, value, array_descr.clone());
                }
                flat_box_index += 1;
            }
        }

        let null = self.const_int(0);
        self.vable_setfield_descr(vable_opref, null, info.token_field_descr());
    }

    /// pyjitpl.py:1120-1146 `_nonstandard_virtualizable(pc, box, fielddescr)`.
    ///
    /// ```text
    ///  def _nonstandard_virtualizable(self, pc, box, fielddescr):
    ///      # returns True if 'box' is actually not the "standard" virtualizable
    ///      # that is stored in metainterp.virtualizable_boxes[-1]
    ///      if self.metainterp.heapcache.is_known_nonstandard_virtualizable(box):
    ///          self.metainterp.staticdata.profiler.count_ops(rop.PTR_EQ, Counters.HEAPCACHED_OPS)
    ///          return True
    ///      if box is self.metainterp.forced_virtualizable:
    ///          self.metainterp.forced_virtualizable = None
    ///      if (self.metainterp.jitdriver_sd.virtualizable_info is not None or
    ///          self.metainterp.jitdriver_sd.greenfield_info is not None):
    ///          standard_box = self.metainterp.virtualizable_boxes[-1]
    ///          if standard_box is box:
    ///              return False
    ///          vinfo = self.metainterp.jitdriver_sd.virtualizable_info
    ///          if vinfo is fielddescr.get_vinfo():
    ///              eqbox = self.metainterp.execute_and_record(rop.PTR_EQ, None,
    ///                                                         box, standard_box)
    ///              eqbox = self.implement_guard_value(eqbox, pc)
    ///              isstandard = eqbox.getint()
    ///              if isstandard:
    ///                  if box.type == 'r':
    ///                      self.metainterp.replace_box(box, standard_box)
    ///                  return False
    ///      if not self.metainterp.heapcache.is_unescaped(box):
    ///          self.emit_force_virtualizable(fielddescr, box)
    ///      self.metainterp.heapcache.nonstandard_virtualizables_now_known(box)
    ///      return True
    /// ```
    ///
    /// In pyre this is the LIVE entry path used by the jitcode machine
    /// (`vable_*_indexed`) at trace time. The pyjitpl::nonstandard_virtualizable
    /// duplicate is reachable only from the legacy `opimpl_*_vable` test
    /// surface. The two implementations carry the same line-by-line shape so
    /// the structural divergence is duplication-only — fixing the type-tag
    /// refactor will let us collapse them into a single entry point.
    fn is_nonstandard_virtualizable(&mut self, vable_opref: OpRef) -> bool {
        // Step 1: heapcache short-circuit.
        //     if self.metainterp.heapcache.is_known_nonstandard_virtualizable(box):
        //         return True
        if self
            .heap_cache
            .is_known_nonstandard_virtualizable(vable_opref)
        {
            return true;
        }
        // Step 2: forced_virtualizable reset on identity.
        //     if box is self.metainterp.forced_virtualizable:
        //         self.metainterp.forced_virtualizable = None
        if self.forced_virtualizable == Some(vable_opref) {
            self.forced_virtualizable = None;
        }
        // Step 3: standard_box identity check.
        //     standard_box = self.metainterp.virtualizable_boxes[-1]
        //     if standard_box is box:
        //         return False
        let standard_box = self
            .virtualizable_boxes
            .as_ref()
            .and_then(|boxes| boxes.last().copied());
        let Some(standard_box) = standard_box else {
            // No boxes → treat as nonstandard.
            return true;
        };
        if standard_box == vable_opref {
            return false;
        }
        // Step 4: PTR_EQ + implement_guard_value + replace_box.
        //     vinfo = self.metainterp.jitdriver_sd.virtualizable_info
        //     if vinfo is fielddescr.get_vinfo():
        //         eqbox = self.metainterp.execute_and_record(
        //             rop.PTR_EQ, None, box, standard_box)
        //         eqbox = self.implement_guard_value(eqbox, pc)
        //         isstandard = eqbox.getint()
        //         if isstandard:
        //             if box.type == 'r':
        //                 self.metainterp.replace_box(box, standard_box)
        //             return False
        //
        // Pyre's vable_*_indexed always passes
        // `standard_virtualizable_box()` as `vable_opref`, so Step 3 above
        // always returns `false` and Step 4 is structurally unreachable in
        // the live path. Emit the PTR_EQ + GUARD_VALUE shape so a future
        // alias-producing caller traps into bridge compilation rather than
        // silently misclassifying the box. The hardcoded `isstandard = 0`
        // encodes pyre's architectural invariant that distinct OpRefs map
        // to distinct concrete pointers; the conditional `replace_box` is
        // emitted line-by-line so a future alias path lights it up.
        let eqbox = self.record_op(OpCode::PtrEq, &[vable_opref, standard_box]);
        let isstandard: i64 = 0;
        let const_isstandard = self.const_int(isstandard);
        self.record_guard(OpCode::GuardValue, &[eqbox, const_isstandard], 0);
        #[allow(unreachable_code)]
        if isstandard != 0 {
            // box.type == 'r' (virtualizables are Ref).
            // pyjitpl.py:1141 `self.metainterp.replace_box(box, standard_box)`.
            // The trace_ctx-side replace_box runs the
            // virtualizable_boxes / heap_cache walks AND queues the
            // recorder rewrite. The framestack / virtualref_boxes walk
            // lives on `MetaInterp::replace_box` (pyjitpl.rs); both
            // share this single helper for the trace-context portion.
            self.replace_box(vable_opref, standard_box);
            return false;
        }
        // Step 5a: emit_force_virtualizable.
        //     if not self.metainterp.heapcache.is_unescaped(box):
        //         self.emit_force_virtualizable(fielddescr, box)
        //
        //     def emit_force_virtualizable(self, fielddescr, box):
        //         vinfo = fielddescr.get_vinfo()
        //         token_descr = vinfo.vable_token_descr
        //         tokenbox = mi.execute_and_record(
        //             rop.GETFIELD_GC_R, token_descr, box)
        //         condbox = mi.execute_and_record(
        //             rop.PTR_NE, None, tokenbox, CONST_NULL)
        //         funcbox = ConstInt(rffi.cast(Signed, vinfo.clear_vable_ptr))
        //         self.execute_varargs(
        //             rop.COND_CALL, [condbox, funcbox, box],
        //             vinfo.clear_vable_descr, False, False)
        if !self.heap_cache.is_unescaped(vable_opref) {
            if let Some(info) = self.virtualizable_info.clone() {
                let token_descr = info.token_field_descr();
                let tokenbox =
                    self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], token_descr);
                let null_ref = self.const_null();
                let condbox = self.record_op(OpCode::PtrNe, &[tokenbox, null_ref]);
                if let (Some(clear_fn), Some(clear_descr)) =
                    (info.clear_vable_fn, info.clear_vable_descr.clone())
                {
                    let funcbox = self.const_int(clear_fn as usize as i64);
                    self.recorder.record_op_with_descr(
                        OpCode::CondCallN,
                        &[condbox, funcbox, vable_opref],
                        clear_descr,
                    );
                }
            }
        }
        // Step 5b: mark this box as a known nonstandard virtualizable so
        // future accesses short-circuit at Step 1.
        //     self.metainterp.heapcache.nonstandard_virtualizables_now_known(box)
        self.heap_cache
            .nonstandard_virtualizables_now_known(vable_opref);
        true
    }

    /// pyjitpl.py:1167-1172 `opimpl_getfield_vable_i(box, fielddescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_getfield_vable_i(self, box, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self.opimpl_getfield_gc_i(box, fielddescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    pub fn vable_getfield_int(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        if self.is_nonstandard_virtualizable(vable_opref) {
            // self.opimpl_getfield_gc_i(box, fielddescr)
            let op = self.record_op_with_descr(OpCode::GetfieldGcI, &[vable_opref], fielddescr);
            return (op, Value::Void);
        }
        // self.metainterp.check_synchronized_virtualizable() — no-op in pyre.
        // index = self._get_virtualizable_field_index(fielddescr)
        // return self.metainterp.virtualizable_boxes[index]
        let index = self
            .virtualizable_info
            .as_ref()
            .and_then(|info| info.static_field_by_descr(&fielddescr));
        if let Some(idx) = index {
            if let Some(entry) = self.virtualizable_entry_at(idx) {
                return entry;
            }
        }
        // Fallback for tests/missing layout
        let op = self.record_op_with_descr(OpCode::GetfieldGcI, &[vable_opref], fielddescr);
        (op, Value::Void)
    }

    /// Record a virtualizable field read with an explicit field descriptor.
    pub fn vable_getfield_int_descr(&mut self, vable_opref: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::GetfieldGcI, &[vable_opref], descr)
    }

    /// pyjitpl.py:1188-1199 `_opimpl_setfield_vable(box, valuebox, fielddescr, pc)`.
    ///
    /// ```text
    ///  def _opimpl_setfield_vable(self, box, valuebox, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self._opimpl_setfield_gc_any(box, valuebox, fielddescr)
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      self.metainterp.virtualizable_boxes[index] = valuebox
    ///      self.metainterp.synchronize_virtualizable()
    ///      # XXX only the index'th field needs to be synchronized, really
    /// ```
    pub fn vable_setfield(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        if self.is_nonstandard_virtualizable(vable_opref) {
            // self._opimpl_setfield_gc_any(box, valuebox, fielddescr)
            self.record_op_with_descr(OpCode::SetfieldGc, &[vable_opref, value], fielddescr);
            return;
        }
        // index = self._get_virtualizable_field_index(fielddescr)
        // self.metainterp.virtualizable_boxes[index] = valuebox
        let index = self
            .virtualizable_info
            .as_ref()
            .and_then(|info| info.static_field_by_descr(&fielddescr));
        if let Some(idx) = index {
            if self.set_virtualizable_entry_at(idx, value, concrete) {
                // pyjitpl.py:3446 write_boxes parity: mirror the updated
                // shadow slot back into the live virtualizable.
                self.synchronize_virtualizable();
                return;
            }
        }
        // Fallback: emit the heap op when the layout is unavailable.
        self.record_op_with_descr(OpCode::SetfieldGc, &[vable_opref, value], fielddescr);
    }

    /// Record a virtualizable field write with an explicit field descriptor.
    pub fn vable_setfield_descr(&mut self, vable_opref: OpRef, value: OpRef, descr: DescrRef) {
        self.record_op_with_descr(OpCode::SetfieldGc, &[vable_opref, value], descr);
    }

    /// pyjitpl.py:1173-1179 `opimpl_getfield_vable_r(box, fielddescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_getfield_vable_r(self, box, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self.opimpl_getfield_gc_r(box, fielddescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    pub fn vable_getfield_ref(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        if self.is_nonstandard_virtualizable(vable_opref) {
            let op = self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fielddescr);
            return (op, Value::Void);
        }
        let index = self
            .virtualizable_info
            .as_ref()
            .and_then(|info| info.static_field_by_descr(&fielddescr));
        if let Some(idx) = index {
            if let Some(entry) = self.virtualizable_entry_at(idx) {
                return entry;
            }
        }
        let op = self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fielddescr);
        (op, Value::Void)
    }

    /// Record a virtualizable ref field read with an explicit field descriptor.
    pub fn vable_getfield_ref_descr(&mut self, vable_opref: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], descr)
    }

    /// pyjitpl.py:1180-1186 `opimpl_getfield_vable_f(box, fielddescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_getfield_vable_f(self, box, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self.opimpl_getfield_gc_f(box, fielddescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    pub fn vable_getfield_float(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        if self.is_nonstandard_virtualizable(vable_opref) {
            let op = self.record_op_with_descr(OpCode::GetfieldGcF, &[vable_opref], fielddescr);
            return (op, Value::Void);
        }
        let index = self
            .virtualizable_info
            .as_ref()
            .and_then(|info| info.static_field_by_descr(&fielddescr));
        if let Some(idx) = index {
            if let Some(entry) = self.virtualizable_entry_at(idx) {
                return entry;
            }
        }
        let op = self.record_op_with_descr(OpCode::GetfieldGcF, &[vable_opref], fielddescr);
        (op, Value::Void)
    }

    /// Record a virtualizable array item read (GETARRAYITEM_GC_I).
    pub fn vable_getarrayitem_int(&mut self, array_opref: OpRef, index: OpRef) -> OpRef {
        let zero = self.const_int(0); // descr placeholder
        self.record_op(OpCode::GetarrayitemGcI, &[array_opref, index, zero])
    }

    /// Standard virtualizable array item read (int).
    /// `array_field_offset` identifies which array field, `item_index` is the element index.
    /// If standard boxes are active, reads from the flat box array directly.
    pub fn vable_getarrayitem_int_vable(
        &mut self,
        array_opref: OpRef,
        fdescr: &DescrRef,
        item_index: usize,
    ) -> (OpRef, Value) {
        if let Some(flat_idx) = self.vable_array_flat_index(fdescr, item_index) {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        let index = self.const_int(item_index as i64);
        let zero = self.const_int(0);
        let op = self.record_op(OpCode::GetarrayitemGcI, &[array_opref, index, zero]);
        (op, Value::Void)
    }

    /// pyjitpl.py:1201-1216 `_get_arrayitem_vable_index(pc, arrayfielddescr, indexbox)`.
    ///
    /// ```text
    ///  def _get_arrayitem_vable_index(self, pc, arrayfielddescr, indexbox):
    ///      indexbox = self.implement_guard_value(indexbox, pc)
    ///      vinfo = self.metainterp.jitdriver_sd.virtualizable_info
    ///      virtualizable_box = self.metainterp.virtualizable_boxes[-1]
    ///      virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///      arrayindex = vinfo.array_field_by_descrs[arrayfielddescr]
    ///      index = indexbox.getint()
    ///      assert 0 <= index < vinfo.get_array_length(virtualizable, arrayindex)
    ///      return vinfo.get_index_in_array(virtualizable, arrayindex, index)
    /// ```
    fn get_arrayitem_vable_index(
        &mut self,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: &DescrRef,
    ) -> Option<usize> {
        // indexbox = self.implement_guard_value(indexbox, pc)
        let promoted_index = if index.is_constant() {
            index
        } else {
            self.promote_int(index, index_runtime_value, 0)
        };
        let _ = promoted_index;
        let item_index = usize::try_from(index_runtime_value).ok()?;
        // arrayindex = vinfo.array_field_by_descrs[arrayfielddescr]
        // assert 0 <= index < vinfo.get_array_length(virtualizable, arrayindex)
        // return vinfo.get_index_in_array(virtualizable, arrayindex, index)
        self.vable_array_flat_index(fdescr, item_index)
    }

    /// pyjitpl.py:1218-1230 `_opimpl_getarrayitem_vable(box, indexbox, fdescr, adescr, pc)`
    /// (int variant via `opimpl_getarrayitem_vable_i = _opimpl_getarrayitem_vable`).
    ///
    /// ```text
    ///  def _opimpl_getarrayitem_vable(self, box, indexbox, fdescr, adescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fdescr):
    ///          arraybox = self.opimpl_getfield_gc_r(box, fdescr)
    ///          ...
    ///          return self.opimpl_getarrayitem_gc_i(arraybox, indexbox, adescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_arrayitem_vable_index(pc, fdescr, indexbox)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    pub fn vable_getarrayitem_int_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
    ) -> (OpRef, Value) {
        if self.is_nonstandard_virtualizable(vable_opref) {
            // arraybox = self.opimpl_getfield_gc_r(box, fdescr)
            // return self.opimpl_getarrayitem_gc_i(arraybox, indexbox, adescr)
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
            return (self.vable_getarrayitem_int(array_opref, index), Value::Void);
        }
        // index = self._get_arrayitem_vable_index(pc, fdescr, indexbox)
        // return self.metainterp.virtualizable_boxes[index]
        if let Some(flat_idx) = self.get_arrayitem_vable_index(index, index_runtime_value, &fdescr)
        {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        // Fallback: vable layout missing — go through getfield + arrayitem.
        let array_opref =
            self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr.clone());
        if let Ok(item_index) = usize::try_from(index_runtime_value) {
            self.vable_getarrayitem_int_vable(array_opref, &fdescr, item_index)
        } else {
            (self.vable_getarrayitem_int(array_opref, index), Value::Void)
        }
    }

    /// Standard virtualizable array item read (ref).
    pub fn vable_getarrayitem_ref_vable(
        &mut self,
        array_opref: OpRef,
        fdescr: &DescrRef,
        item_index: usize,
    ) -> (OpRef, Value) {
        if let Some(flat_idx) = self.vable_array_flat_index(fdescr, item_index) {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        let index = self.const_int(item_index as i64);
        let zero = self.const_int(0);
        let op = self.record_op(OpCode::GetarrayitemGcR, &[array_opref, index, zero]);
        (op, Value::Void)
    }

    /// pyjitpl.py:1218-1234 `_opimpl_getarrayitem_vable` — ref variant.
    pub fn vable_getarrayitem_ref_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
    ) -> (OpRef, Value) {
        if self.is_nonstandard_virtualizable(vable_opref) {
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
            return (self.vable_getarrayitem_ref(array_opref, index), Value::Void);
        }
        if let Some(flat_idx) = self.get_arrayitem_vable_index(index, index_runtime_value, &fdescr)
        {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        let array_opref =
            self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr.clone());
        if let Ok(item_index) = usize::try_from(index_runtime_value) {
            self.vable_getarrayitem_ref_vable(array_opref, &fdescr, item_index)
        } else {
            (self.vable_getarrayitem_ref(array_opref, index), Value::Void)
        }
    }

    /// Standard virtualizable array item read (float).
    pub fn vable_getarrayitem_float_vable(
        &mut self,
        array_opref: OpRef,
        fdescr: &DescrRef,
        item_index: usize,
    ) -> (OpRef, Value) {
        if let Some(flat_idx) = self.vable_array_flat_index(fdescr, item_index) {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        let index = self.const_int(item_index as i64);
        let zero = self.const_int(0);
        let op = self.record_op(OpCode::GetarrayitemGcF, &[array_opref, index, zero]);
        (op, Value::Void)
    }

    /// pyjitpl.py:1218-1234 `_opimpl_getarrayitem_vable` — float variant.
    pub fn vable_getarrayitem_float_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
    ) -> (OpRef, Value) {
        if self.is_nonstandard_virtualizable(vable_opref) {
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
            return (
                self.vable_getarrayitem_float(array_opref, index),
                Value::Void,
            );
        }
        if let Some(flat_idx) = self.get_arrayitem_vable_index(index, index_runtime_value, &fdescr)
        {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        let array_opref =
            self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr.clone());
        if let Ok(item_index) = usize::try_from(index_runtime_value) {
            self.vable_getarrayitem_float_vable(array_opref, &fdescr, item_index)
        } else {
            (
                self.vable_getarrayitem_float(array_opref, index),
                Value::Void,
            )
        }
    }

    /// Standard virtualizable array item write.
    /// `array_field_offset` identifies which array field, `item_index` is the element index.
    /// If standard boxes are active, writes to the flat box array directly.
    pub fn vable_setarrayitem_vable(
        &mut self,
        array_opref: OpRef,
        fdescr: &DescrRef,
        item_index: usize,
        value: OpRef,
        concrete: Value,
    ) {
        let flat_idx = self.vable_array_flat_index(fdescr, item_index);
        if let Some(idx) = flat_idx {
            if self.set_virtualizable_entry_at(idx, value, concrete) {
                return;
            }
        }
        let index = self.const_int(item_index as i64);
        let zero = self.const_int(0);
        self.record_op(OpCode::SetarrayitemGc, &[array_opref, index, value, zero]);
    }

    /// pyjitpl.py:1236-1247 `_opimpl_setarrayitem_vable(box, indexbox, valuebox, fdescr, adescr, pc)`.
    pub fn vable_setarrayitem_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        if self.is_nonstandard_virtualizable(vable_opref) {
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
            self.vable_setarrayitem(array_opref, index, value);
            return;
        }
        if let Some(flat_idx) = self.get_arrayitem_vable_index(index, index_runtime_value, &fdescr)
        {
            if self.set_virtualizable_entry_at(flat_idx, value, concrete) {
                // pyjitpl.py:3446 write_boxes parity: mirror the updated
                // shadow slot back into the live virtualizable.
                self.synchronize_virtualizable();
                return;
            }
        }
        let array_opref =
            self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr.clone());
        if let Ok(item_index) = usize::try_from(index_runtime_value) {
            self.vable_setarrayitem_vable(array_opref, &fdescr, item_index, value, concrete);
        } else {
            self.vable_setarrayitem(array_opref, index, value);
        }
    }

    /// pyjitpl.py:1253-1263 `opimpl_arraylen_vable(box, fdescr, adescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_arraylen_vable(self, box, fdescr, adescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fdescr):
    ///          arraybox = self.opimpl_getfield_gc_r(box, fdescr)
    ///          return self.opimpl_arraylen_gc(arraybox, adescr)
    ///      vinfo = self.metainterp.jitdriver_sd.virtualizable_info
    ///      virtualizable_box = self.metainterp.virtualizable_boxes[-1]
    ///      virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///      arrayindex = vinfo.array_field_by_descrs[fdescr]
    ///      result = vinfo.get_array_length(virtualizable, arrayindex)
    ///      return ConstInt(result)
    /// ```
    pub fn vable_arraylen_vable(&mut self, vable_opref: OpRef, fdescr: DescrRef) -> OpRef {
        if self.is_nonstandard_virtualizable(vable_opref) {
            // arraybox = self.opimpl_getfield_gc_r(box, fdescr)
            // return self.opimpl_arraylen_gc(arraybox, adescr)
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
            return self.record_op(OpCode::ArraylenGc, &[array_opref]);
        }
        // arrayindex = vinfo.array_field_by_descrs[fdescr]
        // result = vinfo.get_array_length(virtualizable, arrayindex)
        // return ConstInt(result)
        if let (Some(info), Some(lengths)) =
            (&self.virtualizable_info, &self.virtualizable_array_lengths)
        {
            if let Some(array_idx) = info.array_field_by_descr(&fdescr) {
                if let Some(&length) = lengths.get(array_idx) {
                    return self.const_int(length as i64);
                }
            }
        }
        // Fallback when the layout is unavailable.
        let array_opref = self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
        self.record_op(OpCode::ArraylenGc, &[array_opref])
    }

    /// Compute the flat index into virtualizable_boxes for an array element.
    /// Returns `None` if standard virtualizable is not active or the array field is unknown.
    fn vable_array_flat_index(&self, fdescr: &DescrRef, item_index: usize) -> Option<usize> {
        let info = self.virtualizable_info.as_ref()?;
        let lengths = self.virtualizable_array_lengths.as_ref()?;
        let array_idx = info.array_field_by_descr(fdescr)?;
        Some(info.get_index_in_array(array_idx, item_index, lengths))
    }

    /// Record a virtualizable array item read with an explicit array descriptor.
    pub fn vable_getarrayitem_int_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        descr: DescrRef,
    ) -> OpRef {
        self.record_op_with_descr(OpCode::GetarrayitemGcI, &[array_opref, index], descr)
    }

    /// Record a virtualizable array item read (GETARRAYITEM_GC_R).
    pub fn vable_getarrayitem_ref(&mut self, array_opref: OpRef, index: OpRef) -> OpRef {
        let zero = self.const_int(0); // descr placeholder
        self.record_op(OpCode::GetarrayitemGcR, &[array_opref, index, zero])
    }

    /// Record a virtualizable array item read with an explicit array descriptor.
    pub fn vable_getarrayitem_ref_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        descr: DescrRef,
    ) -> OpRef {
        self.record_op_with_descr(OpCode::GetarrayitemGcR, &[array_opref, index], descr)
    }

    /// Record a virtualizable array item read (GETARRAYITEM_GC_F).
    pub fn vable_getarrayitem_float(&mut self, array_opref: OpRef, index: OpRef) -> OpRef {
        let zero = self.const_int(0); // descr placeholder
        self.record_op(OpCode::GetarrayitemGcF, &[array_opref, index, zero])
    }

    /// Record a virtualizable array item write (SETARRAYITEM_GC).
    pub fn vable_setarrayitem(&mut self, array_opref: OpRef, index: OpRef, value: OpRef) {
        let zero = self.const_int(0); // descr placeholder
        self.record_op(OpCode::SetarrayitemGc, &[array_opref, index, value, zero]);
    }

    /// Record a virtualizable array item write with an explicit array descriptor.
    pub fn vable_setarrayitem_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        value: OpRef,
        descr: DescrRef,
    ) {
        self.record_op_with_descr(OpCode::SetarrayitemGc, &[array_opref, index, value], descr);
    }

    /// Record a ref-returning call to a may-force function.
    pub fn call_may_force_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a void-returning call to a may-force function.
    pub fn call_may_force_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_void_typed(func_ptr, args, &arg_types);
    }

    /// Record a call with GIL release (for C extensions / external libs).
    ///
    /// In RPython this is `call_release_gil`. The GIL is released before the
    /// call and reacquired after. Used for long-running C functions.
    pub fn call_release_gil_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_release_gil_int_typed(func_ptr, args, &arg_types)
    }

    /// Record a call to a loop-invariant function.
    ///
    /// The result is cached for the duration of one loop iteration.
    /// In RPython, `@jit.loop_invariant` marks such functions.
    pub fn call_loopinvariant_int(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_loopinvariant_int_typed(func_ptr, args, &arg_types)
    }

    /// Record GUARD_NOT_FORCED (must follow a call_may_force).
    pub fn guard_not_forced(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNotForced, &[], num_live)
    }

    // ── CALL_MAY_FORCE with virtualizable synchronization ─────────

    fn call_may_force_with_jitstate_sync_impl<S, R>(
        &mut self,
        state: &S,
        num_live: usize,
        record_call: impl FnOnce(&mut Self) -> R,
    ) -> (R, crate::jit_state::ResidualVirtualizableSync)
    where
        S: crate::jit_state::JitState,
    {
        state.sync_virtualizable_before_residual_call(self);
        let result = record_call(self);
        let sync = state.sync_virtualizable_after_residual_call(self);
        if !sync.forced {
            self.guard_not_forced(num_live);
        }
        (result, sync)
    }

    /// Callback-based virtualizable sync for CALL_MAY_FORCE.
    ///
    /// Uses JitState's `sync_virtualizable_before/after_residual_call`
    /// methods to emit the appropriate SETFIELD/GETFIELD ops. This is
    /// the preferred API for interpreters that implement the JitState
    /// virtualizable sync hooks.
    ///
    /// Returns `(call_result, sync)` where `sync` reports any updated field
    /// OpRefs and whether the residual call forced the standard virtualizable.
    pub fn call_may_force_with_jitstate_sync_int<S: crate::jit_state::JitState>(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        state: &S,
        num_live: usize,
    ) -> (OpRef, crate::jit_state::ResidualVirtualizableSync) {
        self.call_may_force_with_jitstate_sync_impl(state, num_live, |ctx| {
            ctx.call_may_force_int_typed(func_ptr, args, arg_types)
        })
    }

    /// Ref-returning variant of [`call_may_force_with_jitstate_sync_int`].
    pub fn call_may_force_with_jitstate_sync_ref<S: crate::jit_state::JitState>(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        state: &S,
        num_live: usize,
    ) -> (OpRef, crate::jit_state::ResidualVirtualizableSync) {
        self.call_may_force_with_jitstate_sync_impl(state, num_live, |ctx| {
            ctx.call_may_force_ref_typed(func_ptr, args, arg_types)
        })
    }

    /// Float-returning variant of [`call_may_force_with_jitstate_sync_int`].
    pub fn call_may_force_with_jitstate_sync_float<S: crate::jit_state::JitState>(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        state: &S,
        num_live: usize,
    ) -> (OpRef, crate::jit_state::ResidualVirtualizableSync) {
        self.call_may_force_with_jitstate_sync_impl(state, num_live, |ctx| {
            ctx.call_may_force_float_typed(func_ptr, args, arg_types)
        })
    }

    /// Void-returning variant of [`call_may_force_with_jitstate_sync_int`].
    pub fn call_may_force_with_jitstate_sync_void<S: crate::jit_state::JitState>(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        state: &S,
        num_live: usize,
    ) -> crate::jit_state::ResidualVirtualizableSync {
        let (_, sync) = self.call_may_force_with_jitstate_sync_impl(state, num_live, |ctx| {
            ctx.call_may_force_void_typed(func_ptr, args, arg_types)
        });
        sync
    }

    /// Record GUARD_NO_EXCEPTION (check no pending exception).
    pub fn guard_no_exception(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNoException, &[], num_live)
    }

    /// Record GUARD_NOT_INVALIDATED (check loop not invalidated).
    pub fn guard_not_invalidated(&mut self, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardNotInvalidated, &[], num_live)
    }

    // ── Generic typed call ──────────────────────────────────────────

    /// Record a function call with explicit argument and return types.
    ///
    /// All type-specific call convenience methods delegate to this.
    /// `opcode` selects the call family (CallI/R/F/N, CallPureI/R/F/N, etc.).
    pub fn call_typed(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = make_call_descr(arg_types, ret_type);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr);
        // heapcache.py: cache invalidation based on effect level.
        // EF_ELIDABLE / EF_LOOPINVARIANT (CallPure*, CallLoopinvariant*):
        //   no invalidation at all.
        // EF_CAN_RAISE / EF_CANNOT_RAISE (Call*, CallMayForce*):
        //   escape args + invalidate escaped caches.
        // EF_RANDOM_EFFECTS: handled separately via invalidate_all_caches.
        if !opcode.is_call_pure() && !opcode.is_call_loopinvariant() {
            self.heap_cache._escape_argboxes(args);
            self.heap_cache.invalidate_caches_for_escaped();
        }
        result
    }

    pub fn call_void_typed(&mut self, func_ptr: *const (), args: &[OpRef], arg_types: &[Type]) {
        let _ = self.call_typed(OpCode::CallN, func_ptr, args, arg_types, Type::Void);
    }

    /// pyjitpl.py:3553-3579: record_result_of_call_pure.
    ///
    /// Patch a CALL into a CALL_PURE. Called after a pure call executes
    /// during tracing with no exception.
    ///
    /// `concrete_arg_values` contains the execution-time values for ALL
    /// args (pyjitpl.py:3572 `[executor.constant_from_op(a) for a in
    /// normargboxes]`). Used as the full cache key.
    pub fn record_result_of_call_pure(
        &mut self,
        op: OpRef,
        argboxes: &[OpRef],
        concrete_arg_values: &[Value],
        descr: DescrRef,
        patch_pos: TracePosition,
        opcode: OpCode,
        result_value: Value,
    ) -> OpRef {
        let resbox_as_const = result_value;
        // pyjitpl.py:3557-3561: COND_CALL_VALUE ignores the 'value' arg
        let is_cond_value = opcode.is_cond_call_value();
        let norm_start = if is_cond_value { 1 } else { 0 };
        let normargboxes = &argboxes[norm_start..];
        let norm_values = &concrete_arg_values[norm_start..];
        // pyjitpl.py:3562-3565: check if all args are Const
        let all_const = normargboxes
            .iter()
            .all(|arg| self.constants.get_value(*arg).is_some());
        if all_const {
            // pyjitpl.py:3566-3569: all-constants → cut the CALL
            self.recorder.cut(patch_pos);
            let const_opref = match resbox_as_const {
                Value::Int(v) => self.constants.get_or_insert(v),
                Value::Float(v) => self
                    .constants
                    .get_or_insert_typed(v.to_bits() as i64, Type::Float),
                Value::Ref(r) => self
                    .constants
                    .get_or_insert_typed(r.as_usize() as i64, Type::Ref),
                Value::Void => self.constants.get_or_insert(0),
            };
            return const_opref;
        }
        // pyjitpl.py:3572-3573: constant_from_op(a) for ALL args
        let arg_consts: Vec<Value> = norm_values.to_vec();
        self.call_pure_results.insert(arg_consts, resbox_as_const);
        // pyjitpl.py:3574-3575: COND_CALL_VALUE remains as-is
        if is_cond_value {
            return op;
        }
        // pyjitpl.py:3576-3579: cut CALL, re-record as CALL_PURE
        let ret_type = match resbox_as_const {
            Value::Int(_) => Type::Int,
            Value::Ref(_) => Type::Ref,
            Value::Float(_) => Type::Float,
            Value::Void => Type::Void,
        };
        let pure_opcode = OpCode::call_pure_for_type(ret_type);
        self.recorder.cut(patch_pos);
        self.recorder
            .record_op_with_descr(pure_opcode, argboxes, descr)
    }

    /// pyjitpl.py:2397 + compile.py:221: take call_pure_results for
    /// passing to the optimizer.
    pub fn take_call_pure_results(&mut self) -> std::collections::HashMap<Vec<Value>, Value> {
        std::mem::take(&mut self.call_pure_results)
    }

    // ── conditional_call / record_known_result (jtransform.py:1665, 292) ──

    /// RPython pyjitpl.py opimpl_conditional_call_ir_v: emit CondCallN.
    pub fn cond_call_void_typed(
        &mut self,
        condition: i64,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let cond_ref = self.constants.get_or_insert(condition);
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = make_call_descr(arg_types, Type::Void);
        let mut call_args = vec![cond_ref, func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CondCallN, &call_args, descr);
    }

    /// RPython pyjitpl.py opimpl_conditional_call_value_ir_i: emit CondCallValueI.
    pub fn cond_call_value_int_typed(
        &mut self,
        value: i64,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        let value_ref = self.constants.get_or_insert(value);
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = make_call_descr(arg_types, Type::Int);
        let mut call_args = vec![value_ref, func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CondCallValueI, &call_args, descr)
    }

    /// RPython pyjitpl.py opimpl_conditional_call_value_ir_r: emit CondCallValueR.
    pub fn cond_call_value_ref_typed(
        &mut self,
        value: i64,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        let value_ref = self.constants.get_or_insert(value);
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = make_call_descr(arg_types, Type::Ref);
        let mut call_args = vec![value_ref, func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op_with_descr(OpCode::CondCallValueR, &call_args, descr)
    }

    /// RPython pyjitpl.py opimpl_record_known_result_i / _r: emit RecordKnownResult.
    pub fn record_known_result_typed(
        &mut self,
        result_value: i64,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let result_ref = self.constants.get_or_insert(result_value);
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let _descr = make_call_descr(arg_types, Type::Void);
        let mut call_args = vec![result_ref, func_ref];
        call_args.extend_from_slice(args);
        self.recorder
            .record_op(OpCode::RecordKnownResult, &call_args);
    }

    pub fn call_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallI, func_ptr, args, arg_types, Type::Int)
    }

    pub fn call_elidable_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureI, func_ptr, args, arg_types, Type::Int)
    }

    // ── Ref/Float call variants ─────────────────────────────────────

    /// Record a ref-returning function call (CallR).
    pub fn call_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning function call (CallF).
    pub fn call_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_float_typed(func_ptr, args, &arg_types)
    }

    /// Record a ref-returning elidable (pure) call (CallPureR).
    pub fn call_elidable_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_elidable_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning elidable (pure) call (CallPureF).
    pub fn call_elidable_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_elidable_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallR, func_ptr, args, arg_types, Type::Ref)
    }

    pub fn call_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallF, func_ptr, args, arg_types, Type::Float)
    }

    pub fn call_elidable_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureR, func_ptr, args, arg_types, Type::Ref)
    }

    pub fn call_elidable_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_typed(OpCode::CallPureF, func_ptr, args, arg_types, Type::Float)
    }

    fn call_family_typed(
        &mut self,
        opcode: OpCode,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr = make_call_may_force_descr(arg_types, ret_type);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr);
        // heapcache.py: may-force calls escape args and invalidate caches.
        // (GuardNotForced after the call also invalidates, but we need
        // the escape marking here for correctness.)
        self.heap_cache._escape_argboxes(args);
        self.heap_cache.invalidate_caches_for_escaped();
        result
    }

    pub fn call_may_force_void_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Void),
            func_ptr,
            args,
            arg_types,
            Type::Void,
        );
    }

    pub fn call_may_force_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
        )
    }

    pub fn call_may_force_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Ref),
            func_ptr,
            args,
            arg_types,
            Type::Ref,
        )
    }

    /// Record a float-returning may-force call (CallMayForceF).
    pub fn call_may_force_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_may_force_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_may_force_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_may_force_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
        )
    }

    /// Record a void-returning GIL-release call (CallReleaseGilN).
    pub fn call_release_gil_void(&mut self, func_ptr: *const (), args: &[OpRef]) {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_release_gil_void_typed(func_ptr, args, &arg_types);
    }

    /// Record a ref-returning GIL-release call (CallReleaseGilR).
    pub fn call_release_gil_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_release_gil_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning GIL-release call (CallReleaseGilF).
    pub fn call_release_gil_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_release_gil_float_typed(func_ptr, args, &arg_types)
    }

    /// Record a ref-returning loop-invariant call (CallLoopinvariantR).
    pub fn call_loopinvariant_ref(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_loopinvariant_ref_typed(func_ptr, args, &arg_types)
    }

    /// Record a float-returning loop-invariant call (CallLoopinvariantF).
    pub fn call_loopinvariant_float(&mut self, func_ptr: *const (), args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_loopinvariant_float_typed(func_ptr, args, &arg_types)
    }

    pub fn call_release_gil_void_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_family_typed(
            OpCode::call_release_gil_for_type(Type::Void),
            func_ptr,
            args,
            arg_types,
            Type::Void,
        );
    }

    pub fn call_release_gil_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_release_gil_for_type(Type::Int),
            func_ptr,
            args,
            arg_types,
            Type::Int,
        )
    }

    pub fn call_release_gil_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_release_gil_for_type(Type::Ref),
            func_ptr,
            args,
            arg_types,
            Type::Ref,
        )
    }

    pub fn call_release_gil_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_family_typed(
            OpCode::call_release_gil_for_type(Type::Float),
            func_ptr,
            args,
            arg_types,
            Type::Float,
        )
    }

    pub fn call_loopinvariant_void_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Void);
    }

    pub fn call_loopinvariant_int_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Int)
    }

    pub fn call_loopinvariant_ref_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Ref)
    }

    pub fn call_loopinvariant_float_typed(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_loopinvariant_impl(func_ptr, args, arg_types, Type::Float)
    }

    /// pyjitpl.py:2081-2104: loop-invariant call with heapcache caching.
    /// Pure-like: does NOT invalidate caches. Result cached by (descr, arg0).
    fn call_loopinvariant_impl(
        &mut self,
        func_ptr: *const (),
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let func_ref = self.constants.get_or_insert(func_ptr as usize as i64);
        let descr_index = func_ref.0;
        let arg0_int = args.first().map(|a| a.0 as i64).unwrap_or(0);
        // heapcache: check loop-invariant cache
        if let Some((cached, _resvalue)) = self
            .heap_cache
            .call_loopinvariant_lookup(descr_index, arg0_int)
        {
            // Legacy trace_ctx helper does not yet thread the concrete
            // resvalue from this call site; the cached symbolic OpRef
            // is enough for the consumers of this method.
            return cached;
        }
        let opcode = OpCode::call_loopinvariant_for_type(ret_type);
        let descr = make_call_descr(arg_types, ret_type);
        let mut call_args = vec![func_ref];
        call_args.extend_from_slice(args);
        let result = self
            .recorder
            .record_op_with_descr(opcode, &call_args, descr);
        // Loop-invariant calls don't invalidate caches (like pure calls).
        // Concrete resvalue is unknown to this legacy helper; pass 0.
        self.heap_cache
            .call_loopinvariant_cache(descr_index, arg0_int, result, 0);
        result
    }

    // ── CALL_ASSEMBLER ────────────────────────────────────────────

    fn call_assembler_typed(
        &mut self,
        opcode: OpCode,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
        ret_type: Type,
    ) -> OpRef {
        let descr = make_call_assembler_descr(
            target.number,
            arg_types,
            ret_type,
            target.virtualizable_arg_index,
        );
        self.record_op_with_descr(opcode, args, descr)
    }

    /// Emit CALL_ASSEMBLER_I by token number, without needing a `&JitCellToken`.
    ///
    /// Assumes all args are `Type::Int`. For mixed-type args, use
    /// `call_assembler_int_by_number_typed` instead.
    pub fn call_assembler_int_by_number(&mut self, target_number: u64, args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_assembler_descr(
            target_number,
            &arg_types,
            Type::Int,
            self.driver_descriptor
                .as_ref()
                .and_then(JitDriverStaticData::virtualizable_arg_index),
        );
        self.record_op_with_descr(OpCode::CallAssemblerI, args, descr)
    }

    /// Emit CALL_ASSEMBLER_I by token number with explicit arg types.
    pub fn call_assembler_int_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        let descr = make_call_assembler_descr(
            target_number,
            arg_types,
            Type::Int,
            self.driver_descriptor
                .as_ref()
                .and_then(JitDriverStaticData::virtualizable_arg_index),
        );
        self.record_op_with_descr(OpCode::CallAssemblerI, args, descr)
    }

    /// Emit CALL_ASSEMBLER_R by token number with explicit arg types.
    /// resoperation.py:1251 call_assembler_for_descr: result_type=Ref → CALL_ASSEMBLER_R.
    pub fn call_assembler_ref_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        let descr = make_call_assembler_descr(
            target_number,
            arg_types,
            Type::Ref,
            self.driver_descriptor
                .as_ref()
                .and_then(JitDriverStaticData::virtualizable_arg_index),
        );
        self.record_op_with_descr(OpCode::CallAssemblerR, args, descr)
    }

    /// rewrite.py:665-695 handle_call_assembler parity.
    /// Emit CALL_ASSEMBLER with only the frame reference as arg; the backend
    /// expands to the full callee inputarg layout via `VableExpansion`.
    pub fn call_assembler_with_vable_expansion(
        &mut self,
        target_number: u64,
        frame_arg: OpRef,
        result_type: Type,
        expansion: majit_ir::VableExpansion,
    ) -> OpRef {
        self.call_assembler_with_vable_expansion_args(
            target_number,
            &[frame_arg],
            &[Type::Ref],
            result_type,
            expansion,
        )
    }

    /// pyjitpl.py:3589-3609 direct_assembler_call parity.
    /// Emit CALL_ASSEMBLER with multiple red args + VableExpansion.
    /// The backend reads some fields from args[0] (frame) and uses
    /// arg_overrides/const_overrides for callee-specific values.
    pub fn call_assembler_with_vable_expansion_args(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
        result_type: Type,
        expansion: majit_ir::VableExpansion,
    ) -> OpRef {
        let opcode = match result_type {
            Type::Int => OpCode::CallAssemblerI,
            Type::Ref => OpCode::CallAssemblerR,
            Type::Float => OpCode::CallAssemblerF,
            Type::Void => OpCode::CallAssemblerN,
        };
        let descr =
            make_call_assembler_descr_with_vable(target_number, arg_types, result_type, expansion);
        self.record_op_with_descr(opcode, args, descr)
    }

    /// Emit CALL_ASSEMBLER_N (void). Assumes all args are `Type::Int`.
    /// For mixed-type args, use `call_assembler_void_typed`.
    pub fn call_assembler_void(&mut self, target: &JitCellToken, args: &[OpRef]) {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_assembler_void_typed(target, args, &arg_types);
    }

    /// Emit CALL_ASSEMBLER_I. Assumes all args are `Type::Int`.
    /// For mixed-type args, use `call_assembler_int_typed`.
    pub fn call_assembler_int(&mut self, target: &JitCellToken, args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_assembler_int_typed(target, args, &arg_types)
    }

    /// Emit CALL_ASSEMBLER_R. Assumes all args are `Type::Int`.
    /// For mixed-type args, use `call_assembler_ref_typed`.
    pub fn call_assembler_ref(&mut self, target: &JitCellToken, args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_assembler_ref_typed(target, args, &arg_types)
    }

    /// Emit CALL_ASSEMBLER_F. Assumes all args are `Type::Int`.
    /// For mixed-type args, use `call_assembler_float_typed`.
    pub fn call_assembler_float(&mut self, target: &JitCellToken, args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        self.call_assembler_float_typed(target, args, &arg_types)
    }

    pub fn call_assembler_void_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) {
        let _ = self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Void),
            target,
            args,
            arg_types,
            Type::Void,
        );
    }

    pub fn call_assembler_int_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Int),
            target,
            args,
            arg_types,
            Type::Int,
        )
    }

    pub fn call_assembler_ref_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Ref),
            target,
            args,
            arg_types,
            Type::Ref,
        )
    }

    pub fn call_assembler_float_typed(
        &mut self,
        target: &JitCellToken,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        self.call_assembler_typed(
            OpCode::call_assembler_for_type(Type::Float),
            target,
            args,
            arg_types,
            Type::Float,
        )
    }

    // ── Exception handling ──────────────────────────────────────────

    /// Record GUARD_EXCEPTION: assert that the pending exception matches
    /// the given class, and produce a ref to the exception value.
    pub fn guard_exception(&mut self, exc_class: OpRef, num_live: usize) -> OpRef {
        self.record_guard(OpCode::GuardException, &[exc_class], num_live)
    }

    /// Record SAVE_EXCEPTION: capture the pending exception value as a ref.
    pub fn save_exception(&mut self) -> OpRef {
        self.record_op(OpCode::SaveException, &[])
    }

    /// Record SAVE_EXC_CLASS: capture the pending exception's class as an int.
    pub fn save_exc_class(&mut self) -> OpRef {
        self.record_op(OpCode::SaveExcClass, &[])
    }

    /// Record RESTORE_EXCEPTION: restore exception state from saved
    /// class and value refs.
    pub fn restore_exception(&mut self, exc_class: OpRef, exc_value: OpRef) {
        self.record_op(OpCode::RestoreException, &[exc_class, exc_value]);
    }

    // ── Object allocation ───────────────────────────────────────────

    /// Record NEW: allocate a new object described by `descr`.
    pub fn record_new(&mut self, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::New, &[], descr)
    }

    /// Record NEW_WITH_VTABLE: allocate a new object with an explicit vtable pointer.
    pub fn record_new_with_vtable(&mut self, vtable: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewWithVtable, &[vtable], descr)
    }

    /// Record NEW_ARRAY: allocate a new array with the given length.
    pub fn record_new_array(&mut self, length: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewArray, &[length], descr)
    }

    /// Record NEW_ARRAY_CLEAR: allocate a zero-initialized array.
    pub fn record_new_array_clear(&mut self, length: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::NewArrayClear, &[length], descr)
    }

    // ── Virtual references ────────────────────────────────────────

    /// Record VIRTUAL_REF_R: create a virtual reference (ref-typed result).
    ///
    /// `virtual_obj` is the real object being wrapped.
    /// `cindex` = ConstInt(len(virtualref_boxes) // 2) — pair index
    /// (pyjitpl.py:1805-1806 parity).
    ///
    /// The optimizer replaces this with a virtual struct, so if the vref
    /// never escapes, no allocation happens.
    pub fn virtual_ref_r(&mut self, virtual_obj: OpRef, cindex: OpRef) -> OpRef {
        self.record_op(OpCode::VirtualRefR, &[virtual_obj, cindex])
    }

    /// Record VIRTUAL_REF_I: create a virtual reference (int-typed result).
    /// `cindex` = ConstInt(len(virtualref_boxes) // 2) — pair index.
    pub fn virtual_ref_i(&mut self, virtual_obj: OpRef, cindex: OpRef) -> OpRef {
        self.record_op(OpCode::VirtualRefI, &[virtual_obj, cindex])
    }

    /// Record VIRTUAL_REF_FINISH: finalize a virtual reference.
    ///
    /// `vref` is the virtual reference to finalize.
    /// `virtual_obj` is the real object (or NULL/0 if the frame is being left normally).
    pub fn virtual_ref_finish(&mut self, vref: OpRef, virtual_obj: OpRef) {
        self.record_op(OpCode::VirtualRefFinish, &[vref, virtual_obj]);
    }

    /// Record FORCE_TOKEN: capture the current JIT frame address.
    pub fn force_token(&mut self) -> OpRef {
        self.record_op(OpCode::ForceToken, &[])
    }

    // ── Overflow-checked arithmetic ────────────────────────────────

    /// Record overflow-checked integer add + GuardNoOverflow.
    ///
    /// Returns the result OpRef. On overflow at trace time, the caller
    /// should abort tracing.
    pub fn int_add_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntAddOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    /// Record overflow-checked integer sub + GuardNoOverflow.
    pub fn int_sub_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntSubOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    /// Record overflow-checked integer mul + GuardNoOverflow.
    pub fn int_mul_ovf(&mut self, lhs: OpRef, rhs: OpRef, num_live: usize) -> OpRef {
        let result = self.record_op(OpCode::IntMulOvf, &[lhs, rhs]);
        self.record_guard(OpCode::GuardNoOverflow, &[], num_live);
        result
    }

    // ── String operations ───────────────────────────────────────────

    /// Record NEWSTR: allocate a new string with given length.
    pub fn newstr(&mut self, length: OpRef) -> OpRef {
        self.record_op(OpCode::Newstr, &[length])
    }

    /// Record STRLEN: get string length.
    pub fn strlen(&mut self, string: OpRef) -> OpRef {
        self.record_op(OpCode::Strlen, &[string])
    }

    /// Record STRGETITEM: read character at index.
    pub fn strgetitem(&mut self, string: OpRef, index: OpRef) -> OpRef {
        self.record_op(OpCode::Strgetitem, &[string, index])
    }

    /// Record STRSETITEM: write character at index.
    pub fn strsetitem(&mut self, string: OpRef, index: OpRef, value: OpRef) {
        self.record_op(OpCode::Strsetitem, &[string, index, value]);
    }

    /// Record COPYSTRCONTENT: copy characters between strings.
    pub fn copystrcontent(
        &mut self,
        src: OpRef,
        dst: OpRef,
        src_start: OpRef,
        dst_start: OpRef,
        length: OpRef,
    ) {
        self.record_op(
            OpCode::Copystrcontent,
            &[src, dst, src_start, dst_start, length],
        );
    }

    /// Record STRHASH: compute string hash.
    pub fn strhash(&mut self, string: OpRef) -> OpRef {
        self.record_op(OpCode::Strhash, &[string])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit_state::JitState;
    use majit_backend::JitCellToken;
    use majit_ir::Type;

    extern "C" fn dummy_call_target() {}

    #[test]
    fn const_type_honors_resume_data_override() {
        let mut ctx = TraceCtx::for_test(0);
        let c = ctx.const_int(0);
        // Box.type immutability: a ConstInt's intrinsic type is always Int,
        // recorded in the constant pool at allocation time.
        assert_eq!(ctx.const_type(c), Some(Type::Int));
        ctx.mark_const_type(c, Type::Ref);
        // mark_type → numbering_type_overrides takes priority over the
        // intrinsic Int; this is the raw-pointer-ConstInt retag path.
        assert_eq!(ctx.const_type(c), Some(Type::Ref));
    }

    #[derive(Clone, Copy)]
    struct TestSyncField {
        field_descr_idx: u32,
        value: OpRef,
        field_type: Type,
    }

    struct TestSyncState {
        vable_ref: OpRef,
        fields: Vec<TestSyncField>,
        forced: bool,
    }

    impl JitState for TestSyncState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _: usize, _: &()) -> () {}
        fn extract_live(&self, _: &()) -> Vec<i64> {
            Vec::new()
        }
        fn create_sym(_: &(), _: usize) -> () {}
        fn is_compatible(&self, _: &()) -> bool {
            true
        }
        fn restore(&mut self, _: &(), _: &[i64]) {}
        fn collect_jump_args(_: &()) -> Vec<OpRef> {
            Vec::new()
        }
        fn validate_close(_: &(), _: &()) -> bool {
            true
        }

        fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
            for field in &self.fields {
                let descr = crate::fail_descr::make_fail_descr(field.field_descr_idx as usize);
                ctx.record_op_with_descr(OpCode::SetfieldGc, &[self.vable_ref, field.value], descr);
            }
        }

        fn sync_virtualizable_after_residual_call(
            &self,
            ctx: &mut TraceCtx,
        ) -> crate::jit_state::ResidualVirtualizableSync {
            if self.forced {
                return crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: Vec::new(),
                    forced: true,
                };
            }
            let updated_fields = self
                .fields
                .iter()
                .map(|field| {
                    let opcode = OpCode::getfield_for_type(field.field_type);
                    let descr = crate::fail_descr::make_fail_descr(field.field_descr_idx as usize);
                    let new_ref = ctx.record_op_with_descr(opcode, &[self.vable_ref], descr);
                    (field.field_descr_idx, new_ref)
                })
                .collect();
            crate::jit_state::ResidualVirtualizableSync {
                updated_fields,
                forced: false,
            }
        }
    }

    fn make_ctx_with_mixed_inputs() -> (TraceCtx, [OpRef; 3]) {
        let mut recorder = Trace::new();
        let r = recorder.record_input_arg(Type::Ref);
        let f = recorder.record_input_arg(Type::Float);
        let i = recorder.record_input_arg(Type::Int);
        (TraceCtx::new(recorder, 0), [r, f, i])
    }

    fn take_single_call_descr(ctx: TraceCtx, jump_args: &[OpRef]) -> (Vec<Type>, OpCode) {
        let mut recorder = ctx.recorder;
        recorder.close_loop(jump_args);
        let trace = recorder.get_trace();
        let call_op = &trace.ops[0];
        let arg_types = call_op
            .descr
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .expect("call op should carry CallDescr")
            .arg_types()
            .to_vec();
        (arg_types, call_op.opcode)
    }

    fn take_single_call_op(ctx: TraceCtx, jump_args: &[OpRef]) -> majit_ir::Op {
        let mut recorder = ctx.recorder;
        recorder.close_loop(jump_args);
        let mut trace = recorder.get_trace();
        trace.ops.remove(0)
    }

    #[test]
    fn call_may_force_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_may_force_ref_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallMayForceR);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_release_gil_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_release_gil_float_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallReleaseGilF);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_loopinvariant_typed_preserves_mixed_arg_types() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let _ = ctx.call_loopinvariant_int_typed(
            dummy_call_target as *const (),
            &args,
            &[Type::Ref, Type::Float, Type::Int],
        );
        let (arg_types, opcode) = take_single_call_descr(ctx, &args);
        assert_eq!(opcode, OpCode::CallLoopinvariantI);
        assert_eq!(arg_types, &[Type::Ref, Type::Float, Type::Int]);
    }

    #[test]
    fn call_assembler_typed_preserves_mixed_arg_types_and_target_token() {
        let (mut ctx, args) = make_ctx_with_mixed_inputs();
        let mut token = JitCellToken::new(777);
        token.virtualizable_arg_index = Some(1);
        let _ = ctx.call_assembler_ref_typed(&token, &args, &[Type::Ref, Type::Float, Type::Int]);
        let op = take_single_call_op(ctx, &args);
        assert_eq!(op.opcode, OpCode::CallAssemblerR);
        assert_eq!(op.args.as_slice(), &args);
        let call_descr = op
            .descr
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .expect("call op should carry CallDescr");
        assert_eq!(call_descr.arg_types(), &[Type::Ref, Type::Float, Type::Int]);
        assert_eq!(call_descr.call_target_token(), Some(777));
        assert_eq!(call_descr.call_virtualizable_index(), Some(1));
    }

    fn take_all_ops(ctx: TraceCtx) -> Vec<majit_ir::Op> {
        let mut recorder = ctx.recorder;
        let num_inputs = recorder.num_inputargs();
        let jump_args: Vec<OpRef> = (0..num_inputs).map(|i| OpRef(i as u32)).collect();
        recorder.close_loop(&jump_args);
        let trace = recorder.get_trace();
        // Return only non-JUMP ops
        trace
            .ops
            .iter()
            .filter(|op| op.opcode != OpCode::Jump)
            .cloned()
            .collect()
    }

    #[test]
    fn call_may_force_with_jitstate_sync_emits_setfield_before_and_getfield_after() {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        let state = TestSyncState {
            vable_ref: vable,
            fields: vec![TestSyncField {
                field_descr_idx: 42,
                value: field_val,
                field_type: Type::Int,
            }],
            forced: false,
        };

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[field_val],
            &[Type::Int],
            &state,
            2,
        );

        assert!(result.0 > 0);
        assert_eq!(sync.updated_fields.len(), 1);
        assert_eq!(sync.updated_fields[0].0, 42);
        assert_ne!(sync.updated_fields[0].1, field_val);

        let ops = take_all_ops(ctx);
        assert!(ops.len() >= 4);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_void_emits_correct_sequence() {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        let state = TestSyncState {
            vable_ref: vable,
            fields: vec![TestSyncField {
                field_descr_idx: 10,
                value: field_val,
                field_type: Type::Int,
            }],
            forced: false,
        };

        let sync = ctx.call_may_force_with_jitstate_sync_void(
            dummy_call_target as *const (),
            &[field_val],
            &[Type::Int],
            &state,
            2,
        );

        assert_eq!(sync.updated_fields.len(), 1);
        assert_eq!(sync.updated_fields[0].0, 10);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceN);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_multiple_fields() {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let int_val = recorder.record_input_arg(Type::Int);
        let ref_val = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(recorder, 0);

        let state = TestSyncState {
            vable_ref: vable,
            fields: vec![
                TestSyncField {
                    field_descr_idx: 0,
                    value: int_val,
                    field_type: Type::Int,
                },
                TestSyncField {
                    field_descr_idx: 1,
                    value: ref_val,
                    field_type: Type::Ref,
                },
            ],
            forced: false,
        };

        let (_, sync) = ctx.call_may_force_with_jitstate_sync_ref(
            dummy_call_target as *const (),
            &[int_val],
            &[Type::Int],
            &state,
            3,
        );

        assert_eq!(sync.updated_fields.len(), 2);
        assert_eq!(sync.updated_fields[0].0, 0);
        assert_eq!(sync.updated_fields[1].0, 1);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[2].opcode, OpCode::CallMayForceR);
        assert_eq!(ops[3].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[4].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[5].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_empty_jitstate_sync_behaves_like_plain_call() {
        let mut recorder = Trace::new();
        let val = recorder.record_input_arg(Type::Int);
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(recorder, 0);

        let state = TestSyncState {
            vable_ref: vable,
            fields: Vec::new(),
            forced: false,
        };

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[val],
            &[Type::Int],
            &state,
            1,
        );

        assert!(result.0 > 0);
        assert!(sync.updated_fields.is_empty());

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[1].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_float_field() {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let float_val = recorder.record_input_arg(Type::Float);
        let mut ctx = TraceCtx::new(recorder, 0);

        let state = TestSyncState {
            vable_ref: vable,
            fields: vec![TestSyncField {
                field_descr_idx: 5,
                value: float_val,
                field_type: Type::Float,
            }],
            forced: false,
        };

        let (_, sync) = ctx.call_may_force_with_jitstate_sync_float(
            dummy_call_target as *const (),
            &[float_val],
            &[Type::Float],
            &state,
            2,
        );

        assert_eq!(sync.updated_fields.len(), 1);
        assert_eq!(sync.updated_fields[0].0, 5);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceF);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcF);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_default_noop() {
        use crate::jit_state::JitState;

        #[derive(Default)]
        struct NoVableState;

        impl JitState for NoVableState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) -> () {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) -> () {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
        }

        let mut recorder = Trace::new();
        let val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);
        let state = NoVableState;

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[val],
            &[Type::Int],
            &state,
            1,
        );

        // Default JitState does no sync => no extra ops
        assert!(result.0 > 0);
        assert!(!sync.forced);
        assert!(sync.updated_fields.is_empty());

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[1].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_custom_impl() {
        use crate::jit_state::JitState;

        struct VableState {
            vable_ref: OpRef,
            field_val: OpRef,
        }

        impl JitState for VableState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) -> () {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) -> () {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }

            fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
                // Write field 0 to heap
                let fd = majit_ir::make_field_descr(0, 8, Type::Int, majit_ir::ArrayFlag::Signed);
                ctx.vable_setfield(self.vable_ref, fd, self.field_val, Value::Int(0));
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                // Re-read field 0 from heap
                let fd = majit_ir::make_field_descr(0, 8, Type::Int, majit_ir::ArrayFlag::Signed);
                let (new_ref, _) = ctx.vable_getfield_int(self.vable_ref, fd);
                crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: vec![(0, new_ref)],
                    forced: false,
                }
            }
        }

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);
        let state = VableState {
            vable_ref: vable,
            field_val,
        };

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[field_val],
            &[Type::Int],
            &state,
            2,
        );

        assert!(result.0 > 0);
        assert!(!sync.forced);
        assert_eq!(sync.updated_fields.len(), 1);
        assert_eq!(sync.updated_fields[0].0, 0);

        let ops = take_all_ops(ctx);
        // SetfieldGc(before) + CallMayForceI + GetfieldGcI(after) + GuardNotForced
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_skips_guard_when_forced() {
        use crate::jit_state::JitState;

        struct ForcedState;

        impl JitState for ForcedState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) -> () {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) -> () {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                _ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: Vec::new(),
                    forced: true,
                }
            }
        }

        let mut recorder = Trace::new();
        let val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);
        let state = ForcedState;

        let (_result, sync) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[val],
            &[Type::Int],
            &state,
            1,
        );

        assert!(sync.forced);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::CallMayForceI);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_ref_custom_impl() {
        use crate::jit_state::JitState;

        struct RefState {
            vable_ref: OpRef,
            field_val: OpRef,
        }

        impl JitState for RefState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) -> () {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) -> () {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }

            fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
                let fd = majit_ir::make_field_descr(0, 8, Type::Ref, majit_ir::ArrayFlag::Pointer);
                ctx.vable_setfield(
                    self.vable_ref,
                    fd,
                    self.field_val,
                    Value::Ref(majit_ir::GcRef::NULL),
                );
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                let fd = majit_ir::make_field_descr(0, 8, Type::Ref, majit_ir::ArrayFlag::Pointer);
                let (new_ref, _) = ctx.vable_getfield_ref(self.vable_ref, fd);
                crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: vec![(0, new_ref)],
                    forced: false,
                }
            }
        }

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(recorder, 0);
        let state = RefState {
            vable_ref: vable,
            field_val,
        };

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_ref(
            dummy_call_target as *const (),
            &[field_val],
            &[Type::Ref],
            &state,
            2,
        );

        assert!(result.0 > 0);
        assert!(!sync.forced);
        assert_eq!(sync.updated_fields.len(), 1);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceR);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_float_custom_impl() {
        use crate::jit_state::JitState;

        struct FloatState {
            vable_ref: OpRef,
            field_val: OpRef,
        }

        impl JitState for FloatState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) -> () {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) -> () {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }

            fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
                let fd = majit_ir::make_field_descr(0, 8, Type::Float, majit_ir::ArrayFlag::Float);
                ctx.vable_setfield(self.vable_ref, fd, self.field_val, Value::Float(0.0));
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                let fd = majit_ir::make_field_descr(0, 8, Type::Float, majit_ir::ArrayFlag::Float);
                let (new_ref, _) = ctx.vable_getfield_float(self.vable_ref, fd);
                crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: vec![(0, new_ref)],
                    forced: false,
                }
            }
        }

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Float);
        let mut ctx = TraceCtx::new(recorder, 0);
        let state = FloatState {
            vable_ref: vable,
            field_val,
        };

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_float(
            dummy_call_target as *const (),
            &[field_val],
            &[Type::Float],
            &state,
            2,
        );

        assert!(result.0 > 0);
        assert!(!sync.forced);
        assert_eq!(sync.updated_fields.len(), 1);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceF);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcF);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_void_skips_guard_when_forced() {
        use crate::jit_state::JitState;

        struct ForcedVoidState;

        impl JitState for ForcedVoidState {
            type Meta = ();
            type Sym = ();
            type Env = ();

            fn build_meta(&self, _: usize, _: &()) -> () {}
            fn extract_live(&self, _: &()) -> Vec<i64> {
                Vec::new()
            }
            fn create_sym(_: &(), _: usize) -> () {}
            fn is_compatible(&self, _: &()) -> bool {
                true
            }
            fn restore(&mut self, _: &(), _: &[i64]) {}
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                _ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: Vec::new(),
                    forced: true,
                }
            }
        }

        let mut recorder = Trace::new();
        let val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);
        let state = ForcedVoidState;

        let sync = ctx.call_may_force_with_jitstate_sync_void(
            dummy_call_target as *const (),
            &[val],
            &[Type::Int],
            &state,
            1,
        );

        assert!(sync.forced);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::CallMayForceN);
    }

    // --- virtualizable_boxes tests ---

    fn make_test_vable_info() -> crate::virtualizable::VirtualizableInfo {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_field("sp", Type::Int, 16);
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        info
    }

    fn make_test_vable_info_with_array() -> crate::virtualizable::VirtualizableInfo {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Int,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        info
    }

    // Test helper: typed placeholder matching each slot's declared type so
    // the Box's (OpRef, concrete) pair stays internally consistent — the
    // RPython `virtualizable_boxes[index] = valuebox` invariant.  Tests
    // only inspect OpRef plumbing; the concrete half is never read.
    fn ph(ty: Type) -> Value {
        match ty {
            Type::Int => Value::Int(0),
            Type::Float => Value::Float(0.0),
            Type::Ref => Value::Ref(majit_ir::GcRef::NULL),
            Type::Void => Value::Void,
        }
    }

    #[test]
    fn standard_vable_getfield_reads_from_boxes() {
        let info = make_test_vable_info();
        let fd8 = info.static_field_descr(0);
        let fd16 = info.static_field_descr(1);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int); // pc
        let box1 = recorder.record_input_arg(Type::Int); // sp
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // getfield with offset=8 → static field 0 → box0
        let (result, _) = ctx.vable_getfield_int(vable, fd8);
        assert_eq!(result, box0);
        // getfield with offset=16 → static field 1 → box1
        let (result, _) = ctx.vable_getfield_int(vable, fd16);
        assert_eq!(result, box1);

        // No heap ops should have been emitted
        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "standard vable getfield should not emit ops"
        );
    }

    #[test]
    fn standard_vable_setfield_writes_to_boxes() {
        let info = make_test_vable_info();
        let fd8 = info.static_field_descr(0);
        let fd16 = info.static_field_descr(1);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let box1 = recorder.record_input_arg(Type::Int);
        let new_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // setfield offset=8 → updates box0
        ctx.vable_setfield(vable, fd8.clone(), new_val, ph(Type::Int));

        // Box 0 should now be new_val
        let (result, _) = ctx.vable_getfield_int(vable, fd8);
        assert_eq!(result, new_val);
        // Box 1 unchanged
        let (result, _) = ctx.vable_getfield_int(vable, fd16);
        assert_eq!(result, box1);

        // No heap ops should have been emitted
        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "standard vable setfield should not emit ops"
        );
    }

    #[test]
    fn nonstandard_vable_getfield_emits_heap_op() {
        // Without init_virtualizable_boxes, falls back to GETFIELD_GC_I
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(recorder, 0);

        let fd8 = majit_ir::make_field_descr(8, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        let _result = ctx.vable_getfield_int(vable, fd8);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcI);
    }

    #[test]
    fn nonstandard_vable_setfield_emits_heap_op() {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        let fd8 = majit_ir::make_field_descr(8, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        ctx.vable_setfield(vable, fd8, val, ph(Type::Int));

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn standard_vable_getfield_unknown_offset_emits_heap_op() {
        let info = make_test_vable_info();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let box1 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // Unknown offset (999) → fallback to heap op
        let fd999 = majit_ir::make_field_descr(999, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        let _result = ctx.vable_getfield_int(vable, fd999);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcI);
    }

    #[test]
    fn standard_vable_getfield_ref_reads_from_boxes() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("obj", Type::Ref, 8);
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        let fd8 = info.static_field_descr(0);

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(&info, vable, ph(Type::Ref), &[box0], &[ph(Type::Ref)], &[]);

        let (result, _) = ctx.vable_getfield_ref(vable, fd8);
        assert_eq!(result, box0);

        let ops = take_all_ops(ctx);
        assert!(ops.is_empty());
    }

    #[test]
    fn standard_vable_getfield_float_reads_from_boxes() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("val", Type::Float, 8);
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        let fd8 = info.static_field_descr(0);

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Float);
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0],
            &[ph(Type::Float)],
            &[],
        );

        let (result, _) = ctx.vable_getfield_float(vable, fd8);
        assert_eq!(result, box0);

        let ops = take_all_ops(ctx);
        assert!(ops.is_empty());
    }

    #[test]
    fn vable_getarrayitem_reads_from_boxes() {
        let info = make_test_vable_info_with_array();
        let fd24 = info.array_pointer_field_descr(0);
        // 1 static field (pc) + 3 array elements
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let box_arr1 = recorder.record_input_arg(Type::Int);
        let box_arr2 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0, box_arr1, box_arr2],
            &[ph(Type::Int), ph(Type::Int), ph(Type::Int), ph(Type::Int)],
            &[3], // array has 3 elements
        );

        // Array field offset=24, item_index=0 → box_arr0
        let (r0, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 0);
        assert_eq!(r0, box_arr0);
        // item_index=1 → box_arr1
        let (r1, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 1);
        assert_eq!(r1, box_arr1);
        // item_index=2 → box_arr2
        let (r2, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 2);
        assert_eq!(r2, box_arr2);

        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "standard vable getarrayitem should not emit ops"
        );
    }

    #[test]
    fn vable_setarrayitem_writes_to_boxes() {
        let info = make_test_vable_info_with_array();
        let fd24 = info.array_pointer_field_descr(0);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let box_arr1 = recorder.record_input_arg(Type::Int);
        let new_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0, box_arr1],
            &[ph(Type::Int), ph(Type::Int), ph(Type::Int)],
            &[2], // array has 2 elements
        );

        // Write to array[1]
        ctx.vable_setarrayitem_vable(vable, &fd24, 1, new_val, ph(Type::Int));

        // Read back: array[0] unchanged, array[1] updated
        let (r0, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 0);
        assert_eq!(r0, box_arr0);
        let (r1, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 1);
        assert_eq!(r1, new_val);

        let ops = take_all_ops(ctx);
        assert!(ops.is_empty());
    }

    #[test]
    fn vable_getarrayitem_unknown_array_emits_heap_op() {
        let info = make_test_vable_info_with_array();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );

        // Unknown array field offset → fallback
        let fd999 = majit_ir::make_field_descr(999, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        let _r = ctx.vable_getarrayitem_int_vable(vable, &fd999, 0);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetarrayitemGcI);
    }

    #[test]
    fn collect_virtualizable_boxes_returns_current_state() {
        let info = make_test_vable_info();
        let fd8 = info.static_field_descr(0);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let box1 = recorder.record_input_arg(Type::Int);
        let new_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        // Before init: None
        assert!(ctx.collect_virtualizable_boxes().is_none());

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // After init: has boxes (field0, field1, vable_ref sentinel)
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes, vec![box0, box1, vable]);

        // After mutation
        ctx.vable_setfield(vable, fd8, new_val, ph(Type::Int));
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes, vec![new_val, box1, vable]);
    }

    #[test]
    fn gen_store_back_in_vable_uses_field_and_array_descrs() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Ref,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Ref),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Ref);
        let box_arr1 = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(recorder, 0);
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0, box_arr1],
            &[ph(Type::Int), ph(Type::Ref), ph(Type::Ref)],
            &[2],
        );

        ctx.gen_store_back_in_vable(vable);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(
            ops[0].descr.as_ref().map(|d| d.index()),
            Some(info.static_field_descr(0).index())
        );
        assert_eq!(ops[1].opcode, OpCode::GetfieldGcR);
        assert_eq!(
            ops[1].descr.as_ref().map(|d| d.index()),
            Some(info.array_pointer_field_descr(0).index())
        );
        assert_eq!(ops[2].opcode, OpCode::SetarrayitemGc);
        assert_eq!(
            ops[2].descr.as_ref().map(|d| d.index()),
            Some(info.array_item_descr(0).index())
        );
        assert_eq!(ops[3].opcode, OpCode::SetarrayitemGc);
        assert_eq!(
            ops[3].descr.as_ref().map(|d| d.index()),
            Some(info.array_item_descr(0).index())
        );
        assert_eq!(ops[4].opcode, OpCode::SetfieldGc);
        assert_eq!(
            ops[4].descr.as_ref().map(|d| d.index()),
            Some(info.token_field_descr().index())
        );
    }

    #[test]
    fn gen_store_back_in_vable_ignores_nonstandard_virtualizable() {
        let info = make_test_vable_info_with_array();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let other_vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );

        ctx.gen_store_back_in_vable(other_vable);

        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "nonstandard virtualizable must not use standard store-back path"
        );
    }
}
