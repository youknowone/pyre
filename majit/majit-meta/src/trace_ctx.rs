//! No direct RPython equivalent — unified trace recording context
//! (RPython splits this across MetaInterp.history, compile.py, and Trace).

use majit_ir::{DescrRef, GreenKey, OpCode, OpRef, Type};
use majit_trace::heapcache::HeapCache;
use majit_trace::recorder::{Trace, TracePosition};

use majit_codegen::JitCellToken;

use crate::TraceAction;
use crate::call_descr::{make_call_assembler_descr, make_call_descr, make_call_may_force_descr};
use crate::constant_pool::ConstantPool;
use crate::fail_descr::{make_fail_descr, make_fail_descr_typed};
use crate::jitdriver::JitDriverStaticData;
use crate::symbolic_stack::SymbolicStack;
use crate::virtualizable::VirtualizableInfo;

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
    /// VirtualizableInfo for the standard virtualizable (if any).
    virtualizable_info: Option<VirtualizableInfo>,
    /// Lengths of each virtualizable array field, needed for flat index computation.
    virtualizable_array_lengths: Option<Vec<usize>>,
    /// Header PC at which this trace started (0 = function entry).
    pub header_pc: usize,
    /// Pending OpRef replacements from inline callee returns.
    /// Applied when the trace is finalized (close_loop/compile).
    replacements: Vec<(OpRef, OpRef)>,
    /// pyjitpl.py:3030 current_merge_points — loop headers visited during
    /// tracing with their trace positions. First visit records the key +
    /// position; second visit closes the loop.
    current_merge_points: Vec<MergePoint>,
    /// pyjitpl.py:2398: tracing-time heap cache.
    /// Tracks field/array values, allocations, escape status, and class/nullity
    /// knowledge during tracing to avoid recording redundant operations.
    heap_cache: HeapCache,
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
    pub fn add_merge_point(&mut self, key: u64, live_args: Vec<OpRef>, live_arg_types: Vec<Type>) {
        let position = self.recorder.get_position();
        self.current_merge_points.push(MergePoint {
            green_key: key,
            position,
            original_boxes: live_args,
            original_box_types: live_arg_types,
        });
    }

    /// pyjitpl.py:2908 — bridge traces start with empty merge points.
    pub fn clear_merge_points(&mut self) {
        self.current_merge_points.clear();
    }

    /// pyjitpl.py:2988: find merge point by key, searching in reverse
    /// order (most recent first, matching RPython's range(len-1, -1, -1)).
    pub fn get_merge_point(&self, key: u64) -> Option<&MergePoint> {
        self.current_merge_points
            .iter()
            .rev()
            .find(|mp| mp.green_key == key)
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

    /// pyjitpl.py:2951, 2418: reset heap cache at loop header / retrace.
    pub fn reset_heap_cache(&mut self) {
        self.heap_cache.reset();
    }

    /// Create a standalone TraceCtx for testing or external use.
    pub fn for_test(num_inputs: usize) -> Self {
        let mut recorder = Trace::new();
        for _ in 0..num_inputs {
            recorder.record_input_arg(majit_ir::Type::Int);
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
            virtualizable_info: None,
            virtualizable_array_lengths: None,
            header_pc: 0,
            replacements: Vec::new(),
            current_merge_points: vec![MergePoint {
                green_key,
                position: initial_position,
                original_box_types: initial_types,
                original_boxes: initial_boxes.clone(),
            }],
            heap_cache: HeapCache::new(),
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
            virtualizable_info: None,
            virtualizable_array_lengths: None,
            header_pc: 0,
            replacements: Vec::new(),
            current_merge_points: vec![MergePoint {
                green_key,
                position: initial_position,
                original_box_types: initial_boxes.iter().map(|_| Type::Ref).collect(),
                original_boxes: initial_boxes.clone(),
            }],
            heap_cache: HeapCache::new(),
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

    /// Apply all pending replacements to the trace ops.
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

    pub fn find_biggest_inline_function(&self) -> Option<u64> {
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
    /// RPython parity: Ref constants preserve their type so guard
    /// fail_args are correctly typed during guard failure recovery.
    pub fn const_ref(&mut self, value: i64) -> OpRef {
        self.constants
            .get_or_insert_typed(value, majit_ir::Type::Ref)
    }

    /// Return the type of a constant OpRef, if recorded.
    pub fn const_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        self.constants.constant_type(opref)
    }

    /// Return the concrete value for a constant OpRef, if it is a pooled constant.
    pub fn const_value(&self, opref: OpRef) -> Option<i64> {
        self.constants.as_ref().get(&opref.0).copied()
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

    /// Whether the trace has exceeded the maximum allowed length.
    pub fn is_too_long(&self) -> bool {
        self.recorder.is_too_long()
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
        let const_ref = self.const_int(runtime_value);
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
    /// `input_oprefs` contains one OpRef per static field + array element,
    /// in the same flat layout as `VirtualizableInfo::get_index_in_array`.
    /// `vable_ref` is the OpRef of the virtualizable object (frame pointer).
    /// Boxes layout: [field0, ..., fieldN, arr[0], ..., arr[M], vable_ref]
    /// where `boxes[-1]` is the standard virtualizable identity (RPython parity).
    pub fn init_virtualizable_boxes(
        &mut self,
        info: &VirtualizableInfo,
        vable_ref: OpRef,
        input_oprefs: &[OpRef],
        array_lengths: &[usize],
    ) {
        let mut boxes = input_oprefs.to_vec();
        boxes.push(vable_ref); // RPython: virtualizable_boxes[-1] = vable identity
        self.virtualizable_boxes = Some(boxes);
        self.virtualizable_info = Some(info.clone());
        self.virtualizable_array_lengths = Some(array_lengths.to_vec());
    }

    /// Collect the current virtualizable boxes (for close_loop / finish).
    /// Returns `None` if no standard virtualizable is active.
    pub fn collect_virtualizable_boxes(&self) -> Option<Vec<OpRef>> {
        self.virtualizable_boxes.clone()
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

    /// Update a standard virtualizable box by flat index.
    pub fn set_virtualizable_box_at(&mut self, index: usize, value: OpRef) -> bool {
        if let Some(boxes) = &mut self.virtualizable_boxes {
            if let Some(slot) = boxes.get_mut(index) {
                *slot = value;
                return true;
            }
        }
        false
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

    /// Canonical virtualizable metadata for the active standard virtualizable.
    pub fn virtualizable_info(&self) -> Option<&VirtualizableInfo> {
        self.virtualizable_info.as_ref()
    }

    /// Cached array lengths for the active standard virtualizable.
    pub fn virtualizable_array_lengths(&self) -> Option<&[usize]> {
        self.virtualizable_array_lengths.as_deref()
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

    /// Consume `hint(frame, force_virtualizable=True)` during tracing.
    ///
    /// RPython equivalent: `MetaInterp.gen_store_back_in_vable(box)`
    ///
    /// Emits SETFIELD_GC for each static field + SETARRAYITEM_GC for each
    /// array element to flush virtualizable boxes back to the heap.
    /// Finally emits SETFIELD_GC(vable_token, NULL) to clear the token.
    ///
    /// Only operates on the standard virtualizable (identified by
    /// `vable_opref` matching the frame reference in boxes). Nonstandard
    /// or virtual virtualizables are ignored (RPython parity).
    pub fn gen_store_back_in_vable(&mut self, vable_opref: OpRef) {
        let (info, boxes, lengths) = match (
            self.virtualizable_info.clone(),
            self.virtualizable_boxes.clone(),
            self.virtualizable_array_lengths.clone(),
        ) {
            (Some(info), Some(boxes), Some(lengths)) => (info, boxes, lengths),
            _ => return,
        };

        if boxes.last().copied() != Some(vable_opref) {
            return;
        }

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

    /// RPython pyjitpl.py:1114 `_nonstandard_virtualizable()`.
    ///
    /// Returns true if `vable_opref` is NOT the standard virtualizable
    /// (i.e., `vable_opref != virtualizable_boxes[-1]`). Nonstandard
    /// virtualizables must fall back to heap operations.
    fn is_nonstandard_virtualizable(&self, vable_opref: OpRef) -> bool {
        if let Some(ref boxes) = self.virtualizable_boxes {
            if let Some(&vbox) = boxes.last() {
                return vbox != vable_opref;
            }
        }
        true // No boxes → treat as nonstandard
    }

    /// Record a virtualizable field read (GETFIELD_GC_I/R/F).
    ///
    /// RPython pyjitpl.py:1161 `opimpl_getfield_vable_i`.
    ///
    /// Standard virtualizable: returns box value directly (no heap op).
    /// Nonstandard (escaped/virtual): falls back to GETFIELD_GC.
    pub fn vable_getfield_int(&mut self, vable_opref: OpRef, field_offset: usize) -> OpRef {
        if !self.is_nonstandard_virtualizable(vable_opref) {
            if let (Some(boxes), Some(info)) = (&self.virtualizable_boxes, &self.virtualizable_info)
            {
                if let Some(index) = info.static_field_index(field_offset) {
                    return boxes[index];
                }
            }
        }
        let offset_ref = self.const_int(field_offset as i64);
        self.record_op(OpCode::GetfieldGcI, &[vable_opref, offset_ref])
    }

    /// Record a virtualizable field read with an explicit field descriptor.
    pub fn vable_getfield_int_descr(&mut self, vable_opref: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::GetfieldGcI, &[vable_opref], descr)
    }

    /// Record a virtualizable field write (SETFIELD_GC).
    ///
    /// RPython pyjitpl.py:1183 `_opimpl_setfield_vable`.
    ///
    /// Standard: updates box directly. Nonstandard: falls back to SETFIELD_GC.
    pub fn vable_setfield(&mut self, vable_opref: OpRef, field_offset: usize, value: OpRef) {
        if !self.is_nonstandard_virtualizable(vable_opref) {
            let index = self
                .virtualizable_info
                .as_ref()
                .and_then(|info| info.static_field_index(field_offset));
            if let Some(idx) = index {
                if let Some(boxes) = &mut self.virtualizable_boxes {
                    boxes[idx] = value;
                    return;
                }
            }
        }
        let offset_ref = self.const_int(field_offset as i64);
        self.record_op(OpCode::SetfieldGc, &[vable_opref, offset_ref, value]);
    }

    /// Record a virtualizable field write with an explicit field descriptor.
    pub fn vable_setfield_descr(&mut self, vable_opref: OpRef, value: OpRef, descr: DescrRef) {
        self.record_op_with_descr(OpCode::SetfieldGc, &[vable_opref, value], descr);
    }

    /// Record a virtualizable ref field read (GETFIELD_GC_R).
    ///
    /// RPython pyjitpl.py:1168 `opimpl_getfield_vable_r`.
    pub fn vable_getfield_ref(&mut self, vable_opref: OpRef, field_offset: usize) -> OpRef {
        if !self.is_nonstandard_virtualizable(vable_opref) {
            if let (Some(boxes), Some(info)) = (&self.virtualizable_boxes, &self.virtualizable_info)
            {
                if let Some(index) = info.static_field_index(field_offset) {
                    return boxes[index];
                }
            }
        }
        let offset_ref = self.const_int(field_offset as i64);
        self.record_op(OpCode::GetfieldGcR, &[vable_opref, offset_ref])
    }

    /// Record a virtualizable ref field read with an explicit field descriptor.
    pub fn vable_getfield_ref_descr(&mut self, vable_opref: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], descr)
    }

    /// Record a virtualizable float field read (GETFIELD_GC_F).
    ///
    /// RPython pyjitpl.py:1175 `opimpl_getfield_vable_f`.
    pub fn vable_getfield_float(&mut self, vable_opref: OpRef, field_offset: usize) -> OpRef {
        if !self.is_nonstandard_virtualizable(vable_opref) {
            if let (Some(boxes), Some(info)) = (&self.virtualizable_boxes, &self.virtualizable_info)
            {
                if let Some(index) = info.static_field_index(field_offset) {
                    return boxes[index];
                }
            }
        }
        let offset_ref = self.const_int(field_offset as i64);
        self.record_op(OpCode::GetfieldGcF, &[vable_opref, offset_ref])
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
        array_field_offset: usize,
        item_index: usize,
    ) -> OpRef {
        if let Some(flat_idx) = self.vable_array_flat_index(array_field_offset, item_index) {
            if let Some(boxes) = &self.virtualizable_boxes {
                return boxes[flat_idx];
            }
        }
        let index = self.const_int(item_index as i64);
        let zero = self.const_int(0);
        self.record_op(OpCode::GetarrayitemGcI, &[array_opref, index, zero])
    }

    /// Virtualizable array item read with an index OpRef.
    ///
    /// If the index is a known constant and the vable is standard, reads
    /// directly from `virtualizable_boxes`. Otherwise falls back to heap ops.
    pub fn vable_getarrayitem_int_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        array_field_offset: usize,
    ) -> OpRef {
        if let Some(item_index) = self
            .const_value(index)
            .and_then(|value| usize::try_from(value).ok())
        {
            if !self.is_nonstandard_virtualizable(vable_opref) {
                if let Some(flat_idx) = self.vable_array_flat_index(array_field_offset, item_index)
                {
                    if let Some(boxes) = &self.virtualizable_boxes {
                        return boxes[flat_idx];
                    }
                }
            }
            let array_opref = self.vable_getfield_ref(vable_opref, array_field_offset);
            return self.vable_getarrayitem_int_vable(array_opref, array_field_offset, item_index);
        }
        let array_opref = self.vable_getfield_ref(vable_opref, array_field_offset);
        self.vable_getarrayitem_int(array_opref, index)
    }

    /// Standard virtualizable array item read (ref).
    pub fn vable_getarrayitem_ref_vable(
        &mut self,
        array_opref: OpRef,
        array_field_offset: usize,
        item_index: usize,
    ) -> OpRef {
        if let Some(flat_idx) = self.vable_array_flat_index(array_field_offset, item_index) {
            if let Some(boxes) = &self.virtualizable_boxes {
                return boxes[flat_idx];
            }
        }
        let index = self.const_int(item_index as i64);
        let zero = self.const_int(0);
        self.record_op(OpCode::GetarrayitemGcR, &[array_opref, index, zero])
    }

    /// Virtualizable array ref item read with an index OpRef.
    pub fn vable_getarrayitem_ref_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        array_field_offset: usize,
    ) -> OpRef {
        if let Some(item_index) = self
            .const_value(index)
            .and_then(|value| usize::try_from(value).ok())
        {
            if !self.is_nonstandard_virtualizable(vable_opref) {
                if let Some(flat_idx) = self.vable_array_flat_index(array_field_offset, item_index)
                {
                    if let Some(boxes) = &self.virtualizable_boxes {
                        return boxes[flat_idx];
                    }
                }
            }
            let array_opref = self.vable_getfield_ref(vable_opref, array_field_offset);
            return self.vable_getarrayitem_ref_vable(array_opref, array_field_offset, item_index);
        }
        let array_opref = self.vable_getfield_ref(vable_opref, array_field_offset);
        self.vable_getarrayitem_ref(array_opref, index)
    }

    /// Standard virtualizable array item read (float).
    pub fn vable_getarrayitem_float_vable(
        &mut self,
        array_opref: OpRef,
        array_field_offset: usize,
        item_index: usize,
    ) -> OpRef {
        if let Some(flat_idx) = self.vable_array_flat_index(array_field_offset, item_index) {
            if let Some(boxes) = &self.virtualizable_boxes {
                return boxes[flat_idx];
            }
        }
        let index = self.const_int(item_index as i64);
        let zero = self.const_int(0);
        self.record_op(OpCode::GetarrayitemGcF, &[array_opref, index, zero])
    }

    /// Virtualizable array float item read with an index OpRef.
    pub fn vable_getarrayitem_float_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        array_field_offset: usize,
    ) -> OpRef {
        if let Some(item_index) = self
            .const_value(index)
            .and_then(|value| usize::try_from(value).ok())
        {
            if !self.is_nonstandard_virtualizable(vable_opref) {
                if let Some(flat_idx) = self.vable_array_flat_index(array_field_offset, item_index)
                {
                    if let Some(boxes) = &self.virtualizable_boxes {
                        return boxes[flat_idx];
                    }
                }
            }
            let array_opref = self.vable_getfield_ref(vable_opref, array_field_offset);
            return self.vable_getarrayitem_float_vable(
                array_opref,
                array_field_offset,
                item_index,
            );
        }
        let array_opref = self.vable_getfield_ref(vable_opref, array_field_offset);
        self.vable_getarrayitem_float(array_opref, index)
    }

    /// Standard virtualizable array item write.
    /// `array_field_offset` identifies which array field, `item_index` is the element index.
    /// If standard boxes are active, writes to the flat box array directly.
    pub fn vable_setarrayitem_vable(
        &mut self,
        array_opref: OpRef,
        array_field_offset: usize,
        item_index: usize,
        value: OpRef,
    ) {
        let flat_idx = self.vable_array_flat_index(array_field_offset, item_index);
        if let Some(idx) = flat_idx {
            if let Some(boxes) = &mut self.virtualizable_boxes {
                boxes[idx] = value;
                return;
            }
        }
        let index = self.const_int(item_index as i64);
        let zero = self.const_int(0);
        self.record_op(OpCode::SetarrayitemGc, &[array_opref, index, value, zero]);
    }

    /// Virtualizable array item write with an index OpRef.
    pub fn vable_setarrayitem_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        array_field_offset: usize,
        value: OpRef,
    ) {
        if let Some(item_index) = self
            .const_value(index)
            .and_then(|raw| usize::try_from(raw).ok())
        {
            if !self.is_nonstandard_virtualizable(vable_opref) {
                if let Some(idx) = self.vable_array_flat_index(array_field_offset, item_index) {
                    if let Some(boxes) = &mut self.virtualizable_boxes {
                        boxes[idx] = value;
                        return;
                    }
                }
            }
            let array_opref = self.vable_getfield_ref(vable_opref, array_field_offset);
            self.vable_setarrayitem_vable(array_opref, array_field_offset, item_index, value);
            return;
        }
        let array_opref = self.vable_getfield_ref(vable_opref, array_field_offset);
        self.vable_setarrayitem(array_opref, index, value);
    }

    /// Virtualizable array length.
    ///
    /// Standard virtualizable reads the cached array length directly.
    /// Nonstandard virtualizable falls back to heap array length.
    pub fn vable_arraylen_vable(&mut self, vable_opref: OpRef, array_field_offset: usize) -> OpRef {
        if !self.is_nonstandard_virtualizable(vable_opref) {
            if let (Some(info), Some(lengths)) =
                (&self.virtualizable_info, &self.virtualizable_array_lengths)
            {
                if let Some(array_idx) = info.array_field_index(array_field_offset) {
                    if let Some(&length) = lengths.get(array_idx) {
                        return self.const_int(length as i64);
                    }
                }
            }
        }
        let array_opref = self.vable_getfield_ref(vable_opref, array_field_offset);
        self.record_op(OpCode::ArraylenGc, &[array_opref])
    }

    /// Compute the flat index into virtualizable_boxes for an array element.
    /// Returns `None` if standard virtualizable is not active or the array field is unknown.
    fn vable_array_flat_index(
        &self,
        array_field_offset: usize,
        item_index: usize,
    ) -> Option<usize> {
        let info = self.virtualizable_info.as_ref()?;
        let lengths = self.virtualizable_array_lengths.as_ref()?;
        let array_idx = info.array_field_index(array_field_offset)?;
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
        // heapcache.py: pure/elidable calls (CallPure*) don't invalidate.
        // Non-pure calls escape their arguments and invalidate caches.
        if !opcode.is_call_pure() {
            self.heap_cache.mark_escaped_args(args);
            self.heap_cache.invalidate_caches_for_escaped();
        }
        result
    }

    pub fn call_void_typed(&mut self, func_ptr: *const (), args: &[OpRef], arg_types: &[Type]) {
        let _ = self.call_typed(OpCode::CallN, func_ptr, args, arg_types, Type::Void);
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
        self.heap_cache.mark_escaped_args(args);
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
        if let Some(cached) = self
            .heap_cache
            .call_loopinvariant_lookup(descr_index, arg0_int)
        {
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
        self.heap_cache
            .call_loopinvariant_cache(descr_index, arg0_int, result);
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
        let descr = make_call_assembler_descr(target.number, arg_types, ret_type);
        self.record_op_with_descr(opcode, args, descr)
    }

    /// Emit CALL_ASSEMBLER_I by token number, without needing a `&JitCellToken`.
    ///
    /// Assumes all args are `Type::Int`. For mixed-type args, use
    /// `call_assembler_int_by_number_typed` instead.
    pub fn call_assembler_int_by_number(&mut self, target_number: u64, args: &[OpRef]) -> OpRef {
        let arg_types: Vec<Type> = args.iter().map(|_| Type::Int).collect();
        let descr = make_call_assembler_descr(target_number, &arg_types, Type::Int);
        self.record_op_with_descr(OpCode::CallAssemblerI, args, descr)
    }

    /// Emit CALL_ASSEMBLER_I by token number with explicit arg types.
    pub fn call_assembler_int_by_number_typed(
        &mut self,
        target_number: u64,
        args: &[OpRef],
        arg_types: &[Type],
    ) -> OpRef {
        let descr = make_call_assembler_descr(target_number, arg_types, Type::Int);
        self.record_op_with_descr(OpCode::CallAssemblerI, args, descr)
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
    /// `force_token` is the force token for the current JIT frame.
    ///
    /// The optimizer replaces this with a virtual struct, so if the vref
    /// never escapes, no allocation happens.
    pub fn virtual_ref_r(&mut self, virtual_obj: OpRef, force_token: OpRef) -> OpRef {
        self.record_op(OpCode::VirtualRefR, &[virtual_obj, force_token])
    }

    /// Record VIRTUAL_REF_I: create a virtual reference (int-typed result).
    pub fn virtual_ref_i(&mut self, virtual_obj: OpRef, force_token: OpRef) -> OpRef {
        self.record_op(OpCode::VirtualRefI, &[virtual_obj, force_token])
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

    // ── Convenience methods for common trace patterns ───────────────

    /// Pop two operands, record a binary operation, push the result.
    ///
    /// Handles the common stack pattern: `a, b → op(a, b)`.
    /// Note: pops in stack order (top first), but passes to IR as `[second, first]`
    /// so that the left operand comes first.
    pub fn trace_binop(&mut self, stack: &mut SymbolicStack, opcode: OpCode) {
        let r1 = stack.pop().unwrap();
        let r2 = stack.pop().unwrap();
        let result = self.record_op(opcode, &[r2, r1]);
        stack.push(result);
    }

    /// Push a constant integer value onto the symbolic stack.
    pub fn trace_push_const(&mut self, stack: &mut SymbolicStack, value: i64) {
        let opref = self.const_int(value);
        stack.push(opref);
    }

    /// Pop one value and call a void function with it.
    ///
    /// Common pattern for output operations (e.g., POPNUM, POPCHAR).
    pub fn trace_call_void_1(&mut self, stack: &mut SymbolicStack, func_ptr: *const ()) {
        let value = stack.pop().unwrap();
        self.call_void(func_ptr, &[value]);
    }

    /// Pop a boolean-like stack value, record the matching guard, and return
    /// whether tracing should continue or close the loop.
    ///
    /// `branch_taken` is the runtime branch result, while `taken_when_true`
    /// describes whether the interpreter takes the branch on a non-zero value.
    pub fn trace_branch_guard(
        &mut self,
        stack: &mut SymbolicStack,
        branch_taken: bool,
        taken_when_true: bool,
        num_live: usize,
        close_loop_on_taken: bool,
    ) -> TraceAction {
        let cond = stack.pop().unwrap();
        let opcode = if branch_taken == taken_when_true {
            OpCode::GuardTrue
        } else {
            OpCode::GuardFalse
        };
        self.record_guard(opcode, &[cond], num_live);
        if branch_taken && close_loop_on_taken {
            TraceAction::CloseLoop
        } else {
            TraceAction::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit_state::JitState;
    use majit_codegen::JitCellToken;
    use majit_ir::Type;

    extern "C" fn dummy_call_target() {}

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
        let token = JitCellToken::new(777);
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
                ctx.vable_setfield(self.vable_ref, 0, self.field_val);
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                // Re-read field 0 from heap
                let new_ref = ctx.vable_getfield_int(self.vable_ref, 0);
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
                ctx.vable_setfield(self.vable_ref, 0, self.field_val);
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                let new_ref = ctx.vable_getfield_ref(self.vable_ref, 0);
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
                ctx.vable_setfield(self.vable_ref, 0, self.field_val);
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                let new_ref = ctx.vable_getfield_float(self.vable_ref, 0);
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
        info
    }

    fn make_test_vable_info_with_array() -> crate::virtualizable::VirtualizableInfo {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field("locals", Type::Int, 24);
        info
    }

    #[test]
    fn standard_vable_getfield_reads_from_boxes() {
        let info = make_test_vable_info();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int); // pc
        let box1 = recorder.record_input_arg(Type::Int); // sp
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(&info, vable, &[box0, box1], &[]);

        // getfield with offset=8 → static field 0 → box0
        let result = ctx.vable_getfield_int(vable, 8);
        assert_eq!(result, box0);
        // getfield with offset=16 → static field 1 → box1
        let result = ctx.vable_getfield_int(vable, 16);
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
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let box1 = recorder.record_input_arg(Type::Int);
        let new_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(&info, vable, &[box0, box1], &[]);

        // setfield offset=8 → updates box0
        ctx.vable_setfield(vable, 8, new_val);

        // Box 0 should now be new_val
        let result = ctx.vable_getfield_int(vable, 8);
        assert_eq!(result, new_val);
        // Box 1 unchanged
        let result = ctx.vable_getfield_int(vable, 16);
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

        let _result = ctx.vable_getfield_int(vable, 8);

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

        ctx.vable_setfield(vable, 8, val);

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

        ctx.init_virtualizable_boxes(&info, vable, &[box0, box1], &[]);

        // Unknown offset (999) → fallback to heap op
        let _result = ctx.vable_getfield_int(vable, 999);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcI);
    }

    #[test]
    fn standard_vable_getfield_ref_reads_from_boxes() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("obj", Type::Ref, 8);

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(&info, vable, &[box0], &[]);

        let result = ctx.vable_getfield_ref(vable, 8);
        assert_eq!(result, box0);

        let ops = take_all_ops(ctx);
        assert!(ops.is_empty());
    }

    #[test]
    fn standard_vable_getfield_float_reads_from_boxes() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("val", Type::Float, 8);

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Float);
        let mut ctx = TraceCtx::new(recorder, 0);

        ctx.init_virtualizable_boxes(&info, vable, &[box0], &[]);

        let result = ctx.vable_getfield_float(vable, 8);
        assert_eq!(result, box0);

        let ops = take_all_ops(ctx);
        assert!(ops.is_empty());
    }

    #[test]
    fn vable_getarrayitem_reads_from_boxes() {
        let info = make_test_vable_info_with_array();
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
            &[box_pc, box_arr0, box_arr1, box_arr2],
            &[3], // array has 3 elements
        );

        // Array field offset=24, item_index=0 → box_arr0
        let r0 = ctx.vable_getarrayitem_int_vable(vable, 24, 0);
        assert_eq!(r0, box_arr0);
        // item_index=1 → box_arr1
        let r1 = ctx.vable_getarrayitem_int_vable(vable, 24, 1);
        assert_eq!(r1, box_arr1);
        // item_index=2 → box_arr2
        let r2 = ctx.vable_getarrayitem_int_vable(vable, 24, 2);
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
            &[box_pc, box_arr0, box_arr1],
            &[2], // array has 2 elements
        );

        // Write to array[1]
        ctx.vable_setarrayitem_vable(vable, 24, 1, new_val);

        // Read back: array[0] unchanged, array[1] updated
        let r0 = ctx.vable_getarrayitem_int_vable(vable, 24, 0);
        assert_eq!(r0, box_arr0);
        let r1 = ctx.vable_getarrayitem_int_vable(vable, 24, 1);
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

        ctx.init_virtualizable_boxes(&info, vable, &[box_pc, box_arr0], &[1]);

        // Unknown array field offset → fallback
        let _r = ctx.vable_getarrayitem_int_vable(vable, 999, 0);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetarrayitemGcI);
    }

    #[test]
    fn collect_virtualizable_boxes_returns_current_state() {
        let info = make_test_vable_info();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let box1 = recorder.record_input_arg(Type::Int);
        let new_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(recorder, 0);

        // Before init: None
        assert!(ctx.collect_virtualizable_boxes().is_none());

        ctx.init_virtualizable_boxes(&info, vable, &[box0, box1], &[]);

        // After init: has boxes (field0, field1, vable_ref sentinel)
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes, vec![box0, box1, vable]);

        // After mutation
        ctx.vable_setfield(vable, 8, new_val);
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes, vec![new_val, box1, vable]);
    }

    #[test]
    fn gen_store_back_in_vable_uses_field_and_array_descrs() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field("locals", Type::Ref, 24);

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Ref);
        let box_arr1 = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(recorder, 0);
        ctx.init_virtualizable_boxes(&info, vable, &[box_pc, box_arr0, box_arr1], &[2]);

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
        ctx.init_virtualizable_boxes(&info, vable, &[box_pc, box_arr0], &[1]);

        ctx.gen_store_back_in_vable(other_vable);

        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "nonstandard virtualizable must not use standard store-back path"
        );
    }
}
