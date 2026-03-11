/// Heap optimization pass: caches field/array reads and eliminates redundant loads/stores.
///
/// Translated from rpython/jit/metainterp/optimizeopt/heap.py.
///
/// Optimizations performed:
/// - Read-after-write elimination: SETFIELD then GETFIELD on same obj/field -> use cached value
/// - Write-after-write elimination: two SETFIELDs on same obj/field -> only keep the last
/// - Read-after-read elimination: two GETFIELDs on same obj/field -> reuse first result
/// - Same for array items (SETARRAYITEM_GC / GETARRAYITEM_GC) with constant index
/// - Cache invalidation on calls and side-effecting operations
/// - Lazy set emission: SETFIELD_GC is delayed until a guard or side-effecting op forces it
/// - GUARD_NOT_INVALIDATED deduplication
use std::collections::HashMap;

use majit_ir::{Op, OpCode, OpRef};

use crate::{OptContext, OptimizationPass, PassResult};

/// Cache key for a field access: (struct OpRef, field descriptor index).
type FieldKey = (OpRef, u32);

/// Cache key for an array item access: (array OpRef, descriptor index, constant array index).
type ArrayItemKey = (OpRef, u32, i64);

/// Heap optimization pass.
///
/// Caches field and array item values to eliminate redundant loads, and delays
/// store emission (lazy sets) to enable write-after-write elimination.
pub struct OptHeap {
    /// Cached field values: field_key -> value OpRef.
    cached_fields: HashMap<FieldKey, OpRef>,
    /// Pending (lazy) setfields: field_key -> the SETFIELD_GC op.
    lazy_setfields: HashMap<FieldKey, Op>,
    /// Cached array items: array_item_key -> value OpRef.
    cached_arrayitems: HashMap<ArrayItemKey, OpRef>,
    /// Pending (lazy) setarrayitems: array_item_key -> the SETARRAYITEM_GC op.
    lazy_setarrayitems: HashMap<ArrayItemKey, Op>,
    /// Whether we've already emitted a GUARD_NOT_INVALIDATED.
    seen_guard_not_invalidated: bool,
}

impl OptHeap {
    pub fn new() -> Self {
        OptHeap {
            cached_fields: HashMap::new(),
            lazy_setfields: HashMap::new(),
            cached_arrayitems: HashMap::new(),
            lazy_setarrayitems: HashMap::new(),
            seen_guard_not_invalidated: false,
        }
    }

    /// Build the field cache key from a GETFIELD or SETFIELD op.
    ///
    /// For GETFIELD_GC_I/R/F: args = [obj], descr = field descriptor.
    /// For SETFIELD_GC: args = [obj, value], descr = field descriptor.
    fn field_key(op: &Op) -> Option<FieldKey> {
        let descr = op.descr.as_ref()?;
        let obj = op.arg(0);
        Some((obj, descr.index()))
    }

    /// Build the array item cache key from a GETARRAYITEM or SETARRAYITEM op.
    ///
    /// For GETARRAYITEM_GC_I/R/F: args = [array, index], descr = array descriptor.
    /// For SETARRAYITEM_GC: args = [array, index, value], descr = array descriptor.
    /// Returns None if index is not a known constant.
    fn arrayitem_key(op: &Op, ctx: &OptContext) -> Option<ArrayItemKey> {
        let descr = op.descr.as_ref()?;
        let array = op.arg(0);
        let index_ref = op.arg(1);
        let index_val = ctx.get_constant_int(index_ref)?;
        Some((array, descr.index(), index_val))
    }

    /// Force all pending lazy setfields: emit the stored ops and cache their values.
    fn force_all_lazy_setfields(&mut self, ctx: &mut OptContext) {
        let pending: Vec<(FieldKey, Op)> = self.lazy_setfields.drain().collect();
        for (key, op) in pending {
            // The written value is the second arg of SETFIELD_GC.
            let value_ref = op.arg(1);
            ctx.emit(op);
            self.cached_fields.insert(key, value_ref);
        }
    }

    /// Force all pending lazy setarrayitems: emit the stored ops and cache their values.
    fn force_all_lazy_setarrayitems(&mut self, ctx: &mut OptContext) {
        let pending: Vec<(ArrayItemKey, Op)> = self.lazy_setarrayitems.drain().collect();
        for (key, op) in pending {
            // The written value is the third arg of SETARRAYITEM_GC.
            let value_ref = op.arg(2);
            ctx.emit(op);
            self.cached_arrayitems.insert(key, value_ref);
        }
    }

    /// Force all pending lazy stores (both fields and array items).
    fn force_all_lazy(&mut self, ctx: &mut OptContext) {
        self.force_all_lazy_setfields(ctx);
        self.force_all_lazy_setarrayitems(ctx);
    }

    /// Invalidate all caches (field and array). Called on side-effecting operations.
    fn invalidate_caches(&mut self) {
        self.cached_fields.clear();
        self.cached_arrayitems.clear();
    }

    // ── Handlers for specific opcodes ──

    fn optimize_getfield(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let key = match Self::field_key(op) {
            Some(k) => k,
            None => return PassResult::Emit(op.clone()),
        };

        // Check lazy set first: if there is a pending SETFIELD for this key,
        // the value is the second arg of that pending op.
        if let Some(lazy_op) = self.lazy_setfields.get(&key) {
            let cached = lazy_op.arg(1);
            ctx.replace_op(op.pos, cached);
            return PassResult::Remove;
        }

        // Check read cache.
        if let Some(&cached) = self.cached_fields.get(&key) {
            let cached = ctx.get_replacement(cached);
            ctx.replace_op(op.pos, cached);
            return PassResult::Remove;
        }

        // Cache miss: emit the load and cache the result.
        self.cached_fields.insert(key, op.pos);
        PassResult::Emit(op.clone())
    }

    fn optimize_setfield(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let key = match Self::field_key(op) {
            Some(k) => k,
            None => return PassResult::Emit(op.clone()),
        };

        let new_value = op.arg(1);

        // Check if we already have this value cached (writing the same value again).
        if let Some(lazy_op) = self.lazy_setfields.get(&key) {
            if lazy_op.arg(1) == new_value {
                // Writing the same value as the pending lazy set -> redundant.
                return PassResult::Remove;
            }
        } else if let Some(&cached) = self.cached_fields.get(&key) {
            let cached = ctx.get_replacement(cached);
            if cached == new_value {
                // Writing the same value already in the cache -> redundant.
                return PassResult::Remove;
            }
        }

        // Write-after-write: if there is already a lazy set for this key,
        // replace it (the old write is dead).
        // Either way, store as a new lazy set.
        self.lazy_setfields.insert(key, op.clone());
        self.cached_fields.remove(&key);

        PassResult::Remove
    }

    fn optimize_getarrayitem(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let key = match Self::arrayitem_key(op, ctx) {
            Some(k) => k,
            None => return PassResult::Emit(op.clone()),
        };

        // Check lazy set first.
        if let Some(lazy_op) = self.lazy_setarrayitems.get(&key) {
            let cached = lazy_op.arg(2);
            ctx.replace_op(op.pos, cached);
            return PassResult::Remove;
        }

        // Check read cache.
        if let Some(&cached) = self.cached_arrayitems.get(&key) {
            let cached = ctx.get_replacement(cached);
            ctx.replace_op(op.pos, cached);
            return PassResult::Remove;
        }

        // Cache miss: emit and cache.
        self.cached_arrayitems.insert(key, op.pos);
        PassResult::Emit(op.clone())
    }

    fn optimize_setarrayitem(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let key = match Self::arrayitem_key(op, ctx) {
            Some(k) => k,
            None => {
                // Non-constant index: force all lazy array stores and invalidate array cache.
                self.force_all_lazy_setarrayitems(ctx);
                self.cached_arrayitems.clear();
                return PassResult::Emit(op.clone());
            }
        };

        let new_value = op.arg(2);

        // Check if writing the same value.
        if let Some(lazy_op) = self.lazy_setarrayitems.get(&key) {
            if lazy_op.arg(2) == new_value {
                return PassResult::Remove;
            }
        } else if let Some(&cached) = self.cached_arrayitems.get(&key) {
            let cached = ctx.get_replacement(cached);
            if cached == new_value {
                return PassResult::Remove;
            }
        }

        // Write-after-write or new lazy set.
        self.lazy_setarrayitems.insert(key, op.clone());
        self.cached_arrayitems.remove(&key);

        PassResult::Remove
    }

    /// Handle operations that may have side effects.
    /// Forces lazy sets and invalidates caches as needed.
    fn handle_side_effects(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let opcode = op.opcode;

        // Guards: force lazy sets but keep caches (guards don't mutate the heap).
        if opcode.is_guard() {
            self.force_all_lazy(ctx);
            return PassResult::Emit(op.clone());
        }

        // Final operations (Jump, Finish): force everything.
        if opcode.is_final() {
            self.force_all_lazy(ctx);
            return PassResult::Emit(op.clone());
        }

        // Calls: force lazy sets and invalidate caches.
        if opcode.is_call() {
            self.force_all_lazy(ctx);
            self.invalidate_caches();
            return PassResult::Emit(op.clone());
        }

        // Other side-effecting ops: force and invalidate.
        if !opcode.has_no_side_effect() && !opcode.is_ovf() {
            self.force_all_lazy(ctx);
            self.invalidate_caches();
            return PassResult::Emit(op.clone());
        }

        // Pure / no-side-effect / overflow ops: pass through.
        PassResult::Emit(op.clone())
    }
}

impl Default for OptHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for OptHeap {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        match op.opcode {
            // ── Field reads ──
            OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => {
                self.optimize_getfield(op, ctx)
            }

            // ── Field writes ──
            OpCode::SetfieldGc => self.optimize_setfield(op, ctx),

            // ── Array item reads ──
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcF => {
                self.optimize_getarrayitem(op, ctx)
            }

            // ── Array item writes ──
            OpCode::SetarrayitemGc => self.optimize_setarrayitem(op, ctx),

            // ── GUARD_NOT_INVALIDATED deduplication ──
            OpCode::GuardNotInvalidated => {
                if self.seen_guard_not_invalidated {
                    PassResult::Remove
                } else {
                    self.seen_guard_not_invalidated = true;
                    self.force_all_lazy(ctx);
                    PassResult::Emit(op.clone())
                }
            }

            // ── Quasi-immutable field: treat as read + guard_not_invalidated ──
            // The QUASIIMMUT_FIELD op marks a field that rarely changes.
            // The optimizer replaces the field read with the cached value and
            // emits GUARD_NOT_INVALIDATED to ensure validity.
            OpCode::QuasiimmutField => {
                if !self.seen_guard_not_invalidated {
                    self.seen_guard_not_invalidated = true;
                    // Emit a GUARD_NOT_INVALIDATED for the quasi-immutable promise.
                    let guard_op = Op::new(OpCode::GuardNotInvalidated, &[]);
                    self.force_all_lazy(ctx);
                    ctx.emit(guard_op);
                }
                // The QUASIIMMUT_FIELD itself is a no-op marker.
                PassResult::Remove
            }

            // ── SETFIELD_RAW / SETARRAYITEM_RAW: no effect on GC caches ──
            OpCode::SetfieldRaw | OpCode::SetarrayitemRaw => PassResult::Emit(op.clone()),

            // ── Everything else: check for side effects ──
            _ => self.handle_side_effects(op, ctx),
        }
    }

    fn setup(&mut self) {
        self.cached_fields.clear();
        self.lazy_setfields.clear();
        self.cached_arrayitems.clear();
        self.lazy_setarrayitems.clear();
        self.seen_guard_not_invalidated = false;
    }

    fn flush(&mut self) {
        // All lazy sets should have been forced by the final op (Jump/Finish).
        // Clear state as a safety measure.
        debug_assert!(
            self.lazy_setfields.is_empty() && self.lazy_setarrayitems.is_empty(),
            "OptHeap: unflushed lazy sets at end of trace"
        );
        self.cached_fields.clear();
        self.lazy_setfields.clear();
        self.cached_arrayitems.clear();
        self.lazy_setarrayitems.clear();
    }

    fn name(&self) -> &'static str {
        "heap"
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use majit_ir::{Descr, DescrRef, Op, OpCode, OpRef};

    use crate::optimizer::Optimizer;
    use crate::{OptContext, OptimizationPass, PassResult};

    use super::OptHeap;

    /// Minimal descriptor for tests, identified by its index.
    #[derive(Debug)]
    struct TestDescr(u32);

    impl Descr for TestDescr {
        fn index(&self) -> u32 {
            self.0
        }
    }

    fn descr(idx: u32) -> DescrRef {
        Arc::new(TestDescr(idx))
    }

    /// Helper: assign sequential positions to ops.
    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
    }

    /// Run a single OptHeap pass over the given ops.
    fn run_heap_opt(ops: &mut [Op]) -> Vec<Op> {
        assign_positions(ops);
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptHeap::new()));
        opt.optimize(ops)
    }

    // ── Test 1: SETFIELD then GETFIELD → read from cache ──

    #[test]
    fn test_setfield_then_getfield_cached() {
        // setfield_gc(p0, i1, descr=d0)
        // i2 = getfield_gc_i(p0, descr=d0)   <- should be eliminated, replaced by i1
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            // Terminate trace so lazy set is forced.
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Expect: SETFIELD_GC (forced by Jump) + Jump. The GETFIELD is eliminated.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 2: Two GETFIELDs on same object/field → second eliminated ──

    #[test]
    fn test_getfield_read_after_read() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // i2 = getfield_gc_i(p0, descr=d0)   <- eliminated, reuse i1
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Only the first GETFIELD + Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 3: SETFIELD then SETFIELD → first eliminated (write-after-write) ──

    #[test]
    fn test_setfield_write_after_write() {
        // setfield_gc(p0, i1, descr=d0)
        // setfield_gc(p0, i2, descr=d0)   <- first is dead
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(102)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Only the second SETFIELD (forced at Jump) + Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[0].args[1], OpRef(102)); // second value
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 4: SETFIELD then CALL then GETFIELD → cache invalidated ──

    #[test]
    fn test_setfield_call_invalidates_cache() {
        // setfield_gc(p0, i1, descr=d0)
        // call_n(...)
        // i2 = getfield_gc_i(p0, descr=d0)   <- cache invalidated by call, must emit
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // SETFIELD (forced before call) + CALL + GETFIELD (re-emitted) + Jump.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Test 5: SETFIELD on different objects → both cached independently ──

    #[test]
    fn test_setfield_different_objects() {
        // setfield_gc(p0, i1, descr=d0)
        // setfield_gc(p1, i2, descr=d0)
        // i3 = getfield_gc_i(p0, descr=d0)   <- cached from first set (i1)
        // i4 = getfield_gc_i(p1, descr=d0)   <- cached from second set (i2)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(200), OpRef(201)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(200)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Both SETFIELDs (forced at Jump) + Jump. Both GETFIELDs eliminated.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 6: Array items: SETARRAYITEM then GETARRAYITEM → cached ──

    #[test]
    fn test_setarrayitem_then_getarrayitem_cached() {
        // setarrayitem_gc(p0, i_idx, i_val, descr=d0)
        // i2 = getarrayitem_gc_i(p0, i_idx, descr=d0)   <- eliminated
        // We need i_idx to be a known constant.
        let d = descr(0);
        let idx = OpRef(50);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(100), idx, OpRef(101)],
                d.clone(),
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        // We need to make the index a known constant in the context.
        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx, majit_ir::Value::Int(3));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                PassResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                PassResult::Remove => {}
                PassResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                PassResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        // SETARRAYITEM (forced at Jump) + Jump. GETARRAYITEM eliminated.
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(opcodes, vec![OpCode::SetarrayitemGc, OpCode::Jump]);
    }

    // ── Test 7: Guard forces lazy sets ──

    #[test]
    fn test_guard_forces_lazy_setfield() {
        // setfield_gc(p0, i1, descr=d0)     <- lazy, not emitted yet
        // guard_true(i_cond)                <- forces the lazy set
        // i2 = getfield_gc_i(p0, descr=d0) <- still cached (guards don't invalidate)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::with_descr(OpCode::GuardTrue, &[OpRef(200)], descr(99)),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // SETFIELD (forced by guard) + GUARD_TRUE + Jump.
        // GETFIELD is eliminated (cache survives guards).
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::GuardTrue);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 8: GUARD_NOT_INVALIDATED deduplication ──

    #[test]
    fn test_guard_not_invalidated_dedup() {
        let d = descr(99);
        let mut ops = vec![
            Op::with_descr(OpCode::GuardNotInvalidated, &[], d.clone()),
            Op::with_descr(OpCode::GuardNotInvalidated, &[], d.clone()),
            Op::with_descr(OpCode::GuardNotInvalidated, &[], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Only one GUARD_NOT_INVALIDATED + Jump.
        let gni_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNotInvalidated)
            .count();
        assert_eq!(gni_count, 1);
        assert_eq!(result.last().unwrap().opcode, OpCode::Jump);
    }

    // ── Test 9: Different field descriptors are independent ──

    #[test]
    fn test_different_field_descriptors() {
        // setfield_gc(p0, i1, descr=d0)
        // i2 = getfield_gc_i(p0, descr=d1)   <- different descriptor, NOT cached
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d0),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD (emitted during propagation) + SETFIELD (forced at Jump) + Jump.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 10: SETFIELD_RAW does not affect GC caches ──

    #[test]
    fn test_setfield_raw_no_effect_on_gc_cache() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // setfield_raw(p1, i2, descr=d1)     <- RAW, no effect on GC caches
        // i3 = getfield_gc_i(p0, descr=d0)   <- still cached from first read
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0.clone()),
            Op::with_descr(OpCode::SetfieldRaw, &[OpRef(200), OpRef(201)], d1),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + SETFIELD_RAW + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::SetfieldRaw);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 11: Writing same value is redundant ──

    #[test]
    fn test_setfield_same_value_redundant() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // setfield_gc(p0, i1, descr=d0)   <- writing back the same value, redundant
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            // pos=0 will be the GETFIELD result; setfield writes it back
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(0)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + Jump only. SETFIELD removed (writing same value).
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 12: Pure/overflow ops don't invalidate ──

    #[test]
    fn test_pure_ops_dont_invalidate() {
        // i1 = getfield_gc_i(p0, descr=d0)
        // i2 = int_add(i1, i1)              <- pure, no invalidation
        // i3 = getfield_gc_i(p0, descr=d0)  <- still cached
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + INT_ADD + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::IntAdd);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 13: Ref and float field variants ──

    #[test]
    fn test_getfield_ref_cached() {
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    #[test]
    fn test_getfield_float_cached() {
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcF);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 14: Array write-after-write ──

    #[test]
    fn test_setarrayitem_write_after_write() {
        let d = descr(0);
        let idx = OpRef(50);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(100), idx, OpRef(101)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(100), idx, OpRef(102)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx, majit_ir::Value::Int(5));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                PassResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                PassResult::Remove => {}
                PassResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                PassResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        // Only the second SETARRAYITEM (forced at Jump) + Jump.
        let result_opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(result_opcodes, vec![OpCode::SetarrayitemGc, OpCode::Jump]);
        // Verify it's the second value.
        let set_op = &ctx.new_operations[0];
        assert_eq!(set_op.args[2], OpRef(102));
    }

    // ── Test 15: Overflow ops don't invalidate caches ──

    #[test]
    fn test_overflow_ops_dont_invalidate() {
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::IntAddOvf, &[OpRef(0), OpRef(0)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + INT_ADD_OVF + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::IntAddOvf);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 16: Multiple fields on same object ──

    #[test]
    fn test_multiple_fields_same_object() {
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d0.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(102)], d1.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Both SETFIELDs (forced at Jump) + Jump. Both GETFIELDs eliminated.
        assert_eq!(result.len(), 3);
        let set_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::SetfieldGc)
            .count();
        assert_eq!(set_count, 2);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }
}
