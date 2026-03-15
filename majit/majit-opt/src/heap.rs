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
use std::collections::{HashMap, HashSet};

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
///
/// Green field optimization: immutable field caches survive cache invalidation
/// by calls and side-effecting operations. When an immutable field is read from
/// a constant object, the result is also a constant (green field folding).
///
/// Aliasing analysis: objects allocated during the trace (NEW, NEW_WITH_VTABLE,
/// NEW_ARRAY, etc.) cannot alias each other or pre-existing objects. Their field
/// caches survive writes to other objects. Objects that haven't escaped (not
/// passed to calls or stored into the heap) keep their caches across calls.
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
    /// Descriptor indices known to be immutable (green fields).
    /// Cached values for these descriptors survive invalidation.
    immutable_field_descrs: HashSet<u32>,

    // ── Aliasing analysis state ──

    /// Objects allocated during this trace (NEW/NEW_WITH_VTABLE/NEW_ARRAY/etc.).
    /// These cannot alias each other or pre-existing (input arg) objects.
    seen_allocation: HashSet<OpRef>,
    /// Subset of `seen_allocation`: objects that haven't escaped.
    /// An object escapes when it is passed to a call or stored into another
    /// object's field via SETFIELD_GC / SETARRAYITEM_GC.
    /// Caches for unescaped objects survive calls (calls can't access them).
    unescaped: HashSet<OpRef>,
}

impl OptHeap {
    pub fn new() -> Self {
        OptHeap {
            cached_fields: HashMap::new(),
            lazy_setfields: HashMap::new(),
            cached_arrayitems: HashMap::new(),
            lazy_setarrayitems: HashMap::new(),
            seen_guard_not_invalidated: false,
            immutable_field_descrs: HashSet::new(),
            seen_allocation: HashSet::new(),
            unescaped: HashSet::new(),
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

    /// Invalidate caches on calls and other side-effecting operations.
    ///
    /// Caches that survive:
    /// - Immutable (green) field caches: values never change.
    /// - Unescaped object caches: calls cannot access objects that haven't
    ///   been passed to a call or stored into the heap.
    fn invalidate_caches(&mut self) {
        let has_survivors =
            !self.immutable_field_descrs.is_empty() || !self.unescaped.is_empty();

        if has_survivors {
            self.cached_fields.retain(|&(obj, descr_idx), _| {
                if self.immutable_field_descrs.contains(&descr_idx) {
                    return true;
                }
                if self.unescaped.contains(&obj) {
                    return true;
                }
                false
            });
            self.cached_arrayitems
                .retain(|&(obj, _, _), _| self.unescaped.contains(&obj));
        } else {
            self.cached_fields.clear();
            self.cached_arrayitems.clear();
        }
    }

    /// Invalidate field caches affected by a write to `obj` for field `field_idx`.
    ///
    /// Uses aliasing analysis: objects allocated in this trace (seen_allocation)
    /// cannot alias each other. A write to a seen-allocation object only
    /// invalidates caches for unknown-origin objects with the same field.
    /// A write to an unknown-origin object invalidates all non-seen-allocation
    /// caches for that field.
    fn invalidate_field_caches_for_write(&mut self, obj: OpRef, field_idx: u32) {
        let obj_is_seen_alloc = self.seen_allocation.contains(&obj);

        self.cached_fields.retain(|&(cached_obj, cached_field), _| {
            if cached_field != field_idx {
                return true;
            }
            // The exact same (obj, field) entry will be replaced after this,
            // so removing it here is fine.
            if cached_obj == obj {
                return false;
            }
            if obj_is_seen_alloc {
                // Writer is a seen allocation. It can't alias other seen allocations,
                // so keep their caches. Only invalidate unknown-origin objects.
                self.seen_allocation.contains(&cached_obj)
            } else {
                // Writer is unknown origin. It might alias other unknown-origin
                // objects, but can't alias seen allocations.
                self.seen_allocation.contains(&cached_obj)
            }
        });
    }

    /// Mark call arguments as escaped.
    fn mark_args_escaped(&mut self, op: &Op) {
        for &arg in &op.args {
            self.unescaped.remove(&arg);
        }
    }

    // ── Handlers for specific opcodes ──

    fn optimize_getfield(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let key = match Self::field_key(op) {
            Some(k) => k,
            None => return PassResult::Emit(op.clone()),
        };

        // Register immutable field descriptors so their cache entries survive
        // invalidation by calls and side-effecting operations.
        if let Some(descr) = &op.descr {
            if descr.is_always_pure() {
                self.immutable_field_descrs.insert(key.1);
            }
        }

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

        let (obj, field_idx) = key;
        let new_value = op.arg(1);

        // The stored value escapes (it becomes reachable via the heap).
        self.unescaped.remove(&new_value);

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

        // Aliasing-aware invalidation: only invalidate caches that might
        // be affected by this write.
        self.invalidate_field_caches_for_write(obj, field_idx);

        // Write-after-write: if there is already a lazy set for this key,
        // replace it (the old write is dead).
        // Either way, store as a new lazy set.
        self.lazy_setfields.insert(key, op.clone());

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
        // The stored value escapes (becomes reachable via the heap).
        let stored_value = op.arg(2);
        self.unescaped.remove(&stored_value);

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
    /// Tracks allocations for aliasing analysis.
    fn handle_side_effects(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult {
        let opcode = op.opcode;

        // Track allocations for aliasing analysis.
        if opcode.is_malloc() {
            self.seen_allocation.insert(op.pos);
            self.unescaped.insert(op.pos);
            return PassResult::Emit(op.clone());
        }

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

        // Calls: mark arguments as escaped, force lazy sets, and invalidate.
        if opcode.is_call() {
            self.mark_args_escaped(op);
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
        self.immutable_field_descrs.clear();
        self.seen_allocation.clear();
        self.unescaped.clear();
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
        self.immutable_field_descrs.clear();
        self.seen_allocation.clear();
        self.unescaped.clear();
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

    /// Descriptor for immutable (green) fields. `is_always_pure()` returns true.
    #[derive(Debug)]
    struct ImmutableDescr(u32);

    impl Descr for ImmutableDescr {
        fn index(&self) -> u32 {
            self.0
        }

        fn is_always_pure(&self) -> bool {
            true
        }
    }

    fn descr(idx: u32) -> DescrRef {
        Arc::new(TestDescr(idx))
    }

    fn immutable_descr(idx: u32) -> DescrRef {
        Arc::new(ImmutableDescr(idx))
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

    // ── Green field optimization tests ──

    // ── Test 17: Immutable field cache survives call invalidation ──

    #[test]
    fn test_immutable_field_survives_call() {
        // i1 = getfield_gc_i(p0, descr=immutable_d0)
        // call_n(...)                              <- invalidates mutable caches
        // i2 = getfield_gc_i(p0, descr=immutable_d0) <- still cached (immutable)
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + CALL + Jump. Second GETFIELD eliminated (immutable survives call).
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 18: Mutable field cache is still invalidated by call ──

    #[test]
    fn test_mutable_field_invalidated_by_call() {
        // i1 = getfield_gc_i(p0, descr=mutable_d0)
        // call_n(...)
        // i2 = getfield_gc_i(p0, descr=mutable_d0) <- re-emitted (mutable, invalidated)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + CALL + GETFIELD (re-emitted) + Jump.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Test 19: Mixed immutable and mutable fields: only mutable invalidated ──

    #[test]
    fn test_mixed_immutable_mutable_fields() {
        // i1 = getfield_gc_i(p0, descr=immut_d0)
        // i2 = getfield_gc_i(p0, descr=mut_d1)
        // call_n(...)
        // i3 = getfield_gc_i(p0, descr=immut_d0)  <- cached (immutable survives)
        // i4 = getfield_gc_i(p0, descr=mut_d1)    <- re-emitted (mutable invalidated)
        let d_immut = immutable_descr(0);
        let d_mut = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d_immut.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d_mut.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d_immut.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d_mut.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD(immut) + GETFIELD(mut) + CALL + GETFIELD(mut, re-emitted) + Jump.
        // GETFIELD(immut) after call is eliminated.
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI); // immutable, first read
        assert_eq!(result[1].opcode, OpCode::GetfieldGcI); // mutable, first read
        assert_eq!(result[2].opcode, OpCode::CallN);
        assert_eq!(result[3].opcode, OpCode::GetfieldGcI); // mutable, re-emitted
        assert_eq!(result[4].opcode, OpCode::Jump);
    }

    // ── Test 20: Immutable field from non-constant object still gets read-cache ──

    #[test]
    fn test_immutable_field_read_cache_no_constant() {
        // Even without a constant source object, immutable fields benefit from
        // read-after-read caching that survives side effects.
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Only first GETFIELD + Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    // ── Test 21: Immutable Ref and Float field variants survive call ──

    #[test]
    fn test_immutable_field_ref_survives_call() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcR, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcR);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_immutable_field_float_survives_call() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcF, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcF);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 22: Immutable field survives multiple calls ──

    #[test]
    fn test_immutable_field_survives_multiple_calls() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::new(OpCode::CallN, &[OpRef(201)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + CALL + CALL + Jump. Second GETFIELD eliminated.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[2].opcode, OpCode::CallN);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Test 23: Different objects with same immutable descr are independent ──

    #[test]
    fn test_immutable_field_different_objects() {
        let d = immutable_descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(200)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(300)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()), // cached
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(200)], d.clone()), // cached
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Both initial GETFIELDs + CALL + Jump. Both post-call GETFIELDs eliminated.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[1].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[2].opcode, OpCode::CallN);
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    // ── Aliasing analysis tests ──

    // ── Test 24: Two NEW objects don't alias — write to one preserves cache of the other ──

    #[test]
    fn test_seen_allocation_no_alias() {
        // p0 = new()
        // p1 = new()
        // setfield_gc(p0, i10, descr=d0)
        // i1 = getfield_gc_i(p0, descr=d0)   <- cached from set
        // setfield_gc(p1, i20, descr=d0)      <- same field, different seen alloc
        // i2 = getfield_gc_i(p0, descr=d0)   <- still cached! p1 can't alias p0
        let d = descr(0);
        let mut ops = vec![
            Op::new(OpCode::New, &[]),                                              // pos=0 -> p0
            Op::new(OpCode::New, &[]),                                              // pos=1 -> p1
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d.clone()),  // set p0.f = i10
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()),            // read p0.f -> cached
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(1), OpRef(20)], d.clone()),  // set p1.f = i20
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()),            // read p0.f -> still cached
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // NEW + NEW + SETFIELD(p0) + SETFIELD(p1) + Jump.
        // Both GETFIELDs eliminated (p0 cache survives write to p1).
        let opcodes: Vec<_> = result.iter().map(|o| o.opcode).collect();
        assert!(
            !opcodes.contains(&OpCode::GetfieldGcI),
            "both GETFIELDs should be eliminated, got: {opcodes:?}"
        );
        assert_eq!(
            result.iter().filter(|o| o.opcode == OpCode::SetfieldGc).count(),
            2
        );
    }

    // ── Test 25: Unknown-origin object write invalidates other unknown caches ──

    #[test]
    fn test_unknown_object_write_invalidates() {
        // p0 = InputRef(100), p1 = InputRef(200)  — both unknown origin
        // i1 = getfield_gc_i(p0, descr=d0)
        // setfield_gc(p1, i20, descr=d0)   <- p1 is unknown, might alias p0
        // i2 = getfield_gc_i(p0, descr=d0) <- must re-emit (might have been clobbered)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(200), OpRef(20)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD + SETFIELD + GETFIELD (re-emitted) + Jump.
        let get_count = result.iter().filter(|o| o.opcode == OpCode::GetfieldGcI).count();
        assert_eq!(get_count, 2, "second GETFIELD must be re-emitted for unknown-origin objects");
    }

    // ── Test 26: Unescaped allocation's cache survives call ──

    #[test]
    fn test_unescaped_survives_call() {
        // p0 = new()
        // setfield_gc(p0, i10, descr=d0)
        // call_n(some_func)               <- p0 NOT passed to call
        // i1 = getfield_gc_i(p0, descr=d0) <- still cached (p0 is unescaped)
        let d = descr(0);
        let mut ops = vec![
            Op::new(OpCode::New, &[]),                                              // pos=0 -> p0
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),                                  // some unrelated func
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // NEW + SETFIELD(forced by call) + CALL + Jump.
        // GETFIELD eliminated (unescaped object cache survives call).
        let opcodes: Vec<_> = result.iter().map(|o| o.opcode).collect();
        assert!(
            !opcodes.contains(&OpCode::GetfieldGcI),
            "GETFIELD should be eliminated for unescaped object, got: {opcodes:?}"
        );
    }

    // ── Test 27: Escaped allocation's cache is invalidated by call ──

    #[test]
    fn test_escaped_invalidated_by_call() {
        // p0 = new()
        // setfield_gc(p0, i10, descr=d0)
        // call_n(p0)                       <- p0 is passed to call, escapes
        // i1 = getfield_gc_i(p0, descr=d0) <- must re-emit (call might have modified p0)
        let d = descr(0);
        let mut ops = vec![
            Op::new(OpCode::New, &[]),                                              // pos=0 -> p0
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(0)]),                                    // pass p0 to call
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // NEW + SETFIELD + CALL + GETFIELD (re-emitted) + Jump.
        let get_count = result.iter().filter(|o| o.opcode == OpCode::GetfieldGcI).count();
        assert_eq!(get_count, 1, "GETFIELD must be re-emitted after escape via call");
    }

    // ── Test 28: SetfieldGc marks stored value as escaped ──

    #[test]
    fn test_setfield_marks_escape() {
        // p0 = new()       <- unescaped
        // p1 = new()       <- unescaped
        // setfield_gc(p0, i10, descr=d0)
        // setfield_gc(p1, p0, descr=d1)   <- p0 is stored into p1's field, p0 escapes
        // call_n(p1)                       <- p1 escapes; p0 already escaped via setfield
        // i1 = getfield_gc_i(p0, descr=d0) <- must re-emit (p0 escaped)
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::new(OpCode::New, &[]),                                              // pos=0 -> p0
            Op::new(OpCode::New, &[]),                                              // pos=1 -> p1
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d0.clone()), // p0.f0 = i10
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(1), OpRef(0)], d1.clone()),  // p1.f1 = p0 (p0 escapes)
            Op::new(OpCode::CallN, &[OpRef(1)]),                                    // call(p1) (p1 escapes)
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d0.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // p0 escaped via setfield, so its cache is invalidated by the call.
        let get_count = result.iter().filter(|o| o.opcode == OpCode::GetfieldGcI).count();
        assert_eq!(get_count, 1, "GETFIELD must be re-emitted after p0 escaped via setfield");
    }

    // ── Test 29: Seen-allocation cache survives write from unknown-origin object ──

    #[test]
    fn test_seen_alloc_survives_unknown_write() {
        // p0 = new()
        // setfield_gc(p0, i10, descr=d0)
        // setfield_gc(p_unknown, i20, descr=d0)  <- unknown-origin write, same field
        // i1 = getfield_gc_i(p0, descr=d0)       <- still cached (p0 is seen alloc, can't alias unknown)
        let d = descr(0);
        let mut ops = vec![
            Op::new(OpCode::New, &[]),                                              // pos=0 -> p0
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(20)], d.clone()), // unknown object
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // NEW + SETFIELD(p0) + SETFIELD(unknown) + Jump.
        // GETFIELD eliminated (p0 cache survives write to unknown object).
        let opcodes: Vec<_> = result.iter().map(|o| o.opcode).collect();
        assert!(
            !opcodes.contains(&OpCode::GetfieldGcI),
            "GETFIELD should be eliminated for seen-alloc object, got: {opcodes:?}"
        );
    }

    // ── Test 30: Different field descriptors are not affected by aliasing ──

    #[test]
    fn test_aliasing_different_fields_independent() {
        // Even with unknown-origin objects, writes to field d0 don't
        // invalidate caches for field d1.
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(200), OpRef(20)], d0.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1.clone()), // different field
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GETFIELD(d1) + SETFIELD(d0) + Jump. Second GETFIELD eliminated.
        let get_count = result.iter().filter(|o| o.opcode == OpCode::GetfieldGcI).count();
        assert_eq!(get_count, 1, "write to d0 should not invalidate d1 cache");
    }

    // ── Test 31: Unescaped array cache survives call ──

    #[test]
    fn test_unescaped_array_survives_call() {
        // p0 = new_array(5)
        // setarrayitem_gc(p0, idx, i10, descr=d0)
        // call_n(some_func)                <- p0 not passed
        // i1 = getarrayitem_gc_i(p0, idx, descr=d0) <- still cached
        let d = descr(0);
        let idx = OpRef(50);
        let mut ops = vec![
            Op::new(OpCode::NewArray, &[OpRef(5)]),                                     // pos=0 -> p0
            Op::with_descr(OpCode::SetarrayitemGc, &[OpRef(0), idx, OpRef(10)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(0), idx], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

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
                PassResult::Emit(emitted) => { ctx.emit(emitted); }
                PassResult::Remove => {}
                PassResult::Replace(replaced) => { ctx.emit(replaced); }
                PassResult::PassOn => { ctx.emit(resolved); }
            }
        }

        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert!(
            !opcodes.contains(&OpCode::GetarrayitemGcI),
            "GETARRAYITEM should be eliminated for unescaped array, got: {opcodes:?}"
        );
    }

    // ── Test 32: Multiple calls — unescaped object stays cached ──

    #[test]
    fn test_unescaped_survives_multiple_calls() {
        // p0 = new()
        // setfield_gc(p0, i10, descr=d0)
        // call_n(f1)
        // call_n(f2)
        // i1 = getfield_gc_i(p0, descr=d0) <- still cached
        let d = descr(0);
        let mut ops = vec![
            Op::new(OpCode::New, &[]),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::new(OpCode::CallN, &[OpRef(201)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let opcodes: Vec<_> = result.iter().map(|o| o.opcode).collect();
        assert!(
            !opcodes.contains(&OpCode::GetfieldGcI),
            "GETFIELD should be eliminated for unescaped object across multiple calls, got: {opcodes:?}"
        );
    }
}
