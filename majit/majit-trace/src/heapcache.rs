/// Simplified heap cache for the tracing phase.
///
/// During tracing, the heap cache tracks field reads/writes to eliminate
/// redundant loads. If we read a field from an object and it was already
/// read or written in the same trace, we can reuse the cached value.
///
/// This is a simplified version of rpython/jit/metainterp/heapcache.py.
/// Phase 0 omits versioning, array caches, and escape tracking.
use std::collections::{HashMap, HashSet};

use majit_ir::{GcRef, OpCode, OpRef};

/// Heap cache for the tracing interpreter.
///
/// Tracks field values, known classes, and allocation status during
/// a single trace recording session.
pub struct HeapCache {
    /// Field cache: (object_ref, field_descr_index) -> cached value.
    field_cache: HashMap<(OpRef, u32), OpRef>,

    /// Array item cache: (array_ref, index_opref, descr_index) -> cached value.
    /// heapcache.py: `cached_arrayitems`.
    array_cache: HashMap<(OpRef, OpRef, u32), OpRef>,

    /// Known class map: object_ref -> class pointer.
    known_class: HashMap<OpRef, GcRef>,

    /// Quasi-immutable fields known in this trace.
    /// heapcache.py: `quasi_immut_known`.
    quasi_immut_known: HashSet<(OpRef, u32)>,

    /// Set of OpRefs known to be newly allocated and not yet escaped.
    is_unescaped: HashSet<OpRef>,

    /// Set of OpRefs for which we saw the allocation during this trace.
    seen_allocation: HashSet<OpRef>,

    /// heapcache.py: known nullity — values known to be null or non-null.
    known_nullity: HashMap<OpRef, bool>,

    /// heapcache.py: cached_arraylen — cached array lengths.
    cached_arraylen: HashMap<(OpRef, u32), OpRef>,

    /// heapcache.py: likely_virtual — values likely to be virtual objects.
    likely_virtual: HashSet<OpRef>,

    /// heapcache.py: loop-invariant call result cache.
    /// (func_ptr, args_hash) → result OpRef.
    loopinvariant_call_cache: HashMap<(OpRef, u64), OpRef>,

    /// heapcache.py: escape dependencies.
    /// When value V is stored into container C via SETFIELD_GC(C, V),
    /// record V → C. If V later escapes, C must also be marked escaped.
    escape_deps: HashMap<OpRef, Vec<OpRef>>,
}

impl HeapCache {
    /// Create a new, empty heap cache.
    pub fn new() -> Self {
        HeapCache {
            field_cache: HashMap::new(),
            array_cache: HashMap::new(),
            known_class: HashMap::new(),
            quasi_immut_known: HashSet::new(),
            is_unescaped: HashSet::new(),
            seen_allocation: HashSet::new(),
            known_nullity: HashMap::new(),
            cached_arraylen: HashMap::new(),
            likely_virtual: HashSet::new(),
            loopinvariant_call_cache: HashMap::new(),
            escape_deps: HashMap::new(),
        }
    }

    /// Look up a cached field value.
    ///
    /// Returns the OpRef that holds the value of `(obj, field_index)` if
    /// it was previously read or written in this trace, or None.
    pub fn getfield_cached(&self, obj: OpRef, field_index: u32) -> Option<OpRef> {
        self.field_cache.get(&(obj, field_index)).copied()
    }

    /// Record a field value (either from a GETFIELD result or a SETFIELD value).
    ///
    /// After `setfield_cached(obj, field_index, value)`, any subsequent
    /// `getfield_cached(obj, field_index)` will return `Some(value)`.
    ///
    /// When the object is not unescaped, this also invalidates entries for
    /// the same field on other objects (aliasing).
    pub fn setfield_cached(&mut self, obj: OpRef, field_index: u32, value: OpRef) {
        let obj_is_unescaped = self.is_unescaped.contains(&obj);
        if !obj_is_unescaped {
            // Potential aliasing: clear all cached values for this field
            // from objects that are not known-unescaped.
            self.field_cache.retain(|&(cached_obj, cached_field), _| {
                if cached_field != field_index {
                    return true;
                }
                // Keep entries for unescaped objects (no aliasing possible)
                self.is_unescaped.contains(&cached_obj)
            });
        }
        self.field_cache.insert((obj, field_index), value);
    }

    /// Record a field read without aliasing concerns (e.g., after GETFIELD
    /// where the value is now known).
    pub fn getfield_now_known(&mut self, obj: OpRef, field_index: u32, value: OpRef) {
        self.field_cache.insert((obj, field_index), value);
    }

    /// Invalidate all caches. Called when a side-effecting operation occurs
    /// that could modify heap state (e.g., an unknown CALL).
    pub fn invalidate_caches(&mut self) {
        self.field_cache.clear();
    }

    /// Invalidate caches for escaped objects only.
    /// Unescaped objects cannot be affected by external calls.
    pub fn invalidate_caches_for_escaped(&mut self) {
        self.field_cache
            .retain(|&(obj, _), _| self.is_unescaped.contains(&obj));
    }

    /// Record a new object allocation. The object is marked as unescaped
    /// and seen-allocation.
    pub fn new_object(&mut self, opref: OpRef) {
        self.is_unescaped.insert(opref);
        self.seen_allocation.insert(opref);
    }

    /// Mark an object as escaped. Its cached fields for aliased accesses
    /// may no longer be trusted after external calls.
    pub fn mark_escaped(&mut self, opref: OpRef) {
        self.is_unescaped.remove(&opref);
    }

    /// heapcache.py: _escape_box — recursively escape an object and
    /// all values stored into it via SETFIELD_GC.
    pub fn mark_escaped_recursive(&mut self, opref: OpRef) {
        if !self.is_unescaped.remove(&opref) {
            return; // already escaped or not tracked
        }
        // Propagate escape to all values stored in this object's fields.
        if let Some(deps) = self.escape_deps.remove(&opref) {
            for dep in deps {
                self.mark_escaped_recursive(dep);
            }
        }
    }

    /// Record that the class of an object is now known (e.g., after GUARD_CLASS).
    pub fn class_now_known(&mut self, opref: OpRef, class: GcRef) {
        self.known_class.insert(opref, class);
    }

    /// Check if the class of an object is known.
    pub fn is_class_known(&self, opref: OpRef) -> bool {
        self.known_class.contains_key(&opref)
    }

    /// Get the known class of an object, if available.
    pub fn get_known_class(&self, opref: OpRef) -> Option<GcRef> {
        self.known_class.get(&opref).copied()
    }

    /// Check if an object is unescaped (allocated in this trace and not
    /// yet passed to external code).
    pub fn is_unescaped(&self, opref: OpRef) -> bool {
        self.is_unescaped.contains(&opref)
    }

    /// Check if we saw the allocation of this object in the current trace.
    pub fn saw_allocation(&self, opref: OpRef) -> bool {
        self.seen_allocation.contains(&opref)
    }

    /// Notify the cache about an operation, potentially invalidating entries.
    ///
    /// This should be called for every operation during tracing, so the cache
    /// can track which operations affect heap state.
    pub fn notify_op(&mut self, opcode: OpCode, args: &[OpRef], result: OpRef) {
        if opcode.is_malloc() {
            self.new_object(result);
            return;
        }
        // heapcache.py: SETFIELD_GC tracking.
        // The written value becomes reachable from the container.
        // If the container later escapes, the value also escapes.
        if opcode == OpCode::SetfieldGc && args.len() >= 2 {
            let container = args[0];
            let value = args[1];
            // Record dependency: if container escapes, value escapes too.
            self.escape_deps
                .entry(container)
                .or_default()
                .push(value);
            // If container is already escaped, mark value as escaped now.
            if !self.is_unescaped.contains(&container) {
                self.mark_escaped_recursive(value);
            }
        }
        // heapcache.py: GUARD_CLASS/GUARD_NONNULL_CLASS → known class.
        if opcode == OpCode::GuardClass || opcode == OpCode::GuardNonnullClass {
            if args.len() >= 2 {
                if let Some(class_val) = args.get(1) {
                    self.class_now_known(args[0], GcRef(class_val.0 as usize));
                }
            }
            self.nullity_now_known(args[0], true);
        }
        // heapcache.py: GUARD_NONNULL → known non-null.
        if opcode == OpCode::GuardNonnull && !args.is_empty() {
            self.nullity_now_known(args[0], true);
        }
        // Calls may force/escape objects.
        if opcode.is_call() {
            for &arg in args {
                self.mark_escaped(arg);
            }
            if opcode.has_no_side_effect() {
                return;
            }
            self.invalidate_caches_for_escaped();
        }
    }

    /// heapcache.py: invalidate_caches_varargs(descrs, args)
    /// Selectively invalidate caches based on effect info.
    pub fn invalidate_caches_varargs(&mut self, has_side_effects: bool) {
        if has_side_effects {
            self.invalidate_caches_for_escaped();
        }
    }

    // ── Array item caching (RPython heapcache.py cached_arrayitems) ──

    /// Look up a cached array item value.
    pub fn getarrayitem_cache(&self, array: OpRef, index: OpRef, descr: u32) -> Option<OpRef> {
        self.array_cache.get(&(array, index, descr)).copied()
    }

    /// Record an array item write.
    pub fn setarrayitem_cache(&mut self, array: OpRef, index: OpRef, descr: u32, value: OpRef) {
        self.array_cache.insert((array, index, descr), value);
    }

    /// Record an array item read.
    pub fn getarrayitem_now_known(
        &mut self,
        array: OpRef,
        index: OpRef,
        descr: u32,
        value: OpRef,
    ) {
        self.array_cache.insert((array, index, descr), value);
    }

    /// Invalidate array caches for a specific array.
    pub fn invalidate_array_cache(&mut self, array: OpRef) {
        self.array_cache.retain(|&(a, _, _), _| a != array);
    }

    // ── Quasi-immutable tracking (RPython heapcache.py quasi_immut_known) ──

    /// Record that a quasi-immutable field is known.
    pub fn quasi_immut_now_known(&mut self, obj: OpRef, field_index: u32) {
        self.quasi_immut_known.insert((obj, field_index));
    }

    /// Check if a quasi-immutable field is already known.
    pub fn is_quasi_immut_known(&self, obj: OpRef, field_index: u32) -> bool {
        self.quasi_immut_known.contains(&(obj, field_index))
    }

    // ── Nullity tracking (heapcache.py nullity_now_known / is_nullity_known) ──

    /// Record that a value's nullity is known.
    /// heapcache.py: nullity_now_known(box, is_nonnull)
    pub fn nullity_now_known(&mut self, opref: OpRef, is_nonnull: bool) {
        self.known_nullity.insert(opref, is_nonnull);
    }

    /// Check if a value's nullity is known.
    /// heapcache.py: is_nullity_known(box)
    pub fn is_nullity_known(&self, opref: OpRef) -> Option<bool> {
        self.known_nullity.get(&opref).copied()
    }

    // ── Array length caching (heapcache.py arraylen_now_known / arraylen) ──

    /// Record a known array length.
    /// heapcache.py: arraylen_now_known(array, length)
    pub fn arraylen_now_known(&mut self, array: OpRef, descr: u32, length: OpRef) {
        self.cached_arraylen.insert((array, descr), length);
    }

    /// Look up a cached array length.
    /// heapcache.py: arraylen(array)
    pub fn arraylen(&self, array: OpRef, descr: u32) -> Option<OpRef> {
        self.cached_arraylen.get(&(array, descr)).copied()
    }

    // ── Likely virtual tracking (heapcache.py is_likely_virtual) ──

    /// Mark a value as likely virtual.
    /// heapcache.py: HF_LIKELY_VIRTUAL flag
    pub fn mark_likely_virtual(&mut self, opref: OpRef) {
        self.likely_virtual.insert(opref);
    }

    /// Check if a value is likely virtual.
    pub fn is_likely_virtual(&self, opref: OpRef) -> bool {
        self.likely_virtual.contains(&opref)
    }

    // ── Loop-invariant call result caching ──

    /// Record a loop-invariant call result.
    /// heapcache.py: call_loopinvariant_known_result
    pub fn call_loopinvariant_cache(&mut self, func: OpRef, args_hash: u64, result: OpRef) {
        self.loopinvariant_call_cache.insert((func, args_hash), result);
    }

    /// Look up a cached loop-invariant call result.
    pub fn call_loopinvariant_lookup(&self, func: OpRef, args_hash: u64) -> Option<OpRef> {
        self.loopinvariant_call_cache.get(&(func, args_hash)).copied()
    }

    // ── Reset variants ──

    /// Reset the entire cache state.
    pub fn reset(&mut self) {
        self.field_cache.clear();
        self.array_cache.clear();
        self.known_class.clear();
        self.quasi_immut_known.clear();
        self.is_unescaped.clear();
        self.seen_allocation.clear();
        self.known_nullity.clear();
        self.cached_arraylen.clear();
        self.likely_virtual.clear();
        self.loopinvariant_call_cache.clear();
        self.escape_deps.clear();
    }

    /// Reset but keep likely-virtual markers.
    /// heapcache.py: reset_keep_likely_virtuals()
    pub fn reset_keep_likely_virtuals(&mut self) {
        self.field_cache.clear();
        self.array_cache.clear();
        self.known_class.clear();
        self.quasi_immut_known.clear();
        self.is_unescaped.clear();
        self.seen_allocation.clear();
        self.known_nullity.clear();
        self.cached_arraylen.clear();
        self.loopinvariant_call_cache.clear();
        // likely_virtual is NOT cleared
    }
}

impl Default for HeapCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_cache_basic() {
        let mut cache = HeapCache::new();
        let obj = OpRef(0);
        let field = 1;
        let val = OpRef(2);

        assert_eq!(cache.getfield_cached(obj, field), None);

        cache.getfield_now_known(obj, field, val);
        assert_eq!(cache.getfield_cached(obj, field), Some(val));
    }

    #[test]
    fn test_field_cache_overwrite() {
        let mut cache = HeapCache::new();
        let obj = OpRef(0);
        let field = 1;

        cache.getfield_now_known(obj, field, OpRef(10));
        assert_eq!(cache.getfield_cached(obj, field), Some(OpRef(10)));

        cache.getfield_now_known(obj, field, OpRef(20));
        assert_eq!(cache.getfield_cached(obj, field), Some(OpRef(20)));
    }

    #[test]
    fn test_setfield_aliasing() {
        let mut cache = HeapCache::new();
        let obj_a = OpRef(0);
        let obj_b = OpRef(1);
        let field = 5;

        // Both objects have a known field value
        cache.getfield_now_known(obj_a, field, OpRef(10));
        cache.getfield_now_known(obj_b, field, OpRef(20));

        // Writing to obj_a (which is NOT unescaped) should invalidate
        // obj_b's field cache for the same field (potential aliasing).
        cache.setfield_cached(obj_a, field, OpRef(30));
        assert_eq!(cache.getfield_cached(obj_a, field), Some(OpRef(30)));
        assert_eq!(cache.getfield_cached(obj_b, field), None); // invalidated
    }

    #[test]
    fn test_setfield_no_aliasing_for_unescaped() {
        let mut cache = HeapCache::new();
        let obj_a = OpRef(0);
        let obj_b = OpRef(1);
        let field = 5;

        // obj_a is a newly allocated object
        cache.new_object(obj_a);
        cache.getfield_now_known(obj_a, field, OpRef(10));
        cache.getfield_now_known(obj_b, field, OpRef(20));

        // Writing to obj_a (unescaped) does NOT cause aliasing invalidation
        cache.setfield_cached(obj_a, field, OpRef(30));
        assert_eq!(cache.getfield_cached(obj_a, field), Some(OpRef(30)));
        assert_eq!(cache.getfield_cached(obj_b, field), Some(OpRef(20))); // preserved
    }

    #[test]
    fn test_invalidate_caches() {
        let mut cache = HeapCache::new();
        cache.getfield_now_known(OpRef(0), 1, OpRef(10));
        cache.getfield_now_known(OpRef(1), 2, OpRef(20));

        cache.invalidate_caches();
        assert_eq!(cache.getfield_cached(OpRef(0), 1), None);
        assert_eq!(cache.getfield_cached(OpRef(1), 2), None);
    }

    #[test]
    fn test_invalidate_caches_for_escaped() {
        let mut cache = HeapCache::new();
        let escaped_obj = OpRef(0);
        let unescaped_obj = OpRef(1);

        cache.new_object(unescaped_obj);
        cache.getfield_now_known(escaped_obj, 1, OpRef(10));
        cache.getfield_now_known(unescaped_obj, 1, OpRef(20));

        cache.invalidate_caches_for_escaped();
        assert_eq!(cache.getfield_cached(escaped_obj, 1), None);
        assert_eq!(cache.getfield_cached(unescaped_obj, 1), Some(OpRef(20)));
    }

    #[test]
    fn test_new_object() {
        let mut cache = HeapCache::new();
        let obj = OpRef(5);

        assert!(!cache.is_unescaped(obj));
        assert!(!cache.saw_allocation(obj));

        cache.new_object(obj);
        assert!(cache.is_unescaped(obj));
        assert!(cache.saw_allocation(obj));
    }

    #[test]
    fn test_mark_escaped() {
        let mut cache = HeapCache::new();
        let obj = OpRef(5);

        cache.new_object(obj);
        assert!(cache.is_unescaped(obj));

        cache.mark_escaped(obj);
        assert!(!cache.is_unescaped(obj));
        // saw_allocation is permanent
        assert!(cache.saw_allocation(obj));
    }

    #[test]
    fn test_known_class() {
        let mut cache = HeapCache::new();
        let obj = OpRef(0);
        let cls = GcRef(0x1000);

        assert!(!cache.is_class_known(obj));
        assert_eq!(cache.get_known_class(obj), None);

        cache.class_now_known(obj, cls);
        assert!(cache.is_class_known(obj));
        assert_eq!(cache.get_known_class(obj), Some(cls));
    }

    #[test]
    fn test_notify_op_malloc() {
        let mut cache = HeapCache::new();
        let result = OpRef(3);

        cache.notify_op(OpCode::New, &[], result);
        assert!(cache.is_unescaped(result));
        assert!(cache.saw_allocation(result));
    }

    #[test]
    fn test_reset() {
        let mut cache = HeapCache::new();
        cache.new_object(OpRef(0));
        cache.class_now_known(OpRef(0), GcRef(0x1000));
        cache.getfield_now_known(OpRef(0), 1, OpRef(10));

        cache.reset();
        assert!(!cache.is_unescaped(OpRef(0)));
        assert!(!cache.is_class_known(OpRef(0)));
        assert_eq!(cache.getfield_cached(OpRef(0), 1), None);
    }

    #[test]
    fn test_different_fields_independent() {
        let mut cache = HeapCache::new();
        let obj = OpRef(0);

        cache.getfield_now_known(obj, 1, OpRef(10));
        cache.getfield_now_known(obj, 2, OpRef(20));

        // Writing field 1 should not affect field 2
        cache.setfield_cached(obj, 1, OpRef(30));
        assert_eq!(cache.getfield_cached(obj, 1), Some(OpRef(30)));
        assert_eq!(cache.getfield_cached(obj, 2), Some(OpRef(20)));
    }
}
