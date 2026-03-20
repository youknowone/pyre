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
use std::hash::{Hash, Hasher};

use majit_ir::{OopSpecIndex, Op, OpCode, OpRef};

use crate::{OptContext, Optimization, OptimizationResult};

/// Hash the argument OpRefs of an operation for loop-invariant call caching.
fn hash_args(args: &[OpRef]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for arg in args {
        arg.0.hash(&mut hasher);
    }
    hasher.finish()
}

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
    /// Immutable (pure) field cache — separate from cached_fields to survive
    /// all invalidation. RPython heap.py: is_always_pure() fields are never
    /// invalidated (their values never change).
    immutable_cached_fields: HashMap<FieldKey, OpRef>,
    /// Pending (lazy) setfields: field_key -> the SETFIELD_GC op.
    lazy_setfields: HashMap<FieldKey, Op>,
    /// Cached array items: array_item_key -> value OpRef.
    cached_arrayitems: HashMap<ArrayItemKey, OpRef>,
    /// Pending (lazy) setarrayitems: array_item_key -> the SETARRAYITEM_GC op.
    lazy_setarrayitems: HashMap<ArrayItemKey, Op>,
    /// Whether we've already emitted a GUARD_NOT_INVALIDATED.
    seen_guard_not_invalidated: bool,
    /// Postponed operation: held back until the next GUARD_NO_EXCEPTION.
    /// RPython heap.py: `postponed_op` — delays emission of operations
    /// that may raise (CALL_MAY_FORCE, comparison ops) until we see
    /// a GUARD_NO_EXCEPTION, ensuring correct exception semantics.
    postponed_op: Option<Op>,
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

    // ── Nullity tracking ──
    /// Values known to be non-null: proven by guards (GuardNonnull, GuardClass,
    /// GuardNonnullClass, GuardValue) or by allocation (New, NewWithVtable, etc.).
    /// Used to eliminate redundant GuardNonnull checks.
    known_nonnull: HashSet<OpRef>,

    /// Cache for loop-invariant call results: (descr_index, args_hash) -> result OpRef.
    /// Survives calls and side-effecting operations (only cleared on setup).
    loopinvariant_cache: HashMap<(u32, u64), OpRef>,

    /// Whether the last call is known to not raise (for GUARD_NO_EXCEPTION dedup).
    /// RPython heap.py: consecutive GUARD_NO_EXCEPTION can be deduplicated.
    last_call_did_not_raise: bool,

    /// Fields known to be quasi-immutable: (obj, field_idx) -> cached value OpRef.
    /// Populated by QUASIIMMUT_FIELD, consumed by subsequent GETFIELD_GC_*.
    /// Survives calls (guarded by GUARD_NOT_INVALIDATED).
    quasi_immut_cache: HashMap<FieldKey, OpRef>,
    /// heap.py: cached array lengths.
    cached_arraylens: HashMap<(OpRef, u32), OpRef>,
    /// heap.py: variable-index array cache.
    /// Key: (array, descr_index, index_opref) → value OpRef.
    cached_arrayitems_var: HashMap<(OpRef, u32, OpRef), OpRef>,
    /// heap.py: arrayinfo.getlenbound() — minimum known array lengths.
    /// When GETARRAYITEM_GC with constant index N is seen, the array
    /// must have length >= N+1. Tracked per array OpRef.
    array_min_lengths: HashMap<OpRef, i64>,
    /// RPython: Phase 2 (flush=False) — don't force lazy sets on JUMP.
    /// In RPython OptUnroll intercepts JUMP before OptHeap sees it.
    skip_flush_on_final: bool,
}

impl OptHeap {
    pub fn new() -> Self {
        OptHeap {
            cached_fields: HashMap::new(),
            immutable_cached_fields: HashMap::new(),
            lazy_setfields: HashMap::new(),
            cached_arrayitems: HashMap::new(),
            lazy_setarrayitems: HashMap::new(),
            skip_flush_on_final: false,
            seen_guard_not_invalidated: false,
            postponed_op: None,
            immutable_field_descrs: HashSet::new(),
            seen_allocation: HashSet::new(),
            unescaped: HashSet::new(),
            known_nonnull: HashSet::new(),
            loopinvariant_cache: HashMap::new(),
            last_call_did_not_raise: false,
            quasi_immut_cache: HashMap::new(),
            cached_arraylens: HashMap::new(),
            cached_arrayitems_var: HashMap::new(),
            array_min_lengths: HashMap::new(),
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

    /// heap.py: variable-index array key — use the OpRef itself as
    /// the key when the index is not a known constant. This allows
    /// caching array reads where the same variable index is used twice.
    fn arrayitem_key_variable(op: &Op) -> Option<(OpRef, u32, OpRef)> {
        let descr = op.descr.as_ref()?;
        let array = op.arg(0);
        let index_ref = op.arg(1);
        Some((array, descr.index(), index_ref))
    }

    /// heap.py: force_lazy_set — emit lazy setfields.
    /// If any lazy setfield argument references the postponed_op,
    /// emit the postponed_op first (RPython heap.py exact logic).
    fn force_all_lazy_setfields(&mut self, ctx: &mut OptContext) {
        // heap.py: check if any lazy set references the postponed op
        if let Some(ref postponed) = self.postponed_op {
            let postponed_pos = postponed.pos;
            let needs_postponed = self
                .lazy_setfields
                .values()
                .any(|op| op.args.iter().any(|a| *a == postponed_pos));
            if needs_postponed {
                if let Some(p) = self.postponed_op.take() {
                    ctx.emit(p);
                }
            }
        }
        let pending: Vec<(FieldKey, Op)> = self.lazy_setfields.drain().collect();
        for (key, mut op) in pending {
            // Resolve forwarding — virtual refs may have been forced by now
            for arg in op.args.iter_mut() {
                *arg = ctx.get_replacement(*arg);
            }
            // Skip if value ref doesn't correspond to any defined op.
            // This happens when a virtual was force-emitted but the lazy_set's
            // value OpRef points to an intermediate forwarding position that
            // was never assigned to an output op.
            let value_ref = ctx.get_replacement(op.arg(1));
            let num_inputs = ctx.num_inputs() as u32;
            // RPython heap.py: _force_elements emits ALL SetfieldGc
            // without checking if the value is defined. Constants
            // (OpRef >= 10_000) are always valid — they are compile-time
            // constant pool entries (e.g., type pointers). Skipping
            // them causes GC crashes because newly allocated objects
            // lack their type_id field.
            let is_constant = value_ref.0 >= 10_000;
            let value_is_undefined = !is_constant
                && !value_ref.is_none()
                && value_ref.0 >= num_inputs
                && !ctx
                    .new_operations
                    .iter()
                    .any(|o| o.pos == value_ref);
            if value_is_undefined {
                continue;
            }
            ctx.emit(op);
            self.cached_fields.insert(key, value_ref);
        }
    }

    /// Force all pending lazy setarrayitems: emit the stored ops and cache their values.
    fn force_all_lazy_setarrayitems(&mut self, ctx: &mut OptContext) {
        let pending: Vec<(ArrayItemKey, Op)> = self.lazy_setarrayitems.drain().collect();
        for (key, mut op) in pending {
            for arg in &mut op.args {
                *arg = ctx.get_replacement(*arg);
            }
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

    /// heap.py: force_lazy_sets_for_guard()
    ///
    /// For guards, we don't need to force ALL lazy sets. Virtual values
    /// that would be written by lazy sets can be recorded as pendingfields
    /// instead (they'll be materialized on guard failure). Only non-virtual
    /// lazy sets need to be emitted before the guard.
    ///
    /// Returns the list of ops that should go into pendingfields.
    fn force_lazy_sets_for_guard(&mut self, ctx: &mut OptContext) -> Vec<Op> {
        let mut pendingfields = Vec::new();

        // Force lazy setfields: virtual values → pendingfields, rest → emit
        let field_entries: Vec<(FieldKey, Op)> = self.lazy_setfields.drain().collect();
        for (key, op) in field_entries {
            let value_ref = ctx.get_replacement(op.arg(1));
            // heap.py: if value is virtual or unescaped, defer to pendingfields
            let is_virtual = matches!(
                ctx.get_ptr_info(value_ref),
                Some(info) if info.is_virtual()
            );
            if is_virtual || self.unescaped.contains(&value_ref) {
                pendingfields.push(op);
            } else {
                ctx.emit(op);
                self.cached_fields.insert(key, value_ref);
            }
        }

        // Force lazy setarrayitems: same logic.
        // heap.py line 631-632: the struct/array pointer must NOT be virtual
        // (only the stored value can be virtual/unescaped).
        let array_entries: Vec<(ArrayItemKey, Op)> = self.lazy_setarrayitems.drain().collect();
        for (key, op) in array_entries {
            let value_ref = ctx.get_replacement(op.arg(2));
            let is_virtual = matches!(
                ctx.get_ptr_info(value_ref),
                Some(info) if info.is_virtual()
            );
            if is_virtual || self.unescaped.contains(&value_ref) {
                pendingfields.push(op);
            } else {
                ctx.emit(op);
                self.cached_arrayitems.insert(key, value_ref);
            }
        }

        pendingfields
    }

    /// Invalidate caches on calls and other side-effecting operations.
    ///
    /// Caches that survive:
    /// - Immutable (green) field caches: values never change.
    /// - Unescaped object caches: calls cannot access objects that haven't
    ///   been passed to a call or stored into the heap.
    fn invalidate_caches(&mut self) {
        let has_survivors = !self.immutable_field_descrs.is_empty() || !self.unescaped.is_empty();

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
            self.cached_arrayitems_var
                .retain(|&(obj, _, _), _| self.unescaped.contains(&obj));
        } else {
            self.cached_fields.clear();
            self.immutable_cached_fields.clear();
            self.cached_arrayitems.clear();
            self.cached_arrayitems_var.clear();
            self.array_min_lengths.clear();
        }

        // Nullity: allocated objects are permanently non-null.
        // Other nonnull knowledge is invalidated conservatively.
        if !self.seen_allocation.is_empty() {
            self.known_nonnull
                .retain(|v| self.seen_allocation.contains(v));
        } else {
            self.known_nonnull.clear();
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

    /// Invalidate only array cache entries affected by an ARRAYCOPY/ARRAYMOVE.
    ///
    /// Instead of clearing all array caches, only remove entries for the
    /// destination array within the copied index range. Entries for other
    /// arrays, or entries outside the range, are kept.
    fn invalidate_array_caches_for_copy(
        &mut self,
        dest_ref: OpRef,
        dest_start: Option<i64>,
        length: Option<i64>,
    ) {
        self.cached_arrayitems
            .retain(|&(obj, _descr_idx, index), _| {
                if obj != dest_ref {
                    return true; // different array, keep
                }
                // If we know both start and length, only invalidate entries within range
                if let (Some(start), Some(len)) = (dest_start, length) {
                    if index < start || index >= start + len {
                        return true; // outside copy range, keep
                    }
                }
                false // within range or unknown range, invalidate
            });
        self.cached_arrayitems_var
            .retain(|&(obj, _descr_idx, _index_ref), _| obj != dest_ref);
    }

    /// Extract OopSpecIndex from a call op's descriptor, if available.
    fn get_oopspec_index(op: &Op) -> OopSpecIndex {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.effect_info().oopspec_index)
            .unwrap_or(OopSpecIndex::None)
    }

    /// Mark call arguments as escaped.
    fn mark_args_escaped(&mut self, op: &Op) {
        for &arg in &op.args {
            self.unescaped.remove(&arg);
        }
    }

    /// heap.py: check if a call has random effects (EffectInfo).
    /// Calls with HAS_RANDOM_EFFECTS invalidate all caches.
    /// Calls without it only invalidate non-immutable/non-unescaped entries.
    fn call_has_random_effects(op: &Op) -> bool {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.effect_info().has_random_effects())
            .unwrap_or(true) // conservative: assume random effects if unknown
    }

    /// heap.py: check if a call can invalidate quasi-immutable fields.
    fn call_can_invalidate(op: &Op) -> bool {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.effect_info().can_invalidate())
            .unwrap_or(true)
    }

    /// heap.py: check if a call forces virtual/virtualizable objects.
    fn call_forces_virtual(op: &Op) -> bool {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.effect_info().forces_virtual_or_virtualizable())
            .unwrap_or(false)
    }

    /// heap.py: force_from_effectinfo(effectinfo)
    ///
    /// Selective cache invalidation based on EffectInfo bitstrings.
    /// Instead of invalidating all caches, only force/invalidate
    /// fields and arrays that the call may read or write.
    fn force_from_effectinfo(&mut self, op: &Op, ctx: &mut OptContext) {
        let ei = match op.descr.as_ref().and_then(|d| d.as_call_descr()) {
            Some(cd) => cd.effect_info().clone(),
            None => {
                self.force_all_lazy(ctx);
                self.invalidate_caches();
                return;
            }
        };

        // If all bitstrings are zero (no specific field info), fall back
        // to conservative invalidation of all non-immutable/non-unescaped.
        let has_bitstrings = ei.readonly_descrs_fields != 0
            || ei.write_descrs_fields != 0
            || ei.readonly_descrs_arrays != 0
            || ei.write_descrs_arrays != 0;
        if !has_bitstrings {
            self.force_all_lazy(ctx);
            self.cached_fields.retain(|&(obj, descr_idx), _| {
                self.immutable_field_descrs.contains(&descr_idx) || self.unescaped.contains(&obj)
            });
            self.cached_arrayitems
                .retain(|&(obj, _, _), _| self.unescaped.contains(&obj));
            self.cached_arrayitems_var
                .retain(|&(obj, _, _), _| self.unescaped.contains(&obj));
            if !self.seen_allocation.is_empty() {
                self.known_nonnull
                    .retain(|v| self.seen_allocation.contains(v));
            } else {
                self.known_nonnull.clear();
            }
            return;
        }

        // Force/invalidate field caches based on read/write bitstrings
        let field_keys: Vec<(OpRef, u32)> = self.cached_fields.keys().copied().collect();
        for (obj, descr_idx) in field_keys {
            if ei.check_readonly_descr_field(descr_idx) {
                // Call reads this field → force lazy set (but keep cache)
                if let Some(lazy_op) = self.lazy_setfields.remove(&(obj, descr_idx)) {
                    ctx.emit(lazy_op);
                }
            }
            if ei.check_write_descr_field(descr_idx) {
                // Call writes this field → force lazy set AND invalidate cache
                if let Some(lazy_op) = self.lazy_setfields.remove(&(obj, descr_idx)) {
                    ctx.emit(lazy_op);
                }
                if !self.immutable_field_descrs.contains(&descr_idx) {
                    self.cached_fields.remove(&(obj, descr_idx));
                }
            }
        }

        // Force/invalidate array caches
        let array_keys: Vec<(OpRef, u32, i64)> = self.cached_arrayitems.keys().copied().collect();
        for (obj, descr_idx, index) in array_keys {
            if ei.check_readonly_descr_array(descr_idx) {
                if let Some(lazy_op) = self.lazy_setarrayitems.remove(&(obj, descr_idx, index)) {
                    ctx.emit(lazy_op);
                }
            }
            if ei.check_write_descr_array(descr_idx) {
                if let Some(lazy_op) = self.lazy_setarrayitems.remove(&(obj, descr_idx, index)) {
                    ctx.emit(lazy_op);
                }
                self.cached_arrayitems.remove(&(obj, descr_idx, index));
            }
        }
        self.cached_arrayitems_var
            .retain(|&(_, descr_idx, _), _| !ei.check_write_descr_array(descr_idx));

        // Remaining lazy sets for unaffected fields stay lazy.
        // Nonnull tracking: keep for allocated objects only.
        if !self.seen_allocation.is_empty() {
            self.known_nonnull
                .retain(|v| self.seen_allocation.contains(v));
        } else {
            self.known_nonnull.clear();
        }
    }

    // ── Handlers for specific opcodes ──

    fn optimize_getfield(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let key = match Self::field_key(op) {
            Some(k) => k,
            None => return OptimizationResult::Emit(op.clone()),
        };

        // Register immutable field descriptors so their cache entries survive
        // invalidation by calls and side-effecting operations.
        if let Some(descr) = &op.descr {
            if descr.is_always_pure() {
                self.immutable_field_descrs.insert(key.1);
            }
        }

        // Constant-fold: if the source is a compile-time constant address,
        // read the field value directly from memory. This is safe for
        // immutable fields (ob_type, etc.). PyPy heap.py does this for
        // quasi-immutable fields guarded by GUARD_NOT_INVALIDATED.
        // Constant-fold: source is a compile-time constant address.
        // Read the field value directly from memory at optimization time.
        // Safe for immutable fields (ob_type). PyPy heap.py handles this
        // via quasi-immutable guarded by GUARD_NOT_INVALIDATED.
        if op.num_args() >= 1 {
            if let Some(addr) = ctx.get_constant_int(op.arg(0)) {
                if addr != 0 {
                    if let Some(fd) = op.descr.as_ref().and_then(|d| d.as_field_descr()) {
                        let offset = fd.offset();
                        let field_size = fd.field_size();
                        if field_size <= 8 {
                            let ptr = (addr as usize + offset) as *const u8;
                            let mut buf = [0u8; 8];
                            unsafe {
                                std::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), field_size);
                            }
                            let folded = i64::from_ne_bytes(buf);
                            let const_ref = ctx.find_or_record_constant_int(op.pos, folded);
                            if const_ref != op.pos {
                                ctx.replace_op(op.pos, const_ref);
                            }
                            return OptimizationResult::Remove;
                        }
                    }
                }
            }
        }

        // Check lazy set first: if there is a pending SETFIELD for this key,
        // the value is the second arg of that pending op.
        if let Some(lazy_op) = self.lazy_setfields.get(&key) {
            let cached = lazy_op.arg(1);
            ctx.replace_op(op.pos, cached);
            return OptimizationResult::Remove;
        }

        if let Some(&cached) = ctx.imported_short_fields.get(&key) {
            let cached = ctx.force_op_from_preamble(cached);
            self.cached_fields.entry(key).or_insert(cached);
        }

        // Check immutable field cache first — these survive all invalidation.
        if let Some(&cached) = self.immutable_cached_fields.get(&key) {
            let cached = ctx.get_replacement(cached);
            ctx.replace_op(op.pos, cached);
            return OptimizationResult::Remove;
        }

        // Check read cache.
        if let Some(&cached) = self.cached_fields.get(&key) {
            let cached = ctx.get_replacement(cached);
            ctx.replace_op(op.pos, cached);
            return OptimizationResult::Remove;
        }

        // Check quasi-immutable cache: if this field was marked by
        // QUASIIMMUT_FIELD, the value is stable (guarded by GUARD_NOT_INVALIDATED).
        if let Some(&qi_cached) = self.quasi_immut_cache.get(&key) {
            if !qi_cached.is_none() {
                // Subsequent read: reuse the cached value.
                let qi_cached = ctx.get_replacement(qi_cached);
                ctx.replace_op(op.pos, qi_cached);
                return OptimizationResult::Remove;
            }
            // First read after QUASIIMMUT_FIELD: emit the load, then cache
            // the result so it survives calls (unlike normal mutable fields).
            self.quasi_immut_cache.insert(key, op.pos);
            self.cached_fields.insert(key, op.pos);
            return OptimizationResult::Emit(op.clone());
        }

        // Cache miss: emit the load and cache the result.
        // heap.py line 652: make_nonnull(op.getarg(0))
        let struct_ref = ctx.get_replacement(op.arg(0));
        self.known_nonnull.insert(struct_ref);
        self.cached_fields.insert(key, op.pos);
        // Save immutable fields in the permanent cache — they survive all
        // invalidation because the value never changes.
        if self.immutable_field_descrs.contains(&key.1) {
            self.immutable_cached_fields.insert(key, op.pos);
        }
        // heap.py postprocess_GETFIELD_GC_I: structinfo.setfield(descr, op)
        // Record the field value in ptr_info so other passes can see it.
        if let Some(info) = ctx.get_ptr_info_mut(struct_ref) {
            info.set_field(key.1, op.pos);
        }
        OptimizationResult::Emit(op.clone())
    }

    fn optimize_setfield(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let key = match Self::field_key(op) {
            Some(k) => k,
            None => return OptimizationResult::Emit(op.clone()),
        };

        let (obj, field_idx) = key;
        let new_value = op.arg(1);

        // The stored value escapes (it becomes reachable via the heap).
        self.unescaped.remove(&new_value);

        // Check if we already have this value cached (writing the same value again).
        if let Some(lazy_op) = self.lazy_setfields.get(&key) {
            if lazy_op.arg(1) == new_value {
                // Writing the same value as the pending lazy set -> redundant.
                return OptimizationResult::Remove;
            }
        } else if let Some(&cached) = self.cached_fields.get(&key) {
            let cached = ctx.get_replacement(cached);
            if cached == new_value {
                // Writing the same value already in the cache -> redundant.
                return OptimizationResult::Remove;
            }
        }

        // Aliasing-aware invalidation: only invalidate caches that might
        // be affected by this write.
        self.invalidate_field_caches_for_write(obj, field_idx);

        // Write-after-write: if there is already a lazy set for this key,
        // replace it (the old write is dead).
        // Either way, store as a new lazy set.
        self.lazy_setfields.insert(key, op.clone());

        // heap.py: cf.do_setfield updates info as well
        let obj = ctx.get_replacement(obj);
        if let Some(info) = ctx.get_ptr_info_mut(obj) {
            info.set_field(field_idx, new_value);
        }

        OptimizationResult::Remove
    }

    fn optimize_getarrayitem(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // Try constant-index cache first.
        if let Some(key) = Self::arrayitem_key(op, ctx) {
            if let Some(lazy_op) = self.lazy_setarrayitems.get(&key) {
                let cached = lazy_op.arg(2);
                ctx.replace_op(op.pos, cached);
                return OptimizationResult::Remove;
            }
            if let Some(&cached) = ctx.imported_short_arrayitems.get(&key) {
                let cached = ctx.force_op_from_preamble(cached);
                self.cached_arrayitems.entry(key).or_insert(cached);
            }
            if let Some(&cached) = self.cached_arrayitems.get(&key) {
                let cached = ctx.get_replacement(cached);
                ctx.replace_op(op.pos, cached);
                return OptimizationResult::Remove;
            }
            self.cached_arrayitems.insert(key, op.pos);
            // heap.py line 701: make_nonnull(op.getarg(0))
            let array_ref = ctx.get_replacement(op.arg(0));
            self.known_nonnull.insert(array_ref);
            // heap.py line 681: arrayinfo.getlenbound(None).make_gt_const(index)
            // Record that array length >= index + 1
            let (_, _, const_index) = key;
            if const_index >= 0 {
                let min_len = const_index + 1;
                let entry = self.array_min_lengths.entry(array_ref).or_insert(0);
                if min_len > *entry {
                    *entry = min_len;
                }
            }
            if let Some(info) = ctx.get_ptr_info_mut(array_ref) {
                info.set_item(key.2 as usize, op.pos);
            }
            return OptimizationResult::Emit(op.clone());
        }

        // heap.py: variable-index cache — same array + same index OpRef.
        if let Some(var_key) = Self::arrayitem_key_variable(op) {
            if let Some(&cached) = self.cached_arrayitems_var.get(&var_key) {
                let cached = ctx.get_replacement(cached);
                ctx.replace_op(op.pos, cached);
                return OptimizationResult::Remove;
            }
            self.cached_arrayitems_var.insert(var_key, op.pos);
        }

        // heap.py line 701: make_nonnull(op.getarg(0))
        self.known_nonnull.insert(ctx.get_replacement(op.arg(0)));
        OptimizationResult::Emit(op.clone())
    }

    fn optimize_setarrayitem(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // The stored value escapes (becomes reachable via the heap).
        let stored_value = op.arg(2);
        self.unescaped.remove(&stored_value);

        let key = match Self::arrayitem_key(op, ctx) {
            Some(k) => k,
            None => {
                // Non-constant index: force all lazy array stores and invalidate
                // both constant-index and variable-index caches.
                // heap.py: ArrayCachedItem.invalidate() calls parent.clear_varindex()
                self.force_all_lazy_setarrayitems(ctx);
                self.cached_arrayitems.clear();
                self.cached_arrayitems_var.clear();
                self.array_min_lengths.clear();
                // heap.py: cache_varindex_write — cache this write so that
                // a subsequent read with the same variable index can hit.
                if let Some(var_key) = Self::arrayitem_key_variable(op) {
                    self.cached_arrayitems_var.insert(var_key, op.arg(2));
                }
                return OptimizationResult::Emit(op.clone());
            }
        };

        let new_value = op.arg(2);

        // Check if writing the same value.
        if let Some(lazy_op) = self.lazy_setarrayitems.get(&key) {
            if lazy_op.arg(2) == new_value {
                return OptimizationResult::Remove;
            }
        } else if let Some(&cached) = self.cached_arrayitems.get(&key) {
            let cached = ctx.get_replacement(cached);
            if cached == new_value {
                return OptimizationResult::Remove;
            }
        }

        // Write-after-write or new lazy set.
        self.lazy_setarrayitems.insert(key, op.clone());
        self.cached_arrayitems.remove(&key);
        if let Some(info) = ctx.get_ptr_info_mut(ctx.get_replacement(op.arg(0))) {
            info.set_item(key.2 as usize, new_value);
        }
        // heap.py: ArrayCachedItem.invalidate() calls parent.clear_varindex()
        // — writing to any constant index invalidates variable-index cache
        // for the same array+descr, since the variable index could match.
        let (array, descr_idx, _) = key;
        self.cached_arrayitems_var
            .retain(|&(a, d, _), _| !(a == array && d == descr_idx));

        OptimizationResult::Remove
    }

    /// Handle CALL_LOOPINVARIANT_*: cache the result by (descr_index, args_hash).
    ///
    /// If the same call (same descriptor + same arguments) was already seen,
    /// replace with the cached result. Otherwise, emit and cache the result.
    fn optimize_call_loopinvariant(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let descr_idx = op.descr.as_ref().map(|d| d.index()).unwrap_or(0);
        let args_hash = hash_args(&op.args);
        let cache_key = (descr_idx, args_hash);

        if let Some(&cached_result) = self.loopinvariant_cache.get(&cache_key) {
            let cached_result = ctx.get_replacement(cached_result);
            ctx.replace_op(op.pos, cached_result);
            return OptimizationResult::Remove;
        }

        // First time: cache the result, then treat as a normal call for
        // heap cache purposes (force lazy sets, invalidate mutable caches).
        self.loopinvariant_cache.insert(cache_key, op.pos);
        self.mark_args_escaped(op);
        self.force_all_lazy(ctx);
        self.invalidate_caches();
        OptimizationResult::Emit(op.clone())
    }

    /// Handle operations that may have side effects.
    /// Forces lazy sets and invalidates caches as needed.
    /// Tracks allocations for aliasing analysis.
    fn handle_side_effects(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let opcode = op.opcode;

        // Track allocations for aliasing analysis.
        // Allocated objects are always non-null.
        if opcode.is_malloc() {
            self.seen_allocation.insert(op.pos);
            self.unescaped.insert(op.pos);
            self.known_nonnull.insert(op.pos);
            return OptimizationResult::Emit(op.clone());
        }

        // Note: postponed_op (from CallMayForce) must only be emitted at
        // GuardNotForced, not at arbitrary guards. RPython's emit() callback
        // calls emit_postponed_op() before every op, but the postpone→emit
        // cycle is specifically CallMayForce→GuardNotForced. Don't emit here.

        // Guards: force lazy sets but keep caches (guards don't mutate the heap).
        // Track nullity implications from guards.
        if opcode.is_guard() {
            // GuardNonnull on a value already known non-null is redundant.
            if opcode == OpCode::GuardNonnull {
                let arg = op.arg(0);
                if self.known_nonnull.contains(&arg) || self.seen_allocation.contains(&arg) {
                    return OptimizationResult::Remove;
                }
                self.known_nonnull.insert(arg);
            }

            // GuardClass / GuardNonnullClass / GuardValue imply non-null.
            match opcode {
                OpCode::GuardClass | OpCode::GuardNonnullClass | OpCode::GuardValue => {
                    self.known_nonnull.insert(op.arg(0));
                }
                _ => {}
            }

            // heap.py: guards only force non-virtual lazy sets.
            // Virtual lazy sets stay deferred — the virtual object hasn't
            // escaped, so the write can be postponed. RPython stores them
            // in optimizer.pendingfields for guard resume data; here we
            // keep them in lazy_setfields/lazy_setarrayitems until the
            // final JUMP/Finish forces everything.
            let pending_virtual = self.force_lazy_sets_for_guard(ctx);
            for pending_op in pending_virtual {
                if pending_op.opcode == OpCode::SetarrayitemGc {
                    let descr_idx = pending_op.descr.as_ref().map_or(0, |d| d.index());
                    if let Some(index) = ctx.get_constant_int(pending_op.arg(1)) {
                        self.lazy_setarrayitems
                            .insert((pending_op.arg(0), descr_idx, index), pending_op);
                    } else {
                        ctx.emit(pending_op);
                    }
                } else {
                    let descr_idx = pending_op.descr.as_ref().map_or(0, |d| d.index());
                    self.lazy_setfields
                        .insert((pending_op.arg(0), descr_idx), pending_op);
                }
            }
            return OptimizationResult::Emit(op.clone());
        }

        // Final operations (Jump, Finish): force everything.
        // RPython Phase 2 (flush=False): OptUnroll intercepts JUMP before OptHeap.
        // Skip lazy set flush to keep virtuals alive.
        if opcode.is_final() {
            if !self.skip_flush_on_final {
                self.force_all_lazy(ctx);
            }
            return OptimizationResult::Emit(op.clone());
        }

        // Calls: mark arguments as escaped, force lazy sets, and invalidate.
        if opcode.is_call() {
            let oopspec = Self::get_oopspec_index(op);
            match oopspec {
                // heap.py: DICT_LOOKUP caching — consecutive dict lookups
                // on the same dict with the same key can be deduplicated.
                OopSpecIndex::DictLookup => {
                    self.mark_args_escaped(op);
                    self.force_all_lazy(ctx);
                    // Invalidate dict-related caches but keep field/array caches.
                    // Dict operations don't affect struct fields.
                    return OptimizationResult::Emit(op.clone());
                }
                OopSpecIndex::Arraycopy | OopSpecIndex::Arraymove => {
                    // ARRAYCOPY/ARRAYMOVE: only invalidate affected array entries.
                    // Call args: [func_addr, source, dest, source_start, dest_start, length, ...]
                    // args[2] = dest array, args[4] = dest_start, args[5] = length
                    self.mark_args_escaped(op);
                    self.force_all_lazy(ctx);

                    let dest_ref = if op.args.len() > 2 {
                        op.arg(2)
                    } else {
                        OpRef::NONE
                    };
                    let dest_start = if op.args.len() > 4 {
                        ctx.get_constant_int(op.arg(4))
                    } else {
                        None
                    };
                    let length = if op.args.len() > 5 {
                        ctx.get_constant_int(op.arg(5))
                    } else {
                        None
                    };

                    if !dest_ref.is_none() {
                        self.invalidate_array_caches_for_copy(dest_ref, dest_start, length);
                    } else {
                        self.invalidate_caches();
                    }

                    return OptimizationResult::Emit(op.clone());
                }
                _ => {
                    self.mark_args_escaped(op);
                    // heap.py: force_from_effectinfo — selective cache
                    // invalidation using EffectInfo bitstrings.
                    if Self::call_has_random_effects(op) {
                        self.force_all_lazy(ctx);
                        self.invalidate_caches();
                    } else {
                        self.force_from_effectinfo(op, ctx);
                    }
                    if Self::call_can_invalidate(op) {
                        self.seen_guard_not_invalidated = false;
                    }
                    self.last_call_did_not_raise = false;
                    return OptimizationResult::Emit(op.clone());
                }
            }
        }

        // Other side-effecting ops: force and invalidate.
        if !opcode.has_no_side_effect() && !opcode.is_ovf() {
            self.force_all_lazy(ctx);
            self.invalidate_caches();
            // Any side-effecting op may raise — reset exception dedup state.
            self.last_call_did_not_raise = false;
            return OptimizationResult::Emit(op.clone());
        }

        // Pure / no-side-effect / overflow ops: pass through.
        OptimizationResult::Emit(op.clone())
    }
}

impl Default for OptHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptHeap {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        match op.opcode {
            // ── Field reads ──
            OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => {
                self.optimize_getfield(op, ctx)
            }

            // ── Raw field reads/writes ──
            // Keep these conservative. The standard heap.py cache/postprocess
            // logic applies to GC field descriptors, while raw field traffic is
            // used by compatibility seams that intentionally reload state from
            // memory instead of carrying it through loop args.
            OpCode::GetfieldRawI | OpCode::GetfieldRawR | OpCode::GetfieldRawF => {
                OptimizationResult::Emit(op.clone())
            }
            OpCode::SetfieldRaw => OptimizationResult::Emit(op.clone()),

            // ── Field writes ──
            OpCode::SetfieldGc => self.optimize_setfield(op, ctx),

            // ── Array item reads ──
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcF => {
                self.optimize_getarrayitem(op, ctx)
            }

            // ── Raw array item reads/writes ──
            // Same rationale as raw fields above: keep exact ordering and
            // dynamic indices visible until we have RPython-style virtualizable
            // handling for these buffers.
            OpCode::GetarrayitemRawI | OpCode::GetarrayitemRawR | OpCode::GetarrayitemRawF => {
                OptimizationResult::Emit(op.clone())
            }

            // ── Array item writes ──
            OpCode::SetarrayitemGc => self.optimize_setarrayitem(op, ctx),
            OpCode::SetarrayitemRaw => OptimizationResult::Emit(op.clone()),

            // ── heap.py: Interior field reads (array-of-structs pattern) ──
            OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF => {
                // Interior fields use the same field cache as regular fields.
                // The key is (array_opref, interior_field_descr_index).
                self.optimize_getfield(op, ctx)
            }

            // ── heap.py: Interior field writes ──
            OpCode::SetinteriorfieldGc => self.optimize_setfield(op, ctx),

            // ── heap.py: ARRAYLEN_GC — cache array lengths ──
            OpCode::ArraylenGc => {
                let array = ctx.get_replacement(op.arg(0));
                let descr_idx = op.descr.as_ref().map(|d| d.index()).unwrap_or(0);
                if let Some(&cached) = self.cached_arraylens.get(&(array, descr_idx)) {
                    let cached = ctx.get_replacement(cached);
                    ctx.replace_op(op.pos, cached);
                    return OptimizationResult::Remove;
                }
                self.cached_arraylens.insert((array, descr_idx), op.pos);
                // heap.py: transfer array length bound to the ARRAYLEN result.
                // intbounds can use this to eliminate length guards.
                if let Some(&min_len) = self.array_min_lengths.get(&array) {
                    let entry = ctx.int_lower_bounds.entry(op.pos).or_insert(0);
                    if min_len > *entry {
                        *entry = min_len;
                    }
                }
                OptimizationResult::Emit(op.clone())
            }

            // ── heap.py: STRLEN/UNICODELEN — cache like ARRAYLEN ──
            OpCode::Strlen | OpCode::Unicodelen => {
                let str_ref = op.arg(0);
                let key = (str_ref, op.opcode as u32 + 0xFF00);
                if let Some(&cached) = self.cached_arraylens.get(&key) {
                    let cached = ctx.get_replacement(cached);
                    ctx.replace_op(op.pos, cached);
                    return OptimizationResult::Remove;
                }
                self.cached_arraylens.insert(key, op.pos);
                OptimizationResult::PassOn // let intbounds set non-negative
            }

            // ── heap.py: Allocation tracking ──
            OpCode::New | OpCode::NewWithVtable | OpCode::NewArray | OpCode::NewArrayClear => {
                self.seen_allocation.insert(op.pos);
                self.unescaped.insert(op.pos);
                self.known_nonnull.insert(op.pos);
                OptimizationResult::PassOn
            }

            // ── heap.py: CALL_MAY_FORCE — postpone until GUARD_NOT_FORCED ──
            // These calls may force virtualizable objects, so we defer emission
            // until the guard arrives, ensuring correct exception semantics.
            OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN => {
                // RPython emitting_operation: calls go through
                // force_from_effectinfo (selective) or clean_caches,
                // NOT force_all_lazy. force_all_lazy is only in flush().
                self.mark_args_escaped(op);
                // Postpone the call — it will be emitted when GUARD_NOT_FORCED arrives.
                self.postponed_op = Some(op.clone());
                self.last_call_did_not_raise = false;
                if Self::call_has_random_effects(op) {
                    self.invalidate_caches();
                } else {
                    self.force_from_effectinfo(op, ctx);
                }
                if Self::call_can_invalidate(op) {
                    self.seen_guard_not_invalidated = false;
                }
                return OptimizationResult::Remove;
            }

            // heap.py: GUARD_NOT_FORCED — emit the postponed call_may_force,
            // then handle as a guard. RPython uses force_lazy_sets_for_guard
            // (not force_all_lazy) — immutable caches survive.
            OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
                if let Some(postponed) = self.postponed_op.take() {
                    ctx.emit(postponed);
                }
                // RPython emitting_operation for guards:
                //   self.optimizer.pendingfields = self.force_lazy_sets_for_guard()
                let pending_virtual = self.force_lazy_sets_for_guard(ctx);
                for pending_op in pending_virtual {
                    if pending_op.opcode == OpCode::SetarrayitemGc {
                        let descr_idx = pending_op.descr.as_ref().map_or(0, |d| d.index());
                        if let Some(index) = ctx.get_constant_int(pending_op.arg(1)) {
                            self.lazy_setarrayitems
                                .insert((pending_op.arg(0), descr_idx, index), pending_op);
                        } else {
                            ctx.emit(pending_op);
                        }
                    } else {
                        let descr_idx = pending_op.descr.as_ref().map_or(0, |d| d.index());
                        self.lazy_setfields
                            .insert((pending_op.arg(0), descr_idx), pending_op);
                    }
                }
                return OptimizationResult::Emit(op.clone());
            }

            // ── heap.py: COND_CALL handling ──
            OpCode::CondCallN => {
                self.force_all_lazy(ctx);
                self.invalidate_caches();
                self.last_call_did_not_raise = false;
                OptimizationResult::PassOn
            }

            // ── GUARD_NO_EXCEPTION ──
            // RPython heap.py: emit any postponed op before the guard,
            // then deduplicate consecutive GUARD_NO_EXCEPTION.
            OpCode::GuardNoException => {
                // Emit postponed op if any (RPython heap.py postponed_op)
                if let Some(postponed) = self.postponed_op.take() {
                    ctx.emit(postponed);
                }
                if self.last_call_did_not_raise {
                    return OptimizationResult::Remove;
                }
                self.last_call_did_not_raise = true;
                self.force_all_lazy(ctx);
                return OptimizationResult::Emit(op.clone());
            }

            // heap.py: GUARD_EXCEPTION — emit postponed op, then pass through.
            // Unlike GUARD_NO_EXCEPTION, this does NOT deduplicate.
            OpCode::GuardException => {
                if let Some(postponed) = self.postponed_op.take() {
                    ctx.emit(postponed);
                }
                self.last_call_did_not_raise = false;
                self.force_all_lazy(ctx);
                return OptimizationResult::Emit(op.clone());
            }

            // ── GUARD_NOT_INVALIDATED deduplication ──
            OpCode::GuardNotInvalidated => {
                if self.seen_guard_not_invalidated {
                    OptimizationResult::Remove
                } else {
                    self.seen_guard_not_invalidated = true;
                    self.force_all_lazy(ctx);
                    OptimizationResult::Emit(op.clone())
                }
            }

            // heap.py: optimize_QUASIIMMUT_FIELD
            //
            // Records the quasi-immutable dependency so that future
            // GETFIELD_GC on this (obj, field) is treated as pure.
            // Does NOT emit GUARD_NOT_INVALIDATED here — the trace
            // already contains one where needed.
            OpCode::QuasiimmutField => {
                let obj = op.arg(0);
                if let Some(descr) = &op.descr {
                    let field_idx = descr.index();
                    self.quasi_immut_cache.insert((obj, field_idx), OpRef::NONE);
                }
                OptimizationResult::Remove
            }

            // ── heap.py: RAW_LOAD — cache raw memory reads ──
            OpCode::RawLoadI | OpCode::RawLoadF => {
                // Raw loads use the field cache with a synthetic key.
                if let Some(key) = Self::field_key(op) {
                    if let Some(&cached) = self.cached_fields.get(&key) {
                        let cached = ctx.get_replacement(cached);
                        ctx.replace_op(op.pos, cached);
                        return OptimizationResult::Remove;
                    }
                    self.cached_fields.insert(key, op.pos);
                }
                OptimizationResult::Emit(op.clone())
            }

            // ── heap.py: RAW_STORE — invalidate raw memory cache ──
            OpCode::RawStore => {
                if let Some(key) = Self::field_key(op) {
                    self.cached_fields.insert(key, op.arg(1));
                }
                OptimizationResult::Emit(op.clone())
            }

            // ── GC_LOAD / GC_LOAD_INDEXED: generic memory loads ──
            // These could read from any field/array slot, so force all
            // pending lazy writes to ensure correct values.
            OpCode::GcLoadI
            | OpCode::GcLoadR
            | OpCode::GcLoadF
            | OpCode::GcLoadIndexedI
            | OpCode::GcLoadIndexedR
            | OpCode::GcLoadIndexedF => {
                self.force_all_lazy_setfields(ctx);
                self.force_all_lazy_setarrayitems(ctx);
                self.known_nonnull.insert(op.arg(0));
                OptimizationResult::Emit(op.clone())
            }

            // ── Loop-invariant calls: cache results across the trace ──
            OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN => self.optimize_call_loopinvariant(op, ctx),

            // ── Everything else: check for side effects ──
            _ => self.handle_side_effects(op, ctx),
        }
    }

    fn setup(&mut self) {
        self.cached_fields.clear();
        self.immutable_cached_fields.clear();
        self.lazy_setfields.clear();
        self.cached_arrayitems.clear();
        self.lazy_setarrayitems.clear();
        self.seen_guard_not_invalidated = false;
        self.postponed_op = None;
        self.immutable_field_descrs.clear();
        self.seen_allocation.clear();
        self.unescaped.clear();
        self.known_nonnull.clear();
        self.loopinvariant_cache.clear();
        self.last_call_did_not_raise = false;
        self.quasi_immut_cache.clear();
        self.cached_arraylens.clear();
        self.cached_arrayitems_var.clear();
        self.array_min_lengths.clear();
    }

    fn flush(&mut self) {
        // RPython parity: flush pending state without discarding the heap cache.
        // export_state() runs after flush() and still expects cached field/array
        // reads to be available for short preamble construction.
        debug_assert!(
            self.lazy_setfields.is_empty() && self.lazy_setarrayitems.is_empty(),
            "OptHeap: unflushed lazy sets at end of trace"
        );
    }

    /// heap.py: produce_potential_short_preamble_ops(sb)
    ///
    /// Add cached field/array reads to the short preamble so bridges
    /// can re-populate the optimizer's cache.
    fn produce_potential_short_preamble_ops(&self, sb: &mut crate::shortpreamble::ShortBoxes) {
        for (&(obj, descr_idx), &cached_val) in &self.cached_fields {
            if cached_val.is_none() || obj.is_none() {
                continue;
            }
            if sb.lookup_label_arg(obj).is_none() {
                continue;
            }
            let mut op = Op::new(OpCode::GetfieldGcI, &[obj]);
            op.pos = cached_val;
            sb.add_heap_op(op, descr_idx);
        }

        for (&(obj, descr_idx, index), &cached_val) in &self.cached_arrayitems {
            if cached_val.is_none() || obj.is_none() {
                continue;
            }
            if sb.lookup_label_arg(obj).is_none() {
                continue;
            }
            let idx_ref = OpRef(index as u32);
            let mut op = Op::new(OpCode::GetarrayitemGcI, &[obj, idx_ref]);
            op.pos = cached_val;
            sb.add_heap_op(op, descr_idx);
        }
    }

    fn name(&self) -> &'static str {
        "heap"
    }

    fn set_skip_flush_on_final(&mut self, enabled: bool) {
        self.skip_flush_on_final = enabled;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use majit_ir::{
        CallDescr, Descr, DescrRef, EffectInfo, ExtraEffect, OopSpecIndex, Op, OpCode, OpRef,
    };

    use crate::optimizer::Optimizer;
    use crate::{OptContext, Optimization, OptimizationResult, PtrInfo};

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

    #[test]
    fn test_imported_short_field_cache_replays_into_heap() {
        let d = descr(55);
        let mut heap = OptHeap::new();
        let mut ctx = OptContext::with_num_inputs(4, 2);
        ctx.imported_short_fields
            .insert((OpRef(0), d.index()), OpRef(1));

        let mut op = Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d);
        op.pos = OpRef(2);

        let result = heap.optimize_getfield(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(ctx.get_replacement(OpRef(2)), OpRef(1));
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
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        // SETARRAYITEM (forced at Jump) + Jump. GETARRAYITEM eliminated.
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(opcodes, vec![OpCode::SetarrayitemGc, OpCode::Jump]);
    }

    #[test]
    fn test_getarrayitem_postprocess_updates_ptr_info() {
        let d = descr(0);
        let idx = OpRef(50);
        let mut op = Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d.clone());
        op.pos = OpRef(200);

        let mut ctx = OptContext::new(256);
        ctx.make_constant(idx, majit_ir::Value::Int(3));
        ctx.set_ptr_info(OpRef(100), PtrInfo::virtual_array(d, 8));

        let mut pass = OptHeap::new();
        pass.setup();

        let result = pass.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Emit(_)));
        assert_eq!(
            ctx.get_ptr_info(OpRef(100))
                .and_then(|info| info.get_item(3)),
            Some(OpRef(200))
        );
    }

    #[test]
    fn test_setarrayitem_postprocess_updates_ptr_info() {
        let d = descr(0);
        let idx = OpRef(50);
        let op = Op::with_descr(
            OpCode::SetarrayitemGc,
            &[OpRef(100), idx, OpRef(101)],
            d.clone(),
        );

        let mut ctx = OptContext::new(256);
        ctx.make_constant(idx, majit_ir::Value::Int(3));
        ctx.set_ptr_info(OpRef(100), PtrInfo::virtual_array(d, 8));

        let mut pass = OptHeap::new();
        pass.setup();

        let result = pass.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_ptr_info(OpRef(100))
                .and_then(|info| info.get_item(3)),
            Some(OpRef(101))
        );
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
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
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
            Op::new(OpCode::New, &[]), // pos=0 -> p0
            Op::new(OpCode::New, &[]), // pos=1 -> p1
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d.clone()), // set p0.f = i10
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()), // read p0.f -> cached
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(1), OpRef(20)], d.clone()), // set p1.f = i20
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()), // read p0.f -> still cached
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
            result
                .iter()
                .filter(|o| o.opcode == OpCode::SetfieldGc)
                .count(),
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
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "second GETFIELD must be re-emitted for unknown-origin objects"
        );
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
            Op::new(OpCode::New, &[]), // pos=0 -> p0
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]), // some unrelated func
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
            Op::new(OpCode::New, &[]), // pos=0 -> p0
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(0)]), // pass p0 to call
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // NEW + SETFIELD + CALL + GETFIELD (re-emitted) + Jump.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "GETFIELD must be re-emitted after escape via call"
        );
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
            Op::new(OpCode::New, &[]), // pos=0 -> p0
            Op::new(OpCode::New, &[]), // pos=1 -> p1
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(10)], d0.clone()), // p0.f0 = i10
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(1), OpRef(0)], d1.clone()), // p1.f1 = p0 (p0 escapes)
            Op::new(OpCode::CallN, &[OpRef(1)]), // call(p1) (p1 escapes)
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d0.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // p0 escaped via setfield, so its cache is invalidated by the call.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "GETFIELD must be re-emitted after p0 escaped via setfield"
        );
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
            Op::new(OpCode::New, &[]), // pos=0 -> p0
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
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
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
            Op::new(OpCode::NewArray, &[OpRef(5)]), // pos=0 -> p0
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(0), idx, OpRef(10)],
                d.clone(),
            ),
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
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
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

    // ── Loop-invariant call caching tests ──

    // ── Test 33: Two identical CallLoopinvariantI → second removed ──

    #[test]
    fn test_loopinvariant_call_cached() {
        let d = descr(10);
        let mut ops = vec![
            Op::with_descr(
                OpCode::CallLoopinvariantI,
                &[OpRef(100), OpRef(101)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallLoopinvariantI,
                &[OpRef(100), OpRef(101)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // Only the first CallLoopinvariantI + Jump.
        let loopinv_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CallLoopinvariantI)
            .count();
        assert_eq!(
            loopinv_count, 1,
            "second identical loopinvariant call should be eliminated"
        );
        assert_eq!(result.last().unwrap().opcode, OpCode::Jump);
    }

    // ── Test 34: Different args → both kept ──

    #[test]
    fn test_loopinvariant_different_args_not_cached() {
        let d = descr(10);
        let mut ops = vec![
            Op::with_descr(
                OpCode::CallLoopinvariantI,
                &[OpRef(100), OpRef(101)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallLoopinvariantI,
                &[OpRef(100), OpRef(102)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let loopinv_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CallLoopinvariantI)
            .count();
        assert_eq!(loopinv_count, 2, "different args should keep both calls");
    }

    // ── Test 35: Cache survives intervening CallN ──

    #[test]
    fn test_loopinvariant_survives_call() {
        let d = descr(10);
        let mut ops = vec![
            Op::with_descr(
                OpCode::CallLoopinvariantI,
                &[OpRef(100), OpRef(101)],
                d.clone(),
            ),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(
                OpCode::CallLoopinvariantI,
                &[OpRef(100), OpRef(101)],
                d.clone(),
            ),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // CallLoopinvariantI + CallN + Jump. Second loopinvariant removed.
        let loopinv_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CallLoopinvariantI)
            .count();
        assert_eq!(
            loopinv_count, 1,
            "loopinvariant cache should survive intervening calls"
        );
        assert!(
            result.iter().any(|o| o.opcode == OpCode::CallN),
            "the intervening CallN should remain"
        );
    }

    // ── Test 36: Different descriptor → both kept ──

    #[test]
    fn test_loopinvariant_different_descr_not_cached() {
        let d1 = descr(10);
        let d2 = descr(20);
        let mut ops = vec![
            Op::with_descr(OpCode::CallLoopinvariantI, &[OpRef(100), OpRef(101)], d1),
            Op::with_descr(OpCode::CallLoopinvariantI, &[OpRef(100), OpRef(101)], d2),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let loopinv_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::CallLoopinvariantI)
            .count();
        assert_eq!(
            loopinv_count, 2,
            "different descriptors should keep both calls"
        );
    }

    // ── Nullity tracking tests ──

    // ── Test 37: GuardNonnull after allocation is removed ──

    #[test]
    fn test_guard_nonnull_after_allocation() {
        // p0 = new()
        // guard_nonnull(p0)   <- redundant, allocation is always non-null
        let mut ops = vec![
            Op::new(OpCode::New, &[]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 0,
            "guard_nonnull after allocation should be removed"
        );
    }

    // ── Test 38: GuardNonnull after GuardNonnull is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_nonnull() {
        // guard_nonnull(p0)
        // guard_nonnull(p0)   <- redundant
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(nonnull_count, 1, "second guard_nonnull should be removed");
    }

    // ── Test 39: GuardNonnull after GuardClass is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_class() {
        // guard_class(p0, cls)  <- implies non-null
        // guard_nonnull(p0)     <- redundant
        let mut ops = vec![
            Op::new(OpCode::GuardClass, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 0,
            "guard_nonnull after guard_class should be removed"
        );
    }

    // ── Test 40: GuardNonnull on unknown input arg is kept ──

    #[test]
    fn test_guard_nonnull_unknown_not_removed() {
        // guard_nonnull(p0)  <- first time seeing p0, must keep
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(nonnull_count, 1, "guard_nonnull on unknown should be kept");
    }

    // ── Test 41: Nonnull from allocation survives call ──

    #[test]
    fn test_known_nonnull_survives_call_for_allocation() {
        // p0 = new()
        // call_n(some_func)    <- invalidates caches, but not allocation nonnull
        // guard_nonnull(p0)    <- still redundant (allocation is always non-null)
        let mut ops = vec![
            Op::new(OpCode::New, &[]),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 0,
            "guard_nonnull after allocation should be removed even after call"
        );
    }

    // ── Test 42: Nonnull from guard does NOT survive call ──

    #[test]
    fn test_known_nonnull_from_guard_invalidated_by_call() {
        // guard_nonnull(p0)
        // call_n(some_func)   <- invalidates guard-derived nonnull
        // guard_nonnull(p0)   <- must re-emit
        let mut ops = vec![
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 2,
            "guard_nonnull after call should be re-emitted for non-allocation values"
        );
    }

    // ── Test 43: GuardNonnull after GuardNonnullClass is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_nonnull_class() {
        // guard_nonnull_class(p0, cls) <- implies non-null
        // guard_nonnull(p0)            <- redundant
        let mut ops = vec![
            Op::new(OpCode::GuardNonnullClass, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 0,
            "guard_nonnull after guard_nonnull_class should be removed"
        );
    }

    // ── Test 44: GuardNonnull after GuardValue is removed ──

    #[test]
    fn test_guard_nonnull_after_guard_value() {
        // guard_value(p0, c) <- implies non-null
        // guard_nonnull(p0)  <- redundant
        let mut ops = vec![
            Op::new(OpCode::GuardValue, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 0,
            "guard_nonnull after guard_value should be removed"
        );
    }

    // ── Test 45: GuardNonnull after NewWithVtable is removed ──

    #[test]
    fn test_guard_nonnull_after_new_with_vtable() {
        let mut ops = vec![
            Op::new(OpCode::NewWithVtable, &[]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 0,
            "guard_nonnull after new_with_vtable should be removed"
        );
    }

    // ── Test 46: GuardNonnull after NewArray is removed ──

    #[test]
    fn test_guard_nonnull_after_new_array() {
        let mut ops = vec![
            Op::new(OpCode::NewArray, &[OpRef(5)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 0,
            "guard_nonnull after new_array should be removed"
        );
    }

    // ── Call descriptor with OopSpecIndex for arraycopy tests ──

    /// Call descriptor with configurable EffectInfo for testing.
    #[derive(Debug)]
    struct TestCallDescr {
        idx: u32,
        effect: EffectInfo,
    }

    impl Descr for TestCallDescr {
        fn index(&self) -> u32 {
            self.idx
        }
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[majit_ir::Type] {
            &[]
        }
        fn result_type(&self) -> majit_ir::Type {
            majit_ir::Type::Void
        }
        fn result_size(&self) -> usize {
            0
        }
        fn effect_info(&self) -> &EffectInfo {
            &self.effect
        }
    }

    fn call_descr(idx: u32, effect: EffectInfo) -> DescrRef {
        Arc::new(TestCallDescr { idx, effect })
    }

    fn arraycopy_descr(idx: u32) -> DescrRef {
        Arc::new(TestCallDescr {
            idx,
            effect: EffectInfo {
                extra_effect: ExtraEffect::CanRaise,
                oopspec_index: OopSpecIndex::Arraycopy,
                ..Default::default()
            },
        })
    }

    #[test]
    fn test_call_may_force_uses_effectinfo_to_keep_unaffected_field_cache() {
        let d0 = descr(0);
        let call_d = call_descr(
            70,
            EffectInfo {
                extra_effect: ExtraEffect::CanRaise,
                write_descrs_fields: 1u64 << 1,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0.clone()),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "CallMayForce with unrelated write bit should preserve cached GETFIELD"
        );
    }

    #[test]
    fn test_call_may_force_uses_effectinfo_to_invalidate_written_field_cache() {
        let d0 = descr(0);
        let call_d = call_descr(
            71,
            EffectInfo {
                extra_effect: ExtraEffect::CanRaise,
                write_descrs_fields: 1u64 << 0,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0.clone()),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "CallMayForce with matching write bit must invalidate cached GETFIELD"
        );
    }

    #[test]
    fn test_call_may_force_resets_guard_not_invalidated_when_call_can_invalidate() {
        let call_d = call_descr(
            72,
            EffectInfo {
                extra_effect: ExtraEffect::CanRaise,
                can_invalidate: true,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNotInvalidated)
            .count();
        assert_eq!(
            guard_count, 2,
            "CallMayForce that can invalidate must keep the later GuardNotInvalidated"
        );
    }

    #[test]
    fn test_call_may_force_keeps_unaffected_variable_index_array_cache() {
        let d0 = descr(0);
        let idx = OpRef(50);
        let call_d = call_descr(
            73,
            EffectInfo {
                extra_effect: ExtraEffect::CanRaise,
                write_descrs_arrays: 1u64 << 1,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d0.clone()),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let mut ctx = OptContext::new(ops.len() + 64);
        assign_positions(&mut ops);
        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "CallMayForce with unrelated array write bit should preserve variable-index cache"
        );
    }

    #[test]
    fn test_call_may_force_invalidates_written_variable_index_array_cache() {
        let d0 = descr(0);
        let idx = OpRef(50);
        let call_d = call_descr(
            74,
            EffectInfo {
                extra_effect: ExtraEffect::CanRaise,
                write_descrs_arrays: 1u64 << 0,
                ..Default::default()
            },
        );
        let mut ops = vec![
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d0.clone()),
            Op::with_descr(OpCode::CallMayForceN, &[OpRef(200)], call_d),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d0),
            Op::new(OpCode::Jump, &[]),
        ];
        let mut ctx = OptContext::new(ops.len() + 64);
        assign_positions(&mut ops);
        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "CallMayForce with matching array write bit must invalidate variable-index cache"
        );
    }

    // ── ARRAYCOPY cache invalidation tests ──

    // ── Test 47: ARRAYCOPY only invalidates destination range ──

    #[test]
    fn test_arraycopy_only_invalidates_dest_range() {
        // p_src = OpRef(100), p_dst = OpRef(200)
        // setarrayitem_gc(p_dst, idx=0, val=i10)   <- cache dest[0]
        // setarrayitem_gc(p_dst, idx=5, val=i11)   <- cache dest[5]
        // call_n(arraycopy_func, p_src, p_dst, src_start=0, dst_start=2, length=3)
        //   -> copies to dest[2..5], so dest[0] survives, dest[5] survives
        // getarrayitem_gc_i(p_dst, idx=0)           <- still cached
        // getarrayitem_gc_i(p_dst, idx=5)           <- still cached
        let d = descr(0);
        let ac_d = arraycopy_descr(50);
        let idx0 = OpRef(60);
        let idx5 = OpRef(61);
        let dst_start_ref = OpRef(62);
        let length_ref = OpRef(63);
        let src_start_ref = OpRef(64);

        let mut ops = vec![
            // pos=0: setarrayitem_gc(dst, idx=0, val)
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(200), idx0, OpRef(10)],
                d.clone(),
            ),
            // pos=1: setarrayitem_gc(dst, idx=5, val)
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(200), idx5, OpRef(11)],
                d.clone(),
            ),
            // pos=2: call_n(func, src, dst, src_start, dst_start, length)
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef(300),
                    OpRef(100),
                    OpRef(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            // pos=3: getarrayitem_gc_i(dst, idx=0)
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(200), idx0], d.clone()),
            // pos=4: getarrayitem_gc_i(dst, idx=5)
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(200), idx5], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx0, majit_ir::Value::Int(0));
        ctx.make_constant(idx5, majit_ir::Value::Int(5));
        ctx.make_constant(dst_start_ref, majit_ir::Value::Int(2));
        ctx.make_constant(length_ref, majit_ir::Value::Int(3));
        ctx.make_constant(src_start_ref, majit_ir::Value::Int(0));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        // Both GETARRAYITEMs should be eliminated (dest[0] and dest[5] are outside [2..5)).
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        let get_count = opcodes
            .iter()
            .filter(|&&o| o == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 0,
            "dest[0] and dest[5] outside copy range [2..5) should be cached, got: {opcodes:?}"
        );
    }

    // ── Test 48: ARRAYCOPY with unknown length invalidates all dest entries ──

    #[test]
    fn test_arraycopy_unknown_range_invalidates_all() {
        // setarrayitem_gc(p_dst, idx=0, val=i10)
        // call_n(arraycopy_func, src, dst, src_start, dst_start, length)
        //   length is NOT a constant -> invalidates all dest array entries
        // getarrayitem_gc_i(p_dst, idx=0)   <- must re-emit
        let d = descr(0);
        let ac_d = arraycopy_descr(50);
        let idx0 = OpRef(60);
        let dst_start_ref = OpRef(62);
        let length_ref = OpRef(63); // NOT constant
        let src_start_ref = OpRef(64);

        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(200), idx0, OpRef(10)],
                d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef(300),
                    OpRef(100),
                    OpRef(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(200), idx0], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx0, majit_ir::Value::Int(0));
        ctx.make_constant(dst_start_ref, majit_ir::Value::Int(2));
        ctx.make_constant(src_start_ref, majit_ir::Value::Int(0));
        // length_ref is NOT a constant

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        // GETARRAYITEM must be re-emitted (unknown length invalidates all dest entries).
        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "unknown arraycopy length must invalidate all dest array entries"
        );
    }

    #[test]
    fn test_arraycopy_invalidates_dest_variable_index_cache() {
        let d = descr(0);
        let ac_d = arraycopy_descr(50);
        let idx = OpRef(60);
        let dst_start_ref = OpRef(62);
        let length_ref = OpRef(63);
        let src_start_ref = OpRef(64);

        let mut ops = vec![
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(200), idx], d.clone()),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef(300),
                    OpRef(100),
                    OpRef(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(200), idx], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(dst_start_ref, majit_ir::Value::Int(2));
        ctx.make_constant(length_ref, majit_ir::Value::Int(3));
        ctx.make_constant(src_start_ref, majit_ir::Value::Int(0));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "arraycopy must invalidate variable-index cache for destination array"
        );
    }

    #[test]
    fn test_arraycopy_keeps_other_array_variable_index_cache() {
        let d = descr(0);
        let ac_d = arraycopy_descr(50);
        let idx = OpRef(60);
        let dst_start_ref = OpRef(62);
        let length_ref = OpRef(63);
        let src_start_ref = OpRef(64);

        let mut ops = vec![
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(400), idx], d.clone()),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef(300),
                    OpRef(100),
                    OpRef(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(400), idx], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(dst_start_ref, majit_ir::Value::Int(2));
        ctx.make_constant(length_ref, majit_ir::Value::Int(3));
        ctx.make_constant(src_start_ref, majit_ir::Value::Int(0));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "arraycopy should preserve variable-index cache for unrelated arrays"
        );
    }

    #[test]
    fn test_arraycopy_preserves_unrelated_field_cache() {
        let field_d = descr(0);
        let array_d = descr(1);
        let ac_d = arraycopy_descr(50);
        let idx0 = OpRef(60);
        let dst_start_ref = OpRef(62);
        let length_ref = OpRef(63);
        let src_start_ref = OpRef(64);

        let mut ops = vec![
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(500)], field_d.clone()),
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(200), idx0, OpRef(10)],
                array_d.clone(),
            ),
            Op::with_descr(
                OpCode::CallN,
                &[
                    OpRef(300),
                    OpRef(100),
                    OpRef(200),
                    src_start_ref,
                    dst_start_ref,
                    length_ref,
                ],
                ac_d,
            ),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(500)], field_d),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx0, majit_ir::Value::Int(0));
        ctx.make_constant(dst_start_ref, majit_ir::Value::Int(0));
        ctx.make_constant(length_ref, majit_ir::Value::Int(1));
        ctx.make_constant(src_start_ref, majit_ir::Value::Int(0));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "arraycopy should not invalidate unrelated field caches"
        );
    }

    // ── Quasi-immutable field tests ──

    // ── Test 49: QUASIIMMUT_FIELD caches value across calls ──

    #[test]
    fn test_quasiimmut_field_caches_value() {
        // heap.py: QUASIIMMUT_FIELD records the dependency, does NOT emit
        // GUARD_NOT_INVALIDATED. The trace itself already contains the guard.
        //
        // quasiimmut_field(p0, descr=d0)
        // guard_not_invalidated()             <- from the trace
        // i1 = getfield_gc_i(p0, descr=d0)   <- first read, cached as quasi-immut
        // call_n(some_func)                   <- would normally invalidate, but quasi-immut survives
        // i2 = getfield_gc_i(p0, descr=d0)   <- reuses cached value
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::QuasiimmutField, &[OpRef(100)], d.clone()),
            Op::new(OpCode::GuardNotInvalidated, &[]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GUARD_NOT_INVALIDATED + GETFIELD (first read) + CALL + Jump.
        // Second GETFIELD eliminated (quasi-immut cache survives call).
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 1,
            "second GETFIELD after call should be eliminated for quasi-immutable field"
        );
        // GUARD_NOT_INVALIDATED from the trace should survive.
        let gni_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNotInvalidated)
            .count();
        assert_eq!(gni_count, 1, "GUARD_NOT_INVALIDATED should survive");
    }

    // ── Test 50: GC_LOAD forces lazy setfields ──

    #[test]
    fn test_gc_load_forces_lazy_setfields() {
        // setfield_gc(p0, i1, descr=d0)   <- lazy, not emitted yet
        // i2 = gc_load_i(p1, offset, size) <- generic load, forces all lazy writes
        // The SETFIELD must be emitted before the GC_LOAD.
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::new(OpCode::GcLoadI, &[OpRef(200), OpRef(8), OpRef(4)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // SETFIELD (forced by GcLoadI) + GcLoadI + Jump.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::GcLoadI);
        assert_eq!(result[2].opcode, OpCode::Jump);
    }

    // ── Test 51: GC_LOAD marks base as nonnull ──

    #[test]
    fn test_gc_load_marks_nonnull() {
        // i1 = gc_load_i(p0, offset, size)  <- dereferences p0, so p0 is nonnull
        // guard_nonnull(p0)                  <- redundant
        let mut ops = vec![
            Op::new(OpCode::GcLoadI, &[OpRef(100), OpRef(8), OpRef(4)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        let nonnull_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNonnull)
            .count();
        assert_eq!(
            nonnull_count, 0,
            "guard_nonnull after gc_load should be removed"
        );
    }

    // ── Test 52: QUASIIMMUT_FIELD on field 0 doesn't affect field 1 ──

    #[test]
    fn test_quasiimmut_field_different_field_not_cached() {
        // quasiimmut_field(p0, descr=d0)      <- marks field 0 as quasi-immut
        // i1 = getfield_gc_i(p0, descr=d1)   <- different field, NOT quasi-immut
        // call_n(some_func)
        // i2 = getfield_gc_i(p0, descr=d1)   <- must re-emit (d1 is mutable)
        let d0 = descr(0);
        let d1 = descr(1);
        let mut ops = vec![
            Op::with_descr(OpCode::QuasiimmutField, &[OpRef(100)], d0),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1.clone()),
            Op::new(OpCode::CallN, &[OpRef(200)]),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d1.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // GUARD_NOT_INVALIDATED + GETFIELD(d1) + CALL + GETFIELD(d1, re-emitted) + Jump.
        let get_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GetfieldGcI)
            .count();
        assert_eq!(
            get_count, 2,
            "quasi-immut on field 0 should not affect field 1"
        );
    }

    // ── Test 53: Bytearray-as-array heap cache verification ──
    //
    // RPython treats bytearray as regular arrays with item_size=1.
    // Verify the heap cache works correctly with byte-sized array items.

    /// Array descriptor with item_size=1 (byte array).
    #[derive(Debug)]
    struct ByteArrayDescr(u32);

    impl Descr for ByteArrayDescr {
        fn index(&self) -> u32 {
            self.0
        }
        fn as_array_descr(&self) -> Option<&dyn majit_ir::ArrayDescr> {
            Some(self)
        }
    }

    impl majit_ir::ArrayDescr for ByteArrayDescr {
        fn base_size(&self) -> usize {
            8 // typical GC header
        }
        fn item_size(&self) -> usize {
            1 // byte-sized items
        }
        fn type_id(&self) -> u32 {
            0
        }
        fn item_type(&self) -> majit_ir::Type {
            majit_ir::Type::Int
        }
    }

    fn byte_array_descr(idx: u32) -> DescrRef {
        Arc::new(ByteArrayDescr(idx))
    }

    #[test]
    fn test_bytearray_setitem_then_getitem_cached() {
        // setarrayitem_gc(p0, idx, val, descr=byte_array)
        // i2 = getarrayitem_gc_i(p0, idx, descr=byte_array)  <- eliminated
        let d = byte_array_descr(50);
        let idx = OpRef(60);
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

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx, majit_ir::Value::Int(5)); // byte index 5

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        // SETARRAYITEM_GC (forced at Jump) + Jump. GETARRAYITEM eliminated.
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(
            opcodes,
            vec![OpCode::SetarrayitemGc, OpCode::Jump],
            "byte-array getitem should be cached after setitem"
        );
    }

    #[test]
    fn test_bytearray_different_indices_not_cached() {
        // setarrayitem_gc(p0, idx=5, val, descr=byte_array)
        // i2 = getarrayitem_gc_i(p0, idx=6, descr=byte_array)  <- NOT cached (different index)
        let d = byte_array_descr(50);
        let idx5 = OpRef(60);
        let idx6 = OpRef(61);
        let mut ops = vec![
            Op::with_descr(
                OpCode::SetarrayitemGc,
                &[OpRef(100), idx5, OpRef(101)],
                d.clone(),
            ),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx6], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops);

        let mut ctx = OptContext::new(ops.len());
        ctx.make_constant(idx5, majit_ir::Value::Int(5));
        ctx.make_constant(idx6, majit_ir::Value::Int(6));

        let mut pass = OptHeap::new();
        pass.setup();

        for op in &ops {
            let mut resolved = op.clone();
            for arg in &mut resolved.args {
                *arg = ctx.get_replacement(*arg);
            }
            match pass.propagate_forward(&resolved, &mut ctx) {
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        // GETARRAYITEM must be emitted (not cached — different index).
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert!(
            opcodes.contains(&OpCode::GetarrayitemGcI),
            "different byte-array index should not use cache: {:?}",
            opcodes
        );
    }

    #[test]
    fn test_bytearray_read_after_read_cached() {
        // i1 = getarrayitem_gc_i(p0, idx=3, descr=byte_array)
        // i2 = getarrayitem_gc_i(p0, idx=3, descr=byte_array)  <- eliminated (same read)
        let d = byte_array_descr(50);
        let idx = OpRef(60);
        let mut ops = vec![
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d.clone()),
            Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d.clone()),
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
                OptimizationResult::Emit(emitted) => {
                    ctx.emit(emitted);
                }
                OptimizationResult::Remove => {}
                OptimizationResult::Replace(replaced) => {
                    ctx.emit(replaced);
                }
                OptimizationResult::PassOn => {
                    ctx.emit(resolved);
                }
            }
        }

        // Only one GETARRAYITEM + Jump: the second read is eliminated.
        let get_count = ctx
            .new_operations
            .iter()
            .filter(|o| o.opcode == OpCode::GetarrayitemGcI)
            .count();
        assert_eq!(get_count, 1, "byte-array read-after-read should be cached");
    }

    #[test]
    fn test_arraylen_caching() {
        // Two ARRAYLEN_GC on the same array → second eliminated.
        let d = descr(42);
        let mut ops = vec![
            {
                let mut op = Op::new(OpCode::ArraylenGc, &[OpRef(100)]);
                op.descr = Some(d.clone());
                op
            },
            {
                let mut op = Op::new(OpCode::ArraylenGc, &[OpRef(100)]);
                op.descr = Some(d);
                op
            },
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptHeap::new()));
        let result = opt.optimize(&ops);
        let len_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::ArraylenGc)
            .count();
        assert_eq!(len_count, 1, "duplicate ARRAYLEN_GC should be cached");
    }
}
