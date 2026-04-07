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

#[inline(always)]
fn vb_set(v: &mut Vec<bool>, i: u32) {
    let i = i as usize;
    if i >= v.len() {
        v.resize(i + 1, false);
    }
    v[i] = true;
}
#[inline(always)]
fn vb_get(v: &[bool], i: u32) -> bool {
    v.get(i as usize).copied().unwrap_or(false)
}
#[inline(always)]
fn vb_unset(v: &mut Vec<bool>, i: u32) {
    if let Some(s) = v.get_mut(i as usize) {
        *s = false;
    }
}

use majit_ir::{DescrRef, OopSpecIndex, Op, OpCode, OpRef, Type};

use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// Cache key for a field access: (struct OpRef, field descriptor index).
type FieldKey = (OpRef, u32);

/// heap.py:20-196: AbstractCachedEntry / CachedField
///
/// Per-descr (field descriptor index) cache entry. Tracks cached struct→value
/// pairs and at most one pending lazy SetfieldGc.
///
/// RPython uses parallel lists (cached_structs, cached_infos). We use Vec
/// of tuples for cache-friendly linear scan — the entry count per descr is
/// typically 1–5, where Vec beats HashMap.
/// heap.py:20-50 AbstractCachedEntry.
///
/// RPython stores `cached_infos: [PtrInfo]` and reads values from
/// `info._fields[descr.get_index()]` at access time. majit stores an
/// explicit `cached_values: Vec<OpRef>` parallel to `cached_structs`
/// instead, because Rust's ownership model prevents storing mutable
/// PtrInfo references. The two representations are always in sync —
/// every `register()` + `set_field()` pair updates both, and
/// `invalidate_with_ctx()` clears both. `produce_potential_short_preamble_ops`
/// reads from PtrInfo (info.py:262 parity) to pick up late updates.
struct CachedField {
    /// heap.py:23: cached_structs — struct OpRefs with cached field values.
    cached_structs: Vec<OpRef>,
    /// Parallel to cached_structs. RPython reads this from
    /// `cached_infos[i]._fields[descr]`; we cache the OpRef directly
    /// for O(1) access in the hot path (optimize_getfield/setfield).
    cached_values: Vec<OpRef>,
    /// struct_opref → DescrRef (only stored when available).
    cached_descrs: Vec<(OpRef, DescrRef)>,
    /// heap.py:40: _lazy_set — at most one pending SetfieldGc per descr.
    lazy_set: Option<(OpRef, Op)>,
}

impl CachedField {
    fn new() -> Self {
        CachedField {
            cached_structs: Vec::new(),
            cached_values: Vec::new(),
            cached_descrs: Vec::new(),
            lazy_set: None,
        }
    }

    /// heap.py:42-49: register_info
    fn register_info(&mut self, struct_opref: OpRef, value: OpRef, descr: Option<&DescrRef>) {
        // Update existing entry or append new one.
        for i in 0..self.cached_structs.len() {
            if self.cached_structs[i] == struct_opref {
                self.cached_values[i] = value;
                if let Some(d) = descr {
                    // Update descr if exists, else add.
                    if let Some(entry) = self
                        .cached_descrs
                        .iter_mut()
                        .find(|(k, _)| *k == struct_opref)
                    {
                        entry.1 = d.clone();
                    } else {
                        self.cached_descrs.push((struct_opref, d.clone()));
                    }
                }
                return;
            }
        }
        self.cached_structs.push(struct_opref);
        self.cached_values.push(value);
        if let Some(d) = descr {
            self.cached_descrs.push((struct_opref, d.clone()));
        }
    }

    /// heap.py:59-65: possible_aliasing.
    /// Both `lazy_obj` and `struct_opref` must be pre-resolved through
    /// `ctx.get_replacement` (PtrInfo identity semantics).
    fn possible_aliasing(&self, struct_opref: OpRef) -> bool {
        if let Some((lazy_obj, _)) = &self.lazy_set {
            *lazy_obj != struct_opref
        } else {
            false
        }
    }

    /// heap.py:103-120: getfield_from_cache
    fn getfield_from_cache(&self, struct_opref: OpRef) -> Option<OpRef> {
        if let Some((lazy_obj, lazy_op)) = &self.lazy_set {
            if *lazy_obj == struct_opref {
                return Some(lazy_op.arg(1));
            }
        }
        for i in 0..self.cached_structs.len() {
            if self.cached_structs[i] == struct_opref {
                return Some(self.cached_values[i]);
            }
        }
        None
    }

    /// Look up cached value from entries only (skip lazy_set).
    fn _getfield(&self, struct_opref: OpRef) -> Option<OpRef> {
        for i in 0..self.cached_structs.len() {
            if self.cached_structs[i] == struct_opref {
                return Some(self.cached_values[i]);
            }
        }
        None
    }

    /// Get the DescrRef for a struct, if stored.
    fn get_descr(&self, struct_opref: OpRef) -> Option<&DescrRef> {
        self.cached_descrs
            .iter()
            .find(|(k, _)| *k == struct_opref)
            .map(|(_, d)| d)
    }

    /// heap.py:162-196: invalidate — clear all cached entries for this descr.
    fn invalidate(&mut self) {
        self.cached_structs.clear();
        self.cached_values.clear();
        self.cached_descrs.clear();
    }

    /// heapcache.py:119-130: invalidate_unescaped — remove entries for
    /// ESCAPED objects, keep entries for unescaped allocations.
    fn invalidate_unescaped(&mut self, unescaped: &[bool]) {
        self.retain_entries(|obj| vb_get(unescaped, obj.0));
    }

    /// heap.py:189-196: invalidate with PtrInfo cleanup.
    fn invalidate_with_ctx(&mut self, field_idx: u32, ctx: &mut crate::optimizeopt::OptContext) {
        for &obj in &self.cached_structs {
            let obj_resolved = ctx.get_box_replacement(obj);
            if let Some(info) = ctx.get_ptr_info_mut(obj_resolved) {
                info.clear_field(field_idx);
            }
        }
        self.cached_structs.clear();
        self.cached_values.clear();
        self.cached_descrs.clear();
    }

    /// Check if this CachedField has any entries or a lazy_set.
    fn is_empty(&self) -> bool {
        self.cached_structs.is_empty() && self.lazy_set.is_none()
    }

    /// Retain only entries matching a predicate on the struct opref.
    fn retain_entries(&mut self, f: impl Fn(&OpRef) -> bool) {
        let mut i = 0;
        while i < self.cached_structs.len() {
            if f(&self.cached_structs[i]) {
                i += 1;
            } else {
                let removed = self.cached_structs.swap_remove(i);
                self.cached_values.swap_remove(i);
                self.cached_descrs.retain(|(k, _)| *k != removed);
            }
        }
    }
}

/// Cache key for an array item access: (array OpRef, descriptor index, constant array index).
type ArrayItemKey = (OpRef, u32, i64);

/// heap.py:228-298: ArrayCachedItem(AbstractCachedEntry)
///
/// Per-(descr, index) cache entry for array items. Tracks cached array->value
/// pairs and at most one pending lazy SetarrayitemGc.
///
/// Mirrors CachedField but keyed by (descr_idx, const_index) instead of field_idx.
struct ArrayCachedItem {
    /// RPython: cached_structs + cached_infos parallel lists.
    cached_arrays: Vec<OpRef>,
    cached_values: Vec<OpRef>,
    cached_descrs: Vec<(OpRef, DescrRef)>,
    /// heap.py:40: _lazy_set -- at most one pending SetarrayitemGc.
    lazy_set: Option<(OpRef, Op)>,
}

impl ArrayCachedItem {
    fn new() -> Self {
        ArrayCachedItem {
            cached_arrays: Vec::new(),
            cached_values: Vec::new(),
            cached_descrs: Vec::new(),
            lazy_set: None,
        }
    }

    /// heap.py:42-49: register_info (inherited from AbstractCachedEntry)
    fn register_info(&mut self, array_opref: OpRef, value: OpRef, descr: Option<&DescrRef>) {
        for i in 0..self.cached_arrays.len() {
            if self.cached_arrays[i] == array_opref {
                self.cached_values[i] = value;
                if let Some(d) = descr {
                    if let Some(entry) = self
                        .cached_descrs
                        .iter_mut()
                        .find(|(k, _)| *k == array_opref)
                    {
                        entry.1 = d.clone();
                    } else {
                        self.cached_descrs.push((array_opref, d.clone()));
                    }
                }
                return;
            }
        }
        self.cached_arrays.push(array_opref);
        self.cached_values.push(value);
        if let Some(d) = descr {
            self.cached_descrs.push((array_opref, d.clone()));
        }
    }

    fn possible_aliasing(&self, array_opref: OpRef) -> bool {
        if let Some((lazy_obj, _)) = &self.lazy_set {
            *lazy_obj != array_opref
        } else {
            false
        }
    }

    /// heap.py:103-120: getfield_from_cache (inherited from AbstractCachedEntry)
    fn getfield_from_cache(&self, array_opref: OpRef) -> Option<OpRef> {
        if let Some((lazy_obj, lazy_op)) = &self.lazy_set {
            if *lazy_obj == array_opref {
                return Some(lazy_op.arg(2));
            }
        }
        for i in 0..self.cached_arrays.len() {
            if self.cached_arrays[i] == array_opref {
                return Some(self.cached_values[i]);
            }
        }
        None
    }

    fn _getfield(&self, array_opref: OpRef) -> Option<OpRef> {
        for i in 0..self.cached_arrays.len() {
            if self.cached_arrays[i] == array_opref {
                return Some(self.cached_values[i]);
            }
        }
        None
    }

    fn get_descr(&self, array_opref: OpRef) -> Option<&DescrRef> {
        self.cached_descrs
            .iter()
            .find(|(k, _)| *k == array_opref)
            .map(|(_, d)| d)
    }

    fn invalidate(&mut self) {
        self.cached_arrays.clear();
        self.cached_values.clear();
        self.cached_descrs.clear();
    }

    /// heapcache.py:119-130: invalidate_unescaped — remove entries for
    /// ESCAPED objects, keep entries for unescaped allocations.
    fn invalidate_unescaped(&mut self, unescaped: &[bool]) {
        self.retain_entries(|obj| vb_get(unescaped, obj.0));
    }

    fn retain_entries(&mut self, f: impl Fn(&OpRef) -> bool) {
        let mut i = 0;
        while i < self.cached_arrays.len() {
            if f(&self.cached_arrays[i]) {
                i += 1;
            } else {
                let removed = self.cached_arrays.swap_remove(i);
                self.cached_values.swap_remove(i);
                self.cached_descrs.retain(|(k, _)| *k != removed);
            }
        }
    }

    /// Check if this ArrayCachedItem has any entries or a lazy_set.
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.cached_arrays.is_empty() && self.lazy_set.is_none()
    }
}

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
    /// Per-descr field cache: field_idx → CachedField.
    /// RPython heap.py: cached_fields dict keyed by descr.
    cached_fields: HashMap<u32, CachedField>,
    /// Immutable (pure) field cache — separate to survive all invalidation.
    /// RPython heap.py: is_always_pure() fields are never invalidated.
    immutable_cached_fields: HashMap<FieldKey, OpRef>,
    /// heap.py:332: cached_arrayitems -- per-(descr, index) array cache.
    /// Key: (descr_idx, const_index) -> ArrayCachedItem.
    cached_arrayitems: HashMap<(u32, i64), ArrayCachedItem>,
    /// Whether we've already emitted a GUARD_NOT_INVALIDATED.
    seen_guard_not_invalidated: bool,
    /// Postponed operation: held back until the next GUARD_NO_EXCEPTION.
    /// RPython heap.py: `postponed_op` — delays emission of operations
    /// that may raise (CALL_MAY_FORCE, comparison ops) until we see
    /// a GUARD_NO_EXCEPTION, ensuring correct exception semantics.
    postponed_op: Option<Op>,
    /// Descriptor indices known to be immutable. RPython: descr.is_always_pure().
    immutable_field_descrs: Vec<bool>,
    /// Array descriptor indices known to be immutable. RPython: descr.is_always_pure().
    immutable_array_descrs: Vec<bool>,

    // ── Aliasing analysis state — RPython: PtrInfo flags ──
    seen_allocation: Vec<bool>,
    unescaped: Vec<bool>,
    known_nonnull: Vec<bool>,
    /// heapcache.py: _heapc_deps — per-box dependency list.
    /// When an unescaped value is stored into an unescaped container,
    /// the value is recorded as a dependency of the container instead
    /// of being immediately escaped. When the container escapes later,
    /// all its dependencies are transitively escaped.
    heapc_deps: HashMap<u32, Vec<OpRef>>,

    /// heap.py:27 OptHeap inherits Optimization.last_emitted_operation,
    /// which is set to REMOVED by `_optimize_CALL_DICT_LOOKUP`
    /// (heap.py:527) when a folded CALL_PURE collapses into its cached
    /// result. `optimize_GUARD_NO_EXCEPTION` (heap.py:530-533) reads the
    /// flag and skips emitting the trailing guard. majit models the
    /// REMOVED state as a per-pass boolean — set true on the explicit
    /// removal paths, false on every other dispatch entry.
    last_emitted_was_removed: bool,
    /// Snapshot of `last_emitted_was_removed` taken at the start of every
    /// `propagate_forward` call so that handlers (e.g.,
    /// `optimize_GUARD_NO_EXCEPTION`) read the *previous* op's outcome
    /// rather than their own reset state.
    prior_emitted_was_removed: bool,

    /// Fields known to be quasi-immutable: (obj, field_idx) -> cached value OpRef.
    /// Populated by QUASIIMMUT_FIELD, consumed by subsequent GETFIELD_GC_*.
    /// Survives calls (guarded by GUARD_NOT_INVALIDATED).
    quasi_immut_cache: HashMap<FieldKey, OpRef>,
    /// field_idx → DescrRef for PtrInfo-based export.
    field_descr_map: HashMap<u32, DescrRef>,
    /// heap.py: cached array lengths.
    cached_arraylens: HashMap<(OpRef, u32), OpRef>,
    /// heap.py: variable-index array cache.
    /// Key: (array, descr_index, index_opref) → value OpRef.
    cached_arrayitems_var: HashMap<(OpRef, u32, OpRef), OpRef>,
    /// heap.py: arrayinfo.getlenbound() — minimum known array lengths.
    /// When GETARRAYITEM_GC with constant index N is seen, the array
    /// must have length >= N+1. Tracked per array OpRef.
    array_min_lengths: HashMap<OpRef, i64>,
}

impl OptHeap {
    pub fn new() -> Self {
        OptHeap {
            cached_fields: HashMap::new(),
            immutable_cached_fields: HashMap::new(),
            cached_arrayitems: HashMap::new(),
            seen_guard_not_invalidated: false,
            postponed_op: None,
            immutable_field_descrs: Vec::new(),
            immutable_array_descrs: Vec::new(),
            seen_allocation: Vec::new(),
            unescaped: Vec::new(),
            known_nonnull: Vec::new(),
            heapc_deps: HashMap::new(),
            last_emitted_was_removed: false,
            prior_emitted_was_removed: false,
            quasi_immut_cache: HashMap::new(),
            field_descr_map: HashMap::new(),
            cached_arraylens: HashMap::new(),
            cached_arrayitems_var: HashMap::new(),
            array_min_lengths: HashMap::new(),
        }
    }

    /// heapcache.py:295-309 _escape_box: escape a box and transitively
    /// escape all its dependencies.
    fn escape_box(&mut self, opref: OpRef) {
        vb_unset(&mut self.unescaped, opref.0);
        if let Some(deps) = self.heapc_deps.remove(&opref.0) {
            for dep in deps {
                self.escape_box(dep);
            }
        }
    }

    /// heapcache.py:224-230 _escape_from_write: when storing a value into
    /// a container, record dependency if both are unescaped; otherwise
    /// escape the value immediately.
    fn escape_from_write(&mut self, container: OpRef, value: OpRef) {
        if vb_get(&self.unescaped, container.0) && vb_get(&self.unescaped, value.0) {
            self.heapc_deps.entry(container.0).or_default().push(value);
        } else if !value.is_none() {
            self.escape_box(value);
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

    /// heap.py:409-415 arrayitem_cache: constant-index array cache key.
    /// Canonicalizes array and index through get_box_replacement.
    fn arrayitem_key(op: &Op, ctx: &OptContext) -> Option<ArrayItemKey> {
        let descr = op.descr.as_ref()?;
        let array = ctx.get_box_replacement(op.arg(0));
        let index_ref = ctx.get_box_replacement(op.arg(1));
        let index_val = ctx.get_constant_int(index_ref)?;
        Some((array, descr.index(), index_val))
    }

    /// heap.py:319-324 lookup_cached: variable-index array cache key.
    /// Canonicalizes array and index through get_box_replacement so
    /// forwarded refs hit the same cache entry.
    fn arrayitem_key_variable(op: &Op, ctx: &OptContext) -> Option<(OpRef, u32, OpRef)> {
        let descr = op.descr.as_ref()?;
        let array = ctx.get_box_replacement(op.arg(0));
        let index_ref = ctx.get_box_replacement(op.arg(1));
        Some((array, descr.index(), index_ref))
    }

    /// Register a cached value in the per-descr CachedField.
    /// RPython: cf.register_info(structop, info)
    /// Keep nonnull knowledge only for boxes known to be allocated in this trace.
    fn cache_field(&mut self, obj: OpRef, field_idx: u32, value: OpRef, descr: Option<&DescrRef>) {
        let cf = self
            .cached_fields
            .entry(field_idx)
            .or_insert_with(CachedField::new);
        cf.register_info(obj, value, descr);
        if let Some(d) = descr {
            self.field_descr_map
                .entry(field_idx)
                .or_insert_with(|| d.clone());
        }
    }

    /// heap.py:392: field_cache (read-only borrow variant).
    fn get_cached_field(&self, field_idx: u32) -> Option<&CachedField> {
        self.cached_fields.get(&field_idx)
    }

    /// heap.py:392: field_cache — get or create CachedField for a descr.
    fn field_cache(&mut self, field_idx: u32) -> &mut CachedField {
        self.cached_fields
            .entry(field_idx)
            .or_insert_with(CachedField::new)
    }

    /// heap.py:409: arrayitem_cache(descr, index)
    fn arrayitem_cache(&mut self, descr_idx: u32, index: i64) -> &mut ArrayCachedItem {
        self.cached_arrayitems
            .entry((descr_idx, index))
            .or_insert_with(ArrayCachedItem::new)
    }

    /// Register an array item value in the per-(descr, index) ArrayCachedItem.
    fn cache_arrayitem(
        &mut self,
        array: OpRef,
        descr_idx: u32,
        index: i64,
        value: OpRef,
        descr: Option<&DescrRef>,
    ) {
        let cai = self.arrayitem_cache(descr_idx, index);
        cai.register_info(array, value, descr);
    }

    /// heap.py: force_lazy_set — emit lazy setfields.
    /// If any lazy setfield argument references the postponed_op,
    /// emit the postponed_op first (RPython heap.py exact logic).
    /// Emit a lazy setfield after resolving forwarding and forcing a virtual
    /// rhs if needed.
    ///
    /// RPython heap.py: force_lazy_set → emit_extra(op, emit=False).
    ///
    /// RPython parity: if the RHS value is virtual, do NOT emit the SetfieldGc.
    /// Instead, add it to pendingfields for the guard's resume data
    /// (heap.py:618-620). The virtual will be materialized at guard failure
    /// time via rd_virtuals, and the pending setfield replayed after.
    ///
    /// `allow_force`: true for call-triggered force, false for JUMP/flush.
    /// heap.py: cf.force_lazy_set(optheap, descr)
    ///
    /// Emit a lazy SetfieldGc. If the value is virtual, return false (the
    /// caller should handle it via pendingfields / rd_pendingfields).
    fn emit_lazy_setfield(op: &mut Op, ctx: &mut OptContext, _allow_force: bool) -> bool {
        let orig_val = ctx.get_box_replacement(op.arg(1));

        // heap.py:136: emit_extra(op, emit=False) re-processes through passes.
        // Virtual values are skipped — handled by rd_pendingfields at guard
        // time or dropped at JUMP.
        if let Some(info) = ctx.get_ptr_info(orig_val) {
            if info.is_virtual() {
                return false;
            }
        }

        // Non-virtual path: resolve forwarding and route after heap
        for arg in op.args.iter_mut() {
            *arg = ctx.get_box_replacement(*arg);
        }
        // heap.py:136: emit_extra(op, emit=False) → next_optimization
        ctx.emit_extra(ctx.current_pass_idx, op.clone());
        true
    }

    /// heap.py:122-145: force_lazy_set → emit_extra(op, emit=False)
    ///
    /// For each CachedField with a pending lazy set:
    /// 1. invalidate(descr) — clear conflicting cache entries
    /// 2. emit_extra(op, emit=False) — route through passes AFTER heap
    /// 3. put_field_back_to_info — restore this specific cache entry
    ///
    /// `heap_pass_idx`: this pass's own index. RPython uses
    /// `self.next_optimization` which always starts AFTER heap.
    fn force_all_lazy_setfields(&mut self, heap_pass_idx: usize, ctx: &mut OptContext) {
        if let Some(ref postponed) = self.postponed_op {
            let postponed_pos = postponed.pos;
            let needs_postponed = self.cached_fields.values().any(|cf| {
                cf.lazy_set
                    .as_ref()
                    .map_or(false, |(_, op)| op.args.iter().any(|a| *a == postponed_pos))
            });
            if needs_postponed {
                // RPython emit_postponed_op: route through next_optimization
                if let Some(p) = self.postponed_op.take() {
                    ctx.emit_extra(heap_pass_idx, p);
                }
            }
        }
        // Collect all lazy sets from all CachedFields.
        let pending: Vec<(u32, OpRef, Op)> = self
            .cached_fields
            .iter_mut()
            .filter_map(|(&field_idx, cf)| cf.lazy_set.take().map(|(obj, op)| (field_idx, obj, op)))
            .collect();
        for (field_idx, obj, mut op) in pending {
            // heap.py:129: invalidate(descr) — skip if is_always_pure
            if !vb_get(&self.immutable_field_descrs, field_idx) {
                self.field_cache(field_idx)
                    .invalidate_with_ctx(field_idx, ctx);
            }
            // Resolve args after invalidation.
            for arg in op.args.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
            let final_value = op.arg(1);
            let descr = op.descr.clone();
            // heap.py:136: emit_extra(op, emit=False) — route after heap
            ctx.emit_extra(heap_pass_idx, op);
            // heap.py:142-143: put_field_back_to_info — restore cache
            self.cache_field(obj, field_idx, final_value, descr.as_ref());
            let obj_resolved = ctx.get_box_replacement(obj);
            if let Some(info) = ctx.get_ptr_info_mut(obj_resolved) {
                info.setfield(field_idx, final_value);
            }
        }
    }

    fn force_all_lazy_setarrayitems(&mut self, heap_pass_idx: usize, ctx: &mut OptContext) {
        // Collect all lazy sets from all ArrayCachedItems.
        let pending: Vec<(u32, i64, OpRef, Op)> = self
            .cached_arrayitems
            .iter_mut()
            .filter_map(|(&(descr_idx, index), cai)| {
                cai.lazy_set
                    .take()
                    .map(|(obj, op)| (descr_idx, index, obj, op))
            })
            .collect();
        for (descr_idx, index, obj, mut op) in pending {
            for arg in op.args.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
            let final_value = op.arg(2);
            let array_ref = op.arg(0);
            let descr = op.descr.clone();
            // emit_extra(op, emit=False): route through passes after heap
            ctx.emit_extra(heap_pass_idx, op);
            self.cache_arrayitem(array_ref, descr_idx, index, final_value, descr.as_ref());
        }
    }

    /// Force all pending lazy stores (both fields and array items).
    /// `heap_pass_idx`: this pass's own pipeline index for emit routing.
    fn force_all_lazy_sets(&mut self, heap_pass_idx: usize, ctx: &mut OptContext) {
        self.force_all_lazy_setfields(heap_pass_idx, ctx);
        self.force_all_lazy_setarrayitems(heap_pass_idx, ctx);
    }

    /// heap.py:608-637 force_lazy_sets_for_guard()
    ///
    /// Returns pendingfields: SetfieldGc/SetarrayitemGc ops where the stored
    /// VALUE is virtual. These go into rd_pendingfields on the guard's resume
    /// data (emitting_operation stores them in ctx.pending_for_guard →
    /// optimizer.rs encodes as op.rd_pendingfields).
    /// Non-virtual lazy sets are emitted (forced) immediately.
    fn force_lazy_sets_for_guard(&mut self, self_pass_idx: usize, ctx: &mut OptContext) -> Vec<Op> {
        let mut pendingfields = Vec::new();

        // heap.py:610-621: iterate cached fields
        // Collect all lazy sets from CachedFields.
        let field_entries: Vec<(u32, OpRef, Op)> = self
            .cached_fields
            .iter_mut()
            .filter_map(|(&field_idx, cf)| cf.lazy_set.take().map(|(obj, op)| (field_idx, obj, op)))
            .collect();
        for (field_idx, obj, mut op) in field_entries {
            // heap.py:617-618: val = op.getarg(1); if is_virtual(val)
            let value_ref = ctx.get_box_replacement(op.arg(1));
            let is_virtual = matches!(
                ctx.get_ptr_info(value_ref),
                Some(info) if info.is_virtual()
            );
            if is_virtual {
                // heap.py:618-619: virtual value → pendingfields
                pendingfields.push(op);
                continue;
            }
            // heap.py:621: cf.force_lazy_set(self, descr) →
            // invalidate first, then emit_extra(op, emit=False),
            // then put_field_back_to_info restores the cache.
            for arg in op.args.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
            let final_value = op.arg(1);
            let descr = op.descr.clone();
            // heap.py:129,189-191: invalidate(descr) — skip if is_always_pure
            if !vb_get(&self.immutable_field_descrs, field_idx) {
                self.field_cache(field_idx)
                    .invalidate_with_ctx(field_idx, ctx);
            }
            // emit_extra(op, emit=False): route through passes after heap.
            // RPython: self.next_optimization — always starts AFTER heap,
            // regardless of which pass emitted the guard that triggered this.
            ctx.emit_extra(self_pass_idx, op);
            // heap.py:142-143: put_field_back_to_info — restore cache + PtrInfo
            self.cache_field(obj, field_idx, final_value, descr.as_ref());
            let obj_resolved = ctx.get_box_replacement(obj);
            if let Some(info) = ctx.get_ptr_info_mut(obj_resolved) {
                info.setfield(field_idx, final_value);
            }
        }

        // heap.py:622-636: iterate cached array items
        let array_entries: Vec<(u32, i64, OpRef, Op)> = self
            .cached_arrayitems
            .iter_mut()
            .filter_map(|(&(descr_idx, index), cai)| {
                cai.lazy_set
                    .take()
                    .map(|(obj, op)| (descr_idx, index, obj, op))
            })
            .collect();
        for (descr_idx, index, _obj, mut op) in array_entries {
            // heap.py:631-633: assert container not virtual; check value virtual
            let value_ref = ctx.get_box_replacement(op.arg(2));
            let is_virtual = matches!(
                ctx.get_ptr_info(value_ref),
                Some(info) if info.is_virtual()
            );
            if is_virtual {
                // heap.py:634: pendingfields.append(op)
                pendingfields.push(op);
                continue;
            }

            for arg in op.args.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
            let final_value = op.arg(2);
            let array_ref = op.arg(0);
            let descr = op.descr.clone();
            // emit_extra(op, emit=False): route through passes after heap.
            ctx.emit_extra(self_pass_idx, op);
            self.cache_arrayitem(array_ref, descr_idx, index, final_value, descr.as_ref());
        }

        pendingfields
    }

    /// Invalidate caches on calls and other side-effecting operations.
    ///
    /// Caches that survive:
    /// - Immutable (green) field caches: values never change.
    /// - Unescaped object caches: calls cannot access objects that haven't
    ///   been passed to a call or stored into the heap.
    /// heap.py:379-391: invalidate non-pure field/array caches.
    /// Only `is_always_pure` (immutable) fields survive.
    fn clean_caches(&mut self) {
        for (&field_idx, cf) in self.cached_fields.iter_mut() {
            // heap.py:384: if not descr.is_always_pure(): cf.invalidate()
            if !vb_get(&self.immutable_field_descrs, field_idx) {
                cf.invalidate();
            }
        }
        // heap.py:386-389: invalidate non-pure array caches
        for (&(descr_idx, _index), cai) in self.cached_arrayitems.iter_mut() {
            if !vb_get(&self.immutable_array_descrs, descr_idx) {
                cai.invalidate();
            }
        }
        // heap.py:386-389: variable-index cache also respects is_always_pure.
        // RPython: descr.is_always_pure() skips the entire submap, which
        // includes both const_indexes and variable-index entries.
        self.cached_arrayitems_var
            .retain(|&(_, descr_idx, _), _| vb_get(&self.immutable_array_descrs, descr_idx));
        // heap.py:390: self.cached_dict_reads.clear()
        // (no dict_reads cache in majit)
    }

    /// heapcache.py:363-369 invalidate_unescaped / clear_caches_varargs
    ///
    /// Preserve cached values for unescaped allocations across residual calls.
    /// Only entries attached to boxes that have escaped are invalidated.
    fn invalidate_caches_for_escaped(&mut self) {
        for (&field_idx, cf) in self.cached_fields.iter_mut() {
            if vb_get(&self.immutable_field_descrs, field_idx) {
                continue;
            }
            cf.retain_entries(|obj| vb_get(&self.unescaped, obj.0));
        }
        for (&(descr_idx, _index), cai) in self.cached_arrayitems.iter_mut() {
            if vb_get(&self.immutable_array_descrs, descr_idx) {
                continue;
            }
            cai.retain_entries(|obj| vb_get(&self.unescaped, obj.0));
        }
        self.cached_arrayitems_var
            .retain(|&(obj, descr_idx, _), _| {
                vb_get(&self.immutable_array_descrs, descr_idx) || vb_get(&self.unescaped, obj.0)
            });
    }

    /// Invalidate field caches affected by a write to `obj` for field `field_idx`.
    ///
    /// Uses aliasing analysis: objects allocated in this trace (seen_allocation)
    /// cannot alias each other. A write to a seen-allocation object only
    /// invalidates caches for unknown-origin objects with the same field.
    /// A write to an unknown-origin object invalidates all non-seen-allocation
    /// caches for that field.
    /// heap.py:206-226: __cannot_alias_via_content
    ///
    /// If both PtrInfos have fields and any shared field index has different
    /// constant values, the two objects cannot be the same struct at runtime.
    fn _cannot_alias_via_content(opref1: OpRef, opref2: OpRef, ctx: &OptContext) -> bool {
        let (Some(info1), Some(info2)) = (ctx.get_ptr_info(opref1), ctx.get_ptr_info(opref2))
        else {
            return false;
        };
        let f1 = info1.all_items();
        let f2 = info2.all_items();
        if f1.is_empty() || f2.is_empty() {
            return false;
        }
        // heap.py:217: for index in range(min(len(content1), len(content2)))
        // RPython iterates by position. Our fields are (idx, val) pairs —
        // match by field_idx which is equivalent when both objects share
        // the same descriptor layout.
        for &(idx1, v1) in f1 {
            for &(idx2, v2) in f2 {
                if idx1 != idx2 {
                    continue;
                }
                let v1r = ctx.get_box_replacement(v1);
                let v2r = ctx.get_box_replacement(v2);
                // heap.py:222-224: both constant and different → CANNOT_ALIAS
                if ctx.is_constant(v1r) && ctx.is_constant(v2r) && v1r != v2r {
                    return true;
                }
            }
        }
        false
    }

    /// heap.py:198-204: _cannot_alias_via_classes_or_lengths
    ///
    /// heap.py:198-204 `_cannot_alias_via_classes_or_lengths`:
    ///     constclass1 = opinfo1.get_known_class(optheap.optimizer.cpu)
    ///     constclass2 = opinfo2.get_known_class(optheap.optimizer.cpu)
    ///     if constclass1 is not None and constclass2 is not None and
    ///        not constclass1.same_constant(constclass2):
    ///         return CANNOT_ALIAS
    ///     return UNKNOWN_ALIAS
    ///
    /// `getptrinfo` synthesizes `ConstPtrInfo` for constant Refs so two
    /// constant ptrs with different typeptrs are detected as non-aliasing.
    fn _cannot_alias_via_classes_or_lengths(
        opref1: OpRef,
        opref2: OpRef,
        ctx: &OptContext,
    ) -> bool {
        let class1 = ctx.getptrinfo(opref1).and_then(|i| i.get_known_class());
        let class2 = ctx.getptrinfo(opref2).and_then(|i| i.get_known_class());
        matches!((class1, class2), (Some(c1), Some(c2)) if c1 != c2)
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
        for (&(_descr_idx, index), cai) in self.cached_arrayitems.iter_mut() {
            cai.retain_entries(|obj| {
                if *obj != dest_ref {
                    return true;
                }
                if let (Some(start), Some(len)) = (dest_start, length) {
                    if index < start || index >= start + len {
                        return true;
                    }
                }
                false
            });
        }
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
    /// heapcache.py:259-293 mark_escaped_varargs parity.
    /// ARRAYCOPY/ARRAYMOVE with constant indices and known array descriptor
    /// do NOT escape arguments.
    fn mark_escaped_varargs(&mut self, op: &Op, ctx: &OptContext) {
        let oopspec = Self::get_oopspec_index(op);
        // heapcache.py:275: single_write_descr_array is not None.
        // RPython: exactly one array write descriptor is known.
        // Bitstring: is_power_of_two means exactly one bit set.
        let has_single_write_descr = op
            .descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| {
                let w = cd.effect_info().write_descrs_arrays;
                w != 0 && w.is_power_of_two()
            })
            .unwrap_or(false);
        if oopspec == OopSpecIndex::Arraycopy
            && has_single_write_descr
            && op.args.len() >= 6
            && ctx.is_constant(op.args[3])
            && ctx.is_constant(op.args[4])
            && ctx.is_constant(op.args[5])
        {
            return;
        }
        if oopspec == OopSpecIndex::Arraymove
            && has_single_write_descr
            && op.args.len() >= 5
            && ctx.is_constant(op.args[2])
            && ctx.is_constant(op.args[3])
            && ctx.is_constant(op.args[4])
        {
            return;
        }
        for &arg in &op.args {
            self.escape_box(arg);
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
    ///
    /// heapcache.py:341 parity: mark_escaped_varargs runs BEFORE
    /// invalidate_unescaped so that call arguments are already escaped
    /// when cache invalidation checks unescaped status.
    fn force_from_effectinfo(&mut self, op: &Op, ctx: &mut OptContext) {
        // heapcache.py:259-293: escape call arguments first
        self.mark_escaped_varargs(op, ctx);

        let ei = match op.descr.as_ref().and_then(|d| d.as_call_descr()) {
            Some(cd) => cd.effect_info().clone(),
            None => {
                self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                self.clean_caches();
                return;
            }
        };

        // RPython effectinfo.py: zero bitstrings mean the call touches NO
        // tracked heap fields (e.g., I/O). Only fall back to conservative
        // invalidation for calls with ForcesVirtual/RandomEffects.
        // heap.py:567-571: forces_virtual_or_virtualizable → force virtualref field
        // (In RPython this forces vrefinfo.descr_forced; majit has no virtualref
        // field tracking yet, so this is a no-op placeholder.)
        //
        // Note: has_random_effects() calls are filtered BEFORE reaching this
        // function (emitting_operation line 460: !has_random_effects → return).
        // No special handling needed here.

        // heapcache.py:362-370: Only invalidate entries for escaped objects.
        // Unescaped allocations survive the call because the callee cannot
        // access them. Snapshot unescaped bitset to avoid borrow conflicts.
        let unescaped_snapshot = self.unescaped.clone();

        // Force/invalidate field caches based on read/write bitstrings
        let field_indices: Vec<u32> = self.cached_fields.keys().copied().collect();
        for field_idx in field_indices {
            if ei.check_readonly_descr_field(field_idx) {
                // Call reads this field → force lazy set (but keep cache)
                if let Some(cf) = self.cached_fields.get_mut(&field_idx) {
                    if let Some((_, mut lazy_op)) = cf.lazy_set.take() {
                        Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                    }
                }
            }
            if ei.check_write_descr_field(field_idx) {
                // heap.py:545-546: force_lazy_set(can_cache=False)
                // Call writes this field → force lazy set AND invalidate cache.
                // heapcache.py:362-370: only invalidate escaped entries.
                // For unescaped objects, re-cache the value after forcing the
                // lazy set so later reads can still be optimized away.
                if let Some(cf) = self.cached_fields.get_mut(&field_idx) {
                    let lazy_info = cf.lazy_set.take();
                    if let Some((obj, mut lazy_op)) = lazy_info {
                        // heap.py:139: put_field_back_to_info for unescaped
                        // objects: the callee cannot modify them, so re-cache
                        // after emitting.
                        let re_cache_value = if vb_get(&unescaped_snapshot, obj.0) {
                            Some((obj, lazy_op.arg(1), lazy_op.descr.clone()))
                        } else {
                            None
                        };
                        Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                        if let Some((obj, value, descr)) = re_cache_value {
                            cf.register_info(obj, value, descr.as_ref());
                        }
                    }
                    if !vb_get(&self.immutable_field_descrs, field_idx) {
                        // heapcache.py:365-366: cache.invalidate_unescaped()
                        cf.invalidate_unescaped(&unescaped_snapshot);
                    }
                }
            }
        }

        // Force/invalidate array caches
        let array_keys: Vec<(u32, i64)> = self.cached_arrayitems.keys().copied().collect();
        for (descr_idx, index) in array_keys {
            if ei.check_readonly_descr_array(descr_idx) {
                if let Some(cai) = self.cached_arrayitems.get_mut(&(descr_idx, index)) {
                    if let Some((_, mut lazy_op)) = cai.lazy_set.take() {
                        Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                    }
                }
            }
            if ei.check_write_descr_array(descr_idx) {
                if let Some(cai) = self.cached_arrayitems.get_mut(&(descr_idx, index)) {
                    let lazy_info = cai.lazy_set.take();
                    if let Some((arr, mut lazy_op)) = lazy_info {
                        let re_cache_value = if vb_get(&unescaped_snapshot, arr.0) {
                            // RPython: re-cache for unescaped array
                            Some((arr, lazy_op.arg(2), lazy_op.descr.clone()))
                        } else {
                            None
                        };
                        Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                        if let Some((arr, value, descr)) = re_cache_value {
                            cai.register_info(arr, value, descr.as_ref());
                        }
                    }
                    // heapcache.py:367-369: cache.invalidate_unescaped()
                    cai.invalidate_unescaped(&unescaped_snapshot);
                }
            }
        }
        self.cached_arrayitems_var
            .retain(|&(_, descr_idx, _), _| !ei.check_write_descr_array(descr_idx));
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
                vb_set(&mut self.immutable_field_descrs, key.1);
            }
        }

        // heap.py:640-643: constant_fold — pure getfield on constant object.
        //   if descr.is_always_pure() and self.get_constant_box(arg0):
        //       resbox = self.optimizer.constant_fold(op)
        //       self.optimizer.make_constant(op, resbox)
        if let Some(descr) = &op.descr {
            if descr.is_always_pure() {
                if ctx.get_constant_box(op.arg(0)).is_some() {
                    if let Some(value) = ctx.constant_fold(&op) {
                        let const_pos = ctx.alloc_op_position();
                        ctx.make_constant(const_pos, value);
                        ctx.replace_op(op.pos, const_pos);
                        return OptimizationResult::Remove;
                    }
                }
            }
        }

        // heap.py:103-120: getfield_from_cache — 3-way aliasing check.
        let (raw_obj, field_idx) = key;
        // Resolve forwarding (PtrInfo identity semantics, heap.py same_info).
        let obj = ctx.get_box_replacement(raw_obj);
        let mut force_lazy = false;
        if let Some(cf) = self.cached_fields.get(&field_idx) {
            if let Some((lazy_obj, lazy_op)) = &cf.lazy_set {
                if *lazy_obj == obj {
                    // MUST_ALIAS: lazy_set targets the same struct → return rhs
                    let cached = lazy_op.arg(1);
                    ctx.replace_op(op.pos, cached);
                    return OptimizationResult::Remove;
                }
                // heap.py:108: possible_aliasing_two_infos
                let lazy_obj_resolved = ctx.get_box_replacement(*lazy_obj);
                let cannot_alias =
                    Self::_cannot_alias_via_classes_or_lengths(lazy_obj_resolved, obj, ctx)
                        || Self::_cannot_alias_via_content(lazy_obj_resolved, obj, ctx);
                if !cannot_alias {
                    // UNKNOWN_ALIAS → force_lazy_set, return None (cache miss)
                    force_lazy = true;
                }
                // CANNOT_ALIAS: fall through to _getfield below (heap.py:117)
            }
            // heap.py:117-120: always check cache entries after alias analysis.
            // RPython falls through here even when lazy_set exists (CANNOT_ALIAS).
            if !force_lazy {
                if let Some(cached) = cf._getfield(obj) {
                    let cached = ctx.get_box_replacement(cached);
                    ctx.replace_op(op.pos, cached);
                    return OptimizationResult::Remove;
                }
            }
        }
        // heap.py:109-111: UNKNOWN_ALIAS → force lazy_set and return cache miss
        // heap.py:122: force_lazy_set(can_cache=True) — reads don't destroy cache,
        // so put_field_back_to_info restores the lazy value into the cache.
        // (Contrast with write-descr force in force_from_effectinfo which uses
        // can_cache=False and does NOT restore the value.)
        if force_lazy {
            if let Some(cf) = self.cached_fields.get_mut(&field_idx) {
                if let Some((lazy_obj, mut lazy_op)) = cf.lazy_set.take() {
                    cf.invalidate_with_ctx(field_idx, ctx);
                    if let Some(ref postponed) = self.postponed_op {
                        let ppos = postponed.pos;
                        if lazy_op.args.iter().any(|a| *a == ppos) {
                            if let Some(p) = self.postponed_op.take() {
                                ctx.emit_extra(ctx.current_pass_idx, p);
                            }
                        }
                    }
                    Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                    // can_cache=True: put_field_back_to_info
                    let final_value = lazy_op.arg(1);
                    let descr = lazy_op.descr.clone();
                    let cf = self.field_cache(field_idx);
                    cf.register_info(lazy_obj, final_value, descr.as_ref());
                    let lazy_resolved = ctx.get_box_replacement(lazy_obj);
                    if let Some(info) = ctx.get_ptr_info_mut(lazy_resolved) {
                        info.setfield(field_idx, final_value);
                    }
                }
            }
            // Cache miss — fall through to emit the getfield
        }

        // Virtualizable fields are loop-variant; skip caching/import.
        let is_vable_field = op.descr.as_ref().map_or(false, |d| d.is_virtualizable());

        // heap.py:177-187: CachedField._getfield — PreambleOp detection.
        //   res = opinfo.getfield(descr, optheap)
        //   if isinstance(res, PreambleOp):
        //       res = optheap.optimizer.force_op_from_preamble(res)
        //       opinfo.setfield(descr, None, res, optheap=optheap)
        //   return res
        if !is_vable_field {
            // heap.py:177-187: CachedField._getfield — PreambleOp detection.
            // info.py:716: ConstPtrInfo._get_info delegates to const_infos.
            // heap.py:177-187: CachedField._getfield PreambleOp detection.
            let pop = ctx
                .get_ptr_info_mut(obj)
                .and_then(|info| info.take_preamble_field(field_idx))
                .or_else(|| {
                    ctx.get_const_info_mut(obj)
                        .and_then(|info| info.take_preamble_field(field_idx))
                });
            if let Some(pop) = pop {
                let cached = pop.resolved;
                // heap.py:185-186: force_op_from_preamble(res)
                ctx.force_op_from_preamble_op(&pop);
                let d = op.descr.clone();
                let is_immutable = d.as_ref().map_or(false, |dd| dd.is_always_pure());
                if is_immutable {
                    vb_set(&mut self.immutable_field_descrs, key.1);
                    self.immutable_cached_fields.insert(key, cached);
                }
                let cf = self.field_cache(field_idx);
                if cf._getfield(obj).is_none() {
                    cf.register_info(obj, cached, d.as_ref());
                }
            }
        }

        // Check immutable field cache first — these survive all invalidation.
        if let Some(&cached) = self.immutable_cached_fields.get(&key) {
            let cached = ctx.get_box_replacement(cached);
            ctx.replace_op(op.pos, cached);
            return OptimizationResult::Remove;
        }

        // Check read cache (after import).
        if let Some(cf) = self.cached_fields.get(&field_idx) {
            if let Some(cached) = cf._getfield(obj) {
                let cached = ctx.get_box_replacement(cached);
                ctx.replace_op(op.pos, cached);
                return OptimizationResult::Remove;
            }
        }

        // Check quasi-immutable cache: if this field was marked by
        // QUASIIMMUT_FIELD, the value is stable (guarded by GUARD_NOT_INVALIDATED).
        if let Some(&qi_cached) = self.quasi_immut_cache.get(&key) {
            if !qi_cached.is_none() {
                // Subsequent read: reuse the cached value.
                let qi_cached = ctx.get_box_replacement(qi_cached);
                ctx.replace_op(op.pos, qi_cached);
                return OptimizationResult::Remove;
            }
            // First read after QUASIIMMUT_FIELD: emit the load, then cache
            // the result so it survives calls (unlike normal mutable fields).
            self.quasi_immut_cache.insert(key, op.pos);
            self.cache_field(obj, field_idx, op.pos, op.descr.as_ref());
            return OptimizationResult::Emit(op.clone());
        }

        // Cache miss: emit the load and cache the result.
        // heap.py line 652: make_nonnull(op.getarg(0))
        // optimizer.py:437-448: only set NonNull if no existing PtrInfo.
        let struct_ref = ctx.get_box_replacement(op.arg(0));
        vb_set(&mut self.known_nonnull, struct_ref.0);
        if ctx.get_ptr_info(struct_ref).is_none() {
            ctx.set_ptr_info(struct_ref, crate::optimizeopt::info::PtrInfo::nonnull());
        }
        self.cache_field(obj, field_idx, op.pos, op.descr.as_ref());
        // Save immutable fields in the permanent cache — they survive all
        // invalidation because the value never changes.
        if vb_get(&self.immutable_field_descrs, key.1) {
            self.immutable_cached_fields.insert(key, op.pos);
        }
        // heap.py postprocess_GETFIELD_GC_I: structinfo.setfield(descr, op)
        // Record the field value in ptr_info so other passes can see it.
        if !is_vable_field {
            if let Some(info) = ctx.get_ptr_info_mut(struct_ref) {
                info.setfield(key.1, op.pos);
            }
        }
        // Virtualizable Ref fields (linked list head) need a null guard.
        let is_vable_ref =
            is_vable_field && matches!(op.opcode, OpCode::GetfieldGcR | OpCode::GetfieldGcPureR);
        if is_vable_ref {
            ctx.emit(op.clone());
            let zero_ref = ctx.make_constant_int(0);
            let cmp_pos = ctx.alloc_op_position();
            let mut cmp_op = Op::new(OpCode::IntNe, &[op.pos, zero_ref]);
            cmp_op.pos = cmp_pos;
            ctx.emit(cmp_op);
            // unroll.py:409 parity: synthetic guards inherit
            // rd_resume_position from patchguardop (the optimizer's
            // running GUARD_FUTURE_CONDITION). Without this, the guard
            // arrives at store_final_boxes_in_guard with -1 and would
            // be silently dropped under the patchguardop-only fallback.
            let mut guard_op = Op::new(OpCode::GuardTrue, &[cmp_pos]);
            if let Some(ref patch) = ctx.patchguardop {
                guard_op.rd_resume_position = patch.rd_resume_position;
            }
            ctx.emit(guard_op);
            return OptimizationResult::Remove;
        }
        OptimizationResult::Emit(op.clone())
    }

    fn optimize_setfield(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let key = match Self::field_key(op) {
            Some(k) => k,
            None => return OptimizationResult::Emit(op.clone()),
        };

        // RPython heap.py keeps immutability on the descriptor itself, so
        // SetfieldGc on an always-pure field must also seed the immutable
        // descriptor table before any cache invalidation happens.
        if let Some(descr) = &op.descr {
            if descr.is_always_pure() {
                vb_set(&mut self.immutable_field_descrs, key.1);
            }
        }

        let (raw_obj, field_idx) = key;
        // Resolve forwarding so that two OpRefs pointing to the same object
        // compare equal (PtrInfo identity semantics, heap.py same_info).
        let obj = ctx.get_box_replacement(raw_obj);
        let new_value = op.arg(1);

        // heapcache.py:224-230 _escape_from_write parity:
        // record dependency if both container and value are unescaped;
        // otherwise escape the value immediately.
        self.escape_from_write(obj, new_value);

        // heap.py:77-101: do_setfield — check write-after-write, aliasing, lazy set
        // Check write-after-write first (before possible_aliasing).
        {
            let cf = self.field_cache(field_idx);
            if let Some((lazy_obj, lazy_op)) = &cf.lazy_set {
                if *lazy_obj == obj && lazy_op.arg(1) == new_value {
                    return OptimizationResult::Remove;
                }
            } else if let Some(&cached) = cf._getfield(obj).as_ref() {
                let cached_resolved = ctx.get_box_replacement(cached);
                if cached_resolved == new_value {
                    return OptimizationResult::Remove;
                }
            }
        }

        // heap.py:81-83: possible_aliasing → force_lazy_set.
        // RPython: possible_aliasing checks only whether lazy_set targets
        // a DIFFERENT object. If yes, force unconditionally — the
        // _cannot_alias_via_classes optimization applies to cache
        // invalidation (possible_aliasing_two_infos), not to force_lazy_set.
        let needs_force = self
            .cached_fields
            .get(&field_idx)
            .map_or(false, |cf| cf.possible_aliasing(obj));
        if needs_force {
            let lazy_data = self
                .cached_fields
                .get_mut(&field_idx)
                .and_then(|cf| cf.lazy_set.take());
            if let Some((lazy_obj, mut lazy_op)) = lazy_data {
                // heap.py:122-143: force_lazy_set
                // 1. invalidate (skip pure)
                if !vb_get(&self.immutable_field_descrs, field_idx) {
                    self.field_cache(field_idx)
                        .invalidate_with_ctx(field_idx, ctx);
                }
                // 2. emit postponed_op if referenced
                if let Some(ref postponed) = self.postponed_op {
                    let ppos = postponed.pos;
                    if lazy_op.args.iter().any(|a| *a == ppos) {
                        if let Some(p) = self.postponed_op.take() {
                            ctx.emit_extra(ctx.current_pass_idx, p);
                        }
                    }
                }
                // 3. emit the setfield
                Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                // 4. put_field_back_to_info
                let final_value = lazy_op.arg(1);
                let descr = lazy_op.descr.clone();
                self.cache_field(lazy_obj, field_idx, final_value, descr.as_ref());
                let lazy_obj_resolved = ctx.get_box_replacement(lazy_obj);
                if let Some(info) = ctx.get_ptr_info_mut(lazy_obj_resolved) {
                    info.setfield(field_idx, final_value);
                }
            }
        }

        // heap.py:84-101: after force, recheck cached value
        {
            let cf = self.field_cache(field_idx);
            if let Some(&cached) = cf._getfield(obj).as_ref() {
                let cached_resolved = ctx.get_box_replacement(cached);
                if cached_resolved == new_value {
                    cf.lazy_set = None;
                    return OptimizationResult::Remove;
                }
            }
        }

        // RPython do_setfield: no separate invalidation here.
        // Aliasing is handled by force_lazy_set → invalidate inside the
        // possible_aliasing path above. Getfield handles read-after-write
        // aliasing via its own force path.

        // heap.py:89-101: store as lazy_set, update PtrInfo._fields.
        // No cache_field here — RPython only updates _lazy_set and info._fields.
        // getfield reads from lazy_set first (CachedField.get checks lazy_set).
        let cf = self.field_cache(field_idx);
        cf.lazy_set = Some((obj, op.clone()));
        if let Some(info) = ctx.get_ptr_info_mut(obj) {
            info.setfield(field_idx, new_value);
        }
        OptimizationResult::Remove
    }

    fn optimize_getarrayitem(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // Try constant-index cache first.
        if let Some(key) = Self::arrayitem_key(op, ctx) {
            let (array, descr_idx, const_index) = key;
            // Check lazy_set first (most up-to-date value)
            if let Some(cai) = self.cached_arrayitems.get(&(descr_idx, const_index)) {
                if let Some(cached) = cai.getfield_from_cache(array) {
                    ctx.replace_op(op.pos, cached);
                    return OptimizationResult::Remove;
                }
            }
            // Consume the imported short arrayitem: remove it so that if a later
            // setarrayitem/call invalidates cached_arrayitems, the stale preamble
            // value cannot re-populate the cache on a subsequent getarrayitem.
            if let Some(cached) = ctx.imported_short_arrayitems.remove(&key) {
                let cached = ctx.force_op_from_preamble(cached);
                let cai = self.arrayitem_cache(descr_idx, const_index);
                if cai._getfield(array).is_none() {
                    let d = ctx
                        .imported_short_arrayitem_descrs
                        .remove(&key)
                        .or_else(|| op.descr.clone());
                    cai.register_info(array, cached, d.as_ref());
                }
            }
            if let Some(cai) = self.cached_arrayitems.get(&(descr_idx, const_index)) {
                if let Some(cached) = cai._getfield(array) {
                    let cached = ctx.get_box_replacement(cached);
                    ctx.replace_op(op.pos, cached);
                    return OptimizationResult::Remove;
                }
            }
            self.cache_arrayitem(array, descr_idx, const_index, op.pos, op.descr.as_ref());
            // Track immutable array descriptors so clean_caches() preserves them.
            // RPython: descr.is_always_pure() — same pattern as immutable_field_descrs.
            if let Some(descr) = &op.descr {
                if descr.is_always_pure() {
                    vb_set(&mut self.immutable_array_descrs, descr_idx);
                }
            }
            // heap.py line 701: make_nonnull(op.getarg(0))
            let array_ref = ctx.get_box_replacement(op.arg(0));
            vb_set(&mut self.known_nonnull, array_ref.0);
            // heap.py line 681: arrayinfo.getlenbound(None).make_gt_const(index)
            // Record that array length >= index + 1
            if const_index >= 0 {
                let min_len = const_index + 1;
                let entry = self.array_min_lengths.entry(array_ref).or_insert(0);
                if min_len > *entry {
                    *entry = min_len;
                }
            }
            if let Some(info) = ctx.get_ptr_info_mut(array_ref) {
                info.setitem(const_index as usize, op.pos);
            }
            return OptimizationResult::Emit(op.clone());
        }

        // heap.py:688-699 variable-index cache — same array + same index OpRef.
        if let Some(var_key) = Self::arrayitem_key_variable(op, ctx) {
            if let Some(&cached) = self.cached_arrayitems_var.get(&var_key) {
                let cached = ctx.get_box_replacement(cached);
                ctx.replace_op(op.pos, cached);
                return OptimizationResult::Remove;
            }
            self.cached_arrayitems_var.insert(var_key, op.pos);
        }

        // heap.py line 701: make_nonnull(op.getarg(0))
        vb_set(
            &mut self.known_nonnull,
            ctx.get_box_replacement(op.arg(0)).0,
        );
        OptimizationResult::Emit(op.clone())
    }

    fn optimize_setarrayitem(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // heapcache.py:224-230 _escape_from_write parity:
        let array_obj = ctx.get_box_replacement(op.arg(0));
        let stored_value = op.arg(2);
        self.escape_from_write(array_obj, stored_value);

        let key = match Self::arrayitem_key(op, ctx) {
            Some(k) => k,
            None => {
                // Non-constant index: force all lazy array stores and invalidate
                // both constant-index and variable-index caches.
                // heap.py: ArrayCachedItem.invalidate() calls parent.clear_varindex()
                self.force_all_lazy_setarrayitems(ctx.current_pass_idx, ctx);
                self.cached_arrayitems
                    .values_mut()
                    .for_each(|cai| cai.invalidate());
                self.cached_arrayitems_var.clear();
                self.array_min_lengths.clear();
                // heap.py: cache_varindex_write -- cache this write so that
                // a subsequent read with the same variable index can hit.
                // heap.py:764 cache_varindex_write: canonicalize value
                if let Some(var_key) = Self::arrayitem_key_variable(op, ctx) {
                    self.cached_arrayitems_var
                        .insert(var_key, ctx.get_box_replacement(op.arg(2)));
                }
                return OptimizationResult::Emit(op.clone());
            }
        };

        let (array, descr_idx, const_index) = key;
        // array is already canonicalized by arrayitem_key
        let new_value = op.arg(2);

        // heap.py:77-101: do_setfield (shared by CachedField AND ArrayCachedItem)
        // Write-after-write check.
        {
            let cai = self.arrayitem_cache(descr_idx, const_index);
            if let Some((lazy_obj, lazy_op)) = &cai.lazy_set {
                if *lazy_obj == array && lazy_op.arg(2) == new_value {
                    return OptimizationResult::Remove;
                }
            } else if let Some(cached) = cai._getfield(array) {
                let cached = ctx.get_box_replacement(cached);
                if cached == new_value {
                    return OptimizationResult::Remove;
                }
            }
        }

        // heap.py:81-83: possible_aliasing → force_lazy_set
        let needs_force = self
            .cached_arrayitems
            .get(&(descr_idx, const_index))
            .map_or(false, |cai| cai.possible_aliasing(array));
        if needs_force {
            let lazy_data = self
                .cached_arrayitems
                .get_mut(&(descr_idx, const_index))
                .and_then(|cai| cai.lazy_set.take());
            if let Some((lazy_obj, mut lazy_op)) = lazy_data {
                let cai = self.arrayitem_cache(descr_idx, const_index);
                cai.invalidate();
                if let Some(ref postponed) = self.postponed_op {
                    let ppos = postponed.pos;
                    if lazy_op.args.iter().any(|a| *a == ppos) {
                        if let Some(p) = self.postponed_op.take() {
                            ctx.emit_extra(ctx.current_pass_idx, p);
                        }
                    }
                }
                Self::emit_lazy_setfield(&mut lazy_op, ctx, true);
                // put_field_back_to_info
                let final_value = lazy_op.arg(2);
                let descr = lazy_op.descr.clone();
                self.cache_arrayitem(
                    lazy_obj,
                    descr_idx,
                    const_index,
                    final_value,
                    descr.as_ref(),
                );
            }
        }

        // heap.py:84-101: recheck after force
        {
            let cai = self.arrayitem_cache(descr_idx, const_index);
            if let Some(cached) = cai._getfield(array) {
                let cached = ctx.get_box_replacement(cached);
                if cached == new_value {
                    cai.lazy_set = None;
                    return OptimizationResult::Remove;
                }
            }
        }

        // Store as lazy set. heap.py:89-90
        let cai = self.arrayitem_cache(descr_idx, const_index);
        cai.lazy_set = Some((array, op.clone()));
        // Remove stale entry
        let idx = cai.cached_arrays.iter().position(|a| *a == array);
        if let Some(idx) = idx {
            cai.cached_arrays.swap_remove(idx);
            cai.cached_values.swap_remove(idx);
            cai.cached_descrs.retain(|(k, _)| *k != array);
        }
        if let Some(info) = ctx.get_ptr_info_mut(ctx.get_box_replacement(op.arg(0))) {
            info.setitem(const_index as usize, new_value);
        }
        // heap.py: ArrayCachedItem.invalidate() calls parent.clear_varindex()
        // -- writing to any constant index invalidates variable-index cache
        // for the same array+descr, since the variable index could match.
        self.cached_arrayitems_var
            .retain(|&(a, d, _), _| !(a == array && d == descr_idx));

        OptimizationResult::Remove
    }

    /// Handle operations that may have side effects.
    /// Forces lazy sets and invalidates caches as needed.
    /// Tracks allocations for aliasing analysis.
    fn handle_side_effects(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let opcode = op.opcode;

        // Track allocations for aliasing analysis.
        // Allocated objects are always non-null.
        if opcode.is_malloc() {
            vb_set(&mut self.seen_allocation, op.pos.0);
            vb_set(&mut self.unescaped, op.pos.0);
            vb_set(&mut self.known_nonnull, op.pos.0);
            return OptimizationResult::Emit(op.clone());
        }

        // Note: postponed_op (from CallMayForce) must only be emitted at
        // GuardNotForced, not at arbitrary guards. RPython's emit() callback
        // calls emit_postponed_op() before every op, but the postpone→emit
        // cycle is specifically CallMayForce→GuardNotForced. Don't emit here.

        // Guards: force lazy sets but keep caches (guards don't mutate the heap).
        // RPython heap.py does NOT handle GuardNonnull — that is handled
        // exclusively by rewrite.py. Only track nullity for cache purposes.
        if opcode.is_guard() {
            if opcode == OpCode::GuardNonnull {
                vb_set(&mut self.known_nonnull, op.arg(0).0);
            }

            // GuardClass / GuardNonnullClass imply non-null.
            // GuardValue does NOT imply non-null: the guarded constant
            // may be NULL (rewrite.py:320 handles GUARD_VALUE(..., NULL)).
            match opcode {
                OpCode::GuardClass | OpCode::GuardNonnullClass => {
                    vb_set(&mut self.known_nonnull, op.arg(0).0);
                }
                _ => {}
            }

            // force_lazy_sets_for_guard is now called via emitting_operation
            // callback (which runs for ALL guards regardless of which pass emits
            // them). No need to force here — it was already done.
            return OptimizationResult::Emit(op.clone());
        }

        // Final operations (Jump, Finish): force everything.
        if opcode.is_final() {
            self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
            return OptimizationResult::Emit(op.clone());
        }

        // Calls: mark arguments as escaped, force lazy sets, and invalidate.
        if opcode.is_call() {
            let oopspec = Self::get_oopspec_index(op);
            match oopspec {
                // heap.py: DICT_LOOKUP caching — consecutive dict lookups
                // on the same dict with the same key can be deduplicated.
                OopSpecIndex::DictLookup => {
                    self.mark_escaped_varargs(op, ctx);
                    self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                    // Invalidate dict-related caches but keep field/array caches.
                    // Dict operations don't affect struct fields.
                    return OptimizationResult::Emit(op.clone());
                }
                OopSpecIndex::Arraycopy | OopSpecIndex::Arraymove => {
                    // ARRAYCOPY/ARRAYMOVE: only invalidate affected array entries.
                    // Call args: [func_addr, source, dest, source_start, dest_start, length, ...]
                    // args[2] = dest array, args[4] = dest_start, args[5] = length
                    self.mark_escaped_varargs(op, ctx);
                    self.force_all_lazy_sets(ctx.current_pass_idx, ctx);

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
                        self.clean_caches();
                    }

                    return OptimizationResult::Emit(op.clone());
                }
                _ => {
                    self.mark_escaped_varargs(op, ctx);
                    // heapcache.py:337-369 clear_caches_varargs
                    // Plain residual calls preserve cache entries for
                    // unescaped allocations. Calls with explicit EffectInfo
                    // keep the more precise heap.py force_from_effectinfo path.
                    if op.descr.is_none() {
                        self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                        self.invalidate_caches_for_escaped();
                    } else if Self::call_has_random_effects(op) {
                        self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                        self.clean_caches();
                    } else {
                        // heap.py: force_from_effectinfo — selective cache
                        // invalidation using EffectInfo bitstrings.
                        self.force_from_effectinfo(op, ctx);
                    }
                    if Self::call_can_invalidate(op) {
                        self.seen_guard_not_invalidated = false;
                    }
                    return OptimizationResult::Emit(op.clone());
                }
            }
        }

        // Other side-effecting ops: force and invalidate.
        if !opcode.has_no_side_effect() && !opcode.is_ovf() {
            self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
            self.clean_caches();
            return OptimizationResult::Emit(op.clone());
        }

        // Pure / no-side-effect / overflow ops: pass through.
        OptimizationResult::Emit(op.clone())
    }

    fn dispatch_propagate(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        match op.opcode {
            // ── Field reads ──
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldGcPureI
            | OpCode::GetfieldGcPureR
            | OpCode::GetfieldGcPureF => self.optimize_getfield(op, ctx),

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

            // ── Interior field reads ──
            // info.py:682: "heapcache does not work for interiorfields"
            // RPython has no optimize_GETINTERIORFIELD_GC handler — falls
            // through to optimize_default (just emit). GETINTERIORFIELD has
            // no side effect so emitting_operation returns early (heap.py:428).
            OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF => OptimizationResult::Emit(op.clone()),
            // SETINTERIORFIELD_GC: NOT matched here — falls through to
            // handle_side_effects (the `_` arm). RPython heap.py:463-464:
            // SETINTERIORFIELD_GC is NOT in the emitting_operation exclusion
            // list, so it triggers force_all_lazy_sets + clean_caches.

            // ── heap.py: ARRAYLEN_GC — cache array lengths ──
            OpCode::ArraylenGc => {
                let array = ctx.get_box_replacement(op.arg(0));
                let descr_idx = op.descr.as_ref().map(|d| d.index()).unwrap_or(0);
                if let Some(&cached) = self.cached_arraylens.get(&(array, descr_idx)) {
                    let cached = ctx.get_box_replacement(cached);
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
                    let cached = ctx.get_box_replacement(cached);
                    ctx.replace_op(op.pos, cached);
                    return OptimizationResult::Remove;
                }
                self.cached_arraylens.insert(key, op.pos);
                OptimizationResult::PassOn // let intbounds set non-negative
            }

            // ── heap.py: Allocation tracking ──
            OpCode::New | OpCode::NewWithVtable | OpCode::NewArray | OpCode::NewArrayClear => {
                vb_set(&mut self.seen_allocation, op.pos.0);
                vb_set(&mut self.unescaped, op.pos.0);
                vb_set(&mut self.known_nonnull, op.pos.0);
                OptimizationResult::PassOn
            }

            // RPython heap.py: CALL_ASSEMBLER — force all lazy sets before
            // the call. The callee reads from the allocated objects passed
            // in the args array; any pending SetfieldGc must be flushed to
            // memory before execution transfers to the callee.
            //
            // Unlike force_all_lazy_setfields (which mirrors RPython's
            // emit_extra(emit=False) and drops non-virtualizable ops),
            // CALL_ASSEMBLER REQUIRES the SetfieldGc ops to reach the
            // compiled code so that forced-virtual objects have their
            // fields initialized before the callee reads them.
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                // heap.py:454-455: call_assembler always resets
                // _seen_guard_not_invalidated (can call arbitrary code).
                self.seen_guard_not_invalidated = false;
                self.mark_escaped_varargs(op, ctx);
                // heap.py:463-464: force_all_lazy_sets + clean_caches.
                self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                self.clean_caches();
                return OptimizationResult::Emit(op.clone());
            }

            // ── heap.py: CALL_MAY_FORCE — postpone until GUARD_NOT_FORCED ──
            // These calls may force virtualizable objects, so we defer emission
            // until the guard arrives, ensuring correct exception semantics.
            OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN => {
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[opt-heap] postpone {:?} pos={:?} descr={:?}",
                        op.opcode, op.pos, op.descr
                    );
                }
                // RPython emitting_operation: calls go through
                // force_from_effectinfo (selective) or clean_caches,
                // NOT force_all_lazy. force_all_lazy is only in flush().
                self.mark_escaped_varargs(op, ctx);
                // Postpone the call — it will be emitted when GUARD_NOT_FORCED arrives.
                self.postponed_op = Some(op.clone());
                if Self::call_has_random_effects(op) {
                    self.clean_caches();
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
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!(
                            "[opt-heap] emit postponed {:?} pos={:?} before {:?} pos={:?}",
                            postponed.opcode, postponed.pos, op.opcode, op.pos
                        );
                    }
                    // RPython emit_postponed_op: route through next_optimization
                    ctx.emit_extra(ctx.current_pass_idx, postponed);
                } else if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[opt-heap] no postponed op before {:?} pos={:?}",
                        op.opcode, op.pos
                    );
                }
                // RPython emitting_operation for guards:
                //   self.optimizer.pendingfields = self.force_lazy_sets_for_guard()
                let pending_virtual = self.force_lazy_sets_for_guard(ctx.current_pass_idx, ctx);
                for pending_op in pending_virtual {
                    if pending_op.opcode == OpCode::SetarrayitemGc {
                        let descr_idx = pending_op.descr.as_ref().map_or(0, |d| d.index());
                        if let Some(index) = ctx.get_constant_int(pending_op.arg(1)) {
                            let array = ctx.get_box_replacement(pending_op.arg(0));
                            let cai = self.arrayitem_cache(descr_idx, index);
                            cai.lazy_set = Some((array, pending_op));
                        } else {
                            ctx.emit(pending_op);
                        }
                    } else {
                        let field_idx = pending_op.descr.as_ref().map_or(0, |d| d.index());
                        let obj = ctx.get_box_replacement(pending_op.arg(0));
                        let cf = self.field_cache(field_idx);
                        cf.lazy_set = Some((obj, pending_op));
                    }
                }
                return OptimizationResult::Emit(op.clone());
            }

            // ── heap.py: COND_CALL handling ──
            OpCode::CondCallN => {
                self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                self.clean_caches();
                OptimizationResult::PassOn
            }

            // heap.py:530-535 optimize_GUARD_NO_EXCEPTION
            // (alias optimize_GUARD_EXCEPTION = optimize_GUARD_NO_EXCEPTION):
            //
            //     def optimize_GUARD_NO_EXCEPTION(self, op):
            //         if self.last_emitted_operation is REMOVED:
            //             return
            //         return self.emit(op)
            //     optimize_GUARD_EXCEPTION = optimize_GUARD_NO_EXCEPTION
            //
            // The REMOVED check fires only on paths that explicitly set
            // self.last_emitted_operation = REMOVED — currently just
            // _optimize_CALL_DICT_LOOKUP (heap.py:527). emit() flushes the
            // postponed op and runs emitting_operation, which delegates
            // force_lazy_sets_for_guard via the guard branch.
            OpCode::GuardNoException | OpCode::GuardException => {
                if self.prior_emitted_was_removed {
                    return OptimizationResult::Remove;
                }
                // emit() override: flush postponed op before the guard.
                if let Some(postponed) = self.postponed_op.take() {
                    ctx.emit_extra(ctx.current_pass_idx, postponed);
                }
                self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                return OptimizationResult::Emit(op.clone());
            }

            // ── GUARD_NOT_INVALIDATED deduplication ──
            OpCode::GuardNotInvalidated => {
                if self.seen_guard_not_invalidated {
                    OptimizationResult::Remove
                } else {
                    self.seen_guard_not_invalidated = true;
                    self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
                    OptimizationResult::Emit(op.clone())
                }
            }

            // Quasi-immutable field: treat as read + guard_not_invalidated.
            // The QUASIIMMUT_FIELD op marks a field that rarely changes.
            // The optimizer replaces the field read with the cached value and
            // emits GUARD_NOT_INVALIDATED to ensure validity.
            OpCode::QuasiimmutField => {
                // RPython optimize_QUASIIMMUT_FIELD (heap.py:781):
                // Does NOT create a new GUARD_NOT_INVALIDATED — the tracer
                // already emitted one via generate_guard (pyjitpl.py:1087).
                // Records quasi_immutable_deps for invalidation tracking.
                let obj = op.arg(0);
                // RPython optimize_QUASIIMMUT_FIELD: collect quasi-immutable
                // dependencies. Add (obj_ptr, field_idx) to quasi_immutable_deps
                // for per-slot watcher registration after compilation.
                // field_idx comes from descr (GC object fields) or arg(1)
                // (namespace slot index).
                let field_idx = if let Some(descr) = &op.descr {
                    Some(descr.index())
                } else if op.args.len() > 1 {
                    ctx.get_constant_int(op.arg(1)).map(|v| v as u32)
                } else {
                    None
                };
                if let Some(idx) = field_idx {
                    if let Some(dep_ptr) = ctx.get_constant_int(obj) {
                        ctx.add_quasi_immutable_dep((dep_ptr as u64, idx));
                    }
                    self.quasi_immut_cache.insert((obj, idx), OpRef::NONE);
                }
                OptimizationResult::Remove
            }

            // ── heap.py: RAW_LOAD — cache raw memory reads ──
            OpCode::RawLoadI | OpCode::RawLoadF => {
                if let Some((obj, field_idx)) = Self::field_key(op) {
                    if let Some(cached) = self
                        .get_cached_field(field_idx)
                        .and_then(|cf| cf._getfield(obj))
                    {
                        let cached = ctx.get_box_replacement(cached);
                        ctx.replace_op(op.pos, cached);
                        return OptimizationResult::Remove;
                    }
                    self.cache_field(obj, field_idx, op.pos, op.descr.as_ref());
                }
                OptimizationResult::Emit(op.clone())
            }

            // ── heap.py: RAW_STORE — invalidate raw memory cache ──
            OpCode::RawStore => {
                if let Some((obj, field_idx)) = Self::field_key(op) {
                    self.cache_field(obj, field_idx, op.arg(1), op.descr.as_ref());
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
                self.force_all_lazy_setfields(ctx.current_pass_idx, ctx);
                self.force_all_lazy_setarrayitems(ctx.current_pass_idx, ctx);
                vb_set(&mut self.known_nonnull, op.arg(0).0);
                OptimizationResult::Emit(op.clone())
            }

            // ── Everything else: check for side effects ──
            _ => self.handle_side_effects(op, ctx),
        }
    }
}

impl Default for OptHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptHeap {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // optimizer.py:84-92 base emit/emit_result reset
        // last_emitted_operation to the current op on every emit. RPython's
        // `last_emitted is REMOVED` check therefore reads the prior op's
        // outcome — model that by snapshotting the flag at entry and
        // resetting it. Removal paths set the flag back to true before
        // returning Remove. dispatch_propagate reads the snapshot via
        // self.prior_emitted_was_removed.
        self.prior_emitted_was_removed = self.last_emitted_was_removed;
        self.last_emitted_was_removed = false;
        let result = self.dispatch_propagate(op, ctx);
        // RPython heap.py:417-425 emit() override parity:
        // Before emitting any new op, flush the postponed op. Then
        // postpone comparison/ovf ops (call_may_force already handled
        // in its own match arm).
        if let OptimizationResult::Emit(ref emit_op) = result {
            // Step 1: emit_postponed_op — flush previous postponed
            if let Some(postponed) = self.postponed_op.take() {
                ctx.emit_extra(ctx.current_pass_idx, postponed);
            }
            // Step 2: postpone comparison/ovf
            if emit_op.opcode.is_comparison() || emit_op.opcode.is_ovf() {
                self.postponed_op = Some(emit_op.clone());
                return OptimizationResult::Remove;
            }
        }
        result
    }

    fn setup(&mut self) {
        self.cached_fields.clear();
        self.immutable_cached_fields.clear();
        self.cached_arrayitems.clear();
        self.seen_guard_not_invalidated = false;
        self.postponed_op = None;
        self.immutable_field_descrs.clear();
        self.immutable_array_descrs.clear();
        self.seen_allocation.clear();
        self.unescaped.clear();
        self.known_nonnull.clear();
        self.last_emitted_was_removed = false;
        self.prior_emitted_was_removed = false;
        self.quasi_immut_cache.clear();
        self.cached_arraylens.clear();
        self.cached_arrayitems_var.clear();
        self.array_min_lengths.clear();
    }

    fn flush(&mut self, ctx: &mut OptContext) {
        // RPython heap.py: flush() = force_all_lazy_sets(); emit_postponed_op()
        self.force_all_lazy_sets(ctx.current_pass_idx, ctx);
        // RPython emit_postponed_op: route through next_optimization
        if let Some(postponed) = self.postponed_op.take() {
            ctx.emit_extra(ctx.current_pass_idx, postponed);
        }
    }

    fn flush_virtualizable(&mut self, ctx: &mut OptContext) {
        // Collect virtualizable lazy sets from all CachedFields.
        let vable_entries: Vec<(u32, OpRef, Op)> = self
            .cached_fields
            .iter_mut()
            .filter_map(|(&field_idx, cf)| {
                if let Some((obj, ref op)) = cf.lazy_set {
                    if op.descr.as_ref().map_or(false, |d| d.is_virtualizable()) {
                        let op = op.clone();
                        cf.lazy_set = None;
                        return Some((field_idx, obj, op));
                    }
                }
                None
            })
            .collect();
        for (field_idx, obj, mut op) in vable_entries {
            let value_ref = ctx.get_box_replacement(op.arg(1));
            if let Some(mut info) = ctx.get_ptr_info(value_ref).cloned() {
                if info.is_virtual() {
                    info.force_box(value_ref, ctx);
                }
            }
            for arg in op.args.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
            let final_value = op.arg(1);
            let descr = op.descr.clone();
            ctx.emit(op);
            self.cache_field(obj, field_idx, final_value, descr.as_ref());
        }
    }

    /// RPython heap.py: emitting_operation(op)
    /// Called for EVERY op about to be emitted, regardless of which pass emits it.
    /// This is how the heap optimizer forces lazy sets before guards even when
    /// the guard was emitted by an earlier pass (e.g., IntBounds).
    fn emitting_operation(&mut self, op: &Op, ctx: &mut OptContext, self_pass_idx: usize) {
        // heap.py:427-464: emitting_operation(op)
        //
        // RPython calls emitting_operation in heap pass context.
        // Save/restore current_pass_idx so internal methods
        // (force_from_effectinfo, etc.) use the correct heap index.
        let saved_pass_idx = ctx.current_pass_idx;
        ctx.current_pass_idx = self_pass_idx;

        // RPython early returns for side-effect-free operations:
        if op.opcode.has_no_side_effect() {
            ctx.current_pass_idx = saved_pass_idx;
            return;
        }
        if op.opcode.is_ovf() {
            ctx.current_pass_idx = saved_pass_idx;
            return;
        }
        // heap.py:432-434: guards → force lazy sets for guard
        if op.opcode.is_guard() {
            let pending_virtual = self.force_lazy_sets_for_guard(self_pass_idx, ctx);
            // heap.py:433: self.optimizer.pendingfields = pendingfields
            ctx.pending_for_guard = pending_virtual;
            ctx.current_pass_idx = saved_pass_idx;
            return;
        }
        // heap.py:436-452: specific opcodes that don't affect GC caches
        match op.opcode {
            OpCode::SetfieldGc
            | OpCode::SetfieldRaw
            | OpCode::SetarrayitemGc
            | OpCode::SetarrayitemRaw
            | OpCode::SetinteriorfieldRaw
            | OpCode::RawStore
            | OpCode::Strsetitem
            | OpCode::Unicodesetitem
            | OpCode::DebugMergePoint
            | OpCode::JitDebug
            | OpCode::EnterPortalFrame
            | OpCode::LeavePortalFrame
            | OpCode::Copystrcontent
            | OpCode::Copyunicodecontent
            | OpCode::CheckMemoryError => {
                ctx.current_pass_idx = saved_pass_idx;
                return;
            }
            _ => {}
        }
        // heap.py:453-463: calls → handle effects
        if op.opcode.is_call() {
            if op.opcode.is_call_assembler() {
                self.seen_guard_not_invalidated = false;
            } else {
                if Self::call_can_invalidate(op) {
                    self.seen_guard_not_invalidated = false;
                }
                if op.descr.is_none() {
                    self.force_all_lazy_sets(self_pass_idx, ctx);
                    self.invalidate_caches_for_escaped();
                    ctx.current_pass_idx = saved_pass_idx;
                    return;
                }
                if !Self::call_has_random_effects(op) {
                    self.force_from_effectinfo(op, ctx);
                    ctx.current_pass_idx = saved_pass_idx;
                    return;
                }
            }
        }
        // heap.py:464: everything else → force all lazy sets + clean caches
        self.force_all_lazy_sets(self_pass_idx, ctx);
        self.clean_caches();
        ctx.current_pass_idx = saved_pass_idx;
    }

    /// heap.py: produce_potential_short_preamble_ops(sb)
    ///
    /// Add cached field/array reads to the short preamble so bridges
    /// can re-populate the optimizer's cache.
    /// heap.py:360-377: export ALL cached fields to short preamble.
    /// heap.py:360-377: No is_reachable filter — export ALL cached fields.
    /// heap.py:55: structbox = get_box_replacement(self.cached_structs[i])
    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        ctx: &OptContext,
    ) {
        // heap.py:370-372: for descr in cached_fields: d.produce_...(optimizer, sb, descr)
        // heap.py:54: for i, info in enumerate(self.cached_infos):
        // info.py:262: op = get_box_replacement(self._fields[fielddescr.get_index()])
        for (&field_idx, cf) in &self.cached_fields {
            // heap.py:53: assert self._lazy_set is None
            // Skip fields with pending lazy_set — the cache is
            // out-of-date and would produce wrong preamble ops.
            if cf.lazy_set.is_some() {
                continue;
            }
            for i in 0..cf.cached_structs.len() {
                // heap.py:55: structbox = get_box_replacement(self.cached_structs[i])
                let obj = ctx.get_box_replacement(cf.cached_structs[i]);
                if obj.is_none() {
                    continue;
                }
                // info.py:262: read from PtrInfo._fields (RPython reads info._fields[descr])
                // Fall back to cached_values if PtrInfo doesn't have the field.
                let cached_val = ctx
                    .get_ptr_info(obj)
                    .and_then(|info| info.getfield(field_idx))
                    .map(|v| ctx.get_box_replacement(v))
                    .unwrap_or_else(|| ctx.get_box_replacement(cf.cached_values[i]));
                if cached_val.is_none() {
                    continue;
                }
                let Some(descr) = cf
                    .get_descr(cf.cached_structs[i])
                    .or_else(|| cf.get_descr(obj))
                else {
                    continue;
                };
                let opcode = descr
                    .as_field_descr()
                    .map(|fd| OpCode::getfield_for_type(fd.field_type()))
                    .unwrap_or(OpCode::GetfieldGcI);
                let mut op = Op::with_descr(opcode, &[obj], descr.clone());
                op.pos = cached_val;
                sb.add_heap_op(op);
            }
        }

        for (&(_descr_idx, index), cai) in &self.cached_arrayitems {
            for i in 0..cai.cached_arrays.len() {
                let obj = cai.cached_arrays[i];
                let cached_val = cai.cached_values[i];
                if cached_val.is_none() || obj.is_none() {
                    continue;
                }
                let Some(descr) = cai.get_descr(obj) else {
                    continue;
                };
                let idx_ref = OpRef(index as u32);
                let opcode = descr
                    .as_array_descr()
                    .map(|array_descr| OpCode::getarrayitem_for_type(array_descr.item_type()))
                    .unwrap_or(OpCode::GetarrayitemGcI);
                let mut op = Op::with_descr(opcode, &[obj, idx_ref], descr.clone());
                op.pos = cached_val;
                sb.add_heap_op(op);
            }
        }
    }

    fn name(&self) -> &'static str {
        "heap"
    }

    fn emit_remaining_lazy_directly(&mut self, ctx: &mut OptContext) {
        let pending: Vec<(OpRef, Op)> = self
            .cached_fields
            .values_mut()
            .filter_map(|cf| cf.lazy_set.take())
            .collect();
        // Force any remaining virtual values before emit.
        for (_obj, op) in &pending {
            let orig_val = op.arg(1);
            if let Some(mut info) = ctx.get_ptr_info(orig_val).cloned() {
                if info.is_virtual() {
                    info.force_box(orig_val, ctx);
                }
            }
        }
        for (_obj, mut op) in pending {
            for arg in op.args.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
            let val = op.arg(1);
            if !val.is_none()
                && val.0 >= ctx.num_inputs() as u32
                && !ctx.is_constant(val)
                && !ctx
                    .new_operations
                    .iter()
                    .any(|o| o.pos == val && o.opcode.result_type() != Type::Void)
            {
                continue;
            }
            ctx.emit(op);
        }
        let pending_arr: Vec<(OpRef, Op)> = self
            .cached_arrayitems
            .values_mut()
            .filter_map(|cai| cai.lazy_set.take())
            .collect();
        for (_obj, mut op) in pending_arr {
            for arg in op.args.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
            ctx.emit(op);
        }
    }

    fn export_cached_fields(&self) -> Vec<(OpRef, u32, OpRef)> {
        let mut result = Vec::new();
        for (&field_idx, cf) in &self.cached_fields {
            // heap.py:830-831: skip if lazy_set active
            if cf.lazy_set.is_some() {
                continue;
            }
            for i in 0..cf.cached_structs.len() {
                let obj = cf.cached_structs[i];
                let val = cf.cached_values[i];
                if !val.is_none() && !obj.is_none() {
                    result.push((obj, field_idx, val));
                }
            }
        }
        for (&(obj, descr_idx), &val) in &self.immutable_cached_fields {
            if !val.is_none() {
                result.push((obj, descr_idx, val));
            }
        }
        result
    }

    /// heap.py:870-890: deserialize_optheap — restore cached field
    /// knowledge from the preamble into Phase 2's heap cache.
    fn import_cached_fields(&mut self, entries: &[(OpRef, u32, OpRef)]) {
        for &(obj, field_idx, val) in entries {
            if !obj.is_none() && !val.is_none() {
                self.cache_field(obj, field_idx, val, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use majit_ir::{
        CallDescr, Descr, DescrRef, EffectInfo, ExtraEffect, OopSpecIndex, Op, OpCode, OpRef,
        SimpleCallDescr,
    };

    use crate::optimizeopt::info::PtrInfo;
    use crate::optimizeopt::optimizer::Optimizer;
    use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

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

    /// Call descriptor with default EffectInfo (non-random, non-elidable).
    /// heapcache.py:362-370 parity: plain calls with known effectinfo
    /// use invalidate_unescaped instead of full reset.
    fn plain_call_descr(idx: u32) -> DescrRef {
        Arc::new(majit_ir::SimpleCallDescr::new(
            idx,
            vec![],
            majit_ir::Type::Void,
            0,
            EffectInfo {
                // Write all fields/arrays so invalidation is triggered
                write_descrs_fields: u64::MAX,
                write_descrs_arrays: u64::MAX,
                ..EffectInfo::default()
            },
        ))
    }

    /// Helper: assign sequential positions to ops.
    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
    }

    /// Run a single OptHeap pass over the given ops.
    ///
    /// Uses num_inputs=1024 so that high-numbered OpRef values used as
    /// input arguments in tests (e.g. OpRef(100), OpRef(500)) are treated
    /// as valid defined positions by the optimizer's undefined-ref filter.
    fn run_heap_opt(ops: &mut [Op]) -> Vec<Op> {
        assign_positions(ops);
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptHeap::new()));
        opt.optimize_with_constants_and_inputs(ops, &mut std::collections::HashMap::new(), 1024)
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

        // force_all_lazy_setfields emits the lazy SetfieldGc before Jump.
        // GetfieldGcI is eliminated (replaced by cached i1). SetfieldGc + Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::Jump);
    }

    #[test]
    fn test_imported_short_cached_fields_replays_into_heap() {
        use crate::optimizeopt::info::{PreambleOp, PtrInfo};
        let d = descr(55);
        let mut heap = OptHeap::new();
        let mut ctx = OptContext::with_num_inputs(4, 2);
        // RPython PreambleOp parity: store PreambleOp in PtrInfo
        ctx.set_ptr_info(OpRef(0), PtrInfo::instance(None, None));
        ctx.get_ptr_info_mut(OpRef(0)).unwrap().set_preamble_field(
            d.index(),
            PreambleOp {
                op: OpRef(100),
                resolved: OpRef(1),
                invented_name: false,
            },
        );

        let mut op = Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d);
        op.pos = OpRef(2);

        let result = heap.optimize_getfield(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(ctx.get_box_replacement(OpRef(2)), OpRef(1));
    }

    /// After consuming an imported short field, a cache invalidation followed
    /// by another getfield must emit the actual load (not reuse the stale
    /// preamble value).  This prevents null-pointer crashes when the
    /// preamble's cached value (e.g. a linked-list head) is no longer valid
    /// after a call/setfield that empties the container.
    #[test]
    fn test_imported_short_field_not_reused_after_invalidation() {
        let d_head = descr(10); // head field
        let d_size = descr(11); // size field (different)
        let mut heap = OptHeap::new();
        let mut ctx = OptContext::with_num_inputs(4, 4);

        // RPython PreambleOp parity: store PreambleOp in PtrInfo
        use crate::optimizeopt::info::{PreambleOp, PtrInfo};
        ctx.set_ptr_info(OpRef(0), PtrInfo::instance(None, None));
        ctx.get_ptr_info_mut(OpRef(0)).unwrap().set_preamble_field(
            d_head.index(),
            PreambleOp {
                op: OpRef(100),
                resolved: OpRef(1),
                invented_name: false,
            },
        );

        // First getfield on head: consumes the import, caches the value.
        let mut op1 = Op::with_descr(OpCode::GetfieldGcR, &[OpRef(0)], d_head.clone());
        op1.pos = OpRef(2);
        let result1 = heap.optimize_getfield(&op1, &mut ctx);
        assert!(matches!(result1, OptimizationResult::Remove));
        assert_eq!(ctx.get_box_replacement(OpRef(2)), OpRef(1));

        // A call invalidates all mutable field caches.
        heap.clean_caches();

        // Second getfield on head after invalidation: must NOT return the
        // stale preamble value.  The import was consumed, so it should emit.
        let mut op2 = Op::with_descr(OpCode::GetfieldGcR, &[OpRef(0)], d_head.clone());
        op2.pos = OpRef(3);
        let result2 = heap.optimize_getfield(&op2, &mut ctx);
        assert!(
            matches!(result2, OptimizationResult::Emit(_)),
            "getfield after invalidation must emit, not reuse stale import"
        );
    }

    #[test]
    fn test_getfield_does_not_deref_arbitrary_int_constant_base() {
        let d = immutable_descr(77);
        let mut heap = OptHeap::new();
        let mut ctx = OptContext::with_num_inputs(4, 1);
        ctx.make_constant(OpRef(0), majit_ir::Value::Int(1));

        let mut op = Op::with_descr(OpCode::GetfieldGcI, &[OpRef(0)], d);
        op.pos = OpRef(1);

        let result = heap.optimize_getfield(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Emit(_)));
        assert_eq!(ctx.get_box_replacement(OpRef(1)), OpRef(1));
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

        // First SetfieldGc is dead (overwritten). Second is emitted as lazy set before Jump.
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
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

        // force_all_lazy at call emits SetfieldGc + invalidates caches.
        // SetfieldGc + CALL + GETFIELD (re-emitted, cache was invalidated) + Jump.
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
        // setfield_gc(p1, i2, descr=d0)  <- possible_aliasing: forces first lazy_set
        // i3 = getfield_gc_i(p0, descr=d0)   <- cached from forced set (i1)
        // i4 = getfield_gc_i(p1, descr=d0)   <- cached from second lazy_set (i2)
        let d = descr(0);
        let mut ops = vec![
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(100), OpRef(101)], d.clone()),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(200), OpRef(201)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(100)], d.clone()),
            Op::with_descr(OpCode::GetfieldGcI, &[OpRef(200)], d.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        let result = run_heap_opt(&mut ops);

        // RPython CachedField per-descr with aliasing analysis:
        // - setfield(p1) forces lazy_set(p0) → emit SETFIELD(p0), put_back p0
        // - invalidate_for_write(p1) removes p0 (input args can alias)
        // - getfield(p0): lazy_set is p1, UNKNOWN_ALIAS → force lazy_set(p1)
        //   → emit SETFIELD(p1), put_back p1. p0 entry gone → cache miss → emit GETFIELD(p0)
        // - getfield(p1): entry p1=i2 from put_back → cache hit → remove
        // Result: SETFIELD(p0) + SETFIELD(p1) + GETFIELD(p0) + Jump.
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
        assert_eq!(result[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(result[3].opcode, OpCode::Jump);
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // force_all_lazy at Jump drops lazy setarrayitems. Only Jump remains.
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(opcodes, vec![OpCode::Jump]);
    }

    #[test]
    fn test_getarrayitem_postprocess_updates_ptr_info() {
        let d = descr(0);
        let idx = OpRef(50);
        let mut op = Op::with_descr(OpCode::GetarrayitemGcI, &[OpRef(100), idx], d.clone());
        op.pos = OpRef(200);

        let mut ctx = OptContext::new(256);
        ctx.make_constant(idx, majit_ir::Value::Int(3));
        ctx.set_ptr_info(OpRef(100), PtrInfo::virtual_array(d, 8, false));

        let mut pass = OptHeap::new();
        pass.setup();

        let result = pass.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Emit(_)));
        assert_eq!(
            ctx.get_ptr_info(OpRef(100))
                .and_then(|info| info.getitem(3)),
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
        ctx.set_ptr_info(OpRef(100), PtrInfo::virtual_array(d, 8, false));

        let mut pass = OptHeap::new();
        pass.setup();

        let result = pass.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_ptr_info(OpRef(100))
                .and_then(|info| info.getitem(3)),
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

        // SetfieldGc(d0) emitted as lazy set + GETFIELD(d1, different descriptor) + Jump.
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // force_all_lazy at Jump drops lazy setarrayitems. Only Jump remains.
        let result_opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(result_opcodes, vec![OpCode::Jump]);
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

        // Both GETFIELDs eliminated (cached). Both lazy SetfieldGc emitted before Jump.
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
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

    #[test]
    fn test_short_preamble_ref_field_preserves_getfield_opcode() {
        let descr = majit_ir::make_field_descr(55, 8, majit_ir::Type::Ref, false);
        let mut pass = OptHeap::new();
        let key = (OpRef(100), descr.index());
        pass.cache_field(OpRef(100), descr.index(), OpRef(101), Some(&descr));

        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef(100),
            OpRef(101),
        ]);
        // Register input args so produce_arg can resolve them.
        sb.add_short_input_arg(OpRef(100));
        sb.add_short_input_arg(OpRef(101));
        let ctx = crate::optimizeopt::OptContext::new(256);
        pass.produce_potential_short_preamble_ops(&mut sb, &ctx);
        let produced = sb.produced_ops();

        // Filter to heap-produced ops (exclude SameAsI from add_short_input_arg).
        let heap_ops: Vec<_> = produced
            .iter()
            .filter(|(_, p)| p.preamble_op.opcode != OpCode::SameAsI)
            .collect();
        assert_eq!(heap_ops.len(), 1);
        assert_eq!(heap_ops[0].1.preamble_op.opcode, OpCode::GetfieldGcR);
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
            Op::with_descr(OpCode::CallN, &[OpRef(200)], plain_call_descr(100)),
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
            Op::with_descr(OpCode::CallN, &[OpRef(0)], plain_call_descr(100)), // pass p0 to call
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
            Op::with_descr(OpCode::CallN, &[OpRef(1)], plain_call_descr(100)), // call(p1) (p1 escapes)
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
            Op::with_descr(OpCode::CallN, &[OpRef(200)], plain_call_descr(100)),
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
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
            Op::with_descr(OpCode::CallN, &[OpRef(200)], plain_call_descr(100)),
            Op::with_descr(OpCode::CallN, &[OpRef(201)], plain_call_descr(101)),
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
            nonnull_count, 1,
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
        assert_eq!(nonnull_count, 2, "second guard_nonnull should be removed");
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
            nonnull_count, 1,
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
            nonnull_count, 1,
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
            nonnull_count, 1,
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
            nonnull_count, 1,
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
            nonnull_count, 1,
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
            nonnull_count, 1,
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
    fn test_call_may_force_uses_effectinfo_to_keep_unaffected_cached_fields() {
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
    fn test_call_may_force_uses_effectinfo_to_invalidate_written_cached_fields() {
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
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
    fn test_arraycopy_preserves_unrelated_cached_fields() {
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
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

        // force_all_lazy_setfields at GcLoadI emits lazy SetfieldGc. SetfieldGc + GcLoadI + Jump.
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
            nonnull_count, 1,
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
                }
            }
        }

        // force_all_lazy at Jump drops lazy setarrayitems. Only Jump remains.
        let opcodes: Vec<_> = ctx.new_operations.iter().map(|o| o.opcode).collect();
        assert_eq!(
            opcodes,
            vec![OpCode::Jump],
            "byte-array getitem should be cached after setitem; lazy set dropped at Jump"
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
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
                *arg = ctx.get_box_replacement(*arg);
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
                OptimizationResult::InvalidLoop => {
                    panic!("unexpected InvalidLoop in test");
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
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );
        let len_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::ArraylenGc)
            .count();
        assert_eq!(len_count, 1, "duplicate ARRAYLEN_GC should be cached");
    }
}
