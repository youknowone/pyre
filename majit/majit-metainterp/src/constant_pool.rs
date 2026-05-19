//! Constant pool for trace recording with GC root tracking.
//!
//! RPython manages constants implicitly in Trace — ConstPtr boxes are
//! GC-managed objects, so GC can update them when objects move.
//!
//! majit stores Ref constants as raw i64 entries in a VecAssoc,
//! invisible to GC. To achieve RPython parity, Ref constants are rooted
//! on the shadow stack (gcreftracer.py:GCREFTRACER parity). GC's
//! walk_roots updates shadow stack entries in place; refresh_from_gc
//! copies updated values back to the VecAssoc before consumption.
//!
//! Internal storage is `VecAssoc<u32, Value>`.  Production callers and
//! `Backend::set_constants_pool` consume the typed `VecAssoc<u32,
//! Const>` shape.  The remaining raw-`i64` egress paths are:
//!
//!   * `into_inner` / `raw_bits` — raw `i64` readers kept for
//!     integer-typed regalloc consumers (offsets, sizes, scales) in
//!     `dynasm/regalloc.rs` and parity-test helpers. `raw_bits`
//!     round-trips through `as_raw_i64()` without information loss for
//!     integer constants.
//!   * `into_inner_typed` / `snapshot` / `get_value` — typed `Value`
//!     egress, used by callers that need to distinguish
//!     `Value::Int`/`Float`/`Ref`.

use majit_gc::shadow_stack;
use majit_ir::{GcRef, OpRef, Type, Value, VecAssoc};

/// Encode a typed `Value` to the raw `i64` shape used by the legacy
/// backend boundary.
fn value_to_raw_bits(value: &Value) -> i64 {
    match value {
        Value::Int(v) => *v,
        Value::Ref(r) => r.0 as i64,
        Value::Float(f) => f.to_bits() as i64,
        // history.py:220/261/307 — only ConstInt/ConstFloat/ConstPtr
        // exist upstream; there is no `ConstVoid` class. A Void constant
        // reaching the backend boundary indicates a bookkeeping bug;
        // surface it instead of silently lowering to a zero Int payload.
        Value::Void => panic!(
            "value_to_raw_bits: Value::Void has no constant lowering (no ConstVoid upstream)"
        ),
    }
}

/// Constant pool for trace recording.
///
/// Manages the mapping from constant-namespace OpRef to typed `Value`
/// payloads. Each `get_or_insert{,_typed}` call mints a fresh OpRef per
/// `history.py:220/261/307` ConstInt/ConstFloat/ConstPtr `__init__`
/// fresh-allocation; value-equality between two constant OpRefs uses
/// `same_constant` (history.py:204) — not OpRef identity. Resume
/// serialization dedup (`resume.py:148-181 large_ints`/`refs`) is a
/// separate concern that lives in `resume.rs`.
///
/// Storage shape: `VecAssoc<u32, Value>` mirrors RPython where each
/// `ConstInt/ConstFloat/ConstPtr` Box carries `.type` intrinsically
/// (history.py:220/261/307). Type rides on the `Value` variant tag,
/// eliminating the lockstep risk between value and type maps.
///
/// gcreftracer.py parity: Ref-typed constants are pushed onto the GC
/// shadow stack so that GC can trace and update them if objects move.
/// On consumption (into_inner / snapshot), the VecAssoc is refreshed
/// from the shadow stack to pick up any GC-updated pointers.
pub struct ConstantPool {
    /// Keyed by OpRef.0 (tagged constant value, i.e. index | CONST_BIT).
    /// `Value` carries type intrinsically (history.py:220 box.type).
    constants: VecAssoc<u32, Value>,
    /// Zero-based counter for allocating new constant indices.
    next_const_idx: u32,
    /// gcreftracer.py parity: (OpRef key, shadow stack index) for each
    /// rooted Ref constant. walk_roots updates shadow stack entries;
    /// refresh_from_gc copies values back to `constants`.
    rooted_refs: Vec<(u32, usize)>,
    /// Shadow stack depth at pool creation. release_roots pops to here.
    shadow_stack_base: usize,
    /// opencoder.py:484 `self._bigints = []` parity. Dense per-pool
    /// storage of non-small-int constant values, in mint order. Slot
    /// `bigints[k] = (opref_raw, value)` where `opref_raw =
    /// opref.raw()` for the k-th ConstInt minted. Populated alongside
    /// `constants` VecAssoc as a structural mirror; consumers continue
    /// to use the VecAssoc until subsequent slices migrate readers.
    bigints: Vec<(u32, i64)>,
    /// opencoder.py:486 `self._floats = []` parity. Dense per-pool
    /// storage of ConstFloat constants in mint order. Slot
    /// `floats[k] = (opref_raw, bit_pattern)` for the k-th
    /// ConstFloat minted.
    floats: Vec<(u32, i64)>,
    /// opencoder.py:482 `self._refs = [lltype.nullptr(llmemory.GCREF.TO)]`
    /// parity. Dense per-pool storage of ConstPtr constants in mint
    /// order. Slot `refs[k] = (opref_raw, gc_addr)` for the k-th
    /// ConstPtr minted. RPython prepends a null sentinel at index 0;
    /// pyre tracks every minted ConstPtr including null values, so
    /// the sentinel slot is not pre-seeded here.
    refs: Vec<(u32, i64)>,
}

impl ConstantPool {
    pub fn new() -> Self {
        ConstantPool {
            constants: VecAssoc::new(),
            next_const_idx: 0,
            rooted_refs: Vec::new(),
            shadow_stack_base: shadow_stack::depth(),
            bigints: Vec::new(),
            floats: Vec::new(),
            refs: Vec::new(),
        }
    }

    /// Mint a fresh ConstInt OpRef for `value`.
    ///
    /// `history.py:220 ConstInt.__init__` allocates a new Box on every
    /// call; equality between two ConstInt boxes for the same value is
    /// `Const.same_constant` (history.py:204/244), not Box identity.
    /// Per-value dedup belongs to the resume tagging memo
    /// (`resume.py:148-172 ResumeDataLoopMemo.large_ints`), which lives
    /// in `resume.rs` and is unaffected by this method.
    pub fn get_or_insert(&mut self, value: i64) -> OpRef {
        let opref = OpRef::const_int(self.next_const_idx);
        self.next_const_idx += 1;
        self.constants.insert(opref.raw(), Value::Int(value));
        // opencoder.py:621 `self._bigints.append(value)` parity:
        // structural mirror of the VecAssoc write into the dense
        // per-pool Vec.
        self.bigints.push((opref.raw(), value));
        opref
    }

    /// Mint a fresh typed constant OpRef.
    ///
    /// `history.py:220/261/307 ConstInt/ConstFloat/ConstPtr.__init__`
    /// allocate fresh Boxes per call; equality is `Const.same_constant`
    /// (history.py:204). The Ref arm additionally roots the value on
    /// the shadow stack so the GC can update it (gcreftracer.py).
    /// `rooted_refs` (alongside `refresh_from_gc`) is the single source
    /// of truth for which slots track a Ref constant; the post-GC
    /// address is written back to `constants[opref_key]` on each refresh.
    pub fn get_or_insert_typed(&mut self, value: i64, tp: Type) -> OpRef {
        match tp {
            Type::Void => panic!("Void constants are not supported"),
            Type::Float => {
                let opref = OpRef::const_float(self.next_const_idx);
                self.next_const_idx += 1;
                self.constants
                    .insert(opref.raw(), Value::Float(f64::from_bits(value as u64)));
                // opencoder.py:627 `self._floats.append(box.getfloatstorage())`
                // parity.
                self.floats.push((opref.raw(), value));
                opref
            }
            Type::Int => {
                let opref = OpRef::const_int(self.next_const_idx);
                self.next_const_idx += 1;
                self.constants.insert(opref.raw(), Value::Int(value));
                // opencoder.py:621 `self._bigints.append(value)` parity.
                self.bigints.push((opref.raw(), value));
                opref
            }
            Type::Ref => {
                let opref = OpRef::const_ptr(self.next_const_idx);
                self.next_const_idx += 1;
                self.constants
                    .insert(opref.raw(), Value::Ref(GcRef(value as usize)));
                // opencoder.py:598 `self._refs.append(addr)` parity.
                self.refs.push((opref.raw(), value));
                // gcreftracer.py: non-null Ref constants must be rooted
                // on the shadow stack so the GC can update them when
                // objects move. One root per ConstPtr mint mirrors
                // upstream's per-Box rooting via the ConstPtr.value
                // GCREF reachability.
                if value != 0 {
                    let ss_idx = shadow_stack::push(GcRef(value as usize));
                    self.rooted_refs.push((opref.raw(), ss_idx));
                }
                opref
            }
        }
    }

    /// opencoder.py:484 `_bigints` accessor. Returns the dense per-pool
    /// storage of `ConstInt` constants in mint order. Each entry is
    /// `(opref_raw, value)` where `opref_raw = const_int_opref.raw()`.
    pub fn bigints(&self) -> &[(u32, i64)] {
        &self.bigints
    }

    /// opencoder.py:486 `_floats` accessor. Returns the dense per-pool
    /// storage of `ConstFloat` constants in mint order. Each entry is
    /// `(opref_raw, bit_pattern)` where the bit pattern is the i64
    /// reinterpret of the `f64` value (`history.py:265
    /// ConstFloat.getfloatstorage`).
    pub fn floats(&self) -> &[(u32, i64)] {
        &self.floats
    }

    /// opencoder.py:482 `_refs` accessor. Returns the dense per-pool
    /// storage of `ConstPtr` constants in mint order. Each entry is
    /// `(opref_raw, gc_address)`. RPython prepends a null sentinel at
    /// index 0; pyre tracks every minted ConstPtr including null
    /// values so the slot 0 is the first minted ConstPtr (not a
    /// pre-seeded null).
    pub fn refs(&self) -> &[(u32, i64)] {
        &self.refs
    }

    /// Get the type of a constant.
    ///
    /// `history.py:220/261/307` — ConstInt/ConstFloat/ConstPtr `.type` is
    /// pinned at construction. The typed `OpRef` variant tag carries the
    /// same information, so `opref.ty()` is the sole authoritative
    /// source. Returns `None` only for `OpRef::None`.
    pub fn constant_type(&self, opref: OpRef) -> Option<Type> {
        opref.ty()
    }

    /// pyjitpl.py:3572 executor.constant_from_op(a) parity:
    /// return the typed Value for a constant OpRef, or None if
    /// the OpRef is not a known constant.
    ///
    /// In RPython every Const subclass guarantees its payload type
    /// matches its class identity (`history.py:220 ConstInt.type = INT`,
    /// `:261 ConstFloat.type = FLOAT`, `:307 ConstPtr.type = REF`). Mirror
    /// that contract by checking the OpRef variant tag against the stored
    /// `Value` variant: a mismatch indicates two pools were crossed (a
    /// caller bug) and would otherwise return a Value carrying the wrong
    /// type for the OpRef's class.
    pub fn get_value(&self, opref: OpRef) -> Option<Value> {
        let stored = self.constants.get(&opref.raw()).cloned()?;
        debug_assert!(
            opref.ty().map(|t| t == stored.get_type()).unwrap_or(true),
            "ConstantPool::get_value: OpRef variant {:?} (type {:?}) does not match \
             stored Value type {:?} — history.py:220/261/307 disjoint subclass invariant",
            opref,
            opref.ty(),
            stored.get_type(),
        );
        Some(stored)
    }

    /// history.py:204 / :244 `Const.same_constant` — true when two
    /// constant OpRefs name the same `(type, value)` pair. Each
    /// `get_or_insert{,_typed}` call mints a fresh OpRef (per
    /// history.py:220/261/307 fresh-alloc), so two ConstInt/ConstFloat/
    /// ConstPtr OpRefs for the same value have distinct raw indices
    /// and compare unequal under `OpRef::eq`. `same_constant` is the
    /// upstream-orthogonal value-equality path; callers comparing
    /// Const boxes for "is this the same value" must use this method
    /// directly.
    ///
    /// Returns `false` for any non-constant OpRef. RPython's
    /// `Const.same_constant` is defined only on the `Const` hierarchy
    /// (history.py:204-208 — base `raise NotImplementedError`); calling
    /// it on a non-constant Box is a type error upstream. We mirror
    /// that contract by short-circuiting non-constants to `false` even
    /// when `a == b` (e.g. `OpRef::NONE == OpRef::NONE`).
    pub fn same_constant(&self, a: OpRef, b: OpRef) -> bool {
        if !a.is_constant() || !b.is_constant() {
            return false;
        }
        if a == b {
            return true;
        }
        // history.py:251 ConstInt.same_constant / :292 ConstFloat /
        // :338 ConstPtr — each subclass's `same_constant` short-circuits
        // to `false` when the operand is not an instance of the same
        // subclass. ConstInt/ConstFloat/ConstPtr are disjoint Const
        // subclasses (history.py:220/261/307). Read each operand's
        // class identity from the typed OpRef variant tag.
        if a.ty() != b.ty() {
            return false;
        }
        let av = self.constants.get(&a.raw());
        let bv = self.constants.get(&b.raw());
        match (av, bv) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        }
    }

    /// Update constants map from shadow stack — GC may have moved Ref objects.
    /// gcreftracer.py:gcrefs_trace parity.
    ///
    /// `rooted_refs` is populated only by `get_or_insert_typed` under
    /// `tp == Type::Ref`, so every entry here is Ref-typed by
    /// construction (`history.py:307 ConstPtr.type = REF`).
    pub(crate) fn refresh_from_gc(&mut self) {
        for &(opref_key, ss_idx) in &self.rooted_refs {
            let current = shadow_stack::get(ss_idx);
            self.constants.insert(opref_key, Value::Ref(current));
            // Mirror the GC update into the dense `refs` pool so it
            // stays consistent with `constants`. The entry was appended
            // by `get_or_insert_typed(_, Ref)`; its raw key is unique
            // per mint, so the first match is the right slot.
            if let Some(slot) = self.refs.iter_mut().find(|(k, _)| *k == opref_key) {
                slot.1 = current.0 as i64;
            }
        }
    }

    /// Release shadow stack roots.
    /// gcreftracer.py parity: release GC roots for this pool's constants.
    /// XXX majit-only: in RPython, ConstantPool consumption is strictly
    /// LIFO so pop_to always succeeds. In majit, ExportedState may pop
    /// the shadow stack between this pool's creation and release. Until
    /// the LIFO ordering is enforced structurally, guard against this.
    fn release_roots(&mut self) {
        if !self.rooted_refs.is_empty() {
            let current = shadow_stack::depth();
            if current >= self.shadow_stack_base {
                shadow_stack::pop_to(self.shadow_stack_base);
            }
            self.rooted_refs.clear();
        }
    }

    /// Consume the pool and return the raw-bits map.
    ///
    /// The raw-bits view is preserved for backend / parity-print
    /// consumers that still operate on `i64` payloads. Each
    /// `Value` is lowered via `value_to_raw_bits`.
    pub fn into_inner(mut self) -> VecAssoc<u32, i64> {
        self.refresh_from_gc();
        let constants = std::mem::take(&mut self.constants);
        self.release_roots();
        let mut out: VecAssoc<u32, i64> = VecAssoc::new();
        for (k, v) in constants.into_iter() {
            out.insert(k, value_to_raw_bits(&v));
        }
        out
    }

    /// Consume the pool, returning the canonical typed `VecAssoc<u32,
    /// Value>` — matching RPython's `Const(value)` Box model where
    /// `ConstInt/ConstFloat/ConstPtr` (history.py:220/261/307) carry
    /// their value as a typed instance attribute.
    pub fn into_inner_typed(mut self) -> VecAssoc<u32, Value> {
        self.refresh_from_gc();
        let constants = std::mem::take(&mut self.constants);
        self.release_roots();
        constants
    }

    /// Get a mutable reference to the inner constants pool (typed).
    pub fn as_mut(&mut self) -> &mut VecAssoc<u32, Value> {
        &mut self.constants
    }

    /// Ensure `next_const_idx` is beyond the given const-namespace key.
    /// Used by bridge injection: constants with pre-assigned indices must
    /// not be overwritten by subsequent `get_or_insert` allocations.
    pub fn reserve_index_past(&mut self, opref_key: u32) {
        let raw_idx = opref_key & !(1 << 31);
        self.next_const_idx = self.next_const_idx.max(raw_idx + 1);
    }

    /// Get a shared reference to the inner constants pool (typed).
    pub fn as_ref(&self) -> &VecAssoc<u32, Value> {
        &self.constants
    }

    /// Look up a constant's raw `i64` bits (legacy backend boundary).
    /// Returns `None` if `opref` is not in the pool.
    pub fn raw_bits(&self, opref: OpRef) -> Option<i64> {
        self.constants.get(&opref.raw()).map(value_to_raw_bits)
    }

    /// Clone the constants pool without consuming it, returning the
    /// typed `Value` shape. Refreshes from GC first to pick up moved
    /// Ref pointers.
    pub fn snapshot(&mut self) -> VecAssoc<u32, Value> {
        self.refresh_from_gc();
        self.constants.clone()
    }
}

impl Default for ConstantPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ConstantPool {
    fn drop(&mut self) {
        self.release_roots();
    }
}

impl majit_trace::heapcache::SameConstantOracle for ConstantPool {
    fn same_constant(&self, a: OpRef, b: OpRef) -> bool {
        ConstantPool::same_constant(self, a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_constant_distinct_mints_within_pool_satisfy_predicate() {
        // history.py:204 same_constant on a single ConstantPool: two
        // get_or_insert calls for the same value mint distinct ConstInt
        // OpRefs (history.py:220 fresh-alloc) that satisfy same_constant.
        let mut pool = ConstantPool::new();
        let a = pool.get_or_insert(42);
        let b = pool.get_or_insert(42);
        assert_ne!(a, b, "fresh mints must produce distinct OpRefs");
        assert!(pool.same_constant(a, b));
    }

    #[test]
    fn same_constant_value_aware_across_independent_inserts() {
        // history.py:244 ConstInt.same_constant: two ConstInt instances
        // with the same value compare equal even though they're distinct
        // Box objects in RPython. This helper extends that semantics to
        // pyre OpRefs that may have been minted at different ConstantPool
        // indices (cross-pool deserialisation).
        let mut pool = ConstantPool::new();
        let a = pool.get_or_insert(42);
        // Manually insert a second slot with the same value, bypassing
        // the dedup path (simulates cross-pool composition).
        let b_idx = pool.next_const_idx;
        pool.next_const_idx += 1;
        let b = OpRef::const_int(b_idx);
        pool.constants.insert(b.raw(), Value::Int(42));
        assert_ne!(a, b, "different idx slots must be != under variant Eq");
        assert!(
            pool.same_constant(a, b),
            "same_constant must be value-aware",
        );
    }

    #[test]
    fn same_constant_disjoint_subclasses_are_unequal() {
        // history.py:220 / :261 / :307: ConstInt and ConstPtr are
        // disjoint Const subclasses; same_constant returns false across
        // type boundaries even when the underlying value matches.
        let mut pool = ConstantPool::new();
        let i = pool.get_or_insert_typed(0, Type::Int);
        let p = pool.get_or_insert_typed(0, Type::Ref);
        assert_ne!(i, p);
        assert!(!pool.same_constant(i, p));
    }

    #[test]
    fn same_constant_rejects_non_constants() {
        let pool = ConstantPool::new();
        let inputarg = OpRef::input_arg_int(3);
        let op = OpRef::int_op(7);
        assert!(!pool.same_constant(inputarg, op));
        assert!(!pool.same_constant(inputarg, inputarg.with_raw(99)));
    }

    #[test]
    fn same_constant_handles_none() {
        // history.py:204-208 — `Const.same_constant` is defined only on
        // the Const hierarchy. `OpRef::NONE` is not a constant, so the
        // helper must return false even when both operands compare
        // equal under variant-aware Eq.
        let pool = ConstantPool::new();
        assert!(!pool.same_constant(OpRef::NONE, OpRef::NONE));
        assert!(!pool.same_constant(OpRef::NONE, OpRef::const_int(0)));
    }

    /// `get_or_insert(0)` and `get_or_insert_typed(0, Ref)` must NOT alias —
    /// `history.py:220/307` ConstInt(0) and ConstPtr(NULL) are different
    /// classes. Distinct OpRef discriminates the two paths even when the
    /// raw value is identical.
    #[test]
    fn int_ref_zero_does_not_alias() {
        let mut pool = ConstantPool::new();
        let i_zero = pool.get_or_insert(0);
        let r_null = pool.get_or_insert_typed(0, Type::Ref);
        assert_ne!(i_zero, r_null);
        assert_eq!(pool.constant_type(i_zero), Some(Type::Int));
        assert_eq!(pool.constant_type(r_null), Some(Type::Ref));
    }

    /// `history.py:220 ConstInt.__init__` is fresh-alloc per call;
    /// equality between two ConstInt boxes for the same value uses
    /// `Const.same_constant` (history.py:204/244). Pin this contract so
    /// callers that rely on value-equality across two mint operations
    /// migrate to `ConstantPool::same_constant` rather than `OpRef::eq`.
    #[test]
    fn int_value_equality_uses_same_constant_not_eq() {
        let mut pool = ConstantPool::new();
        let a = pool.get_or_insert(42);
        let b = pool.get_or_insert(42);
        assert!(
            pool.same_constant(a, b),
            "two get_or_insert(42) calls must satisfy same_constant"
        );
    }

    /// `history.py:307 ConstPtr.__init__` is fresh-alloc per call;
    /// equality between two ConstPtr boxes for the same address uses
    /// `Const.same_constant`. Mirror the Int pin above for the Ref arm.
    #[test]
    fn ref_value_equality_uses_same_constant_not_eq() {
        let mut pool = ConstantPool::new();
        let a = pool.get_or_insert_typed(0xdead_beef, Type::Ref);
        let b = pool.get_or_insert_typed(0xdead_beef, Type::Ref);
        assert!(
            pool.same_constant(a, b),
            "two get_or_insert_typed(_, Ref) calls must satisfy same_constant"
        );
    }

    /// opencoder.py:482/484/486 `_refs`/`_bigints`/`_floats` parity:
    /// the dense per-pool Vecs grow in mint order, independently of
    /// each other. A mixed mint sequence should leave each Vec dense
    /// (no gaps) with one entry per same-kind mint.
    #[test]
    fn dense_pools_grow_independently_per_kind() {
        let mut pool = ConstantPool::new();
        let i0 = pool.get_or_insert(7);
        let f0 = pool.get_or_insert_typed(0x4000_0000_0000_0000, Type::Float);
        let i1 = pool.get_or_insert_typed(13, Type::Int);
        let r0 = pool.get_or_insert_typed(0, Type::Ref);
        let r1 = pool.get_or_insert_typed(0xdead_beef, Type::Ref);
        assert_eq!(pool.bigints().len(), 2);
        assert_eq!(pool.bigints()[0], (i0.raw(), 7));
        assert_eq!(pool.bigints()[1], (i1.raw(), 13));
        assert_eq!(pool.floats().len(), 1);
        assert_eq!(pool.floats()[0], (f0.raw(), 0x4000_0000_0000_0000));
        assert_eq!(pool.refs().len(), 2);
        assert_eq!(pool.refs()[0], (r0.raw(), 0));
        assert_eq!(pool.refs()[1], (r1.raw(), 0xdead_beef));
    }
}
