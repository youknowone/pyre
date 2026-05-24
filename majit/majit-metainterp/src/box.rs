//! BoxPool — pyre-only side-table mapping flat `OpRef` indices to PyPy-style
//! `Box` identities.  Box/BoxKind/BoxRef/Forwarded themselves now live in
//! `majit_ir::box_ref`.

use majit_ir::OpRef;

pub use majit_ir::box_ref::{BoxRef, Forwarded, PtrInfoBorrowMut};

/// Encapsulated `BoxRef` storage for `OptContext` (Codex plan step 1).
///
/// Indexed by `OpRef` raw position. `BoxRef._forwarded` is the
/// authoritative PyPy-style storage; `BoxPool` only maps pyre's flat
/// `OpRef` indices to those Box identities.
///
/// Sparse via `Vec<Option<BoxRef>>` so positions skipped during pool
/// extension (e.g. constant-namespace claims via `allocate_next_pos_raw`)
/// stay `None` instead of producing Void filler boxes. PyPy's box-per-Box
/// model has no filler analogue — every Box is constructed by
/// `ResOperation()` or `InputArg()` at its real type.
#[derive(Clone, Debug, Default)]
pub struct BoxPool {
    inner: Vec<Option<BoxRef>>,
}

impl BoxPool {
    /// Defensive bound on `idx`/`capacity` passed to `set`/`with_capacity`/
    /// `from_slots`.  A leaked constant-namespace OpRef (raw `>= CONST_BIT
    /// = 1 << 31`) or `TempVar` sentinel (`raw >= 0xFFFF_0000`) reaching
    /// these entry points would otherwise resize the underlying
    /// `Vec<Option<BoxRef>>` to multi-GiB.  Real traces top out at
    /// `O(10^5)` ops; `10_000_000` is ~3 orders of magnitude of headroom
    /// while still much smaller than `CONST_BIT`, so any namespace bleed
    /// panics immediately instead of OOMing.
    const SANE_IDX_BOUND: usize = 10_000_000;

    pub fn new() -> Self {
        Self::default()
    }

    /// Preallocate the slot table with `capacity` entries reserved.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(
            capacity < Self::SANE_IDX_BOUND,
            "BoxPool::with_capacity({capacity}) exceeds SANE_IDX_BOUND ({}); \
             caller likely passed a raw OpRef payload with CONST_BIT/sentinel set",
            Self::SANE_IDX_BOUND
        );
        Self {
            inner: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Read `box_pool[opref]` — returns `Some(&BoxRef)` only for a
    /// materialized slot in the body/inputarg namespaces. Out-of-bounds
    /// and tombstoned slots return `None`. OpRefs in Const/TempVar/None
    /// namespaces — which never have a `box_pool` slot by construction
    /// (constants live in `const_pool`, TempVars are regalloc-only) —
    /// also return `None`, so the caller chain falls through to whatever
    /// fallback was already in place for the absent-slot path.
    pub fn get(&self, opref: OpRef) -> Option<&BoxRef> {
        let idx = match opref {
            OpRef::IntOp(p)
            | OpRef::FloatOp(p)
            | OpRef::RefOp(p)
            | OpRef::VoidOp(p)
            | OpRef::InputArgInt(p)
            | OpRef::InputArgFloat(p)
            | OpRef::InputArgRef(p) => p as usize,
            OpRef::ConstInt(_)
            | OpRef::ConstFloat(_)
            | OpRef::ConstPtr(_)
            | OpRef::TempVar(_)
            | OpRef::None => return None,
        };
        self.inner.get(idx)?.as_ref()
    }

    /// Positional accessor by raw recording slot — used when the
    /// caller carries an integer position rather than an `OpRef`
    /// (recorder tests; GC snapshot rewrite paths that store
    /// `iter_indexed` slot indices; `allocate_next_pos_raw`'s
    /// constant-slot probe). Production callers with an `OpRef` in
    /// scope use `get` instead.
    pub fn get_at_position(&self, position: usize) -> Option<&BoxRef> {
        self.inner.get(position)?.as_ref()
    }

    /// `box_pool[opref] = Some(value)`; extends with `None` padding to
    /// reach the slot. Returns a clone of the installed BoxRef.
    ///
    /// Takes an `OpRef` rather than raw `usize` so the namespace
    /// invariants are enforced at the type level: only body and
    /// InputArg variants reach the underlying `Vec`. A constant or
    /// `TempVar` reaching this entry point would otherwise resize the
    /// Vec to multi-GiB (raw payload >= `CONST_BIT = 1 << 31` or
    /// `SENTINEL_BASE = 0xFFFF_0000`).
    pub fn set(&mut self, opref: OpRef, value: BoxRef) -> BoxRef {
        let idx = match opref {
            OpRef::IntOp(p)
            | OpRef::FloatOp(p)
            | OpRef::RefOp(p)
            | OpRef::VoidOp(p)
            | OpRef::InputArgInt(p)
            | OpRef::InputArgFloat(p)
            | OpRef::InputArgRef(p) => p as usize,
            OpRef::ConstInt(_) | OpRef::ConstFloat(_) | OpRef::ConstPtr(_) => panic!(
                "BoxPool::set rejects constant OpRefs ({opref:?}); \
                 constants live in `const_pool`, not the box pool"
            ),
            OpRef::TempVar(_) => panic!(
                "BoxPool::set rejects TempVar OpRefs ({opref:?}); \
                 TempVars are regalloc-only and have no Box identity"
            ),
            OpRef::None => panic!("BoxPool::set rejects OpRef::None"),
        };
        assert!(
            idx < Self::SANE_IDX_BOUND,
            "BoxPool::set({opref:?}, idx={idx}) exceeds SANE_IDX_BOUND ({})",
            Self::SANE_IDX_BOUND
        );
        if idx >= self.inner.len() {
            self.inner.resize(idx + 1, None);
        }
        self.inner[idx] = Some(value.clone());
        value
    }

    /// Iterate over `(idx, &BoxRef)` for every materialized slot,
    /// skipping `None` holes.
    pub fn iter_indexed(&self) -> impl Iterator<Item = (usize, &BoxRef)> {
        self.inner
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.as_ref().map(|b| (i, b)))
    }

    /// Iterate over `&BoxRef` for every materialized slot, skipping
    /// `None` holes. Use when caller does not need the index.
    pub fn iter(&self) -> impl Iterator<Item = &BoxRef> {
        self.inner.iter().filter_map(|b| b.as_ref())
    }

    /// Sequentially append a fully-typed BoxRef. Used by recorder /
    /// history reconstruction where positions are dense (every op
    /// gets a Box).
    pub fn push(&mut self, value: BoxRef) {
        self.inner.push(Some(value));
    }

    /// Slice 8.C: bind ResOp boxes to their matching `OpRc` so subsequent
    /// `BoxRef::set_forwarded_*` calls dual-write through to `Op.forwarded`.
    /// `num_inputargs` is the count of InputArg boxes at the head of the
    /// pool — ResOp boxes occupy `box_pool[num_inputargs..]` and correspond
    /// 1:1 to `ops[i - num_inputargs]`.
    pub fn bind_ops(&self, num_inputargs: usize, ops: &[majit_ir::OpRc]) {
        for (pool_idx, box_opt) in self.inner.iter().enumerate().skip(num_inputargs) {
            if let Some(boxref) = box_opt {
                let op_idx = pool_idx - num_inputargs;
                if let Some(op) = ops.get(op_idx) {
                    if boxref.is_resop() {
                        boxref.bind_op(op);
                    }
                }
            }
        }
    }

    /// Slice 8.D InputArg counterpart of `bind_ops`. Binds the head
    /// `inputargs.len()` slots of the pool to their matching `InputArgRc`
    /// so `BoxRef::set_forwarded_*` routes through `inputarg.forwarded`
    /// (`resoperation.py:700 AbstractInputArg._forwarded`). Index `i`
    /// in `inputargs` corresponds to `box_pool[i]`.
    pub fn bind_inputargs(&self, inputargs: &[majit_ir::InputArgRc]) {
        for (i, ia) in inputargs.iter().enumerate() {
            if let Some(Some(boxref)) = self.inner.get(i) {
                if boxref.is_inputarg() {
                    boxref.bind_inputarg(ia);
                }
            }
        }
    }

    /// Drop trailing entries until `len() <= new_len`. Mirrors PyPy
    /// recorder savepoint rollback (`recorder.py savepoint.restore`).
    pub fn truncate(&mut self, new_len: usize) {
        self.inner.truncate(new_len);
    }

    /// Grow the slot table with `None` entries until `len() >=
    /// new_len`. No-op when already at or past `new_len`. Mirrors
    /// `BoxPool::set` upper-end grow but without planting any BoxRef —
    /// used by bridge / phase-2 setup that reserves positions before a
    /// producer materialises them via `ensure_box`.
    pub fn pad_none_to(&mut self, new_len: usize) {
        assert!(
            new_len < Self::SANE_IDX_BOUND,
            "BoxPool::pad_none_to({new_len}) exceeds SANE_IDX_BOUND ({})",
            Self::SANE_IDX_BOUND
        );
        if new_len > self.inner.len() {
            self.inner.resize(new_len, None);
        }
    }

    /// Build from a `Vec<Option<BoxRef>>` snapshot table.
    pub fn from_slots(slots: Vec<Option<BoxRef>>) -> Self {
        assert!(
            slots.len() < Self::SANE_IDX_BOUND,
            "BoxPool::from_slots(len={}) exceeds SANE_IDX_BOUND ({}); \
             slot table likely indexed by raw OpRef payload with CONST_BIT/sentinel set",
            slots.len(),
            Self::SANE_IDX_BOUND
        );
        Self { inner: slots }
    }
}

#[cfg(test)]
impl From<Vec<BoxRef>> for BoxPool {
    fn from(inner: Vec<BoxRef>) -> Self {
        let mut slots = Vec::with_capacity(inner.len());
        slots.extend(inner.into_iter().map(Some));
        Self { inner: slots }
    }
}

// No `impl From<BoxPool> for Vec<BoxRef>` — the natural body
// (`pool.inner.into_iter().flatten().collect()`) drops `None` holes,
// which would silently collapse the sparse position layout and break
// `OpRef::raw() as usize` index lookups against the result. Callers
// that need the raw slot table must use `BoxPool::into_slots()` and
// keep working in `Vec<Option<BoxRef>>` shape.
