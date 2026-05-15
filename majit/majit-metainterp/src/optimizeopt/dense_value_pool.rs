//! `Vec`-backed dense constant pool.
//!
//! Replaces `HashMap<u32, Value>` for per-trace constant storage,
//! mirroring `opencoder.py:482-486` upstream (`self._refs = [...]`,
//! `self._bigints = []`, `self._floats = []`) — PyPy uses
//! position-indexed lists, not a dict. Pyre's variants
//! (`OpRef::ConstInt(idx)` / `ConstFloat(idx)` / `ConstPtr(idx)`)
//! currently share a single index namespace; the per-type three-Vec
//! split lands in a later slice without affecting this storage.
//!
//! HashMap-shaped methods (`get`, `insert`, `iter`, `is_empty`, ...)
//! are preserved so call sites do not need to learn about index
//! arithmetic.

use majit_ir::Value;

/// Vec-backed dense store keyed by `OpRef::const_index()`. Slot
/// `i` holds `Some(value)` when index `i` was assigned by
/// `make_constant` / `seed_constant`; gaps remain `None`.
#[derive(Clone, Debug, Default)]
pub struct DenseValuePool {
    slots: Vec<Option<Value>>,
}

impl DenseValuePool {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Slot count (includes None gaps). Mirrors `HashMap::len()`
    /// returning the total entries — note: gaps count toward `len`.
    /// Most callers used `is_empty()` not `len()`, so this is rare.
    pub fn _slot_count(&self) -> usize {
        self.slots.len()
    }

    /// True when no slot has been allocated. Direct PyPy parity:
    /// `len(trace._bigints) == 0 and len(trace._floats) == 0 and ...`.
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.is_none())
    }

    /// HashMap-style get. `idx` is a `&u32` to match the
    /// `HashMap::get(&key)` shape that call sites already use.
    pub fn get(&self, idx: &u32) -> Option<&Value> {
        self.slots.get(*idx as usize)?.as_ref()
    }

    /// HashMap-style insert. Resizes with `None` padding if the
    /// index exceeds the current vec length.
    pub fn insert(&mut self, idx: u32, value: Value) {
        let i = idx as usize;
        if i >= self.slots.len() {
            self.slots.resize(i + 1, None);
        }
        self.slots[i] = Some(value);
    }

    /// Iterate `(idx, &Value)` over present slots only, in index
    /// order. Replaces the `HashMap::iter()` `(&u32, &Value)` pattern;
    /// the iteration order is now deterministic (was non-deterministic
    /// under the previous HashMap).
    pub fn iter(&self) -> impl Iterator<Item = (u32, &Value)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|v| (i as u32, v)))
    }

    /// Largest occupied index, or `None` when empty. Matches the
    /// `self.const_pool.keys().max()` pattern in
    /// `optimizer.rs::after_propagate_forward`.
    pub fn max_index(&self) -> Option<u32> {
        for (i, s) in self.slots.iter().enumerate().rev() {
            if s.is_some() {
                return Some(i as u32);
            }
        }
        None
    }

    /// HashMap::entry-style upsert: `or_insert_with(...)` semantics.
    /// Used by unroll for first-seen wins on shared constants.
    pub fn entry_or_insert_with<F: FnOnce() -> Value>(&mut self, idx: u32, f: F) -> &mut Value {
        let i = idx as usize;
        if i >= self.slots.len() {
            self.slots.resize(i + 1, None);
        }
        if self.slots[i].is_none() {
            self.slots[i] = Some(f());
        }
        self.slots[i].as_mut().unwrap()
    }
}
