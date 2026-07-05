//! Associative map used by majit.
//!
//! Previously backed by `vecmap_rs::VecMap` (linear-scan Vec). Replaced by
//! `indexmap::IndexMap` for O(1) keyed access while preserving insertion order.
//! The `VecMap` name is kept as a type alias to minimise churn across callers.

pub use indexmap::IndexMap as VecMap;

/// The compiled-trace constant pool: position → constant value.
///
/// Backed by [`indexmap::IndexMap`] rather than a linear-scan Vec because the
/// pool is built by inserting one entry per const-folded position (up to the
/// full trace length) and read back by keyed lookup and in-order iteration.
pub type ConstMap<V> = indexmap::IndexMap<u32, V>;

// ConstLookup already implemented via ConstMap = IndexMap<u32, V> = VecMap<u32, V>.

/// `entry().or_insert_with(...)` / `entry().or_default()` shortcuts.
pub trait VecMapExt<K, V> {
    fn entry_or_insert_with<F: FnOnce() -> V>(&mut self, key: K, f: F) -> &mut V;
    fn entry_or_default(&mut self, key: K) -> &mut V
    where
        V: Default;
}

impl<K: Eq + std::hash::Hash, V> VecMapExt<K, V> for indexmap::IndexMap<K, V> {
    fn entry_or_insert_with<F: FnOnce() -> V>(&mut self, key: K, f: F) -> &mut V {
        self.entry(key).or_insert_with(f)
    }
    fn entry_or_default(&mut self, key: K) -> &mut V
    where
        V: Default,
    {
        self.entry(key).or_default()
    }
}
