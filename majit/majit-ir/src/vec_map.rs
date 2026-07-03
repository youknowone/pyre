//! Vec-backed associative map used by majit.
//!
//! The implementation lives in `vecmap_rs::VecMap`; this module re-exports
//! it together with the small extension API majit expects.
//!
//! [`VecMapExt`] re-exposes the `entry_or_insert_with`, `entry_or_default`
//! and `iter_entries_mut` shortcut methods majit uses, so caller sites keep
//! their `map.entry_or_insert_with(k, f)` shape without going through the
//! intermediate `Entry` value.

pub use vecmap_rs::VecMap;

/// The compiled-trace constant pool: position → constant value.
///
/// Backed by [`indexmap::IndexMap`] rather than [`VecMap`] because the pool is
/// built by inserting one entry per const-folded position (up to the full
/// trace length) and read back by keyed lookup and in-order iteration.
/// `VecMap`'s `entry`/`get`/`insert` are linear scans (`iter().position`), so a
/// large trace's pool made those O(n²); `IndexMap` gives O(1) keyed access
/// while preserving insertion order, so codegen that iterates the pool is
/// unaffected.
pub type ConstMap<V> = indexmap::IndexMap<u32, V>;

impl<V> crate::resoperation::ConstLookup<V> for VecMap<u32, V> {
    fn lookup(&self, key: u32) -> Option<&V> {
        self.get(&key)
    }
}

/// `entry().or_insert_with(...)` / `entry().or_default()` shortcuts.
pub trait VecMapExt<K, V> {
    fn entry_or_insert_with<F: FnOnce() -> V>(&mut self, key: K, f: F) -> &mut V;
    fn entry_or_default(&mut self, key: K) -> &mut V
    where
        V: Default;
    /// Mutable access to both key and value. Use only when the key payload is
    /// itself part of a GC-traced object graph and must be updated in place.
    fn iter_entries_mut(&mut self) -> vecmap_rs::map::IterMut2<'_, K, V>;
}

impl<K: Eq, V> VecMapExt<K, V> for vecmap_rs::VecMap<K, V> {
    fn entry_or_insert_with<F: FnOnce() -> V>(&mut self, key: K, f: F) -> &mut V {
        self.entry(key).or_insert_with(f)
    }
    fn entry_or_default(&mut self, key: K) -> &mut V
    where
        V: Default,
    {
        self.entry(key).or_default()
    }
    fn iter_entries_mut(&mut self) -> vecmap_rs::map::IterMut2<'_, K, V> {
        use vecmap_rs::map::MutableKeys;
        self.iter_mut2()
    }
}
