//! Helpers and type aliases for `indexmap::IndexMap`.

/// The compiled-trace constant pool: position → constant value.
///
/// Backed by [`indexmap::IndexMap`] rather than a linear-scan Vec because the
/// pool is built by inserting one entry per const-folded position (up to the
/// full trace length) and read back by keyed lookup and in-order iteration.
pub type ConstMap<V> = indexmap::IndexMap<u32, V>;

/// `entry().or_insert_with(...)` / `entry().or_default()` shortcuts.
pub trait IndexMapExt<K, V> {
    fn entry_or_insert_with<F: FnOnce() -> V>(&mut self, key: K, f: F) -> &mut V;
    fn entry_or_default(&mut self, key: K) -> &mut V
    where
        V: Default;
}

impl<K: Eq + std::hash::Hash, V> IndexMapExt<K, V> for indexmap::IndexMap<K, V> {
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
