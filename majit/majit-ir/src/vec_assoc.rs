//! Compatibility re-export for the Vec-backed associative map used by
//! majit.
//!
//! `VecAssoc` used to be a local Vec-backed map.  The implementation now
//! lives in `vecmap_rs::VecMap`; keep this module and type alias so the
//! existing `majit_ir::VecAssoc` / `majit_ir::vec_assoc::VecAssoc` public
//! surface remains stable while using the shared container.
//!
//! [`VecMapExt`] re-exposes the `entry_or_insert_with`, `entry_or_default`
//! and `iter_entries_mut` shortcut methods the previous local `VecAssoc`
//! provided, so caller sites keep their `map.entry_or_insert_with(k, f)`
//! shape without going through the intermediate `Entry` value.

pub type VecAssoc<K, V> = vecmap_rs::VecMap<K, V>;

impl<V> crate::resoperation::ConstLookup<V> for VecAssoc<u32, V> {
    fn lookup(&self, key: u32) -> Option<&V> {
        self.get(&key)
    }
}

/// `entry().or_insert_with(...)` / `entry().or_default()` shortcuts the
/// original [`VecAssoc`] exposed as inherent methods.  Provided as an
/// extension trait so the type alias for `vecmap_rs::VecMap` keeps the
/// caller-facing API surface unchanged.
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
