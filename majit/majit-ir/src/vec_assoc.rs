//! Vec-backed associative containers used to replace small `HashMap`s
//! per the house no-HashMap rule (`AGENTS.md` §2 + stricter project
//! policy).
//!
//! Each container keeps `(key, value)` pairs in a `Vec` and performs
//! linear scans on insert/lookup. The intended use is for pools whose
//! live size per trace stays small (typically < a few dozen entries),
//! where O(n) operations are cheap and faithful to the upstream
//! algorithm (PyPy uses `dict` here only for object-identity lookup,
//! not for size scaling).

use serde::{Deserialize, Serialize};

/// Vec-backed associative container with `HashMap`-shaped get / insert /
/// entry-or-insert / clear methods. Equality on the key uses `==`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VecAssoc<K: Eq, V> {
    entries: Vec<(K, V)>,
}

impl<K: Eq, V> Default for VecAssoc<K, V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<K: Eq, V> VecAssoc<K, V> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrow the internal `Vec<(K, V)>` as a slice. Useful for handing
    /// the contents to helpers that take `&[(K, V)]` without needing to
    /// know about VecAssoc.
    pub fn as_slice(&self) -> &[(K, V)] {
        &self.entries
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: std::borrow::Borrow<Q>,
        Q: ?Sized + Eq,
    {
        self.entries
            .iter()
            .find(|(k, _)| k.borrow() == key)
            .map(|(_, v)| v)
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: std::borrow::Borrow<Q>,
        Q: ?Sized + Eq,
    {
        self.entries
            .iter_mut()
            .find(|(k, _)| (*k).borrow() == key)
            .map(|(_, v)| v)
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: ?Sized + Eq,
    {
        self.entries.iter().any(|(k, _)| k.borrow() == key)
    }

    /// Dict-assignment semantics: overwrite existing value or append a
    /// fresh entry.
    pub fn insert(&mut self, key: K, value: V) {
        if let Some(entry) = self.entries.iter_mut().find(|(k, _)| k == &key) {
            entry.1 = value;
        } else {
            self.entries.push((key, value));
        }
    }

    /// `HashMap::entry(k).or_default()` parity — inserts `V::default()`
    /// when the key is missing.
    pub fn entry_or_default(&mut self, key: K) -> &mut V
    where
        V: Default,
    {
        self.entry_or_insert_with(key, V::default)
    }

    /// `HashMap::entry(k).or_insert_with(...)` parity.
    pub fn entry_or_insert_with<F: FnOnce() -> V>(&mut self, key: K, f: F) -> &mut V {
        let idx = match self.entries.iter().position(|(k, _)| k == &key) {
            Some(i) => i,
            None => {
                self.entries.push((key, f()));
                self.entries.len() - 1
            }
        };
        &mut self.entries[idx].1
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// `HashMap::retain(|k, v| ...)` parity: in-place filter that keeps
    /// only the entries for which the closure returns true.
    pub fn retain<F: FnMut(&K, &mut V) -> bool>(&mut self, mut f: F) {
        self.entries.retain_mut(|(k, v)| f(k, v));
    }

    /// `HashMap::remove(k)` parity: remove and return the value if present.
    /// Order of remaining entries is preserved (uses `Vec::remove`).
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: std::borrow::Borrow<Q>,
        Q: ?Sized + Eq,
    {
        let idx = self.entries.iter().position(|(k, _)| k.borrow() == key)?;
        Some(self.entries.remove(idx).1)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.entries.iter_mut().map(|(k, v)| (&*k, v))
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.entries.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, v)| v)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.entries.iter_mut().map(|(_, v)| v)
    }
}

impl<K: Eq, V> FromIterator<(K, V)> for VecAssoc<K, V> {
    /// Build a VecAssoc from an iterator of pairs. Duplicate keys collapse
    /// to the last inserted value (dict-assignment semantics).
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut out = Self::new();
        for (k, v) in iter {
            out.insert(k, v);
        }
        out
    }
}

impl<K: Eq, V, const N: usize> From<[(K, V); N]> for VecAssoc<K, V> {
    fn from(arr: [(K, V); N]) -> Self {
        arr.into_iter().collect()
    }
}

impl<K: Eq, V> IntoIterator for VecAssoc<K, V> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<(K, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a, K: Eq, V> IntoIterator for &'a VecAssoc<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::iter::Map<std::slice::Iter<'a, (K, V)>, fn(&'a (K, V)) -> (&'a K, &'a V)>;
    fn into_iter(self) -> Self::IntoIter {
        fn split<'a, K, V>(t: &'a (K, V)) -> (&'a K, &'a V) {
            (&t.0, &t.1)
        }
        self.entries
            .iter()
            .map(split as fn(&'a (K, V)) -> (&'a K, &'a V))
    }
}

impl<K, V, Q> std::ops::Index<&Q> for VecAssoc<K, V>
where
    K: Eq + std::borrow::Borrow<Q>,
    Q: ?Sized + Eq,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<V> crate::resoperation::ConstLookup<V> for VecAssoc<u32, V> {
    fn lookup(&self, key: u32) -> Option<&V> {
        self.get(&key)
    }
}
