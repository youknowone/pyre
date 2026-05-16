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

/// Vec-backed associative container with `HashMap`-shaped get / insert /
/// entry-or-insert / clear methods. Equality on the key uses `==`.
#[derive(Clone, Debug)]
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

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.entries
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
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

    /// `HashMap::remove(k)` parity: remove and return the value if present.
    /// Order of remaining entries is preserved (uses `Vec::remove`).
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.entries.iter().position(|(k, _)| k == key)?;
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
}
