//! Vec-backed membership set used to replace small `HashSet`s
//! per the house no-HashMap rule (`AGENTS.md` §2 + stricter project
//! policy).
//!
//! Stores entries in insertion order in a `Vec<T>` and performs
//! linear scans on insert/contains. Intended for sets whose live size
//! per trace stays small, matching the PyPy upstream which uses
//! Python `set`/`dict` here purely for object-identity membership,
//! not for size scaling.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VecSet<T: Eq> {
    entries: Vec<T>,
}

impl<T: Eq> Default for VecSet<T> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<T: Eq> VecSet<T> {
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

    pub fn as_slice(&self) -> &[T] {
        &self.entries
    }

    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: std::borrow::Borrow<Q>,
        Q: ?Sized + Eq,
    {
        self.entries.iter().any(|t| t.borrow() == value)
    }

    /// `HashSet::insert(v)` parity: returns true iff the value was new.
    pub fn insert(&mut self, value: T) -> bool {
        if self.entries.iter().any(|t| t == &value) {
            false
        } else {
            self.entries.push(value);
            true
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// `HashSet::remove(v)` parity: remove and return whether the value
    /// was present. Preserves order of remaining entries.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: std::borrow::Borrow<Q>,
        Q: ?Sized + Eq,
    {
        if let Some(idx) = self.entries.iter().position(|t| t.borrow() == value) {
            self.entries.remove(idx);
            true
        } else {
            false
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.entries.iter()
    }
}

impl<T: Eq> FromIterator<T> for VecSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut out = Self::new();
        for v in iter {
            out.insert(v);
        }
        out
    }
}

impl<T: Eq> Extend<T> for VecSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for v in iter {
            self.insert(v);
        }
    }
}

impl<T: Eq, const N: usize> From<[T; N]> for VecSet<T> {
    fn from(arr: [T; N]) -> Self {
        arr.into_iter().collect()
    }
}

impl<T: Eq> IntoIterator for VecSet<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a, T: Eq> IntoIterator for &'a VecSet<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}
