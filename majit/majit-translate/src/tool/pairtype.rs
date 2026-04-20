//! RPython `rpython/tool/pairtype.py` — `pair()` / `pairtype()` /
//! `DoubleDispatchRegistry` port.
//!
//! Upstream's `pair(a, b)` machinery drives `binaryop.py` /
//! `unaryop.py` class-extension dispatch. Rust port mirrors the
//! `DoubleDispatchRegistry` precisely (pairtype.py:75-96) and models
//! the pair-MRO walk with [`pairmro`]. `pair()` / `pairtype()`
//! themselves are tightly bound to Python's metaclass machinery and
//! have no direct Rust analog — the `SomeValueTag` hierarchy defined
//! in `annotator::model` plays the role of the pair class, and this
//! module supplies the registry + MRO walk that indexes it.

use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

/// RPython `pairmro(cls1, cls2)` (pairtype.py:65-73).
///
/// ```python
/// def pairmro(cls1, cls2):
///     for base2 in cls2.__mro__:
///         for base1 in cls1.__mro__:
///             yield (base1, base2)
/// ```
///
/// Resolution order on pairs of types for double dispatch.
/// Compatible with the MRO of `pairtype(cls1, cls2)`.
pub fn pairmro<'a, T1: Copy, T2: Copy>(
    mro1: &'a [T1],
    mro2: &'a [T2],
) -> impl Iterator<Item = (T1, T2)> + 'a {
    mro2.iter()
        .flat_map(move |b2| mro1.iter().map(move |b1| (*b1, *b2)))
}

/// RPython `class DoubleDispatchRegistry` (pairtype.py:75-96).
///
/// A mapping of pairs of types to arbitrary objects respecting
/// inheritance. Upstream keeps a `_registry` dict plus a `_cache` that
/// is rebuilt after every `__setitem__` (the cache is a copy of the
/// registry at the time of the last update; subsequent `__getitem__`
/// calls walk `pairmro` within that snapshot).
///
/// Rust port stores the registry in a `HashMap` and keeps the cache
/// semantic by re-copying on every insert. Lookup walks `pairmro` on
/// caller-supplied MRO lists (enum variants don't carry
/// class-hierarchy metadata; the caller — currently
/// `SomeValueTag::mro` — owns that information).
pub struct DoubleDispatchRegistry<K1, K2, V>
where
    K1: Copy + Eq + Hash,
    K2: Copy + Eq + Hash,
{
    /// RPython `self._registry = {}` (pairtype.py:80). Stored with
    /// `Rc<V>` because upstream's `_cache = self._registry.copy()`
    /// shares Python references (no deep clone); Rust matches that
    /// with reference-counted sharing.
    _registry: HashMap<(K1, K2), Rc<V>>,
    /// RPython `self._cache = {}` (pairtype.py:81). Starts as a copy of
    /// `_registry` and grows on MRO hits so subsequent lookups for the
    /// same clspair are O(1).
    _cache: HashMap<(K1, K2), Rc<V>>,
}

impl<K1, K2, V> DoubleDispatchRegistry<K1, K2, V>
where
    K1: Copy + Eq + Hash,
    K2: Copy + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            _registry: HashMap::new(),
            _cache: HashMap::new(),
        }
    }

    /// RPython `__setitem__(self, clspair, value)` (pairtype.py:94-96).
    ///
    /// ```python
    /// def __setitem__(self, clspair, value):
    ///     self._registry[clspair] = value
    ///     self._cache = self._registry.copy()
    /// ```
    pub fn set(&mut self, clspair: (K1, K2), value: V) {
        let rc = Rc::new(value);
        self._registry.insert(clspair, Rc::clone(&rc));
        // Upstream `_cache = self._registry.copy()` — shallow copy of
        // the dict, same value references. Rust Rc::clone matches.
        self._cache = self
            ._registry
            .iter()
            .map(|(k, v)| (*k, Rc::clone(v)))
            .collect();
    }

    /// RPython `__getitem__(self, clspair)` (pairtype.py:83-92).
    ///
    /// ```python
    /// def __getitem__(self, clspair):
    ///     try:
    ///         return self._cache[clspair]
    ///     except KeyError:
    ///         cls1, cls2 = clspair
    ///         for c1, c2 in pairmro(cls1, cls2):
    ///             if (c1, c2) in self._cache:
    ///                 return self._cache[(c1, c2)]
    ///         else:
    ///             raise
    /// ```
    pub fn get(&self, clspair: (K1, K2), mro1: &[K1], mro2: &[K2]) -> Option<&V> {
        if let Some(v) = self._cache.get(&clspair) {
            return Some(v.as_ref());
        }
        for (c1, c2) in pairmro(mro1, mro2) {
            if let Some(v) = self._cache.get(&(c1, c2)) {
                return Some(v.as_ref());
            }
        }
        None
    }
}

impl<K1, K2, V> Default for DoubleDispatchRegistry<K1, K2, V>
where
    K1: Copy + Eq + Hash,
    K2: Copy + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairmro_product_order_matches_upstream() {
        let mro1 = [1u8, 10];
        let mro2 = [2u8, 20];
        let got: Vec<_> = pairmro(&mro1, &mro2).collect();
        // upstream: outer loop over mro2, inner over mro1.
        assert_eq!(got, vec![(1, 2), (10, 2), (1, 20), (10, 20)]);
    }

    #[test]
    fn registry_returns_exact_match_first() {
        let mut reg: DoubleDispatchRegistry<i32, i32, &'static str> = DoubleDispatchRegistry::new();
        reg.set((1, 2), "exact");
        reg.set((0, 0), "fallback");
        assert_eq!(reg.get((1, 2), &[1, 0], &[2, 0]), Some(&"exact"));
    }

    #[test]
    fn registry_walks_pairmro_on_miss() {
        let mut reg: DoubleDispatchRegistry<i32, i32, &'static str> = DoubleDispatchRegistry::new();
        reg.set((0, 0), "fallback");
        assert_eq!(reg.get((1, 2), &[1, 0], &[2, 0]), Some(&"fallback"));
    }

    #[test]
    fn registry_missing_returns_none() {
        let reg: DoubleDispatchRegistry<i32, i32, &'static str> = DoubleDispatchRegistry::new();
        assert_eq!(reg.get((1, 2), &[1], &[2]), None);
    }
}
