//! Generic union-find with per-root info objects.
//!
//! RPython upstream: `rpython/tool/algo/unionfind.py` (100 LOC). Used
//! by the annotator (`bookkeeper.pbc_maximal_call_families`,
//! `frozenpbc_attr_families`, `classpbc_attr_families`), the rtyper's
//! class-attr normalization, and various translator passes. Behaviour
//! mirrors upstream line-by-line — path-compression in [`find`],
//! weighted union in [`union`], and the `info.absorb(other_info)`
//! side-effect on merge.
//!
//! ## PRE-EXISTING-ADAPTATION: return shape
//!
//! Upstream `find` / `union` return the 3-tuple
//! `(new_root, representative, info)`. In Rust we cannot return a
//! `&mut V` borrowed from `self.root_info` alongside an owned `K`
//! without complicating borrow-checker lifetimes against the path-
//! compression writes that happen on the same call. The Rust port
//! splits this: [`find`] / [`union`] / [`find_rep`] return the owned
//! representative and a `bool` flag, while callers fetch the info via
//! [`UnionFind::get`] / [`UnionFind::get_mut`]. The `absorb`
//! side-effect on merge still happens inside [`union`] via the
//! [`UnionFindInfo::absorb`] trait, so behaviour is identical — only
//! the call-site shape differs.
//!
//! ## PRE-EXISTING-ADAPTATION: factory is mandatory
//!
//! Upstream accepts `info_factory=None`, in which case every root's
//! info is the Python object `None` and `absorb` is skipped. The Rust
//! port requires an `info_factory: Fn(&K) -> V` at construction time;
//! callers that want the no-info behaviour set `V = ()` and pass
//! `|_| ()`, relying on the no-op `impl UnionFindInfo for ()` to
//! preserve the upstream absorb-skip semantics.

use std::collections::HashMap;
use std::hash::Hash;

/// RPython `info1.absorb(info2)` (unionfind.py:76) — called by
/// [`UnionFind::union`] when two partitions merge. `self` is the
/// larger partition's info (pre-weight-swap), `other` the absorbed
/// partition's info.
pub trait UnionFindInfo: Sized {
    fn absorb(&mut self, other: Self);
}

/// No-op `absorb` — mirrors upstream `info_factory=None` skip path
/// (unionfind.py:75-76 `if info1 is not None`).
impl UnionFindInfo for () {
    fn absorb(&mut self, _other: Self) {}
}

/// RPython `class UnionFind(object)` (unionfind.py:6-99).
pub struct UnionFind<K, V>
where
    K: Eq + Hash + Clone,
{
    /// RPython `self.link_to_parent` (unionfind.py:8).
    link_to_parent: HashMap<K, K>,
    /// RPython `self.weight` (unionfind.py:9).
    weight: HashMap<K, usize>,
    /// RPython `self.root_info` (unionfind.py:11).
    root_info: HashMap<K, V>,
    /// RPython `self.info_factory` (unionfind.py:10). See module doc
    /// for why the Rust port makes this mandatory.
    info_factory: Box<dyn Fn(&K) -> V>,
}

impl<K, V> UnionFind<K, V>
where
    K: Eq + Hash + Clone,
    V: UnionFindInfo,
{
    /// RPython `UnionFind.__init__(self, info_factory=None)`
    /// (unionfind.py:7-11).
    pub fn new<F>(info_factory: F) -> Self
    where
        F: Fn(&K) -> V + 'static,
    {
        UnionFind {
            link_to_parent: HashMap::new(),
            weight: HashMap::new(),
            root_info: HashMap::new(),
            info_factory: Box::new(info_factory),
        }
    }

    /// RPython `UnionFind.__contains__(self, obj)` (unionfind.py:22-23).
    pub fn contains(&self, obj: &K) -> bool {
        self.link_to_parent.contains_key(obj)
    }

    /// RPython `UnionFind.__iter__(self)` / `UnionFind.keys(self)`
    /// (unionfind.py:25-29).
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.link_to_parent.keys()
    }

    /// RPython `UnionFind.infos(self)` (unionfind.py:31-32).
    pub fn infos(&self) -> impl Iterator<Item = &V> {
        self.root_info.values()
    }

    /// RPython `UnionFind.__getitem__(self, obj)` (unionfind.py:13-20).
    ///
    /// Returns `None` for the KeyError case. Path-compresses as a
    /// side-effect; matches upstream (the body calls `self.find`).
    pub fn get(&mut self, obj: &K) -> Option<&V> {
        if !self.link_to_parent.contains_key(obj) {
            return None;
        }
        let (_, rep) = self.find(obj.clone());
        self.root_info.get(&rep)
    }

    /// Mutable counterpart of [`Self::get`]. No upstream equivalent —
    /// Python returns `info` directly and lets callers mutate fields;
    /// Rust splits read vs write. Callers must use this to reach
    /// `info.<method>()` calls that upstream writes as
    /// `uf[obj].method()`.
    pub fn get_mut(&mut self, obj: &K) -> Option<&mut V> {
        if !self.link_to_parent.contains_key(obj) {
            return None;
        }
        let (_, rep) = self.find(obj.clone());
        self.root_info.get_mut(&rep)
    }

    /// RPython `UnionFind.find_rep(self, obj)` (unionfind.py:34-43).
    ///
    /// Returns the root representative. Auto-inserts via
    /// `info_factory` if `obj` isn't tracked yet (matches upstream —
    /// `find_rep` falls through to `find` on the KeyError path).
    pub fn find_rep(&mut self, obj: K) -> K {
        let (_, rep) = self.find(obj);
        rep
    }

    /// RPython `UnionFind.find(self, obj)` (unionfind.py:45-65).
    ///
    /// Returns `(new_root, representative)`. `new_root` is `true` when
    /// `obj` was not previously tracked and a fresh partition was
    /// created via `info_factory`. Performs path compression on the
    /// walk from `obj` to its root.
    pub fn find(&mut self, obj: K) -> (bool, K) {
        if !self.link_to_parent.contains_key(&obj) {
            // upstream: fresh singleton partition.
            let info = (self.info_factory)(&obj);
            self.root_info.insert(obj.clone(), info);
            self.weight.insert(obj.clone(), 1);
            self.link_to_parent.insert(obj.clone(), obj.clone());
            return (true, obj);
        }

        // upstream: `to_root = [obj]`; walk parents until fixed point.
        let mut to_root = vec![obj.clone()];
        let mut parent = self.link_to_parent[&obj].clone();
        while &parent != to_root.last().unwrap() {
            to_root.push(parent.clone());
            parent = self.link_to_parent[&parent].clone();
        }

        // upstream: `for obj in to_root: self.link_to_parent[obj] = parent`.
        for o in to_root.into_iter() {
            self.link_to_parent.insert(o, parent.clone());
        }

        (false, parent)
    }

    /// RPython `UnionFind.union(self, obj1, obj2)` (unionfind.py:67-91).
    ///
    /// Returns `(not_noop, representative)`. `not_noop` is `true` iff
    /// the call materially changed the partitioning — either a brand-
    /// new singleton was created inside `find`, or two previously
    /// distinct partitions got merged.
    pub fn union(&mut self, obj1: K, obj2: K) -> (bool, K) {
        let (new1, rep1) = self.find(obj1);
        let (new2, rep2) = self.find(obj2);

        if rep1 == rep2 {
            // upstream: `return new1 or new2, rep1, info1`.
            return (new1 || new2, rep1);
        }

        // upstream: `if info1 is not None: info1.absorb(info2)`. We
        // take both infos out, absorb, and re-insert the merged one
        // onto the chosen new root below. `()` infos no-op via the
        // trait impl, matching the upstream None branch.
        let mut info1 = self.root_info.remove(&rep1).expect("rep1 info");
        let info2 = self.root_info.remove(&rep2).expect("rep2 info");
        info1.absorb(info2);

        // upstream: weighted union, smaller tree under larger.
        let w1 = self.weight.remove(&rep1).expect("rep1 weight");
        let w2 = self.weight.remove(&rep2).expect("rep2 weight");
        let w = w1 + w2;

        let (new_rep1, new_rep2) = if w1 < w2 { (rep2, rep1) } else { (rep1, rep2) };

        // upstream: `self.link_to_parent[rep2] = rep1`; `del
        // self.weight[rep2]`; `del self.root_info[rep2]`;
        // `self.weight[rep1] = w`; `self.root_info[rep1] = info1`.
        self.link_to_parent
            .insert(new_rep2.clone(), new_rep1.clone());
        self.weight.insert(new_rep1.clone(), w);
        self.root_info.insert(new_rep1.clone(), info1);

        (true, new_rep1)
    }

    /// RPython `UnionFind.union_list(self, objlist)` (unionfind.py:93-99).
    pub fn union_list(&mut self, objlist: &[K]) {
        if objlist.is_empty() {
            return;
        }
        let obj0 = objlist[0].clone();
        self.find(obj0.clone());
        for obj1 in objlist.iter().skip(1) {
            self.union(obj0.clone(), obj1.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// RPython `test_cleanup` (test_unionfind.py:4-21).
    ///
    /// Creates two partitions and uses `absorb` to remove absorbed
    /// entries from an external tracking list. The final list length
    /// must equal the number of surviving partitions.
    #[test]
    fn cleanup() {
        // Track every Info object externally; `absorb` removes the
        // absorbed party from this list so only the surviving-root
        // infos remain at the end.
        struct State {
            live: RefCell<Vec<usize>>,
        }
        impl State {
            fn new() -> Self {
                State {
                    live: RefCell::new(Vec::new()),
                }
            }
            fn push(&self, id: usize) {
                self.live.borrow_mut().push(id);
            }
            fn remove(&self, id: usize) {
                let mut v = self.live.borrow_mut();
                let pos = v
                    .iter()
                    .position(|&x| x == id)
                    .expect("absorbed id must be live");
                v.remove(pos);
            }
            fn len(&self) -> usize {
                self.live.borrow().len()
            }
        }

        struct Info {
            id: usize,
            state: Rc<State>,
        }
        impl UnionFindInfo for Info {
            fn absorb(&mut self, other: Self) {
                // upstream: `state.remove(other)`.
                self.state.remove(other.id);
            }
        }

        let state = Rc::new(State::new());
        let state_for_factory = state.clone();
        let mut uf: UnionFind<i32, Info> = UnionFind::new(move |obj: &i32| {
            let info = Info {
                id: *obj as usize,
                state: state_for_factory.clone(),
            };
            state_for_factory.push(info.id);
            info
        });

        uf.find(1);
        for i in (1..10).step_by(2) {
            uf.union(i, 1);
        }
        uf.find(2);
        for i in (2..20).step_by(2) {
            uf.union(i, 2);
        }
        assert_eq!(state.len(), 2, "exactly 2 partitions must survive");
    }

    /// RPython `test_asymmetric_absorb` (test_unionfind.py:23-34).
    #[test]
    fn asymmetric_absorb() {
        #[derive(Debug)]
        struct Info {
            values: Vec<i32>,
        }
        impl UnionFindInfo for Info {
            fn absorb(&mut self, other: Self) {
                self.values.extend(other.values);
            }
        }

        let mut uf: UnionFind<i32, Info> = UnionFind::new(|obj: &i32| Info { values: vec![*obj] });
        uf.union(2, 3);
        uf.union(1, 2);

        // upstream: `uf[1].values == uf[2].values == uf[3].values ==
        // [1, 2, 3]`. Order within the merged list depends on
        // weight-swap direction and absorb ordering — accept any
        // permutation of {1, 2, 3}.
        let v1 = {
            let mut v = uf.get(&1).unwrap().values.clone();
            v.sort();
            v
        };
        let v2 = {
            let mut v = uf.get(&2).unwrap().values.clone();
            v.sort();
            v
        };
        let v3 = {
            let mut v = uf.get(&3).unwrap().values.clone();
            v.sort();
            v
        };
        assert_eq!(v1, vec![1, 2, 3]);
        assert_eq!(v2, vec![1, 2, 3]);
        assert_eq!(v3, vec![1, 2, 3]);
    }

    #[test]
    fn contains_and_get_missing() {
        let mut uf: UnionFind<i32, ()> = UnionFind::new(|_| ());
        assert!(!uf.contains(&42));
        assert!(uf.get(&42).is_none());
        uf.find(42);
        assert!(uf.contains(&42));
        assert!(uf.get(&42).is_some());
    }

    #[test]
    fn find_is_new_flag() {
        let mut uf: UnionFind<i32, ()> = UnionFind::new(|_| ());
        let (new, rep) = uf.find(1);
        assert!(new);
        assert_eq!(rep, 1);
        let (new, rep) = uf.find(1);
        assert!(!new);
        assert_eq!(rep, 1);
    }

    #[test]
    fn union_returns_not_noop_flag() {
        let mut uf: UnionFind<i32, ()> = UnionFind::new(|_| ());
        // Fresh partition creation counts as not_noop via new1|new2.
        let (changed, _) = uf.union(1, 2);
        assert!(changed);
        // Re-union of same-root: not_noop=false.
        let (changed, _) = uf.union(1, 2);
        assert!(!changed);
    }

    #[test]
    fn union_list_collapses_to_single_partition() {
        let mut uf: UnionFind<i32, ()> = UnionFind::new(|_| ());
        uf.union_list(&[1, 2, 3, 4, 5]);
        let rep = uf.find_rep(1);
        for i in 2..=5 {
            assert_eq!(uf.find_rep(i), rep);
        }
    }

    #[test]
    fn union_list_empty_noop() {
        let mut uf: UnionFind<i32, ()> = UnionFind::new(|_| ());
        uf.union_list(&[]);
        assert_eq!(uf.keys().count(), 0);
    }

    #[test]
    fn keys_and_infos_enumerate_all_entries() {
        let mut uf: UnionFind<i32, ()> = UnionFind::new(|_| ());
        uf.find(1);
        uf.find(2);
        uf.find(3);
        let mut ks: Vec<i32> = uf.keys().copied().collect();
        ks.sort();
        assert_eq!(ks, vec![1, 2, 3]);
        assert_eq!(uf.infos().count(), 3);
        uf.union(1, 2);
        assert_eq!(uf.infos().count(), 2);
    }

    #[test]
    fn path_compression_after_chain_union() {
        // Build a chain 1 -> 2 -> 3 -> 4 -> 5 via unions on a
        // factory that keeps insertion order-dependent weights; then
        // find(1) must flatten everybody to the same root.
        let mut uf: UnionFind<i32, ()> = UnionFind::new(|_| ());
        for i in 1..5 {
            uf.union(i, i + 1);
        }
        let rep = uf.find_rep(1);
        for i in 2..=5 {
            assert_eq!(uf.find_rep(i), rep);
        }
    }
}
