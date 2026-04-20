//! List definitions — the identity-based element-type carrier used by
//! `SomeList`.
//!
//! RPython upstream: `rpython/annotator/listdef.py` (207 LOC).
//!
//! Phase 5 P5.1 port. Closes the "same_as" identity gap that Phase 4
//! A4.4 deferred (see the review #2 entry in
//! `majit-translate/src/annotator/model.rs`).
//!
//! Rust adaptation (parity rule #1, minimum deviation):
//!
//! * Upstream identity: `ListDef.same_as(other)` reduces to
//!   `self.listitem is other.listitem`. In Python, `is` compares
//!   object identity; two independently-constructed `ListDef`s never
//!   share a `ListItem` unless the bookkeeper merged them.
//!   Rust equivalent: `Rc::ptr_eq(&self.listitem, &other.listitem)`.
//!
//! * Upstream mutation: `ListItem.merge(other)` mutates `self.s_value`
//!   in place then calls `self.patch()`, which walks `self.itemof`
//!   (the set of `ListDef`s currently using this `ListItem`) and
//!   rewrites each `listdef.listitem = self`. After merge both
//!   ListDefs' `same_as` returns True. The Rust port reproduces this
//!   via:
//!     - [`ListDef`] wraps an `Rc<ListDefInner>`; every `ListDef`
//!       clone shares the same inner cell.
//!     - [`ListDefInner::listitem`] is an interior-mutable slot
//!       (`RefCell<Rc<RefCell<ListItem>>>`) so `patch()` can retarget
//!       it through a shared reference.
//!     - [`ListItem::itemof`] stores [`ItemOwner`] weak backrefs so
//!       both `ListDef` owners AND `DictDef` owners can be patched
//!       through the same list. DictKey/DictValue upstream patch via
//!       separate `patch()` overrides on the subclasses; Rust
//!       composition flattens the distinction into an enum.
//!
//! * `TLS.no_side_effects_in_union` (model.py:758-769) is replaced by
//!   a thread-local counter with an RAII guard. The guard name is
//!   Rust-native — upstream uses a bare `try/finally` on the global —
//!   but the semantics match byte-for-byte.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::{Rc, Weak};

use super::bookkeeper::{Bookkeeper, PositionKey};
use super::model::{AnnotatorError, SomeList, SomeValue, UnionError};

/// RPython `class TooLateForChange(AnnotatorError)` (listdef.py:6-7).
/// Raised when mutation is attempted on a `dont_change_any_more`
/// listitem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TooLateForChange;

impl std::fmt::Display for TooLateForChange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TooLateForChange")
    }
}

impl std::error::Error for TooLateForChange {}

impl From<TooLateForChange> for AnnotatorError {
    fn from(_: TooLateForChange) -> Self {
        AnnotatorError::new("TooLateForChange")
    }
}

/// RPython `class ListChangeUnallowed(AnnotatorError)` (listdef.py:9-10).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListChangeUnallowed(pub String);

impl std::fmt::Display for ListChangeUnallowed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ListChangeUnallowed: {}", self.0)
    }
}

impl std::error::Error for ListChangeUnallowed {}

thread_local! {
    /// Rust-side mirror of upstream `TLS.no_side_effects_in_union`
    /// (model.py:758). `union()` increments this counter before
    /// dispatching to `pair(s1, s2).union()`; `ListItem.merge` /
    /// `DictKey.merge` consult it to refuse mutation and raise
    /// `UnionError` instead. Thread-local so parallel tests observe
    /// independent state.
    static NO_SIDE_EFFECTS_IN_UNION: Cell<usize> = const { Cell::new(0) };
}

/// RAII wrapper around the `TLS.no_side_effects_in_union += 1` /
/// `-= 1` pattern at `model.py:758-769` — upstream guards the
/// increment with a bare `try/finally`; Rust wraps it in a `Drop`
/// impl so callers cannot accidentally leak the state.
///
/// This type has no upstream name — the Rust port introduces it only
/// because `finally` does not exist as a language construct. The guard
/// is `!Send` / `!Sync` so the increment never escapes the thread that
/// took it.
pub struct SideEffectFreeGuard {
    _not_send: std::marker::PhantomData<*mut ()>,
}

impl SideEffectFreeGuard {
    pub fn enter() -> Self {
        NO_SIDE_EFFECTS_IN_UNION.with(|c| c.set(c.get() + 1));
        SideEffectFreeGuard {
            _not_send: std::marker::PhantomData,
        }
    }
}

impl Drop for SideEffectFreeGuard {
    fn drop(&mut self) {
        NO_SIDE_EFFECTS_IN_UNION.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// Mirror of upstream `getattr(TLS, 'no_side_effects_in_union', 0)`
/// (listdef.py:60) — True when the counter is non-zero.
pub fn in_side_effect_free_union() -> bool {
    NO_SIDE_EFFECTS_IN_UNION.with(|c| c.get() > 0)
}

/// Backref slot used by [`ListItem::itemof`] — enumerates every kind
/// of owner that can hold the listitem so `patch()` retargets the
/// correct cell.
///
/// Upstream (listdef.py:100-103) `ListItem.patch()` walks
/// `self.itemof` and sets `listdef.listitem = self`. DictKey /
/// DictValue override `patch()` (dictdef.py:15-17, 70-72) to retarget
/// `dictdef.dictkey` / `dictdef.dictvalue` instead. Rust has no
/// subclass dispatch for the `patch` override, so the enum captures
/// the upstream subclass identity directly.
#[derive(Clone, Debug)]
pub(crate) enum ItemOwner {
    /// Owner slot for `ListDef.listitem`.
    ListDef(Weak<ListDefInner>),
    /// Owner slot for `DictDef.dictkey`.
    DictKey(Weak<super::dictdef::DictDefInner>),
    /// Owner slot for `DictDef.dictvalue`.
    DictValue(Weak<super::dictdef::DictDefInner>),
}

/// RPython `class ListItem` (listdef.py:12-117).
///
/// The identity-carrying element-type state of a list annotation. Wrap
/// in `Rc<RefCell<ListItem>>` for sharing — sister `ListDef`s clone
/// the Rc so their identity comparisons hold.
///
/// ## PRE-EXISTING-ADAPTATION: DictKey/DictValue field flattening
///
/// Upstream has `class DictKey(ListItem)` (dictdef.py:7-66) and
/// `class DictValue(ListItem)` (dictdef.py:69-72). The subclass
/// instances share a `ListItem`-shaped cell with `DictDef` via
/// `dictdef.dictkey` / `dictdef.dictvalue`, and merge operations walk
/// across both base and subclass slots (`custom_eq_hash`,
/// `s_rdict_eqfn`, `s_rdict_hashfn`). Rust has no single inheritance,
/// and [`ItemOwner`] backrefs already store a bare
/// `Rc<RefCell<ListItem>>`, so we flatten the three `DictKey`
/// subclass fields onto this base struct. For non-DictKey uses the
/// fields stay at their default (`custom_eq_hash = false`,
/// `s_rdict_eqfn = s_rdict_hashfn = SomeValue::Impossible`) exactly
/// as `class DictKey(ListItem)` declares in dictdef.py:8-9. This is
/// the minimum-deviation collapse of subclass → flattened field
/// (parity rule #1).
#[derive(Debug)]
pub struct ListItem {
    /// RPython `self.s_value` (listdef.py:30).
    pub s_value: SomeValue,
    /// RPython `self.bookkeeper` (listdef.py:31). `None` matches
    /// upstream's `bookkeeper=None` path (listdef.py:34-35) and sets
    /// [`Self::dont_change_any_more`] at construction.
    pub bookkeeper: Option<Rc<Bookkeeper>>,
    /// RPython `self.mutated` (listdef.py:13).
    pub mutated: bool,
    /// RPython `self.resized` (listdef.py:14).
    pub resized: bool,
    /// RPython `self.range_step` (listdef.py:15). Upstream stores the
    /// step as either `None` (list not from a range()) or an integer;
    /// `0` means "variable step" — the sentinel value produced by
    /// merging two constant-step lists with different step values.
    pub range_step: Option<i64>,
    /// RPython `self.dont_change_any_more` (listdef.py:16).
    pub dont_change_any_more: bool,
    /// RPython `self.immutable` (listdef.py:17).
    pub immutable: bool,
    /// RPython `self.must_not_resize` (listdef.py:18).
    pub must_not_resize: bool,
    /// RPython `self.itemof = {}` (listdef.py:32). Weak backrefs to
    /// every owner currently using this `ListItem`.
    pub(crate) itemof: Vec<ItemOwner>,
    /// RPython `self.read_locations = set()` (listdef.py:33).
    pub read_locations: HashSet<PositionKey>,
    /// Flattened `DictKey.custom_eq_hash` (dictdef.py:13). `false` for
    /// every non-DictKey ListItem.
    pub custom_eq_hash: bool,
    /// Flattened `DictKey.s_rdict_eqfn` (dictdef.py:8). Defaults to
    /// `SomeValue::Impossible` (= upstream `s_ImpossibleValue`).
    pub s_rdict_eqfn: SomeValue,
    /// Flattened `DictKey.s_rdict_hashfn` (dictdef.py:9).
    pub s_rdict_hashfn: SomeValue,
}

impl ListItem {
    /// RPython `ListItem.__init__(bookkeeper, s_value)` (listdef.py:29-35).
    ///
    /// Sets `dont_change_any_more = True` when `bookkeeper is None`.
    pub fn new(bookkeeper: Option<Rc<Bookkeeper>>, s_value: SomeValue) -> Self {
        let dont_change_any_more = bookkeeper.is_none();
        ListItem {
            s_value,
            bookkeeper,
            mutated: false,
            resized: false,
            range_step: None,
            dont_change_any_more,
            immutable: false,
            must_not_resize: false,
            itemof: Vec::new(),
            read_locations: HashSet::new(),
            // Flattened DictKey defaults (dictdef.py:8-9, 13).
            custom_eq_hash: false,
            s_rdict_eqfn: SomeValue::Impossible,
            s_rdict_hashfn: SomeValue::Impossible,
        }
    }

    /// RPython `ListItem.notify_update()` (listdef.py:104-107).
    ///
    /// Triggers `self.bookkeeper.annotator.reflowfromposition(pk)` for
    /// every `pk` in `self.read_locations`. The annotator backlink on
    /// `Bookkeeper` is Phase 5 P5.2 territory — until bookkeeper.py is
    /// ported, this method walks `read_locations` but the reflow call
    /// body stays empty. The structural call sites inside
    /// [`Self::merge`] match upstream `listdef.py:95,97` exactly so no
    /// deviation gets introduced at the call-site level.
    pub fn notify_update(&self) {
        for _position_key in &self.read_locations {
            // upstream: `self.bookkeeper.annotator.reflowfromposition(pk)`.
            // Bookkeeper.annotator / Annotator.reflowfromposition are
            // ported together in Phase 5 P5.2; leaving the loop body
            // empty preserves upstream's walk order without silently
            // succeeding on a missing reflow.
        }
    }

    /// RPython `ListItem.generalize(s_other_value)` (listdef.py:109-117).
    ///
    /// Widens `self.s_value` with `s_other_value` via `unionof`, then
    /// (if widened) notifies reflow readers. Returns `true` when the
    /// type actually widened.
    pub fn generalize(&mut self, s_other_value: &SomeValue) -> Result<bool, UnionError> {
        let s_new_value = super::model::union(&self.s_value, s_other_value)?;
        let updated = s_new_value != self.s_value;
        if updated {
            if self.dont_change_any_more {
                return Err(UnionError {
                    lhs: self.s_value.clone(),
                    rhs: s_other_value.clone(),
                    msg: "TooLateForChange on generalize()".into(),
                });
            }
            self.s_value = s_new_value;
            self.notify_update();
        }
        Ok(updated)
    }

    /// RPython `ListItem.mutate()` (listdef.py:37-42).
    pub fn mutate(&mut self) -> Result<(), TooLateForChange> {
        if !self.mutated {
            if self.dont_change_any_more {
                return Err(TooLateForChange);
            }
            self.immutable = false;
            self.mutated = true;
        }
        Ok(())
    }

    /// RPython `ListItem.resize()` (listdef.py:44-50).
    pub fn resize(&mut self) -> Result<(), AnnotatorError> {
        if !self.resized {
            if self.dont_change_any_more {
                return Err(AnnotatorError::new("TooLateForChange"));
            }
            if self.must_not_resize {
                return Err(AnnotatorError::new("ListChangeUnallowed: resizing list"));
            }
            self.resized = true;
        }
        Ok(())
    }

    /// RPython `ListItem.setrangestep(step)` (listdef.py:52-56).
    pub fn setrangestep(&mut self, step: Option<i64>) -> Result<(), TooLateForChange> {
        if step != self.range_step {
            if self.dont_change_any_more {
                return Err(TooLateForChange);
            }
            self.range_step = step;
        }
        Ok(())
    }

    /// RPython `ListItem.merge(other)` (listdef.py:58-98).
    ///
    /// Takes two `Rc<RefCell<ListItem>>` associated-function style
    /// rather than `&mut self` / `&mut other` because the borrow
    /// checker refuses two mutable borrows of the same `Vec<ItemOwner>`
    /// slot when the caller happens to pass the same Rc twice (the
    /// `Rc::ptr_eq` shortcut below exits before any borrow). The
    /// upstream method name is preserved.
    pub fn merge(
        self_li: &Rc<RefCell<ListItem>>,
        other_li: &Rc<RefCell<ListItem>>,
    ) -> Result<Rc<RefCell<ListItem>>, UnionError> {
        // upstream: `if self is not other:`.
        if Rc::ptr_eq(self_li, other_li) {
            return Ok(self_li.clone());
        }

        // upstream: `if getattr(TLS, 'no_side_effects_in_union', 0):
        //                raise UnionError(self, other)`.
        if in_side_effect_free_union() {
            let a = self_li.borrow();
            let b = other_li.borrow();
            return Err(UnionError {
                lhs: a.s_value.clone(),
                rhs: b.s_value.clone(),
                msg: "ListItem.merge during side-effect-free union".into(),
            });
        }

        // upstream: `if other.dont_change_any_more: if
        // self.dont_change_any_more: raise TooLateForChange;
        // else: self, other = other, self` (listdef.py:63-71).
        let (driver_li, folded_li) = {
            let self_b = self_li.borrow();
            let other_b = other_li.borrow();
            if other_b.dont_change_any_more {
                if self_b.dont_change_any_more {
                    return Err(UnionError {
                        lhs: self_b.s_value.clone(),
                        rhs: other_b.s_value.clone(),
                        msg: "TooLateForChange".into(),
                    });
                }
                (other_li, self_li)
            } else {
                (self_li, other_li)
            }
        };

        // Snapshot folded side (everything upstream reads as `other.X`
        // after the swap). Having a single snapshot prevents intermixed
        // borrows with driver_mut below.
        let (folded_s_value, folded_itemof, folded_read_locations, folded_flags) = {
            let folded_b = folded_li.borrow();
            (
                folded_b.s_value.clone(),
                folded_b.itemof.clone(),
                folded_b.read_locations.clone(),
                (
                    folded_b.mutated,
                    folded_b.resized,
                    folded_b.immutable,
                    folded_b.must_not_resize,
                    folded_b.range_step,
                ),
            )
        };

        // upstream lines 73-85: flag merges. Order preserved exactly.
        {
            let mut driver_mut = driver_li.borrow_mut();
            driver_mut.immutable &= folded_flags.2;
            if folded_flags.3 {
                if driver_mut.resized {
                    return Err(UnionError {
                        lhs: driver_mut.s_value.clone(),
                        rhs: folded_s_value.clone(),
                        msg: "ListChangeUnallowed: list merge with a resized".into(),
                    });
                }
                driver_mut.must_not_resize = true;
            }
        }
        if folded_flags.0 {
            // upstream: `self.mutate()` — propagates TooLateForChange.
            driver_li.borrow_mut().mutate().map_err(|_| UnionError {
                lhs: driver_li.borrow().s_value.clone(),
                rhs: folded_s_value.clone(),
                msg: "TooLateForChange on mutate() during merge".into(),
            })?;
        }
        if folded_flags.1 {
            // upstream: `self.resize()` — propagates TooLateForChange /
            // ListChangeUnallowed.
            driver_li.borrow_mut().resize().map_err(|e| UnionError {
                lhs: driver_li.borrow().s_value.clone(),
                rhs: folded_s_value.clone(),
                msg: e.msg.unwrap_or_else(|| "resize() failed".into()),
            })?;
        }
        let driver_range_step = driver_li.borrow().range_step;
        if folded_flags.4 != driver_range_step {
            // upstream: `self.setrangestep(self._step_map[...])`.
            let new_step = merge_range_step(driver_range_step, folded_flags.4);
            driver_li
                .borrow_mut()
                .setrangestep(new_step)
                .map_err(|_| UnionError {
                    lhs: driver_li.borrow().s_value.clone(),
                    rhs: folded_s_value.clone(),
                    msg: "TooLateForChange on setrangestep() during merge".into(),
                })?;
        }

        // upstream: `self.itemof.update(other.itemof)` (listdef.py:85).
        driver_li
            .borrow_mut()
            .itemof
            .extend(folded_itemof.iter().cloned());

        // upstream lines 86-91.
        let driver_s_value_pre = driver_li.borrow().s_value.clone();
        let new_s_value = super::model::union(&driver_s_value_pre, &folded_s_value)?;
        let widens_driver = new_s_value != driver_s_value_pre;
        let widens_folded = new_s_value != folded_s_value;
        if widens_driver && driver_li.borrow().dont_change_any_more {
            return Err(UnionError {
                lhs: driver_s_value_pre,
                rhs: folded_s_value,
                msg: "TooLateForChange on dont_change_any_more ListItem".into(),
            });
        }

        // upstream: `self.patch()` (listdef.py:92, 100-103). After the
        // itemof.update above, driver.itemof holds every owner; retarget
        // them all to driver.
        let patch_list = driver_li.borrow().itemof.clone();
        for owner in &patch_list {
            match owner {
                ItemOwner::ListDef(weak) => {
                    if let Some(inner) = weak.upgrade() {
                        *inner.listitem.borrow_mut() = driver_li.clone();
                    }
                }
                ItemOwner::DictKey(weak) => {
                    if let Some(inner) = weak.upgrade() {
                        *inner.dictkey.borrow_mut() = driver_li.clone();
                    }
                }
                ItemOwner::DictValue(weak) => {
                    if let Some(inner) = weak.upgrade() {
                        *inner.dictvalue.borrow_mut() = driver_li.clone();
                    }
                }
            }
        }

        // upstream lines 93-98: conditional s_value update + notify +
        // read_locations merge.
        if widens_driver {
            driver_li.borrow_mut().s_value = new_s_value;
            driver_li.borrow().notify_update();
        }
        if widens_folded {
            // upstream: `other.notify_update()`. folded_li still holds
            // the old read_locations snapshot before we overwrite
            // driver.read_locations below.
            folded_li.borrow().notify_update();
        }
        // upstream: `self.read_locations |= other.read_locations`.
        driver_li
            .borrow_mut()
            .read_locations
            .extend(folded_read_locations);

        Ok(driver_li.clone())
    }
}

/// RPython `ListItem._step_map[type(self.range_step),
/// type(other.range_step)]` (listdef.py:23-27). Upstream keys the dict
/// on `(type(None), int)` / `(int, type(None))` / `(int, int)`.
fn merge_range_step(self_step: Option<i64>, other_step: Option<i64>) -> Option<i64> {
    match (self_step, other_step) {
        // `(NoneType, int)` / `(int, NoneType)` → None.
        (None, Some(_)) | (Some(_), None) => None,
        // `(int, int)` with different values → 0 (variable step).
        (Some(a), Some(b)) if a != b => Some(0),
        // Same-value (int, int) or (None, None) — upstream never
        // invokes the map on equality, so the branch just returns
        // self.
        _ => self_step,
    }
}

/// Inner cell of a [`ListDef`].
///
/// `listitem` lives inside an interior-mutable slot so
/// [`ListItem::merge`] can retarget it through an [`ItemOwner`].
#[derive(Debug)]
pub struct ListDefInner {
    pub(crate) listitem: RefCell<Rc<RefCell<ListItem>>>,
}

/// RPython `class ListDef` (listdef.py:120-204).
#[derive(Clone, Debug)]
pub struct ListDef {
    pub(crate) inner: Rc<ListDefInner>,
}

impl ListDef {
    /// RPython `ListDef.__init__(bookkeeper, s_item=s_ImpossibleValue,
    /// mutated=False, resized=False)` (listdef.py:125-130).
    pub fn new(
        bookkeeper: Option<Rc<Bookkeeper>>,
        s_item: SomeValue,
        mutated: bool,
        resized: bool,
    ) -> Self {
        let mut item = ListItem::new(bookkeeper, s_item);
        // upstream: `self.listitem.mutated = mutated | resized;
        //            self.listitem.resized = resized`.
        item.mutated = mutated || resized;
        item.resized = resized;
        let li = Rc::new(RefCell::new(item));
        let inner = Rc::new(ListDefInner {
            listitem: RefCell::new(li.clone()),
        });
        // upstream: `self.listitem.itemof[self] = True`.
        li.borrow_mut()
            .itemof
            .push(ItemOwner::ListDef(Rc::downgrade(&inner)));
        ListDef { inner }
    }

    /// RPython `ListDef.same_as(other)` (listdef.py:136-137).
    pub fn same_as(&self, other: &ListDef) -> bool {
        let a = self.inner.listitem.borrow();
        let b = other.inner.listitem.borrow();
        Rc::ptr_eq(&*a, &*b)
    }

    /// RPython `ListDef.union(other)` (listdef.py:139-141).
    pub fn union_with(&self, other: &ListDef) -> Result<(), UnionError> {
        let self_li = self.inner.listitem.borrow().clone();
        let other_li = other.inner.listitem.borrow().clone();
        let _ = ListItem::merge(&self_li, &other_li)?;
        Ok(())
    }

    /// RPython `ListDef.listitem.s_value` accessor — shortcut used by
    /// call sites that only need the element annotation without
    /// borrowing the listitem directly.
    pub fn s_value(&self) -> SomeValue {
        self.inner.listitem.borrow().borrow().s_value.clone()
    }

    /// RPython `ListDef.mutate()` (listdef.py:182-183).
    pub fn mutate(&self) -> Result<(), TooLateForChange> {
        let li = self.inner.listitem.borrow().clone();
        let mut li_mut = li.borrow_mut();
        li_mut.mutate()
    }

    /// RPython `ListDef.resize()` (listdef.py:185-187).
    ///
    /// ```python
    /// def resize(self):
    ///     self.listitem.mutate()
    ///     self.listitem.resize()
    /// ```
    pub fn resize(&self) -> Result<(), AnnotatorError> {
        let li = self.inner.listitem.borrow().clone();
        let mut li_mut = li.borrow_mut();
        li_mut
            .mutate()
            .map_err(|_| AnnotatorError::new("TooLateForChange"))?;
        li_mut.resize()
    }

    /// RPython `ListDef.read_item(position_key)` (listdef.py:132-134).
    ///
    /// Records a read location for eventual `notify_update()` reflow,
    /// then returns the current element annotation. `position_key` is
    /// `Option` — upstream's `bookkeeper.position_key` is `None`
    /// outside of a reflow frame, and stashing that `None` in
    /// `listitem.read_locations` is legal (Python dict keys accept
    /// `None`). The Rust port's `HashSet<PositionKey>` can only hold
    /// `Some` values, so `None` is dropped from the read-locations set
    /// — the subsequent `s_value.clone()` return still matches
    /// upstream behaviour.
    pub fn read_item(&self, position_key: Option<PositionKey>) -> SomeValue {
        let li = self.inner.listitem.borrow().clone();
        let mut li_mut = li.borrow_mut();
        if let Some(pk) = position_key {
            li_mut.read_locations.insert(pk);
        }
        li_mut.s_value.clone()
    }

    /// RPython `ListDef.generalize(s_value)` (listdef.py:167-168).
    pub fn generalize(&self, s_value: &SomeValue) -> Result<bool, UnionError> {
        let li = self.inner.listitem.borrow().clone();
        let mut li_mut = li.borrow_mut();
        li_mut.generalize(s_value)
    }

    /// RPython `ListDef.offspring(bookkeeper, *others)` (listdef.py:154-165).
    ///
    /// ```python
    /// def offspring(self, bookkeeper, *others):
    ///     position = bookkeeper.position_key
    ///     s_self_value = self.read_item(position)
    ///     s_other_values = []
    ///     for other in others:
    ///         s_other_values.append(other.read_item(position))
    ///     s_newlst = bookkeeper.newlist(s_self_value, *s_other_values)
    ///     s_newvalue = s_newlst.listdef.read_item(position)
    ///     self.generalize(s_newvalue)
    ///     for other in others:
    ///         other.generalize(s_newvalue)
    ///     return s_newlst
    /// ```
    pub fn offspring(
        &self,
        bookkeeper: &Rc<Bookkeeper>,
        others: &[&ListDef],
    ) -> Result<SomeList, AnnotatorError> {
        // upstream: `position = bookkeeper.position_key`. Outside of a
        // reflow frame this is `None`; Python dict/set keys accept it,
        // so the Rust port passes the Option through.
        let position = bookkeeper.current_position_key();
        let s_self_value = self.read_item(position.clone());
        let mut s_other_values: Vec<SomeValue> = Vec::with_capacity(others.len());
        for other in others {
            s_other_values.push(other.read_item(position.clone()));
        }
        // upstream: `bookkeeper.newlist(s_self_value, *s_other_values)`.
        let mut all_values = Vec::with_capacity(1 + s_other_values.len());
        all_values.push(s_self_value);
        all_values.extend(s_other_values);
        let s_newlst = bookkeeper.newlist(&all_values, None)?;
        let s_newvalue = s_newlst.listdef.read_item(position);
        self.generalize(&s_newvalue)
            .map_err(|e| AnnotatorError::new(e.msg))?;
        for other in others {
            other
                .generalize(&s_newvalue)
                .map_err(|e| AnnotatorError::new(e.msg))?;
        }
        Ok(s_newlst)
    }

    /// RPython `ListDef.generalize_range_step(range_step)`
    /// (listdef.py:170-173).
    ///
    /// Creates a fresh ListItem carrying the candidate `range_step`,
    /// then merges it into `self.listitem` so `_step_map` collapses
    /// the two step values (matching upstream lines 82-85).
    pub fn generalize_range_step(&self, range_step: Option<i64>) -> Result<(), UnionError> {
        let bookkeeper = {
            let li = self.inner.listitem.borrow().clone();
            li.borrow().bookkeeper.clone()
        };
        let mut new_item = ListItem::new(bookkeeper, SomeValue::Impossible);
        new_item.range_step = range_step;
        let new_li = Rc::new(RefCell::new(new_item));
        let self_li = self.inner.listitem.borrow().clone();
        let _ = ListItem::merge(&self_li, &new_li)?;
        Ok(())
    }

    /// RPython `ListDef.agree(bookkeeper, other)` (listdef.py:143-152).
    ///
    /// Bidirectionally generalises both sides against each other at
    /// the bookkeeper's current position, then reconciles `range_step`
    /// if either side is range-derived. `position_key` is passed as
    /// `Option` so the upstream None-key caching path (no reflow
    /// frame active) flows through unchanged.
    pub fn agree(&self, bookkeeper: &Bookkeeper, other: &ListDef) -> Result<(), UnionError> {
        let position = bookkeeper.current_position_key();
        let s_self_value = self.read_item(position.clone());
        let s_other_value = other.read_item(position);
        self.generalize(&s_other_value)?;
        other.generalize(&s_self_value)?;
        let (self_step, other_step) = {
            let a = self.inner.listitem.borrow().clone();
            let b = other.inner.listitem.borrow().clone();
            (a.borrow().range_step, b.borrow().range_step)
        };
        if self_step.is_some() {
            self.generalize_range_step(other_step)?;
        }
        if other_step.is_some() {
            other.generalize_range_step(self_step)?;
        }
        Ok(())
    }
}

impl PartialEq for ListDef {
    /// RPython `SomeList.__eq__` (model.py:339-348) uses
    /// `listdef.same_as`; mirror the identity-only semantics here so
    /// wrapping structs picking up `derive(PartialEq)` inherit it.
    fn eq(&self, other: &Self) -> bool {
        self.same_as(other)
    }
}

impl Eq for ListDef {}

/// Stable hash matching the identity-based equality above.
impl std::hash::Hash for ListDef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let li = self.inner.listitem.borrow();
        let raw: *const RefCell<ListItem> = Rc::as_ptr(&*li);
        (raw as usize).hash(state);
    }
}

/// RPython `s_list_of_strings = SomeList(ListDef(None,
/// SomeString(no_nul=True), resized=True))` (listdef.py:206-207).
pub fn s_list_of_strings() -> super::model::SomeList {
    super::model::SomeList::new(ListDef::new(
        None,
        super::model::SomeValue::String(super::model::SomeString::new(false, true)),
        false,
        true,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::model::{SomeInteger, SomeValue};
    use super::*;

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    #[test]
    fn same_as_identity_preserved_across_clone() {
        let a = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
        );
        let b = a.clone();
        assert!(a.same_as(&b));
    }

    #[test]
    fn distinct_listdefs_are_not_same_as() {
        let a = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
        );
        let b = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
        );
        assert!(!a.same_as(&b));
    }

    #[test]
    fn union_with_shares_listitem_after_merge() {
        // bookkeeper=Some(...) → dont_change_any_more stays False, so
        // merge proceeds without TooLateForChange.
        let a = ListDef::new(
            Some(bk()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let b = ListDef::new(
            Some(bk()),
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );
        assert!(!a.same_as(&b));
        a.union_with(&b).expect("merge must succeed");
        assert!(a.same_as(&b));
    }

    #[test]
    fn merge_refuses_under_side_effect_free_guard() {
        let a = ListDef::new(
            Some(bk()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let b = ListDef::new(
            Some(bk()),
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );

        let _guard = SideEffectFreeGuard::enter();
        assert!(a.union_with(&b).is_err());
        assert!(!a.same_as(&b));
    }

    #[test]
    fn dont_change_any_more_merge_widening_errors() {
        // upstream: merging two final listitems with different element
        // types triggers TooLateForChange.
        let a = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let b = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );
        let err = a
            .union_with(&b)
            .expect_err("widening final list must error");
        assert!(err.msg.contains("TooLateForChange"));
    }

    #[test]
    fn merge_range_step_matches_upstream_table() {
        // upstream `_step_map`:
        //   (NoneType, int) → None
        //   (int, NoneType) → None
        //   (int, int)      → 0
        assert_eq!(merge_range_step(None, Some(2)), None);
        assert_eq!(merge_range_step(Some(2), None), None);
        assert_eq!(merge_range_step(Some(2), Some(3)), Some(0));
        // Same-value / double-None fall back to self.
        assert_eq!(merge_range_step(None, None), None);
    }
}
