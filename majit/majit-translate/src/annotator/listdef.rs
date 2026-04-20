//! List definitions — the identity-based element-type carrier used by
//! `SomeList`.
//!
//! RPython upstream: `rpython/annotator/listdef.py` (207 LOC).
//!
//! Phase 5 P5.1 port — closes the "same_as" identity gap that Phase 4
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
//!     - [`ListItem::itemof`] stores `Weak<ListDefInner>` backrefs
//!       that survive merges but don't hold the ListDef alive.
//!
//! * Bookkeeper integration: upstream takes a live `bookkeeper`
//!   reference that informs `notify_update()` → `reflowfromposition`.
//!   The bookkeeper is Phase 5 (bookkeeper.py, 614 LOC) and has not
//!   landed yet; `ListItem::bookkeeper` is therefore `None` at the
//!   Phase 5 P5.1 cut and `notify_update` is a no-op. Sites that
//!   construct lists under `bookkeeper = None` inherit upstream's
//!   `dont_change_any_more = True` branch (listdef.py:34-35), so
//!   merges of two dont-change listitems raise [`TooLateForChange`].
//!
//! * `TLS.no_side_effects_in_union` is replaced by an [`AtomicUsize`]
//!   counter — [`enter_side_effect_free_union`] /
//!   [`leave_side_effect_free_union`]. `ListItem::merge` short-
//!   circuits to [`UnionError`] when the counter is non-zero.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use super::model::{AnnotatorError, SomeValue, UnionError};

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
    /// RPython `TLS.no_side_effects_in_union` surrogate (model.py:758).
    /// Incremented by `model::contains`, read by [`merge_items`] to
    /// refuse mutation and surface a [`UnionError`] instead.
    /// Thread-local so parallel tests do not observe each other's
    /// guard state.
    static NO_SIDE_EFFECTS_IN_UNION: Cell<usize> = const { Cell::new(0) };
}

/// RAII-style guard used by callers that need the "refuse mutation"
/// contract (`model::contains`, pattern-checking code paths).
pub struct SideEffectFreeGuard {
    // Ensure the guard is !Send / !Sync so it never escapes the
    // thread that set up the TLS increment.
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

/// True when merges must refuse mutation and raise UnionError. Callers
/// outside this module typically enter the mode via
/// [`SideEffectFreeGuard::enter`].
pub fn in_side_effect_free_union() -> bool {
    NO_SIDE_EFFECTS_IN_UNION.with(|c| c.get() > 0)
}

/// RPython `ListItem._step_map` (listdef.py:23-27). The upstream dict
/// is keyed by `(type(self.range_step), type(other.range_step))`
/// where each type is `NoneType` or `int`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeStep {
    /// upstream `range_step = None` — the list is not from a
    /// `range()` call. `_step_map` uses `type(None)`.
    NotRange,
    /// upstream `range_step = <int>`.
    Constant(i64),
    /// upstream `range_step = 0` — "variable step", set when two
    /// const-step lists with different steps merge.
    Variable,
}

impl RangeStep {
    /// RPython `_step_map[type(self), type(other)]` (listdef.py:83-84).
    fn merge(self, other: RangeStep) -> RangeStep {
        match (self, other) {
            // (None, int) / (int, None) → None.
            (RangeStep::NotRange, RangeStep::Constant(_))
            | (RangeStep::Constant(_), RangeStep::NotRange) => RangeStep::NotRange,
            // (int, int) of different values → 0 (variable step).
            (RangeStep::Constant(a), RangeStep::Constant(b)) if a != b => RangeStep::Variable,
            // Everything else including same-value (int, int) collapses
            // to self; upstream only invokes _step_map when values
            // differ, so the branch never fires on equality.
            _ => self,
        }
    }
}

/// RPython `class ListItem` (listdef.py:12-117).
///
/// The identity-carrying element-type state of a list annotation. Wrap
/// in `Rc<RefCell<ListItem>>` for sharing — sister `ListDef`s clone
/// the Rc so their identity comparisons hold.
#[derive(Debug)]
pub struct ListItem {
    /// RPython `self.s_value` (listdef.py:30).
    pub s_value: SomeValue,
    /// RPython `self.mutated` (listdef.py:13).
    pub mutated: bool,
    /// RPython `self.resized` (listdef.py:14).
    pub resized: bool,
    /// RPython `self.range_step` (listdef.py:15).
    pub range_step: RangeStep,
    /// RPython `self.dont_change_any_more` (listdef.py:16). Set when
    /// the bookkeeper is None or when the annotator has declared
    /// annotations final.
    pub dont_change_any_more: bool,
    /// RPython `self.immutable` (listdef.py:17). Set by
    /// `_immutable_fields_` hints.
    pub immutable: bool,
    /// RPython `self.must_not_resize` (listdef.py:18). Set by
    /// `make_sure_not_resized()`.
    pub must_not_resize: bool,
    /// RPython `self.itemof` (listdef.py:32). Weak backrefs to every
    /// [`ListDefInner`] currently using this `ListItem`, so
    /// `merge()` can retarget each ListDef's `listitem` slot.
    pub(crate) itemof: Vec<Weak<ListDefInner>>,
    // `read_locations` and `bookkeeper` are deferred to the
    // bookkeeper.py port. Adding the fields as unit-typed
    // placeholders keeps the struct shape close to upstream for
    // future diffs.
    pub(crate) _read_locations_placeholder: (),
    pub(crate) _bookkeeper_placeholder: (),
}

impl ListItem {
    /// RPython `ListItem.__init__(bookkeeper, s_value)` (listdef.py:29-35).
    ///
    /// `bookkeeper=None` path — sets `dont_change_any_more = True`
    /// matching upstream line 34-35. Use [`Self::with_bookkeeper`] to
    /// produce a mutable variant.
    pub fn new(s_value: SomeValue) -> Self {
        ListItem {
            s_value,
            mutated: false,
            resized: false,
            range_step: RangeStep::NotRange,
            dont_change_any_more: true,
            immutable: false,
            must_not_resize: false,
            itemof: Vec::new(),
            _read_locations_placeholder: (),
            _bookkeeper_placeholder: (),
        }
    }

    /// RPython `ListItem.__init__(bookkeeper=<live>, s_value)` path —
    /// `dont_change_any_more` stays False. Phase 5 P5.1 uses a unit
    /// placeholder for the bookkeeper; Phase 5's bookkeeper.py port
    /// replaces this with the real carrier.
    pub fn with_bookkeeper(s_value: SomeValue) -> Self {
        let mut li = Self::new(s_value);
        li.dont_change_any_more = false;
        li
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
}

/// Inner cell of a [`ListDef`].
///
/// `listitem` lives inside an interior-mutable slot so
/// [`ListItem::merge`] can retarget it through a `Weak<ListDefInner>`.
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
    /// RPython `ListDef.__init__(bookkeeper=None, s_item, mutated=False,
    /// resized=False)` (listdef.py:125-130) — `bookkeeper=None` path.
    /// Created listitems are marked `dont_change_any_more = True`.
    pub fn new(s_item: SomeValue) -> Self {
        Self::construct(ListItem::new(s_item), false, false)
    }

    /// RPython `ListDef.__init__(bookkeeper=<live>, s_item, …)` path.
    /// The resulting `ListItem` is mutable — `dont_change_any_more`
    /// stays False so the bookkeeper-aware merge path succeeds.
    pub fn mutable(s_item: SomeValue) -> Self {
        Self::construct(ListItem::with_bookkeeper(s_item), false, false)
    }

    /// RPython variant that sets `mutated` / `resized` at construction
    /// time (used by `s_list_of_strings` etc.). Uses the `None`
    /// bookkeeper path to match upstream.
    pub fn with_flags(s_item: SomeValue, mutated: bool, resized: bool) -> Self {
        Self::construct(ListItem::new(s_item), mutated, resized)
    }

    fn construct(mut item: ListItem, mutated: bool, resized: bool) -> Self {
        item.mutated = mutated || resized;
        item.resized = resized;
        let li = Rc::new(RefCell::new(item));
        let inner = Rc::new(ListDefInner {
            listitem: RefCell::new(li.clone()),
        });
        li.borrow_mut().itemof.push(Rc::downgrade(&inner));
        ListDef { inner }
    }

    /// RPython `ListDef.same_as(other)` (listdef.py:136-137).
    ///
    /// `self.listitem is other.listitem` — identity, not structural
    /// equality.
    pub fn same_as(&self, other: &ListDef) -> bool {
        let a = self.inner.listitem.borrow();
        let b = other.inner.listitem.borrow();
        Rc::ptr_eq(&*a, &*b)
    }

    /// RPython `ListDef.union(other)` (listdef.py:139-141).
    ///
    /// Merges `other`'s `ListItem` into `self`'s, patching every
    /// sister `ListDef` (including `other` itself) to share the
    /// merged cell. Returns `self` for upstream method-chaining.
    pub fn union_with(&self, other: &ListDef) -> Result<(), UnionError> {
        let self_li = self.inner.listitem.borrow().clone();
        let other_li = other.inner.listitem.borrow().clone();
        let _ = merge_items(&self_li, &other_li)?;
        Ok(())
    }

    /// RPython `ListDef.s_value` accessor — shortcut to the current
    /// `ListItem`'s element annotation.
    pub fn s_value(&self) -> SomeValue {
        self.inner.listitem.borrow().borrow().s_value.clone()
    }

    /// RPython `ListDef.mutate()` / `resize()` / `never_resize()`
    /// (listdef.py:182-192). Full port lands with Phase 5's
    /// bookkeeper-aware annotator; included as forwarders for now.
    pub fn mutate(&self) -> Result<(), TooLateForChange> {
        let li = self.inner.listitem.borrow().clone();
        let mut li_mut = li.borrow_mut();
        li_mut.mutate()
    }
}

impl PartialEq for ListDef {
    // RPython `SomeList.__eq__` at model.py:339-348 calls
    // `listdef.same_as(other.listdef)`. Mirror the identity semantics
    // directly on [`ListDef`] so any downstream `derive(PartialEq)`
    // picks up the correct behaviour.
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

/// RPython `ListItem.merge(other)` (listdef.py:58-98) — free function
/// form so `DictKey` / `DictValue` can reuse the core merge loop.
pub(crate) fn merge_items(
    self_li: &Rc<RefCell<ListItem>>,
    other_li: &Rc<RefCell<ListItem>>,
) -> Result<Rc<RefCell<ListItem>>, UnionError> {
    // upstream: `if self is not other`.
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

    // upstream dont_change_any_more dance (listdef.py:63-71).
    // If other is final but self isn't, swap so the final one is
    // `self` — merges into it then raise TooLateForChange if it widens.
    let (driver_li, folded_li) = {
        let self_b = self_li.borrow();
        let other_b = other_li.borrow();
        if other_b.dont_change_any_more && !self_b.dont_change_any_more {
            (other_li, self_li)
        } else {
            (self_li, other_li)
        }
    };

    // Snapshot flags / s_value / itemof before mutating.
    let (folded_s_value, folded_itemof, folded_flags) = {
        let folded_b = folded_li.borrow();
        (
            folded_b.s_value.clone(),
            folded_b.itemof.clone(),
            (
                folded_b.mutated,
                folded_b.resized,
                folded_b.immutable,
                folded_b.must_not_resize,
                folded_b.range_step,
            ),
        )
    };

    let (driver_s_value, driver_range_step) = {
        let driver_b = driver_li.borrow();
        (driver_b.s_value.clone(), driver_b.range_step)
    };

    // Union element types (may recurse but never takes a side-effect-
    // free guard — the outer caller already decided whether to mutate).
    let new_s_value = super::model::union(&driver_s_value, &folded_s_value)?;
    let widens = new_s_value != driver_s_value;

    // upstream: `if s_new_value != s_value: if self.dont_change_any_more:
    //                raise TooLateForChange`.
    if widens && driver_li.borrow().dont_change_any_more {
        return Err(UnionError {
            lhs: driver_s_value,
            rhs: folded_s_value,
            msg: "TooLateForChange on dont_change_any_more ListItem".into(),
        });
    }

    // upstream flag merges (listdef.py:73-85).
    {
        let mut driver_mut = driver_li.borrow_mut();
        driver_mut.immutable &= folded_flags.2;
        if folded_flags.3 {
            // other.must_not_resize — upstream raises
            // ListChangeUnallowed if self.resized.
            if driver_mut.resized {
                return Err(UnionError {
                    lhs: driver_mut.s_value.clone(),
                    rhs: folded_s_value,
                    msg: "ListChangeUnallowed: list merge with a resized".into(),
                });
            }
            driver_mut.must_not_resize = true;
        }
        if folded_flags.0 {
            let _ = driver_mut.mutate();
        }
        if folded_flags.1 {
            let _ = driver_mut.resize();
        }
        if folded_flags.4 != driver_range_step {
            driver_mut.range_step = driver_range_step.merge(folded_flags.4);
        }
        // upstream: `self.itemof.update(other.itemof)`.
        driver_mut.itemof.extend(folded_itemof.iter().cloned());
        if widens {
            driver_mut.s_value = new_s_value;
        }
    }

    // upstream: `self.patch()` — retarget every ListDefInner in the
    // just-extended itemof to point at self.
    for weak in &folded_itemof {
        if let Some(inner) = weak.upgrade() {
            *inner.listitem.borrow_mut() = driver_li.clone();
        }
    }

    Ok(driver_li.clone())
}

/// RPython `s_list_of_strings = SomeList(ListDef(None,
/// SomeString(no_nul=True), resized=True))` (listdef.py:206-207).
pub fn s_list_of_strings() -> super::model::SomeList {
    super::model::SomeList::new(ListDef::with_flags(
        super::model::SomeValue::String(super::model::SomeString::new(false, true)),
        false,
        true,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::model::{SomeInteger, SomeValue};
    use super::*;

    #[test]
    fn same_as_identity_preserved_across_clone() {
        let a = ListDef::new(SomeValue::Integer(SomeInteger::default()));
        let b = a.clone();
        assert!(a.same_as(&b));
    }

    #[test]
    fn distinct_listdefs_are_not_same_as() {
        let a = ListDef::new(SomeValue::Integer(SomeInteger::default()));
        let b = ListDef::new(SomeValue::Integer(SomeInteger::default()));
        assert!(!a.same_as(&b));
    }

    #[test]
    fn union_with_shares_listitem_after_merge() {
        let a = ListDef::new(SomeValue::Integer(SomeInteger::new(true, false)));
        let b = ListDef::new(SomeValue::Integer(SomeInteger::new(false, false)));
        assert!(!a.same_as(&b));

        // By default both listdefs are dont_change_any_more = true
        // (bookkeeper=None). Allow the merge by clearing the flag.
        a.inner.listitem.borrow().borrow_mut().dont_change_any_more = false;
        b.inner.listitem.borrow().borrow_mut().dont_change_any_more = false;

        a.union_with(&b).expect("merge must succeed");
        assert!(a.same_as(&b));
    }

    #[test]
    fn merge_refuses_under_side_effect_free_guard() {
        let a = ListDef::new(SomeValue::Integer(SomeInteger::new(true, false)));
        let b = ListDef::new(SomeValue::Integer(SomeInteger::new(false, false)));

        a.inner.listitem.borrow().borrow_mut().dont_change_any_more = false;
        b.inner.listitem.borrow().borrow_mut().dont_change_any_more = false;

        let _guard = SideEffectFreeGuard::enter();
        assert!(a.union_with(&b).is_err());
        // Without merging, same_as stays false.
        assert!(!a.same_as(&b));
    }

    #[test]
    fn dont_change_any_more_merge_widening_errors() {
        // upstream: merging two final listitems with different element
        // types triggers TooLateForChange.
        let a = ListDef::new(SomeValue::Integer(SomeInteger::new(true, false)));
        let b = ListDef::new(SomeValue::Integer(SomeInteger::new(false, false)));
        // Both stay dont_change_any_more = true (bookkeeper=None path).
        let err = a
            .union_with(&b)
            .expect_err("widening final list must error");
        assert!(err.msg.contains("TooLateForChange"));
    }
}
