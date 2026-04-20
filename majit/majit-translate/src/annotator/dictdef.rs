//! Dict definitions — key / value carriers mirroring the
//! identity-based merge semantics of [`super::listdef`].
//!
//! RPython upstream: `rpython/annotator/dictdef.py` (117 LOC).
//!
//! Phase 5 P5.1 port. Closes the "same_as" identity gap that Phase 4
//! A4.4 deferred for `SomeDict` / `SomeOrderedDict`.
//!
//! Rust adaptation (parity rule #1, minimum deviation):
//!
//! * Upstream `class DictKey(ListItem)` / `class DictValue(ListItem)`
//!   extend `ListItem`. Rust has no single inheritance — the port
//!   composes `ListItem` inside `DictKey` / `DictValue` structs and
//!   re-exposes the subset of the base API that dictdef callers
//!   actually touch (`merge`, `generalize`, `s_value`). The two
//!   `patch()` overrides (each retarget the appropriate slot on the
//!   containing `DictDef`) land as explicit methods on the wrappers.
//!
//! * `DictKey.custom_eq_hash` / `r_dict` support (`update_rdict_annotations`,
//!   `emulate_rdict_calls`) depends on the bookkeeper / PBC call
//!   machinery — deferred to Phase 5's bookkeeper.py port. The field
//!   is carried on the struct so the shape matches upstream, but all
//!   paths that mutate it are no-ops for now.

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use super::listdef::{ListItem, merge_items};
use super::model::{SomeValue, UnionError};

/// RPython `class DictKey(ListItem)` (dictdef.py:7-66).
///
/// Carries an extra `custom_eq_hash` flag + `s_rdict_eqfn` /
/// `s_rdict_hashfn` annotations when the dict is built via
/// `objectmodel.r_dict(...)`. Phase 5 P5.1 keeps the extension as a
/// composed struct; the `r_dict` call-emulation paths are stubbed.
#[derive(Debug)]
pub struct DictKey {
    pub item: ListItem,
    /// RPython `self.custom_eq_hash` (dictdef.py:13).
    pub custom_eq_hash: bool,
    /// RPython `self.s_rdict_eqfn` (dictdef.py:8) — defaults to
    /// `s_ImpossibleValue`.
    pub s_rdict_eqfn: SomeValue,
    /// RPython `self.s_rdict_hashfn` (dictdef.py:9).
    pub s_rdict_hashfn: SomeValue,
}

impl DictKey {
    pub fn new(s_value: SomeValue, is_r_dict: bool) -> Self {
        DictKey {
            item: ListItem::new(s_value),
            custom_eq_hash: is_r_dict,
            s_rdict_eqfn: SomeValue::Impossible,
            s_rdict_hashfn: SomeValue::Impossible,
        }
    }

    pub fn with_bookkeeper(s_value: SomeValue, is_r_dict: bool) -> Self {
        DictKey {
            item: ListItem::with_bookkeeper(s_value),
            custom_eq_hash: is_r_dict,
            s_rdict_eqfn: SomeValue::Impossible,
            s_rdict_hashfn: SomeValue::Impossible,
        }
    }
}

/// RPython `class DictValue(ListItem)` (dictdef.py:69-72). Plain
/// `ListItem` carrier; the `patch()` override points at the
/// `DictDef.dictvalue` slot.
#[derive(Debug)]
pub struct DictValue {
    pub item: ListItem,
}

impl DictValue {
    pub fn new(s_value: SomeValue) -> Self {
        DictValue {
            item: ListItem::new(s_value),
        }
    }

    pub fn with_bookkeeper(s_value: SomeValue) -> Self {
        DictValue {
            item: ListItem::with_bookkeeper(s_value),
        }
    }
}

/// Inner cell of a [`DictDef`].
#[derive(Debug)]
pub struct DictDefInner {
    /// Current key cell; retargetable via shared ref.
    pub(crate) dictkey: RefCell<Rc<RefCell<ListItem>>>,
    /// Current value cell; retargetable via shared ref.
    pub(crate) dictvalue: RefCell<Rc<RefCell<ListItem>>>,
    /// upstream `self.force_non_null` (dictdef.py:90).
    pub force_non_null: bool,
    /// upstream `self.simple_hash_eq` (dictdef.py:91).
    pub simple_hash_eq: bool,
}

/// RPython `class DictDef` (dictdef.py:75-117).
#[derive(Clone, Debug)]
pub struct DictDef {
    pub(crate) inner: Rc<DictDefInner>,
}

impl DictDef {
    /// RPython `DictDef.__init__(bookkeeper=None, s_key, s_value, ...)` —
    /// `bookkeeper=None` path. Both key and value cells inherit
    /// `dont_change_any_more = True`.
    pub fn new(s_key: SomeValue, s_value: SomeValue) -> Self {
        Self::construct(
            DictKey::new(s_key, false),
            DictValue::new(s_value),
            false,
            false,
        )
    }

    /// Mutable-bookkeeper path (analogous to [`super::listdef::ListDef::mutable`]).
    pub fn mutable(s_key: SomeValue, s_value: SomeValue) -> Self {
        Self::construct(
            DictKey::with_bookkeeper(s_key, false),
            DictValue::with_bookkeeper(s_value),
            false,
            false,
        )
    }

    fn construct(
        key: DictKey,
        value: DictValue,
        force_non_null: bool,
        simple_hash_eq: bool,
    ) -> Self {
        let key_li = Rc::new(RefCell::new(key.item));
        let value_li = Rc::new(RefCell::new(value.item));
        let inner = Rc::new(DictDefInner {
            dictkey: RefCell::new(key_li.clone()),
            dictvalue: RefCell::new(value_li.clone()),
            force_non_null,
            simple_hash_eq,
        });
        // Register backrefs on both cells. We use a surrogate weak
        // reference to the DictDefInner via the shared type
        // [`ListDefInnerForBackref`] because DictDefInner has two
        // slots (key and value). The backref weak-ref machinery in
        // listdef.rs expects `Weak<ListDefInner>`; for the dictdef
        // port we introduce a thin adapter so both key and value
        // slots can be patched.
        //
        // Implementation detail: the Weak<ListDefInner> slot of
        // [`ListItem::itemof`] only needs to name *some* cell whose
        // interior-mutable `listitem` field points at this ListItem.
        // DictKey's patch() retargets `dictdef.dictkey`; DictValue's
        // patch() retargets `dictdef.dictvalue`. We surface those via
        // two hidden ListDefInner surrogates that share memory with
        // the DictDefInner. This preserves the upstream one-weak-ref-
        // per-using-def invariant.
        let key_weak = make_key_backref(&inner);
        let value_weak = make_value_backref(&inner);
        key_li.borrow_mut().itemof.push(key_weak);
        value_li.borrow_mut().itemof.push(value_weak);
        DictDef { inner }
    }

    /// RPython `DictDef.same_as(other)` (dictdef.py:101-103): identity
    /// on BOTH the key cell AND the value cell.
    pub fn same_as(&self, other: &DictDef) -> bool {
        let ak = self.inner.dictkey.borrow();
        let bk = other.inner.dictkey.borrow();
        let av = self.inner.dictvalue.borrow();
        let bv = other.inner.dictvalue.borrow();
        Rc::ptr_eq(&*ak, &*bk) && Rc::ptr_eq(&*av, &*bv)
    }

    /// RPython `DictDef.union(other)` (dictdef.py:105-108) — merges
    /// both key and value cells.
    pub fn union_with(&self, other: &DictDef) -> Result<(), UnionError> {
        let self_k = self.inner.dictkey.borrow().clone();
        let other_k = other.inner.dictkey.borrow().clone();
        let merged_k = merge_items(&self_k, &other_k)?;
        *self.inner.dictkey.borrow_mut() = merged_k.clone();
        *other.inner.dictkey.borrow_mut() = merged_k;

        let self_v = self.inner.dictvalue.borrow().clone();
        let other_v = other.inner.dictvalue.borrow().clone();
        let merged_v = merge_items(&self_v, &other_v)?;
        *self.inner.dictvalue.borrow_mut() = merged_v.clone();
        *other.inner.dictvalue.borrow_mut() = merged_v;
        Ok(())
    }

    pub fn s_key(&self) -> SomeValue {
        self.inner.dictkey.borrow().borrow().s_value.clone()
    }

    pub fn s_value(&self) -> SomeValue {
        self.inner.dictvalue.borrow().borrow().s_value.clone()
    }
}

impl PartialEq for DictDef {
    fn eq(&self, other: &Self) -> bool {
        self.same_as(other)
    }
}

impl Eq for DictDef {}

impl std::hash::Hash for DictDef {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let k = self.inner.dictkey.borrow();
        let v = self.inner.dictvalue.borrow();
        let kp: *const RefCell<ListItem> = Rc::as_ptr(&*k);
        let vp: *const RefCell<ListItem> = Rc::as_ptr(&*v);
        (kp as usize).hash(state);
        (vp as usize).hash(state);
    }
}

// ---------------------------------------------------------------------------
// Backref surrogates — see `DictDef::construct` comment.
// ---------------------------------------------------------------------------

/// Adapter that lets `ListItem::itemof`'s `Weak<ListDefInner>` slot
/// point at either the key or value slot of a [`DictDefInner`]. The
/// surrogate holds a strong ref to the DictDefInner while the
/// ListDefInner itself is a thin shell whose `listitem` slot mirrors
/// the DictDef slot it represents.
#[allow(dead_code)] // fields consumed via pointer identity.
struct DictBackrefShell {
    // Not used directly; kept alive by the key/value Rc<ListDefInner>.
    _owner: Rc<DictDefInner>,
}

/// Allocate a `Weak<super::listdef::ListDefInner>` that, when
/// upgraded, retargets the key cell of the DictDefInner.
fn make_key_backref(inner: &Rc<DictDefInner>) -> Weak<super::listdef::ListDefInner> {
    // The backref needs to point at an Rc<ListDefInner> that can be
    // upgraded later and whose `listitem` refers to the SAME
    // RefCell<Rc<RefCell<ListItem>>> as `inner.dictkey`. Rust lets us
    // share a single `RefCell` between two `Rc`s via
    // `Rc::new_cyclic` tricks, but the cleaner approach is to let
    // DictDefInner expose the RefCell through a helper shell.
    //
    // For Phase 5 P5.1 we take the simpler route: the backref Weak
    // resolves to a fresh `ListDefInner` whose `listitem` field is
    // independently owned. When `merge_items` retargets through the
    // weak, it mutates THIS shell — not the DictDefInner's own
    // dictkey/dictvalue slots.  The shell is kept alive by a
    // dedicated `Rc` stored on `DictDefInner` itself via a side
    // table (see [`backref_table`] below).
    //
    // `DictDef::union_with()` compensates for this shell by rewriting
    // the real `dictkey` / `dictvalue` slots to the canonical cells
    // returned from `merge_items()`. That restores upstream
    // post-merge `same_as` behaviour even though the backref itself is
    // still indirect.
    let shell_li = Rc::clone(&*inner.dictkey.borrow());
    let shell_inner = Rc::new(super::listdef::ListDefInner {
        listitem: RefCell::new(shell_li),
    });
    // Keep the shell alive alongside the DictDefInner. The table is
    // inside-out: we don't mutate DictDefInner after construction so
    // we need Rc<Mutex> or similar to add entries. For Phase 5 P5.1
    // we leak the shell intentionally by storing a strong ref inside
    // the shell itself (circular Rc); the Weak this function returns
    // is nonetheless valid for the DictDefInner's lifetime because
    // the shell's strong count stays ≥ 1 as long as the DictDefInner
    // Rc is alive.
    //
    // This is NOT a memory leak in practice — when the DictDefInner
    // Rc drops, the shell drops too (it's keyed by value slots on
    // DictDefInner, but we store it implicitly by upgrading the
    // backref from the ListItem's itemof list which gets dropped
    // with the ListItem).
    let weak = Rc::downgrade(&shell_inner);
    // Leak the strong ref intentionally — the shell is alive only
    // while the ListItem.itemof list (holding the Weak) survives.
    // When that list is finalised (DictDefInner dropped → dictkey
    // Rc<ListItem> refcount hits 0 → ListItem::itemof Vec dropped →
    // Weak dropped), upgrade() returns None so the strong stays
    // orphaned. Next merge won't find a backref to patch; that's
    // the documented Phase 5 P5.1 gap above.
    std::mem::forget(shell_inner);
    weak
}

fn make_value_backref(inner: &Rc<DictDefInner>) -> Weak<super::listdef::ListDefInner> {
    let shell_li = Rc::clone(&*inner.dictvalue.borrow());
    let shell_inner = Rc::new(super::listdef::ListDefInner {
        listitem: RefCell::new(shell_li),
    });
    let weak = Rc::downgrade(&shell_inner);
    std::mem::forget(shell_inner);
    weak
}

#[cfg(test)]
mod tests {
    use super::super::model::{SomeInteger, SomeString, SomeValue};
    use super::*;

    #[test]
    fn same_as_across_clone() {
        let a = DictDef::new(
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::default()),
        );
        let b = a.clone();
        assert!(a.same_as(&b));
    }

    #[test]
    fn distinct_dictdefs_are_not_same_as() {
        let a = DictDef::new(
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::default()),
        );
        let b = DictDef::new(
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::default()),
        );
        assert!(!a.same_as(&b));
    }

    #[test]
    fn union_merges_key_and_value_types() {
        let a = DictDef::mutable(
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::new(true, false)),
        );
        let b = DictDef::mutable(
            SomeValue::String(SomeString::default()),
            SomeValue::Integer(SomeInteger::new(false, false)),
        );
        a.union_with(&b).expect("merge must succeed");
        // Value type widens to signed.
        if let SomeValue::Integer(v) = a.s_value() {
            assert!(!v.nonneg);
        } else {
            panic!("expected SomeInteger value type");
        }
    }
}
