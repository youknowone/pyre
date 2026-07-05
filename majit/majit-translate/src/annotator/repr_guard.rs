//! Shared recursion guard for `Debug` of the interior-mutable annotation
//! nodes ([`super::listdef::ListDef`] / [`super::dictdef::DictDef`]), whose
//! shared item cell can point back to a `SomeValue::List` / `Dict` that owns
//! them — a legal self-referential annotation such as `l = []; l.append(l)`.
//! Mirrors the thread-local `reprdict` guard in `SomeObject.__repr__`
//! (model.py:68-90): a node already being formatted renders as an elision
//! instead of recursing forever.

use std::cell::RefCell;

thread_local! {
    static REPR_GUARD: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// RAII token for `REPR_GUARD`. [`ReprGuard::enter`] returns `None` when
/// `id` is already on the stack (a cycle), signalling the caller to elide
/// rather than recurse; the pushed id is popped on drop.
pub(crate) struct ReprGuard(usize);

impl ReprGuard {
    pub(crate) fn enter(id: usize) -> Option<ReprGuard> {
        REPR_GUARD.with(|g| {
            let mut g = g.borrow_mut();
            if g.contains(&id) {
                None
            } else {
                g.push(id);
                Some(ReprGuard(id))
            }
        })
    }
}

impl Drop for ReprGuard {
    fn drop(&mut self) {
        REPR_GUARD.with(|g| {
            let mut g = g.borrow_mut();
            if let Some(pos) = g.iter().rposition(|&x| x == self.0) {
                g.remove(pos);
            }
        });
    }
}
