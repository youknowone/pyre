//! `rpython/jit/metainterp/quasiimmut.py` — the loop-invalidation half of an
//! `_immutable_fields_` entry spelled with a `?`.
//!
//! Upstream has one `QuasiImmut` class serving every quasi-immutable field: the
//! rtyper synthesises a hidden `mutate_<name>` pointer field per declaration
//! (`get_mutate_field_name`), `get_current_qmut_instance` fills it in on the
//! first registration, and the invalidation function the rtyper installs on the
//! real field's write path nulls it and sweeps it. Pyre has no rtyper, so
//! [`QuasiImmutField`] is that hidden field written out, and the two `?` fields
//! this tree declares — `W_TypeObject._version_tag`
//! (typeobject.py:177) and `ModuleDictStrategy.version` (celldict.py:34) —
//! share it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

/// `quasiimmut.py:55-109 QuasiImmut` — the loops that baked one quasi-immutable
/// field's value as a constant, and must be revoked when it changes.
///
/// The flag stands in for upstream's `looptoken` + `cpu.invalidate_loop`: the
/// backend already routes `GUARD_NOT_INVALIDATED` through a per-artifact
/// `AtomicBool`, so setting it is what `looptoken.invalidated = True` buys.
pub struct QuasiImmut {
    /// `quasiimmut.py:62-63` — weak so a retired loop drops out instead of
    /// being kept alive by the object whose field it read.
    looptokens_wrefs: Vec<std::sync::Weak<AtomicBool>>,
    /// `quasiimmut.py:57 compress_limit = 30`.
    compress_limit: usize,
}

impl Default for QuasiImmut {
    fn default() -> Self {
        Self::new()
    }
}

impl QuasiImmut {
    /// `quasiimmut.py:59-64 __init__`. The initial limit is the growth formula
    /// in [`Self::compress_looptokens_list`] evaluated at length zero.
    pub fn new() -> Self {
        Self {
            looptokens_wrefs: Vec::new(),
            compress_limit: 30,
        }
    }

    /// `quasiimmut.py:72-75 register_loop_token`.
    pub fn register_loop_token(&mut self, flag: &Arc<AtomicBool>) {
        if self.looptokens_wrefs.len() > self.compress_limit {
            self.compress_looptokens_list();
        }
        self.looptokens_wrefs.push(Arc::downgrade(flag));
    }

    /// `quasiimmut.py:77-82 compress_looptokens_list` — drop the entries whose
    /// loop is gone and re-derive the limit from what is left, so an object that
    /// is recompiled against many times and never mutated cannot grow an
    /// unbounded list.
    ///
    /// Upstream's note that already-invalidated tokens must be kept applies
    /// here too: the flag stays live while its artifact does, and re-flipping
    /// an already-set flag is what keeps a multiply-invalidated loop revoked.
    fn compress_looptokens_list(&mut self) {
        self.looptokens_wrefs.retain(|w| w.strong_count() > 0);
        self.compress_limit = (self.looptokens_wrefs.len() + 15) * 2;
    }

    /// `quasiimmut.py:84-109 invalidate` — every loop recorded here becomes
    /// invalid, so each `GUARD_NOT_INVALIDATED` in it (and in its bridges) must
    /// now fail. The list is emptied like upstream's `self.looptokens_wrefs =
    /// []`; the caller then drops the instance itself, so a loop that recompiles
    /// registers against a fresh one.
    ///
    /// Taking the list is what bounds the walk. Keeping the live entries would
    /// be doubly wrong: the flag is already `true`, so re-storing it does
    /// nothing, and the list would only ever grow — one entry per compiled loop
    /// that folded a field read. A module-level `except X as e:` runs `del e`
    /// every iteration and `delitem` calls `mutated()`, so retaining would make
    /// each iteration walk every loop ever compiled against the object:
    /// O(compiled loops) per store, and the JIT compiling more loops would make
    /// the interpreter slower.
    pub fn invalidate(&mut self) {
        for wref in std::mem::take(&mut self.looptokens_wrefs) {
            if let Some(flag) = wref.upgrade() {
                flag.store(true, Ordering::Release);
            }
        }
    }

    /// Number of entries still recorded, for tests.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.looptokens_wrefs.len()
    }

    /// The current growth limit, for tests.
    #[cfg(test)]
    pub(crate) fn compress_limit(&self) -> usize {
        self.compress_limit
    }
}

/// The hidden `mutate_<name>` field the rtyper synthesises for one `?`
/// declaration (`quasiimmut.py get_mutate_field_name`), written out because
/// pyre has no rtyper to synthesise it.
///
/// Null until the first loop registers (`get_current_qmut_instance`,
/// quasiimmut.py:116-126); nulled and dropped by the invalidation function
/// (`make_invalidation_function._invalidate_now`, quasiimmut.py:129-134), so
/// the next registration starts from a fresh instance and the identity a
/// revalidation compares really did change.
///
/// The pointer is the single source of truth, exactly as upstream's field is.
/// The lock is the nogil adaptation: upstream gets the null-then-sweep pair for
/// free from the GIL, while pyre runs Python threads on real OS threads, so
/// without it two mutators free one instance twice and a mutator can free the
/// instance a compiling thread is pushing into. Only the dereferencing paths
/// take it — [`Self::is_installed`] deliberately does not, because that bare
/// test is the whole cost a mutation on an object no loop watches has to pay.
///
/// Both owners are allocated non-moving (`try_gc_alloc_stable_raw` /
/// `malloc_typed`), so the lock cannot be remapped out from under a holder and
/// no address-striped indirection is needed. The critical section allocates
/// nothing GC-managed and crosses no safepoint, so it cannot park a mutator the
/// collector is waiting for.
pub struct QuasiImmutField {
    ptr: AtomicPtr<QuasiImmut>,
    lock: parking_lot::Mutex<()>,
}

impl Default for QuasiImmutField {
    fn default() -> Self {
        Self::new()
    }
}

impl QuasiImmutField {
    pub const fn new() -> Self {
        Self {
            ptr: AtomicPtr::new(std::ptr::null_mut()),
            lock: parking_lot::Mutex::new(()),
        }
    }

    /// Whether any loop has registered since the last invalidation — the
    /// `if not qmut_ptr` test that guards `_invalidate_now`'s body
    /// (quasiimmut.py:130). Lock-free so it can stay inside a trace.
    #[inline]
    pub fn is_installed(&self) -> bool {
        !self.ptr.load(Ordering::Acquire).is_null()
    }

    /// `quasiimmut.py:116-126 get_current_qmut_instance` followed by
    /// `:72-75 register_loop_token`: create the instance if the field is still
    /// null, then record this loop's invalidation flag on it.
    pub fn register_loop_token(&self, flag: &Arc<AtomicBool>) {
        let _guard = self.lock.lock();
        let mut qmut_ptr = self.ptr.load(Ordering::Acquire);
        if qmut_ptr.is_null() {
            qmut_ptr = Box::into_raw(Box::new(QuasiImmut::new()));
            self.ptr.store(qmut_ptr, Ordering::Release);
        }
        // Safe: the pointer is only ever cleared under the same lock, and the
        // instance is freed by whoever wins that clear.
        unsafe { (*qmut_ptr).register_loop_token(flag) };
    }

    /// `quasiimmut.py:129-134 make_invalidation_function._invalidate_now` —
    /// unlink the instance, then flip every loop flag it recorded.
    ///
    /// Re-reads the pointer under the lock rather than trusting an earlier
    /// [`Self::is_installed`]: two threads mutating the same object both see it
    /// installed, and only the one that wins the swap owns the [`Box`].
    pub fn invalidate(&self) {
        let Some(mut qmut) = self.take() else {
            return;
        };
        qmut.invalidate();
    }

    /// Unlink the instance and hand it to the caller. The owner's destructor
    /// uses this to reclaim a box that was never invalidated.
    pub fn take(&self) -> Option<Box<QuasiImmut>> {
        let qmut_ptr = {
            let _guard = self.lock.lock();
            self.ptr.swap(std::ptr::null_mut(), Ordering::AcqRel)
        };
        if qmut_ptr.is_null() {
            return None;
        }
        // Safe: the swap is what transfers ownership, and it happens once.
        Some(unsafe { Box::from_raw(qmut_ptr) })
    }
}

impl Drop for QuasiImmutField {
    fn drop(&mut self) {
        let qmut_ptr = *self.ptr.get_mut();
        if !qmut_ptr.is_null() {
            drop(unsafe { Box::from_raw(qmut_ptr) });
        }
    }
}

/// The residual half of an invalidation: unlink the instance and flip every
/// loop flag it recorded.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py:139`) for the
/// reason upstream's own walk is out of line — it hangs off the residual
/// `jit_force_quasi_immutable` path and never appears in a trace — and because
/// the lock and the `Vec` walk have no lowering. A free function taking a raw
/// pointer, because that is the shape `majit_macros` can emit a call target
/// for; the callers keep their [`QuasiImmutField::is_installed`] test traced,
/// so a mutation on an object no loop watches still makes no call.
///
/// # Safety
/// `field` must point at a live [`QuasiImmutField`].
#[majit_macros::dont_look_inside]
pub unsafe fn sweep_quasi_immut_field(field: *const QuasiImmutField) {
    unsafe { (*field).invalidate() };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `quasiimmut.py:84-109 invalidate` flips every registered flag and empties
    /// the list, and a flag whose loop is gone is simply skipped.
    #[test]
    fn invalidate_flips_live_flags_and_clears() {
        let mut qi = QuasiImmut::new();
        let live = Arc::new(AtomicBool::new(false));
        let dead = Arc::new(AtomicBool::new(false));
        qi.register_loop_token(&live);
        qi.register_loop_token(&dead);
        drop(dead);

        qi.invalidate();
        assert!(live.load(Ordering::Acquire), "a live loop must be revoked");
        assert_eq!(
            qi.len(),
            0,
            "invalidate empties the list; a recompile registers again",
        );

        // A second invalidate with nothing registered is a no-op, and the flag
        // stays set — `GUARD_NOT_INVALIDATED` has no un-set edge.
        qi.invalidate();
        assert!(live.load(Ordering::Acquire));
    }

    /// `quasiimmut.py:72-82` — an object recompiled against many times and never
    /// mutated must not grow an unbounded watcher list.
    #[test]
    fn register_compresses_dead_loop_tokens() {
        let mut qi = QuasiImmut::new();
        assert_eq!(qi.compress_limit(), 30, "quasiimmut.py:57 compress_limit");
        for _ in 0..500 {
            // Each "recompile" drops its artifact immediately, so every
            // registered weak ref is already dead by the next round.
            let flag = Arc::new(AtomicBool::new(false));
            qi.register_loop_token(&flag);
        }
        assert!(
            qi.len() <= qi.compress_limit() + 1,
            "compress must bound the list, got {} against limit {}",
            qi.len(),
            qi.compress_limit(),
        );
        // With every entry dead the limit collapses back to the empty-list
        // value rather than ratcheting upward.
        assert_eq!(qi.compress_limit(), 30);
    }

    /// `_invalidate_now` nulls the field before sweeping, so a later
    /// registration starts from a fresh instance.
    #[test]
    fn invalidate_unlinks_before_sweeping() {
        let field = QuasiImmutField::new();
        assert!(!field.is_installed());

        let flag = Arc::new(AtomicBool::new(false));
        field.register_loop_token(&flag);
        assert!(field.is_installed());

        field.invalidate();
        assert!(flag.load(Ordering::Acquire));
        assert!(
            !field.is_installed(),
            "the field is nulled before the sweep"
        );

        // A second invalidation with nothing installed must not double-free.
        field.invalidate();

        // A recompile registers against a fresh instance rather than reviving
        // the swept one.  Only the address would witness that directly, and the
        // allocator is free to hand the freed box straight back, so the
        // observable is that the new flag rides its own list.
        let flag2 = Arc::new(AtomicBool::new(false));
        field.register_loop_token(&flag2);
        assert!(field.is_installed());
        field.invalidate();
        assert!(flag2.load(Ordering::Acquire));
    }

    /// Registration runs on the compiling thread while any other Python thread
    /// can be mutating the same object. Upstream is safe here only because of
    /// the GIL; pyre has none, so the get-or-create, the push and the free have
    /// to be serialised or this races the instance to a double free.
    ///
    /// Reverting [`QuasiImmutField::take`] to an unlocked load/store aborts this
    /// with heap corruption inside `Vec`.
    #[test]
    fn publication_is_serialised_across_threads() {
        const ROUNDS: usize = 200_000;

        let field = QuasiImmutField::new();
        let stop = AtomicBool::new(false);

        std::thread::scope(|scope| {
            // Two compiling threads keep re-installing the instance, so the
            // mutators below keep finding one to free.
            for _ in 0..2 {
                let field = &field;
                let stop = &stop;
                scope.spawn(move || {
                    let flag = Arc::new(AtomicBool::new(false));
                    while !stop.load(Ordering::Relaxed) {
                        field.register_loop_token(&flag);
                    }
                });
            }
            let mutators: Vec<_> = (0..4)
                .map(|_| {
                    let field = &field;
                    scope.spawn(move || {
                        for _ in 0..ROUNDS {
                            field.invalidate();
                        }
                    })
                })
                .collect();
            for mutator in mutators {
                mutator.join().expect("mutator thread faulted");
            }
            stop.store(true, Ordering::Relaxed);
        });

        field.invalidate();
        assert!(!field.is_installed());
    }
}
