//! `rpython/jit/metainterp/quasiimmut.py` — the loop-invalidation half of an
//! `_immutable_fields_` entry spelled with a `?`.
//!
//! Upstream has one `QuasiImmut` class serving every quasi-immutable field: the
//! rtyper synthesises a hidden `mutate_<name>` pointer field per declaration
//! (`get_mutate_field_name`), `get_current_qmut_instance` fills it in on the
//! first registration, and the invalidation function the rtyper installs on the
//! real field's write path nulls it and sweeps it. Pyre has no rtyper, so
//! [`QuasiImmutField`] is that hidden field written out — one per `?`
//! declaration, embedded in the owner the declaration belongs to, and reached
//! by the record-time resolver in `pyre-jit-trace::state quasi_immut_descr`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

/// `quasiimmut.py:54-110 QuasiImmut` — the loops that baked one quasi-immutable
/// field's value as a constant, and must be revoked when it changes.
///
/// The flag stands in for upstream's `looptoken` + `cpu.invalidate_loop`: the
/// backend already routes `GUARD_NOT_INVALIDATED` through a per-artifact
/// `AtomicBool`, so setting it is what `looptoken.invalidated = True` buys.
///
/// Upstream reaches an instance twice — `QuasiImmutDescr.__init__` binds it
/// while recording (`pyjitpl.py:1081`) and `compile.py:204-207` registers the
/// finished loop on that same object — so the instance outlives the field slot
/// that published it. It is shared through an [`Arc`] here for that reason, so
/// the methods take `&self` and reach the list through a lock.
pub struct QuasiImmut {
    tokens: parking_lot::Mutex<LoopTokens>,
    /// Whether [`QuasiImmutField::invalidate`] has already unlinked and swept
    /// this instance. Upstream asks the same question by identity —
    /// `quasiimmut.py:153 if qmut is not self.qmut` compares the recorded
    /// instance against the field's current one, and the two differ exactly
    /// when the recorded one was unlinked, since a fresh instance is only ever
    /// minted into a nulled field.
    unlinked: AtomicBool,
}

/// The list `quasiimmut.py:59-63 __init__` sets up, behind the lock that keeps
/// a registration and a sweep from interleaving.
struct LoopTokens {
    /// `quasiimmut.py:61-63` — weak so a retired loop drops out instead of
    /// being kept alive by the object whose field it read.
    looptokens_wrefs: Vec<std::sync::Weak<AtomicBool>>,
    /// `quasiimmut.py:56 compress_limit = 30`.
    compress_limit: usize,
}

impl LoopTokens {
    /// `quasiimmut.py:77-82 compress_looptokens_list` — drop the entries whose
    /// loop is gone and re-derive the limit from what is left, so an object that
    /// is recompiled against many times and never mutated cannot grow an
    /// unbounded list.
    ///
    /// Upstream's note that already-invalidated tokens must be kept applies
    /// here too: the flag stays live while its artifact does, and re-flipping
    /// an already-set flag is what keeps a multiply-invalidated loop revoked.
    fn compress(&mut self) {
        self.looptokens_wrefs.retain(|w| w.strong_count() > 0);
        self.compress_limit = (self.looptokens_wrefs.len() + 15) * 2;
    }
}

impl Default for QuasiImmut {
    fn default() -> Self {
        Self::new()
    }
}

impl QuasiImmut {
    /// `quasiimmut.py:59-63 __init__`. The initial limit is the growth formula
    /// in [`LoopTokens::compress`] evaluated at length zero.
    pub fn new() -> Self {
        Self {
            tokens: parking_lot::Mutex::new(LoopTokens {
                looptokens_wrefs: Vec::new(),
                compress_limit: 30,
            }),
            unlinked: AtomicBool::new(false),
        }
    }

    /// `quasiimmut.py:151-153` — whether this instance is still the one the
    /// field publishes, which is what `is_still_valid_for` tests before a loop
    /// that folded the field is allowed to stand.
    pub fn is_current(&self) -> bool {
        !self.unlinked.load(Ordering::Acquire)
    }

    /// `quasiimmut.py:72-75 register_loop_token`.
    ///
    /// A caller holding a recorded instance can reach here after the sweep:
    /// upstream cannot, because the GIL spans its optimize-and-compile, but
    /// pyre's runs with other threads live. The loop is then born invalid —
    /// its `GUARD_NOT_INVALIDATED` has to fail on the first entry, since the
    /// value it baked is the pre-mutation one and no later sweep will reach a
    /// list this instance has already emptied.
    pub fn register_loop_token(&self, flag: &Arc<AtomicBool>) {
        let mut tokens = self.tokens.lock();
        if self.unlinked.load(Ordering::Acquire) {
            flag.store(true, Ordering::Release);
            return;
        }
        if tokens.looptokens_wrefs.len() > tokens.compress_limit {
            tokens.compress();
        }
        tokens.looptokens_wrefs.push(Arc::downgrade(flag));
    }

    /// `quasiimmut.py:84-110 invalidate` — every loop recorded here becomes
    /// invalid, so each `GUARD_NOT_INVALIDATED` in it (and in its bridges) must
    /// now fail. The list is emptied like upstream's `self.looptokens_wrefs =
    /// []`, and the instance is marked unlinked under the same lock so a
    /// registration that arrives afterwards cannot be dropped on the floor.
    ///
    /// Taking the list is what bounds the walk. Keeping the live entries would
    /// be doubly wrong: the flag is already `true`, so re-storing it does
    /// nothing, and the list would only ever grow — one entry per compiled loop
    /// that folded a field read. A module-level `except X as e:` runs `del e`
    /// every iteration and `delitem` calls `mutated()`, so retaining would make
    /// each iteration walk every loop ever compiled against the object:
    /// O(compiled loops) per store, and the JIT compiling more loops would make
    /// the interpreter slower.
    pub fn invalidate(&self) {
        let wrefs = {
            let mut tokens = self.tokens.lock();
            self.unlinked.store(true, Ordering::Release);
            std::mem::take(&mut tokens.looptokens_wrefs)
        };
        for wref in wrefs {
            if let Some(flag) = wref.upgrade() {
                flag.store(true, Ordering::Release);
            }
        }
    }

    /// Number of entries still recorded, for tests.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.tokens.lock().looptokens_wrefs.len()
    }

    /// The current growth limit, for tests.
    #[cfg(test)]
    pub(crate) fn compress_limit(&self) -> usize {
        self.tokens.lock().compress_limit
    }
}

/// The hidden `mutate_<name>` field the rtyper synthesises for one `?`
/// declaration (`quasiimmut.py get_mutate_field_name`), written out because
/// pyre has no rtyper to synthesise it.
///
/// Null until the first read is recorded (`get_current_qmut_instance`,
/// quasiimmut.py:17-27); nulled by the invalidation function
/// (`make_invalidation_function._invalidate_now`, quasiimmut.py:33-38), so
/// the next recording starts from a fresh instance and the identity a
/// revalidation compares really did change. The stored pointer is one strong
/// [`Arc`] reference, so an instance a trace is still holding survives being
/// unlinked here — which is what lets a compile register on the instance its
/// recording resolved.
///
/// The pointer is the single source of truth, exactly as upstream's field is.
/// The lock is the nogil adaptation: upstream gets the null-then-sweep pair for
/// free from the GIL, while pyre runs Python threads on real OS threads, so
/// without it two mutators unlink one instance twice. Only the paths that
/// publish or unpublish take it — [`Self::is_installed`] deliberately does not,
/// because that bare test is the whole cost a mutation on an object no loop
/// watches has to pay.
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

    /// Whether any read has been recorded since the last invalidation — the
    /// `if not qmut_ptr` test that guards `_invalidate_now`'s body
    /// (quasiimmut.py:41). Lock-free so it can stay inside a trace.
    #[inline]
    pub fn is_installed(&self) -> bool {
        !self.ptr.load(Ordering::Acquire).is_null()
    }

    /// `quasiimmut.py:17-27 get_current_qmut_instance` — the field's instance,
    /// created when it is still null, handed back so the caller can hold it.
    ///
    /// Upstream calls this while RECORDING the read: `pyjitpl.py:1081` builds a
    /// `QuasiImmutDescr`, whose `__init__` `bh_setfield_gc_r`s a fresh instance
    /// into the hidden field (`quasiimmut.py:26`) and keeps it as
    /// `self.qmut`. That binding is what `heap.py:818 is_still_valid_for` and
    /// `compile.py:204-207 register_loop_token` both read later; pyre returns it
    /// for the same two consumers.
    pub fn get_current_qmut_instance(&self) -> Arc<QuasiImmut> {
        let _guard = self.lock.lock();
        let qmut_ptr = self.ptr.load(Ordering::Acquire);
        if qmut_ptr.is_null() {
            let fresh = Arc::new(QuasiImmut::new());
            let raw = Arc::into_raw(fresh).cast_mut();
            // One reference stays in the field, one goes to the caller.
            unsafe { Arc::increment_strong_count(raw) };
            self.ptr.store(raw, Ordering::Release);
            return unsafe { Arc::from_raw(raw) };
        }
        // Safe: the field holds a strong reference for as long as the pointer
        // is published, and it is only unpublished under the same lock.
        unsafe {
            Arc::increment_strong_count(qmut_ptr);
            Arc::from_raw(qmut_ptr)
        }
    }

    /// `quasiimmut.py:33-38 make_invalidation_function._invalidate_now` —
    /// unlink the instance, then flip every loop flag it recorded.
    ///
    /// Re-reads the pointer under the lock rather than trusting an earlier
    /// [`Self::is_installed`]: two threads mutating the same object both see it
    /// installed, and only the one that wins the swap gets the reference.
    pub fn invalidate(&self) {
        let Some(qmut) = self.take() else {
            return;
        };
        qmut.invalidate();
    }

    /// Unlink the instance and hand the field's reference to the caller, who
    /// drops it. [`Self::invalidate`] sweeps first; a swept owner's destructor
    /// calls it on its own to release an instance that was never invalidated.
    ///
    /// The reference, not the allocation: a recording still holding this
    /// instance keeps it alive, so an owner reclaimed mid-compile cannot leave
    /// the compiler registering into freed memory.
    pub fn take(&self) -> Option<Arc<QuasiImmut>> {
        let qmut_ptr = {
            let _guard = self.lock.lock();
            self.ptr.swap(std::ptr::null_mut(), Ordering::AcqRel)
        };
        if qmut_ptr.is_null() {
            return None;
        }
        // Safe: the swap is what transfers the field's reference, and it
        // happens once.
        Some(unsafe { Arc::from_raw(qmut_ptr) })
    }
}

impl Drop for QuasiImmutField {
    fn drop(&mut self) {
        let qmut_ptr = *self.ptr.get_mut();
        if !qmut_ptr.is_null() {
            drop(unsafe { Arc::from_raw(qmut_ptr) });
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

    /// `quasiimmut.py:84-110 invalidate` flips every registered flag and empties
    /// the list, and a flag whose loop is gone is simply skipped.
    #[test]
    fn invalidate_flips_live_flags_and_clears() {
        let qi = QuasiImmut::new();
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
        let qi = QuasiImmut::new();
        assert_eq!(qi.compress_limit(), 30, "quasiimmut.py:56 compress_limit");
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
    /// recording starts from a fresh instance.
    #[test]
    fn invalidate_unlinks_before_sweeping() {
        let field = QuasiImmutField::new();
        assert!(!field.is_installed());

        let flag = Arc::new(AtomicBool::new(false));
        field.get_current_qmut_instance().register_loop_token(&flag);
        assert!(field.is_installed());

        field.invalidate();
        assert!(flag.load(Ordering::Acquire));
        assert!(
            !field.is_installed(),
            "the field is nulled before the sweep"
        );

        // A second invalidation with nothing installed must not double-free.
        field.invalidate();

        let flag2 = Arc::new(AtomicBool::new(false));
        field
            .get_current_qmut_instance()
            .register_loop_token(&flag2);
        assert!(field.is_installed());
        field.invalidate();
        assert!(flag2.load(Ordering::Acquire));
    }

    /// `quasiimmut.py:151-153 if qmut is not self.qmut` — a recording holds the
    /// instance it resolved, so an invalidation that lands before the compile
    /// is visible on that handle rather than being hidden by the field having
    /// already minted a replacement.
    #[test]
    fn a_recorded_instance_reports_a_sweep_and_is_not_reused() {
        let field = QuasiImmutField::new();
        let recorded = field.get_current_qmut_instance();
        assert!(recorded.is_current());

        field.invalidate();
        assert!(!recorded.is_current(), "the recorded instance was swept");

        // The field mints a fresh instance rather than republishing the one the
        // recording still holds.
        let fresh = field.get_current_qmut_instance();
        assert!(fresh.is_current());
        assert!(!Arc::ptr_eq(&recorded, &fresh));

        // Registering on the swept instance cannot be silently lost: the loop
        // folded a pre-mutation value, so it is born invalid.
        let flag = Arc::new(AtomicBool::new(false));
        recorded.register_loop_token(&flag);
        assert!(flag.load(Ordering::Acquire));
        assert_eq!(recorded.len(), 0, "a swept list must not grow again");
    }

    /// The premise the sweep hooks rest on. `property_destructor` and its two
    /// twins in `pyre-jit/src/eval.rs` run when the collector reclaims an owner,
    /// which can happen while a compile that recorded a read of its field is
    /// still in flight. Taking the field's reference back must not disturb that
    /// compile's own.
    ///
    /// This is what carrying the instance rather than re-resolving `(owner,
    /// field index)` bought: the older shape handed the compiler an address to
    /// resolve later, so reclaiming here would have been a use-after-free.
    #[test]
    fn a_swept_owner_hands_back_only_its_own_reference() {
        let field = QuasiImmutField::new();
        // What recording bound onto the marker descr.
        let recorded = field.get_current_qmut_instance();
        assert_eq!(Arc::strong_count(&recorded), 2, "the field's and ours");

        // The collector sweeps the owner and the destructor runs.
        drop(field.take());
        assert_eq!(
            Arc::strong_count(&recorded),
            1,
            "only the compile's is left"
        );

        // The compile can still finish against it. Reclamation is not
        // revocation: nothing invalidated the field, so the instance is current
        // and the registration is recorded rather than born-invalid.
        assert!(recorded.is_current());
        let flag = Arc::new(AtomicBool::new(false));
        recorded.register_loop_token(&flag);
        assert!(!flag.load(Ordering::Acquire));
        assert_eq!(recorded.len(), 1);
    }

    /// Recording runs on one thread while any other Python thread can be
    /// mutating the same object. Upstream is safe here only because of the GIL;
    /// pyre has none, so the get-or-create, the push and the unlink have to be
    /// serialised or this races the instance to a double free.
    ///
    /// Reverting [`QuasiImmutField::take`] to an unlocked load/store aborts this
    /// with heap corruption inside `Vec`.
    #[test]
    fn publication_is_serialised_across_threads() {
        const ROUNDS: usize = 200_000;

        let field = QuasiImmutField::new();
        let stop = AtomicBool::new(false);

        std::thread::scope(|scope| {
            // Two recording threads keep re-installing the instance, so the
            // mutators below keep finding one to sweep.
            for _ in 0..2 {
                let field = &field;
                let stop = &stop;
                scope.spawn(move || {
                    let flag = Arc::new(AtomicBool::new(false));
                    while !stop.load(Ordering::Relaxed) {
                        field.get_current_qmut_instance().register_loop_token(&flag);
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
