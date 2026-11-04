//! `rpython/jit/metainterp/quasiimmut.py`'s `QuasiImmut` class: the loop-token
//! list and its sweep. The hidden `mutate_<name>` word pair and `QuasiImmutField`
//! stay with the host object model.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// `quasiimmut.py QuasiImmut` — the loops that baked one quasi-immutable
/// field's value as a constant, and must be revoked when it changes.
///
/// Like upstream, the registry holds the owning loop token, not a particular
/// loop/bridge machine-code fragment.  Invalidating the token both removes it
/// from warm entry and activates every still-unpatched
/// `GUARD_NOT_INVALIDATED` generation owned by that token.
///
/// Upstream reaches an instance twice — `QuasiImmutDescr.__init__` binds it
/// while recording (`pyjitpl.py`) and `compile.py:204-207` registers the  allow-line-citation
/// finished loop on that same object — so the instance outlives the field slot
/// that published it. It is shared through an [`Arc`] here for that reason, so
/// the methods take `&self` and reach the list through a lock.
pub struct QuasiImmut {
    tokens: parking_lot::Mutex<LoopTokens>,
    /// Whether [`QuasiImmutField::invalidate`] has already unlinked and swept
    /// this instance. Upstream asks the same question by identity —
    /// `quasiimmut.py QuasiImmutDescr.is_still_valid_for if qmut is not self.qmut` compares the recorded
    /// instance against the field's current one, and the two differ exactly
    /// when the recorded one was unlinked, since a fresh instance is only ever
    /// minted into a nulled field.
    unlinked: AtomicBool,
}

/// The list `quasiimmut.py __init__` sets up, behind the lock that keeps
/// a registration and a sweep from interleaving.
struct LoopTokens {
    /// `quasiimmut.py:61-63` — weak so a retired loop drops out instead of  allow-line-citation
    /// being kept alive by the object whose field it read.
    looptokens_wrefs: Vec<std::sync::Weak<dyn crate::QuasiImmutLoopToken>>,
    /// `quasiimmut.py compress_limit = 30`.
    compress_limit: usize,
}

impl LoopTokens {
    /// `quasiimmut.py compress_looptokens_list` — drop the entries whose
    /// loop is gone and re-derive the limit from what is left, so an object that
    /// is recompiled against many times and never mutated cannot grow an
    /// unbounded list.
    ///
    /// Upstream's note that already-invalidated tokens must be kept applies
    /// here too: a repeated invalidation must reach the same token again so
    /// the backend can activate guards compiled since the previous call.
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
    /// `quasiimmut.py __init__`. The initial limit is the growth formula
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

    /// `quasiimmut.py:151-153` — whether this instance is still the one the  allow-line-citation
    /// field publishes, which is what `is_still_valid_for` tests before a loop
    /// that folded the field is allowed to stand.
    pub fn is_current(&self) -> bool {
        !self.unlinked.load(Ordering::Acquire)
    }

    /// `quasiimmut.py register_loop_token`.
    ///
    /// A caller holding a recorded instance can reach here after the sweep:
    /// upstream cannot, because the GIL spans its optimize-and-compile, but
    /// pyre's runs with other threads live. The loop is then born invalid —
    /// its `GUARD_NOT_INVALIDATED` has to fail on the first entry, since the
    /// value it baked is the pre-mutation one and no later sweep will reach a
    /// list this instance has already emptied.
    pub fn register_loop_token(&self, token: &Arc<dyn crate::QuasiImmutLoopToken>) {
        let mut tokens = self.tokens.lock();
        if self.unlinked.load(Ordering::Acquire) {
            token.invalidate_for_quasi_immut();
            return;
        }
        if tokens.looptokens_wrefs.len() > tokens.compress_limit {
            tokens.compress();
        }
        tokens.looptokens_wrefs.push(Arc::downgrade(token));
    }

    /// `quasiimmut.py invalidate` — every loop recorded here becomes
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
            if let Some(token) = wref.upgrade() {
                token.invalidate_for_quasi_immut();
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

#[cfg(test)]
#[derive(Debug)]
struct TestLoopToken {
    invalidated: Arc<AtomicBool>,
}

#[cfg(test)]
impl crate::QuasiImmutLoopToken for TestLoopToken {
    fn invalidate_for_quasi_immut(&self) {
        self.invalidated.store(true, Ordering::Release);
    }
}

#[cfg(test)]
fn test_loop_token(invalidated: &Arc<AtomicBool>) -> Arc<dyn crate::QuasiImmutLoopToken> {
    Arc::new(TestLoopToken {
        invalidated: invalidated.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `quasiimmut.py invalidate` flips every registered flag and empties
    /// the list, and a flag whose loop is gone is simply skipped.
    #[test]
    fn invalidate_flips_live_flags_and_clears() {
        let qi = QuasiImmut::new();
        let live = Arc::new(AtomicBool::new(false));
        let dead = Arc::new(AtomicBool::new(false));
        let live_token = test_loop_token(&live);
        let dead_token = test_loop_token(&dead);
        qi.register_loop_token(&live_token);
        qi.register_loop_token(&dead_token);
        drop(dead_token);

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

    /// `quasiimmut.py register_loop_token` — an object recompiled against many times and never
    /// mutated must not grow an unbounded watcher list.
    #[test]
    fn register_compresses_dead_loop_tokens() {
        let qi = QuasiImmut::new();
        assert_eq!(qi.compress_limit(), 30, "quasiimmut.py:56 compress_limit"); // allow-line-citation
        for _ in 0..500 {
            // Each "recompile" drops its artifact immediately, so every
            // registered weak ref is already dead by the next round.
            let flag = Arc::new(AtomicBool::new(false));
            let token = test_loop_token(&flag);
            qi.register_loop_token(&token);
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

    /// `quasiimmut.py QuasiImmutDescr.is_still_valid_for if qmut is not self.qmut` — a recording holds the
    /// instance it resolved, so an invalidation that lands before the compile
    /// is visible on that handle rather than being hidden by the field having
    /// already minted a replacement.
    #[test]
    fn a_recorded_instance_reports_a_sweep_and_is_not_reused() {
        let recorded = Arc::new(QuasiImmut::new());
        assert!(recorded.is_current());

        recorded.invalidate();
        assert!(!recorded.is_current(), "the recorded instance was swept");

        // A replacement is a fresh instance rather than republishing the one the
        // recording still holds.
        let fresh = Arc::new(QuasiImmut::new());
        assert!(fresh.is_current());
        assert!(!Arc::ptr_eq(&recorded, &fresh));

        // Registering on the swept instance cannot be silently lost: the loop
        // folded a pre-mutation value, so it is born invalid.
        let flag = Arc::new(AtomicBool::new(false));
        let token = test_loop_token(&flag);
        recorded.register_loop_token(&token);
        assert!(flag.load(Ordering::Acquire));
        assert_eq!(recorded.len(), 0, "a swept list must not grow again");
    }

    /// Reclamation is not revocation: nothing invalidated the field, so the
    /// instance is current and the registration is recorded rather than
    /// born-invalid.
    #[test]
    fn a_current_instance_records_the_registration() {
        let recorded = QuasiImmut::new();
        assert!(recorded.is_current());
        let flag = Arc::new(AtomicBool::new(false));
        let token = test_loop_token(&flag);
        recorded.register_loop_token(&token);
        assert!(!flag.load(Ordering::Acquire));
        assert_eq!(recorded.len(), 1);
    }
}
