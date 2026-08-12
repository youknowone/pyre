//! Debug-gated detector for two `OpRef` variants colliding on one `raw()` key.
//!
//! `OpRef::raw()` (`resoperation.rs:211`) maps eight variants onto one `u32`,
//! so `InputArgInt(7)`, `IntOp(7)` and `InputArgRef(7)` are the same key. The
//! enum's own doc says why that tag is load-bearing: it "keeps the disjoint
//! RPython Box classes disjoint even when Pyre's flat encoding reuses the same
//! raw position across InputArg / ResOp / Const namespaces". A table keyed on
//! `raw()` throws the tag away, and the collapse is invisible at the call site
//! — it yields a wrong answer, not a type error.
//!
//! Within one `TraceRecorder` the collapse is harmless: `record_input_arg`,
//! `record_op`, `record_op_with_descr`, the guard recorders, `Jump` and
//! `Finish` all mint from a single `op_count` that each of them increments,
//! and input args are registered before any op (`recorder.rs:200`), so `raw()`
//! is injective there by construction. The hazard is therefore never "this
//! site calls `raw()`" — it is **two numbering scopes meeting at one key**,
//! which is what this module measures.
//!
//! A collision is a difference of *variant*, never of payload. At a rewrite
//! boundary the payload legitimately changes — a translation cache answering
//! `InputArgRef(0) -> InputArgRef(4824)` is the healthy case, same bank, new
//! position — so comparing whole boxes reports correct behaviour as a defect.
//! Measured on the tree `e939e25b713`, which a rebase has since rewritten —
//! find it by subject, "majit-ir: add a debug-gated detector for two OpRef
//! variants on one raw() key". Comparing whole boxes produced 182619 reports on
//! dualtape, of which 182619 differed only in payload and none differed in
//! variant. The printed line said "claimed by two variants" and named the same
//! variant twice, 182619 times.
//!
//! What this instrument cannot see. `note_key` returns early on a `None` or
//! constant `OpRef`, which is correct — an inline Const has no `raw()` to key
//! on, and `raw()` panics on one. But that blind spot coincides exactly with
//! the positional mask that blanks an inline frame's identity-slot prefix to
//! `Const(0)`: where the mask applies, a collision is masked into a constant
//! and this detector declines it. So a zero from a masked range is not evidence
//! of absence — a sound zero and an uninformative one are spelled identically.
//! Widening the identity range past the mask does distinguish them, but only as
//! a throwaway diagnostic that corrupts the run it measures: body-allocated
//! registers sit just past the identity ends, because `next_reg` is raised to
//! `max(max_pre_bound, split_identity_floor, float_identity_floor)` (see
//! `jitcode_lower/api.rs`). So any widening that reaches them also blanks
//! ordinary working registers — which `int_identity_reserved_end`'s own doc in
//! `pyjitpl/dispatch.rs` already calls out as dropping live data.
//! Distinguishing the two zeroes soundly needs the collision observed BEFORE
//! the trim runs, not a wider trim.
//!
//! WARNING: Those two floor terms are not gated alike, and the int/ref one is the
//! weaker: `split_identity_floor` is **inert (0) unless `split_dispatch`**,
//! while `float_identity_floor` is deliberately ungated. So without
//! `split_dispatch` the body's registers are not held above the int/ref
//! identity ends at all, and a widening there reaches live data sooner, not
//! later. The hazard is worse in that arm, not absent from it — do not read
//! the floor as a guard.
//!
//! It is also not the only predicate on this hazard, and the two are
//! complementary rather than redundant. This one fires at the *query*, naming
//! the key whose two claimants disagree; the `translate_trace_iter_opref` bank
//! tripwire fires at the *answer*, naming the operand the lookup resolved to.
//! One reports that a slot is contested, the other what came back out of it.
//!
//! Gate: `MAJIT_OPREF_VARIANT_AUDIT=1` reports, `=abort` panics on the first
//! collision. Off, `note` is one thread-local read and a return.
//!
//! All state is thread-local, including the mode: a process-global counter
//! would make two tests running in parallel read each other's collisions, and
//! a process-global mode would let a test that disables the audit silence a
//! concurrent one. Per-thread state is also the honest scope — a trace is
//! recorded, optimized and compiled on one thread.
//!
//! Retires when the two namespaces are made structurally uncollidable: at that
//! point a collision is unrepresentable rather than merely unobserved, and
//! this instrument has nothing left to detect.

use crate::resoperation::OpRef;
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::{Discriminant, discriminant};

const OFF: u8 = 1;
const REPORT: u8 = 2;
const ABORT: u8 = 3;

/// `(label, scope, key)` — one slot of one table.
type Key = (&'static str, usize, u32);

/// One distinct collision, plus how often it recurred.
///
/// Deduping and counting answer different questions and neither substitutes
/// for the other: how many slots are affected, and how hot each one is. A run
/// that prints one line per occurrence reports 182619 problems when it has
/// found 1618; a run that only dedupes cannot tell a one-shot from a hot loop.
struct Collision {
    label: &'static str,
    scope: usize,
    key: u32,
    first: OpRef,
    now: OpRef,
    occurrences: usize,
}

#[derive(Default)]
struct Audit {
    /// `None` until this thread has read the environment.
    mode: Option<u8>,
    /// The OpRef that first claimed each key.
    ///
    /// `scope` is caller-supplied and identifies the *table*, not the process:
    /// two different caches may legitimately assign different variants to the
    /// same raw, and only a collision *within one table* is a defect. Callers
    /// pass something stable for the table's lifetime, e.g. `slice.as_ptr()`.
    seen: HashMap<Key, OpRef>,
    /// Keyed by slot *and* by the ordered pair of variants, so one slot that
    /// collides two different ways stays two findings.
    collisions: HashMap<(Key, Discriminant<OpRef>, Discriminant<OpRef>), Collision>,
    collision_occurrences: usize,
    /// Every non-const note this thread accepted while enabled.
    ///
    /// Without it a reported zero has two readings — "no collision" and "the
    /// instrumented site never executed" — and the caller cannot tell them
    /// apart. A collision count is only a result when it is paired with proof
    /// the detector was reached, so a verification criterion has to read
    /// `collisions() == 0 && notes() > 0`, never the first conjunct alone.
    notes: usize,
    /// Notes whose key was already claimed and whose OpRef has the SAME
    /// variant as the stored entry — payload ignored, so a legitimate rewrite
    /// (`InputArgRef(0)` answered by `InputArgRef(4824)`) counts here, and so
    /// does re-noting a byte-identical box.
    ///
    /// WARNING: This is an exact definition and it is this module's alone. It is not
    /// interchangeable with any other instrument's "same"/"unchanged" tally:
    /// those may count identical boxes only, or count calls rather than notes,
    /// and this one does neither. Anyone registering a prediction against it
    /// must register against this sentence, not against a number from a
    /// different instrument.
    revisits_same_variant: usize,
}

thread_local! {
    static AUDIT: RefCell<Audit> = RefCell::new(Audit::default());
}

fn resolve_mode() -> u8 {
    match std::env::var("MAJIT_OPREF_VARIANT_AUDIT") {
        Ok(v) if v == "abort" => ABORT,
        Ok(v) if v != "0" && !v.is_empty() => REPORT,
        _ => OFF,
    }
}

/// Whether the audit is collecting on this thread. Callers may use this to
/// skip computing a scope value; `note` re-checks, so it is an optimization.
#[inline]
pub fn enabled() -> bool {
    AUDIT.with(|a| {
        let mut audit = a.borrow_mut();
        *audit.mode.get_or_insert_with(resolve_mode) >= REPORT
    })
}

/// Collision *occurrences* on this thread since start or since [`reset`] —
/// every firing, including repeats of one already reported.
///
/// WARNING: Never read this alone. Zero collisions and an instrument that was never
/// reached produce the identical number; pair it with [`notes`].
pub fn collisions() -> usize {
    AUDIT.with(|a| a.borrow().collision_occurrences)
}

/// Distinct `(slot, variant-pair)` collisions — how many places are affected,
/// as opposed to how often they fire ([`collisions`]).
pub fn distinct_collisions() -> usize {
    AUDIT.with(|a| a.borrow().collisions.len())
}

/// Notes that landed on an already-claimed key carrying the SAME variant.
///
/// See the field's doc for the exact definition. In short: a revisit, payload
/// ignored — the healthy translation-cache answer and the byte-identical
/// re-note both count here, and neither is a collision.
pub fn revisits_same_variant() -> usize {
    AUDIT.with(|a| a.borrow().revisits_same_variant)
}

/// Notes accepted on this thread while enabled, collisions included.
///
/// This is the reached-ness half of any verdict: `collisions() == 0` means
/// "no collision" only when `notes() > 0`. Zero here means the instrumented
/// code did not run — a dead instrument, not a clean one.
pub fn notes() -> usize {
    AUDIT.with(|a| a.borrow().notes)
}

/// Distinct `(label, scope, key)` triples seen on this thread.
pub fn keys_seen() -> usize {
    AUDIT.with(|a| a.borrow().seen.len())
}

/// Forget every key and zero this thread's counters.
pub fn reset() {
    AUDIT.with(|a| {
        let mut audit = a.borrow_mut();
        audit.seen.clear();
        audit.collisions.clear();
        audit.collision_occurrences = 0;
        audit.notes = 0;
        audit.revisits_same_variant = 0;
    });
}

/// Print this thread's totals and one line per distinct collision.
///
/// Also runs from `Audit`'s destructor at thread exit, so a gated run gets a
/// total without the subject having to call anything. That is best-effort —
/// a thread-local destructor is not guaranteed to run on the main thread — so
/// a caller that needs the summary should call this itself rather than assume
/// teardown produced it.
pub fn report_summary() {
    AUDIT.with(|a| print_summary(&a.borrow()));
}

fn print_summary(audit: &Audit) {
    if audit.mode.unwrap_or(OFF) < REPORT {
        return;
    }
    eprintln!(
        "[opref-audit] SUMMARY notes={} keys={} revisits_same_variant={} \
         distinct_collisions={} collision_occurrences={}",
        audit.notes,
        audit.seen.len(),
        audit.revisits_same_variant,
        audit.collisions.len(),
        audit.collision_occurrences,
    );
    let mut rows: Vec<&Collision> = audit.collisions.values().collect();
    rows.sort_by(|l, r| r.occurrences.cmp(&l.occurrences));
    for row in rows {
        eprintln!(
            "[opref-audit]   x{} {}: key={} (scope={:#x}) first {:?}, now {:?}",
            row.occurrences, row.label, row.key, row.scope, row.first, row.now,
        );
    }
}

impl Drop for Audit {
    fn drop(&mut self) {
        print_summary(self);
    }
}

/// Force this thread's mode, bypassing the environment, so a test can exercise
/// the detector regardless of how the run was invoked.
pub fn set_mode_for_test(on: bool) {
    AUDIT.with(|a| a.borrow_mut().mode = Some(if on { REPORT } else { OFF }));
}

/// Record that `opref` was used as a `raw()` key in `label`'s table `scope`.
///
/// Fires when an `OpRef` of a *different variant* already claimed that key in
/// that table. Here the key is derived from the payload, so two boxes sharing
/// a key can only differ in their variant anyway — the tag `raw()` drops.
/// That is not true of [`note_key`], and assuming it there is what made the
/// first version of this module report 182619 false positives.
///
/// Inline-`Const` variants are skipped: they carry their value inline and have
/// no `raw()` encoding at all (`raw()` panics on them by design).
#[inline]
pub fn note(label: &'static str, scope: usize, opref: OpRef) {
    if opref.is_none() || opref.is_constant() {
        return;
    }
    note_key(label, scope, opref.raw(), opref);
}

/// Record that `opref` claims the key `key` in `label`'s table `scope`, where
/// `key` is supplied rather than taken from `opref.raw()`.
///
/// Needed wherever the collision is between a *lookup* and the *entry it
/// lands on* rather than between two lookups. A `raw()`-keyed translation
/// cache is the case that matters: the query `InputArgInt(0)` and the stored
/// operand `InputArgRef(4824)` both claim slot 0, but they carry different
/// raws, so noting each under its own `raw()` files them under different keys
/// and nothing is ever compared. Both must be noted under the *slot*.
///
/// Getting this wrong is silent — the detector reports zero and reads as a
/// clean result, which is indistinguishable from an instrument that cannot
/// fire.
///
/// WARNING: Because the key is supplied, the payload is free to differ from it, and
/// a differing payload is the NORMAL answer of a translation cache. Only the
/// variant is compared. See the module doc for what comparing whole boxes
/// measured on dualtape.
#[inline]
pub fn note_key(label: &'static str, scope: usize, key: u32, opref: OpRef) {
    if opref.is_none() || opref.is_constant() {
        return;
    }
    let report = AUDIT.with(|a| {
        let mut audit = a.borrow_mut();
        if *audit.mode.get_or_insert_with(resolve_mode) < REPORT {
            return None;
        }
        audit.notes += 1;
        let abort = audit.mode == Some(ABORT);
        let slot = (label, scope, key);
        let Some(&first) = audit.seen.get(&slot) else {
            audit.seen.insert(slot, opref);
            return None;
        };
        if discriminant(&first) == discriminant(&opref) {
            audit.revisits_same_variant += 1;
            return None;
        }
        audit.collision_occurrences += 1;
        let occurrences = {
            let row = audit
                .collisions
                .entry((slot, discriminant(&first), discriminant(&opref)))
                .or_insert(Collision {
                    label,
                    scope,
                    key,
                    first,
                    now: opref,
                    occurrences: 0,
                });
            row.occurrences += 1;
            row.occurrences
        };
        // Print once per distinct collision. A hot slot would otherwise emit
        // one line per firing and bury every other finding in the run.
        if occurrences > 1 {
            return None;
        }
        Some((
            first,
            abort,
            audit.collisions.len(),
            audit.collision_occurrences,
        ))
    });
    let Some((first, abort, distinct, total)) = report else {
        return;
    };
    let message = format!(
        "[opref-audit] {label}: key={key} claimed by two OpRef variants in one table \
         (scope={scope:#x}): first {first:?}, now {opref:?} \
         [distinct #{distinct}, occurrences so far {total}]"
    );
    eprintln!("{message}");
    assert!(!abort, "{message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// POSITIVE CONTROL. Two variants sharing a payload must fire.
    ///
    /// Without this the detector's silence is worthless: an assertion that
    /// only ever sees singletons reads exactly like one that never runs.
    #[test]
    fn two_variants_on_one_key_fire() {
        set_mode_for_test(true);
        reset();
        note("t", 0, OpRef::input_arg_int(7));
        assert_eq!(
            collisions(),
            0,
            "the first claim on a key is never a collision"
        );
        note("t", 0, OpRef::int_op(7));
        assert_eq!(
            collisions(),
            1,
            "InputArgInt(7) and IntOp(7) share raw=7 and must be reported"
        );
    }

    /// POSITIVE CONTROL for the translation-cache shape (#53 / dualtape).
    ///
    /// The collision there is between a *lookup* and the *entry it lands on*,
    /// not between two lookups: the query `InputArgInt(0)` reaches slot 0,
    /// which the recorder filled for `InputArgRef(4824)`. The two carry
    /// different raws, so noting each under its own `raw()` files them under
    /// different keys and nothing is compared — the detector reads zero and is
    /// blind, not clean. Both must be noted under the SLOT.
    #[test]
    fn a_lookup_colliding_with_its_stored_entry_fires() {
        set_mode_for_test(true);
        reset();
        let slot = 0;
        note_key("cache", 0, slot, OpRef::input_arg_int(0));
        note_key("cache", 0, slot, OpRef::input_arg_ref(4824));
        assert_eq!(
            collisions(),
            1,
            "the query's bank and the stored entry's bank differ at slot 0"
        );

        // And the wiring that would have missed it: keying each by its own
        // raw() puts them in different buckets and reports nothing.
        reset();
        note("cache", 0, OpRef::input_arg_int(0));
        note("cache", 0, OpRef::input_arg_ref(4824));
        assert_eq!(
            collisions(),
            0,
            "documents the blind wiring — raw 0 and raw 4824 are different keys"
        );
    }

    /// REGRESSION TEST for this module's own first version.
    ///
    /// It compared whole boxes, so a translation cache answering a query with
    /// a rewritten position — same bank, new slot — was reported as a variant
    /// collision. On dualtape that fired 182619 times and every one of them
    /// was this shape, the exact spelling below among them (24024 firings at
    /// key 0). The detector was named for variants and compared payloads.
    #[test]
    fn a_rewrite_that_keeps_the_variant_is_silent() {
        set_mode_for_test(true);
        reset();
        let slot = 0;
        note_key("cache", 0, slot, OpRef::input_arg_ref(0));
        note_key("cache", 0, slot, OpRef::input_arg_ref(4824));
        assert_eq!(
            collisions(),
            0,
            "same bank, new position is what a translation cache is FOR"
        );
        assert_eq!(
            revisits_same_variant(),
            1,
            "and it must be counted, not merely ignored"
        );
    }

    /// The counters must partition the notes, so no note can be silently
    /// dropped into a category nobody reads. Any future early return that
    /// forgets to classify its note breaks this identity.
    #[test]
    fn every_note_lands_in_exactly_one_category() {
        set_mode_for_test(true);
        reset();
        note_key("t", 0, 3, OpRef::input_arg_int(3)); // new key
        note_key("t", 0, 3, OpRef::input_arg_int(99)); // revisit, same variant
        note_key("t", 0, 3, OpRef::int_op(3)); // collision
        note_key("t", 0, 4, OpRef::int_op(4)); // new key
        assert_eq!(notes(), 4);
        assert_eq!(
            keys_seen() + revisits_same_variant() + collisions(),
            notes(),
            "first claims + same-variant revisits + collisions must be every note"
        );
    }

    /// Dedupe and count are separate facts and the detector owes both: one
    /// line per distinct collision, and a tally of how hot each one is.
    #[test]
    fn a_recurring_collision_is_one_finding_with_many_occurrences() {
        set_mode_for_test(true);
        reset();
        note_key("t", 0, 5, OpRef::input_arg_int(5));
        for _ in 0..7 {
            note_key("t", 0, 5, OpRef::int_op(5));
        }
        assert_eq!(distinct_collisions(), 1, "one slot, one variant pair");
        assert_eq!(collisions(), 7, "and the firing rate is not thrown away");
    }

    /// CONTROL FOR THE LIVENESS COUNTER ITSELF.
    ///
    /// `notes()` exists so a reported `collisions() == 0` can be told apart
    /// from an instrument that never ran. A counter that is itself never
    /// incremented would reintroduce the same defect one level down, so it is
    /// asserted to move — and to move on the *silent* path, which is exactly
    /// the case a zero-collision verdict rests on.
    #[test]
    fn notes_counts_every_accepted_note() {
        set_mode_for_test(true);
        reset();
        assert_eq!(notes(), 0, "reset must zero the liveness counter");
        note("t", 0, OpRef::int_op(1));
        note("t", 0, OpRef::int_op(2));
        note("t", 0, OpRef::int_op(1));
        assert_eq!(notes(), 3, "every accepted note counts, repeats included");
        assert_eq!(keys_seen(), 2, "two distinct keys");
        assert_eq!(
            collisions(),
            0,
            "the reached-ness proof must hold on the silent path"
        );

        // Constants carry their value inline and have no raw() encoding, so
        // they are skipped entirely and must not inflate the reached count.
        note("t", 0, OpRef::const_int(9));
        assert_eq!(notes(), 3);
    }

    /// Gated off, the liveness counter must stay at zero too — otherwise a
    /// disabled run would look reached.
    #[test]
    fn disabled_records_no_notes() {
        set_mode_for_test(false);
        reset();
        note("t", 0, OpRef::int_op(1));
        assert_eq!(notes(), 0);
        assert_eq!(collisions(), 0);
    }

    /// NEGATIVE CONTROL. The same box seen repeatedly is the common case and
    /// must stay silent, or the detector drowns its own signal.
    #[test]
    fn one_variant_seen_repeatedly_is_silent() {
        set_mode_for_test(true);
        reset();
        for _ in 0..8 {
            note("t", 0, OpRef::int_op(7));
        }
        assert_eq!(collisions(), 0);
    }

    /// The same raw in two different tables is legal — a per-table numbering
    /// may assign the same position to different variants. Scoping is what
    /// keeps the detector from reporting correct behaviour as a defect.
    #[test]
    fn same_raw_in_two_scopes_is_silent() {
        set_mode_for_test(true);
        reset();
        note("t", 0x1000, OpRef::input_arg_int(7));
        note("t", 0x2000, OpRef::int_op(7));
        assert_eq!(collisions(), 0);
    }

    /// Distinct raws never collide however many variants are involved.
    #[test]
    fn distinct_raws_are_silent() {
        set_mode_for_test(true);
        reset();
        note("t", 0, OpRef::input_arg_int(0));
        note("t", 0, OpRef::int_op(1));
        note("t", 0, OpRef::input_arg_ref(2));
        assert_eq!(collisions(), 0);
    }

    /// Gated off, the detector must not report — otherwise every host pays for
    /// an instrument they did not ask for.
    #[test]
    fn disabled_records_nothing() {
        set_mode_for_test(false);
        reset();
        note("t", 0, OpRef::input_arg_int(7));
        note("t", 0, OpRef::int_op(7));
        assert_eq!(collisions(), 0);
    }
}
