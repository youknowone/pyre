//! Debug-gated attribution for writes into a frame's `int_regs`.
//!
//! Answers one question: **which code last wrote the slot a reader is about to
//! read?** A register file is a `Vec<Option<OpRef>>` written from ten
//! production sites across two files, and a value read out of it carries no
//! record of which of them put it there. Every existing instrument in this area
//! reports what a slot *holds*; none reports where it *came from*.
//!
//! The writer is not named by hand. `#[track_caller]` makes each note carry its
//! own call site, so a site cannot be mislabelled and a site that is moved
//! re-reports itself at the new position.
//!
//! ## Why the frame is identified by its register array's address
//!
//! Attribution needs to distinguish "the root frame's slot 4" from "some
//! callee's slot 4", and frame depth is not usable: three of the writers fill a
//! callee frame *before* it is pushed, so at the moment of the write its depth
//! does not exist yet. The address of the `int_regs` backing buffer is
//! available at every site and is the same object the reader later indexes.
//!
//! WARNING: An address identifies a live allocation, not a frame for all time. A
//! popped frame frees its buffer and a later frame may be handed the same
//! address, so two entries under one address may belong to two frames. That is
//! sound for the question asked here — the read is attributed to the most
//! recent write to *the same live buffer*, which is the writer whose value the
//! reader is holding — but it is NOT sound for counting distinct frames, and
//! this module deliberately reports no such count.
//!
//! ## STOP: Which configuration a report describes
//!
//! **Arm A only; `split_dispatch = true` is unexercised by any tracked crate.**
//!
//! Every measurement taken with this module so far ran with `split_dispatch`
//! off, because nothing in the repository turns it on: the only tracked
//! occurrence of the spelling is a doc comment in
//! `majit-macros/src/jit_interp/jitcode_lower/dispatch.rs`, and `git log -S`
//! over `majit/examples/` is empty for all history. The two crates that do set
//! it live in untracked nested checkouts, which every repo-scoped census
//! silently misses.
//!
//! That matters because the arms disagree about this module's own subject. With
//! the flag off, a virtualizable identity sits low and the positional mask
//! diverts it; with the flag on, the identity is body-allocated, tracks
//! `split_identity_floor`, and the mask diverts nothing. **A zero from this
//! instrument is therefore a statement about the arm that was built, not about
//! majit** — and the arm where the interesting defect is inert is exactly the
//! one a default build measures. See [`declare_arm`].
//!
//! ## Reading a zero
//!
//! `reads` and `writes` are separate counters because a site that recorded no
//! write has two readings — "it ran and wrote elsewhere" and "it never ran" —
//! and excluding a candidate on the strength of the wrong one is the failure
//! this instrument exists to avoid. A site absent from the report wrote
//! nothing; the report says so explicitly rather than leaving it to be inferred
//! from an absence.
//!
//! Gate: `MAJIT_REG_WRITE_AUDIT=1`. Off, every entry point is one thread-local
//! read and a return. All state is thread-local, matching `opref_audit`: a
//! trace is recorded on one thread, and a process-global table would let two
//! tests in parallel attribute each other's writes.

use crate::resoperation::OpRef;
use std::cell::RefCell;
use std::collections::HashMap;
use std::panic::Location;

/// `(file, line)` of a writer. `Location` is not used as a key directly so the
/// key stays a plain comparable pair.
type Site = (&'static str, u32);

/// One distinct thing a reader observed: a slot, the variant it held, and the
/// site that last wrote it.
#[derive(PartialEq, Eq, Hash)]
struct Observation {
    slot: usize,
    /// `Debug` of the `OpRef`, e.g. `InputArgInt(0)`. Kept as text because the
    /// point of the report is to be read, and the payload is part of the
    /// witness being attributed.
    value: String,
    writer: Option<Site>,
}

#[derive(Default)]
struct Audit {
    /// `None` until this thread has read the environment.
    on: Option<bool>,
    /// `(int_regs buffer address, slot) -> site that last wrote it`.
    last_writer: HashMap<(usize, usize), Site>,
    /// Per-site write tally. A site with no entry here never wrote.
    writes: HashMap<Site, usize>,
    /// Slots each site wrote, so "ran but never touched slot 4" is a statement
    /// the report can make rather than one the reader has to infer.
    slots: HashMap<Site, Vec<usize>>,
    /// Distinct observations, with how often each recurred.
    observations: HashMap<Observation, usize>,
    reads: usize,
    /// The configuration this run measured, as declared by the subject.
    ///
    /// WARNING: Not derivable here. `split_dispatch` is a macro-time flag in the
    /// *consumer* crate; by the time this module sees a write, the choice is
    /// baked into generated code and nothing at this layer can read it back.
    /// So it is declared, not detected — and left `None` when nobody declared
    /// it, never defaulted to a plausible arm.
    ///
    /// This exists because a zero from this instrument is configuration-shaped.
    /// Measured on dualtape: with `split_dispatch` off the identity sits at
    /// slot 2 and the mask diverts it; with the flag on the same run diverts
    /// nothing. An undeclared report is therefore a number without the one
    /// fact needed to read it, and would be cited later as "measured clean" on
    /// the arm where the subject is inert.
    arm: Option<&'static str>,
    /// Slots the reader visited but resolved WITHOUT indexing `int_regs` —
    /// keyed by `(slot, which branch took it)`.
    ///
    /// Without this, "slot 4 was never read" has two readings: the reader never
    /// visited slot 4, or it visited and a mask answered on its behalf before
    /// the instrumented line. Those are opposite conclusions — the second says
    /// the instrument is blind exactly where the question is — and they are
    /// spelled identically in a table that only records what reached the read.
    /// Together with [`Audit::reads`] these partition every slot the liveness
    /// iterator yields.
    diverted: HashMap<(usize, &'static str), usize>,
}

thread_local! {
    static AUDIT: RefCell<Audit> = RefCell::new(Audit::default());
}

fn resolve() -> bool {
    matches!(std::env::var("MAJIT_REG_WRITE_AUDIT"), Ok(v) if v != "0" && !v.is_empty())
}

#[inline]
fn enabled(audit: &mut Audit) -> bool {
    *audit.on.get_or_insert_with(resolve)
}

/// Record that the caller wrote `slot` of the `int_regs` buffer at `regs`.
///
/// `regs` is the buffer address (`self.int_regs.as_ptr() as usize`), not the
/// frame — see the module doc for why depth is unavailable at three of the
/// writers.
#[track_caller]
#[inline]
pub fn note_int_write(regs: usize, slot: usize, _opref: Option<OpRef>) {
    // WARNING: Resolved HERE, not inside the closure below. `#[track_caller]` does not
    // propagate into a nested closure: `Location::caller()` called inside
    // `AUDIT.with(|a| …)` resolves against the closure, which is defined in this
    // file, so every wired site reports this module's own line instead of its
    // own. That failure is uniform — one plausible-looking writer for every
    // site — and it reads as a finding rather than as a broken instrument. The
    // three controls below exist because it is not visible by inspection.
    let loc = Location::caller();
    AUDIT.with(|a| {
        let mut audit = a.borrow_mut();
        if !enabled(&mut audit) {
            return;
        }
        let site: Site = (loc.file(), loc.line());
        audit.last_writer.insert((regs, slot), site);
        *audit.writes.entry(site).or_insert(0) += 1;
        let seen = audit.slots.entry(site).or_default();
        if !seen.contains(&slot) {
            seen.push(slot);
        }
    });
}

/// Record that a reader took `opref` out of `slot` of the buffer at `regs`, and
/// attribute it to whichever site last wrote that slot.
///
/// A read whose slot has no recorded writer is kept with `writer: None` rather
/// than dropped: an unattributed read is the finding, not a gap to hide.
pub fn note_int_read(regs: usize, slot: usize, opref: OpRef) {
    AUDIT.with(|a| {
        let mut audit = a.borrow_mut();
        if !enabled(&mut audit) {
            return;
        }
        audit.reads += 1;
        let writer = audit.last_writer.get(&(regs, slot)).copied();
        let obs = Observation {
            slot,
            value: format!("{opref:?}"),
            writer,
        };
        *audit.observations.entry(obs).or_insert(0) += 1;
    });
}

/// Record that the reader visited `slot` but was answered by `which` instead of
/// by the register file, so no read of `int_regs[slot]` took place.
pub fn note_int_read_diverted(slot: usize, which: &'static str) {
    AUDIT.with(|a| {
        let mut audit = a.borrow_mut();
        if !enabled(&mut audit) {
            return;
        }
        *audit.diverted.entry((slot, which)).or_insert(0) += 1;
    });
}

/// Slots the reader visited without indexing the register file, as
/// `(slot, branch, count)`.
pub fn diverted_reads() -> Vec<(usize, &'static str, usize)> {
    AUDIT.with(|a| {
        let mut rows: Vec<(usize, &'static str, usize)> = a
            .borrow()
            .diverted
            .iter()
            .map(|(&(slot, which), &n)| (slot, which, n))
            .collect();
        rows.sort_unstable();
        rows
    })
}

/// Print every distinct observation and every writer's tally.
pub fn report(subject: &str) {
    AUDIT.with(|a| print_table(&a.borrow(), subject));
}

fn print_table(audit: &Audit, header: &str) {
    if audit.on != Some(true) {
        return;
    }
    let diverted_total: usize = audit.diverted.values().sum();
    eprintln!(
        "[reg-write] ===== {header}: arm={} reads={} diverted={} \
         distinct_observations={} writer_sites={} =====",
        audit.arm.unwrap_or("UNDECLARED"),
        audit.reads,
        diverted_total,
        audit.observations.len(),
        audit.writes.len(),
    );
    if audit.arm.is_none() {
        // Loud, and on the same line-prefix as the data, so a grep that
        // collects the table also collects the caveat. A quiet default would
        // let an arm-A zero be quoted as a property of the mechanism.
        eprintln!(
            "[reg-write]   WARNING: NO ARM DECLARED — this table describes ONE configuration and \
             does not say which. Call declare_arm() from the subject. A zero here is not a \
             statement about majit; it is a statement about the arm that was built."
        );
    }
    let mut rows: Vec<(&Observation, &usize)> = audit.observations.iter().collect();
    rows.sort_by(|l, r| l.0.slot.cmp(&r.0.slot).then(r.1.cmp(l.1)));
    for (obs, count) in rows {
        let writer = match obs.writer {
            Some((file, line)) => format!("{file}:{line}"),
            None => "UNATTRIBUTED (no recorded write to this slot)".to_string(),
        };
        eprintln!(
            "[reg-write]   int_regs[{}] = {} <- {} (x{count})",
            obs.slot, obs.value, writer
        );
    }
    let mut diverted: Vec<(&(usize, &'static str), &usize)> = audit.diverted.iter().collect();
    diverted.sort_unstable();
    for ((slot, which), count) in diverted {
        eprintln!("[reg-write]   DIVERTED slot {slot} answered by {which} (x{count})");
    }
    let mut sites: Vec<(&Site, &usize)> = audit.writes.iter().collect();
    sites.sort_by(|l, r| r.1.cmp(l.1));
    for ((file, line), count) in sites {
        let mut slots = audit.slots[&(*file, *line)].clone();
        slots.sort_unstable();
        eprintln!("[reg-write]   WRITER {file}:{line} wrote {count}x, slots {slots:?}");
    }
}

/// Printed at thread exit so a gated run produces its table without the
/// subject having to call anything. Best-effort, exactly as in `opref_audit`:
/// a thread-local destructor is not guaranteed to run on the main thread, so a
/// caller that needs the table should call [`report`] itself.
impl Drop for Audit {
    fn drop(&mut self) {
        print_table(self, "thread exit");
    }
}

/// Declare which configuration this run was built in, e.g.
/// `"dualtape split_dispatch=false"`.
///
/// STOP: Declaring a value this crate cannot check is on the caller. The point is
/// not verification — it is that the report carries the fact at all, so a table
/// cannot be quoted without the arm it belongs to.
pub fn declare_arm(arm: &'static str) {
    AUDIT.with(|a| a.borrow_mut().arm = Some(arm));
}

/// The declared arm, or `None` if the subject never declared one.
pub fn arm() -> Option<&'static str> {
    AUDIT.with(|a| a.borrow().arm)
}

/// Forget every attribution and zero the counters.
///
/// WARNING: Deliberately does NOT clear the declared arm: the arm is a property of the
/// binary, and a `reset()` between two measurements in one process cannot have
/// changed it. Clearing it would silently downgrade every later report to
/// UNDECLARED.
pub fn reset() {
    AUDIT.with(|a| {
        let mut audit = a.borrow_mut();
        audit.last_writer.clear();
        audit.writes.clear();
        audit.slots.clear();
        audit.observations.clear();
        audit.reads = 0;
    });
}

/// Force this thread's mode, so a test can exercise the instrument regardless
/// of how the run was invoked.
pub fn set_mode_for_test(on: bool) {
    AUDIT.with(|a| a.borrow_mut().on = Some(on));
}

/// Every writer site recorded on this thread, as `("file", line, writes)`.
pub fn writer_sites() -> Vec<(&'static str, u32, usize)> {
    AUDIT.with(|a| {
        let audit = a.borrow();
        let mut rows: Vec<(&'static str, u32, usize)> =
            audit.writes.iter().map(|(&(f, l), &n)| (f, l, n)).collect();
        rows.sort_unstable();
        rows
    })
}

/// The site that last wrote `slot` of the buffer at `regs`, if any.
pub fn writer_of(regs: usize, slot: usize) -> Option<(&'static str, u32)> {
    AUDIT.with(|a| a.borrow().last_writer.get(&(regs, slot)).copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// POSITIVE CONTROL. A write then a read of the same slot must attribute
    /// the read to the writing line, and the line must be this test's own.
    #[test]
    fn a_read_is_attributed_to_the_line_that_wrote_the_slot() {
        set_mode_for_test(true);
        reset();
        let regs = 0x1000;
        let write_line = line() + 1;
        note_int_write(regs, 4, Some(OpRef::input_arg_int(0)));
        assert_eq!(
            writer_of(regs, 4),
            Some((file!(), write_line)),
            "#[track_caller] must report the CALL site, not this module"
        );
    }

    /// Two different lines writing the same slot: the LAST one owns the read.
    /// Without this the instrument could report a stale attribution and read
    /// exactly like a correct one.
    #[test]
    fn the_last_writer_wins() {
        set_mode_for_test(true);
        reset();
        let regs = 0x2000;
        note_int_write(regs, 4, Some(OpRef::input_arg_int(0)));
        let second = line() + 1;
        note_int_write(regs, 4, Some(OpRef::int_op(9)));
        assert_eq!(writer_of(regs, 4), Some((file!(), second)));
    }

    /// Two buffers are two frames. A write to one must not attribute a read of
    /// the other — that collapse is what makes a slot-keyed table say "the
    /// root frame's slot 4" when it means some callee's.
    #[test]
    fn two_buffers_do_not_share_attributions() {
        set_mode_for_test(true);
        reset();
        note_int_write(0x3000, 4, Some(OpRef::input_arg_int(0)));
        assert!(
            writer_of(0x4000, 4).is_none(),
            "a different register file must not inherit an attribution"
        );
    }

    /// NEGATIVE CONTROL for the third value. A slot nobody wrote must stay
    /// UNATTRIBUTED rather than borrow the nearest writer, because "nobody
    /// wrote this" is a finding.
    #[test]
    fn an_unwritten_slot_stays_unattributed() {
        set_mode_for_test(true);
        reset();
        let regs = 0x5000;
        note_int_write(regs, 1, Some(OpRef::input_arg_int(0)));
        note_int_read(regs, 4, OpRef::input_arg_int(0));
        assert!(writer_of(regs, 4).is_none());
        AUDIT.with(|a| {
            let audit = a.borrow();
            let obs = audit.observations.keys().next().expect("one observation");
            assert_eq!(obs.slot, 4);
            assert!(obs.writer.is_none(), "must not be attributed to slot 1");
        });
    }

    /// A site that ran and wrote only other slots is distinguishable from a
    /// site that never ran. This is the distinction the whole instrument turns
    /// on: excluding a candidate needs the first, and only the first.
    #[test]
    fn a_site_that_wrote_elsewhere_is_not_a_site_that_never_ran() {
        set_mode_for_test(true);
        reset();
        let regs = 0x6000;
        let ran = line() + 1;
        note_int_write(regs, 7, Some(OpRef::int_op(1)));
        let sites = writer_sites();
        assert_eq!(sites, vec![(file!(), ran, 1)], "the site that ran reports");
        assert!(
            writer_of(regs, 4).is_none(),
            "and it is still not the writer of slot 4"
        );
    }

    /// The arm is never guessed, and `reset` must not silently drop it.
    ///
    /// Both halves matter. Defaulting an undeclared arm to a plausible string
    /// would make an arm-A zero readable as a claim about majit; clearing it on
    /// `reset` would downgrade the second of two measurements in one process to
    /// UNDECLARED without anyone touching the declaration.
    #[test]
    fn the_arm_is_declared_never_defaulted_and_survives_reset() {
        set_mode_for_test(true);
        reset();
        assert_eq!(arm(), None, "an undeclared arm must stay unknown");
        declare_arm("dualtape split_dispatch=false");
        reset();
        assert_eq!(
            arm(),
            Some("dualtape split_dispatch=false"),
            "reset clears measurements, not the configuration they were taken in"
        );
    }

    /// Gated off, nothing is recorded — otherwise every host pays for an
    /// instrument they did not ask for, and a disabled run looks reached.
    #[test]
    fn disabled_records_nothing() {
        set_mode_for_test(false);
        reset();
        note_int_write(0x7000, 4, Some(OpRef::input_arg_int(0)));
        note_int_read(0x7000, 4, OpRef::input_arg_int(0));
        assert!(writer_of(0x7000, 4).is_none());
        assert!(writer_sites().is_empty());
    }

    /// `line!()` at the call site, so the controls above compute the expected
    /// line rather than hard-coding one that rots on the next edit.
    #[track_caller]
    fn line() -> u32 {
        Location::caller().line()
    }
}
