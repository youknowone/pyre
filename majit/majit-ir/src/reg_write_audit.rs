//! Debug-gated attribution for writes into a frame's `int_regs`.
//!
//! Each write records its `#[track_caller]` location, allowing a later register
//! read to identify the code that produced its value. Frames are keyed by the
//! address of the live `int_regs` buffer because callees may be populated before
//! they have a stack depth. The key is suitable for latest-writer attribution,
//! not for counting frames across buffer reuse.
//!
//! Reads, writes, and reads diverted before indexing the register file are
//! counted separately so a zero can be distinguished from an unexecuted path.
//! Reports also name the caller-declared configuration arm; see [`declare_arm`].
//!
//! Build with `jit-audits` to compile the instrumentation sites. In that
//! build, `MAJIT_REG_WRITE_AUDIT=1` enables collection; with the environment
//! gate off, every entry point is one thread-local read and a return. Without
//! the Cargo feature, the module and its call sites are absent. State is
//! thread-local because each trace is processed on one thread and parallel
//! tests must not share diagnostic observations.

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
    /// The macro-time configuration declared by the consumer. It cannot be
    /// recovered from generated register writes at this layer.
    arm: Option<&'static str>,
    /// Slots the reader visited but resolved WITHOUT indexing `int_regs` —
    /// keyed by `(slot, which branch took it)`.
    ///
    /// Together with [`Audit::reads`], these partition the slots visited by the
    /// liveness iterator.
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
    // Resolve before entering the closure: `#[track_caller]` does not propagate
    // through it, and resolving inside would attribute every write to this file.
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

/// Declare the caller-known configuration this run was built in.
///
/// This records rather than verifies the configuration; the caller owns the
/// value because this crate cannot recover macro-time consumer settings.
pub fn declare_arm(arm: &'static str) {
    AUDIT.with(|a| a.borrow_mut().arm = Some(arm));
}

/// The declared arm, or `None` if the subject never declared one.
pub fn arm() -> Option<&'static str> {
    AUDIT.with(|a| a.borrow().arm)
}

/// Forget every attribution and zero the counters.
///
/// The declared arm is retained because resetting counters does not change the
/// binary's configuration.
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
