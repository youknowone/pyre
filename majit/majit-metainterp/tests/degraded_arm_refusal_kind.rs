//! `refusal_kind` against the reasons the example crates actually record.
//!
//! The classifier keys on prose fragments emitted by the macro, so the only
//! thing that makes it trustworthy is a corpus of strings taken from real runs.
//! Every reason below was captured under `MAJIT_LOG=1` — the recorder prints it
//! itself, so no probe edit is needed to re-take them:
//!
//! ```text
//! MAJIT_LOG=1 cargo test --manifest-path majit/examples/<crate>/Cargo.toml -- --nocapture
//! ```
//!
//! When a family is added, add it HERE as well as in `refusal_kind`. This file
//! is the one place the whole known corpus is visible; the example gates each
//! see only their own crate's reasons and cannot tell a new mechanism from a
//! reworded one.

use majit_metainterp::{REFUSAL_SEPARATOR, RefusalKind, refusal_kind, refusal_kinds};

const TL_ROLL: &str = "arm body writes a green this lowering path cannot carry \
                       back to the caller (lowering stopped at this statement; \
                       any further blockers follow): pc += 1;";

/// `tl`'s `ROLL` again, originally recorded with the pc-return gate term dropped
/// so the arm reached the channel. The mechanism refusing it changes; the arm
/// name does not. This pair is the whole reason `RefusalKind` exists — see
/// `refusal_kind_separates_the_two_reasons_one_arm_produced`.
///
/// A MUTATED BUILD IS NO LONGER NEEDED TO SEE THIS. Now that lowering keeps
/// walking after its first refusal, the unmodified build records this exact
/// string as the SECOND member of `TL_ROLL_ACCUMULATED` — asserted byte for
/// byte by `the_accumulated_tl_reason_is_its_two_known_members`. The A/B that
/// produced it stays documented because it is how the second mechanism was
/// first found, not because it is still the only way to reach it.
const TL_ROLL_ON_PC_CHANNEL: &str = "arm body has a statement the lowerer \
                                     cannot express: \
                                     storage_roll(state.stack.as_mut_ptr() as \
                                     usize, state.stackpos, r);";

/// `tl`'s `PUSHARG` — an unlowerable store into the stack array.
const TL_PUSHARG: &str = "arm body has a statement the lowerer cannot express: \
                          state.stack [state.stackpos as usize] = inputarg;";

/// `braininterp`'s loop-end arm, refused because it encloses loop control.
const BF_LOOP_END: &str = "arm body encloses a `break`/`continue` that cannot \
                           be lowered in place: if state.tape [state.pointer as \
                           usize] != 0";

/// `tiny2`/`tiny3`'s `OP_PUSH_INT` and `OP_PUSH_FLOAT` — the widening operand
/// read.
const TINY_PUSH: &str = "arm body has a statement the lowerer cannot express: \
                         let value = i64 ::from_le_bytes([program [pc], program \
                         [pc + 1], program [pc + ...)";

const TL_ROLL_ACCUMULATED: &str = "arm body writes a green this lowering path \
                                   cannot carry back to the caller (lowering \
                                   stopped at this statement; any further \
                                   blockers follow): pc += 1; || arm body has a \
                                   statement the lowerer cannot express: \
                                   storage_roll(state.stack.as_mut_ptr() as \
                                   usize, state.stackpos, r);";

/// `tlr`'s allocation arm, with green writeback and reallocation blockers.
const TLR_ALLOCATE: &str = "arm body writes a green this lowering path cannot \
                            carry back to the caller (lowering stopped at this \
                            statement; any further blockers follow): pc += 1; || \
                            arm body has a statement the lowerer cannot express: \
                            state.regs = vec! [0; n];";

/// `lower_stmt.rs`'s `lower_return_stmt` — the sibling guard to `break`/`continue`.
const SRC_ENCLOSED_RETURN: &str =
    "arm body encloses a `return` that cannot be lowered in place: return acc;";

/// `api.rs`'s `try_generate_jitcode_body_parts_with_caller_bindings` and
/// `try_generate_jitcode_pc_return_body_with_caller_bindings`. No
/// `{what}: {spelling}` shape — a bare `Err`.
const SRC_EMPTY_BODY: &str = "arm body has no statements to lower";

/// `api.rs`'s `try_generate_jitcode_pc_return_body_with_caller_bindings`, on the
/// pc-return path.
const SRC_NO_PC_BINDING: &str = "arm body has no `pc` binding for the pc-return writeback";

/// `dispatch.rs`'s `lower_dispatch_chain` — the only family raised at INSTALL rather than by the
/// statement lowerer, so it names no offending statement at all.
const SRC_UNSUPPORTED_CALL_POLICY: &str =
    "arm body lowering resolved an unsupported call policy at install";

/// `lowerer.rs`'s `take_body_failure_reason`. Its own doc keeps this exact wording so unconverted refusal
/// sites stay greppable, which is why it gets a variant instead of falling into
/// `Unclassified`: reaching it means "a site is unconverted", not "majit grew a
/// mechanism".
const SRC_UNREACHED_FALLBACK: &str = "arm body could not be lowered to a sub-JitCode";

#[test]
fn refusal_kind_covers_every_reachable_producer_string() {
    assert_eq!(
        refusal_kind(SRC_ENCLOSED_RETURN),
        RefusalKind::EnclosedReturn
    );
    assert_eq!(refusal_kind(SRC_EMPTY_BODY), RefusalKind::EmptyBody);
    assert_eq!(refusal_kind(SRC_NO_PC_BINDING), RefusalKind::NoPcBinding);
    assert_eq!(
        refusal_kind(SRC_UNSUPPORTED_CALL_POLICY),
        RefusalKind::UnsupportedCallPolicy
    );
    assert_eq!(
        refusal_kind(SRC_UNREACHED_FALLBACK),
        RefusalKind::UnreachedLoweringFallback
    );
}

/// `UnsupportedCallPolicy` names no offending statement, and the others do.
///
/// Seven of the eight families are minted by the statement lowerer and carry
/// `arm body {what}: {spelling}` — `record_body_failure` in `lower_stmt.rs`
/// appends `: ` and a (possibly truncated) rendering of the statement. Two of
/// those seven are bare `Err`s with no statement in hand, and the eighth is
/// raised at INSTALL, before any statement is under consideration.
///
/// This is the one family where the usual evidence is **structurally absent**:
/// a gate that pairs `(arm, RefusalKind)` with "and the reason names this source
/// snippet" — which every example gate does — has nothing to match here. Pinned
/// as a test rather than left in a doc comment, because the next person to write
/// such a gate will discover it as a puzzling failure otherwise.
#[test]
fn only_the_statement_lowerer_families_name_an_offending_statement() {
    // `: ` introduces the statement spelling. The install-time string has no
    // statement, so it must not look as though it does.
    for (reason, kind) in [
        (TL_ROLL, RefusalKind::GreenWriteback),
        (TL_PUSHARG, RefusalKind::UnlowerableStmt),
        (BF_LOOP_END, RefusalKind::EnclosedBreakContinue),
        (SRC_ENCLOSED_RETURN, RefusalKind::EnclosedReturn),
    ] {
        assert_eq!(refusal_kind(reason), kind);
        assert!(
            reason.contains(": "),
            "{kind:?} is minted by record_body_failure and must carry `: <stmt>`: {reason}"
        );
    }
    for (reason, kind) in [
        (
            SRC_UNSUPPORTED_CALL_POLICY,
            RefusalKind::UnsupportedCallPolicy,
        ),
        (SRC_EMPTY_BODY, RefusalKind::EmptyBody),
        (SRC_NO_PC_BINDING, RefusalKind::NoPcBinding),
        (
            SRC_UNREACHED_FALLBACK,
            RefusalKind::UnreachedLoweringFallback,
        ),
    ] {
        assert_eq!(refusal_kind(reason), kind);
        assert!(
            !reason.contains(": "),
            "{kind:?} names no offending statement, so a snippet-keyed gate has \
             nothing to match — it must not appear to carry one: {reason}"
        );
    }
}

/// Every reachable string maps to a DISTINCT family.
///
/// A consolidation that quietly collapses two families into one passes every
/// existing gate — the arms still degrade, the names are unchanged, and each
/// crate compares against whatever the merged value now is. This is the check
/// that a merge cannot silently pass.
#[test]
fn the_eight_reachable_strings_do_not_collapse_onto_each_other() {
    let all = [
        TL_ROLL,
        TL_PUSHARG,
        BF_LOOP_END,
        SRC_ENCLOSED_RETURN,
        SRC_EMPTY_BODY,
        SRC_NO_PC_BINDING,
        SRC_UNSUPPORTED_CALL_POLICY,
        SRC_UNREACHED_FALLBACK,
    ];
    let mut kinds: Vec<RefusalKind> = all.iter().map(|s| refusal_kind(s)).collect();
    let before = kinds.len();
    kinds.sort();
    kinds.dedup();
    assert_eq!(
        kinds.len(),
        before,
        "two reachable reason strings now classify the same, so any gate \
         comparing them is blind to a change between those two mechanisms: {:?}",
        all.iter()
            .map(|s| (refusal_kind(s), &s[..40.min(s.len())]))
            .collect::<Vec<_>>()
    );
    assert!(
        !kinds.contains(&RefusalKind::Unclassified),
        "a reachable producer string reached Unclassified — add its family here \
         and in refusal_kind, do not widen an existing fragment to absorb it"
    );
}

/// Checks every refusal string captured from the example corpus against its
/// expected family.
#[test]
fn refusal_kind_classifies_the_recorded_corpus() {
    assert_eq!(refusal_kind(TL_ROLL), RefusalKind::GreenWriteback);
    assert_eq!(
        refusal_kind(TL_ROLL_ON_PC_CHANNEL),
        RefusalKind::UnlowerableStmt
    );
    assert_eq!(refusal_kind(TL_PUSHARG), RefusalKind::UnlowerableStmt);
    assert_eq!(
        refusal_kind(BF_LOOP_END),
        RefusalKind::EnclosedBreakContinue
    );
    assert_eq!(refusal_kind(TINY_PUSH), RefusalKind::UnlowerableStmt);
    assert_eq!(
        refusal_kind(TL_ROLL_ACCUMULATED),
        RefusalKind::GreenWriteback
    );
    assert_eq!(refusal_kind(TLR_ALLOCATE), RefusalKind::GreenWriteback);
}

/// An accumulated reason exposes both its writeback and statement-lowering
/// blockers.
#[test]
fn an_accumulated_reason_reports_every_blocker() {
    let kinds = refusal_kinds(TLR_ALLOCATE);
    assert_eq!(
        kinds,
        vec![RefusalKind::GreenWriteback, RefusalKind::UnlowerableStmt],
        "tlr's ALLOCATE has two blockers and this must read both, the green \
         advance first; a length of 1 means this crate's REFUSAL_SEPARATOR no \
         longer matches the one the recorded string was minted with"
    );
    assert!(
        TLR_ALLOCATE.contains("state.regs = vec! [0; n];"),
        "the member behind the head must still name the reallocation — that \
         statement is the only thing this reason reports that nothing else does"
    );
}

/// The accumulated TL reason is exactly its two known members joined by the
/// shared separator.
#[test]
fn the_accumulated_tl_reason_is_its_two_known_members() {
    assert_eq!(
        TL_ROLL_ACCUMULATED,
        format!("{TL_ROLL}{REFUSAL_SEPARATOR}{TL_ROLL_ON_PC_CHANNEL}"),
        "the recorded string must be its head and its tail joined by the \
         separator; a mismatch means one of the three literals was re-recorded \
         without the others"
    );
}

/// The property the classifier exists for: one arm, two mechanisms, and the
/// name is the same in both.
///
/// A gate keyed on the degraded-arm NAME reads identically for these two — that
/// is exactly what happened, across three A/B arms, with the suite green in all
/// three. If a later edit makes these classify the same, every such gate goes
/// quietly blind again.
///
/// Implied by `refusal_kind_classifies_the_recorded_corpus` and cannot fail
/// while that passes: it pins both strings to distinct variants already. Kept as
/// a separate test because it names the property and fails with a message about
/// it, not as independent coverage. Do not cite it as a second check.
#[test]
fn refusal_kind_separates_the_two_reasons_one_arm_produced() {
    assert_ne!(
        refusal_kind(TL_ROLL),
        refusal_kind(TL_ROLL_ON_PC_CHANNEL),
        "the two reasons ROLL produced now classify the same, so a cause gate \
         can no longer distinguish the arms it exists to distinguish"
    );
}

/// A mechanism no fragment matches must reach `Unclassified` rather than land in
/// a known family.
///
/// Do not "fix" a failing gate by widening a fragment until this passes as
/// something else. `Unclassified` reaching a gate means majit grew a refusal
/// family; the fix is to add the family here, with its recorded string.
#[test]
fn an_unknown_mechanism_is_not_bucketed_into_a_known_family() {
    assert_eq!(
        refusal_kind("arm body does something no fragment has been written for"),
        RefusalKind::Unclassified
    );
    // Sharper: a reason containing the word "cannot" but not the fragment must
    // still be unclassified. `braininterp`'s reads "cannot be lowered in place",
    // which is one word away from the `cannot express` fragment.
    assert_eq!(
        refusal_kind("arm body cannot be lowered for a brand new reason"),
        RefusalKind::Unclassified
    );
}
