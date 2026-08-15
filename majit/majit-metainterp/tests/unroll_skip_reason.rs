//! A suppressed unroll has two possible causes, and the log has to say which.
//!
//! `compile_loop` skips the unroll optimizer when either the `PYRE_NO_UNROLL`
//! env override is set OR the jitdriver's `enable_opts` does not list `unroll`.
//! Both paths used to emit the same line — `[jit] PYRE_NO_UNROLL: skipping
//! unroll optimizer` — and raise the same `InvalidLoop("PYRE_NO_UNROLL")`.
//!
//! That is not a cosmetic difference. A frontend that deliberately leaves
//! `unroll` out of its `enable_opts` (measuring that peeling the preamble costs
//! it more than it returns is a normal outcome for a loop body with nothing
//! loop-invariant to hoist) gets a log naming an environment variable. A reader
//! checks their shell, finds it unset, and has been sent after a cause that
//! does not exist — the real one is a line of the frontend's own setup.

use majit_metainterp::unroll_skip_reason;

fn opts(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_string()).collect()
}

/// The full option list, as a frontend that wants unrolling would pass it.
const ALL_OPTS: &[&str] = &[
    "intbounds",
    "rewrite",
    "virtualize",
    "string",
    "pure",
    "earlyforce",
    "heap",
    "unroll",
];

#[test]
fn each_cause_is_named_as_itself() {
    let with_unroll = opts(ALL_OPTS);
    let reason = unroll_skip_reason(true, &with_unroll)
        .expect("the env override suppresses unrolling whatever the opts say");
    assert!(
        reason.contains("PYRE_NO_UNROLL"),
        "the env override must name the variable a reader can check; got {reason:?}",
    );

    // `ALL_OPTS` minus `unroll` — the shape a frontend that opted out passes.
    let without_unroll = opts(&ALL_OPTS[..ALL_OPTS.len() - 1]);
    let reason = unroll_skip_reason(false, &without_unroll)
        .expect("an enable_opts list without `unroll` suppresses unrolling");
    assert!(
        reason.contains("enable_opts"),
        "the configuration cause must name the configuration, not the env \
         variable a reader would then look for in vain; got {reason:?}",
    );
    assert!(
        !reason.contains("PYRE_NO_UNROLL"),
        "…and must not name the env override at all: it is unset in exactly \
         this case, which is what made the old message misleading; got {reason:?}",
    );
}

/// The control. Without it every assertion above is satisfied by a function
/// that suppresses unrolling unconditionally, which would be a far worse defect
/// than the message it was meant to fix.
#[test]
fn a_driver_that_asked_for_unrolling_gets_it() {
    assert_eq!(
        unroll_skip_reason(false, &opts(ALL_OPTS)),
        None,
        "`unroll` is listed and the env override is unset, so nothing suppresses it",
    );
}

/// An empty list is the absent-configuration case, not a request for unrolling.
///
/// Worth pinning separately: "no opts were configured" and "opts were
/// configured and `unroll` was left out" arrive here as the same value, and
/// both must suppress. A predicate that treated empty as "unrestricted" would
/// unroll for a driver that never asked.
#[test]
fn an_empty_option_list_suppresses_rather_than_permits() {
    assert!(
        unroll_skip_reason(false, &[]).is_some(),
        "an empty enable_opts does not list `unroll`, so unrolling stays off",
    );
}

/// Substring matching would accept a longer option that merely contains the
/// name. The comparison is on whole entries and this is what says so.
#[test]
fn a_different_option_containing_the_name_does_not_enable_unrolling() {
    assert!(
        unroll_skip_reason(false, &opts(&["unroll_safe", "heap"])).is_some(),
        "only an exact `unroll` entry enables unrolling",
    );
}
