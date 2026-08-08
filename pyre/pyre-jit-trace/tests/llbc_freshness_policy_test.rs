//! `PYRE_LLBC_STRICT` / `PYRE_LLBC_SKIP_FINGERPRINT_CHECK` interaction.
//!
//! Two switches on one guard, and before this was fixed they had two
//! independent ways of silently disarming it:
//!
//! * the opt-out was read first and returned, so `SKIP` defeated `STRICT` with
//!   no diagnostic that the escalation had been overridden; and
//! * the predicates disagreed — `SKIP` was a presence test (`…=0` still
//!   skipped) while `STRICT` demanded exactly `"1"` (`…=true` did nothing).
//!   The permissive parse was on the dangerous door and the exact one on the
//!   safe door.
//!
//! The rule these tests encode: **an unrecognised value is never silently the
//! same as unset**, and **asking for strict while also asking to skip is a
//! contradiction, not a precedence question.**

use pyre_jit_trace::llbc_fingerprint::{freshness_policy, FreshnessMode};

#[test]
fn both_unset_is_strict_and_says_nothing() {
    let p = freshness_policy(None, None);
    assert_eq!(p.mode, FreshnessMode::Strict);
    assert!(p.diagnostics.is_empty(), "{:?}", p.diagnostics);
}

#[test]
fn strict_one_promotes_to_error() {
    let p = freshness_policy(Some("1"), None);
    assert_eq!(p.mode, FreshnessMode::Strict);
    assert!(p.diagnostics.is_empty(), "{:?}", p.diagnostics);
}

/// The opt-out is honoured, but it announces itself. Returning silently would
/// make "opted out" and "checked and fresh" the same absence of output — the
/// conflation the UNKNOWN channel exists to prevent.
#[test]
fn skip_one_opts_out_but_says_so() {
    let p = freshness_policy(None, Some("1"));
    assert_eq!(p.mode, FreshnessMode::Skip);
    assert_eq!(p.diagnostics.len(), 1, "{:?}", p.diagnostics);
    assert!(p.diagnostics[0].contains("PYRE_LLBC_SKIP_FINGERPRINT_CHECK"));
}

/// The headline regression: asking for both is refused, not resolved by
/// precedence. Honouring SKIP lies to the caller; honouring STRICT overrides
/// an opt-out that exists for offline and vendored builds.
#[test]
fn strict_and_skip_together_is_refused() {
    let p = freshness_policy(Some("1"), Some("1"));
    assert_eq!(p.mode, FreshnessMode::Refuse);
    let joined = p.diagnostics.join("\n");
    assert!(joined.contains("PYRE_LLBC_STRICT"), "{joined}");
    assert!(
        joined.contains("PYRE_LLBC_SKIP_FINGERPRINT_CHECK"),
        "{joined}"
    );
}

/// An unrecognised strict spelling is reported while retaining the safe
/// strict-by-default policy.
#[test]
fn an_unrecognised_strict_value_is_not_silently_unset() {
    let p = freshness_policy(Some("true"), None);
    assert_eq!(p.mode, FreshnessMode::Strict);
    assert_eq!(p.diagnostics.len(), 1, "{:?}", p.diagnostics);
    assert!(p.diagnostics[0].contains("PYRE_LLBC_STRICT"));
    assert!(p.diagnostics[0].contains("true"));
}

/// `PYRE_LLBC_SKIP_FINGERPRINT_CHECK=0` used to skip — the presence test read
/// a value that plainly says "off" as "on".
#[test]
fn skip_zero_does_not_skip() {
    let p = freshness_policy(None, Some("0"));
    assert_eq!(p.mode, FreshnessMode::Strict);
    assert_eq!(p.diagnostics.len(), 1, "{:?}", p.diagnostics);
    assert!(p.diagnostics[0].contains("PYRE_LLBC_SKIP_FINGERPRINT_CHECK"));
}

/// And the two defects compounding: a junk opt-out value must not disarm a
/// correctly spelled escalation.
#[test]
fn a_junk_skip_value_does_not_defeat_strict() {
    let p = freshness_policy(Some("1"), Some("0"));
    assert_eq!(p.mode, FreshnessMode::Strict);
    assert_eq!(p.diagnostics.len(), 1, "{:?}", p.diagnostics);
    assert!(p.diagnostics[0].contains("PYRE_LLBC_SKIP_FINGERPRINT_CHECK"));
}

#[test]
fn an_empty_value_is_unrecognised_not_set() {
    assert_eq!(freshness_policy(None, Some("")).mode, FreshnessMode::Strict);
    assert_eq!(freshness_policy(Some(""), None).mode, FreshnessMode::Strict);
}
