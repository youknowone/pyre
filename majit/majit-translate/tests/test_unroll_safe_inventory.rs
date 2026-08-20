//! Every `unroll_safe` in the shipped interpreter LLBC is one that was
//! reviewed.
//!
//! `unroll_safe` is not a loop annotation.  `codewriter/policy.rs`
//! `look_inside_graph` cancels `contains_loop` for a hinted graph, so the
//! attribute changes *what the walker descends into*: the hinted graph and
//! its whole callee closure enter the candidate set.  The tree's history
//! with that is why this file exists — the same three attributes were added,
//! reverted for a SIGBUS traced to an sret ABI mismatch in a callee the hint
//! newly reached, and only re-landed once that callee was published
//! correctly.
//!
//! So an addition here is a descent-scope change that needs its own
//! evidence, and this test makes adding one without saying so fail.  It is
//! deliberately a *subset* check rather than an equality check: a developer's
//! `build/llbc` is routinely older than the source (a `pyre-interpreter`
//! edit is invisible until re-extraction), and a stale artefact must not
//! produce a false red.  A stale artefact can only under-report, which
//! passes; a new hint can only over-report, which fails.

use majit_charon_reader::Llbc;
use majit_translate::front::llbc_hints::harvest_hints_from_llbcs;

const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc"
);

/// The leaf name of every function reviewed as `unroll_safe`, with the
/// upstream decorator it mirrors.
///
/// Matched on the leaf rather than the full path because the harvester
/// spells a method with its impl block (`pyframe::<Impl>::fast2locals`) and
/// a free function without one (`builtins::leading_non_null_count`); the
/// leaves are unambiguous across the interpreter.
const REVIEWED_UNROLL_SAFE: &[(&str, &str)] = &[
    // `abstractinst.py` carries `@jit.unroll_safe` on all three.
    (
        "isinstance",
        "abstractinst.py p_recursive_isinstance_w caller",
    ),
    ("issubclass", "abstractinst.py abstract_issubclass_w caller"),
    (
        "p_abstract_issubclass_w",
        "abstractinst.py _abstract_issubclass_w",
    ),
    // `pyframe.py` `fast2locals`.
    ("fast2locals", "pyframe.py fast2locals"),
    // No upstream counterpart by name; the loop is a bounded scan of a
    // fixed-size argument slice.
    ("leading_non_null_count", "flat builtin-keyword ABI scan"),
];

/// `builtins::leading_non_null_count` has carried its own `unroll_safe`
/// since it was introduced and appears in every recorded cache snapshot, so
/// its absence means the artefact is too old to say anything — skip loudly
/// rather than pass on a read that proves nothing.
const CONTROL: &str = "leading_non_null_count";

fn harvested_unroll_safe() -> Option<Vec<String>> {
    if !std::path::Path::new(INTERPRETER_LLBC).is_file() {
        eprintln!(
            "skipping: {INTERPRETER_LLBC} is missing; run \
             `python3 scripts/extract-llbc.py pyre-interpreter`"
        );
        return None;
    }
    let llbc = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter.ullbc");
    let hints = harvest_hints_from_llbcs(std::slice::from_ref(&llbc));
    let mut paths: Vec<String> = hints
        .iter()
        .filter(|(_, values)| values.iter().any(|h| h == "unroll_safe"))
        .map(|(path, _)| path.clone())
        .collect();
    paths.sort();
    if !paths.iter().any(|p| leaf(p) == CONTROL) {
        eprintln!(
            "skipping: {INTERPRETER_LLBC} carries no `unroll_safe` on {CONTROL}, \
             so it predates the hint inventory entirely; re-extract to exercise \
             this test (harvested: {paths:?})"
        );
        return None;
    }

    // `REVIEWED_UNROLL_SAFE` and the subset check below both match on the
    // leaf, on the stated assumption that leaves are unambiguous across the
    // interpreter.  Nothing else verifies that.  If an unreviewed function
    // elsewhere later takes an already-reviewed leaf name and is hinted, the
    // subset check would pass on the strength of the other function's review
    // — the one outcome this file exists to prevent.
    let mut leaves: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for path in &paths {
        if let Some(previous) = leaves.insert(leaf(path), path.as_str()) {
            panic!(
                "leaf `{}` is ambiguous between {previous} and {path}. \
                 REVIEWED_UNROLL_SAFE matches by leaf, so it cannot tell them \
                 apart and one would ride on the other's review; key the \
                 inventory by full path before adding either.",
                leaf(path),
            );
        }
    }
    Some(paths)
}

fn leaf(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path)
}

#[test]
fn every_unroll_safe_in_the_shipped_llbc_is_a_reviewed_one() {
    let Some(paths) = harvested_unroll_safe() else {
        return;
    };
    for path in &paths {
        assert!(
            REVIEWED_UNROLL_SAFE
                .iter()
                .any(|(name, _)| *name == leaf(path)),
            "{path} carries `unroll_safe` but is not in REVIEWED_UNROLL_SAFE. \
             The attribute admits this graph and its callee closure into the \
             candidate set, so it needs its own evidence — measure \
             `fbw_rolled_back_with_effects` (a rise is a correctness verdict, \
             not a statistic) and the per-fixture jitstats before adding it, \
             then list it here. Harvested: {paths:?}",
        );
    }
}

/// `_unpackiterable_known_length_jitlook` quotes upstream's
/// `@jit.unroll_safe` in its own doc comment, which reads as an unfinished
/// port and has been picked up as one.  It is not.
///
/// Upstream hints that body for `unpackiterable_unroll`, whose
/// `expected_length` is an UNPACK_SEQUENCE oparg; `unpackiterable` reaches
/// it through `_unpackiterable_known_length`, which is
/// `@jit.dont_look_inside` — "the JIT stopped looking inside already".  pyre
/// has neither `unpackiterable_unroll` nor the shim, so `unpackiterable` is
/// the body's only caller: the hinted path upstream keeps closed.  Carrying
/// the attribute alone inverts that decision instead of matching it.
#[test]
fn the_known_length_unpack_body_stays_unhinted_without_its_shim() {
    let Some(paths) = harvested_unroll_safe() else {
        return;
    };
    assert!(
        !paths
            .iter()
            .any(|p| leaf(p) == "_unpackiterable_known_length_jitlook"),
        "`unroll_safe` on _unpackiterable_known_length_jitlook opens the path \
         upstream fences with the `@jit.dont_look_inside` shim \
         `_unpackiterable_known_length`, which pyre does not have. Port the \
         shim (and `unpackiterable_unroll`) with an ABI-correct publication \
         first — the signature returns `Result<Vec<PyObjectRef>, PyError>`, \
         which `helper_call_kind_for_type` answers `Unsupported` for. \
         Harvested: {paths:?}",
    );
}
