//! Keeps `pyre-wasm-runner`'s positional fbw label array synchronized with
//! `pyre_jit_trace::trace::fbw_diag::LABELS`.
//!
//! The runner links no pyre crate — it is a wasmtime host that reaches the
//! counters through the `pyre_fbw_diag` wasm export — so it cannot import the
//! authority and instead restates it as a positional array. Both sides are
//! length-checked by rustc against their own constant (`RING_BASE` here,
//! `FBW_SLOTS` there), which catches a slot added on one side only; neither
//! compiler sees the SPELLINGS, so a rename drifts silently and every tally
//! from the divergence onward is printed under the wrong key. This test
//! compares the two source declarations as text, the same way
//! `majit-metainterp/tests/mc_diag_mirror.rs` does for `MC_DIAG_LABELS`.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // `<root>/pyre/pyre-jit-trace` -> `<root>`
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("CARGO_MANIFEST_DIR has fewer than two ancestors")
        .to_path_buf()
}

fn read(path: &Path) -> String {
    // A missing file must FAIL rather than skip. This test's whole job is to
    // compare two files; one that cannot find them and passes anyway is the
    // failure mode it exists to prevent, one level up.
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {} — {e}", path.display()))
}

/// The string literals of the first array literal following `anchor`.
///
/// Both declarations carry a `[` in their TYPE (`[&str; RING_BASE]` /
/// `[&str; FBW_SLOTS]`), so scanning for the first `[` after the name lands
/// there and yields zero entries. Anchor on the name, then step to the `= [`
/// that opens the value.
fn array_after(text: &str, anchor: &str, what: &str) -> Vec<String> {
    let at = text
        .find(anchor)
        .unwrap_or_else(|| panic!("{what}: anchor {anchor:?} not found"));
    let eq = text[at..]
        .find("= [")
        .unwrap_or_else(|| panic!("{what}: no `= [` after anchor {anchor:?}"))
        + at;
    let open = eq + text[eq..].find('[').expect("`= [` contains a `[`");

    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut end = open;
    for (i, b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    assert!(depth == 0 && end > open, "{what}: unbalanced array literal");

    let mut out = Vec::new();
    for line in text[open + 1..end].lines() {
        // Whole-line comments are dropped so a `"` inside prose cannot be
        // mistaken for an entry. Entries are one per line in both files.
        let line = line.trim();
        if line.starts_with("//") {
            continue;
        }
        let mut rest = line;
        while let Some(s) = rest.find('"') {
            let after = &rest[s + 1..];
            let Some(e) = after.find('"') else { break };
            out.push(after[..e].to_string());
            rest = &after[e + 1..];
        }
    }
    out
}

/// The integer literal a `usize` constant is declared with.
fn declared_count(src: &str, anchor: &str, what: &str) -> usize {
    let at = src
        .find(anchor)
        .unwrap_or_else(|| panic!("{what}: {anchor:?} declaration not found"));
    let tail = &src[at + anchor.len()..];
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits
        .parse()
        .unwrap_or_else(|e| panic!("{what}: count is not a number ({digits:?}) — {e}"))
}

fn sources() -> (String, String) {
    let root = repo_root();
    (
        read(&root.join("pyre/pyre-jit-trace/src/trace.rs")),
        read(&root.join("pyre/pyre-wasm-runner/src/main.rs")),
    )
}

fn extract(trace_src: &str, runner_src: &str) -> (Vec<String>, Vec<String>, usize) {
    let slots = declared_count(trace_src, "pub const RING_BASE: usize = ", "pyre-jit-trace");
    let authority = array_after(trace_src, "pub const LABELS", "pyre-jit-trace");
    let runner = array_after(runner_src, "let fbw_labels", "runner");
    (authority, runner, slots)
}

/// The slots whose two spellings disagree, over the shorter of the two.
fn divergences(authority: &[String], runner: &[String]) -> Vec<String> {
    (0..authority.len().min(runner.len()))
        .filter(|&i| authority[i] != runner[i])
        .map(|i| {
            format!(
                "  slot {i}: fbw_diag={:?} runner={:?}",
                authority[i], runner[i]
            )
        })
        .collect()
}

/// Validates the source parser against the compiler-enforced slot count before
/// using the parser to diagnose mirror drift.
#[test]
fn slots_agree_with_the_declared_count() {
    let (trace_src, runner_src) = sources();
    let (authority, _runner, slots) = extract(&trace_src, &runner_src);
    assert_eq!(
        authority.len(),
        slots,
        "the fbw_diag-side extraction found {} labels but RING_BASE is {slots}. \
         rustc already enforces that array's length, so THE PARSER IN THIS FILE \
         IS WRONG — do not touch fbw_diag::LABELS. Most likely the anchor now \
         matches a different array, or an entry is not one-per-line.",
        authority.len(),
    );
}

/// The runner's own constant must agree too — otherwise a slot added on one
/// side compiles on both and only the zip length changes.
#[test]
fn the_runner_declares_the_same_slot_count() {
    let (trace_src, runner_src) = sources();
    let slots = declared_count(
        &trace_src,
        "pub const RING_BASE: usize = ",
        "pyre-jit-trace",
    );
    let mirrored = declared_count(&runner_src, "const FBW_SLOTS: usize = ", "runner");
    assert_eq!(
        mirrored, slots,
        "pyre-wasm-runner's FBW_SLOTS is {mirrored} but \
         pyre_jit_trace::trace::fbw_diag::RING_BASE is {slots}. The runner reads \
         `pyre_fbw_diag(i)` for i in 0..FBW_SLOTS, so a low count silently drops \
         the tail slots from the wasm `[jit-stats] fbw_diag` line and a high one \
         reads past the tallies into the ring.",
    );
}

#[test]
fn the_runner_mirror_matches_fbw_diag_labels() {
    let (trace_src, runner_src) = sources();
    let (authority, runner, slots) = extract(&trace_src, &runner_src);

    assert_eq!(
        runner.len(),
        slots,
        "pyre-wasm-runner's fbw label array has {} entries but RING_BASE is \
         {slots}. A tally slot was added to fbw_diag::LABELS without being \
         appended to the mirror in pyre-wasm-runner/src/main.rs (or vice versa). \
         APPEND it — inserting in the middle silently renames every tally after \
         the insertion point.\n\
         BUT FIRST: a count that is far off rather than off by a few means the \
         `let fbw_labels` anchor matched a different array, and the extraction is \
         at fault, not the mirror.",
        runner.len(),
    );

    let divergent = divergences(&authority, &runner);
    assert!(
        divergent.is_empty(),
        "the pyre-wasm-runner fbw mirror no longer matches \
         pyre_jit_trace::trace::fbw_diag::LABELS.\n{}\n\
         The runner zips these labels against `pyre_fbw_diag(i)` BY INDEX, so \
         every tally from the first divergent slot onward is printed under the \
         wrong name — and check.py merges every `[jit-stats]` line into one flat \
         key->value map, so a wrong name is compared against the wrong baseline \
         rather than reported as missing. Fix the mirror in \
         pyre-wasm-runner/src/main.rs to match pyre-jit-trace, which is the \
         authority: its array is length-checked against RING_BASE, which is the \
         same constant the counter array itself is sized by.",
        divergent.join("\n"),
    );
}

/// The text comparison above proves the two ARRAYS agree; it cannot prove
/// either one agrees with the slot CONSTANTS, which is what the bump sites
/// actually index by. Both arrays staying in step while `LABELS` shifts
/// against `WALKS`/`MIDBODY_LATCH`/... would rename every tally on both
/// backends at once, which no diff between them can see. Bind each label to
/// its own constant.
///
/// Two entries deliberately break the mechanical `ESCAPE_*` -> `fbw_escape_*`
/// reading — `ESCAPE_FORCE_BY_PORTAL` is `fbw_force_by_portal`, not
/// `fbw_escape_force_by_portal` — which is why this is written out rather than
/// derived from the constant names.
#[test]
fn each_label_sits_at_its_own_slot_constant() {
    use pyre_jit_trace::trace::fbw_diag as d;

    let bindings = [
        (d::WALKS, "fbw_walks"),
        (d::ROLLED_BACK_WITH_EFFECTS, "fbw_rolled_back_with_effects"),
        (d::MIDBODY_LATCH, "fbw_midbody_latch"),
        (
            d::MIDBODY_LATCH_NEW_UNJOURNALED,
            "fbw_midbody_latch_new_unjournaled",
        ),
        (d::ESCAPE_PLAIN_FALLBACK, "fbw_escape_plain_fallback"),
        (
            d::ESCAPE_PLAIN_FALLBACK_UNCLEAN,
            "fbw_escape_plain_fallback_unclean",
        ),
        (d::ESCAPE_PORTAL_ONLY, "fbw_escape_portal_only"),
        (
            d::ESCAPE_PUBLISHED_CALLEE_ONLY,
            "fbw_escape_published_callee_only",
        ),
        (
            d::ESCAPE_PORTAL_AND_PUBLISHED_CALLEE,
            "fbw_escape_portal_and_published_callee",
        ),
        (d::ESCAPE_FORCE_BY_PORTAL, "fbw_force_by_portal"),
        (d::ESCAPE_FORCE_BY_CALLEE_ONLY, "fbw_force_by_callee_only"),
        (
            d::STORE_JOURNAL_ROLLBACK_FAILED,
            "fbw_store_journal_rollback_failed",
        ),
        (
            d::BLACKHOLE_ADOPTED_SINGLE_FRAME,
            "fbw_blackhole_adopted_single_frame",
        ),
        (
            d::BLACKHOLE_ADOPTED_MULTI_FRAME,
            "fbw_blackhole_adopted_multi_frame",
        ),
        (d::GATE_DECLINED_SHAPE, "gate_declined_shape"),
        (
            d::GATE_DECLINED_FOR_ITER_REGION,
            "gate_declined_for_iter_region",
        ),
        (
            d::GATE_DECLINED_FUNCTION_ENTRY,
            "gate_declined_function_entry",
        ),
        (d::BRIDGE_EC_FROM_PORTAL_RED, "bridge_ec_from_portal_red"),
        (d::BRIDGE_EC_MISSING, "bridge_ec_missing"),
    ];

    // Exhaustiveness: a slot added to `fbw_diag` grows `LABELS` (rustc enforces
    // that) and the runner mirror (the length check above enforces that), but
    // neither would notice that its constant went unbound here. Require the
    // bound slots to be exactly `0..RING_BASE`.
    let mut bound: Vec<usize> = bindings.iter().map(|&(slot, _)| slot).collect();
    bound.sort_unstable();
    assert_eq!(
        bound,
        (0..d::RING_BASE).collect::<Vec<_>>(),
        "the slot constants bound below are not exactly 0..RING_BASE — a new \
         fbw_diag slot was added without binding its constant to its label here",
    );

    for (slot, expected) in bindings {
        assert_eq!(
            d::LABELS[slot],
            expected,
            "slot {slot} is named {:?} but the constant that indexes it means \
             {expected:?}. A slot constant was inserted or renumbered without \
             moving its entry in LABELS, so every bump from this slot on is \
             reported under another slot's key — on BOTH backends at once, \
             which is why the runner-mirror diff cannot see it.",
            d::LABELS[slot],
        );
    }
}

/// POSITIVE CONTROL — proves the checks above can fail.
///
/// A guard that has never been observed failing asserts nothing. Both drift
/// shapes are injected into the REAL runner source, in memory, and the checks
/// are required to catch them. Perturbing the real text rather than a
/// hand-written fixture is what makes this a control over the ACTUAL
/// extraction: a fixture would only exercise `divergences`, leaving the
/// anchors — the part that is easiest to get silently wrong — untested.
#[test]
fn the_mirror_check_catches_injected_drift() {
    let (trace_src, runner_src) = sources();
    let (authority, runner, slots) = extract(&trace_src, &runner_src);
    assert_eq!(runner.len(), slots, "control needs a green starting tree");

    // Both injections edit only the text FROM the anchor onward, so a future
    // comment that happens to quote one of these keys ahead of the array
    // cannot absorb the perturbation and turn this control silently green.
    let anchor_at = runner_src
        .find("let fbw_labels")
        .expect("control: runner anchor already validated above");
    let (head, tail) = runner_src.split_at(anchor_at);
    let inject = |from: &str, to: &str| format!("{head}{}", tail.replacen(from, to, 1));

    // Shape 1 — a rename. Repoint one slot at a name from a different slot so
    // the entry stays a plausible fbw key and only the ORDER is wrong.
    let renamed = inject(
        &format!("\"{}\",", runner[1]),
        &format!("\"{}\",", runner[2]),
    );
    assert_ne!(renamed, runner_src, "control: rename injection was a no-op");
    let drifted = array_after(&renamed, "let fbw_labels", "runner");
    assert!(
        !divergences(&authority, &drifted).is_empty(),
        "a renamed slot went undetected — `divergences` or the runner anchor is \
         broken, and this file is guarding nothing",
    );

    // Shape 2 — a dropped slot. The length check, not `divergences`, is what
    // has to catch this: removing the LAST entry leaves every surviving slot
    // correctly named.
    let dropped = inject(&format!("\"{}\",\n", runner[slots - 1]), "");
    assert_ne!(dropped, runner_src, "control: drop injection was a no-op");
    let short = array_after(&dropped, "let fbw_labels", "runner");
    assert_eq!(
        short.len(),
        slots - 1,
        "the drop injection did not shorten the extracted array, so the length \
         assertion in the mirror test is untested",
    );
    assert!(
        divergences(&authority, &short).is_empty(),
        "control assumption broken: dropping the LAST slot should leave every \
         remaining slot correctly named, so only the length check can catch it",
    );
}
