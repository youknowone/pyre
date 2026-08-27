//! Keeps `pyre-wasm-runner`'s positional bridge-diag labels synchronized with
//! [`majit_backend_wasm::BRIDGE_DIAG_LABELS`].
//!
//! The runner is a wasmtime host that reaches the tallies through the
//! `pyre_jit_bridge_diag` guest export; it links no majit crate, so it
//! restates the keys as a positional array. `BRIDGE_DIAG` is sized by the
//! authority array, which catches a tally added without a name on THIS side —
//! but nothing on the runner's side sees either the count or the spellings.
//! Both drift shapes are silent: a short mirror stops printing the tail (slot
//! 56 was bumped in the guest and reported by nobody until this test was
//! written), and a rename prints every tally from the divergence onward under
//! the wrong key.
//!
//! The authority is LINKED, not parsed. Only the runner is read as text.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // `<root>/majit/majit-backend-wasm` -> `<root>`
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("CARGO_MANIFEST_DIR has fewer than two ancestors")
        .to_path_buf()
}

fn runner_source() -> String {
    // A missing file must FAIL rather than skip. This test's whole job is to
    // read that file; one that cannot find it and passes anyway is the failure
    // mode it exists to prevent, one level up.
    let path = repo_root().join("pyre/pyre-wasm-runner/src/main.rs");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {} — {e}", path.display()))
}

/// The string literals of the first array literal following `anchor`.
///
/// The anchor is load-bearing and getting it wrong is silent: `main.rs` holds
/// four positional label arrays, and selecting the wrong one reports total
/// drift manufactured entirely by the selector. Anchor on the export name,
/// which occurs once, then step to the `= [` that opens the value — a
/// declaration can carry a `[` in its TYPE, which scanning for the first `[`
/// after the anchor would land on.
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

const RUNNER_ANCHOR: &str = "\"pyre_jit_bridge_diag\"";

fn mirror(runner_src: &str) -> Vec<String> {
    array_after(runner_src, RUNNER_ANCHOR, "runner")
}

/// The slots whose two spellings disagree, over the shorter of the two.
fn divergences(authority: &[&str], runner: &[String]) -> Vec<String> {
    (0..authority.len().min(runner.len()))
        .filter(|&i| authority[i] != runner[i].as_str())
        .map(|i| {
            format!(
                "  slot {i}: majit={:?} runner={:?}",
                authority[i], runner[i]
            )
        })
        .collect()
}

#[test]
fn the_runner_mirror_matches_bridge_diag_labels() {
    let runner_src = runner_source();
    let authority = majit_backend_wasm::BRIDGE_DIAG_LABELS;
    let runner = mirror(&runner_src);

    assert_eq!(
        runner.len(),
        authority.len(),
        "pyre-wasm-runner's bridge_diag label array has {} entries but \
         BRIDGE_DIAG_LABELS has {}. A tally was added to one side only. APPEND \
         it — inserting in the middle silently renames every tally after the \
         insertion point.\n\
         BUT FIRST: a count that is far off rather than off by a few means the \
         {RUNNER_ANCHOR} anchor matched a different array, and the extraction \
         is at fault rather than the mirror. main.rs holds four positional \
         label arrays.",
        runner.len(),
        authority.len(),
    );

    let divergent = divergences(authority, &runner);
    assert!(
        divergent.is_empty(),
        "the pyre-wasm-runner bridge_diag mirror no longer matches \
         majit_backend_wasm::BRIDGE_DIAG_LABELS.\n{}\n\
         The runner zips these labels against `pyre_jit_bridge_diag(i)` BY \
         INDEX, so every tally from the first divergent slot onward is printed \
         under the wrong name — and check.py folds every `[jit-stats]` line \
         into one flat key->value map, so a wrong name is compared against the \
         wrong baseline rather than reported as missing. majit is the \
         authority: `BRIDGE_DIAG` itself is sized by that array's length.",
        divergent.join("\n"),
    );
}

/// POSITIVE CONTROL — proves the test above can fail.
///
/// A guard that has never been observed failing asserts nothing. Both drift
/// shapes are injected into the REAL runner source, in memory, and the checks
/// are required to catch them. Perturbing the real text rather than a
/// hand-written fixture is what makes this a control over the ACTUAL
/// extraction, anchors included.
///
/// Nothing is written to disk; the test perturbs an in-memory copy.
#[test]
fn the_mirror_check_catches_injected_drift() {
    let runner_src = runner_source();
    let authority = majit_backend_wasm::BRIDGE_DIAG_LABELS;
    let runner = mirror(&runner_src);
    assert_eq!(
        runner.len(),
        authority.len(),
        "control needs a green starting tree",
    );

    // Both injections edit only the text FROM the anchor onward, so a comment
    // elsewhere that happens to quote one of these keys cannot absorb the
    // perturbation and turn this control silently green.
    let anchor_at = runner_src
        .find(RUNNER_ANCHOR)
        .expect("control: runner anchor already validated above");
    let (head, tail) = runner_src.split_at(anchor_at);
    let inject = |from: &str, to: &str| format!("{head}{}", tail.replacen(from, to, 1));

    // Shape 1 — a rename. Only the element-wise comparison can see it, since
    // the count is unchanged.
    const INJECTED: &str = "XX_perturbed_by_the_positive_control";
    let renamed = inject(&format!("\"{}\",", runner[1]), &format!("\"{INJECTED}\","));
    assert_ne!(renamed, runner_src, "control: rename injection was a no-op");
    let drifted = mirror(&renamed);
    assert_eq!(
        drifted.len(),
        authority.len(),
        "a rename must not change the count — otherwise this control proves \
         the length check, not the comparison",
    );
    let caught = divergences(authority, &drifted);
    // A COUNT is not an identity: exactly one mismatch is also what a
    // MISALIGNED extraction produces. Pin WHICH slot fired and WHAT it read.
    assert_eq!(caught.len(), 1, "expected one divergence, got {caught:?}");
    assert!(
        caught[0].starts_with("  slot 1: ") && caught[0].contains(INJECTED),
        "the comparison reported one divergence, but not the one that was \
         injected — expected slot 1 carrying {INJECTED:?}, got {:?}. The count \
         matched by coincidence, so this control was passing without observing \
         the perturbation: suspect the extraction anchor, not `divergences`.",
        caught[0],
    );

    // Shape 2 — a dropped slot. Removing the LAST entry leaves every surviving
    // slot correctly named, so only the length check can catch it. This is the
    // shape that was live before this file existed.
    let last = runner.last().expect("runner mirror is empty");
    let dropped = inject(&format!("\"{last}\",\n"), "");
    assert_ne!(dropped, runner_src, "control: drop injection was a no-op");
    let short = mirror(&dropped);
    assert_eq!(
        short.len(),
        authority.len() - 1,
        "the drop injection did not shorten the extracted array, so the length \
         assertion in the mirror test is untested",
    );
    assert!(
        divergences(authority, &short).is_empty(),
        "control assumption broken: dropping the LAST slot should leave every \
         remaining slot correctly named, so only the length check can catch it",
    );
}
