//! Keeps `pyre-wasm-runner`'s positional diagnostic labels synchronized with
//! `MC_DIAG_LABELS`.
//!
//! The runner intentionally does not depend on the metainterpreter crate, so
//! this test compares the two source declarations and validates the parser
//! against `MC_DIAG_SLOTS`.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // `<root>/majit/majit-metainterp` -> `<root>`
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
/// The anchor is load-bearing on BOTH sides, and getting it wrong is silent:
///
/// * `main.rs` has THREE `let labels = [` arrays. A bare `let labels = ` match
///   takes the first (abort reasons, 6 entries) and reports 64 spurious
///   mismatches — a total-drift verdict manufactured entirely by the selector.
/// * `MC_DIAG_LABELS: [&str; MC_DIAG_SLOTS] = [...]` has a bracket in its TYPE.
///   Scanning for the first `[` after the name lands there and yields 0
///   entries.
///
/// So: anchor on something unique, then step to the `= [` that opens the value.
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

fn declared_slots(majit_src: &str) -> usize {
    let anchor = "pub const MC_DIAG_SLOTS: usize = ";
    let at = majit_src
        .find(anchor)
        .expect("majit: `MC_DIAG_SLOTS` declaration not found");
    let tail = &majit_src[at + anchor.len()..];
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits
        .parse()
        .unwrap_or_else(|e| panic!("majit: MC_DIAG_SLOTS is not a number ({digits:?}) — {e}"))
}

fn sources() -> (String, String) {
    let root = repo_root();
    (
        read(&root.join("majit/majit-metainterp/src/lib.rs")),
        read(&root.join("pyre/pyre-wasm-runner/src/main.rs")),
    )
}

fn extract(majit_src: &str, runner_src: &str) -> (Vec<String>, Vec<String>, usize) {
    let slots = declared_slots(majit_src);
    let majit = array_after(majit_src, "pub const MC_DIAG_LABELS", "majit");
    let runner = array_after(runner_src, "\"pyre_jit_mc_diag\"", "runner");
    (majit, runner, slots)
}

/// The slots whose two spellings disagree, over the shorter of the two.
fn divergences(majit: &[String], runner: &[String]) -> Vec<String> {
    (0..majit.len().min(runner.len()))
        .filter(|&i| majit[i] != runner[i])
        .map(|i| format!("  slot {i}: majit={:?} runner={:?}", majit[i], runner[i]))
        .collect()
}

/// Validates the source parser against the compiler-enforced slot count before
/// using the parser to diagnose mirror drift.
#[test]
fn slots_agree_with_the_declared_count() {
    let (majit_src, runner_src) = sources();
    let (majit, _runner, slots) = extract(&majit_src, &runner_src);
    assert_eq!(
        majit.len(),
        slots,
        "the majit-side extraction found {} labels but MC_DIAG_SLOTS is {slots}. \
         rustc already enforces that array's length, so THE PARSER IN THIS FILE \
         IS WRONG — do not touch MC_DIAG_LABELS. Most likely the anchor now \
         matches a different array, or an entry is not one-per-line.",
        majit.len(),
    );
}

#[test]
fn the_runner_mirror_matches_mc_diag_labels() {
    let (majit_src, runner_src) = sources();
    let (majit, runner, slots) = extract(&majit_src, &runner_src);

    assert_eq!(
        runner.len(),
        slots,
        "pyre-wasm-runner's mc_diag label array has {} entries but \
         MC_DIAG_SLOTS is {slots}. A new slot was added to \
         majit_metainterp::MC_DIAG_LABELS without being appended to the mirror \
         in pyre-wasm-runner/src/main.rs (or vice versa). APPEND it — inserting \
         in the middle silently renames every tally after the insertion point.\n\
         BUT FIRST: a count that is far off rather than off by a few means the \
         `\"pyre_jit_mc_diag\"` anchor matched a different array, and the \
         extraction is at fault, not the mirror. main.rs has three \
         `let labels = [` literals; the first is a 6-entry abort-reason list.",
        runner.len(),
    );

    let divergent = divergences(&majit, &runner);
    assert!(
        divergent.is_empty(),
        "the pyre-wasm-runner mc_diag mirror no longer matches \
         majit_metainterp::MC_DIAG_LABELS.\n{}\n\
         The runner zips these labels against `mc_diag(i)` BY INDEX, so every \
         tally from the first divergent slot onward is printed under the wrong \
         name. Fix the mirror in pyre-wasm-runner/src/main.rs to match majit — \
         majit is the authority, since its array is length-checked against \
         MC_DIAG itself.",
        divergent.join("\n"),
    );
}

/// POSITIVE CONTROL — proves the two tests above can fail.
///
/// A guard that has never been observed failing asserts nothing; `assert!(x
/// == x)` also passes on every tree. So both drift shapes are injected into
/// the REAL runner source, in memory, and the checks are required to catch
/// them. Perturbing the real text rather than a hand-written fixture is what
/// makes this a control over the ACTUAL extraction: a fixture would only
/// exercise `divergences`, leaving the anchors — the part that has already
/// been wrong twice — untested.
///
/// Nothing is written to disk; the test perturbs an in-memory copy.
#[test]
fn a_perturbed_mirror_is_detected() {
    let (majit_src, runner_src) = sources();
    let (majit, runner, slots) = extract(&majit_src, &runner_src);
    let last = runner.last().expect("runner mirror is empty").clone();

    // `replacen(.., 1)` takes the FIRST occurrence in the whole file, not the
    // first inside the mirror. Today the last label occurs exactly once, but a
    // future label colliding with a string in one of the three earlier
    // `let labels = [` arrays would put the perturbation OUTSIDE the extracted
    // region — and this test would then fail saying the element-wise comparison
    // missed a rename, accusing `divergences`, which would be innocent. Assert
    // the target is unique so a broken control reports itself as broken.
    let needle = format!("\"{last}\",");
    assert_eq!(
        runner_src.matches(&needle).count(),
        1,
        "the perturbation target {needle:?} occurs more than once in main.rs, so \
         `replacen` may patch text outside the mc_diag array and this control \
         would be measuring the wrong region. Pick a unique target — do NOT \
         read a failure below as mirror drift until this passes.",
    );

    // Shape 1 — RENAMED in place. The count still matches, so only the
    // element-wise comparison can see it.
    const INJECTED: &str = "XX_perturbed_by_the_positive_control";
    let renamed = runner_src.replacen(&needle, &format!("\"{INJECTED}\","), 1);
    assert_ne!(renamed, runner_src, "the perturbation did not apply");
    let (_, perturbed, _) = extract(&majit_src, &renamed);
    assert_eq!(
        perturbed.len(),
        slots,
        "a rename must not change the count — otherwise this control proves \
         the length check, not the comparison",
    );
    let caught = divergences(&majit, &perturbed);
    assert_eq!(
        caught.len(),
        1,
        "the element-wise comparison did not catch a single renamed label; \
         it reported {caught:?}",
    );

    // A COUNT is not an identity. Exactly one mismatch is also what a
    // MISALIGNED extraction produces — shift the runner array by one and
    // some other single slot disagrees, giving `len() == 1` while the
    // perturbation was never seen. Since positional agreement is the whole
    // property under guard, pin WHICH slot fired and WHAT it read: the
    // divergence must name the last slot and carry the injected string.
    assert!(
        caught[0].starts_with(&format!("  slot {}: ", slots - 1)) && caught[0].contains(INJECTED),
        "the comparison reported one divergence, but not the one that was \
         injected — expected slot {} carrying {INJECTED:?}, got {:?}. The \
         count matched by coincidence, so this control was passing without \
         observing the perturbation: suspect the extraction anchors, not \
         `divergences`.",
        slots - 1,
        caught[0],
    );

    // Shape 2 — DELETED. The remaining labels still agree pairwise up to the
    // deletion point, so only the length check can see it.
    let deleted = runner_src.replacen(&needle, "", 1);
    assert_ne!(
        deleted, runner_src,
        "the deletion did not apply. Asserted separately from the length \
         check below so a no-op substitution reports itself as such, rather \
         than as `divergences` failing to shorten",
    );
    let (_, shortened, _) = extract(&majit_src, &deleted);
    assert_eq!(
        shortened.len(),
        slots - 1,
        "removing one label did not shorten the extracted mirror by one",
    );
    assert!(
        divergences(&majit, &shortened).is_empty(),
        "a trailing deletion is invisible to the element-wise comparison by \
         construction — if this fires, `divergences` is no longer bounded by \
         the shorter side and shape 2 is being caught for the wrong reason",
    );
}
