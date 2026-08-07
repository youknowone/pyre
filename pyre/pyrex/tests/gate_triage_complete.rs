//! Every `PYRE_*` environment gate read from a workspace member's Rust source
//! must have a live entry in `pyre/gate-triage.md`.
//!
//! The charter (§3.6) says a gate is a staging area, not a home, and
//! `gate-triage.md` is the standing list of what to retire and when. That list
//! only works if a gate joins it when it is born: audited by hand, it drifted to
//! 63% empty — 66 of 105 live gates were absent — because nothing failed when a
//! new gate skipped it. This test is that failure.
//!
//! Adding a gate therefore costs one row. The row is cheap; the alternative is
//! another hand audit that goes stale the week after it lands.
//!
//! **Scope: Rust only.** `PYRE_CHECK_PYPY3`, `PYRE_CHECK_PYTHON3`,
//! `PYRE_SHARED_BUILD` and `PYRE_SYNTH_PYPY` are live gates read from `check.py`,
//! `check_synthetic.py`, the CI workflows and `scripts/llbc_extract.py`, and
//! nothing here sees them. A Python- or YAML-only gate can still enter
//! undocumented; whether to widen this scan is tracked as F5 in `rework.md`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    // <root>/pyre/pyrex/ -> <root>
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("pyrex manifest sits two levels below the repo root")
        .to_path_buf()
}

/// Workspace member directories, read from the root `Cargo.toml`.
///
/// Deriving the search roots from the member list rather than walking the whole
/// repository is what keeps untracked scratch copies of source files out: a
/// `.rs` file only reaches a compiler if it belongs to a member crate, so only
/// those can hold a live gate. It also needs no maintenance — a new crate has to
/// join `members` to build at all.
fn workspace_member_dirs(root: &Path) -> Vec<PathBuf> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("read root Cargo.toml");
    // Anchored at line start: `default-members = [` sits above `members = [`
    // and an unanchored search finds that one instead.
    let after = manifest
        .split_once("\nmembers = [")
        .expect("root Cargo.toml has a members list")
        .1;
    let list = after.split_once(']').expect("members list is closed").0;
    let dirs: Vec<PathBuf> = list
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let line = line.strip_prefix('"')?;
            Some(root.join(line.split_once('"')?.0))
        })
        .collect();
    assert!(
        dirs.len() > 10,
        "parsed only {} workspace members — the Cargo.toml parse is wrong",
        dirs.len()
    );
    dirs
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // A member crate's own `target/` from a standalone build.
            if entry.file_name() != "target" {
                collect_rs(&path, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The function names that read an environment variable in this tree.
///
/// `env::var` (with its `_os` suffix) is the common one, but a search for it
/// alone misses the host seam: `host_os::var` on the sandbox path and
/// `host_seam::ops::getenv`, which takes a *byte* string. `PYRE_STDLIB` is read
/// through both seam forms in `pyre-interpreter/src/importing.rs` and stays
/// visible to a `std::env` search only because unrelated `env::var` readers
/// exist in `pyre-wasm-runner`. A wasm- or sandbox-only gate would have no such
/// cover and would bypass the brake entirely.
const READ_FORMS: [&str; 3] = ["env::var", "host_os::var", "getenv"];

/// Gate names this text reads from the environment.
///
/// Matches the read forms above rather than every mention of the name, so a gate
/// discussed in a comment or held in a Rust const does not count as live. That
/// is the same distinction `gate-triage.md` §2 draws by hand.
fn gates_read_by(text: &str) -> Vec<&str> {
    let mut found = Vec::new();
    for form in READ_FORMS {
        for (at, _) in text.match_indices(form) {
            let rest = &text[at + form.len()..];
            // `env::var_os` is the same read wearing a suffix.
            let rest = rest.strip_prefix("_os").unwrap_or(rest);
            let Some(rest) = rest.strip_prefix('(') else {
                continue;
            };
            let rest = rest.trim_start();
            // `host_seam::ops::getenv` names its gate with a byte string.
            let rest = rest.strip_prefix('b').unwrap_or(rest);
            let Some(rest) = rest.strip_prefix('"') else {
                continue;
            };
            let Some(end) = rest.find('"') else { continue };
            let name = &rest[..end];
            if name.starts_with("PYRE_") {
                found.push(name);
            }
        }
    }
    found
}

#[test]
fn gates_read_by_matches_the_read_forms_and_nothing_else() {
    let sample = r#"
        std::env::var("PYRE_A").is_ok();
        env::var_os("PYRE_B").is_none();
        std::env::var(
            "PYRE_C",
        );
        host_os::var("PYRE_D").ok();
        crate::host_seam::ops::getenv(b"PYRE_E");
        // PYRE_MENTIONED_IN_A_COMMENT
        const PYRE_CONST: &str = "PYRE_NOT_A_READ";
        other::var("PYRE_NOT_ENV");
        env::var("HOME");
    "#;
    let mut got = gates_read_by(sample);
    // Sorted: the scan groups by read form, so the order carries no meaning.
    got.sort_unstable();
    assert_eq!(got, vec!["PYRE_A", "PYRE_B", "PYRE_C", "PYRE_D", "PYRE_E"]);
}

/// Does this `##` heading introduce a section that records history?
///
/// §1/§1b/§1c list gates whose readers were deleted, §2 lists names that were
/// never env vars, and §3 lists names with no read site. A name there is the
/// record of a gate that is *gone*, so it must not satisfy the brake — otherwise
/// re-introducing a reader for `PYRE_SINGLE_PASS` lands green on the strength of
/// its own retirement row, with neither its new polarity nor a retirement plan
/// written down.
fn is_history_heading(heading: &str) -> bool {
    let lower = heading.to_ascii_lowercase();
    // "retired", not "retire": §4 is a *live* section whose heading says when
    // its gates will go ("retire when the epic closes").
    lower.contains("retired") || lower.contains("dead") || lower.contains("not gates")
}

/// Every `PYRE_*` token the triage document lists as a live gate.
///
/// Tokenized rather than substring-searched: `contains("PYRE_A")` is satisfied
/// by a documented `PYRE_ANCHOR_STRICT`, so a new gate whose name is a prefix of
/// a listed one would slip through the brake unnoticed. Scoped to the live
/// sections for the reason in `is_history_heading`. `###` subsections inherit
/// their `##` parent, which is what keeps §6a–§6c live under §6.
fn gates_documented_in(triage: &str) -> BTreeSet<&str> {
    let mut found = BTreeSet::new();
    let mut live = true;
    for line in triage.lines() {
        if let Some(heading) = line.strip_prefix("## ") {
            live = !is_history_heading(heading);
        }
        if !live {
            continue;
        }
        for (at, _) in line.match_indices("PYRE_") {
            // A name preceded by a name character is the tail of a longer token.
            if line[..at]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                continue;
            }
            let end = line[at..]
                .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
                .map_or(line.len(), |off| at + off);
            found.insert(&line[at..end]);
        }
    }
    found
}

#[test]
fn every_live_pyre_gate_has_a_gate_triage_entry() {
    let root = repo_root();
    let triage_path = root.join("pyre/gate-triage.md");
    let triage = std::fs::read_to_string(&triage_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", triage_path.display()));
    let documented = gates_documented_in(&triage);

    // Two anchors on the live/history split, because a heading reword would
    // otherwise change what this test accepts without failing. One on each side:
    // a retired name must not count, and a listed live name must.
    assert!(
        !documented.contains("PYRE_SINGLE_PASS"),
        "PYRE_SINGLE_PASS is named only in a retirement section and has no read \
         site, so it must not count as documented — is_history_heading no longer \
         matches gate-triage.md's headings"
    );
    assert!(
        documented.contains("PYRE_JD1"),
        "PYRE_JD1 is listed live in §6a but did not count as documented — \
         is_history_heading is excluding a live section"
    );

    let mut sources = Vec::new();
    for member in workspace_member_dirs(&root) {
        collect_rs(&member, &mut sources);
    }
    assert!(
        sources.len() > 100,
        "found only {} .rs files across the workspace members — the walk is not \
         reaching the tree",
        sources.len()
    );
    // This file's own fixture spells out `env::var("PYRE_A")` and friends, which
    // are test data rather than gates.
    let self_path = root.join(file!());

    let mut missing: BTreeSet<(String, String)> = BTreeSet::new();
    for path in &sources {
        if *path == self_path {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        for name in gates_read_by(&text) {
            if !documented.contains(name) {
                let rel = path.strip_prefix(&root).unwrap_or(path);
                missing.insert((name.to_string(), rel.display().to_string()));
            }
        }
    }

    if !missing.is_empty() {
        let listed = missing
            .iter()
            .map(|(name, file)| format!("  {name}  ({file})"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "{} PYRE_* gate(s) are read from the environment but have no entry in \
             pyre/gate-triage.md:\n{listed}\n\n\
             Add a row for each: what it gates, its default polarity, and — for a \
             default-ON experiment — the epic whose close retires it.",
            missing.len()
        );
    }
}
