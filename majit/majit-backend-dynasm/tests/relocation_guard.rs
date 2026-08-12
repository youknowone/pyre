//! `codebuf::finalize_writable` copies finalized code to an arena address that
//! differs from the one dynasm assembled it at. That is only sound while every
//! relocation this backend emits is PC-relative, because the address-dependent
//! ones live in dynasm's private `managed` list and no out-of-crate caller can
//! re-apply them after the move.
//!
//! `PatchLoc::needs_adjustment` is true for exactly two relocation kinds, and
//! the dynasm macro produces them from exactly two source forms:
//!
//! * `RelToAbs` — an `extern <addr>` jump target.
//! * `AbsToRel` — a label operand encoded absolutely, which on x86 means an
//!   8-byte jump-target immediate (`QWORD ->label` and friends). aarch64 emits
//!   every jump target with `relative_encoding = true`, so it cannot reach this
//!   kind at all.
//!
//! Neither form appears in this backend today. This test fails if one lands, so
//! the copy in `finalize_writable` stays sound.

use std::fs;
use std::path::{Path, PathBuf};

/// Data directives and operand size prefixes that encode a value absolutely.
/// Paired with a label reference (`->name` / `=>expr`) each yields `AbsToRel`.
const ABSOLUTE_ENCODINGS: [&str; 6] = ["QWORD", ".qword", ".u64", ".i64", "DQWORD", ".dword"];

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("backend src tree is readable") {
        let path = entry.expect("readable dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// The pieces of a source line that can be dynasm statements. Statements are
/// `;`-separated inside `dynasm!`, so anything after a `;` is a candidate;
/// ordinary Rust statements end with `;` and leave an empty tail.
fn statement_fragments(line: &str) -> impl Iterator<Item = &str> {
    line.split(';')
        .skip(1)
        .map(|fragment| fragment.split("//").next().unwrap_or("").trim())
        .filter(|fragment| !fragment.is_empty())
}

/// Why `fragment` would emit a relocation that a buffer move invalidates.
fn address_dependent_reason(fragment: &str) -> Option<&'static str> {
    if fragment
        .split_whitespace()
        .any(|token| token.trim_matches(',') == "extern")
    {
        return Some("`extern` jump target emits a RelToAbs relocation");
    }
    let mentions_label = fragment.contains("->") || fragment.contains("=>");
    if mentions_label
        && ABSOLUTE_ENCODINGS
            .iter()
            .any(|encoding| fragment.contains(encoding))
    {
        return Some("absolutely encoded label operand emits an AbsToRel relocation");
    }
    None
}

fn scan(text: &str) -> Vec<(usize, &'static str)> {
    let mut hits = Vec::new();
    for (index, line) in text.lines().enumerate() {
        // A `//`-led line is prose; dynasm statements never start there.
        if line.trim_start().starts_with("//") {
            continue;
        }
        for fragment in statement_fragments(line) {
            if let Some(reason) = address_dependent_reason(fragment) {
                hits.push((index + 1, reason));
            }
        }
    }
    hits
}

/// The scan above is what makes the corpus result below mean anything, so pin
/// that it separates the two forms it names from the ones this backend does use.
#[test]
fn scan_separates_address_dependent_forms() {
    let relative = "\
        dynasm!(self.mc\n\
            ; mov rax, QWORD helper as _\n\
            ; jne =>skip_wb\n\
            ; b ->slowpath\n\
        );\n";
    assert_eq!(scan(relative), Vec::new(), "PC-relative forms must pass");

    let rel_to_abs = "        ; call extern helper as _\n";
    assert_eq!(
        scan(rel_to_abs).len(),
        1,
        "an `extern` jump target must be caught"
    );

    let abs_to_rel = "        ; mov rax, QWORD =>resume_label\n";
    assert_eq!(
        scan(abs_to_rel).len(),
        1,
        "an 8-byte label operand must be caught"
    );
}

#[test]
fn no_address_dependent_relocations() {
    let mut sources = Vec::new();
    rust_sources(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut sources,
    );
    assert!(
        sources.len() > 1,
        "source scan found nothing to check: {sources:?}"
    );

    let mut violations = Vec::new();
    for path in &sources {
        let text = fs::read_to_string(path).expect("source file is utf-8");
        for (line_number, reason) in scan(&text) {
            violations.push(format!("{}:{line_number}: {reason}", path.display()));
        }
    }

    assert!(
        violations.is_empty(),
        "address-dependent relocations would survive the copy in \
         `codebuf::finalize_writable`, which moves finalized code to an arena \
         address dynasm never sees:\n{}",
        violations.join("\n")
    );
}
