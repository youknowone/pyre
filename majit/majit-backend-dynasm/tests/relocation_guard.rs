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
//!
//! Both forms are assembly syntax, so the scan reads only the inside of a
//! `dynasm!` body. Reading whole Rust files instead would collide with the host
//! language: `unsafe extern "C" { .. }` is not a jump target, and a `match` arm
//! next to an unrelated `QWORD` operand is not a label reference.

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

/// One `;`-separated statement of a `dynasm!` body.
#[derive(Debug, PartialEq, Eq)]
struct Statement {
    /// 1-based line the statement's first non-blank character sits on.
    line: usize,
    /// The statement with newlines and comments flattened to spaces.
    text: String,
}

/// Advance `index` past a `//` line comment or a `"…"` literal starting there,
/// returning the index just after it, or `None` if neither starts at `index`.
fn skip_comment_or_string(chars: &[char], index: usize) -> Option<usize> {
    match (chars[index], chars.get(index + 1)) {
        ('/', Some('/')) => {
            let mut end = index;
            while end < chars.len() && chars[end] != '\n' {
                end += 1;
            }
            Some(end)
        }
        ('"', _) => {
            let mut end = index + 1;
            while end < chars.len() {
                match chars[end] {
                    '\\' => end += 2,
                    '"' => return Some(end + 1),
                    _ => end += 1,
                }
            }
            Some(chars.len())
        }
        _ => None,
    }
}

/// Every statement inside every `dynasm!` body in `text`, plus the number of
/// bodies walked.
///
/// A dynasm statement is `;`-separated and freely wraps across lines, so a
/// per-line split would let `; mov rax, QWORD` / `->label` through on the
/// continuation line. Splitting only at the body's own nesting depth keeps a
/// `;` inside an interpolated Rust expression from cutting a statement in two.
///
/// The body count is returned so a caller can tell "no statement uses an
/// address-dependent form" apart from "the scan found no assembly at all".
fn dynasm_statements(text: &str) -> (Vec<Statement>, usize) {
    let chars: Vec<char> = text.chars().collect();
    let needle: Vec<char> = "dynasm!".chars().collect();
    let newlines: Vec<usize> = chars
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c == '\n')
        .map(|(index, _)| index)
        .collect();
    let line_of = |index: usize| newlines.partition_point(|&nl| nl < index) + 1;

    let mut statements = Vec::new();
    let mut bodies = 0;
    let mut index = 0;
    while index < chars.len() {
        if let Some(next) = skip_comment_or_string(&chars, index) {
            index = next;
            continue;
        }
        if !chars[index..].starts_with(&needle[..]) {
            index += 1;
            continue;
        }
        let mut open = index + needle.len();
        while chars.get(open).is_some_and(|c| c.is_whitespace()) {
            open += 1;
        }
        if chars.get(open) != Some(&'(') {
            index += needle.len();
            continue;
        }
        bodies += 1;

        // `open` is depth 1, so statement separators are the `;` at depth 1.
        let mut depth = 1usize;
        let mut current = String::new();
        let mut start = open + 1;
        let mut cursor = open + 1;
        let mut flush = |current: &mut String, start: usize| {
            let piece = current.trim();
            if !piece.is_empty() {
                // Report the line the statement's first real character is on,
                // not the separator's, so a wrapped statement points at its head.
                let head = (start..chars.len())
                    .find(|&i| !chars[i].is_whitespace())
                    .unwrap_or(start);
                statements.push(Statement {
                    line: line_of(head),
                    text: piece.to_string(),
                });
            }
            current.clear();
        };
        while cursor < chars.len() && depth > 0 {
            if let Some(next) = skip_comment_or_string(&chars, cursor) {
                // A comment is not part of the statement; a string literal is
                // kept so an interpolated operand still reads as one piece.
                if chars[cursor] == '"' {
                    current.extend(&chars[cursor..next]);
                } else {
                    current.push(' ');
                }
                cursor = next;
                continue;
            }
            match chars[cursor] {
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(chars[cursor]);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    current.push(chars[cursor]);
                }
                ';' if depth == 1 => {
                    flush(&mut current, start);
                    start = cursor + 1;
                }
                '\n' => current.push(' '),
                c => current.push(c),
            }
            cursor += 1;
        }
        flush(&mut current, start);
        index = cursor + 1;
    }
    (statements, bodies)
}

/// Why `statement` would emit a relocation that a buffer move invalidates.
fn address_dependent_reason(statement: &str) -> Option<&'static str> {
    if statement
        .split_whitespace()
        .any(|token| token.trim_matches(',') == "extern")
    {
        return Some("`extern` jump target emits a RelToAbs relocation");
    }
    let mentions_label = statement.contains("->") || statement.contains("=>");
    if mentions_label
        && ABSOLUTE_ENCODINGS
            .iter()
            .any(|encoding| statement.contains(encoding))
    {
        return Some("absolutely encoded label operand emits an AbsToRel relocation");
    }
    None
}

fn scan(text: &str) -> Vec<(usize, &'static str)> {
    let (statements, _) = dynasm_statements(text);
    statements
        .into_iter()
        .filter_map(|statement| {
            address_dependent_reason(&statement.text).map(|reason| (statement.line, reason))
        })
        .collect()
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

    let rel_to_abs = "dynasm!(self.mc ; call extern helper as _);\n";
    assert_eq!(
        scan(rel_to_abs).len(),
        1,
        "an `extern` jump target must be caught"
    );

    let abs_to_rel = "dynasm!(self.mc ; mov rax, QWORD =>resume_label);\n";
    assert_eq!(
        scan(abs_to_rel).len(),
        1,
        "an 8-byte label operand must be caught"
    );

    // A statement that wraps: the continuation line carries no `;` of its own,
    // so a per-line scan would miss both of these, and the reported line is the
    // statement's head rather than its tail.
    let wrapped_abs_to_rel =
        "dynasm!(self.mc\n    ; mov rax, QWORD\n          ->resume_label\n);\n";
    assert_eq!(
        scan(wrapped_abs_to_rel),
        vec![(
            2,
            "absolutely encoded label operand emits an AbsToRel relocation"
        )],
        "a wrapped 8-byte label operand must be caught on its opening line"
    );
    let wrapped_extern = "dynasm!(self.mc\n    ; call\n          extern helper as _\n);\n";
    assert_eq!(
        scan(wrapped_extern),
        vec![(2, "`extern` jump target emits a RelToAbs relocation")],
        "a wrapped `extern` jump target must be caught on its opening line"
    );

    // Both forms are assembly syntax. Outside a `dynasm!` body the same spellings
    // are ordinary Rust and mean something else entirely — these are the shapes
    // that a whole-file scan reports as hits.
    let host_extern = "unsafe extern \"C\" {\n    fn sys_icache_invalidate(p: *mut u8);\n}\n";
    assert_eq!(
        scan(host_extern),
        Vec::new(),
        "a Rust `extern` block is not a jump target"
    );
    let host_match_arm = "match placement {\n    Gpr(d) => emit(d, QWORD_MASK),\n}\n";
    assert_eq!(
        scan(host_match_arm),
        Vec::new(),
        "a `match` arm next to a QWORD operand is not a label reference"
    );

    // A nested `;` belongs to the interpolated Rust expression, not to dynasm,
    // so it must not cut the statement it sits inside.
    let nested_semicolon = "dynasm!(self.mc ; mov rax, QWORD { let x = 1; x } ->label);\n";
    assert_eq!(
        scan(nested_semicolon).len(),
        1,
        "a `;` inside an interpolated expression must not split the statement"
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
    let mut bodies = 0;
    let mut statements = 0;
    for path in &sources {
        let text = fs::read_to_string(path).expect("source file is utf-8");
        let (found, file_bodies) = dynasm_statements(&text);
        bodies += file_bodies;
        statements += found.len();
        for statement in found {
            if let Some(reason) = address_dependent_reason(&statement.text) {
                violations.push(format!("{}:{}: {reason}", path.display(), statement.line));
            }
        }
    }

    // An empty result means nothing only if there was assembly to read.
    assert!(
        bodies > 1000 && statements > bodies,
        "scan read {statements} statements from {bodies} `dynasm!` bodies, \
         far below this backend's emitter — the parse, not the code, is what is empty"
    );

    assert!(
        violations.is_empty(),
        "address-dependent relocations would survive the copy in \
         `codebuf::finalize_writable`, which moves finalized code to an arena \
         address dynasm never sees:\n{}",
        violations.join("\n")
    );
}
