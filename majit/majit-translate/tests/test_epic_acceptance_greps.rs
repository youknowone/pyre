//! Source-level acceptance gates for structural RPython parity.
//!
//! The tests reject variant-keyed JitCode maps, a separate exception opname
//! family, and the legacy `compile_pyre_interpreter` entry point. They scan
//! every MAJIT crate's `src` tree after stripping comments and normalizing
//! whitespace, with positive controls that prove each selector still matches.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use walkdir::WalkDir;

// Forbidden selectors

/// Spellings that would introduce a variant-keyed JitCode map. Source is
/// whitespace-normalized before matching, so ordinary and turbofish forms of
/// both hash and ordered maps are listed explicitly.
const VARIANT_KEYED_MAP: &[&str] = &[
    "HashMap<Instruction",
    "HashMap::<Instruction",
    "BTreeMap<Instruction",
    "BTreeMap::<Instruction",
];

const NEW_OPNAME_FAMILY: &[&str] = &["_may_raise", "OpKind::Try", "TryOp"];

/// `fn compile_pyre_interpreter` with its whitespace removed.
const LEGACY_ENTRY_POINT: &[&str] = &["fncompile_pyre_interpreter"];

/// Below this the walk is assumed broken rather than the tree assumed empty.
/// A BRAKE, not a census pin — it is far below the population so that adding
/// or deleting sources never reds this gate for an unrelated reason.
const MIN_POPULATION: usize = 100;

/// Likewise a brake: `majit/` has 14 crate `src` trees today.
const MIN_CRATES: usize = 5;

// Comment stripping
/// Overwrite `[from, to)` with spaces, leaving newlines in place so byte
/// offsets and line numbers both survive.
fn blank(out: &mut [u8], from: usize, to: usize) {
    for byte in &mut out[from..to] {
        if *byte != b'\n' {
            *byte = b' ';
        }
    }
}

/// If a raw-string literal opens at `i`, return `(first body byte, hash count)`.
fn raw_string_open(b: &[u8], i: usize) -> Option<(usize, usize)> {
    // A raw string may not be the tail of an identifier (`foo_r"x"` is not one).
    if i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_') {
        return None;
    }
    let mut p = i;
    if b.get(p) == Some(&b'b') {
        p += 1;
    }
    if b.get(p) != Some(&b'r') {
        return None;
    }
    p += 1;
    let mut hashes = 0;
    while b.get(p) == Some(&b'#') {
        hashes += 1;
        p += 1;
    }
    if b.get(p) != Some(&b'"') {
        return None;
    }
    Some((p + 1, hashes))
}

fn raw_string_end(b: &[u8], body_start: usize, hashes: usize) -> usize {
    let mut p = body_start;
    while p < b.len() {
        if b[p] == b'"' && (0..hashes).all(|k| b.get(p + 1 + k) == Some(&b'#')) {
            return p + 1 + hashes;
        }
        p += 1;
    }
    b.len()
}

fn string_end(b: &[u8], i: usize) -> usize {
    let mut p = i + 1;
    while p < b.len() {
        match b[p] {
            b'\\' => p += 2,
            b'"' => return p + 1,
            _ => p += 1,
        }
    }
    b.len()
}

/// `'a'` / `'\n'` are char literals; `'a` in `&'a T` is a lifetime and must not
/// swallow what follows. The two are told apart by where the closing quote sits.
fn char_literal_end(b: &[u8], i: usize) -> usize {
    match b.get(i + 1) {
        Some(&b'\\') => {
            let limit = (i + 8).min(b.len());
            (i + 2..limit)
                .find(|&p| b[p] == b'\'')
                .map_or(i + 1, |p| p + 1)
        }
        Some(&c) if c.is_ascii() && b.get(i + 2) == Some(&b'\'') => i + 3,
        // A multi-byte char: the quote follows its continuation bytes.
        Some(&c) if !c.is_ascii() => {
            let limit = (i + 6).min(b.len());
            (i + 2..limit)
                .find(|&p| b[p] == b'\'')
                .map_or(i + 1, |p| p + 1)
        }
        _ => i + 1,
    }
}

/// Rust block comments nest, so this counts depth rather than scanning for the
/// first `*/`.
fn block_comment_end(b: &[u8], i: usize) -> usize {
    let mut depth = 1usize;
    let mut p = i + 2;
    while p < b.len() {
        if b[p] == b'/' && b.get(p + 1) == Some(&b'*') {
            depth += 1;
            p += 2;
        } else if b[p] == b'*' && b.get(p + 1) == Some(&b'/') {
            depth -= 1;
            p += 2;
            if depth == 0 {
                return p;
            }
        } else {
            p += 1;
        }
    }
    b.len()
}

/// Blank every comment, preserving length, byte offsets and line breaks.
///
/// String contents are KEPT. A string is not a comment, and keeping it is the
/// over-reporting direction — the safe one for a gate. It is also the whole
/// point: the previous selector cut each line at its first `//`, so a line
/// carrying `"scheme://host"` lost everything after it.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = b.to_vec();
    let mut i = 0usize;
    while i < b.len() {
        if let Some((body, hashes)) = raw_string_open(b, i) {
            i = raw_string_end(b, body, hashes);
            continue;
        }
        match b[i] {
            b'"' => i = string_end(b, i),
            b'\'' => i = char_literal_end(b, i),
            b'/' if b.get(i + 1) == Some(&b'/') => {
                let end = b[i..]
                    .iter()
                    .position(|&c| c == b'\n')
                    .map_or(b.len(), |off| i + off);
                blank(&mut out, i, end);
                i = end;
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                let end = block_comment_end(b, i);
                blank(&mut out, i, end);
                i = end;
            }
            _ => i += 1,
        }
    }
    String::from_utf8(out).expect("comment bytes are replaced with ASCII spaces")
}

/// Drop every whitespace byte, recording the source line each surviving byte
/// came from. Dropping newlines too is what makes a generic wrapped across
/// lines matchable; the forbidden spellings contain no whitespace, so nothing
/// about them is lost.
fn normalise(src: &str) -> (String, Vec<u32>) {
    let mut dense = String::with_capacity(src.len());
    let mut lines = Vec::with_capacity(src.len());
    let mut line = 1u32;
    for ch in src.chars() {
        if ch == '\n' {
            line += 1;
            continue;
        }
        if ch.is_whitespace() {
            continue;
        }
        dense.push(ch);
        // One entry per UTF-8 byte: `str::find` returns byte offsets, so the
        // line map has to be indexed the same way.
        for _ in 0..ch.len_utf8() {
            lines.push(line);
        }
    }
    (dense, lines)
}

// Source population

struct Scanned {
    rel: PathBuf,
    dense: String,
    lines: Vec<u32>,
}

/// `.../majit`, the parent of every crate directory.
fn majit_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate directory has a parent")
        .to_path_buf()
}

/// True for `<crate>/src/**.rs` and nothing else — in particular not for
/// `<crate>/tests/**`, which is what stops this gate reading its own body.
fn is_crate_source(rel: &Path) -> bool {
    let mut parts = rel.components();
    let _crate_dir = parts.next();
    parts.next().is_some_and(|c| c.as_os_str() == "src")
        && rel.extension().and_then(|s| s.to_str()) == Some("rs")
}

fn population() -> &'static Vec<Scanned> {
    static FILES: OnceLock<Vec<Scanned>> = OnceLock::new();
    FILES.get_or_init(|| {
        let root = majit_root();
        WalkDir::new(&root)
            .into_iter()
            // Descend only `majit/<crate>/src/**`. Pruning at depth 2 is not
            // cosmetic: `majit/charon-corpus/target` exists on a built tree,
            // and an unpruned walk would descend the whole build directory.
            .filter_entry(|e| e.depth() != 2 || e.file_name() == OsStr::new("src"))
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let path = entry.into_path();
                let rel = path.strip_prefix(&root).ok()?.to_path_buf();
                if !is_crate_source(&rel) {
                    return None;
                }
                let src = fs::read_to_string(&path).ok()?;
                let (dense, lines) = normalise(&strip_comments(&src));
                Some(Scanned { rel, dense, lines })
            })
            .collect()
    })
}

fn scan_forbidden(label: &str, patterns: &[&str]) -> Vec<String> {
    let files = population();
    let crates: std::collections::BTreeSet<_> = files
        .iter()
        .filter_map(|f| f.rel.components().next())
        .collect();

    // The denominator, printed on every run: a zero below is only readable
    // beside the size of the population that produced it.
    eprintln!(
        "[{label}] scanned {} files across {} crates under majit/<crate>/src",
        files.len(),
        crates.len()
    );
    assert!(
        files.len() >= MIN_POPULATION && crates.len() >= MIN_CRATES,
        "acceptance-grep population collapsed to {} files / {} crates — the \
         walk is broken, so a green here would mean nothing. Expected at \
         least {MIN_POPULATION} files across {MIN_CRATES} crates.",
        files.len(),
        crates.len()
    );

    let mut hits = Vec::new();
    for file in files {
        for &pattern in patterns {
            let mut from = 0;
            while let Some(at) = file.dense[from..].find(pattern) {
                let at = from + at;
                hits.push(format!(
                    "{}:{}: {}",
                    file.rel.display(),
                    file.lines[at],
                    pattern
                ));
                from = at + 1;
            }
        }
    }
    if !hits.is_empty() {
        eprintln!("[{label}] {} forbidden occurrence(s):", hits.len());
        for hit in &hits {
            eprintln!("  {hit}");
        }
    }
    hits
}

// Acceptance gates

/// Anchor: `rpython/jit/codewriter/call.py:87` keeps `self.jitcodes = {}`
/// keyed by graph identity; there is no per-opcode-variant JitCode map
/// anywhere in upstream. pyre's canonical key is `CallPath` (the graph
/// identity surrogate), matching upstream.
#[test]
fn no_variant_keyed_jitcode_map() {
    let hits = scan_forbidden("no_variant_keyed_jitcode_map", VARIANT_KEYED_MAP);
    assert!(
        hits.is_empty(),
        "Plan acceptance violated: a variant-keyed Instruction → ... map \
         reintroduces the CPython-3.13-bytecode coupling that was removed. \
         Use `HashMap<CallPath, ...>` instead — see \
         `rpython/jit/codewriter/call.py:87 self.jitcodes = {{}}`."
    );
}

/// Anchor: `rpython/jit/codewriter/jtransform.py:456 rewrite_op_direct_call`
/// handles raising calls through the existing `residual_call_{r,v,f}_*`
/// family; `rpython/translator/exceptiontransform.py` lowers exception edges
/// to additional successors on the existing `Terminator`. No new opname
/// family, no new `OpKind` variant.
#[test]
fn no_new_opname_family_for_exceptions() {
    let hits = scan_forbidden("no_new_opname_family_for_exceptions", NEW_OPNAME_FAMILY);
    assert!(
        hits.is_empty(),
        "Plan acceptance violated: exception-carrying operations must be \
         routed through the existing `residual_call_*` opname family \
         (`rpython/jit/codewriter/jtransform.py:rewrite_op_direct_call`) \
         and `?`/`PyResult` must lower to exceptional successor edges on \
         the existing `Terminator` (`rpython/translator/exceptiontransform.py`). \
         A fresh `_may_raise` / `OpKind::Try` / `TryOp` is a deviation \
         and must be removed."
    );
}

/// The legacy `compile_pyre_interpreter` entry point must not re-emerge. It
/// is replaced with `CodeWriter::make_jitcodes` (upstream `codewriter.py:74`).
/// A surviving definition means the pyre-bytecode-walking path lives in
/// parallel with the graph-keyed one.
#[test]
fn no_legacy_compile_pyre_interpreter() {
    let hits = scan_forbidden("no_legacy_compile_pyre_interpreter", LEGACY_ENTRY_POINT);
    assert!(
        hits.is_empty(),
        "Plan acceptance violated: `compile_pyre_interpreter` resurfaced. \
         It is replaced with `CodeWriter::make_jitcodes` \
         (`rpython/jit/codewriter/codewriter.py:74`). Remove the legacy \
         definition."
    );
}

/// Positive control for the three absence gates. It runs their real
/// strip/normalize/find pipeline over awkward spellings and confirms that code
/// is found while commented mentions are ignored.
#[test]
fn selector_detects_every_forbidden_spelling() {
    const SYNTHETIC: &str = r###"
// a comment naming HashMap<Instruction must not count
//! nor a doc line naming OpKind::Try or TryOp
/* a block /* nested */ comment naming HashMap<Instruction */
fn truncated_by_a_string() {
    let _u = "scheme://host";
    type M1 = HashMap<Instruction, u32>;
}
type M2 = HashMap<
    Instruction,
    u32,
>;
fn turbofish() { let _ = HashMap::<Instruction, u32>::new(); }
fn ordered() -> BTreeMap<Instruction, u32> { todo!() }
fn raising(x: u8) -> bool { x._may_raise() }
struct S { t: TryOp, k: OpKind::Try }
fn compile_pyre_interpreter() {}
"###;

    let (dense, _lines) = normalise(&strip_comments(SYNTHETIC));

    // (pattern, expected count) — every count is deliberate, not a floor.
    let expected: &[(&str, usize)] = &[
        ("HashMap<Instruction", 2), // the string-truncated line + the wrapped generic
        ("HashMap::<Instruction", 1),
        ("BTreeMap<Instruction", 1),
        ("_may_raise", 1),
        ("OpKind::Try", 1),
        ("TryOp", 1),
        ("fncompile_pyre_interpreter", 1),
    ];
    for &(pattern, want) in expected {
        let got = dense.matches(pattern).count();
        assert_eq!(
            got, want,
            "selector self-test: `{pattern}` found {got} time(s), expected \
             {want}. The selector has stopped seeing a spelling it is \
             responsible for, which would make the three gates above pass \
             vacuously."
        );
    }

    // The negative half: nothing before the first `fn` is code, so the three
    // commented mentions must contribute nothing. Without this the test could
    // be satisfied by a selector that never strips comments at all.
    let stripped = strip_comments(SYNTHETIC);
    let commented_prefix = &stripped[..stripped.find("fn truncated").expect("marker present")];
    for leaked in ["HashMap<Instruction", "OpKind::Try", "TryOp"] {
        assert!(
            !commented_prefix.contains(leaked),
            "selector self-test: `{leaked}` survived comment stripping — \
             line, doc and nested block comments must all be blanked."
        );
    }
}
