//! Epic acceptance invariants — the "what this epic must NOT introduce"
//! side of the parity contract in `docs/plans/lucky-growing-puzzle.md`.
//!
//! The plan's verification protocol lists three grep invariants the
//! codebase must uphold after the epic lands. Each invariant maps to a
//! specific RPython structural decision that pyre must NOT diverge from:
//!
//! | Invariant | RPython anchor | Why |
//! |---|---|---|
//! | `HashMap<Instruction, ...>` = 0 | `rpython/jit/codewriter/call.py:87 self.jitcodes = {}` | upstream keys jitcodes by `graph`, not by opcode variant. A variant-keyed map in pyre would mean the codewriter lowers per-opcode-arm instead of per-graph — that is the CPython-3.13-bytecode coupling the epic removes. |
//! | `_may_raise \| TryOp \| OpKind::Try` = 0 | `rpython/jit/codewriter/jtransform.py:456 rewrite_op_direct_call` + `rpython/translator/exceptiontransform.py` | upstream routes raising calls through the existing `residual_call_*` opname family and lowers `?`/`PyResult` to exceptional successor edges on the existing `Terminator`. A fresh opname family would be a NEW-DEVIATION. |
//! | `compile_pyre_interpreter` = 0 | (pyre-specific legacy name) | Phase E replaces the pyre-specific opcode-walking entry point with `CodeWriter::make_jitcodes`. A lingering `compile_pyre_interpreter` definition means the legacy path survives in parallel. |
//!
//! The tests walk `majit-translate/src/` and strip line comments before
//! matching; a mention inside `//` / `//!` is acceptable (typically this
//! very file's anchor comment referencing the forbidden pattern). Any
//! non-comment occurrence is a hard failure.

use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Walk production source tree. Returns (relative_path, source) pairs.
fn production_rs_files(root: &Path) -> Vec<(PathBuf, String)> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext == "rs")
                .unwrap_or(false)
        })
        .filter_map(|e| {
            let path = e.into_path();
            let rel = path.strip_prefix(root).ok()?.to_path_buf();
            let src = fs::read_to_string(&path).ok()?;
            Some((rel, src))
        })
        .collect()
}

/// Strip `//` line comments (including `//!` doc lines and `///`
/// rustdoc lines). Matches rustc lexer semantics for single-line comments.
/// Block comments are left in place — forbidden strings inside `/* */`
/// are rare enough that stripping them is not worth the parser.
fn strip_line_comments(line: &str) -> &str {
    match line.find("//") {
        Some(idx) => &line[..idx],
        None => line,
    }
}

fn scan_forbidden(
    label: &str,
    root: &Path,
    is_forbidden: impl Fn(&str) -> bool,
) -> Vec<(PathBuf, usize, String)> {
    let mut hits = Vec::new();
    for (rel, src) in production_rs_files(root) {
        for (lineno, line) in src.lines().enumerate() {
            let stripped = strip_line_comments(line);
            if is_forbidden(stripped) {
                hits.push((rel.clone(), lineno + 1, line.to_string()));
            }
        }
    }
    if !hits.is_empty() {
        eprintln!(
            "[{label}] {} forbidden occurrence(s) in majit-translate/src/:",
            hits.len()
        );
        for (rel, lineno, line) in &hits {
            eprintln!("  {}:{}: {}", rel.display(), lineno, line.trim());
        }
    }
    hits
}

fn src_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Plan verification protocol item #5: `rg "HashMap<Instruction" majit/` = 0.
///
/// Anchor: `rpython/jit/codewriter/call.py:87` keeps `self.jitcodes = {}`
/// keyed by graph identity; there is no per-opcode-variant JitCode map
/// anywhere in upstream. pyre's canonical key is `CallPath` (the graph
/// identity surrogate), matching upstream.
#[test]
fn no_variant_keyed_jitcode_map() {
    let hits = scan_forbidden("no_variant_keyed_jitcode_map", &src_root(), |line| {
        line.contains("HashMap<Instruction")
    });
    assert!(
        hits.is_empty(),
        "Plan acceptance violated: a variant-keyed Instruction → ... map \
         reintroduces the CPython-3.13-bytecode coupling that Phase E \
         removed. Use `HashMap<CallPath, ...>` instead — see \
         `rpython/jit/codewriter/call.py:87 self.jitcodes = {{}}`."
    );
}

/// Plan verification protocol item #4: `rg "_may_raise|TryOp|OpKind::Try"
/// majit/` = 0.
///
/// Anchor: `rpython/jit/codewriter/jtransform.py:456 rewrite_op_direct_call`
/// handles raising calls through the existing `residual_call_{r,v,f}_*`
/// family; `rpython/translator/exceptiontransform.py` lowers exception
/// edges to additional successors on the existing `Terminator`. No new
/// opname family, no new `OpKind` variant.
#[test]
fn no_new_opname_family_for_exceptions() {
    let hits = scan_forbidden("no_new_opname_family_for_exceptions", &src_root(), |line| {
        line.contains("_may_raise") || line.contains("OpKind::Try") || line.contains("TryOp")
    });
    assert!(
        hits.is_empty(),
        "Plan acceptance violated: exception-carrying operations must be \
         routed through the existing `residual_call_*` opname family \
         (`rpython/jit/codewriter/jtransform.py:rewrite_op_direct_call`) \
         and `?`/`PyResult` must lower to exceptional successor edges on \
         the existing `Terminator` (`rpython/translator/exceptiontransform.py`). \
         A fresh `_may_raise` / `OpKind::Try` / `TryOp` is NEW-DEVIATION \
         and must be removed."
    );
}

/// Plan verification protocol item: the legacy `compile_pyre_interpreter`
/// entry point must not re-emerge after Phase E. Phase E replaces it with
/// `CodeWriter::make_jitcodes` (upstream `codewriter.py:74`). A surviving
/// definition means the pyre-bytecode-walking path lives in parallel with
/// the graph-keyed one.
#[test]
fn no_legacy_compile_pyre_interpreter() {
    let hits = scan_forbidden("no_legacy_compile_pyre_interpreter", &src_root(), |line| {
        line.contains("fn compile_pyre_interpreter")
    });
    assert!(
        hits.is_empty(),
        "Plan acceptance violated: `compile_pyre_interpreter` resurfaced. \
         Phase E replaces it with `CodeWriter::make_jitcodes` \
         (`rpython/jit/codewriter/codewriter.py:74`). Remove the legacy \
         definition."
    );
}
