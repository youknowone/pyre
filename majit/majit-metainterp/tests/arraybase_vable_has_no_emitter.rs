//! `BC_ARRAYBASE_VABLE` is defined, decoded, walked, wired into the blackhole
//! and assemblable — and **nothing emits it**. This test is what keeps that
//! sentence true.
//!
//! The opcode exists ahead of its only intended producer: a `#[jit_interp]`
//! lowering for `state.<varr>.as_mut_ptr()`, held back because the walk
//! resumes at the residual's own argument byte rather than at the next opcode
//! (see the `BC_ARRAYBASE_VABLE` comment in `majit-translate`'s `insns.rs`).
//! Until that is fixed, an emitter would turn a dormant opcode into a wrong
//! answer.
//!
//! ## Why a test and not a comment
//!
//! The opcode already carries a comment saying nothing emits it. Prose
//! describing an **absence** is uniquely fragile: it stays on the page,
//! reading as current, at the exact moment someone adds the emitter that makes
//! it false. Nothing about writing that emitter brings the reader past the
//! sentence. A test converts "this claim quietly went stale" into a red.
//!
//! ## Why the emitter and not the byte
//!
//! `vable_arraybase_with_base` is the single choke point — the only way the
//! opcode's byte reaches a jitcode. Scanning for `BC_ARRAYBASE_VABLE` instead
//! would match its definition, its name-map registration and its handler
//! wiring, all of which are supposed to exist, so the interesting signal would
//! have to be separated from three legitimate ones.
//!
//! The emitter is located by **symbol, not by path**: the file it lives in has
//! moved once already and may again.

use std::path::{Path, PathBuf};

/// The only way to put `BC_ARRAYBASE_VABLE` into a jitcode.
const EMITTER: &str = "vable_arraybase_with_base";

/// Crates scanned. The Rust workspace lives under these two roots; everything
/// else at the repository root is Python, fixtures, or untracked scratch.
const ROOTS: [&str; 2] = ["majit", "pyre"];

fn repo_root() -> PathBuf {
    // `majit/majit-metainterp` -> `majit` -> repository root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("manifest dir has two ancestors")
        .to_path_buf()
}

fn collect_rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            // `target` holds generated copies of the sources being scanned;
            // counting them would report the same call site many times over.
            if name == "target" || name.starts_with('.') {
                continue;
            }
            collect_rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Every reference to [`EMITTER`], split into its definition and its callers.
///
/// Comment lines are dropped first: a doc comment naming the emitter (this
/// file's own header does it, and the opcode's documentation may come to)
/// is prose about the symbol, not a use of it.
fn scan() -> (Vec<String>, Vec<String>) {
    let root = repo_root();
    let mut files = Vec::new();
    for r in ROOTS {
        collect_rust_sources(&root.join(r), &mut files);
    }
    assert!(
        files.len() > 100,
        "source walk found only {} files under {ROOTS:?} — the scan is \
         mis-rooted at {}, so a zero result would mean nothing",
        files.len(),
        root.display(),
    );

    let mut definitions = Vec::new();
    let mut callers = Vec::new();
    for file in files {
        // This file names the emitter in prose and in `EMITTER`; excluding it
        // by path keeps the const from counting as a call site.
        if file.ends_with("arraybase_vable_has_no_emitter.rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        if !text.contains(EMITTER) {
            continue;
        }
        for (i, line) in text.lines().enumerate() {
            if !line.contains(EMITTER) || line.trim_start().starts_with("//") {
                continue;
            }
            let site = format!(
                "{}:{}: {}",
                file.strip_prefix(&root).unwrap_or(&file).display(),
                i + 1,
                line.trim(),
            );
            if line.contains(&format!("fn {EMITTER}")) {
                definitions.push(site);
            } else {
                callers.push(site);
            }
        }
    }
    (definitions, callers)
}

#[test]
fn arraybase_vable_has_no_emitter() {
    let (definitions, callers) = scan();

    // A missing definition fails as loudly as an extra caller: it means the
    // opcode's producer was deleted or renamed, and this test would otherwise
    // keep passing while guarding nothing.
    assert_eq!(
        definitions.len(),
        1,
        "expected exactly one `{EMITTER}` definition, found {}:\n  {}",
        definitions.len(),
        definitions.join("\n  "),
    );

    assert!(
        callers.is_empty(),
        "`{EMITTER}` now has {} caller(s):\n  {}\n\n\
         `BC_ARRAYBASE_VABLE` is deliberately unreachable — it is defined, \
         decoded and wired, but no lowering emits it, because the walk resumes \
         at the residual's argument byte instead of the next opcode and the \
         program returns a wrong answer rather than aborting.\n\n\
         If you are landing the lowering as part of fixing that: delete this \
         test in the same commit, and say in the message that the opcode now \
         has a producer. If you are not, this is the bug this test exists to \
         catch.",
        callers.len(),
        callers.join("\n  "),
    );
}
