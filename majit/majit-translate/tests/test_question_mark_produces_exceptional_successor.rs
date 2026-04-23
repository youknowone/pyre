//! Phase C acceptance anchor (lucky-growing-puzzle).
//!
//! Test name matches the plan's stated acceptance command
//! (`cargo test -p majit-translate test_question_mark_produces_exceptional_successor`).
//!
//! ## RPython line-by-line references
//!
//! - `rpython/flowspace/flowcontext.py:130-143` — exception link carries
//!   fresh prevblock-side `Variable('last_exception')` +
//!   `Variable('last_exc_value')` plus `case is Exception`.
//! - `rpython/flowspace/model.py:21-25` — `FunctionGraph.exceptblock`
//!   has two inputargs `(etype, evalue)`.
//! - `rpython/flowspace/model.py:214` — `Block.canraise` is true iff
//!   `exitswitch is c_last_exception`.
//! - `rpython/jit/codewriter/jtransform.py:456-471` —
//!   `handle_residual_call` emits `residual_call_*` followed by a
//!   trailing `-live-` whenever `calldescr_canraise(calldescr)`.
//! - `rpython/jit/codewriter/flatten.py:177-278 insert_exits()` — a
//!   canraise block flattens to `catch_exception` + normal link +
//!   `goto_if_exception_mismatch` / reraise tail.
//!
//! ## What this test anchors
//!
//! pyre already lowers Rust `?` to a two-exit Link structure + fresh
//! `last_exception` / `last_exc_value` extravars — same shape RPython
//! produces from its `flowspace/flowcontext.py:130-143`. Phase A's
//! discovery matrix counts canraise blocks per handler; this test
//! asserts the **structural** invariants explicitly on a minimal
//! handler so any regression surfaces with a clear message (not
//! buried inside a 28-row matrix).
//!
//! No new `OpKind`, no new opname, no new JitCode key.

use majit_translate::front::ast::build_function_graph_pub;
use majit_translate::model::ExitSwitch;
use syn::{File, Item};

const PYOPCODE_SRC: &str = include_str!("../../../pyre/pyre-interpreter/src/pyopcode.rs");

fn parse_pyopcode() -> File {
    syn::parse_file(PYOPCODE_SRC).expect("pyopcode.rs must parse")
}

fn find_fn<'a>(file: &'a File, name: &str) -> &'a syn::ItemFn {
    file.items
        .iter()
        .find_map(|item| match item {
            Item::Fn(func) if func.sig.ident == name => Some(func),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected `fn {name}` in pyopcode.rs"))
}

/// RPython `flowspace/model.py:214` — `block.canraise` is true iff
/// `block.exitswitch is c_last_exception`. Pyre's `Block::canraise()`
/// already mirrors this semantic; the helper here makes the assertion
/// read against the raw ExitSwitch so the message can cite model.py.
fn block_is_canraise(block: &majit_translate::model::Block) -> bool {
    matches!(block.exitswitch, Some(ExitSwitch::LastException))
}

#[test]
fn test_question_mark_produces_exceptional_successor() {
    // Pick a small handler that contains at least one `?` call:
    // `opcode_load_const` body is `executor.load_const(&code.constants[idx])?`
    // followed by the `Ok(StepResult::Continue)` fall-through. This is
    // the minimal shape for the parity assertion.
    let file = parse_pyopcode();
    let func = find_fn(&file, "opcode_load_const");

    let sf = build_function_graph_pub(func);

    // RPython `flowspace/model.py:214` — at least one block must have
    // `exitswitch == c_last_exception` once the lowering has emitted
    // the `?`-edge. If this is 0, front/ast.rs `Expr::Try` silently
    // dropped the exception link.
    let canraise_blocks: Vec<_> = sf
        .graph
        .blocks
        .iter()
        .filter(|b| block_is_canraise(b))
        .collect();
    assert!(
        !canraise_blocks.is_empty(),
        "`opcode_load_const` contains a `?` call but no canraise block \
         was produced — front/ast.rs Expr::Try lowering regressed"
    );

    // RPython `flowcontext.py:141-143`: canraise block has **two**
    // Links — `[0]` is the normal fall-through (exitcase == None),
    // `[1]` is the exception link (case is Exception, with extravars
    // last_exception / last_exc_value).
    for block in &canraise_blocks {
        assert_eq!(
            block.exits.len(),
            2,
            "canraise block must have exactly 2 exits (normal + \
             exception), mirrors flowcontext.py:141-143"
        );
        assert!(
            block.exits[0].exitcase.is_none(),
            "exits[0] must be the normal fall-through (exitcase=None); \
             flowcontext.py:141 `Link(vars=[], egg, case=None)`"
        );
        assert!(
            block.exits[1].catches_all_exceptions(),
            "exits[1] must have `case is Exception` — the upstream \
             shape from flowcontext.py:141-143"
        );
        assert!(
            block.exits[1].last_exception.is_some() && block.exits[1].last_exc_value.is_some(),
            "exits[1] must carry extravars last_exception / \
             last_exc_value populated by flowcontext.py:143 \
             `link.extravars(...)`"
        );
    }

    // RPython `flowspace/model.py:21-25` — `FunctionGraph.exceptblock`
    // always carries **two** inputargs `(etype, evalue)`. The
    // `?`-exception Link feeds into those two slots; a mismatched
    // arity would make the per-handler jitcode's exception link
    // un-emittable by flatten/liveness.
    assert_eq!(
        sf.graph.block(sf.graph.exceptblock).inputargs.len(),
        2,
        "exceptblock arity must stay (etype, evalue) per \
         flowspace/model.py:21-25"
    );
}

#[test]
fn no_new_opname_introduced_for_may_raise_paths() {
    // Plan acceptance criterion: `rg "_may_raise" majit/majit-translate/src/ = 0`.
    // RPython `jtransform.py:467-470` folds raising calls into the
    // existing `residual_call_%s_%s` family (residual_call_r_r,
    // residual_call_ir_v, ...) — no `residual_call_may_raise` opname.
    // Pyre must stay on that family; the canraise signal rides on the
    // trailing `-live-` (OpKind::Live) emitted by jtransform.rs:1741-1746.
    //
    // This test is a low-cost regression fence: if anyone ever adds a
    // `_may_raise`-suffixed opname, the grep floor trips before it
    // reaches master.
    // Plain substring scan — `_may_raise` is distinctive enough that a
    // full regex engine is not needed. We intentionally skip doc
    // comments + test files so the acceptance floor matches the
    // production emit surface only.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders: Vec<String> = Vec::new();
    walk_rs_files(&root, &mut |path, contents| {
        for (i, line) in contents.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with("*") {
                continue;
            }
            if line.contains("_may_raise") {
                offenders.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
            }
        }
    });
    assert!(
        offenders.is_empty(),
        "no `_may_raise` opnames are permitted — RPython jtransform.py:467-470 \
         keeps the raising path on the existing residual_call_* family:\n{}",
        offenders.join("\n")
    );
}

fn walk_rs_files<F: FnMut(&std::path::Path, &str)>(dir: &std::path::Path, f: &mut F) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, f);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                f(&path, &contents);
            }
        }
    }
}
