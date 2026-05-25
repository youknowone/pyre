//! Phase B.2: callcontrol trait-method resolution over real pyre-interpreter
//! sources.
//!
//! RPython's `bookkeeper.py:431 MethodDesc` keys method resolution on
//! `(classdef, method_name)` — i.e. the concrete receiver class must be
//! known.  pyre's `CallControl::resolve_method(name, Some(receiver_type))`
//! is the analogue: the caller supplies the concrete receiver and the
//! resolver finds the unique impl for that `(receiver, name)` pair.
//!
//! This test reads `pyre-interpreter/src/pyopcode.rs` (for trait
//! declarations) and `pyre-interpreter/src/eval.rs` (for
//! `impl <Trait> for PyFrame` blocks), builds a `CallControl`, and
//! asserts that the method names invoked by the super-instruction
//! helpers all resolve to a concrete PyFrame graph when the receiver
//! is the concrete type `PyFrame`.
//!
//! Generic-receiver resolution (e.g. `resolve_method(name, Some("E"))`)
//! is intentionally NOT tested here: upstream method resolution requires
//! a concrete classdef key (`bookkeeper.py:431`), and "unique impl
//! across the entire program" is a pyre-specific closed-world shortcut,
//! not parity.  That shortcut lives in `CallControl` for now, but test
//! oracles track upstream's contract — concrete receiver only.

use std::path::PathBuf;

use majit_translate::{
    ParsedInterpreter, build_semantic_program_from_parsed_files, call::CallControl,
    extract_opcode_dispatch_receiver_traits, extract_trait_impls, parse_source,
};

fn pyre_file_path(relative: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("..");
    p.push("..");
    for segment in relative.split('/') {
        p.push(segment);
    }
    p
}

fn parse_pyre_file(relative: &str) -> ParsedInterpreter {
    let path = pyre_file_path(relative);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));
    parse_source(&src)
}

/// Mirror `pyre-jit-trace/build.rs::collect_rs_files`: walk every `.rs`
/// under `pyre-object/src` and `pyre-interpreter/src` so
/// `build_semantic_program_from_parsed_files` sees the same
/// whole-program scope as production.
fn collect_pyre_interpreter_program_inputs() -> Vec<ParsedInterpreter> {
    let mut out = Vec::new();
    for dir in ["pyre/pyre-object/src", "pyre/pyre-interpreter/src"] {
        let root = pyre_file_path(dir);
        collect_rs_under(&root, &mut out);
    }
    out
}

fn collect_rs_under(dir: &std::path::Path, out: &mut Vec<ParsedInterpreter>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_under(&path, out);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(&path) else {
            continue;
        };
        out.push(parse_source(&src));
    }
}

#[test]
fn resolve_super_inst_method_calls_against_pyframe_impls() {
    let pyopcode = parse_pyre_file("pyre/pyre-interpreter/src/pyopcode.rs");
    let eval = parse_pyre_file("pyre/pyre-interpreter/src/eval.rs");

    // Step 1: confirm the dispatch function's generic receiver carries every
    // handler trait. Phase B.1 already established the where-clause parsing;
    // this reaffirms the pyre-interpreter signature hasn't drifted out from
    // under us.
    let bindings = extract_opcode_dispatch_receiver_traits(&pyopcode);
    let executor_traits = bindings
        .traits_by_receiver
        .get("executor")
        .expect("executor receiver binding");
    for expected in [
        "LocalOpcodeHandler",
        "SharedOpcodeHandler",
        "ArithmeticOpcodeHandler",
    ] {
        assert!(
            executor_traits.iter().any(|t| t == expected),
            "executor missing expected trait `{}`, got {:?}",
            expected,
            executor_traits
        );
    }

    // Step 2: collect trait impls from both files.  Production at
    // `lib.rs:317-342` populates `program.fn_return_types` /
    // `struct_fields` / `known_struct_names` via
    // `build_semantic_program_from_parsed_files` BEFORE
    // `extract_trait_impls` so the impl-body lowering can resolve
    // user-defined method-return types
    // (`bookkeeper.getdesc(...).find_method` upstream parity at
    // `unaryop.py:206-213`).  Without it,
    // `!self.<user_method>()` patterns surface
    // `UnaryNotUnknownOperand` at `front/ast.rs:3713`.  Walk the full
    // pyre-interpreter + pyre-object source set so the walker has the
    // same whole-program scope as `pyre-jit-trace/build.rs`.
    let parsed_files = collect_pyre_interpreter_program_inputs();
    let program = build_semantic_program_from_parsed_files(&parsed_files)
        .expect("pyre-interpreter source must lower without FlowingError");
    let mut impls = Vec::new();
    impls.extend(
        extract_trait_impls(
            &pyopcode,
            &program.struct_fields,
            &program.fn_return_types,
            &program.known_struct_names,
            None,
        )
        .expect("pyopcode trait impls must lower"),
    );
    impls.extend(
        extract_trait_impls(
            &eval,
            &program.struct_fields,
            &program.fn_return_types,
            &program.known_struct_names,
            None,
        )
        .expect("eval trait impls must lower"),
    );

    let pyframe_impl_count = impls.iter().filter(|i| i.for_type == "PyFrame").count();
    assert!(
        pyframe_impl_count >= 9,
        "expected at least 9 `impl ... for PyFrame` blocks across \
         pyopcode.rs + eval.rs, found {}",
        pyframe_impl_count
    );

    // Step 3: feed the impls to a fresh CallControl.
    let mut cc = CallControl::new();
    for imp in &impls {
        for method in &imp.methods {
            if let Some(graph) = method.graph.clone() {
                cc.register_trait_method(&method.name, Some(&imp.trait_name), &imp.for_type, graph);
            }
        }
    }

    // Step 4: every method name invoked by the super-instruction helpers
    // must resolve against the concrete `PyFrame` receiver.  This mirrors
    // RPython `bookkeeper.py:431 MethodDesc` which keys on
    // `(classdef, method_name)` — concrete class required.
    //
    // The list below mirrors the trait methods called from the bodies of
    // `opcode_load_fast_load_fast`, `opcode_store_fast_load_fast`,
    // `opcode_store_fast_store_fast`, and `opcode_load_fast_pair_checked`
    // in pyopcode.rs.
    let required_methods = [
        "load_local_value",
        "load_local_checked_value",
        "store_local_value",
        "push_value",
        "pop_value",
    ];

    for name in required_methods {
        let via_concrete = cc.resolve_method(name, Some("PyFrame"), None);
        assert!(
            via_concrete.is_some(),
            "`{}` did not resolve against PyFrame receiver; impl count = {}",
            name,
            impls
                .iter()
                .flat_map(|i| i.methods.iter())
                .filter(|m| m.name == name)
                .count()
        );
    }
}
