//! Phase B.2: callcontrol trait-method resolution over real pyre-interpreter
//! sources.
//!
//! For line-by-line parity with RPython's `rpython/jit/codewriter/call.py` we
//! need every `executor.method(...)` call in `execute_opcode_step` and its
//! helper `opcode_*` functions to resolve to a single concrete graph. RPython
//! achieves this because the annotator's `SomeInstance(classdef)` already
//! identifies a unique class and `bookkeeper.py:318 methoddesc` tracks the
//! graph per method. pyre's analogue is `CallControl::resolve_method` combined
//! with the `extract_trait_impls` / `extract_opcode_dispatch_receiver_traits`
//! pass.
//!
//! This test reads `pyre-interpreter/src/pyopcode.rs` (for trait declarations
//! and the dispatch function) and `pyre-interpreter/src/eval.rs` (for the
//! `impl <Trait> for PyFrame` blocks), builds a `CallControl`, and asserts
//! that the method names invoked by the super-instruction helpers
//! (`opcode_load_fast_load_fast`, `opcode_load_fast_pair_checked`,
//! `opcode_store_fast_load_fast`, `opcode_store_fast_store_fast`) all resolve
//! to a concrete PyFrame graph — both when the receiver is the concrete type
//! `PyFrame` and when it is the generic parameter `E` (the actual dispatch
//! site in pyopcode.rs).

use std::path::PathBuf;

use majit_translate::{
    ParsedInterpreter, call::CallControl, extract_opcode_dispatch_receiver_traits,
    extract_trait_impls, front::StructFieldRegistry, parse_source,
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

    // Step 2: collect trait impls from both files. Empty registries are
    // sufficient because the resolution here keys on (trait_name, impl_type)
    // and does not need struct-layout information.
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = std::collections::HashMap::new();
    let empty_struct_names = std::collections::HashSet::new();
    let mut impls = Vec::new();
    impls.extend(extract_trait_impls(
        &pyopcode,
        &empty_registry,
        &empty_fn_ret,
        &empty_struct_names,
    ));
    impls.extend(extract_trait_impls(
        &eval,
        &empty_registry,
        &empty_fn_ret,
        &empty_struct_names,
    ));

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
    // must resolve — both against the concrete PyFrame receiver and against
    // the generic parameter `E` (which is the actual receiver at the dispatch
    // site `executor.load_local_value(...)`).
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
        let via_concrete = cc.resolve_method(name, Some("PyFrame"));
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
        let via_generic = cc.resolve_method(name, Some("E"));
        assert!(
            via_generic.is_some(),
            "`{}` did not resolve against generic receiver `E`; \
             CallControl should pick the unique concrete impl (PyFrame) \
             when the receiver is a generic parameter",
            name
        );
    }
}
