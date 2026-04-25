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
    impls.extend(
        extract_trait_impls(
            &pyopcode,
            &empty_registry,
            &empty_fn_ret,
            &empty_struct_names,
        )
        .expect("pyopcode trait impls must lower"),
    );
    impls.extend(
        extract_trait_impls(&eval, &empty_registry, &empty_fn_ret, &empty_struct_names)
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
    }
}
