//! Phase B (lucky-growing-puzzle) — Bookkeeper trait resolution coverage
//! against the real pyre-interpreter source.
//!
//! ## RPython positioning
//!
//! Upstream resolves `executor.method(...)` calls inside the interpreter
//! dispatch via `rpython/annotator/bookkeeper.py:318 methoddesc` +
//! `rpython/jit/codewriter/call.py:94-155 guess_call_kind`. Pyre's
//! analogue lives in `majit-translate/src/parse.rs::extract_opcode_dispatch_receiver_traits`
//! (collects `<E: TraitA + TraitB + ...>` bounds from
//! `execute_opcode_step`) plus `lib.rs::resolve_handler_calls` /
//! `push_matching_trait_methods` (walks trait impls and picks the
//! concrete or default method). That machinery is already written;
//! this test only verifies it produces the expected coverage against
//! the **real** `pyre-interpreter/src/pyopcode.rs` +
//! `pyre-interpreter/src/eval.rs` pair so Phase E (`make_jitcodes`)
//! can depend on every trait method resolving to a PyFrame graph.
//!
//! RPython parity references consulted:
//! - `rpython/annotator/bookkeeper.py:318` — methoddesc dispatch
//! - `rpython/jit/codewriter/call.py:94-155` — guess_call_kind +
//!   direct_call resolution
//!
//! ## What this test enforces
//!
//! - **Receiver bindings** (B.1): `extract_opcode_dispatch_receiver_traits`
//!   on the real `execute_opcode_step<E: ...>(executor: &mut E, ...)`
//!   produces `traits_by_receiver["executor"]` containing every handler
//!   trait used by the dispatch.
//! - **Unique PyFrame implementations** (B.2): for every trait in those
//!   bindings, exactly one concrete `impl <Trait> for PyFrame {...}`
//!   exists — no two distinct impls claim the same trait against
//!   `PyFrame`. If ambiguity appeared, resolution would become
//!   non-deterministic.
//! - **Method closure** (B.2): every method declared in an
//!   `impl <Trait> for PyFrame` is backed by a real `FunctionGraph`
//!   (methods without graphs would leave Phase E with a hole).
//!
//! This is strictly a parity-verification harness. No production code
//! is touched.

use std::collections::{BTreeMap, HashMap, HashSet};

use majit_translate::{
    ParsedInterpreter, TraitImplInfo, extract_opcode_dispatch_receiver_traits, extract_trait_impls,
    front::StructFieldRegistry, parse_source,
};

/// Source baked into the test binary so the assertions do not drift
/// when the checkout is moved. Mirrors the list in
/// `majit-translate/src/generated.rs:109-114` so resolution operates on
/// the exact set of files Phase E will feed into `make_jitcodes`.
const PYOPCODE_SRC: &str = include_str!("../../../pyre/pyre-interpreter/src/pyopcode.rs");
const EVAL_SRC: &str = include_str!("../../../pyre/pyre-interpreter/src/eval.rs");
const PYFRAME_SRC: &str = include_str!("../../../pyre/pyre-interpreter/src/pyframe.rs");
const SHARED_OPCODE_SRC: &str = include_str!("../../../pyre/pyre-interpreter/src/shared_opcode.rs");

fn parse_pyre_source(src: &str) -> ParsedInterpreter {
    parse_source(src)
}

fn collect_all_trait_impls() -> Vec<TraitImplInfo> {
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret: HashMap<String, String> = HashMap::new();
    let empty_struct_names: HashSet<String> = HashSet::new();
    let mut out = Vec::new();
    for src in [PYOPCODE_SRC, EVAL_SRC, PYFRAME_SRC, SHARED_OPCODE_SRC] {
        let parsed = parse_pyre_source(src);
        out.extend(extract_trait_impls(
            &parsed,
            &empty_registry,
            &empty_fn_ret,
            &empty_struct_names,
        ));
    }
    out
}

/// Canonical list of handler traits the Phase B/E pipeline depends on.
/// If `execute_opcode_step` grows a new trait bound, add it here so the
/// test becomes the compile-time contract update, not a silent drift.
const EXPECTED_HANDLER_TRAITS: &[&str] = &[
    "OpcodeStepExecutor",
    "SharedOpcodeHandler",
    "ConstantOpcodeHandler",
    "LocalOpcodeHandler",
    "NamespaceOpcodeHandler",
    "StackOpcodeHandler",
    "IterOpcodeHandler",
    "TruthOpcodeHandler",
    "ControlFlowOpcodeHandler",
    "BranchOpcodeHandler",
    "ArithmeticOpcodeHandler",
];

#[test]
fn execute_opcode_step_receiver_binds_every_handler_trait() {
    // B.1: run `extract_opcode_dispatch_receiver_traits` against the
    // real pyopcode.rs and assert it picks up both the direct bound
    // (`<E: OpcodeStepExecutor>`) and the `where E: ...` clause that
    // names every concrete handler trait. RPython parity:
    // `rpython/annotator/bookkeeper.py:318 methoddesc` dispatches per
    // trait, so the bookkeeper must see the full trait set.
    let parsed = parse_pyre_source(PYOPCODE_SRC);
    let bindings = extract_opcode_dispatch_receiver_traits(&parsed);

    let executor_type_root = bindings
        .type_root_by_receiver
        .get("executor")
        .expect("executor receiver missing type_root");
    assert_eq!(
        executor_type_root, "E",
        "execute_opcode_step is <E: ...>(executor: &mut E); \
         type_root must stay `E` for bookkeeper to correlate trait \
         bounds with the receiver parameter"
    );

    let traits = bindings
        .traits_by_receiver
        .get("executor")
        .expect("executor receiver missing trait bindings");

    for expected in EXPECTED_HANDLER_TRAITS {
        assert!(
            traits.iter().any(|t| t == expected),
            "execute_opcode_step bindings missing `{expected}`: got {traits:?}"
        );
    }
}

/// Return the subset of `impls` where `for_type == "PyFrame"`.
fn pyframe_impls(impls: &[TraitImplInfo]) -> Vec<&TraitImplInfo> {
    impls.iter().filter(|i| i.for_type == "PyFrame").collect()
}

#[test]
fn every_handler_trait_has_a_unique_pyframe_impl() {
    // B.2: for each of the canonical handler traits there must be
    // exactly one concrete `impl <Trait> for PyFrame` across the
    // pyre-interpreter source. RPython parity:
    // `rpython/jit/codewriter/call.py:94-155 guess_call_kind` resolves
    // to a single implementation; ambiguity (two impls of the same
    // trait for the same type) would trip upstream too.
    let impls = collect_all_trait_impls();
    let pyframe = pyframe_impls(&impls);

    let mut per_trait_impl_count: BTreeMap<&str, usize> = BTreeMap::new();
    for imp in &pyframe {
        *per_trait_impl_count
            .entry(imp.trait_name.as_str())
            .or_insert(0) += 1;
    }

    for expected in EXPECTED_HANDLER_TRAITS {
        let count = per_trait_impl_count.get(expected).copied().unwrap_or(0);
        assert_eq!(
            count, 1,
            "expected exactly one `impl {expected} for PyFrame` \
             (ambiguity would make trait resolution non-deterministic), \
             got {count}; per-trait counts: {per_trait_impl_count:?}"
        );
    }
}

#[test]
fn pyframe_trait_methods_all_have_function_graphs() {
    // B.2: every method declared on a `impl <Trait> for PyFrame` must
    // be backed by a real `FunctionGraph`. A missing graph would leave
    // Phase E (`CodeWriter::make_jitcodes`) with an unresolvable
    // direct_call target for the corresponding opcode handler.
    let impls = collect_all_trait_impls();
    let pyframe = pyframe_impls(&impls);
    assert!(
        !pyframe.is_empty(),
        "expected at least one `impl <Trait> for PyFrame` across pyre-interpreter"
    );

    let mut missing: Vec<(String, String)> = Vec::new();
    for imp in &pyframe {
        for method in &imp.methods {
            if method.graph.is_none() {
                missing.push((imp.trait_name.clone(), method.name.clone()));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "PyFrame trait methods without FunctionGraph: {missing:?}"
    );
}

#[test]
fn trait_method_resolution_is_globally_unambiguous() {
    // B.2 corner-case: verify **no** trait method across the whole
    // pyre-interpreter has two concrete implementations claiming the
    // same (trait_name, method_name, for_type) triple. The dispatcher
    // at `lib.rs::resolve_handler_calls` already picks the first
    // concrete match; duplicate entries would hide whichever impl is
    // actually compiled into the binary, making the parity choice
    // invisible to reviewers.
    let impls = collect_all_trait_impls();
    let mut seen: HashMap<(String, String, String), usize> = HashMap::new();
    for imp in &impls {
        // Skip the synthetic "<default methods of T>" bucket — by
        // construction it collides with every concrete impl of the
        // same method, and the dispatcher handles that via the
        // `is_default_methods` branch at lib.rs:960.
        if imp.for_type.starts_with("<default methods of ") {
            continue;
        }
        for method in &imp.methods {
            let key = (
                imp.trait_name.clone(),
                method.name.clone(),
                imp.for_type.clone(),
            );
            *seen.entry(key).or_insert(0) += 1;
        }
    }
    let duplicates: Vec<_> = seen.iter().filter(|&(_, &n)| n > 1).collect();
    assert!(
        duplicates.is_empty(),
        "ambiguous trait method resolution: {duplicates:?} — \
         each (trait, method, for_type) triple must map to a single impl"
    );
}
