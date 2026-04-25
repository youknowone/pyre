//! End-to-end composition test: Rust-AST adapter output flows through
//! `RPythonAnnotator::build_types`.
//!
//! This is the payoff test for plan step M2.5g (Position-2 adaptation
//! "Rust AST adapter into unchanged flowspace", see
//! `~/.claude/plans/annotator-monomorphization-tier1-abstract-lake.md`):
//! it proves that a `syn::ItemFn` fed through
//! [`build_host_function_from_rust`] produces a `(HostObject, PyGraph)`
//! pair that upstream's annotator pipeline consumes without
//! modification — the same `build_types` → `get_call_parameters`
//! → `FunctionDesc.specialize` → `cachedgraph` → `buildgraph`
//! → `translator.buildflowgraph` → `_prebuilt_graphs` short-circuit
//! already exercised in unit tests, but driven here from a real
//! adapter-built graph rather than a hand-rolled stub.
//!
//! Upstream analogue:
//!   `annrpython.py:73-97 build_types` dispatches through
//!   `get_call_parameters(function, args_s)` → `FunctionDesc.specialize`
//!   → `cachedgraph` → `buildgraph` → `translator.buildflowgraph(pyobj)`.
//!   The `_prebuilt_graphs` short-circuit at `translator.py:50-51`
//!   returns the graph we seeded without running `build_flow`.

use majit_translate::annotator::annrpython::RPythonAnnotator;
use majit_translate::translator::interactive::Translation;

fn parse_item_fn(src: &str) -> syn::ItemFn {
    syn::parse_str::<syn::ItemFn>(src).expect("test fixture must parse as a single ItemFn")
}

#[test]
fn adapter_output_flows_through_build_types_for_constant_return() {
    // `fn one() -> i64 { 1 }` — the simplest constant-return case the
    // feasibility probe (`annotator_monomorphization` plan, feasibility
    // probe findings #1) validated for the bytecode path. Re-validate
    // the same contract for the Rust-source adapter path.
    //
    // `Translation::from_rust_item_fn` bundles the adapter call +
    // `_prebuilt_graphs` seed per upstream `interactive.py:25-26`.
    let item = parse_item_fn("fn one() -> i64 { 1 }");
    let (t, host) = Translation::from_rust_item_fn(&item).expect("translation");

    let ann = RPythonAnnotator::new_with_translator(
        Some(std::rc::Rc::clone(&t.context)),
        None,
        None,
        false,
    );
    let result = ann
        .build_types(&host, &[], true, false)
        .expect("build_types must succeed");
    // Constant-int return resolves through the returnblock's Link args
    // (the constant-carrying path in the feasibility probe's case #1).
    // Accept any integer annotation — narrower assertions live in the
    // annotator's own unit tests.
    match result {
        Some(sv) => {
            let ty = format!("{sv:?}");
            assert!(
                ty.contains("Integer"),
                "expected SomeValue::Integer, got {ty}"
            );
        }
        None => panic!("build_types should return an annotation for a literal-return function"),
    }
}

#[test]
fn adapter_output_main_entry_point_populates_translator_entry_graph() {
    // Upstream `annrpython.py:87-88`:
    //   `if main_entry_point: self.translator.entry_point_graph = flowgraph`.
    // Confirms the hook the future codewriter wiring (plan M3.2) will
    // use to retrieve the specialized portal graph.
    let item = parse_item_fn("fn one() -> i64 { 1 }");
    let (t, host) = Translation::from_rust_item_fn(&item).expect("translation");

    let ann = RPythonAnnotator::new_with_translator(
        Some(std::rc::Rc::clone(&t.context)),
        None,
        None,
        false,
    );
    let _ = ann
        .build_types(&host, &[], true, true)
        .expect("build_types must succeed");
    assert!(
        ann.translator.entry_point_graph.borrow().is_some(),
        "main_entry_point=true must populate translator.entry_point_graph"
    );
}

#[test]
fn adapter_output_preserves_prebuilt_graph_pop_semantics() {
    // Upstream `translator.py:50-51` pops the entry out of
    // `_prebuilt_graphs` on first lookup. The adapter-produced pair
    // should obey the same contract — subsequent lookups go to
    // `build_flow`, which for an adapter-only HostObject has no
    // bytecode and therefore surfaces an error. This pins the
    // single-consumption contract without asserting the fallback error
    // text (which is an implementation detail of `buildflowgraph`).
    let item = parse_item_fn("fn one() -> i64 { 1 }");
    let (t, host) = Translation::from_rust_item_fn(&item).expect("translation");

    // Capture the prebuilt Rc so we can verify `buildflowgraph` returns
    // the same instance on the first hit.
    let prebuilt = t
        .context
        ._prebuilt_graphs
        .borrow()
        .get(&host)
        .cloned()
        .expect("seeded by Translation::from_rust_item_fn");

    let first = t
        .context
        .buildflowgraph(host.clone(), false)
        .expect("first");
    assert!(std::rc::Rc::ptr_eq(&first, &prebuilt));

    // After the pop, the entry is gone.
    assert!(!t.context._prebuilt_graphs.borrow().contains_key(&host));
}
