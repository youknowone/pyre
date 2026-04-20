//! End-to-end integration test for `flowspace::objspace::build_flow`.
//!
//! RPython basis: `rpython/flowspace/test/test_objspace.py` end-to-end
//! style, but focused on a narrow smoke path that exercises every
//! flowspace module landed by Phases 1–3:
//!
//!   source →  rustpython-compiler  →  CodeObject
//!    └→  HostCode::from_code
//!    └→  GraphFunc::from_host_code
//!    └→  objspace::build_flow
//!    └→  checkgraph(graph)
//!
//! This is the phase-exit test for Phase 3 F3.9. It does not assert
//! fine-grained SpaceOperation ordering — that lives inside the
//! per-handler unit tests on flowcontext.rs. What this test does
//! pin is: build_flow never panics on a realistic `def f(x): return x
//! + 1` input and the resulting graph passes checkgraph.

use majit_translate::flowspace::bytecode::{ConstantData, HostCode};
use majit_translate::flowspace::model::{BlockRefExt, ConstValue, Constant, GraphFunc, checkgraph};
use majit_translate::flowspace::objspace::build_flow;
use rustpython_compiler::{Mode, compile as rp_compile};
use rustpython_compiler_core::bytecode::CodeObject;

fn compile_first_function(src: &str) -> CodeObject {
    let module = rp_compile(src, Mode::Exec, "<pyre>".into(), Default::default())
        .expect("compile should succeed");
    module
        .constants
        .iter()
        .find_map(|c| match c {
            ConstantData::Code { code } => Some((**code).clone()),
            _ => None,
        })
        .expect("source should contain at least one function body")
}

fn empty_globals() -> Constant {
    Constant::new(ConstValue::Dict(Default::default()))
}

fn graph_func_from_source(src: &str) -> GraphFunc {
    let code = compile_first_function(src);
    let host = HostCode::from_code(&code);
    GraphFunc::from_host_code(host, empty_globals(), Vec::new())
}

#[test]
fn build_flow_for_constant_return() {
    // simplest case: `def f(): return 1` — one block, one return.
    let func = graph_func_from_source("def f():\n    return 1\n");
    let graph = build_flow(func).expect("build_flow must succeed");

    // Smoke-check: iterblocks visits at least the return block and the
    // startblock.
    let blocks = graph.iterblocks();
    assert!(
        blocks.len() >= 1,
        "expected at least one block, got {}",
        blocks.len()
    );

    checkgraph(&graph);
}

#[test]
fn build_flow_for_one_arg_adder() {
    // `def f(x): return x + 1` — exercises LoadFast + LoadConst +
    // BinaryOp + ReturnValue handler chain.
    let func = graph_func_from_source("def f(x):\n    return x + 1\n");
    let graph = build_flow(func).expect("build_flow must succeed");

    assert_eq!(graph.name, "f");
    let startblock = graph.startblock.clone();
    assert_eq!(
        startblock.borrow().inputargs.len(),
        1,
        "one formal arg → one inputarg"
    );

    checkgraph(&graph);
}

#[test]
fn build_flow_rejects_missing_co_newlocals() {
    // Construct a GraphFunc whose HostCode has co_flags = 0. This
    // mirrors the upstream `_assert_rpythonic` branch that rejects
    // exec-level code objects (no CO_NEWLOCALS).
    let code = compile_first_function("def f():\n    return 1\n");
    let mut host = HostCode::from_code(&code);
    host.co_flags = 0; // strip CO_NEWLOCALS
    let func = GraphFunc::from_host_code(host, empty_globals(), Vec::new());

    let err = build_flow(func).expect_err("missing CO_NEWLOCALS must reject");
    assert!(format!("{err:?}").contains("CO_NEWLOCALS"));
}

#[test]
fn build_flow_rejects_closure_bearing_func() {
    let code = compile_first_function("def f():\n    return 1\n");
    let host = HostCode::from_code(&code);
    let mut func = GraphFunc::from_host_code(host, empty_globals(), Vec::new());
    func.closure.push(Constant::new(ConstValue::Int(1)));

    let err = build_flow(func).expect_err("closure-bearing func must reject");
    assert!(format!("{err:?}").contains("closures"));
}
