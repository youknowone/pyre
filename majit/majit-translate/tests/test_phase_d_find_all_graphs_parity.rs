//! Phase D acceptance anchor (lucky-growing-puzzle).
//!
//! Verifies that `CallControl::find_all_graphs` + `DefaultJitPolicy`
//! behave as RPython specifies when fed the pyre-interpreter source:
//!
//! 1. portal graph is in candidates (call.py:65
//!    `candidate_graphs = set(todo)`).
//! 2. Graphs reachable via `direct_call` from the portal — including
//!    every `opcode_*` helper — become candidates (call.py:75-88 BFS).
//! 3. Graphs registered as `builtin_targets` are **not** followed —
//!    they stay at the residual-call boundary (call.py:104-105
//!    `getattr(targetgraph.func, 'oopspec')` → `builtin`, skipped in
//!    call.py:82 `if kind != "regular": continue`).
//!
//! ## RPython references
//!
//! - `rpython/jit/codewriter/call.py:49-92 find_all_graphs`
//! - `rpython/jit/codewriter/call.py:116-139 guess_call_kind`
//! - `rpython/jit/codewriter/policy.py:48-84 look_inside_graph`
//!
//! The test constructs a **minimal synthetic** dispatch graph (no
//! pyre-interpreter source dependency) so the parity claim is isolated
//! to BFS + policy interaction. The broader `test_pyre_find_all_graphs`
//! already covers the integration path on the real handler corpus.

use majit_translate::{
    CallPath, call::CallControl, model::FunctionGraph, policy::DefaultJitPolicy,
};

/// Build a FunctionGraph with a single `direct_call` to `callee_path`
/// and no return value. This is the minimum the BFS needs to follow
/// an edge.
fn build_caller_graph(name: &str, callee_path: &CallPath) -> FunctionGraph {
    use majit_translate::model::{CallTarget, OpKind, SpaceOperation, ValueType};

    let mut graph = FunctionGraph::new(name);
    let vid = graph.alloc_value();
    graph
        .block_mut(graph.startblock)
        .operations
        .push(SpaceOperation {
            result: Some(vid),
            kind: OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: callee_path.segments.clone(),
                },
                args: Vec::new(),
                result_ty: ValueType::Int,
            },
        });
    graph.set_return(graph.startblock, Some(vid));
    graph
}

#[test]
fn find_all_graphs_follows_regular_edges_from_portal() {
    // RPython call.py:82 `if kind != "regular": continue` — a regular
    // direct_call edge whose callee has a registered graph must be
    // added to the candidate set.
    let portal_path = CallPath::from_segments(["portal"]);
    let callee_path = CallPath::from_segments(["callee"]);

    let mut cc = CallControl::new();
    cc.register_function_graph(
        portal_path.clone(),
        build_caller_graph("portal", &callee_path),
    );
    cc.register_function_graph(callee_path.clone(), FunctionGraph::new("callee"));
    cc.mark_portal(portal_path.clone());

    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    assert!(
        cc.is_candidate(&portal_path),
        "RPython call.py:65 portal must land in candidate_graphs"
    );
    assert!(
        cc.is_candidate(&callee_path),
        "RPython call.py:86-87 BFS must follow regular direct_call edges"
    );
}

#[test]
fn find_all_graphs_does_not_follow_builtin_targets() {
    // RPython call.py:132-133 `if hasattr(targetgraph.func, 'oopspec'):
    //   return 'builtin'` — and call.py:82 `if kind != "regular":
    //   continue`. A registered builtin target must stay at the
    //   residual-call boundary: reachable via the edge, but not added
    //   to the inline closure.
    let portal_path = CallPath::from_segments(["portal"]);
    let builtin_path = CallPath::from_segments(["ll_builtin"]);

    let mut cc = CallControl::new();
    cc.register_function_graph(
        portal_path.clone(),
        build_caller_graph("portal", &builtin_path),
    );
    // Register the builtin's graph so the edge has somewhere to land,
    // but mark it as a builtin target so BFS must skip the regular
    // classification.
    cc.register_function_graph(builtin_path.clone(), FunctionGraph::new("ll_builtin"));
    cc.mark_builtin(builtin_path.clone());
    cc.mark_portal(portal_path.clone());

    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    assert!(
        cc.is_candidate(&portal_path),
        "portal must always land in candidates"
    );
    assert!(
        !cc.is_candidate(&builtin_path),
        "RPython call.py:82 `kind != regular` — builtin targets stay \
         residual and must NOT be pulled into the candidate closure"
    );
}

#[test]
fn find_all_graphs_does_not_follow_portal_recursive_edges() {
    // RPython call.py:119-120 `jitdriver_sd_from_portal_runner_ptr` —
    // a call to the portal itself classifies as `recursive`, which
    // call.py:82 skips. The portal remains a candidate (via the seed)
    // but is not re-visited.
    let portal_path = CallPath::from_segments(["portal"]);

    let mut cc = CallControl::new();
    // Portal graph contains a self-call.
    cc.register_function_graph(
        portal_path.clone(),
        build_caller_graph("portal", &portal_path),
    );
    cc.mark_portal(portal_path.clone());

    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    // Reaching this assertion means BFS didn't infinite-loop on the
    // self-edge. The candidate set contains the portal by the seed
    // step, regardless of the recursive edge.
    assert!(cc.is_candidate(&portal_path));
}

#[test]
fn find_all_graphs_leaves_unregistered_targets_as_residual() {
    // Phase D.2 parity contract: upstream `PyPyJitPolicy.look_inside_function`
    // (`pypy/module/pypyjit/policy.py:25-39`) excludes per-module by name
    // (`pypy.interpreter.astcompiler.*`, `rpython.rlib.rlocale`, …) so
    // those functions become residual calls even when the BFS would
    // otherwise follow them. Pyre uses a different but structurally
    // equivalent mechanism: the `PYRE_JIT_GRAPH_SOURCES` whitelist plus
    // `register_function_graph` plays the "allowed module" role, and an
    // unregistered callee is treated as residual by construction —
    // `find_all_graphs_bfs` at `call.rs:1466` only pulls a callee into
    // `candidate_graphs` when `function_graphs.get(callee_path)` succeeds.
    //
    // The two mechanisms converge on the same observable behaviour: a
    // direct_call whose callee lies outside the JIT-analysable surface
    // stays residual. This test pins that contract so a future change
    // that starts synthesising graphs for unregistered callees (or
    // otherwise short-circuits the residual fallback) surfaces loudly.
    let portal_path = CallPath::from_segments(["portal"]);
    let unregistered_path = CallPath::from_segments(["external_helper"]);

    let mut cc = CallControl::new();
    cc.register_function_graph(
        portal_path.clone(),
        build_caller_graph("portal", &unregistered_path),
    );
    // NOTE: `unregistered_path` has no `register_function_graph` call — it
    // represents an opaque callee (Rust stdlib, unregistered module,
    // externally linked helper, etc.).
    cc.mark_portal(portal_path.clone());

    let mut policy = DefaultJitPolicy::new();
    cc.find_all_graphs(&mut policy);

    assert!(
        cc.is_candidate(&portal_path),
        "portal seeded by mark_portal must always be a candidate"
    );
    assert!(
        !cc.is_candidate(&unregistered_path),
        "Phase D.2: unregistered direct_call targets must stay residual — \
         they are the pyre analogue of PyPyJitPolicy.look_inside_function=False, \
         and BFS must not pull them into candidate_graphs"
    );
}
