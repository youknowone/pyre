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
