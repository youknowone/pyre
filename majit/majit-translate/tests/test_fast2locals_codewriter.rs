//! `PyFrame::fast2locals` driven through the codewriter, against the shipped
//! interpreter LLBC.
//!
//! A graph containing a loop is normally residualized rather than looked
//! inside (`policy.py:50,61-62 contains_loop`), so nothing in the shipped
//! build exercises the front-through-assembler path on a looping graph that
//! reads the frame's locals array.  `fast2locals` is the smallest real graph
//! with that shape, and driving it here rather than waiting for a policy that
//! admits it is what surfaced the element-type defect below.

use majit_charon_reader::Llbc;
use majit_translate::codewriter::call::CallControl;
use majit_translate::codewriter::codewriter::CodeWriter;
use majit_translate::codewriter::jitcode::JitCode;
use majit_translate::front::mir::lower_fun_decl;
use majit_translate::model::{CallTarget, FunctionGraph, OpKind, ValueType};
use majit_translate::{CallPath, GraphTransformConfig, VirtualizableFieldDescriptor};

const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc"
);

/// Lower `pyframe::<Impl>::fast2locals` out of the shipped LLBC, or `None`
/// when the artefact is absent so the tests degrade to a skip.
fn lower_fast2locals() -> Option<FunctionGraph> {
    if !std::path::Path::new(INTERPRETER_LLBC).is_file() {
        eprintln!("skipping: {INTERPRETER_LLBC} is missing");
        return None;
    }
    let llbc = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter.ullbc");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path().ends_with("::fast2locals"))
        .expect("fast2locals present in the shipped LLBC");
    Some(lower_fun_decl(&llbc, fd).expect("lower fast2locals"))
}

/// The virtualizable configuration the production pipeline passes
/// (`pyre-jit-trace/src/virtualizable_spec.rs PYFRAME_VABLE_ARRAYS`).
fn config() -> GraphTransformConfig {
    GraphTransformConfig {
        vable_arrays: vec![VirtualizableFieldDescriptor::new(
            "locals_cells_stack_w",
            Some("PyFrame".to_string()),
            0,
        )],
        ..Default::default()
    }
}

/// The whole pipeline — jtransform, regalloc, flatten, assemble — runs to
/// completion on `fast2locals`.
///
/// `front::iter_next` typed the `range` iterator's element as a GC
/// reference, which put the loop counter in the ref register bank and made
/// every `array[i]` in the body a `getarrayitem_gc` with a ref index.
/// `assembler.rs` asserts that index kind, so it aborted the build outright
/// — asserting that this call returns at all is the assertion.
///
/// The bare `CallControl` here is not a simplification: the production
/// pipeline also declines `fast2locals` in the real rtyper's two-phase
/// prepass and types it with the legacy walker, which is the tier that reads
/// `result_ty` off the op.
#[test]
fn fast2locals_assembles() {
    let Some(graph) = lower_fast2locals() else {
        return;
    };
    let path = CallPath::from_segments(["pyre_interpreter", "pyframe", "fast2locals"]);
    let mut callcontrol = CallControl::new();
    callcontrol.register_function_graph(path.clone(), graph.clone());
    let mut codewriter = CodeWriter::new();
    let jitcode = std::sync::Arc::new(JitCode::new("fast2locals"));
    codewriter.transform_graph_to_jitcode(
        &graph,
        &path,
        &mut callcontrol,
        &config(),
        &jitcode,
        false,
        0,
    );
}

/// A `for i in a..b` loop yields an integer, not a GC reference.
///
/// `front::range_iter` reroutes the exclusive int `Range` onto the `range()`
/// builtin plus the `iter` bridge so the receiver selects upstream's
/// `RangeIteratorRepr`, whose `rtype_next` (`rrange.py ll_rangenext_*`)
/// yields a `Signed`.  The `[__iter_next]` marker `front::iter_next` folds
/// the `next` diamond into has to agree: the legacy type walker reads
/// `result_ty` straight off the op, and it is the tier `fast2locals` lands
/// in.
#[test]
fn a_range_loops_next_yields_an_int_element() {
    let Some(graph) = lower_fast2locals() else {
        return;
    };

    // The `range()` results, so an `[__iter_next]` can be attributed to a
    // range rather than to a slice/Vec iterator.
    let mut range_results = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                ..
            } = &op.kind
                && segments.len() == 1
                && segments[0] == "__pyre_range"
                && let Some(result) = &op.result
            {
                range_results.push(result.clone());
            }
        }
    }
    assert!(
        !range_results.is_empty(),
        "`fast2locals` has two `for i in 0..n` loops; neither reached \
         `front::range_iter`, so this test no longer exercises anything"
    );

    // An iterator built over one of those ranges, threaded to its `next`.
    let iterators: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                args,
                ..
            } if segments.last().is_some_and(|s| s == "iter")
                && args.first().is_some_and(|a| range_results.contains(a)) =>
            {
                op.result.clone()
            }
            _ => None,
        })
        .collect();
    assert!(
        !iterators.is_empty(),
        "no `iter` op over a `range()` result — the reroute changed shape"
    );

    let mut checked = 0;
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                args,
                result_ty,
            } = &op.kind
            else {
                continue;
            };
            // `front::iter_next`'s marker spelling; the module is
            // crate-private, so match the reserved segment directly rather
            // than widening its visibility for a test.
            if segments.len() != 1 || segments[0] != "__iter_next" {
                continue;
            }
            // The iterator reaches `next` through the loop header's phi, so
            // match on the value that entered it rather than on the operand.
            if !args
                .first()
                .is_some_and(|a| iterators.contains(a) || reaches(&graph, a, &iterators))
            {
                continue;
            }
            checked += 1;
            assert_eq!(
                result_ty,
                &ValueType::Int,
                "the `range` iterator's `next` yields a GC reference, so the \
                 loop counter lands in the ref register bank",
            );
        }
    }
    assert!(
        checked > 0,
        "no `[__iter_next]` was attributed to a range iterator"
    );
}

/// Whether `var` is one of `sources`, following block inputargs back through
/// every predecessor's link argument in the matching slot.  The loop-carried
/// iterator arrives at `next` as a phi, not as the `iter` op's result.
fn reaches(
    graph: &FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
    sources: &[majit_translate::flowspace::model::Variable],
) -> bool {
    let mut visited = Vec::new();
    let mut stack = vec![var.clone()];
    while let Some(v) = stack.pop() {
        if visited.contains(&v) {
            continue;
        }
        if sources.contains(&v) {
            return true;
        }
        visited.push(v.clone());
        for b in &graph.blocks {
            let Some(slot) = b.inputargs.iter().position(|iv| iv == &v) else {
                continue;
            };
            for pb in &graph.blocks {
                for link in &pb.exits {
                    if link.target == b.id
                        && let Some(majit_translate::model::LinkArg::Value(src)) =
                            link.args.get(slot)
                    {
                        stack.push(src.clone());
                    }
                }
            }
        }
    }
    false
}
