//! One-level `*mut bool` actuals passed to a fn-pointer formal.
//!
//! The formal already lowers as `rffi.CArrayPtr(Bool)`. The producer
//! must seed / cast the actual to the same lltype so `SomePtr.call`
//! (`_func::call`) sees matching argument types.

mod common;

use common::{CORPUS_LLBC, load_llbc, lower_function};
use majit_translate::annotator::model::SomeValue;
use majit_translate::front::mir::LowerError;
use majit_translate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, ValueType};
use majit_translate::translator::rtyper::legacy_annotator;
use majit_translate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use majit_translate::translator::rtyper::lltypesystem::rffi::CArrayPtr;

const FIXTURES: [&str; 2] = ["cell_ptr_forward", "cell_ptr_from_local"];

fn actual_passed_to_callback(graph: &FunctionGraph) -> majit_translate::flowspace::model::Variable {
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::IndirectCall { args, .. } => {
                    return args
                        .first()
                        .cloned()
                        .expect("indirect_call of fn(*mut bool) takes the cell pointer");
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments.last().map(String::as_str) == Some("__dyn_call") => {
                    let Some(LinkArg::Value(v)) = args.get(1) else {
                        panic!("__dyn_call threads the cell pointer as args[1]; got {args:?}");
                    };
                    return v.clone();
                }
                _ => {}
            }
        }
    }
    panic!("{}: no call through the fn-pointer formal", graph.name);
}

fn assert_someptr_carrayptr_bool(sv: &SomeValue, where_: &str) {
    let SomeValue::Ptr(ptr) = sv else {
        panic!("{where_}: expected SomePtr(CArrayPtr(Bool)), got {sv:?}");
    };
    assert_eq!(
        LowLevelType::Ptr(Box::new(ptr.ll_ptrtype.clone())),
        CArrayPtr(LowLevelType::Bool),
        "{where_}: actual lltype is CArrayPtr(Bool)"
    );
}

fn annotate_and_assert_actual(graph: &FunctionGraph, name: &str) {
    // Same public annotator `tests/test_import_mir.rs` runs on a graph.
    // `CodeWriter::dual_gate_skip_reason` rtypes corpus IndirectCall /
    // residual helpers and Skips with `KeyError` on an unbound `v`; that
    // is a dual-gate rtyper harness gap, not the producer lltype.
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        legacy_annotator::annotate(graph);
    }));
    if let Err(payload) = panicked {
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "<unrecognised panic payload>".to_string());
        panic!("{name}: annotator panicked: {msg}");
    }
    let actual = actual_passed_to_callback(graph);
    let annotation = actual
        .annotation
        .borrow()
        .clone()
        .unwrap_or_else(|| panic!("{name}: actual has an annotation"));
    assert_someptr_carrayptr_bool(&annotation, name);
}

#[test]
fn cell_ptr_callback_actuals_annotate_as_carrayptr_bool() {
    let llbc = load_llbc(CORPUS_LLBC);
    for name in FIXTURES {
        let graph = lower_function(llbc, name).unwrap_or_else(|err| {
            panic!("{name}: lowering failed: {err}");
        });
        if name == "cell_ptr_forward" {
            let input = graph
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .find_map(|op| match &op.kind {
                    OpKind::Input {
                        name: n,
                        ty,
                        class_root,
                    } if n == "p" => Some((ty.clone(), class_root.clone())),
                    _ => None,
                })
                .expect("cell_ptr_forward has Input p");
            assert_eq!(input.0, ValueType::Int, "p is Int-banked");
            assert_eq!(
                input.1.as_deref(),
                Some("*mut bool"),
                "p class_root is *mut bool"
            );
        }
        annotate_and_assert_actual(&graph, name);
    }
}

fn graph_ops(graph: &FunctionGraph) -> Vec<&OpKind> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .map(|op| &op.kind)
        .collect()
}

fn opkind_is_raw(kind: &OpKind) -> bool {
    matches!(
        kind,
        OpKind::RawMalloc { .. }
            | OpKind::RawFree { .. }
            | OpKind::RawLoad { .. }
            | OpKind::RawStore { .. }
    )
}

fn graph_has_raw_cell_op(graph: &FunctionGraph) -> bool {
    graph_ops(graph).into_iter().any(opkind_is_raw)
}

fn input_named<'a>(
    graph: &'a FunctionGraph,
    name: &str,
) -> (
    &'a majit_translate::flowspace::model::Variable,
    &'a ValueType,
    &'a Option<String>,
) {
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::Input {
                name: n,
                ty,
                class_root,
            } = &op.kind
                && n == name
            {
                let var = op
                    .result
                    .as_ref()
                    .unwrap_or_else(|| panic!("{}: Input {name} has a result", graph.name));
                return (var, ty, class_root);
            }
        }
    }
    panic!("{}: no Input {name}", graph.name);
}

fn aliases_cell(
    graph: &FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
    cell: &majit_translate::flowspace::model::Variable,
) -> bool {
    let mut cur = var.clone();
    for _ in 0..16 {
        if cur == *cell {
            return true;
        }
        let mut next = None;
        for block in &graph.blocks {
            for op in &block.operations {
                let Some(result) = op.result.as_ref() else {
                    continue;
                };
                if result != &cur {
                    continue;
                }
                next = match &op.kind {
                    OpKind::UnaryOp { operand, .. } => Some(operand.clone()),
                    OpKind::Call { args, .. } => {
                        args.iter().find_map(|arg| arg.as_variable().cloned())
                    }
                    _ => None,
                };
            }
        }
        let Some(next) = next else {
            return false;
        };
        cur = next;
    }
    false
}

fn raw_malloc_cells(graph: &FunctionGraph) -> Vec<majit_translate::flowspace::model::Variable> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::RawMalloc { .. } => op.result.clone(),
            _ => None,
        })
        .collect()
}

fn actual_passed_to_bool_mut_set(
    graph: &FunctionGraph,
) -> majit_translate::flowspace::model::Variable {
    let mut indirect = None;
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments.last().map(String::as_str) == Some("bool_mut_set") => {
                    let Some(LinkArg::Value(v)) = args.first() else {
                        panic!(
                            "{}: bool_mut_set call takes the cell pointer; got {args:?}",
                            graph.name
                        );
                    };
                    return v.clone();
                }
                OpKind::IndirectCall { args, .. } if indirect.is_none() => {
                    indirect = args.first().cloned();
                }
                _ => {}
            }
        }
    }
    indirect.unwrap_or_else(|| panic!("{}: no call to bool_mut_set", graph.name))
}

#[test]
fn direct_shared_bool_ref_uses_scalar_alias() {
    let llbc = load_llbc(CORPUS_LLBC);
    let read = lower_function(llbc, "bool_ref_read").unwrap_or_else(|err| {
        panic!("bool_ref_read: lowering failed: {err}");
    });
    let caller = lower_function(llbc, "bool_ref_direct_caller").unwrap_or_else(|err| {
        panic!("bool_ref_direct_caller: lowering failed: {err}");
    });

    let (p, ty, class_root) = input_named(&read, "p");
    assert_ne!(
        p.concretetype(),
        Some(CArrayPtr(LowLevelType::Bool)),
        "bool_ref_read p is not stamped CArrayPtr(Bool); ty={ty:?} class_root={class_root:?}"
    );
    assert_ne!(
        class_root.as_deref(),
        Some("*mut bool"),
        "bool_ref_read p is not the cell-pointer class_root"
    );

    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        legacy_annotator::annotate(&read);
    }));
    if let Err(payload) = panicked {
        let msg = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "<unrecognised panic payload>".to_string());
        panic!("bool_ref_read: annotator panicked: {msg}");
    }
    if let Some(annotation) = p.annotation.borrow().clone()
        && let SomeValue::Ptr(ptr) = &*annotation
    {
        assert_ne!(
            LowLevelType::Ptr(Box::new(ptr.ll_ptrtype.clone())),
            CArrayPtr(LowLevelType::Bool),
            "bool_ref_read p is not annotated CArrayPtr(Bool); got {annotation:?}"
        );
    }

    for (name, graph) in [
        ("bool_ref_read", &read),
        ("bool_ref_direct_caller", &caller),
    ] {
        assert!(
            !graph_has_raw_cell_op(graph),
            "{name} has no raw cell op; ops={:?}",
            graph_ops(graph)
        );
    }
}

#[test]
fn direct_mut_bool_ref_spills_cell() {
    let llbc = load_llbc(CORPUS_LLBC);
    let setter = lower_function(llbc, "bool_mut_set").unwrap_or_else(|err| {
        panic!("bool_mut_set: lowering failed: {err}");
    });
    let caller = lower_function(llbc, "bool_mut_direct_caller").unwrap_or_else(|err| {
        panic!("bool_mut_direct_caller: lowering failed: {err}");
    });

    let cells = raw_malloc_cells(&caller);
    assert!(
        !cells.is_empty(),
        "bool_mut_direct_caller contains a raw cell allocation; ops={:?}",
        graph_ops(&caller)
    );
    let actual = actual_passed_to_bool_mut_set(&caller);
    assert!(
        cells
            .iter()
            .any(|cell| aliases_cell(&caller, &actual, cell)),
        "bool_mut_set actual is the cell (or a cast of it); actual={actual:?} cells={cells:?} ops={:?}",
        graph_ops(&caller)
    );
    assert!(
        graph_ops(&setter)
            .iter()
            .any(|kind| matches!(kind, OpKind::RawStore { .. })),
        "bool_mut_set contains a raw store; ops={:?}",
        graph_ops(&setter)
    );

    let (p, _, _) = input_named(&setter, "p");
    assert_eq!(
        p.concretetype(),
        Some(CArrayPtr(LowLevelType::Bool)),
        "bool_mut_set p is stamped CArrayPtr(Bool)"
    );
    assert_eq!(
        actual.concretetype(),
        Some(CArrayPtr(LowLevelType::Bool)),
        "bool_mut_direct_caller actual is stamped CArrayPtr(Bool)"
    );

    for (name, graph) in [
        ("bool_mut_set", &setter),
        ("bool_mut_direct_caller", &caller),
    ] {
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            legacy_annotator::annotate(graph);
        }));
        if let Err(payload) = panicked {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<unrecognised panic payload>".to_string());
            panic!("{name}: annotator panicked: {msg}");
        }
    }
    let p_ann = p
        .annotation
        .borrow()
        .clone()
        .unwrap_or_else(|| panic!("bool_mut_set: p has an annotation"));
    assert_someptr_carrayptr_bool(&p_ann, "bool_mut_set p");
    let actual_ann = actual
        .annotation
        .borrow()
        .clone()
        .unwrap_or_else(|| panic!("bool_mut_direct_caller: actual has an annotation"));
    assert_someptr_carrayptr_bool(&actual_ann, "bool_mut_direct_caller actual");
}

fn actual_passed_to_bool_ref_read(
    graph: &FunctionGraph,
) -> majit_translate::flowspace::model::Variable {
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } = &op.kind
                && segments.last().map(String::as_str) == Some("bool_ref_read")
            {
                let Some(LinkArg::Value(v)) = args.first() else {
                    panic!(
                        "{}: bool_ref_read call takes the loaded bool; got {args:?}",
                        graph.name
                    );
                };
                return v.clone();
            }
        }
    }
    panic!("{}: no call to bool_ref_read", graph.name)
}

fn producing_op<'a>(
    graph: &'a FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<&'a OpKind> {
    for block in &graph.blocks {
        for op in &block.operations {
            if op.result.as_ref() == Some(var) {
                return Some(&op.kind);
            }
        }
    }
    None
}

#[test]
fn direct_mut_then_shared_bool_ref_loads_cell() {
    let llbc = load_llbc(CORPUS_LLBC);
    let setter = lower_function(llbc, "bool_mut_set").unwrap_or_else(|err| {
        panic!("bool_mut_set: lowering failed: {err}");
    });
    let read = lower_function(llbc, "bool_ref_read").unwrap_or_else(|err| {
        panic!("bool_ref_read: lowering failed: {err}");
    });
    let caller = lower_function(llbc, "bool_mut_then_shared_direct_caller").unwrap_or_else(|err| {
        panic!("bool_mut_then_shared_direct_caller: lowering failed: {err}");
    });

    let cells = raw_malloc_cells(&caller);
    assert!(
        !cells.is_empty(),
        "bool_mut_then_shared_direct_caller contains a raw cell allocation; ops={:?}",
        graph_ops(&caller)
    );
    let mut_actual = actual_passed_to_bool_mut_set(&caller);
    assert!(
        cells
            .iter()
            .any(|cell| aliases_cell(&caller, &mut_actual, cell)),
        "bool_mut_set actual is the cell (or a cast of it); actual={mut_actual:?} cells={cells:?} ops={:?}",
        graph_ops(&caller)
    );
    assert!(
        graph_ops(&setter)
            .iter()
            .any(|kind| matches!(kind, OpKind::RawStore { .. })),
        "bool_mut_set contains a raw store; ops={:?}",
        graph_ops(&setter)
    );

    let (p, _, _) = input_named(&setter, "p");
    assert_eq!(
        p.concretetype(),
        Some(CArrayPtr(LowLevelType::Bool)),
        "bool_mut_set p is stamped CArrayPtr(Bool)"
    );
    assert_eq!(
        mut_actual.concretetype(),
        Some(CArrayPtr(LowLevelType::Bool)),
        "bool_mut_then_shared_direct_caller bool_mut_set actual is stamped CArrayPtr(Bool)"
    );

    let shared_actual = actual_passed_to_bool_ref_read(&caller);
    assert_ne!(
        shared_actual.concretetype(),
        Some(CArrayPtr(LowLevelType::Bool)),
        "bool_ref_read actual is not stamped CArrayPtr(Bool); ops={:?}",
        graph_ops(&caller)
    );
    assert_ne!(
        shared_actual,
        mut_actual,
        "bool_ref_read actual is not the bool_mut_set cell pointer; shared={shared_actual:?} mut={mut_actual:?} ops={:?}",
        graph_ops(&caller)
    );
    match producing_op(&caller, &shared_actual) {
        Some(OpKind::RawLoad { base, item_ty, .. }) => {
            assert!(
                cells.iter().any(|cell| aliases_cell(&caller, base, cell))
                    || (caller
                        .blocks
                        .iter()
                        .any(|block| block.inputargs.contains(base))
                        && base.concretetype() == Some(LowLevelType::Signed)),
                "bool_ref_read RawLoad base is the cell (or a cast of it); base={base:?} cells={cells:?} ops={:?}",
                graph_ops(&caller)
            );
            assert_eq!(
                *item_ty,
                ValueType::Bool,
                "bool_ref_read RawLoad item_ty is Bool"
            );
        }
        other => panic!(
            "bool_ref_read actual is the result of RawLoad; got {other:?} ops={:?}",
            graph_ops(&caller)
        ),
    }

    for (name, graph) in [
        ("bool_mut_set", &setter),
        ("bool_ref_read", &read),
        ("bool_mut_then_shared_direct_caller", &caller),
    ] {
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            legacy_annotator::annotate(graph);
        }));
        if let Err(payload) = panicked {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<unrecognised panic payload>".to_string());
            panic!("{name}: annotator panicked: {msg}");
        }
    }
    let p_ann = p
        .annotation
        .borrow()
        .clone()
        .unwrap_or_else(|| panic!("bool_mut_set: p has an annotation"));
    assert_someptr_carrayptr_bool(&p_ann, "bool_mut_set p");
    let mut_actual_ann = mut_actual.annotation.borrow().clone().unwrap_or_else(|| {
        panic!("bool_mut_then_shared_direct_caller: bool_mut_set actual has an annotation")
    });
    assert_someptr_carrayptr_bool(
        &mut_actual_ann,
        "bool_mut_then_shared_direct_caller bool_mut_set actual",
    );
    let shared_actual_ann = shared_actual
        .annotation
        .borrow()
        .clone()
        .unwrap_or_else(|| {
            panic!("bool_mut_then_shared_direct_caller: bool_ref_read actual has an annotation")
        });
    assert!(
        matches!(*shared_actual_ann, SomeValue::Bool(_)),
        "bool_mut_then_shared_direct_caller: bool_ref_read actual is SomeBool; got {shared_actual_ann:?}"
    );
}

#[test]
fn cell_ptr_field_read_is_carrayptr_bool() {
    let llbc = load_llbc(CORPUS_LLBC);
    let call = lower_function(llbc, "cell_ptr_field_call").unwrap_or_else(|err| {
        panic!("cell_ptr_field_call: lowering failed: {err}");
    });
    let store = lower_function(llbc, "cell_ptr_field_store").unwrap_or_else(|err| {
        panic!("cell_ptr_field_store: lowering failed: {err}");
    });

    let mut field_read_res = None;
    for block in &call.blocks {
        for op in &block.operations {
            if let OpKind::FieldRead { field, .. } = &op.kind
                && field.name == "p"
            {
                let res = op
                    .result
                    .as_ref()
                    .unwrap_or_else(|| panic!("cell_ptr_field_call: FieldRead p has a result"));
                assert_eq!(
                    res.concretetype(),
                    Some(CArrayPtr(LowLevelType::Bool)),
                    "cell_ptr_field_call FieldRead p is stamped CArrayPtr(Bool)"
                );
                field_read_res = Some(res.clone());
            }
        }
    }
    let field_read_res = field_read_res.unwrap_or_else(|| {
        panic!(
            "cell_ptr_field_call: FieldRead of p; ops={:?}",
            graph_ops(&call)
        )
    });

    let actual = actual_passed_to_callback(&call);
    assert!(
        actual == field_read_res || actual.concretetype() == Some(CArrayPtr(LowLevelType::Bool)),
        "cell_ptr_field_call actual is the FieldRead p (or same concretetype); actual={actual:?} field={field_read_res:?} ops={:?}",
        graph_ops(&call)
    );
    annotate_and_assert_actual(&call, "cell_ptr_field_call");

    let mut saw_store = false;
    for block in &store.blocks {
        for op in &block.operations {
            if let OpKind::FieldWrite { field, value, .. } = &op.kind
                && field.name == "p"
            {
                let LinkArg::Value(v) = value else {
                    panic!("cell_ptr_field_store FieldWrite p value is a Variable; got {value:?}");
                };
                assert_eq!(
                    v.concretetype(),
                    Some(CArrayPtr(LowLevelType::Bool)),
                    "cell_ptr_field_store FieldWrite p is stamped CArrayPtr(Bool)"
                );
                saw_store = true;
            }
        }
    }
    assert!(
        saw_store,
        "cell_ptr_field_store: FieldWrite of p; ops={:?}",
        graph_ops(&store)
    );
}

#[test]
fn gcref_local_raw_address_fails_closed() {
    let result = lower_function(load_llbc(CORPUS_LLBC), "gcref_raw_local_to_callback");
    assert!(
        matches!(
            result,
            Err(LowerError::Unsupported(ref message))
                if message.contains("raw address of a bare GCREF local has no traced cell storage")
        ),
        "bare GCREF local must decline instead of producing a GCREF word as CArrayPtr(GCREF): {result:?}"
    );
}
