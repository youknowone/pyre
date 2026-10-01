//! A call that returns a non-byte raw scalar pointer uses the integer
//! bank. `history.py` `getkind` banks a raw `Ptr` as `int`, and
//! `pygraph_initial_block` already records that parameter as `Int`.
//! `CallControl::getcalldescr` compares the two.
//!
//! `IntArray::base` and `FloatArray::base` return the GC block header,
//! so those calls stay `Ref`. An integer cast to a non-byte raw scalar
//! stays in the int bank.
//!
//! A byte pointer stays `Ref`: pyre erases a GC reference to `*mut u8`.

use majit_charon_reader::ullbc::NameSeg;
use majit_charon_reader::{FunDecl, Llbc};
use majit_translate::{
    front::mir::{LowerContext, lower_fun_decl, lower_function},
    model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType},
};

const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc"
);
const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);

fn defining_op<'a>(
    graph: &'a FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<&'a OpKind> {
    graph.blocks.iter().find_map(|block| {
        block
            .operations
            .iter()
            .find_map(|op| (op.result.as_ref() == Some(var)).then_some(&op.kind))
    })
}

fn root_op<'a>(
    graph: &'a FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> Option<&'a OpKind> {
    let mut current = var.clone();
    for _ in 0..32 {
        if let Some(op) = defining_op(graph, &current) {
            return Some(op);
        }
        let mut next = None;
        for block in &graph.blocks {
            let Some(slot) = block.inputargs.iter().position(|arg| arg == &current) else {
                continue;
            };
            for pred in &graph.blocks {
                for exit in &pred.exits {
                    if exit.target == block.id
                        && let Some(src) = exit.args.get(slot).and_then(LinkArg::as_variable)
                        && src != &current
                    {
                        next = Some(src.clone());
                    }
                }
            }
        }
        match next {
            Some(src) => current = src,
            None => return None,
        }
    }
    None
}

fn returned_var(graph: &FunctionGraph) -> majit_translate::flowspace::model::Variable {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.exits)
        .find(|link| link.target == graph.returnblock)
        .and_then(|link| link.args.first())
        .and_then(LinkArg::as_variable)
        .cloned()
        .unwrap_or_else(|| panic!("{} does not return a variable", graph.name))
}

fn call_leaf(target: &CallTarget) -> Option<&str> {
    match target {
        CallTarget::FunctionPath { segments, .. } => segments.last().map(String::as_str),
        CallTarget::Method { name, .. } => Some(name.as_str()),
        _ => None,
    }
}

fn ops(graph: &FunctionGraph) -> impl Iterator<Item = &SpaceOperation> {
    graph.blocks.iter().flat_map(|block| &block.operations)
}

fn call_result<'a>(kind: &'a OpKind) -> Option<(&'a ValueType, Option<&'a str>)> {
    match kind {
        OpKind::Call {
            result_ty, target, ..
        } => Some((result_ty, call_leaf(target))),
        OpKind::IndirectCall { result_ty, .. } => Some((result_ty, None)),
        _ => None,
    }
}

/// `name_path` renders every inherent impl as `<Impl>`, so
/// `ActionFlag::ticker_addr` and `SpaceActionFlag::ticker_addr` share one
/// suffix. The impl segment's `Self` type is the receiver.
fn receiver_path(llbc: &Llbc, fd: &FunDecl) -> Option<String> {
    let id = fd.item_meta.name.iter().find_map(|seg| match seg {
        NameSeg::Other(value) => value
            .pointer("/Impl/Ty/skip_binder/Deduplicated")
            .and_then(|value| value.as_u64()),
        NameSeg::Ident { .. } => None,
    })?;
    let def_id = llbc.dedup_to_adt_def_id(id)?;
    Some(llbc.type_by_id(def_id)?.item_meta.name_path())
}

fn find_method<'a>(llbc: &'a Llbc, leaf: &str, owner: &str) -> &'a FunDecl {
    let suffix = format!("::{leaf}");
    let owner_suffix = format!("::{owner}");
    llbc.iter_local_fns()
        .find(|fd| {
            fd.item_meta.name_path().ends_with(&suffix)
                && receiver_path(llbc, fd)
                    .is_some_and(|path| path == owner || path.ends_with(&owner_suffix))
        })
        .unwrap_or_else(|| panic!("missing {owner}::{leaf}"))
}

fn find_exact<'a>(llbc: &'a Llbc, path: &str) -> &'a FunDecl {
    llbc.iter_local_fns()
        .find(|fd| fd.item_meta.name_path() == path)
        .unwrap_or_else(|| panic!("missing {path}"))
}

fn call_results_named<'a>(graph: &'a FunctionGraph, leaf: &str) -> Vec<&'a ValueType> {
    ops(graph)
        .filter_map(|op| {
            let (ty, name) = call_result(&op.kind)?;
            (name == Some(leaf)).then_some(ty)
        })
        .collect()
}

fn assert_returned_call(graph: &FunctionGraph, leaf: &str) {
    match root_op(graph, &returned_var(graph)).and_then(call_result) {
        Some((ValueType::Int, Some(got))) if got == leaf => {}
        other => panic!(
            "{} must return an int `{leaf}` call, got {other:?}",
            graph.name
        ),
    }
}

#[test]
fn call_returned_raw_scalar_pointer_is_int() {
    let llbc = Llbc::load(INTERPRETER_LLBC).expect("pyre-interpreter.ullbc is already extracted");
    let context = LowerContext::new(&llbc);

    let inner = lower_fun_decl(&context, find_method(&llbc, "ticker_addr", "ActionFlag"))
        .expect("lower ActionFlag::ticker_addr");
    assert_returned_call(&inner, "as_ptr");

    let producer = lower_fun_decl(
        &context,
        find_method(&llbc, "ticker_addr", "SpaceActionFlag"),
    )
    .expect("lower SpaceActionFlag::ticker_addr");
    assert_returned_call(&producer, "ticker_addr");
}

#[test]
fn typed_array_base_call_stays_ref() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("pyre-object.ullbc is already extracted");
    let context = LowerContext::new(&llbc);
    for path in [
        "pyre_object::int_array::<Impl>::index",
        "pyre_object::float_array::<Impl>::index",
    ] {
        let graph = lower_fun_decl(&context, find_exact(&llbc, path)).unwrap_or_else(|err| {
            panic!("lower {path}: {err}");
        });
        let bases = call_results_named(&graph, "base");
        assert!(
            !bases.is_empty(),
            "{path} must call base, ops in {}",
            graph.name
        );
        assert!(
            bases.iter().all(|ty| matches!(ty, ValueType::Ref(_))),
            "{path} base() stays the GC block Ref, got {bases:?}"
        );
    }
}

#[test]
fn integer_cast_to_raw_scalar_stays_int() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build/llbc/pyre-module.ullbc"
    ))
    .expect("pyre-module.ullbc is already extracted");
    let context = LowerContext::new(&llbc);
    let graph = lower_fun_decl(&context, find_method(&llbc, "as_ptr", "HashStateStorage"))
        .expect("lower HashStateStorage::as_ptr");
    let leaves: Vec<_> = ops(&graph)
        .filter_map(|op| call_result(&op.kind).and_then(|(_, leaf)| leaf))
        .collect();
    assert!(
        leaves.contains(&"cast_ptr_to_int"),
        "the field address still crosses into the integer, got {leaves:?}"
    );
    assert!(
        !leaves.contains(&"cast_int_to_ptr"),
        "usize as *const usize stays in the int bank, got {leaves:?} in {}",
        graph.name
    );
    match root_op(&graph, &returned_var(&graph)) {
        Some(OpKind::BinOp { result_ty, .. })
            if matches!(result_ty, ValueType::Int | ValueType::Unsigned) => {}
        other => panic!(
            "{} must return the aligned integer word, got {other:?}",
            graph.name
        ),
    }

    let widened = lower_fun_decl(
        &context,
        find_method(&llbc, "as_mut_ptr", "HashStateStorage"),
    )
    .expect("lower HashStateStorage::as_mut_ptr");
    assert_returned_call(&widened, "as_ptr");
}

#[test]
fn sizehint_i64_cast_stays_gcarray_write() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("pyre-object.ullbc is already extracted");
    let graph = lower_function(&llbc, "pyre_object::listobject::set_sizehint_state_value")
        .expect("lower set_sizehint_state_value");
    let raw_stores = ops(&graph)
        .filter(|op| matches!(op.kind, OpKind::RawStore { .. }))
        .count();
    assert_eq!(
        raw_stores, 0,
        "set_sizehint_state_value must keep the GcArray write"
    );
    let ptr_to_int = call_results_named(&graph, "cast_ptr_to_int");
    assert!(
        ptr_to_int.is_empty(),
        "*mut u8 as *mut i64 must not become cast_ptr_to_int, got {ptr_to_int:?}"
    );
    assert!(
        ops(&graph).any(|op| {
            matches!(
                &op.kind,
                OpKind::ArrayWrite {
                    item_ty: ValueType::Int,
                    array_type_id: Some(array_type_id),
                    ..
                } if array_type_id == "[i64]"
            )
        }),
        "set_sizehint_state_value must write GcArray(Signed)"
    );
}

#[test]
fn call_returned_byte_pointer_stays_ref() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("pyre-object.ullbc is already extracted");
    let graph = lower_function(&llbc, "gc_hook::try_gc_alloc_young_nonmoving_raw")
        .expect("lower try_gc_alloc_young_nonmoving_raw");
    let byte_calls: Vec<_> = ops(&graph)
        .filter_map(|op| call_result(&op.kind))
        .filter(|(_, leaf)| *leaf == Some("try_gc_alloc_stable_raw"))
        .map(|(ty, _)| ty.clone())
        .collect();
    assert!(
        !byte_calls.is_empty(),
        "try_gc_alloc_young_nonmoving_raw must call try_gc_alloc_stable_raw, ops in {}",
        graph.name
    );
    assert!(
        byte_calls.iter().all(|ty| matches!(ty, ValueType::Ref(_))),
        "*mut u8 call result stays Ref, got {byte_calls:?} in {}",
        graph.name
    );
}
