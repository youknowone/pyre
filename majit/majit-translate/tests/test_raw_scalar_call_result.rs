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
//!
//! A call that passes `&` / `&mut` of a primitive into a raw
//! scalar-pointer parameter materializes an address first
//! (`RawMalloc` / `RawStore` / `RawFree`). `history.py` `getkind` banks
//! that parameter as `int`, and `Rvalue::Ref` aliases the pointee word.
//! A mutable raw parameter copies the written word back into the
//! borrowed local.

use majit_charon_reader::ullbc::NameSeg;
use majit_charon_reader::{FunDecl, Llbc};
use majit_translate::{
    front::mir::{LowerContext, lower_fun_decl, lower_function},
    model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType},
};
use serde_json::{Value, json};

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

fn i64_ty() -> Value {
    json!({"Scalar": {"Integer": {"Signed": "I64"}}})
}

fn u64_ty() -> Value {
    json!({"Scalar": {"Integer": {"Unsigned": "U64"}}})
}

fn u8_ty() -> Value {
    json!({"Scalar": {"Integer": {"Unsigned": "U8"}}})
}

fn raw_ptr(pointee: &Value, kind: &str) -> Value {
    json!({"RawPtr": [pointee, kind]})
}

fn borrow_ty(pointee: &Value, kind: &str) -> Value {
    json!({"Ref": ["Erased", pointee, kind]})
}

fn place(id: u64, ty: &Value) -> Value {
    json!({"kind": {"Local": id}, "ty": ty})
}

/// How `write_hash` passes local 2 into `sink_pair`.
enum Pass {
    /// `_b = &kind word`, then pass `_b`. `ret_local` is copied into the
    /// return slot after the call (`2` is the word, `1` is `flag`).
    Borrow { kind: &'static str, ret_local: u64 },
    /// Pass local 2 itself.
    Direct { ret_local: u64 },
    /// Copy the word, then pass `&mut` of the original, then return the copy.
    CopyThenMutBorrow,
}

fn lower_probe(word_name: &str, word_ty: &Value, sink_param: &Value, pass: Pass) -> FunctionGraph {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str]| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let ret = i64_ty();
    let flag = i64_ty();
    let local = |index: u64, name: Option<&str>, ty: &Value| json!({"index": index, "name": name, "span": span, "ty": ty});
    let mut locals = vec![
        local(0, None, &ret),
        local(1, Some("flag"), &flag),
        local(2, Some(word_name), word_ty),
    ];
    let (statements, call_args, dest, ret_local) = match pass {
        Pass::Borrow { kind, ret_local } => {
            let borrowed = borrow_ty(word_ty, kind);
            locals.push(local(3, None, &borrowed));
            locals.push(local(4, None, &ret));
            let word = place(2, word_ty);
            let borrow = place(3, &borrowed);
            (
                vec![
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": word, "kind": kind, "ptr_metadata": null
                    }}]}}),
                ],
                vec![json!({"Copy": place(1, &flag)}), json!({"Move": borrow})],
                place(4, &ret),
                ret_local,
            )
        }
        Pass::Direct { ret_local } => {
            locals.push(local(3, None, &ret));
            (
                vec![],
                vec![
                    json!({"Copy": place(1, &flag)}),
                    json!({"Move": place(2, word_ty)}),
                ],
                place(3, &ret),
                ret_local,
            )
        }
        Pass::CopyThenMutBorrow => {
            let borrowed = borrow_ty(word_ty, "Mut");
            locals.push(local(3, None, word_ty));
            locals.push(local(4, None, &borrowed));
            locals.push(local(5, None, &ret));
            let word = place(2, word_ty);
            let borrow = place(4, &borrowed);
            (
                vec![
                    json!({"span": span, "kind": {"Assign": [
                        place(3, word_ty),
                        {"Use": [{"Copy": word.clone()}, "Yes"]}
                    ]}}),
                    json!({"span": span, "kind": {"Assign": [borrow.clone(), {"Ref": {
                        "place": word, "kind": "Mut", "ptr_metadata": null
                    }}]}}),
                ],
                vec![json!({"Copy": place(1, &flag)}), json!({"Move": borrow})],
                place(5, &ret),
                3,
            )
        }
    };
    let ret_place_ty = if ret_local == 1 {
        flag.clone()
    } else {
        word_ty.clone()
    };
    let fun = |id: u64, name: &[&str], inputs: Vec<Value>, body: Value| {
        json!({
            "def_id": id,
            "item_meta": meta(name),
            "signature": {"is_unsafe": false, "inputs": inputs, "output": ret.clone()},
            "body": body
        })
    };
    let caller = fun(
        0,
        &["probe", "write_hash"],
        vec![flag.clone(), word_ty.clone()],
        json!({"Unstructured": {"span": span, "locals": {"arg_count": 2, "locals": locals}, "body": [
            {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                    "args": call_args, "dest": dest},
                "target": 1, "on_unwind": 2
            }}}},
            {"statements": [{"span": span, "kind": {"Assign": [
                place(0, &ret),
                {"Use": [{"Copy": place(ret_local, &ret_place_ty)}, "Yes"]}
            ]}}], "terminator": {"span": span, "kind": "Return"}},
            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
        ]}}),
    );
    let sink = fun(
        1,
        &["probe", "sink_pair"],
        vec![flag, sink_param.clone()],
        json!("Opaque"),
    );
    let file = json!({"charon_version": "0.1.201", "has_errors": false, "translated": {
        "crate_name": "probe", "type_decls": [], "fun_decls": [caller, sink],
        "global_decls": [], "trait_decls": [], "trait_impls": []
    }});
    let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("probe fixture parses");
    lower_function(&llbc, "write_hash").unwrap_or_else(|err| panic!("lower write_hash: {err}"))
}

fn op_lines(graph: &FunctionGraph) -> String {
    ops(graph)
        .map(|op| format!("{:?}", op.kind))
        .collect::<Vec<_>>()
        .join("\n")
}

fn input_var<'a>(
    graph: &'a FunctionGraph,
    name: &str,
) -> &'a majit_translate::flowspace::model::Variable {
    ops(graph)
        .find_map(|op| match &op.kind {
            OpKind::Input { name: got, .. } if got == name => op.result.as_ref(),
            _ => None,
        })
        .unwrap_or_else(|| {
            panic!(
                "missing input {name} in {}\n{}",
                graph.name,
                op_lines(graph)
            )
        })
}

fn sink_call(graph: &FunctionGraph) -> &OpKind {
    let mut found = None;
    for op in ops(graph) {
        let OpKind::Call { target, .. } = &op.kind else {
            continue;
        };
        if call_leaf(target) == Some("sink_pair") {
            assert!(found.is_none(), "two sink_pair calls\n{}", op_lines(graph));
            found = Some(&op.kind);
        }
    }
    found.unwrap_or_else(|| panic!("missing sink_pair\n{}", op_lines(graph)))
}

fn call_arg<'a>(kind: &'a OpKind, index: usize) -> &'a majit_translate::flowspace::model::Variable {
    let OpKind::Call { args, .. } = kind else {
        panic!("not a call");
    };
    args.get(index)
        .and_then(LinkArg::as_variable)
        .unwrap_or_else(|| panic!("call arg {index} is not a variable: {kind:?}"))
}

fn raw_ops<'a>(
    graph: &'a FunctionGraph,
    pred: impl Fn(&OpKind) -> bool,
) -> Vec<&'a SpaceOperation> {
    ops(graph).filter(|op| pred(&op.kind)).collect()
}

fn assert_no_raw_address(graph: &FunctionGraph, label: &str) {
    let spilled = raw_ops(graph, |kind| {
        matches!(
            kind,
            OpKind::RawMalloc { .. }
                | OpKind::RawStore { .. }
                | OpKind::RawLoad { .. }
                | OpKind::RawFree { .. }
        )
    });
    assert!(
        spilled.is_empty(),
        "{label} must pass the word, got\n{}",
        op_lines(graph)
    );
}

fn assert_address_spill(graph: &FunctionGraph, copy_out: bool, return_is_reload: bool) {
    let flag = input_var(graph, "flag");
    let keyhash = input_var(graph, "keyhash");
    let mallocs = raw_ops(graph, |kind| matches!(kind, OpKind::RawMalloc { .. }));
    assert_eq!(
        mallocs.len(),
        1,
        "one raw spill allocation\n{}",
        op_lines(graph)
    );
    let ptr = mallocs[0].result.as_ref().expect("RawMalloc result");
    match &mallocs[0].kind {
        OpKind::RawMalloc { owner, zero: false } if owner == "Tuple<i64>" => {}
        other => panic!("spill owner is Tuple<i64>, got {other:?}"),
    }
    let stores = raw_ops(graph, |kind| matches!(kind, OpKind::RawStore { .. }));
    assert_eq!(
        stores.len(),
        1,
        "one store of the word\n{}",
        op_lines(graph)
    );
    match &stores[0].kind {
        OpKind::RawStore {
            base,
            offset,
            value,
            item_ty: ValueType::Int,
            itemsize: 8,
            is_item_signed: true,
        } if base == ptr && value == keyhash => {
            assert!(
                matches!(defining_op(graph, offset), Some(OpKind::ConstInt(0))),
                "store offset is the zero constant"
            );
        }
        other => panic!("store must write keyhash through the spill, got {other:?}"),
    }
    let call = sink_call(graph);
    assert_eq!(call_arg(call, 0), flag, "flag stays the first argument");
    assert_eq!(
        call_arg(call, 1),
        ptr,
        "the raw parameter receives the address\n{}",
        op_lines(graph)
    );
    let loads = raw_ops(graph, |kind| matches!(kind, OpKind::RawLoad { .. }));
    let frees = raw_ops(graph, |kind| matches!(kind, OpKind::RawFree { .. }));
    assert_eq!(frees.len(), 1, "the spill is freed\n{}", op_lines(graph));
    assert!(
        matches!(&frees[0].kind, OpKind::RawFree { ptr: freed } if freed == ptr),
        "RawFree releases the spill"
    );
    let call_block = graph
        .blocks
        .iter()
        .find(|block| {
            block
                .operations
                .iter()
                .any(|op| std::ptr::eq(&op.kind, call))
        })
        .expect("call block");
    let pos = |pred: &dyn Fn(&OpKind) -> bool| {
        call_block
            .operations
            .iter()
            .position(|op| pred(&op.kind))
            .expect("op in the call block")
    };
    let malloc_at = pos(&|kind| matches!(kind, OpKind::RawMalloc { .. }));
    let store_at = pos(&|kind| matches!(kind, OpKind::RawStore { .. }));
    let call_at = pos(&|kind| std::ptr::eq(kind, call));
    let free_at = pos(&|kind| matches!(kind, OpKind::RawFree { .. }));
    assert!(malloc_at < store_at && store_at < call_at && call_at < free_at);
    if copy_out {
        assert_eq!(
            loads.len(),
            1,
            "mutable out-parameter reloads\n{}",
            op_lines(graph)
        );
        match &loads[0].kind {
            OpKind::RawLoad {
                base,
                offset,
                item_ty: ValueType::Int,
                itemsize: 8,
                is_item_signed: true,
            } if base == ptr => {
                assert!(matches!(
                    defining_op(graph, offset),
                    Some(OpKind::ConstInt(0))
                ));
            }
            other => panic!("reload must read the spill, got {other:?}"),
        }
        let load_at = pos(&|kind| matches!(kind, OpKind::RawLoad { .. }));
        assert!(call_at < load_at && load_at < free_at);
    } else {
        assert!(
            loads.is_empty(),
            "a shared or const parameter is not written back\n{}",
            op_lines(graph)
        );
    }
    let returned = root_op(graph, &returned_var(graph));
    if return_is_reload {
        match returned {
            Some(OpKind::RawLoad {
                base,
                item_ty: ValueType::Int,
                itemsize: 8,
                is_item_signed: true,
                ..
            }) if base == ptr => {}
            other => panic!("the caller must observe the reloaded word, got {other:?}"),
        }
    } else {
        match returned {
            Some(OpKind::Input {
                name,
                ty: ValueType::Int,
                ..
            }) if name == "keyhash" => {}
            other => panic!("the returned word must stay the pre-call input, got {other:?}"),
        }
    }
}

#[test]
fn mut_borrow_of_i64_passed_to_raw_pointer_reloads_the_word() {
    let word = i64_ty();
    let graph = lower_probe(
        "keyhash",
        &word,
        &raw_ptr(&word, "Mut"),
        Pass::Borrow {
            kind: "Mut",
            ret_local: 2,
        },
    );
    assert_address_spill(&graph, true, true);
}

#[test]
fn pre_call_copy_keeps_the_word_the_borrow_reloads() {
    let word = i64_ty();
    let graph = lower_probe(
        "keyhash",
        &word,
        &raw_ptr(&word, "Mut"),
        Pass::CopyThenMutBorrow,
    );
    assert_address_spill(&graph, true, false);
}

#[test]
fn shared_borrow_of_i64_into_const_pointer_is_not_copied_out() {
    let word = i64_ty();
    let graph = lower_probe(
        "keyhash",
        &word,
        &raw_ptr(&word, "Const"),
        Pass::Borrow {
            kind: "Shared",
            ret_local: 2,
        },
    );
    assert_address_spill(&graph, false, false);
}

#[test]
fn raw_scalar_address_spill_leaves_other_arguments_as_the_word() {
    let i64_ty = i64_ty();
    let cases: [(&str, &str, Value, Value, Pass); 5] = [
        (
            "&mut i64 parameter",
            "keyhash",
            i64_ty.clone(),
            borrow_ty(&i64_ty, "Mut"),
            Pass::Borrow {
                kind: "Mut",
                ret_local: 2,
            },
        ),
        (
            "already raw *mut i64",
            "keyhash",
            raw_ptr(&i64_ty, "Mut"),
            raw_ptr(&i64_ty, "Mut"),
            Pass::Direct { ret_local: 2 },
        ),
        (
            "&mut u8 into *mut u8",
            "byte",
            u8_ty(),
            raw_ptr(&u8_ty(), "Mut"),
            Pass::Borrow {
                kind: "Mut",
                ret_local: 1,
            },
        ),
        (
            "i64 passed by value",
            "keyhash",
            i64_ty.clone(),
            i64_ty.clone(),
            Pass::Direct { ret_local: 2 },
        ),
        (
            "&mut i64 into *mut u64",
            "keyhash",
            i64_ty.clone(),
            raw_ptr(&u64_ty(), "Mut"),
            Pass::Borrow {
                kind: "Mut",
                ret_local: 2,
            },
        ),
    ];
    for (label, word_name, word_ty, sink_param, pass) in cases {
        let graph = lower_probe(word_name, &word_ty, &sink_param, pass);
        assert_no_raw_address(&graph, label);
        let call = sink_call(&graph);
        assert_eq!(
            call_arg(call, 0),
            input_var(&graph, "flag"),
            "{label}: flag"
        );
        assert_eq!(
            call_arg(call, 1),
            input_var(&graph, word_name),
            "{label}: the argument stays the word\n{}",
            op_lines(&graph)
        );
    }
}
