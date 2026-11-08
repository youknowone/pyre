//! `w_dict_lookup`'s `DictStrategy::getitem` reads `method_getitem` from
//! the vtable word of `DictStrategyRef.imp`, not from the data pointer.

mod common;

use std::collections::HashSet;

use common::{
    OBJECT_LLBC, build_semantic_program_from_llbcs_with_static_addrs_and_function_names, load_llbc,
    lower_function, with_shared_lowering,
};
use majit_charon_reader::Llbc;
use majit_translate::HostStaticAddrs;
use majit_translate::call::CallControl;
use majit_translate::codewriter::jtransform::{GraphTransformConfig, Transformer};
use majit_translate::flowspace::model::Variable;
use majit_translate::model::{
    BlockId, CallFuncPtr, CallTarget, ConcreteType, FunctionGraph, OpKind, SpaceOperation,
    ValueType, VecFieldPart,
};

fn object_llbc() -> &'static Llbc {
    load_llbc(OBJECT_LLBC)
}

fn assert_vtable_reads(graph: &FunctionGraph, method_name: &str) {
    let mut reads = 0usize;
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::FieldRead { base, field, .. } = &op.kind else {
                continue;
            };
            if field.name != method_name {
                continue;
            }
            reads += 1;
            let mut seen = HashSet::new();
            assert!(
                reaches_imp_fat_len(graph, block.id, base, &mut seen),
                "{method_name} base does not reach FatLen of imp\n{}",
                graph.dump()
            );
            assert_eq!(
                FunctionGraph::concretetype_of(base),
                ConcreteType::Signed,
                "vtable word must be an int so getfield_raw_r loads it"
            );
        }
    }
    assert!(reads > 0, "lowered no {method_name} read\n{}", graph.dump());
    let still_calls_strategy = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            let OpKind::Call { target, .. } = &op.kind else {
                return false;
            };
            call_names_get_strategy(target)
        })
    });
    assert!(
        !still_calls_strategy,
        "w_dict_get_strategy stayed a call, so its vtable word was dropped\n{}",
        graph.dump()
    );
    let fat_vtable_left = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| match &op.kind {
            OpKind::FieldRead { field, .. } => {
                field.name == "imp" && field.vec_part == Some(VecFieldPart::FatVtable)
            }
            _ => false,
        })
    });
    assert!(
        !fat_vtable_left,
        "imp.vtable GetfieldGcR next to imp.len GetfieldGcI trips make_equal_to Box.type\n{}",
        graph.dump()
    );
    // The IndirectCall receiver is the fat DATA word (`keep_dyn_call_receiver`),
    // not a cast of the vtable word. Unused `cast_int_to_ptr` of FatLen
    // is dropped by `drop_unused_fat_len_casts`.
}

fn is_cast_int_to_ptr(kind: &OpKind) -> bool {
    match kind {
        OpKind::UnaryOp { op, .. } if op == "cast_int_to_ptr" => true,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } if segments.last().map(|s| s.as_str()) == Some("cast_int_to_ptr") => true,
        _ => false,
    }
}

#[test]
fn w_dict_lookup_reads_getitem_from_the_vtable_word() {
    let llbc = object_llbc();
    let graph = lower_function(llbc, "w_dict_lookup").expect("lower w_dict_lookup");
    assert_vtable_reads(&graph, "method_getitem");
}

fn call_names_get_strategy(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => segments
            .iter()
            .any(|segment| segment == "w_dict_get_strategy"),
        CallTarget::Method { name, .. } => name == "w_dict_get_strategy",
        _ => false,
    }
}

fn reaches_imp_fat_len(
    graph: &FunctionGraph,
    block: BlockId,
    var: &Variable,
    seen: &mut HashSet<(usize, u64)>,
) -> bool {
    if !seen.insert((block.0, var.id())) {
        return false;
    }
    if let Some((def_block, op)) = producer(graph, var.id()) {
        return match &op.kind {
            OpKind::FieldRead { field, ty, .. }
                if field.name == "imp"
                    && matches!(
                        field.vec_part,
                        Some(VecFieldPart::FatLen | VecFieldPart::FatMeta)
                    )
                    && *ty == ValueType::Int
                    && FunctionGraph::concretetype_of(var) == ConcreteType::Signed =>
            {
                true
            }
            OpKind::UnaryOp { op, operand, .. } if op == "same_as" => {
                reaches_imp_fat_len(graph, def_block, operand, seen)
            }
            _ => false,
        };
    }
    let Some(block_ref) = graph.blocks.get(block.0) else {
        return false;
    };
    let preds = graph.predecessors(block);
    if preds.is_empty() {
        return false;
    }
    let index = block_ref.inputargs.iter().position(|arg| arg == var);
    preds.iter().all(|pred| {
        let Some(pred_block) = graph.blocks.get(pred.0) else {
            return false;
        };
        let incoming: Vec<_> = pred_block
            .exits
            .iter()
            .filter(|link| link.target == block)
            .collect();
        if incoming.is_empty() {
            return false;
        }
        incoming.iter().all(|link| {
            let src = if let Some(index) = index {
                link.args.get(index).and_then(|arg| arg.as_variable())
            } else {
                // Dominating use: the method_* block reads the metadata
                // word without taking it as an inputarg.
                Some(var)
            };
            src.is_some_and(|src| reaches_imp_fat_len(graph, *pred, src, seen))
        })
    })
}

#[test]
fn w_dict_getitem_str_hashed_assembles_with_vtable_word() {
    let llbc = object_llbc();
    let graph =
        lower_function(llbc, "w_dict_getitem_str_hashed").expect("lower w_dict_getitem_str_hashed");
    assert_vtable_reads(&graph, "method_getitem_str_hashed");
    // `join_blocks` renames the indirect call's `hash_` onto the link arg.
    // That arg has to stay defined after the fat-dyn splice, or liveness
    // panics while assembling this graph.
    let segments: Vec<String> = graph.name.split("::").map(str::to_string).collect();
    let path = majit_translate::CallPath::from_segments(segments);
    let mut callcontrol = majit_translate::codewriter::call::CallControl::new();
    callcontrol.set_struct_fields(with_shared_lowering(llbc, |ctx| {
        ctx.struct_field_registry()
    }));
    callcontrol.register_function_graph(path.clone(), graph.clone());
    let mut codewriter = majit_translate::codewriter::codewriter::CodeWriter::new();
    let jitcode = std::sync::Arc::new(majit_translate::codewriter::jitcode::JitCode::new(
        "w_dict_getitem_str_hashed",
    ));
    let cfg = majit_translate::GraphTransformConfig::default();
    codewriter.transform_graph_to_jitcode(
        &graph,
        &path,
        &mut callcontrol,
        &cfg,
        &jitcode,
        false,
        0,
    );
}

fn has_cast_int_to_ptr(graph: &FunctionGraph) -> bool {
    graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| is_cast_int_to_ptr(&op.kind))
    })
}

#[test]
fn w_dict_store_has_no_cast_int_to_ptr_of_vtable_word() {
    let llbc = object_llbc();
    let graph = lower_function(llbc, "w_dict_store").expect("lower w_dict_store");
    assert_vtable_reads(&graph, "method_setitem");
    assert!(
        !has_cast_int_to_ptr(&graph),
        "leftover FatVtable users are the DATA word (Ref), not CastIntToPtr of FatMeta\n{}",
        graph.dump()
    );
}

fn call_leaf_is(target: &CallTarget, leaf: &str) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            segments.last().is_some_and(|name| name == leaf)
        }
        _ => false,
    }
}

#[test]
fn eq_w_compares_utf8_fields_with_ll_streq() {
    let llbc = object_llbc();
    let graph = lower_function(llbc, "eq_w").expect("lower eq_w");
    assert!(
        graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(&op.kind, OpKind::Call { target, .. } if call_leaf_is(target, "jit_ll_streq"))
            })
        }),
        "eq_w must call ll_streq on the two `_utf8` STRs \
         (unicodeobject.py eq_w: self._utf8 == w_other._utf8)\n{}",
        graph.dump()
    );
    let residual_payload = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            let OpKind::Call { target, .. } = &op.kind else {
                return false;
            };
            call_leaf_is(target, "utf8_payload_bytes")
        })
    });
    assert!(
        !residual_payload,
        "eq_w must not go through utf8_payload_bytes; the compare is ll_streq \
         of the two value fields\n{}",
        graph.dump()
    );
}

fn value_type_kind(ty: &ValueType) -> Option<char> {
    match ty {
        ValueType::Int
        | ValueType::Unsigned
        | ValueType::Int128
        | ValueType::UInt128
        | ValueType::SingleFloat
        | ValueType::Bool => Some('i'),
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => Some('r'),
        ValueType::Float => Some('f'),
        ValueType::Void | ValueType::State | ValueType::Unknown => None,
    }
}

fn concrete_kind(ty: ConcreteType) -> Option<char> {
    match ty {
        ConcreteType::Signed => Some('i'),
        ConcreteType::GcRef => Some('r'),
        ConcreteType::Float => Some('f'),
        ConcreteType::Void => None,
        ConcreteType::Unknown => None,
    }
}

fn var_kind(graph: &FunctionGraph, block: BlockId, var: &Variable) -> Option<char> {
    var_kind_from(graph, block, var, &mut HashSet::new())
}

fn var_kind_from(
    graph: &FunctionGraph,
    block: BlockId,
    var: &Variable,
    seen: &mut HashSet<(usize, u64)>,
) -> Option<char> {
    if !seen.insert((block.0, var.id())) {
        return None;
    }
    if let Some(kind) = concrete_kind(FunctionGraph::concretetype_of(var)) {
        return Some(kind);
    }
    if let Some((def_block, op)) = producer(graph, var.id()) {
        return match &op.kind {
            OpKind::FieldRead { ty, field, .. } => {
                if field.vec_part == Some(VecFieldPart::FatData) && *ty == ValueType::Int {
                    Some('i')
                } else {
                    value_type_kind(ty)
                }
            }
            OpKind::ConstInt(_) | OpKind::ConstUInt(_) => Some('i'),
            OpKind::ConstRefAddr(_) => Some('r'),
            OpKind::Input { ty, .. } => value_type_kind(ty),
            OpKind::UnaryOp { operand, .. } => var_kind_from(graph, def_block, operand, seen),
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if segments.last().map(|s| s.as_str()) == Some("cast_int_to_ptr")
                || segments.last().map(|s| s.as_str()) == Some("same_as") =>
            {
                args.first()
                    .and_then(|arg| arg.as_variable())
                    .and_then(|src| var_kind_from(graph, def_block, src, seen))
            }
            _ => None,
        };
    }
    let block_ref = graph.blocks.get(block.0)?;
    let index = block_ref.inputargs.iter().position(|arg| arg == var)?;
    let mut kinds = Vec::new();
    for pred in graph.predecessors(block) {
        let pred_block = graph.blocks.get(pred.0)?;
        for link in pred_block.exits.iter().filter(|link| link.target == block) {
            let src = link.args.get(index)?.as_variable()?;
            kinds.push(var_kind_from(graph, pred, src, seen)?);
        }
    }
    let first = *kinds.first()?;
    kinds.iter().all(|kind| *kind == first).then_some(first)
}

fn first_input_kind(graph: &FunctionGraph) -> Option<(char, ValueType)> {
    let block = graph.blocks.get(graph.startblock.0)?;
    for op in &block.operations {
        let OpKind::Input { ty, .. } = &op.kind else {
            continue;
        };
        if matches!(ty, ValueType::Void) {
            continue;
        }
        let kind = value_type_kind(ty).or_else(|| {
            op.result
                .as_ref()
                .and_then(|var| var_kind(graph, graph.startblock, var))
        })?;
        return Some((kind, ty.clone()));
    }
    None
}

fn first_indirect_call(
    graph: &FunctionGraph,
) -> Option<(BlockId, &[Variable], Option<&(String, String)>)> {
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::IndirectCall {
                args, family_key, ..
            } = &op.kind
            {
                return Some((block.id, args.as_slice(), family_key.as_ref()));
            }
        }
    }
    None
}

/// Dyn `DictStrategy::length` passes the fat DATA word as Ref
/// (RPython `DictStrategy` instance). The ZST impl keeps that same
/// first inputarg so the `indirect_call` family is one FUNC type.
#[test]
fn w_dict_len_indirect_call_receiver_kind_matches_zst_length_self() {
    let llbc = object_llbc();
    let caller = lower_function(llbc, "w_dict_len").expect("lower w_dict_len");
    let callee = lower_function(llbc, "kwargsdict::<Impl>::length")
        .expect("lower kwargsdict::<Impl>::length");
    if let Ok(path) = std::env::var("DUMP_W_DICT_LEN") {
        std::fs::write(format!("{path}.caller"), caller.dump()).expect("write w_dict_len dump");
        std::fs::write(format!("{path}.callee"), callee.dump())
            .expect("write kwargsdict length dump");
    }
    let (call_block, call_args, family) = first_indirect_call(&caller).unwrap_or_else(|| {
        panic!(
            "w_dict_len must emit IndirectCall for DictStrategy::length\n{}",
            caller.dump()
        )
    });
    assert!(
        !has_cast_int_to_ptr(&caller),
        "leftover FatVtable users are the DATA word (Ref), not CastIntToPtr of FatMeta\n{}",
        caller.dump()
    );
    assert!(
        family.is_some_and(|(trait_root, method)| {
            trait_root.contains("DictStrategy") && method == "length"
        }),
        "IndirectCall family must be DictStrategy::length, got {family:?}\n{}",
        caller.dump()
    );
    let recv = call_args
        .first()
        .unwrap_or_else(|| panic!("dyn length call must pass a receiver\n{}", caller.dump()));
    assert_ne!(
        FunctionGraph::concretetype_of(recv),
        ConcreteType::Void,
        "dyn call receiver must not be Void\n{}",
        caller.dump()
    );
    let recv_kind = var_kind(&caller, call_block, recv).unwrap_or_else(|| {
        panic!(
            "dyn call receiver has no kind (var={recv:?})\n{}",
            caller.dump()
        )
    });
    let (callee_kind, callee_ty) = first_input_kind(&callee).unwrap_or_else(|| {
        panic!(
            "kwargsdict::<Impl>::length must keep a non-Void self input\n{}",
            callee.dump()
        )
    });
    assert_eq!(
        recv_kind,
        callee_kind,
        "caller receiver kind {recv_kind} != callee first inputarg kind {callee_kind} ({callee_ty:?})\ncaller:\n{}\ncallee:\n{}",
        caller.dump(),
        callee.dump()
    );
    assert_eq!(
        recv_kind, 'r',
        "DictStrategy self is a GC instance (Ref); mixed ZST+GC family follows that kind, got {recv_kind}"
    );

    let caller_segments: Vec<String> = caller.name.split("::").map(str::to_string).collect();
    let callee_segments: Vec<String> = callee.name.split("::").map(str::to_string).collect();
    let caller_path = majit_translate::CallPath::from_segments(caller_segments);
    let callee_path = majit_translate::CallPath::from_segments(callee_segments);
    let mut callcontrol = majit_translate::codewriter::call::CallControl::new();
    callcontrol.set_struct_fields(with_shared_lowering(llbc, |ctx| {
        ctx.struct_field_registry()
    }));
    callcontrol.register_function_graph(caller_path.clone(), caller.clone());
    callcontrol.register_function_graph(callee_path.clone(), callee.clone());
    let cfg = majit_translate::GraphTransformConfig::default();
    let mut codewriter = majit_translate::codewriter::codewriter::CodeWriter::new();
    let caller_jit = std::sync::Arc::new(majit_translate::codewriter::jitcode::JitCode::new(
        "w_dict_len",
    ));
    let callee_jit = std::sync::Arc::new(majit_translate::codewriter::jitcode::JitCode::new(
        "kwargsdict_length",
    ));
    codewriter.transform_graph_to_jitcode(
        &caller,
        &caller_path,
        &mut callcontrol,
        &cfg,
        &caller_jit,
        false,
        0,
    );
    codewriter.transform_graph_to_jitcode(
        &callee,
        &callee_path,
        &mut callcontrol,
        &cfg,
        &callee_jit,
        false,
        0,
    );
    let callee_args = &callee_jit.calldescr().arg_classes;
    assert!(
        callee_args.starts_with('r'),
        "ZST length impl arg_classes must start with r (DictStrategy instance self), got {callee_args:?}\n{}",
        callee.dump()
    );
    if let Ok(path) = std::env::var("DUMP_W_DICT_LEN") {
        std::fs::write(
            format!("{path}.arg_classes"),
            format!(
                "w_dict_len recv_kind={recv_kind}\nkwargsdict::<Impl>::length arg_classes={callee_args}\n"
            ),
        )
        .expect("write arg_classes dump");
    }
}

fn producer(graph: &FunctionGraph, var_id: u64) -> Option<(BlockId, &SpaceOperation)> {
    graph.blocks.iter().find_map(|block| {
        block
            .operations
            .iter()
            .find(|op| {
                op.result
                    .as_ref()
                    .is_some_and(|result| result.id() == var_id)
            })
            .map(|op| (block.id, op))
    })
}

/// Trait-impl graph whose `FUNC.RESULT` is `return_type`, with a Ref
/// receiver plus `extra_args` Ref formals.
fn trait_method_graph(name: &str, extra_args: usize, return_type: &str) -> FunctionGraph {
    let mut graph = FunctionGraph::new(name);
    graph
        .push_op_var(
            graph.startblock,
            OpKind::Input {
                name: "self".into(),
                ty: ValueType::Ref(None),
                class_root: None,
            },
            true,
        )
        .unwrap();
    for i in 0..extra_args {
        graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: format!("a{i}"),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
    }
    graph.set_return(graph.startblock, None);
    graph.with_return_type(return_type)
}

/// Dyn-vtable residual of a trait method whose caller still carries the
/// front-end unit/GC-reference `Ref` shell. `family_key` is cleared the
/// way `lower_indirect_calls` leaves it; `graphs` is the family witness.
fn transform_indirect_vtable_residual_graph(
    trait_root: &str,
    method: &str,
    impl_type: &str,
    extra_args: usize,
    callee: FunctionGraph,
) -> (char, char, usize) {
    let mut cc = CallControl::new();
    cc.register_trait_method(method, Some(trait_root), impl_type, callee);
    let family = cc.all_impls_for_indirect(trait_root, method);
    assert_eq!(family.len(), 1);

    let mut graph = FunctionGraph::new("caller");
    let bb = graph.startblock;
    let funcptr = graph.alloc_value_var_with_type(ConcreteType::Signed);
    let args: Vec<_> = (0..=extra_args)
        .map(|_| graph.alloc_value_var_with_type(ConcreteType::GcRef))
        .collect();
    let result = graph.alloc_value_var_with_type(ConcreteType::GcRef);
    graph.block_mut(bb).operations.push(SpaceOperation {
        result: Some(result),
        kind: OpKind::IndirectCall {
            funcptr,
            args,
            graphs: Some(family),
            family_key: None,
            result_ty: ValueType::Ref(None),
        },
    });
    graph.set_return(bb, None);

    let config = GraphTransformConfig::default();
    let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
    let transformed = transformer.transform(&graph);
    transformed
        .graph
        .block(bb)
        .operations
        .iter()
        .find_map(|op| match &op.kind {
            OpKind::CallResidual {
                funcptr: CallFuncPtr::Value(_),
                descriptor,
                result_kind,
                indirect_targets: None,
                ..
            } => Some((*result_kind, descriptor.result_type, descriptor.result_size)),
            _ => None,
        })
        .expect("indirect call lowered to a residual")
}

fn transform_indirect_vtable_residual(
    method: &str,
    extra_args: usize,
    return_type: &str,
) -> (char, char, usize) {
    transform_indirect_vtable_residual_graph(
        "Store",
        method,
        "IdentityStore",
        extra_args,
        trait_method_graph("IdentityStore::method", extra_args, return_type),
    )
}

/// `fn(&self, ...) -> ()` trait method: the unit `()` shell is Ref on the
/// call, `FUNC.RESULT` is Void, residual is `_v` with `result_size` 0.
#[test]
fn indirect_unit_trait_method_residual_is_void() {
    let (result_kind, result_type, result_size) =
        transform_indirect_vtable_residual("store", 3, "()");
    assert_eq!(result_kind, 'v');
    assert_eq!(result_type, 'v');
    assert_eq!(result_size, 0);
}

/// A trait method returning a GC reference (`ref` token) stays a Ref
/// residual.
#[test]
fn indirect_gcref_trait_method_residual_stays_ref() {
    let (result_kind, result_type, result_size) =
        transform_indirect_vtable_residual("load", 2, "ref");
    assert_eq!(result_kind, 'r');
    assert_eq!(result_type, 'r');
    assert_eq!(result_size, 8);
}

/// MIR-built `IdentityDictStrategy::setitem(&self, ...) -> ()` declares
/// Void `FUNC.RESULT`; an indirect vtable residual through that graph is
/// `_v` with `result_size` 0.
#[test]
fn mir_identity_dict_setitem_declares_void_and_indirect_residual_is_void() {
    let llbc = object_llbc();
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        std::slice::from_ref(llbc),
        HostStaticAddrs::default(),
        &["identitydict"],
        &["setitem"],
    )
    .expect("build DictStrategy::setitem graphs");
    let setitems: Vec<_> = program
        .functions
        .iter()
        .map(|function| {
            (
                function.name.as_str(),
                function.self_ty_root.as_deref(),
                function.trait_root.as_deref(),
                function.return_type.clone(),
            )
        })
        .collect();
    let func = program
        .functions
        .iter()
        .find(|function| {
            function.name == "setitem"
                && function.trait_root.as_deref() == Some("DictStrategy")
                && function
                    .self_ty_root
                    .as_deref()
                    .is_some_and(|owner| owner.ends_with("IdentityDictStrategy"))
        })
        .unwrap_or_else(|| panic!("IdentityDictStrategy::setitem not in {setitems:?}"));
    assert_eq!(
        func.return_type.as_deref(),
        Some("()"),
        "header FUNC.RESULT: {:?}",
        func.return_type
    );
    assert_eq!(
        func.graph().return_type.as_deref(),
        Some("()"),
        "graph FUNC.RESULT: {:?}",
        func.graph().return_type
    );
    let (result_kind, result_type, result_size) = transform_indirect_vtable_residual_graph(
        "DictStrategy",
        "setitem",
        "IdentityDictStrategy",
        3,
        func.graph().clone(),
    );
    assert_eq!(result_kind, 'v');
    assert_eq!(result_type, 'v');
    assert_eq!(result_size, 0);
}

fn call_ops_with_leaf<'a>(
    graph: &'a FunctionGraph,
    leaf: &str,
) -> Vec<(
    &'a [majit_translate::model::LinkArg],
    &'a ValueType,
    BlockId,
)> {
    let mut out = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call {
                target,
                args,
                result_ty,
            } = &op.kind
            else {
                continue;
            };
            if call_leaf_is(target, leaf) {
                out.push((args.as_slice(), result_ty, block.id));
            }
        }
    }
    out
}

/// Checkpoint B: ordinary add is the vtable `indirect_call`
/// (`jtransform.py handle_regular_indirect_call`). remove/has_key keep
/// the pointer-to-ObjectKey trampolines.
#[test]
fn w_set_add_checked_is_set_strategy_add_indirect_call() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let graph = lower_function(llbc, "w_set_add_checked").expect("lower w_set_add_checked");
    let dump = graph.dump();
    assert!(
        call_ops_with_leaf(&graph, "set_strategy_add_ptr").is_empty(),
        "w_set_add_checked must not residualize through set_strategy_add_ptr\n{dump}"
    );
    let old_add_leaf = concat!("set_strategy_add_", "key_ptr");
    assert!(
        call_ops_with_leaf(&graph, old_add_leaf).is_empty(),
        "old pointer-to-ObjectKey add leaf must not remain\n{dump}"
    );
    assert!(
        !dump.contains("__dyn_call"),
        "known SetStrategy::add family must not use __dyn_call\n{dump}"
    );

    let (call_block, call_args, family) = first_indirect_call(&graph).unwrap_or_else(|| {
        panic!("w_set_add_checked must emit IndirectCall for SetStrategy::add\n{dump}")
    });
    assert!(
        family.is_some_and(|(trait_root, method)| {
            trait_root.contains("SetStrategy") && method == "add"
        }),
        "IndirectCall family must be SetStrategy::add, got {family:?}\n{dump}"
    );
    assert!(
        call_args.len() >= 3,
        "SetStrategy::add keeps self plus (w_set, w_key), got {} args\n{dump}",
        call_args.len()
    );
    let recv = call_args
        .first()
        .unwrap_or_else(|| panic!("dyn add call must pass a receiver\n{dump}"));
    assert_ne!(
        FunctionGraph::concretetype_of(recv),
        ConcreteType::Void,
        "dyn call receiver must not be Void\n{dump}"
    );
    let recv_kind = var_kind(&graph, call_block, recv)
        .unwrap_or_else(|| panic!("dyn add receiver has no kind (var={recv:?})\n{dump}"));
    assert_eq!(
        recv_kind, 'r',
        "SetStrategy self is a GC instance (Ref) even for ZST impls, got {recv_kind}"
    );

    let contains = lower_function(llbc, "w_set_contains_key_checked")
        .expect("lower w_set_contains_key_checked");
    assert!(
        !call_ops_with_leaf(&contains, "set_strategy_has_key_ptr").is_empty(),
        "has_key keeps set_strategy_has_key_ptr\n{}",
        contains.dump()
    );
    let discard =
        lower_function(llbc, "w_set_discard_key_checked").expect("lower w_set_discard_key_checked");
    assert!(
        !call_ops_with_leaf(&discard, "set_strategy_remove_key_ptr").is_empty(),
        "remove keeps set_strategy_remove_key_ptr\n{}",
        discard.dump()
    );
}

/// `lib.rs` family registration of `(self_ty_root, trait_root)` plus the
/// default-body arm for a generic `Self`. Mirrors production so a missing
/// unwrapped member is a discovery bug, not a test-only substitution.
fn register_production_set_strategy_add_family(
    cc: &mut CallControl,
    program: &majit_translate::SemanticProgram,
) -> Vec<String> {
    let mut members = Vec::new();
    for func in &program.functions {
        let Some(trait_leaf) = func.trait_root.as_deref() else {
            continue;
        };
        if trait_leaf != "SetStrategy" {
            continue;
        }
        let leaf = func.name.rsplit("::").next().unwrap_or(func.name.as_str());
        if leaf != "add" && !leaf.starts_with("add__spec_") {
            continue;
        }
        match func.self_ty_root.as_deref() {
            Some(owner) => {
                let path = match func.trait_impl_identity.as_deref() {
                    Some(identity) => {
                        let mut segments: Vec<String> = owner
                            .split("::")
                            .filter(|seg| !seg.is_empty())
                            .map(str::to_string)
                            .collect();
                        segments.push(format!("<impl {identity}>"));
                        segments.push(func.name.clone());
                        majit_translate::CallPath::from_segments(segments)
                    }
                    None => majit_translate::CallPath::for_impl_method(owner, &func.name),
                };
                cc.register_trait_family_path(&func.name, trait_leaf, path.clone());
                cc.register_function_graph(path.clone(), func.graph().clone());
                members.push(format!(
                    "{path} (owner={owner}, identity={:?}, return={:?})",
                    func.trait_impl_identity, func.return_type
                ));
            }
            None => {
                let impl_type = format!("<default methods of {trait_leaf}>");
                cc.register_trait_method(
                    &func.name,
                    Some(trait_leaf),
                    &impl_type,
                    func.graph().clone(),
                );
                members.push(format!(
                    "{impl_type}::{} (generic Self, identity={:?}, return={:?})",
                    func.name, func.trait_impl_identity, func.return_type
                ));
            }
        }
    }
    members
}

const SET_STRATEGY_ADD_IMPLS: [&str; 6] = [
    "EmptySetStrategy",
    "ObjectSetStrategy",
    "IntegerSetStrategy",
    "BytesSetStrategy",
    "AsciiSetStrategy",
    "IdentitySetStrategy",
];

fn family_member_names(family: &[majit_translate::CallPath]) -> Vec<String> {
    family.iter().map(|path| path.to_string()).collect()
}

fn family_covers_impl(family: &[majit_translate::CallPath], impl_leaf: &str) -> bool {
    family.iter().any(|path| {
        path.segments
            .iter()
            .any(|segment| segment == impl_leaf || segment.contains(impl_leaf))
    })
}

/// `rpbc.rs lower_indirect_calls_with` fills `(SetStrategy, add)` with
/// Empty, Object, and the four unwrapped monomorphized impl graphs.
/// `jtransform.rs Transformer::lower_indirect_call_op` then emits
/// `int_guard_value` plus `IndirectCallTargets` (`guess_call_kind` Regular).
#[test]
fn set_strategy_add_family_has_six_impls_and_regular_ict() {
    let Some(llbc) = object_llbc() else {
        return;
    };
    let mut graph = lower_function(llbc, "w_set_add_checked").expect("lower w_set_add_checked");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        std::slice::from_ref(llbc),
        HostStaticAddrs::default(),
        &["setobject"],
        &["add", "w_set_add_checked"],
    )
    .expect("build SetStrategy::add graphs");

    let mut cc = CallControl::new();
    let registered = register_production_set_strategy_add_family(&mut cc, &program);
    let family = cc.all_impls_for_indirect("SetStrategy", "add");
    let names = family_member_names(&family);
    let missing: Vec<&str> = SET_STRATEGY_ADD_IMPLS
        .iter()
        .copied()
        .filter(|impl_leaf| !family_covers_impl(&family, impl_leaf))
        .collect();
    assert!(
        missing.is_empty() && family.len() >= 6,
        "SetStrategy::add family must include Empty, Object, and the four unwrapped impls; \
         missing={missing:?} family={names:?} registered={registered:?}"
    );

    majit_translate::translator::rtyper::rpbc::lower_indirect_calls(&mut graph, &cc);
    let filled = graph.blocks.iter().find_map(|block| {
        block.operations.iter().find_map(|op| match &op.kind {
            OpKind::IndirectCall {
                graphs, family_key, ..
            } => Some((graphs.clone(), family_key.clone())),
            _ => None,
        })
    });
    let (graphs, family_key) = filled.unwrap_or_else(|| {
        panic!(
            "lower_indirect_calls must keep the SetStrategy::add IndirectCall\n{}",
            graph.dump()
        )
    });
    assert!(
        family_key.is_none(),
        "rtyper must consume the family key, got {family_key:?}"
    );
    let graphs = graphs.unwrap_or_else(|| {
        panic!("known SetStrategy::add family must fill graphs, not collapse to unknown")
    });
    assert!(
        graphs.len() >= 6,
        "filled graphs must keep all six impls, got {}",
        graphs.len()
    );

    let caller_path = majit_translate::CallPath::from_segments(["w_set_add_checked"]);
    cc.register_function_graph(caller_path.clone(), graph.clone());
    cc.find_helper_graphs(
        &mut majit_translate::policy::DefaultJitPolicy::new(),
        &[caller_path],
    );
    let config = GraphTransformConfig::default();
    let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
    let transformed = transformer.transform(&graph);
    let has_guard = transformed.graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| matches!(&op.kind, OpKind::GuardValue { kind_char: 'i', .. }))
    });
    let ict_len = transformed.graph.blocks.iter().find_map(|block| {
        block.operations.iter().find_map(|op| match &op.kind {
            OpKind::CallResidual {
                indirect_targets: Some(targets),
                ..
            } => Some(targets.lst.len()),
            _ => None,
        })
    });
    assert!(
        has_guard,
        "regular indirect add must emit int_guard_value\n{}",
        transformed.graph.dump()
    );
    let ict_len = ict_len.unwrap_or_else(|| {
        panic!(
            "regular indirect add must emit IndirectCallTargets\n{}",
            transformed.graph.dump()
        )
    });
    assert!(
        ict_len >= 6,
        "IndirectCallTargets must list all six impl jitcodes, got {ict_len}"
    );
}
