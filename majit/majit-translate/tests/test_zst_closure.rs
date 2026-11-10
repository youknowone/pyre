//! A closure that captures nothing is a layout-size-0 ADT. Its low-level
//! repr is `FunctionRepr` Void (`rpbc.py`); the specialized `CallPath`
//! names the body. Capturing envs stay a `Ref` value.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_function;
use majit_translate::model::{CallTarget, ConcreteType, FunctionGraph, LinkArg, OpKind};
use serde_json::{Value, json};

fn zst_closure_and_then() -> Llbc {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: Vec<Value>| {
        json!({
            "name": path,
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let ident = |name: &str| json!({"Ident": [name, 0]});
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let adt = |id| json!({"Adt": {"id": id, "generics": generics}});
    let option_ty = json!({"Adt": {
        "id": 0,
        "generics": {"regions": [], "types": [{"Scalar": {"Integer": {"Signed": "I64"}}}], "const_generics": [], "trait_refs": []}
    }});
    let word = json!({"Scalar": {"Integer": {"Signed": "I64"}}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let closure_ty = adt(1);
    let layout = json!([{
        "key": "fixture-target",
        "value": {"size": 0, "align": 1, "variant_layouts": [{"field_offsets": []}], "repr": {"transparent": false}}
    }]);
    let impl_seg = json!({"Impl": {"Ty": {
        "params": {"regions": [], "types": [], "const_generics": [], "trait_clauses": [], "regions_outlive": [], "types_outlive": [], "trait_type_constraints": []},
        "skip_binder": {"Value": [0, {"Adt": {"id": 0, "generics": generics}}]},
        "kind": "InherentImplBlock"
    }}});
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(vec![ident("core"), ident("option"), ident("Option")]),
                 "kind": {"Enum": [
                    {"name": "None", "fields": [], "discriminant": {"Scalar": {"Unsigned": ["U8", "0"]}}},
                    {"name": "Some", "fields": [{"name": null, "ty": word, "attr_info": {"attributes": [], "inline": null, "rename": null, "public": false}}],
                     "discriminant": {"Scalar": {"Unsigned": ["U8", "1"]}}}
                 ]}},
                {"def_id": 1, "item_meta": meta(vec![ident("fixture"), ident("use_and_then"), ident("closure")]),
                 "kind": {"Struct": []},
                 "layout": layout,
                 "src": {"Closure": {"info": {"kind": "FnOnce"}}}}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(vec![ident("fixture"), ident("use_and_then")]),
                 "signature": {"is_unsafe": false, "inputs": [option_ty], "output": option_ty},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": option_ty},
                        {"index": 1, "name": "opt", "span": span, "ty": option_ty},
                        {"index": 2, "name": "closure", "span": span, "ty": closure_ty}
                    ]},
                    "body": [
                        {"statements": [{"span": span, "kind": {"Assign": [place(2, &closure_ty),
                            {"Aggregate": [{"Adt": [1, null, null, generics]}, []]}]}}],
                         "kind": "Regular",
                         "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [{"Move": place(1, &option_ty)}, {"Move": place(2, &closure_ty)}],
                                "dest": place(0, &option_ty)},
                            "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(vec![ident("core"), ident("option"), ident("Option"), impl_seg, ident("and_then")]),
                 "signature": {"is_unsafe": false, "inputs": [option_ty, closure_ty], "output": option_ty},
                 "body": "Opaque"}
            ],
            "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("closure fixture parses")
}

fn defined_by_const_none(
    graph: &FunctionGraph,
    var: &majit_translate::flowspace::model::Variable,
) -> bool {
    graph.blocks.iter().any(|block| {
        block
            .operations
            .iter()
            .any(|op| op.result.as_ref() == Some(var) && matches!(op.kind, OpKind::ConstNone))
    })
}

fn fn_item_param_llbc() -> Llbc {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: Vec<Value>| {
        json!({
            "name": path,
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let ident = |name: &str| json!({"Ident": [name, 0]});
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let unit = json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": {"types": []}}});
    let fn_ty = json!({"FnDef": {"kind": {"Fun": 1}, "generics": generics}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(vec![ident("fixture"), ident("pass_fn_item")]),
                 "signature": {"is_unsafe": false, "inputs": [], "output": unit},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 0, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": unit}
                    ]},
                    "body": [
                        {"statements": [],
                         "kind": "Regular",
                         "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                                "args": [{"Const": [
                                    {"FnDef": {"kind": {"Fun": 1}, "generics": generics}},
                                    fn_ty
                                ]}],
                                "dest": place(0, &unit)},
                            "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(vec![ident("fixture"), ident("flush_unlocked")]),
                 "signature": {"is_unsafe": false, "inputs": [], "output": unit},
                 "body": "Opaque"},
                {"def_id": 2, "item_meta": meta(vec![ident("fixture"), ident("takes_fn_item")]),
                 "signature": {"is_unsafe": false, "inputs": [fn_ty], "output": unit},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": unit},
                        {"index": 1, "name": "body", "span": span, "ty": fn_ty}
                    ]},
                    "body": [
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}}
                    ]
                 }}}
            ],
            "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("fn-item fixture parses")
}

fn operand_is_void_none(graph: &FunctionGraph, arg: &LinkArg) -> bool {
    let LinkArg::Value(var) = arg else {
        return false;
    };
    FunctionGraph::concretetype_of(var) == ConcreteType::Void || defined_by_const_none(graph, var)
}

#[test]
fn noncapturing_closure_passed_to_and_then_is_void() {
    let graph = lower_function(&zst_closure_and_then(), "use_and_then").expect("and_then lowers");
    let mut closure_operands = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            match target {
                CallTarget::Method { name, .. } if name == "and_then" && args.len() >= 2 => {
                    closure_operands.push(&args[1]);
                }
                CallTarget::FunctionPath { segments, .. }
                    if segments.last().map(String::as_str) == Some("and_then")
                        && args.len() >= 2 =>
                {
                    closure_operands.push(&args[1]);
                }
                _ => {}
            }
        }
    }
    assert!(
        !closure_operands.is_empty(),
        "the lowered graph must pass the closure to and_then"
    );
    assert!(
        closure_operands
            .iter()
            .all(|arg| operand_is_void_none(&graph, arg)),
        "FunctionRepr.convert_const of a capture-less closure is None / Void"
    );
}

#[test]
fn fn_item_param_and_actual_are_void() {
    let llbc = fn_item_param_llbc();
    let callee = lower_function(&llbc, "takes_fn_item").expect("takes_fn_item lowers");
    let mut saw_void_input = false;
    for block in &callee.blocks {
        for op in &block.operations {
            if let OpKind::Input { name, ty, .. } = &op.kind
                && name == "body"
            {
                assert_eq!(ty, &majit_translate::model::ValueType::Void);
                saw_void_input = true;
            }
        }
    }
    assert!(saw_void_input, "callee Input of a FnDef param is Void");

    let caller = lower_function(&llbc, "pass_fn_item").expect("pass_fn_item lowers");
    let mut saw_void_actual = false;
    for block in &caller.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            let CallTarget::FunctionPath { segments, .. } = target else {
                continue;
            };
            if segments.last().map(String::as_str) != Some("takes_fn_item") {
                continue;
            }
            assert!(
                args.iter().all(|arg| operand_is_void_none(&caller, arg)),
                "caller actual of a FnDef argument is Void, not ConstFnAddr"
            );
            saw_void_actual = true;
        }
    }
    assert!(saw_void_actual, "caller must call takes_fn_item");
}

fn fn_pointer_param_llbc() -> Llbc {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: Vec<Value>| {
        json!({
            "name": path,
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let ident = |name: &str| json!({"Ident": [name, 0]});
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let unit = json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": {"types": []}}});
    let fn_ptr_ty = json!({"FnPtr": {
        "skip_binder": {"inputs": [], "output": {"Tuple": []}, "is_unsafe": false}
    }});
    let fn_def_ty = json!({"FnDef": {"kind": {"Fun": 1}, "generics": generics}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(vec![ident("fixture"), ident("pass_fn_ptr")]),
                 "signature": {"is_unsafe": false, "inputs": [], "output": unit},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 0, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": unit}
                    ]},
                    "body": [
                        {"statements": [],
                         "kind": "Regular",
                         "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                                "args": [{"Const": [
                                    {"FnDef": {"kind": {"Fun": 1}, "generics": generics}},
                                    fn_def_ty
                                ]}],
                                "dest": place(0, &unit)},
                            "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(vec![ident("fixture"), ident("hint_really")]),
                 "signature": {"is_unsafe": false, "inputs": [], "output": unit},
                 "body": "Opaque"},
                {"def_id": 2, "item_meta": meta(vec![ident("fixture"), ident("takes_fn_ptr")]),
                 "signature": {"is_unsafe": false, "inputs": [fn_ptr_ty], "output": unit},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": unit},
                        {"index": 1, "name": "function", "span": span, "ty": fn_ptr_ty}
                    ]},
                    "body": [
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}}
                    ]
                 }}}
            ],
            "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("fn-pointer fixture parses")
}

fn operand_is_fn_const(graph: &FunctionGraph, arg: &LinkArg) -> bool {
    let LinkArg::Value(var) = arg else {
        return false;
    };
    graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            if op.result.as_ref() != Some(var) {
                return false;
            }
            match &op.kind {
                OpKind::ConstFnAddr { .. } => true,
                OpKind::Call { target, args, .. } => {
                    args.is_empty() && majit_translate::model::fn_const_segments(target).is_some()
                }
                _ => false,
            }
        })
    })
}

#[test]
fn fn_pointer_param_and_actual_stay_int() {
    let llbc = fn_pointer_param_llbc();
    let callee = lower_function(&llbc, "takes_fn_ptr").expect("takes_fn_ptr lowers");
    let mut saw_int_input = false;
    for block in &callee.blocks {
        for op in &block.operations {
            if let OpKind::Input { name, ty, .. } = &op.kind
                && name == "function"
            {
                assert_eq!(ty, &majit_translate::model::ValueType::Int);
                saw_int_input = true;
            }
        }
    }
    assert!(saw_int_input, "callee Input of a FnPtr param is Int");

    let caller = lower_function(&llbc, "pass_fn_ptr").expect("pass_fn_ptr lowers");
    let mut saw_fn_const_actual = false;
    for block in &caller.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            let CallTarget::FunctionPath { segments, .. } = target else {
                continue;
            };
            if segments.last().map(String::as_str) != Some("takes_fn_ptr") {
                continue;
            }
            assert!(
                args.iter().all(|arg| operand_is_fn_const(&caller, arg)),
                "caller actual of a FnPtr argument is ConstFnAddr / __fn_const"
            );
            saw_fn_const_actual = true;
        }
    }
    assert!(saw_fn_const_actual, "caller must call takes_fn_ptr");
}

/// `conditional_call3` reifies a function item through
/// `UnaryOp(Cast FnDefToFnPtr, Const FnDef)` into a `fn(..)` local.
/// The Const's own type is still `FnDef`; the Cast dest is `FnPtr`, so
/// the actual stays `__fn_const` (`Ptr(FuncType)`).
fn fn_pointer_via_cast_llbc() -> Llbc {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: Vec<Value>| {
        json!({
            "name": path,
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let ident = |name: &str| json!({"Ident": [name, 0]});
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let unit = json!({"Adt": {"id": 0, "builtin": "Tuple", "generics": {"types": []}}});
    let fn_ptr_ty = json!({"FnPtr": {
        "skip_binder": {"inputs": [], "output": {"Tuple": []}, "is_unsafe": false}
    }});
    let fn_def_ty = json!({"FnDef": {"kind": {"Fun": 1}, "generics": generics}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(vec![ident("fixture"), ident("pass_fn_ptr_via_cast")]),
                 "signature": {"is_unsafe": false, "inputs": [], "output": unit},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 0, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": unit},
                        {"index": 1, "name": "function", "span": span, "ty": fn_ptr_ty}
                    ]},
                    "body": [
                        {"statements": [{"span": span, "kind": {"Assign": [
                            place(1, &fn_ptr_ty),
                            {"UnaryOp": [
                                {"Cast": "FnDefToFnPtr"},
                                {"Const": [
                                    {"FnDef": {"kind": {"Fun": 1}, "generics": generics}},
                                    fn_def_ty
                                ]}
                            ]}
                        ]}}],
                         "kind": "Regular",
                         "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 2}, "generics": generics}},
                                "args": [{"Move": place(1, &fn_ptr_ty)}],
                                "dest": place(0, &unit)},
                            "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(vec![ident("fixture"), ident("hint_really")]),
                 "signature": {"is_unsafe": false, "inputs": [], "output": unit},
                 "body": "Opaque"},
                {"def_id": 2, "item_meta": meta(vec![ident("fixture"), ident("takes_fn_ptr")]),
                 "signature": {"is_unsafe": false, "inputs": [fn_ptr_ty], "output": unit},
                 "body": "Opaque"}
            ],
            "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("fn-pointer-via-cast fixture parses")
}

#[test]
fn fn_def_to_fn_ptr_cast_keeps_fn_const() {
    let llbc = fn_pointer_via_cast_llbc();
    let caller = lower_function(&llbc, "pass_fn_ptr_via_cast").expect("cast caller lowers");
    let mut saw_fn_const_actual = false;
    for block in &caller.blocks {
        for op in &block.operations {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            let CallTarget::FunctionPath { segments, .. } = target else {
                continue;
            };
            if segments.last().map(String::as_str) != Some("takes_fn_ptr") {
                continue;
            }
            assert!(
                args.iter().all(|arg| operand_is_fn_const(&caller, arg)),
                "FnDefToFnPtr Cast dest is FnPtr: the actual is ConstFnAddr, not Void"
            );
            saw_fn_const_actual = true;
        }
    }
    assert!(saw_fn_const_actual, "caller must call takes_fn_ptr");
}

/// `FnOnce::call_once(self, args)` of a capture-less closure: both the
/// env and the `(Infallible,)` Args pack are FunctionRepr Void, so
/// FUNC.ARGS drops them.
fn captureless_call_once_llbc() -> Llbc {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: Vec<Value>| {
        json!({
            "name": path,
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let ident = |name: &str| json!({"Ident": [name, 0]});
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let unit = json!({"Adt": {"id": 2, "builtin": "Tuple", "generics": {"types": []}}});
    let env_ty = json!({"Adt": {"id": 0, "generics": generics}});
    let infallible_ty = json!({"Adt": {"id": 1, "generics": generics}});
    let args_ty = json!({"Adt": {
        "id": 3,
        "builtin": "Tuple",
        "generics": {"regions": [], "types": [infallible_ty], "const_generics": [], "trait_refs": []}
    }});
    let layout0 = json!([{
        "key": "fixture-target",
        "value": {"size": 0, "align": 1, "variant_layouts": [{"field_offsets": []}], "repr": {"transparent": false}}
    }]);
    let layout_enum = json!([{
        "key": "fixture-target",
        "value": {"size": 0, "align": 1, "variant_layouts": [], "repr": {"transparent": false}}
    }]);
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(vec![ident("fixture"), ident("try_get"), ident("closure")]),
                 "kind": {"Struct": []}, "layout": layout0,
                 "src": {"Closure": {"info": {"kind": "FnOnce"}}}},
                {"def_id": 1, "item_meta": meta(vec![ident("core"), ident("convert"), ident("Infallible")]),
                 "kind": {"Enum": []}, "layout": layout_enum}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(vec![ident("fixture"), ident("try_get"), ident("closure"),
                    ident("call_once")]),
                 "signature": {"is_unsafe": false, "inputs": [env_ty, args_ty], "output": unit},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 2, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": unit},
                        {"index": 1, "name": "self", "span": span, "ty": env_ty},
                        {"index": 2, "name": "args", "span": span, "ty": args_ty}
                    ]},
                    "body": [
                        {"statements": [], "kind": "Regular",
                         "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular",
                         "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}}
            ],
            "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("call_once fixture parses")
}

#[test]
fn captureless_call_once_callee_inputs_are_void() {
    let graph = lower_function(&captureless_call_once_llbc(), "call_once")
        .expect("capture-less call_once lowers");
    let mut inputs = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            if let OpKind::Input { name, ty, .. } = &op.kind {
                inputs.push((name.clone(), ty.clone()));
            }
        }
    }
    assert!(
        inputs
            .iter()
            .any(|(name, ty)| name == "self" && *ty == majit_translate::model::ValueType::Void),
        "capture-less env Input is Void; inputs={inputs:?}"
    );
    assert!(
        inputs
            .iter()
            .any(|(name, ty)| name == "args" && *ty == majit_translate::model::ValueType::Void),
        "void-ZST Args tuple Input is Void; inputs={inputs:?}"
    );
}
