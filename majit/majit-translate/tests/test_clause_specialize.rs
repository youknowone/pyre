//! Concrete generic instantiations get distinct graphs.
//!
//! `RDict::get` for a `String` key must reach `<str as PartialEq>::eq`.
//! The `ObjectKey` instantiation must reach `ObjectKey::eq`. The two
//! graphs do not share a key.

use majit_charon_reader::Llbc;
use majit_translate::{
    HostStaticAddrs,
    front::mir::{
        build_semantic_program_from_llbcs_with_static_addrs_and_function_names, lower_function,
    },
    model::{CallTarget, ConcreteType, FunctionGraph, OpKind, ValueType},
};

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);

#[test]
fn rdict_string_and_objectkey_instantiations_resolve_distinct_eq() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["celldict", "dictmultiobject", "rordereddict"],
        &[
            "_orig_module_dict_entries_get",
            "_orig_dict_entries_probe_object",
        ],
    )
    .expect("lower the two RDict callers");

    let specialized: Vec<_> = program
        .functions
        .iter()
        .filter(|f| f.name.contains("__s"))
        .collect();
    assert!(
        specialized.len() >= 2,
        "expected a specialized graph per instantiation, got {}",
        specialized.len()
    );
    let names: Vec<_> = specialized.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names.len(),
        names.iter().collect::<std::collections::HashSet<_>>().len(),
        "instantiation keys collided: {names:?}"
    );

    let string_eq = specialized.iter().any(|f| graph_calls_str_eq(&f.graph));
    let object_eq = specialized
        .iter()
        .any(|f| graph_calls_object_key_eq(&f.graph));
    assert!(
        string_eq,
        "String-key instantiation did not reach <str as PartialEq>::eq"
    );
    assert!(
        object_eq,
        "ObjectKey instantiation did not reach ObjectKey::eq; graphs: {names:?}"
    );
    assert!(
        specialized
            .iter()
            .filter(|f| graph_calls_str_eq(&f.graph))
            .all(|f| !graph_calls_object_key_eq(&f.graph))
    );
}

fn graph_calls_str_eq(graph: &majit_translate::model::FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::BinOp { op, .. } if op == "eq" => true,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                segments.len() >= 4
                    && segments[segments.len() - 4] == "str"
                    && segments[segments.len() - 3] == "traits"
                    && segments[segments.len() - 2] == "<Impl>"
                    && segments[segments.len() - 1] == "eq"
            }
            _ => false,
        })
}

/// `fn mk<T: Default>() -> T` at `T = i64` and `T = String` is two graphs.
/// The i64 copy's produced value is `Int`. The String copy's produced
/// value is the ref bank (`Str`, same register class as `Ref`).
#[test]
fn mk_i64_and_string_spec_graphs_have_int_and_ref_returns() {
    let llbc = Llbc::from_slice(mk_fixture_llbc().as_bytes()).expect("parse mk fixture");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["fixture"],
        &["use_i64", "use_string"],
    )
    .expect("lower mk callers");
    let specs: Vec<_> = program
        .functions
        .iter()
        .filter(|f| f.name.contains("mk__s"))
        .collect();
    assert_eq!(
        specs.len(),
        2,
        "expected two mk instantiations, got {:?}",
        specs.iter().map(|f| &f.name).collect::<Vec<_>>()
    );
    assert_ne!(specs[0].name, specs[1].name);
    let mut kinds = Vec::new();
    for spec in &specs {
        let kind = graph_return_kind(&spec.graph);
        assert!(
            kind == "Int" || kind == "Ref",
            "{} return kind {kind}, ops {:?}",
            spec.name,
            op_kinds(&spec.graph)
        );
        kinds.push(kind);
    }
    kinds.sort();
    assert_eq!(kinds, vec!["Int", "Ref"]);
}

/// `fn eqv<Q: Eq, K: Borrow<Q>>` at `Q = K = String` aliases
/// `Borrow::borrow` and calls the `String` eq impl.
#[test]
fn string_eqv_spec_aliases_borrow_and_calls_string_eq() {
    let llbc = Llbc::load(OBJECT_LLBC).expect("load pyre-object.ullbc");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["celldict", "dictmultiobject", "rordereddict"],
        &[
            "_orig_module_dict_entries_get",
            "_orig_dict_entries_probe_object",
        ],
    )
    .expect("lower the two RDict callers");
    let eqv: Vec<_> = program
        .functions
        .iter()
        .filter(|f| f.name.contains("equivalent__s") && graph_calls_str_eq(&f.graph))
        .collect();
    assert!(
        !eqv.is_empty(),
        "no String equivalent copy; specs: {:?}",
        program
            .functions
            .iter()
            .filter(|f| f.name.contains("__s"))
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
    );
    for spec in &eqv {
        assert!(
            !graph_calls_borrow(&spec.graph),
            "{} still calls Borrow::borrow",
            spec.name
        );
        assert!(
            graph_calls_string_eq_impl(&spec.graph),
            "{} does not call the String eq impl",
            spec.name
        );
    }
}

/// A spec copy at `V = ()` calls `put<V>(slot: &mut V, v: V) -> V`.
/// The specialized `put` graph keeps one non-void argument, the `Ref`
/// slot; the unit value is dropped.
#[test]
fn put_unit_spec_graph_drops_the_value_arg() {
    let llbc = Llbc::from_slice(put_fixture_llbc().as_bytes()).expect("parse put fixture");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["fixture"],
        &["use_unit"],
    )
    .expect("lower use_unit");
    let puts: Vec<_> = program
        .functions
        .iter()
        .filter(|f| f.name.contains("put__s"))
        .collect();
    assert_eq!(
        puts.len(),
        1,
        "expected one specialized put, got {:?}",
        program
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
    );
    let kinds = non_void_input_kinds(&puts[0].graph);
    assert_eq!(
        kinds,
        vec!["Ref"],
        "{} non-void inputs {kinds:?}, ops {:?}",
        puts[0].name,
        op_kinds(&puts[0].graph)
    );
}

/// A spec copy calls `pyre_object::lltype::malloc_typed`. The call stays
/// on the bare path: `HOST_ENV` already owns that builtin, so no
/// `malloc_typed__s` graph is built.
#[test]
fn spec_copy_keeps_bare_lltype_malloc_typed() {
    let llbc =
        Llbc::from_slice(malloc_typed_fixture_llbc().as_bytes()).expect("parse malloc fixture");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["fixture"],
        &["use_wrap"],
    )
    .expect("lower use_wrap");
    let wraps: Vec<_> = program
        .functions
        .iter()
        .filter(|f| f.name.contains("wrap__s"))
        .collect();
    assert_eq!(
        wraps.len(),
        1,
        "expected one specialized wrap, got {:?}",
        program
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        graph_calls_bare_malloc_typed(&wraps[0].graph),
        "{} dropped the bare malloc_typed call, ops {:?}",
        wraps[0].name,
        op_kinds(&wraps[0].graph)
    );
    assert!(
        program
            .functions
            .iter()
            .all(|f| !f.name.contains("malloc_typed__s")),
        "host builtin was copied: {:?}",
        program
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .filter(|name| name.contains("malloc"))
            .collect::<Vec<_>>()
    );
}

/// `fn f(s: &mut S) { replace(&mut s.u, ()) }` where `S.u` is `()`.
/// The borrow aliases a Void field read: no `getfield` of `u`.
#[test]
fn unit_field_borrow_is_void_and_emits_no_getfield() {
    let llbc = Llbc::from_slice(unit_field_fixture_llbc().as_bytes()).expect("parse S fixture");
    let graph = lower_function(&llbc, "f").expect("lower f");
    assert!(
        !graph_reads_field(&graph, "u"),
        "zero-sized field u must not be a getfield, ops {:?}",
        op_kinds(&graph)
    );
    let arg = replace_first_arg(&graph);
    assert_eq!(
        FunctionGraph::concretetype_of(arg),
        ConcreteType::Void,
        "&mut s.u must be Void, ops {:?}",
        op_kinds(&graph)
    );
}

fn graph_reads_field(graph: &FunctionGraph, name: &str) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field.name == name))
}

fn replace_first_arg(graph: &FunctionGraph) -> &majit_translate::flowspace::model::Variable {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find_map(|op| match &op.kind {
            OpKind::Call { args, .. } if !args.is_empty() => args[0].as_variable(),
            _ => None,
        })
        .expect("put call")
}

fn non_void_input_kinds(graph: &majit_translate::model::FunctionGraph) -> Vec<&'static str> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| match &op.kind {
            OpKind::Input { ty, .. } if !matches!(ty, ValueType::Void) => match ty {
                ValueType::Ref(_) | ValueType::Str => Some("Ref"),
                ValueType::Int | ValueType::Unsigned => Some("Int"),
                _ => Some("other"),
            },
            _ => None,
        })
        .collect()
}

fn graph_return_kind(graph: &majit_translate::model::FunctionGraph) -> &'static str {
    let mut saw_int = false;
    let mut saw_ref = false;
    for op in graph.blocks.iter().flat_map(|block| &block.operations) {
        let ty = match &op.kind {
            OpKind::Call { result_ty, .. }
            | OpKind::UnaryOp { result_ty, .. }
            | OpKind::BinOp { result_ty, .. } => Some(result_ty),
            OpKind::ConstInt(_) => return "Int",
            _ => None,
        };
        match ty {
            Some(majit_translate::model::ValueType::Int)
            | Some(majit_translate::model::ValueType::Unsigned) => saw_int = true,
            Some(majit_translate::model::ValueType::Ref(_))
            | Some(majit_translate::model::ValueType::Str) => saw_ref = true,
            _ => {}
        }
    }
    if saw_int && !saw_ref {
        "Int"
    } else if saw_ref && !saw_int {
        "Ref"
    } else {
        "mixed"
    }
}

fn op_kinds(graph: &majit_translate::model::FunctionGraph) -> Vec<String> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .map(|op| format!("{:?}", op.kind))
        .collect()
}

fn call_leaf_is_borrow(segments: &[String]) -> bool {
    segments.last().map(String::as_str) == Some("borrow")
}

fn graph_calls_borrow(graph: &majit_translate::model::FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => call_leaf_is_borrow(segments),
            OpKind::Call {
                target:
                    CallTarget::Method {
                        name,
                        resolved_path,
                        ..
                    },
                ..
            } => {
                name == "borrow"
                    || resolved_path
                        .as_ref()
                        .is_some_and(|path| call_leaf_is_borrow(&path.segments))
            }
            _ => false,
        })
}

fn graph_calls_string_eq_impl(graph: &majit_translate::model::FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                segments.len() >= 4
                    && segments[segments.len() - 4] == "str"
                    && segments[segments.len() - 3] == "traits"
                    && segments[segments.len() - 2] == "<Impl>"
                    && (segments[segments.len() - 1] == "eq"
                        || segments[segments.len() - 1].starts_with("eq__s"))
            }
            OpKind::BinOp { op, .. } if op == "eq" => true,
            _ => false,
        })
}

fn mk_fixture_llbc() -> String {
    use serde_json::json;
    let span =
        json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str], local: bool| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": local
        })
    };
    let i64_ty = json!({"Literal": {"Int": "I64"}});
    let string_ty = json!({"Adt": {"id": {"Adt": 0}, "generics": {"regions": [], "types": [], "const_generics": [], "trait_refs": []}}});
    let tvar = json!({"TypeVar": {"Bound": [0, 0]}});
    let empty_g = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let impl_ref = |id: u64, ty: &serde_json::Value| json!({"kind": {"TraitImpl": {"id": id, "generics": {"regions": [], "types": [ty], "const_generics": [], "trait_refs": []}}}});
    let opaque = |id: u64, path: &[&str], out: &serde_json::Value| {
        json!({
            "def_id": id,
            "item_meta": meta(path, false),
            "signature": {"is_unsafe": false, "inputs": [], "output": out},
            "body": null
        })
    };
    let body = |ret: &serde_json::Value, func: serde_json::Value| {
        json!({
            "Unstructured": {
                "span": span,
                "locals": {"arg_count": 0, "locals": [{"index": 0, "name": null, "span": span, "ty": ret}]},
                "body": [
                    {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                        "call": {"func": func, "args": [], "dest": {"kind": {"Local": 0}, "ty": ret}},
                        "target": 2,
                        "on_unwind": 1
                    }}}},
                    {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}},
                    {"statements": [], "terminator": {"span": span, "kind": "Return"}}
                ]
            }
        })
    };
    let call_mk = |id: u64, name: &str, ty: &serde_json::Value, impl_id: u64| {
        json!({
            "def_id": id,
            "item_meta": meta(&["fixture", name], true),
            "signature": {"is_unsafe": false, "inputs": [], "output": ty},
            "body": body(ty, json!({"Regular": {
                "kind": {"Fun": {"Regular": 3}},
                "generics": {"regions": [], "types": [ty], "const_generics": [], "trait_refs": [impl_ref(impl_id, ty)]}
            }}))
        })
    };
    let mk = json!({
        "def_id": 3,
        "item_meta": meta(&["fixture", "mk"], true),
        "signature": {"is_unsafe": false, "inputs": [], "output": tvar},
        "generics": {
            "regions": [],
            "types": [{"index": 0, "name": "T"}],
            "const_generics": [],
            "trait_clauses": [{"clause_id": 0}],
            "regions_outlive": [],
            "types_outlive": [],
            "trait_type_constraints": []
        },
        "body": body(&tvar, json!({"Regular": {
            "kind": {"Trait": [{"kind": {"Clause": {"Bound": [0, 0]}}}, 0, 0]},
            "generics": empty_g
        }}))
    });
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "fun_decls": [
                opaque(0, &["core", "default", "Default", "default"], &tvar),
                opaque(1, &["core", "default", "Default", "default"], &i64_ty),
                opaque(2, &["alloc", "string", "String", "default"], &string_ty),
                mk,
                call_mk(4, "use_i64", &i64_ty, 0),
                call_mk(5, "use_string", &string_ty, 1)
            ],
            "global_decls": [],
            "type_decls": [{
                "def_id": 0,
                "item_meta": meta(&["alloc", "string", "String"], false),
                "kind": "Opaque",
                "src": "TopLevel"
            }],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["core", "default", "Default"], false)
            }],
            "trait_impls": [
                {
                    "methods": [{"kind": {"TraitMethod": [0, 0]}, "skip_binder": {"id": 1}}],
                    "impl_trait": {"id": 0, "generics": {"regions": [], "types": [i64_ty], "const_generics": [], "trait_refs": []}},
                    "implied_trait_refs": []
                },
                {
                    "methods": [{"kind": {"TraitMethod": [0, 0]}, "skip_binder": {"id": 2}}],
                    "impl_trait": {"id": 0, "generics": {"regions": [], "types": [string_ty], "const_generics": [], "trait_refs": []}},
                    "implied_trait_refs": []
                }
            ]
        }
    });
    file.to_string()
}

fn put_fixture_llbc() -> String {
    use serde_json::json;
    let span =
        json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str], local: bool| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": local
        })
    };
    let empty_g = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let unit = json!({"Adt": {"id": "Tuple", "generics": empty_g}});
    let tvar = json!({"TypeVar": {"Bound": [0, 0]}});
    let slot_ty = json!({"Ref": {"region": "Erased", "ty": tvar, "kind": "Mut"}});
    let place = |id: u64, ty: &serde_json::Value| json!({"kind": {"Local": id}, "ty": ty});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| json!({"index": index, "name": name, "span": span, "ty": ty});
    let impl_ref = json!({
        "kind": {"TraitImpl": {"id": 0, "generics": {"regions": [], "types": [unit], "const_generics": [], "trait_refs": []}}}
    });
    let generics = |clauses: bool| {
        json!({
            "regions": [],
            "types": [{"index": 0, "name": "V"}],
            "const_generics": [],
            "trait_clauses": if clauses { json!([{"clause_id": 0}]) } else { json!([]) },
            "regions_outlive": [],
            "types_outlive": [],
            "trait_type_constraints": []
        })
    };
    let sig = json!({
        "is_unsafe": false,
        "inputs": [slot_ty, tvar],
        "output": tvar
    });
    let put_body = json!({
        "Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 2,
                "locals": [local(0, None, &tvar), local(1, Some("slot"), &slot_ty), local(2, Some("v"), &tvar)]
            },
            "body": [
                {"statements": [], "terminator": {"span": span, "kind": "Return"}}
            ]
        }
    });
    let fill_body = json!({
        "Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 2,
                "locals": [
                    local(0, None, &tvar),
                    local(1, Some("slot"), &slot_ty),
                    local(2, Some("v"), &tvar),
                    local(3, Some("tmp"), &tvar)
                ]
            },
            "body": [
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Trait": [{"kind": {"Clause": {"Bound": [0, 0]}}}, 0, 0]},
                            "generics": empty_g
                        }},
                        "args": [],
                        "dest": place(3, &tvar)
                    },
                    "target": 2,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}},
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Fun": {"Regular": 2}},
                            "generics": {"regions": [], "types": [tvar], "const_generics": [], "trait_refs": []}
                        }},
                        "args": [{"Copy": place(1, &slot_ty)}, {"Copy": place(2, &tvar)}],
                        "dest": place(0, &tvar)
                    },
                    "target": 3,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "Return"}}
            ]
        }
    });
    let use_body = json!({
        "Unstructured": {
            "span": span,
            "locals": {"arg_count": 0, "locals": [local(0, None, &unit)]},
            "body": [
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Fun": {"Regular": 3}},
                            "generics": {"regions": [], "types": [unit], "const_generics": [], "trait_refs": [impl_ref]}
                        }},
                        "args": [],
                        "dest": place(0, &unit)
                    },
                    "target": 2,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}},
                {"statements": [], "terminator": {"span": span, "kind": "Return"}}
            ]
        }
    });
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "fun_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(&["core", "default", "Default", "default"], false),
                    "signature": {"is_unsafe": false, "inputs": [], "output": tvar},
                    "body": null
                },
                {
                    "def_id": 1,
                    "item_meta": meta(&["core", "default", "Default", "default"], false),
                    "signature": {"is_unsafe": false, "inputs": [], "output": unit},
                    "body": null
                },
                {
                    "def_id": 2,
                    "item_meta": meta(&["fixture", "put"], true),
                    "signature": sig,
                    "generics": generics(false),
                    "body": put_body
                },
                {
                    "def_id": 3,
                    "item_meta": meta(&["fixture", "fill"], true),
                    "signature": sig,
                    "generics": generics(true),
                    "body": fill_body
                },
                {
                    "def_id": 4,
                    "item_meta": meta(&["fixture", "use_unit"], true),
                    "signature": {"is_unsafe": false, "inputs": [], "output": unit},
                    "body": use_body
                }
            ],
            "global_decls": [],
            "type_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["core", "default", "Default"], false)
            }],
            "trait_impls": [{
                "methods": [{"kind": {"TraitMethod": [0, 0]}, "skip_binder": {"id": 1}}],
                "impl_trait": {"id": 0, "generics": {"regions": [], "types": [unit], "const_generics": [], "trait_refs": []}},
                "implied_trait_refs": []
            }]
        }
    });
    file.to_string()
}

fn unit_field_fixture_llbc() -> String {
    use serde_json::json;
    let span =
        json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str], local: bool| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": local
        })
    };
    let empty_g = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let i64_ty = json!({"Literal": {"Int": "I64"}});
    let unit = json!({"Adt": {"id": "Tuple", "generics": {"types": []}}});
    let s_ty = json!({"Adt": {"id": {"Adt": 0}, "generics": empty_g}});
    let s_ref = json!({"Ref": ["Erased", s_ty, "Mut"]});
    let place = |id: u64, ty: &serde_json::Value| json!({"kind": {"Local": id}, "ty": ty});
    let local = |index: u64, name: Option<&str>, ty: &serde_json::Value| json!({"index": index, "name": name, "span": span, "ty": ty});
    let field_u = json!({
        "kind": {"Projection": [
            {"kind": {"Projection": [place(1, &s_ref), "Deref"]}, "ty": s_ty},
            {"Field": [{"Adt": [0, null]}, 1]}
        ]},
        "ty": unit
    });
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "type_decls": [{
                "def_id": 0,
                "item_meta": meta(&["fixture", "S"], true),
                "kind": {"Struct": [
                    {"name": "a", "ty": i64_ty, "attr_info": null},
                    {"name": "u", "ty": unit, "attr_info": null}
                ]}
            }],
            "fun_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(&["fixture", "f"], true),
                    "signature": {"is_unsafe": false, "inputs": [s_ref], "output": unit},
                    "body": {"Unstructured": {
                        "span": span,
                        "locals": {"arg_count": 1, "locals": [
                            local(0, None, &unit),
                            local(1, Some("s"), &s_ref),
                            local(2, Some("slot"), &unit),
                            local(3, Some("val"), &unit)
                        ]},
                        "body": [
                            {"statements": [
                                {"span": span, "kind": {"Assign": [
                                    place(2, &unit),
                                    {"Ref": {"place": field_u, "kind": "Mut", "ptr_metadata": "None"}}
                                ]}},
                                {"span": span, "kind": {"Assign": [
                                    place(3, &unit),
                                    {"Aggregate": [{"Adt": ["Tuple", null, null, empty_g]}, []]}
                                ]}}
                            ], "terminator": {"span": span, "kind": {"Call": {
                                "call": {
                                    "func": {"Regular": {
                                        "kind": {"Fun": {"Regular": 1}},
                                        "generics": empty_g
                                    }},
                                    "args": [{"Copy": place(2, &unit)}, {"Copy": place(3, &unit)}],
                                    "dest": place(0, &unit)
                                },
                                "target": 1,
                                "on_unwind": 2
                            }}}},
                            {"statements": [], "terminator": {"span": span, "kind": "Return"}},
                            {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}}
                        ]
                    }}
                },
                {
                    "def_id": 1,
                    "item_meta": meta(&["fixture", "put"], true),
                    "signature": {"is_unsafe": false, "inputs": [unit, unit], "output": unit},
                    "body": null
                }
            ],
            "global_decls": [],
            "trait_decls": [],
            "trait_impls": []
        }
    });
    file.to_string()
}

fn graph_calls_bare_malloc_typed(graph: &FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => {
                segments.last().map(String::as_str) == Some("malloc_typed")
                    && segments.iter().any(|segment| segment == "lltype")
            }
            _ => false,
        })
}

fn malloc_typed_fixture_llbc() -> String {
    use serde_json::json;
    let span =
        json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str], local: bool| {
        json!({
            "name": path.iter().map(|seg| json!({"Ident": [seg, 0]})).collect::<Vec<_>>(),
            "span": span,
            "source_text": null,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
            "is_local": local
        })
    };
    let i64_ty = json!({"Literal": {"Int": "I64"}});
    let tvar = json!({"TypeVar": {"Bound": [0, 0]}});
    let empty_g = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let impl_ref = json!({
        "kind": {"TraitImpl": {"id": 0, "generics": {"regions": [], "types": [i64_ty], "const_generics": [], "trait_refs": []}}}
    });
    let generics = json!({
        "regions": [],
        "types": [{"index": 0, "name": "T"}],
        "const_generics": [],
        "trait_clauses": [{"clause_id": 0}],
        "regions_outlive": [],
        "types_outlive": [],
        "trait_type_constraints": []
    });
    let ret_body = |ty: &serde_json::Value| {
        json!({
            "Unstructured": {
                "span": span,
                "locals": {"arg_count": 0, "locals": [{"index": 0, "name": null, "span": span, "ty": ty}]},
                "body": [
                    {"statements": [], "terminator": {"span": span, "kind": "Return"}}
                ]
            }
        })
    };
    let place = |id: u64, ty: &serde_json::Value| json!({"kind": {"Local": id}, "ty": ty});
    let local = |index: u64, ty: &serde_json::Value| json!({"index": index, "name": null, "span": span, "ty": ty});
    let wrap_body = json!({
        "Unstructured": {
            "span": span,
            "locals": {
                "arg_count": 0,
                "locals": [local(0, &i64_ty), local(1, &i64_ty)]
            },
            "body": [
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Trait": [{"kind": {"Clause": {"Bound": [0, 0]}}}, 0, 0]},
                            "generics": empty_g
                        }},
                        "args": [],
                        "dest": place(1, &i64_ty)
                    },
                    "target": 2,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}},
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Fun": {"Regular": 2}},
                            "generics": {"regions": [], "types": [i64_ty], "const_generics": [], "trait_refs": []}
                        }},
                        "args": [],
                        "dest": place(0, &i64_ty)
                    },
                    "target": 3,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "Return"}}
            ]
        }
    });
    let use_body = json!({
        "Unstructured": {
            "span": span,
            "locals": {"arg_count": 0, "locals": [local(0, &i64_ty)]},
            "body": [
                {"statements": [], "terminator": {"span": span, "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {
                            "kind": {"Fun": {"Regular": 3}},
                            "generics": {"regions": [], "types": [i64_ty], "const_generics": [], "trait_refs": [impl_ref]}
                        }},
                        "args": [],
                        "dest": place(0, &i64_ty)
                    },
                    "target": 2,
                    "on_unwind": 1
                }}}},
                {"statements": [], "terminator": {"span": span, "kind": "UnwindResume"}},
                {"statements": [], "terminator": {"span": span, "kind": "Return"}}
            ]
        }
    });
    let file = json!({
        "charon_version": "0.1.201",
        "has_errors": false,
        "translated": {
            "crate_name": "fixture",
            "fun_decls": [
                {
                    "def_id": 0,
                    "item_meta": meta(&["core", "marker", "Marker", "mark"], false),
                    "signature": {"is_unsafe": false, "inputs": [], "output": tvar},
                    "body": null
                },
                {
                    "def_id": 1,
                    "item_meta": meta(&["fixture", "mark_i64"], false),
                    "signature": {"is_unsafe": false, "inputs": [], "output": i64_ty},
                    "body": null
                },
                {
                    "def_id": 2,
                    "item_meta": meta(&["pyre_object", "lltype", "malloc_typed"], true),
                    "signature": {"is_unsafe": false, "inputs": [], "output": i64_ty},
                    "generics": {
                        "regions": [],
                        "types": [{"index": 0, "name": "T"}],
                        "const_generics": [],
                        "trait_clauses": [],
                        "regions_outlive": [],
                        "types_outlive": [],
                        "trait_type_constraints": []
                    },
                    "body": ret_body(&i64_ty)
                },
                {
                    "def_id": 3,
                    "item_meta": meta(&["fixture", "wrap"], true),
                    "signature": {"is_unsafe": false, "inputs": [], "output": i64_ty},
                    "generics": generics,
                    "body": wrap_body
                },
                {
                    "def_id": 4,
                    "item_meta": meta(&["fixture", "use_wrap"], true),
                    "signature": {"is_unsafe": false, "inputs": [], "output": i64_ty},
                    "body": use_body
                }
            ],
            "global_decls": [],
            "type_decls": [],
            "trait_decls": [{
                "def_id": 0,
                "item_meta": meta(&["core", "marker", "Marker"], false)
            }],
            "trait_impls": [{
                "methods": [{"kind": {"TraitMethod": [0, 0]}, "skip_binder": {"id": 1}}],
                "impl_trait": {"id": 0, "generics": {"regions": [], "types": [i64_ty], "const_generics": [], "trait_refs": []}},
                "implied_trait_refs": []
            }]
        }
    });
    file.to_string()
}

fn path_is_object_key_eq(segments: &[String]) -> bool {
    segments.iter().any(|segment| segment == "ObjectKey")
        && segments.last().map(String::as_str) == Some("eq")
}

fn graph_calls_object_key_eq(graph: &majit_translate::model::FunctionGraph) -> bool {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .any(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => path_is_object_key_eq(segments),
            OpKind::Call {
                target:
                    CallTarget::Method {
                        name,
                        receiver_root,
                        resolved_path,
                        ..
                    },
                ..
            } => {
                resolved_path
                    .as_ref()
                    .is_some_and(|path| path_is_object_key_eq(&path.segments))
                    || (name == "eq"
                        && receiver_root
                            .as_deref()
                            .is_some_and(|root| root.contains("ObjectKey")))
            }
            _ => false,
        })
}
