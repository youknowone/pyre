//! Concrete generic instantiations get distinct graphs.
//!
//! `RDict::get` for a `String` key must reach `<str as PartialEq>::eq`.
//! The `ObjectKey` instantiation must reach `ObjectKey::eq`. The two
//! graphs do not share a key.

use majit_charon_reader::Llbc;
use majit_translate::{
    HostStaticAddrs,
    front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names,
    model::{CallTarget, OpKind},
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
    let object_eq = specialized.iter().any(|f| graph_calls_object_key_eq(&f.graph));
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
    graph.blocks.iter().flat_map(|block| &block.operations).any(|op| {
        match &op.kind {
            OpKind::BinOp { op, .. } if op == "eq" => true,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.len() >= 4
                && segments[segments.len() - 4] == "str"
                && segments[segments.len() - 3] == "traits"
                && segments[segments.len() - 2] == "<Impl>"
                && segments[segments.len() - 1] == "eq",
            _ => false,
        }
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
    graph.blocks.iter().flat_map(|block| &block.operations).any(|op| {
        match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.len() >= 4
                && segments[segments.len() - 4] == "str"
                && segments[segments.len() - 3] == "traits"
                && segments[segments.len() - 2] == "<Impl>"
                && (segments[segments.len() - 1] == "eq"
                    || segments[segments.len() - 1].starts_with("eq__s")),
            OpKind::BinOp { op, .. } if op == "eq" => true,
            _ => false,
        }
    })
}

fn mk_fixture_llbc() -> String {
    use serde_json::json;
    let span = json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
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
    let impl_ref = |id: u64, ty: &serde_json::Value| {
        json!({"kind": {"TraitImpl": {"id": id, "generics": {"regions": [], "types": [ty], "const_generics": [], "trait_refs": []}}}})
    };
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


fn path_is_object_key_eq(segments: &[String]) -> bool {
    segments.iter().any(|segment| segment == "ObjectKey") && segments.last().map(String::as_str) == Some("eq")
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
