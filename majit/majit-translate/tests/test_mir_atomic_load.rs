//! Atomic ordering must survive the Rust-input boundary. RPython's
//! flowcontext.FlowContext.record_block propagates FlowingError rather than
//! publishing a graph with an unsupported operation erased.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::{LowerError, lower_function};
use serde_json::{Value, json};

fn fixture(ordering: Option<&str>) -> Llbc {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str]| {
        json!({
            "name": path.iter().map(|s| json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let adt = |id| json!({"Adt": {"id": id, "generics": generics}});
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    // `AtomicU64` is the alias `Atomic<u64>`.
    let atomic = json!({"Adt": {"id": 0, "generics":
        {"regions": [], "types": [word], "const_generics": [], "trait_refs": []}}});
    let receiver = json!({"Ref": ["Erased", atomic, "Shared"]});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let variants = ["Relaxed", "Acquire", "SeqCst", "Release", "AcqRel"];
    let statements = ordering
        .map(|name| {
            let index = variants.iter().position(|v| *v == name).unwrap();
            json!([{"span": span, "kind": {"Assign": [place(2, &adt(1)),
            {"Aggregate": [{"Adt": [1, index, null, generics]}, []]}]}}])
        })
        .unwrap_or_else(|| json!([]));
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "sync", "atomic", "Atomic"]), "kind": "Opaque"},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Ordering"]),
                 "kind": {"Enum": variants.iter().enumerate().map(|(i, name)| json!({
                    "name": name, "fields": [],
                    "discriminant": {"Scalar": {"Unsigned": ["U8", i.to_string()]}}
                 })).collect::<Vec<_>>()}}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "read_atomic"]),
                 "signature": {"is_unsafe": false,
                    "inputs": if ordering.is_some() {vec![receiver.clone()]} else {vec![receiver.clone(), adt(1)]},
                    "output": word},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": if ordering.is_some() {1} else {2}, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": word},
                        {"index": 1, "name": "slot", "span": span, "ty": receiver},
                        {"index": 2, "name": "ordering", "span": span, "ty": adt(1)}
                    ]},
                    "body": [
                        {"statements": statements, "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [{"Copy": place(1, &receiver)}, {"Copy": place(2, &adt(1))}],
                                "dest": place(0, &word)}, "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Atomic", "load"]),
                 "signature": {"is_unsafe": false, "inputs": [receiver, adt(1)], "output": word}, "body": "Opaque"}
            ], "global_decls": [], "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("atomic fixture parses")
}

#[test]
fn atomic_load_relaxed_keeps_existing_lowering() {
    lower_function(&fixture(Some("Relaxed")), "read_atomic").expect("known Relaxed load");
}

#[test]
fn atomic_load_must_not_erase_ordering() {
    for ordering in [
        Some("Acquire"),
        Some("SeqCst"),
        Some("Release"),
        Some("AcqRel"),
        None,
    ] {
        let result = lower_function(&fixture(ordering), "read_atomic");
        assert!(
            matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering")),
            "{ordering:?}: must reject unsupported ordering, got {result:?}"
        );
    }
}

#[test]
#[ignore = "requires freshly extracted pyre-object LLBC; run extraction --check first"]
fn atomic_load_real_acquire_reader_of_quasi_field_lowers() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-object.ullbc"
    ))
    .expect("load fresh pyre-object corpus");
    // Quasi-immutable `version_tag: AtomicU64`. The Acquire load lowers
    // to a FieldRead so `record_quasiimmut_field` still guards it.
    let graph = lower_function(&llbc, "w_type_get_version_tag")
        .expect("Acquire load of quasi version_tag lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "lowered w_type_get_version_tag has no version_tag field read"
    );
    // get_instantiate publishes the cached W_TypeObject to allocators. Its
    // readers and late writers are all GIL-serialized, so the source spells a
    // Relaxed load — the one ordering the frontend lowers to a plain read.
    let result = lower_function(&llbc, "get_instantiate");
    assert!(
        result.is_ok(),
        "get_instantiate: the Relaxed reader must keep lowering: {result:?}"
    );
}

#[test]
#[ignore = "loads target/llbc/pyre-interpreter.ullbc"]
fn acquire_quasi_w_globals_lowers_across_the_block_copy() {
    let llbc = Llbc::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    ))
    .expect("fresh pyre-interpreter corpus");
    let graph = lower_function(&llbc, "w_code_get_w_globals")
        .expect("Acquire load of quasi w_globals lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. } if field.name == "w_globals"
            )
        })
    });
    assert!(
        saw,
        "lowered w_code_get_w_globals has no w_globals field read"
    );
}

fn quasi_atomic_u64_field_fixture(owner: &str, field: &str, entries: Option<&str>) -> Llbc {
    let span = json!({"data": {"file_id": 0,
        "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
    let meta = |path: &[&str]| {
        json!({
            "name": path.iter().map(|s| json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span, "source_text": null, "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    };
    let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
    let word = json!({"Scalar": {"Integer": {"Unsigned": "U64"}}});
    let atomic = json!({"Adt": {"id": 0, "generics":
        {"regions": [], "types": [word], "const_generics": [], "trait_refs": []}}});
    let owner_adt = json!({"Adt": {"id": 2, "generics": generics, "builtin": null}});
    let owner_ref = json!({"Ref": ["Erased", owner_adt, "Shared"]});
    let recv_ty = json!({"Ref": ["Erased", atomic, "Shared"]});
    let adt_ord = json!({"Adt": {"id": 1, "generics": generics}});
    let place = |id, ty: &Value| json!({"kind": {"Local": id}, "ty": ty});
    let field_place = json!({
        "kind": {"Projection": [
            {"kind": {"Projection": [place(1, &owner_ref), "Deref"]}, "ty": owner_adt},
            {"Field": [null, 0]}
        ]},
        "ty": atomic
    });
    let variants = ["Relaxed", "Acquire", "SeqCst", "Release", "AcqRel"];
    let attr = json!({"attributes": [], "inline": null, "rename": null, "public": true});
    let global_decls = match entries {
        Some(joined) => json!([{
            "def_id": 0,
            "item_meta": meta(&["fixture", &format!("_immutable_fields_{owner}")]),
            "global_kind": "NamedConst",
            "value": {"Value": [1, [{"Str": joined}, {"Ref": [
                "Erased",
                {"Scalar": {"Integer": {"Unsigned": "U8"}}},
                "Shared"
            ]}]]}
        }]),
        None => json!([]),
    };
    let file = json!({"charon_version": "0.1.201", "has_errors": false,
        "translated": {"crate_name": "fixture",
            "type_decls": [
                {"def_id": 0, "item_meta": meta(&["core", "sync", "atomic", "Atomic"]), "kind": "Opaque"},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Ordering"]),
                 "kind": {"Enum": variants.iter().enumerate().map(|(i, name)| json!({
                    "name": name, "fields": [],
                    "discriminant": {"Scalar": {"Unsigned": ["U8", i.to_string()]}}
                 })).collect::<Vec<_>>()}},
                {"def_id": 2, "item_meta": meta(&["fixture", owner]),
                 "kind": {"Struct": [{
                    "name": field, "ty": atomic, "attr_info": attr
                 }]}}
            ],
            "fun_decls": [
                {"def_id": 0, "item_meta": meta(&["fixture", "read_field"]),
                 "signature": {"is_unsafe": false,
                    "inputs": [owner_ref.clone()],
                    "output": word},
                 "body": {"Unstructured": {"span": span,
                    "locals": {"arg_count": 1, "locals": [
                        {"index": 0, "name": null, "span": span, "ty": word},
                        {"index": 1, "name": "owner", "span": span, "ty": owner_ref},
                        {"index": 2, "name": "ordering", "span": span, "ty": adt_ord},
                        {"index": 3, "name": "recv", "span": span, "ty": recv_ty}
                    ]},
                    "body": [
                        {"statements": [
                            {"span": span, "kind": {"Assign": [place(2, &adt_ord),
                                {"Aggregate": [{"Adt": [1, 1, null, generics]}, []]}]}},
                            {"span": span, "kind": {"Assign": [place(3, &recv_ty),
                                {"Ref": {"place": field_place, "kind": "Shared", "ptr_metadata": null}}]}}
                        ], "terminator": {"span": span, "kind": {"Call": {
                            "call": {"func": {"Regular": {"kind": {"Fun": 1}, "generics": generics}},
                                "args": [{"Copy": place(3, &recv_ty)}, {"Copy": place(2, &adt_ord)}],
                                "dest": place(0, &word)}, "target": 1, "on_unwind": 2
                        }}}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "Return"}},
                        {"statements": [], "kind": "Regular", "terminator": {"span": span, "kind": "UnwindResume"}}
                    ]
                 }}},
                {"def_id": 1, "item_meta": meta(&["core", "sync", "atomic", "Atomic", "load"]),
                 "signature": {"is_unsafe": false, "inputs": [recv_ty, adt_ord], "output": word}, "body": "Opaque"}
            ], "global_decls": global_decls, "trait_decls": [], "trait_impls": []
        }
    });
    Llbc::from_slice(file.to_string().as_bytes()).expect("atomic field fixture parses")
}

#[test]
fn acquire_quasi_atomic_field_lowers_and_non_quasi_still_errors() {
    let graph = lower_function(
        &quasi_atomic_u64_field_fixture("Owner", "version_tag", Some("version_tag?")),
        "read_field",
    )
    .expect("Acquire load of quasi AtomicU64 field lowers");
    let saw = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                majit_translate::model::OpKind::FieldRead { field, .. }
                    if field.name == "version_tag"
            )
        })
    });
    assert!(
        saw,
        "lowered quasi AtomicU64 field has no version_tag field read"
    );

    let result = lower_function(
        &quasi_atomic_u64_field_fixture("Plain", "hot_counter", None),
        "read_field",
    );
    assert!(
        matches!(result, Err(LowerError::Unsupported(ref message))
            if message.contains("atomic load ordering Acquire")
                && message.contains("requires address-preserving ordered lowering")),
        "Acquire load of a non-quasi AtomicU64 field must stay unsupported, got {result:?}"
    );
}
