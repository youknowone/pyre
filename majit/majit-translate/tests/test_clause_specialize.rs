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
