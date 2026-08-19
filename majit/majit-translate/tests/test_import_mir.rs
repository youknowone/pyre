use majit_charon_reader::Llbc;
use majit_translate::{
    HostStaticAddrs,
    front::mir::build_semantic_program_from_llbcs_with_static_addrs_and_function_names,
    model::{CallTarget, OpKind},
};

const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc"
);

#[test]
fn dunder_import_lowers_rust_string_find_and_slices_to_rpython_ops() {
    let llbc = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter.ullbc");
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &[llbc],
        HostStaticAddrs::default(),
        &["importing"],
        &["dunder_import"],
    )
    .expect("lower importing::dunder_import");
    let function = program
        .functions
        .iter()
        .find(|f| f.name == "dunder_import")
        .expect("dunder_import graph");

    assert_eq!(
        program
            .struct_fields
            .field_type("W_TupleObject", "wrappeditems"),
        Some("[*mut PyObject]"),
        "PyPy tuple storage must project as GcArray(OBJECTPTR), not ItemsBlock"
    );

    let ops = function
        .graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect::<Vec<_>>();
    assert!(ops.iter().any(|op| matches!(
        &op.kind,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            ..
        } if segments == &["__getslice_rangeto".to_string()]
    )));
    assert!(ops.iter().any(|op| matches!(
        &op.kind,
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            ..
        } if name == "find"
    )));
    assert!(!ops.iter().any(|op| matches!(
        &op.kind,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments },
            ..
        } if segments.starts_with(&["core".to_string(), "str".to_string()])
            && matches!(segments.last().map(String::as_str), Some("find" | "index"))
    )));
}
