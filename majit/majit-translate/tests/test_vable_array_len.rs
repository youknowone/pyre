//! `FixedObjectArray::len` on the virtualizable array must lower to
//! `OpKind::ArrayLen`, not to a `__len` call.
//!
//! The codewriter's virtualizable protocol requires the array to be
//! consumed by an array operation in its defining block.  A call passes it
//! as an argument instead, which reaches `jtransform`'s
//! `handle_residual_call` and aborts the build with "a virtualizable array
//! is passed around".

use majit_charon_reader::Llbc;
use majit_translate::front::mir::lower_fun_decl;
use majit_translate::model::{CallTarget, OpKind};

const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc"
);

/// `PyFrame::_check_stack_index` is `index >= self.stack_base() && index <
/// locals_w!(self).len()` — a read of the virtualizable array whose only
/// consumer is `len`.
#[test]
fn vable_array_len_lowers_to_arraylen_not_a_call() {
    if !std::path::Path::new(INTERPRETER_LLBC).is_file() {
        eprintln!(
            "skipping: {INTERPRETER_LLBC} is missing; run \
             `python3 scripts/extract-llbc.py pyre-interpreter`"
        );
        return;
    }

    let llbc = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter.ullbc");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path().ends_with("::_check_stack_index"))
        .expect("_check_stack_index present in the shipped LLBC");
    let graph = lower_fun_decl(&llbc, fd).expect("lower _check_stack_index");

    let mut array_var = None;
    let mut defining_block = None;
    for (index, block) in graph.blocks.iter().enumerate() {
        for op in &block.operations {
            if let OpKind::FieldRead { field, .. } = &op.kind
                && field.name == "locals_cells_stack_w"
            {
                array_var = op.result.clone();
                defining_block = Some(index);
            }
        }
    }
    let array_var = array_var.expect("a read of the virtualizable array");
    let defining_block = defining_block.expect("the array's defining block");

    let mut len_block = None;
    for (index, block) in graph.blocks.iter().enumerate() {
        for op in &block.operations {
            match &op.kind {
                OpKind::ArrayLen { base, .. } if *base == array_var => {
                    len_block = Some(index);
                }
                // The `__len` retargeting `is_container_len` gives the
                // non-virtualizable length-prefixed containers.
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    args,
                    ..
                } if segments == &["__len".to_string()] && args.contains(&array_var) => {
                    panic!(
                        "the virtualizable array reaches a `__len` call as an argument, \
                         which trips `_check_no_vable_array` via `handle_residual_call`"
                    );
                }
                _ => {}
            }
        }
    }

    assert_eq!(
        len_block,
        Some(defining_block),
        "`ArrayLen` must consume the array in the block that reads it"
    );
}
