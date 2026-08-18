//! Pyre's signed string-find compatibility call to RPython lowering.
//!
//! PyPy reads the dot position as a Signed and slices on it:
//! `dotindex = name.find(".")` then `name[:dotindex]`
//! (`pypy/module/_frozen_importlib/interp_import.py:76-79`).
//! Rust's standard `str::find` returns `Option<usize>` and has no start
//! argument, so the interpreter exposes an exact native compatibility helper.
//! During source translation this adapter replaces that helper with the
//! existing RPython `str.find(char, start) -> Signed` method operation.

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, OpKind, SpaceOperation, ValueType};

pub(crate) fn is_rpython_str_find_char(segments: &[String]) -> bool {
    segments.ends_with(&["importing".to_string(), "rpython_str_find_char".to_string()])
}

pub(crate) fn is_rpython_str_slice_prefix(segments: &[String]) -> bool {
    segments.ends_with(&[
        "importing".to_string(),
        "rpython_str_slice_prefix".to_string(),
    ])
}

pub(crate) fn emit_rpython_str_find_char(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    args: &[Variable],
) -> Result<Variable, String> {
    if args.len() != 3 {
        return Err(format!(
            "rpython_str_find_char expected string, char, and start, got {} arguments",
            args.len()
        ));
    }
    let result = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(result.clone()),
        kind: OpKind::Call {
            target: CallTarget::method("find", None),
            args: args.to_vec(),
            result_ty: ValueType::Int,
        },
    });
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_only_the_importing_compatibility_helper() {
        let path = |parts: &[&str]| parts.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(is_rpython_str_find_char(&path(&[
            "pyre_interpreter",
            "importing",
            "rpython_str_find_char"
        ])));
        assert!(!is_rpython_str_find_char(&path(&[
            "core", "str", "<Impl>", "find"
        ])));
        assert!(is_rpython_str_slice_prefix(&path(&[
            "pyre_interpreter",
            "importing",
            "rpython_str_slice_prefix"
        ])));
    }

    #[test]
    fn emits_the_existing_signed_rpython_find_method() {
        let mut graph = FunctionGraph::new("test_rpython_str_find_char");
        let block = graph.startblock;
        let haystack = graph
            .push_op_var(block, OpKind::ConstStr(b"a.b".to_vec()), true)
            .unwrap();
        let needle = graph
            .push_op_var(block, OpKind::ConstStr(b".".to_vec()), true)
            .unwrap();
        let start = graph.push_op_var(block, OpKind::ConstInt(1), true).unwrap();
        let result =
            emit_rpython_str_find_char(&mut graph, block, &[haystack, needle, start]).unwrap();

        assert!(graph.block(block).operations.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                result_ty: ValueType::Int,
                ..
            } if op.result.as_ref() == Some(&result) && name == "find"
        )));
    }
}
