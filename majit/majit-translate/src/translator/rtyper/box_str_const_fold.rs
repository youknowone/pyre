//! Pre-transform fold of boxed string literals.
//!
//! String views are identity aliases in the model graph: references, pointer
//! casts, and `Wtf8::new` do not emit operations. A call that boxes such a
//! view therefore receives the string literal's `Variable` directly. Calls
//! split MIR basic blocks, so the literal definition may be in a straight-line
//! predecessor rather than in the call's block.

use crate::flowspace::model::Variable;
use crate::model::{BlockId, CallTarget, FunctionGraph, LinkArg, OpKind};

const BOX_STR_CONSTANT_PATH: [&str; 3] = ["pyre_object", "unicodeobject", "box_str_constant"];

fn is_box_str_constant_call(kind: &OpKind) -> Option<&Variable> {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments },
        args,
        ..
    } = kind
    else {
        return None;
    };
    let [arg] = args.as_slice() else {
        return None;
    };
    (segments
        .iter()
        .map(String::as_str)
        .eq(BOX_STR_CONSTANT_PATH))
    .then_some(arg)
}

/// Resolve `value` to a string literal that dominates its use.
///
/// The walk accepts only straight-line control flow. At a block input it also
/// follows the sole incoming link, preserving the input-argument position.
/// Multiple predecessors or multiple incoming links make the value ambiguous
/// and stop the fold.
fn dominating_literal(
    graph: &FunctionGraph,
    use_block: BlockId,
    use_op_index: usize,
    value: &Variable,
) -> Option<Vec<u8>> {
    let mut block_id = use_block;
    let mut before = use_op_index;
    let mut value = value.clone();
    let mut seen = Vec::new();

    loop {
        if seen.contains(&block_id) {
            return None;
        }
        seen.push(block_id);

        let block = graph.block(block_id);
        if let Some(producer) = block.operations[..before]
            .iter()
            .rev()
            .find(|op| op.result.as_ref() == Some(&value))
        {
            return match &producer.kind {
                OpKind::ConstStr(bytes) => Some(bytes.clone()),
                _ => None,
            };
        }

        if let Some(slot) = block.inputargs.iter().position(|arg| arg == &value) {
            let predecessors = graph.predecessors(block_id);
            let [predecessor] = predecessors.as_slice() else {
                return None;
            };
            let predecessor_block = graph.block(*predecessor);
            let mut incoming = predecessor_block
                .exits
                .iter()
                .filter(|link| link.target == block_id);
            let link = incoming.next()?;
            if incoming.next().is_some() {
                return None;
            }
            let LinkArg::Value(incoming_value) = link.args.get(slot)? else {
                return None;
            };
            value = incoming_value.clone();
            block_id = *predecessor;
            before = predecessor_block.operations.len();
            continue;
        }

        let predecessors = graph.predecessors(block_id);
        let [predecessor] = predecessors.as_slice() else {
            return None;
        };
        block_id = *predecessor;
        before = graph.block(block_id).operations.len();
    }
}

/// Replace `box_str_constant` calls over proven string literals with the
/// literal constant while preserving each call's result variable.
pub fn fold_box_str_constants(graph: &mut FunctionGraph) {
    let mut rewrites = Vec::new();
    for block in &graph.blocks {
        for (op_index, op) in block.operations.iter().enumerate() {
            let Some(arg) = is_box_str_constant_call(&op.kind) else {
                continue;
            };
            if let Some(bytes) = dominating_literal(graph, block.id, op_index, arg) {
                rewrites.push((block.id, op_index, bytes));
            }
        }
    }

    for (block_id, op_index, bytes) in rewrites {
        graph.block_mut(block_id).operations[op_index].kind = OpKind::ConstStr(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ValueType;

    fn str_const_call(payload: &str) -> OpKind {
        OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["__str_const".to_string(), payload.to_string()],
            },
            args: vec![],
            result_ty: ValueType::Ref(None),
        }
    }

    fn box_str_constant_call(arg: Variable) -> OpKind {
        OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: BOX_STR_CONSTANT_PATH.map(str::to_string).to_vec(),
            },
            args: vec![arg],
            result_ty: ValueType::Ref(None),
        }
    }

    #[test]
    fn folds_frontend_literal_view_across_straight_line_blocks() {
        let mut graph = FunctionGraph::new("box_literal");
        let entry = graph.startblock;
        let literal = graph
            .push_op_var(entry, str_const_call("__instancecheck__"), true)
            .expect("string literal must produce a value");
        let call_block = graph.create_block();
        graph.set_goto(entry, call_block, vec![]);
        let boxed = graph
            .push_op_var(call_block, box_str_constant_call(literal.clone()), true)
            .expect("box call must produce a value");

        crate::translator::rtyper::str_const_fold::fold_str_consts(&mut graph);
        fold_box_str_constants(&mut graph);

        assert_eq!(
            graph.block(entry).operations[0].kind,
            OpKind::ConstStr(b"__instancecheck__".to_vec())
        );
        let folded = &graph.block(call_block).operations[0];
        assert_eq!(folded.result.as_ref(), Some(&boxed));
        assert_eq!(folded.kind, OpKind::ConstStr(b"__instancecheck__".to_vec()));
    }

    #[test]
    fn leaves_dynamic_argument_call_unchanged() {
        let mut graph = FunctionGraph::new("box_dynamic");
        let entry = graph.startblock;
        let input = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "value".to_string(),
                    ty: ValueType::Str,
                    class_root: None,
                },
                true,
            )
            .expect("input must produce a value");
        let original = box_str_constant_call(input);
        graph.push_op_var(entry, original.clone(), true);

        fold_box_str_constants(&mut graph);

        assert_eq!(graph.block(entry).operations[1].kind, original);
    }
}
