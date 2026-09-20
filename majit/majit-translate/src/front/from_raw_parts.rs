//! `slice::from_raw_parts{,_mut}(chars, len)` of a length-prefixed GC string
//! header → identity on that header.
//!
//! ## Positioning
//!
//! `core::slice::raw::from_raw_parts` is Opaque in the LLBC.  A handful of
//! adapters already lower it during body lowering by aliasing the destination
//! to the block header — `bytes_block_chars` (address-taken `chars` FieldRead),
//! `BytesArray`/`UnicodeArray::as_slice` (items-view), `RBigInt::digits`
//! (capacity-length typed items).  Those folds key on the *enclosing function
//! name*, so a helper with the same callee shape
//! (`utf8_payload_bytes`: `from_raw_parts(header.add(CHARS_OFFSET), len)`)
//! never reaches them and stays a Phase-A leaf.
//!
//! The translated STR / `BytesBlock` is a length-prefixed GC struct whose
//! `chars` array begins at the descr `base_size`.  A slice over that array is
//! the header itself in the list/string model: `__len` and byte reads dispatch
//! through `arraylen_gc` / `strgetitem`.  Aliasing the slice to the header is
//! therefore the same identity those name-gated folds already emit.
//!
//! Arbitrary `from_raw_parts(p, n)` is not that shape.  `n` may be a logical
//! length below capacity (object-list pop, a buffer exporter's window); dropping
//! it would let a consumer observe capacity.  This pass only fires when the
//! pointer is recovered as:
//! - an address-taken `chars` FieldRead (the `bytes_block_chars` shape), or
//! - a `<*const u8>::add(header, 2*word)` whose length operand is the same
//!   header's `bh_lowlevel_string_len` / `length` field (`utf8_payload_bytes`).
//!
//! Fail-safe: any other residual is left untouched.

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, FunctionGraph, OpKind, SpaceOperation, ValueType};

/// Rewrite every residual `from_raw_parts{,_mut}` whose pointer is a
/// length-prefixed GC-string header view into `same_as(header)`.  Returns
/// the number of sites rewritten.
pub(crate) fn rewire_from_raw_parts_sites(graph: &mut FunctionGraph) -> usize {
    let mut sites: Vec<(usize, usize, Variable)> = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            let Some(header) = header_for_from_raw_parts(graph, op) else {
                continue;
            };
            if !var_live_in_block(graph, bi, &header) {
                continue;
            }
            sites.push((bi, oi, header));
        }
    }
    let rewritten = sites.len();
    for (bi, oi, header) in sites {
        let result_ty = match &graph.blocks[bi].operations[oi].kind {
            OpKind::Call { result_ty, .. } => result_ty.clone(),
            _ => ValueType::Ref(None),
        };
        graph.blocks[bi].operations[oi].kind = OpKind::UnaryOp {
            op: "same_as".to_string(),
            operand: header,
            result_ty,
        };
    }
    rewritten
}

fn header_for_from_raw_parts(graph: &FunctionGraph, op: &SpaceOperation) -> Option<Variable> {
    let OpKind::Call { target, args, .. } = &op.kind else {
        return None;
    };
    if args.len() != 2 || !is_from_raw_parts_target(target) {
        return None;
    }
    let ptr = args[0].clone().into_variable();
    let len = args[1].clone().into_variable();
    if let Some(header) = chars_field_header(graph, &ptr) {
        return Some(header);
    }
    let header = ptr_add_chars_offset_header(graph, &ptr)?;
    len_names_same_header(graph, &len, &header).then_some(header)
}

fn is_from_raw_parts_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    matches!(
        segments.last().map(String::as_str),
        Some("from_raw_parts" | "from_raw_parts_mut")
    ) && segments.iter().any(|s| s == "slice" || s == "raw")
}

fn is_ptr_add_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    matches!(
        segments.last().map(String::as_str),
        Some("add" | "wrapping_add")
    ) && segments
        .iter()
        .any(|s| s == "const_ptr" || s == "mut_ptr" || s == "ptr")
}

fn producer<'a>(graph: &'a FunctionGraph, var: &Variable) -> Option<&'a SpaceOperation> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .find(|op| op.result.as_ref() == Some(var))
}

fn chars_field_header(graph: &FunctionGraph, ptr: &Variable) -> Option<Variable> {
    match &producer(graph, ptr)?.kind {
        OpKind::FieldRead { base, field, .. }
            if field.name == "chars" && field.taken_by_address =>
        {
            Some(base.clone())
        }
        _ => None,
    }
}

fn is_chars_offset(n: i64) -> bool {
    // STR / rpy_string chars sit two words past the header (`hash`, then `len`).
    // Translate-time `usize` is the host's; wasm32 leftovers use a 4-byte word.
    n == 8 || n == 16
}

fn const_int_of(graph: &FunctionGraph, var: &Variable) -> Option<i64> {
    match &producer(graph, var)?.kind {
        OpKind::ConstInt(n) => Some(*n),
        OpKind::ConstUInt(n) => i64::try_from(*n).ok(),
        _ => None,
    }
}

fn ptr_add_chars_offset_header(graph: &FunctionGraph, ptr: &Variable) -> Option<Variable> {
    match &producer(graph, ptr)?.kind {
        OpKind::Call { target, args, .. } if args.len() == 2 && is_ptr_add_target(target) => {
            let offset = args[1].clone().into_variable();
            is_chars_offset(const_int_of(graph, &offset)?).then(|| args[0].clone().into_variable())
        }
        OpKind::BinOp { op, lhs, rhs, .. } if op == "add" || op == "int_add" => {
            is_chars_offset(const_int_of(graph, rhs)?).then(|| lhs.clone())
        }
        _ => None,
    }
}

fn len_names_same_header(graph: &FunctionGraph, len: &Variable, header: &Variable) -> bool {
    let Some(op) = producer(graph, len) else {
        return false;
    };
    match &op.kind {
        OpKind::FieldRead { base, field, .. }
            if (field.name == "length" || field.name == "len") && base == header =>
        {
            true
        }
        OpKind::Call { target, args, .. } if !args.is_empty() => {
            let leaf = match target {
                CallTarget::FunctionPath { segments, .. } => segments.last().map(String::as_str),
                CallTarget::Method { name, .. } => Some(name.as_str()),
                _ => None,
            };
            let names_len = leaf.is_some_and(|leaf| {
                leaf == "bh_lowlevel_string_len" || leaf.ends_with("lowlevel_string_len")
            });
            names_len && args[0].clone().into_variable() == *header
        }
        _ => false,
    }
}

fn var_live_in_block(graph: &FunctionGraph, block_idx: usize, var: &Variable) -> bool {
    let block = &graph.blocks[block_idx];
    block.inputargs.iter().any(|input| input == var)
        || block
            .operations
            .iter()
            .any(|op| op.result.as_ref() == Some(var))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FieldDescriptor;

    fn from_raw_parts_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "slice", "raw", "from_raw_parts"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn ptr_add_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "ptr", "const_ptr", "<Impl>", "add"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn residual_from_raw_parts(g: &FunctionGraph) -> bool {
        g.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target, .. } if is_from_raw_parts_target(target)
            )
        })
    }

    #[test]
    fn chars_field_from_raw_parts_aliases_header() {
        let mut g = FunctionGraph::new("test_from_raw_parts_chars");
        let a = g.startblock;
        let header = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let chars = g
            .push_op_var(
                a,
                OpKind::FieldRead {
                    base: header.clone(),
                    field: FieldDescriptor {
                        name: "chars".to_string(),
                        owner_root: Some("BytesBlock".to_string()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: true,
                    },
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        let len = g.push_op_var(a, OpKind::ConstInt(4), true).unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![chars, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 1);
        assert!(
            !residual_from_raw_parts(&g),
            "header-view from_raw_parts must be gone"
        );
        assert!(g.blocks[a.0].operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::UnaryOp { op, operand, .. } if op == "same_as" && operand == &header
            )
        }));
    }

    #[test]
    fn utf8_payload_add_from_raw_parts_aliases_header() {
        let mut g = FunctionGraph::new("test_from_raw_parts_utf8");
        let a = g.startblock;
        let header = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let offset = g.push_op_var(a, OpKind::ConstInt(16), true).unwrap();
        let chars = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: ptr_add_target(),
                    args: crate::model::call_args(vec![header.clone(), offset]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let len = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: ["pyre_object", "lowlevel_string", "bh_lowlevel_string_len"]
                            .iter()
                            .map(|s| s.to_string())
                            .collect(),
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![header.clone()]),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![chars, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 1);
        assert!(!residual_from_raw_parts(&g));
        assert!(g.blocks[a.0].operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::UnaryOp { op, operand, .. } if op == "same_as" && operand == &header
            )
        }));
    }

    #[test]
    fn arbitrary_from_raw_parts_is_left_residual() {
        let mut g = FunctionGraph::new("test_from_raw_parts_arbitrary");
        let a = g.startblock;
        let ptr = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let len = g.push_op_var(a, OpKind::ConstInt(4), true).unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![ptr, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 0);
        assert!(
            residual_from_raw_parts(&g),
            "an unproven ptr+len view must stay residual"
        );
    }
}
