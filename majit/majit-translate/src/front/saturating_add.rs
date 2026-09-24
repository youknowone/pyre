//! `{u64,usize}::saturating_add(a, b)` → unsigned clamp diamond.
//!
//! ## Positioning
//!
//! `core::num::<Impl>::saturating_add` is a foreign leaf whose body is Opaque
//! in the LLBC, so the caller emits a residual `saturating_add` call.  Every
//! portal-reachable site named by the census (`W_ListObject::object_grow`,
//! `IntArray::grow`, `UnicodeArray::{empty,with_capacity}`) is word-sized
//! unsigned.  Wrapping add in the word bank plus `uint_lt(sum, a)` is the
//! carry test; the overflow arm yields the max of the op's `Unsigned`
//! lowleveltype (`unsigned_repr`: `u64::MAX` when the word is 8 bytes,
//! `u32::MAX` when it is 4).
//!
//! A narrow `u16`/`u32` saturating add is **not** that test: `u32::MAX + 1`
//! does not wrap in the word bank, so `sum < a` stays false and the rewrite
//! would return `2^32` where the source saturates at `u32::MAX`.  Capture
//! therefore keeps the dest-atom gate in `front::mir`; this pass only
//! rewrites residual calls whose result Variable was recorded there.
//!
//! Signed `saturating_add` clamps at `TYPE_MAX` with a different diamond
//! (`add_ovf` + a const) and is left residual.
//!
//! ## The rewrite
//!
//! Block A holds the residual `saturating_add` call producing `r`.  The
//! rewrite:
//! 1. drops the call and emits `sum = a + b`, `ovf = uint_lt(sum, a)`;
//! 2. the true arm (`ovf`) builds `r = Unsigned max`;
//! 3. the false arm builds `r = sum`;
//! 4. both arms forward to B.
//!
//! Fail-safe: a site whose producer is not a 2-arg `saturating_add` Call
//! is left untouched. `int_add` and `int_sub` are separate ops, so add
//! sites are [`SaturatingAddSite`]s, not `SaturatingSubSite`s.

use crate::flowspace::model::Variable;
use crate::front::bool_then::{close_goto_mixed, map_source, reproduce_exit_args};
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// A recognized word-sized `{usize,u64}::saturating_add(a, b)` call.
/// `int_add` carries its own op, so this is not a [`crate::front::saturating_sub::SaturatingSubSite`].
#[derive(Clone)]
pub(crate) struct SaturatingAddSite {
    /// The `saturating_add` call result. Operands are read from that call.
    pub result_var: Variable,
}

pub(crate) fn rewire_saturating_add_call_sites(
    graph: &mut FunctionGraph,
    sites: &[SaturatingAddSite],
) -> usize {
    rewire_saturating_add_call_sites_for(graph, sites, crate::layout::target_word_size())
}

pub(crate) fn rewire_saturating_add_call_sites_for(
    graph: &mut FunctionGraph,
    sites: &[SaturatingAddSite],
    word_bytes: usize,
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_saturating_add_site(graph, &site.result_var, word_bytes) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {}
        }
    }
    rewritten
}

fn is_saturating_add_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    segments.last().map(String::as_str) == Some("saturating_add")
        && segments.iter().any(|s| s.as_str() == "num")
}

fn rewire_one_saturating_add_site(
    graph: &mut FunctionGraph,
    result_var: &Variable,
    word_bytes: usize,
) -> Result<(), String> {
    let name = graph.name.clone();
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(result_var))
        })
        .ok_or_else(|| format!("{name}: saturating_add result var has no producer block"))?;

    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(result_var))
        .ok_or_else(|| format!("{name}: saturating_add call op not found in block {a}"))?;
    if ci + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: saturating_add call is not the last op of block {a}"
        ));
    }

    let (addend_a, addend_b) = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call { target, args, .. }
            if args.len() == 2 && is_saturating_add_target(target) =>
        {
            (
                args[0].clone().into_variable(),
                args[1].clone().into_variable(),
            )
        }
        other => {
            return Err(format!(
                "{name}: saturating_add producer op is not a 2-arg saturating_add call: {other:?}"
            ));
        }
    };

    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!(
            "{name}: saturating_add call block {a} does not have a single exit"
        ));
    };
    if exit.exitcase.is_some() || exit.last_exception.is_some() || exit.last_exc_value.is_some() {
        return Err(format!(
            "{name}: saturating_add call block {a} exit is not a plain goto"
        ));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;

    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != *result_var
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.remove(ci);
    let sum = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(sum.clone()),
        kind: OpKind::BinOp {
            op: "add".to_string(),
            lhs: addend_a.clone(),
            rhs: addend_b,
            result_ty: ValueType::Unsigned,
        },
    });
    let ovf = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(ovf.clone()),
        kind: OpKind::BinOp {
            op: "uint_lt".to_string(),
            lhs: sum.clone(),
            rhs: addend_a,
            result_ty: ValueType::Int,
        },
    });

    let mut else_sources = carried.clone();
    if !else_sources.contains(&sum) {
        else_sources.push(sum.clone());
    }
    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(carried.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(else_sources.len());

    let then_result = graph.alloc_value_var();
    graph.block_mut(then_bb).operations.push(SpaceOperation {
        result: Some(then_result.clone()),
        kind: OpKind::ConstUInt(crate::front::checked_arith_uint::unsigned_word_max(
            word_bytes,
        )),
    });
    let then_link_args = reproduce_exit_args(
        &saved_exit,
        result_var,
        &then_result,
        &carried,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_link_args);

    let sum_in_else = map_source(&else_sources, &else_inputs, &sum)
        .ok_or_else(|| format!("{name}: wrapping sum not threaded into else arm"))?;
    let else_link_args = reproduce_exit_args(
        &saved_exit,
        result_var,
        &sum_in_else,
        &else_sources,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_link_args);

    graph.set_branch(a_id, ovf, then_bb, carried, else_bb, else_sources);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn emit_call(g: &mut FunctionGraph, a: crate::model::BlockId, args: Vec<Variable>) -> Variable {
        g.push_op_var(
            a,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec![
                        "core".into(),
                        "num".into(),
                        "<Impl>".into(),
                        "saturating_add".into(),
                    ],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(args),
                result_ty: ValueType::Unsigned,
            },
            true,
        )
        .unwrap()
    }

    #[test]
    fn rewrite_lifts_saturating_add_to_carry_clamp_diamond() {
        let mut g = FunctionGraph::new("test_saturating_add");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = emit_call(&mut g, a, vec![av.clone(), bv.clone()]);

        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_add_call_sites(
            &mut g,
            &[SaturatingAddSite {
                result_var: r.clone(),
            }],
        );
        assert_eq!(rewritten, 1, "the saturating_add site must be rewritten");

        assert!(
            !g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("saturating_add")
            )),
            "residual saturating_add call removed from A"
        );
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "add"
            )),
            "A wrapping-adds a + b"
        );
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "uint_lt"
            )),
            "A tests the unsigned carry"
        );
        assert_eq!(g.blocks[a.0].exits.len(), 2, "A branches to MAX / sum arms");
        let maxes = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| matches!(&op.kind, OpKind::ConstUInt(u64::MAX)))
            .count();
        assert!(maxes >= 1, "the overflow arm builds u64::MAX");
    }

    #[test]
    fn rewrite_declines_saturating_sub_result_var() {
        let mut g = FunctionGraph::new("test_saturating_add_decline_sub");
        let a = g.startblock;
        let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
        let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let r = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "core".into(),
                            "num".into(),
                            "<Impl>".into(),
                            "saturating_sub".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![av, bv]),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let (b, _) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![r.clone()]);

        let rewritten = rewire_saturating_add_call_sites(
            &mut g,
            &[SaturatingAddSite {
                result_var: r.clone(),
            }],
        );
        assert_eq!(rewritten, 0, "a saturating_sub producer declines");
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("saturating_sub")
            )),
            "residual saturating_sub call is left untouched"
        );
    }

    fn overflow_const(g: &FunctionGraph) -> u64 {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .find_map(|op| match op.kind {
                OpKind::ConstUInt(n) => Some(n),
                _ => None,
            })
            .expect("overflow clamp const")
    }

    #[test]
    fn saturating_add_clamps_at_unsigned_lowleveltype_max() {
        for word_bytes in [8usize, 4] {
            let unsigned_max = match word_bytes {
                8 => u64::MAX,
                4 => u32::MAX as u64,
                _ => unreachable!(),
            };
            let mut g = FunctionGraph::new("test_saturating_add_width");
            let a = g.startblock;
            let av = g.push_op_var(a, OpKind::ConstInt(7), true).unwrap();
            let bv = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
            let r = emit_call(&mut g, a, vec![av, bv]);
            let (b, _b_args) = g.create_block_with_arg_vars(1);
            g.set_return(b, None);
            g.set_goto(a, b, vec![r.clone()]);
            let rewritten = rewire_saturating_add_call_sites_for(
                &mut g,
                &[SaturatingAddSite { result_var: r }],
                word_bytes,
            );
            assert_eq!(rewritten, 1);
            assert_eq!(overflow_const(&g), unsigned_max);
        }
    }
}
