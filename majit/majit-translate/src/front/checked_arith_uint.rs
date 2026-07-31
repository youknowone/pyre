//! `usize::checked_{add,mul}()` → native unsigned-overflow ops + a
//! virtualized `Option`.
//!
//! ## Positioning
//!
//! `core::num::<Impl>::checked_add` / `checked_mul` on **unsigned** operands
//! are Opaque core bodies (Charon cannot extract `core`), so the caller
//! carries a residual `checked_*` call the rtyper census cannot type.
//! [`crate::front::checked_arith`] lowers the **signed** match-shape into a
//! `*_ovf` op + `OverflowError` edge, but that path is wrong for unsigned
//! arithmetic: RPython has no unsigned overflow op (`r_uint` wraps), so a
//! signed `mul_ovf` would diverge for operands in `(i64::MAX, usize::MAX]`.
//!
//! The unsigned overflow tests have native, wrapping-safe forms:
//!   - `checked_mul(x, y)` overflows iff the high word of the widening
//!     product is non-zero: `uint_mul_high(x, y) != 0`.
//!   - `checked_add(x, y)` overflows iff the sum wrapped below an addend
//!     (carry): `uint_lt(x + y, x)`.
//!
//! ## The rewrite (`rewire_one_checked_arith_uint_site`)
//!
//! Block A's last op is the residual `opt = checked_*(x, y)` call.  The
//! rewrite drops that call and, **in place**, emits the native sequence plus a
//! virtualized `Option` reusing `opt` as the ctor result:
//!   - `checked_mul`: `lo = mul(x, y)`, `hi = uint_mul_high(x, y)`,
//!     `disc = eq(hi, 0)`, value = `lo`.
//!   - `checked_add`: `sum = add(x, y)`, `ovf = uint_lt(sum, x)`,
//!     `disc = eq(ovf, 0)`, value = `sum`.
//!   - `opt = Some/None` aggregate with `__discriminant = disc` (dynamic,
//!     `1` = `Some` when no overflow, `0` = `None` on overflow) and
//!     `__pos_0 = value`.
//!
//! Unlike [`crate::front::checked_arith`] / [`crate::front::option_try`] this
//! is **order-independent**: it produces a valid virtualized `Option` value
//! that the downstream `?` / match / let-else consumes unchanged — no diamond
//! matching, no exit rewiring.
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the census Skip / legacy-walker
//! fallback is unchanged.  Gating (an unsigned `checked_{add,mul}` producer
//! still present as its `Call`) is applied both at the recording site (only
//! unsigned operands are recorded) and here (the producer must still be the
//! `Call`, so a site [`crate::front::checked_arith`] already rewrote is
//! skipped).

use crate::flowspace::model::Variable;
use crate::front::bool_then::emit_option_variant_dynamic;
use crate::model::{CallTarget, FunctionGraph, OpKind, SpaceOperation, ValueType};

/// A recorded unsigned `checked_{add,mul}` call whose result is an
/// `Option<T>`, captured during body lowering with the `Option`/`Some` owners
/// and payload type resolved from the destination type.
#[derive(Clone)]
pub(crate) struct CheckedArithUintSite {
    /// The `checked_*` call result (the `Option<T>` value) — locates block A
    /// and is reused as the virtualized `Option` ctor result.
    pub opt: Variable,
    /// The `Option` enum root `name_path` — the `__discriminant` field owner
    /// and the ctor owner.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload owner.
    pub some_owner: String,
    /// The `Option` payload `T` — the `Some::__pos_0` field kind.
    pub payload_ty: ValueType,
}

/// Rewrite every recorded unsigned `checked_{add,mul}` site into the native
/// overflow-test + virtualized `Option` shape.  Returns the number of sites
/// rewritten; declined sites keep their residual call (census Skip).
pub(crate) fn rewire_checked_arith_uint_sites(
    graph: &mut FunctionGraph,
    sites: &[CheckedArithUintSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_checked_arith_uint_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                if std::env::var_os("PYRE_MIR_FRONTEND_DEBUG").is_some() {
                    eprintln!(
                        "[checked_arith_uint] {} decline at {:?}: {_decline}",
                        graph.name, site.opt
                    );
                }
                // Leave the residual `checked_*` call; the unregistered callee
                // makes the rtyper census Skip this graph (no regression).
            }
        }
    }
    rewritten
}

/// The unsigned `checked_*` operators this pass lowers.  `checked_sub` is
/// intentionally absent: unsigned subtraction underflow is a `uint_lt(x, y)`
/// guard, but no census site spells it in the `?`-shape, so it stays a
/// residual.
#[derive(Clone, Copy)]
enum UintArith {
    Add,
    Mul,
}

impl UintArith {
    fn from_leaf(leaf: &str) -> Option<Self> {
        match leaf {
            "checked_add" => Some(UintArith::Add),
            "checked_mul" => Some(UintArith::Mul),
            _ => None,
        }
    }
}

fn rewire_one_checked_arith_uint_site(
    graph: &mut FunctionGraph,
    site: &CheckedArithUintSite,
) -> Result<(), String> {
    let name = graph.name.clone();
    let opt = &site.opt;

    // Block A: the `checked_*()` residual call producing `opt`, closed by
    // `lower_call` with a single forwarding exit.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(opt))
        })
        .ok_or_else(|| format!("{name}: checked_* result var has no producer block"))?;

    // The call must still be A's last op (a `[..]::checked_add/checked_mul`
    // 2-arg FunctionPath call); a site `checked_arith`'s first pass already
    // rewrote is now a `*_ovf` BinOp producer — skip it.  The overflow op is
    // resolved here (before any mutation) so an unsupported leaf declines
    // without touching the graph.
    let call_idx = graph.blocks[a].operations.len() - 1;
    let (lhs, rhs, arith) = match &graph.blocks[a].operations[call_idx] {
        SpaceOperation {
            result: Some(r),
            kind:
                OpKind::Call {
                    target: target @ CallTarget::FunctionPath { segments },
                    args,
                    ..
                },
        } if r == opt
            && args.len() == 2
            && crate::front::checked_arith::is_checked_arith_target(target) =>
        {
            let leaf = segments.last().map(String::as_str).unwrap_or_default();
            let arith = UintArith::from_leaf(leaf).ok_or_else(|| {
                format!("{name}: unsigned checked lowering does not handle {leaf}")
            })?;
            (args[0].clone(), args[1].clone(), arith)
        }
        _ => {
            return Err(format!(
                "{name}: block {a} last op is not the 2-arg checked_* call producing {opt:?}"
            ));
        }
    };

    // --- All structural validation passed; mutate the graph. ---

    let a_id = graph.blocks[a].id;
    // Drop the residual call (A's last op) so `opt` is produced solely by the
    // virtualized ctor appended below, then re-emit the native sequence in
    // place of the removed call.
    graph.blocks[a].operations.truncate(call_idx);

    // The two operators differ only in the payload value and the overflow
    // test; build the `Option` once from the resulting `(value, disc)`.
    let (value, disc) = match arith {
        UintArith::Mul => {
            let lo = push_binop(
                graph,
                a_id,
                "mul",
                lhs.clone(),
                rhs.clone(),
                ValueType::Unsigned,
            );
            let hi = push_binop(graph, a_id, "uint_mul_high", lhs, rhs, ValueType::Unsigned);
            let disc = push_no_overflow_disc(graph, a_id, hi);
            (lo, disc)
        }
        UintArith::Add => {
            let sum = push_binop(
                graph,
                a_id,
                "add",
                lhs.clone(),
                rhs.clone(),
                ValueType::Unsigned,
            );
            // Unsigned carry: the wrapped sum is strictly below an addend.
            let ovf = push_binop(graph, a_id, "uint_lt", sum.clone(), lhs, ValueType::Int);
            let disc = push_no_overflow_disc(graph, a_id, ovf);
            (sum, disc)
        }
    };

    emit_option_variant_dynamic(
        graph,
        a_id,
        opt.clone(),
        &site.option_owner,
        disc,
        Some((&site.some_owner, value, site.payload_ty.clone())),
    );
    Ok(())
}

/// `disc = eq(test, 0)` — the `Option` tag: `1` (`Some`) when the overflow
/// `test` is `0` (no overflow), `0` (`None`) when it is non-zero.
fn push_no_overflow_disc(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    test: Variable,
) -> Variable {
    let zero = push_const_int(graph, block, 0);
    push_binop(graph, block, "eq", test, zero, ValueType::Int)
}

pub(crate) fn push_binop(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    op: &str,
    lhs: Variable,
    rhs: Variable,
    result_ty: ValueType,
) -> Variable {
    let res = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::BinOp {
            op: op.to_string(),
            lhs,
            rhs,
            result_ty,
        },
    });
    res
}

pub(crate) fn push_const_int(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    value: i64,
) -> Variable {
    let res = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::ConstInt(value),
    });
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn checked_target(leaf: &str) -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "num", "<Impl>", leaf]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    fn site_for(opt: &Variable) -> CheckedArithUintSite {
        CheckedArithUintSite {
            opt: opt.clone(),
            option_owner: "core::option::Option".to_string(),
            some_owner: "core::option::Option::Some".to_string(),
            payload_ty: ValueType::Unsigned,
        }
    }

    /// The op names / owners the virtualized `Option` tail carries, in order.
    fn tail_binops(graph: &FunctionGraph, a: usize) -> Vec<String> {
        graph.blocks[a]
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::BinOp { op: name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    fn build_checked_site(leaf: &str) -> (FunctionGraph, Variable, usize) {
        let mut g = FunctionGraph::new("test_checked_uint");
        let a = g.startblock;
        let x = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let y = g.push_op_var(a, OpKind::ConstInt(8), true).unwrap();
        // Block A's last op = the residual unsigned `checked_*(x, y)`.
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(leaf),
                    args: vec![x, y],
                    result_ty: ValueType::Ref(Some("core::option::Option".into())),
                },
                true,
            )
            .unwrap();
        // A continuation consuming `opt` (single forwarding goto exit).
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(a, cont, vec![opt.clone()]);
        (g, opt, a.0)
    }

    #[test]
    fn checked_mul_lowers_to_uint_mul_high_and_virtualized_option() {
        let (mut g, opt, a) = build_checked_site("checked_mul");
        let rewritten = rewire_checked_arith_uint_sites(&mut g, &[site_for(&opt)]);
        assert_eq!(rewritten, 1, "the unsigned checked_mul site must rewrite");

        // No residual `checked_*` call survives in block A.
        assert!(
            !g.blocks[a].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { .. },
                    ..
                }
            )),
            "the residual checked_mul call must be gone"
        );
        // Overflow tests: `mul` (low word) + `uint_mul_high` (high word) + `eq`.
        assert_eq!(tail_binops(&g, a), vec!["mul", "uint_mul_high", "eq"]);

        // `opt` is now produced by the transparent Option ctor.
        let ctor = g.blocks[a]
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&opt))
            .expect("opt must be produced");
        assert!(
            matches!(
                &ctor.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            ),
            "opt must be the synthetic transparent Option ctor result"
        );
        // The aggregate writes the dynamic `__discriminant` and the `__pos_0`
        // payload keyed to the Some owner.
        let disc_write = g.blocks[a].operations.iter().any(|op| {
            matches!(&op.kind, OpKind::FieldWrite { base, field, .. }
                if *base == opt && field.name == "__discriminant")
        });
        let pos0_write = g.blocks[a].operations.iter().any(|op| {
            matches!(&op.kind, OpKind::FieldWrite { base, field, .. }
                if *base == opt && field.name == "__pos_0"
                    && field.owner_root.as_deref() == Some("core::option::Option::Some"))
        });
        assert!(disc_write, "the __discriminant tag write must be present");
        assert!(pos0_write, "the __pos_0 payload write must be present");
    }

    #[test]
    fn checked_add_lowers_to_uint_lt_carry_test() {
        let (mut g, opt, a) = build_checked_site("checked_add");
        let rewritten = rewire_checked_arith_uint_sites(&mut g, &[site_for(&opt)]);
        assert_eq!(rewritten, 1, "the unsigned checked_add site must rewrite");
        // Carry test: `add` (wrapping sum) + `uint_lt(sum, x)` + `eq`.
        assert_eq!(tail_binops(&g, a), vec!["add", "uint_lt", "eq"]);
    }

    #[test]
    fn declines_when_producer_is_no_longer_the_checked_call() {
        // A site `checked_arith` already rewrote (producer is now a `mul_ovf`
        // BinOp, not the `Call`) must be left untouched.
        let mut g = FunctionGraph::new("test_already_rewritten");
        let a = g.startblock;
        let x = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let y = g.push_op_var(a, OpKind::ConstInt(8), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::BinOp {
                    op: "mul_ovf".to_string(),
                    lhs: x,
                    rhs: y,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(a, cont, vec![opt.clone()]);

        let rewritten = rewire_checked_arith_uint_sites(&mut g, &[site_for(&opt)]);
        assert_eq!(rewritten, 0, "a non-Call producer must decline");
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "mul_ovf"
            )),
            "the mul_ovf producer must survive untouched"
        );
    }
}
