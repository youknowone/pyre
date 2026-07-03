//! `bigint::BigInt::div_rem()` → modeled `(quotient, remainder)` tuple.
//!
//! ## Positioning
//!
//! malachite's `BigInt::div_rem(&self, &Self) -> (Self, Self)` returns a
//! `(BigInt, BigInt)` tuple whose body is Opaque in the LLBC, so the caller
//! emits a residual `div_rem` FunctionPath call — an unregistered callee the
//! rtyper census Skips.  The value flows into a downstream `let (q, r) = …`
//! destructure the front lowers as `__pos_0` / `__pos_1` tuple field reads.
//! This pass supplies that producer, sourcing each half from a
//! `#[dont_look_inside]` residual and reassembling the pair as the same
//! synthetic-`Tuple` aggregate `Rvalue::Aggregate` emits:
//!
//! ```text
//!     t = num.div_rem(&den)                  // residual `div_rem` call
//! becomes
//!     q = jit_bigint_div(num, den)           // truncated quotient
//!     r = jit_bigint_rem(num, den)           // truncated remainder
//!     t = Tuple { __pos_0: q, __pos_1: r }
//! ```
//!
//! `num` / `den` are the raw `*mut BigInt` the front models a `BigInt` value
//! as (a classdef-less `SomeInstance` GcRef), matching both helpers' i64-over-
//! `*mut BigInt` ABI; each helper allocates its result in the collecting
//! nursery.  Sibling of [`crate::front::bigint_to_i64`] — the same branchless
//! in-place splice, building a two-field `Ref` tuple instead of an `Option`.
//!
//! It is **fail-safe**: a site whose producer op is not the expected 2-arg
//! residual call is left untouched, and the unregistered `div_rem` callee
//! keeps the rtyper census Skip (no regression vs the legacy walker).

use crate::flowspace::model::Variable;
use crate::model::{
    CallTarget, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

/// The `#[dont_look_inside]` residuals the synth calls, spelled to match their
/// `jit_fnaddr` bindings.
const DIV_PATH: [&str; 4] = [
    "pyre_interpreter",
    "objspace",
    "descroperation",
    "jit_bigint_div",
];
const REM_PATH: [&str; 4] = [
    "pyre_interpreter",
    "objspace",
    "descroperation",
    "jit_bigint_rem",
];

/// A recognized `bigint::BigInt::div_rem()` call site captured during body
/// lowering (`front::mir` `recognize_bigint_div_rem_site`).  The synthetic
/// `Tuple` owner is a constant, so the site carries only the result var.
#[derive(Clone)]
pub(crate) struct BigIntDivRemSite {
    /// The `div_rem` call result (the `(BigInt, BigInt)` tuple) — locates the
    /// producer op.
    pub result_var: Variable,
}

/// Rewrite every recorded `bigint::BigInt::div_rem()` call site into the
/// modeled quotient/remainder tuple.  Fail-safe: a site whose producer op is
/// not the expected 2-arg residual call is left untouched (Skip).  Returns the
/// number of sites rewritten.
pub(crate) fn rewire_bigint_div_rem_call_sites(
    graph: &mut FunctionGraph,
    sites: &[BigIntDivRemSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_bigint_div_rem_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                // Leave the residual `div_rem` call; the unregistered callee
                // keeps the rtyper census Skip for this graph.
            }
        }
    }
    rewritten
}

fn rewire_one_bigint_div_rem_site(
    graph: &mut FunctionGraph,
    site: &BigIntDivRemSite,
) -> Result<(), String> {
    let name = graph.name.clone();
    let (a, call_idx) = graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(bi, b)| {
            b.operations
                .iter()
                .position(|op| op.result.as_ref() == Some(&site.result_var))
                .map(|oi| (bi, oi))
        })
        .ok_or_else(|| format!("{name}: bigint div_rem result var has no producer op"))?;

    // The producer must be the 2-arg residual call (numerator, denominator).
    let (num, den) = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call { args, .. } if args.len() == 2 => (args[0].clone(), args[1].clone()),
        other => {
            return Err(format!(
                "{name}: bigint div_rem producer op is not a 2-arg call: {other:?}"
            ));
        }
    };

    // --- Structural validation passed; splice the producer. ---
    // Key the rebuilt tuple's `__pos_N` owner off the destructure reads so
    // the writes land on the same (possibly per-shape-suffixed) classdef.
    let owner = graph.tuple_owner_for_var(&site.result_var);
    let q = graph.alloc_value_var();
    let r = graph.alloc_value_var();
    let inserts = build_div_rem_tuple(&site.result_var, num, den, q, r, &owner);

    let ops = &mut graph.blocks[a].operations;
    ops.remove(call_idx);
    for (offset, op) in inserts.into_iter().enumerate() {
        ops.insert(call_idx + offset, op);
    }
    Ok(())
}

/// A residual `CallTarget::FunctionPath` for a `#[dont_look_inside]` helper.
fn functionpath(segments: &[&str]) -> CallTarget {
    CallTarget::FunctionPath {
        segments: segments.iter().map(|s| s.to_string()).collect(),
    }
}

/// A synthetic-`Tuple` `__pos_<idx>` `FieldWrite` of a `Ref` element, matching
/// the `Rvalue::Aggregate` non-Adt tuple chain (owner_root `"Tuple"`, element
/// ty `Ref(None)`).
fn tuple_field_write(base: &Variable, idx: usize, value: Variable, owner: &str) -> SpaceOperation {
    SpaceOperation {
        result: None,
        kind: OpKind::FieldWrite {
            base: base.clone(),
            field: FieldDescriptor {
                name: format!("__pos_{idx}"),
                owner_root: Some(owner.to_string()),
                owner_id: None,
            },
            value: LinkArg::Value(value),
            ty: ValueType::Ref(None),
        },
    }
}

/// Build the modeled quotient/remainder tuple bound to `result_var`: the two
/// `#[dont_look_inside]` residuals + the synthetic-`Tuple` ctor + `__pos_0` /
/// `__pos_1` writes, the same transparent-ctor + `FieldWrite` chain
/// `Rvalue::Aggregate` emits for a non-Adt tuple.
fn build_div_rem_tuple(
    result_var: &Variable,
    num: Variable,
    den: Variable,
    q: Variable,
    r: Variable,
    owner: &str,
) -> [SpaceOperation; 5] {
    [
        SpaceOperation {
            result: Some(q.clone()),
            kind: OpKind::Call {
                target: functionpath(&DIV_PATH),
                args: vec![num.clone(), den.clone()],
                result_ty: ValueType::Ref(None),
            },
        },
        SpaceOperation {
            result: Some(r.clone()),
            kind: OpKind::Call {
                target: functionpath(&REM_PATH),
                args: vec![num, den],
                result_ty: ValueType::Ref(None),
            },
        },
        SpaceOperation {
            result: Some(result_var.clone()),
            kind: OpKind::Call {
                target: CallTarget::synthetic_transparent_ctor(owner.to_string()),
                args: Vec::new(),
                result_ty: ValueType::Ref(Some(owner.to_string())),
            },
        },
        tuple_field_write(result_var, 0, q, owner),
        tuple_field_write(result_var, 1, r, owner),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn div_rem_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["bigint", "BigInt", "div_rem"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    fn calls_to<'a>(g: &'a FunctionGraph, leaf: &'a str) -> usize {
        g.blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments },
                        ..
                    } if segments.last().map(String::as_str) == Some(leaf)
                )
            })
            .count()
    }

    /// Build `t = num.div_rem(&den)` and assert the rewrite drops the `div_rem`
    /// call, emits the `jit_bigint_div` / `jit_bigint_rem` residuals, and
    /// builds the `Tuple { __pos_0, __pos_1 }` aggregate on the original result.
    #[test]
    fn rewrite_lifts_div_rem_to_modeled_tuple() {
        let mut g = FunctionGraph::new("test_bigint_div_rem");
        let a = g.startblock;
        let num = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let den = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: div_rem_target(),
                    args: vec![num.clone(), den.clone()],
                    result_ty: ValueType::Ref(Some("Tuple".into())),
                },
                true,
            )
            .unwrap();
        let (b, _b_args) = g.create_block_with_arg_vars(1);
        g.set_return(b, None);
        g.set_goto(a, b, vec![result.clone()]);

        let rewritten = rewire_bigint_div_rem_call_sites(
            &mut g,
            &[BigIntDivRemSite {
                result_var: result.clone(),
            }],
        );
        assert_eq!(rewritten, 1, "the div_rem site must be rewritten");
        assert_eq!(calls_to(&g, "div_rem"), 0, "residual div_rem call removed");
        assert_eq!(
            calls_to(&g, "jit_bigint_div"),
            1,
            "quotient residual emitted"
        );
        assert_eq!(
            calls_to(&g, "jit_bigint_rem"),
            1,
            "remainder residual emitted"
        );
        // The aggregate: __pos_0 + __pos_1 writes on the result var.
        let pos_writes: Vec<usize> = g
            .blocks
            .iter()
            .flat_map(|blk| &blk.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldWrite { field, .. } if field.name.starts_with("__pos_") => field
                    .name
                    .strip_prefix("__pos_")
                    .and_then(|n| n.parse().ok()),
                _ => None,
            })
            .collect();
        assert_eq!(pos_writes, vec![0, 1], "two positional tuple writes");
        let ctor_binds_result = g.blocks[a.0].operations.iter().any(|op| {
            op.result.as_ref() == Some(&result)
                && matches!(&op.kind, OpKind::Call { args, .. } if args.is_empty())
        });
        assert!(ctor_binds_result, "result var re-bound to the Tuple ctor");
    }

    /// A producer op that is not a 2-arg call declines (fail-safe).
    #[test]
    fn rewrite_declines_when_producer_not_binary_call() {
        let mut g = FunctionGraph::new("test_bigint_div_rem_decline");
        let a = g.startblock;
        let num = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let result = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: div_rem_target(),
                    args: vec![num],
                    result_ty: ValueType::Ref(Some("Tuple".into())),
                },
                true,
            )
            .unwrap();
        g.set_return(a, None);

        let rewritten =
            rewire_bigint_div_rem_call_sites(&mut g, &[BigIntDivRemSite { result_var: result }]);
        assert_eq!(rewritten, 0, "a non-binary producer declines");
        assert_eq!(
            calls_to(&g, "div_rem"),
            1,
            "residual call survives on decline"
        );
    }
}
