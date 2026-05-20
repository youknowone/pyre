//! Legacy type resolution pass — transitional cutover input + reference.
//!
//! TODO(retire-legacy-resolve): flat `ConcreteType` enum with ad-hoc
//! lowering and no upstream `rpython/` counterpart — the orthodox port
//! is [`crate::translator::rtyper::rtyper::RPythonTyper`]
//! (`rtyper.py:RPythonTyper` + `rmodel.py:Repr` hierarchy), which
//! produces per-`Variable.concretetype` `LowLevelType` directly.
//!
//! This file remains because:
//!   * `dual_gate_check_with_registry`
//!     ([`crate::translator::rtyper::cutover`]) still compares the real
//!     path against `resolve_types(graph, &annotations)` for parity
//!     diff, and
//!   * `transform_graph_to_jitcode`
//!     ([`crate::jit_codewriter::codewriter`]) calls
//!     `resolve_rewritten_types(...)` to merge the post-jtransform
//!     `result_kind` declarations with `merge_synth_kinds`, and falls
//!     back to `legacy_state` (`resolve_types(graph, &annotations)`)
//!     when the dual-gate Skip-classifies.
//!
//! Retirement drops both consumers and this file together once Skip
//! categories close.
//!
//! Transforms annotated ValueTypes into concrete low-level types
//! and specializes operations accordingly.

use std::collections::HashMap;

use crate::flowspace::model::ConstValue;
use crate::jit_codewriter::annotation_state::{AnnotationState, somevalue_to_valuetype};
use crate::jit_codewriter::type_state::{
    ConcreteType, authoritative_result_types, kind_char_to_concrete, valuetype_to_concrete,
};
use crate::model::{FunctionGraph, Link, LinkArg, OpKind, ValueId};

/// Resolve annotations to concrete types.
///
/// RPython equivalent: `RPythonTyper.specialize_block()` — walks
/// each block and converts annotation → Repr → lowleveltype.
///
/// Result is committed through `graph.set_concretetype_inline` for
/// every populated `ValueId`; downstream consumers read kinds via
/// `graph.concretetype(v)` (`getkind(v.concretetype)`).
pub fn resolve_types(graph: &FunctionGraph, annotations: &AnnotationState) {
    // Walk the orthodox `some_values` map (RPython `Variable.annotation`
    // analogue).  `Unknown`-projected entries never reach this loop
    // because `AnnotationState::set` removes the `some_values` slot on
    // `Unknown`; non-`Unknown` `valuetype_to_concrete` outputs always
    // produce a concrete kind, so every iteration commits a real
    // `graph.set_concretetype_inline` write.
    for (&vid, rc_some) in &annotations.some_values {
        let vtype = somevalue_to_valuetype(rc_some);
        let concrete = valuetype_to_concrete(&vtype);
        graph.set_concretetype_inline(vid, concrete);
    }

    // Resolve from ops with explicit type info
    for block in &graph.blocks {
        // Resolve inputargs (Phi nodes) from annotations
        for vid in block.inputarg_value_ids(graph) {
            if graph.concretetype(vid) == ConcreteType::Unknown {
                let vtype = annotations.get(vid);
                let concrete = valuetype_to_concrete(&vtype);
                if concrete != ConcreteType::Unknown {
                    graph.set_concretetype_inline(vid, concrete);
                }
            }
        }
        for op in &block.operations {
            if let Some(result) = op.registered_result_value_id(graph) {
                let inferred = infer_concrete_from_op(&op.kind);
                if inferred != ConcreteType::Unknown {
                    // The op carries authoritative result kind (post-
                    // jtransform `float_*` rewrite, explicit Call /
                    // FieldRead / etc. result_ty).  Override any
                    // annotation default — annrpython defaults BinOp /
                    // UnaryOp result to Int, which would shadow a later
                    // Float inference for `float_add` / `float_neg` /
                    // etc. produced by jtransform's float-operand
                    // rewrite arm.
                    graph.set_concretetype_inline(result, inferred);
                }
            }
        }
    }

    // Cross-block: propagate through Link args → target inputargs.
    // Keep the exception-link split explicit, mirroring upstream's
    // `_convert_link()` handling of `last_exception` /
    // `last_exc_value` before the per-arg conversion loop.
    for block in &graph.blocks {
        for link in &block.exits {
            if link_is_raise_like(link) {
                convert_raise_link(graph, link);
            } else {
                convert_link(graph, link);
            }
        }
    }

    // Backward-constraint: an op whose RPython decorator declares
    // integer operands (`@arguments("i", ...)` in `blackhole.py`) cannot
    // run on `Ref`-classified operands — upstream `rtyper.py:specialize`
    // determines operand kind from the op's `concretetype`, not from a
    // fallback.  Pyre's front-end leaves many `OpKind::Input` values
    // with `ty: Unknown` (path references not yet bound by the annotator);
    // without this pass they end up defaulted to `GcRef` by the backfill
    // below and produce pyre-only `int_*/{ri,ir,rr,r}` keys. Upgrade
    // Unknown operands of pure integer ops to `Signed`, matching
    // RPython's mandatory kind assignment.  `ptr_eq` / `ptr_ne` remain
    // Ref-Ref (rewritten in `jtransform` per `rpython/jit/codewriter/
    // jtransform.py:1243-1255 _rewrite_cmp_ptrs`) and are skipped.
    /// Operations whose RPython decorator requires integer operands on
    /// both sides (`blackhole.py:459+ @arguments("i", "i", returns="i")`).
    /// Unlike the comparison ops `lt`/`le`/`gt`/`ge`, no ptr-specific
    /// variant exists in RPython — these must see Signed operands.
    fn canonical_int_binop(op: &str) -> Option<&'static str> {
        match op {
            "add" | "add_assign" => Some("add"),
            "sub" | "sub_assign" => Some("sub"),
            "mul" | "mul_assign" => Some("mul"),
            "div" | "div_assign" => Some("div"),
            "mod" | "mod_assign" => Some("mod"),
            "and" | "bitand" | "bitand_assign" => Some("and"),
            "or" | "bitor" | "bitor_assign" => Some("or"),
            "xor" | "bitxor" | "bitxor_assign" => Some("xor"),
            "rshift" | "shr" | "rshift_assign" => Some("rshift"),
            "lshift" | "shl" | "lshift_assign" => Some("lshift"),
            "lt" => Some("lt"),
            "le" => Some("le"),
            "gt" => Some("gt"),
            "ge" => Some("ge"),
            _ => None,
        }
    }
    /// `eq`/`ne` may still be rewritten to `ptr_eq`/`ptr_ne` in
    /// `jtransform.rs` when both operands are Ref (RPython
    /// `_rewrite_cmp_ptrs`). Skip them here so the jtransform pass
    /// observes the original Ref-Ref shape.
    fn is_int_unop(op: &str) -> bool {
        matches!(op, "neg" | "invert" | "not")
    }
    fn is_identity_unop(op: &str) -> bool {
        matches!(op, "same_as")
    }
    // Iterate: a backward-inferred Signed on one ValueId may feed
    // another op's operand through Link propagation, so run until
    // fixed-point.  RPython's `rtyper.py:specialize` is a single
    // forward pass because every Variable has a `concretetype` from
    // annotation; pyre's partial typing means downstream ops can
    // depend on upstream inferences that only this pass supplies.
    loop {
        let mut changed = false;
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::BinOp {
                        op: opname,
                        lhs,
                        rhs,
                        ..
                    } if canonical_int_binop(opname).is_some() => {
                        // Only upgrade Unknown operands.  Forcing
                        // GcRef → Signed here is unsound: a value
                        // classified Ref may alias with pointer uses
                        // elsewhere in the graph, and demoting it to
                        // Int would misroute those other uses
                        // (regalloc puts the value in an int register,
                        // subsequent field reads dereference garbage).
                        // The remaining pyre-only `int_*/{ri,ir,rr}`
                        // keys need `cast_ptr_to_int` insertion in
                        // jtransform, not a type override in the
                        // rtyper.
                        let lhs_vid = graph
                            .value_id_of(lhs)
                            .expect("BinOp.lhs has backing ValueId");
                        let rhs_vid = graph
                            .value_id_of(rhs)
                            .expect("BinOp.rhs has backing ValueId");
                        changed |= maybe_seed_concrete_type(graph, lhs_vid, ConcreteType::Signed);
                        changed |= maybe_seed_concrete_type(graph, rhs_vid, ConcreteType::Signed);
                        if let Some(result) = op.registered_result_value_id(graph) {
                            // RPython rtyper resolves an `add` whose
                            // operands are both `lltype.Float` to
                            // `float_add` directly (`rfloat.py
                            // rtype_add`); the result inherits Float
                            // (or Int for comparisons).  When pyre's
                            // upstream typing already classified both
                            // operands as Float, mirror that: seed the
                            // result as Float for arithmetic ops and
                            // Int for comparisons.  jtransform will
                            // then rewrite the BinOp's `op` from `add`
                            // to `float_add` (etc.) and insert
                            // `cast_int_to_float` for mixed int/float
                            // operands, keeping IR/regalloc consistent.
                            let lhs_float = graph.concretetype(lhs_vid) == ConcreteType::Float;
                            let rhs_float = graph.concretetype(rhs_vid) == ConcreteType::Float;
                            let any_float = lhs_float || rhs_float;
                            let both_numeric = matches!(
                                graph.concretetype(lhs_vid),
                                ConcreteType::Signed | ConcreteType::Float
                            ) && matches!(
                                graph.concretetype(rhs_vid),
                                ConcreteType::Signed | ConcreteType::Float
                            );
                            // Mirror jtransform's float-rewrite set
                            // (`jit_codewriter/jtransform.rs`):
                            // arithmetic `add/sub/mul/div` →
                            // `float_*` returns Float; comparisons
                            // `lt/le/gt/ge/eq/ne` → `float_*` returns
                            // Int (Bool — RPython `rfloat.py:133`,
                            // `blackhole.py:731 bhimpl_float_eq`).
                            // `mod` is also Float when a Float operand is
                            // present, but jtransform lowers it to the
                            // residual `ll_math_fmod` helper rather than a
                            // non-RPython `float_mod` opcode.
                            let is_compare =
                                matches!(opname.as_str(), "lt" | "le" | "gt" | "ge" | "eq" | "ne");
                            let is_arith =
                                matches!(opname.as_str(), "add" | "sub" | "mul" | "div" | "mod");
                            if any_float && both_numeric && (is_arith || is_compare) {
                                let target = if is_compare {
                                    ConcreteType::Signed
                                } else {
                                    ConcreteType::Float
                                };
                                if graph.concretetype(result) != target {
                                    graph.set_concretetype_inline(result, target);
                                    changed = true;
                                }
                            } else if !any_float || !both_numeric {
                                changed |=
                                    maybe_seed_concrete_type(graph, result, ConcreteType::Signed);
                            }
                        }
                    }
                    OpKind::UnaryOp {
                        op: opname,
                        operand,
                        ..
                    } if is_int_unop(opname) => {
                        let operand_vid = graph
                            .value_id_of(operand)
                            .expect("UnaryOp.operand has backing ValueId");
                        changed |=
                            maybe_seed_concrete_type(graph, operand_vid, ConcreteType::Signed);
                        if let Some(result) = op.registered_result_value_id(graph) {
                            // Same Float-operand override as the BinOp
                            // arm above.  Unary `neg` on a Float
                            // returns Float (`float_neg`).
                            let operand_float =
                                graph.concretetype(operand_vid) == ConcreteType::Float;
                            if operand_float && opname == "neg" {
                                if graph.concretetype(result) != ConcreteType::Float {
                                    graph.set_concretetype_inline(result, ConcreteType::Float);
                                    changed = true;
                                }
                            } else {
                                changed |=
                                    maybe_seed_concrete_type(graph, result, ConcreteType::Signed);
                            }
                        }
                    }
                    OpKind::UnaryOp {
                        op: opname,
                        operand,
                        ..
                    } if opname == "cast_int_to_float" => {
                        // RPython `rfloat.py` rtype_int → calls
                        // `hop.inputarg(Signed, ...)` before emitting
                        // `cast_int_to_float`, so the operand's
                        // concretetype is Signed by construction.
                        // jtransform's `coerce_operand_to_float`
                        // (jit_codewriter/jtransform.rs) emits this
                        // op only when get_value_kind(operand) == 'i'
                        // in pass 1.  Pass 2 runs on the rewritten
                        // graph from a fresh state and may lose that
                        // backward-constraint upgrade if the operand
                        // lacks a definitive def-site classification
                        // (e.g., Call with Unknown result_ty), which
                        // would surface as `cast_int_to_float/r>f`
                        // at the assembler.  Re-seed Signed here so
                        // pass 2 converges to the same operand kind.
                        let operand_vid = graph
                            .value_id_of(operand)
                            .expect("UnaryOp.operand has backing ValueId");
                        changed |=
                            maybe_seed_concrete_type(graph, operand_vid, ConcreteType::Signed);
                    }
                    OpKind::UnaryOp {
                        op: opname,
                        operand,
                        ..
                    } if is_identity_unop(opname) => {
                        if let Some(result) = op.registered_result_value_id(graph) {
                            let operand_vid = graph
                                .value_id_of(operand)
                                .expect("UnaryOp.operand has backing ValueId");
                            let operand_ty = graph.concretetype(operand_vid);
                            let result_ty = graph.concretetype(result);
                            changed |= maybe_seed_concrete_type(graph, result, operand_ty);
                            changed |= maybe_seed_concrete_type(graph, operand_vid, result_ty);
                        }
                    }
                    _ => {}
                }
            }
            // Re-propagate along links after each op round so backward
            // inferences reach the linked ValueIds. RPython has concrete
            // types on both ends by this point and `_convert_link()` emits
            // conversions where needed; this legacy pass has no conversion
            // insertion, so Unknown values joined by a link must converge to
            // the known low-level type before regalloc/flatten.
            for link in &block.exits {
                changed |= if link_is_raise_like(link) {
                    converge_raise_link(graph, link)
                } else {
                    converge_link(graph, link)
                };
            }
        }
        if !changed {
            break;
        }
    }

    // Backfill any `Variable` that's referenced by the graph but was
    // never typed (e.g. Link args pointing at values with no producer
    // op, synthetic values whose declaring site didn't seed a
    // concretetype).  RPython's rtyper cannot leave any `Variable`
    // untyped — `lltype.Signed / Ptr / Float` is mandatory — so the
    // assembler is entitled to assume every register has a class.
    // Default untyped values to `GcRef` (the same safe default
    // jtransform's `get_value_kind` picks) so `build_value_kinds` +
    // `perform_register_allocation` always produce a coloring for
    // every reachable value.
    let mut seen: std::collections::HashSet<ValueId> = std::collections::HashSet::new();
    for block in &graph.blocks {
        for v in block.inputarg_value_ids(graph) {
            seen.insert(v);
        }
        for op in &block.operations {
            for v in crate::inline::op_value_refs(&op.kind, Some(graph)) {
                seen.insert(v);
            }
            if let Some(r) = op.registered_result_value_id(graph) {
                seen.insert(r);
            }
        }
        for link in &block.exits {
            for arg in &link.args {
                if let Some(v) = arg.as_value(graph) {
                    seen.insert(v);
                }
            }
            for arg in link.last_exception.iter().chain(link.last_exc_value.iter()) {
                if let Some(v) = arg.as_value(graph) {
                    seen.insert(v);
                }
            }
        }
    }
    for v in seen {
        if graph.concretetype(v) == ConcreteType::Unknown {
            graph.set_concretetype_inline(v, ConcreteType::GcRef);
        }
    }

    // RPython parity: every `graph.set_concretetype_inline(vid, ct)`
    // above publishes the resolved kind on each Variable's
    // `concretetype` cell, matching `rtyper.py:258 v.concretetype = ...`.
    // Downstream consumers read kinds via `graph.concretetype(v)`
    // (i.e. `getkind(v.concretetype)`) directly without a separate
    // `apply_to_graph` publish step.
}

fn const_value_to_concrete(value: &ConstValue) -> ConcreteType {
    match value {
        ConstValue::Int(_)
        | ConstValue::Bool(_)
        | ConstValue::SpecTag(_)
        | ConstValue::LLAddress(_) => ConcreteType::Signed,
        ConstValue::Float(_) => ConcreteType::Float,
        ConstValue::Placeholder => ConcreteType::Unknown,
        ConstValue::Atom(_)
        | ConstValue::Dict(_)
        | ConstValue::ByteStr(_)
        | ConstValue::UniStr(_)
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::None
        | ConstValue::Code(_)
        | ConstValue::LLPtr(_)
        | ConstValue::Function(_)
        | ConstValue::HostObject(_) => ConcreteType::GcRef,
    }
}

fn link_is_raise_like(link: &Link) -> bool {
    link.last_exception.is_some() && link.last_exc_value.is_some()
}

fn convert_link(graph: &FunctionGraph, link: &Link) {
    let target_block = graph.block(link.target);
    let target_vids = target_block.inputarg_value_ids(graph);
    for (dst, src) in target_vids.iter().zip(link.args.iter()) {
        // A `LinkArg::Const(_)` whose `concretetype` carries a
        // construction-site repr (e.g. `set_return`'s
        // `Constant(None, concretetype=Void)`) is authoritative —
        // mirrors RPython `_convert_link()` materialising
        // `inputconst(r_to, value)` whose own `concretetype` is the
        // destination repr (`rpython/rtyper/rmodel.py:inputconst`
        // + `rpython/rtyper/rnone.py:48`).  Force-write so an
        // earlier annotation-derived kind (e.g. `Ref → GcRef` from
        // `valuetype_to_concrete` on the upstream `SomeNone → Ref`
        // projection) does not shadow it.
        if let LinkArg::Const(value) = src
            && let Some(lltype) = value.concretetype.as_ref()
        {
            graph.set_concretetype_inline(*dst, crate::model::getkind(lltype));
            continue;
        }
        let _ = maybe_seed_concrete_type(graph, *dst, link_arg_concrete_type(graph, src));
    }
}

fn convert_raise_link(graph: &FunctionGraph, link: &Link) {
    if let Some(value) = link.last_exception.as_ref().and_then(|a| a.as_value(graph)) {
        let _ = maybe_seed_concrete_type(graph, value, ConcreteType::Signed);
    }
    if let Some(value) = link.last_exc_value.as_ref().and_then(|a| a.as_value(graph)) {
        let _ = maybe_seed_concrete_type(graph, value, ConcreteType::GcRef);
    }

    let target_block = graph.block(link.target);
    let target_vids = target_block.inputarg_value_ids(graph);
    for (dst, src) in target_vids.iter().zip(link.args.iter()) {
        let src_ty = if Some(src) == link.last_exception.as_ref() {
            ConcreteType::Signed
        } else if Some(src) == link.last_exc_value.as_ref() {
            ConcreteType::GcRef
        } else {
            link_arg_concrete_type(graph, src)
        };
        let _ = maybe_seed_concrete_type(graph, *dst, src_ty);
    }
}

fn converge_link(graph: &FunctionGraph, link: &Link) -> bool {
    let mut changed = false;
    let target_block = graph.block(link.target);
    let target_vids = target_block.inputarg_value_ids(graph);
    for (dst, src) in target_vids.iter().zip(link.args.iter()) {
        match src {
            LinkArg::Value(var) => {
                // After the Variable cutover, `as_value(graph) == None`
                // means the link references a `Variable` the graph
                // never registered — malformed link metadata.  Silently
                // skipping convergence would let the dual-gate baseline
                // pass with degraded types and hide the producer bug;
                // mirror the `legacy_annotator::link_arg_type`
                // expect-style contract instead.
                let src_vid = src.as_value(graph).unwrap_or_else(|| {
                    panic!(
                        "converge_link: LinkArg::Value references Variable \
                         {var:?} that is not registered on the graph — \
                         malformed link.args"
                    )
                });
                let src_ty = graph.concretetype(src_vid);
                let dst_ty = graph.concretetype(*dst);
                changed |= maybe_seed_concrete_type(graph, *dst, src_ty);
                changed |= maybe_seed_concrete_type(graph, src_vid, dst_ty);
            }
            LinkArg::Const(_) => {
                // Route through `link_arg_concrete_type` so the
                // `Constant.concretetype` construction-site hint
                // (e.g. `set_return`'s `concretetype=Void`) is
                // honoured here too — bypassing it would let
                // `const_value_to_concrete(&value.value)` default
                // `None → GcRef` and shadow the Void write.
                changed |=
                    maybe_seed_concrete_type(graph, *dst, link_arg_concrete_type(graph, src));
            }
        }
    }
    changed
}

fn converge_raise_link(graph: &FunctionGraph, link: &Link) -> bool {
    let mut changed = false;
    if let Some(value) = link.last_exception.as_ref().and_then(|a| a.as_value(graph)) {
        changed |= maybe_seed_concrete_type(graph, value, ConcreteType::Signed);
    }
    if let Some(value) = link.last_exc_value.as_ref().and_then(|a| a.as_value(graph)) {
        changed |= maybe_seed_concrete_type(graph, value, ConcreteType::GcRef);
    }

    let target_block = graph.block(link.target);
    let target_vids = target_block.inputarg_value_ids(graph);
    for (dst, src) in target_vids.iter().zip(link.args.iter()) {
        let src_ty = if Some(src) == link.last_exception.as_ref() {
            ConcreteType::Signed
        } else if Some(src) == link.last_exc_value.as_ref() {
            ConcreteType::GcRef
        } else {
            link_arg_concrete_type(graph, src)
        };
        changed |= maybe_seed_concrete_type(graph, *dst, src_ty);
        // For LinkArg::Value, propagate the dst kind back to the src
        // Variable.  `as_value(graph) == None` after the Variable
        // cutover means malformed link metadata (the LinkArg::Const
        // arm exits early); fail loud rather than silently skip the
        // back-propagation.
        if let LinkArg::Value(var) = src {
            let src_vid = src.as_value(graph).unwrap_or_else(|| {
                panic!(
                    "converge_raise_link: LinkArg::Value references Variable \
                     {var:?} that is not registered on the graph — \
                     malformed link.args"
                )
            });
            let dst_ty = graph.concretetype(*dst);
            changed |= maybe_seed_concrete_type(graph, src_vid, dst_ty);
        }
    }
    changed
}

fn link_arg_concrete_type(graph: &FunctionGraph, src: &LinkArg) -> ConcreteType {
    match src {
        // After the Variable cutover an unregistered LinkArg::Value
        // is malformed metadata — fail loud rather than degrading to
        // `Unknown` which would let the dual-gate baseline pass with
        // missing kinds and mask producer bugs.  Mirrors the
        // `legacy_annotator::link_arg_type` contract.
        LinkArg::Value(var) => {
            let vid = src.as_value(graph).unwrap_or_else(|| {
                panic!(
                    "link_arg_concrete_type: LinkArg::Value references Variable \
                     {var:?} that is not registered on the graph — \
                     malformed link.args"
                )
            });
            graph.concretetype(vid)
        }
        // RPython `pairtype(Repr, NoneRepr).convert_from_to`
        // (`rpython/rtyper/rnone.py:48`) emits `inputconst(Void, None)`
        // when None flows into a `NoneRepr` target; the symmetric
        // `pairtype(NoneRepr, Repr).convert_from_to`
        // (`rpython/rtyper/rnone.py:58`) emits `inputconst(r_to, None)`
        // which is a null pointer for `Ptr`/ref targets.  Pyre's
        // construction sites that already know the target repr write
        // it onto `Constant.concretetype` at the construction site
        // (e.g. `set_return` wires `Constant(None, concretetype=Void)`
        // per `flowcontext.py:687-689` + `:1232-1236`); honour that
        // construction-site hint here.  Falls back to the value-only
        // default (`getkind(Ptr) == GcRef` for None) when the
        // construction site did not set a target repr.
        LinkArg::Const(value) => match value.concretetype.as_ref() {
            Some(lltype) => crate::model::getkind(lltype),
            None => const_value_to_concrete(&value.value),
        },
    }
}

fn maybe_seed_concrete_type(graph: &FunctionGraph, dst: ValueId, src_ty: ConcreteType) -> bool {
    if graph.concretetype(dst) == ConcreteType::Unknown && src_ty != ConcreteType::Unknown {
        // RPython parity: `rtyper.py:258 v.concretetype = ...` writes the
        // resolved kind inline on the Variable as soon as the resolver
        // knows it.  Pyre's iterative build mirrors that by publishing
        // through `graph.set_concretetype_inline` so `graph.concretetype(v)`
        // sees the kind during the same pass.
        graph.set_concretetype_inline(dst, src_ty);
        true
    } else {
        false
    }
}

fn infer_concrete_from_op(kind: &OpKind) -> ConcreteType {
    match kind {
        OpKind::ConstInt(_) => ConcreteType::Signed,
        // RPython `getkind(lltype.Bool) == 'int'` (flatten.py:getkind);
        // codewriter folds Bool storage to int kind.
        OpKind::ConstBool(_) => ConcreteType::Signed,
        OpKind::ConstFloat(_) => ConcreteType::Float,
        // RPython `rpython/annotator/annrpython.py` types every Variable
        // at annotation time, so `OpKind::Input` reaching rtyper has a
        // concrete type.  pyre's front-end (`front/ast.rs` Expr::Path
        // lowering) re-emits a fresh `OpKind::Input { ty: Unknown }` for
        // each source-level identifier reference instead of binding the
        // name to the inputarg's Variable.  Leave Unknown so the
        // integer-op backward-constraint pass (`resolve_types`) can
        // upgrade operands of pure integer ops to `Signed` before the
        // final GcRef backfill.
        OpKind::Input { ty, .. } => valuetype_to_concrete(ty),
        // Field / array reads whose declared `ty` is pyre-only Unknown
        // default to `ConcreteType::GcRef` so the value reaches
        // regalloc. RPython's rtyper resolves the field / element type
        // from the struct's `concretetype`; pyre's struct registry can
        // lack the entry (Rust generics, unsupported declarations),
        // leaving `ValueType::Unknown`.  Ref is the conservative
        // default — if the underlying value is actually Int, the
        // canonical `getfield_gc_i/rd>i` key still carries the
        // correct result kind at assembler emit time.
        OpKind::FieldRead { ty, .. } => {
            let c = valuetype_to_concrete(ty);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::ArrayRead { item_ty, .. } => {
            let c = valuetype_to_concrete(item_ty);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::InteriorFieldRead { item_ty, .. } => {
            let c = valuetype_to_concrete(item_ty);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::Call { result_ty, .. } => valuetype_to_concrete(result_ty),
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => kind_char_to_concrete(*result_kind),
        OpKind::UnaryOp { op, result_ty, .. } if op == "same_as" => {
            valuetype_to_concrete(result_ty)
        }
        // For `BinOp` / `UnaryOp` whose `result_ty` is still
        // `ValueType::Unknown`, return `ConcreteType::Unknown` rather
        // than defaulting to `Signed`.  The override at line ~55 only
        // fires when `inferred != Unknown`; previously the `Signed`
        // fallback overrode the annotator's value (e.g. the `neg(Float)
        // → Float` annotation produced by the dedicated `op == "neg"`
        // arm in `infer_op_type`), poisoning downstream phi-merge
        // inputargs whose link arg types are correctly Float.  With
        // `Unknown` here the override is skipped and the annotator's
        // value persists; the loop below still upgrades the concrete
        // type to `Float` for `neg(Float)` etc. via the special arms.
        OpKind::BinOp { result_ty, .. } | OpKind::UnaryOp { result_ty, .. } => {
            valuetype_to_concrete(result_ty)
        }
        // Vtable funcptr extraction returns an integer pointer (RPython
        // `op.args[0]` of `indirect_call` is `Ptr(FuncType)`).
        OpKind::VtableMethodPtr { .. } => ConcreteType::Signed,
        OpKind::IndirectCall { result_ty, .. } => valuetype_to_concrete(result_ty),
        // Virtualizable field / array reads inherit the RPython
        // canonical result kind from the declared field/element type.
        OpKind::VableFieldRead { ty, .. } => {
            let c = valuetype_to_concrete(ty);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::VableArrayRead { item_ty, .. } => {
            let c = valuetype_to_concrete(item_ty);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        // pyre-only `OpKind::Abort` (`front/ast.rs` lowering of Rust
        // syntax not yet ported — macros, unsupported literals,
        // fallback expressions).  Fall back to GcRef so these values
        // still get a regalloc coloring and the assembler's
        // `lookup_reg_with_kind` covers every operand. RPython has no
        // analogue; porting each producer path individually eliminates
        // the `abort/` wire keys.
        OpKind::Abort { .. } => ConcreteType::GcRef,
        _ => ConcreteType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, ValueType, exception_exitcase,
    };
    use crate::translator::rtyper::legacy_annotator as annotate;

    #[test]
    fn resolves_int_types() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        graph.set_return(entry, Some(graph.must_variable(v)));

        let annotations = annotate::annotate(&graph);
        resolve_types(&graph, &annotations);
        assert_eq!(graph.concretetype(v), ConcreteType::Signed);
    }

    #[test]
    fn resolves_ref_field() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let base = graph.alloc_value();
        let base_var = graph.must_variable(base);
        let v = graph
            .push_op(
                entry,
                OpKind::FieldRead {
                    base: base_var,
                    field: crate::model::FieldDescriptor::new("obj", None),
                    ty: ValueType::Ref,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(graph.must_variable(v)));

        let annotations = annotate::annotate(&graph);
        resolve_types(&graph, &annotations);
        assert_eq!(graph.concretetype(v), ConcreteType::GcRef);
    }

    #[test]
    fn resolves_phi_through_link_args() {
        let mut graph = FunctionGraph::new("phi");
        let entry = graph.startblock;
        let val = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let (target, phi_args) = graph.create_block_with_args(1);
        let phi = phi_args[0];
        let val_var = graph.must_variable(val);
        graph.set_goto(entry, target, vec![val_var]);
        graph.set_return(target, Some(graph.must_variable(phi)));

        let annotations = annotate::annotate(&graph);
        resolve_types(&graph, &annotations);
        assert_eq!(graph.concretetype(phi), ConcreteType::Signed);
    }

    #[test]
    fn backward_constraint_types_unknown_int_binop_operands_as_signed() {
        let mut graph = FunctionGraph::new("int_backprop");
        let entry = graph.startblock;
        let lhs = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "lhs".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let rhs = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "rhs".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let lhs_var = graph.must_variable(lhs);
        let rhs_var = graph.must_variable(rhs);
        let result = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "add".to_string(),
                    lhs: lhs_var,
                    rhs: rhs_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(graph.must_variable(result)));

        let annotations = annotate::annotate(&graph);
        resolve_types(&graph, &annotations);
        assert_eq!(graph.concretetype(lhs), ConcreteType::Signed);
        assert_eq!(graph.concretetype(rhs), ConcreteType::Signed);
        assert_eq!(graph.concretetype(result), ConcreteType::Signed);
    }

    #[test]
    fn backward_constraint_types_frontend_bitop_operands_as_signed() {
        let mut graph = FunctionGraph::new("bitxor_backprop");
        let entry = graph.startblock;
        let lhs = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "lhs".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let rhs = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "rhs".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let lhs_var = graph.must_variable(lhs);
        let rhs_var = graph.must_variable(rhs);
        let result = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "bitxor".to_string(),
                    lhs: lhs_var,
                    rhs: rhs_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(graph.must_variable(result)));

        let annotations = annotate::annotate(&graph);
        resolve_types(&graph, &annotations);
        assert_eq!(graph.concretetype(lhs), ConcreteType::Signed);
        assert_eq!(graph.concretetype(rhs), ConcreteType::Signed);
        assert_eq!(graph.concretetype(result), ConcreteType::Signed);
    }

    #[test]
    fn same_as_preserves_ref_classification() {
        let mut graph = FunctionGraph::new("same_as_ref");
        let entry = graph.startblock;
        let value = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "obj".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let value_var = graph.must_variable(value);
        let alias = graph
            .push_op(
                entry,
                OpKind::UnaryOp {
                    op: "same_as".to_string(),
                    operand: value_var,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(graph.must_variable(alias)));

        let annotations = annotate::annotate(&graph);
        resolve_types(&graph, &annotations);
        assert_eq!(graph.concretetype(value), ConcreteType::GcRef);
        assert_eq!(graph.concretetype(alias), ConcreteType::GcRef);
    }

    #[test]
    fn same_as_propagates_signed_without_forcing_unknown_identity_to_int() {
        let mut graph = FunctionGraph::new("same_as_int");
        let entry = graph.startblock;
        let value = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let value_var = graph.must_variable(value);
        let alias = graph
            .push_op(
                entry,
                OpKind::UnaryOp {
                    op: "same_as".to_string(),
                    operand: value_var,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(graph.must_variable(alias)));

        let annotations = annotate::annotate(&graph);
        resolve_types(&graph, &annotations);
        assert_eq!(graph.concretetype(value), ConcreteType::Signed);
        assert_eq!(graph.concretetype(alias), ConcreteType::Signed);
    }

    #[test]
    fn backward_constraint_propagates_signed_back_through_link_source() {
        let mut graph = FunctionGraph::new("link_backprop");
        let entry = graph.startblock;
        let src = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "x".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let one = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let (target, phi_args) = graph.create_block_with_args(1);
        let phi = phi_args[0];
        let src_var = graph.must_variable(src);
        graph.set_goto(entry, target, vec![src_var]);
        let phi_var = graph.must_variable(phi);
        let one_var = graph.must_variable(one);
        let result = graph
            .push_op(
                target,
                OpKind::BinOp {
                    op: "add".to_string(),
                    lhs: phi_var,
                    rhs: one_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(target, Some(graph.must_variable(result)));

        let annotations = annotate::annotate(&graph);
        resolve_types(&graph, &annotations);
        assert_eq!(graph.concretetype(phi), ConcreteType::Signed);
        assert_eq!(graph.concretetype(src), ConcreteType::Signed);
        assert_eq!(graph.concretetype(result), ConcreteType::Signed);
    }

    #[test]
    fn unknown_input_without_integer_constraint_backfills_as_gcref() {
        let mut graph = FunctionGraph::new("unknown_backfill");
        let entry = graph.startblock;
        let value = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "obj".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(graph.must_variable(value)));

        let annotations = annotate::annotate(&graph);
        resolve_types(&graph, &annotations);
        assert_eq!(graph.concretetype(value), ConcreteType::GcRef);
    }

    #[test]
    fn resolves_raise_link_exception_pair() {
        let mut graph = FunctionGraph::new("raise_link");
        let entry = graph.startblock;
        let (exc_block, etype, evalue) = graph.exceptblock_args();
        let last_exception = graph.alloc_value();
        let last_exc_value = graph.alloc_value();
        let last_exception_var = graph.must_variable(last_exception);
        let last_exc_value_var = graph.must_variable(last_exc_value);
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![
                Link::from_variables(
                    &graph,
                    vec![last_exception_var.clone(), last_exc_value_var.clone()],
                    exc_block,
                    Some(exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::Value(last_exception_var)),
                    Some(LinkArg::Value(last_exc_value_var)),
                ),
            ],
        );

        let annotations = annotate::annotate(&graph);
        resolve_types(&graph, &annotations);
        assert_eq!(graph.concretetype(last_exception), ConcreteType::Signed);
        assert_eq!(graph.concretetype(last_exc_value), ConcreteType::GcRef);
        assert_eq!(graph.concretetype(etype), ConcreteType::Signed);
        assert_eq!(graph.concretetype(evalue), ConcreteType::GcRef);
    }
}
