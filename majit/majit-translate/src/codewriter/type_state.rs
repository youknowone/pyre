//! Concrete-kind helpers — `ConcreteType` projection from
//! `LowLevelType`, `ValueType`, op-result kinds, plus
//! `apply_from_flowspace_variables` (copies rtyper-typed lltypes onto
//! the matching legacy Variables, propagating `Variable.concretetype`
//! into every alias).
//!
//! Type kinds flow through `Variable.concretetype`
//! (`rpython/flowspace/model.py Variable.__slots__ = [..., "concretetype"]`;
//! `:355 Constant.__slots__ = ["concretetype"]`) — set inline by the
//! rtyper via `RPythonTyper.setconcretetype()`
//! (`rpython/rtyper/rtyper.py v.concretetype = ...`).  Pyre
//! reproduces this through `FunctionGraph::set_concretetype_of_inline`
//! writes followed by `FunctionGraph::concretetype_of(&v)` reads
//! (which routes to the backing `Variable.concretetype` cell).  No
//! external slot table survives.

use std::collections::{HashMap, HashSet};

use crate::flowspace::model::Variable;
use crate::model::{FunctionGraph, OpKind, SpaceOperation, ValueType};

/// Re-export the canonical [`ConcreteType`] from [`crate::model`].
///
/// The kind enum used to live here as a side-table value type;
/// after the medium-term parity push it lives on each backing
/// `Variable.concretetype` cell carried inline on each backing
/// `Variable` referenced by the IR (mirroring upstream
/// `Variable.concretetype` line-for-line).  The alias keeps existing
/// imports working while consumers migrate to reading
/// `FunctionGraph::concretetype_of(&v)`.
pub use crate::model::ConcreteType;

/// Copy each typed Variable's `concretetype` onto the legacy graph
/// Variable it was seeded from, so subsequent
/// `FunctionGraph::concretetype_of(&v)` reads route through the
/// rtyper's `Variable.concretetype` directly.
///
/// `value_to_var` is keyed by the legacy graph Variable's object
/// identity (`legacy_var -> typed_var`).  Each legacy Variable's
/// `Rc<RefCell>` concretetype cell is shared across every reference to
/// it in the graph — `Block.inputargs`, op operands, `Link.args`,
/// `exitswitch`, `last_exception`, `last_exc_value` — so a single write
/// onto the key Variable propagates everywhere, mirroring upstream
/// `v.concretetype = T` attribute aliasing (`history.py getkind`
/// reads `v.concretetype` off the Variable).
///
/// A typed Variable whose `concretetype` is still `None` (rtyper hasn't
/// processed it yet) leaves its legacy counterpart untouched —
/// equivalent to RPython's "no `.concretetype` attribute" window before
/// `setconcretetype` runs.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn apply_from_flowspace_variables(
    value_to_var: &crate::translator::rtyper::flowspace_adapter::LegacyToTyped,
) {
    for (legacy_var, typed_var) in value_to_var.iter() {
        let Some(ct) = typed_var.concretetype() else {
            continue;
        };
        legacy_var.set_concretetype(Some(ct));
    }
}

/// Publish the Constant half of the accepted rtyper projection. Upstream
/// reads Constant.concretetype directly; the transitional MIR emitter also
/// has Variables for constant definitions and eliminated phi aliases.
/// Call only after the unchanged dual-gate comparison has accepted them.
#[expect(
    clippy::mutable_key_type,
    reason = "Variable keys use immutable identity, like RPython graph variables"
)]
pub(crate) fn apply_from_flowspace_constants(
    constants: &std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
    >,
) {
    for (legacy, lltype) in constants {
        legacy.set_concretetype(Some(lltype.clone()));
    }
}

/// `ValueType` → `ConcreteType` projection used by both
/// `resolve_types` (legacy graph walk) and `authoritative_result_types`
/// (post-jtransform op-result projection).
///
/// `Bool` collapses to `Signed` because RPython `BoolRepr.lowleveltype
/// = Bool` lifts to LL `Signed` for the codewriter; the legacy resolver
/// followed the same collapse and the post-jtransform projection
/// matches it.
pub(crate) fn valuetype_to_concrete(vt: &ValueType) -> ConcreteType {
    match vt {
        // `Unsigned` shares the `Signed` ConcreteType — the codewriter
        // / regalloc do not distinguish signedness (`getkind(Unsigned)
        // == 'int'`); only the rtyper picks `IntegerRepr.lowleveltype
        // = Unsigned` based on `SomeInteger.unsigned`.
        ValueType::Int | ValueType::Unsigned | ValueType::Bool => ConcreteType::Signed,
        ValueType::Ref(_) | ValueType::Str | ValueType::StringBuilder => ConcreteType::GcRef,
        ValueType::Float => ConcreteType::Float,
        ValueType::Void => ConcreteType::Void,
        ValueType::State
        | ValueType::Unknown
        | ValueType::Int128
        | ValueType::UInt128
        | ValueType::SingleFloat => ConcreteType::Unknown,
    }
}

/// `result_kind: char` → `ConcreteType` projection used by jtransform
/// call families (`CallElidable` / `CallResidual` / `CallMayForce` /
/// `InlineCall` / `RecursiveCall`).
pub(crate) fn kind_char_to_concrete(kind: char) -> ConcreteType {
    match kind {
        'i' => ConcreteType::Signed,
        'r' => ConcreteType::GcRef,
        'f' => ConcreteType::Float,
        'v' => ConcreteType::Void,
        _ => ConcreteType::Unknown,
    }
}

fn concrete_if_known(concrete: ConcreteType) -> Option<ConcreteType> {
    if concrete == ConcreteType::Unknown {
        None
    } else {
        Some(concrete)
    }
}

/// Per-op `ConcreteType` declared by the rewritten graph's op-result
/// fields (`result_ty` / `result_kind`).  Authoritative for op-result
/// kinds because the rewriter declares them at lowering time, so this
/// projection wins over `original` operand inferences in
/// [`merge_synth_kinds`]'s precedence chain.
pub(crate) fn authoritative_result_type_from_op(kind: &OpKind) -> Option<ConcreteType> {
    match kind {
        OpKind::ConstInt(_) => Some(ConcreteType::Signed),
        OpKind::ConstUInt(_) => Some(ConcreteType::Signed),
        OpKind::ConstInt128(_) | OpKind::ConstUInt128(_) => None,
        OpKind::ConstBool(_) => Some(ConcreteType::Signed),
        // `_we_are_jitted` symbolic — `Bool` concretetype folds to int
        // kind; folded to `ConstBool(true)` by `jtransform` before emit.
        OpKind::ConstSymbolic { .. } => Some(ConcreteType::Signed),
        OpKind::ConstFloat(_) => Some(ConcreteType::Float),
        OpKind::ConstStr(_) => Some(ConcreteType::GcRef),
        OpKind::Input { ty, .. } => concrete_if_known(valuetype_to_concrete(ty)),
        OpKind::FieldRead { ty, .. } | OpKind::VableFieldRead { ty, .. } => {
            concrete_if_known(valuetype_to_concrete(ty))
        }
        OpKind::ArrayRead { item_ty, .. }
        | OpKind::InteriorFieldRead { item_ty, .. }
        | OpKind::VableArrayRead { item_ty, .. }
        // `raw_load_i/f` answers the element kind its descr names, the
        // same projection the GC array reads use.
        | OpKind::RawLoad { item_ty, .. } => {
            concrete_if_known(valuetype_to_concrete(item_ty))
        }
        OpKind::Call { result_ty, .. }
        | OpKind::IndirectCall { result_ty, .. }
        | OpKind::BinOp { result_ty, .. }
        | OpKind::UnaryOp { result_ty, .. } => concrete_if_known(valuetype_to_concrete(result_ty)),
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => {
            concrete_if_known(kind_char_to_concrete(*result_kind))
        }
        OpKind::VtableMethodPtr { .. } => Some(ConcreteType::Signed),
        // `arraylen_vable/rdd>i` answers a length, never the element kind.
        OpKind::VableArrayLen { .. } => Some(ConcreteType::Signed),
        _ => None,
    }
}

/// Walk the rewritten graph and collect every op-result that carries an
/// authoritative `ConcreteType` (per-op declaration), keyed on the backing
/// [`Variable`].  Feeds [`merge_synth_kinds`]'s `post_result` lane.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub(crate) fn authoritative_result_types(graph: &FunctionGraph) -> HashMap<Variable, ConcreteType> {
    let mut result = HashMap::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let Some(var) = op.result.as_ref() else {
                continue;
            };
            if let Some(concrete) = authoritative_result_type_from_op(&op.kind) {
                result.insert(var.clone(), concrete);
            }
        }
    }
    result
}

// `build_value_kinds` retired — the regalloc / flatten / assemble
// pipeline now reads kinds straight off the Variable's
// `.concretetype` cell (the upstream-orthodox source).
// Per-Variable `RegKind` projections happen at the use site via
// `regalloc::perform_register_allocation`'s internal
// `concretetype_to_regkind`, matching RPython's
// `getkind(v.concretetype)` access pattern bit for bit.

/// Stamp GC `getfield`/`setfield` bases into the Ref bank before regalloc.
///
/// RPython `history.getkind`: a `Ptr` whose `TO._gckind != 'raw'` is
/// `"ref"`, so `bhimpl_getfield_gc_*` is always
/// `@arguments("cpu", "r", "d", returns="X")`.  Pyre's MIR sometimes
/// leaves a GC object pointer as `Signed` (a Rust raw pointer /
/// address-sized word).  The assembler then keys the first argcode off
/// that bank and emits the pyre-only `getfield_gc_*/id>X` form.
///
/// Walk every surviving GC field/array access (unknown owner defaults
/// to `_gckind='gc'`, matching `jtransform.py rewrite_op_getfield`'s
/// `getattr(STRUCT, '_gckind', 'gc')`) and publish `GcRef` on a
/// `Signed`/`Unknown`/`Void` base.
///
/// Raw owners (`CallControl::struct_storage_for` →
/// `is_gc_managed=false`) are left alone: a `Signed` base stays an
/// address (`getfield_raw` / `setfield_raw`), a `GcRef` base stays a
/// Ref-banked raw struct (`PyType` / `CLASSTYPE` travels through the
/// Ref bank even though its storage is not a GC object).
///
/// `ArrayRead` / `ArrayWrite` / `ArrayLen` are the GC-array family
/// (`jtransform.py` emits `getarrayitem_gc` / `arraylen_gc` only when
/// `ARRAY._gckind == 'gc'`; raw arrays go through `raw_load` /
/// `getarrayitem_raw`).  A Signed base on those ops is the same
/// mis-banked GC pointer FieldRead had.
///
/// A Signed base that `make_three_lists_from_vars` already stored in an
/// int argument list keeps that cell. The list was split from the Signed
/// kind, and `emit_list_of_kind` requires the cell to still say int.
/// The access reads `cast_int_to_ptr` of the word, so regalloc colours a
/// ref base and the call list keeps the Signed argument.
pub(crate) fn promote_gc_field_bases(
    graph: &mut FunctionGraph,
    callcontrol: Option<&crate::call::CallControl>,
) {
    let int_args = int_argument_var_ids(graph);
    let graph_name = graph.name.clone();
    let mut redirects: Vec<(usize, usize)> = Vec::new();
    for (block_index, block) in graph.blocks.iter().enumerate() {
        for (op_index, op) in block.operations.iter().enumerate() {
            let Some((base, force_gc)) = gc_access_base(&op.kind, callcontrol) else {
                continue;
            };
            if !force_gc {
                continue;
            }
            if FunctionGraph::concretetype_of(base) == ConcreteType::Signed
                && int_args.contains(&base.id())
            {
                redirects.push((block_index, op_index));
                continue;
            }
            stamp_gc_ref_base(base, &graph_name);
        }
    }
    // Later inserts shift higher indices in the same block. Walk back so
    // each recorded index still names the access.
    for (block_index, op_index) in redirects.into_iter().rev() {
        redirect_signed_base_through_cast(graph, block_index, op_index);
    }
}

fn gc_access_base<'a>(
    kind: &'a OpKind,
    callcontrol: Option<&crate::call::CallControl>,
) -> Option<(&'a Variable, bool)> {
    match kind {
        OpKind::FieldRead { base, field, .. } | OpKind::FieldWrite { base, field, .. } => {
            Some((base, field_owner_is_gc(field, callcontrol)))
        }
        OpKind::VableFieldRead { base, .. }
        | OpKind::VableFieldWrite { base, .. }
        | OpKind::VableArrayRead { base, .. }
        | OpKind::VableArrayWrite { base, .. }
        | OpKind::VableArrayLen { base, .. } => Some((base, true)),
        // `nolength` is a raw items region (`ARRAY._gckind == 'raw'`).
        // Its base is the address integer (`getkind` → int). Promoting
        // that address to `GcRef` puts an `int_add` result in the ref
        // bank (`int_add/ii>r`), which no blackhole handler has.
        // A length-prefixed array is the GC family and stays `GcRef`.
        OpKind::ArrayRead { base, nolength, .. }
        | OpKind::ArrayWrite { base, nolength, .. }
        | OpKind::ArrayLen { base, nolength, .. } => Some((base, !nolength)),
        _ => None,
    }
}

fn gc_access_base_mut(kind: &mut OpKind) -> Option<&mut Variable> {
    match kind {
        OpKind::FieldRead { base, .. }
        | OpKind::FieldWrite { base, .. }
        | OpKind::VableFieldRead { base, .. }
        | OpKind::VableFieldWrite { base, .. }
        | OpKind::VableArrayRead { base, .. }
        | OpKind::VableArrayWrite { base, .. }
        | OpKind::VableArrayLen { base, .. }
        | OpKind::ArrayRead { base, .. }
        | OpKind::ArrayWrite { base, .. }
        | OpKind::ArrayLen { base, .. } => Some(base),
        _ => None,
    }
}

/// Variable ids sitting in an int argument list. `emit_list_of_kind`
/// asserts each of those cells is still Signed at assemble time.
fn int_argument_var_ids(graph: &FunctionGraph) -> HashSet<u64> {
    let mut ids = HashSet::new();
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::CallResidual { args_i, .. }
                | OpKind::CallMayForce { args_i, .. }
                | OpKind::CallElidable { args_i, .. }
                | OpKind::InlineCall { args_i, .. }
                | OpKind::ConditionalCall { args_i, .. }
                | OpKind::ConditionalCallValue { args_i, .. }
                | OpKind::RecordKnownResult { args_i, .. } => {
                    extend_var_ids(&mut ids, args_i);
                }
                OpKind::RecursiveCall {
                    greens_i, reds_i, ..
                }
                | OpKind::JitMergePoint {
                    greens_i, reds_i, ..
                } => {
                    extend_var_ids(&mut ids, greens_i);
                    extend_var_ids(&mut ids, reds_i);
                }
                _ => {}
            }
        }
    }
    ids
}

fn extend_var_ids(ids: &mut HashSet<u64>, vars: &[Variable]) {
    for var in vars {
        ids.insert(var.id());
    }
}

fn redirect_signed_base_through_cast(
    graph: &mut FunctionGraph,
    block_index: usize,
    op_index: usize,
) {
    let original = gc_access_base_mut(&mut graph.blocks[block_index].operations[op_index].kind)
        .expect("redirect target is a GC field or array access")
        .clone();
    let cast_result = graph.alloc_value_var_with_type(ConcreteType::GcRef);
    *gc_access_base_mut(&mut graph.blocks[block_index].operations[op_index].kind)
        .expect("redirect target is a GC field or array access") = cast_result.clone();
    graph.blocks[block_index].operations.insert(
        op_index,
        SpaceOperation {
            result: Some(cast_result),
            kind: OpKind::UnaryOp {
                op: "cast_int_to_ptr".into(),
                operand: original,
                result_ty: ValueType::Ref(None),
            },
        },
    );
}

fn stamp_gc_ref_base(base: &crate::flowspace::model::Variable, graph_name: &str) {
    match FunctionGraph::concretetype_of(base) {
        ConcreteType::GcRef => {}
        ConcreteType::Float => panic!(
            "GC field/array base {base:?} has Float concretetype — \
             history.getkind(Ptr(GC)) is 'ref' (graph {graph_name})"
        ),
        ConcreteType::Signed | ConcreteType::Unknown | ConcreteType::Void => {
            FunctionGraph::set_concretetype_of_inline(base, ConcreteType::GcRef);
        }
    }
}

pub(crate) fn field_owner_is_gc(
    field: &crate::model::FieldDescriptor,
    callcontrol: Option<&crate::call::CallControl>,
) -> bool {
    let Some(owner) = field.owner_root.as_deref() else {
        return true;
    };
    callcontrol
        .and_then(|cc| cc.struct_storage_for(owner))
        .map(|(is_gc, _)| is_gc)
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FieldDescriptor, OpKind, SpaceOperation, ValueType};

    fn push_input(
        graph: &mut FunctionGraph,
        name: &str,
        ty: ValueType,
    ) -> crate::flowspace::model::Variable {
        let var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: name.into(),
                    ty,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(graph.startblock, var.clone());
        var
    }

    #[test]
    fn promote_gc_field_bases_lifts_a_signed_gc_owner_into_the_ref_bank() {
        let mut graph = FunctionGraph::new("int_base_getfield");
        let base = push_input(&mut graph, "obj", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));
        FunctionGraph::set_concretetype_of_inline(&base, ConcreteType::Signed);

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&base),
            ConcreteType::GcRef,
            "a GC FieldRead base that arrived as Signed must be published as GcRef"
        );
    }

    #[test]
    fn promote_gc_field_bases_lifts_a_signed_array_read_base() {
        let mut graph = FunctionGraph::new("int_base_getarrayitem");
        let base = push_input(&mut graph, "arr", ValueType::Int);
        let index = push_input(&mut graph, "i", ValueType::Int);
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayRead {
                    base: base.clone(),
                    index,
                    item_ty: ValueType::Int,
                    array_type_id: None,
                    nolength: false,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));
        FunctionGraph::set_concretetype_of_inline(&base, ConcreteType::Signed);

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&base),
            ConcreteType::GcRef,
            "a GC ArrayRead base that arrived as Signed must be published as GcRef"
        );
    }

    #[test]
    fn promote_gc_field_bases_leaves_a_headerless_array_address_signed() {
        let (array_type_id, nolength) =
            crate::front::mir::fixed_array_index_identity(false, "&[u32]");
        assert!(nolength, "a [u32] item run has no length header");
        assert!(crate::front::typestr::nolength_from_array_type_id(
            array_type_id.as_deref()
        ));

        let mut graph = FunctionGraph::new("raw_slice_address");
        let addr = push_input(&mut graph, "addr", ValueType::Int);
        let index = push_input(&mut graph, "i", ValueType::Int);
        let value = push_input(&mut graph, "v", ValueType::Int);
        FunctionGraph::set_concretetype_of_inline(&addr, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&index, ConcreteType::Signed);
        graph
            .block_mut(graph.startblock)
            .operations
            .push(crate::model::SpaceOperation {
                result: None,
                kind: OpKind::ArrayWrite {
                    base: addr.clone(),
                    index,
                    value: crate::model::LinkArg::Value(value),
                    item_ty: ValueType::Int,
                    array_type_id,
                    nolength,
                },
            });

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&addr),
            ConcreteType::Signed,
            "nolength raw slice base stays an int"
        );
    }

    #[test]
    fn promote_gc_field_bases_lifts_an_unknown_base() {
        let mut graph = FunctionGraph::new("unknown_base_getfield");
        let base = push_input(&mut graph, "obj", ValueType::Ref(None));
        let result = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result));
        assert_eq!(FunctionGraph::concretetype_of(&base), ConcreteType::Unknown);

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&base),
            ConcreteType::GcRef,
            "the post-rewrite pass still publishes an untyped GC field base"
        );
    }

    #[test]
    fn promote_gc_field_bases_casts_a_signed_base_that_is_an_int_call_argument() {
        let mut graph = FunctionGraph::new("int_arg_and_field");
        let base = push_input(&mut graph, "obj", ValueType::Int);
        let other = push_input(&mut graph, "n", ValueType::Ref(None));
        FunctionGraph::set_concretetype_of_inline(&base, ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&other, ConcreteType::GcRef);
        let read = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(read));
        graph
            .block_mut(graph.startblock)
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::InlineCall {
                    jitcode: crate::jitcode::JitCodeHandle::new(std::sync::Arc::new(
                        crate::jitcode::JitCode::new("callee"),
                    )),
                    args_i: vec![base.clone()],
                    args_r: vec![other],
                    args_f: Vec::new(),
                    result_kind: 'r',
                },
            });

        promote_gc_field_bases(&mut graph, None);

        assert_eq!(
            FunctionGraph::concretetype_of(&base),
            ConcreteType::Signed,
            "an int-list argument stays Signed so emit_list_of_kind still matches"
        );
        let ops = &graph.block(graph.startblock).operations;
        let cast_at = ops
            .iter()
            .position(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, operand, .. }
                        if op == "cast_int_to_ptr" && operand.id() == base.id()
                )
            })
            .expect("cast_int_to_ptr of the signed argument");
        let field_at = ops
            .iter()
            .position(|op| matches!(&op.kind, OpKind::FieldRead { .. }))
            .expect("field read");
        assert!(cast_at < field_at, "the cast dominates the field read");
        let OpKind::FieldRead {
            base: field_base, ..
        } = &ops[field_at].kind
        else {
            unreachable!("field read");
        };
        assert_ne!(field_base.id(), base.id());
        assert_eq!(
            FunctionGraph::concretetype_of(field_base),
            ConcreteType::GcRef
        );
        assert!(ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::InlineCall { args_i, .. } if args_i.iter().any(|arg| arg.id() == base.id())
        )));
    }

    #[test]
    fn field_owner_is_gc_defaults_unknown_owners_to_gc() {
        let named = FieldDescriptor::new("x", Some("Point".into()));
        let unnamed = FieldDescriptor::new("x", None);
        assert!(
            field_owner_is_gc(&named, None),
            "jtransform.py getattr(STRUCT, '_gckind', 'gc') defaults to gc"
        );
        assert!(
            field_owner_is_gc(&unnamed, None),
            "a descriptor with no owner_root is treated as GC"
        );
    }
}
