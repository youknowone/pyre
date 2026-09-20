//! `Layout::from_size_align(size, const_align).ok()` → a native power-of-two
//! bound check + a virtualized `Option<Layout>` (nested transparent ctors), and
//! the sibling `Layout::from_size_align(size, const_align).expect(msg)` → the
//! same bound check + a by-value virtualized `Layout` (no `Option` wrapper),
//! branching to an implicit raise on the overflowed (`Err`) case
//! ([`rewire_from_size_align_expect_sites`]).
//!
//! ## Positioning
//!
//! `core::alloc::layout::Layout::from_size_align` and `core::result::Result::ok`
//! are Opaque core bodies (Charon cannot extract `core`), so the caller carries
//! two residual calls the rtyper census cannot type:
//!   - `res = from_size_align(size, align)` → `Result<Layout, LayoutError>`,
//!   - `opt = res.ok()` → `Option<Layout>`.
//!     Both cross a value the census cannot model as a residual return (`Layout` is
//!     a 2-word `{size, align}` aggregate; `Result`/`Option` of it likewise), so
//!     the pair is the standing wall in `rlist::try_typed_items_block_layout`
//!     once `checked_mul`/`checked_add` are natively lowered
//!     ([`crate::front::checked_arith_uint`]).
//!
//! `from_size_align(size, align)` returns `Ok(Layout { size, align })` iff
//! `align.is_power_of_two()` and `size <= isize::MAX - (align - 1)`.  The
//! frontend folds `align_of::<T>()` — ADT layouts and primitive widths
//! including `usize` — to a compile-time `ConstInt` power of two, so
//! `align.is_power_of_two()` is statically true and the bound `isize::MAX -
//! (align - 1)` is a constant.  The residual pair therefore has a native form:
//!   - `too_big = uint_lt(bound, size)` — `size > bound`, the overflowed case;
//!   - the `Option` tag is `Some` (`1`) iff `too_big` is `0` (fits), else `None`.
//!
//! ## The rewrite (`rewire_one_from_size_align_site`)
//!
//! The `.ok()` residual (block Q) produces the `Option<Layout>`; its single arg
//! is the `from_size_align` residual result (block P).  The rewrite drops both
//! residuals and, **in place** in block P, emits the native sequence plus a
//! virtualized nested aggregate reusing the `from_size_align` result var as the
//! `Option` ctor result:
//!   - `bound = ConstInt(isize::MAX - (align - 1))`, `too_big = uint_lt(bound,
//!     size)`, `disc = eq(too_big, 0)`;
//!   - `layout = <transparent Layout ctor>{ __pos_0 = size, __pos_1 = align }`;
//!   - `opt = <transparent Option ctor>{ __discriminant = disc, __pos_0(Some) =
//!     layout }`.
//!     Block Q's `ok` call is deleted and its result aliased to the block-P
//!     `Option` value threaded across the P→Q edge — no diamond, no exit rewiring.
//!
//! Like [`crate::front::checked_arith_uint`] it is **order-independent** and
//! **fail-safe**: any structural mismatch returns `Err`, the caller leaves both
//! residual calls untouched, and the unregistered callees keep the rtyper
//! census Skip (no regression).

use crate::flowspace::model::Variable;
use crate::front::bool_then::{
    close_goto_mixed, emit_option_variant, map_source, reproduce_exit_args,
};
use crate::front::checked_arith_uint::{push_binop, push_const_int};
use crate::model::{
    BlockId, CallTarget, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

/// A recorded `Layout::from_size_align(..).ok()` call whose result is an
/// `Option<Layout>`, captured during body lowering with the `Option`/`Some`
/// owners, the `Layout` payload owner, and the payload type resolved from the
/// `.ok()` destination type.
#[derive(Clone)]
pub(crate) struct FromSizeAlignSite {
    /// The `.ok()` call result (the `Option<Layout>` value) — locates block Q.
    pub ok_result: Variable,
    /// The `Option` enum root `name_path` — the `__discriminant` field owner
    /// and the ctor owner.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload owner.
    pub some_owner: String,
    /// The `Option` payload `Layout` projected to a [`ValueType`] — the
    /// `Some::__pos_0` field kind (a `Ref` to the `Layout` aggregate).
    pub payload_ty: ValueType,
    /// The `Layout` ADT `name_path` — the nested transparent-ctor owner and its
    /// `__pos_0`/`__pos_1` field owner.
    pub layout_owner: String,
}

/// A recorded `Layout::from_size_align(..).expect(msg)` call whose result is a
/// by-value `Layout` (the `Result::Ok` payload), captured during body lowering.
/// The `.expect()` receiver is a `Result<Layout, LayoutError>`, so — unlike the
/// `.ok()` shape — there is no `Option` wrapper: the rewrite virtualizes the
/// `Layout` directly and raises on the overflowed (`Err`) case.
#[derive(Clone)]
pub(crate) struct FromSizeAlignExpectSite {
    /// The `.expect()` call result (the by-value `Layout`) — locates block Q.
    pub result_var: Variable,
    /// The `Layout` ADT `name_path` — the transparent-ctor owner and its
    /// `__pos_0`/`__pos_1` field owner.
    pub layout_owner: String,
}

/// Rewrite every recorded `from_size_align(..).ok()` site into the native
/// bound-check + virtualized `Option<Layout>` shape.  Returns the number of
/// sites rewritten; declined sites keep their residual calls (census Skip).
pub(crate) fn rewire_from_size_align_sites(
    graph: &mut FunctionGraph,
    sites: &[FromSizeAlignSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_from_size_align_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                if std::env::var_os("MAJIT_MIR_FRONTEND_DEBUG").is_some() {
                    eprintln!(
                        "[from_size_align] {} decline at {:?}: {_decline}",
                        graph.name, site.ok_result
                    );
                }
                // Leave both residual calls; the unregistered callees make the
                // rtyper census Skip this graph (no regression).
            }
        }
    }
    rewritten
}

/// `true` iff `path` names the `Layout` ADT, whatever crate Charon prefixes
/// (`core::alloc::layout::Layout` from the TypeDecl, `alloc::layout::Layout`
/// after `strip_crate_prefix`).
pub(crate) fn is_layout_adt_owner(path: &str) -> bool {
    path == "core::alloc::layout::Layout" || path == "alloc::layout::Layout"
}

/// `true` iff `segments` names `Layout::from_size_align`.
///
/// Two Charon spellings reach the residual:
/// - associated-function owner qualification `[.., "layout", "Layout", "from_size_align"]`
///   (the call-site `CallTarget::FunctionPath` after `impl_method_owner_for_fundecl`);
/// - the FunDecl `name_path` `[.., "layout", "<Impl>", "from_size_align"]`
///   (`core::alloc::layout::<Impl>::from_size_align`), used when the Method hint
///   is declined and the raw declaration path is kept.
fn is_layout_from_size_align(segments: &[String]) -> bool {
    if segments.last().map(String::as_str) != Some("from_size_align") {
        return false;
    }
    match segments
        .get(segments.len().wrapping_sub(2))
        .map(String::as_str)
    {
        Some("Layout") => true,
        Some("<Impl>") => {
            segments
                .get(segments.len().wrapping_sub(3))
                .map(String::as_str)
                == Some("layout")
        }
        _ => false,
    }
}

fn is_layout_from_size_align_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => is_layout_from_size_align(segments),
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => name == "from_size_align" && receiver_root.as_deref() == Some("Layout"),
        _ => false,
    }
}

/// `true` iff `segments` names `Result::ok` — Method-hint owner qualification
/// `[.., "result", "Result", "ok"]` or the FunDecl `name_path`
/// `[.., "result", "<Impl>", "ok"]`.
fn result_method_segments(segments: &[String], leaf: &str) -> bool {
    if segments.last().map(String::as_str) != Some(leaf) {
        return false;
    }
    match segments
        .get(segments.len().wrapping_sub(2))
        .map(String::as_str)
    {
        Some("Result") => true,
        Some("<Impl>") => {
            segments
                .get(segments.len().wrapping_sub(3))
                .map(String::as_str)
                == Some("result")
        }
        _ => false,
    }
}

fn is_result_ok_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => name == "ok" && receiver_root.as_deref() == Some("Result"),
        CallTarget::FunctionPath { segments, .. } => result_method_segments(segments, "ok"),
        _ => false,
    }
}

fn is_result_expect_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => name == "expect" && receiver_root.as_deref() == Some("Result"),
        CallTarget::FunctionPath { segments, .. } => result_method_segments(segments, "expect"),
        _ => false,
    }
}

fn result_ok_receiver(kind: &OpKind) -> Option<LinkArg> {
    match kind {
        OpKind::Call { target, args, .. } if args.len() == 1 && is_result_ok_target(target) => {
            Some(args[0].clone())
        }
        _ => None,
    }
}

fn result_expect_receiver(kind: &OpKind) -> Option<LinkArg> {
    match kind {
        OpKind::Call { target, args, .. } if args.len() == 2 && is_result_expect_target(target) => {
            Some(args[0].clone())
        }
        _ => None,
    }
}

/// Locate the unique predecessor whose last op is threaded into `recv` on
/// the single edge into `q`.  Returns `(p_index, produced_var, q_input)`.
/// `produced_var` is the predecessor result; `q_input` is the block-Q
/// inputarg that received it — they differ when the edge is an SSA copy.
fn find_from_size_align_pred(
    graph: &FunctionGraph,
    q: usize,
    recv: &LinkArg,
    name: &str,
) -> Result<(usize, Variable, Variable), String> {
    graph
        .blocks
        .iter()
        .enumerate()
        .find_map(|(index, block)| {
            let [exit] = block.exits.as_slice() else {
                return None;
            };
            if exit.target != graph.blocks[q].id {
                return None;
            }
            let produced = block.operations.last()?.result.as_ref()?;
            exit.args
                .iter()
                .zip(&graph.blocks[q].inputargs)
                .find_map(|(arg, input)| {
                    let LinkArg::Value(source) = arg else {
                        return None;
                    };
                    (source == produced && (input == recv || source == recv))
                        .then(|| (index, source.clone(), input.clone()))
                })
        })
        .ok_or_else(|| format!("{name}: from_size_align result is not threaded into the consumer"))
}

/// The numeric align passed to `from_size_align`, recovered from a
/// `ConstInt` / `ConstUInt` producer.  Follows a unique SSA copy across
/// block inputargs (`resolve_to_producer_op`) so a folded `align_of` that
/// lives in a predecessor still counts; a merge or a residual call
/// declines.
fn folded_align_const(
    graph: &FunctionGraph,
    align_arg: &Variable,
    name: &str,
) -> Result<i64, String> {
    let decline = format!("{name}: from_size_align align arg is not a folded ConstInt");
    let (block_id, idx) = crate::front::mir::resolve_to_producer_op(graph, align_arg)
        .ok_or_else(|| decline.clone())?;
    let block = graph
        .blocks
        .iter()
        .find(|b| b.id == block_id)
        .ok_or_else(|| decline.clone())?;
    let align = match block.operations.get(idx).map(|op| &op.kind) {
        Some(OpKind::ConstInt(n)) => *n,
        Some(OpKind::ConstUInt(n)) => i64::try_from(*n).map_err(|_| decline.clone())?,
        _ => return Err(decline),
    };
    if align <= 0 || !(align as u64).is_power_of_two() {
        return Err(format!(
            "{name}: from_size_align align {align} is not a power of two"
        ));
    }
    Ok(align)
}

fn from_size_align_operands(kind: &OpKind) -> Option<(Variable, Variable)> {
    match kind {
        OpKind::Call { target, args, .. }
            if args.len() == 2 && is_layout_from_size_align_target(target) =>
        {
            Some((
                args[0].clone().into_variable(),
                args[1].clone().into_variable(),
            ))
        }
        _ => None,
    }
}

fn rewire_one_from_size_align_site(
    graph: &mut FunctionGraph,
    site: &FromSizeAlignSite,
) -> Result<(), String> {
    let name = graph.name.clone();
    let ok = &site.ok_result;

    // Block Q: the `.ok()` residual call producing `ok`, closed by `lower_call`
    // with a single forwarding exit.  The call must still be Q's last op — a
    // `Result::ok` Method call with a single arg (the `from_size_align` result).
    let q = graph
        .blocks
        .iter()
        .position(|b| b.operations.iter().any(|op| op.result.as_ref() == Some(ok)))
        .ok_or_else(|| format!("{name}: .ok() result var has no producer block"))?;
    let ok_idx = graph.blocks[q].operations.len() - 1;
    let ok_op = &graph.blocks[q].operations[ok_idx];
    let ok_arg = match (&ok_op.result, result_ok_receiver(&ok_op.kind)) {
        (Some(r), Some(arg)) if r == ok => arg,
        _ => {
            return Err(format!(
                "{name}: block {q} last op is not the Result::ok call producing {ok:?}"
            ));
        }
    };

    // Block P: the `from_size_align` residual producing `fsa_res` as its last
    // op — a 2-arg `[..]::Layout::from_size_align` FunctionPath call.
    let (p, fsa_res, fsa_in_q) = find_from_size_align_pred(graph, q, &ok_arg, &name)?;
    let fsa_idx = graph.blocks[p].operations.len() - 1;
    let (size, align_arg) = match &graph.blocks[p].operations[fsa_idx] {
        SpaceOperation {
            result: Some(r),
            kind,
        } if r == &fsa_res => from_size_align_operands(kind).ok_or_else(|| {
            format!("{name}: block {p} last op is not the 2-arg Layout::from_size_align call")
        })?,
        _ => {
            return Err(format!(
                "{name}: block {p} last op is not the 2-arg Layout::from_size_align call"
            ));
        }
    };

    // Folded `align_of::<T>()` — a `ConstInt` power of two, possibly an SSA
    // copy of a predecessor's const (MIR often emits `align_of` in its own
    // block).  A residual call or a phi merge declines.
    let align = folded_align_const(graph, &align_arg, &name)?;
    // `Ok` iff `size <= isize::MAX - (align - 1)`.  `isize::MAX == i64::MAX`.
    let bound = i64::MAX - (align - 1);

    // --- All structural validation passed; mutate the graph. ---

    crate::front::bool_then::validate_dynamic_option_exit(graph, graph.blocks[p].id)?;
    // The rewritten .ok() forwards Q's own input, not the now-removed P
    // result. RPython Link.args -> Block.inputargs is the value boundary.
    // Extra Q operations reading a legacy cross-block alias need their own
    // normalization before this fold can remove that alias's definition.
    if ok_arg != fsa_in_q && ok_idx != 0 {
        return Err(format!(
            "{name}: .ok() has an unnormalized cross-block operand"
        ));
    }

    let p_id = graph.blocks[p].id;
    // Drop the residual `from_size_align` call (P's last op) so `fsa_res` is
    // produced solely by the virtualized `Option` ctor appended below.
    graph.blocks[p].operations.truncate(fsa_idx);

    // Native bound check: `too_big = uint_lt(bound, size)` (size > bound → the
    // overflowed case), `disc = eq(too_big, 0)` (Some when it fits, else None).
    let bound_var = push_const_int(graph, p_id, bound);
    let too_big = push_binop(
        graph,
        p_id,
        "uint_lt",
        bound_var,
        size.clone(),
        ValueType::Int,
    );
    let zero = push_const_int(graph, p_id, 0);
    let disc = push_binop(graph, p_id, "eq", too_big, zero, ValueType::Int);

    // Forward the old continuation's live values through both arms. Layout
    // is constructed only after the bound check succeeds, just as the native
    // from_size_align source does; the None arm has no payload allocation.
    let saved_exit = graph.block(p_id).exits[0].clone();
    let mut carried = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(value) = arg
            && value != &fsa_res
            && !carried.contains(value)
        {
            carried.push(value.clone());
        }
    }
    let mut some_sources = carried.clone();
    for value in [&size, &align_arg] {
        if !some_sources.contains(value) {
            some_sources.push(value.clone());
        }
    }
    let (some_block, some_inputs) = graph.create_block_with_arg_vars(some_sources.len());
    let (none_block, none_inputs) = graph.create_block_with_arg_vars(carried.len());
    let layout = graph.alloc_value_var();
    build_layout_aggregate(
        graph,
        some_block,
        &site.layout_owner,
        layout.clone(),
        map_source(&some_sources, &some_inputs, &size).expect("size threaded"),
        map_source(&some_sources, &some_inputs, &align_arg).expect("align threaded"),
    );
    let some = emit_option_variant(
        graph,
        some_block,
        &site.option_owner,
        1,
        Some((&site.some_owner, layout, site.payload_ty.clone())),
    );
    let none = emit_option_variant(graph, none_block, &site.option_owner, 0, None);
    for (arm, value, sources, inputs) in [
        (some_block, some, &some_sources, &some_inputs),
        (none_block, none, &carried, &none_inputs),
    ] {
        let args = reproduce_exit_args(&saved_exit, &fsa_res, &value, sources, inputs, &name)
            .expect("all continuation values threaded");
        close_goto_mixed(graph, arm, saved_exit.target, args);
    }
    graph.set_branch(p_id, disc, some_block, some_sources, none_block, carried);

    // Block Q: drop the `.ok()` call and alias its result to the block-P
    // `Option` threaded across the P→Q edge (`fsa_res` is a Q inputarg).
    graph.blocks[q].operations.truncate(ok_idx);
    for exit in &mut graph.blocks[q].exits {
        for arg in &mut exit.args {
            if let LinkArg::Value(v) = arg
                && v == ok
            {
                *v = fsa_in_q.clone();
            }
        }
    }

    Ok(())
}

/// Rewrite every recorded `from_size_align(..).expect(msg)` site into the
/// native bound-check + a by-value virtualized `Layout`, raising on overflow.
/// Returns the number of sites rewritten; declined sites keep their residual
/// calls (census Skip).
pub(crate) fn rewire_from_size_align_expect_sites(
    graph: &mut FunctionGraph,
    sites: &[FromSizeAlignExpectSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_from_size_align_expect_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                if std::env::var_os("MAJIT_MIR_FRONTEND_DEBUG").is_some() {
                    eprintln!(
                        "[from_size_align expect] {} decline at {:?}: {_decline}",
                        graph.name, site.result_var
                    );
                }
                // Leave both residual calls; the unregistered callees make the
                // rtyper census Skip this graph (no regression).
            }
        }
    }
    rewritten
}

fn rewire_one_from_size_align_expect_site(
    graph: &mut FunctionGraph,
    site: &FromSizeAlignExpectSite,
) -> Result<(), String> {
    let name = graph.name.clone();
    let result = &site.result_var;

    // Block Q: the `.expect()` residual call producing `result`, closed by
    // `lower_call` with a single forwarding exit.  The call must still be Q's
    // last op — a `Result::expect` Method call with two args (the
    // `from_size_align` result and the panic message).
    let q = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(result))
        })
        .ok_or_else(|| format!("{name}: .expect() result var has no producer block"))?;
    let expect_idx = graph.blocks[q].operations.len() - 1;
    let expect_op = &graph.blocks[q].operations[expect_idx];
    let expect_arg = match (
        expect_op.result.as_ref(),
        result_expect_receiver(&expect_op.kind),
    ) {
        (Some(r), Some(arg)) if r == result => arg,
        _ => {
            return Err(format!(
                "{name}: block {q} last op is not the Result::expect call producing {result:?}"
            ));
        }
    };

    // Block P: the `from_size_align` residual producing `fsa_res` as its last op
    // — a 2-arg `[..]::Layout::from_size_align` FunctionPath call.  The expect
    // receiver is often Q's inputarg (an SSA copy), not P's result var.
    let (p, fsa_res, fsa_in_q) = find_from_size_align_pred(graph, q, &expect_arg, &name)?;
    let fsa_idx = graph.blocks[p].operations.len() - 1;
    let (size, align_arg) = match &graph.blocks[p].operations[fsa_idx] {
        SpaceOperation {
            result: Some(r),
            kind,
        } if r == &fsa_res => from_size_align_operands(kind).ok_or_else(|| {
            format!("{name}: block {p} last op is not the 2-arg Layout::from_size_align call")
        })?,
        _ => {
            return Err(format!(
                "{name}: block {p} last op is not the 2-arg Layout::from_size_align call"
            ));
        }
    };

    // Folded `align_of::<T>()` — a `ConstInt` power of two, possibly an SSA
    // copy of a predecessor's const.  A residual call or a phi merge declines.
    let align = folded_align_const(graph, &align_arg, &name)?;
    // `Ok` iff `size <= isize::MAX - (align - 1)`.  `isize::MAX == i64::MAX`.
    let bound = i64::MAX - (align - 1);

    // Block P must forward `fsa_res` to Q via a single plain goto — the shape
    // the branch replaces (the `Ok` arm re-forwards P's original link args).
    let q_id = graph.blocks[q].id;
    let p_goto_args: Vec<Variable> = {
        let [exit] = graph.blocks[p].exits.as_slice() else {
            return Err(format!(
                "{name}: from_size_align block {p} is not single-exit"
            ));
        };
        if exit.exitcase.is_some() || exit.target != q_id {
            return Err(format!(
                "{name}: from_size_align block {p} exit is not a plain goto to the expect block"
            ));
        }
        exit.args
            .iter()
            .map(|a| match a {
                LinkArg::Value(v) => Ok(v.clone()),
                _ => Err(format!(
                    "{name}: from_size_align P->Q link carries a non-Value arg"
                )),
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    if !p_goto_args.contains(&fsa_res) {
        return Err(format!(
            "{name}: fsa_res is not forwarded across the P->Q edge"
        ));
    }

    // --- All structural validation passed; mutate the graph. ---

    let p_id = graph.blocks[p].id;
    // Drop the residual `from_size_align` call (P's last op).
    graph.blocks[p].operations.truncate(fsa_idx);

    // Native bound check: `too_big = uint_lt(bound, size)` (size > bound → the
    // overflowed case), `disc = eq(too_big, 0)` (Ok when it fits, else Err).
    let bound_var = push_const_int(graph, p_id, bound);
    let too_big = push_binop(
        graph,
        p_id,
        "uint_lt",
        bound_var,
        size.clone(),
        ValueType::Int,
    );
    let zero = push_const_int(graph, p_id, 0);
    let disc = push_binop(graph, p_id, "eq", too_big, zero, ValueType::Int);

    // The by-value `Layout { size, align }` reuses `fsa_res` — the value P
    // already forwards to Q — so the `Ok` arm needs no fresh threading.
    build_layout_aggregate(
        graph,
        p_id,
        &site.layout_owner,
        fsa_res.clone(),
        size,
        align_arg,
    );

    // `Err` arm (overflow): raise the implicit `AssertionError` — `expect`
    // panics on `Err`, the never-`Err` contract making this the "shouldn't
    // occur" shape `remove_assertion_errors` prunes.
    let (else_bb, _else_inputs) = graph.create_block_with_arg_vars(0);
    graph.set_raise_implicit(
        else_bb,
        "Layout::from_size_align().expect() on overflowing size",
    );

    // P branches on `disc`: fits (`1`) → Q with P's original link args (the
    // reused `fsa_res` now carrying the virtualized `Layout`); overflow (`0`) →
    // the raise arm.
    graph.set_branch(p_id, disc, q_id, p_goto_args, else_bb, Vec::new());

    // Block Q: drop the `.expect()` call and alias its result to the Q input
    // that received P's virtualized `Layout` (`fsa_res` is P's result; the
    // edge copy is `fsa_in_q`).
    graph.blocks[q].operations.truncate(expect_idx);
    for exit in &mut graph.blocks[q].exits {
        for arg in &mut exit.args {
            if let LinkArg::Value(v) = arg
                && v == result
            {
                *v = fsa_in_q.clone();
            }
        }
    }

    Ok(())
}

/// Build a `Layout { __pos_0 = size, __pos_1 = align }` transparent-ctor
/// aggregate in `block`, writing it into `result`.  Both fields are `Unsigned`
/// (`usize`); the aggregate is virtualized (the ctor + `FieldWrite` chain folds
/// to SSA), so nothing materializes a real `Layout`.  `result` is a fresh var
/// for the `.ok()` `Some` payload, or the reused `from_size_align` result var
/// for the by-value `.expect()` shape.
fn build_layout_aggregate(
    graph: &mut FunctionGraph,
    block: BlockId,
    layout_owner: &str,
    result: Variable,
    size: Variable,
    align: Variable,
) {
    let mut owner_path = crate::model::split_qualified_path(layout_owner);
    let ctor_name = owner_path.pop().unwrap_or_default();
    let ctor_target = if owner_path.is_empty() {
        CallTarget::synthetic_transparent_ctor(ctor_name)
    } else {
        CallTarget::synthetic_transparent_ctor_with_owner(owner_path, ctor_name)
    };
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(result.clone()),
        kind: OpKind::Call {
            target: ctor_target,
            args: Vec::new(),
            result_ty: ValueType::Ref(Some(layout_owner.to_string())),
        },
    });
    for (name, value) in [("__pos_0", size), ("__pos_1", align)] {
        graph.block_mut(block).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: result.clone(),
                field: FieldDescriptor {
                    name: name.to_string(),
                    owner_root: Some(layout_owner.to_string()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                },
                value: LinkArg::Value(value),
                ty: ValueType::Unsigned,
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::CallTarget;

    fn fsa_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["alloc", "layout", "Layout", "from_size_align"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn ok_target() -> CallTarget {
        CallTarget::Method {
            name: "ok".to_string(),
            receiver_root: Some("Result".to_string()),
            resolved_path: None,
            fun_decl_id: None,
        }
    }

    fn site_for(ok_result: &Variable) -> FromSizeAlignSite {
        FromSizeAlignSite {
            ok_result: ok_result.clone(),
            option_owner: "core::option::Option".to_string(),
            some_owner: "core::option::Option::Some".to_string(),
            payload_ty: ValueType::Ref(Some("core::alloc::layout::Layout".to_string())),
            layout_owner: "core::alloc::layout::Layout".to_string(),
        }
    }

    /// Build P (from_size_align) → Q (.ok()) → returnblock with `align` folded
    /// to a `ConstInt` in P.  Returns the graph and the `.ok()` result var.
    fn build_site() -> (FunctionGraph, Variable) {
        let mut g = FunctionGraph::new("test_from_size_align");
        let p = g.startblock;
        let size = g.push_op_var(p, OpKind::ConstInt(64), true).unwrap();
        let align = g.push_op_var(p, OpKind::ConstInt(8), true).unwrap();
        let fsa = g
            .push_op_var(
                p,
                OpKind::Call {
                    target: fsa_target(),
                    args: crate::model::call_args(vec![size, align]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        // Q consumes its own inputarg, reached by the P result's link arg.
        let (q, q_args) = g.create_block_with_arg_vars(1);
        let ok = g
            .push_op_var(
                q,
                OpKind::Call {
                    target: ok_target(),
                    args: crate::model::call_args(vec![q_args[0].clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(q, cont, vec![ok.clone()]);
        g.set_goto(p, q, vec![fsa]);
        (g, ok)
    }

    #[test]
    fn from_size_align_ok_lowers_to_bound_check_and_nested_option() {
        let (mut g, ok) = build_site();
        let q = g
            .blocks
            .iter()
            .find(|block| {
                block
                    .operations
                    .iter()
                    .any(|op| op.result.as_ref() == Some(&ok))
            })
            .expect("ok block")
            .id;
        let q_input = g.block(q).inputargs[0].clone();
        let rewritten = rewire_from_size_align_sites(&mut g, &[site_for(&ok)]);
        assert_eq!(rewritten, 1, "the from_size_align().ok() site must rewrite");
        assert!(
            matches!(&g.block(q).exits[0].args[0], LinkArg::Value(value) if value == &q_input),
            ".ok() must forward the Q input, not the deleted P result"
        );

        // No residual FunctionPath / Method call survives.
        let has_residual = g.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { .. } | CallTarget::Method { .. },
                    ..
                }
            )
        });
        assert!(
            !has_residual,
            "both from_size_align and ok residuals must be gone"
        );

        // The bound-check binops (`uint_lt` + `eq`) landed.
        let binops: Vec<String> = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::BinOp { op, .. } => Some(op.clone()),
                _ => None,
            })
            .collect();
        assert!(binops.contains(&"uint_lt".to_string()));
        assert!(binops.contains(&"eq".to_string()));

        // A nested Layout and the two concrete Option variants exist;
        // the enum base must never be allocated and given a payload.
        let ctor_owners: Vec<String> = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { name, .. },
                    ..
                } => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert!(
            ctor_owners.contains(&"Layout".to_string()),
            "Layout ctor present"
        );
        assert!(ctor_owners.contains(&"Some".to_string()));
        assert!(ctor_owners.contains(&"None".to_string()));
        assert!(!ctor_owners.contains(&"Option".to_string()));
        assert!(!g.block(g.startblock).operations.iter().any(|op| matches!(&op.kind,
            OpKind::Call { target: CallTarget::SyntheticTransparentCtor { name, .. }, .. } if name == "Layout")),
            "Layout allocation belongs after the successful bound check");
        let none_arm = g.blocks.iter().find(|block| block.operations.iter().any(|op| matches!(&op.kind,
            OpKind::Call { target: CallTarget::SyntheticTransparentCtor { name, .. }, .. } if name == "None")))
            .expect("None arm");
        assert!(!none_arm.operations.iter().any(|op| matches!(&op.kind,
            OpKind::Call { target: CallTarget::SyntheticTransparentCtor { name, .. }, .. } if name == "Layout")));

        // The Layout aggregate writes `__pos_0` and `__pos_1`.
        let layout_fields: Vec<String> = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldWrite { field, .. }
                    if field.owner_root.as_deref() == Some("core::alloc::layout::Layout") =>
                {
                    Some(field.name.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(layout_fields, vec!["__pos_0", "__pos_1"]);
    }

    /// Real MIR emits `align_of` (folded to `ConstInt`) in its own block and
    /// SSA-copies it into the `from_size_align` block.  The rewrite must
    /// follow that unique incoming link.
    #[test]
    fn from_size_align_ok_lowers_when_align_const_is_ssa_copied() {
        let mut g = FunctionGraph::new("test_from_size_align_ssa_align");
        let consts = g.startblock;
        let size = g.push_op_var(consts, OpKind::ConstInt(64), true).unwrap();
        let align = g.push_op_var(consts, OpKind::ConstInt(8), true).unwrap();
        let (p, p_args) = g.create_block_with_arg_vars(2);
        let fsa = g
            .push_op_var(
                p,
                OpKind::Call {
                    target: fsa_target(),
                    args: crate::model::call_args(vec![p_args[0].clone(), p_args[1].clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (q, q_args) = g.create_block_with_arg_vars(1);
        let ok = g
            .push_op_var(
                q,
                OpKind::Call {
                    target: ok_target(),
                    args: crate::model::call_args(vec![q_args[0].clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(q, cont, vec![ok.clone()]);
        g.set_goto(p, q, vec![fsa]);
        g.set_goto(consts, p, vec![size, align]);

        let rewritten = rewire_from_size_align_sites(&mut g, &[site_for(&ok)]);
        assert_eq!(rewritten, 1, "SSA-copied align ConstInt must rewrite");
        assert!(
            !residual_from_size_align_survives(&g),
            "from_size_align residual must be gone"
        );
    }

    #[test]
    fn declines_when_align_is_not_a_folded_const() {
        let mut g = FunctionGraph::new("test_dynamic_align");
        let p = g.startblock;
        let size = g.push_op_var(p, OpKind::ConstInt(64), true).unwrap();
        // A non-const align (a plain input-threaded value) has no ConstInt
        // producer, so the bound cannot be materialized — decline.
        let align = g
            .push_op_var(
                p,
                OpKind::Input {
                    name: "a".to_string(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let fsa = g
            .push_op_var(
                p,
                OpKind::Call {
                    target: fsa_target(),
                    args: crate::model::call_args(vec![size, align]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (q, _q_args) = g.create_block_with_arg_vars(1);
        let ok = g
            .push_op_var(
                q,
                OpKind::Call {
                    target: ok_target(),
                    args: crate::model::call_args(vec![fsa.clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(q, cont, vec![ok.clone()]);
        g.set_goto(p, q, vec![fsa]);

        let rewritten = rewire_from_size_align_sites(&mut g, &[site_for(&ok)]);
        assert_eq!(rewritten, 0, "a non-const align must decline");
    }

    fn expect_target() -> CallTarget {
        CallTarget::Method {
            name: "expect".to_string(),
            receiver_root: Some("Result".to_string()),
            resolved_path: None,
            fun_decl_id: None,
        }
    }

    /// Build P (from_size_align) → Q (.expect(msg)) → returnblock with `align`
    /// folded to a `ConstInt` in P.  Returns the graph and the `.expect()`
    /// result var (the by-value `Layout`).
    fn build_expect_site() -> (FunctionGraph, Variable) {
        let mut g = FunctionGraph::new("test_from_size_align_expect");
        let p = g.startblock;
        let size = g.push_op_var(p, OpKind::ConstInt(64), true).unwrap();
        let align = g.push_op_var(p, OpKind::ConstInt(8), true).unwrap();
        let fsa = g
            .push_op_var(
                p,
                OpKind::Call {
                    target: fsa_target(),
                    args: crate::model::call_args(vec![size, align]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (q, _q_args) = g.create_block_with_arg_vars(1);
        let msg = g.push_op_var(q, OpKind::ConstInt(1), true).unwrap();
        let layout = g
            .push_op_var(
                q,
                OpKind::Call {
                    target: expect_target(),
                    args: crate::model::call_args(vec![fsa.clone(), msg]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(q, cont, vec![layout.clone()]);
        g.set_goto(p, q, vec![fsa]);
        (g, layout)
    }

    fn expect_site_for(result_var: &Variable) -> FromSizeAlignExpectSite {
        FromSizeAlignExpectSite {
            result_var: result_var.clone(),
            layout_owner: "core::alloc::layout::Layout".to_string(),
        }
    }

    #[test]
    fn from_size_align_expect_lowers_to_bound_check_and_by_value_layout() {
        let (mut g, layout) = build_expect_site();
        let rewritten = rewire_from_size_align_expect_sites(&mut g, &[expect_site_for(&layout)]);
        assert_eq!(
            rewritten, 1,
            "the from_size_align().expect() site must rewrite"
        );

        // No residual FunctionPath / Method call survives.
        let has_residual = g.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { .. } | CallTarget::Method { .. },
                    ..
                }
            )
        });
        assert!(
            !has_residual,
            "both from_size_align and expect residuals must be gone"
        );

        // The bound-check binops (`uint_lt` + `eq`) landed.
        let binops: Vec<String> = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::BinOp { op, .. } => Some(op.clone()),
                _ => None,
            })
            .collect();
        assert!(binops.contains(&"uint_lt".to_string()));
        assert!(binops.contains(&"eq".to_string()));

        // A by-value `Layout` transparent ctor exists — and NO `Option` ctor
        // (unlike the `.ok()` shape).
        let ctor_owners: Vec<String> = g
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { name, .. },
                    ..
                } => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert!(
            ctor_owners.contains(&"Layout".to_string()),
            "Layout ctor present"
        );
        assert!(
            !ctor_owners.contains(&"Option".to_string()),
            "no Option wrapper"
        );

        // Exactly one arm raises to the exceptblock (the overflow `Err` arm).
        let raises = g
            .blocks
            .iter()
            .filter(|blk| blk.exits.iter().any(|link| link.target == g.exceptblock))
            .count();
        assert_eq!(raises, 1, "the overflow arm raises to exceptblock");
    }

    /// Real MIR threads the `from_size_align` result into `.expect()` as a
    /// block inputarg (an SSA copy), not as the predecessor's result var.
    #[test]
    fn from_size_align_expect_lowers_when_result_is_ssa_threaded() {
        let mut g = FunctionGraph::new("test_from_size_align_expect_ssa");
        let p = g.startblock;
        let size = g.push_op_var(p, OpKind::ConstInt(64), true).unwrap();
        let align = g.push_op_var(p, OpKind::ConstInt(8), true).unwrap();
        let fsa = g
            .push_op_var(
                p,
                OpKind::Call {
                    target: fsa_target(),
                    args: crate::model::call_args(vec![size, align]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (q, q_args) = g.create_block_with_arg_vars(1);
        let msg = g.push_op_var(q, OpKind::ConstInt(1), true).unwrap();
        let layout = g
            .push_op_var(
                q,
                OpKind::Call {
                    target: expect_target(),
                    args: crate::model::call_args(vec![q_args[0].clone(), msg]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(q, cont, vec![layout.clone()]);
        g.set_goto(p, q, vec![fsa]);

        let rewritten = rewire_from_size_align_expect_sites(&mut g, &[expect_site_for(&layout)]);
        assert_eq!(rewritten, 1, "SSA-threaded expect receiver must rewrite");
        assert!(
            !residual_from_size_align_survives(&g),
            "from_size_align residual must be gone"
        );
        assert!(
            matches!(&g.block(q).exits[0].args[0], LinkArg::Value(value) if value == &q_args[0]),
            ".expect() must forward the Q input, not the deleted P result"
        );
    }

    #[test]
    fn expect_declines_when_align_is_not_a_folded_const() {
        let mut g = FunctionGraph::new("test_expect_dynamic_align");
        let p = g.startblock;
        let size = g.push_op_var(p, OpKind::ConstInt(64), true).unwrap();
        let align = g
            .push_op_var(
                p,
                OpKind::Input {
                    name: "a".to_string(),
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let fsa = g
            .push_op_var(
                p,
                OpKind::Call {
                    target: fsa_target(),
                    args: crate::model::call_args(vec![size, align]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (q, _q_args) = g.create_block_with_arg_vars(1);
        let msg = g.push_op_var(q, OpKind::ConstInt(1), true).unwrap();
        let layout = g
            .push_op_var(
                q,
                OpKind::Call {
                    target: expect_target(),
                    args: crate::model::call_args(vec![fsa.clone(), msg]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(q, cont, vec![layout.clone()]);
        g.set_goto(p, q, vec![fsa]);

        let rewritten = rewire_from_size_align_expect_sites(&mut g, &[expect_site_for(&layout)]);
        assert_eq!(rewritten, 0, "a non-const align must decline");
    }

    fn residual_from_size_align_survives(g: &FunctionGraph) -> bool {
        g.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target, .. } if is_layout_from_size_align_target(target)
            )
        })
    }

    fn fsa_impl_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["alloc", "layout", "<Impl>", "from_size_align"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn fsa_method_target() -> CallTarget {
        CallTarget::Method {
            name: "from_size_align".to_string(),
            receiver_root: Some("Layout".to_string()),
            resolved_path: None,
        }
    }

    fn ok_impl_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "result", "<Impl>", "ok"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn build_ok_site_with(fsa: CallTarget, ok: CallTarget) -> (FunctionGraph, Variable) {
        let mut g = FunctionGraph::new("test_from_size_align_spelling");
        let p = g.startblock;
        let size = g.push_op_var(p, OpKind::ConstInt(64), true).unwrap();
        let align = g.push_op_var(p, OpKind::ConstInt(8), true).unwrap();
        let fsa_res = g
            .push_op_var(
                p,
                OpKind::Call {
                    target: fsa,
                    args: crate::model::call_args(vec![size, align]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (q, q_args) = g.create_block_with_arg_vars(1);
        let ok_res = g
            .push_op_var(
                q,
                OpKind::Call {
                    target: ok,
                    args: crate::model::call_args(vec![q_args[0].clone()]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(q, cont, vec![ok_res.clone()]);
        g.set_goto(p, q, vec![fsa_res]);
        (g, ok_res)
    }

    #[test]
    fn layout_from_size_align_accepts_impl_declaration_spelling() {
        let layout: Vec<String> = ["alloc", "layout", "Layout", "from_size_align"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let impl_path: Vec<String> = ["alloc", "layout", "<Impl>", "from_size_align"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let core_impl: Vec<String> = ["core", "alloc", "layout", "<Impl>", "from_size_align"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let unrelated: Vec<String> = ["core", "num", "<Impl>", "from_size_align"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(is_layout_from_size_align(&layout));
        assert!(is_layout_from_size_align(&impl_path));
        assert!(is_layout_from_size_align(&core_impl));
        assert!(!is_layout_from_size_align(&unrelated));
        assert!(is_layout_adt_owner("core::alloc::layout::Layout"));
        assert!(is_layout_adt_owner("alloc::layout::Layout"));
        assert!(!is_layout_adt_owner("core::result::Result"));
    }

    #[test]
    fn from_size_align_ok_impl_spelling_lowers() {
        let (mut g, ok) = build_ok_site_with(fsa_impl_target(), ok_target());
        let rewritten = rewire_from_size_align_sites(&mut g, &[site_for(&ok)]);
        assert_eq!(rewritten, 1, "the <Impl> FunDecl spelling must rewrite");
        assert!(
            !residual_from_size_align_survives(&g),
            "the <Impl> residual must be gone"
        );
    }

    #[test]
    fn from_size_align_ok_method_target_lowers() {
        let (mut g, ok) = build_ok_site_with(fsa_method_target(), ok_target());
        let rewritten = rewire_from_size_align_sites(&mut g, &[site_for(&ok)]);
        assert_eq!(
            rewritten, 1,
            "the Method Layout::from_size_align form must rewrite"
        );
        assert!(
            !residual_from_size_align_survives(&g),
            "the Method residual must be gone"
        );
    }

    #[test]
    fn from_size_align_ok_functionpath_result_ok_lowers() {
        let (mut g, ok) = build_ok_site_with(fsa_target(), ok_impl_target());
        let rewritten = rewire_from_size_align_sites(&mut g, &[site_for(&ok)]);
        assert_eq!(
            rewritten, 1,
            "Result::ok as core::result::<Impl>::ok must rewrite"
        );
        assert!(
            !residual_from_size_align_survives(&g),
            "from_size_align residual must be gone when .ok() is a FunctionPath"
        );
    }

    /// The four census callers must consume their `from_size_align` residual.
    /// Ignored: loads the real extracted LLBCs.
    #[test]
    #[ignore]
    fn four_from_size_align_callers_have_no_residual() {
        use crate::front::mir::lower_function;
        use majit_charon_reader::Llbc;

        let object = Llbc::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../build/llbc/pyre-object.ullbc"
        ))
        .expect("load pyre-object.ullbc");
        let rlib = Llbc::load(crate::runtime_names::artifacts::MAJIT_RLIB_ULLBC)
            .expect("load majit-rlib.ullbc");

        for (llbc, name) in [
            (&object, "bh_alloc_lowlevel_string"),
            (&object, "alloc_raw_utf8_payload"),
            (&object, "try_items_block_layout"),
            (&rlib, "try_typed_items_block_layout"),
        ] {
            let graph =
                lower_function(llbc, name).unwrap_or_else(|err| panic!("lower {name}: {err}"));
            assert!(
                !residual_from_size_align_survives(&graph),
                "{name} still has a residual Layout::from_size_align"
            );
        }
    }
}
