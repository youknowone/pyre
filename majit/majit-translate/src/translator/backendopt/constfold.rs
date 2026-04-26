//! Port subset of `rpython/translator/backendopt/constfold.py`.

use std::collections::HashMap;

use crate::flowspace::model::{
    BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, LinkRef, SpaceOperation,
    Variable,
};
use crate::translator::simplify;

/// RPython `fold_op_list(block, constants, exit_early=False,
/// exc_catch=False)` at `constfold.py:10-100`.
fn fold_op_list(
    block: &BlockRef,
    constants: &mut HashMap<Variable, Constant>,
    exit_early: bool,
) -> FoldOpListResult {
    let operations = block.borrow().operations.clone();
    let mut newops = Vec::new();
    let mut folded_count = 0;

    for (index, spaceop) in operations.iter().enumerate() {
        let mut vargsmodif = false;
        let mut vargs = Vec::with_capacity(spaceop.args.len());
        let mut args = Vec::with_capacity(spaceop.args.len());
        for v in &spaceop.args {
            match v {
                Hlvalue::Constant(c) => {
                    args.push(c.value.clone());
                    vargs.push(v.clone());
                }
                Hlvalue::Variable(var) => {
                    if let Some(c) = constants.get(var) {
                        vargsmodif = true;
                        args.push(c.value.clone());
                        vargs.push(Hlvalue::Constant(c.clone()));
                    } else {
                        vargs.push(v.clone());
                    }
                }
            }
        }

        if args.len() == vargs.len() {
            if let Some(result) = eval_llop(&spaceop.opname, &args) {
                if let Hlvalue::Variable(result_var) = &spaceop.result {
                    let constant = constant_for_result(result, &spaceop.result);
                    constants.insert(result_var.clone(), constant);
                    folded_count += 1;
                    continue;
                }
            }
        } else if matches!(spaceop.opname.as_str(), "ptr_eq" | "int_eq")
            && spaceop.args.len() == 2
            && spaceop.args[0] == spaceop.args[1]
        {
            if let Hlvalue::Variable(result_var) = &spaceop.result {
                constants.insert(
                    result_var.clone(),
                    constant_for_result(ConstValue::Bool(true), &spaceop.result),
                );
                folded_count += 1;
                continue;
            }
        } else if matches!(spaceop.opname.as_str(), "ptr_ne" | "int_ne")
            && spaceop.args.len() == 2
            && spaceop.args[0] == spaceop.args[1]
        {
            if let Hlvalue::Variable(result_var) = &spaceop.result {
                constants.insert(
                    result_var.clone(),
                    constant_for_result(ConstValue::Bool(false), &spaceop.result),
                );
                folded_count += 1;
                continue;
            }
        }

        // Upstream `constfold.py:60-82`: deal with `int_add_ovf` etc.
        // when the op is the last in a `c_last_exception` block and
        // both args fold to constants. The block is rewired to skip
        // either the overflow link (success) or the normal link
        // (always-overflows).
        if index == operations.len() - 1
            && block.borrow().canraise()
            && args.len() == vargs.len()
            && args.len() == 2
            && spaceop.opname.ends_with("_ovf")
            && !exit_early
        {
            match fold_ovf_op(&spaceop.opname, &args) {
                None => {
                    // Upstream `:68-76`: always overflows — drop the
                    // op, drop the normal link, keep the OverflowError
                    // link with its exitcase / last_exception cleared.
                    let mut b = block.borrow_mut();
                    b.exitswitch = None;
                    let exc_link = b.exits[1].clone();
                    drop(b);
                    {
                        let mut l = exc_link.borrow_mut();
                        debug_assert_eq!(
                            match l.exitcase.as_ref() {
                                Some(Hlvalue::Constant(c)) =>
                                    c.value.host_class_name().map(str::to_owned),
                                _ => None,
                            },
                            Some("OverflowError".to_string()),
                            "constfold.py:70 _ovf overflow link expected OverflowError exitcase"
                        );
                        l.exitcase = None;
                        l.llexitcase = None;
                        l.last_exception = None;
                        l.last_exc_value = None;
                    }
                    block.recloseblock(vec![exc_link]);
                    folded_count += 1;
                    continue;
                }
                Some(result) => {
                    // Upstream `:77-82`: doesn't overflow — keep the
                    // normal link (`exits[0]`) and bind the constant
                    // result.
                    let normal_link = {
                        let mut b = block.borrow_mut();
                        b.exitswitch = None;
                        b.exits[0].clone()
                    };
                    block.recloseblock(vec![normal_link]);
                    if let Hlvalue::Variable(result_var) = &spaceop.result {
                        constants.insert(
                            result_var.clone(),
                            constant_for_result(result, &spaceop.result),
                        );
                    }
                    folded_count += 1;
                    continue;
                }
            }
        }

        if exit_early {
            let _ = folded_count;
            return FoldOpListResult::Count;
        }
        if vargsmodif {
            if spaceop.opname == "indirect_call"
                && matches!(vargs.first(), Some(Hlvalue::Constant(_)))
            {
                newops.push(SpaceOperation::new(
                    "direct_call",
                    vargs[..vargs.len().saturating_sub(1)].to_vec(),
                    spaceop.result.clone(),
                ));
            } else {
                newops.push(SpaceOperation::new(
                    spaceop.opname.clone(),
                    vargs,
                    spaceop.result.clone(),
                ));
            }
        } else {
            newops.push(spaceop.clone());
        }

        let _ = index;
    }

    if exit_early {
        let _ = folded_count;
        FoldOpListResult::Count
    } else {
        FoldOpListResult::Ops(newops)
    }
}

enum FoldOpListResult {
    Count,
    Ops(Vec<SpaceOperation>),
}

/// RPython `constant_fold_block(block)` at `constfold.py:116-135`.
///
/// Upstream `:122 remaining_exits = [link for link in block.exits
/// if link.llexitcase == switch]` filters the surviving exit by the
/// **lltype-level** exit value (`llexitcase`), not the flow-level
/// exitcase. Pre-rtype the two are equal; post-rtype they diverge —
/// rtyped switches carry the low-level integer in `llexitcase` while
/// `exitcase` retains the flow-level value, so a flow-keyed match
/// would pick the wrong exit on rtyped switches.
pub fn constant_fold_block(block: &BlockRef) {
    let mut constants = HashMap::new();
    let FoldOpListResult::Ops(newops) = fold_op_list(block, &mut constants, false) else {
        unreachable!("exit_early=false returns Ops")
    };
    block.borrow_mut().operations = newops;

    if constants.is_empty() {
        return;
    }

    let const_switch = {
        let b = block.borrow();
        match &b.exitswitch {
            Some(Hlvalue::Variable(v)) => constants.get(v).cloned(),
            _ => None,
        }
    };
    if let Some(c) = const_switch {
        // Upstream `:121-132`: filter by `link.llexitcase == switch`.
        let exits_snapshot: Vec<LinkRef> = block.borrow().exits.iter().cloned().collect();
        let mut remaining_exits: Vec<LinkRef> = exits_snapshot
            .iter()
            .filter(|link| match &link.borrow().llexitcase {
                Some(Hlvalue::Constant(lc)) => lc.value == c.value,
                _ => false,
            })
            .cloned()
            .collect();
        if remaining_exits.is_empty() {
            // Upstream `:125-127`: fall through to the explicit
            // `'default'` exit (last in `block.exits`). Upstream's
            // `block.exits[-1].exitcase == 'default'` compares against
            // a Python 2 byte literal — `ConstValue::string_eq` accepts
            // either ByteStr or UniStr in the Rust port (Item 2 string
            // split keeps the bytes shape but tolerates either side
            // until callers rewrap).
            if let Some(last) = exits_snapshot.last() {
                let is_default = matches!(
                    &last.borrow().exitcase,
                    Some(Hlvalue::Constant(c)) if c.value.string_eq("default")
                );
                assert!(
                    is_default,
                    "constfold.py:126 expected last exit to be 'default' fallback"
                );
                remaining_exits.push(last.clone());
            }
        }
        assert_eq!(
            remaining_exits.len(),
            1,
            "constfold.py:128 exactly one exit must survive after llexitcase match"
        );
        {
            let mut l = remaining_exits[0].borrow_mut();
            l.exitcase = None;
            l.llexitcase = None;
        }
        block.borrow_mut().exitswitch = None;
        block.recloseblock(remaining_exits);
    }

    for link in &block.borrow().exits {
        let mut link = link.borrow_mut();
        for arg in &mut link.args {
            if let Some(Hlvalue::Variable(v)) = arg {
                if let Some(c) = constants.get(v) {
                    *arg = Some(Hlvalue::Constant(c.clone()));
                }
            }
        }
    }
}

/// RPython `same_constant(c1, c2)` at `constfold.py:306-314`.
///
/// Upstream `:308 assert c1.concretetype == c2.concretetype` requires
/// both constants to be at the same lltype before any value comparison
/// — different lltypes with the same Python value must NOT diffuse,
/// because they flow into different storage. Upstream then special-
/// cases `lltype.Ptr` GC pointers (compares by `value` only) and
/// otherwise compares Constant identity (`c1 == c2` over Constant).
///
/// The local port enforces the concretetype-equality precondition.
/// When either side has no recorded concretetype the call returns
/// false, matching upstream's "must agree" rule rather than upstream's
/// assert (the assert is only safe when annotations have run; the
/// pyre-side caller may run pre-anno).
fn same_constant(c1: &Hlvalue, c2: &Hlvalue) -> bool {
    let (Hlvalue::Constant(a), Hlvalue::Constant(b)) = (c1, c2) else {
        return false;
    };
    match (a.concretetype.as_ref(), b.concretetype.as_ref()) {
        (Some(ta), Some(tb)) if ta == tb => a.value == b.value,
        _ => false,
    }
}

/// RPython `constant_diffuse(graph)` at `constfold.py:260-304`.
fn constant_diffuse(graph: &FunctionGraph) -> usize {
    use crate::flowspace::model::BlockKey;

    let mut count = 0;

    // Upstream `:262-272`: after `exitswitch vexit`, replace `vexit`
    // with the corresponding constant if it also appears on the
    // outgoing links.
    for block in graph.iterblocks() {
        let (vexit, exits) = {
            let b = block.borrow();
            let vexit = match &b.exitswitch {
                Some(Hlvalue::Variable(v)) => Some(v.clone()),
                _ => None,
            };
            (vexit, b.exits.clone())
        };
        let Some(vexit) = vexit else {
            continue;
        };
        for link in exits {
            let (carries_vexit, is_default, llexitcase, exitcase_concrete) = {
                let l = link.borrow();
                let is_default = matches!(
                    &l.exitcase,
                    Some(Hlvalue::Constant(c)) if c.value.string_eq("default")
                );
                let carries_vexit = l
                    .args
                    .iter()
                    .any(|arg| matches!(arg, Some(Hlvalue::Variable(v)) if v == &vexit));
                let llexitcase = l.llexitcase.clone();
                let exitcase_concrete = match &l.llexitcase {
                    Some(Hlvalue::Constant(c)) => c.concretetype.clone(),
                    _ => None,
                };
                (carries_vexit, is_default, llexitcase, exitcase_concrete)
            };
            if !carries_vexit || is_default {
                continue;
            }
            let Some(Hlvalue::Constant(case_const)) = llexitcase else {
                continue;
            };
            // Upstream `:269-270`: `remap = {vexit: Constant(
            // link.llexitcase, vexit.concretetype)}`. The Rust port
            // re-wraps the constant so the resulting `Constant` carries
            // `vexit.concretetype` rather than the raw exitcase
            // concretetype, mirroring the upstream substitution.
            let concretetype = vexit.concretetype().or(exitcase_concrete);
            let remapped = match concretetype {
                Some(t) => Constant::with_concretetype(case_const.value.clone(), t),
                None => Constant::new(case_const.value.clone()),
            };
            let mut l = link.borrow_mut();
            for arg in l.args.iter_mut() {
                if matches!(arg, Some(Hlvalue::Variable(v)) if v == &vexit) {
                    *arg = Some(Hlvalue::Constant(remapped.clone()));
                    count += 1;
                }
            }
        }
    }

    // Upstream `:277-303`: hoist constants common to all incoming
    // links into block-prefix `same_as` ops.
    let entrymap = crate::flowspace::model::mkentrymap(graph);
    let startkey = BlockKey::of(&graph.startblock);
    for block in graph.iterblocks() {
        let key = BlockKey::of(&block);
        if key == startkey {
            continue;
        }
        if block.borrow().exits.is_empty() {
            continue;
        }
        let Some(links) = entrymap.get(&key) else {
            continue;
        };
        if links.is_empty() {
            continue;
        }
        let firstlink = links[0].clone();
        let firstargs = firstlink.borrow().args.clone();
        let rest = &links[1..];

        let mut diffuse: Vec<(usize, Constant)> = Vec::new();
        for (i, c) in firstargs.iter().enumerate() {
            let Some(Hlvalue::Constant(c)) = c else {
                continue;
            };
            let mut all_same = true;
            for lnk in rest {
                let l = lnk.borrow();
                let other = l.args.get(i).cloned().flatten();
                let matched = match other {
                    Some(other) => same_constant(&Hlvalue::Constant(c.clone()), &other),
                    None => false,
                };
                if !matched {
                    all_same = false;
                    break;
                }
            }
            if all_same {
                diffuse.push((i, c.clone()));
            }
        }
        // Upstream `:293`: `diffuse.reverse()` — process highest index
        // first so earlier `del` operations don't shift later ones.
        diffuse.reverse();
        if diffuse.is_empty() {
            continue;
        }
        let mut same_as_ops = Vec::with_capacity(diffuse.len());
        for (i, c) in &diffuse {
            for lnk in links {
                lnk.borrow_mut().args.remove(*i);
            }
            let v = block.borrow_mut().inputargs.remove(*i);
            same_as_ops.push(SpaceOperation::new(
                "same_as",
                vec![Hlvalue::Constant(c.clone())],
                v,
            ));
            count += 1;
        }
        // Upstream `:301`: `block.operations = same_as + block.operations`.
        // Insert in original (pre-reverse) order so the prefix shape
        // matches `[same_as_for_low_index, ..., same_as_for_high_index]`.
        same_as_ops.reverse();
        let mut b = block.borrow_mut();
        let mut new_ops = same_as_ops;
        new_ops.extend(b.operations.drain(..));
        b.operations = new_ops;
        drop(b);
        // Upstream `:302-303`: `if same_as: constant_fold_block(block)`.
        constant_fold_block(&block);
    }

    count
}

/// RPython `constant_fold_graph(graph)` at `constfold.py:316-370`.
///
/// PRE-EXISTING-ADAPTATION: upstream's loop body does TWO things per
/// iteration — block-level diffusion (`constant_diffuse`) AND
/// link-level constant propagation that splits the target block at the
/// folded prefix and rewires every incoming constant edge through a
/// new sub-block. The link-level pass handles `_ovf` overflow blocks
/// specially (`:333-364`): when both args fold to constants, the link
/// is rewired directly to the success or overflow exit and the
/// original block becomes dead.
///
/// The local port runs only the block-level pass; the link-level pass
/// requires `prepare_constant_fold_link`, `complete_constants`,
/// `rewire_link_for_known_exitswitch`, `rewire_links`, `split_block`,
/// and `insert_empty_block` — none of which are ported yet. Convergence
/// path = port those six helpers from `constfold.py:156-256` +
/// `simplify.split_block` + `unsimplify.insert_empty_block`, then
/// expand this loop to match upstream `:325-370`.
pub fn constant_fold_graph(graph: &FunctionGraph) {
    for block in graph.iterblocks() {
        if !block.borrow().operations.is_empty() {
            constant_fold_block(&block);
        }
    }
    loop {
        let diffused = constant_diffuse(graph);
        // Upstream `:327-364`: link-level constant propagation —
        // skipped here pending the helper port noted above. Without
        // this pass, constants flowing across links into folded
        // operations stay un-folded, producing fewer optimisations
        // than upstream but no incorrect rewrites.
        if diffused == 0 {
            break;
        }
        simplify::eliminate_empty_blocks(graph);
        simplify::join_blocks(graph);
    }
}

/// RPython `replace_symbolic(graph, symbolic, value)` at
/// `constfold.py:372-383`.
///
/// Upstream's `arg.value is symbolic` (Python `is`) compares object
/// identity — only literally the same object passed in by the caller
/// matches. Rust's `PartialEq` cannot express identity for cloneable
/// `ConstValue`s, so the local port relies on the caller passing a
/// `ConstValue` shape that user code provably cannot construct (e.g.
/// `ConstValue::SpecTag(unique_u64)`). Passing a `ConstValue::Str`
/// or any other user-reachable variant would be **NEW-DEVIATION** —
/// any same-valued constant in user code would be silently rewritten.
pub fn replace_symbolic(graph: &FunctionGraph, symbolic: &ConstValue, value: Constant) -> bool {
    debug_assert!(
        matches!(symbolic, ConstValue::SpecTag(_) | ConstValue::Atom(_)),
        "replace_symbolic: caller must supply an identity-bearing ConstValue \
         (SpecTag/Atom) so user-reachable values cannot collide; got {symbolic:?}"
    );
    let mut result = false;
    for block in graph.iterblocks() {
        {
            let mut b = block.borrow_mut();
            for op in &mut b.operations {
                for arg in &mut op.args {
                    if matches!(arg, Hlvalue::Constant(c) if &c.value == symbolic) {
                        *arg = Hlvalue::Constant(value.clone());
                        result = true;
                    }
                }
            }
            if matches!(&b.exitswitch, Some(Hlvalue::Constant(c)) if &c.value == symbolic) {
                b.exitswitch = Some(Hlvalue::Constant(value.clone()));
                result = true;
            }
        }
    }
    result
}

/// Stable, process-unique `SpecTag` id reserved for the
/// `rpython.rlib.jit._we_are_jitted` symbolic singleton. Picked at the
/// far end of the `u64` space so no `SpecTag` allocated by
/// `unroll.py` (which counts up from 0) can collide.
///
/// Once `rlib/jit.py` lands locally, `_we_are_jitted` becomes
/// `Constant(ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID), lltype.Signed)`
/// at every emit site. Until then no user code produces this tag, so
/// the call below sweeps zero matches and `replace_we_are_jitted`
/// returns `false` — exactly upstream's "no `we_are_jitted()` call
/// sites" behaviour, but driven by identity rather than string match.
pub const WE_ARE_JITTED_TAG_ID: u64 = u64::MAX - 0x57E_A1E_71D;

/// RPython `replace_we_are_jitted(graph)` at `constfold.py:385-392`.
///
/// Upstream:
/// ```python
/// def replace_we_are_jitted(graph):
///     from rpython.rlib import jit
///     replacement = Constant(0, lltype.Signed)
///     did_replacement = replace_symbolic(graph, jit._we_are_jitted, replacement)
///     if did_replacement:
///         constant_fold_graph(graph)
/// ```
///
/// `replace_symbolic` keys on Python `is` identity. The Rust port
/// uses [`WE_ARE_JITTED_TAG_ID`] as the identity-bearing
/// `ConstValue::SpecTag` shape; passing any user-reachable
/// `ConstValue` would be a NEW-DEVIATION (silent rewriting of
/// matching values). Once `rlib/jit.py` is ported and emits
/// `Constant(ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID), Signed)` at
/// every `we_are_jitted()` call site, this function performs the
/// upstream replacement.
pub fn replace_we_are_jitted(graph: &FunctionGraph) -> bool {
    let symbolic = ConstValue::SpecTag(WE_ARE_JITTED_TAG_ID);
    let replacement = Constant::with_concretetype(
        ConstValue::Int(0),
        crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Signed,
    );
    let did_replacement = replace_symbolic(graph, &symbolic, replacement);
    if did_replacement {
        constant_fold_graph(graph);
    }
    did_replacement
}

fn constant_for_result(value: ConstValue, result: &Hlvalue) -> Constant {
    match result {
        Hlvalue::Variable(v) => match v.concretetype() {
            Some(t) => Constant::with_concretetype(value, t),
            None => Constant::new(value),
        },
        Hlvalue::Constant(c) => match &c.concretetype {
            Some(t) => Constant::with_concretetype(value, t.clone()),
            None => Constant::new(value),
        },
    }
}

fn eval_llop(opname: &str, args: &[ConstValue]) -> Option<ConstValue> {
    Some(match (opname, args) {
        ("same_as", [value]) => value.clone(),
        ("bool_not", [ConstValue::Bool(value)]) => ConstValue::Bool(!value),
        ("int_is_true", [ConstValue::Int(value)]) => ConstValue::Bool(*value != 0),
        ("int_neg", [ConstValue::Int(value)]) => ConstValue::Int(value.checked_neg()?),
        ("int_abs", [ConstValue::Int(value)]) => ConstValue::Int(value.checked_abs()?),
        ("int_invert", [ConstValue::Int(value)]) => ConstValue::Int(!value),
        ("int_add", [ConstValue::Int(a), ConstValue::Int(b)]) => {
            ConstValue::Int(a.checked_add(*b)?)
        }
        ("int_sub", [ConstValue::Int(a), ConstValue::Int(b)]) => {
            ConstValue::Int(a.checked_sub(*b)?)
        }
        ("int_mul", [ConstValue::Int(a), ConstValue::Int(b)]) => {
            ConstValue::Int(a.checked_mul(*b)?)
        }
        ("int_floordiv", [ConstValue::Int(_), ConstValue::Int(0)]) => return None,
        ("int_floordiv", [ConstValue::Int(a), ConstValue::Int(b)]) => {
            ConstValue::Int(a.checked_div(*b)?)
        }
        ("int_mod", [ConstValue::Int(_), ConstValue::Int(0)]) => return None,
        ("int_mod", [ConstValue::Int(a), ConstValue::Int(b)]) => {
            ConstValue::Int(a.checked_rem(*b)?)
        }
        ("int_lt", [ConstValue::Int(a), ConstValue::Int(b)]) => ConstValue::Bool(a < b),
        ("int_le", [ConstValue::Int(a), ConstValue::Int(b)]) => ConstValue::Bool(a <= b),
        ("int_eq", [ConstValue::Int(a), ConstValue::Int(b)]) => ConstValue::Bool(a == b),
        ("int_ne", [ConstValue::Int(a), ConstValue::Int(b)]) => ConstValue::Bool(a != b),
        ("int_gt", [ConstValue::Int(a), ConstValue::Int(b)]) => ConstValue::Bool(a > b),
        ("int_ge", [ConstValue::Int(a), ConstValue::Int(b)]) => ConstValue::Bool(a >= b),
        ("int_and", [ConstValue::Int(a), ConstValue::Int(b)]) => ConstValue::Int(a & b),
        ("int_or", [ConstValue::Int(a), ConstValue::Int(b)]) => ConstValue::Int(a | b),
        ("int_xor", [ConstValue::Int(a), ConstValue::Int(b)]) => ConstValue::Int(a ^ b),
        ("ptr_eq", [a, b]) => ConstValue::Bool(a == b),
        ("ptr_ne", [a, b]) => ConstValue::Bool(a != b),
        _ => return None,
    })
}

/// Port of upstream `fold_ovf_op(spaceop, args)` (`constfold.py:102-114`).
///
/// Returns `Some(result)` when the overflow-checked op folds without
/// overflow, and `None` when the operation would always overflow at
/// compile time. The `Result::Err` variant from
/// `i64::checked_{add,sub,mul}` is the Rust analogue of upstream's
/// `rarithmetic.ovfcheck` raising `OverflowError`.
pub(crate) fn fold_ovf_op(opname: &str, args: &[ConstValue]) -> Option<ConstValue> {
    match (opname, args) {
        ("int_add_ovf" | "int_add_nonneg_ovf", [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_add(*b).map(ConstValue::Int)
        }
        ("int_sub_ovf", [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_sub(*b).map(ConstValue::Int)
        }
        ("int_mul_ovf", [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_mul(*b).map(ConstValue::Int)
        }
        _ => None,
    }
}
