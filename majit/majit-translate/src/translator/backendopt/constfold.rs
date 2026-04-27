//! Port subset of `rpython/translator/backendopt/constfold.py`.

use std::collections::HashMap;

use crate::flowspace::model::{
    BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, LinkArg,
    LinkRef, SpaceOperation, Variable,
};
use crate::translator::rtyper::lltypesystem::lloperation::ll_operations;
use crate::translator::simplify;
use crate::translator::unsimplify::{insert_empty_block, split_block};

/// Heterogeneous map used by the link-level constant-fold pass.
/// Upstream `constfold.py:156-256` carries values that may be either
/// fold-result Constants (added by `fold_op_list`), link-arg Hlvalues
/// (added by `complete_constants`), or fresh Variables (added by
/// `prepare_constant_fold_link`'s indirect-call rewrite). Python's
/// untyped `dict` admits this; the Rust port widens the value type to
/// `Hlvalue` so the same map can carry every shape upstream encounters.
pub(crate) type LinkConstants = HashMap<Variable, Hlvalue>;

/// Per-(block, link) split entry collected by `prepare_constant_fold_link`
/// and consumed by `rewire_links`. Mirrors upstream's tuple
/// `(folded_count, link, constants)` at `constfold.py:230-231`.
struct LinkSplit {
    folded_count: usize,
    link: LinkRef,
    constants: LinkConstants,
}

/// Map of pending splits, keyed by the target block. Each entry holds
/// a strong `BlockRef` (since `BlockKey` is a raw pointer) plus the
/// list of split records collected from incoming links.
type SplitBlocks = HashMap<BlockKey, (BlockRef, Vec<LinkSplit>)>;

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

        // Upstream `constfold.py:27-82`: `try: op = getattr(llop,
        // spaceop.opname)`. Missing-from-registry → AttributeError →
        // `pass` → fall through to the exit-early/append branch.
        if let Some(op_desc) = ll_operations().get(spaceop.opname.as_str()) {
            if !op_desc.sideeffects && args.len() == vargs.len() {
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

            // Upstream `constfold.py:60-82`: deal with `int_add_ovf`
            // etc. when the op is the last in a `c_last_exception`
            // block and both args fold to constants. Still inside the
            // `else:` branch of the `try: op = getattr(...)` at
            // upstream, so the registry lookup gates this path too.
            if index == operations.len() - 1
                && block.borrow().canraise()
                && args.len() == vargs.len()
                && args.len() == 2
                && spaceop.opname.ends_with("_ovf")
                && !exit_early
            {
                match fold_ovf_op(&spaceop.opname, &args) {
                    None => {
                        // Upstream `:68-76`: always overflows — drop
                        // the op, drop the normal link, keep the
                        // OverflowError link with its exitcase /
                        // last_exception cleared.
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
                        // Upstream `:77-82`: doesn't overflow — keep
                        // the normal link (`exits[0]`) and bind the
                        // constant result.
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
        }

        if exit_early {
            return FoldOpListResult::Count(folded_count);
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
        FoldOpListResult::Count(folded_count)
    } else {
        FoldOpListResult::Ops(newops)
    }
}

/// RPython `fold_op_list` returns `folded_count` (int) when
/// `exit_early=True` and `newops` (list) otherwise. The Rust enum fuses
/// both shapes; `Count(usize)` carries the upstream `folded_count`
/// value verbatim because upstream `prepare_constant_fold_link`
/// (`constfold.py:205-231`) reads it to decide how many ops to skip
/// when splitting the target block — making the count load-bearing
/// rather than ornamental observability.
enum FoldOpListResult {
    Count(usize),
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

/// RPython `complete_constants(link, constants)` at
/// `constfold.py:156-165`.
///
/// Walks `zip(link.args, link.target.inputargs)` and, for each
/// target inputarg `v2` not yet present in `constants`, records the
/// link-side value (`v1`) under that key. Pre-existing entries are
/// asserted to match (upstream's `assert constants[v2] is v1` —
/// Python `is`). The local port narrows that to value-equality on
/// `Hlvalue`, which is a tightening only for Constant entries (any
/// two equal-valued Constants compare equal here, but upstream would
/// have allowed two non-identical `Constant(0, Signed)` instances —
/// the constfold pipeline never produces such pairs in practice).
pub(crate) fn complete_constants(link: &LinkRef, constants: &mut LinkConstants) {
    let l = link.borrow();
    let target = l
        .target
        .as_ref()
        .expect("complete_constants: link must have a target")
        .clone();
    let target_inputargs = target.borrow().inputargs.clone();
    assert_eq!(
        l.args.len(),
        target_inputargs.len(),
        "complete_constants: link.args / target.inputargs arity mismatch",
    );
    for (v1, v2) in l.args.iter().zip(target_inputargs.iter()) {
        let Hlvalue::Variable(v2_var) = v2 else {
            // Upstream `block.inputargs` is exclusively Variables;
            // skip defensively.
            continue;
        };
        let Some(v1_value) = v1 else {
            // Transient `None` arg — the constfold caller always
            // supplies fully-materialised links, so just skip.
            continue;
        };
        match constants.get(v2_var) {
            Some(existing) => {
                assert_eq!(
                    existing, v1_value,
                    "complete_constants: conflicting entry for target inputarg",
                );
            }
            None => {
                constants.insert(v2_var.clone(), v1_value.clone());
            }
        }
    }
}

/// RPython `rewire_link_for_known_exitswitch(link1, llexitvalue)` at
/// `constfold.py:167-193`.
///
/// When `link1.target` is an op-less block whose only role is to
/// switch on a constant value, rewire `link1` directly to the chosen
/// successor. The chosen exit is the one whose `llexitcase` matches
/// `llexitvalue`, falling back to the trailing `'default'` exit if
/// present. If neither matches, upstream returns silently and so does
/// the local port.
pub(crate) fn rewire_link_for_known_exitswitch(link1: &LinkRef, llexitvalue: &ConstValue) {
    let block = link1
        .borrow()
        .target
        .clone()
        .expect("rewire_link_for_known_exitswitch: link1 has no target");
    let exits = block.borrow().exits.clone();
    if exits.is_empty() {
        return;
    }
    let last_is_default = matches!(
        &exits.last().unwrap().borrow().exitcase,
        Some(Hlvalue::Constant(c)) if c.value.string_eq("default")
    );
    let (defaultexit, nondefaultexits): (Option<LinkRef>, Vec<LinkRef>) = if last_is_default {
        let mut e = exits;
        let last = e.pop();
        (last, e)
    } else {
        (None, exits)
    };

    let nextlink = nondefaultexits
        .iter()
        .find(|link| {
            matches!(
                &link.borrow().llexitcase,
                Some(Hlvalue::Constant(c)) if &c.value == llexitvalue,
            )
        })
        .cloned()
        .or(defaultexit);

    let Some(nextlink) = nextlink else {
        return;
    };

    let inputargs = block.borrow().inputargs.clone();
    let link1_args = link1.borrow().args.clone();
    assert_eq!(
        inputargs.len(),
        link1_args.len(),
        "rewire_link_for_known_exitswitch: arity mismatch link1.args / target.inputargs",
    );

    // `blockmapping = dict(zip(block.inputargs, link1.args))`.
    let mut blockmapping: HashMap<Variable, LinkArg> = HashMap::new();
    for (v, a) in inputargs.iter().zip(link1_args.iter()) {
        if let Hlvalue::Variable(var) = v {
            blockmapping.insert(var.clone(), a.clone());
        }
    }

    let nextlink_args = nextlink.borrow().args.clone();
    let new_target = nextlink.borrow().target.clone();
    let mut new_args: Vec<LinkArg> = Vec::with_capacity(nextlink_args.len());
    for arg in nextlink_args {
        match arg {
            Some(Hlvalue::Variable(v)) => {
                let mapped = blockmapping.get(&v).cloned().expect(
                    "rewire_link_for_known_exitswitch: nextlink arg references \
                         a Variable not in the switch block's inputargs — \
                         upstream `blockmapping[v]` would have raised KeyError",
                );
                new_args.push(mapped);
            }
            other => new_args.push(other),
        }
    }

    let mut l = link1.borrow_mut();
    l.target = new_target;
    l.args = new_args;
}

/// RPython `prepare_constant_fold_link(link, constants, splitblocks)`
/// at `constfold.py:195-231`.
///
/// Tries to fold the prefix of the link's target block under the
/// constants known on this link. If folding succeeds, records a split
/// entry so `rewire_links` can later re-route the link past the folded
/// prefix.
///
/// `constants` is the heterogeneous link-level map: at entry it holds
/// Constants only (built by the caller from the link's args zipped
/// with the target's inputargs). `fold_op_list` may add more Constant
/// entries; the indirect-call rewrite below adds the fresh Variable
/// produced for the rebound `direct_call` result.
fn prepare_constant_fold_link(
    link: &LinkRef,
    constants: &mut LinkConstants,
    splitblocks: &mut SplitBlocks,
) {
    let block = link
        .borrow()
        .target
        .clone()
        .expect("prepare_constant_fold_link: link has no target");

    // Upstream `:198-203`: target has no operations — only chance to
    // fold is via the exitswitch lookup.
    if block.borrow().operations.is_empty() {
        let exitswitch_var = match &block.borrow().exitswitch {
            Some(Hlvalue::Variable(v)) => Some(v.clone()),
            _ => None,
        };
        if let Some(v) = exitswitch_var {
            if let Some(Hlvalue::Constant(c)) = constants.get(&v) {
                let llexitvalue = c.value.clone();
                rewire_link_for_known_exitswitch(link, &llexitvalue);
            }
        }
        return;
    }

    // Build a Constant-only narrow view to feed `fold_op_list`. At
    // this point in the pipeline (per upstream invariant) `constants`
    // contains only Constants; `complete_constants` and the
    // indirect-call rewrite that widen to Variables have not run yet.
    let mut narrow: HashMap<Variable, Constant> = constants
        .iter()
        .filter_map(|(k, v)| match v {
            Hlvalue::Constant(c) => Some((k.clone(), c.clone())),
            _ => None,
        })
        .collect();
    let folded_count = match fold_op_list(&block, &mut narrow, true) {
        FoldOpListResult::Count(n) => n,
        FoldOpListResult::Ops(_) => {
            unreachable!("fold_op_list(exit_early=true) returns Count")
        }
    };
    // Reflect any new fold results back into the wide map. Upstream's
    // single dict gets these by direct mutation; the Rust port's
    // narrow→wide bridge mirrors that effect.
    for (k, c) in narrow {
        constants.insert(k, Hlvalue::Constant(c));
    }

    let n_total = block.borrow().operations.len();
    let n = if block.borrow().canraise() {
        n_total - 1
    } else {
        n_total
    };

    let mut effective_link = link.clone();
    let mut effective_folded = folded_count;

    // Upstream `:211-227`: if the next non-folded op is an
    // `indirect_call` whose function pointer is now constant, rewrite
    // it to a `direct_call` and tuck the new op into a synthetic
    // block via `insert_empty_block`.
    if effective_folded < n {
        let nextop = block.borrow().operations[effective_folded].clone();
        if nextop.opname == "indirect_call" {
            let nextop_arg0_var = match nextop.args.first() {
                Some(Hlvalue::Variable(v)) => Some(v.clone()),
                _ => None,
            };
            if let Some(arg0_var) = nextop_arg0_var {
                if let Some(arg0_value) = constants.get(&arg0_var).cloned() {
                    // Build callargs: resolved function constant, then
                    // `nextop.args[1:-1]` mapped through `constants1`
                    // (a copy widened with `complete_constants`).
                    let mut callargs: Vec<Hlvalue> = vec![arg0_value];

                    let mut constants1 = constants.clone();
                    complete_constants(link, &mut constants1);

                    let len = nextop.args.len();
                    let mid = if len >= 2 {
                        &nextop.args[1..len - 1]
                    } else {
                        &[][..]
                    };
                    for v in mid {
                        match v {
                            Hlvalue::Variable(var) => {
                                callargs.push(
                                    constants1
                                        .get(var)
                                        .cloned()
                                        .unwrap_or_else(|| Hlvalue::Variable(var.clone())),
                                );
                            }
                            other => callargs.push(other.clone()),
                        }
                    }

                    let Hlvalue::Variable(orig_result) = &nextop.result else {
                        panic!(
                            "prepare_constant_fold_link: indirect_call result is not a Variable"
                        );
                    };
                    // Upstream `v_result = Variable(nextop.result)` —
                    // a fresh Variable named after the original, with
                    // the same concretetype. `Variable::copy` matches
                    // both invariants.
                    let v_result = orig_result.copy();
                    constants.insert(orig_result.clone(), Hlvalue::Variable(v_result.clone()));

                    let callop =
                        SpaceOperation::new("direct_call", callargs, Hlvalue::Variable(v_result));
                    let newblock = insert_empty_block(link, vec![callop]);
                    let exits = newblock.borrow().exits.clone();
                    assert_eq!(
                        exits.len(),
                        1,
                        "insert_empty_block should leave exactly one exit"
                    );
                    let new_link = exits.into_iter().next().unwrap();
                    debug_assert!(matches!(
                        new_link.borrow().target.as_ref(),
                        Some(t) if std::rc::Rc::ptr_eq(t, &block)
                    ));
                    effective_link = new_link;
                    effective_folded += 1;
                }
            }
        }
    }

    if effective_folded > 0 {
        let key = BlockKey::of(&block);
        splitblocks
            .entry(key)
            .or_insert_with(|| (block.clone(), Vec::new()))
            .1
            .push(LinkSplit {
                folded_count: effective_folded,
                link: effective_link,
                constants: constants.clone(),
            });
    }
}

/// RPython `rewire_links(splitblocks, graph)` at `constfold.py:233-256`.
///
/// For each block with pending splits, sort the splits by descending
/// folded position so the block can be split from the end forward
/// without invalidating earlier positions. Each split rewires its
/// originating link past the folded prefix, threading the resolved
/// constants through the splitlink's args.
fn rewire_links(splitblocks: SplitBlocks) {
    for (_key, (block, mut splits)) in splitblocks {
        // Upstream `splits.sort(); splits.reverse()` — descending
        // folded_count.
        splits.sort_by(|a, b| b.folded_count.cmp(&a.folded_count));
        for LinkSplit {
            folded_count: position,
            link,
            mut constants,
        } in splits
        {
            debug_assert!(matches!(
                link.borrow().target.as_ref(),
                Some(t) if std::rc::Rc::ptr_eq(t, &block)
            ));

            let n_ops = block.borrow().operations.len();
            let no_exitswitch = block.borrow().exitswitch.is_none();
            let splitlink: LinkRef = if position == n_ops && no_exitswitch {
                // Upstream `:243-247`: a split here would leave nothing
                // in the suffix, so reuse `block.exits[0]` directly.
                let exits = block.borrow().exits.clone();
                assert_eq!(exits.len(), 1, "rewire_links shortcut requires single exit",);
                exits.into_iter().next().unwrap()
            } else {
                // Upstream `:249-251`.
                let l = split_block(&block, position, None);
                debug_assert_eq!(block.borrow().exits.len(), 1);
                l
            };

            // Upstream `:252-253` invariants.
            debug_assert!(matches!(
                link.borrow().target.as_ref(),
                Some(t) if std::rc::Rc::ptr_eq(t, &block)
            ));
            debug_assert!(matches!(
                splitlink.borrow().prevblock.as_ref().and_then(|w| w.upgrade()),
                Some(p) if std::rc::Rc::ptr_eq(&p, &block)
            ));

            complete_constants(&link, &mut constants);

            // Upstream `:255 args = [constants.get(v, v) for v in splitlink.args]`.
            let splitlink_args = splitlink.borrow().args.clone();
            let new_args: Vec<LinkArg> = splitlink_args
                .into_iter()
                .map(|arg| match arg {
                    Some(Hlvalue::Variable(v)) => {
                        if let Some(c) = constants.get(&v).cloned() {
                            Some(c)
                        } else {
                            Some(Hlvalue::Variable(v))
                        }
                    }
                    other => other,
                })
                .collect();

            let new_target = splitlink.borrow().target.clone();
            let mut l = link.borrow_mut();
            l.args = new_args;
            l.target = new_target;
        }
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
/// Two-phase fold:
/// - Phase 1 (`:317-320`): per-block constant fold.
/// - Phase 2 (`:325-370`): fixpoint loop combining `constant_diffuse`
///   (block-level constant promotion / `same_as` hoisting) with the
///   link-level pass — for each incoming constant link, fold the
///   target block's prefix and either (a) rewire the link past the
///   folded prefix (via `prepare_constant_fold_link` →
///   `rewire_links` → `split_block`), or (b) handle the
///   single-`_ovf`-op block special case directly here, routing the
///   link to either the OverflowError exit or the normal exit
///   depending on whether the fold raised.
pub fn constant_fold_graph(graph: &FunctionGraph) {
    for block in graph.iterblocks() {
        if !block.borrow().operations.is_empty() {
            constant_fold_block(&block);
        }
    }
    loop {
        let diffused = constant_diffuse(graph);
        let mut splitblocks: SplitBlocks = HashMap::new();

        for link in graph.iterlinks() {
            // `:329-332`: build initial Constant-only constants from
            // link.args ↦ target.inputargs zip.
            let target = match link.borrow().target.as_ref() {
                Some(t) => t.clone(),
                None => continue,
            };
            let target_inputargs = target.borrow().inputargs.clone();
            let link_args = link.borrow().args.clone();
            let mut constants: LinkConstants = HashMap::new();
            for (v1, v2) in link_args.iter().zip(target_inputargs.iter()) {
                let Hlvalue::Variable(v2_var) = v2 else {
                    continue;
                };
                if let Some(Hlvalue::Constant(c)) = v1 {
                    constants.insert(v2_var.clone(), Hlvalue::Constant(c.clone()));
                }
            }
            if constants.is_empty() {
                continue;
            }

            // `:334-336`: detect single-op overflow block.
            let (lastop, is_ovfflow_block) = {
                let tb = target.borrow();
                let lastop = tb.operations.last().cloned();
                let is_ovf = tb.operations.len() == 1
                    && tb.canraise()
                    && lastop
                        .as_ref()
                        .map_or(false, |op| op.opname.ends_with("_ovf"));
                (lastop, is_ovf)
            };

            if !is_ovfflow_block {
                prepare_constant_fold_link(&link, &mut constants, &mut splitblocks);
                continue;
            }

            // `:341-364`: ovf-block special case. Try to fold the
            // single op directly with link-level constants. If any
            // arg is non-constant, abandon (no fold for THIS link).
            let lastop = lastop.expect("ovfflow_block requires last op");
            let mut all_const = true;
            let mut constargs: Vec<ConstValue> = Vec::with_capacity(lastop.args.len());
            for arg in &lastop.args {
                match arg {
                    Hlvalue::Constant(c) => constargs.push(c.value.clone()),
                    Hlvalue::Variable(v) => {
                        if let Some(Hlvalue::Constant(c)) = constants.get(v) {
                            constargs.push(c.value.clone());
                        } else {
                            all_const = false;
                            break;
                        }
                    }
                }
            }
            if !all_const {
                continue;
            }

            let res = fold_ovf_op(&lastop.opname, &constargs);
            let target_exits = target.borrow().exits.clone();
            assert!(
                target_exits.len() >= 2,
                "ovfflow_block must have a normal exit and an OverflowError exit"
            );

            let targetlink = match res {
                None => {
                    // `:355-356`: always overflows. Upstream
                    // `assert targetlink.exitcase is OverflowError`.
                    let exit1 = target_exits[1].clone();
                    debug_assert_eq!(
                        exit1.borrow().exitcase.as_ref().and_then(|hv| match hv {
                            Hlvalue::Constant(c) => c.value.host_class_name().map(str::to_owned),
                            _ => None,
                        }),
                        Some("OverflowError".to_string()),
                        "constfold.py:356 ovf overflow link expected OverflowError exitcase",
                    );
                    exit1
                }
                Some(value) => {
                    // `:358-361`: doesn't overflow. Upstream
                    // `assert targetlink.exitcase is None`.
                    let exit0 = target_exits[0].clone();
                    debug_assert!(
                        exit0.borrow().exitcase.is_none(),
                        "constfold.py:360 ovf normal link expected exitcase=None",
                    );
                    if let Hlvalue::Variable(result_var) = &lastop.result {
                        constants.insert(
                            result_var.clone(),
                            Hlvalue::Constant(constant_for_result(value, &lastop.result)),
                        );
                    }
                    exit0
                }
            };

            complete_constants(&link, &mut constants);

            // `:363 link.args = [constants.get(v, v) for v in targetlink.args]`.
            let targetlink_args = targetlink.borrow().args.clone();
            let new_args: Vec<LinkArg> = targetlink_args
                .into_iter()
                .map(|arg| match arg {
                    Some(Hlvalue::Variable(v)) => {
                        if let Some(c) = constants.get(&v).cloned() {
                            Some(c)
                        } else {
                            Some(Hlvalue::Variable(v))
                        }
                    }
                    other => other,
                })
                .collect();
            let new_target = targetlink.borrow().target.clone();
            let mut l = link.borrow_mut();
            l.args = new_args;
            l.target = new_target;
        }

        let had_splitblocks = !splitblocks.is_empty();
        if had_splitblocks {
            rewire_links(splitblocks);
        }
        if diffused == 0 && !had_splitblocks {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{Block, FunctionGraph, Link};

    fn var(name: &str) -> Variable {
        Variable::named(name)
    }

    fn hv(v: &Variable) -> Hlvalue {
        Hlvalue::Variable(v.clone())
    }

    fn const_int(n: i64) -> Hlvalue {
        Hlvalue::Constant(Constant::new(ConstValue::Int(n)))
    }

    /// `complete_constants` records the link's value for each target
    /// inputarg not yet present in `constants`.
    #[test]
    fn complete_constants_inserts_missing_target_inputargs() {
        let v_target_0 = var("t0");
        let v_target_1 = var("t1");
        let target = Block::shared(vec![hv(&v_target_0), hv(&v_target_1)]);
        let v_link_0 = var("l0");
        let link = Link::new(vec![hv(&v_link_0), const_int(42)], Some(target), None).into_ref();

        let mut constants: LinkConstants = HashMap::new();
        complete_constants(&link, &mut constants);

        // t0 → Variable l0; t1 → Constant 42.
        assert_eq!(constants.len(), 2);
        assert!(matches!(
            constants.get(&v_target_0),
            Some(Hlvalue::Variable(v)) if v.id() == v_link_0.id()
        ));
        assert!(matches!(
            constants.get(&v_target_1),
            Some(Hlvalue::Constant(c)) if c.value == ConstValue::Int(42)
        ));
    }

    /// Pre-existing entries in `constants` that match the link's value
    /// are tolerated; mismatches panic per upstream's `assert
    /// constants[v2] is v1`.
    #[test]
    fn complete_constants_tolerates_matching_preexisting_entry() {
        let v_target = var("t0");
        let target = Block::shared(vec![hv(&v_target)]);
        let link = Link::new(vec![const_int(7)], Some(target), None).into_ref();

        let mut constants: LinkConstants = HashMap::new();
        constants.insert(v_target.clone(), const_int(7));
        complete_constants(&link, &mut constants);

        assert_eq!(constants.len(), 1);
    }

    #[test]
    #[should_panic(expected = "conflicting entry")]
    fn complete_constants_panics_on_conflicting_preexisting_entry() {
        let v_target = var("t0");
        let target = Block::shared(vec![hv(&v_target)]);
        let link = Link::new(vec![const_int(7)], Some(target), None).into_ref();

        let mut constants: LinkConstants = HashMap::new();
        constants.insert(v_target.clone(), const_int(99));
        complete_constants(&link, &mut constants);
    }

    /// `rewire_link_for_known_exitswitch` jumps `link1` directly to
    /// the matching exit's target, propagating its args through the
    /// block's inputarg → link1.args mapping.
    #[test]
    fn rewire_link_for_known_exitswitch_routes_to_matching_exit() {
        let v_in = var("in");
        let switch_block = Block::shared(vec![hv(&v_in)]);
        let dest_a = Block::shared(vec![hv(&var("a_in"))]);
        let dest_b = Block::shared(vec![hv(&var("b_in"))]);

        let case_a = Constant::with_concretetype(
            ConstValue::Int(0),
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Signed,
        );
        let case_b = Constant::with_concretetype(
            ConstValue::Int(1),
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Signed,
        );
        let exit_a = Link::new(
            vec![hv(&v_in)],
            Some(dest_a.clone()),
            Some(Hlvalue::Constant(case_a.clone())),
        )
        .into_ref();
        exit_a.borrow_mut().llexitcase = Some(Hlvalue::Constant(case_a));
        let exit_b = Link::new(
            vec![hv(&v_in)],
            Some(dest_b.clone()),
            Some(Hlvalue::Constant(case_b.clone())),
        )
        .into_ref();
        exit_b.borrow_mut().llexitcase = Some(Hlvalue::Constant(case_b));
        switch_block.closeblock(vec![exit_a, exit_b]);

        let v_caller = var("caller_v");
        let link1 = Link::new(vec![hv(&v_caller)], Some(switch_block), None).into_ref();

        rewire_link_for_known_exitswitch(&link1, &ConstValue::Int(1));

        let l = link1.borrow();
        assert!(matches!(
            l.target.as_ref(),
            Some(t) if std::rc::Rc::ptr_eq(t, &dest_b)
        ));
        // The destination's inputarg is fed `v_caller` (the original
        // caller variable, since exit_b's args was [v_in] and
        // blockmapping[v_in] = v_caller).
        assert_eq!(l.args.len(), 1);
        assert!(matches!(
            &l.args[0],
            Some(Hlvalue::Variable(v)) if v.id() == v_caller.id()
        ));
    }

    /// When no exit's `llexitcase` matches, fall through to the
    /// trailing `'default'` exit if present.
    #[test]
    fn rewire_link_for_known_exitswitch_falls_through_to_default() {
        let v_in = var("in");
        let switch_block = Block::shared(vec![hv(&v_in)]);
        let dest_a = Block::shared(vec![hv(&var("a_in"))]);
        let dest_default = Block::shared(vec![hv(&var("d_in"))]);

        let case_a = Constant::with_concretetype(
            ConstValue::Int(0),
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Signed,
        );
        let exit_a = Link::new(
            vec![hv(&v_in)],
            Some(dest_a),
            Some(Hlvalue::Constant(case_a.clone())),
        )
        .into_ref();
        exit_a.borrow_mut().llexitcase = Some(Hlvalue::Constant(case_a));
        let default_exit = Link::new(
            vec![hv(&v_in)],
            Some(dest_default.clone()),
            Some(Hlvalue::Constant(Constant::new(ConstValue::ByteStr(
                b"default".to_vec(),
            )))),
        )
        .into_ref();
        switch_block.closeblock(vec![exit_a, default_exit]);

        let v_caller = var("caller_v");
        let link1 = Link::new(vec![hv(&v_caller)], Some(switch_block), None).into_ref();

        // Looking for case `42` which doesn't match `0` — must fall
        // through to default.
        rewire_link_for_known_exitswitch(&link1, &ConstValue::Int(42));

        let l = link1.borrow();
        assert!(matches!(
            l.target.as_ref(),
            Some(t) if std::rc::Rc::ptr_eq(t, &dest_default)
        ));
    }

    /// `constant_fold_graph` link-level pass: when an incoming link's
    /// arg is a constant that the target block's prefix can fold
    /// against, the link is rewired past the folded prefix.
    ///
    ///   A: link.args=[Const(5)] → B
    ///   B: inputargs=[x], ops=[int_neg(x) → r], link.args=[r] → ret
    ///
    /// After: A's link should jump directly to (a successor of) B
    /// with the folded constant carried through.
    #[test]
    fn constant_fold_graph_rewires_link_past_folded_prefix() {
        let v_x = var("x");
        let v_r = var("r");
        let v_ret = var("retv");

        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let block_b = Block::shared(vec![hv(&v_x)]);
        block_b.borrow_mut().operations.push(SpaceOperation::new(
            "int_neg",
            vec![hv(&v_x)],
            hv(&v_r),
        ));
        let exit_b = Link::new(vec![hv(&v_r)], Some(graph.returnblock.clone()), None).into_ref();
        block_b.closeblock(vec![exit_b]);

        // Block A → Block B via a constant link arg.
        let exit_a = Link::new(
            vec![Hlvalue::Constant(Constant::new(ConstValue::Int(5)))],
            Some(block_b.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![exit_a]);

        // Make returnblock's inputarg distinct so the assertion below
        // is unambiguous.
        let _ = v_ret;

        constant_fold_graph(&graph);

        // After fold: every link transitively reachable from start
        // that lands on returnblock must carry Const(-5) — proving
        // int_neg(5) = -5 propagated through whatever block was left
        // intact by the link-level rewire.
        let mut found_neg5 = false;
        for link in graph.iterlinks() {
            let l = link.borrow();
            let target_is_return = l
                .target
                .as_ref()
                .map(|t| std::rc::Rc::ptr_eq(t, &graph.returnblock))
                .unwrap_or(false);
            if !target_is_return {
                continue;
            }
            for arg in &l.args {
                if matches!(arg, Some(Hlvalue::Constant(c)) if c.value == ConstValue::Int(-5)) {
                    found_neg5 = true;
                }
            }
        }
        assert!(
            found_neg5,
            "expected a link to returnblock carrying Constant(Int(-5)) after fold"
        );
    }

    /// When neither `llexitcase` matches AND there is no `'default'`
    /// exit, leave `link1` untouched and return silently
    /// (upstream `:182-184`).
    #[test]
    fn rewire_link_for_known_exitswitch_returns_silently_when_unmatched_no_default() {
        let v_in = var("in");
        let switch_block = Block::shared(vec![hv(&v_in)]);
        let dest_a = Block::shared(vec![hv(&var("a_in"))]);
        let case_a = Constant::with_concretetype(
            ConstValue::Int(0),
            crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Signed,
        );
        let exit_a = Link::new(
            vec![hv(&v_in)],
            Some(dest_a),
            Some(Hlvalue::Constant(case_a.clone())),
        )
        .into_ref();
        exit_a.borrow_mut().llexitcase = Some(Hlvalue::Constant(case_a));
        switch_block.closeblock(vec![exit_a]);

        let v_caller = var("caller_v");
        let link1_target_orig = switch_block.clone();
        let link1 = Link::new(vec![hv(&v_caller)], Some(switch_block), None).into_ref();

        rewire_link_for_known_exitswitch(&link1, &ConstValue::Int(99));

        // link1 unchanged.
        let l = link1.borrow();
        assert!(matches!(
            l.target.as_ref(),
            Some(t) if std::rc::Rc::ptr_eq(t, &link1_target_orig)
        ));
    }
}
