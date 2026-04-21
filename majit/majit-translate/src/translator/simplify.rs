//! RPython `rpython/translator/simplify.py` — graph-level transformations
//! invoked from `RPythonAnnotator.simplify()` (annrpython.py:336-371)
//! and `simplify_graph` (simplify.py:1075+).
//!
//! Only the subset reachable from the annotator port lands here
//! initially; downstream simplification passes arrive with the
//! rtyper port.

use std::collections::{HashMap, HashSet};

use crate::flowspace::model::{
    BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphRef, HOST_ENV,
    Hlvalue, HostObject, LinkRef, SpaceOperation, Variable, checkgraph, mkentrymap,
};
use crate::translator::backendopt::ssa::DataFlowFamilyBuilder;
use crate::translator::translator::TranslationContext;

/// RPython `eliminate_empty_blocks(graph)` (simplify.py:52-69).
///
/// ```python
/// def eliminate_empty_blocks(graph):
///     """Eliminate basic blocks that do not contain any operations.
///     When this happens, we need to replace the preceeding link with
///     the following link.  Arguments of the links should be updated."""
///     for link in list(graph.iterlinks()):
///         while not link.target.operations:
///             block1 = link.target
///             if block1.exitswitch is not None:
///                 break
///             if not block1.exits:
///                 break
///             exit = block1.exits[0]
///             assert block1 is not exit.target, (
///                 "the graph contains an empty infinite loop")
///             subst = dict(zip(block1.inputargs, link.args))
///             link.args = [v.replace(subst) for v in exit.args]
///             link.target = exit.target
/// ```
///
/// The Rust port mirrors upstream's single-threaded mutation pattern:
/// walk each snapshot link, collapse chains of empty successor
/// blocks, and rewrite the link's `args` / `target` in place.
pub fn eliminate_empty_blocks(graph: &FunctionGraph) {
    // upstream: `for link in list(graph.iterlinks()):`
    for link_ref in graph.iterlinks() {
        loop {
            // upstream: `while not link.target.operations:`
            let (is_empty, has_switch, has_exits, next_exit, target_rc) = {
                let link = link_ref.borrow();
                let Some(target) = link.target.as_ref() else {
                    break;
                };
                let target_b = target.borrow();
                let is_empty = target_b.operations.is_empty();
                let has_switch = target_b.exitswitch.is_some();
                let has_exits = !target_b.exits.is_empty();
                let next_exit = target_b.exits.first().cloned();
                (is_empty, has_switch, has_exits, next_exit, target.clone())
            };
            if !is_empty {
                break;
            }
            if has_switch {
                // upstream: `if block1.exitswitch is not None: break`.
                break;
            }
            if !has_exits {
                // upstream: `if not block1.exits: break`.
                break;
            }
            // upstream: `exit = block1.exits[0]`.
            let Some(exit_ref) = next_exit else {
                break;
            };

            // upstream: `assert block1 is not exit.target, ...`.
            {
                let exit = exit_ref.borrow();
                if let Some(exit_target) = &exit.target {
                    assert!(
                        !std::rc::Rc::ptr_eq(exit_target, &target_rc),
                        "the graph contains an empty infinite loop"
                    );
                }
            }

            // Build `subst = dict(zip(block1.inputargs, link.args))` —
            // the Link carries `Vec<LinkArg> = Vec<Option<Hlvalue>>`,
            // so we keep entries with concrete values only and skip
            // the transient-merge `None` slots.
            let subst: HashMap<Variable, Hlvalue> = {
                let target_b = target_rc.borrow();
                let link = link_ref.borrow();
                target_b
                    .inputargs
                    .iter()
                    .zip(link.args.iter())
                    .filter_map(|(formal, actual)| {
                        let Hlvalue::Variable(v) = formal else {
                            return None;
                        };
                        actual.as_ref().map(|a| (v.clone(), a.clone()))
                    })
                    .collect()
            };

            // upstream: `link.args = [v.replace(subst) for v in exit.args]`.
            let new_args: Vec<Option<Hlvalue>> = {
                let exit = exit_ref.borrow();
                exit.args
                    .iter()
                    .map(|arg| {
                        arg.as_ref().map(|v| match v {
                            Hlvalue::Variable(var) => subst
                                .get(var)
                                .cloned()
                                .unwrap_or_else(|| Hlvalue::Variable(var.clone())),
                            other => other.clone(),
                        })
                    })
                    .collect()
            };

            // upstream: `link.target = exit.target`.
            let new_target = exit_ref.borrow().target.clone();

            {
                let mut link = link_ref.borrow_mut();
                link.args = new_args;
                link.target = new_target;
            }
            // upstream: the while-loop re-evaluates on the rewritten
            // link so chains of empty blocks collapse in one pass.
        }
    }
}

/// Inline helper — upstream's `.replace(mapping)` when `mapping` carries
/// `dict[Variable, Hlvalue]`. The Rust port's `Hlvalue.replace` method
/// narrows its signature to `HashMap<Variable, Variable>` because that
/// is all its existing callers need (`copygraph`, `renamevariables`);
/// `join_blocks` needs the upstream-wide semantics (a Variable can map
/// to a Constant when the caller threads through `link.args`). This
/// helper keeps the wide semantics local to `simplify.rs` instead of
/// widening the public API surface.
fn rename_hl(v: &Hlvalue, mapping: &HashMap<Variable, Hlvalue>) -> Hlvalue {
    match v {
        Hlvalue::Variable(var) => mapping
            .get(var)
            .cloned()
            .unwrap_or_else(|| Hlvalue::Variable(var.clone())),
        Hlvalue::Constant(_) => v.clone(),
    }
}

fn rename_op(op: &SpaceOperation, mapping: &HashMap<Variable, Hlvalue>) -> SpaceOperation {
    // upstream: `op = op.replace(renaming)`.
    let mut new_op = SpaceOperation {
        opname: op.opname.clone(),
        args: op.args.iter().map(|a| rename_hl(a, mapping)).collect(),
        result: rename_hl(&op.result, mapping),
        offset: op.offset,
    };
    // upstream: indirect_call with a Constant callee is rewritten back
    // to direct_call (simplify.py:293-297).
    if new_op.opname == "indirect_call" {
        if matches!(new_op.args.first(), Some(Hlvalue::Constant(_))) {
            assert!(
                matches!(new_op.args.last(), Some(Hlvalue::Constant(_))),
                "indirect_call's trailing argument must be a Constant"
            );
            new_op.args.pop();
            new_op.opname = "direct_call".to_string();
        }
    }
    new_op
}

fn rename_link_args(link: &LinkRef, mapping: &HashMap<Variable, Hlvalue>) -> LinkRef {
    // upstream's `link.replace(mapping)` path — rebuild a fresh Link
    // with renamed args / last_exception / last_exc_value.
    let src = link.borrow();
    let mut newlink = crate::flowspace::model::Link::new_mergeable(
        src.args
            .iter()
            .map(|a| a.as_ref().map(|v| rename_hl(v, mapping)))
            .collect(),
        src.target.clone(),
        src.exitcase.clone(),
    );
    newlink.prevblock = src.prevblock.clone();
    newlink.llexitcase = src.llexitcase.clone();
    newlink.last_exception = src.last_exception.as_ref().map(|v| rename_hl(v, mapping));
    newlink.last_exc_value = src.last_exc_value.as_ref().map(|v| rename_hl(v, mapping));
    newlink.into_ref()
}

/// RPython `replace_exitswitch_by_constant(block, const)`
/// (simplify.py:36-48).
///
/// ```python
/// def replace_exitswitch_by_constant(block, const):
///     newexits = [link for link in block.exits
///                      if link.exitcase == const.value]
///     if len(newexits) == 0:
///         newexits = [link for link in block.exits
///                      if link.exitcase == 'default']
///     assert len(newexits) == 1
///     newexits[0].exitcase = None
///     if hasattr(newexits[0], 'llexitcase'):
///         newexits[0].llexitcase = None
///     block.exitswitch = None
///     block.recloseblock(*newexits)
///     return newexits
/// ```
pub fn replace_exitswitch_by_constant(block: &BlockRef, const_: &Constant) -> Vec<LinkRef> {
    // upstream: `if link.exitcase == const.value` — our `Link.exitcase`
    // is `Option<Hlvalue>` wrapping a Constant carrying `ConstValue`;
    // compare on the inner `ConstValue`.
    let cases_eq = |link: &LinkRef| match &link.borrow().exitcase {
        Some(Hlvalue::Constant(c)) => c.value == const_.value,
        _ => false,
    };
    let default_case = |link: &LinkRef| {
        matches!(
            &link.borrow().exitcase,
            Some(Hlvalue::Constant(c)) if matches!(&c.value, ConstValue::Str(s) if s == "default")
        )
    };

    let exits_snapshot: Vec<LinkRef> = block.borrow().exits.iter().cloned().collect();
    let mut newexits: Vec<LinkRef> = exits_snapshot
        .iter()
        .filter(|l| cases_eq(l))
        .cloned()
        .collect();
    if newexits.is_empty() {
        newexits = exits_snapshot
            .iter()
            .filter(|l| default_case(l))
            .cloned()
            .collect();
    }
    assert_eq!(
        newexits.len(),
        1,
        "replace_exitswitch_by_constant: no unique surviving exit"
    );
    {
        let mut l = newexits[0].borrow_mut();
        l.exitcase = None;
        l.llexitcase = None;
    }
    block.borrow_mut().exitswitch = None;
    block.recloseblock(newexits.clone());
    newexits
}

/// RPython `CanRemove = {...}` (simplify.py:405-417).
///
/// Upstream also fills `CanRemove` with everything from
/// `lloperation.enum_ops_without_sideeffects()`; that rtyper-phase
/// table is not yet ported, so the Rust port includes only the
/// explicit high-level opname list. Annotator-phase callers
/// (`transform_dead_op_vars` via `translator/transform.py`) see the
/// exact same subset they'd see upstream during the annotator pass.
fn can_remove_opnames() -> &'static HashSet<&'static str> {
    use std::sync::OnceLock;
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        [
            "newtuple",
            "newlist",
            "newdict",
            "bool",
            "is_",
            "id",
            "type",
            "issubtype",
            "isinstance",
            "repr",
            "str",
            "len",
            "hash",
            "getattr",
            "getitem",
            "pos",
            "neg",
            "abs",
            "hex",
            "oct",
            "ord",
            "invert",
            "add",
            "sub",
            "mul",
            "truediv",
            "floordiv",
            "div",
            "mod",
            "divmod",
            "pow",
            "lshift",
            "rshift",
            "and_",
            "or_",
            "xor",
            "int",
            "float",
            "long",
            "lt",
            "le",
            "eq",
            "ne",
            "gt",
            "ge",
            "cmp",
            "coerce",
            "contains",
            "iter",
            "get",
        ]
        .into_iter()
        .collect()
    })
}

/// RPython `CanRemoveBuiltins = {hasattr: True}` (simplify.py:418-420).
///
/// Compared by HostObject identity against `HOST_ENV.lookup_builtin("hasattr")`.
fn can_remove_builtins() -> Vec<HostObject> {
    let mut v = Vec::new();
    if let Some(h) = HOST_ENV.lookup_builtin("hasattr") {
        v.push(h);
    }
    v
}

/// RPython `get_graph(arg, translator)` (simplify.py:20-33).
///
/// Returns the `FunctionGraph` associated with a `direct_call`'s
/// constant callee, or `None` when the callee is a Variable or a
/// non-function constant. The Rust port matches on the user-function
/// shape exposed via `HostObject::user_function()` and walks the
/// translator's `graphs` list for a graph whose name matches the
/// function's `name` attribute — a lightweight stand-in for upstream's
/// `lltype._ptr._obj.graph` traversal.
fn get_graph_for_call(arg: &Hlvalue, translator: &TranslationContext) -> Option<GraphRef> {
    let Hlvalue::Constant(c) = arg else {
        return None;
    };
    let ConstValue::HostObject(h) = &c.value else {
        return None;
    };
    let gf = h.user_function()?;
    let fn_name = &gf.name;
    for g in translator.graphs.borrow().iter() {
        if g.borrow().name == *fn_name {
            return Some(g.clone());
        }
    }
    None
}

/// RPython `has_no_side_effects(translator, graph, seen=None)`
/// (simplify.py:355+). The full upstream logic recurses through
/// `lloperation.LL_OPERATIONS[...].sideeffects` and the callgraph;
/// neither piece is ported yet. Conservatively return `false` — the
/// direct_call pruning branch then leaves the op alone, matching the
/// upstream call shape when `translator is None`.
fn has_no_side_effects(_translator: &TranslationContext, _graph: &GraphRef) -> bool {
    false
}

/// RPython `transform_dead_op_vars(graph, translator=None)`
/// (simplify.py:397-401). Thin wrapper around
/// `transform_dead_op_vars_in_blocks`.
pub fn transform_dead_op_vars(graph: &FunctionGraph, translator: Option<&TranslationContext>) {
    transform_dead_op_vars_in_blocks(&graph.iterblocks(), &[graph as *const _], translator, graph);
}

/// RPython `transform_dead_op_vars_in_blocks(blocks, graphs, translator=None)`
/// (simplify.py:422-524). The `graphs` parameter degenerates to
/// a single graph reference plus the optional translator; the
/// multi-graph case (upstream when called from `transform.py`) routes
/// through `translator.annotator.annotated[block].startblock` which
/// the Rust port models via the `annotator_lookup` closure argument
/// on callers that need it (`translator/transform.rs::
/// transform_dead_op_vars`).
pub fn transform_dead_op_vars_in_blocks(
    blocks: &[BlockRef],
    _graphs_sentinel: &[*const FunctionGraph],
    translator: Option<&TranslationContext>,
    single_graph: &FunctionGraph,
) {
    // upstream: `set_of_blocks = set(blocks)`.
    let set_of_blocks: HashSet<BlockKey> = blocks.iter().map(BlockKey::of).collect();
    // upstream: `start_blocks = {graphs[0].startblock}` when single-
    // graph. Multi-graph path (upstream path through
    // `translator.annotator.annotated[block].startblock`) lands with
    // the `ann.translator` wiring at the transform.py call site —
    // here we operate on one graph at a time.
    let start_block_key = BlockKey::of(&single_graph.startblock);
    let can_remove_ops = can_remove_opnames();
    let can_remove_builtins_list = can_remove_builtins();

    // `read_vars`: set of Variables whose value is read downstream.
    let mut read_vars: HashSet<Variable> = HashSet::new();
    // `dependencies`: map from Var → [dependency Vars].
    let mut dependencies: HashMap<Variable, HashSet<Variable>> = HashMap::new();

    let canremove_op = |op: &SpaceOperation, block: &BlockRef| -> bool {
        if !can_remove_ops.contains(op.opname.as_str()) {
            return false;
        }
        // upstream: `op is not block.raising_op`.
        let raising = block.borrow().raising_op().cloned();
        match raising {
            Some(r) => *op != r,
            None => true,
        }
    };
    let add_read_var = |read_vars: &mut HashSet<Variable>, v: &Hlvalue| {
        if let Hlvalue::Variable(var) = v {
            read_vars.insert(var.clone());
        }
    };
    let add_dep =
        |deps: &mut HashMap<Variable, HashSet<Variable>>, dest: &Hlvalue, src: &Hlvalue| {
            if let (Hlvalue::Variable(dv), Hlvalue::Variable(sv)) = (dest, src) {
                deps.entry(dv.clone()).or_default().insert(sv.clone());
            }
        };

    // upstream: `for block in blocks: compute read_vars + dependencies`.
    for block in blocks {
        let b = block.borrow();
        for op in &b.operations {
            if !canremove_op(op, block) {
                // upstream: `read_vars.update(op.args)`.
                for a in &op.args {
                    add_read_var(&mut read_vars, a);
                }
            } else {
                for a in &op.args {
                    add_dep(&mut dependencies, &op.result, a);
                }
            }
        }
        // upstream: `if isinstance(block.exitswitch, Variable): read_vars.add(block.exitswitch)`.
        if let Some(Hlvalue::Variable(sw)) = &b.exitswitch {
            read_vars.insert(sw.clone());
        }

        if !b.exits.is_empty() {
            for link in &b.exits {
                let link_b = link.borrow();
                let Some(target) = link_b.target.as_ref() else {
                    continue;
                };
                let target_key = BlockKey::of(target);
                let target_inputargs = target.borrow().inputargs.clone();
                let link_args: Vec<Option<Hlvalue>> = link_b.args.clone();
                let in_set = set_of_blocks.contains(&target_key);
                for (arg_opt, tgt) in link_args.iter().zip(target_inputargs.iter()) {
                    let Some(arg) = arg_opt else { continue };
                    if !in_set {
                        // upstream: `read_vars.add(arg); read_vars.add(targetarg)`.
                        add_read_var(&mut read_vars, arg);
                        add_read_var(&mut read_vars, tgt);
                    } else {
                        add_dep(&mut dependencies, tgt, arg);
                    }
                }
            }
        } else {
            // upstream: return/except blocks implicitly use inputargs.
            for a in &b.inputargs {
                add_read_var(&mut read_vars, a);
            }
        }

        // upstream: start block's inputargs always live.
        if BlockKey::of(block) == start_block_key {
            for a in &b.inputargs {
                add_read_var(&mut read_vars, a);
            }
        }
    }

    // upstream: `flow_read_var_backward(set(read_vars))`.
    let mut pending: Vec<Variable> = read_vars.iter().cloned().collect();
    while let Some(var) = pending.pop() {
        if let Some(deps) = dependencies.get(&var).cloned() {
            for prev in deps {
                if read_vars.insert(prev.clone()) {
                    pending.push(prev);
                }
            }
        }
    }

    // upstream: iterate blocks again, prune removable ops whose result
    // isn't read, prune link args whose target-inputarg isn't read.
    for block in blocks {
        // upstream: backward walk over operations.
        let ops_len = block.borrow().operations.len();
        let mut i = ops_len;
        while i > 0 {
            i -= 1;
            let (op, result_used, opname, first_arg_builtin) = {
                let b = block.borrow();
                let op = b.operations[i].clone();
                let result_used = match &op.result {
                    Hlvalue::Variable(v) => read_vars.contains(v),
                    Hlvalue::Constant(_) => true,
                };
                let first_arg_builtin = match op.args.first() {
                    Some(Hlvalue::Constant(Constant {
                        value: ConstValue::HostObject(h),
                        ..
                    })) => Some(h.clone()),
                    _ => None,
                };
                let opname = op.opname.clone();
                (op, result_used, opname, first_arg_builtin)
            };
            if result_used {
                continue;
            }
            if canremove_op(&op, block) {
                block.borrow_mut().operations.remove(i);
            } else if opname == "simple_call" {
                if let Some(h) = first_arg_builtin {
                    if can_remove_builtins_list.iter().any(|b| b == &h) {
                        block.borrow_mut().operations.remove(i);
                    }
                }
            } else if opname == "direct_call" {
                if let Some(trans) = translator {
                    let Some(callee_arg) = op.args.first() else {
                        continue;
                    };
                    if let Some(graph) = get_graph_for_call(callee_arg, trans) {
                        let is_raising = block
                            .borrow()
                            .raising_op()
                            .cloned()
                            .map(|r| r == op)
                            .unwrap_or(false);
                        if has_no_side_effects(trans, &graph) && !is_raising {
                            block.borrow_mut().operations.remove(i);
                        }
                    }
                }
            }
        }

        // upstream: output vars never used — drop their positions from
        // link.args. Must happen before block.inputargs is shrunk so
        // the same-index cross-block invariant holds.
        let exits_snapshot: Vec<LinkRef> = block.borrow().exits.iter().cloned().collect();
        for link in exits_snapshot {
            let target = link.borrow().target.clone();
            let Some(target) = target else { continue };
            let target_inputargs = target.borrow().inputargs.clone();
            let args_len = link.borrow().args.len();
            assert_eq!(args_len, target_inputargs.len(), "link arity mismatch");
            // upstream: back-to-front deletion.
            let mut i = args_len;
            while i > 0 {
                i -= 1;
                let tgt = &target_inputargs[i];
                let drop = match tgt {
                    Hlvalue::Variable(v) => !read_vars.contains(v),
                    Hlvalue::Constant(_) => false,
                };
                if drop {
                    link.borrow_mut().args.remove(i);
                }
            }
        }
    }

    // upstream: final pass — drop unused inputargs (matching link.args
    // are already gone).
    for block in blocks {
        let inputargs = block.borrow().inputargs.clone();
        let mut i = inputargs.len();
        while i > 0 {
            i -= 1;
            let drop = match &inputargs[i] {
                Hlvalue::Variable(v) => !read_vars.contains(v),
                Hlvalue::Constant(_) => false,
            };
            if drop {
                block.borrow_mut().inputargs.remove(i);
            }
        }
    }
}

/// RPython `class Representative` (simplify.py:526-531).
///
/// Info payload attached to each UF partition in
/// `remove_identical_vars_SSA`. Stores the per-partition
/// "representative" value; `absorb` is a no-op so the info that wins
/// the weighted union keeps its `rep`.
#[derive(Clone, Debug)]
struct Representative {
    rep: Hlvalue,
}

impl crate::tool::algo::unionfind::UnionFindInfo for Representative {
    fn absorb(&mut self, _other: Self) {}
}

/// RPython `all_equal(lst)` (simplify.py:533-535).
fn all_equal_hl(lst: &[Hlvalue]) -> bool {
    match lst.first() {
        None => true,
        Some(first) => lst.iter().skip(1).all(|x| x == first),
    }
}

/// RPython `isspecialvar(v)` (simplify.py:537-538).
fn isspecialvar(v: &Hlvalue) -> bool {
    match v {
        Hlvalue::Variable(var) => {
            let p = var.name_prefix();
            p == "last_exception_" || p == "last_exc_value_"
        }
        _ => false,
    }
}

/// Variable→Hlvalue counterpart of `Block::renamevariables` that
/// upstream's `block.renamevariables(mapping)` needs when the mapping
/// values are mixed Variable/Constant (as happens in
/// `remove_identical_vars_SSA`). The Rust port's
/// `Block::renamevariables` narrows to `HashMap<Variable, Variable>`;
/// see the note above `rename_hl` for why we keep the wide path
/// file-local.
fn renamevariables_hl(block: &BlockRef, mapping: &HashMap<Variable, Hlvalue>) {
    if mapping.is_empty() {
        return;
    }
    let new_ops: Vec<SpaceOperation> = {
        let b = block.borrow();
        b.operations
            .iter()
            .map(|op| rename_op(op, mapping))
            .collect()
    };
    let new_exitswitch = {
        let b = block.borrow();
        b.exitswitch.as_ref().map(|sw| rename_hl(sw, mapping))
    };
    {
        let mut b = block.borrow_mut();
        b.operations = new_ops;
        b.exitswitch = new_exitswitch;
    }
    // link.args are Vec<Option<Hlvalue>>; rename in place.
    let exits_snapshot: Vec<LinkRef> = block.borrow().exits.iter().cloned().collect();
    for link in exits_snapshot {
        let mut l = link.borrow_mut();
        l.args = l
            .args
            .iter()
            .map(|arg| arg.as_ref().map(|v| rename_hl(v, mapping)))
            .collect();
        if let Some(sw) = &l.last_exception {
            l.last_exception = Some(rename_hl(sw, mapping));
        }
        if let Some(sw) = &l.last_exc_value {
            l.last_exc_value = Some(rename_hl(sw, mapping));
        }
    }
}

/// RPython `remove_identical_vars_SSA(graph)` (simplify.py:540-595).
///
/// Upstream variant that uses its own `UnionFind(Representative)`
/// instead of `DataFlowFamilyBuilder`, inlining the phi-node collapse
/// loop over block inputs with linked `link.args[i]` as phi sources.
pub fn remove_identical_vars_ssa(graph: &FunctionGraph) {
    use crate::tool::algo::unionfind::UnionFind;

    let mut uf: UnionFind<Hlvalue, Representative> =
        UnionFind::new(|k: &Hlvalue| Representative { rep: k.clone() });

    // upstream: `entrymap = mkentrymap(graph); del entrymap[startblock];
    // entrymap.pop(returnblock, None); entrymap.pop(exceptblock, None)`.
    let mut entrymap = mkentrymap(graph);
    entrymap.remove(&BlockKey::of(&graph.startblock));
    entrymap.remove(&BlockKey::of(&graph.returnblock));
    entrymap.remove(&BlockKey::of(&graph.exceptblock));

    // upstream: `inputs[block] = zip(block.inputargs, zip(*[link.args
    // for link in links]))`. Rust: Vec<(Variable, Vec<Hlvalue>)> per
    // block. We key by BlockKey and carry a BlockRef alongside for
    // later iteration.
    let mut inputs: HashMap<BlockKey, (BlockRef, Vec<(Variable, Vec<Hlvalue>)>)> = HashMap::new();
    for (bkey, links) in entrymap.iter() {
        let block = links
            .first()
            .and_then(|l| l.borrow().target.clone())
            .expect("entrymap link missing target");
        let inputargs = block.borrow().inputargs.clone();
        let mut phis: Vec<(Variable, Vec<Hlvalue>)> = Vec::with_capacity(inputargs.len());
        for (i, input) in inputargs.iter().enumerate() {
            let Hlvalue::Variable(input_v) = input else {
                continue;
            };
            let mut phi_args: Vec<Hlvalue> = Vec::with_capacity(links.len());
            for link in links {
                let la: Vec<Option<Hlvalue>> = link.borrow().args.clone();
                let Some(Some(a)) = la.get(i).cloned() else {
                    panic!(
                        "remove_identical_vars_SSA: link.args[{i}] missing at block {:?}",
                        bkey
                    );
                };
                phi_args.push(a);
            }
            phis.push((input_v.clone(), phi_args));
        }
        inputs.insert(bkey.clone(), (block.clone(), phis));
    }

    // upstream: `progress = True; while progress: ... for block in
    // inputs: if simplify_phis(block): progress = True`.
    let block_keys: Vec<BlockKey> = inputs.keys().cloned().collect();
    let mut progress = true;
    while progress {
        progress = false;
        for bkey in &block_keys {
            if simplify_phis_inner(&mut uf, inputs.get_mut(bkey).unwrap()) {
                progress = true;
            }
        }
    }

    // upstream: `renaming = dict((key, uf[key].rep) for key in uf)` —
    // filter to Variable keys only (Constants-as-keys occur when
    // linked args brought them into the UF, but we only rename
    // Variables in block storage slots).
    let renaming: HashMap<Variable, Hlvalue> = {
        let keys: Vec<Hlvalue> = uf.keys().cloned().collect();
        let mut out: HashMap<Variable, Hlvalue> = HashMap::new();
        for k in keys {
            let Hlvalue::Variable(kv) = k.clone() else {
                continue;
            };
            if let Some(info) = uf.get(&k) {
                out.insert(kv, info.rep.clone());
            }
        }
        out
    };

    // upstream: rewrite each block's inputargs and every incoming
    // link's args to match the pruned `inputs[block]`. The order
    // matters — inputargs shrink before renamevariables runs so the
    // eliminated inputs' references in ops get the rep instead.
    for (bkey, (block, phis)) in inputs.iter() {
        let links = entrymap
            .get(bkey)
            .cloned()
            .expect("entrymap lookup consistent with inputs");
        let new_inputs: Vec<Hlvalue> = phis
            .iter()
            .map(|(v, _)| Hlvalue::Variable(v.clone()))
            .collect();
        // `per_link_args[link_idx][phi_idx] = phis[phi_idx].1[link_idx]`.
        let per_link_args: Vec<Vec<Option<Hlvalue>>> = (0..links.len())
            .map(|li| phis.iter().map(|(_, pa)| Some(pa[li].clone())).collect())
            .collect();
        block.borrow_mut().inputargs = new_inputs;
        assert_eq!(links.len(), per_link_args.len());
        for (link, args) in links.iter().zip(per_link_args.into_iter()) {
            link.borrow_mut().args = args;
        }
    }

    for (_, (block, _)) in inputs.iter() {
        renamevariables_hl(block, &renaming);
    }
}

/// Inner of `remove_identical_vars_ssa`'s `simplify_phis(block)` closure
/// (simplify.py:555-573).
fn simplify_phis_inner(
    uf: &mut crate::tool::algo::unionfind::UnionFind<Hlvalue, Representative>,
    slot: &mut (BlockRef, Vec<(Variable, Vec<Hlvalue>)>),
) -> bool {
    let (_block, phis) = slot;
    let mut to_remove: Vec<usize> = Vec::new();
    let mut unique_phis: HashMap<Vec<Hlvalue>, Variable> = HashMap::new();
    for (i, (input, phi_args)) in phis.iter().enumerate() {
        // upstream: `new_args = [uf.find_rep(arg) for arg in phi_args]`.
        let new_args: Vec<Hlvalue> = phi_args.iter().map(|a| uf.find_rep(a.clone())).collect();
        // upstream: `if all_equal(new_args) and not isspecialvar(new_args[0]):`
        let first = new_args.first().cloned();
        if let Some(first) = first {
            if all_equal_hl(&new_args) && !isspecialvar(&first) {
                uf.union(first, Hlvalue::Variable(input.clone()));
                to_remove.push(i);
                continue;
            }
        }
        // upstream: else branch — group by identical phi-tuple.
        let key = new_args;
        if let Some(existing) = unique_phis.get(&key).cloned() {
            uf.union(
                Hlvalue::Variable(existing),
                Hlvalue::Variable(input.clone()),
            );
            to_remove.push(i);
        } else {
            unique_phis.insert(key, input.clone());
        }
    }
    // upstream: `for i in reversed(to_remove): del phis[i]`.
    for i in to_remove.iter().rev() {
        phis.remove(*i);
    }
    !to_remove.is_empty()
}

/// RPython `remove_identical_vars(graph)` (simplify.py:597-653).
///
/// ```python
/// def remove_identical_vars(graph):
///     """When the same variable is passed multiple times into the next
///     block, pass it only once."""
///     builder = DataFlowFamilyBuilder(graph)
///     variable_families = builder.get_variable_families()  # vertical removal
///     while True:
///         if not builder.merge_identical_phi_nodes():    # horizontal removal
///             break
///         if not builder.complete():                     # vertical removal
///             break
///     for block, links in mkentrymap(graph).items():
///         if block is graph.startblock:
///             continue
///         renaming = {}
///         family2blockvar = {}
///         kills = []
///         for i, v in enumerate(block.inputargs):
///             v1 = variable_families.find_rep(v)
///             if v1 in family2blockvar:
///                 renaming[v] = family2blockvar[v1]
///                 kills.append(i)
///             else:
///                 family2blockvar[v1] = v
///         if renaming:
///             block.renamevariables(renaming)
///             kills.reverse()
///             for i in kills:
///                 del block.inputargs[i]
///                 for link in links:
///                     del link.args[i]
/// ```
pub fn remove_identical_vars(graph: &FunctionGraph) {
    let mut builder = DataFlowFamilyBuilder::new(graph);
    // upstream: `builder.get_variable_families()` triggers initial
    // vertical removal (calls `complete()`); we discard the returned
    // handle and drive the subsequent steps on the same builder.
    let _ = builder.get_variable_families();
    loop {
        // upstream: horizontal removal first, then vertical.
        if !builder.merge_identical_phi_nodes() {
            break;
        }
        if !builder.complete() {
            break;
        }
    }
    let variable_families = &mut builder.variable_families;

    let entrymap = mkentrymap(graph);
    for (block_key, links) in entrymap.iter() {
        // upstream: `if block is graph.startblock: continue`.
        if *block_key == BlockKey::of(&graph.startblock) {
            continue;
        }
        let block = links
            .first()
            .and_then(|l| l.borrow().target.clone())
            .expect("link.target missing");

        let mut renaming: HashMap<Variable, Variable> = HashMap::new();
        let mut family2blockvar: HashMap<Hlvalue, Variable> = HashMap::new();
        let mut kills: Vec<usize> = Vec::new();

        let inputargs_snapshot: Vec<Hlvalue> = block.borrow().inputargs.clone();
        for (i, input) in inputargs_snapshot.iter().enumerate() {
            let Hlvalue::Variable(v) = input else {
                continue;
            };
            let v1 = variable_families.find_rep(Hlvalue::Variable(v.clone()));
            if let Some(existing) = family2blockvar.get(&v1) {
                renaming.insert(v.clone(), existing.clone());
                kills.push(i);
            } else {
                family2blockvar.insert(v1, v.clone());
            }
        }
        if !renaming.is_empty() {
            block.borrow_mut().renamevariables(&renaming);
            // upstream: `kills.reverse()` + `del block.inputargs[i]`
            // + `del link.args[i]` for each incoming link.
            kills.reverse();
            for i in &kills {
                block.borrow_mut().inputargs.remove(*i);
                for link in links {
                    link.borrow_mut().args.remove(*i);
                }
            }
        }
    }
}

/// RPython `simplify_exceptions(graph)` (simplify.py:110-170).
///
/// Collapses the `except Exception:` chain-of-is_/issubtype tests
/// produced by the flowspace into a single list of `exitcase=cls`
/// links on the raising block.
///
/// ```python
/// def simplify_exceptions(graph):
///     renaming = {}
///     for block in graph.iterblocks():
///         if not (block.canraise
///                 and block.exits[-1].exitcase is Exception):
///             continue
///         covered = [link.exitcase for link in block.exits[1:-1]]
///         seen = []
///         preserve = list(block.exits[:-1])
///         exc = block.exits[-1]
///         last_exception = exc.last_exception
///         last_exc_value = exc.last_exc_value
///         query = exc.target
///         switches = []
///         while len(query.exits) == 2:
///             newrenaming = {}
///             for lprev, ltarg in zip(exc.args, query.inputargs):
///                 newrenaming[ltarg] = lprev.replace(renaming)
///             op = query.operations[0]
///             if not (op.opname in ("is_", "issubtype") and
///                     op.args[0].replace(newrenaming) == last_exception):
///                 break
///             renaming.update(newrenaming)
///             case = query.operations[0].args[-1].value
///             assert issubclass(case, py.builtin.BaseException)
///             lno, lyes = query.exits
///             assert lno.exitcase == False and lyes.exitcase == True
///             if case not in seen:
///                 is_covered = False
///                 for cov in covered:
///                     if issubclass(case, cov):
///                         is_covered = True
///                         break
///                 if not is_covered:
///                     switches.append( (case, lyes) )
///                 seen.append(case)
///             exc = lno
///             query = exc.target
///         if Exception not in seen:
///             switches.append( (Exception, exc) )
///         exits = []
///         for case, oldlink in switches:
///             link = oldlink.replace(renaming)
///             link.last_exception = last_exception
///             link.last_exc_value = last_exc_value
///             renaming2 = {}
///             for v in link.getextravars():
///                 renaming2[v] = Variable(v)
///             link = link.replace(renaming2)
///             link.exitcase = case
///             exits.append(link)
///         block.recloseblock(*(preserve + exits))
/// ```
pub fn simplify_exceptions(graph: &FunctionGraph) {
    let exception_class = HOST_ENV
        .lookup_builtin("Exception")
        .expect("HOST_ENV missing Exception");
    let mut renaming: HashMap<Variable, Hlvalue> = HashMap::new();

    for block in graph.iterblocks() {
        // upstream: `if not (block.canraise and block.exits[-1].exitcase is Exception): continue`.
        let (canraise, last_is_exception, exits_snapshot) = {
            let b = block.borrow();
            let canraise = b.canraise();
            let last_is_exception = b
                .exits
                .last()
                .and_then(|l| l.borrow().exitcase.clone())
                .map(|c| match c {
                    Hlvalue::Constant(Constant {
                        value: ConstValue::HostObject(h),
                        ..
                    }) => h == exception_class,
                    _ => false,
                })
                .unwrap_or(false);
            let exits_snapshot: Vec<LinkRef> = b.exits.iter().cloned().collect();
            (canraise, last_is_exception, exits_snapshot)
        };
        if !(canraise && last_is_exception) {
            continue;
        }

        // upstream: `covered = [link.exitcase for link in block.exits[1:-1]]`.
        let covered: Vec<Option<Hlvalue>> = exits_snapshot
            .iter()
            .skip(1)
            .take(exits_snapshot.len().saturating_sub(2))
            .map(|l| l.borrow().exitcase.clone())
            .collect();
        // upstream: `preserve = list(block.exits[:-1])`.
        let preserve: Vec<LinkRef> = exits_snapshot[..exits_snapshot.len().saturating_sub(1)]
            .iter()
            .cloned()
            .collect();
        let mut seen: Vec<Hlvalue> = Vec::new();
        let mut switches: Vec<(Hlvalue, LinkRef)> = Vec::new();

        // upstream: `exc = block.exits[-1]; ... query = exc.target`.
        let mut exc: LinkRef = exits_snapshot
            .last()
            .cloned()
            .expect("canraise block has ≥2 exits");
        let last_exception = exc.borrow().last_exception.clone();
        let last_exc_value = exc.borrow().last_exc_value.clone();
        let mut query: Option<BlockRef> = exc.borrow().target.clone();

        // upstream: `while len(query.exits) == 2:`.
        while let Some(q) = query.clone() {
            if q.borrow().exits.len() != 2 {
                break;
            }
            // upstream: `newrenaming[ltarg] = lprev.replace(renaming)`.
            let mut newrenaming: HashMap<Variable, Hlvalue> = HashMap::new();
            let exc_args: Vec<Option<Hlvalue>> = exc.borrow().args.clone();
            let q_inputargs = q.borrow().inputargs.clone();
            for (lprev_opt, ltarg) in exc_args.iter().zip(q_inputargs.iter()) {
                let Some(lprev) = lprev_opt else { continue };
                let Hlvalue::Variable(tgt_v) = ltarg else {
                    continue;
                };
                let replaced = rename_hl(lprev, &renaming);
                newrenaming.insert(tgt_v.clone(), replaced);
            }
            // upstream: `op = query.operations[0]`.
            let q_ops = q.borrow().operations.clone();
            let Some(op) = q_ops.first() else { break };
            // upstream: `if not (op.opname in ("is_", "issubtype") and
            //              op.args[0].replace(newrenaming) == last_exception): break`.
            if !(op.opname == "is_" || op.opname == "issubtype") {
                break;
            }
            let Some(last_exc_hl) = last_exception.as_ref() else {
                break;
            };
            let Some(op_arg0) = op.args.first() else {
                break;
            };
            if rename_hl(op_arg0, &newrenaming) != *last_exc_hl {
                break;
            }
            // upstream: `renaming.update(newrenaming)`.
            for (k, v) in newrenaming.into_iter() {
                renaming.insert(k, v);
            }
            // upstream: `case = query.operations[0].args[-1].value`.
            let Some(case_arg) = op.args.last() else {
                break;
            };
            let case_hl = case_arg.clone();
            // upstream: `assert issubclass(case, BaseException)` — debug.
            debug_assert!(match &case_hl {
                Hlvalue::Constant(Constant {
                    value: ConstValue::HostObject(h),
                    ..
                }) => h.is_class(),
                _ => false,
            });
            // upstream: `lno, lyes = query.exits; assert lno.exitcase == False and
            // lyes.exitcase == True`. Our boolean switch invariant
            // (checkgraph) maps exits[0] → False, exits[1] → True.
            let q_exits = q.borrow().exits.clone();
            let Some(lno) = q_exits.first().cloned() else {
                break;
            };
            let Some(lyes) = q_exits.get(1).cloned() else {
                break;
            };
            // upstream: `if case not in seen: ... switches.append((case, lyes))`.
            if !seen.iter().any(|s| *s == case_hl) {
                let is_covered = covered.iter().any(|cov_opt| match cov_opt {
                    Some(cov) => is_exitcase_subclass(&case_hl, cov),
                    None => false,
                });
                if !is_covered {
                    switches.push((case_hl.clone(), lyes));
                }
                seen.push(case_hl);
            }
            // upstream: `exc = lno; query = exc.target`.
            exc = lno;
            query = exc.borrow().target.clone();
        }
        // upstream: `if Exception not in seen: switches.append((Exception, exc))`.
        let exception_hlvalue = Hlvalue::Constant(Constant::new(ConstValue::HostObject(
            exception_class.clone(),
        )));
        if !seen.iter().any(|s| *s == exception_hlvalue) {
            switches.push((exception_hlvalue.clone(), exc));
        }

        // upstream: `exits = []; for case, oldlink in switches:
        //   link = oldlink.replace(renaming); ... link.exitcase = case; exits.append(link)`.
        let mut new_exits: Vec<LinkRef> = Vec::new();
        for (case_hl, oldlink) in switches {
            let link = rename_link_args(&oldlink, &renaming);
            {
                let mut lb = link.borrow_mut();
                lb.last_exception = last_exception.clone();
                lb.last_exc_value = last_exc_value.clone();
            }
            // upstream: `renaming2[v] = Variable(v)` for each extra
            // var — fresh identity, upstream Variable(v) copies prefix
            // via rename; our Variable::copy does the same.
            let mut renaming2: HashMap<Variable, Hlvalue> = HashMap::new();
            {
                let lb = link.borrow();
                if let Some(Hlvalue::Variable(v)) = &lb.last_exception {
                    renaming2.insert(v.clone(), Hlvalue::Variable(v.copy()));
                }
                if let Some(Hlvalue::Variable(v)) = &lb.last_exc_value {
                    renaming2.insert(v.clone(), Hlvalue::Variable(v.copy()));
                }
            }
            // Apply renaming2 in place on the fresh link's args, etc.
            let link2 = rename_link_args(&link, &renaming2);
            link2.borrow_mut().exitcase = Some(case_hl);
            new_exits.push(link2);
        }
        let mut combined: Vec<LinkRef> = preserve;
        combined.extend(new_exits);
        block.recloseblock(combined);
    }
}

/// RPython `coalesce_bool(graph)` (simplify.py:656-699).
///
/// Collapses the two-step pattern
/// ```text
/// block: bool(v); exitswitch=result_v
///   ├─False→ block2: bool(result_v_again); exitswitch=...
///   └─True→  block2_same
/// ```
/// into a direct jump from `block` to `block2`'s true/false targets.
/// Upstream walks candidate bool-ended blocks, rewrites each
/// `bool`-exitcase-keyed exit to point at `block2`'s outgoing arm, and
/// repeats until a fixed point.
pub fn coalesce_bool(graph: &FunctionGraph) {
    // upstream: list of (block, [(case, target_block)]) candidate tuples.
    let mut candidates: Vec<(BlockRef, Vec<(bool, BlockRef)>)> = Vec::new();
    for block in graph.iterblocks() {
        let is_bool = {
            let b = block.borrow();
            b.operations
                .last()
                .map(|op| op.opname == "bool")
                .unwrap_or(false)
        };
        if !is_bool {
            continue;
        }
        let tgts = has_bool_exitpath(&block);
        if !tgts.is_empty() {
            candidates.push((block, tgts));
        }
    }

    while let Some((cand, tgts)) = candidates.pop() {
        let cand_exits_snapshot: Vec<LinkRef> = cand.borrow().exits.iter().cloned().collect();
        let mut new_exits: Vec<LinkRef> = cand_exits_snapshot.clone();

        for (case_bool, tgt) in tgts {
            // upstream: `exit = cand.exits[case]` where `case` is bool
            // (Python bool subclasses int → cand.exits[0/1]).
            let idx = if case_bool { 1 } else { 0 };
            let Some(exit_link) = cand_exits_snapshot.get(idx).cloned() else {
                continue;
            };
            // upstream: `rrenaming = dict(zip(tgt.inputargs, exit.args));
            //  rrenaming[tgt.operations[0].result] = cand.operations[-1].result`.
            let mut rrenaming: HashMap<Variable, Hlvalue> = HashMap::new();
            let tgt_inputargs = tgt.borrow().inputargs.clone();
            let exit_args: Vec<Option<Hlvalue>> = exit_link.borrow().args.clone();
            for (ltarg, lprev_opt) in tgt_inputargs.iter().zip(exit_args.iter()) {
                let Hlvalue::Variable(tgt_v) = ltarg else {
                    continue;
                };
                let Some(lprev) = lprev_opt else { continue };
                rrenaming.insert(tgt_v.clone(), lprev.clone());
            }
            let tgt_first_result = {
                let b = tgt.borrow();
                b.operations
                    .first()
                    .map(|op| op.result.clone())
                    .and_then(|r| match r {
                        Hlvalue::Variable(v) => Some(v),
                        _ => None,
                    })
            };
            let cand_last_result = {
                let b = cand.borrow();
                b.operations.last().map(|op| op.result.clone())
            };
            if let (Some(tgt_first), Some(cand_last)) = (tgt_first_result, cand_last_result) {
                rrenaming.insert(tgt_first, cand_last);
            }
            // upstream: `newlink = tgt.exits[case].copy(rename)`.
            let tgt_exits = tgt.borrow().exits.clone();
            let Some(tgt_exit) = tgt_exits.get(idx).cloned() else {
                continue;
            };
            let newlink = rename_link_args(&tgt_exit, &rrenaming);
            new_exits[idx] = newlink;
        }
        cand.recloseblock(new_exits);
        // upstream: retry against the same cand in case another bool-
        // chain opened up.
        let again = has_bool_exitpath(&cand);
        if !again.is_empty() {
            candidates.push((cand, again));
        }
    }
}

/// Helper for `coalesce_bool` — mirrors upstream's nested
/// `has_bool_exitpath(block)` closure (simplify.py:663-677).
fn has_bool_exitpath(block: &BlockRef) -> Vec<(bool, BlockRef)> {
    let mut tgts: Vec<(bool, BlockRef)> = Vec::new();
    let b = block.borrow();
    let Some(start_op) = b.operations.last() else {
        return tgts;
    };
    let Some(cond_v) = start_op.args.first().cloned() else {
        return tgts;
    };
    if b.exitswitch.as_ref() != Some(&start_op.result) {
        return tgts;
    }
    for exit in &b.exits {
        let exit_b = exit.borrow();
        let Some(tgt) = exit_b.target.clone() else {
            continue;
        };
        // upstream: `if tgt == block: continue`.
        if std::rc::Rc::ptr_eq(&tgt, block) {
            continue;
        }
        let tgt_b = tgt.borrow();
        if tgt_b.operations.len() != 1 || tgt_b.operations[0].opname != "bool" {
            continue;
        }
        let tgt_op = &tgt_b.operations[0];
        if tgt_b.exitswitch.as_ref() != Some(&tgt_op.result) {
            continue;
        }
        // upstream: `rrenaming.get(tgt_op.args[0]) == cond_v`.
        let Some(tgt_op_arg0) = tgt_op.args.first() else {
            continue;
        };
        let Hlvalue::Variable(arg_var) = tgt_op_arg0 else {
            continue;
        };
        // Build rrenaming just enough to check `arg_var → cond_v`.
        let position = tgt_b
            .inputargs
            .iter()
            .position(|ia| matches!(ia, Hlvalue::Variable(v) if v == arg_var));
        let Some(pos) = position else { continue };
        let Some(link_arg) = exit_b.args.get(pos).cloned().flatten() else {
            continue;
        };
        if link_arg != cond_v {
            continue;
        }
        let case_bool = match &exit_b.exitcase {
            Some(Hlvalue::Constant(Constant {
                value: ConstValue::Bool(b),
                ..
            })) => *b,
            _ => continue,
        };
        drop(tgt_b);
        tgts.push((case_bool, tgt));
    }
    tgts
}

/// RPython `transform_xxxitem(graph)` (simplify.py:172-186).
///
/// ```python
/// def transform_xxxitem(graph):
///     # xxx setitem too
///     for block in graph.iterblocks():
///         if block.canraise:
///             last_op = block.raising_op
///             if last_op.opname == 'getitem':
///                 postfx = []
///                 for exit in block.exits:
///                     if exit.exitcase is IndexError:
///                         postfx.append('idx')
///                 if postfx:
///                     Op = getattr(op, '_'.join(['getitem'] + postfx))
///                     newop = Op(*last_op.args)
///                     newop.result = last_op.result
///                     block.operations[-1] = newop
/// ```
///
/// `op.getitem_idx` is the sole descendant: when `IndexError` appears
/// in a `getitem` block's exit cases, we rewrite the raising op to
/// `getitem_idx`. Upstream's open-ended postfx list (`['idx', 'key']`
/// etc.) does not materialise — `op.getitem_idx_key` does exist but
/// the pass only builds an `'idx'` postfx.
pub fn transform_xxxitem(graph: &FunctionGraph) {
    let index_error = HOST_ENV
        .lookup_builtin("IndexError")
        .expect("HOST_ENV missing IndexError");
    for block in graph.iterblocks() {
        let (canraise, is_getitem) = {
            let b = block.borrow();
            let canraise = b.canraise();
            let is_getitem = canraise
                && b.raising_op()
                    .map(|op| op.opname == "getitem")
                    .unwrap_or(false);
            (canraise, is_getitem)
        };
        if !canraise || !is_getitem {
            continue;
        }
        // upstream: `for exit in block.exits: if exit.exitcase is IndexError: postfx.append('idx')`.
        let has_idx = {
            let b = block.borrow();
            b.exits.iter().any(|link| match &link.borrow().exitcase {
                Some(Hlvalue::Constant(Constant {
                    value: ConstValue::HostObject(h),
                    ..
                })) => h == &index_error,
                _ => false,
            })
        };
        if !has_idx {
            continue;
        }
        // upstream: `newop = Op(*last_op.args); newop.result = last_op.result;
        // block.operations[-1] = newop`. The Rust port rewrites opname
        // in place since SpaceOperation is already the concrete carrier.
        let mut b = block.borrow_mut();
        if let Some(op) = b.operations.last_mut() {
            op.opname = "getitem_idx".to_string();
        }
    }
}

/// RPython `remove_dead_exceptions(graph)` (simplify.py:189-216).
///
/// ```python
/// def remove_dead_exceptions(graph):
///     """Exceptions can be removed if they are unreachable"""
///     def issubclassofmember(cls, seq):
///         for member in seq:
///             if member and issubclass(cls, member):
///                 return True
///         return False
///     for block in list(graph.iterblocks()):
///         if not block.canraise:
///             continue
///         exits = []
///         seen = []
///         for link in block.exits:
///             case = link.exitcase
///             if issubclassofmember(case, seen):
///                 continue
///             while len(exits) > 1:
///                 prev = exits[-1]
///                 if not (issubclass(prev.exitcase, link.exitcase) and
///                     prev.target is link.target and prev.args == link.args):
///                     break
///                 exits.pop()
///             exits.append(link)
///             seen.append(case)
///         block.recloseblock(*exits)
/// ```
pub fn remove_dead_exceptions(graph: &FunctionGraph) {
    // upstream: `for block in list(graph.iterblocks())` — snapshot.
    for block in graph.iterblocks() {
        if !block.borrow().canraise() {
            continue;
        }
        let exits_snapshot: Vec<LinkRef> = block.borrow().exits.iter().cloned().collect();
        let mut new_exits: Vec<LinkRef> = Vec::new();
        let mut seen: Vec<Option<Hlvalue>> = Vec::new();

        for link in exits_snapshot {
            let case = link.borrow().exitcase.clone();

            // upstream: `if issubclassofmember(case, seen): continue`.
            if issubclassofmember(case.as_ref(), &seen) {
                continue;
            }

            // upstream: merge the previous case if it's a subclass of
            // the current one and the link shape matches.
            while new_exits.len() > 1 {
                let prev = new_exits.last().expect("len > 1 ⇒ non-empty");
                let (subclass_ok, target_same, args_same) = {
                    let prev_b = prev.borrow();
                    let link_b = link.borrow();
                    let subclass_ok = match (&prev_b.exitcase, &link_b.exitcase) {
                        (Some(a), Some(b)) => is_exitcase_subclass(a, b),
                        _ => false,
                    };
                    let target_same = match (prev_b.target.as_ref(), link_b.target.as_ref()) {
                        (Some(a), Some(b)) => std::rc::Rc::ptr_eq(a, b),
                        _ => false,
                    };
                    let args_same = prev_b.args == link_b.args;
                    (subclass_ok, target_same, args_same)
                };
                if !(subclass_ok && target_same && args_same) {
                    break;
                }
                new_exits.pop();
            }
            new_exits.push(link);
            seen.push(case);
        }
        block.recloseblock(new_exits);
    }
}

fn issubclassofmember(cls: Option<&Hlvalue>, seq: &[Option<Hlvalue>]) -> bool {
    let Some(cls) = cls else {
        return false;
    };
    for member in seq {
        // upstream: `if member and issubclass(cls, member): return True`.
        let Some(member) = member else { continue };
        if is_exitcase_subclass(cls, member) {
            return true;
        }
    }
    false
}

/// `issubclass(exitcase_cls, other_cls)` over Hlvalue-wrapped Constant
/// HostObjects. Returns `false` for any non-class payload (upstream's
/// Python `issubclass` raises TypeError on non-classes, but upstream
/// only ever calls this with exception classes).
fn is_exitcase_subclass(cls: &Hlvalue, other: &Hlvalue) -> bool {
    match (cls, other) {
        (
            Hlvalue::Constant(Constant {
                value: ConstValue::HostObject(a),
                ..
            }),
            Hlvalue::Constant(Constant {
                value: ConstValue::HostObject(b),
                ..
            }),
        ) => a.is_subclass_of(b),
        _ => false,
    }
}

/// RPython `remove_assertion_errors(graph)` (simplify.py:321-346).
///
/// ```python
/// def remove_assertion_errors(graph):
///     """Remove branches that go directly to raising an AssertionError,
///     assuming that AssertionError shouldn't occur at run-time.  Note that
///     this is how implicit exceptions are removed (see _implicit_ in
///     flowcontext.py).
///     """
///     for block in list(graph.iterblocks()):
///         for i in range(len(block.exits)-1, -1, -1):
///             exit = block.exits[i]
///             if not (exit.target is graph.exceptblock and
///                     exit.args[0] == Constant(AssertionError)):
///                 continue
///             if len(block.exits) < 2:
///                 break
///             if block.canraise:
///                 if exit.exitcase is None:
///                     break
///                 if len(block.exits) == 2:
///                     block.exitswitch = None
///                     exit.exitcase = None
///             lst = list(block.exits)
///             del lst[i]
///             block.recloseblock(*lst)
/// ```
pub fn remove_assertion_errors(graph: &FunctionGraph) {
    // upstream: `for block in list(graph.iterblocks())` — snapshot
    // blocks so recloseblock mid-iteration is safe.
    let assert_err_class: HostObject = HOST_ENV
        .lookup_builtin("AssertionError")
        .expect("HOST_ENV missing AssertionError");
    let exceptblock_key = BlockKey::of(&graph.exceptblock);

    for block in graph.iterblocks() {
        let mut i = block.borrow().exits.len();
        while i > 0 {
            i -= 1;
            let (targets_except, args_is_assert_err, canraise, exitcase_none, exits_len) = {
                let b = block.borrow();
                let Some(exit) = b.exits.get(i) else {
                    break;
                };
                let exit_b = exit.borrow();
                let targets_except = exit_b
                    .target
                    .as_ref()
                    .map(|t| BlockKey::of(t) == exceptblock_key)
                    .unwrap_or(false);
                let args_first = exit_b.args.first().cloned().flatten();
                let args_is_assert_err = matches!(
                    args_first,
                    Some(Hlvalue::Constant(Constant {
                        value: ConstValue::HostObject(ref h),
                        ..
                    })) if h == &assert_err_class
                );
                (
                    targets_except,
                    args_is_assert_err,
                    b.canraise(),
                    exit_b.exitcase.is_none(),
                    b.exits.len(),
                )
            };

            // upstream: `if not (exit.target is graph.exceptblock and
            // exit.args[0] == Constant(AssertionError)): continue`.
            if !(targets_except && args_is_assert_err) {
                continue;
            }
            // upstream: `if len(block.exits) < 2: break`.
            if exits_len < 2 {
                break;
            }
            if canraise {
                // upstream: `if exit.exitcase is None: break`.
                if exitcase_none {
                    break;
                }
                // upstream: `if len(block.exits) == 2: block.exitswitch
                // = None; exit.exitcase = None` — the surviving exit
                // (the non-i one) gets promoted to an unconditional
                // link below once we delete exit i.
                if exits_len == 2 {
                    // upstream mutates the `exit` being removed to
                    // have `exitcase = None`, but since we drop it in
                    // the same block this is effectively no-op for
                    // the outcome. Still mirror the mutation for
                    // parity — a reader reconstructing upstream
                    // semantics shouldn't be surprised.
                    block.borrow_mut().exitswitch = None;
                    if let Some(e) = block.borrow().exits.get(i) {
                        e.borrow_mut().exitcase = None;
                    }
                }
            }
            // upstream: `lst = list(block.exits); del lst[i];
            // block.recloseblock(*lst)`.
            let mut lst: Vec<LinkRef> = block.borrow().exits.iter().cloned().collect();
            lst.remove(i);
            block.recloseblock(lst);
        }
    }
}

/// RPython `transform_ovfcheck(graph)` (simplify.py:71-108).
///
/// Rewrites `simple_call(ovfcheck, result_of_prev_op)` into the
/// `_ovf` variant of the previous operation. When `ovfcheck` sits at
/// the start of its block, upstream collapses the predecessor link
/// first via `join_blocks` and re-enters; the Rust port mirrors that
/// fixpoint loop.
pub fn transform_ovfcheck(graph: &FunctionGraph) {
    let ovfcheck_sentinel: HostObject = HOST_ENV
        .lookup_builtin("ovfcheck")
        .expect("HOST_ENV missing ovfcheck sentinel");

    loop {
        let mut any_block_needs_merge = false;

        for block in graph.iterblocks() {
            // upstream: `for i in range(len(block.operations)-1, -1, -1)`.
            let ops_len = block.borrow().operations.len();
            let mut i = ops_len;
            while i > 0 {
                i -= 1;
                let (is_ovfcheck_call, result_of_call) = {
                    let b = block.borrow();
                    let op = &b.operations[i];
                    let matches_ovfcheck = op.opname == "simple_call"
                        && matches!(
                            op.args.first(),
                            Some(Hlvalue::Constant(Constant {
                                value: ConstValue::HostObject(h),
                                ..
                            })) if h == &ovfcheck_sentinel
                        );
                    (matches_ovfcheck, op.result.clone())
                };
                if !is_ovfcheck_call {
                    continue;
                }

                // upstream: hard case — `ovfcheck` at block start.
                if i == 0 {
                    let entrymap = mkentrymap(graph);
                    let Some(links) = entrymap.get(&BlockKey::of(&block)) else {
                        break;
                    };
                    assert_eq!(
                        links.len(),
                        1,
                        "ovfcheck at block start requires single entry"
                    );
                    let Some(prevblock) = links[0]
                        .borrow()
                        .prevblock
                        .as_ref()
                        .and_then(|w| w.upgrade())
                    else {
                        break;
                    };
                    {
                        let first_exit = prevblock.borrow().exits.first().cloned();
                        if let Some(fe) = first_exit {
                            assert!(matches!(
                                fe.borrow().target.as_ref(),
                                Some(t) if std::rc::Rc::ptr_eq(t, &block)
                            ));
                        }
                    }
                    prevblock.borrow_mut().exitswitch = None;
                    prevblock.recloseblock(vec![links[0].clone()]);
                    any_block_needs_merge = true;
                    break;
                }

                // upstream: `op1 = block.operations[i - 1]`. Must be
                // overflow-capable.
                let (op1_opname, op1_result) = {
                    let b = block.borrow();
                    let op1 = &b.operations[i - 1];
                    (op1.opname.clone(), op1.result.clone())
                };
                let kind = crate::flowspace::operation::OpKind::from_opname(&op1_opname)
                    .unwrap_or_else(|| {
                        panic!("ovfcheck on unknown opname {op1_opname} in {}", graph.name)
                    });
                let ovf_kind = kind.ovf_variant().unwrap_or_else(|| {
                    panic!(
                        "ovfcheck in {}: Operation {op1_opname} has no overflow variant",
                        graph.name
                    )
                });
                // upstream: `op1_ovf = op1.ovfchecked()`; we just flip
                // opname in place — args/result/offset already match.
                {
                    let mut b = block.borrow_mut();
                    b.operations[i - 1].opname = ovf_kind.opname().to_string();
                    b.operations.remove(i);
                }
                let Hlvalue::Variable(result_v) = result_of_call else {
                    continue;
                };
                let mut renaming: HashMap<Variable, Hlvalue> = HashMap::new();
                renaming.insert(result_v, op1_result);
                renamevariables_hl(&block, &renaming);
            }
        }

        if !any_block_needs_merge {
            break;
        }
        // upstream: after merging, re-run join_blocks + retry.
        join_blocks(graph);
    }
}

/// RPython `all_passes = [...]` (simplify.py:1060-1073) + `simplify_graph`
/// (simplify.py:1075-1081).
///
/// `all_passes` upstream is a list of callables; the Rust port uses a
/// `fn(&FunctionGraph)` slice so every entry has the same call shape.
/// `transform_dead_op_vars(graph, translator=None)` and
/// `SSA_to_SSI(graph, annotator=None)` are wrapped to match — both
/// accept `translator=None` / `annotator=None` by default and the
/// Rust wrappers pass `None`.
pub fn all_passes() -> &'static [fn(&FunctionGraph)] {
    &[
        dead_op_vars_shim,
        eliminate_empty_blocks,
        remove_assertion_errors,
        remove_identical_vars_ssa,
        constfold_exitswitch,
        remove_trivial_links,
        crate::translator::backendopt::ssa::ssa_to_ssi,
        coalesce_bool,
        transform_ovfcheck,
        simplify_exceptions,
        transform_xxxitem,
        remove_dead_exceptions,
    ]
}

fn dead_op_vars_shim(graph: &FunctionGraph) {
    transform_dead_op_vars(graph, None);
}

/// RPython `simplify_graph(graph, passes=True)` (simplify.py:1075-1081).
///
/// ```python
/// def simplify_graph(graph, passes=True):
///     if passes is True:
///         passes = all_passes
///     for pass_ in passes:
///         pass_(graph)
///     checkgraph(graph)
/// ```
pub fn simplify_graph(graph: &FunctionGraph, passes: Option<&[fn(&FunctionGraph)]>) {
    let default_passes = all_passes();
    let passes = passes.unwrap_or(default_passes);
    for pass_ in passes {
        pass_(graph);
    }
    checkgraph(graph);
}

/// RPython `cleanup_graph(graph)` (simplify.py:1083-1088).
///
/// ```python
/// def cleanup_graph(graph):
///     checkgraph(graph)
///     eliminate_empty_blocks(graph)
///     join_blocks(graph)
///     remove_identical_vars(graph)
///     checkgraph(graph)
/// ```
pub fn cleanup_graph(graph: &FunctionGraph) {
    checkgraph(graph);
    eliminate_empty_blocks(graph);
    join_blocks(graph);
    remove_identical_vars(graph);
    checkgraph(graph);
}

/// RPython `constfold_exitswitch(graph)` (simplify.py:218-239).
///
/// ```python
/// def constfold_exitswitch(graph):
///     block = graph.startblock
///     seen = set([block])
///     stack = list(block.exits)
///     while stack:
///         link = stack.pop()
///         target = link.target
///         if target in seen:
///             continue
///         source = link.prevblock
///         switch = source.exitswitch
///         if (isinstance(switch, Constant) and not source.canraise):
///             exits = replace_exitswitch_by_constant(source, switch)
///             stack.extend(exits)
///         else:
///             seen.add(target)
///             stack.extend(target.exits)
/// ```
pub fn constfold_exitswitch(graph: &FunctionGraph) {
    let mut seen: HashSet<BlockKey> = HashSet::new();
    seen.insert(BlockKey::of(&graph.startblock));
    let mut stack: Vec<LinkRef> = graph.startblock.borrow().exits.iter().cloned().collect();

    while let Some(link) = stack.pop() {
        let (prev_rc, target_rc) = {
            let l = link.borrow();
            (
                l.prevblock.as_ref().and_then(|w| w.upgrade()),
                l.target.clone(),
            )
        };
        let Some(target_rc) = target_rc else {
            continue;
        };
        if seen.contains(&BlockKey::of(&target_rc)) {
            continue;
        }
        let Some(prev_rc) = prev_rc else {
            continue;
        };

        // upstream: `switch = source.exitswitch`.
        let (is_const_switch, const_val, is_canraise) = {
            let b = prev_rc.borrow();
            match &b.exitswitch {
                Some(Hlvalue::Constant(c)) => (true, Some(c.clone()), b.canraise()),
                _ => (false, None, b.canraise()),
            }
        };

        if is_const_switch && !is_canraise {
            let const_val = const_val.expect("const_val set when is_const_switch");
            let new_exits = replace_exitswitch_by_constant(&prev_rc, &const_val);
            stack.extend(new_exits);
        } else {
            seen.insert(BlockKey::of(&target_rc));
            let more: Vec<LinkRef> = target_rc.borrow().exits.iter().cloned().collect();
            stack.extend(more);
        }
    }
}

/// RPython `remove_trivial_links(graph)` (simplify.py:242-268).
///
/// ```python
/// def remove_trivial_links(graph):
///     """Remove trivial links by merging their source and target blocks
///
///     A link is trivial if it has no arguments, is the single exit of its
///     source and the single parent of its target.
///     """
///     entrymap = mkentrymap(graph)
///     block = graph.startblock
///     seen = set([block])
///     stack = list(block.exits)
///     while stack:
///         link = stack.pop()
///         if link.target in seen:
///             continue
///         source = link.prevblock
///         target = link.target
///         if (not link.args and source.exitswitch is None and
///                 len(entrymap[target]) == 1 and
///                 target.exits):  # stop at the returnblock
///             assert len(source.exits) == 1
///             source.operations.extend(target.operations)
///             source.exitswitch = newexitswitch = target.exitswitch
///             source.recloseblock(*target.exits)
///             stack.extend(source.exits)
///         else:
///             seen.add(target)
///             stack.extend(target.exits)
/// ```
pub fn remove_trivial_links(graph: &FunctionGraph) {
    let entrymap = mkentrymap(graph);
    let mut seen: HashSet<BlockKey> = HashSet::new();
    seen.insert(BlockKey::of(&graph.startblock));
    let mut stack: Vec<LinkRef> = graph.startblock.borrow().exits.iter().cloned().collect();

    while let Some(link) = stack.pop() {
        let (prev_rc, target_rc, link_args_empty) = {
            let l = link.borrow();
            let prev = l.prevblock.as_ref().and_then(|w| w.upgrade());
            let target = l.target.clone();
            // upstream `not link.args` — treat `Vec<Option<Hlvalue>>` as
            // empty when there are no args at all. Transient-merge
            // `None` slots are valid args in upstream semantics.
            let args_empty = l.args.is_empty();
            (prev, target, args_empty)
        };
        let Some(target_rc) = target_rc else {
            continue;
        };
        if seen.contains(&BlockKey::of(&target_rc)) {
            continue;
        }
        let Some(prev_rc) = prev_rc else {
            continue;
        };

        let source_switch_none = prev_rc.borrow().exitswitch.is_none();
        let target_entry_count = entrymap
            .get(&BlockKey::of(&target_rc))
            .map(Vec::len)
            .unwrap_or(0);
        let target_has_exits = !target_rc.borrow().exits.is_empty();

        let is_trivial =
            link_args_empty && source_switch_none && target_entry_count == 1 && target_has_exits;

        if is_trivial {
            assert_eq!(
                prev_rc.borrow().exits.len(),
                1,
                "remove_trivial_links: source block has exitswitch=None but != 1 exit"
            );
            // upstream: `source.operations.extend(target.operations)`.
            let target_ops = target_rc.borrow().operations.clone();
            prev_rc.borrow_mut().operations.extend(target_ops);
            // upstream: `source.exitswitch = target.exitswitch`.
            let new_switch = target_rc.borrow().exitswitch.clone();
            prev_rc.borrow_mut().exitswitch = new_switch;
            // upstream: `source.recloseblock(*target.exits)`.
            let target_exits: Vec<LinkRef> = target_rc.borrow().exits.iter().cloned().collect();
            prev_rc.recloseblock(target_exits);
            let source_exits: Vec<LinkRef> = prev_rc.borrow().exits.iter().cloned().collect();
            stack.extend(source_exits);
        } else {
            seen.insert(BlockKey::of(&target_rc));
            let more: Vec<LinkRef> = target_rc.borrow().exits.iter().cloned().collect();
            stack.extend(more);
        }
    }
}

/// RPython `join_blocks(graph)` (simplify.py:271-319).
///
/// ```python
/// def join_blocks(graph):
///     """Links can be deleted if they are the single exit of a block and
///     the single entry point of the next block.  When this happens, we can
///     append all the operations of the following block to the preceeding
///     block (but renaming variables with the appropriate arguments.)
///     """
///     entrymap = mkentrymap(graph)
///     block = graph.startblock
///     seen = {block: True}
///     stack = list(block.exits)
///     while stack:
///         link = stack.pop()
///         if (link.prevblock.exitswitch is None and
///             len(entrymap[link.target]) == 1 and
///             link.target.exits):  # stop at the returnblock
///             assert len(link.prevblock.exits) == 1
///             renaming = {}
///             for vprev, vtarg in zip(link.args, link.target.inputargs):
///                 renaming[vtarg] = vprev
///             def rename_op(op):
///                 op = op.replace(renaming)
///                 # special case...
///                 if op.opname == 'indirect_call':
///                     if isinstance(op.args[0], Constant):
///                         assert isinstance(op.args[-1], Constant)
///                         del op.args[-1]
///                         op.opname = 'direct_call'
///                 return op
///             for op in link.target.operations:
///                 link.prevblock.operations.append(rename_op(op))
///             exits = []
///             for exit in link.target.exits:
///                 newexit = exit.replace(renaming)
///                 exits.append(newexit)
///             if link.target.exitswitch:
///                 newexitswitch = link.target.exitswitch.replace(renaming)
///             else:
///                 newexitswitch = None
///             link.prevblock.exitswitch = newexitswitch
///             link.prevblock.recloseblock(*exits)
///             if (isinstance(newexitswitch, Constant) and
///                     not link.prevblock.canraise):
///                 exits = replace_exitswitch_by_constant(link.prevblock,
///                                                        newexitswitch)
///             stack.extend(exits)
///         else:
///             if link.target not in seen:
///                 stack.extend(link.target.exits)
///                 seen[link.target] = True
/// ```
pub fn join_blocks(graph: &FunctionGraph) {
    let entrymap = mkentrymap(graph);
    let mut seen: HashSet<BlockKey> = HashSet::new();
    seen.insert(BlockKey::of(&graph.startblock));

    // upstream: `stack = list(block.exits)`.
    let mut stack: Vec<LinkRef> = graph.startblock.borrow().exits.iter().cloned().collect();

    while let Some(link) = stack.pop() {
        // Snapshot the fields we need before re-borrowing the Link
        // itself (Rust borrow-checker keeps mutation and reads split).
        let (prev_rc, target_rc, target_inputargs, link_args_all_some, link_args) = {
            let l = link.borrow();
            let prev = l.prevblock.as_ref().and_then(|w| w.upgrade());
            let target = l.target.clone();
            let target_inputargs = target
                .as_ref()
                .map(|t| t.borrow().inputargs.clone())
                .unwrap_or_default();
            let args_all_some = l.args.iter().all(|a| a.is_some());
            let args: Vec<Hlvalue> = l.args.iter().filter_map(|a| a.clone()).collect();
            (prev, target, target_inputargs, args_all_some, args)
        };
        let Some(prev_rc) = prev_rc else {
            continue;
        };
        let Some(target_rc) = target_rc else {
            continue;
        };

        let prev_switch_is_none = prev_rc.borrow().exitswitch.is_none();
        let target_entry_count = entrymap
            .get(&BlockKey::of(&target_rc))
            .map(Vec::len)
            .unwrap_or(0);
        let target_has_exits = !target_rc.borrow().exits.is_empty();

        // upstream: `link.prevblock.exitswitch is None and
        // len(entrymap[link.target]) == 1 and link.target.exits`.
        let can_join = prev_switch_is_none
            && target_entry_count == 1
            && target_has_exits
            && link_args_all_some;

        if can_join {
            assert_eq!(
                prev_rc.borrow().exits.len(),
                1,
                "join_blocks: prevblock has exitswitch=None but != 1 exit"
            );

            // upstream: `renaming[vtarg] = vprev` for each pair.
            let mut renaming: HashMap<Variable, Hlvalue> = HashMap::new();
            for (vprev, vtarg) in link_args.iter().zip(target_inputargs.iter()) {
                if let Hlvalue::Variable(tgt) = vtarg {
                    renaming.insert(tgt.clone(), vprev.clone());
                }
            }

            // upstream: append renamed target.operations to prevblock.
            let target_ops: Vec<SpaceOperation> = target_rc.borrow().operations.clone();
            for op in &target_ops {
                prev_rc
                    .borrow_mut()
                    .operations
                    .push(rename_op(op, &renaming));
            }

            // upstream: `exits = [exit.replace(renaming) for exit in target.exits]`.
            let target_exits: Vec<LinkRef> = target_rc.borrow().exits.iter().cloned().collect();
            let mut new_exits: Vec<LinkRef> = target_exits
                .iter()
                .map(|e| rename_link_args(e, &renaming))
                .collect();

            // upstream: `newexitswitch = target.exitswitch.replace(renaming) or None`.
            let new_exitswitch = target_rc
                .borrow()
                .exitswitch
                .as_ref()
                .map(|sw| rename_hl(sw, &renaming));
            prev_rc.borrow_mut().exitswitch = new_exitswitch.clone();
            prev_rc.recloseblock(new_exits.clone());

            // upstream: constant-fold the new switch when prevblock is
            // not a can-raise block (simplify.py:311-314).
            if let Some(Hlvalue::Constant(const_)) = &new_exitswitch {
                if !prev_rc.borrow().canraise() {
                    new_exits = replace_exitswitch_by_constant(&prev_rc, const_);
                }
            }

            stack.extend(new_exits);
        } else {
            // upstream: `if link.target not in seen: stack.extend(link.target.exits)`.
            let target_key = BlockKey::of(&target_rc);
            if !seen.contains(&target_key) {
                seen.insert(target_key);
                let more: Vec<LinkRef> = target_rc.borrow().exits.iter().cloned().collect();
                stack.extend(more);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{Block, FunctionGraph, Hlvalue, Link, Variable};
    use std::rc::Rc;

    #[test]
    fn elimination_collapses_single_empty_block() {
        // start -> empty (no ops, 1 formal arg) -> return
        // empty.exits[0].args = [empty.inputargs[0]]
        // After: start.exits[0] rewrites to point at return with the
        // substituted arg from start's link.args.
        let formal = Variable::new();
        let empty = Block::shared(vec![Hlvalue::Variable(formal.clone())]);
        let start = Block::shared(vec![]);

        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        // start -> empty with args [int(42)]
        let actual = Hlvalue::Constant(crate::flowspace::model::Constant::new(
            crate::flowspace::model::ConstValue::Int(42),
        ));
        let start_link = Link::new(vec![actual], Some(empty.clone()), None).into_ref();
        start.borrow_mut().closeblock(vec![start_link]);

        // empty -> return with args [formal]
        let empty_link = Link::new(
            vec![Hlvalue::Variable(formal.clone())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        empty.borrow_mut().closeblock(vec![empty_link]);

        eliminate_empty_blocks(&graph);

        // start -> returnblock now (empty collapsed).
        let s = start.borrow();
        let exit = s.exits[0].borrow();
        assert!(Rc::ptr_eq(exit.target.as_ref().unwrap(), &returnblock));
        // And the link args carry the substituted constant.
        assert!(matches!(
            exit.args[0].as_ref(),
            Some(Hlvalue::Constant(c)) if matches!(c.value, crate::flowspace::model::ConstValue::Int(42))
        ));
    }

    #[test]
    fn elimination_keeps_blocks_with_operations() {
        // start -> body (has an op) -> return — body is NOT empty.
        let v_in = Variable::new();
        let v_out = Variable::new();
        let body = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        body.borrow_mut()
            .operations
            .push(crate::flowspace::model::SpaceOperation::new(
                "op".to_string(),
                vec![Hlvalue::Variable(v_in.clone())],
                Hlvalue::Variable(v_out.clone()),
            ));
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        let start_link = Link::new(
            vec![Hlvalue::Variable(Variable::new())],
            Some(body.clone()),
            None,
        )
        .into_ref();
        start.borrow_mut().closeblock(vec![start_link]);

        let body_link = Link::new(
            vec![Hlvalue::Variable(v_out)],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        body.borrow_mut().closeblock(vec![body_link]);

        eliminate_empty_blocks(&graph);

        // start -> body unchanged.
        let s = start.borrow();
        let exit = s.exits[0].borrow();
        assert!(Rc::ptr_eq(exit.target.as_ref().unwrap(), &body));
    }

    #[test]
    fn join_blocks_merges_linear_chain() {
        // start -> mid -> return.
        // `mid` has one entry (from start) and non-empty exits ⇒ its
        // operations get folded into start, and start takes over mid's
        // exit to returnblock.
        let v_in = Variable::new();
        let v_out = Variable::new();
        let mid = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        mid.borrow_mut()
            .operations
            .push(crate::flowspace::model::SpaceOperation::new(
                "int_add",
                vec![
                    Hlvalue::Variable(v_in.clone()),
                    Hlvalue::Constant(crate::flowspace::model::Constant::new(
                        crate::flowspace::model::ConstValue::Int(1),
                    )),
                ],
                Hlvalue::Variable(v_out.clone()),
            ));

        let start_in = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(start_in.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        let start_link = Link::new(
            vec![Hlvalue::Variable(start_in.clone())],
            Some(mid.clone()),
            None,
        )
        .into_ref();
        start.borrow_mut().closeblock(vec![start_link.clone()]);
        start_link.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        let mid_link = Link::new(
            vec![Hlvalue::Variable(v_out.clone())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        mid.borrow_mut().closeblock(vec![mid_link.clone()]);
        mid_link.borrow_mut().prevblock = Some(Rc::downgrade(&mid));

        join_blocks(&graph);

        // `mid`'s op should now be on `start`, with v_in renamed to
        // start_in in the folded op's args.
        let s = start.borrow();
        assert_eq!(s.operations.len(), 1);
        assert_eq!(s.operations[0].opname, "int_add");
        assert!(matches!(
            &s.operations[0].args[0],
            Hlvalue::Variable(v) if *v == start_in
        ));
        // start's exit should now target returnblock (mid got folded).
        assert_eq!(s.exits.len(), 1);
        assert!(Rc::ptr_eq(
            s.exits[0].borrow().target.as_ref().unwrap(),
            &returnblock
        ));
    }

    #[test]
    fn join_blocks_keeps_diamond() {
        // start -> {a, b} — both target the same block `merge`. `merge`
        // has 2 entries ⇒ join_blocks must NOT fold either into start
        // (entrymap[merge] != 1).
        let v = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let merge = Block::shared(vec![Hlvalue::Variable(Variable::new())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        // Boolean switch on `v`: False -> merge, True -> merge.
        start.borrow_mut().exitswitch = Some(Hlvalue::Variable(v.clone()));
        let left = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(merge.clone()),
            Some(Hlvalue::Constant(crate::flowspace::model::Constant::new(
                crate::flowspace::model::ConstValue::Bool(false),
            ))),
        )
        .into_ref();
        let right = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(merge.clone()),
            Some(Hlvalue::Constant(crate::flowspace::model::Constant::new(
                crate::flowspace::model::ConstValue::Bool(true),
            ))),
        )
        .into_ref();
        start
            .borrow_mut()
            .closeblock(vec![left.clone(), right.clone()]);
        left.borrow_mut().prevblock = Some(Rc::downgrade(&start));
        right.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        let merge_link = Link::new(
            vec![merge.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        merge.borrow_mut().closeblock(vec![merge_link.clone()]);
        merge_link.borrow_mut().prevblock = Some(Rc::downgrade(&merge));

        let ops_before = merge.borrow().operations.len();
        join_blocks(&graph);
        let ops_after = merge.borrow().operations.len();

        // Start still has 2 exits, merge still has 1 op-list, returnblock unchanged.
        assert_eq!(start.borrow().exits.len(), 2);
        assert_eq!(ops_before, ops_after);
    }

    #[test]
    fn constfold_exitswitch_picks_matching_exit() {
        // start has a Bool(true) exitswitch with two exits:
        //   exitcase=False -> left
        //   exitcase=True  -> right (should survive)
        // After folding, start has a single exit with exitcase=None
        // targeting `right`.
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let v = Variable::new();
        let left = Block::shared(vec![Hlvalue::Variable(Variable::new())]);
        let right = Block::shared(vec![Hlvalue::Variable(Variable::new())]);
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        start.borrow_mut().exitswitch = Some(Hlvalue::Constant(C::new(CV::Bool(true))));
        let l_link = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(left.clone()),
            Some(Hlvalue::Constant(C::new(CV::Bool(false)))),
        )
        .into_ref();
        let r_link = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(right.clone()),
            Some(Hlvalue::Constant(C::new(CV::Bool(true)))),
        )
        .into_ref();
        start
            .borrow_mut()
            .closeblock(vec![l_link.clone(), r_link.clone()]);
        l_link.borrow_mut().prevblock = Some(Rc::downgrade(&start));
        r_link.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        // left and right each exit to returnblock so the graph is
        // structurally closed (though checkgraph is not invoked here).
        let left_end = Link::new(
            vec![left.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        left.borrow_mut().closeblock(vec![left_end.clone()]);
        left_end.borrow_mut().prevblock = Some(Rc::downgrade(&left));
        let right_end = Link::new(
            vec![right.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        right.borrow_mut().closeblock(vec![right_end.clone()]);
        right_end.borrow_mut().prevblock = Some(Rc::downgrade(&right));

        constfold_exitswitch(&graph);

        let s = start.borrow();
        assert!(s.exitswitch.is_none());
        assert_eq!(s.exits.len(), 1);
        assert!(s.exits[0].borrow().exitcase.is_none());
        assert!(Rc::ptr_eq(
            s.exits[0].borrow().target.as_ref().unwrap(),
            &right
        ));
    }

    #[test]
    fn remove_trivial_links_merges_empty_link() {
        // start -> mid (via trivial link, no args) -> return.
        // `mid` has 1 op and 1 entry/exit. The source→mid link has no
        // args, so remove_trivial_links folds mid into start.
        let v = Variable::new();
        let mid = Block::shared(vec![]);
        mid.borrow_mut()
            .operations
            .push(crate::flowspace::model::SpaceOperation::new(
                "int_zero",
                vec![],
                Hlvalue::Variable(v.clone()),
            ));
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        // Trivial link start -> mid (no args — mid.inputargs also empty).
        let start_link = Link::new(vec![], Some(mid.clone()), None).into_ref();
        start.borrow_mut().closeblock(vec![start_link.clone()]);
        start_link.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        let mid_link = Link::new(
            vec![Hlvalue::Variable(v.clone())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        mid.borrow_mut().closeblock(vec![mid_link.clone()]);
        mid_link.borrow_mut().prevblock = Some(Rc::downgrade(&mid));

        remove_trivial_links(&graph);

        let s = start.borrow();
        // mid's op now lives on start.
        assert_eq!(s.operations.len(), 1);
        assert_eq!(s.operations[0].opname, "int_zero");
        // start's exit now targets returnblock directly.
        assert_eq!(s.exits.len(), 1);
        assert!(Rc::ptr_eq(
            s.exits[0].borrow().target.as_ref().unwrap(),
            &returnblock
        ));
    }

    #[test]
    fn remove_identical_vars_dedupes_duplicate_phi_inputs() {
        // A block receives two input args whose phi nodes pull from
        // the same Constant on every incoming link. `remove_identical_vars`
        // should collapse them to a single inputarg.
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let v_a = Variable::new();
        let v_b = Variable::new();
        let body = Block::shared(vec![
            Hlvalue::Variable(v_a.clone()),
            Hlvalue::Variable(v_b.clone()),
        ]);
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        // start -> body with [Const(1), Const(1)] — both phis see the
        // same value so merge_identical_phi_nodes unifies v_a / v_b.
        let link_sb = Link::new(
            vec![
                Hlvalue::Constant(C::new(CV::Int(1))),
                Hlvalue::Constant(C::new(CV::Int(1))),
            ],
            Some(body.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link_sb]);
        let link_br = Link::new(
            vec![body.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        body.closeblock(vec![link_br.clone()]);

        remove_identical_vars(&graph);

        // body should now have exactly one inputarg — the duplicate
        // was pruned, and the entrymap link's args list shrank to
        // match.
        assert_eq!(body.borrow().inputargs.len(), 1);
        assert_eq!(start.borrow().exits[0].borrow().args.len(), 1);
    }

    #[test]
    fn transform_xxxitem_rewrites_getitem_to_idx() {
        // Block raising `getitem` with an IndexError exit → opname
        // swaps to `getitem_idx`.
        use crate::flowspace::model::{
            ConstValue as CV, Constant as C, HOST_ENV as HE, LAST_EXCEPTION, SpaceOperation as SO,
        };
        let index_err = HE.lookup_builtin("IndexError").unwrap();
        let v_in = Variable::new();
        let block = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        block.borrow_mut().operations.push(SO::new(
            "getitem",
            vec![
                Hlvalue::Variable(v_in.clone()),
                Hlvalue::Constant(C::new(CV::Int(0))),
            ],
            Hlvalue::Variable(Variable::new()),
        ));
        block.borrow_mut().exitswitch =
            Some(Hlvalue::Constant(C::new(CV::Atom(LAST_EXCEPTION.clone()))));
        let graph = FunctionGraph::new("f", block.clone());
        let ok_link = Link::new(
            vec![block.borrow().operations[0].result.clone()],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        let err_link = Link::new(
            vec![
                Hlvalue::Constant(C::new(CV::HostObject(index_err.clone()))),
                Hlvalue::Constant(C::new(CV::HostObject(index_err.clone()))),
            ],
            Some(graph.exceptblock.clone()),
            Some(Hlvalue::Constant(C::new(CV::HostObject(index_err.clone())))),
        )
        .into_ref();
        block.closeblock(vec![ok_link, err_link]);

        transform_xxxitem(&graph);

        assert_eq!(block.borrow().operations[0].opname, "getitem_idx");
    }

    #[test]
    fn remove_dead_exceptions_noop_on_non_raising_block() {
        // No canraise block → function is effectively a no-op and
        // doesn't panic.
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let link = Link::new(
            vec![Hlvalue::Constant(C::new(CV::Int(0)))],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);
        remove_dead_exceptions(&graph);
    }

    #[test]
    fn simplify_exceptions_noop_on_simple_graph() {
        // No Exception-terminated canraise block → pass is a no-op.
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let link = Link::new(
            vec![Hlvalue::Constant(C::new(CV::Int(0)))],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);
        simplify_exceptions(&graph);
    }

    #[test]
    fn remove_identical_vars_ssa_dedupes_duplicate_phis() {
        // Same fixture as remove_identical_vars_dedupes_duplicate_phi_inputs
        // — verify the SSA variant collapses constant-fed duplicate
        // phis to a single input.
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let v_a = Variable::new();
        let v_b = Variable::new();
        let body = Block::shared(vec![
            Hlvalue::Variable(v_a.clone()),
            Hlvalue::Variable(v_b.clone()),
        ]);
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        let link_sb = Link::new(
            vec![
                Hlvalue::Constant(C::new(CV::Int(1))),
                Hlvalue::Constant(C::new(CV::Int(1))),
            ],
            Some(body.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link_sb]);
        let link_br = Link::new(
            vec![body.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        body.closeblock(vec![link_br]);

        remove_identical_vars_ssa(&graph);
        // body lost at least one inputarg.
        assert!(body.borrow().inputargs.len() < 2);
    }

    #[test]
    fn transform_dead_op_vars_drops_unused_pure_ops() {
        // Block: two unused `add` ops whose results don't flow into
        // link.args / exitswitch / return. transform_dead_op_vars
        // should drop them.
        use crate::flowspace::model::{ConstValue as CV, Constant as C, SpaceOperation as SO};
        let v_live = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(v_live.clone())]);
        // dead_a = add v_live, 1
        // dead_b = add v_live, 2
        let dead_a = Variable::new();
        let dead_b = Variable::new();
        start.borrow_mut().operations.push(SO::new(
            "add",
            vec![
                Hlvalue::Variable(v_live.clone()),
                Hlvalue::Constant(C::new(CV::Int(1))),
            ],
            Hlvalue::Variable(dead_a.clone()),
        ));
        start.borrow_mut().operations.push(SO::new(
            "add",
            vec![
                Hlvalue::Variable(v_live.clone()),
                Hlvalue::Constant(C::new(CV::Int(2))),
            ],
            Hlvalue::Variable(dead_b.clone()),
        ));

        let graph = FunctionGraph::new("f", start.clone());
        let link = Link::new(
            vec![Hlvalue::Variable(v_live.clone())],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);

        transform_dead_op_vars(&graph, None);

        // Both dead ops are gone; block.operations is empty.
        assert!(start.borrow().operations.is_empty());
    }

    #[test]
    fn coalesce_bool_noop_on_simple_graph() {
        // No bool op at block tail → pass is a no-op.
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        let link = Link::new(
            vec![Hlvalue::Constant(C::new(CV::Int(0)))],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);
        coalesce_bool(&graph);
    }

    #[test]
    fn remove_assertion_errors_drops_assertion_branch() {
        // A canraise block with two exits:
        //   exit[0] (exitcase=None)        → returnblock (non-ex path)
        //   exit[1] (exitcase=AssertionError cls) → exceptblock
        //     with args=[Constant(AssertionError), Constant(instance)]
        // remove_assertion_errors should drop exit[1] and promote the
        // surviving exit — but because exit[0].exitcase was already
        // None and exits_len goes from 2→1, the exitswitch is cleared.
        use crate::flowspace::model::{
            ConstValue as CV, Constant as C, HOST_ENV as HE, SpaceOperation as SO,
        };
        let assert_err = HE.lookup_builtin("AssertionError").unwrap();
        let v_in = Variable::new();
        let body = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        // A trivial op so canraise's raising_op check passes.
        body.borrow_mut().operations.push(SO::new(
            "simple_call",
            vec![Hlvalue::Variable(v_in.clone())],
            Hlvalue::Variable(Variable::new()),
        ));
        body.borrow_mut().exitswitch = Some(Hlvalue::Constant(C::new(CV::Atom(
            crate::flowspace::model::LAST_EXCEPTION.clone(),
        ))));

        let start_in = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(start_in.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();
        let exceptblock = graph.exceptblock.clone();

        let start_link = Link::new(
            vec![Hlvalue::Variable(start_in.clone())],
            Some(body.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![start_link]);

        // Non-exceptional exit (exitcase=None) body → return.
        let ok_link = Link::new(
            vec![body.borrow().operations[0].result.clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        // Exceptional exit (exitcase=AssertionError) body → exceptblock.
        let err_link = Link::new(
            vec![
                Hlvalue::Constant(C::new(CV::HostObject(assert_err.clone()))),
                Hlvalue::Constant(C::new(CV::HostObject(assert_err.clone()))),
            ],
            Some(exceptblock.clone()),
            Some(Hlvalue::Constant(C::new(CV::HostObject(
                assert_err.clone(),
            )))),
        )
        .into_ref();
        body.closeblock(vec![ok_link.clone(), err_link.clone()]);

        remove_assertion_errors(&graph);

        // body should have exactly one exit (the non-exceptional one).
        assert_eq!(body.borrow().exits.len(), 1);
        assert!(body.borrow().exitswitch.is_none());
        assert!(Rc::ptr_eq(
            body.borrow().exits[0].borrow().target.as_ref().unwrap(),
            &returnblock
        ));
    }

    #[test]
    fn transform_ovfcheck_rewrites_add_to_add_ovf() {
        // Block operations: `t = add v1 v2`; `r = simple_call(ovfcheck, t)`.
        // After transform_ovfcheck, the block should contain only
        // `r = add_ovf v1 v2` (t absorbed into r via renaming).
        use crate::flowspace::model::{
            ConstValue as CV, Constant as C, HOST_ENV as HE, SpaceOperation as SO,
        };
        let ovf_sentinel = HE.lookup_builtin("ovfcheck").unwrap();
        let v1 = Variable::new();
        let v2 = Variable::new();
        let t = Variable::new();
        let r = Variable::new();
        let start = Block::shared(vec![
            Hlvalue::Variable(v1.clone()),
            Hlvalue::Variable(v2.clone()),
        ]);
        start.borrow_mut().operations.push(SO::new(
            "add",
            vec![Hlvalue::Variable(v1.clone()), Hlvalue::Variable(v2.clone())],
            Hlvalue::Variable(t.clone()),
        ));
        start.borrow_mut().operations.push(SO::new(
            "simple_call",
            vec![
                Hlvalue::Constant(C::new(CV::HostObject(ovf_sentinel.clone()))),
                Hlvalue::Variable(t.clone()),
            ],
            Hlvalue::Variable(r.clone()),
        ));

        let graph = FunctionGraph::new("f", start.clone());
        let link = Link::new(
            vec![Hlvalue::Variable(r.clone())],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link.clone()]);

        transform_ovfcheck(&graph);

        let b = start.borrow();
        assert_eq!(b.operations.len(), 1);
        assert_eq!(b.operations[0].opname, "add_ovf");
        // The `return` link's arg was renamed from r → t.
        let link_arg = link.borrow().args[0].clone();
        assert!(matches!(link_arg, Some(Hlvalue::Variable(v)) if v == t));
    }

    #[test]
    fn simplify_graph_runs_all_passes_on_valid_graph() {
        // Smoke-test the driver end-to-end: minimum valid graph, all
        // 12 passes applied, checkgraph bookend. Must not panic.
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("g", start.clone());
        let link = Link::new(
            vec![Hlvalue::Constant(C::new(CV::Int(7)))],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);
        simplify_graph(&graph, None);
    }

    #[test]
    fn cleanup_graph_composes_passes_on_valid_graph() {
        // Smoke test — cleanup_graph on a minimum valid graph must
        // not panic (checkgraph bookends + the three inner passes are
        // all idempotent on a well-formed shape).
        use crate::flowspace::model::{ConstValue as CV, Constant as C};
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("g", start.clone());
        let link = Link::new(
            vec![Hlvalue::Constant(C::new(CV::Int(7)))],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![link]);

        cleanup_graph(&graph);
    }

    #[test]
    fn remove_trivial_links_preserves_links_carrying_args() {
        // start -> mid with 1 arg. `link.args` is non-empty so
        // remove_trivial_links must NOT fold (args-less precondition
        // fails); mid's op stays on mid.
        let v_in = Variable::new();
        let mid = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        mid.borrow_mut()
            .operations
            .push(crate::flowspace::model::SpaceOperation::new(
                "noop",
                vec![Hlvalue::Variable(v_in.clone())],
                Hlvalue::Variable(Variable::new()),
            ));
        let start_in = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(start_in.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        let returnblock = graph.returnblock.clone();

        let start_link = Link::new(
            vec![Hlvalue::Variable(start_in.clone())],
            Some(mid.clone()),
            None,
        )
        .into_ref();
        start.borrow_mut().closeblock(vec![start_link.clone()]);
        start_link.borrow_mut().prevblock = Some(Rc::downgrade(&start));

        let mid_link = Link::new(
            vec![Hlvalue::Variable(Variable::new())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        mid.borrow_mut().closeblock(vec![mid_link.clone()]);
        mid_link.borrow_mut().prevblock = Some(Rc::downgrade(&mid));

        remove_trivial_links(&graph);

        // start still has no operations (not folded).
        assert!(start.borrow().operations.is_empty());
        // start's exit still targets mid (not rewired).
        assert!(Rc::ptr_eq(
            start.borrow().exits[0].borrow().target.as_ref().unwrap(),
            &mid
        ));
    }
}
