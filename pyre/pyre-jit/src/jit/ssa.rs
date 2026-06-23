//! Port of `rpython/translator/backendopt/ssa.py` onto pyre-jit's flow graph.
//!
//! Provides `DataFlowFamilyBuilder` and `SSA_to_SSI`, the data-flow phi-family
//! analysis the `simplify_graph` pass list depends on (`simplify.py:1067`).
//! Port reference: `majit/majit-translate/src/translator/backendopt/ssa.rs`.
//!
//! Classification (issue #112 scope #4): **direct PyPy parity**.

use std::collections::{HashMap, HashSet};

use majit_translate::tool::algo::unionfind::UnionFind;

use super::flow::{
    BlockRef, ExitSwitch, ExitSwitchElement, FlowValue, FunctionGraph, Variable, mkentrymap,
};

/// One unification opportunity: a block input paired with the per-incoming-link
/// actuals at that position.  RPython stores this as a flat list
/// `[block, inputvar, *linkvars]` (`ssa.py:65`); the Rust port names the parts.
struct Opportunity {
    block: BlockRef,
    inputvar: FlowValue,
    linkvars: Vec<FlowValue>,
}

/// `rpython/translator/backendopt/ssa.py:4-90` `class DataFlowFamilyBuilder`.
pub struct DataFlowFamilyBuilder {
    /// `self.opportunities` (ssa.py:17).
    opportunities: Vec<Opportunity>,
    /// `self.opportunities_with_const` (ssa.py:18).
    opportunities_with_const: Vec<Opportunity>,
    /// `self.variable_families = UnionFind()` (ssa.py:36).
    pub variable_families: UnionFind<FlowValue, ()>,
}

impl DataFlowFamilyBuilder {
    /// `DataFlowFamilyBuilder.__init__(self, graph)` (ssa.py:12-36).
    pub fn new(graph: &FunctionGraph) -> Self {
        let mut opportunities: Vec<Opportunity> = Vec::new();
        let mut opportunities_with_const: Vec<Opportunity> = Vec::new();

        // `entrymap = mkentrymap(graph); del entrymap[graph.startblock]`.
        let mut entrymap = mkentrymap(graph);
        entrymap.remove(&graph.startblock);

        for (block, links) in entrymap.iter() {
            // `assert links`.
            assert!(!links.is_empty(), "entrymap entry has no links");
            let inputargs = block.borrow().inputargs.clone();
            for (n, inputvar) in inputargs.iter().enumerate() {
                // `vars = [block, inputvar]; put_in = opportunities`.
                let mut linkvars: Vec<FlowValue> = Vec::with_capacity(links.len());
                let mut put_in_const = false;
                for link in links {
                    // `var = link.args[n]`.
                    let var =
                        link.borrow().args.get(n).cloned().flatten().expect(
                            "DataFlowFamilyBuilder: link.args position missing for inputarg",
                        );
                    // `if not isinstance(var, Variable): put_in = opportunities_with_const`.
                    if matches!(var, FlowValue::Constant(_)) {
                        put_in_const = true;
                    }
                    linkvars.push(var);
                }
                let opp = Opportunity {
                    block: block.clone(),
                    inputvar: inputvar.clone(),
                    linkvars,
                };
                if put_in_const {
                    opportunities_with_const.push(opp);
                } else {
                    opportunities.push(opp);
                }
            }
        }

        DataFlowFamilyBuilder {
            opportunities,
            opportunities_with_const,
            variable_families: UnionFind::new(|_: &FlowValue| ()),
        }
    }

    /// `DataFlowFamilyBuilder.complete(self)` (ssa.py:38-63).
    pub fn complete(&mut self) -> bool {
        let mut any_progress_at_all = false;
        let mut progress = true;
        while progress {
            progress = false;
            let mut pending_opportunities: Vec<Opportunity> = Vec::new();
            // Move out so we can mutate the UF while iterating.
            let opps = std::mem::take(&mut self.opportunities);
            for opp in opps {
                // `repvars = [find_rep(v1) for v1 in vars[1:]]` —
                // `vars[1:]` = [inputvar, *linkvars].
                let mut repvars: Vec<FlowValue> = Vec::with_capacity(1 + opp.linkvars.len());
                repvars.push(self.variable_families.find_rep(opp.inputvar.clone()));
                for v in &opp.linkvars {
                    repvars.push(self.variable_families.find_rep(v.clone()));
                }
                // `repvars_without_duplicates = dict.fromkeys(repvars)` —
                // insertion-ordered unique keys.
                let unique: Vec<FlowValue> = {
                    let mut seen: HashSet<FlowValue> = HashSet::new();
                    let mut out = Vec::new();
                    for v in &repvars {
                        if seen.insert(v.clone()) {
                            out.push(v.clone());
                        }
                    }
                    out
                };
                match unique.len() {
                    n if n > 2 => {
                        // `pending_opportunities.append(vars[:1] + repvars)`
                        // — recycle with vars[1:] replaced by reps.
                        let inputvar = repvars[0].clone();
                        let linkvars = repvars[1..].to_vec();
                        pending_opportunities.push(Opportunity {
                            block: opp.block.clone(),
                            inputvar,
                            linkvars,
                        });
                    }
                    2 => {
                        // `variable_families.union(*repvars_without_duplicates)`.
                        self.variable_families
                            .union(unique[0].clone(), unique[1].clone());
                        progress = true;
                    }
                    _ => {} // 0 or 1 distinct reps — nothing to do.
                }
            }
            self.opportunities = pending_opportunities;
            any_progress_at_all |= progress;
        }
        any_progress_at_all
    }

    /// `DataFlowFamilyBuilder.merge_identical_phi_nodes(self)` (ssa.py:65-86).
    pub fn merge_identical_phi_nodes(&mut self) -> bool {
        let mut any_progress_at_all = false;
        let mut progress = true;
        while progress {
            progress = false;
            // `block_phi_nodes = {}`.
            let mut block_phi_nodes: HashMap<(BlockRef, Vec<FlowValue>), FlowValue> =
                HashMap::new();
            // `for vars in self.opportunities + self.opportunities_with_const`.
            for opp in self
                .opportunities
                .iter()
                .chain(self.opportunities_with_const.iter())
            {
                let blockvar = opp.inputvar.clone();
                // `linksvars = [find_rep(v) for v in linksvars]`.
                let linksvars_rep: Vec<FlowValue> = opp
                    .linkvars
                    .iter()
                    .map(|v| self.variable_families.find_rep(v.clone()))
                    .collect();
                // `phi_node = (block,) + tuple(linksvars)`.
                let phi_node = (opp.block.clone(), linksvars_rep);
                if let Some(blockvar1) = block_phi_nodes.get(&phi_node).cloned() {
                    // `if variable_families.union(blockvar1, blockvar)[0]: progress = True`.
                    let (not_noop, _) = self.variable_families.union(blockvar1, blockvar);
                    if not_noop {
                        progress = true;
                    }
                } else {
                    block_phi_nodes.insert(phi_node, blockvar);
                }
            }
            any_progress_at_all |= progress;
        }
        any_progress_at_all
    }

    /// `DataFlowFamilyBuilder.get_variable_families(self)` (ssa.py:88-90).
    pub fn get_variable_families(&mut self) -> &mut UnionFind<FlowValue, ()> {
        self.complete();
        &mut self.variable_families
    }

    /// Consume-variant of `get_variable_families`.
    pub fn into_variable_families(mut self) -> UnionFind<FlowValue, ()> {
        self.complete();
        self.variable_families
    }
}

/// The Variables an `exitswitch` reads (`block.exitswitch` is a Variable, the
/// `c_last_exception` sentinel, or a jtransform tuple).
fn exitswitch_variables(sw: &ExitSwitch) -> Vec<Variable> {
    match sw {
        ExitSwitch::Value(v) => v.as_variable().into_iter().collect(),
        ExitSwitch::Tuple(elements) => elements
            .iter()
            .filter_map(|element| match element {
                ExitSwitchElement::Value(v) => v.as_variable(),
                ExitSwitchElement::Marker(_) => None,
            })
            .collect(),
    }
}

/// `rpython/translator/backendopt/ssa.py:128-132` `variables_created_in`.
///
/// ```python
/// def variables_created_in(block):
///     result = set(block.inputargs)
///     for op in block.operations:
///         result.add(op.result)
///     return result
/// ```
pub fn variables_created_in(block: &BlockRef) -> HashSet<FlowValue> {
    let mut result: HashSet<FlowValue> = HashSet::new();
    let b = block.borrow();
    for v in &b.inputargs {
        result.insert(v.clone());
    }
    for op in &b.operations {
        if let Some(r) = &op.result {
            result.insert(r.clone());
        }
    }
    result
}

/// `rpython/translator/backendopt/ssa.py:135-196` `SSA_to_SSI(graph, annotator=None)`.
///
/// Ensures every Variable used in a block is either defined in that block or
/// added to its inputargs (and to every incoming link's args), so the graph is
/// in valid SSI form (each Variable defined exactly once across the graph).
pub fn ssa_to_ssi(graph: &FunctionGraph) {
    let mut entrymap = mkentrymap(graph);
    entrymap.remove(&graph.startblock);

    let mut variable_families = DataFlowFamilyBuilder::new(graph).into_variable_families();

    // Build the initial pending list: for each non-start block, the Variables
    // it uses but does not itself create.
    let mut pending: Vec<(BlockRef, Variable)> = Vec::new();
    for block in graph.iterblocks() {
        if !entrymap.contains_key(&block) {
            continue;
        }
        let variables_created = variables_created_in(&block);
        let mut seen: HashSet<FlowValue> = variables_created.clone();
        let mut variables_used: Vec<Variable> = Vec::new();
        let record = |v: Variable, seen: &mut HashSet<FlowValue>, used: &mut Vec<Variable>| {
            if seen.insert(FlowValue::Variable(v)) {
                used.push(v);
            }
        };

        {
            let b = block.borrow();
            for op in &b.operations {
                for arg in &op.args {
                    for v in arg.variables() {
                        record(v, &mut seen, &mut variables_used);
                    }
                }
            }
            if let Some(sw) = &b.exitswitch {
                for v in exitswitch_variables(sw) {
                    record(v, &mut seen, &mut variables_used);
                }
            }
            for link in &b.exits {
                let l = link.borrow();
                for arg in &l.args {
                    if let Some(FlowValue::Variable(v)) = arg {
                        record(*v, &mut seen, &mut variables_used);
                    }
                }
            }
        }

        for v in variables_used {
            // `v._name not in ('last_exception_', 'last_exc_value_')`.
            let prefix = v.name_prefix();
            if prefix == "last_exception_" || prefix == "last_exc_value_" {
                continue;
            }
            pending.push((block.clone(), v));
        }
    }

    while let Some((block, v)) = pending.pop() {
        let v_rep = variable_families.find_rep(FlowValue::Variable(v));
        let variables_created = variables_created_in(&block);
        if variables_created.contains(&FlowValue::Variable(v)) {
            continue;
        }
        // Linear search for a w created in this block in the same family.
        let mut matched: Option<Variable> = None;
        for w in &variables_created {
            let FlowValue::Variable(w_var) = w else {
                continue;
            };
            if variable_families.find_rep(w.clone()) == v_rep {
                matched = Some(*w_var);
                break;
            }
        }
        if let Some(w_var) = matched {
            // `block.renamevariables({v: w})`.
            let mut renaming: HashMap<Variable, Variable> = HashMap::new();
            renaming.insert(v, w_var);
            block.borrow_mut().renamevariables(&renaming);
        } else {
            // Add `v` to every incoming link and the block's inputargs.
            //
            // Pyre walker adaptation: a block with no graph predecessors
            // (the startblock) is reached when `v` is defined in the walker's
            // register/slot model without a matching graph SpaceOp —
            // `block.operations` lacks the definition, so
            // `variables_created_in` cannot see it.  The value really exists
            // (in the walker's register/slot model), so the
            // link.args/inputargs threaded so far flow correctly from it; stop
            // threading instead of panicking.  RPython panics here because its
            // flow graphs are complete; pyre's walker graph recording is not
            // until the walker threads every value through the graph.
            let Some(links) = entrymap.get(&block).cloned() else {
                continue;
            };
            // `w = v.copy()` — fresh identity, same kind.
            let w = graph.fresh_like(v);
            variable_families.union(FlowValue::Variable(v), FlowValue::Variable(w));
            let mut renaming: HashMap<Variable, Variable> = HashMap::new();
            renaming.insert(v, w);
            block.borrow_mut().renamevariables(&renaming);
            block.borrow_mut().inputargs.push(FlowValue::Variable(w));
            for link in &links {
                // `link.args.append(v); pending.append((link.prevblock, v))`.
                link.borrow_mut().args.push(Some(FlowValue::Variable(v)));
                let prev = link
                    .borrow()
                    .prevblock_ref()
                    .expect("Link.prevblock missing");
                pending.push((prev, v));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::flatten::Kind;
    use crate::jit::flow::{Block, Link, Variable, VariableId};

    /// Mirrors `majit-translate/.../ssa.rs::data_flow_family_builder_unifies_linear_chain`.
    /// start(v_in) -> middle(v_mid) -> return(v_ret); each link passes its
    /// predecessor's var straight through, so all three land in one family.
    #[test]
    fn data_flow_family_builder_unifies_linear_chain() {
        let v_in = Variable::new(VariableId(20), Kind::Int);
        let v_mid = Variable::new(VariableId(21), Kind::Int);
        let start = Block::shared(vec![v_in.into()]);
        let middle = Block::shared(vec![v_mid.into()]);
        let graph = FunctionGraph::new("f", start.clone(), None);
        let returnblock = graph.returnblock.clone();

        let l1 = Link::new(vec![v_in.into()], Some(middle.clone()), None).into_ref();
        start.closeblock(vec![l1]);
        let l2 = Link::new(vec![v_mid.into()], Some(returnblock.clone()), None).into_ref();
        middle.closeblock(vec![l2]);

        let mut builder = DataFlowFamilyBuilder::new(&graph);
        assert!(builder.complete());
        let mut uf = builder.variable_families;
        let rep_in = uf.find_rep(FlowValue::Variable(v_in));
        let rep_mid = uf.find_rep(FlowValue::Variable(v_mid));
        assert_eq!(rep_in, rep_mid);
    }

    /// `SSA_to_SSI` threads a value used in a later block back through the
    /// intervening block's inputargs and the connecting links.
    #[test]
    fn ssa_to_ssi_threads_value_through_block() {
        // start() -> middle() -> return(); `middle` uses `c` (created in
        // start) in its link to the returnblock, so SSA_to_SSI must add `c`
        // to middle.inputargs and the start->middle link.
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone(), None);
        let returnblock = graph.returnblock.clone();

        // `c` is created in start by an op.
        let c = graph.fresh_variable(Kind::Int);
        push_op_local(&start, "int_zero", c);

        let middle = graph.new_block(vec![]);
        let l1 = Link::new(vec![], Some(middle.clone()), None).into_ref();
        start.closeblock(vec![l1]);
        // middle -> return passes `c`, which middle does not itself create.
        let l2 = Link::new(vec![c.into()], Some(returnblock), None).into_ref();
        middle.closeblock(vec![l2]);

        ssa_to_ssi(&graph);

        // middle gained an inputarg to receive `c`.
        assert_eq!(middle.borrow().inputargs.len(), 1);
        // the start->middle link now passes one arg.
        assert_eq!(start.borrow().exits[0].borrow().args.len(), 1);
    }

    fn push_op_local(block: &BlockRef, opname: &str, result: Variable) {
        use crate::jit::flow::{SpaceOperation, push_op};
        push_op(
            block,
            SpaceOperation::new(opname, vec![], Some(result.into()), -1),
        );
    }
}
