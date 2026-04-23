//! Polymorphic Bound Callable (PBC) representation — lowers indirect
//! method calls on `dyn Trait` receivers.
//!
//! RPython equivalent:
//! `rpython/rtyper/rpbc.py:199-217 FunctionReprBase.call`. RPython's
//! rtyper materialises the funcptr Variable via
//! `convert_to_concrete_llfn` and then emits
//! `genop('indirect_call', [funcptr, *args, c_graphs])` carrying the
//! full row-of-graphs as the trailing Constant. Pyre's rtyper-equivalent
//! does the same shape over `OpKind::Call { target: CallTarget::Indirect
//! { trait_root, method_name }, .. }` sites: insert a `VtableMethodPtr`
//! to materialise the funcptr ValueId, then replace the original Call
//! with an `OpKind::IndirectCall` carrying funcptr + args + the family
//! `graphs` (full `c_graphs`, not yet filtered by JIT candidates —
//! that filtering happens later in `call.py::graphs_from(op)`).
//!
//! After this pass, no `CallTarget::Indirect` survives in the graph;
//! `jtransform` only sees `OpKind::IndirectCall` for indirect dispatch.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::annotator::argument::ArgumentsForTranslation;
use crate::annotator::bookkeeper::{Bookkeeper, PositionKey};
use crate::annotator::description::{CallFamily, CallTableRow, DescKey};
use crate::annotator::model::{AnnotatorError, SomePBC};
use crate::call::CallControl;
use crate::flowspace::argument::CallShape;
use crate::flowspace::pygraph::PyGraph;
use crate::model::{
    BlockId, CallTarget, FunctionGraph as JitFunctionGraph, OpKind, SpaceOperation,
};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{_ptr, FuncType, PtrTarget};
use crate::translator::rtyper::rclass;
use crate::translator::rtyper::rtyper::RPythonTyper;
// `TypeResolutionState` lives at `jit_codewriter/type_state.rs` — see the
// PRE-EXISTING-ADAPTATION header there. `lower_indirect_calls` threads a
// `&mut TypeResolutionState` so the inserted `VtableMethodPtr` funcptr
// gets recorded as Signed for downstream regalloc / flatten.
use crate::jit_codewriter::type_state::TypeResolutionState;

/// RPython `ConcreteCallTableRow(dict)` (rpbc.py:71-82).
#[derive(Clone, Debug)]
pub struct ConcreteCallTableRow {
    pub row: HashMap<DescKey, _ptr>,
    pub fntype: FuncType,
    pub attrname: Option<String>,
}

type ConcreteCallTableRowRef = Rc<RefCell<ConcreteCallTableRow>>;

impl ConcreteCallTableRow {
    /// RPython `ConcreteCallTableRow.from_row(cls, rtyper, row)`.
    pub fn from_row(rtyper: &RPythonTyper, row: &CallTableRow) -> Result<Self, TyperError> {
        let mut concrete_row = HashMap::new();
        let mut last_llfn = None;
        for (funcdesc, graph) in row {
            let llfn = rtyper.getcallable(graph)?;
            last_llfn = Some(llfn.clone());
            concrete_row.insert(*funcdesc, llfn);
        }
        let fntype = match last_llfn
            .expect("ConcreteCallTableRow::from_row requires a non-empty row")
            ._TYPE
            .TO
            .clone()
        {
            PtrTarget::Func(func) => func,
            other => panic!("ConcreteCallTableRow expects Func ptr, got {other:?}"),
        };
        Ok(ConcreteCallTableRow {
            row: concrete_row,
            fntype,
            attrname: None,
        })
    }
}

impl PartialEq for ConcreteCallTableRow {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row
    }
}

impl Eq for ConcreteCallTableRow {}

/// RPython `LLCallTable(object)` (rpbc.py:85-123).
#[derive(Clone, Debug, Default)]
pub struct LLCallTable {
    /// Upstream `self.table` keyed by `(shape, index)`.
    pub table: HashMap<(CallShape, usize), ConcreteCallTableRowRef>,
    /// Upstream `self.uniquerows`.
    pub uniquerows: Vec<ConcreteCallTableRowRef>,
}

impl LLCallTable {
    pub fn lookup(&self, row: &ConcreteCallTableRow) -> Result<(usize, ConcreteCallTableRow), ()> {
        for (existingindex, existingrow) in self.uniquerows.iter().enumerate() {
            let existingrow = existingrow.borrow();
            if row.fntype != existingrow.fntype {
                continue;
            }
            let mut mismatched = false;
            for (funcdesc, llfn) in &row.row {
                if let Some(existing_llfn) = existingrow.row.get(funcdesc) {
                    if llfn != existing_llfn {
                        mismatched = true;
                        break;
                    }
                }
            }
            if mismatched {
                continue;
            }
            let mut merged = ConcreteCallTableRow {
                row: row.row.clone(),
                fntype: row.fntype.clone(),
                attrname: row.attrname.clone(),
            };
            for (desc, llfn) in &existingrow.row {
                merged.row.insert(*desc, llfn.clone());
            }
            if merged.row.len() == row.row.len() + existingrow.row.len() {
                // no common funcdesc, not a match
            } else {
                return Ok((existingindex, merged));
            }
        }
        Err(())
    }

    pub fn add(&mut self, row: ConcreteCallTableRow) {
        match self.lookup(&row) {
            Err(()) => self.uniquerows.push(Rc::new(RefCell::new(row))),
            Ok((index, merged)) => {
                if merged == *self.uniquerows[index].borrow() {
                    return;
                }
                self.uniquerows.remove(index);
                self.add(merged);
            }
        }
    }
}

/// RPython `build_concrete_calltable(rtyper, callfamily)` (rpbc.py:125-157).
pub fn build_concrete_calltable(
    rtyper: &RPythonTyper,
    callfamily: &CallFamily,
) -> Result<LLCallTable, TyperError> {
    let mut llct = LLCallTable::default();
    let mut concreterows: HashMap<(CallShape, usize), ConcreteCallTableRow> = HashMap::new();

    for (shape, rows) in &callfamily.calltables {
        for (index, row) in rows.iter().enumerate() {
            let concreterow = ConcreteCallTableRow::from_row(rtyper, row)?;
            concreterows.insert((shape.clone(), index), concreterow.clone());
            llct.add(concreterow);
        }
    }

    for ((shape, index), row) in concreterows {
        let (existingindex, biggerrow) = llct.lookup(&row).expect("row must exist in LLCallTable");
        let canonical_row = Rc::clone(&llct.uniquerows[existingindex]);
        assert!(biggerrow == *canonical_row.borrow());
        llct.table.insert((shape, index), canonical_row);
    }

    if llct.uniquerows.len() == 1 {
        llct.uniquerows[0].borrow_mut().attrname = None;
    } else {
        for (finalindex, row) in llct.uniquerows.iter_mut().enumerate() {
            row.borrow_mut().attrname = Some(format!("variant{finalindex}"));
        }
    }

    Ok(llct)
}

/// RPython `get_concrete_calltable(rtyper, callfamily)` (rpbc.py:159-174).
pub fn get_concrete_calltable(
    rtyper: &RPythonTyper,
    callfamily: &Rc<RefCell<CallFamily>>,
) -> Result<LLCallTable, AnnotatorError> {
    let key = Rc::as_ptr(callfamily) as usize;
    {
        let concrete_calltables = rtyper.concrete_calltables.borrow();
        if let Some((llct, oldsize)) = concrete_calltables.get(&key) {
            if *oldsize != callfamily.borrow().total_calltable_size {
                return Err(AnnotatorError::new("call table was unexpectedly extended"));
            }
            return Ok(llct.clone());
        }
    }

    let llct = build_concrete_calltable(rtyper, &callfamily.borrow())
        .map_err(|e| AnnotatorError::new(e.to_string()))?;
    let oldsize = callfamily.borrow().total_calltable_size;
    rtyper
        .concrete_calltables
        .borrow_mut()
        .insert(key, (llct.clone(), oldsize));
    Ok(llct)
}

/// RPython `FunctionReprBase.call()` row-selection prefix
/// (rpbc.py:214-218).
#[derive(Clone, Debug)]
pub struct SelectedCallFamilyRow {
    pub shape: CallShape,
    pub index: usize,
    pub row_of_graphs: CallTableRow,
    pub anygraph: Rc<PyGraph>,
}

/// Select the `(shape, index)` row witness for a `SomePBC` call family,
/// mirroring the first half of `FunctionReprBase.call()`:
///
/// * `descs = list(s_pbc.descriptions)`
/// * `shape, index = self.callfamily.find_row(...)`
/// * `row_of_graphs = self.callfamily.calltables[shape][index]`
/// * `anygraph = row_of_graphs.itervalues().next()`
pub fn select_call_family_row(
    bookkeeper: &Rc<Bookkeeper>,
    callfamily: &Rc<RefCell<CallFamily>>,
    s_pbc: &SomePBC,
    args: &ArgumentsForTranslation,
    op_key: Option<PositionKey>,
) -> Result<SelectedCallFamilyRow, AnnotatorError> {
    let descs: Vec<_> = s_pbc.descriptions.values().cloned().collect();
    let (shape, index) = callfamily
        .borrow()
        .find_row(bookkeeper, &descs, args, op_key)?;
    let row_of_graphs = callfamily
        .borrow()
        .calltables
        .get(&shape)
        .and_then(|table| table.get(index))
        .cloned()
        .ok_or_else(|| AnnotatorError::new("FunctionReprBase.call: missing calltable row"))?;
    let anygraph = row_of_graphs
        .values()
        .next()
        .cloned()
        .ok_or_else(|| AnnotatorError::new("FunctionReprBase.call: empty row_of_graphs"))?;
    Ok(SelectedCallFamilyRow {
        shape,
        index,
        row_of_graphs,
        anygraph,
    })
}

/// Walk every `OpKind::Call` whose target is `CallTarget::Indirect` and
/// rewrite it into the RPython-orthodox pair:
///
/// 1. `OpKind::VtableMethodPtr { receiver, trait_root, method_name }`
///    → produces `funcptr: ValueId` (Int kind)
/// 2. `OpKind::IndirectCall { funcptr, args, graphs, result_ty }`
///    → mirrors RPython `indirect_call(funcptr, *args, c_graphs)`
///
/// `args` remain the full call argument list, including the receiver.
/// RPython's `convert_to_concrete_llfn` materialises the funcptr but the
/// eventual `indirect_call` still receives `self, ...` as ordinary args.
pub fn lower_indirect_calls(
    graph: &mut JitFunctionGraph,
    type_state: &mut TypeResolutionState,
    call_control: &CallControl,
) {
    // Collect the (block, op_index) sites first so the rewrite below
    // can mutate the graph without aliasing the borrow.
    let sites: Vec<(usize, usize)> = graph
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(bid, block)| {
            block
                .operations
                .iter()
                .enumerate()
                .filter_map(move |(oi, op)| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::Indirect { .. },
                        ..
                    } => Some((bid, oi)),
                    _ => None,
                })
        })
        .collect();

    // Process in reverse so earlier indices stay valid as later sites
    // grow by 1 op (the inserted `VtableMethodPtr`).
    for (bid, oi) in sites.into_iter().rev() {
        let block_id = BlockId(bid);
        let op = graph.blocks[bid].operations[oi].clone();
        let (target, args, result_ty, result) = match op.kind {
            OpKind::Call {
                target,
                args,
                result_ty,
            } => (target, args, result_ty, op.result),
            _ => unreachable!("site filter mismatch"),
        };
        let (trait_root, method_name) = match target {
            CallTarget::Indirect {
                trait_root,
                method_name,
            } => (trait_root, method_name),
            _ => unreachable!("site filter mismatch"),
        };
        let receiver = *args
            .first()
            .expect("dyn-Trait method call must have a receiver arg");
        // RPython rclass.py:371-377 (condensed into a single op).
        let funcptr = rclass::class_get_method_ptr(
            graph,
            type_state,
            block_id,
            oi,
            receiver,
            trait_root.clone(),
            method_name.clone(),
        );
        // RPython rpbc.py:216 `c_graphs = row_of_graphs.values()` — full
        // family without JIT candidate filtering.
        //
        // `None` means "unknown family" — matches
        // `rpython/translator/backendopt/graphanalyze.py:117`, where
        // `graphs is None` short-circuits family analyzers to
        // `top_result()` (conservative: canraise, can_invalidate, etc.
        // default to True). `Some(vec![])` would instead be treated as
        // "empty family → No/false" by every analyzer, which is unsafe
        // when external/unregistered impls might still be called.
        // Pyre's source-only parser only sees `#[trait_method]`-
        // registered impls; if none are registered for a
        // `(trait, method)` family, we don't know whether that's
        // because the family is truly empty or because the impls live
        // outside the analyzed sources, so we take the conservative
        // `None` path.
        let family = call_control.all_impls_for_indirect(&trait_root, &method_name);
        let graphs = if family.is_empty() {
            None
        } else {
            Some(family)
        };
        // The original Call op is now at index `oi + 1` because
        // `class_get_method_ptr` inserted `VtableMethodPtr` at `oi`.
        graph.blocks[bid].operations[oi + 1] = SpaceOperation {
            result,
            kind: OpKind::IndirectCall {
                funcptr,
                args,
                graphs,
                result_ty,
            },
        };
    }
}

/// Debug-build invariant: after `lower_indirect_calls` runs, no
/// `CallTarget::Indirect` may survive in the graph. Callers can run
/// this assert to catch missed sites early.
#[cfg(debug_assertions)]
pub fn assert_no_indirect_call_targets(graph: &JitFunctionGraph) {
    for (bid, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            if let OpKind::Call {
                target: CallTarget::Indirect { .. },
                ..
            } = &op.kind
            {
                panic!(
                    "post-lowering invariant violation: \
                     CallTarget::Indirect survived at block {bid} op {oi}: {:?}",
                    op
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::argument::ArgumentsForTranslation;
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::description::{DescEntry, FunctionDesc};
    use crate::annotator::model::{SomeInteger, SomePBC, SomeValue};
    use crate::call::CallControl;
    use crate::flowspace::model::{
        Block as FlowBlock, BlockRefExt as FlowBlockRefExt, ConstValue as FlowConstValue,
        Constant as FlowConstant, FunctionGraph as FlowFunctionGraph, GraphFunc as FlowGraphFunc,
        Hlvalue as FlowHlvalue, Link as FlowLink, Variable as FlowVariable,
    };
    use crate::model::{FunctionGraph, OpKind, ValueType};
    use crate::translate_legacy::annotator::annrpython::annotate;
    use crate::translate_legacy::rtyper::rtyper::resolve_types;
    use crate::translator::rtyper::lltypesystem::lltype::{_ptr, FuncType, functionptr};

    fn make_rtyper() -> (Rc<RPythonAnnotator>, RPythonTyper) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        (ann, rtyper)
    }

    fn make_pygraph(name: &str) -> Rc<PyGraph> {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let mut arg_var = FlowVariable::named("x");
        arg_var.concretetype = Some(LowLevelType::Signed);
        arg_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        let arg = FlowHlvalue::Variable(arg_var);
        let startblock = FlowBlock::shared(vec![arg.clone()]);
        let mut ret_var = FlowVariable::new();
        ret_var.concretetype = Some(LowLevelType::Void);
        ret_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        let graph = FlowFunctionGraph::with_return_var(
            name,
            startblock.clone(),
            FlowHlvalue::Variable(ret_var),
        );
        let link = Rc::new(std::cell::RefCell::new(FlowLink::new(
            vec![arg],
            Some(graph.returnblock.clone()),
            None,
        )));
        startblock.closeblock(vec![link]);
        Rc::new(PyGraph {
            graph: Rc::new(std::cell::RefCell::new(graph)),
            func: FlowGraphFunc::new(
                name,
                FlowConstant::new(FlowConstValue::Dict(Default::default())),
            ),
            signature: std::cell::RefCell::new(crate::flowspace::argument::Signature::new(
                vec!["x".to_string()],
                None,
                None,
            )),
            defaults: std::cell::RefCell::new(None),
            access_directly: std::cell::Cell::new(false),
        })
    }

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    fn int_sig(names: &[&str]) -> crate::flowspace::argument::Signature {
        crate::flowspace::argument::Signature::new(
            names.iter().map(|name| name.to_string()).collect(),
            None,
            None,
        )
    }

    fn fake_llfn(name: &str, arity: usize) -> _ptr {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        functionptr(
            FuncType {
                args: vec![LowLevelType::Signed; arity],
                result: LowLevelType::Void,
            },
            name,
            None,
            None,
        )
    }

    #[test]
    fn build_concrete_calltable_merges_overlapping_rows() {
        let (_ann, rtyper) = make_rtyper();
        let mut family = CallFamily::new(DescKey(1));
        let shared = make_pygraph("shared");
        let mut row0 = CallTableRow::new();
        row0.insert(DescKey(10), shared.clone());
        row0.insert(DescKey(11), make_pygraph("left_only"));
        let mut row1 = CallTableRow::new();
        row1.insert(DescKey(10), shared);
        row1.insert(DescKey(12), make_pygraph("right_only"));
        family.calltables.insert(
            CallShape {
                shape_cnt: 1,
                shape_keys: Vec::new(),
                shape_star: false,
            },
            vec![row0, row1],
        );

        let llct = build_concrete_calltable(&rtyper, &family).unwrap();
        assert_eq!(llct.uniquerows.len(), 1);
        let merged = llct.uniquerows[0].borrow();
        assert_eq!(merged.row.len(), 3);
        assert_eq!(merged.attrname, None);
        assert_eq!(llct.table.len(), 2);
        assert_eq!(llct.table.values().next().unwrap().borrow().row.len(), 3);
    }

    #[test]
    fn build_concrete_calltable_table_rows_alias_uniquerows() {
        let (_ann, rtyper) = make_rtyper();
        let mut family = CallFamily::new(DescKey(1));
        let mut row0 = CallTableRow::new();
        row0.insert(DescKey(10), make_pygraph("a"));
        family.calltables.insert(
            CallShape {
                shape_cnt: 1,
                shape_keys: Vec::new(),
                shape_star: false,
            },
            vec![row0],
        );

        let llct = build_concrete_calltable(&rtyper, &family).unwrap();
        let shared = llct.table.values().next().expect("table row missing");
        assert!(Rc::ptr_eq(shared, &llct.uniquerows[0]));
    }

    #[test]
    fn build_concrete_calltable_assigns_variant_names_to_multiple_uniquerows() {
        let (_ann, rtyper) = make_rtyper();
        let mut family = CallFamily::new(DescKey(1));
        let mut row0 = CallTableRow::new();
        row0.insert(DescKey(10), make_pygraph("a"));
        let mut row1 = CallTableRow::new();
        row1.insert(DescKey(11), make_pygraph("b"));
        family.calltables.insert(
            CallShape {
                shape_cnt: 1,
                shape_keys: Vec::new(),
                shape_star: false,
            },
            vec![row0, row1],
        );

        let llct = build_concrete_calltable(&rtyper, &family).unwrap();
        assert_eq!(llct.uniquerows.len(), 2);
        let mut attrnames: Vec<_> = llct
            .uniquerows
            .iter()
            .map(|row| row.borrow().attrname.clone().unwrap())
            .collect();
        attrnames.sort();
        assert_eq!(
            attrnames,
            vec!["variant0".to_string(), "variant1".to_string()]
        );
        let mut table_attrnames: Vec<_> = llct
            .table
            .values()
            .map(|row| row.borrow().attrname.clone().unwrap())
            .collect();
        table_attrnames.sort();
        assert_eq!(
            table_attrnames,
            vec!["variant0".to_string(), "variant1".to_string()]
        );
    }

    #[test]
    fn llcalltable_lookup_rejects_rows_with_mismatched_fntype() {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let mut llct = LLCallTable::default();
        llct.uniquerows
            .push(Rc::new(RefCell::new(ConcreteCallTableRow {
                row: HashMap::from([(DescKey(10), fake_llfn("a", 1))]),
                fntype: FuncType {
                    args: vec![LowLevelType::Signed],
                    result: LowLevelType::Void,
                },
                attrname: None,
            })));

        let row = ConcreteCallTableRow {
            row: HashMap::from([(DescKey(10), fake_llfn("a", 1))]),
            fntype: FuncType {
                args: vec![LowLevelType::Signed, LowLevelType::Signed],
                result: LowLevelType::Void,
            },
            attrname: None,
        };

        assert!(llct.lookup(&row).is_err());
    }

    #[test]
    fn get_concrete_calltable_caches_and_rejects_extension() {
        let (_ann, rtyper) = make_rtyper();
        let family = Rc::new(RefCell::new(CallFamily::new(DescKey(1))));
        let mut row0 = CallTableRow::new();
        row0.insert(DescKey(10), make_pygraph("a"));
        family.borrow_mut().calltables.insert(
            CallShape {
                shape_cnt: 1,
                shape_keys: Vec::new(),
                shape_star: false,
            },
            vec![row0],
        );
        family.borrow_mut().total_calltable_size = 1;

        let first = get_concrete_calltable(&rtyper, &family).unwrap();
        let second = get_concrete_calltable(&rtyper, &family).unwrap();
        assert_eq!(first.uniquerows.len(), second.uniquerows.len());
        assert_eq!(rtyper.concrete_calltables.borrow().len(), 1);

        family.borrow_mut().total_calltable_size = 2;
        let err = get_concrete_calltable(&rtyper, &family).unwrap_err();
        assert!(
            err.to_string()
                .contains("call table was unexpectedly extended")
        );
    }

    #[test]
    fn select_call_family_row_matches_find_row_and_returns_witness_graph() {
        let bk = bk();
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            "f",
            int_sig(&["x"]),
            None,
            None,
        )));
        let cached_graph = make_pygraph("f_graph");
        fd.borrow().cache.borrow_mut().insert(
            crate::annotator::description::GraphCacheKey::None,
            cached_graph.clone(),
        );
        let args = ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
            None,
        );
        FunctionDesc::consider_call_site(&[fd.clone()], &args, &SomeValue::Impossible, None)
            .unwrap();
        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd.clone())], false);
        let callfamily = fd.borrow().base.getcallfamily().unwrap();

        let selected = select_call_family_row(&bk, &callfamily, &s_pbc, &args, None).unwrap();

        assert_eq!(selected.shape.shape_cnt, 1);
        assert_eq!(selected.index, 0);
        assert_eq!(selected.row_of_graphs.len(), 1);
        assert!(Rc::ptr_eq(&selected.anygraph.graph, &cached_graph.graph));
    }

    fn build_indirect_graph() -> FunctionGraph {
        let mut graph = FunctionGraph::new("outer");
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "h".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_return(graph.startblock, None);
        graph
    }

    /// Boundary: pre-lowering graph has `CallTarget::Indirect`; post-lowering
    /// graph has `VtableMethodPtr + IndirectCall` and zero `Indirect`
    /// targets. Mirrors RPython rpbc.py:199-217 emit shape.
    #[test]
    fn lower_indirect_calls_eliminates_call_target_indirect() {
        let mut cc = CallControl::new();
        cc.register_trait_method("run", Some("Handler"), "A", FunctionGraph::new("A::run"));
        cc.register_trait_method("run", Some("Handler"), "B", FunctionGraph::new("B::run"));
        cc.find_all_graphs_for_tests();

        let mut graph = build_indirect_graph();

        // Pre-lowering: a CallTarget::Indirect exists.
        let pre_indirect = graph.blocks[graph.startblock.0]
            .operations
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::Indirect { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(pre_indirect, 1, "expected 1 Indirect Call pre-lowering");

        let annotations = annotate(&graph);
        let mut type_state = resolve_types(&graph, &annotations);
        lower_indirect_calls(&mut graph, &mut type_state, &cc);

        // Post-lowering: invariant — zero Indirect targets.
        assert_no_indirect_call_targets(&graph);

        // Post-lowering: exactly one VtableMethodPtr and one IndirectCall.
        let ops = &graph.blocks[graph.startblock.0].operations;
        let vtable_ptr_count = ops
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::VtableMethodPtr { .. }))
            .count();
        let indirect_call_count = ops
            .iter()
            .filter(|op| matches!(&op.kind, OpKind::IndirectCall { .. }))
            .count();
        assert_eq!(vtable_ptr_count, 1);
        assert_eq!(indirect_call_count, 1);
    }

    /// Regression: inherent (non-trait) method calls — `CallTarget::Method`
    /// targets — must pass through `lower_indirect_calls` unchanged.
    /// RPython rpbc.py:199-217 only rewrites the indirect-call dispatch
    /// (`s_pbc.callfamily`), inherent method calls are statically resolved
    /// upstream by the rtyper.
    #[test]
    fn inherent_method_unchanged_by_lowering() {
        let mut cc = CallControl::new();
        cc.register_function_graph(
            crate::parse::CallPath::from_segments(["Foo", "bar"]),
            FunctionGraph::new("Foo::bar"),
        );

        let mut graph = FunctionGraph::new("outer");
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "f".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::method("bar", Some("Foo".to_string())),
                args: vec![receiver],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_return(graph.startblock, None);

        let pre_ops_len = graph.blocks[graph.startblock.0].operations.len();
        let annotations = annotate(&graph);
        let mut type_state = resolve_types(&graph, &annotations);
        lower_indirect_calls(&mut graph, &mut type_state, &cc);

        // Same op count; the Call op survives untouched.
        let ops = &graph.blocks[graph.startblock.0].operations;
        assert_eq!(ops.len(), pre_ops_len);
        let method_count = ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::Method { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(method_count, 1, "Method target must survive lowering");
        assert!(
            !ops.iter()
                .any(|op| matches!(&op.kind, OpKind::VtableMethodPtr { .. })),
            "inherent call must not produce VtableMethodPtr"
        );
        assert!(
            !ops.iter()
                .any(|op| matches!(&op.kind, OpKind::IndirectCall { .. })),
            "inherent call must not produce IndirectCall"
        );
    }
}
