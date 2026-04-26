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
use crate::annotator::description::{CallFamily, CallTableRow, DescEntry, DescKey};
use crate::annotator::model::{AnnotatorError, DescKind, SomeObjectTrait, SomePBC};
use crate::call::CallControl;
use crate::flowspace::argument::CallShape;
use crate::flowspace::pygraph::PyGraph;
use crate::model::{
    BlockId, CallTarget, FunctionGraph as JitFunctionGraph, OpKind, SpaceOperation,
};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, DelayedPointer, FuncType, MallocFlavor, Ptr as LLPtr, PtrTarget, StructType,
    malloc as ll_malloc, nullptr as ll_nullptr,
};
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

/// RPython `small_cand(rtyper, s_pbc)` (rpbc.py:26-34).
///
/// ```python
/// def small_cand(rtyper, s_pbc):
///     if 1 < len(s_pbc.descriptions) < rtyper.getconfig().translation.withsmallfuncsets:
///         callfamily = s_pbc.any_description().getcallfamily()
///         llct = get_concrete_calltable(rtyper, callfamily)
///         if (len(llct.uniquerows) == 1 and
///                 (not s_pbc.subset_of or small_cand(rtyper, s_pbc.subset_of))):
///             return True
///     return False
/// ```
///
/// `withsmallfuncsets = 0` is upstream's default; with that value,
/// the strict `< withsmallfuncsets` bound is never satisfied and the
/// helper returns `false` without probing `callfamily`.
pub fn small_cand(rtyper: &RPythonTyper, s_pbc: &SomePBC) -> Result<bool, TyperError> {
    let withsmallfuncsets = rtyper
        .getconfig()
        .map(|cfg| cfg.translation.withsmallfuncsets)
        .unwrap_or(0);
    let nb = s_pbc.descriptions.len();
    if !(1 < nb && nb < withsmallfuncsets) {
        return Ok(false);
    }
    let Some(sample) = s_pbc.any_description() else {
        return Ok(false);
    };
    let sample_ref = sample.as_function().ok_or_else(|| {
        TyperError::message("small_cand: any_description() is not a FunctionDesc")
    })?;
    let callfamily = sample_ref.borrow().base.getcallfamily().map_err(|e| {
        TyperError::message(format!(
            "small_cand: getcallfamily failed: {}",
            e.msg.as_deref().unwrap_or("<no message>")
        ))
    })?;
    let llct = get_concrete_calltable(rtyper, &callfamily).map_err(|e| {
        TyperError::message(format!(
            "small_cand: get_concrete_calltable failed: {}",
            e.msg.as_deref().unwrap_or("<no message>")
        ))
    })?;
    if llct.uniquerows.len() != 1 {
        return Ok(false);
    }
    match &s_pbc.subset_of {
        None => Ok(true),
        Some(parent) => small_cand(rtyper, parent),
    }
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
pub(crate) mod tests {
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

    pub(crate) fn make_pygraph(name: &str) -> Rc<PyGraph> {
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
        let mut arg_var = FlowVariable::named("x");
        arg_var.set_concretetype(Some(LowLevelType::Signed));
        arg_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        let arg = FlowHlvalue::Variable(arg_var);
        let startblock = FlowBlock::shared(vec![arg.clone()]);
        let mut ret_var = FlowVariable::new();
        ret_var.set_concretetype(Some(LowLevelType::Void));
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
    fn small_cand_returns_false_for_singleton_descriptions() {
        let (_ann, rtyper) = make_rtyper();
        let bk = bk();
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            bk,
            None,
            "single",
            int_sig(&["x"]),
            None,
            None,
        )));
        // `1 < len(descs)` is `1 < 1` → false → early return.
        let s_pbc_one = SomePBC::new(vec![DescEntry::Function(fd)], false);
        assert!(!small_cand(&rtyper, &s_pbc_one).unwrap());
    }

    #[test]
    fn small_cand_returns_false_when_withsmallfuncsets_is_zero() {
        let (_ann, rtyper) = make_rtyper();
        let bk = bk();
        let fd_a = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            "a",
            int_sig(&["x"]),
            None,
            None,
        )));
        let fd_b = Rc::new(RefCell::new(FunctionDesc::new(
            bk,
            None,
            "b",
            int_sig(&["x"]),
            None,
            None,
        )));
        let s_pbc = SomePBC::new(
            vec![DescEntry::Function(fd_a), DescEntry::Function(fd_b)],
            false,
        );
        // Default withsmallfuncsets=0 → `2 < 0` is false → early return false.
        assert!(!small_cand(&rtyper, &s_pbc).unwrap());
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

// =====================================================================
// rpbc.py:177-372 — FunctionReprBase + FunctionRepr
//
// Upstream builds the full Repr hierarchy over SomePBC. The slice
// ported here covers the degenerate path used by `convert_desc_or_const`
// for class-attr method slots: `SomePBC` with exactly one
// `DescKind::Function` description and `can_be_None=False`. All other
// shapes (FunctionsPBCRepr / SmallFunctionSetPBCRepr / getFrozenPBCRepr
// / ClassesPBCRepr / MethodsPBCRepr / MethodOfFrozenPBCRepr) still
// surface as `MissingRTypeOperation` from `rtyper_makerepr`.
// =====================================================================

use crate::flowspace::model::{ConstValue, Constant, GraphKey, Hlvalue};
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{Repr, ReprState};
use std::rc::Weak;
use std::sync::Arc;

/// RPython `class FunctionReprBase(Repr)` (rpbc.py:177-221).
///
/// Shared state for every function-PBC repr. Concrete subclasses
/// override `lowleveltype` and the call-dispatch `call()` body —
/// upstream has `FunctionRepr`, `FunctionsPBCRepr`, and
/// `SmallFunctionSetPBCRepr` subclassing this.
///
/// Pyre composes rather than inherits: concrete reprs (for now only
/// [`FunctionRepr`]) embed [`FunctionReprBase`] as a field so the
/// shared fields live in one place while each repr declares its own
/// `Repr` trait impl.
#[derive(Debug)]
pub struct FunctionReprBase {
    /// RPython `self.rtyper = rtyper` (rpbc.py:179). Weak because the
    /// rtyper owns the strong reference and each repr lives inside
    /// `RPythonTyper.reprs` cache.
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.s_pbc = s_pbc` (rpbc.py:180). Stored by value;
    /// upstream keeps the original Python reference, pyre clones the
    /// structured `SomePBC`.
    pub s_pbc: SomePBC,
    /// RPython `self.callfamily = s_pbc.any_description().getcallfamily()`
    /// (rpbc.py:181). Kept optional because test harnesses may build a
    /// `SomePBC` whose descriptions have no attached callfamily yet —
    /// upstream raises in that case; pyre defers the raise until the
    /// dispatcher that actually needs the callfamily fires.
    pub callfamily: Option<Rc<RefCell<CallFamily>>>,
}

impl FunctionReprBase {
    /// RPython `FunctionReprBase.__init__(self, rtyper, s_pbc)`
    /// (rpbc.py:178-181).
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Self, TyperError> {
        // upstream: `self.callfamily =
        // s_pbc.any_description().getcallfamily()`.
        //
        // If `any_description()` returns `None` the PBC is empty — upstream
        // would have tripped the `assert len(descs) > 0` in the ctor.
        let callfamily = match s_pbc.any_description() {
            Some(entry) => match entry {
                DescEntry::Function(rc) => {
                    let fd = rc.borrow();
                    match fd.base.getcallfamily() {
                        Ok(family) => Some(family),
                        Err(_) => None,
                    }
                }
                // MethodDesc / FrozenDesc / MethodOfFrozenDesc /
                // ClassDesc also expose `getcallfamily` via the base
                // `Desc`; only FunctionRepr needs the stored handle
                // today so we leave those routes unpopulated until
                // their concrete reprs land.
                _ => None,
            },
            None => None,
        };
        Ok(FunctionReprBase {
            rtyper: Rc::downgrade(rtyper),
            s_pbc,
            callfamily,
        })
    }

    /// RPython `FunctionReprBase.get_s_callable(self)` (rpbc.py:183-184).
    pub fn get_s_callable(&self) -> &SomePBC {
        &self.s_pbc
    }

    /// RPython `FunctionReprBase.get_s_signatures(self, shape)`
    /// (rpbc.py:189-191).
    ///
    /// ```python
    /// def get_s_signatures(self, shape):
    ///     funcdesc = self.s_pbc.any_description()
    ///     return funcdesc.get_s_signatures(shape)
    /// ```
    ///
    /// Only meaningful when the pbc's representative description is a
    /// `FunctionDesc`; other `DescEntry` variants return `None` since
    /// upstream's `get_s_signatures` method lives on `FunctionDesc` only.
    pub fn get_s_signatures(
        &self,
        shape: &CallShape,
    ) -> Result<
        Vec<(
            Vec<crate::annotator::model::SomeValue>,
            crate::annotator::model::SomeValue,
        )>,
        TyperError,
    > {
        let Some(entry) = self.s_pbc.any_description() else {
            return Err(TyperError::message(
                "FunctionReprBase.get_s_signatures: s_pbc is empty",
            ));
        };
        let DescEntry::Function(rc) = entry else {
            return Err(TyperError::message(
                "FunctionReprBase.get_s_signatures: representative is not a FunctionDesc",
            ));
        };
        rc.borrow()
            .get_s_signatures(shape)
            .map_err(|e| TyperError::message(e.to_string()))
    }

    // Deferred ports — the remaining members of upstream
    // `FunctionReprBase` (rpbc.py:193-217) are:
    //
    //   * `rtype_simple_call(self, hop)`
    //   * `rtype_call_args(self, hop)`
    //   * `call(self, opname, hop)` — row selection + callparse +
    //     gencall
    //
    // All three depend on upstream `rpython/rtyper/callparse.py` and
    // `HighLevelOp.exception_is_here()` (ExceptionData). The row-
    // selection half of `call()` is already available as the free
    // function `select_call_family_row` (rpbc.rs:260-306); the rest
    // lands with the ExceptionData + callparse ports.
}

/// RPython `class FunctionRepr(FunctionReprBase)` (rpbc.py:315-369).
///
/// The repr for a statically known constant function — `SomePBC` with
/// a single `FunctionDesc` and `can_be_None=False`. `lowleveltype` is
/// `Void` because the callable is identity-encoded at compile time;
/// `convert_desc` / `convert_const` both return `None` (no runtime
/// value), and calls route through `convert_to_concrete_llfn` (port
/// pending).
#[derive(Debug)]
pub struct FunctionRepr {
    base: FunctionReprBase,
    state: ReprState,
    lltype: LowLevelType,
}

impl FunctionRepr {
    /// RPython `FunctionRepr(rtyper, s_pbc)` — inherits `__init__` from
    /// `FunctionReprBase` and sets `lowleveltype = Void` as a
    /// class-level attribute (rpbc.py:318).
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Self, TyperError> {
        Ok(FunctionRepr {
            base: FunctionReprBase::new(rtyper, s_pbc)?,
            state: ReprState::new(),
            lltype: LowLevelType::Void,
        })
    }

    /// Access to the embedded [`FunctionReprBase`] state.
    pub fn base(&self) -> &FunctionReprBase {
        &self.base
    }

    /// RPython `FunctionRepr.convert_to_concrete_llfn(v, shape, index,
    /// llop)` (rpbc.py:326-336).
    ///
    /// ```python
    /// def convert_to_concrete_llfn(self, v, shape, index, llop):
    ///     assert v.concretetype == Void
    ///     funcdesc, = self.s_pbc.descriptions
    ///     row_of_one_graph = self.callfamily.calltables[shape][index]
    ///     graph = row_of_one_graph[funcdesc]
    ///     llfn = self.rtyper.getcallable(graph)
    ///     return inputconst(typeOf(llfn), llfn)
    /// ```
    ///
    /// `llop` is part of the upstream signature for subclasses that
    /// materialise funcptrs via `get_specfunc_row`; `FunctionRepr` is
    /// `Void`-typed so it returns a compile-time `Constant` directly
    /// without consuming `llop`.
    pub fn convert_to_concrete_llfn(
        &self,
        v: &crate::flowspace::model::Hlvalue,
        shape: &CallShape,
        index: usize,
        _llops: &Rc<RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
    ) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
        // upstream: `assert v.concretetype == Void`.
        let v_ty = match v {
            crate::flowspace::model::Hlvalue::Variable(var) => var.concretetype(),
            crate::flowspace::model::Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        if v_ty != Some(LowLevelType::Void) {
            return Err(TyperError::message(
                "FunctionRepr.convert_to_concrete_llfn: v.concretetype != Void",
            ));
        }

        // upstream: `funcdesc, = self.s_pbc.descriptions`.
        if self.base.s_pbc.descriptions.len() != 1 {
            return Err(TyperError::message(
                "FunctionRepr.convert_to_concrete_llfn: expected a single FunctionDesc",
            ));
        }
        let funcdesc_entry = self
            .base
            .s_pbc
            .descriptions
            .values()
            .next()
            .expect("len checked");
        // `build_calltable_row` keys the row on `desc.rowkey()` (the
        // stable `Desc.identity`), not on pointer identity — so
        // lookup here has to follow the same route.
        let funcdesc_rowkey = funcdesc_entry
            .rowkey()
            .map_err(|e| TyperError::message(e.to_string()))?;

        // upstream: `self.callfamily.calltables[shape][index]`.
        let callfamily = self.base.callfamily.as_ref().ok_or_else(|| {
            TyperError::message("FunctionRepr.convert_to_concrete_llfn: callfamily not available")
        })?;
        let family_ref = callfamily.borrow();
        let table = family_ref.calltables.get(shape).ok_or_else(|| {
            TyperError::message("FunctionRepr.convert_to_concrete_llfn: shape not in calltables")
        })?;
        let row = table.get(index).ok_or_else(|| {
            TyperError::message("FunctionRepr.convert_to_concrete_llfn: row index out of range")
        })?;

        // upstream: `graph = row_of_one_graph[funcdesc]`.
        let graph = row.get(&funcdesc_rowkey).cloned().ok_or_else(|| {
            TyperError::message(
                "FunctionRepr.convert_to_concrete_llfn: funcdesc not in row_of_one_graph",
            )
        })?;

        // upstream: `llfn = self.rtyper.getcallable(graph); return
        //            inputconst(typeOf(llfn), llfn)`.
        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message(
                "FunctionRepr.convert_to_concrete_llfn: rtyper weak reference dropped",
            )
        })?;
        build_llfn_constant(&rtyper, &graph)
    }

    /// RPython `FunctionRepr.get_unique_llfn(self)` (rpbc.py:338-360).
    ///
    /// ```python
    /// def get_unique_llfn(self):
    ///     funcdesc, = self.s_pbc.descriptions
    ///     tables = []        # find the simple call in the calltable
    ///     for shape, table in self.callfamily.calltables.items():
    ///         if not shape[1] and not shape[2]:
    ///             tables.append(table)
    ///     if len(tables) != 1:
    ///         raise TyperError("cannot pass a function with various call shapes")
    ///     table, = tables
    ///     graphs = []
    ///     for row in table:
    ///         if funcdesc in row:
    ///             graphs.append(row[funcdesc])
    ///     if not graphs:
    ///         raise TyperError("cannot pass here a function that is not called")
    ///     graph = graphs[0]
    ///     if graphs != [graph] * len(graphs):
    ///         raise TyperError("cannot pass a specialized function here")
    ///     llfn = self.rtyper.getcallable(graph)
    ///     return inputconst(typeOf(llfn), llfn)
    /// ```
    pub fn get_unique_llfn(&self) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
        // upstream: `funcdesc, = self.s_pbc.descriptions`.
        if self.base.s_pbc.descriptions.len() != 1 {
            return Err(TyperError::message(
                "FunctionRepr.get_unique_llfn: expected a single FunctionDesc",
            ));
        }
        let funcdesc_entry = self
            .base
            .s_pbc
            .descriptions
            .values()
            .next()
            .expect("len checked");
        // upstream `funcdesc in row` / `row[funcdesc]` — calltable rows
        // are keyed by `rowkey()` (stable identity), not by
        // pointer-derived `desc_key()`.
        let funcdesc_rowkey = funcdesc_entry
            .rowkey()
            .map_err(|e| TyperError::message(e.to_string()))?;

        let callfamily = self.base.callfamily.as_ref().ok_or_else(|| {
            TyperError::message("FunctionRepr.get_unique_llfn: callfamily not available")
        })?;
        let family_ref = callfamily.borrow();

        // upstream: walk `self.callfamily.calltables.items()` and keep
        // every table whose shape has no kwargs and no `*args`. `shape[1]`
        // is `shape_keys` and `shape[2]` is `shape_star`.
        let mut tables: Vec<&Vec<CallTableRow>> = Vec::new();
        for (shape, table) in &family_ref.calltables {
            if shape.shape_keys.is_empty() && !shape.shape_star {
                tables.push(table);
            }
        }
        if tables.len() != 1 {
            return Err(TyperError::message(
                "cannot pass a function with various call shapes",
            ));
        }
        let table = tables[0];

        // upstream: `graphs = [row[funcdesc] for row in table if
        //                      funcdesc in row]`.
        let mut graphs: Vec<Rc<PyGraph>> = Vec::new();
        for row in table {
            if let Some(graph) = row.get(&funcdesc_rowkey) {
                graphs.push(graph.clone());
            }
        }
        if graphs.is_empty() {
            return Err(TyperError::message(
                "cannot pass here a function that is not called",
            ));
        }
        let first = graphs[0].clone();
        // upstream: `if graphs != [graph] * len(graphs)` — every entry
        // must be the same graph (by identity).
        if graphs.iter().any(|g| !Rc::ptr_eq(g, &first)) {
            return Err(TyperError::message(
                "cannot pass a specialized function here",
            ));
        }

        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("FunctionRepr.get_unique_llfn: rtyper weak reference dropped")
        })?;
        build_llfn_constant(&rtyper, &first)
    }

    /// RPython `FunctionRepr.get_concrete_llfn(s_pbc, args_s, op)`
    /// (rpbc.py:362-369).
    ///
    /// ```python
    /// def get_concrete_llfn(self, s_pbc, args_s, op):
    ///     bk = self.rtyper.annotator.bookkeeper
    ///     funcdesc, = s_pbc.descriptions
    ///     with bk.at_position(None):
    ///         argspec = simple_args(args_s)
    ///         graph = funcdesc.get_graph(argspec, op)
    ///     llfn = self.rtyper.getcallable(graph)
    ///     return inputconst(typeOf(llfn), llfn)
    /// ```
    ///
    /// `op` is upstream's SpaceOperation identity key; the Rust port
    /// threads it as `Option<PositionKey>` to match
    /// [`FunctionDesc::get_graph`](crate::annotator::description::FunctionDesc::get_graph).
    pub fn get_concrete_llfn(
        &self,
        s_pbc: &SomePBC,
        args_s: Vec<crate::annotator::model::SomeValue>,
        op_key: Option<PositionKey>,
    ) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
        // upstream: `funcdesc, = s_pbc.descriptions`.
        if s_pbc.descriptions.len() != 1 {
            return Err(TyperError::message(
                "FunctionRepr.get_concrete_llfn: expected a single FunctionDesc",
            ));
        }
        let funcdesc_rc = match s_pbc.descriptions.values().next().expect("len checked") {
            DescEntry::Function(rc) => rc.clone(),
            _ => {
                return Err(TyperError::message(
                    "FunctionRepr.get_concrete_llfn: single desc is not a FunctionDesc",
                ));
            }
        };

        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("FunctionRepr.get_concrete_llfn: rtyper weak reference dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("FunctionRepr.get_concrete_llfn: annotator weak reference dropped")
        })?;

        // upstream: `with bk.at_position(None): argspec = simple_args(args_s);
        //            graph = funcdesc.get_graph(argspec, op)`.
        let graph = {
            let _guard = annotator.bookkeeper.at_position(None);
            let argspec = crate::annotator::argument::simple_args(args_s);
            funcdesc_rc
                .borrow()
                .get_graph(&argspec, op_key)
                .map_err(|e| TyperError::message(e.to_string()))?
        };

        build_llfn_constant(&rtyper, &graph)
    }

    /// RPython `FunctionReprBase.call(self, hop)` (rpbc.py:199-221).
    /// FunctionRepr override — delegates to the shared free helper
    /// [`pbc_call_via_concrete_llfn`] with `FunctionRepr`'s
    /// `convert_to_concrete_llfn`.
    pub fn call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> Result<Option<Hlvalue>, TyperError> {
        pbc_call_via_concrete_llfn(&self.base, self, hop, |v, shape, idx, llops| {
            self.convert_to_concrete_llfn(v, shape, idx, llops)
        })
    }
}

/// Shared body of upstream `FunctionReprBase.call(self, hop)`
/// (rpbc.py:199-221):
///
/// ```python
/// def call(self, hop):
///     bk = self.rtyper.annotator.bookkeeper
///     args = hop.spaceop.build_args(hop.args_s[1:])
///     s_pbc = hop.args_s[0]   # possibly more precise than self.s_pbc
///     descs = list(s_pbc.descriptions)
///     shape, index = self.callfamily.find_row(bk, descs, args, hop.spaceop)
///     row_of_graphs = self.callfamily.calltables[shape][index]
///     anygraph = row_of_graphs.itervalues().next()
///     vfn = hop.inputarg(self, arg=0)
///     vlist = [self.convert_to_concrete_llfn(vfn, shape, index, hop.llops)]
///     vlist += callparse.callparse(self.rtyper, anygraph, hop)
///     rresult = callparse.getrresult(self.rtyper, anygraph)
///     hop.exception_is_here()
///     if isinstance(vlist[0], Constant):
///         v = hop.genop('direct_call', vlist, resulttype=rresult)
///     else:
///         vlist.append(hop.inputconst(Void, row_of_graphs.values()))
///         v = hop.genop('indirect_call', vlist, resulttype=rresult)
///     if hop.r_result is impossible_repr:
///         return None
///     else:
///         return hop.llops.convertvar(v, rresult, hop.r_result)
/// ```
///
/// Pyre composes rather than inherits — upstream's `call` lives on
/// `FunctionReprBase` and dispatches through virtual
/// `convert_to_concrete_llfn`. The Rust port encodes the polymorphism
/// as a `FnOnce` parameter so each concrete repr (`FunctionRepr`,
/// `FunctionsPBCRepr`, …) can plug in its own
/// `convert_to_concrete_llfn` body without losing the line-by-line
/// match against upstream `FunctionReprBase.call`.
pub(super) fn pbc_call_via_concrete_llfn<F>(
    base: &FunctionReprBase,
    self_repr: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    convert_to_concrete_llfn: F,
) -> Result<Option<Hlvalue>, TyperError>
where
    F: FnOnce(
        &Hlvalue,
        &CallShape,
        usize,
        &Rc<RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
    ) -> Result<Hlvalue, TyperError>,
{
    use crate::annotator::bookkeeper::build_args_for_op;
    use crate::annotator::model::SomeValue;
    use crate::translator::rtyper::callparse::{self, RResult};
    use crate::translator::rtyper::rmodel::impossible_repr;
    use crate::translator::rtyper::rtyper::GenopResult;

    // upstream: `bk = self.rtyper.annotator.bookkeeper`.
    let rtyper = base.rtyper.upgrade().ok_or_else(|| {
        TyperError::message("FunctionReprBase.call: rtyper weak reference dropped")
    })?;
    let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
        TyperError::message("FunctionReprBase.call: annotator weak reference dropped")
    })?;
    let bookkeeper = annotator.bookkeeper.clone();

    // upstream: `args = hop.spaceop.build_args(hop.args_s[1:])`.
    //
    // Pyre's analogue is `build_args_for_op(opname, args_s)` —
    // bookkeeper.rs:2110 — which mirrors the upstream
    // `CallOp.build_args` polymorphic dispatch
    // (`flowspace/operation.py:678 simple_call`,
    // `:699 call_args`).
    let args_s_full = hop.args_s.borrow().clone();
    if args_s_full.is_empty() {
        return Err(TyperError::message(
            "FunctionReprBase.call: hop.args_s must contain the receiver",
        ));
    }
    let args = build_args_for_op(&hop.spaceop.opname, &args_s_full[1..])
        .map_err(|e| TyperError::message(e.to_string()))?;

    // upstream: `s_pbc = hop.args_s[0]; descs = list(s_pbc.descriptions);
    //            shape, index = self.callfamily.find_row(bk, descs, args, hop.spaceop)
    //            row_of_graphs = self.callfamily.calltables[shape][index]
    //            anygraph = row_of_graphs.itervalues().next()`.
    //
    // Reuses `select_call_family_row` (rpbc.rs:277) which mirrors the
    // four lines above. `op_key` is `None` because `HighLevelOp` does
    // not carry its enclosing block/graph identity — upstream caches
    // `find_row` keyed on `hop.spaceop` Python identity. The cache
    // miss is benign: `find_row` recomputes deterministically.
    let s_pbc = match args_s_full.first().cloned() {
        Some(SomeValue::PBC(pbc)) => pbc,
        Some(other) => {
            return Err(TyperError::message(format!(
                "FunctionReprBase.call: hop.args_s[0] is not a SomePBC: {other:?}"
            )));
        }
        None => unreachable!("emptiness checked above"),
    };
    let callfamily = base
        .callfamily
        .as_ref()
        .ok_or_else(|| TyperError::message("FunctionReprBase.call: callfamily not available"))?;
    // upstream `rpbc.py:204` passes `hop.spaceop` to `find_row`; pyre
    // carries the equivalent (graph + block + op_index identity) on
    // [`HighLevelOp::position_key`], stamped by `highlevelops_with_graph`
    // during `specialize_block`. Falls back to `None` for test-only
    // hops constructed via the bare [`HighLevelOp::new`] ctor — those
    // never reach a real callfamily lookup.
    let op_key = hop.position_key.borrow().clone();
    let row = select_call_family_row(&bookkeeper, callfamily, &s_pbc, &args, op_key)
        .map_err(|e| TyperError::message(e.to_string()))?;

    // upstream: `vfn = hop.inputarg(self, arg=0)`.
    let vfn = hop.inputarg(self_repr, 0)?;

    // upstream: `vlist = [self.convert_to_concrete_llfn(vfn, shape, index, hop.llops)]`.
    //
    // Pass the `Rc<RefCell<LowLevelOpList>>` directly so the multi-row
    // `FunctionsPBCRepr` branch can `borrow_mut()` to emit the `getfield`
    // op (rpbc.py:308-312). Single-row paths take a no-op pass-through
    // and never touch the inner buffer.
    let vlist0 = convert_to_concrete_llfn(&vfn, &row.shape, row.index, &hop.llops)?;
    let mut vlist: Vec<Hlvalue> = vec![vlist0];

    // upstream: `vlist += callparse.callparse(self.rtyper, anygraph, hop)`.
    vlist.extend(callparse::callparse(&rtyper, &row.anygraph, hop, None)?);

    // upstream: `rresult = callparse.getrresult(self.rtyper, anygraph)`.
    let rresult = callparse::getrresult(&rtyper, &row.anygraph)?;

    // upstream: `hop.exception_is_here()`.
    hop.exception_is_here()?;

    // upstream:
    //     if isinstance(vlist[0], Constant):
    //         v = hop.genop('direct_call', vlist, resulttype=rresult)
    //     else:
    //         vlist.append(hop.inputconst(Void, row_of_graphs.values()))
    //         v = hop.genop('indirect_call', vlist, resulttype=rresult)
    let resulttype = match &rresult {
        RResult::Repr(r) => GenopResult::Repr(r.clone()),
        // upstream `resulttype=lltype.Void` enters the LLType arm
        // (vresult.concretetype = Void; returns `Some(vresult)`), not
        // `GenopResult::Void` (which would return `None`).
        RResult::Void => GenopResult::LLType(LowLevelType::Void),
    };
    let v = if matches!(&vlist[0], Hlvalue::Constant(_)) {
        hop.genop("direct_call", vlist, resulttype)
    } else {
        // upstream `rpbc.py:216`:
        //     vlist.append(hop.inputconst(Void, row_of_graphs.values()))
        // pyre carries the candidate graphs as a `ConstValue::Graphs`
        // (graph identities via `GraphKey::as_usize`) since `Constant`
        // cannot hold raw `Rc<RefCell<FunctionGraph>>` without breaking
        // the wider `Sync`/`Hash` invariants (see `ConstValue::Graphs`
        // doc on `flowspace/model.rs`). Sorted so the constant is
        // deterministic across HashMap iteration orders — order has no
        // semantic meaning at the indirect_call protocol level
        // (`lower_indirect_calls` at rpbc.rs:319 still recovers the
        // graphs from `OpKind::Call { CallTarget::Indirect, .. }`,
        // which is the authoritative channel).
        let mut graph_ids: Vec<usize> = row
            .row_of_graphs
            .values()
            .map(|g| GraphKey::of(&g.graph).as_usize())
            .collect();
        graph_ids.sort_unstable();
        vlist.push(Hlvalue::Constant(
            crate::translator::rtyper::rtyper::HighLevelOp::inputconst(
                &LowLevelType::Void,
                &ConstValue::Graphs(graph_ids),
            )?,
        ));
        hop.genop("indirect_call", vlist, resulttype)
    };

    // upstream:
    //     if hop.r_result is impossible_repr:
    //         return None
    //     else:
    //         return hop.llops.convertvar(v, rresult, hop.r_result)
    let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
        TyperError::message("FunctionReprBase.call: hop.r_result not initialised")
    })?;
    // upstream `r is impossible_repr` is a Python identity check;
    // pyre stores the singleton `Arc<VoidRepr>` so the underlying
    // `VoidRepr` address is stable — compare via raw pointer through
    // `&dyn Repr` (matches the pattern at `rtyper.rs:5638`
    // `Arc::ptr_eq`).
    let imp = impossible_repr();
    let imp_ptr = (imp.as_ref() as *const _ as *const ()) as usize;
    let r_result_ptr = (r_result.as_ref() as *const dyn Repr as *const ()) as usize;
    if r_result_ptr == imp_ptr {
        return Ok(None);
    }
    // `genop` always returns `Some(Variable)` because we passed a
    // typed `resulttype`; the `None` case (upstream `resulttype=None`)
    // does not apply here.
    let v_h = v.ok_or_else(|| {
        TyperError::message("FunctionReprBase.call: genop returned None despite typed resulttype")
    })?;
    let rresult_repr: std::sync::Arc<dyn Repr> = match rresult {
        RResult::Repr(r) => r,
        // Void rresult only reaches convertvar when r_result is also
        // impossible_repr — which already short-circuited above.
        // Defensive fallback: treat as the impossible_repr singleton
        // so identity in convertvar holds.
        RResult::Void => impossible_repr() as std::sync::Arc<dyn Repr>,
    };
    let converted =
        hop.llops
            .borrow_mut()
            .convertvar(v_h, rresult_repr.as_ref(), r_result.as_ref())?;
    Ok(Some(converted))
}

/// Build an `inputconst(typeOf(llfn), llfn)` Constant from an
/// `Rc<PyGraph>`, mirroring upstream's boilerplate tail of all three
/// `FunctionRepr` llfn helpers.
fn build_llfn_constant(
    rtyper: &Rc<RPythonTyper>,
    graph: &Rc<PyGraph>,
) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
    let func_ptr = rtyper.getcallable(graph)?;
    let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
    let c = crate::translator::rtyper::rmodel::inputconst_from_lltype(
        &func_ptr_type,
        &crate::flowspace::model::ConstValue::LLPtr(Box::new(func_ptr)),
    )?;
    Ok(crate::flowspace::model::Hlvalue::Constant(c))
}

impl Repr for FunctionRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "FunctionRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::FunctionRepr
    }

    /// RPython `FunctionReprBase.get_r_implfunc(self)` (rpbc.py:186-187) —
    /// upstream `return self, 0`. Inherited by `FunctionRepr` via the
    /// `FunctionReprBase` base class.
    fn get_r_implfunc(&self) -> Result<(&dyn Repr, usize), TyperError> {
        Ok((self, 0))
    }

    /// RPython `FunctionRepr.convert_desc(self, funcdesc)`
    /// (rpbc.py:320-321):
    ///
    /// ```python
    /// def convert_desc(self, funcdesc):
    ///     return None
    /// ```
    ///
    /// Returns the "no runtime value" sentinel as a Void-typed
    /// `Constant(None, Void)`. Upstream's `None` return gets wrapped
    /// by `convert_desc_or_const` into a `Constant(None, Void)` at the
    /// call site in `rclass.setup_vtable`.
    fn convert_desc(&self, _desc: &DescEntry) -> Result<Constant, TyperError> {
        Ok(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    }

    /// RPython `FunctionRepr.convert_const(self, value)`
    /// (rpbc.py:323-324):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     return None
    /// ```
    ///
    /// Mirrors `convert_desc`: no runtime value, Void-typed `Constant`.
    fn convert_const(&self, _value: &ConstValue) -> Result<Constant, TyperError> {
        Ok(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    }

    /// RPython `FunctionReprBase.rtype_simple_call(self, hop)`
    /// (rpbc.py:193-194) — `return self.call(hop)`. Inherited by
    /// `FunctionRepr` via `class FunctionRepr(FunctionReprBase)`.
    fn rtype_simple_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.call(hop)
    }

    /// RPython `FunctionReprBase.rtype_call_args(self, hop)`
    /// (rpbc.py:196-197) — `return self.call(hop)`. Inherited by
    /// `FunctionRepr` via `class FunctionRepr(FunctionReprBase)`.
    fn rtype_call_args(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.call(hop)
    }
}

/// RPython `class FunctionsPBCRepr(CanBeNull, FunctionReprBase)`
/// (rpbc.py:224-312).
///
/// ```python
/// class FunctionsPBCRepr(CanBeNull, FunctionReprBase):
///     def __init__(self, rtyper, s_pbc):
///         FunctionReprBase.__init__(self, rtyper, s_pbc)
///         llct = get_concrete_calltable(self.rtyper, self.callfamily)
///         self.concretetable = llct.table
///         self.uniquerows = llct.uniquerows
///         if len(llct.uniquerows) == 1:
///             row = llct.uniquerows[0]
///             self.lowleveltype = row.fntype
///         else:
///             self.lowleveltype = self.setup_specfunc()
///         self.funccache = {}
/// ```
///
/// Pyre lands the single-row case (one uniquerow → `lowleveltype = row.fntype`)
/// and defers the multi-row `setup_specfunc` branch. The `CanBeNull`
/// mixin (upstream rmodel.py:251-260 — `rtype_bool` via
/// `ptr_nonzero`) is not yet wired and will land when `rtype_bool` is
/// routed through the pairtype dispatcher.
#[derive(Debug)]
pub struct FunctionsPBCRepr {
    /// Inherited state from `FunctionReprBase` (rpbc.py:178-181).
    pub base: FunctionReprBase,
    /// RPython `self.concretetable = llct.table` (rpbc.py:230).
    pub concretetable: HashMap<(CallShape, usize), ConcreteCallTableRowRef>,
    /// RPython `self.uniquerows = llct.uniquerows` (rpbc.py:231).
    pub uniquerows: Vec<ConcreteCallTableRowRef>,
    /// RPython `self.lowleveltype = row.fntype` (rpbc.py:234) or the
    /// deferred `self.setup_specfunc()` result (rpbc.py:239). Wrapped
    /// in `Ptr(...)` so the type is a function pointer as upstream —
    /// `row.fntype` is a `FuncType` and the pointer wrap mirrors
    /// `typeOf(llfn)` which returns `Ptr(FuncType)`.
    lltype: LowLevelType,
    /// RPython `self.funccache = {}` (rpbc.py:240). Keyed on
    /// `DescKey` (pointer-identity) matching `convert_desc`'s
    /// upstream `self.funccache[funcdesc]` lookup.
    pub funccache: RefCell<HashMap<DescKey, Constant>>,
    state: ReprState,
}

impl FunctionsPBCRepr {
    /// RPython `FunctionsPBCRepr(rtyper, s_pbc)` (rpbc.py:227-240).
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Self, TyperError> {
        let base = FunctionReprBase::new(rtyper, s_pbc)?;
        let callfamily = base.callfamily.clone().ok_or_else(|| {
            TyperError::message("FunctionsPBCRepr: sample FunctionDesc has no callfamily")
        })?;
        // upstream: `llct = get_concrete_calltable(self.rtyper, self.callfamily)`.
        let llct = get_concrete_calltable(rtyper, &callfamily)
            .map_err(|e| TyperError::message(e.to_string()))?;
        // upstream rpbc.py:232-239:
        //     if len(llct.uniquerows) == 1:
        //         row = llct.uniquerows[0]
        //         self.lowleveltype = row.fntype
        //     else:
        //         self.lowleveltype = self.setup_specfunc()
        let lltype = if llct.uniquerows.len() == 1 {
            let row = llct.uniquerows[0].borrow();
            LowLevelType::Ptr(Box::new(LLPtr {
                TO: PtrTarget::Func(row.fntype.clone()),
            }))
        } else {
            Self::setup_specfunc(&llct.uniquerows)?
        };
        Ok(FunctionsPBCRepr {
            base,
            concretetable: llct.table,
            uniquerows: llct.uniquerows,
            lltype,
            funccache: RefCell::new(HashMap::new()),
            state: ReprState::new(),
        })
    }

    /// RPython `FunctionsPBCRepr.setup_specfunc(self)` (rpbc.py:242-247):
    ///
    /// ```python
    /// def setup_specfunc(self):
    ///     fields = []
    ///     for row in self.uniquerows:
    ///         fields.append((row.attrname, row.fntype))
    ///     kwds = {'hints': {'immutable': True, 'static_immutable': True}}
    ///     return Ptr(Struct('specfunc', *fields, **kwds))
    /// ```
    ///
    /// Each multi-row `uniquerow` carries a unique `variant{N}` attrname
    /// stamped in [`get_concrete_calltable`] (rpbc.rs:174-176); we
    /// surface a `TyperError::message` if attrname is `None` to catch
    /// the impossible single-row-leaks-into-multi-row case.
    fn setup_specfunc(uniquerows: &[ConcreteCallTableRowRef]) -> Result<LowLevelType, TyperError> {
        let mut fields: Vec<(String, LowLevelType)> = Vec::with_capacity(uniquerows.len());
        for row_ref in uniquerows {
            let row = row_ref.borrow();
            let attrname = row.attrname.clone().ok_or_else(|| {
                TyperError::message(
                    "FunctionsPBCRepr.setup_specfunc: multi-row uniquerow has no \
                     attrname (get_concrete_calltable should have stamped variant{N})",
                )
            })?;
            let fnptr_ty = LowLevelType::Ptr(Box::new(LLPtr {
                TO: PtrTarget::Func(row.fntype.clone()),
            }));
            fields.push((attrname, fnptr_ty));
        }
        // upstream `kwds = {'hints': {'immutable': True, 'static_immutable': True}}`
        let hints = vec![
            ("immutable".to_string(), ConstValue::Bool(true)),
            ("static_immutable".to_string(), ConstValue::Bool(true)),
        ];
        let struct_t = StructType::with_hints("specfunc", fields, hints);
        Ok(LowLevelType::Ptr(Box::new(LLPtr {
            TO: PtrTarget::Struct(struct_t),
        })))
    }

    /// RPython `FunctionsPBCRepr.get_specfunc_row(self, llop, v, c_rowname,
    /// resulttype)` (rpbc.py:252-253):
    ///
    /// ```python
    /// def get_specfunc_row(self, llop, v, c_rowname, resulttype):
    ///     return llop.genop('getfield', [v, c_rowname], resulttype=resulttype)
    /// ```
    fn get_specfunc_row(
        &self,
        llop: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
        v: &Hlvalue,
        attrname: &str,
        resulttype: LowLevelType,
    ) -> Result<Hlvalue, TyperError> {
        let cname = Constant::with_concretetype(ConstValue::byte_str(attrname), LowLevelType::Void);
        let var = llop
            .genop(
                "getfield",
                vec![v.clone(), Hlvalue::Constant(cname)],
                crate::translator::rtyper::rtyper::GenopResult::LLType(resulttype),
            )
            .ok_or_else(|| {
                TyperError::message(
                    "FunctionsPBCRepr.get_specfunc_row: genop(getfield) returned None \
                     despite typed resulttype",
                )
            })?;
        Ok(Hlvalue::Variable(var))
    }

    /// RPython `FunctionsPBCRepr.convert_to_concrete_llfn(self, v,
    /// shape, index, llop)` (rpbc.py:300-312):
    ///
    /// ```python
    /// def convert_to_concrete_llfn(self, v, shape, index, llop):
    ///     assert v.concretetype == self.lowleveltype
    ///     if len(self.uniquerows) == 1:
    ///         return v
    ///     else:
    ///         row = self.concretetable[shape, index]
    ///         cname = inputconst(Void, row.attrname)
    ///         return self.get_specfunc_row(llop, v, cname, row.fntype)
    /// ```
    ///
    /// Single-row port: returns `v` unchanged. The multi-row branch
    /// requires `setup_specfunc` (rpbc.py:242-247) which is deferred
    /// in tandem with `FunctionsPBCRepr::new`'s multi-row constructor
    /// path; both surface `MissingRTypeOperation` when reached.
    pub fn convert_to_concrete_llfn(
        &self,
        v: &Hlvalue,
        shape: &CallShape,
        index: usize,
        llops: &Rc<RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
    ) -> Result<Hlvalue, TyperError> {
        // upstream: `assert v.concretetype == self.lowleveltype`.
        let v_ty = match v {
            Hlvalue::Variable(var) => var.concretetype(),
            Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        if v_ty.as_ref() != Some(&self.lltype) {
            return Err(TyperError::message(format!(
                "FunctionsPBCRepr.convert_to_concrete_llfn: v.concretetype \
                 {:?} != self.lowleveltype {:?}",
                v_ty, self.lltype,
            )));
        }
        if self.uniquerows.len() == 1 {
            // upstream: `return v`.
            return Ok(v.clone());
        }
        // upstream rpbc.py:308-312:
        //     row = self.concretetable[shape, index]
        //     cname = inputconst(Void, row.attrname)
        //     return self.get_specfunc_row(llop, v, cname, row.fntype)
        let row_ref = self
            .concretetable
            .get(&(shape.clone(), index))
            .cloned()
            .ok_or_else(|| {
                TyperError::message(format!(
                    "FunctionsPBCRepr.convert_to_concrete_llfn: \
                     concretetable has no row for (shape={shape:?}, index={index})"
                ))
            })?;
        let row = row_ref.borrow();
        let attrname = row.attrname.clone().ok_or_else(|| {
            TyperError::message(
                "FunctionsPBCRepr.convert_to_concrete_llfn: \
                 multi-row concretetable entry has no attrname",
            )
        })?;
        let resulttype = LowLevelType::Ptr(Box::new(LLPtr {
            TO: PtrTarget::Func(row.fntype.clone()),
        }));
        let mut llop = llops.borrow_mut();
        self.get_specfunc_row(&mut llop, v, &attrname, resulttype)
    }

    /// RPython `FunctionReprBase.call(self, hop)` (rpbc.py:199-221).
    /// FunctionsPBCRepr override — delegates to the shared free helper
    /// [`pbc_call_via_concrete_llfn`] with `FunctionsPBCRepr`'s
    /// `convert_to_concrete_llfn`. For the single-row case the funcptr
    /// arrives as a `Variable` from `inputarg(self, 0)` (the arg
    /// carries `lowleveltype = Ptr(FuncType(...))`), so the dispatch
    /// takes the `indirect_call` branch.
    pub fn call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> Result<Option<Hlvalue>, TyperError> {
        pbc_call_via_concrete_llfn(&self.base, self, hop, |v, shape, idx, llops| {
            self.convert_to_concrete_llfn(v, shape, idx, llops)
        })
    }
}

impl Repr for FunctionsPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "FunctionsPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::FunctionsPBCRepr
    }

    /// RPython `FunctionReprBase.get_r_implfunc(self)` (rpbc.py:186-187) —
    /// upstream `return self, 0`. Inherited by `FunctionsPBCRepr` via the
    /// `FunctionReprBase` base class.
    fn get_r_implfunc(&self) -> Result<(&dyn Repr, usize), TyperError> {
        Ok((self, 0))
    }

    /// RPython `FunctionsPBCRepr.convert_desc(self, funcdesc)`
    /// (rpbc.py:255-287).
    ///
    /// Single-row port: pulls the `_ptr` stored at `row[funcdesc]`
    /// and caches the resulting `Constant` in `funccache`. The
    /// multi-row branch (build a `specfunc` Struct with one field per
    /// uniquerow) and the "funcdesc missing from the row" rffi-cast
    /// fallback both surface as `MissingRTypeOperation` until the
    /// spec-func malloc path lands.
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        let desc_rowkey = desc
            .rowkey()
            .map_err(|e| TyperError::message(e.to_string()))?;

        // upstream: `try: return self.funccache[funcdesc]`.
        if let Some(cached) = self.funccache.borrow().get(&desc_rowkey) {
            return Ok(cached.clone());
        }

        // upstream rpbc.py:261-271:
        //     llfns = {}
        //     found_anything = False
        //     for row in self.uniquerows:
        //         if funcdesc in row:
        //             llfn = row[funcdesc]
        //             found_anything = True
        //         else:
        //             llfn = nullptr(row.fntype.TO)
        //         llfns[row.attrname] = llfn
        let mut llfns: Vec<(Option<String>, _ptr)> = Vec::with_capacity(self.uniquerows.len());
        let mut last_llfn: Option<_ptr> = None;
        let mut found_anything = false;
        for row_ref in &self.uniquerows {
            let row = row_ref.borrow();
            let llfn = match row.row.get(&desc_rowkey) {
                Some(ptr) => {
                    found_anything = true;
                    ptr.clone()
                }
                None => ll_nullptr(LowLevelType::Func(Box::new(row.fntype.clone())))
                    .map_err(TyperError::message)?,
            };
            last_llfn = Some(llfn.clone());
            llfns.push((row.attrname.clone(), llfn));
        }

        // upstream rpbc.py:272-285:
        //     if len(self.uniquerows) == 1:
        //         if found_anything:
        //             result = llfn
        //         else:
        //             # extremely rare case
        //             result = rffi.cast(self.lowleveltype, ~len(self.funccache))
        //     else:
        //         result = self.create_specfunc()
        //         for attrname, llfn in llfns.items():
        //             setattr(result, attrname, llfn)
        let result_ptr: _ptr = if self.uniquerows.len() == 1 {
            if found_anything {
                last_llfn.expect("single-row loop must populate last_llfn when found_anything")
            } else {
                // upstream rpbc.py:275-280:
                //     # extremely rare case, shown only sometimes by
                //     # test_bug_callfamily: don't emit NULL, because
                //     # that would be interpreted as equal to None...
                //     # It should never be called anyway.
                //     result = rffi.cast(self.lowleveltype, ~len(self.funccache))
                //
                // Upstream's `~len(funccache)` casts an integer to a
                // function pointer; the value is unique per call, so
                // any two missing-from-row descs get distinct
                // non-NULL sentinels (so equality checks against None
                // and against each other resolve correctly).
                //
                // pyre's parity equivalent: a fresh `_ptr` carrying
                // [`DelayedPointer`] as the `_obj0` slot. `_ptr`'s
                // PartialEq (lltype.rs:1048-1051) falls back to
                // `_identity` for any non-Ok(Some) `_obj0`, and each
                // `_ptr::new` allocation gets a fresh identity. The
                // result: never-NULL, distinct-per-call, never
                // dereferenceable — same observable semantics as
                // upstream's int-encoded cast without needing the
                // llmemory/rffi int-to-ptr machinery.
                let LowLevelType::Ptr(boxed) = &self.lltype else {
                    return Err(TyperError::message(
                        "FunctionsPBCRepr.convert_desc: single-row lltype is not Ptr",
                    ));
                };
                _ptr::new_with_solid(
                    (**boxed).clone(),
                    Err(DelayedPointer),
                    /* _solid = */ true,
                )
            }
        } else {
            // upstream `self.create_specfunc()` ≡ `malloc(self.lowleveltype.TO,
            // immortal=True)` (rpbc.py:249-250). Here `self.lltype` is
            // `Ptr(Struct('specfunc', ...))`, so the `.TO` peels back to
            // the Struct.
            let LowLevelType::Ptr(struct_ptr) = &self.lltype else {
                return Err(TyperError::message(
                    "FunctionsPBCRepr.convert_desc: multi-row lltype is not Ptr(Struct)",
                ));
            };
            let PtrTarget::Struct(struct_t) = &struct_ptr.TO else {
                return Err(TyperError::message(
                    "FunctionsPBCRepr.convert_desc: multi-row lltype.TO is not Struct",
                ));
            };
            let struct_lltype = LowLevelType::Struct(Box::new(struct_t.clone()));
            let mut specfunc_ptr = ll_malloc(struct_lltype, None, MallocFlavor::Raw, true)
                .map_err(TyperError::message)?;
            // upstream `for attrname, llfn in llfns.items(): setattr(result, attrname, llfn)`.
            for (attrname_opt, llfn) in &llfns {
                let attrname = attrname_opt.as_deref().ok_or_else(|| {
                    TyperError::message(
                        "FunctionsPBCRepr.convert_desc: multi-row uniquerow has no attrname",
                    )
                })?;
                let llfn_value =
                    crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Ptr(Box::new(
                        llfn.clone(),
                    ));
                specfunc_ptr
                    .setattr(attrname, llfn_value)
                    .map_err(TyperError::message)?;
            }
            specfunc_ptr
        };

        // upstream `self.funccache[funcdesc] = result; return result`.
        let c = crate::translator::rtyper::rmodel::inputconst_from_lltype(
            &self.lltype,
            &ConstValue::LLPtr(Box::new(result_ptr)),
        )?;
        self.funccache.borrow_mut().insert(desc_rowkey, c.clone());
        Ok(c)
    }

    /// RPython `CanBeNull.rtype_bool(self, hop)` (rmodel.py:255-260) —
    /// `FunctionsPBCRepr` inherits this via `class FunctionsPBCRepr(CanBeNull,
    /// FunctionReprBase)` (rpbc.py:224). Delegates to
    /// [`crate::translator::rtyper::rmodel::can_be_null_rtype_bool`] which
    /// emits the constant fast-path or a `ptr_nonzero` guard.
    fn rtype_bool(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        crate::translator::rtyper::rmodel::can_be_null_rtype_bool(self, hop)
    }

    /// RPython `FunctionsPBCRepr.convert_const(self, value)`
    /// (rpbc.py:289-298).
    ///
    /// Ports the `value is None` arm: upstream emits
    /// `nullptr(self.lowleveltype.TO)`. The MethodType /
    /// staticmethod unwrapping and the `bookkeeper.getdesc(value)`
    /// dispatch to `convert_desc` both defer pending host-object
    /// infrastructure.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if matches!(value, ConstValue::None) {
            let LowLevelType::Ptr(ptr) = &self.lltype else {
                return Err(TyperError::message(
                    "FunctionsPBCRepr.convert_const: lowleveltype is not Ptr",
                ));
            };
            let null_ptr = ptr.as_ref().clone()._defl();
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(null_ptr)),
                self.lltype.clone(),
            ));
        }
        Err(TyperError::missing_rtype_operation(
            "FunctionsPBCRepr.convert_const (rpbc.py:290-298) \
             non-None branch requires bookkeeper.getdesc dispatch",
        ))
    }

    /// RPython `FunctionReprBase.rtype_simple_call(self, hop)`
    /// (rpbc.py:193-194) — `return self.call(hop)`. Inherited by
    /// `FunctionsPBCRepr` via `class FunctionsPBCRepr(CanBeNull,
    /// FunctionReprBase)`.
    fn rtype_simple_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.call(hop)
    }

    /// RPython `FunctionReprBase.rtype_call_args(self, hop)`
    /// (rpbc.py:196-197) — `return self.call(hop)`. Inherited by
    /// `FunctionsPBCRepr` via `class FunctionsPBCRepr(CanBeNull,
    /// FunctionReprBase)`.
    fn rtype_call_args(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.call(hop)
    }
}

/// RPython `class SingleFrozenPBCRepr(Repr)` (rpbc.py:635-662).
///
/// Void-typed repr for a single-FrozenDesc `SomePBC` with
/// `can_be_None=False`. `convert_desc` asserts desc-identity with the
/// stored `frozendesc` then emits `Constant(None, Void)`; `convert_const`
/// is the same sentinel (upstream `return None`).
#[derive(Debug)]
pub struct SingleFrozenPBCRepr {
    /// RPython `self.frozendesc = frozendesc` (rpbc.py:639-640).
    pub frozendesc: DescEntry,
    state: ReprState,
    lltype: LowLevelType,
}

impl SingleFrozenPBCRepr {
    /// RPython `SingleFrozenPBCRepr.__init__(self, frozendesc)`
    /// (rpbc.py:639-640).
    pub fn new(frozendesc: DescEntry) -> Self {
        SingleFrozenPBCRepr {
            frozendesc,
            state: ReprState::new(),
            lltype: LowLevelType::Void,
        }
    }
}

impl Repr for SingleFrozenPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "SingleFrozenPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::SingleFrozenPBCRepr
    }

    /// RPython `SingleFrozenPBCRepr.convert_desc(self, frozendesc)`
    /// (rpbc.py:647-649):
    ///
    /// ```python
    /// def convert_desc(self, frozendesc):
    ///     assert frozendesc is self.frozendesc
    ///     return object()  # lowleveltype is Void
    /// ```
    ///
    /// Pyre's identity check uses [`DescEntry::desc_key`] which wraps
    /// `Rc::as_ptr` — equivalent to upstream's `frozendesc is
    /// self.frozendesc`. The sentinel is emitted as a `Constant(None,
    /// Void)` so `convert_desc_or_const` gets a typed return instead
    /// of upstream's untagged Python `object()`.
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        if desc.desc_key() != self.frozendesc.desc_key() {
            return Err(TyperError::message(
                "SingleFrozenPBCRepr.convert_desc: frozendesc identity mismatch",
            ));
        }
        Ok(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    }

    /// RPython `SingleFrozenPBCRepr.convert_const(self, value)`
    /// (rpbc.py:651-652): upstream `return None`. Mirrors
    /// `FunctionRepr.convert_const` — Void sentinel.
    fn convert_const(&self, _value: &ConstValue) -> Result<Constant, TyperError> {
        Ok(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ))
    }

    /// RPython `SingleFrozenPBCRepr.rtype_getattr(_, hop)`
    /// (rpbc.py:642-645):
    ///
    /// ```python
    /// def rtype_getattr(_, hop):
    ///     if not hop.s_result.is_constant():
    ///         raise TyperError("getattr on a constant PBC returns a non-constant")
    ///     return hop.inputconst(hop.r_result, hop.s_result.const)
    /// ```
    ///
    /// Structurally identical to
    /// [`MultipleUnrelatedFrozenPBCRepr::rtype_getattr`] — Void-typed
    /// reprs carry no runtime fields, so `getattr` is only valid when
    /// the annotator can hand back a constant.
    fn rtype_getattr(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        let s_const = hop
            .s_result
            .borrow()
            .as_ref()
            .and_then(|s| s.const_())
            .cloned();
        let Some(value) = s_const else {
            return Err(TyperError::message(
                "getattr on a constant PBC returns a non-constant",
            ));
        };
        let r_result = hop
            .r_result
            .borrow()
            .clone()
            .ok_or_else(|| TyperError::message("HighLevelOp.r_result is not set"))?;
        let c = crate::translator::rtyper::rtyper::HighLevelOp::inputconst(&*r_result, &value)?;
        Ok(Some(crate::flowspace::model::Hlvalue::Constant(c)))
    }
}

/// RPython `class MultipleUnrelatedFrozenPBCRepr(MultipleFrozenPBCReprBase)`
/// (rpbc.py:675-711).
///
/// Representation for a `SomePBC` of frozen PBCs that have no common
/// access set. The only operation upstream allows is `is` comparison,
/// lowered to `adr_eq` at pairtype dispatch.
///
/// ```python
/// class MultipleUnrelatedFrozenPBCRepr(MultipleFrozenPBCReprBase):
///     lowleveltype = llmemory.Address
///     EMPTY = Struct('pbc', hints={'immutable': True, 'static_immutable': True})
///
///     def __init__(self, rtyper):
///         self.rtyper = rtyper
///         self.converted_pbc_cache = {}
///
///     def null_instance(self):
///         return llmemory.Address._defl()
/// ```
///
/// Pyre currently lands the shell: lowleveltype is
/// [`LowLevelType::Address`], `null_instance` returns the
/// [`LowLevelValue::Address`] NULL sentinel, and `convert_desc` /
/// `convert_const` / `convert_pbc` / `create_instance` / `rtype_getattr`
/// all surface [`TyperError::missing_rtype_operation`] until
/// `Bookkeeper::getdesc` + `rtyper.getrepr` + `fakeaddress` body land.
#[derive(Debug)]
pub struct MultipleUnrelatedFrozenPBCRepr {
    /// RPython `self.rtyper = rtyper` (rpbc.py:682).
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.converted_pbc_cache = {}` (rpbc.py:683).
    /// Keyed by `DescEntry::desc_key()` for pointer-identity semantics
    /// matching upstream's dict-by-Python-identity. Value is the
    /// `_address` produced by `convert_pbc(pbcptr)` —
    /// `LowLevelValue::Address(_)` so callers can surface it
    /// directly as a Constant.
    pub converted_pbc_cache: RefCell<
        HashMap<
            crate::annotator::description::DescKey,
            crate::translator::rtyper::lltypesystem::lltype::_address,
        >,
    >,
    state: ReprState,
    lltype: LowLevelType,
}

impl MultipleUnrelatedFrozenPBCRepr {
    /// RPython `MultipleUnrelatedFrozenPBCRepr.__init__(self, rtyper)`
    /// (rpbc.py:681-683).
    pub fn new(rtyper: &Rc<RPythonTyper>) -> Self {
        MultipleUnrelatedFrozenPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            converted_pbc_cache: RefCell::new(HashMap::new()),
            state: ReprState::new(),
            lltype: LowLevelType::Address,
        }
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.create_instance(self)`
    /// (rpbc.py:702-703):
    ///
    /// ```python
    /// EMPTY = Struct('pbc', hints={'immutable': True, 'static_immutable': True})
    /// def create_instance(self):
    ///     return malloc(self.EMPTY, immortal=True)
    /// ```
    ///
    /// Allocates a fresh empty `Struct('pbc', ...)` carrier. Used by
    /// [`Self::convert_desc`] when the matching SomePBC desc's repr
    /// is `Void` (the desc carries no fields, so a placeholder
    /// allocation is needed for `fakeaddress` to point at).
    pub fn create_instance(&self) -> Result<_ptr, TyperError> {
        // upstream `Struct('pbc', hints={'immutable': True,
        // 'static_immutable': True})` — fields-empty struct, Raw flavor
        // (default for `Struct(...)` without `Gc` prefix).
        let body = crate::translator::rtyper::lltypesystem::lltype::StructType::with_hints(
            "pbc",
            vec![],
            vec![
                ("immutable".into(), ConstValue::Bool(true)),
                ("static_immutable".into(), ConstValue::Bool(true)),
            ],
        );
        let body_ty = LowLevelType::Struct(Box::new(body));
        crate::translator::rtyper::lltypesystem::lltype::malloc(
            body_ty,
            None,
            crate::translator::rtyper::lltypesystem::lltype::MallocFlavor::Raw,
            true,
        )
        .map_err(TyperError::message)
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.convert_pbc(self, pbcptr)`
    /// (rpbc.py:699-700):
    ///
    /// ```python
    /// def convert_pbc(self, pbcptr):
    ///     return llmemory.fakeaddress(pbcptr)
    /// ```
    ///
    /// Wraps a live `_ptr` in [`_address::Fake`] so it flows through
    /// `Address`-typed slots. Used by [`Self::convert_desc`] after
    /// the per-frozendesc repr produced its `_ptr` value.
    pub fn convert_pbc(
        &self,
        pbcptr: _ptr,
    ) -> crate::translator::rtyper::lltypesystem::lltype::_address {
        crate::translator::rtyper::lltypesystem::lltype::_address::Fake(Box::new(pbcptr))
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.null_instance(self)`
    /// (rpbc.py:705-706):
    ///
    /// ```python
    /// def null_instance(self):
    ///     return llmemory.Address._defl()
    /// ```
    ///
    /// Returns the NULL fakeaddress sentinel. Consumers use this for
    /// the `can_be_None` fallback in `convert_const` and for pairtype
    /// dispatch defaults.
    pub fn null_instance(&self) -> Constant {
        let value = match LowLevelType::Address._defl() {
            crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Address(addr) => {
                ConstValue::LLAddress(addr)
            }
            other => {
                unreachable!("Address._defl() must yield LowLevelValue::Address, got {other:?}")
            }
        };
        Constant::with_concretetype(value, LowLevelType::Address)
    }
}

impl Repr for MultipleUnrelatedFrozenPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "MultipleUnrelatedFrozenPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::MultipleUnrelatedFrozenPBCRepr
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.convert_desc(self, frozendesc)`
    /// (rpbc.py:685-697):
    ///
    /// ```python
    /// def convert_desc(self, frozendesc):
    ///     try:
    ///         return self.converted_pbc_cache[frozendesc]
    ///     except KeyError:
    ///         r = self.rtyper.getrepr(annmodel.SomePBC([frozendesc]))
    ///         if r.lowleveltype is Void:
    ///             pbc = self.create_instance()
    ///         else:
    ///             pbc = r.convert_desc(frozendesc)
    ///         convpbc = self.convert_pbc(pbc)
    ///         self.converted_pbc_cache[frozendesc] = convpbc
    ///         return convpbc
    /// ```
    ///
    /// Routes the frozen desc through its concrete per-desc repr,
    /// extracts the `_ptr`, and wraps it via `convert_pbc` (=
    /// `fakeaddress`) into the `Address`-typed cache.
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        let desc_key = desc.desc_key();
        if let Some(cached) = self.converted_pbc_cache.borrow().get(&desc_key) {
            return Ok(Constant::with_concretetype(
                ConstValue::LLAddress(cached.clone()),
                LowLevelType::Address,
            ));
        }

        let DescEntry::Frozen(_) = desc else {
            return Err(TyperError::message(format!(
                "MultipleUnrelatedFrozenPBCRepr.convert_desc: non-Frozen desc {desc:?}"
            )));
        };

        // upstream `r = self.rtyper.getrepr(annmodel.SomePBC([frozendesc]))`.
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message(
                "MultipleUnrelatedFrozenPBCRepr.convert_desc: rtyper weak ref dropped",
            )
        })?;
        let s_pbc = SomePBC::new(vec![desc.clone()], false);
        let r = rtyper.getrepr(&crate::annotator::model::SomeValue::PBC(s_pbc))?;

        // upstream `if r.lowleveltype is Void: pbc = self.create_instance()
        // else: pbc = r.convert_desc(frozendesc)`.
        let pbc = if matches!(r.lowleveltype(), LowLevelType::Void) {
            self.create_instance()?
        } else {
            let converted = r.convert_desc(desc)?;
            match converted.value {
                ConstValue::LLPtr(p) => *p,
                other => {
                    return Err(TyperError::message(format!(
                        "MultipleUnrelatedFrozenPBCRepr.convert_desc: per-desc repr \
                         {} returned non-LLPtr value {other:?}",
                        r.class_name()
                    )));
                }
            }
        };

        // upstream `convpbc = self.convert_pbc(pbc); self.converted_pbc_cache[frozendesc] = convpbc`.
        let convpbc = self.convert_pbc(pbc);
        self.converted_pbc_cache
            .borrow_mut()
            .insert(desc_key, convpbc.clone());
        Ok(Constant::with_concretetype(
            ConstValue::LLAddress(convpbc),
            LowLevelType::Address,
        ))
    }

    /// RPython `MultipleFrozenPBCReprBase.convert_const(self, pbc)`
    /// (rpbc.py:666-670):
    ///
    /// ```python
    /// def convert_const(self, pbc):
    ///     if pbc is None:
    ///         return self.null_instance()
    ///     frozendesc = self.rtyper.annotator.bookkeeper.getdesc(pbc)
    ///     return self.convert_desc(frozendesc)
    /// ```
    ///
    /// `None` → `null_instance()` (NULL fakeaddress). Non-None
    /// HostObject → `bk.getdesc(host) → convert_desc(...)`.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if matches!(value, ConstValue::None) {
            return Ok(self.null_instance());
        }
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "MultipleUnrelatedFrozenPBCRepr.convert_const: non-None value must \
                 be a HostObject (frozen pbc), got {value:?}"
            )));
        };
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message(
                "MultipleUnrelatedFrozenPBCRepr.convert_const: rtyper weak ref dropped",
            )
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message(
                "MultipleUnrelatedFrozenPBCRepr.convert_const: annotator weak ref dropped",
            )
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        self.convert_desc(&desc)
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.rtype_getattr(_, hop)`
    /// (rpbc.py:708-711):
    ///
    /// ```python
    /// def rtype_getattr(_, hop):
    ///     if not hop.s_result.is_constant():
    ///         raise TyperError("getattr on a constant PBC returns a non-constant")
    ///     return hop.inputconst(hop.r_result, hop.s_result.const)
    /// ```
    ///
    /// A MU repr carries no structural fields, so upstream only allows
    /// getattr when the annotator has already proved the result constant.
    fn rtype_getattr(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        let s_const = hop
            .s_result
            .borrow()
            .as_ref()
            .and_then(|s| s.const_())
            .cloned();
        let Some(value) = s_const else {
            return Err(TyperError::message(
                "getattr on a constant PBC returns a non-constant",
            ));
        };
        let r_result = hop
            .r_result
            .borrow()
            .clone()
            .ok_or_else(|| TyperError::message("HighLevelOp.r_result is not set"))?;
        let c = crate::translator::rtyper::rtyper::HighLevelOp::inputconst(&*r_result, &value)?;
        Ok(Some(crate::flowspace::model::Hlvalue::Constant(c)))
    }
}

/// RPython `pairtype(MultipleUnrelatedFrozenPBCRepr,
/// MultipleUnrelatedFrozenPBCRepr).rtype_is_` (rpbc.py:713-725).
///
/// ```python
/// def rtype_is_((robj1, robj2), hop):
///     if isinstance(robj1, MultipleUnrelatedFrozenPBCRepr):
///         r = robj1
///     else:
///         r = robj2
///     vlist = hop.inputargs(r, r)
///     return hop.genop('adr_eq', vlist, resulttype=Bool)
/// ```
///
/// Both operands are already MU repr, so the receiver selection is a
/// no-op — `r1` is used directly. The upstream block also covers
/// `pairtype(MU, SingleFrozen)` / `pairtype(SingleFrozen, MU)` shapes,
/// which require a `SingleFrozen -> MU` convert_from_to path (null
/// address materialisation) that is not yet ported; those arms still
/// fall through to the generic `(Repr, Repr)` dispatcher and surface
/// `is of instances of the non-pointers` TyperError until the
/// conversion helper lands.
pub fn pair_mu_mu_rtype_is_(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
    use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult};
    let v_list = hop.inputargs(vec![ConvertedTo::Repr(r1), ConvertedTo::Repr(r2)])?;
    Ok(hop
        .genop("adr_eq", v_list, GenopResult::LLType(LowLevelType::Bool))
        .expect("adr_eq with Bool result returns a value"))
}

/// RPython `class MultipleFrozenPBCRepr(MultipleFrozenPBCReprBase)`
/// (rpbc.py:728-800).
///
/// Representation for a SomePBC of frozen PBCs that share a common
/// `ClassAttrFamily`-style access set. The vtable layout is a custom
/// `Struct('pbc', ...)` whose fields are one per attribute the access
/// set tracks; each frozen desc materialises into an immortal
/// allocation of that struct, with attribute values written eagerly
/// from the matching `frozendesc.attrcache`.
///
/// ```python
/// class MultipleFrozenPBCRepr(MultipleFrozenPBCReprBase):
///     def __init__(self, rtyper, access_set):
///         self.rtyper = rtyper
///         self.access_set = access_set
///         self.pbc_type = ForwardReference()
///         self.lowleveltype = Ptr(self.pbc_type)
///         self.pbc_cache = {}
///
///     def _setup_repr(self):
///         llfields = self._setup_repr_fields()
///         kwds = {'hints': {'immutable': True, 'static_immutable': True}}
///         self.pbc_type.become(Struct('pbc', *llfields, **kwds))
///     ...
/// ```
#[derive(Debug)]
pub struct MultipleFrozenPBCRepr {
    /// RPython `self.rtyper = rtyper` (rpbc.py:732).
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.access_set = access_set` (rpbc.py:733).
    pub access_set: Rc<RefCell<crate::annotator::description::FrozenAttrFamily>>,
    /// RPython `self.pbc_type = ForwardReference()` (rpbc.py:734).
    /// Stored as a clone alongside [`Self::lltype`] so callers can
    /// observe the resolved struct after `_setup_repr` runs `become`
    /// against either reference (Rust's `ForwardReference` shares its
    /// `Arc<Mutex<…>>` slot across clones).
    pbc_type: crate::translator::rtyper::lltypesystem::lltype::ForwardReference,
    /// RPython `self.lowleveltype = Ptr(self.pbc_type)` (rpbc.py:735).
    lltype: LowLevelType,
    /// RPython `self.pbc_cache = {}` (rpbc.py:736). Caches the
    /// per-frozendesc materialised `_ptr` so repeated `convert_desc`
    /// calls return identity-equal pointers and `convert_const` cycles
    /// terminate.
    pbc_cache: RefCell<HashMap<crate::annotator::description::DescKey, _ptr>>,
    /// RPython `self.fieldmap` populated by `_setup_repr_fields`
    /// (rpbc.py:757). Maps each tracked attrname to its mangled struct
    /// field name + the per-attr Repr for the value type.
    fieldmap: RefCell<HashMap<String, (String, Arc<dyn Repr>)>>,
    state: ReprState,
}

impl MultipleFrozenPBCRepr {
    /// RPython `MultipleFrozenPBCRepr.__init__(self, rtyper, access_set)`
    /// (rpbc.py:731-736).
    pub fn new(
        rtyper: &Rc<RPythonTyper>,
        access_set: Rc<RefCell<crate::annotator::description::FrozenAttrFamily>>,
    ) -> Self {
        let pbc_type = crate::translator::rtyper::lltypesystem::lltype::ForwardReference::new();
        let pbc_type_clone = pbc_type.clone();
        let lltype = LowLevelType::Ptr(Box::new(
            crate::translator::rtyper::lltypesystem::lltype::Ptr {
                TO: crate::translator::rtyper::lltypesystem::lltype::PtrTarget::ForwardReference(
                    pbc_type_clone,
                ),
            },
        ));
        MultipleFrozenPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            access_set,
            pbc_type,
            lltype,
            pbc_cache: RefCell::new(HashMap::new()),
            fieldmap: RefCell::new(HashMap::new()),
            state: ReprState::new(),
        }
    }

    /// RPython `MultipleFrozenPBCRepr._setup_repr_fields(self)`
    /// (rpbc.py:755-767):
    ///
    /// ```python
    /// def _setup_repr_fields(self):
    ///     fields = []
    ///     self.fieldmap = {}
    ///     if self.access_set is not None:
    ///         attrlist = self.access_set.attrs.keys()
    ///         attrlist.sort()
    ///         for attr in attrlist:
    ///             s_value = self.access_set.attrs[attr]
    ///             r_value = self.rtyper.getrepr(s_value)
    ///             mangled_name = mangle('pbc', attr)
    ///             fields.append((mangled_name, r_value.lowleveltype))
    ///             self.fieldmap[attr] = mangled_name, r_value
    ///     return fields
    /// ```
    fn setup_repr_fields(&self) -> Result<Vec<(String, LowLevelType)>, TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("MultipleFrozenPBCRepr._setup_repr_fields: rtyper weak ref dropped")
        })?;
        // upstream `attrlist = self.access_set.attrs.keys(); attrlist.sort()`.
        let attrs_snapshot: Vec<(String, crate::annotator::model::SomeValue)> = {
            let caf = self.access_set.borrow();
            let mut attrs: Vec<(String, crate::annotator::model::SomeValue)> = caf
                .attrs
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            attrs.sort_by(|a, b| a.0.cmp(&b.0));
            attrs
        };
        let mut fields = Vec::with_capacity(attrs_snapshot.len());
        let mut new_fieldmap: HashMap<String, (String, Arc<dyn Repr>)> = HashMap::new();
        for (attr, s_value) in attrs_snapshot {
            let r_value = rtyper.getrepr(&s_value)?;
            let mangled_name = crate::translator::rtyper::rmodel::mangle("pbc", &attr);
            fields.push((mangled_name.clone(), r_value.lowleveltype().clone()));
            new_fieldmap.insert(attr, (mangled_name, r_value));
        }
        *self.fieldmap.borrow_mut() = new_fieldmap;
        Ok(fields)
    }

    /// RPython `MultipleFrozenPBCRepr.create_instance(self)` (rpbc.py:743-744):
    ///
    /// ```python
    /// def create_instance(self):
    ///     return malloc(self.pbc_type, immortal=True)
    /// ```
    pub fn create_instance(&self) -> Result<_ptr, TyperError> {
        let resolved = self.pbc_type.resolved().ok_or_else(|| {
            TyperError::message(
                "MultipleFrozenPBCRepr.create_instance: pbc_type ForwardReference \
                 not resolved — call setup() first",
            )
        })?;
        crate::translator::rtyper::lltypesystem::lltype::malloc(
            resolved,
            None,
            crate::translator::rtyper::lltypesystem::lltype::MallocFlavor::Raw,
            true,
        )
        .map_err(TyperError::message)
    }

    /// RPython `MultipleFrozenPBCRepr.null_instance(self)` (rpbc.py:746-747):
    ///
    /// ```python
    /// def null_instance(self):
    ///     return nullptr(self.pbc_type)
    /// ```
    pub fn null_instance(&self) -> Result<_ptr, TyperError> {
        let LowLevelType::Ptr(ptr) = &self.lltype else {
            return Err(TyperError::message(
                "MultipleFrozenPBCRepr.null_instance: lowleveltype is not Ptr",
            ));
        };
        Ok(ptr.as_ref().clone()._defl())
    }

    /// RPython `MultipleFrozenPBCRepr.getfield(self, vpbc, attr, llops)`
    /// (rpbc.py:749-753):
    ///
    /// ```python
    /// def getfield(self, vpbc, attr, llops):
    ///     mangled_name, r_value = self.fieldmap[attr]
    ///     cmangledname = inputconst(Void, mangled_name)
    ///     return llops.genop('getfield', [vpbc, cmangledname], resulttype=r_value)
    /// ```
    pub fn getfield(
        &self,
        vpbc: Hlvalue,
        attr: &str,
        llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
    ) -> Result<crate::flowspace::model::Variable, TyperError> {
        let (mangled_name, r_value) =
            self.fieldmap.borrow().get(attr).cloned().ok_or_else(|| {
                TyperError::message(format!(
                    "MultipleFrozenPBCRepr.getfield: attr {attr:?} not in fieldmap"
                ))
            })?;
        let cname =
            Constant::with_concretetype(ConstValue::byte_str(mangled_name), LowLevelType::Void);
        Ok(llops
            .genop(
                "getfield",
                vec![vpbc, Hlvalue::Constant(cname)],
                crate::translator::rtyper::rtyper::GenopResult::LLType(
                    r_value.lowleveltype().clone(),
                ),
            )
            .expect("getfield with non-Void result yields a Variable"))
    }
}

impl Repr for MultipleFrozenPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "MultipleFrozenPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::MultipleFrozenPBCRepr
    }

    /// RPython `MultipleFrozenPBCRepr._setup_repr(self)` (rpbc.py:738-741):
    ///
    /// ```python
    /// def _setup_repr(self):
    ///     llfields = self._setup_repr_fields()
    ///     kwds = {'hints': {'immutable': True, 'static_immutable': True}}
    ///     self.pbc_type.become(Struct('pbc', *llfields, **kwds))
    /// ```
    fn _setup_repr(&self) -> Result<(), TyperError> {
        let llfields = self.setup_repr_fields()?;
        let body = crate::translator::rtyper::lltypesystem::lltype::StructType::with_hints(
            "pbc",
            llfields,
            vec![
                ("immutable".into(), ConstValue::Bool(true)),
                ("static_immutable".into(), ConstValue::Bool(true)),
            ],
        );
        self.pbc_type
            .r#become(LowLevelType::Struct(Box::new(body)))
            .map_err(TyperError::message)?;
        Ok(())
    }

    /// RPython `MultipleFrozenPBCRepr.convert_desc(self, frozendesc)`
    /// (rpbc.py:769-790):
    ///
    /// ```python
    /// def convert_desc(self, frozendesc):
    ///     if (self.access_set is not None and
    ///             frozendesc not in self.access_set.descs):
    ///         raise TyperError("not found in PBC access set: %r" % (frozendesc,))
    ///     try:
    ///         return self.pbc_cache[frozendesc]
    ///     except KeyError:
    ///         self.setup()
    ///         result = self.create_instance()
    ///         self.pbc_cache[frozendesc] = result
    ///         for attr, (mangled_name, r_value) in self.fieldmap.items():
    ///             if r_value.lowleveltype is Void:
    ///                 continue
    ///             try:
    ///                 thisattrvalue = frozendesc.attrcache[attr]
    ///             except KeyError:
    ///                 if frozendesc.warn_missing_attribute(attr):
    ///                     warning("Desc %r has no attribute %r" % (frozendesc, attr))
    ///                 continue
    ///             llvalue = r_value.convert_const(thisattrvalue)
    ///             setattr(result, mangled_name, llvalue)
    ///         return result
    /// ```
    ///
    /// Cache-before-init order matches upstream rpbc.py:778 (parity
    /// with the TupleRepr / InstanceRepr fixes from the recent reviewer
    /// pass): the `_ptr` is inserted into `pbc_cache` before any
    /// recursive `r_value.convert_const(...)` so cyclic frozen graphs
    /// terminate via the cached entry.
    fn convert_desc(
        &self,
        desc: &crate::annotator::description::DescEntry,
    ) -> Result<Constant, TyperError> {
        // upstream `if (self.access_set is not None and
        // frozendesc not in self.access_set.descs)` — pyre's
        // `access_set` is always Some on this concrete repr (the
        // `None` arm in upstream describes the abstract base class
        // before the per-attr family is known), so the membership
        // check is unconditional here.
        //
        // PRE-EXISTING-DEVIATION: pyre keys
        // `FrozenAttrFamily.descs` on `FrozenDesc.base.identity`
        // (counter-based DescKey) because that is the key
        // `frozenpbc_attr_families` UnionFind uses; the parallel
        // `DescEntry::desc_key()` returns the `Rc::as_ptr`-based
        // identity used elsewhere. Look up by `base.identity` so
        // membership matches what `mergeattrfamilies` populated.
        let crate::annotator::description::DescEntry::Frozen(fd_rc) = desc else {
            return Err(TyperError::message(format!(
                "MultipleFrozenPBCRepr.convert_desc: non-Frozen desc {desc:?}"
            )));
        };
        let identity_key = fd_rc.borrow().base.identity;
        if !self.access_set.borrow().descs.contains_key(&identity_key) {
            return Err(TyperError::message(format!(
                "MultipleFrozenPBCRepr.convert_desc: {desc:?} not found in PBC access set"
            )));
        }
        let desc_key = desc.desc_key();

        // upstream `return self.pbc_cache[frozendesc]` fast path.
        if let Some(cached) = self.pbc_cache.borrow().get(&desc_key) {
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(cached.clone())),
                self.lltype.clone(),
            ));
        }

        // upstream `self.setup()`. Repr::setup is idempotent.
        Repr::setup(self as &dyn Repr)?;

        // upstream `result = self.create_instance(); self.pbc_cache[frozendesc] = result`.
        let result = self.create_instance()?;
        self.pbc_cache.borrow_mut().insert(desc_key, result);

        // upstream loop over fieldmap, writing each attr's converted
        // value into the cached pointer through brief borrow_mut bursts.
        let frozendesc = fd_rc.clone();
        let fieldmap_snapshot: Vec<(String, String, Arc<dyn Repr>)> = self
            .fieldmap
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.0.clone(), v.1.clone()))
            .collect();
        for (attr, mangled_name, r_value) in fieldmap_snapshot {
            if matches!(r_value.lowleveltype(), LowLevelType::Void) {
                continue;
            }
            // upstream `frozendesc.attrcache[attr]`. Missing attrs
            // emit a warning upstream; pyre swallows the miss
            // silently — `warn_missing_attribute` is not yet ported
            // and the prebuilt-instance path tolerates partial fills.
            let attrvalue = match frozendesc.borrow().attrcache.borrow().get(&attr) {
                Some(v) => v.clone(),
                None => continue,
            };
            let item_const = r_value.convert_const(&attrvalue)?;
            let llval = crate::translator::rtyper::rclass::constant_to_lowlevel_value(&item_const)?;
            let mut cache = self.pbc_cache.borrow_mut();
            let result_in_cache = cache.get_mut(&desc_key).ok_or_else(|| {
                TyperError::message(
                    "MultipleFrozenPBCRepr.convert_desc: cached pointer disappeared mid-init",
                )
            })?;
            result_in_cache
                .setattr(&mangled_name, llval)
                .map_err(TyperError::message)?;
        }

        // upstream `return result` — clone out the cached entry as a
        // Constant.
        let cached = self
            .pbc_cache
            .borrow()
            .get(&desc_key)
            .cloned()
            .ok_or_else(|| {
                TyperError::message(
                    "MultipleFrozenPBCRepr.convert_desc: cached pointer missing on return",
                )
            })?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(cached)),
            self.lltype.clone(),
        ))
    }

    /// RPython `MultipleFrozenPBCReprBase.convert_const(self, pbc)`
    /// (rpbc.py:666-670):
    ///
    /// ```python
    /// def convert_const(self, pbc):
    ///     if pbc is None:
    ///         return self.null_instance()
    ///     frozendesc = self.rtyper.annotator.bookkeeper.getdesc(pbc)
    ///     return self.convert_desc(frozendesc)
    /// ```
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if matches!(value, ConstValue::None) {
            let null = self.null_instance()?;
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(null)),
                self.lltype.clone(),
            ));
        }
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "MultipleFrozenPBCRepr.convert_const: non-None value must be a \
                 HostObject (frozen pbc), got {value:?}"
            )));
        };
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("MultipleFrozenPBCRepr.convert_const: rtyper weak ref dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("MultipleFrozenPBCRepr.convert_const: annotator weak ref dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        self.convert_desc(&desc)
    }

    /// RPython `MultipleFrozenPBCRepr.rtype_getattr(self, hop)`
    /// (rpbc.py:792-800):
    ///
    /// ```python
    /// def rtype_getattr(self, hop):
    ///     if hop.s_result.is_constant():
    ///         return hop.inputconst(hop.r_result, hop.s_result.const)
    ///     attr = hop.args_s[1].const
    ///     vpbc, vattr = hop.inputargs(self, Void)
    ///     v_res = self.getfield(vpbc, attr, hop.llops)
    ///     mangled_name, r_res = self.fieldmap[attr]
    ///     return hop.llops.convertvar(v_res, r_res, hop.r_result)
    /// ```
    fn rtype_getattr(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rtyper::{ConvertedTo, HighLevelOp};

        // upstream const-result fast path.
        let const_fast_path: Option<(Arc<dyn Repr>, ConstValue)> = {
            let s_result_borrow = hop.s_result.borrow();
            let r_result_borrow = hop.r_result.borrow();
            match (s_result_borrow.as_ref(), r_result_borrow.as_ref()) {
                (Some(s_result), Some(r_result)) => {
                    s_result.const_().map(|cv| (r_result.clone(), cv.clone()))
                }
                _ => None,
            }
        };
        if let Some((r_result, const_val)) = const_fast_path {
            let c = HighLevelOp::inputconst(ConvertedTo::Repr(r_result.as_ref()), &const_val)?;
            return Ok(Some(Hlvalue::Constant(c)));
        }

        // upstream `attr = hop.args_s[1].const`.
        let attr: String = {
            let args_s = hop.args_s.borrow();
            let s_attr = args_s.get(1).cloned().ok_or_else(|| {
                TyperError::message("MultipleFrozenPBCRepr.rtype_getattr: missing args_s[1]")
            })?;
            match s_attr.const_().cloned() {
                Some(value) => value.as_text().unwrap_or("<non-string>").to_string(),
                other => {
                    return Err(TyperError::message(format!(
                        "MultipleFrozenPBCRepr.rtype_getattr: non-constant attribute name (got {other:?})"
                    )));
                }
            }
        };

        // upstream `vpbc, vattr = hop.inputargs(self, Void)`.
        let v_args = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::LowLevelType(&LowLevelType::Void),
        ])?;
        let vpbc = v_args[0].clone();

        // upstream `v_res = self.getfield(vpbc, attr, hop.llops)`.
        let v_res = {
            let mut llops = hop.llops.borrow_mut();
            self.getfield(vpbc, &attr, &mut *llops)?
        };

        // upstream `mangled_name, r_res = self.fieldmap[attr]; return
        // hop.llops.convertvar(v_res, r_res, hop.r_result)`.
        let r_res = self
            .fieldmap
            .borrow()
            .get(&attr)
            .map(|(_, r)| r.clone())
            .ok_or_else(|| {
                TyperError::message(format!(
                    "MultipleFrozenPBCRepr.rtype_getattr: attr {attr:?} not in fieldmap"
                ))
            })?;
        let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
            TyperError::message("MultipleFrozenPBCRepr.rtype_getattr: hop.r_result missing")
        })?;
        let result = {
            let mut llops = hop.llops.borrow_mut();
            llops.convertvar(Hlvalue::Variable(v_res), r_res.as_ref(), r_result.as_ref())?
        };
        Ok(Some(result))
    }
}

/// RPython `adjust_shape(hop2, s_shape)` (rpbc.py:1120-1124):
///
/// ```python
/// def adjust_shape(hop2, s_shape):
///     new_shape = (s_shape.const[0]+1,) + s_shape.const[1:]
///     c_shape = Constant(new_shape)
///     s_shape = hop2.rtyper.annotator.bookkeeper.immutablevalue(new_shape)
///     hop2.v_s_insertfirstarg(c_shape, s_shape) # reinsert adjusted shape
/// ```
///
/// `s_shape.const` is a `(shape_cnt, shape_keys, shape_star)` tuple
/// constant carried by `call_args` ops; the helper bumps `shape_cnt`
/// by 1 to account for the bound-self arg that
/// [`MethodOfFrozenPBCRepr::redispatch_call`] /
/// [`MethodsPBCRepr::redispatch_call`] (rpbc.py:894, :1195) prepend.
pub(super) fn adjust_shape(
    hop2: &crate::translator::rtyper::rtyper::HighLevelOp,
    s_shape: &crate::annotator::model::SomeValue,
) -> Result<(), TyperError> {
    use crate::annotator::model::SomeValue;
    use crate::flowspace::model::Constant as FlowConstant;

    // upstream: `s_shape.const` — pull the underlying tuple ConstValue.
    let shape_const = s_shape
        .const_()
        .cloned()
        .ok_or_else(|| TyperError::message("adjust_shape: s_shape is not a Constant"))?;
    let items = match &shape_const {
        ConstValue::Tuple(items) => items,
        other => {
            return Err(TyperError::message(format!(
                "adjust_shape: s_shape.const is not a Tuple: {other:?}"
            )));
        }
    };
    if items.len() != 3 {
        return Err(TyperError::message(
            "adjust_shape: shape tuple must have 3 elements",
        ));
    }
    // upstream: `(s_shape.const[0]+1,) + s_shape.const[1:]`.
    let bumped_cnt = match &items[0] {
        ConstValue::Int(n) => ConstValue::Int(n + 1),
        other => {
            return Err(TyperError::message(format!(
                "adjust_shape: shape_cnt is not Int: {other:?}"
            )));
        }
    };
    let new_shape = ConstValue::Tuple(vec![bumped_cnt, items[1].clone(), items[2].clone()]);
    // upstream: `c_shape = Constant(new_shape)` — Void-typed constant
    // (matches the encoding `bookkeeper::call_shape_from_const` reads).
    let c_shape = Hlvalue::Constant(FlowConstant::with_concretetype(
        new_shape.clone(),
        LowLevelType::Void,
    ));
    // upstream: `s_shape = hop2.rtyper.annotator.bookkeeper.immutablevalue(new_shape)`.
    let bookkeeper = hop2
        .rtyper
        .annotator
        .upgrade()
        .ok_or_else(|| TyperError::message("adjust_shape: annotator weak ref dropped"))?
        .bookkeeper
        .clone();
    let s_new_shape = bookkeeper
        .immutablevalue(&new_shape)
        .map_err(|e| TyperError::message(e.to_string()))?;
    // upstream: `hop2.v_s_insertfirstarg(c_shape, s_shape)`.
    hop2.v_s_insertfirstarg(c_shape, s_new_shape)?;
    // Suppress unused-import warning for SomeValue when only used in
    // the parameter type.
    let _ = SomeValue::Impossible;
    Ok(())
}

/// RPython `class MethodOfFrozenPBCRepr(Repr)` (rpbc.py:844-911).
///
/// ```python
/// class MethodOfFrozenPBCRepr(Repr):
///     """Representation selected for a PBC of method object(s) of frozen PBCs.
///     It assumes that all methods are the same function bound to different PBCs.
///     The low-level representation can then be a pointer to that PBC."""
///
///     def __init__(self, rtyper, s_pbc):
///         self.rtyper = rtyper
///         funcdescs = set([desc.funcdesc for desc in s_pbc.descriptions])
///         assert len(funcdescs) == 1
///         self.funcdesc = funcdescs.pop()
///         if s_pbc.can_be_none():
///             raise TyperError(...)
///         im_selves = [desc.frozendesc for desc in s_pbc.descriptions]
///         self.s_im_self = annmodel.SomePBC(im_selves)
///         self.r_im_self = rtyper.getrepr(self.s_im_self)
///         self.lowleveltype = self.r_im_self.lowleveltype
/// ```
#[derive(Debug)]
pub struct MethodOfFrozenPBCRepr {
    /// RPython `self.rtyper = rtyper` (rpbc.py:850).
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.funcdesc = funcdescs.pop()` (rpbc.py:853) — the
    /// single shared underlying `FunctionDesc` across all bound
    /// methods in `s_pbc`.
    pub funcdesc: Rc<RefCell<crate::annotator::description::FunctionDesc>>,
    /// RPython `self.s_im_self = SomePBC(im_selves)` (rpbc.py:867) —
    /// the PBC of the bound `frozendesc`s.
    pub s_im_self: SomePBC,
    /// RPython `self.r_im_self = rtyper.getrepr(self.s_im_self)`
    /// (rpbc.py:868).
    pub r_im_self: std::sync::Arc<dyn Repr>,
    /// RPython `self.lowleveltype = self.r_im_self.lowleveltype`
    /// (rpbc.py:869) — pointer-to-bound-PBC. Stored explicitly because
    /// `Repr::lowleveltype` returns `&LowLevelType` and we cannot
    /// borrow through `r_im_self`.
    lltype: LowLevelType,
    state: ReprState,
}

impl MethodOfFrozenPBCRepr {
    /// RPython `MethodOfFrozenPBCRepr.__init__(self, rtyper, s_pbc)`
    /// (rpbc.py:849-869).
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Self, TyperError> {
        // upstream: `funcdescs = set([desc.funcdesc for desc in
        //                              s_pbc.descriptions]); assert
        //            len(funcdescs) == 1; self.funcdesc =
        //            funcdescs.pop()`.
        let mut funcdesc_set: std::collections::BTreeMap<
            crate::annotator::description::DescKey,
            Rc<RefCell<crate::annotator::description::FunctionDesc>>,
        > = std::collections::BTreeMap::new();
        let mut frozendescs: Vec<crate::annotator::description::DescEntry> = Vec::new();
        for entry in s_pbc.descriptions.values() {
            let mof = entry.as_method_of_frozen().ok_or_else(|| {
                TyperError::message(
                    "MethodOfFrozenPBCRepr: every description must be a MethodOfFrozenDesc",
                )
            })?;
            let mof_b = mof.borrow();
            let fd = mof_b.funcdesc.clone();
            let fd_key = crate::annotator::description::DescKey::from_rc(&fd);
            funcdesc_set.entry(fd_key).or_insert(fd);
            frozendescs.push(crate::annotator::description::DescEntry::Frozen(
                mof_b.frozendesc.clone(),
            ));
        }
        if funcdesc_set.len() != 1 {
            return Err(TyperError::message(
                "MethodOfFrozenPBCRepr: all bound methods must share a single FunctionDesc",
            ));
        }
        let funcdesc = funcdesc_set.into_iter().next().expect("len checked").1;

        // upstream: `if s_pbc.can_be_none(): raise TyperError(...)`.
        if s_pbc.can_be_none {
            return Err(TyperError::message(
                "unsupported: variable of type method-of-frozen-PBC or None",
            ));
        }

        // upstream: `im_selves = [desc.frozendesc for desc in
        //                          s_pbc.descriptions];
        //            self.s_im_self = SomePBC(im_selves);
        //            self.r_im_self = rtyper.getrepr(self.s_im_self);
        //            self.lowleveltype = self.r_im_self.lowleveltype`.
        let s_im_self = SomePBC::new(frozendescs, false);
        let r_im_self =
            rtyper.getrepr(&crate::annotator::model::SomeValue::PBC(s_im_self.clone()))?;
        let lltype = r_im_self.lowleveltype().clone();

        Ok(MethodOfFrozenPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            funcdesc,
            s_im_self,
            r_im_self,
            lltype,
            state: ReprState::new(),
        })
    }

    /// RPython `MethodOfFrozenPBCRepr.convert_desc(self, mdesc)`
    /// (rpbc.py:878-882):
    ///
    /// ```python
    /// def convert_desc(self, mdesc):
    ///     if mdesc.funcdesc is not self.funcdesc:
    ///         raise TyperError("not a method bound on %r: %r" % ...)
    ///     return self.r_im_self.convert_desc(mdesc.frozendesc)
    /// ```
    pub fn convert_desc(
        &self,
        desc: &crate::annotator::description::DescEntry,
    ) -> Result<Constant, TyperError> {
        let mof = desc.as_method_of_frozen().ok_or_else(|| {
            TyperError::message(
                "MethodOfFrozenPBCRepr.convert_desc: desc is not a MethodOfFrozenDesc",
            )
        })?;
        let mof_b = mof.borrow();
        // upstream: `if mdesc.funcdesc is not self.funcdesc: raise`.
        if !Rc::ptr_eq(&mof_b.funcdesc, &self.funcdesc) {
            return Err(TyperError::message(format!(
                "not a method bound on {:?}: {:?}",
                self.funcdesc.borrow().name,
                mof_b.funcdesc.borrow().name,
            )));
        }
        // upstream: `return self.r_im_self.convert_desc(mdesc.frozendesc)`.
        let frozen_entry =
            crate::annotator::description::DescEntry::Frozen(mof_b.frozendesc.clone());
        self.r_im_self.convert_desc(&frozen_entry)
    }

    /// RPython `MethodOfFrozenPBCRepr.redispatch_call(self, hop,
    /// call_args)` (rpbc.py:894-911).
    ///
    /// ```python
    /// def redispatch_call(self, hop, call_args):
    ///     s_function = annmodel.SomePBC([self.funcdesc])
    ///     hop2 = hop.copy()
    ///     hop2.args_s[0] = self.s_im_self
    ///     hop2.args_r[0] = self.r_im_self
    ///     if isinstance(hop2.args_v[0], Constant):
    ///         boundmethod = hop2.args_v[0].value
    ///         hop2.args_v[0] = Constant(boundmethod.im_self)
    ///     if call_args:
    ///         hop2.swap_fst_snd_args()
    ///         _, s_shape = hop2.r_s_popfirstarg()
    ///         adjust_shape(hop2, s_shape)
    ///     c = Constant("obscure-don't-use-me")
    ///     hop2.v_s_insertfirstarg(c, s_function)
    ///     return hop2.dispatch()
    /// ```
    pub fn redispatch_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
        call_args: bool,
    ) -> Result<Option<Hlvalue>, TyperError> {
        use crate::annotator::description::DescEntry;
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::Constant as FlowConstant;

        // upstream: `s_function = SomePBC([self.funcdesc])`.
        let s_function = SomeValue::PBC(SomePBC::new(
            vec![DescEntry::Function(self.funcdesc.clone())],
            false,
        ));
        // upstream: `hop2 = hop.copy()`.
        let hop2 = hop.copy();
        // upstream: `hop2.args_s[0] = self.s_im_self;
        //            hop2.args_r[0] = self.r_im_self`.
        hop2.args_s.borrow_mut()[0] = SomeValue::PBC(self.s_im_self.clone());
        hop2.args_r.borrow_mut()[0] = Some(self.r_im_self.clone());

        // upstream:
        //     if isinstance(hop2.args_v[0], Constant):
        //         boundmethod = hop2.args_v[0].value
        //         hop2.args_v[0] = Constant(boundmethod.im_self)
        //
        // PRE-EXISTING-ADAPTATION: pyre's `ConstValue` does not yet
        // carry a "bound method" host-object variant from which
        // `boundmethod.im_self` can be extracted. The Constant arm of
        // upstream is a compile-time short-circuit that re-bases the
        // hop's first arg on the frozen instance's host-side
        // representation; pyre defers it until the bookkeeper carries
        // a `BoundMethod` Constant kind. The Variable arm runs
        // unmodified — `convertvar` will do the type adjustment under
        // `args_r[0] = self.r_im_self`.
        if matches!(hop2.args_v.borrow().get(0), Some(Hlvalue::Constant(_))) {
            return Err(TyperError::missing_rtype_operation(
                "MethodOfFrozenPBCRepr.redispatch_call (rpbc.py:900-902) \
                 Constant boundmethod.im_self extraction port pending",
            ));
        }

        // upstream: `if call_args: swap_fst_snd_args(); _, s_shape =
        //                          r_s_popfirstarg(); adjust_shape(hop2, s_shape)`.
        if call_args {
            hop2.swap_fst_snd_args();
            let (_r, s_shape) = hop2.r_s_popfirstarg();
            adjust_shape(&hop2, &s_shape)?;
        }

        // upstream: `c = Constant("obscure-don't-use-me")`. Void-typed
        // sentinel — the hop2 path never dereferences it, but the
        // dispatcher needs *some* args_v[0] to feed
        // `hop.inputarg(self_repr, 0)`. Upstream stores no
        // concretetype on this Constant; pyre tags Void to satisfy
        // genop's concretetype assertion downstream.
        let c = Hlvalue::Constant(FlowConstant::with_concretetype(
            ConstValue::byte_str("obscure-don't-use-me"),
            LowLevelType::Void,
        ));
        // upstream: `hop2.v_s_insertfirstarg(c, s_function)`.
        hop2.v_s_insertfirstarg(c, s_function)?;
        // upstream: `return hop2.dispatch()`.
        hop2.dispatch()
    }
}

impl Repr for MethodOfFrozenPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "MethodOfFrozenPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::MethodOfFrozenPBCRepr
    }

    /// RPython `MethodOfFrozenPBCRepr.get_r_implfunc(self)`
    /// (rpbc.py:874-876).
    fn get_r_implfunc(&self) -> Result<(&dyn Repr, usize), TyperError> {
        // upstream: `r_func = self.rtyper.getrepr(self.get_s_callable());
        //            return r_func, 1`.
        // We cannot return a `&dyn Repr` to a temporary Arc, so the
        // implfunc accessor is unsupported until callers route through
        // an owned-Arc API. Surface as a typed error so the missing
        // dependency is explicit.
        Err(TyperError::missing_rtype_operation(
            "MethodOfFrozenPBCRepr.get_r_implfunc (rpbc.py:874-876) \
             requires &dyn Repr from a freshly-built FunctionRepr",
        ))
    }

    /// RPython `MethodOfFrozenPBCRepr.convert_desc` (rpbc.py:878-882) —
    /// thin Repr-trait forwarder; the body lives on the inherent impl.
    fn convert_desc(
        &self,
        desc: &crate::annotator::description::DescEntry,
    ) -> Result<Constant, TyperError> {
        MethodOfFrozenPBCRepr::convert_desc(self, desc)
    }

    /// RPython `MethodOfFrozenPBCRepr.convert_const(self, method)`
    /// (rpbc.py:884-886).
    ///
    /// PRE-EXISTING-ADAPTATION: upstream calls `bookkeeper.getdesc(method)`
    /// on a host-side Python bound method. Pyre's bookkeeper accepts
    /// `HostObject` only; bound-method `HostObject`s are not yet
    /// produced anywhere in the pipeline. Defers to convert_desc by
    /// way of `getdesc` once that bridge lands.
    fn convert_const(&self, _value: &ConstValue) -> Result<Constant, TyperError> {
        Err(TyperError::missing_rtype_operation(
            "MethodOfFrozenPBCRepr.convert_const (rpbc.py:884-886) \
             requires bookkeeper.getdesc on a HostObject(BoundMethod)",
        ))
    }

    /// RPython `MethodOfFrozenPBCRepr.rtype_simple_call(self, hop)`
    /// (rpbc.py:888-889) — `return self.redispatch_call(hop, call_args=False)`.
    fn rtype_simple_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.redispatch_call(hop, false)
    }

    /// RPython `MethodOfFrozenPBCRepr.rtype_call_args(self, hop)`
    /// (rpbc.py:891-892) — `return self.redispatch_call(hop, call_args=True)`.
    fn rtype_call_args(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.redispatch_call(hop, true)
    }
}

/// RPython `class ClassesPBCRepr(Repr)` (rpbc.py:920-968).
///
/// PBC repr for a `SomePBC` whose kind is `DescKind::Class`. Upstream:
///
/// ```python
/// def __init__(self, rtyper, s_pbc):
///     self.rtyper = rtyper
///     self.s_pbc = s_pbc
///     if s_pbc.is_constant():
///         self.lowleveltype = Void
///     else:
///         self.lowleveltype = self.getlowleveltype()
///
/// def getlowleveltype(self):
///     return CLASSTYPE
/// ```
///
/// Both arms of `__init__` are now ported; `getlowleveltype()` returns
/// `CLASSTYPE` directly and is independent of `init_vtable`. The
/// downstream `convert_desc` non-Void path still routes to
/// `getruntime` which is blocked on `ClassRepr.init_vtable`.
#[derive(Debug)]
pub struct ClassesPBCRepr {
    /// RPython `self.rtyper = rtyper` (rpbc.py:924). Weak backref.
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.s_pbc = s_pbc` (rpbc.py:925).
    pub s_pbc: SomePBC,
    state: ReprState,
    /// RPython `self.lowleveltype` — `Void` on the constant branch,
    /// `CLASSTYPE` (Ptr(OBJECT_VTABLE)) otherwise via
    /// [`Self::getlowleveltype`].
    lltype: LowLevelType,
}

impl ClassesPBCRepr {
    /// RPython `ClassesPBCRepr.__init__(self, rtyper, s_pbc)`
    /// (rpbc.py:923-932). Both constant and non-constant arms ported.
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Self, TyperError> {
        // upstream rpbc.py:928 — `if s_pbc.is_constant():`. The pyre
        // `SomePBC::is_constant` mirrors model.py:532-537 which only
        // sets `const_box` when `len==1 && !can_be_none &&
        // desc.pyobj is not None`, so a single-desc PBC whose
        // ClassDesc has no live pyobj falls through to the non-constant
        // arm.
        let lltype = if s_pbc.is_constant() {
            LowLevelType::Void
        } else {
            // upstream rpbc.py:932 — `self.lowleveltype = self.getlowleveltype()`.
            Self::getlowleveltype()
        };
        Ok(ClassesPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            s_pbc,
            state: ReprState::new(),
            lltype,
        })
    }

    /// RPython `ClassesPBCRepr.getlowleveltype(self)` (rpbc.py:1080-1081).
    ///
    /// ```python
    /// def getlowleveltype(self):
    ///     return CLASSTYPE
    /// ```
    pub fn getlowleveltype() -> LowLevelType {
        crate::translator::rtyper::rclass::CLASSTYPE.clone()
    }

    /// RPython `ClassesPBCRepr.get_access_set(self, attrname)`
    /// (rpbc.py:934-948):
    ///
    /// ```python
    /// def get_access_set(self, attrname):
    ///     classdescs = list(self.s_pbc.descriptions)
    ///     access = classdescs[0].queryattrfamily(attrname)
    ///     for classdesc in classdescs[1:]:
    ///         access1 = classdesc.queryattrfamily(attrname)
    ///         assert access1 is access       # XXX not implemented
    ///     if access is None:
    ///         raise rclass.MissingRTypeAttribute(attrname)
    ///     commonbase = access.commonbase
    ///     class_repr = rclass.getclassrepr(self.rtyper, commonbase)
    ///     return access, class_repr
    /// ```
    ///
    /// Returns the `(ClassAttrFamily, ClassRepr)` pair that hosts the
    /// shared vtable slot for `attrname`. The `ClassAttrFamily.commonbase`
    /// is populated by
    /// [`crate::translator::rtyper::normalizecalls::merge_classpbc_getattr_into_classdef`];
    /// missing means `merge_classpbc_getattr_into_classdef` was either
    /// never run or skipped this family because it has fewer than two
    /// distinct ClassDescs (single-class PBCs read attrs through their
    /// concrete `ClassRepr` directly, not through this dispatch).
    pub fn get_access_set(
        &self,
        attrname: &str,
    ) -> Result<
        (
            Rc<RefCell<crate::annotator::description::ClassAttrFamily>>,
            crate::translator::rtyper::rclass::ClassReprArc,
        ),
        TyperError,
    > {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.get_access_set: rtyper weak ref dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.get_access_set: annotator weak ref dropped")
        })?;

        // upstream `classdescs = list(self.s_pbc.descriptions)`. Pyre
        // stores `descriptions: BTreeMap<DescKey, DescEntry>` so
        // iteration is deterministic — the "first" desc is therefore
        // well-defined for the assert-all-equal walk below.
        let mut class_descs: Vec<Rc<RefCell<crate::annotator::classdesc::ClassDesc>>> = Vec::new();
        for entry in self.s_pbc.descriptions.values() {
            if let DescEntry::Class(rc) = entry {
                class_descs.push(rc.clone());
            } else {
                return Err(TyperError::message(format!(
                    "ClassesPBCRepr.get_access_set: non-Class desc {entry:?} in s_pbc",
                )));
            }
        }
        let Some((first, rest)) = class_descs.split_first() else {
            return Err(TyperError::message(
                "ClassesPBCRepr.get_access_set: empty descriptions",
            ));
        };

        // upstream `access = classdescs[0].queryattrfamily(attrname); for
        // classdesc in classdescs[1:]: access1 = classdesc.queryattrfamily(attrname);
        // assert access1 is access`.
        let access = crate::annotator::classdesc::ClassDesc::queryattrfamily(first, attrname);
        for other in rest {
            let other_access =
                crate::annotator::classdesc::ClassDesc::queryattrfamily(other, attrname);
            match (&access, &other_access) {
                (Some(a), Some(b)) if Rc::ptr_eq(a, b) => {}
                _ => {
                    return Err(TyperError::message(format!(
                        "ClassesPBCRepr.get_access_set: descs disagree on \
                         attrfamily for {attrname:?} — upstream marks this \
                         path as 'XXX not implemented'"
                    )));
                }
            }
        }
        // upstream `if access is None: raise rclass.MissingRTypeAttribute(attrname)`.
        let access = access.ok_or_else(|| {
            TyperError::missing_rtype_operation(format!(
                "ClassesPBCRepr.get_access_set: no access set for attribute \
                 {attrname:?} (rclass.MissingRTypeAttribute)"
            ))
        })?;

        // upstream `commonbase = access.commonbase`. Populated by
        // `normalizecalls.merge_classpbc_getattr_into_classdef`
        // (normalizecalls.py:232).
        let commonbase = access.borrow().commonbase.clone().ok_or_else(|| {
            TyperError::message(
                "ClassesPBCRepr.get_access_set: ClassAttrFamily.commonbase missing — \
                     merge_classpbc_getattr_into_classdef must run before rtyping",
            )
        })?;

        // upstream `class_repr = rclass.getclassrepr(self.rtyper, commonbase)`.
        let class_repr =
            crate::translator::rtyper::rclass::getclassrepr_arc(&rtyper, Some(&commonbase))?;
        let _ = annotator; // silence unused after future-proofing the upgrade() guard.
        Ok((access, class_repr))
    }
}

impl Repr for ClassesPBCRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ClassesPBCRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::ClassesPBCRepr
    }

    /// RPython `ClassesPBCRepr.convert_desc(self, desc)` (rpbc.py:
    /// :950-957).
    ///
    /// ```python
    /// def convert_desc(self, desc):
    ///     if desc not in self.s_pbc.descriptions:
    ///         raise TyperError("%r not in %r" % (desc, self))
    ///     if self.lowleveltype is Void:
    ///         return None
    ///     subclassdef = desc.getuniqueclassdef()
    ///     r_subclass = rclass.getclassrepr(self.rtyper, subclassdef)
    ///     return r_subclass.getruntime(self.lowleveltype)
    /// ```
    ///
    /// Both Void and non-Void arms ported. The non-Void arm walks
    /// `desc.getuniqueclassdef()` → `getclassrepr_arc(rtyper, classdef)`
    /// → `ClassReprArc::getruntime(self.lowleveltype)` to materialise
    /// the vtable pointer constant.
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        if !self.s_pbc.descriptions.contains_key(&desc.desc_key()) {
            return Err(TyperError::message(format!(
                "ClassesPBCRepr.convert_desc: {desc:?} not in {:?}",
                self.s_pbc
            )));
        }
        if matches!(self.lltype, LowLevelType::Void) {
            return Ok(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            ));
        }
        // upstream: `subclassdef = desc.getuniqueclassdef();
        //           r_subclass = rclass.getclassrepr(self.rtyper, subclassdef);
        //           return r_subclass.getruntime(self.lowleveltype)`.
        let DescEntry::Class(class_rc) = desc else {
            return Err(TyperError::message(format!(
                "ClassesPBCRepr.convert_desc: non-Class desc {desc:?} reached the \
                 non-Void arm",
            )));
        };
        let subclassdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc)
            .map_err(|e| TyperError::message(e.to_string()))?;
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.convert_desc: rtyper weak ref dropped")
        })?;
        let r_subclass =
            crate::translator::rtyper::rclass::getclassrepr_arc(&rtyper, Some(&subclassdef))?;
        let vtable_ptr = r_subclass.getruntime(&self.lltype)?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(vtable_ptr)),
            self.lltype.clone(),
        ))
    }

    /// RPython `ClassesPBCRepr.convert_const(self, cls)` (rpbc.py:959-968).
    ///
    /// ```python
    /// def convert_const(self, cls):
    ///     if cls is None:
    ///         if self.lowleveltype is Void:
    ///             return None
    ///         else:
    ///             T = self.lowleveltype
    ///             return nullptr(T.TO)
    ///     bk = self.rtyper.annotator.bookkeeper
    ///     classdesc = bk.getdesc(cls)
    ///     return self.convert_desc(classdesc)
    /// ```
    ///
    /// All three arms ported:
    /// - `cls is None` + `Void` → `Constant(None, Void)`
    /// - `cls is None` + non-Void → `Constant(LLPtr(_defl), Ptr(...))`
    /// - non-None → `bk.getdesc(cls); self.convert_desc(desc)` (the
    ///   non-Void downstream path still surfaces the `getruntime`
    ///   blocker through `convert_desc`).
    /// RPython `ClassesPBCRepr.rtype_getattr(self, hop)` (rpbc.py:970-987):
    ///
    /// ```python
    /// def rtype_getattr(self, hop):
    ///     if hop.s_result.is_constant():
    ///         return hop.inputconst(hop.r_result, hop.s_result.const)
    ///     else:
    ///         attr = hop.args_s[1].const
    ///         if attr == '__name__':
    ///             from rpython.rtyper.lltypesystem import rstr
    ///             class_repr = self.rtyper.rootclass_repr
    ///             vcls, vattr = hop.inputargs(class_repr, Void)
    ///             cname = inputconst(Void, 'name')
    ///             return hop.genop('getfield', [vcls, cname],
    ///                              resulttype = Ptr(rstr.STR))
    ///         access_set, class_repr = self.get_access_set(attr)
    ///         vcls, vattr = hop.inputargs(class_repr, Void)
    ///         v_res = class_repr.getpbcfield(vcls, access_set, attr, hop.llops)
    ///         s_res = access_set.s_value
    ///         r_res = self.rtyper.getrepr(s_res)
    ///         return hop.llops.convertvar(v_res, r_res, hop.r_result)
    /// ```
    ///
    /// The `__name__` arm currently surfaces a missing-op TyperError —
    /// both `rstr.STR` and the `OBJECT_VTABLE.{name,instantiate}`
    /// fields are pre-existing blockers tracked separately. The body
    /// here matches upstream's structure so the arm uncloses to a
    /// single-line `genop` once those land.
    fn rtype_getattr(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rclass::ClassReprArc;
        use crate::translator::rtyper::rtyper::{ConvertedTo, HighLevelOp};

        // upstream `if hop.s_result.is_constant(): return hop.inputconst(hop.r_result, hop.s_result.const)`.
        let const_fast_path: Option<(Arc<dyn Repr>, ConstValue)> = {
            let s_result_borrow = hop.s_result.borrow();
            let r_result_borrow = hop.r_result.borrow();
            match (s_result_borrow.as_ref(), r_result_borrow.as_ref()) {
                (Some(s_result), Some(r_result)) => {
                    s_result.const_().map(|cv| (r_result.clone(), cv.clone()))
                }
                _ => None,
            }
        };
        if let Some((r_result, const_val)) = const_fast_path {
            let c = HighLevelOp::inputconst(ConvertedTo::Repr(r_result.as_ref()), &const_val)?;
            return Ok(Some(Hlvalue::Constant(c)));
        }

        // upstream `attr = hop.args_s[1].const`.
        let attr: String = {
            let args_s = hop.args_s.borrow();
            let s_attr = args_s.get(1).cloned().ok_or_else(|| {
                TyperError::message(
                    "ClassesPBCRepr.rtype_getattr: missing attribute argument args_s[1]",
                )
            })?;
            match s_attr.const_().cloned() {
                Some(value) => value.as_text().unwrap_or("<non-string>").to_string(),
                other => {
                    return Err(TyperError::message(format!(
                        "ClassesPBCRepr.rtype_getattr: non-constant attribute name \
                         (got {other:?})"
                    )));
                }
            }
        };

        // upstream `if attr == '__name__':
        //     class_repr = self.rtyper.rootclass_repr
        //     vcls, vattr = hop.inputargs(class_repr, Void)
        //     cname = inputconst(Void, 'name')
        //     return hop.genop('getfield', [vcls, cname], resulttype=Ptr(rstr.STR))`.
        if attr == "__name__" {
            let rtyper = self.rtyper.upgrade().ok_or_else(|| {
                TyperError::message(
                    "ClassesPBCRepr.rtype_getattr('__name__'): rtyper weak ref dropped",
                )
            })?;
            let root_repr = rtyper.rootclass_repr.borrow().clone().ok_or_else(|| {
                TyperError::message(
                    "ClassesPBCRepr.rtype_getattr('__name__'): rootclass_repr is None — \
                         call RPythonTyper::initialize_exceptiondata first",
                )
            })?;
            let root_repr_arc: Arc<dyn Repr> = root_repr.clone();
            let v_args = hop.inputargs(vec![
                ConvertedTo::Repr(root_repr_arc.as_ref()),
                ConvertedTo::LowLevelType(&LowLevelType::Void),
            ])?;
            let vcls = v_args[0].clone();
            let cname =
                Constant::with_concretetype(ConstValue::byte_str("name"), LowLevelType::Void);
            let strptr = crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone();
            let v_res = {
                let mut llops = hop.llops.borrow_mut();
                llops
                    .genop(
                        "getfield",
                        vec![vcls, Hlvalue::Constant(cname)],
                        crate::translator::rtyper::rtyper::GenopResult::LLType(strptr),
                    )
                    .expect("getfield with non-Void result yields a Variable")
            };
            return Ok(Some(Hlvalue::Variable(v_res)));
        }

        // upstream `access_set, class_repr = self.get_access_set(attr)`.
        let (access_set, class_repr) = self.get_access_set(&attr)?;
        let class_repr_inst = match &class_repr {
            ClassReprArc::Inst(inst) => inst.clone(),
            ClassReprArc::Root(_) => {
                return Err(TyperError::message(
                    "ClassesPBCRepr.rtype_getattr: commonbase resolved to RootClassRepr — \
                     unexpected; root has no pbcfields",
                ));
            }
        };
        let class_repr_arc: Arc<dyn Repr> = class_repr.as_repr();

        // upstream `vcls, vattr = hop.inputargs(class_repr, Void)`.
        let v_args = hop.inputargs(vec![
            ConvertedTo::Repr(class_repr_arc.as_ref()),
            ConvertedTo::LowLevelType(&LowLevelType::Void),
        ])?;
        let vcls = v_args[0].clone();

        // upstream `v_res = class_repr.getpbcfield(vcls, access_set, attr, hop.llops)`.
        let access_set_id = Rc::as_ptr(&access_set) as usize;
        let v_res = {
            let mut llops = hop.llops.borrow_mut();
            class_repr_inst.getpbcfield(vcls, access_set_id, &attr, &mut *llops)?
        };

        // upstream `s_res = access_set.s_value; r_res = self.rtyper.getrepr(s_res)`.
        let s_res = access_set.borrow().s_value.clone();
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.rtype_getattr: rtyper weak ref dropped")
        })?;
        let r_res = rtyper.getrepr(&s_res)?;

        // upstream `return hop.llops.convertvar(v_res, r_res, hop.r_result)`.
        let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
            TyperError::message(
                "ClassesPBCRepr.rtype_getattr: hop.r_result missing on the non-const arm",
            )
        })?;
        let result = {
            let mut llops = hop.llops.borrow_mut();
            llops.convertvar(Hlvalue::Variable(v_res), r_res.as_ref(), r_result.as_ref())?
        };
        Ok(Some(result))
    }

    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if matches!(value, ConstValue::None) {
            // upstream `if self.lowleveltype is Void: return None`.
            if matches!(self.lltype, LowLevelType::Void) {
                return Ok(Constant::with_concretetype(
                    ConstValue::None,
                    LowLevelType::Void,
                ));
            }
            // upstream `T = self.lowleveltype; return nullptr(T.TO)`.
            let LowLevelType::Ptr(ptr) = &self.lltype else {
                return Err(TyperError::message(format!(
                    "ClassesPBCRepr.convert_const: lowleveltype is neither Void nor Ptr, got {:?}",
                    self.lltype,
                )));
            };
            let null_ptr = ptr.as_ref().clone()._defl();
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(null_ptr)),
                self.lltype.clone(),
            ));
        }
        // upstream `bk = self.rtyper.annotator.bookkeeper;
        //          classdesc = bk.getdesc(cls);
        //          return self.convert_desc(classdesc)`.
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "ClassesPBCRepr.convert_const: non-None value must be a HostObject \
                 carrying a class, got {value:?}",
            )));
        };
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.convert_const: rtyper weak ref dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("ClassesPBCRepr.convert_const: annotator weak ref dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        self.convert_desc(&desc)
    }
}

/// RPython `getFrozenPBCRepr(rtyper, s_pbc)` (rpbc.py:610-632).
///
/// ```python
/// def getFrozenPBCRepr(rtyper, s_pbc):
///     descs = list(s_pbc.descriptions)
///     assert len(descs) >= 1
///     if len(descs) == 1 and not s_pbc.can_be_None:
///         return SingleFrozenPBCRepr(descs[0])
///     else:
///         access = descs[0].queryattrfamily()
///         for desc in descs[1:]:
///             access1 = desc.queryattrfamily()
///             if access1 is not access:
///                 try:
///                     return rtyper.pbc_reprs['unrelated']
///                 except KeyError:
///                     result = MultipleUnrelatedFrozenPBCRepr(rtyper)
///                     rtyper.pbc_reprs['unrelated'] = result
///                     return result
///         try:
///             return rtyper.pbc_reprs[access]
///         except KeyError:
///             result = MultipleFrozenPBCRepr(rtyper, access)
///             rtyper.pbc_reprs[access] = result
///             rtyper.add_pendingsetup(result)
///             return result
/// ```
///
/// Pyre's port currently handles:
/// * single-desc + `!can_be_None` → [`SingleFrozenPBCRepr`],
/// * multi-desc with any two descs whose `queryattrfamily` differ →
///   [`MultipleUnrelatedFrozenPBCRepr`] (fresh instance per call;
///   `rtyper.pbc_reprs['unrelated']` cache deferred),
/// * multi-desc with a common access set → `MissingRTypeOperation`
///   pending [`MultipleFrozenPBCRepr`].
///
/// Called from both `somepbc_rtyper_makerepr` DescKind::Frozen and the
/// Function-kind uncallable branch — `FunctionDesc.queryattrfamily` is
/// the base `None` stub, so multi-FunctionDesc always appears "common
/// access set = None" and routes to the still-pending `MultipleFrozenPBCRepr`
/// arm.
pub fn get_frozen_pbc_repr(
    rtyper: &Rc<RPythonTyper>,
    s_pbc: &SomePBC,
) -> Result<std::sync::Arc<dyn Repr>, TyperError> {
    if s_pbc.descriptions.is_empty() {
        return Err(TyperError::message(
            "getFrozenPBCRepr: empty descriptions (rpbc.py:612 assertion)",
        ));
    }
    if s_pbc.descriptions.len() == 1 && !s_pbc.can_be_none {
        let frozendesc = s_pbc
            .descriptions
            .values()
            .next()
            .expect("SomePBC guarantees non-empty descriptions")
            .clone();
        return Ok(std::sync::Arc::new(SingleFrozenPBCRepr::new(frozendesc)));
    }
    // upstream rpbc.py:616-625 — walk descs, collect each
    // `queryattrfamily()` (Python-object identity). If any two differ,
    // route to MultipleUnrelatedFrozenPBCRepr.
    //
    // FrozenDesc.queryattrfamily returns `Option<Rc<RefCell<FrozenAttrFamily>>>`;
    // FunctionDesc (via base `Desc`) returns `Option<()>` = None. Pyre
    // collapses both to `Option<usize>` keyed on `Rc::as_ptr`.
    let access_keys: Vec<Option<usize>> = s_pbc
        .descriptions
        .values()
        .map(|d| match d {
            DescEntry::Frozen(fd) => fd
                .borrow()
                .queryattrfamily()
                .map(|family| Rc::as_ptr(&family) as usize),
            _ => None,
        })
        .collect();
    let first = access_keys.first().copied().unwrap_or(None);
    let all_same = access_keys.iter().all(|key| *key == first);
    if !all_same {
        // upstream rpbc.py:621-624 — `rtyper.pbc_reprs['unrelated']`
        // singleton cache. Fetch if present, otherwise insert a fresh
        // repr and return it.
        use crate::translator::rtyper::rtyper::PbcReprKey;
        if let Some(cached) = rtyper.pbc_reprs.borrow().get(&PbcReprKey::Unrelated) {
            return Ok(cached.clone());
        }
        let fresh: std::sync::Arc<dyn Repr> =
            std::sync::Arc::new(MultipleUnrelatedFrozenPBCRepr::new(rtyper));
        rtyper
            .pbc_reprs
            .borrow_mut()
            .insert(PbcReprKey::Unrelated, fresh.clone());
        return Ok(fresh);
    }
    // upstream rpbc.py:626-632 — common-access-set arm.
    //
    // ```python
    // try:
    //     return rtyper.pbc_reprs[access]
    // except KeyError:
    //     result = MultipleFrozenPBCRepr(rtyper, access)
    //     rtyper.pbc_reprs[access] = result
    //     rtyper.add_pendingsetup(result)
    //     return result
    // ```
    //
    // All descs share `first` (verified above) — the access family is
    // either `Some(usize)` or `None`. None means every desc has no
    // attrfamily; upstream still routes to `MultipleFrozenPBCRepr` with
    // `access_set=None`, but the Rust port requires a live family Rc to
    // construct the repr. Resolve the family from the first frozen
    // desc's `queryattrfamily()` so the repr can read its `attrs`.
    let Some(access_key) = first else {
        return Err(TyperError::missing_rtype_operation(
            "getFrozenPBCRepr: MultipleFrozenPBCRepr with access_set=None \
             (rpbc.py:626-632 + 758) is upstream's no-attrs path — pyre \
             defers it because no current consumer needs a zero-field \
             struct PBC",
        ));
    };
    use crate::translator::rtyper::rtyper::PbcReprKey;
    if let Some(cached) = rtyper
        .pbc_reprs
        .borrow()
        .get(&PbcReprKey::Access(access_key))
    {
        return Ok(cached.clone());
    }
    // Fetch the live FrozenAttrFamily — every desc shared the same key
    // so any of them yields the same family.
    let access_family = s_pbc
        .descriptions
        .values()
        .find_map(|d| match d {
            DescEntry::Frozen(fd) => fd.borrow().queryattrfamily(),
            _ => None,
        })
        .ok_or_else(|| {
            TyperError::message(
                "getFrozenPBCRepr: invariant — first desc's queryattrfamily \
                 returned Some(key) but no descs surface a family",
            )
        })?;
    let fresh: std::sync::Arc<dyn Repr> =
        std::sync::Arc::new(MultipleFrozenPBCRepr::new(rtyper, access_family));
    rtyper
        .pbc_reprs
        .borrow_mut()
        .insert(PbcReprKey::Access(access_key), fresh.clone());
    rtyper.add_pendingsetup(fresh.clone());
    Ok(fresh)
}

/// RPython `SomePBC.rtyper_makerepr` single-FunctionDesc arm
/// (rpbc.py:35-62, limited to the degenerate branch `len(descriptions)
/// == 1 and not can_be_None`). Other shapes surface as
/// `MissingRTypeOperation` until their concrete reprs land.
///
/// Outstanding SomePBC repr stubs (each would be a full Repr class
/// with `_setup_repr` / `convert_desc` / `convert_const` /
/// `rtype_simple_call` / `rtype_call_args` / `call`):
///
///   * `SmallFunctionSetPBCRepr` (rpbc.py:47, class at rpbc.py:~393) —
///     Small multi-desc Function PBCs chosen by
///     `small_cand(rtyper, s_pbc)` (already ported at rpbc.rs
///     `small_cand`). Blocked on `Char` / `Signed` integer-indexed
///     vtable layout + `inputconst(Char, i)` call-site lowering.
///   * `MethodsPBCRepr` (rpbc.py:53, class at rpbc.py:~1126) —
///     DescKind::Method. Blocked on upstream `MethodDesc` /
///     `get_funcdesc_or_implfunc` split, plus bound-method
///     `convert_desc_or_const` that unwraps `im_func` / `im_self`.
///   * `MethodOfFrozenPBCRepr` (rpbc.py:57, class at rpbc.py:~869) —
///     DescKind::MethodOfFrozen. Blocked on `MethodOfFrozenDesc`
///     receiver-repr delegation (`get_r_implfunc` returns
///     `(r_func, 1)`) and the associated frozen-receiver Constant
///     build.
///
/// All three ultimately call through `FunctionReprBase.call` whose
/// port is blocked on ExceptionData + callparse (see the deferred-
/// ports comment at the end of the `FunctionReprBase` impl above).
pub fn somepbc_rtyper_makerepr(
    s_pbc: &SomePBC,
    rtyper: &Rc<RPythonTyper>,
) -> Result<std::sync::Arc<dyn Repr>, TyperError> {
    let kind = s_pbc.get_kind().map_err(|e| {
        TyperError::message(format!(
            "SomePBC.rtyper_makerepr: {}",
            e.msg.as_deref().unwrap_or("<no message>")
        ))
    })?;
    match kind {
        DescKind::Function => {
            if s_pbc.descriptions.len() == 1 && !s_pbc.can_be_none {
                Ok(
                    std::sync::Arc::new(FunctionRepr::new(rtyper, s_pbc.clone())?)
                        as std::sync::Arc<dyn Repr>,
                )
            } else {
                // rpbc.py:42-49 — sample = self.any_description();
                // callfamily = sample.querycallfamily(); if callfamily
                // and callfamily.total_calltable_size > 0:
                //     getRepr = FunctionsPBCRepr
                //     if small_cand(rtyper, self):
                //         getRepr = SmallFunctionSetPBCRepr
                // else:
                //     getRepr = getFrozenPBCRepr
                let sample = s_pbc.any_description().ok_or_else(|| {
                    TyperError::message("SomePBC.rtyper_makerepr: empty descriptions")
                })?;
                let callfamily = sample
                    .as_function()
                    .ok_or_else(|| {
                        TyperError::message(
                            "SomePBC.rtyper_makerepr: Function-kind sample is not a FunctionDesc",
                        )
                    })?
                    .borrow()
                    .base
                    .querycallfamily();
                let callable = callfamily
                    .as_ref()
                    .map(|cf| cf.borrow().total_calltable_size > 0)
                    .unwrap_or(false);
                if callable {
                    if small_cand(rtyper, s_pbc)? {
                        Err(TyperError::missing_rtype_operation(
                            "SomePBC.rtyper_makerepr: SmallFunctionSetPBCRepr \
                             (rpbc.py:47) port pending",
                        ))
                    } else {
                        Ok(
                            std::sync::Arc::new(FunctionsPBCRepr::new(rtyper, s_pbc.clone())?)
                                as std::sync::Arc<dyn Repr>,
                        )
                    }
                } else {
                    // rpbc.py:49 — uncallable Function-kind PBCs route
                    // through `getFrozenPBCRepr`. The multi-desc path
                    // inside the factory still surfaces
                    // `MissingRTypeOperation`, but single-desc
                    // !can_be_None (degenerate uncallable) lands
                    // directly as `SingleFrozenPBCRepr`.
                    get_frozen_pbc_repr(rtyper, s_pbc)
                }
            }
        }
        DescKind::Class => {
            // rpbc.py:50-52 — `getRepr = ClassesPBCRepr`. Pyre's port
            // only handles the constant-class branch today (lowleveltype
            // = Void); non-constant shapes surface `MissingRTypeOperation`
            // from inside `ClassesPBCRepr::new`.
            Ok(
                std::sync::Arc::new(ClassesPBCRepr::new(rtyper, s_pbc.clone())?)
                    as std::sync::Arc<dyn Repr>,
            )
        }
        DescKind::Method => Err(TyperError::missing_rtype_operation(
            "SomePBC.rtyper_makerepr: MethodsPBCRepr (rpbc.py:53-54) port pending",
        )),
        DescKind::Frozen => get_frozen_pbc_repr(rtyper, s_pbc),
        DescKind::MethodOfFrozen => {
            // rpbc.py:57-58 — `getRepr = MethodOfFrozenPBCRepr`.
            Ok(
                std::sync::Arc::new(MethodOfFrozenPBCRepr::new(rtyper, s_pbc.clone())?)
                    as std::sync::Arc<dyn Repr>,
            )
        }
    }
}

#[cfg(test)]
mod pbc_repr_tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::bookkeeper::Bookkeeper;
    use crate::annotator::description::{DescEntry, FunctionDesc};
    use crate::flowspace::argument::Signature;
    use std::cell::RefCell as StdRefCell;

    fn make_rtyper() -> (Rc<RPythonAnnotator>, Rc<RPythonTyper>) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().unwrap();
        (ann, rtyper)
    }

    fn function_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
        DescEntry::Function(Rc::new(StdRefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            name,
            Signature::new(vec![], None, None),
            None,
            None,
        ))))
    }

    #[test]
    fn function_repr_has_void_lowleveltype_and_pbc_tag() {
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(r.lowleveltype(), &LowLevelType::Void);
        assert_eq!(r.class_name(), "FunctionRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::FunctionRepr);
    }

    #[test]
    fn somepbc_rtyper_makerepr_single_function_desc_without_none_returns_function_repr() {
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "FunctionRepr");
    }

    #[test]
    fn somepbc_rtyper_makerepr_single_function_desc_with_can_be_none_surfaces_pending() {
        // Uncallable (no callfamily wired) → getFrozenPBCRepr branch.
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f], true);
        let err = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap_err();
        assert!(
            err.to_string().contains("getFrozenPBCRepr"),
            "unexpected: {}",
            err
        );
    }

    #[test]
    fn somepbc_rtyper_makerepr_multi_function_descs_without_callfamily_uses_getfrozenpbcrepr() {
        // Multi-FunctionDesc PBC without an attached callfamily → upstream
        // rpbc.py:49 routes to getFrozenPBCRepr.
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let g = function_entry(&ann.bookkeeper, "g");
        let s_pbc = SomePBC::new(vec![f, g], false);
        let err = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap_err();
        assert!(
            err.to_string().contains("getFrozenPBCRepr"),
            "unexpected: {}",
            err
        );
    }

    #[test]
    fn somepbc_rtyper_makerepr_multi_function_descs_with_callfamily_uses_functionspbcrepr() {
        // Multi-FunctionDesc PBC whose sample has total_calltable_size > 0 →
        // upstream rpbc.py:44-45 routes to FunctionsPBCRepr. The single
        // uniquerow branch (rpbc.py:232-234) produces lowleveltype = row.fntype
        // wrapped as Ptr(Func).
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        // Pre-populate each FunctionDesc's graph cache so consider_call_site
        // can build the call table without touching a missing pyobj.
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        let args = ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();
        assert!(
            fd_f.borrow()
                .base
                .getcallfamily()
                .unwrap()
                .borrow()
                .total_calltable_size
                > 0,
            "sanity: consider_call_site should have grown total_calltable_size",
        );

        let s_pbc = SomePBC::new(
            vec![DescEntry::Function(fd_f), DescEntry::Function(fd_g)],
            false,
        );
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "FunctionsPBCRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::FunctionsPBCRepr);
        // Single uniquerow → Ptr(Func(...)) as upstream `typeOf(llfn)`.
        assert!(matches!(
            r.lowleveltype(),
            LowLevelType::Ptr(box_ptr) if matches!(box_ptr.TO, PtrTarget::Func(_))
        ));
    }

    #[test]
    fn function_repr_convert_desc_returns_void_none_constant() {
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f.clone()], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();

        // rpbc.py:320-321 — convert_desc returns None; pyre wraps it in
        // a Void-typed Constant so call sites get a typed value.
        let c = r.convert_desc(&f).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::None));
    }

    #[test]
    fn function_repr_convert_const_returns_void_none_constant() {
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();

        // rpbc.py:323-324 — convert_const returns None regardless of
        // the input value (a constant function PBC has no runtime
        // payload).
        let c = r.convert_const(&ConstValue::Int(123)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::None));
    }

    fn frozen_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
        use crate::annotator::description::FrozenDesc;
        use crate::flowspace::model::HostObject;
        // `Frozen`-kind descriptions require a concrete Python-side
        // object; use a module carrier since upstream frozen descs
        // typically wrap module/class singletons.
        let pyobj = HostObject::new_module(name);
        DescEntry::Frozen(Rc::new(StdRefCell::new(
            FrozenDesc::new(bk.clone(), pyobj).expect("FrozenDesc::new"),
        )))
    }

    fn method_of_frozen_entry(
        bk: &Rc<Bookkeeper>,
        funcdesc: &Rc<StdRefCell<crate::annotator::description::FunctionDesc>>,
        frozen_name: &str,
    ) -> DescEntry {
        use crate::annotator::description::{FrozenDesc, MethodOfFrozenDesc};
        use crate::flowspace::model::HostObject;
        let pyobj = HostObject::new_module(frozen_name);
        let frozendesc = Rc::new(StdRefCell::new(FrozenDesc::new(bk.clone(), pyobj).unwrap()));
        DescEntry::MethodOfFrozen(Rc::new(StdRefCell::new(MethodOfFrozenDesc::new(
            bk.clone(),
            funcdesc.clone(),
            frozendesc,
        ))))
    }

    #[test]
    fn method_of_frozen_pbc_repr_new_succeeds_for_single_methodoffrozen_pbc() {
        // rpbc.py:849-869 — shared FunctionDesc + can_be_None=False
        // builds the repr; lowleveltype = r_im_self.lowleveltype.
        // Single-MethodOfFrozenDesc PBC routes r_im_self through
        // SingleFrozenPBCRepr (Void-typed).
        let (ann, rtyper) = make_rtyper();
        let DescEntry::Function(fd) = function_entry(&ann.bookkeeper, "f") else {
            unreachable!()
        };
        let mof = method_of_frozen_entry(&ann.bookkeeper, &fd, "frozen0");
        let s_pbc = SomePBC::new(vec![mof], false);
        let r = MethodOfFrozenPBCRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(r.class_name(), "MethodOfFrozenPBCRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::MethodOfFrozenPBCRepr);
        // r_im_self is `SingleFrozenPBCRepr` → Void.
        assert_eq!(r.lowleveltype(), &LowLevelType::Void);
        assert_eq!(r.lowleveltype(), r.r_im_self.lowleveltype());
        // The shared FunctionDesc on the repr is the same Rc as the
        // one bound on the MethodOfFrozenDesc.
        assert!(Rc::ptr_eq(&r.funcdesc, &fd));
    }

    // The "mixed funcdescs" rejection branch (rpbc.py:851-853 `assert
    // len(funcdescs) == 1`) is enforced one layer up: `SomePBC::new`
    // (annotator/model.rs:1302-1306) panics with "AnnotatorError: You
    // can't mix a set of methods on a frozen PBC in RPython that are
    // different underlying functions" before the rtyper repr ever
    // sees the SomePBC. The redundant `funcdesc_set.len() != 1`
    // guard inside `MethodOfFrozenPBCRepr::new` is defensive only —
    // the upstream `assert` is the load-bearing check. No separate
    // test is added here because the SomePBC-level enforcement is
    // already exercised by `annotator/model::tests`.

    #[test]
    fn method_of_frozen_pbc_repr_convert_desc_routes_through_r_im_self() {
        // rpbc.py:878-882 — `if mdesc.funcdesc is not self.funcdesc:
        // raise; return self.r_im_self.convert_desc(mdesc.frozendesc)`.
        // The success path returns the Constant produced by the
        // bound-frozendesc's repr (single-frozen → Void None sentinel).
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        let DescEntry::Function(fd) = function_entry(&ann.bookkeeper, "f") else {
            unreachable!()
        };
        let mof = method_of_frozen_entry(&ann.bookkeeper, &fd, "frozen0");
        let s_pbc = SomePBC::new(vec![mof.clone()], false);
        let r = MethodOfFrozenPBCRepr::new(&rtyper, s_pbc).unwrap();
        let c = r.convert_desc(&mof).unwrap();
        // Single-frozen `r_im_self` → Void-typed `None` sentinel
        // matches `SingleFrozenPBCRepr.convert_desc` parity.
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::None));
    }

    #[test]
    fn method_of_frozen_pbc_repr_convert_desc_rejects_foreign_funcdesc() {
        // rpbc.py:879-880 — "not a method bound on %r". Build the
        // repr around `f`, then call convert_desc with a MethodOf-
        // FrozenDesc whose funcdesc is `g`.
        let (ann, rtyper) = make_rtyper();
        let DescEntry::Function(fd_f) = function_entry(&ann.bookkeeper, "f") else {
            unreachable!()
        };
        let DescEntry::Function(fd_g) = function_entry(&ann.bookkeeper, "g") else {
            unreachable!()
        };
        let mof_f = method_of_frozen_entry(&ann.bookkeeper, &fd_f, "frozen0");
        let mof_g = method_of_frozen_entry(&ann.bookkeeper, &fd_g, "frozen0");
        let s_pbc = SomePBC::new(vec![mof_f], false);
        let r = MethodOfFrozenPBCRepr::new(&rtyper, s_pbc).unwrap();
        let err = r.convert_desc(&mof_g).unwrap_err();
        assert!(
            err.to_string().contains("not a method bound on"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn somepbc_rtyper_makerepr_method_of_frozen_kind_routes_to_method_of_frozen_pbc_repr() {
        // rpbc.py:57-58 — `elif issubclass(kind, MethodOfFrozenDesc):
        //                  getRepr = MethodOfFrozenPBCRepr`.
        use crate::annotator::model::SomeValue;
        let (ann, rtyper) = make_rtyper();
        let DescEntry::Function(fd) = function_entry(&ann.bookkeeper, "f") else {
            unreachable!()
        };
        let mof = method_of_frozen_entry(&ann.bookkeeper, &fd, "frozen0");
        let s_pbc = SomePBC::new(vec![mof], false);
        let r = rtyper.getrepr(&SomeValue::PBC(s_pbc)).unwrap();
        assert_eq!(r.repr_class_id(), ReprClassId::MethodOfFrozenPBCRepr);
    }

    #[test]
    fn single_frozen_pbc_repr_convert_desc_identity_check_and_void_sentinel() {
        use crate::flowspace::model::ConstValue;
        let (ann, _rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let r = SingleFrozenPBCRepr::new(f.clone());

        // Same desc identity → Ok(Constant(None, Void)).
        let c = r.convert_desc(&f).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::None));

        // Different desc identity → rejected.
        let other = frozen_entry(&ann.bookkeeper, "frozen1");
        let err = r.convert_desc(&other).unwrap_err();
        assert!(err.to_string().contains("frozendesc identity mismatch"));
    }

    #[test]
    fn single_frozen_pbc_repr_convert_const_returns_void_none() {
        use crate::flowspace::model::ConstValue;
        let (ann, _rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let r = SingleFrozenPBCRepr::new(f);
        let c = r.convert_const(&ConstValue::Int(7)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::None));
    }

    #[test]
    fn multiple_unrelated_frozen_pbc_repr_has_address_lowleveltype() {
        let (_ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        assert_eq!(r.lowleveltype(), &LowLevelType::Address);
        assert_eq!(r.class_name(), "MultipleUnrelatedFrozenPBCRepr");
        assert_eq!(
            r.repr_class_id(),
            ReprClassId::MultipleUnrelatedFrozenPBCRepr
        );
    }

    #[test]
    fn multiple_unrelated_frozen_pbc_repr_null_instance_emits_null_address_constant() {
        let (_ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let null = r.null_instance();
        assert_eq!(null.concretetype, Some(LowLevelType::Address));
        assert!(matches!(
            null.value,
            ConstValue::LLAddress(crate::translator::rtyper::lltypesystem::lltype::_address::Null)
        ));
    }

    #[test]
    fn multiple_unrelated_frozen_pbc_repr_convert_const_none_returns_null_instance() {
        let (_ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let c = r.convert_const(&ConstValue::None).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Address));
        assert!(matches!(
            c.value,
            ConstValue::LLAddress(crate::translator::rtyper::lltypesystem::lltype::_address::Null)
        ));
    }

    /// Non-HostObject non-None values surface a structured TyperError
    /// — upstream rpbc.py:666-670 only feeds HostObjects on the
    /// non-None path, so non-HostObjects are upstream-impossible and
    /// the Rust port rejects them up-front.
    #[test]
    fn multiple_unrelated_frozen_pbc_repr_convert_const_non_hostobject_rejected() {
        let (_ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let err = r.convert_const(&ConstValue::Int(7)).unwrap_err();
        assert!(
            err.to_string().contains("must be a HostObject"),
            "rejection message must be specific, got {err}"
        );
    }

    /// `MultipleUnrelatedFrozenPBCRepr.convert_desc(frozendesc)`
    /// (rpbc.py:685-697) routes the desc through its per-desc repr,
    /// extracts the `_ptr`, wraps it in `_address::Fake`, and caches
    /// the result keyed on `DescEntry::desc_key()`. This test drives
    /// the `r.lowleveltype is Void` arm — `SingleFrozenPBCRepr` is
    /// the per-desc repr for an isolated FrozenDesc and its lltype is
    /// `Void`, so `convert_desc` falls back to `create_instance()`.
    #[test]
    fn multiple_unrelated_frozen_pbc_repr_convert_desc_routes_through_void_per_desc_repr() {
        let (ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let c = r
            .convert_desc(&f)
            .expect("convert_desc Void per-desc repr arm");
        assert_eq!(c.concretetype, Some(LowLevelType::Address));
        let ConstValue::LLAddress(addr) = &c.value else {
            panic!("expected LLAddress, got {:?}", c.value);
        };
        assert!(
            matches!(
                addr,
                crate::translator::rtyper::lltypesystem::lltype::_address::Fake(_)
            ),
            "Void per-desc arm wraps a fresh `pbc` instance via fakeaddress"
        );

        // Repeated convert_desc on the same desc returns the cached value.
        let c2 = r.convert_desc(&f).unwrap();
        assert_eq!(c.value, c2.value);
    }

    #[test]
    fn get_frozen_pbc_repr_single_desc_without_can_be_none_returns_single_frozen_repr() {
        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = get_frozen_pbc_repr(&rtyper, &s_pbc).unwrap();
        assert_eq!(r.class_name(), "SingleFrozenPBCRepr");
    }

    #[test]
    fn get_frozen_pbc_repr_multi_desc_common_access_set_surfaces_pending() {
        // Two FrozenDescs without any `getattrfamily` touch share the
        // same `queryattrfamily() == None` result, so upstream routes
        // them to MultipleFrozenPBCRepr (still pending).
        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let g = frozen_entry(&ann.bookkeeper, "frozen1");
        let s_pbc = SomePBC::new(vec![f, g], false);
        let err = get_frozen_pbc_repr(&rtyper, &s_pbc).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("MultipleFrozenPBCRepr"));
    }

    #[test]
    fn mu_repr_rtype_getattr_requires_constant_s_result() {
        // rpbc.py:708-711 — getattr on a constant PBC must yield a
        // constant; non-constant s_result raises TyperError.
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (_ann, rtyper) = make_rtyper();
        let mu = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let spaceop = SpaceOperation::new(
            OpKind::GetAttr.opname(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // s_result is None (non-constant) → TyperError.
        hop.s_result
            .replace(Some(crate::annotator::model::SomeValue::Impossible));
        hop.r_result.replace(Some(
            std::sync::Arc::new(MultipleUnrelatedFrozenPBCRepr::new(&rtyper))
                as std::sync::Arc<dyn Repr>,
        ));
        let err = mu.rtype_getattr(&hop).unwrap_err();
        assert!(
            err.to_string()
                .contains("getattr on a constant PBC returns a non-constant"),
            "unexpected: {err}",
        );
    }

    #[test]
    fn pair_mu_mu_rtype_is_emits_adr_eq_against_bool_result() {
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let _ = ann;

        // Single MU repr instance shared across both arg slots and the
        // helper call so `inputargs`'s no-op convert path fires.
        let mu_arc: std::sync::Arc<dyn Repr> =
            std::sync::Arc::new(MultipleUnrelatedFrozenPBCRepr::new(&rtyper));

        // Address-typed Variables — inputargs reads args_v / args_r.
        use crate::flowspace::model::Constant;
        let make_address_var = || {
            let v = Variable::new();
            v.set_concretetype(Some(LowLevelType::Address));
            v
        };
        let v1 = Hlvalue::Variable(make_address_var());
        let v2 = Hlvalue::Variable(make_address_var());
        let result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Bool));
        let spaceop = SpaceOperation::new(
            OpKind::Is.opname(),
            vec![v1.clone(), v2.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // Use non-const SomeValue so inputarg skips the const_ shortcut
        // and reaches convertvar directly. Semantically upstream passes
        // SomePBC here; the unit test only cares about the op surface.
        hop.args_v.borrow_mut().push(v1);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(mu_arc.clone()));
        hop.args_v.borrow_mut().push(v2);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(mu_arc.clone()));

        let result = pair_mu_mu_rtype_is_(&*mu_arc, &*mu_arc, &hop).unwrap();
        // adr_eq is emitted as a ll op with Bool result.
        let ops = hop.llops.borrow();
        let adr_eq = ops.ops.iter().find(|op| op.opname == "adr_eq");
        assert!(
            adr_eq.is_some(),
            "expected adr_eq op to be emitted, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>(),
        );
        // Result is a variable pinned to Bool.
        match result {
            Hlvalue::Variable(v) => {
                assert_eq!(v.concretetype(), Some(LowLevelType::Bool));
            }
            Hlvalue::Constant(c) => panic!("unexpected constant result {c:?}"),
        }
        // Silence the unused-const warning for the Constant import.
        let _ = Constant::new(crate::flowspace::model::ConstValue::None);
    }

    #[test]
    fn get_frozen_pbc_repr_unrelated_branch_reuses_rtyper_pbc_reprs_singleton() {
        // upstream rpbc.py:621-624 caches a single MultipleUnrelatedFrozenPBCRepr
        // under `rtyper.pbc_reprs['unrelated']`. Two calls with
        // different SomePBCs (both landing in the unrelated branch)
        // must return the same Arc.
        let (ann, rtyper) = make_rtyper();

        let (f1, g1) = (
            frozen_entry(&ann.bookkeeper, "a"),
            frozen_entry(&ann.bookkeeper, "b"),
        );
        let (DescEntry::Frozen(a_rc), DescEntry::Frozen(b_rc)) = (f1.clone(), g1.clone()) else {
            unreachable!()
        };
        let _ = a_rc.borrow().getattrfamily().unwrap();
        let _ = b_rc.borrow().getattrfamily().unwrap();
        let s_pbc_1 = SomePBC::new(vec![f1, g1], false);
        let r1 = get_frozen_pbc_repr(&rtyper, &s_pbc_1).unwrap();

        let (f2, g2) = (
            frozen_entry(&ann.bookkeeper, "c"),
            frozen_entry(&ann.bookkeeper, "d"),
        );
        let (DescEntry::Frozen(c_rc), DescEntry::Frozen(d_rc)) = (f2.clone(), g2.clone()) else {
            unreachable!()
        };
        let _ = c_rc.borrow().getattrfamily().unwrap();
        let _ = d_rc.borrow().getattrfamily().unwrap();
        let s_pbc_2 = SomePBC::new(vec![f2, g2], false);
        let r2 = get_frozen_pbc_repr(&rtyper, &s_pbc_2).unwrap();

        assert!(
            std::sync::Arc::ptr_eq(&r1, &r2),
            "rtyper.pbc_reprs['unrelated'] must return the same Arc on repeat lookups",
        );
    }

    #[test]
    fn get_frozen_pbc_repr_multi_desc_unrelated_access_sets_routes_to_unrelated_repr() {
        // Give each FrozenDesc its own attrfamily (distinct Rc ptrs),
        // so `queryattrfamily()` returns distinct Python-object
        // identities → upstream rpbc.py:619-625 routes to
        // MultipleUnrelatedFrozenPBCRepr.
        let (ann, rtyper) = make_rtyper();
        let f_entry = frozen_entry(&ann.bookkeeper, "frozen0");
        let g_entry = frozen_entry(&ann.bookkeeper, "frozen1");
        let DescEntry::Frozen(f_rc) = f_entry.clone() else {
            unreachable!("frozen_entry returns DescEntry::Frozen");
        };
        let DescEntry::Frozen(g_rc) = g_entry.clone() else {
            unreachable!("frozen_entry returns DescEntry::Frozen");
        };
        // Materialize distinct FrozenAttrFamily singletons per desc.
        let fam_f = f_rc.borrow().getattrfamily().unwrap();
        let fam_g = g_rc.borrow().getattrfamily().unwrap();
        assert!(!Rc::ptr_eq(&fam_f, &fam_g));
        let s_pbc = SomePBC::new(vec![f_entry, g_entry], false);
        let r = get_frozen_pbc_repr(&rtyper, &s_pbc).unwrap();
        assert_eq!(r.class_name(), "MultipleUnrelatedFrozenPBCRepr");
        assert_eq!(r.lowleveltype(), &LowLevelType::Address);
    }

    /// `get_frozen_pbc_repr` for a multi-FrozenDesc PBC whose descs
    /// share an `Rc::ptr_eq`-identical `FrozenAttrFamily` (after
    /// `mergeattrfamilies`) returns a fresh [`MultipleFrozenPBCRepr`]
    /// keyed in `rtyper.pbc_reprs[Access(family_id)]`. Repeated calls
    /// reuse the cached entry.
    #[test]
    fn get_frozen_pbc_repr_multi_desc_shared_family_returns_multiple_frozen_repr() {
        let (ann, rtyper) = make_rtyper();
        let f_entry = frozen_entry(&ann.bookkeeper, "frozen0");
        let g_entry = frozen_entry(&ann.bookkeeper, "frozen1");
        let DescEntry::Frozen(f_rc) = &f_entry else {
            unreachable!();
        };
        let DescEntry::Frozen(g_rc) = &g_entry else {
            unreachable!();
        };
        // Share a single FrozenAttrFamily across both descs.
        f_rc.borrow()
            .mergeattrfamilies(&[&g_rc.borrow()])
            .expect("mergeattrfamilies");
        let fam_f = f_rc.borrow().getattrfamily().unwrap();
        let fam_g = g_rc.borrow().getattrfamily().unwrap();
        assert!(
            Rc::ptr_eq(&fam_f, &fam_g),
            "post-merge both descs must share the same family Rc"
        );

        let s_pbc = SomePBC::new(vec![f_entry.clone(), g_entry.clone()], false);
        let r = get_frozen_pbc_repr(&rtyper, &s_pbc).expect("MultipleFrozenPBCRepr");
        assert_eq!(r.class_name(), "MultipleFrozenPBCRepr");
        // lowleveltype is Ptr(ForwardReference) before setup.
        let LowLevelType::Ptr(ptr) = r.lowleveltype() else {
            panic!("MultipleFrozenPBCRepr lltype must be Ptr");
        };
        assert!(
            matches!(
                &ptr.TO,
                crate::translator::rtyper::lltypesystem::lltype::PtrTarget::ForwardReference(_)
            ),
            "lowleveltype TO must be ForwardReference until _setup_repr runs"
        );

        // Calling get_frozen_pbc_repr again with an equivalent SomePBC
        // should hit the `pbc_reprs[Access(...)]` singleton cache
        // (rpbc.py:627).
        let s_pbc2 = SomePBC::new(vec![f_entry, g_entry], false);
        let r2 = get_frozen_pbc_repr(&rtyper, &s_pbc2).expect("cached");
        assert!(
            Arc::ptr_eq(&r, &r2),
            "second call must return the same Arc<dyn Repr>"
        );
    }

    /// `MultipleFrozenPBCRepr._setup_repr` runs `_setup_repr_fields`,
    /// converts each access-set attr's `s_value` to a Repr, mangles
    /// the field name with `mangle('pbc', attr)`, and resolves
    /// `pbc_type` to a `Struct('pbc', ...)` carrying those fields.
    #[test]
    fn multiple_frozen_pbc_repr_setup_resolves_pbc_type_struct_with_mangled_fields() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        let (ann, rtyper) = make_rtyper();
        let f_entry = frozen_entry(&ann.bookkeeper, "frozen0");
        let g_entry = frozen_entry(&ann.bookkeeper, "frozen1");
        let DescEntry::Frozen(f_rc) = &f_entry else {
            unreachable!();
        };
        let DescEntry::Frozen(g_rc) = &g_entry else {
            unreachable!();
        };
        f_rc.borrow()
            .mergeattrfamilies(&[&g_rc.borrow()])
            .expect("mergeattrfamilies");
        // Seed a single Int-typed attr `x` on the family.
        {
            let fam = f_rc.borrow().getattrfamily().unwrap();
            fam.borrow_mut()
                .attrs
                .insert("x".into(), SomeValue::Integer(SomeInteger::default()));
        }

        let s_pbc = SomePBC::new(vec![f_entry, g_entry], false);
        let r = get_frozen_pbc_repr(&rtyper, &s_pbc).unwrap();
        Repr::setup(r.as_ref()).expect("setup MultipleFrozenPBCRepr");

        // pbc_type ForwardReference resolved to Struct('pbc', ('pbc_x', Signed)).
        let LowLevelType::Ptr(ptr) = r.lowleveltype() else {
            panic!("MultipleFrozenPBCRepr lltype must be Ptr");
        };
        let to = ptr.TO.clone();
        let resolved = match to {
            crate::translator::rtyper::lltypesystem::lltype::PtrTarget::ForwardReference(fwd) => {
                fwd.resolved().expect("pbc_type resolved after setup")
            }
            other => panic!("Ptr target must remain ForwardReference variant, got {other:?}"),
        };
        let LowLevelType::Struct(body) = resolved else {
            panic!("resolved pbc_type must be Struct");
        };
        assert_eq!(body._name, "pbc");
        assert!(
            body._flds.get("pbc_x").is_some(),
            "Struct('pbc', ...) must carry mangled `pbc_x` field"
        );
    }

    /// `MultipleFrozenPBCRepr.convert_desc(frozendesc)` with attr
    /// values seeded into `frozendesc.attrcache` returns a live
    /// `_ptr` whose struct field equals the converted value. Cache
    /// is populated keyed on the desc identity.
    #[test]
    fn multiple_frozen_pbc_repr_convert_desc_writes_attr_value_into_pbc_struct() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        let (ann, rtyper) = make_rtyper();
        let f_entry = frozen_entry(&ann.bookkeeper, "frozen0");
        let g_entry = frozen_entry(&ann.bookkeeper, "frozen1");
        let DescEntry::Frozen(f_rc) = &f_entry else {
            unreachable!();
        };
        let DescEntry::Frozen(g_rc) = &g_entry else {
            unreachable!();
        };
        f_rc.borrow()
            .mergeattrfamilies(&[&g_rc.borrow()])
            .expect("mergeattrfamilies");
        {
            let fam = f_rc.borrow().getattrfamily().unwrap();
            fam.borrow_mut()
                .attrs
                .insert("x".into(), SomeValue::Integer(SomeInteger::default()));
        }
        // Per-desc attrcache seed.
        f_rc.borrow()
            .create_new_attribute("x", ConstValue::Int(7))
            .unwrap();
        g_rc.borrow()
            .create_new_attribute("x", ConstValue::Int(11))
            .unwrap();

        let s_pbc = SomePBC::new(vec![f_entry.clone(), g_entry.clone()], false);
        let r = get_frozen_pbc_repr(&rtyper, &s_pbc).unwrap();
        Repr::setup(r.as_ref()).expect("setup");

        let c_f = r.convert_desc(&f_entry).expect("convert_desc f");
        let ConstValue::LLPtr(p_f) = &c_f.value else {
            panic!("convert_desc must return LLPtr");
        };
        assert_eq!(
            p_f.getattr("pbc_x").unwrap(),
            crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Signed(7)
        );

        let c_g = r.convert_desc(&g_entry).expect("convert_desc g");
        let ConstValue::LLPtr(p_g) = &c_g.value else {
            panic!("convert_desc must return LLPtr");
        };
        assert_eq!(
            p_g.getattr("pbc_x").unwrap(),
            crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Signed(11)
        );

        // Repeated convert_desc on the same frozendesc returns the
        // identity-equal cached `_ptr`.
        let c_f2 = r.convert_desc(&f_entry).unwrap();
        let ConstValue::LLPtr(p_f2) = &c_f2.value else {
            unreachable!();
        };
        assert_eq!(p_f._hashable_identity(), p_f2._hashable_identity());
    }

    #[test]
    fn somepbc_rtyper_makerepr_single_frozen_desc_without_none_returns_single_frozen_repr() {
        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "SingleFrozenPBCRepr");
    }

    fn class_entry(bk: &Rc<Bookkeeper>, name: &str) -> DescEntry {
        use crate::annotator::classdesc::ClassDesc;
        use crate::flowspace::model::HostObject;
        let pyobj = HostObject::new_class(name, vec![]);
        DescEntry::Class(Rc::new(StdRefCell::new(ClassDesc::new_shell(
            bk,
            pyobj,
            name.to_string(),
        ))))
    }

    #[test]
    fn classes_pbc_repr_constant_branch_has_void_lowleveltype() {
        let (ann, rtyper) = make_rtyper();
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c], false);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(r.lowleveltype(), &LowLevelType::Void);
        assert_eq!(r.class_name(), "ClassesPBCRepr");
        assert_eq!(r.repr_class_id(), ReprClassId::ClassesPBCRepr);
    }

    #[test]
    fn classes_pbc_repr_non_constant_branch_uses_classtype_lowleveltype() {
        let (ann, rtyper) = make_rtyper();
        // can_be_none=true with a single desc: upstream's
        // `s_pbc.is_constant()` returns False, so __init__ routes
        // through `getlowleveltype()` and the repr's lltype is
        // CLASSTYPE.
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c], true);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(
            r.lowleveltype(),
            &crate::translator::rtyper::rclass::CLASSTYPE.clone()
        );
    }

    #[test]
    fn classes_pbc_repr_convert_desc_identity_check() {
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c.clone()], false);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        // Same-identity desc → Void-None sentinel.
        let out = r.convert_desc(&c).unwrap();
        assert_eq!(out.concretetype, Some(LowLevelType::Void));
        assert!(matches!(out.value, ConstValue::None));

        // Other-identity desc → TyperError.
        let other = class_entry(&ann.bookkeeper, "D");
        let err = r.convert_desc(&other).unwrap_err();
        assert!(err.to_string().contains("not in"));
    }

    #[test]
    fn classes_pbc_repr_convert_const_none_returns_void_sentinel() {
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c], false);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        let out = r.convert_const(&ConstValue::None).unwrap();
        assert_eq!(out.concretetype, Some(LowLevelType::Void));
        assert!(matches!(out.value, ConstValue::None));

        // Non-HostObject value rejected with a structured TyperError
        // (upstream surfaces `bk.getdesc` AttributeError on the host
        // side; pyre catches the type mismatch up-front).
        let err = r.convert_const(&ConstValue::Int(7)).unwrap_err();
        assert!(err.to_string().contains("must be a HostObject"));
    }

    #[test]
    fn classes_pbc_repr_convert_const_none_on_non_void_returns_null_ptr() {
        use crate::flowspace::model::ConstValue;
        let (ann, rtyper) = make_rtyper();
        // can_be_none=true forces the non-constant arm, so lltype is
        // CLASSTYPE = Ptr(OBJECT_VTABLE) and convert_const(None) emits
        // `nullptr(T.TO)` as a `ConstValue::LLPtr` carrying the
        // `_defl()` null sentinel.
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c], true);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        let out = r.convert_const(&ConstValue::None).unwrap();
        assert_eq!(
            out.concretetype.as_ref(),
            Some(&crate::translator::rtyper::rclass::CLASSTYPE.clone())
        );
        let ConstValue::LLPtr(p) = &out.value else {
            panic!(
                "non-Void None must produce ConstValue::LLPtr, got {:?}",
                out.value
            );
        };
        // _defl produces a null pointer (no underlying object).
        assert!(!p.nonzero(), "expected null pointer from _defl()");
    }

    #[test]
    fn classes_pbc_repr_convert_const_non_none_routes_through_bookkeeper() {
        use crate::flowspace::model::{ConstValue, HostObject};
        let (ann, rtyper) = make_rtyper();
        // Route the desc construction through `bk.getdesc` so the
        // desc-identity used by `s_pbc` matches the one returned when
        // `convert_const` does the roundtrip — `class_entry` builds a
        // fresh ClassDesc shell that is NOT cached in `bk.descs`.
        let host = HostObject::new_class("C", vec![]);
        let c = ann
            .bookkeeper
            .getdesc(&host)
            .expect("bk.getdesc(host) for fresh class");
        let s_pbc = SomePBC::new(vec![c], false);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        let out = r
            .convert_const(&ConstValue::HostObject(host))
            .expect("non-None HostObject must route through bk.getdesc + convert_desc");
        assert_eq!(out.concretetype, Some(LowLevelType::Void));
        assert!(matches!(out.value, ConstValue::None));
    }

    #[test]
    fn classes_pbc_repr_convert_desc_non_void_returns_vtable_pointer() {
        use crate::flowspace::model::HostObject;
        use crate::translator::rtyper::rclass::CLASSTYPE;
        let (ann, rtyper) = make_rtyper();
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        // Route the ClassDesc through `bk.getdesc` so it's cached and
        // shared between the SomePBC.descriptions seed and the
        // `getuniqueclassdef` lookup invoked by convert_desc.
        let host = HostObject::new_class("C", vec![]);
        let c = ann
            .bookkeeper
            .getdesc(&host)
            .expect("bk.getdesc fresh class");
        // Drive `getuniqueclassdef` so the classdef exists, then assign
        // minimal inheritance ids — `init_vtable → fill_vtable_root`
        // requires `classdef.minid/maxid`. The values are arbitrary
        // here (id assignment is normalizecalls.assign_inheritance_ids
        // territory).
        let DescEntry::Class(class_rc) = &c else {
            unreachable!();
        };
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc)
            .expect("getuniqueclassdef");
        {
            let mut cd = classdef.borrow_mut();
            cd.minid = Some(2);
            cd.maxid = Some(3);
        }
        let s_pbc = SomePBC::new(vec![c.clone()], true);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();
        // Set up the ClassRepr (vtable_type ForwardReference resolves).
        let r_subclass =
            crate::translator::rtyper::rclass::getclassrepr_arc(&rtyper, Some(&classdef)).unwrap();
        Repr::setup(&*r_subclass.as_repr()).expect("setup classrepr");

        let out = r.convert_desc(&c).expect("convert_desc non-Void");
        assert_eq!(out.concretetype.as_ref(), Some(&CLASSTYPE.clone()));
        let ConstValue::LLPtr(p) = &out.value else {
            panic!("expected ConstValue::LLPtr, got {:?}", out.value);
        };
        assert!(p.nonzero(), "vtable pointer must be live (non-null)");
    }

    /// `ClassesPBCRepr.get_access_set(attrname)` resolves the
    /// `ClassAttrFamily` shared by the s_pbc.descriptions and returns
    /// it alongside the `ClassRepr` of the family's `commonbase`. The
    /// `commonbase` field is populated by
    /// [`merge_classpbc_getattr_into_classdef`]; this test runs that
    /// pass first to drive the live setup end-to-end.
    #[test]
    fn classes_pbc_repr_get_access_set_returns_family_and_commonbase_class_repr() {
        use crate::annotator::classdesc::{ClassDef, ClassDesc};
        use crate::annotator::description::DescKey;
        use crate::flowspace::model::HostObject;
        use crate::translator::rtyper::normalizecalls::merge_classpbc_getattr_into_classdef;

        let (ann, rtyper) = make_rtyper();

        // Build Root + Leaf classdescs sharing a ClassAttrFamily for "x".
        // Both descs must live in `bookkeeper.descs` so the merge pass
        // can resolve `family.descs` keys.
        let root_host = HostObject::new_class("pkg.Root", vec![]);
        let root_desc = Rc::new(StdRefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            root_host.clone(),
            "pkg.Root".to_string(),
        )));
        let leaf_host = HostObject::new_class("pkg.Leaf", vec![root_host.clone()]);
        let leaf_desc = Rc::new(StdRefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            leaf_host.clone(),
            "pkg.Leaf".to_string(),
        )));
        leaf_desc.borrow_mut().basedesc = Some(root_desc.clone());
        {
            let mut descs = ann.bookkeeper.descs.borrow_mut();
            descs.insert(root_host.clone(), DescEntry::Class(root_desc.clone()));
            descs.insert(leaf_host.clone(), DescEntry::Class(leaf_desc.clone()));
        }
        // Drive ClassDef creation up front — getuniqueclassdef caches
        // and `merge_classpbc_getattr_into_classdef` calls it again.
        let root_cd = ClassDesc::getuniqueclassdef(&root_desc).unwrap();
        let _leaf_cd = ClassDesc::getuniqueclassdef(&leaf_desc).unwrap();

        // Wire the ClassAttrFamily for "x" via UnionFind union.
        let root_key = DescKey::from_rc(&root_desc);
        let leaf_key = DescKey::from_rc(&leaf_desc);
        ann.bookkeeper.with_classpbc_attr_families("x", |uf| {
            uf.find(root_key);
            uf.union(root_key, leaf_key);
        });

        merge_classpbc_getattr_into_classdef(&ann).expect("merge_classpbc_getattr_into_classdef");

        // Build the ClassesPBCRepr over both descs.
        let s_pbc = SomePBC::new(
            vec![
                DescEntry::Class(root_desc.clone()),
                DescEntry::Class(leaf_desc.clone()),
            ],
            false,
        );
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        // Call get_access_set("x") and verify both halves of the tuple.
        let (access, class_repr) = r.get_access_set("x").expect("get_access_set");
        // The returned access matches what merge wrote to `family.commonbase`.
        let cb = access
            .borrow()
            .commonbase
            .clone()
            .expect("commonbase populated");
        assert!(
            Rc::ptr_eq(&cb, &root_cd),
            "commonbase must equal Root classdef"
        );
        // class_repr is the ClassRepr for Root.
        let class_repr_classdef = match &class_repr {
            crate::translator::rtyper::rclass::ClassReprArc::Inst(inst) => inst.classdef(),
            _ => panic!("commonbase Root must produce ClassReprArc::Inst"),
        };
        assert!(
            Rc::ptr_eq(&class_repr_classdef, &root_cd),
            "class_repr.classdef must equal Root"
        );
    }

    /// `rtype_getattr` const-result fast path (rpbc.py:971-972):
    /// when `hop.s_result.is_constant()` the body short-circuits to
    /// `inputconst(hop.r_result, hop.s_result.const)` BEFORE consulting
    /// the access set or the attrname. The general / `__name__` paths
    /// are never reached.
    #[test]
    fn classes_pbc_repr_rtype_getattr_const_result_short_circuits_to_inputconst() {
        use crate::annotator::classdesc::ClassDesc;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{Hlvalue, HostObject, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let host = HostObject::new_class("pkg.C", vec![]);
        let desc = Rc::new(StdRefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            host.clone(),
            "pkg.C".to_string(),
        )));
        ann.bookkeeper
            .descs
            .borrow_mut()
            .insert(host, DescEntry::Class(desc.clone()));
        let s_pbc = SomePBC::new(vec![DescEntry::Class(desc)], true);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        // Build a getattr hop where s_result is a *constant* integer —
        // should short-circuit to `inputconst` and NEVER touch
        // get_access_set (we leave the bookkeeper unwired so the
        // general path would fail loudly if reached).
        let spaceop = SpaceOperation::new(
            OpKind::GetAttr.opname(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        let mut s_int = SomeInteger::default();
        s_int.base.const_box = Some(Constant::new(ConstValue::Int(7)));
        hop.s_result.replace(Some(SomeValue::Integer(s_int)));
        hop.r_result
            .replace(Some(
                std::sync::Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")))
                    as std::sync::Arc<dyn Repr>,
            ));

        let out = r.rtype_getattr(&hop).expect("const-result path").unwrap();
        let Hlvalue::Constant(c) = out else {
            panic!("const-result must produce a Constant, got {out:?}");
        };
        assert_eq!(c.value, ConstValue::Int(7));
    }

    /// `rtype_getattr('__name__')` (rpbc.py:975-981) routes through
    /// `rootclass_repr` and emits `genop('getfield', [vcls, 'name'],
    /// resulttype=Ptr(rstr.STR))`. Body uncloses now that
    /// `rstr.STR` + `OBJECT_VTABLE.name` have landed.
    #[test]
    fn classes_pbc_repr_rtype_getattr_name_arm_emits_getfield_into_strptr() {
        use crate::annotator::classdesc::ClassDesc;
        use crate::annotator::model::{SomeString, SomeValue};
        use crate::flowspace::model::{Hlvalue, HostObject, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rmodel::SimplePointerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let host = HostObject::new_class("pkg.C", vec![]);
        let desc = Rc::new(StdRefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            host.clone(),
            "pkg.C".to_string(),
        )));
        ann.bookkeeper
            .descs
            .borrow_mut()
            .insert(host, DescEntry::Class(desc.clone()));
        let s_pbc = SomePBC::new(vec![DescEntry::Class(desc.clone())], true);
        let r_arc: std::sync::Arc<dyn Repr> =
            std::sync::Arc::new(ClassesPBCRepr::new(&rtyper, s_pbc).unwrap());

        // Build a hop with non-constant s_result + args_s[1] = Str("__name__").
        let spaceop = SpaceOperation::new(
            OpKind::GetAttr.opname(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.s_result.replace(Some(SomeValue::Impossible));
        // r_result for __name__ is `Ptr(STR)` — use SimplePointerRepr
        // built over `STRPTR`.
        let strptr = crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone();
        hop.r_result.replace(Some(
            std::sync::Arc::new(SimplePointerRepr::new(strptr.clone())) as std::sync::Arc<dyn Repr>,
        ));
        // args_s[0] = receiver (the class PBC), args_s[1] = the
        // attrname constant. Pre-set args_r[0] to the rootclass_repr
        // so `inputargs(class_repr, Void)`'s convertvar takes the
        // identity short-circuit (ClassesPBCRepr → RootClassRepr is
        // a same-lltype conversion that has no explicit pairtype
        // entry yet).
        let root_repr_arc: std::sync::Arc<dyn Repr> = rtyper
            .rootclass_repr
            .borrow()
            .clone()
            .expect("rootclass_repr must be initialised");
        let mut v_recv = Variable::new();
        v_recv.set_concretetype(Some(root_repr_arc.lowleveltype().clone()));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(v_recv));
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(root_repr_arc.clone()));
        let mut s_attr = SomeString::default();
        s_attr.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("__name__")));
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::byte_str("__name__"),
                LowLevelType::Void,
            )));
        hop.args_s.borrow_mut().push(SomeValue::String(s_attr));
        hop.args_r.borrow_mut().push(None);

        let out = r_arc
            .rtype_getattr(&hop)
            .expect("rtype_getattr __name__")
            .expect("non-empty result");
        let Hlvalue::Variable(_) = out else {
            panic!("__name__ arm must return a Variable carrying the genop result");
        };
        let ops = hop.llops.borrow();
        let getfield = ops
            .ops
            .iter()
            .find(|op| op.opname == "getfield")
            .expect("getfield op must be emitted");
        // args[1] is the field-name Void constant `name`.
        let Hlvalue::Constant(field_const) = &getfield.args[1] else {
            panic!("getfield arg[1] must be a Constant");
        };
        assert_eq!(field_const.value, ConstValue::byte_str("name"));
        assert_eq!(field_const.concretetype, Some(LowLevelType::Void));
    }

    /// `get_access_set` returns a structured `MissingRTypeAttribute`
    /// when no ClassAttrFamily exists for the attrname. Upstream
    /// rpbc.py:945 raises directly; pyre surfaces it as a TyperError.
    #[test]
    fn classes_pbc_repr_get_access_set_unknown_attr_raises_missing_rtype_attribute() {
        use crate::annotator::classdesc::ClassDesc;
        use crate::flowspace::model::HostObject;

        let (ann, rtyper) = make_rtyper();
        let host = HostObject::new_class("pkg.C", vec![]);
        let desc = Rc::new(StdRefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            host.clone(),
            "pkg.C".to_string(),
        )));
        ann.bookkeeper
            .descs
            .borrow_mut()
            .insert(host, DescEntry::Class(desc.clone()));

        let s_pbc = SomePBC::new(vec![DescEntry::Class(desc)], true);
        let r = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap();

        let err = r.get_access_set("nonexistent").unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("nonexistent"));
    }

    #[test]
    fn somepbc_rtyper_makerepr_single_class_desc_routes_to_classes_pbc_repr() {
        let (ann, rtyper) = make_rtyper();
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c], false);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "ClassesPBCRepr");
    }

    #[test]
    fn somepbc_rtyper_makerepr_single_frozen_desc_with_can_be_none_surfaces_pending() {
        // Single FrozenDesc with `can_be_None=true` falls out of the
        // SingleFrozenPBCRepr fast-path, hits getFrozenPBCRepr's
        // multi-desc branch, and — with only one access-set (None) —
        // routes to the still-pending MultipleFrozenPBCRepr arm.
        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let s_pbc = SomePBC::new(vec![f], true);
        let err = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("MultipleFrozenPBCRepr"));
    }

    #[test]
    fn function_repr_convert_desc_or_const_routes_desc_through_convert_desc() {
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::translator::rtyper::rmodel::DescOrConst;
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f.clone()], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();

        // Desc arm — routes through convert_desc, which returns the
        // Void-None sentinel.
        let c1 = r
            .convert_desc_or_const(&DescOrConst::Desc(f.clone()))
            .unwrap();
        assert!(matches!(c1.value, ConstValue::None));
        assert_eq!(c1.concretetype, Some(LowLevelType::Void));

        // Const arm — routes through convert_const, same outcome.
        let c2 = r
            .convert_desc_or_const(&DescOrConst::Const(Constant::new(ConstValue::Int(7))))
            .unwrap();
        assert!(matches!(c2.value, ConstValue::None));
        assert_eq!(c2.concretetype, Some(LowLevelType::Void));
    }

    /// Build a single-FunctionDesc callfamily whose simple-call
    /// calltable contains one row pointing at a pre-cached PyGraph.
    /// Mirrors the fixture at
    /// `somepbc_rtyper_makerepr_multi_function_descs_with_callfamily_uses_functionspbcrepr`
    /// but constrained to one desc so `convert_to_concrete_llfn`,
    /// `get_unique_llfn`, and `get_concrete_llfn` all have a
    /// well-formed `(shape, index, funcdesc_key)` triple to look up.
    fn single_funcdesc_with_callfamily(
        ann: &Rc<RPythonAnnotator>,
        name: &str,
    ) -> (Rc<StdRefCell<FunctionDesc>>, CallShape, Rc<PyGraph>) {
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};

        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            name,
            sig,
            None,
            None,
        )));
        let graph_name = format!("{name}_graph");
        let pygraph = super::tests::make_pygraph(&graph_name);
        fd.borrow()
            .cache
            .borrow_mut()
            .insert(GraphCacheKey::None, pygraph.clone());

        let args = ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
            None,
        );
        let shape = args.rawshape();
        FunctionDesc::consider_call_site(&[fd.clone()], &args, &SomeValue::Impossible, None)
            .expect("consider_call_site should populate the calltable");
        (fd, shape, pygraph)
    }

    fn void_variable() -> crate::flowspace::model::Hlvalue {
        let mut v = crate::flowspace::model::Variable::new();
        v.set_concretetype(Some(LowLevelType::Void));
        crate::flowspace::model::Hlvalue::Variable(v)
    }

    fn signed_variable() -> crate::flowspace::model::Hlvalue {
        let mut v = crate::flowspace::model::Variable::new();
        v.set_concretetype(Some(LowLevelType::Signed));
        crate::flowspace::model::Hlvalue::Variable(v)
    }

    fn empty_lloplist(
        rtyper: &Rc<RPythonTyper>,
    ) -> Rc<RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>> {
        Rc::new(RefCell::new(
            crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None),
        ))
    }

    /// Verifies that a `Hlvalue::Constant(LLPtr(Func(...)))` names the
    /// same `PyGraph` identity as `expected`. The `_func` object stores
    /// the graph as a `GraphKey::as_usize()` pointer hash, so identity
    /// comparison runs through [`GraphKey`].
    fn const_points_at_graph(
        value: &crate::flowspace::model::Hlvalue,
        expected: &Rc<PyGraph>,
    ) -> bool {
        use crate::flowspace::model::{ConstValue, GraphKey, Hlvalue};
        use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
        let Hlvalue::Constant(c) = value else {
            return false;
        };
        let ConstValue::LLPtr(llptr) = &c.value else {
            return false;
        };
        let obj = match llptr._obj() {
            Ok(obj) => obj,
            Err(_) => return false,
        };
        let _ptr_obj::Func(func_obj) = obj else {
            return false;
        };
        match func_obj.graph {
            Some(id) => id == GraphKey::of(&expected.graph).as_usize(),
            None => false,
        }
    }

    #[test]
    fn function_repr_convert_to_concrete_llfn_returns_constant_pointing_at_callable_graph() {
        let (ann, rtyper) = make_rtyper();
        let (fd, shape, pygraph) = single_funcdesc_with_callfamily(&ann, "f");

        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd)], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        let llop = empty_lloplist(&rtyper);

        let result = r
            .convert_to_concrete_llfn(&void_variable(), &shape, 0, &llop)
            .unwrap();
        assert!(
            const_points_at_graph(&result, &pygraph),
            "convert_to_concrete_llfn should wrap the row-of-one-graph entry as an LLPtr Constant"
        );
    }

    #[test]
    fn function_repr_convert_to_concrete_llfn_rejects_non_void_input() {
        let (ann, rtyper) = make_rtyper();
        let (fd, shape, _pygraph) = single_funcdesc_with_callfamily(&ann, "f");

        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd)], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        let llop = empty_lloplist(&rtyper);

        let err = r
            .convert_to_concrete_llfn(&signed_variable(), &shape, 0, &llop)
            .unwrap_err();
        assert!(
            err.to_string().contains("concretetype != Void"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn function_repr_get_unique_llfn_returns_constant_for_single_simple_shape() {
        let (ann, rtyper) = make_rtyper();
        let (fd, _shape, pygraph) = single_funcdesc_with_callfamily(&ann, "f");

        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd)], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();

        let result = r.get_unique_llfn().unwrap();
        assert!(
            const_points_at_graph(&result, &pygraph),
            "get_unique_llfn should produce an LLPtr constant aimed at the sole simple-shape graph"
        );
    }

    #[test]
    fn function_repr_get_unique_llfn_errors_when_no_simple_shape_tables_exist() {
        // A FunctionDesc whose callfamily has no calltable entries at
        // all leaves `get_unique_llfn`'s simple-shape filter with an
        // empty `tables` list — `len(tables) != 1` trips upstream's
        // `"cannot pass a function with various call shapes"` branch.
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        // Touch `getcallfamily()` so FunctionReprBase picks up an empty
        // family rather than `None` — otherwise the test would surface
        // the "callfamily not available" guard instead.
        if let DescEntry::Function(rc) = &f {
            rc.borrow().base.getcallfamily().unwrap();
        } else {
            unreachable!();
        }
        let s_pbc = SomePBC::new(vec![f], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        let err = r.get_unique_llfn().unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot pass a function with various call shapes"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn function_repr_get_concrete_llfn_returns_constant_pointing_at_specialized_graph() {
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig,
            None,
            None,
        )));
        // `get_concrete_llfn` calls `funcdesc.get_graph` which routes
        // through `default_specialize` → `maybe_star_args` → cachedgraph
        // with `GraphCacheKey::None` when there are no star args — the
        // pre-populated cache entry is what lets the test avoid a
        // missing-pyobj crash.
        let pygraph = super::tests::make_pygraph("f_graph");
        fd.borrow()
            .cache
            .borrow_mut()
            .insert(GraphCacheKey::None, pygraph.clone());

        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd)], false);
        let r = FunctionRepr::new(&rtyper, s_pbc.clone()).unwrap();

        let result = r
            .get_concrete_llfn(
                &s_pbc,
                vec![SomeValue::Integer(SomeInteger::default())],
                None,
            )
            .unwrap();
        assert!(
            const_points_at_graph(&result, &pygraph),
            "get_concrete_llfn should wrap funcdesc.get_graph's result as an LLPtr Constant"
        );
    }

    #[test]
    fn function_repr_simple_call_emits_direct_call_op() {
        // rpbc.py:199-221 — `FunctionReprBase.call` (Rust binding on
        // `FunctionRepr`). The Void-typed `convert_to_concrete_llfn`
        // returns a Constant funcptr, so the dispatch must take the
        // `direct_call` branch and short-circuit at the
        // `r_result is impossible_repr` check (Void return graph).
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{
            ConstValue, Constant as FlowConstant, Hlvalue, SpaceOperation, Variable,
        };
        use crate::translator::rtyper::rmodel::impossible_repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let (fd, _shape, _pygraph) = single_funcdesc_with_callfamily(&ann, "f");
        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd)], false);
        let r: std::sync::Arc<FunctionRepr> =
            std::sync::Arc::new(FunctionRepr::new(&rtyper, s_pbc.clone()).unwrap());
        let r_dyn: std::sync::Arc<dyn Repr> = r.clone();

        // SpaceOperation: result_var = simple_call(receiver, c_int)
        let mut receiver = Variable::new();
        receiver.set_concretetype(Some(LowLevelType::Void));
        let receiver_h = Hlvalue::Variable(receiver);

        let int_const = FlowConstant::with_concretetype(ConstValue::Int(7), LowLevelType::Signed);
        let int_const_h = Hlvalue::Constant(int_const);

        let mut result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Void));
        let result_h = Hlvalue::Variable(result_var);

        let spaceop = SpaceOperation::new(
            "simple_call",
            vec![receiver_h.clone(), int_const_h.clone()],
            result_h,
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);

        // Seed args_v / args_s / args_r / r_result manually; the test
        // PyGraph carries no annotation propagation so `hop.setup()`
        // would not produce useful Reprs from the spaceop.
        *hop.args_v.borrow_mut() = vec![receiver_h, int_const_h];
        *hop.args_s.borrow_mut() = vec![
            SomeValue::PBC(s_pbc),
            SomeValue::Integer(SomeInteger::default()),
        ];
        *hop.args_r.borrow_mut() = vec![Some(r_dyn.clone()), None];
        // The test PyGraph's returnvar has annotation=Impossible, so
        // upstream rtyper would assign `impossible_repr` to r_result.
        *hop.r_result.borrow_mut() = Some(impossible_repr() as std::sync::Arc<dyn Repr>);

        // upstream rpbc.py:218-219 — `r_result is impossible_repr`
        // short-circuits to None.
        let result = r.rtype_simple_call(&hop).unwrap();
        assert!(
            result.is_none(),
            "Void return + impossible_repr r_result must short-circuit to None"
        );

        // upstream rpbc.py:213 — direct_call is emitted because
        // `convert_to_concrete_llfn` returns a Constant for Void-typed
        // FunctionRepr.
        let llops = hop.llops.borrow();
        let opnames: Vec<&str> = llops.ops.iter().map(|op| op.opname.as_str()).collect();
        assert!(
            opnames.contains(&"direct_call"),
            "expected a direct_call op in llops, got: {opnames:?}"
        );
        assert!(
            !opnames.contains(&"indirect_call"),
            "FunctionRepr (Void-typed funcptr) must not emit indirect_call, got: {opnames:?}"
        );
    }

    #[test]
    fn functions_pbc_repr_simple_call_emits_indirect_call_op() {
        // rpbc.py:199-221 (`FunctionReprBase.call`) +
        // rpbc.py:300-312 (`FunctionsPBCRepr.convert_to_concrete_llfn`).
        // The single-row `convert_to_concrete_llfn` returns `v`
        // unchanged — when `v` is a `Variable` (runtime funcptr),
        // the dispatch must take the `indirect_call` branch and
        // append the `c_graphs` placeholder Constant.
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{
            ConstValue, Constant as FlowConstant, Hlvalue, SpaceOperation, Variable,
        };
        use crate::translator::rtyper::rmodel::impossible_repr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let (fd, _shape, _pygraph) = single_funcdesc_with_callfamily(&ann, "f");
        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd)], false);
        let r: std::sync::Arc<FunctionsPBCRepr> =
            std::sync::Arc::new(FunctionsPBCRepr::new(&rtyper, s_pbc.clone()).unwrap());
        let r_dyn: std::sync::Arc<dyn Repr> = r.clone();
        let funcptr_lltype = r.lowleveltype().clone();

        // SpaceOperation: result_var = simple_call(receiver_funcptr, c_int)
        // The receiver carries the PBCRepr's lowleveltype = Ptr(FuncType(...)),
        // mirroring upstream's `vfn = hop.inputarg(self, arg=0)` path
        // where `self` is the FunctionsPBCRepr.
        let mut receiver = Variable::new();
        receiver.set_concretetype(Some(funcptr_lltype.clone()));
        let receiver_h = Hlvalue::Variable(receiver);

        let int_const = FlowConstant::with_concretetype(ConstValue::Int(7), LowLevelType::Signed);
        let int_const_h = Hlvalue::Constant(int_const);

        let mut result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Void));
        let result_h = Hlvalue::Variable(result_var);

        let spaceop = SpaceOperation::new(
            "simple_call",
            vec![receiver_h.clone(), int_const_h.clone()],
            result_h,
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);

        // Seed args_v / args_s / args_r / r_result manually; the test
        // PyGraph carries no annotation propagation so `hop.setup()`
        // would not produce useful Reprs from the spaceop.
        *hop.args_v.borrow_mut() = vec![receiver_h, int_const_h];
        *hop.args_s.borrow_mut() = vec![
            SomeValue::PBC(s_pbc),
            SomeValue::Integer(SomeInteger::default()),
        ];
        *hop.args_r.borrow_mut() = vec![Some(r_dyn.clone()), None];
        // The test PyGraph's returnvar has annotation=Impossible, so
        // upstream rtyper would assign `impossible_repr` to r_result.
        *hop.r_result.borrow_mut() = Some(impossible_repr() as std::sync::Arc<dyn Repr>);

        // upstream rpbc.py:218-219 — `r_result is impossible_repr`
        // short-circuits to None.
        let result = r.rtype_simple_call(&hop).unwrap();
        assert!(
            result.is_none(),
            "Void return + impossible_repr r_result must short-circuit to None"
        );

        // upstream rpbc.py:215-217 — indirect_call is emitted because
        // `convert_to_concrete_llfn` returned `v` (a Variable funcptr,
        // not a Constant) for the single-row FunctionsPBCRepr case.
        let llops = hop.llops.borrow();
        let opnames: Vec<&str> = llops.ops.iter().map(|op| op.opname.as_str()).collect();
        assert!(
            opnames.contains(&"indirect_call"),
            "expected an indirect_call op in llops, got: {opnames:?}"
        );
        assert!(
            !opnames.contains(&"direct_call"),
            "FunctionsPBCRepr (Variable funcptr) must not emit direct_call, got: {opnames:?}"
        );

        // upstream rpbc.py:216 — the `c_graphs` placeholder Constant
        // is appended as the last arg of the indirect_call op.
        let indirect = llops
            .ops
            .iter()
            .find(|op| op.opname == "indirect_call")
            .expect("indirect_call op must exist");
        assert!(
            matches!(indirect.args.last(), Some(Hlvalue::Constant(_))),
            "indirect_call's last arg must be a Constant (c_graphs placeholder), \
             got: {:?}",
            indirect.args.last()
        );
    }

    #[test]
    fn single_frozen_pbc_repr_rtype_getattr_requires_constant_s_result() {
        // rpbc.py:642-645 — mirror of the MU test; getattr on a Void
        // PBC must surface a constant, otherwise raise.
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let r = SingleFrozenPBCRepr::new(f);
        let spaceop = SpaceOperation::new(
            OpKind::GetAttr.opname(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.s_result
            .replace(Some(crate::annotator::model::SomeValue::Impossible));
        hop.r_result.replace(Some(
            std::sync::Arc::new(SingleFrozenPBCRepr::new(frozen_entry(
                &ann.bookkeeper,
                "frozen_r",
            ))) as std::sync::Arc<dyn Repr>,
        ));
        let err = r.rtype_getattr(&hop).unwrap_err();
        assert!(
            err.to_string()
                .contains("getattr on a constant PBC returns a non-constant"),
            "unexpected: {err}",
        );
    }

    #[test]
    fn functions_pbc_repr_convert_desc_returns_llptr_constant_and_caches_it() {
        // rpbc.py:255-287 — single-row convert_desc must produce a
        // Constant wrapping the row's _ptr, and subsequent calls must
        // hit the funccache.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::ConstValue;

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        let args = ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();

        let f_entry = DescEntry::Function(fd_f.clone());
        let s_pbc = SomePBC::new(vec![f_entry.clone(), DescEntry::Function(fd_g)], false);
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();

        let c1 = r.convert_desc(&f_entry).unwrap();
        assert!(matches!(c1.value, ConstValue::LLPtr(_)));
        // Re-invoking should hit the funccache.
        let before = r.funccache.borrow().len();
        let c2 = r.convert_desc(&f_entry).unwrap();
        assert_eq!(c1, c2);
        assert_eq!(r.funccache.borrow().len(), before);
    }

    #[test]
    fn functions_pbc_repr_convert_const_none_returns_null_pointer_constant() {
        // rpbc.py:294-296 — `value is None` arm emits
        // `nullptr(self.lowleveltype.TO)`.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::ConstValue;

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        let args = ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();

        let s_pbc = SomePBC::new(
            vec![DescEntry::Function(fd_f), DescEntry::Function(fd_g)],
            false,
        );
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();
        let c = r.convert_const(&ConstValue::None).unwrap();
        let ConstValue::LLPtr(llptr) = &c.value else {
            panic!("expected LLPtr");
        };
        assert!(!llptr.nonzero(), "convert_const(None) must be a null ptr");
        assert_eq!(c.concretetype.as_ref(), Some(r.lowleveltype()));
    }

    #[test]
    fn functions_pbc_repr_convert_const_non_none_defers_pending_getdesc() {
        // Non-None convert_const requires bookkeeper.getdesc dispatch;
        // until that lands, the call surfaces MissingRTypeOperation.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::ConstValue;

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &ArgumentsForTranslation::new(
                vec![SomeValue::Integer(SomeInteger::default())],
                None,
                None,
            ),
            &SomeValue::Impossible,
            None,
        )
        .unwrap();
        let s_pbc = SomePBC::new(
            vec![DescEntry::Function(fd_f), DescEntry::Function(fd_g)],
            false,
        );
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();
        let err = r.convert_const(&ConstValue::Int(7)).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("bookkeeper.getdesc"));
    }

    #[test]
    fn functions_pbc_repr_rtype_bool_constant_fast_path_returns_bool_constant() {
        // rmodel.py:255-260 — CanBeNull.rtype_bool fast-path: when the
        // annotator has proved the result constant, emit inputconst(Bool, ...).
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeBool, SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, Constant, Hlvalue, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &ArgumentsForTranslation::new(
                vec![SomeValue::Integer(SomeInteger::default())],
                None,
                None,
            ),
            &SomeValue::Impossible,
            None,
        )
        .unwrap();
        let s_pbc = SomePBC::new(
            vec![DescEntry::Function(fd_f), DescEntry::Function(fd_g)],
            false,
        );
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();

        let spaceop = SpaceOperation::new(
            OpKind::Bool.opname(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops.clone());
        // Constant annotation → fast-path: emits a Bool constant.
        let mut sb = SomeBool::new();
        sb.base.const_box = Some(Constant::new(ConstValue::Bool(true)));
        hop.s_result.replace(Some(SomeValue::Bool(sb)));

        let result = r
            .rtype_bool(&hop)
            .unwrap()
            .expect("fast path returns value");
        let Hlvalue::Constant(c) = result else {
            panic!("expected Bool Constant, got {result:?}");
        };
        assert_eq!(c.concretetype, Some(LowLevelType::Bool));
        assert!(matches!(c.value, ConstValue::Bool(true)));
        assert!(
            llops.borrow().ops.is_empty(),
            "fast-path should emit no ops"
        );
    }

    #[test]
    fn functions_pbc_repr_rtype_bool_non_constant_emits_ptr_nonzero() {
        // rmodel.py:259-260 — non-constant s_result path: emit a
        // ptr_nonzero op against a Bool resulttype.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{Hlvalue, SpaceOperation, Variable};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &ArgumentsForTranslation::new(
                vec![SomeValue::Integer(SomeInteger::default())],
                None,
                None,
            ),
            &SomeValue::Impossible,
            None,
        )
        .unwrap();
        let s_pbc = SomePBC::new(
            vec![DescEntry::Function(fd_f), DescEntry::Function(fd_g)],
            false,
        );
        let r: std::sync::Arc<dyn Repr> =
            std::sync::Arc::new(FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap());

        // Input arg: a Ptr-typed Variable (matches FunctionsPBCRepr lowleveltype).
        let mut arg_var = Variable::new();
        arg_var.set_concretetype(Some(r.lowleveltype().clone()));
        let arg = Hlvalue::Variable(arg_var);
        let spaceop = SpaceOperation::new(
            OpKind::Bool.opname(),
            vec![arg.clone()],
            Hlvalue::Variable(Variable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops.clone());
        hop.args_v.borrow_mut().push(arg);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r.clone()));
        // Non-constant s_result → ptr_nonzero path.
        hop.s_result.replace(Some(SomeValue::Impossible));

        let result = r
            .rtype_bool(&hop)
            .unwrap()
            .expect("ptr_nonzero returns value");
        assert!(matches!(result, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert_eq!(
            ops.ops.len(),
            1,
            "ptr_nonzero should be the only emitted op"
        );
        assert_eq!(ops.ops[0].opname, "ptr_nonzero");
    }

    #[test]
    fn functions_pbc_repr_single_uniquerow_populates_concretetable_and_funccache() {
        // rpbc.py:227-240 — ensure the new single-row ctor path wires
        // `concretetable`, `uniquerows`, and the funccache correctly.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};

        let (ann, rtyper) = make_rtyper();
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd_f = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "g",
            sig,
            None,
            None,
        )));
        for (desc, name) in [(&fd_f, "f_graph"), (&fd_g, "g_graph")] {
            desc.borrow()
                .cache
                .borrow_mut()
                .insert(GraphCacheKey::None, super::tests::make_pygraph(name));
        }
        let args = ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
            None,
        );
        FunctionDesc::consider_call_site(
            &[fd_f.clone(), fd_g.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap();

        let s_pbc = SomePBC::new(
            vec![DescEntry::Function(fd_f), DescEntry::Function(fd_g)],
            false,
        );
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(
            r.uniquerows.len(),
            1,
            "two FunctionDescs sharing a shape merge into one uniquerow"
        );
        assert_eq!(r.concretetable.len(), 1);
        assert!(r.funccache.borrow().is_empty());
    }

    #[test]
    fn function_repr_get_r_implfunc_returns_self_and_zero() {
        // rpbc.py:186-187 — upstream `return self, 0`. Port exposes
        // `(self_as_dyn_repr, 0)`; identity is checked by comparing
        // pointer against the concrete FunctionRepr receiver.
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        let (r_impl, nimplicit) = r.get_r_implfunc().unwrap();
        assert_eq!(nimplicit, 0);
        assert!(std::ptr::eq(
            r_impl as *const dyn Repr as *const (),
            &r as *const FunctionRepr as *const (),
        ));
    }

    #[test]
    fn function_repr_base_get_s_signatures_delegates_to_funcdesc() {
        // rpbc.py:189-191 — upstream `return funcdesc.get_s_signatures(shape)`.
        // Without a populated calltable, `FunctionDesc.get_s_signatures`
        // returns an empty Vec (its `calltables.get(shape)` miss path).
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        if let DescEntry::Function(rc) = &f {
            rc.borrow().base.getcallfamily().unwrap();
        }
        let s_pbc = SomePBC::new(vec![f], false);
        let r = FunctionRepr::new(&rtyper, s_pbc).unwrap();
        let sigs = r
            .base()
            .get_s_signatures(&crate::flowspace::argument::CallShape {
                shape_cnt: 0,
                shape_keys: Vec::new(),
                shape_star: false,
            })
            .unwrap();
        assert!(sigs.is_empty());
    }
}
