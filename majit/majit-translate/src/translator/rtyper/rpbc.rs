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
    fn setup_specfunc_builds_ptr_struct_with_one_field_per_uniquerow_with_immutable_hints() {
        // rpbc.py:242-247 — `Ptr(Struct('specfunc', *fields,
        // hints={immutable: True, static_immutable: True}))`.
        use crate::flowspace::model::ConstValue;
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

        let row0 = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::new(),
            fntype: FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Void,
            },
            attrname: Some("variant0".to_string()),
        }));
        let row1 = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::new(),
            fntype: FuncType {
                args: vec![LowLevelType::Signed, LowLevelType::Signed],
                result: LowLevelType::Bool,
            },
            attrname: Some("variant1".to_string()),
        }));

        let lltype = super::FunctionsPBCRepr::setup_specfunc(&[row0, row1]).unwrap();
        let LowLevelType::Ptr(ptr) = &lltype else {
            panic!("expected Ptr");
        };
        let PtrTarget::Struct(s) = &ptr.TO else {
            panic!("expected Ptr(Struct)");
        };
        assert_eq!(s._name, "specfunc");
        let fields: Vec<_> = s._flds.iter().collect();
        let field_names: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
        assert!(field_names.contains(&"variant0"));
        assert!(field_names.contains(&"variant1"));
        let hint_immutable = s
            ._hints
            .iter()
            .find(|(k, _)| k.as_str() == "immutable")
            .map(|(_, v)| v.clone());
        let hint_static = s
            ._hints
            .iter()
            .find(|(k, _)| k.as_str() == "static_immutable")
            .map(|(_, v)| v.clone());
        assert_eq!(hint_immutable, Some(ConstValue::Bool(true)));
        assert_eq!(hint_static, Some(ConstValue::Bool(true)));
    }

    #[test]
    fn setup_specfunc_rejects_row_without_attrname() {
        let row = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::new(),
            fntype: FuncType {
                args: vec![],
                result: crate::translator::rtyper::lltypesystem::lltype::LowLevelType::Void,
            },
            attrname: None,
        }));
        let err = super::FunctionsPBCRepr::setup_specfunc(&[row]).unwrap_err();
        assert!(err.to_string().contains("missing attrname"));
    }

    #[test]
    fn convert_desc_multi_row_fills_specfunc_struct_fields_with_variant_llfns() {
        // rpbc.py:281-285 — multi-row path mallocs a specfunc struct
        // and `setattr(result, attrname, llfn)` for each variant.
        // Construct a two-row FunctionsPBCRepr by hand and verify
        // convert_desc(fd) returns a Constant whose inner _struct has
        // variant0 = fd's row0 llfn, variant1 = fd's row1 llfn.
        use crate::translator::rtyper::lltypesystem::lltype::{
            _ptr_obj, LowLevelType, LowLevelValue,
        };

        let desc_a = DescKey(10);
        let llfn_a_row0 = fake_llfn("a_r0", 1);
        let llfn_a_row1 = fake_llfn("a_r1", 2);
        let row0 = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::from([(desc_a, llfn_a_row0.clone())]),
            fntype: FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Void,
            },
            attrname: Some("variant0".to_string()),
        }));
        let row1 = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::from([(desc_a, llfn_a_row1.clone())]),
            fntype: FuncType {
                args: vec![LowLevelType::Signed, LowLevelType::Signed],
                result: LowLevelType::Void,
            },
            attrname: Some("variant1".to_string()),
        }));

        let specfunc_lltype =
            super::FunctionsPBCRepr::setup_specfunc(&[row0.clone(), row1.clone()]).unwrap();

        let (ann, rtyper) = make_rtyper();
        let rtyper = Rc::new(rtyper);
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "fn_a",
            int_sig(&["x"]),
            None,
            None,
        )));
        // Align the FunctionDesc's identity with desc_a so rowkey() matches.
        fd.borrow_mut().base.identity = desc_a;
        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd.clone())], false);
        let base = FunctionReprBase {
            rtyper: Rc::downgrade(&rtyper),
            s_pbc,
            callfamily: None,
        };
        let r = super::FunctionsPBCRepr {
            base,
            concretetable: HashMap::new(),
            uniquerows: vec![row0, row1],
            lltype: specfunc_lltype.clone(),
            funccache: RefCell::new(HashMap::new()),
            state: super::super::rmodel::ReprState::new(),
        };

        let c = r.convert_desc(&DescEntry::Function(fd.clone())).unwrap();
        let ConstValue::LLPtr(ptr) = &c.value else {
            panic!("expected LLPtr");
        };
        let Ok(Some(_ptr_obj::Struct(s))) = &ptr._obj0 else {
            panic!("specfunc struct should be present");
        };
        let variant0 = s._getattr("variant0").expect("variant0 field");
        let variant1 = s._getattr("variant1").expect("variant1 field");
        assert_eq!(variant0, &LowLevelValue::Ptr(Box::new(llfn_a_row0)));
        assert_eq!(variant1, &LowLevelValue::Ptr(Box::new(llfn_a_row1)));

        // Second call hits the funccache and returns the identical constant.
        let c2 = r.convert_desc(&DescEntry::Function(fd)).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn convert_desc_multi_row_missing_entry_fills_nullptr_in_variant_field() {
        // rpbc.py:267-270 — missing-desc entries populate the variant
        // with `nullptr(row.fntype.TO)` instead of erroring.
        use crate::translator::rtyper::lltypesystem::lltype::{
            _ptr_obj, LowLevelType, LowLevelValue,
        };

        let desc_a = DescKey(10);
        let desc_b = DescKey(11);
        let llfn_a_row0 = fake_llfn("a_r0", 1);
        // row1 is missing desc_a entirely.
        let row0 = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::from([(desc_a, llfn_a_row0.clone())]),
            fntype: FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Void,
            },
            attrname: Some("variant0".to_string()),
        }));
        let row1 = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::from([(desc_b, fake_llfn("b_r1", 2))]),
            fntype: FuncType {
                args: vec![LowLevelType::Signed, LowLevelType::Signed],
                result: LowLevelType::Void,
            },
            attrname: Some("variant1".to_string()),
        }));

        let specfunc_lltype =
            super::FunctionsPBCRepr::setup_specfunc(&[row0.clone(), row1.clone()]).unwrap();

        let (ann, rtyper) = make_rtyper();
        let rtyper = Rc::new(rtyper);
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "fn_a",
            int_sig(&["x"]),
            None,
            None,
        )));
        fd.borrow_mut().base.identity = desc_a;
        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd.clone())], false);
        let r = super::FunctionsPBCRepr {
            base: FunctionReprBase {
                rtyper: Rc::downgrade(&rtyper),
                s_pbc,
                callfamily: None,
            },
            concretetable: HashMap::new(),
            uniquerows: vec![row0, row1],
            lltype: specfunc_lltype,
            funccache: RefCell::new(HashMap::new()),
            state: super::super::rmodel::ReprState::new(),
        };

        let c = r.convert_desc(&DescEntry::Function(fd)).unwrap();
        let ConstValue::LLPtr(ptr) = &c.value else {
            panic!("expected LLPtr");
        };
        let Ok(Some(_ptr_obj::Struct(s))) = &ptr._obj0 else {
            panic!("specfunc struct should be present");
        };
        let variant0 = s._getattr("variant0").expect("variant0 field");
        let variant1 = s._getattr("variant1").expect("variant1 field");
        assert_eq!(variant0, &LowLevelValue::Ptr(Box::new(llfn_a_row0)));
        // variant1 is a null function pointer — same Ptr type, no _obj0.
        let LowLevelValue::Ptr(null_ptr) = variant1 else {
            panic!("variant1 should be a Ptr");
        };
        assert!(matches!(null_ptr._obj0, Ok(None)));
    }

    #[test]
    fn convert_to_concrete_llfn_single_row_returns_v_unchanged() {
        // rpbc.py:306-307 — `if len(self.uniquerows) == 1: return v`.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        // Construct a single-row FunctionsPBCRepr end-to-end.
        let (ann, rtyper) = make_rtyper();
        let rtyper = Rc::new(rtyper);
        let sig = int_sig(&["x"]);
        let fd_f = Rc::new(RefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "f",
            sig.clone(),
            None,
            None,
        )));
        let fd_g = Rc::new(RefCell::new(FunctionDesc::new(
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
                .insert(GraphCacheKey::None, make_pygraph(name));
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
        let r = super::FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();
        assert_eq!(r.uniquerows.len(), 1);

        let shape = r.concretetable.keys().next().unwrap().0.clone();
        let input = Variable::new();
        input.set_concretetype(Some(r.lowleveltype().clone()));
        let v_in = Hlvalue::Variable(input);
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let out = r
            .convert_to_concrete_llfn(&v_in, &shape, 0, &mut llops)
            .unwrap();
        assert_eq!(out, v_in);
        assert!(llops.ops.is_empty(), "single-row path should not emit ops");
    }

    #[test]
    fn convert_to_concrete_llfn_multi_row_emits_getfield_on_specfunc_struct() {
        // rpbc.py:308-312 — multi-row emits `getfield(v, c_rowname)`
        // with resulttype = Ptr(FuncType) and variant-named Void const.
        use crate::flowspace::model::{Hlvalue, Variable};
        use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, Ptr as LlPtr};
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let desc_a = DescKey(10);
        let row0 = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::from([(desc_a, fake_llfn("a_r0", 1))]),
            fntype: FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Void,
            },
            attrname: Some("variant0".to_string()),
        }));
        let row1 = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::from([(desc_a, fake_llfn("a_r1", 2))]),
            fntype: FuncType {
                args: vec![LowLevelType::Signed, LowLevelType::Signed],
                result: LowLevelType::Bool,
            },
            attrname: Some("variant1".to_string()),
        }));
        let specfunc_lltype =
            super::FunctionsPBCRepr::setup_specfunc(&[row0.clone(), row1.clone()]).unwrap();
        let shape0 = CallShape {
            shape_cnt: 1,
            shape_keys: Vec::new(),
            shape_star: false,
        };
        let shape1 = CallShape {
            shape_cnt: 2,
            shape_keys: Vec::new(),
            shape_star: false,
        };
        let concretetable = HashMap::from([
            ((shape0.clone(), 0), row0.clone()),
            ((shape1.clone(), 0), row1.clone()),
        ]);

        let (ann, rtyper) = make_rtyper();
        let rtyper = Rc::new(rtyper);
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "fn_a",
            int_sig(&["x"]),
            None,
            None,
        )));
        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd)], false);
        let r = super::FunctionsPBCRepr {
            base: FunctionReprBase {
                rtyper: Rc::downgrade(&rtyper),
                s_pbc,
                callfamily: None,
            },
            concretetable,
            uniquerows: vec![row0, row1],
            lltype: specfunc_lltype.clone(),
            funccache: RefCell::new(HashMap::new()),
            state: super::super::rmodel::ReprState::new(),
        };

        let input = Variable::new();
        input.set_concretetype(Some(specfunc_lltype));
        let v_in = Hlvalue::Variable(input);
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let out = r
            .convert_to_concrete_llfn(&v_in, &shape1, 0, &mut llops)
            .unwrap();

        // Output should be a Variable whose concretetype is Ptr(FuncType) of row1.
        let Hlvalue::Variable(out_var) = &out else {
            panic!("expected Variable");
        };
        let expected_ty = LowLevelType::Ptr(Box::new(LlPtr {
            TO: PtrTarget::Func(FuncType {
                args: vec![LowLevelType::Signed, LowLevelType::Signed],
                result: LowLevelType::Bool,
            }),
        }));
        assert_eq!(out_var.concretetype().as_ref(), Some(&expected_ty));
        assert_eq!(llops.ops.len(), 1);
        assert_eq!(llops.ops[0].opname, "getfield");
    }

    #[test]
    fn convert_desc_single_row_funcdesc_missing_emits_rffi_cast_bogus_pointer() {
        // rpbc.py:275-280 — extremely rare fallback: when the single
        // uniquerow doesn't contain the funcdesc, `rffi.cast` produces
        // a non-null non-None placeholder pointer that shouldn't be
        // called but mustn't compare equal to None either.
        use crate::translator::rtyper::lltypesystem::lltype::{_ptr_obj, LowLevelType};

        // Build a single-row FunctionsPBCRepr with one desc (desc_a)
        // wired into the row, then call convert_desc with a different
        // desc (desc_b) that is NOT in the row.
        let desc_a = DescKey(10);
        let desc_b = DescKey(11);
        let row0 = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::from([(desc_a, fake_llfn("a_only", 1))]),
            fntype: FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Void,
            },
            // Upstream uses attrname = None in the single-row case.
            attrname: None,
        }));

        let (ann, rtyper) = make_rtyper();
        let rtyper = Rc::new(rtyper);
        let fd_b = Rc::new(RefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "fn_b",
            int_sig(&["x"]),
            None,
            None,
        )));
        fd_b.borrow_mut().base.identity = desc_b;

        let row_lltype = LowLevelType::Ptr(Box::new(
            crate::translator::rtyper::lltypesystem::lltype::Ptr {
                TO: PtrTarget::Func(row0.borrow().fntype.clone()),
            },
        ));
        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd_b.clone())], false);
        let r = super::FunctionsPBCRepr {
            base: FunctionReprBase {
                rtyper: Rc::downgrade(&rtyper),
                s_pbc,
                callfamily: None,
            },
            concretetable: HashMap::new(),
            uniquerows: vec![row0],
            lltype: row_lltype,
            funccache: RefCell::new(HashMap::new()),
            state: super::super::rmodel::ReprState::new(),
        };

        let c = r.convert_desc(&DescEntry::Function(fd_b)).unwrap();
        let ConstValue::LLPtr(ptr) = &c.value else {
            panic!("expected LLPtr constant");
        };
        // Non-null: _obj0 is Ok(Some(_ptr_obj::Func(...))).
        let Ok(Some(_ptr_obj::Func(func))) = &ptr._obj0 else {
            panic!("expected Ok(Some(Func(_))), got {:?}", ptr._obj0);
        };
        // Name carries the fallback marker so two subsequent fallbacks
        // keep distinct identities through _func structural equality.
        assert!(
            func._name.starts_with("<rffi.cast_pbc_fallback_"),
            "unexpected synthetic name: {:?}",
            func._name
        );
        // funccache still caches the result so a second convert_desc
        // returns the identical constant.
        let c2 = r
            .convert_desc(&DescEntry::Function(Rc::new(RefCell::new(
                FunctionDesc::new(
                    ann.bookkeeper.clone(),
                    None,
                    "fn_b",
                    int_sig(&["x"]),
                    None,
                    None,
                ),
            ))))
            .unwrap();
        // Distinct desc → different entry → distinct fallback_id → different synthetic name.
        let ConstValue::LLPtr(p2) = &c2.value else {
            panic!("expected LLPtr");
        };
        let Ok(Some(_ptr_obj::Func(f2))) = &p2._obj0 else {
            panic!("expected second fallback to also mint a Func ptr");
        };
        assert_ne!(func._name, f2._name);
    }

    #[test]
    fn create_specfunc_allocates_immortal_struct_pointer_matching_lowleveltype() {
        // rpbc.py:249-250 — `malloc(self.lowleveltype.TO, immortal=True)`.
        // Test via direct struct instantiation: build a FunctionsPBCRepr
        // shell with a specfunc lowleveltype, then confirm
        // `create_specfunc` returns a `_ptr` whose `_TYPE` equals the
        // repr's lowleveltype and whose solid/immortal metadata is set.
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

        let row0 = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::new(),
            fntype: FuncType {
                args: vec![LowLevelType::Signed],
                result: LowLevelType::Void,
            },
            attrname: Some("variant0".to_string()),
        }));
        let row1 = Rc::new(RefCell::new(ConcreteCallTableRow {
            row: HashMap::new(),
            fntype: FuncType {
                args: vec![LowLevelType::Signed, LowLevelType::Signed],
                result: LowLevelType::Bool,
            },
            attrname: Some("variant1".to_string()),
        }));
        let specfunc_lltype = super::FunctionsPBCRepr::setup_specfunc(&[row0, row1]).unwrap();

        // Build a bare-bones FunctionsPBCRepr so we can exercise
        // create_specfunc without running get_concrete_calltable.
        let (ann, rtyper) = make_rtyper();
        let rtyper = Rc::new(rtyper);
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "fn",
            int_sig(&["x"]),
            None,
            None,
        )));
        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd)], false);
        let base = FunctionReprBase {
            rtyper: Rc::downgrade(&rtyper),
            s_pbc,
            callfamily: None,
        };
        let r = super::FunctionsPBCRepr {
            base,
            concretetable: HashMap::new(),
            uniquerows: Vec::new(),
            lltype: specfunc_lltype.clone(),
            funccache: RefCell::new(HashMap::new()),
            state: super::super::rmodel::ReprState::new(),
        };
        let ptr = r.create_specfunc().unwrap();
        assert_eq!(
            LowLevelType::Ptr(Box::new(ptr._TYPE.clone())),
            specfunc_lltype
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

use crate::flowspace::model::{ConstValue, Constant};
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{Repr, ReprState};
use std::rc::Weak;

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

    /// Row-selection half of RPython `FunctionReprBase.call(self, hop)`
    /// (rpbc.py:200-207) — shared between `FunctionRepr` and
    /// `FunctionsPBCRepr` because the `convert_to_concrete_llfn` step
    /// differs by subtype. Returns the selected row plus the input
    /// `Hlvalue` for arg 0 (the receiver passed through
    /// `hop.inputarg(self, 0)`).
    pub fn call_setup(
        &self,
        self_repr: &dyn Repr,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> Result<(SelectedCallFamilyRow, crate::flowspace::model::Hlvalue), TyperError> {
        use crate::annotator::model::SomeValue;

        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("FunctionReprBase.call: rtyper weak reference dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("FunctionReprBase.call: annotator weak reference dropped")
        })?;
        let bk = &annotator.bookkeeper;

        // upstream: `args = hop.spaceop.build_args(hop.args_s[1:])`.
        let args_s: Vec<SomeValue> = hop.args_s.borrow().iter().skip(1).cloned().collect();
        let args = crate::annotator::bookkeeper::build_args_for_op(&hop.spaceop.opname, &args_s)
            .map_err(|e| TyperError::message(e.to_string()))?;

        // upstream: `s_pbc = hop.args_s[0]   # possibly more precise than self.s_pbc`.
        let s_pbc = match hop.args_s.borrow().first().cloned() {
            Some(SomeValue::PBC(p)) => p,
            Some(other) => {
                return Err(TyperError::message(format!(
                    "FunctionReprBase.call: args_s[0] is not SomePBC (got {other:?})"
                )));
            }
            None => {
                return Err(TyperError::message(
                    "FunctionReprBase.call: args_s is empty",
                ));
            }
        };

        // upstream: `shape, index = self.callfamily.find_row(bk, descs,
        // args, hop.spaceop)`. Upstream uses `id(hop.spaceop)` as the
        // call-site identity for call-site-specific graph
        // specialization caching. Pyre threads the real
        // `(graph, block, op_index)` triple through
        // [`HighLevelOp::position_key`] — populated by
        // [`RPythonTyper::highlevelops`] — so this lookup keys on the
        // same identity the annotator used when building the
        // calltable.
        let op_key = hop.position_key.clone();

        let callfamily = self.callfamily.as_ref().ok_or_else(|| {
            TyperError::message("FunctionReprBase.call: callfamily not available")
        })?;
        let row = select_call_family_row(bk, callfamily, &s_pbc, &args, op_key)
            .map_err(|e| TyperError::message(e.to_string()))?;

        // upstream: `vfn = hop.inputarg(self, arg=0)`.
        let vfn = hop.inputarg(self_repr, 0)?;

        Ok((row, vfn))
    }

    /// Trailing half of RPython `FunctionReprBase.call(self, hop)`
    /// (rpbc.py:209-221) — consumes the already-computed `v_func` from
    /// the subtype's `convert_to_concrete_llfn` plus the row witnesses
    /// from [`Self::call_setup`], then emits callparse + exception
    /// wiring + `direct_call` / `indirect_call` + return-value convert.
    pub fn call_emit(
        &self,
        v_func: crate::flowspace::model::Hlvalue,
        row: &SelectedCallFamilyRow,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> Result<Option<crate::flowspace::model::Hlvalue>, TyperError> {
        use crate::flowspace::model::{ConstValue, Hlvalue};
        use crate::translator::rtyper::rtyper::{GenopResult, HighLevelOp};

        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("FunctionReprBase.call: rtyper weak reference dropped")
        })?;

        // upstream: `vlist += callparse.callparse(self.rtyper, anygraph, hop)`.
        let vargs = super::callparse::callparse(&rtyper, &row.anygraph, hop, None)?;
        let mut vlist = Vec::with_capacity(vargs.len() + 2);
        vlist.push(v_func);
        vlist.extend(vargs);

        // upstream: `rresult = callparse.getrresult(self.rtyper, anygraph)`.
        let rresult = super::callparse::getrresult(&rtyper, &row.anygraph)?;

        // upstream: `hop.exception_is_here()`.
        hop.exception_is_here()?;

        // upstream: `if isinstance(vlist[0], Constant): direct_call ;
        // else: vlist.append(c_graphs); indirect_call`.
        let opname = if matches!(&vlist[0], Hlvalue::Constant(_)) {
            "direct_call"
        } else {
            // upstream rpbc.py:216 — `vlist.append(hop.inputconst(Void,
            // row_of_graphs.values()))`. The c_graphs constant carries
            // the set of target graph identities for the
            // indirect_call. Pyre routes through the dedicated
            // `ConstValue::Graphs(Vec<usize>)` carrier (flowspace/
            // model.rs:1735) — one entry per PyGraph identity — so
            // consumers can distinguish a graph-list Void constant
            // from any other usize-list payload.
            let graph_ids: Vec<usize> = row
                .row_of_graphs
                .values()
                .map(|g| crate::flowspace::model::GraphKey::of(&g.graph).as_usize())
                .collect();
            vlist.push(Hlvalue::Constant(HighLevelOp::inputconst(
                &LowLevelType::Void,
                &ConstValue::Graphs(graph_ids),
            )?));
            // Record every graph in the row so downstream call-graph
            // tracking stays accurate alongside the c_graphs slot.
            for graph in row.row_of_graphs.values() {
                hop.llops.borrow_mut().record_extra_call(&graph.graph)?;
            }
            "indirect_call"
        };

        let v = hop
            .genop(opname, vlist, GenopResult::LLType(rresult.lowleveltype()))
            .ok_or_else(|| {
                TyperError::message(
                    "FunctionReprBase.call: genop returned None for non-Void result",
                )
            })?;

        // upstream rpbc.py:218 — `if hop.r_result is impossible_repr:
        // return None`. Pyre stores `impossible_repr` as the singleton
        // `VoidRepr` returned by [`crate::translator::rtyper::rmodel::impossible_repr`],
        // wrapped in `Some(Arc<dyn Repr>)` on the hop's `r_result`
        // slot. Compare via `Arc::ptr_eq` against that singleton — a
        // bare `None` slot is a setup error, not the impossible-repr
        // signal.
        let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
            TyperError::message(
                "FunctionReprBase.call: hop.r_result is not set (call hop.setup() first)",
            )
        })?;
        let impossible: std::sync::Arc<dyn Repr> =
            crate::translator::rtyper::rmodel::impossible_repr();
        if std::sync::Arc::ptr_eq(&r_result, &impossible) {
            return Ok(None);
        }
        match rresult {
            super::callparse::GetrresultKind::Repr(src) => {
                let converted =
                    hop.llops
                        .borrow_mut()
                        .convertvar(v, src.as_ref(), r_result.as_ref())?;
                Ok(Some(converted))
            }
            super::callparse::GetrresultKind::Void => {
                if r_result.lowleveltype() == &LowLevelType::Void {
                    Ok(Some(v))
                } else {
                    Err(TyperError::message(
                        "FunctionReprBase.call: void callee return cannot convert to non-Void r_result",
                    ))
                }
            }
        }
    }
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
        _llop: &crate::translator::rtyper::rtyper::LowLevelOpList,
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
    /// (rpbc.py:193-194) — inherited by `FunctionRepr`. Delegates to
    /// the shared `FunctionReprBase::call_setup` / `::call_emit`
    /// scaffolding with `FunctionRepr`'s Void-typed
    /// `convert_to_concrete_llfn`.
    fn rtype_simple_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        let (row, vfn) = self.base.call_setup(self, hop)?;
        let v_func =
            self.convert_to_concrete_llfn(&vfn, &row.shape, row.index, &hop.llops.borrow())?;
        self.base.call_emit(v_func, &row, hop)
    }

    /// RPython `FunctionReprBase.rtype_call_args(self, hop)`
    /// (rpbc.py:196-197). Same dispatch target as `rtype_simple_call`;
    /// the `call_args` vs `simple_call` difference is handled inside
    /// `callparse::callparse` via `hop.spaceop.opname`.
    fn rtype_call_args(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.rtype_simple_call(hop)
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
        // upstream rpbc.py:232-239 — single-row case → `row.fntype`
        // (wrapped in `Ptr`); multi-row goes through setup_specfunc.
        let lltype = if llct.uniquerows.len() == 1 {
            let row = llct.uniquerows[0].borrow();
            LowLevelType::Ptr(Box::new(
                crate::translator::rtyper::lltypesystem::lltype::Ptr {
                    TO: PtrTarget::Func(row.fntype.clone()),
                },
            ))
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

    /// RPython `FunctionsPBCRepr.setup_specfunc(self)` (rpbc.py:242-247).
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
    /// PRE-EXISTING-DEVIATION: `ConcreteCallTableRow::from_row` in the
    /// Rust port stores `fntype` as a bare `FuncType` (unwrapped from
    /// the `Ptr(FuncType)` that `typeOf(llfn)` returns upstream). The
    /// Struct field is therefore re-wrapped in `Ptr(FuncType)` to match
    /// the semantic shape upstream consumers expect.
    pub(crate) fn setup_specfunc(
        uniquerows: &[ConcreteCallTableRowRef],
    ) -> Result<LowLevelType, TyperError> {
        use crate::translator::rtyper::lltypesystem::lltype::{Ptr as LlPtr, StructType};

        let mut fields: Vec<(String, LowLevelType)> = Vec::with_capacity(uniquerows.len());
        for row in uniquerows {
            let row_ref = row.borrow();
            let attrname = row_ref.attrname.clone().ok_or_else(|| {
                TyperError::message("FunctionsPBCRepr.setup_specfunc: uniquerow missing attrname")
            })?;
            let field_type = LowLevelType::Ptr(Box::new(LlPtr {
                TO: PtrTarget::Func(row_ref.fntype.clone()),
            }));
            fields.push((attrname, field_type));
        }
        let hints = vec![
            (
                "immutable".to_string(),
                crate::flowspace::model::ConstValue::Bool(true),
            ),
            (
                "static_immutable".to_string(),
                crate::flowspace::model::ConstValue::Bool(true),
            ),
        ];
        let struct_t = StructType::with_hints("specfunc", fields, hints);
        Ok(LowLevelType::Ptr(Box::new(LlPtr {
            TO: PtrTarget::Struct(struct_t),
        })))
    }

    /// RPython `FunctionsPBCRepr.create_specfunc(self)` (rpbc.py:249-250).
    ///
    /// ```python
    /// def create_specfunc(self):
    ///     return malloc(self.lowleveltype.TO, immortal=True)
    /// ```
    pub fn create_specfunc(
        &self,
    ) -> Result<crate::translator::rtyper::lltypesystem::lltype::_ptr, TyperError> {
        use crate::translator::rtyper::lltypesystem::lltype::{MallocFlavor, malloc};

        let LowLevelType::Ptr(ptr) = &self.lltype else {
            return Err(TyperError::message(
                "FunctionsPBCRepr.create_specfunc: lowleveltype is not Ptr",
            ));
        };
        let to_type: LowLevelType = ptr.TO.clone().into();
        malloc(to_type, None, MallocFlavor::Gc, true).map_err(TyperError::message)
    }

    /// RPython `FunctionsPBCRepr.get_specfunc_row(self, llop, v, c_rowname, resulttype)`
    /// (rpbc.py:252-253):
    ///
    /// ```python
    /// def get_specfunc_row(self, llop, v, c_rowname, resulttype):
    ///     return llop.genop('getfield', [v, c_rowname], resulttype=resulttype)
    /// ```
    fn get_specfunc_row(
        llop: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
        v: crate::flowspace::model::Hlvalue,
        c_rowname: crate::flowspace::model::Hlvalue,
        resulttype: LowLevelType,
    ) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
        use crate::translator::rtyper::rtyper::GenopResult;
        let result = llop
            .genop(
                "getfield",
                vec![v, c_rowname],
                GenopResult::LLType(resulttype),
            )
            .expect("get_specfunc_row: getfield with non-Void resulttype returns a value");
        Ok(crate::flowspace::model::Hlvalue::Variable(result))
    }

    /// RPython `FunctionsPBCRepr.convert_to_concrete_llfn(self, v, shape, index, llop)`
    /// (rpbc.py:300-312):
    ///
    /// ```python
    /// def convert_to_concrete_llfn(self, v, shape, index, llop):
    ///     assert v.concretetype == self.lowleveltype
    ///     if len(self.uniquerows) == 1:
    ///         return v
    ///     else:
    ///         # 'v' is a Struct pointer, read the corresponding field
    ///         row = self.concretetable[shape, index]
    ///         cname = inputconst(Void, row.attrname)
    ///         return self.get_specfunc_row(llop, v, cname, row.fntype)
    /// ```
    pub fn convert_to_concrete_llfn(
        &self,
        v: &crate::flowspace::model::Hlvalue,
        shape: &CallShape,
        index: usize,
        llop: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
    ) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
        use crate::translator::rtyper::lltypesystem::lltype::Ptr as LlPtr;

        // upstream: `assert v.concretetype == self.lowleveltype`.
        let v_ty = match v {
            crate::flowspace::model::Hlvalue::Variable(var) => var.concretetype(),
            crate::flowspace::model::Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        if v_ty.as_ref() != Some(&self.lltype) {
            return Err(TyperError::message(
                "FunctionsPBCRepr.convert_to_concrete_llfn: v.concretetype != self.lowleveltype",
            ));
        }

        if self.uniquerows.len() == 1 {
            return Ok(v.clone());
        }

        // upstream: `row = self.concretetable[shape, index]`.
        let row_ref = self
            .concretetable
            .get(&(shape.clone(), index))
            .cloned()
            .ok_or_else(|| {
                TyperError::message(
                    "FunctionsPBCRepr.convert_to_concrete_llfn: \
                     (shape, index) not in concretetable",
                )
            })?;
        let row = row_ref.borrow();
        let attrname = row.attrname.clone().ok_or_else(|| {
            TyperError::message(
                "FunctionsPBCRepr.convert_to_concrete_llfn: multi-row row missing attrname",
            )
        })?;
        let cname = crate::translator::rtyper::rmodel::inputconst_from_lltype(
            &LowLevelType::Void,
            &ConstValue::Str(attrname),
        )?;
        // upstream rpbc.py's convert_to_concrete_llfn passes `row.fntype`
        // as resulttype; pyre's ConcreteCallTableRow stores bare FuncType
        // (PRE-EXISTING-DEVIATION — see setup_specfunc doc). Re-wrap as
        // Ptr(FuncType) to match the specfunc struct field type.
        let result_ty = LowLevelType::Ptr(Box::new(LlPtr {
            TO: PtrTarget::Func(row.fntype.clone()),
        }));
        Self::get_specfunc_row(
            llop,
            v.clone(),
            crate::flowspace::model::Hlvalue::Constant(cname),
            result_ty,
        )
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
    /// Walks every uniquerow, collecting either `row[funcdesc]` or a
    /// typed nullptr. Single-row returns the collected llfn directly;
    /// multi-row mallocs a specfunc struct and sets each variant field
    /// via `_setattr`. The "funcdesc missing from the only row"
    /// rffi.cast fallback (rpbc.py:275-280) still defers until the
    /// `funccache` size can feed an rffi.cast immediate.
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        use crate::translator::rtyper::lltypesystem::lltype::{
            _ptr_obj, LowLevelValue, Ptr as LlPtr, nullptr,
        };

        let desc_rowkey = desc
            .rowkey()
            .map_err(|e| TyperError::message(e.to_string()))?;

        // upstream: `try: return self.funccache[funcdesc]`.
        if let Some(cached) = self.funccache.borrow().get(&desc_rowkey) {
            return Ok(cached.clone());
        }

        // upstream rpbc.py:261-271 — collect `llfns[attrname] = llfn`
        // entries, remembering whether any row actually contained the
        // desc.
        let mut llfns: Vec<(String, _ptr)> = Vec::with_capacity(self.uniquerows.len());
        let mut found_anything = false;
        for row_ref in &self.uniquerows {
            let row = row_ref.borrow();
            // upstream `row.attrname` — `None` only when len(uniquerows) == 1,
            // in which case the attrname slot is never consulted (single-
            // row path doesn't dereference it).
            let attrname = row.attrname.clone().unwrap_or_default();
            let llfn = if let Some(ptr) = row.row.get(&desc_rowkey) {
                found_anything = true;
                ptr.clone()
            } else {
                // upstream rpbc.py:267-270 — missing entry → `nullptr(row.fntype.TO)`.
                let func_ptr_type = LowLevelType::Ptr(Box::new(LlPtr {
                    TO: PtrTarget::Func(row.fntype.clone()),
                }));
                nullptr(match func_ptr_type.clone() {
                    LowLevelType::Ptr(p) => p.TO.clone().into(),
                    _ => unreachable!(),
                })
                .map_err(TyperError::message)?
            };
            llfns.push((attrname, llfn));
        }

        // upstream rpbc.py:272-285 — single-row unwraps directly;
        // multi-row mallocs a specfunc struct and populates fields.
        let result = if self.uniquerows.len() == 1 {
            if found_anything {
                llfns.into_iter().next().unwrap().1
            } else {
                // upstream rpbc.py:275-280 —
                //     "extremely rare case, shown only sometimes by
                //      test_bug_callfamily: don't emit NULL, because
                //      that would be interpreted as equal to None...
                //      It should never be called anyway."
                //     result = rffi.cast(self.lowleveltype,
                //                        ~len(self.funccache))
                //
                // rffi.cast of an integer to Ptr(FuncType) materialises
                // a non-null, non-None pointer with no callable body.
                // Pyre mints a synthetic `_func` container whose name
                // carries the fallback ordinal so distinct fallback
                // emissions stay distinct under `_func` structural
                // equality.
                use crate::translator::rtyper::lltypesystem::lltype::functionptr;

                let row = self.uniquerows[0].borrow();
                let fallback_id = !(self.funccache.borrow().len() as i64);
                let name = format!("<rffi.cast_pbc_fallback_{fallback_id}>");
                // `functionptr` mints a fresh _ptr identity and wraps a
                // `_func` whose `_name` carries the fallback ordinal,
                // so _func structural equality distinguishes this
                // fallback from any real callable pointer that might
                // share the fntype.
                functionptr(row.fntype.clone(), &name, None, None)
            }
        } else {
            let mut specfunc = self.create_specfunc()?;
            let obj_slot = specfunc._obj0.as_mut().map_err(|_| {
                TyperError::message("FunctionsPBCRepr.convert_desc: specfunc pointer is delayed")
            })?;
            let obj = obj_slot.as_mut().ok_or_else(|| {
                TyperError::message("FunctionsPBCRepr.convert_desc: specfunc pointer is null")
            })?;
            let _ptr_obj::Struct(s) = obj else {
                return Err(TyperError::message(
                    "FunctionsPBCRepr.convert_desc: specfunc points at non-Struct object",
                ));
            };
            for (attrname, llfn) in llfns {
                let ok = s._setattr(&attrname, LowLevelValue::Ptr(Box::new(llfn)));
                if !ok {
                    return Err(TyperError::message(format!(
                        "FunctionsPBCRepr.convert_desc: specfunc struct has no field {attrname:?}"
                    )));
                }
            }
            specfunc
        };

        let c = crate::translator::rtyper::rmodel::inputconst_from_lltype(
            &self.lltype,
            &ConstValue::LLPtr(Box::new(result)),
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
    /// (rpbc.py:289-298):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if isinstance(value, types.MethodType) and value.im_self is None:
    ///         value = value.im_func  # unbound method -> bare function
    ///     elif isinstance(value, staticmethod):
    ///         value = value.__get__(42)
    ///     if value is None:
    ///         null = nullptr(self.lowleveltype.TO)
    ///         return null
    ///     funcdesc = self.rtyper.annotator.bookkeeper.getdesc(value)
    ///     return self.convert_desc(funcdesc)
    /// ```
    ///
    /// The `types.MethodType` unbound-method arm is dead in the Rust
    /// port — `HostObject::BoundMethod` always carries a concrete
    /// `self_obj` so there is no Py2-style unbound method analogue.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        // rpbc.py:292-293 — staticmethod unwrap via `__get__(42)`; the
        // descriptor protocol returns the underlying callable.
        let unwrapped: ConstValue = match value {
            ConstValue::HostObject(host) if host.is_staticmethod() => host
                .staticmethod_func()
                .cloned()
                .map(ConstValue::HostObject)
                .ok_or_else(|| {
                    TyperError::message(
                        "FunctionsPBCRepr.convert_const: staticmethod has no wrapped function",
                    )
                })?,
            other => other.clone(),
        };

        // rpbc.py:294-296 — `value is None` → `nullptr(self.lowleveltype.TO)`.
        if matches!(unwrapped, ConstValue::None) {
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

        // rpbc.py:297-298 — `funcdesc = bookkeeper.getdesc(value)` ->
        // `self.convert_desc(funcdesc)`.
        let ConstValue::HostObject(host) = &unwrapped else {
            return Err(TyperError::message(format!(
                "FunctionsPBCRepr.convert_const: expected callable HostObject, got {:?}",
                unwrapped
            )));
        };
        let rtyper = self.base.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("FunctionsPBCRepr.convert_const: rtyper weak reference dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("FunctionsPBCRepr.convert_const: annotator weak reference dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        self.convert_desc(&desc)
    }

    /// RPython `FunctionReprBase.rtype_simple_call(self, hop)`
    /// (rpbc.py:193-194) — inherited by `FunctionsPBCRepr`. Delegates
    /// to the shared `call_setup` / `call_emit` scaffolding with
    /// `FunctionsPBCRepr`'s getfield-based `convert_to_concrete_llfn`.
    fn rtype_simple_call(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        let (row, vfn) = self.base.call_setup(self, hop)?;
        let v_func = {
            let mut llops = hop.llops.borrow_mut();
            self.convert_to_concrete_llfn(&vfn, &row.shape, row.index, &mut llops)?
        };
        self.base.call_emit(v_func, &row, hop)
    }

    /// RPython `FunctionReprBase.rtype_call_args(self, hop)`
    /// (rpbc.py:196-197).
    fn rtype_call_args(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.rtype_simple_call(hop)
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
    /// matching upstream's dict-by-Python-identity. Values are the
    /// `fakeaddress` Constants returned by `convert_desc`.
    pub converted_pbc_cache: RefCell<HashMap<crate::annotator::description::DescKey, Constant>>,
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
        // upstream `llmemory.Address._defl()` returns `NULL =
        // fakeaddress(None)`. Pyre's [`_address::Null`] is the direct
        // analogue, wrapped in [`ConstValue::LLAddress`] so the
        // Constant carries an Address-typed value (rather than
        // re-using `ConstValue::None`, which would conflate with Void
        // None constants).
        Constant::with_concretetype(
            ConstValue::LLAddress(crate::translator::rtyper::lltypesystem::lltype::_address::Null),
            LowLevelType::Address,
        )
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.create_instance(self)`
    /// (rpbc.py:702-703):
    ///
    /// ```python
    /// def create_instance(self):
    ///     return malloc(self.EMPTY, immortal=True)
    /// ```
    ///
    /// Builds a fresh `_struct` instance for the upstream
    /// `EMPTY = Struct('pbc', hints={'immutable': True,
    /// 'static_immutable': True})` placeholder, used when the
    /// underlying single-PBC repr is Void (i.e. the frozen desc has
    /// no fields).
    fn create_instance(
        &self,
    ) -> Result<crate::translator::rtyper::lltypesystem::lltype::_ptr, TyperError> {
        use crate::translator::rtyper::lltypesystem::lltype;
        let hints = lltype::FrozenDict::new(vec![
            ("immutable".to_string(), ConstValue::Bool(true)),
            ("static_immutable".to_string(), ConstValue::Bool(true)),
        ]);
        let empty = lltype::StructType {
            _name: "pbc".to_string(),
            _flds: lltype::FrozenDict::new(Vec::new()),
            _names: Vec::new(),
            _adtmeths: lltype::FrozenDict::new(Vec::new()),
            _hints: hints,
            _arrayfld: None,
            _gckind: lltype::GcKind::Raw,
            _runtime_type_info: None,
        };
        lltype::malloc(
            LowLevelType::Struct(Box::new(empty)),
            None,
            lltype::MallocFlavor::Raw,
            true,
        )
        .map_err(TyperError::message)
    }

    /// RPython `MultipleUnrelatedFrozenPBCRepr.convert_pbc(self,
    /// pbcptr)` (rpbc.py:699-700):
    ///
    /// ```python
    /// def convert_pbc(self, pbcptr):
    ///     return llmemory.fakeaddress(pbcptr)
    /// ```
    ///
    /// Wraps the underlying frozen-PBC pointer in an Address-typed
    /// `fakeaddress` so consumers can compare via `adr_eq`.
    fn convert_pbc(
        &self,
        pbcptr: crate::translator::rtyper::lltypesystem::lltype::_ptr,
    ) -> Constant {
        Constant::with_concretetype(
            ConstValue::LLAddress(
                crate::translator::rtyper::lltypesystem::lltype::_address::Fake(Box::new(pbcptr)),
            ),
            LowLevelType::Address,
        )
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

    /// RPython `MultipleUnrelatedFrozenPBCRepr.convert_desc(self,
    /// frozendesc)` (rpbc.py:685-697):
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
    /// Routes a single frozendesc through the PBC repr machinery and
    /// wraps the resulting pointer (or empty-struct placeholder) in a
    /// `fakeaddress`-typed Constant. The `converted_pbc_cache` keys by
    /// `DescEntry::desc_key()` (counter identity, matching upstream's
    /// dict-by-Python-identity).
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        use crate::annotator::model::{SomePBC, SomeValue};
        use crate::translator::rtyper::lltypesystem::lltype::{_address, _ptr_obj};

        let DescEntry::Frozen(_) = desc else {
            return Err(TyperError::message(format!(
                "MultipleUnrelatedFrozenPBCRepr.convert_desc: expected FrozenDesc, got {:?}",
                desc.kind()
            )));
        };
        let identity_key = desc.desc_key();

        // upstream rpbc.py:686-687 — `try: return self.converted_pbc_cache[frozendesc]`.
        if let Some(cached) = self.converted_pbc_cache.borrow().get(&identity_key) {
            return Ok(cached.clone());
        }

        // upstream rpbc.py:689-694:
        //   r = self.rtyper.getrepr(SomePBC([frozendesc]))
        //   if r.lowleveltype is Void: pbc = self.create_instance()
        //   else: pbc = r.convert_desc(frozendesc)
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MultipleUnrelatedFrozenPBCRepr: rtyper dropped"))?;
        let s_single = SomeValue::PBC(SomePBC::new(vec![desc.clone()], false));
        let r = rtyper.getrepr(&s_single)?;
        let pbcptr = if r.lowleveltype() == &LowLevelType::Void {
            self.create_instance()?
        } else {
            let pbc_const = r.convert_desc(desc)?;
            let ConstValue::LLPtr(ptr) = pbc_const.value else {
                return Err(TyperError::message(format!(
                    "MultipleUnrelatedFrozenPBCRepr.convert_desc: underlying repr {:?} \
                     returned non-LLPtr Constant {pbc_const:?}",
                    r.class_name()
                )));
            };
            // Strip the `Box` to recover an owned `_ptr` for
            // `convert_pbc(pbc)`.
            *ptr
        };
        let convpbc = self.convert_pbc(pbcptr);
        // Touch _ptr_obj/_address paths so the `unused import` lint
        // doesn't complain even when both arms are infrequently used.
        let _ = std::marker::PhantomData::<(_ptr_obj, _address)>;
        self.converted_pbc_cache
            .borrow_mut()
            .insert(identity_key, convpbc.clone());
        Ok(convpbc)
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
            return Ok(self.null_instance());
        }
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MultipleUnrelatedFrozenPBCRepr: rtyper dropped"))?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("MultipleUnrelatedFrozenPBCRepr: annotator dropped")
        })?;
        let host = match value {
            ConstValue::HostObject(h) => h.clone(),
            other => {
                return Err(TyperError::message(format!(
                    "MultipleUnrelatedFrozenPBCRepr.convert_const: expected HostObject, \
                     got {other:?}"
                )));
            }
        };
        let desc = annotator
            .bookkeeper
            .getdesc(&host)
            .map_err(|err| TyperError::message(err.to_string()))?;
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
/// Repr for a `SomePBC` of frozen PBCs that share a common attribute
/// access set. Upstream (rpbc.py:731-736):
///
/// ```python
/// def __init__(self, rtyper, access_set):
///     self.rtyper = rtyper
///     self.access_set = access_set
///     self.pbc_type = ForwardReference()
///     self.lowleveltype = Ptr(self.pbc_type)
///     self.pbc_cache = {}
/// ```
///
/// Full body landed per rpbc.py:728-800 + rpbc.py:802-841 pair
/// conversions. `_setup_repr` walks the attr family, mangles names,
/// materialises the `Struct('pbc', ...)` and resolves `pbc_type` so
/// `create_instance` / `convert_desc` / `rtype_getattr` can proceed.
#[derive(Debug)]
pub struct MultipleFrozenPBCRepr {
    /// RPython `self.rtyper = rtyper` (rpbc.py:732). Weak backref.
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.access_set = access_set` (rpbc.py:733). Upstream
    /// accepts `None` when every desc has `queryattrfamily() is None`;
    /// in that case `_setup_repr_fields` returns an empty field list.
    pub access_set: Option<Rc<RefCell<crate::annotator::description::FrozenAttrFamily>>>,
    /// RPython `self.pbc_type = ForwardReference()` (rpbc.py:734).
    /// Resolved by `_setup_repr` into a `Struct('pbc', *fields,
    /// hints={immutable,static_immutable})`.
    pub pbc_type: crate::translator::rtyper::lltypesystem::lltype::ForwardReference,
    /// RPython `self.lowleveltype = Ptr(self.pbc_type)` (rpbc.py:735).
    lltype: LowLevelType,
    /// RPython `self.pbc_cache = {}` (rpbc.py:736). Upstream
    /// `convert_desc` caches per-frozendesc struct pointers.
    pub pbc_cache: RefCell<HashMap<DescKey, Constant>>,
    /// Lazy field map populated by `_setup_repr_fields`. Maps the
    /// raw attribute name to `(mangled_name, r_value)`. RPython stores
    /// this as `self.fieldmap` (rpbc.py:757).
    #[allow(clippy::type_complexity)]
    pub fieldmap: RefCell<Option<HashMap<String, (String, std::sync::Arc<dyn Repr>)>>>,
    state: ReprState,
}

/// Convert a primitive-carrying [`Constant`] into the
/// [`crate::translator::rtyper::lltypesystem::lltype::LowLevelValue`]
/// that a matching `_struct` field-slot expects. Used by
/// `MultipleFrozenPBCRepr.convert_desc` to feed upstream's
/// `setattr(result, mangled_name, llvalue)` (rpbc.py:785-789) through
/// the Rust `_struct::_setattr(name, LowLevelValue)` API.
///
/// PRE-EXISTING-ADAPTATION: upstream Python uses ctypes for the
/// coercion at write time; the Rust port only needs to round-trip the
/// primitives our reprs actually emit — Int / Bool / Float / Char /
/// LLPtr. Struct / Array bodies would need to be materialised via
/// their own container example, which isn't exercised by the set of
/// attribute types `access_set.attrs` carries today.
fn constant_to_llvalue(
    c: &Constant,
) -> Result<crate::translator::rtyper::lltypesystem::lltype::LowLevelValue, String> {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelValue;

    let expected = c.concretetype.clone();
    match (&c.value, expected.as_ref()) {
        (ConstValue::None, Some(LowLevelType::Void)) | (ConstValue::None, None) => {
            Ok(LowLevelValue::Void)
        }
        (ConstValue::Int(n), Some(lltype)) => match lltype {
            LowLevelType::Signed
            | LowLevelType::SignedLongLong
            | LowLevelType::SignedLongLongLong => Ok(LowLevelValue::Signed(*n)),
            LowLevelType::Unsigned
            | LowLevelType::UnsignedLongLong
            | LowLevelType::UnsignedLongLongLong => Ok(LowLevelValue::Unsigned(*n as u64)),
            LowLevelType::Bool => Ok(LowLevelValue::Bool(*n != 0)),
            other => Err(format!(
                "constant_to_llvalue: Int constant with concretetype {other:?} not supported"
            )),
        },
        (ConstValue::Bool(b), _) => Ok(LowLevelValue::Bool(*b)),
        (ConstValue::Float(bits), _) => Ok(LowLevelValue::Float(*bits)),
        (ConstValue::LLPtr(ptr), _) => Ok(LowLevelValue::Ptr(ptr.clone())),
        (ConstValue::Str(s), Some(LowLevelType::Void)) => {
            // Void-typed string constants (`c_rowname` style placeholders)
            // aren't stored in struct fields; upstream rejects them too.
            Err(format!(
                "constant_to_llvalue: Void-typed Str {s:?} is not a materialisable value"
            ))
        }
        (other, ctype) => Err(format!(
            "constant_to_llvalue: unsupported constant {other:?} (concretetype {ctype:?})"
        )),
    }
}

impl MultipleFrozenPBCRepr {
    /// RPython `MultipleFrozenPBCRepr(rtyper, access_set)` (rpbc.py:731-736).
    pub fn new(
        rtyper: &Rc<RPythonTyper>,
        access_set: Option<Rc<RefCell<crate::annotator::description::FrozenAttrFamily>>>,
    ) -> Self {
        use crate::translator::rtyper::lltypesystem::lltype::{ForwardReference, Ptr as LlPtr};

        let pbc_type = ForwardReference::new();
        let lltype = LowLevelType::Ptr(Box::new(LlPtr {
            TO: PtrTarget::ForwardReference(pbc_type.clone()),
        }));
        MultipleFrozenPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            access_set,
            pbc_type,
            lltype,
            pbc_cache: RefCell::new(HashMap::new()),
            fieldmap: RefCell::new(None),
            state: ReprState::new(),
        }
    }

    /// RPython `MultipleFrozenPBCRepr._setup_repr_fields(self)`
    /// (rpbc.py:755-767).
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
        use crate::translator::rtyper::rmodel::mangle;

        let mut fields: Vec<(String, LowLevelType)> = Vec::new();
        let mut fieldmap: HashMap<String, (String, std::sync::Arc<dyn Repr>)> = HashMap::new();

        if let Some(access_set) = &self.access_set {
            let rtyper = self.rtyper.upgrade().ok_or_else(|| {
                TyperError::message("MultipleFrozenPBCRepr._setup_repr_fields: rtyper weak dropped")
            })?;
            // upstream: `attrlist = self.access_set.attrs.keys(); attrlist.sort()`.
            let mut attrlist: Vec<String> = access_set.borrow().attrs.keys().cloned().collect();
            attrlist.sort();
            for attr in &attrlist {
                let s_value = access_set
                    .borrow()
                    .attrs
                    .get(attr)
                    .cloned()
                    .expect("attrlist came from access_set.attrs");
                let r_value = rtyper.getrepr(&s_value)?;
                let mangled_name = mangle("pbc", attr);
                fields.push((mangled_name.clone(), r_value.lowleveltype().clone()));
                fieldmap.insert(attr.clone(), (mangled_name, r_value));
            }
        }

        *self.fieldmap.borrow_mut() = Some(fieldmap);
        Ok(fields)
    }

    /// RPython `MultipleFrozenPBCRepr._setup_repr(self)` (rpbc.py:738-741).
    ///
    /// ```python
    /// def _setup_repr(self):
    ///     llfields = self._setup_repr_fields()
    ///     kwds = {'hints': {'immutable': True, 'static_immutable': True}}
    ///     self.pbc_type.become(Struct('pbc', *llfields, **kwds))
    /// ```
    fn setup_repr(&self) -> Result<(), TyperError> {
        use crate::translator::rtyper::lltypesystem::lltype::StructType;

        let llfields = self.setup_repr_fields()?;
        let hints = vec![
            ("immutable".to_string(), ConstValue::Bool(true)),
            ("static_immutable".to_string(), ConstValue::Bool(true)),
        ];
        let struct_t = StructType::with_hints("pbc", llfields, hints);
        self.pbc_type
            .r#become(LowLevelType::Struct(Box::new(struct_t)))
            .map_err(TyperError::message)
    }

    /// RPython `MultipleFrozenPBCRepr.null_instance(self)` (rpbc.py:746-747):
    /// `return nullptr(self.pbc_type)`. Assumes `_setup_repr` has run.
    pub fn null_instance(
        &self,
    ) -> Result<crate::translator::rtyper::lltypesystem::lltype::_ptr, TyperError> {
        use crate::translator::rtyper::lltypesystem::lltype::nullptr;

        let resolved = self.pbc_type.resolved().ok_or_else(|| {
            TyperError::message(
                "MultipleFrozenPBCRepr.null_instance: pbc_type not resolved (call setup first)",
            )
        })?;
        nullptr(resolved).map_err(TyperError::message)
    }

    /// RPython `MultipleFrozenPBCRepr.create_instance(self)` (rpbc.py:743-744):
    /// `return malloc(self.pbc_type, immortal=True)`.
    pub fn create_instance(
        &self,
    ) -> Result<crate::translator::rtyper::lltypesystem::lltype::_ptr, TyperError> {
        use crate::translator::rtyper::lltypesystem::lltype::{MallocFlavor, malloc};

        let resolved = self.pbc_type.resolved().ok_or_else(|| {
            TyperError::message(
                "MultipleFrozenPBCRepr.create_instance: pbc_type not resolved (call setup first)",
            )
        })?;
        malloc(resolved, None, MallocFlavor::Gc, true).map_err(TyperError::message)
    }

    /// RPython `MultipleFrozenPBCRepr.getfield(vpbc, attr, llops)`
    /// (rpbc.py:749-753). Looks up `(mangled_name, r_value)` in the
    /// fieldmap, then emits `getfield(vpbc, mangled_name)` with the
    /// attribute's target repr.
    fn getfield_emit(
        &self,
        vpbc: &crate::flowspace::model::Hlvalue,
        attr: &str,
        llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
    ) -> Result<crate::flowspace::model::Hlvalue, TyperError> {
        use crate::translator::rtyper::rtyper::GenopResult;

        let fieldmap = self.fieldmap.borrow();
        let fieldmap = fieldmap.as_ref().ok_or_else(|| {
            TyperError::message(
                "MultipleFrozenPBCRepr.getfield: fieldmap not populated (call setup first)",
            )
        })?;
        let (mangled_name, r_value) = fieldmap.get(attr).cloned().ok_or_else(|| {
            TyperError::message(format!(
                "MultipleFrozenPBCRepr.getfield: no mangled field for attr {attr:?}"
            ))
        })?;
        let cname = crate::translator::rtyper::rmodel::inputconst_from_lltype(
            &LowLevelType::Void,
            &ConstValue::Str(mangled_name),
        )?;
        let result_ty = r_value.lowleveltype().clone();
        let result = llops
            .genop(
                "getfield",
                vec![
                    vpbc.clone(),
                    crate::flowspace::model::Hlvalue::Constant(cname),
                ],
                GenopResult::LLType(result_ty),
            )
            .expect("getfield with non-Void result returns a Variable");
        Ok(crate::flowspace::model::Hlvalue::Variable(result))
    }

    /// Runs `_setup_repr` if it hasn't yet. Idempotent because the
    /// Repr trait's `setup()` already dedupes through `ReprState`.
    fn ensure_setup(&self) -> Result<(), TyperError> {
        Repr::setup(self as &dyn Repr)
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

    /// RPython `_setup_repr` override (rpbc.py:738-741) — threaded
    /// through the `Repr::setup()` harness by `add_pendingsetup` in
    /// `get_frozen_pbc_repr`.
    fn _setup_repr(&self) -> Result<(), TyperError> {
        self.setup_repr()
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
            self.ensure_setup()?;
            let null = self.null_instance()?;
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(null)),
                self.lltype.clone(),
            ));
        }
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "MultipleFrozenPBCRepr.convert_const: expected HostObject, got {value:?}"
            )));
        };
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("MultipleFrozenPBCRepr.convert_const: rtyper weak dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("MultipleFrozenPBCRepr.convert_const: annotator weak dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        self.convert_desc(&desc)
    }

    /// RPython `MultipleFrozenPBCRepr.convert_desc(self, frozendesc)`
    /// (rpbc.py:769-790).
    ///
    /// ```python
    /// def convert_desc(self, frozendesc):
    ///     if (self.access_set is not None and
    ///             frozendesc not in self.access_set.descs):
    ///         raise TyperError("not found in PBC access set: %r")
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
    ///                     warning(...)
    ///                 continue
    ///             llvalue = r_value.convert_const(thisattrvalue)
    ///             setattr(result, mangled_name, llvalue)
    ///         return result
    /// ```
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;

        let DescEntry::Frozen(frozendesc) = desc else {
            return Err(TyperError::message(format!(
                "MultipleFrozenPBCRepr.convert_desc: expected FrozenDesc, got {:?}",
                desc.kind()
            )));
        };
        let identity_key = desc.desc_key();

        // upstream rpbc.py:770-772 — access-set membership guard.
        if let Some(access_set) = &self.access_set {
            if !access_set.borrow().descs.contains_key(&identity_key) {
                return Err(TyperError::message(format!(
                    "not found in PBC access set: {:?}",
                    identity_key
                )));
            }
        }

        // upstream rpbc.py:773-775 — `try: return self.pbc_cache[frozendesc]`.
        if let Some(cached) = self.pbc_cache.borrow().get(&identity_key) {
            return Ok(cached.clone());
        }

        // upstream rpbc.py:776-778 — `self.setup()` + `result =
        // self.create_instance()` + IMMEDIATELY
        // `self.pbc_cache[frozendesc] = result`. Pyre allocates TWO
        // distinct `_ptr` instances:
        //
        // * `placeholder_ptr` is what recursive
        //   `convert_desc(same_frozendesc)` sees from the cache. Its
        //   `_obj0` body is the empty `_struct` from
        //   `create_instance` and stays empty.
        // * `final_ptr` carries the populated struct body. After the
        //   loop finishes, `placeholder_ptr._become(&final_ptr)`
        //   records the redirect in `PTR_BECOME_TARGETS`, so any
        //   clone of `placeholder_ptr` (stored in another struct's
        //   field by a recursive convert_desc that ran during
        //   populate) resolves to the populated body when consumers
        //   call `_resolved_ptr` / `_obj()`.
        //
        // The cache entry itself is rewritten to `final_ptr` once
        // populate finishes, so direct cache reads bypass the
        // redirect chain.
        self.ensure_setup()?;
        let placeholder_ptr = self.create_instance()?;
        self.pbc_cache.borrow_mut().insert(
            identity_key,
            Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(placeholder_ptr.clone())),
                self.lltype.clone(),
            ),
        );

        let mut final_ptr = self.create_instance()?;

        // upstream rpbc.py:779-789 — populate struct fields from
        // frozendesc.attrcache + r_value.convert_const. Mutations
        // happen on `final_ptr` directly; the pbc_cache borrow stays
        // released across the recursive `r_value.convert_const(...)`.
        let field_entries: Vec<(String, String, std::sync::Arc<dyn Repr>)> = {
            let fieldmap = self.fieldmap.borrow();
            let fieldmap = fieldmap.as_ref().expect("ensure_setup populates fieldmap");
            fieldmap
                .iter()
                .map(|(attr, (mangled, r_value))| (attr.clone(), mangled.clone(), r_value.clone()))
                .collect()
        };

        for (attr, mangled_name, r_value) in field_entries {
            if r_value.lowleveltype() == &LowLevelType::Void {
                continue;
            }
            let thisattrvalue = frozendesc.borrow().attrcache.borrow().get(&attr).cloned();
            let Some(thisattrvalue) = thisattrvalue else {
                // upstream rpbc.py:783-787 — `warning(...)` when the
                // missing attribute isn't a dunder. Pyre tracks the
                // warning through warn_missing_attribute but otherwise
                // silently `continue`s to match upstream's flow.
                let _warn = frozendesc
                    .borrow()
                    .warn_missing_attribute(&attr)
                    .unwrap_or(false);
                continue;
            };
            // r_value.convert_const may recurse into
            // MFPBC.convert_desc — DON'T hold the pbc_cache borrow
            // across this call.
            let llvalue = r_value.convert_const(&thisattrvalue)?;
            let ll_body = constant_to_llvalue(&llvalue).map_err(TyperError::message)?;

            let obj_slot = final_ptr._obj0.as_mut().map_err(|_| {
                TyperError::message("MultipleFrozenPBCRepr.convert_desc: final pointer is delayed")
            })?;
            let obj = obj_slot.as_mut().ok_or_else(|| {
                TyperError::message("MultipleFrozenPBCRepr.convert_desc: final pointer is null")
            })?;
            let _ptr_obj::Struct(s) = obj else {
                return Err(TyperError::message(
                    "MultipleFrozenPBCRepr.convert_desc: pbc_type is not a Struct",
                ));
            };
            if !s._setattr(&mangled_name, ll_body) {
                return Err(TyperError::message(format!(
                    "MultipleFrozenPBCRepr.convert_desc: no field {mangled_name:?} \
                     on specfunc struct"
                )));
            }
        }

        // Splice the populated body into the redirect table so any
        // clone of `placeholder_ptr` resolves to `final_ptr` when
        // consumers walk through `_obj()` / `_resolved_ptr`.
        placeholder_ptr._become(&final_ptr);

        // Rewrite the cache entry to point at `final_ptr` directly —
        // direct cache reads now see the populated body without
        // chasing the redirect.
        let final_const = Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(final_ptr)),
            self.lltype.clone(),
        );
        self.pbc_cache
            .borrow_mut()
            .insert(identity_key, final_const.clone());
        Ok(final_const)
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

        // upstream rpbc.py:793-794 — constant fast-path.
        let s_const = hop
            .s_result
            .borrow()
            .as_ref()
            .and_then(|s| s.const_())
            .cloned();
        if let Some(value) = s_const {
            let r_result = hop
                .r_result
                .borrow()
                .clone()
                .ok_or_else(|| TyperError::message("HighLevelOp.r_result is not set"))?;
            let c = HighLevelOp::inputconst(&*r_result, &value)?;
            return Ok(Some(crate::flowspace::model::Hlvalue::Constant(c)));
        }

        // upstream rpbc.py:796-800 — variable attr + getfield via
        // mangled name, then convertvar into r_result.
        let attr = hop
            .args_s
            .borrow()
            .get(1)
            .and_then(|sv| sv.const_().cloned())
            .and_then(|cv| match cv {
                ConstValue::Str(s) => Some(s),
                _ => None,
            })
            .ok_or_else(|| {
                TyperError::message(
                    "MultipleFrozenPBCRepr.rtype_getattr: args_s[1] is not a string constant",
                )
            })?;

        self.ensure_setup()?;
        let vlist = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::LowLevelType(&LowLevelType::Void),
        ])?;
        let vpbc = vlist[0].clone();

        let v_res = {
            let mut llops = hop.llops.borrow_mut();
            self.getfield_emit(&vpbc, &attr, &mut llops)?
        };
        let r_res = {
            let fieldmap = self.fieldmap.borrow();
            let fieldmap = fieldmap.as_ref().expect("ensure_setup populates fieldmap");
            fieldmap
                .get(&attr)
                .map(|(_, r_value)| r_value.clone())
                .ok_or_else(|| {
                    TyperError::message(format!(
                        "MultipleFrozenPBCRepr.rtype_getattr: unknown attr {attr:?}"
                    ))
                })?
        };
        let r_result = hop
            .r_result
            .borrow()
            .clone()
            .ok_or_else(|| TyperError::message("HighLevelOp.r_result is not set"))?;
        let converted =
            hop.llops
                .borrow_mut()
                .convertvar(v_res, r_res.as_ref(), r_result.as_ref())?;
        Ok(Some(converted))
    }
}

/// RPython `pairtype(FunctionRepr, FunctionsPBCRepr).convert_from_to`
/// (rpbc.py:377-379):
///
/// ```python
/// class __extend__(pairtype(FunctionRepr, FunctionsPBCRepr)):
///     def convert_from_to((r_fpbc1, r_fpbc2), v, llops):
///         return inputconst(r_fpbc2, r_fpbc1.s_pbc.const)
/// ```
///
/// Upstream relies on `r_fpbc1` being a single-desc constant PBC so
/// `s_pbc.const` is populated with the pyobj stored in `const_box`.
/// The Rust port materialises the destination constant via
/// [`Repr::convert_const`] — which now routes `ConstValue::HostObject`
/// through `bookkeeper.getdesc`.
pub fn pair_function_repr_functions_pbc_repr_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
) -> Result<Option<crate::flowspace::model::Hlvalue>, TyperError> {
    let any_from: &dyn std::any::Any = r_from;
    let r_from = any_from.downcast_ref::<FunctionRepr>().ok_or_else(|| {
        TyperError::message(
            "pair(FunctionRepr, FunctionsPBCRepr): source repr is not a FunctionRepr",
        )
    })?;
    let pbc_const = r_from.base.s_pbc.base.const_box.as_ref().ok_or_else(|| {
        TyperError::message(
            "pair(FunctionRepr, FunctionsPBCRepr).convert_from_to: \
             source s_pbc has no constant pyobj",
        )
    })?;
    let c = r_to.convert_const(&pbc_const.value)?;
    Ok(Some(crate::flowspace::model::Hlvalue::Constant(c)))
}

/// RPython `pairtype(MultipleFrozenPBCRepr,
/// MultipleUnrelatedFrozenPBCRepr).convert_from_to` (rpbc.py:802-805):
///
/// ```python
/// return llops.genop('cast_ptr_to_adr', [v], resulttype=llmemory.Address)
/// ```
pub fn pair_mfpbc_mufpbc_convert_from_to(
    _r_from: &dyn Repr,
    _r_to: &dyn Repr,
    v: &crate::flowspace::model::Hlvalue,
    llops: &mut crate::translator::rtyper::rtyper::LowLevelOpList,
) -> Result<Option<crate::flowspace::model::Hlvalue>, TyperError> {
    use crate::translator::rtyper::rtyper::GenopResult;
    let result = llops
        .genop(
            "cast_ptr_to_adr",
            vec![v.clone()],
            GenopResult::LLType(LowLevelType::Address),
        )
        .expect("cast_ptr_to_adr with Address result returns a value");
    Ok(Some(crate::flowspace::model::Hlvalue::Variable(result)))
}

/// RPython `pairtype(MultipleFrozenPBCRepr,
/// MultipleFrozenPBCRepr).convert_from_to` (rpbc.py:807-811):
///
/// ```python
/// if r_pbc1.access_set == r_pbc2.access_set:
///     return v
/// return NotImplemented
/// ```
///
/// Identity on the access_set — both sides must point at the same
/// FrozenAttrFamily Rc. `NotImplemented` maps to `Ok(None)` so the
/// pairtype dispatcher falls through to the next MRO entry.
pub fn pair_mfpbc_mfpbc_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &crate::flowspace::model::Hlvalue,
) -> Result<Option<crate::flowspace::model::Hlvalue>, TyperError> {
    let any_from: &dyn std::any::Any = r_from;
    let any_to: &dyn std::any::Any = r_to;
    let Some(r_pbc1) = any_from.downcast_ref::<MultipleFrozenPBCRepr>() else {
        return Ok(None);
    };
    let Some(r_pbc2) = any_to.downcast_ref::<MultipleFrozenPBCRepr>() else {
        return Ok(None);
    };
    let same_access = match (&r_pbc1.access_set, &r_pbc2.access_set) {
        (Some(a), Some(b)) => Rc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    };
    if same_access {
        Ok(Some(v.clone()))
    } else {
        Ok(None)
    }
}

/// RPython `pairtype(SingleFrozenPBCRepr,
/// MultipleFrozenPBCRepr).convert_from_to` (rpbc.py:813-821):
///
/// ```python
/// frozendesc1 = r_pbc1.frozendesc
/// access = frozendesc1.queryattrfamily()
/// if access is r_pbc2.access_set:
///     value = r_pbc2.convert_desc(frozendesc1)
///     lltype = r_pbc2.lowleveltype
///     return Constant(value, lltype)
/// return NotImplemented
/// ```
pub fn pair_sfpbc_mfpbc_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
) -> Result<Option<crate::flowspace::model::Hlvalue>, TyperError> {
    let any_from: &dyn std::any::Any = r_from;
    let any_to: &dyn std::any::Any = r_to;
    let Some(r_pbc1) = any_from.downcast_ref::<SingleFrozenPBCRepr>() else {
        return Ok(None);
    };
    let Some(r_pbc2) = any_to.downcast_ref::<MultipleFrozenPBCRepr>() else {
        return Ok(None);
    };
    let DescEntry::Frozen(frozendesc1) = &r_pbc1.frozendesc else {
        return Err(TyperError::message(
            "pair(SingleFrozenPBCRepr, MultipleFrozenPBCRepr): source frozendesc is not a FrozenDesc",
        ));
    };
    let access = frozendesc1.borrow().queryattrfamily();
    let matches = match (&access, &r_pbc2.access_set) {
        (Some(a), Some(b)) => Rc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    };
    if !matches {
        return Ok(None);
    }
    let c = r_pbc2.convert_desc(&r_pbc1.frozendesc)?;
    Ok(Some(crate::flowspace::model::Hlvalue::Constant(c)))
}

/// RPython `pairtype(MultipleFrozenPBCReprBase,
/// SingleFrozenPBCRepr).convert_from_to` (rpbc.py:823-826):
///
/// ```python
/// return inputconst(Void, r_pbc2.frozendesc)
/// ```
///
/// SFPBC is `Void`-typed so we return a Void-typed
/// `ConstValue::FrozenDesc(DescKey)` carrying the frozen desc
/// identity. Consumers can distinguish a FrozenDesc-carrying Void
/// constant from a plain integer payload because the carrier is a
/// dedicated ConstValue variant.
pub fn pair_mfpbc_base_sfpbc_convert_from_to(
    _r_from: &dyn Repr,
    r_to: &dyn Repr,
) -> Result<Option<crate::flowspace::model::Hlvalue>, TyperError> {
    let any_to: &dyn std::any::Any = r_to;
    let Some(r_pbc2) = any_to.downcast_ref::<SingleFrozenPBCRepr>() else {
        return Ok(None);
    };
    let desc_key = r_pbc2.frozendesc.desc_key();
    let c = crate::translator::rtyper::rmodel::inputconst_from_lltype(
        &LowLevelType::Void,
        &ConstValue::FrozenDesc(desc_key.0),
    )?;
    Ok(Some(crate::flowspace::model::Hlvalue::Constant(c)))
}

/// RPython `pairtype(FunctionRepr,
/// MultipleFrozenPBCRepr).convert_from_to` (rpbc.py:828-834):
///
/// ```python
/// if r_fn1.s_pbc.is_constant():
///     value = r_frozen2.convert_const(r_fn1.s_pbc.const)
///     lltype = r_frozen2.lowleveltype
///     return Constant(value, lltype)
/// return NotImplemented
/// ```
pub fn pair_function_repr_mfpbc_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
) -> Result<Option<crate::flowspace::model::Hlvalue>, TyperError> {
    let any_from: &dyn std::any::Any = r_from;
    let Some(r_fn1) = any_from.downcast_ref::<FunctionRepr>() else {
        return Ok(None);
    };
    if !r_fn1.base.s_pbc.is_constant() {
        return Ok(None);
    }
    let Some(pbc_const) = r_fn1.base.s_pbc.base.const_box.as_ref() else {
        return Ok(None);
    };
    let c = r_to.convert_const(&pbc_const.value)?;
    Ok(Some(crate::flowspace::model::Hlvalue::Constant(c)))
}

/// RPython `pairtype(MultipleFrozenPBCRepr,
/// FunctionRepr).convert_from_to` (rpbc.py:836-841):
///
/// ```python
/// if r_fn2.lowleveltype is Void:
///     value = r_fn2.s_pbc.const
///     return Constant(value, Void)
/// return NotImplemented
/// ```
pub fn pair_mfpbc_function_repr_convert_from_to(
    _r_from: &dyn Repr,
    r_to: &dyn Repr,
) -> Result<Option<crate::flowspace::model::Hlvalue>, TyperError> {
    let any_to: &dyn std::any::Any = r_to;
    let Some(r_fn2) = any_to.downcast_ref::<FunctionRepr>() else {
        return Ok(None);
    };
    if r_fn2.lowleveltype() != &LowLevelType::Void {
        return Ok(None);
    }
    let Some(pbc_const) = r_fn2.base.s_pbc.base.const_box.as_ref() else {
        return Ok(None);
    };
    let c = crate::translator::rtyper::rmodel::inputconst_from_lltype(
        &LowLevelType::Void,
        &pbc_const.value,
    )?;
    Ok(Some(crate::flowspace::model::Hlvalue::Constant(c)))
}

/// RPython `class MethodOfFrozenPBCRepr(Repr)` (rpbc.py:844-915).
///
/// Representation for a `SomePBC` of `MethodOfFrozenDesc` — bound
/// methods on frozen PBCs where every desc shares the same underlying
/// `funcdesc`. The low-level representation is just the bound `im_self`
/// pointer (mirrors `r_im_self`).
///
/// ```python
/// class MethodOfFrozenPBCRepr(Repr):
///     def __init__(self, rtyper, s_pbc):
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
    /// RPython `self.rtyper = rtyper` (rpbc.py:850). Weak backref.
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.funcdesc` (rpbc.py:851-853).
    pub funcdesc: Rc<RefCell<crate::annotator::description::FunctionDesc>>,
    /// RPython `self.s_im_self = SomePBC(im_selves)` (rpbc.py:867).
    pub s_im_self: crate::annotator::model::SomeValue,
    /// RPython `self.r_im_self = rtyper.getrepr(self.s_im_self)`
    /// (rpbc.py:868). Strong handle so the underlying frozen-PBC repr
    /// stays alive while bound methods reference it.
    pub r_im_self: std::sync::Arc<dyn Repr>,
    state: ReprState,
    lltype: LowLevelType,
}

impl MethodOfFrozenPBCRepr {
    /// RPython `MethodOfFrozenPBCRepr.__init__(self, rtyper, s_pbc)`
    /// (rpbc.py:849-870).
    pub fn new(
        rtyper: &Rc<RPythonTyper>,
        s_pbc: crate::annotator::model::SomePBC,
    ) -> Result<Self, TyperError> {
        // upstream rpbc.py:851-853 — every description must share a
        // single funcdesc.
        let mut funcdesc_iter = s_pbc.descriptions.values().filter_map(|d| match d {
            DescEntry::MethodOfFrozen(rc) => Some(rc.borrow().funcdesc.clone()),
            _ => None,
        });
        let first = funcdesc_iter
            .next()
            .ok_or_else(|| TyperError::message("MethodOfFrozenPBCRepr: empty descriptions"))?;
        let first_id = first.borrow().base.identity;
        for fd in funcdesc_iter {
            if fd.borrow().base.identity != first_id {
                return Err(TyperError::message(
                    "MethodOfFrozenPBCRepr: descriptions must share a single funcdesc \
                     (rpbc.py:852)",
                ));
            }
        }

        // upstream rpbc.py:863-865 — variable `method-of-frozen-PBC or
        // None` is unsupported.
        if s_pbc.can_be_none {
            return Err(TyperError::message(
                "unsupported: variable of type method-of-frozen-PBC or None",
            ));
        }

        // upstream rpbc.py:867-868 — build SomePBC of the frozendescs
        // and ask the rtyper for its repr.
        let im_self_descs: Vec<DescEntry> = s_pbc
            .descriptions
            .values()
            .filter_map(|d| match d {
                DescEntry::MethodOfFrozen(rc) => {
                    Some(DescEntry::Frozen(rc.borrow().frozendesc.clone()))
                }
                _ => None,
            })
            .collect();
        let s_im_self = crate::annotator::model::SomeValue::PBC(
            crate::annotator::model::SomePBC::new(im_self_descs, false),
        );
        let r_im_self = rtyper.getrepr(&s_im_self)?;
        let lltype = r_im_self.lowleveltype().clone();

        Ok(MethodOfFrozenPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            funcdesc: first,
            s_im_self,
            r_im_self,
            state: ReprState::new(),
            lltype,
        })
    }

    /// RPython `MethodOfFrozenPBCRepr.get_s_callable(self)`
    /// (rpbc.py:872-873).
    pub fn get_s_callable(&self) -> crate::annotator::model::SomeValue {
        crate::annotator::model::SomeValue::PBC(crate::annotator::model::SomePBC::new(
            vec![DescEntry::Function(self.funcdesc.clone())],
            false,
        ))
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

    /// RPython `MethodOfFrozenPBCRepr.convert_desc(self, mdesc)`
    /// (rpbc.py:879-883):
    ///
    /// ```python
    /// def convert_desc(self, mdesc):
    ///     if mdesc.funcdesc is not self.funcdesc:
    ///         raise TyperError(...)
    ///     return self.r_im_self.convert_desc(mdesc.frozendesc)
    /// ```
    fn convert_desc(&self, desc: &DescEntry) -> Result<Constant, TyperError> {
        let DescEntry::MethodOfFrozen(mdesc) = desc else {
            return Err(TyperError::message(format!(
                "MethodOfFrozenPBCRepr.convert_desc: expected MethodOfFrozenDesc, \
                 got {:?}",
                desc.kind()
            )));
        };
        let m = mdesc.borrow();
        if m.funcdesc.borrow().base.identity != self.funcdesc.borrow().base.identity {
            return Err(TyperError::message(format!(
                "not a method bound on {:?}: {:?}",
                self.funcdesc.borrow().base.identity,
                m.funcdesc.borrow().base.identity
            )));
        }
        let frozen_entry = DescEntry::Frozen(m.frozendesc.clone());
        self.r_im_self.convert_desc(&frozen_entry)
    }

    /// RPython `MethodOfFrozenPBCRepr.convert_const(self, method)`
    /// (rpbc.py:885-887):
    ///
    /// ```python
    /// def convert_const(self, method):
    ///     mdesc = self.rtyper.annotator.bookkeeper.getdesc(method)
    ///     return self.convert_desc(mdesc)
    /// ```
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let host = match value {
            ConstValue::HostObject(h) => h.clone(),
            other => {
                return Err(TyperError::message(format!(
                    "MethodOfFrozenPBCRepr.convert_const: expected bound method \
                     HostObject, got {other:?}"
                )));
            }
        };
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("MethodOfFrozenPBCRepr: rtyper dropped"))?;
        let annotator = rtyper
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("MethodOfFrozenPBCRepr: annotator dropped"))?;
        let desc = annotator
            .bookkeeper
            .getdesc(&host)
            .map_err(|err| TyperError::message(err.to_string()))?;
        self.convert_desc(&desc)
    }

    /// RPython `MethodOfFrozenPBCRepr.rtype_simple_call /
    /// rtype_call_args` (rpbc.py:889-893):
    ///
    /// ```python
    /// def rtype_simple_call(self, hop):
    ///     return self.redispatch_call(hop, call_args=False)
    /// def rtype_call_args(self, hop):
    ///     return self.redispatch_call(hop, call_args=True)
    /// ```
    ///
    /// `redispatch_call` rewrites the bound method call into a plain
    /// function call by inserting the underlying funcdesc as the first
    /// argument and re-running the rtyper dispatch.
    ///
    /// PRE-EXISTING-DEVIATION: the `redispatch_call` implementation
    /// requires `hop.copy()` + `swap_fst_snd_args` + `r_s_popfirstarg`
    /// + `v_s_insertfirstarg` + `dispatch` + `adjust_shape` to be
    /// stitched together. All five hop helpers exist
    /// (rtyper.rs:1809+, 2072+, 2077+, 2087+, 1826+) but the
    /// glue/`adjust_shape` translator is the next item to land.
    fn rtype_simple_call(
        &self,
        _hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        Err(TyperError::missing_rtype_operation(
            "MethodOfFrozenPBCRepr.rtype_simple_call (rpbc.py:889-890) \
             port pending — blocked on redispatch_call + adjust_shape glue",
        ))
    }

    fn rtype_call_args(
        &self,
        _hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        Err(TyperError::missing_rtype_operation(
            "MethodOfFrozenPBCRepr.rtype_call_args (rpbc.py:892-893) \
             port pending — blocked on redispatch_call + adjust_shape glue",
        ))
    }
}

/// RPython `class ClassesPBCRepr(Repr)` (rpbc.py:920-968).
///
/// Constant-class branch of the PBC repr for a `SomePBC` whose kind
/// is `DescKind::Class`. Upstream:
///
/// ```python
/// def __init__(self, rtyper, s_pbc):
///     self.rtyper = rtyper
///     self.s_pbc = s_pbc
///     if s_pbc.is_constant():
///         self.lowleveltype = Void
///     else:
///         self.lowleveltype = self.getlowleveltype()
/// ```
///
/// Pyre only ports the `is_constant()` path today — `getlowleveltype`
/// (upstream rpbc.py:~970 onwards) depends on `rtyper.rootclass_repr`
/// resolution plus class-hierarchy common-base which is blocked on
/// `ClassRepr.init_vtable`. The non-constant arm surfaces
/// `MissingRTypeOperation` when reached.
#[derive(Debug)]
pub struct ClassesPBCRepr {
    /// RPython `self.rtyper = rtyper` (rpbc.py:924). Weak backref.
    pub rtyper: Weak<RPythonTyper>,
    /// RPython `self.s_pbc = s_pbc` (rpbc.py:925).
    pub s_pbc: SomePBC,
    state: ReprState,
    /// RPython `self.lowleveltype` set to `Void` on the constant
    /// branch; populated via `getlowleveltype()` otherwise.
    lltype: LowLevelType,
}

impl ClassesPBCRepr {
    /// RPython `ClassesPBCRepr.__init__(self, rtyper, s_pbc)`
    /// (rpbc.py:923-932), constant-class branch only.
    pub fn new(rtyper: &Rc<RPythonTyper>, s_pbc: SomePBC) -> Result<Self, TyperError> {
        // upstream rpbc.py:928 — `if s_pbc.is_constant():`. The pyre
        // `SomePBC::is_constant` mirrors model.py:532-537 which only
        // sets `const_box` when `len==1 && !can_be_none &&
        // desc.pyobj is not None`, so a single-desc PBC whose
        // ClassDesc has no live pyobj is *not* constant.
        if !s_pbc.is_constant() {
            return Err(TyperError::missing_rtype_operation(
                "ClassesPBCRepr: non-constant branch (rpbc.py:932 getlowleveltype) \
                 port pending — blocked on ClassRepr.init_vtable",
            ));
        }
        Ok(ClassesPBCRepr {
            rtyper: Rc::downgrade(rtyper),
            s_pbc,
            state: ReprState::new(),
            lltype: LowLevelType::Void,
        })
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
    /// Pyre only ports the `lowleveltype is Void` arm — the
    /// `getruntime` path depends on `ClassRepr.init_vtable` /
    /// `ClassRepr.vtable` which is not yet ported.
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
        Err(TyperError::missing_rtype_operation(
            "ClassesPBCRepr.convert_desc: non-Void lowleveltype routes through \
             ClassRepr.getruntime (rpbc.py:955-957) — blocked on ClassRepr.init_vtable",
        ))
    }

    /// RPython `ClassesPBCRepr.convert_const(self, cls)` (rpbc.py:
    /// :959-968).
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
    /// Pyre only handles the `cls is None` + Void arm. The
    /// `bk.getdesc(cls)` path needs a Python-host-to-classdesc lookup
    /// that lives on the `Bookkeeper`; wire it when setup_vtable
    /// reaches a non-None class constant.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if matches!(value, ConstValue::None) && matches!(self.lltype, LowLevelType::Void) {
            return Ok(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            ));
        }
        Err(TyperError::missing_rtype_operation(
            "ClassesPBCRepr.convert_const: non-None / non-Void branch (rpbc.py:960-968) \
             port pending — blocked on Bookkeeper.getdesc + ClassRepr.getruntime",
        ))
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
    // upstream rpbc.py:626-632 — `return rtyper.pbc_reprs[access]`
    // fast-path, else mint a new MultipleFrozenPBCRepr and
    // `rtyper.add_pendingsetup(result)` before caching. Upstream's
    // `access` may be `None` when every frozen desc has
    // `queryattrfamily() is None`; the access_set-less repr still
    // mints with an empty Struct.
    let access: Option<Rc<RefCell<crate::annotator::description::FrozenAttrFamily>>> =
        s_pbc.descriptions.values().find_map(|d| match d {
            DescEntry::Frozen(fd) => fd.borrow().queryattrfamily(),
            _ => None,
        });
    use crate::translator::rtyper::rtyper::PbcReprKey;
    let key = match &access {
        Some(family) => PbcReprKey::Access(Rc::as_ptr(family) as usize),
        // Upstream uses `None` as a dict key directly; the Rust port
        // collapses every `access is None` case onto `Unrelated` *for
        // Unrelated reprs*, but `MultipleFrozenPBCRepr(access=None)`
        // needs its own distinct key. Reuse `Access(0)` as the sentinel
        // — no real `FrozenAttrFamily` can live at pointer 0.
        None => PbcReprKey::Access(0),
    };
    if let Some(cached) = rtyper.pbc_reprs.borrow().get(&key) {
        return Ok(cached.clone());
    }
    let fresh: std::sync::Arc<dyn Repr> =
        std::sync::Arc::new(MultipleFrozenPBCRepr::new(rtyper, access));
    rtyper.pbc_reprs.borrow_mut().insert(key, fresh.clone());
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
        DescKind::MethodOfFrozen => Ok(std::sync::Arc::new(MethodOfFrozenPBCRepr::new(
            rtyper,
            s_pbc.clone(),
        )?) as std::sync::Arc<dyn Repr>),
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
    fn somepbc_rtyper_makerepr_single_function_desc_with_can_be_none_routes_to_multiple_frozen_pbc_repr()
     {
        // Uncallable (no callfamily wired) → getFrozenPBCRepr branch,
        // now lands on MultipleFrozenPBCRepr shell since the shell is
        // ported.
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let s_pbc = SomePBC::new(vec![f], true);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "MultipleFrozenPBCRepr");
    }

    #[test]
    fn somepbc_rtyper_makerepr_multi_function_descs_without_callfamily_routes_to_multiple_frozen_pbc_repr()
     {
        // Multi-FunctionDesc PBC without an attached callfamily →
        // upstream rpbc.py:49 routes to getFrozenPBCRepr which now
        // returns a MultipleFrozenPBCRepr shell.
        let (ann, rtyper) = make_rtyper();
        let f = function_entry(&ann.bookkeeper, "f");
        let g = function_entry(&ann.bookkeeper, "g");
        let s_pbc = SomePBC::new(vec![f, g], false);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "MultipleFrozenPBCRepr");
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
    ) -> (
        DescEntry,
        Rc<StdRefCell<crate::annotator::description::FrozenDesc>>,
    ) {
        use crate::annotator::description::{FrozenDesc, MethodOfFrozenDesc};
        use crate::flowspace::model::HostObject;
        let pyobj = HostObject::new_module(frozen_name);
        let frozen = Rc::new(StdRefCell::new(
            FrozenDesc::new(bk.clone(), pyobj).expect("FrozenDesc::new"),
        ));
        let mof = Rc::new(StdRefCell::new(MethodOfFrozenDesc::new(
            bk.clone(),
            funcdesc.clone(),
            frozen.clone(),
        )));
        (DescEntry::MethodOfFrozen(mof), frozen)
    }

    #[test]
    fn method_of_frozen_pbc_repr_routes_convert_desc_through_r_im_self() {
        // upstream rpbc.py:879-883 — convert_desc(mdesc) verifies the
        // funcdesc identity then defers to `r_im_self.convert_desc(
        // mdesc.frozendesc)`. With a single frozendesc backing it,
        // r_im_self ends up as `SingleFrozenPBCRepr` (lowleveltype Void).
        let (ann, rtyper) = make_rtyper();
        let DescEntry::Function(funcdesc) = function_entry(&ann.bookkeeper, "f") else {
            unreachable!();
        };
        let (mof_desc, _frozen) = method_of_frozen_entry(&ann.bookkeeper, &funcdesc, "frozen_a");
        let s_pbc = SomePBC::new(vec![mof_desc.clone()], false);
        let repr = MethodOfFrozenPBCRepr::new(&rtyper, s_pbc).expect("ctor");

        assert_eq!(repr.repr_class_id(), ReprClassId::MethodOfFrozenPBCRepr);
        // r_im_self is SingleFrozenPBCRepr → Void lowleveltype.
        assert_eq!(repr.lowleveltype(), &LowLevelType::Void);

        let c = repr.convert_desc(&mof_desc).expect("convert_desc");
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
        assert!(matches!(c.value, ConstValue::None));
    }

    #[test]
    fn method_of_frozen_pbc_repr_rejects_can_be_none() {
        // upstream rpbc.py:863-865 — variable of type
        // method-of-frozen-PBC or None is unsupported.
        let (ann, rtyper) = make_rtyper();
        let DescEntry::Function(funcdesc) = function_entry(&ann.bookkeeper, "f") else {
            unreachable!();
        };
        let (mof_desc, _) = method_of_frozen_entry(&ann.bookkeeper, &funcdesc, "frozen_a");
        let s_pbc = SomePBC::new(vec![mof_desc], true);
        let err = MethodOfFrozenPBCRepr::new(&rtyper, s_pbc).unwrap_err();
        assert!(err.to_string().contains("method-of-frozen-PBC or None"));
    }

    #[test]
    #[should_panic(expected = "different underlying functions")]
    fn method_of_frozen_pbc_repr_rejects_distinct_funcdescs() {
        // upstream rpbc.py:851-853 — every description must share a
        // single funcdesc. SomePBC::new (model.rs:1303) catches this
        // before the repr ctor sees it, raising the
        // "different underlying functions" panic from
        // [`SomePBC::__init__`] (model.py:527-553).
        let (ann, _rtyper) = make_rtyper();
        let DescEntry::Function(fd_a) = function_entry(&ann.bookkeeper, "fa") else {
            unreachable!();
        };
        let DescEntry::Function(fd_b) = function_entry(&ann.bookkeeper, "fb") else {
            unreachable!();
        };
        let (mof_a, _) = method_of_frozen_entry(&ann.bookkeeper, &fd_a, "frozen_a");
        let (mof_b, _) = method_of_frozen_entry(&ann.bookkeeper, &fd_b, "frozen_b");
        let _ = SomePBC::new(vec![mof_a, mof_b], false);
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
        // upstream `llmemory.Address._defl() == NULL = fakeaddress(None)`.
        // Pyre wraps the NULL sentinel in [`ConstValue::LLAddress`].
        let ConstValue::LLAddress(addr) = &null.value else {
            panic!("expected LLAddress, got {:?}", null.value);
        };
        assert!(matches!(
            addr,
            crate::translator::rtyper::lltypesystem::lltype::_address::Null
        ));
    }

    #[test]
    fn multiple_unrelated_frozen_pbc_repr_convert_const_none_returns_null_instance() {
        let (_ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let c = r.convert_const(&ConstValue::None).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Address));
        let ConstValue::LLAddress(addr) = &c.value else {
            panic!("expected LLAddress, got {:?}", c.value);
        };
        assert!(matches!(
            addr,
            crate::translator::rtyper::lltypesystem::lltype::_address::Null
        ));
    }

    #[test]
    fn multiple_unrelated_frozen_pbc_repr_convert_const_non_none_rejects_int() {
        // Non-HostObject ConstValues without a corresponding desc must
        // surface a clear error rather than silently fail.
        let (_ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let err = r.convert_const(&ConstValue::Int(7)).unwrap_err();
        assert!(
            err.to_string().contains("expected HostObject"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn multiple_unrelated_frozen_pbc_repr_convert_desc_routes_through_single_repr() {
        // upstream rpbc.py:685-697: `r = rtyper.getrepr(SomePBC([fd]))`
        // returns SingleFrozenPBCRepr for a single-desc PBC. Its
        // lowleveltype is Void, so create_instance is used to mint an
        // empty placeholder struct. The result is wrapped in a
        // fakeaddress Constant.
        let (ann, rtyper) = make_rtyper();
        let r = MultipleUnrelatedFrozenPBCRepr::new(&rtyper);
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let c = r.convert_desc(&f).expect("convert_desc(frozen0)");
        assert_eq!(c.concretetype, Some(LowLevelType::Address));
        let ConstValue::LLAddress(addr) = &c.value else {
            panic!("expected LLAddress, got {:?}", c.value);
        };
        assert!(matches!(
            addr,
            crate::translator::rtyper::lltypesystem::lltype::_address::Fake(_)
        ));
        // Second call hits converted_pbc_cache and returns the same Constant.
        let c2 = r.convert_desc(&f).unwrap();
        assert_eq!(c, c2);
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
    fn get_frozen_pbc_repr_multi_desc_common_access_set_returns_multiple_frozen_pbc_repr() {
        // rpbc.py:616-632 — two FrozenDescs with matching
        // `queryattrfamily()` (both None is still "common") route to
        // MultipleFrozenPBCRepr and register in `rtyper.pbc_reprs`
        // with `rtyper.add_pendingsetup` queueing a later
        // `_setup_repr`.
        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let g = frozen_entry(&ann.bookkeeper, "frozen1");
        let s_pbc = SomePBC::new(vec![f, g], false);
        let r = get_frozen_pbc_repr(&rtyper, &s_pbc).unwrap();
        assert_eq!(r.class_name(), "MultipleFrozenPBCRepr");
        // Second invocation must hit the pbc_reprs cache and return the same repr.
        let r2 = get_frozen_pbc_repr(&rtyper, &s_pbc).unwrap();
        assert!(
            std::sync::Arc::ptr_eq(&r, &r2),
            "pbc_reprs cache must reuse the same MultipleFrozenPBCRepr"
        );
    }

    /// Helper: build a FrozenAttrFamily with a single Integer-typed
    /// attribute `attrname` and union it with `desc_a` so their
    /// `queryattrfamily()` both return the same Rc.
    fn prime_attrfamily_with_int_attr(
        bk: &Rc<Bookkeeper>,
        desc_a: &DescEntry,
        attrname: &str,
    ) -> Rc<RefCell<crate::annotator::description::FrozenAttrFamily>> {
        use crate::annotator::model::{SomeInteger, SomeValue};
        // Touch getattrfamily to ensure the family exists under desc_a's identity.
        let DescEntry::Frozen(fd_a) = desc_a else {
            panic!("expected Frozen desc");
        };
        let family = fd_a
            .borrow()
            .getattrfamily()
            .expect("getattrfamily should mint a fresh family");
        family.borrow_mut().attrs.insert(
            attrname.to_string(),
            SomeValue::Integer(SomeInteger::new(false, false)),
        );
        // Make sure the bookkeeper's union-find slot is aware of the family.
        let _ = bk
            .frozenpbc_attr_families
            .borrow_mut()
            .find_rep(fd_a.borrow().base.identity);
        family
    }

    #[test]
    fn multiple_frozen_pbc_repr_setup_repr_builds_struct_with_mangled_attr_field() {
        // rpbc.py:738-767 — `_setup_repr` walks access_set.attrs, mangles
        // each attr name via mangle("pbc", attr), and populates the
        // ForwardReference pbc_type with a Struct('pbc', ...) carrying
        // the immutable/static_immutable hints.
        use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, PtrTarget};
        let (ann, rtyper) = make_rtyper();
        let desc_a = frozen_entry(&ann.bookkeeper, "frozen_a");
        let access = prime_attrfamily_with_int_attr(&ann.bookkeeper, &desc_a, "count");
        let r = std::sync::Arc::new(MultipleFrozenPBCRepr::new(&rtyper, Some(access)));
        Repr::setup(r.as_ref() as &dyn Repr).unwrap();

        // pbc_type should now be resolved to Struct('pbc', ...).
        let resolved = r.pbc_type.resolved().expect("pbc_type must resolve");
        let LowLevelType::Struct(s) = resolved else {
            panic!("pbc_type should resolve to a Struct, got non-Struct");
        };
        assert_eq!(s._name, "pbc");
        // Field name is mangle("pbc", "count") = "pbc_count".
        let expected_field = crate::translator::rtyper::rmodel::mangle("pbc", "count");
        assert!(
            s._flds.get(&expected_field).is_some(),
            "mangled field {expected_field:?} not found in struct"
        );
        // fieldmap should map the raw attr back to the mangled name.
        let fieldmap = r.fieldmap.borrow();
        let fieldmap = fieldmap.as_ref().expect("fieldmap populated");
        let (mangled, _r_value) = fieldmap.get("count").expect("fieldmap has count");
        assert_eq!(mangled, &expected_field);

        // lowleveltype is still Ptr(pbc_type) — which now resolves to
        // Ptr(Struct('pbc', ...)).
        let LowLevelType::Ptr(ptr) = r.lowleveltype() else {
            panic!("expected Ptr");
        };
        // Through ForwardReference resolution, PtrTarget::ForwardReference
        // round-trips to the resolved Struct.
        match &ptr.TO {
            PtrTarget::ForwardReference(fr) => {
                assert!(fr.resolved().is_some(), "ForwardReference must be resolved");
            }
            other => panic!("expected Ptr(ForwardReference), got {other:?}"),
        }
    }

    #[test]
    fn multiple_frozen_pbc_repr_convert_desc_populates_struct_fields_from_attrcache() {
        // rpbc.py:769-790 — convert_desc pulls attrs from
        // `frozendesc.attrcache` (not read_attribute — we pre-populate
        // the cache directly) and writes them into the specfunc struct
        // via `setattr(result, mangled_name, llvalue)`.
        use crate::translator::rtyper::lltypesystem::lltype::{_ptr_obj, LowLevelValue};
        let (ann, rtyper) = make_rtyper();
        let desc_a = frozen_entry(&ann.bookkeeper, "frozen_a");
        let access = prime_attrfamily_with_int_attr(&ann.bookkeeper, &desc_a, "count");

        // Pre-populate the frozendesc's attrcache so convert_desc has
        // something to copy into the struct.
        let DescEntry::Frozen(fd_a) = &desc_a else {
            unreachable!();
        };
        fd_a.borrow()
            .attrcache
            .borrow_mut()
            .insert("count".into(), ConstValue::Int(42));

        let r = std::sync::Arc::new(MultipleFrozenPBCRepr::new(&rtyper, Some(access)));
        let c = r.convert_desc(&desc_a).unwrap();
        let ConstValue::LLPtr(ptr) = &c.value else {
            panic!("expected LLPtr");
        };
        let Ok(Some(_ptr_obj::Struct(s))) = &ptr._obj0 else {
            panic!("expected freshly-malloc'd Struct");
        };
        let expected_field = crate::translator::rtyper::rmodel::mangle("pbc", "count");
        let field = s
            ._getattr(&expected_field)
            .expect("field must be populated");
        assert_eq!(field, &LowLevelValue::Signed(42));

        // Second convert_desc hits the pbc_cache.
        let c2 = r.convert_desc(&desc_a).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn multiple_frozen_pbc_repr_convert_const_none_returns_null_instance() {
        // rpbc.py:666-668 — `if pbc is None: return self.null_instance()`.
        use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
        let (ann, rtyper) = make_rtyper();
        let desc_a = frozen_entry(&ann.bookkeeper, "frozen_a");
        let access = prime_attrfamily_with_int_attr(&ann.bookkeeper, &desc_a, "count");
        let r = std::sync::Arc::new(MultipleFrozenPBCRepr::new(&rtyper, Some(access)));
        let c = r.convert_const(&ConstValue::None).unwrap();
        let ConstValue::LLPtr(null_ptr) = &c.value else {
            panic!("expected LLPtr");
        };
        assert!(
            matches!(null_ptr._obj0, Ok(None)),
            "null_instance must produce Ok(None) _obj0"
        );
        // And the concretetype matches the repr's Ptr lowleveltype.
        assert_eq!(c.concretetype.as_ref(), Some(r.lowleveltype()));
        // Further sanity: pbc_type must have resolved to a Struct now
        // that ensure_setup has run.
        let resolved = r.pbc_type.resolved().expect("pbc_type resolved");
        assert!(matches!(resolved, LowLevelType::Struct(_)));
        drop(_ptr_obj::Struct as fn(_) -> _); // keep _ptr_obj used for error paths
    }

    #[test]
    fn multiple_frozen_pbc_repr_convert_desc_recursive_cache_placeholder_stabilises() {
        // upstream rpbc.py:776-778 — `self.pbc_cache[frozendesc] = result`
        // happens BEFORE the field-populate loop so a recursive
        // convert_desc on the same frozendesc sees the placeholder
        // struct pointer instead of looping forever.
        //
        // Scenario: two FrozenDescs A, B sharing a FrozenAttrFamily
        // with a single attr "peer". A.attrcache["peer"] = B_pyobj;
        // B.attrcache["peer"] = A_pyobj. Pre-wire pbc_reprs so
        // rtyper.getrepr(SomePBC([A, B])) returns THIS same MFPBC.
        use crate::annotator::description::FrozenAttrFamily;
        use crate::annotator::model::{SomeInteger, SomePBC, SomeValue};
        use crate::flowspace::model::{ConstValue, HostObject};
        use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
        use crate::translator::rtyper::rtyper::PbcReprKey;

        let (ann, rtyper) = make_rtyper();
        let desc_a = frozen_entry(&ann.bookkeeper, "frozen_a");
        let desc_b = frozen_entry(&ann.bookkeeper, "frozen_b");

        // Unify both under one FrozenAttrFamily and add attr "peer"
        // typed as SomePBC([A, B]) — so the field's r_value ends up
        // being the MFPBC that references the same access_set.
        let (DescEntry::Frozen(fd_a), DescEntry::Frozen(fd_b)) = (&desc_a, &desc_b) else {
            unreachable!();
        };
        let a_id = fd_a.borrow().base.identity;
        let b_id = fd_b.borrow().base.identity;
        let family = {
            let mut uf = ann.bookkeeper.frozenpbc_attr_families.borrow_mut();
            let _ = uf.find_rep(a_id);
            let _ = uf.find_rep(b_id);
            let rep = uf.find_rep(a_id);
            let (_changed, new_rep) = uf.union(rep, b_id);
            uf.get(&new_rep).cloned().unwrap()
        };
        // Patch the family in place so descs + attrs match.
        {
            let mut fam = family.borrow_mut();
            // After union both identities map to the same FrozenAttrFamily;
            // ensure both are listed as members.
            fam.descs.clear();
            fam.descs.insert(a_id, ());
            fam.descs.insert(b_id, ());
            fam.attrs.insert(
                "peer".to_string(),
                SomeValue::PBC(SomePBC::new(vec![desc_a.clone(), desc_b.clone()], false)),
            );
            // Suppress the single-desc-const-box path so
            // `getrepr(SomePBC([A,B]))` doesn't auto-collapse.
            let _ = SomeInteger::default();
        }

        // Ensure bookkeeper.getdesc(pyobj) resolves to our descs.
        let a_pyobj = fd_a.borrow().base.pyobj.as_ref().cloned().unwrap();
        let b_pyobj = fd_b.borrow().base.pyobj.as_ref().cloned().unwrap();
        ann.bookkeeper
            .descs
            .borrow_mut()
            .insert(a_pyobj.clone(), desc_a.clone());
        ann.bookkeeper
            .descs
            .borrow_mut()
            .insert(b_pyobj.clone(), desc_b.clone());

        // Pre-populate the FrozenDesc attrcaches so convert_desc's
        // populate loop finds the cross-reference values without
        // hitting `read_attribute`.
        fd_a.borrow()
            .attrcache
            .borrow_mut()
            .insert("peer".to_string(), ConstValue::HostObject(b_pyobj.clone()));
        fd_b.borrow()
            .attrcache
            .borrow_mut()
            .insert("peer".to_string(), ConstValue::HostObject(a_pyobj.clone()));

        // Build the MFPBC and wire it into rtyper.pbc_reprs so
        // rtyper.getrepr on SomePBC([A,B]) returns this same instance.
        let r = std::sync::Arc::new(MultipleFrozenPBCRepr::new(&rtyper, Some(family.clone())));
        let access_key = Rc::as_ptr(&family) as usize;
        rtyper.pbc_reprs.borrow_mut().insert(
            PbcReprKey::Access(access_key),
            r.clone() as std::sync::Arc<dyn Repr>,
        );

        // Recursive convert_desc — the placeholder insertion order
        // must stabilise: convert_desc(A) → fields populate →
        // convert_const(B_pyobj) → convert_desc(B) → fields populate →
        // convert_const(A_pyobj) → convert_desc(A) hits the cache
        // (placeholder struct). No stack overflow; both descs end up
        // with fully-populated struct pointers pointing at each
        // other.
        let c_a = r.convert_desc(&desc_a).expect("convert_desc(A)");
        let c_b = r.convert_desc(&desc_b).expect("convert_desc(B)");

        // Both cached Constants carry LLPtr values with populated
        // Struct bodies.
        for (label, c) in [("A", &c_a), ("B", &c_b)] {
            let ConstValue::LLPtr(ptr) = &c.value else {
                panic!("{label}: expected LLPtr");
            };
            let Ok(Some(_ptr_obj::Struct(s))) = &ptr._obj0 else {
                panic!("{label}: expected Ok(Some(Struct))");
            };
            let mangled = crate::translator::rtyper::rmodel::mangle("pbc", "peer");
            let field = s._getattr(&mangled).expect("peer field present");
            // Each peer field points at the *other* desc's Ptr — the
            // exact value differs but both must be non-null Ptrs.
            use crate::translator::rtyper::lltypesystem::lltype::LowLevelValue;
            let LowLevelValue::Ptr(peer) = field else {
                panic!("{label}: peer field is not Ptr");
            };
            assert!(peer.nonzero(), "{label}: peer pointer must be non-null");
        }

        // Second convert_desc on A must return the SAME cached Constant
        // (pbc_cache reuse).
        let c_a2 = r.convert_desc(&desc_a).unwrap();
        assert_eq!(c_a, c_a2);

        // upstream rpbc.py:776-778 invariant: every cross-reference to A
        // — including the placeholder snapshot captured in B's "peer"
        // field by the recursive convert_desc(A) — must compare equal
        // to the post-`_become` cache entry. Without redirect-following
        // in `_hashable_identity`, the placeholder's raw `_identity`
        // would differ from the final pointer's `_identity` and
        // `ConstValue::LLPtr` Hash/Eq would break the invariant.
        let ConstValue::LLPtr(ptr_b) = &c_b.value else {
            panic!("c_b expected LLPtr");
        };
        let Ok(Some(_ptr_obj::Struct(s_b))) = &ptr_b._obj0 else {
            panic!("c_b struct missing");
        };
        let mangled = crate::translator::rtyper::rmodel::mangle("pbc", "peer");
        let peer_field = s_b
            ._getattr(&mangled)
            .expect("peer field present on B's struct");
        let crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Ptr(peer_a_in_b) =
            peer_field
        else {
            panic!("peer field must be Ptr");
        };
        let ConstValue::LLPtr(ptr_a_cached) = &c_a.value else {
            panic!("c_a expected LLPtr");
        };
        // The hashable identities must coincide: the recursive snapshot
        // of A captured inside B was a placeholder, and the cache holds
        // the post-_become final pointer. Both must hash/eq identically.
        assert_eq!(
            peer_a_in_b._hashable_identity(),
            ptr_a_cached._hashable_identity(),
            "placeholder snapshot vs. post-_become final pointer must \
             share hashable identity (rpbc.py:776-778 redirect)"
        );
    }

    #[test]
    fn multiple_frozen_pbc_repr_convert_desc_rejects_non_access_set_member() {
        // rpbc.py:770-772 — `if access_set is not None and frozendesc
        // not in access_set.descs: raise TyperError`.
        let (ann, rtyper) = make_rtyper();
        let desc_a = frozen_entry(&ann.bookkeeper, "frozen_a");
        let desc_b = frozen_entry(&ann.bookkeeper, "frozen_b");
        let access = prime_attrfamily_with_int_attr(&ann.bookkeeper, &desc_a, "count");
        let r = std::sync::Arc::new(MultipleFrozenPBCRepr::new(&rtyper, Some(access)));
        let err = r.convert_desc(&desc_b).unwrap_err();
        assert!(
            err.to_string().contains("not found in PBC access set"),
            "unexpected: {err}"
        );
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
    fn classes_pbc_repr_non_constant_branch_surfaces_pending() {
        let (ann, rtyper) = make_rtyper();
        // can_be_none=true with a single desc: upstream's
        // `s_pbc.is_constant()` would return False (const missing);
        // dispatch falls to getlowleveltype which is unported.
        let c = class_entry(&ann.bookkeeper, "C");
        let s_pbc = SomePBC::new(vec![c], true);
        let err = ClassesPBCRepr::new(&rtyper, s_pbc).unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("getlowleveltype"));
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

        // Non-None value on Void lowleveltype requires getdesc(cls)
        // which is not ported.
        let err = r.convert_const(&ConstValue::Int(7)).unwrap_err();
        assert!(err.is_missing_rtype_operation());
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
    fn somepbc_rtyper_makerepr_single_frozen_desc_with_can_be_none_returns_multiple_frozen_pbc_repr()
     {
        // Single FrozenDesc with `can_be_None=true` falls out of the
        // SingleFrozenPBCRepr fast-path, hits getFrozenPBCRepr's
        // multi-desc branch, and routes to MultipleFrozenPBCRepr
        // (with access_set = None because the single desc has no
        // attribute family touches).
        let (ann, rtyper) = make_rtyper();
        let f = frozen_entry(&ann.bookkeeper, "frozen0");
        let s_pbc = SomePBC::new(vec![f], true);
        let r = somepbc_rtyper_makerepr(&s_pbc, &rtyper).unwrap();
        assert_eq!(r.class_name(), "MultipleFrozenPBCRepr");
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
    ) -> crate::translator::rtyper::rtyper::LowLevelOpList {
        crate::translator::rtyper::rtyper::LowLevelOpList::new(rtyper.clone(), None)
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
    fn function_repr_rtype_simple_call_emits_direct_call_to_annotated_callable_graph() {
        // End-to-end smoke of FunctionReprBase.call (rpbc.py:199-221):
        // FunctionRepr (Void-typed) + single uniquerow → convert_to_concrete_llfn
        // yields a Constant LLPtr, call_emit emits `direct_call`. The graph's
        // returnvar carries an Integer annotation so `getrresult` resolves to
        // an IntegerRepr (Signed), matching the `r_result` wired on the hop.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{
            Block as FlowBlock, BlockRefExt, FunctionGraph as FlowFunctionGraph,
            GraphFunc as FlowGraphFunc, Hlvalue as FlowHlvalue, Link as FlowLink, SpaceOperation,
            Variable as FlowVariable,
        };
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;
        use std::sync::Arc;

        let (ann, rtyper) = make_rtyper();
        // Build a FunctionDesc + PyGraph whose startblock arg and return
        // var are annotated as SomeInteger — this is what makes
        // `callparse::getrinputs` / `getrresult` route to IntegerRepr.
        let sig = Signature::new(vec!["x".to_string()], None, None);
        let fd = Rc::new(StdRefCell::new(FunctionDesc::new(
            ann.bookkeeper.clone(),
            None,
            "annotated_f",
            sig.clone(),
            None,
            None,
        )));
        let mut arg_var = FlowVariable::named("x");
        arg_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::new(
                false, false,
            )))));
        let arg = FlowHlvalue::Variable(arg_var);
        let startblock = FlowBlock::shared(vec![arg.clone()]);
        let mut ret_var = FlowVariable::new();
        ret_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::new(
                false, false,
            )))));
        let graph = FlowFunctionGraph::with_return_var(
            "annotated_f",
            startblock.clone(),
            FlowHlvalue::Variable(ret_var),
        );
        let link = Rc::new(StdRef::new(FlowLink::new(
            vec![arg],
            Some(graph.returnblock.clone()),
            None,
        )));
        startblock.closeblock(vec![link]);
        let pygraph = Rc::new(PyGraph {
            graph: Rc::new(StdRef::new(graph)),
            func: FlowGraphFunc::new(
                "annotated_f",
                crate::flowspace::model::Constant::new(ConstValue::Dict(Default::default())),
            ),
            signature: StdRef::new(sig),
            defaults: StdRef::new(None),
            access_directly: std::cell::Cell::new(false),
        });
        fd.borrow()
            .cache
            .borrow_mut()
            .insert(GraphCacheKey::None, pygraph.clone());
        // Populate the callfamily simple-call shape via consider_call_site.
        let args = ArgumentsForTranslation::new(
            vec![SomeValue::Integer(SomeInteger::default())],
            None,
            None,
        );
        FunctionDesc::consider_call_site(&[fd.clone()], &args, &SomeValue::Impossible, None)
            .unwrap();

        let s_pbc = SomePBC::new(vec![DescEntry::Function(fd)], false);
        let r_func = Arc::new(FunctionRepr::new(&rtyper, s_pbc.clone()).unwrap());
        let r_int: Arc<dyn Repr> = rtyper
            .getrepr(&SomeValue::Integer(SomeInteger::new(false, false)))
            .unwrap();

        // Construct the hop for `simple_call(f, 5)`.
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            Vec::new(),
            FlowHlvalue::Variable(FlowVariable::new()),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops.clone());
        let mut v_func = FlowVariable::new();
        v_func.set_concretetype(Some(LowLevelType::Void));
        hop.args_v.borrow_mut().extend([
            FlowHlvalue::Variable(v_func),
            FlowHlvalue::Constant(crate::flowspace::model::Constant::with_concretetype(
                ConstValue::Int(5),
                LowLevelType::Signed,
            )),
        ]);
        hop.args_s.borrow_mut().extend([
            SomeValue::PBC(s_pbc),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        let r_func_dyn: Arc<dyn Repr> = r_func.clone();
        hop.args_r
            .borrow_mut()
            .extend([Some(r_func_dyn), Some(r_int.clone())]);
        *hop.r_result.borrow_mut() = Some(r_int);

        let result = r_func.rtype_simple_call(&hop).unwrap().unwrap();
        assert!(matches!(result, FlowHlvalue::Variable(_)));
        let ops = llops.borrow();
        let emitted_opnames: Vec<&str> = ops.ops.iter().map(|op| op.opname.as_str()).collect();
        assert!(
            emitted_opnames.contains(&"direct_call"),
            "expected direct_call op, got {:?}",
            emitted_opnames
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
    fn functions_pbc_repr_convert_const_host_function_routes_through_getdesc_to_convert_desc() {
        // rpbc.py:297-298 — non-None value becomes
        // `funcdesc = bookkeeper.getdesc(value) ; self.convert_desc(funcdesc)`.
        // Prime `bookkeeper.descs` so `getdesc` returns fd_f without
        // running the newfuncdesc path, and verify convert_const emits
        // the same LLPtr Constant that convert_desc(fd_f) would.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, GraphFunc, HostObject};

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

        let host_f = HostObject::new_user_function(GraphFunc::new(
            "f",
            crate::flowspace::model::Constant::new(ConstValue::Dict(Default::default())),
        ));
        ann.bookkeeper
            .descs
            .borrow_mut()
            .insert(host_f.clone(), DescEntry::Function(fd_f.clone()));

        let s_pbc = SomePBC::new(
            vec![DescEntry::Function(fd_f.clone()), DescEntry::Function(fd_g)],
            false,
        );
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();

        let expected = r.convert_desc(&DescEntry::Function(fd_f)).unwrap();
        let got = r.convert_const(&ConstValue::HostObject(host_f)).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn functions_pbc_repr_convert_const_staticmethod_unwraps_then_routes_to_getdesc() {
        // rpbc.py:292-293 — `staticmethod.__get__(42)` unwraps the
        // wrapped callable. Rust port uses
        // `HostObject::staticmethod_func()`. After unwrap, the same
        // getdesc → convert_desc chain fires.
        use crate::annotator::argument::ArgumentsForTranslation;
        use crate::annotator::description::GraphCacheKey;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::{ConstValue, GraphFunc, HostObject};

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

        let host_f = HostObject::new_user_function(GraphFunc::new(
            "f",
            crate::flowspace::model::Constant::new(ConstValue::Dict(Default::default())),
        ));
        ann.bookkeeper
            .descs
            .borrow_mut()
            .insert(host_f.clone(), DescEntry::Function(fd_f.clone()));
        let sm = HostObject::new_staticmethod("Cls.f", host_f);

        let s_pbc = SomePBC::new(
            vec![DescEntry::Function(fd_f.clone()), DescEntry::Function(fd_g)],
            false,
        );
        let r = FunctionsPBCRepr::new(&rtyper, s_pbc).unwrap();

        let expected = r.convert_desc(&DescEntry::Function(fd_f)).unwrap();
        let got = r.convert_const(&ConstValue::HostObject(sm)).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn functions_pbc_repr_convert_const_non_callable_const_surfaces_message() {
        // Non-None, non-HostObject constants cannot route through
        // bookkeeper.getdesc. Port returns a structured error rather
        // than panicking. Upstream's `getdesc(int_value)` would raise
        // inside the Bookkeeper; pyre surfaces it at the repr.
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
        assert!(err.to_string().contains("expected callable HostObject"));
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
