//! RPython `rpython/rtyper/rlist.py` + `lltypesystem/rlist.py` —
//! minimal `FixedSizeListRepr` slice covering the `len(list)` lowering.
//!
//! Rust slices (`&[T]`) and fixed arrays annotate as a non-resized
//! `SomeList` (`bookkeeper`), so `slice.len()` — routed to the rtyper's
//! `len` operation by `flowspace_adapter` — lands on this repr's
//! `rtype_len`.
//!
//! The resized `ListRepr` lands its data shape (`Ptr(GcStruct("list",
//! ("length", Signed), ("items", Ptr(GcArray(ITEM)))))`) and the
//! `rtype_len` lowering (`ll_length` reads the `length` field). The
//! `ADTIList` / `ADTIFixedList` adtmeth tables, `rtype_method_append` /
//! `getitem` / `setitem` / iterators, the `_ll_list_resize*` family and
//! the `pairtype(BaseListRepr, BaseListRepr)` operations are deferred to
//! follow-on slices. For `FixedSizeListRepr` only the data shape
//! `Ptr(GcArray(ITEM))` and the `ll_fixed_length` lowering land.

use std::rc::Rc;
use std::sync::Arc;

use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    ArrayType, LowLevelType, Ptr, PtrTarget, StructType,
};
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, HighLevelOp, RPythonTyper, helper_pygraph_from_graph, variable_with_lltype,
    void_field_const,
};

/// RPython `class FixedSizeListRepr(AbstractFixedSizeListRepr,
/// BaseListRepr)` (`lltypesystem/rlist.py:173-187`):
///
/// ```python
/// def _setup_repr(self):
///     if 'item_repr' not in self.__dict__:
///         self.external_item_repr, self.item_repr = externalvsinternal(
///             self.rtyper, self._item_repr_computer(), gcref=True)
///     if isinstance(self.LIST, GcForwardReference):
///         ITEMARRAY = self.get_itemarray_lowleveltype()
///         self.LIST.become(ITEMARRAY)
/// ```
///
/// `LIST` becomes a bare `GcArray(ITEM)` (no `length`/`items` wrapper —
/// that is the resized `ListRepr`), so the low-level type is
/// `Ptr(GcArray(ITEM))`.
#[derive(Debug)]
pub struct FixedSizeListRepr {
    state: ReprState,
    lltype: LowLevelType,
    /// `self.item_repr` (`lltypesystem/rlist.py:177`) — the internal
    /// (gcref-wrapped) element repr.
    #[allow(dead_code)]
    item_repr: Arc<dyn Repr>,
}

impl FixedSizeListRepr {
    pub fn new(rtyper: &Rc<RPythonTyper>, item_repr: Arc<dyn Repr>) -> Result<Self, TyperError> {
        // `externalvsinternal(rtyper, item_repr, gcref=True)` —
        // gc `InstanceRepr` items become the generic `Ptr(OBJECT)`
        // gcref so the array element type is never a gc container
        // (which `ArrayType::gc` rejects); non-instance reprs pass
        // through unchanged.
        let (_external, internal) =
            crate::translator::rtyper::rclass::externalvsinternal(rtyper, item_repr)?;
        let item_lltype = internal.lowleveltype().clone();
        let arr = ArrayType::gc(item_lltype);
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(arr),
        }));
        Ok(FixedSizeListRepr {
            state: ReprState::new(),
            lltype,
            item_repr: internal,
        })
    }
}

impl Repr for FixedSizeListRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "FixedSizeListRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::FixedSizeListRepr
    }

    /// RPython `AbstractBaseListRepr.rtype_len(self, hop)`
    /// (`rlist.py:124-130`):
    ///
    /// ```python
    /// def rtype_len(self, hop):
    ///     v_lst, = hop.inputargs(self)
    ///     if hop.args_s[0].listdef.listitem.resized:
    ///         ll_func = ll_len
    ///     else:
    ///         ll_func = ll_len_foldable
    ///     return hop.gendirectcall(ll_func, v_lst)
    /// ```
    ///
    /// `FixedSizeListRepr` is only minted for the non-resized case, so
    /// the lowering is the `ll_len_foldable` path: `l.ll_length()` →
    /// `ll_fixed_length` (`lltypesystem/rlist.py:395-396`) = `len(l)` =
    /// the `getarraysize` op on the `Ptr(GcArray)` receiver. The
    /// `len_foldable` oopspec is a tracing-time JIT hint; the lowered
    /// op is the bare `getarraysize`.
    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let ptr_lltype = self.lltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_fixed_length".to_string(),
            vec![ptr_lltype],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_fixed_length_helper_graph("ll_fixed_length", ptr_for_builder.clone())
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    }
}

/// Synthesise `LLHelpers`-style `ll_fixed_length`
/// (`lltypesystem/rlist.py:395-396`):
///
/// ```python
/// def ll_fixed_length(l):
///     return len(l)
/// ```
///
/// Single-block graph: `getarraysize(l) -> Signed`. Unlike
/// `ll_strlen` (which `getsubstruct`s the nested `chars` array first),
/// the `FixedSizeListRepr` receiver IS the `Ptr(GcArray)`, so the
/// length is one op.
pub(crate) fn build_ll_fixed_length_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("l", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_len = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(arg)],
        Hlvalue::Variable(v_len.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_len)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string()],
        func,
    ))
}

/// RPython `class ListRepr(AbstractListRepr, BaseListRepr)`
/// (`lltypesystem/rlist.py:107-133`):
///
/// ```python
/// def _setup_repr(self):
///     if 'item_repr' not in self.__dict__:
///         self.external_item_repr, self.item_repr = externalvsinternal(
///             self.rtyper, self._item_repr_computer(), gcref=True)
///     if isinstance(self.LIST, GcForwardReference):
///         ITEM = self.item_repr.lowleveltype
///         ITEMARRAY = self.get_itemarray_lowleveltype()
///         self.LIST.become(GcStruct("list", ("length", Signed),
///                                           ("items", Ptr(ITEMARRAY)),
///                                   adtmeths = ADTIList({...}),
///                                   hints = {'list': True}))
/// ```
///
/// Unlike [`FixedSizeListRepr`] (whose `LIST` is the bare
/// `GcArray(ITEM)`), the resized list wraps the array in a `length` +
/// `items` header struct so `append`/resize can grow `items` while the
/// `length` counter tracks the live element count. The low-level type
/// is therefore `Ptr(GcStruct("list", length, items))`.
///
/// This slice lands the data shape and `rtype_len`; the `ADTIList`
/// adtmeth table and the `append`/`getitem`/`setitem`/resize ops are
/// follow-on slices.
#[derive(Debug)]
pub struct ListRepr {
    state: ReprState,
    lltype: LowLevelType,
    /// `self.item_repr` (`rlist.py:111`) — the internal (gcref-wrapped)
    /// element repr.
    #[allow(dead_code)]
    item_repr: Arc<dyn Repr>,
}

impl ListRepr {
    pub fn new(rtyper: &Rc<RPythonTyper>, item_repr: Arc<dyn Repr>) -> Result<Self, TyperError> {
        // `externalvsinternal(rtyper, item_repr, gcref=True)` — same
        // gcref normalisation as `FixedSizeListRepr`: gc `InstanceRepr`
        // items become the generic `Ptr(OBJECT)` gcref so the array
        // element type is never a gc container.
        let (_external, internal) =
            crate::translator::rtyper::rclass::externalvsinternal(rtyper, item_repr)?;
        let item_lltype = internal.lowleveltype().clone();
        // upstream `get_itemarray_lowleveltype()` — `GcArray(ITEM)` (the
        // `ADTIFixedList` adtmeths it carries are unused until the array
        // ops land, so the bare array suffices for this slice).
        let itemarray = ArrayType::gc(item_lltype);
        let items_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(itemarray),
        }));
        // upstream `GcStruct("list", ("length", Signed), ("items",
        // Ptr(ITEMARRAY)), hints={'list': True})`.
        let list_struct = StructType::gc_with_hints(
            "list",
            vec![
                ("length".to_string(), LowLevelType::Signed),
                ("items".to_string(), items_ptr),
            ],
            vec![("list".to_string(), ConstValue::Bool(true))],
        );
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(list_struct),
        }));
        Ok(ListRepr {
            state: ReprState::new(),
            lltype,
            item_repr: internal,
        })
    }
}

impl Repr for ListRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ListRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::ListRepr
    }

    /// RPython `AbstractBaseListRepr.rtype_len(self, hop)`
    /// (`rlist.py:124-130`): the resized list takes the `ll_len`
    /// (non-foldable) branch, which reads the struct `length` field —
    /// `ll_length(l)` = `l.length` (`lltypesystem/rlist.py` ADTIList).
    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let ptr_lltype = self.lltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_length".to_string(),
            vec![ptr_lltype],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_length_helper_graph("ll_length", ptr_for_builder.clone())
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    }
}

/// Synthesise `ll_length(l) -> Signed` (`lltypesystem/rlist.py` ADTIList
/// `ll_length`):
///
/// ```python
/// def ll_length(l):
///     return l.length
/// ```
///
/// Single-block graph: `getfield(l, "length") -> Signed`. The receiver
/// is the `Ptr(GcStruct("list", ...))`, so the length is the struct's
/// `length` header field (vs `FixedSizeListRepr`'s `getarraysize`).
pub(crate) fn build_ll_length_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("l", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_len = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(arg), void_field_const("length")],
        Hlvalue::Variable(v_len.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_len)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string()],
        func,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rint::IntegerRepr;

    fn fresh_rtyper() -> Rc<RPythonTyper> {
        let ann = RPythonAnnotator::new(None, None, None, false);
        Rc::new(RPythonTyper::new(&ann))
    }

    /// `initialize_exceptiondata` sets the typer's self-weak, required by
    /// `rtyper_makerepr`'s `SomeList` arm (`self_rc()`).
    fn fresh_rtyper_live() -> Rc<RPythonTyper> {
        let rtyper = fresh_rtyper();
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        rtyper
    }

    #[test]
    fn list_repr_lltype_is_ptr_gcstruct_length_items() {
        // `LIST = GcStruct("list", ("length", Signed), ("items",
        // Ptr(GcArray(ITEM))))`, lowleveltype `Ptr(LIST)`.
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = ListRepr::new(&rtyper, r_int).expect("ListRepr::new");
        assert_eq!(repr.class_name(), "ListRepr");

        let LowLevelType::Ptr(ptr) = repr.lowleveltype() else {
            panic!("ListRepr lltype must be a Ptr");
        };
        let PtrTarget::Struct(body) = &ptr.TO else {
            panic!("ListRepr Ptr target must be a Struct");
        };
        assert_eq!(body._name, "list");
        assert_eq!(body._flds.get("length"), Some(&LowLevelType::Signed));
        let items = body._flds.get("items").expect("items field");
        let LowLevelType::Ptr(items_ptr) = items else {
            panic!("items field must be a Ptr");
        };
        assert!(
            matches!(items_ptr.TO, PtrTarget::Array(_)),
            "items must point to a GcArray"
        );
    }

    #[test]
    fn makerepr_resized_somelist_routes_to_list_repr() {
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeList, SomeValue};
        use crate::translator::rtyper::rmodel::rtyper_makerepr;

        let rtyper = fresh_rtyper_live();
        // `resized=true` → the `ListRepr` (resized) branch.
        let ldef = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            true,
        );
        let sv = SomeValue::List(SomeList::new(ldef));
        let repr = rtyper_makerepr(&sv, &rtyper).expect("rtyper_makerepr resized list");
        assert_eq!(repr.class_name(), "ListRepr");
        assert_eq!(repr.repr_class_id(), ReprClassId::ListRepr);
    }

    #[test]
    fn makerepr_nonresized_somelist_routes_to_fixed_size_list_repr() {
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeList, SomeValue};
        use crate::translator::rtyper::rmodel::rtyper_makerepr;

        let rtyper = fresh_rtyper_live();
        // `resized=false` → the `FixedSizeListRepr` branch (unchanged).
        let ldef = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );
        let sv = SomeValue::List(SomeList::new(ldef));
        let repr = rtyper_makerepr(&sv, &rtyper).expect("rtyper_makerepr non-resized list");
        assert_eq!(repr.class_name(), "FixedSizeListRepr");
        assert_eq!(repr.repr_class_id(), ReprClassId::FixedSizeListRepr);
    }
}
