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
    /// (gcref-wrapped) element repr; its lowleveltype is the array
    /// element type and the `getitem` result type.
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

    /// RPython `pair(AbstractBaseListRepr, IntegerRepr).rtype_getitem`
    /// (`rlist.py:247-267`):
    ///
    /// ```python
    /// def rtype_getitem((r_lst, r_int), hop, checkidx=False):
    ///     v_lst, v_index = hop.inputargs(r_lst, Signed)
    ///     ...
    ///     spec = dum_nocheck
    ///     hop.exception_cannot_occur()
    ///     ...
    ///     if hop.args_s[1].nonneg:
    ///         llfn = ll_getitem_nonneg
    ///     ...
    ///     v_res = hop.gendirectcall(llfn, ..., v_lst, v_index)
    ///     return r_lst.recast(hop.llops, v_res)
    /// ```
    ///
    /// `FixedSizeListRepr` is the non-resized list — a Rust slice
    /// (`&[T]`) or fixed array, indexed by `usize` values that annotate
    /// as a non-negative `SomeInteger`. Only the `ll_getitem_nonneg` +
    /// `dum_nocheck` branch arises, and that chain collapses through
    /// `ll_getitem_foldable_nonneg` → `ll_fixed_getitem_fast(l, index)` →
    /// `l[index]` (`lltypesystem/rlist.py:402-405`) to the bare
    /// `getarrayitem` on the `Ptr(GcArray)` receiver. The negative-index
    /// (`ll_getitem`) and `checkidx` (IndexError-raising) branches surface
    /// a `TyperError` until those helpers land — Rust slice indexing never
    /// produces them. `recast` is a no-op here (the item lltype is already
    /// the array element type).
    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::annotator::model::SomeValue;
        let s1 = hop
            .args_s
            .borrow()
            .get(1)
            .cloned()
            .ok_or_else(|| TyperError::message("list rtype_getitem: args_s[1] missing"))?;
        let nonneg = match &s1 {
            SomeValue::Integer(i) => i.nonneg,
            other => {
                return Err(TyperError::message(format!(
                    "list rtype_getitem: args_s[1] must be SomeInteger, got {other:?}"
                )));
            }
        };
        if !nonneg {
            return Err(TyperError::message(
                "list rtype_getitem: negative-index ll_getitem branch not yet ported",
            ));
        }
        let args = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ])?;
        hop.exception_cannot_occur()?;
        let item_lltype = self.item_repr.lowleveltype().clone();
        let ptr_lltype = self.lltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let item_for_builder = item_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_fixed_getitem_fast".to_string(),
            vec![ptr_lltype, LowLevelType::Signed],
            item_lltype,
            move |_rtyper, _args, _result| {
                build_ll_fixed_getitem_fast_helper_graph(
                    "ll_fixed_getitem_fast",
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, args)
    }

    /// RPython `pair(AbstractBaseListRepr, IntegerRepr).rtype_setitem`
    /// (`rlist.py:272-284`):
    ///
    /// ```python
    /// def rtype_setitem((r_lst, r_int), hop):
    ///     if hop.has_implicit_exception(IndexError):
    ///         spec = dum_checkidx
    ///     else:
    ///         spec = dum_nocheck
    ///     v_func = hop.inputconst(Void, spec)
    ///     v_lst, v_index, v_item = hop.inputargs(r_lst, Signed, r_lst.item_repr)
    ///     if hop.args_s[1].nonneg:
    ///         llfn = ll_setitem_nonneg
    ///     else:
    ///         llfn = ll_setitem
    ///     hop.exception_is_here()
    ///     return hop.gendirectcall(llfn, v_func, v_lst, v_index, v_item)
    /// ```
    ///
    /// `FixedSizeListRepr` handles the `dum_nocheck` + nonneg fast path:
    /// `ll_setitem_nonneg(dum_nocheck, l, index, item)` collapses (no
    /// IndexError branch, `index >= 0` is a debug `ll_assert`) through
    /// `l.ll_setitem_fast` → `ll_fixed_setitem_fast(l, index, item)` →
    /// `l[index] = item` (`lltypesystem/rlist.py:407-410`) to the bare
    /// `setarrayitem` on the `Ptr(GcArray)` receiver. The third inputarg
    /// converts to `item_repr` (the gcref-wrapped element repr). The
    /// `checkidx` (implicit-IndexError) and negative-index (`ll_setitem`)
    /// branches surface a `TyperError` until those helpers land — Rust
    /// slice indexing never produces them.
    fn rtype_setitem(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::annotator::model::SomeValue;
        if hop.has_implicit_exception("IndexError") {
            return Err(TyperError::message(
                "list rtype_setitem: checkidx IndexError branch not yet ported",
            ));
        }
        let s1 = hop
            .args_s
            .borrow()
            .get(1)
            .cloned()
            .ok_or_else(|| TyperError::message("list rtype_setitem: args_s[1] missing"))?;
        let nonneg = match &s1 {
            SomeValue::Integer(i) => i.nonneg,
            other => {
                return Err(TyperError::message(format!(
                    "list rtype_setitem: args_s[1] must be SomeInteger, got {other:?}"
                )));
            }
        };
        if !nonneg {
            return Err(TyperError::message(
                "list rtype_setitem: negative-index ll_setitem branch not yet ported",
            ));
        }
        let args = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::LowLevelType(&LowLevelType::Signed),
            ConvertedTo::Repr(self.item_repr.as_ref()),
        ])?;
        hop.exception_is_here()?;
        let item_lltype = self.item_repr.lowleveltype().clone();
        let ptr_lltype = self.lltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let item_for_builder = item_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_fixed_setitem_fast".to_string(),
            vec![ptr_lltype, LowLevelType::Signed, item_lltype],
            LowLevelType::Void,
            move |_rtyper, _args, _result| {
                build_ll_fixed_setitem_fast_helper_graph(
                    "ll_fixed_setitem_fast",
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, args)
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

/// Synthesise `ll_fixed_getitem_fast` (`lltypesystem/rlist.py:402-405`):
///
/// ```python
/// def ll_fixed_getitem_fast(l, index):
///     ll_assert(index < len(l), "fixed getitem out of bounds")
///     return l[index]
/// ```
///
/// The `ll_assert` is a debug-only bound check (no production op); the
/// `FixedSizeListRepr` receiver IS the `Ptr(GcArray)`, so the body is the
/// single `getarrayitem(l, index) -> ITEM` op (unlike `ll_stritem_nonneg`,
/// which `getsubstruct`s the nested `chars` array first).
pub(crate) fn build_ll_fixed_getitem_fast_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let l_arg = variable_with_lltype("l", ptr_lltype);
    let index_arg = variable_with_lltype("index", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(index_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_item = variable_with_lltype("item", item_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem",
        vec![Hlvalue::Variable(l_arg), Hlvalue::Variable(index_arg)],
        Hlvalue::Variable(v_item.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_item)],
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
        vec!["l".to_string(), "index".to_string()],
        func,
    ))
}

/// Synthesise `ll_fixed_setitem_fast` (`lltypesystem/rlist.py:407-410`):
///
/// ```python
/// def ll_fixed_setitem_fast(l, index, item):
///     ll_assert(index < len(l), "fixed setitem out of bounds")
///     l[index] = item
/// ```
///
/// The `ll_assert` is a debug-only bound check (no production op); the
/// `FixedSizeListRepr` receiver IS the `Ptr(GcArray)`, so the body is the
/// single `setarrayitem(l, index, item)` op. The function falls off the
/// end (`return None`), so the returnblock receives the `Void` `None`
/// constant.
pub(crate) fn build_ll_fixed_setitem_fast_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let l_arg = variable_with_lltype("l", ptr_lltype);
    let index_arg = variable_with_lltype("index", LowLevelType::Signed);
    let item_arg = variable_with_lltype("item", item_lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l_arg.clone()),
        Hlvalue::Variable(index_arg.clone()),
        Hlvalue::Variable(item_arg.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(l_arg),
            Hlvalue::Variable(index_arg),
            Hlvalue::Variable(item_arg),
        ],
        Hlvalue::Variable(v_void),
    ));
    let none_const = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ));
    startblock.closeblock(vec![
        Link::new(vec![none_const], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["l".to_string(), "index".to_string(), "item".to_string()],
        func,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::listdef::ListDef;
    use crate::annotator::model::{SomeInteger, SomeList, SomeValue};
    use crate::flowspace::model::Variable;
    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rint::{IntegerRepr, signed_repr};
    use crate::translator::rtyper::rmodel::rtyper_makerepr;
    use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};

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

    /// rlist.py:247-267 nonneg + checkidx=False branch — `getitem` on a
    /// `FixedSizeListRepr` lowers to a `direct_call` of
    /// `ll_fixed_getitem_fast` (a single `getarrayitem` on the
    /// `Ptr(GcArray)` receiver), preceded by `hop.exception_cannot_occur()`.
    #[test]
    fn fixed_size_list_getitem_nonneg_emits_direct_call_to_ll_fixed_getitem_fast() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        // A single shared repr instance: `convertvar` short-circuits on
        // `std::ptr::eq` of the `&dyn Repr`, so `args_r[0]` and the `self`
        // routed through `rtype_getitem` must be the same object.
        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_list), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ false,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ true, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = list_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("list getitem nonneg: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "checkidx=False path must call hop.exception_cannot_occur()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_fixed_getitem_fast"),
            "expected 'll_fixed_getitem_fast' in {dbg}"
        );
    }

    /// Negative-index annotation (`args_s[1].nonneg == false`) is not yet
    /// ported (the `ll_getitem` neg-fix branch) — it surfaces a
    /// `TyperError` so the subject stays on the legacy walker rather than
    /// miscompiling. Rust slice indexing never produces it.
    #[test]
    fn fixed_size_list_getitem_negative_index_is_deferred() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_list), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ false,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ false, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);
        assert!(list_repr.rtype_getitem(&hop).is_err());
    }

    /// rlist.py:272-284 nonneg + dum_nocheck branch — `setitem` on a
    /// `FixedSizeListRepr` lowers to a `direct_call` of
    /// `ll_fixed_setitem_fast` (a single `setarrayitem` on the
    /// `Ptr(GcArray)` receiver), preceded by `hop.exception_is_here()`.
    #[test]
    fn fixed_size_list_setitem_nonneg_emits_direct_call_to_ll_fixed_setitem_fast() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();
        // The third inputarg converts to `item_repr`; `convertvar`
        // short-circuits on `std::ptr::eq`, so `args_r[2]` must be the
        // exact stored `item_repr` instance.
        let item_repr = list_repr.item_repr.clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_item = Variable::new();
        v_item.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "setitem".to_string(),
                vec![
                    Hlvalue::Variable(v_list),
                    Hlvalue::Variable(v_idx),
                    Hlvalue::Variable(v_item),
                ],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ true,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ true, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
            Some(item_repr),
        ]);

        let result = list_repr
            .rtype_setitem(&hop)
            .unwrap_or_else(|err| panic!("list setitem nonneg: {err:?}"));
        assert!(result.is_some());
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_setitem must call hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_fixed_setitem_fast"),
            "expected 'll_fixed_setitem_fast' in {dbg}"
        );
    }

    /// Negative-index annotation (`args_s[1].nonneg == false`) is not yet
    /// ported (the `ll_setitem` neg-fix branch); like the `checkidx`
    /// (implicit-IndexError) branch it surfaces a `TyperError` so the
    /// subject stays on the legacy walker. Rust slice indexing never
    /// produces a negative index.
    #[test]
    fn fixed_size_list_setitem_negative_index_is_deferred() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let list_lltype = list_repr.lowleveltype().clone();
        let item_repr = list_repr.item_repr.clone();
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_idx = Variable::new();
        v_idx.set_concretetype(Some(LowLevelType::Signed));
        let v_item = Variable::new();
        v_item.set_concretetype(Some(LowLevelType::Signed));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "setitem".to_string(),
                vec![
                    Hlvalue::Variable(v_list),
                    Hlvalue::Variable(v_idx),
                    Hlvalue::Variable(v_item),
                ],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ true,
                /* resized */ false,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ false, false)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
            Some(item_repr),
        ]);
        assert!(list_repr.rtype_setitem(&hop).is_err());
    }
}
