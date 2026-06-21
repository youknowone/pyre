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
    /// produces them.
    ///
    /// The upstream result `recast` (`rlist.py:266`
    /// `return r_lst.recast(hop.llops, v_res)` → `convertvar(v, item_repr,
    /// external_item_repr)`) is omitted: `FixedSizeListRepr::new` keeps only
    /// the internal `item_repr` and drops the `external_item_repr` half of
    /// `externalvsinternal`, so getitem returns the internal repr directly.
    /// That is correct for every list a live subject builds today — a
    /// primitive item has `external == internal`, making recast an identity —
    /// but a GC-instance list (`external != internal`) would return the
    /// internal/root repr instead of the concrete external repr. Deferred to
    /// the #305 slice that models `external_item_repr`: pyre's `convertvar`
    /// keys identity on `Arc::ptr_eq`, so a same-lltype-different-`Arc`
    /// recast added now would `TyperError` every currently-green getitem.
    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::annotator::model::SomeValue;
        if hop.has_implicit_exception("IndexError") {
            return Err(TyperError::message(
                "list rtype_getitem: checkidx IndexError branch not yet ported",
            ));
        }
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

    /// RPython `AbstractBaseListRepr.rtype_method_reverse(self, hop)`
    /// (`rlist.py:138-143`):
    ///
    /// ```python
    /// def rtype_method_reverse(self, hop):
    ///     v_lst, = hop.inputargs(self)
    ///     hop.exception_cannot_occur()
    ///     hop.gendirectcall(ll_reverse, v_lst)
    /// ```
    ///
    /// `ll_reverse` (`rlist.py:677-686`) is an in-place swap loop over the
    /// `FixedSizeListRepr` receiver (the bare `Ptr(GcArray)`): it reads both
    /// endpoints, writes them crossed, and walks `i` up / `length_1_i` down
    /// toward the middle. The lowered body is the multi-block CFG built by
    /// [`build_ll_reverse_helper_graph`]. `reverse` returns `None` (void).
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        match method_name {
            "reverse" => {
                let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
                hop.exception_cannot_occur()?;
                let item_lltype = self.item_repr.lowleveltype().clone();
                let ptr_lltype = self.lltype.clone();
                let ptr_for_builder = ptr_lltype.clone();
                let item_for_builder = item_lltype.clone();
                let helper = hop.rtyper.lowlevel_helper_function_with_builder(
                    "ll_reverse".to_string(),
                    vec![ptr_lltype],
                    LowLevelType::Void,
                    move |_rtyper, _args, _result| {
                        build_ll_reverse_helper_graph(
                            "ll_reverse",
                            ptr_for_builder.clone(),
                            item_for_builder.clone(),
                        )
                    },
                )?;
                hop.gendirectcall(&helper, vlist)
            }
            _ => Err(TyperError::message(format!(
                "missing FixedSizeListRepr.rtype_method_{method_name}"
            ))),
        }
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

    /// RPython `pair(AbstractBaseListRepr, IntegerRepr).rtype_getitem`
    /// (`rlist.py:247-267`) for the resized list. Shares the dispatch with
    /// [`FixedSizeListRepr::rtype_getitem`] (nonneg + `dum_nocheck` fast
    /// path, `ll_getitem_nonneg` → `l.ll_getitem_fast(index)`), but the
    /// resized receiver is the `Ptr(GcStruct("list", length, items))`
    /// header, so `ll_getitem_fast` (`lltypesystem/rlist.py:259-262`) reads
    /// the `items` array out of the struct first
    /// (`l.ll_items()[index]` = `getfield(l, "items")` then
    /// `getarrayitem`). The negative-index (`ll_getitem`) and `checkidx`
    /// (IndexError-raising) branches surface a `TyperError` until those
    /// helpers land.
    ///
    /// The upstream result `recast` (`rlist.py:266`) is omitted for the
    /// same reason as `FixedSizeListRepr`: `ListRepr::new` keeps only the
    /// internal `item_repr`, an identity recast for the primitive items a
    /// live subject builds today. A GC-instance list (`external !=
    /// internal`) is deferred to the #305 `external_item_repr` slice.
    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::annotator::model::SomeValue;
        if hop.has_implicit_exception("IndexError") {
            return Err(TyperError::message(
                "list rtype_getitem: checkidx IndexError branch not yet ported",
            ));
        }
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
            "ll_getitem_fast".to_string(),
            vec![ptr_lltype, LowLevelType::Signed],
            item_lltype,
            move |_rtyper, _args, _result| {
                build_ll_getitem_fast_helper_graph(
                    "ll_getitem_fast",
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, args)
    }

    /// RPython `pair(AbstractBaseListRepr, IntegerRepr).rtype_setitem`
    /// (`rlist.py:272-284`) for the resized list. Shares the dispatch with
    /// [`FixedSizeListRepr::rtype_setitem`] (nonneg + `dum_nocheck`,
    /// `ll_setitem_nonneg` → `l.ll_setitem_fast(index, item)`), but the
    /// resized receiver reads the `items` array out of the
    /// `Ptr(GcStruct("list", length, items))` header first
    /// (`lltypesystem/rlist.py:264-267` `l.ll_items()[index] = item` =
    /// `getfield(l, "items")` then `setarrayitem`). The `checkidx`
    /// (implicit-IndexError) and negative-index (`ll_setitem`) branches
    /// surface a `TyperError` until those helpers land.
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
            "ll_setitem_fast".to_string(),
            vec![ptr_lltype, LowLevelType::Signed, item_lltype],
            LowLevelType::Void,
            move |_rtyper, _args, _result| {
                build_ll_setitem_fast_helper_graph(
                    "ll_setitem_fast",
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, args)
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

/// Synthesise the resized-list `ll_getitem_fast`
/// (`lltypesystem/rlist.py:259-262`):
///
/// ```python
/// def ll_getitem_fast(l, index):
///     ll_assert(index < l.ll_length(), "getitem out of bounds")
///     return l.ll_items()[index]
/// ```
///
/// The `ll_assert` is a debug-only bound check (no production op). Unlike
/// [`build_ll_fixed_getitem_fast_helper_graph`] (whose receiver IS the
/// `Ptr(GcArray)`), the resized receiver is the
/// `Ptr(GcStruct("list", length, items))` header, so the body reads the
/// `items` array out of the struct first: `getfield(l, "items")` →
/// `getarrayitem(items, index) -> ITEM`.
pub(crate) fn build_ll_getitem_fast_helper_graph(
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

    let items_ptr_lltype = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(ArrayType::gc(item_lltype.clone())),
    }));
    let v_items = variable_with_lltype("items", items_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l_arg), void_field_const("items")],
        Hlvalue::Variable(v_items.clone()),
    ));
    let v_item = variable_with_lltype("item", item_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarrayitem",
        vec![Hlvalue::Variable(v_items), Hlvalue::Variable(index_arg)],
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

/// Synthesise the resized-list `ll_setitem_fast`
/// (`lltypesystem/rlist.py:264-267`):
///
/// ```python
/// def ll_setitem_fast(l, index, item):
///     ll_assert(index < l.ll_length(), "setitem out of bounds")
///     l.ll_items()[index] = item
/// ```
///
/// The `ll_assert` is a debug-only bound check (no production op). The
/// resized receiver reads the `items` array out of the
/// `Ptr(GcStruct("list", length, items))` header first:
/// `getfield(l, "items")` → `setarrayitem(items, index, item)`. The
/// function falls off the end (`return None`), so the returnblock receives
/// the `Void` `None` constant.
pub(crate) fn build_ll_setitem_fast_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let l_arg = variable_with_lltype("l", ptr_lltype);
    let index_arg = variable_with_lltype("index", LowLevelType::Signed);
    let item_arg = variable_with_lltype("item", item_lltype.clone());
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

    let items_ptr_lltype = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(ArrayType::gc(item_lltype)),
    }));
    let v_items = variable_with_lltype("items", items_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l_arg), void_field_const("items")],
        Hlvalue::Variable(v_items.clone()),
    ));
    let v_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(v_items),
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

/// Synthesise `ll_reverse` (`rlist.py:677-686`):
///
/// ```python
/// def ll_reverse(l):
///     length = l.ll_length()
///     i = 0
///     length_1_i = length-1-i
///     while i < length_1_i:
///         tmp = l.ll_getitem_fast(i)
///         l.ll_setitem_fast(i, l.ll_getitem_fast(length_1_i))
///         l.ll_setitem_fast(length_1_i, tmp)
///         i += 1
///         length_1_i -= 1
/// ```
///
/// In-place swap loop over the `FixedSizeListRepr` receiver (the bare
/// `Ptr(GcArray)`): `ll_length` / `ll_getitem_fast` / `ll_setitem_fast` each
/// collapse to a single `getarraysize` / `getarrayitem` / `setarrayitem` op
/// (no `items` header indirection, unlike the resized list). Four-block CFG:
/// - **startblock**: `getarraysize(l) -> length`, `int_sub(length, 1) ->
///   length_1_i` (the `length - 1 - i` initial folds `- i` against the
///   loop-seeding constant `i = 0`); links to `block_loop_cond(l, 0,
///   length_1_i)`.
/// - **block_loop_cond**: `int_lt(i, length_1_i) -> cond`. `True` →
///   `block_loop_body`; `False` → returnblock (`None`). The strict `<` leaves
///   an odd-length list's middle element in place.
/// - **block_loop_body**: read BOTH endpoints before writing either (so the
///   swap captures pre-swap values) — `getarrayitem(l, i) -> tmp`,
///   `getarrayitem(l, length_1_i) -> v`, `setarrayitem(l, i, v)`,
///   `setarrayitem(l, length_1_i, tmp)` — then `int_add(i, 1)`,
///   `int_sub(length_1_i, 1)`, back to `block_loop_cond`.
pub(crate) fn build_ll_reverse_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let signed_const = |n: i64| {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(n),
            LowLevelType::Signed,
        ))
    };
    let bool_const = |b: bool| {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Bool(b),
            LowLevelType::Bool,
        ))
    };

    let l_arg = variable_with_lltype("l", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(l_arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // The loop blocks carry (l, i, length_1_i) as their own fresh inputargs.
    let l_cond = variable_with_lltype("l", ptr_lltype.clone());
    let i_cond = variable_with_lltype("i", LowLevelType::Signed);
    let j_cond = variable_with_lltype("length_1_i", LowLevelType::Signed);
    let block_loop_cond = Block::shared(vec![
        Hlvalue::Variable(l_cond.clone()),
        Hlvalue::Variable(i_cond.clone()),
        Hlvalue::Variable(j_cond.clone()),
    ]);

    let l_body = variable_with_lltype("l", ptr_lltype);
    let i_body = variable_with_lltype("i", LowLevelType::Signed);
    let j_body = variable_with_lltype("length_1_i", LowLevelType::Signed);
    let block_loop_body = Block::shared(vec![
        Hlvalue::Variable(l_body.clone()),
        Hlvalue::Variable(i_body.clone()),
        Hlvalue::Variable(j_body.clone()),
    ]);

    // ---- startblock: length = getarraysize(l); length_1_i = length - 1.
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(l_arg.clone())],
        Hlvalue::Variable(length.clone()),
    ));
    let length_1_i = variable_with_lltype("length_1_i", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_sub",
        vec![Hlvalue::Variable(length), signed_const(1)],
        Hlvalue::Variable(length_1_i.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_arg),
                signed_const(0),
                Hlvalue::Variable(length_1_i),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- block_loop_cond: int_lt(i, length_1_i). True -> body; False -> return None.
    let cond = variable_with_lltype("cond", LowLevelType::Bool);
    block_loop_cond
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(i_cond.clone()),
                Hlvalue::Variable(j_cond.clone()),
            ],
            Hlvalue::Variable(cond.clone()),
        ));
    block_loop_cond.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
    let none_const = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_cond),
                Hlvalue::Variable(i_cond),
                Hlvalue::Variable(j_cond),
            ],
            Some(block_loop_body.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![none_const],
            Some(graph.returnblock.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // ---- block_loop_body: read both endpoints, write them crossed, step indices.
    let tmp = variable_with_lltype("tmp", item_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(l_body.clone()),
                Hlvalue::Variable(i_body.clone()),
            ],
            Hlvalue::Variable(tmp.clone()),
        ));
    let v = variable_with_lltype("v", item_lltype);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(l_body.clone()),
                Hlvalue::Variable(j_body.clone()),
            ],
            Hlvalue::Variable(v.clone()),
        ));
    let w_i = variable_with_lltype("v", LowLevelType::Void);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(l_body.clone()),
                Hlvalue::Variable(i_body.clone()),
                Hlvalue::Variable(v),
            ],
            Hlvalue::Variable(w_i),
        ));
    let w_j = variable_with_lltype("v", LowLevelType::Void);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(l_body.clone()),
                Hlvalue::Variable(j_body.clone()),
                Hlvalue::Variable(tmp),
            ],
            Hlvalue::Variable(w_j),
        ));
    let i_next = variable_with_lltype("i", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_body.clone()), signed_const(1)],
            Hlvalue::Variable(i_next.clone()),
        ));
    let j_next = variable_with_lltype("length_1_i", LowLevelType::Signed);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(j_body.clone()), signed_const(1)],
            Hlvalue::Variable(j_next.clone()),
        ));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_body),
                Hlvalue::Variable(i_next),
                Hlvalue::Variable(j_next),
            ],
            Some(block_loop_cond.clone()),
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

    /// A caught `IndexError` (`hop.has_implicit_exception("IndexError")`)
    /// requires the `checkidx` path (`ll_getitem`), which is not yet
    /// ported — `rtype_getitem` must surface a `TyperError` rather than
    /// silently dropping the bounds check via the `dum_nocheck` fast path.
    /// Mirrors the `rtype_setitem` checkidx guard.
    #[test]
    fn fixed_size_list_getitem_checkidx_indexerror_is_deferred() {
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
        // A caught IndexError exitcase on this getitem.
        let exitblock = std::rc::Rc::new(std::cell::RefCell::new(Block::new(vec![])));
        let cls_index = crate::flowspace::model::HOST_ENV
            .lookup_exception_class("IndexError")
            .expect("IndexError class");
        let link_index = std::rc::Rc::new(std::cell::RefCell::new(Link::new(
            vec![],
            Some(exitblock),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                cls_index,
            )))),
        )));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_list), Hlvalue::Variable(v_idx)],
                Hlvalue::Variable(v_result),
            ),
            vec![link_index],
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
        let err = list_repr
            .rtype_getitem(&hop)
            .expect_err("checkidx IndexError must defer to a TyperError");
        assert!(
            format!("{err}").contains("checkidx IndexError"),
            "expected checkidx IndexError deferral message, got {err}"
        );
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

    /// rlist.py:247-267 nonneg + checkidx=False branch — `getitem` on a
    /// resized `ListRepr` lowers to a `direct_call` of `ll_getitem_fast`
    /// (the resized helper, distinct from `FixedSizeListRepr`'s
    /// `ll_fixed_getitem_fast`), preceded by `hop.exception_cannot_occur()`.
    #[test]
    fn resized_list_getitem_nonneg_emits_direct_call_to_ll_getitem_fast() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let list_repr: Arc<ListRepr> = Arc::new(
            ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new"),
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
                /* resized */ true,
            ))),
            SomeValue::Integer(SomeInteger::new(/* nonneg */ true, false)),
        ]);
        hop.args_r.borrow_mut().extend([
            Some(list_repr.clone() as Arc<dyn Repr>),
            Some(signed_repr() as Arc<dyn Repr>),
        ]);

        let result = list_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("resized list getitem nonneg: {err:?}"));
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
            dbg.contains("ll_getitem_fast"),
            "expected 'll_getitem_fast' in {dbg}"
        );
    }

    /// rlist.py:272-284 nonneg + dum_nocheck branch — `setitem` on a
    /// resized `ListRepr` lowers to a `direct_call` of `ll_setitem_fast`
    /// (the resized helper, distinct from `FixedSizeListRepr`'s
    /// `ll_fixed_setitem_fast`), preceded by `hop.exception_is_here()`.
    #[test]
    fn resized_list_setitem_nonneg_emits_direct_call_to_ll_setitem_fast() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let list_repr: Arc<ListRepr> = Arc::new(
            ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new"),
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
                /* resized */ true,
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
            .unwrap_or_else(|err| panic!("resized list setitem nonneg: {err:?}"));
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
            dbg.contains("ll_setitem_fast"),
            "expected 'll_setitem_fast' in {dbg}"
        );
    }

    /// The resized `ll_getitem_fast` helper body reads the `items` array
    /// out of the `length`/`items` header struct before indexing:
    /// `getfield(l, "items")` → `getarrayitem(items, index)` (vs
    /// `FixedSizeListRepr`'s single `getarrayitem` on the bare array).
    #[test]
    fn build_ll_getitem_fast_helper_reads_items_then_getarrayitem() {
        let rtyper = fresh_rtyper();
        let repr = ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new");
        let pygraph = build_ll_getitem_fast_helper_graph(
            "ll_getitem_fast",
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_getitem_fast_helper_graph");
        let graph = pygraph.graph.borrow();
        let ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(ops, vec!["getfield", "getarrayitem"]);
    }

    /// The resized `ll_setitem_fast` helper body reads the `items` array
    /// out of the `length`/`items` header struct before storing:
    /// `getfield(l, "items")` → `setarrayitem(items, index, item)`.
    #[test]
    fn build_ll_setitem_fast_helper_reads_items_then_setarrayitem() {
        let rtyper = fresh_rtyper();
        let repr = ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new");
        let pygraph = build_ll_setitem_fast_helper_graph(
            "ll_setitem_fast",
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_setitem_fast_helper_graph");
        let graph = pygraph.graph.borrow();
        let ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(ops, vec!["getfield", "setarrayitem"]);
    }

    /// `slice.reverse()` rtypes through `rtype_method("reverse")` to a
    /// `direct_call(ll_reverse, v_lst)` (`rlist.py:138-143`); `reverse`
    /// returns `None` (void), and the path calls
    /// `hop.exception_cannot_occur()`.
    #[test]
    fn fixed_size_list_reverse_emits_direct_call_to_ll_reverse() {
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
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                vec![Hlvalue::Variable(v_list)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .extend([SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                /* mutated */ true,
                /* resized */ false,
            )))]);
        hop.args_r
            .borrow_mut()
            .extend([Some(list_repr.clone() as Arc<dyn Repr>)]);

        let result = list_repr
            .rtype_method("reverse", &hop)
            .unwrap_or_else(|err| panic!("list reverse: {err:?}"));
        assert!(result.is_some());
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_method_reverse must call hop.exception_cannot_occur()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(dbg.contains("ll_reverse"), "expected 'll_reverse' in {dbg}");
    }

    /// An unknown method name surfaces a `TyperError` so the subject stays
    /// on the legacy walker rather than miscompiling.
    #[test]
    fn fixed_size_list_unknown_method_is_deferred() {
        let rtyper = fresh_rtyper_live();
        let list_repr: Arc<FixedSizeListRepr> = Arc::new(
            FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
                .expect("FixedSizeListRepr::new"),
        );
        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_repr.lowleveltype().clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Void));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "simple_call".to_string(),
                vec![Hlvalue::Variable(v_list)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops,
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_r
            .borrow_mut()
            .extend([Some(list_repr.clone() as Arc<dyn Repr>)]);
        assert!(list_repr.rtype_method("sort", &hop).is_err());
    }

    /// The `ll_reverse` helper is a four-block swap loop: `startblock`
    /// (`getarraysize` + `int_sub`) → `block_loop_cond` (`int_lt`) →
    /// `block_loop_body` (two `getarrayitem` reads BEFORE two `setarrayitem`
    /// writes, then `int_add` / `int_sub`) → back to the cond block.
    #[test]
    fn build_ll_reverse_helper_has_swap_loop_blocks() {
        let rtyper = fresh_rtyper();
        let repr = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let pygraph = build_ll_reverse_helper_graph(
            "ll_reverse",
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_reverse_helper_graph");
        let graph = pygraph.graph.borrow();
        let block_op_seqs: Vec<Vec<String>> = graph
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect()
            })
            .collect();
        // The startblock signature distinguishes reverse from get/setitem.
        assert!(
            block_op_seqs.contains(&vec!["getarraysize".to_string(), "int_sub".to_string()]),
            "startblock must be getarraysize + int_sub, got {block_op_seqs:?}"
        );
        // The loop-condition block.
        assert!(
            block_op_seqs.contains(&vec!["int_lt".to_string()]),
            "expected an int_lt condition block, got {block_op_seqs:?}"
        );
        // The swap body: both reads precede both writes, then the two steps.
        assert!(
            block_op_seqs.contains(&vec![
                "getarrayitem".to_string(),
                "getarrayitem".to_string(),
                "setarrayitem".to_string(),
                "setarrayitem".to_string(),
                "int_add".to_string(),
                "int_sub".to_string(),
            ]),
            "expected the read-both-before-write swap body, got {block_op_seqs:?}"
        );
    }
}
