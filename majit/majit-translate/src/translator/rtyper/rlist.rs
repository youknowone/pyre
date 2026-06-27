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
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    Array, LowLevelType, Ptr, PtrTarget, Struct,
};
use crate::translator::rtyper::lltypesystem::rstr::sub_helper_funcptr_constant;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, HighLevelOp, LowLevelFunction, RPythonTyper, constant_with_lltype, exception_args,
    helper_pygraph_from_graph, variable_with_lltype, void_field_const,
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
    /// `self.external_item_repr` (`lltypesystem/rlist.py:177`) — the
    /// external element repr `recast` ([`list_recast`]) converts a getitem
    /// result back to. For primitive items this is a clone of the same
    /// cached `Arc` as `item_repr`, so the recast short-circuits to
    /// identity; for a gc `InstanceRepr` it is the concrete repr while
    /// `item_repr` is the generic `OBJECTPTR` root.
    external_item_repr: Arc<dyn Repr>,
}

impl FixedSizeListRepr {
    pub fn new(rtyper: &Rc<RPythonTyper>, item_repr: Arc<dyn Repr>) -> Result<Self, TyperError> {
        // `externalvsinternal(rtyper, item_repr, gcref=True)` —
        // gc `InstanceRepr` items become the generic `Ptr(OBJECT)`
        // gcref so the array element type is never a gc container
        // (which `Array::gc` rejects); non-instance reprs pass
        // through unchanged.
        let (external_item_repr, internal) =
            crate::translator::rtyper::rclass::externalvsinternal(rtyper, item_repr, true)?;
        let item_lltype = internal.lowleveltype().clone();
        let arr = Array::gc(item_lltype);
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(arr),
        }));
        Ok(FixedSizeListRepr {
            state: ReprState::new(),
            lltype,
            item_repr: internal,
            external_item_repr,
        })
    }
}

/// RPython `AbstractBaseListRepr.recast(self, llops, v)` (`rlist.py:67-68`):
///
/// ```python
/// def recast(self, llops, v):
///     return llops.convertvar(v, self.item_repr, self.external_item_repr)
/// ```
///
/// Converts a `getitem` result from the internal element repr (the array
/// element type) back to the external repr the caller annotated. For the
/// primitive items a live subject builds, `item_repr` and
/// `external_item_repr` are clones of the same cached `Arc`, so `convertvar`
/// short-circuits to identity (its `ptr::eq` guard) and emits no op. A
/// gc-instance element list (`external != internal`) downcasts the generic
/// `OBJECTPTR` internal repr to the concrete external `InstanceRepr` via the
/// `cast_pointer` `pair(InstanceRepr, InstanceRepr)` arm.
fn list_recast(
    hop: &HighLevelOp,
    item_repr: &Arc<dyn Repr>,
    external_item_repr: &Arc<dyn Repr>,
    v: Hlvalue,
) -> Result<Hlvalue, TyperError> {
    hop.llops
        .borrow_mut()
        .convertvar(v, item_repr.as_ref(), external_item_repr.as_ref())
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
    /// (`&[T]`) or fixed array. The `(checkidx, nonneg)` selection is
    /// shared with the resized list through [`list_rtype_getitem`]: the
    /// nonneg + `dum_nocheck` fast path collapses through
    /// `ll_getitem_foldable_nonneg` → `ll_fixed_getitem_fast(l, index)` →
    /// `l[index]` (`lltypesystem/rlist.py:402-405`) to the bare
    /// `getarrayitem` on the `Ptr(GcArray)` receiver, while the
    /// negative-index (`ll_fixed_getitem`) and `checkidx`
    /// (IndexError-raising `ll_fixed_getitem_*_checked`) helpers fold / window
    /// the index before dispatching to that fast helper. Rust slice indexing
    /// only ever exercises the nonneg + `dum_nocheck` path.
    ///
    /// The upstream result `recast` (`rlist.py:266`
    /// `return r_lst.recast(hop.llops, v_res)`) is applied via
    /// [`list_recast`]: `convertvar(v_res, item_repr, external_item_repr)`.
    /// For the primitive items a live subject builds `external == internal`
    /// (clones of the same cached `Arc`), so the recast short-circuits to
    /// identity and emits no op; a GC-instance list downcasts the internal
    /// root repr to the concrete external repr.
    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_res = list_rtype_getitem(
            hop,
            self,
            ListLayout::Fixed,
            &self.lltype,
            self.item_repr.lowleveltype(),
        )?;
        list_recast(hop, &self.item_repr, &self.external_item_repr, v_res).map(Some)
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
    /// Shares the `(checkidx, nonneg)` selection with the resized list via
    /// [`list_rtype_setitem`]. The `dum_nocheck` + nonneg fast path
    /// (`ll_setitem_nonneg(dum_nocheck, l, index, item)`, no IndexError
    /// branch — `index >= 0` is a debug `ll_assert`) collapses through
    /// `l.ll_setitem_fast` → `ll_fixed_setitem_fast(l, index, item)` →
    /// `l[index] = item` (`lltypesystem/rlist.py:407-410`) to the bare
    /// `setarrayitem` on the `Ptr(GcArray)` receiver; the negative-index
    /// (`ll_fixed_setitem`) and `checkidx` (IndexError-raising
    /// `ll_fixed_setitem_*_checked`) helpers fold / window the index first.
    /// The third inputarg converts to the internal `item_repr` (the
    /// gcref-wrapped element repr), so `rtype_setitem` does not `recast`.
    fn rtype_setitem(&self, hop: &HighLevelOp) -> RTypeResult {
        list_rtype_setitem(
            hop,
            self,
            ListLayout::Fixed,
            &self.lltype,
            self.item_repr.lowleveltype(),
            self.item_repr.as_ref(),
        )
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

    /// RPython `lltypesystem/rlist.py` `make_iterator_repr` on the
    /// `AbstractBaseListRepr`: the no-variant case mints
    /// `ListIteratorRepr(self)`; the `("reversed",)` variant
    /// (`ReversedListIteratorRepr`) is deferred.
    fn make_iterator_repr(
        &self,
        variant: &[String],
        foldable: bool,
    ) -> Result<Arc<dyn Repr>, TyperError> {
        if !variant.is_empty() {
            return Err(TyperError::missing_rtype_operation(
                "FixedSizeListRepr.make_iterator_repr: non-default variant \
                 (reversed) deferred",
            ));
        }
        Ok(Arc::new(ListIteratorRepr::new(
            self.lltype.clone(),
            self.item_repr.clone(),
            self.external_item_repr.clone(),
            true,
            foldable,
        )?))
    }
}

/// RPython `class AbstractListIteratorRepr(IteratorRepr)` (`rlist.py:437`).
///
/// Concrete list iterator layouts live in the lltypesystem-specific module.
/// The generic iterator lowering methods are still deferred.
#[derive(Debug, Default)]
pub struct AbstractListIteratorRepr;

fn rlist_runtime_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!("rlist.{name} — list helper deferred"))
}

pub fn rtype_newlist() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("rtype_newlist"))
}

pub fn rtype_alloc_and_set() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("rtype_alloc_and_set"))
}

pub fn _ll_zero_or_null() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("_ll_zero_or_null"))
}

pub fn _null_of_type() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("_null_of_type"))
}

pub fn ll_alloc_and_set() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_alloc_and_set"))
}

pub fn _ll_alloc_and_set_nojit() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("_ll_alloc_and_set_nojit"))
}

pub fn _ll_alloc_and_set_jit() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("_ll_alloc_and_set_jit"))
}

pub fn _ll_alloc_and_clear() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("_ll_alloc_and_clear"))
}

pub fn _ll_alloc_and_set_nonnull() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("_ll_alloc_and_set_nonnull"))
}

pub fn ll_null_item() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_null_item"))
}

#[allow(non_snake_case)]
pub fn listItemType() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("listItemType"))
}

pub fn ll_arraycopy() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_arraycopy"))
}

pub fn ll_arraymove() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_arraymove"))
}

pub fn ll_copy() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_copy"))
}

pub fn ll_list_is_true() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_list_is_true"))
}

pub fn ll_list_is_true_foldable() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_list_is_true_foldable"))
}

pub fn ll_append() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_append"))
}

pub fn ll_prepend() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_prepend"))
}

pub fn ll_concat() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_concat"))
}

pub fn ll_insert_nonneg() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_insert_nonneg"))
}

pub fn ll_pop_nonneg() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_pop_nonneg"))
}

pub fn ll_pop_default() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_pop_default"))
}

pub fn ll_pop_zero() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_pop_zero"))
}

pub fn ll_pop() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_pop"))
}

pub fn ll_delitem_nonneg() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_delitem_nonneg"))
}

pub fn ll_delitem() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_delitem"))
}

pub fn ll_extend() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend"))
}

pub fn ll_extend_with_str() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend_with_str"))
}

pub fn ll_extend_with_str_slice_startonly() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend_with_str_slice_startonly"))
}

pub fn ll_extend_with_str_slice_startstop() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend_with_str_slice_startstop"))
}

pub fn ll_extend_with_str_slice_minusone() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend_with_str_slice_minusone"))
}

pub fn ll_extend_with_char_count() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_extend_with_char_count"))
}

pub fn ll_listslice_startonly() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listslice_startonly"))
}

pub fn ll_listslice_startstop() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listslice_startstop"))
}

pub fn ll_listslice_minusone() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listslice_minusone"))
}

pub fn ll_listdelslice_startonly() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listdelslice_startonly"))
}

pub fn ll_listdelslice_startstop() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listdelslice_startstop"))
}

pub fn ll_listsetslice() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listsetslice"))
}

pub fn listeq_unroll_case() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("listeq_unroll_case"))
}

pub fn ll_listeq() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listeq"))
}

pub fn ll_listcontains() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listcontains"))
}

pub fn ll_listindex() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listindex"))
}

pub fn ll_listremove() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_listremove"))
}

pub fn ll_inplace_mul() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_inplace_mul"))
}

pub fn ll_mul() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_mul"))
}

pub fn ll_mul_loop() -> Result<(), TyperError> {
    Err(rlist_runtime_deferred("ll_mul_loop"))
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
    /// `self.external_item_repr` (`lltypesystem/rlist.py:110`) — the
    /// external element repr `recast` ([`list_recast`]) converts a getitem
    /// result back to (a clone of `item_repr` for primitives, the concrete
    /// `InstanceRepr` for a gc-instance element list).
    external_item_repr: Arc<dyn Repr>,
}

impl ListRepr {
    pub fn new(rtyper: &Rc<RPythonTyper>, item_repr: Arc<dyn Repr>) -> Result<Self, TyperError> {
        // `externalvsinternal(rtyper, item_repr, gcref=True)` — same
        // gcref normalisation as `FixedSizeListRepr`: gc `InstanceRepr`
        // items become the generic `Ptr(OBJECT)` gcref so the array
        // element type is never a gc container.
        let (external_item_repr, internal) =
            crate::translator::rtyper::rclass::externalvsinternal(rtyper, item_repr, true)?;
        let item_lltype = internal.lowleveltype().clone();
        // upstream `get_itemarray_lowleveltype()` — `GcArray(ITEM)` (the
        // `ADTIFixedList` adtmeths it carries are unused until the array
        // ops land, so the bare array suffices for this slice).
        let itemarray = Array::gc(item_lltype);
        let items_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(itemarray),
        }));
        // upstream `GcStruct("list", ("length", Signed), ("items",
        // Ptr(ITEMARRAY)), hints={'list': True})`.
        let list_struct = Struct::gc_with_hints(
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
            external_item_repr,
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
    /// (IndexError-raising `ll_getitem_*_checked`) helpers fold / window the
    /// index before dispatching to that fast helper — all selected by the
    /// shared [`list_rtype_getitem`].
    ///
    /// The upstream result `recast` (`rlist.py:266`) is applied via
    /// [`list_recast`] — identical to `FixedSizeListRepr`: an identity
    /// short-circuit for the primitive items a live subject builds, a
    /// `cast_pointer` downcast for a gc-instance element list.
    fn rtype_getitem(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_res = list_rtype_getitem(
            hop,
            self,
            ListLayout::Resized,
            &self.lltype,
            self.item_repr.lowleveltype(),
        )?;
        list_recast(hop, &self.item_repr, &self.external_item_repr, v_res).map(Some)
    }

    /// RPython `pair(AbstractBaseListRepr, IntegerRepr).rtype_setitem`
    /// (`rlist.py:272-284`) for the resized list. Shares the dispatch with
    /// [`FixedSizeListRepr::rtype_setitem`] (nonneg + `dum_nocheck`,
    /// `ll_setitem_nonneg` → `l.ll_setitem_fast(index, item)`), but the
    /// resized receiver reads the `items` array out of the
    /// `Ptr(GcStruct("list", length, items))` header first
    /// (`lltypesystem/rlist.py:264-267` `l.ll_items()[index] = item` =
    /// `getfield(l, "items")` then `setarrayitem`). The negative-index
    /// (`ll_setitem`) and `checkidx` (IndexError-raising
    /// `ll_setitem_*_checked`) helpers fold / window the index before
    /// dispatching to that fast helper — all selected by the shared
    /// [`list_rtype_setitem`].
    fn rtype_setitem(&self, hop: &HighLevelOp) -> RTypeResult {
        list_rtype_setitem(
            hop,
            self,
            ListLayout::Resized,
            &self.lltype,
            self.item_repr.lowleveltype(),
            self.item_repr.as_ref(),
        )
    }

    /// RPython `AbstractBaseListRepr.rtype_method_reverse(self, hop)`
    /// (`rlist.py:138-143`) — defined on the common base, so it applies to
    /// the resized [`ListRepr`] as well as [`FixedSizeListRepr`]. The body
    /// shape is identical (`inputargs(self)`, `exception_cannot_occur()`,
    /// `direct_call(ll_reverse, v_lst)`); only the lowered `ll_reverse`
    /// differs — the resized receiver reads `length`/`items` out of the
    /// `Ptr(GcStruct("list", length, items))` header
    /// ([`build_ll_reverse_resized_helper_graph`]) rather than `getarraysize`
    /// / bare `getarrayitem` on a `Ptr(GcArray)`.
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
                        build_ll_reverse_resized_helper_graph(
                            "ll_reverse",
                            ptr_for_builder.clone(),
                            item_for_builder.clone(),
                        )
                    },
                )?;
                hop.gendirectcall(&helper, vlist)
            }
            _ => Err(TyperError::message(format!(
                "missing ListRepr.rtype_method_{method_name}"
            ))),
        }
    }

    /// RPython `lltypesystem/rlist.py` `make_iterator_repr` on the
    /// `AbstractBaseListRepr`: the no-variant case mints
    /// `ListIteratorRepr(self)`; the `("reversed",)` variant
    /// (`ReversedListIteratorRepr`) is deferred. The resized receiver is
    /// flagged so `ll_listnext` reads `length` via the struct header.
    fn make_iterator_repr(
        &self,
        variant: &[String],
        foldable: bool,
    ) -> Result<Arc<dyn Repr>, TyperError> {
        if !variant.is_empty() {
            return Err(TyperError::missing_rtype_operation(
                "ListRepr.make_iterator_repr: non-default variant (reversed) deferred",
            ));
        }
        // A resized list is always `mutated`, so `foldable` is false here; it is
        // threaded for signature parity and gated again by `list_is_fixed`.
        Ok(Arc::new(ListIteratorRepr::new(
            self.lltype.clone(),
            self.item_repr.clone(),
            self.external_item_repr.clone(),
            false,
            foldable,
        )?))
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

/// Synthesise `ll_getitem_foldable_nonneg` (`rlist.py:721-724`):
///
/// ```python
/// def ll_getitem_foldable_nonneg(l, index):
///     ll_assert(index >= 0, "unexpectedly negative list getitem index")
///     return l.ll_getitem_fast(index)
/// ll_getitem_foldable_nonneg.oopspec = 'list.getitem_foldable(l, index)'
/// ```
///
/// Identical body to [`build_ll_fixed_getitem_fast_helper_graph`] (the
/// `ll_assert` is a debug-only bound check), except the element read is the
/// FOLDABLE `getarrayitem_pure`. `rtype_getitem` (rlist.py:255-258) selects
/// this helper instead of `ll_fixed_getitem_fast` when `not
/// listitem.mutated`, so the immutable element load can be folded / CSE'd;
/// the `oopspec = 'list.getitem_foldable'` is realised here by the distinct
/// `getarrayitem_pure` opname, which `lloperation` marks `canfold=True`.
pub(crate) fn build_ll_fixed_getitem_fast_foldable_helper_graph(
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
        "getarrayitem_pure",
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
        TO: PtrTarget::Array(Array::gc(item_lltype.clone())),
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
        TO: PtrTarget::Array(Array::gc(item_lltype)),
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

/// Synthesise the resized-list `ll_reverse` (`rlist.py:677-686`):
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
/// Same four-block swap loop as [`build_ll_reverse_helper_graph`], but the
/// resized receiver is the `Ptr(GcStruct("list", length, items))` header, so
/// the `ll_length` / `ll_getitem_fast` / `ll_setitem_fast` adtmeths read the
/// header rather than a bare array: `length` is `getfield(l, "length")` (vs
/// `getarraysize`), and each element access reads the `items` array out of
/// the struct first — `getfield(l, "items")` then `getarrayitem` /
/// `setarrayitem` (matching [`build_ll_getitem_fast_helper_graph`] /
/// [`build_ll_setitem_fast_helper_graph`]). The repeated `items` reads fold
/// in the malloc/CSE passes, mirroring upstream's per-adtmeth `l.ll_items()`.
pub(crate) fn build_ll_reverse_resized_helper_graph(
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
    let items_ptr_lltype = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::gc(item_lltype.clone())),
    }));

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

    // ---- startblock: length = getfield(l, "length"); length_1_i = length - 1.
    let length = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(l_arg.clone()), void_field_const("length")],
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

    // ---- block_loop_body: read BOTH endpoints (each via getfield "items" +
    // getarrayitem) before writing either, then step the indices.
    let items_tmp = variable_with_lltype("items", items_ptr_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(l_body.clone()), void_field_const("items")],
            Hlvalue::Variable(items_tmp.clone()),
        ));
    let tmp = variable_with_lltype("tmp", item_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(items_tmp),
                Hlvalue::Variable(i_body.clone()),
            ],
            Hlvalue::Variable(tmp.clone()),
        ));
    let items_v = variable_with_lltype("items", items_ptr_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(l_body.clone()), void_field_const("items")],
            Hlvalue::Variable(items_v.clone()),
        ));
    let v = variable_with_lltype("v", item_lltype);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getarrayitem",
            vec![
                Hlvalue::Variable(items_v),
                Hlvalue::Variable(j_body.clone()),
            ],
            Hlvalue::Variable(v.clone()),
        ));
    let items_wi = variable_with_lltype("items", items_ptr_lltype.clone());
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(l_body.clone()), void_field_const("items")],
            Hlvalue::Variable(items_wi.clone()),
        ));
    let w_i = variable_with_lltype("v", LowLevelType::Void);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(items_wi),
                Hlvalue::Variable(i_body.clone()),
                Hlvalue::Variable(v),
            ],
            Hlvalue::Variable(w_i),
        ));
    let items_wj = variable_with_lltype("items", items_ptr_lltype);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(l_body.clone()), void_field_const("items")],
            Hlvalue::Variable(items_wj.clone()),
        ));
    let w_j = variable_with_lltype("v", LowLevelType::Void);
    block_loop_body
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "setarrayitem",
            vec![
                Hlvalue::Variable(items_wj),
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

/// `FixedSizeListRepr` (bare `Ptr(GcArray)`) vs the resized `ListRepr`
/// (`Ptr(GcStruct("list", length, items))`) differ only in how `ll_length` /
/// `ll_getitem_fast` / `ll_setitem_fast` reach the element array, so the
/// checked / negative-index `ll_getitem` / `ll_setitem` CFGs
/// (`rlist.py:688-748`) are written once and parameterised by this layout.
#[derive(Clone, Copy)]
enum ListLayout {
    Fixed,
    Resized,
}

impl ListLayout {
    fn getitem_fast_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_getitem_fast",
            ListLayout::Resized => "ll_getitem_fast",
        }
    }
    /// rlist.py:721-724 `ll_getitem_foldable_nonneg` — the foldable
    /// counterpart of `getitem_fast_name`, a DISTINCT function so the
    /// helper cache never serves a foldable graph to a mutated list (or
    /// vice-versa). Only the Fixed layout reaches it (Resized ⟹ mutated).
    fn getitem_fast_foldable_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_getitem_fast_foldable",
            ListLayout::Resized => "ll_getitem_fast_foldable",
        }
    }
    fn setitem_fast_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_setitem_fast",
            ListLayout::Resized => "ll_setitem_fast",
        }
    }
    fn getitem_neg_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_getitem",
            ListLayout::Resized => "ll_getitem",
        }
    }
    fn getitem_nonneg_checked_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_getitem_nonneg_checked",
            ListLayout::Resized => "ll_getitem_nonneg_checked",
        }
    }
    fn getitem_checked_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_getitem_checked",
            ListLayout::Resized => "ll_getitem_checked",
        }
    }
    fn setitem_neg_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_setitem",
            ListLayout::Resized => "ll_setitem",
        }
    }
    fn setitem_nonneg_checked_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_setitem_nonneg_checked",
            ListLayout::Resized => "ll_setitem_nonneg_checked",
        }
    }
    fn setitem_checked_name(self) -> &'static str {
        match self {
            ListLayout::Fixed => "ll_fixed_setitem_checked",
            ListLayout::Resized => "ll_setitem_checked",
        }
    }
}

fn signed_const(n: i64) -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::Int(n),
        LowLevelType::Signed,
    ))
}

fn bool_const(b: bool) -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::Bool(b),
        LowLevelType::Bool,
    ))
}

/// Push the `ll_length` read for `layout` onto `block` and return the Signed
/// length var: `getarraysize(l)` (Fixed, bare `Ptr(GcArray)`) or
/// `getfield(l, "length")` (Resized, struct header).
fn emit_list_length_read(block: &BlockRef, layout: ListLayout, l: &Variable) -> Variable {
    let length = variable_with_lltype("length", LowLevelType::Signed);
    let op = match layout {
        ListLayout::Fixed => SpaceOperation::new(
            "getarraysize",
            vec![Hlvalue::Variable(l.clone())],
            Hlvalue::Variable(length.clone()),
        ),
        ListLayout::Resized => SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(l.clone()), void_field_const("length")],
            Hlvalue::Variable(length.clone()),
        ),
    };
    block.borrow_mut().operations.push(op);
    length
}

/// Build (or retrieve cached) the layout's `ll_*_getitem_fast` sub-helper and
/// return a funcptr `Constant` to `direct_call` it (the `basegetitem` of
/// `rlist.py:266`).
fn list_getitem_fast_funcptr(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    foldable: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<Constant, TyperError> {
    // rlist.py:264-266 — the `basegetitem` passed into every (nonneg×checkidx)
    // wrapper is the foldable `ll_getitem_foldable_nonneg` (rlist.py:721-724)
    // when the list item is not mutated.  Only Fixed has a foldable element
    // load; Resized is always mutated, so `foldable` is never set for it.
    let foldable_fixed = foldable && matches!(layout, ListLayout::Fixed);
    let name = if foldable_fixed {
        layout.getitem_fast_foldable_name().to_string()
    } else {
        layout.getitem_fast_name().to_string()
    };
    let name_owned = name.clone();
    let ptr_for_builder = ptr_lltype.clone();
    let item_for_builder = item_lltype.clone();
    let inner = rtyper.lowlevel_helper_function_with_builder(
        name,
        vec![ptr_lltype, LowLevelType::Signed],
        item_lltype,
        move |_rtyper, _args, _result| match layout {
            ListLayout::Fixed if foldable_fixed => {
                build_ll_fixed_getitem_fast_foldable_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                )
            }
            ListLayout::Fixed => build_ll_fixed_getitem_fast_helper_graph(
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            ListLayout::Resized => build_ll_getitem_fast_helper_graph(
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
        },
    )?;
    sub_helper_funcptr_constant(rtyper, &inner)
}

/// rlist.py:697-714 `ll_getitem` with `func is dum_nocheck` — negative index,
/// no bound check: `if index < 0: index += l.ll_length(); return
/// basegetitem(l, index)`. 3-block CFG (start → block_neg_fix → block_dispatch)
/// forwarding the possibly-fixed index to a `direct_call` of the layout's
/// `ll_*_getitem_fast`.
fn build_ll_list_getitem_neg_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    foldable: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast = list_getitem_fast_funcptr(
        rtyper,
        layout,
        foldable,
        ptr_lltype.clone(),
        item_lltype.clone(),
    )?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_fix = variable_with_lltype("l", ptr_lltype.clone());
    let i_fix = variable_with_lltype("index", LowLevelType::Signed);
    let block_neg_fix = Block::shared(vec![
        Hlvalue::Variable(l_fix.clone()),
        Hlvalue::Variable(i_fix.clone()),
    ]);

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
    ]);

    // ---- start: is_neg = int_lt(index, 0); branch.
    let is_neg = variable_with_lltype("is_neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(i.clone()), signed_const(0)],
        Hlvalue::Variable(is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(l.clone()), Hlvalue::Variable(i.clone())],
            Some(block_neg_fix.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(l), Hlvalue::Variable(i)],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // ---- block_neg_fix: length = <len read>; i_fixed = int_add(index, length).
    let length = emit_list_length_read(&block_neg_fix, layout, &l_fix);
    let i_fixed = variable_with_lltype("index", LowLevelType::Signed);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_fix), Hlvalue::Variable(length)],
            Hlvalue::Variable(i_fixed.clone()),
        ));
    block_neg_fix.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(l_fix), Hlvalue::Variable(i_fixed)],
            Some(block_dispatch.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- block_dispatch: c = direct_call(fast, l, index); return c.
    let c = variable_with_lltype("c", item_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
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

/// rlist.py:688-692 `ll_getitem_nonneg` with `func is dum_checkidx` —
/// nonneg index, bound check: `if index >= l.ll_length(): raise IndexError;
/// return basegetitem(l, index)`. 2-block CFG plus graph.exceptblock.
fn build_ll_list_getitem_nonneg_checked_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    foldable: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast = list_getitem_fast_funcptr(
        rtyper,
        layout,
        foldable,
        ptr_lltype.clone(),
        item_lltype.clone(),
    )?;
    let exc_args = exception_args("IndexError")?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
    ]);

    // ---- start: length = <len read>; oob = int_ge(index, length); branch.
    let length = emit_list_length_read(&startblock, layout, &l);
    let oob = variable_with_lltype("oob", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(i.clone()), Hlvalue::Variable(length)],
        Hlvalue::Variable(oob.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(l), Hlvalue::Variable(i)],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // ---- block_dispatch: c = direct_call(fast, l, index); return c.
    let c = variable_with_lltype("c", item_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
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

/// rlist.py:697-714 `ll_getitem` with `func is dum_checkidx` — the negative
/// index is folded in (`index += length`) then the `0 <= index < length`
/// window is enforced, raising IndexError otherwise. The r_uint window test
/// is lowered to the signed-explicit `index >= length or index < 0` form
/// (matching the `ll_stritem_checked` lowering): 5-block CFG (start →
/// block_neg_fix → block_check_high → block_check_low → block_dispatch) plus
/// graph.exceptblock.
fn build_ll_list_getitem_checked_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    foldable: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast = list_getitem_fast_funcptr(
        rtyper,
        layout,
        foldable,
        ptr_lltype.clone(),
        item_lltype.clone(),
    )?;
    let exc_args = exception_args("IndexError")?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
    ]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_fix = variable_with_lltype("l", ptr_lltype.clone());
    let i_fix_u = variable_with_lltype("index", LowLevelType::Unsigned);
    let len_fix_u = variable_with_lltype("length", LowLevelType::Unsigned);
    let block_fixup = Block::shared(vec![
        Hlvalue::Variable(l_fix.clone()),
        Hlvalue::Variable(i_fix_u.clone()),
        Hlvalue::Variable(len_fix_u.clone()),
    ]);

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
    ]);

    // ---- start: length = <len read>; index_u = cast_int_to_uint(index);
    //      length_u = cast_int_to_uint(length); oob = uint_ge(index_u, length_u);
    //      branch.  The common 0 <= index < length case falls straight through
    //      with no add (`ll_getitem`, rlist.py:699).
    let length = emit_list_length_read(&startblock, layout, &l);
    let i_u = variable_with_lltype("index", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(i.clone())],
        Hlvalue::Variable(i_u.clone()),
    ));
    let len_u = variable_with_lltype("length", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(length)],
        Hlvalue::Variable(len_u.clone()),
    ));
    let oob = variable_with_lltype("oob", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "uint_ge",
        vec![
            Hlvalue::Variable(i_u.clone()),
            Hlvalue::Variable(len_u.clone()),
        ],
        Hlvalue::Variable(oob.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l.clone()),
                Hlvalue::Variable(i_u),
                Hlvalue::Variable(len_u),
            ],
            Some(block_fixup.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(l), Hlvalue::Variable(i)],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // ---- block_fixup: index_u = uint_add(index_u, length_u);
    //      if uint_ge(index_u, length_u): raise IndexError;
    //      index = intmask(index_u); -> dispatch.
    let i_fixed_u = variable_with_lltype("index", LowLevelType::Unsigned);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_add",
            vec![
                Hlvalue::Variable(i_fix_u),
                Hlvalue::Variable(len_fix_u.clone()),
            ],
            Hlvalue::Variable(i_fixed_u.clone()),
        ));
    let oob2 = variable_with_lltype("oob", LowLevelType::Bool);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_ge",
            vec![
                Hlvalue::Variable(i_fixed_u.clone()),
                Hlvalue::Variable(len_fix_u),
            ],
            Hlvalue::Variable(oob2.clone()),
        ));
    let i_fixed = variable_with_lltype("index", LowLevelType::Signed);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_uint_to_int",
            vec![Hlvalue::Variable(i_fixed_u)],
            Hlvalue::Variable(i_fixed.clone()),
        ));
    block_fixup.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob2));
    block_fixup.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(l_fix), Hlvalue::Variable(i_fixed)],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // ---- block_dispatch: c = direct_call(fast, l, index); return c.
    let c = variable_with_lltype("c", item_lltype);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
            ],
            Hlvalue::Variable(c.clone()),
        ));
    block_dispatch.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(c)],
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

/// Build (or retrieve cached) the list-getitem helper for the
/// `(checkidx, nonneg)` combination (`rlist.py:247-267`), selecting the
/// `dum_nocheck` fast path, the negative-index `ll_getitem`, the nonneg
/// `dum_checkidx` `ll_getitem_nonneg`, or the full checked `ll_getitem`.
fn list_getitem_helper(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    checkidx: bool,
    nonneg: bool,
    foldable: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<LowLevelFunction, TyperError> {
    // rlist.py:255-266 `basegetitem` selection by the listdef's `mutated` flag,
    // passed into ALL four (nonneg×checkidx) wrappers: the foldable
    // `ll_getitem_foldable_nonneg` (rlist.py:721-724) when not mutated, else
    // `ll_getitem_fast`.  Each foldable wrapper gets a distinct `_foldable`
    // cache name so `lowlevel_helper_function_with_builder` never serves a
    // foldable graph to a mutated list of the same item_lltype.  Only Fixed has
    // a foldable element load; Resized is always mutated (the `mutated | resized`
    // invariant in ListDef construction) so `foldable` is never set for it.
    let base = match (checkidx, nonneg) {
        (false, true) => layout.getitem_fast_name(),
        (false, false) => layout.getitem_neg_name(),
        (true, true) => layout.getitem_nonneg_checked_name(),
        (true, false) => layout.getitem_checked_name(),
    };
    let name = if foldable {
        format!("{base}_foldable")
    } else {
        base.to_string()
    };
    let name_owned = name.clone();
    let ptr_for_builder = ptr_lltype.clone();
    let item_for_builder = item_lltype.clone();
    rtyper.lowlevel_helper_function_with_builder(
        name,
        vec![ptr_lltype, LowLevelType::Signed],
        item_lltype,
        move |rtyper_inner, _args, _result| match (checkidx, nonneg) {
            (false, true) => match layout {
                // rlist.py:721-724 `ll_getitem_foldable_nonneg` — the Fixed
                // fast load, but foldable (`oopspec = 'list.getitem_foldable'`).
                // The Resized layout is always mutated so `foldable` can only
                // fire on Fixed.
                ListLayout::Fixed if foldable => build_ll_fixed_getitem_fast_foldable_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                ),
                ListLayout::Fixed => build_ll_fixed_getitem_fast_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                ),
                ListLayout::Resized => build_ll_getitem_fast_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                ),
            },
            (false, false) => build_ll_list_getitem_neg_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                foldable,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            (true, true) => build_ll_list_getitem_nonneg_checked_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                foldable,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            (true, false) => build_ll_list_getitem_checked_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                foldable,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
        },
    )
}

/// Shared `pair(AbstractBaseListRepr, IntegerRepr).rtype_getitem`
/// (`rlist.py:247-267`) for both list layouts. The `getitem_idx` op collapses
/// onto `getitem` in the rtyper dispatch, so `hop.has_implicit_exception` is
/// the `dum_checkidx` selector — the same caught-IndexError signal
/// `rtype_setitem` uses (`rlist.py:273`). Returns the (internal-repr) element
/// value; the caller applies `recast`.
fn list_rtype_getitem(
    hop: &HighLevelOp,
    self_repr: &dyn Repr,
    layout: ListLayout,
    ptr_lltype: &LowLevelType,
    item_lltype: &LowLevelType,
) -> Result<Hlvalue, TyperError> {
    use crate::annotator::model::SomeValue;
    let checkidx = hop.has_implicit_exception("IndexError");
    // rlist.py:255-258 `basegetitem` selection by the listdef's mutated
    // flag: `ll_getitem_fast` (non-foldable) when `listdef.listitem.mutated`,
    // else `ll_getitem_foldable_nonneg` (foldable, `oopspec =
    // 'list.getitem_foldable(l, index)'`, rlist.py:721-724). `mutated` is the
    // gate, NOT the layout — a non-resized FixedSizeListRepr that is still
    // mutated (setitem without resize) is non-foldable; a Resized list carries
    // the `mutated | resized` construction invariant (listdef.py:128 /
    // listdef.rs) so it is always mutated ⟹ never foldable.
    let s0 = hop
        .args_s
        .borrow()
        .first()
        .cloned()
        .ok_or_else(|| TyperError::message("list rtype_getitem: args_s[0] missing"))?;
    let foldable = match &s0 {
        SomeValue::List(lst) => !lst.listdef.listitem_rc().borrow().mutated,
        other => {
            return Err(TyperError::message(format!(
                "list rtype_getitem: args_s[0] must be SomeList, got {other:?}"
            )));
        }
    };
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
    let args = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
    ])?;
    if checkidx {
        hop.exception_is_here()?;
    } else {
        hop.exception_cannot_occur()?;
    }
    let helper = list_getitem_helper(
        &hop.rtyper,
        layout,
        checkidx,
        nonneg,
        foldable,
        ptr_lltype.clone(),
        item_lltype.clone(),
    )?;
    hop.gendirectcall(&helper, args)?
        .ok_or_else(|| TyperError::message("list getitem helper unexpectedly returned Void"))
}

/// The `None` (Void) constant a void-returning helper's returnblock link
/// carries (`rlist.py` setitem helpers return nothing).
fn none_void() -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::None,
        LowLevelType::Void,
    ))
}

/// Build (or retrieve cached) the layout's `ll_*_setitem_fast` sub-helper and
/// return a funcptr `Constant` to `direct_call` it (the `basesetitem` of
/// `rlist.py:283`).
fn list_setitem_fast_funcptr(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<Constant, TyperError> {
    let name = layout.setitem_fast_name().to_string();
    let name_owned = name.clone();
    let ptr_for_builder = ptr_lltype.clone();
    let item_for_builder = item_lltype.clone();
    let inner = rtyper.lowlevel_helper_function_with_builder(
        name,
        vec![ptr_lltype, LowLevelType::Signed, item_lltype],
        LowLevelType::Void,
        move |_rtyper, _args, _result| match layout {
            ListLayout::Fixed => build_ll_fixed_setitem_fast_helper_graph(
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            ListLayout::Resized => build_ll_setitem_fast_helper_graph(
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
        },
    )?;
    sub_helper_funcptr_constant(rtyper, &inner)
}

/// rlist.py:716-734 `ll_setitem` with `func is dum_nocheck` — negative index,
/// no bound check: `if index < 0: index += l.ll_length(); l.ll_setitem_fast(
/// index, item)`. 3-block CFG forwarding the possibly-fixed index + item to a
/// `direct_call` of the layout's `ll_*_setitem_fast` (Void).
fn build_ll_list_setitem_neg_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast =
        list_setitem_fast_funcptr(rtyper, layout, ptr_lltype.clone(), item_lltype.clone())?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let item = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(item.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_fix = variable_with_lltype("l", ptr_lltype.clone());
    let i_fix = variable_with_lltype("index", LowLevelType::Signed);
    let item_fix = variable_with_lltype("item", item_lltype.clone());
    let block_neg_fix = Block::shared(vec![
        Hlvalue::Variable(l_fix.clone()),
        Hlvalue::Variable(i_fix.clone()),
        Hlvalue::Variable(item_fix.clone()),
    ]);

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let item_disp = variable_with_lltype("item", item_lltype);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
        Hlvalue::Variable(item_disp.clone()),
    ]);

    // ---- start: is_neg = int_lt(index, 0); branch.
    let is_neg = variable_with_lltype("is_neg", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lt",
        vec![Hlvalue::Variable(i.clone()), signed_const(0)],
        Hlvalue::Variable(is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l.clone()),
                Hlvalue::Variable(i.clone()),
                Hlvalue::Variable(item.clone()),
            ],
            Some(block_neg_fix.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l),
                Hlvalue::Variable(i),
                Hlvalue::Variable(item),
            ],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // ---- block_neg_fix: length = <len read>; i_fixed = int_add(index, length).
    let length = emit_list_length_read(&block_neg_fix, layout, &l_fix);
    let i_fixed = variable_with_lltype("index", LowLevelType::Signed);
    block_neg_fix
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(i_fix), Hlvalue::Variable(length)],
            Hlvalue::Variable(i_fixed.clone()),
        ));
    block_neg_fix.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l_fix),
                Hlvalue::Variable(i_fixed),
                Hlvalue::Variable(item_fix),
            ],
            Some(block_dispatch.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ---- block_dispatch: direct_call(fast, l, index, item); return None.
    let v_void = variable_with_lltype("v", LowLevelType::Void);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
                Hlvalue::Variable(item_disp),
            ],
            Hlvalue::Variable(v_void),
        ));
    block_dispatch.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
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

/// rlist.py:716-720 `ll_setitem_nonneg` with `func is dum_checkidx` — nonneg
/// index, bound check: `if index >= l.ll_length(): raise IndexError;
/// l.ll_setitem_fast(index, item)`. 2-block CFG plus graph.exceptblock.
fn build_ll_list_setitem_nonneg_checked_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast =
        list_setitem_fast_funcptr(rtyper, layout, ptr_lltype.clone(), item_lltype.clone())?;
    let exc_args = exception_args("IndexError")?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let item = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(item.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let item_disp = variable_with_lltype("item", item_lltype);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
        Hlvalue::Variable(item_disp.clone()),
    ]);

    // ---- start: length = <len read>; oob = int_ge(index, length); branch.
    let length = emit_list_length_read(&startblock, layout, &l);
    let oob = variable_with_lltype("oob", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(i.clone()), Hlvalue::Variable(length)],
        Hlvalue::Variable(oob.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l),
                Hlvalue::Variable(i),
                Hlvalue::Variable(item),
            ],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // ---- block_dispatch: direct_call(fast, l, index, item); return None.
    let v_void = variable_with_lltype("v", LowLevelType::Void);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
                Hlvalue::Variable(item_disp),
            ],
            Hlvalue::Variable(v_void),
        ));
    block_dispatch.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
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

/// rlist.py:716-734 `ll_setitem` with `func is dum_checkidx` — fold the
/// negative index in (`index += length`) then enforce the `0 <= index <
/// length` window, raising IndexError otherwise (the r_uint window test
/// lowered to the signed-explicit `index >= length or index < 0` form, as in
/// [`build_ll_list_getitem_checked_helper_graph`]): 5-block CFG plus
/// graph.exceptblock.
fn build_ll_list_setitem_checked_helper_graph(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    name: &str,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let c_fast =
        list_setitem_fast_funcptr(rtyper, layout, ptr_lltype.clone(), item_lltype.clone())?;
    let exc_args = exception_args("IndexError")?;

    let l = variable_with_lltype("l", ptr_lltype.clone());
    let i = variable_with_lltype("index", LowLevelType::Signed);
    let item = variable_with_lltype("item", item_lltype.clone());
    let startblock = Block::shared(vec![
        Hlvalue::Variable(l.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(item.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let l_fix = variable_with_lltype("l", ptr_lltype.clone());
    let i_fix_u = variable_with_lltype("index", LowLevelType::Unsigned);
    let len_fix_u = variable_with_lltype("length", LowLevelType::Unsigned);
    let item_fix = variable_with_lltype("item", item_lltype.clone());
    let block_fixup = Block::shared(vec![
        Hlvalue::Variable(l_fix.clone()),
        Hlvalue::Variable(i_fix_u.clone()),
        Hlvalue::Variable(len_fix_u.clone()),
        Hlvalue::Variable(item_fix.clone()),
    ]);

    let l_disp = variable_with_lltype("l", ptr_lltype);
    let i_disp = variable_with_lltype("index", LowLevelType::Signed);
    let item_disp = variable_with_lltype("item", item_lltype);
    let block_dispatch = Block::shared(vec![
        Hlvalue::Variable(l_disp.clone()),
        Hlvalue::Variable(i_disp.clone()),
        Hlvalue::Variable(item_disp.clone()),
    ]);

    // ---- start: length = <len read>; index_u = cast_int_to_uint(index);
    //      length_u = cast_int_to_uint(length); oob = uint_ge(index_u, length_u);
    //      branch.  The common 0 <= index < length case falls straight through
    //      with no add (`ll_setitem`, rlist.py:737).
    let length = emit_list_length_read(&startblock, layout, &l);
    let i_u = variable_with_lltype("index", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(i.clone())],
        Hlvalue::Variable(i_u.clone()),
    ));
    let len_u = variable_with_lltype("length", LowLevelType::Unsigned);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(length)],
        Hlvalue::Variable(len_u.clone()),
    ));
    let oob = variable_with_lltype("oob", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "uint_ge",
        vec![
            Hlvalue::Variable(i_u.clone()),
            Hlvalue::Variable(len_u.clone()),
        ],
        Hlvalue::Variable(oob.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(l.clone()),
                Hlvalue::Variable(i_u),
                Hlvalue::Variable(len_u),
                Hlvalue::Variable(item.clone()),
            ],
            Some(block_fixup.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l),
                Hlvalue::Variable(i),
                Hlvalue::Variable(item),
            ],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // ---- block_fixup: index_u = uint_add(index_u, length_u);
    //      if uint_ge(index_u, length_u): raise IndexError;
    //      index = intmask(index_u); -> dispatch.
    let i_fixed_u = variable_with_lltype("index", LowLevelType::Unsigned);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_add",
            vec![
                Hlvalue::Variable(i_fix_u),
                Hlvalue::Variable(len_fix_u.clone()),
            ],
            Hlvalue::Variable(i_fixed_u.clone()),
        ));
    let oob2 = variable_with_lltype("oob", LowLevelType::Bool);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "uint_ge",
            vec![
                Hlvalue::Variable(i_fixed_u.clone()),
                Hlvalue::Variable(len_fix_u),
            ],
            Hlvalue::Variable(oob2.clone()),
        ));
    let i_fixed = variable_with_lltype("index", LowLevelType::Signed);
    block_fixup
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_uint_to_int",
            vec![Hlvalue::Variable(i_fixed_u)],
            Hlvalue::Variable(i_fixed.clone()),
        ));
    block_fixup.borrow_mut().exitswitch = Some(Hlvalue::Variable(oob2));
    block_fixup.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(bool_const(true)),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(l_fix),
                Hlvalue::Variable(i_fixed),
                Hlvalue::Variable(item_fix),
            ],
            Some(block_dispatch.clone()),
            Some(bool_const(false)),
        )
        .into_ref(),
    ]);

    // ---- block_dispatch: direct_call(fast, l, index, item); return None.
    let v_void = variable_with_lltype("v", LowLevelType::Void);
    block_dispatch
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(c_fast),
                Hlvalue::Variable(l_disp),
                Hlvalue::Variable(i_disp),
                Hlvalue::Variable(item_disp),
            ],
            Hlvalue::Variable(v_void),
        ));
    block_dispatch.closeblock(vec![
        Link::new(vec![none_void()], Some(graph.returnblock.clone()), None).into_ref(),
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

/// Build (or retrieve cached) the list-setitem helper for the
/// `(checkidx, nonneg)` combination (`rlist.py:272-284`).
fn list_setitem_helper(
    rtyper: &RPythonTyper,
    layout: ListLayout,
    checkidx: bool,
    nonneg: bool,
    ptr_lltype: LowLevelType,
    item_lltype: LowLevelType,
) -> Result<LowLevelFunction, TyperError> {
    let name = match (checkidx, nonneg) {
        (false, true) => layout.setitem_fast_name(),
        (false, false) => layout.setitem_neg_name(),
        (true, true) => layout.setitem_nonneg_checked_name(),
        (true, false) => layout.setitem_checked_name(),
    }
    .to_string();
    let name_owned = name.clone();
    let ptr_for_builder = ptr_lltype.clone();
    let item_for_builder = item_lltype.clone();
    rtyper.lowlevel_helper_function_with_builder(
        name,
        vec![ptr_lltype, LowLevelType::Signed, item_lltype],
        LowLevelType::Void,
        move |rtyper_inner, _args, _result| match (checkidx, nonneg) {
            (false, true) => match layout {
                ListLayout::Fixed => build_ll_fixed_setitem_fast_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                ),
                ListLayout::Resized => build_ll_setitem_fast_helper_graph(
                    &name_owned,
                    ptr_for_builder.clone(),
                    item_for_builder.clone(),
                ),
            },
            (false, false) => build_ll_list_setitem_neg_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            (true, true) => build_ll_list_setitem_nonneg_checked_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
            (true, false) => build_ll_list_setitem_checked_helper_graph(
                rtyper_inner,
                layout,
                &name_owned,
                ptr_for_builder.clone(),
                item_for_builder.clone(),
            ),
        },
    )
}

/// Shared `pair(AbstractBaseListRepr, IntegerRepr).rtype_setitem`
/// (`rlist.py:272-284`) for both list layouts. `hop.has_implicit_exception`
/// is the `dum_checkidx` selector; the item is the third inputarg, converted
/// to the internal `item_repr` (so `rtype_setitem` does not `recast`).
fn list_rtype_setitem(
    hop: &HighLevelOp,
    self_repr: &dyn Repr,
    layout: ListLayout,
    ptr_lltype: &LowLevelType,
    item_lltype: &LowLevelType,
    item_repr: &dyn Repr,
) -> RTypeResult {
    use crate::annotator::model::SomeValue;
    let checkidx = hop.has_implicit_exception("IndexError");
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
    let args = hop.inputargs(vec![
        ConvertedTo::Repr(self_repr),
        ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ConvertedTo::Repr(item_repr),
    ])?;
    hop.exception_is_here()?;
    let helper = list_setitem_helper(
        &hop.rtyper,
        layout,
        checkidx,
        nonneg,
        ptr_lltype.clone(),
        item_lltype.clone(),
    )?;
    hop.gendirectcall(&helper, args)
}

/// RPython `class ListIteratorRepr(AbstractListIteratorRepr)`
/// (`lltypesystem/rlist.py:453-461`):
///
/// ```python
/// class ListIteratorRepr(AbstractListIteratorRepr):
///     def __init__(self, r_list):
///         self.r_list = r_list
///         self.lowleveltype = Ptr(GcStruct('listiter',
///             ('list', r_list.lowleveltype),
///             ('index', Signed)))
///         self.ll_listiter = ll_listiter
///         self.ll_listnext = ll_listnext
///         self.ll_getnextindex = ll_getnextindex
/// ```
///
/// The Rust port stores the list's `lowleveltype` + `item_repr` + resized
/// flag rather than the full `r_list` repr: `make_iterator_repr` is a
/// `&self` method on the list repr and cannot reproduce the `Arc<dyn
/// Repr>` the upstream `r_list` field holds, so the iterator carries
/// exactly the data its `ll_listiter` / `ll_listnext` helpers consume —
/// the iter struct shape, the element repr, and the `getarraysize`-vs-
/// `length`-field length distinction.
#[derive(Debug)]
pub struct ListIteratorRepr {
    state: ReprState,
    /// `Ptr(GcStruct('listiter', ('list', LIST), ('index', Signed)))`.
    lltype: LowLevelType,
    /// `r_list.lowleveltype` — the `list` field type and the receiver
    /// `ll_listnext` reads through.
    list_lltype: LowLevelType,
    /// `r_list.item_repr` — the element repr `ll_listnext` returns.
    item_repr: Arc<dyn Repr>,
    /// `r_list.external_item_repr` (`lltypesystem/rlist.py:457`
    /// `self.external_item_repr = r_list.external_item_repr`) — the surface
    /// element repr `rtype_next` recasts the `ll_listnext` result back to
    /// (identity for primitive items; see [`list_recast`]).
    external_item_repr: Arc<dyn Repr>,
    /// The `ll_length` read-out SHAPE only: `true` for `FixedSizeListRepr`
    /// (length via `getarraysize` on the bare `Ptr(GcArray)`), `false` for
    /// the resized `ListRepr` (length via `getfield(l, "length")` on the
    /// header struct). Distinct from [`Self::foldable`]: the length read-out is
    /// the same `getarraysize`/`length` op whether or not the element load
    /// folds.
    list_is_fixed: bool,
    /// `not r_list.listitem.mutated` (`lltypesystem/rlist.py:462-466`): selects
    /// `ll_listnext_foldable` over `ll_listnext`, so an unmutated
    /// `FixedSizeListRepr`'s element load lowers to the PURE `getarrayitem_pure`
    /// the optimizer can fold / CSE across iterations. Only meaningful together
    /// with `list_is_fixed` — a `FixedSizeListRepr` can still be `mutated`
    /// (in-place setitem without resize) and a resized `ListRepr` is always
    /// `mutated`, so `rtype_next` gates the foldable read on
    /// `list_is_fixed && foldable`.
    foldable: bool,
}

impl ListIteratorRepr {
    pub fn new(
        list_lltype: LowLevelType,
        item_repr: Arc<dyn Repr>,
        external_item_repr: Arc<dyn Repr>,
        list_is_fixed: bool,
        foldable: bool,
    ) -> Result<Self, TyperError> {
        // upstream `Ptr(GcStruct('listiter', ('list', r_list.lowleveltype),
        // ('index', Signed)))`.
        let listiter_struct = Struct::gc(
            "listiter",
            vec![
                ("list".to_string(), list_lltype.clone()),
                ("index".to_string(), LowLevelType::Signed),
            ],
        );
        let lltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(listiter_struct),
        }));
        Ok(ListIteratorRepr {
            state: ReprState::new(),
            lltype,
            list_lltype,
            item_repr,
            external_item_repr,
            list_is_fixed,
            foldable,
        })
    }
}

impl Repr for ListIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ListIteratorRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::ListIteratorRepr
    }

    /// RPython `IteratorRepr.rtype_iter(self, hop)` (rmodel.py:266-268) —
    /// `iter(iter(x)) <==> iter(x)`: the iterator is its own iterator, so
    /// the op is the identity on the receiver.
    fn rtype_iter(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        Ok(Some(vlist[0].clone()))
    }

    /// RPython `AbstractListIteratorRepr.newiter(self, hop)`
    /// (`rlist.py:439-442`):
    ///
    /// ```python
    /// def newiter(self, hop):
    ///     v_lst, = hop.inputargs(self.r_list)
    ///     citerptr = hop.inputconst(Void, self.lowleveltype)
    ///     return hop.gendirectcall(self.ll_listiter, citerptr, v_lst)
    /// ```
    ///
    /// The Void `citerptr` type-tag is baked into the `ll_listiter`
    /// helper's `malloc` op (the helper is minted per-`ListIteratorRepr`),
    /// matching how `TupleRepr.newtuple` bakes the struct lltype into its
    /// malloc rather than threading a Void runtime arg.
    fn newiter(&self, hop: &HighLevelOp) -> RTypeResult {
        // upstream `v_lst, = hop.inputargs(self.r_list)`. The iter()
        // operand's repr (the list repr) is `hop.args_r[0]`; using it as
        // the conversion target keeps the `convertvar` identity
        // short-circuit. (A non-primitive `Ptr` list lltype has no
        // primitive repr, so `ConvertedTo::LowLevelType` cannot convert
        // it — the list repr the operand already carries is required.)
        let r_list = {
            let args_r = hop.args_r.borrow();
            args_r
                .first()
                .and_then(|o| o.clone())
                .ok_or_else(|| TyperError::message("ListIteratorRepr.newiter: arg0 repr missing"))?
        };
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(r_list.as_ref())])?;
        hop.exception_cannot_occur()?;
        let list_lltype = self.list_lltype.clone();
        let listiter_lltype = self.lltype.clone();
        let list_for_builder = list_lltype.clone();
        let iter_for_builder = listiter_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_listiter".to_string(),
            vec![list_lltype],
            listiter_lltype,
            move |_rtyper, _args, _result| {
                build_ll_listiter_helper_graph(
                    "ll_listiter",
                    list_for_builder.clone(),
                    iter_for_builder.clone(),
                )
            },
        )?;
        hop.gendirectcall(&helper, vlist)
    }

    /// RPython `AbstractListIteratorRepr.rtype_next(self, hop)`
    /// (`rlist.py:444-449`):
    ///
    /// ```python
    /// def rtype_next(self, hop):
    ///     v_iter, = hop.inputargs(self)
    ///     hop.has_implicit_exception(StopIteration)
    ///     hop.exception_is_here()
    ///     v_res = hop.gendirectcall(self.ll_listnext, v_iter)
    ///     return self.r_list.recast(hop.llops, v_res)
    /// ```
    ///
    /// `ll_listnext` (the `index >= ll_length()` bounds-check that raises
    /// `StopIteration`) lowers to [`build_ll_listnext_helper_graph`]. For an
    /// unmutated `FixedSizeListRepr` (`self.foldable && self.list_is_fixed`) the
    /// helper is `ll_listnext_foldable` (`lltypesystem/rlist.py:462-466,
    /// 484-491`), whose element read is the PURE `getarrayitem_pure`
    /// (`ll_getitem_foldable_nonneg`, the `list.getitem_foldable` oopspec) the
    /// optimizer can fold / CSE across iterations; every other list keeps the
    /// plain `getarrayitem`. The two helpers carry distinct names so the
    /// per-signature helper cache never serves a folded body for a mutated list
    /// of the same element type.
    /// The upstream result `recast` (`rlist.py:449` `self.r_list.recast`,
    /// `rlist.py:67`) converts the `ll_listnext` result back to
    /// `external_item_repr` via [`list_recast`] — identity for primitive
    /// items (no op emitted), pairtype dispatch for a GC-instance element
    /// list.
    fn rtype_next(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_iter = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        hop.has_implicit_exception("StopIteration");
        hop.exception_is_here()?;
        let item_lltype = self.item_repr.lowleveltype().clone();
        let iter_lltype = self.lltype.clone();
        let list_lltype = self.list_lltype.clone();
        let list_is_fixed = self.list_is_fixed;
        // lltypesystem/rlist.py:462-466 — an unmutated FixedSizeListRepr reads
        // each element through the PURE `getarrayitem_pure`
        // (`ll_getitem_foldable_nonneg`); every other list takes the plain
        // getarrayitem. A resized list is always mutated, so `list_is_fixed`
        // also gates it.
        let foldable = self.foldable && list_is_fixed;
        let helper_name = if foldable {
            "ll_listnext_foldable"
        } else {
            "ll_listnext"
        };
        let iter_for_builder = iter_lltype.clone();
        let list_for_builder = list_lltype.clone();
        let item_for_builder = item_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            helper_name.to_string(),
            vec![iter_lltype],
            item_lltype,
            move |_rtyper, _args, _result| {
                build_ll_listnext_helper_graph(
                    helper_name,
                    iter_for_builder.clone(),
                    list_for_builder.clone(),
                    item_for_builder.clone(),
                    list_is_fixed,
                    foldable,
                )
            },
        )?;
        let v_res = hop
            .gendirectcall(&helper, v_iter)?
            .ok_or_else(|| TyperError::message("list rtype_next: ll_listnext returned Void"))?;
        Ok(Some(list_recast(
            hop,
            &self.item_repr,
            &self.external_item_repr,
            v_res,
        )?))
    }
}

/// Synthesise `ll_listiter` (`lltypesystem/rlist.py:470-474`):
///
/// ```python
/// def ll_listiter(ITERPTR, lst):
///     iter = malloc(ITERPTR.TO)
///     iter.list = lst
///     iter.index = 0
///     return iter
/// ```
///
/// Single-block graph: `malloc(listiter struct)` → `setfield(iter,
/// "list", lst)` → `setfield(iter, "index", 0)` → return iter. The
/// `ITERPTR` type-tag is baked into the `malloc` op's Void operand (the
/// helper is minted with the iter lltype known), so the runtime signature
/// is `ll_listiter(lst)`.
pub(crate) fn build_ll_listiter_helper_graph(
    name: &str,
    list_lltype: LowLevelType,
    listiter_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let lst_arg = variable_with_lltype("lst", list_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(lst_arg.clone())]);
    let return_var = variable_with_lltype("result", listiter_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // upstream `malloc(ITERPTR.TO)` — the Void `c1` carries the inner
    // listiter Struct lltype, `cflags` the gc flavor, exactly as
    // `TupleRepr.newtuple` encodes its malloc.
    let LowLevelType::Ptr(ptr) = &listiter_lltype else {
        return Err(TyperError::message(
            "build_ll_listiter_helper_graph: listiter lltype is not Ptr",
        ));
    };
    let inner_struct = match &ptr.TO {
        PtrTarget::Struct(body) => body.clone(),
        other => {
            return Err(TyperError::message(format!(
                "build_ll_listiter_helper_graph: Ptr target must be Struct, got {other:?}"
            )));
        }
    };
    let c1 = Constant::with_concretetype(
        ConstValue::LowLevelType(Box::new(LowLevelType::Struct(Box::new(inner_struct)))),
        LowLevelType::Void,
    );
    let cflags = Constant::with_concretetype(ConstValue::byte_str("flavor=gc"), LowLevelType::Void);
    let v_iter = variable_with_lltype("iter", listiter_lltype.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "malloc",
        vec![Hlvalue::Constant(c1), Hlvalue::Constant(cflags)],
        Hlvalue::Variable(v_iter.clone()),
    ));
    // iter.list = lst
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            void_field_const("list"),
            Hlvalue::Variable(lst_arg),
        ],
        Hlvalue::Variable(variable_with_lltype("v0", LowLevelType::Void)),
    ));
    // iter.index = 0
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setfield",
        vec![
            Hlvalue::Variable(v_iter.clone()),
            void_field_const("index"),
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Int(0),
                LowLevelType::Signed,
            )),
        ],
        Hlvalue::Variable(variable_with_lltype("v1", LowLevelType::Void)),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_iter)],
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
        vec!["lst".to_string()],
        func,
    ))
}

/// Synthesise `ll_listnext` (`lltypesystem/rlist.py:476-482`):
///
/// ```python
/// def ll_listnext(iter):
///     l = iter.list
///     index = iter.index
///     if index >= l.ll_length():
///         raise StopIteration
///     iter.index = index + 1
///     return l.ll_getitem_fast(index)
/// ```
///
/// Two-block CFG mirroring [`lowlevel_range_check_helper_graph`]'s
/// raise-to-`exceptblock` shape:
/// - **startblock**: `l = getfield(iter, "list")`,
///   `index = getfield(iter, "index")`, `len = ll_length(l)`
///   (`getarraysize(l)` for the fixed array receiver, `getfield(l,
///   "length")` for the resized header struct), `cond = int_lt(index,
///   len)`. `exitswitch = cond`: the `false` (out-of-bounds) exit links to
///   `graph.exceptblock` with `exception_args("StopIteration")`, the `true`
///   exit carries `(iter, l, index)` to the continue block.
/// - **continue**: `iter.index = int_add(index, 1)`, then
///   `res = ll_getitem_fast(l, index)` (`getarrayitem(l, index)` for the
///   fixed array; `getfield(l, "items")` then `getarrayitem(items, index)`
///   for the resized struct), return `res`.
pub(crate) fn build_ll_listnext_helper_graph(
    name: &str,
    iter_lltype: LowLevelType,
    list_lltype: LowLevelType,
    item_lltype: LowLevelType,
    list_is_fixed: bool,
    foldable: bool,
) -> Result<PyGraph, TyperError> {
    // The resized list keeps its element array in the "items" field; the
    // fixed list IS the bare `Ptr(GcArray)`.
    let items_lltype = if list_is_fixed {
        None
    } else {
        let extracted = match &list_lltype {
            LowLevelType::Ptr(p) => match &p.TO {
                PtrTarget::Struct(s) => s._flds.get("items").cloned(),
                _ => None,
            },
            _ => None,
        };
        Some(extracted.ok_or_else(|| {
            TyperError::message(
                "build_ll_listnext_helper_graph: resized list lltype missing items field",
            )
        })?)
    };

    let iter_arg = variable_with_lltype("iter", iter_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(iter_arg.clone())]);
    let return_var = variable_with_lltype("result", item_lltype.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // upstream `raise StopIteration` — the [etype, evalue] pair the
    // exceptblock link carries.
    let exc_args = exception_args("StopIteration")?;

    // startblock: l = iter.list; index = iter.index; len = ll_length(l);
    //             cond = index < len.
    let v_l = variable_with_lltype("l", list_lltype.clone());
    let v_index = variable_with_lltype("index", LowLevelType::Signed);
    let v_len = variable_with_lltype("len", LowLevelType::Signed);
    let v_cond = variable_with_lltype("cond", LowLevelType::Bool);
    {
        let mut b = startblock.borrow_mut();
        b.operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(iter_arg.clone()),
                void_field_const("list"),
            ],
            Hlvalue::Variable(v_l.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(iter_arg.clone()),
                void_field_const("index"),
            ],
            Hlvalue::Variable(v_index.clone()),
        ));
        if list_is_fixed {
            b.operations.push(SpaceOperation::new(
                "getarraysize",
                vec![Hlvalue::Variable(v_l.clone())],
                Hlvalue::Variable(v_len.clone()),
            ));
        } else {
            b.operations.push(SpaceOperation::new(
                "getfield",
                vec![Hlvalue::Variable(v_l.clone()), void_field_const("length")],
                Hlvalue::Variable(v_len.clone()),
            ));
        }
        b.operations.push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(v_index.clone()),
                Hlvalue::Variable(v_len.clone()),
            ],
            Hlvalue::Variable(v_cond.clone()),
        ));
        b.exitswitch = Some(Hlvalue::Variable(v_cond.clone()));
    }

    // continue block receives (iter, l, index).
    let c_iter = variable_with_lltype("iter", iter_lltype);
    let c_l = variable_with_lltype("l", list_lltype);
    let c_index = variable_with_lltype("index", LowLevelType::Signed);
    let cont = Block::shared(vec![
        Hlvalue::Variable(c_iter.clone()),
        Hlvalue::Variable(c_l.clone()),
        Hlvalue::Variable(c_index.clone()),
    ]);

    startblock.closeblock(vec![
        // index < len -> continue (carry iter, l, index).
        Link::new(
            vec![
                Hlvalue::Variable(iter_arg),
                Hlvalue::Variable(v_l),
                Hlvalue::Variable(v_index),
            ],
            Some(cont.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        // index >= len -> raise StopIteration.
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    // continue: iter.index = index + 1; res = ll_getitem_fast(l, index).
    let v_newindex = variable_with_lltype("newindex", LowLevelType::Signed);
    let v_res = variable_with_lltype("res", item_lltype.clone());
    {
        let mut b = cont.borrow_mut();
        b.operations.push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(c_index.clone()),
                constant_with_lltype(ConstValue::Int(1), LowLevelType::Signed),
            ],
            Hlvalue::Variable(v_newindex.clone()),
        ));
        b.operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(c_iter),
                void_field_const("index"),
                Hlvalue::Variable(v_newindex),
            ],
            Hlvalue::Variable(variable_with_lltype("v", LowLevelType::Void)),
        ));
        if let Some(items_lltype) = items_lltype {
            let v_items = variable_with_lltype("items", items_lltype);
            b.operations.push(SpaceOperation::new(
                "getfield",
                vec![Hlvalue::Variable(c_l.clone()), void_field_const("items")],
                Hlvalue::Variable(v_items.clone()),
            ));
            b.operations.push(SpaceOperation::new(
                "getarrayitem",
                vec![Hlvalue::Variable(v_items), Hlvalue::Variable(c_index)],
                Hlvalue::Variable(v_res.clone()),
            ));
        } else {
            // `ll_listnext_foldable` (rlist.py:484-491) reads the element via
            // `ll_getitem_foldable_nonneg` = the PURE `getarrayitem_pure`; the
            // plain `ll_listnext` uses the non-pure `getarrayitem`. Only the
            // fixed list folds (a resized list is always mutated).
            let read_op = if foldable {
                "getarrayitem_pure"
            } else {
                "getarrayitem"
            };
            b.operations.push(SpaceOperation::new(
                read_op,
                vec![Hlvalue::Variable(c_l), Hlvalue::Variable(c_index)],
                Hlvalue::Variable(v_res.clone()),
            ));
        }
    }
    cont.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_res)],
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
        vec!["iter".to_string()],
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

    /// rlist.py:247-267 nonneg + checkidx=False branch on an UNMUTATED
    /// `FixedSizeListRepr` — `basegetitem = ll_getitem_foldable_nonneg`
    /// (rlist.py:258), so `getitem` lowers to a `direct_call` of the foldable
    /// `ll_fixed_getitem_fast_foldable` (a single `getarrayitem_pure` on the
    /// `Ptr(GcArray)` receiver), preceded by `hop.exception_cannot_occur()`.
    #[test]
    fn fixed_size_list_getitem_nonneg_unmutated_emits_foldable_helper() {
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
            .unwrap_or_else(|err| panic!("list getitem nonneg unmutated: {err:?}"));
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
            dbg.contains("ll_fixed_getitem_fast_foldable"),
            "unmutated list must select the foldable helper, got {dbg}"
        );
    }

    /// rlist.py:255-256 — a MUTATED `FixedSizeListRepr` (in-place setitem
    /// without resize) keeps `basegetitem = ll_getitem_fast`, so `getitem`
    /// lowers to the NON-foldable `ll_fixed_getitem_fast` (plain
    /// `getarrayitem`), proving the foldable selection is gated on
    /// `listitem.mutated`, not on the Fixed layout.
    #[test]
    fn fixed_size_list_getitem_nonneg_mutated_emits_nonfoldable_helper() {
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
                /* mutated */ true,
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
            .unwrap_or_else(|err| panic!("list getitem nonneg mutated: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_fixed_getitem_fast") && !dbg.contains("_foldable"),
            "mutated list must select the non-foldable helper, got {dbg}"
        );
    }

    /// The foldable helper body [`build_ll_fixed_getitem_fast_foldable_helper_graph`]
    /// emits the PURE `getarrayitem_pure` element load (rlist.py:721-724
    /// `oopspec = 'list.getitem_foldable'`), while the non-foldable
    /// [`build_ll_fixed_getitem_fast_helper_graph`] emits a plain
    /// `getarrayitem`.
    #[test]
    fn fixed_getitem_fast_foldable_body_emits_getarrayitem_pure() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let repr = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let ptr = repr.lowleveltype().clone();
        let item = LowLevelType::Signed;

        let foldable = build_ll_fixed_getitem_fast_foldable_helper_graph(
            "ll_fixed_getitem_fast_foldable",
            ptr.clone(),
            item.clone(),
        )
        .expect("build foldable helper");
        let nonfoldable =
            build_ll_fixed_getitem_fast_helper_graph("ll_fixed_getitem_fast", ptr, item)
                .expect("build non-foldable helper");

        // `block_op_sequences` also visits the (op-less) returnblock.
        assert_eq!(
            block_op_sequences(&foldable),
            vec![vec!["getarrayitem_pure".to_string()], vec![]],
            "foldable helper body must be the pure element load"
        );
        assert_eq!(
            block_op_sequences(&nonfoldable),
            vec![vec!["getarrayitem".to_string()], vec![]],
            "non-foldable helper body must be the plain element load"
        );
    }

    /// rlist.py:247-267 negative-index branch (`args_s[1].nonneg == false`,
    /// no caught IndexError → `dum_nocheck` `ll_getitem`) — `getitem` lowers
    /// to a `direct_call` of the neg-fix helper `ll_fixed_getitem` (not the
    /// `_fast` helper), preceded by `hop.exception_cannot_occur()`.
    #[test]
    fn fixed_size_list_getitem_negative_index_emits_direct_call_to_ll_fixed_getitem() {
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

        let result = list_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("list getitem neg: {err:?}"));
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
            dbg.contains("ll_fixed_getitem") && !dbg.contains("ll_fixed_getitem_fast"),
            "expected the neg helper 'll_fixed_getitem' (not '_fast') in {dbg}"
        );
    }

    /// rlist.py:247-267 checkidx branch (`hop.has_implicit_exception(
    /// "IndexError")`, nonneg index → `ll_getitem_nonneg` with
    /// `dum_checkidx`) — `getitem` lowers to a `direct_call` of the
    /// bound-checking helper `ll_fixed_getitem_nonneg_checked`, preceded by
    /// `hop.exception_is_here()`.
    #[test]
    fn fixed_size_list_getitem_checkidx_emits_direct_call_to_nonneg_checked() {
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
        let result = list_repr
            .rtype_getitem(&hop)
            .unwrap_or_else(|err| panic!("list getitem checkidx: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "checkidx=True path must call hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_fixed_getitem_nonneg_checked"),
            "expected 'll_fixed_getitem_nonneg_checked' in {dbg}"
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

    /// rlist.py:272-284 negative-index branch (`args_s[1].nonneg == false`,
    /// no caught IndexError → `dum_nocheck` `ll_setitem`) — `setitem` lowers
    /// to a `direct_call` of the neg-fix helper `ll_fixed_setitem` (not the
    /// `_fast` helper), preceded by `hop.exception_is_here()`.
    #[test]
    fn fixed_size_list_setitem_negative_index_emits_direct_call_to_ll_fixed_setitem() {
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

        let result = list_repr
            .rtype_setitem(&hop)
            .unwrap_or_else(|err| panic!("list setitem neg: {err:?}"));
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
            dbg.contains("ll_fixed_setitem") && !dbg.contains("ll_fixed_setitem_fast"),
            "expected the neg helper 'll_fixed_setitem' (not '_fast') in {dbg}"
        );
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

    #[test]
    fn runtime_helper_surface_is_explicitly_deferred() {
        let _iter = AbstractListIteratorRepr;

        let err = ll_append().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_append"));

        let err = ll_listslice_startstop().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_listslice_startstop"));
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

    /// The resized [`ListRepr`] inherits `reverse` from the common base
    /// (`rlist.py:138-143`); it rtypes through `rtype_method("reverse")` to a
    /// `direct_call(ll_reverse, v_lst)` just like `FixedSizeListRepr`, and the
    /// path calls `hop.exception_cannot_occur()`.
    #[test]
    fn resized_list_reverse_emits_direct_call_to_ll_reverse() {
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
                /* resized */ true,
            )))]);
        hop.args_r
            .borrow_mut()
            .extend([Some(list_repr.clone() as Arc<dyn Repr>)]);

        let result = list_repr
            .rtype_method("reverse", &hop)
            .unwrap_or_else(|err| panic!("resized list reverse: {err:?}"));
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

    /// The resized `ll_reverse` is the same four-block swap loop as the
    /// fixed-size one, but the startblock reads `length` via `getfield`
    /// (vs `getarraysize`) and every element access reads the `items` array
    /// out of the struct header first (`getfield` before each
    /// `getarrayitem` / `setarrayitem`).
    #[test]
    fn build_ll_reverse_resized_helper_has_swap_loop_blocks() {
        let rtyper = fresh_rtyper();
        let repr = ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new");
        let pygraph = build_ll_reverse_resized_helper_graph(
            "ll_reverse",
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_reverse_resized_helper_graph");
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
        // The resized startblock reads `length` from the struct header.
        assert!(
            block_op_seqs.contains(&vec!["getfield".to_string(), "int_sub".to_string()]),
            "startblock must be getfield + int_sub, got {block_op_seqs:?}"
        );
        assert!(
            block_op_seqs.contains(&vec!["int_lt".to_string()]),
            "expected an int_lt condition block, got {block_op_seqs:?}"
        );
        // The swap body reads `items` (getfield) before each array op; both
        // reads precede both writes, then the two index steps.
        assert!(
            block_op_seqs.contains(&vec![
                "getfield".to_string(),
                "getarrayitem".to_string(),
                "getfield".to_string(),
                "getarrayitem".to_string(),
                "getfield".to_string(),
                "setarrayitem".to_string(),
                "getfield".to_string(),
                "setarrayitem".to_string(),
                "int_add".to_string(),
                "int_sub".to_string(),
            ]),
            "expected the items-indirected swap body, got {block_op_seqs:?}"
        );
    }

    /// Count the outgoing links across all reachable blocks of `pygraph`
    /// whose target is `graph.exceptblock` (the IndexError-raising arms).
    fn count_links_to_exceptblock(pygraph: &PyGraph) -> usize {
        let graph = pygraph.graph.borrow();
        graph
            .iterblocks()
            .iter()
            .flat_map(|b| b.borrow().exits.clone())
            .filter(|link| {
                link.borrow()
                    .target
                    .as_ref()
                    .is_some_and(|t| std::rc::Rc::ptr_eq(t, &graph.exceptblock))
            })
            .count()
    }

    fn block_op_sequences(pygraph: &PyGraph) -> Vec<Vec<String>> {
        pygraph
            .graph
            .borrow()
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .map(|op| op.opname.clone())
                    .collect()
            })
            .collect()
    }

    /// rlist.py:699-714 checked `ll_getitem` (`dum_checkidx`, negative index
    /// folded in) lowers to the unsigned-window test `r_uint(index) >=
    /// r_uint(length)`: the start reads length (`getarraysize` on the
    /// `FixedSizeListRepr` receiver), casts index/length to unsigned and
    /// `uint_ge`-branches; the fixup block does `uint_add`, a second `uint_ge`
    /// raising IndexError, and `intmask` (`cast_uint_to_int`) feeding the
    /// `direct_call` dispatch.  Only the post-add bound check raises (one link
    /// to exceptblock); the common `0 <= index < length` case falls through.
    #[test]
    fn build_ll_list_getitem_checked_helper_fixed_has_window_checks() {
        // Keep `ann` alive for the duration: building the inner
        // `ll_*_getitem_fast` sub-helper annotates it through the typer's
        // weak annotator reference.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let repr = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let pygraph = build_ll_list_getitem_checked_helper_graph(
            &rtyper,
            ListLayout::Fixed,
            "ll_fixed_getitem_checked",
            /* foldable */ false,
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_list_getitem_checked_helper_graph");
        let seqs = block_op_sequences(&pygraph);
        assert!(
            seqs.contains(&vec![
                "getarraysize".to_string(),
                "cast_int_to_uint".to_string(),
                "cast_int_to_uint".to_string(),
                "uint_ge".to_string(),
            ]),
            "start must read length (getarraysize), cast to unsigned, then uint_ge, got {seqs:?}"
        );
        assert!(
            seqs.contains(&vec![
                "uint_add".to_string(),
                "uint_ge".to_string(),
                "cast_uint_to_int".to_string(),
            ]),
            "expected a uint_add/uint_ge/cast_uint_to_int fixup block, got {seqs:?}"
        );
        assert!(
            seqs.contains(&vec!["direct_call".to_string()]),
            "expected a direct_call dispatch block, got {seqs:?}"
        );
        assert_eq!(
            count_links_to_exceptblock(&pygraph),
            1,
            "only the post-add bound check raises IndexError"
        );
    }

    /// rlist.py:737-742 checked `ll_setitem` for the resized [`ListRepr`]: the
    /// length read is `getfield(l, "length")` (struct header) not
    /// `getarraysize`, the unsigned-window test then casts/`uint_ge`-branches
    /// and the fixup block (`uint_add`/`uint_ge`/`cast_uint_to_int`) threads the
    /// `item` operand through to a `direct_call` of `ll_setitem_fast`; only the
    /// post-add bound check raises IndexError.
    #[test]
    fn build_ll_list_setitem_checked_helper_resized_reads_length_via_getfield() {
        // Keep `ann` alive (see the getitem-checked test above).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");
        let repr = ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new");
        let pygraph = build_ll_list_setitem_checked_helper_graph(
            &rtyper,
            ListLayout::Resized,
            "ll_setitem_checked",
            repr.lowleveltype().clone(),
            LowLevelType::Signed,
        )
        .expect("build_ll_list_setitem_checked_helper_graph");
        let seqs = block_op_sequences(&pygraph);
        assert!(
            seqs.contains(&vec![
                "getfield".to_string(),
                "cast_int_to_uint".to_string(),
                "cast_int_to_uint".to_string(),
                "uint_ge".to_string(),
            ]),
            "resized start must read length (getfield), cast to unsigned, then uint_ge, got {seqs:?}"
        );
        assert!(
            seqs.contains(&vec![
                "uint_add".to_string(),
                "uint_ge".to_string(),
                "cast_uint_to_int".to_string(),
            ]),
            "expected a uint_add/uint_ge/cast_uint_to_int fixup block, got {seqs:?}"
        );
        assert!(
            seqs.contains(&vec!["direct_call".to_string()]),
            "expected a direct_call dispatch block, got {seqs:?}"
        );
        assert_eq!(
            count_links_to_exceptblock(&pygraph),
            1,
            "only the post-add bound check raises IndexError"
        );
    }

    /// `ListIteratorRepr`'s lowleveltype is `Ptr(GcStruct("listiter",
    /// ("list", LIST), ("index", Signed)))` (`lltypesystem/rlist.py:455-458`).
    #[test]
    fn list_iterator_repr_lltype_is_ptr_gcstruct_list_index() {
        let rtyper = fresh_rtyper();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let r_iter = ListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
            false,
        )
        .expect("ListIteratorRepr::new");
        assert_eq!(r_iter.class_name(), "ListIteratorRepr");
        assert_eq!(r_iter.repr_class_id(), ReprClassId::ListIteratorRepr);

        let LowLevelType::Ptr(ptr) = r_iter.lowleveltype() else {
            panic!("ListIteratorRepr lltype must be a Ptr");
        };
        let PtrTarget::Struct(body) = &ptr.TO else {
            panic!("ListIteratorRepr Ptr target must be a Struct");
        };
        assert_eq!(body._name, "listiter");
        assert_eq!(body._flds.get("index"), Some(&LowLevelType::Signed));
        // the `list` field carries the list repr's own lowleveltype.
        assert_eq!(body._flds.get("list"), Some(r_list.lowleveltype()));
    }

    /// `SomeIterator(SomeList)` routes through `SomeIterator.rtyper_makerepr`
    /// → `r_container.make_iterator_repr()` → `ListIteratorRepr`
    /// (`rmodel.py:274-282`).
    #[test]
    fn makerepr_somelist_iterator_routes_to_list_iterator_repr() {
        let rtyper = fresh_rtyper_live();
        let ldef = ListDef::new(
            None,
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
            false,
        );
        let s_list = SomeValue::List(SomeList::new(ldef));
        let s_iter =
            SomeValue::Iterator(crate::annotator::model::SomeIterator::new(s_list, vec![]));
        let repr = rtyper_makerepr(&s_iter, &rtyper).expect("rtyper_makerepr list iterator");
        assert_eq!(repr.class_name(), "ListIteratorRepr");
        assert_eq!(repr.repr_class_id(), ReprClassId::ListIteratorRepr);
    }

    /// `ll_listiter` body is `malloc(listiter)` → `setfield(iter, "list",
    /// lst)` → `setfield(iter, "index", 0)` (`lltypesystem/rlist.py:470-474`).
    #[test]
    fn build_ll_listiter_helper_emits_malloc_then_two_setfields() {
        let rtyper = fresh_rtyper();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let r_iter = ListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
            false,
        )
        .expect("ListIteratorRepr::new");
        let pygraph = build_ll_listiter_helper_graph(
            "ll_listiter",
            r_list.lowleveltype().clone(),
            r_iter.lowleveltype().clone(),
        )
        .expect("build_ll_listiter_helper_graph");
        let graph = pygraph.graph.borrow();
        let ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(ops, vec!["malloc", "setfield", "setfield"]);
    }

    /// `iter(list)` rtypes through the default `Repr.rtype_iter`
    /// (`make_iterator_repr().newiter(hop)`) to a `direct_call(ll_listiter,
    /// v_lst)` (`rmodel.py:229-231` + `rlist.py:439-442`).
    #[test]
    fn fixed_size_list_iter_emits_direct_call_to_ll_listiter() {
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
        let iter_lltype = ListIteratorRepr::new(
            list_lltype.clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
            false,
        )
        .expect("ListIteratorRepr::new")
        .lowleveltype()
        .clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_list = Variable::new();
        v_list.set_concretetype(Some(list_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(iter_lltype));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "iter".to_string(),
                vec![Hlvalue::Variable(v_list)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s
            .borrow_mut()
            .push(SomeValue::List(SomeList::new(ListDef::new(
                None,
                SomeValue::Integer(SomeInteger::new(false, false)),
                false,
                false,
            ))));
        hop.args_r
            .borrow_mut()
            .push(Some(list_repr.clone() as Arc<dyn Repr>));

        let result = list_repr
            .rtype_iter(&hop)
            .unwrap_or_else(|err| panic!("list iter: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_listiter"),
            "expected 'll_listiter' in {dbg}"
        );
    }

    /// `ll_listnext` over a fixed list: startblock bounds-checks via
    /// `getfield`/`getfield`/`getarraysize`/`int_lt` and the continue block
    /// `int_add`/`setfield`/`getarrayitem` (`lltypesystem/rlist.py:476-482`).
    /// The out-of-bounds exit links to the graph's `exceptblock`.
    #[test]
    fn build_ll_listnext_helper_fixed_bounds_checks_and_getarrayitem() {
        let rtyper = fresh_rtyper_live();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let r_iter = ListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
            false,
        )
        .expect("ListIteratorRepr::new");
        let pygraph = build_ll_listnext_helper_graph(
            "ll_listnext",
            r_iter.lowleveltype().clone(),
            r_list.lowleveltype().clone(),
            LowLevelType::Signed,
            true,
            false,
        )
        .expect("build_ll_listnext_helper_graph");
        let graph = pygraph.graph.borrow();
        let start_ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(
            start_ops,
            vec!["getfield", "getfield", "getarraysize", "int_lt"]
        );
        // startblock branches on the bounds-check; one exit raises via the
        // graph's exceptblock.
        let start = graph.startblock.borrow();
        assert!(start.exitswitch.is_some(), "bounds-check exitswitch");
        let except_key = crate::flowspace::model::BlockKey::of(&graph.exceptblock);
        let raises = start.exits.iter().any(|lnk| {
            lnk.borrow()
                .target
                .as_ref()
                .is_some_and(|t| crate::flowspace::model::BlockKey::of(t) == except_key)
        });
        assert!(
            raises,
            "one startblock exit must link to exceptblock (raise StopIteration)"
        );
        // the non-raising exit's continue block reads the element.
        let cont_ops: Vec<Vec<String>> = graph
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
        assert!(
            cont_ops.iter().any(|seq| seq
                == &vec![
                    "int_add".to_string(),
                    "setfield".to_string(),
                    "getarrayitem".to_string()
                ]),
            "continue block must int_add/setfield/getarrayitem, got {cont_ops:?}"
        );
    }

    /// `ll_listnext_foldable` over an unmutated fixed list reads the element via
    /// the PURE `getarrayitem_pure` (`lltypesystem/rlist.py:484-491` →
    /// `ll_getitem_foldable_nonneg`), so the trace optimizer can fold / CSE the
    /// load across iterations.
    #[test]
    fn build_ll_listnext_foldable_helper_emits_getarrayitem_pure() {
        let rtyper = fresh_rtyper_live();
        let r_list = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let r_iter = ListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            true,
            true,
        )
        .expect("ListIteratorRepr::new");
        let pygraph = build_ll_listnext_helper_graph(
            "ll_listnext_foldable",
            r_iter.lowleveltype().clone(),
            r_list.lowleveltype().clone(),
            LowLevelType::Signed,
            true,
            true,
        )
        .expect("build_ll_listnext_helper_graph");
        let graph = pygraph.graph.borrow();
        let cont_ops: Vec<Vec<String>> = graph
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
        assert!(
            cont_ops.iter().any(|seq| seq
                == &vec![
                    "int_add".to_string(),
                    "setfield".to_string(),
                    "getarrayitem_pure".to_string()
                ]),
            "foldable continue block must emit getarrayitem_pure, got {cont_ops:?}"
        );
    }

    /// `ll_listnext` over a resized list reads `length` from the header and
    /// `items` array before `getarrayitem` (`lltypesystem/rlist.py` ADTIList).
    #[test]
    fn build_ll_listnext_helper_resized_reads_length_and_items() {
        let rtyper = fresh_rtyper_live();
        let r_list = ListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>).expect("ListRepr::new");
        let r_iter = ListIteratorRepr::new(
            r_list.lowleveltype().clone(),
            signed_repr() as Arc<dyn Repr>,
            signed_repr() as Arc<dyn Repr>,
            false,
            false,
        )
        .expect("ListIteratorRepr::new");
        let pygraph = build_ll_listnext_helper_graph(
            "ll_listnext",
            r_iter.lowleveltype().clone(),
            r_list.lowleveltype().clone(),
            LowLevelType::Signed,
            false,
            false,
        )
        .expect("build_ll_listnext_helper_graph");
        let graph = pygraph.graph.borrow();
        let start_ops: Vec<_> = graph
            .startblock
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        // resized length via getfield "length" (not getarraysize).
        assert_eq!(
            start_ops,
            vec!["getfield", "getfield", "getfield", "int_lt"]
        );
        let cont_ops: Vec<Vec<String>> = graph
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
        assert!(
            cont_ops.iter().any(|seq| seq
                == &vec![
                    "int_add".to_string(),
                    "setfield".to_string(),
                    "getfield".to_string(),
                    "getarrayitem".to_string()
                ]),
            "resized continue must read items array before getarrayitem, got {cont_ops:?}"
        );
    }

    /// `next(iter)` rtypes through `ListIteratorRepr::rtype_next` to a
    /// `direct_call(ll_listnext, v_iter)`, recording the implicit
    /// `StopIteration` (`rlist.py:444-449`).
    #[test]
    fn list_iterator_next_emits_direct_call_to_ll_listnext() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata in test setup");

        let list_repr = FixedSizeListRepr::new(&rtyper, signed_repr() as Arc<dyn Repr>)
            .expect("FixedSizeListRepr::new");
        let iter_repr: Arc<ListIteratorRepr> = Arc::new(
            ListIteratorRepr::new(
                list_repr.lowleveltype().clone(),
                signed_repr() as Arc<dyn Repr>,
                signed_repr() as Arc<dyn Repr>,
                true,
                false,
            )
            .expect("ListIteratorRepr::new"),
        );
        let iter_lltype = iter_repr.lowleveltype().clone();

        let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
            rtyper.clone(),
            None,
        )));
        let v_iter = Variable::new();
        v_iter.set_concretetype(Some(iter_lltype));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Signed));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "next".to_string(),
                vec![Hlvalue::Variable(v_iter)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops.clone(),
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().push(SomeValue::Iterator(
            crate::annotator::model::SomeIterator::new(
                SomeValue::List(SomeList::new(ListDef::new(
                    None,
                    SomeValue::Integer(SomeInteger::new(false, false)),
                    false,
                    false,
                ))),
                vec![],
            ),
        ));
        hop.args_r
            .borrow_mut()
            .push(Some(iter_repr.clone() as Arc<dyn Repr>));

        let result = iter_repr
            .rtype_next(&hop)
            .unwrap_or_else(|err| panic!("list next: {err:?}"));
        assert!(matches!(result, Some(Hlvalue::Variable(_))));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "direct_call");
        assert!(
            ops._called_exception_is_here_or_cannot_occur,
            "rtype_next must call hop.exception_is_here()"
        );
        let Hlvalue::Constant(c) = &ops.ops[0].args[0] else {
            panic!("expected Constant funcptr as direct_call arg 0");
        };
        let dbg = format!("{:?}", c.value);
        assert!(
            dbg.contains("ll_listnext"),
            "expected 'll_listnext' in {dbg}"
        );
    }
}
