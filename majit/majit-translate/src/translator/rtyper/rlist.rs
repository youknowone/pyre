//! RPython `rpython/rtyper/rlist.py` + `lltypesystem/rlist.py` —
//! minimal `FixedSizeListRepr` slice covering the `len(list)` lowering.
//!
//! Rust slices (`&[T]`) and fixed arrays annotate as a non-resized
//! `SomeList` (`bookkeeper`), so `slice.len()` — routed to the rtyper's
//! `len` operation by `flowspace_adapter` — lands on this repr's
//! `rtype_len`.
//!
//! The full list repr surface (`ListRepr` resized `GcStruct("list",
//! length, items)`, the `ADTIList` / `ADTIFixedList` adtmeth tables,
//! `rtype_method_append` / `getitem` / `setitem` / iterators, and the
//! `pairtype(BaseListRepr, BaseListRepr)` operations) is deferred to
//! follow-on slices. Today only the data shape `Ptr(GcArray(ITEM))`
//! and the `ll_fixed_length` lowering land.

use std::rc::Rc;
use std::sync::Arc;

use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{ArrayType, LowLevelType, Ptr, PtrTarget};
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, HighLevelOp, RPythonTyper, helper_pygraph_from_graph, variable_with_lltype,
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
