//! RPython `rpython/rtyper/lltypesystem/rordereddict.py`.
//!
//! PyPy aliases all dict annotations to `SomeOrderedDict`, so this is
//! the default concrete repr selected by `SomeDict.rtyper_makerepr`.
//! This slice lands `OrderedDictRepr`'s low-level table shape and the public
//! constant/helper names around the low-level method section. The dense
//! lookup/resizing/helper family remains a follow-up line-by-line port.
#![allow(non_camel_case_types, non_snake_case)]

use std::rc::Rc;
use std::sync::{Arc, LazyLock};

use crate::annotator::dictdef::DictDef;
use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    Array, GCREF, LowLevelType, Ptr, PtrTarget, Struct,
};
use crate::translator::rtyper::rdict::{AbstractDictIteratorRepr, AbstractDictRepr};
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, HighLevelOp, RPythonTyper, constant_with_lltype, exception_args,
    helper_pygraph_from_graph, variable_with_lltype, void_field_const,
};

fn ptr_to_gc_array(of: LowLevelType) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::gc(of)),
    }))
}

/// RPython `DICTINDEX_*` pointer aliases (`rordereddict.py:489-492`).
///
/// PyPy uses width-specific `rffi.UCHAR` / `USHORT` / `UINT` element types.
/// Pyre's current `lltype`/`rffi` surface does not yet distinguish those C
/// integer widths, so all four aliases carry the available unsigned primitive
/// while preserving the exact public names and `Ptr(GcArray(...))` shape.
pub static DICTINDEX_LONG: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));
pub static DICTINDEX_INT: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));
pub static DICTINDEX_SHORT: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));
pub static DICTINDEX_BYTE: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_gc_array(LowLevelType::Unsigned));

/// RPython `IS_64BIT = sys.maxint != 2 ** 31 - 1` (`rordereddict.py:494`).
pub const IS_64BIT: bool = usize::BITS == 64;

/// RPython lookup table selector constants (`rordereddict.py:496-503`).
pub const FUNC_SHIFT: i64 = if IS_64BIT { 3 } else { 2 };
pub const FUNC_MASK: i64 = if IS_64BIT { 0x07 } else { 0x03 };
pub const FUNC_BYTE: i64 = 0;
pub const FUNC_SHORT: i64 = 1;
pub const FUNC_INT: i64 = 2;
pub const FUNC_LONG: i64 = if IS_64BIT { 3 } else { 2 };
pub const FUNC_MUST_REINDEX: i64 = if IS_64BIT { 4 } else { 3 };

/// RPython `TYPE_*` aliases (`rordereddict.py:504-507`). See
/// [`DICTINDEX_BYTE`] for the current width-collapsing note.
pub const TYPE_BYTE: LowLevelType = LowLevelType::Unsigned;
pub const TYPE_SHORT: LowLevelType = LowLevelType::Unsigned;
pub const TYPE_INT: LowLevelType = LowLevelType::Unsigned;
pub const TYPE_LONG: LowLevelType = LowLevelType::Unsigned;

pub const PERTURB_SHIFT: i64 = 5;
pub const FREE: i64 = 0;
pub const DELETED: i64 = 1;
pub const VALID_OFFSET: i64 = 2;
pub const MIN_INDEXES_MINUS_ENTRIES: i64 = VALID_OFFSET + 1;

pub const FLAG_LOOKUP: i64 = 0;
pub const FLAG_STORE: i64 = 1;

/// RPython `DICT_INITSIZE = 16` (`rordereddict.py:1156`).
pub const DICT_INITSIZE: i64 = 16;

/// RPython `class OrderedDictRepr(AbstractDictRepr)`
/// (`lltypesystem/rordereddict.py:173`).
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct OrderedDictRepr {
    pub base: AbstractDictRepr,
    pub DICTKEY: LowLevelType,
    pub DICTVALUE: LowLevelType,
    /// RPython `Struct("odictentry", ...)`.
    pub DICTENTRY: Struct,
    pub DICTENTRYARRAY: Array,
    /// RPython `GcStruct("dicttable", ...)`.
    pub DICT: Struct,
    lowleveltype: LowLevelType,
}

impl OrderedDictRepr {
    pub fn new(
        rtyper: Rc<RPythonTyper>,
        key_repr: Arc<dyn Repr>,
        value_repr: Arc<dyn Repr>,
        dictdef: DictDef,
        custom_eq_hash_repr: Option<(Arc<dyn Repr>, Arc<dyn Repr>)>,
        force_non_null: bool,
        simple_hash_eq: bool,
    ) -> Result<Self, TyperError> {
        let custom_eq_hash = custom_eq_hash_repr.is_some();
        let (external_key_repr, key_repr) =
            AbstractDictRepr::pickrepr(&rtyper, key_repr, custom_eq_hash)?;
        let (external_value_repr, value_repr) = if custom_eq_hash {
            (value_repr.clone(), value_repr)
        } else {
            crate::translator::rtyper::rclass::externalvsinternal(&rtyper, value_repr, true)?
        };
        let dictkey_lltype = key_repr.lowleveltype().clone();
        let dictvalue_lltype = value_repr.lowleveltype().clone();

        let mut entryfields = vec![
            ("key".into(), dictkey_lltype.clone()),
            ("f_valid".into(), LowLevelType::Bool),
            ("value".into(), dictvalue_lltype.clone()),
        ];
        if !simple_hash_eq {
            entryfields.push(("f_hash".into(), LowLevelType::Signed));
        }
        let dictentry = Struct::new("odictentry", entryfields);
        let dictentryarray = Array::gc(LowLevelType::Struct(Box::new(dictentry.clone())));
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(dictentryarray.clone()),
        }));
        let mut fields = vec![
            ("num_live_items".into(), LowLevelType::Signed),
            ("num_ever_used_items".into(), LowLevelType::Signed),
            ("resize_counter".into(), LowLevelType::Signed),
            ("indexes".into(), GCREF.clone()),
            ("lookup_function_no".into(), LowLevelType::Signed),
            ("entries".into(), entries_ptr),
        ];
        if let Some((r_rdict_eqfn, r_rdict_hashfn)) = &custom_eq_hash_repr {
            fields.push(("fnkeyeq".into(), r_rdict_eqfn.lowleveltype().clone()));
            fields.push(("fnkeyhash".into(), r_rdict_hashfn.lowleveltype().clone()));
        }
        let dict = Struct::gc_with_hints(
            "dicttable",
            fields,
            vec![("dict".into(), ConstValue::Bool(true))],
        );
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(dict.clone()),
        }));

        Ok(OrderedDictRepr {
            base: AbstractDictRepr {
                state: ReprState::new(),
                rtyper,
                external_key_repr,
                key_repr,
                external_value_repr,
                value_repr,
                dictdef,
                custom_eq_hash,
                force_non_null,
                simple_hash_eq,
                custom_eq_hash_repr,
            },
            DICTKEY: dictkey_lltype,
            DICTVALUE: dictvalue_lltype,
            DICTENTRY: dictentry,
            DICTENTRYARRAY: dictentryarray,
            DICT: dict,
            lowleveltype,
        })
    }
}

impl Repr for OrderedDictRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.base.state
    }

    fn class_name(&self) -> &'static str {
        "OrderedDictRepr"
    }

    fn repr_class_id(&self) -> crate::translator::rtyper::pairtype::ReprClassId {
        crate::translator::rtyper::pairtype::ReprClassId::OrderedDictRepr
    }

    fn compact_repr(&self) -> String {
        self.base.compact_repr()
    }

    /// RPython `OrderedDictRepr.rtype_len(self, hop)`
    /// (`rordereddict.py:274-276`): `hop.gendirectcall(ll_dict_len, v_dict)`.
    /// `ll_dict_len(d)` (`rordereddict.py:648-649`) returns the
    /// `num_live_items` header field.
    fn rtype_len(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_dict = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let ptr_lltype = self.lowleveltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_dict_len".to_string(),
            vec![ptr_lltype],
            LowLevelType::Signed,
            move |_rtyper, _args, _result| {
                build_ll_dict_len_helper_graph("ll_dict_len", ptr_for_builder.clone())
            },
        )?;
        hop.gendirectcall(&helper, v_dict)
    }

    /// RPython `OrderedDictRepr.rtype_bool(self, hop)`
    /// (`rordereddict.py:278-280`): `hop.gendirectcall(ll_dict_bool, v_dict)`.
    /// `ll_dict_bool(d)` (`rordereddict.py:651-653`) is `bool(d) and
    /// d.num_live_items != 0` — the explicit `bool(d)` guard lets a None-typed
    /// dict read False without dereferencing, so this overrides the
    /// `int_is_true(len)` default (`rmodel.py:199-207`) which would deref the
    /// possibly-null receiver.
    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        let v_dict = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        let ptr_lltype = self.lowleveltype.clone();
        let ptr_for_builder = ptr_lltype.clone();
        let helper = hop.rtyper.lowlevel_helper_function_with_builder(
            "ll_dict_bool".to_string(),
            vec![ptr_lltype],
            LowLevelType::Bool,
            move |_rtyper, _args, _result| {
                build_ll_dict_bool_helper_graph("ll_dict_bool", ptr_for_builder.clone())
            },
        )?;
        hop.gendirectcall(&helper, v_dict)
    }
}

/// Synthesise `ll_dict_len(d) -> Signed` (`rordereddict.py:648-649`):
///
/// ```python
/// def ll_dict_len(d):
///     return d.num_live_items
/// ```
///
/// Single-block graph: `getfield(d, "num_live_items") -> Signed`, the live
/// entry count tracked in the `dicttable` header.
pub(crate) fn build_ll_dict_len_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("d", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_len = variable_with_lltype("num_live_items", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(arg), void_field_const("num_live_items")],
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
        vec!["d".to_string()],
        func,
    ))
}

/// Synthesise `ll_dict_bool(d) -> Bool` (`rordereddict.py:651-653`):
///
/// ```python
/// def ll_dict_bool(d):
///     # check if a dict is True, allowing for None
///     return bool(d) and d.num_live_items != 0
/// ```
///
/// Two-block CFG plus the returnblock:
/// - **start**: `v_nz = ptr_nonzero(d)`; branch on it. True → `check_len`
///   (forwarding `d`), False → returnblock(`False`) without dereferencing.
/// - **check_len**: `getfield(d, "num_live_items")`, `int_ne(n, 0)` →
///   returnblock(result).
pub(crate) fn build_ll_dict_bool_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("d", ptr_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_zero = || constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed);

    // check_len inputarg: same `d` ptr forwarded through the True branch.
    let d_for_len = variable_with_lltype("d", ptr_lltype);
    let block_check_len = Block::shared(vec![Hlvalue::Variable(d_for_len.clone())]);

    // ---- start: ptr_nonzero(d); branch on the result.
    let v_nz = variable_with_lltype("v_nz", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(arg.clone())],
        Hlvalue::Variable(v_nz.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(v_nz));
    let start_true_link = Link::new(
        vec![Hlvalue::Variable(arg)],
        Some(block_check_len.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let start_false_link = Link::new(
        vec![bool_false()],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    startblock.closeblock(vec![start_true_link, start_false_link]);

    // ---- check_len: getfield(num_live_items); int_ne(n, 0).
    let v_count = variable_with_lltype("num_live_items", LowLevelType::Signed);
    block_check_len
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "getfield",
            vec![
                Hlvalue::Variable(d_for_len),
                void_field_const("num_live_items"),
            ],
            Hlvalue::Variable(v_count.clone()),
        ));
    let v_result = variable_with_lltype("result", LowLevelType::Bool);
    block_check_len
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_ne",
            vec![Hlvalue::Variable(v_count), signed_zero()],
            Hlvalue::Variable(v_result.clone()),
        ));
    block_check_len.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(v_result)],
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
        vec!["d".to_string()],
        func,
    ))
}

pub fn ll_newdict_size(_dict: &Struct, _length_estimate: usize) -> Result<(), TyperError> {
    Err(TyperError::missing_rtype_operation(
        "lltypesystem.rordereddict.ll_newdict_size — ordered hash table allocation deferred",
    ))
}

fn ordered_dict_runtime_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "lltypesystem.rordereddict.{name} — ordered hash table runtime deferred"
    ))
}

pub fn ll_call_lookup_function() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_call_lookup_function"))
}

pub fn get_ll_dict() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("get_ll_dict"))
}

pub fn ll_no_initial_index() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_no_initial_index"))
}

pub fn ll_malloc_indexes_and_choose_lookup() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "ll_malloc_indexes_and_choose_lookup",
    ))
}

pub fn ll_clear_indexes() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_clear_indexes"))
}

/// Synthesise `_ll_write_indexes(d, i, value, T)` (`rordereddict.py:558-563`):
///
/// ```python
/// def _ll_write_indexes(d, i, value, T):
///     INDEXES = _ll_ptr_to_array_of(T)
///     indexes = lltype.cast_opaque_ptr(INDEXES, d.indexes)
///     cast_value = rffi.cast(T, value)
///     ll_assert(intmask(cast_value) == value, "...")   # debug-only, omitted
///     indexes[i] = cast_value
/// ```
///
/// Single-block graph storing `value` into the sparse index array at slot `i`.
/// `cast_opaque_ptr(INDEXES, d.indexes)` lowers to `cast_pointer` (GCREF ->
/// INDEXES); `rffi.cast(T, value)` narrows the Signed slot value to the
/// unsigned index element type via `cast_int_to_uint`. All `DICTINDEX_*` widths
/// collapse to `Ptr(GcArray(Unsigned))` here, so this one impl serves every
/// FUNC_* width. The `ll_assert` is a debug-only range check with no runtime
/// effect after translation and is not modelled.
pub fn build_ll_write_indexes_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    index_elem_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let indexes_ptr_lltype = _ll_ptr_to_array_of(index_elem_lltype.clone());

    let d = variable_with_lltype("d", dict_ptr_lltype);
    let i = variable_with_lltype("i", LowLevelType::Signed);
    let value = variable_with_lltype("value", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(d.clone()),
        Hlvalue::Variable(i.clone()),
        Hlvalue::Variable(value.clone()),
    ]);
    let return_var = variable_with_lltype("result", LowLevelType::Void);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    // indexes = cast_opaque_ptr(INDEXES, d.indexes)
    let v_gcref = variable_with_lltype("indexes_gcref", GCREF.clone());
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getfield",
        vec![Hlvalue::Variable(d), void_field_const("indexes")],
        Hlvalue::Variable(v_gcref.clone()),
    ));
    let v_indexes = variable_with_lltype("indexes", indexes_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_pointer",
        vec![Hlvalue::Variable(v_gcref)],
        Hlvalue::Variable(v_indexes.clone()),
    ));
    // cast_value = rffi.cast(T, value): narrow Signed slot value to the
    // unsigned index element width.
    let v_cast = variable_with_lltype("cast_value", index_elem_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_int_to_uint",
        vec![Hlvalue::Variable(value)],
        Hlvalue::Variable(v_cast.clone()),
    ));
    // indexes[i] = cast_value
    let v_void = variable_with_lltype("v", LowLevelType::Void);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "setarrayitem",
        vec![
            Hlvalue::Variable(v_indexes),
            Hlvalue::Variable(i),
            Hlvalue::Variable(v_cast),
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
        vec!["d".to_string(), "i".to_string(), "value".to_string()],
        func,
    ))
}

/// Select the key-equality op for the simple-hash-eq `checkingkey == key`
/// comparison (`direct_compare`), keyed on the entry key lltype. Custom
/// `keyeq`/`paranoia` keys are out of scope (PBC `hlinvoke`).
fn direct_compare_op(key_lltype: &LowLevelType) -> Result<&'static str, TyperError> {
    Ok(match key_lltype {
        LowLevelType::Signed | LowLevelType::Bool => "int_eq",
        LowLevelType::Unsigned => "uint_eq",
        LowLevelType::Char => "char_eq",
        LowLevelType::UniChar => "unichar_eq",
        LowLevelType::Ptr(_) => "ptr_eq",
        other => {
            return Err(TyperError::message(format!(
                "ll_dict_lookup direct_compare unsupported key lltype {other:?}"
            )));
        }
    })
}

/// Synthesise `ll_dict_lookup(d, key, hash, store_flag, T) -> Signed`
/// (`rordereddict.py:1038-1106`), the open-addressing perturb-probe that
/// every dict access routes through.
///
/// ```python
/// def ll_dict_lookup(d, key, hash, store_flag, T):
///     INDEXES = _ll_ptr_to_array_of(T)
///     entries = d.entries
///     indexes = lltype.cast_opaque_ptr(INDEXES, d.indexes)
///     mask = len(indexes) - 1
///     i = r_uint(hash & mask)
///     index = rffi.cast(lltype.Signed, indexes[intmask(i)])
///     if index >= VALID_OFFSET:
///         checkingkey = entries[index - VALID_OFFSET].key
///         if checkingkey == key:                 # direct_compare, keyeq is None
///             return index - VALID_OFFSET
///         deletedslot = -1
///     elif index == DELETED:
///         deletedslot = intmask(i)
///     else:                                       # pristine -- lookup failed
///         if store_flag == FLAG_STORE:
///             _ll_write_indexes(d, i, d.num_ever_used_items + VALID_OFFSET, T)
///         return -1
///     perturb = r_uint(hash)
///     while 1:
///         i = (i << 2) + i + perturb + 1
///         i = i & mask
///         index = rffi.cast(lltype.Signed, indexes[intmask(i)])
///         if index == FREE:
///             if store_flag == FLAG_STORE:
///                 if deletedslot == -1:
///                     deletedslot = intmask(i)
///                 _ll_write_indexes(d, deletedslot,
///                                   d.num_ever_used_items + VALID_OFFSET, T)
///             return -1
///         elif index >= VALID_OFFSET:
///             checkingkey = entries[index - VALID_OFFSET].key
///             if checkingkey == key:
///                 return index - VALID_OFFSET
///         elif deletedslot == -1:
///             deletedslot = intmask(i)
///         perturb >>= PERTURB_SHIFT
/// ```
///
/// **Scope (faithful subset):** simple-hash-eq + `direct_compare`. The
/// `d.keyeq is not None` / `d.paranoia` branches (custom `__eq__`, PBC
/// `hlinvoke`, lookup restart) are skipped — they are statically dead when
/// `keyeq is None`, exactly as RPython folds them. All `DICTINDEX_*` widths
/// collapse to `Ptr(GcArray(Unsigned))`, so this one graph serves every
/// `FUNC_*` width (the `T`/`ll_call_lookup_function` 4-way dispatch is inert).
///
/// **Unsigned arithmetic is load-bearing, not cosmetic:** `i` and `perturb`
/// are `r_uint`. `perturb >>= PERTURB_SHIFT` is a *logical* shift
/// (`uint_rshift`); a Signed `int_rshift` would sign-extend for the common
/// negative `hash`, walking a different probe sequence. So `i`/`perturb` are
/// modelled `Unsigned` end to end (`cast_int_to_uint` at the boundaries,
/// `cast_uint_to_int` for `intmask(i)` indexing). The `& mask` keeps the
/// signed/unsigned index value bit-identical, but the shift is not.
///
/// **Store path inlined:** `store_flag == FLAG_STORE` writes the index slot
/// via `cast_int_to_uint` + `setarrayitem` on the already-extracted `indexes`
/// — byte-equivalent to [`build_ll_write_indexes_helper_graph`]'s body (the
/// `indexes` pointer is invariant across a lookup, which never resizes). The
/// standalone `_ll_write_indexes` helper remains for the non-inlined callers
/// (`ll_dict_store_clean`, reindex) ported in later slices.
///
/// 13-block CFG (plus the returnblock). First-try-before-loop mirrors the
/// "do the first try before any looping" optimisation; the loop body
/// re-derives `i`, reads the slot, and 3-way branches FREE / VALID / DELETED
/// before the `perturb` shift back-edge.
pub fn build_ll_dict_lookup_helper_graph(
    name: &str,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
    index_elem_lltype: LowLevelType,
    key_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let indexes_ptr_lltype = _ll_ptr_to_array_of(index_elem_lltype.clone());
    let eq_op = direct_compare_op(&key_lltype)?;

    // Value/const constructors.
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let unsigned = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Unsigned);
    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let key_field = || void_field_const("key");
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let sig = || LowLevelType::Signed;
    let uns = || LowLevelType::Unsigned;
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);

    // ---- startblock inputargs: (d, key, hash, store_flag).
    let d = new_var("d", dict_ptr_lltype.clone());
    let key = new_var("key", key_lltype.clone());
    let hash = new_var("hash", sig());
    let store_flag = new_var("store_flag", sig());
    let startblock = Block::shared(vec![var(&d), var(&key), var(&hash), var(&store_flag)]);
    let return_var = new_var("result", sig());
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    // Pre-create every downstream block with fresh inputarg copies so the
    // back-edge target exists when each block is closed.
    // block_first_valid / block_first_notvalid carry the post-first-probe state:
    //   (d, entries, indexes, mask_u, key, store_flag, hash, i, index).
    let make_inv = |suffix: &str| {
        (
            new_var("d", dict_ptr_lltype.clone()),
            new_var("entries", entries_ptr_lltype.clone()),
            new_var("indexes", indexes_ptr_lltype.clone()),
            new_var("mask_u", uns()),
            new_var("key", key_lltype.clone()),
            new_var(&format!("store_flag{suffix}"), sig()),
        )
    };

    // block_first_valid.
    let (fv_d, fv_entries, fv_indexes, fv_mask, fv_key, fv_sf) = make_inv("");
    let fv_hash = new_var("hash", sig());
    let fv_i = new_var("i", uns());
    let fv_index = new_var("index", sig());
    let block_first_valid = Block::shared(vec![
        var(&fv_d),
        var(&fv_entries),
        var(&fv_indexes),
        var(&fv_mask),
        var(&fv_key),
        var(&fv_sf),
        var(&fv_hash),
        var(&fv_i),
        var(&fv_index),
    ]);

    // block_first_notvalid.
    let (nv_d, nv_entries, nv_indexes, nv_mask, nv_key, nv_sf) = make_inv("");
    let nv_hash = new_var("hash", sig());
    let nv_i = new_var("i", uns());
    let nv_index = new_var("index", sig());
    let block_first_notvalid = Block::shared(vec![
        var(&nv_d),
        var(&nv_entries),
        var(&nv_indexes),
        var(&nv_mask),
        var(&nv_key),
        var(&nv_sf),
        var(&nv_hash),
        var(&nv_i),
        var(&nv_index),
    ]);

    // block_first_pristine_store: (d, indexes, store_flag, i).
    let ps_d = new_var("d", dict_ptr_lltype.clone());
    let ps_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let ps_sf = new_var("store_flag", sig());
    let ps_i = new_var("i", uns());
    let block_first_pristine_store =
        Block::shared(vec![var(&ps_d), var(&ps_indexes), var(&ps_sf), var(&ps_i)]);

    // block_store_at: (d, indexes, slot) — inlined _ll_write_indexes + return -1.
    let st_d = new_var("d", dict_ptr_lltype.clone());
    let st_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let st_slot = new_var("slot", sig());
    let block_store_at = Block::shared(vec![var(&st_d), var(&st_indexes), var(&st_slot)]);

    // block_loop_init: (d, entries, indexes, mask_u, key, store_flag, hash, i, deletedslot).
    let (li_d, li_entries, li_indexes, li_mask, li_key, li_sf) = make_inv("");
    let li_hash = new_var("hash", sig());
    let li_i = new_var("i", uns());
    let li_ds = new_var("deletedslot", sig());
    let block_loop_init = Block::shared(vec![
        var(&li_d),
        var(&li_entries),
        var(&li_indexes),
        var(&li_mask),
        var(&li_key),
        var(&li_sf),
        var(&li_hash),
        var(&li_i),
        var(&li_ds),
    ]);

    // block_loop_body: (d, entries, indexes, mask_u, key, store_flag, perturb, i, deletedslot).
    let (lb_d, lb_entries, lb_indexes, lb_mask, lb_key, lb_sf) = make_inv("");
    let lb_perturb = new_var("perturb", uns());
    let lb_i = new_var("i", uns());
    let lb_ds = new_var("deletedslot", sig());
    let block_loop_body = Block::shared(vec![
        var(&lb_d),
        var(&lb_entries),
        var(&lb_indexes),
        var(&lb_mask),
        var(&lb_key),
        var(&lb_sf),
        var(&lb_perturb),
        var(&lb_i),
        var(&lb_ds),
    ]);

    // block_loop_notfree: loop_body + index.
    let (nf_d, nf_entries, nf_indexes, nf_mask, nf_key, nf_sf) = make_inv("");
    let nf_perturb = new_var("perturb", uns());
    let nf_i = new_var("i", uns());
    let nf_ds = new_var("deletedslot", sig());
    let nf_index = new_var("index", sig());
    let block_loop_notfree = Block::shared(vec![
        var(&nf_d),
        var(&nf_entries),
        var(&nf_indexes),
        var(&nf_mask),
        var(&nf_key),
        var(&nf_sf),
        var(&nf_perturb),
        var(&nf_i),
        var(&nf_ds),
        var(&nf_index),
    ]);

    // block_loop_valid: loop_body + index.
    let (lv_d, lv_entries, lv_indexes, lv_mask, lv_key, lv_sf) = make_inv("");
    let lv_perturb = new_var("perturb", uns());
    let lv_i = new_var("i", uns());
    let lv_ds = new_var("deletedslot", sig());
    let lv_index = new_var("index", sig());
    let block_loop_valid = Block::shared(vec![
        var(&lv_d),
        var(&lv_entries),
        var(&lv_indexes),
        var(&lv_mask),
        var(&lv_key),
        var(&lv_sf),
        var(&lv_perturb),
        var(&lv_i),
        var(&lv_ds),
        var(&lv_index),
    ]);

    // block_loop_deleted: loop_body shape.
    let (ld_d, ld_entries, ld_indexes, ld_mask, ld_key, ld_sf) = make_inv("");
    let ld_perturb = new_var("perturb", uns());
    let ld_i = new_var("i", uns());
    let ld_ds = new_var("deletedslot", sig());
    let block_loop_deleted = Block::shared(vec![
        var(&ld_d),
        var(&ld_entries),
        var(&ld_indexes),
        var(&ld_mask),
        var(&ld_key),
        var(&ld_sf),
        var(&ld_perturb),
        var(&ld_i),
        var(&ld_ds),
    ]);

    // block_perturb_shift: loop_body shape.
    let (sh_d, sh_entries, sh_indexes, sh_mask, sh_key, sh_sf) = make_inv("");
    let sh_perturb = new_var("perturb", uns());
    let sh_i = new_var("i", uns());
    let sh_ds = new_var("deletedslot", sig());
    let block_perturb_shift = Block::shared(vec![
        var(&sh_d),
        var(&sh_entries),
        var(&sh_indexes),
        var(&sh_mask),
        var(&sh_key),
        var(&sh_sf),
        var(&sh_perturb),
        var(&sh_i),
        var(&sh_ds),
    ]);

    // block_loop_free: (d, indexes, store_flag, i, deletedslot).
    let lf_d = new_var("d", dict_ptr_lltype.clone());
    let lf_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let lf_sf = new_var("store_flag", sig());
    let lf_i = new_var("i", uns());
    let lf_ds = new_var("deletedslot", sig());
    let block_loop_free = Block::shared(vec![
        var(&lf_d),
        var(&lf_indexes),
        var(&lf_sf),
        var(&lf_i),
        var(&lf_ds),
    ]);

    // block_free_choose_slot: (d, indexes, i, deletedslot).
    let fc_d = new_var("d", dict_ptr_lltype.clone());
    let fc_indexes = new_var("indexes", indexes_ptr_lltype.clone());
    let fc_i = new_var("i", uns());
    let fc_ds = new_var("deletedslot", sig());
    let block_free_choose_slot =
        Block::shared(vec![var(&fc_d), var(&fc_indexes), var(&fc_i), var(&fc_ds)]);

    // ===== startblock =====
    let entries = new_var("entries", entries_ptr_lltype.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("entries")],
        &entries,
    );
    let gcref = new_var("indexes_gcref", GCREF.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&d), void_field_const("indexes")],
        &gcref,
    );
    let indexes = new_var("indexes", indexes_ptr_lltype.clone());
    push(&startblock, "cast_pointer", vec![var(&gcref)], &indexes);
    let len = new_var("len", sig());
    push(&startblock, "getarraysize", vec![var(&indexes)], &len);
    let mask = new_var("mask", sig());
    push(&startblock, "int_sub", vec![var(&len), signed(1)], &mask);
    let mask_u = new_var("mask_u", uns());
    push(&startblock, "cast_int_to_uint", vec![var(&mask)], &mask_u);
    let hashmask = new_var("hashmask", sig());
    push(
        &startblock,
        "int_and",
        vec![var(&hash), var(&mask)],
        &hashmask,
    );
    let i0 = new_var("i", uns());
    push(&startblock, "cast_int_to_uint", vec![var(&hashmask)], &i0);
    let i0_s = new_var("i_s", sig());
    push(&startblock, "cast_uint_to_int", vec![var(&i0)], &i0_s);
    let elem0 = new_var("elem", uns());
    push(
        &startblock,
        "getarrayitem",
        vec![var(&indexes), var(&i0_s)],
        &elem0,
    );
    let index0 = new_var("index", sig());
    push(&startblock, "cast_uint_to_int", vec![var(&elem0)], &index0);
    let ge0 = new_var("ge", LowLevelType::Bool);
    push(
        &startblock,
        "int_ge",
        vec![var(&index0), signed(VALID_OFFSET)],
        &ge0,
    );
    startblock.borrow_mut().exitswitch = Some(var(&ge0));
    let first_args = vec![
        var(&d),
        var(&entries),
        var(&indexes),
        var(&mask_u),
        var(&key),
        var(&store_flag),
        var(&hash),
        var(&i0),
        var(&index0),
    ];
    startblock.closeblock(vec![
        Link::new(
            first_args.clone(),
            Some(block_first_valid.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            first_args,
            Some(block_first_notvalid.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ===== block_first_valid: checkingkey == key on the first probe. =====
    let fv_slot = new_var("slot", sig());
    push(
        &block_first_valid,
        "int_sub",
        vec![var(&fv_index), signed(VALID_OFFSET)],
        &fv_slot,
    );
    let fv_ckey = new_var("checkingkey", key_lltype.clone());
    push(
        &block_first_valid,
        "getinteriorfield",
        vec![var(&fv_entries), var(&fv_slot), key_field()],
        &fv_ckey,
    );
    let fv_eq = new_var("keyeq", LowLevelType::Bool);
    push(
        &block_first_valid,
        eq_op,
        vec![var(&fv_ckey), var(&fv_key)],
        &fv_eq,
    );
    block_first_valid.borrow_mut().exitswitch = Some(var(&fv_eq));
    block_first_valid.closeblock(vec![
        Link::new(
            vec![var(&fv_slot)],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&fv_d),
                var(&fv_entries),
                var(&fv_indexes),
                var(&fv_mask),
                var(&fv_key),
                var(&fv_sf),
                var(&fv_hash),
                var(&fv_i),
                signed(-1),
            ],
            Some(block_loop_init.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ===== block_first_notvalid: DELETED vs pristine FREE. =====
    let nv_i_s = new_var("i_s", sig());
    push(
        &block_first_notvalid,
        "cast_uint_to_int",
        vec![var(&nv_i)],
        &nv_i_s,
    );
    let nv_is_deleted = new_var("is_deleted", LowLevelType::Bool);
    push(
        &block_first_notvalid,
        "int_eq",
        vec![var(&nv_index), signed(DELETED)],
        &nv_is_deleted,
    );
    block_first_notvalid.borrow_mut().exitswitch = Some(var(&nv_is_deleted));
    block_first_notvalid.closeblock(vec![
        Link::new(
            vec![
                var(&nv_d),
                var(&nv_entries),
                var(&nv_indexes),
                var(&nv_mask),
                var(&nv_key),
                var(&nv_sf),
                var(&nv_hash),
                var(&nv_i),
                var(&nv_i_s),
            ],
            Some(block_loop_init.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![var(&nv_d), var(&nv_indexes), var(&nv_sf), var(&nv_i)],
            Some(block_first_pristine_store.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ===== block_first_pristine_store: store at i iff FLAG_STORE, else -1. =====
    let ps_i_s = new_var("i_s", sig());
    push(
        &block_first_pristine_store,
        "cast_uint_to_int",
        vec![var(&ps_i)],
        &ps_i_s,
    );
    let ps_is_store = new_var("is_store", LowLevelType::Bool);
    push(
        &block_first_pristine_store,
        "int_eq",
        vec![var(&ps_sf), signed(FLAG_STORE)],
        &ps_is_store,
    );
    block_first_pristine_store.borrow_mut().exitswitch = Some(var(&ps_is_store));
    block_first_pristine_store.closeblock(vec![
        Link::new(
            vec![var(&ps_d), var(&ps_indexes), var(&ps_i_s)],
            Some(block_store_at.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed(-1)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ===== block_store_at: indexes[slot] = num_ever_used + VALID_OFFSET; return -1. =====
    let st_neu = new_var("num_ever_used_items", sig());
    push(
        &block_store_at,
        "getfield",
        vec![var(&st_d), void_field_const("num_ever_used_items")],
        &st_neu,
    );
    let st_value = new_var("value", sig());
    push(
        &block_store_at,
        "int_add",
        vec![var(&st_neu), signed(VALID_OFFSET)],
        &st_value,
    );
    let st_cast = new_var("cast_value", index_elem_lltype.clone());
    push(
        &block_store_at,
        "cast_int_to_uint",
        vec![var(&st_value)],
        &st_cast,
    );
    let st_void = new_var("v", LowLevelType::Void);
    push(
        &block_store_at,
        "setarrayitem",
        vec![var(&st_indexes), var(&st_slot), var(&st_cast)],
        &st_void,
    );
    block_store_at.closeblock(vec![
        Link::new(vec![signed(-1)], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // ===== block_loop_init: perturb = r_uint(hash); enter loop. =====
    let li_perturb = new_var("perturb", uns());
    push(
        &block_loop_init,
        "cast_int_to_uint",
        vec![var(&li_hash)],
        &li_perturb,
    );
    block_loop_init.closeblock(vec![
        Link::new(
            vec![
                var(&li_d),
                var(&li_entries),
                var(&li_indexes),
                var(&li_mask),
                var(&li_key),
                var(&li_sf),
                var(&li_perturb),
                var(&li_i),
                var(&li_ds),
            ],
            Some(block_loop_body.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ===== block_loop_body: i = ((i<<2)+i+perturb+1)&mask; read slot; branch FREE. =====
    let lb_ish = new_var("ish", uns());
    push(
        &block_loop_body,
        "uint_lshift",
        vec![var(&lb_i), signed(2)],
        &lb_ish,
    );
    let lb_ipi = new_var("ipi", uns());
    push(
        &block_loop_body,
        "uint_add",
        vec![var(&lb_ish), var(&lb_i)],
        &lb_ipi,
    );
    let lb_ipp = new_var("ipp", uns());
    push(
        &block_loop_body,
        "uint_add",
        vec![var(&lb_ipi), var(&lb_perturb)],
        &lb_ipp,
    );
    let lb_iinc = new_var("iinc", uns());
    push(
        &block_loop_body,
        "uint_add",
        vec![var(&lb_ipp), unsigned(1)],
        &lb_iinc,
    );
    let lb_inew = new_var("i", uns());
    push(
        &block_loop_body,
        "uint_and",
        vec![var(&lb_iinc), var(&lb_mask)],
        &lb_inew,
    );
    let lb_inew_s = new_var("i_s", sig());
    push(
        &block_loop_body,
        "cast_uint_to_int",
        vec![var(&lb_inew)],
        &lb_inew_s,
    );
    let lb_elem = new_var("elem", uns());
    push(
        &block_loop_body,
        "getarrayitem",
        vec![var(&lb_indexes), var(&lb_inew_s)],
        &lb_elem,
    );
    let lb_index = new_var("index", sig());
    push(
        &block_loop_body,
        "cast_uint_to_int",
        vec![var(&lb_elem)],
        &lb_index,
    );
    let lb_is_free = new_var("is_free", LowLevelType::Bool);
    push(
        &block_loop_body,
        "int_eq",
        vec![var(&lb_index), signed(FREE)],
        &lb_is_free,
    );
    block_loop_body.borrow_mut().exitswitch = Some(var(&lb_is_free));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![
                var(&lb_d),
                var(&lb_indexes),
                var(&lb_sf),
                var(&lb_inew),
                var(&lb_ds),
            ],
            Some(block_loop_free.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&lb_d),
                var(&lb_entries),
                var(&lb_indexes),
                var(&lb_mask),
                var(&lb_key),
                var(&lb_sf),
                var(&lb_perturb),
                var(&lb_inew),
                var(&lb_ds),
                var(&lb_index),
            ],
            Some(block_loop_notfree.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ===== block_loop_notfree: index >= VALID_OFFSET vs DELETED. =====
    let nf_ge = new_var("ge", LowLevelType::Bool);
    push(
        &block_loop_notfree,
        "int_ge",
        vec![var(&nf_index), signed(VALID_OFFSET)],
        &nf_ge,
    );
    block_loop_notfree.borrow_mut().exitswitch = Some(var(&nf_ge));
    let nf_carry = vec![
        var(&nf_d),
        var(&nf_entries),
        var(&nf_indexes),
        var(&nf_mask),
        var(&nf_key),
        var(&nf_sf),
        var(&nf_perturb),
        var(&nf_i),
        var(&nf_ds),
    ];
    let mut nf_valid_args = nf_carry.clone();
    nf_valid_args.push(var(&nf_index));
    block_loop_notfree.closeblock(vec![
        Link::new(
            nf_valid_args,
            Some(block_loop_valid.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            nf_carry,
            Some(block_loop_deleted.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ===== block_loop_valid: checkingkey == key on a probed slot. =====
    let lv_slot = new_var("slot", sig());
    push(
        &block_loop_valid,
        "int_sub",
        vec![var(&lv_index), signed(VALID_OFFSET)],
        &lv_slot,
    );
    let lv_ckey = new_var("checkingkey", key_lltype.clone());
    push(
        &block_loop_valid,
        "getinteriorfield",
        vec![var(&lv_entries), var(&lv_slot), key_field()],
        &lv_ckey,
    );
    let lv_eq = new_var("keyeq", LowLevelType::Bool);
    push(
        &block_loop_valid,
        eq_op,
        vec![var(&lv_ckey), var(&lv_key)],
        &lv_eq,
    );
    block_loop_valid.borrow_mut().exitswitch = Some(var(&lv_eq));
    block_loop_valid.closeblock(vec![
        Link::new(
            vec![var(&lv_slot)],
            Some(graph.returnblock.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&lv_d),
                var(&lv_entries),
                var(&lv_indexes),
                var(&lv_mask),
                var(&lv_key),
                var(&lv_sf),
                var(&lv_perturb),
                var(&lv_i),
                var(&lv_ds),
            ],
            Some(block_perturb_shift.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ===== block_loop_deleted: record first deleted slot (deletedslot == -1). =====
    let ld_i_s = new_var("i_s", sig());
    push(
        &block_loop_deleted,
        "cast_uint_to_int",
        vec![var(&ld_i)],
        &ld_i_s,
    );
    let ld_ds_m1 = new_var("ds_is_m1", LowLevelType::Bool);
    push(
        &block_loop_deleted,
        "int_eq",
        vec![var(&ld_ds), signed(-1)],
        &ld_ds_m1,
    );
    block_loop_deleted.borrow_mut().exitswitch = Some(var(&ld_ds_m1));
    let ld_head = vec![
        var(&ld_d),
        var(&ld_entries),
        var(&ld_indexes),
        var(&ld_mask),
        var(&ld_key),
        var(&ld_sf),
        var(&ld_perturb),
        var(&ld_i),
    ];
    let mut ld_set_args = ld_head.clone();
    ld_set_args.push(var(&ld_i_s));
    let mut ld_keep_args = ld_head;
    ld_keep_args.push(var(&ld_ds));
    block_loop_deleted.closeblock(vec![
        Link::new(
            ld_set_args,
            Some(block_perturb_shift.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            ld_keep_args,
            Some(block_perturb_shift.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ===== block_perturb_shift: perturb >>= PERTURB_SHIFT; back-edge to loop body. =====
    let sh_perturb_new = new_var("perturb", uns());
    push(
        &block_perturb_shift,
        "uint_rshift",
        vec![var(&sh_perturb), signed(PERTURB_SHIFT)],
        &sh_perturb_new,
    );
    block_perturb_shift.closeblock(vec![
        Link::new(
            vec![
                var(&sh_d),
                var(&sh_entries),
                var(&sh_indexes),
                var(&sh_mask),
                var(&sh_key),
                var(&sh_sf),
                var(&sh_perturb_new),
                var(&sh_i),
                var(&sh_ds),
            ],
            Some(block_loop_body.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ===== block_loop_free: store at deletedslot iff FLAG_STORE, else -1. =====
    let lf_is_store = new_var("is_store", LowLevelType::Bool);
    push(
        &block_loop_free,
        "int_eq",
        vec![var(&lf_sf), signed(FLAG_STORE)],
        &lf_is_store,
    );
    block_loop_free.borrow_mut().exitswitch = Some(var(&lf_is_store));
    block_loop_free.closeblock(vec![
        Link::new(
            vec![var(&lf_d), var(&lf_indexes), var(&lf_i), var(&lf_ds)],
            Some(block_free_choose_slot.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![signed(-1)],
            Some(graph.returnblock.clone()),
            Some(bool_false()),
        )
        .into_ref(),
    ]);

    // ===== block_free_choose_slot: deletedslot==-1 ? i : deletedslot, then store. =====
    let fc_i_s = new_var("i_s", sig());
    push(
        &block_free_choose_slot,
        "cast_uint_to_int",
        vec![var(&fc_i)],
        &fc_i_s,
    );
    let fc_ds_m1 = new_var("ds_is_m1", LowLevelType::Bool);
    push(
        &block_free_choose_slot,
        "int_eq",
        vec![var(&fc_ds), signed(-1)],
        &fc_ds_m1,
    );
    block_free_choose_slot.borrow_mut().exitswitch = Some(var(&fc_ds_m1));
    block_free_choose_slot.closeblock(vec![
        Link::new(
            vec![var(&fc_d), var(&fc_indexes), var(&fc_i_s)],
            Some(block_store_at.clone()),
            Some(bool_true()),
        )
        .into_ref(),
        Link::new(
            vec![var(&fc_d), var(&fc_indexes), var(&fc_ds)],
            Some(block_store_at.clone()),
            Some(bool_false()),
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
        vec![
            "d".to_string(),
            "key".to_string(),
            "hash".to_string(),
            "store_flag".to_string(),
        ],
        func,
    ))
}

/// Synthesise `_ll_dictnext(iter) -> Signed` (`rordereddict.py:1229-1259`),
/// the forward dict-iterator step shared by keys/values/items iteration.
///
/// ```python
/// def _ll_dictnext(iter):
///     dict = iter.dict
///     if dict:
///         entries = dict.entries
///         index = iter.index
///         entries_len = dict.num_ever_used_items
///         while index < entries_len:
///             nextindex = index + 1
///             if entries.valid(index):
///                 iter.index = nextindex
///                 return index
///             else:
///                 if index == (dict.lookup_function_no >> FUNC_SHIFT):
///                     dict.lookup_function_no += (1 << FUNC_SHIFT)
///             index = nextindex
///         iter.dict = nullptr(...)   # clear the reference, prevent restarts
///     raise StopIteration
/// ```
///
/// `entries.valid(index)` is the simple-hash-eq `f_valid` flag
/// (`getinteriorfield(entries, index, "f_valid")`); the dummy-obj valid path
/// is out of scope. The `else` arm is the `popitem(last=False)` fast-forward
/// hack that bumps the high bits of `lookup_function_no` so a repeated
/// iteration over a shrinking prefix starts later; the `ll_assert` guarding
/// it is debug-only and omitted. Self-contained: no lookup, hash, or malloc —
/// a direct entries-array walk.
///
/// 8-block CFG plus the returnblock and exceptblock. Both the null-`dict`
/// guard and the loop-exhausted tail raise `StopIteration` via `exceptblock`.
pub fn build_ll_dictnext_helper_graph(
    name: &str,
    iter_ptr_lltype: LowLevelType,
    dict_ptr_lltype: LowLevelType,
    entries_ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let signed = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);
    let var = |v: &Variable| Hlvalue::Variable(v.clone());
    let null_dict = || {
        Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            dict_ptr_lltype.clone(),
        ))
    };
    let push = |block: &BlockRef, opname: &str, args: Vec<Hlvalue>, result: &Variable| {
        block.borrow_mut().operations.push(SpaceOperation::new(
            opname,
            args,
            Hlvalue::Variable(result.clone()),
        ));
    };
    let sig = || LowLevelType::Signed;
    let new_var = |n: &str, t: LowLevelType| variable_with_lltype(n, t);

    let exc_args = exception_args("StopIteration")?;

    // ---- startblock inputargs: (iter).
    let iter = new_var("iter", iter_ptr_lltype.clone());
    let startblock = Block::shared(vec![var(&iter)]);
    let return_var = new_var("result", sig());
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock.clone(), var(&return_var));

    // Pre-create the downstream blocks with fresh inputarg copies.
    // block_setup: (iter, dict).
    let su_iter = new_var("iter", iter_ptr_lltype.clone());
    let su_dict = new_var("dict", dict_ptr_lltype.clone());
    let block_setup = Block::shared(vec![var(&su_iter), var(&su_dict)]);

    // block_loop_cond: (iter, dict, entries, index, entries_len).
    let lc_iter = new_var("iter", iter_ptr_lltype.clone());
    let lc_dict = new_var("dict", dict_ptr_lltype.clone());
    let lc_entries = new_var("entries", entries_ptr_lltype.clone());
    let lc_index = new_var("index", sig());
    let lc_len = new_var("entries_len", sig());
    let block_loop_cond = Block::shared(vec![
        var(&lc_iter),
        var(&lc_dict),
        var(&lc_entries),
        var(&lc_index),
        var(&lc_len),
    ]);

    // block_loop_body: same shape as loop_cond.
    let lb_iter = new_var("iter", iter_ptr_lltype.clone());
    let lb_dict = new_var("dict", dict_ptr_lltype.clone());
    let lb_entries = new_var("entries", entries_ptr_lltype.clone());
    let lb_index = new_var("index", sig());
    let lb_len = new_var("entries_len", sig());
    let block_loop_body = Block::shared(vec![
        var(&lb_iter),
        var(&lb_dict),
        var(&lb_entries),
        var(&lb_index),
        var(&lb_len),
    ]);

    // block_return_valid: (iter, nextindex, index).
    let rv_iter = new_var("iter", iter_ptr_lltype.clone());
    let rv_nextindex = new_var("nextindex", sig());
    let rv_index = new_var("index", sig());
    let block_return_valid = Block::shared(vec![var(&rv_iter), var(&rv_nextindex), var(&rv_index)]);

    // block_invalid: (iter, dict, entries, nextindex, entries_len, index).
    let iv_iter = new_var("iter", iter_ptr_lltype.clone());
    let iv_dict = new_var("dict", dict_ptr_lltype.clone());
    let iv_entries = new_var("entries", entries_ptr_lltype.clone());
    let iv_nextindex = new_var("nextindex", sig());
    let iv_len = new_var("entries_len", sig());
    let iv_index = new_var("index", sig());
    let block_invalid = Block::shared(vec![
        var(&iv_iter),
        var(&iv_dict),
        var(&iv_entries),
        var(&iv_nextindex),
        var(&iv_len),
        var(&iv_index),
    ]);

    // block_bump: (iter, dict, entries, nextindex, entries_len, lfn).
    let bp_iter = new_var("iter", iter_ptr_lltype.clone());
    let bp_dict = new_var("dict", dict_ptr_lltype.clone());
    let bp_entries = new_var("entries", entries_ptr_lltype.clone());
    let bp_nextindex = new_var("nextindex", sig());
    let bp_len = new_var("entries_len", sig());
    let bp_lfn = new_var("lfn", sig());
    let block_bump = Block::shared(vec![
        var(&bp_iter),
        var(&bp_dict),
        var(&bp_entries),
        var(&bp_nextindex),
        var(&bp_len),
        var(&bp_lfn),
    ]);

    // block_clear: (iter).
    let cl_iter = new_var("iter", iter_ptr_lltype.clone());
    let block_clear = Block::shared(vec![var(&cl_iter)]);

    // ===== startblock: dict = iter.dict; if dict raise-guard. =====
    let dict0 = new_var("dict", dict_ptr_lltype.clone());
    push(
        &startblock,
        "getfield",
        vec![var(&iter), void_field_const("dict")],
        &dict0,
    );
    let nz = new_var("nz", LowLevelType::Bool);
    push(&startblock, "ptr_nonzero", vec![var(&dict0)], &nz);
    startblock.borrow_mut().exitswitch = Some(var(&nz));
    startblock.closeblock(vec![
        Link::new(
            vec![var(&iter), var(&dict0)],
            Some(block_setup.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    // ===== block_setup: read entries, index, entries_len; enter loop. =====
    let su_entries = new_var("entries", entries_ptr_lltype.clone());
    push(
        &block_setup,
        "getfield",
        vec![var(&su_dict), void_field_const("entries")],
        &su_entries,
    );
    let su_index = new_var("index", sig());
    push(
        &block_setup,
        "getfield",
        vec![var(&su_iter), void_field_const("index")],
        &su_index,
    );
    let su_len = new_var("entries_len", sig());
    push(
        &block_setup,
        "getfield",
        vec![var(&su_dict), void_field_const("num_ever_used_items")],
        &su_len,
    );
    block_setup.closeblock(vec![
        Link::new(
            vec![
                var(&su_iter),
                var(&su_dict),
                var(&su_entries),
                var(&su_index),
                var(&su_len),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ===== block_loop_cond: while index < entries_len. =====
    let lt = new_var("lt", LowLevelType::Bool);
    push(
        &block_loop_cond,
        "int_lt",
        vec![var(&lc_index), var(&lc_len)],
        &lt,
    );
    block_loop_cond.borrow_mut().exitswitch = Some(var(&lt));
    block_loop_cond.closeblock(vec![
        Link::new(
            vec![
                var(&lc_iter),
                var(&lc_dict),
                var(&lc_entries),
                var(&lc_index),
                var(&lc_len),
            ],
            Some(block_loop_body.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![var(&lc_iter)],
            Some(block_clear.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    // ===== block_loop_body: nextindex = index + 1; branch on entries.valid. =====
    let lb_next = new_var("nextindex", sig());
    push(
        &block_loop_body,
        "int_add",
        vec![var(&lb_index), signed(1)],
        &lb_next,
    );
    let lb_valid = new_var("valid", LowLevelType::Bool);
    push(
        &block_loop_body,
        "getinteriorfield",
        vec![
            var(&lb_entries),
            var(&lb_index),
            void_field_const("f_valid"),
        ],
        &lb_valid,
    );
    block_loop_body.borrow_mut().exitswitch = Some(var(&lb_valid));
    block_loop_body.closeblock(vec![
        Link::new(
            vec![var(&lb_iter), var(&lb_next), var(&lb_index)],
            Some(block_return_valid.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&lb_iter),
                var(&lb_dict),
                var(&lb_entries),
                var(&lb_next),
                var(&lb_len),
                var(&lb_index),
            ],
            Some(block_invalid.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    // ===== block_return_valid: iter.index = nextindex; return index. =====
    let rv_void = new_var("v", LowLevelType::Void);
    push(
        &block_return_valid,
        "setfield",
        vec![var(&rv_iter), void_field_const("index"), var(&rv_nextindex)],
        &rv_void,
    );
    block_return_valid.closeblock(vec![
        Link::new(vec![var(&rv_index)], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    // ===== block_invalid: popitem(last=False) fast-forward hack. =====
    let iv_lfn = new_var("lfn", sig());
    push(
        &block_invalid,
        "getfield",
        vec![var(&iv_dict), void_field_const("lookup_function_no")],
        &iv_lfn,
    );
    let iv_shifted = new_var("shifted", sig());
    push(
        &block_invalid,
        "int_rshift",
        vec![var(&iv_lfn), signed(FUNC_SHIFT)],
        &iv_shifted,
    );
    let iv_eq = new_var("is_head", LowLevelType::Bool);
    push(
        &block_invalid,
        "int_eq",
        vec![var(&iv_index), var(&iv_shifted)],
        &iv_eq,
    );
    block_invalid.borrow_mut().exitswitch = Some(var(&iv_eq));
    block_invalid.closeblock(vec![
        Link::new(
            vec![
                var(&iv_iter),
                var(&iv_dict),
                var(&iv_entries),
                var(&iv_nextindex),
                var(&iv_len),
                var(&iv_lfn),
            ],
            Some(block_bump.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![
                var(&iv_iter),
                var(&iv_dict),
                var(&iv_entries),
                var(&iv_nextindex),
                var(&iv_len),
            ],
            Some(block_loop_cond.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    // ===== block_bump: lookup_function_no += (1 << FUNC_SHIFT); loop back. =====
    let bp_new_lfn = new_var("lfn", sig());
    push(
        &block_bump,
        "int_add",
        vec![var(&bp_lfn), signed(1 << FUNC_SHIFT)],
        &bp_new_lfn,
    );
    let bp_void = new_var("v", LowLevelType::Void);
    push(
        &block_bump,
        "setfield",
        vec![
            var(&bp_dict),
            void_field_const("lookup_function_no"),
            var(&bp_new_lfn),
        ],
        &bp_void,
    );
    block_bump.closeblock(vec![
        Link::new(
            vec![
                var(&bp_iter),
                var(&bp_dict),
                var(&bp_entries),
                var(&bp_nextindex),
                var(&bp_len),
            ],
            Some(block_loop_cond.clone()),
            None,
        )
        .into_ref(),
    ]);

    // ===== block_clear: iter.dict = nullptr; raise StopIteration. =====
    let cl_void = new_var("v", LowLevelType::Void);
    push(
        &block_clear,
        "setfield",
        vec![var(&cl_iter), void_field_const("dict"), null_dict()],
        &cl_void,
    );
    block_clear.closeblock(vec![
        Link::new(exc_args, Some(graph.exceptblock.clone()), None).into_ref(),
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

pub fn ll_call_insert_clean_function() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "ll_call_insert_clean_function",
    ))
}

pub fn ll_call_delete_by_entry_index() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "ll_call_delete_by_entry_index",
    ))
}

pub fn ll_valid_from_flag() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_valid_from_flag"))
}

pub fn ll_valid_from_key() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_valid_from_key"))
}

pub fn ll_valid_from_value() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_valid_from_value"))
}

pub fn ll_mark_deleted_in_flag() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_mark_deleted_in_flag"))
}

pub fn ll_mark_deleted_in_key() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_mark_deleted_in_key"))
}

pub fn ll_mark_deleted_in_value() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_mark_deleted_in_value"))
}

pub fn ll_hash_from_cache() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_hash_from_cache"))
}

pub fn ll_hash_recomputed() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_hash_recomputed"))
}

pub fn ll_hash_custom_fast() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_hash_custom_fast"))
}

pub fn ll_keyhash_custom() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_keyhash_custom"))
}

pub fn ll_keyeq_custom() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_keyeq_custom"))
}

pub fn ll_dict_len() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_len"))
}

pub fn ll_dict_bool() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_bool"))
}

pub fn ll_dict_getitem() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_getitem"))
}

pub fn ll_dict_getitem_with_hash() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_getitem_with_hash"))
}

pub fn ll_dict_setitem() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_setitem"))
}

pub fn ll_dict_setitem_with_hash() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_setitem_with_hash"))
}

pub fn _ll_dict_setitem_lookup_done() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "_ll_dict_setitem_lookup_done",
    ))
}

pub fn _ll_dict_rescue() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_dict_rescue"))
}

pub fn _ll_dict_insert_no_index() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_dict_insert_no_index"))
}

pub fn ll_len_of_d_indexes() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_len_of_d_indexes"))
}

pub fn _ll_len_of_d_indexes() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_len_of_d_indexes"))
}

/// RPython `_overallocate_entries_len` (`rordereddict.py:748-757`).
pub fn _overallocate_entries_len(baselen: usize) -> usize {
    let newsize = baselen + (baselen >> 3);
    newsize + 8
}

pub fn ll_dict_grow() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_grow"))
}

pub fn _ll_dict_entries_size_too_big() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "_ll_dict_entries_size_too_big",
    ))
}

pub fn ll_dict_remove_deleted_items() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "ll_dict_remove_deleted_items",
    ))
}

pub fn ll_dict_delitem() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_delitem"))
}

pub fn ll_dict_delitem_with_hash() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_delitem_with_hash"))
}

pub fn ll_dict_delitem_if_value_is() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_delitem_if_value_is"))
}

pub fn _ll_dict_del_entry() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_dict_del_entry"))
}

pub fn _ll_dict_del() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_dict_del"))
}

pub fn ll_dict_resize() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_resize"))
}

pub fn _ll_dict_resize_to() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_dict_resize_to"))
}

pub fn ll_ensure_indexes() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_ensure_indexes"))
}

pub fn ll_dict_create_initial_index() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "ll_dict_create_initial_index",
    ))
}

pub fn ll_dict_rehash_after_translation() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "ll_dict_rehash_after_translation",
    ))
}

pub fn ll_dict_reindex() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_reindex"))
}

/// RPython `_ll_ptr_to_array_of(T)` (`rordereddict.py:1033-1035`).
pub fn _ll_ptr_to_array_of(T: LowLevelType) -> LowLevelType {
    ptr_to_gc_array(T)
}

pub fn ll_dict_store_clean() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_store_clean"))
}

pub fn ll_dict_delete_by_entry_index() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "ll_dict_delete_by_entry_index",
    ))
}

pub fn _ll_empty_array() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_empty_array"))
}

pub fn ll_newdict() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_newdict"))
}

pub fn _ll_malloc_dict() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_malloc_dict"))
}

pub fn _ll_malloc_entries() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_malloc_entries"))
}

pub fn _ll_free_entries() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_free_entries"))
}

pub fn ll_dictiter_reversed() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dictiter_reversed"))
}

pub fn _ll_dictnext_reversed() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_dictnext_reversed"))
}

pub fn ll_dict_get() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_get"))
}

pub fn ll_dict_setdefault() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_setdefault"))
}

pub fn ll_dict_copy() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_copy"))
}

pub fn ll_dict_clear() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_clear"))
}

pub fn ll_dict_update() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_update"))
}

pub fn ll_prepare_dict_update() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_prepare_dict_update"))
}

pub fn recast() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("recast"))
}

pub fn _make_ll_keys_values_items() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_make_ll_keys_values_items"))
}

pub fn ll_dict_keys() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_keys"))
}

pub fn ll_dict_values() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_values"))
}

pub fn ll_dict_items() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_items"))
}

pub fn ll_dict_contains() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_contains"))
}

pub fn ll_dict_contains_with_hash() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_contains_with_hash"))
}

pub fn _ll_getnextitem() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("_ll_getnextitem"))
}

pub fn ll_dict_popitem() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_popitem"))
}

pub fn ll_dict_pop() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_pop"))
}

pub fn ll_dict_pop_default() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_pop_default"))
}

pub fn ll_dict_move_to_end() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_move_to_end"))
}

pub fn ll_dict_move_to_last() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_move_to_last"))
}

pub fn ll_dict_move_to_first() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred("ll_dict_move_to_first"))
}

pub fn _ll_dict_move_to_first_shift_items() -> Result<(), TyperError> {
    Err(ordered_dict_runtime_deferred(
        "_ll_dict_move_to_first_shift_items",
    ))
}

/// RPython `get_ll_dictiter(DICTPTR)` (`rordereddict.py:1187-1190`).
pub fn get_ll_dictiter(DICTPTR: LowLevelType) -> LowLevelType {
    let dictiter = Struct::gc(
        "dictiter",
        vec![
            ("dict".into(), DICTPTR),
            ("index".into(), LowLevelType::Signed),
        ],
    );
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(dictiter),
    }))
}

/// RPython `class DictIteratorRepr(AbstractDictIteratorRepr)`
/// (`lltypesystem/rordereddict.py:1192`).
#[derive(Debug)]
pub struct DictIteratorRepr {
    pub base: AbstractDictIteratorRepr,
    pub r_dict_lowleveltype: LowLevelType,
}

impl DictIteratorRepr {
    pub fn new(r_dict_lowleveltype: LowLevelType, variant: impl Into<String>) -> Self {
        let variant = variant.into();
        let lowleveltype = get_ll_dictiter(r_dict_lowleveltype.clone());
        DictIteratorRepr {
            base: AbstractDictIteratorRepr::new(lowleveltype, vec![variant]),
            r_dict_lowleveltype,
        }
    }
}

impl Repr for DictIteratorRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        self.base.lowleveltype()
    }

    fn state(&self) -> &ReprState {
        self.base.state()
    }

    fn class_name(&self) -> &'static str {
        "DictIteratorRepr"
    }

    fn repr_class_id(&self) -> crate::translator::rtyper::pairtype::ReprClassId {
        crate::translator::rtyper::pairtype::ReprClassId::DictIteratorRepr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::dictdef::DictDef;
    use crate::annotator::model::{SomeInteger, SomeString, SomeValue};
    use crate::translator::rtyper::pairtype::ReprClassId;
    use crate::translator::rtyper::rint::signed_repr;
    use crate::translator::rtyper::rstr::string_repr;

    #[test]
    fn ordereddictrepr_builds_sparse_index_dicttable_shape() {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );

        let repr = OrderedDictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("ordered dict repr");

        assert_eq!(repr.repr_class_id(), ReprClassId::OrderedDictRepr);
        assert_eq!(repr.DICTENTRY._name, "odictentry");
        assert_eq!(
            repr.DICTENTRY._names,
            vec!["key", "f_valid", "value", "f_hash"]
        );
        assert_eq!(
            repr.DICT._names,
            vec![
                "num_live_items",
                "num_ever_used_items",
                "resize_counter",
                "indexes",
                "lookup_function_no",
                "entries"
            ]
        );
        assert_eq!(
            ReprClassId::OrderedDictRepr.mro(),
            &[
                ReprClassId::OrderedDictRepr,
                ReprClassId::AbstractDictRepr,
                ReprClassId::Repr
            ]
        );
    }

    fn ptr_gcarray_of(value: &LowLevelType) -> &Array {
        let LowLevelType::Ptr(ptr) = value else {
            panic!("expected Ptr(GcArray), got {value:?}");
        };
        let PtrTarget::Array(array) = &ptr.TO else {
            panic!("expected Ptr(GcArray), got {value:?}");
        };
        array
    }

    #[test]
    fn lookup_constants_match_64bit_upstream_layout() {
        assert!(IS_64BIT, "test host is expected to match PyPy's 64-bit arm");
        assert_eq!(FUNC_SHIFT, 3);
        assert_eq!(FUNC_MASK, 0x07);
        assert_eq!(FUNC_BYTE, 0);
        assert_eq!(FUNC_SHORT, 1);
        assert_eq!(FUNC_INT, 2);
        assert_eq!(FUNC_LONG, 3);
        assert_eq!(FUNC_MUST_REINDEX, 4);
        assert_eq!(PERTURB_SHIFT, 5);
        assert_eq!(FREE, 0);
        assert_eq!(DELETED, 1);
        assert_eq!(VALID_OFFSET, 2);
        assert_eq!(MIN_INDEXES_MINUS_ENTRIES, 3);
        assert_eq!(FLAG_LOOKUP, 0);
        assert_eq!(FLAG_STORE, 1);
        assert_eq!(DICT_INITSIZE, 16);
    }

    #[test]
    fn dictindex_aliases_keep_ptr_gcarray_shape() {
        for alias in [
            &*DICTINDEX_BYTE,
            &*DICTINDEX_SHORT,
            &*DICTINDEX_INT,
            &*DICTINDEX_LONG,
        ] {
            assert_eq!(ptr_gcarray_of(alias).OF, LowLevelType::Unsigned);
        }
        assert_eq!(TYPE_BYTE, LowLevelType::Unsigned);
        assert_eq!(TYPE_SHORT, LowLevelType::Unsigned);
        assert_eq!(TYPE_INT, LowLevelType::Unsigned);
        assert_eq!(TYPE_LONG, LowLevelType::Unsigned);
    }

    #[test]
    fn runtime_helper_surface_is_explicitly_deferred() {
        let err = ll_call_lookup_function().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_call_lookup_function"));

        let err = ll_dict_len().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_dict_len"));

        let err = ll_dict_getitem().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_dict_getitem"));

        let err = ll_dict_keys().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_dict_keys"));

        let err = ll_dict_move_to_first().expect_err("runtime helper deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_dict_move_to_first"));
    }

    #[test]
    fn overallocate_entries_len_matches_upstream_growth_pattern() {
        let got: Vec<_> = [0, 8, 17, 27, 38, 50, 64, 80]
            .into_iter()
            .map(_overallocate_entries_len)
            .collect();
        assert_eq!(got, vec![8, 17, 27, 38, 50, 64, 80, 98]);
    }

    #[test]
    fn get_ll_dictiter_builds_dict_and_index_fields() {
        let dictptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(Struct::gc("dicttable", vec![])),
        }));
        let iterptr = get_ll_dictiter(dictptr.clone());
        let LowLevelType::Ptr(ptr) = iterptr else {
            panic!("expected Ptr(GcStruct), got {iterptr:?}");
        };
        let PtrTarget::Struct(iter) = &ptr.TO else {
            panic!("expected Ptr(GcStruct), got {ptr:?}");
        };
        assert_eq!(iter._name, "dictiter");
        assert_eq!(iter._names, vec!["dict", "index"]);
        assert_eq!(iter._flds.get("dict"), Some(&dictptr));
        assert_eq!(iter._flds.get("index"), Some(&LowLevelType::Signed));
    }

    #[test]
    fn dictiteratorrepr_extends_abstract_iterator_repr() {
        let dictptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(Struct::gc("dicttable", vec![])),
        }));
        let repr = DictIteratorRepr::new(dictptr.clone(), "keys");

        assert_eq!(repr.class_name(), "DictIteratorRepr");
        assert_eq!(repr.repr_class_id(), ReprClassId::DictIteratorRepr);
        assert_eq!(repr.base.variant, vec!["keys".to_string()]);
        assert_eq!(repr.r_dict_lowleveltype, dictptr);
        assert_eq!(
            ReprClassId::DictIteratorRepr.mro(),
            &[
                ReprClassId::DictIteratorRepr,
                ReprClassId::AbstractDictIteratorRepr,
                ReprClassId::Repr
            ]
        );
    }

    fn sample_dict_ptr_lltype() -> LowLevelType {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let repr = OrderedDictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("ordered dict repr");
        repr.lowleveltype().clone()
    }

    /// `ll_dict_len` is a one-block graph reading the `num_live_items`
    /// header field and returning it as `Signed`.
    #[test]
    fn build_ll_dict_len_reads_num_live_items_field() {
        let helper = build_ll_dict_len_helper_graph("ll_dict_len", sample_dict_ptr_lltype())
            .expect("build_ll_dict_len_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_len");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getfield"]);
        let field = &startblock.operations[0].args[1];
        assert!(
            matches!(field, Hlvalue::Constant(c) if c.value == ConstValue::byte_str("num_live_items")),
            "len helper must read num_live_items, got {field:?}"
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "ll_dict_len returns Signed"
        );
    }

    /// `ll_dict_bool` branches on `ptr_nonzero(d)`: the False arm returns the
    /// `False` constant without dereferencing the receiver; the True arm
    /// forwards `d` into a block that reads `num_live_items` and compares it
    /// `!= 0`.
    #[test]
    fn build_ll_dict_bool_guards_null_then_checks_num_live_items() {
        let helper = build_ll_dict_bool_helper_graph("ll_dict_bool", sample_dict_ptr_lltype())
            .expect("build_ll_dict_bool_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_bool");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();

        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["ptr_nonzero"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);

        // False arm: returns the False constant straight to the returnblock.
        let false_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit link present");
        let false_first_arg = false_link
            .borrow()
            .args
            .first()
            .and_then(|opt| opt.as_ref())
            .cloned()
            .expect("False link first arg present");
        assert!(matches!(
            false_first_arg,
            Hlvalue::Constant(c) if c.value == ConstValue::Bool(false)
        ));

        // True arm: forwards `d` into check_len = getfield(num_live_items) +
        // int_ne(n, 0).
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let check_len = true_link
            .borrow()
            .target
            .clone()
            .expect("True link target block");
        let check_ops: Vec<String> = check_len
            .borrow()
            .operations
            .iter()
            .map(|op| op.opname.clone())
            .collect();
        assert_eq!(check_ops, vec!["getfield", "int_ne"]);
        let getfield_field = check_len.borrow().operations[0].args[1].clone();
        assert!(
            matches!(getfield_field, Hlvalue::Constant(c) if c.value == ConstValue::byte_str("num_live_items")),
            "bool helper must read num_live_items"
        );

        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Bool),
            "ll_dict_bool returns Bool"
        );
    }

    /// `_ll_write_indexes` is a single-block store: getfield(d,"indexes") ->
    /// cast_pointer (GCREF->INDEXES) -> cast_int_to_uint(value) ->
    /// setarrayitem(indexes, i, cast_value), Void return.
    #[test]
    fn build_ll_write_indexes_casts_gcref_then_stores_slot() {
        let helper = build_ll_write_indexes_helper_graph(
            "_ll_write_indexes",
            sample_dict_ptr_lltype(),
            LowLevelType::Unsigned,
        )
        .expect("build_ll_write_indexes_helper_graph");
        assert_eq!(helper.func.name, "_ll_write_indexes");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 3); // d, i, value
        let ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            ops,
            vec![
                "getfield",
                "cast_pointer",
                "cast_int_to_uint",
                "setarrayitem"
            ]
        );
        let field = &startblock.operations[0].args[1];
        assert!(
            matches!(field, Hlvalue::Constant(c) if c.value == ConstValue::byte_str("indexes")),
            "first op must read the indexes field, got {field:?}"
        );
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(ret.concretetype.borrow().clone(), Some(LowLevelType::Void));
    }

    /// Int-keyed sample `(dict_ptr, entries_ptr, key_lltype)` for the lookup
    /// builder, derived from a freshly built `OrderedDictRepr`.
    fn sample_dict_lookup_lltypes() -> (LowLevelType, LowLevelType, LowLevelType) {
        let ann = Rc::new(RPythonAnnotator::new(None, None, None, false));
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().expect("rtyper init");
        let dictdef = DictDef::new(
            None,
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let repr = OrderedDictRepr::new(
            rtyper,
            signed_repr() as Arc<dyn Repr>,
            string_repr() as Arc<dyn Repr>,
            dictdef,
            None,
            false,
            false,
        )
        .expect("ordered dict repr");
        let entries_ptr = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(repr.DICTENTRYARRAY.clone()),
        }));
        (
            repr.lowleveltype().clone(),
            entries_ptr,
            repr.DICTKEY.clone(),
        )
    }

    /// Walk every block reachable from `start`, returning the visited block
    /// count and the flattened op-name list.
    fn walk_blocks(start: &BlockRef) -> (usize, Vec<String>) {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![start.clone()];
        let mut ops = Vec::new();
        let mut count = 0usize;
        while let Some(b) = stack.pop() {
            if !seen.insert(Rc::as_ptr(&b) as usize) {
                continue;
            }
            count += 1;
            let bb = b.borrow();
            for op in &bb.operations {
                ops.push(op.opname.clone());
            }
            for link in &bb.exits {
                if let Some(t) = link.borrow().target.clone() {
                    stack.push(t);
                }
            }
        }
        (count, ops)
    }

    /// `ll_dict_lookup` is the open-addressing perturb-probe. Validate the
    /// first-probe header, the full 13-block + returnblock CFG shape, the
    /// unsigned probe arithmetic (logical `uint_rshift` for the perturb
    /// shift), the interior key read, the inlined store, and the Signed slot
    /// return.
    #[test]
    fn build_ll_dict_lookup_assembles_perturb_probe_cfg() {
        let (dict_ptr, entries_ptr, key_lltype) = sample_dict_lookup_lltypes();
        let helper = build_ll_dict_lookup_helper_graph(
            "ll_dict_lookup",
            dict_ptr,
            entries_ptr,
            LowLevelType::Unsigned,
            key_lltype,
        )
        .expect("build_ll_dict_lookup_helper_graph");
        assert_eq!(helper.func.name, "ll_dict_lookup");
        let inner = helper.graph.borrow();

        // First-probe header: read entries + indexes, cast the GCREF index
        // array, derive mask, hash & mask, read+cast the slot, branch on
        // index >= VALID_OFFSET.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            start_ops,
            vec![
                "getfield",         // d.entries
                "getfield",         // d.indexes (GCREF)
                "cast_pointer",     // cast_opaque_ptr(INDEXES, ...)
                "getarraysize",     // len(indexes)
                "int_sub",          // mask = len - 1
                "cast_int_to_uint", // mask_u
                "int_and",          // hash & mask
                "cast_int_to_uint", // i = r_uint(...)
                "cast_uint_to_int", // intmask(i)
                "getarrayitem",     // indexes[intmask(i)]
                "cast_uint_to_int", // rffi.cast(Signed, ...)
                "int_ge",           // index >= VALID_OFFSET
            ]
        );
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        // 13 work blocks + the returnblock are all reachable.
        let (block_count, ops) = walk_blocks(&inner.startblock);
        assert_eq!(block_count, 14, "13 work blocks + returnblock");

        // Distinctive ops of the probe must all appear.
        for needed in [
            "uint_lshift",      // i << 2
            "uint_add",         // + i / + perturb / + 1
            "uint_and",         // & mask
            "uint_rshift",      // perturb >>= PERTURB_SHIFT (logical!)
            "getinteriorfield", // entries[slot].key
            "int_eq",           // FREE / DELETED / FLAG_STORE / key (Signed)
            "setarrayitem",     // inlined _ll_write_indexes store
        ] {
            assert!(
                ops.iter().any(|o| o == needed),
                "lookup CFG must emit {needed}, got {ops:?}"
            );
        }

        // Returns the Signed entry slot (index - VALID_OFFSET) or -1.
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "ll_dict_lookup returns Signed"
        );
    }

    /// `_ll_dictnext` walks the entries array; on a valid entry it advances
    /// `iter.index` and returns the slot, on exhaustion it nulls `iter.dict`
    /// and raises StopIteration. Validate the null-guard header, the 8-block +
    /// return + except CFG, the `f_valid` interior read, the field writes, the
    /// popitem fast-forward hack ops, and the Signed return.
    #[test]
    fn build_ll_dictnext_walks_entries_then_raises_stopiteration() {
        let (dict_ptr, entries_ptr, _key) = sample_dict_lookup_lltypes();
        let iter_ptr = get_ll_dictiter(dict_ptr.clone());
        let helper =
            build_ll_dictnext_helper_graph("_ll_dictnext", iter_ptr, dict_ptr, entries_ptr)
                .expect("build_ll_dictnext_helper_graph");
        assert_eq!(helper.func.name, "_ll_dictnext");
        let inner = helper.graph.borrow();

        // Null-guard header: dict = iter.dict; ptr_nonzero(dict); branch.
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["getfield", "ptr_nonzero"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);
        drop(startblock);

        // 8 work blocks + returnblock + exceptblock all reachable.
        let (block_count, ops) = walk_blocks(&inner.startblock);
        assert_eq!(block_count, 10, "8 work blocks + returnblock + exceptblock");

        for needed in [
            "getinteriorfield", // entries.valid(index) = f_valid
            "setfield",         // iter.index = nextindex / iter.dict = null / lfn bump
            "int_lt",           // index < entries_len
            "int_add",          // nextindex / lfn += (1 << FUNC_SHIFT)
            "int_rshift",       // lookup_function_no >> FUNC_SHIFT
            "int_eq",           // index == (lookup_function_no >> FUNC_SHIFT)
            "ptr_nonzero",      // if dict
        ] {
            assert!(
                ops.iter().any(|o| o == needed),
                "dictnext CFG must emit {needed}, got {ops:?}"
            );
        }

        // Returns the Signed entry index.
        let Hlvalue::Variable(ret) = &inner.returnblock.borrow().inputargs[0] else {
            panic!("returnblock inputarg must be a Variable");
        };
        assert_eq!(
            ret.concretetype.borrow().clone(),
            Some(LowLevelType::Signed),
            "_ll_dictnext returns Signed"
        );
    }
}
