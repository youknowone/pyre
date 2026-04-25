//! RPython `rpython/rtyper/rtuple.py` — `TupleRepr` for RPython tuples.
//!
//! Upstream rtuple.py (414 LOC) covers:
//! * `TUPLE_TYPE(field_lltypes)` (rtuple.py:119-126) — Void for empty
//!   tuples, `Ptr(GcStruct('tuple%d', ('item0', T0), ...))` otherwise.
//! * `class TupleRepr(Repr)` (rtuple.py:129+) — items_r,
//!   external_items_r, fieldnames, lltypes, tuple_cache, lowleveltype.
//! * `getitem` / `getitem_internal` (rtuple.py:144-150).
//! * `newtuple` / `newtuple_cached` / `_rtype_newtuple` (rtuple.py:153-182).
//! * `convert_const` / `instantiate` (rtuple.py:184-204).
//! * pair-type / iterator / hash / eq / str (rtuple.py:200-414).
//!
//! This file lands the **minimal slice** required to wire
//! [`SomeTuple.rtyper_makerepr`] (rmodel.rs) to a real repr instead of
//! `MissingRTypeOperation`. Concretely:
//!
//! | upstream | Rust mirror |
//! |---|---|
//! | `TUPLE_TYPE` (rtuple.py:119-126) | [`tuple_type`] |
//! | `TupleRepr.__init__` (rtuple.py:131-142) | [`TupleRepr::new`] |
//! | `TupleRepr.lowleveltype` | [`Repr::lowleveltype`] impl |
//! | `convert_const(())` empty-tuple Void arm | [`Repr::convert_const`] |
//!
//! Methods that emit ops via `llops` (`getitem` / `newtuple` /
//! `instantiate`-driven non-empty `convert_const`), the tuple_cache,
//! pair-type conversions, and rtype_* dispatchers land in follow-up
//! commits.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::flowspace::model::{ConstValue, Constant, Hlvalue, Variable};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    self, _ptr, LowLevelType, MallocFlavor, Ptr, PtrTarget, StructType,
};
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{Repr, ReprState};
use crate::translator::rtyper::rtyper::{GenopResult, LowLevelOpList, RPythonTyper};

/// RPython `TUPLE_TYPE(field_lltypes)` (rtuple.py:119-126).
///
/// ```python
/// def TUPLE_TYPE(field_lltypes):
///     if len(field_lltypes) == 0:
///         return Void      # empty tuple
///     else:
///         fields = [('item%d' % i, TYPE) for i, TYPE in enumerate(field_lltypes)]
///         kwds = {'hints': {'immutable': True, 'noidentity': True}}
///         return Ptr(GcStruct('tuple%d' % len(field_lltypes), *fields, **kwds))
/// ```
pub fn tuple_type(field_lltypes: &[LowLevelType]) -> LowLevelType {
    if field_lltypes.is_empty() {
        return LowLevelType::Void;
    }
    let n = field_lltypes.len();
    let name = format!("tuple{n}");
    let fields = field_lltypes
        .iter()
        .enumerate()
        .map(|(i, t)| (format!("item{i}"), t.clone()))
        .collect();
    let body = StructType::gc_with_hints(
        &name,
        fields,
        vec![
            ("immutable".into(), ConstValue::Bool(true)),
            ("noidentity".into(), ConstValue::Bool(true)),
        ],
    );
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(body),
    }))
}

/// RPython `pairtype(TupleRepr, TupleRepr).rtype_is_` (rtuple.py:355-356):
///
/// ```python
/// def rtype_is_((robj1, robj2), hop):
///     raise TyperError("cannot compare tuples with 'is'")
/// ```
///
/// Eagerly rejected by the rtyper so the generic `(Repr,
/// Repr).rtype_is_` arm cannot fall through to a `ptr_eq` on tuple
/// pointers — upstream considers identity comparison on tuple
/// values structurally meaningless. Both args are accepted as
/// `&dyn Repr` even though only the type name matters; the parity
/// of the error message preserves caller-visible behaviour.
pub fn pair_tuple_tuple_rtype_is_(
    _r1: &dyn Repr,
    _r2: &dyn Repr,
    _hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> Result<Option<Hlvalue>, TyperError> {
    Err(TyperError::message("cannot compare tuples with 'is'"))
}

/// RPython `pairtype(TupleRepr, TupleRepr).convert_from_to`
/// (rtuple.py:340-353):
///
/// ```python
/// def convert_from_to((r_from, r_to), v, llops):
///     if len(r_from.items_r) == len(r_to.items_r):
///         if r_from.lowleveltype == r_to.lowleveltype:
///             return v
///         n = len(r_from.items_r)
///         items_v = []
///         for i in range(n):
///             item_v = r_from.getitem_internal(llops, v, i)
///             item_v = llops.convertvar(item_v,
///                                           r_from.items_r[i],
///                                           r_to.items_r[i])
///             items_v.append(item_v)
///         return r_from.newtuple(llops, r_to, items_v)
///     return NotImplemented
/// ```
///
/// Same-arity tuple-to-tuple conversion via per-position
/// `getitem_internal` + `convertvar` then `newtuple` on the
/// destination repr. Different-arity returns `Ok(None)` (upstream's
/// `NotImplemented`).
pub fn pair_tuple_tuple_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let any_from: &dyn std::any::Any = r_from;
    let r_from_t = any_from.downcast_ref::<TupleRepr>().ok_or_else(|| {
        TyperError::message("pair_tuple_tuple_convert_from_to: r_from is not a TupleRepr")
    })?;
    let any_to: &dyn std::any::Any = r_to;
    let r_to_t = any_to.downcast_ref::<TupleRepr>().ok_or_else(|| {
        TyperError::message("pair_tuple_tuple_convert_from_to: r_to is not a TupleRepr")
    })?;
    // upstream rtuple.py:341 — different arity → NotImplemented.
    if r_from_t.items_r.len() != r_to_t.items_r.len() {
        return Ok(None);
    }
    // upstream rtuple.py:342-343 — same lowleveltype is identity.
    if r_from_t.lowleveltype() == r_to_t.lowleveltype() {
        return Ok(Some(v.clone()));
    }
    // upstream rtuple.py:344-351 — per-item getitem_internal +
    // convertvar to the matching destination items_r position.
    let n = r_from_t.items_r.len();
    let mut items_v: Vec<Hlvalue> = Vec::with_capacity(n);
    for i in 0..n {
        let v_internal = r_from_t.getitem_internal(llops, v.clone(), i)?;
        let item_v = llops.convertvar(
            Hlvalue::Variable(v_internal),
            r_from_t.items_r[i].as_ref(),
            r_to_t.items_r[i].as_ref(),
        )?;
        items_v.push(item_v);
    }
    // upstream rtuple.py:352 — `r_from.newtuple(llops, r_to, items_v)`.
    let result = TupleRepr::newtuple(llops, r_to_t, items_v)?;
    Ok(Some(result))
}

/// RPython `pairtype(TupleRepr, TupleRepr).rtype_add` (rtuple.py:319-327):
///
/// ```python
/// def rtype_add((r_tup1, r_tup2), hop):
///     v_tuple1, v_tuple2 = hop.inputargs(r_tup1, r_tup2)
///     vlist = []
///     for i in range(len(r_tup1.items_r)):
///         vlist.append(r_tup1.getitem_internal(hop.llops, v_tuple1, i))
///     for i in range(len(r_tup2.items_r)):
///         vlist.append(r_tup2.getitem_internal(hop.llops, v_tuple2, i))
///     return r_tup1.newtuple_cached(hop, vlist)
/// rtype_inplace_add = rtype_add
/// ```
///
/// Concatenates two tuples by emitting per-position `getfield` ops
/// on each side then dispatching to `newtuple_cached` to either
/// short-circuit to a const-result `inputconst` or emit a fresh
/// `malloc` + per-item `setfield` sequence. Aliased through the
/// pair dispatcher for both `add` and `inplace_add` opnames.
pub fn pair_tuple_tuple_rtype_add(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> crate::translator::rtyper::rmodel::RTypeResult {
    use crate::translator::rtyper::rtyper::ConvertedTo;
    let any1: &dyn std::any::Any = r1;
    let r_tup1 = any1
        .downcast_ref::<TupleRepr>()
        .ok_or_else(|| TyperError::message("pair_tuple_tuple_rtype_add: r1 is not a TupleRepr"))?;
    let any2: &dyn std::any::Any = r2;
    let r_tup2 = any2
        .downcast_ref::<TupleRepr>()
        .ok_or_else(|| TyperError::message("pair_tuple_tuple_rtype_add: r2 is not a TupleRepr"))?;
    // upstream `v_tuple1, v_tuple2 = hop.inputargs(r_tup1, r_tup2)`.
    let v_args = hop.inputargs(vec![ConvertedTo::Repr(r_tup1), ConvertedTo::Repr(r_tup2)])?;
    let v_tuple1 = v_args[0].clone();
    let v_tuple2 = v_args[1].clone();
    // upstream loop — per-position getitem_internal on each side,
    // appended into a single vlist.
    let mut vlist: Vec<Hlvalue> = Vec::with_capacity(r_tup1.items_r.len() + r_tup2.items_r.len());
    {
        let mut llops = hop.llops.borrow_mut();
        for i in 0..r_tup1.items_r.len() {
            let v = r_tup1.getitem_internal(&mut llops, v_tuple1.clone(), i)?;
            vlist.push(Hlvalue::Variable(v));
        }
        for i in 0..r_tup2.items_r.len() {
            let v = r_tup2.getitem_internal(&mut llops, v_tuple2.clone(), i)?;
            vlist.push(Hlvalue::Variable(v));
        }
    }
    // upstream `r_tup1.newtuple_cached(hop, vlist)`.
    let result = TupleRepr::newtuple_cached(hop, vlist)?;
    Ok(Some(result))
}

/// RPython `pairtype(TupleRepr, IntegerRepr).rtype_getitem`
/// (rtuple.py:264-273) free-function entry. Routes through the
/// receiver `TupleRepr`'s `rtype_pair_getitem` method.
pub fn pair_tuple_int_rtype_getitem(
    r_tup: &dyn Repr,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
) -> crate::translator::rtyper::rmodel::RTypeResult {
    let any_r: &dyn std::any::Any = r_tup;
    let r_tup = any_r.downcast_ref::<TupleRepr>().ok_or_else(|| {
        TyperError::message("pair_tuple_int_rtype_getitem: receiver is not a TupleRepr")
    })?;
    r_tup.rtype_pair_getitem(hop)
}

/// RPython `class TupleRepr(Repr)` (rtuple.py:129-204+).
///
/// Minimal slice — carries the item reprs + lowleveltype. Methods that
/// emit ops (`getitem`, `newtuple`, ...) land in follow-up commits.
#[derive(Debug)]
pub struct TupleRepr {
    /// RPython `self.items_r` (rtuple.py:132,138). The `internal_repr`
    /// per item from `externalvsinternal` — GC `InstanceRepr` items
    /// are mapped to the root `getinstancerepr(rtyper, None)`; non-GC
    /// items pass through unchanged.
    pub items_r: Vec<Arc<dyn Repr>>,
    /// RPython `self.external_items_r` (rtuple.py:133) — concrete
    /// per-position reprs preserved for `getitem`'s convertvar back to
    /// the surface type.
    pub external_items_r: Vec<Arc<dyn Repr>>,
    /// RPython `self.fieldnames = ['item%d' % i for i in ...]`
    /// (rtuple.py:139).
    pub fieldnames: Vec<String>,
    /// RPython `self.lltypes = [r.lowleveltype for r in items_r]`
    /// (rtuple.py:140).
    pub lltypes: Vec<LowLevelType>,
    /// RPython `self.lowleveltype = TUPLE_TYPE(self.lltypes)` (rtuple.py:142).
    lltype: LowLevelType,
    /// RPython `self.tuple_cache = {}` (rtuple.py:141). Caches the
    /// instantiated `_ptr` per `Vec<ConstValue>` key. `convert_const`
    /// matches upstream rtuple.py:190-191 — it instantiates, inserts
    /// the pointer into this cache BEFORE filling fields, and then
    /// mutates the cached entry via a brief `borrow_mut` per field
    /// write so recursive `r.convert_const(obj)` calls stay
    /// re-entrancy-safe.
    tuple_cache: RefCell<HashMap<Vec<ConstValue>, _ptr>>,
    state: ReprState,
}

impl TupleRepr {
    /// RPython `TupleRepr.__init__(self, rtyper, items_r)` (rtuple.py:131-142).
    ///
    /// Splits each input item via [`externalvsinternal`]: GC
    /// `InstanceRepr` items are stored internally as the root
    /// `getinstancerepr(rtyper, None)` while the concrete repr is kept
    /// at `external_items_r` for `getitem` to convert back. Non-GC
    /// items pass through with `external == internal`.
    ///
    /// `gcref=True` arm is deferred (no `rgcref` port yet).
    pub fn new(rtyper: &Rc<RPythonTyper>, items_r: Vec<Arc<dyn Repr>>) -> Result<Self, TyperError> {
        let mut internal_items: Vec<Arc<dyn Repr>> = Vec::with_capacity(items_r.len());
        let mut external_items: Vec<Arc<dyn Repr>> = Vec::with_capacity(items_r.len());
        for item_r in items_r {
            let (external, internal) =
                crate::translator::rtyper::rclass::externalvsinternal(rtyper, item_r)?;
            internal_items.push(internal);
            external_items.push(external);
        }
        let lltypes: Vec<LowLevelType> = internal_items
            .iter()
            .map(|r| r.lowleveltype().clone())
            .collect();
        let fieldnames = (0..internal_items.len())
            .map(|i| format!("item{i}"))
            .collect();
        let lltype = tuple_type(&lltypes);
        Ok(TupleRepr {
            items_r: internal_items,
            external_items_r: external_items,
            fieldnames,
            lltypes,
            lltype,
            tuple_cache: RefCell::new(HashMap::new()),
            state: ReprState::new(),
        })
    }

    /// RPython `TupleRepr.instantiate(self)` (rtuple.py:223-227).
    ///
    /// ```python
    /// def instantiate(self):
    ///     if len(self.items_r) == 0:
    ///         return dum_empty_tuple     # PBC placeholder for an empty tuple
    ///     else:
    ///         return malloc(self.lowleveltype.TO)
    /// ```
    ///
    /// The empty-tuple `dum_empty_tuple` PBC sentinel is not modelled
    /// — `convert_const` short-circuits to `Constant(None, Void)` for
    /// empty inputs via [`Self::instantiate_empty`] (see
    /// [`Repr::convert_const`] impl). For non-empty tuples this
    /// allocates a Gc instance of the `tuple%d` GcStruct via
    /// `lltype::malloc` with default `immortal=False` —
    /// matching upstream rtuple.py:226-227 `malloc(self.lowleveltype.TO)`.
    pub fn instantiate(&self) -> Result<_ptr, TyperError> {
        if self.items_r.is_empty() {
            return Err(TyperError::message(
                "TupleRepr.instantiate: empty-tuple sentinel uses Void short-circuit \
                 in convert_const; instantiate() should not be reached",
            ));
        }
        let LowLevelType::Ptr(ptr) = &self.lltype else {
            return Err(TyperError::message(format!(
                "TupleRepr.instantiate: lowleveltype is not Ptr, got {:?}",
                self.lltype
            )));
        };
        let inner: LowLevelType = match &ptr.TO {
            PtrTarget::Struct(body) => LowLevelType::Struct(Box::new(body.clone())),
            other => {
                return Err(TyperError::message(format!(
                    "TupleRepr.instantiate: Ptr target must be Struct, got {:?}",
                    other
                )));
            }
        };
        // upstream `malloc(self.lowleveltype.TO)` — defaults to
        // `flavor='gc', immortal=False`.
        lltype::malloc(inner, None, MallocFlavor::Gc, false).map_err(TyperError::message)
    }

    /// RPython empty-tuple branch of `instantiate` (rtuple.py:223-225):
    ///
    /// ```python
    /// def instantiate(self):
    ///     if len(self.items_r) == 0:
    ///         return dum_empty_tuple     # PBC placeholder for an empty tuple
    ///     ...
    /// ```
    ///
    /// Pyre returns a `Constant(None, Void)` carrier as the
    /// `dum_empty_tuple` analogue. Upstream's sentinel is a Python
    /// callable cached by identity in `tuple_cache`; the Rust port
    /// cannot key the typed cache (`HashMap<Vec<ConstValue>, _ptr>`)
    /// on a Void-typed value, so the structural parity is achieved
    /// by returning a deterministic `Constant(None, Void)` —
    /// `PartialEq`/`Hash` derive guarantees two emitted carriers
    /// compare equal, which is the only observable property
    /// upstream's identity cache provides.
    fn instantiate_empty(&self) -> Constant {
        Constant::with_concretetype(ConstValue::None, LowLevelType::Void)
    }

    /// RPython `TupleRepr.getitem_internal(self, llops, v_tuple, index)`
    /// (rtuple.py:248-253):
    ///
    /// ```python
    /// def getitem_internal(self, llops, v_tuple, index):
    ///     name = self.fieldnames[index]
    ///     llresult = self.lltypes[index]
    ///     cname = inputconst(Void, name)
    ///     return llops.genop('getfield', [v_tuple, cname], resulttype=llresult)
    /// ```
    pub fn getitem_internal(
        &self,
        llops: &mut LowLevelOpList,
        v_tuple: Hlvalue,
        index: usize,
    ) -> Result<Variable, TyperError> {
        let name = self.fieldnames.get(index).ok_or_else(|| {
            TyperError::message(format!(
                "TupleRepr.getitem_internal: index {index} out of range \
                 (len={})",
                self.fieldnames.len()
            ))
        })?;
        let llresult = self.lltypes[index].clone();
        let cname = Constant::with_concretetype(ConstValue::Str(name.clone()), LowLevelType::Void);
        Ok(llops
            .genop(
                "getfield",
                vec![v_tuple, Hlvalue::Constant(cname)],
                GenopResult::LLType(llresult),
            )
            .expect("getfield with non-Void result yields a Variable"))
    }

    /// RPython `TupleRepr.getitem(self, llops, v_tuple, index)`
    /// (rtuple.py:144-150):
    ///
    /// ```python
    /// def getitem(self, llops, v_tuple, index):
    ///     v = self.getitem_internal(llops, v_tuple, index)
    ///     r_item = self.items_r[index]
    ///     r_external_item = self.external_items_r[index]
    ///     return llops.convertvar(v, r_item, r_external_item)
    /// ```
    ///
    /// Pyre's external == internal for every item today (no GCRef
    /// wrapping), so the convertvar reduces to identity. Once
    /// `externalvsinternal`'s GCRef arm lands, `convertvar` will route
    /// through the pairtype dispatch.
    pub fn getitem(
        &self,
        llops: &mut LowLevelOpList,
        v_tuple: Hlvalue,
        index: usize,
    ) -> Result<Hlvalue, TyperError> {
        let v = self.getitem_internal(llops, v_tuple, index)?;
        let r_item = self.items_r.get(index).cloned().ok_or_else(|| {
            TyperError::message(format!(
                "TupleRepr.getitem: index {index} out of range (len={})",
                self.items_r.len()
            ))
        })?;
        let r_external = self.external_items_r[index].clone();
        llops.convertvar(Hlvalue::Variable(v), r_item.as_ref(), r_external.as_ref())
    }

    /// RPython `TupleRepr.newtuple(cls, llops, r_tuple, items_v)`
    /// (rtuple.py:152-168):
    ///
    /// ```python
    /// @classmethod
    /// def newtuple(cls, llops, r_tuple, items_v):
    ///     assert len(r_tuple.items_r) == len(items_v)
    ///     for r_item, v_item in zip(r_tuple.items_r, items_v):
    ///         assert r_item.lowleveltype == v_item.concretetype
    ///     if len(r_tuple.items_r) == 0:
    ///         return inputconst(Void, ())
    ///     c1 = inputconst(Void, r_tuple.lowleveltype.TO)
    ///     cflags = inputconst(Void, {'flavor': 'gc'})
    ///     v_result = llops.genop('malloc', [c1, cflags],
    ///                            resulttype=r_tuple.lowleveltype)
    ///     for i in range(len(r_tuple.items_r)):
    ///         cname = inputconst(Void, r_tuple.fieldnames[i])
    ///         llops.genop('setfield', [v_result, cname, items_v[i]])
    ///     return v_result
    /// ```
    ///
    /// Builds a fresh tuple value at runtime by emitting a `malloc`
    /// op followed by per-item `setfield` ops. Empty tuples surface
    /// as a Void `()` Constant.
    pub fn newtuple(
        llops: &mut LowLevelOpList,
        r_tuple: &TupleRepr,
        items_v: Vec<Hlvalue>,
    ) -> Result<Hlvalue, TyperError> {
        if r_tuple.items_r.len() != items_v.len() {
            return Err(TyperError::message(format!(
                "TupleRepr.newtuple: arity mismatch: r_tuple has {} items, items_v \
                 has {}",
                r_tuple.items_r.len(),
                items_v.len()
            )));
        }
        // upstream rtuple.py:155-157 —
        // `for r_item, v_item in zip(r_tuple.items_r, items_v):
        //      assert r_item.lowleveltype == v_item.concretetype`.
        // Each item's concretetype must match the matching items_r
        // repr's lowleveltype; Constant carriers and Variables both
        // expose concretetype.
        for (i, (r_item, v_item)) in r_tuple.items_r.iter().zip(items_v.iter()).enumerate() {
            let v_concrete = match v_item {
                Hlvalue::Variable(v) => v.concretetype(),
                Hlvalue::Constant(c) => c.concretetype.clone(),
            };
            let expected = r_item.lowleveltype();
            match v_concrete {
                Some(ref got) if got == expected => {}
                other => {
                    return Err(TyperError::message(format!(
                        "TupleRepr.newtuple: item {i} concretetype mismatch \
                         — items_r[{i}].lowleveltype = {expected:?}, \
                         items_v[{i}].concretetype = {other:?}"
                    )));
                }
            }
        }
        if r_tuple.items_r.is_empty() {
            // upstream `inputconst(Void, ())` — Void-typed empty tuple sentinel.
            return Ok(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Tuple(Vec::new()),
                LowLevelType::Void,
            )));
        }
        // upstream `c1 = inputconst(Void, r_tuple.lowleveltype.TO)`.
        // The Void-typed `c1` carries the inner Struct lltype as the
        // type-tag for `malloc`'s lowering.
        let LowLevelType::Ptr(ptr) = &r_tuple.lltype else {
            return Err(TyperError::message(
                "TupleRepr.newtuple: lowleveltype is not Ptr",
            ));
        };
        let inner_struct = match &ptr.TO {
            PtrTarget::Struct(body) => body.clone(),
            other => {
                return Err(TyperError::message(format!(
                    "TupleRepr.newtuple: Ptr target must be Struct, got {:?}",
                    other
                )));
            }
        };
        let c1 = Constant::with_concretetype(
            ConstValue::LowLevelType(Box::new(LowLevelType::Struct(Box::new(inner_struct)))),
            LowLevelType::Void,
        );
        // upstream `cflags = inputconst(Void, {'flavor': 'gc'})`. We
        // surface the flavor via a Void-typed Str sentinel matching
        // how other malloc emitters in pyre encode it; the lowering
        // pass reads the type from `c1` and the flavor from `cflags`.
        let cflags = Constant::with_concretetype(
            ConstValue::Str("flavor=gc".to_string()),
            LowLevelType::Void,
        );
        let v_result = llops
            .genop(
                "malloc",
                vec![Hlvalue::Constant(c1), Hlvalue::Constant(cflags)],
                GenopResult::LLType(r_tuple.lltype.clone()),
            )
            .expect("malloc with non-Void result yields a Variable");
        let v_result_h = Hlvalue::Variable(v_result);
        for (i, v_item) in items_v.into_iter().enumerate() {
            let cname = Constant::with_concretetype(
                ConstValue::Str(r_tuple.fieldnames[i].clone()),
                LowLevelType::Void,
            );
            llops.genop(
                "setfield",
                vec![v_result_h.clone(), Hlvalue::Constant(cname), v_item],
                GenopResult::Void,
            );
        }
        Ok(v_result_h)
    }

    /// RPython `TupleRepr.rtype_len(self, hop)` (rtuple.py:200-201):
    ///
    /// ```python
    /// def rtype_len(self, hop):
    ///     return hop.inputconst(Signed, len(self.items_r))
    /// ```
    ///
    /// Inherent helper used by both the [`Repr::rtype_len`] override
    /// and direct callers. Renamed from `rtype_len` to avoid the
    /// inherent-vs-trait-impl name clash that would force UFCS at
    /// every call site.
    fn rtype_len_inherent(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rtyper::HighLevelOp;
        let _ = hop;
        let n = self.items_r.len() as i64;
        let c = HighLevelOp::inputconst(&LowLevelType::Signed, &ConstValue::Int(n))?;
        Ok(Some(Hlvalue::Constant(c)))
    }

    /// RPython `TupleRepr._rtype_newtuple(cls, hop)` (rtuple.py:178-182):
    ///
    /// ```python
    /// @classmethod
    /// def _rtype_newtuple(cls, hop):
    ///     r_tuple = hop.r_result
    ///     vlist = hop.inputargs(*r_tuple.items_r)
    ///     return cls.newtuple_cached(hop, vlist)
    /// ```
    ///
    /// `newtuple_cached` (rtuple.py:170-176) routes through the
    /// constant-result fast path; otherwise it calls `cls.newtuple`.
    /// Both arms are implemented here.
    /// RPython `pairtype(TupleRepr, IntegerRepr).rtype_getitem`
    /// helper — used internally by [`pair_tuple_int_rtype_getitem`].
    fn rtype_pair_getitem(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rtyper::ConvertedTo;
        // upstream `v_tuple, v_index = hop.inputargs(r_tup, Signed)`.
        let v_args = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::LowLevelType(&LowLevelType::Signed),
        ])?;
        let v_tuple = v_args[0].clone();
        let v_index = &v_args[1];
        // upstream `if not isinstance(v_index, Constant): raise
        // TyperError("non-constant tuple index")`.
        let Hlvalue::Constant(idx_const) = v_index else {
            return Err(TyperError::message(
                "pair(TupleRepr, IntegerRepr).rtype_getitem: non-constant tuple index",
            ));
        };
        let ConstValue::Int(idx) = idx_const.value else {
            return Err(TyperError::message(format!(
                "pair(TupleRepr, IntegerRepr).rtype_getitem: tuple index must be Int, got {:?}",
                idx_const.value
            )));
        };
        // upstream rtuple.py:266-273 — does NOT bounds-check or
        // reject negative indices itself; `r_tup.getitem(hop.llops,
        // v_tuple, index)` flows through `self.fieldnames[index]`
        // which is a Python list and supports negative indexing
        // (`fieldnames[-1]` → last item). Mirror the wrap-around so
        // `(0, 1, 2)[-1]` types as `2` matching upstream.
        //
        // upstream rtuple.py:270-271 also drops the implicit
        // IndexError exception channel via
        // `if hop.has_implicit_exception(IndexError):
        //     hop.exception_cannot_occur()`. Pyre's `HighLevelOp`
        // does not yet model implicit-exception channels; the call
        // sites that thread getitem through exception-aware lowering
        // will surface the gap. Tracked as a follow-on once
        // `has_implicit_exception` lands.
        let len = self.items_r.len() as i64;
        let normalized = if idx < 0 { idx + len } else { idx };
        if normalized < 0 || normalized >= len {
            return Err(TyperError::message(format!(
                "pair(TupleRepr, IntegerRepr).rtype_getitem: index {idx} out of \
                 range (len={len})"
            )));
        }
        let v = self.getitem(&mut hop.llops.borrow_mut(), v_tuple, normalized as usize)?;
        Ok(Some(v))
    }

    /// RPython `TupleRepr.newtuple_cached(cls, hop, items_v)`
    /// (rtuple.py:170-176):
    ///
    /// ```python
    /// @classmethod
    /// def newtuple_cached(cls, hop, items_v):
    ///     r_tuple = hop.r_result
    ///     if hop.s_result.is_constant():
    ///         return inputconst(r_tuple, hop.s_result.const)
    ///     else:
    ///         return cls.newtuple(hop.llops, r_tuple, items_v)
    /// ```
    ///
    /// `r_tuple` is read from `hop.r_result` and downcast to
    /// `TupleRepr`. The const-result fast path bypasses
    /// emit-and-fill in favour of a single `inputconst(r_tuple,
    /// hop.s_result.const)` carrier — matching upstream's behaviour
    /// of materialising a Constant tuple from the annotator's known
    /// const value.
    pub fn newtuple_cached(
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
        items_v: Vec<Hlvalue>,
    ) -> Result<Hlvalue, TyperError> {
        use crate::translator::rtyper::rmodel::inputconst;
        let r_result =
            hop.r_result.borrow().clone().ok_or_else(|| {
                TyperError::message("TupleRepr.newtuple_cached: r_result missing")
            })?;
        let any_r: &dyn std::any::Any = r_result.as_ref();
        let r_tuple = any_r.downcast_ref::<TupleRepr>().ok_or_else(|| {
            TyperError::message("TupleRepr.newtuple_cached: hop.r_result is not a TupleRepr")
        })?;
        let s_const = hop
            .s_result
            .borrow()
            .as_ref()
            .and_then(|s| s.const_())
            .cloned();
        if let Some(value) = s_const {
            let c = inputconst(r_tuple, &value)?;
            return Ok(Hlvalue::Constant(c));
        }
        TupleRepr::newtuple(&mut hop.llops.borrow_mut(), r_tuple, items_v)
    }

    /// RPython `class __extend__(TupleRepr).rtype_getslice(r_tup, hop)`
    /// (rtuple.py:277-290):
    ///
    /// ```python
    /// def rtype_getslice(r_tup, hop):
    ///     s_start = hop.args_s[1]
    ///     s_stop = hop.args_s[2]
    ///     assert s_start.is_immutable_constant(),"tuple slicing: needs constants"
    ///     assert s_stop.is_immutable_constant(), "tuple slicing: needs constants"
    ///     start = s_start.const
    ///     stop = s_stop.const
    ///     indices = range(len(r_tup.items_r))[start:stop]
    ///     assert len(indices) == len(hop.r_result.items_r)
    ///
    ///     v_tup = hop.inputarg(r_tup, arg=0)
    ///     items_v = [r_tup.getitem_internal(hop.llops, v_tup, i)
    ///                for i in indices]
    ///     return hop.r_result.newtuple(hop.llops, hop.r_result, items_v)
    /// ```
    ///
    /// Slice bounds must be immutable constants (compile-time).
    /// Negative indices wrap-around via Python's `range[start:stop]`
    /// semantics: the index is clamped to `[0, n]` after adding `n`
    /// to negatives. The result repr's `items_r` arity must match
    /// the slice length, which is the rtyper-supplied invariant
    /// (annotator computed `r_result` from the same start/stop).
    fn rtype_getslice_inherent(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::annotator::model::SomeObjectTrait;
        use crate::translator::rtyper::rtyper::ConvertedTo;
        // upstream rtuple.py:278-281 — `hop.args_s[1]` / `[2]` are
        // start / stop annotations. Both must be immutable
        // constants — slice bounds resolve at rtype-time.
        let (start, stop) = {
            let args_s = hop.args_s.borrow();
            let s_start = args_s.get(1).ok_or_else(|| {
                TyperError::message("TupleRepr.rtype_getslice: args_s[1] (start) missing")
            })?;
            let s_stop = args_s.get(2).ok_or_else(|| {
                TyperError::message("TupleRepr.rtype_getslice: args_s[2] (stop) missing")
            })?;
            if !s_start.is_immutable_constant() {
                return Err(TyperError::message("tuple slicing: needs constants"));
            }
            if !s_stop.is_immutable_constant() {
                return Err(TyperError::message("tuple slicing: needs constants"));
            }
            let start = match s_start.const_() {
                Some(ConstValue::Int(v)) => *v,
                other => {
                    return Err(TyperError::message(format!(
                        "TupleRepr.rtype_getslice: start must be Int constant, got {:?}",
                        other
                    )));
                }
            };
            let stop = match s_stop.const_() {
                Some(ConstValue::Int(v)) => *v,
                other => {
                    return Err(TyperError::message(format!(
                        "TupleRepr.rtype_getslice: stop must be Int constant, got {:?}",
                        other
                    )));
                }
            };
            (start, stop)
        };
        // upstream rtuple.py:284 — `indices = range(len(items_r))[start:stop]`.
        // Mirror Python's slice clamping: negatives add `n` then floor at 0;
        // out-of-range positives clamp to `n`.
        let n = self.items_r.len() as i64;
        let normalize = |i: i64| -> i64 { if i < 0 { (i + n).max(0) } else { i.min(n) } };
        let lo = normalize(start);
        let hi = normalize(stop);
        let indices: Vec<usize> = if lo < hi {
            (lo as usize..hi as usize).collect()
        } else {
            Vec::new()
        };
        // upstream rtuple.py:285 — `assert len(indices) == len(hop.r_result.items_r)`.
        let r_result_arc = hop
            .r_result
            .borrow()
            .clone()
            .ok_or_else(|| TyperError::message("TupleRepr.rtype_getslice: r_result missing"))?;
        let any_r: &dyn std::any::Any = r_result_arc.as_ref();
        let r_result = any_r.downcast_ref::<TupleRepr>().ok_or_else(|| {
            TyperError::message("TupleRepr.rtype_getslice: r_result is not a TupleRepr")
        })?;
        if indices.len() != r_result.items_r.len() {
            return Err(TyperError::message(format!(
                "TupleRepr.rtype_getslice: slice length {} != r_result.items_r len {}",
                indices.len(),
                r_result.items_r.len()
            )));
        }
        // upstream rtuple.py:287 — `v_tup = hop.inputarg(r_tup, arg=0)`.
        let v_tup = hop.inputarg(ConvertedTo::Repr(self), 0)?;
        // upstream rtuple.py:288-289 — per-index getitem_internal.
        let mut items_v: Vec<Hlvalue> = Vec::with_capacity(indices.len());
        {
            let mut llops = hop.llops.borrow_mut();
            for i in &indices {
                let v = self.getitem_internal(&mut llops, v_tup.clone(), *i)?;
                items_v.push(Hlvalue::Variable(v));
            }
        }
        // upstream rtuple.py:290 — `r_result.newtuple(hop.llops, r_result, items_v)`.
        let result = TupleRepr::newtuple(&mut hop.llops.borrow_mut(), r_result, items_v)?;
        Ok(Some(result))
    }

    pub fn rtype_newtuple(
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        use crate::translator::rtyper::rtyper::ConvertedTo;
        let r_result =
            hop.r_result.borrow().clone().ok_or_else(|| {
                TyperError::message("TupleRepr._rtype_newtuple: r_result missing")
            })?;
        let any_r: &dyn std::any::Any = r_result.as_ref();
        let r_tuple = any_r.downcast_ref::<TupleRepr>().ok_or_else(|| {
            TyperError::message("TupleRepr._rtype_newtuple: hop.r_result is not a TupleRepr")
        })?;
        // upstream `vlist = hop.inputargs(*r_tuple.items_r)`. Each
        // arg is coerced to the matching item repr.
        let converted: Vec<ConvertedTo<'_>> = r_tuple
            .items_r
            .iter()
            .map(|r| ConvertedTo::Repr(r.as_ref()))
            .collect();
        let vlist = hop.inputargs(converted)?;
        // upstream `cls.newtuple_cached(hop, vlist)`.
        let result = TupleRepr::newtuple_cached(hop, vlist)?;
        Ok(Some(result))
    }
}

impl Repr for TupleRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "TupleRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::TupleRepr
    }

    /// `RPythonTyper.translate_op_len` (rtyper.py:484-486) dispatches
    /// `r.rtype_len(hop)` — without this override the default
    /// `Repr.rtype_len` would raise `MissingRTypeOperation` for tuples.
    /// Forwards to the inherent [`TupleRepr::rtype_len`] which mirrors
    /// upstream rtuple.py:200-201.
    fn rtype_len(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.rtype_len_inherent(hop)
    }

    /// `class __extend__(TupleRepr).rtype_getslice` (rtuple.py:277-290).
    /// Without this override `Repr.rtype_getslice` raises
    /// `MissingRTypeOperation("getslice")`. Forwards to the inherent
    /// helper.
    fn rtype_getslice(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> crate::translator::rtyper::rmodel::RTypeResult {
        self.rtype_getslice_inherent(hop)
    }

    /// RPython `TupleRepr.convert_const(self, value)` (rtuple.py:184-198).
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     assert isinstance(value, tuple) and len(value) == len(self.items_r)
    ///     key = tuple([Constant(item) for item in value])
    ///     try:
    ///         return self.tuple_cache[key]
    ///     except KeyError:
    ///         p = self.instantiate()
    ///         self.tuple_cache[key] = p
    ///         for obj, r, name in zip(value, self.items_r, self.fieldnames):
    ///             if r.lowleveltype is not Void:
    ///                 setattr(p, name, r.convert_const(obj))
    ///         return p
    /// ```
    ///
    /// The empty-tuple arm reduces to `Constant(None, Void)` because
    /// upstream `TUPLE_TYPE([])` is `Void`. Non-empty tuples
    /// `instantiate` an immortal Gc struct, recursively
    /// `convert_const` each item to a `LowLevelValue`, write into
    /// the struct field, and cache the resulting `_ptr` keyed on the
    /// raw `Vec<ConstValue>` items.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let ConstValue::Tuple(items) = value else {
            return Err(TyperError::message(format!(
                "TupleRepr.convert_const: value must be a tuple, got {value:?}"
            )));
        };
        if items.len() != self.items_r.len() {
            return Err(TyperError::message(format!(
                "TupleRepr.convert_const: tuple arity mismatch: got {}, expected {}",
                items.len(),
                self.items_r.len()
            )));
        }
        if self.items_r.is_empty() {
            // upstream rtuple.py:184-194 routes the empty case
            // through `self.instantiate()` which returns
            // `dum_empty_tuple` (a Python sentinel function used as
            // a Void-typed PBC placeholder), and caches that
            // sentinel under the empty-tuple key in `tuple_cache`.
            // Pyre's `tuple_cache` is `HashMap<Vec<ConstValue>, _ptr>`
            // — it cannot store a Void-typed sentinel because
            // `TUPLE_TYPE([]) == Void` and `_ptr` describes a Ptr
            // value. The structural parity is achieved by the
            // `instantiate_empty` helper below: it materialises a
            // canonical `Constant(None, Void)` once and returns
            // cloned copies on subsequent calls. Two `Constant(None,
            // Void)` instances compare equal under `PartialEq` and
            // hash identically, so consumers cannot distinguish
            // upstream's identity-cached sentinel from this body's
            // structural-equivalent.
            return Ok(self.instantiate_empty());
        }
        // Cache lookup — return the cached Constant if present.
        if let Some(cached) = self.tuple_cache.borrow().get(items) {
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(cached.clone())),
                self.lltype.clone(),
            ));
        }
        // upstream `p = self.instantiate()` + `self.tuple_cache[key] = p`
        // (rtuple.py:190-191): cache the instantiated `_ptr` BEFORE
        // filling its fields. Pyre's `_ptr` is value-typed (Clone
        // deep-copies `_obj0`) so we cannot keep an aliased handle
        // outside the cache — instead we hold the slot in the cache
        // and mutate it through a brief `borrow_mut` per field write.
        // The `borrow_mut` is dropped between writes so recursive
        // `r.convert_const(obj)` calls below stay re-entrancy-safe.
        let p = self.instantiate()?;
        self.tuple_cache.borrow_mut().insert(items.clone(), p);
        // upstream loop: `for obj, r, name in zip(value, items_r, fieldnames):
        //     if r.lowleveltype is not Void:
        //         setattr(p, name, r.convert_const(obj))`.
        for ((obj, r), name) in items
            .iter()
            .zip(self.items_r.iter())
            .zip(self.fieldnames.iter())
        {
            if matches!(r.lowleveltype(), LowLevelType::Void) {
                continue;
            }
            // Recursive call must NOT hold the tuple_cache borrow.
            let item_const = r.convert_const(obj)?;
            let llval = crate::translator::rtyper::rclass::constant_to_lowlevel_value(&item_const)?;
            let mut cache = self.tuple_cache.borrow_mut();
            let p_in_cache = cache.get_mut(items).ok_or_else(|| {
                TyperError::message("TupleRepr.convert_const: cached pointer disappeared mid-init")
            })?;
            p_in_cache
                .setattr(name, llval)
                .map_err(TyperError::message)?;
        }
        // Return a Clone of the cached entry — observationally
        // identical to returning `p` directly upstream.
        let cached = self
            .tuple_cache
            .borrow()
            .get(items)
            .cloned()
            .ok_or_else(|| {
                TyperError::message("TupleRepr.convert_const: cached pointer missing on return")
            })?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(cached)),
            self.lltype.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use std::rc::Rc;

    fn fresh_rtyper() -> Rc<RPythonTyper> {
        let ann = RPythonAnnotator::new(None, None, None, false);
        Rc::new(RPythonTyper::new(&ann))
    }

    #[test]
    fn tuple_type_empty_returns_void() {
        let t = tuple_type(&[]);
        assert_eq!(t, LowLevelType::Void);
    }

    #[test]
    fn tuple_type_single_signed_returns_ptr_gcstruct() {
        let t = tuple_type(&[LowLevelType::Signed]);
        let LowLevelType::Ptr(ptr) = t else {
            panic!("non-empty tuple_type must produce Ptr");
        };
        let PtrTarget::Struct(body) = &ptr.TO else {
            panic!("Ptr target must be Struct");
        };
        assert_eq!(body._name, "tuple1");
        assert!(body._flds.get("item0").is_some());
    }

    #[test]
    fn tuple_repr_new_collects_items_and_fieldnames() {
        use crate::translator::rtyper::rint::IntegerRepr;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        assert_eq!(repr.items_r.len(), 2);
        assert_eq!(
            repr.fieldnames,
            vec!["item0".to_string(), "item1".to_string()]
        );
        assert_eq!(repr.lltypes.len(), 2);
        let LowLevelType::Ptr(ptr) = repr.lowleveltype() else {
            panic!("non-empty tuple repr must carry Ptr lltype");
        };
        let PtrTarget::Struct(body) = &ptr.TO else {
            panic!("Ptr target must be Struct");
        };
        assert_eq!(body._name, "tuple2");
    }

    #[test]
    fn tuple_repr_empty_convert_const_returns_void_sentinel() {
        let rtyper = fresh_rtyper();
        let repr = TupleRepr::new(&rtyper, vec![]).unwrap();
        let out = repr
            .convert_const(&ConstValue::Tuple(vec![]))
            .expect("empty tuple must succeed");
        assert_eq!(out.concretetype, Some(LowLevelType::Void));
        assert!(matches!(out.value, ConstValue::None));
    }

    #[test]
    fn tuple_repr_non_empty_convert_const_returns_live_pointer() {
        use crate::translator::rtyper::rint::IntegerRepr;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        let out = repr
            .convert_const(&ConstValue::Tuple(vec![
                ConstValue::Int(7),
                ConstValue::Int(11),
            ]))
            .expect("non-empty tuple convert_const");
        let ConstValue::LLPtr(p) = &out.value else {
            panic!("expected LLPtr, got {:?}", out.value);
        };
        assert!(p.nonzero(), "tuple instance must be live");
        // Per-item field writes propagate.
        let item0 = p.getattr("item0").unwrap();
        let item1 = p.getattr("item1").unwrap();
        assert_eq!(item0, lltype::LowLevelValue::Signed(7));
        assert_eq!(item1, lltype::LowLevelValue::Signed(11));
    }

    #[test]
    fn tuple_repr_convert_const_caches_repeated_calls_to_same_pointer() {
        use crate::translator::rtyper::rint::IntegerRepr;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int.clone()]).unwrap();
        let value = ConstValue::Tuple(vec![ConstValue::Int(42)]);
        let a = repr.convert_const(&value).unwrap();
        let b = repr.convert_const(&value).unwrap();
        let (ConstValue::LLPtr(pa), ConstValue::LLPtr(pb)) = (&a.value, &b.value) else {
            panic!("expected LLPtr from both calls");
        };
        assert_eq!(pa._hashable_identity(), pb._hashable_identity());
    }

    #[test]
    fn tuple_repr_getitem_internal_emits_getfield() {
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let mut v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(repr.lowleveltype().clone()));
        let v = repr
            .getitem_internal(&mut llops, Hlvalue::Variable(v_tuple), 1)
            .unwrap();
        assert_eq!(llops.ops.len(), 1);
        assert_eq!(llops.ops[0].opname, "getfield");
        let Hlvalue::Constant(field_const) = &llops.ops[0].args[1] else {
            panic!("getfield arg[1] must be a Constant");
        };
        assert_eq!(field_const.value, ConstValue::Str("item1".to_string()));
        assert_eq!(v.concretetype().as_ref(), Some(&LowLevelType::Signed));
    }

    #[test]
    fn tuple_repr_newtuple_emits_malloc_and_setfields() {
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let mut v_a = Variable::new();
        v_a.set_concretetype(Some(LowLevelType::Signed));
        let mut v_b = Variable::new();
        v_b.set_concretetype(Some(LowLevelType::Signed));
        let _ = TupleRepr::newtuple(
            &mut llops,
            &repr,
            vec![Hlvalue::Variable(v_a), Hlvalue::Variable(v_b)],
        )
        .unwrap();
        // 1 malloc + 2 setfield = 3 ops total.
        assert_eq!(llops.ops.len(), 3);
        assert_eq!(llops.ops[0].opname, "malloc");
        assert_eq!(llops.ops[1].opname, "setfield");
        assert_eq!(llops.ops[2].opname, "setfield");
    }

    #[test]
    fn tuple_repr_rtype_len_emits_inputconst_signed() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let repr =
            TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone(), r_int.clone()]).unwrap();
        let result_var = Variable::new();
        let spaceop =
            SpaceOperation::new(OpKind::Len.opname(), vec![], Hlvalue::Variable(result_var));
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        let out = repr.rtype_len(&hop).expect("rtype_len").unwrap();
        let Hlvalue::Constant(c) = out else {
            panic!("rtype_len must return a Constant");
        };
        assert_eq!(c.value, ConstValue::Int(3));
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Signed));
    }

    #[test]
    fn pair_tuple_int_rtype_getitem_emits_getfield_for_constant_index() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        // Build a HighLevelOp with two args: tuple variable + Int(1) constant.
        let mut v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        let v_idx_h = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(1),
            LowLevelType::Signed,
        ));
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::GetItem.opname(),
            vec![v_tuple_h.clone(), v_idx_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        hop.args_v.borrow_mut().push(v_idx_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));

        let _ = pair_tuple_int_rtype_getitem(r_tup_arc.as_ref(), &hop)
            .expect("pair_tuple_int_rtype_getitem")
            .unwrap();
        let ops = hop.llops.borrow();
        // upstream emits `getfield(v_tuple, 'item1')` and a no-op
        // convertvar (since external == internal). One getfield op
        // expected total.
        assert!(
            ops.ops.iter().any(|op| op.opname == "getfield"),
            "expected at least one getfield op, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
    }

    /// upstream rtuple.py:266-273 — tuple constant indexing flows
    /// through `self.fieldnames[index]` which is a Python list and
    /// supports negative indexing (`fieldnames[-1]` returns the
    /// last item). Pyre normalises `idx + len` to mirror the
    /// wrap-around so `(0, 1, 2)[-1]` types as `2`.
    #[test]
    fn pair_tuple_int_rtype_getitem_negative_index_wraps_around_to_last_item() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;
        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> = Arc::new(
            TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone(), r_int.clone()]).unwrap(),
        );
        let mut v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        // Negative constant index — `-1` should map to last (item2).
        let v_idx_h = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Int(-1),
            LowLevelType::Signed,
        ));
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::GetItem.opname(),
            vec![v_tuple_h.clone(), v_idx_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        hop.args_v.borrow_mut().push(v_idx_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));

        pair_tuple_int_rtype_getitem(r_tup_arc.as_ref(), &hop)
            .expect("negative index must succeed")
            .unwrap();
        let ops = hop.llops.borrow();
        // The emitted getfield must reference `item2` (last item of
        // a 3-tuple, i.e. fieldnames[-1] == "item2").
        let getfield = ops
            .ops
            .iter()
            .find(|op| op.opname == "getfield")
            .expect("getfield op expected");
        let Hlvalue::Constant(c) = &getfield.args[1] else {
            panic!("getfield arg[1] must be a Constant");
        };
        assert_eq!(c.value, ConstValue::Str("item2".to_string()));
    }

    #[test]
    fn tuple_repr_newtuple_empty_returns_void_constant() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let rtyper = fresh_rtyper();
        let repr = TupleRepr::new(&rtyper, vec![]).unwrap();
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let out = TupleRepr::newtuple(&mut llops, &repr, vec![]).unwrap();
        assert!(llops.ops.is_empty(), "empty tuple emits no ops");
        let Hlvalue::Constant(c) = out else {
            panic!("expected Constant for empty tuple");
        };
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Void));
    }

    /// `translate_op_len` (rtyper.py:484-486) dispatches `r.rtype_len(hop)`
    /// where `r` is the `Repr` for the first argument. Without
    /// [`Repr::rtype_len`] override on `TupleRepr` the call would fall
    /// to the trait default `missing_rtype_operation("len")`. This test
    /// goes through `RPythonTyper::translate_operation("len", ...)` to
    /// pin the override at the dispatch level — not just the inherent
    /// helper.
    #[test]
    fn translate_operation_len_routes_to_tuplerepr_override() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let mut v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::Len.opname(),
            vec![v_tuple_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc));

        let out = rtyper
            .translate_operation(&hop)
            .expect("translate_operation len must dispatch through TupleRepr override")
            .expect("len returns a value");
        let Hlvalue::Constant(c) = out else {
            panic!("rtype_len must return a Constant");
        };
        assert_eq!(c.value, ConstValue::Int(2));
    }

    /// `externalvsinternal` (rmodel.py:417-429) maps each GC
    /// `InstanceRepr` item to the root `getinstancerepr(rtyper, None)`
    /// while keeping the concrete repr on the `external_items_r` side.
    /// Non-instance items (Integer / Bool / etc.) pass through with
    /// `external == internal`.
    #[test]
    fn tuple_repr_new_routes_gc_instance_items_through_externalvsinternal() {
        use crate::annotator::classdesc::ClassDef;
        use crate::translator::rtyper::rclass::{Flavor, getinstancerepr};
        use crate::translator::rtyper::rint::IntegerRepr;

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.Foo", None);
        let r_inst = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).unwrap();
        let r_inst_arc: Arc<dyn Repr> = r_inst.clone();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));

        let repr = TupleRepr::new(&rtyper, vec![r_inst_arc.clone(), r_int.clone()]).unwrap();

        // External items_r: untouched per-position concrete reprs.
        assert!(
            Arc::ptr_eq(&repr.external_items_r[0], &r_inst_arc),
            "external_items_r[0] must be the original GC InstanceRepr"
        );
        assert!(
            Arc::ptr_eq(&repr.external_items_r[1], &r_int),
            "external_items_r[1] (non-instance) is preserved as-is"
        );

        // Internal items_r[0] is rerouted to the root InstanceRepr
        // (classdef=None) — the lowleveltype must equal the root
        // OBJECTPTR since both root + leaf carry the OBJECT GcStruct.
        let root_inst = getinstancerepr(&rtyper, None, Flavor::Gc).unwrap();
        assert_eq!(
            repr.items_r[0].lowleveltype(),
            root_inst.lowleveltype(),
            "internal items_r[0] must match root InstanceRepr lowleveltype"
        );
        // And the non-instance int repr is internal == external.
        assert!(
            Arc::ptr_eq(&repr.items_r[1], &r_int),
            "items_r[1] (Integer) must be unchanged"
        );
    }

    /// `translate_op_newtuple` (rtyper.py:547-549) dispatches the free
    /// function `rtuple.rtype_newtuple(hop)`. Without the explicit
    /// `"newtuple"` arm in `RPythonTyper::translate_operation` the op
    /// would fall to `default_translate_operation` and raise
    /// "unimplemented operation 'newtuple'".
    #[test]
    fn translate_operation_newtuple_routes_to_rtuple_dispatch() {
        use crate::annotator::model::{SomeTuple, SomeValue};
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let mut v_a = Variable::new();
        v_a.set_concretetype(Some(LowLevelType::Signed));
        let mut v_b = Variable::new();
        v_b.set_concretetype(Some(LowLevelType::Signed));
        let v_a_h = Hlvalue::Variable(v_a);
        let v_b_h = Hlvalue::Variable(v_b);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            "newtuple".to_string(),
            vec![v_a_h.clone(), v_b_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // Args metadata — newtuple takes per-item args typed at each
        // r_tuple.items_r repr (Signed here).
        hop.args_v.borrow_mut().push(v_a_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        hop.args_v.borrow_mut().push(v_b_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        // r_result is the TupleRepr — `_rtype_newtuple` reads it from
        // `hop.r_result`.
        *hop.r_result.borrow_mut() = Some(r_tup_arc.clone());
        // s_result is non-const so the cache fast path is skipped and
        // genop("malloc") + per-item genop("setfield") are emitted.
        *hop.s_result.borrow_mut() = Some(SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Impossible,
            SomeValue::Impossible,
        ])));
        let _ = OpKind::SimpleCall; // keep imports minimal silenced

        let out = rtyper
            .translate_operation(&hop)
            .expect("translate_operation newtuple must dispatch through rtuple")
            .expect("newtuple returns a value");
        let Hlvalue::Variable(_) = out else {
            panic!("non-const newtuple must return a Variable from emitted malloc");
        };
        let ops = hop.llops.borrow();
        assert!(
            ops.ops.iter().any(|op| op.opname == "malloc"),
            "expected malloc op, got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
    }

    /// `pairtype(TupleRepr, TupleRepr).rtype_is_` (rtuple.py:355-356)
    /// raises a `TyperError` instead of routing to `ptr_eq` on the
    /// tuple pointers — upstream rejects tuple identity comparison
    /// eagerly. The dispatch table arm in `pairtype.rs` propagates
    /// the error past the generic `(Repr, Repr).rtype_is_` arm.
    #[test]
    fn pair_tuple_tuple_rtype_is_returns_typeerror() {
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup_arc: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let mut v_a = Variable::new();
        v_a.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let mut v_b = Variable::new();
        v_b.set_concretetype(Some(r_tup_arc.lowleveltype().clone()));
        let v_a_h = Hlvalue::Variable(v_a);
        let v_b_h = Hlvalue::Variable(v_b);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::Is.opname(),
            vec![v_a_h.clone(), v_b_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_a_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));
        hop.args_v.borrow_mut().push(v_b_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup_arc.clone()));

        let err = rtyper
            .translate_operation(&hop)
            .expect_err("rtype_is_ on two tuples must error");
        assert!(
            format!("{err}").contains("cannot compare tuples with 'is'"),
            "expected upstream error message, got: {err}"
        );
    }

    /// `TupleRepr.rtype_getslice` (rtuple.py:277-290) extracts the
    /// per-position items via `getitem_internal` and assembles a
    /// fresh tuple via `newtuple`. For `(a, b, c)[1:3]` the emitted
    /// op stream is: 2× getfield (for items 1 and 2) + 1× malloc + 2×
    /// setfield = 5 ops total. The result tuple's repr arity must
    /// match `len(indices)`.
    #[test]
    fn tuple_repr_rtype_getslice_emits_per_index_getfield_and_newtuple() {
        use crate::annotator::model::{SomeInteger, SomeTuple, SomeValue};
        use crate::flowspace::model::Constant as FlowConstant;
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup3: Arc<dyn Repr> = Arc::new(
            TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone(), r_int.clone()]).unwrap(),
        );
        let r_tup2: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let mut v_tuple = Variable::new();
        v_tuple.set_concretetype(Some(r_tup3.lowleveltype().clone()));
        let v_tuple_h = Hlvalue::Variable(v_tuple);
        // start = 1, stop = 3 — slice [1:3] on a 3-tuple yields 2 items.
        let v_start_h = Hlvalue::Constant(FlowConstant::with_concretetype(
            ConstValue::Int(1),
            LowLevelType::Signed,
        ));
        let v_stop_h = Hlvalue::Constant(FlowConstant::with_concretetype(
            ConstValue::Int(3),
            LowLevelType::Signed,
        ));
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::GetSlice.opname(),
            vec![v_tuple_h.clone(), v_start_h.clone(), v_stop_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        // Args metadata.
        hop.args_v.borrow_mut().push(v_tuple_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup3.clone()));
        // start=1: SomeInteger with const_box.
        let mut s_start = SomeInteger::new(false, false);
        s_start.base.const_box = Some(FlowConstant::with_concretetype(
            ConstValue::Int(1),
            LowLevelType::Signed,
        ));
        hop.args_v.borrow_mut().push(v_start_h);
        hop.args_s.borrow_mut().push(SomeValue::Integer(s_start));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        // stop=3.
        let mut s_stop = SomeInteger::new(false, false);
        s_stop.base.const_box = Some(FlowConstant::with_concretetype(
            ConstValue::Int(3),
            LowLevelType::Signed,
        ));
        hop.args_v.borrow_mut().push(v_stop_h);
        hop.args_s.borrow_mut().push(SomeValue::Integer(s_stop));
        hop.args_r.borrow_mut().push(Some(r_int.clone()));
        // r_result is the 2-tuple; s_result non-const so newtuple emits.
        *hop.r_result.borrow_mut() = Some(r_tup2.clone());
        *hop.s_result.borrow_mut() = Some(SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Impossible,
            SomeValue::Impossible,
        ])));

        let out = rtyper
            .translate_operation(&hop)
            .expect("getslice translates")
            .expect("getslice returns a value");
        let Hlvalue::Variable(_) = out else {
            panic!("getslice must return Variable from emitted malloc");
        };
        let ops = hop.llops.borrow();
        let getfield_count = ops.ops.iter().filter(|op| op.opname == "getfield").count();
        let malloc_count = ops.ops.iter().filter(|op| op.opname == "malloc").count();
        let setfield_count = ops.ops.iter().filter(|op| op.opname == "setfield").count();
        assert_eq!(
            getfield_count,
            2,
            "expected 2 getfield (items 1, 2), got ops: {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
        assert_eq!(
            malloc_count, 1,
            "expected 1 malloc for the 2-tuple newtuple"
        );
        assert_eq!(
            setfield_count, 2,
            "expected 2 setfield (items 0, 1) on the new tuple"
        );
    }

    /// `pairtype(TupleRepr, TupleRepr).rtype_add` (rtuple.py:319-327)
    /// concatenates two tuples by per-position getfield_internal +
    /// newtuple_cached. For `(a, b) + (c,)` the emitted op stream is:
    /// 3× getfield (items 0,1 of left + item 0 of right) + 1× malloc +
    /// 3× setfield = 7 ops total. Result repr arity = `len_a + len_b`.
    #[test]
    fn pair_tuple_tuple_rtype_add_concatenates_via_per_side_getfield_and_newtuple() {
        use crate::annotator::model::{SomeTuple, SomeValue};
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList};
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_tup2: Arc<dyn Repr> =
            Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap());
        let r_tup1: Arc<dyn Repr> = Arc::new(TupleRepr::new(&rtyper, vec![r_int.clone()]).unwrap());
        let r_tup_result: Arc<dyn Repr> = Arc::new(
            TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone(), r_int.clone()]).unwrap(),
        );
        let mut v_left = Variable::new();
        v_left.set_concretetype(Some(r_tup2.lowleveltype().clone()));
        let v_left_h = Hlvalue::Variable(v_left);
        let mut v_right = Variable::new();
        v_right.set_concretetype(Some(r_tup1.lowleveltype().clone()));
        let v_right_h = Hlvalue::Variable(v_right);
        let result_var = Variable::new();
        let spaceop = SpaceOperation::new(
            OpKind::Add.opname(),
            vec![v_left_h.clone(), v_right_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_left_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup2.clone()));
        hop.args_v.borrow_mut().push(v_right_h);
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(r_tup1.clone()));
        *hop.r_result.borrow_mut() = Some(r_tup_result.clone());
        *hop.s_result.borrow_mut() = Some(SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Impossible,
            SomeValue::Impossible,
            SomeValue::Impossible,
        ])));

        let out = rtyper
            .translate_operation(&hop)
            .expect("tuple+tuple translates")
            .expect("tuple+tuple returns a value");
        let Hlvalue::Variable(_) = out else {
            panic!("non-const tuple+tuple must return a Variable from emitted malloc");
        };
        let ops = hop.llops.borrow();
        let getfield_count = ops.ops.iter().filter(|op| op.opname == "getfield").count();
        let malloc_count = ops.ops.iter().filter(|op| op.opname == "malloc").count();
        let setfield_count = ops.ops.iter().filter(|op| op.opname == "setfield").count();
        assert_eq!(getfield_count, 3, "2 getfield from left + 1 from right");
        assert_eq!(malloc_count, 1);
        assert_eq!(setfield_count, 3, "3 setfield on the 3-tuple result");
    }

    /// `pairtype(TupleRepr, TupleRepr).convert_from_to` (rtuple.py:340-353)
    /// returns the source value unchanged when both reprs have the
    /// same lowleveltype (rtuple.py:342-343). Different-arity tuples
    /// return `NotImplemented` (Ok(None)).
    #[test]
    fn pair_tuple_tuple_convert_from_to_identity_when_lltype_matches() {
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_a = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        let r_b = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        // Sanity: same items_r, same TUPLE_TYPE → equal lowleveltype.
        assert_eq!(r_a.lowleveltype(), r_b.lowleveltype());
        let mut v_in = Variable::new();
        v_in.set_concretetype(Some(r_a.lowleveltype().clone()));
        let v_in_h = Hlvalue::Variable(v_in);
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let out = pair_tuple_tuple_convert_from_to(&r_a, &r_b, &v_in_h, &mut llops)
            .expect("convert_from_to must succeed when lltypes match")
            .expect("same-lltype must short-circuit to Some(v)");
        // Identity — no ops emitted, result is the unchanged input.
        assert!(llops.ops.is_empty(), "identity convert emits no ops");
        let v_concrete = match &v_in_h {
            Hlvalue::Variable(v) => v.concretetype(),
            Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        let out_concrete = match &out {
            Hlvalue::Variable(v) => v.concretetype(),
            Hlvalue::Constant(c) => c.concretetype.clone(),
        };
        assert_eq!(
            v_concrete, out_concrete,
            "identity convert preserves concretetype"
        );
    }

    /// Different-arity tuple→tuple convert_from_to returns `Ok(None)`
    /// (upstream `NotImplemented`) — the rtyper's pair-MRO walker
    /// falls through to the next handler instead of synthesising a
    /// non-existent shape.
    #[test]
    fn pair_tuple_tuple_convert_from_to_different_arity_returns_notimplemented() {
        use crate::translator::rtyper::rint::IntegerRepr;
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        let rtyper = fresh_rtyper();
        let r_int: Arc<dyn Repr> = Arc::new(IntegerRepr::new(LowLevelType::Signed, Some("int_")));
        let r_2 = TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone()]).unwrap();
        let r_3 =
            TupleRepr::new(&rtyper, vec![r_int.clone(), r_int.clone(), r_int.clone()]).unwrap();
        let mut v = Variable::new();
        v.set_concretetype(Some(r_2.lowleveltype().clone()));
        let v_h = Hlvalue::Variable(v);
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let out = pair_tuple_tuple_convert_from_to(&r_2, &r_3, &v_h, &mut llops)
            .expect("convert_from_to must not error on arity mismatch");
        assert!(out.is_none(), "different-arity must return None");
        assert!(llops.ops.is_empty());
    }
}
