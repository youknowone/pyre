//! RPython `rpython/rtyper/lltypesystem/rgcref.py` — generic GCREF
//! wrapper repr.
//!
//! Upstream uses `GCRefRepr` when a container stores arbitrary GC
//! pointers as `llmemory.GCREF` while preserving the original external
//! repr. The port mirrors the cache/keying, constant opaque casts, and
//! pairtype conversions used by `externalvsinternal(..., gcref=True)`.
//! The `GCRefRepr.get_ll_dummyval_obj` producer and the conditional
//! `GCRefRepr.ll_str` wrapper are deferred as latent surfaces (see their
//! doc comments for the precise blockers).

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Arc;

use crate::flowspace::model::{
    Block, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link, SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    GCREF, GcKind, LowLevelType, LowLevelValue, Ptr, cast_opaque_ptr, getfunctionptr,
};
use crate::translator::rtyper::rclass::{Flavor, getinstancerepr};
use crate::translator::rtyper::rmodel::{DummyValueBuilder, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    GenopResult, LowLevelFunction, LowLevelOpList, RPythonTyper, helper_pygraph_from_graph,
    variable_with_lltype,
};

/// RPython `UNKNOWN = object()` (`rgcref.py:6`): sentinel marking a
/// `_ll_eq_func` / `_ll_hash_func` slot as not-yet-computed, distinct from a
/// computed `None` (the base repr has no eq/hash helper).
#[derive(Debug, Clone)]
enum LlHelperCache {
    Unknown,
    Computed(Option<LowLevelFunction>),
}

#[derive(Debug)]
pub struct GCRefRepr {
    r_base: Arc<dyn Repr>,
    lltype: LowLevelType,
    state: ReprState,
    /// `self._ll_eq_func = UNKNOWN` (`rgcref.py:21`), memoized on first
    /// `get_ll_eq_function` call.
    _ll_eq_func: RefCell<LlHelperCache>,
    /// `self._ll_hash_func = UNKNOWN` (`rgcref.py:22`).
    _ll_hash_func: RefCell<LlHelperCache>,
}

impl GCRefRepr {
    /// RPython `GCRefRepr.make(r_base, cache)` (`rgcref.py:11-17`),
    /// folding in `__init__` (`rgcref.py:20-28`).
    ///
    /// The conditional `self.ll_str` wrapper (`rgcref.py:24-28`, installed
    /// only `if hasattr(r_base, 'll_str')`) is omitted. Upstream the
    /// `ll_str` attribute is carried by `VoidRepr` (`rmodel.py:358`, an
    /// unreachable stub), `TupleRepr` (`rtuple.py:212`,
    /// `ll_str = property(gen_str_function)`), and the string reprs; pyre's
    /// `Repr` trait has no `ll_str` slot and `rtuple::gen_str_function` is
    /// itself a deferred stub, so no `r_base` supplies `ll_str` and the
    /// `hasattr` guard is vacuously false — there is nothing to wrap. The
    /// wrapper ports once a `Repr::ll_str` surface and `gen_str_function`
    /// land.
    pub fn make(
        r_base: Arc<dyn Repr>,
        cache: &RefCell<HashMap<usize, Arc<GCRefRepr>>>,
    ) -> Arc<GCRefRepr> {
        let key = Arc::as_ptr(&r_base) as *const () as usize;
        if let Some(existing) = cache.borrow().get(&key) {
            return existing.clone();
        }
        let repr = Arc::new(GCRefRepr {
            r_base,
            lltype: GCREF.clone(),
            state: ReprState::new(),
            _ll_eq_func: RefCell::new(LlHelperCache::Unknown),
            _ll_hash_func: RefCell::new(LlHelperCache::Unknown),
        });
        cache.borrow_mut().insert(key, repr.clone());
        repr
    }

    pub fn r_base(&self) -> Arc<dyn Repr> {
        self.r_base.clone()
    }
}

impl Repr for GCRefRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "GCRefRepr"
    }

    fn repr_class_id(&self) -> super::super::pairtype::ReprClassId {
        super::super::pairtype::ReprClassId::GCRefRepr
    }

    /// RPython `GCRefRepr.convert_const` (`rgcref.py:29-30`).
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let base = self.r_base.convert_const(value)?;
        let ConstValue::LLPtr(ptr) = base.value else {
            return Err(TyperError::message(format!(
                "GCRefRepr.convert_const: base repr returned non-pointer constant {:?}",
                base.value
            )));
        };
        let cast =
            cast_opaque_ptr(gcref_ptr().as_ref(), ptr.as_ref()).map_err(TyperError::message)?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(cast)),
            self.lltype.clone(),
        ))
    }

    /// RPython `GCRefRepr.get_ll_eq_function` (`rgcref.py:32-46`): compute the
    /// wrapper once, caching it in `_ll_eq_func` past the `UNKNOWN` sentinel.
    fn get_ll_eq_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        if matches!(*self._ll_eq_func.borrow(), LlHelperCache::Unknown) {
            let ll_eq_func = match self.r_base.get_ll_eq_function(rtyper)? {
                None => None,
                Some(base_eq) => {
                    let name = format!("ll_gcref_eq_{}", self.r_base.lowleveltype().short_name());
                    let base_lltype = self.r_base.lowleveltype().clone();
                    Some(rtyper.lowlevel_helper_function_with_builder(
                        name.clone(),
                        vec![self.lltype.clone(), self.lltype.clone()],
                        LowLevelType::Bool,
                        move |_rtyper, args, result| {
                            build_gcref_wrapper_graph(
                                &name,
                                args,
                                result,
                                &base_lltype,
                                base_eq.clone(),
                                "ptr",
                            )
                        },
                    )?)
                }
            };
            *self._ll_eq_func.borrow_mut() = LlHelperCache::Computed(ll_eq_func);
        }
        match &*self._ll_eq_func.borrow() {
            LlHelperCache::Computed(func) => Ok(func.clone()),
            LlHelperCache::Unknown => unreachable!("_ll_eq_func computed above"),
        }
    }

    /// RPython `GCRefRepr.get_ll_hash_function` (`rgcref.py:48-62`).
    fn get_ll_hash_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        if matches!(*self._ll_hash_func.borrow(), LlHelperCache::Unknown) {
            let ll_hash_func = match self.r_base.get_ll_hash_function(rtyper)? {
                None => None,
                Some(base_hash) => {
                    let name = format!("ll_gcref_hash_{}", self.r_base.lowleveltype().short_name());
                    let base_lltype = self.r_base.lowleveltype().clone();
                    Some(rtyper.lowlevel_helper_function_with_builder(
                        name.clone(),
                        vec![self.lltype.clone()],
                        LowLevelType::Signed,
                        move |_rtyper, args, result| {
                            build_gcref_wrapper_graph(
                                &name,
                                args,
                                result,
                                &base_lltype,
                                base_hash.clone(),
                                "ptr",
                            )
                        },
                    )?)
                }
            };
            *self._ll_hash_func.borrow_mut() = LlHelperCache::Computed(ll_hash_func);
        }
        match &*self._ll_hash_func.borrow() {
            LlHelperCache::Computed(func) => Ok(func.clone()),
            LlHelperCache::Unknown => unreachable!("_ll_hash_func computed above"),
        }
    }
}

/// RPython `class DummyValueBuilderGCRef(object)` (`rgcref.py:74-104`).
///
/// `ll_dummy_value` (`rgcref.py:93-104`) casts the base-instance dummy
/// (`getinstancerepr(None)` → [`DummyValueBuilder`] over `TYPE.TO`) to
/// `GCREF` and memoises it under `GCREF` in
/// `RPythonTyper.cache_dummy_values`. Since pyre stores only `rtyper_id`
/// for identity, the typer is threaded in at call time.
///
/// The producer side — `GCRefRepr.get_ll_dummyval_obj` (`rgcref.py:58-59`)
/// returning this builder — is deferred with the generic
/// `Repr.get_ll_dummyval_obj`: its only consumers are the unported
/// dict/ordereddict `ENTRIES.dummy_obj` fields.
#[derive(Clone, Debug)]
pub struct DummyValueBuilderGCRef {
    rtyper_id: usize,
}

impl DummyValueBuilderGCRef {
    pub fn new(rtyper: &RPythonTyper) -> Self {
        DummyValueBuilderGCRef {
            rtyper_id: rtyper as *const RPythonTyper as usize,
        }
    }

    pub fn rtyper_id(&self) -> usize {
        self.rtyper_id
    }

    pub fn _freeze_(&self) -> bool {
        true
    }

    pub fn ll_dummy_value(&self, rtyper: &Rc<RPythonTyper>) -> Result<LowLevelValue, TyperError> {
        if let Some(p) = rtyper.cache_dummy_values.borrow().get(&*GCREF) {
            return Ok(p.clone());
        }
        let rinstbase = getinstancerepr(rtyper, None, Flavor::Gc)?;
        let LowLevelType::Ptr(ptr) = rinstbase.lowleveltype() else {
            return Err(TyperError::message(
                "DummyValueBuilderGCRef: instance repr lowleveltype must be Ptr",
            ));
        };
        let type_to = LowLevelType::from(ptr.TO.clone());
        let val = DummyValueBuilder::new(rtyper, type_to).ll_dummy_value(rtyper)?;
        let LowLevelValue::Ptr(inner_ptr) = &val else {
            return Err(TyperError::message(
                "DummyValueBuilderGCRef: base dummy value must be Ptr",
            ));
        };
        let p = cast_opaque_ptr(&gcref_ptr(), inner_ptr).map_err(TyperError::message)?;
        let p = LowLevelValue::Ptr(Box::new(p));
        rtyper
            .cache_dummy_values
            .borrow_mut()
            .insert((*GCREF).clone(), p.clone());
        Ok(p)
    }
}

impl PartialEq for DummyValueBuilderGCRef {
    fn eq(&self, other: &Self) -> bool {
        self.rtyper_id == other.rtyper_id
    }
}

impl Eq for DummyValueBuilderGCRef {}

impl Hash for DummyValueBuilderGCRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        GCREF.clone().hash(state);
    }
}

fn gcref_ptr() -> Box<Ptr> {
    let LowLevelType::Ptr(ptr) = GCREF.clone() else {
        unreachable!("GCREF must be Ptr(GcOpaqueType('GCREF'))");
    };
    ptr
}

fn is_same_repr(a: &dyn Repr, b: &dyn Repr) -> bool {
    std::ptr::eq(
        a as *const dyn Repr as *const (),
        b as *const dyn Repr as *const (),
    )
}

fn as_gcref_repr(repr: &dyn Repr) -> Result<&GCRefRepr, TyperError> {
    let any: &dyn std::any::Any = repr;
    any.downcast_ref::<GCRefRepr>().ok_or_else(|| {
        TyperError::message(format!("expected GCRefRepr, got {}", repr.repr_string()))
    })
}

/// RPython `pairtype(GCRefRepr, Repr).convert_from_to` (`rgcref.py:52-56`).
pub(crate) fn pair_gcref_repr_convert_from_to(
    _r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    if let LowLevelType::Ptr(ptr) = r_to.lowleveltype() {
        if ptr._gckind() == GcKind::Gc {
            let converted = llops
                .genop(
                    "cast_opaque_ptr",
                    vec![v.clone()],
                    GenopResult::LLType(r_to.lowleveltype().clone()),
                )
                .map(Hlvalue::Variable);
            return Ok(converted);
        }
    }
    Ok(None)
}

/// RPython `pairtype(Repr, GCRefRepr).convert_from_to` (`rgcref.py:58-63`).
pub(crate) fn pair_repr_gcref_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let r_to_gcref = as_gcref_repr(r_to)?;
    let base = r_to_gcref.r_base();
    let mut value = v.clone();
    if !is_same_repr(r_from, base.as_ref()) {
        value = llops.convertvar(value, r_from, base.as_ref())?;
    }
    let converted = llops
        .genop(
            "cast_opaque_ptr",
            vec![value],
            GenopResult::LLType(r_to.lowleveltype().clone()),
        )
        .map(Hlvalue::Variable);
    Ok(converted)
}

fn build_gcref_wrapper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    base_lltype: &LowLevelType,
    base_helper: LowLevelFunction,
    arg_prefix: &str,
) -> Result<PyGraph, TyperError> {
    let gcref = GCREF.clone();
    let input_vars: Vec<_> = args
        .iter()
        .enumerate()
        .map(|(i, arg)| variable_with_lltype(&format!("{arg_prefix}{i}"), arg.clone()))
        .collect();
    let startblock = Block::shared(
        input_vars
            .iter()
            .cloned()
            .map(Hlvalue::Variable)
            .collect::<Vec<_>>(),
    );
    let return_var = variable_with_lltype("result", result.clone());
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var.clone()),
    );

    let base_graph = base_helper.graph.as_ref().ok_or_else(|| {
        TyperError::missing_rtype_operation(format!(
            "low-level helper {} has no annotated helper graph",
            base_helper.name
        ))
    })?;
    let func_ptr = getfunctionptr(&base_graph.graph, |value| match value {
        Hlvalue::Variable(v) => v
            .concretetype()
            .ok_or_else(|| TyperError::message("helper graph variable missing concretetype")),
        Hlvalue::Constant(c) => c
            .concretetype
            .clone()
            .ok_or_else(|| TyperError::message("helper graph constant missing concretetype")),
    })?;
    let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
    let mut call_args = vec![Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::LLPtr(Box::new(func_ptr)),
        func_ptr_type,
    ))];
    for (i, input) in input_vars.iter().enumerate() {
        if args.get(i) != Some(&gcref) {
            return Err(TyperError::message(format!(
                "GCRef wrapper expected GCREF arg, got {:?}",
                args.get(i)
            )));
        }
        let casted = variable_with_lltype(&format!("{arg_prefix}{i}_base"), base_lltype.clone());
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "cast_opaque_ptr",
            vec![Hlvalue::Variable(input.clone())],
            Hlvalue::Variable(casted.clone()),
        ));
        call_args.push(Hlvalue::Variable(casted));
    }

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "direct_call",
        call_args,
        Hlvalue::Variable(return_var.clone()),
    ));

    use crate::flowspace::model::BlockRefExt;
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(return_var)],
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
        (0..args.len())
            .map(|i| format!("{arg_prefix}{i}"))
            .collect(),
        func,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::model::Variable;
    use crate::translator::rtyper::rclass::{Flavor, getinstancerepr};
    use crate::translator::rtyper::rint::signed_repr;
    use crate::translator::rtyper::rtyper::RPythonTyper;

    #[test]
    fn make_caches_by_base_repr_identity() {
        let cache = RefCell::new(HashMap::new());
        let base = signed_repr() as Arc<dyn Repr>;
        let a = GCRefRepr::make(base.clone(), &cache);
        let b = GCRefRepr::make(base, &cache);
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.lowleveltype(), &GCREF.clone());
    }

    #[test]
    fn dummy_value_builder_gcref_freezes_and_compares_by_rtyper_identity() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper_a = RPythonTyper::new(&ann);
        let rtyper_b = RPythonTyper::new(&ann);
        let a1 = DummyValueBuilderGCRef::new(&rtyper_a);
        let a2 = DummyValueBuilderGCRef::new(&rtyper_a);
        let b = DummyValueBuilderGCRef::new(&rtyper_b);

        assert!(a1._freeze_());
        assert_eq!(a1, a2);
        assert_ne!(a1, b);

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        a1.hash(&mut h1);
        a2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn dummy_value_builder_gcref_casts_base_instance_dummy_and_caches_under_gcref() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().unwrap();
        let builder = DummyValueBuilderGCRef::new(&rtyper);

        let v1 = builder.ll_dummy_value(&rtyper).expect("gcref dummy value");
        assert!(matches!(v1, LowLevelValue::Ptr(_)));
        assert!(rtyper.cache_dummy_values.borrow().contains_key(&*GCREF));
        // second call returns the cached GCREF cast, not a fresh one.
        let v2 = builder.ll_dummy_value(&rtyper).expect("cached gcref dummy");
        assert_eq!(v1, v2);
    }

    #[test]
    fn externalvsinternal_gcref_true_wraps_gc_pointer_repr() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().unwrap();
        let item = getinstancerepr(&rtyper, None, Flavor::Gc).unwrap() as Arc<dyn Repr>;

        let (external, internal) =
            crate::translator::rtyper::rclass::externalvsinternal(&rtyper, item.clone(), true)
                .unwrap();
        assert!(is_same_repr(external.as_ref(), item.as_ref()));
        assert_eq!(
            internal.repr_class_id(),
            super::super::super::pairtype::ReprClassId::GCRefRepr
        );
        assert_eq!(internal.lowleveltype(), &GCREF.clone());
    }

    #[test]
    fn pair_repr_gcref_emits_cast_opaque_ptr() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().unwrap();
        let base = getinstancerepr(&rtyper, None, Flavor::Gc).unwrap() as Arc<dyn Repr>;
        let gcref = GCRefRepr::make(base.clone(), &rtyper.gcrefreprcache) as Arc<dyn Repr>;
        let var = Variable::named("p");
        var.set_concretetype(Some(base.lowleveltype().clone()));
        let mut llops = LowLevelOpList::new(rtyper, None);

        let converted = pair_repr_gcref_convert_from_to(
            base.as_ref(),
            gcref.as_ref(),
            &Hlvalue::Variable(var),
            &mut llops,
        )
        .unwrap()
        .expect("conversion should emit a value");
        assert_eq!(llops.ops.len(), 1);
        assert_eq!(llops.ops[0].opname, "cast_opaque_ptr");
        let converted_type = match converted {
            Hlvalue::Variable(v) => v.concretetype(),
            Hlvalue::Constant(c) => c.concretetype,
        };
        assert_eq!(converted_type.as_ref(), Some(&GCREF.clone()));
    }
}
