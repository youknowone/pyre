//! `rpython/rtyper/rweakref.py` — RTyping of RPython-level weakrefs.

use std::cell::RefCell;
use std::fmt;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use crate::flowspace::model::{ConstValue, Constant, Hlvalue};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::llmemory;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, GcKind, LowLevelType, LowLevelValue, MallocFlavor, OpaqueType, Ptr, PtrTarget, Struct,
    WEAKREF_PTR, cast_opaque_ptr, frozendict, malloc, nullptr,
};
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{
    RTypeResult, Repr, ReprState, gc_flavor_const, lowlevel_type_const,
};
use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult, HighLevelOp, RPythonTyper};

// ─── BaseWeakRefRepr trait ───────────────────────────────────────────

/// rweakref.py:23-48 `class BaseWeakRefRepr(Repr)`.
///
/// Shared interface for `WeakRefRepr` (native) and
/// `EmulatedWeakRefRepr` (rweakref=False fallback).
pub trait BaseWeakRefReprTrait: Repr {
    /// rweakref.py:58/80 `_weakref_create(hop, v_inst)`.
    fn weakref_create(&self, hop: &HighLevelOp, v_inst: Hlvalue) -> RTypeResult;

    /// rweakref.py:62/91 `_weakref_deref(hop, v_wref)`.
    fn weakref_deref(&self, hop: &HighLevelOp, v_wref: Hlvalue) -> RTypeResult;
}

/// Test whether a `&dyn Repr` is a `BaseWeakRefRepr` subclass.
pub fn is_base_weakref_repr(r: &dyn Repr) -> bool {
    let any_r: &dyn std::any::Any = r;
    any_r.downcast_ref::<WeakRefRepr>().is_some()
        || any_r.downcast_ref::<EmulatedWeakRefRepr>().is_some()
}

/// Downcast `&dyn Repr` to `&dyn BaseWeakRefReprTrait`.
pub fn as_base_weakref_repr(r: &dyn Repr) -> Option<&dyn BaseWeakRefReprTrait> {
    let any_r: &dyn std::any::Any = r;
    if let Some(w) = any_r.downcast_ref::<WeakRefRepr>() {
        return Some(w as &dyn BaseWeakRefReprTrait);
    }
    if let Some(e) = any_r.downcast_ref::<EmulatedWeakRefRepr>() {
        return Some(e as &dyn BaseWeakRefReprTrait);
    }
    None
}

fn ptr_target_as_lowleveltype(target: &PtrTarget) -> LowLevelType {
    match target {
        PtrTarget::Func(t) => LowLevelType::Func(Box::new(t.clone())),
        PtrTarget::Struct(t) => LowLevelType::Struct(Box::new(t.clone())),
        PtrTarget::Array(t) => LowLevelType::Array(Box::new(t.clone())),
        PtrTarget::FixedSizeArray(t) => LowLevelType::FixedSizeArray(Box::new(t.clone())),
        PtrTarget::Opaque(t) => LowLevelType::Opaque(Box::new(t.clone())),
        PtrTarget::ForwardReference(t) => LowLevelType::ForwardReference(Box::new(t.clone())),
    }
}

fn ptr_pointee_type(ptr_type: &LowLevelType) -> Result<LowLevelType, TyperError> {
    let LowLevelType::Ptr(ptr) = ptr_type else {
        return Err(TyperError::message(format!(
            "weakref repr lowleveltype must be Ptr(...), got {ptr_type:?}",
        )));
    };
    Ok(ptr_target_as_lowleveltype(&ptr.TO))
}

fn null_ptr_constant(ptr_type: &LowLevelType) -> Result<Constant, TyperError> {
    let pointee = ptr_pointee_type(ptr_type)?;
    let ptr = nullptr(pointee).map_err(TyperError::message)?;
    Ok(Constant::with_concretetype(
        ConstValue::LLPtr(Box::new(ptr)),
        ptr_type.clone(),
    ))
}

/// rweakref.py:53 `dead_wref = llmemory.dead_wref` — the single prebuilt
/// dead-weakref pointer, a `WeakRefRepr` class attribute (not recomputed
/// per `convert_const`). Delegates to [`llmemory::dead_wref`] so the
/// singleton lives at its upstream home.
fn native_dead_wref_constant() -> Result<Constant, TyperError> {
    Ok(Constant::with_concretetype(
        ConstValue::LLPtr(Box::new(llmemory::dead_wref())),
        WEAKREF_PTR.clone(),
    ))
}

/// rweakref.py:27-39 `BaseWeakRefRepr.convert_const(self, value)` — the
/// shared body. `None` maps to a null pointer; a dead weakref to the
/// per-repr `dead_wref`; a live weakref converts its referent through
/// the referent's own repr (`rtyper.bindingrepr(Constant(instance))`)
/// and hands the low-level instance to the per-repr `do_weakref_create`.
fn base_weakref_convert_const(
    rtyper: &Weak<RPythonTyper>,
    value: &ConstValue,
    null_lowleveltype: &LowLevelType,
    dead_wref: impl FnOnce() -> Result<Constant, TyperError>,
    do_weakref_create: impl FnOnce(&_ptr) -> Result<Constant, TyperError>,
) -> Result<Constant, TyperError> {
    match value {
        ConstValue::None => null_ptr_constant(null_lowleveltype),
        ConstValue::HostObject(obj) if obj.is_weakref() => match obj.weakref_referent() {
            Some(None) => dead_wref(),
            Some(Some(instance)) => {
                // rweakref.py:33-39 — `bk = self.rtyper.annotator.bookkeeper`
                // is implicit in `bindingrepr`.
                let rtyper = rtyper.upgrade().ok_or_else(|| {
                    TyperError::message(
                        "BaseWeakRefRepr.convert_const: rtyper weak reference dropped",
                    )
                })?;
                let instance_value = ConstValue::HostObject(instance.clone());
                let repr = rtyper
                    .bindingrepr(&Hlvalue::Constant(Constant::new(instance_value.clone())))?;
                let llinstance = repr.convert_const(&instance_value)?;
                let ConstValue::LLPtr(llinstance) = &llinstance.value else {
                    return Err(TyperError::message(format!(
                        "BaseWeakRefRepr.convert_const: referent repr produced a \
                         non-pointer low-level constant {:?}",
                        llinstance.value,
                    )));
                };
                do_weakref_create(llinstance)
            }
            None => unreachable!("is_weakref checked above"),
        },
        _ => Err(TyperError::message(format!(
            "BaseWeakRefRepr.convert_const: expected None or weakref.ref, got {value:?}",
        ))),
    }
}

/// rweakref.py:74 `dead_wref = lltype.malloc(lowleveltype.TO,
/// immortal=True, zero=True)` — evaluated once at class-definition time,
/// shared by every `EmulatedWeakRefRepr`. Cached thread-locally for the
/// same single-identity reason as [`native_dead_wref_constant`].
fn emulated_dead_wref_constant(ptr_type: &LowLevelType) -> Result<Constant, TyperError> {
    thread_local! {
        static DEAD_WREF: RefCell<Option<_ptr>> = const { RefCell::new(None) };
    }
    DEAD_WREF.with(|cell| -> Result<Constant, TyperError> {
        if cell.borrow().is_none() {
            let pointee = ptr_pointee_type(ptr_type)?;
            let ptr = malloc(pointee, None, MallocFlavor::Gc, true).map_err(TyperError::message)?;
            *cell.borrow_mut() = Some(ptr);
        }
        let ptr = cell.borrow().as_ref().unwrap().clone();
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(ptr)),
            ptr_type.clone(),
        ))
    })
}

fn gcref_type() -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Opaque(OpaqueType::gc("GCREF")),
    }))
}

fn ref_field_const() -> Result<Constant, TyperError> {
    HighLevelOp::inputconst(&LowLevelType::Void, &ConstValue::byte_str("ref"))
}

// ─── WeakRefRepr ─────────────────────────────────────────────────────

/// rweakref.py:51-64 `class WeakRefRepr(BaseWeakRefRepr)`.
/// `lowleveltype = WeakRefPtr`.
#[derive(Debug)]
pub struct WeakRefRepr {
    state: ReprState,
    /// rweakref.py:24-25 `BaseWeakRefRepr.__init__(self, rtyper): self
    /// .rtyper = rtyper`. `convert_const` reads it to `bindingrepr` a
    /// live prebuilt referent. `Weak` to avoid a refcount cycle, mirroring
    /// `InstanceRepr`'s `rtyper` back-edge.
    rtyper: Weak<RPythonTyper>,
}

impl WeakRefRepr {
    pub fn new(rtyper: Weak<RPythonTyper>) -> Self {
        WeakRefRepr {
            state: ReprState::new(),
            rtyper,
        }
    }

    /// rweakref.py:55-56 `do_weakref_create(self, llinstance):
    /// return llmemory.weakref_create(llinstance)`.
    fn do_weakref_create(&self, llinstance: &_ptr) -> Result<Constant, TyperError> {
        let wref = llmemory::weakref_create(llinstance).map_err(TyperError::message)?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(wref)),
            WEAKREF_PTR.clone(),
        ))
    }
}

impl Repr for WeakRefRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &WEAKREF_PTR
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "WeakRefRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::WeakRefRepr
    }

    /// rweakref.py:27-39 (shared `BaseWeakRefRepr.convert_const`).
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        base_weakref_convert_const(
            &self.rtyper,
            value,
            self.lowleveltype(),
            native_dead_wref_constant,
            |llinstance| self.do_weakref_create(llinstance),
        )
    }

    /// rweakref.py:41-48
    fn rtype_simple_call(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        hop.exception_cannot_occur()?;
        let r_result = hop.r_result.borrow();
        let r_result = r_result
            .as_ref()
            .ok_or_else(|| TyperError::message("WeakRefRepr.rtype_simple_call: no r_result"))?;
        if *r_result.lowleveltype() == LowLevelType::Void {
            Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            ))))
        } else {
            self.weakref_deref(hop, vlist.into_iter().next().unwrap())
        }
    }
}

impl BaseWeakRefReprTrait for WeakRefRepr {
    /// rweakref.py:58-60
    fn weakref_create(&self, hop: &HighLevelOp, v_inst: Hlvalue) -> RTypeResult {
        Ok(hop.genop(
            "weakref_create",
            vec![v_inst],
            GenopResult::LLType(WEAKREF_PTR.clone()),
        ))
    }

    /// rweakref.py:62-64
    fn weakref_deref(&self, hop: &HighLevelOp, v_wref: Hlvalue) -> RTypeResult {
        let r_result = hop.r_result.borrow();
        let r_result = r_result
            .as_ref()
            .ok_or_else(|| TyperError::message("WeakRefRepr._weakref_deref: no r_result"))?;
        Ok(hop.genop(
            "weakref_deref",
            vec![v_wref],
            GenopResult::Repr(r_result.clone()),
        ))
    }
}

impl fmt::Display for WeakRefRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr_string())
    }
}

// ─── EmulatedWeakRefRepr ─────────────────────────────────────────────

/// rweakref.py:67-96 `class EmulatedWeakRefRepr(BaseWeakRefRepr)`.
/// For `rweakref=False`, emulates weakrefs with strong references.
/// `lowleveltype = Ptr(GcStruct('EmulatedWeakRef', ('ref', GCREF)))`.
#[derive(Debug)]
pub struct EmulatedWeakRefRepr {
    state: ReprState,
    lltype: LowLevelType,
    /// rweakref.py:24-25 — see [`WeakRefRepr::rtyper`].
    rtyper: Weak<RPythonTyper>,
}

/// `Ptr(GcStruct('EmulatedWeakRef', ('ref', GCREF)))` (rweakref.py:71-72).
fn emulated_weakref_lltype() -> LowLevelType {
    let gcref = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Opaque(OpaqueType::gc("GCREF")),
    }));
    let flds = frozendict::new(vec![("ref".into(), gcref)]);
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(Struct {
            _name: "EmulatedWeakRef".into(),
            _flds: flds,
            _names: vec!["ref".into()],
            _adtmeths: frozendict::new(vec![]),
            _hints: frozendict::new(vec![]),
            _arrayfld: None,
            _gckind: GcKind::Gc,
            _runtime_type_info: None,
        }),
    }))
}

impl EmulatedWeakRefRepr {
    pub fn new(rtyper: Weak<RPythonTyper>) -> Self {
        EmulatedWeakRefRepr {
            state: ReprState::new(),
            lltype: emulated_weakref_lltype(),
            rtyper,
        }
    }

    /// rweakref.py:75-78
    ///
    /// ```python
    /// def do_weakref_create(self, llinstance):
    ///     p = lltype.malloc(self.lowleveltype.TO, immortal=True)
    ///     p.ref = lltype.cast_opaque_ptr(llmemory.GCREF, llinstance)
    ///     return p
    /// ```
    fn do_weakref_create(&self, llinstance: &_ptr) -> Result<Constant, TyperError> {
        let pointee = ptr_pointee_type(&self.lltype)?;
        let mut p = malloc(pointee, None, MallocFlavor::Gc, true).map_err(TyperError::message)?;
        // `p.ref = cast_opaque_ptr(GCREF, llinstance)` (rweakref.py:77): wrap
        // the instance in a hidden GCREF opaque so `weakref_deref` can reveal
        // the original container, rather than just re-typing the pointer.
        let LowLevelType::Ptr(gcref_ptr) = gcref_type() else {
            unreachable!("gcref_type() is a Ptr");
        };
        let gcref = cast_opaque_ptr(&gcref_ptr, llinstance).map_err(TyperError::message)?;
        p.setattr("ref", LowLevelValue::Ptr(Box::new(gcref)))
            .map_err(TyperError::message)?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(p)),
            self.lltype.clone(),
        ))
    }
}

impl Repr for EmulatedWeakRefRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "EmulatedWeakRefRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::EmulatedWeakRefRepr
    }

    /// rweakref.py:27-39 (shared `BaseWeakRefRepr.convert_const`).
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        base_weakref_convert_const(
            &self.rtyper,
            value,
            &self.lltype,
            || emulated_dead_wref_constant(&self.lltype),
            |llinstance| self.do_weakref_create(llinstance),
        )
    }

    /// rweakref.py:41-48
    fn rtype_simple_call(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        hop.exception_cannot_occur()?;
        let r_result = hop.r_result.borrow();
        let r_result = r_result.as_ref().ok_or_else(|| {
            TyperError::message("EmulatedWeakRefRepr.rtype_simple_call: no r_result")
        })?;
        if *r_result.lowleveltype() == LowLevelType::Void {
            Ok(Some(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            ))))
        } else {
            self.weakref_deref(hop, vlist.into_iter().next().unwrap())
        }
    }
}

impl BaseWeakRefReprTrait for EmulatedWeakRefRepr {
    /// rweakref.py:80-89
    fn weakref_create(&self, hop: &HighLevelOp, v_inst: Hlvalue) -> RTypeResult {
        let c_type = lowlevel_type_const(ptr_pointee_type(&self.lltype)?);
        let c_flags = gc_flavor_const()?;
        let v_ptr = hop
            .genop(
                "malloc",
                vec![c_type, c_flags],
                GenopResult::LLType(self.lltype.clone()),
            )
            .ok_or_else(|| TyperError::message("EmulatedWeakRefRepr: malloc genop failed"))?;
        let v_gcref = hop
            .genop(
                "cast_opaque_ptr",
                vec![v_inst],
                GenopResult::LLType(gcref_type()),
            )
            .ok_or_else(|| {
                TyperError::message("EmulatedWeakRefRepr: cast_opaque_ptr genop failed")
            })?;
        let c_ref = ref_field_const()?;
        hop.genop(
            "setfield",
            vec![v_ptr.clone(), Hlvalue::Constant(c_ref), v_gcref],
            GenopResult::Void,
        );
        Ok(Some(v_ptr))
    }

    /// rweakref.py:91-96
    fn weakref_deref(&self, hop: &HighLevelOp, v_wref: Hlvalue) -> RTypeResult {
        let c_ref = ref_field_const()?;
        let v_gcref = hop
            .genop(
                "getfield",
                vec![v_wref, Hlvalue::Constant(c_ref)],
                GenopResult::LLType(gcref_type()),
            )
            .ok_or_else(|| TyperError::message("EmulatedWeakRefRepr: getfield genop failed"))?;
        let r_result = hop.r_result.borrow();
        let r_result = r_result.as_ref().ok_or_else(|| {
            TyperError::message("EmulatedWeakRefRepr._weakref_deref: no r_result")
        })?;
        Ok(hop.genop(
            "cast_opaque_ptr",
            vec![v_gcref],
            GenopResult::Repr(r_result.clone()),
        ))
    }
}

impl fmt::Display for EmulatedWeakRefRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.repr_string())
    }
}

// ─── rtyper_makerepr dispatch ────────────────────────────────────────

/// rweakref.py:13-17 `SomeWeakRef.rtyper_makerepr`.
pub fn weakref_makerepr(rtyper: &RPythonTyper) -> Result<Arc<dyn Repr>, TyperError> {
    let rweakref = rtyper
        .getconfig()
        .map(|c| c.translation.rweakref)
        .unwrap_or(true);
    let rtyper_weak = Rc::downgrade(&rtyper.self_rc()?);
    if rweakref {
        Ok(Arc::new(WeakRefRepr::new(rtyper_weak)) as Arc<dyn Repr>)
    } else {
        Ok(Arc::new(EmulatedWeakRefRepr::new(rtyper_weak)) as Arc<dyn Repr>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::HostObject;

    #[test]
    fn weakref_convert_const_none_is_null_weakref_ptr() {
        let repr = WeakRefRepr::new(Weak::new());
        let c = repr.convert_const(&ConstValue::None).unwrap();
        assert_eq!(c.concretetype, Some(WEAKREF_PTR.clone()));
        let ConstValue::LLPtr(ptr) = c.value else {
            panic!("None weakref constant must be a low-level null pointer");
        };
        assert!(!ptr.nonzero());
    }

    #[test]
    fn weakref_convert_const_dead_host_weakref_is_dead_wref_ptr() {
        let repr = WeakRefRepr::new(Weak::new());
        let dead = HostObject::new_weakref("weakref.ref", None);
        let c = repr
            .convert_const(&ConstValue::HostObject(dead))
            .expect("dead host weakref must convert to llmemory.dead_wref shape");
        assert_eq!(c.concretetype, Some(WEAKREF_PTR.clone()));
        let ConstValue::LLPtr(ptr) = c.value else {
            panic!("dead weakref constant must be a low-level weakref pointer");
        };
        assert!(ptr.nonzero());
    }

    /// rweakref.py:54 `dead_wref = llmemory.dead_wref` is a single shared
    /// prebuilt value. `_ptr` equality respects container identity, so
    /// two conversions of distinct dead weakrefs must compare equal — the
    /// singleton invariant.
    #[test]
    fn dead_wref_is_a_single_shared_value_across_conversions() {
        let repr = WeakRefRepr::new(Weak::new());
        let c1 = repr
            .convert_const(&ConstValue::HostObject(HostObject::new_weakref(
                "weakref.ref",
                None,
            )))
            .unwrap();
        let c2 = repr
            .convert_const(&ConstValue::HostObject(HostObject::new_weakref(
                "weakref.ref",
                None,
            )))
            .unwrap();
        assert_eq!(c1.value, c2.value);

        let er = EmulatedWeakRefRepr::new(Weak::new());
        let e1 = er
            .convert_const(&ConstValue::HostObject(HostObject::new_weakref(
                "weakref.ref",
                None,
            )))
            .unwrap();
        let e2 = er
            .convert_const(&ConstValue::HostObject(HostObject::new_weakref(
                "weakref.ref",
                None,
            )))
            .unwrap();
        assert_eq!(e1.value, e2.value);
    }

    #[test]
    fn emulated_weakref_convert_const_none_is_null_emulated_ptr() {
        let repr = EmulatedWeakRefRepr::new(Weak::new());
        let c = repr.convert_const(&ConstValue::None).unwrap();
        assert_eq!(c.concretetype, Some(repr.lltype.clone()));
        let ConstValue::LLPtr(ptr) = c.value else {
            panic!("None emulated weakref constant must be a low-level null pointer");
        };
        assert!(!ptr.nonzero());
    }

    fn gc_instance(name: &str) -> _ptr {
        // An instance is a concrete `Ptr(GcStruct)` (an `InstanceRepr`'s
        // lowleveltype), so `cast_opaque_ptr(GCREF, llinstance)` takes the
        // concrete→opaque path and wraps it in a hidden GCREF opaque.
        let struct_t = LowLevelType::Struct(Box::new(Struct::gc(name, vec![])));
        malloc(struct_t, None, MallocFlavor::Gc, true).unwrap()
    }

    #[test]
    fn native_do_weakref_create_yields_nonzero_weakref_ptr() {
        // rweakref.py:55-56 `return llmemory.weakref_create(llinstance)`.
        let repr = WeakRefRepr::new(Weak::new());
        let c = repr.do_weakref_create(&gc_instance("Inst")).unwrap();
        assert_eq!(c.concretetype, Some(WEAKREF_PTR.clone()));
        let ConstValue::LLPtr(ptr) = c.value else {
            panic!("weakref_create must produce a low-level pointer");
        };
        assert!(ptr.nonzero());
    }

    #[test]
    fn emulated_do_weakref_create_stores_gcref_in_ref_field() {
        // rweakref.py:75-78 — malloc EmulatedWeakRef, p.ref = cast_opaque_ptr.
        let repr = EmulatedWeakRefRepr::new(Weak::new());
        let c = repr.do_weakref_create(&gc_instance("Inst")).unwrap();
        assert_eq!(c.concretetype, Some(repr.lltype.clone()));
        let ConstValue::LLPtr(ptr) = c.value else {
            panic!("emulated weakref_create must produce a low-level pointer");
        };
        let LowLevelValue::Ptr(gcref) = ptr.getattr("ref").unwrap() else {
            panic!("EmulatedWeakRef.ref must be a GCREF pointer");
        };
        assert!(gcref.nonzero());
    }

    /// rweakref.py:37 `repr = self.rtyper.bindingrepr(Constant(instance))`
    /// — a live referent needs the typer. With a dropped `Weak`, the live
    /// arm surfaces a structured error instead of panicking.
    #[test]
    fn live_weakref_convert_const_without_rtyper_errors() {
        let repr = WeakRefRepr::new(Weak::new());
        let live = HostObject::new_weakref(
            "weakref.ref",
            Some(HostObject::new_weakref("referent", None)),
        );
        let err = repr
            .convert_const(&ConstValue::HostObject(live))
            .expect_err("live weakref with a dropped rtyper must error");
        assert!(err.to_string().contains("rtyper weak reference dropped"));
    }
}
