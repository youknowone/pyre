//! `rpython/rtyper/llannotation.py` port.
//!
//! Carries the scalar / pointer branches that `SomePtr`,
//! `SomeInteriorPtr`, `SomeLLADTMeth`, `_ptrEntry.compute_annotation()`,
//! and low-level call-site annotation need. Address-family support
//! still lands in later line-by-line ports.
//!
//! Upstream, this module hosts:
//!
//! * `annotation_to_lltype` / `lltype_to_annotation` / `ll_to_annotation`
//!   (llannotation.py:147-200).
//! * `class SomeInteriorPtr(SomePtr)` (llannotation.py:67-70).
//! * `class SomeLLADTMeth(SomeObject)` (llannotation.py:72-93).
//! * `pairtype(SomePtr, SomeInteger)` / `pairtype(SomePtr, SomeObject)`
//!   binary dispatch (llannotation.py:100-128).
//!
//! In addition, upstream's `class SomePtr` (lltype.py:1518-1577) hosts
//! its own `getattr` / `setattr` / `len` / `bool` / `call` methods —
//! the Rust port re-homes those methods here next to `SomeInteriorPtr`
//! so every pointer-annotation surface sits in a single file. The
//! `SomePtr` struct itself mirrors its upstream home at
//! `lltypesystem/lltype.rs` and is re-exported from `annotator/model.rs`
//! so the `SomeValue::Ptr` variant keeps using the annotator-model path.

use std::collections::HashMap;

use super::lltypesystem::lltype;
use crate::annotator::argument::ArgumentsForTranslation;
use crate::annotator::bookkeeper;
use crate::annotator::model::{
    AnnotatorError, KnownType, SomeBool, SomeChar, SomeFloat, SomeInteger, SomeLongFloat,
    SomeObjectBase, SomeObjectTrait, SomePtr, SomeSingleFloat, SomeUnicodeCodePoint, SomeValue,
    SomeValueTag, s_bool, s_none,
};
use crate::flowspace::model::{ConstValue, Constant};
use crate::flowspace::operation::{CanOnlyThrow, HLOperation, OpKind, Specialization};
use crate::tool::pairtype::DoubleDispatchRegistry;

// =====================================================================
// llannotation.py:147-200 — annotation ⇄ lltype helpers.
// =====================================================================

/// RPython `annotation_to_lltype(s_val, info=None)` (llannotation.py:147-169).
pub fn annotation_to_lltype(
    s_val: &SomeValue,
    info: Option<&str>,
) -> Result<lltype::LowLevelType, AnnotatorError> {
    if let SomeValue::InteriorPtr(s_ptr) = s_val {
        let p = &s_ptr.ll_ptrtype;
        if p.offsets
            .iter()
            .any(|offset| matches!(offset, lltype::InteriorOffset::Index(0)))
        {
            let zero_count = p
                .offsets
                .iter()
                .filter(|offset| matches!(offset, lltype::InteriorOffset::Index(0)))
                .count();
            assert_eq!(zero_count, 1);
            let parent_ptr = lltype::Ptr {
                TO: match &*p.PARENTTYPE {
                    lltype::LowLevelType::Struct(t) => lltype::PtrTarget::Struct((**t).clone()),
                    lltype::LowLevelType::Array(t) => lltype::PtrTarget::Array((**t).clone()),
                    lltype::LowLevelType::FixedSizeArray(t) => {
                        lltype::PtrTarget::FixedSizeArray((**t).clone())
                    }
                    lltype::LowLevelType::Opaque(t) => lltype::PtrTarget::Opaque((**t).clone()),
                    other => panic!("SomeInteriorPtr parent must be container type, got {other:?}"),
                },
            };
            return Ok(lltype::LowLevelType::Ptr(Box::new(lltype::Ptr {
                TO: lltype::PtrTarget::Struct(
                    parent_ptr._example()._interior_ptr_type_with_index(&p.TO),
                ),
            })));
        }
        return Ok(lltype::LowLevelType::Ptr(Box::new(lltype::Ptr {
            TO: match &*p.PARENTTYPE {
                lltype::LowLevelType::Struct(t) => lltype::PtrTarget::Struct((**t).clone()),
                lltype::LowLevelType::Array(t) => lltype::PtrTarget::Array((**t).clone()),
                lltype::LowLevelType::FixedSizeArray(t) => {
                    lltype::PtrTarget::FixedSizeArray((**t).clone())
                }
                lltype::LowLevelType::Opaque(t) => lltype::PtrTarget::Opaque((**t).clone()),
                other => panic!("SomeInteriorPtr parent must be container type, got {other:?}"),
            },
        })));
    }
    if let SomeValue::Ptr(s_ptr) = s_val {
        return Ok(lltype::LowLevelType::from(s_ptr.ll_ptrtype.clone()));
    }
    if let SomeValue::Integer(s_int) = s_val {
        return Ok(lltype::build_number(None, s_int.knowntype()));
    }

    let result = match s_val {
        SomeValue::None_(_) | SomeValue::Impossible => Some(lltype::LowLevelType::Void),
        SomeValue::Bool(_) => Some(lltype::LowLevelType::Bool),
        SomeValue::Float(_) => Some(lltype::LowLevelType::Float),
        SomeValue::SingleFloat(_) => Some(lltype::LowLevelType::SingleFloat),
        SomeValue::LongFloat(_) => Some(lltype::LowLevelType::LongFloat),
        SomeValue::Char(_) => Some(lltype::LowLevelType::Char),
        SomeValue::UnicodeCodePoint(_) => Some(lltype::LowLevelType::UniChar),
        _ => None,
    };
    if let Some(result) = result {
        return Ok(result);
    }

    let prefix = info.map(|s| format!("{s}: ")).unwrap_or_default();
    Err(AnnotatorError::new(format!(
        "{prefix}should return a low-level type,\ngot instead {s_val:?}"
    )))
}

/// RPython `lltype_to_annotation(T)` (llannotation.py:172-185).
///
/// Upstream falls back to `SomePtr(T)` / `SomeInteriorPtr(T)` for the
/// non-primitive cases — `SomePtr.__init__` asserts `isinstance(T,
/// Ptr)`, so bare `Struct`/`Array`/`Func`/`Opaque`/`ForwardReference`
/// would raise `AssertionError` at construction. The Rust port makes
/// that rejection explicit: callers must hand in `Ptr(T)` or
/// `InteriorPtr` rather than the bare container type.
pub fn lltype_to_annotation<T>(t: T) -> SomeValue
where
    T: Into<lltype::LowLevelType>,
{
    let ty = t.into();
    match ty {
        lltype::LowLevelType::Void => s_none(),
        lltype::LowLevelType::Bool => s_bool(),
        lltype::LowLevelType::Float => SomeValue::Float(SomeFloat::new()),
        lltype::LowLevelType::SingleFloat => SomeValue::SingleFloat(SomeSingleFloat::new()),
        lltype::LowLevelType::LongFloat => SomeValue::LongFloat(SomeLongFloat::new()),
        lltype::LowLevelType::Char => SomeValue::Char(SomeChar::new(false)),
        lltype::LowLevelType::UniChar => {
            SomeValue::UnicodeCodePoint(SomeUnicodeCodePoint::new(false))
        }
        lltype::LowLevelType::Signed => {
            SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::Int))
        }
        lltype::LowLevelType::Unsigned => {
            SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::Ruint))
        }
        lltype::LowLevelType::SignedLongLong => {
            SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::LongLong))
        }
        lltype::LowLevelType::SignedLongLongLong => SomeValue::Integer(
            SomeInteger::new_with_knowntype(false, KnownType::LongLongLong),
        ),
        lltype::LowLevelType::UnsignedLongLong => {
            SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::ULongLong))
        }
        lltype::LowLevelType::UnsignedLongLongLong => SomeValue::Integer(
            SomeInteger::new_with_knowntype(false, KnownType::ULongLongLong),
        ),
        lltype::LowLevelType::InteriorPtr(t) => SomeValue::InteriorPtr(SomeInteriorPtr::new(*t)),
        lltype::LowLevelType::Ptr(t) => SomeValue::Ptr(SomePtr::new(*t)),
        other => {
            debug_assert!(other.is_container_type());
            panic!(
                "lltype_to_annotation: container type {:?} must be wrapped in Ptr(T) or \
                 InteriorPtr — upstream SomePtr.__init__ asserts isinstance(T, Ptr)",
                other
            )
        }
    }
}

/// RPython `ll_to_annotation(v)` (llannotation.py:190-200).
pub fn ll_to_annotation(v: lltype::LowLevelValue) -> SomeValue {
    if let lltype::LowLevelValue::InteriorPtr(ptr) = v {
        return SomeValue::InteriorPtr(SomeInteriorPtr::new(ptr._TYPE()));
    }
    lltype_to_annotation(lltype::typeOf_value(&v))
}

// =====================================================================
// lltype.py:1530-1577 — SomePtr pointer-specific methods.
// =====================================================================

impl SomePtr {
    /// RPython `SomePtr.bool(self)` (lltype.py:1566-1570).
    pub fn bool(&self) -> SomeValue {
        let mut result = SomeBool::new();
        if self.is_constant()
            && let Some(c) = &self.base.const_box
            && let Some(truthy) = c.value.truthy()
        {
            result.base.const_box = Some(Constant::new(ConstValue::Bool(truthy)));
        }
        SomeValue::Bool(result)
    }

    /// RPython `SomePtr.len(self)` (lltype.py:1550-1555). Propagates the
    /// `TypeError` upstream raises from `_fixedlength` on non-array
    /// pointers as an `AnnotatorError`.
    pub fn len(&self) -> Result<SomeValue, AnnotatorError> {
        let length = self
            .ll_ptrtype
            ._example()
            ._fixedlength()
            .map_err(AnnotatorError::new)?;
        if let Some(length) = length {
            bookkeeper::immutablevalue(&ConstValue::Int(length))
        } else {
            Ok(SomeValue::Integer(SomeInteger::new(true, false)))
        }
    }

    /// RPython `SomePtr.getattr(self, s_attr)` (lltype.py:1531-1548).
    pub fn getattr(&self, s_attr: &SomeValue) -> Result<SomeValue, AnnotatorError> {
        if !s_attr.is_constant() {
            return Err(AnnotatorError::new(format!(
                "getattr on ptr {:?} with non-constant field-name",
                self.ll_ptrtype
            )));
        }
        let Some(ConstValue::Str(attr)) = s_attr.const_() else {
            panic!(
                "SomePtr.getattr expects a constant string field-name, got {:?}",
                s_attr.const_()
            );
        };
        let example = self.ll_ptrtype._example();
        match example._lookup_adtmeth(attr) {
            Err(lltype::AttributeError) => {
                let v = example.getattr(attr).map_err(AnnotatorError::new)?;
                Ok(ll_to_annotation(v))
            }
            Ok(lltype::LowLevelAdtMember::Method { ll_ptrtype, func }) => {
                Ok(SomeValue::LLADTMeth(SomeLLADTMeth::new(ll_ptrtype, func)))
            }
            Ok(lltype::LowLevelAdtMember::Value(v)) => bookkeeper::immutablevalue(&v),
        }
    }

    /// RPython `SomePtr.call(self, args)` (lltype.py:1567-1577).
    pub fn call(&self, args: &ArgumentsForTranslation) -> Result<SomeValue, AnnotatorError> {
        let (args_s, kwds_s) = args
            .unpack()
            .map_err(|err| AnnotatorError::new(err.getmsg()))?;
        if !kwds_s.is_empty() {
            return Err(AnnotatorError::new(
                "keyword arguments to call to a low-level fn ptr",
            ));
        }
        let info = "argument to ll function pointer call";
        let llargs = args_s
            .iter()
            .map(|s_arg| annotation_to_lltype(s_arg, Some(info)).map(|t| t._defl()))
            .collect::<Result<Vec<_>, _>>()?;
        let v = self.ll_ptrtype._example().call(&llargs);
        Ok(ll_to_annotation(v))
    }

    /// RPython `SomePtr.setattr(self, s_attr, s_value)` (lltype.py:1557-1564).
    pub fn setattr(
        &self,
        s_attr: &SomeValue,
        s_value: &SomeValue,
    ) -> Result<SomeValue, AnnotatorError> {
        if !s_attr.is_constant() {
            return Err(AnnotatorError::new(format!(
                "setattr on ptr {:?} with non-constant field-name",
                self.ll_ptrtype
            )));
        }
        let Some(ConstValue::Str(attr)) = s_attr.const_() else {
            panic!(
                "SomePtr.setattr expects a constant string field-name, got {:?}",
                s_attr.const_()
            );
        };
        let mut example = self.ll_ptrtype._example();
        if example.getattr(attr).map_err(AnnotatorError::new)? != lltype::LowLevelValue::Void {
            let v_lltype = annotation_to_lltype(s_value, None)?;
            example
                .setattr(attr, v_lltype._defl())
                .map_err(AnnotatorError::new)?;
        }
        Ok(SomeValue::Impossible)
    }
}

// =====================================================================
// llannotation.py:67-70 — class SomeInteriorPtr(SomePtr)
// =====================================================================

/// RPython `class SomeInteriorPtr(SomePtr)` (llannotation.py:67-70).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeInteriorPtr {
    pub base: SomeObjectBase,
    pub ll_ptrtype: lltype::InteriorPtr,
}

impl SomeInteriorPtr {
    pub fn new(ll_ptrtype: lltype::InteriorPtr) -> Self {
        SomeInteriorPtr {
            base: SomeObjectBase::new(KnownType::LlPtr, true),
            ll_ptrtype,
        }
    }

    pub fn bool(&self) -> SomeValue {
        let mut result = SomeBool::new();
        if self.is_constant()
            && let Some(c) = &self.base.const_box
            && let Some(truthy) = c.value.truthy()
        {
            result.base.const_box = Some(Constant::new(ConstValue::Bool(truthy)));
        }
        SomeValue::Bool(result)
    }

    pub fn len(&self) -> Result<SomeValue, AnnotatorError> {
        let length = self
            .ll_ptrtype
            ._example()
            ._fixedlength()
            .map_err(AnnotatorError::new)?;
        if let Some(length) = length {
            bookkeeper::immutablevalue(&ConstValue::Int(length))
        } else {
            Ok(SomeValue::Integer(SomeInteger::new(true, false)))
        }
    }

    pub fn getattr(&self, s_attr: &SomeValue) -> Result<SomeValue, AnnotatorError> {
        if !s_attr.is_constant() {
            return Err(AnnotatorError::new(format!(
                "getattr on ptr {:?} with non-constant field-name",
                self.ll_ptrtype
            )));
        }
        let Some(ConstValue::Str(attr)) = s_attr.const_() else {
            panic!(
                "SomeInteriorPtr.getattr expects a constant string field-name, got {:?}",
                s_attr.const_()
            );
        };
        let example = self.ll_ptrtype._example();
        match example._lookup_adtmeth(attr) {
            Err(lltype::AttributeError) => {
                let v = example.getattr(attr).map_err(AnnotatorError::new)?;
                Ok(ll_to_annotation(v))
            }
            Ok(lltype::LowLevelAdtMember::Method { ll_ptrtype, func }) => {
                Ok(SomeValue::LLADTMeth(SomeLLADTMeth::new(ll_ptrtype, func)))
            }
            Ok(lltype::LowLevelAdtMember::Value(v)) => bookkeeper::immutablevalue(&v),
        }
    }

    pub fn call(&self, args: &ArgumentsForTranslation) -> Result<SomeValue, AnnotatorError> {
        let (args_s, kwds_s) = args
            .unpack()
            .map_err(|err| AnnotatorError::new(err.getmsg()))?;
        if !kwds_s.is_empty() {
            return Err(AnnotatorError::new(
                "keyword arguments to call to a low-level fn ptr",
            ));
        }
        let info = "argument to ll function pointer call";
        let llargs = args_s
            .iter()
            .map(|s_arg| annotation_to_lltype(s_arg, Some(info)).map(|t| t._defl()))
            .collect::<Result<Vec<_>, _>>()?;
        let v = self.ll_ptrtype._example().call(&llargs);
        Ok(ll_to_annotation(v))
    }

    pub fn setattr(
        &self,
        s_attr: &SomeValue,
        s_value: &SomeValue,
    ) -> Result<SomeValue, AnnotatorError> {
        if !s_attr.is_constant() {
            return Err(AnnotatorError::new(format!(
                "setattr on ptr {:?} with non-constant field-name",
                self.ll_ptrtype
            )));
        }
        let Some(ConstValue::Str(attr)) = s_attr.const_() else {
            panic!(
                "SomeInteriorPtr.setattr expects a constant string field-name, got {:?}",
                s_attr.const_()
            );
        };
        let mut example = self.ll_ptrtype._example();
        if example.getattr(attr).map_err(AnnotatorError::new)? != lltype::LowLevelValue::Void {
            let v_lltype = annotation_to_lltype(s_value, None)?;
            example
                .setattr(attr, v_lltype._defl())
                .map_err(AnnotatorError::new)?;
        }
        Ok(SomeValue::Impossible)
    }
}

impl SomeObjectTrait for SomeInteriorPtr {
    fn knowntype(&self) -> KnownType {
        KnownType::LlPtr
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

// =====================================================================
// llannotation.py:72-93 — class SomeLLADTMeth(SomeObject)
// =====================================================================

/// RPython `class SomeLLADTMeth(SomeObject)` (llannotation.py:72-83).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeLLADTMeth {
    pub base: SomeObjectBase,
    pub ll_ptrtype: lltype::LowLevelPointerType,
    pub func: ConstValue,
}

impl SomeLLADTMeth {
    pub fn new(ll_ptrtype: lltype::LowLevelPointerType, func: ConstValue) -> Self {
        SomeLLADTMeth {
            base: SomeObjectBase::new(KnownType::Object, true),
            ll_ptrtype,
            func,
        }
    }

    /// RPython `SomeLLADTMeth.call(self, args)` (llannotation.py:82-87).
    pub fn call(&self, args: &ArgumentsForTranslation) -> Result<SomeValue, AnnotatorError> {
        let s_func = bookkeeper::immutablevalue(&self.func)?;
        s_func.call(&args.prepend(lltype_to_annotation(self.ll_ptrtype.clone())))
    }
}

impl SomeObjectTrait for SomeLLADTMeth {
    fn knowntype(&self) -> KnownType {
        KnownType::Object
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

// =====================================================================
// llannotation.py:100-128 — pairtype(SomePtr, SomeInteger/Object)
// =====================================================================

/// Register the `SomePtr`/`SomeInteger` and `SomePtr`/`SomeObject`
/// pair dispatch tables. Invoked from `annotator::binaryop::init` at
/// `_REGISTRY_DOUBLE` construction time; upstream runs as the `class
/// __extend__(pairtype(...))` module-import side effect.
pub fn init_pairtypes(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    init_ptr_integer_pairtype(reg);
    init_ptr_object_pairtype(reg);
}

fn register(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
    op: OpKind,
    tag1: SomeValueTag,
    tag2: SomeValueTag,
    spec: Specialization,
) {
    reg.entry(op).or_default().set((tag1, tag2), spec);
}

fn init_ptr_integer_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::Ptr,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(ptr_integer_getitem),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::SetItem,
        SomeValueTag::Ptr,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(ptr_integer_setitem),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

fn ptr_integer_getitem(
    ann: &crate::annotator::annrpython::RPythonAnnotator,
    hl: &HLOperation,
) -> SomeValue {
    use crate::annotator::model::s_impossible_value;
    match ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible) {
        SomeValue::Ptr(p) => match p.ll_ptrtype._example().getitem(0) {
            Ok(v) => ll_to_annotation(v),
            Err(err) if err.contains("out of bounds") => s_impossible_value(),
            Err(err) => panic!("{err}"),
        },
        SomeValue::InteriorPtr(p) => match p.ll_ptrtype._example().getitem(0) {
            Ok(v) => ll_to_annotation(v),
            Err(err) if err.contains("out of bounds") => s_impossible_value(),
            Err(err) => panic!("{err}"),
        },
        _ => panic!("ptr_integer_getitem: arg 0 not ptr-like"),
    }
}

fn ptr_integer_setitem(
    ann: &crate::annotator::annrpython::RPythonAnnotator,
    hl: &HLOperation,
) -> SomeValue {
    let s_value = ann
        .annotation(&hl.args[2])
        .expect("ptr_integer_setitem: missing value arg");
    match ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible) {
        SomeValue::Ptr(p) => {
            let mut example = p.ll_ptrtype._example();
            match example.getitem(0) {
                Ok(lltype::LowLevelValue::Void) => {}
                Ok(_) => {
                    let v_lltype = annotation_to_lltype(&s_value, None)
                        .expect("ptr_integer_setitem: annotation_to_lltype failed");
                    example
                        .setitem(0, v_lltype._defl())
                        .unwrap_or_else(|err| panic!("{err}"));
                }
                Err(err) => panic!("{err}"),
            }
        }
        SomeValue::InteriorPtr(p) => {
            let mut example = p.ll_ptrtype._example();
            match example.getitem(0) {
                Ok(lltype::LowLevelValue::Void) => {}
                Ok(_) => {
                    let v_lltype = annotation_to_lltype(&s_value, None)
                        .expect("ptr_integer_setitem: annotation_to_lltype failed");
                    example
                        .setitem(0, v_lltype._defl())
                        .unwrap_or_else(|err| panic!("{err}"));
                }
                Err(err) => panic!("{err}"),
            }
        }
        _ => panic!("ptr_integer_setitem: arg 0 not ptr-like"),
    }
    SomeValue::Impossible
}

fn init_ptr_object_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::Ptr,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|ann, hl| {
                let p = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
                let obj = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
                panic!(
                    "AnnotatorError: ptr {:?} getitem index not an int: {:?}",
                    match p {
                        SomeValue::Ptr(p) => lltype::LowLevelType::from(p.ll_ptrtype),
                        SomeValue::InteriorPtr(p) => lltype::LowLevelType::from(p.ll_ptrtype),
                        _ => panic!("ptr_object_getitem: arg 0 not ptr-like"),
                    },
                    obj
                );
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    register(
        reg,
        OpKind::SetItem,
        SomeValueTag::Ptr,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|ann, hl| {
                let p = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
                let obj = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
                panic!(
                    "AnnotatorError: ptr {:?} setitem index not an int: {:?}",
                    match p {
                        SomeValue::Ptr(p) => lltype::LowLevelType::from(p.ll_ptrtype),
                        SomeValue::InteriorPtr(p) => lltype::LowLevelType::from(p.ll_ptrtype),
                        _ => panic!("ptr_object_setitem: arg 0 not ptr-like"),
                    },
                    obj
                );
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{KnownType, SomeBool};

    #[test]
    fn annotation_to_lltype_maps_scalar_and_pointer_annotations() {
        let signed = annotation_to_lltype(
            &SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::Int)),
            None,
        )
        .unwrap();
        assert_eq!(signed, lltype::LowLevelType::Signed);

        let unsigned = annotation_to_lltype(
            &SomeValue::Integer(SomeInteger::new_with_knowntype(false, KnownType::Ruint)),
            None,
        )
        .unwrap();
        assert_eq!(unsigned, lltype::LowLevelType::Unsigned);

        let ptr = lltype::Ptr {
            TO: lltype::PtrTarget::Func(lltype::FuncType {
                args: vec![lltype::LowLevelType::Signed],
                result: lltype::LowLevelType::Void,
            }),
        };
        let s_ptr = SomeValue::Ptr(SomePtr::new(ptr.clone()));
        assert_eq!(
            annotation_to_lltype(&s_ptr, None).unwrap(),
            lltype::LowLevelType::from(ptr)
        );
        assert_eq!(
            annotation_to_lltype(&SomeValue::Bool(SomeBool::new()), None).unwrap(),
            lltype::LowLevelType::Bool
        );

        let s_interior = SomeValue::InteriorPtr(SomeInteriorPtr::new(lltype::InteriorPtr {
            PARENTTYPE: Box::new(lltype::LowLevelType::Struct(Box::new(
                lltype::StructType::new("S", vec![("x".into(), lltype::LowLevelType::Signed)]),
            ))),
            TO: Box::new(lltype::LowLevelType::Signed),
            offsets: vec![lltype::InteriorOffset::Field("x".into())],
        }));
        assert!(matches!(
            annotation_to_lltype(&s_interior, None).unwrap(),
            lltype::LowLevelType::Ptr(_)
        ));
    }

    #[test]
    fn lltype_to_annotation_maps_scalars_and_ptrs() {
        assert!(matches!(
            lltype_to_annotation(lltype::LowLevelType::Void),
            SomeValue::None_(_)
        ));
        assert!(matches!(
            lltype_to_annotation(lltype::LowLevelType::Signed),
            SomeValue::Integer(_)
        ));
        let SomeValue::Integer(i) = lltype_to_annotation(lltype::LowLevelType::SignedLongLong)
        else {
            panic!("expected SomeInteger");
        };
        assert_eq!(i.knowntype(), KnownType::LongLong);
        let SomeValue::Integer(i) = lltype_to_annotation(lltype::LowLevelType::UnsignedLongLong)
        else {
            panic!("expected SomeInteger");
        };
        assert_eq!(i.knowntype(), KnownType::ULongLong);
        assert!(i.unsigned);
        assert!(matches!(
            lltype_to_annotation(lltype::LowLevelType::Bool),
            SomeValue::Bool(_)
        ));
        assert!(matches!(
            lltype_to_annotation(lltype::LowLevelType::UniChar),
            SomeValue::UnicodeCodePoint(_)
        ));

        let ptr = lltype::Ptr {
            TO: lltype::PtrTarget::Func(lltype::FuncType {
                args: vec![],
                result: lltype::LowLevelType::Void,
            }),
        };
        let SomeValue::Ptr(s_ptr) = lltype_to_annotation(ptr.clone()) else {
            panic!("expected SomePtr");
        };
        assert_eq!(s_ptr.ll_ptrtype, ptr);

        let interior = lltype::InteriorPtr {
            PARENTTYPE: Box::new(lltype::LowLevelType::Struct(Box::new(
                lltype::StructType::new("S", vec![("x".into(), lltype::LowLevelType::Signed)]),
            ))),
            TO: Box::new(lltype::LowLevelType::Signed),
            offsets: vec![lltype::InteriorOffset::Field("x".into())],
        };
        let SomeValue::InteriorPtr(s_ptr) = lltype_to_annotation(
            lltype::LowLevelType::InteriorPtr(Box::new(interior.clone())),
        ) else {
            panic!("expected SomeInteriorPtr");
        };
        assert_eq!(s_ptr.ll_ptrtype, interior);
    }

    #[test]
    #[should_panic(expected = "must be wrapped in Ptr(T)")]
    fn lltype_to_annotation_rejects_bare_struct() {
        // upstream llannotation.py:184-185 falls back to SomePtr(T),
        // whose __init__ asserts T is a Ptr. A bare Struct therefore
        // trips the SomePtr assertion; the Rust port surfaces this as
        // an explicit panic at the dispatch site.
        let _ = lltype_to_annotation(lltype::LowLevelType::Struct(Box::new(
            lltype::StructType::new("S", vec![("x".into(), lltype::LowLevelType::Signed)]),
        )));
    }

    #[test]
    #[should_panic(expected = "must be wrapped in Ptr(T)")]
    fn lltype_to_annotation_rejects_bare_func() {
        let _ = lltype_to_annotation(lltype::LowLevelType::Func(Box::new(lltype::FuncType {
            args: vec![],
            result: lltype::LowLevelType::Void,
        })));
    }

    #[test]
    #[should_panic(expected = "must be wrapped in Ptr(T)")]
    fn lltype_to_annotation_rejects_bare_array() {
        let _ = lltype_to_annotation(lltype::LowLevelType::Array(Box::new(
            lltype::ArrayType::new(lltype::LowLevelType::Signed),
        )));
    }

    #[test]
    fn ll_to_annotation_maps_low_level_values() {
        assert!(matches!(
            ll_to_annotation(lltype::LowLevelValue::Void),
            SomeValue::None_(_)
        ));
        assert!(matches!(
            ll_to_annotation(lltype::LowLevelValue::Signed(0)),
            SomeValue::Integer(_)
        ));
        assert!(matches!(
            ll_to_annotation(lltype::LowLevelValue::Bool(false)),
            SomeValue::Bool(_)
        ));
        assert!(matches!(
            ll_to_annotation(lltype::LowLevelValue::Char('\0')),
            SomeValue::Char(_)
        ));

        let interior = lltype::_interior_ptr {
            _T: lltype::LowLevelType::Signed,
            _parent: lltype::LowLevelValue::Struct(Box::new(
                lltype::StructType::new("S", vec![("x".into(), lltype::LowLevelType::Signed)])
                    ._container_example(),
            )),
            _offsets: vec![lltype::InteriorOffset::Field("x".into())],
        };
        assert!(matches!(
            ll_to_annotation(lltype::LowLevelValue::InteriorPtr(Box::new(interior))),
            SomeValue::InteriorPtr(_)
        ));
    }

    #[test]
    #[should_panic(expected = "must be wrapped in Ptr(T)")]
    fn ll_to_annotation_rejects_bare_struct_value() {
        let value = lltype::LowLevelValue::Struct(Box::new(
            lltype::StructType::new("S", vec![("x".into(), lltype::LowLevelType::Signed)])
                ._container_example(),
        ));
        let _ = ll_to_annotation(value);
    }
}
