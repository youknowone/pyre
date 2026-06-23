//! RPython `rpython/rtyper/lltypesystem/ll2ctypes.py`.
//!
//! Upstream bridges live lltype containers to Python's `ctypes` runtime. Pyre
//! does not embed Python ctypes, but the module name and public conversion entry
//! points are still part of the RPython parity surface: `rffi.cast()` lowers via
//! `force_cast`, `rffi.ptradd()` specializes through `force_ptradd`, and address
//! helpers refer to `cast_adr_to_int`. This file mirrors that surface and
//! implements the parts that are purely lltype-structural; runtime ctypes
//! allocation/conversion returns an explicit `MissingRTypeOperation`.
#![allow(non_camel_case_types, non_snake_case)]

use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    _address, ArrayType, LowLevelType, PtrTarget,
};

pub const _POSIX: bool = cfg!(unix);
pub const _MS_WINDOWS: bool = cfg!(windows);
pub const _FREEBSD: bool = cfg!(target_os = "freebsd");
pub const _64BIT: bool = usize::BITS == 64;

/// RPython `far_regions = None`.
pub const FAR_REGIONS_ENABLED: bool = false;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CTypesType {
    Void,
    Bool,
    Signed,
    Unsigned,
    SignedLongLong,
    UnsignedLongLong,
    Float,
    SingleFloat,
    LongFloat,
    Char,
    UniChar,
    Address,
    Ptr(Box<LowLevelType>),
    Struct(String),
    Array(Box<LowLevelType>),
    FixedSizeArray {
        of: Box<LowLevelType>,
        length: usize,
    },
    Opaque(String),
    Func {
        args: Vec<LowLevelType>,
        result: Box<LowLevelType>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CTypesValue {
    Void,
    Bool(bool),
    Int(i64),
    Float(u64),
    Byte(u8),
    UniChar(char),
    Address(i64),
    Deferred {
        typ: CTypesType,
        reason: &'static str,
    },
}

pub fn allocate_ctypes(ctype: &CTypesType) -> Result<CTypesValue, TyperError> {
    Err(TyperError::missing_rtype_operation(format!(
        "ll2ctypes.allocate_ctypes({ctype:?}) requires runtime ctypes storage"
    )))
}

pub fn do_allocation_in_far_regions() -> Result<(), TyperError> {
    Err(TyperError::missing_rtype_operation(
        "ll2ctypes.do_allocation_in_far_regions requires mmap-backed ctypes storage",
    ))
}

pub fn get_rtyper() -> Result<(), TyperError> {
    Err(TyperError::missing_rtype_operation(
        "ll2ctypes.get_rtyper runtime hook deferred",
    ))
}

/// RPython `get_ctypes_type(T)` for the structural subset pyre can represent
/// without Python ctypes classes.
pub fn get_ctypes_type(T: &LowLevelType) -> Result<CTypesType, TyperError> {
    match T {
        LowLevelType::Void => Ok(CTypesType::Void),
        LowLevelType::Bool => Ok(CTypesType::Bool),
        LowLevelType::Signed => Ok(CTypesType::Signed),
        LowLevelType::Unsigned => Ok(CTypesType::Unsigned),
        LowLevelType::SignedLongLong | LowLevelType::SignedLongLongLong => {
            Ok(CTypesType::SignedLongLong)
        }
        LowLevelType::UnsignedLongLong | LowLevelType::UnsignedLongLongLong => {
            Ok(CTypesType::UnsignedLongLong)
        }
        LowLevelType::Float => Ok(CTypesType::Float),
        LowLevelType::SingleFloat => Ok(CTypesType::SingleFloat),
        LowLevelType::LongFloat => Ok(CTypesType::LongFloat),
        LowLevelType::Char => Ok(CTypesType::Char),
        LowLevelType::UniChar => Ok(CTypesType::UniChar),
        LowLevelType::Address => Ok(CTypesType::Address),
        LowLevelType::Ptr(ptr) => Ok(CTypesType::Ptr(Box::new(match &ptr.TO {
            PtrTarget::Func(t) => LowLevelType::Func(Box::new(t.clone())),
            PtrTarget::Struct(t) => LowLevelType::Struct(Box::new(t.clone())),
            PtrTarget::Array(t) => LowLevelType::Array(Box::new(t.clone())),
            PtrTarget::FixedSizeArray(t) => LowLevelType::FixedSizeArray(Box::new(t.clone())),
            PtrTarget::Opaque(t) => LowLevelType::Opaque(Box::new(t.clone())),
            PtrTarget::ForwardReference(t) => LowLevelType::ForwardReference(Box::new(t.clone())),
        }))),
        LowLevelType::Struct(t) => Ok(CTypesType::Struct(t._name.clone())),
        LowLevelType::Array(t) => Ok(CTypesType::Array(Box::new(t.OF.clone()))),
        LowLevelType::FixedSizeArray(t) => Ok(CTypesType::FixedSizeArray {
            of: Box::new(t.OF.clone()),
            length: t.length,
        }),
        LowLevelType::Opaque(t) => Ok(CTypesType::Opaque(t.tag.clone())),
        LowLevelType::Func(t) => Ok(CTypesType::Func {
            args: t.args.clone(),
            result: Box::new(t.result.clone()),
        }),
        LowLevelType::ForwardReference(t) => {
            let Some(real) = t.resolved() else {
                return Err(TyperError::missing_rtype_operation(
                    "ll2ctypes.get_ctypes_type unresolved ForwardReference",
                ));
            };
            get_ctypes_type(&real)
        }
        LowLevelType::InteriorPtr(_) => Err(TyperError::missing_rtype_operation(
            "ll2ctypes.get_ctypes_type InteriorPtr runtime bridge deferred",
        )),
    }
}

pub fn build_ctypes_struct(T: &LowLevelType) -> Result<CTypesType, TyperError> {
    match T {
        LowLevelType::Struct(_) => get_ctypes_type(T),
        other => Err(TyperError::message(format!(
            "build_ctypes_struct expects Struct, got {other:?}"
        ))),
    }
}

pub fn build_ctypes_array(T: &LowLevelType) -> Result<CTypesType, TyperError> {
    match T {
        LowLevelType::Array(_) | LowLevelType::FixedSizeArray(_) => get_ctypes_type(T),
        other => Err(TyperError::message(format!(
            "build_ctypes_array expects Array, got {other:?}"
        ))),
    }
}

pub fn get_ctypes_array_of_size(
    FIELDTYPE: &LowLevelType,
    max_n: usize,
) -> Result<CTypesType, TyperError> {
    Ok(CTypesType::FixedSizeArray {
        of: Box::new(FIELDTYPE.clone()),
        length: max_n,
    })
}

pub fn lltype2ctypes(llobj: &ConstValue) -> Result<CTypesValue, TyperError> {
    match llobj {
        ConstValue::Bool(value) => Ok(CTypesValue::Bool(*value)),
        ConstValue::Int(value) => Ok(CTypesValue::Int(*value)),
        ConstValue::Float(value) => Ok(CTypesValue::Float(*value)),
        ConstValue::ByteStr(bytes) if bytes.len() == 1 => Ok(CTypesValue::Byte(bytes[0])),
        ConstValue::UniStr(text) if text.chars().count() == 1 => {
            Ok(CTypesValue::UniChar(text.chars().next().unwrap()))
        }
        ConstValue::None => Ok(CTypesValue::Address(0)),
        other => Err(TyperError::missing_rtype_operation(format!(
            "ll2ctypes.lltype2ctypes runtime conversion deferred for {other:?}"
        ))),
    }
}

pub fn ctypes2lltype(T: &LowLevelType, cobj: &CTypesValue) -> Result<ConstValue, TyperError> {
    match (T, cobj) {
        (LowLevelType::Void, CTypesValue::Void) => Ok(ConstValue::None),
        (LowLevelType::Bool, CTypesValue::Bool(value)) => Ok(ConstValue::Bool(*value)),
        (LowLevelType::Signed, CTypesValue::Int(value))
        | (LowLevelType::Unsigned, CTypesValue::Int(value))
        | (LowLevelType::SignedLongLong, CTypesValue::Int(value))
        | (LowLevelType::SignedLongLongLong, CTypesValue::Int(value))
        | (LowLevelType::UnsignedLongLong, CTypesValue::Int(value))
        | (LowLevelType::UnsignedLongLongLong, CTypesValue::Int(value)) => {
            Ok(ConstValue::Int(*value))
        }
        (LowLevelType::Float, CTypesValue::Float(value))
        | (LowLevelType::SingleFloat, CTypesValue::Float(value))
        | (LowLevelType::LongFloat, CTypesValue::Float(value)) => Ok(ConstValue::Float(*value)),
        (LowLevelType::Char, CTypesValue::Byte(value)) => Ok(ConstValue::ByteStr(vec![*value])),
        (LowLevelType::UniChar, CTypesValue::UniChar(value)) => {
            Ok(ConstValue::UniStr(value.to_string()))
        }
        (LowLevelType::Ptr(_), CTypesValue::Address(0)) => Ok(ConstValue::None),
        _ => Err(TyperError::missing_rtype_operation(format!(
            "ll2ctypes.ctypes2lltype runtime conversion deferred for {T:?} <- {cobj:?}"
        ))),
    }
}

pub fn uninitialized2ctypes(T: &LowLevelType) -> Result<CTypesValue, TyperError> {
    let typ = get_ctypes_type(T)?;
    Ok(CTypesValue::Deferred {
        typ,
        reason: "uninitialized",
    })
}

pub fn force_cast(RESTYPE: &LowLevelType, value: &ConstValue) -> Result<ConstValue, TyperError> {
    if RESTYPE.contains_value(value) {
        return Ok(value.clone());
    }
    match (RESTYPE, value) {
        (LowLevelType::Bool, ConstValue::Int(value)) => Ok(ConstValue::Bool(*value != 0)),
        (
            LowLevelType::Signed
            | LowLevelType::Unsigned
            | LowLevelType::SignedLongLong
            | LowLevelType::SignedLongLongLong
            | LowLevelType::UnsignedLongLong
            | LowLevelType::UnsignedLongLongLong,
            ConstValue::Bool(value),
        ) => Ok(ConstValue::Int(if *value { 1 } else { 0 })),
        (LowLevelType::Char, ConstValue::Int(value)) if (0..=255).contains(value) => {
            Ok(ConstValue::ByteStr(vec![*value as u8]))
        }
        (LowLevelType::Signed, ConstValue::ByteStr(bytes)) if bytes.len() == 1 => {
            Ok(ConstValue::Int(bytes[0] as i64))
        }
        _ => Err(TyperError::missing_rtype_operation(format!(
            "ll2ctypes.force_cast({RESTYPE:?}, {value:?}) requires runtime ctypes cast"
        ))),
    }
}

pub fn typecheck_ptradd(T: &LowLevelType) -> Result<&ArrayType, TyperError> {
    let LowLevelType::Ptr(ptr) = T else {
        return Err(TyperError::message("ptradd() expects a pointer type"));
    };
    let PtrTarget::Array(array) = &ptr.TO else {
        return Err(TyperError::message("ptradd() expects Ptr(Array)"));
    };
    if !matches!(array._hints.get("nolength"), Some(ConstValue::Bool(true))) {
        return Err(TyperError::message(
            "ptradd() expects a pointer to a no-length array",
        ));
    }
    Ok(array)
}

pub fn force_ptradd(
    ptr_type: &LowLevelType,
    _ptr_value: &ConstValue,
    _n: i64,
) -> Result<ConstValue, TyperError> {
    typecheck_ptradd(ptr_type)?;
    Err(TyperError::missing_rtype_operation(
        "ll2ctypes.force_ptradd requires runtime pointer storage; rtyper lowers it to direct_ptradd",
    ))
}

pub fn cast_adr_to_int(addr: &ConstValue) -> Result<i64, TyperError> {
    match addr {
        ConstValue::Int(value) => Ok(*value),
        ConstValue::None => Ok(0),
        ConstValue::LLAddress(_address::Null) => Ok(0),
        ConstValue::LLAddress(_address::IntCast(value)) => Ok(*value),
        ConstValue::LLAddress(_address::Fake(_)) => Err(TyperError::missing_rtype_operation(
            "ll2ctypes.cast_adr_to_int for fakeaddress requires runtime ctypes storage",
        )),
        other => Err(TyperError::missing_rtype_operation(format!(
            "ll2ctypes.cast_adr_to_int runtime conversion deferred for {other:?}"
        ))),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotCtypesAllocatedStructure;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _parentable_mixin;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _struct_mixin;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _fixedsizedarray_mixin;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _array_mixin;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _array_of_unknown_length;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _array_of_known_length;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _lladdress {
    pub intval: i64,
}

impl _lladdress {
    pub fn new(intval: i64) -> Self {
        _lladdress { intval }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _llgcopaque {
    pub intval: i64,
}

impl _llgcopaque {
    pub fn new(intval: i64) -> Self {
        _llgcopaque { intval }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ForceCastEntry;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ForcePtrAddEntry;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CastAdrToIntEntry;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LL2CtypesCallable;

pub fn get_ctypes_callable() -> Result<LL2CtypesCallable, TyperError> {
    Err(TyperError::missing_rtype_operation(
        "ll2ctypes.get_ctypes_callable runtime callback bridge deferred",
    ))
}

pub fn get_ctypes_trampoline() -> Result<LL2CtypesCallable, TyperError> {
    Err(TyperError::missing_rtype_operation(
        "ll2ctypes.get_ctypes_trampoline runtime callback bridge deferred",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::lltypesystem::lltype::{ArrayType, Ptr};

    #[test]
    fn get_ctypes_type_classifies_primitives_and_arrays() {
        assert_eq!(
            get_ctypes_type(&LowLevelType::Signed).unwrap(),
            CTypesType::Signed
        );
        let array = LowLevelType::Array(Box::new(ArrayType::with_hints(
            LowLevelType::Char,
            vec![("nolength".into(), ConstValue::Bool(true))],
        )));
        assert_eq!(
            get_ctypes_type(&array).unwrap(),
            CTypesType::Array(Box::new(LowLevelType::Char))
        );
    }

    #[test]
    fn force_cast_handles_exact_scalar_cases_without_runtime_ctypes() {
        assert_eq!(
            force_cast(&LowLevelType::Bool, &ConstValue::Int(1)).unwrap(),
            ConstValue::Bool(true)
        );
        assert_eq!(
            force_cast(&LowLevelType::Char, &ConstValue::Int(65)).unwrap(),
            ConstValue::byte_str("A")
        );
        assert!(force_cast(&LowLevelType::Char, &ConstValue::Int(256)).is_err());
    }

    #[test]
    fn typecheck_ptradd_requires_ptr_to_nolength_array() {
        let ptr_type = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(ArrayType::with_hints(
                LowLevelType::Char,
                vec![("nolength".into(), ConstValue::Bool(true))],
            )),
        }));
        assert!(typecheck_ptradd(&ptr_type).is_ok());

        let sized_ptr_type = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Array(ArrayType::new(LowLevelType::Char)),
        }));
        assert!(typecheck_ptradd(&sized_ptr_type).is_err());
    }
}
