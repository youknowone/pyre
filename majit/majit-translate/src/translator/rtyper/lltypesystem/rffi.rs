//! RPython `rpython/rtyper/lltypesystem/rffi.py`.
//!
//! This slice exposes the exact lltype-facing surface that can be represented
//! with pyre's current `lltype` port: errno flags, C array/struct helper
//! constructors, callback pointer types, and the standard raw pointer aliases.
//! Width-specific C integer aliases and full wrapper-generation behavior remain
//! deferred until the corresponding RPython platform/type metadata is ported.
#![allow(non_snake_case)]

use std::sync::LazyLock;

use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, ArrayType, FixedSizeArrayType, FuncType, LowLevelType, OpaqueType, Ptr, PtrTarget,
    StructType, functionptr_with_external_name,
};

/// RPython `RFFI_SAVE_ERRNO` and related bit flags (`rffi.py:62-73`).
pub const RFFI_SAVE_ERRNO: i64 = 1;
pub const RFFI_READSAVED_ERRNO: i64 = 2;
pub const RFFI_ZERO_ERRNO_BEFORE: i64 = 4;
pub const RFFI_FULL_ERRNO: i64 = RFFI_SAVE_ERRNO | RFFI_READSAVED_ERRNO;
pub const RFFI_FULL_ERRNO_ZERO: i64 = RFFI_SAVE_ERRNO | RFFI_ZERO_ERRNO_BEFORE;
pub const RFFI_SAVE_LASTERROR: i64 = 8;
pub const RFFI_READSAVED_LASTERROR: i64 = 16;
pub const RFFI_SAVE_WSALASTERROR: i64 = 32;
pub const RFFI_FULL_LASTERROR: i64 = RFFI_SAVE_LASTERROR | RFFI_READSAVED_LASTERROR;
pub const RFFI_ERR_NONE: i64 = 0;
pub const RFFI_ERR_ALL: i64 = RFFI_FULL_ERRNO | RFFI_FULL_LASTERROR;
pub const RFFI_ALT_ERRNO: i64 = 64;

/// RPython primitive aliases (`rffi.py:739-779`).
pub const CHAR: LowLevelType = LowLevelType::Char;
pub const DOUBLE: LowLevelType = LowLevelType::Float;
pub const LONGDOUBLE: LowLevelType = LowLevelType::LongFloat;
pub const FLOAT: LowLevelType = LowLevelType::SingleFloat;
pub const SIGNED: LowLevelType = LowLevelType::Signed;
pub const UNSIGNED: LowLevelType = LowLevelType::Unsigned;

fn ptr_to_array(of: LowLevelType, hints: Vec<(String, ConstValue)>) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(ArrayType::with_hints(of, hints)),
    }))
}

fn nolength_hints() -> Vec<(String, ConstValue)> {
    vec![("nolength".into(), ConstValue::Bool(true))]
}

fn void_hints(render_as_const: bool) -> Vec<(String, ConstValue)> {
    let mut hints = nolength_hints();
    hints.push(("render_as_void".into(), ConstValue::Bool(true)));
    if render_as_const {
        hints.push(("render_as_const".into(), ConstValue::Bool(true)));
    }
    hints
}

fn const_hints() -> Vec<(String, ConstValue)> {
    let mut hints = nolength_hints();
    hints.push(("render_as_const".into(), ConstValue::Bool(true)));
    hints
}

/// `void *` (`rffi.py:746`), represented by upstream as a no-length char array
/// pointer with the `render_as_void` hint.
pub static VOIDP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LowLevelType::Char, void_hints(false)));

/// `const void *` (`rffi.py:747`).
pub static CONST_VOIDP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LowLevelType::Char, void_hints(true)));

/// `void **` (`rffi.py:751`).
pub static VOIDPP: LazyLock<LowLevelType> = LazyLock::new(|| CArrayPtr((*VOIDP).clone()));

/// `char *` (`rffi.py:754`).
pub static CCHARP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LowLevelType::Char, nolength_hints()));

/// `const char *` (`rffi.py:757-758`).
pub static CONST_CCHARP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LowLevelType::Char, const_hints()));

/// `wchar_t *` (`rffi.py:761`).
pub static CWCHARP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LowLevelType::UniChar, nolength_hints()));

pub static DOUBLEP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(DOUBLE, nolength_hints()));
pub static FLOATP: LazyLock<LowLevelType> = LazyLock::new(|| ptr_to_array(FLOAT, nolength_hints()));
pub static LONGDOUBLEP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(LONGDOUBLE, nolength_hints()));
pub static SIGNEDP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(SIGNED, nolength_hints()));
pub static SIGNEDPP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array((*SIGNEDP).clone(), nolength_hints()));
pub static UNSIGNEDP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array(UNSIGNED, nolength_hints()));
pub static CCHARPP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array((*CCHARP).clone(), nolength_hints()));
pub static CWCHARPP: LazyLock<LowLevelType> =
    LazyLock::new(|| ptr_to_array((*CWCHARP).clone(), nolength_hints()));

/// RPython `CStruct(name, *fields, **kwds)` (`rffi.py:614-626`).
///
/// Upstream prefixes every field with `c_` and adds the C rendering hints.
pub fn CStruct(name: &str, fields: Vec<(String, LowLevelType)>) -> StructType {
    CStruct_with_hints(name, fields, vec![])
}

pub fn CStruct_with_hints(
    name: &str,
    fields: Vec<(String, LowLevelType)>,
    mut hints: Vec<(String, ConstValue)>,
) -> StructType {
    hints.push(("external".into(), ConstValue::byte_str("C")));
    hints.push(("c_name".into(), ConstValue::byte_str(name)));
    let c_fields = fields
        .into_iter()
        .map(|(field, typ)| (format!("c_{field}"), typ))
        .collect();
    StructType::with_hints(name, c_fields, hints)
}

/// RPython `CStructPtr(*args, **kwds)` (`rffi.py:628-629`).
pub fn CStructPtr(name: &str, fields: Vec<(String, LowLevelType)>) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(CStruct(name, fields)),
    }))
}

/// RPython `CFixedArray(tp, size)` (`rffi.py:631-633`).
pub fn CFixedArray(tp: LowLevelType, size: usize) -> FixedSizeArrayType {
    FixedSizeArrayType::new(tp, size)
}

/// RPython `CArray(tp)` (`rffi.py:635-637`).
pub fn CArray(tp: LowLevelType) -> ArrayType {
    ArrayType::with_hints(tp, nolength_hints())
}

/// RPython `CArrayPtr(tp)` (`rffi.py:639-641`).
pub fn CArrayPtr(tp: LowLevelType) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(CArray(tp)),
    }))
}

/// RPython `CCallback(args, res)` (`rffi.py:643-645`).
pub fn CCallback(args: Vec<LowLevelType>, res: LowLevelType) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Func(FuncType { args, result: res }),
    }))
}

/// RPython `COpaque(...)` (`rffi.py:647-672`), narrowed to the type identity
/// available in the current `lltype::OpaqueType` port.
pub fn COpaque(name: Option<&str>) -> OpaqueType {
    OpaqueType::new(name.unwrap_or("C"))
}

/// RPython `COpaquePtr(*args, **kwds)` (`rffi.py:674-676`).
pub fn COpaquePtr(name: Option<&str>) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Opaque(COpaque(name)),
    }))
}

/// Minimal `llexternal` surface backed by the same function-pointer metadata
/// used by `extfunc.py`. Full wrapper generation remains with the broader
/// `rffi.py` runtime port.
pub fn llexternal(name: &str, args: Vec<LowLevelType>, result: LowLevelType) -> _ptr {
    functionptr_with_external_name(FuncType { args, result }, name, None)
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(non_snake_case)]
pub struct CConstant {
    pub c_name: String,
    pub TP: LowLevelType,
}

impl CConstant {
    pub fn new(c_name: impl Into<String>, TP: LowLevelType) -> Self {
        CConstant {
            c_name: c_name.into(),
            TP,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr_array_of(value: &LowLevelType) -> &ArrayType {
        let LowLevelType::Ptr(ptr) = value else {
            panic!("expected Ptr(Array), got {value:?}");
        };
        let PtrTarget::Array(array) = &ptr.TO else {
            panic!("expected Ptr(Array), got {value:?}");
        };
        array
    }

    #[test]
    fn errno_flags_match_upstream_bit_layout() {
        assert_eq!(RFFI_FULL_ERRNO, 3);
        assert_eq!(RFFI_FULL_ERRNO_ZERO, 5);
        assert_eq!(RFFI_FULL_LASTERROR, 24);
        assert_eq!(RFFI_ERR_ALL, 27);
        assert_eq!(RFFI_ALT_ERRNO, 64);
    }

    #[test]
    fn carray_builds_nolength_raw_array() {
        let array = CArray(LowLevelType::Char);
        assert_eq!(array.OF, LowLevelType::Char);
        assert_eq!(array._hints.get("nolength"), Some(&ConstValue::Bool(true)));
    }

    #[test]
    fn pointer_aliases_have_rffi_array_hints() {
        let voidp = ptr_array_of(&VOIDP);
        assert_eq!(voidp.OF, LowLevelType::Char);
        assert_eq!(voidp._hints.get("nolength"), Some(&ConstValue::Bool(true)));
        assert_eq!(
            voidp._hints.get("render_as_void"),
            Some(&ConstValue::Bool(true))
        );

        let const_charp = ptr_array_of(&CONST_CCHARP);
        assert_eq!(const_charp.OF, LowLevelType::Char);
        assert_eq!(
            const_charp._hints.get("render_as_const"),
            Some(&ConstValue::Bool(true))
        );

        let charpp = ptr_array_of(&CCHARPP);
        assert_eq!(charpp.OF, (*CCHARP).clone());
    }

    #[test]
    fn cstruct_prefixes_fields_and_records_c_hints() {
        let c_struct = CStruct(
            "demo",
            vec![("x".into(), LowLevelType::Signed), ("y".into(), DOUBLE)],
        );
        assert_eq!(c_struct._names, vec!["c_x", "c_y"]);
        assert_eq!(
            c_struct._hints.get("external"),
            Some(&ConstValue::byte_str("C"))
        );
        assert_eq!(
            c_struct._hints.get("c_name"),
            Some(&ConstValue::byte_str("demo"))
        );
    }

    #[test]
    fn ccallback_returns_ptr_to_functype() {
        let callback = CCallback(vec![SIGNED], LowLevelType::Void);
        let LowLevelType::Ptr(ptr) = callback else {
            panic!("expected Ptr(Func)");
        };
        let PtrTarget::Func(func_t) = ptr.TO else {
            panic!("expected Ptr(Func)");
        };
        assert_eq!(func_t.args, vec![LowLevelType::Signed]);
        assert_eq!(func_t.result, LowLevelType::Void);
    }
}
