//! RPython `rpython/rtyper/lltypesystem/rffi.py`.
//!
//! This slice exposes the exact lltype-facing surface that can be represented
//! with pyre's current `lltype` port: errno flags, C array/struct helper
//! constructors, callback pointer types, and the standard raw pointer aliases.
//! Width-specific C integer aliases and full wrapper-generation behavior remain
//! deferred until the corresponding RPython platform/type metadata is ported.
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, Array, FixedSizeArray, FuncType, LowLevelType, LowLevelValue, MallocFlavor, OpaqueType,
    Ptr, PtrTarget, Struct, functionptr_with_external_name, malloc,
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

/// RPython `_isfunctype(TP)` (`rffi.py:43-48`).
pub fn _isfunctype(TP: &LowLevelType) -> bool {
    matches!(TP, LowLevelType::Ptr(ptr) if matches!(ptr.TO, PtrTarget::Func(_)))
}

/// RPython `_isllptr(p)` (`rffi.py:50-52`). The Rust signature accepts only
/// `lltype::_ptr`, so every value reaching this helper is a low-level pointer.
pub fn _isllptr(_p: &_ptr) -> bool {
    true
}

/// RPython `_IsLLPtrEntry(ExtRegistryEntry)` (`rffi.py:53-60`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct _IsLLPtrEntry;

/// RPython primitive aliases (`rffi.py:739-779`).
pub const CHAR: LowLevelType = LowLevelType::Char;
pub const DOUBLE: LowLevelType = LowLevelType::Float;
pub const LONGDOUBLE: LowLevelType = LowLevelType::LongFloat;
pub const FLOAT: LowLevelType = LowLevelType::SingleFloat;
pub const SIGNED: LowLevelType = LowLevelType::Signed;
pub const UNSIGNED: LowLevelType = LowLevelType::Unsigned;
pub const r_singlefloat: LowLevelType = LowLevelType::SingleFloat;
pub const NULL: Option<()> = None;

/// RPython `TYPES` seed list (`rffi.py:503-538`), narrowed only where the
/// platform-dependent `CompilationError` branch cannot be evaluated in this
/// static port.
pub static TYPES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    let mut types = vec![
        "short",
        "unsigned short",
        "int",
        "unsigned int",
        "long",
        "unsigned long",
        "signed char",
        "unsigned char",
        "long long",
        "unsigned long long",
        "size_t",
        "time_t",
        "wchar_t",
        "uintptr_t",
        "intptr_t",
        "void*",
    ];
    #[cfg(not(windows))]
    {
        types.extend([
            "mode_t",
            "pid_t",
            "ssize_t",
            "ptrdiff_t",
            "int_least8_t",
            "uint_least8_t",
            "int_least16_t",
            "uint_least16_t",
            "int_least32_t",
            "uint_least32_t",
            "int_least64_t",
            "uint_least64_t",
            "int_fast8_t",
            "uint_fast8_t",
            "int_fast16_t",
            "uint_fast16_t",
            "int_fast32_t",
            "uint_fast32_t",
            "int_fast64_t",
            "uint_fast64_t",
            "intmax_t",
            "uintmax_t",
        ]);
    }
    types
});

fn rffi_name_from_c_type(mut name: &str) -> String {
    if let Some(rest) = name.strip_prefix("unsigned") {
        name = rest.trim_start();
        format!("u{}", name.replace(' ', ""))
    } else {
        name.replace(' ', "")
    }
}

/// RPython `populate_inttypes()` (`rffi.py:541-559`).
pub fn populate_inttypes() -> Vec<String> {
    TYPES
        .iter()
        .map(|name| rffi_name_from_c_type(name))
        .collect()
}

/// RPython `setup()` (`rffi.py:562-575`), represented by the primitive
/// low-level types available before the full platform cache is ported.
pub fn setup() -> Vec<LowLevelType> {
    populate_inttypes()
        .into_iter()
        .map(|name| {
            if name == "void*" {
                (*VOIDP).clone()
            } else if name.starts_with('u') || name == "size_t" || name == "uintptr_t" {
                LowLevelType::Unsigned
            } else if name == "wchar_t" {
                LowLevelType::UniChar
            } else {
                LowLevelType::Signed
            }
        })
        .collect()
}

pub static NUMBER_TYPES: LazyLock<Vec<LowLevelType>> = LazyLock::new(setup);

/// RPython wrap-around integer class names (`rffi.py:578-584`). The current
/// lltype port carries the low-level primitive identity, not rarithmetic's
/// generated Python class object.
pub const r_int_real: LowLevelType = LowLevelType::Signed;
pub const r_uint_real: LowLevelType = LowLevelType::Unsigned;

fn ptr_to_array(of: LowLevelType, hints: Vec<(String, ConstValue)>) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(Array::with_hints(of, hints)),
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
pub fn CStruct(name: &str, fields: Vec<(String, LowLevelType)>) -> Struct {
    CStruct_with_hints(name, fields, vec![])
}

pub fn CStruct_with_hints(
    name: &str,
    fields: Vec<(String, LowLevelType)>,
    mut hints: Vec<(String, ConstValue)>,
) -> Struct {
    hints.push(("external".into(), ConstValue::byte_str("C")));
    hints.push(("c_name".into(), ConstValue::byte_str(name)));
    let c_fields = fields
        .into_iter()
        .map(|(field, typ)| (format!("c_{field}"), typ))
        .collect();
    Struct::with_hints(name, c_fields, hints)
}

/// RPython `CStructPtr(*args, **kwds)` (`rffi.py:628-629`).
pub fn CStructPtr(name: &str, fields: Vec<(String, LowLevelType)>) -> LowLevelType {
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(CStruct(name, fields)),
    }))
}

/// RPython `CFixedArray(tp, size)` (`rffi.py:631-633`).
pub fn CFixedArray(tp: LowLevelType, size: usize) -> FixedSizeArray {
    FixedSizeArray::new(tp, size)
}

/// RPython `CArray(tp)` (`rffi.py:635-637`).
pub fn CArray(tp: LowLevelType) -> Array {
    Array::with_hints(tp, nolength_hints())
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

/// RPython `make(STRUCT, **fields)` (`rffi.py:1299-1305`).
pub fn make(STRUCT: Struct, fields: Vec<(String, LowLevelValue)>) -> Result<_ptr, String> {
    let mut ptr = malloc(
        LowLevelType::Struct(Box::new(STRUCT)),
        None,
        MallocFlavor::Raw,
        false,
    )?;
    for (name, value) in fields {
        ptr.setattr(&name, value)?;
    }
    Ok(ptr)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CallbackHolder {
    pub callbacks: HashMap<String, bool>,
}

impl CallbackHolder {
    pub fn new() -> Self {
        CallbackHolder {
            callbacks: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeepaliveKeeper {
    pub stuff_to_keepalive: Vec<Option<ConstValue>>,
    pub free_positions: Vec<usize>,
}

/// RPython `_KEEPER_CACHE = {}` (`rffi.py:456`). This is intentionally a
/// side cache because upstream uses a module-level dict keyed by low-level
/// type for callback keepalive slots.
pub static _KEEPER_CACHE: LazyLock<Mutex<HashMap<LowLevelType, KeepaliveKeeper>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn _keeper_for_type(TP: LowLevelType) -> KeepaliveKeeper {
    let mut cache = _KEEPER_CACHE.lock().expect("_KEEPER_CACHE poisoned");
    cache.entry(TP).or_default().clone()
}

pub fn register_keepalive(TP: LowLevelType, obj: ConstValue) -> usize {
    let mut cache = _KEEPER_CACHE.lock().expect("_KEEPER_CACHE poisoned");
    let keeper = cache.entry(TP).or_default();
    if let Some(pos) = keeper.free_positions.pop() {
        keeper.stuff_to_keepalive[pos] = Some(obj);
        pos
    } else {
        let pos = keeper.stuff_to_keepalive.len();
        keeper.stuff_to_keepalive.push(Some(obj));
        pos
    }
}

pub fn get_keepalive_object(pos: usize, TP: LowLevelType) -> Option<ConstValue> {
    let cache = _KEEPER_CACHE.lock().expect("_KEEPER_CACHE poisoned");
    cache
        .get(&TP)
        .and_then(|keeper| keeper.stuff_to_keepalive.get(pos).cloned())
        .flatten()
}

pub fn unregister_keepalive(pos: usize, TP: LowLevelType) {
    let mut cache = _KEEPER_CACHE.lock().expect("_KEEPER_CACHE poisoned");
    if let Some(keeper) = cache.get_mut(&TP)
        && pos < keeper.stuff_to_keepalive.len()
    {
        keeper.stuff_to_keepalive[pos] = None;
        keeper.free_positions.push(pos);
    }
}

fn deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "lltypesystem.rffi.{name} — C wrapper/runtime buffer behavior deferred"
    ))
}

pub fn _make_wrapper_for() -> Result<(), TyperError> {
    Err(deferred("_make_wrapper_for"))
}

pub fn llexternal_use_eci() -> Result<(), TyperError> {
    Err(deferred("llexternal_use_eci"))
}

pub fn generate_macro_wrapper() -> Result<(), TyperError> {
    Err(deferred("generate_macro_wrapper"))
}

pub fn CExternVariable() -> Result<(), TyperError> {
    Err(deferred("CExternVariable"))
}

pub fn make_string_mappings() -> Result<(), TyperError> {
    Err(deferred("make_string_mappings"))
}

pub fn constcharp2str() -> Result<(), TyperError> {
    Err(deferred("constcharp2str"))
}

pub fn constcharpsize2str() -> Result<(), TyperError> {
    Err(deferred("constcharpsize2str"))
}

pub fn str2constcharp() -> Result<(), TyperError> {
    Err(deferred("str2constcharp"))
}

pub fn _deprecated_get_nonmovingbuffer() -> Result<(), TyperError> {
    Err(deferred("_deprecated_get_nonmovingbuffer"))
}

pub fn get_nonmovingbuffer() -> Result<(), TyperError> {
    Err(deferred("get_nonmovingbuffer"))
}

pub fn get_nonmovingbuffer_final_null() -> Result<(), TyperError> {
    Err(deferred("get_nonmovingbuffer_final_null"))
}

pub fn free_nonmovingbuffer() -> Result<(), TyperError> {
    Err(deferred("free_nonmovingbuffer"))
}

pub fn wcharpsize2utf8() -> Result<(), TyperError> {
    Err(deferred("wcharpsize2utf8"))
}

pub fn wcharp2utf8() -> Result<(), TyperError> {
    Err(deferred("wcharp2utf8"))
}

pub fn wcharp2utf8n() -> Result<(), TyperError> {
    Err(deferred("wcharp2utf8n"))
}

pub fn utf82wcharp() -> Result<(), TyperError> {
    Err(deferred("utf82wcharp"))
}

pub fn utf82wcharp_ex() -> Result<(), TyperError> {
    Err(deferred("utf82wcharp_ex"))
}

pub fn liststr2charpp() -> Result<(), TyperError> {
    Err(deferred("liststr2charpp"))
}

pub fn ll_liststr2charpp() -> Result<(), TyperError> {
    Err(deferred("ll_liststr2charpp"))
}

pub fn free_charpp() -> Result<(), TyperError> {
    Err(deferred("free_charpp"))
}

pub fn charpp2liststr() -> Result<(), TyperError> {
    Err(deferred("charpp2liststr"))
}

pub fn cast() -> Result<(), TyperError> {
    Err(deferred("cast"))
}

pub fn ptradd() -> Result<(), TyperError> {
    Err(deferred("ptradd"))
}

pub fn size_and_sign() -> Result<(), TyperError> {
    Err(deferred("size_and_sign"))
}

pub fn sizeof() -> Result<(), TyperError> {
    Err(deferred("sizeof"))
}

pub fn offsetof() -> Result<(), TyperError> {
    Err(deferred("offsetof"))
}

/// RPython `MakeEntry(ExtRegistryEntry)` (`rffi.py:1320`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MakeEntry;

pub fn structcopy() -> Result<(), TyperError> {
    Err(deferred("structcopy"))
}

pub fn _get_structcopy_fn() -> Result<(), TyperError> {
    Err(deferred("_get_structcopy_fn"))
}

pub fn setintfield() -> Result<(), TyperError> {
    Err(deferred("setintfield"))
}

pub fn getintfield() -> Result<(), TyperError> {
    Err(deferred("getintfield"))
}

/// RPython scoped raw string buffer context managers (`rffi.py:1386-1492`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_str2charp;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_unicode2wcharp;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_utf82wcharp;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_nonmovingbuffer;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_view_charp;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_nonmoving_unicodebuffer;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_alloc_buffer;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_alloc_unicodebuffer;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct scoped_alloc_utf8buffer;

pub fn c_memcpy() -> Result<(), TyperError> {
    Err(deferred("c_memcpy"))
}

pub fn c_memset() -> Result<(), TyperError> {
    Err(deferred("c_memset"))
}

/// RPython `TEST_RAW_ADDR_KEEP_ALIVE = {}` (`rffi.py:1512`).
pub static TEST_RAW_ADDR_KEEP_ALIVE: LazyLock<Mutex<HashMap<String, LowLevelType>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn get_raw_address_of_string() -> Result<(), TyperError> {
    Err(deferred("get_raw_address_of_string"))
}

/// RPython `_StrFinalizerQueue(rgc.FinalizerQueue)` (`rffi.py:1538`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct _StrFinalizerQueue {
    pub raw_copies: HashMap<usize, LowLevelType>,
}

/// RPython `_fq_addr_from_string = _StrFinalizerQueue()` (`rffi.py:1557`).
pub static _fq_addr_from_string: LazyLock<Mutex<_StrFinalizerQueue>> =
    LazyLock::new(|| Mutex::new(_StrFinalizerQueue::default()));

pub fn _get_raw_address_buf_from_string() -> Result<(), TyperError> {
    Err(deferred("_get_raw_address_buf_from_string"))
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

    fn ptr_array_of(value: &LowLevelType) -> &Array {
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
    fn functype_and_llptr_predicates_match_upstream_shape() {
        let callback = CCallback(vec![LowLevelType::Signed], LowLevelType::Void);
        assert!(_isfunctype(&callback));
        assert!(!_isfunctype(&LowLevelType::Signed));

        let ptr = llexternal("demo", vec![], LowLevelType::Void);
        assert!(_isllptr(&ptr));
    }

    #[test]
    fn inttype_population_preserves_rffi_names() {
        let names = populate_inttypes();
        assert!(names.contains(&"short".to_string()));
        assert!(names.contains(&"ushort".to_string()));
        assert!(names.contains(&"int".to_string()));
        assert!(names.contains(&"uint".to_string()));
        assert!(names.contains(&"void*".to_string()));
        assert!(NUMBER_TYPES.len() >= names.len());
        assert_eq!(r_int_real, LowLevelType::Signed);
        assert_eq!(r_uint_real, LowLevelType::Unsigned);
        assert_eq!(r_singlefloat, LowLevelType::SingleFloat);
        assert_eq!(NULL, None);
    }

    #[test]
    fn keepalive_cache_reuses_freed_slots_per_type() {
        let tp = LowLevelType::Signed;
        let pos = register_keepalive(tp.clone(), ConstValue::Int(41));
        assert_eq!(
            get_keepalive_object(pos, tp.clone()),
            Some(ConstValue::Int(41))
        );

        unregister_keepalive(pos, tp.clone());
        assert_eq!(get_keepalive_object(pos, tp.clone()), None);

        let reused = register_keepalive(tp.clone(), ConstValue::Int(42));
        assert_eq!(reused, pos);
        assert_eq!(get_keepalive_object(pos, tp), Some(ConstValue::Int(42)));
    }

    #[test]
    fn deferred_runtime_helpers_name_the_original_surface() {
        let err = make_string_mappings().expect_err("string runtime is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("make_string_mappings"));

        let err = generate_macro_wrapper().expect_err("macro wrapper is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("generate_macro_wrapper"));

        let err = ll_liststr2charpp().expect_err("char** runtime is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("ll_liststr2charpp"));

        let err = c_memcpy().expect_err("native memcpy binding is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("c_memcpy"));

        let err = get_raw_address_of_string().expect_err("raw string address helper is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("get_raw_address_of_string"));

        let _scoped = scoped_alloc_buffer;
        let _finalizer = _StrFinalizerQueue::default();
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
    fn make_mallocs_raw_struct_and_populates_fields() {
        let s = Struct::new("point", vec![("x".into(), LowLevelType::Signed)]);
        let ptr = make(s, vec![("x".into(), LowLevelValue::Signed(7))])
            .expect("rffi.make should allocate a raw struct pointer");

        assert_eq!(ptr.getattr("x").unwrap(), LowLevelValue::Signed(7));
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
