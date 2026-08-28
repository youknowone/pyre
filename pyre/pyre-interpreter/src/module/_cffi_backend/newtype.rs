//! Building ctypes — PyPy:
//! `pypy/module/_cffi_backend/newtype.py`.
//!
//! `newtype.py` memoises through `UniqueCache` plus a weak `_pointer_type` /
//! `_array_types` on each ctype.  Here every ctype is rooted for the process
//! (see [`super::ctypeobj::new_ctype`]), so the memo can be a plain map: a
//! program declares a bounded set of C types once, and pointer compatibility
//! is decided by object identity, which only a memo can keep true.

use crate::PyError;
use pyre_object::PyObjectRef;
use std::collections::HashMap;
use std::ffi::{c_char, c_double, c_float, c_int, c_long, c_longlong, c_short};
use std::sync::{Mutex, OnceLock};

use super::ctypeobj::{self, W_CType};
use super::misc;

/// `newtype.py alignment(TYPE)` — `offsetof(struct {char x; TYPE y;}, y)`,
/// which is the type's alignment.
macro_rules! prim {
    ($kind:expr, $ty:ty) => {
        (
            $kind,
            std::mem::size_of::<$ty>() as i64,
            std::mem::align_of::<$ty>() as i64,
        )
    };
    ($kind:expr, $ty:ty, $rep:literal) => {
        (
            $kind,
            (std::mem::size_of::<$ty>() * $rep) as i64,
            std::mem::align_of::<$ty>() as i64,
        )
    };
}

/// `newtype.py PRIMITIVE_TYPES` — the kind, size and alignment of every name
/// `new_primitive_type` accepts.
fn primitive_types() -> &'static HashMap<&'static str, (i64, i64, i64)> {
    static TABLE: OnceLock<HashMap<&'static str, (i64, i64, i64)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        use ctypeobj::{
            KIND_PRIM_BOOL, KIND_PRIM_CHAR, KIND_PRIM_COMPLEX, KIND_PRIM_FLOAT,
            KIND_PRIM_LONGDOUBLE, KIND_PRIM_SIGNED, KIND_PRIM_UNICHAR, KIND_PRIM_UNSIGNED,
        };
        let mut t: HashMap<&'static str, (i64, i64, i64)> = HashMap::new();
        t.insert("char", prim!(KIND_PRIM_CHAR, c_char));
        t.insert("wchar_t", prim!(KIND_PRIM_UNICHAR, u32));
        t.insert("signed char", prim!(KIND_PRIM_SIGNED, i8));
        t.insert("short", prim!(KIND_PRIM_SIGNED, c_short));
        t.insert("int", prim!(KIND_PRIM_SIGNED, c_int));
        t.insert("long", prim!(KIND_PRIM_SIGNED, c_long));
        t.insert("long long", prim!(KIND_PRIM_SIGNED, c_longlong));
        t.insert("unsigned char", prim!(KIND_PRIM_UNSIGNED, u8));
        t.insert("unsigned short", prim!(KIND_PRIM_UNSIGNED, c_short));
        t.insert("unsigned int", prim!(KIND_PRIM_UNSIGNED, c_int));
        t.insert("unsigned long", prim!(KIND_PRIM_UNSIGNED, c_long));
        t.insert("unsigned long long", prim!(KIND_PRIM_UNSIGNED, c_longlong));
        t.insert("float", prim!(KIND_PRIM_FLOAT, c_float));
        t.insert("double", prim!(KIND_PRIM_FLOAT, c_double));
        t.insert(
            "long double",
            (
                KIND_PRIM_LONGDOUBLE,
                misc::sizeof_long_double(),
                misc::alignof_long_double(),
            ),
        );
        t.insert("_Bool", prim!(KIND_PRIM_BOOL, bool));
        for name in ["float _Complex", "_cffi_float_complex_t"] {
            t.insert(name, prim!(KIND_PRIM_COMPLEX, c_float, 2));
        }
        for name in ["double _Complex", "_cffi_double_complex_t"] {
            t.insert(name, prim!(KIND_PRIM_COMPLEX, c_double, 2));
        }
        // `eptypesize` — the fixed-width names, whose sizes are their names.
        for (name, kind, size) in [
            ("int8_t", KIND_PRIM_SIGNED, 1),
            ("uint8_t", KIND_PRIM_UNSIGNED, 1),
            ("int16_t", KIND_PRIM_SIGNED, 2),
            ("uint16_t", KIND_PRIM_UNSIGNED, 2),
            ("int32_t", KIND_PRIM_SIGNED, 4),
            ("uint32_t", KIND_PRIM_UNSIGNED, 4),
            ("int64_t", KIND_PRIM_SIGNED, 8),
            ("uint64_t", KIND_PRIM_UNSIGNED, 8),
            ("int_least8_t", KIND_PRIM_SIGNED, 1),
            ("uint_least8_t", KIND_PRIM_UNSIGNED, 1),
            ("int_least16_t", KIND_PRIM_SIGNED, 2),
            ("uint_least16_t", KIND_PRIM_UNSIGNED, 2),
            ("int_least32_t", KIND_PRIM_SIGNED, 4),
            ("uint_least32_t", KIND_PRIM_UNSIGNED, 4),
            ("int_least64_t", KIND_PRIM_SIGNED, 8),
            ("uint_least64_t", KIND_PRIM_UNSIGNED, 8),
            ("char16_t", KIND_PRIM_UNICHAR, 2),
            ("char32_t", KIND_PRIM_UNICHAR, 4),
        ] {
            t.insert(name, (kind, size, size));
        }
        // The `fast` names are the platform's own widths, which C picks for
        // speed rather than size.
        for (name, kind, ty_size) in [
            ("int_fast8_t", KIND_PRIM_SIGNED, std::mem::size_of::<i8>()),
            (
                "uint_fast8_t",
                KIND_PRIM_UNSIGNED,
                std::mem::size_of::<u8>(),
            ),
            (
                "int_fast16_t",
                KIND_PRIM_SIGNED,
                std::mem::size_of::<c_long>(),
            ),
            (
                "uint_fast16_t",
                KIND_PRIM_UNSIGNED,
                std::mem::size_of::<c_long>(),
            ),
            (
                "int_fast32_t",
                KIND_PRIM_SIGNED,
                std::mem::size_of::<c_long>(),
            ),
            (
                "uint_fast32_t",
                KIND_PRIM_UNSIGNED,
                std::mem::size_of::<c_long>(),
            ),
            (
                "int_fast64_t",
                KIND_PRIM_SIGNED,
                std::mem::size_of::<c_long>(),
            ),
            (
                "uint_fast64_t",
                KIND_PRIM_UNSIGNED,
                std::mem::size_of::<c_long>(),
            ),
        ] {
            t.insert(name, (kind, ty_size as i64, ty_size as i64));
        }
        t.insert("intptr_t", prim!(KIND_PRIM_SIGNED, isize));
        t.insert("uintptr_t", prim!(KIND_PRIM_UNSIGNED, usize));
        t.insert("size_t", prim!(KIND_PRIM_UNSIGNED, usize));
        t.insert("ssize_t", prim!(KIND_PRIM_SIGNED, isize));
        t.insert("ptrdiff_t", prim!(KIND_PRIM_SIGNED, isize));
        t.insert("intmax_t", prim!(KIND_PRIM_SIGNED, c_longlong));
        t.insert("uintmax_t", prim!(KIND_PRIM_UNSIGNED, c_longlong));
        t
    })
}

/// `UniqueCache.primitives`.
fn primitive_cache() -> &'static Mutex<HashMap<&'static str, usize>> {
    static CACHE: OnceLock<Mutex<HashMap<&'static str, usize>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `UniqueCache.ctvoid`.
static VOID_TYPE: OnceLock<usize> = OnceLock::new();

/// `W_CTypePointer._array_types`, keyed by the pointer ctype's address and
/// the array length.
fn array_cache() -> &'static Mutex<HashMap<(usize, i64), usize>> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, i64), usize>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `newtype.py _new_primitive_type`.
pub fn new_primitive_type(name: &str) -> Result<PyObjectRef, PyError> {
    let Some((&key, &(kind, size, align))) = primitive_types().get_key_value(name) else {
        return Err(PyError::key_error(format!("{name:?}")));
    };
    {
        let cache = primitive_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(&existing) = cache.get(key) {
            return Ok(existing as PyObjectRef);
        }
    }
    let word = std::mem::size_of::<isize>() as i64;
    let mut flags = 0;
    match kind {
        ctypeobj::KIND_PRIM_CHAR
        | ctypeobj::KIND_PRIM_UNICHAR
        | ctypeobj::KIND_PRIM_SIGNED
        | ctypeobj::KIND_PRIM_UNSIGNED
        | ctypeobj::KIND_PRIM_BOOL => {
            flags |= ctypeobj::F_PRIMITIVE_INTEGER;
        }
        _ => {}
    }
    match kind {
        ctypeobj::KIND_PRIM_SIGNED => {
            if size <= word {
                flags |= ctypeobj::F_VALUE_FITS_LONG;
            }
            if size < word {
                flags |= ctypeobj::F_VALUE_SMALLER_THAN_LONG;
            }
        }
        ctypeobj::KIND_PRIM_UNSIGNED | ctypeobj::KIND_PRIM_BOOL => {
            if size < word {
                flags |= ctypeobj::F_VALUE_FITS_LONG;
            }
            if size <= word {
                flags |= ctypeobj::F_VALUE_FITS_ULONG;
            }
        }
        ctypeobj::KIND_PRIM_UNICHAR => {
            // `char16_t` and `char32_t` are always unsigned; only `wchar_t`
            // follows the platform's signedness.
            if key == "wchar_t" && wchar_is_signed() {
                flags |= ctypeobj::F_SIGNED_WCHAR;
            }
        }
        _ => {}
    }
    let obj = ctypeobj::new_ctype(
        kind,
        size,
        key,
        key.len() as i64,
        align,
        pyre_object::PY_NULL,
        -1,
        flags,
    );
    let mut cache = primitive_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Ok(*cache.entry(key).or_insert(obj as usize) as PyObjectRef)
}

/// `rfficache.signof_c_type('wchar_t')`.
fn wchar_is_signed() -> bool {
    // The C `wchar_t` is signed on the platforms whose `wchar_t` is `int`
    // and unsigned where it is `unsigned short` or `unsigned int`.
    cfg!(not(windows))
}

/// `newtype.py new_void_type`.
pub fn new_void_type() -> PyObjectRef {
    *VOID_TYPE.get_or_init(|| {
        ctypeobj::new_ctype(
            ctypeobj::KIND_VOID,
            -1,
            "void",
            "void".len() as i64,
            -1,
            pyre_object::PY_NULL,
            -1,
            0,
        ) as usize
    }) as PyObjectRef
}

/// `newtype.py _new_voidp_type`.
pub fn new_voidp_type() -> Result<PyObjectRef, PyError> {
    new_pointer_type(new_void_type())
}

/// `newtype.py _new_chara_type`.
pub fn new_chara_type() -> Result<PyObjectRef, PyError> {
    let w_char = new_primitive_type("char")?;
    let w_charp = new_pointer_type(w_char)?;
    new_array_type(w_charp, -1)
}

/// `newtype.py _new_pointer_type` — `W_CType._pointer_type`'s memo, which
/// `convert_from_object` relies on to decide pointer compatibility.
pub fn new_pointer_type(w_ctitem: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ctitem = ctypeobj::ctype_arg(w_ctitem)?;
    if !ctitem.pointer_type.is_null() {
        return Ok(ctitem.pointer_type);
    }
    // `W_CTypePointer.__init__`: an array's pointer needs the parenthesised
    // spelling so `int(*)[5]` does not read as `int *[5]`.
    let extra = if ctitem.kind == ctypeobj::KIND_ARRAY {
        "(*)"
    } else {
        " *"
    };
    let (name, name_position) = ctitem.insert_name(extra, 2);
    let mut flags = ctypeobj::F_NONFUNC_POINTER_OR_ARRAY;
    if ctitem.kind == ctypeobj::KIND_VOID {
        flags |= ctypeobj::F_VOID_PTR | ctypeobj::F_VOIDCHAR_PTR;
    }
    if ctitem.kind == ctypeobj::KIND_PRIM_CHAR {
        flags |= ctypeobj::F_VOIDCHAR_PTR;
    }
    if ctitem.size == 1 {
        flags |= ctypeobj::F_ONEBYTE_PTR;
    }
    if ctitem.name() == "struct _IO_FILE" || ctitem.name() == "FILE" {
        flags |= ctypeobj::F_FILE_PTR;
    }
    flags |= accept_str_flag(ctitem);
    let obj = ctypeobj::new_ctype(
        ctypeobj::KIND_POINTER,
        std::mem::size_of::<*const u8>() as i64,
        &name,
        name_position,
        -1,
        w_ctitem,
        -1,
        flags,
    );
    // Re-read the item: building the pointer allocated, and the memo must
    // land on the object the caller named.
    let ctitem = ctypeobj::ctype_arg(w_ctitem)?;
    ctitem.pointer_type = obj;
    Ok(obj)
}

/// `W_CTypePtrOrArray.accept_str` — whether a `bytes` initialises it.
fn accept_str_flag(ctitem: &W_CType) -> i64 {
    let accepts = ctitem.kind == ctypeobj::KIND_VOID
        || ctitem.kind == ctypeobj::KIND_PRIM_CHAR
        || (ctitem.has(ctypeobj::F_PRIMITIVE_INTEGER) && ctitem.size == 1);
    if accepts { ctypeobj::F_ACCEPT_STR } else { 0 }
}

/// `newtype.py _new_array_type`.
pub fn new_array_type(w_ctptr: PyObjectRef, length: i64) -> Result<PyObjectRef, PyError> {
    let ctptr = ctypeobj::ctype_arg(w_ctptr)?;
    if ctptr.kind != ctypeobj::KIND_POINTER {
        return Err(PyError::type_error("first arg must be a pointer ctype"));
    }
    let key = (w_ctptr as usize, length);
    {
        let cache = array_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(&existing) = cache.get(&key) {
            return Ok(existing as PyObjectRef);
        }
    }
    let ctitem = super::ctypeptr::item_of(ctptr)?;
    if ctitem.size < 0 {
        return Err(PyError::value_error(format!(
            "array item of unknown size: '{}'",
            ctitem.name()
        )));
    }
    let (arraysize, extra) = if length < 0 {
        (-1, "[]".to_string())
    } else {
        let arraysize = length
            .checked_mul(ctitem.size)
            .filter(|&n| n >= 0)
            .ok_or_else(|| PyError::overflow_error("array size would overflow a ssize_t"))?;
        (arraysize, format!("[{length}]"))
    };
    let (name, name_position) = ctitem.insert_name(&extra, 0);
    let flags = ctypeobj::F_NONFUNC_POINTER_OR_ARRAY | accept_str_flag(ctitem);
    let obj = ctypeobj::new_ctype(
        ctypeobj::KIND_ARRAY,
        arraysize,
        &name,
        name_position,
        -1,
        ctptr.ctitem,
        length,
        flags,
    );
    // `W_CTypeArray.ctptr`, re-read for the same reason as above.
    let array = ctypeobj::ctype_arg(obj)?;
    array.ctptr = w_ctptr;
    let mut cache = array_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Ok(*cache.entry(key).or_insert(obj as usize) as PyObjectRef)
}

/// `W_CTypePointer.cache_array_type` — the `T[]` a slice of a `T *` reads
/// through.
pub fn cached_unbounded_array_type(w_ctptr: PyObjectRef) -> Result<PyObjectRef, PyError> {
    new_array_type(w_ctptr, -1)
}

/// `func.py new_array_type`'s length argument: `None` is "no length".
pub fn array_length_arg(w_length: PyObjectRef) -> Result<i64, PyError> {
    if unsafe { pyre_object::pyobject::is_none(w_length) } {
        return Ok(-1);
    }
    // `space.getindex_w(w_length, space.w_OverflowError)` — a length wider
    // than a `ssize_t` is an OverflowError rather than a silent clamp.
    let length = crate::baseobjspace::index_int_w_preserve_negative(w_length)?;
    if length < 0 {
        return Err(PyError::value_error("negative array length"));
    }
    Ok(length)
}

// ── structs and unions ──────────────────────────────────────────────────

/// `newtype.py SF_MSVC_BITFIELDS`.
pub const SF_MSVC_BITFIELDS: i64 = 0x01;
/// `newtype.py SF_GCC_ARM_BITFIELDS`.
pub const SF_GCC_ARM_BITFIELDS: i64 = 0x02;
/// `newtype.py SF_GCC_X86_BITFIELDS`.
pub const SF_GCC_X86_BITFIELDS: i64 = 0x10;
/// `newtype.py SF_GCC_BIG_ENDIAN`.
pub const SF_GCC_BIG_ENDIAN: i64 = 0x04;
/// `newtype.py SF_GCC_LITTLE_ENDIAN`.
pub const SF_GCC_LITTLE_ENDIAN: i64 = 0x40;
/// `newtype.py SF_PACKED`.
pub const SF_PACKED: i64 = 0x08;
/// `newtype.py SF_STD_FIELD_POS`.
pub const SF_STD_FIELD_POS: i64 = 0x80;

/// `newtype.py SF_DEFAULT_PACKING`.
const SF_DEFAULT_PACKING: i64 = if cfg!(windows) { 8 } else { 0x4000_0000 };

/// `newtype.py DEFAULT_SFLAGS_PLATFORM`.
const DEFAULT_SFLAGS_PLATFORM: i64 = if cfg!(windows) {
    SF_MSVC_BITFIELDS
} else if cfg!(any(target_arch = "arm", target_arch = "aarch64")) {
    SF_GCC_ARM_BITFIELDS
} else {
    SF_GCC_X86_BITFIELDS
};

/// `newtype.py DEFAULT_SFLAGS_ENDIAN`.
const DEFAULT_SFLAGS_ENDIAN: i64 = if cfg!(target_endian = "big") {
    SF_GCC_BIG_ENDIAN
} else {
    SF_GCC_LITTLE_ENDIAN
};

/// `newtype.py complete_sflags`.
fn complete_sflags(mut sflags: i64) -> i64 {
    if sflags & (SF_MSVC_BITFIELDS | SF_GCC_ARM_BITFIELDS | SF_GCC_X86_BITFIELDS) == 0 {
        sflags |= DEFAULT_SFLAGS_PLATFORM;
    }
    if sflags & (SF_GCC_BIG_ENDIAN | SF_GCC_LITTLE_ENDIAN) == 0 {
        sflags |= DEFAULT_SFLAGS_ENDIAN;
    }
    sflags
}

/// `ffi_obj.py get_ffi_error` — the class `FFI.error` names.  `ffi_obj` is
/// not ported yet, so the class lives here until it has an owner to publish
/// it from; it is the object identity that matters, and there is one.
pub fn ffi_error() -> PyObjectRef {
    static FFI_ERROR: OnceLock<usize> = OnceLock::new();
    *FFI_ERROR.get_or_init(|| {
        let w_exception = crate::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before _cffi_backend init");
        crate::builtins::make_exc_type("ffi.error", crate::builtins::exc_exception_new, w_exception)
            as usize
    }) as PyObjectRef
}

/// `newtype.py new_struct_type`.
pub fn new_struct_type(name: &str) -> PyObjectRef {
    new_struct_or_union(ctypeobj::KIND_STRUCT, name)
}

/// `newtype.py new_union_type`.
pub fn new_union_type(name: &str) -> PyObjectRef {
    new_struct_or_union(ctypeobj::KIND_UNION, name)
}

/// `W_CTypeStructOrUnion.__init__` — born opaque, with the name as its own
/// insertion point.
fn new_struct_or_union(kind: i64, name: &str) -> PyObjectRef {
    ctypeobj::new_ctype(
        kind,
        -1,
        name,
        name.len() as i64,
        -1,
        pyre_object::PY_NULL,
        -1,
        0,
    )
}

/// `newtype.py detect_custom_layout`.
pub fn detect_custom_layout(
    ct: &mut W_CType,
    sflags: i64,
    cdef_value: i64,
    compiler_value: i64,
    msg: &str,
) -> Result<(), PyError> {
    if compiler_value == cdef_value {
        return Ok(());
    }
    if sflags & SF_STD_FIELD_POS != 0 {
        let name = ct.name();
        let mut err = PyError::value_error(format!(
            "{name}: {msg} (cdef says {cdef_value}, but C compiler says {compiler_value}). fix it or use \"...;\" as the last field in the cdef for {name} to make it flexible"
        ));
        err.exc_object = crate::builtins::exc_exception_new(&[
            ffi_error(),
            pyre_object::w_str_new(&format!(
                "{name}: {msg} (cdef says {cdef_value}, but C compiler says {compiler_value}). fix it or use \"...;\" as the last field in the cdef for {name} to make it flexible"
            )),
        ])?;
        return Err(err);
    }
    ct.flags |= ctypeobj::F_CUSTOM_FIELD_POS;
    Ok(())
}

/// `newtype.py roundup_bytes`.
fn roundup_bytes(bytes: i64, bit: i64) -> i64 {
    bytes + i64::from(bit > 0)
}

/// One entry of the `fields` argument: `(name, ctype[, bitsize[, offset]])`.
struct FieldDescr {
    fname: String,
    w_ftype: PyObjectRef,
    fbitsize: i64,
    foffset: i64,
}

/// `newtype.py complete_struct_or_union`.
pub fn complete_struct_or_union(
    w_ctype: PyObjectRef,
    w_fields: PyObjectRef,
    totalsize: i64,
    totalalignment: i64,
    sflags: i64,
    pack: i64,
) -> Result<(), PyError> {
    let sflags = complete_sflags(sflags);
    let (sflags, pack) = if sflags & SF_PACKED != 0 {
        (sflags, 1)
    } else if pack <= 0 {
        (sflags, SF_DEFAULT_PACKING)
    } else {
        (sflags | SF_PACKED, pack)
    };
    let ct = ctypeobj::ctype_arg(w_ctype)?;
    if !ct.is_struct_or_union() || ct.size >= 0 {
        return Err(PyError::type_error(
            "first arg must be a non-initialized struct or union ctype",
        ));
    }
    let is_union = ct.kind == ctypeobj::KIND_UNION;

    // The field descriptors are read out first: each one is a tuple whose
    // unpacking allocates, and nothing below may hold a stale reference.
    let descrs = read_field_descrs(ct, w_fields)?;

    let roots = pyre_object::gc_roots::push_roots();
    let list_slot = roots.base();
    let _ = roots.pin_root(pyre_object::w_list_new(Vec::new()));
    let dict_slot = list_slot + 1;
    let _ = roots.pin_root(pyre_object::dictmultiobject::w_dict_new());
    let append = |w_field: PyObjectRef| unsafe {
        pyre_object::listobject::w_list_append(roots.get(list_slot), w_field);
    };
    let record = |name: &str, w_field: PyObjectRef| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str(roots.get(dict_slot), name, w_field);
    };

    let mut alignment = 1i64;
    // The real offset is `byteoffset + bitoffset * 8`, counted in bits.
    let mut byteoffset = 0i64;
    let mut bitoffset = 0i64;
    let mut byteoffsetmax = 0i64;
    let mut prev_bitfield_size = 0i64;
    let mut prev_bitfield_free = 0i64;
    ct.flags &= !ctypeobj::F_CUSTOM_FIELD_POS;
    let mut with_var_array = false;
    let mut with_packed_change = false;

    for (i, descr) in descrs.iter().enumerate() {
        let ftype = ctypeobj::ctype_arg(descr.w_ftype)?;
        let fname = descr.fname.as_str();
        let fbitsize = descr.fbitsize;
        let foffset = descr.foffset;
        if unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(roots.get(dict_slot), fname).is_some()
        } {
            return Err(PyError::key_error(format!(
                "duplicate field name '{fname}'"
            )));
        }
        if ftype.size < 0 {
            if ftype.kind == ctypeobj::KIND_ARRAY
                && fbitsize < 0
                && (i == descrs.len() - 1 || foffset != -1)
            {
                with_var_array = true;
            } else {
                return Err(PyError::type_error(format!(
                    "field '{}.{fname}' has ctype '{}' of unknown size",
                    ct.name(),
                    ftype.name()
                )));
            }
        } else if ftype.is_struct_or_union() {
            ftype.force_lazy_struct()?;
            // A var-sized array anywhere inside a field propagates outward,
            // so a struct holding such a struct is var-sized too.
            if ftype.has(ctypeobj::F_WITH_VAR_ARRAY) {
                with_var_array = true;
            }
        }

        if is_union {
            byteoffset = 0;
            bitoffset = 0;
        }

        // Skip the alignment bump for an anonymous bitfield, or under SF_PACKED.
        let falignorg = ftype.alignof()?;
        let falign = pack.min(falignorg);
        let mut do_align = true;
        if sflags & SF_GCC_ARM_BITFIELDS == 0 && fbitsize >= 0 {
            do_align = if sflags & SF_MSVC_BITFIELDS == 0 {
                // Anonymous bitfields of any size do not cause alignment.
                !fname.is_empty()
            } else {
                // Zero-sized bitfields do not cause alignment.
                fbitsize > 0
            };
        }
        if alignment < falign && do_align {
            alignment = falign;
        }

        let fflags = if is_union && i > 0 {
            super::ctypestruct::BF_IGNORE_IN_CTOR
        } else {
            0
        };

        if fbitsize < 0 {
            // Not a bitfield: the common case.
            let bs_flag = if ftype.kind == ctypeobj::KIND_ARRAY && ftype.length <= 0 {
                super::ctypestruct::BS_EMPTY_ARRAY
            } else {
                super::ctypestruct::BS_REGULAR
            };
            // Pad to the next byte, then to `falign` or `falignorg` bytes.
            byteoffset = roundup_bytes(byteoffset, bitoffset);
            bitoffset = 0;
            let byteoffsetorg = (byteoffset + falignorg - 1) & !(falignorg - 1);
            byteoffset = (byteoffset + falign - 1) & !(falign - 1);
            if byteoffsetorg != byteoffset {
                with_packed_change = true;
            }
            if foffset >= 0 {
                // A forced field position: the offset just computed only
                // decides whether the layout is a custom one.
                detect_custom_layout(
                    ct,
                    sflags,
                    byteoffset,
                    foffset,
                    &format!("wrong offset for field '{fname}'"),
                )?;
                byteoffset = foffset;
            }
            if fname.is_empty() && ftype.is_struct_or_union() {
                // A nested anonymous struct or union: its fields are spliced
                // in at this offset under their own names.
                for w_srcfld in super::ctypestruct::fields_list_of(ftype)? {
                    let srcfld = super::ctypestruct::W_CField::from_obj(w_srcfld)
                        .ok_or_else(|| PyError::system_error("field list holds a non-field"))?;
                    let w_fld = super::ctypestruct::make_shifted(srcfld, byteoffset, fflags);
                    append(w_fld);
                    if let Some(name) = super::ctypestruct::name_of_field(ftype, w_srcfld)? {
                        record(&name, w_fld);
                    }
                }
                // Such a structure can never be passed by value.
                ct.flags |= ctypeobj::F_CUSTOM_FIELD_POS;
            } else {
                let w_fld =
                    super::ctypestruct::new_cfield(descr.w_ftype, byteoffset, bs_flag, -1, fflags);
                append(w_fld);
                record(fname, w_fld);
            }
            if ftype.size >= 0 {
                byteoffset += ftype.size;
            }
            prev_bitfield_size = 0;
        } else {
            // A bitfield.
            if foffset >= 0 {
                return Err(PyError::type_error(format!(
                    "field '{}.{fname}' is a bitfield, but a fixed offset is specified",
                    ct.name()
                )));
            }
            if !(ftype.kind == ctypeobj::KIND_PRIM_SIGNED
                || ftype.kind == ctypeobj::KIND_PRIM_UNSIGNED
                || ftype.is_char_or_unichar())
            {
                return Err(PyError::type_error(format!(
                    "field '{}.{fname}' declared as '{}' cannot be a bit field",
                    ct.name(),
                    ftype.name()
                )));
            }
            if fbitsize > 8 * ftype.size {
                return Err(PyError::type_error(format!(
                    "bit field '{}.{fname}' is declared '{}:{fbitsize}', which exceeds the width of the type",
                    ct.name(),
                    ftype.name()
                )));
            }
            // Where the theoretical field covering a whole `ftype` starts;
            // the real bitfield lives inside it.
            let mut field_offset_bytes = byteoffset & !(falign - 1);
            if fbitsize == 0 {
                if !fname.is_empty() {
                    return Err(PyError::type_error(format!(
                        "field '{}.{fname}' is declared with :0",
                        ct.name()
                    )));
                }
                if sflags & SF_MSVC_BITFIELDS == 0 {
                    // GCC's notion of "ftype :0;" pads to a value aligned for
                    // `ftype`.
                    if roundup_bytes(byteoffset, bitoffset) > field_offset_bytes {
                        field_offset_bytes += falign;
                    }
                    byteoffset = field_offset_bytes;
                    bitoffset = 0;
                }
                // MSVC's notion is mostly ignored: it only separates other
                // bitfields, forcing them into separate words.
                prev_bitfield_size = 0;
            } else {
                let mut bitshift;
                if sflags & SF_MSVC_BITFIELDS == 0 {
                    // GCC's algorithm: the field can start where we are if it
                    // would fit entirely into an aligned `ftype` field.
                    let bits_already_occupied = (byteoffset - field_offset_bytes) * 8 + bitoffset;
                    if bits_already_occupied + fbitsize > 8 * ftype.size {
                        if sflags & SF_PACKED != 0 && bits_already_occupied & 7 != 0 {
                            return Err(PyError::not_implemented(format!(
                                "with 'packed', gcc would compile field '{}.{fname}' to reuse some bits in the previous field",
                                ct.name()
                            )));
                        }
                        field_offset_bytes += falign;
                        byteoffset = field_offset_bytes;
                        bitoffset = 0;
                        bitshift = 0;
                    } else {
                        bitshift = bits_already_occupied;
                    }
                    bitoffset += fbitsize;
                    byteoffset += bitoffset >> 3;
                    bitoffset &= 7;
                } else {
                    // MSVC's algorithm: a bitfield takes the full width of its
                    // declared type, and shares bits only with a previous
                    // bitfield of the same size.
                    if prev_bitfield_size == ftype.size && prev_bitfield_free >= fbitsize {
                        bitshift = 8 * prev_bitfield_size - prev_bitfield_free;
                    } else {
                        byteoffset = roundup_bytes(byteoffset, bitoffset);
                        bitoffset = 0;
                        byteoffset = (byteoffset + falign - 1) & !(falign - 1);
                        byteoffset += ftype.size;
                        bitshift = 0;
                        prev_bitfield_size = ftype.size;
                        prev_bitfield_free = 8 * prev_bitfield_size;
                    }
                    prev_bitfield_free -= fbitsize;
                    field_offset_bytes = byteoffset - ftype.size;
                }
                if sflags & SF_GCC_BIG_ENDIAN != 0 {
                    bitshift = 8 * ftype.size - fbitsize - bitshift;
                }
                if !fname.is_empty() {
                    let w_fld = super::ctypestruct::new_cfield(
                        descr.w_ftype,
                        field_offset_bytes,
                        bitshift,
                        fbitsize,
                        fflags,
                    );
                    append(w_fld);
                    record(fname, w_fld);
                }
            }
        }
        byteoffsetmax = byteoffsetmax.max(roundup_bytes(byteoffset, bitoffset));
    }

    // As in C, a structure whose size would be zero is one byte instead; a
    // manually-specified total size of zero is still honoured.
    let alignedsize = ((byteoffsetmax + alignment - 1) & !(alignment - 1)).max(1);
    let totalsize = if totalsize < 0 {
        alignedsize
    } else {
        detect_custom_layout(ct, sflags, alignedsize, totalsize, "wrong total size")?;
        if totalsize < byteoffsetmax {
            return Err(PyError::type_error(format!(
                "{} cannot be of size {totalsize}: there are fields at least up to {byteoffsetmax}",
                ct.name()
            )));
        }
        totalsize
    };
    let totalalignment = if totalalignment < 0 {
        alignment
    } else {
        detect_custom_layout(
            ct,
            sflags,
            alignment,
            totalalignment,
            "wrong total alignment",
        )?;
        totalalignment
    };

    ct.size = totalsize;
    ct.align = totalalignment;
    ct.fields_list = roots.get(list_slot);
    ct.fields_dict = roots.get(dict_slot);
    if with_var_array {
        ct.flags |= ctypeobj::F_WITH_VAR_ARRAY;
    }
    if with_packed_change {
        ct.flags |= ctypeobj::F_WITH_PACKED_CHANGE;
    }
    // The ctype is old-gen and both containers are young, so the barrier has
    // to run after the two writes.
    pyre_object::gc_hook::try_gc_write_barrier_managed(w_ctype.cast::<u8>());
    Ok(())
}

/// The `(name, ctype[, bitsize[, offset]])` tuples `complete_struct_or_union`
/// is given, read before anything allocates on their behalf.
fn read_field_descrs(ct: &W_CType, w_fields: PyObjectRef) -> Result<Vec<FieldDescr>, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let fields_slot = roots.base();
    let _ = roots.pin_root(w_fields);
    let fields_w = crate::baseobjspace::fixedview(roots.get(fields_slot), -1)?;
    let items_slot = pyre_object::gc_roots::shadow_stack_len();
    for &w_field in &fields_w {
        let _ = roots.pin_root(w_field);
    }
    let mut out = Vec::with_capacity(fields_w.len());
    for i in 0..fields_w.len() {
        let field_w = crate::baseobjspace::fixedview(roots.get(items_slot + i), -1)?;
        if !(2..=4).contains(&field_w.len()) {
            return Err(PyError::type_error("bad field descr"));
        }
        let part_slot = pyre_object::gc_roots::shadow_stack_len();
        for &w_part in &field_w {
            let _ = roots.pin_root(w_part);
        }
        let fname = crate::baseobjspace::text_w(roots.get(part_slot))?.to_string();
        let w_ftype = roots.get(part_slot + 1);
        if ctypeobj::ctype_at(w_ftype).is_none() {
            return Err(PyError::type_error(format!(
                "field '{}.{fname}' must be a ctype",
                ct.name()
            )));
        }
        let fbitsize = if field_w.len() > 2 {
            crate::baseobjspace::int_w(roots.get(part_slot + 2))?
        } else {
            -1
        };
        let foffset = if field_w.len() > 3 {
            crate::baseobjspace::int_w(roots.get(part_slot + 3))?
        } else {
            -1
        };
        out.push(FieldDescr {
            fname,
            w_ftype,
            fbitsize,
            foffset,
        });
    }
    Ok(out)
}

// ── enums ───────────────────────────────────────────────────────────────

/// `newtype.py new_enum_type`.
pub fn new_enum_type(
    name: &str,
    w_enumerators: PyObjectRef,
    w_enumvalues: PyObjectRef,
    w_basectype: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let base_slot = roots.base();
    let _ = roots.pin_root(w_basectype);
    let enumerators_w = crate::baseobjspace::fixedview(w_enumerators, -1)?;
    let names_slot = pyre_object::gc_roots::shadow_stack_len();
    for &w in &enumerators_w {
        let _ = roots.pin_root(w);
    }
    let enumvalues_w = crate::baseobjspace::fixedview(w_enumvalues, -1)?;
    let values_slot = pyre_object::gc_roots::shadow_stack_len();
    for &w in &enumvalues_w {
        let _ = roots.pin_root(w);
    }
    if enumerators_w.len() != enumvalues_w.len() {
        return Err(PyError::value_error("tuple args must have the same size"));
    }
    let basectype = ctypeobj::ctype_arg(roots.get(base_slot))?;
    if basectype.kind != ctypeobj::KIND_PRIM_SIGNED
        && basectype.kind != ctypeobj::KIND_PRIM_UNSIGNED
    {
        return Err(PyError::type_error(
            "expected a primitive signed or unsigned base type",
        ));
    }
    // Writing each value through the base type is what detects an
    // out-of-range or badly typed one.
    let probe = super::cdataobj::raw_alloc(basectype.size, true)?;
    for i in 0..enumvalues_w.len() {
        let written = unsafe {
            ctypeobj::convert_from_object(
                ctypeobj::ctype_arg(roots.get(base_slot))?,
                probe,
                roots.get(values_slot + i),
            )
        };
        if let Err(e) = written {
            unsafe { libc::free(probe.cast::<libc::c_void>()) };
            return Err(e);
        }
    }
    unsafe { libc::free(probe.cast::<libc::c_void>()) };

    let basectype = ctypeobj::ctype_arg(roots.get(base_slot))?;
    let w_ctype = ctypeobj::new_ctype(
        basectype.kind,
        basectype.size,
        name,
        name.len() as i64,
        basectype.align,
        pyre_object::PY_NULL,
        -1,
        basectype.flags | ctypeobj::F_ENUM,
    );
    let ctype_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_ctype);
    let e2v_slot = ctype_slot + 1;
    let _ = roots.pin_root(pyre_object::dictmultiobject::w_dict_new());
    let v2e_slot = e2v_slot + 1;
    let _ = roots.pin_root(pyre_object::dictmultiobject::w_dict_new());
    // Walking backwards is what lets an earlier duplicate win, as upstream's
    // reversed fill does.
    for i in (0..enumerators_w.len()).rev() {
        let w_name = roots.get(names_slot + i);
        let w_value = roots.get(values_slot + i);
        crate::baseobjspace::setitem(roots.get(e2v_slot), w_name, w_value)?;
        crate::baseobjspace::setitem(roots.get(v2e_slot), w_value, w_name)?;
    }
    let ct = ctypeobj::ctype_arg(roots.get(ctype_slot))?;
    ct.enumerators2values = roots.get(e2v_slot);
    ct.enumvalues2erators = roots.get(v2e_slot);
    pyre_object::gc_hook::try_gc_write_barrier_managed(roots.get(ctype_slot).cast::<u8>());
    Ok(roots.get(ctype_slot))
}

// ── function types ──────────────────────────────────────────────────────

/// `UniqueCache.functions` — keyed by the result ctype, the argument ctypes,
/// the ellipsis flag and the ABI, all of which decide the type's identity.
type FunctionKey = (usize, Vec<usize>, bool, i64);

fn function_cache() -> &'static Mutex<HashMap<FunctionKey, usize>> {
    static CACHE: OnceLock<Mutex<HashMap<FunctionKey, usize>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `newtype.py new_function_type`.
pub fn new_function_type(
    w_fargs: PyObjectRef,
    w_fresult: PyObjectRef,
    ellipsis: bool,
    abi: i64,
) -> Result<PyObjectRef, PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let result_slot = roots.base();
    let _ = roots.pin_root(w_fresult);
    let args_w = crate::baseobjspace::fixedview(w_fargs, -1)?;
    let mut fargs = Vec::with_capacity(args_w.len());
    for w_farg in args_w {
        let Some(farg) = ctypeobj::ctype_at(w_farg) else {
            return Err(PyError::type_error(
                "first arg must be a tuple of ctype objects",
            ));
        };
        // An array argument decays to the pointer it was built from.
        fargs.push(if farg.kind == ctypeobj::KIND_ARRAY {
            farg.ctptr
        } else {
            w_farg
        });
    }
    build_function_type(&fargs, roots.get(result_slot), ellipsis, abi)
}

/// `newtype.py _new_function_type` and `_build_function_type`.
pub fn build_function_type(
    fargs: &[PyObjectRef],
    w_fresult: PyObjectRef,
    ellipsis: bool,
    abi: i64,
) -> Result<PyObjectRef, PyError> {
    let key: FunctionKey = (
        w_fresult as usize,
        fargs.iter().map(|&a| a as usize).collect(),
        ellipsis,
        abi,
    );
    {
        let cache = function_cache()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(&existing) = cache.get(&key) {
            return Ok(existing as PyObjectRef);
        }
    }
    let fresult = ctypeobj::ctype_arg(w_fresult)?;
    if (fresult.size < 0 && fresult.kind != ctypeobj::KIND_VOID)
        || fresult.kind == ctypeobj::KIND_ARRAY
    {
        return Err(PyError::type_error(
            if fresult.is_struct_or_union() && fresult.size < 0 {
                format!("result type '{}' is opaque", fresult.name())
            } else {
                format!("invalid result type: '{}'", fresult.name())
            },
        ));
    }
    // `W_CTypeFunc.__init__`.
    let (extra, xpos) = super::ctypefunc::compute_extra_text(fargs, ellipsis, abi);
    let (name, name_position) = fresult.insert_name(&extra, xpos);
    let w_ctype = ctypeobj::new_ctype(
        ctypeobj::KIND_FUNC,
        std::mem::size_of::<*const u8>() as i64,
        &name,
        name_position,
        -1,
        w_fresult,
        -1,
        if ellipsis { ctypeobj::F_ELLIPSIS } else { 0 },
    );
    let roots = pyre_object::gc_roots::push_roots();
    let ctype_slot = roots.base();
    let _ = roots.pin_root(w_ctype);
    let fargs_slot = ctype_slot + 1;
    let _ = roots.pin_root(pyre_object::w_tuple_new(fargs.to_vec()));
    let ct = ctypeobj::ctype_arg(roots.get(ctype_slot))?;
    ct.fargs = roots.get(fargs_slot);
    ct.abi = abi;
    pyre_object::gc_hook::try_gc_write_barrier_managed(roots.get(ctype_slot).cast::<u8>());
    if !ellipsis {
        // A function taking '...' is stored without a cif at all: its cif is
        // computed per call from the types actually passed.  For every other
        // one it is computed once, here.  A NotImplementedError is eaten; the
        // call itself raises it if one is ever made.
        match super::ctypefunc::build_cif_descr(fargs, w_fresult, abi, None) {
            Ok(cif) => ct.cif_descr = cif,
            Err(e) if e.kind == crate::PyErrorKind::NotImplementedError => {}
            Err(e) => return Err(e),
        }
    }
    let mut cache = function_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Ok(*cache.entry(key).or_insert(roots.get(ctype_slot) as usize) as PyObjectRef)
}
