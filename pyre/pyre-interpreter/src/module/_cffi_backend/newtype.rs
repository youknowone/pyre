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
