//! `float` -- PyPy `cpyext/floatobject.py`.

use super::object::argument;
use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use std::ffi::{c_char, c_double, c_int};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_FromDouble(value: c_double) -> *mut CPyObject {
    pyobject::make_ref(pyre_object::w_float_new(value))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_AsDouble(object: *mut CPyObject) -> c_double {
    let Some(object) = argument(object) else {
        return -1.0;
    };
    trap(crate::baseobjspace::float_w(object)).unwrap_or(-1.0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::is_float(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && super::object::is_exactly(object, &pyre_object::FLOAT_TYPE)) as c_int
}

/// `floatobject.py PyFloat_AS_DOUBLE` — the read [`PyFloat_AsDouble`] makes,
/// under a name whose macro spelling promises no error checking.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_AS_DOUBLE(object: *mut CPyObject) -> c_double {
    unsafe { PyFloat_AsDouble(object) }
}

/// `floatobject.py PyFloat_FromString` — `float(o)`.
///
/// The argument is named `str` upstream and read as one, but nothing here
/// narrows it: `float` is what decides what it accepts.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_FromString(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    let float_type = crate::typedef::gettypeobject(&pyre_object::FLOAT_TYPE);
    super::object::result(crate::call::call_function_impl_result(float_type, &[object]))
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyFloat_FromDouble as *const ());
    std::hint::black_box(PyFloat_AsDouble as *const ());
    std::hint::black_box(PyFloat_FromString as *const ());
    std::hint::black_box(PyFloat_AS_DOUBLE as *const ());
    std::hint::black_box(PyFloat_Check as *const ());
    std::hint::black_box(PyFloat_CheckExact as *const ());
    std::hint::black_box(PyFloat_Pack2 as *const ());
    std::hint::black_box(PyFloat_Pack4 as *const ());
    std::hint::black_box(PyFloat_Pack8 as *const ());
    std::hint::black_box(PyFloat_Unpack2 as *const ());
    std::hint::black_box(PyFloat_Unpack4 as *const ());
    std::hint::black_box(PyFloat_Unpack8 as *const ());
}

// ── the pack and unpack entry points ────────────────────────────────────
//
// `_PyFloat_InitState` decides between three formats by looking at the bits
// of a chosen value; here `f32` and `f64` are IEEE 754 binary32 and binary64
// by the language's own definition, so the arithmetic-and-masks path those
// entry points keep for a machine that is neither has no machine to run on
// and is not written.

/// Write `bytes` to `data`, which the caller has ordered already.
fn store(data: *mut c_char, bytes: &[u8]) {
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data as *mut u8, bytes.len()) };
}

fn overflow(code: char) -> c_int {
    super::pyerrors::set_pending_error(crate::PyError::new(
        crate::PyErrorKind::OverflowError,
        format!("float too large to pack with {code} format"),
    ));
    -1
}

/// `floatobject.c:1993 PyFloat_Pack2`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_Pack2(x: c_double, data: *mut c_char, le: c_int) -> c_int {
    // The rounding and the `OverflowError` it raises are `pack_half`'s.
    let Some(bits) = trap(crate::module::r#struct::pack_half(x)) else {
        return -1;
    };
    let bytes = match le != 0 {
        true => bits.to_le_bytes(),
        false => bits.to_be_bytes(),
    };
    store(data, &bytes);
    0
}

/// `floatobject.c:2101 PyFloat_Pack4`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_Pack4(x: c_double, data: *mut c_char, le: c_int) -> c_int {
    let narrowed = x as f32;
    if narrowed.is_infinite() && !x.is_infinite() {
        return overflow('f');
    }
    let mut bits = narrowed.to_bits();
    if x.is_nan() {
        // Narrowing quiets a signalling NaN, so a payload that survived it
        // says the value was one and the quiet bit goes back off.
        let wide = x.to_bits();
        if wide & (1 << 51) == 0 && bits & 0x3fffff != 0 {
            bits &= !(1 << 22);
        }
    }
    let bytes = match le != 0 {
        true => bits.to_le_bytes(),
        false => bits.to_be_bytes(),
    };
    store(data, &bytes);
    0
}

/// `floatobject.c:2247 PyFloat_Pack8`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_Pack8(x: c_double, data: *mut c_char, le: c_int) -> c_int {
    let bytes = match le != 0 {
        true => x.to_bits().to_le_bytes(),
        false => x.to_bits().to_be_bytes(),
    };
    store(data, &bytes);
    0
}

/// The `N` bytes at `data` as an integer, read in the order `le` asks for.
fn load<const N: usize>(data: *const c_char, le: c_int) -> [u8; N] {
    let mut raw = [0u8; N];
    unsafe { std::ptr::copy_nonoverlapping(data as *const u8, raw.as_mut_ptr(), N) };
    if le == 0 {
        raw.reverse();
    }
    raw
}

/// `floatobject.c:2379 PyFloat_Unpack2`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_Unpack2(data: *const c_char, le: c_int) -> c_double {
    crate::module::r#struct::unpack_half(u16::from_le_bytes(load(data, le)))
}

/// `floatobject.c:2435 PyFloat_Unpack4`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_Unpack4(data: *const c_char, le: c_int) -> c_double {
    let narrow = f32::from_bits(u32::from_le_bytes(load(data, le)));
    let widened = narrow as f64;
    if narrow.is_nan() && narrow.to_bits() & (1 << 22) == 0 {
        // Widening quiets it, and the value it came from was signalling.
        return f64::from_bits(widened.to_bits() & !(1u64 << 51));
    }
    widened
}

/// `floatobject.c:2549 PyFloat_Unpack8`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyFloat_Unpack8(data: *const c_char, le: c_int) -> c_double {
    f64::from_bits(u64::from_le_bytes(load(data, le)))
}
