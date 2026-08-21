//! `complex` -- PyPy `cpyext/complexobject.py`.

use super::object::{argument, is_exactly};
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::{c_double, c_int};

/// `Py_complex` — the pair a complex is carried across the boundary as.
///
/// Upstream reaches this one through a pointer, because lltype cannot return a
/// struct; here it is returned by value, which is the convention an extension
/// compiled against CPython's header already calls with.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CPyComplex {
    pub real: c_double,
    pub imag: c_double,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyComplex_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    if object.is_null() {
        return 0;
    }
    if unsafe { pyre_object::is_complex(object) } {
        return 1;
    }
    match crate::typedef::gettypefor(&pyre_object::COMPLEX_TYPE) {
        Some(complex_type) => unsafe {
            crate::baseobjspace::isinstance_w(object, complex_type.as_ptr()) as c_int
        },
        None => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyComplex_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && is_exactly(object, &pyre_object::COMPLEX_TYPE)) as c_int
}

/// `complexobject.py PyComplex_FromDoubles`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyComplex_FromDoubles(real: c_double, imag: c_double) -> *mut CPyObject {
    pyobject::make_ref(pyre_object::complexobject::w_complex_new(real, imag))
}

/// `complexobject.py _PyComplex_FromCComplex`, which upstream spells with a
/// pointer for want of a struct argument.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyComplex_FromCComplex(value: CPyComplex) -> *mut CPyObject {
    unsafe { PyComplex_FromDoubles(value.real, value.imag) }
}

/// The pair behind `object`, through `__complex__` where it is not a complex.
///
/// `complex()` itself is not what this is: that reads a `str`, which no
/// conversion of a *number* may accept.
fn unpack(object: PyObjectRef) -> Result<(f64, f64), crate::PyError> {
    if unsafe { pyre_object::is_complex(object) } {
        return Ok(unsafe {
            (
                pyre_object::complexobject::w_complex_get_real(object),
                pyre_object::complexobject::w_complex_get_imag(object),
            )
        });
    }
    if let Some(method) = unsafe { crate::baseobjspace::lookup(object, "__complex__") } {
        let converted = crate::call::call_function_impl_result(method, &[object])?;
        if !unsafe { pyre_object::is_complex(converted) } {
            return Err(crate::PyError::type_error(
                "__complex__ should return a complex object".to_string(),
            ));
        }
        return Ok(unsafe {
            (
                pyre_object::complexobject::w_complex_get_real(converted),
                pyre_object::complexobject::w_complex_get_imag(converted),
            )
        });
    }
    // Anything that is a float, or that can be read as one.
    Ok((crate::baseobjspace::float_w(object)?, 0.0))
}

/// `complexobject.py PyComplex_RealAsDouble`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyComplex_RealAsDouble(object: *mut CPyObject) -> c_double {
    let Some(object) = argument(object) else {
        return -1.0;
    };
    if unsafe { pyre_object::is_complex(object) } {
        return unsafe { pyre_object::complexobject::w_complex_get_real(object) };
    }
    super::pyerrors::trap(crate::baseobjspace::float_w(object)).unwrap_or(-1.0)
}

/// `complexobject.py PyComplex_ImagAsDouble`.
///
/// Anything that is not a complex has no imaginary part to report, and this
/// answers 0.0 for it without asking whether it is a number at all.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyComplex_ImagAsDouble(object: *mut CPyObject) -> c_double {
    let Some(object) = argument(object) else {
        return -1.0;
    };
    if unsafe { pyre_object::is_complex(object) } {
        unsafe { pyre_object::complexobject::w_complex_get_imag(object) }
    } else {
        0.0
    }
}

/// `complexobject.py _PyComplex_AsCComplex`.
///
/// The failure answer is a real part of -1.0 with the error recorded, which is
/// what a caller distinguishes by asking `PyErr_Occurred`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyComplex_AsCComplex(object: *mut CPyObject) -> CPyComplex {
    let failed = CPyComplex {
        real: -1.0,
        imag: 0.0,
    };
    let Some(object) = argument(object) else {
        return failed;
    };
    match super::pyerrors::trap(unpack(object)) {
        Some((real, imag)) => CPyComplex { real, imag },
        None => failed,
    }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyComplex_Check as *const ());
    std::hint::black_box(PyComplex_CheckExact as *const ());
    std::hint::black_box(PyComplex_FromDoubles as *const ());
    std::hint::black_box(PyComplex_FromCComplex as *const ());
    std::hint::black_box(PyComplex_RealAsDouble as *const ());
    std::hint::black_box(PyComplex_ImagAsDouble as *const ());
    std::hint::black_box(PyComplex_AsCComplex as *const ());
}
