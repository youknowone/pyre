//! `float` -- PyPy `cpyext/floatobject.py`.

use super::object::argument;
use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use std::ffi::{c_double, c_int};

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

pub(super) fn ensure_linked() {
    std::hint::black_box(PyFloat_FromDouble as *const ());
    std::hint::black_box(PyFloat_AsDouble as *const ());
    std::hint::black_box(PyFloat_Check as *const ());
    std::hint::black_box(PyFloat_CheckExact as *const ());
}
