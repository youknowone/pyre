//! The number protocol -- PyPy `cpyext/number.py`.
//!
//! Each entry point is the interpreter's own operator, not a direct slot call:
//! the operand on either side may be a pyre object, and the binary-operator
//! dispatch is what decides between the two.

use super::object::{argument, arguments, result};
use super::pyerrors::trap;
use super::pyobject::CPyObject;
use crate::bytecode::BinaryOperator;
use pyre_object::PyObjectRef;
use std::ffi::c_int;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Add(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::add(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Subtract(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::sub(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Multiply(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::mul(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_MatrixMultiply(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::matmul(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_FloorDivide(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::floordiv(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_TrueDivide(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::truediv(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Remainder(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::mod_(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Divmod(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::divmod(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Lshift(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::lshift(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Rshift(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::rshift(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_And(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::and_(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Xor(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::xor(left, right))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Or(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::or_(left, right))
}

/// The in-place forms consult `__iadd__` and its siblings before falling back
/// to the binary operator, which is what `binary_value` does for the in-place
/// opcodes.

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceAdd(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceAdd,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceSubtract(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceSubtract,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceMultiply(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceMultiply,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceMatrixMultiply(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceMatrixMultiply,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceFloorDivide(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceFloorDivide,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceTrueDivide(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceTrueDivide,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceRemainder(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceRemainder,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceLshift(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceLshift,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceRshift(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceRshift,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceAnd(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceAnd,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceXor(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceXor,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlaceOr(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some([left, right]) = arguments([left, right]) else {
        return std::ptr::null_mut();
    };
    result(crate::opcode_ops::binary_value(
        left,
        right,
        BinaryOperator::InplaceOr,
    ))
}

fn absolute(object: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    crate::builtins::builtin_abs(&[object])
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Negative(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::neg(object))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Positive(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::pos(object))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Invert(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::invert(object))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Absolute(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(absolute(object))
}

/// `PyNumber_Power(base, exponent, modulus)` — `Py_None` for the two-argument
/// form, which is the only one the interpreter's `pow` operator covers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Power(
    base: *mut CPyObject,
    exponent: *mut CPyObject,
    modulus: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([base, exponent, modulus]);
    let Some([base, exponent]) = arguments([base, exponent]) else {
        return std::ptr::null_mut();
    };
    let modulus = unsafe { super::pyobject::from_ref(modulus) };
    if modulus.is_null() || unsafe { pyre_object::is_none(modulus) } {
        return result(crate::pow(base, exponent));
    }
    result(crate::objspace::descroperation::pow3(
        base, exponent, modulus,
    ))
}

/// `PyNumber_InPlacePower(base, exponent, modulus)` — a modulus other than
/// `None` has no in-place operator to reach, so it is refused
/// (`number.py:146-152`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_InPlacePower(
    base: *mut CPyObject,
    exponent: *mut CPyObject,
    modulus: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([base, exponent, modulus]);
    let Some([base, exponent]) = arguments([base, exponent]) else {
        return std::ptr::null_mut();
    };
    let modulus = unsafe { super::pyobject::from_ref(modulus) };
    if !modulus.is_null() && !unsafe { pyre_object::is_none(modulus) } {
        return result(Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "PyNumber_InPlacePower with non-None modulus is not supported",
        )));
    }
    result(crate::opcode_ops::binary_value(
        base,
        exponent,
        BinaryOperator::InplacePower,
    ))
}

/// `PyNumber_ToBase(n, base)` — `n.__index__()` written in `base` behind its
/// `0b`/`0o`/`0x` marker (`number.py:57-83`).
///
/// Any other base is a `SystemError`: the marker table has no entry for it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_ToBase(object: *mut CPyObject, base: c_int) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    let prefix = match base {
        2 => "0b",
        8 => "0o",
        16 => "0x",
        // Base ten carries no marker, so it is the plain decimal spelling of
        // the index rather than a radix format.
        10 => {
            let decimal = crate::baseobjspace::space_index(object)
                .and_then(|index| crate::builtins::builtin_str(&[index]));
            return result(decimal);
        }
        _ => {
            return result(Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                "PyNumber_ToBase: base must be 2, 8, 10 or 16",
            )));
        }
    };
    result(
        crate::builtins::format_index_radix(object, base as u32, prefix)
            .map(|written| pyre_object::w_str_new(&written)),
    )
}

/// `nb_index`, `nb_int` or `nb_float` — the test `PyNumber_Check` performs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { super::pyobject::from_ref(object) };
    if object.is_null() {
        return 0;
    }
    let numeric = unsafe {
        pyre_object::pyobject::is_int_or_long(object)
            || pyre_object::is_float(object)
            || crate::baseobjspace::lookup(object, "__index__").is_some()
            || crate::baseobjspace::lookup(object, "__int__").is_some()
            || crate::baseobjspace::lookup(object, "__float__").is_some()
    };
    numeric as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Index(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::space_index(object))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_Float(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::float_w(object).map(pyre_object::w_float_new))
}

/// `PyNumber_AsSsize_t(object, exc)` — `exc` is the class an overflow is
/// reported with, and NULL asks for the clamp instead.
///
/// `getindex_w` is the clamping half: an index too large for the machine word
/// comes back as `i64::MAX` or `i64::MIN` rather than as an error, which is
/// what a NULL `exc` wants and what a non-NULL one must not get.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyNumber_AsSsize_t(object: *mut CPyObject, exc: *mut CPyObject) -> isize {
    super::object::realize_all([object, exc]);
    let Some(object) = argument(object) else {
        return -1;
    };
    if exc.is_null() {
        return match trap(crate::baseobjspace::getindex_w(object)) {
            Some(value) => value as isize,
            None => -1,
        };
    }
    let Some(index) = trap(crate::baseobjspace::space_index(object)) else {
        return -1;
    };
    match crate::baseobjspace::int_w(index) {
        Ok(value) => value as isize,
        Err(error) if error.kind == crate::PyErrorKind::OverflowError => {
            let message = format!(
                "cannot fit '{}' into an index-sized integer",
                crate::type_methods::arg_type_name(object)
            );
            let Ok(message) = std::ffi::CString::new(message) else {
                return -1;
            };
            unsafe { super::pyerrors::PyErr_SetString(exc, message.as_ptr()) };
            -1
        }
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            -1
        }
    }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyNumber_Add as *const ());
    std::hint::black_box(PyNumber_Subtract as *const ());
    std::hint::black_box(PyNumber_Multiply as *const ());
    std::hint::black_box(PyNumber_MatrixMultiply as *const ());
    std::hint::black_box(PyNumber_FloorDivide as *const ());
    std::hint::black_box(PyNumber_TrueDivide as *const ());
    std::hint::black_box(PyNumber_Remainder as *const ());
    std::hint::black_box(PyNumber_Divmod as *const ());
    std::hint::black_box(PyNumber_Lshift as *const ());
    std::hint::black_box(PyNumber_Rshift as *const ());
    std::hint::black_box(PyNumber_And as *const ());
    std::hint::black_box(PyNumber_Xor as *const ());
    std::hint::black_box(PyNumber_Or as *const ());
    std::hint::black_box(PyNumber_InPlaceAdd as *const ());
    std::hint::black_box(PyNumber_InPlaceSubtract as *const ());
    std::hint::black_box(PyNumber_InPlaceMultiply as *const ());
    std::hint::black_box(PyNumber_InPlaceMatrixMultiply as *const ());
    std::hint::black_box(PyNumber_InPlaceFloorDivide as *const ());
    std::hint::black_box(PyNumber_InPlaceTrueDivide as *const ());
    std::hint::black_box(PyNumber_InPlaceRemainder as *const ());
    std::hint::black_box(PyNumber_InPlaceLshift as *const ());
    std::hint::black_box(PyNumber_InPlaceRshift as *const ());
    std::hint::black_box(PyNumber_InPlaceAnd as *const ());
    std::hint::black_box(PyNumber_InPlaceXor as *const ());
    std::hint::black_box(PyNumber_InPlaceOr as *const ());
    std::hint::black_box(PyNumber_Negative as *const ());
    std::hint::black_box(PyNumber_Positive as *const ());
    std::hint::black_box(PyNumber_Invert as *const ());
    std::hint::black_box(PyNumber_Absolute as *const ());
    std::hint::black_box(PyNumber_Power as *const ());
    std::hint::black_box(PyNumber_InPlacePower as *const ());
    std::hint::black_box(PyNumber_ToBase as *const ());
    std::hint::black_box(PyNumber_Check as *const ());
    std::hint::black_box(PyNumber_Index as *const ());
    std::hint::black_box(PyNumber_Float as *const ());
    std::hint::black_box(PyNumber_AsSsize_t as *const ());
}
