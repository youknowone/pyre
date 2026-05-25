//! cmath module — PyPy: pypy/module/cmath/
//!
//! Complex math functions via `pymath::cmath`.  pyre lacks
//! `W_ComplexObject` so the real-valued subset is registered; complex
//! arithmetic will require a follow-up.  `infj` / `nanj` are deferred
//! along with the complex type.

use crate::module::math::interp_math;
use pyre_object::*;

fn phase_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(floatobject::w_float_new(
        interp_math::get_double(args[0]).atan2(0.0),
    ))
}

fn polar_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let x = interp_math::get_double(args[0]);
    Ok(w_tuple_new(vec![
        floatobject::w_float_new(x.abs()),
        floatobject::w_float_new(0.0),
    ]))
}

fn rect_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let r = interp_math::get_double(args[0]);
    let phi = interp_math::get_double(args[1]);
    Ok(floatobject::w_float_new(r * phi.cos()))
}

fn isfinite_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(interp_math::get_double(args[0]).is_finite()))
}

fn isinf_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(interp_math::get_double(args[0]).is_infinite()))
}

fn isnan_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(interp_math::get_double(args[0]).is_nan()))
}

crate::py_module! {
    "cmath",
    interpleveldefs: {
        "pi"  => floatobject::w_float_new(pymath::math::PI),
        "e"   => floatobject::w_float_new(pymath::math::E),
        "tau" => floatobject::w_float_new(pymath::math::TAU),
        "inf" => floatobject::w_float_new(pymath::math::INF),
        "nan" => floatobject::w_float_new(pymath::math::NAN),
    },
    functions: {
        "phase" / 1 = phase_impl,
        "polar" / 1 = polar_impl,
        "rect"  / 2 = rect_impl,

        "isfinite" / 1 = isfinite_impl,
        "isinf"    / 1 = isinf_impl,
        "isnan"    / 1 = isnan_impl,

        // Real-valued forwards (pending complex type)
        "sqrt"  / 1 = interp_math::sqrt,
        "exp"   / 1 = interp_math::exp,
        "log10" / 1 = interp_math::log10,
        "sin"   / 1 = interp_math::sin,
        "cos"   / 1 = interp_math::cos,
        "tan"   / 1 = interp_math::tan,
        "asin"  / 1 = interp_math::asin,
        "acos"  / 1 = interp_math::acos,
        "atan"  / 1 = interp_math::atan,
        "sinh"  / 1 = interp_math::sinh,
        "cosh"  / 1 = interp_math::cosh,
        "tanh"  / 1 = interp_math::tanh,
        "asinh" / 1 = interp_math::asinh,
        "acosh" / 1 = interp_math::acosh,
        "atanh" / 1 = interp_math::atanh,
        "log"   / * = interp_math::log,
    },
}
