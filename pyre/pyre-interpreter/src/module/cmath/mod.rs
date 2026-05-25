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

        "phase" => crate::make_builtin_function_with_arity("phase", phase_impl, 1),
        "polar" => crate::make_builtin_function_with_arity("polar", polar_impl, 1),
        "rect"  => crate::make_builtin_function_with_arity("rect",  rect_impl,  2),

        "isfinite" => crate::make_builtin_function_with_arity("isfinite", isfinite_impl, 1),
        "isinf"    => crate::make_builtin_function_with_arity("isinf",    isinf_impl,    1),
        "isnan"    => crate::make_builtin_function_with_arity("isnan",    isnan_impl,    1),

        // Real-valued forwards (pending complex type)
        "sqrt"  => crate::make_builtin_function_with_arity("sqrt",  interp_math::sqrt,  1),
        "exp"   => crate::make_builtin_function_with_arity("exp",   interp_math::exp,   1),
        "log10" => crate::make_builtin_function_with_arity("log10", interp_math::log10, 1),
        "sin"   => crate::make_builtin_function_with_arity("sin",   interp_math::sin,   1),
        "cos"   => crate::make_builtin_function_with_arity("cos",   interp_math::cos,   1),
        "tan"   => crate::make_builtin_function_with_arity("tan",   interp_math::tan,   1),
        "asin"  => crate::make_builtin_function_with_arity("asin",  interp_math::asin,  1),
        "acos"  => crate::make_builtin_function_with_arity("acos",  interp_math::acos,  1),
        "atan"  => crate::make_builtin_function_with_arity("atan",  interp_math::atan,  1),
        "sinh"  => crate::make_builtin_function_with_arity("sinh",  interp_math::sinh,  1),
        "cosh"  => crate::make_builtin_function_with_arity("cosh",  interp_math::cosh,  1),
        "tanh"  => crate::make_builtin_function_with_arity("tanh",  interp_math::tanh,  1),
        "asinh" => crate::make_builtin_function_with_arity("asinh", interp_math::asinh, 1),
        "acosh" => crate::make_builtin_function_with_arity("acosh", interp_math::acosh, 1),
        "atanh" => crate::make_builtin_function_with_arity("atanh", interp_math::atanh, 1),
        // `log` takes optional base — variable arity
        "log"   => crate::make_builtin_function("log", interp_math::log),
    }
}
