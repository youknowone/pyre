//! cmath module — PyPy: pypy/module/cmath/
//!
//! Complex math functions via `pymath::cmath`.  pyre lacks
//! `W_ComplexObject` so the real-valued subset is registered; complex
//! arithmetic will require a follow-up.  `infj` / `nanj` are deferred
//! along with the complex type.

use crate::module::math::interp_math;
use pyre_object::*;

/// `cmath.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)` — complex
/// `_Py_c_isclose` equivalent over the two operands' components.
fn isclose_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["rel_tol", "abs_tol"], "isclose")?;
    if pos.len() < 2 {
        return Err(crate::PyError::type_error(
            "isclose() missing required argument",
        ));
    }
    let val = |o: PyObjectRef| {
        unsafe { crate::objspace::descroperation::complex_val(o) }
            .ok_or_else(|| crate::PyError::type_error("must be a number"))
    };
    let (ar, ai) = val(pos[0])?;
    let (br, bi) = val(pos[1])?;
    let tol = |name: &str, default: f64| -> Result<f64, crate::PyError> {
        match crate::builtins::kwarg_get(kwargs, name) {
            Some(v) => crate::baseobjspace::float_w(v),
            None => Ok(default),
        }
    };
    let rel_tol = tol("rel_tol", 1e-9)?;
    let abs_tol = tol("abs_tol", 0.0)?;
    if rel_tol < 0.0 || abs_tol < 0.0 {
        return Err(crate::PyError::value_error(
            "tolerances must be non-negative",
        ));
    }
    // Exact equality (covers the inf == inf case).
    if ar == br && ai == bi {
        return Ok(w_bool_from(true));
    }
    // Any infinity that is not an exact match is not close.
    if ar.is_infinite() || ai.is_infinite() || br.is_infinite() || bi.is_infinite() {
        return Ok(w_bool_from(false));
    }
    let diff = (ar - br).hypot(ai - bi);
    let mag_a = ar.hypot(ai);
    let mag_b = br.hypot(bi);
    let close = diff <= (rel_tol * mag_a.max(mag_b)).max(abs_tol);
    Ok(w_bool_from(close))
}

fn polar_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let x = interp_math::get_double(args[0]);
    Ok(w_tuple_new(vec![
        floatobject::w_float_new(x.abs()),
        floatobject::w_float_new(0.0),
    ]))
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
        "phase" / 1 = |args| Ok(floatobject::w_float_new(interp_math::get_double(args[0]).atan2(0.0))),
        "polar" / 1 = polar_impl,
        "rect"  / 2 = |args| Ok(floatobject::w_float_new(
            interp_math::get_double(args[0]) * interp_math::get_double(args[1]).cos()
        )),

        "isfinite" / 1 = |args| Ok(w_bool_from(interp_math::get_double(args[0]).is_finite())),
        "isinf"    / 1 = |args| Ok(w_bool_from(interp_math::get_double(args[0]).is_infinite())),
        "isnan"    / 1 = |args| Ok(w_bool_from(interp_math::get_double(args[0]).is_nan())),
        "isclose"  / * = isclose_impl,

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
