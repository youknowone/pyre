//! cmath module — complex math functions via pymath::cmath.
//!
//! PyPy equivalent: pypy/module/cmath/
//!
//! Stub: complex number type not yet implemented in pyre.
//! Functions are registered so `import cmath` succeeds, but complex
//! arithmetic requires W_ComplexObject (future work).

use crate::{PyNamespace, builtin_code_new, namespace_store};
use pyre_object::*;

pub fn init(ns: &mut PyNamespace) {
    // Constants
    namespace_store(ns, "pi", floatobject::w_float_new(pymath::math::PI));
    namespace_store(ns, "e", floatobject::w_float_new(pymath::math::E));
    namespace_store(ns, "tau", floatobject::w_float_new(pymath::math::TAU));
    namespace_store(ns, "inf", floatobject::w_float_new(pymath::math::INF));
    namespace_store(ns, "nan", floatobject::w_float_new(pymath::math::NAN));
    // infj, nanj would need complex type

    // Real-valued functions (work on float, stub for complex)
    namespace_store(
        ns,
        "phase",
        builtin_code_new("phase", |args| {
            Ok(floatobject::w_float_new(
                super::interp_math::get_double(args[0]).atan2(0.0),
            ))
        }),
    );
    namespace_store(
        ns,
        "polar",
        builtin_code_new("polar", |args| {
            let x = super::interp_math::get_double(args[0]);
            Ok(w_tuple_new(vec![
                floatobject::w_float_new(x.abs()),
                floatobject::w_float_new(0.0),
            ]))
        }),
    );
    namespace_store(
        ns,
        "rect",
        builtin_code_new("rect", |args| {
            let r = super::interp_math::get_double(args[0]);
            let phi = super::interp_math::get_double(args[1]);
            Ok(floatobject::w_float_new(r * phi.cos()))
        }),
    );
    namespace_store(
        ns,
        "isfinite",
        builtin_code_new("isfinite", |args| {
            Ok(w_bool_from(
                super::interp_math::get_double(args[0]).is_finite(),
            ))
        }),
    );
    namespace_store(
        ns,
        "isinf",
        builtin_code_new("isinf", |args| {
            Ok(w_bool_from(
                super::interp_math::get_double(args[0]).is_infinite(),
            ))
        }),
    );
    namespace_store(
        ns,
        "isnan",
        builtin_code_new("isnan", |args| {
            Ok(w_bool_from(
                super::interp_math::get_double(args[0]).is_nan(),
            ))
        }),
    );

    // Forward trig/exp functions to math equivalents for real input
    for (name, func) in [
        (
            "sqrt",
            super::interp_math::sqrt as fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
        ),
        ("exp", super::interp_math::exp),
        ("log", super::interp_math::log),
        ("log10", super::interp_math::log10),
        ("sin", super::interp_math::sin),
        ("cos", super::interp_math::cos),
        ("tan", super::interp_math::tan),
        ("asin", super::interp_math::asin),
        ("acos", super::interp_math::acos),
        ("atan", super::interp_math::atan),
        ("sinh", super::interp_math::sinh),
        ("cosh", super::interp_math::cosh),
        ("tanh", super::interp_math::tanh),
        ("asinh", super::interp_math::asinh),
        ("acosh", super::interp_math::acosh),
        ("atanh", super::interp_math::atanh),
    ] {
        namespace_store(ns, name, builtin_code_new(name, func));
    }
}
