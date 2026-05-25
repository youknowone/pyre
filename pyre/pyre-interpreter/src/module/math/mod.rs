//! math module — PyPy: pypy/module/math/
//!
//! Function bodies live in `interp_math`; this declarative table mirrors
//! `moduledef.py` interpleveldefs.

pub mod interp_math;

use interp_math as m;

crate::py_module! {
    "math",
    interpleveldefs: {
        "e"   => pyre_object::floatobject::w_float_new(pymath::math::E),
        "pi"  => pyre_object::floatobject::w_float_new(pymath::math::PI),
        "tau" => pyre_object::floatobject::w_float_new(pymath::math::TAU),
        "inf" => pyre_object::floatobject::w_float_new(pymath::math::INF),
        "nan" => pyre_object::floatobject::w_float_new(pymath::math::NAN),
    },
    functions: {
        // Trigonometric
        "sin"   / 1 = m::sin,
        "cos"   / 1 = m::cos,
        "tan"   / 1 = m::tan,
        "asin"  / 1 = m::asin,
        "acos"  / 1 = m::acos,
        "atan"  / 1 = m::atan,
        "atan2" / 2 = m::atan2,
        "sinh"  / 1 = m::sinh,
        "cosh"  / 1 = m::cosh,
        "tanh"  / 1 = m::tanh,
        "asinh" / 1 = m::asinh,
        "acosh" / 1 = m::acosh,
        "atanh" / 1 = m::atanh,

        // Exponential / logarithmic
        "sqrt"  / 1 = m::sqrt,
        "cbrt"  / 1 = m::cbrt,
        "exp"   / 1 = m::exp,
        "exp2"  / 1 = m::exp2,
        "expm1" / 1 = m::expm1,
        "log"   / * = m::log,
        "log2"  / 1 = m::log2,
        "log10" / 1 = m::log10,
        "log1p" / 1 = m::log1p,
        "pow"   / 2 = m::pow,

        // Gamma / error
        "erf"    / 1 = m::erf,
        "erfc"   / 1 = m::erfc,
        "gamma"  / 1 = m::gamma,
        "lgamma" / 1 = m::lgamma,

        // Rounding / truncation
        "floor" / 1 = m::floor,
        "ceil"  / 1 = m::ceil,
        "trunc" / 1 = m::trunc,

        // Floating-point manipulation
        "fabs"      / 1 = m::fabs,
        "fmod"      / 2 = m::fmod,
        "copysign"  / 2 = m::copysign,
        "remainder" / 2 = m::remainder,
        "frexp"     / 1 = m::frexp,
        "ldexp"     / 2 = m::ldexp,
        "modf"      / 1 = m::modf,
        "nextafter" / * = m::nextafter,
        "ulp"       / 1 = m::ulp,
        "fma"       / 3 = m::fma,

        // Classification
        "isinf"    / 1 = m::isinf,
        "isnan"    / 1 = m::isnan,
        "isfinite" / 1 = m::isfinite,
        "isclose"  / * = m::isclose,

        // Conversion
        "degrees" / 1 = m::degrees,
        "radians" / 1 = m::radians,

        // Multi-dimensional
        "hypot" / * = m::hypot,
        "dist"  / 2 = m::dist,

        // Aggregation
        "fsum"    / 1 = m::fsum,
        "prod"    / * = m::prod,
        "sumprod" / 2 = m::sumprod,

        // Integer math
        "factorial" / 1 = m::factorial,
        "gcd"   / * = m::gcd,
        "lcm"   / * = m::lcm,
        "comb"  / 2 = m::comb,
        "perm"  / * = m::perm,
        "isqrt" / 1 = m::isqrt,
    },
}
