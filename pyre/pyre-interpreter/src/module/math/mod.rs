//! math module — PyPy: pypy/module/math/
//!
//! Function bodies live in `interp_math`; this declarative table mirrors
//! `moduledef.py` interpleveldefs.

pub mod interp_math;

use interp_math as m;

crate::py_module! {
    "math",
    interpleveldefs: {
        // Constants
        "e" => pyre_object::floatobject::w_float_new(pymath::math::E),
        "pi" => pyre_object::floatobject::w_float_new(pymath::math::PI),
        "tau" => pyre_object::floatobject::w_float_new(pymath::math::TAU),
        "inf" => pyre_object::floatobject::w_float_new(pymath::math::INF),
        "nan" => pyre_object::floatobject::w_float_new(pymath::math::NAN),

        // Trigonometric
        "sin"   => crate::make_builtin_function_with_arity("sin",   m::sin,   1),
        "cos"   => crate::make_builtin_function_with_arity("cos",   m::cos,   1),
        "tan"   => crate::make_builtin_function_with_arity("tan",   m::tan,   1),
        "asin"  => crate::make_builtin_function_with_arity("asin",  m::asin,  1),
        "acos"  => crate::make_builtin_function_with_arity("acos",  m::acos,  1),
        "atan"  => crate::make_builtin_function_with_arity("atan",  m::atan,  1),
        "atan2" => crate::make_builtin_function_with_arity("atan2", m::atan2, 2),
        "sinh"  => crate::make_builtin_function_with_arity("sinh",  m::sinh,  1),
        "cosh"  => crate::make_builtin_function_with_arity("cosh",  m::cosh,  1),
        "tanh"  => crate::make_builtin_function_with_arity("tanh",  m::tanh,  1),
        "asinh" => crate::make_builtin_function_with_arity("asinh", m::asinh, 1),
        "acosh" => crate::make_builtin_function_with_arity("acosh", m::acosh, 1),
        "atanh" => crate::make_builtin_function_with_arity("atanh", m::atanh, 1),

        // Exponential / logarithmic
        "sqrt"  => crate::make_builtin_function_with_arity("sqrt",  m::sqrt,  1),
        "cbrt"  => crate::make_builtin_function_with_arity("cbrt",  m::cbrt,  1),
        "exp"   => crate::make_builtin_function_with_arity("exp",   m::exp,   1),
        "exp2"  => crate::make_builtin_function_with_arity("exp2",  m::exp2,  1),
        "expm1" => crate::make_builtin_function_with_arity("expm1", m::expm1, 1),
        "log"   => crate::make_builtin_function("log", m::log),
        "log2"  => crate::make_builtin_function_with_arity("log2",  m::log2,  1),
        "log10" => crate::make_builtin_function_with_arity("log10", m::log10, 1),
        "log1p" => crate::make_builtin_function_with_arity("log1p", m::log1p, 1),
        "pow"   => crate::make_builtin_function_with_arity("pow",   m::pow,   2),

        // Gamma / error
        "erf"    => crate::make_builtin_function_with_arity("erf",    m::erf,    1),
        "erfc"   => crate::make_builtin_function_with_arity("erfc",   m::erfc,   1),
        "gamma"  => crate::make_builtin_function_with_arity("gamma",  m::gamma,  1),
        "lgamma" => crate::make_builtin_function_with_arity("lgamma", m::lgamma, 1),

        // Rounding / truncation
        "floor" => crate::make_builtin_function_with_arity("floor", m::floor, 1),
        "ceil"  => crate::make_builtin_function_with_arity("ceil",  m::ceil,  1),
        "trunc" => crate::make_builtin_function_with_arity("trunc", m::trunc, 1),

        // Floating-point manipulation
        "fabs"      => crate::make_builtin_function_with_arity("fabs",      m::fabs,      1),
        "fmod"      => crate::make_builtin_function_with_arity("fmod",      m::fmod,      2),
        "copysign"  => crate::make_builtin_function_with_arity("copysign",  m::copysign,  2),
        "remainder" => crate::make_builtin_function_with_arity("remainder", m::remainder, 2),
        "frexp"     => crate::make_builtin_function_with_arity("frexp",     m::frexp,     1),
        "ldexp"     => crate::make_builtin_function_with_arity("ldexp",     m::ldexp,     2),
        "modf"      => crate::make_builtin_function_with_arity("modf",      m::modf,      1),
        "nextafter" => crate::make_builtin_function("nextafter", m::nextafter),
        "ulp"       => crate::make_builtin_function_with_arity("ulp",       m::ulp,       1),
        "fma"       => crate::make_builtin_function_with_arity("fma",       m::fma,       3),

        // Classification
        "isinf"    => crate::make_builtin_function_with_arity("isinf",    m::isinf,    1),
        "isnan"    => crate::make_builtin_function_with_arity("isnan",    m::isnan,    1),
        "isfinite" => crate::make_builtin_function_with_arity("isfinite", m::isfinite, 1),
        "isclose"  => crate::make_builtin_function("isclose", m::isclose),

        // Conversion
        "degrees" => crate::make_builtin_function_with_arity("degrees", m::degrees, 1),
        "radians" => crate::make_builtin_function_with_arity("radians", m::radians, 1),

        // Multi-dimensional
        "hypot" => crate::make_builtin_function("hypot", m::hypot),
        "dist"  => crate::make_builtin_function_with_arity("dist", m::dist, 2),

        // Aggregation
        "fsum"    => crate::make_builtin_function_with_arity("fsum",    m::fsum,    1),
        "prod"    => crate::make_builtin_function("prod", m::prod),
        "sumprod" => crate::make_builtin_function_with_arity("sumprod", m::sumprod, 2),

        // Integer math
        "factorial" => crate::make_builtin_function_with_arity("factorial", m::factorial, 1),
        "gcd"   => crate::make_builtin_function("gcd", m::gcd),
        "lcm"   => crate::make_builtin_function("lcm", m::lcm),
        "comb"  => crate::make_builtin_function_with_arity("comb", m::comb, 2),
        "perm"  => crate::make_builtin_function("perm", m::perm),
        "isqrt" => crate::make_builtin_function_with_arity("isqrt", m::isqrt, 1),
    }
}
