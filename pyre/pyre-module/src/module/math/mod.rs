//! math module — PyPy: pypy/module/math/
//!
//! Function bodies live in `interp_math`; this declarative table mirrors
//! `moduledef.py` interpleveldefs.

pub mod interp_math;

use interp_math as m;

pyre_interpreter::py_module! {
    "math",
    interpleveldefs: {
        "e"   => pyre_object::floatobject::w_float_new(pymath::math::E),
        "pi"  => pyre_object::floatobject::w_float_new(pymath::math::PI),
        "tau" => pyre_object::floatobject::w_float_new(pymath::math::TAU),
        "inf" => pyre_object::floatobject::w_float_new(pymath::math::INF),
        "nan" => pyre_object::floatobject::w_float_new(pymath::math::NAN),
    },
    module_functions: {
        // Trigonometric. The one- and two-argument float builtins are
        // installed in `extra_init` as `__majit_wrap_math_*`.

        // Exponential / logarithmic. sqrt/cbrt/log/log1p/exp/exp2/expm1/pow
        // are installed in `extra_init`.
        "log2"  / 1 = m::log2,
        "log10" / 1 = m::log10,

        // Gamma / error. erf/erfc/gamma/lgamma are installed in `extra_init`.

        // Rounding / truncation: floor/ceil/trunc are installed in `extra_init`.

        // Floating-point manipulation. `fabs`, `ulp`, `frexp`, `ldexp`,
        // `fmod`, `copysign` and `remainder` are installed in `extra_init`.
        "modf"      / 1 = m::modf,
        "nextafter" / * = m::nextafter,
        "fma"       / 3 = m::fma,

        // Classification. `isclose` is installed in `extra_init`.
        "isinf"    / 1 = m::isinf,
        "isnan"    / 1 = m::isnan,
        "isfinite" / 1 = m::isfinite,

        // Conversion. `degrees` and `radians` are installed in `extra_init`.

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
        // `isqrt` is installed in `extra_init`.
    },
    extra_init: |ns| {
        // Module builtin, not a method descriptor: `BuiltinCode.func` is the
        // gateway itself so builtin-call descent walks it.
        let install = |name: &'static str, func: pyre_interpreter::BuiltinCodeFn| {
            pyre_interpreter::module_ns_store(
                ns,
                name,
                pyre_interpreter::make_module_builtin_function_with_arity(name, func, 1),
            );
        };
        install("sqrt", m::__majit_wrap_math_sqrt);
        install("sin", m::__majit_wrap_math_sin);
        install("cos", m::__majit_wrap_math_cos);
        install("tan", m::__majit_wrap_math_tan);
        install("asin", m::__majit_wrap_math_asin);
        install("acos", m::__majit_wrap_math_acos);
        install("atan", m::__majit_wrap_math_atan);
        install("tanh", m::__majit_wrap_math_tanh);
        install("asinh", m::__majit_wrap_math_asinh);
        install("acosh", m::__majit_wrap_math_acosh);
        install("atanh", m::__majit_wrap_math_atanh);
        install("log1p", m::__majit_wrap_math_log1p);
        install("cbrt", m::__majit_wrap_math_cbrt);
        install("erf", m::__majit_wrap_math_erf);
        install("erfc", m::__majit_wrap_math_erfc);
        install("ulp", m::__majit_wrap_math_ulp);
        install("degrees", m::__majit_wrap_math_degrees);
        install("radians", m::__majit_wrap_math_radians);
        install("fabs", m::__majit_wrap_math_fabs);
        install("exp", m::__majit_wrap_math_exp);
        install("exp2", m::__majit_wrap_math_exp2);
        install("expm1", m::__majit_wrap_math_expm1);
        install("sinh", m::__majit_wrap_math_sinh);
        install("cosh", m::__majit_wrap_math_cosh);
        install("gamma", m::__majit_wrap_math_gamma);
        install("lgamma", m::__majit_wrap_math_lgamma);
        install("floor", m::__majit_wrap_math_floor);
        install("ceil", m::__majit_wrap_math_ceil);
        install("trunc", m::__majit_wrap_math_trunc);
        // `frexp` stays on the walker fold `math_frexp`: its gateway would
        // root the mantissa box across the exponent box, and a root bracket
        // does not lower in a walked body.
        install("frexp", m::frexp);
        install("isqrt", m::__majit_wrap_math_isqrt);
        // Optional base, or keyword tolerances: not a fixed arity-1 builtin.
        pyre_interpreter::module_ns_store(
            ns,
            "log",
            pyre_interpreter::make_module_builtin_function("log", m::__majit_wrap_math_log),
        );
        pyre_interpreter::module_ns_store(
            ns,
            "isclose",
            pyre_interpreter::make_module_builtin_function("isclose", m::__majit_wrap_math_isclose),
        );
        let install2 = |name: &'static str, func: pyre_interpreter::BuiltinCodeFn| {
            pyre_interpreter::module_ns_store(
                ns,
                name,
                pyre_interpreter::make_module_builtin_function_with_arity(name, func, 2),
            );
        };
        install2("pow", m::__majit_wrap_math_pow);
        install2("fmod", m::__majit_wrap_math_fmod);
        install2("copysign", m::__majit_wrap_math_copysign);
        install2("remainder", m::__majit_wrap_math_remainder);
        install2("atan2", m::__majit_wrap_math_atan2);
        install2("ldexp", m::__majit_wrap_math_ldexp);
    },
}
