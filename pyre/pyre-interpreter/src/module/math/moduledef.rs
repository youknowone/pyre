//! Math module definition — complete CPython parity via pymath crate.
//!
//! PyPy equivalent: pypy/module/math/moduledef.py

use super::interp_math;
use crate::{PyNamespace, namespace_store, w_builtin_func_new};

pub fn init(ns: &mut PyNamespace) {
    // ── Constants (PyPy: interp_math.State.__init__) ─────────────────
    namespace_store(
        ns,
        "e",
        pyre_object::floatobject::w_float_new(pymath::math::E),
    );
    namespace_store(
        ns,
        "pi",
        pyre_object::floatobject::w_float_new(pymath::math::PI),
    );
    namespace_store(
        ns,
        "tau",
        pyre_object::floatobject::w_float_new(pymath::math::TAU),
    );
    namespace_store(
        ns,
        "inf",
        pyre_object::floatobject::w_float_new(pymath::math::INF),
    );
    namespace_store(
        ns,
        "nan",
        pyre_object::floatobject::w_float_new(pymath::math::NAN),
    );

    // ── Trigonometric ────────────────────────────────────────────────
    namespace_store(ns, "sin", w_builtin_func_new("sin", interp_math::sin));
    namespace_store(ns, "cos", w_builtin_func_new("cos", interp_math::cos));
    namespace_store(ns, "tan", w_builtin_func_new("tan", interp_math::tan));
    namespace_store(ns, "asin", w_builtin_func_new("asin", interp_math::asin));
    namespace_store(ns, "acos", w_builtin_func_new("acos", interp_math::acos));
    namespace_store(ns, "atan", w_builtin_func_new("atan", interp_math::atan));
    namespace_store(ns, "atan2", w_builtin_func_new("atan2", interp_math::atan2));
    namespace_store(ns, "sinh", w_builtin_func_new("sinh", interp_math::sinh));
    namespace_store(ns, "cosh", w_builtin_func_new("cosh", interp_math::cosh));
    namespace_store(ns, "tanh", w_builtin_func_new("tanh", interp_math::tanh));
    namespace_store(ns, "asinh", w_builtin_func_new("asinh", interp_math::asinh));
    namespace_store(ns, "acosh", w_builtin_func_new("acosh", interp_math::acosh));
    namespace_store(ns, "atanh", w_builtin_func_new("atanh", interp_math::atanh));

    // ── Exponential / logarithmic ───────────────────────────────────
    namespace_store(ns, "sqrt", w_builtin_func_new("sqrt", interp_math::sqrt));
    namespace_store(ns, "cbrt", w_builtin_func_new("cbrt", interp_math::cbrt));
    namespace_store(ns, "exp", w_builtin_func_new("exp", interp_math::exp));
    namespace_store(ns, "exp2", w_builtin_func_new("exp2", interp_math::exp2));
    namespace_store(ns, "expm1", w_builtin_func_new("expm1", interp_math::expm1));
    namespace_store(ns, "log", w_builtin_func_new("log", interp_math::log));
    namespace_store(ns, "log2", w_builtin_func_new("log2", interp_math::log2));
    namespace_store(ns, "log10", w_builtin_func_new("log10", interp_math::log10));
    namespace_store(ns, "log1p", w_builtin_func_new("log1p", interp_math::log1p));
    namespace_store(ns, "pow", w_builtin_func_new("pow", interp_math::pow));

    // ── Gamma / error ───────────────────────────────────────────────
    namespace_store(ns, "erf", w_builtin_func_new("erf", interp_math::erf));
    namespace_store(ns, "erfc", w_builtin_func_new("erfc", interp_math::erfc));
    namespace_store(ns, "gamma", w_builtin_func_new("gamma", interp_math::gamma));
    namespace_store(
        ns,
        "lgamma",
        w_builtin_func_new("lgamma", interp_math::lgamma),
    );

    // ── Rounding / truncation ───────────────────────────────────────
    namespace_store(ns, "floor", w_builtin_func_new("floor", interp_math::floor));
    namespace_store(ns, "ceil", w_builtin_func_new("ceil", interp_math::ceil));
    namespace_store(ns, "trunc", w_builtin_func_new("trunc", interp_math::trunc));

    // ── Floating-point manipulation ─────────────────────────────────
    namespace_store(ns, "fabs", w_builtin_func_new("fabs", interp_math::fabs));
    namespace_store(ns, "fmod", w_builtin_func_new("fmod", interp_math::fmod));
    namespace_store(
        ns,
        "copysign",
        w_builtin_func_new("copysign", interp_math::copysign),
    );
    namespace_store(
        ns,
        "remainder",
        w_builtin_func_new("remainder", interp_math::remainder),
    );
    namespace_store(ns, "frexp", w_builtin_func_new("frexp", interp_math::frexp));
    namespace_store(ns, "ldexp", w_builtin_func_new("ldexp", interp_math::ldexp));
    namespace_store(ns, "modf", w_builtin_func_new("modf", interp_math::modf));
    namespace_store(
        ns,
        "nextafter",
        w_builtin_func_new("nextafter", interp_math::nextafter),
    );
    namespace_store(ns, "ulp", w_builtin_func_new("ulp", interp_math::ulp));
    namespace_store(ns, "fma", w_builtin_func_new("fma", interp_math::fma));

    // ── Classification ──────────────────────────────────────────────
    namespace_store(ns, "isinf", w_builtin_func_new("isinf", interp_math::isinf));
    namespace_store(ns, "isnan", w_builtin_func_new("isnan", interp_math::isnan));
    namespace_store(
        ns,
        "isfinite",
        w_builtin_func_new("isfinite", interp_math::isfinite),
    );
    namespace_store(
        ns,
        "isclose",
        w_builtin_func_new("isclose", interp_math::isclose),
    );

    // ── Conversion ──────────────────────────────────────────────────
    namespace_store(
        ns,
        "degrees",
        w_builtin_func_new("degrees", interp_math::degrees),
    );
    namespace_store(
        ns,
        "radians",
        w_builtin_func_new("radians", interp_math::radians),
    );

    // ── Multi-dimensional ───────────────────────────────────────────
    namespace_store(ns, "hypot", w_builtin_func_new("hypot", interp_math::hypot));
    namespace_store(ns, "dist", w_builtin_func_new("dist", interp_math::dist));

    // ── Aggregation ─────────────────────────────────────────────────
    namespace_store(ns, "fsum", w_builtin_func_new("fsum", interp_math::fsum));
    namespace_store(ns, "prod", w_builtin_func_new("prod", interp_math::prod));

    // ── Integer math ────────────────────────────────────────────────
    namespace_store(
        ns,
        "factorial",
        w_builtin_func_new("factorial", interp_math::factorial),
    );
    namespace_store(ns, "gcd", w_builtin_func_new("gcd", interp_math::gcd));
    namespace_store(ns, "lcm", w_builtin_func_new("lcm", interp_math::lcm));
    namespace_store(ns, "comb", w_builtin_func_new("comb", interp_math::comb));
    namespace_store(ns, "perm", w_builtin_func_new("perm", interp_math::perm));
    namespace_store(ns, "isqrt", w_builtin_func_new("isqrt", interp_math::isqrt));
}
