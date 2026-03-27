//! Math module definition — complete CPython parity via pymath crate.
//!
//! PyPy equivalent: pypy/module/math/moduledef.py

use super::interp_math;
use crate::{PyNamespace, builtin_code_new, namespace_store};

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
    namespace_store(ns, "sin", builtin_code_new("sin", interp_math::sin));
    namespace_store(ns, "cos", builtin_code_new("cos", interp_math::cos));
    namespace_store(ns, "tan", builtin_code_new("tan", interp_math::tan));
    namespace_store(ns, "asin", builtin_code_new("asin", interp_math::asin));
    namespace_store(ns, "acos", builtin_code_new("acos", interp_math::acos));
    namespace_store(ns, "atan", builtin_code_new("atan", interp_math::atan));
    namespace_store(ns, "atan2", builtin_code_new("atan2", interp_math::atan2));
    namespace_store(ns, "sinh", builtin_code_new("sinh", interp_math::sinh));
    namespace_store(ns, "cosh", builtin_code_new("cosh", interp_math::cosh));
    namespace_store(ns, "tanh", builtin_code_new("tanh", interp_math::tanh));
    namespace_store(ns, "asinh", builtin_code_new("asinh", interp_math::asinh));
    namespace_store(ns, "acosh", builtin_code_new("acosh", interp_math::acosh));
    namespace_store(ns, "atanh", builtin_code_new("atanh", interp_math::atanh));

    // ── Exponential / logarithmic ───────────────────────────────────
    namespace_store(ns, "sqrt", builtin_code_new("sqrt", interp_math::sqrt));
    namespace_store(ns, "cbrt", builtin_code_new("cbrt", interp_math::cbrt));
    namespace_store(ns, "exp", builtin_code_new("exp", interp_math::exp));
    namespace_store(ns, "exp2", builtin_code_new("exp2", interp_math::exp2));
    namespace_store(ns, "expm1", builtin_code_new("expm1", interp_math::expm1));
    namespace_store(ns, "log", builtin_code_new("log", interp_math::log));
    namespace_store(ns, "log2", builtin_code_new("log2", interp_math::log2));
    namespace_store(ns, "log10", builtin_code_new("log10", interp_math::log10));
    namespace_store(ns, "log1p", builtin_code_new("log1p", interp_math::log1p));
    namespace_store(ns, "pow", builtin_code_new("pow", interp_math::pow));

    // ── Gamma / error ───────────────────────────────────────────────
    namespace_store(ns, "erf", builtin_code_new("erf", interp_math::erf));
    namespace_store(ns, "erfc", builtin_code_new("erfc", interp_math::erfc));
    namespace_store(ns, "gamma", builtin_code_new("gamma", interp_math::gamma));
    namespace_store(
        ns,
        "lgamma",
        builtin_code_new("lgamma", interp_math::lgamma),
    );

    // ── Rounding / truncation ───────────────────────────────────────
    namespace_store(ns, "floor", builtin_code_new("floor", interp_math::floor));
    namespace_store(ns, "ceil", builtin_code_new("ceil", interp_math::ceil));
    namespace_store(ns, "trunc", builtin_code_new("trunc", interp_math::trunc));

    // ── Floating-point manipulation ─────────────────────────────────
    namespace_store(ns, "fabs", builtin_code_new("fabs", interp_math::fabs));
    namespace_store(ns, "fmod", builtin_code_new("fmod", interp_math::fmod));
    namespace_store(
        ns,
        "copysign",
        builtin_code_new("copysign", interp_math::copysign),
    );
    namespace_store(
        ns,
        "remainder",
        builtin_code_new("remainder", interp_math::remainder),
    );
    namespace_store(ns, "frexp", builtin_code_new("frexp", interp_math::frexp));
    namespace_store(ns, "ldexp", builtin_code_new("ldexp", interp_math::ldexp));
    namespace_store(ns, "modf", builtin_code_new("modf", interp_math::modf));
    namespace_store(
        ns,
        "nextafter",
        builtin_code_new("nextafter", interp_math::nextafter),
    );
    namespace_store(ns, "ulp", builtin_code_new("ulp", interp_math::ulp));
    namespace_store(ns, "fma", builtin_code_new("fma", interp_math::fma));

    // ── Classification ──────────────────────────────────────────────
    namespace_store(ns, "isinf", builtin_code_new("isinf", interp_math::isinf));
    namespace_store(ns, "isnan", builtin_code_new("isnan", interp_math::isnan));
    namespace_store(
        ns,
        "isfinite",
        builtin_code_new("isfinite", interp_math::isfinite),
    );
    namespace_store(
        ns,
        "isclose",
        builtin_code_new("isclose", interp_math::isclose),
    );

    // ── Conversion ──────────────────────────────────────────────────
    namespace_store(
        ns,
        "degrees",
        builtin_code_new("degrees", interp_math::degrees),
    );
    namespace_store(
        ns,
        "radians",
        builtin_code_new("radians", interp_math::radians),
    );

    // ── Multi-dimensional ───────────────────────────────────────────
    namespace_store(ns, "hypot", builtin_code_new("hypot", interp_math::hypot));
    namespace_store(ns, "dist", builtin_code_new("dist", interp_math::dist));

    // ── Aggregation ─────────────────────────────────────────────────
    namespace_store(ns, "fsum", builtin_code_new("fsum", interp_math::fsum));
    namespace_store(ns, "prod", builtin_code_new("prod", interp_math::prod));

    // ── Integer math ────────────────────────────────────────────────
    namespace_store(
        ns,
        "factorial",
        builtin_code_new("factorial", interp_math::factorial),
    );
    namespace_store(ns, "gcd", builtin_code_new("gcd", interp_math::gcd));
    namespace_store(ns, "lcm", builtin_code_new("lcm", interp_math::lcm));
    namespace_store(ns, "comb", builtin_code_new("comb", interp_math::comb));
    namespace_store(ns, "perm", builtin_code_new("perm", interp_math::perm));
    namespace_store(ns, "isqrt", builtin_code_new("isqrt", interp_math::isqrt));
}
