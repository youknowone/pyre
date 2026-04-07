//! Math module definition — complete CPython parity via pymath crate.
//!
//! PyPy equivalent: pypy/module/math/moduledef.py

use super::interp_math;
use crate::{PyNamespace, make_builtin_function, namespace_store};

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
    namespace_store(ns, "sin", make_builtin_function("sin", interp_math::sin));
    namespace_store(ns, "cos", make_builtin_function("cos", interp_math::cos));
    namespace_store(ns, "tan", make_builtin_function("tan", interp_math::tan));
    namespace_store(ns, "asin", make_builtin_function("asin", interp_math::asin));
    namespace_store(ns, "acos", make_builtin_function("acos", interp_math::acos));
    namespace_store(ns, "atan", make_builtin_function("atan", interp_math::atan));
    namespace_store(
        ns,
        "atan2",
        make_builtin_function("atan2", interp_math::atan2),
    );
    namespace_store(ns, "sinh", make_builtin_function("sinh", interp_math::sinh));
    namespace_store(ns, "cosh", make_builtin_function("cosh", interp_math::cosh));
    namespace_store(ns, "tanh", make_builtin_function("tanh", interp_math::tanh));
    namespace_store(
        ns,
        "asinh",
        make_builtin_function("asinh", interp_math::asinh),
    );
    namespace_store(
        ns,
        "acosh",
        make_builtin_function("acosh", interp_math::acosh),
    );
    namespace_store(
        ns,
        "atanh",
        make_builtin_function("atanh", interp_math::atanh),
    );

    // ── Exponential / logarithmic ───────────────────────────────────
    namespace_store(ns, "sqrt", make_builtin_function("sqrt", interp_math::sqrt));
    namespace_store(ns, "cbrt", make_builtin_function("cbrt", interp_math::cbrt));
    namespace_store(ns, "exp", make_builtin_function("exp", interp_math::exp));
    namespace_store(ns, "exp2", make_builtin_function("exp2", interp_math::exp2));
    namespace_store(
        ns,
        "expm1",
        make_builtin_function("expm1", interp_math::expm1),
    );
    namespace_store(ns, "log", make_builtin_function("log", interp_math::log));
    namespace_store(ns, "log2", make_builtin_function("log2", interp_math::log2));
    namespace_store(
        ns,
        "log10",
        make_builtin_function("log10", interp_math::log10),
    );
    namespace_store(
        ns,
        "log1p",
        make_builtin_function("log1p", interp_math::log1p),
    );
    namespace_store(ns, "pow", make_builtin_function("pow", interp_math::pow));

    // ── Gamma / error ───────────────────────────────────────────────
    namespace_store(ns, "erf", make_builtin_function("erf", interp_math::erf));
    namespace_store(ns, "erfc", make_builtin_function("erfc", interp_math::erfc));
    namespace_store(
        ns,
        "gamma",
        make_builtin_function("gamma", interp_math::gamma),
    );
    namespace_store(
        ns,
        "lgamma",
        make_builtin_function("lgamma", interp_math::lgamma),
    );

    // ── Rounding / truncation ───────────────────────────────────────
    namespace_store(
        ns,
        "floor",
        make_builtin_function("floor", interp_math::floor),
    );
    namespace_store(ns, "ceil", make_builtin_function("ceil", interp_math::ceil));
    namespace_store(
        ns,
        "trunc",
        make_builtin_function("trunc", interp_math::trunc),
    );

    // ── Floating-point manipulation ─────────────────────────────────
    namespace_store(ns, "fabs", make_builtin_function("fabs", interp_math::fabs));
    namespace_store(ns, "fmod", make_builtin_function("fmod", interp_math::fmod));
    namespace_store(
        ns,
        "copysign",
        make_builtin_function("copysign", interp_math::copysign),
    );
    namespace_store(
        ns,
        "remainder",
        make_builtin_function("remainder", interp_math::remainder),
    );
    namespace_store(
        ns,
        "frexp",
        make_builtin_function("frexp", interp_math::frexp),
    );
    namespace_store(
        ns,
        "ldexp",
        make_builtin_function("ldexp", interp_math::ldexp),
    );
    namespace_store(ns, "modf", make_builtin_function("modf", interp_math::modf));
    namespace_store(
        ns,
        "nextafter",
        make_builtin_function("nextafter", interp_math::nextafter),
    );
    namespace_store(ns, "ulp", make_builtin_function("ulp", interp_math::ulp));
    namespace_store(ns, "fma", make_builtin_function("fma", interp_math::fma));

    // ── Classification ──────────────────────────────────────────────
    namespace_store(
        ns,
        "isinf",
        make_builtin_function("isinf", interp_math::isinf),
    );
    namespace_store(
        ns,
        "isnan",
        make_builtin_function("isnan", interp_math::isnan),
    );
    namespace_store(
        ns,
        "isfinite",
        make_builtin_function("isfinite", interp_math::isfinite),
    );
    namespace_store(
        ns,
        "isclose",
        make_builtin_function("isclose", interp_math::isclose),
    );

    // ── Conversion ──────────────────────────────────────────────────
    namespace_store(
        ns,
        "degrees",
        make_builtin_function("degrees", interp_math::degrees),
    );
    namespace_store(
        ns,
        "radians",
        make_builtin_function("radians", interp_math::radians),
    );

    // ── Multi-dimensional ───────────────────────────────────────────
    namespace_store(
        ns,
        "hypot",
        make_builtin_function("hypot", interp_math::hypot),
    );
    namespace_store(ns, "dist", make_builtin_function("dist", interp_math::dist));

    // ── Aggregation ─────────────────────────────────────────────────
    namespace_store(ns, "fsum", make_builtin_function("fsum", interp_math::fsum));
    namespace_store(ns, "prod", make_builtin_function("prod", interp_math::prod));

    // ── Integer math ────────────────────────────────────────────────
    namespace_store(
        ns,
        "factorial",
        make_builtin_function("factorial", interp_math::factorial),
    );
    namespace_store(ns, "gcd", make_builtin_function("gcd", interp_math::gcd));
    namespace_store(ns, "lcm", make_builtin_function("lcm", interp_math::lcm));
    namespace_store(ns, "comb", make_builtin_function("comb", interp_math::comb));
    namespace_store(ns, "perm", make_builtin_function("perm", interp_math::perm));
    namespace_store(
        ns,
        "isqrt",
        make_builtin_function("isqrt", interp_math::isqrt),
    );
}
