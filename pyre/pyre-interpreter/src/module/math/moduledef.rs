//! Math module definition — complete CPython parity via pymath crate.
//!
//! PyPy equivalent: pypy/module/math/moduledef.py

use super::interp_math;
use crate::{DictStorage, dict_storage_store, make_builtin_function};

pub fn init(ns: &mut DictStorage) {
    // ── Constants (PyPy: interp_math.State.__init__) ─────────────────
    dict_storage_store(
        ns,
        "e",
        pyre_object::floatobject::w_float_new(pymath::math::E),
    );
    dict_storage_store(
        ns,
        "pi",
        pyre_object::floatobject::w_float_new(pymath::math::PI),
    );
    dict_storage_store(
        ns,
        "tau",
        pyre_object::floatobject::w_float_new(pymath::math::TAU),
    );
    dict_storage_store(
        ns,
        "inf",
        pyre_object::floatobject::w_float_new(pymath::math::INF),
    );
    dict_storage_store(
        ns,
        "nan",
        pyre_object::floatobject::w_float_new(pymath::math::NAN),
    );

    // ── Trigonometric ────────────────────────────────────────────────
    dict_storage_store(ns, "sin", make_builtin_function("sin", interp_math::sin));
    dict_storage_store(ns, "cos", make_builtin_function("cos", interp_math::cos));
    dict_storage_store(ns, "tan", make_builtin_function("tan", interp_math::tan));
    dict_storage_store(ns, "asin", make_builtin_function("asin", interp_math::asin));
    dict_storage_store(ns, "acos", make_builtin_function("acos", interp_math::acos));
    dict_storage_store(ns, "atan", make_builtin_function("atan", interp_math::atan));
    dict_storage_store(
        ns,
        "atan2",
        make_builtin_function("atan2", interp_math::atan2),
    );
    dict_storage_store(ns, "sinh", make_builtin_function("sinh", interp_math::sinh));
    dict_storage_store(ns, "cosh", make_builtin_function("cosh", interp_math::cosh));
    dict_storage_store(ns, "tanh", make_builtin_function("tanh", interp_math::tanh));
    dict_storage_store(
        ns,
        "asinh",
        make_builtin_function("asinh", interp_math::asinh),
    );
    dict_storage_store(
        ns,
        "acosh",
        make_builtin_function("acosh", interp_math::acosh),
    );
    dict_storage_store(
        ns,
        "atanh",
        make_builtin_function("atanh", interp_math::atanh),
    );

    // ── Exponential / logarithmic ───────────────────────────────────
    dict_storage_store(ns, "sqrt", make_builtin_function("sqrt", interp_math::sqrt));
    dict_storage_store(ns, "cbrt", make_builtin_function("cbrt", interp_math::cbrt));
    dict_storage_store(ns, "exp", make_builtin_function("exp", interp_math::exp));
    dict_storage_store(ns, "exp2", make_builtin_function("exp2", interp_math::exp2));
    dict_storage_store(
        ns,
        "expm1",
        make_builtin_function("expm1", interp_math::expm1),
    );
    dict_storage_store(ns, "log", make_builtin_function("log", interp_math::log));
    dict_storage_store(ns, "log2", make_builtin_function("log2", interp_math::log2));
    dict_storage_store(
        ns,
        "log10",
        make_builtin_function("log10", interp_math::log10),
    );
    dict_storage_store(
        ns,
        "log1p",
        make_builtin_function("log1p", interp_math::log1p),
    );
    dict_storage_store(ns, "pow", make_builtin_function("pow", interp_math::pow));

    // ── Gamma / error ───────────────────────────────────────────────
    dict_storage_store(ns, "erf", make_builtin_function("erf", interp_math::erf));
    dict_storage_store(ns, "erfc", make_builtin_function("erfc", interp_math::erfc));
    dict_storage_store(
        ns,
        "gamma",
        make_builtin_function("gamma", interp_math::gamma),
    );
    dict_storage_store(
        ns,
        "lgamma",
        make_builtin_function("lgamma", interp_math::lgamma),
    );

    // ── Rounding / truncation ───────────────────────────────────────
    dict_storage_store(
        ns,
        "floor",
        make_builtin_function("floor", interp_math::floor),
    );
    dict_storage_store(ns, "ceil", make_builtin_function("ceil", interp_math::ceil));
    dict_storage_store(
        ns,
        "trunc",
        make_builtin_function("trunc", interp_math::trunc),
    );

    // ── Floating-point manipulation ─────────────────────────────────
    dict_storage_store(ns, "fabs", make_builtin_function("fabs", interp_math::fabs));
    dict_storage_store(ns, "fmod", make_builtin_function("fmod", interp_math::fmod));
    dict_storage_store(
        ns,
        "copysign",
        make_builtin_function("copysign", interp_math::copysign),
    );
    dict_storage_store(
        ns,
        "remainder",
        make_builtin_function("remainder", interp_math::remainder),
    );
    dict_storage_store(
        ns,
        "frexp",
        make_builtin_function("frexp", interp_math::frexp),
    );
    dict_storage_store(
        ns,
        "ldexp",
        make_builtin_function("ldexp", interp_math::ldexp),
    );
    dict_storage_store(ns, "modf", make_builtin_function("modf", interp_math::modf));
    dict_storage_store(
        ns,
        "nextafter",
        make_builtin_function("nextafter", interp_math::nextafter),
    );
    dict_storage_store(ns, "ulp", make_builtin_function("ulp", interp_math::ulp));
    dict_storage_store(ns, "fma", make_builtin_function("fma", interp_math::fma));

    // ── Classification ──────────────────────────────────────────────
    dict_storage_store(
        ns,
        "isinf",
        make_builtin_function("isinf", interp_math::isinf),
    );
    dict_storage_store(
        ns,
        "isnan",
        make_builtin_function("isnan", interp_math::isnan),
    );
    dict_storage_store(
        ns,
        "isfinite",
        make_builtin_function("isfinite", interp_math::isfinite),
    );
    dict_storage_store(
        ns,
        "isclose",
        make_builtin_function("isclose", interp_math::isclose),
    );

    // ── Conversion ──────────────────────────────────────────────────
    dict_storage_store(
        ns,
        "degrees",
        make_builtin_function("degrees", interp_math::degrees),
    );
    dict_storage_store(
        ns,
        "radians",
        make_builtin_function("radians", interp_math::radians),
    );

    // ── Multi-dimensional ───────────────────────────────────────────
    dict_storage_store(
        ns,
        "hypot",
        make_builtin_function("hypot", interp_math::hypot),
    );
    dict_storage_store(ns, "dist", make_builtin_function("dist", interp_math::dist));

    // ── Aggregation ─────────────────────────────────────────────────
    dict_storage_store(ns, "fsum", make_builtin_function("fsum", interp_math::fsum));
    dict_storage_store(ns, "prod", make_builtin_function("prod", interp_math::prod));
    dict_storage_store(
        ns,
        "sumprod",
        make_builtin_function("sumprod", interp_math::sumprod),
    );

    // ── Integer math ────────────────────────────────────────────────
    dict_storage_store(
        ns,
        "factorial",
        make_builtin_function("factorial", interp_math::factorial),
    );
    dict_storage_store(ns, "gcd", make_builtin_function("gcd", interp_math::gcd));
    dict_storage_store(ns, "lcm", make_builtin_function("lcm", interp_math::lcm));
    dict_storage_store(ns, "comb", make_builtin_function("comb", interp_math::comb));
    dict_storage_store(ns, "perm", make_builtin_function("perm", interp_math::perm));
    dict_storage_store(
        ns,
        "isqrt",
        make_builtin_function("isqrt", interp_math::isqrt),
    );
}
