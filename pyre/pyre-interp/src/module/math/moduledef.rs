//! Math module definition.
//!
//! PyPy equivalent: pypy/module/math/moduledef.py
//!
//! Maps Python-visible names to interp-level implementations,
//! matching PyPy's `interpleveldefs` dict.

use pyre_runtime::{PyNamespace, namespace_store, w_builtin_func_new};

use super::interp_math;

/// Install all math module names into the given namespace.
///
/// PyPy equivalent: Module(MixedModule) with interpleveldefs dict
/// in pypy/module/math/moduledef.py
pub fn init(ns: &mut PyNamespace) {
    // ── Constants ────────────────────────────────────────────────────
    // PyPy: interp_math.State.__init__ → w_e, w_pi
    namespace_store(
        ns,
        "e",
        pyre_object::floatobject::w_float_new(std::f64::consts::E),
    );
    namespace_store(
        ns,
        "pi",
        pyre_object::floatobject::w_float_new(std::f64::consts::PI),
    );
    namespace_store(
        ns,
        "tau",
        pyre_object::floatobject::w_float_new(std::f64::consts::TAU),
    );
    namespace_store(
        ns,
        "inf",
        pyre_object::floatobject::w_float_new(f64::INFINITY),
    );
    namespace_store(ns, "nan", pyre_object::floatobject::w_float_new(f64::NAN));

    // ── interpleveldefs ──────────────────────────────────────────────
    // PyPy: moduledef.py interpleveldefs = { 'sqrt': 'interp_math.sqrt', ... }
    namespace_store(ns, "sqrt", w_builtin_func_new("sqrt", interp_math::sqrt));
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
    namespace_store(ns, "exp", w_builtin_func_new("exp", interp_math::exp));
    namespace_store(ns, "expm1", w_builtin_func_new("expm1", interp_math::expm1));
    namespace_store(ns, "log", w_builtin_func_new("log", interp_math::log));
    namespace_store(ns, "log2", w_builtin_func_new("log2", interp_math::log2));
    namespace_store(ns, "log10", w_builtin_func_new("log10", interp_math::log10));
    namespace_store(ns, "log1p", w_builtin_func_new("log1p", interp_math::log1p));
    namespace_store(ns, "pow", w_builtin_func_new("pow", interp_math::pow));
    namespace_store(ns, "hypot", w_builtin_func_new("hypot", interp_math::hypot));
    namespace_store(ns, "fabs", w_builtin_func_new("fabs", interp_math::fabs));
    namespace_store(ns, "fmod", w_builtin_func_new("fmod", interp_math::fmod));
    namespace_store(ns, "floor", w_builtin_func_new("floor", interp_math::floor));
    namespace_store(ns, "ceil", w_builtin_func_new("ceil", interp_math::ceil));
    namespace_store(ns, "trunc", w_builtin_func_new("trunc", interp_math::trunc));
    namespace_store(
        ns,
        "copysign",
        w_builtin_func_new("copysign", interp_math::copysign),
    );
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
    namespace_store(ns, "isinf", w_builtin_func_new("isinf", interp_math::isinf));
    namespace_store(ns, "isnan", w_builtin_func_new("isnan", interp_math::isnan));
    namespace_store(
        ns,
        "isfinite",
        w_builtin_func_new("isfinite", interp_math::isfinite),
    );
    namespace_store(
        ns,
        "factorial",
        w_builtin_func_new("factorial", interp_math::factorial),
    );
    namespace_store(ns, "gcd", w_builtin_func_new("gcd", interp_math::gcd));
    namespace_store(ns, "erf", w_builtin_func_new("erf", interp_math::erf));
    namespace_store(ns, "erfc", w_builtin_func_new("erfc", interp_math::erfc));
    namespace_store(ns, "gamma", w_builtin_func_new("gamma", interp_math::gamma));
    namespace_store(
        ns,
        "lgamma",
        w_builtin_func_new("lgamma", interp_math::lgamma),
    );
}
