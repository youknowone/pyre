//! Math function implementations — delegating to pymath crate.
//!
//! PyPy equivalent: pypy/module/math/interp_math.py
//!
//! All functions delegate to `pymath::math` for CPython-exact results.

use majit_rlib::rbigint::{RBigInt as BigInt, RBigIntError, RBigIntGcRoot};
use pyre_object::*;

/// Infallible f64 extraction with a `0.0` fallback for a non-convertible
/// argument. Retained for `cmath`, whose flat gateway does not thread the
/// error path; the `math` module uses [`try_get_double`] directly so a
/// non-number raises `TypeError`.
pub fn get_double(obj: PyObjectRef) -> f64 {
    try_get_double(obj).unwrap_or(0.0)
}

/// PyPy: `_get_double` — raises TypeError when the argument has no
/// numeric interpretation (no int/float/bool/long layout and no
/// __float__/__index__ method). mathmodule.c's entry points use this
/// to reject `math.exp("spam")` etc.
pub fn try_get_double(obj: PyObjectRef) -> Result<f64, pyre_interpreter::PyError> {
    unsafe {
        if is_float(obj) {
            return Ok(floatobject::w_float_get_value(obj));
        }
        // Reading the payload answers only for an operand whose Python class
        // is the builtin: `is_int` / `is_long` / `is_bool` read `ob_type`,
        // which a strict subclass shares while overriding `__float__`.  The
        // subclass falls to the ladder below; an inherited `int.__float__`
        // reproduces the payload.  The `float` arm stays ungated because the
        // conversion short-circuits on the layout and ignores an override.
        if pyre_object::is_exact_builtin_instance(obj)
            && let Some(value) = pyre_interpreter::builtins::int_payload_as_f64(obj)
        {
            return value;
        }
    }
    // `__float__` is a type-only special-method lookup (`space.lookup`); an
    // instance attribute named `__float__` is not consulted. A raising
    // `__float__` (descriptor `__get__` or the call itself) propagates
    // instead of being reported as "must be real number".
    match unsafe { pyre_interpreter::baseobjspace::lookup_special(obj, "__float__") } {
        Ok(Some(method)) => {
            let result = pyre_interpreter::builtins::call_and_check(method, &[])?;
            unsafe {
                if is_float(result) {
                    // A strict `float` subclass is accepted but deprecated;
                    // an exact `float` is used as-is.
                    if !is_exact_type(result, &FLOAT_TYPE) {
                        let value_type = pyre_interpreter::type_methods::arg_type_name(obj);
                        let result_type = pyre_interpreter::type_methods::arg_type_name(result);
                        pyre_interpreter::warn::warn_deprecation(&format!(
                            "{value_type}.__float__ returned non-float (type {result_type}).  \
                             The ability to return an instance of a strict subclass of \
                             float is deprecated, and may be removed in a future version \
                             of Python."
                        ))?;
                    }
                    return Ok(floatobject::w_float_get_value(result));
                }
            }
            // descroperation.py:891 — a non-float result (including int/long)
            // is rejected rather than coerced.
            let result_type = unsafe { pyre_object::type_name_of(result) };
            return Err(pyre_interpreter::PyError::type_error(format!(
                "__float__ returned non-float (type '{result_type}')",
            )));
        }
        Ok(None) => {}
        Err(err) => return Err(err),
    }
    match pyre_interpreter::baseobjspace::getattr_str(obj, "__index__") {
        Ok(method) => {
            let result = pyre_interpreter::builtins::call_and_check(method, &[])?;
            unsafe {
                if is_int(result) {
                    return Ok(w_int_get_value(result) as f64);
                }
                if is_long(result) {
                    let v = jit_bigint_to_f64_or_nan(w_long_get_value(result));
                    if !v.is_finite() {
                        return Err(pyre_interpreter::PyError::overflow_error(
                            "int too large to convert to float",
                        ));
                    }
                    return Ok(v);
                }
            }
        }
        Err(err) if err.kind != pyre_interpreter::PyErrorKind::AttributeError => return Err(err),
        Err(_) => {}
    }
    Err(pyre_interpreter::PyError::type_error(format!(
        "must be real number, not {}",
        pyre_interpreter::type_methods::arg_type_name(obj)
    )))
}

type PyResult = Result<PyObjectRef, pyre_interpreter::PyError>;

fn map_err(r: pymath::Result<f64>) -> PyResult {
    match r {
        Ok(v) => Ok(floatobject::w_float_new(v)),
        Err(pymath::Error::EDOM) => {
            Err(pyre_interpreter::PyError::value_error("math domain error"))
        }
        Err(pymath::Error::ERANGE) => Err(pyre_interpreter::PyError::overflow_error(
            "math range error",
        )),
    }
}

fn map_int_err(e: pymath::Error) -> pyre_interpreter::PyError {
    match e {
        pymath::Error::EDOM => pyre_interpreter::PyError::value_error("math domain error"),
        pymath::Error::ERANGE => pyre_interpreter::PyError::overflow_error("math range error"),
    }
}

fn map_rbigint_err(e: RBigIntError) -> pyre_interpreter::PyError {
    match e {
        RBigIntError::Memory => pyre_interpreter::PyError::memory_error(""),
        RBigIntError::Overflow | RBigIntError::FloatDivisionOverflow => {
            pyre_interpreter::PyError::overflow_error("math range error")
        }
        RBigIntError::DivisionByZero => {
            pyre_interpreter::PyError::zero_division("integer division by zero")
        }
        _ => pyre_interpreter::PyError::value_error("math domain error"),
    }
}

/// `float.__repr__` of a finite value, used to embed the offending operand
/// in a domain-error message.
fn float_repr(val: f64) -> String {
    if val.is_nan() {
        "nan".to_owned()
    } else if val.is_infinite() {
        if val.is_sign_positive() {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        }
    } else {
        pyre_interpreter::display::format_float_repr(val)
    }
}

// ── 1-arg float→float via pymath ─────────────────────────────────────

/// Domain pin via pymath; box the pymath success value.  The walker
/// still descends the unboxed leaf, which recomputes the operation so
/// the generated jitcode has a body.  crates.io pymath cannot be walked
/// (it is not in the Charon artefact).
fn math1_pymath(
    name: &str,
    args: &[PyObjectRef],
    compute: fn(f64) -> Result<f64, pymath::Error>,
    edom: fn(f64) -> String,
) -> PyResult {
    if args.len() != 1 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{name}() takes exactly one argument"
        )));
    }
    let val = try_get_double(args[0])?;
    match compute(val) {
        Ok(v) => pyre_interpreter::objspace::descroperation::_float_pos(v),
        Err(pymath::Error::EDOM) => Err(pyre_interpreter::PyError::value_error(edom(val))),
        Err(pymath::Error::ERANGE) => Err(pyre_interpreter::PyError::overflow_error(
            "math range error",
        )),
    }
}

// Trigonometric
pub fn sin(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("sin", args, pymath::math::sin, |v| {
        format!("expected a finite input, got {}", float_repr(v))
    })
}

pub fn cos(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("cos", args, pymath::math::cos, |v| {
        format!("expected a finite input, got {}", float_repr(v))
    })
}
pub fn tan(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("tan", args, pymath::math::tan, |v| {
        format!("expected a finite input, got {}", float_repr(v))
    })
}
pub fn asin(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("asin", args, pymath::math::asin, |v| {
        format!(
            "expected a number in range from -1 up to 1, got {}",
            float_repr(v)
        )
    })
}
pub fn acos(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("acos", args, pymath::math::acos, |v| {
        format!(
            "expected a number in range from -1 up to 1, got {}",
            float_repr(v)
        )
    })
}
pub fn atan(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("atan", args, pymath::math::atan, |_| {
        "math domain error".to_string()
    })
}
pub fn sinh(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("sinh", args, pymath::math::sinh, |_| {
        "math domain error".to_string()
    })
}
pub fn cosh(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("cosh", args, pymath::math::cosh, |_| {
        "math domain error".to_string()
    })
}
pub fn tanh(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("tanh", args, pymath::math::tanh, |_| {
        "math domain error".to_string()
    })
}
pub fn asinh(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("asinh", args, pymath::math::asinh, |_| {
        "math domain error".to_string()
    })
}
pub fn acosh(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("acosh", args, pymath::math::acosh, |v| {
        format!(
            "expected argument value not less than 1, got {}",
            float_repr(v)
        )
    })
}
pub fn atanh(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("atanh", args, pymath::math::atanh, |v| {
        format!("expected a number between -1 and 1, got {}", float_repr(v))
    })
}

// Exponential / logarithmic
pub fn sqrt(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("sqrt", args, pymath::math::sqrt, |v| {
        format!("expected a nonnegative input, got {}", float_repr(v))
    })
}

/// Arity / keyword / domain residual of `sqrt`. The positional-count check is
/// the one `py_checked_arity_fn` used to run before this body.
#[majit_macros::dont_look_inside]
fn sqrt_slow(args: &[PyObjectRef]) -> PyResult {
    pyre_interpreter::gateway::check_declared_positional_arity("sqrt", 1, args)?;
    sqrt(args)
}

/// interp_math.py `sqrt` = `math1(space, math.sqrt, w_x)`; ll_math.py `ll_math_sqrt`.
pub fn __majit_wrap_math_sqrt(args: &[PyObjectRef]) -> PyResult {
    if args.len() == 1 {
        let w_x = args[0];
        let x =
            if unsafe { pyre_object::is_exact_builtin_instance(w_x) && pyre_object::is_float(w_x) }
            {
                Some(unsafe { pyre_object::w_float_get_value(w_x) })
            } else if unsafe {
                pyre_object::is_exact_builtin_instance(w_x)
                    && (pyre_object::is_int(w_x) || pyre_object::is_bool(w_x))
            } {
                // `_get_double` coerces via `space.float`; a machine int's `i64 as f64`
                // is that value. NaN cannot appear. A non-finite result is ±inf.
                Some(unsafe { pyre_object::w_int_get_value(w_x) } as f64)
            } else {
                None
            };
        if let Some(x) = x {
            // ll_math.py `ll_math_sqrt`: `x < 0.0` raises; NaN fails `x >= 0.0`.
            if x >= 0.0 {
                if x.is_finite() {
                    return pyre_interpreter::objspace::descroperation::_float_sqrt(x);
                }
                return Ok(pyre_object::floatobject::w_float_new(x));
            }
        }
    }
    sqrt_slow(args)
}

#[cfg(not(target_arch = "wasm32"))]
#[linkme::distributed_slice(pyre_interpreter::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
#[allow(non_upper_case_globals)]
static __majit_builtin_wrapper_target_math_sqrt:
    pyre_interpreter::gateway::BuiltinWrapperDescriptor =
    pyre_interpreter::gateway::BuiltinWrapperDescriptor {
        path: concat!(module_path!(), "::", stringify!(__majit_wrap_math_sqrt)),
        func: __majit_wrap_math_sqrt,
    };

/// One jitcode per builtin: the domain test is expanded into the function,
/// and the leaf is a direct call. `$domain` binds the operand with `|x|`.
macro_rules! majit_math1_gateway {
    ($name:ident, $leaf:path, total) => {
        majit_math1_gateway!(@emit $name, x, { return $leaf(x); });
    };
    ($name:ident, $leaf:path, |$x:ident| $domain:expr) => {
        majit_math1_gateway!(@emit $name, $x, {
            if $domain {
                return $leaf($x);
            }
        });
    };
    (@emit $name:ident, $x:ident, $fast:stmt) => {
        ::paste::paste! {
            #[majit_macros::dont_look_inside]
            fn [<$name _slow>](args: &[PyObjectRef]) -> PyResult {
                pyre_interpreter::gateway::check_declared_positional_arity(
                    stringify!($name),
                    1,
                    args,
                )?;
                $name(args)
            }

            /// interp_math.py `math1`: exact float, or a machine int/bool read
            /// as `f64`. Outside the domain the original body runs.
            pub fn [<__majit_wrap_math_ $name>](args: &[PyObjectRef]) -> PyResult {
                if args.len() == 1 {
                    let w_x = args[0];
                    let $x = if unsafe {
                        pyre_object::is_exact_builtin_instance(w_x)
                            && pyre_object::is_float(w_x)
                    } {
                        Some(unsafe { pyre_object::w_float_get_value(w_x) })
                    } else if unsafe {
                        pyre_object::is_exact_builtin_instance(w_x)
                            && (pyre_object::is_int(w_x) || pyre_object::is_bool(w_x))
                    } {
                        Some(unsafe { pyre_object::w_int_get_value(w_x) } as f64)
                    } else {
                        None
                    };
                    if let Some($x) = $x {
                        $fast
                    }
                }
                [<$name _slow>](args)
            }

            #[cfg(not(target_arch = "wasm32"))]
            #[linkme::distributed_slice(pyre_interpreter::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
            #[allow(non_upper_case_globals)]
            static [<__majit_builtin_wrapper_target_math_ $name>]:
                pyre_interpreter::gateway::BuiltinWrapperDescriptor =
                pyre_interpreter::gateway::BuiltinWrapperDescriptor {
                    path: concat!(
                        module_path!(),
                        "::",
                        stringify!([<__majit_wrap_math_ $name>]),
                    ),
                    func: [<__majit_wrap_math_ $name>],
                };
        }
    };
}

use pyre_interpreter::objspace::descroperation::{
    _float_acos, _float_acosh, _float_asin, _float_asinh, _float_atan, _float_atanh, _float_cbrt,
    _float_cos, _float_degrees, _float_erf, _float_erfc, _float_log, _float_log1p, _float_radians,
    _float_sin, _float_tan, _float_tanh, _float_ulp,
};

majit_math1_gateway!(sin, _float_sin, |x| x.is_finite());
majit_math1_gateway!(cos, _float_cos, |x| x.is_finite());
majit_math1_gateway!(tan, _float_tan, |x| x.is_finite());
majit_math1_gateway!(asin, _float_asin, |x| x >= -1.0 && x <= 1.0);
majit_math1_gateway!(acos, _float_acos, |x| x >= -1.0 && x <= 1.0);
majit_math1_gateway!(atan, _float_atan, |x| x.is_finite());
majit_math1_gateway!(tanh, _float_tanh, |x| x.is_finite());
majit_math1_gateway!(asinh, _float_asinh, |x| x.is_finite());
majit_math1_gateway!(acosh, _float_acosh, |x| x.is_finite() && x >= 1.0);
majit_math1_gateway!(atanh, _float_atanh, |x| x.is_finite()
    && x > -1.0
    && x < 1.0);
majit_math1_gateway!(log1p, _float_log1p, |x| x.is_finite() && x > -1.0);
majit_math1_gateway!(cbrt, _float_cbrt, total);
majit_math1_gateway!(erf, _float_erf, total);
majit_math1_gateway!(erfc, _float_erfc, total);
majit_math1_gateway!(ulp, _float_ulp, total);
majit_math1_gateway!(degrees, _float_degrees, total);
majit_math1_gateway!(radians, _float_radians, total);

/// `math1` whose leaf can overflow: call the raw `f64` sibling, and a
/// non-finite result (the fold's result check) stays in the original body.
/// `$guard` binds the operand and the raw result with `|x, y|`.
macro_rules! majit_math1_raw_gateway {
    ($name:ident, $raw:path, |$x:ident, $y:ident| $guard:expr) => {
        ::paste::paste! {
            #[majit_macros::dont_look_inside]
            fn [<$name _slow>](args: &[PyObjectRef]) -> PyResult {
                pyre_interpreter::gateway::check_declared_positional_arity(
                    stringify!($name),
                    1,
                    args,
                )?;
                $name(args)
            }

            /// interp_math.py `math1`: exact float, or a machine int/bool read
            /// as `f64`. A non-finite operand or raw result runs `$name`.
            pub fn [<__majit_wrap_math_ $name>](args: &[PyObjectRef]) -> PyResult {
                if args.len() == 1 {
                    let w_x = args[0];
                    let $x = if unsafe {
                        pyre_object::is_exact_builtin_instance(w_x)
                            && pyre_object::is_float(w_x)
                    } {
                        Some(unsafe { pyre_object::w_float_get_value(w_x) })
                    } else if unsafe {
                        pyre_object::is_exact_builtin_instance(w_x)
                            && (pyre_object::is_int(w_x) || pyre_object::is_bool(w_x))
                    } {
                        Some(unsafe { pyre_object::w_int_get_value(w_x) } as f64)
                    } else {
                        None
                    };
                    if let Some($x) = $x {
                        if $x.is_finite() {
                            let $y = $raw($x);
                            if $guard {
                                return pyre_interpreter::objspace::descroperation::_float_pos($y);
                            }
                        }
                    }
                }
                [<$name _slow>](args)
            }

            #[cfg(not(target_arch = "wasm32"))]
            #[linkme::distributed_slice(pyre_interpreter::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
            #[allow(non_upper_case_globals)]
            static [<__majit_builtin_wrapper_target_math_ $name>]:
                pyre_interpreter::gateway::BuiltinWrapperDescriptor =
                pyre_interpreter::gateway::BuiltinWrapperDescriptor {
                    path: concat!(
                        module_path!(),
                        "::",
                        stringify!([<__majit_wrap_math_ $name>]),
                    ),
                    func: [<__majit_wrap_math_ $name>],
                };
        }
    };
}

use pyre_interpreter::objspace::descroperation::{
    _float_cosh, _float_exp, _float_exp2, _float_expm1, _float_gamma_raw, _float_lgamma_raw,
    _float_sinh,
};

// The overflow direction is pinned before the C call rather than read off
// its result: `ll_math` raises `OverflowError` from inside the same graph,
// while a gateway can only hand the operand back to the slow path, which
// must not be reachable once the call has run.  Below `709` (`1023` for the
// base-2 form) the result is finite; the band up to the true bound stays in
// the slow path.
majit_math1_gateway!(exp, _float_exp, |x| x.is_finite() && x < 709.0);
majit_math1_gateway!(exp2, _float_exp2, |x| x.is_finite() && x < 1023.0);
majit_math1_gateway!(expm1, _float_expm1, |x| x.is_finite() && x < 709.0);
majit_math1_gateway!(sinh, _float_sinh, |x| x > -709.0 && x < 709.0);
majit_math1_gateway!(cosh, _float_cosh, |x| x > -709.0 && x < 709.0);
// A pole and an overflow both come back non-finite (`NaN` / inf).
majit_math1_raw_gateway!(gamma, _float_gamma_raw, |x, y| y.is_finite());
majit_math1_raw_gateway!(lgamma, _float_lgamma_raw, |x, y| y.is_finite());

/// interp_math.py `math2`. `$fast` binds the operands with `|x, y|` and returns on the
/// fast arm. Two arguments are read directly; anything else is `_slow`.
macro_rules! majit_math2_gateway {
    ($name:ident, |$x:ident, $y:ident| $fast:block) => {
        ::paste::paste! {
            #[majit_macros::dont_look_inside]
            fn [<$name _slow>](args: &[PyObjectRef]) -> PyResult {
                pyre_interpreter::gateway::check_declared_positional_arity(
                    stringify!($name),
                    2,
                    args,
                )?;
                $name(args)
            }

            pub fn [<__majit_wrap_math_ $name>](args: &[PyObjectRef]) -> PyResult {
                if args.len() == 2 {
                    let w_x = args[0];
                    let w_y = args[1];
                    let $x = if unsafe {
                        pyre_object::is_exact_builtin_instance(w_x) && pyre_object::is_float(w_x)
                    } {
                        Some(unsafe { pyre_object::w_float_get_value(w_x) })
                    } else if unsafe {
                        pyre_object::is_exact_builtin_instance(w_x)
                            && (pyre_object::is_int(w_x) || pyre_object::is_bool(w_x))
                    } {
                        Some(unsafe { pyre_object::w_int_get_value(w_x) } as f64)
                    } else {
                        None
                    };
                    let $y = if unsafe {
                        pyre_object::is_exact_builtin_instance(w_y) && pyre_object::is_float(w_y)
                    } {
                        Some(unsafe { pyre_object::w_float_get_value(w_y) })
                    } else if unsafe {
                        pyre_object::is_exact_builtin_instance(w_y)
                            && (pyre_object::is_int(w_y) || pyre_object::is_bool(w_y))
                    } {
                        Some(unsafe { pyre_object::w_int_get_value(w_y) } as f64)
                    } else {
                        None
                    };
                    if let (Some($x), Some($y)) = ($x, $y) {
                        $fast
                    }
                }
                [<$name _slow>](args)
            }

            #[cfg(not(target_arch = "wasm32"))]
            #[linkme::distributed_slice(pyre_interpreter::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
            #[allow(non_upper_case_globals)]
            static [<__majit_builtin_wrapper_target_math_ $name>]:
                pyre_interpreter::gateway::BuiltinWrapperDescriptor =
                pyre_interpreter::gateway::BuiltinWrapperDescriptor {
                    path: concat!(
                        module_path!(),
                        "::",
                        stringify!([<__majit_wrap_math_ $name>]),
                    ),
                    func: [<__majit_wrap_math_ $name>],
                };
        }
    };
}

use pyre_interpreter::objspace::descroperation::{
    _float_atan2, _float_copysign, _float_fmod, _float_isclose, _float_ldexp_raw, _float_pos,
    _float_pow, _float_remainder, _int_frexp_exponent_raw, _int_from_ceil, _int_from_floor,
    _int_from_trunc, _int_isqrt,
};

// ll_math.py `ll_math_pow` on the arm where the C call can neither overflow
// nor leave the real line, pinned before the call for the reason `exp`
// gives. The frexp exponent `e` of a normal finite `x` is in
// `[-1021, 1024]` (a zero, subnormal, infinity or NaN falls outside), and
// `|x|` lies in `[2**(e-1), 2**e)`, so `|log2(|x|)| <= |e| + 1` and
// `|y| * (|e| + 1) < 1000` keeps the result inside the double range. A
// negative base needs an integral `y` (below `2**53`, where the
// float-to-int cast is exact).
majit_math2_gateway!(pow, |x, y| {
    let e = _int_frexp_exponent_raw(x);
    if e >= -1021 && e <= 1024 {
        let e_abs = if e < 0 { -e } else { e };
        let y_abs = if y < 0.0 { -y } else { y };
        if y_abs * ((e_abs + 1) as f64) < 1000.0
            && (x > 0.0 || (y_abs < 9007199254740992.0 && ((y as i64) as f64) == y))
        {
            return _float_pow(x, y);
        }
    }
});
// `y != 0` is the fold's `YNonZero` pin. Both zeros compare equal.
majit_math2_gateway!(fmod, |x, y| {
    if x.is_finite() && y.is_finite() && y != 0.0 {
        return _float_fmod(x, y);
    }
});
majit_math2_gateway!(remainder, |x, y| {
    if x.is_finite() && y.is_finite() && y != 0.0 {
        return _float_remainder(x, y);
    }
});
// `copysign` and `atan2` are total on every exact float or machine int.
majit_math2_gateway!(copysign, |x, y| {
    return _float_copysign(x, y);
});
majit_math2_gateway!(atan2, |x, y| {
    return _float_atan2(x, y);
});

/// `floor` / `ceil` / `trunc` on an exact float inside the signed machine
/// range. `2**63` is the strict upper bound (`i64::MAX` is not a float).
/// NaN, infinities, out-of-range values and non-floats stay in `_slow`.
macro_rules! majit_math_round_gateway {
    ($name:ident, $leaf:path) => {
        ::paste::paste! {
            #[majit_macros::dont_look_inside]
            fn [<$name _slow>](args: &[PyObjectRef]) -> PyResult {
                pyre_interpreter::gateway::check_declared_positional_arity(
                    stringify!($name),
                    1,
                    args,
                )?;
                $name(args)
            }

            pub fn [<__majit_wrap_math_ $name>](args: &[PyObjectRef]) -> PyResult {
                if args.len() == 1 {
                    let w_x = args[0];
                    if unsafe {
                        pyre_object::is_exact_builtin_instance(w_x) && pyre_object::is_float(w_x)
                    } {
                        let x = unsafe { pyre_object::w_float_get_value(w_x) };
                        const SIGNED_MIN_AS_FLOAT: f64 = -9223372036854775808.0;
                        const SIGNED_LIMIT_AS_FLOAT: f64 = 9223372036854775808.0;
                        if x >= SIGNED_MIN_AS_FLOAT && x < SIGNED_LIMIT_AS_FLOAT {
                            return $leaf(x);
                        }
                    }
                }
                [<$name _slow>](args)
            }

            #[cfg(not(target_arch = "wasm32"))]
            #[linkme::distributed_slice(pyre_interpreter::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
            #[allow(non_upper_case_globals)]
            static [<__majit_builtin_wrapper_target_math_ $name>]:
                pyre_interpreter::gateway::BuiltinWrapperDescriptor =
                pyre_interpreter::gateway::BuiltinWrapperDescriptor {
                    path: concat!(
                        module_path!(),
                        "::",
                        stringify!([<__majit_wrap_math_ $name>]),
                    ),
                    func: [<__majit_wrap_math_ $name>],
                };
        }
    };
}

majit_math_round_gateway!(floor, _int_from_floor);
majit_math_round_gateway!(ceil, _int_from_ceil);
majit_math_round_gateway!(trunc, _int_from_trunc);

/// Arity / keyword / domain residual of `fabs`.
#[majit_macros::dont_look_inside]
fn fabs_slow(args: &[PyObjectRef]) -> PyResult {
    pyre_interpreter::gateway::check_declared_positional_arity("fabs", 1, args)?;
    fabs(args)
}

/// interp_math.py `fabs` = `math1` of `math.fabs`. Every exact float or
/// machine int/bool is in domain; `ll_math_fabs` raises for none of them.
pub fn __majit_wrap_math_fabs(args: &[PyObjectRef]) -> PyResult {
    if args.len() == 1 {
        let w_x = args[0];
        let x =
            if unsafe { pyre_object::is_exact_builtin_instance(w_x) && pyre_object::is_float(w_x) }
            {
                Some(unsafe { pyre_object::w_float_get_value(w_x) })
            } else if unsafe {
                pyre_object::is_exact_builtin_instance(w_x)
                    && (pyre_object::is_int(w_x) || pyre_object::is_bool(w_x))
            } {
                Some(unsafe { pyre_object::w_int_get_value(w_x) } as f64)
            } else {
                None
            };
        if let Some(x) = x {
            return pyre_interpreter::objspace::descroperation::_float_abs(x);
        }
    }
    fabs_slow(args)
}

#[cfg(not(target_arch = "wasm32"))]
#[linkme::distributed_slice(pyre_interpreter::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
#[allow(non_upper_case_globals)]
static __majit_builtin_wrapper_target_math_fabs:
    pyre_interpreter::gateway::BuiltinWrapperDescriptor =
    pyre_interpreter::gateway::BuiltinWrapperDescriptor {
        path: concat!(module_path!(), "::", stringify!(__majit_wrap_math_fabs)),
        func: __majit_wrap_math_fabs,
    };

/// `log` keeps its optional base, so this residual is the original body
/// rather than a fixed arity-1 check.
#[majit_macros::dont_look_inside]
fn log_slow(args: &[PyObjectRef]) -> PyResult {
    log(args)
}

/// interp_math.py `log` = `loghelper` of one operand (`ll_math_log`) when the
/// value is finite and strictly positive. A base, a non-positive input, or a
/// non-machine number stays in `log`.
pub fn __majit_wrap_math_log(args: &[PyObjectRef]) -> PyResult {
    if args.len() == 1 {
        let w_x = args[0];
        let x =
            if unsafe { pyre_object::is_exact_builtin_instance(w_x) && pyre_object::is_float(w_x) }
            {
                Some(unsafe { pyre_object::w_float_get_value(w_x) })
            } else if unsafe {
                pyre_object::is_exact_builtin_instance(w_x)
                    && (pyre_object::is_int(w_x) || pyre_object::is_bool(w_x))
            } {
                Some(unsafe { pyre_object::w_int_get_value(w_x) } as f64)
            } else {
                None
            };
        if let Some(x) = x {
            if x.is_finite() && x > 0.0 {
                return _float_log(x);
            }
        }
    }
    log_slow(args)
}

#[cfg(not(target_arch = "wasm32"))]
#[linkme::distributed_slice(pyre_interpreter::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
#[allow(non_upper_case_globals)]
static __majit_builtin_wrapper_target_math_log:
    pyre_interpreter::gateway::BuiltinWrapperDescriptor =
    pyre_interpreter::gateway::BuiltinWrapperDescriptor {
        path: concat!(module_path!(), "::", stringify!(__majit_wrap_math_log)),
        func: __majit_wrap_math_log,
    };

/// Every `math` gateway with the path its `BUILTIN_WRAPPER_DESCRIPTORS` entry
/// carries.  wasm32 links no descriptor slice, so `publish_optional_fnaddrs`
/// binds these paths there instead, the way `jit_fnaddr` binds the
/// interpreter's own gateways; the descent finds the jitcode by that path.
macro_rules! math_gateway_fnaddrs {
    ($($name:ident),* $(,)?) => {
        ::paste::paste! {
            pub fn math_gateway_fnaddrs() -> Vec<(&'static str, *const ())> {
                vec![$((
                    concat!(module_path!(), "::__majit_wrap_math_", stringify!($name)),
                    [<__majit_wrap_math_ $name>] as *const (),
                )),*]
            }
        }
    };
}

math_gateway_fnaddrs!(
    sqrt, sin, cos, tan, asin, acos, atan, tanh, asinh, acosh, atanh, log1p, cbrt, erf, erfc, ulp,
    degrees, radians, fabs, exp, exp2, expm1, sinh, cosh, gamma, lgamma, floor, ceil, trunc, isqrt,
    log, isclose, pow, fmod, copysign, remainder, atan2, ldexp
);

/// The name of the canonical `math` builtin `callable` is, or `None` for any
/// other value.  Only `frexp` keeps a walker fold, so it is the one name
/// answered; a value rebound under it carries a different builtin code and
/// answers `None`.
pub fn math_builtin_name(callable: PyObjectRef) -> Option<&'static str> {
    unsafe {
        if callable.is_null() || !pyre_interpreter::is_function(callable) {
            return None;
        }
        let code = pyre_interpreter::function_get_code(callable) as PyObjectRef;
        let is_frexp = !code.is_null()
            && pyre_interpreter::gateway::is_builtin_code(code)
            && pyre_interpreter::gateway::builtin_code_get(code) as usize
                == frexp as *const () as usize;
        is_frexp.then_some("frexp")
    }
}

/// `ldexp` off its exact arm: a zero, subnormal or non-finite `x`, an
/// exponent outside the normal range, or a result that would round,
/// overflow or underflow.
#[majit_macros::dont_look_inside]
fn ldexp_slow(args: &[PyObjectRef]) -> PyResult {
    pyre_interpreter::gateway::check_declared_positional_arity("ldexp", 2, args)?;
    ldexp(args)
}

/// interp_math.py `ldexp`. A bool exponent, a non-int, or an overflowing
/// scale stays in `ldexp`.
pub fn __majit_wrap_math_ldexp(args: &[PyObjectRef]) -> PyResult {
    if args.len() == 2 {
        let w_x = args[0];
        let w_i = args[1];
        let x =
            if unsafe { pyre_object::is_exact_builtin_instance(w_x) && pyre_object::is_float(w_x) }
            {
                Some(unsafe { pyre_object::w_float_get_value(w_x) })
            } else if unsafe {
                pyre_object::is_exact_builtin_instance(w_x)
                    && (pyre_object::is_int(w_x) || pyre_object::is_bool(w_x))
            } {
                Some(unsafe { pyre_object::w_int_get_value(w_x) } as f64)
            } else {
                None
            };
        let exp = if unsafe {
            pyre_object::is_exact_builtin_instance(w_i)
                && pyre_object::is_int(w_i)
                && !pyre_object::is_bool(w_i)
        } {
            Some(unsafe { pyre_object::w_int_get_value(w_i) })
        } else {
            None
        };
        if let (Some(x), Some(exp)) = (x, exp) {
            // ll_math.py `ll_math_ldexp` where scaling is exact: a normal
            // `x` and a normal `2**exp` whose product stays normal, i.e. the
            // frexp exponent `e` of `x` has `-1021 <= e + exp <= 1024`.
            if x.is_finite()
                && (x >= f64::MIN_POSITIVE || x <= -f64::MIN_POSITIVE)
                && exp >= -1022
                && exp <= 1023
            {
                let e = _int_frexp_exponent_raw(x) + exp;
                if e >= -1021 && e <= 1024 {
                    return _float_pos(_float_ldexp_raw(x, exp));
                }
            }
        }
    }
    ldexp_slow(args)
}

#[cfg(not(target_arch = "wasm32"))]
#[linkme::distributed_slice(pyre_interpreter::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
#[allow(non_upper_case_globals)]
static __majit_builtin_wrapper_target_math_ldexp:
    pyre_interpreter::gateway::BuiltinWrapperDescriptor =
    pyre_interpreter::gateway::BuiltinWrapperDescriptor {
        path: concat!(module_path!(), "::", stringify!(__majit_wrap_math_ldexp)),
        func: __majit_wrap_math_ldexp,
    };

/// `isqrt` on an exact positive machine int below `2**53`. Zero, a bool,
/// a long and a negative stay in `isqrt`.
#[majit_macros::dont_look_inside]
fn isqrt_slow(args: &[PyObjectRef]) -> PyResult {
    pyre_interpreter::gateway::check_declared_positional_arity("isqrt", 1, args)?;
    isqrt(args)
}

/// app_math.py `isqrt` on a `W_IntObject` that fits an exact `f64`.
pub fn __majit_wrap_math_isqrt(args: &[PyObjectRef]) -> PyResult {
    if args.len() == 1 {
        let w_n = args[0];
        if unsafe {
            pyre_object::is_exact_builtin_instance(w_n)
                && pyre_object::is_int(w_n)
                && !pyre_object::is_bool(w_n)
        } {
            let n = unsafe { pyre_object::w_int_get_value(w_n) };
            const EXACT_FLOAT_INT: i64 = 1 << 53;
            if n >= 1 && n < EXACT_FLOAT_INT {
                return _int_isqrt(n);
            }
        }
    }
    isqrt_slow(args)
}

#[cfg(not(target_arch = "wasm32"))]
#[linkme::distributed_slice(pyre_interpreter::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
#[allow(non_upper_case_globals)]
static __majit_builtin_wrapper_target_math_isqrt:
    pyre_interpreter::gateway::BuiltinWrapperDescriptor =
    pyre_interpreter::gateway::BuiltinWrapperDescriptor {
        path: concat!(module_path!(), "::", stringify!(__majit_wrap_math_isqrt)),
        func: __majit_wrap_math_isqrt,
    };

/// Keyword tolerances stay in `isclose`. This residual is the original body.
#[majit_macros::dont_look_inside]
fn isclose_slow(args: &[PyObjectRef]) -> PyResult {
    isclose(args)
}

/// interp_math.py `isclose` with both tolerances defaulted, on two finite
/// exact floats or machine ints.
pub fn __majit_wrap_math_isclose(args: &[PyObjectRef]) -> PyResult {
    if args.len() == 2 {
        let w_a = args[0];
        let w_b = args[1];
        let a =
            if unsafe { pyre_object::is_exact_builtin_instance(w_a) && pyre_object::is_float(w_a) }
            {
                Some(unsafe { pyre_object::w_float_get_value(w_a) })
            } else if unsafe {
                pyre_object::is_exact_builtin_instance(w_a)
                    && (pyre_object::is_int(w_a) || pyre_object::is_bool(w_a))
            } {
                Some(unsafe { pyre_object::w_int_get_value(w_a) } as f64)
            } else {
                None
            };
        let b =
            if unsafe { pyre_object::is_exact_builtin_instance(w_b) && pyre_object::is_float(w_b) }
            {
                Some(unsafe { pyre_object::w_float_get_value(w_b) })
            } else if unsafe {
                pyre_object::is_exact_builtin_instance(w_b)
                    && (pyre_object::is_int(w_b) || pyre_object::is_bool(w_b))
            } {
                Some(unsafe { pyre_object::w_int_get_value(w_b) } as f64)
            } else {
                None
            };
        if let (Some(a), Some(b)) = (a, b) {
            if a.is_finite() && b.is_finite() {
                return _float_isclose(a, b);
            }
        }
    }
    isclose_slow(args)
}

#[cfg(not(target_arch = "wasm32"))]
#[linkme::distributed_slice(pyre_interpreter::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
#[allow(non_upper_case_globals)]
static __majit_builtin_wrapper_target_math_isclose:
    pyre_interpreter::gateway::BuiltinWrapperDescriptor =
    pyre_interpreter::gateway::BuiltinWrapperDescriptor {
        path: concat!(module_path!(), "::", stringify!(__majit_wrap_math_isclose)),
        func: __majit_wrap_math_isclose,
    };

/// Raw counterparts of `ll_math_floor` / `ll_math_ceil` for a guarded JIT fast
/// path.  Both are total on finite input and cannot raise, so the walker emits
/// them as pure elidable calls; the trace guards the rounded value into the
/// machine range before casting.
pub extern "C" fn jit_math_floor_raw(x: f64) -> f64 {
    x.floor()
}

pub extern "C" fn jit_math_ceil_raw(x: f64) -> f64 {
    x.ceil()
}

// ── raw helpers and identity table for the generic float folds ───────

macro_rules! jit_raw2 {
    ($helper:ident, $name:ident) => {
        pub extern "C" fn $helper(x: f64, y: f64) -> f64 {
            match pymath::math::$name(x, y) {
                Ok(v) => v,
                Err(_) => f64::NAN,
            }
        }
    };
}

jit_raw2!(jit_math_pow, pow);
jit_raw2!(jit_math_fmod, fmod);
jit_raw2!(jit_math_copysign, copysign);
jit_raw2!(jit_math_remainder, remainder);
jit_raw2!(jit_math_atan2, atan2);

/// `ll_math.py math_hypot` — the C `hypot` llexternal, not the raising
/// `ll_math_hypot` wrapper.  Two finite arguments that overflow return
/// +inf; the wrapper (and `complex_abs`) turn that into OverflowError.
pub extern "C" fn jit_math_hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}

/// `ll_math.py` C llexternals for the Opaque `f64` inherent methods
/// `ll_math::f64_method_llexternal` names.
/// IEEE, no raise — the `ll_math_*` wrappers stay around them.
macro_rules! jit_math_raw1 {
    ($($helper:ident => $method:ident),* $(,)?) => {
        $(
            pub extern "C" fn $helper(x: f64) -> f64 {
                x.$method()
            }
        )*
    };
}

jit_math_raw1! {
    jit_math_log_raw => ln,
    jit_math_log10_raw => log10,
    jit_math_log1p_raw => ln_1p,
    jit_math_exp_raw => exp,
    jit_math_exp2_raw => exp2,
    jit_math_expm1_raw => exp_m1,
    jit_math_sqrt_raw => sqrt,
    jit_math_cbrt_raw => cbrt,
    jit_math_sin_raw => sin,
    jit_math_cos_raw => cos,
    jit_math_tan_raw => tan,
    jit_math_asin_raw => asin,
    jit_math_acos_raw => acos,
    jit_math_atan_raw => atan,
    jit_math_sinh_raw => sinh,
    jit_math_cosh_raw => cosh,
    jit_math_tanh_raw => tanh,
    jit_math_asinh_raw => asinh,
    jit_math_acosh_raw => acosh,
    jit_math_atanh_raw => atanh,
}

pub extern "C" fn jit_math_pow_raw(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// The C `fmod` llexternal, which is also what `%` over two floats lowers
/// to: `lloperation.py` has no `float_mod`, so the codewriter emits a
/// residual call of this name instead.
pub extern "C" fn jit_math_fmod_raw(x: f64, y: f64) -> f64 {
    x % y
}

pub fn cbrt(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("cbrt", args, pymath::math::cbrt, |_| {
        "math domain error".to_string()
    })
}
pub fn exp(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("exp", args, pymath::math::exp, |_| {
        "math domain error".to_string()
    })
}
pub fn exp2(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("exp2", args, pymath::math::exp2, |_| {
        "math domain error".to_string()
    })
}
pub fn expm1(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("expm1", args, pymath::math::expm1, |_| {
        "math domain error".to_string()
    })
}
pub fn log1p(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("log1p", args, pymath::math::log1p, |v| {
        format!("expected argument value > -1, got {}", float_repr(v))
    })
}

// Gamma / error
pub fn erf(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("erf", args, pymath::math::erf, |_| {
        "math domain error".to_string()
    })
}
pub fn erfc(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("erfc", args, pymath::math::erfc, |_| {
        "math domain error".to_string()
    })
}
pub fn gamma(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("gamma", args, pymath::math::gamma, |v| {
        format!(
            "expected a noninteger or positive integer, got {}",
            float_repr(v)
        )
    })
}
pub fn lgamma(args: &[PyObjectRef]) -> PyResult {
    math1_pymath("lgamma", args, pymath::math::lgamma, |v| {
        format!(
            "expected a noninteger or positive integer, got {}",
            float_repr(v)
        )
    })
}

// Misc
/// `math.fabs` after `_get_double`: the unboxed `_float_abs` leaf.
pub fn fabs(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 1 {
        return Err(pyre_interpreter::PyError::type_error(
            "fabs() takes exactly one argument",
        ));
    }
    pyre_interpreter::objspace::descroperation::_float_abs(try_get_double(args[0])?)
}
pub fn ulp(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 1 {
        return Err(pyre_interpreter::PyError::type_error(
            "ulp() takes exactly one argument",
        ));
    }
    pyre_interpreter::objspace::descroperation::_float_pos(pymath::math::ulp(try_get_double(
        args[0],
    )?))
}

// ── 2-arg float→float via pymath ─────────────────────────────────────

/// Domain pin via pymath; box the pymath success value.  The walker
/// still descends the unboxed two-arg leaf.
fn math2_pymath(
    name: &str,
    args: &[PyObjectRef],
    compute: fn(f64, f64) -> Result<f64, pymath::Error>,
) -> PyResult {
    if args.len() != 2 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{name}() takes exactly 2 arguments"
        )));
    }
    let x = try_get_double(args[0])?;
    let y = try_get_double(args[1])?;
    match compute(x, y) {
        Ok(v) => pyre_interpreter::objspace::descroperation::_float_pos(v),
        Err(pymath::Error::EDOM) => {
            Err(pyre_interpreter::PyError::value_error("math domain error"))
        }
        Err(pymath::Error::ERANGE) => Err(pyre_interpreter::PyError::overflow_error(
            "math range error",
        )),
    }
}

pub fn pow(args: &[PyObjectRef]) -> PyResult {
    math2_pymath("pow", args, pymath::math::pow)
}
pub fn fmod(args: &[PyObjectRef]) -> PyResult {
    math2_pymath("fmod", args, pymath::math::fmod)
}
pub fn copysign(args: &[PyObjectRef]) -> PyResult {
    math2_pymath("copysign", args, pymath::math::copysign)
}
pub fn remainder(args: &[PyObjectRef]) -> PyResult {
    math2_pymath("remainder", args, pymath::math::remainder)
}
pub fn atan2(args: &[PyObjectRef]) -> PyResult {
    math2_pymath("atan2", args, pymath::math::atan2)
}

pub fn hypot(args: &[PyObjectRef]) -> PyResult {
    let args = no_keywords(args, "hypot")?;
    let coords: Vec<f64> = args
        .iter()
        .map(|&a| try_get_double(a))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(floatobject::w_float_new(pymath::math::hypot(&coords)))
}

pub fn dist(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "dist() takes exactly 2 arguments",
        ));
    }
    let collect_coords = |obj: PyObjectRef| -> Result<Vec<f64>, pyre_interpreter::PyError> {
        let items = pyre_interpreter::builtins::collect_iterable(obj)?;
        // Conversion can call an element's `__float__`, so publish the whole
        // materialized sequence before converting any member. `pin_roots`
        // writes every pointer before the first forwarding query; a
        // per-item `pin_root` would let that query collect the rest.
        let _roots = pyre_object::gc_roots::push_roots();
        let items_base = pyre_object::gc_roots::pin_roots(&items);
        (0..items.len())
            .map(|index| {
                try_get_double(pyre_object::gc_roots::shadow_stack_get(items_base + index))
            })
            .collect()
    };
    let p = collect_coords(args[0])?;
    let q = collect_coords(args[1])?;
    if p.len() != q.len() {
        return Err(pyre_interpreter::PyError::value_error(
            "both points must have the same number of dimensions",
        ));
    }
    Ok(floatobject::w_float_new(pymath::math::dist(&p, &q)))
}

// ── Integer-returning functions ──────────────────────────────────────

/// Invoke `__ceil__`/`__floor__`/`__trunc__` looked up on the argument's
/// type (special-method semantics, so an instance attribute is ignored).
///
/// `math.ceil`/`math.floor` fall back to `__float__` coercion when the
/// dunder is absent, so `ceil(FloatLike(...))` works; `math.trunc` has no
/// such fallback and requires `__trunc__`.
fn math_unary_int(
    args: &[PyObjectRef],
    dunder: &str,
    fname: &str,
    fallback_float: bool,
) -> PyResult {
    if args.len() != 1 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "{fname}() takes exactly 1 argument",
        )));
    }
    // `interp_math.py`'s `floor` / `ceil` / `trunc`:
    //
    //     w_descr = space.lookup(w_x, '__floor__')
    //     if w_descr is not None:
    //         return space.get_and_call_function(w_descr, w_x)
    //
    // The unbound descriptor is called with the object leading the
    // positionals, so a plain `float` argument does not pay for a bound method
    // object per call.  A descriptor whose `__get__` itself raises (e.g.
    // BadDescr.__get__ → ValueError) still propagates that error:
    // `get_and_call_function` binds through `get` for every descriptor other
    // than a function or method descriptor, and neither of those runs user
    // code to bind.  `lookup` reads the type MRO only, so an instance
    // attribute of the same name stays ignored.
    if let Some(w_descr) = unsafe { pyre_interpreter::baseobjspace::lookup(args[0], dunder) }
        && let Some(w_type) = pyre_interpreter::typedef::r#type(args[0])
    {
        return unsafe {
            pyre_interpreter::baseobjspace::get_and_call_function(
                w_descr,
                args[0],
                w_type.as_ptr(),
                &[],
            )
        };
    }
    if !fallback_float {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "type {} doesn't define {dunder} method",
            pyre_interpreter::baseobjspace::object_functionstr_type_name(args[0])
        )));
    }
    // Fall back to `__float__` coercion.  `try_get_double` already reports
    // "must be real number, not X" for an operand with no numeric
    // interpretation, and its contract is that everything else — a raising
    // `__float__`, or the `OverflowError` for an int too wide for an f64 —
    // propagates; relabelling here would swallow exactly those.
    let v = try_get_double(args[0])?;
    // `float_to_pyint` is `newlong_from_float`: NaN raises, an infinity raises,
    // and a finite value outside the machine range becomes a long.  A direct
    // `as i64` saturates instead, so `floor(FloatLike(1e300))` answered
    // `i64::MAX` and `floor(FloatLike(nan))` answered `0`.
    pyre_interpreter::typedef::float_to_pyint(
        v,
        match dunder {
            "__ceil__" => pyre_interpreter::typedef::FloatToIntMode::Ceil,
            "__floor__" => pyre_interpreter::typedef::FloatToIntMode::Floor,
            _ => pyre_interpreter::typedef::FloatToIntMode::Trunc,
        },
    )
}

pub fn floor(args: &[PyObjectRef]) -> PyResult {
    math_unary_int(args, "__floor__", "floor", true)
}

pub fn ceil(args: &[PyObjectRef]) -> PyResult {
    math_unary_int(args, "__ceil__", "ceil", true)
}

pub fn trunc(args: &[PyObjectRef]) -> PyResult {
    math_unary_int(args, "__trunc__", "trunc", false)
}

// ── Special signatures ──────────────────────────────────────────────

/// `loghelper(arg, func)` — one operand's logarithm, where `base` spells which
/// `func` upstream passed: `0.0` for `m_log`, `2.0` for `m_log2` and `10.0`
/// for `m_log10`.
///
/// An integer operand is read from its payload rather than coerced, so a value
/// no `float` can hold still has a logarithm, and only a non-integer reaches
/// the general coercion, where an overridden `__float__` is what answers.  The
/// two arms also state their refusal differently — an integer can be
/// arbitrarily large, so its message carries no value — and which spelling a
/// program sees says which arm read the operand.
fn loghelper(w_x: PyObjectRef, base: f64) -> Result<f64, pyre_interpreter::PyError> {
    unsafe {
        if pyre_object::is_bool(w_x) || pyre_object::is_int(w_x) || pyre_object::is_long(w_x) {
            let num_owned;
            let num: &BigInt = if pyre_object::is_long(w_x) {
                pyre_object::w_long_get_value(w_x)
            } else if pyre_object::is_bool(w_x) {
                num_owned = BigInt::from(pyre_object::w_bool_get_value(w_x) as i64);
                &num_owned
            } else {
                num_owned = BigInt::from(pyre_object::w_int_get_value(w_x));
                &num_owned
            };
            if num.int_le(0) {
                return Err(pyre_interpreter::PyError::value_error(
                    "expected a positive input",
                ));
            }
            // `PyLong_AsDouble` first, so a value a `float` can hold takes the
            // logarithm of that conversion and answers as the float beside it
            // does.  Only a value that overflows falls back to the scaled
            // double, whose top bits are all the logarithm can be read from.
            return match num.tofloat() {
                Ok(x) => Ok(log_of_double(x, base)),
                Err(_) => num.log(base).map_err(map_rbigint_err),
            };
        }
    }
    let x = try_get_double(w_x)?;
    // NaN propagates through log.
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    // Domain error for x <= 0 (but x == +inf is fine).
    if x <= 0.0 {
        return Err(pyre_interpreter::PyError::value_error(format!(
            "expected a positive input, got {}",
            float_repr(x)
        )));
    }
    Ok(log_of_double(x, base))
}

/// The `func` [`loghelper`] was handed, spelled as the base it takes.
fn log_of_double(x: f64, base: f64) -> f64 {
    if base == 10.0 {
        x.log10()
    } else if base == 2.0 {
        x.log2()
    } else {
        x.ln()
    }
}

/// A `math` entry point declared `METH_VARARGS` takes no keywords at all, so
/// one is rejected before the arguments are read — otherwise the trailing
/// marker dict reaches the body as one more operand.
pub(crate) fn no_keywords<'a>(
    args: &'a [PyObjectRef],
    name: &str,
) -> Result<&'a [PyObjectRef], pyre_interpreter::PyError> {
    let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    if pyre_interpreter::builtins::has_real_kwargs(kwargs) {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "math.{name}() takes no keyword arguments"
        )));
    }
    Ok(positional)
}

pub fn log(args: &[PyObjectRef]) -> PyResult {
    let args = no_keywords(args, "log")?;
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "log expected at least 1 argument, got 0",
        ));
    }
    if args.len() > 2 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "log expected at most 2 arguments, got {}",
            args.len()
        )));
    }
    // `math_log_impl` is two `loghelper` calls and a division.  So the
    // argument is what a refusal names even when the base is out of the domain
    // too, a base of 1 leaves a zero denominator rather than a domain error,
    // and the result is one natural logarithm over another — not the logarithm
    // taken in that base, which rounds elsewhere.
    let num = loghelper(args[0], 0.0)?;
    let Some(base) = args.get(1).copied() else {
        return Ok(floatobject::w_float_new(num));
    };
    let den = loghelper(base, 0.0)?;
    if den == 0.0 {
        return Err(pyre_interpreter::PyError::zero_division("division by zero"));
    }
    Ok(floatobject::w_float_new(num / den))
}

pub fn log10(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 1 {
        return Err(pyre_interpreter::PyError::type_error(
            "log10() takes exactly 1 argument",
        ));
    }
    Ok(floatobject::w_float_new(loghelper(args[0], 10.0)?))
}

pub fn log2(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 1 {
        return Err(pyre_interpreter::PyError::type_error(
            "log2() takes exactly 1 argument",
        ));
    }
    Ok(floatobject::w_float_new(loghelper(args[0], 2.0)?))
}

pub fn degrees(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "degrees() takes exactly 1 argument",
        ));
    }
    pyre_interpreter::objspace::descroperation::_float_pos(pymath::math::degrees(try_get_double(
        args[0],
    )?))
}

pub fn radians(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "radians() takes exactly 1 argument",
        ));
    }
    pyre_interpreter::objspace::descroperation::_float_pos(pymath::math::radians(try_get_double(
        args[0],
    )?))
}

pub fn isinf(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "isinf() takes exactly 1 argument",
        ));
    }
    Ok(w_bool_from(pymath::math::isinf(try_get_double(args[0])?)))
}

pub fn isnan(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "isnan() takes exactly 1 argument",
        ));
    }
    Ok(w_bool_from(pymath::math::isnan(try_get_double(args[0])?)))
}

pub fn isfinite(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "isfinite() takes exactly 1 argument",
        ));
    }
    Ok(w_bool_from(pymath::math::isfinite(try_get_double(
        args[0],
    )?)))
}

pub fn isclose(args: &[PyObjectRef]) -> PyResult {
    let (pos, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    if pos.len() < 2 {
        // `_PyArg_ParseStackAndKeywords` names the first slot it could not
        // fill; both are positional-only, so a keyword never fills one.
        let missing = if pos.is_empty() {
            "a' (pos 1"
        } else {
            "b' (pos 2"
        };
        return Err(pyre_interpreter::PyError::type_error(format!(
            "isclose() missing required argument '{missing})"
        )));
    }
    if pos.len() > 2 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "isclose() takes exactly 2 positional arguments ({} given)",
            pos.len()
        )));
    }
    // `rel_tol` and `abs_tol` are the only (keyword-only) parameters.
    pyre_interpreter::builtins::kwarg_reject_unknown(kwargs, &["rel_tol", "abs_tol"], "isclose")?;
    // `interp_math.py`'s `isclose` — all four operands are converted, in this
    // order, before anything about them is checked, so a non-numeric `a` is
    // reported even when a tolerance is negative.  An omitted tolerance
    // arrives upstream as an already-wrapped float, so converting it can
    // neither raise nor reach `__float__`; `None` stands in for that here and
    // `pymath` supplies the same defaults.
    let a = try_get_double(pos[0])?;
    let b = try_get_double(pos[1])?;
    let read = |name: &str| -> Result<Option<f64>, pyre_interpreter::PyError> {
        match pyre_interpreter::builtins::kwarg_get(kwargs, name) {
            Some(v) => Ok(Some(try_get_double(v)?)),
            None => Ok(None),
        }
    };
    let rel_tol = read("rel_tol")?;
    let abs_tol = read("abs_tol")?;
    // `isclose` — the sanity check on the tolerances runs
    // after those conversions and before the comparison, and names them.
    // `pymath` reports the same rejection as EDOM, which `map_int_err`
    // relabels "math domain error".
    if rel_tol.is_some_and(|t| t < 0.0) || abs_tol.is_some_and(|t| t < 0.0) {
        return Err(pyre_interpreter::PyError::value_error(
            "tolerances must be non-negative",
        ));
    }
    match pymath::math::isclose(a, b, rel_tol, abs_tol) {
        Ok(v) => Ok(w_bool_from(v)),
        Err(e) => Err(map_int_err(e)),
    }
}

pub fn factorial(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "factorial() takes exactly 1 argument",
        ));
    }
    // pypy/module/math/app_math.py:factorial —
    //     if '__index__' not in dir(n):
    //         raise TypeError("'%s' object cannot be interpreted as an integer"
    //                         % type(n).__name__)
    // The check is on `__index__` alone, so floats are rejected for the same
    // reason strings are rather than by a numeric-value test. `dir(n)` only
    // reports membership, so the check must not bind the descriptor — the one
    // binding belongs to `get_bigint` below.
    if unsafe { pyre_interpreter::baseobjspace::lookup(args[0], "__index__") }.is_none() {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "'{}' object cannot be interpreted as an integer",
            pyre_interpreter::baseobjspace::object_functionstr_type_name(args[0])
        )));
    }
    let n_big = get_bigint(args[0])?;
    if n_big.int_lt(0) {
        return Err(pyre_interpreter::PyError::value_error(
            "factorial() not defined for negative values",
        ));
    }
    let n = if jit_bigint_to_i64_fits(&n_big) != 0 {
        jit_bigint_to_i64_value(&n_big)
    } else {
        return Err(pyre_interpreter::PyError::overflow_error(format!(
            "factorial() argument should not exceed {}",
            i64::MAX
        )));
    };

    // pypy/module/math/app_math.py:factorial — balanced odd-product tree.
    fn fac_odd(low: i64, high: i64, gap: i64) -> BigInt {
        if low + gap >= high {
            let mut result = BigInt::one();
            let mut i = low;
            while i < high {
                result = result.int_mul(i);
                i += 2;
            }
            return result;
        }
        let mid = ((low + high) >> 1) | 1;
        fac_odd(low, mid, gap).mul(&fac_odd(mid, high, gap))
    }
    fn fac1(x: i64, gap: i64) -> (BigInt, BigInt, i64) {
        if x <= 2 {
            return (BigInt::one(), BigInt::one(), x - 1);
        }
        let x2 = x >> 1;
        let (f, mut g, shift) = fac1(x2, gap);
        g = g.mul(&fac_odd((x2 + 1) | 1, x + 1, gap));
        (f.mul(&g), g, shift + x2)
    }

    let result = if n <= 100 {
        let mut result = BigInt::one();
        let mut i = 2;
        while i <= n {
            result = result.int_mul(i);
            i += 1;
        }
        result
    } else {
        let gap = 100.max(n >> 7);
        let (result, _, shift) = fac1(n, gap);
        result.lshift(shift).map_err(map_rbigint_err)?
    };
    Ok(bigint_to_pyint(&result))
}

/// Convert any int/long/bool to a BigInt for math.gcd/lcm/factorial
/// overflow-safe handling. PyPy: space.bigint_w() which traverses the
/// W_IntObject/W_LongObject/W_BoolObject union and materializes rbigint.
/// Raises TypeError for non-integer inputs via `__index__` dunder, matching
/// CPython's `_PyLong_FromNbIndexOrNbInt`.
fn get_bigint(obj: PyObjectRef) -> Result<BigInt, pyre_interpreter::PyError> {
    unsafe {
        if pyre_object::is_long(obj) {
            return Ok(pyre_object::w_long_get_value(obj).translated_alias());
        }
        if pyre_object::is_int(obj) {
            return Ok(BigInt::from(pyre_object::w_int_get_value(obj)));
        }
        if pyre_object::is_bool(obj) {
            return Ok(BigInt::from(if pyre_object::w_bool_get_value(obj) {
                1
            } else {
                0
            }));
        }
        if pyre_object::is_float(obj) {
            return Err(pyre_interpreter::PyError::type_error(
                "'float' object cannot be interpreted as an integer",
            ));
        }
    }
    // __index__ dunder — descroperation.py `_index`: type-only special-method
    // lookup, then propagate a raising `__index__` instead of masking it with
    // the generic "object cannot be interpreted as an integer".
    match unsafe { pyre_interpreter::baseobjspace::lookup_special(obj, "__index__") } {
        Ok(Some(method)) => {
            let result = pyre_interpreter::builtins::call_and_check(method, &[])?;
            unsafe {
                if pyre_object::is_int(result) {
                    return Ok(BigInt::from(pyre_object::w_int_get_value(result)));
                }
                if pyre_object::is_long(result) {
                    return Ok(pyre_object::w_long_get_value(result).translated_alias());
                }
            }
            // descroperation.py:612 — __index__ returned non-int (type %T)
            let result_type = unsafe { (*(*result).ob_type).name };
            return Err(pyre_interpreter::PyError::type_error(format!(
                "__index__ returned non-int (type '{result_type}')",
            )));
        }
        Ok(None) => {}
        Err(err) => return Err(err),
    }
    Err(pyre_interpreter::PyError::type_error(
        "object cannot be interpreted as an integer",
    ))
}

/// `space.abs(space.index(w))` in the machine-word domain.  `None` is
/// `gcd_two`'s `except OverflowError` direction, which replays the
/// pair in the rbigint domain: `is_long` values never fit, and `i64::MIN` is
/// the one machine int whose absolute value leaves the range.
fn index_abs_machine_word(obj: PyObjectRef) -> Option<i64> {
    if !unsafe { pyre_object::is_int(obj) } {
        return None;
    }
    unsafe { pyre_object::w_int_get_value(obj) }.checked_abs()
}

pub fn gcd(args: &[PyObjectRef]) -> PyResult {
    let args = no_keywords(args, "gcd")?;
    // `interp_math.py`'s `gcd_two` reads both operands as Signed and only
    // falls back to rbigint when one overflows.  Taking the pair through
    // `get_bigint` unconditionally allocates five digit blocks and runs a
    // divmod to reduce two machine words.
    if let [a, b] = args
        && let (Some(a), Some(b)) = (index_abs_machine_word(*a), index_abs_machine_word(*b))
    {
        return Ok(w_int_new(majit_rlib::rbigint::gcd_binary(a, b)));
    }
    // RPython's GC transform roots this running rbigint across the next
    // argument's potentially user-defined `__index__` call.
    let mut result = RBigIntGcRoot::new(BigInt::zero());
    for &arg in args {
        *result = result.gcd(&get_bigint(arg)?).map_err(map_rbigint_err)?;
    }
    Ok(bigint_to_pyint(&result))
}

pub fn lcm(args: &[PyObjectRef]) -> PyResult {
    let args = no_keywords(args, "lcm")?;
    if args.is_empty() {
        return Ok(w_int_new(1));
    }
    // app_math.py keeps `res` live while each later `index()` can execute
    // arbitrary Python and collect.
    let mut result = RBigIntGcRoot::new(get_bigint(args[0])?);
    for &arg in &args[1..] {
        // Every argument goes through `__index__` even once the running result
        // is zero: `math_lcm_impl` only short-circuits the arithmetic, so
        // `math.lcm(0, 1.5)` still raises TypeError.  `app_math.lcm` returns
        // early instead and skips the remaining conversions.
        let value = get_bigint(arg)?;
        if result.is_zero() {
            continue;
        }
        if value.is_zero() {
            *result = BigInt::zero();
            continue;
        }
        let divisor = result.gcd(&value).map_err(map_rbigint_err)?;
        *result = result
            .floordiv(&divisor)
            .map_err(map_rbigint_err)?
            .mul(&value)
            .abs();
    }
    let result = result.abs();
    Ok(bigint_to_pyint(&result))
}

/// `w_int_new` when the value fits an i64, else `w_long_new`.
fn bigint_to_pyint(b: &BigInt) -> PyObjectRef {
    if jit_bigint_to_i64_fits(b) != 0 {
        w_int_new(jit_bigint_to_i64_value(b))
    } else {
        w_long_new(b.translated_alias())
    }
}

/// The value of an operand that is already a machine int.  `None` covers
/// everything `space.index` would have to run to answer — a long, or an
/// object with `__index__`.
fn machine_word_int(obj: PyObjectRef) -> Option<i64> {
    unsafe { pyre_object::is_int(obj).then(|| pyre_object::w_int_get_value(obj)) }
}

/// `comb(n, k)` in the machine-word domain, for `0 <= k <= n`.
///
/// `None` is the direction that replays the pair in the rbigint domain: an
/// intermediate that leaves the machine range.  Each step is the exact
/// `C(n, i-1) * (n - i + 1) / i = C(n, i)`, so the running value is a real
/// binomial coefficient throughout and only the last multiplication before
/// the answer itself grows out of range can overflow.
fn comb_machine_word(n: i64, k: i64) -> Option<i64> {
    let k = k.min(n - k);
    let mut result: i64 = 1;
    for i in 1..=k {
        result = result.checked_mul(n - i + 1)?;
        result /= i;
    }
    Some(result)
}

/// `perm(n, k)` in the machine-word domain, for `0 <= k <= n`: the falling
/// factorial `n * (n-1) * ... * (n-k+1)`.  `None` replays the pair in the
/// rbigint domain.
fn perm_machine_word(n: i64, k: i64) -> Option<i64> {
    let mut result: i64 = 1;
    for i in 0..k {
        result = result.checked_mul(n - i)?;
    }
    Some(result)
}

pub fn comb(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "comb() takes exactly two arguments",
        ));
    }
    // `get_bigint` allocates a digit block per operand before the reduction
    // below allocates another per multiplication and per divmod.  A pair of
    // machine ints answers the same value with neither.  The two rejections
    // keep their order, so `comb(-1, -1)` still names `n`.
    if let [n, k] = args
        && let (Some(n), Some(k)) = (machine_word_int(*n), machine_word_int(*k))
    {
        if n < 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "n must be a non-negative integer",
            ));
        }
        if k < 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "k must be a non-negative integer",
            ));
        }
        if k > n {
            return Ok(w_int_new(0));
        }
        if let Some(result) = comb_machine_word(n, k) {
            return Ok(w_int_new(result));
        }
    }
    // `n` is an unboxed rbigint local across `index(k)`, exactly the kind of
    // local rooted automatically by RPython's GC transform.
    let n_big = RBigIntGcRoot::new(get_bigint(args[0])?);
    let k_big = get_bigint(args[1])?;

    if n_big.int_lt(0) {
        return Err(pyre_interpreter::PyError::value_error(
            "n must be a non-negative integer",
        ));
    }
    if k_big.int_lt(0) {
        return Err(pyre_interpreter::PyError::value_error(
            "k must be a non-negative integer",
        ));
    }

    if k_big.gt(&n_big) {
        return Ok(w_int_new(0));
    }

    let n_minus_k = &*n_big - &k_big;
    let k = if n_minus_k.lt(&k_big) {
        n_minus_k
    } else {
        k_big
    };
    if k.is_zero() {
        return Ok(w_int_new(1));
    }

    // pypy/module/math/app_math.py:comb — preserve its occasional fraction
    // reduction, including a bigint loop index.
    let mut numerator = n_big.translated_alias();
    let mut denominator = BigInt::one();
    let mut i = BigInt::one();
    while i.lt(&k) {
        numerator = numerator.mul(&n_big.sub(&i));
        denominator = denominator.mul(&i.int_add(1));
        if i.int_and_(15).is_zero() {
            numerator = numerator.floordiv(&denominator).map_err(map_rbigint_err)?;
            denominator = BigInt::one();
        }
        i = i.int_add(1);
    }
    Ok(bigint_to_pyint(
        &numerator.floordiv(&denominator).map_err(map_rbigint_err)?,
    ))
}

pub fn perm(args: &[PyObjectRef]) -> PyResult {
    let args = no_keywords(args, "perm")?;
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "perm() takes at least 1 argument",
        ));
    }
    if args.len() > 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "perm() takes at most 2 arguments",
        ));
    }
    // `perm(n, k)` over machine ints whose falling factorial stays in range
    // answers without a digit block per multiplication, the same way `comb`
    // does.  `perm(n)` and `perm(n, None)` mean k = n, which only fits for
    // n <= 20 and otherwise falls through.
    if let Some(n) = args.first().copied().and_then(machine_word_int)
        && let Some(k) = match args.get(1).copied() {
            None => Some(n),
            Some(k) if unsafe { pyre_object::is_none(k) } => Some(n),
            Some(k) => machine_word_int(k),
        }
    {
        if n < 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "n must be a non-negative integer",
            ));
        }
        if k < 0 {
            return Err(pyre_interpreter::PyError::value_error(
                "k must be a non-negative integer",
            ));
        }
        if k > n {
            return Ok(w_int_new(0));
        }
        if let Some(result) = perm_machine_word(n, k) {
            return Ok(w_int_new(result));
        }
    }
    // Keep `n` rooted while a non-None `k` invokes its `__index__`.
    let n_big = RBigIntGcRoot::new(get_bigint(args[0])?);
    if n_big.int_lt(0) {
        return Err(pyre_interpreter::PyError::value_error(
            "n must be a non-negative integer",
        ));
    }
    // perm(n, None) means k = n (factorial).
    let k_big = if args.len() >= 2 && !unsafe { pyre_object::is_none(args[1]) } {
        Some(get_bigint(args[1])?)
    } else {
        None
    };
    if let Some(ref k_val) = k_big {
        if k_val.int_lt(0) {
            return Err(pyre_interpreter::PyError::value_error(
                "k must be a non-negative integer",
            ));
        }
        if k_val > &n_big {
            return Ok(w_int_new(0));
        }
    }
    let k = k_big.unwrap_or_else(|| n_big.translated_alias());

    fn product_range(low: &BigInt, high: &BigInt, gap: &BigInt) -> Result<BigInt, RBigIntError> {
        if low.add(gap).ge(high) {
            let mut result = BigInt::one();
            let mut i = low.translated_alias();
            while i.lt(high) {
                result = result.mul(&i);
                i = i.int_add(1);
            }
            return Ok(result);
        }
        let mid = low.add(high).rshift(1, false)?;
        Ok(product_range(low, &mid, gap)?.mul(&product_range(&mid, high, gap)?))
    }

    let low = n_big.sub(&k).int_add(1);
    let high = n_big.int_add(1);
    let result = if k.int_le(100) {
        product_range(&low, &high, &BigInt::fromint(100))
    } else {
        let shifted = k.rshift(7, false).map_err(map_rbigint_err)?;
        let gap = if shifted.int_lt(100) {
            BigInt::fromint(100)
        } else {
            shifted
        };
        product_range(&low, &high, &gap)
    }
    .map_err(map_rbigint_err)?;
    Ok(bigint_to_pyint(&result))
}

pub fn isqrt(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "isqrt() takes exactly 1 argument",
        ));
    }
    let n = get_bigint(args[0])?;
    let value = n.isqrt().map_err(|_| {
        pyre_interpreter::PyError::value_error("isqrt() argument must be nonnegative")
    })?;
    Ok(bigint_to_pyint(&value))
}

pub fn fsum(args: &[PyObjectRef]) -> PyResult {
    // interp_math.py fsum: consume the iterator once while maintaining the
    // partials array.  The old port first materialized every boxed element and
    // then built a second Vec<f64>; besides diverging from upstream, that kept
    // one shadow-stack root per input alive until the entire iterable had been
    // consumed.
    let _roots = pyre_object::gc_roots::push_roots();
    let iterable_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args[0]);
    let w_iter = pyre_interpreter::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(
        iterable_slot,
    ))?;
    let iter_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_iter);

    let mut inf_sum = 0.0;
    let mut special_sum = 0.0;
    let mut partials: Vec<f64> = Vec::new();
    loop {
        let w_value = match pyre_interpreter::baseobjspace::next(
            pyre_object::gc_roots::shadow_stack_get(iter_slot),
        ) {
            Ok(value) => value,
            Err(err) if err.matches_stop_iteration() => break,
            Err(err) => return Err(err),
        };
        // `_get_double` can invoke user code.  Keep the yielded object rooted
        // only across that conversion, exactly like the translated livevar at
        // this point in the upstream loop.
        let original = {
            let _value_root = pyre_object::gc_roots::push_roots();
            let value_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_value);
            try_get_double(pyre_object::gc_roots::shadow_stack_get(value_slot))?
        };
        let mut value = original;
        let mut added = 0;
        for index in 0..partials.len() {
            let mut partial = partials[index];
            if value.abs() < partial.abs() {
                std::mem::swap(&mut value, &mut partial);
            }
            let hi = value + partial;
            let yr = hi - value;
            let lo = partial - yr;
            if lo != 0.0 {
                partials[added] = lo;
                added += 1;
            }
            value = hi;
        }
        partials.truncate(added);
        if value != 0.0 {
            if !value.is_finite() {
                if original.is_finite() {
                    return map_err(Err(pymath::Error::ERANGE));
                }
                if original.is_infinite() {
                    inf_sum += original;
                }
                special_sum += original;
                partials.clear();
            } else {
                partials.push(value);
            }
        }
    }

    if special_sum != 0.0 {
        if inf_sum.is_nan() {
            return map_err(Err(pymath::Error::EDOM));
        }
        return Ok(floatobject::w_float_new(special_sum));
    }
    let mut hi = 0.0;
    let mut lo = 0.0;
    let mut index = partials.len();
    if index > 0 {
        index -= 1;
        hi = partials[index];
        while index > 0 {
            index -= 1;
            let value = hi;
            let partial = partials[index];
            hi = value + partial;
            let yr = hi - value;
            lo = partial - yr;
            if lo != 0.0 {
                break;
            }
        }
        if index > 0
            && ((lo < 0.0 && partials[index - 1] < 0.0) || (lo > 0.0 && partials[index - 1] > 0.0))
        {
            let doubled = lo * 2.0;
            let value = hi + doubled;
            let yr = value - hi;
            if doubled == yr {
                hi = value;
            }
        }
    }
    Ok(floatobject::w_float_new(hi))
}

pub fn prod(args: &[PyObjectRef]) -> PyResult {
    // math.prod(iterable, *, start=1) — PyPy: pypy/module/math/interp_math.py
    // prod iterates with `space.mul` and returns the accumulated product.
    // `start` is keyword-only; positional `start` raises TypeError.
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "prod() takes at least 1 argument",
        ));
    }
    // Detect the __pyre_kw__ dict tail used by CALL_KW for builtin
    // functions with keyword arguments. PyPy: Arguments.parse_into_scope
    // splits positional from keyword before the call; pyre's dispatch
    // leaves them combined, so we unpack here.
    let is_kwargs = unsafe {
        let last = *args.last().unwrap();
        pyre_object::is_dict(last)
            && pyre_object::w_dict_getitem_str(last, "__pyre_kw__")
                .is_some_and(pyre_object::kw_marker::is_kw_marker_sentinel)
    };
    let (positional, start) = if is_kwargs {
        let kwargs = *args.last().unwrap();
        // `prod(iterable, /, *, start=1)` — `start` is the only accepted
        // keyword; any other is an unexpected-keyword TypeError.
        for (k, _) in unsafe { pyre_object::w_dict_items(kwargs) } {
            let name = unsafe { pyre_object::w_str_get_wtf8(k) };
            match name.as_str() {
                Ok("__pyre_kw__") | Ok("start") => {}
                _ => {
                    return Err(pyre_interpreter::PyError::type_error(format!(
                        "prod() got an unexpected keyword argument '{name}'"
                    )));
                }
            }
        }
        let start_key = pyre_object::unicodeobject::intern_str_value("start");
        let start =
            unsafe { pyre_object::w_dict_lookup(kwargs, start_key) }.unwrap_or(w_int_new(1));
        (&args[..args.len() - 1], start)
    } else if args.len() >= 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "prod() takes only one positional argument (the iterable)",
        ));
    } else {
        (&args[..1], w_int_new(1))
    };
    if positional.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "prod() takes at least 1 argument",
        ));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let acc_slot = pyre_object::gc_roots::pin_roots(&[start, positional[0]]);
    let items = pyre_interpreter::builtins::collect_iterable(
        pyre_object::gc_roots::shadow_stack_get(acc_slot + 1),
    )?;
    // `collect_iterable` returns unrooted pointers. Publish the whole
    // slice before any forwarding query; a per-item `pin_root` is a
    // safepoint that can move the still-unpinned tail.
    let items_base = pyre_object::gc_roots::pin_roots(&items);
    // Multiplication can call `__mul__`; reload both operands after every
    // collection and keep the running product in its rooted slot.
    for index in 0..items.len() {
        let product = pyre_interpreter::baseobjspace::mul(
            pyre_object::gc_roots::shadow_stack_get(acc_slot),
            pyre_object::gc_roots::shadow_stack_get(items_base + index),
        )?;
        pyre_object::gc_roots::shadow_stack_set(acc_slot, product);
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(acc_slot))
}

/// math.sumprod(p, q) — multiply paired elements, then sum. Added in
/// CPython 3.12. PyPy equivalent: not yet landed; here we follow
/// mathmodule.c `math_sumprod_impl` semantics using the generic
/// `space.mul` + `space.add` loop.
pub fn sumprod(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "sumprod() takes exactly 2 arguments",
        ));
    }
    let p = pyre_interpreter::builtins::collect_iterable(args[0])?;
    // Collecting the second input runs its iterator, so publish the first
    // materialized sequence before that call. `pin_roots` writes every
    // pointer before the first forwarding query.
    let _roots = pyre_object::gc_roots::push_roots();
    let p_base = pyre_object::gc_roots::pin_roots(&p);
    let q = pyre_interpreter::builtins::collect_iterable(args[1])?;
    // `mul` and `add` dispatch to the operands' `__mul__` / `__add__`, so a
    // Decimal or Fraction element makes every turn a collection point.  Both
    // collected sequences and the running total are native locals no root
    // walker updates, so publish them and read each operand back per turn.
    let q_base = pyre_object::gc_roots::pin_roots(&q);
    if p.len() != q.len() {
        return Err(pyre_interpreter::PyError::value_error(
            "Inputs are not the same length",
        ));
    }
    // The accumulator starts as int 0 so type coercion follows the pure
    // Python `total = 0; total += p_i * q_i` recipe: int stays int, and a
    // Decimal/Fraction/float product widens the running total on first add.
    let acc_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_int_new(0));
    for index in 0..p.len() {
        let prod = pyre_interpreter::baseobjspace::mul(
            pyre_object::gc_roots::shadow_stack_get(p_base + index),
            pyre_object::gc_roots::shadow_stack_get(q_base + index),
        )?;
        // `prod` is fresh and `add` runs Python; it needs a root of its own for
        // that call, released again each turn so the bracket stays fixed-size.
        let iteration_roots = pyre_object::gc_roots::push_roots();
        let prod_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = iteration_roots.pin_root(prod);
        let total = pyre_interpreter::baseobjspace::add(
            pyre_object::gc_roots::shadow_stack_get(acc_slot),
            pyre_object::gc_roots::shadow_stack_get(prod_slot),
        )?;
        drop(iteration_roots);
        pyre_object::gc_roots::shadow_stack_set(acc_slot, total);
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(acc_slot))
}

pub fn frexp(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "frexp() takes exactly 1 argument",
        ));
    }
    let (m, e) = pymath::math::frexp(try_get_double(args[0])?);
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(floatobject::w_float_new(m));
    fields.push(w_int_new(e as i64));
    Ok(w_tuple_new(fields.take()))
}

pub fn ldexp(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "ldexp() takes exactly 2 arguments",
        ));
    }
    // PyPy: pypy/module/math/interp_math.py::ldexp — second argument
    // must be an integer (via `__index__`), not a float.
    // interp_math.py::ldexp evaluates x before converting the exponent.
    // Besides preserving callback order, this avoids retaining an unboxed
    // exponent rbigint across x.__float__.
    let x = try_get_double(args[0])?;
    let exp_big = get_bigint(args[1])?;
    // Short-circuit special cases so an overflowing exponent doesn't
    // mask inf/nan propagation.
    if x.is_nan() {
        return Ok(floatobject::w_float_new(x));
    }
    if x.is_infinite() || x == 0.0 {
        return Ok(floatobject::w_float_new(x));
    }
    // Clamp the exponent to i32 range. Out-of-range exponents either
    // underflow to 0 (negative, finite x) or overflow to OverflowError.
    let exp = if jit_bigint_to_i64_fits(&exp_big) != 0 {
        i32::try_from(jit_bigint_to_i64_value(&exp_big)).ok()
    } else {
        None
    };
    let Some(exp) = exp else {
        // Sign of the exponent decides the result shape.
        if exp_big.int_lt(0) {
            let signed = if x.is_sign_positive() { 0.0 } else { -0.0 };
            return Ok(floatobject::w_float_new(signed));
        }
        return Err(pyre_interpreter::PyError::overflow_error(
            "math range error",
        ));
    };
    map_err(pymath::math::ldexp(x, exp))
}

pub fn modf(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "modf() takes exactly 1 argument",
        ));
    }
    let (frac, integer) = pymath::math::modf(try_get_double(args[0])?);
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(floatobject::w_float_new(frac));
    fields.push(floatobject::w_float_new(integer));
    Ok(w_tuple_new(fields.take()))
}

pub fn nextafter(args: &[PyObjectRef]) -> PyResult {
    let is_kwargs = !args.is_empty()
        && unsafe {
            let last = *args.last().unwrap();
            pyre_object::is_dict(last)
                && pyre_object::w_dict_getitem_str(last, "__pyre_kw__")
                    .is_some_and(pyre_object::kw_marker::is_kw_marker_sentinel)
        };
    let (pos, kwargs) = if is_kwargs {
        (&args[..args.len() - 1], Some(*args.last().unwrap()))
    } else {
        (args, None)
    };
    if pos.len() != 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "nextafter() takes exactly 2 positional arguments",
        ));
    }
    let steps = match kwargs.and_then(|kw| unsafe { pyre_object::w_dict_getitem_str(kw, "steps") })
    {
        Some(s) => {
            use num_traits::ToPrimitive;
            let b = get_bigint(s)?;
            if b.int_lt(0) {
                return Err(pyre_interpreter::PyError::value_error(
                    "steps must be a non-negative integer",
                ));
            }
            Some(if jit_bigint_to_u64_fits(&b) != 0 {
                jit_bigint_to_u64_value(&b)
            } else {
                u64::MAX
            })
        }
        None => None,
    };
    Ok(floatobject::w_float_new(pymath::math::nextafter(
        try_get_double(pos[0])?,
        try_get_double(pos[1])?,
        steps,
    )))
}

pub fn fma(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 3 {
        return Err(pyre_interpreter::PyError::type_error(
            "fma() takes exactly 3 arguments",
        ));
    }
    map_err(pymath::math::fma(
        try_get_double(args[0])?,
        try_get_double(args[1])?,
        try_get_double(args[2])?,
    ))
}
