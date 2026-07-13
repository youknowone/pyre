//! Math function implementations — delegating to pymath crate.
//!
//! PyPy equivalent: pypy/module/math/interp_math.py
//!
//! All functions delegate to `pymath::math` for CPython-exact results.

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
pub fn try_get_double(obj: PyObjectRef) -> Result<f64, crate::PyError> {
    unsafe {
        if is_int(obj) {
            return Ok(w_int_get_value(obj) as f64);
        }
        if is_float(obj) {
            return Ok(floatobject::w_float_get_value(obj));
        }
        if is_long(obj) {
            // A Python int is always finite, so a non-finite conversion means
            // the magnitude exceeds f64 range — PyFloat_AsDouble raises here.
            let v = jit_bigint_to_f64_or_nan(w_long_get_value(obj));
            if !v.is_finite() {
                return Err(crate::PyError::overflow_error(
                    "int too large to convert to float",
                ));
            }
            return Ok(v);
        }
        if is_bool(obj) {
            return Ok(if w_bool_get_value(obj) { 1.0 } else { 0.0 });
        }
    }
    // `__float__` is a type-only special-method lookup (`space.lookup`); an
    // instance attribute named `__float__` is not consulted. A raising
    // `__float__` (descriptor `__get__` or the call itself) propagates
    // instead of being reported as "must be real number".
    match unsafe { crate::baseobjspace::lookup_special(obj, "__float__") } {
        Ok(Some(method)) => {
            let result = crate::builtins::call_and_check(method, &[])?;
            unsafe {
                if is_float(result) {
                    // A strict `float` subclass is accepted but deprecated;
                    // an exact `float` is used as-is.
                    if !is_exact_type(result, &FLOAT_TYPE) {
                        let value_type = crate::type_methods::arg_type_name(obj);
                        let result_type = crate::type_methods::arg_type_name(result);
                        crate::warn::warn_deprecation(&format!(
                            "{value_type}.__float__ returned non-float (type {result_type}).  \
                             The ability to return an instance of a strict subclass of \
                             float is deprecated, and may be removed in a future version \
                             of Python."
                        ));
                    }
                    return Ok(floatobject::w_float_get_value(result));
                }
            }
            // descroperation.py:891 — a non-float result (including int/long)
            // is rejected rather than coerced.
            let result_type = unsafe { pyre_object::type_name_of(result) };
            return Err(crate::PyError::type_error(format!(
                "__float__ returned non-float (type '{result_type}')",
            )));
        }
        Ok(None) => {}
        Err(err) => return Err(err),
    }
    match crate::baseobjspace::getattr_str(obj, "__index__") {
        Ok(method) => {
            let result = crate::builtins::call_and_check(method, &[])?;
            unsafe {
                if is_int(result) {
                    return Ok(w_int_get_value(result) as f64);
                }
                if is_long(result) {
                    let v = jit_bigint_to_f64_or_nan(w_long_get_value(result));
                    if !v.is_finite() {
                        return Err(crate::PyError::overflow_error(
                            "int too large to convert to float",
                        ));
                    }
                    return Ok(v);
                }
            }
        }
        Err(err) if err.kind != crate::PyErrorKind::AttributeError => return Err(err),
        Err(_) => {}
    }
    Err(crate::PyError::type_error("must be real number"))
}

type PyResult = Result<PyObjectRef, crate::PyError>;

fn map_err(r: pymath::Result<f64>) -> PyResult {
    match r {
        Ok(v) => Ok(floatobject::w_float_new(v)),
        Err(pymath::Error::EDOM) => Err(crate::PyError::value_error("math domain error")),
        Err(pymath::Error::ERANGE) => Err(crate::PyError::overflow_error("math range error")),
    }
}

fn map_int_err(e: pymath::Error) -> crate::PyError {
    match e {
        pymath::Error::EDOM => crate::PyError::value_error("math domain error"),
        pymath::Error::ERANGE => crate::PyError::overflow_error("math range error"),
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
        crate::display::format_float_repr(val)
    }
}

// ── 1-arg float→float via pymath ─────────────────────────────────────

macro_rules! pm1 {
    ($name:ident) => {
        pub fn $name(args: &[PyObjectRef]) -> PyResult {
            if args.len() != 1 {
                return Err(crate::PyError::type_error(concat!(
                    stringify!($name),
                    "() takes exactly one argument"
                )));
            }
            map_err(pymath::math::$name(try_get_double(args[0])?))
        }
    };
}

/// Like `pm1!`, but an `EDOM` result becomes a value-carrying message
/// ("<prefix>, got <repr>") instead of the generic "math domain error".
macro_rules! pm1_edom {
    ($name:ident, $prefix:literal) => {
        pub fn $name(args: &[PyObjectRef]) -> PyResult {
            if args.len() != 1 {
                return Err(crate::PyError::type_error(concat!(
                    stringify!($name),
                    "() takes exactly one argument"
                )));
            }
            let val = try_get_double(args[0])?;
            match pymath::math::$name(val) {
                Ok(v) => Ok(floatobject::w_float_new(v)),
                Err(pymath::Error::EDOM) => Err(crate::PyError::value_error(format!(
                    concat!($prefix, ", got {}"),
                    float_repr(val)
                ))),
                Err(pymath::Error::ERANGE) => {
                    Err(crate::PyError::overflow_error("math range error"))
                }
            }
        }
    };
}

macro_rules! pm1_plain {
    ($name:ident) => {
        pub fn $name(args: &[PyObjectRef]) -> PyResult {
            if args.len() != 1 {
                return Err(crate::PyError::type_error(concat!(
                    stringify!($name),
                    "() takes exactly one argument"
                )));
            }
            Ok(floatobject::w_float_new(pymath::math::$name(
                try_get_double(args[0])?,
            )))
        }
    };
}

// Trigonometric
pm1_edom!(sin, "expected a finite input");
pm1_edom!(cos, "expected a finite input");
pm1_edom!(tan, "expected a finite input");
pm1_edom!(asin, "expected a number in range from -1 up to 1");
pm1_edom!(acos, "expected a number in range from -1 up to 1");
pm1!(atan);
pm1!(sinh);
pm1!(cosh);
pm1!(tanh);
pm1!(asinh);
pm1!(acosh);
pm1_edom!(atanh, "expected a number between -1 and 1");

// Exponential / logarithmic
pm1_edom!(sqrt, "expected a nonnegative input");
pm1!(cbrt);
pm1!(exp);
pm1!(exp2);
pm1!(expm1);
pm1!(log1p);

// Gamma / error
pm1!(erf);
pm1!(erfc);
pm1_edom!(gamma, "expected a noninteger or positive integer");
pm1!(lgamma);

// Misc
pm1!(fabs);
pm1_plain!(ulp);

// ── 2-arg float→float via pymath ─────────────────────────────────────

macro_rules! pm2 {
    ($name:ident) => {
        pub fn $name(args: &[PyObjectRef]) -> PyResult {
            if args.len() != 2 {
                return Err(crate::PyError::type_error(concat!(
                    stringify!($name),
                    "() takes exactly 2 arguments"
                )));
            }
            let x = try_get_double(args[0])?;
            let y = try_get_double(args[1])?;
            map_err(pymath::math::$name(x, y))
        }
    };
}

pm2!(pow);
pm2!(fmod);
pm2!(copysign);
pm2!(remainder);

pub fn atan2(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(
            "atan2() takes exactly 2 arguments",
        ));
    }
    let x = try_get_double(args[0])?;
    let y = try_get_double(args[1])?;
    map_err(pymath::math::atan2(x, y))
}

pub fn hypot(args: &[PyObjectRef]) -> PyResult {
    let coords: Vec<f64> = args
        .iter()
        .map(|&a| try_get_double(a))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(floatobject::w_float_new(pymath::math::hypot(&coords)))
}

pub fn dist(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(
            "dist() takes exactly 2 arguments",
        ));
    }
    let p: Vec<f64> = crate::builtins::collect_iterable(args[0])?
        .iter()
        .map(|&a| try_get_double(a))
        .collect::<Result<Vec<_>, _>>()?;
    let q: Vec<f64> = crate::builtins::collect_iterable(args[1])?
        .iter()
        .map(|&a| try_get_double(a))
        .collect::<Result<Vec<_>, _>>()?;
    if p.len() != q.len() {
        return Err(crate::PyError::value_error(
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
        return Err(crate::PyError::type_error(format!(
            "{fname}() takes exactly 1 argument",
        )));
    }
    // If the descriptor itself raises (e.g. BadDescr.__get__ → ValueError),
    // propagate that error rather than silently falling back to float.
    match unsafe { crate::baseobjspace::lookup_special(args[0], dunder) } {
        Ok(Some(method)) => {
            crate::call::clear_call_error();
            let result = crate::call_function(method, &[]);
            if !result.is_null() {
                return Ok(result);
            }
            if let Some(err) = crate::call::take_call_error() {
                return Err(err);
            }
        }
        Ok(None) => {}
        Err(err) => return Err(err),
    }
    if !fallback_float {
        return Err(crate::PyError::type_error(format!(
            "type {} doesn't define {fname}() method",
            crate::baseobjspace::object_functionstr_type_name(args[0])
        )));
    }
    // Fall back to `__float__` coercion — `try_get_double` raises TypeError
    // when the operand has no numeric interpretation.
    let v = try_get_double(args[0])
        .map_err(|_| crate::PyError::type_error(format!("type has no {fname}() method")))?;
    Ok(w_int_new(match dunder {
        "__ceil__" => v.ceil() as i64,
        "__floor__" => v.floor() as i64,
        _ => v.trunc() as i64,
    }))
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

/// Compute `log(n)` for arbitrarily large integers by bit-shifting off the
/// top 53 bits into an f64 mantissa and adding `e * SHIFT * log(2)`.
///
/// PyPy: rpython/rlib/rbigint.py::_loghelper —
///     x, e = _AsScaledDouble(arg)
///     return func(x) + e * SHIFT * func(2.0)
///
/// Here we pick SHIFT=1 so `e` is the number of bits shifted off.
fn bigint_log(n: &malachite_bigint::BigInt, base: f64) -> Result<f64, crate::PyError> {
    use num_traits::{Signed, ToPrimitive};
    if !n.is_positive() {
        return Err(crate::PyError::value_error("math domain error"));
    }
    // Extract bit length and shift down so the value fits in an f64 mantissa.
    let bits = n.bits() as usize;
    let shift = if bits > 60 { bits - 60 } else { 0 };
    let shifted = n >> shift;
    let x: f64 = jit_bigint_to_f64_or_inf(&shifted);
    if !x.is_finite() {
        return Err(crate::PyError::overflow_error("int too large"));
    }
    // log(n) = log(x) + shift * log(2)
    let log_x = if base == 10.0 {
        x.log10()
    } else if base == 2.0 {
        x.log2()
    } else {
        x.ln()
    };
    let log_two = if base == 10.0 {
        2f64.log10()
    } else if base == 2.0 {
        1.0 // log2(2) = 1
    } else {
        2f64.ln()
    };
    let mut result = log_x + shift as f64 * log_two;
    // If base != 0 and != {e, 10, 2}, divide by log(base).
    if base != 0.0 && base != 10.0 && base != 2.0 {
        result /= base.ln();
    }
    Ok(result)
}

/// Special-case integer arguments to avoid overflow, and give the domain
/// error the value-carrying message except for an int argument, whose error
/// carries no value.
fn log_any(w_x: PyObjectRef, base: f64) -> PyResult {
    use num_traits::Signed;
    unsafe {
        if pyre_object::is_bool(w_x) || pyre_object::is_int(w_x) || pyre_object::is_long(w_x) {
            let num_owned;
            let num: &malachite_bigint::BigInt = if pyre_object::is_long(w_x) {
                pyre_object::w_long_get_value(w_x)
            } else if pyre_object::is_bool(w_x) {
                num_owned =
                    malachite_bigint::BigInt::from(pyre_object::w_bool_get_value(w_x) as i64);
                &num_owned
            } else {
                num_owned = malachite_bigint::BigInt::from(pyre_object::w_int_get_value(w_x));
                &num_owned
            };
            if !num.is_positive() {
                return Err(crate::PyError::value_error("expected a positive input"));
            }
            return Ok(floatobject::w_float_new(bigint_log(num, base)?));
        }
    }
    let x = try_get_double(w_x)?;
    // NaN propagates through log.
    if x.is_nan() {
        return Ok(floatobject::w_float_new(f64::NAN));
    }
    // Domain error for x <= 0 (but x == +inf is fine).
    if x <= 0.0 {
        return Err(crate::PyError::value_error(format!(
            "expected a positive input, got {}",
            float_repr(x)
        )));
    }
    if base == 10.0 {
        Ok(floatobject::w_float_new(x.log10()))
    } else if base == 2.0 {
        Ok(floatobject::w_float_new(x.log2()))
    } else if base == 0.0 {
        Ok(floatobject::w_float_new(x.ln()))
    } else {
        Ok(floatobject::w_float_new(x.ln() / base.ln()))
    }
}

pub fn log(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() || args.len() > 2 {
        return Err(crate::PyError::type_error("log() takes 1 or 2 arguments"));
    }
    let base = if args.len() >= 2 {
        // The base is validated before the argument, so log(x, base) with a
        // non-positive base reports the base rather than the argument.
        let b = try_get_double(args[1])?;
        if b <= 0.0 {
            return Err(crate::PyError::value_error(format!(
                "expected a positive input, got {}",
                float_repr(b)
            )));
        }
        if b == 1.0 {
            return Err(crate::PyError::value_error("math domain error"));
        }
        b
    } else {
        0.0
    };
    log_any(args[0], base)
}

pub fn log10(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(
            "log10() takes exactly 1 argument",
        ));
    }
    log_any(args[0], 10.0)
}

pub fn log2(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(
            "log2() takes exactly 1 argument",
        ));
    }
    log_any(args[0], 2.0)
}

pub fn degrees(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "degrees() takes exactly 1 argument",
        ));
    }
    Ok(floatobject::w_float_new(pymath::math::degrees(
        try_get_double(args[0])?,
    )))
}

pub fn radians(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "radians() takes exactly 1 argument",
        ));
    }
    Ok(floatobject::w_float_new(pymath::math::radians(
        try_get_double(args[0])?,
    )))
}

pub fn isinf(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "isinf() takes exactly 1 argument",
        ));
    }
    Ok(w_bool_from(pymath::math::isinf(try_get_double(args[0])?)))
}

pub fn isnan(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "isnan() takes exactly 1 argument",
        ));
    }
    Ok(w_bool_from(pymath::math::isnan(try_get_double(args[0])?)))
}

pub fn isfinite(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "isfinite() takes exactly 1 argument",
        ));
    }
    Ok(w_bool_from(pymath::math::isfinite(try_get_double(
        args[0],
    )?)))
}

pub fn isclose(args: &[PyObjectRef]) -> PyResult {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(
            "isclose() takes exactly 2 positional arguments",
        ));
    }
    // `rel_tol` and `abs_tol` are the only (keyword-only) parameters.
    crate::builtins::kwarg_reject_unknown(kwargs, &["rel_tol", "abs_tol"], "isclose")?;
    let read = |name: &str| -> Result<Option<f64>, crate::PyError> {
        match crate::builtins::kwarg_get(kwargs, name) {
            Some(v) => Ok(Some(try_get_double(v)?)),
            None => Ok(None),
        }
    };
    let rel_tol = read("rel_tol")?;
    let abs_tol = read("abs_tol")?;
    match pymath::math::isclose(
        try_get_double(pos[0])?,
        try_get_double(pos[1])?,
        rel_tol,
        abs_tol,
    ) {
        Ok(v) => Ok(w_bool_from(v)),
        Err(e) => Err(map_int_err(e)),
    }
}

pub fn factorial(args: &[PyObjectRef]) -> PyResult {
    use malachite_bigint::BigInt;
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "factorial() takes exactly 1 argument",
        ));
    }
    // PyPy: pypy/module/math/app_math.py factorial — reject floats that aren't
    // exact integers, and negative x.
    unsafe {
        if pyre_object::is_float(args[0]) {
            return Err(crate::PyError::type_error(
                "factorial() only accepts integral values",
            ));
        }
    }
    let n_big = get_bigint(args[0])?;
    use num_traits::Signed;
    if n_big.is_negative() {
        return Err(crate::PyError::value_error(
            "factorial() not defined for negative values",
        ));
    }
    let n = if jit_bigint_to_i64_fits(&n_big) != 0 {
        jit_bigint_to_i64_value(&n_big)
    } else {
        return Err(crate::PyError::overflow_error(
            "factorial() argument should not exceed i64::MAX",
        ));
    };
    // Straightforward BigInt multiplication; overflow impossible with
    // arbitrary precision. Faster algorithms (binary split) exist in
    // pypy/module/math/app_math.py but structural correctness is what
    // matters here.
    let mut result = BigInt::from(1);
    for i in 2..=n {
        result *= BigInt::from(i);
    }
    if jit_bigint_to_i64_fits(&result) != 0 {
        Ok(w_int_new(jit_bigint_to_i64_value(&result)))
    } else {
        Ok(w_long_new(result))
    }
}

/// Convert any int/long/bool to a BigInt for math.gcd/lcm/factorial
/// overflow-safe handling. PyPy: space.bigint_w() which traverses the
/// W_IntObject/W_LongObject/W_BoolObject union and materializes rbigint.
/// Raises TypeError for non-integer inputs via `__index__` dunder, matching
/// CPython's `_PyLong_FromNbIndexOrNbInt`.
fn get_bigint(obj: PyObjectRef) -> Result<malachite_bigint::BigInt, crate::PyError> {
    use malachite_bigint::BigInt;
    unsafe {
        if pyre_object::is_long(obj) {
            return Ok(pyre_object::w_long_get_value(obj).clone());
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
            return Err(crate::PyError::type_error(
                "'float' object cannot be interpreted as an integer",
            ));
        }
    }
    // __index__ dunder — descroperation.py `_index`: type-only special-method
    // lookup, then propagate a raising `__index__` instead of masking it with
    // the generic "object cannot be interpreted as an integer".
    match unsafe { crate::baseobjspace::lookup_special(obj, "__index__") } {
        Ok(Some(method)) => {
            let result = crate::builtins::call_and_check(method, &[])?;
            unsafe {
                if pyre_object::is_int(result) {
                    return Ok(BigInt::from(pyre_object::w_int_get_value(result)));
                }
                if pyre_object::is_long(result) {
                    return Ok(pyre_object::w_long_get_value(result).clone());
                }
            }
            // descroperation.py:612 — __index__ returned non-int (type %T)
            let result_type = unsafe { (*(*result).ob_type).name };
            return Err(crate::PyError::type_error(format!(
                "__index__ returned non-int (type '{result_type}')",
            )));
        }
        Ok(None) => {}
        Err(err) => return Err(err),
    }
    Err(crate::PyError::type_error(
        "object cannot be interpreted as an integer",
    ))
}

pub fn gcd(args: &[PyObjectRef]) -> PyResult {
    use malachite_bigint::BigInt;
    let refs: Vec<BigInt> = args
        .iter()
        .map(|&a| get_bigint(a))
        .collect::<Result<Vec<_>, _>>()?;
    let ref_slices: Vec<&BigInt> = refs.iter().collect();
    let result = pymath::math::integer::gcd(&ref_slices);
    if jit_bigint_to_i64_fits(&result) != 0 {
        Ok(w_int_new(jit_bigint_to_i64_value(&result)))
    } else {
        Ok(w_long_new(result))
    }
}

pub fn lcm(args: &[PyObjectRef]) -> PyResult {
    use malachite_bigint::BigInt;
    let refs: Vec<BigInt> = args
        .iter()
        .map(|&a| get_bigint(a))
        .collect::<Result<Vec<_>, _>>()?;
    let ref_slices: Vec<&BigInt> = refs.iter().collect();
    let result = pymath::math::integer::lcm(&ref_slices);
    if jit_bigint_to_i64_fits(&result) != 0 {
        Ok(w_int_new(jit_bigint_to_i64_value(&result)))
    } else {
        Ok(w_long_new(result))
    }
}

/// `w_int_new` when the value fits an i64, else `w_long_new`.
fn bigint_to_pyint(b: malachite_bigint::BigInt) -> PyObjectRef {
    use num_traits::ToPrimitive;
    match b.to_i64() {
        Some(i) => w_int_new(i),
        None => w_long_new(b),
    }
}

fn biguint_to_pyint(b: malachite_bigint::BigUint) -> PyObjectRef {
    bigint_to_pyint(malachite_bigint::BigInt::from(b))
}

pub fn comb(args: &[PyObjectRef]) -> PyResult {
    use num_traits::{Signed, ToPrimitive};

    if args.len() != 2 {
        return Err(crate::PyError::type_error(
            "comb() takes exactly two arguments",
        ));
    }
    let n_big = get_bigint(args[0])?;
    let k_big = get_bigint(args[1])?;

    if n_big.is_negative() {
        return Err(crate::PyError::value_error(
            "n must be a non-negative integer",
        ));
    }
    if k_big.is_negative() {
        return Err(crate::PyError::value_error(
            "k must be a non-negative integer",
        ));
    }

    // Fast path: n fits in i64.
    if let Some(ni) = n_big.to_i64() {
        // k out of range [0, n] means the result is 0.
        let ki = match k_big.to_i64() {
            Some(k) if (0..=ni).contains(&k) => k,
            _ => return Ok(w_int_new(0)),
        };
        // Symmetry C(n, k) == C(n, n-k): compute with the smaller index.
        let ki = ki.min(ni - ki);
        if ki > 1 {
            let v = pymath::math::integer::comb(ni, ki).map_err(map_int_err)?;
            return Ok(biguint_to_pyint(v));
        }
        if ki == 0 {
            return Ok(w_int_new(1));
        }
        return Ok(bigint_to_pyint(n_big)); // ki == 1
    }

    // BigInt path: n does not fit i64. Reduce by symmetry, then the smaller
    // index must fit u64 for the divide-and-conquer product.
    let n_minus_k = &n_big - &k_big;
    if n_minus_k.is_negative() {
        return Ok(w_int_new(0));
    }
    let effective_k = if n_minus_k < k_big {
        &n_minus_k
    } else {
        &k_big
    };
    // `comb_bigint` takes the smaller factor as a `u64`; a value past
    // `u64::MAX` is unreachable for any computable input (the product would
    // need >2**64 multiplications) and is rejected rather than truncated.
    let ki: u64 = match effective_k.to_u64() {
        Some(k) => k,
        None => {
            return Err(crate::PyError::overflow_error(format!(
                "min(n - k, k) must not exceed {}",
                u64::MAX
            )));
        }
    };
    Ok(biguint_to_pyint(pymath::math::comb_bigint(&n_big, ki)))
}

pub fn perm(args: &[PyObjectRef]) -> PyResult {
    use num_traits::{Signed, ToPrimitive};

    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "perm() takes at least 1 argument",
        ));
    }
    if args.len() > 2 {
        return Err(crate::PyError::type_error(
            "perm() takes at most 2 arguments",
        ));
    }
    let n_big = get_bigint(args[0])?;
    if n_big.is_negative() {
        return Err(crate::PyError::value_error(
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
        if k_val.is_negative() {
            return Err(crate::PyError::value_error(
                "k must be a non-negative integer",
            ));
        }
        if k_val > &n_big {
            return Ok(w_int_new(0));
        }
    }
    // k (falling-factorial length) must fit u64 for the product.
    let ki: u64 = match &k_big {
        None => n_big.to_u64().ok_or_else(|| {
            crate::PyError::overflow_error(format!("n must not exceed {}", u64::MAX))
        })?,
        Some(k_val) => k_val.to_u64().ok_or_else(|| {
            crate::PyError::overflow_error(format!("k must not exceed {}", u64::MAX))
        })?,
    };
    // Fast path: n fits in i64 and k > 1 uses the i64 kernel.
    if let Some(ni) = n_big.to_i64() {
        if ni >= 0 && ki > 1 {
            let v = pymath::math::integer::perm(ni, Some(ki as i64)).map_err(map_int_err)?;
            return Ok(biguint_to_pyint(v));
        }
    }
    Ok(biguint_to_pyint(pymath::math::perm_bigint(&n_big, ki)))
}

pub fn isqrt(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "isqrt() takes exactly 1 argument",
        ));
    }
    let n = get_bigint(args[0])?;
    match pymath::math::integer::isqrt(&n) {
        Ok(v) => {
            use num_traits::ToPrimitive;
            match v.to_i64() {
                Some(i) => Ok(w_int_new(i)),
                None => Ok(w_long_new(v)),
            }
        }
        Err(e) => Err(map_int_err(e)),
    }
}

pub fn fsum(args: &[PyObjectRef]) -> PyResult {
    let items = crate::builtins::collect_iterable(args[0])?;
    let floats: Vec<f64> = items
        .iter()
        .map(|&a| try_get_double(a))
        .collect::<Result<Vec<_>, _>>()?;
    map_err(pymath::math::fsum(floats))
}

pub fn prod(args: &[PyObjectRef]) -> PyResult {
    // math.prod(iterable, *, start=1) — PyPy: pypy/module/math/interp_math.py
    // prod iterates with `space.mul` and returns the accumulated product.
    // `start` is keyword-only; positional `start` raises TypeError.
    if args.is_empty() {
        return Err(crate::PyError::type_error(
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
            && pyre_object::w_dict_lookup(last, pyre_object::w_str_new("__pyre_kw__"))
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
                    return Err(crate::PyError::type_error(format!(
                        "prod() got an unexpected keyword argument '{name}'"
                    )));
                }
            }
        }
        let start_key = pyre_object::w_str_new("start");
        let start =
            unsafe { pyre_object::w_dict_lookup(kwargs, start_key) }.unwrap_or(w_int_new(1));
        (&args[..args.len() - 1], start)
    } else if args.len() >= 2 {
        return Err(crate::PyError::type_error(
            "prod() takes only one positional argument (the iterable)",
        ));
    } else {
        (&args[..1], w_int_new(1))
    };
    if positional.is_empty() {
        return Err(crate::PyError::type_error(
            "prod() takes at least 1 argument",
        ));
    }
    let iterable = positional[0];
    let items = crate::builtins::collect_iterable(iterable)?;
    let mut acc = start;
    for item in items {
        acc = crate::baseobjspace::mul(acc, item)?;
    }
    Ok(acc)
}

/// math.sumprod(p, q) — multiply paired elements, then sum. Added in
/// CPython 3.12. PyPy equivalent: not yet landed; here we follow
/// mathmodule.c `math_sumprod_impl` semantics using the generic
/// `space.mul` + `space.add` loop.
pub fn sumprod(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(
            "sumprod() takes exactly 2 arguments",
        ));
    }
    let p = crate::builtins::collect_iterable(args[0])?;
    let q = crate::builtins::collect_iterable(args[1])?;
    if p.len() != q.len() {
        return Err(crate::PyError::value_error(
            "Inputs are not the same length",
        ));
    }
    // The accumulator starts as int 0 so type coercion follows the pure
    // Python `total = 0; total += p_i * q_i` recipe: int stays int, and a
    // Decimal/Fraction/float product widens the running total on first add.
    let mut acc: PyObjectRef = w_int_new(0);
    for (a, b) in p.iter().zip(q.iter()) {
        let prod = crate::baseobjspace::mul(*a, *b)?;
        acc = crate::baseobjspace::add(acc, prod)?;
    }
    Ok(acc)
}

pub fn frexp(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "frexp() takes exactly 1 argument",
        ));
    }
    let (m, e) = pymath::math::frexp(try_get_double(args[0])?);
    Ok(w_tuple_new(vec![
        floatobject::w_float_new(m),
        w_int_new(e as i64),
    ]))
}

pub fn ldexp(args: &[PyObjectRef]) -> PyResult {
    use num_traits::ToPrimitive;
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "ldexp() takes exactly 2 arguments",
        ));
    }
    // PyPy: pypy/module/math/interp_math.py::ldexp — second argument
    // must be an integer (via `__index__`), not a float.
    let exp_big = get_bigint(args[1])?;
    let x = try_get_double(args[0])?;
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
    let exp = match exp_big.to_i32() {
        Some(v) => v,
        None => {
            // Sign of the exponent decides the result shape.
            if exp_big.sign() == malachite_bigint::Sign::Minus {
                let signed = if x.is_sign_positive() { 0.0 } else { -0.0 };
                return Ok(floatobject::w_float_new(signed));
            }
            return Err(crate::PyError::overflow_error("math range error"));
        }
    };
    map_err(pymath::math::ldexp(x, exp))
}

pub fn modf(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "modf() takes exactly 1 argument",
        ));
    }
    let (frac, integer) = pymath::math::modf(try_get_double(args[0])?);
    Ok(w_tuple_new(vec![
        floatobject::w_float_new(frac),
        floatobject::w_float_new(integer),
    ]))
}

pub fn nextafter(args: &[PyObjectRef]) -> PyResult {
    let is_kwargs = !args.is_empty()
        && unsafe {
            let last = *args.last().unwrap();
            pyre_object::is_dict(last)
                && pyre_object::w_dict_lookup(last, pyre_object::w_str_new("__pyre_kw__")).is_some()
        };
    let (pos, kwargs) = if is_kwargs {
        (&args[..args.len() - 1], Some(*args.last().unwrap()))
    } else {
        (&args[..], None)
    };
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(
            "nextafter() takes exactly 2 positional arguments",
        ));
    }
    let steps = match kwargs
        .and_then(|kw| unsafe { pyre_object::w_dict_lookup(kw, pyre_object::w_str_new("steps")) })
    {
        Some(s) => {
            use num_traits::ToPrimitive;
            let b = get_bigint(s)?;
            if b.sign() == malachite_bigint::Sign::Minus {
                return Err(crate::PyError::value_error(
                    "steps must be a non-negative integer",
                ));
            }
            Some(b.to_u64().unwrap_or(u64::MAX))
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
        return Err(crate::PyError::type_error(
            "fma() takes exactly 3 arguments",
        ));
    }
    map_err(pymath::math::fma(
        try_get_double(args[0])?,
        try_get_double(args[1])?,
        try_get_double(args[2])?,
    ))
}
