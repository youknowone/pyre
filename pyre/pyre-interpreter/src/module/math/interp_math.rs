//! Math function implementations — delegating to pymath crate.
//!
//! PyPy equivalent: pypy/module/math/interp_math.py
//!
//! All functions delegate to `pymath::math` for CPython-exact results.

use pyre_object::*;

/// Extract f64 from an int, float, or long object.
/// PyPy: `_get_double(space, w_x)` — falls back to `__float__` for
/// Fraction/Decimal/custom number types. Infallible callers should
/// use `get_double_or_default` which retains the legacy 0.0 fallback
/// for backward compatibility.
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
            use num_traits::ToPrimitive;
            return Ok(w_long_get_value(obj).to_f64().unwrap_or(f64::NAN));
        }
        if is_bool(obj) {
            return Ok(if w_bool_get_value(obj) { 1.0 } else { 0.0 });
        }
    }
    if let Ok(method) = crate::baseobjspace::getattr(obj, "__float__") {
        let result = crate::call_function(method, &[obj]);
        if !result.is_null() {
            unsafe {
                if is_float(result) {
                    return Ok(floatobject::w_float_get_value(result));
                }
                if is_int(result) {
                    return Ok(w_int_get_value(result) as f64);
                }
            }
        }
    }
    if let Ok(method) = crate::baseobjspace::getattr(obj, "__index__") {
        let result = crate::call_function(method, &[obj]);
        if !result.is_null() {
            unsafe {
                if is_int(result) {
                    return Ok(w_int_get_value(result) as f64);
                }
            }
        }
    }
    Err(crate::PyError::type_error("must be real number"))
}

fn get_int(obj: PyObjectRef) -> i64 {
    unsafe {
        if is_int(obj) {
            return w_int_get_value(obj);
        }
        if is_bool(obj) {
            return if w_bool_get_value(obj) { 1 } else { 0 };
        }
        if is_float(obj) {
            return floatobject::w_float_get_value(obj) as i64;
        }
    }
    0
}

type PyResult = Result<PyObjectRef, crate::PyError>;

fn map_err(r: pymath::Result<f64>) -> PyResult {
    match r {
        Ok(v) => Ok(floatobject::w_float_new(v)),
        Err(e) => Err(crate::PyError::value_error(format!(
            "math domain error: {e:?}"
        ))),
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
pm1!(sin);
pm1!(cos);
pm1!(tan);
pm1!(asin);
pm1!(acos);
pm1!(atan);
pm1!(sinh);
pm1!(cosh);
pm1!(tanh);
pm1!(asinh);
pm1!(acosh);
pm1!(atanh);

// Exponential / logarithmic
pm1!(sqrt);
pm1!(cbrt);
pm1!(exp);
pm1!(exp2);
pm1!(expm1);
pm1!(log1p);

// Gamma / error
pm1!(erf);
pm1!(erfc);
pm1!(gamma);
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
    let coords: Vec<f64> = args.iter().map(|&a| get_double(a)).collect();
    Ok(floatobject::w_float_new(pymath::math::hypot(&coords)))
}

pub fn dist(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("dist requires 2 arguments"));
    }
    let p: Vec<f64> = crate::builtins::collect_iterable(args[0])?
        .iter()
        .map(|&a| get_double(a))
        .collect();
    let q: Vec<f64> = crate::builtins::collect_iterable(args[1])?
        .iter()
        .map(|&a| get_double(a))
        .collect();
    if p.len() != q.len() {
        return Err(crate::PyError::value_error(
            "both points must have the same number of dimensions",
        ));
    }
    Ok(floatobject::w_float_new(pymath::math::dist(&p, &q)))
}

// ── Integer-returning functions ──────────────────────────────────────

/// Invoke `__ceil__`/`__floor__`/`__trunc__` or fall back to converting
/// the argument to a float via `__float__`. Raises TypeError when the
/// argument has no numeric interpretation — PyPy: mathmodule.c
/// math_1_impl's `double_from_object` routine.
fn math_unary_int(args: &[PyObjectRef], dunder: &str, fname: &str) -> PyResult {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "{fname}() takes exactly 1 argument",
        )));
    }
    // Prefer the dunder so subclasses of float with `__ceil__` use their
    // override even when the parent float path would also succeed. If the
    // descriptor itself raises (e.g. BadDescr.__get__ → ValueError),
    // propagate that error rather than silently falling back to float.
    match crate::baseobjspace::getattr(args[0], dunder) {
        Ok(method) => {
            crate::call::clear_call_error();
            let result = crate::call_function(method, &[args[0]]);
            if !result.is_null() {
                return Ok(result);
            }
            if let Some(err) = crate::call::take_call_error() {
                return Err(err);
            }
        }
        Err(err) if err.kind != crate::PyErrorKind::AttributeError => {
            return Err(err);
        }
        _ => {}
    }
    // Fall back to `__float__` coercion — mathmodule.c uses PyNumber_Float
    // in math_1_impl for this role. `try_get_double` raises TypeError
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
    math_unary_int(args, "__floor__", "floor")
}

pub fn ceil(args: &[PyObjectRef]) -> PyResult {
    math_unary_int(args, "__ceil__", "ceil")
}

pub fn trunc(args: &[PyObjectRef]) -> PyResult {
    math_unary_int(args, "__trunc__", "trunc")
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
    let x: f64 = shifted
        .to_f64()
        .ok_or_else(|| crate::PyError::overflow_error("int too large"))?;
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

/// PyPy: pypy/module/math/interp_math.py::_log_any —
/// special-case long arguments to avoid overflow in `get_double`.
fn log_any(w_x: PyObjectRef, base: f64) -> PyResult {
    unsafe {
        if pyre_object::is_long(w_x) {
            let num = pyre_object::w_long_get_value(w_x);
            let r = bigint_log(num, base)?;
            if base != 0.0 && base != 10.0 && base != 2.0 {
                if base <= 0.0 || base.is_nan() {
                    return Err(crate::PyError::value_error("math domain error"));
                }
            }
            return Ok(floatobject::w_float_new(r));
        }
    }
    let x = get_double(w_x);
    // NaN propagates through log.
    if x.is_nan() {
        return Ok(floatobject::w_float_new(f64::NAN));
    }
    // CPython: domain error for x <= 0 (but x == +inf is fine).
    if x <= 0.0 {
        return Err(crate::PyError::value_error("math domain error"));
    }
    if base == 10.0 {
        Ok(floatobject::w_float_new(x.log10()))
    } else if base == 2.0 {
        Ok(floatobject::w_float_new(x.log2()))
    } else if base == 0.0 {
        Ok(floatobject::w_float_new(x.ln()))
    } else {
        if base <= 0.0 || base.is_nan() {
            return Err(crate::PyError::value_error("math domain error"));
        }
        Ok(floatobject::w_float_new(x.ln() / base.ln()))
    }
}

pub fn log(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() || args.len() > 2 {
        return Err(crate::PyError::type_error("log() takes 1 or 2 arguments"));
    }
    let base = if args.len() >= 2 {
        get_double(args[1])
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
    Ok(floatobject::w_float_new(pymath::math::degrees(get_double(
        args[0],
    ))))
}

pub fn radians(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "radians() takes exactly 1 argument",
        ));
    }
    Ok(floatobject::w_float_new(pymath::math::radians(get_double(
        args[0],
    ))))
}

pub fn isinf(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "isinf() takes exactly 1 argument",
        ));
    }
    Ok(w_bool_from(pymath::math::isinf(get_double(args[0]))))
}

pub fn isnan(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "isnan() takes exactly 1 argument",
        ));
    }
    Ok(w_bool_from(pymath::math::isnan(get_double(args[0]))))
}

pub fn isfinite(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "isfinite() takes exactly 1 argument",
        ));
    }
    Ok(w_bool_from(pymath::math::isfinite(get_double(args[0]))))
}

pub fn isclose(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("isclose requires 2 arguments"));
    }
    let rel_tol = args.get(2).map(|&a| get_double(a));
    let abs_tol = args.get(3).map(|&a| get_double(a));
    match pymath::math::isclose(get_double(args[0]), get_double(args[1]), rel_tol, abs_tol) {
        Ok(v) => Ok(w_bool_from(v)),
        Err(e) => Err(crate::PyError::value_error(format!("{e:?}"))),
    }
}

pub fn factorial(args: &[PyObjectRef]) -> PyResult {
    use malachite_bigint::BigInt;
    use num_traits::ToPrimitive;
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
    let n = match n_big.to_i64() {
        Some(v) => v,
        None => {
            return Err(crate::PyError::overflow_error(
                "factorial() argument should not exceed i64::MAX",
            ));
        }
    };
    if n < 0 {
        return Err(crate::PyError::value_error(
            "factorial() not defined for negative values",
        ));
    }
    // Straightforward BigInt multiplication; overflow impossible with
    // arbitrary precision. Faster algorithms (binary split) exist in
    // pypy/module/math/app_math.py but structural correctness is what
    // matters here.
    let mut result = BigInt::from(1);
    for i in 2..=n {
        result *= BigInt::from(i);
    }
    Ok(result
        .to_i64()
        .map(w_int_new)
        .unwrap_or_else(|| w_long_new(result)))
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
    // __index__ dunder — PyPy: descroperation.py space.index.
    if let Ok(method) = crate::baseobjspace::getattr(obj, "__index__") {
        let result = crate::call_function(method, &[obj]);
        if !result.is_null() {
            unsafe {
                if pyre_object::is_int(result) {
                    return Ok(BigInt::from(pyre_object::w_int_get_value(result)));
                }
                if pyre_object::is_long(result) {
                    return Ok(pyre_object::w_long_get_value(result).clone());
                }
            }
        }
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
    use num_traits::ToPrimitive;
    Ok(result
        .to_i64()
        .map(w_int_new)
        .unwrap_or_else(|| w_long_new(result)))
}

pub fn lcm(args: &[PyObjectRef]) -> PyResult {
    use malachite_bigint::BigInt;
    let refs: Vec<BigInt> = args
        .iter()
        .map(|&a| get_bigint(a))
        .collect::<Result<Vec<_>, _>>()?;
    let ref_slices: Vec<&BigInt> = refs.iter().collect();
    let result = pymath::math::integer::lcm(&ref_slices);
    use num_traits::ToPrimitive;
    Ok(result
        .to_i64()
        .map(w_int_new)
        .unwrap_or_else(|| w_long_new(result)))
}

pub fn comb(args: &[PyObjectRef]) -> PyResult {
    use num_traits::ToPrimitive;
    if args.len() < 2 {
        return Err(crate::PyError::type_error("comb() takes 2 arguments"));
    }
    let n_big = get_bigint(args[0])?;
    let k_big = get_bigint(args[1])?;
    let n = n_big.to_i64().ok_or_else(|| {
        crate::PyError::overflow_error("comb() argument should not exceed i64::MAX")
    })?;
    let k = k_big.to_i64().ok_or_else(|| {
        crate::PyError::overflow_error("comb() argument should not exceed i64::MAX")
    })?;
    match pymath::math::integer::comb(n, k) {
        Ok(v) => match v.to_i64() {
            Some(i) => Ok(w_int_new(i)),
            None => Ok(w_long_new(malachite_bigint::BigInt::from(v))),
        },
        Err(e) => Err(crate::PyError::value_error(format!("{e:?}"))),
    }
}

pub fn perm(args: &[PyObjectRef]) -> PyResult {
    use num_traits::ToPrimitive;
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "perm() takes at least 1 argument",
        ));
    }
    let n_big = get_bigint(args[0])?;
    let n = n_big.to_i64().ok_or_else(|| {
        crate::PyError::overflow_error("perm() argument should not exceed i64::MAX")
    })?;
    // perm(n, None) means "unlimited" — treat as perm(n).
    let k = if args.len() >= 2 && !unsafe { pyre_object::is_none(args[1]) } {
        let k_big = get_bigint(args[1])?;
        Some(k_big.to_i64().ok_or_else(|| {
            crate::PyError::overflow_error("perm() argument should not exceed i64::MAX")
        })?)
    } else {
        None
    };
    match pymath::math::integer::perm(n, k) {
        Ok(v) => match v.to_i64() {
            Some(i) => Ok(w_int_new(i)),
            None => Ok(w_long_new(malachite_bigint::BigInt::from(v))),
        },
        Err(e) => Err(crate::PyError::value_error(format!("{e:?}"))),
    }
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
        Err(e) => Err(crate::PyError::value_error(format!("{e:?}"))),
    }
}

pub fn fsum(args: &[PyObjectRef]) -> PyResult {
    let items = crate::builtins::collect_iterable(args[0])?;
    let floats: Vec<f64> = items.iter().map(|&a| get_double(a)).collect();
    match pymath::math::fsum(floats) {
        Ok(v) => Ok(floatobject::w_float_new(v)),
        Err(e) => Err(crate::PyError::value_error(format!("{e:?}"))),
    }
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
            && pyre_object::w_dict_lookup(last, pyre_object::w_str_new("__pyre_kw__")).is_some()
    };
    let (positional, start) = if is_kwargs {
        let kwargs = *args.last().unwrap();
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
    if args.len() < 2 {
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
    let mut acc: PyObjectRef = floatobject::w_float_new(0.0);
    let mut all_int = true;
    for (a, b) in p.iter().zip(q.iter()) {
        unsafe {
            if !(pyre_object::is_int(*a) || pyre_object::is_long(*a))
                || !(pyre_object::is_int(*b) || pyre_object::is_long(*b))
            {
                all_int = false;
                break;
            }
        }
    }
    if all_int {
        acc = w_int_new(0);
    }
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
    let (m, e) = pymath::math::frexp(get_double(args[0]));
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
    let x = get_double(args[0]);
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
    let (frac, integer) = pymath::math::modf(get_double(args[0]));
    Ok(w_tuple_new(vec![
        floatobject::w_float_new(frac),
        floatobject::w_float_new(integer),
    ]))
}

pub fn nextafter(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "nextafter() takes at least 2 arguments",
        ));
    }
    Ok(floatobject::w_float_new(pymath::math::nextafter(
        get_double(args[0]),
        get_double(args[1]),
        None,
    )))
}

pub fn fma(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 3 {
        return Err(crate::PyError::type_error(
            "fma() takes exactly 3 arguments",
        ));
    }
    map_err(pymath::math::fma(
        get_double(args[0]),
        get_double(args[1]),
        get_double(args[2]),
    ))
}
