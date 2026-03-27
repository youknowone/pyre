//! Math function implementations — delegating to pymath crate.
//!
//! PyPy equivalent: pypy/module/math/interp_math.py
//!
//! All functions delegate to `pymath::math` for CPython-exact results.

use pyre_object::*;

/// Extract f64 from an int, float, or long object.
/// PyPy: `_get_double(space, w_x)`
pub fn get_double(obj: PyObjectRef) -> f64 {
    unsafe {
        if is_int(obj) {
            return w_int_get_value(obj) as f64;
        }
        if is_float(obj) {
            return floatobject::w_float_get_value(obj);
        }
        if is_long(obj) {
            use num_traits::ToPrimitive;
            return w_long_get_value(obj).to_f64().unwrap_or(f64::NAN);
        }
        if is_bool(obj) {
            return if w_bool_get_value(obj) { 1.0 } else { 0.0 };
        }
    }
    0.0
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
            map_err(pymath::math::$name(get_double(args[0])))
        }
    };
}

macro_rules! pm1_plain {
    ($name:ident) => {
        pub fn $name(args: &[PyObjectRef]) -> PyResult {
            Ok(floatobject::w_float_new(pymath::math::$name(get_double(
                args[0],
            ))))
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
pm1!(log10);
pm1!(log2);

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
            map_err(pymath::math::$name(
                get_double(args[0]),
                get_double(args[1]),
            ))
        }
    };
}

pm2!(pow);
pm2!(fmod);
pm2!(copysign);
pm2!(remainder);

pub fn atan2(args: &[PyObjectRef]) -> PyResult {
    map_err(pymath::math::atan2(
        get_double(args[0]),
        get_double(args[1]),
    ))
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
    Ok(floatobject::w_float_new(pymath::math::dist(&p, &q)))
}

// ── Integer-returning functions ──────────────────────────────────────

pub fn floor(args: &[PyObjectRef]) -> PyResult {
    Ok(w_int_new(get_double(args[0]).floor() as i64))
}

pub fn ceil(args: &[PyObjectRef]) -> PyResult {
    Ok(w_int_new(get_double(args[0]).ceil() as i64))
}

pub fn trunc(args: &[PyObjectRef]) -> PyResult {
    Ok(w_int_new(get_double(args[0]).trunc() as i64))
}

// ── Special signatures ──────────────────────────────────────────────

pub fn log(args: &[PyObjectRef]) -> PyResult {
    let base = if args.len() >= 2 {
        Some(get_double(args[1]))
    } else {
        None
    };
    map_err(pymath::math::log(get_double(args[0]), base))
}

pub fn degrees(args: &[PyObjectRef]) -> PyResult {
    Ok(floatobject::w_float_new(pymath::math::degrees(get_double(
        args[0],
    ))))
}

pub fn radians(args: &[PyObjectRef]) -> PyResult {
    Ok(floatobject::w_float_new(pymath::math::radians(get_double(
        args[0],
    ))))
}

pub fn isinf(args: &[PyObjectRef]) -> PyResult {
    Ok(w_bool_from(pymath::math::isinf(get_double(args[0]))))
}

pub fn isnan(args: &[PyObjectRef]) -> PyResult {
    Ok(w_bool_from(pymath::math::isnan(get_double(args[0]))))
}

pub fn isfinite(args: &[PyObjectRef]) -> PyResult {
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
    let n = get_int(args[0]);
    match pymath::math::integer::factorial(n) {
        Ok(v) => {
            use num_traits::ToPrimitive;
            match v.to_i64() {
                Some(i) => Ok(w_int_new(i)),
                None => Ok(w_long_new(malachite_bigint::BigInt::from(v))),
            }
        }
        Err(e) => Err(crate::PyError::value_error(format!("{e:?}"))),
    }
}

pub fn gcd(args: &[PyObjectRef]) -> PyResult {
    use malachite_bigint::BigInt;
    let refs: Vec<BigInt> = args.iter().map(|&a| BigInt::from(get_int(a))).collect();
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
    let refs: Vec<BigInt> = args.iter().map(|&a| BigInt::from(get_int(a))).collect();
    let ref_slices: Vec<&BigInt> = refs.iter().collect();
    let result = pymath::math::integer::lcm(&ref_slices);
    use num_traits::ToPrimitive;
    Ok(result
        .to_i64()
        .map(w_int_new)
        .unwrap_or_else(|| w_long_new(result)))
}

pub fn comb(args: &[PyObjectRef]) -> PyResult {
    let n = get_int(args[0]);
    let k = get_int(args[1]);
    match pymath::math::integer::comb(n, k) {
        Ok(v) => {
            use num_traits::ToPrimitive;
            match v.to_i64() {
                Some(i) => Ok(w_int_new(i)),
                None => Ok(w_long_new(malachite_bigint::BigInt::from(v))),
            }
        }
        Err(e) => Err(crate::PyError::value_error(format!("{e:?}"))),
    }
}

pub fn perm(args: &[PyObjectRef]) -> PyResult {
    let n = get_int(args[0]);
    let k = if args.len() >= 2 {
        Some(get_int(args[1]))
    } else {
        None
    };
    match pymath::math::integer::perm(n, k) {
        Ok(v) => {
            use num_traits::ToPrimitive;
            match v.to_i64() {
                Some(i) => Ok(w_int_new(i)),
                None => Ok(w_long_new(malachite_bigint::BigInt::from(v))),
            }
        }
        Err(e) => Err(crate::PyError::value_error(format!("{e:?}"))),
    }
}

pub fn isqrt(args: &[PyObjectRef]) -> PyResult {
    use malachite_bigint::BigInt;
    let n = BigInt::from(get_int(args[0]));
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
    let items = crate::builtins::collect_iterable(args[0])?;
    let floats: Vec<f64> = items.iter().map(|&a| get_double(a)).collect();
    let start = args.get(1).map(|&a| get_double(a));
    Ok(floatobject::w_float_new(pymath::math::prod(floats, start)))
}

pub fn frexp(args: &[PyObjectRef]) -> PyResult {
    let (m, e) = pymath::math::frexp(get_double(args[0]));
    Ok(w_tuple_new(vec![
        floatobject::w_float_new(m),
        w_int_new(e as i64),
    ]))
}

pub fn ldexp(args: &[PyObjectRef]) -> PyResult {
    map_err(pymath::math::ldexp(
        get_double(args[0]),
        get_int(args[1]) as i32,
    ))
}

pub fn modf(args: &[PyObjectRef]) -> PyResult {
    let (frac, integer) = pymath::math::modf(get_double(args[0]));
    Ok(w_tuple_new(vec![
        floatobject::w_float_new(frac),
        floatobject::w_float_new(integer),
    ]))
}

pub fn nextafter(args: &[PyObjectRef]) -> PyResult {
    Ok(floatobject::w_float_new(pymath::math::nextafter(
        get_double(args[0]),
        get_double(args[1]),
        None,
    )))
}

pub fn fma(args: &[PyObjectRef]) -> PyResult {
    map_err(pymath::math::fma(
        get_double(args[0]),
        get_double(args[1]),
        get_double(args[2]),
    ))
}
