//! Math function implementations.
//!
//! PyPy equivalent: pypy/module/math/interp_math.py
//!
//! Each function follows PyPy's pattern:
//!   - `_get_double(space, w_x)` → extract f64 from int/float/long
//!   - `math1(space, f, w_x)` → 1-arg wrapper that calls `f(x)` and returns float
//!   - `math2(space, f, w_x, w_y)` → 2-arg wrapper

use pyre_object::*;

/// Extract f64 from an int, float, or long object.
///
/// PyPy equivalent: `_get_double(space, w_x)` in interp_math.py
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
    }
    panic!("a float is required")
}

// ── macro wrappers matching PyPy's math1 / math2 patterns ────────────

macro_rules! math1 {
    ($name:ident, $f:expr) => {
        pub fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            assert!(
                args.len() == 1,
                concat!(stringify!($name), "() takes exactly one argument")
            );
            let x = get_double(args[0]);
            Ok(floatobject::w_float_new($f(x)))
        }
    };
}

macro_rules! math2 {
    ($name:ident, $f:expr) => {
        pub fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            assert!(
                args.len() == 2,
                concat!(stringify!($name), "() takes exactly two arguments")
            );
            let x = get_double(args[0]);
            let y = get_double(args[1]);
            Ok(floatobject::w_float_new($f(x, y)))
        }
    };
}

// ── 1-argument functions (PyPy: math1 wrappers in interp_math.py) ────

math1!(sqrt, f64::sqrt);
math1!(sin, f64::sin);
math1!(cos, f64::cos);
math1!(tan, f64::tan);
math1!(asin, f64::asin);
math1!(acos, f64::acos);
math1!(atan, f64::atan);
math1!(sinh, f64::sinh);
math1!(cosh, f64::cosh);
math1!(tanh, f64::tanh);
math1!(asinh, f64::asinh);
math1!(acosh, f64::acosh);
math1!(atanh, f64::atanh);
math1!(exp, f64::exp);
math1!(expm1, |x: f64| x.exp_m1());
math1!(log1p, |x: f64| x.ln_1p());
math1!(fabs, f64::abs);
math1!(erf, |x: f64| libm::erf(x));
math1!(erfc, |x: f64| libm::erfc(x));
math1!(gamma, |x: f64| libm::tgamma(x));
math1!(lgamma, |x: f64| libm::lgamma(x));

// ── 2-argument functions (PyPy: math2 wrappers in interp_math.py) ────

math2!(pow, f64::powf);
math2!(atan2, f64::atan2);
math2!(hypot, f64::hypot);
math2!(copysign, f64::copysign);
math2!(fmod, |x: f64, y: f64| x % y);

// ── functions with non-standard signatures ───────────────────────────

pub fn floor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "floor() takes exactly one argument");
    let x = get_double(args[0]);
    Ok(w_int_new(x.floor() as i64))
}

pub fn ceil(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "ceil() takes exactly one argument");
    let x = get_double(args[0]);
    Ok(w_int_new(x.ceil() as i64))
}

/// PyPy equivalent: interp_math.trunc
pub fn trunc(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "trunc() takes exactly one argument");
    let x = get_double(args[0]);
    Ok(w_int_new(x.trunc() as i64))
}

/// PyPy equivalent: interp_math.log (supports optional base argument)
pub fn log(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let x = get_double(args[0]);
    let result = if args.len() == 2 {
        let base = get_double(args[1]);
        x.ln() / base.ln()
    } else {
        x.ln()
    };
    Ok(floatobject::w_float_new(result))
}

pub fn log2(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "log2() takes exactly one argument");
    Ok(floatobject::w_float_new(get_double(args[0]).log2()))
}

pub fn log10(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "log10() takes exactly one argument");
    Ok(floatobject::w_float_new(get_double(args[0]).log10()))
}

/// PyPy equivalent: interp_math.degrees
pub fn degrees(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "degrees() takes exactly one argument");
    Ok(floatobject::w_float_new(get_double(args[0]).to_degrees()))
}

/// PyPy equivalent: interp_math.radians
pub fn radians(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "radians() takes exactly one argument");
    Ok(floatobject::w_float_new(get_double(args[0]).to_radians()))
}

/// PyPy equivalent: interp_math.isinf
pub fn isinf(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "isinf() takes exactly one argument");
    Ok(w_bool_from(get_double(args[0]).is_infinite()))
}

/// PyPy equivalent: interp_math.isnan
pub fn isnan(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "isnan() takes exactly one argument");
    Ok(w_bool_from(get_double(args[0]).is_nan()))
}

pub fn isfinite(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "isfinite() takes exactly one argument");
    Ok(w_bool_from(get_double(args[0]).is_finite()))
}

/// PyPy equivalent: app_math.factorial (app-level in PyPy, interp-level here)
pub fn factorial(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "factorial() takes exactly one argument");
    let n = unsafe { w_int_get_value(args[0]) };
    assert!(n >= 0, "factorial() not defined for negative values");
    let mut result: i64 = 1;
    for i in 2..=n {
        result = result.checked_mul(i).expect("factorial overflow");
    }
    Ok(w_int_new(result))
}

pub fn gcd(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "gcd() takes exactly two arguments");
    let mut a = unsafe { w_int_get_value(args[0]) }.unsigned_abs();
    let mut b = unsafe { w_int_get_value(args[1]) }.unsigned_abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    Ok(w_int_new(a as i64))
}
