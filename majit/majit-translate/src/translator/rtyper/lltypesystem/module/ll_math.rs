//! RPython `rpython/rtyper/lltypesystem/module/ll_math.py` parity helpers.
//!
//! Upstream also owns the C `rffi.llexternal` registration layer here.  This
//! Rust module exposes the semantic helper names that the codewriter already
//! refers to (`ll_math_fmod`, `ll_math_sqrt`, `ll_math_pow`, ...), keeping the
//! module path aligned while the full `rffi` external-function registry remains
//! unported.

#![allow(non_upper_case_globals)]

use std::fmt;

pub const use_library_isinf_isnan: bool = false;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MathExternal {
    pub name: String,
}

pub fn math_llexternal(name: &str) -> MathExternal {
    MathExternal {
        name: name.to_owned(),
    }
}

pub const UNARY_MATH_FUNCTIONS: &[&str] = &[
    "acos", "asin", "atan", "ceil", "cosh", "exp", "fabs", "sinh", "tan", "tanh", "acosh", "asinh",
    "atanh", "expm1",
];

pub const UNARY_MATH_FUNCTIONS_CAN_OVERFLOW: &[&str] = &["cosh", "exp", "sinh", "expm1"];

pub const UNARY_MATH_FUNCTIONS_C99: &[&str] = &["acosh", "asinh", "atanh", "expm1"];

pub const unary_math_functions: &[&str] = UNARY_MATH_FUNCTIONS;
pub const unary_math_functions_can_overflow: &[&str] = UNARY_MATH_FUNCTIONS_CAN_OVERFLOW;
pub const unary_math_functions_c99: &[&str] = UNARY_MATH_FUNCTIONS_C99;

const VERY_LARGE_FLOAT: f64 = f64::INFINITY;

pub const ERANGE: i32 = 34;
pub const EDOM: i32 = 33;
pub const INT_MAX: i64 = i32::MAX as i64;
pub const INT_MIN: i64 = i32::MIN as i64;

pub fn _lib_isnan(y: f64) -> i32 {
    ll_math_isnan(y) as i32
}

pub fn _lib_finite(y: f64) -> i32 {
    ll_math_isfinite(y) as i32
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MathError {
    ValueError(&'static str),
    OverflowError(&'static str),
}

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathError::ValueError(msg) => f.write_str(msg),
            MathError::OverflowError(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for MathError {}

pub fn ll_math_isnan(y: f64) -> bool {
    y != y
}

pub fn ll_math_isinf(y: f64) -> bool {
    (y + VERY_LARGE_FLOAT) == y || y == f64::NEG_INFINITY
}

pub fn ll_math_isfinite(y: f64) -> bool {
    (y - y) == 0.0
}

pub fn ll_math_floor(x: f64) -> f64 {
    x.floor()
}

pub fn sqrt_nonneg(x: f64) -> f64 {
    x.sqrt()
}

pub fn ll_math_copysign(x: f64, y: f64) -> f64 {
    x.copysign(y)
}

pub fn ll_math_atan2(y: f64, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }

    if !y.is_finite() {
        if y.is_nan() {
            return f64::NAN;
        }
        if x.is_infinite() {
            if ll_math_copysign(1.0, x) == 1.0 {
                return ll_math_copysign(0.25 * std::f64::consts::PI, y);
            }
            return ll_math_copysign(0.75 * std::f64::consts::PI, y);
        }
        return ll_math_copysign(0.5 * std::f64::consts::PI, y);
    }

    if x.is_infinite() || y == 0.0 {
        if ll_math_copysign(1.0, x) == 1.0 {
            return ll_math_copysign(0.0, y);
        }
        return ll_math_copysign(std::f64::consts::PI, y);
    }

    y.atan2(x)
}

pub fn ll_math_frexp(x: f64) -> (f64, i64) {
    if !x.is_finite() || x == 0.0 {
        return (x, 0);
    }

    let bits = x.to_bits();
    let sign = bits & (1_u64 << 63);
    let exponent = ((bits >> 52) & 0x7ff) as i64;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent == 0 {
        let scaled = x * ((1_u64 << 52) as f64);
        let (mantissa, exponent) = ll_math_frexp(scaled);
        return (mantissa, exponent - 52);
    }

    let mantissa_bits = sign | (1022_u64 << 52) | fraction;
    (f64::from_bits(mantissa_bits), exponent - 1022)
}

pub fn ll_math_ldexp(x: f64, exp: i64) -> Result<f64, MathError> {
    if x == 0.0 || !x.is_finite() {
        return Ok(x);
    }
    let r = x * 2.0_f64.powi(exp.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
    if r.is_infinite() {
        return Err(MathError::OverflowError("math range error"));
    }
    Ok(r)
}

pub fn ll_math_modf(x: f64) -> (f64, f64) {
    if !x.is_finite() {
        if x.is_nan() {
            return (x, x);
        }
        return (ll_math_copysign(0.0, x), x);
    }
    let intpart = x.trunc();
    (x - intpart, intpart)
}

pub fn ll_math_fmod(x: f64, y: f64) -> Result<f64, MathError> {
    if y.is_infinite() && x.is_finite() {
        return Ok(x);
    }

    let r = x % y;
    if r.is_nan() && !x.is_nan() && !y.is_nan() {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(r)
}

pub fn ll_math_hypot(x: f64, y: f64) -> Result<f64, MathError> {
    if x.is_infinite() {
        return Ok(x.abs());
    }
    if y.is_infinite() {
        return Ok(y.abs());
    }

    let r = x.hypot(y);
    if r.is_infinite() && x.is_finite() && y.is_finite() {
        return Err(MathError::OverflowError("math range error"));
    }
    if r.is_nan() && !x.is_nan() && !y.is_nan() {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(r)
}

pub fn ll_math_pow(x: f64, y: f64) -> Result<f64, MathError> {
    if y.is_nan() {
        if x == 1.0 {
            return Ok(1.0);
        }
        return Ok(y);
    }

    if !x.is_finite() {
        if x.is_nan() {
            if y == 0.0 {
                return Ok(1.0);
            }
            return Ok(x);
        }
        let odd_y = !y.is_infinite() && (y.abs() % 2.0) == 1.0;
        if y > 0.0 {
            return Ok(if odd_y { x } else { x.abs() });
        }
        if y == 0.0 {
            return Ok(1.0);
        }
        return Ok(if odd_y { ll_math_copysign(0.0, x) } else { 0.0 });
    }

    if y.is_infinite() {
        if x.abs() == 1.0 {
            return Ok(1.0);
        }
        if y > 0.0 && x.abs() > 1.0 {
            return Ok(y);
        }
        if y < 0.0 && x.abs() < 1.0 {
            if x == 0.0 {
                return Err(MathError::ValueError("0**-inf: divide by zero"));
            }
            return Ok(-y);
        }
        return Ok(0.0);
    }

    let r = x.powf(y);
    if r.is_nan() && !x.is_nan() && !y.is_nan() {
        return Err(MathError::ValueError("math domain error"));
    }
    if r.is_infinite() {
        if x == 0.0 {
            return Err(MathError::ValueError("math domain error"));
        }
        return Err(MathError::OverflowError("math range error"));
    }
    Ok(r)
}

pub fn ll_math_sqrt(x: f64) -> Result<f64, MathError> {
    if x < 0.0 {
        return Err(MathError::ValueError("math domain error"));
    }
    if x.is_finite() {
        return Ok(sqrt_nonneg(x));
    }
    Ok(x)
}

pub fn ll_math_log(x: f64) -> Result<f64, MathError> {
    if x <= 0.0 {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(x.ln())
}

pub fn ll_math_log10(x: f64) -> Result<f64, MathError> {
    if x <= 0.0 {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(x.log10())
}

pub fn ll_math_log1p(x: f64) -> Result<f64, MathError> {
    if x == 0.0 {
        return Ok(x);
    }
    if x <= -1.0 {
        if x == -1.0 {
            return Err(MathError::OverflowError("math range  error"));
        }
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(x.ln_1p())
}

pub fn ll_math_sin(x: f64) -> Result<f64, MathError> {
    if x.is_infinite() {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(x.sin())
}

pub fn ll_math_cos(x: f64) -> Result<f64, MathError> {
    if x.is_infinite() {
        return Err(MathError::ValueError("math domain error"));
    }
    Ok(x.cos())
}

fn check_unary_math(x: f64, r: f64, can_overflow: bool) -> Result<f64, MathError> {
    if r.is_nan() && !x.is_nan() {
        return Err(MathError::ValueError("math domain error"));
    }
    if can_overflow && r.is_infinite() && x.is_finite() {
        return Err(MathError::OverflowError("math range error"));
    }
    Ok(r)
}

pub fn ll_math_acos(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.acos(), false)
}

pub fn ll_math_asin(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.asin(), false)
}

pub fn ll_math_atan(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.atan(), false)
}

pub fn ll_math_ceil(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.ceil(), false)
}

pub fn ll_math_cosh(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.cosh(), true)
}

pub fn ll_math_exp(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.exp(), true)
}

pub fn ll_math_fabs(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.abs(), false)
}

pub fn ll_math_sinh(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.sinh(), true)
}

pub fn ll_math_tan(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.tan(), false)
}

pub fn ll_math_tanh(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.tanh(), false)
}

pub fn ll_math_acosh(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.acosh(), false)
}

pub fn ll_math_asinh(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.asinh(), false)
}

pub fn ll_math_atanh(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.atanh(), false)
}

pub fn ll_math_expm1(x: f64) -> Result<f64, MathError> {
    check_unary_math(x, x.exp_m1(), true)
}

pub type UnaryMathFunction = fn(f64) -> Result<f64, MathError>;

pub fn new_unary_math_function(
    name: &str,
    _can_overflow: bool,
    _c99: bool,
) -> Option<UnaryMathFunction> {
    match name {
        "acos" => Some(ll_math_acos),
        "asin" => Some(ll_math_asin),
        "atan" => Some(ll_math_atan),
        "ceil" => Some(ll_math_ceil),
        "cosh" => Some(ll_math_cosh),
        "exp" => Some(ll_math_exp),
        "fabs" => Some(ll_math_fabs),
        "sinh" => Some(ll_math_sinh),
        "tan" => Some(ll_math_tan),
        "tanh" => Some(ll_math_tanh),
        "acosh" => Some(ll_math_acosh),
        "asinh" => Some(ll_math_asinh),
        "atanh" => Some(ll_math_atanh),
        "expm1" => Some(ll_math_expm1),
        _ => None,
    }
}

pub fn _revdb_frexp(x: f64) -> (f64, i64) {
    ll_math_frexp(x)
}

pub fn _revdb_modf(x: f64) -> (f64, f64) {
    ll_math_modf(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_predicates_match_ll_math_branches() {
        assert!(ll_math_isnan(f64::NAN));
        assert!(ll_math_isinf(f64::INFINITY));
        assert!(ll_math_isinf(f64::NEG_INFINITY));
        assert!(!ll_math_isinf(1.0));
        assert!(ll_math_isfinite(1.0));
        assert!(!ll_math_isfinite(f64::INFINITY));
        assert!(!ll_math_isfinite(f64::NAN));
    }

    #[test]
    fn fmod_preserves_finite_x_for_infinite_y_and_errors_on_zero_divisor() {
        assert_eq!(ll_math_fmod(3.5, f64::INFINITY), Ok(3.5));
        assert_eq!(
            ll_math_fmod(3.5, 0.0),
            Err(MathError::ValueError("math domain error"))
        );
    }

    #[test]
    fn sqrt_log_and_log1p_surface_domain_errors() {
        assert_eq!(
            ll_math_sqrt(-1.0),
            Err(MathError::ValueError("math domain error"))
        );
        assert_eq!(
            ll_math_log(0.0),
            Err(MathError::ValueError("math domain error"))
        );
        assert_eq!(
            ll_math_log1p(-1.0),
            Err(MathError::OverflowError("math range  error"))
        );
    }

    #[test]
    fn pow_handles_upstream_ieee_special_cases() {
        assert_eq!(ll_math_pow(1.0, f64::NAN), Ok(1.0));
        assert_eq!(ll_math_pow(f64::NAN, 0.0), Ok(1.0));
        assert_eq!(ll_math_pow(f64::INFINITY, -2.0), Ok(0.0));
        assert_eq!(
            ll_math_pow(0.0, f64::NEG_INFINITY),
            Err(MathError::ValueError("0**-inf: divide by zero"))
        );
    }

    #[test]
    fn frexp_matches_python_shape_for_normal_and_special_values() {
        assert_eq!(ll_math_frexp(0.0), (0.0, 0));
        assert_eq!(ll_math_frexp(f64::INFINITY), (f64::INFINITY, 0));
        let (mantissa, exponent) = ll_math_frexp(8.0);
        assert_eq!(mantissa, 0.5);
        assert_eq!(exponent, 4);
    }

    #[test]
    fn generated_unary_math_symbols_surface_domain_and_overflow_checks() {
        assert_eq!(
            ll_math_acos(2.0),
            Err(MathError::ValueError("math domain error"))
        );
        assert_eq!(
            ll_math_exp(10000.0),
            Err(MathError::OverflowError("math range error"))
        );
        assert_eq!(ll_math_fabs(-2.5), Ok(2.5));
    }

    #[test]
    fn top_level_parity_surface_keeps_upstream_names() {
        assert!(!use_library_isinf_isnan);
        assert_eq!(ERANGE, 34);
        assert_eq!(EDOM, 33);
        assert_eq!(INT_MAX, i32::MAX as i64);
        assert_eq!(INT_MIN, i32::MIN as i64);
        assert_eq!(_lib_isnan(f64::NAN), 1);
        assert_eq!(_lib_finite(1.0), 1);
        assert_eq!(sqrt_nonneg(4.0), 2.0);
        assert_eq!(unary_math_functions, UNARY_MATH_FUNCTIONS);
        assert_eq!(
            unary_math_functions_can_overflow,
            UNARY_MATH_FUNCTIONS_CAN_OVERFLOW
        );
        assert_eq!(unary_math_functions_c99, UNARY_MATH_FUNCTIONS_C99);
        assert_eq!(math_llexternal("log1p").name, "log1p");

        let exp = new_unary_math_function("exp", true, false).expect("exp helper");
        assert_eq!(exp(0.0), Ok(1.0));
        assert!(new_unary_math_function("missing", false, false).is_none());
        assert_eq!(_revdb_frexp(8.0), ll_math_frexp(8.0));
        assert_eq!(_revdb_modf(1.25), ll_math_modf(1.25));
    }
}
