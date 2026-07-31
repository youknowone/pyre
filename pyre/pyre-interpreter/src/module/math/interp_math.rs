//! Math function implementations — delegating to pymath crate.
//!
//! PyPy equivalent: pypy/module/math/interp_math.py
//!
//! All functions delegate to `pymath::math` for CPython-exact results.

use pyre_object::rbigint::{RBigInt as BigInt, RBigIntError, RBigIntGcRoot};
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
                        ))?;
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

fn map_rbigint_err(e: RBigIntError) -> crate::PyError {
    match e {
        RBigIntError::Memory => crate::PyError::memory_error(""),
        RBigIntError::Overflow | RBigIntError::FloatDivisionOverflow => {
            crate::PyError::overflow_error("math range error")
        }
        RBigIntError::DivisionByZero => crate::PyError::zero_division("integer division by zero"),
        _ => crate::PyError::value_error("math domain error"),
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

static MATH_SQRT_WRAPPER: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static MATH_LOG_WRAPPER: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static MATH_COS_WRAPPER: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static MATH_SIN_WRAPPER: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static MATH_FREXP_WRAPPER: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static MATH_LDEXP_WRAPPER: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static MATH_ISQRT_WRAPPER: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Record the checked-arity wrapper pointers installed by `py_module!`.
///
/// `py_checked_arity_fn!` wraps each `interp_math` body in a non-capturing
/// closure, so a BuiltinCode stores the wrapper pointer rather than (for
/// example) [`frexp`] itself.  The wrappers are immutable process code, hence
/// process-global `OnceLock<usize>` is the same ownership shape as pyre's
/// other immortal runtime metadata and needs no GC rooting.
pub fn register_jit_builtin_wrappers(ns: PyObjectRef) {
    for (name, slot) in [
        ("sqrt", &MATH_SQRT_WRAPPER),
        ("log", &MATH_LOG_WRAPPER),
        ("cos", &MATH_COS_WRAPPER),
        ("sin", &MATH_SIN_WRAPPER),
        ("frexp", &MATH_FREXP_WRAPPER),
        ("ldexp", &MATH_LDEXP_WRAPPER),
        ("isqrt", &MATH_ISQRT_WRAPPER),
    ] {
        let callable = crate::module_ns_get(ns, name)
            .unwrap_or_else(|| panic!("math.{name} missing after module registration"));
        let wrapper = unsafe {
            let code = crate::function_get_code(callable) as PyObjectRef;
            debug_assert!(crate::gateway::is_builtin_code(code));
            crate::gateway::builtin_code_get(code) as usize
        };
        let installed = slot.get_or_init(|| wrapper);
        debug_assert_eq!(*installed, wrapper);
    }
}

unsafe fn math_builtin_wrapper_matches(
    callable: PyObjectRef,
    expected: &std::sync::OnceLock<usize>,
) -> bool {
    unsafe {
        if callable.is_null() || !crate::is_function(callable) {
            return false;
        }
        let code = crate::function_get_code(callable) as PyObjectRef;
        !code.is_null()
            && crate::gateway::is_builtin_code(code)
            && expected
                .get()
                .is_some_and(|addr| *addr == crate::gateway::builtin_code_get(code) as usize)
    }
}

/// True iff `callable` is the canonical builtin `math.sqrt` function object.
/// The JIT walker uses the builtin-code native fn-pointer identity to
/// distinguish it from a value rebound under the same `math.sqrt` name, so a
/// monkeypatched `math.sqrt` correctly declines the pure-inline specialization.
pub fn is_math_sqrt_function(callable: PyObjectRef) -> bool {
    unsafe { math_builtin_wrapper_matches(callable, &MATH_SQRT_WRAPPER) }
}

pub fn is_math_log_function(callable: PyObjectRef) -> bool {
    unsafe { math_builtin_wrapper_matches(callable, &MATH_LOG_WRAPPER) }
}

pub fn is_math_cos_function(callable: PyObjectRef) -> bool {
    unsafe { math_builtin_wrapper_matches(callable, &MATH_COS_WRAPPER) }
}

pub fn is_math_sin_function(callable: PyObjectRef) -> bool {
    unsafe { math_builtin_wrapper_matches(callable, &MATH_SIN_WRAPPER) }
}

/// Callable-identity probes used by the meta-trace walker.  As with
/// [`is_math_sqrt_function`], compare the immutable BuiltinCode function
/// pointer rather than a module/name string so rebinding `math.frexp` or
/// `math.ldexp` cannot enter a specialization for the old callable.
pub fn is_math_frexp_function(callable: PyObjectRef) -> bool {
    unsafe { math_builtin_wrapper_matches(callable, &MATH_FREXP_WRAPPER) }
}

pub fn is_math_ldexp_function(callable: PyObjectRef) -> bool {
    unsafe { math_builtin_wrapper_matches(callable, &MATH_LDEXP_WRAPPER) }
}

pub fn is_math_isqrt_function(callable: PyObjectRef) -> bool {
    unsafe { math_builtin_wrapper_matches(callable, &MATH_ISQRT_WRAPPER) }
}

/// Raw, allocation-free counterparts of RPython's `ll_math_frexp` result
/// components.  The translated PyPy trace carries the pair as two unboxed
/// values before `space.newtuple2` virtualizes; pyre's walker emits one pure
/// call per component because the IR has no multi-result call opcode.
pub extern "C" fn jit_math_frexp_mantissa(x: f64) -> f64 {
    pymath::math::frexp(x).0
}

pub extern "C" fn jit_math_frexp_exponent(x: f64) -> i64 {
    pymath::math::frexp(x).1 as i64
}

/// Non-raising raw `ldexp` for a guarded JIT fast path.  RPython lowers
/// `ll_math_ldexp` to the platform operation and guards its exceptional
/// result.  On overflow, return signed infinity so the walker's finite-result
/// guard deoptimizes and the ordinary builtin re-executes to raise
/// `OverflowError`; successful finite/underflow results are returned exactly.
pub extern "C" fn jit_math_ldexp_raw(x: f64, exp: i64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let Ok(exp) = i32::try_from(exp) else {
        return if exp < 0 {
            0.0f64.copysign(x)
        } else {
            f64::INFINITY.copysign(x)
        };
    };
    match pymath::math::ldexp(x, exp) {
        Ok(result) => result,
        Err(_) => f64::INFINITY.copysign(x),
    }
}

/// Allocation-free exact machine-integer arm of `app_math.isqrt`.
///
/// PyPy meta-traces the app-level implementation with an unboxed Signed when
/// the argument is a `W_IntObject`.  The native wrapper otherwise promotes
/// that value to `rbigint` before running the same algorithm, hiding the
/// unboxed arm from pyre's walker.  Use a hardware square-root estimate and
/// exact division-based corrections; the latter preserve integer semantics
/// even where `f64` cannot represent every input.
pub extern "C" fn jit_math_isqrt_i64(n: i64) -> i64 {
    debug_assert!(n >= 0);
    if n == 0 {
        return 0;
    }
    let n = n as u64;
    let mut root = (n as f64).sqrt() as u64;
    while root + 1 <= n / (root + 1) {
        root += 1;
    }
    while root > n / root {
        root -= 1;
    }
    root as i64
}

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
fn bigint_log(n: &BigInt, base: f64) -> Result<f64, crate::PyError> {
    if n.int_le(0) {
        return Err(crate::PyError::value_error("math domain error"));
    }
    n.log(base).map_err(map_rbigint_err)
}

/// Special-case integer arguments to avoid overflow, and give the domain
/// error the value-carrying message except for an int argument, whose error
/// carries no value.
fn log_any(w_x: PyObjectRef, base: f64) -> PyResult {
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
    if n_big.int_lt(0) {
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
fn get_bigint(obj: PyObjectRef) -> Result<BigInt, crate::PyError> {
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
                    return Ok(pyre_object::w_long_get_value(result).translated_alias());
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
    // RPython's GC transform roots this running rbigint across the next
    // argument's potentially user-defined `__index__` call.
    let mut result = RBigIntGcRoot::new(BigInt::zero());
    for &arg in args {
        *result = result.gcd(&get_bigint(arg)?).map_err(map_rbigint_err)?;
    }
    Ok(bigint_to_pyint(&result))
}

pub fn lcm(args: &[PyObjectRef]) -> PyResult {
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

pub fn comb(args: &[PyObjectRef]) -> PyResult {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(
            "comb() takes exactly two arguments",
        ));
    }
    // `n` is an unboxed rbigint local across `index(k)`, exactly the kind of
    // local rooted automatically by RPython's GC transform.
    let n_big = RBigIntGcRoot::new(get_bigint(args[0])?);
    let k_big = get_bigint(args[1])?;

    if n_big.int_lt(0) {
        return Err(crate::PyError::value_error(
            "n must be a non-negative integer",
        ));
    }
    if k_big.int_lt(0) {
        return Err(crate::PyError::value_error(
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
    // Keep `n` rooted while a non-None `k` invokes its `__index__`.
    let n_big = RBigIntGcRoot::new(get_bigint(args[0])?);
    if n_big.int_lt(0) {
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
        if k_val.int_lt(0) {
            return Err(crate::PyError::value_error(
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
        return Err(crate::PyError::type_error(
            "isqrt() takes exactly 1 argument",
        ));
    }
    let n = get_bigint(args[0])?;
    let value = n
        .isqrt()
        .map_err(|_| crate::PyError::value_error("isqrt() argument must be nonnegative"))?;
    Ok(bigint_to_pyint(&value))
}

pub fn fsum(args: &[PyObjectRef]) -> PyResult {
    // interp_math.py:572-633: consume the iterator once while maintaining the
    // partials array.  The old port first materialized every boxed element and
    // then built a second Vec<f64>; besides diverging from upstream, that kept
    // one shadow-stack root per input alive until the entire iterable had been
    // consumed.
    let _roots = pyre_object::gc_roots::push_roots();
    let iterable_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(args[0]);
    let w_iter = crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(iterable_slot))?;
    let iter_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_iter);

    let mut inf_sum = 0.0;
    let mut special_sum = 0.0;
    let mut partials: Vec<f64> = Vec::new();
    loop {
        let w_value =
            match crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(iter_slot)) {
                Ok(value) => value,
                Err(err) if err.kind == crate::PyErrorKind::StopIteration => break,
                Err(err) => return Err(err),
            };
        // `_get_double` can invoke user code.  Keep the yielded object rooted
        // only across that conversion, exactly like the translated livevar at
        // this point in the upstream loop.
        let original = {
            let _value_root = pyre_object::gc_roots::push_roots();
            let value_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(w_value);
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
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
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
        return Err(crate::PyError::overflow_error("math range error"));
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
                && pyre_object::w_dict_getitem_str(last, "__pyre_kw__")
                    .is_some_and(pyre_object::kw_marker::is_kw_marker_sentinel)
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
    let steps = match kwargs.and_then(|kw| unsafe { pyre_object::w_dict_getitem_str(kw, "steps") })
    {
        Some(s) => {
            use num_traits::ToPrimitive;
            let b = get_bigint(s)?;
            if b.int_lt(0) {
                return Err(crate::PyError::value_error(
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
