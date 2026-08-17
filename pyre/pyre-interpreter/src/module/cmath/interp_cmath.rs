//! cmath module implementations — PyPy: pypy/module/cmath/interp_cmath.py
//!
//! Complex math via `pymath::cmath` (the `rpython.rlib.rcomplex` role,
//! with CPython's special-value tables).  Arguments are unpacked through
//! `builtins::complex_coerce`, the `unpackcomplex` port, so `__complex__`
//! / `__float__` / `__index__` objects are accepted like PyPy's
//! `space.unpackcomplex`.

use num_complex::Complex64;
use pymath::cmath as pmc;
use pyre_object::*;

type PyResult = Result<PyObjectRef, crate::PyError>;

/// `space.unpackcomplex(w_z)` — reuse the `complexobject.py unpackcomplex`
/// port that `complex()` construction goes through.
fn unpack(obj: PyObjectRef) -> Result<Complex64, crate::PyError> {
    let (re, im) = crate::builtins::complex_coerce(obj)?;
    Ok(Complex64::new(re, im))
}

/// `call_c_func` (interp_cmath.py:19) — errno-style failures become the
/// fixed cmath messages.
fn map_err(e: pymath::Error) -> crate::PyError {
    match e {
        pymath::Error::EDOM => crate::PyError::value_error("math domain error"),
        pymath::Error::ERANGE => crate::PyError::overflow_error("math range error"),
    }
}

/// `space.newcomplex(resx, resy)`.
fn wrap(z: Complex64) -> PyObjectRef {
    complexobject::w_complex_new(z.re, z.im)
}

/// `unaryfn` (interp_cmath.py:29) — unpack, compute, wrap.  Arity is
/// enforced by the `/ 1` registration.
macro_rules! cm1 {
    ($name:ident) => {
        pub fn $name(args: &[PyObjectRef]) -> PyResult {
            pmc::$name(unpack(args[0])?).map(wrap).map_err(map_err)
        }
    };
}

cm1!(sqrt);
cm1!(exp);
cm1!(log10);
cm1!(sin);
cm1!(cos);
cm1!(tan);
cm1!(asin);
cm1!(acos);
cm1!(atan);
cm1!(sinh);
cm1!(cosh);
cm1!(tanh);
cm1!(asinh);
cm1!(acosh);
cm1!(atanh);

/// `wrapped_log` (interp_cmath.py:80) — with a base, `log(z)/log(base)`;
/// `pymath::cmath::log` carries the `_Py_c_quot` division itself.
pub fn log(args: &[PyObjectRef]) -> PyResult {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "cmath.log() takes no keyword arguments",
        ));
    }
    if pos.is_empty() {
        return Err(crate::PyError::type_error(
            "log expected at least 1 argument, got 0",
        ));
    }
    if pos.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "log expected at most 2 arguments, got {}",
            pos.len()
        )));
    }
    let z = unpack(pos[0])?;
    let base = pos.get(1).map(|&b| unpack(b)).transpose()?;
    pmc::log(z, base).map(wrap).map_err(map_err)
}

/// `wrapped_phase` — a float result, not a complex.
pub fn phase(args: &[PyObjectRef]) -> PyResult {
    let phi = pmc::phase(unpack(args[0])?).map_err(map_err)?;
    Ok(floatobject::w_float_new(phi))
}

/// `wrapped_polar` — `(r, phi)` tuple.
pub fn polar(args: &[PyObjectRef]) -> PyResult {
    let (r, phi) = pmc::polar(unpack(args[0])?).map_err(map_err)?;
    Ok(w_tuple_new(vec![
        floatobject::w_float_new(r),
        floatobject::w_float_new(phi),
    ]))
}

/// `wrapped_rect` — arguments go through `space.float_w`, so a complex
/// operand is rejected rather than unpacked.
pub fn rect(args: &[PyObjectRef]) -> PyResult {
    let r = crate::baseobjspace::float_w(args[0])?;
    let phi = crate::baseobjspace::float_w(args[1])?;
    pmc::rect(r, phi).map(wrap).map_err(map_err)
}

/// `wrapped_isfinite` — both components finite.
pub fn isfinite(args: &[PyObjectRef]) -> PyResult {
    Ok(w_bool_from(pmc::isfinite(unpack(args[0])?)))
}

/// `wrapped_isinf` — either component infinite.
pub fn isinf(args: &[PyObjectRef]) -> PyResult {
    Ok(w_bool_from(pmc::isinf(unpack(args[0])?)))
}

/// `wrapped_isnan` — either component NaN.
pub fn isnan(args: &[PyObjectRef]) -> PyResult {
    Ok(w_bool_from(pmc::isnan(unpack(args[0])?)))
}

/// `cmath.isclose(a, b, *, rel_tol=1e-09, abs_tol=0.0)` — complex
/// `_Py_c_isclose` equivalent over the two operands' components.
pub fn isclose(args: &[PyObjectRef]) -> PyResult {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["rel_tol", "abs_tol"], "isclose")?;
    if pos.len() < 2 {
        return Err(crate::PyError::type_error(
            "isclose() missing required argument",
        ));
    }
    let (ar, ai) = crate::builtins::complex_coerce(pos[0])?;
    let (br, bi) = crate::builtins::complex_coerce(pos[1])?;
    let tol = |name: &str, default: f64| -> Result<f64, crate::PyError> {
        match crate::builtins::kwarg_get(kwargs, name) {
            Some(v) => crate::baseobjspace::float_w(v),
            None => Ok(default),
        }
    };
    let rel_tol = tol("rel_tol", 1e-9)?;
    let abs_tol = tol("abs_tol", 0.0)?;
    if rel_tol < 0.0 || abs_tol < 0.0 {
        return Err(crate::PyError::value_error(
            "tolerances must be non-negative",
        ));
    }
    // Exact equality (covers the inf == inf case).
    if ar == br && ai == bi {
        return Ok(w_bool_from(true));
    }
    // Any infinity that is not an exact match is not close.
    if ar.is_infinite() || ai.is_infinite() || br.is_infinite() || bi.is_infinite() {
        return Ok(w_bool_from(false));
    }
    let diff = (ar - br).hypot(ai - bi);
    let mag_a = ar.hypot(ai);
    let mag_b = br.hypot(bi);
    let close = diff <= (rel_tol * mag_a.max(mag_b)).max(abs_tol);
    Ok(w_bool_from(close))
}
