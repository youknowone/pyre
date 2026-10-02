//! cmath module implementations — PyPy: pypy/module/cmath/interp_cmath.py
//!
//! Complex math via `pymath::cmath` (the `rpython.rlib.rcomplex` role,
//! with CPython's special-value tables).  Arguments are unpacked through
//! `builtins::complex_coerce`, the `unpackcomplex` port, so `__complex__`,
//! then `__index__`, then `__float__` are accepted like PyPy's
//! `space.unpackcomplex`.

use num_complex::Complex64;
use pymath::cmath as pmc;
use pyre_object::*;

type PyResult = Result<PyObjectRef, pyre_interpreter::PyError>;

/// `space.unpackcomplex(w_z)` — reuse the `complexobject.py unpackcomplex`
/// port that `complex()` construction goes through.
fn unpack(obj: PyObjectRef) -> Result<Complex64, pyre_interpreter::PyError> {
    let (re, im) = pyre_interpreter::builtins::complex_coerce(obj)?;
    Ok(Complex64::new(re, im))
}

/// `call_c_func` (interp_cmath.py) — errno-style failures become the
/// fixed cmath messages.
fn map_err(e: pymath::Error) -> pyre_interpreter::PyError {
    match e {
        pymath::Error::EDOM => pyre_interpreter::PyError::value_error("math domain error"),
        pymath::Error::ERANGE => pyre_interpreter::PyError::overflow_error("math range error"),
    }
}

/// `space.newcomplex(resx, resy)`.
fn wrap(z: Complex64) -> PyObjectRef {
    complexobject::w_complex_new(z.re, z.im)
}

/// `unaryfn` (interp_cmath.py) — unpack, compute, wrap.  Arity is
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

/// `wrapped_log` (interp_cmath.py) — with a base, `log(z)/log(base)`;
/// `pymath::cmath::log` carries the `_Py_c_quot` division itself.
pub fn log(args: &[PyObjectRef]) -> PyResult {
    let (pos, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    if pyre_interpreter::builtins::has_real_kwargs(kwargs) {
        return Err(pyre_interpreter::PyError::type_error(
            "cmath.log() takes no keyword arguments",
        ));
    }
    if pos.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "log expected at least 1 argument, got 0",
        ));
    }
    if pos.len() > 2 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "log expected at most 2 arguments, got {}",
            pos.len()
        )));
    }
    let w_z = pos[0];
    let mut w_base = pos.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let z = pyre_object::with_roots!(w_base => unpack(w_z))?;
    let base = if w_base.is_null() {
        None
    } else {
        Some(unpack(w_base)?)
    };
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
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(floatobject::w_float_new(r));
    fields.push(floatobject::w_float_new(phi));
    Ok(w_tuple_new(fields.take()))
}

/// `wrapped_rect` — arguments go through `space.float_w`, so a complex
/// operand is rejected rather than unpacked.
pub fn rect(args: &[PyObjectRef]) -> PyResult {
    let w_r = args[0];
    let mut w_phi = args[1];
    let r = pyre_object::with_roots!(w_phi => pyre_interpreter::baseobjspace::float_w(w_r))?;
    let phi = pyre_interpreter::baseobjspace::float_w(w_phi)?;
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
    let (pos, mut kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    pyre_interpreter::builtins::kwarg_reject_unknown(kwargs, &["rel_tol", "abs_tol"], "isclose")?;
    if pos.len() < 2 {
        return Err(pyre_interpreter::PyError::type_error(
            "isclose() missing required argument",
        ));
    }
    // `b` and the keywords are read back after `a`'s `__complex__` ran.
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[pos[0], pos[1], kwargs.unwrap_or(pyre_object::PY_NULL)]);
    let a = pyre_interpreter::builtins::complex_coerce(roots.get(base));
    let w_b = roots.get(base + 1);
    let w = roots.get(base + 2);
    kwargs = if w.is_null() { None } else { Some(w) };
    drop(roots);
    let (ar, ai) = a?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[w_b, kwargs.unwrap_or(pyre_object::PY_NULL)]);
    let b = pyre_interpreter::builtins::complex_coerce(roots.get(base));
    let w = roots.get(base + 1);
    kwargs = if w.is_null() { None } else { Some(w) };
    drop(roots);
    let (br, bi) = b?;
    let tol = |kwargs: Option<PyObjectRef>,
               name: &str,
               default: f64|
     -> Result<f64, pyre_interpreter::PyError> {
        match pyre_interpreter::builtins::kwarg_get(kwargs, name) {
            Some(v) => pyre_interpreter::baseobjspace::float_w(v),
            None => Ok(default),
        }
    };
    // `abs_tol` is looked up after `rel_tol`'s `__float__` ran.
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
    let w = roots.get(base);
    let rel_tol = tol(if w.is_null() { None } else { Some(w) }, "rel_tol", 1e-9);
    let w = roots.get(base);
    kwargs = if w.is_null() { None } else { Some(w) };
    drop(roots);
    let rel_tol = rel_tol?;
    let abs_tol = tol(kwargs, "abs_tol", 0.0)?;
    if rel_tol < 0.0 || abs_tol < 0.0 {
        return Err(pyre_interpreter::PyError::value_error(
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
