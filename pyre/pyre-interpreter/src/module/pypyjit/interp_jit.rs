//! `pypy/module/pypyjit/interp_jit.py` — the public `set_param` entry.
//!
//! The rest of that module (the driver, dispatch, `residual_call`) stays in
//! the JIT crates. This file owns only the function `moduledef.py` binds as
//! `interp_jit.set_param`.

/// interp_jit.py — `set_param(space, __args__)`.
///
/// Accepts the PyPy calling conventions:
///   * `set_param("name=value,name=value")` / `set_param("off")` /
///     `set_param("default")` — the positional string form.
///   * `set_param(name=value, ...)` — keyword arguments.
///
/// Both forms funnel through the JIT's authoritative `set_user_param` parser
/// (`call::set_jit_param_string`); the keyword form additionally validates each
/// name against `unroll_parameters`, reading the JIT's own table rather than
/// duplicating it.
pub(super) fn set_param(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    let (pos, mut kwds) = pyre_interpreter::builtins::split_builtin_kwargs(args);

    // interp_jit.py:147-148 — at most one non-keyword argument.
    if pos.len() > 1 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "set_param() takes at most 1 non-keyword argument, {} given",
            pos.len()
        )));
    }

    // interp_jit.py:151-156 — positional string → set_user_param(None, text).
    if let Some(&text_obj) = pos.first() {
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.pin_roots(&[kwds.unwrap_or(pyre_object::PY_NULL)]);
        let text = pyre_interpreter::baseobjspace::text_w(text_obj);
        let w = roots.get(base);
        kwds = if w.is_null() { None } else { Some(w) };
        drop(roots);
        let text = text?;
        if pyre_interpreter::call::set_jit_param_string(text).is_err() {
            return Err(pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::ValueError,
                "error in JIT parameters string".to_string(),
            ));
        }
    }

    // interp_jit.py:159-170 — keyword arguments. Re-serialize each `name=value`
    // pair into one parameter string so the JIT-side parser stays the single
    // source of truth. `enable_opts` carries a string value; every other
    // parameter is an integer (`space.int_w` rejects a non-int value here) whose
    // name is looked up in `unroll_parameters` (rlib/jit.py) before it is
    // accepted.
    if let Some(kw_dict) = kwds {
        let mut parts: Vec<String> = Vec::new();
        for (k, v) in unsafe { pyre_object::dictmultiobject::w_dict_items(kw_dict) } {
            if !unsafe { pyre_object::is_str(k) } {
                continue;
            }
            let key = unsafe { pyre_object::w_str_get_wtf8(k) };
            if key == "__pyre_kw__" {
                continue;
            }
            if key == "enable_opts" {
                let value = pyre_interpreter::baseobjspace::text_w(v)?;
                parts.push(format!("{key}={value}"));
            } else {
                let value = pyre_interpreter::baseobjspace::int_w(v)?;
                let known = majit_metainterp::jit::UNROLL_PARAMETERS
                    .iter()
                    .any(|&(name, _)| key == name && name != "enable_opts");
                if !known {
                    return Err(pyre_interpreter::PyError::type_error(format!(
                        "no JIT parameter '{key}'"
                    )));
                }
                parts.push(format!("{key}={value}"));
            }
        }
        if !parts.is_empty()
            && pyre_interpreter::call::set_jit_param_string(&parts.join(",")).is_err()
        {
            return Err(pyre_interpreter::PyError::new(
                pyre_interpreter::PyErrorKind::ValueError,
                "error in JIT parameters string".to_string(),
            ));
        }
    }

    Ok(pyre_object::w_none())
}
