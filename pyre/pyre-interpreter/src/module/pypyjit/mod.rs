//! `pypyjit` module — PyPy: pypy/module/pypyjit/
//!
//! Exposes `set_param`, the runtime JIT-parameter control from
//! `interp_jit.py:138-167`. The JIT itself lives in the higher `pyre-jit`
//! crate, so this routes through the `SET_JIT_PARAM_STRING_HOOK` /
//! `SET_JIT_PARAM_HOOK` that pyre-jit registers at boot (`call.rs`). Because
//! the hooks are in-process function pointers rather than an env lever, a
//! `pypyjit.set_param(...)` call configures the warmstate on every backend
//! including the wasm guest (which sees no environment).

/// interp_jit.py:138-167 — `set_param(space, __args__)`.
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
fn set_param(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let (pos, kwds) = crate::builtins::split_builtin_kwargs(args);

    // interp_jit.py:147-148 — at most one non-keyword argument.
    if pos.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "set_param() takes at most 1 non-keyword argument, {} given",
            pos.len()
        )));
    }

    // interp_jit.py:151-156 — positional string → set_user_param(None, text).
    if let Some(&text_obj) = pos.first() {
        let text = crate::baseobjspace::text_w(text_obj)?;
        if crate::call::set_jit_param_string(&text).is_err() {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "error in JIT parameters string".to_string(),
            ));
        }
    }

    // interp_jit.py:159-170 — keyword arguments. Re-serialize each `name=value`
    // pair into one parameter string so the JIT-side parser stays the single
    // source of truth. `enable_opts` carries a string value; every other
    // parameter is an integer (`space.int_w` rejects a non-int value here) whose
    // name is looked up in `unroll_parameters` (rlib/jit.py:606) before it is
    // accepted.
    if let Some(kw_dict) = kwds {
        let mut parts: Vec<String> = Vec::new();
        for (k, v) in unsafe { pyre_object::dictmultiobject::w_dict_items(kw_dict) } {
            if !unsafe { pyre_object::is_str(k) } {
                continue;
            }
            let key = unsafe { pyre_object::w_str_get_value(k) };
            if key == "__pyre_kw__" {
                continue;
            }
            if key == "enable_opts" {
                let value = crate::baseobjspace::text_w(v)?;
                parts.push(format!("{key}={value}"));
            } else {
                let value = crate::baseobjspace::int_w(v)?;
                let known = majit_metainterp::jit::UNROLL_PARAMETERS
                    .iter()
                    .any(|&(name, _)| name == key && name != "enable_opts");
                if !known {
                    return Err(crate::PyError::type_error(format!(
                        "no JIT parameter '{key}'"
                    )));
                }
                parts.push(format!("{key}={value}"));
            }
        }
        if !parts.is_empty() && crate::call::set_jit_param_string(&parts.join(",")).is_err() {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "error in JIT parameters string".to_string(),
            ));
        }
    }

    Ok(pyre_object::w_none())
}

crate::py_module! {
    "pypyjit",
    functions: {
        "set_param" / * = |args| set_param(args),
    }
}
