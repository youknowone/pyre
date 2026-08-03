//! `_bisect` accelerator module.
//!
//! PyPy keeps the algorithms app-level in `lib-python/3/bisect.py`; this
//! interpreter-level surface follows those loops while providing the optional
//! accelerator imported by that module. RustPython's corresponding owner is
//! `crates/stdlib/src/bisect.rs`.

use pyre_object::*;

struct BisectArgs {
    a: PyObjectRef,
    x: PyObjectRef,
    lo: i64,
    hi: i64,
    key: Option<PyObjectRef>,
}

fn argument(
    positional: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
    index: usize,
    name: &str,
    function: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    let positional_value = positional.get(index).copied();
    let keyword_value = crate::builtins::kwarg_get(kwargs, name);
    if positional_value.is_some() && keyword_value.is_some() {
        return Err(crate::PyError::type_error(format!(
            "{function}() got multiple values for argument '{name}'"
        )));
    }
    Ok(positional_value.or(keyword_value))
}

fn index_value(value: PyObjectRef) -> Result<i64, crate::PyError> {
    let index = crate::baseobjspace::space_index(value)?;
    crate::baseobjspace::int_w(index)
}

fn parse_args(args: &[PyObjectRef], function: &str) -> Result<BisectArgs, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["a", "x", "lo", "hi", "key"], function)?;
    if positional.len() > 4 {
        return Err(crate::PyError::type_error(format!(
            "{function}() takes at most 4 positional arguments ({} given)",
            positional.len()
        )));
    }

    let a = argument(positional, kwargs, 0, "a", function)?.ok_or_else(|| {
        crate::PyError::type_error(format!(
            "{function}() missing required argument 'a' (pos 1)"
        ))
    })?;
    let x = argument(positional, kwargs, 1, "x", function)?.ok_or_else(|| {
        crate::PyError::type_error(format!(
            "{function}() missing required argument 'x' (pos 2)"
        ))
    })?;
    let lo = match argument(positional, kwargs, 2, "lo", function)? {
        Some(value) => index_value(value)?,
        None => 0,
    };
    if lo < 0 {
        return Err(crate::PyError::value_error("lo must be non-negative"));
    }
    let hi = match argument(positional, kwargs, 3, "hi", function)? {
        Some(value) if !unsafe { is_none(value) } => index_value(value)?,
        _ => crate::baseobjspace::len_w(a)?,
    };
    let key = crate::builtins::kwarg_get(kwargs, "key").filter(|value| !unsafe { is_none(*value) });
    Ok(BisectArgs { a, x, lo, hi, key })
}

fn call_one(callable: PyObjectRef, arg: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    crate::call::call_function_impl_result(callable, &[arg])
}

fn less_than(left: PyObjectRef, right: PyObjectRef) -> Result<bool, crate::PyError> {
    let result = crate::objspace::descroperation::compare(
        left,
        right,
        crate::objspace::descroperation::CompareOp::Lt,
    )?;
    crate::baseobjspace::is_true(result)
}

fn bisect(mut parsed: BisectArgs, right: bool) -> Result<i64, crate::PyError> {
    while parsed.lo < parsed.hi {
        // Written this way instead of `(lo + hi) / 2` so a search spanning
        // `sys.maxsize` cannot overflow, matching `_bisectmodule.c`.
        let mid = parsed.lo + (parsed.hi - parsed.lo) / 2;
        let mut item = crate::baseobjspace::getitem(parsed.a, w_int_new(mid))?;
        if let Some(key) = parsed.key {
            item = call_one(key, item)?;
        }
        let is_less = if right {
            less_than(parsed.x, item)?
        } else {
            less_than(item, parsed.x)?
        };
        if is_less == right {
            parsed.hi = mid;
        } else {
            parsed.lo = mid + 1;
        }
    }
    Ok(parsed.lo)
}

fn bisect_left(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_int_new(bisect(parse_args(args, "bisect_left")?, false)?))
}

fn bisect_right(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_int_new(bisect(parse_args(args, "bisect_right")?, true)?))
}

fn insort(
    args: &[PyObjectRef],
    right: bool,
    function: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let mut parsed = parse_args(args, function)?;
    let original_x = parsed.x;
    if let Some(key) = parsed.key {
        parsed.x = call_one(key, original_x)?;
    }
    let a = parsed.a;
    let index = bisect(parsed, right)?;
    let insert = crate::baseobjspace::getattr_str(a, "insert")?;
    crate::call::call_function_impl_result(insert, &[w_int_new(index), original_x])?;
    Ok(w_none())
}

fn insort_left(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    insort(args, false, "insort_left")
}

fn insort_right(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    insort(args, true, "insort_right")
}

pub fn init(ns: PyObjectRef) {
    let left = crate::gateway::with_module(
        "_bisect",
        crate::make_module_builtin_function("bisect_left", bisect_left),
    );
    let right = crate::gateway::with_module(
        "_bisect",
        crate::make_module_builtin_function("bisect_right", bisect_right),
    );
    let insert_left = crate::gateway::with_module(
        "_bisect",
        crate::make_module_builtin_function("insort_left", insort_left),
    );
    let insert_right = crate::gateway::with_module(
        "_bisect",
        crate::make_module_builtin_function("insort_right", insort_right),
    );
    crate::module_ns_store(ns, "bisect_left", left);
    crate::module_ns_store(ns, "bisect_right", right);
    crate::module_ns_store(ns, "bisect", right);
    crate::module_ns_store(ns, "insort_left", insert_left);
    crate::module_ns_store(ns, "insort_right", insert_right);
    crate::module_ns_store(ns, "insort", insert_right);
}
