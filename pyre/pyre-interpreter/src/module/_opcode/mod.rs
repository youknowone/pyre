//! _opcode module — PyPy: `pypy/module/_opcode/`.
//!
//! Stub providing stack_effect + has_arg / has_const / has_name /
//! has_jump and related classifiers — enough for opcode.py to import.
//! Returns neutral values (0 for stack_effect, False for has_*, empty
//! lists / dicts for the rest).  Full implementations would mirror
//! Python/compile.c.

use pyre_object::*;

fn get_special_method_names(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_list_new(vec![
        w_str_new("__enter__"),
        w_str_new("__exit__"),
        w_str_new("__aenter__"),
        w_str_new("__aexit__"),
    ]))
}

crate::py_module! {
    "_opcode",
    functions: {
        "stack_effect"             / 3 = |_| Ok(w_int_new(0)),
        "get_executor"             / 0 = |_| Ok(w_none()),
        "get_specialization_stats" / 0 = |_| Ok(w_dict_new()),
        "get_intrinsic1_descs"     / 0 = |_| Ok(w_list_new(vec![])),
        "get_intrinsic2_descs"     / 0 = |_| Ok(w_list_new(vec![])),
        "get_opname"               / 1 = |args| Ok(w_str_new(
            &format!("<{}>", args.first().map(|a| unsafe { w_int_get_value(*a) }).unwrap_or(0))
        )),
        "get_nb_ops"               / 0 = |_| Ok(w_list_new(vec![])),
        "get_special_method_names" / 0 = get_special_method_names,
        "get_executor_count"       / 0 = |_| Ok(w_int_new(0)),
        "get_hot_code"             / 0 = |_| Ok(w_list_new(vec![])),
    },
    extra_init: |ns| {
        for name in [
            "has_arg", "has_const", "has_name", "has_jump", "has_jrel",
            "has_jabs", "has_free", "has_local", "has_exc",
        ] {
            crate::dict_storage_store(
                ns, name,
                crate::make_builtin_function_with_arity(name, |_| Ok(w_bool_from(false)), 0),
            );
        }
    }
}
