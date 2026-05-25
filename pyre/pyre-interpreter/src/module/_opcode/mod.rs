//! _opcode module — PyPy: `pypy/module/_opcode/`.
//!
//! Stub providing stack_effect + has_arg / has_const / has_name /
//! has_jump and related classifiers — enough for opcode.py to import.
//! Returns neutral values (0 for stack_effect, False for has_*, empty
//! lists / dicts for the rest).  Full implementations would mirror
//! CPython Python/compile.c.

crate::py_module! {
    "_opcode",
    interpleveldefs: {
        "stack_effect" => crate::make_builtin_function_with_arity(
            "stack_effect", |_| Ok(pyre_object::w_int_new(0)), 3),
        "get_executor" => crate::make_builtin_function_with_arity(
            "get_executor", |_| Ok(pyre_object::w_none()), 0),
        "get_specialization_stats" => crate::make_builtin_function_with_arity(
            "get_specialization_stats", |_| Ok(pyre_object::w_dict_new()), 0),
        "get_intrinsic1_descs" => crate::make_builtin_function_with_arity(
            "get_intrinsic1_descs", |_| Ok(pyre_object::w_list_new(vec![])), 0),
        "get_intrinsic2_descs" => crate::make_builtin_function_with_arity(
            "get_intrinsic2_descs", |_| Ok(pyre_object::w_list_new(vec![])), 0),
        "get_opname" => crate::make_builtin_function_with_arity(
            "get_opname",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_str_new("<0>"));
                }
                let code = unsafe { pyre_object::w_int_get_value(args[0]) };
                Ok(pyre_object::w_str_new(&format!("<{code}>")))
            },
            1,
        ),
        "get_nb_ops" => crate::make_builtin_function_with_arity(
            "get_nb_ops", |_| Ok(pyre_object::w_list_new(vec![])), 0),
        "get_special_method_names" => crate::make_builtin_function_with_arity(
            "get_special_method_names",
            |_| {
                Ok(pyre_object::w_list_new(vec![
                    pyre_object::w_str_new("__enter__"),
                    pyre_object::w_str_new("__exit__"),
                    pyre_object::w_str_new("__aenter__"),
                    pyre_object::w_str_new("__aexit__"),
                ]))
            },
            0,
        ),
        "get_executor_count" => crate::make_builtin_function_with_arity(
            "get_executor_count", |_| Ok(pyre_object::w_int_new(0)), 0),
        "get_hot_code" => crate::make_builtin_function_with_arity(
            "get_hot_code", |_| Ok(pyre_object::w_list_new(vec![])), 0),
    },
    extra_init: |ns| {
        for name in [
            "has_arg",
            "has_const",
            "has_name",
            "has_jump",
            "has_jrel",
            "has_jabs",
            "has_free",
            "has_local",
            "has_exc",
        ] {
            crate::dict_storage_store(
                ns,
                name,
                crate::make_builtin_function_with_arity(
                    name,
                    |_| Ok(pyre_object::w_bool_from(false)),
                    0,
                ),
            );
        }
    }
}
