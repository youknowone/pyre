//! _opcode implementation — PyPy: pypy/module/_opcode/interp_opcode.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// _opcode stub — PyPy: pypy/module/_opcode (CPython's opcode introspection).
/// opcode.py requires stack_effect + has_arg/has_const/has_name/has_jump and
/// related classifiers. Our stubs return neutral values; full implementations
/// would mirror CPython Python/compile.c.
pub fn register_module(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "stack_effect",
        crate::make_builtin_function_with_arity(
            "stack_effect",
            |_| Ok(pyre_object::w_int_new(0)),
            3,
        ),
    );
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
    crate::dict_storage_store(
        ns,
        "get_executor",
        crate::make_builtin_function_with_arity("get_executor", |_| Ok(pyre_object::w_none()), 0),
    );
    crate::dict_storage_store(
        ns,
        "get_specialization_stats",
        crate::make_builtin_function_with_arity(
            "get_specialization_stats",
            |_| Ok(pyre_object::w_dict_new()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_intrinsic1_descs",
        crate::make_builtin_function_with_arity(
            "get_intrinsic1_descs",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_intrinsic2_descs",
        crate::make_builtin_function_with_arity(
            "get_intrinsic2_descs",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_opname",
        crate::make_builtin_function_with_arity(
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
    );
    crate::dict_storage_store(
        ns,
        "get_nb_ops",
        crate::make_builtin_function_with_arity(
            "get_nb_ops",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_special_method_names",
        crate::make_builtin_function_with_arity(
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
    );
    crate::dict_storage_store(
        ns,
        "get_executor_count",
        crate::make_builtin_function_with_arity(
            "get_executor_count",
            |_| Ok(pyre_object::w_int_new(0)),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_hot_code",
        crate::make_builtin_function_with_arity(
            "get_hot_code",
            |_| Ok(pyre_object::w_list_new(vec![])),
            0,
        ),
    );
}
