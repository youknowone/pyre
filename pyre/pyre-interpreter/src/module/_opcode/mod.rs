//! _opcode module — PyPy: `pypy/module/_opcode/`.
//!
//! Stub providing stack_effect + has_arg / has_const / has_name /
//! has_jump and related classifiers — enough for opcode.py to import.
//! Returns neutral values (0 for stack_effect, False for has_*, empty
//! lists / dicts for the rest).  Full implementations would mirror
//! Python/compile.c.

use pyre_object::*;

fn stub_int_zero(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_int_new(0))
}

fn stub_none(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_none())
}

fn stub_empty_dict(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_dict_new())
}

fn stub_empty_list(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_list_new(vec![]))
}

fn stub_false(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(false))
}

fn get_opname(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_str_new("<0>"));
    }
    let code = unsafe { w_int_get_value(args[0]) };
    Ok(w_str_new(&format!("<{code}>")))
}

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
        "stack_effect"             / 3 = stub_int_zero,
        "get_executor"             / 0 = stub_none,
        "get_specialization_stats" / 0 = stub_empty_dict,
        "get_intrinsic1_descs"     / 0 = stub_empty_list,
        "get_intrinsic2_descs"     / 0 = stub_empty_list,
        "get_opname"               / 1 = get_opname,
        "get_nb_ops"               / 0 = stub_empty_list,
        "get_special_method_names" / 0 = get_special_method_names,
        "get_executor_count"       / 0 = stub_int_zero,
        "get_hot_code"             / 0 = stub_empty_list,
    },
    extra_init: |ns| {
        for name in [
            "has_arg", "has_const", "has_name", "has_jump", "has_jrel",
            "has_jabs", "has_free", "has_local", "has_exc",
        ] {
            crate::dict_storage_store(
                ns, name,
                crate::make_builtin_function_with_arity(name, stub_false, 0),
            );
        }
    }
}
