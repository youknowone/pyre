//! _functools module — CPython accelerator imported by
//! `lib-python/3/functools.py`.
//!
use pyre_object::*;

const ATTR_CMP: &str = "cmp";
const ATTR_OBJ: &str = "obj";

fn key_wrapper_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::setattr_str(args[0], ATTR_OBJ, args[1])?;
    Ok(w_none())
}

fn key_wrapper_compare(
    args: &[PyObjectRef],
    op: crate::objspace::descroperation::CompareOp,
) -> Result<PyObjectRef, crate::PyError> {
    let self_type = crate::typedef::r#type(args[0]).expect("key wrapper has a type");
    if crate::typedef::r#type(args[1]) != Some(self_type) {
        return Ok(w_not_implemented());
    }
    let cmp = unsafe {
        crate::baseobjspace::lookup_in_type(self_type, ATTR_CMP)
            .expect("key wrapper type has its comparator")
    };
    let lhs = crate::baseobjspace::getattr_str(args[0], ATTR_OBJ)?;
    let rhs = crate::baseobjspace::getattr_str(args[1], ATTR_OBJ)?;
    let result = crate::call::call_function_impl_result(cmp, &[lhs, rhs])?;
    crate::objspace::descroperation::compare(result, w_int_new(0), op)
}

fn key_wrapper_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    key_wrapper_compare(args, crate::objspace::descroperation::CompareOp::Lt)
}

fn key_wrapper_le(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    key_wrapper_compare(args, crate::objspace::descroperation::CompareOp::Le)
}

fn key_wrapper_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    key_wrapper_compare(args, crate::objspace::descroperation::CompareOp::Eq)
}

fn key_wrapper_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    key_wrapper_compare(args, crate::objspace::descroperation::CompareOp::Gt)
}

fn key_wrapper_ge(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    key_wrapper_compare(args, crate::objspace::descroperation::CompareOp::Ge)
}

fn key_wrapper_hash(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Err(crate::PyError::type_error(
        "unhashable type: 'functools.KeyWrapper'",
    ))
}

fn cmp_to_key(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cmp = args[0];
    let key_type = crate::typedef::make_builtin_type("functools.KeyWrapper", |ns| {
        crate::dict_storage_store(ns, ATTR_CMP, cmp);
        crate::dict_storage_store(
            ns,
            "__init__",
            crate::make_builtin_function_with_arity("__init__", key_wrapper_init, 2),
        );
        for (name, func) in [
            ("__lt__", key_wrapper_lt as fn(&[PyObjectRef]) -> _),
            ("__le__", key_wrapper_le as fn(&[PyObjectRef]) -> _),
            ("__eq__", key_wrapper_eq as fn(&[PyObjectRef]) -> _),
            ("__gt__", key_wrapper_gt as fn(&[PyObjectRef]) -> _),
            ("__ge__", key_wrapper_ge as fn(&[PyObjectRef]) -> _),
        ] {
            crate::dict_storage_store(
                ns,
                name,
                crate::make_builtin_function_with_arity(name, func, 2),
            );
        }
        crate::dict_storage_store(
            ns,
            "__hash__",
            crate::make_builtin_function_with_arity("__hash__", key_wrapper_hash, 1),
        );
    });
    unsafe { pyre_object::w_type_set_hasdict(key_type, true) };
    Ok(key_type)
}

crate::py_module! {
    "_functools",
    functions: {
        "reduce"     / * = |_| Err(crate::PyError::type_error("reduce not implemented")),
        "cmp_to_key" / 1 = cmp_to_key,
    },
}
