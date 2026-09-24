//! `_types` native type-object exports.

use pyre_object::*;
#[cfg(not(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
)))]
use std::sync::OnceLock;

fn store(ns: PyObjectRef, name: &str, ty: PyObjectRef) {
    pyre_interpreter::module_ns_store(ns, name, ty);
}

#[cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]
fn capsule_type() -> PyObjectRef {
    pyre_interpreter::cpyext::capsule::capsule_type()
}

#[cfg(not(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
)))]
/// The build carries no capsules at all, so the name answers with a type that
/// can produce none: `PyCapsule_Type` has no `tp_new` and no
/// `Py_TPFLAGS_BASETYPE`, and a capsule only ever comes from `PyCapsule_New`.
fn capsule_type() -> PyObjectRef {
    static CAPSULE_TYPE: OnceLock<usize> = OnceLock::new();
    *CAPSULE_TYPE.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type("PyCapsule", |ns| unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                "__new__",
                pyre_interpreter::typedef::make_new_descr(|_| {
                    Err(pyre_interpreter::PyError::type_error(
                        "cannot create 'PyCapsule' instances",
                    ))
                }),
            );
        });
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(tp);
            pyre_object::w_type_set_acceptable_as_base_class(tp, false);
        }
        tp as usize
    }) as PyObjectRef
}

pub fn init(ns: PyObjectRef) -> Result<(), pyre_interpreter::PyError> {
    let function_type =
        pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::function::FUNCTION_TYPE);
    store(
        ns,
        "AsyncGeneratorType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::generator::ASYNC_GENERATOR_TYPE),
    );
    store(
        ns,
        "BuiltinFunctionType",
        pyre_interpreter::typedef::gettypeobject(
            &pyre_interpreter::function::BUILTIN_FUNCTION_TYPE,
        ),
    );
    store(
        ns,
        "BuiltinMethodType",
        pyre_interpreter::typedef::gettypeobject(
            &pyre_interpreter::function::BUILTIN_FUNCTION_TYPE,
        ),
    );
    store(ns, "CapsuleType", capsule_type());
    store(
        ns,
        "CellType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::nestedscope::CELL_TYPE),
    );
    store(
        ns,
        "ClassMethodDescriptorType",
        pyre_interpreter::typedef::gettypeobject(
            &pyre_interpreter::function::CLASSMETHOD_DESCRIPTOR_TYPE,
        ),
    );
    store(
        ns,
        "CodeType",
        pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::pycode::CODE_TYPE),
    );
    store(
        ns,
        "CoroutineType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::generator::COROUTINE_TYPE),
    );
    store(
        ns,
        "EllipsisType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::ELLIPSIS_TYPE),
    );
    store(
        ns,
        "FrameType",
        pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::pyframe::FRAME_TYPE),
    );
    store(ns, "FunctionType", function_type);
    store(
        ns,
        "GeneratorType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::generator::GENERATOR_TYPE),
    );
    store(
        ns,
        "GenericAlias",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::GENERIC_ALIAS_TYPE),
    );
    store(
        ns,
        "GetSetDescriptorType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::typedef::GETSET_DESCRIPTOR_TYPE),
    );
    store(ns, "LambdaType", function_type);
    store(
        ns,
        "MappingProxyType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::MAPPING_PROXY_TYPE),
    );
    store(
        ns,
        "MemberDescriptorType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::typedef::MEMBER_TYPE),
    );
    store(
        ns,
        "MethodDescriptorType",
        pyre_interpreter::typedef::gettypeobject(
            &pyre_interpreter::function::METHOD_DESCRIPTOR_TYPE,
        ),
    );
    store(
        ns,
        "MethodType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::function::METHOD_TYPE),
    );
    store(
        ns,
        "MethodWrapperType",
        pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::function::METHOD_WRAPPER_TYPE),
    );
    store(
        ns,
        "ModuleType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::MODULE_TYPE),
    );
    store(
        ns,
        "NoneType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::NONE_TYPE),
    );
    store(
        ns,
        "NotImplementedType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::NOTIMPLEMENTED_TYPE),
    );
    store(
        ns,
        "SimpleNamespace",
        pyre_interpreter::module::sys::vm::simple_namespace_type(),
    );
    store(
        ns,
        "TracebackType",
        pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::pytraceback::PYTRACEBACK_TYPE),
    );
    store(
        ns,
        "UnionType",
        pyre_interpreter::typedef::gettypeobject(&pyre_object::UNION_TYPE),
    );
    store(
        ns,
        "WrapperDescriptorType",
        pyre_interpreter::typedef::gettypeobject(&pyre_interpreter::function::SLOT_WRAPPER_TYPE),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn capsule_type_publishes_null_tp_new() {
        pyre_interpreter::typedef::init_typeobjects();
        let capsule_type = super::capsule_type();
        assert!(unsafe { pyre_object::w_type_disallows_instantiation(capsule_type) });

        let flags = pyre_interpreter::baseobjspace::getattr_str(capsule_type, "__flags__")
            .expect("PyCapsule.__flags__ lookup failed");
        assert_ne!(unsafe { pyre_object::w_int_get_value(flags) } & (1 << 7), 0);
    }
}
