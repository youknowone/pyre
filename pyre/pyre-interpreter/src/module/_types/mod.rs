//! `_types` native type-object exports.

use pyre_object::*;
#[cfg(not(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
)))]
use std::sync::OnceLock;

fn store(ns: PyObjectRef, name: &str, ty: PyObjectRef) {
    crate::module_ns_store(ns, name, ty);
}

#[cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]
fn capsule_type() -> PyObjectRef {
    crate::cpyext::capsule::capsule_type()
}

#[cfg(not(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
)))]
fn capsule_type() -> PyObjectRef {
    static CAPSULE_TYPE: OnceLock<usize> = OnceLock::new();
    *CAPSULE_TYPE.get_or_init(|| crate::typedef::make_builtin_type("PyCapsule", |_| {}) as usize)
        as PyObjectRef
}

pub fn init(ns: PyObjectRef) {
    let function_type = crate::typedef::gettypeobject(&crate::function::FUNCTION_TYPE);
    store(
        ns,
        "AsyncGeneratorType",
        crate::typedef::gettypeobject(&pyre_object::generator::ASYNC_GENERATOR_TYPE),
    );
    store(
        ns,
        "BuiltinFunctionType",
        crate::typedef::gettypeobject(&crate::function::BUILTIN_FUNCTION_TYPE),
    );
    store(
        ns,
        "BuiltinMethodType",
        crate::typedef::gettypeobject(&crate::function::BUILTIN_FUNCTION_TYPE),
    );
    store(ns, "CapsuleType", capsule_type());
    store(
        ns,
        "CellType",
        crate::typedef::gettypeobject(&pyre_object::nestedscope::CELL_TYPE),
    );
    store(
        ns,
        "ClassMethodDescriptorType",
        crate::typedef::gettypeobject(&crate::function::CLASSMETHOD_DESCRIPTOR_TYPE),
    );
    store(
        ns,
        "CodeType",
        crate::typedef::gettypeobject(&crate::pycode::CODE_TYPE),
    );
    store(
        ns,
        "CoroutineType",
        crate::typedef::gettypeobject(&pyre_object::generator::COROUTINE_TYPE),
    );
    store(
        ns,
        "EllipsisType",
        crate::typedef::gettypeobject(&pyre_object::ELLIPSIS_TYPE),
    );
    store(
        ns,
        "FrameType",
        crate::typedef::gettypeobject(&crate::pyframe::FRAME_TYPE),
    );
    store(ns, "FunctionType", function_type);
    store(
        ns,
        "GeneratorType",
        crate::typedef::gettypeobject(&pyre_object::generator::GENERATOR_TYPE),
    );
    store(
        ns,
        "GenericAlias",
        crate::typedef::gettypeobject(&pyre_object::GENERIC_ALIAS_TYPE),
    );
    store(
        ns,
        "GetSetDescriptorType",
        crate::typedef::gettypeobject(&pyre_object::typedef::GETSET_DESCRIPTOR_TYPE),
    );
    store(ns, "LambdaType", function_type);
    store(
        ns,
        "MappingProxyType",
        crate::typedef::gettypeobject(&pyre_object::MAPPING_PROXY_TYPE),
    );
    store(
        ns,
        "MemberDescriptorType",
        crate::typedef::gettypeobject(&pyre_object::typedef::MEMBER_TYPE),
    );
    store(
        ns,
        "MethodDescriptorType",
        crate::typedef::gettypeobject(&crate::function::METHOD_DESCRIPTOR_TYPE),
    );
    store(
        ns,
        "MethodType",
        crate::typedef::gettypeobject(&pyre_object::function::METHOD_TYPE),
    );
    store(
        ns,
        "MethodWrapperType",
        crate::typedef::gettypeobject(&crate::function::METHOD_WRAPPER_TYPE),
    );
    store(
        ns,
        "ModuleType",
        crate::typedef::gettypeobject(&pyre_object::MODULE_TYPE),
    );
    store(
        ns,
        "NoneType",
        crate::typedef::gettypeobject(&pyre_object::NONE_TYPE),
    );
    store(
        ns,
        "NotImplementedType",
        crate::typedef::gettypeobject(&pyre_object::NOTIMPLEMENTED_TYPE),
    );
    store(
        ns,
        "SimpleNamespace",
        crate::module::sys::vm::simple_namespace_type(),
    );
    store(
        ns,
        "TracebackType",
        crate::typedef::gettypeobject(&crate::pytraceback::PYTRACEBACK_TYPE),
    );
    store(
        ns,
        "UnionType",
        crate::typedef::gettypeobject(&pyre_object::UNION_TYPE),
    );
    store(
        ns,
        "WrapperDescriptorType",
        crate::typedef::gettypeobject(&crate::function::SLOT_WRAPPER_TYPE),
    );
}
