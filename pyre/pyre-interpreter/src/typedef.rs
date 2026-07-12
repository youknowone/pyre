//! TypeDef — builtin type descriptor registry.
//!
//! PyPy equivalent: pypy/interpreter/typedef.py
//!
//! Each builtin type (list, str, dict, tuple, int, float, bool, etc.)
//! gets a W_TypeObject with methods pre-installed in its namespace.
//! `getattr` looks up the type object from the registry and searches
//! its namespace via MRO, exactly like user-defined classes.
//!
//! This eliminates the `builtin_type_method` match-based dispatch and
//! unifies attribute lookup for all object types.

use std::collections::HashMap;
use std::sync::OnceLock;

use pyre_object::pyobject::*;
use pyre_object::*;
use rustpython_wtf8::{CodePoint, Wtf8Buf};

use crate::{
    DictStorage, dict_storage_store, make_builtin_function, make_builtin_function_with_arity,
};

/// Compatibility stand-ins for PyPy `typedef.py` API (type descriptor helpers).
#[derive(Debug, Default)]
pub struct TypeDef;

impl TypeDef {
    pub fn new(_name: &str, _base: Option<crate::W_Root>, _dict: Option<PyObjectRef>) -> Self {
        let _ = (_name, _base, _dict);
        Self
    }
}

#[derive(Debug, Default)]
pub struct GetSetProperty;

#[derive(Debug, Default)]
pub struct Member;

#[derive(Debug, Default)]
pub struct ClassAttr;

pub fn default_identity_hash(_space: PyObjectRef, _w_obj: PyObjectRef) -> PyObjectRef {
    let _ = _space;
    let _ = _w_obj;
    0 as *mut pyre_object::PyObject
}

pub fn get_unique_interplevel_subclass(_space: PyObjectRef, cls: PyObjectRef) -> PyObjectRef {
    let _ = _space;
    cls
}

pub fn _getusercls(_cls: PyObjectRef, _reallywantdict: bool) -> PyObjectRef {
    let _ = _reallywantdict;
    _cls
}

pub fn _share_methods(_copycls: PyObjectRef, _subcls: PyObjectRef) {
    let _ = (_copycls, _subcls);
}

pub fn use_special_method_shortcut(_name: &str, _checkerfunc: Option<PyObjectRef>) -> bool {
    let _ = (_name, _checkerfunc);
    false
}

pub fn make_descr_typecheck_wrapper<T, F, A>(
    _tag: T,
    _func: F,
    _extraargs: A,
    _cls: Option<PyObjectRef>,
) -> PyObjectRef
where
    F: Fn() -> PyObjectRef,
{
    let _ = (_tag, _extraargs, _cls);
    _func()
}

pub fn _make_descr_typecheck_wrapper<T, F, A>(
    _tag: T,
    _func: F,
    _extraargs: A,
    _cls: PyObjectRef,
    _use_closure: bool,
) -> PyObjectRef
where
    F: Fn() -> PyObjectRef,
{
    let _ = (_tag, _extraargs, _cls, _use_closure);
    _func()
}

pub fn interp_attrproperty(
    _name: &str,
    cls: PyObjectRef,
    _doc: Option<&str>,
    _wrapfn: Option<PyObjectRef>,
) -> PyObjectRef {
    let _ = (_name, _doc, _wrapfn);
    cls
}

pub fn interp_attrproperty_w(_name: &str, cls: PyObjectRef, _doc: Option<&str>) -> PyObjectRef {
    let _ = (_name, _doc);
    cls
}

pub fn generic_new_descr(_w_type: PyObjectRef) -> PyObjectRef {
    _w_type
}

pub fn generic_ne(_space: PyObjectRef, w_obj1: PyObjectRef, w_obj2: PyObjectRef) -> PyObjectRef {
    let _ = (_space, w_obj1, w_obj2);
    PY_NULL
}

pub fn fget_co_varnames(_space: PyObjectRef, _code: PyObjectRef) -> PyObjectRef {
    let _ = (_space, _code);
    PY_NULL
}

pub fn fget_co_argcount(_space: PyObjectRef, _code: PyObjectRef) -> PyObjectRef {
    let _ = (_space, _code);
    PY_NULL
}

pub fn fget_co_flags(_space: PyObjectRef, _code: PyObjectRef) -> PyObjectRef {
    let _ = (_space, _code);
    PY_NULL
}

pub fn fget_co_consts(_space: PyObjectRef, _code: PyObjectRef) -> PyObjectRef {
    let _ = (_space, _code);
    PY_NULL
}

pub fn make_weakref_descr(_cls: PyObjectRef) -> PyObjectRef {
    _cls
}

pub fn always_none(_self: PyObjectRef, _obj: PyObjectRef) -> PyObjectRef {
    let _ = (_self, _obj);
    PY_NULL
}

/// Global typeobject cache: maps static PyType pointer → W_TypeObject (as usize).
///
/// PyPy equivalent: space.gettypeobject(cls.typedef) → cached W_TypeObject
/// Stored as usize to satisfy Send+Sync requirements of OnceLock.
pub static TYPEOBJECT_CACHE: OnceLock<HashMap<usize, usize>> = OnceLock::new();

/// Get the cached W_TypeObject for a builtin runtime type.
///
/// PyPy: `space.gettypefor(cls)` / `space.gettypeobject(typedef)`
///
/// Reads the per-type `instantiate` slot (`rclass.py:739-743`
/// `new_instance`), an `AtomicPtr` seeded by `init_typeobjects()`'s
/// `set_instantiate` loop — the same source that backs
/// `TYPEOBJECT_CACHE`. The slot read is a single field load the JIT can
/// model, where the `usize`-keyed `HashMap` lookup is not.
pub fn gettypefor(tp: *const PyType) -> Option<PyObjectRef> {
    if tp.is_null() {
        return None;
    }
    let w = unsafe { pyre_object::get_instantiate(&*tp) };
    if w.is_null() { None } else { Some(w) }
}

/// Get the W_TypeObject for any PyObjectRef.
///
/// RPython: `space.type(w_obj)` → `jit.promote(w_obj.__class__); w_obj.getclass(space)`
///
/// With `w_class` on PyObject, this is a direct field read. Falls back to
/// `gettypefor(ob_type)` for objects created before init_typeobjects()
/// (singletons such as None/True/False/Ellipsis live in read-only static
/// memory, so we never write w_class back into them).
pub fn r#type(obj: PyObjectRef) -> Option<PyObjectRef> {
    if obj.is_null() {
        return None;
    }
    // A tagged immediate is an exact builtin `int`; its type object is
    // `gettypefor(&INT_TYPE)`, synthesized before the `w_class`/`ob_type`
    // derefs below (which would fault on the immediate). Gated on
    // `CAN_BE_TAGGED` (default false), so the derefs stay the only live path.
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return gettypefor(&pyre_object::INT_TYPE);
    }
    unsafe {
        // Exception instances share a single W_BaseException layout
        // but carry an `ExcKind` tag that names the real Python class.
        // `__new__` paths (exc_new_wrapper) overwrite `w_class` with the
        // exact class that was called — including user subclasses such as
        // `class MyErr(Exception): pass`. Trust `w_class` whenever it has
        // been specialised away from the generic `EXCEPTION_TYPE` stub
        // installed by `w_exception_new`; fall back to the kind-tag
        // registry only for internal raise paths (`PyError::value_error`
        // etc.) that bypass `__new__`.
        if pyre_object::is_exception(obj) {
            let w_class = (*obj).w_class;
            let exc_stub =
                pyre_object::get_instantiate(&pyre_object::interp_exceptions::EXCEPTION_TYPE);
            if !w_class.is_null() && !std::ptr::eq(w_class, exc_stub) {
                return Some(w_class);
            }
            let kind = pyre_object::w_exception_get_kind(obj);
            let cls = pyre_object::interp_exceptions::lookup_exc_class_for_kind(kind);
            if !cls.is_null() {
                return Some(cls);
            }
        }
        let w_class = (*obj).w_class;
        if !w_class.is_null() {
            return Some(w_class);
        }
        // Fallback for objects created before init_typeobjects (None, True,
        // False, Ellipsis, NotImplemented). These are `static`s in RODATA,
        // so writing to (*obj).w_class would SIGBUS — just look it up via
        // gettypefor(), which reads an AtomicPtr on the PyType.
        let tp = (*obj).ob_type;
        gettypefor(tp)
    }
}

/// Initialize the type registry with all builtin types.
///
/// PyPy: each W_XxxObject.typedef = TypeDef("xxx", ...) is set at
/// module load time. In pyre, we do it once at startup.
///
/// Must be called before any getattr on builtin objects.
pub fn init_typeobjects() {
    TYPEOBJECT_CACHE.get_or_init(|| {
        // Seed preorder `subclassrange_{min,max}` on every PyType
        // reachable from `INSTANCE_TYPE` so `ll_isinstance` works on
        // the interpreter-only test path that skips the JIT init.
        // JIT init re-computes these via `gc.subclass_range` and
        // overwrites with identical values (idempotent).  Calling
        // `mark_subclass_ranges_initialized` afterwards stops the
        // pyre-object-internal `is_exception` fallback from
        // overwriting with the object-only subset (which would lose
        // the cross-crate `CODE_TYPE` / `PYTRACEBACK_TYPE` ranges).
        pyre_object::pyobject::compute_subclass_ranges_from(
            &[
                pyre_object::pyobject::all_foreign_pytypes(),
                crate::all_foreign_pytypes(),
            ],
            &[&pyre_object::INSTANCE_TYPE],
        );
        pyre_object::pyobject::mark_subclass_ranges_initialized();
        let mut reg: HashMap<usize, usize> = HashMap::new();

        // 'object' first — PyPy: objectobject.py W_ObjectObject.typedef
        // MRO = [object]. All other types inherit from object.
        let object_type = new_root_typeobject("object", init_object_type);
        reg.insert(
            &INSTANCE_TYPE as *const PyType as usize,
            object_type as usize,
        );
        let _ = W_OBJECT_TYPEOBJECT.set(object_type as usize);

        // type — PyPy: typeobject.py, bases=(object,)
        // type.__new__(metatype, name, bases, dict) creates new types
        let type_type = new_typeobject_with_base("type", init_type_type, object_type);
        // hasdict/weakrefable/acceptable now set by typedef.py:34,37,43 logic
        // in new_typeobject_with_base_and_layout from init_type_type's dict contents.
        // typeobject.py:691-701 W_TypeObject._lifeline_/getweakref/setweakref/
        // delweakref — every type object supports weakrefs regardless of the
        // `weakrefable` flag inferred from its dict.  Mark the metaclass so that
        // instances of `type` (i.e. all classes) route through the weakref
        // side table; subclassed metaclasses inherit it via copy_flags_from_bases.
        unsafe { pyre_object::w_type_set_weakrefable(type_type, true) };
        reg.insert(&TYPE_TYPE as *const PyType as usize, type_type as usize);
        let _ = W_TYPE_TYPEOBJECT.set(type_type as usize);

        // int — intobject.py W_IntObject.typedef, bases=(object,)
        // Layout = INT_TYPE because instances are W_IntObject.
        let int_type = new_typeobject_with_base_and_layout(
            "int",
            init_int_type,
            object_type,
            &INT_TYPE as *const PyType,
        );
        reg.insert(&INT_TYPE as *const PyType as usize, int_type as usize);
        // W_LongObject shares the `int` Python identity but has its own
        // layout PyType (`LONG_TYPE`). Map it to the same `int_type`
        // W_TypeObject so the `set_instantiate` loop caches the class on
        // `LONG_TYPE` too — `w_long_new` stamps `w_class =
        // get_instantiate(INT_TYPE)`, and `is_exact_builtin_instance`
        // reads `get_instantiate(ob_type)` where `ob_type == LONG_TYPE`;
        // both must resolve to the same object.
        reg.insert(
            &pyre_object::LONG_TYPE as *const PyType as usize,
            int_type as usize,
        );

        // float — floatobject.py, bases=(object,)
        reg.insert(
            &FLOAT_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "float",
                init_float_type,
                object_type,
                &FLOAT_TYPE as *const PyType,
            ) as usize,
        );

        // complex — complexobject.c, bases=(object,)
        reg.insert(
            &pyre_object::COMPLEX_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "complex",
                init_complex_type,
                object_type,
                &pyre_object::COMPLEX_TYPE as *const PyType,
            ) as usize,
        );

        // array.array — interp_array.py, bases=(object,)
        reg.insert(
            &pyre_object::interp_array::ARRAY_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "array.array",
                crate::module::array::init_array_type,
                object_type,
                &pyre_object::interp_array::ARRAY_TYPE as *const PyType,
            ) as usize,
        );

        // bool — boolobject.py, bases=(int,)
        // Layout = BOOL_TYPE (not INT_TYPE: different struct size).
        // boolobject.py:110 W_BoolObject.typedef.acceptable_as_base_class = False
        let bool_type = new_typeobject_with_base_and_layout(
            "bool",
            init_bool_type,
            int_type,
            &BOOL_TYPE as *const PyType,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(bool_type, false) };
        reg.insert(&BOOL_TYPE as *const PyType as usize, bool_type as usize);

        // str — PyPy: unicodeobject.py, bases=(object,)
        reg.insert(
            &STR_TYPE as *const PyType as usize,
            new_typeobject_with_base("str", init_str_type, object_type) as usize,
        );

        // list — PyPy: listobject.py, bases=(object,)
        reg.insert(
            &LIST_TYPE as *const PyType as usize,
            new_typeobject_with_base("list", init_list_type, object_type) as usize,
        );

        // tuple — PyPy: tupleobject.py, bases=(object,)
        reg.insert(
            &TUPLE_TYPE as *const PyType as usize,
            new_typeobject_with_base("tuple", init_tuple_type, object_type) as usize,
        );

        // dict — PyPy: dictmultiobject.py, bases=(object,)
        let dict_type = new_typeobject_with_base("dict", init_dict_type, object_type);
        reg.insert(&DICT_TYPE as *const PyType as usize, dict_type as usize);
        // `pypy/objspace/std/dictmultiobject.py:67
        // allocate_instance(W_ModuleDictObject, space.w_dict)` —
        // module dicts surface as Python's `dict`.  Register the
        // sibling `MODULE_DICT_TYPE` static under the same dict
        // W_TypeObject so `type(g) is dict` and
        // `isinstance(g, dict)` hold on `W_ModuleDictObject`
        // instances even though they carry a different Rust
        // layout / GC type id.
        reg.insert(
            &pyre_object::dictmultiobject::MODULE_DICT_TYPE as *const PyType as usize,
            dict_type as usize,
        );
        unsafe {
            pyre_object::set_instantiate(
                &pyre_object::dictmultiobject::MODULE_DICT_TYPE,
                dict_type,
            );
        }

        // mappingproxy — `pypy/objspace/std/dictproxyobject.py:103`
        // `W_DictProxyObject.typedef = TypeDef('mappingproxy', ...)`,
        // bases=(object,).  The TypeDef surface (keys/values/items/get/
        // copy/__or__/__ror__/__ior__/__reversed__/cmp methods) is
        // populated by `init_mappingproxy_type` so `cls.__dict__.keys()`
        // and friends dispatch through the registered descriptors.
        reg.insert(
            &pyre_object::MAPPING_PROXY_TYPE as *const PyType as usize,
            new_typeobject_with_base("mappingproxy", init_mappingproxy_type, object_type) as usize,
        );
        // module — `pypy/interpreter/module.py Module.typedef`, bases=(object,).
        // `Module` carries a custom Rust layout (name + w_dict), so
        // instances are produced by `w_module_new` at import time, not by the
        // generic `object.__new__`.  Registering the W_TypeObject gives
        // `type(m)` a real type (was the bare name string), so `m.__class__`,
        // `__flags__`, `isinstance(m, object)` and the inherited
        // `object.__reduce_ex__` all resolve.  `get_instantiate(&MODULE_TYPE)`
        // (read by `w_module_new`) is wired by the `set_instantiate` loop below.
        reg.insert(
            &pyre_object::MODULE_TYPE as *const PyType as usize,
            new_typeobject_with_base("module", init_module_type, object_type) as usize,
        );
        // `pypy/objspace/std/dictmultiobject.py:449/459/469` —
        // dict_keys / dict_values / dict_items.  PyPy registers
        // each as a distinct TypeDef but they share the
        // _iter_keys/_iter_values/_iter_items dispatch.  Pyre's
        // baseobjspace iter/len/contains arms cover the runtime
        // semantics, so the per-typedef init body stays empty for
        // now — what matters is that `type(d.keys())` resolves to
        // the right W_TypeObject (otherwise `builtin_type` falls
        // back to a str return).  Mark these non-base-acceptable to
        // mirror PyPy's `acceptable_as_base_class = False`.
        // dict_keys / dict_items get the SetLikeDictView surface
        // per dictmultiobject.py:1802-1829 / 1773-1800; dict_values
        // stops at the common slots per dictmultiobject.py:1831-1840
        // (values views are intentionally NOT set-like).
        let dict_keys_type =
            new_typeobject_with_base("dict_keys", init_dict_view_set_like_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(dict_keys_type, false) };
        reg.insert(
            &pyre_object::dictmultiobject::DICT_KEYS_TYPE as *const PyType as usize,
            dict_keys_type as usize,
        );
        let dict_values_type =
            new_typeobject_with_base("dict_values", init_dict_view_values_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(dict_values_type, false) };
        reg.insert(
            &pyre_object::dictmultiobject::DICT_VALUES_TYPE as *const PyType as usize,
            dict_values_type as usize,
        );
        let dict_items_type =
            new_typeobject_with_base("dict_items", init_dict_view_set_like_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(dict_items_type, false) };
        reg.insert(
            &pyre_object::dictmultiobject::DICT_ITEMS_TYPE as *const PyType as usize,
            dict_items_type as usize,
        );

        // traceback — `pypy/interpreter/pytraceback.py:17-101
        // PyTraceback.typedef`.  Read-only-ish: `tb_next` accepts a
        // chain rewrite, `tb_lineno` / `tb_lasti` are read+write to
        // mirror PyPy's getsetters.  `acceptable_as_base_class=False`
        // matches PyPy's `pytraceback.py` which never sets it (TypeDef
        // defaults).
        let traceback_type =
            new_typeobject_with_base("traceback", init_pytraceback_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(traceback_type, false) };
        reg.insert(
            &crate::pytraceback::PYTRACEBACK_TYPE as *const PyType as usize,
            traceback_type as usize,
        );

        // frame — PyPy: typedef.py:736-753 PyFrame.typedef.
        // `assert not PyFrame.typedef.acceptable_as_base_class` (typedef.py:754)
        // — no `__new__`, cannot be subclassed.
        let frame_type = new_typeobject_with_base("frame", init_frame_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(frame_type, false) };
        reg.insert(
            &crate::pyframe::FRAME_TYPE as *const PyType as usize,
            frame_type as usize,
        );

        // function — PyPy: funcobject.py
        // Functions are descriptors: function.__get__ returns a bound method.
        let function_type = new_typeobject_with_base("function", init_function_type, object_type);
        // typedef.py:742 Function.typedef.acceptable_as_base_class = False
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(function_type, false) };
        // typedef.py:735/740 — Function exposes __dict__ and __weakref__.
        unsafe {
            pyre_object::w_type_set_hasdict(function_type, true);
            pyre_object::w_type_set_weakrefable(function_type, true);
            // typedef.py:807 `method_descriptor=True` → typeobject.py:256
            // `flag_method_descriptor` (the LOAD_METHOD fast-path gate,
            // callmethod.py:66).
            pyre_object::typeobject::w_type_set_flag_method_descriptor(function_type, true);
        }
        reg.insert(
            &crate::FUNCTION_TYPE as *const PyType as usize,
            function_type as usize,
        );

        // builtin_function — PyPy: typedef.py BuiltinFunction.typedef
        // Mirrors Function.typedef except `__get__` is intentionally absent.
        let builtin_function_type =
            new_typeobject_with_base("builtin_function", init_builtin_function_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(builtin_function_type, false) };
        unsafe {
            pyre_object::w_type_set_hasdict(builtin_function_type, true);
            pyre_object::w_type_set_weakrefable(builtin_function_type, true);
        }
        reg.insert(
            &crate::BUILTIN_FUNCTION_TYPE as *const PyType as usize,
            builtin_function_type as usize,
        );

        // builtin-code — PyPy: BuiltinCode.typedef = TypeDef('builtin-code', ...)
        reg.insert(
            &crate::BUILTIN_CODE_TYPE as *const PyType as usize,
            new_typeobject_with_base("builtin-code", init_builtin_code_type, object_type) as usize,
        );

        // typedef.py:765 Method.typedef.acceptable_as_base_class = False
        let method_type = new_typeobject_with_base("method", init_method_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(method_type, false) };
        // typedef.py:763 — Method exposes __weakref__.
        unsafe { pyre_object::w_type_set_weakrefable(method_type, true) };
        reg.insert(
            &pyre_object::function::METHOD_TYPE as *const PyType as usize,
            method_type as usize,
        );

        // typedef.py:664 PyCode.typedef.acceptable_as_base_class = False
        let code_type = new_typeobject_with_base("code", init_code_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(code_type, false) };
        reg.insert(
            &crate::pycode::CODE_TYPE as *const PyType as usize,
            code_type as usize,
        );

        // typedef.py:500 Member.typedef.acceptable_as_base_class = False
        let member_desc_type = new_typeobject_with_base(
            "member_descriptor",
            init_member_descriptor_type,
            object_type,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(member_desc_type, false) };
        reg.insert(
            &pyre_object::typedef::MEMBER_TYPE as *const PyType as usize,
            member_desc_type as usize,
        );

        // staticmethod — PyPy: function.py StaticMethod, bases=(object,)
        reg.insert(
            &pyre_object::function::STATICMETHOD_TYPE as *const PyType as usize,
            new_typeobject_with_base("staticmethod", init_staticmethod_type, object_type) as usize,
        );

        // classmethod — PyPy: function.py ClassMethod, bases=(object,)
        reg.insert(
            &pyre_object::function::CLASSMETHOD_TYPE as *const PyType as usize,
            new_typeobject_with_base("classmethod", init_classmethod_type, object_type) as usize,
        );

        // property — PyPy: descriptor.py W_Property, bases=(object,)
        reg.insert(
            &pyre_object::descriptor::PROPERTY_TYPE as *const PyType as usize,
            new_typeobject_with_base("property", init_property_type, object_type) as usize,
        );

        // exception — pyre uses one shared W_TypeObject for all builtin
        // exception instances; the per-class hierarchy lives in the namespace
        // (see make_exc_type in builtins.rs).  Registering it here lets
        // typedef::r#type return a non-null type for raised exception objects.
        reg.insert(
            &pyre_object::interp_exceptions::EXCEPTION_TYPE as *const PyType as usize,
            new_typeobject_with_base("exception", |_| {}, object_type) as usize,
        );

        // NoneType — bases=(object,)
        reg.insert(
            &NONE_TYPE as *const PyType as usize,
            new_typeobject_with_base("NoneType", |_| {}, object_type) as usize,
        );

        // typedef.py:941-946 Ellipsis.typedef.
        let ellipsis_type = new_typeobject_with_base("ellipsis", init_ellipsis_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(ellipsis_type, false) };
        reg.insert(
            &ELLIPSIS_TYPE as *const PyType as usize,
            ellipsis_type as usize,
        );

        // typedef.py:948-954 NotImplemented.typedef.
        let notimplemented_type =
            new_typeobject_with_base("NotImplementedType", init_notimplemented_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(notimplemented_type, false) };
        reg.insert(
            &pyre_object::pyobject::NOTIMPLEMENTED_TYPE as *const PyType as usize,
            notimplemented_type as usize,
        );

        // types.UnionType — PyPy: _pypy_generic_alias.py UnionType, bases=(object,)
        // `__slots__` includes `__weakref__` (`_pypy_generic_alias.py:247`),
        // so a union is weak-referenceable.
        let union_type = new_typeobject_with_base("types.UnionType", init_union_type, object_type);
        unsafe { pyre_object::w_type_set_weakrefable(union_type, true) };
        reg.insert(
            &pyre_object::UNION_TYPE as *const PyType as usize,
            union_type as usize,
        );

        // types.GenericAlias — PyPy: _pypy_generic_alias.py GenericAlias,
        // bases=(object,).  `__slots__` includes `__weakref__`
        // (`_pypy_generic_alias.py:17`), so an alias is weak-referenceable.
        let generic_alias_type = new_typeobject_with_base(
            "types.GenericAlias",
            crate::_pypy_generic_alias::init_generic_alias_type,
            object_type,
        );
        unsafe { pyre_object::w_type_set_weakrefable(generic_alias_type, true) };
        reg.insert(
            &pyre_object::GENERIC_ALIAS_TYPE as *const PyType as usize,
            generic_alias_type as usize,
        );

        // slice — PyPy: sliceobject.py, bases=(object,)
        reg.insert(
            &pyre_object::sliceobject::SLICE_TYPE as *const PyType as usize,
            new_typeobject_with_base("slice", init_slice_type, object_type) as usize,
        );

        // re.Pattern / re.Match — PyPy: module/_sre/interp_sre.py
        // W_SRE_Pattern.typedef (:641) / W_SRE_Match.typedef (:869);
        // neither is acceptable_as_base_class (:669/:896).
        let sre_pattern_type = new_typeobject_with_base(
            "re.Pattern",
            crate::module::_sre::interp_sre::init_sre_pattern_type,
            object_type,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(sre_pattern_type, false) };
        reg.insert(
            &pyre_object::interp_sre::SRE_PATTERN_TYPE as *const PyType as usize,
            sre_pattern_type as usize,
        );
        let sre_match_type = new_typeobject_with_base(
            "re.Match",
            crate::module::_sre::interp_sre::init_sre_match_type,
            object_type,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(sre_match_type, false) };
        reg.insert(
            &pyre_object::interp_sre::SRE_MATCH_TYPE as *const PyType as usize,
            sre_match_type as usize,
        );

        // _sre.SRE_Scanner — W_SRE_Scanner.typedef (:949); the iterator
        // behind Pattern.finditer/scanner; not acceptable_as_base_class
        // (:957).
        let sre_scanner_type = new_typeobject_with_base(
            "_sre.SRE_Scanner",
            crate::module::_sre::interp_sre::init_sre_scanner_type,
            object_type,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(sre_scanner_type, false) };
        reg.insert(
            &pyre_object::interp_sre::SRE_SCANNER_TYPE as *const PyType as usize,
            sre_scanner_type as usize,
        );

        // bytearray — PyPy: bytearrayobject.py, bases=(object,)
        reg.insert(
            &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const PyType as usize,
            new_typeobject_with_base("bytearray", init_bytearray_type, object_type) as usize,
        );

        // bytes — PyPy: bytesobject.py W_BytesObject, bases=(object,)
        reg.insert(
            &pyre_object::bytesobject::BYTES_TYPE as *const PyType as usize,
            new_typeobject_with_base("bytes", init_bytes_type, object_type) as usize,
        );

        // set / frozenset — PyPy: setobject.py, bases=(object,).
        // Both carry their own layout typedef so check_user_subclass's layout
        // safety check (typeobject.py:520-523) can reject foreign-layout
        // subclasses (e.g. subclass adds __slots__).
        reg.insert(
            &pyre_object::setobject::SET_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "set",
                init_set_type,
                object_type,
                &pyre_object::setobject::SET_TYPE as *const PyType,
            ) as usize,
        );
        reg.insert(
            &pyre_object::setobject::FROZENSET_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "frozenset",
                init_frozenset_type,
                object_type,
                &pyre_object::setobject::FROZENSET_TYPE as *const PyType,
            ) as usize,
        );

        // Foreign PyType statics that have no per-type init function but
        // still need a W_TypeObject so `gettypefor(&XXX_TYPE)` returns
        // it — used by `type(g).__name__`,
        // `isinstance(x, type(x))`, and the descriptor protocol's
        // `space.type(w_obj)` invariants.  Without a registered
        // W_TypeObject the 1-arg `type(obj)` fallback at
        // `builtins.rs:1003` would return the type's *name* as a
        // `str`, breaking every downstream identity check.
        //
        // Empty init body matches PyPy typedefs that expose only
        // protocol slots filled in by the runtime (e.g. generator's
        // `send`/`throw`/`close` in `pypy/interpreter/generator.py`):
        // pyre carries those slots elsewhere in the dispatch path so
        // the typedef itself stays empty.
        reg.insert(
            &pyre_object::descriptor::SUPER_TYPE as *const PyType as usize,
            new_typeobject_with_base("super", |_| {}, object_type) as usize,
        );
        let generator_type = new_typeobject_with_base("generator", |_| {}, object_type);
        // `Py_TPFLAGS_DISALLOW_INSTANTIATION` — a generator is produced
        // only by calling a generator function, never by `generator()`,
        // so `tp_new` is NULL and pickling refuses it.
        unsafe { pyre_object::w_type_set_disallow_instantiation(generator_type) };
        reg.insert(
            &pyre_object::generator::GENERATOR_TYPE as *const PyType as usize,
            generator_type as usize,
        );
        reg.insert(
            &pyre_object::functional::RANGE_ITER_TYPE as *const PyType as usize,
            new_typeobject_with_base("range_iterator", |_| {}, object_type) as usize,
        );
        // rangeobject.c PyRange_Type carries no Py_TPFLAGS_BASETYPE, so
        // `range` is not an acceptable base class.
        let range_type = new_typeobject_with_base(
            "range",
            |ns| {
                dict_storage_store(ns, "__new__", make_new_descr(range_descr_new));
            },
            object_type,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(range_type, false) };
        reg.insert(
            &pyre_object::functional::RANGE_TYPE as *const PyType as usize,
            range_type as usize,
        );
        // memoryobject.py:731 W_MemoryView.typedef.acceptable_as_base_class = False
        let memoryview_type = new_typeobject_with_base(
            "memoryview",
            crate::builtins::init_memoryview_type,
            object_type,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(memoryview_type, false) };
        reg.insert(
            &pyre_object::memoryview::MEMORYVIEW_TYPE as *const PyType as usize,
            memoryview_type as usize,
        );
        let seq_iterator_type = new_typeobject_with_base("iterator", |_| {}, object_type);
        // `Py_TPFLAGS_DISALLOW_INSTANTIATION` — an iterator is produced only by
        // `iter(obj)`, never by `iterator()`, so `tp_new` is NULL.
        unsafe { pyre_object::w_type_set_disallow_instantiation(seq_iterator_type) };
        reg.insert(
            &pyre_object::iterobject::SEQ_ITER_TYPE as *const PyType as usize,
            seq_iterator_type as usize,
        );
        reg.insert(
            &pyre_object::functional::LONG_RANGE_ITER_TYPE as *const PyType as usize,
            // `W_AbstractRangeIterator.typedef` names every range-iterator
            // class `range_iterator` (`functional.py`); the word-fit and bignum
            // iterators share that public name though they are distinct types.
            new_typeobject_with_base("range_iterator", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::functional::ENUMERATE_TYPE as *const PyType as usize,
            new_typeobject_with_base(
                "enumerate",
                |ns| {
                    dict_storage_store(ns, "__new__", make_new_descr(enumerate_descr_new));
                },
                object_type,
            ) as usize,
        );
        reg.insert(
            &pyre_object::functional::REVERSED_TYPE as *const PyType as usize,
            new_typeobject_with_base(
                "reversed",
                |ns| {
                    dict_storage_store(ns, "__new__", make_new_descr(reversed_descr_new));
                },
                object_type,
            ) as usize,
        );
        reg.insert(
            &pyre_object::functional::FILTER_TYPE as *const PyType as usize,
            new_typeobject_with_base(
                "filter",
                |ns| {
                    dict_storage_store(ns, "__new__", make_new_descr(filter_descr_new));
                },
                object_type,
            ) as usize,
        );
        reg.insert(
            &pyre_object::functional::MAP_TYPE as *const PyType as usize,
            new_typeobject_with_base(
                "map",
                |ns| {
                    dict_storage_store(ns, "__new__", make_new_descr(map_descr_new));
                },
                object_type,
            ) as usize,
        );
        reg.insert(
            &pyre_object::functional::ZIP_TYPE as *const PyType as usize,
            new_typeobject_with_base(
                "zip",
                |ns| {
                    dict_storage_store(ns, "__new__", make_new_descr(zip_descr_new));
                },
                object_type,
            ) as usize,
        );
        reg.insert(
            &pyre_object::dictmultiobject::DICT_KEYITERATOR_TYPE as *const PyType as usize,
            new_typeobject_with_base("dict_keyiterator", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::dictmultiobject::DICT_VALUEITERATOR_TYPE as *const PyType as usize,
            new_typeobject_with_base("dict_valueiterator", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::dictmultiobject::DICT_ITEMITERATOR_TYPE as *const PyType as usize,
            new_typeobject_with_base("dict_itemiterator", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::nestedscope::CELL_TYPE as *const PyType as usize,
            new_typeobject_with_base("cell", init_cell_type, object_type) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::COUNT_TYPE as *const PyType as usize,
            new_typeobject_with_base("itertools.count", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::REPEAT_TYPE as *const PyType as usize,
            new_typeobject_with_base("itertools.repeat", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::TAKEWHILE_TYPE as *const PyType as usize,
            new_typeobject_with_base("itertools.takewhile", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::DROPWHILE_TYPE as *const PyType as usize,
            new_typeobject_with_base("itertools.dropwhile", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::FILTERFALSE_TYPE as *const PyType as usize,
            new_typeobject_with_base("itertools.filterfalse", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::PAIRWISE_TYPE as *const PyType as usize,
            new_typeobject_with_base("itertools.pairwise", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::CYCLE_TYPE as *const PyType as usize,
            new_typeobject_with_base("itertools.cycle", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::CHAIN_TYPE as *const PyType as usize,
            new_typeobject_with_base("itertools.chain", |_| {}, object_type) as usize,
        );
        // `pypy/objspace/std/specialisedtupleobject.py` — three SpecialisedTuple
        // variants share the public `tuple` PyType name, so all three
        // foreign statics map to a "tuple" typedef.  `gettypefor` keys
        // by static address (each variant has its own
        // `&SPECIALISED_TUPLE_..._TYPE`), so a separate
        // W_TypeObject per variant is required — they just present
        // the same `__name__` to user code (PyPy parity).
        reg.insert(
            &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_II_TYPE as *const PyType
                as usize,
            new_typeobject_with_base("tuple", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_FF_TYPE as *const PyType
                as usize,
            new_typeobject_with_base("tuple", |_| {}, object_type) as usize,
        );
        reg.insert(
            &pyre_object::specialisedtupleobject::SPECIALISED_TUPLE_OO_TYPE as *const PyType
                as usize,
            new_typeobject_with_base("tuple", |_| {}, object_type) as usize,
        );

        // rclass.py:739-743 parity — cache W_TypeObject on each PyType
        // so allocators can set w_class at allocation time (like RPython's
        // `self.setfield(vptr, '__class__', ctypeptr, llops)` in new_instance).
        for (&pytype_addr, &w_typeobject_addr) in &reg {
            let tp = unsafe { &*(pytype_addr as *const PyType) };
            let w_typeobject = w_typeobject_addr as PyObjectRef;
            pyre_object::pyobject::set_instantiate(tp, w_typeobject);
        }
        // pypy/objspace/std/objspace.py:104-108 — set
        // `flag_map_or_seq` on W_TypeObject for dict / list / tuple.
        // PyPy stores this marker on `W_TypeObject` (typeobject.py:169),
        // not on the low-level OBJECT_VTABLE / PyType.  Heap types copy
        // it from their bases in `inherit_flag_map_or_seq`, mirroring
        // typeobject.py:1495.
        for (pytype, flag) in [
            (&pyre_object::pyobject::DICT_TYPE, b'M'),
            (&pyre_object::pyobject::LIST_TYPE, b'S'),
            (&pyre_object::pyobject::TUPLE_TYPE, b'S'),
            // rangeobject.c PyRange_Type carries Py_TPFLAGS_SEQUENCE.
            (&pyre_object::functional::RANGE_TYPE, b'S'),
        ] {
            let w_typeobject = *reg
                .get(&(pytype as *const PyType as usize))
                .expect("built-in type object must be registered before flag_map_or_seq init")
                as PyObjectRef;
            unsafe {
                pyre_object::typeobject::w_type_set_flag_map_or_seq(w_typeobject, flag);
            }
        }
        // Set w_class on all built-in type objects to `type`.
        // baseobjspace.py:76 getclass() — for type objects, the class
        // is the metatype (default: `type`).
        let w_type_type = W_TYPE_TYPEOBJECT
            .get()
            .map(|v| *v as PyObjectRef)
            .unwrap_or(PY_NULL);
        for &w_typeobject_addr in reg.values() {
            let w_typeobj = w_typeobject_addr as PyObjectRef;
            unsafe {
                if (*w_typeobj).w_class.is_null() {
                    (*w_typeobj).w_class = w_type_type;
                }
            }
        }

        reg
    });

    patch_builtin_function_descriptors();
    patch_frame_traceback_descriptors();
    patch_getset_descriptor_metadata();
    patch_typeobject_descriptor_names();
}

/// `typedef.py:58 add_entries` parity — walk every registered
/// W_TypeObject's namespace and stamp each `GetSetProperty`'s
/// `name` slot with the dict-key it lives under, when the slot
/// still holds the `<generic property>` sentinel.  PyPy's
/// `add_entries` runs at TypeDef construction time and writes
/// `getset.name = key` so descriptor introspection
/// (`type.__dict__['<key>'].__name__`) returns the same string the
/// dict was keyed by.  Pyre's `init_<type>_type` helpers store
/// descriptors via `make_getset_descriptor` (no name), so without
/// this pass every descriptor's `__name__` would surface as the
/// sentinel.  Explicit names passed via `make_*_named` survive
/// (the sentinel-only check skips them).
fn patch_typeobject_descriptor_names() {
    let Some(reg) = TYPEOBJECT_CACHE.get() else {
        return;
    };
    for (_pytype_addr, &w_typeobject_addr) in reg {
        let tp = w_typeobject_addr as PyObjectRef;
        if tp.is_null() {
            continue;
        }
        let dict_ptr = unsafe { pyre_object::w_type_get_dict_ptr(tp) } as *mut DictStorage;
        if dict_ptr.is_null() {
            continue;
        }
        let ns = unsafe { &*dict_ptr };
        let entries: Vec<(String, PyObjectRef)> = ns
            .keys()
            .filter_map(|k| ns.get(k).map(|&v| (k.to_string(), v)))
            .collect();
        for (key, value) in entries {
            if value.is_null() {
                continue;
            }
            if !unsafe { pyre_object::typedef::is_getset_property(value) } {
                continue;
            }
            let cur = unsafe { pyre_object::typedef::w_getset_get_name(value) };
            let is_sentinel = cur.is_null()
                || (unsafe { pyre_object::is_str(cur) }
                    && unsafe { pyre_object::w_str_get_value(cur) } == "<generic property>");
            if !is_sentinel {
                continue;
            }
            let new_name = pyre_object::w_str_new(&key);
            unsafe { pyre_object::typedef::w_getset_set_name(value, new_name) };
        }
    }
}

/// The global `object` type object, accessible from builtins.
static W_OBJECT_TYPEOBJECT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
/// The global `type` type object.
static W_TYPE_TYPEOBJECT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Get the wrapped `type` typeobject.
///
/// `dont_look_inside` keeps the JIT from tracing into the `OnceLock`
/// read: the slot is set once at startup and holds the runtime
/// typeobject address, which has no registry-resolvable accessor, so
/// the call stays a residual returning that pointer (the trace-side twin
/// registers the fnaddr in `jit_trace_fnaddrs`).
#[majit_macros::dont_look_inside]
pub fn w_type() -> PyObjectRef {
    W_TYPE_TYPEOBJECT
        .get()
        .map(|v| *v as PyObjectRef)
        .unwrap_or(PY_NULL)
}

pub fn gettypeobject(tp: &PyType) -> PyObjectRef {
    gettypefor(tp as *const PyType).unwrap_or(PY_NULL)
}

/// Get the wrapped `object` typeobject.
///
/// `dont_look_inside` for the same reason as [`w_type`].
#[majit_macros::dont_look_inside]
pub fn w_object() -> PyObjectRef {
    W_OBJECT_TYPEOBJECT
        .get()
        .map(|v| *v as PyObjectRef)
        .unwrap_or(PY_NULL)
}

/// Stamp the builtin `__new__` carrier in `ns` with `__self__ =
/// type_obj` (the type that defines `tp_new`), mirroring
/// `typeobject.c add_tp_new_wrapper`.  `copyreg._reduce_ex` walks the
/// MRO testing `base.__new__.__self__ is base`, so each builtin type
/// that defines `__new__` must carry its own type as the wrapper's
/// `__self__`.  Inherited `__new__` keeps the ancestor's stamp
/// (`function_set_new_self` only writes when unset).
///
/// # Safety
/// `ns_ptr` must be a valid, live `DictStorage`; `type_obj` a valid type.
unsafe fn stamp_new_descr_self(ns_ptr: *mut DictStorage, type_obj: PyObjectRef) {
    if let Some(w_new) = (*ns_ptr).get("__new__").copied() {
        if !w_new.is_null() && pyre_object::function::is_staticmethod(w_new) {
            let inner = pyre_object::function::w_staticmethod_get_func(w_new);
            if !inner.is_null() && crate::function::is_function(inner) {
                crate::function::function_set_new_self(inner, type_obj);
            }
        }
    }
    // typeobject.py:1738-1742 — `if isinstance(descrvalue, GetSetProperty):
    // descrvalue = descrvalue.copy_for_type(w_type)`.  Bind every reqcls-less
    // GetSetProperty in the namespace to its owning type so that
    // `T.__dict__['x'].__objclass__` (descr_get_objclass reads `w_objclass`)
    // resolves instead of raising "generic self has no __objclass__".
    let mut rebinds: Vec<(String, PyObjectRef)> = Vec::new();
    for (key, &descr) in (*ns_ptr).entries() {
        if !descr.is_null() && pyre_object::typedef::is_getset_property(descr) {
            let bound = copy_for_type(descr, type_obj);
            if !std::ptr::eq(bound, descr) {
                rebinds.push((key.to_string(), bound));
            }
        }
    }
    for (key, bound) in rebinds {
        (*ns_ptr).insert(key, bound);
    }
}

/// Create the root `object` type. MRO = [object].
fn new_root_typeobject(name: &str, init: fn(&mut DictStorage)) -> PyObjectRef {
    let mut ns = Box::new(DictStorage::new());
    ns.fix_ptr();
    init(&mut ns);
    let ns_ptr = Box::into_raw(ns);
    let type_obj = w_type_new_builtin(
        name,
        PY_NULL,
        ns_ptr as *mut u8,
        &INSTANCE_TYPE as *const PyType,
    );
    // typeobject.py:1261-1280 setup_builtin_type — root type gets its own Layout.
    unsafe {
        let layout = pyre_object::typeobject::leak_layout(pyre_object::typeobject::Layout {
            typedef: &INSTANCE_TYPE as *const PyType,
            nslots: 0,
            newslotnames: vec![],
            base_layout: std::ptr::null(),
            acceptable_as_base_class: true, // object has __new__
            typedef_hasdict: false,         // object typedef declares no __dict__
        });
        pyre_object::w_type_set_layout(type_obj, layout);
        // object: hasdict=False, weakrefable=False (bare object() has no __dict__)
        pyre_object::w_type_set_hasdict(type_obj, false);
        pyre_object::w_type_set_weakrefable(type_obj, false);
    }
    unsafe { w_type_set_mro(type_obj, vec![type_obj]) };
    unsafe { stamp_new_descr_self(ns_ptr, type_obj) };
    type_obj
}

/// Create a builtin type with a single base. MRO = [self] + base.mro().
/// Layout defaults to INSTANCE_TYPE (general object layout).
fn new_typeobject_with_base(
    name: &str,
    init: impl FnOnce(&mut DictStorage),
    base: PyObjectRef,
) -> PyObjectRef {
    new_typeobject_with_base_and_layout(name, init, base, &INSTANCE_TYPE as *const PyType)
}

/// Create a builtin type with explicit layout PyType.
///
/// typeobject.py:1261-1280 setup_builtin_type parity: each builtin type
/// gets its own Layout based on its instancetypedef. Types that share
/// the same typedef as their base reuse the parent's Layout object.
fn new_typeobject_with_base_and_layout(
    name: &str,
    init: impl FnOnce(&mut DictStorage),
    base: PyObjectRef,
    layout_pytype: *const PyType,
) -> PyObjectRef {
    let mut ns = Box::new(DictStorage::new());
    ns.fix_ptr();
    init(&mut ns);
    let ns_ptr = Box::into_raw(ns);
    let bases = w_tuple_new(vec![base]);
    let type_obj = w_type_new_builtin(name, bases, ns_ptr as *mut u8, layout_pytype);

    // typeobject.py:1273-1280 setup_builtin_type:
    //   parent_layout = w_bestbase.layout
    //   if parent_layout.typedef is instancetypedef:
    //       return parent_layout      ← reuse
    //   return Layout(instancetypedef, 0, base_layout=parent_layout)
    unsafe {
        let parent_layout = pyre_object::w_type_get_layout_ptr(base);
        let reuse = if !parent_layout.is_null() {
            std::ptr::eq((*parent_layout).typedef, layout_pytype)
        } else {
            false
        };
        let has_dict = (*ns_ptr).get("__dict__").is_some();
        let has_weakref = (*ns_ptr).get("__weakref__").is_some();
        let layout = if reuse {
            parent_layout
        } else {
            let has_new = (*ns_ptr).get("__new__").is_some();
            pyre_object::typeobject::leak_layout(pyre_object::typeobject::Layout {
                typedef: layout_pytype,
                nslots: 0,
                newslotnames: vec![],
                base_layout: parent_layout,
                acceptable_as_base_class: has_new,
                // typedef.py:40 `hasdict = '__dict__' in rawdict` — a typedef
                // that declares `__dict__` does its own dict management, so
                // mapdict must not add a second one (typeobject.py:253-257).
                typedef_hasdict: has_dict,
            })
        };
        pyre_object::w_type_set_layout(type_obj, layout);
        // typedef.py:39-41: inherit from bases
        let base_hasdict = pyre_object::w_type_get_hasdict(base);
        let base_weakrefable = pyre_object::w_type_get_weakrefable(base);
        pyre_object::w_type_set_hasdict(type_obj, has_dict || base_hasdict);
        pyre_object::w_type_set_weakrefable(type_obj, has_weakref || base_weakrefable);
    }

    // MRO = [self] + base_mro
    let base_mro = unsafe { w_type_get_mro(base) };
    let mut mro = vec![type_obj];
    if !base_mro.is_null() {
        mro.extend_from_slice(unsafe { &*base_mro });
    } else {
        mro.push(base);
    }
    unsafe { w_type_set_mro(type_obj, mro) };
    unsafe { stamp_new_descr_self(ns_ptr, type_obj) };
    type_obj
}

/// Create a named builtin type inheriting from multiple `bases`.
///
/// The first entry is the primary base (drives layout/hasdict/weakref
/// inheritance, like `w_bestbase` in typeobject.py:setup_builtin_type);
/// the full tuple is recorded as `__bases__` and the MRO is the C3
/// linearization (`compute_default_mro`).  Used for builtin exception
/// classes with more than one base, e.g.
/// `class UnsupportedOperation(OSError, ValueError)`.
pub fn make_builtin_type_with_bases(
    name: &str,
    init: impl FnOnce(&mut DictStorage),
    bases: &[PyObjectRef],
) -> PyObjectRef {
    let layout_pytype = &INSTANCE_TYPE as *const PyType;
    let base = bases[0];
    let mut ns = Box::new(DictStorage::new());
    ns.fix_ptr();
    init(&mut ns);
    let ns_ptr = Box::into_raw(ns);
    let bases_tuple = w_tuple_new(bases.to_vec());
    let type_obj = w_type_new_builtin(name, bases_tuple, ns_ptr as *mut u8, layout_pytype);

    unsafe {
        let parent_layout = pyre_object::w_type_get_layout_ptr(base);
        let reuse = if !parent_layout.is_null() {
            std::ptr::eq((*parent_layout).typedef, layout_pytype)
        } else {
            false
        };
        let has_dict = (*ns_ptr).get("__dict__").is_some();
        let has_weakref = (*ns_ptr).get("__weakref__").is_some();
        let layout = if reuse {
            parent_layout
        } else {
            let has_new = (*ns_ptr).get("__new__").is_some();
            pyre_object::typeobject::leak_layout(pyre_object::typeobject::Layout {
                typedef: layout_pytype,
                nslots: 0,
                newslotnames: vec![],
                base_layout: parent_layout,
                acceptable_as_base_class: has_new,
                typedef_hasdict: false,
            })
        };
        pyre_object::w_type_set_layout(type_obj, layout);
        // typedef.py:39-41: inherit hasdict/weakrefable from any base.
        let mut hasdict = has_dict;
        let mut weakrefable = has_weakref;
        for &b in bases {
            hasdict |= pyre_object::w_type_get_hasdict(b);
            weakrefable |= pyre_object::w_type_get_weakrefable(b);
        }
        pyre_object::w_type_set_hasdict(type_obj, hasdict);
        pyre_object::w_type_set_weakrefable(type_obj, weakrefable);
    }

    // MRO = C3 linearization over the recorded `__bases__`.
    let mro = unsafe { crate::baseobjspace::compute_default_mro(type_obj) };
    unsafe { w_type_set_mro(type_obj, mro) };
    unsafe { stamp_new_descr_self(ns_ptr, type_obj) };
    type_obj
}

/// Create a named builtin type inheriting from `object`.
///
/// Used by extension modules (e.g. _sre) to define their own types.
/// typeobject.py:174 `is_heaptype=False` — builtin type.
pub fn make_builtin_type(name: &str, init: impl FnOnce(&mut DictStorage)) -> PyObjectRef {
    new_typeobject_with_base(name, init, w_object())
}

/// Create a named builtin type inheriting from `base`.
pub fn make_builtin_type_with_base(
    name: &str,
    init: impl FnOnce(&mut DictStorage),
    base: PyObjectRef,
) -> PyObjectRef {
    new_typeobject_with_base(name, init, base)
}

/// Create a named builtin type whose instances live behind a custom
/// `layout_pytype` (the `*const PyType` stored in `ob_header.ob_type`
/// for new instances).  Used for W_Root subclasses that allocate
/// their own typed payload (e.g. `GetSetProperty`) rather than
/// piggy-backing on `INSTANCE_TYPE`.  Mirrors `typeobject.py:1273-1280
/// setup_builtin_type`'s explicit-layout branch.
pub fn make_builtin_type_with_layout(
    name: &str,
    init: impl FnOnce(&mut DictStorage),
    base: PyObjectRef,
    layout_pytype: *const PyType,
) -> PyObjectRef {
    new_typeobject_with_base_and_layout(name, init, base, layout_pytype)
}

/// int.__new__(cls, *args) — PyPy: intobject.py descr__new__
///
/// If cls is the builtin int type, returns a plain W_IntObject.
/// If cls is a subclass of int, returns a W_ObjectObject with the
/// int value stored internally (for int subclasses like IntFlag).
fn int_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = if args.is_empty() {
        std::ptr::null_mut() as PyObjectRef
    } else {
        args[0]
    };
    // intobject.py _new_int → check_user_subclass
    if !cls.is_null() && unsafe { pyre_object::is_type(cls) } {
        if let Some(w_int) = gettypefor(&pyre_object::INT_TYPE) {
            check_user_subclass(w_int, cls)?;
        }
    }
    let value = crate::builtins::builtin_int(&args[1..])?;
    // If cls is int itself (or null), return a plain int.
    if cls.is_null() || !unsafe { pyre_object::is_type(cls) } {
        return Ok(value);
    }
    let int_typeobj = gettypefor(&pyre_object::INT_TYPE);
    if int_typeobj.map_or(false, |t| std::ptr::eq(cls, t)) {
        return Ok(value);
    }
    // cls is a subclass of int. Create a unique instance (bypassing the
    // small-int cache so each has its own identity). Set w_class = cls so
    // type()/isinstance() see the subclass while preserving the underlying
    // int/long storage layout for arithmetic. A magnitude that overflows
    // i64 is a W_LongObject; cloning its BigInt keeps the value intact
    // (w_int_get_value would truncate it to a garbage machine word).
    let obj = if unsafe { pyre_object::is_long(value) } {
        let big = unsafe { pyre_object::w_long_get_value(value) }.clone();
        pyre_object::w_long_new(big)
    } else {
        let int_val = unsafe { pyre_object::w_int_get_value(value) };
        pyre_object::w_int_new_unique(int_val)
    };
    unsafe {
        (*obj).w_class = cls;
    }
    Ok(obj)
}

/// `float.__new__(cls, *args)` — PyPy: floatobject.py descr__new__.
///
/// If cls is the builtin float type, returns a plain W_FloatObject.
/// If cls is a float subclass (e.g. test_math's `class FloatCeil(float)`),
/// returns a fresh W_FloatObject with `w_class = cls` so `type(obj) == cls`
/// and `__ceil__`/`__floor__`/`__trunc__` dunders on the subclass dispatch.
fn float_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        args[0]
    };
    // floatobject.py descr__new__: `w_x` is positional-only, so a surplus
    // positional or any keyword is rejected by builtinclass_new_args_check
    // (skipped when a subtype overrides __init__, which absorbs the surplus).
    // Feed `builtin_float` only the value positionals so the trailing
    // `__pyre_kw__` marker dict never leaks as the value on the subtype path.
    let (value_positional, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    builtinclass_new_args_check(
        "float",
        gettypeobject(&pyre_object::FLOAT_TYPE),
        cls,
        value_positional.len().saturating_sub(1),
        crate::builtins::has_real_kwargs(kwargs),
    )?;
    let value = crate::builtins::builtin_float(value_positional)?;
    // tp_new_wrapper (subclass_to_tag) rejects a non-type or non-subtype cls
    // and returns None for base `float`; a strict subclass retags a fresh
    // W_FloatObject so setattr / w_class on it don't clobber the value-cached
    // singleton.
    let sub = match subclass_to_tag(cls, &pyre_object::FLOAT_TYPE)? {
        Some(sub) => sub,
        None => return Ok(value),
    };
    let float_val = unsafe { pyre_object::w_float_get_value(value) };
    let obj = pyre_object::w_float_new(float_val);
    unsafe {
        (*obj).w_class = sub;
    }
    Ok(obj)
}

fn complex_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        args[0]
    };
    let value = crate::builtins::builtin_complex(&args[1..])?;
    if cls.is_null() || !unsafe { pyre_object::is_type(cls) } {
        return Ok(value);
    }
    let complex_typeobj = gettypefor(&pyre_object::COMPLEX_TYPE);
    if complex_typeobj.map_or(false, |t| std::ptr::eq(cls, t)) {
        return Ok(value);
    }
    // Subclass path — retag a fresh W_ComplexObject with the subclass.
    let (re, im) = unsafe {
        (
            pyre_object::w_complex_get_real(value),
            pyre_object::w_complex_get_imag(value),
        )
    };
    let obj = pyre_object::w_complex_new(re, im);
    unsafe {
        (*obj).w_class = cls;
    }
    Ok(obj)
}

/// Wrap a `__new__` builtin function in a staticmethod descriptor.
///
/// `__new__` must NOT bind a receiver — calling `cls.__new__(other_cls, ...)`
/// passes `other_cls` as the first argument, not `cls`. PyPy/CPython model
/// this by automatically wrapping `__new__` definitions in `staticmethod` at
/// type-creation time. pyre's TypeDef registry uses this helper at install
/// time so each builtin type's `__new__` slot already carries the correct
/// non-binding descriptor.
pub(crate) fn make_new_descr(
    func: fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
) -> PyObjectRef {
    // `BuiltinFunction`-typed so `type(int.__new__)` differs from a user
    // `def`'s `function`, letting `copyreg._reduce_ex`'s
    // `isinstance(new, type(int.__new__))` match only builtin `tp_new`
    // wrappers (mirrors `builtin_function_or_method`).  `__self__` is
    // stamped at type-finalisation via `stamp_new_descr_self`.
    let f = crate::gateway::make_builtin_function_as_builtin("__new__", func);
    pyre_object::w_staticmethod_new(f)
}

/// Wrap a `maketrans` builtin function in a staticmethod descriptor.
///
/// `str.maketrans` / `bytes.maketrans` / `bytearray.maketrans` are static
/// methods: an instance call such as `b''.maketrans(a, b)` must read `a`/`b`
/// as the two arguments, not bind the receiver as the first one.
fn make_maketrans_descr(
    func: fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
) -> PyObjectRef {
    pyre_object::w_staticmethod_new(make_builtin_function("maketrans", func))
}

/// `moduleobject.c module_new` — allocate an anonymous `Module`
/// (empty name, fresh dict).  The name is seeded by `__init__`, so
/// `__new__` ignores its arguments.  A subclass instance is retagged
/// with the actual class.
fn module_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_module = pyre_object::w_module_new("", std::ptr::null_mut());
    if let Some(cls) = args.first().copied() {
        if !cls.is_null() {
            unsafe { (*w_module).w_class = cls };
        }
    }
    Ok(w_module)
}

/// `moduleobject.c module_init` / `module.py:18-24 Module.__init__` —
/// `module.__init__(self, name, doc=None)`.  Seeds the `name` field plus
/// `__name__` / `__doc__` / `__package__` / `__loader__` / `__spec__`
/// in the module dict.
fn module_descr_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, _kwargs) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() < 2 {
        return Err(crate::PyError::type_error(
            "module.__init__() missing required argument 'name' (pos 1)".to_string(),
        ));
    }
    let self_ = positional[0];
    let w_name = positional[1];
    let w_doc = positional
        .get(2)
        .copied()
        .unwrap_or_else(pyre_object::w_none);
    let name = crate::baseobjspace::text_w(w_name)?;
    unsafe { pyre_object::w_module_set_name(self_, name) };
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(self_) };
    unsafe {
        pyre_object::w_dict_setitem_str(w_dict, "__name__", w_name);
        pyre_object::w_dict_setitem_str(w_dict, "__doc__", w_doc);
        pyre_object::w_dict_setitem_str(w_dict, "__package__", pyre_object::w_none());
        pyre_object::w_dict_setitem_str(w_dict, "__loader__", pyre_object::w_none());
        pyre_object::w_dict_setitem_str(w_dict, "__spec__", pyre_object::w_none());
    }
    Ok(pyre_object::w_none())
}

/// `module.py Module.typedef` — wire `__new__` / `__init__` so
/// `type(m)(name)` builds a real module.  `module` defines its own
/// `tp_new`, so `module.__new__ is not object.__new__`.
fn init_module_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(module_descr_new));
    dict_storage_store(
        ns,
        "__init__",
        make_builtin_function("__init__", module_descr_init),
    );
}

fn ellipsis_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if let Some(w_ellipsis) = gettypefor(&pyre_object::ELLIPSIS_TYPE) {
        check_user_subclass(w_ellipsis, cls)?;
    }
    Ok(pyre_object::special::w_ellipsis())
}

fn init_ellipsis_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(ellipsis_descr_new));
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function_with_arity("__repr__", |_args| Ok(w_str_new("Ellipsis")), 1),
    );
    dict_storage_store(
        ns,
        "__reduce__",
        make_builtin_function_with_arity("__reduce__", |_args| Ok(w_str_new("Ellipsis")), 1),
    );
}

/// special.py:20: NotImplemented.descr_new_notimplemented
fn notimplemented_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if let Some(w_notimplemented) = gettypefor(&pyre_object::pyobject::NOTIMPLEMENTED_TYPE) {
        check_user_subclass(w_notimplemented, cls)?;
    }
    Ok(pyre_object::special::w_not_implemented())
}

/// typedef.py:948-954 NotImplemented.typedef
fn init_notimplemented_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(notimplemented_descr_new));
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function_with_arity("__repr__", |_args| Ok(w_str_new("NotImplemented")), 1),
    );
    dict_storage_store(
        ns,
        "__reduce__",
        make_builtin_function_with_arity("__reduce__", |_args| Ok(w_str_new("NotImplemented")), 1),
    );
    // special.py:28-33 descr_bool
    dict_storage_store(
        ns,
        "__bool__",
        make_builtin_function_with_arity(
            "__bool__",
            |_args| {
                crate::warn::warn_deprecation(
                    "NotImplemented should not be used in a boolean context",
                );
                Ok(pyre_object::boolobject::w_bool_from(true))
            },
            1,
        ),
    );
}

/// `str.__new__(cls, *args)` — PyPy: unicodeobject.py descr__new__
///
/// `cls` is `str` itself: return the plain `W_UnicodeObject` from `builtin_str`.
/// `cls` is a `str` subclass: build the value, then allocate a fresh
/// `W_UnicodeObject` tagged with `__class__ = cls` so `type(obj) == cls` while
/// the underlying layout still satisfies `is_str()` for the JIT fast path.
fn str_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        args[0]
    };
    let value = crate::builtins::builtin_str(&args[1..])?;
    if cls.is_null() || !unsafe { pyre_object::is_type(cls) } {
        return Ok(value);
    }
    let str_typeobj = gettypefor(&pyre_object::STR_TYPE);
    if str_typeobj.map_or(false, |t| std::ptr::eq(cls, t)) {
        return Ok(value);
    }
    let s_owned = unsafe { pyre_object::w_str_get_value(value) }.to_string();
    let obj = pyre_object::w_str_new(&s_owned);
    // Tag with subclass type so type(obj) returns cls.
    unsafe {
        (*(obj as *mut pyre_object::PyObject)).w_class = cls;
    }
    Ok(obj)
}

/// dict.__new__(cls, *args) — if cls is a dict subclass, create an instance
/// with a backing dict for storage. PyPy: dictmultiobject.py descr__new__
fn dict_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        args[0]
    };
    let dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);

    // If cls IS dict (not a subclass), use normal dict constructor
    if cls.is_null() || std::ptr::eq(cls, dict_type) {
        return crate::builtins::builtin_dict_ctor(&args[1..]);
    }

    // cls is a dict subclass — create the instance with an empty backing
    // dict. `__new__` must NOT populate from the constructor arguments:
    // `object.__new__`/`dict.__new__` ignore them, and filling is the job of
    // `__init__` (the inherited `dict.__init__` for a plain subclass, or the
    // subclass override). Pre-filling here double-applies the argument for a
    // subclass whose `__init__` accumulates (e.g. `Counter`, whose `update`
    // adds rather than sets).
    let instance = pyre_object::w_instance_new(cls);
    let backing = pyre_object::w_dict_new();
    let _ = crate::baseobjspace::setattr_str(instance, "__dict_data__", backing);
    Ok(instance)
}
/// boolobject.py descr_new — bool.__new__(cls, obj=False)
///
/// check_user_subclass prevents subclassing (acceptable_as_base_class=False).
/// Only positional obj argument accepted.
fn bool_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // args[0] = w_booltype (cls)
    let w_booltype = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if let Some(w_bool) = gettypefor(&pyre_object::BOOL_TYPE) {
        check_user_subclass(w_bool, w_booltype)?;
    }
    // boolobject.py: descr_new(space, w_booltype, w_obj)
    // Takes exactly (cls) or (cls, obj). No extra args, no kwargs.
    if args.len() > 2 {
        return Err(crate::PyError::type_error(
            "bool expected at most 1 argument, got more",
        ));
    }
    // args[1] = w_obj (default: False)
    let w_obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    if w_obj.is_null() {
        return Ok(pyre_object::w_bool_from(false));
    }
    // Validate __bool__ return type and handle __bool__=None / __len__=None.
    // PyPy: space.is_true validates these conditions.
    // Use space.lookup (resolves type via type(obj)) — works for both
    // W_ObjectObject and int/float subclass instances.
    unsafe {
        if let Some(w_type) = crate::typedef::r#type(w_obj) {
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__bool__") {
                if pyre_object::is_none(method) {
                    return Err(crate::PyError::type_error(
                        "object of this type has no bool()",
                    ));
                }
                let result = crate::call_function(method, &[w_obj]);
                if !result.is_null() {
                    if !pyre_object::is_bool(result) {
                        let tp_name = (*(*result).ob_type).name;
                        return Err(crate::PyError::type_error(format!(
                            "__bool__ should return bool, returned {}",
                            tp_name,
                        )));
                    }
                    return Ok(result);
                }
            }
            if let Some(len_m) = crate::baseobjspace::lookup_in_type(w_type, "__len__") {
                if pyre_object::is_none(len_m) {
                    return Err(crate::PyError::type_error(
                        "object of this type has no len()",
                    ));
                }
                // __len__ returning negative → ValueError
                let len_result = crate::call_function(len_m, &[w_obj]);
                if !len_result.is_null() && pyre_object::is_int(len_result) {
                    let v = pyre_object::w_int_get_value(len_result);
                    if v < 0 {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::ValueError,
                            "__len__() should return >= 0".to_string(),
                        ));
                    }
                    return Ok(pyre_object::w_bool_from(v != 0));
                }
            }
        }
    }
    Ok(pyre_object::w_bool_from(crate::baseobjspace::is_true(
        w_obj,
    )?))
}
/// When `cls` is a user subclass of the builtin `base` (not `base`
/// itself, not null/non-type), return it so `__new__` can tag the fresh
/// builtin instance's `w_class`; otherwise `None`.  Mirrors the
/// subclass-tagging path `str`/`int`/`float` `__new__` already use so
/// `type(obj)` / `isinstance` / overridden-dunder dispatch see the
/// subclass while the object keeps its builtin layout.
fn subclass_to_tag(
    cls: PyObjectRef,
    base: &'static pyre_object::PyType,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    if cls.is_null() {
        return Ok(None);
    }
    let base_obj = match gettypefor(base) {
        Some(t) => t,
        None => return Ok(Some(cls)),
    };
    // `cls` is the builtin base itself → keep the canonical layout, no retag.
    if std::ptr::eq(cls, base_obj) {
        return Ok(None);
    }
    // tp_new_wrapper rejects a non-type, or a type that is not a subtype of the
    // builtin: `range.__new__(int, 1)` must raise, not stamp a W_Range as int
    // (which later dispatch would read through an incompatible layout).
    if !unsafe { pyre_object::is_type(cls) } {
        let base_name = unsafe { pyre_object::w_type_get_name(base_obj) };
        return Err(crate::PyError::type_error(format!(
            "{base_name}.__new__(X): X is not a type object"
        )));
    }
    if !unsafe { crate::baseobjspace::issubtype_w(cls, base_obj) } {
        let base_name = unsafe { pyre_object::w_type_get_name(base_obj) };
        let cls_name = unsafe { pyre_object::w_type_get_name(cls) };
        return Err(crate::PyError::type_error(format!(
            "{base_name}.__new__({cls_name}): {cls_name} is not a subtype of {base_name}"
        )));
    }
    Ok(Some(cls))
}

/// `list.__new__(cls, *args)` — `listobject.py:descr__new__` allocates a
/// `W_ListObject` of `w_listtype`.  `builtin_list_ctor` always returns a
/// fresh list, so a subclass instance is the same object with `w_class`
/// retagged.
fn list_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (params, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let cls = params.first().copied().unwrap_or(pyre_object::PY_NULL);
    builtinclass_new_args_check(
        "list",
        gettypeobject(&pyre_object::LIST_TYPE),
        cls,
        params.len().saturating_sub(2),
        crate::builtins::has_real_kwargs(kwargs),
    )?;
    let value = crate::builtins::builtin_list_ctor(params.get(1..).unwrap_or(&[]))?;
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::LIST_TYPE)? {
        unsafe {
            (*value).w_class = sub;
        }
    }
    Ok(value)
}

/// `tuple.__new__(cls, *args)` — `tupleobject.py:descr__new__` allocates
/// a `W_TupleObject` of `w_tupletype`.  `builtin_tuple` may return the
/// argument tuple unchanged, so the subclass path rebuilds a fresh tuple
/// before retagging to avoid aliasing the input.
fn tuple_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (params, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let cls = params.first().copied().unwrap_or(pyre_object::PY_NULL);
    builtinclass_new_args_check(
        "tuple",
        gettypeobject(&pyre_object::TUPLE_TYPE),
        cls,
        params.len().saturating_sub(2),
        crate::builtins::has_real_kwargs(kwargs),
    )?;
    let value = crate::builtins::builtin_tuple(params.get(1..).unwrap_or(&[]))?;
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::TUPLE_TYPE)? {
        let n = unsafe { pyre_object::w_tuple_len(value) };
        let items: Vec<PyObjectRef> = (0..n)
            .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(value, i as i64) })
            .collect();
        // Canonical array-backed layout (ob_type == TUPLE_TYPE) so the
        // subclass tag never lands on an arity-2 specialised tuple.
        let fresh = pyre_object::w_tuple_new_array_backed(items);
        unsafe {
            (*fresh).w_class = sub;
        }
        return Ok(fresh);
    }
    Ok(value)
}
/// `enumerate.__new__(cls, iterable, start=0)` — `functional.py:253-275
/// W_Enumerate.descr___new__`.  `builtin_enumerate` builds a fresh
/// `W_Enumerate`; a subclass instance is the same object with `w_class`
/// retagged (the instance keeps the `enumerate` GC tag so iteration still
/// dispatches through the builtin `__next__`).
fn enumerate_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let value = crate::builtins::builtin_enumerate(args.get(1..).unwrap_or(&[]))?;
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::functional::ENUMERATE_TYPE)? {
        unsafe {
            (*value).w_class = sub;
        }
    }
    Ok(value)
}

/// `map.__new__(cls, func, *iterables, strict=False)` — `functional.py:888-902
/// W_Map.descr___new__`.
fn map_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let value = crate::builtins::builtin_map(args.get(1..).unwrap_or(&[]))?;
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::functional::MAP_TYPE)? {
        unsafe {
            (*value).w_class = sub;
        }
    }
    Ok(value)
}

/// `filter.__new__(cls, predicate, iterable)` — `functional.py:917-925
/// W_Filter.descr___new__`.
fn filter_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let value = crate::builtins::builtin_filter(args.get(1..).unwrap_or(&[]))?;
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::functional::FILTER_TYPE)? {
        unsafe {
            (*value).w_class = sub;
        }
    }
    Ok(value)
}

/// `zip.__new__(cls, *iterables, strict=False)` — `functional.py:1101-1105
/// W_Zip.descr___new__`.
fn zip_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let value = crate::builtins::builtin_zip(args.get(1..).unwrap_or(&[]))?;
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::functional::ZIP_TYPE)? {
        unsafe {
            (*value).w_class = sub;
        }
    }
    Ok(value)
}

/// `reversed.__new__(cls, sequence)` — `functional.py:330-359
/// W_ReversedIterator`.  `builtin_reversed` returns a `W_ReversedIterator`
/// only for the exact builtin-sequence fast path; for a range or a
/// `__reversed__`-defining object it returns a foreign iterator, which must
/// NOT be retagged to the subclass.  Retag only the canonical reversed
/// object.
fn reversed_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let value = crate::builtins::builtin_reversed(args.get(1..).unwrap_or(&[]))?;
    if unsafe { pyre_object::functional::is_reversed(value) } {
        if let Some(sub) = subclass_to_tag(cls, &pyre_object::functional::REVERSED_TYPE)? {
            unsafe {
                (*value).w_class = sub;
            }
        }
    }
    Ok(value)
}

/// `range.__new__(cls, stop)` / `range.__new__(cls, start, stop[, step])` —
/// `rangeobject.py descr_new`.  `builtin_range` builds a fresh `W_Range`; a
/// subclass instance is the same object with `w_class` retagged.
fn range_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let value = crate::builtins::builtin_range(args.get(1..).unwrap_or(&[]))?;
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::functional::RANGE_TYPE)? {
        unsafe {
            (*value).w_class = sub;
        }
    }
    Ok(value)
}

// dict_new handled by dict_descr_new above (supports dict subclasses)

/// typeobject.py:511-524 W_TypeObject.check_user_subclass.
///
/// ```text
/// def check_user_subclass(self, w_subtype):
///     if not isinstance(w_subtype, W_TypeObject):
///         raise TypeError("X is not a type object ('%T')", w_subtype)
///     if not w_subtype.issubtype(self):
///         raise TypeError("%N.__new__(%N): %N is not a subtype of %N", ...)
///     if self.layout.typedef is not w_subtype.layout.typedef:
///         raise TypeError("%N.__new__(%N) is not safe, use %N.__new__()", ...)
///     return w_subtype
/// ```
fn check_user_subclass(w_self: PyObjectRef, w_subtype: PyObjectRef) -> Result<(), crate::PyError> {
    if w_subtype.is_null() || !unsafe { pyre_object::is_type(w_subtype) } {
        let self_name = unsafe { pyre_object::w_type_get_name(w_self) };
        return Err(crate::PyError::type_error(format!(
            "{}.__new__(X): X is not a type object",
            self_name,
        )));
    }
    if std::ptr::eq(w_subtype, w_self) {
        return Ok(());
    }
    let mro_ptr = unsafe { pyre_object::w_type_get_mro(w_subtype) };
    let is_sub =
        !mro_ptr.is_null() && unsafe { (*mro_ptr).iter().any(|&t| std::ptr::eq(t, w_self)) };
    if !is_sub {
        let self_name = unsafe { pyre_object::w_type_get_name(w_self) };
        let sub_name = unsafe { pyre_object::w_type_get_name(w_subtype) };
        return Err(crate::PyError::type_error(format!(
            "{}.__new__({}): {} is not a subtype of {}",
            self_name, sub_name, sub_name, self_name,
        )));
    }
    // typeobject.py:520-523 — layout safety. The base allocator only knows
    // how to fill the parent layout; if the subtype introduces extra slots
    // (different layout typedef), allocating through it would corrupt the
    // foreign layout.
    let self_layout = unsafe { pyre_object::w_type_get_layout_ptr(w_self) };
    let sub_layout = unsafe { pyre_object::w_type_get_layout_ptr(w_subtype) };
    let self_typedef = if self_layout.is_null() {
        std::ptr::null()
    } else {
        unsafe { (*self_layout).typedef }
    };
    let sub_typedef = if sub_layout.is_null() {
        std::ptr::null()
    } else {
        unsafe { (*sub_layout).typedef }
    };
    if !std::ptr::eq(self_typedef, sub_typedef) {
        let self_name = unsafe { pyre_object::w_type_get_name(w_self) };
        let sub_name = unsafe { pyre_object::w_type_get_name(w_subtype) };
        return Err(crate::PyError::type_error(format!(
            "{}.__new__({}) is not safe, use {}.__new__()",
            self_name, sub_name, sub_name,
        )));
    }
    Ok(())
}

fn set_alloc_for_class(
    cls: PyObjectRef,
    exact_type: PyObjectRef,
    frozen: bool,
) -> Result<PyObjectRef, crate::PyError> {
    // typeobject.py:511 allocate_instance → check_user_subclass.
    check_user_subclass(exact_type, cls)?;
    let obj = if frozen {
        pyre_object::w_frozenset_new()
    } else {
        pyre_object::w_set_new()
    };
    if !std::ptr::eq(cls, exact_type) {
        unsafe {
            (*obj).w_class = cls;
        }
    }
    Ok(obj)
}

/// `set.__new__(cls, ...)` — PyPy: setobject.py W_SetObject.descr_new.
///
/// PyPy declares the inner function as `descr_new(space, w_settype,
/// __args__)`. `__args__` is the gateway sentinel for variadic positional
/// arguments, so gateway.py:723-727 sets `maxargs = sys.maxint`; the body
/// ignores everything past `w_settype`. The actual argument count check
/// lives on `descr_init`, which type.__call__ runs after `__new__`.
fn set_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (params, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let cls = params.first().copied().unwrap_or(pyre_object::PY_NULL);
    let set_type = crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE);
    // `set.__new__` ignores its extra arguments and leaves positional-count
    // validation to `__init__`; only a plain `set(...)` whose `__init__` is not
    // overridden rejects keywords here (`_PyArg_NoKeywords`).  A subclass that
    // defines an `__init__` taking keywords must still receive them, so the
    // keyword check is skipped whenever the subtype overrides `__init__`
    // (passing 0 for `positional_extra` keeps the surplus-positional report in
    // `__init__`).
    builtinclass_new_args_check(
        "set",
        set_type,
        cls,
        0,
        crate::builtins::has_real_kwargs(kwargs),
    )?;
    set_alloc_for_class(cls, set_type, false)
}

/// `objspace/std/util.py:107` `builtinclass_new_args_check` — shared surplus
/// argument validation for the `__new__` of one-argument builtin classes
/// (`float`, `tuple`, `list`, `frozenset`, `itertools.cycle`).
///
/// The check is skipped when `w_subtyp` overrides `__init__`, because that
/// `__init__` consumes the surplus arguments (`space.getattr(base, '__init__')
/// is space.getattr(sub, '__init__')` — modelled here as MRO-lookup identity).
/// When it applies, a surplus positional wins over a keyword.
///
/// `positional_extra` is `len(__args__.arguments_w)` (positionals beyond the
/// single accepted argument); `has_keywords` is `__args__.keyword_names_w`.
fn builtinclass_new_args_check(
    name: &str,
    w_basetyp: PyObjectRef,
    w_subtyp: PyObjectRef,
    positional_extra: usize,
    has_keywords: bool,
) -> Result<(), crate::PyError> {
    let init_matches = w_subtyp.is_null()
        || std::ptr::eq(w_basetyp, w_subtyp)
        || unsafe {
            match (
                crate::baseobjspace::lookup_in_type(w_basetyp, "__init__"),
                crate::baseobjspace::lookup_in_type(w_subtyp, "__init__"),
            ) {
                (Some(b), Some(s)) => std::ptr::eq(b, s),
                (None, None) => true,
                _ => false,
            }
        };
    if init_matches {
        if positional_extra > 0 {
            return Err(crate::PyError::type_error(format!(
                "{name} expected at most 1 argument, got {}",
                positional_extra + 1,
            )));
        }
        if has_keywords {
            return Err(crate::PyError::type_error(format!(
                "{name}() takes no keyword arguments"
            )));
        }
    }
    Ok(())
}

/// `frozenset.__new__(cls, [iterable])` — PyPy: setobject.py W_FrozensetObject.descr_new2.
fn frozenset_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let frozenset_type = crate::typedef::gettypeobject(&pyre_object::setobject::FROZENSET_TYPE);
    builtinclass_new_args_check(
        "frozenset",
        frozenset_type,
        cls,
        args.len().saturating_sub(2),
        crate::builtins::has_real_kwargs(kwargs),
    )?;
    let iterable = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);

    // setobject.py:616-618 — reuse the argument only when the target type is
    // exactly `frozenset` and the argument's implementation class is exactly
    // `W_FrozensetObject` (`type(w_iterable) is W_FrozensetObject`); a subclass
    // instance retags `w_class` and is rebuilt.
    if !iterable.is_null()
        && std::ptr::eq(cls, frozenset_type)
        && unsafe {
            pyre_object::pyobject::is_exact_type(iterable, &pyre_object::setobject::FROZENSET_TYPE)
        }
    {
        return Ok(iterable);
    }

    let obj = set_alloc_for_class(cls, frozenset_type, true)?;
    if !iterable.is_null() {
        let items = crate::builtins::collect_iterable(iterable)?;
        for item in items {
            unsafe { pyre_object::w_set_add(obj, item) };
        }
    }
    Ok(obj)
}

/// `set.__init__(self, [iterable])` — PyPy: setobject.py W_SetObject.descr_init.
///
/// PyPy parses `__args__` against `init_signature = Signature(['some_iterable'])`
/// so anything beyond `(self, iterable)` raises TypeError; pyre enforces the
/// same maxargs explicitly here.
fn set_descr_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let set_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    // gateway.interp2app(W_SetObject.descr_init) enforces that `self` is a
    // W_SetObject before the body runs; without this check pyre would cast
    // arbitrary args[0] values straight to the set layout below.
    if set_obj.is_null() || !unsafe { pyre_object::is_set(set_obj) } {
        let tp_name = if set_obj.is_null() {
            "NoneType".to_string()
        } else {
            unsafe { (*(*set_obj).ob_type).name.to_string() }
        };
        return Err(crate::PyError::type_error(format!(
            "descriptor '__init__' requires a 'set' object but received a '{}'",
            tp_name,
        )));
    }
    // setobject.py:161 `descr_init(self, space, w_iterable=None, __posonly__=None)`
    // — `iterable` is a single positional-only optional argument.  Parse the
    // gateway args against that signature (gateway interp2app `parse_into_scope`)
    // so a keyword raises the matching TypeError instead of leaking the kwargs
    // dict as the iterable: `set(iterable=[1])` → "set.__init__() got a
    // positional-only argument passed as keyword argument: 'iterable'",
    // `set(x=1)` → "set.__init__() got an unexpected keyword argument 'x'".
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    let mut keyword_names_w: Vec<PyObjectRef> = Vec::new();
    let mut keywords_w: Vec<PyObjectRef> = Vec::new();
    if let Some(dict) = kwargs {
        for (key, val) in unsafe { pyre_object::w_dict_str_entries_wtf8(dict) } {
            if key.as_str() == Ok("__pyre_kw__") {
                continue;
            }
            keyword_names_w.push(pyre_object::w_str_from_wtf8(key));
            keywords_w.push(val);
        }
    }
    let signature = crate::gateway::Signature::new(vec!["self", "iterable"], None, None, 0, 2);
    let arguments = crate::argument::Arguments::with_kw(positional, &keyword_names_w, &keywords_w);
    let defaults = [pyre_object::w_none()];
    let mut scope_w = vec![pyre_object::PY_NULL; signature.scope_length()];
    arguments.parse_into_scope(
        set_obj,
        &mut scope_w,
        "set.__init__",
        &signature,
        Some(&defaults),
        pyre_object::PY_NULL,
    )?;
    let w_iterable = scope_w[1];

    let existing = unsafe { pyre_object::w_set_items(set_obj) };
    for item in existing {
        unsafe {
            pyre_object::w_set_discard(set_obj, item);
        }
    }
    // setobject.py:1722 `_initialize_set` populates from the iterable when it
    // is not None (the parsed default).
    if !w_iterable.is_null() && !unsafe { pyre_object::is_none(w_iterable) } {
        let items = crate::builtins::collect_iterable(w_iterable)?;
        for item in items {
            unsafe { pyre_object::w_set_add(set_obj, item) };
        }
    }
    Ok(pyre_object::w_none())
}

// ── List TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/listobject.py TypeDef("list", ...)

/// Name of `obj`'s type, for operand-type error messages.
fn arg_type_name(obj: PyObjectRef) -> String {
    unsafe {
        match r#type(obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => (*(*obj).ob_type).name.to_string(),
        }
    }
}

fn init_list_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(list_descr_new));
    // listobject.py:2486 __class_getitem__ = interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    dict_storage_store(
        ns,
        "__class_getitem__",
        pyre_object::function::w_classmethod_new(make_builtin_function(
            "__class_getitem__",
            crate::_pypy_generic_alias::generic_alias_class_getitem,
        )),
    );
    dict_storage_store(
        ns,
        "append",
        make_builtin_function_with_arity("append", crate::type_methods::list_method_append, 2),
    );
    dict_storage_store(
        ns,
        "extend",
        make_builtin_function_with_arity("extend", crate::type_methods::list_method_extend, 2),
    );
    dict_storage_store(
        ns,
        "copy",
        make_builtin_function_with_arity("copy", crate::type_methods::list_method_copy, 1),
    );
    dict_storage_store(
        ns,
        "insert",
        make_builtin_function_with_arity("insert", crate::type_methods::list_method_insert, 3),
    );
    dict_storage_store(
        ns,
        "pop",
        make_builtin_function("pop", crate::type_methods::list_method_pop),
    );
    dict_storage_store(
        ns,
        "clear",
        make_builtin_function_with_arity("clear", crate::type_methods::list_method_clear, 1),
    );
    dict_storage_store(
        ns,
        "reverse",
        make_builtin_function_with_arity("reverse", crate::type_methods::list_method_reverse, 1),
    );
    dict_storage_store(
        ns,
        "sort",
        make_builtin_function("sort", crate::type_methods::list_method_sort),
    );
    dict_storage_store(
        ns,
        "index",
        make_builtin_function("index", crate::type_methods::list_method_index),
    );
    dict_storage_store(
        ns,
        "count",
        make_builtin_function_with_arity("count", crate::type_methods::list_method_count, 2),
    );
    dict_storage_store(
        ns,
        "remove",
        make_builtin_function_with_arity("remove", crate::type_methods::list_method_remove, 2),
    );
    // Container slots exposed as callable dunders.  `__getitem__` binds the
    // direct slot body (`getitem_slot`) rather than the operator entry, so a
    // subclass override's `super().__getitem__` reaches the inherited builtin
    // subscript instead of re-entering override dispatch.
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                crate::type_methods::arity_exact(args, "list.__getitem__", 1)?;
                crate::baseobjspace::getitem_slot(args[0], args[1])
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__setitem__",
        make_builtin_function_with_arity(
            "__setitem__",
            |args| {
                crate::type_methods::arity_exact_unpack(args, "__setitem__", 2)?;
                crate::baseobjspace::setitem_slot(args[0], args[1], args[2])?;
                Ok(pyre_object::w_none())
            },
            3,
        ),
    );
    dict_storage_store(
        ns,
        "__delitem__",
        make_builtin_function_with_arity(
            "__delitem__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                crate::baseobjspace::delitem_slot(args[0], args[1])?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function_with_arity(
            "__len__",
            |args| {
                crate::type_methods::arity_slot(args, 0)?;
                crate::baseobjspace::len_slot(args[0])
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function_with_arity(
            "__contains__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                let found = crate::baseobjspace::contains_slot(args[0], args[1])?;
                Ok(pyre_object::w_bool_from(found))
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__iter__",
        // Build the storage iterator directly rather than re-entering
        // `space.iter()` — a `list` subclass that calls `super().__iter__()`
        // would otherwise be re-dispatched back to its own override.
        make_builtin_function_with_arity(
            "__iter__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                if obj.is_null() {
                    return Ok(pyre_object::w_none());
                }
                crate::type_methods::arity_slot(args, 0)?;
                Ok(pyre_object::w_seq_iter_new(obj, unsafe {
                    pyre_object::w_list_len(obj)
                }))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__reversed__",
        make_builtin_function_with_arity(
            "__reversed__",
            |args| {
                // `listobject.py:737 descr_reversed` — a lazy reverse iterator
                // over the list, the same `W_ReversedIterator` representation as
                // `reversed(list)` (walks `getitem(seq, remaining)` downward).
                crate::type_methods::arity_no_args(args, "list.__reversed__")?;
                let obj = args[0];
                let n = unsafe { pyre_object::w_list_len(obj) } as i64;
                Ok(pyre_object::functional::w_reversed_new(obj, n - 1))
            },
            1,
        ),
    );
    // Arithmetic slots.  `listobject.c:list_concat` rejects a non-list
    // operand with TypeError (it does not return NotImplemented);
    // `list_repeat` requires an integer count.
    dict_storage_store(
        ns,
        "__add__",
        make_builtin_function_with_arity(
            "__add__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                if unsafe { pyre_object::is_list(args[1]) } {
                    unsafe { crate::objspace::descroperation::list_concat(args[0], args[1]) }
                } else {
                    Err(crate::PyError::type_error(format!(
                        "can only concatenate list (not \"{}\") to list",
                        arg_type_name(args[1])
                    )))
                }
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__mul__",
        make_builtin_function_with_arity("__mul__", list_descr_mul, 2),
    );
    dict_storage_store(
        ns,
        "__rmul__",
        make_builtin_function_with_arity("__rmul__", list_descr_mul, 2),
    );
    dict_storage_store(
        ns,
        "__iadd__",
        make_builtin_function_with_arity(
            "__iadd__",
            |args| {
                // Validate the slot arity before delegating so a bad call
                // reports "expected 1 argument, got M" rather than the
                // `extend` message surfaced by the shared implementation.
                crate::type_methods::arity_slot(args, 1)?;
                crate::type_methods::list_method_extend(args)?;
                Ok(args[0])
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__imul__",
        make_builtin_function_with_arity(
            "__imul__",
            |args| {
                // listobject.py descr_inplace_mul: the count goes through
                // `__index__`; a non-index operand becomes NotImplemented.
                crate::type_methods::arity_slot(args, 1)?;
                let Some(w_count) = list_repeat_index(args[1])? else {
                    return Ok(pyre_object::w_not_implemented());
                };
                unsafe { crate::objspace::descroperation::list_inplace_repeat(args[0], w_count)? };
                Ok(args[0])
            },
            2,
        ),
    );
    for (name, func) in [
        ("__eq__", list_dunder_eq as DunderFn),
        ("__ne__", list_dunder_ne),
        ("__lt__", list_dunder_lt),
        ("__le__", list_dunder_le),
        ("__gt__", list_dunder_gt),
        ("__ge__", list_dunder_ge),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
}

/// Coerce a `list * n` / `list *= n` repeat count through `__index__`
/// (`space.getindex_w`).  An operand without `__index__` yields `None`, which
/// the caller maps to NotImplemented so the `*`/`*=` operator can try a
/// reflected `__rmul__` and otherwise emit the "can't multiply sequence by
/// non-int" message; any other coercion error propagates.
fn list_repeat_index(w_obj: PyObjectRef) -> Result<Option<PyObjectRef>, crate::PyError> {
    match crate::baseobjspace::space_index(w_obj) {
        Ok(w_count) => Ok(Some(w_count)),
        Err(e) if e.kind == crate::PyErrorKind::TypeError => Ok(None),
        Err(e) => Err(e),
    }
}

/// `listobject.c:list_repeat` — `list * n` / `n * list`.  The count goes
/// through `__index__`, so any object implementing it repeats the list.
fn list_descr_mul(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_slot(args, 1)?;
    let Some(w_count) = list_repeat_index(args[1])? else {
        return Ok(pyre_object::w_not_implemented());
    };
    unsafe { crate::objspace::descroperation::list_repeat(args[0], w_count) }
}

// ── Str TypeDef ──────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/unicodeobject.py TypeDef("str", ...)

fn init_str_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(str_descr_new));
    dict_storage_store(
        ns,
        "__format__",
        make_builtin_function_with_arity(
            "__format__",
            crate::type_methods::builtin_value_format,
            2,
        ),
    );
    dict_storage_store(
        ns,
        "join",
        make_builtin_function_with_arity("join", crate::type_methods::str_method_join, 2),
    );
    dict_storage_store(
        ns,
        "split",
        make_builtin_function("split", crate::type_methods::str_method_split),
    );
    dict_storage_store(
        ns,
        "rsplit",
        make_builtin_function("rsplit", crate::type_methods::str_method_rsplit),
    );
    dict_storage_store(
        ns,
        "splitlines",
        make_builtin_function("splitlines", crate::type_methods::str_method_splitlines),
    );
    dict_storage_store(
        ns,
        "partition",
        make_builtin_function("partition", crate::type_methods::str_method_partition),
    );
    dict_storage_store(
        ns,
        "rpartition",
        make_builtin_function("rpartition", crate::type_methods::str_method_rpartition),
    );
    dict_storage_store(
        ns,
        "zfill",
        make_builtin_function("zfill", crate::type_methods::str_method_zfill),
    );
    dict_storage_store(
        ns,
        "casefold",
        make_builtin_function("casefold", crate::type_methods::str_method_casefold),
    );
    dict_storage_store(
        ns,
        "swapcase",
        make_builtin_function("swapcase", crate::type_methods::str_method_swapcase),
    );
    dict_storage_store(
        ns,
        "expandtabs",
        make_builtin_function("expandtabs", crate::type_methods::str_method_expandtabs),
    );
    dict_storage_store(
        ns,
        "format_map",
        make_builtin_function("format_map", crate::type_methods::str_method_format_map),
    );
    dict_storage_store(
        ns,
        "strip",
        make_builtin_function("strip", crate::type_methods::str_method_strip),
    );
    dict_storage_store(
        ns,
        "lstrip",
        make_builtin_function("lstrip", crate::type_methods::str_method_lstrip),
    );
    dict_storage_store(
        ns,
        "rstrip",
        make_builtin_function("rstrip", crate::type_methods::str_method_rstrip),
    );
    dict_storage_store(
        ns,
        "startswith",
        make_builtin_function("startswith", crate::type_methods::str_method_startswith),
    );
    dict_storage_store(
        ns,
        "endswith",
        make_builtin_function("endswith", crate::type_methods::str_method_endswith),
    );
    dict_storage_store(
        ns,
        "replace",
        make_builtin_function("replace", crate::type_methods::str_method_replace),
    );
    dict_storage_store(
        ns,
        "find",
        make_builtin_function("find", crate::type_methods::str_method_find),
    );
    dict_storage_store(
        ns,
        "rfind",
        make_builtin_function("rfind", crate::type_methods::str_method_rfind),
    );
    dict_storage_store(
        ns,
        "rindex",
        make_builtin_function("rindex", crate::type_methods::str_method_rindex),
    );
    dict_storage_store(
        ns,
        "upper",
        make_builtin_function_with_arity("upper", crate::type_methods::str_method_upper, 1),
    );
    dict_storage_store(
        ns,
        "lower",
        make_builtin_function_with_arity("lower", crate::type_methods::str_method_lower, 1),
    );
    dict_storage_store(
        ns,
        "format",
        make_builtin_function("format", crate::type_methods::str_method_format),
    );
    dict_storage_store(
        ns,
        "encode",
        make_builtin_function("encode", crate::type_methods::str_method_encode),
    );
    dict_storage_store(
        ns,
        "isdigit",
        make_builtin_function_with_arity("isdigit", crate::type_methods::str_method_isdigit, 1),
    );
    dict_storage_store(
        ns,
        "isdecimal",
        make_builtin_function_with_arity("isdecimal", crate::type_methods::str_method_isdecimal, 1),
    );
    dict_storage_store(
        ns,
        "isnumeric",
        make_builtin_function_with_arity("isnumeric", crate::type_methods::str_method_isnumeric, 1),
    );
    dict_storage_store(
        ns,
        "istitle",
        make_builtin_function_with_arity("istitle", crate::type_methods::str_method_istitle, 1),
    );
    dict_storage_store(
        ns,
        "isalpha",
        make_builtin_function_with_arity("isalpha", crate::type_methods::str_method_isalpha, 1),
    );
    dict_storage_store(
        ns,
        "isidentifier",
        make_builtin_function_with_arity(
            "isidentifier",
            crate::type_methods::str_method_isidentifier,
            1,
        ),
    );
    dict_storage_store(
        ns,
        "zfill",
        make_builtin_function_with_arity("zfill", crate::type_methods::str_method_zfill, 2),
    );
    dict_storage_store(
        ns,
        "count",
        make_builtin_function("count", crate::type_methods::str_method_count),
    );
    dict_storage_store(
        ns,
        "index",
        make_builtin_function("index", crate::type_methods::str_method_index),
    );
    dict_storage_store(
        ns,
        "title",
        make_builtin_function_with_arity("title", crate::type_methods::str_method_title, 1),
    );
    dict_storage_store(
        ns,
        "capitalize",
        make_builtin_function_with_arity(
            "capitalize",
            crate::type_methods::str_method_capitalize,
            1,
        ),
    );
    dict_storage_store(
        ns,
        "swapcase",
        make_builtin_function_with_arity("swapcase", crate::type_methods::str_method_swapcase, 1),
    );
    dict_storage_store(
        ns,
        "center",
        make_builtin_function("center", crate::type_methods::str_method_center),
    );
    dict_storage_store(
        ns,
        "ljust",
        make_builtin_function("ljust", crate::type_methods::str_method_ljust),
    );
    dict_storage_store(
        ns,
        "rjust",
        make_builtin_function("rjust", crate::type_methods::str_method_rjust),
    );
    dict_storage_store(
        ns,
        "isspace",
        make_builtin_function_with_arity("isspace", crate::type_methods::str_method_isspace, 1),
    );
    dict_storage_store(
        ns,
        "isprintable",
        make_builtin_function_with_arity(
            "isprintable",
            crate::type_methods::str_method_isprintable,
            1,
        ),
    );
    dict_storage_store(
        ns,
        "isupper",
        make_builtin_function_with_arity("isupper", crate::type_methods::str_method_isupper, 1),
    );
    dict_storage_store(
        ns,
        "islower",
        make_builtin_function_with_arity("islower", crate::type_methods::str_method_islower, 1),
    );
    dict_storage_store(
        ns,
        "isalnum",
        make_builtin_function_with_arity("isalnum", crate::type_methods::str_method_isalnum, 1),
    );
    dict_storage_store(
        ns,
        "isascii",
        make_builtin_function_with_arity("isascii", crate::type_methods::str_method_isascii, 1),
    );
    dict_storage_store(
        ns,
        "partition",
        make_builtin_function_with_arity("partition", crate::type_methods::str_method_partition, 2),
    );
    dict_storage_store(
        ns,
        "rpartition",
        make_builtin_function_with_arity(
            "rpartition",
            crate::type_methods::str_method_rpartition,
            2,
        ),
    );
    dict_storage_store(
        ns,
        "splitlines",
        make_builtin_function("splitlines", crate::type_methods::str_method_splitlines),
    );
    dict_storage_store(
        ns,
        "removeprefix",
        make_builtin_function_with_arity(
            "removeprefix",
            crate::type_methods::str_method_removeprefix,
            2,
        ),
    );
    dict_storage_store(
        ns,
        "removesuffix",
        make_builtin_function_with_arity(
            "removesuffix",
            crate::type_methods::str_method_removesuffix,
            2,
        ),
    );
    dict_storage_store(
        ns,
        "expandtabs",
        make_builtin_function("expandtabs", crate::type_methods::str_method_expandtabs),
    );
    dict_storage_store(
        ns,
        "translate",
        make_builtin_function_with_arity("translate", crate::type_methods::str_method_translate, 2),
    );
    // str dunder methods
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function_with_arity(
            "__contains__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                Ok(pyre_object::w_bool_from(
                    crate::baseobjspace::contains_slot(args[0], args[1])?,
                ))
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function_with_arity(
            "__len__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                crate::type_methods::arity_slot(args, 0)?;
                crate::baseobjspace::len_slot(args[0])
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                crate::baseobjspace::getitem_slot(args[0], args[1])
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function_with_arity(
            "__iter__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                crate::baseobjspace::iter(args[0])
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__add__",
        make_builtin_function_with_arity(
            "__add__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                // Self-contained concat: returning NotImplemented for a
                // non-str operand lets the `+` operator emit the
                // "can only concatenate" message, and avoids the
                // recursion that delegating back to `add` would cause
                // (descroperation::add re-dispatches to this dunder).
                if unsafe { pyre_object::is_str(args[1]) } {
                    unsafe { crate::objspace::descroperation::str_concat(args[0], args[1]) }
                } else {
                    Ok(pyre_object::w_not_implemented())
                }
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__mul__",
        make_builtin_function_with_arity(
            "__mul__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                if unsafe { pyre_object::pyobject::is_int_or_long(args[1]) } {
                    unsafe { crate::objspace::descroperation::str_repeat(args[0], args[1]) }
                } else {
                    let count = crate::builtins::space_index_w(args[1])?;
                    unsafe {
                        crate::objspace::descroperation::str_repeat(args[0], w_int_new(count))
                    }
                }
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__mod__",
        make_builtin_function_with_arity(
            "__mod__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                crate::baseobjspace::mod_(args[0], args[1])
            },
            2,
        ),
    );
    // maketrans — PyPy: unicodeobject.py descr_maketrans
    dict_storage_store(
        ns,
        "maketrans",
        make_maketrans_descr(|args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error(
                    "maketrans expected at least 1 argument, got 0",
                ));
            }
            if args.len() > 3 {
                return Err(crate::PyError::type_error(format!(
                    "maketrans expected at most 3 arguments, got {}",
                    args.len()
                )));
            }

            let d = pyre_object::w_dict_new();
            if args.len() >= 2 {
                if !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "first maketrans argument must be a string if there is a second argument",
                    ));
                }
                if !unsafe { pyre_object::is_str(args[1]) } {
                    return Err(crate::PyError::type_error(format!(
                        "maketrans() argument 2 must be str, not {}",
                        crate::type_methods::arg_type_name(args[1])
                    )));
                }
                if args.len() == 3 && !unsafe { pyre_object::is_str(args[2]) } {
                    return Err(crate::PyError::type_error(format!(
                        "maketrans() argument 3 must be str, not {}",
                        crate::type_methods::arg_type_name(args[2])
                    )));
                }

                let x = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
                let y = unsafe { pyre_object::w_str_get_wtf8(args[1]) };
                if x.code_points().count() != y.code_points().count() {
                    return Err(crate::PyError::value_error(
                        "the first two maketrans arguments must have equal length",
                    ));
                }
                for (xc, yc) in x.code_points().zip(y.code_points()) {
                    unsafe {
                        pyre_object::w_dict_store(
                            d,
                            pyre_object::w_int_new(xc.to_u32() as i64),
                            pyre_object::w_int_new(yc.to_u32() as i64),
                        );
                    }
                }
                if args.len() == 3 {
                    let z = unsafe { pyre_object::w_str_get_wtf8(args[2]) };
                    for zc in z.code_points() {
                        unsafe {
                            pyre_object::w_dict_store(
                                d,
                                pyre_object::w_int_new(zc.to_u32() as i64),
                                pyre_object::w_none(),
                            );
                        }
                    }
                }
            } else {
                if !unsafe { pyre_object::is_dict(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "if you give only one argument to maketrans it must be a dict",
                    ));
                }
                let src = args[0];
                unsafe {
                    // `w_dict_items` dispatches through `is_module_dict`
                    // so `str.maketrans(some_module.__dict__)` walks the
                    // strategy storage when handed a W_ModuleDictObject.
                    for (k, v) in pyre_object::w_dict_items(src) {
                        let ord_key = if pyre_object::is_int(k) {
                            k
                        } else if pyre_object::is_str(k) {
                            let s = pyre_object::w_str_get_wtf8(k);
                            let mut cps = s.code_points();
                            let cp = cps.next();
                            if cp.is_none() || cps.next().is_some() {
                                return Err(crate::PyError::value_error(
                                    "string keys in translate table must be of length 1",
                                ));
                            }
                            pyre_object::w_int_new(cp.unwrap().to_u32() as i64)
                        } else {
                            return Err(crate::PyError::type_error(
                                "keys in translate table must be strings or integers",
                            ));
                        };
                        pyre_object::w_dict_store(d, ord_key, v);
                    }
                }
            }
            Ok(d)
        }),
    );
    for (name, func) in [
        ("__eq__", str_dunder_eq as DunderFn),
        ("__ne__", str_dunder_ne),
        ("__lt__", str_dunder_lt),
        ("__le__", str_dunder_le),
        ("__gt__", str_dunder_gt),
        ("__ge__", str_dunder_ge),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
    // unicodeobject.py descr_getnewargs — `(W_UnicodeObject(self._utf8),)`:
    // a fresh plain str from the contents, so a str subclass reduces to str.
    dict_storage_store(
        ns,
        "__getnewargs__",
        make_builtin_function_with_arity(
            "__getnewargs__",
            |args| {
                let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_str_from_wtf8(s.to_owned()),
                ]))
            },
            1,
        ),
    );
}

// ── Dict TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/dictmultiobject.py TypeDef("dict", ...)

fn init_dict_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(dict_descr_new));
    // dictmultiobject.py:446 __class_getitem__ = interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    dict_storage_store(
        ns,
        "__class_getitem__",
        pyre_object::function::w_classmethod_new(make_builtin_function(
            "__class_getitem__",
            crate::_pypy_generic_alias::generic_alias_class_getitem,
        )),
    );
    // `dictmultiobject.py:137-138 descr_init` →
    // `init_or_update(space, self, __args__, 'dict')`
    dict_storage_store(
        ns,
        "__init__",
        make_builtin_function("__init__", |args| {
            crate::type_methods::dict_init_or_update(args, "dict")
        }),
    );
    dict_storage_store(
        ns,
        "get",
        make_builtin_function("get", crate::type_methods::dict_method_get),
    );
    dict_storage_store(
        ns,
        "keys",
        make_builtin_function_with_arity("keys", crate::type_methods::dict_method_keys, 1),
    );
    dict_storage_store(
        ns,
        "values",
        make_builtin_function_with_arity("values", crate::type_methods::dict_method_values, 1),
    );
    dict_storage_store(
        ns,
        "items",
        make_builtin_function_with_arity("items", crate::type_methods::dict_method_items, 1),
    );
    dict_storage_store(
        ns,
        "update",
        make_builtin_function("update", crate::type_methods::dict_method_update),
    );
    dict_storage_store(
        ns,
        "pop",
        make_builtin_function("pop", crate::type_methods::dict_method_pop),
    );
    dict_storage_store(
        ns,
        "popitem",
        make_builtin_function_with_arity("popitem", crate::type_methods::dict_method_popitem, 1),
    );
    dict_storage_store(
        ns,
        "setdefault",
        make_builtin_function("setdefault", crate::type_methods::dict_method_setdefault),
    );
    dict_storage_store(
        ns,
        "__setitem__",
        make_builtin_function_with_arity(
            "__setitem__",
            |args| {
                crate::type_methods::arity_exact_unpack(args, "__setitem__", 2)?;
                // For plain dict: direct store. For dict subclass instance: use backing dict.
                unsafe {
                    if pyre_object::is_dict(args[0]) {
                        crate::type_methods::dict_store_checked(args[0], args[1], args[2])?;
                    } else if pyre_object::is_instance(args[0]) {
                        // dict subclass — store in __dict_data__ backing dict
                        if let Ok(backing) =
                            crate::baseobjspace::getattr_str(args[0], "__dict_data__")
                        {
                            if pyre_object::is_dict(backing) {
                                crate::type_methods::dict_store_checked(backing, args[1], args[2])?;
                            }
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            3,
        ),
    );
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                crate::type_methods::arity_exact(args, "dict.__getitem__", 1)?;
                unsafe {
                    if pyre_object::is_dict(args[0]) {
                        return crate::baseobjspace::getitem(args[0], args[1]);
                    }
                    if pyre_object::is_instance(args[0]) {
                        if let Ok(backing) =
                            crate::baseobjspace::getattr_str(args[0], "__dict_data__")
                        {
                            if pyre_object::is_dict(backing) {
                                // `dictmultiobject.py:166-170` — on a miss,
                                // dispatch `__missing__` against the SUBCLASS
                                // instance's type, not the plain-`dict` backing
                                // (so e.g. `defaultdict.__missing__` fires).
                                return match pyre_object::dictmultiobject::w_dict_lookup_checked(
                                    backing, args[1],
                                ) {
                                    Ok(Some(val)) => Ok(val),
                                    Ok(None) => crate::baseobjspace::dict_missing_or_key_error(
                                        args[0], args[1],
                                    ),
                                    Err(_) => Err(crate::baseobjspace::take_pending_hash_error()),
                                };
                            }
                        }
                    }
                }
                crate::baseobjspace::getitem(args[0], args[1])
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function_with_arity(
            "__contains__",
            |args| {
                crate::type_methods::arity_exact(args, "dict.__contains__", 1)?;
                let dict = crate::type_methods::resolve_dict_backing(args[0]);
                if !dict.is_null() {
                    return match unsafe {
                        pyre_object::dictmultiobject::w_dict_lookup_checked(dict, args[1])
                    } {
                        Ok(v) => Ok(pyre_object::w_bool_from(v.is_some())),
                        Err(_) => Err(crate::baseobjspace::take_pending_hash_error()),
                    };
                }
                Ok(pyre_object::w_bool_from(
                    crate::baseobjspace::contains_slot(args[0], args[1]).unwrap_or(false),
                ))
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function_with_arity(
            "__len__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                crate::type_methods::arity_slot(args, 0)?;
                let dict = crate::type_methods::resolve_dict_backing(args[0]);
                if !dict.is_null() {
                    return Ok(pyre_object::w_int_new(
                        unsafe { pyre_object::w_dict_len(dict) } as i64,
                    ));
                }
                crate::baseobjspace::len_slot(args[0])
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function_with_arity(
            "__repr__",
            |args| {
                // `dictmultiobject.py:130-150 descr_repr`.  Registered as a
                // method (not only the `py_repr` fast path) so dict-subclass
                // instances and `super().__repr__()` format their backing.
                if args.is_empty() {
                    return Ok(pyre_object::w_str_new("{}"));
                }
                let recv = args[0];
                let dict = crate::type_methods::resolve_dict_backing(recv);
                if dict.is_null() {
                    // Unbound `dict.__repr__(x)` on a non-dict receiver —
                    // reject it like a builtin descriptor rather than
                    // formatting an empty `{}`.
                    let tp_name = unsafe { (*(*recv).ob_type).name };
                    return Err(crate::PyError::type_error(format!(
                        "descriptor '__repr__' for 'dict' objects \
                         doesn't apply to a '{tp_name}' object"
                    )));
                }
                unsafe { Ok(pyre_object::w_str_new(&crate::display::dict_repr(dict)?)) }
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function_with_arity(
            "__iter__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                let dict = crate::type_methods::resolve_dict_backing(args[0]);
                if !dict.is_null() {
                    // Iterate over dict keys
                    return crate::baseobjspace::iter(dict);
                }
                crate::baseobjspace::iter(args[0])
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__delitem__",
        make_builtin_function_with_arity(
            "__delitem__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                // For plain dict: direct delete. For dict subclass instance: use backing dict.
                unsafe {
                    if pyre_object::is_dict(args[0]) {
                        crate::baseobjspace::delitem_slot(args[0], args[1])?;
                    } else if pyre_object::is_instance(args[0]) {
                        // dict subclass — delete from __dict_data__ backing dict
                        if let Ok(backing) =
                            crate::baseobjspace::getattr_str(args[0], "__dict_data__")
                        {
                            if pyre_object::is_dict(backing) {
                                crate::baseobjspace::delitem(backing, args[1])?;
                            }
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__eq__",
        make_builtin_function_with_arity(
            "__eq__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                // A dict subclass is instance-represented (the mapping lives in
                // the `__dict_data__` backing), so `compare` would not see it as
                // a dict and would re-dispatch to this `__eq__`, recursing.
                // Resolve each operand to its backing dict first; exact dicts
                // and non-dict operands are left unchanged for `compare`.
                let resolve = |o: PyObjectRef| {
                    let backing = crate::type_methods::resolve_dict_backing(o);
                    if backing.is_null() { o } else { backing }
                };
                let a = resolve(args[0]);
                let b = resolve(args[1]);
                // `dictmultiobject.py descr_eq`: a non-dict operand yields
                // NotImplemented. Handing it to `compare` would re-dispatch
                // to this `__eq__` (the operand is not a dict for compare's
                // fast path) and recurse.
                if !unsafe { pyre_object::is_dict(b) } {
                    return Ok(pyre_object::w_not_implemented());
                }
                crate::baseobjspace::compare(a, b, crate::baseobjspace::CompareOp::Eq)
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__or__",
        make_builtin_function_with_arity(
            "__or__",
            |args| {
                // `pypy/objspace/std/dictmultiobject.py:288 descr_or`:
                //   def descr_or(self, space, w_other):
                //       if not space.isinstance_w(w_other, space.w_dict):
                //           return space.w_NotImplemented
                //       new = self.descr_copy(space)
                //       new.descr_update(space, w_other)
                //       return new
                crate::type_methods::arity_slot(args, 1)?;
                let src = crate::type_methods::resolve_dict_backing(args[0]);
                let other = crate::type_methods::resolve_dict_backing(args[1]);
                if other.is_null() {
                    return Ok(pyre_object::w_not_implemented());
                }
                // `descr_copy` then `descr_update`: copy LHS, overlay
                // RHS — both reads go through `w_dict_items` so a
                // dict backed by a `dict_storage_proxy` (globals() /
                // module.__dict__) contributes its storage-only
                // entries too, matching PyPy's storage-strategy
                // delitem/iter parity.
                let dst = pyre_object::w_dict_new();
                if !src.is_null() {
                    for (k, v) in unsafe { pyre_object::w_dict_items(src) } {
                        unsafe { pyre_object::w_dict_store(dst, k, v) };
                    }
                }
                for (k, v) in unsafe { pyre_object::w_dict_items(other) } {
                    unsafe { pyre_object::w_dict_store(dst, k, v) };
                }
                Ok(dst)
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__ror__",
        make_builtin_function_with_arity(
            "__ror__",
            |args| {
                // `dictmultiobject.py:295 descr_ror`: `other | dict` copies
                // the right-hand-side base (other) and overlays self.
                crate::type_methods::arity_slot(args, 1)?;
                let self_ = crate::type_methods::resolve_dict_backing(args[0]);
                let other = crate::type_methods::resolve_dict_backing(args[1]);
                if other.is_null() {
                    return Ok(pyre_object::w_not_implemented());
                }
                let dst = pyre_object::w_dict_new();
                for (k, v) in unsafe { pyre_object::w_dict_items(other) } {
                    unsafe { pyre_object::w_dict_store(dst, k, v) };
                }
                if !self_.is_null() {
                    for (k, v) in unsafe { pyre_object::w_dict_items(self_) } {
                        unsafe { pyre_object::w_dict_store(dst, k, v) };
                    }
                }
                Ok(dst)
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__ior__",
        make_builtin_function_with_arity(
            "__ior__",
            |args| {
                // `dictmultiobject.py:303 descr_ior`: in-place update via
                // `update1`, returns self.
                crate::type_methods::arity_slot(args, 1)?;
                let self_ = crate::type_methods::resolve_dict_backing(args[0]);
                if !self_.is_null() {
                    crate::type_methods::dict_update1(self_, args[1])?;
                }
                Ok(args[0])
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__reversed__",
        make_builtin_function_with_arity(
            "__reversed__",
            |args| {
                // `dictmultiobject.py:207 descr_reversed`: reverse iterator
                // over the dict keys.
                let d = crate::type_methods::resolve_dict_backing(args[0]);
                let mut keys: Vec<PyObjectRef> = if d.is_null() {
                    Vec::new()
                } else {
                    unsafe { pyre_object::w_dict_items(d) }
                        .into_iter()
                        .map(|(k, _)| k)
                        .collect()
                };
                keys.reverse();
                let n = keys.len();
                let list = pyre_object::w_list_new(keys);
                Ok(pyre_object::w_seq_iter_new(list, n))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "copy",
        make_builtin_function_with_arity("copy", crate::type_methods::dict_method_copy, 1),
    );
    dict_storage_store(
        ns,
        "clear",
        make_builtin_function_with_arity(
            "clear",
            |args| {
                // `pypy/objspace/std/dictmultiobject.py:1374
                // W_DictMultiObject.descr_clear` — empties every entry
                // regardless of key type by dispatching through the
                // strategy's `clear` (`celldict.py:162-164` for
                // module dicts).  `w_dict_clear` does the dispatch.
                if !args.is_empty() {
                    let d = crate::type_methods::resolve_dict_backing(args[0]);
                    if !d.is_null() {
                        unsafe { pyre_object::dictmultiobject::w_dict_clear(d) };
                    }
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    // dict.fromkeys(iterable, value=None) — classmethod
    dict_storage_store(
        ns,
        "fromkeys",
        pyre_object::function::w_classmethod_new(make_builtin_function("fromkeys", |args| {
            // classmethod: args[0] is the bound cls; the user arguments are
            // fromkeys(iterable, value=None).
            let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            let (iterable, value) = if args.len() >= 3 {
                (args[1], args[2])
            } else if args.len() == 2 {
                (args[1], pyre_object::w_none())
            } else {
                return Err(crate::PyError::type_error(
                    "fromkeys expected at least 1 argument, got 0",
                ));
            };
            let items = crate::builtins::collect_iterable(iterable)?;
            // dictmultiobject.py:120-134 descr_fromkeys — for `dict` itself,
            // fill a fresh dict directly; for a dict subclass, construct an
            // instance via `cls()` and route through `space.setitem` so the
            // result is an instance of the subclass.
            let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
            if cls.is_null() || crate::baseobjspace::is_w(cls, w_dict_type) {
                let d = pyre_object::w_dict_new();
                for key in items {
                    unsafe { pyre_object::w_dict_store(d, key, value) };
                }
                Ok(d)
            } else {
                let d = crate::call::call_function_impl_result(cls, &[])?;
                for key in items {
                    crate::baseobjspace::setitem(d, key, value)?;
                }
                Ok(d)
            }
        })),
    );
}

// ── Mappingproxy TypeDef ─────────────────────────────────────────────
//
// `pypy/objspace/std/dictproxyobject.py:103` —
// `W_DictProxyObject.typedef = TypeDef('mappingproxy', ...)`.  All
// methods forward to `self.w_mapping` (the wrapped W_DictObject);
// pyre routes through `resolve_dict_backing`, which now unwraps the
// proxy to its inner dict so the dict-method bodies stay shared.

/// `pypy/objspace/std/dictmultiobject.py` —
/// `W_DictViewKeysObject` / `W_DictViewValuesObject` /
/// `W_DictViewItemsObject` typedef bodies. Pyre dispatches the
/// runtime methods (`__iter__` / `__len__` / `__contains__` /
/// `__repr__`) directly through baseobjspace + display arms keyed on
/// the view's PyType, so dispatch works without typedef registration.
/// Common slots shared across all three dict_view typedefs per
/// `dictmultiobject.py:1773-1788 / 1802-1813 / 1831-1840`:
/// `__iter__`, `__len__`, `__reversed__`, `__repr__`, `mapping`.
/// `dict_values` stops here; `dict_keys` / `dict_items` extend with
/// the SetLikeDictView surface in
/// `init_dict_view_set_like_type` below.
fn init_dict_view_common_slots(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function_with_arity("__iter__", |args| crate::baseobjspace::iter(args[0]), 1),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function_with_arity(
            "__len__",
            |args| crate::baseobjspace::len_slot(args[0]),
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__reversed__",
        make_builtin_function_with_arity(
            "__reversed__",
            |args| {
                let view = args[0];
                let mut snapshot = crate::type_methods::dict_view_snapshot(view);
                snapshot.reverse();
                let n = snapshot.len();
                let list = pyre_object::w_list_new(snapshot);
                Ok(pyre_object::w_seq_iter_new(list, n))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function_with_arity(
            "__repr__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_str_new(""));
                }
                Ok(pyre_object::w_str_new(&unsafe {
                    crate::display::py_repr(args[0])?
                }))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "mapping",
        make_getset_descriptor(make_builtin_function_with_arity(
            "mapping",
            |args| {
                let view = args[1];
                if view.is_null() {
                    return Ok(pyre_object::w_none());
                }
                let dict = unsafe { pyre_object::dictmultiobject::w_dict_view_get_dict(view) };
                if dict.is_null() {
                    return Ok(pyre_object::w_dict_proxy_new(pyre_object::w_dict_new()));
                }
                Ok(pyre_object::w_dict_proxy_new(dict))
            },
            2,
        )),
    );
}

/// `dictmultiobject.py` `W_DictViewKeysObject` /
/// `W_DictViewItemsObject`
/// typedef body — common slots plus `__contains__` and the
/// SetLikeDictView surface (comparisons, set ops, isdisjoint).
fn init_dict_view_set_like_type(ns: &mut DictStorage) {
    init_dict_view_common_slots(ns);
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function_with_arity(
            "__contains__",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(false));
                }
                Ok(pyre_object::w_bool_from(crate::baseobjspace::contains(
                    args[0], args[1],
                )?))
            },
            2,
        ),
    );
    register_dict_view_set_operators(ns);
}

/// `dictmultiobject.py:1831-1840 W_DictViewValuesObject.typedef` —
/// common slots only.  Values views are NOT set-like in PyPy
/// (`dictmultiobject.py:1619-1623 _is_set_like` excludes them) and
/// have no `__contains__` / set ops / comparisons of their own;
/// equality falls through to `object.__eq__`'s identity check.
fn init_dict_view_values_type(ns: &mut DictStorage) {
    init_dict_view_common_slots(ns);
}

/// `pypy/interpreter/pytraceback.py:17-101 PyTraceback.typedef` —
/// the four Python-visible getsets.
///
/// ```python
/// PyTraceback.typedef = TypeDef("traceback",
///     __new__ = interp2app(PyTraceback.descr_new),
///     __dir__ = interp2app(PyTraceback.descr__dir__),
///     __reduce__ = interp2app(PyTraceback.descr__reduce__),
///     __setstate__ = interp2app(PyTraceback.descr__setstate__),
///     tb_frame  = GetSetProperty(PyTraceback.descr_get_tb_frame),
///     tb_lasti  = GetSetProperty(PyTraceback.descr_get_tb_lasti,
///                                PyTraceback.descr_set_tb_lasti),
///     tb_lineno = GetSetProperty(PyTraceback.descr_get_tb_lineno,
///                                PyTraceback.descr_set_tb_lineno),
///     tb_next   = GetSetProperty(PyTraceback.descr_get_next,
///                                PyTraceback.descr_set_next),
/// )
/// ```
///
/// Pyre wires `tb_lasti`, `tb_lineno`, `tb_next`, `tb_frame`,
/// `__new__`, `__dir__`.
///   - `tb_frame` returns the live `PyFrame` (`FRAME_TYPE`) when it is
///     GC-owned, else a `sys.namespace` stub for a non-Gc / freed
///     frame (see the getter below).
///   - `__new__` = `TracebackType(tb_next, tb_frame, tb_lasti,
///     tb_lineno)` (3.7+ constructor), taking a live `frame` object.
///   - `__reduce__` / `__setstate__` are intentionally NOT wired:
///     CPython 3.14 tracebacks are not picklable (`pickle.dumps(tb)`
///     raises `TypeError: cannot pickle 'traceback' object`, and
///     `traceback` has no `__setstate__`).  PyPy's `_pickle_support`
///     path is PyPy-specific and would add non-CPython behavior, so it
///     is deliberately omitted (behavior authority = CPython 3.14).
/// `TracebackType(tb_next, tb_frame, tb_lasti, tb_lineno)` — the 3.7+
/// traceback constructor.  `args[0]` is the class; the four positional
/// arguments follow.  `tb_next` is a traceback or `None`; `tb_frame`
/// must be a `frame`; `tb_lasti` / `tb_lineno` are ints.  CPython's
/// `tb_lasti` is a byte offset, so it is halved to pyre's instruction-
/// unit form for storage (the `tb_lasti` getter multiplies back by 2).
fn traceback_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 5 {
        return Err(crate::PyError::type_error(format!(
            "TracebackType() takes exactly 4 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let w_next = args[1];
    let w_frame = args[2];
    let w_lasti = args[3];
    let w_lineno = args[4];

    // tb_next: a traceback or None.
    let next = if unsafe { pyre_object::is_none(w_next) } {
        pyre_object::PY_NULL
    } else if unsafe { crate::pytraceback::is_pytraceback(w_next) } {
        w_next
    } else {
        return Err(crate::PyError::type_error(format!(
            "expected traceback object or None, got '{}'",
            type_name_of(w_next)
        )));
    };

    // tb_frame: must be a `frame` object (`FRAME_TYPE`).
    if w_frame.is_null()
        || !unsafe { pyre_object::py_type_check(w_frame, &crate::pyframe::FRAME_TYPE) }
    {
        return Err(crate::PyError::type_error(format!(
            "TracebackType() argument 'tb_frame' must be frame, not {}",
            type_name_of(w_frame)
        )));
    }
    let frame = w_frame as *mut crate::pyframe::PyFrame;

    // tb_lasti / tb_lineno: integers.  `tb_lasti` arrives as a CPython
    // byte offset; store the instruction-unit form (`/ 2`).
    if !unsafe { pyre_object::is_int(w_lasti) } {
        return Err(crate::PyError::type_error(format!(
            "an integer is required (got type {})",
            type_name_of(w_lasti)
        )));
    }
    if !unsafe { pyre_object::is_int(w_lineno) } {
        return Err(crate::PyError::type_error(format!(
            "an integer is required (got type {})",
            type_name_of(w_lineno)
        )));
    }
    let lasti = unsafe { pyre_object::w_int_get_value(w_lasti) } / 2;
    let lineno = unsafe { pyre_object::w_int_get_value(w_lineno) };
    let w_code = unsafe { (*frame).fget_f_code() };

    Ok(crate::pytraceback::w_pytraceback_new(
        frame, lasti, next, lineno, w_code,
    ))
}

fn init_pytraceback_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(traceback_descr_new));
    // pytraceback.py:45-49 descr_get_tb_lasti / descr_set_tb_lasti.
    //
    // pyre stores `lasti` as an instruction-unit index (`PyFrame.last_instr`
    // increments by 1 per instruction), whereas CPython's `tb_lasti` is a
    // byte offset (2 bytes per code unit).  Report the byte-offset form so
    // `code.co_positions()` consumers — `traceback._get_code_position` does
    // `instruction_index // 2` — recover the right instruction.
    let lasti_getter = make_builtin_function_with_arity(
        "tb_lasti",
        |args| {
            let tb = args[1];
            if tb.is_null() {
                return Ok(pyre_object::w_none());
            }
            let lasti = unsafe { crate::pytraceback::w_pytraceback_get_lasti(tb) };
            Ok(pyre_object::w_int_new(lasti * 2))
        },
        2,
    );
    let lasti_setter = make_builtin_function_with_arity(
        "tb_lasti",
        |args| {
            let tb = args[1];
            let w_value = args[2];
            if tb.is_null() {
                return Ok(pyre_object::w_none());
            }
            // Inverse of the getter: incoming byte offset → instruction index.
            let v = crate::baseobjspace::int_w(w_value)?;
            unsafe { crate::pytraceback::w_pytraceback_set_lasti(tb, v / 2) };
            Ok(pyre_object::w_none())
        },
        3,
    );
    dict_storage_store(
        ns,
        "tb_lasti",
        make_getset_property_named(lasti_getter, lasti_setter, pyre_object::PY_NULL, "tb_lasti"),
    );

    // pytraceback.py:39-43 descr_get_tb_lineno / descr_set_tb_lineno.
    let lineno_getter = make_builtin_function_with_arity(
        "tb_lineno",
        |args| {
            let tb = args[1];
            if tb.is_null() {
                return Ok(pyre_object::w_none());
            }
            let n = unsafe { crate::pytraceback::w_pytraceback_get_lineno(tb) };
            Ok(pyre_object::w_int_new(n))
        },
        2,
    );
    let lineno_setter = make_builtin_function_with_arity(
        "tb_lineno",
        |args| {
            let tb = args[1];
            let w_value = args[2];
            if tb.is_null() {
                return Ok(pyre_object::w_none());
            }
            let v = crate::baseobjspace::int_w(w_value)?;
            unsafe { crate::pytraceback::w_pytraceback_set_lineno(tb, v) };
            Ok(pyre_object::w_none())
        },
        3,
    );
    dict_storage_store(
        ns,
        "tb_lineno",
        make_getset_property_named(
            lineno_getter,
            lineno_setter,
            pyre_object::PY_NULL,
            "tb_lineno",
        ),
    );

    // pytraceback.py:51-62 descr_get_next / descr_set_next — setter
    // walks the proposed chain for self-references (`:57-61
    // traceback loop detected`).
    let next_getter = make_builtin_function_with_arity(
        "tb_next",
        |args| {
            let tb = args[1];
            if tb.is_null() {
                return Ok(pyre_object::w_none());
            }
            let nxt = unsafe { crate::pytraceback::w_pytraceback_get_w_next(tb) };
            if nxt.is_null() {
                return Ok(pyre_object::w_none());
            }
            Ok(nxt)
        },
        2,
    );
    let next_setter = make_builtin_function_with_arity(
        "tb_next",
        |args| {
            let tb = args[1];
            let mut w_new = args[2];
            if tb.is_null() {
                return Ok(pyre_object::w_none());
            }
            // pytraceback.py:55 `w_next = space.interp_w(PyTraceback,
            // w_next, can_be_None=True)` — None / null → PY_NULL chain
            // terminator; anything else must be a PyTraceback.
            if w_new.is_null() || unsafe { pyre_object::is_none(w_new) } {
                w_new = pyre_object::PY_NULL;
            } else if !unsafe { crate::pytraceback::is_pytraceback(w_new) } {
                return Err(crate::PyError::type_error(
                    "expected traceback object or None".to_string(),
                ));
            }
            if unsafe { crate::pytraceback::w_pytraceback_set_w_next(tb, w_new) }.is_err() {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::ValueError,
                    "traceback loop detected".to_string(),
                ));
            }
            Ok(pyre_object::w_none())
        },
        3,
    );
    dict_storage_store(
        ns,
        "tb_next",
        make_getset_property_named(next_getter, next_setter, pyre_object::PY_NULL, "tb_next"),
    );

    // pytraceback.py:34 descr_get_tb_frame — return the live `PyFrame`
    // itself (`FRAME_TYPE` typedef) as the user-visible `frame` object.
    // The traceback keeps the raising frame's chain reachable through
    // `pytraceback_object_custom_trace`, so a GC-owned frame is still
    // alive here.  The guard must match the custom_trace's guard
    // (`try_gc_owns_object`): only frames forwarded as managed edges
    // survive; a non-Gc frame falls back to the `sys.namespace` stub
    // built from the retained `w_code` + stamped line number.
    //
    // A frame is non-Gc only when the GC stable-alloc hook was never
    // pytraceback.py:34 descr_get_tb_frame — return the live `PyFrame`
    // itself (`FRAME_TYPE` typedef) as the user-visible `frame` object.
    // The GC subsystem is installed at boot (`init_gc_subsystem`), so all
    // frames — including under `PYRE_JIT=0` — are GC-owned oldgen blocks
    // that stay alive as long as the traceback references them.
    let frame_getter = make_builtin_function_with_arity(
        "tb_frame",
        |args| {
            let tb = args[1];
            if tb.is_null() {
                return Ok(pyre_object::w_none());
            }
            let frame = unsafe { crate::pytraceback::w_pytraceback_get_frame(tb) };
            if frame.is_null() {
                return Ok(pyre_object::w_none());
            }
            // Mark escaped so the JIT keeps the frame materialised for
            // the exposed reference (pyframe.py:176 `mark_as_escaped`),
            // mirroring `sys._getframe`.
            unsafe { (*frame).mark_as_escaped() };
            Ok(frame as pyre_object::PyObjectRef)
        },
        2,
    );
    dict_storage_store(
        ns,
        "tb_frame",
        make_getset_property_named(
            frame_getter,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            "tb_frame",
        ),
    );
    // `pytraceback.py:99-101 descr__dir__` — returns the list of
    // public traceback attribute names.
    dict_storage_store(
        ns,
        "__dir__",
        make_builtin_function_with_arity(
            "__dir__",
            |_args| {
                Ok(pyre_object::w_list_new(vec![
                    pyre_object::w_str_new("tb_frame"),
                    pyre_object::w_str_new("tb_next"),
                    pyre_object::w_str_new("tb_lasti"),
                    pyre_object::w_str_new("tb_lineno"),
                ]))
            },
            1,
        ),
    );
}

/// `pypy/interpreter/typedef.py:736-753 PyFrame.typedef` — the `frame`
/// type's getset descriptors + `clear` / `__repr__`.  The receiver
/// (`args[1]`) is a live `PyFrame` object (its `ob_header.ob_type` is
/// `FRAME_TYPE`); every field access casts it to `*mut PyFrame`.  A read
/// through a null / already-freed receiver returns `None` rather than
/// dereferencing.  `f_lineno`'s setter is [`PyFrame::fset_f_lineno`],
/// which validates the line-jump via `mark_stacks`; the read-only getsets
/// and `f_trace*` setters mirror `pyframe.py:641-806` directly.
fn init_frame_type(ns: &mut DictStorage) {
    use crate::pyframe::PyFrame;

    // Helper: resolve the receiver to `&mut PyFrame`, or return `w_none()`
    // (the closures each inline this because Rust closures can't share a
    // borrow-returning helper cleanly).
    fn frame_ptr(w_obj: pyre_object::PyObjectRef) -> *mut PyFrame {
        w_obj as *mut PyFrame
    }

    // f_code — read-only; the `PyCode` wrapper (pyframe.py:641 fget_code).
    let code_getter = make_builtin_function_with_arity(
        "f_code",
        |args| {
            let f = frame_ptr(args[1]);
            if f.is_null() {
                return Ok(pyre_object::w_none());
            }
            Ok(unsafe { &*f }.fget_f_code())
        },
        2,
    );
    dict_storage_store(
        ns,
        "f_code",
        make_getset_descriptor_named(code_getter, "f_code"),
    );

    // f_globals — read-only (pyframe.py:647 fget_w_globals).
    let globals_getter = make_builtin_function_with_arity(
        "f_globals",
        |args| {
            let f = frame_ptr(args[1]);
            if f.is_null() {
                return Ok(pyre_object::w_none());
            }
            let w = unsafe { &*f }.get_w_globals();
            Ok(if w.is_null() {
                pyre_object::w_none()
            } else {
                w
            })
        },
        2,
    );
    dict_storage_store(
        ns,
        "f_globals",
        make_getset_descriptor_named(globals_getter, "f_globals"),
    );

    // f_locals — read-only; runs `fast2locals` (pyframe.py:644
    // fget_getdictscope), so it needs `&mut` and can raise.
    let locals_getter = make_builtin_function_with_arity(
        "f_locals",
        |args| {
            let f = frame_ptr(args[1]);
            if f.is_null() {
                return Ok(pyre_object::w_none());
            }
            let w = unsafe { &mut *f }.getdictscope()?;
            Ok(if w.is_null() {
                pyre_object::w_dict_new()
            } else {
                w
            })
        },
        2,
    );
    dict_storage_store(
        ns,
        "f_locals",
        make_getset_descriptor_named(locals_getter, "f_locals"),
    );

    // f_back — read-only; the next non-hidden frame (pyframe.py:767).
    let back_getter = make_builtin_function_with_arity(
        "f_back",
        |args| {
            let f = frame_ptr(args[1]);
            if f.is_null() {
                return Ok(pyre_object::w_none());
            }
            let back = unsafe { &*f }.fget_f_back();
            Ok(if back.is_null() {
                pyre_object::w_none()
            } else {
                // Exposing the frame to app level: mark escaped so the JIT
                // materialises it (pyframe.py:176), mirroring `_getframe`.
                unsafe { (*back).mark_as_escaped() };
                back as pyre_object::PyObjectRef
            })
        },
        2,
    );
    dict_storage_store(
        ns,
        "f_back",
        make_getset_descriptor_named(back_getter, "f_back"),
    );

    // f_lasti — read-only bytecode offset (pyframe.py:770).
    //
    // pyre stores `last_instr` as an instruction-unit index (increments
    // by 1 per instruction); CPython's `f_lasti` is a byte offset
    // (2 bytes per code unit).  Report the byte-offset form (× 2) so
    // `dis` / `code.co_positions()` consumers that do `f_lasti // 2`
    // recover the right instruction — the same adaptation `tb_lasti`
    // uses (`typedef.rs` tb_lasti getter).
    let lasti_getter = make_builtin_function_with_arity(
        "f_lasti",
        |args| {
            let f = frame_ptr(args[1]);
            if f.is_null() {
                return Ok(pyre_object::w_none());
            }
            Ok(pyre_object::w_int_new(
                unsafe { &*f }.fget_f_lasti() as i64 * 2,
            ))
        },
        2,
    );
    dict_storage_store(
        ns,
        "f_lasti",
        make_getset_descriptor_named(lasti_getter, "f_lasti"),
    );

    // f_builtins — read-only builtin dict (pyframe.py:761).
    let builtins_getter = make_builtin_function_with_arity(
        "f_builtins",
        |args| {
            let f = frame_ptr(args[1]);
            if f.is_null() {
                return Ok(pyre_object::w_none());
            }
            let w = unsafe { &*f }.fget_f_builtins();
            Ok(if w.is_null() {
                pyre_object::w_none()
            } else {
                w
            })
        },
        2,
    );
    dict_storage_store(
        ns,
        "f_builtins",
        make_getset_descriptor_named(builtins_getter, "f_builtins"),
    );

    // f_lineno — read/write (pyframe.py:654 fget_f_lineno / :666 fset).
    // The getter returns `None` for an untraced frame whose line is -1;
    // the setter is `fset_f_lineno`, which validates the debugger
    // line-jump via `mark_stacks` and raises `ValueError` on an illegal
    // target (only permitted from within a trace function).
    let lineno_getter = make_builtin_function_with_arity(
        "f_lineno",
        |args| {
            let f = frame_ptr(args[1]);
            if f.is_null() {
                return Ok(pyre_object::w_none());
            }
            let frame = unsafe { &*f };
            let lineno = frame.get_last_lineno();
            if frame.get_w_f_trace().is_null() {
                if lineno == -1 {
                    return Ok(pyre_object::w_none());
                }
                return Ok(pyre_object::w_int_new(lineno as i64));
            }
            let lineno = if lineno == -1 {
                frame
                    .code()
                    .first_line_number
                    .map_or(-1, |n| n.get() as isize)
            } else {
                lineno
            };
            Ok(pyre_object::w_int_new(lineno as i64))
        },
        2,
    );
    let lineno_setter = make_builtin_function_with_arity(
        "f_lineno",
        |args| {
            let f = frame_ptr(args[1]);
            if f.is_null() {
                return Ok(pyre_object::w_none());
            }
            let new_lineno = crate::baseobjspace::int_w(args[2])
                .map_err(|_| crate::PyError::value_error("lineno must be an integer"))?;
            unsafe { &mut *f }.fset_f_lineno(new_lineno as isize)?;
            Ok(pyre_object::w_none())
        },
        3,
    );
    dict_storage_store(
        ns,
        "f_lineno",
        make_getset_property_named(
            lineno_getter,
            lineno_setter,
            pyre_object::PY_NULL,
            "f_lineno",
        ),
    );

    // f_trace — read/write/delete (pyframe.py:773-785).
    let trace_getter = make_builtin_function_with_arity(
        "f_trace",
        |args| {
            let f = frame_ptr(args[1]);
            if f.is_null() {
                return Ok(pyre_object::w_none());
            }
            let w = unsafe { &*f }.fget_f_trace();
            Ok(if w.is_null() {
                pyre_object::w_none()
            } else {
                w
            })
        },
        2,
    );
    let trace_setter = make_builtin_function_with_arity(
        "f_trace",
        |args| {
            let f = frame_ptr(args[1]);
            if !f.is_null() {
                unsafe { &mut *f }.fset_f_trace(args[2]);
            }
            Ok(pyre_object::w_none())
        },
        3,
    );
    let trace_deleter = make_builtin_function_with_arity(
        "f_trace",
        |args| {
            let f = frame_ptr(args[1]);
            if !f.is_null() {
                unsafe { &mut *f }.fdel_f_trace();
            }
            Ok(pyre_object::w_none())
        },
        2,
    );
    dict_storage_store(
        ns,
        "f_trace",
        make_getset_property_named(trace_getter, trace_setter, trace_deleter, "f_trace"),
    );

    // f_trace_lines — read/write bool (pyframe.py:787-791).
    let trace_lines_getter = make_builtin_function_with_arity(
        "f_trace_lines",
        |args| {
            let f = frame_ptr(args[1]);
            if f.is_null() {
                return Ok(pyre_object::w_none());
            }
            Ok(pyre_object::w_bool_from(
                unsafe { &*f }.fget_f_trace_lines(),
            ))
        },
        2,
    );
    let trace_lines_setter = make_builtin_function_with_arity(
        "f_trace_lines",
        |args| {
            let f = frame_ptr(args[1]);
            if !f.is_null() {
                let v = crate::baseobjspace::is_true(args[2])?;
                unsafe { &mut *f }.fset_f_trace_lines(v);
            }
            Ok(pyre_object::w_none())
        },
        3,
    );
    dict_storage_store(
        ns,
        "f_trace_lines",
        make_getset_property_named(
            trace_lines_getter,
            trace_lines_setter,
            pyre_object::PY_NULL,
            "f_trace_lines",
        ),
    );

    // f_trace_opcodes — read/write bool (pyframe.py:793-797).
    let trace_opcodes_getter = make_builtin_function_with_arity(
        "f_trace_opcodes",
        |args| {
            let f = frame_ptr(args[1]);
            if f.is_null() {
                return Ok(pyre_object::w_none());
            }
            Ok(pyre_object::w_bool_from(
                unsafe { &*f }.fget_f_trace_opcodes(),
            ))
        },
        2,
    );
    let trace_opcodes_setter = make_builtin_function_with_arity(
        "f_trace_opcodes",
        |args| {
            let f = frame_ptr(args[1]);
            if !f.is_null() {
                let v = crate::baseobjspace::is_true(args[2])?;
                unsafe { &mut *f }.fset_f_trace_opcodes(v);
            }
            Ok(pyre_object::w_none())
        },
        3,
    );
    dict_storage_store(
        ns,
        "f_trace_opcodes",
        make_getset_property_named(
            trace_opcodes_getter,
            trace_opcodes_setter,
            pyre_object::PY_NULL,
            "f_trace_opcodes",
        ),
    );

    // clear() — interp2app (pyframe.py:805 descr_clear).
    dict_storage_store(
        ns,
        "clear",
        make_builtin_function_with_arity(
            "clear",
            |args| {
                let f = frame_ptr(args[0]);
                if !f.is_null() {
                    unsafe { &mut *f }.descr_clear()?;
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // __repr__ — interp2app (pyframe.py:849 descr_repr).
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function_with_arity(
            "__repr__",
            |args| {
                let f = frame_ptr(args[0]);
                if f.is_null() {
                    return Ok(pyre_object::w_str_new("<frame (null)>"));
                }
                Ok(pyre_object::w_str_new(&unsafe { &*f }.descr_repr()))
            },
            1,
        ),
    );
}

/// `pypy/objspace/std/dictmultiobject.py:1605-1623`
/// `_all_contained_in` + `_is_set_like` — shared helpers for
/// `SetLikeDictView`'s comparison + set-op dispatch.  Pyre folds
/// the three view types into one `W_DictViewObject`, so kind-aware
/// branching happens here.
fn dict_view_is_set_like(obj: pyre_object::PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    unsafe {
        if pyre_object::is_set(obj) || pyre_object::is_frozenset(obj) {
            return true;
        }
        if pyre_object::dictmultiobject::is_dict_view(obj) {
            let kind = pyre_object::dictmultiobject::w_dict_view_get_kind(obj);
            return matches!(
                kind,
                pyre_object::dictmultiobject::DictViewKind::Keys
                    | pyre_object::dictmultiobject::DictViewKind::Items
            );
        }
        false
    }
}

fn dict_view_all_contained_in(
    view: pyre_object::PyObjectRef,
    other: pyre_object::PyObjectRef,
) -> Result<bool, crate::PyError> {
    let snapshot = crate::type_methods::dict_view_snapshot(view);
    for item in snapshot {
        if !crate::baseobjspace::contains(other, item)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[derive(Clone, Copy)]
enum DictViewCmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

fn dict_view_compare(
    self_view: pyre_object::PyObjectRef,
    other: pyre_object::PyObjectRef,
    op: DictViewCmp,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if !dict_view_is_set_like(other) {
        // PyPy returns NotImplemented; pyre's compare path turns
        // that into the python `NotImplemented` singleton through
        // the bytecode dispatch, so emit it directly here.
        return Ok(pyre_object::w_not_implemented());
    }
    let self_len = unsafe { crate::baseobjspace::len(self_view)? };
    let other_len = unsafe { crate::baseobjspace::len(other)? };
    let self_n = unsafe { pyre_object::w_int_get_value(self_len) };
    let other_n = unsafe { pyre_object::w_int_get_value(other_len) };
    let result = match op {
        // dictmultiobject.py:1628-1635 descr_eq
        DictViewCmp::Eq => self_n == other_n && dict_view_all_contained_in(self_view, other)?,
        DictViewCmp::Ne => !(self_n == other_n && dict_view_all_contained_in(self_view, other)?),
        // dictmultiobject.py:1637-1642 descr_lt
        DictViewCmp::Lt => self_n < other_n && dict_view_all_contained_in(self_view, other)?,
        DictViewCmp::Le => self_n <= other_n && dict_view_all_contained_in(self_view, other)?,
        // dictmultiobject.py:1651-1656 descr_gt — flips direction.
        DictViewCmp::Gt => self_n > other_n && dict_view_all_contained_in(other, self_view)?,
        DictViewCmp::Ge => self_n >= other_n && dict_view_all_contained_in(other, self_view)?,
    };
    Ok(pyre_object::w_bool_from(result))
}

/// `dictmultiobject.py:1665-1690 descr_isdisjoint` — iterate other,
/// reject as soon as any item is in self.  Pyre's snapshot-based
/// `contains` over the view materialises the (k, v) tuple wrapping
/// for items views, matching the PyPy semantics.
fn dict_view_isdisjoint(
    self_view: pyre_object::PyObjectRef,
    other: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if std::ptr::eq(self_view, other) {
        let n = unsafe { crate::baseobjspace::len(self_view)? };
        return Ok(pyre_object::w_bool_from(
            unsafe { pyre_object::w_int_get_value(n) } == 0,
        ));
    }
    let other_items = crate::builtins::collect_iterable(other)?;
    for item in other_items {
        if crate::baseobjspace::contains(self_view, item)? {
            return Ok(pyre_object::w_bool_from(false));
        }
    }
    Ok(pyre_object::w_bool_from(true))
}

/// `dictmultiobject.py:1692-1708 _as_set_op` — produce a fresh set
/// holding the result of the named set operation between
/// `self_view` and `other`.  PyPy delegates to set's `_update`
/// methods for efficiency; pyre computes the result inline because
/// pyre's set typedef does not expose the in-place mutators yet.
/// Both shapes (forward and reflected) are reachable from the same
/// helper because set ops on the supported subset (-, &, |, ^) are
/// commutative under "build set(LHS) and combine with RHS"
/// semantics — the reverse caller just swaps the operand order.
#[derive(Clone, Copy)]
enum DictViewSetOp {
    Sub,
    And,
    Or,
    Xor,
}

fn dict_view_set_op_compute(
    self_view: pyre_object::PyObjectRef,
    other: pyre_object::PyObjectRef,
    op: DictViewSetOp,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let self_items = crate::type_methods::dict_view_snapshot(self_view);
    let other_items = crate::builtins::collect_iterable(other)?;
    // Materialise `other` as a set for O(1) `contains` lookups.  `set`
    // requires hashable elements; PyPy raises TypeError naturally
    // through the underlying set constructor, which is what `_as_set_op`
    // surfaces too.
    let other_set = pyre_object::w_set_from_items(&other_items);
    let result_items: Vec<pyre_object::PyObjectRef> = match op {
        // dictmultiobject.py:1705 sub → difference_update on set(self)
        DictViewSetOp::Sub => self_items
            .into_iter()
            .filter(|&item| !unsafe { pyre_object::w_set_contains(other_set, item) })
            .collect(),
        // dictmultiobject.py:1706 and → intersection_update
        DictViewSetOp::And => self_items
            .into_iter()
            .filter(|&item| unsafe { pyre_object::w_set_contains(other_set, item) })
            .collect(),
        // dictmultiobject.py:1707 or → update (set union, dedup via set ctor below)
        DictViewSetOp::Or => {
            let mut combined: Vec<pyre_object::PyObjectRef> = self_items;
            combined.extend(other_items);
            combined
        }
        // dictmultiobject.py:1708 xor → symmetric_difference_update
        DictViewSetOp::Xor => {
            let self_set = pyre_object::w_set_from_items(&self_items);
            let mut out: Vec<pyre_object::PyObjectRef> = self_items
                .into_iter()
                .filter(|&item| !unsafe { pyre_object::w_set_contains(other_set, item) })
                .collect();
            for item in other_items {
                if !unsafe { pyre_object::w_set_contains(self_set, item) } {
                    out.push(item);
                }
            }
            out
        }
    };
    Ok(pyre_object::w_set_from_items(&result_items))
}

fn dict_view_set_op(
    self_view: pyre_object::PyObjectRef,
    other: pyre_object::PyObjectRef,
    op_name: &str,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let op = match op_name {
        "difference_update" => DictViewSetOp::Sub,
        "intersection_update" => DictViewSetOp::And,
        "update" => DictViewSetOp::Or,
        "symmetric_difference_update" => DictViewSetOp::Xor,
        _ => return Err(crate::PyError::type_error("unknown set op")),
    };
    dict_view_set_op_compute(self_view, other, op)
}

fn dict_view_rset_op(
    self_view: pyre_object::PyObjectRef,
    other: pyre_object::PyObjectRef,
    op_name: &str,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let op = match op_name {
        // PyPy's reverse ops swap operand order: `other - self_view`,
        // `other & self_view`, etc.  Sub/And are not commutative, so
        // the swap matters; Or/Xor are commutative.
        "difference_update" => {
            // other - self_view
            let other_items = crate::builtins::collect_iterable(other)?;
            let self_items = crate::type_methods::dict_view_snapshot(self_view);
            let self_set = pyre_object::w_set_from_items(&self_items);
            let result_items: Vec<pyre_object::PyObjectRef> = other_items
                .into_iter()
                .filter(|&item| !unsafe { pyre_object::w_set_contains(self_set, item) })
                .collect();
            return Ok(pyre_object::w_set_from_items(&result_items));
        }
        "intersection_update" => DictViewSetOp::And,
        "update" => DictViewSetOp::Or,
        "symmetric_difference_update" => DictViewSetOp::Xor,
        _ => return Err(crate::PyError::type_error("unknown set op")),
    };
    dict_view_set_op_compute(self_view, other, op)
}

// Top-level fn-pointer dispatchers for each comparator and set op
// (`make_builtin_function_with_arity` requires a `fn` pointer — closures
// that capture per-op state are not allowed, so each spec gets its own
// thin wrapper that calls into the shared `dict_view_*` helpers).
fn dict_view_descr_eq(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_compare(args[0], args[1], DictViewCmp::Eq)
}
fn dict_view_descr_ne(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_compare(args[0], args[1], DictViewCmp::Ne)
}
fn dict_view_descr_lt(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_compare(args[0], args[1], DictViewCmp::Lt)
}
fn dict_view_descr_le(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_compare(args[0], args[1], DictViewCmp::Le)
}
fn dict_view_descr_gt(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_compare(args[0], args[1], DictViewCmp::Gt)
}
fn dict_view_descr_ge(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_compare(args[0], args[1], DictViewCmp::Ge)
}
fn dict_view_descr_isdisjoint(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "isdisjoint() takes exactly one argument",
        ));
    }
    dict_view_isdisjoint(args[0], args[1])
}
fn dict_view_descr_sub(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_set_op(args[0], args[1], "difference_update")
}
fn dict_view_descr_and(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_set_op(args[0], args[1], "intersection_update")
}
fn dict_view_descr_or(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_set_op(args[0], args[1], "update")
}
fn dict_view_descr_xor(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_set_op(args[0], args[1], "symmetric_difference_update")
}
fn dict_view_descr_rsub(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_rset_op(args[0], args[1], "difference_update")
}
fn dict_view_descr_rand(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_rset_op(args[0], args[1], "intersection_update")
}
fn dict_view_descr_ror(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_rset_op(args[0], args[1], "update")
}
fn dict_view_descr_rxor(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_not_implemented());
    }
    dict_view_rset_op(args[0], args[1], "symmetric_difference_update")
}

fn register_dict_view_set_operators(ns: &mut DictStorage) {
    // Comparisons (Items/Keys only — Values returns NotImplemented
    // because `dict_view_is_set_like` rejects non-set-like LHS).
    for (name, func) in [
        ("__eq__", dict_view_descr_eq as fn(&[_]) -> _),
        ("__ne__", dict_view_descr_ne),
        ("__lt__", dict_view_descr_lt),
        ("__le__", dict_view_descr_le),
        ("__gt__", dict_view_descr_gt),
        ("__ge__", dict_view_descr_ge),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
    // `dictmultiobject.py:1797 isdisjoint = interp2app(descr_isdisjoint)`
    dict_storage_store(
        ns,
        "isdisjoint",
        make_builtin_function_with_arity("isdisjoint", dict_view_descr_isdisjoint, 2),
    );
    // `dictmultiobject.py:1705-1708 _as_set_op` — set ops route
    // through `set(self).METHOD(other)`; reflected variants build
    // `set(other)` and merge self in.
    for (name, func) in [
        ("__sub__", dict_view_descr_sub as fn(&[_]) -> _),
        ("__and__", dict_view_descr_and),
        ("__or__", dict_view_descr_or),
        ("__xor__", dict_view_descr_xor),
        ("__rsub__", dict_view_descr_rsub),
        ("__rand__", dict_view_descr_rand),
        ("__ror__", dict_view_descr_ror),
        ("__rxor__", dict_view_descr_rxor),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
}

/// `dictproxyobject.py:20 descr_new(space, w_type, w_mapping)` — wrap a
/// mapping (exposes `__getitem__`, not a list/tuple) in a read-only
/// proxy.  `types.MappingProxyType(d)` (`type(type.__dict__)(d)`)
/// resolves here; without it the type-call fell through to the default
/// `object.__new__`, producing a proxy with an empty/NULL mapping.
fn mappingproxy_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // args[0] = cls, args[1] = mapping.
    let w_mapping = match args.get(1) {
        Some(&m) if !m.is_null() => m,
        _ => {
            return Err(crate::PyError::type_error(
                "mappingproxy() missing required argument 'mapping' (pos 1)",
            ));
        }
    };
    let has_getitem = r#type(w_mapping)
        .map(|t| unsafe { crate::baseobjspace::lookup_in_type(t, "__getitem__") }.is_some())
        .unwrap_or(false);
    let is_seq = unsafe { pyre_object::is_list(w_mapping) || pyre_object::is_tuple(w_mapping) };
    if !has_getitem || is_seq {
        let tp = unsafe { (*(*w_mapping).ob_type).name };
        return Err(crate::PyError::type_error(format!(
            "mappingproxy() argument must be a mapping, not {tp}"
        )));
    }
    Ok(pyre_object::w_dict_proxy_new(w_mapping))
}

fn init_mappingproxy_type(ns: &mut DictStorage) {
    // dictproxyobject.py:105 __new__=interp2app(descr_new)
    dict_storage_store(ns, "__new__", make_new_descr(mappingproxy_descr_new));
    // dictproxyobject.py:117 __class_getitem__ = interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    dict_storage_store(
        ns,
        "__class_getitem__",
        pyre_object::function::w_classmethod_new(make_builtin_function(
            "__class_getitem__",
            crate::_pypy_generic_alias::generic_alias_class_getitem,
        )),
    );
    // dictproxyobject.py:32 descr_len → space.len(self.w_mapping)
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            crate::type_methods::arity_slot(args, 0)?;
            crate::baseobjspace::len_slot(args[0])
        }),
    );
    // dictproxyobject.py:35 descr_getitem → space.getitem(self.w_mapping, w_key)
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function("__getitem__", |args| {
            crate::type_methods::arity_slot(args, 1)?;
            crate::baseobjspace::getitem(args[0], args[1])
        }),
    );
    // dictproxyobject.py:38 descr_contains → space.contains(self.w_mapping, w_key)
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function("__contains__", |args| {
            crate::type_methods::arity_slot(args, 1)?;
            Ok(pyre_object::w_bool_from(crate::baseobjspace::contains(
                args[0], args[1],
            )?))
        }),
    );
    // dictproxyobject.py:41 descr_iter → space.iter(self.w_mapping)
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            crate::baseobjspace::iter(args[0])
        }),
    );
    // dictproxyobject.py:47 descr_repr →
    // `b"mappingproxy(%s)" % space.utf8_w(space.repr(self.w_mapping))`
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function("__repr__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new("mappingproxy({})"));
            }
            unsafe { Ok(pyre_object::w_str_new(&crate::display::py_repr(args[0])?)) }
        }),
    );
    // dictproxyobject.py:44 descr_str → space.str(self.w_mapping)
    dict_storage_store(
        ns,
        "__str__",
        make_builtin_function("__str__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new(""));
            }
            unsafe { Ok(pyre_object::w_str_new(&crate::display::py_str(args[0])?)) }
        }),
    );
    // dictproxyobject.py:67 descr_ior → unconditional TypeError; the
    // proxy is read-only so in-place merge is rejected by name even
    // when the rhs would otherwise be acceptable for `__or__`.
    dict_storage_store(
        ns,
        "__ior__",
        make_builtin_function("__ior__", |_args| {
            Err(crate::PyError::type_error(
                "'|=' is not supported by mappingproxy; use '|' instead",
            ))
        }),
    );
    // dictproxyobject.py:51 descr_or →
    // `copy_self.update(w_other); return copy_self`.  Implemented via
    // `dict_method_copy` (unwraps proxy through resolve_dict_backing)
    // followed by an items merge from `w_other`.
    dict_storage_store(
        ns,
        "__or__",
        make_builtin_function("__or__", |args| {
            crate::type_methods::arity_slot(args, 1)?;
            let lhs = args[0];
            let rhs = unsafe {
                if pyre_object::is_dict_proxy(args[1]) {
                    pyre_object::w_dict_proxy_get_mapping(args[1])
                } else {
                    args[1]
                }
            };
            if !unsafe { pyre_object::is_dict(rhs) } {
                return Ok(pyre_object::w_not_implemented());
            }
            let new_dict = crate::type_methods::dict_method_copy(&[lhs])?;
            crate::type_methods::dict_method_update(&[new_dict, rhs])?;
            Ok(new_dict)
        }),
    );
    // dictproxyobject.py:60 descr_ror →
    // `space.call_method(w_other, '__or__', self.w_mapping)`.
    dict_storage_store(
        ns,
        "__ror__",
        make_builtin_function("__ror__", |args| {
            crate::type_methods::arity_slot(args, 1)?;
            let self_mapping = unsafe {
                if pyre_object::is_dict_proxy(args[0]) {
                    pyre_object::w_dict_proxy_get_mapping(args[0])
                } else {
                    args[0]
                }
            };
            let lhs = args[1];
            if !unsafe { pyre_object::is_dict(lhs) } {
                return Ok(pyre_object::w_not_implemented());
            }
            let new_dict = crate::type_methods::dict_method_copy(&[lhs])?;
            crate::type_methods::dict_method_update(&[new_dict, self_mapping])?;
            Ok(new_dict)
        }),
    );
    // dictproxyobject.py:87 descr_reversed →
    // `space.call_method(self.w_mapping, '__reversed__')`.  Pyre lacks
    // a dedicated reverse iterator on dict, so fall back to building
    // a list of keys in reverse insertion order.
    dict_storage_store(
        ns,
        "__reversed__",
        make_builtin_function("__reversed__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            let dict = crate::type_methods::resolve_dict_backing(args[0]);
            if dict.is_null() {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            let mut keys: Vec<pyre_object::PyObjectRef> = unsafe {
                pyre_object::w_dict_items(dict)
                    .into_iter()
                    .map(|(k, _)| k)
                    .collect()
            };
            keys.reverse();
            crate::baseobjspace::iter(pyre_object::w_list_new(keys))
        }),
    );
    // dictproxyobject.py:71 get_w / 75 keys_w / 78 values_w / 81 items_w /
    // 84 copy_w — forward through `dict_method_*` (which unwraps the
    // proxy via `resolve_dict_backing`).
    dict_storage_store(
        ns,
        "get",
        make_builtin_function("get", crate::type_methods::dict_method_get),
    );
    dict_storage_store(
        ns,
        "keys",
        make_builtin_function("keys", crate::type_methods::dict_method_keys),
    );
    dict_storage_store(
        ns,
        "values",
        make_builtin_function("values", crate::type_methods::dict_method_values),
    );
    dict_storage_store(
        ns,
        "items",
        make_builtin_function("items", crate::type_methods::dict_method_items),
    );
    dict_storage_store(
        ns,
        "copy",
        make_builtin_function("copy", crate::type_methods::dict_method_copy),
    );
    // dictproxyobject.py:91-100 cmp methods (eq/ne/lt/le/gt/ge) →
    // `getattr(space, op)(self.w_mapping, w_other)`.  Pyre routes
    // through `space.compare`; the proxy's `space.eq`/`space.lt`/etc.
    // path runs the same `resolve_dict_backing` unwrap.  Each
    // comparison gets its own `fn` so the pointer stays
    // non-capturing.
    fn cmp_helper(
        args: &[PyObjectRef],
        op: crate::baseobjspace::CompareOp,
    ) -> Result<PyObjectRef, crate::PyError> {
        crate::type_methods::arity_slot(args, 1)?;
        // descr_op → getattr(space, op)(self.w_mapping, w_other): the
        // comparison runs on the wrapped mapping, not the proxy itself.
        let self_mapping = unsafe {
            if pyre_object::is_dict_proxy(args[0]) {
                pyre_object::w_dict_proxy_get_mapping(args[0])
            } else {
                args[0]
            }
        };
        crate::baseobjspace::compare(self_mapping, args[1], op)
    }
    fn proxy_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Eq)
    }
    fn proxy_ne(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Ne)
    }
    fn proxy_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Lt)
    }
    fn proxy_le(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Le)
    }
    fn proxy_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Gt)
    }
    fn proxy_ge(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        cmp_helper(args, crate::baseobjspace::CompareOp::Ge)
    }
    dict_storage_store(ns, "__eq__", make_builtin_function("__eq__", proxy_eq));
    dict_storage_store(ns, "__ne__", make_builtin_function("__ne__", proxy_ne));
    dict_storage_store(ns, "__lt__", make_builtin_function("__lt__", proxy_lt));
    dict_storage_store(ns, "__le__", make_builtin_function("__le__", proxy_le));
    dict_storage_store(ns, "__gt__", make_builtin_function("__gt__", proxy_gt));
    dict_storage_store(ns, "__ge__", make_builtin_function("__ge__", proxy_ge));
}

// ── Tuple TypeDef ────────────────────────────────────────────────────

fn init_tuple_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(tuple_descr_new));
    // tupleobject.py:354 __class_getitem__ = interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    dict_storage_store(
        ns,
        "__class_getitem__",
        pyre_object::function::w_classmethod_new(make_builtin_function(
            "__class_getitem__",
            crate::_pypy_generic_alias::generic_alias_class_getitem,
        )),
    );
    dict_storage_store(
        ns,
        "index",
        make_builtin_function("index", crate::type_methods::tuple_method_index),
    );
    dict_storage_store(
        ns,
        "count",
        make_builtin_function_with_arity("count", crate::type_methods::tuple_method_count, 2),
    );
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function_with_arity(
            "__contains__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                Ok(pyre_object::w_bool_from(
                    crate::baseobjspace::contains_slot(args[0], args[1]).unwrap_or(false),
                ))
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function_with_arity(
            "__len__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                crate::type_methods::arity_slot(args, 0)?;
                Ok(pyre_object::w_int_new(
                    unsafe { pyre_object::w_tuple_len(args[0]) } as i64,
                ))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__iter__",
        // Build the storage iterator directly rather than re-entering
        // `space.iter()` — a `tuple` subclass that calls `super().__iter__()`
        // would otherwise be re-dispatched back to its own override.
        make_builtin_function_with_arity(
            "__iter__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                if obj.is_null() {
                    return Ok(pyre_object::w_none());
                }
                Ok(pyre_object::w_seq_iter_new(obj, unsafe {
                    pyre_object::w_tuple_len(obj)
                }))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                crate::baseobjspace::getitem_slot(args[0], args[1])
            },
            2,
        ),
    );
    // `tupleobject.c:tuple_concat` rejects a non-tuple operand with
    // TypeError; `*` requires an integer count.
    dict_storage_store(
        ns,
        "__add__",
        make_builtin_function_with_arity(
            "__add__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                if unsafe { pyre_object::is_tuple(args[1]) } {
                    unsafe { crate::objspace::descroperation::tuple_concat(args[0], args[1]) }
                } else {
                    Err(crate::PyError::type_error(format!(
                        "can only concatenate tuple (not \"{}\") to tuple",
                        arg_type_name(args[1])
                    )))
                }
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__mul__",
        make_builtin_function_with_arity("__mul__", tuple_descr_mul, 2),
    );
    dict_storage_store(
        ns,
        "__rmul__",
        make_builtin_function_with_arity("__rmul__", tuple_descr_mul, 2),
    );
    for (name, func) in [
        ("__eq__", tuple_dunder_eq as DunderFn),
        ("__ne__", tuple_dunder_ne),
        ("__lt__", tuple_dunder_lt),
        ("__le__", tuple_dunder_le),
        ("__gt__", tuple_dunder_gt),
        ("__ge__", tuple_dunder_ge),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
    // tupleobject.py descr_getnewargs — `((self-copy),)`
    dict_storage_store(
        ns,
        "__getnewargs__",
        make_builtin_function_with_arity(
            "__getnewargs__",
            |args| {
                let items = unsafe { pyre_object::w_tuple_items_copy_as_vec(args[0]) };
                Ok(pyre_object::w_tuple_new(vec![pyre_object::w_tuple_new(
                    items,
                )]))
            },
            1,
        ),
    );
}

/// `tupleobject.c` `tuple * n` / `n * tuple`.  A non-integer count
/// raises the `__index__` TypeError.
fn tuple_descr_mul(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_slot(args, 1)?;
    if unsafe { pyre_object::pyobject::is_int_or_long(args[1]) } {
        crate::objspace::descroperation::mul(args[0], args[1])
    } else {
        // NotImplemented lets the `*` operator try a reflected `__rmul__`
        // and otherwise emit the "can't multiply sequence by non-int"
        // message, instead of this method's own slot error.
        Ok(pyre_object::w_not_implemented())
    }
}

// ── Int/Float/Bool TypeDef (minimal) ─────────────────────────────────

// ── Type TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/typeobject.py TypeDef("type", ...)

/// types.UnionType — PyPy: _pypy_generic_alias.py UnionType
/// sliceobject.py:148 `W_SliceObject.descr_indices`.
fn slice_method_indices(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "indices() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let length = crate::builtins::getindex_w(args[1])?;
    if length < 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "length should not be negative".to_string(),
        ));
    }
    let (start, stop, step) = unsafe {
        crate::sliceobject::indices3(
            pyre_object::sliceobject::w_slice_get_start(args[0]),
            pyre_object::sliceobject::w_slice_get_stop(args[0]),
            pyre_object::sliceobject::w_slice_get_step(args[0]),
            length,
        )?
    };
    Ok(w_tuple_new(vec![
        pyre_object::w_int_new(start),
        pyre_object::w_int_new(stop),
        pyre_object::w_int_new(step),
    ]))
}

/// sliceobject.py `W_SliceObject.descr__new__` — `slice([start,] stop[, step])`.
fn slice_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (params, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    if crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "slice() takes no keyword arguments",
        ));
    }
    let none = pyre_object::w_none();
    let (start, stop, step) = match params {
        [stop] => (none, *stop, none),
        [start, stop] => (*start, *stop, none),
        [start, stop, step] => (*start, *stop, *step),
        [] => {
            return Err(crate::PyError::type_error(
                "slice expected at least 1 argument, got 0",
            ));
        }
        _ => {
            return Err(crate::PyError::type_error(format!(
                "slice expected at most 3 arguments, got {}",
                params.len()
            )));
        }
    };
    Ok(pyre_object::sliceobject::w_slice_new(start, stop, step))
}

fn slice_getter(
    args: &[PyObjectRef],
    field: unsafe fn(PyObjectRef) -> PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    // sliceobject.py:191 `slicewprop.fget` — applied to a non-slice
    // receiver raises TypeError("descriptor is for 'slice'").
    if unsafe { pyre_object::sliceobject::is_slice(self_) } {
        Ok(unsafe { field(self_) })
    } else {
        Err(crate::PyError::type_error("descriptor is for 'slice'"))
    }
}

/// sliceobject.py `descr_repr` — `"slice(%r, %r, %r)"`.
fn slice_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_str_new(&unsafe { crate::display::py_repr(args[0])? }))
}

/// sliceobject.py `descr_eq` / `descr_ne` — compare the three components.
/// `slice is slice` is always equal even with non-comparable params.
fn slice_components_eq(a: PyObjectRef, b: PyObjectRef) -> Result<bool, crate::PyError> {
    unsafe {
        Ok(crate::baseobjspace::eq_w(
            pyre_object::sliceobject::w_slice_get_start(a),
            pyre_object::sliceobject::w_slice_get_start(b),
        )? && crate::baseobjspace::eq_w(
            pyre_object::sliceobject::w_slice_get_stop(a),
            pyre_object::sliceobject::w_slice_get_stop(b),
        )? && crate::baseobjspace::eq_w(
            pyre_object::sliceobject::w_slice_get_step(a),
            pyre_object::sliceobject::w_slice_get_step(b),
        )?)
    }
}

fn slice_descr_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (a, b) = (args[0], args[1]);
    if a == b {
        return Ok(pyre_object::w_bool_from(true));
    }
    if unsafe { pyre_object::sliceobject::is_slice(b) } {
        Ok(pyre_object::w_bool_from(slice_components_eq(a, b)?))
    } else {
        Ok(pyre_object::w_not_implemented())
    }
}

fn slice_descr_ne(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (a, b) = (args[0], args[1]);
    if a == b {
        return Ok(pyre_object::w_bool_from(false));
    }
    if unsafe { pyre_object::sliceobject::is_slice(b) } {
        if slice_components_eq(a, b)? {
            Ok(pyre_object::w_bool_from(false))
        } else {
            Ok(pyre_object::w_bool_from(true))
        }
    } else {
        Ok(pyre_object::w_not_implemented())
    }
}

/// sliceobject.py `descr_lt` — lexicographic on (start, stop, step).
fn slice_descr_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (a, b) = (args[0], args[1]);
    if a == b {
        return Ok(pyre_object::w_bool_from(false));
    }
    if unsafe { pyre_object::sliceobject::is_slice(b) } {
        slice_lt_components(a, b)
    } else {
        Ok(pyre_object::w_not_implemented())
    }
}

fn slice_lt_components(a: PyObjectRef, b: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let (sa, sb) = unsafe {
        (
            pyre_object::sliceobject::w_slice_get_start(a),
            pyre_object::sliceobject::w_slice_get_start(b),
        )
    };
    if crate::baseobjspace::eq_w(sa, sb)? {
        let (ta, tb) = unsafe {
            (
                pyre_object::sliceobject::w_slice_get_stop(a),
                pyre_object::sliceobject::w_slice_get_stop(b),
            )
        };
        if crate::baseobjspace::eq_w(ta, tb)? {
            let (pa, pb) = unsafe {
                (
                    pyre_object::sliceobject::w_slice_get_step(a),
                    pyre_object::sliceobject::w_slice_get_step(b),
                )
            };
            crate::baseobjspace::compare(pa, pb, crate::baseobjspace::CompareOp::Lt)
        } else {
            crate::baseobjspace::compare(ta, tb, crate::baseobjspace::CompareOp::Lt)
        }
    } else {
        crate::baseobjspace::compare(sa, sb, crate::baseobjspace::CompareOp::Lt)
    }
}

/// sliceobject.py `descr__reduce__` — `(type(self), (start, stop, step))`.
fn slice_descr_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let s = args[0];
    let ty = r#type(s).unwrap_or(pyre_object::PY_NULL);
    let components = unsafe {
        w_tuple_new(vec![
            pyre_object::sliceobject::w_slice_get_start(s),
            pyre_object::sliceobject::w_slice_get_stop(s),
            pyre_object::sliceobject::w_slice_get_step(s),
        ])
    };
    Ok(w_tuple_new(vec![ty, components]))
}

fn init_slice_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(slice_descr_new));
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function("__repr__", slice_descr_repr),
    );
    dict_storage_store(
        ns,
        "__eq__",
        make_builtin_function("__eq__", slice_descr_eq),
    );
    dict_storage_store(
        ns,
        "__ne__",
        make_builtin_function("__ne__", slice_descr_ne),
    );
    dict_storage_store(
        ns,
        "__lt__",
        make_builtin_function("__lt__", slice_descr_lt),
    );
    // sliceobject.py:205 `__hash__ = None` — slice is unhashable, consistent
    // with the value-based `__eq__`.  hash() raises via the unhashable
    // ladder in `builtins::try_hash_value`; the dict entry surfaces
    // `slice.__hash__ is None` to introspection.
    dict_storage_store(ns, "__hash__", pyre_object::w_none());
    dict_storage_store(
        ns,
        "__reduce__",
        make_builtin_function("__reduce__", slice_descr_reduce),
    );
    dict_storage_store(
        ns,
        "start",
        make_getset_descriptor_named(
            make_builtin_function_with_arity(
                "start",
                |args| slice_getter(args, pyre_object::sliceobject::w_slice_get_start),
                2,
            ),
            "start",
        ),
    );
    dict_storage_store(
        ns,
        "stop",
        make_getset_descriptor_named(
            make_builtin_function_with_arity(
                "stop",
                |args| slice_getter(args, pyre_object::sliceobject::w_slice_get_stop),
                2,
            ),
            "stop",
        ),
    );
    dict_storage_store(
        ns,
        "step",
        make_getset_descriptor_named(
            make_builtin_function_with_arity(
                "step",
                |args| slice_getter(args, pyre_object::sliceobject::w_slice_get_step),
                2,
            ),
            "step",
        ),
    );
    dict_storage_store(
        ns,
        "indices",
        make_builtin_function("indices", slice_method_indices),
    );
}

/// `UnionType.__getitem__` (`_pypy_generic_alias.py:312`) — substitute the
/// free parameters with `items`, then fold the substituted members back into
/// a union with `|`.
fn union_getitem(args: &[PyObjectRef]) -> crate::PyResult {
    let self_ = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if !unsafe { pyre_object::is_union(self_) } {
        return Err(crate::PyError::type_error(
            "descriptor '__getitem__' requires a 'types.UnionType' object",
        ));
    }
    let items_raw = args.get(1).copied().unwrap_or_else(pyre_object::w_none);
    let items = if unsafe { pyre_object::is_tuple(items_raw) } {
        items_raw
    } else {
        pyre_object::w_tuple_new(vec![items_raw])
    };
    let params = unsafe { pyre_object::w_union_get_parameters(self_) };
    let union_args = unsafe { pyre_object::w_union_get_args(self_) };
    let newargs = crate::_pypy_generic_alias::subs_parameters(self_, union_args, params, items)?;
    if newargs.is_empty() {
        // `if len(newargs) == 0: return UnionType(())` — unreachable for a
        // real union (always ≥1 member), kept for parity.
        return Ok(pyre_object::w_union_from_members(
            Vec::new(),
            pyre_object::w_tuple_new(vec![]),
        ));
    }
    // `curr = newargs[0]; for i in range(1, ...): curr |= newargs[i]`.
    let mut curr = newargs[0];
    for &next in &newargs[1..] {
        curr = crate::objspace::descroperation::or_(curr, next)?;
    }
    Ok(curr)
}

/// `UnionType.__class_getitem__(items)` — `typing.Union` is bound to this
/// type, so `Union[int, str]` folds the members back into `int | str`.  A
/// single member is returned unwrapped (`Union[int]` is `int`).
fn union_class_getitem(args: &[PyObjectRef]) -> crate::PyResult {
    // args[0] = cls (UnionType), args[1] = items.
    let items_raw = args.get(1).copied().unwrap_or_else(pyre_object::w_none);
    let items: Vec<PyObjectRef> = if unsafe { pyre_object::is_tuple(items_raw) } {
        let len = unsafe { pyre_object::w_tuple_len(items_raw) };
        (0..len)
            .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(items_raw, i as i64) })
            .collect()
    } else {
        vec![items_raw]
    };
    if items.is_empty() {
        return Err(crate::PyError::type_error(
            "Cannot take a Union of no types.",
        ));
    }
    let mut curr = items[0];
    for &next in &items[1..] {
        curr = crate::_pypy_generic_alias::create_union(curr, next)?;
    }
    Ok(curr)
}

fn init_union_type(ns: &mut DictStorage) {
    // UnionType.__args__ — returns the tuple of union member types
    let args_getter = make_builtin_function_with_arity(
        "__args__",
        |args| {
            let self_ = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if unsafe { pyre_object::is_union(self_) } {
                Ok(unsafe { pyre_object::w_union_get_args(self_) })
            } else {
                Ok(pyre_object::PY_NULL)
            }
        },
        2,
    );
    dict_storage_store(ns, "__args__", make_getset_descriptor(args_getter));
    // UnionType.__parameters__ — the slot stored at construction from the raw
    // constructor operands (`_pypy_generic_alias.py:264`
    // `self.__parameters__ = _collect_parameters(args)`).
    let params_getter = make_builtin_function_with_arity(
        "__parameters__",
        |args| {
            let self_ = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if unsafe { pyre_object::is_union(self_) } {
                Ok(unsafe { pyre_object::w_union_get_parameters(self_) })
            } else {
                Ok(pyre_object::PY_NULL)
            }
        },
        2,
    );
    dict_storage_store(ns, "__parameters__", make_getset_descriptor(params_getter));
    // UnionType.__getitem__ (`_pypy_generic_alias.py:312`) — substitute the
    // free parameters with `items`, then fold the results back into a union
    // with `|`.
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function("__getitem__", union_getitem),
    );
    // UnionType.__class_getitem__ — `typing.Union` is this type, so
    // `Union[int, str]` folds members into a union.
    dict_storage_store(
        ns,
        "__class_getitem__",
        pyre_object::function::w_classmethod_new(make_builtin_function(
            "__class_getitem__",
            union_class_getitem,
        )),
    );
    // UnionType.__or__ — PyPy: UnionType.__or__ → _create_union
    dict_storage_store(
        ns,
        "__or__",
        make_builtin_function_with_arity(
            "__or__",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("__or__ requires 2 arguments"));
                }
                crate::_pypy_generic_alias::create_union(args[0], args[1])
            },
            2,
        ),
    );
    // UnionType.__ror__
    dict_storage_store(
        ns,
        "__ror__",
        make_builtin_function_with_arity(
            "__ror__",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("__ror__ requires 2 arguments"));
                }
                crate::_pypy_generic_alias::create_union(args[1], args[0])
            },
            2,
        ),
    );
    // UnionType.__eq__ — `set(self.__args__) == set(other.__args__)`
    // (`_pypy_generic_alias.py:270`).
    dict_storage_store(
        ns,
        "__eq__",
        make_builtin_function_with_arity(
            "__eq__",
            |args| {
                let self_ = args[0];
                let other = args[1];
                if !unsafe { pyre_object::is_union(other) } {
                    return Ok(pyre_object::w_not_implemented());
                }
                Ok(pyre_object::w_bool_from(
                    crate::_pypy_generic_alias::union_set_eq(self_, other)?,
                ))
            },
            2,
        ),
    );
}

thread_local! {
    static GETSET_DESCRIPTOR_TYPE: std::cell::OnceCell<pyre_object::PyObjectRef>
        = const { std::cell::OnceCell::new() };
}

fn getset_descriptor_type() -> pyre_object::PyObjectRef {
    GETSET_DESCRIPTOR_TYPE.with(|cell| {
        *cell.get_or_init(|| {
            // `typedef.py:444 GetSetProperty.typedef = TypeDef(
            // "getset_descriptor", ...)`.  Pyre owns the static
            // `GETSET_DESCRIPTOR_TYPE` PyType so GetSetProperty
            // instances carry it as `ob_type` (not the catch-all
            // `INSTANCE_TYPE`).  `make_builtin_type_with_layout`
            // wires the layout so `setup_builtin_type` records the
            // explicit typedef per `typeobject.py:1273-1280`.
            let tp = make_builtin_type_with_layout(
                "getset_descriptor",
                init_getset_descriptor_type,
                w_object(),
                &pyre_object::typedef::GETSET_DESCRIPTOR_TYPE as *const PyType,
            );
            // typedef.py:446 assert not GetSetProperty.typedef.acceptable_as_base_class
            unsafe { pyre_object::w_type_set_acceptable_as_base_class(tp, false) };
            // `init_typeobjects` would normally hand the W_TypeObject
            // to `set_instantiate(pytype, w_typeobject)` so allocators
            // can stamp `ob_header.w_class` at construction time
            // (see typedef.rs around `for (pytype, w_type) in reg`).
            // `getset_descriptor_type()` is called from inside the
            // init loop *as* a builder for descriptors that other
            // typedefs install, so the post-loop `set_instantiate`
            // pass can race the first GetSetProperty alloc.
            // Setting it eagerly here keeps `w_class` non-null for
            // every descriptor regardless of allocation order.
            pyre_object::pyobject::set_instantiate(
                &pyre_object::typedef::GETSET_DESCRIPTOR_TYPE,
                tp,
            );
            tp
        })
    })
}

/// typedef.py:378-382 readonly_attribute
///
/// ```python
/// def readonly_attribute(self, space):   # overwritten in cpyext
///     if self.name == '<generic property>':
///         raise oefmt(space.w_AttributeError, "readonly attribute")
///     else:
///         raise oefmt(space.w_AttributeError, "readonly attribute '%s'", self.name)
/// ```
///
/// PyPy raises `AttributeError`, not `TypeError`; the message keeps
/// the descriptor's `name` so `e.args[0]` matches CPython /
/// inspect.py expectations.
fn readonly_attribute(descr: pyre_object::PyObjectRef) -> crate::PyError {
    let name_obj = read_descr_name(descr);
    let name = if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
        Some(unsafe { pyre_object::w_str_get_value(name_obj) })
    } else {
        None
    };
    match name {
        Some(n) if n != "<generic property>" => {
            crate::PyError::attribute_error(format!("readonly attribute '{}'", n))
        }
        _ => crate::PyError::attribute_error("readonly attribute".to_string()),
    }
}

/// typedef.py:308-415 GetSetProperty.typedef = TypeDef("getset_descriptor", ...)
fn init_getset_descriptor_type(ns: &mut DictStorage) {
    // typedef.py:347-365 GetSetProperty.descr_property_get
    //
    // ```python
    // @unwrap_spec(w_cls = WrappedDefault(None))
    // def descr_property_get(self, space, w_obj, w_cls=None):
    //     """property.__get__(obj[, type]) -> value
    //     Read the value of the property of the given obj."""
    //     # XXX HAAAAAAAAAAAACK (but possibly a good one)
    //     if (space.is_w(w_obj, space.w_None)
    //         and not space.is_w(w_cls, space.type(space.w_None))):
    //         #print self, w_obj, w_cls
    //         if space.is_w(w_cls, space.w_None):
    //             raise oefmt(space.w_TypeError, "__get__(None, None) is invalid")
    //         return self
    //     else:
    //         try:
    //             return self.fget(self, space, w_obj)
    //         except DescrMismatch:
    //             return w_obj.descr_call_mismatch(
    //                 space, '__getattribute__',
    //                 self.reqcls, Arguments(space, [w_obj,
    //                                                space.newtext(self.name)]))
    // ```
    dict_storage_store(
        ns,
        "__get__",
        make_builtin_function("__get__", |args| {
            let w_self = args[0];
            let w_obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let w_cls = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
            let w_obj_is_none = !w_obj.is_null() && unsafe { pyre_object::is_none(w_obj) };
            let none_type =
                crate::typedef::r#type(pyre_object::w_none()).unwrap_or(pyre_object::PY_NULL);
            let w_cls_is_none_type = !w_cls.is_null() && std::ptr::eq(w_cls, none_type);
            // typedef.py:352-353 if w_obj is None and w_cls is not type(None):
            if w_obj_is_none && !w_cls_is_none_type {
                // typedef.py:355 if w_cls is None: raise TypeError
                if w_cls.is_null() || unsafe { pyre_object::is_none(w_cls) } {
                    return Err(crate::PyError::type_error(
                        "__get__(None, None) is invalid".to_string(),
                    ));
                }
                // typedef.py:357 return self
                return Ok(w_self);
            }
            // typedef.py:359-365 try: return self.fget(self, space, w_obj)
            //                    except DescrMismatch: descr_call_mismatch(...)
            let reqcls = read_reqcls(w_self);
            // pyre's typecheck wrapper equivalent: descr_self_interp_w runs
            // before the inner function so DescrMismatch is raised the same
            // way PyPy's `_make_descr_typecheck_wrapper` does.
            if !reqcls.is_null() {
                if let Err(e) = crate::baseobjspace::descr_self_interp_w(reqcls, w_obj) {
                    if e.kind == crate::PyErrorKind::DescrMismatch {
                        return Err(crate::baseobjspace::descr_call_mismatch(
                            w_obj,
                            "__getattribute__",
                            reqcls,
                        ));
                    }
                    return Err(e);
                }
            }
            let fget = read_fget(w_self);
            if fget.is_null() {
                return Err(readonly_attribute(w_self));
            }
            match crate::call::call_function_impl_result(fget, &[w_self, w_obj]) {
                Ok(v) => Ok(v),
                Err(e) if e.kind == crate::PyErrorKind::DescrMismatch => Err(
                    crate::baseobjspace::descr_call_mismatch(w_obj, "__getattribute__", reqcls),
                ),
                Err(e) => Err(e),
            }
        }),
    );
    // typedef.py:373-386 GetSetProperty.descr_property_set
    //
    // ```python
    // def descr_property_set(self, space, w_obj, w_value):
    //     fset = self.fset
    //     if fset is None:
    //         raise self.readonly_attribute(space)
    //     try:
    //         fset(self, space, w_obj, w_value)
    //     except DescrMismatch:
    //         w_obj.descr_call_mismatch(
    //             space, '__setattr__',
    //             self.reqcls, Arguments(space, [w_obj,
    //                                            space.newtext(self.name),
    //                                            w_value]))
    // ```
    dict_storage_store(
        ns,
        "__set__",
        make_builtin_function_with_arity(
            "__set__",
            |args| {
                let w_self = args[0];
                let w_obj = args[1];
                let w_value = args[2];
                let fset = read_fset(w_self);
                if fset.is_null() || unsafe { pyre_object::is_none(fset) } {
                    return Err(readonly_attribute(w_self));
                }
                let reqcls = read_reqcls(w_self);
                if !reqcls.is_null() {
                    if let Err(e) = crate::baseobjspace::descr_self_interp_w(reqcls, w_obj) {
                        if e.kind == crate::PyErrorKind::DescrMismatch {
                            return Err(crate::baseobjspace::descr_call_mismatch(
                                w_obj,
                                "__setattr__",
                                reqcls,
                            ));
                        }
                        return Err(e);
                    }
                }
                match crate::call::call_function_impl_result(fset, &[w_self, w_obj, w_value]) {
                    Ok(_) => Ok(pyre_object::w_none()),
                    Err(e) if e.kind == crate::PyErrorKind::DescrMismatch => Err(
                        crate::baseobjspace::descr_call_mismatch(w_obj, "__setattr__", reqcls),
                    ),
                    Err(e) => Err(e),
                }
            },
            3,
        ),
    );
    // typedef.py:388-400 GetSetProperty.descr_property_del
    //
    // ```python
    // def descr_property_del(self, space, w_obj):
    //     fdel = self.fdel
    //     if fdel is None:
    //         raise oefmt(space.w_AttributeError, "cannot delete attribute")
    //     try:
    //         fdel(self, space, w_obj)
    //     except DescrMismatch:
    //         w_obj.descr_call_mismatch(
    //             space, '__delattr__',
    //             self.reqcls, Arguments(space, [w_obj,
    //                                            space.newtext(self.name)]))
    // ```
    dict_storage_store(
        ns,
        "__delete__",
        make_builtin_function_with_arity(
            "__delete__",
            |args| {
                let w_self = args[0];
                let w_obj = args[1];
                let fdel = read_fdel(w_self);
                if fdel.is_null() || unsafe { pyre_object::is_none(fdel) } {
                    // typedef.py:404-405:
                    //   raise oefmt(space.w_AttributeError,
                    //       "cannot delete '%s' attribute of immutable type '%N'",
                    //       self.name, w_obj)
                    let name_obj = read_descr_name(w_self);
                    let name = if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
                        unsafe { pyre_object::w_str_get_value(name_obj) }
                    } else {
                        "<generic property>"
                    };
                    let type_name = unsafe {
                        match crate::typedef::r#type(w_obj) {
                            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
                            None => (*(*w_obj).ob_type).name.to_string(),
                        }
                    };
                    return Err(crate::PyError::attribute_error(format!(
                        "cannot delete '{name}' attribute of immutable type '{type_name}'"
                    )));
                }
                let reqcls = read_reqcls(w_self);
                if !reqcls.is_null() {
                    if let Err(e) = crate::baseobjspace::descr_self_interp_w(reqcls, w_obj) {
                        if e.kind == crate::PyErrorKind::DescrMismatch {
                            return Err(crate::baseobjspace::descr_call_mismatch(
                                w_obj,
                                "__delattr__",
                                reqcls,
                            ));
                        }
                        return Err(e);
                    }
                }
                match crate::call::call_function_impl_result(fdel, &[w_self, w_obj]) {
                    Ok(_) => Ok(pyre_object::w_none()),
                    Err(e) if e.kind == crate::PyErrorKind::DescrMismatch => Err(
                        crate::baseobjspace::descr_call_mismatch(w_obj, "__delattr__", reqcls),
                    ),
                    Err(e) => Err(e),
                }
            },
            2,
        ),
    );
    // The four metadata getsets (typedef.py:470-473
    // __name__/__qualname__/__objclass__/__doc__) cannot be
    // installed inside this function — each one allocates a fresh
    // `GetSetProperty` via `make_getset_descriptor`, which
    // funnels through `getset_descriptor_type()`'s OnceCell, and we
    // are currently *inside* that OnceCell's init closure.
    // Re-entering `OnceCell::get_or_init` is undefined behaviour
    // (the cell is already mutably borrowed), so the post-init
    // helper `patch_getset_descriptor_metadata` stamps them after
    // the OnceCell finishes, mirroring how
    // `patch_builtin_function_descriptors` patches the
    // BuiltinFunction `reqcls` slot.
}

/// typedef.py:465-474 metadata getsets on `GetSetProperty.typedef`,
/// installed in a post-init pass per the comment above
/// `init_getset_descriptor_type`.
///
/// ```python
/// __name__ = interp_attrproperty('name', cls=GetSetProperty,
///                                 wrapfn="newtext_or_none"),
/// __qualname__ = GetSetProperty(GetSetProperty.descr_get_qualname),
/// __objclass__ = GetSetProperty(GetSetProperty.descr_get_objclass),
/// __doc__ = interp_attrproperty('doc', cls=GetSetProperty,
///                                wrapfn="newtext_or_none"),
/// ```
fn patch_getset_descriptor_metadata() {
    let tp = getset_descriptor_type();
    if tp.is_null() {
        return;
    }
    let dict_ptr = unsafe { pyre_object::w_type_get_dict_ptr(tp) } as *mut DictStorage;
    if dict_ptr.is_null() {
        return;
    }
    let ns = unsafe { &mut *dict_ptr };
    // typedef.py:470 __name__
    dict_storage_store(
        ns,
        "__name__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__name__",
            |args| {
                let descr = args[1];
                if descr.is_null() {
                    return Ok(pyre_object::w_none());
                }
                let name = unsafe { pyre_object::typedef::w_getset_get_name(descr) };
                if name.is_null() {
                    return Ok(pyre_object::w_none());
                }
                Ok(name)
            },
            2,
        )),
    );
    // typedef.py:471 __qualname__ = GetSetProperty(descr_get_qualname)
    //
    // ```python
    // def descr_get_qualname(self, space):
    //     if self.w_qualname is None:
    //         self.w_qualname = self._calculate_qualname(space)
    //     return self.w_qualname
    //
    // def _calculate_qualname(self, space):
    //     if self.reqcls is None:
    //         type_qualname = '?'
    //     else:
    //         w_type = space.gettypeobject(self.reqcls.typedef)
    //         type_qualname = space.text_w(
    //             space.getattr(w_type, space.newtext('__qualname__')))
    //     qualname = "%s.%s" % (type_qualname, self.name)
    //     return space.newtext(qualname)
    // ```
    dict_storage_store(
        ns,
        "__qualname__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__qualname__",
            |args| {
                let descr = args[1];
                if descr.is_null() {
                    return Ok(pyre_object::w_none());
                }
                unsafe {
                    let cached = pyre_object::typedef::w_getset_get_qualname(descr);
                    if !cached.is_null() {
                        return Ok(cached);
                    }
                    // typedef.py:425-432 _calculate_qualname:
                    //   if self.reqcls is None: type_qualname = '?'
                    //   else:
                    //       w_type = space.gettypeobject(self.reqcls.typedef)
                    //       type_qualname = space.text_w(
                    //           space.getattr(w_type, space.newtext('__qualname__')))
                    //
                    // PyPy reads the bound class's `__qualname__`
                    // (which respects nested-class scoping and any
                    // explicit `__qualname__` assignment in the class
                    // body), NOT the bare `__name__`.  Pyre's
                    // `getattr(w_type, '__qualname__')` resolves
                    // through the type-side __qualname__ getset that
                    // already mirrors PyPy's lookup-then-fallback
                    // chain (`baseobjspace.rs:4004-4009`).
                    let reqcls = pyre_object::typedef::w_getset_get_reqcls(descr);
                    let type_qualname = if reqcls.is_null() {
                        "?".to_string()
                    } else {
                        match crate::baseobjspace::getattr_str(reqcls, "__qualname__") {
                            Ok(qn) if pyre_object::is_str(qn) => {
                                pyre_object::w_str_get_value(qn).to_string()
                            }
                            // PyPy raises through here on AttributeError;
                            // pyre falls back to the bare type name to
                            // avoid surfacing an unrelated AttributeError
                            // when introspecting `descr.__qualname__`.
                            _ => pyre_object::w_type_get_name(reqcls).to_string(),
                        }
                    };
                    let name_obj = pyre_object::typedef::w_getset_get_name(descr);
                    let name = if !name_obj.is_null() && pyre_object::is_str(name_obj) {
                        pyre_object::w_str_get_value(name_obj).to_string()
                    } else {
                        "<generic property>".to_string()
                    };
                    let combined = pyre_object::w_str_new(&format!("{type_qualname}.{name}"));
                    pyre_object::typedef::w_getset_set_qualname(descr, combined);
                    Ok(combined)
                }
            },
            2,
        )),
    );
    // typedef.py:472 __objclass__ = GetSetProperty(descr_get_objclass)
    //
    // ```python
    // def descr_get_objclass(self, space):
    //     if self.w_objclass is not None:
    //         return self.w_objclass
    //     if self.reqcls is not None:
    //         return space.gettypeobject(self.reqcls.typedef)
    //     raise oefmt(space.w_AttributeError,
    //                 "generic self has no __objclass__")
    // ```
    dict_storage_store(
        ns,
        "__objclass__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__objclass__",
            |args| {
                let descr = args[1];
                if descr.is_null() {
                    return Err(crate::PyError::attribute_error(
                        "generic self has no __objclass__",
                    ));
                }
                unsafe {
                    let w_objclass = pyre_object::typedef::w_getset_get_objclass(descr);
                    if !w_objclass.is_null() {
                        return Ok(w_objclass);
                    }
                    let reqcls = pyre_object::typedef::w_getset_get_reqcls(descr);
                    if !reqcls.is_null() {
                        return Ok(reqcls);
                    }
                    Err(crate::PyError::attribute_error(
                        "generic self has no __objclass__",
                    ))
                }
            },
            2,
        )),
    );
    // typedef.py:473 __doc__ = interp_attrproperty('doc', ...)
    dict_storage_store(
        ns,
        "__doc__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__doc__",
            |args| {
                let descr = args[1];
                if descr.is_null() {
                    return Ok(pyre_object::w_none());
                }
                let doc = unsafe { pyre_object::typedef::w_getset_get_doc(descr) };
                if doc.is_null() {
                    return Ok(pyre_object::w_none());
                }
                Ok(doc)
            },
            2,
        )),
    );
}

/// `GetSetProperty(fget)` — read-only getset descriptor with no required class.
///
/// PyPy: `GetSetProperty(fget)` (typedef.py:312-325).
fn make_getset_descriptor(getter: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    make_getset_property_full(
        getter,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        None,
    )
}

/// `GetSetProperty(fget)` with an explicit `name`.  Mirrors
/// `typedef.py:58 add_entries` which stamps the dict-key as the
/// descriptor's `name` (so `dict_descr.__name__` is `"__dict__"`,
/// `weakref_descr.__name__` is `"__weakref__"`, etc.) — without this
/// pyre's descriptors would all surface as `"<generic property>"`.
pub(crate) fn make_getset_descriptor_named(
    getter: pyre_object::PyObjectRef,
    name: &str,
) -> pyre_object::PyObjectRef {
    make_getset_property_full(
        getter,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        Some(name),
    )
}

/// `GetSetProperty(fget, fset, fdel)` — full getset descriptor with no
/// required class. Equivalent to PyPy's `GetSetProperty(fget, fset, fdel)`
/// invocation with `cls=None`.
fn make_getset_property(
    fget: pyre_object::PyObjectRef,
    fset: pyre_object::PyObjectRef,
    fdel: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    make_getset_property_full(fget, fset, fdel, pyre_object::PY_NULL, None)
}

/// `GetSetProperty(fget, fset, fdel)` with explicit `name` — see
/// `make_getset_descriptor_named` for the typedef.py:58 motivation.
fn make_getset_property_named(
    fget: pyre_object::PyObjectRef,
    fset: pyre_object::PyObjectRef,
    fdel: pyre_object::PyObjectRef,
    name: &str,
) -> pyre_object::PyObjectRef {
    make_getset_property_full(fget, fset, fdel, pyre_object::PY_NULL, Some(name))
}

/// `GetSetProperty(fget, fset, fdel, cls=cls)` — full getset descriptor
/// with a required class for descriptor-level type enforcement.
///
/// PyPy: `GetSetProperty(...)` (typedef.py:312-325) with the `cls` keyword.
/// `cls` is stored as `reqcls` and `descr_self_interp_w` raises
/// `DescrMismatch` when a wrong-class instance reaches `__get__/__set__/__delete__`.
fn make_getset_property_full(
    fget: pyre_object::PyObjectRef,
    fset: pyre_object::PyObjectRef,
    fdel: pyre_object::PyObjectRef,
    cls: pyre_object::PyObjectRef,
    name: Option<&str>,
) -> pyre_object::PyObjectRef {
    // Force `getset_descriptor_type` registration so the static
    // PyType's `instantiate` slot points at the W_TypeObject before
    // the first allocation reads it.  Returns the (cached)
    // PyObjectRef back; the W_TypeObject side is not used for the
    // alloc itself — the static `GETSET_DESCRIPTOR_TYPE` PyType is.
    let _ = getset_descriptor_type();
    // typedef.py:346 `self.name = name if name is not None else
    // '<generic property>'` — pyre stamps the literal sentinel when
    // no explicit name is supplied, so `make_getset_descriptor` keeps
    // the PyPy-default sentinel for callers that don't override it.
    let resolved_name = match name {
        Some(n) => pyre_object::w_str_new(n),
        None => pyre_object::w_str_new("<generic property>"),
    };
    pyre_object::typedef::w_getset_property_new(
        fget,
        fset,
        fdel,
        pyre_object::PY_NULL, // doc
        cls,
        false, // use_closure
        resolved_name,
    )
}

fn init_type_type(ns: &mut DictStorage) {
    // type.__new__(metatype, name, bases, dict) — creates new type
    dict_storage_store(
        ns,
        "__new__",
        make_new_descr(crate::builtins::type_descr_new),
    );
    // `type[int]` builds a GenericAlias, but `type` carries no
    // `__class_getitem__` in its dict — `descroperation.getitem` special-cases
    // `is_w(w_obj, w_type)` (`descroperation.py:362`).  The wiring lives in
    // `baseobjspace::getitem_type`, so `hasattr(type, "__class_getitem__")`
    // stays False to match.
    // type.__init__ — no-op for now
    dict_storage_store(
        ns,
        "__init__",
        make_builtin_function("__init__", |_| Ok(pyre_object::w_none())),
    );
    // type.__call__(cls, *args) — typeobject.c type_call.  The implicit
    // instantiation path handles `Cls()` directly, but a custom metaclass
    // whose `__call__` delegates via `super().__call__(...)` needs this
    // entry to resolve to the default __new__/__init__ behaviour.
    dict_storage_store(
        ns,
        "__call__",
        make_builtin_function("__call__", |args| {
            let Some((&cls, rest)) = args.split_first() else {
                return Err(crate::PyError::type_error(
                    "type.__call__() takes at least 1 argument (0 given)",
                ));
            };
            crate::call::type_call_instantiate(cls, rest)
        }),
    );
    // type.__annotations__ / __dict__ / __mro__ / __name__ / __bases__
    // are exposed as getset descriptors so
    // `type.__dict__['<name>'].__get__(cls)` invokes the underlying getter
    // and returns the real value (matching CPython's getset_descriptor).
    //
    // PyPy: pypy/objspace/std/typeobject.py get_annotations / descr_getdict
    // / descr_getmro / descr_getname / descr_getbases.
    // GetSetProperty fget callbacks receive (descriptor_self, w_obj) so the
    // wrapped object is at args[1] (matches PyPy's typecheck wrapper that
    // passes (closure, space, w_obj)).
    let annotations_getter = make_builtin_function_with_arity(
        "__annotations__",
        |args| {
            // GetSetProperty fget callbacks receive (descriptor_self, w_obj),
            // so the cls is at args[1].
            let cls = args[1];
            // PEP 649 path: bytecode emits `__annotate_func__` (== `__annotate__`).
            // Call it with format=1 (VALUE) to materialise the dict.
            if let Some(annotate_fn) =
                unsafe { crate::baseobjspace::lookup_in_type(cls, "__annotate_func__") }
                    .or_else(|| unsafe { crate::baseobjspace::lookup_in_type(cls, "__annotate__") })
            {
                if !annotate_fn.is_null() && !unsafe { pyre_object::is_none(annotate_fn) } {
                    return Ok(crate::call::call_function_impl_raw(
                        annotate_fn,
                        &[pyre_object::w_int_new(1)],
                    ));
                }
            }
            Ok(pyre_object::w_dict_new())
        },
        2,
    );
    dict_storage_store(
        ns,
        "__annotations__",
        make_getset_descriptor(annotations_getter),
    );

    let mro_getter = make_builtin_function_with_arity(
        "__mro__",
        |args| {
            let cls = args[1];
            unsafe {
                let mro_ptr = pyre_object::w_type_get_mro(cls);
                if mro_ptr.is_null() {
                    return Ok(pyre_object::w_tuple_new(vec![]));
                }
                Ok(pyre_object::w_tuple_new((*mro_ptr).clone()))
            }
        },
        2,
    );
    dict_storage_store(ns, "__mro__", make_getset_descriptor(mro_getter));

    // typeobject.py:1237 descr__flags — the `tp_flags` bitmask.
    let flags_getter = make_builtin_function_with_arity(
        "__flags__",
        |args| {
            Ok(pyre_object::w_int_new(unsafe {
                pyre_object::w_type_get_flags(args[1])
            }))
        },
        2,
    );
    dict_storage_store(ns, "__flags__", make_getset_descriptor(flags_getter));

    // `type.mro(cls)` — typeobject.c `mro_external` / `type.mro`: the method
    // form returns the MRO as a fresh list (the `__mro__` getset above
    // returns the tuple).  Bound as a regular method, so `cls` is at args[0].
    let mro_method = make_builtin_function("mro", |args| {
        let cls = args[0];
        unsafe {
            let mro_ptr = pyre_object::w_type_get_mro(cls);
            if mro_ptr.is_null() {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            Ok(pyre_object::w_list_new((*mro_ptr).clone()))
        }
    });
    dict_storage_store(ns, "mro", mro_method);

    // typeobject.py:1269-1272 descr___subclasses__ — return the list of
    // immediate subclasses recorded in `weak_subclasses` (dead weakrefs
    // filtered out by `w_type_get_subclasses`).
    let subclasses_method = make_builtin_function("__subclasses__", |args| {
        // `type.__subclasses__` resolves unbound when read off `type` itself
        // (it lives in `type`'s own dict, so it is not bound to a metatype
        // instance); calling it without the class argument is the
        // "unbound method ... needs an argument" TypeError, not a crash.
        let cls = match args.first() {
            Some(&c) if unsafe { pyre_object::is_type(c) } => c,
            _ => {
                return Err(crate::PyError::type_error(
                    "unbound method type.__subclasses__() needs an argument",
                ));
            }
        };
        let subs = unsafe { pyre_object::w_type_get_subclasses(cls) };
        Ok(pyre_object::w_list_new(subs))
    });
    dict_storage_store(ns, "__subclasses__", subclasses_method);

    // `pypy/objspace/std/typeobject.py:614-624 get_module` /
    // `:1241-1247 descr_get__module` / `descr_set__module`.
    // For heaptype (user-defined classes) the value is read from /
    // written to the class's `__dict__`; for builtin types getter
    // derives the module from the qualified name (everything before
    // the rightmost dot, default `"builtins"`).  PyPy's
    // `getdictvalue` returns the stored value verbatim — including
    // `None` — so the getter must NOT fall through to the dot-split
    // when the dict entry exists but happens to be None.
    let module_getter = make_builtin_function_with_arity(
        "__module__",
        |args| {
            let cls = args[1];
            // Reached as `type.__module__`: this getset lives on `type`'s own
            // dict, so the descriptor protocol binds it with a null instance.
            // There is no class to inspect, so use the builtin default that
            // the dot-split would produce for the unqualified name `type`.
            if cls.is_null() {
                return Ok(pyre_object::w_str_new("builtins"));
            }
            // `typeobject.py:614-617 get_module`:
            //     if self.is_heaptype():
            //         return self.getdictvalue(space, '__module__')
            // Only a heaptype reads `__module__` from its dict; a builtin
            // type derives it from the qualified name.  `lookup_in_type`
            // filters out null entries but preserves `w_none()`, matching
            // PyPy's "value present even if it's None" semantic.
            if unsafe { pyre_object::w_type_is_heaptype(cls) } {
                if let Some(v) = unsafe { crate::baseobjspace::lookup_in_type(cls, "__module__") } {
                    if !v.is_null() {
                        return Ok(v);
                    }
                }
            }
            // Builtin-name dot split fallback (`typeobject.py:619-624`).
            let name = unsafe { pyre_object::w_type_get_name(cls) };
            let mod_name = match name.rfind('.') {
                Some(dot) => name[..dot].to_string(),
                None => "builtins".to_string(),
            };
            Ok(pyre_object::w_str_new(&mod_name))
        },
        2,
    );
    let module_setter = make_builtin_function_with_arity(
        "__module__",
        |args| {
            // `typeobject.py:1245-1247`:
            //     def descr_set__module(space, w_type, w_value):
            //         w_type.setdictvalue(space, '__module__', w_value)
            // Writes directly into the type's namespace dict so
            // `A.__module__ = "x"` is reflected in `A.__dict__`.
            let cls = args[1];
            let value = args[2];
            unsafe {
                if pyre_object::is_type(cls) {
                    let dict_ptr = pyre_object::w_type_get_dict_ptr(cls) as *mut crate::DictStorage;
                    if !dict_ptr.is_null() {
                        crate::dict_storage_store(&mut *dict_ptr, "__module__", value);
                    }
                }
            }
            Ok(pyre_object::w_none())
        },
        3,
    );
    dict_storage_store(
        ns,
        "__module__",
        make_getset_property_named(
            module_getter,
            module_setter,
            pyre_object::PY_NULL,
            "__module__",
        ),
    );

    let dict_getter = make_builtin_function_with_arity(
        "__dict__",
        |args| {
            let cls = args[1];
            unsafe {
                let ns_ptr = pyre_object::typeobject::w_type_get_dict_ptr(cls);
                if ns_ptr.is_null() {
                    return Ok(pyre_object::w_dict_proxy_new(pyre_object::w_dict_new()));
                }
                // `pypy/objspace/std/typeobject.py:1277 descr_get_dict`
                // returns `W_DictProxyObject(w_dict)` — read-only **live**
                // view.  Wrap the type's canonical W_DictObject so
                // subsequent `cls.x = 1` setattrs flow through the
                // dict_storage_proxy and become visible on the proxy.
                // Instance flavor: a type's namespace is a regular
                // W_DictObject, not a module-strategy dict.
                let canonical = crate::baseobjspace::dict_storage_to_dict_kind(
                    ns_ptr as *const DictStorage,
                    crate::baseobjspace::DictWrapKind::Instance,
                );
                Ok(pyre_object::w_dict_proxy_new(canonical))
            }
        },
        2,
    );
    dict_storage_store(ns, "__dict__", make_getset_descriptor(dict_getter));

    let name_getter = make_builtin_function_with_arity(
        "__name__",
        |args| unsafe {
            let name = pyre_object::w_type_get_name(args[1]);
            Ok(pyre_object::w_str_new(name))
        },
        2,
    );
    // typeobject.py:1046 descr_set__name__
    let name_setter = make_builtin_function_with_arity(
        "__name__",
        |args| {
            let w_type = args[1];
            let w_value = args[2];
            // typeobject.py:1048 — only heap types may be renamed.
            if !unsafe { pyre_object::w_type_is_heaptype(w_type) } {
                return Err(crate::PyError::type_error(format!(
                    "can't set {}.__name__",
                    unsafe { pyre_object::w_type_get_name(w_type) }
                )));
            }
            // typeobject.py:1050 — `space.isinstance_w(w_value, space.w_text)`
            // accepts str and any str subclass, not only the exact type.
            if !unsafe { crate::baseobjspace::isinstance_str_w(w_value) } {
                return Err(crate::PyError::type_error(format!(
                    "can only assign string to {}.__name__, not '{}'",
                    unsafe { pyre_object::w_type_get_name(w_type) },
                    type_name_of(w_value)
                )));
            }
            // typeobject.py:1054 text_w — read through the surrogate-aware
            // WTF-8 view so a lone surrogate does not panic before the
            // checks below run.
            let wtf8 = unsafe { pyre_object::w_str_get_wtf8(w_value) };
            // typeobject.py:1055 — reject embedded null characters.
            for cp in wtf8.code_points() {
                if cp.to_u32() == 0 {
                    return Err(crate::PyError::value_error(
                        "type name must not contain null characters",
                    ));
                }
            }
            // typeobject.py:1057 _check_surrogate.
            crate::builtins::check_surrogate(w_value)?;
            // typeobject.py:1058 `w_type.name = name` — surrogate-free, so
            // the str view is valid UTF-8.
            let name = unsafe { pyre_object::w_str_get_value(w_value) };
            unsafe { pyre_object::w_type_set_name(w_type, name) };
            Ok(pyre_object::w_none())
        },
        3,
    );
    dict_storage_store(
        ns,
        "__name__",
        make_getset_property_named(name_getter, name_setter, pyre_object::PY_NULL, "__name__"),
    );

    let bases_getter = make_builtin_function_with_arity(
        "__bases__",
        |args| unsafe {
            let bases = pyre_object::w_type_get_bases(args[1]);
            if bases.is_null() {
                return Ok(pyre_object::w_tuple_new(vec![]));
            }
            Ok(bases)
        },
        2,
    );
    let bases_setter = make_builtin_function_with_arity("__bases__", type_set_bases, 3);
    dict_storage_store(
        ns,
        "__bases__",
        make_getset_property_named(
            bases_getter,
            bases_setter,
            pyre_object::PY_NULL,
            "__bases__",
        ),
    );
}

/// `type.__bases__` setter (typeobject.py:1064-1105 `descr_set__bases__`).
/// Heap types only; the new bases must be a non-empty tuple of classes whose
/// best base shares the current instance layout (so instances stay valid).
/// On success the MRO is recomputed and the type is re-registered on its new
/// bases.
fn type_set_bases(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        let w_type = args[1];
        let w_value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        let type_name = pyre_object::w_type_get_name(w_type);
        if !pyre_object::w_type_is_heaptype(w_type) {
            return Err(crate::PyError::type_error(format!(
                "can't set {type_name}.__bases__"
            )));
        }
        if w_value.is_null() || !pyre_object::is_tuple(w_value) {
            return Err(crate::PyError::type_error(format!(
                "can only assign tuple to {type_name}.__bases__"
            )));
        }
        let n = pyre_object::w_tuple_len(w_value);
        if n == 0 {
            return Err(crate::PyError::type_error(format!(
                "can only assign non-empty tuple to {type_name}.__bases__"
            )));
        }
        // find_best_base: pick the base with the most-derived instance layout.
        let mut w_bestbase = pyre_object::PY_NULL;
        let mut best_layout: *const pyre_object::typeobject::Layout = std::ptr::null();
        for i in 0..n {
            let Some(w_base) = pyre_object::w_tuple_getitem(w_value, i as i64) else {
                continue;
            };
            if std::ptr::eq(w_base, w_type) {
                return Err(crate::PyError::type_error(
                    "a __bases__ item causes an inheritance cycle",
                ));
            }
            if !pyre_object::is_type(w_base) {
                return Err(crate::PyError::type_error(format!(
                    "{type_name}.__bases__ must be tuple of classes, not '{}'",
                    (*(*w_base).ob_type).name
                )));
            }
            let cand_layout = pyre_object::w_type_get_layout_ptr(w_base);
            if best_layout.is_null()
                || (cand_layout != best_layout
                    && !cand_layout.is_null()
                    && (*cand_layout).issublayout(best_layout))
            {
                w_bestbase = w_base;
                best_layout = cand_layout;
            }
        }
        // Instances keep their current layout, so the new best base must share
        // it (no instance-size change).  Adding layout-neutral mixin bases such
        // as Generic is fine; switching to an incompatible solid base is not.
        let cur_layout = pyre_object::w_type_get_layout_ptr(w_type);
        if best_layout != cur_layout {
            return Err(crate::PyError::type_error(format!(
                "__bases__ assignment: '{}' object layout differs from '{type_name}'",
                pyre_object::w_type_get_name(w_bestbase)
            )));
        }
        pyre_object::typeobject::w_type_set_bases(w_type, w_value);
        let mro = crate::baseobjspace::compute_mro(w_type);
        pyre_object::w_type_set_mro(w_type, mro);
        pyre_object::typeobject::w_type_ready(w_type);
        Ok(pyre_object::w_none())
    }
}

/// function/builtin_function_or_method — PyPy: function.py Function typedef
/// descr_function_get (function.py:462): always returns a Method.
/// PyPy: shared `Function.typedef.rawdict` entries that BuiltinFunction.typedef
/// inherits via `TypeDef("builtin_function", **Function.typedef.rawdict)`.
///
/// Slots that exist on `Function.typedef` *and* on `BuiltinFunction.typedef`
/// belong here so the two initializers stay structurally aligned with PyPy's
/// `**rawdict` pattern. Function-only slots (currently just `__get__`) and
/// BuiltinFunction-only overrides (`__new__`, `__self__`, `__repr__`)
/// live in their respective wrappers.
fn init_function_type_common(ns: &mut DictStorage) {
    // `pypy/interpreter/typedef.py:802 __doc__ = getset_func_doc` —
    // `getset_func_doc = GetSetProperty(Function.fget_func_doc,
    // fset_func_doc, fdel_func_doc)` (typedef.py:758-760) lives on
    // `Function.typedef`'s rawdict so it is inherited by
    // `BuiltinFunction.typedef` via `**Function.typedef.rawdict`
    // (typedef.py:899).  Registering the descriptor here mirrors that
    // shape so `del f.__doc__` on a user-defined function reaches the
    // typedef `__delete__` slot (and through it
    // `function_del_doc`'s sticky-None write — function.py:455-457),
    // not the fall-through "no attribute" path.  The `_check_code_mutable`
    // gate inside the setter/deleter still raises `TypeError` for
    // builtin functions (`can_change_code = False`).
    let doc_getter = make_builtin_function("__doc__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_none());
        }
        Ok(unsafe { crate::function::fget_func_doc(func) })
    });
    let doc_setter = make_builtin_function("__doc__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_doc(func, value)? };
        Ok(pyre_object::w_none())
    });
    let doc_deleter = make_builtin_function("__doc__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fdel_func_doc(func)? };
        Ok(pyre_object::w_none())
    });
    dict_storage_store(
        ns,
        "__doc__",
        make_getset_property(doc_getter, doc_setter, doc_deleter),
    );
    // `pypy/interpreter/typedef.py:811 __annotations__ =
    // getset_func_annotations` →
    // `getset_func_annotations = GetSetProperty(Function.fget_func_annotations,
    //                                            Function.fset_func_annotations,
    //                                            Function.fdel_func_annotations)`
    // (typedef.py:787-789).  Without this descriptor, `f.__annotations__
    // = X` falls through to the generic `setdictvalue` which would
    // shadow the `Function.w_ann` slot (the getattr fast path reads
    // `w_ann` directly).  The setter validates the new value as a
    // dict per `function.py:557-558` and clears the slot on `None`.
    let ann_getter = make_builtin_function("__annotations__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_dict_new());
        }
        Ok(unsafe { crate::function::function_get_annotations(func) })
    });
    let ann_setter = make_builtin_function("__annotations__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_annotations(func, value)? };
        Ok(pyre_object::w_none())
    });
    let ann_deleter = make_builtin_function("__annotations__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fdel_func_annotations(func)? };
        Ok(pyre_object::w_none())
    });
    dict_storage_store(
        ns,
        "__annotations__",
        make_getset_property(ann_getter, ann_setter, ann_deleter),
    );
    // ── Remaining `pypy/interpreter/typedef.py:758-815 Function.typedef`
    // GetSetProperty entries.  Installing each as a typedef descriptor
    // is what makes user-level `f.__name__ = "x"` go through the
    // validating `function.py:fset_func_name` path instead of the
    // generic `setdictvalue` fall-through.  Reads keep using the
    // baseobjspace fast path (it produces the same value the descriptor
    // `__get__` would return); the descriptor's role is to enforce the
    // setter / deleter type checks PyPy applies before mutating the
    // function instance.
    //
    // `typedef.py:780 getset_func_name = GetSetProperty(fget_func_name,
    //                                                    fset_func_name)`.
    let name_getter = make_builtin_function("__name__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_str_new(""));
        }
        Ok(unsafe { crate::function::fget_func_name(func) })
    });
    let name_setter = make_builtin_function("__name__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_name(func, value)? };
        Ok(pyre_object::w_none())
    });
    dict_storage_store(
        ns,
        "__name__",
        make_getset_property(name_getter, name_setter, pyre_object::PY_NULL),
    );
    // `typedef.py:782 getset_func_qualname = GetSetProperty(
    //   Function.fget_func_qualname, Function.fset_func_qualname)`.
    // Both getter and setter wired so `f.__qualname__ = "C.m"`
    // reaches `fset_func_qualname`'s str validation
    // (function.py:476-485) instead of falling through to
    // `setdictvalue` and silently shadowing the slot.
    let qualname_getter = make_builtin_function("__qualname__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_str_new(""));
        }
        let s = unsafe { crate::function::function_get_qualname(func) };
        Ok(pyre_object::w_str_new(&s))
    });
    let qualname_setter = make_builtin_function("__qualname__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_qualname(func, value)? };
        Ok(pyre_object::w_none())
    });
    dict_storage_store(
        ns,
        "__qualname__",
        make_getset_property(qualname_getter, qualname_setter, pyre_object::PY_NULL),
    );
    // `typedef.py:768-770 getset___module__ = GetSetProperty(
    //   Function.fget___module__, fset___module__, fdel___module__)`.
    let module_getter = make_builtin_function("__module__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_none());
        }
        Ok(unsafe { crate::function::fget___module__(func) })
    });
    let module_setter = make_builtin_function("__module__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset___module__(func, value)? };
        Ok(pyre_object::w_none())
    });
    let module_deleter = make_builtin_function("__module__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fdel___module__(func)? };
        Ok(pyre_object::w_none())
    });
    dict_storage_store(
        ns,
        "__module__",
        make_getset_property(module_getter, module_setter, module_deleter),
    );
    // `typedef.py:772-774 getset_func_defaults = GetSetProperty(
    //   Function.fget_func_defaults, fset_func_defaults, fdel_func_defaults)`.
    let defaults_getter = make_builtin_function("__defaults__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_none());
        }
        Ok(unsafe { crate::function::fget_func_defaults(func) })
    });
    let defaults_setter = make_builtin_function("__defaults__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_defaults(func, value)? };
        Ok(pyre_object::w_none())
    });
    let defaults_deleter = make_builtin_function("__defaults__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fdel_func_defaults(func)? };
        Ok(pyre_object::w_none())
    });
    dict_storage_store(
        ns,
        "__defaults__",
        make_getset_property(defaults_getter, defaults_setter, defaults_deleter),
    );
    // `typedef.py:775-777 getset_func_kwdefaults = GetSetProperty(...)`.
    let kwdefaults_getter = make_builtin_function("__kwdefaults__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_none());
        }
        Ok(unsafe { crate::function::fget_func_kwdefaults(func) })
    });
    let kwdefaults_setter = make_builtin_function("__kwdefaults__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_kwdefaults(func, value)? };
        Ok(pyre_object::w_none())
    });
    let kwdefaults_deleter = make_builtin_function("__kwdefaults__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fdel_func_kwdefaults(func)? };
        Ok(pyre_object::w_none())
    });
    dict_storage_store(
        ns,
        "__kwdefaults__",
        make_getset_property(kwdefaults_getter, kwdefaults_setter, kwdefaults_deleter),
    );
    // `typedef.py:778-779 getset_func_code = GetSetProperty(
    //   Function.fget_func_code, fset_func_code)`.
    let code_getter = make_builtin_function("__code__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_none());
        }
        let raw = unsafe { crate::function::fget_func_code(func) };
        Ok(raw as pyre_object::PyObjectRef)
    });
    let code_setter = make_builtin_function("__code__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_code(func, value)? };
        Ok(pyre_object::w_none())
    });
    dict_storage_store(
        ns,
        "__code__",
        make_getset_property(code_getter, code_setter, pyre_object::PY_NULL),
    );
    // `typedef.py:813 __closure__ = GetSetProperty(Function.fget_func_closure)`
    // — read-only.
    let closure_getter = make_builtin_function("__closure__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_none());
        }
        Ok(unsafe { crate::function::fget_func_closure(func) })
    });
    dict_storage_store(ns, "__closure__", make_getset_descriptor(closure_getter));
    // `typedef.py:826 __globals__ = interp_attrproperty_w('w_func_globals',
    // cls=Function)` — read-only canonical W_DictObject view of the
    // function's globals storage.  `interp_attrproperty_w`
    // (`typedef.py:465-474`) fetches the attribute and substitutes
    // `space.w_None` when the slot is `None`.  pyre's
    // `function_get_globals_obj` returns `PY_NULL` for builtins
    // allocated with a null storage pointer (gateway.rs:661-700);
    // route that through `w_None` so `BuiltinFunction.__globals__`
    // observes `None` rather than a raw null leak — the literal
    // `if w_value is None` arm of fget.
    let globals_getter = make_builtin_function("__globals__", |args| {
        let func = args[1];
        let w_value = unsafe { crate::function::function_get_globals_obj(func) };
        if w_value.is_null() {
            Ok(pyre_object::w_none())
        } else {
            Ok(w_value)
        }
    });
    dict_storage_store(ns, "__globals__", make_getset_descriptor(globals_getter));
    // `func.__builtins__` — `_PyEval_BuiltinsFromGlobals(globals)`: look up
    // `__builtins__` in the function's globals; a Module yields its dict,
    // any other value is returned directly, and an absent key falls back to
    // the default builtin Module.  `pick_builtin_obj` already performs that
    // resolution (honoring a custom `__builtins__`); convert the Module it
    // returns to its dict so callers see a mapping (annotationlib's
    // `{**annotate.__builtins__, **annotate.__globals__}`).
    let func_builtins_getter = make_builtin_function("__builtins__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_none());
        }
        let w_globals = unsafe { crate::function::function_get_globals_obj(func) };
        let exec_ctx = crate::call::take_last_exec_ctx();
        let w_builtin = crate::baseobjspace::pick_builtin_obj(w_globals, exec_ctx);
        if w_builtin.is_null() {
            Ok(pyre_object::w_none())
        } else if unsafe { pyre_object::is_module(w_builtin) } {
            Ok(unsafe { pyre_object::w_module_get_w_dict(w_builtin) })
        } else {
            Ok(w_builtin)
        }
    });
    dict_storage_store(
        ns,
        "__builtins__",
        make_getset_descriptor(func_builtins_getter),
    );
    // `pypy/interpreter/typedef.py:805 __objclass__ = getset_func_objclass`
    //
    // ```python
    // getset_func_objclass = GetSetProperty(Function.fget_func_objclass)
    // ```
    //
    // Read-only descriptor that surfaces `self.w_objclass` for
    // introspection helpers (`inspect.getfullargspec` etc.); raises
    // AttributeError when no class is bound (`function.py:498-501`).
    dict_storage_store(
        ns,
        "__objclass__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__objclass__",
            |args| {
                let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                if func.is_null() {
                    return Err(crate::PyError::attribute_error("__objclass__"));
                }
                unsafe { crate::function::fget_func_objclass(func) }
            },
            2,
        )),
    );
    // `pypy/interpreter/typedef.py:806 __text_signature__ =
    // getset_func_text_signature` —
    //
    // ```python
    // getset_func_text_signature = GetSetProperty(
    //     Function.fget_func_text_signature,
    //     Function.fset_func_text_signature)
    // ```
    let text_signature_getter = make_builtin_function("__text_signature__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Err(crate::PyError::attribute_error("__text_signature__"));
        }
        unsafe { crate::function::fget_func_text_signature(func) }
    });
    let text_signature_setter = make_builtin_function("__text_signature__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        if !func.is_null() {
            unsafe { crate::function::fset_func_text_signature(func, value) };
        }
        Ok(pyre_object::w_none())
    });
    dict_storage_store(
        ns,
        "__text_signature__",
        make_getset_property(
            text_signature_getter,
            text_signature_setter,
            pyre_object::PY_NULL,
        ),
    );
    // `pypy/interpreter/typedef.py:809 __defaults_count__ =
    // GetSetProperty(Function.fget_defaults_count)` — a PyPy
    // extension that lets `inspect.py` distinguish "no default" from
    // "default is None" when introspecting builtins like `dict.pop`.
    //
    // ```python
    // def fget_defaults_count(self, space):
    //     return space.newint(len(self.defs_w))
    // ```
    //
    // Pyre stores `defs_w` as either a tuple PyObjectRef or PY_NULL
    // (the latter mirrors PyPy's empty-list `[]`).
    dict_storage_store(
        ns,
        "__defaults_count__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__defaults_count__",
            |args| {
                let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                if func.is_null() {
                    return Ok(pyre_object::w_int_new(0));
                }
                let defaults = unsafe { crate::function::function_get_defaults(func) };
                let n = if defaults.is_null() {
                    0
                } else if unsafe { pyre_object::is_tuple(defaults) } {
                    unsafe { pyre_object::w_tuple_len(defaults) as i64 }
                } else {
                    0
                };
                Ok(pyre_object::w_int_new(n))
            },
            2,
        )),
    );
}

fn init_function_type(ns: &mut DictStorage) {
    init_function_type_common(ns);
    // `funcobject.c func_new` — `FunctionType(code, globals, name=None,
    // argdefs=None, closure=None, kwdefaults=None)`.
    dict_storage_store(
        ns,
        "__new__",
        make_new_descr(crate::function::descr_function_new),
    );
    dict_storage_store(
        ns,
        "__get__",
        make_builtin_function("__get__", |args| {
            let w_function = args.first().copied().unwrap_or(pyre_object::w_none());
            let w_obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let w_cls = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
            // function.py:464-470 descr_function_get
            //
            //   asking_for_function = (
            //       space.is_w(w_cls, space.w_None)
            //       or (
            //           space.is_w(w_obj, space.w_None)
            //           and not space.is_w(w_cls, space.type(space.w_None))
            //       )
            //   )
            //
            // The class-access case (`w_obj == None and w_cls is some type`)
            // returns the bare function — that's how `cls.func` stays callable
            // as a plain function rather than a bound method.
            let cls_is_none = unsafe { w_cls.is_null() || pyre_object::is_none(w_cls) };
            let obj_is_none = unsafe { w_obj.is_null() || pyre_object::is_none(w_obj) };
            let cls_is_none_type = std::ptr::eq(w_cls, gettypeobject(&pyre_object::NONE_TYPE));
            let asking_for_function = cls_is_none || (obj_is_none && !cls_is_none_type);
            if asking_for_function {
                Ok(w_function)
            } else {
                // function.py:470  Method(space, w_function, w_obj, w_cls)
                Ok(pyre_object::w_method_new(w_function, w_obj, w_cls))
            }
        }),
    );
}

/// PyPy typedef.py:813-820:
///
/// ```text
/// BuiltinFunction.typedef = TypeDef("builtin_function",
///                                   **Function.typedef.rawdict)
/// BuiltinFunction.typedef.rawdict.update({
///     '__new__': interp2app(BuiltinFunction.descr_builtinfunction__new__.im_func),
///     '__self__': GetSetProperty(always_none, cls=BuiltinFunction),
///     '__repr__': interp2app(BuiltinFunction.descr_function_repr),
///     '__doc__': getset_func_doc,
/// })
/// del BuiltinFunction.typedef.rawdict['__get__']
/// ```
///
/// `init_function_type_common` provides the shared `**rawdict` slots; the
/// missing `dict_storage_store(ns, "__get__", ...)` call after it expresses the
/// `del rawdict['__get__']` step. The `update({...})` overrides go below as
/// pyre starts modeling them.
fn init_builtin_function_type(ns: &mut DictStorage) {
    init_function_type_common(ns);
    dict_storage_store(
        ns,
        "__new__",
        make_new_descr(|_args| {
            Err(crate::PyError::type_error(
                "cannot create 'builtin_function' instances",
            ))
        }),
    );

    // typedef.py:816 GetSetProperty(always_none, cls=BuiltinFunction). The
    // `cls=` argument routes through descr_self_interp_w so wrong-class
    // instances raise DescrMismatch instead of silently returning None.
    // `init_builtin_function_type` runs while the BuiltinFunction
    // W_TypeObject is still under construction, so `cls` cannot be
    // resolved here; `patch_builtin_function_descriptors` runs after the
    // type cache is populated and writes the missing reqcls.
    // A builtin `__new__` carrier reports its defining type as `__self__`
    // (`typeobject.c add_tp_new_wrapper`), stamped via
    // `stamp_new_descr_self`; every other builtin function keeps the
    // `always_none` behaviour.
    let self_getter = make_builtin_function_with_arity(
        "__self__",
        |args| {
            let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if func.is_null() {
                return Ok(pyre_object::w_none());
            }
            Ok(unsafe { crate::function::function_get_self_or_none(func) })
        },
        2,
    );
    dict_storage_store(ns, "__self__", make_getset_descriptor(self_getter));

    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function_with_arity(
            "__repr__",
            |args| {
                let func = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let name = if func.is_null() {
                    "<unknown>"
                } else {
                    unsafe { crate::function_get_name(func) }
                };
                Ok(pyre_object::w_str_new(&format!(
                    "<built-in function {name}>"
                )))
            },
            1,
        ),
    );

    // `pypy/interpreter/typedef.py:899-906`
    // `BuiltinFunction.typedef.rawdict.update({...})` re-asserts
    // `__doc__` from the inherited `Function.typedef.rawdict` slot
    // (also `getset_func_doc`).  Pyre installs `__doc__` once in
    // `init_function_type_common` so both function types share the
    // same getter/setter/deleter; the `_check_code_mutable` gate
    // inside the setter/deleter raises `TypeError` for builtin
    // instances because `can_change_code = False`.
}

/// typedef.py:816,818 wires `cls=BuiltinFunction` on the `__self__` and
/// `__doc__` GetSetProperty entries; the inner `init_builtin_function_type`
/// runs while the W_TypeObject is still under construction, so the reqcls
/// patch happens here, after `init_typeobjects` has filled the cache and
/// the BuiltinFunction typeobject is reachable.
fn patch_builtin_function_descriptors() {
    let bf_type =
        gettypefor(&crate::BUILTIN_FUNCTION_TYPE as *const PyType).unwrap_or(pyre_object::PY_NULL);
    if bf_type.is_null() {
        return;
    }
    let dict_ptr = unsafe { pyre_object::w_type_get_dict_ptr(bf_type) } as *mut DictStorage;
    if dict_ptr.is_null() {
        return;
    }
    let ns = unsafe { &*dict_ptr };
    for name in ["__self__", "__doc__"] {
        if let Some(&descr) = ns.get(name) {
            if unsafe { pyre_object::typedef::is_getset_property(descr) } {
                // typedef.py:818 `cls=BuiltinFunction` — patch the
                // `reqcls` slot in place now that the BuiltinFunction
                // typeobject exists.  GetSetProperty's reqcls is a
                // single PyObjectRef field, so this is a one-line
                // store rather than the previous side-table read /
                // mutate / write back dance.
                unsafe { pyre_object::typedef::w_getset_set_reqcls(descr, bf_type) };
            }
        }
    }
}

/// typedef.py:736-770 — `PyFrame.typedef` / `PyTraceback.typedef` build
/// their getsets as `GetSetProperty(PyFrame.fget_*, cls=PyFrame)` /
/// `GetSetProperty(PyTraceback.descr_*, cls=PyTraceback)`.  The `cls`
/// stamps `reqcls`, so a getset invoked with a foreign receiver
/// (`type(f).f_code.__get__(1, int)`) raises the descriptor
/// `TypeError` in `__get__`/`__set__` instead of reaching the closure,
/// which casts the receiver straight to `*mut PyFrame` /
/// `*mut PyTraceback` and would otherwise read at struct offsets on
/// arbitrary memory.  The frame/traceback getsets are created
/// reqcls-less (`make_getset_descriptor_named`), so patch the slot in
/// place once both typeobjects exist — the same shape as
/// `patch_builtin_function_descriptors`.
fn patch_frame_traceback_descriptors() {
    for layout in [
        &crate::pyframe::FRAME_TYPE as *const PyType,
        &crate::pytraceback::PYTRACEBACK_TYPE as *const PyType,
    ] {
        let w_type = gettypefor(layout).unwrap_or(pyre_object::PY_NULL);
        if w_type.is_null() {
            continue;
        }
        let dict_ptr = unsafe { pyre_object::w_type_get_dict_ptr(w_type) } as *mut DictStorage;
        if dict_ptr.is_null() {
            continue;
        }
        let ns = unsafe { &*dict_ptr };
        let descrs: Vec<PyObjectRef> = ns
            .entries()
            .filter_map(|(_, &descr)| {
                (!descr.is_null() && unsafe { pyre_object::typedef::is_getset_property(descr) })
                    .then_some(descr)
            })
            .collect();
        for descr in descrs {
            unsafe { pyre_object::typedef::w_getset_set_reqcls(descr, w_type) };
        }
    }
}

/// BuiltinCode.typedef (typedef.py) — code object attributes for builtins.
///
/// PyPy exposes co_name, co_varnames, co_argcount, co_flags, co_consts.
/// No __get__ — BuiltinCode is a code object, not a descriptor.
fn init_builtin_code_type(ns: &mut DictStorage) {
    let co_name_getter = make_builtin_function_with_arity(
        "co_name",
        |args| {
            let code = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if code.is_null() {
                return Ok(pyre_object::w_none());
            }
            let name = unsafe { crate::builtin_code_name(code) };
            Ok(pyre_object::w_str_new(name))
        },
        2,
    );
    dict_storage_store(ns, "co_name", make_getset_descriptor(co_name_getter));

    // Signature-derived attrs (fget_co_argcount etc., typedef.py). A
    // builtin code with no recorded Signature reports zero/empty so
    // inspect.signature() degrades to an empty signature instead of
    // raising AttributeError.
    fn code_sig(args: &[pyre_object::PyObjectRef]) -> Option<&'static crate::gateway::Signature> {
        let code = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if code.is_null() {
            None
        } else {
            unsafe { crate::builtin_code_get_signature(code) }
        }
    }
    let argcount_getter = make_builtin_function_with_arity(
        "co_argcount",
        |args| {
            Ok(pyre_object::w_int_new(
                code_sig(args).map_or(0, |s| s.num_argnames()) as i64,
            ))
        },
        2,
    );
    dict_storage_store(ns, "co_argcount", make_getset_descriptor(argcount_getter));
    let posonly_getter = make_builtin_function_with_arity(
        "co_posonlyargcount",
        |args| {
            Ok(pyre_object::w_int_new(
                code_sig(args).map_or(0, |s| s.num_posonlyargnames()) as i64,
            ))
        },
        2,
    );
    dict_storage_store(
        ns,
        "co_posonlyargcount",
        make_getset_descriptor(posonly_getter),
    );
    let kwonly_getter = make_builtin_function_with_arity(
        "co_kwonlyargcount",
        |args| {
            Ok(pyre_object::w_int_new(
                code_sig(args).map_or(0, |s| s.num_kwonlyargnames()) as i64,
            ))
        },
        2,
    );
    dict_storage_store(
        ns,
        "co_kwonlyargcount",
        make_getset_descriptor(kwonly_getter),
    );
    let varnames_getter = make_builtin_function_with_arity(
        "co_varnames",
        |args| {
            let names = code_sig(args)
                .map(|s| {
                    s.getallvarnames()
                        .iter()
                        .map(|n| pyre_object::w_str_new(n))
                        .collect()
                })
                .unwrap_or_default();
            Ok(pyre_object::w_tuple_new(names))
        },
        2,
    );
    dict_storage_store(ns, "co_varnames", make_getset_descriptor(varnames_getter));
    let flags_getter = make_builtin_function_with_arity(
        "co_flags",
        |args| {
            let mut flags = 0i64;
            if let Some(s) = code_sig(args) {
                if s.has_vararg() {
                    flags |= 0x04; // CO_VARARGS
                }
                if s.has_kwarg() {
                    flags |= 0x08; // CO_VARKEYWORDS
                }
            }
            Ok(pyre_object::w_int_new(flags))
        },
        2,
    );
    dict_storage_store(ns, "co_flags", make_getset_descriptor(flags_getter));
}

fn init_method_type(ns: &mut DictStorage) {
    // typedef.py:839-840 ─
    //   __func__ = interp_attrproperty_w('w_function', cls=Method),
    //   __self__ = interp_attrproperty_w('w_instance', cls=Method),
    // — both read-only.  `interp_attrproperty_w` (typedef.py:465-474)
    // fetches the attribute and substitutes `space.w_None` when the
    // slot is `None`; the accessor returns w_method_get_func /
    // w_method_get_self raw, so a null `w_function` / `w_instance`
    // (unbound creation paths) leaked through the descriptor.  Mirror
    // the upstream `if w_value is None: return space.w_None` arm.
    let func_getter = make_builtin_function_with_arity(
        "__func__",
        |args| {
            let method = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if !unsafe { pyre_object::function::is_method(method) } {
                return Ok(pyre_object::w_none());
            }
            let w_value = unsafe { pyre_object::w_method_get_func(method) };
            if w_value.is_null() {
                Ok(pyre_object::w_none())
            } else {
                Ok(w_value)
            }
        },
        2,
    );
    dict_storage_store(ns, "__func__", make_getset_descriptor(func_getter));
    let self_getter = make_builtin_function_with_arity(
        "__self__",
        |args| {
            let method = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if !unsafe { pyre_object::function::is_method(method) } {
                return Ok(pyre_object::w_none());
            }
            let w_value = unsafe { pyre_object::w_method_get_self(method) };
            if w_value.is_null() {
                Ok(pyre_object::w_none())
            } else {
                Ok(w_value)
            }
        },
        2,
    );
    dict_storage_store(ns, "__self__", make_getset_descriptor(self_getter));
}

fn init_code_type(ns: &mut DictStorage) {
    // code.replace(**kwargs) — pycode.py:543-550 W_PyCode.descr_replace →
    // reconstruct the code object with the given co_* fields overridden.
    dict_storage_store(
        ns,
        "replace",
        make_builtin_function("replace", |args| unsafe {
            crate::pycode::code_replace(args)
        }),
    );
    // `pypy/interpreter/typedef.py:720`
    // `co_exceptiontable = interp_attrproperty('co_exceptiontable', cls=PyCode,
    //                                          wrapfn="newbytes")`.
    //
    // Read-only attribute exposing the raw varint-packed table.  The
    // matching getset descriptor wraps the field as a `bytes` object
    // (PyPy `wrapfn="newbytes"`).  `args[0]` is the descriptor itself,
    // `args[1]` is the PyCode instance (typedef.py:467-470 calling
    // convention via `descr_property_get`).
    dict_storage_store(
        ns,
        "co_exceptiontable",
        make_getset_descriptor(make_builtin_function_with_arity(
            "co_exceptiontable",
            |args| {
                let w_self = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                if w_self.is_null() {
                    return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&[]));
                }
                if !unsafe { crate::pycode::is_code(w_self) } {
                    return Err(crate::PyError::type_error(
                        "descriptor 'co_exceptiontable' requires a 'code' object",
                    ));
                }
                let bytes = unsafe { crate::pycode::w_code_exceptiontable(w_self) };
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
            },
            2,
        )),
    );
    // code.co_positions() — PEP 657 per-instruction source positions
    // (`pycode.py` exposes `co_positions` via `co_positions_iterator`).
    // Yields one `(start_line, end_line, start_col, end_col)` tuple per
    // instruction from the decoded `CodeObject.locations` table.
    // `traceback._get_code_position` indexes the result at `tb_lasti // 2`,
    // so the `tb_lasti` getter reports the byte-offset form
    // (`2 * instruction_index`) to land on the same entry.
    //
    // The column fields are reported as `None`: pyre lacks the
    // `compile(PyCF_ONLY_AST)` AST surface that `traceback`'s 3.14 caret
    // anchoring (`_should_show_carets` / `_byte_offset_to_character_offset`)
    // re-parses, and a `None` column makes `format_frame_summary`
    // (`traceback.py:556-561`) take the column-free single-line branch —
    // the same graceful degradation CPython uses when range info is absent
    // (`-X no_debug_ranges`).  Line numbers stay exact.
    dict_storage_store(
        ns,
        "co_positions",
        make_builtin_function("co_positions", |args| {
            let w_self = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if w_self.is_null() || !unsafe { crate::pycode::is_code(w_self) } {
                return Err(crate::PyError::type_error(
                    "descriptor 'co_positions' requires a 'code' object",
                ));
            }
            let code_ptr =
                unsafe { crate::pycode::w_code_get_ptr(w_self) } as *const crate::CodeObject;
            let rows: Vec<pyre_object::PyObjectRef> = if code_ptr.is_null() {
                Vec::new()
            } else {
                let code = unsafe { &*code_ptr };
                code.locations
                    .iter()
                    .map(|(start, end)| {
                        pyre_object::w_tuple_new(vec![
                            pyre_object::w_int_new(start.line.get() as i64),
                            pyre_object::w_int_new(end.line.get() as i64),
                            pyre_object::w_none(),
                            pyre_object::w_none(),
                        ])
                    })
                    .collect()
            };
            let n = rows.len();
            Ok(pyre_object::w_seq_iter_new(
                pyre_object::w_list_new(rows),
                n,
            ))
        }),
    );
}

/// typedef.py:533-540 Member.typedef
fn init_member_descriptor_type(ns: &mut DictStorage) {
    // typedef.py:535 __get__ = interp2app(Member.descr_member_get)
    dict_storage_store(
        ns,
        "__get__",
        make_builtin_function("__get__", |args| {
            let descr = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if descr.is_null() || !unsafe { pyre_object::typedef::is_member(descr) } {
                return Ok(pyre_object::w_none());
            }
            let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            // typedef.py:507-508: if space.is_w(w_obj, space.w_None): return self
            if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
                return Ok(descr);
            }
            // typedef.py:510: self.typecheck(space, w_obj)
            unsafe {
                let w_cls = pyre_object::w_member_get_cls(descr);
                if !w_cls.is_null()
                    && pyre_object::is_type(w_cls)
                    && !crate::baseobjspace::isinstance_w(obj, w_cls)
                {
                    let slot_name = pyre_object::w_member_get_name(descr);
                    return Err(crate::PyError::type_error(format!(
                        "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                        slot_name,
                        pyre_object::w_type_get_name(w_cls),
                        (*(*obj).ob_type).name,
                    )));
                }
            }
            // typedef.py:511-516: w_result = w_obj.getslotvalue(self.index);
            // None → AttributeError("'%T' object has no attribute '%s'").
            let slot_name = unsafe { pyre_object::w_member_get_name(descr) };
            let index = unsafe { pyre_object::w_member_get_index(descr) };
            let found = if unsafe { pyre_object::is_instance(obj) } {
                unsafe { crate::objspace::std::mapdict::getslotvalue(obj, index) }
            } else {
                crate::baseobjspace::native_slot_get(obj, slot_name)
            };
            match found {
                Some(v) => Ok(v),
                None => Err(crate::PyError::new(
                    crate::PyErrorKind::AttributeError,
                    format!(
                        "'{}' object has no attribute '{}'",
                        unsafe { (*(*obj).ob_type).name },
                        slot_name,
                    ),
                )),
            }
        }),
    );
    // typedef.py:536 __set__ = interp2app(Member.descr_member_set)
    dict_storage_store(
        ns,
        "__set__",
        make_builtin_function("__set__", |args| {
            let descr = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if descr.is_null() || !unsafe { pyre_object::typedef::is_member(descr) } {
                return Ok(pyre_object::w_none());
            }
            let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
            // typedef.py:521: self.typecheck(space, w_obj)
            unsafe {
                let w_cls = pyre_object::w_member_get_cls(descr);
                if !w_cls.is_null()
                    && pyre_object::is_type(w_cls)
                    && !crate::baseobjspace::isinstance_w(obj, w_cls)
                {
                    let slot_name = pyre_object::w_member_get_name(descr);
                    return Err(crate::PyError::type_error(format!(
                        "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                        slot_name,
                        pyre_object::w_type_get_name(w_cls),
                        (*(*obj).ob_type).name,
                    )));
                }
            }
            // typedef.py:522: w_obj.setslotvalue(self.index, w_value)
            let index = unsafe { pyre_object::w_member_get_index(descr) };
            if unsafe { pyre_object::is_instance(obj) } {
                unsafe { crate::objspace::std::mapdict::setslotvalue(obj, index, value) };
            } else {
                let slot_name = unsafe { pyre_object::w_member_get_name(descr) };
                if !crate::baseobjspace::native_slot_set(obj, slot_name, value) {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::AttributeError,
                        format!(
                            "'{}' object attribute '{}' is read-only",
                            unsafe { (*(*obj).ob_type).name },
                            slot_name,
                        ),
                    ));
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    // typedef.py:537 __delete__ = interp2app(Member.descr_member_del)
    dict_storage_store(
        ns,
        "__delete__",
        make_builtin_function("__delete__", |args| {
            let descr = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if descr.is_null() || !unsafe { pyre_object::typedef::is_member(descr) } {
                return Ok(pyre_object::w_none());
            }
            let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            // typedef.py:526: self.typecheck(space, w_obj)
            unsafe {
                let w_cls = pyre_object::w_member_get_cls(descr);
                if !w_cls.is_null()
                    && pyre_object::is_type(w_cls)
                    && !crate::baseobjspace::isinstance_w(obj, w_cls)
                {
                    let slot_name = pyre_object::w_member_get_name(descr);
                    return Err(crate::PyError::type_error(format!(
                        "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                        slot_name,
                        pyre_object::w_type_get_name(w_cls),
                        (*(*obj).ob_type).name,
                    )));
                }
            }
            // typedef.py:527-531: success = w_obj.delslotvalue(self.index)
            let slot_name = unsafe { pyre_object::w_member_get_name(descr) };
            let index = unsafe { pyre_object::w_member_get_index(descr) };
            let removed = if unsafe { pyre_object::is_instance(obj) } {
                unsafe { crate::objspace::std::mapdict::delslotvalue(obj, index) }
            } else {
                crate::baseobjspace::native_slot_del(obj, slot_name)
            };
            if !removed {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::AttributeError,
                    slot_name.to_string(),
                ));
            }
            Ok(pyre_object::w_none())
        }),
    );
    // typedef.py:538 __name__ = interp_attrproperty('name', ...)
    let name_getter = make_builtin_function_with_arity(
        "__name__",
        |args| {
            let member = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if member.is_null() || !unsafe { pyre_object::typedef::is_member(member) } {
                return Ok(pyre_object::w_none());
            }
            Ok(pyre_object::w_str_new(unsafe {
                pyre_object::w_member_get_name(member)
            }))
        },
        2,
    );
    dict_storage_store(ns, "__name__", make_getset_descriptor(name_getter));
    // typedef.py:539 `__objclass__ = interp_attrproperty_w('w_cls',
    // cls=Member)` — read-only.  `interp_attrproperty_w`
    // (typedef.py:465-474) fetches the attribute and substitutes
    // `space.w_None` when the slot is `None`; mirror that fget shape
    // arm-for-arm.  The `is_member` guard stays as a defensive type
    // check at the builtin-function boundary (PyPy's
    // `descr_property_get` rejects non-Member instances before
    // reaching fget; pyre's GetSetProperty path is less strict).
    let objclass_getter = make_builtin_function_with_arity(
        "__objclass__",
        |args| {
            let member = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if !unsafe { pyre_object::typedef::is_member(member) } {
                return Ok(pyre_object::w_none());
            }
            let w_value = unsafe { pyre_object::w_member_get_cls(member) };
            if w_value.is_null() {
                Ok(pyre_object::w_none())
            } else {
                Ok(w_value)
            }
        },
        2,
    );
    dict_storage_store(ns, "__objclass__", make_getset_descriptor(objclass_getter));
}

/// `nestedscope.py:Cell` typedef.  PyPy `typedef.py:934-952 Cell.typedef`:
///
/// ```python
/// Cell.typedef = TypeDef("cell",
///     ...
///     __repr__     = interp2app(Cell.descr__repr__),
///     ...
///     cell_contents= GetSetProperty(
///         Cell.descr__cell_contents,
///         Cell.descr_set_cell_contents,
///         Cell.descr_del_cell_contents,
///         cls=Cell),
/// )
/// ```
///
/// Only the user-visible read/write/delete of `cell_contents` is ported
/// here.  `__eq__`/`__ne__`/`__lt__`/`__gt__`/`__le__`/`__ge__` cell-vs-cell
/// comparisons (`nestedscope.py:9-19 make_cell_cmp`) and `__hash__ = None`
/// remain unimplemented as a wider parity gap — they are not needed for
/// the descriptor-on-tuple-of-cells path that motivates this work.
fn init_cell_type(ns: &mut DictStorage) {
    // `nestedscope.py:112-116 descr__cell_contents`:
    //
    //     def descr__cell_contents(self, space):
    //         try:
    //             return self.get()
    //         except ValueError:
    //             raise oefmt(space.w_ValueError, "Cell is empty")
    //
    // `Cell.get()` (`nestedscope.py:31-44`) raises `ValueError` when
    // `self.w_value is None`.  Pyre represents an empty cell as
    // `contents = PY_NULL`, so the null-pointer check below mirrors the
    // upstream `self.w_value is None` test.
    let cell_contents_getter = make_builtin_function_with_arity(
        "cell_contents",
        |args| {
            let cell = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if cell.is_null() || !unsafe { pyre_object::is_cell(cell) } {
                return Err(crate::PyError::type_error(
                    "descriptor 'cell_contents' for 'cell' objects doesn't apply",
                ));
            }
            let v = unsafe { pyre_object::w_cell_get(cell) };
            if v.is_null() {
                return Err(crate::PyError::value_error("Cell is empty"));
            }
            Ok(v)
        },
        2,
    );
    // `nestedscope.py:118-119 descr_set_cell_contents`:
    //
    //     def descr_set_cell_contents(self, space, w_value):
    //         return self.set(w_value)
    let cell_contents_setter = make_builtin_function_with_arity(
        "cell_contents",
        |args| {
            let cell = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let w_value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
            if cell.is_null() || !unsafe { pyre_object::is_cell(cell) } {
                return Err(crate::PyError::type_error(
                    "descriptor 'cell_contents' for 'cell' objects doesn't apply",
                ));
            }
            unsafe { pyre_object::w_cell_set(cell, w_value) };
            Ok(pyre_object::w_none())
        },
        3,
    );
    // `nestedscope.py:121-125 descr_del_cell_contents`:
    //
    //     def descr_del_cell_contents(self, space):
    //         try:
    //             return self.delete()
    //         except ValueError:
    //             pass # CPython ignores it
    //
    // Pyre clears the cell to PY_NULL so a subsequent read raises the
    // same `Cell is empty` message; the `ValueError` from
    // `Cell.delete()` is swallowed per the upstream comment.
    let cell_contents_deleter = make_builtin_function_with_arity(
        "cell_contents",
        |args| {
            let cell = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if cell.is_null() || !unsafe { pyre_object::is_cell(cell) } {
                return Err(crate::PyError::type_error(
                    "descriptor 'cell_contents' for 'cell' objects doesn't apply",
                ));
            }
            unsafe { pyre_object::w_cell_set(cell, pyre_object::PY_NULL) };
            Ok(pyre_object::w_none())
        },
        2,
    );
    dict_storage_store(
        ns,
        "cell_contents",
        make_getset_property_named(
            cell_contents_getter,
            cell_contents_setter,
            cell_contents_deleter,
            "cell_contents",
        ),
    );
}

/// `staticmethod.__new__(cls, func)` — PyPy: function.py StaticMethod.descr__new__
fn init_staticmethod_type(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__new__",
        make_builtin_function("__new__", |args| {
            // staticmethod(func) — args[0] is cls (staticmethod type), args[1] is func
            let func = if args.len() > 1 {
                args[1]
            } else {
                pyre_object::w_none()
            };
            Ok(pyre_object::function::w_staticmethod_new(func))
        }),
    );
    // `typedef.py:866 __get__ = interp2app(
    //     StaticMethod.descr_staticmethod_get)`.  `function.py:691-693`:
    //
    //     def descr_staticmethod_get(self, w_obj, w_cls=None):
    //         """staticmethod(x).__get__(obj[, type]) -> x"""
    //         return self.w_function
    //
    // Arity 3 covers `__get__(self, obj, cls=None)`.  `args[0]` is the
    // staticmethod instance; the remaining slots are ignored beyond the
    // type guard.  Returning `w_function` without binding is the
    // canonical staticmethod semantic (`function.py:864 …does not
    // receive an implicit first argument`).
    dict_storage_store(
        ns,
        "__get__",
        make_builtin_function_with_arity(
            "__get__",
            |args| {
                let sm = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                if !unsafe { pyre_object::function::is_staticmethod(sm) } {
                    return Err(crate::PyError::type_error(
                        "descriptor '__get__' requires a 'staticmethod' object",
                    ));
                }
                let w_func = unsafe { pyre_object::function::w_staticmethod_get_func(sm) };
                if w_func.is_null() {
                    Ok(pyre_object::w_none())
                } else {
                    Ok(w_func)
                }
            },
            3,
        ),
    );
    // typedef.py:870-871 ─
    //   __func__ = interp_attrproperty_w('w_function', cls=StaticMethod),
    //   __wrapped__ = interp_attrproperty_w('w_function', cls=StaticMethod),
    // — same `w_function` slot, two aliases, both routed through
    // the interp_attrproperty_w fget shape (typedef.py:465-474):
    // substitute w_None when the fetched slot is None.
    fn staticmethod_func_attr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if !unsafe { pyre_object::function::is_staticmethod(obj) } {
            return Ok(pyre_object::w_none());
        }
        let w_value = unsafe { pyre_object::function::w_staticmethod_get_func(obj) };
        if w_value.is_null() {
            Ok(pyre_object::w_none())
        } else {
            Ok(w_value)
        }
    }
    dict_storage_store(
        ns,
        "__func__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__func__",
            staticmethod_func_attr,
            2,
        )),
    );
    dict_storage_store(
        ns,
        "__wrapped__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__wrapped__",
            staticmethod_func_attr,
            2,
        )),
    );
    // typedef.py:872 `__isabstractmethod__ = GetSetProperty(
    //     StaticMethod.descr_isabstract)`.  function.py:705-706:
    //
    //     def descr_isabstract(self, space):
    //         return space.newbool(space.isabstractmethod_w(self.w_function))
    //
    // `baseobjspace.isabstractmethod_w` already factored above.
    dict_storage_store(
        ns,
        "__isabstractmethod__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__isabstractmethod__",
            |args| {
                let sm = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                if !unsafe { pyre_object::function::is_staticmethod(sm) } {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let func = unsafe { pyre_object::function::w_staticmethod_get_func(sm) };
                let result = crate::baseobjspace::isabstractmethod_w(func)?;
                Ok(pyre_object::w_bool_from(result))
            },
            2,
        )),
    );
}

/// `classmethod.__new__(cls, func)` — PyPy: function.py ClassMethod.descr__new__
fn init_classmethod_type(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__new__",
        make_builtin_function("__new__", |args| {
            let func = if args.len() > 1 {
                args[1]
            } else {
                pyre_object::w_none()
            };
            Ok(pyre_object::function::w_classmethod_new(func))
        }),
    );
    // `typedef.py:883 __get__ = interp2app(
    //     ClassMethod.descr_classmethod_get)`.  `function.py:738-748`:
    //
    //     def descr_classmethod_get(self, space, w_obj, w_klass=None):
    //         if space.is_none(w_klass):
    //             w_klass = space.type(w_obj)
    //         w_func = self.w_function
    //         w_bound = space.get(w_func, w_klass, w_klass)
    //         if w_bound is not w_func:
    //             return w_bound
    //         # the object doesn't have a get, but it might still be
    //         # callable, so make a Method object
    //         return Method(space, w_func, w_klass)
    //
    // The two branches collapse into a single `Method(func, klass)`
    // construction because pyre's `w_method_new` is the same shape
    // that `Function.descr_function_get` would return when
    // `w_func.__get__(klass, klass)` fires.  This matches the
    // pre-existing hardcoded classmethod arm in
    // `baseobjspace::get` (`baseobjspace.rs:5420-5427`).
    dict_storage_store(
        ns,
        "__get__",
        make_builtin_function_with_arity(
            "__get__",
            |args| {
                let cm = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                if !unsafe { pyre_object::function::is_classmethod(cm) } {
                    return Err(crate::PyError::type_error(
                        "descriptor '__get__' requires a 'classmethod' object",
                    ));
                }
                let w_obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                let mut w_klass = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
                if w_klass.is_null() || unsafe { pyre_object::is_none(w_klass) } {
                    w_klass = crate::typedef::r#type(w_obj).unwrap_or(pyre_object::PY_NULL);
                }
                let w_func = unsafe { pyre_object::function::w_classmethod_get_func(cm) };
                Ok(pyre_object::w_method_new(w_func, w_klass, w_klass))
            },
            3,
        ),
    );
    // typedef.py:884-885 ─
    //   __func__ = interp_attrproperty_w('w_function', cls=ClassMethod),
    //   __wrapped__ = interp_attrproperty_w('w_function', cls=ClassMethod),
    fn classmethod_func_attr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if !unsafe { pyre_object::function::is_classmethod(obj) } {
            return Ok(pyre_object::w_none());
        }
        let w_value = unsafe { pyre_object::function::w_classmethod_get_func(obj) };
        if w_value.is_null() {
            Ok(pyre_object::w_none())
        } else {
            Ok(w_value)
        }
    }
    dict_storage_store(
        ns,
        "__func__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__func__",
            classmethod_func_attr,
            2,
        )),
    );
    dict_storage_store(
        ns,
        "__wrapped__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__wrapped__",
            classmethod_func_attr,
            2,
        )),
    );
    // typedef.py:886 `__isabstractmethod__ = GetSetProperty(
    //     ClassMethod.descr_isabstract)`.  function.py:760-761:
    //
    //     def descr_isabstract(self, space):
    //         return space.newbool(space.isabstractmethod_w(self.w_function))
    dict_storage_store(
        ns,
        "__isabstractmethod__",
        make_getset_descriptor(make_builtin_function_with_arity(
            "__isabstractmethod__",
            |args| {
                let cm = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                if !unsafe { pyre_object::function::is_classmethod(cm) } {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let func = unsafe { pyre_object::function::w_classmethod_get_func(cm) };
                let result = crate::baseobjspace::isabstractmethod_w(func)?;
                Ok(pyre_object::w_bool_from(result))
            },
            2,
        )),
    );
}

/// `property.__new__(cls, fget=None, fset=None, fdel=None, doc=None)`
/// — descriptor.py W_Property.descr_new
fn init_property_type(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__new__",
        make_builtin_function("__new__", |args| {
            // args[0] is cls; fget/fset/fdel/doc follow.
            // descriptor.py:186-189 `@unwrap_spec(w_fget=..., w_fset=...,
            // w_fdel=..., w_doc=...)` — keyword forms bind too.
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(
                kwargs,
                &["fget", "fset", "fdel", "doc"],
                "property",
            )?;
            let arg = |idx: usize, name: &str| {
                pos.get(idx)
                    .copied()
                    .or_else(|| crate::builtins::kwarg_get(kwargs, name))
                    .unwrap_or(pyre_object::PY_NULL)
            };
            let cls = pos.first().copied().unwrap_or(pyre_object::PY_NULL);
            let fget = arg(1, "fget");
            let fset = arg(2, "fset");
            let fdel = arg(3, "fdel");
            let w_doc = arg(4, "doc");
            let prop = pyre_object::w_property_new(fget, fset, fdel);
            // typeobject.py:511 `allocate_instance(W_Property, w_subtype)`
            // — `generic_new_descr(W_Property)` honours the subtype, so a
            // `property` subclass instance keeps its own class.
            let property_type =
                crate::typedef::gettypeobject(&pyre_object::descriptor::PROPERTY_TYPE);
            if !cls.is_null() && !std::ptr::eq(cls, property_type) {
                check_user_subclass(property_type, cls)?;
                unsafe {
                    (*prop).w_class = cls;
                }
            }
            unsafe {
                // descriptor.py:193 `self.w_doc = w_doc`
                if !w_doc.is_null() && !pyre_object::is_none(w_doc) {
                    pyre_object::descriptor::w_property_set_doc(prop, w_doc);
                } else if !fget.is_null() && !pyre_object::is_none(fget) {
                    // descriptor.py:195-204 — without an explicit doc,
                    // inherit `fget.__doc__` and mark `getter_doc`.
                    // (The subclass `space.setattr` branch at :202-203
                    // is folded into the field write: pyre property
                    // subclass instances share the W_Property
                    // layout, so the slot is the only storage.)
                    if let Ok(getter_doc) = crate::baseobjspace::getattr_str(fget, "__doc__") {
                        if !getter_doc.is_null() && !pyre_object::is_none(getter_doc) {
                            pyre_object::descriptor::w_property_set_getter_doc(prop, getter_doc);
                        }
                    }
                }
            }
            Ok(prop)
        }),
    );
    // descriptor.py W_Property.typedef `__get__` / `__set__` / `__delete__`
    // — the implicit descriptor path special-cases properties, but the
    // type-dict entries make `prop.__get__`, `hasattr(prop, '__get__')`,
    // and `_is_descriptor` see the property as a descriptor.
    dict_storage_store(
        ns,
        "__get__",
        make_builtin_function("__get__", crate::baseobjspace::property_descr_get_impl),
    );
    dict_storage_store(
        ns,
        "__set__",
        make_builtin_function("__set__", crate::baseobjspace::property_descr_set_impl),
    );
    dict_storage_store(
        ns,
        "__delete__",
        make_builtin_function(
            "__delete__",
            crate::baseobjspace::property_descr_delete_impl,
        ),
    );
}

/// `self` as a plain int — `int.real` / `numerator` / `conjugate` /
/// `as_integer_ratio` return the integer value, so a `bool` receiver
/// yields `1` / `0` rather than itself.
fn int_as_plain_int(args: &[PyObjectRef]) -> PyObjectRef {
    let obj = args.first().copied().unwrap_or(pyre_object::w_int_new(0));
    unsafe {
        if pyre_object::is_bool(obj) {
            return pyre_object::w_int_new(pyre_object::w_bool_get_value(obj) as i64);
        }
    }
    obj
}

// ── Numeric binary-op dunders ────────────────────────────────────────
// Each computes the concrete int/long/float result when the operand is
// numerically compatible, else returns NotImplemented so the interpreter
// can try the reflected method.  These resolve to the `*_builtin`
// type-slot computations, not the operator-level dispatch — a slot wired
// to the operator would re-enter it when the other operand is a numeric
// subclass that overrides the special method, and recurse without bound.
macro_rules! int_binop_fwd {
    ($name:ident, $op:path) => {
        fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::arity_slot(args, 1)?;
            if unsafe { pyre_object::pyobject::is_int_or_long(args[1]) } {
                $op(args[0], args[1])
            } else {
                Ok(pyre_object::w_not_implemented())
            }
        }
    };
}
macro_rules! int_binop_rev {
    ($name:ident, $op:path) => {
        fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::arity_slot(args, 1)?;
            if unsafe { pyre_object::pyobject::is_int_or_long(args[1]) } {
                $op(args[1], args[0])
            } else {
                Ok(pyre_object::w_not_implemented())
            }
        }
    };
}
macro_rules! float_binop_fwd {
    ($name:ident, $op:path) => {
        fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::arity_slot(args, 1)?;
            let b = args[1];
            if unsafe {
                pyre_object::pyobject::is_float(b) || pyre_object::pyobject::is_int_or_long(b)
            } {
                $op(args[0], b)
            } else {
                Ok(pyre_object::w_not_implemented())
            }
        }
    };
}
macro_rules! float_binop_rev {
    ($name:ident, $op:path) => {
        fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::arity_slot(args, 1)?;
            let b = args[1];
            if unsafe {
                pyre_object::pyobject::is_float(b) || pyre_object::pyobject::is_int_or_long(b)
            } {
                $op(b, args[0])
            } else {
                Ok(pyre_object::w_not_implemented())
            }
        }
    };
}
fn complex_binop_operand(b: PyObjectRef) -> bool {
    unsafe {
        pyre_object::is_complex(b)
            || pyre_object::is_float(b)
            || pyre_object::pyobject::is_int_or_long(b)
            || pyre_object::is_bool(b)
    }
}
macro_rules! complex_binop_fwd {
    ($name:ident, $op:path) => {
        fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::arity_slot(args, 1)?;
            if complex_binop_operand(args[1]) {
                $op(args[0], args[1])
            } else {
                Ok(pyre_object::w_not_implemented())
            }
        }
    };
}
macro_rules! complex_binop_rev {
    ($name:ident, $op:path) => {
        fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::arity_slot(args, 1)?;
            if complex_binop_operand(args[1]) {
                $op(args[1], args[0])
            } else {
                Ok(pyre_object::w_not_implemented())
            }
        }
    };
}

int_binop_fwd!(int_dunder_add, crate::objspace::descroperation::add_builtin);
int_binop_rev!(
    int_dunder_radd,
    crate::objspace::descroperation::add_builtin
);
int_binop_fwd!(int_dunder_sub, crate::objspace::descroperation::sub_builtin);
int_binop_rev!(
    int_dunder_rsub,
    crate::objspace::descroperation::sub_builtin
);
int_binop_fwd!(int_dunder_mul, crate::objspace::descroperation::mul_builtin);
int_binop_rev!(
    int_dunder_rmul,
    crate::objspace::descroperation::mul_builtin
);
int_binop_fwd!(
    int_dunder_truediv,
    crate::objspace::descroperation::truediv_builtin
);
int_binop_rev!(
    int_dunder_rtruediv,
    crate::objspace::descroperation::truediv_builtin
);
int_binop_fwd!(
    int_dunder_floordiv,
    crate::objspace::descroperation::floordiv_builtin
);
int_binop_rev!(
    int_dunder_rfloordiv,
    crate::objspace::descroperation::floordiv_builtin
);
int_binop_fwd!(int_dunder_mod, crate::objspace::descroperation::mod_builtin);
int_binop_rev!(
    int_dunder_rmod,
    crate::objspace::descroperation::mod_builtin
);
int_binop_fwd!(
    int_dunder_divmod,
    crate::objspace::descroperation::divmod_builtin
);
int_binop_rev!(
    int_dunder_rdivmod,
    crate::objspace::descroperation::divmod_builtin
);
/// `int.__rpow__(self, base[, mod])` — the reflected slot accepts an
/// optional modulus argument, so it validates arity as one-or-two.
fn int_dunder_rpow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_pow(args)?;
    if unsafe { pyre_object::pyobject::is_int_or_long(args[1]) } {
        crate::objspace::descroperation::pow_builtin(args[1], args[0])
    } else {
        Ok(pyre_object::w_not_implemented())
    }
}
int_binop_fwd!(
    int_dunder_lshift,
    crate::objspace::descroperation::lshift_builtin
);
int_binop_rev!(
    int_dunder_rlshift,
    crate::objspace::descroperation::lshift_builtin
);
int_binop_fwd!(
    int_dunder_rshift,
    crate::objspace::descroperation::rshift_builtin
);
int_binop_rev!(
    int_dunder_rrshift,
    crate::objspace::descroperation::rshift_builtin
);
int_binop_fwd!(int_dunder_and, crate::objspace::descroperation::and_builtin);
int_binop_rev!(
    int_dunder_rand,
    crate::objspace::descroperation::and_builtin
);
int_binop_fwd!(int_dunder_or, crate::objspace::descroperation::or_builtin);
int_binop_rev!(int_dunder_ror, crate::objspace::descroperation::or_builtin);
int_binop_fwd!(int_dunder_xor, crate::objspace::descroperation::xor_builtin);
int_binop_rev!(
    int_dunder_rxor,
    crate::objspace::descroperation::xor_builtin
);

/// `int.__pow__(self, exp[, mod])` — optional modulus routes through the
/// three-argument modular power.
fn int_dunder_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_pow(args)?;
    if unsafe { pyre_object::pyobject::is_int_or_long(args[1]) } {
        if args.len() >= 3 {
            if unsafe { pyre_object::pyobject::is_none(args[2]) } {
                crate::objspace::descroperation::pow_builtin(args[0], args[1])
            } else {
                crate::objspace::descroperation::pow3(args[0], args[1], args[2])
            }
        } else {
            crate::objspace::descroperation::pow_builtin(args[0], args[1])
        }
    } else {
        Ok(pyre_object::w_not_implemented())
    }
}

float_binop_fwd!(
    float_dunder_add,
    crate::objspace::descroperation::add_builtin
);
float_binop_rev!(
    float_dunder_radd,
    crate::objspace::descroperation::add_builtin
);
float_binop_fwd!(
    float_dunder_sub,
    crate::objspace::descroperation::sub_builtin
);
float_binop_rev!(
    float_dunder_rsub,
    crate::objspace::descroperation::sub_builtin
);
float_binop_fwd!(
    float_dunder_mul,
    crate::objspace::descroperation::mul_builtin
);
float_binop_rev!(
    float_dunder_rmul,
    crate::objspace::descroperation::mul_builtin
);
float_binop_fwd!(
    float_dunder_truediv,
    crate::objspace::descroperation::truediv_builtin
);
float_binop_rev!(
    float_dunder_rtruediv,
    crate::objspace::descroperation::truediv_builtin
);
float_binop_fwd!(
    float_dunder_floordiv,
    crate::objspace::descroperation::floordiv_builtin
);
float_binop_rev!(
    float_dunder_rfloordiv,
    crate::objspace::descroperation::floordiv_builtin
);
float_binop_fwd!(
    float_dunder_mod,
    crate::objspace::descroperation::mod_builtin
);
float_binop_rev!(
    float_dunder_rmod,
    crate::objspace::descroperation::mod_builtin
);
float_binop_fwd!(
    float_dunder_divmod,
    crate::objspace::descroperation::divmod_builtin
);
float_binop_rev!(
    float_dunder_rdivmod,
    crate::objspace::descroperation::divmod_builtin
);
/// `float.__pow__` / `__rpow__` — the ternary-power slot accepts an
/// optional modulus argument, so arity is validated as one-or-two.
fn float_dunder_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_pow(args)?;
    let b = args[1];
    if unsafe { pyre_object::pyobject::is_float(b) || pyre_object::pyobject::is_int_or_long(b) } {
        crate::objspace::descroperation::pow_builtin(args[0], b)
    } else {
        Ok(pyre_object::w_not_implemented())
    }
}
fn float_dunder_rpow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_pow(args)?;
    let b = args[1];
    if unsafe { pyre_object::pyobject::is_float(b) || pyre_object::pyobject::is_int_or_long(b) } {
        crate::objspace::descroperation::pow_builtin(b, args[0])
    } else {
        Ok(pyre_object::w_not_implemented())
    }
}

complex_binop_fwd!(
    complex_dunder_add,
    crate::objspace::descroperation::add_builtin
);
complex_binop_rev!(
    complex_dunder_radd,
    crate::objspace::descroperation::add_builtin
);
complex_binop_fwd!(
    complex_dunder_sub,
    crate::objspace::descroperation::sub_builtin
);
complex_binop_rev!(
    complex_dunder_rsub,
    crate::objspace::descroperation::sub_builtin
);
complex_binop_fwd!(
    complex_dunder_mul,
    crate::objspace::descroperation::mul_builtin
);
complex_binop_rev!(
    complex_dunder_rmul,
    crate::objspace::descroperation::mul_builtin
);
complex_binop_fwd!(
    complex_dunder_truediv,
    crate::objspace::descroperation::truediv_builtin
);
complex_binop_rev!(
    complex_dunder_rtruediv,
    crate::objspace::descroperation::truediv_builtin
);
/// `complex.__pow__` / `__rpow__` — the ternary-power slot accepts an
/// optional modulus argument, so arity is validated as one-or-two.
fn complex_dunder_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_pow(args)?;
    if complex_binop_operand(args[1]) {
        crate::objspace::descroperation::pow_builtin(args[0], args[1])
    } else {
        Ok(pyre_object::w_not_implemented())
    }
}
fn complex_dunder_rpow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_pow(args)?;
    if complex_binop_operand(args[1]) {
        crate::objspace::descroperation::pow_builtin(args[1], args[0])
    } else {
        Ok(pyre_object::w_not_implemented())
    }
}

// Rich comparison dunders (`__eq__` / `__ne__` / `__lt__` / `__le__` /
// `__gt__` / `__ge__`).  Each built-in numeric / sequence type only
// compares against operands of an accepted type and returns
// `NotImplemented` otherwise, so the reflected comparison on the other
// operand gets a chance (`1 == 1.0` succeeds through `float.__eq__`).
// When the operand passes the guard the value comparison is delegated to
// `descroperation::compare_slot` (the direct slot body), whose matching-type
// fast paths return without re-entering override dispatch, so a subclass
// override's `super().__eq__` (etc.) resolves to the inherited comparison.
fn cmp_guard_int(b: PyObjectRef) -> bool {
    unsafe { pyre_object::pyobject::is_int_or_long(b) }
}
fn cmp_guard_float(b: PyObjectRef) -> bool {
    unsafe { pyre_object::pyobject::is_float(b) || pyre_object::pyobject::is_int_or_long(b) }
}
fn cmp_guard_complex(b: PyObjectRef) -> bool {
    complex_binop_operand(b)
}
fn cmp_guard_str(b: PyObjectRef) -> bool {
    unsafe { pyre_object::is_str(b) }
}
fn cmp_guard_list(b: PyObjectRef) -> bool {
    unsafe { pyre_object::is_list(b) }
}
fn cmp_guard_tuple(b: PyObjectRef) -> bool {
    unsafe { pyre_object::is_tuple(b) }
}
fn cmp_guard_bytes(b: PyObjectRef) -> bool {
    unsafe { pyre_object::bytesobject::is_bytes(b) }
}
fn cmp_guard_bytearray(b: PyObjectRef) -> bool {
    unsafe { pyre_object::bytesobject::is_bytes_like(b) }
}

macro_rules! cmp_dunder {
    ($name:ident, $op:ident, $guard:path) => {
        fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::arity_slot(args, 1)?;
            if $guard(args[1]) {
                crate::objspace::descroperation::compare_slot(
                    args[0],
                    args[1],
                    crate::objspace::descroperation::CompareOp::$op,
                )
            } else {
                Ok(pyre_object::w_not_implemented())
            }
        }
    };
}

macro_rules! cmp_dunder_set {
    ($eq:ident, $ne:ident, $lt:ident, $le:ident, $gt:ident, $ge:ident, $guard:path) => {
        cmp_dunder!($eq, Eq, $guard);
        cmp_dunder!($ne, Ne, $guard);
        cmp_dunder!($lt, Lt, $guard);
        cmp_dunder!($le, Le, $guard);
        cmp_dunder!($gt, Gt, $guard);
        cmp_dunder!($ge, Ge, $guard);
    };
}

cmp_dunder_set!(
    int_dunder_eq,
    int_dunder_ne,
    int_dunder_lt,
    int_dunder_le,
    int_dunder_gt,
    int_dunder_ge,
    cmp_guard_int
);
cmp_dunder_set!(
    float_dunder_eq,
    float_dunder_ne,
    float_dunder_lt,
    float_dunder_le,
    float_dunder_gt,
    float_dunder_ge,
    cmp_guard_float
);
cmp_dunder!(complex_dunder_eq, Eq, cmp_guard_complex);
cmp_dunder!(complex_dunder_ne, Ne, cmp_guard_complex);
// complexobject.py:459 `_fail_cmp` — complex defines no ordering, so
// __lt__/__le__/__gt__/__ge__ return NotImplemented; the `<` operator then
// raises TypeError through the comparison fallback.
macro_rules! complex_fail_cmp {
    ($name:ident) => {
        fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::arity_slot(args, 1)?;
            Ok(pyre_object::w_not_implemented())
        }
    };
}
complex_fail_cmp!(complex_dunder_lt);
complex_fail_cmp!(complex_dunder_le);
complex_fail_cmp!(complex_dunder_gt);
complex_fail_cmp!(complex_dunder_ge);
cmp_dunder_set!(
    str_dunder_eq,
    str_dunder_ne,
    str_dunder_lt,
    str_dunder_le,
    str_dunder_gt,
    str_dunder_ge,
    cmp_guard_str
);
cmp_dunder_set!(
    list_dunder_eq,
    list_dunder_ne,
    list_dunder_lt,
    list_dunder_le,
    list_dunder_gt,
    list_dunder_ge,
    cmp_guard_list
);
cmp_dunder_set!(
    tuple_dunder_eq,
    tuple_dunder_ne,
    tuple_dunder_lt,
    tuple_dunder_le,
    tuple_dunder_gt,
    tuple_dunder_ge,
    cmp_guard_tuple
);
cmp_dunder_set!(
    bytes_dunder_eq,
    bytes_dunder_ne,
    bytes_dunder_lt,
    bytes_dunder_le,
    bytes_dunder_gt,
    bytes_dunder_ge,
    cmp_guard_bytes
);
cmp_dunder_set!(
    bytearray_dunder_eq,
    bytearray_dunder_ne,
    bytearray_dunder_lt,
    bytearray_dunder_le,
    bytearray_dunder_gt,
    bytearray_dunder_ge,
    cmp_guard_bytearray
);

type DunderFn = fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>;

fn init_int_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(int_descr_new));
    dict_storage_store(
        ns,
        "bit_length",
        make_builtin_function_with_arity(
            "bit_length",
            |args| {
                // `intobject.py descr_bit_length` — number of bits in the
                // absolute value, so long/bigint operands must route
                // through their magnitude rather than the i64 fast path
                // (which leaves out-of-range values at 0).
                let bits = if !args.is_empty()
                    && unsafe { pyre_object::pyobject::is_int_or_long(args[0]) }
                {
                    unsafe { crate::builtins::obj_to_bigint(args[0]).bits() }
                } else {
                    0
                };
                Ok(pyre_object::w_int_new(bits as i64))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "bit_count",
        // PyPy `intobject.py:649-650 descr_bit_count` parity:
        // `space.newint(_bit_count(self.intval))`.  Routes through
        // `pyre_object::int_bit_count` (`@jit.elidable` parity port of
        // `_bit_count`) so the call graph matches upstream
        // `descr_bit_count -> _bit_count` 1:1.
        make_builtin_function_with_arity(
            "bit_count",
            |args| {
                let count = if args.is_empty() {
                    0
                } else if unsafe { pyre_object::is_int(args[0]) } {
                    // Small-int fast path — `@jit.elidable` `_bit_count`.
                    pyre_object::int_bit_count(unsafe { pyre_object::w_int_get_value(args[0]) })
                } else if unsafe { pyre_object::pyobject::is_int_or_long(args[0]) } {
                    // long/bigint: population count of the magnitude, so the
                    // i64 fast path (which leaves out-of-range values at 0)
                    // does not undercount.
                    unsafe {
                        crate::builtins::obj_to_bigint(args[0])
                            .iter_u32_digits()
                            .map(|d| d.count_ones() as i64)
                            .sum()
                    }
                } else {
                    0
                };
                Ok(pyre_object::w_int_new(count))
            },
            1,
        ),
    );
    // int.to_bytes(length=1, byteorder='big', *, signed=False)
    // PyPy: longobject.py descr_to_bytes
    dict_storage_store(
        ns,
        "to_bytes",
        make_builtin_function("to_bytes", |args| {
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(
                kwargs,
                &["length", "byteorder", "signed"],
                "to_bytes",
            )?;
            crate::builtins::kwarg_reject_duplicate(
                kwargs,
                "to_bytes",
                "length",
                pos.get(1).is_some(),
            )?;
            crate::builtins::kwarg_reject_duplicate(
                kwargs,
                "to_bytes",
                "byteorder",
                pos.get(2).is_some(),
            )?;
            if pos.len() > 3 {
                return Err(crate::PyError::type_error(format!(
                    "to_bytes() takes at most 2 positional arguments ({} given)",
                    pos.len() - 1
                )));
            }
            let val = if !pos.is_empty() && unsafe { pyre_object::pyobject::is_int_or_long(pos[0]) }
            {
                unsafe { crate::builtins::obj_to_bigint(pos[0]) }
            } else {
                malachite_bigint::BigInt::from(0)
            };
            let length_obj = pos
                .get(1)
                .copied()
                .or_else(|| crate::builtins::kwarg_get(kwargs, "length"));
            let length_i = match length_obj {
                Some(o) => crate::builtins::space_index_w(o)?,
                None => 1,
            };
            if length_i < 0 {
                return Err(crate::PyError::value_error(
                    "length argument must be non-negative",
                ));
            }
            let length = length_i as usize;
            let little_endian = match pos
                .get(2)
                .copied()
                .or_else(|| crate::builtins::kwarg_get(kwargs, "byteorder"))
            {
                None => false,
                Some(o) if unsafe { pyre_object::is_str(o) } => {
                    match unsafe { pyre_object::w_str_get_value(o) } {
                        "little" => true,
                        "big" => false,
                        _ => {
                            return Err(crate::PyError::value_error(
                                "byteorder must be either 'little' or 'big'",
                            ));
                        }
                    }
                }
                Some(o) => {
                    return Err(crate::PyError::type_error(format!(
                        "expected str, got {} object",
                        unsafe { (*(*o).ob_type).name }
                    )));
                }
            };
            let signed = crate::builtins::kwarg_get(kwargs, "signed")
                .map(crate::baseobjspace::is_true)
                .transpose()?
                .unwrap_or(false);
            let bits = length * 8;
            let zero = malachite_bigint::BigInt::from(0);
            let limit = malachite_bigint::BigInt::from(1) << bits;
            let encoded = if bits == 0 {
                if val != zero {
                    return Err(crate::PyError::overflow_error("int too big to convert"));
                }
                zero.clone()
            } else if signed {
                let half = if bits == 0 {
                    malachite_bigint::BigInt::from(0)
                } else {
                    malachite_bigint::BigInt::from(1) << (bits - 1)
                };
                if val < -half.clone() || val >= half {
                    return Err(crate::PyError::overflow_error("int too big to convert"));
                }
                if val < zero { val + &limit } else { val }
            } else {
                if val < zero {
                    return Err(crate::PyError::overflow_error(
                        "can't convert negative int to unsigned",
                    ));
                }
                if val >= limit {
                    return Err(crate::PyError::overflow_error("int too big to convert"));
                }
                val
            };
            let mut bytes = vec![0u8; length];
            use num_traits::ToPrimitive;
            for i in 0..length {
                let shift = if little_endian { i } else { length - 1 - i } * 8;
                let byte = (&encoded >> shift) & malachite_bigint::BigInt::from(0xff);
                bytes[i] = byte.to_u8().unwrap_or(0);
            }
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
        }),
    );
    // int.from_bytes(bytes, byteorder='big', *, signed=False) — classmethod.
    dict_storage_store(
        ns,
        "from_bytes",
        pyre_object::function::w_classmethod_new(make_builtin_function(
            "from_bytes",
            int_from_bytes,
        )),
    );
    // int.__index__ / __int__ / __trunc__ — identity
    for method in ["__index__", "__int__", "__trunc__"] {
        dict_storage_store(
            ns,
            method,
            make_builtin_function_with_arity(
                method,
                |args| Ok(args.first().copied().unwrap_or(pyre_object::w_int_new(0))),
                1,
            ),
        );
    }
    // int.conjugate — identity (bool → int)
    dict_storage_store(
        ns,
        "conjugate",
        make_builtin_function_with_arity("conjugate", |args| Ok(int_as_plain_int(args)), 1),
    );
    // int.as_integer_ratio — (self, 1)
    dict_storage_store(
        ns,
        "as_integer_ratio",
        make_builtin_function_with_arity(
            "as_integer_ratio",
            |args| {
                Ok(pyre_object::w_tuple_new(vec![
                    int_as_plain_int(args),
                    pyre_object::w_int_new(1),
                ]))
            },
            1,
        ),
    );
    // int.real / int.imag / int.numerator — properties
    // True.real → 1 (int, not bool), False.real → 0
    dict_storage_store(
        ns,
        "real",
        pyre_object::w_property_new(
            make_builtin_function_with_arity("real", |args| Ok(int_as_plain_int(args)), 1),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
    );
    dict_storage_store(
        ns,
        "imag",
        pyre_object::w_property_new(
            make_builtin_function_with_arity("imag", |_| Ok(pyre_object::w_int_new(0)), 1),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
    );
    dict_storage_store(
        ns,
        "numerator",
        pyre_object::w_property_new(
            make_builtin_function_with_arity("numerator", |args| Ok(int_as_plain_int(args)), 1),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
    );
    let denom_getter =
        make_builtin_function_with_arity("denominator", |_| Ok(pyre_object::w_int_new(1)), 1);
    dict_storage_store(ns, "denominator", make_getset_descriptor(denom_getter));
    // Unary / conversion slots exposed as callable dunders.  These have
    // no NotImplemented dispatch, so each delegates to the object-space
    // op, which fast-paths the concrete int (no re-dispatch through the
    // dunder).  Binary arithmetic dunders are registered separately.
    dict_storage_store(
        ns,
        "__round__",
        make_builtin_function("__round__", crate::builtins::builtin_round),
    );
    dict_storage_store(
        ns,
        "__format__",
        make_builtin_function_with_arity(
            "__format__",
            crate::type_methods::builtin_value_format,
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__float__",
        make_builtin_function_with_arity("__float__", crate::builtins::builtin_float, 1),
    );
    dict_storage_store(
        ns,
        "__abs__",
        make_builtin_function_with_arity("__abs__", crate::builtins::builtin_abs, 1),
    );
    dict_storage_store(
        ns,
        "__neg__",
        make_builtin_function_with_arity(
            "__neg__",
            |args| crate::objspace::descroperation::neg(args[0]),
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__pos__",
        make_builtin_function_with_arity(
            "__pos__",
            |args| crate::objspace::descroperation::pos(args[0]),
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__invert__",
        make_builtin_function_with_arity(
            "__invert__",
            |args| crate::objspace::descroperation::invert(args[0]),
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__bool__",
        make_builtin_function_with_arity(
            "__bool__",
            |args| {
                Ok(pyre_object::w_bool_from(crate::baseobjspace::is_true_slot(
                    args[0],
                )?))
            },
            1,
        ),
    );
    // `int.__floor__` / `int.__ceil__` return the int itself.
    dict_storage_store(
        ns,
        "__floor__",
        make_builtin_function_with_arity("__floor__", |args| Ok(args[0]), 1),
    );
    dict_storage_store(
        ns,
        "__ceil__",
        make_builtin_function_with_arity("__ceil__", |args| Ok(args[0]), 1),
    );
    // Binary arithmetic / bitwise dunders (forward + reflected).
    for (name, func) in [
        ("__add__", int_dunder_add as DunderFn),
        ("__radd__", int_dunder_radd),
        ("__sub__", int_dunder_sub),
        ("__rsub__", int_dunder_rsub),
        ("__mul__", int_dunder_mul),
        ("__rmul__", int_dunder_rmul),
        ("__truediv__", int_dunder_truediv),
        ("__rtruediv__", int_dunder_rtruediv),
        ("__floordiv__", int_dunder_floordiv),
        ("__rfloordiv__", int_dunder_rfloordiv),
        ("__mod__", int_dunder_mod),
        ("__rmod__", int_dunder_rmod),
        ("__divmod__", int_dunder_divmod),
        ("__rdivmod__", int_dunder_rdivmod),
        ("__rpow__", int_dunder_rpow),
        ("__lshift__", int_dunder_lshift),
        ("__rlshift__", int_dunder_rlshift),
        ("__rshift__", int_dunder_rshift),
        ("__rrshift__", int_dunder_rrshift),
        ("__and__", int_dunder_and),
        ("__rand__", int_dunder_rand),
        ("__or__", int_dunder_or),
        ("__ror__", int_dunder_ror),
        ("__xor__", int_dunder_xor),
        ("__rxor__", int_dunder_rxor),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
    // `__pow__` takes an optional modulus, so it is variadic.
    dict_storage_store(
        ns,
        "__pow__",
        make_builtin_function("__pow__", int_dunder_pow),
    );
    for (name, func) in [
        ("__eq__", int_dunder_eq as DunderFn),
        ("__ne__", int_dunder_ne),
        ("__lt__", int_dunder_lt),
        ("__le__", int_dunder_le),
        ("__gt__", int_dunder_gt),
        ("__ge__", int_dunder_ge),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
    // intobject.py descr_getnewargs — `(wrapint(self.intval),)`: a fresh
    // plain int from the value, so an int subclass (e.g. bool) reduces to
    // the base int.
    dict_storage_store(
        ns,
        "__getnewargs__",
        make_builtin_function_with_arity(
            "__getnewargs__",
            |args| {
                let v = unsafe { pyre_object::w_int_get_value(args[0]) };
                Ok(pyre_object::w_tuple_new(vec![pyre_object::w_int_new(v)]))
            },
            1,
        ),
    );
}
/// Complex `repr` (`Xj` for a pure-`+0` real part, else `(re±imj)`),
/// delegated to `rustpython_literal::complex::to_string`.
pub(crate) fn complex_repr_string(re: f64, im: f64) -> String {
    rustpython_literal::complex::to_string(re, im)
}

fn init_complex_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(complex_descr_new));
    let repr = |args: &[PyObjectRef]| {
        let (re, im) = unsafe {
            (
                pyre_object::w_complex_get_real(args[0]),
                pyre_object::w_complex_get_imag(args[0]),
            )
        };
        Ok(pyre_object::w_str_new(&complex_repr_string(re, im)))
    };
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function_with_arity("__repr__", repr, 1),
    );
    dict_storage_store(
        ns,
        "__str__",
        make_builtin_function_with_arity("__str__", repr, 1),
    );
    dict_storage_store(
        ns,
        "__format__",
        make_builtin_function_with_arity(
            "__format__",
            crate::type_methods::builtin_value_format,
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__hash__",
        make_builtin_function_with_arity(
            "__hash__",
            |args| {
                let (re, im) = unsafe {
                    (
                        pyre_object::w_complex_get_real(args[0]),
                        pyre_object::w_complex_get_imag(args[0]),
                    )
                };
                Ok(pyre_object::w_int_new(
                    crate::objspace::descroperation::complex_hash(re, im),
                ))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__bool__",
        make_builtin_function_with_arity(
            "__bool__",
            |args| {
                let (re, im) = unsafe {
                    (
                        pyre_object::w_complex_get_real(args[0]),
                        pyre_object::w_complex_get_imag(args[0]),
                    )
                };
                Ok(pyre_object::w_bool_from(re != 0.0 || im != 0.0))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__abs__",
        make_builtin_function_with_arity(
            "__abs__",
            |args| {
                let (re, im) = unsafe {
                    (
                        pyre_object::w_complex_get_real(args[0]),
                        pyre_object::w_complex_get_imag(args[0]),
                    )
                };
                Ok(pyre_object::w_float_new(re.hypot(im)))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__neg__",
        make_builtin_function_with_arity(
            "__neg__",
            |args| crate::objspace::descroperation::neg(args[0]),
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__pos__",
        make_builtin_function_with_arity(
            "__pos__",
            |args| crate::objspace::descroperation::pos(args[0]),
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__complex__",
        make_builtin_function_with_arity(
            "__complex__",
            |args| {
                // Return a plain `complex` with the same components.
                let (re, im) = unsafe {
                    (
                        pyre_object::w_complex_get_real(args[0]),
                        pyre_object::w_complex_get_imag(args[0]),
                    )
                };
                Ok(pyre_object::w_complex_new(re, im))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "conjugate",
        make_builtin_function_with_arity(
            "conjugate",
            |args| {
                let (re, im) = unsafe {
                    (
                        pyre_object::w_complex_get_real(args[0]),
                        pyre_object::w_complex_get_imag(args[0]),
                    )
                };
                Ok(pyre_object::w_complex_new(re, -im))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__getnewargs__",
        make_builtin_function_with_arity(
            "__getnewargs__",
            |args| {
                let (re, im) = unsafe {
                    (
                        pyre_object::w_complex_get_real(args[0]),
                        pyre_object::w_complex_get_imag(args[0]),
                    )
                };
                Ok(pyre_object::w_tuple_new(vec![pyre_object::w_complex_new(
                    re, im,
                )]))
            },
            1,
        ),
    );
    // complex.real / complex.imag — read-only float components.
    dict_storage_store(
        ns,
        "real",
        pyre_object::w_property_new(
            make_builtin_function_with_arity(
                "real",
                |args| {
                    Ok(pyre_object::w_float_new(unsafe {
                        pyre_object::w_complex_get_real(args[0])
                    }))
                },
                1,
            ),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
    );
    dict_storage_store(
        ns,
        "imag",
        pyre_object::w_property_new(
            make_builtin_function_with_arity(
                "imag",
                |args| {
                    Ok(pyre_object::w_float_new(unsafe {
                        pyre_object::w_complex_get_imag(args[0])
                    }))
                },
                1,
            ),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
    );
    for (name, func) in [
        ("__add__", complex_dunder_add as DunderFn),
        ("__radd__", complex_dunder_radd),
        ("__sub__", complex_dunder_sub),
        ("__rsub__", complex_dunder_rsub),
        ("__mul__", complex_dunder_mul),
        ("__rmul__", complex_dunder_rmul),
        ("__truediv__", complex_dunder_truediv),
        ("__rtruediv__", complex_dunder_rtruediv),
        ("__pow__", complex_dunder_pow),
        ("__rpow__", complex_dunder_rpow),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
    for (name, func) in [
        ("__eq__", complex_dunder_eq as DunderFn),
        ("__ne__", complex_dunder_ne),
        ("__lt__", complex_dunder_lt),
        ("__le__", complex_dunder_le),
        ("__gt__", complex_dunder_gt),
        ("__ge__", complex_dunder_ge),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
}

fn init_float_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(float_descr_new));
    // float.__getformat__(kind) → returns the format string for the
    // given kind. PyPy: floatobject.py W_FloatObject.descr__getformat__.
    // Both 'double' and 'float' are IEEE 754 little-endian on x86/ARM.
    dict_storage_store(
        ns,
        "__getformat__",
        make_builtin_function("__getformat__", |args| {
            // Python classmethod signature: float.__getformat__(kind).
            // pyre may pass either (kind,) or (self, kind); accept both by
            // scanning for the first str argument.
            let kind = args
                .iter()
                .find_map(|&a| unsafe {
                    if pyre_object::is_str(a) {
                        Some(pyre_object::w_str_get_value(a).to_string())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    crate::PyError::type_error(
                        "__getformat__() argument must be 'double' or 'float'",
                    )
                })?;
            match kind.as_str() {
                "double" | "float" => Ok(pyre_object::w_str_new("IEEE, little-endian")),
                _ => Err(crate::PyError::value_error(
                    "__getformat__() argument must be 'double' or 'float'",
                )),
            }
        }),
    );
    dict_storage_store(
        ns,
        "hex",
        make_builtin_function_with_arity(
            "hex",
            |args| {
                // float.hex() — floatobject.c float_hex.  C99 hex-float
                // literal round-trippable through float.fromhex.
                if args.is_empty() {
                    return Err(crate::PyError::type_error("hex() requires self"));
                }
                let v = unsafe { pyre_object::w_float_get_value(args[0]) };
                Ok(pyre_object::w_str_new(&float_hex_repr(v)))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "fromhex",
        make_builtin_function("fromhex", |args| {
            // float.fromhex(s) — PyPy: floatobject.py descr_fromhex.
            // Parse hexadecimal floating-point literals like '0x1.8p3'.
            let s_arg = args
                .iter()
                .find_map(|&a| unsafe {
                    if pyre_object::is_str(a) {
                        Some(pyre_object::w_str_get_value(a).to_string())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    crate::PyError::type_error("fromhex() requires a string argument")
                })?;
            // Delegate parsing to the shared hex-float reader, which rounds
            // round-half-even over the full exponent range (subnormals down to
            // 0x1p-1074), accepts the inf/nan spellings, handles surrounding
            // ASCII whitespace itself, and flags overflow distinctly.
            match rustpython_common::float_ops::from_hex(&s_arg) {
                Ok(v) => Ok(pyre_object::w_float_new(v)),
                Err(e) => {
                    use rustpython_common::float_ops::HexFloatError;
                    Err(match e {
                        HexFloatError::Overflow => crate::PyError::overflow_error(
                            "hexadecimal value too large to represent as a float",
                        ),
                        HexFloatError::TooLong => {
                            crate::PyError::value_error("hexadecimal string too long to convert")
                        }
                        HexFloatError::Invalid => {
                            crate::PyError::value_error("invalid hexadecimal floating-point string")
                        }
                    })
                }
            }
        }),
    );
    dict_storage_store(
        ns,
        "is_integer",
        make_builtin_function_with_arity(
            "is_integer",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let v = unsafe { pyre_object::w_float_get_value(args[0]) };
                Ok(pyre_object::w_bool_from(v.is_finite() && v == v.trunc()))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "as_integer_ratio",
        make_builtin_function_with_arity(
            "as_integer_ratio",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "as_integer_ratio() requires self",
                    ));
                }
                let v = unsafe { pyre_object::w_float_get_value(args[0]) };
                // Exact numerator/denominator via the shared rational
                // decomposition (full exponent range, reduced to lowest terms).
                let (numer, denom) =
                    rustpython_common::int::float_to_ratio(v).ok_or_else(|| {
                        if v.is_infinite() {
                            crate::PyError::overflow_error(
                                "cannot convert Infinity to integer ratio",
                            )
                        } else {
                            crate::PyError::value_error("cannot convert NaN to integer ratio")
                        }
                    })?;
                let to_pyint = |b: malachite_bigint::BigInt| {
                    if pyre_object::jit_bigint_to_i64_fits(&b) != 0 {
                        pyre_object::w_int_new(pyre_object::jit_bigint_to_i64_value(&b))
                    } else {
                        pyre_object::w_long_new(b)
                    }
                };
                Ok(pyre_object::w_tuple_new(vec![
                    to_pyint(numer),
                    to_pyint(denom),
                ]))
            },
            1,
        ),
    );
    // float.conjugate — identity for a real number.
    dict_storage_store(
        ns,
        "conjugate",
        make_builtin_function_with_arity(
            "conjugate",
            |args| {
                Ok(args
                    .first()
                    .copied()
                    .unwrap_or(pyre_object::w_float_new(0.0)))
            },
            1,
        ),
    );
    // float.real / float.imag — a float is its own real part; imag is 0.0.
    dict_storage_store(
        ns,
        "real",
        pyre_object::w_property_new(
            make_builtin_function_with_arity(
                "real",
                |args| {
                    Ok(args
                        .first()
                        .copied()
                        .unwrap_or(pyre_object::w_float_new(0.0)))
                },
                1,
            ),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
    );
    dict_storage_store(
        ns,
        "imag",
        pyre_object::w_property_new(
            make_builtin_function_with_arity("imag", |_| Ok(pyre_object::w_float_new(0.0)), 1),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ),
    );
    // floatobject.py:713/715/449-455 — __int__/__trunc__ go through
    // descr_trunc (truncate-toward-zero), __floor__ / __ceil__ run
    // math.floor/ceil first, then newint_from_float.
    fn float_trunc_method(
        args: &[*mut pyre_object::PyObject],
    ) -> Result<*mut pyre_object::PyObject, crate::PyError> {
        if args.is_empty() {
            return Err(crate::PyError::type_error("__trunc__() requires self"));
        }
        let v = unsafe { pyre_object::w_float_get_value(args[0]) };
        float_to_pyint(v, FloatToIntMode::Trunc)
    }
    fn float_int_method(
        args: &[*mut pyre_object::PyObject],
    ) -> Result<*mut pyre_object::PyObject, crate::PyError> {
        if args.is_empty() {
            return Err(crate::PyError::type_error("__int__() requires self"));
        }
        let v = unsafe { pyre_object::w_float_get_value(args[0]) };
        float_to_pyint(v, FloatToIntMode::Trunc)
    }
    fn float_floor_method(
        args: &[*mut pyre_object::PyObject],
    ) -> Result<*mut pyre_object::PyObject, crate::PyError> {
        if args.is_empty() {
            return Err(crate::PyError::type_error("__floor__() requires self"));
        }
        let v = unsafe { pyre_object::w_float_get_value(args[0]) };
        float_to_pyint(v, FloatToIntMode::Floor)
    }
    fn float_ceil_method(
        args: &[*mut pyre_object::PyObject],
    ) -> Result<*mut pyre_object::PyObject, crate::PyError> {
        if args.is_empty() {
            return Err(crate::PyError::type_error("__ceil__() requires self"));
        }
        let v = unsafe { pyre_object::w_float_get_value(args[0]) };
        float_to_pyint(v, FloatToIntMode::Ceil)
    }
    for (method, func) in [
        (
            "__trunc__",
            float_trunc_method
                as fn(
                    &[*mut pyre_object::PyObject],
                ) -> Result<*mut pyre_object::PyObject, crate::PyError>,
        ),
        ("__int__", float_int_method),
        ("__floor__", float_floor_method),
        ("__ceil__", float_ceil_method),
    ] {
        dict_storage_store(
            ns,
            method,
            make_builtin_function_with_arity(method, func, 1),
        );
    }
    // Unary / conversion slots exposed as callable dunders (no
    // NotImplemented dispatch).  Binary arithmetic dunders are
    // registered separately.
    dict_storage_store(
        ns,
        "__round__",
        make_builtin_function("__round__", crate::builtins::builtin_round),
    );
    dict_storage_store(
        ns,
        "__format__",
        make_builtin_function_with_arity(
            "__format__",
            crate::type_methods::builtin_value_format,
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__float__",
        make_builtin_function_with_arity("__float__", crate::builtins::builtin_float_dunder, 1),
    );
    dict_storage_store(
        ns,
        "__abs__",
        make_builtin_function_with_arity("__abs__", crate::builtins::builtin_abs, 1),
    );
    dict_storage_store(
        ns,
        "__neg__",
        make_builtin_function_with_arity(
            "__neg__",
            |args| crate::objspace::descroperation::neg(args[0]),
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__pos__",
        make_builtin_function_with_arity(
            "__pos__",
            |args| crate::objspace::descroperation::pos(args[0]),
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__bool__",
        make_builtin_function_with_arity(
            "__bool__",
            |args| {
                Ok(pyre_object::w_bool_from(crate::baseobjspace::is_true_slot(
                    args[0],
                )?))
            },
            1,
        ),
    );
    // Binary arithmetic dunders (forward + reflected).  float has no
    // bitwise ops; `__pow__` takes no modulus.
    for (name, func) in [
        ("__add__", float_dunder_add as DunderFn),
        ("__radd__", float_dunder_radd),
        ("__sub__", float_dunder_sub),
        ("__rsub__", float_dunder_rsub),
        ("__mul__", float_dunder_mul),
        ("__rmul__", float_dunder_rmul),
        ("__truediv__", float_dunder_truediv),
        ("__rtruediv__", float_dunder_rtruediv),
        ("__floordiv__", float_dunder_floordiv),
        ("__rfloordiv__", float_dunder_rfloordiv),
        ("__mod__", float_dunder_mod),
        ("__rmod__", float_dunder_rmod),
        ("__divmod__", float_dunder_divmod),
        ("__rdivmod__", float_dunder_rdivmod),
        ("__pow__", float_dunder_pow),
        ("__rpow__", float_dunder_rpow),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
    for (name, func) in [
        ("__eq__", float_dunder_eq as DunderFn),
        ("__ne__", float_dunder_ne),
        ("__lt__", float_dunder_lt),
        ("__le__", float_dunder_le),
        ("__gt__", float_dunder_gt),
        ("__ge__", float_dunder_ge),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
    // floatobject.py descr_getnewargs — `(self.descr_float(),)`: a fresh
    // plain float from the value.
    dict_storage_store(
        ns,
        "__getnewargs__",
        make_builtin_function_with_arity(
            "__getnewargs__",
            |args| {
                let v = unsafe { pyre_object::w_float_get_value(args[0]) };
                Ok(pyre_object::w_tuple_new(vec![pyre_object::w_float_new(v)]))
            },
            1,
        ),
    );
}

#[derive(Copy, Clone)]
pub(crate) enum FloatToIntMode {
    Trunc,
    Floor,
    Ceil,
}

/// `pypy/objspace/std/longobject.py:511-522 newlong_from_float` parity.
/// NaN → ValueError, ±inf → OverflowError; finite values are reduced
/// to int and materialised through the BigInt path so values outside
/// i64 range produce a long rather than saturating.
pub(crate) fn float_to_pyint(v: f64, mode: FloatToIntMode) -> Result<PyObjectRef, crate::PyError> {
    if v.is_nan() {
        return Err(crate::PyError::value_error(
            "cannot convert float NaN to integer",
        ));
    }
    if v.is_infinite() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::OverflowError,
            "cannot convert float infinity to integer",
        ));
    }
    let reduced = match mode {
        FloatToIntMode::Trunc => v.trunc(),
        FloatToIntMode::Floor => v.floor(),
        FloatToIntMode::Ceil => v.ceil(),
    };
    use num_traits::FromPrimitive;
    let big = malachite_bigint::BigInt::from_f64(reduced).expect("finite already checked");
    if pyre_object::jit_bigint_to_i64_fits(&big) != 0 {
        Ok(pyre_object::w_int_new(
            pyre_object::jit_bigint_to_i64_value(&big),
        ))
    } else {
        Ok(pyre_object::w_long_new(big))
    }
}

/// `frexp` — split `x` into mantissa `m` (`0.5 <= |m| < 1`) and
/// exponent `e` so that `x == m * 2**e`.  std has no `frexp`, so the
/// IEEE-754 bits are decomposed directly: clearing the stored exponent
/// to `0x3fe` lands the value in `[0.5, 1)`.  Subnormals are first
/// scaled into the normal range by `2**54`.
fn float_frexp(x: f64) -> (f64, i32) {
    if x == 0.0 {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i32;
    if exp_field == 0 {
        let scaled = (x * 18014398509481984.0).to_bits();
        let m_bits = (scaled & 0x800f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000;
        let e = (((scaled >> 52) & 0x7ff) as i32) - 1022 - 54;
        return (f64::from_bits(m_bits), e);
    }
    let m_bits = (bits & 0x800f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000;
    (f64::from_bits(m_bits), exp_field - 1022)
}

/// Map a 4-bit value to its lowercase hex digit.
fn hex_digit_char(d: i64) -> char {
    if d < 10 {
        (b'0' + d as u8) as char
    } else {
        (b'a' + (d - 10) as u8) as char
    }
}

/// `floatobject.c:float_hex` — render `x` as a C99 hexadecimal float
/// literal (`[-]0x1.hhhhhhhhhhhhhp±d`) round-trippable through
/// `float.fromhex`.  nan / inf reuse the ordinary float repr.
fn float_hex_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        let s = if x > 0.0 { "inf" } else { "-inf" };
        return s.to_string();
    }
    if x == 0.0 {
        let neg = x.to_bits() >> 63 == 1;
        let s = if neg { "-0x0.0p+0" } else { "0x0.0p+0" };
        return s.to_string();
    }
    let ax = if x < 0.0 { -x } else { x };
    let (mut m, mut e) = float_frexp(ax);
    // shift = 1 - max(DBL_MIN_EXP - e, 0), DBL_MIN_EXP = -1021.
    let underflow = -1021 - e;
    let shift = 1 - if underflow > 0 { underflow } else { 0 };
    m *= 2f64.powi(shift);
    e -= shift;

    let lead = m as i64;
    let mut digits = String::new();
    digits.push(hex_digit_char(lead));
    m -= lead as f64;
    digits.push('.');
    for _ in 0..13 {
        m *= 16.0;
        let d = m as i64;
        digits.push(hex_digit_char(d));
        m -= d as f64;
    }
    let (esign, eabs) = if e < 0 { ('-', -e) } else { ('+', e) };
    let sign = if x < 0.0 { "-" } else { "" };
    format!("{sign}0x{digits}p{esign}{eabs}")
}

/// IEEE 754 double decomposition into (mantissa, exponent, sign).
/// PyPy: Lib/fractions.py _decimal_to_ratio uses a similar approach.
fn integer_decode(v: f64) -> (u64, i16, i8) {
    let bits = v.to_bits();
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };
    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}
/// boolobject.py `_make_bitwise_binop` — when both operands are bool the
/// result is bool; a non-bool operand delegates to the int dunder, which
/// returns an int.  `descr_rbinop` reuses `descr_binop`, so the reflected
/// slots bind to the same function.
fn bool_bitwise_binop(
    args: &[PyObjectRef],
    bool_op: unsafe fn(PyObjectRef, PyObjectRef) -> PyObjectRef,
    int_op: fn(PyObjectRef, PyObjectRef) -> Result<PyObjectRef, crate::PyError>,
) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("expected 1 argument, got 0"));
    }
    let a = crate::baseobjspace::unwrap_cell(args[0]);
    let b = crate::baseobjspace::unwrap_cell(args[1]);
    if !unsafe { pyre_object::is_bool(b) } {
        return int_op(a, b);
    }
    Ok(unsafe { bool_op(a, b) })
}

fn bool_dunder_and(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    bool_bitwise_binop(
        args,
        pyre_object::bool_descr_and,
        crate::objspace::descroperation::and_builtin,
    )
}

fn bool_dunder_or(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    bool_bitwise_binop(
        args,
        pyre_object::bool_descr_or,
        crate::objspace::descroperation::or_builtin,
    )
}

fn bool_dunder_xor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    bool_bitwise_binop(
        args,
        pyre_object::bool_descr_xor,
        crate::objspace::descroperation::xor_builtin,
    )
}

/// `bool.__repr__` / `bool.__str__` — boolobject.c `bool_repr` returns
/// "True"/"False" instead of inheriting int's decimal formatter (`tp_str`
/// falls back to `tp_repr`, so both dunders share this).
fn bool_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let truthy = !w_self.is_null() && crate::baseobjspace::is_true(w_self)?;
    Ok(w_str_new(if truthy { "True" } else { "False" }))
}

fn init_bool_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(bool_descr_new));
    dict_storage_store(ns, "__repr__", make_builtin_function("__repr__", bool_repr));
    dict_storage_store(ns, "__str__", make_builtin_function("__str__", bool_repr));
    // boolobject.py:97-106 — bool defines its own bitwise dunders so that
    // `True & True` is `True`; int.__and__ etc. return int.
    for (and_name, rand_name, f) in [
        (
            "__and__",
            "__rand__",
            bool_dunder_and as fn(&[PyObjectRef]) -> _,
        ),
        ("__or__", "__ror__", bool_dunder_or),
        ("__xor__", "__rxor__", bool_dunder_xor),
    ] {
        dict_storage_store(ns, and_name, make_builtin_function(and_name, f));
        dict_storage_store(ns, rand_name, make_builtin_function(rand_name, f));
    }
}

// ── Object TypeDef ───────────────────────────────────────────────────
// PyPy: pypy/objspace/std/objectobject.py TypeDef("object", ...)

/// `object.__new__(cls)` — allocate a bare instance of cls.
///
/// PyPy: objectobject.py descr__new__
fn object_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(
        !args.is_empty(),
        "object.__new__() requires a type argument"
    );
    let cls = crate::baseobjspace::unwrap_cell(args[0]);
    // cls should be a W_TypeObject — create instance of it
    if unsafe { is_type(cls) } {
        return Ok(w_instance_new(cls));
    }
    // Fallback: create bare instance with no type
    Ok(w_instance_new(PY_NULL))
}

/// `object.__init__(self)` — no-op base __init__.
fn object_descr_init(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_none())
}

fn init_object_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(object_descr_new));
    dict_storage_store(
        ns,
        "__init__",
        make_builtin_function("__init__", object_descr_init),
    );
    // PyPy: objectobject.py — default comparison/hash/repr for all objects
    dict_storage_store(
        ns,
        "__eq__",
        make_builtin_function_with_arity(
            "__eq__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                Ok(pyre_object::w_bool_from(std::ptr::eq(args[0], args[1])))
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__ne__",
        // `typeobject.py object_richcompare` — the default `__ne__`
        // negates the (virtually dispatched) `__eq__` result, so a
        // subclass that overrides only `__eq__` still gets a consistent
        // `!=`.  `__eq__` itself falls back to identity here.
        make_builtin_function_with_arity(
            "__ne__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                let eq = crate::baseobjspace::compare(
                    args[0],
                    args[1],
                    crate::baseobjspace::CompareOp::Eq,
                )?;
                // A `NotImplemented` from `__eq__` must pass through so the
                // caller can try the reflected comparison.
                if unsafe { pyre_object::is_not_implemented(eq) } {
                    return Ok(eq);
                }
                Ok(pyre_object::w_bool_from(!crate::baseobjspace::is_true(eq)?))
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__hash__",
        make_builtin_function_with_arity(
            "__hash__",
            |args| {
                Ok(pyre_object::w_int_new(if args.is_empty() {
                    0
                } else {
                    args[0] as i64
                }))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__repr__",
        // PyPy: objectobject.py descr___repr__ — base __repr__ for all objects
        make_builtin_function_with_arity(
            "__repr__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_str_new("<object>"));
                }
                let obj = args[0];
                unsafe {
                    if pyre_object::is_instance(obj) {
                        // `w_obj.getrepr(space, '%s object' % fulltypename)`.
                        let name = crate::baseobjspace::getfulltypename(obj);
                        return Ok(pyre_object::w_str_new(&format!(
                            "<{name} object at {obj:?}>"
                        )));
                    }
                }
                // For non-instances, delegate to display
                Ok(pyre_object::w_str_new(&format!("<object at {:?}>", obj)))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__str__",
        make_builtin_function_with_arity(
            "__str__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_str_new("<object>"));
                }
                // Delegate to __repr__ to avoid infinite recursion
                // PyPy: objectobject.py descr___str__ → space.repr(w_self)
                Ok(pyre_object::w_str_new(&unsafe { crate::py_repr(args[0])? }))
            },
            1,
        ),
    );
    // PyPy: objectobject.py descr___format__
    dict_storage_store(
        ns,
        "__format__",
        make_builtin_function_with_arity(
            "__format__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_str_new(""));
                }
                if args.len() > 1 {
                    // object.__format__(self, format_spec): the spec must be
                    // a `str` (a `bytes` spec is rejected like any other
                    // non-`str`); a non-empty one is unsupported, an empty
                    // one falls through to `str(self)`.
                    let spec =
                        crate::type_methods::read_format_spec(args[1], "__format__() argument")?;
                    if !spec.is_empty() {
                        return Err(crate::PyError::type_error(format!(
                            "unsupported format string passed to {}.__format__",
                            crate::type_methods::arg_type_name(args[0])
                        )));
                    }
                }
                Ok(pyre_object::w_str_new(&unsafe { crate::py_str(args[0])? }))
            },
            2,
        ),
    );
    // objectobject.py descr__reduce__ / descr__reduce_ex__ / descr__getstate__
    dict_storage_store(
        ns,
        "__reduce__",
        make_builtin_function_with_arity(
            "__reduce__",
            |args| crate::reduce_protocol::descr_reduce(args[0]),
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__reduce_ex__",
        make_builtin_function_with_arity(
            "__reduce_ex__",
            |args| {
                let proto = unsafe { pyre_object::w_int_get_value(args[1]) };
                crate::reduce_protocol::descr_reduce_ex(args[0], proto)
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__getstate__",
        make_builtin_function_with_arity(
            "__getstate__",
            |args| crate::reduce_protocol::object_getstate_default(args[0]),
            1,
        ),
    );
    // typeobject.py descr___init_subclass__ — the default accepts no
    // keywords; class-definition keywords reaching it via the builtin
    // kwargs ABI are an error, not silently dropped.
    dict_storage_store(
        ns,
        "__init_subclass__",
        make_builtin_function("__init_subclass__", |args| {
            let (_, kwargs) = crate::builtins::split_builtin_kwargs(args);
            if let Some(kw) = kwargs {
                let has_real_kw = unsafe {
                    pyre_object::w_dict_items(kw).into_iter().any(|(k, _)| {
                        pyre_object::is_str(k)
                            && pyre_object::w_str_get_wtf8(k).as_str() != Ok("__pyre_kw__")
                    })
                };
                if has_real_kw {
                    return Err(crate::PyError::type_error(
                        "__init_subclass__() takes no keyword arguments",
                    ));
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    dict_storage_store(
        ns,
        "__subclasshook__",
        make_builtin_function("__subclasshook__", |_| Ok(pyre_object::w_not_implemented())),
    );
    // PyPy: objectobject.py descr___setattr__
    // object.__setattr__(self, name, value) → setattr dispatch
    dict_storage_store(
        ns,
        "__setattr__",
        make_builtin_function_with_arity(
            "__setattr__",
            |args| {
                if args.len() < 3 {
                    return Err(crate::PyError::type_error(
                        "__setattr__ requires 3 arguments",
                    ));
                }
                if !unsafe { pyre_object::is_str(args[1]) } {
                    return Err(crate::PyError::type_error("attribute name must be string"));
                }
                // `object.__setattr__` is the terminal implementation
                // that writes directly to the instance dict, bypassing
                // any user __setattr__ override.
                let name = unsafe { pyre_object::w_str_get_wtf8(args[1]) };
                match name.as_str() {
                    Ok(s) => crate::baseobjspace::object_setattr(args[0], s, args[2]),
                    Err(_) => unsafe {
                        crate::baseobjspace::object_setattr_surrogate(
                            args[0], args[1], name, args[2],
                        )
                    },
                }
            },
            3,
        ),
    );
    // PyPy: objectobject.py descr___delattr__
    dict_storage_store(
        ns,
        "__delattr__",
        make_builtin_function_with_arity(
            "__delattr__",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "__delattr__ requires 2 arguments",
                    ));
                }
                if !unsafe { pyre_object::is_str(args[1]) } {
                    return Err(crate::PyError::type_error("attribute name must be string"));
                }
                let name = unsafe { pyre_object::w_str_get_wtf8(args[1]) };
                match name.as_str() {
                    Ok(s) => crate::baseobjspace::object_delattr(args[0], s),
                    Err(_) => unsafe {
                        crate::baseobjspace::object_delattr_surrogate(args[0], args[1], name)
                    },
                }
            },
            2,
        ),
    );
    // PyPy: objectobject.py descr___getattribute__
    dict_storage_store(
        ns,
        "__getattribute__",
        make_builtin_function_with_arity(
            "__getattribute__",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "__getattribute__ requires 2 arguments",
                    ));
                }
                if !unsafe { pyre_object::is_str(args[1]) } {
                    return Err(crate::PyError::type_error("attribute name must be string"));
                }
                let name = unsafe { pyre_object::w_str_get_wtf8(args[1]) };
                match name.as_str() {
                    Ok(s) => crate::baseobjspace::object_getattribute(args[0], s),
                    Err(_) => unsafe {
                        crate::baseobjspace::object_getattribute_surrogate(args[0], args[1], name)
                    },
                }
            },
            2,
        ),
    );
}

fn bytearray_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let value = bytearray_descr_new_impl(args)?;
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::bytearrayobject::BYTEARRAY_TYPE)? {
        let data = unsafe { pyre_object::bytesobject::bytes_like_data(value).to_vec() };
        let fresh = pyre_object::bytearrayobject::w_bytearray_from_bytes(&data);
        unsafe {
            (*fresh).w_class = sub;
        }
        return Ok(fresh);
    }
    Ok(value)
}

fn bytearray_descr_new_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // args[0] = cls (ignored — bytearray subclasses still allocate the
    // primitive layout). bytearrayobject.py descr_new accepts:
    //   bytearray()           → empty
    //   bytearray(int)        → zero-filled buffer of length n
    //   bytearray(bytes-like) → copy of the contents
    //   bytearray(str, encoding[, errors]) → encoded bytes (encoding ignored)
    // args[0] = cls. `bytearray(source=b'', encoding=None, errors=None)` —
    // every parameter is positional-or-keyword (bytearrayobject.py
    // descr_init shares bytesobject.newbytesdata_w); `encoding`/`errors`
    // are only valid with a str source.
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    // pos[0] is the class; `bytearray(source, encoding, errors)` accepts at
    // most three further positional arguments.
    if pos.len() > 4 {
        return Err(crate::PyError::type_error(&format!(
            "bytearray() takes at most 3 arguments ({} given)",
            pos.len() - 1
        )));
    }
    crate::builtins::kwarg_reject_unknown(kwargs, &["source", "encoding", "errors"], "bytearray")?;
    let source =
        crate::builtins::resolve_pos_or_kw(pos.get(1).copied(), kwargs, "source", "bytearray", 1)?;
    let w_encoding = crate::builtins::resolve_pos_or_kw(
        pos.get(2).copied(),
        kwargs,
        "encoding",
        "bytearray",
        2,
    )?;
    let w_errors =
        crate::builtins::resolve_pos_or_kw(pos.get(3).copied(), kwargs, "errors", "bytearray", 3)?;
    // `text_or_none` unwrap_spec treats an explicit `None` as absent.
    let w_encoding = w_encoding.filter(|&e| !unsafe { pyre_object::is_none(e) });
    let w_errors = w_errors.filter(|&e| !unsafe { pyre_object::is_none(e) });
    let Some(arg) = source else {
        if w_encoding.is_some() || w_errors.is_some() {
            return Err(crate::PyError::type_error(
                "encoding or errors without sequence argument",
            ));
        }
        return Ok(pyre_object::bytearrayobject::w_bytearray_new(0));
    };
    let has_codec = w_encoding.is_some() || w_errors.is_some();
    unsafe {
        // bytearrayobject.py:217 — str source shares bytesobject.newbytesdata_w
        if pyre_object::is_str(arg) {
            let encoding = match w_encoding {
                Some(e) if pyre_object::is_str(e) => pyre_object::w_str_get_value(e),
                _ => {
                    return Err(crate::PyError::type_error(
                        "string argument without an encoding",
                    ));
                }
            };
            let errors = match w_errors {
                Some(e) if pyre_object::is_str(e) => pyre_object::w_str_get_value(e),
                _ => "strict",
            };
            let encoded = crate::type_methods::encode_object(arg, encoding, errors)?;
            return Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(
                &encoded,
            ));
        }
        if has_codec {
            let which = if w_encoding.is_some() {
                "encoding"
            } else {
                "errors"
            };
            return Err(crate::PyError::type_error(format!(
                "{which} without string argument (got '{}' instead)",
                type_name_of(arg)
            )));
        }
        // newbytesdata_w_tail: `getindex_w(source, OverflowError)` — any object
        // exposing __index__ is a count of NUL bytes.  (bytearray does NOT
        // honour __bytes__, so there is no invoke_bytes_method here.)
        if pyre_object::pyobject::is_int_or_long(arg)
            || crate::baseobjspace::lookup(arg, "__index__").is_some()
        {
            let n = match crate::baseobjspace::int_w(crate::baseobjspace::space_index(arg)?) {
                Ok(n) => n,
                Err(e) if e.kind == crate::PyErrorKind::OverflowError => {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::OverflowError,
                        format!(
                            "cannot fit '{}' into an index-sized integer",
                            crate::baseobjspace::object_functionstr_type_name(arg)
                        ),
                    ));
                }
                Err(e) => return Err(e),
            };
            if n < 0 {
                return Err(crate::PyError::value_error("negative count"));
            }
            return Ok(pyre_object::bytearrayobject::w_bytearray_new(n as usize));
        }
        // `_convert_from_buffer_or_iterable`: any buffer exporter — bytes,
        // bytearray, `array.array`, memoryview — yields its raw buffer bytes
        // (`buffer_w(BUF_FULL_RO).as_str()`) before the iterable path; a
        // released memoryview raises first.
        if let Some(b) = crate::typedef::buffer_as_bytes_like(arg)? {
            return Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(
                pyre_object::bytesobject::bytes_like_data(b),
            ));
        }
    }
    // `_from_byte_sequence_loop`: stream the source through `byte_w` (honours
    // __index__, "byte must be in range(0, 256)"; a non-index element → "'X'
    // object cannot be interpreted as an integer").  A source with no __iter__
    // → "cannot convert 'X' object to bytearray"; an error raised by
    // __iter__/__next__ propagates unchanged.
    unsafe {
        let it = match crate::baseobjspace::iter(arg) {
            Ok(it) => it,
            Err(e) => {
                if crate::baseobjspace::lookup(arg, "__iter__").is_none() {
                    return Err(crate::PyError::type_error(format!(
                        "cannot convert '{}' object to bytearray",
                        crate::baseobjspace::object_functionstr_type_name(arg)
                    )));
                }
                return Err(e);
            }
        };
        let mut buf = Vec::new();
        loop {
            match crate::baseobjspace::next(it) {
                Ok(item) => buf.push(crate::baseobjspace::byte_w(item, "byte")?),
                Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
                Err(e) => return Err(e),
            }
        }
        Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(&buf))
    }
}

/// PyPy: bytesobject.py W_BytesObject.typedef
fn init_bytes_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(bytes_descr_new));
    dict_storage_store(
        ns,
        "__bytes__",
        make_builtin_function_with_arity("__bytes__", bytes_method_bytes, 1),
    );
    dict_storage_store(
        ns,
        "decode",
        make_builtin_function("decode", bytes_method_decode),
    );
    dict_storage_store(
        ns,
        "__repr__",
        make_builtin_function_with_arity("__repr__", bytes_method_repr, 1),
    );
    dict_storage_store(
        ns,
        "__str__",
        make_builtin_function_with_arity("__str__", bytes_method_repr, 1),
    );
    dict_storage_store(ns, "hex", make_builtin_function("hex", bytes_method_hex));
    dict_storage_store(ns, "find", make_builtin_function("find", bytes_method_find));
    dict_storage_store(
        ns,
        "rfind",
        make_builtin_function("rfind", bytes_method_rfind),
    );
    dict_storage_store(
        ns,
        "index",
        make_builtin_function("index", bytes_method_index),
    );
    dict_storage_store(
        ns,
        "rindex",
        make_builtin_function("rindex", bytes_method_rindex),
    );
    dict_storage_store(
        ns,
        "count",
        make_builtin_function("count", bytes_method_count),
    );
    dict_storage_store(
        ns,
        "startswith",
        make_builtin_function("startswith", bytes_method_startswith),
    );
    dict_storage_store(
        ns,
        "endswith",
        make_builtin_function("endswith", bytes_method_endswith),
    );
    dict_storage_store(
        ns,
        "upper",
        make_builtin_function("upper", bytes_method_upper),
    );
    dict_storage_store(
        ns,
        "lower",
        make_builtin_function("lower", bytes_method_lower),
    );
    dict_storage_store(
        ns,
        "strip",
        make_builtin_function("strip", bytes_method_strip),
    );
    dict_storage_store(
        ns,
        "lstrip",
        make_builtin_function("lstrip", bytes_method_lstrip),
    );
    dict_storage_store(
        ns,
        "rstrip",
        make_builtin_function("rstrip", bytes_method_rstrip),
    );
    dict_storage_store(
        ns,
        "replace",
        make_builtin_function("replace", bytes_method_replace),
    );
    dict_storage_store(
        ns,
        "split",
        make_builtin_function("split", bytes_method_split),
    );
    dict_storage_store(
        ns,
        "rsplit",
        make_builtin_function("rsplit", bytes_method_rsplit),
    );
    dict_storage_store(ns, "join", make_builtin_function("join", bytes_method_join));
    dict_storage_store(
        ns,
        "partition",
        make_builtin_function("partition", bytes_method_partition),
    );
    dict_storage_store(
        ns,
        "rpartition",
        make_builtin_function("rpartition", bytes_method_rpartition),
    );
    dict_storage_store(
        ns,
        "translate",
        make_builtin_function("translate", bytes_method_translate),
    );
    dict_storage_store(
        ns,
        "isdigit",
        make_builtin_function("isdigit", bytes_method_isdigit),
    );
    dict_storage_store(
        ns,
        "isalpha",
        make_builtin_function("isalpha", bytes_method_isalpha),
    );
    dict_storage_store(
        ns,
        "isalnum",
        make_builtin_function("isalnum", bytes_method_isalnum),
    );
    dict_storage_store(
        ns,
        "isspace",
        make_builtin_function("isspace", bytes_method_isspace),
    );
    dict_storage_store(
        ns,
        "isascii",
        make_builtin_function("isascii", bytes_method_isascii),
    );
    dict_storage_store(
        ns,
        "isupper",
        make_builtin_function("isupper", bytes_method_isupper),
    );
    dict_storage_store(
        ns,
        "islower",
        make_builtin_function("islower", bytes_method_islower),
    );
    dict_storage_store(
        ns,
        "istitle",
        make_builtin_function("istitle", bytes_method_istitle),
    );
    dict_storage_store(
        ns,
        "title",
        make_builtin_function("title", bytes_method_title),
    );
    dict_storage_store(
        ns,
        "capitalize",
        make_builtin_function("capitalize", bytes_method_capitalize),
    );
    dict_storage_store(
        ns,
        "swapcase",
        make_builtin_function("swapcase", bytes_method_swapcase),
    );
    dict_storage_store(
        ns,
        "removeprefix",
        make_builtin_function("removeprefix", bytes_method_removeprefix),
    );
    dict_storage_store(
        ns,
        "removesuffix",
        make_builtin_function("removesuffix", bytes_method_removesuffix),
    );
    dict_storage_store(
        ns,
        "ljust",
        make_builtin_function("ljust", bytes_method_ljust),
    );
    dict_storage_store(
        ns,
        "rjust",
        make_builtin_function("rjust", bytes_method_rjust),
    );
    dict_storage_store(
        ns,
        "center",
        make_builtin_function("center", bytes_method_center),
    );
    dict_storage_store(
        ns,
        "zfill",
        make_builtin_function("zfill", bytes_method_zfill),
    );
    dict_storage_store(
        ns,
        "splitlines",
        make_builtin_function("splitlines", bytes_method_splitlines),
    );
    dict_storage_store(
        ns,
        "expandtabs",
        make_builtin_function("expandtabs", bytes_method_expandtabs),
    );
    dict_storage_store(ns, "maketrans", make_maketrans_descr(bytes_maketrans));
    dict_storage_store(
        ns,
        "fromhex",
        pyre_object::function::w_classmethod_new(make_builtin_function("fromhex", bytes_fromhex)),
    );
    dict_storage_store(
        ns,
        "__add__",
        make_builtin_function_with_arity(
            "__add__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                unsafe { crate::objspace::descroperation::bytes_concat(args[0], args[1]) }
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__mul__",
        make_builtin_function_with_arity("__mul__", |args| bytes_descr_repeat(args), 2),
    );
    dict_storage_store(
        ns,
        "__rmul__",
        make_builtin_function_with_arity("__rmul__", |args| bytes_descr_repeat(args), 2),
    );
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function_with_arity(
            "__contains__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                Ok(pyre_object::w_bool_from(
                    crate::baseobjspace::contains_slot(args[0], args[1])?,
                ))
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                crate::baseobjspace::getitem_slot(args[0], args[1])
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function_with_arity(
            "__iter__",
            |args| {
                crate::type_methods::arity_slot(args, 0)?;
                crate::baseobjspace::iter(args[0])
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function_with_arity(
            "__len__",
            |args| {
                crate::type_methods::arity_slot(args, 0)?;
                crate::baseobjspace::len_slot(args[0])
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__mod__",
        make_builtin_function_with_arity(
            "__mod__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                unsafe { crate::objspace::std::formatting::bytes_format_percent(args[0], args[1]) }
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__rmod__",
        make_builtin_function_with_arity(
            "__rmod__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                if unsafe { pyre_object::bytesobject::is_bytes(args[1]) } {
                    unsafe {
                        crate::objspace::std::formatting::bytes_format_percent(args[1], args[0])
                    }
                } else {
                    Ok(pyre_object::w_not_implemented())
                }
            },
            2,
        ),
    );
    for (name, func) in [
        ("__eq__", bytes_dunder_eq as DunderFn),
        ("__ne__", bytes_dunder_ne),
        ("__lt__", bytes_dunder_lt),
        ("__le__", bytes_dunder_le),
        ("__gt__", bytes_dunder_gt),
        ("__ge__", bytes_dunder_ge),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
    // bytesobject.py descr_getnewargs — a fresh plain bytes from the value,
    // so a bytes subclass reduces to bytes.
    dict_storage_store(
        ns,
        "__getnewargs__",
        make_builtin_function_with_arity(
            "__getnewargs__",
            |args| {
                let data = unsafe { pyre_object::w_bytes_data(args[0]) };
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_bytes_from_bytes(data),
                ]))
            },
            1,
        ),
    );
    // bytes methods are mostly shared with bytearray — add as needed.
}

fn bytes_descr_repeat(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_slot(args, 1)?;
    let Some(count) = list_repeat_index(args[1])? else {
        return Ok(pyre_object::w_not_implemented());
    };
    unsafe { crate::objspace::descroperation::bytes_repeat(args[0], count) }
}

/// `stringmethods.py:_op_val(space, w_sub, allow_char=True)` — the
/// `sub` argument of a bytes search/count method is either a bytes-like
/// object or a single integer in `range(0, 256)` standing for one byte.
fn bytes_sub_arg(w_sub: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    unsafe {
        if let Some(src) = buffer_as_bytes_like(w_sub)? {
            Ok(pyre_object::bytesobject::bytes_like_data(src).to_vec())
        } else if pyre_object::is_int(w_sub) {
            let v = pyre_object::w_int_get_value(w_sub);
            if !(0..=255).contains(&v) {
                return Err(crate::PyError::value_error("byte must be in range(0, 256)"));
            }
            Ok(vec![v as u8])
        } else {
            Err(crate::PyError::type_error(format!(
                "argument should be integer or bytes-like object, not '{}'",
                type_name_of(w_sub)
            )))
        }
    }
}

/// `stringmethods.py:_convert_idx_params` — resolve the optional `start`
/// / `end` search args (PyPy slice semantics) into a byte-offset window
/// `[start, end)` into a bytes-like of length `len`.  Returns `None`
/// when the window is empty because `start` is past the end or past
/// `end` (the search-miss case shared by find / index / count).
fn bytes_idx_window(
    len: usize,
    args: &[PyObjectRef],
) -> Result<Option<(usize, usize)>, crate::PyError> {
    let len_i = len as i64;
    let w_start = if args.len() >= 3 {
        args[2]
    } else {
        pyre_object::w_none()
    };
    let w_end = if args.len() >= 4 {
        args[3]
    } else {
        pyre_object::w_none()
    };
    let (start, end) = crate::sliceobject::unwrap_start_stop(len_i, w_start, w_end)?;
    if start > len_i {
        return Ok(None);
    }
    let end = end.min(len_i);
    if start > end {
        return Ok(None);
    }
    Ok(Some((start as usize, end as usize)))
}

/// First index of `needle` within `hay`; empty needle matches at 0.
fn bytes_find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// Last index of `needle` within `hay`; empty needle matches at `len`.
fn bytes_rfind_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(hay.len());
    }
    if needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len())
        .rev()
        .find(|&i| &hay[i..i + needle.len()] == needle)
}

/// Non-overlapping occurrence count; empty needle yields `len + 1`.
fn bytes_count_subslices(hay: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() {
        return hay.len() + 1;
    }
    let mut count = 0;
    let mut i = 0;
    while i + needle.len() <= hay.len() {
        if &hay[i..i + needle.len()] == needle {
            count += 1;
            i += needle.len();
        } else {
            i += 1;
        }
    }
    count
}

/// `stringmethods.py:descr_find` / `descr_rfind` — search a bytes-like
/// over the codepoint-irrelevant byte window selected by start / end.
fn bytes_search(args: &[PyObjectRef], forward: bool) -> Result<i64, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let sub = bytes_sub_arg(args[1])?;
    let Some((start, end)) = bytes_idx_window(data.len(), args)? else {
        return Ok(-1);
    };
    let window = &data[start..end];
    let pos = if forward {
        bytes_find_subslice(window, &sub)
    } else {
        bytes_rfind_subslice(window, &sub)
    };
    Ok(pos.map(|p| (start + p) as i64).unwrap_or(-1))
}

fn bytes_method_find(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "find", 1)?;
    Ok(pyre_object::w_int_new(bytes_search(args, true)?))
}

fn bytes_method_rfind(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "rfind", 1)?;
    Ok(pyre_object::w_int_new(bytes_search(args, false)?))
}

fn bytes_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "index", 1)?;
    let res = bytes_search(args, true)?;
    if res < 0 {
        return Err(crate::PyError::value_error("subsection not found"));
    }
    Ok(pyre_object::w_int_new(res))
}

fn bytes_method_rindex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "rindex", 1)?;
    let res = bytes_search(args, false)?;
    if res < 0 {
        return Err(crate::PyError::value_error("subsection not found"));
    }
    Ok(pyre_object::w_int_new(res))
}

fn bytes_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "count", 1)?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let sub = bytes_sub_arg(args[1])?;
    let Some((start, end)) = bytes_idx_window(data.len(), args)? else {
        return Ok(pyre_object::w_int_new(0));
    };
    Ok(pyre_object::w_int_new(
        bytes_count_subslices(&data[start..end], &sub) as i64,
    ))
}

/// `stringmethods.py:descr_startswith` / `descr_endswith` — test the
/// byte window `[start, end)` against a single bytes-like prefix or a
/// tuple of bytes-like prefixes.  `forward` selects starts/ends.
fn bytes_prefix_match(
    args: &[PyObjectRef],
    method: &str,
    forward: bool,
) -> Result<bool, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    // `start > len(value)` collapses the window to None → no match.
    let Some((start, end)) = bytes_idx_window(data.len(), args)? else {
        return Ok(false);
    };
    let window = &data[start..end];
    let test = |p: &[u8]| {
        if forward {
            window.starts_with(p)
        } else {
            window.ends_with(p)
        }
    };
    let needle = args[1];
    unsafe {
        if let Some(src) = buffer_as_bytes_like(needle)? {
            return Ok(test(pyre_object::bytesobject::bytes_like_data(src)));
        }
        if pyre_object::is_tuple(needle) {
            let n = pyre_object::w_tuple_len(needle) as i64;
            for i in 0..n {
                let item = pyre_object::w_tuple_getitem(needle, i).expect("index is in range");
                let Some(src) = buffer_as_bytes_like(item)? else {
                    return Err(crate::PyError::type_error(format!(
                        "a bytes-like object is required, not '{}'",
                        type_name_of(item)
                    )));
                };
                if test(pyre_object::bytesobject::bytes_like_data(src)) {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
        Err(crate::PyError::type_error(format!(
            "{method} first arg must be bytes or a tuple of bytes, not {}",
            type_name_of(needle)
        )))
    }
}

fn bytes_method_startswith(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "startswith", 1)?;
    Ok(pyre_object::w_bool_from(bytes_prefix_match(
        args,
        "startswith",
        true,
    )?))
}

fn bytes_method_endswith(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "endswith", 1)?;
    Ok(pyre_object::w_bool_from(bytes_prefix_match(
        args, "endswith", false,
    )?))
}

/// `bytesobject.py:390 descr_upper` — ASCII-only case mapping (bytes
/// outside `a`-`z` / `A`-`Z`, including non-ASCII, are unchanged).
/// `stringmethods.py:_new` — the StringMethods mixin builds its result
/// with `self._new(...)`, which each subclass overrides to produce its
/// own kind.  So a transform on a `bytearray` receiver yields a
/// `bytearray`, while the same transform on `bytes` yields `bytes`.
fn new_bytes_like(recv: PyObjectRef, data: &[u8]) -> PyObjectRef {
    if unsafe { pyre_object::bytearrayobject::is_bytearray(recv) } {
        pyre_object::bytearrayobject::w_bytearray_from_bytes(data)
    } else {
        pyre_object::bytesobject::w_bytes_from_bytes(data)
    }
}

/// Empty result matching the receiver's kind (see [`new_bytes_like`]).
fn empty_bytes_like(recv: PyObjectRef) -> PyObjectRef {
    new_bytes_like(recv, b"")
}

fn bytes_method_upper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "upper")?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let out: Vec<u8> = data.iter().map(|b| b.to_ascii_uppercase()).collect();
    Ok(new_bytes_like(args[0], &out))
}

/// `bytesobject.py:247 descr_lower` — ASCII-only case mapping.
fn bytes_method_lower(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "lower")?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let out: Vec<u8> = data.iter().map(|b| b.to_ascii_lowercase()).collect();
    Ok(new_bytes_like(args[0], &out))
}

/// `stringmethods.py:_strip` / `_strip_none` — trim bytes from the
/// ends.  With no / `None` `chars` arg the default ASCII-whitespace set
/// is stripped (` \t\n\r\x0b\x0c`); with a bytes-like arg any byte in
/// that set is trimmed.  `left` / `right` select the sides.
fn bytes_strip(
    args: &[PyObjectRef],
    left: bool,
    right: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let chars: Option<Vec<u8>> = match args.get(1) {
        Some(&a) if !a.is_null() && unsafe { !pyre_object::is_none(a) } => {
            if let Some(src) = buffer_as_bytes_like(a)? {
                Some(unsafe { pyre_object::bytesobject::bytes_like_data(src) }.to_vec())
            } else {
                return Err(crate::PyError::type_error(format!(
                    "a bytes-like object is required, not '{}'",
                    type_name_of(a)
                )));
            }
        }
        _ => None,
    };
    let in_set = |b: u8| match &chars {
        Some(set) => set.contains(&b),
        None => matches!(b, 0x09 | 0x0a | 0x0b | 0x0c | 0x0d | 0x20),
    };
    let mut lo = 0;
    let mut hi = data.len();
    if left {
        while lo < hi && in_set(data[lo]) {
            lo += 1;
        }
    }
    if right {
        while hi > lo && in_set(data[hi - 1]) {
            hi -= 1;
        }
    }
    Ok(new_bytes_like(args[0], &data[lo..hi]))
}

fn bytes_method_strip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "strip")?;
    bytes_strip(args, true, true)
}

fn bytes_method_lstrip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "lstrip")?;
    bytes_strip(args, true, false)
}

fn bytes_method_rstrip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "rstrip")?;
    bytes_strip(args, false, true)
}

/// Resolve a buffer-providing object to a bytes-like object whose bytes
/// `bytes_like_data` can read: bytes / bytearray resolve to themselves, and a
/// `memoryview` materialises its live view (honouring stride) into a fresh
/// `bytes`.  `Ok(None)` for anything else; a released memoryview is rejected
/// with `ValueError` (`space.buffer_w` calls `_check_released` first).  Lets
/// bytes / bytearray methods accept any buffer argument the way
/// `space.buffer_w(w_obj, space.BUF_SIMPLE)` does upstream, without treating a
/// memoryview as bytes-like elsewhere.
pub(crate) fn buffer_as_bytes_like(
    obj: PyObjectRef,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    if let Some(target) = crate::module::__pypy__::interp_buffer::forwarded_exporter(obj) {
        return buffer_as_bytes_like(target?);
    }
    if unsafe { pyre_object::interp_array::is_array(obj) } {
        return Ok(Some(pyre_object::bytesobject::w_bytes_from_bytes(unsafe {
            pyre_object::interp_array::w_array_bytes(obj)
        })));
    }
    if unsafe { pyre_object::bytesobject::is_bytes_like(obj) } {
        return Ok(Some(obj));
    }
    if unsafe { pyre_object::memoryview::is_w_memoryview(obj) } {
        unsafe { crate::builtins::memoryview_check_released(obj) }?;
        let data = unsafe { crate::builtins::memoryview_gather_bytes(obj) };
        return Ok(Some(pyre_object::bytesobject::w_bytes_from_bytes(&data)));
    }
    Ok(None)
}

/// Require `obj` to be a bytes-like object, returning its bytes; raises
/// the CPython `a bytes-like object is required, not '<type>'` TypeError
/// otherwise.  A memoryview is accepted through its backing buffer.
fn require_bytes_like(obj: PyObjectRef) -> Result<&'static [u8], crate::PyError> {
    match buffer_as_bytes_like(obj)? {
        Some(src) => Ok(unsafe { pyre_object::bytesobject::bytes_like_data(src) }),
        None => Err(crate::PyError::type_error(format!(
            "a bytes-like object is required, not '{}'",
            type_name_of(obj)
        ))),
    }
}

/// The Python-visible class name of `obj` (its `w_class`/type name), used in
/// bytes-method TypeErrors.  More accurate than the raw `ob_type` name for
/// instance-layout objects (e.g. a memoryview reports `memoryview`).
fn type_name_of(obj: PyObjectRef) -> String {
    match r#type(obj) {
        Some(tp) => unsafe { pyre_object::w_type_get_name(tp) }.to_string(),
        None => unsafe { (*(*obj).ob_type).name.to_string() },
    }
}

/// Non-overlapping left-to-right byte replacement, capped at `limit`.
/// An empty `old` inserts `new` before every byte and at the end, per
/// CPython `bytes.replace(b"", ...)`.
fn replace_bytes(data: &[u8], old: &[u8], new: &[u8], limit: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut count = 0;
    if old.is_empty() {
        for &b in data {
            if count < limit {
                out.extend_from_slice(new);
                count += 1;
            }
            out.push(b);
        }
        if count < limit {
            out.extend_from_slice(new);
        }
        return out;
    }
    let mut i = 0;
    while i < data.len() {
        if count < limit && data[i..].starts_with(old) {
            out.extend_from_slice(new);
            i += old.len();
            count += 1;
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

const BYTES_WHITESPACE: [u8; 6] = [0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20];

fn split_bytes_sep(data: &[u8], sep: &[u8], maxsplit: i64) -> Vec<Vec<u8>> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut count = 0i64;
    let mut i = 0;
    while i + sep.len() <= data.len() {
        if (maxsplit < 0 || count < maxsplit) && &data[i..i + sep.len()] == sep {
            parts.push(data[start..i].to_vec());
            i += sep.len();
            start = i;
            count += 1;
        } else {
            i += 1;
        }
    }
    parts.push(data[start..].to_vec());
    parts
}

fn rsplit_bytes_sep(data: &[u8], sep: &[u8], maxsplit: i64) -> Vec<Vec<u8>> {
    let mut parts = Vec::new();
    let mut end = data.len();
    let mut count = 0i64;
    let mut i = data.len();
    while i >= sep.len() {
        if (maxsplit < 0 || count < maxsplit) && &data[i - sep.len()..i] == sep {
            parts.push(data[i..end].to_vec());
            end = i - sep.len();
            i = end;
            count += 1;
        } else {
            i -= 1;
        }
    }
    parts.push(data[..end].to_vec());
    parts.reverse();
    parts
}

fn split_bytes_ws(data: &[u8], maxsplit: i64) -> Vec<Vec<u8>> {
    let is_ws = |b: u8| BYTES_WHITESPACE.contains(&b);
    let mut parts: Vec<Vec<u8>> = Vec::new();
    let n = data.len();
    let mut i = 0;
    loop {
        while i < n && is_ws(data[i]) {
            i += 1;
        }
        if i >= n {
            break;
        }
        if maxsplit >= 0 && parts.len() as i64 >= maxsplit {
            let mut end = n;
            while end > i && is_ws(data[end - 1]) {
                end -= 1;
            }
            parts.push(data[i..end].to_vec());
            break;
        }
        let start = i;
        while i < n && !is_ws(data[i]) {
            i += 1;
        }
        parts.push(data[start..i].to_vec());
    }
    parts
}

fn rsplit_bytes_ws(data: &[u8], maxsplit: i64) -> Vec<Vec<u8>> {
    let is_ws = |b: u8| BYTES_WHITESPACE.contains(&b);
    let mut parts: Vec<Vec<u8>> = Vec::new();
    let mut i = data.len();
    loop {
        while i > 0 && is_ws(data[i - 1]) {
            i -= 1;
        }
        if i == 0 {
            break;
        }
        if maxsplit >= 0 && parts.len() as i64 >= maxsplit {
            let mut start = 0;
            while start < i && is_ws(data[start]) {
                start += 1;
            }
            parts.push(data[start..i].to_vec());
            break;
        }
        let end = i;
        while i > 0 && !is_ws(data[i - 1]) {
            i -= 1;
        }
        parts.push(data[i..end].to_vec());
    }
    parts.reverse();
    parts
}

/// `stringmethods.py:descr_split` / `descr_rsplit` — split a bytes-like
/// on a bytes-like separator (empty separator → ValueError) or, when
/// `sep` is absent / `None`, on runs of ASCII whitespace with empty
/// fields dropped.  `maxsplit < 0` means unlimited.  `forward` selects
/// split vs rsplit.
fn bytes_split(args: &[PyObjectRef], forward: bool) -> Result<PyObjectRef, crate::PyError> {
    // `sep` and `maxsplit` are both positional-or-keyword; `maxsplit`
    // routes through `__index__` (`space_index_w`), so a non-integer
    // (including `None`) raises rather than silently defaulting.
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let fn_name = if forward { "split" } else { "rsplit" };
    crate::builtins::kwarg_reject_unknown(kwargs, &["sep", "maxsplit"], fn_name)?;
    crate::builtins::kwarg_reject_duplicate(kwargs, fn_name, "sep", pos.get(1).is_some())?;
    crate::builtins::kwarg_reject_duplicate(kwargs, fn_name, "maxsplit", pos.get(2).is_some())?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(pos[0]) };
    let maxsplit = match pos
        .get(2)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "maxsplit"))
    {
        Some(m) if !m.is_null() => crate::builtins::space_index_w(m)?,
        _ => -1,
    };
    let sep_arg = pos
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "sep"));
    let sep: Option<Vec<u8>> = match sep_arg {
        Some(o) if !o.is_null() && unsafe { !pyre_object::is_none(o) } => {
            if let Some(src) = buffer_as_bytes_like(o)? {
                Some(unsafe { pyre_object::bytesobject::bytes_like_data(src) }.to_vec())
            } else {
                return Err(crate::PyError::type_error(format!(
                    "a bytes-like object is required, not '{}'",
                    type_name_of(o)
                )));
            }
        }
        _ => None,
    };
    let parts = match sep {
        Some(s) => {
            if s.is_empty() {
                return Err(crate::PyError::value_error("empty separator"));
            }
            if forward {
                split_bytes_sep(data, &s, maxsplit)
            } else {
                rsplit_bytes_sep(data, &s, maxsplit)
            }
        }
        None => {
            if forward {
                split_bytes_ws(data, maxsplit)
            } else {
                rsplit_bytes_ws(data, maxsplit)
            }
        }
    };
    let items: Vec<PyObjectRef> = parts.iter().map(|p| new_bytes_like(pos[0], p)).collect();
    Ok(pyre_object::w_list_new(items))
}

fn bytes_method_split(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "split")?;
    bytes_split(args, true)
}

fn bytes_method_rsplit(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "rsplit")?;
    bytes_split(args, false)
}

/// `stringmethods.py:descr_replace` — replace occurrences of `old` with
/// `new` (both bytes-like); optional `count` caps the replacements (a
/// negative or absent count means "no limit").
fn bytes_method_replace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `replace` is positional-only; any keyword argument is rejected.
    // `count` routes through `__index__` (`space_index_w`), so a
    // non-integer raises rather than silently defaulting to "no limit".
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if kwargs.is_some() {
        return Err(crate::PyError::type_error(format!(
            "{}.replace() takes no keyword arguments",
            unsafe { (*(*pos[0]).ob_type).name }
        )));
    }
    assert!(pos.len() >= 3, "replace() takes at least 2 arguments");
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(pos[0]) };
    let old = require_bytes_like(pos[1])?;
    let new = require_bytes_like(pos[2])?;
    let limit = match pos.get(3) {
        Some(&w_count) if !w_count.is_null() => {
            let c = crate::builtins::space_index_w(w_count)?;
            if c < 0 { usize::MAX } else { c as usize }
        }
        _ => usize::MAX,
    };
    Ok(new_bytes_like(
        pos[0],
        &replace_bytes(data, old, new, limit),
    ))
}

/// `stringmethods.py:descr_join` — concatenate the bytes-like elements
/// of an iterable, inserting the receiver between them.  A non-bytes
/// element raises the CPython `sequence item N: expected a bytes-like
/// object, <T> found` TypeError.
fn bytes_method_join(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "join() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let sep = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let iterable = args[1];
    let items: Vec<PyObjectRef> = unsafe {
        if pyre_object::is_list(iterable) {
            let n = pyre_object::w_list_len(iterable);
            (0..n)
                .filter_map(|i| pyre_object::w_list_getitem(iterable, i as i64))
                .collect()
        } else if pyre_object::is_tuple(iterable) {
            let n = pyre_object::w_tuple_len(iterable);
            (0..n)
                .filter_map(|i| pyre_object::w_tuple_getitem(iterable, i as i64))
                .collect()
        } else {
            crate::builtins::collect_iterable(iterable)?
        }
    };
    let mut out: Vec<u8> = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        if i > 0 {
            out.extend_from_slice(sep);
        }
        let Some(src) = buffer_as_bytes_like(item)? else {
            return Err(crate::PyError::type_error(format!(
                "sequence item {i}: expected a bytes-like object, {} found",
                type_name_of(item)
            )));
        };
        out.extend_from_slice(unsafe { pyre_object::bytesobject::bytes_like_data(src) });
    }
    Ok(new_bytes_like(args[0], &out))
}

/// `stringmethods.py:descr_partition` / `descr_rpartition` — split once
/// at the first / last occurrence of a non-empty bytes-like separator,
/// returning a 3-tuple `(head, sep, tail)`.  Empty separator raises
/// ValueError; when not found the whole value lands in the first
/// (partition) or last (rpartition) slot with empty siblings.
fn bytes_partition(args: &[PyObjectRef], forward: bool) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let sep = require_bytes_like(args[1])?;
    if sep.is_empty() {
        return Err(crate::PyError::value_error("empty separator"));
    }
    let found = if forward {
        bytes_find_subslice(data, sep)
    } else {
        bytes_rfind_subslice(data, sep)
    };
    match found {
        Some(i) => {
            // `stringlib_partition` hands back the separator argument object
            // itself for a bytes receiver (so a memoryview separator survives
            // as the middle element); `bytearray_partition` builds a fresh
            // bytearray slice for all three parts.
            let middle = if unsafe { pyre_object::bytesobject::is_bytes(args[0]) } {
                args[1]
            } else {
                new_bytes_like(args[0], sep)
            };
            Ok(pyre_object::w_tuple_new(vec![
                new_bytes_like(args[0], &data[..i]),
                middle,
                new_bytes_like(args[0], &data[i + sep.len()..]),
            ]))
        }
        None => {
            // A bytearray receiver must not alias into the result tuple
            // (mutating it would mutate the tuple); hand back a fresh copy.
            let whole = new_bytes_like(args[0], data);
            let empty = || empty_bytes_like(args[0]);
            if forward {
                Ok(pyre_object::w_tuple_new(vec![whole, empty(), empty()]))
            } else {
                Ok(pyre_object::w_tuple_new(vec![empty(), empty(), whole]))
            }
        }
    }
}

fn bytes_method_partition(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "partition() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    bytes_partition(args, true)
}

fn bytes_method_rpartition(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "rpartition() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    bytes_partition(args, false)
}

/// Non-empty and every byte satisfies `pred` — the shape shared by
/// `bytes.isdigit` / `isalpha` / `isalnum` / `isspace`.
fn bytes_all_nonempty(data: &[u8], pred: impl Fn(u8) -> bool) -> bool {
    !data.is_empty() && data.iter().all(|&b| pred(b))
}

fn bytes_method_isdigit(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    Ok(pyre_object::w_bool_from(bytes_all_nonempty(data, |b| {
        b.is_ascii_digit()
    })))
}

fn bytes_method_isalpha(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    Ok(pyre_object::w_bool_from(bytes_all_nonempty(data, |b| {
        b.is_ascii_alphabetic()
    })))
}

fn bytes_method_isalnum(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    Ok(pyre_object::w_bool_from(bytes_all_nonempty(data, |b| {
        b.is_ascii_alphanumeric()
    })))
}

fn bytes_method_isspace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    Ok(pyre_object::w_bool_from(bytes_all_nonempty(data, |b| {
        BYTES_WHITESPACE.contains(&b)
    })))
}

/// `bytes.isascii` / `bytearray.isascii` — every byte is <= 0x7F.
/// An empty buffer is ASCII (`descr_isascii` returns True on no bytes).
fn bytes_method_isascii(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    Ok(pyre_object::w_bool_from(data.is_ascii()))
}

/// `bytes.isupper` — at least one cased byte and no lowercase byte.
fn bytes_method_isupper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let mut cased = false;
    for &b in data {
        if b.is_ascii_lowercase() {
            return Ok(pyre_object::w_bool_from(false));
        }
        if b.is_ascii_uppercase() {
            cased = true;
        }
    }
    Ok(pyre_object::w_bool_from(cased))
}

/// `bytes.islower` — at least one cased byte and no uppercase byte.
fn bytes_method_islower(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let mut cased = false;
    for &b in data {
        if b.is_ascii_uppercase() {
            return Ok(pyre_object::w_bool_from(false));
        }
        if b.is_ascii_lowercase() {
            cased = true;
        }
    }
    Ok(pyre_object::w_bool_from(cased))
}

/// `bytes.istitle` — titlecased: every run of cased bytes starts with an
/// uppercase byte followed by lowercase, with at least one cased byte.
fn bytes_method_istitle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let mut cased = false;
    let mut prev_cased = false;
    for &b in data {
        if b.is_ascii_uppercase() {
            if prev_cased {
                return Ok(pyre_object::w_bool_from(false));
            }
            prev_cased = true;
            cased = true;
        } else if b.is_ascii_lowercase() {
            if !prev_cased {
                return Ok(pyre_object::w_bool_from(false));
            }
            prev_cased = true;
            cased = true;
        } else {
            prev_cased = false;
        }
    }
    Ok(pyre_object::w_bool_from(cased))
}

/// `stringmethods.py` justification fill char — defaults to space; a
/// non-length-1 bytes-like raises `<method>() argument 2 must be a
/// single character`.
fn bytes_fill_char(args: &[PyObjectRef], idx: usize, method: &str) -> Result<u8, crate::PyError> {
    match args.get(idx) {
        Some(&f) if !f.is_null() && unsafe { !pyre_object::is_none(f) } => {
            let d = require_bytes_like(f)?;
            if d.len() != 1 {
                return Err(crate::PyError::type_error(format!(
                    "{method}() argument 2 must be a single character"
                )));
            }
            Ok(d[0])
        }
        _ => Ok(b' '),
    }
}

/// `stringmethods.py:descr_ljust` — left-justify within `width`.
fn bytes_method_ljust(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "ljust", 1)?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let width = crate::builtins::space_index_w(args[1])?;
    let fill = bytes_fill_char(args, 2, "ljust")?;
    let len = data.len() as i64;
    if width <= len {
        return Ok(new_bytes_like(args[0], data));
    }
    let mut out = data.to_vec();
    out.resize(width as usize, fill);
    Ok(new_bytes_like(args[0], &out))
}

/// `stringmethods.py:descr_rjust` — right-justify within `width`.
fn bytes_method_rjust(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "rjust", 1)?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let width = crate::builtins::space_index_w(args[1])?;
    let fill = bytes_fill_char(args, 2, "rjust")?;
    let len = data.len() as i64;
    if width <= len {
        return Ok(new_bytes_like(args[0], data));
    }
    let mut out = vec![fill; (width - len) as usize];
    out.extend_from_slice(data);
    Ok(new_bytes_like(args[0], &out))
}

/// `stringmethods.py:descr_center` — center within `width`; the extra
/// fill byte (for odd padding) follows PyPy's `d//2 + (d & width & 1)`
/// left-offset.
fn bytes_method_center(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "center", 1)?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let width = crate::builtins::space_index_w(args[1])?;
    let fill = bytes_fill_char(args, 2, "center")?;
    let len = data.len() as i64;
    if width <= len {
        return Ok(new_bytes_like(args[0], data));
    }
    let d = width - len;
    let offset = (d / 2 + (d & width & 1)) as usize;
    let mut out = vec![fill; offset];
    out.extend_from_slice(data);
    out.resize(width as usize, fill);
    Ok(new_bytes_like(args[0], &out))
}

/// `bytesobject.py:descr_zfill` — left-pad with `b'0'` to `width`,
/// keeping a leading `+`/`-` sign ahead of the zeros.
fn bytes_method_zfill(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_exact(args, "bytes.zfill", 1)?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let width = crate::builtins::space_index_w(args[1])?;
    let len = data.len() as i64;
    if width <= len {
        return Ok(new_bytes_like(args[0], data));
    }
    let pad = (width - len) as usize;
    let mut out = Vec::with_capacity(width as usize);
    let rest = match data.split_first() {
        Some((&first, tail)) if first == b'+' || first == b'-' => {
            out.push(first);
            tail
        }
        _ => data,
    };
    out.resize(out.len() + pad, b'0');
    out.extend_from_slice(rest);
    Ok(new_bytes_like(args[0], &out))
}

/// `bytes.title` — ASCII titlecase: the first alphabetic byte of each
/// run is uppercased, the rest lowercased; non-alphabetic bytes reset
/// the run.
fn bytes_method_title(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let mut prev_cased = false;
    let out: Vec<u8> = data
        .iter()
        .map(|&b| {
            if b.is_ascii_alphabetic() {
                let mapped = if prev_cased {
                    b.to_ascii_lowercase()
                } else {
                    b.to_ascii_uppercase()
                };
                prev_cased = true;
                mapped
            } else {
                prev_cased = false;
                b
            }
        })
        .collect();
    Ok(new_bytes_like(args[0], &out))
}

/// `bytes.capitalize` — ASCII: first byte uppercased, the rest
/// lowercased.
fn bytes_method_capitalize(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let out: Vec<u8> = data
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            if i == 0 {
                b.to_ascii_uppercase()
            } else {
                b.to_ascii_lowercase()
            }
        })
        .collect();
    Ok(new_bytes_like(args[0], &out))
}

/// `bytes.swapcase` — ASCII: swap the case of each cased byte.
fn bytes_method_swapcase(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let out: Vec<u8> = data
        .iter()
        .map(|&b| {
            if b.is_ascii_uppercase() {
                b.to_ascii_lowercase()
            } else if b.is_ascii_lowercase() {
                b.to_ascii_uppercase()
            } else {
                b
            }
        })
        .collect();
    Ok(new_bytes_like(args[0], &out))
}

/// `bytes.removeprefix` — drop a leading bytes-like prefix if present.
fn bytes_method_removeprefix(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, _) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "{}.removeprefix() takes exactly one argument ({} given)",
            unsafe { (*(*pos[0]).ob_type).name },
            pos.len().saturating_sub(1)
        )));
    }
    let args = pos;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let prefix = require_bytes_like(args[1])?;
    let out = if data.starts_with(prefix) {
        &data[prefix.len()..]
    } else {
        data
    };
    Ok(new_bytes_like(args[0], out))
}

/// `bytes.removesuffix` — drop a trailing bytes-like suffix if present.
fn bytes_method_removesuffix(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, _) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "{}.removesuffix() takes exactly one argument ({} given)",
            unsafe { (*(*pos[0]).ob_type).name },
            pos.len().saturating_sub(1)
        )));
    }
    let args = pos;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let suffix = require_bytes_like(args[1])?;
    let out = if !suffix.is_empty() && data.ends_with(suffix) {
        &data[..data.len() - suffix.len()]
    } else {
        data
    };
    Ok(new_bytes_like(args[0], out))
}

/// `bytesobject.py:descr_translate` — map each byte through a 256-entry
/// `table` (or `None` for identity) after dropping any byte present in
/// the optional `delete` set.  `delete` may be positional or the
/// `delete=` keyword.
fn bytes_method_translate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least_positional(args, "translate", 1)?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    let Some(&table_obj) = positional.first() else {
        return Err(crate::PyError::type_error(
            "translate() takes at least 1 positional argument (0 given)",
        ));
    };
    let table: Option<&[u8]> = unsafe {
        if pyre_object::is_none(table_obj) {
            None
        } else if let Some(src) = buffer_as_bytes_like(table_obj)? {
            let t = pyre_object::bytesobject::bytes_like_data(src);
            if t.len() != 256 {
                return Err(crate::PyError::value_error(
                    "translation table must be 256 characters long",
                ));
            }
            Some(t)
        } else {
            return Err(crate::PyError::type_error(format!(
                "a bytes-like object is required, not '{}'",
                type_name_of(table_obj)
            )));
        }
    };
    let delete_obj = positional
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "delete"));
    let mut deleted = [false; 256];
    if let Some(d) = delete_obj {
        if !d.is_null() && unsafe { !pyre_object::is_none(d) } {
            if let Some(src) = buffer_as_bytes_like(d)? {
                for &b in unsafe { pyre_object::bytesobject::bytes_like_data(src) } {
                    deleted[b as usize] = true;
                }
            } else {
                return Err(crate::PyError::type_error(format!(
                    "a bytes-like object is required, not '{}'",
                    type_name_of(d)
                )));
            }
        }
    }
    let mut out = Vec::with_capacity(data.len());
    for &b in data {
        if deleted[b as usize] {
            continue;
        }
        out.push(match table {
            Some(t) => t[b as usize],
            None => b,
        });
    }
    Ok(new_bytes_like(args[0], &out))
}

/// `stringmethods.py:descr_splitlines` — split on `\n`, `\r`, and
/// `\r\n` line boundaries (the byte set; the extended Unicode line
/// terminators are str-only).  `keepends=True` retains the terminator
/// on each emitted line, and a trailing terminator does not produce an
/// extra empty entry.
fn bytes_method_splitlines(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "splitlines")?;
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["keepends"], "splitlines")?;
    crate::builtins::kwarg_reject_duplicate(
        kwargs,
        "splitlines",
        "keepends",
        pos.get(1).is_some(),
    )?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(pos[0]) };
    // keepends is positional-or-keyword.
    let keepends = crate::builtins::kwarg_get(kwargs, "keepends")
        .or_else(|| pos.get(1).copied())
        .map(crate::baseobjspace::is_true)
        .transpose()?
        .unwrap_or(false);
    let args = pos;
    let mut parts: Vec<PyObjectRef> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < data.len() {
        if data[i] == b'\n' || data[i] == b'\r' {
            let mut term_end = i + 1;
            if data[i] == b'\r' && term_end < data.len() && data[term_end] == b'\n' {
                term_end += 1;
            }
            let end = if keepends { term_end } else { i };
            parts.push(new_bytes_like(args[0], &data[start..end]));
            start = term_end;
            i = term_end;
        } else {
            i += 1;
        }
    }
    if start < data.len() {
        parts.push(new_bytes_like(args[0], &data[start..]));
    }
    Ok(pyre_object::w_list_new(parts))
}

/// `stringmethods.py:descr_expandtabs` — replace each `\t` with spaces
/// up to the next multiple of `tabsize`, measured from the start of the
/// current line (the column resets on `\n` / `\r`); a non-positive
/// `tabsize` drops tabs entirely.
fn bytes_method_expandtabs(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "expandtabs")?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let tabsize = match pos
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "tabsize"))
    {
        Some(t) if !t.is_null() => crate::builtins::space_index_w(t)?,
        _ => 8,
    };
    let mut out: Vec<u8> = Vec::with_capacity(data.len());
    let mut col: i64 = 0;
    for &b in data {
        match b {
            b'\t' => {
                if tabsize > 0 {
                    let incr = tabsize - (col % tabsize);
                    col += incr;
                    out.resize(out.len() + incr as usize, b' ');
                }
            }
            b'\n' | b'\r' => {
                out.push(b);
                col = 0;
            }
            _ => {
                out.push(b);
                col += 1;
            }
        }
    }
    Ok(new_bytes_like(args[0], &out))
}

/// `bytesobject.py:descr_maketrans` — build a 256-byte translation table
/// mapping each byte of `frm` to the byte at the same index in `to`;
/// the two bytes-like arguments must have equal length.  Bytes not in
/// `frm` map to themselves.
fn bytes_maketrans(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(
            "maketrans() takes exactly two arguments",
        ));
    }
    let frm = require_bytes_like(args[0])?;
    let to = require_bytes_like(args[1])?;
    if frm.len() != to.len() {
        return Err(crate::PyError::value_error(
            "maketrans arguments must have same length",
        ));
    }
    let mut table: Vec<u8> = (0..=255u8).collect();
    for (&f, &t) in frm.iter().zip(to.iter()) {
        table[f as usize] = t;
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&table))
}

/// `_PyBytes_FromHex` — parse a hex string into bytes.  ASCII whitespace
/// is skipped between byte pairs (but not within one); a stray nibble at
/// the end raises the even-count error, any other non-hex byte raises
/// the positional error.
fn parse_hex_string(args: &[PyObjectRef]) -> Result<Vec<u8>, crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(
            "fromhex() takes exactly one argument",
        ));
    }
    let bytes: &[u8] = match args.first() {
        Some(&a) if unsafe { pyre_object::is_str(a) } => {
            unsafe { pyre_object::w_str_get_value(a) }.as_bytes()
        }
        Some(&a) if unsafe { pyre_object::bytesobject::is_bytes_like(a) } => unsafe {
            pyre_object::bytesobject::bytes_like_data(a)
        },
        Some(&a) => {
            return Err(crate::PyError::type_error(format!(
                "fromhex() argument must be str or bytes-like, not {}",
                unsafe { (*(*a).ob_type).name }
            )));
        }
        None => {
            return Err(crate::PyError::type_error(
                "fromhex() takes exactly one argument",
            ));
        }
    };
    let nibble = |b: u8| -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    };
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        // `Py_ISSPACE`: space, tab, newline, vertical tab, form feed,
        // carriage return.  (`u8::is_ascii_whitespace` omits 0x0b.)
        if matches!(bytes[i], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
            i += 1;
            continue;
        }
        let Some(top) = nibble(bytes[i]) else {
            return Err(crate::PyError::value_error(format!(
                "non-hexadecimal number found in fromhex() arg at position {i}"
            )));
        };
        i += 1;
        if i >= bytes.len() {
            return Err(crate::PyError::value_error(
                "fromhex() arg must contain an even number of hexadecimal digits",
            ));
        }
        let Some(bot) = nibble(bytes[i]) else {
            return Err(crate::PyError::value_error(format!(
                "non-hexadecimal number found in fromhex() arg at position {i}"
            )));
        };
        i += 1;
        out.push((top << 4) | bot);
    }
    Ok(out)
}

// classmethod: args[0] is the bound cls, args[1] the hex string.
// `intobject.py:62 descr_from_bytes` — classmethod
// `(bytes, byteorder='big', *, signed=False)`.  `byteorder` is
// positional-or-keyword; `signed` is keyword-only.  Bound `cls` arrives
// at `args[0]`; the base type returns a plain int, a subclass routes
// through `cls(value)`.
fn int_from_bytes(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let cls = pos.first().copied().unwrap_or(pyre_object::PY_NULL);
    // `bytes` and `byteorder` are the only positional parameters; `signed`
    // is keyword-only, so a third positional is an error.
    if pos.len() > 3 {
        return Err(crate::PyError::type_error(format!(
            "from_bytes() takes at most 2 positional arguments ({} given)",
            pos.len() - 1
        )));
    }
    // `byteorder` and `signed` are the only keywords the gateway signature
    // accepts; anything else is an unexpected-keyword TypeError.
    crate::builtins::kwarg_reject_unknown(kwargs, &["byteorder", "signed"], "from_bytes")?;
    let data_obj = pos.get(1).copied().ok_or_else(|| {
        crate::PyError::type_error("from_bytes() missing required argument 'bytes' (pos 1)")
    })?;
    // `makebytesdata_w` — the buffer protocol, else an iterable of ints.
    let bytes: Vec<u8> = if unsafe { pyre_object::bytesobject::is_bytes_like(data_obj) } {
        unsafe { pyre_object::bytesobject::bytes_like_data(data_obj).to_vec() }
    } else {
        let items = crate::builtins::collect_iterable(data_obj)?;
        let mut v = Vec::with_capacity(items.len());
        for it in items {
            let n = crate::baseobjspace::int_w(it)?;
            if !(0..=255).contains(&n) {
                return Err(crate::PyError::value_error(
                    "bytes must be in range(0, 256)",
                ));
            }
            v.push(n as u8);
        }
        v
    };
    // byteorder is positional-or-keyword; supplying both is an error rather
    // than the keyword silently winning.
    let byteorder_kw = crate::builtins::kwarg_get(kwargs, "byteorder");
    let byteorder_pos = pos.get(2).copied();
    if byteorder_kw.is_some() && byteorder_pos.is_some() {
        return Err(crate::PyError::type_error(
            "got multiple values for argument 'byteorder'",
        ));
    }
    // `byteorder='text'` unwraps through `space.text_w`; a non-str value is a
    // TypeError, and only a str that is neither 'little'/'big' is a ValueError.
    let little_endian = match byteorder_pos.or(byteorder_kw) {
        None => false,
        Some(b) if unsafe { pyre_object::is_str(b) } => {
            match unsafe { pyre_object::w_str_get_value(b) } {
                "little" => true,
                "big" => false,
                _ => {
                    return Err(crate::PyError::value_error(
                        "byteorder must be either 'little' or 'big'",
                    ));
                }
            }
        }
        Some(b) => {
            let tname = unsafe { (*(*b).ob_type).name };
            return Err(crate::PyError::type_error(format!(
                "expected str, got {tname} object"
            )));
        }
    };
    let signed = crate::builtins::kwarg_get(kwargs, "signed")
        .map(crate::baseobjspace::is_true)
        .transpose()?
        .unwrap_or(false);
    let mut val = malachite_bigint::BigInt::from(0);
    if little_endian {
        for &b in bytes.iter().rev() {
            val = (val << 8) + malachite_bigint::BigInt::from(b);
        }
    } else {
        for &b in &bytes {
            val = (val << 8) + malachite_bigint::BigInt::from(b);
        }
    }
    let n = bytes.len();
    if signed && n > 0 {
        let sign_probe = if little_endian {
            bytes[n - 1]
        } else {
            bytes[0]
        };
        if sign_probe & 0x80 != 0 {
            val -= malachite_bigint::BigInt::from(1) << (8 * n);
        }
    }
    let w_result = if pyre_object::jit_bigint_to_i64_fits(&val) != 0 {
        pyre_object::w_int_new(pyre_object::jit_bigint_to_i64_value(&val))
    } else {
        pyre_object::w_long_new(val)
    };
    let base = crate::typedef::gettypeobject(&pyre_object::pyobject::INT_TYPE);
    if cls.is_null() || crate::baseobjspace::is_w(cls, base) {
        Ok(w_result)
    } else {
        crate::call::call_function_impl_result(cls, &[w_result])
    }
}

// `bytesobject.py:587 descr_fromhex` / `bytearrayobject.py:207
// descr_fromhex` — build the base type's value, then route through
// `cls(value)` when called on a subclass.
fn bytes_fromhex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let out = parse_hex_string(&args[1..])?;
    let w_bytes = pyre_object::bytesobject::w_bytes_from_bytes(&out);
    let base = crate::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE);
    if cls.is_null() || crate::baseobjspace::is_w(cls, base) {
        Ok(w_bytes)
    } else {
        crate::call::call_function_impl_result(cls, &[w_bytes])
    }
}

fn bytearray_fromhex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let out = parse_hex_string(&args[1..])?;
    let w_bytearray = pyre_object::bytearrayobject::w_bytearray_from_bytes(&out);
    let base = crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE);
    if cls.is_null() || crate::baseobjspace::is_w(cls, base) {
        Ok(w_bytearray)
    } else {
        crate::call::call_function_impl_result(cls, &[w_bytearray])
    }
}

/// `pypy/objspace/std/bytesobject.py W_BytesObject.descr_hex` —
///
/// ```python
/// def descr_hex(self, space, w_sep=None, w_bytes_per_sep=1):
///     ...
/// ```
///
/// Returns a string of hex pairs.  Optional `sep` (single byte/char)
/// inserts between pairs; `bytes_per_sep` controls the grouping.
pub(crate) fn bytes_method_hex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "hex")?;
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["sep", "bytes_per_sep"], "hex")?;
    crate::builtins::kwarg_reject_duplicate(kwargs, "hex", "sep", pos.get(1).is_some())?;
    crate::builtins::kwarg_reject_duplicate(kwargs, "hex", "bytes_per_sep", pos.get(2).is_some())?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(pos[0]) };
    // No sep / default grouping — produces "ffff" for [0xff, 0xff].
    // The sep + bytes_per_sep kwargs are deferred until first observed
    // need; CPython callers without args hit the hot path.
    let sep_arg = pos
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "sep"));
    if sep_arg.is_none() {
        let mut out = String::with_capacity(data.len() * 2);
        for &b in data {
            out.push_str(&format!("{:02x}", b));
        }
        return Ok(pyre_object::w_str_new(&out));
    }
    // `pypy/objspace/std/bytearrayobject.py:645-687 _binascii_hexstr`
    // sep validation — must be a length-1 ASCII string or length-1
    // bytes; otherwise ValueError per PyPy.
    let sep_obj = sep_arg.unwrap();
    let sep_char: char = if unsafe { pyre_object::is_str(sep_obj) } {
        let s = unsafe { pyre_object::w_str_get_value(sep_obj) };
        let mut chars = s.chars();
        let first = chars.next().ok_or_else(|| {
            crate::PyError::new(crate::PyErrorKind::ValueError, "sep must be length 1.")
        })?;
        if chars.next().is_some() {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "sep must be length 1.",
            ));
        }
        if (first as u32) >= 0x80 {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "sep must be ASCII.",
            ));
        }
        first
    } else if unsafe { pyre_object::is_bytes(sep_obj) } {
        let sep_bytes = unsafe { pyre_object::bytesobject::bytes_like_data(sep_obj) };
        if sep_bytes.len() != 1 {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "sep must be length 1.",
            ));
        }
        sep_bytes[0] as char
    } else {
        return Err(crate::PyError::type_error("sep must be str or bytes"));
    };
    let sep_str = sep_char.to_string();
    // `bytearrayobject.py:680-692` — positive `bytes_per_sep` groups
    // from the right (default), negative groups from the left; zero
    // disables separators entirely.
    let raw_group: i64 = match pos
        .get(2)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "bytes_per_sep"))
    {
        Some(o) => crate::baseobjspace::int_w(o)?,
        None => 1,
    };
    let group = raw_group.unsigned_abs() as usize;
    let group_from_left = raw_group < 0;
    let mut out = String::with_capacity(data.len() * 2 + data.len());
    for (i, b) in data.iter().enumerate() {
        if i > 0 && group != 0 {
            let boundary = if group_from_left {
                i % group == 0
            } else {
                (data.len() - i) % group == 0
            };
            if boundary {
                out.push_str(&sep_str);
            }
        }
        out.push_str(&format!("{:02x}", b));
    }
    Ok(pyre_object::w_str_new(&out))
}

/// interp_codecs.py:298/363 — encode-only handlers raise TypeError on decode
fn decode_error_encode_only_handler() -> crate::PyError {
    crate::PyError::type_error("don't know how to handle UnicodeDecodeError in error callback")
}

/// interp_exceptions.py:1061-1070 W_UnicodeDecodeError.descr_str format
fn unicode_decode_error_msg(
    codec: &str,
    data: &[u8],
    start: usize,
    end: usize,
    reason: &str,
) -> String {
    if end == start + 1 {
        format!(
            "'{codec}' codec can't decode byte 0x{:02x} in position {start}: {reason}",
            data[start]
        )
    } else {
        format!(
            "'{codec}' codec can't decode bytes in position {start}-{}: {reason}",
            end - 1
        )
    }
}

/// unicodehelper.py:13-23 decode_error_handler — raises a structured
/// UnicodeDecodeError, mirroring `OperationError(space.w_UnicodeDecodeError,
/// space.newtuple([encoding, w_s, start, end, msg]))`.  Populates the
/// `.encoding`/`.object`/`.start`/`.end`/`.reason` fields per
/// `W_UnicodeDecodeError.descr_init` (interp_exceptions.py:1041-1059) so
/// the caught exception carries the full attribute set, not just a message.
/// `.object` holds the whole bytes buffer; `start`/`end` index into it.
pub(crate) fn unicode_decode_error(
    encoding: &str,
    data: &[u8],
    start: usize,
    end: usize,
    reason: &str,
) -> crate::PyError {
    let w_encoding = pyre_object::w_str_new(encoding);
    let w_object = pyre_object::w_bytes_from_bytes(data);
    let w_start = pyre_object::w_int_new(start as i64);
    let w_end = pyre_object::w_int_new(end as i64);
    let w_reason = pyre_object::w_str_new(reason);
    // Eager message for PyError.message; descr_str recomputes the same
    // text from the fields (display.rs unicode_decode_error_str).
    let msg = unicode_decode_error_msg(encoding, data, start, end, reason);
    let exc = pyre_object::interp_exceptions::w_exception_new(
        pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError,
        &msg,
    );
    unsafe {
        pyre_object::interp_exceptions::w_exception_set_encoding(exc, w_encoding);
        pyre_object::interp_exceptions::w_exception_set_object(exc, w_object);
        pyre_object::interp_exceptions::w_exception_set_start(exc, w_start);
        pyre_object::interp_exceptions::w_exception_set_end(exc, w_end);
        pyre_object::interp_exceptions::w_exception_set_reason(exc, w_reason);
        // W_BaseException.descr_init: args_w = [encoding, object, start, end, reason]
        let args_list =
            pyre_object::w_list_new(vec![w_encoding, w_object, w_start, w_end, w_reason]);
        pyre_object::interp_exceptions::w_exception_set_args(exc, args_list);
        crate::PyError::from_exc_object(exc)
    }
}

/// interp_exceptions.py:1175-1191 W_UnicodeEncodeError.descr_str format.
/// `w_object` is the str being encoded; the bad code point is read at
/// `start` through the surrogate-aware WTF-8 view.
fn unicode_encode_error_msg(
    codec: &str,
    w_object: PyObjectRef,
    start: usize,
    end: usize,
    reason: &str,
) -> String {
    if end == start + 1 {
        let badchar = unsafe {
            pyre_object::w_str_get_wtf8(w_object)
                .code_points()
                .nth(start)
                .map(|c| c.to_u32())
                .unwrap_or(0)
        };
        let badchar_repr = if badchar <= 0xff {
            format!("'\\x{badchar:02x}'")
        } else if badchar <= 0xffff {
            format!("'\\u{badchar:04x}'")
        } else {
            format!("'\\U{badchar:08x}'")
        };
        format!(
            "'{codec}' codec can't encode character {badchar_repr} in position {start}: {reason}"
        )
    } else {
        format!(
            "'{codec}' codec can't encode characters in position {start}-{}: {reason}",
            end - 1
        )
    }
}

/// unicodehelper.py encode_error_handler — raises a structured
/// UnicodeEncodeError, mirroring `OperationError(space.w_UnicodeEncodeError,
/// space.newtuple([encoding, w_obj, start, end, msg]))`.  Populates the
/// `.encoding`/`.object`/`.start`/`.end`/`.reason` fields per
/// `W_UnicodeEncodeError.descr_init` (interp_exceptions.py:1153-1173) so the
/// caught exception carries the full attribute set, not just a message.
/// `.object` holds the whole str; `start`/`end` index code points into it.
pub(crate) fn unicode_encode_error(
    encoding: &str,
    w_object: PyObjectRef,
    start: usize,
    end: usize,
    reason: &str,
) -> crate::PyError {
    let w_encoding = pyre_object::w_str_new(encoding);
    let w_start = pyre_object::w_int_new(start as i64);
    let w_end = pyre_object::w_int_new(end as i64);
    let w_reason = pyre_object::w_str_new(reason);
    // Eager message for PyError.message; descr_str recomputes the same text
    // from the fields (display.rs unicode_encode_error_str).
    let msg = unicode_encode_error_msg(encoding, w_object, start, end, reason);
    let exc = pyre_object::interp_exceptions::w_exception_new(
        pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError,
        &msg,
    );
    unsafe {
        pyre_object::interp_exceptions::w_exception_set_encoding(exc, w_encoding);
        pyre_object::interp_exceptions::w_exception_set_object(exc, w_object);
        pyre_object::interp_exceptions::w_exception_set_start(exc, w_start);
        pyre_object::interp_exceptions::w_exception_set_end(exc, w_end);
        pyre_object::interp_exceptions::w_exception_set_reason(exc, w_reason);
        // W_BaseException.descr_init: args_w = [encoding, object, start, end, reason]
        let args_list =
            pyre_object::w_list_new(vec![w_encoding, w_object, w_start, w_end, w_reason]);
        pyre_object::interp_exceptions::w_exception_set_args(exc, args_list);
        crate::PyError::from_exc_object(exc)
    }
}

/// unicodehelper.py:15-22 — strict errorhandler raises UnicodeDecodeError
fn utf8_strict_handler(
    data: &[u8],
    start: usize,
    end: usize,
    reason: &str,
) -> Result<(), crate::PyError> {
    Err(unicode_decode_error("utf-8", data, start, end, reason))
}

/// Handle a decode error for non-strict modes.
/// Returns replacement text to append to `out`, or Err for fatal handlers.
/// `start` and `end` define the error span in `data`.
fn utf8_error_handler(
    err_mode: &str,
    data: &[u8],
    start: usize,
    end: usize,
    reason: &str,
    out: &mut Wtf8Buf,
) -> Result<usize, crate::PyError> {
    match err_mode {
        "strict" => {
            utf8_strict_handler(data, start, end, reason)?;
            unreachable!()
        }
        "ignore" => Ok(end),
        "replace" => {
            out.push_char('\u{FFFD}');
            Ok(end)
        }
        // interp_codecs.py:536-555 surrogateescape_errors (decode branch).
        // Escape up to four non-ASCII bytes as lone surrogates 0xdc00+c;
        // refuse to escape ASCII bytes; if none consumed, re-raise.
        "surrogateescape" => {
            let mut consumed = 0;
            while consumed < 4 && consumed < end - start {
                let c = data[start + consumed];
                if c < 128 {
                    // Refuse to escape ASCII bytes.
                    break;
                }
                out.push(CodePoint::from_u32(0xDC00 + c as u32).unwrap());
                consumed += 1;
            }
            if consumed == 0 {
                // codec complained about ASCII byte.
                return Err(unicode_decode_error("utf-8", data, start, end, reason));
            }
            Ok(start + consumed)
        }
        // interp_codecs.py:476-510 surrogatepass_errors (decode branch).
        // Decode a single three-byte UTF-8 surrogate (ED A0..BF 80..BF) at
        // `start`; if it is not a surrogate, re-raise the original error.
        "surrogatepass" => {
            let ch0 = if data.len() > start {
                data[start] as i32
            } else {
                -1
            };
            let ch1 = if data.len() > start + 1 {
                data[start + 1] as i32
            } else {
                -1
            };
            let ch2 = if data.len() > start + 2 {
                data[start + 2] as i32
            } else {
                -1
            };
            let mut ch = 0;
            if ch1 != -1
                && ch2 != -1
                && ch0 & 0xf0 == 0xe0
                && ch1 & 0xc0 == 0x80
                && ch2 & 0xc0 == 0x80
            {
                // it's a three-byte code
                ch = ((ch0 & 0x0f) << 12) + ((ch1 & 0x3f) << 6) + (ch2 & 0x3f);
            }
            if !(0xd800..=0xdfff).contains(&ch) {
                // it's not a surrogate - fail
                ch = 0;
            }
            if ch == 0 {
                return Err(unicode_decode_error("utf-8", data, start, end, reason));
            }
            out.push(CodePoint::from_u32(ch as u32).unwrap());
            Ok(start + 3)
        }
        "backslashreplace" => {
            for &b in &data[start..end] {
                out.push_str(&format!("\\x{:02x}", b));
            }
            Ok(end)
        }
        "xmlcharrefreplace" | "namereplace" => Err(decode_error_encode_only_handler()),
        _ => Err(crate::PyError::new(
            crate::PyErrorKind::LookupError,
            format!("unknown error handler name '{err_mode}'"),
        )),
    }
}

/// runicode.py:118-127 _utf8_code_length table
/// Indexed by (byte - 0x80).  0 = invalid start, 2/3/4 = expected sequence length.
const UTF8_CODE_LENGTH: [u8; 128] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 80-8F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 90-9F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // A0-AF
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // B0-BF
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // C0-C1 + C2-CF
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // D0-DF
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, // E0-EF
    4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // F0-F4 + F5-FF
];

/// rutf8.py:326-328
fn invalid_cont_byte(b: u8) -> bool {
    (b as i8) >= -0x40 // equivalent: b < 0x80 || b > 0xBF
}

/// rutf8.py:339-343
/// Surrogates (ED A0..BF) are always rejected — pyre's Rust String cannot
/// hold surrogate codepoints; the error handler deals with surrogatepass.
fn invalid_byte_2_of_3(ch1: u8, ch2: u8) -> bool {
    invalid_cont_byte(ch2) || (ch1 == 0xE0 && ch2 < 0xA0) || (ch1 == 0xED && ch2 > 0x9F)
}

/// rutf8.py:345-348
fn invalid_byte_2_of_4(ch1: u8, ch2: u8) -> bool {
    invalid_cont_byte(ch2) || (ch1 == 0xF0 && ch2 < 0x90) || (ch1 == 0xF4 && ch2 > 0x8F)
}

/// interp_locale.py:42-46 `charp2uni` — decode a C string the way
/// `str(bytes, 'utf-8', 'surrogateescape')` does: valid UTF-8 passes
/// through and any other byte becomes a lone `0xDC00 + byte` surrogate.
/// `surrogateescape` rescues every byte, so the decode never fails.
pub(crate) fn charp2uni(data: &[u8]) -> PyObjectRef {
    let decoded = decode_utf8_with_errors(data, "surrogateescape")
        .expect("surrogateescape rescues every byte, so the decode never fails");
    pyre_object::w_str_from_wtf8(decoded)
}

/// unicodehelper.py:377-537 _str_decode_utf8_slowpath
/// Structural port of PyPy's _utf8_code_length state machine.
/// PyPy appends raw UTF-8 bytes to a StringBuilder; Rust reconstructs
/// Unicode scalar values via char::from_u32.  Surrogates are always
/// rejected by invalid_byte_2_of_3 and routed to the error handler.
fn decode_utf8_with_errors(data: &[u8], err_mode: &str) -> Result<Wtf8Buf, crate::PyError> {
    let s = data;
    let size = s.len();
    let mut result = Wtf8Buf::new();
    let mut pos = 0;
    let final_ = true; // pyre always decodes complete buffers

    while pos < size {
        let ordch1 = s[pos];
        // unicodehelper.py:394 fast path for ASCII
        if ordch1 <= 0x7F {
            result.push_char(ordch1 as char);
            pos += 1;
            continue;
        }

        // unicodehelper.py:399
        let n = UTF8_CODE_LENGTH[(ordch1 - 0x80) as usize];

        // unicodehelper.py:400 truncated sequence
        if pos + n as usize > size {
            let charsleft = size - pos - 1; // 0, 1, or 2
            // unicodehelper.py:407
            if charsleft == 0 {
                if !final_ {
                    break;
                }
                pos = utf8_error_handler(
                    err_mode,
                    s,
                    pos,
                    pos + 1,
                    "unexpected end of data",
                    &mut result,
                )?;
                continue;
            }
            let ordch2 = s[pos + 1];
            if n == 3 {
                // unicodehelper.py:417-434
                if invalid_byte_2_of_3(ordch1, ordch2) {
                    pos = utf8_error_handler(
                        err_mode,
                        s,
                        pos,
                        pos + 1,
                        "invalid continuation byte",
                        &mut result,
                    )?;
                    continue;
                }
                if !final_ {
                    break;
                }
                pos = utf8_error_handler(
                    err_mode,
                    s,
                    pos,
                    pos + 2,
                    "unexpected end of data",
                    &mut result,
                )?;
                continue;
            } else if n == 4 {
                // unicodehelper.py:435-459
                if invalid_byte_2_of_4(ordch1, ordch2) {
                    pos = utf8_error_handler(
                        err_mode,
                        s,
                        pos,
                        pos + 1,
                        "invalid continuation byte",
                        &mut result,
                    )?;
                    continue;
                }
                if charsleft == 2 && invalid_cont_byte(s[pos + 2]) {
                    pos = utf8_error_handler(
                        err_mode,
                        s,
                        pos,
                        pos + 2,
                        "invalid continuation byte",
                        &mut result,
                    )?;
                    continue;
                }
                if !final_ {
                    break;
                }
                pos = utf8_error_handler(
                    err_mode,
                    s,
                    pos,
                    pos + charsleft + 1,
                    "unexpected end of data",
                    &mut result,
                )?;
                continue;
            }
            unreachable!("n must be 3 or 4 when charsleft > 0");
        }

        // unicodehelper.py:462 n == 0 → invalid start byte
        if n == 0 {
            pos = utf8_error_handler(err_mode, s, pos, pos + 1, "invalid start byte", &mut result)?;
            continue;
        }

        if n == 2 {
            // unicodehelper.py:471-482
            let ordch2 = s[pos + 1];
            if invalid_cont_byte(ordch2) {
                pos = utf8_error_handler(
                    err_mode,
                    s,
                    pos,
                    pos + 1,
                    "invalid continuation byte",
                    &mut result,
                )?;
                continue;
            }
            // 110yyyyy 10zzzzzz
            let cp = ((ordch1 as u32 & 0x1F) << 6) | (ordch2 as u32 & 0x3F);
            if let Some(c) = char::from_u32(cp) {
                result.push_char(c);
            }
            pos += 2;
        } else if n == 3 {
            // unicodehelper.py:484-503
            let ordch2 = s[pos + 1];
            let ordch3 = s[pos + 2];
            if invalid_byte_2_of_3(ordch1, ordch2) {
                pos = utf8_error_handler(
                    err_mode,
                    s,
                    pos,
                    pos + 1,
                    "invalid continuation byte",
                    &mut result,
                )?;
                continue;
            }
            if invalid_cont_byte(ordch3) {
                pos = utf8_error_handler(
                    err_mode,
                    s,
                    pos,
                    pos + 2,
                    "invalid continuation byte",
                    &mut result,
                )?;
                continue;
            }
            // 1110xxxx 10yyyyyy 10zzzzzz
            let cp = ((ordch1 as u32 & 0x0F) << 12)
                | ((ordch2 as u32 & 0x3F) << 6)
                | (ordch3 as u32 & 0x3F);
            if let Some(c) = char::from_u32(cp) {
                result.push_char(c);
            }
            pos += 3;
        } else {
            // n == 4, unicodehelper.py:505-532
            let ordch2 = s[pos + 1];
            let ordch3 = s[pos + 2];
            let ordch4 = s[pos + 3];
            if invalid_byte_2_of_4(ordch1, ordch2) {
                pos = utf8_error_handler(
                    err_mode,
                    s,
                    pos,
                    pos + 1,
                    "invalid continuation byte",
                    &mut result,
                )?;
                continue;
            }
            if invalid_cont_byte(ordch3) {
                pos = utf8_error_handler(
                    err_mode,
                    s,
                    pos,
                    pos + 2,
                    "invalid continuation byte",
                    &mut result,
                )?;
                continue;
            }
            if invalid_cont_byte(ordch4) {
                pos = utf8_error_handler(
                    err_mode,
                    s,
                    pos,
                    pos + 3,
                    "invalid continuation byte",
                    &mut result,
                )?;
                continue;
            }
            // 11110www 10xxxxxx 10yyyyyy 10zzzzzz
            let cp = ((ordch1 as u32 & 0x07) << 18)
                | ((ordch2 as u32 & 0x3F) << 12)
                | ((ordch3 as u32 & 0x3F) << 6)
                | (ordch4 as u32 & 0x3F);
            if let Some(c) = char::from_u32(cp) {
                result.push_char(c);
            }
            pos += 4;
        }
    }
    Ok(result)
}

/// bytesobject.py descr_decode → stringmethods.py:196 decode_object
pub(crate) fn bytes_method_decode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "decode")?;
    // `bytes.decode(encoding='utf-8', errors='strict')` — both parameters
    // are positional-or-keyword, so accept them from either side.
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["encoding", "errors"], "decode")?;
    // `encoding` is positional-or-keyword at position 1; giving it both ways is
    // a TypeError (the rarer 3-positional `errors` over-count is not modelled).
    if pos.len() > 1 && crate::builtins::kwarg_get(kwargs, "encoding").is_some() {
        return Err(crate::PyError::type_error(
            "argument for decode() given by name ('encoding') and position (1)",
        ));
    }
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(pos[0]) };
    let w_encoding = pos
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "encoding"));
    let w_errors = pos
        .get(2)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "errors"));
    // unicodeobject.py:1669 — encoding/errors must be str (space.text_w)
    if let Some(enc) = w_encoding {
        if !unsafe { pyre_object::is_str(enc) } && !unsafe { pyre_object::is_none(enc) } {
            let tn = unsafe { (*(*enc).ob_type).name };
            return Err(crate::PyError::type_error(format!(
                "decode() argument 'encoding' must be str, not {tn}",
            )));
        }
    }
    if let Some(err) = w_errors {
        if !unsafe { pyre_object::is_str(err) } && !unsafe { pyre_object::is_none(err) } {
            let tn = unsafe { (*(*err).ob_type).name };
            return Err(crate::PyError::type_error(format!(
                "decode() argument 'errors' must be str, not {tn}",
            )));
        }
    }
    let encoding = match w_encoding {
        Some(e) if unsafe { pyre_object::is_str(e) } => unsafe {
            pyre_object::w_str_get_value(e).to_string()
        },
        _ => "utf-8".to_string(),
    };
    let errors = match w_errors {
        Some(e) if unsafe { pyre_object::is_str(e) } => unsafe {
            pyre_object::w_str_get_value(e).to_string()
        },
        _ => "strict".to_string(),
    };
    let s = decode_bytes_to_wtf8(data, &encoding, errors.as_str())?;
    Ok(pyre_object::w_str_from_wtf8(s))
}

/// Decode `data` under `encoding`/`errors` into a WTF-8 string, dispatching on
/// the codec name the same way `bytes.decode` does.
pub(crate) fn decode_bytes_to_wtf8(
    data: &[u8],
    encoding: &str,
    errors: &str,
) -> Result<Wtf8Buf, crate::PyError> {
    let err_mode = errors;
    let enc_lower = encoding.to_ascii_lowercase().replace('_', "-");
    let s = match enc_lower.as_str() {
        "utf-8" | "utf8" | "u8" => decode_utf8_with_errors(data, err_mode)?,
        "ascii" | "us-ascii" | "646" => {
            let mut out = Wtf8Buf::new();
            for (i, &b) in data.iter().enumerate() {
                if b >= 0x80 {
                    match err_mode {
                        "strict" => {
                            return Err(unicode_decode_error(
                                "ascii",
                                data,
                                i,
                                i + 1,
                                "ordinal not in range(128)",
                            ));
                        }
                        "ignore" => continue,
                        "replace" => {
                            out.push_char('\u{FFFD}');
                            continue;
                        }
                        // surrogateescape escapes the non-ASCII byte as a lone
                        // surrogate 0xdc00+b (interp_codecs.py:536-555).
                        "surrogateescape" => {
                            out.push(CodePoint::from_u32(0xDC00 + b as u32).unwrap());
                            continue;
                        }
                        // surrogatepass only decodes three-byte UTF-8 surrogate
                        // sequences; a single non-ASCII byte is not one, so it
                        // re-raises (interp_codecs.py:476-510).
                        "surrogatepass" => {
                            return Err(unicode_decode_error(
                                "ascii",
                                data,
                                i,
                                i + 1,
                                "ordinal not in range(128)",
                            ));
                        }
                        "backslashreplace" => {
                            out.push_str(&format!("\\x{:02x}", b));
                            continue;
                        }
                        "xmlcharrefreplace" | "namereplace" => {
                            return Err(decode_error_encode_only_handler());
                        }
                        _ => {
                            return Err(crate::PyError::new(
                                crate::PyErrorKind::LookupError,
                                format!("unknown error handler name '{err_mode}'"),
                            ));
                        }
                    }
                }
                out.push_char(b as char);
            }
            out
        }
        "latin-1" | "latin1" | "iso-8859-1" | "8859" => {
            Wtf8Buf::from_string(data.iter().map(|&b| b as char).collect::<String>())
        }
        "raw-unicode-escape" => crate::type_methods::decode_raw_unicode_escape(data)?,
        _ => {
            if let Some(result) = crate::type_methods::decode_utf16_32(data, &enc_lower, err_mode) {
                result?
            } else {
                let w_data = pyre_object::bytesobject::w_bytes_from_bytes(data);
                let w_text = crate::module::_codecs::decode_text_codec(w_data, encoding, err_mode)?;
                unsafe { pyre_object::w_str_get_wtf8(w_text) }.to_wtf8_buf()
            }
        }
    };
    Ok(s)
}

/// PyPy: bytesobject.py descr_repr — returns a quoted literal like `b'hello'`.
fn bytes_method_bytes(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `W_BytesObject.descr_bytes` ("convert this value to exact type bytes"):
    // an exact `bytes` returns itself; a subclass returns a fresh exact-bytes
    // copy of its value.
    crate::type_methods::require_receiver(args, "__bytes__")?;
    let self_ = args[0];
    if unsafe { pyre_object::pyobject::is_exact_type(self_, &pyre_object::bytesobject::BYTES_TYPE) }
    {
        return Ok(self_);
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(unsafe {
        pyre_object::bytesobject::bytes_like_data(self_)
    }))
}

fn bytes_method_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "__repr__")?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    // Determine preferred quote: single unless the data contains single but
    // not double quote (matches CPython).
    let has_single = data.contains(&b'\'');
    let has_double = data.contains(&b'"');
    let quote: char = if has_single && !has_double { '"' } else { '\'' };
    let mut out = String::with_capacity(data.len() + 3);
    out.push('b');
    out.push(quote);
    for &b in data {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            q if q as char == quote => {
                out.push('\\');
                out.push(quote);
            }
            0x20..=0x7e => out.push(b as char),
            _ => out.push_str(&format!("\\x{:02x}", b)),
        }
    }
    out.push(quote);
    Ok(pyre_object::w_str_new(&out))
}

fn bytes_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let value = bytes_descr_new_impl(args)?;
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::bytesobject::BYTES_TYPE)? {
        // `bytes(b)` may return the argument unchanged, so rebuild a
        // fresh object before retagging to avoid aliasing the input.
        let data = unsafe { pyre_object::bytesobject::bytes_like_data(value).to_vec() };
        let fresh = pyre_object::bytesobject::w_bytes_from_bytes(&data);
        unsafe {
            (*fresh).w_class = sub;
        }
        return Ok(fresh);
    }
    Ok(value)
}

fn bytes_descr_new_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // args[0] = cls. `bytes(source=b'', encoding=None, errors=None)` —
    // every parameter is positional-or-keyword (bytesobject.py descr_new);
    // `encoding`/`errors` are only valid with a str source.
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    // pos[0] is the class; `bytes(source, encoding, errors)` accepts at most
    // three further positional arguments.
    if pos.len() > 4 {
        return Err(crate::PyError::type_error(&format!(
            "bytes() takes at most 3 arguments ({} given)",
            pos.len() - 1
        )));
    }
    crate::builtins::kwarg_reject_unknown(kwargs, &["source", "encoding", "errors"], "bytes")?;
    let source =
        crate::builtins::resolve_pos_or_kw(pos.get(1).copied(), kwargs, "source", "bytes", 1)?;
    let w_encoding =
        crate::builtins::resolve_pos_or_kw(pos.get(2).copied(), kwargs, "encoding", "bytes", 2)?;
    let w_errors =
        crate::builtins::resolve_pos_or_kw(pos.get(3).copied(), kwargs, "errors", "bytes", 3)?;
    // `text_or_none` unwrap_spec treats an explicit `None` as absent.
    let w_encoding = w_encoding.filter(|&e| !unsafe { pyre_object::is_none(e) });
    let w_errors = w_errors.filter(|&e| !unsafe { pyre_object::is_none(e) });
    let Some(arg) = source else {
        // No source → `bytes()` is empty; a stray encoding/errors with no
        // source is the "encoding or errors without sequence argument" error.
        if w_encoding.is_some() || w_errors.is_some() {
            return Err(crate::PyError::type_error(
                "encoding or errors without sequence argument",
            ));
        }
        return Ok(pyre_object::bytesobject::w_bytes_empty());
    };
    let has_codec = w_encoding.is_some() || w_errors.is_some();
    unsafe {
        if pyre_object::is_str(arg) {
            let encoding = match w_encoding {
                Some(e) if pyre_object::is_str(e) => pyre_object::w_str_get_value(e),
                _ => {
                    return Err(crate::PyError::type_error(
                        "string argument without an encoding",
                    ));
                }
            };
            let errors = match w_errors {
                Some(e) if pyre_object::is_str(e) => pyre_object::w_str_get_value(e),
                _ => "strict",
            };
            let encoded = crate::type_methods::encode_object(arg, encoding, errors)?;
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&encoded));
        }
        if has_codec {
            let which = if w_encoding.is_some() {
                "encoding"
            } else {
                "errors"
            };
            return Err(crate::PyError::type_error(format!(
                "{which} without string argument (got '{}' instead)",
                type_name_of(arg)
            )));
        }
        // bytesobject.py:560 — `bytes(bytes_obj)` on an exact `bytes` source
        // returns the argument unmodified (identity).  A subclass source falls
        // through (its bytes are copied); a subclass *request* is retagged by
        // `bytes_descr_new`, which copies before retagging.
        if pyre_object::pyobject::is_exact_type(arg, &pyre_object::bytesobject::BYTES_TYPE) {
            return Ok(arg);
        }
        // bytesobject.py:575 `invoke_bytes_method` — a `__bytes__` special
        // method takes precedence over the count / buffer / iterable paths;
        // its result is returned **unmodified** (even a bytes subclass), so the
        // exact object identity is preserved.  (bytearray does NOT honour
        // __bytes__.)
        if let Some(method) = crate::baseobjspace::lookup(arg, "__bytes__") {
            let w_bytes = crate::builtins::call_and_check(method, &[arg])?;
            if !pyre_object::bytesobject::is_bytes(w_bytes) {
                return Err(crate::PyError::type_error(format!(
                    "__bytes__ returned non-bytes (type {})",
                    type_name_of(w_bytes)
                )));
            }
            return Ok(w_bytes);
        }
        // newbytesdata_w_tail: `getindex_w(source, OverflowError)` — any object
        // exposing __index__ (not just an exact int) is a count of NUL bytes.
        if pyre_object::pyobject::is_int_or_long(arg)
            || crate::baseobjspace::lookup(arg, "__index__").is_some()
        {
            let n = match crate::baseobjspace::int_w(crate::baseobjspace::space_index(arg)?) {
                Ok(n) => n,
                Err(e) if e.kind == crate::PyErrorKind::OverflowError => {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::OverflowError,
                        format!(
                            "cannot fit '{}' into an index-sized integer",
                            crate::baseobjspace::object_functionstr_type_name(arg)
                        ),
                    ));
                }
                Err(e) => return Err(e),
            };
            // bytesobject.py:797 — negative count raises ValueError
            if n < 0 {
                return Err(crate::PyError::value_error("negative count"));
            }
            return Ok(pyre_object::bytesobject::w_bytes_from_bytes(
                &vec![0u8; n as usize],
            ));
        }
        // `_convert_from_buffer_or_iterable`: any buffer exporter — bytes,
        // bytearray, `array.array`, memoryview — yields its raw buffer bytes
        // (`buffer_w(BUF_FULL_RO).as_str()`) before the iterable path; a
        // released memoryview raises first.
        if let Some(b) = crate::typedef::buffer_as_bytes_like(arg)? {
            return Ok(new_bytes_like(
                args[0],
                pyre_object::bytesobject::bytes_like_data(b),
            ));
        }
    }
    // `_from_byte_sequence_loop`: iterate the source, coercing each element
    // with `byte_w` (honours __index__ and range-checks 0..256, "bytes must be
    // in range(0, 256)"; a non-index element → "'X' object cannot be
    // interpreted as an integer").  A source with no __iter__ is the "cannot
    // convert" case; an error raised by __iter__/__next__ propagates unchanged.
    unsafe {
        let it = match crate::baseobjspace::iter(arg) {
            Ok(it) => it,
            Err(e) => {
                if crate::baseobjspace::lookup(arg, "__iter__").is_none() {
                    return Err(crate::PyError::type_error(format!(
                        "cannot convert '{}' object to bytes",
                        crate::baseobjspace::object_functionstr_type_name(arg)
                    )));
                }
                return Err(e);
            }
        };
        let mut buf = Vec::new();
        loop {
            match crate::baseobjspace::next(it) {
                Ok(item) => buf.push(crate::baseobjspace::byte_w(item, "bytes")?),
                Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
                Err(e) => return Err(e),
            }
        }
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buf))
    }
}

/// `space.byte_w` — extract a single byte (`0 <= v < 256`) from an index
/// argument; an invalid index raises TypeError, an out-of-range value ValueError.
fn bytearray_byte_arg(obj: PyObjectRef) -> Result<u8, crate::PyError> {
    // byte_w: getindex_w then range-check to [0, 256).
    let value = crate::baseobjspace::getindex_w(obj)?;
    if !(0..=255).contains(&value) {
        return Err(crate::PyError::value_error("byte must be in range(0, 256)"));
    }
    Ok(value as u8)
}

/// `pypy/objspace/std/bytearrayobject.py descr_inplace_mul` — repeat the
/// bytearray in place while preserving its identity.
fn bytearray_method_imul(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "__imul__", 1)?;
    let ba = args[0];
    // Follow CPython here: a non-index TypeError propagates rather than
    // returning NotImplemented.
    let w_index = crate::baseobjspace::space_index(args[1])?;
    let count = match crate::baseobjspace::int_w(w_index) {
        Ok(v) => v,
        Err(e) if e.kind == crate::PyErrorKind::OverflowError => {
            return Err(crate::PyError::new(
                crate::PyErrorKind::OverflowError,
                format!(
                    "cannot fit '{}' into an index-sized integer",
                    crate::baseobjspace::object_functionstr_type_name(args[1])
                ),
            ));
        }
        Err(e) => return Err(e),
    };
    unsafe {
        crate::builtins::bytearray_check_exports(ba)?;
        let vec = pyre_object::bytearrayobject::w_bytearray_vec_mut(ba);
        if count <= 0 {
            vec.clear();
        } else if count != 1 && !vec.is_empty() {
            vec.len().checked_mul(count as usize).ok_or_else(|| {
                crate::PyError::new(
                    crate::PyErrorKind::OverflowError,
                    "repeated bytes are too long",
                )
            })?;
            let orig = vec.clone();
            for _ in 1..count {
                vec.extend_from_slice(&orig);
            }
        }
    }
    Ok(ba)
}

/// `bytearrayobject.py:descr_append` — append one byte in place.
fn bytearray_method_append(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "append", 1)?;
    unsafe { crate::builtins::bytearray_check_exports(args[0])? };
    let b = bytearray_byte_arg(args[1])?;
    unsafe { pyre_object::bytearrayobject::w_bytearray_vec_mut(args[0]).push(b) };
    Ok(pyre_object::w_none())
}

/// `bytearrayobject.py:descr_extend` — append a bytes-like object's
/// bytes, or each integer yielded by an iterable.
fn bytearray_method_extend(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "extend", 1)?;
    unsafe { crate::builtins::bytearray_check_exports(args[0])? };
    let other = args[1];
    // Materialize the new bytes before mutating so `x.extend(x)` is safe.
    let appended: Vec<u8> = unsafe {
        if pyre_object::bytesobject::is_bytes_like(other) {
            pyre_object::bytesobject::bytes_like_data(other).to_vec()
        } else {
            // `PyObject_GetIter` — a non-iterable operand is the "can't extend" case.
            let it = crate::baseobjspace::iter(other).map_err(|e| {
                if e.kind == crate::PyErrorKind::TypeError {
                    crate::PyError::type_error(format!(
                        "can't extend bytearray with {}",
                        crate::baseobjspace::object_functionstr_type_name(other)
                    ))
                } else {
                    e
                }
            })?;
            let is_str = pyre_object::is_str(other);
            let mut appended = Vec::new();
            loop {
                match crate::baseobjspace::next(it) {
                    Ok(v) => {
                        let b = bytearray_byte_arg(v).map_err(|e| {
                            if is_str && e.kind == crate::PyErrorKind::TypeError {
                                crate::PyError::type_error(
                                    "expected iterable of integers; got: 'str'",
                                )
                            } else {
                                e
                            }
                        })?;
                        appended.push(b);
                    }
                    Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
                    Err(e) => return Err(e),
                }
            }
            appended
        }
    };
    unsafe {
        pyre_object::bytearrayobject::w_bytearray_vec_mut(args[0]).extend_from_slice(&appended)
    };
    Ok(pyre_object::w_none())
}

/// `bytearrayobject.py:descr_insert` — insert one byte before `index`,
/// clamping out-of-range indices (negative counts from the end).
fn bytearray_method_insert(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "insert", 2)?;
    unsafe { crate::builtins::bytearray_check_exports(args[0])? };
    let index = crate::builtins::space_index_w(args[1])?;
    let b = bytearray_byte_arg(args[2])?;
    unsafe {
        let vec = pyre_object::bytearrayobject::w_bytearray_vec_mut(args[0]);
        let len = vec.len() as i64;
        let i = if index < 0 { index + len } else { index };
        vec.insert(i.clamp(0, len) as usize, b);
    }
    Ok(pyre_object::w_none())
}

/// `bytearrayobject.py:descr_remove` — remove the first byte equal to
/// `value`; ValueError when absent.
fn bytearray_method_remove(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_at_least(args, "remove", 1)?;
    unsafe { crate::builtins::bytearray_check_exports(args[0])? };
    let b = bytearray_byte_arg(args[1])?;
    unsafe {
        let vec = pyre_object::bytearrayobject::w_bytearray_vec_mut(args[0]);
        match vec.iter().position(|&x| x == b) {
            Some(pos) => vec.remove(pos),
            None => {
                return Err(crate::PyError::value_error("value not found in bytearray"));
            }
        };
    }
    Ok(pyre_object::w_none())
}

/// `bytearrayobject.py:descr_pop` — remove and return the byte at
/// `index` (default last); IndexError when empty or out of range.
fn bytearray_method_pop(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "pop")?;
    unsafe {
        crate::builtins::bytearray_check_exports(args[0])?;
        let vec = pyre_object::bytearrayobject::w_bytearray_vec_mut(args[0]);
        let len = vec.len() as i64;
        if len == 0 {
            return Err(crate::PyError::new(
                crate::PyErrorKind::IndexError,
                "pop from empty bytearray",
            ));
        }
        let index = match args.get(1) {
            Some(&a) if !a.is_null() && !pyre_object::is_none(a) => {
                crate::builtins::space_index_w(a)?
            }
            _ => -1,
        };
        let i = if index < 0 { index + len } else { index };
        if i < 0 || i >= len {
            return Err(crate::PyError::new(
                crate::PyErrorKind::IndexError,
                "pop index out of range",
            ));
        }
        Ok(pyre_object::w_int_new(vec.remove(i as usize) as i64))
    }
}

/// `bytearrayobject.py:descr_reverse` — reverse the bytes in place.
fn bytearray_method_reverse(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "reverse")?;
    unsafe { pyre_object::bytearrayobject::w_bytearray_vec_mut(args[0]).reverse() };
    Ok(pyre_object::w_none())
}

/// `bytearrayobject.py:descr_clear` — empty the bytearray in place.
fn bytearray_method_clear(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "clear")?;
    unsafe {
        crate::builtins::bytearray_check_exports(args[0])?;
        pyre_object::bytearrayobject::w_bytearray_vec_mut(args[0]).clear();
    };
    Ok(pyre_object::w_none())
}

/// `bytearrayobject.py:descr_copy` — return a new bytearray with the
/// same bytes.
fn bytearray_method_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "copy")?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(data))
}

/// `bytearrayobject.py descr_releasebuffer` — the Python 3.12
/// `__release_buffer__` protocol entry for a released bytearray export.
fn bytearray_method_release_buffer(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unsafe { pyre_object::bytearrayobject::w_bytearray_exports_decref(args[0]) };
    Ok(pyre_object::w_none())
}

/// PyPy: bytearrayobject.py W_BytearrayObject.typedef
fn init_bytearray_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(bytearray_descr_new));
    dict_storage_store(
        ns,
        "__release_buffer__",
        make_builtin_function_with_arity("__release_buffer__", bytearray_method_release_buffer, 2),
    );
    // `bytearrayobject.py W_BytearrayObject.descr_decode` shares the
    // bytes decode machinery — `bytes_method_decode` already pulls the
    // payload via `bytes_like_data`, which handles both kinds.
    dict_storage_store(
        ns,
        "decode",
        make_builtin_function("decode", bytes_method_decode),
    );
    // The scalar-returning read-only methods (int / bool results) read
    // their payload via `bytes_like_data`, which handles both bytes and
    // bytearray, so they share the bytes implementations verbatim.
    dict_storage_store(ns, "find", make_builtin_function("find", bytes_method_find));
    dict_storage_store(
        ns,
        "rfind",
        make_builtin_function("rfind", bytes_method_rfind),
    );
    dict_storage_store(
        ns,
        "index",
        make_builtin_function("index", bytes_method_index),
    );
    dict_storage_store(
        ns,
        "rindex",
        make_builtin_function("rindex", bytes_method_rindex),
    );
    dict_storage_store(
        ns,
        "count",
        make_builtin_function("count", bytes_method_count),
    );
    dict_storage_store(
        ns,
        "startswith",
        make_builtin_function("startswith", bytes_method_startswith),
    );
    dict_storage_store(
        ns,
        "endswith",
        make_builtin_function("endswith", bytes_method_endswith),
    );
    dict_storage_store(
        ns,
        "isdigit",
        make_builtin_function("isdigit", bytes_method_isdigit),
    );
    dict_storage_store(
        ns,
        "isalpha",
        make_builtin_function("isalpha", bytes_method_isalpha),
    );
    dict_storage_store(
        ns,
        "isalnum",
        make_builtin_function("isalnum", bytes_method_isalnum),
    );
    dict_storage_store(
        ns,
        "isspace",
        make_builtin_function("isspace", bytes_method_isspace),
    );
    dict_storage_store(
        ns,
        "isascii",
        make_builtin_function("isascii", bytes_method_isascii),
    );
    dict_storage_store(
        ns,
        "isupper",
        make_builtin_function("isupper", bytes_method_isupper),
    );
    dict_storage_store(
        ns,
        "islower",
        make_builtin_function("islower", bytes_method_islower),
    );
    dict_storage_store(
        ns,
        "istitle",
        make_builtin_function("istitle", bytes_method_istitle),
    );
    dict_storage_store(
        ns,
        "__add__",
        make_builtin_function_with_arity(
            "__add__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                let a = args[0];
                let b = args[1];
                unsafe {
                    let a_data = pyre_object::bytesobject::bytes_like_data(a);
                    // descr_add: a non-buffer operand raises rather than
                    // concatenating as empty.
                    let Some(src) = buffer_as_bytes_like(b)? else {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::TypeError,
                            format!(
                                "can't concat {} to bytearray",
                                crate::type_methods::arg_type_name(b)
                            ),
                        ));
                    };
                    let b_data = pyre_object::bytesobject::bytes_like_data(src).to_vec();
                    let mut result = a_data.to_vec();
                    result.extend_from_slice(&b_data);
                    Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(
                        &result,
                    ))
                }
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__iadd__",
        make_builtin_function_with_arity(
            "__iadd__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                let ba = args[0];
                let other = args[1];
                unsafe {
                    crate::builtins::bytearray_check_exports(ba)?;
                    // descr_inplace_add: a non-buffer operand raises rather
                    // than silently leaving the bytearray unchanged.
                    let Some(src) = buffer_as_bytes_like(other)? else {
                        return Err(crate::PyError::new(
                            crate::PyErrorKind::TypeError,
                            format!(
                                "can't concat {} to bytearray",
                                crate::type_methods::arg_type_name(other)
                            ),
                        ));
                    };
                    let data = pyre_object::bytesobject::bytes_like_data(src).to_vec();
                    pyre_object::bytearrayobject::w_bytearray_extend(ba, &data);
                }
                Ok(ba)
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__imul__",
        make_builtin_function_with_arity("__imul__", bytearray_method_imul, 2),
    );
    // The transform methods read via `bytes_like_data` and build their
    // result with `new_bytes_like`, which yields a bytearray for a
    // bytearray receiver, so they share the bytes implementations.
    dict_storage_store(
        ns,
        "translate",
        make_builtin_function("translate", bytes_method_translate),
    );
    dict_storage_store(
        ns,
        "upper",
        make_builtin_function("upper", bytes_method_upper),
    );
    dict_storage_store(
        ns,
        "lower",
        make_builtin_function("lower", bytes_method_lower),
    );
    dict_storage_store(
        ns,
        "strip",
        make_builtin_function("strip", bytes_method_strip),
    );
    dict_storage_store(
        ns,
        "lstrip",
        make_builtin_function("lstrip", bytes_method_lstrip),
    );
    dict_storage_store(
        ns,
        "rstrip",
        make_builtin_function("rstrip", bytes_method_rstrip),
    );
    dict_storage_store(
        ns,
        "replace",
        make_builtin_function("replace", bytes_method_replace),
    );
    dict_storage_store(
        ns,
        "split",
        make_builtin_function("split", bytes_method_split),
    );
    dict_storage_store(
        ns,
        "rsplit",
        make_builtin_function("rsplit", bytes_method_rsplit),
    );
    dict_storage_store(
        ns,
        "splitlines",
        make_builtin_function("splitlines", bytes_method_splitlines),
    );
    dict_storage_store(ns, "join", make_builtin_function("join", bytes_method_join));
    dict_storage_store(
        ns,
        "partition",
        make_builtin_function("partition", bytes_method_partition),
    );
    dict_storage_store(
        ns,
        "rpartition",
        make_builtin_function("rpartition", bytes_method_rpartition),
    );
    dict_storage_store(
        ns,
        "title",
        make_builtin_function("title", bytes_method_title),
    );
    dict_storage_store(
        ns,
        "capitalize",
        make_builtin_function("capitalize", bytes_method_capitalize),
    );
    dict_storage_store(
        ns,
        "swapcase",
        make_builtin_function("swapcase", bytes_method_swapcase),
    );
    dict_storage_store(
        ns,
        "removeprefix",
        make_builtin_function("removeprefix", bytes_method_removeprefix),
    );
    dict_storage_store(
        ns,
        "removesuffix",
        make_builtin_function("removesuffix", bytes_method_removesuffix),
    );
    dict_storage_store(
        ns,
        "ljust",
        make_builtin_function("ljust", bytes_method_ljust),
    );
    dict_storage_store(
        ns,
        "rjust",
        make_builtin_function("rjust", bytes_method_rjust),
    );
    dict_storage_store(
        ns,
        "center",
        make_builtin_function("center", bytes_method_center),
    );
    dict_storage_store(
        ns,
        "zfill",
        make_builtin_function("zfill", bytes_method_zfill),
    );
    dict_storage_store(
        ns,
        "expandtabs",
        make_builtin_function("expandtabs", bytes_method_expandtabs),
    );
    dict_storage_store(ns, "hex", make_builtin_function("hex", bytes_method_hex));
    dict_storage_store(ns, "maketrans", make_maketrans_descr(bytes_maketrans));
    dict_storage_store(
        ns,
        "fromhex",
        pyre_object::function::w_classmethod_new(make_builtin_function(
            "fromhex",
            bytearray_fromhex,
        )),
    );
    // In-place mutators specific to the mutable bytearray.
    dict_storage_store(
        ns,
        "append",
        make_builtin_function("append", bytearray_method_append),
    );
    dict_storage_store(
        ns,
        "extend",
        make_builtin_function("extend", bytearray_method_extend),
    );
    dict_storage_store(
        ns,
        "insert",
        make_builtin_function("insert", bytearray_method_insert),
    );
    dict_storage_store(
        ns,
        "remove",
        make_builtin_function("remove", bytearray_method_remove),
    );
    dict_storage_store(
        ns,
        "pop",
        make_builtin_function("pop", bytearray_method_pop),
    );
    dict_storage_store(
        ns,
        "reverse",
        make_builtin_function("reverse", bytearray_method_reverse),
    );
    dict_storage_store(
        ns,
        "clear",
        make_builtin_function("clear", bytearray_method_clear),
    );
    dict_storage_store(
        ns,
        "copy",
        make_builtin_function("copy", bytearray_method_copy),
    );
    // Subscript slots exposed as callable dunders.  Each binds the direct
    // slot body so a subclass override's `super().__getitem__` reaches the
    // inherited builtin subscript instead of re-entering override dispatch.
    dict_storage_store(
        ns,
        "__getitem__",
        make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                crate::baseobjspace::getitem_slot(args[0], args[1])
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__setitem__",
        make_builtin_function_with_arity(
            "__setitem__",
            |args| {
                crate::type_methods::arity_exact_unpack(args, "__setitem__", 2)?;
                crate::baseobjspace::setitem_slot(args[0], args[1], args[2])?;
                Ok(pyre_object::w_none())
            },
            3,
        ),
    );
    dict_storage_store(
        ns,
        "__delitem__",
        make_builtin_function_with_arity(
            "__delitem__",
            |args| {
                crate::type_methods::arity_slot(args, 1)?;
                crate::baseobjspace::delitem_slot(args[0], args[1])?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    for (name, func) in [
        ("__eq__", bytearray_dunder_eq as DunderFn),
        ("__ne__", bytearray_dunder_ne),
        ("__lt__", bytearray_dunder_lt),
        ("__le__", bytearray_dunder_le),
        ("__gt__", bytearray_dunder_gt),
        ("__ge__", bytearray_dunder_ge),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 2));
    }
}

// ── set / frozenset TypeDef ──────────────────────────────────────────
// PyPy: pypy/objspace/std/setobject.py W_BaseSetObject.typedef
// pyre splits the shared methods through `init_setlike_common` so the
// frozenset typedef can omit the in-place mutators.

fn init_setlike_common(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "__contains__",
        make_builtin_function_with_arity(
            "__contains__",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(false));
                }
                unsafe {
                    if pyre_object::is_set_or_frozenset(args[0]) {
                        return Ok(pyre_object::w_bool_from(pyre_object::w_set_contains(
                            args[0], args[1],
                        )));
                    }
                }
                Ok(pyre_object::w_bool_from(false))
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__len__",
        make_builtin_function_with_arity(
            "__len__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                unsafe {
                    if pyre_object::is_set_or_frozenset(args[0]) {
                        return Ok(pyre_object::w_int_new(
                            pyre_object::w_set_len(args[0]) as i64
                        ));
                    }
                }
                Ok(pyre_object::w_int_new(0))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function_with_arity(
            "__iter__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                crate::baseobjspace::iter(args[0])
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__bool__",
        make_builtin_function_with_arity(
            "__bool__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_bool_from(false));
                }
                unsafe {
                    if pyre_object::is_set_or_frozenset(args[0]) {
                        return Ok(pyre_object::w_bool_from(
                            pyre_object::w_set_len(args[0]) > 0,
                        ));
                    }
                }
                Ok(pyre_object::w_bool_from(true))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__reduce__",
        make_builtin_function_with_arity(
            "__reduce__",
            |args| crate::reduce_protocol::set_reduce(args[0]),
            1,
        ),
    );
    dict_storage_store(
        ns,
        "__or__",
        make_builtin_function_with_arity("__or__", set_op_or, 2),
    );
    dict_storage_store(
        ns,
        "__and__",
        make_builtin_function_with_arity("__and__", set_op_and, 2),
    );
    dict_storage_store(
        ns,
        "__sub__",
        make_builtin_function_with_arity("__sub__", set_op_sub, 2),
    );
    dict_storage_store(
        ns,
        "__xor__",
        make_builtin_function_with_arity("__xor__", set_op_xor, 2),
    );
    dict_storage_store(
        ns,
        "__eq__",
        make_builtin_function_with_arity("__eq__", set_method_eq, 2),
    );
    dict_storage_store(
        ns,
        "__le__",
        make_builtin_function_with_arity("__le__", set_method_le, 2),
    );
    dict_storage_store(
        ns,
        "__ge__",
        make_builtin_function_with_arity("__ge__", set_method_ge, 2),
    );
    dict_storage_store(
        ns,
        "__lt__",
        make_builtin_function_with_arity(
            "__lt__",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let le = unsafe { pyre_object::w_bool_get_value(set_method_le(args)?) };
                let eq = unsafe { pyre_object::w_bool_get_value(set_method_eq(args)?) };
                Ok(pyre_object::w_bool_from(le && !eq))
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "__gt__",
        make_builtin_function_with_arity(
            "__gt__",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let ge = unsafe { pyre_object::w_bool_get_value(set_method_ge(args)?) };
                let eq = unsafe { pyre_object::w_bool_get_value(set_method_eq(args)?) };
                Ok(pyre_object::w_bool_from(ge && !eq))
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "union",
        make_builtin_function("union", set_method_union),
    );
    dict_storage_store(
        ns,
        "intersection",
        make_builtin_function("intersection", set_method_intersection),
    );
    dict_storage_store(
        ns,
        "difference",
        make_builtin_function("difference", set_method_difference),
    );
    dict_storage_store(
        ns,
        "symmetric_difference",
        make_builtin_function_with_arity(
            "symmetric_difference",
            set_method_symmetric_difference,
            2,
        ),
    );
    dict_storage_store(
        ns,
        "issubset",
        make_builtin_function_with_arity("issubset", set_method_le, 2),
    );
    dict_storage_store(
        ns,
        "issuperset",
        make_builtin_function_with_arity("issuperset", set_method_ge, 2),
    );
    dict_storage_store(
        ns,
        "isdisjoint",
        make_builtin_function_with_arity(
            "isdisjoint",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_bool_from(true));
                }
                let other_items = crate::builtins::collect_iterable(args[1])?;
                unsafe {
                    for item in &other_items {
                        if pyre_object::w_set_contains(args[0], *item) {
                            return Ok(pyre_object::w_bool_from(false));
                        }
                    }
                }
                Ok(pyre_object::w_bool_from(true))
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "copy",
        make_builtin_function_with_arity(
            "copy",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_set_new());
                }
                let items = unsafe { pyre_object::w_set_items(args[0]) };
                unsafe {
                    if pyre_object::is_frozenset(args[0]) {
                        return Ok(pyre_object::w_frozenset_from_items(&items));
                    }
                }
                Ok(pyre_object::w_set_from_items(&items))
            },
            1,
        ),
    );
}

// The `|` / `&` / `-` / `^` operator slots (`nb_or` etc.) require the
// other operand to be a set/frozenset and return NotImplemented otherwise
// — unlike the `union` / `intersection` / … methods, which accept any
// iterable.  `setobject.py descr_or`/`descr_and`/`descr_sub`/`descr_xor`.
fn set_op_requires_set(args: &[pyre_object::PyObjectRef]) -> bool {
    args.len() >= 2 && !unsafe { pyre_object::is_set_or_frozenset(args[1]) }
}
fn set_op_or(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if set_op_requires_set(args) {
        return Ok(pyre_object::w_not_implemented());
    }
    set_method_union(args)
}
fn set_op_and(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if set_op_requires_set(args) {
        return Ok(pyre_object::w_not_implemented());
    }
    set_method_intersection(args)
}
fn set_op_sub(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if set_op_requires_set(args) {
        return Ok(pyre_object::w_not_implemented());
    }
    set_method_difference(args)
}
fn set_op_xor(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if set_op_requires_set(args) {
        return Ok(pyre_object::w_not_implemented());
    }
    set_method_symmetric_difference(args)
}

fn set_method_union(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_set_new());
    }
    let mut items = unsafe { pyre_object::w_set_items(args[0]) };
    for other in &args[1..] {
        let other_items = crate::builtins::collect_iterable(*other)?;
        for item in other_items {
            items.push(item);
        }
    }
    unsafe {
        if pyre_object::is_frozenset(args[0]) {
            return Ok(pyre_object::w_frozenset_from_items(&items));
        }
    }
    Ok(pyre_object::w_set_from_items(&items))
}

fn set_method_intersection(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_set_new());
    }
    let self_items = unsafe { pyre_object::w_set_items(args[0]) };
    let mut result: Vec<pyre_object::PyObjectRef> = self_items;
    for other in &args[1..] {
        let other_items = crate::builtins::collect_iterable(*other)?;
        result.retain(|&item| unsafe {
            other_items
                .iter()
                .any(|&o| pyre_object::w_set_contains(pyre_object::w_set_from_items(&[o]), item))
        });
    }
    unsafe {
        if pyre_object::is_frozenset(args[0]) {
            return Ok(pyre_object::w_frozenset_from_items(&result));
        }
    }
    Ok(pyre_object::w_set_from_items(&result))
}

fn set_method_difference(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_set_new());
    }
    let mut items = unsafe { pyre_object::w_set_items(args[0]) };
    for other in &args[1..] {
        let other_items = crate::builtins::collect_iterable(*other)?;
        let probe = pyre_object::w_set_from_items(&other_items);
        items.retain(|&item| !unsafe { pyre_object::w_set_contains(probe, item) });
    }
    unsafe {
        if pyre_object::is_frozenset(args[0]) {
            return Ok(pyre_object::w_frozenset_from_items(&items));
        }
    }
    Ok(pyre_object::w_set_from_items(&items))
}

fn set_method_symmetric_difference(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        if args.is_empty() {
            return Ok(pyre_object::w_set_new());
        }
        return Ok(args[0]);
    }
    let self_items = unsafe { pyre_object::w_set_items(args[0]) };
    let other_items = crate::builtins::collect_iterable(args[1])?;
    let other_probe = pyre_object::w_set_from_items(&other_items);
    let self_probe = pyre_object::w_set_from_items(&self_items);
    let mut result: Vec<pyre_object::PyObjectRef> = self_items
        .iter()
        .copied()
        .filter(|&item| !unsafe { pyre_object::w_set_contains(other_probe, item) })
        .collect();
    for item in other_items {
        if !unsafe { pyre_object::w_set_contains(self_probe, item) } {
            result.push(item);
        }
    }
    unsafe {
        if pyre_object::is_frozenset(args[0]) {
            return Ok(pyre_object::w_frozenset_from_items(&result));
        }
    }
    Ok(pyre_object::w_set_from_items(&result))
}

fn set_method_eq(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_bool_from(false));
    }
    unsafe {
        if !pyre_object::is_set_or_frozenset(args[1]) {
            return Ok(pyre_object::w_bool_from(false));
        }
        if pyre_object::w_set_len(args[0]) != pyre_object::w_set_len(args[1]) {
            return Ok(pyre_object::w_bool_from(false));
        }
        for item in pyre_object::w_set_items(args[0]) {
            if !pyre_object::w_set_contains(args[1], item) {
                return Ok(pyre_object::w_bool_from(false));
            }
        }
    }
    Ok(pyre_object::w_bool_from(true))
}

fn set_method_le(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_bool_from(true));
    }
    let other_items = crate::builtins::collect_iterable(args[1])?;
    let probe = pyre_object::w_set_from_items(&other_items);
    unsafe {
        for item in pyre_object::w_set_items(args[0]) {
            if !pyre_object::w_set_contains(probe, item) {
                return Ok(pyre_object::w_bool_from(false));
            }
        }
    }
    Ok(pyre_object::w_bool_from(true))
}

fn set_method_ge(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_bool_from(true));
    }
    let other_items = crate::builtins::collect_iterable(args[1])?;
    unsafe {
        for item in other_items {
            if !pyre_object::w_set_contains(args[0], item) {
                return Ok(pyre_object::w_bool_from(false));
            }
        }
    }
    Ok(pyre_object::w_bool_from(true))
}

fn init_set_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(set_descr_new));
    // setobject.py:528 __class_getitem__ = gateway.interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    dict_storage_store(
        ns,
        "__class_getitem__",
        pyre_object::function::w_classmethod_new(make_builtin_function(
            "__class_getitem__",
            crate::_pypy_generic_alias::generic_alias_class_getitem,
        )),
    );
    dict_storage_store(
        ns,
        "__init__",
        make_builtin_function("__init__", set_descr_init),
    );
    init_setlike_common(ns);
    dict_storage_store(
        ns,
        "add",
        make_builtin_function_with_arity(
            "add",
            |args| {
                if args.len() >= 2 {
                    unsafe { pyre_object::w_set_add(args[0], args[1]) };
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "discard",
        make_builtin_function_with_arity(
            "discard",
            |args| {
                if args.len() >= 2 {
                    unsafe { pyre_object::w_set_discard(args[0], args[1]) };
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "remove",
        make_builtin_function_with_arity(
            "remove",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("remove() requires an argument"));
                }
                let removed = unsafe { pyre_object::w_set_discard(args[0], args[1]) };
                if !removed {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::KeyError,
                        "set.remove(x): x not in set",
                    ));
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    dict_storage_store(
        ns,
        "pop",
        make_builtin_function_with_arity(
            "pop",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::KeyError,
                        "pop from an empty set",
                    ));
                }
                let items = unsafe { pyre_object::w_set_items(args[0]) };
                if let Some(&item) = items.first() {
                    unsafe { pyre_object::w_set_discard(args[0], item) };
                    return Ok(item);
                }
                Err(crate::PyError::new(
                    crate::PyErrorKind::KeyError,
                    "pop from an empty set",
                ))
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "clear",
        make_builtin_function_with_arity(
            "clear",
            |args| {
                if !args.is_empty() {
                    let items = unsafe { pyre_object::w_set_items(args[0]) };
                    for item in items {
                        unsafe { pyre_object::w_set_discard(args[0], item) };
                    }
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    dict_storage_store(
        ns,
        "update",
        make_builtin_function("update", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            for other in &args[1..] {
                let other_items = crate::builtins::collect_iterable(*other)?;
                for item in other_items {
                    unsafe { pyre_object::w_set_add(args[0], item) };
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    // `pypy/objspace/std/setobject.py:1188 W_BaseSetObject.descr_difference_update`
    // / `:1217 descr_intersection_update` / `:1244 descr_symmetric_difference_update`
    // — in-place set ops that mirror the non-update variants but
    // mutate `self` instead of returning a fresh set.
    dict_storage_store(
        ns,
        "difference_update",
        make_builtin_function("difference_update", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            for other in &args[1..] {
                let other_items = crate::builtins::collect_iterable(*other)?;
                for item in other_items {
                    unsafe { pyre_object::w_set_discard(args[0], item) };
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    dict_storage_store(
        ns,
        "intersection_update",
        make_builtin_function("intersection_update", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            // Snapshot self's items, drop any not present in EVERY other.
            let self_items = unsafe { pyre_object::w_set_items(args[0]) };
            for item in self_items {
                let mut keep = true;
                for other in &args[1..] {
                    let other_items = crate::builtins::collect_iterable(*other)?;
                    let mut found = false;
                    for &o in other_items.iter() {
                        if crate::baseobjspace::eq_w(item, o)? {
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        keep = false;
                        break;
                    }
                }
                if !keep {
                    unsafe { pyre_object::w_set_discard(args[0], item) };
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    dict_storage_store(
        ns,
        "symmetric_difference_update",
        make_builtin_function("symmetric_difference_update", |args| {
            if args.is_empty() || args.len() < 2 {
                return Ok(pyre_object::w_none());
            }
            let other_items = crate::builtins::collect_iterable(args[1])?;
            for item in other_items {
                // toggle: remove if present, add otherwise
                let self_items = unsafe { pyre_object::w_set_items(args[0]) };
                let mut present = false;
                for &existing in self_items.iter() {
                    if crate::baseobjspace::eq_w(item, existing)? {
                        present = true;
                        break;
                    }
                }
                if present {
                    unsafe { pyre_object::w_set_discard(args[0], item) };
                } else {
                    unsafe { pyre_object::w_set_add(args[0], item) };
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
}

fn init_frozenset_type(ns: &mut DictStorage) {
    dict_storage_store(ns, "__new__", make_new_descr(frozenset_descr_new));
    // setobject.py:661 __class_getitem__ = gateway.interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    dict_storage_store(
        ns,
        "__class_getitem__",
        pyre_object::function::w_classmethod_new(make_builtin_function(
            "__class_getitem__",
            crate::_pypy_generic_alias::generic_alias_class_getitem,
        )),
    );
    init_setlike_common(ns);
}

// ── __dict__ / __weakref__ descriptors ───────────────────────────────

/// typedef.py:561-563 dict_descr.
///
/// ```python
/// dict_descr = GetSetProperty(descr_get_dict, descr_set_dict, descr_del_dict,
///                             doc="dictionary for instance variables (if defined)")
/// dict_descr.name = '__dict__'
/// ```
pub fn dict_descr() -> pyre_object::PyObjectRef {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    let addr = *CACHED.get_or_init(|| {
        let fget = make_builtin_function_with_arity("descr_get_dict", descr_get_dict, 2);
        let fset = make_builtin_function_with_arity("descr_set_dict", descr_set_dict, 3);
        let fdel = make_builtin_function_with_arity("descr_del_dict", descr_del_dict, 2);
        // typedef.py:563 `dict_descr.name = '__dict__'` — pass the
        // explicit name through the constructor so descriptor
        // introspection (`type.__dict__['__dict__'].__name__`) returns
        // `"__dict__"` instead of the `"<generic property>"` sentinel.
        // The earlier setattr fix-up was masked by the new read-only
        // `__name__` getset and silently failed.
        make_getset_property_named(fget, fset, fdel, "__dict__") as usize
    });
    addr as pyre_object::PyObjectRef
}

/// typedef.py:593-595 weakref_descr.
///
/// ```python
/// weakref_descr = GetSetProperty(descr_get_weakref,
///                     doc="list of weak references to the object (if defined)")
/// weakref_descr.name = '__weakref__'
/// ```
pub fn weakref_descr() -> pyre_object::PyObjectRef {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    let addr = *CACHED.get_or_init(|| {
        let fget = make_builtin_function_with_arity("descr_get_weakref", descr_get_weakref, 2);
        // typedef.py:591 `weakref_descr.name = '__weakref__'` —
        // see `dict_descr` for the parity rationale.
        make_getset_descriptor_named(fget, "__weakref__") as usize
    });
    addr as pyre_object::PyObjectRef
}

/// PyPy stores `fget/fset/fdel/doc/reqcls/use_closure/name` directly on
/// the `GetSetProperty` instance fields. pyre's instance dict (mapdict)
/// is thread-local, but `init_typeobjects` runs once globally and the
/// `pypy/interpreter/typedef.py:327-336 GetSetProperty._init` —
/// stores fget/fset/fdel/doc/reqcls/use_closure/name directly on the
/// descriptor instance.  Pyre matches that shape with a real W_Root
/// struct (`pyre_object::typedef::GetSetProperty`); these
/// helpers are thin wrappers over the typed accessors so existing
/// call sites stay readable.
///
/// `cls` is stored as `reqcls` exactly like PyPy. `use_closure` is
/// unused at runtime (pyre has no closure-passing distinction) but
/// still kept on the struct for parity.
fn getset_property_init(
    new: pyre_object::PyObjectRef,
    fget: pyre_object::PyObjectRef,
    fset: pyre_object::PyObjectRef,
    fdel: pyre_object::PyObjectRef,
    doc: pyre_object::PyObjectRef,
    cls: pyre_object::PyObjectRef,
    use_closure: bool,
    name: pyre_object::PyObjectRef,
) {
    // The descriptor struct is allocated by `make_getset_property_full`
    // already filled in (typedef.py:327-336 hands the fully-formed
    // instance back to the caller); this helper survives only as the
    // copy-for-type path that re-stamps an existing GetSetProperty
    // with new bindings.
    let _ = use_closure; // mirrored in the struct but unused here
    let resolved_name = if !name.is_null() && unsafe { pyre_object::is_str(name) } {
        name
    } else {
        pyre_object::w_str_new("<generic property>")
    };
    unsafe {
        let descr = &mut *(new as *mut pyre_object::typedef::GetSetProperty);
        descr.fget = fget;
        descr.fset = fset;
        descr.fdel = fdel;
        descr.doc = doc;
        descr.reqcls = cls;
        descr.name = resolved_name;
        descr.use_closure = use_closure;
    }
}

/// Read the optional `reqcls` field from a getset descriptor.
/// Returns null if no required class is set.
fn read_reqcls(descr: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    if descr.is_null() {
        return pyre_object::PY_NULL;
    }
    let value = unsafe { pyre_object::typedef::w_getset_get_reqcls(descr) };
    if value.is_null() || unsafe { pyre_object::is_none(value) } {
        pyre_object::PY_NULL
    } else {
        value
    }
}

fn read_fget(descr: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    if descr.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { pyre_object::typedef::w_getset_get_fget(descr) }
}

fn read_fset(descr: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    if descr.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { pyre_object::typedef::w_getset_get_fset(descr) }
}

fn read_fdel(descr: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    if descr.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { pyre_object::typedef::w_getset_get_fdel(descr) }
}

fn read_descr_name(descr: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    if descr.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { pyre_object::typedef::w_getset_get_name(descr) }
}

/// typedef.py:337-345 GetSetProperty.copy_for_type.
///
/// ```python
/// def copy_for_type(self, w_objclass):
///     if self.reqcls is None:
///         new = instantiate(GetSetProperty)
///         new._init(self.fget, self.fset, self.fdel, self.doc, self.reqcls,
///                   self.use_closure, self.name)
///         new.w_objclass = w_objclass
///         return new
///     else:
///         return self
/// ```
fn copy_for_type(
    descr: pyre_object::PyObjectRef,
    w_objclass: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    // typedef.py:338 if self.reqcls is None:
    let reqcls = read_reqcls(descr);
    if !reqcls.is_null() {
        // typedef.py:344 return self
        return descr;
    }
    if !unsafe { pyre_object::typedef::is_getset_property(descr) } {
        return descr;
    }
    // typedef.py:350-352 — allocate a fresh GetSetProperty and copy
    // every slot from the source descriptor (reqcls passes through as
    // None per the source's `if self.reqcls is None` precondition).
    let _ = getset_descriptor_type(); // ensure type registered
    let src = unsafe { &*(descr as *const pyre_object::typedef::GetSetProperty) };
    let new = pyre_object::typedef::w_getset_property_new(
        src.fget,
        src.fset,
        src.fdel,
        src.doc,
        pyre_object::PY_NULL,
        src.use_closure,
        src.name,
    );
    // typedef.py:353 new.w_objclass = w_objclass — write directly to
    // the typed slot, mirroring PyPy's instance-field assignment.
    unsafe { pyre_object::typedef::w_getset_set_objclass(new, w_objclass) };
    new
}

/// Public re-export of `copy_for_type` so that
/// `objspace/std/typeobject.py::create_dict_slot`'s pyre equivalent in
/// `call.rs` can call `copy_for_type(dict_descr(), w_self)` directly,
/// matching PyPy's `dict_descr.copy_for_type(w_self)` shape.
pub fn copy_descriptor_for_type(
    descr: pyre_object::PyObjectRef,
    w_objclass: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    copy_for_type(descr, w_objclass)
}

/// typedef.py:541-547 descr_get_dict.
///
/// ```python
/// def descr_get_dict(space, w_obj):
///     w_dict = w_obj.getdict(space)
///     if w_dict is None:
///         raise oefmt(space.w_TypeError,
///                     "descriptor '__dict__' doesn't apply to '%T' objects",
///                     w_obj)
///     return w_dict
/// ```
///
/// In pyre the typecheck wrapper passes (closure, w_obj) — args[0] is
/// the descriptor `self` and args[1] is w_obj. There is no `space`
/// parameter (pyre has no space first-class object).
fn descr_get_dict(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let _closure = args[0];
    let w_obj = args[1];
    let w_dict = crate::baseobjspace::getdict(w_obj);
    if w_dict.is_null() {
        let tp_name = unsafe { (*(*w_obj).ob_type).name };
        return Err(crate::PyError::type_error(format!(
            "descriptor '__dict__' doesn't apply to '{}' objects",
            tp_name,
        )));
    }
    Ok(w_dict)
}

/// typedef.py:549-550 descr_set_dict.
///
/// ```python
/// def descr_set_dict(space, w_obj, w_dict):
///     w_obj.setdict(space, w_dict)
/// ```
fn descr_set_dict(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let _closure = args[0];
    let w_obj = args[1];
    let w_dict = args[2];
    crate::baseobjspace::setdict(w_obj, w_dict)?;
    Ok(pyre_object::w_none())
}

/// typedef.py:552-553 descr_del_dict.
///
/// ```python
/// def descr_del_dict(space, w_obj): # blame CPython for the existence of this one
///     w_obj.setdict(space, space.newdict())
/// ```
fn descr_del_dict(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let _closure = args[0];
    let w_obj = args[1];
    crate::baseobjspace::setdict(w_obj, pyre_object::w_dict_new())?;
    Ok(pyre_object::w_none())
}

/// typedef.py:555-559 descr_get_weakref.
///
/// ```python
/// def descr_get_weakref(space, w_obj):
///     lifeline = w_obj.getweakref()
///     if lifeline is None:
///         return space.w_None
///     return lifeline.get_any_weakref(space)
/// ```
fn descr_get_weakref(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let _closure = args[0];
    let w_obj = args[1];
    let lifeline = crate::baseobjspace::getweakref(w_obj);
    match lifeline {
        None => Ok(pyre_object::w_none()),
        Some(lifeline) => Ok(crate::module::_weakref::interp__weakref::get_any_weakref(
            lifeline,
        )),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_ellipsis_has_registered_typeobject() {
        crate::typedef::init_typeobjects();
        let w_type = crate::typedef::r#type(pyre_object::special::w_ellipsis())
            .expect("Ellipsis should resolve to a W_TypeObject");
        unsafe {
            assert_eq!(pyre_object::w_type_get_name(w_type), "ellipsis");
            assert!(!pyre_object::w_type_get_acceptable_as_base_class(w_type));
        }
    }
}
