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

use crate::{make_builtin_function, make_builtin_function_with_arity};

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

/// `pypy/interpreter/typedef.py:138-143 default_identity_hash`.
///
/// Exact immutable builtin values use their value-derived unique id; ordinary
/// objects use RPython's `compute_identity_hash`, whose translated minimark
/// implementation is `mangle_hash(id_or_identityhash(obj))`.
pub fn default_identity_hash_value(w_obj: PyObjectRef) -> i64 {
    if let Some(w_unique_id) = crate::function::immutable_unique_id(w_obj) {
        return crate::builtins::hash_value(w_unique_id);
    }
    let identity = pyre_object::gc_hook::gc_identity_hash(w_obj as usize) as i64;
    identity ^ (identity >> 4)
}

pub fn default_identity_hash(_space: PyObjectRef, w_obj: PyObjectRef) -> PyObjectRef {
    pyre_object::w_int_new(default_identity_hash_value(w_obj))
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

/// PyPy `pypy/interpreter/typedef.py:598-599 make_weakref_descr` returns
/// the canonical `weakref_descr` installed in a TypeDef. The class argument
/// only participates in annotation in RPython; it is not the descriptor
/// value itself.
pub fn make_weakref_descr(_cls: PyObjectRef) -> PyObjectRef {
    weakref_descr()
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
    // Interpreter-only test path: libtest runs each `#[test]` on a fresh
    // thread and `dict_eq_hook`'s hash hook is thread-local, so install it
    // here — the single type-system entry every dict-building test funnels
    // through — before the builtin type namespaces (GC dicts) are built or
    // probed.  Production installs the hook at boot (`pyre-jit::eval::
    // init_jit_hooks`); this call is compiled out of every non-test build
    // (gated on `cfg(test)` here, or the `test-hooks` feature that
    // downstream test builds such as `pyre-jit` enable via a dev-dependency).
    #[cfg(any(test, feature = "test-hooks"))]
    crate::test_hooks::install_hash_hook();
    TYPEOBJECT_CACHE.get_or_init(|| {
        // Seed `subclassrange_{min,max}` on every registered PyType so
        // `ll_isinstance` works on the interpreter-only test path that
        // skips the JIT init. This uses the same registration-ordered
        // reversed-MRO peer census as GC `assign_inheritance_ids`, so JIT
        // init's later `gc.subclass_range` writeback is byte-identical.
        // Calling
        // `mark_subclass_ranges_initialized` afterwards stops the
        // pyre-object-internal `is_exception` fallback from
        // omitting the cross-crate `CODE_TYPE` / `PYTRACEBACK_TYPE`
        // aliases from a later redundant write.
        let object_aliases = pyre_object::pyobject::all_subclass_range_aliases();
        let interpreter_aliases = crate::all_subclass_range_aliases();
        pyre_object::pyobject::compute_subclass_ranges_from(&[
            &object_aliases,
            &interpreter_aliases,
        ]);
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
        let module_type = new_typeobject_with_base("module", init_module_type, object_type);
        unsafe {
            // module.py Module.getdict plus Module.typedef.__weakref__.
            pyre_object::w_type_set_hasdict(module_type, true);
            pyre_object::w_type_set_weakrefable(module_type, true);
        }
        reg.insert(
            &pyre_object::MODULE_TYPE as *const PyType as usize,
            module_type as usize,
        );
        // `pypy/objspace/std/dictmultiobject.py` dict_keys / dict_values /
        // dict_items TypeDefs, with the Python 3.14 type flags.  PyPy's
        // current TypeDefs expose `new_dict_*` and permit subclassing for its
        // app-level OrderedDict views.  Python 3.14 instead gives all three
        // types a null `tp_new` and clears BASETYPE; pyre targets that newer
        // public contract.  Its OrderedDict views therefore use the 3.14
        // `_collections_abc` bases in `app_odict.py`.
        // dict_keys / dict_items get the SetLikeDictView surface
        // per dictmultiobject.py:1802-1829 / 1773-1800; dict_values
        // stops at the common slots per dictmultiobject.py:1831-1840
        // (values views are intentionally NOT set-like).
        let dict_keys_type =
            new_typeobject_with_base("dict_keys", init_dict_view_keys_type, object_type);
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(dict_keys_type, false);
            pyre_object::w_type_set_disallow_instantiation(dict_keys_type);
        }
        reg.insert(
            &pyre_object::dictmultiobject::DICT_KEYS_TYPE as *const PyType as usize,
            dict_keys_type as usize,
        );
        let dict_values_type =
            new_typeobject_with_base("dict_values", init_dict_view_values_type, object_type);
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(dict_values_type, false);
            pyre_object::w_type_set_disallow_instantiation(dict_values_type);
        }
        reg.insert(
            &pyre_object::dictmultiobject::DICT_VALUES_TYPE as *const PyType as usize,
            dict_values_type as usize,
        );
        let dict_items_type =
            new_typeobject_with_base("dict_items", init_dict_view_items_type, object_type);
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(dict_items_type, false);
            pyre_object::w_type_set_disallow_instantiation(dict_items_type);
        }
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

        // builtin_function — PyPy: typedef.py BuiltinFunction.typedef. The
        // externally visible name follows CPython 3.14.
        let builtin_function_type = new_typeobject_with_base(
            "builtin_function_or_method",
            init_builtin_function_type,
            object_type,
        );
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(builtin_function_type, false);
            pyre_object::w_type_set_disallow_instantiation(builtin_function_type);
        }
        unsafe {
            // CPython 3.14 PyCFunction objects are weakrefable but have no
            // instance dict. PyPy's copied Function rawdict differs here.
            pyre_object::w_type_set_hasdict(builtin_function_type, false);
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

        // NoneType — PyPy noneobject.py, with the Python 3.14 rich-comparison
        // slots exposed by CPython's singleton type.
        let none_type = new_typeobject_with_base("NoneType", init_none_type, object_type);
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(none_type, false) };
        reg.insert(&NONE_TYPE as *const PyType as usize, none_type as usize);

        // setobject.py W_SetIterObject.typedef. Python 3.14 exposes the
        // concrete name as `set_iterator` (PyPy 3.11 used `setiterator`).
        reg.insert(
            &pyre_object::setobject::SET_ITERATOR_TYPE as *const PyType as usize,
            new_typeobject_with_base("set_iterator", init_set_iterator_type, object_type) as usize,
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
        let set_type = new_typeobject_with_base_and_layout(
            "set",
            init_set_type,
            object_type,
            &pyre_object::setobject::SET_TYPE as *const PyType,
        );
        // W_BaseSetObject.getweakref/setweakref/delweakref: both concrete
        // set layouts carry a weakref lifeline.
        unsafe { pyre_object::w_type_set_weakrefable(set_type, true) };
        reg.insert(
            &pyre_object::setobject::SET_TYPE as *const PyType as usize,
            set_type as usize,
        );
        let frozenset_type = new_typeobject_with_base_and_layout(
            "frozenset",
            init_frozenset_type,
            object_type,
            &pyre_object::setobject::FROZENSET_TYPE as *const PyType,
        );
        unsafe { pyre_object::w_type_set_weakrefable(frozenset_type, true) };
        reg.insert(
            &pyre_object::setobject::FROZENSET_TYPE as *const PyType as usize,
            frozenset_type as usize,
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
            new_typeobject_with_base("super", init_super_type, object_type) as usize,
        );
        let generator_type =
            new_typeobject_with_base("generator", init_generator_type, object_type);
        // `Py_TPFLAGS_DISALLOW_INSTANTIATION` — a generator is produced
        // only by calling a generator function, never by `generator()`,
        // so `tp_new` is NULL and pickling refuses it.
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(generator_type);
            pyre_object::w_type_set_acceptable_as_base_class(generator_type, false);
            pyre_object::w_type_set_weakrefable(generator_type, true);
        }
        reg.insert(
            &pyre_object::generator::GENERATOR_TYPE as *const PyType as usize,
            generator_type as usize,
        );
        let coroutine_type =
            new_typeobject_with_base("coroutine", init_coroutine_type, object_type);
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(coroutine_type);
            pyre_object::w_type_set_acceptable_as_base_class(coroutine_type, false);
            pyre_object::w_type_set_weakrefable(coroutine_type, true);
        }
        reg.insert(
            &pyre_object::generator::COROUTINE_TYPE as *const PyType as usize,
            coroutine_type as usize,
        );
        let coroutine_wrapper_type = new_typeobject_with_base(
            "coroutine_wrapper",
            init_coroutine_wrapper_type,
            object_type,
        );
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(coroutine_wrapper_type);
            pyre_object::w_type_set_acceptable_as_base_class(coroutine_wrapper_type, false);
        }
        reg.insert(
            &pyre_object::generator::COROUTINE_WRAPPER_TYPE as *const PyType as usize,
            coroutine_wrapper_type as usize,
        );
        let range_iterator_type =
            new_typeobject_with_base("range_iterator", init_range_iterator_type, object_type);
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(range_iterator_type);
            pyre_object::w_type_set_acceptable_as_base_class(range_iterator_type, false);
        }
        reg.insert(
            &pyre_object::functional::RANGE_ITER_TYPE as *const PyType as usize,
            range_iterator_type as usize,
        );
        // rangeobject.c PyRange_Type carries no Py_TPFLAGS_BASETYPE, so
        // `range` is not an acceptable base class.
        let range_type = new_typeobject_with_base("range", init_range_type, object_type);
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
        unsafe {
            pyre_object::w_type_set_acceptable_as_base_class(memoryview_type, false);
            pyre_object::w_type_set_weakrefable(memoryview_type, true);
        }
        reg.insert(
            &pyre_object::memoryview::MEMORYVIEW_TYPE as *const PyType as usize,
            memoryview_type as usize,
        );
        let seq_iterator_type =
            new_typeobject_with_base("iterator", init_sequence_iterator_type, object_type);
        // `Py_TPFLAGS_DISALLOW_INSTANTIATION` — an iterator is produced only by
        // `iter(obj)`, never by `iterator()`, so `tp_new` is NULL.
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(seq_iterator_type);
            pyre_object::w_type_set_acceptable_as_base_class(seq_iterator_type, false);
        }
        reg.insert(
            &pyre_object::iterobject::SEQ_ITER_TYPE as *const PyType as usize,
            seq_iterator_type as usize,
        );
        let callable_iterator_type = new_typeobject_with_base(
            "callable_iterator",
            init_callable_iterator_type,
            object_type,
        );
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(callable_iterator_type);
            pyre_object::w_type_set_acceptable_as_base_class(callable_iterator_type, false);
        }
        reg.insert(
            &pyre_object::operation::CALLABLE_ITERATOR_TYPE as *const PyType as usize,
            callable_iterator_type as usize,
        );
        for (pytype, name, init) in [
            (
                &pyre_object::iterobject::LIST_ITER_TYPE as *const PyType,
                "list_iterator",
                init_list_iterator_type as fn(PyObjectRef),
            ),
            (
                &pyre_object::iterobject::LIST_REVERSE_ITER_TYPE as *const PyType,
                "list_reverseiterator",
                init_list_reverse_iterator_type as fn(PyObjectRef),
            ),
            (
                &pyre_object::iterobject::TUPLE_ITER_TYPE as *const PyType,
                "tuple_iterator",
                init_tuple_iterator_type as fn(PyObjectRef),
            ),
        ] {
            let iterator_type = new_typeobject_with_base(name, init, object_type);
            unsafe {
                pyre_object::w_type_set_disallow_instantiation(iterator_type);
                pyre_object::w_type_set_acceptable_as_base_class(iterator_type, false);
            }
            reg.insert(pytype as usize, iterator_type as usize);
        }
        let long_range_iterator_type = new_typeobject_with_base(
            "longrange_iterator",
            init_long_range_iterator_type,
            object_type,
        );
        unsafe {
            pyre_object::w_type_set_disallow_instantiation(long_range_iterator_type);
            pyre_object::w_type_set_acceptable_as_base_class(long_range_iterator_type, false);
        }
        reg.insert(
            &pyre_object::functional::LONG_RANGE_ITER_TYPE as *const PyType as usize,
            long_range_iterator_type as usize,
        );
        reg.insert(
            &pyre_object::functional::ENUMERATE_TYPE as *const PyType as usize,
            new_typeobject_with_base("enumerate", init_enumerate_type, object_type) as usize,
        );
        reg.insert(
            &pyre_object::functional::REVERSED_TYPE as *const PyType as usize,
            new_typeobject_with_base("reversed", init_reversed_type, object_type) as usize,
        );
        reg.insert(
            &pyre_object::functional::FILTER_TYPE as *const PyType as usize,
            new_typeobject_with_base("filter", init_filter_type, object_type) as usize,
        );
        reg.insert(
            &pyre_object::functional::MAP_TYPE as *const PyType as usize,
            new_typeobject_with_base("map", init_map_type, object_type) as usize,
        );
        reg.insert(
            &pyre_object::functional::ZIP_TYPE as *const PyType as usize,
            new_typeobject_with_base("zip", init_zip_type, object_type) as usize,
        );
        for (pytype, name, init) in [
            (
                &pyre_object::dictmultiobject::DICT_KEYITERATOR_TYPE as *const PyType,
                "dict_keyiterator",
                init_dict_key_iterator_type as fn(PyObjectRef),
            ),
            (
                &pyre_object::dictmultiobject::DICT_VALUEITERATOR_TYPE as *const PyType,
                "dict_valueiterator",
                init_dict_value_iterator_type as fn(PyObjectRef),
            ),
            (
                &pyre_object::dictmultiobject::DICT_ITEMITERATOR_TYPE as *const PyType,
                "dict_itemiterator",
                init_dict_item_iterator_type as fn(PyObjectRef),
            ),
            (
                &pyre_object::dictmultiobject::DICT_REVERSEKEYITERATOR_TYPE as *const PyType,
                "dict_reversekeyiterator",
                init_dict_reverse_key_iterator_type as fn(PyObjectRef),
            ),
            (
                &pyre_object::dictmultiobject::DICT_REVERSEVALUEITERATOR_TYPE as *const PyType,
                "dict_reversevalueiterator",
                init_dict_reverse_value_iterator_type as fn(PyObjectRef),
            ),
            (
                &pyre_object::dictmultiobject::DICT_REVERSEITEMITERATOR_TYPE as *const PyType,
                "dict_reverseitemiterator",
                init_dict_reverse_item_iterator_type as fn(PyObjectRef),
            ),
        ] {
            let iterator_type = new_typeobject_with_base(name, init, object_type);
            unsafe {
                pyre_object::w_type_set_disallow_instantiation(iterator_type);
                pyre_object::w_type_set_acceptable_as_base_class(iterator_type, false);
            }
            reg.insert(pytype as usize, iterator_type as usize);
        }
        let cell_type = new_typeobject_with_base_and_layout(
            "cell",
            init_cell_type,
            object_type,
            &pyre_object::nestedscope::CELL_TYPE as *const PyType,
        );
        // typedef.py:953 `Cell.typedef.acceptable_as_base_class = False`;
        // CPython 3.14 likewise omits Py_TPFLAGS_BASETYPE.
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(cell_type, false) };
        reg.insert(
            &pyre_object::nestedscope::CELL_TYPE as *const PyType as usize,
            cell_type as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::COUNT_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "itertools.count",
                init_count_type,
                object_type,
                &pyre_object::interp_itertools::COUNT_TYPE as *const PyType,
            ) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::REPEAT_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "itertools.repeat",
                init_repeat_type,
                object_type,
                &pyre_object::interp_itertools::REPEAT_TYPE as *const PyType,
            ) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::TAKEWHILE_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "itertools.takewhile",
                init_takewhile_type,
                object_type,
                &pyre_object::interp_itertools::TAKEWHILE_TYPE as *const PyType,
            ) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::DROPWHILE_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "itertools.dropwhile",
                init_dropwhile_type,
                object_type,
                &pyre_object::interp_itertools::DROPWHILE_TYPE as *const PyType,
            ) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::FILTERFALSE_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "itertools.filterfalse",
                init_filterfalse_type,
                object_type,
                &pyre_object::interp_itertools::FILTERFALSE_TYPE as *const PyType,
            ) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::COMPRESS_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "itertools.compress",
                init_compress_type,
                object_type,
                &pyre_object::interp_itertools::COMPRESS_TYPE as *const PyType,
            ) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::STARMAP_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "itertools.starmap",
                init_starmap_type,
                object_type,
                &pyre_object::interp_itertools::STARMAP_TYPE as *const PyType,
            ) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::ACCUMULATE_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "itertools.accumulate",
                init_accumulate_type,
                object_type,
                &pyre_object::interp_itertools::ACCUMULATE_TYPE as *const PyType,
            ) as usize,
        );
        reg.insert(
            &pyre_object::interp_itertools::ZIP_LONGEST_TYPE as *const PyType as usize,
            new_typeobject_with_base_and_layout(
                "itertools.zip_longest",
                init_zip_longest_type,
                object_type,
                &pyre_object::interp_itertools::ZIP_LONGEST_TYPE as *const PyType,
            ) as usize,
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
            // arraymodule.c arraytype carries Py_TPFLAGS_SEQUENCE.
            (&pyre_object::interp_array::ARRAY_TYPE, b'S'),
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
        // baseobjspace.py getclass() — for type objects, the class
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

    // The `patch_*` passes install descriptors into the shared global type
    // dicts (e.g. `object.__class__`).  `get_or_init` above serializes only
    // the type construction: once it returns, every concurrent
    // `ExecutionContext::new` caller it was blocking falls through to here at
    // once, so an unguarded first-time `type_dict_store` would race a sibling
    // thread's `type_dict_contains` read on the same `IndexMap` and tear its
    // internal index table.  A dedicated `Once` collapses the patch pass to a
    // single writer; it runs after `TYPEOBJECT_CACHE` is populated so
    // `patch_typeobject_descriptor_names` still observes the registry.
    static PATCH_TYPEOBJECTS: std::sync::Once = std::sync::Once::new();
    PATCH_TYPEOBJECTS.call_once(|| {
        patch_object_class_descriptor();
        patch_builtin_function_descriptors();
        patch_function_member_descriptors();
        patch_module_descriptors();
        patch_frame_traceback_descriptors();
        patch_cell_descriptor();
        patch_getset_descriptor_metadata();
        patch_typeobject_descriptor_names();
    });
}

/// Install `object.__class__` after the root object type exists.
///
/// `GetSetProperty` itself inherits from object, so allocating this descriptor
/// inside `init_object_type` would recursively request an object base before
/// `W_OBJECT_TYPEOBJECT` has been published.  PyPy's module-level TypeDef
/// construction has the root available already; pyre mirrors that ordering
/// with this post-registration pass.
fn patch_object_class_descriptor() {
    let object_type = w_object();
    if object_type.is_null()
        || !crate::type_dict_has_storage(object_type)
        || crate::type_dict_contains(object_type, "__class__")
    {
        return;
    }
    let class_getter = make_builtin_function_with_arity(
        "__class__",
        |args| Ok(crate::typedef::r#type(args[1]).unwrap_or(pyre_object::PY_NULL)),
        2,
    );
    let class_setter = make_builtin_function_with_arity(
        "__class__",
        |args| crate::baseobjspace::descr_set___class__(args[1], args[2]),
        3,
    );
    crate::type_dict_store(
        object_type,
        "__class__",
        make_getset_property_full(
            class_getter,
            class_setter,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            object_type,
            Some("__class__"),
        ),
    );
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
        let ns = unsafe { pyre_object::w_type_get_dict_ptr(tp) } as PyObjectRef;
        if ns.is_null() {
            continue;
        }
        let entries: Vec<(String, PyObjectRef)> = unsafe { pyre_object::w_dict_items(ns) }
            .into_iter()
            .filter_map(|(key, value)| {
                unsafe { pyre_object::w_str_get_value_opt(key) }.map(|key| (key.to_owned(), value))
            })
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
/// `ns` must be a valid, live `W_DictObject`; `type_obj` a valid type.
unsafe fn stamp_new_descr_self(ns: PyObjectRef, type_obj: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let save_point = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(ns);
    pyre_object::gc_roots::pin_root(type_obj);

    if let Some(w_new) = pyre_object::w_dict_getitem_str(ns, "__new__") {
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
    let keys: Vec<String> = pyre_object::w_dict_items(ns)
        .into_iter()
        .filter_map(|(key, _)| pyre_object::w_str_get_value_opt(key).map(str::to_owned))
        .collect();
    for key in keys {
        let ns = pyre_object::gc_roots::shadow_stack_get(save_point);
        let type_obj = pyre_object::gc_roots::shadow_stack_get(save_point + 1);
        let Some(descr) = pyre_object::w_dict_getitem_str(ns, &key) else {
            continue;
        };
        // CPython member descriptors carry their defining type in d_type;
        // PyPy's Member receives w_cls while the TypeDef is materialised.
        // Builtin TypeDef initializers necessarily create the descriptor
        // before `type_obj` exists, so fill the equivalent typed field here.
        if !descr.is_null() && pyre_object::is_member(descr) {
            let owner = pyre_object::w_member_get_cls(descr);
            if owner.is_null() {
                pyre_object::w_member_set_cls(descr, type_obj);
            }
        }
        if !descr.is_null() && pyre_object::typedef::is_getset_property(descr) {
            let descr_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(descr);
            let bound = copy_for_type(descr, type_obj);
            if !std::ptr::eq(bound, descr) {
                let ns = pyre_object::gc_roots::shadow_stack_get(save_point);
                let descr = pyre_object::gc_roots::shadow_stack_get(descr_slot);
                if !std::ptr::eq(bound, descr) {
                    pyre_object::w_dict_setitem_str_no_proxy(ns, &key, bound);
                }
            }
        }
    }
}

/// Create the root `object` type. MRO = [object].
fn new_root_typeobject(name: &str, init: fn(PyObjectRef)) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let ns = pyre_object::w_dict_new();
    pyre_object::gc_roots::pin_root(ns);
    init(ns);
    let ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    let type_obj = w_type_new_builtin(
        name,
        PY_NULL,
        ns as *mut u8,
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
    let ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    unsafe { stamp_new_descr_self(ns, type_obj) };
    type_obj
}

/// Create a builtin type with a single base. MRO = [self] + base.mro().
/// Layout defaults to INSTANCE_TYPE (general object layout).
fn new_typeobject_with_base(
    name: &str,
    init: impl FnOnce(PyObjectRef),
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
    init: impl FnOnce(PyObjectRef),
    base: PyObjectRef,
    layout_pytype: *const PyType,
) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let ns = pyre_object::w_dict_new();
    pyre_object::gc_roots::pin_root(ns);
    init(ns);
    let bases = w_tuple_new(vec![base]);
    let ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    let type_obj = w_type_new_builtin(name, bases, ns as *mut u8, layout_pytype);

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
        let has_dict = pyre_object::w_dict_getitem_str(ns, "__dict__").is_some();
        let has_weakref = pyre_object::w_dict_getitem_str(ns, "__weakref__").is_some();
        let layout = if reuse {
            parent_layout
        } else {
            let has_new = pyre_object::w_dict_getitem_str(ns, "__new__").is_some();
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
        mro.extend_from_slice(unsafe { (*base_mro).as_slice() });
    } else {
        mro.push(base);
    }
    unsafe { w_type_set_mro(type_obj, mro) };
    let ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    unsafe { stamp_new_descr_self(ns, type_obj) };
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
    init: impl FnOnce(PyObjectRef),
    bases: &[PyObjectRef],
) -> PyObjectRef {
    let layout_pytype = &INSTANCE_TYPE as *const PyType;
    let base = bases[0];
    let _roots = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let ns = pyre_object::w_dict_new();
    pyre_object::gc_roots::pin_root(ns);
    init(ns);
    let bases_tuple = w_tuple_new(bases.to_vec());
    let ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    let type_obj = w_type_new_builtin(name, bases_tuple, ns as *mut u8, layout_pytype);

    unsafe {
        let parent_layout = pyre_object::w_type_get_layout_ptr(base);
        let reuse = if !parent_layout.is_null() {
            std::ptr::eq((*parent_layout).typedef, layout_pytype)
        } else {
            false
        };
        let has_dict = pyre_object::w_dict_getitem_str(ns, "__dict__").is_some();
        let has_weakref = pyre_object::w_dict_getitem_str(ns, "__weakref__").is_some();
        let layout = if reuse {
            parent_layout
        } else {
            let has_new = pyre_object::w_dict_getitem_str(ns, "__new__").is_some();
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
    let ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    unsafe { stamp_new_descr_self(ns, type_obj) };
    type_obj
}

/// Create a named builtin type inheriting from `object`.
///
/// Used by extension modules (e.g. _sre) to define their own types.
/// typeobject.py:174 `is_heaptype=False` — builtin type.
pub fn make_builtin_type(name: &str, init: impl FnOnce(PyObjectRef)) -> PyObjectRef {
    new_typeobject_with_base(name, init, w_object())
}

/// Create a named builtin type inheriting from `base`.
pub fn make_builtin_type_with_base(
    name: &str,
    init: impl FnOnce(PyObjectRef),
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
    init: impl FnOnce(PyObjectRef),
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
    let w_module = pyre_object::w_module_new("");
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
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if positional.is_empty() {
        return Err(crate::PyError::type_error(
            "descriptor '__init__' of 'module' object needs an argument",
        ));
    }
    let given = positional.len().saturating_sub(1);
    if given > 2 {
        return Err(crate::PyError::type_error(format!(
            "module() takes at most 2 arguments ({given} given)"
        )));
    }
    let mut w_name = positional.get(1).copied();
    let mut w_doc = positional.get(2).copied();
    if let Some(kwargs) = kwargs {
        for (key, value) in unsafe { pyre_object::w_dict_str_entries_wtf8(kwargs) } {
            let Ok(key) = key.as_str() else {
                continue;
            };
            if key == "__pyre_kw__" {
                continue;
            }
            match key {
                "name" if w_name.is_none() => w_name = Some(value),
                "name" => {
                    return Err(crate::PyError::type_error(
                        "argument for module() given by name ('name') and position (1)",
                    ));
                }
                "doc" if w_doc.is_none() => w_doc = Some(value),
                "doc" => {
                    return Err(crate::PyError::type_error(
                        "argument for module() given by name ('doc') and position (2)",
                    ));
                }
                other => {
                    return Err(crate::PyError::type_error(format!(
                        "module() got an unexpected keyword argument '{other}'"
                    )));
                }
            }
        }
    }
    let Some(w_name) = w_name else {
        return Err(crate::PyError::type_error(
            "module() missing required argument 'name' (pos 1)",
        ));
    };
    let self_ = positional[0];
    if !unsafe { pyre_object::is_str(w_name) } {
        let received = if unsafe { pyre_object::is_none(w_name) } {
            "None".to_string()
        } else {
            crate::typedef::r#type(w_name)
                .map(|tp| unsafe { pyre_object::w_type_get_name(tp) }.to_string())
                .unwrap_or_else(|| unsafe { (*(*w_name).ob_type).name }.to_string())
        };
        return Err(crate::PyError::type_error(format!(
            "module() argument 'name' must be str, not {received}"
        )));
    }
    let w_doc = w_doc.unwrap_or_else(pyre_object::w_none);
    let name = unsafe { pyre_object::w_str_get_value(w_name) };
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

/// module.py:126-128 `Module.descr_module__repr__` — delegate to
/// `_frozen_importlib._module_repr`, which implements the spec/file/name
/// precedence shared by CPython 3.14.
fn module_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let module = module_require(args.first().copied().unwrap_or(PY_NULL), "__repr__")?;
    Ok(pyre_object::w_str_new(&module_repr_string(module)?))
}

fn module_require(obj: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    if obj.is_null() || !unsafe { pyre_object::is_module(obj) } {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' for 'module' objects doesn't apply to this object"
        )));
    }
    Ok(obj)
}

pub(crate) fn module_repr_string(module: PyObjectRef) -> Result<String, crate::PyError> {
    let importlib = crate::importing::get_sys_module("_frozen_importlib")
        .or_else(|| crate::importing::get_sys_module("importlib._bootstrap"));
    if let Some(importlib) = importlib {
        let repr_fn = crate::baseobjspace::getattr_str(importlib, "_module_repr")?;
        let result = crate::call::call_function_impl_result(repr_fn, &[module])?;
        return Ok(crate::baseobjspace::text_w(result)?.to_string());
    }
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    let loader = crate::baseobjspace::finditem_str(w_dict, "__loader__")?;
    if let Some(spec) = crate::baseobjspace::finditem_str(w_dict, "__spec__")? {
        if !spec.is_null()
            && !unsafe { pyre_object::is_none(spec) }
            && crate::baseobjspace::is_true(spec)?
        {
            let mut name = crate::baseobjspace::getattr_str(spec, "name")?;
            if unsafe { pyre_object::is_none(name) } {
                name = pyre_object::w_str_new("?");
            }
            let origin = crate::baseobjspace::getattr_str(spec, "origin")?;
            if unsafe { pyre_object::is_none(origin) } {
                let spec_loader = crate::baseobjspace::getattr_str(spec, "loader")?;
                if unsafe { pyre_object::is_none(spec_loader) } {
                    return Ok(format!("<module {}>", unsafe {
                        crate::display::py_repr(name)?
                    }));
                }
                return Ok(format!(
                    "<module {} ({})>",
                    unsafe { crate::display::py_repr(name)? },
                    unsafe { crate::display::py_repr(spec_loader)? }
                ));
            }
            let has_location = crate::baseobjspace::getattr_str(spec, "has_location")?;
            if crate::baseobjspace::is_true(has_location)? {
                return Ok(format!(
                    "<module {} from {}>",
                    unsafe { crate::display::py_repr(name)? },
                    unsafe { crate::display::py_repr(origin)? }
                ));
            }
            return Ok(format!(
                "<module {} ({})>",
                unsafe { crate::display::py_repr(name)? },
                unsafe { crate::display::py_str(origin)? }
            ));
        }
    }
    let name = crate::baseobjspace::finditem_str(w_dict, "__name__")?
        .unwrap_or_else(|| pyre_object::w_str_new("?"));
    let name_repr = unsafe { crate::display::py_repr(name)? };
    if let Some(filename) = crate::baseobjspace::finditem_str(w_dict, "__file__")? {
        return Ok(format!("<module {name_repr} from {}>", unsafe {
            crate::display::py_repr(filename)?
        }));
    }
    if let Some(loader) = loader {
        if !loader.is_null() && !unsafe { pyre_object::is_none(loader) } {
            return Ok(format!("<module {name_repr} ({})>", unsafe {
                crate::display::py_repr(loader)?
            }));
        }
    }
    Ok(format!("<module {name_repr}>"))
}

/// module.py:130-160 `Module.descr_getattribute`. The object-space module
/// path already performs the normal lookup, PEP 562 `__getattr__`, and the
/// module-specific AttributeError wording, so the descriptor is the direct
/// entry point into that same implementation.
fn module_descr_getattribute(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let module = module_require(args.first().copied().unwrap_or(PY_NULL), "__getattribute__")?;
    let name = crate::baseobjspace::text_w(args[1])?;
    crate::baseobjspace::getattr_str(module, name)
}

/// module.py:164-173 `Module.descr_module__dir__`.
fn module_descr_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let module = module_require(args.first().copied().unwrap_or(PY_NULL), "__dir__")?;
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    if w_dict.is_null()
        || (!unsafe { pyre_object::is_dict(w_dict) }
            && !unsafe { pyre_object::dictmultiobject::is_module_dict(w_dict) }
            && !gettypefor(&pyre_object::DICT_TYPE).is_some_and(|dict_type| unsafe {
                crate::baseobjspace::isinstance_w(w_dict, dict_type)
            }))
    {
        return Err(crate::PyError::type_error(format!(
            "{}.__dict__ is not a dictionary",
            unsafe { crate::display::py_repr(module)? }
        )));
    }
    if let Some(w_dir) = crate::baseobjspace::finditem_str(w_dict, "__dir__")? {
        return crate::call::call_function_impl_result(w_dir, &[]);
    }
    crate::builtins::builtin_list_ctor(&[w_dict])
}

fn module_annotations_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let module = module_require(args.get(1).copied().unwrap_or(PY_NULL), "__annotations__")?;
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_dict);
    if let Some(annotations) = crate::baseobjspace::finditem_str(w_dict, "__annotations__")? {
        return Ok(annotations);
    }
    let annotations = match crate::baseobjspace::finditem_str(w_dict, "__annotate__")? {
        Some(annotate) if !annotate.is_null() && !unsafe { pyre_object::is_none(annotate) } => {
            let annotate_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(annotate);
            let format = pyre_object::w_int_new(1);
            let result = crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(annotate_slot),
                &[format],
            )?;
            if !unsafe { pyre_object::is_dict(result) } {
                let received = crate::typedef::r#type(result)
                    .map(|tp| unsafe { pyre_object::w_type_get_name(tp) }.to_string())
                    .unwrap_or_else(|| unsafe { (*(*result).ob_type).name }.to_string());
                return Err(crate::PyError::type_error(format!(
                    "__annotate__ returned non-dict of type '{received}'"
                )));
            }
            // Module annotation functions consult the mutable
            // `__conditional_annotations__` set.  A partially executed
            // module may therefore produce a larger dict on a later read;
            // do not freeze the intermediate result in `__annotations__`.
            return Ok(result);
        }
        _ => pyre_object::w_dict_new(),
    };
    let annotations_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(annotations);
    let key = pyre_object::w_str_new("__annotations__");
    crate::baseobjspace::setitem(
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
        key,
        pyre_object::gc_roots::shadow_stack_get(annotations_slot),
    )?;
    Ok(pyre_object::gc_roots::shadow_stack_get(annotations_slot))
}

fn module_annotations_set(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let module = module_require(args.get(1).copied().unwrap_or(PY_NULL), "__annotations__")?;
    let value = args[2];
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_dict);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(value);
    let annotations_key = pyre_object::w_str_new("__annotations__");
    crate::baseobjspace::setitem(
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
        annotations_key,
        pyre_object::gc_roots::shadow_stack_get(value_slot),
    )?;
    let annotate_key = pyre_object::w_str_new("__annotate__");
    let _ = crate::baseobjspace::delitem(
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
        annotate_key,
    );
    Ok(pyre_object::w_none())
}

fn module_annotations_del(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let module = module_require(args.get(1).copied().unwrap_or(PY_NULL), "__annotations__")?;
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_dict);
    if crate::baseobjspace::finditem_str(w_dict, "__annotations__")?.is_none() {
        return Err(crate::PyError::attribute_error("__annotations__"));
    }
    let annotations_key = pyre_object::w_str_new("__annotations__");
    crate::baseobjspace::delitem(
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
        annotations_key,
    )?;
    let annotate_key = pyre_object::w_str_new("__annotate__");
    let _ = crate::baseobjspace::delitem(
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
        annotate_key,
    );
    Ok(pyre_object::w_none())
}

fn module_annotate_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let module = module_require(args.get(1).copied().unwrap_or(PY_NULL), "__annotate__")?;
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    if let Some(annotate) = crate::baseobjspace::finditem_str(w_dict, "__annotate__")? {
        return Ok(annotate);
    }
    let none = pyre_object::w_none();
    crate::baseobjspace::setitem(w_dict, pyre_object::w_str_new("__annotate__"), none)?;
    Ok(none)
}

fn module_annotate_set(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let module = module_require(args.get(1).copied().unwrap_or(PY_NULL), "__annotate__")?;
    let value = args[2];
    if !unsafe { pyre_object::is_none(value) } && !crate::baseobjspace::callable_w(value) {
        return Err(crate::PyError::type_error(
            "__annotate__ must be callable or None",
        ));
    }
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_dict);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(value);
    let annotate_key = pyre_object::w_str_new("__annotate__");
    crate::baseobjspace::setitem(
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
        annotate_key,
        pyre_object::gc_roots::shadow_stack_get(value_slot),
    )?;
    if !unsafe { pyre_object::is_none(value) } {
        let annotations_key = pyre_object::w_str_new("__annotations__");
        let _ = crate::baseobjspace::delitem(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            annotations_key,
        );
    }
    Ok(pyre_object::w_none())
}

fn module_annotate_del(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Err(crate::PyError::type_error(
        "cannot delete __annotate__ attribute",
    ))
}

/// `module.py Module.typedef` — wire `__new__` / `__init__` so
/// `type(m)(name)` builds a real module.  `module` defines its own
/// `tp_new`, so `module.__new__ is not object.__new__`.
fn init_module_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(module_descr_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__init__",
            make_builtin_function("__init__", module_descr_init),
        )
    };
    for (name, function, arity) in [
        (
            "__repr__",
            module_descr_repr as crate::gateway::BuiltinCodeFn,
            1,
        ),
        ("__getattribute__", module_descr_getattribute, 2),
        ("__dir__", module_descr_dir, 1),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, function, arity),
            )
        };
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__dict__",
            pyre_object::w_member_new_direct(
                pyre_object::MEMBER_MODULE_DICT,
                "__dict__".to_owned(),
                pyre_object::PY_NULL,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__annotations__",
            make_getset_property(
                make_builtin_function_with_arity("__annotations__", module_annotations_get, 2),
                make_builtin_function_with_arity("__annotations__", module_annotations_set, 3),
                make_builtin_function_with_arity("__annotations__", module_annotations_del, 2),
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__annotate__",
            make_getset_property(
                make_builtin_function_with_arity("__annotate__", module_annotate_get, 2),
                make_builtin_function_with_arity("__annotate__", module_annotate_set, 3),
                make_builtin_function_with_arity("__annotate__", module_annotate_del, 2),
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            pyre_object::w_str_new(
                "Create a module object.\n\nThe name must be a string; the optional doc argument can have any type.",
            ),
        )
    };
}

fn singleton_receiver(
    args: &[PyObjectRef],
    owner: &str,
    name: &str,
    predicate: unsafe fn(PyObjectRef) -> bool,
) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if self_.is_null() || !unsafe { predicate(self_) } {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' requires a '{owner}' object but received a '{}'",
            type_name_of(self_),
        )));
    }
    Ok(self_)
}

fn ellipsis_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() > 1 || crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "EllipsisType takes no arguments",
        ));
    }
    let cls = positional.first().copied().unwrap_or(pyre_object::PY_NULL);
    if let Some(w_ellipsis) = gettypefor(&pyre_object::ELLIPSIS_TYPE) {
        check_user_subclass(w_ellipsis, cls)?;
    }
    Ok(pyre_object::special::w_ellipsis())
}

fn init_ellipsis_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new("The type of the Ellipsis singleton."),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(ellipsis_descr_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    singleton_receiver(args, "ellipsis", "__repr__", pyre_object::is_ellipsis)?;
                    Ok(w_str_new("Ellipsis"))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                |args| {
                    singleton_receiver(args, "ellipsis", "__reduce__", pyre_object::is_ellipsis)?;
                    Ok(w_str_new("Ellipsis"))
                },
                1,
            ),
        )
    };
}

/// special.py:20: NotImplemented.descr_new_notimplemented
fn notimplemented_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() > 1 || crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "NotImplementedType takes no arguments",
        ));
    }
    let cls = positional.first().copied().unwrap_or(pyre_object::PY_NULL);
    if let Some(w_notimplemented) = gettypefor(&pyre_object::pyobject::NOTIMPLEMENTED_TYPE) {
        check_user_subclass(w_notimplemented, cls)?;
    }
    Ok(pyre_object::special::w_not_implemented())
}

/// typedef.py:948-954 NotImplemented.typedef
fn init_notimplemented_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::w_dict_setitem_str(
            ns,
            "__doc__",
            pyre_object::w_str_new("The type of the NotImplemented singleton."),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(notimplemented_descr_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    singleton_receiver(
                        args,
                        "NotImplementedType",
                        "__repr__",
                        pyre_object::is_not_implemented,
                    )?;
                    Ok(w_str_new("NotImplemented"))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                |args| {
                    singleton_receiver(
                        args,
                        "NotImplementedType",
                        "__reduce__",
                        pyre_object::is_not_implemented,
                    )?;
                    Ok(w_str_new("NotImplemented"))
                },
                1,
            ),
        )
    };
    // Python 3.14 changed the older PyPy/3.11 deprecation warning into a hard
    // TypeError.  3.14 is pyre's language-version oracle.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__bool__",
            make_builtin_function_with_arity(
                "__bool__",
                |args| {
                    singleton_receiver(
                        args,
                        "NotImplementedType",
                        "__bool__",
                        pyre_object::is_not_implemented,
                    )?;
                    Err(crate::PyError::type_error(
                        "NotImplemented should not be used in a boolean context",
                    ))
                },
                1,
            ),
        )
    };
}

/// noneobject.py `W_NoneObject.typedef`, plus Python 3.14's singleton rich
/// comparison/hash entries.  Equality is identity; ordering returns
/// NotImplemented so the normal comparison dispatcher produces the final
/// TypeError when neither operand handles it.
fn none_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() > 1 || crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error("NoneType takes no arguments"));
    }
    let cls = positional.first().copied().unwrap_or(pyre_object::PY_NULL);
    if let Some(w_none_type) = gettypefor(&pyre_object::NONE_TYPE) {
        check_user_subclass(w_none_type, cls)?;
    }
    Ok(pyre_object::w_none())
}

fn none_ordering(args: &[PyObjectRef], name: &str) -> Result<PyObjectRef, crate::PyError> {
    singleton_receiver(args, "NoneType", name, pyre_object::is_none)?;
    Ok(pyre_object::w_not_implemented())
}

fn none_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    none_ordering(args, "__lt__")
}

fn none_le(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    none_ordering(args, "__le__")
}

fn none_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    none_ordering(args, "__gt__")
}

fn none_ge(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    none_ordering(args, "__ge__")
}

fn init_none_type(ns: PyObjectRef) {
    let entries = [
        (
            "__doc__",
            pyre_object::w_str_new("The type of the None singleton."),
        ),
        ("__new__", make_new_descr(none_descr_new)),
        (
            "__bool__",
            make_builtin_function_with_arity(
                "__bool__",
                |args| {
                    singleton_receiver(args, "NoneType", "__bool__", pyre_object::is_none)?;
                    Ok(pyre_object::w_bool_from(false))
                },
                1,
            ),
        ),
        (
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    singleton_receiver(args, "NoneType", "__repr__", pyre_object::is_none)?;
                    Ok(w_str_new("None"))
                },
                1,
            ),
        ),
        (
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| {
                    let self_ =
                        singleton_receiver(args, "NoneType", "__hash__", pyre_object::is_none)?;
                    Ok(pyre_object::w_int_new(crate::builtins::hash_value(self_)))
                },
                1,
            ),
        ),
        (
            "__eq__",
            make_builtin_function_with_arity(
                "__eq__",
                |args| {
                    singleton_receiver(args, "NoneType", "__eq__", pyre_object::is_none)?;
                    if args.len() >= 2 && unsafe { pyre_object::is_none(args[1]) } {
                        Ok(pyre_object::w_bool_from(true))
                    } else {
                        Ok(pyre_object::w_not_implemented())
                    }
                },
                2,
            ),
        ),
        (
            "__ne__",
            make_builtin_function_with_arity(
                "__ne__",
                |args| {
                    singleton_receiver(args, "NoneType", "__ne__", pyre_object::is_none)?;
                    if args.len() >= 2 && unsafe { pyre_object::is_none(args[1]) } {
                        Ok(pyre_object::w_bool_from(false))
                    } else {
                        Ok(pyre_object::w_not_implemented())
                    }
                },
                2,
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str(ns, name, value) };
    }
    for (name, function) in [
        ("__lt__", none_lt as DunderFn),
        ("__le__", none_le as DunderFn),
        ("__gt__", none_gt as DunderFn),
        ("__ge__", none_ge as DunderFn),
    ] {
        unsafe {
            pyre_object::w_dict_setitem_str(
                ns,
                name,
                make_builtin_function_with_arity(name, function, 2),
            )
        };
    }
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
    if cls.is_null() {
        return Ok(value);
    }
    let str_typeobj = gettypefor(&pyre_object::STR_TYPE);
    if str_typeobj.map_or(false, |t| std::ptr::eq(cls, t)) {
        return Ok(value);
    }
    if !unsafe { pyre_object::is_type(cls) } {
        return Err(crate::PyError::type_error(
            "str.__new__(X): X is not a subtype of str",
        ));
    }
    if let Some(w_str) = str_typeobj {
        if !unsafe { crate::baseobjspace::issubtype_w(cls, w_str) } {
            let cls_name = unsafe { pyre_object::w_type_get_name(cls) };
            return Err(crate::PyError::type_error(format!(
                "str.__new__({cls_name}): {cls_name} is not a subtype of str"
            )));
        }
    }
    let contents = unsafe { pyre_object::w_str_get_wtf8(value) }.to_wtf8_buf();
    Ok(pyre_object::w_str_subclass_from_wtf8(contents, cls))
}

/// `dictmultiobject.py:115-117 descr_new` — allocate the instance and return
/// it; `__args__` is ignored.
///
/// `__new__` must NOT populate from the constructor arguments:
/// `object.__new__`/`dict.__new__` ignore them, and filling is the job of
/// `__init__` (`:137-138 descr_init` → `:1430 init_or_update`, the inherited
/// `dict.__init__` for a plain subclass, or the subclass override).
/// Pre-filling here consumes the argument a second time: a re-iterable one is
/// walked twice (`dict(mapping)` calling `keys`/`__getitem__` twice), a
/// one-shot one raises on the second walk, and a subclass whose `__init__`
/// accumulates (e.g. `Counter`, whose `update` adds rather than sets)
/// double-applies it.
fn dict_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        args[0]
    };
    let dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);

    if cls.is_null() || std::ptr::eq(cls, dict_type) {
        return Ok(pyre_object::w_dict_new());
    }

    // A dict subclass keeps its items in an instance attribute, so the
    // allocation carries an empty backing dict for `__init__` to fill.
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
    let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "bool() takes no keyword arguments",
        ));
    }
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
                        // A tagged immediate is always an exact `int`; name it
                        // without derefing its (non-pointer) tagged bits as
                        // `ob_type`. Mirrors the tag short-circuit in
                        // `builtin_str`. Gated on `CAN_BE_TAGGED`.
                        let tp_name: &str = if pyre_object::tagged_int::CAN_BE_TAGGED
                            && pyre_object::tagged_int::is_tagged_int(result)
                        {
                            "int"
                        } else {
                            (*(*result).ob_type).name
                        };
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

/// `list.__new__(cls, *args)` allocates an empty list.  Population belongs to
/// `list.__init__`, including for subclasses which inherit that initializer.
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
    let value = pyre_object::w_list_new(Vec::new());
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::LIST_TYPE)? {
        unsafe {
            (*value).w_class = sub;
        }
        // objspace.py `allocate_instance`: a builtin-layout subclass still
        // participates in the user-finalizer queue when its Python type has
        // `__del__`. Registration must follow `w_class` tagging so the hook
        // sees the subclass rather than the canonical list type.
        pyre_object::gc_hook::maybe_register_finalizer(value);
    }
    Ok(value)
}

fn list_descr_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (params, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let list = crate::type_methods::require_list_receiver(params, "__init__", false)?;
    // CPython 3.14 clinic/listobject.c.h `list___init__`: keywords are
    // rejected for exact list and subclasses which inherit list.__new__, but
    // ignored here when the subclass overrides __new__ (that override already
    // received them through type.__call__).
    let list_type = gettypeobject(&pyre_object::LIST_TYPE);
    let instance_type = crate::typedef::r#type(list).unwrap_or(list_type);
    let inherits_list_new = unsafe {
        match (
            crate::baseobjspace::lookup_in_type(instance_type, "__new__"),
            crate::baseobjspace::lookup_in_type(list_type, "__new__"),
        ) {
            (Some(instance_new), Some(list_new)) => std::ptr::eq(instance_new, list_new),
            _ => true,
        }
    };
    if inherits_list_new && crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "list() takes no keyword arguments",
        ));
    }
    if params.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "list expected at most 1 argument, got {}",
            params.len() - 1
        )));
    }
    unsafe { pyre_object::w_list_clear(list) };
    if let Some(&iterable) = params.get(1) {
        // listobject.py:557-566 — append as the iterator yields, retaining
        // a completed prefix if a later next() fails.
        crate::type_methods::list_method_extend(&[list, iterable])?;
    }
    Ok(pyre_object::w_none())
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
        pyre_object::gc_hook::maybe_register_finalizer(fresh);
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
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let value = crate::builtins::builtin_map(args.get(1..).unwrap_or(&[]))?;
    pyre_object::gc_roots::pin_root(value);
    let value_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let value = unsafe { pyre_object::gc_roots::shadow_stack_get(value_slot) };
    let cls = unsafe { pyre_object::gc_roots::shadow_stack_get(cls_slot) };
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
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let cls = positional.first().copied().unwrap_or(pyre_object::PY_NULL);
    let args_w = positional.get(1..).unwrap_or(&[]);
    if args_w.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "filter expected 2 arguments, got {}",
            args_w.len()
        )));
    }

    // `space.getattr` and keyword-name inspection are allowed to allocate in
    // the source gateway. Keep the subtype and both positional arguments live
    // before reproducing that conditional keyword check.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(args_w[0]);
    let predicate_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(args_w[1]);
    let iterable_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let kwargs_slot = kwargs.map(|dict| {
        pyre_object::gc_roots::pin_root(dict);
        pyre_object::gc_roots::shadow_stack_len() - 1
    });
    let cls = unsafe { pyre_object::gc_roots::shadow_stack_get(cls_slot) };
    let filter_type =
        gettypefor(&pyre_object::functional::FILTER_TYPE).unwrap_or(pyre_object::PY_NULL);
    let init_matches = std::ptr::eq(cls, filter_type)
        || unsafe {
            match (
                crate::baseobjspace::lookup_in_type(cls, "__init__"),
                crate::baseobjspace::lookup_in_type(filter_type, "__init__"),
            ) {
                (Some(sub), Some(base)) => std::ptr::eq(sub, base),
                (None, None) => true,
                _ => false,
            }
        };
    let rooted_kwargs =
        kwargs_slot.map(|slot| unsafe { pyre_object::gc_roots::shadow_stack_get(slot) });
    if init_matches && crate::builtins::has_real_kwargs(rooted_kwargs) {
        return Err(crate::PyError::type_error(
            "filter() takes no keyword arguments",
        ));
    }
    let value = crate::builtins::builtin_filter(&[
        unsafe { pyre_object::gc_roots::shadow_stack_get(predicate_slot) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(iterable_slot) },
    ])?;
    pyre_object::gc_roots::pin_root(value);
    let value_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let value = unsafe { pyre_object::gc_roots::shadow_stack_get(value_slot) };
    let cls = unsafe { pyre_object::gc_roots::shadow_stack_get(cls_slot) };
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
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let value = crate::builtins::builtin_zip(args.get(1..).unwrap_or(&[]))?;
    pyre_object::gc_roots::pin_root(value);
    let value_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let value = unsafe { pyre_object::gc_roots::shadow_stack_get(value_slot) };
    let cls = unsafe { pyre_object::gc_roots::shadow_stack_get(cls_slot) };
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
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let value = crate::builtins::builtin_reversed(args.get(1..).unwrap_or(&[]))?;
    pyre_object::gc_roots::pin_root(value);
    let value_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let value = unsafe { pyre_object::gc_roots::shadow_stack_get(value_slot) };
    if unsafe { pyre_object::functional::is_reversed(value) } {
        let cls = unsafe { pyre_object::gc_roots::shadow_stack_get(cls_slot) };
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

/// `descriptor.py W_Super.__new__` — allocate an uninitialised field-resident
/// proxy; `__init__` fills it from zero, one, or two user arguments.
fn super_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(PY_NULL);
    let value = pyre_object::descriptor::w_super_new(PY_NULL, PY_NULL);
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::descriptor::SUPER_TYPE)? {
        unsafe { (*value).w_class = sub };
    }
    Ok(value)
}

fn super_descr_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let fresh = crate::builtins::builtin_super(args.get(1..).unwrap_or(&[]))?;
    unsafe {
        pyre_object::descriptor::w_super_set_fields(
            args[0],
            pyre_object::descriptor::w_super_get_type(fresh),
            pyre_object::descriptor::w_super_get_obj(fresh),
        )
    };
    Ok(w_none())
}

fn super_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let start = unsafe { pyre_object::descriptor::w_super_get_type(args[0]) };
    let bound = unsafe { pyre_object::descriptor::w_super_get_obj(args[0]) };
    let start_name = if start.is_null() {
        "NULL".to_string()
    } else {
        unsafe { pyre_object::w_type_get_name(start) }.to_string()
    };
    let bound_name = if bound.is_null() {
        "NULL".to_string()
    } else {
        let bound_type = crate::builtins::super_check(start, bound)?;
        format!("<{} object>", unsafe {
            pyre_object::w_type_get_name(bound_type)
        })
    };
    Ok(w_str_new(&format!(
        "<super: <class '{}'>, {}>",
        start_name, bound_name
    )))
}

fn super_descr_getattribute(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { pyre_object::is_str(args[1]) } {
        return Err(crate::PyError::type_error("attribute name must be string"));
    }
    crate::baseobjspace::getattr_str(args[0], unsafe { pyre_object::w_str_get_value(args[1]) })
}

fn super_descr_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args[0];
    let obj = args.get(1).copied().unwrap_or_else(w_none);
    let bound = unsafe { pyre_object::descriptor::w_super_get_obj(self_) };
    if !bound.is_null() || unsafe { pyre_object::is_none(obj) } {
        return Ok(self_);
    }
    let start = unsafe { pyre_object::descriptor::w_super_get_type(self_) };
    if start.is_null() {
        return Err(crate::PyError::type_error(
            "__get__(x) is invalid on an uninitialized instance of 'super'",
        ));
    }
    let cls = r#type(self_).unwrap_or_else(|| gettypeobject(&pyre_object::descriptor::SUPER_TYPE));
    crate::call::call_function_impl_result(cls, &[start, obj])
}

fn super_getter(args: &[PyObjectRef], field: usize) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.get(1).copied().unwrap_or(PY_NULL);
    if !unsafe { pyre_object::descriptor::is_super(self_) } {
        return Err(crate::PyError::type_error("descriptor is for 'super'"));
    }
    let start = unsafe { pyre_object::descriptor::w_super_get_type(self_) };
    let bound = unsafe { pyre_object::descriptor::w_super_get_obj(self_) };
    Ok(match field {
        0 => {
            if start.is_null() {
                w_none()
            } else {
                start
            }
        }
        1 => {
            if bound.is_null() {
                w_none()
            } else {
                bound
            }
        }
        _ => {
            if bound.is_null() {
                w_none()
            } else {
                crate::builtins::super_check(start, bound)?
            }
        }
    })
}

fn super_get_thisclass(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    super_getter(args, 0)
}

fn super_get_self(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    super_getter(args, 1)
}

fn super_get_self_class(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    super_getter(args, 2)
}

/// PyPy `descriptor.py W_Super.typedef`, with Python 3.14's zero-argument
/// documentation and concrete type surface.
fn init_super_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::w_dict_setitem_str(
            ns,
            "__doc__",
            w_str_new(
                "super() -> same as super(__class__, <first argument>)\n\
                 super(type) -> unbound super object\n\
                 super(type, obj) -> bound super object; requires isinstance(obj, type)\n\
                 super(type, type2) -> bound super object; requires issubclass(type2, type)\n\
                 Typical use to call a cooperative superclass method:\n\
                 class C(B):\n\
                     def meth(self, arg):\n\
                         super().meth(arg)\n\
                 This works for class methods too:\n\
                 class C(B):\n\
                     @classmethod\n\
                     def cmeth(cls, arg):\n\
                         super().cmeth(arg)",
            ),
        )
    };
    for (name, value) in [
        ("__new__", make_new_descr(super_descr_new)),
        (
            "__init__",
            make_builtin_function("__init__", super_descr_init),
        ),
        (
            "__repr__",
            make_builtin_function_with_arity("__repr__", super_descr_repr, 1),
        ),
        (
            "__getattribute__",
            make_builtin_function_with_arity("__getattribute__", super_descr_getattribute, 2),
        ),
        ("__get__", make_builtin_function("__get__", super_descr_get)),
    ] {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
    for (name, getter) in [
        ("__thisclass__", super_get_thisclass as DunderFn),
        ("__self__", super_get_self as DunderFn),
        ("__self_class__", super_get_self_class as DunderFn),
    ] {
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_getset_descriptor_named(
                    make_builtin_function_with_arity(name, getter, 2),
                    name,
                ),
            )
        };
    }
}

/// `functional.py W_Range.descr_repr`.
fn range_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_str_new(&unsafe { crate::display::py_repr(args[0])? }))
}

/// `functional.py W_Range.descr_getitem`.
fn range_descr_getitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::getitem(args[0], args[1])
}

/// `functional.py W_Range.descr_len`, with Python 3.14's direct-descriptor
/// ssize_t overflow behavior for a range whose length does not fit.
fn range_descr_len(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::len_slot(args[0])
}

/// `functional.py W_Range.descr_contains`.
fn range_descr_contains(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(crate::baseobjspace::contains(
        args[0], args[1],
    )?))
}

/// CPython 3.14 `range_richcompare`, layered on PyPy
/// `functional.py W_Range.descr_eq`.  PyPy only publishes `descr_eq`; 3.14
/// takes precedence and exposes the whole rich-comparison slot surface.
fn range_descr_richcompare(
    args: &[PyObjectRef],
    op: crate::baseobjspace::CompareOp,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { pyre_object::is_w_range(args[1]) } {
        return Ok(w_not_implemented());
    }
    match op {
        crate::baseobjspace::CompareOp::Eq | crate::baseobjspace::CompareOp::Ne => {
            let mut result = unsafe { pyre_object::w_range_eq(args[0], args[1]) };
            if matches!(op, crate::baseobjspace::CompareOp::Ne) {
                result = !result;
            }
            Ok(w_bool_from(result))
        }
        crate::baseobjspace::CompareOp::Le
        | crate::baseobjspace::CompareOp::Ge
        | crate::baseobjspace::CompareOp::Lt
        | crate::baseobjspace::CompareOp::Gt => Ok(w_not_implemented()),
    }
}

fn range_descr_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    range_descr_richcompare(args, crate::baseobjspace::CompareOp::Eq)
}

fn range_descr_ne(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    range_descr_richcompare(args, crate::baseobjspace::CompareOp::Ne)
}

fn range_descr_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    range_descr_richcompare(args, crate::baseobjspace::CompareOp::Lt)
}

fn range_descr_le(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    range_descr_richcompare(args, crate::baseobjspace::CompareOp::Le)
}

fn range_descr_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    range_descr_richcompare(args, crate::baseobjspace::CompareOp::Gt)
}

fn range_descr_ge(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    range_descr_richcompare(args, crate::baseobjspace::CompareOp::Ge)
}

/// `functional.py W_Range.descr_bool`.
fn range_descr_bool(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(unsafe { pyre_object::w_range_bool(args[0]) }))
}

fn range_getter(args: &[PyObjectRef], field: usize) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.get(1).copied().unwrap_or(PY_NULL);
    if !unsafe { pyre_object::is_w_range(self_) } {
        return Err(crate::PyError::type_error("descriptor is for 'range'"));
    }
    let (start, stop, step) = unsafe { pyre_object::w_range_fields(self_) };
    Ok(match field {
        0 => start,
        1 => stop,
        _ => step,
    })
}

fn range_get_start(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    range_getter(args, 0)
}

fn range_get_stop(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    range_getter(args, 1)
}

fn range_get_step(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    range_getter(args, 2)
}

/// PyPy `functional.py W_Range.typedef`, kept in the same entry order.
fn init_range_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::w_dict_setitem_str(
            ns,
            "__doc__",
            w_str_new(
                "range(stop) -> range object\n\
                 range(start, stop[, step]) -> range object\n\n\
                 Return an object that produces a sequence of integers from start (inclusive)\n\
                 to stop (exclusive) by step.  range(i, j) produces i, i+1, i+2, ..., j-1.\n\
                 start defaults to 0, and stop is omitted!  range(4) produces 0, 1, 2, 3.\n\
                 These are exactly the valid indices for a list of 4 elements.\n\
                 When step is given, it specifies the increment (or decrement).",
            ),
        )
    };
    let entries = [
        ("__new__", make_new_descr(range_descr_new)),
        (
            "__repr__",
            make_builtin_function_with_arity("__repr__", range_descr_repr, 1),
        ),
        (
            "__getitem__",
            make_builtin_function_with_arity("__getitem__", range_descr_getitem, 2),
        ),
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::range_iter_method, 1),
        ),
        (
            "__len__",
            make_builtin_function_with_arity("__len__", range_descr_len, 1),
        ),
        (
            "__reversed__",
            make_builtin_function_with_arity(
                "__reversed__",
                crate::baseobjspace::range_reversed_method,
                1,
            ),
        ),
        (
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                crate::baseobjspace::range_reduce_method,
                1,
            ),
        ),
        (
            "__contains__",
            make_builtin_function_with_arity("__contains__", range_descr_contains, 2),
        ),
        (
            "__eq__",
            make_builtin_function_with_arity("__eq__", range_descr_eq, 2),
        ),
        (
            "__ne__",
            make_builtin_function_with_arity("__ne__", range_descr_ne, 2),
        ),
        (
            "__lt__",
            make_builtin_function_with_arity("__lt__", range_descr_lt, 2),
        ),
        (
            "__le__",
            make_builtin_function_with_arity("__le__", range_descr_le, 2),
        ),
        (
            "__gt__",
            make_builtin_function_with_arity("__gt__", range_descr_gt, 2),
        ),
        (
            "__ge__",
            make_builtin_function_with_arity("__ge__", range_descr_ge, 2),
        ),
        (
            "__hash__",
            make_builtin_function_with_arity("__hash__", crate::baseobjspace::range_hash_method, 1),
        ),
        (
            "__bool__",
            make_builtin_function_with_arity("__bool__", range_descr_bool, 1),
        ),
        (
            "count",
            make_builtin_function_with_arity("count", crate::baseobjspace::range_count_method, 2),
        ),
        (
            "index",
            make_builtin_function_with_arity("index", crate::baseobjspace::range_index_method, 2),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
    for (name, getter_fn) in [
        ("start", range_get_start as DunderFn),
        ("stop", range_get_stop as DunderFn),
        ("step", range_get_step as DunderFn),
    ] {
        let getter = make_builtin_function_with_arity(name, getter_fn, 2);
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_getset_descriptor_named(getter, name),
            )
        };
    }
}

fn install_functional_entry(ns: PyObjectRef, name: &'static str, value: PyObjectRef) {
    unsafe { pyre_object::w_dict_setitem_str(ns, name, value) };
}

/// PyPy `functional.py W_Enumerate.typedef`.
fn init_enumerate_type(ns: PyObjectRef) {
    install_functional_entry(
        ns,
        "__doc__",
        w_str_new(
            "Return an enumerate object.\n\n  iterable\n    an object supporting iteration\n\nThe enumerate object yields pairs containing a count (from start, which\ndefaults to zero) and a value yielded by the iterable argument.\n\nenumerate is useful for obtaining an indexed list:\n    (0, seq[0]), (1, seq[1]), (2, seq[2]), ...",
        ),
    );
    install_functional_entry(ns, "__new__", make_new_descr(enumerate_descr_new));
    install_functional_entry(
        ns,
        "__iter__",
        make_builtin_function_with_arity("__iter__", crate::baseobjspace::enumerate_iter_method, 1),
    );
    install_functional_entry(
        ns,
        "__next__",
        make_builtin_function_with_arity("__next__", crate::baseobjspace::enumerate_next_method, 1),
    );
    install_functional_entry(
        ns,
        "__reduce__",
        make_builtin_function_with_arity(
            "__reduce__",
            crate::baseobjspace::enumerate_reduce_method,
            1,
        ),
    );
    install_functional_entry(
        ns,
        "__class_getitem__",
        pyre_object::function::w_classmethod_new(make_builtin_function(
            "__class_getitem__",
            crate::_pypy_generic_alias::generic_alias_class_getitem,
        )),
    );
}

/// PyPy `functional.py W_ReversedIterator.typedef`.
fn init_reversed_type(ns: PyObjectRef) {
    install_functional_entry(
        ns,
        "__doc__",
        w_str_new("Return a reverse iterator over the values of the given sequence."),
    );
    install_functional_entry(ns, "__new__", make_new_descr(reversed_descr_new));
    for (name, function, arity) in [
        (
            "__iter__",
            crate::baseobjspace::reversed_iter_method as DunderFn,
            1,
        ),
        ("__next__", crate::baseobjspace::reversed_next_method, 1),
        (
            "__length_hint__",
            crate::baseobjspace::reversed_length_hint_method,
            1,
        ),
        ("__reduce__", crate::baseobjspace::reversed_reduce_method, 1),
        (
            "__setstate__",
            crate::baseobjspace::reversed_setstate_method,
            2,
        ),
    ] {
        install_functional_entry(
            ns,
            name,
            make_builtin_function_with_arity(name, function, arity),
        );
    }
}

/// PyPy `functional.py W_Map.typedef`, plus Python 3.14's exposed state slot.
fn init_map_type(ns: PyObjectRef) {
    install_functional_entry(
        ns,
        "__doc__",
        w_str_new(
            "map(func, *iterables) --> map object\n\nMake an iterator that computes the function using arguments from\neach of the iterables.  Stops when the shortest iterable is exhausted.",
        ),
    );
    install_functional_entry(ns, "__new__", make_new_descr(map_descr_new));
    for (name, function, arity) in [
        (
            "__iter__",
            crate::baseobjspace::map_iter_method as DunderFn,
            1,
        ),
        ("__next__", crate::baseobjspace::map_next_method, 1),
        ("__reduce__", crate::baseobjspace::map_reduce_method, 1),
        ("__setstate__", crate::baseobjspace::map_setstate_method, 2),
    ] {
        install_functional_entry(
            ns,
            name,
            make_builtin_function_with_arity(name, function, arity),
        );
    }
}

/// PyPy `functional.py W_Filter.typedef`.
fn init_filter_type(ns: PyObjectRef) {
    install_functional_entry(
        ns,
        "__doc__",
        w_str_new(
            "filter(function or None, iterable) --> filter object\n\nReturn an iterator yielding those items of iterable for which function(item)\nis true. If function is None, return the items that are true.",
        ),
    );
    install_functional_entry(ns, "__new__", make_new_descr(filter_descr_new));
    for (name, function) in [
        (
            "__iter__",
            crate::baseobjspace::filter_iter_method as DunderFn,
        ),
        ("__next__", crate::baseobjspace::filter_next_method),
        ("__reduce__", crate::baseobjspace::filter_reduce_method),
    ] {
        install_functional_entry(
            ns,
            name,
            make_builtin_function_with_arity(name, function, 1),
        );
    }
}

/// PyPy `functional.py W_Zip.typedef`.
fn init_zip_type(ns: PyObjectRef) {
    install_functional_entry(
        ns,
        "__doc__",
        w_str_new(
            "zip(*iterables) --> A zip object yielding tuples until an input is exhausted.\n\nThe zip object yields n-length tuples, where n is the number of iterables\npassed as positional arguments to zip().  The i-th element in every tuple\ncomes from the i-th iterable argument to zip().  This continues until the\nshortest argument is exhausted.",
        ),
    );
    install_functional_entry(ns, "__new__", make_new_descr(zip_descr_new));
    for (name, function, arity) in [
        (
            "__iter__",
            crate::baseobjspace::zip_iter_method as DunderFn,
            1,
        ),
        ("__next__", crate::baseobjspace::zip_next_method, 1),
        ("__reduce__", crate::baseobjspace::zip_reduce_method, 1),
        ("__setstate__", crate::baseobjspace::zip_setstate_method, 2),
    ] {
        install_functional_entry(
            ns,
            name,
            make_builtin_function_with_arity(name, function, arity),
        );
    }
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
pub(crate) fn check_user_subclass(
    w_self: PyObjectRef,
    w_subtype: PyObjectRef,
) -> Result<(), crate::PyError> {
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
    let is_sub = !mro_ptr.is_null()
        && unsafe {
            (*mro_ptr)
                .as_slice()
                .iter()
                .any(|&t| std::ptr::eq(t, w_self))
        };
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
    // objspace.py:486 `allocate_instance` registers every freshly allocated
    // instance whose class carries `hasuserdel`.  Set/frozenset subclasses use
    // this layout-specific allocator instead of `w_instance_new`, so perform
    // the same post-allocation step after installing the real subclass.
    pyre_object::gc_hook::maybe_register_finalizer(obj);
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

    // setobject.py — reuse the argument only when the target type is
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
        set_init_from_iterable(obj, iterable)?;
    }
    Ok(obj)
}

/// Fill a freshly allocated set/frozenset from `w_iterable`.
///
/// `setobject.py set_strategy_and_setdata` — the storage is set up from
/// the iterable, and a set operand (`:1619-1621`) hands its own over rather
/// than being walked: its elements hashed when they entered it.
fn set_init_from_iterable(
    w_set: PyObjectRef,
    w_iterable: PyObjectRef,
) -> Result<(), crate::PyError> {
    if unsafe { pyre_object::is_set_or_frozenset(w_iterable) } {
        unsafe { pyre_object::w_set_copy_storage_from(w_set, w_iterable) };
        return Ok(());
    }
    // RPython's shadowstack transformation keeps `self` live across
    // `space.listview(w_iterable)`.  `collect_iterable` can execute a long
    // generator and trigger a major collection, so mirror that generated
    // root explicitly: the set body is non-moving old-gen storage, but it
    // must still be marked live instead of being swept while Rust holds the
    // only reference.
    let _roots = pyre_object::gc_roots::push_roots();
    let set_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_set);
    let iterable_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_iterable);
    // Python 3.14 `set_update_dict_lock_held`: an exact dict is walked
    // through its key table and each cached hash is handed directly to the
    // set.  This is observable when a key's `__hash__` has side effects, and
    // it is the cached-hash shape of PyPy's
    // `DictStrategy.getiteritems_with_hash`.  A dict subclass must still use
    // normal iteration because it may override `__iter__`.
    if unsafe {
        pyre_object::is_exact_type(
            pyre_object::gc_roots::shadow_stack_get(iterable_slot),
            &pyre_object::DICT_TYPE,
        )
    } {
        let mut index = 0usize;
        let mut copied_hashed_key = false;
        while let Some(key) = unsafe {
            pyre_object::dictmultiobject::w_dict_nth_hashed_key(
                pyre_object::gc_roots::shadow_stack_get(iterable_slot),
                index,
            )
        } {
            unsafe {
                pyre_object::setobject::w_set_insert_key_checked(
                    pyre_object::gc_roots::shadow_stack_get(set_slot),
                    key,
                )
            }
            .map_err(|_| crate::baseobjspace::take_pending_hash_error())?;
            copied_hashed_key = true;
            index += 1;
        }
        // Empty object-shaped dicts and non-empty ones both finish here.  For
        // an empty dict the active Object/Unicode strategy is detected by a
        // zero length; typed/Empty strategies deliberately fall through.
        let strategy = unsafe {
            pyre_object::w_dict_get_strategy(pyre_object::gc_roots::shadow_stack_get(iterable_slot))
                .strategy_kind()
        };
        if copied_hashed_key
            || matches!(
                strategy,
                pyre_object::dictmultiobject::StrategyKind::Object
                    | pyre_object::dictmultiobject::StrategyKind::Unicode
            )
        {
            return Ok(());
        }
    }
    let items =
        crate::builtins::collect_iterable(pyre_object::gc_roots::shadow_stack_get(iterable_slot))?;
    let w_set = pyre_object::gc_roots::shadow_stack_get(set_slot);
    crate::builtins::builtin_set_add_items(w_set, &items)
}

/// `set.__init__(self, [iterable])` — PyPy: setobject.py W_SetObject.descr_init.
///
/// PyPy parses `__args__` against `init_signature = Signature(['some_iterable'])`
/// so anything beyond `(self, iterable)` raises TypeError; pyre enforces the
/// same maxargs explicitly here.
fn set_descr_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let set_obj = crate::type_methods::require_set_receiver(args, "__init__", false)?;
    // setobject.py `descr_init(self, space, w_iterable=None, __posonly__=None)`
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

    // setobject.py `_initialize_set` — `w_obj.clear()` drops the
    // storage in one go, then the iterable populates it when it is not None
    // (the parsed default).
    unsafe { pyre_object::w_set_clear(set_obj) };
    if !w_iterable.is_null() && !unsafe { pyre_object::is_none(w_iterable) } {
        set_init_from_iterable(set_obj, w_iterable)?;
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

fn init_list_type(ns: PyObjectRef) {
    // listobject.py W_ListObject.typedef, kept in source order.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new(
                "Built-in mutable sequence.\n\nIf no argument is given, the constructor creates a new empty list.\nThe argument must be an iterable if specified.",
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(list_descr_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__init__",
            make_builtin_function("__init__", list_descr_init),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    let list = crate::type_methods::require_list_receiver(args, "__repr__", false)?;
                    Ok(w_str_new(&unsafe { crate::display::list_repr(list)? }))
                },
                1,
            ),
        )
    };
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, "__hash__", w_none()) };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__sizeof__",
            make_builtin_function_with_arity(
                "__sizeof__",
                |args| {
                    let list =
                        crate::type_methods::require_list_receiver(args, "__sizeof__", true)?;
                    // CPython 3.14's PyListObject header is five machine
                    // words; the item array contributes one pointer per
                    // allocated slot. This is the version oracle where PyPy
                    // does not expose a list-specific descriptor.
                    let size = 5 * std::mem::size_of::<usize>()
                        + unsafe { pyre_object::w_list_capacity(list) }
                            * std::mem::size_of::<PyObjectRef>();
                    Ok(w_int_new(size as i64))
                },
                1,
            ),
        )
    };
    // listobject.py:2486 __class_getitem__ = interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                crate::_pypy_generic_alias::generic_alias_class_getitem,
            )),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "append",
            make_builtin_function_with_arity("append", crate::type_methods::list_method_append, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "extend",
            make_builtin_function_with_arity("extend", crate::type_methods::list_method_extend, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "copy",
            make_builtin_function_with_arity("copy", crate::type_methods::list_method_copy, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "insert",
            make_builtin_function_with_arity("insert", crate::type_methods::list_method_insert, 3),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "pop",
            make_builtin_function("pop", crate::type_methods::list_method_pop),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "clear",
            make_builtin_function_with_arity("clear", crate::type_methods::list_method_clear, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "reverse",
            make_builtin_function_with_arity(
                "reverse",
                crate::type_methods::list_method_reverse,
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "sort",
            make_builtin_function("sort", crate::type_methods::list_method_sort),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "index",
            make_builtin_function("index", crate::type_methods::list_method_index),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "count",
            make_builtin_function_with_arity("count", crate::type_methods::list_method_count, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "remove",
            make_builtin_function_with_arity("remove", crate::type_methods::list_method_remove, 2),
        )
    };
    // Container slots exposed as callable dunders.  `__getitem__` binds the
    // direct slot body (`getitem_slot`) rather than the operator entry, so a
    // subclass override's `super().__getitem__` reaches the inherited builtin
    // subscript instead of re-entering override dispatch.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__getitem__",
            make_builtin_function_with_arity(
                "__getitem__",
                |args| {
                    crate::type_methods::require_list_receiver(args, "__getitem__", true)?;
                    crate::type_methods::arity_exact(args, "list.__getitem__", 1)?;
                    crate::baseobjspace::getitem_slot(args[0], args[1])
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__setitem__",
            make_builtin_function_with_arity(
                "__setitem__",
                |args| {
                    crate::type_methods::require_list_receiver(args, "__setitem__", false)?;
                    crate::type_methods::arity_exact_unpack(args, "__setitem__", 2)?;
                    crate::baseobjspace::setitem_slot(args[0], args[1], args[2])?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__delitem__",
            make_builtin_function_with_arity(
                "__delitem__",
                |args| {
                    crate::type_methods::require_list_receiver(args, "__delitem__", false)?;
                    crate::type_methods::arity_slot(args, 1)?;
                    crate::baseobjspace::delitem_slot(args[0], args[1])?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__len__",
            make_builtin_function_with_arity(
                "__len__",
                |args| {
                    crate::type_methods::require_list_receiver(args, "__len__", false)?;
                    crate::type_methods::arity_slot(args, 0)?;
                    crate::baseobjspace::len_slot(args[0])
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__contains__",
            make_builtin_function_with_arity(
                "__contains__",
                |args| {
                    crate::type_methods::require_list_receiver(args, "__contains__", false)?;
                    crate::type_methods::arity_slot(args, 1)?;
                    let found = crate::baseobjspace::contains_slot(args[0], args[1])?;
                    Ok(pyre_object::w_bool_from(found))
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__iter__",
            // Build the storage iterator directly rather than re-entering
            // `space.iter()` — a `list` subclass that calls `super().__iter__()`
            // would otherwise be re-dispatched back to its own override.
            make_builtin_function_with_arity(
                "__iter__",
                |args| {
                    let obj = crate::type_methods::require_list_receiver(args, "__iter__", false)?;
                    crate::type_methods::arity_slot(args, 0)?;
                    Ok(pyre_object::w_list_iter_new(obj))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reversed__",
            make_builtin_function_with_arity(
                "__reversed__",
                |args| {
                    // `listobject.py:737 descr_reversed` — a lazy reverse iterator
                    // over the list, the same `W_ReversedIterator` representation as
                    // `reversed(list)` (walks `getitem(seq, remaining)` downward).
                    let obj =
                        crate::type_methods::require_list_receiver(args, "__reversed__", true)?;
                    crate::type_methods::arity_no_args(args, "list.__reversed__")?;
                    let n = unsafe { pyre_object::w_list_len(obj) } as i64;
                    Ok(pyre_object::w_list_reverse_iter_new(obj, n - 1))
                },
                1,
            ),
        )
    };
    // Arithmetic slots.  `listobject.py:627 descr_add` returns NotImplemented
    // for a non-list operand (so the `+` operator's reflected dispatch runs and
    // a generic "unsupported operand type(s)" TypeError is raised);
    // `list_repeat` requires an integer count.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__add__",
            make_builtin_function_with_arity(
                "__add__",
                |args| {
                    crate::type_methods::require_list_receiver(args, "__add__", false)?;
                    crate::type_methods::arity_slot(args, 1)?;
                    if unsafe { pyre_object::is_list(args[1]) } {
                        unsafe { crate::objspace::descroperation::list_concat(args[0], args[1]) }
                    } else {
                        Ok(pyre_object::w_not_implemented())
                    }
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__mul__",
            make_builtin_function_with_arity("__mul__", list_descr_mul, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__rmul__",
            make_builtin_function_with_arity("__rmul__", list_descr_rmul, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__iadd__",
            make_builtin_function_with_arity(
                "__iadd__",
                |args| {
                    crate::type_methods::require_list_receiver(args, "__iadd__", false)?;
                    // Validate the slot arity before delegating so a bad call
                    // reports "expected 1 argument, got M" rather than the
                    // `extend` message surfaced by the shared implementation.
                    crate::type_methods::arity_slot(args, 1)?;
                    crate::type_methods::list_method_extend(args)?;
                    Ok(args[0])
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__imul__",
            make_builtin_function_with_arity(
                "__imul__",
                |args| {
                    crate::type_methods::require_list_receiver(args, "__imul__", false)?;
                    // listobject.py descr_inplace_mul: the count goes through
                    // `__index__`; a non-index operand becomes NotImplemented.
                    crate::type_methods::arity_slot(args, 1)?;
                    let Some(w_count) = list_repeat_index(args[1])? else {
                        return Ok(pyre_object::w_not_implemented());
                    };
                    unsafe {
                        crate::objspace::descroperation::list_inplace_repeat(args[0], w_count)?
                    };
                    Ok(args[0])
                },
                2,
            ),
        )
    };
    for (name, func) in [
        ("__eq__", list_dunder_eq as DunderFn),
        ("__ne__", list_dunder_ne),
        ("__lt__", list_dunder_lt),
        ("__le__", list_dunder_le),
        ("__gt__", list_dunder_gt),
        ("__ge__", list_dunder_ge),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
}

/// `space.getindex_w(w_obj, space.w_OverflowError)`: run `w_obj.__index__()`
/// and return the resulting int/long object, converting an out-of-index-range
/// value to an `OverflowError` that names the ORIGINAL operand (not the
/// `__index__` result). A non-`__index__` operand raises the `TypeError` from
/// `space.index`. Callers that repeat a sequence share this so `str`/`tuple`
/// honour a custom `__index__` exactly like `list`/`bytes`.
fn getindex_repeat(w_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let w_count = crate::baseobjspace::space_index(w_obj)?;
    match crate::baseobjspace::int_w(w_count) {
        Ok(_) => Ok(w_count),
        Err(e) if e.kind == crate::PyErrorKind::OverflowError => Err(crate::PyError::new(
            crate::PyErrorKind::OverflowError,
            format!(
                "cannot fit '{}' into an index-sized integer",
                crate::baseobjspace::object_functionstr_type_name(w_obj)
            ),
        )),
        Err(e) => Err(e),
    }
}

/// Coerce a `list`/`tuple` `* n` / `*= n` repeat count through `getindex_repeat`
/// (`getindex_w`). An operand without `__index__` yields `None`, which the
/// caller maps to NotImplemented so the `*`/`*=` operator can try a reflected
/// `__rmul__` and otherwise emit the "can't multiply sequence by non-int"
/// message; any other coercion error propagates. This is `descr_mul`'s
/// `try/except TypeError -> NotImplemented` wrapper (`listobject.py`).
fn list_repeat_index(w_obj: PyObjectRef) -> Result<Option<PyObjectRef>, crate::PyError> {
    match getindex_repeat(w_obj) {
        Ok(w_count) => Ok(Some(w_count)),
        Err(e) if e.kind == crate::PyErrorKind::TypeError => Ok(None),
        Err(e) => Err(e),
    }
}

/// `listobject.c:list_repeat` — `list * n` / `n * list`.  The count goes
/// through `__index__`, so any object implementing it repeats the list.
fn list_descr_mul(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    list_descr_mul_impl(args, "__mul__")
}

fn list_descr_rmul(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    list_descr_mul_impl(args, "__rmul__")
}

fn list_descr_mul_impl(args: &[PyObjectRef], name: &str) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_list_receiver(args, name, false)?;
    crate::type_methods::arity_slot(args, 1)?;
    let Some(w_count) = list_repeat_index(args[1])? else {
        return Ok(pyre_object::w_not_implemented());
    };
    unsafe { crate::objspace::descroperation::list_repeat(args[0], w_count) }
}

// ── Str TypeDef ──────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/unicodeobject.py TypeDef("str", ...)

fn str_descr_mul(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_slot(args, 1)?;
    // unicodeobject.py descr_mul = getindex_w (no NotImplemented wrapper),
    // so a non-__index__ operand raises and a custom __index__ is honoured.
    let w_count = if unsafe { pyre_object::pyobject::is_int_or_long(args[1]) } {
        args[1]
    } else {
        getindex_repeat(args[1])?
    };
    unsafe { crate::objspace::descroperation::str_repeat(args[0], w_count) }
}

fn str_descr_rmod(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_slot(args, 1)?;
    // unicodeobject.py:439-443 — only a unicode left operand reaches the
    // formatter; every other direct invocation returns NotImplemented.
    if !unsafe { pyre_object::is_str(args[1]) } {
        return Ok(pyre_object::w_not_implemented());
    }
    crate::baseobjspace::mod_(args[1], args[0])
}

fn init_str_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new(
                "str(object='') -> str\n\
                 str(bytes_or_buffer[, encoding[, errors]]) -> str\n\n\
                 Create a new string object from the given object. If encoding or\n\
                 errors is specified, then the object must expose a data buffer\n\
                 that will be decoded using the given encoding and error handler.\n\
                 Otherwise, returns the result of object.__str__() (if defined)\n\
                 or repr(object).\n\
                 encoding defaults to 'utf-8'.\n\
                 errors defaults to 'strict'.",
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(str_descr_new),
        )
    };
    // unicodeobject.py:330-338 descr_repr / descr_str.  descr_str returns an
    // exact base str for a subtype, which is required by enum.StrEnum's
    // inherited `str.__str__`.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    Ok(pyre_object::w_str_new(&crate::display::format_wtf8_repr(
                        unsafe { pyre_object::w_str_get_wtf8(args[0]) },
                    )))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__str__",
            make_builtin_function_with_arity(
                "__str__",
                |args| {
                    let obj = args[0];
                    if unsafe { pyre_object::pyobject::is_exact_type(obj, &pyre_object::STR_TYPE) }
                    {
                        Ok(obj)
                    } else {
                        Ok(pyre_object::w_str_from_wtf8(
                            unsafe { pyre_object::w_str_get_wtf8(obj) }.to_wtf8_buf(),
                        ))
                    }
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| Ok(w_int_new(crate::builtins::hash_value(args[0]))),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__sizeof__",
            make_builtin_function_with_arity(
                "__sizeof__",
                |args| {
                    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
                    let mut maxchar = 0u32;
                    let mut length = 0usize;
                    for cp in s.code_points() {
                        maxchar = maxchar.max(cp.to_u32());
                        length += 1;
                    }
                    // CPython's compact PEP 393 layout, exposed for compatibility
                    // with test_str.test_raiseMemError.  PyPy documents
                    // `__sizeof__` on str but sys.getsizeof itself remains a
                    // default-returning operation for other objects.
                    let word = std::mem::size_of::<usize>();
                    let struct_size = if maxchar < 0x80 { 5 * word } else { 7 * word };
                    let char_size = if maxchar < 0x100 {
                        1
                    } else if maxchar < 0x10000 {
                        2
                    } else {
                        4
                    };
                    Ok(pyre_object::w_int_new(
                        (struct_size + char_size * (length + 1)) as i64,
                    ))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__format__",
            make_builtin_function_with_arity(
                "__format__",
                crate::type_methods::builtin_value_format,
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "join",
            make_builtin_function_with_arity("join", crate::type_methods::str_method_join, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "split",
            make_builtin_function("split", crate::type_methods::str_method_split),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rsplit",
            make_builtin_function("rsplit", crate::type_methods::str_method_rsplit),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "splitlines",
            make_builtin_function("splitlines", crate::type_methods::str_method_splitlines),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "partition",
            make_builtin_function("partition", crate::type_methods::str_method_partition),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rpartition",
            make_builtin_function("rpartition", crate::type_methods::str_method_rpartition),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "zfill",
            make_builtin_function("zfill", crate::type_methods::str_method_zfill),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "casefold",
            make_builtin_function("casefold", crate::type_methods::str_method_casefold),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "swapcase",
            make_builtin_function("swapcase", crate::type_methods::str_method_swapcase),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "expandtabs",
            make_builtin_function("expandtabs", crate::type_methods::str_method_expandtabs),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "format_map",
            make_builtin_function("format_map", crate::type_methods::str_method_format_map),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "strip",
            make_builtin_function("strip", crate::type_methods::str_method_strip),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "lstrip",
            make_builtin_function("lstrip", crate::type_methods::str_method_lstrip),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rstrip",
            make_builtin_function("rstrip", crate::type_methods::str_method_rstrip),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "startswith",
            make_builtin_function("startswith", crate::type_methods::str_method_startswith),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "endswith",
            make_builtin_function("endswith", crate::type_methods::str_method_endswith),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "replace",
            make_builtin_function("replace", crate::type_methods::str_method_replace),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "find",
            make_builtin_function("find", crate::type_methods::str_method_find),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rfind",
            make_builtin_function("rfind", crate::type_methods::str_method_rfind),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rindex",
            make_builtin_function("rindex", crate::type_methods::str_method_rindex),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "upper",
            make_builtin_function_with_arity("upper", crate::type_methods::str_method_upper, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "lower",
            make_builtin_function_with_arity("lower", crate::type_methods::str_method_lower, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "format",
            make_builtin_function("format", crate::type_methods::str_method_format),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "encode",
            make_builtin_function("encode", crate::type_methods::str_method_encode),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isdigit",
            make_builtin_function_with_arity("isdigit", crate::type_methods::str_method_isdigit, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isdecimal",
            make_builtin_function_with_arity(
                "isdecimal",
                crate::type_methods::str_method_isdecimal,
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isnumeric",
            make_builtin_function_with_arity(
                "isnumeric",
                crate::type_methods::str_method_isnumeric,
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "istitle",
            make_builtin_function_with_arity("istitle", crate::type_methods::str_method_istitle, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isalpha",
            make_builtin_function_with_arity("isalpha", crate::type_methods::str_method_isalpha, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isidentifier",
            make_builtin_function_with_arity(
                "isidentifier",
                crate::type_methods::str_method_isidentifier,
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "zfill",
            make_builtin_function_with_arity("zfill", crate::type_methods::str_method_zfill, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "count",
            make_builtin_function("count", crate::type_methods::str_method_count),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "index",
            make_builtin_function("index", crate::type_methods::str_method_index),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "title",
            make_builtin_function_with_arity("title", crate::type_methods::str_method_title, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "capitalize",
            make_builtin_function_with_arity(
                "capitalize",
                crate::type_methods::str_method_capitalize,
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "swapcase",
            make_builtin_function_with_arity(
                "swapcase",
                crate::type_methods::str_method_swapcase,
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "center",
            make_builtin_function("center", crate::type_methods::str_method_center),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "ljust",
            make_builtin_function("ljust", crate::type_methods::str_method_ljust),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rjust",
            make_builtin_function("rjust", crate::type_methods::str_method_rjust),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isspace",
            make_builtin_function_with_arity("isspace", crate::type_methods::str_method_isspace, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isprintable",
            make_builtin_function_with_arity(
                "isprintable",
                crate::type_methods::str_method_isprintable,
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isupper",
            make_builtin_function_with_arity("isupper", crate::type_methods::str_method_isupper, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "islower",
            make_builtin_function_with_arity("islower", crate::type_methods::str_method_islower, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isalnum",
            make_builtin_function_with_arity("isalnum", crate::type_methods::str_method_isalnum, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isascii",
            make_builtin_function_with_arity("isascii", crate::type_methods::str_method_isascii, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "partition",
            make_builtin_function_with_arity(
                "partition",
                crate::type_methods::str_method_partition,
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rpartition",
            make_builtin_function_with_arity(
                "rpartition",
                crate::type_methods::str_method_rpartition,
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "splitlines",
            make_builtin_function("splitlines", crate::type_methods::str_method_splitlines),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "removeprefix",
            make_builtin_function_with_arity(
                "removeprefix",
                crate::type_methods::str_method_removeprefix,
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "removesuffix",
            make_builtin_function_with_arity(
                "removesuffix",
                crate::type_methods::str_method_removesuffix,
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "expandtabs",
            make_builtin_function("expandtabs", crate::type_methods::str_method_expandtabs),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "translate",
            make_builtin_function_with_arity(
                "translate",
                crate::type_methods::str_method_translate,
                2,
            ),
        )
    };
    // str dunder methods
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__mul__",
            make_builtin_function_with_arity("__mul__", str_descr_mul, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__rmul__",
            make_builtin_function_with_arity("__rmul__", str_descr_mul, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__rmod__",
            make_builtin_function_with_arity("__rmod__", str_descr_rmod, 2),
        )
    };
    // maketrans — PyPy: unicodeobject.py descr_maketrans
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    for (name, func) in [
        ("__eq__", str_dunder_eq as DunderFn),
        ("__ne__", str_dunder_ne),
        ("__lt__", str_dunder_lt),
        ("__le__", str_dunder_le),
        ("__gt__", str_dunder_gt),
        ("__ge__", str_dunder_ge),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
    // unicodeobject.py descr_getnewargs — `(W_UnicodeObject(self._utf8),)`:
    // a fresh plain str from the contents, so a str subclass reduces to str.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
}

// ── Dict TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/dictmultiobject.py TypeDef("dict", ...)

fn init_dict_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new(
                "dict() -> new empty dictionary\n\
                 dict(mapping) -> new dictionary initialized from a mapping object's\n\
                     (key, value) pairs\n\
                 dict(iterable) -> new dictionary initialized as if via:\n\
                     d = {}\n\
                     for k, v in iterable:\n\
                         d[k] = v\n\
                 dict(**kwargs) -> new dictionary initialized with the name=value pairs\n\
                     in the keyword argument list.  For example:  dict(one=1, two=2)",
            ),
        )
    };
    // dictmultiobject.py:421 `__hash__ = None`.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, "__hash__", w_none()) };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(dict_descr_new),
        )
    };
    // dictmultiobject.py:446 __class_getitem__ = interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                crate::_pypy_generic_alias::generic_alias_class_getitem,
            )),
        )
    };
    // `dictmultiobject.py:137-138 descr_init` →
    // `init_or_update(space, self, __args__, 'dict')`
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__init__",
            make_builtin_function("__init__", |args| {
                crate::type_methods::dict_init_or_update(args, "dict")
            }),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "get",
            make_builtin_function("get", crate::type_methods::dict_method_get),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "keys",
            make_builtin_function_with_arity("keys", crate::type_methods::dict_method_keys, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "values",
            make_builtin_function_with_arity("values", crate::type_methods::dict_method_values, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "items",
            make_builtin_function_with_arity("items", crate::type_methods::dict_method_items, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "update",
            make_builtin_function("update", crate::type_methods::dict_method_update),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "pop",
            make_builtin_function("pop", crate::type_methods::dict_method_pop),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "popitem",
            make_builtin_function_with_arity(
                "popitem",
                crate::type_methods::dict_method_popitem,
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "setdefault",
            make_builtin_function("setdefault", crate::type_methods::dict_method_setdefault),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                                    crate::type_methods::dict_store_checked(
                                        backing, args[1], args[2],
                                    )?;
                                }
                            }
                        }
                    }
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                                        Err(_) => {
                                            Err(crate::baseobjspace::take_pending_dict_key_error(
                                                args[1],
                                            ))
                                        }
                                    };
                                }
                            }
                        }
                    }
                    crate::baseobjspace::getitem(args[0], args[1])
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                            Err(_) => {
                                Err(crate::baseobjspace::take_pending_dict_key_error(args[1]))
                            }
                        };
                    }
                    Ok(pyre_object::w_bool_from(
                        crate::baseobjspace::contains_slot(args[0], args[1])?,
                    ))
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                        let tp_name = unsafe { pyre_object::type_name_of(recv) };
                        return Err(crate::PyError::type_error(format!(
                            "descriptor '__repr__' for 'dict' objects \
                         doesn't apply to a '{tp_name}' object"
                        )));
                    }
                    unsafe { Ok(pyre_object::w_str_new(&crate::display::dict_repr(dict)?)) }
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ne__",
            make_builtin_function_with_arity(
                "__ne__",
                |args| {
                    crate::type_methods::arity_slot(args, 1)?;
                    let a = crate::type_methods::resolve_dict_backing(args[0]);
                    let b = crate::type_methods::resolve_dict_backing(args[1]);
                    if a.is_null() || b.is_null() {
                        return Ok(pyre_object::w_not_implemented());
                    }
                    let eq =
                        crate::baseobjspace::compare(a, b, crate::baseobjspace::CompareOp::Eq)?;
                    if unsafe { pyre_object::is_not_implemented(eq) } {
                        return Ok(eq);
                    }
                    Ok(pyre_object::w_bool_from(!crate::baseobjspace::is_true(eq)?))
                },
                2,
            ),
        )
    };
    // CPython 3.14 keeps the inherited object ordering descriptors visible in
    // dict.__dict__; like objectobject.py's generic rich comparison they
    // return NotImplemented.
    for name in ["__lt__", "__le__", "__gt__", "__ge__"] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, |_| Ok(w_not_implemented()), 2),
            )
        };
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__sizeof__",
            make_builtin_function_with_arity(
                "__sizeof__",
                |args| {
                    crate::type_methods::arity_slot(args, 0)?;
                    let backing = crate::type_methods::resolve_dict_backing(args[0]);
                    if backing.is_null() {
                        return Err(crate::PyError::type_error(
                            "descriptor '__sizeof__' for 'dict' objects doesn't apply",
                        ));
                    }
                    let len = unsafe { pyre_object::w_dict_len(backing) };
                    // W_DictObject plus its strategy/storage bookkeeping and
                    // the stored hash/key/value lane for each live entry.
                    let size = pyre_object::dictmultiobject::W_DICT_OBJECT_SIZE
                        + std::mem::size_of::<usize>()
                        + len * 3 * std::mem::size_of::<usize>();
                    Ok(w_int_new(size as i64))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                    // RHS — both reads go through `w_dict_items`, matching
                    // PyPy's storage-strategy delitem/iter parity.
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reversed__",
            make_builtin_function_with_arity(
                "__reversed__",
                |args| {
                    // `dictmultiobject.py:207 descr_reversed`:
                    // `strategy.w_iterreversed(self)` returns the live
                    // `W_DictMultiIterKeysReversedObject` cursor.
                    dict_iterator_receiver(
                        args,
                        "__reversed__",
                        true,
                        "dict",
                        &pyre_object::DICT_TYPE,
                        true,
                    )?;
                    let d = crate::type_methods::resolve_dict_backing(args[0]);
                    Ok(
                        pyre_object::dictmultiobject::w_dict_view_reverse_iterator_new(
                            d,
                            pyre_object::dictmultiobject::DictViewKind::Keys,
                        ),
                    )
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "copy",
            make_builtin_function_with_arity("copy", crate::type_methods::dict_method_copy, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                    crate::type_methods::arity_no_args(args, "dict.clear")?;
                    let d = crate::type_methods::resolve_dict_backing(args[0]);
                    if !d.is_null() {
                        unsafe { pyre_object::dictmultiobject::w_dict_clear(d) };
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        )
    };
    // dict.fromkeys(iterable, value=None) — classmethod
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "fromkeys",
            pyre_object::function::w_classmethod_new(make_builtin_function("fromkeys", |args| {
                // classmethod: args[0] is the bound cls; the user arguments are
                // fromkeys(iterable, value=None).
                let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                crate::type_methods::arity_at_most(args, "fromkeys", 2)?;
                let (iterable, value) = if args.len() >= 3 {
                    (args[1], args[2])
                } else if args.len() == 2 {
                    (args[1], pyre_object::w_none())
                } else {
                    return Err(crate::PyError::type_error(
                        "fromkeys expected at least 1 argument, got 0",
                    ));
                };
                // dictmultiobject.py:120-134 descr_fromkeys — for `dict` itself,
                // fill a fresh dict through the dict's own setitem, which hashes
                // each key; for a dict subclass, construct an instance via `cls()`
                // and route through `space.setitem` so the result is an instance
                // of the subclass.
                let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
                if cls.is_null() || crate::baseobjspace::is_w(cls, w_dict_type) {
                    let d = pyre_object::w_dict_new();
                    // Python 3.14's exact-set/frozenset fast path carries each
                    // entry's cached hash into the new exact dict.  This is the
                    // reverse of `set_update_dict_lock_held` and avoids a
                    // second observable `__hash__` call.  Subclasses still go
                    // through their iterator below.
                    if unsafe {
                        pyre_object::is_exact_type(iterable, &pyre_object::setobject::SET_TYPE)
                            || pyre_object::is_exact_type(
                                iterable,
                                &pyre_object::setobject::FROZENSET_TYPE,
                            )
                    } {
                        let _roots = pyre_object::gc_roots::push_roots();
                        let sp = pyre_object::gc_roots::shadow_stack_len();
                        pyre_object::gc_roots::pin_root(d);
                        pyre_object::gc_roots::pin_root(value);
                        pyre_object::gc_roots::pin_root(iterable);
                        let mut index = 0usize;
                        loop {
                            let iterable = pyre_object::gc_roots::shadow_stack_get(sp + 2);
                            let Some(key) = (unsafe { pyre_object::w_set_key_at(iterable, index) })
                            else {
                                break;
                            };
                            let d = pyre_object::gc_roots::shadow_stack_get(sp);
                            let value = pyre_object::gc_roots::shadow_stack_get(sp + 1);
                            unsafe {
                                pyre_object::w_dict_store_hashed_checked(
                                    d, key.obj, value, key.hash,
                                )
                            }
                            .map_err(|_| {
                                crate::baseobjspace::take_pending_dict_key_error(key.obj)
                            })?;
                            index += 1;
                        }
                        return Ok(pyre_object::gc_roots::shadow_stack_get(sp));
                    }
                    let items = crate::builtins::collect_iterable(iterable)?;
                    // `try_hash_value` may run a user `__hash__` that allocates
                    // and triggers a moving minor collection; `d`, the shared
                    // `value` (reused across every key), and every not-yet-added
                    // key are rooted for the whole loop and reloaded after each
                    // hash.
                    let d = unsafe {
                        let _roots = pyre_object::gc_roots::push_roots();
                        let sp = pyre_object::gc_roots::shadow_stack_len();
                        pyre_object::gc_roots::pin_root(d);
                        pyre_object::gc_roots::pin_root(value);
                        let key_base = sp + 2;
                        for key in items {
                            pyre_object::gc_roots::pin_root(key);
                        }
                        let key_len = pyre_object::gc_roots::shadow_stack_len() - key_base;
                        for i in 0..key_len {
                            let key = pyre_object::gc_roots::shadow_stack_get(key_base + i);
                            let hash = crate::builtins::try_hash_value(key).map_err(|err| {
                                crate::baseobjspace::wrap_dict_key_hash_error(key, err)
                            })?;
                            let d = pyre_object::gc_roots::shadow_stack_get(sp);
                            let key = pyre_object::gc_roots::shadow_stack_get(key_base + i);
                            let value = pyre_object::gc_roots::shadow_stack_get(sp + 1);
                            pyre_object::w_dict_store_hashed_checked(d, key, value, hash).map_err(
                                |_| crate::baseobjspace::take_pending_dict_key_error(key),
                            )?;
                        }
                        pyre_object::gc_roots::shadow_stack_get(sp)
                    };
                    Ok(d)
                } else {
                    let items = crate::builtins::collect_iterable(iterable)?;
                    let d = crate::call::call_function_impl_result(cls, &[])?;
                    for key in items {
                        crate::baseobjspace::setitem(d, key, value)?;
                    }
                    Ok(d)
                }
            })),
        )
    };
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
fn dict_view_reversed(
    args: &[PyObjectRef],
    owner: &str,
    expected: &'static PyType,
    kind: pyre_object::dictmultiobject::DictViewKind,
) -> crate::PyResult {
    let view = dict_iterator_receiver(args, "__reversed__", true, owner, expected, false)?;
    let dict = unsafe { pyre_object::dictmultiobject::w_dict_view_get_dict(view) };
    Ok(pyre_object::dictmultiobject::w_dict_view_reverse_iterator_new(dict, kind))
}

fn dict_keys_reversed(args: &[PyObjectRef]) -> crate::PyResult {
    dict_view_reversed(
        args,
        "dict_keys",
        &pyre_object::dictmultiobject::DICT_KEYS_TYPE,
        pyre_object::dictmultiobject::DictViewKind::Keys,
    )
}

fn dict_values_reversed(args: &[PyObjectRef]) -> crate::PyResult {
    dict_view_reversed(
        args,
        "dict_values",
        &pyre_object::dictmultiobject::DICT_VALUES_TYPE,
        pyre_object::dictmultiobject::DictViewKind::Values,
    )
}

fn dict_items_reversed(args: &[PyObjectRef]) -> crate::PyResult {
    dict_view_reversed(
        args,
        "dict_items",
        &pyre_object::dictmultiobject::DICT_ITEMS_TYPE,
        pyre_object::dictmultiobject::DictViewKind::Items,
    )
}

fn init_dict_view_common_slots(
    ns: PyObjectRef,
    reversed_fn: fn(&[PyObjectRef]) -> crate::PyResult,
) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__iter__",
            make_builtin_function_with_arity(
                "__iter__",
                |args| crate::baseobjspace::iter(args[0]),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__len__",
            make_builtin_function_with_arity(
                "__len__",
                |args| crate::baseobjspace::len_slot(args[0]),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reversed__",
            make_builtin_function_with_arity("__reversed__", reversed_fn, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "mapping",
            make_getset_property_named_doc(
                make_builtin_function_with_arity(
                    "mapping",
                    |args| {
                        let view = args[1];
                        if view.is_null() {
                            return Ok(pyre_object::w_none());
                        }
                        let dict =
                            unsafe { pyre_object::dictmultiobject::w_dict_view_get_dict(view) };
                        if dict.is_null() {
                            return Ok(pyre_object::w_dict_proxy_new(pyre_object::w_dict_new()));
                        }
                        Ok(pyre_object::w_dict_proxy_new(dict))
                    },
                    2,
                ),
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
                "dictionary that this view refers to",
                "mapping",
            ),
        )
    };
}

/// `dictmultiobject.py` `W_DictViewKeysObject` /
/// `W_DictViewItemsObject`
/// typedef body — common slots plus `__contains__` and the
/// SetLikeDictView surface (comparisons, set ops, isdisjoint).
fn init_dict_view_set_like_type(
    ns: PyObjectRef,
    reversed_fn: fn(&[PyObjectRef]) -> crate::PyResult,
) {
    init_dict_view_common_slots(ns, reversed_fn);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // The keys and items views are set-like and therefore unhashable
    // (`dictmultiobject.py:1626 _is_set_like`); the values view is not and
    // keeps `object.__hash__`.  Declare the slot `None` so a `hash()` finds it
    // and rejects the view instead of falling back to the identity hash.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            pyre_object::w_none(),
        )
    };
    register_dict_view_set_operators(ns);
}

/// `dictmultiobject.py:1831-1840 W_DictViewValuesObject.typedef` —
/// common slots only.  Values views are NOT set-like in PyPy
/// (`dictmultiobject.py:1619-1623 _is_set_like` excludes them) and
/// have no `__contains__` / set ops / comparisons of their own;
/// equality falls through to `object.__eq__`'s identity check.
fn init_dict_view_values_type(ns: PyObjectRef) {
    init_dict_view_common_slots(ns, dict_values_reversed);
}

fn init_dict_view_keys_type(ns: PyObjectRef) {
    init_dict_view_set_like_type(ns, dict_keys_reversed);
}

fn init_dict_view_items_type(ns: PyObjectRef) {
    init_dict_view_set_like_type(ns, dict_items_reversed);
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

fn init_pytraceback_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(traceback_descr_new),
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "tb_lasti",
            make_getset_property_named(
                lasti_getter,
                lasti_setter,
                pyre_object::PY_NULL,
                "tb_lasti",
            ),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "tb_lineno",
            make_getset_property_named(
                lineno_getter,
                lineno_setter,
                pyre_object::PY_NULL,
                "tb_lineno",
            ),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "tb_next",
            make_getset_property_named(next_getter, next_setter, pyre_object::PY_NULL, "tb_next"),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "tb_frame",
            make_getset_property_named(
                frame_getter,
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
                "tb_frame",
            ),
        )
    };
    // `pytraceback.py:99-101 descr__dir__` — returns the list of
    // public traceback attribute names.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
}

/// `pypy/interpreter/typedef.py:736-753 PyFrame.typedef` — the `frame`
/// type's getset descriptors + `clear` / `__repr__`.  The receiver
/// (`args[1]`) is a live `PyFrame` object (its `ob_header.ob_type` is
/// `FRAME_TYPE`); every field access casts it to `*mut PyFrame`.  A read
/// through a null / already-freed receiver returns `None` rather than
/// dereferencing.  `f_lineno`'s setter is [`PyFrame::fset_f_lineno`],
/// which validates the line-jump via `mark_stacks`; the read-only getsets
/// and `f_trace*` setters mirror `pyframe.py:641-806` directly.
fn init_frame_type(ns: PyObjectRef) {
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "f_code",
            make_getset_descriptor_named(code_getter, "f_code"),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "f_globals",
            make_getset_descriptor_named(globals_getter, "f_globals"),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "f_locals",
            make_getset_descriptor_named(locals_getter, "f_locals"),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "f_back",
            make_getset_descriptor_named(back_getter, "f_back"),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "f_lasti",
            make_getset_descriptor_named(lasti_getter, "f_lasti"),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "f_builtins",
            make_getset_descriptor_named(builtins_getter, "f_builtins"),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "f_lineno",
            make_getset_property_named(
                lineno_getter,
                lineno_setter,
                pyre_object::PY_NULL,
                "f_lineno",
            ),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "f_trace",
            make_getset_property_named(trace_getter, trace_setter, trace_deleter, "f_trace"),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "f_trace_lines",
            make_getset_property_named(
                trace_lines_getter,
                trace_lines_setter,
                pyre_object::PY_NULL,
                "f_trace_lines",
            ),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "f_trace_opcodes",
            make_getset_property_named(
                trace_opcodes_getter,
                trace_opcodes_setter,
                pyre_object::PY_NULL,
                "f_trace_opcodes",
            ),
        )
    };

    // clear() — interp2app (pyframe.py:805 descr_clear).
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };

    // __repr__ — interp2app (pyframe.py:849 descr_repr).
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
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

/// `dictmultiobject.py:1699-1710 _as_set_op` — build a set from the left
/// operand and run the named in-place set method against the right one.
///
/// Materialising the left operand through the set constructor is what
/// enforces the hash protocol: an unhashable element raises there rather
/// than being dropped or stored unhashed.
fn dict_view_as_set_op(
    lhs: pyre_object::PyObjectRef,
    rhs: pyre_object::PyObjectRef,
    methname: &str,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let w_set_type = crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE);
    let w_set = crate::call::call_function_impl_result(w_set_type, &[lhs])?;
    let method = crate::baseobjspace::getattr_str(w_set, methname)?;
    crate::call::call_function_impl_result(method, &[rhs])?;
    Ok(w_set)
}

/// `dictmultiobject.py:1701-1704 _as_set_op.op` — `set(self)` combined with
/// `w_other`.
fn dict_view_set_op(
    self_view: pyre_object::PyObjectRef,
    other: pyre_object::PyObjectRef,
    op_name: &str,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    dict_view_as_set_op(self_view, other, op_name)
}

/// `dictmultiobject.py:1705-1709 _as_set_op.rop` — the reflected shape builds
/// the set from `w_other` instead, so the non-commutative `-` and `&` keep
/// their operand order.
fn dict_view_rset_op(
    self_view: pyre_object::PyObjectRef,
    other: pyre_object::PyObjectRef,
    op_name: &str,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    dict_view_as_set_op(other, self_view, op_name)
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

fn register_dict_view_set_operators(ns: PyObjectRef) {
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
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
    // `dictmultiobject.py:1797 isdisjoint = interp2app(descr_isdisjoint)`
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isdisjoint",
            make_builtin_function_with_arity("isdisjoint", dict_view_descr_isdisjoint, 2),
        )
    };
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
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
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
        let tp = unsafe { pyre_object::type_name_of(w_mapping) };
        return Err(crate::PyError::type_error(format!(
            "mappingproxy() argument must be a mapping, not {tp}"
        )));
    }
    Ok(pyre_object::w_dict_proxy_new(w_mapping))
}

fn init_mappingproxy_type(ns: PyObjectRef) {
    // Python 3.14 `PyDictProxy_Type`: "Read-only proxy of a mapping."
    // PyPy's module doc spells out the same contract, while its TypeDef
    // predates the explicit type-doc slot.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            pyre_object::w_str_new("Read-only proxy of a mapping."),
        )
    };
    // dictproxyobject.py:105 __new__=interp2app(descr_new)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(mappingproxy_descr_new),
        )
    };
    // dictproxyobject.py:117 __class_getitem__ = interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                crate::_pypy_generic_alias::generic_alias_class_getitem,
            )),
        )
    };
    // dictproxyobject.py:32 descr_len → space.len(self.w_mapping)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__len__",
            make_builtin_function("__len__", |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                crate::type_methods::arity_slot(args, 0)?;
                crate::baseobjspace::len_slot(args[0])
            }),
        )
    };
    // dictproxyobject.py:35 descr_getitem → space.getitem(self.w_mapping, w_key)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__getitem__",
            make_builtin_function("__getitem__", |args| {
                crate::type_methods::arity_slot(args, 1)?;
                crate::baseobjspace::getitem(args[0], args[1])
            }),
        )
    };
    // dictproxyobject.py:38 descr_contains → space.contains(self.w_mapping, w_key)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__contains__",
            make_builtin_function("__contains__", |args| {
                crate::type_methods::arity_slot(args, 1)?;
                Ok(pyre_object::w_bool_from(crate::baseobjspace::contains(
                    args[0], args[1],
                )?))
            }),
        )
    };
    // dictproxyobject.py:41 descr_iter → space.iter(self.w_mapping)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__iter__",
            make_builtin_function("__iter__", |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                crate::baseobjspace::iter(args[0])
            }),
        )
    };
    // dictproxyobject.py:47 descr_repr →
    // `b"mappingproxy(%s)" % space.utf8_w(space.repr(self.w_mapping))`
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function("__repr__", |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_str_new("mappingproxy({})"));
                }
                unsafe { Ok(pyre_object::w_str_new(&crate::display::py_repr(args[0])?)) }
            }),
        )
    };
    // dictproxyobject.py:44 descr_str → space.str(self.w_mapping)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__str__",
            make_builtin_function("__str__", |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_str_new(""));
                }
                unsafe { Ok(pyre_object::w_str_new(&crate::display::py_str(args[0])?)) }
            }),
        )
    };
    // dictproxyobject.py:67 descr_ior → unconditional TypeError; the
    // proxy is read-only so in-place merge is rejected by name even
    // when the rhs would otherwise be acceptable for `__or__`.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ior__",
            make_builtin_function("__ior__", |_args| {
                Err(crate::PyError::type_error(
                    "'|=' is not supported by mappingproxy; use '|' instead",
                ))
            }),
        )
    };
    // Python 3.14 `mappingproxy_hash`: delegate to the wrapped mapping.
    // This is newer than PyPy's current `dictproxyobject.py` TypeDef. A
    // proxy around dict therefore raises `unhashable type: 'dict'`, while a
    // proxy around a custom hashable mapping returns that mapping's hash.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| {
                    let proxy = args[0];
                    if !unsafe { pyre_object::is_dict_proxy(proxy) } {
                        let received = unsafe { (*(*proxy).ob_type).name };
                        return Err(crate::PyError::type_error(format!(
                            "descriptor '__hash__' requires a 'mappingproxy' object but received a '{received}'"
                        )));
                    }
                    let mapping = unsafe { pyre_object::w_dict_proxy_get_mapping(proxy) };
                    Ok(pyre_object::w_int_new(crate::baseobjspace::hash_w_strict(
                        mapping,
                    )?))
                },
                1,
            ),
        )
    };
    // dictproxyobject.py:51 descr_or →
    // `copy_self.update(w_other); return copy_self`.  Implemented via
    // `dict_method_copy` (unwraps proxy through resolve_dict_backing)
    // followed by an items merge from `w_other`.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // dictproxyobject.py:60 descr_ror →
    // `space.call_method(w_other, '__or__', self.w_mapping)`.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // dictproxyobject.py:87 descr_reversed →
    // `space.call_method(self.w_mapping, '__reversed__')`.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reversed__",
            make_builtin_function("__reversed__", |args| {
                dict_iterator_receiver(
                    args,
                    "__reversed__",
                    true,
                    "mappingproxy",
                    &pyre_object::MAPPING_PROXY_TYPE,
                    false,
                )?;
                let dict = crate::type_methods::resolve_dict_backing(args[0]);
                Ok(
                    pyre_object::dictmultiobject::w_dict_view_reverse_iterator_new(
                        dict,
                        pyre_object::dictmultiobject::DictViewKind::Keys,
                    ),
                )
            }),
        )
    };
    // dictproxyobject.py:71 get_w / 75 keys_w / 78 values_w / 81 items_w /
    // 84 copy_w — call the wrapped mapping's method. The mappingproxy
    // constructor accepts any mapping, not only an exact W_DictObject, so
    // routing these through dict_method_* would cast e.g. OrderedDict's
    // subclass layout as a raw dict.
    fn forward_mapping_method(args: &[PyObjectRef], name: &str) -> crate::PyResult {
        let proxy = *args
            .first()
            .ok_or_else(|| crate::PyError::type_error("unbound mappingproxy method"))?;
        let mapping = unsafe {
            if pyre_object::is_dict_proxy(proxy) {
                pyre_object::w_dict_proxy_get_mapping(proxy)
            } else {
                proxy
            }
        };
        let method = crate::baseobjspace::getattr_str(mapping, name)?;
        crate::call::call_function_impl_result(method, &args[1..])
    }
    fn proxy_get(args: &[PyObjectRef]) -> crate::PyResult {
        forward_mapping_method(args, "get")
    }
    fn proxy_keys(args: &[PyObjectRef]) -> crate::PyResult {
        forward_mapping_method(args, "keys")
    }
    fn proxy_values(args: &[PyObjectRef]) -> crate::PyResult {
        forward_mapping_method(args, "values")
    }
    fn proxy_items(args: &[PyObjectRef]) -> crate::PyResult {
        forward_mapping_method(args, "items")
    }
    fn proxy_copy(args: &[PyObjectRef]) -> crate::PyResult {
        forward_mapping_method(args, "copy")
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "get",
            make_builtin_function("get", proxy_get),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "keys",
            make_builtin_function("keys", proxy_keys),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "values",
            make_builtin_function("values", proxy_values),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "items",
            make_builtin_function("items", proxy_items),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "copy",
            make_builtin_function("copy", proxy_copy),
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__eq__",
            make_builtin_function("__eq__", proxy_eq),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ne__",
            make_builtin_function("__ne__", proxy_ne),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__lt__",
            make_builtin_function("__lt__", proxy_lt),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__le__",
            make_builtin_function("__le__", proxy_le),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__gt__",
            make_builtin_function("__gt__", proxy_gt),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ge__",
            make_builtin_function("__ge__", proxy_ge),
        )
    };
}

// ── Tuple TypeDef ────────────────────────────────────────────────────

fn init_tuple_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new(
                "Built-in immutable sequence.\n\nIf no argument is given, the constructor returns an empty tuple.\nIf iterable is specified the tuple is initialized from iterable's items.\n\nIf the argument is a tuple, the return value is the same object.",
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(tuple_descr_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    let tuple =
                        crate::type_methods::require_tuple_receiver(args, "__repr__", false)?;
                    Ok(w_str_new(&unsafe { crate::display::tuple_repr(tuple)? }))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| {
                    let tuple =
                        crate::type_methods::require_tuple_receiver(args, "__hash__", false)?;
                    Ok(w_int_new(crate::builtins::try_hash_value(tuple)?))
                },
                1,
            ),
        )
    };
    // tupleobject.py:354 __class_getitem__ = interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                crate::_pypy_generic_alias::generic_alias_class_getitem,
            )),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "index",
            make_builtin_function("index", crate::type_methods::tuple_method_index),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "count",
            make_builtin_function_with_arity("count", crate::type_methods::tuple_method_count, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__contains__",
            make_builtin_function_with_arity(
                "__contains__",
                |args| {
                    crate::type_methods::require_tuple_receiver(args, "__contains__", false)?;
                    crate::type_methods::arity_slot(args, 1)?;
                    Ok(pyre_object::w_bool_from(
                        crate::baseobjspace::contains_slot(args[0], args[1])?,
                    ))
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__len__",
            make_builtin_function_with_arity(
                "__len__",
                |args| {
                    let tuple =
                        crate::type_methods::require_tuple_receiver(args, "__len__", false)?;
                    crate::type_methods::arity_slot(args, 0)?;
                    Ok(pyre_object::w_int_new(
                        unsafe { pyre_object::w_tuple_len(tuple) } as i64,
                    ))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__iter__",
            // Build the storage iterator directly rather than re-entering
            // `space.iter()` — a `tuple` subclass that calls `super().__iter__()`
            // would otherwise be re-dispatched back to its own override.
            make_builtin_function_with_arity(
                "__iter__",
                |args| {
                    let obj = crate::type_methods::require_tuple_receiver(args, "__iter__", false)?;
                    crate::type_methods::arity_slot(args, 0)?;
                    Ok(pyre_object::w_tuple_iter_new(obj))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__getitem__",
            make_builtin_function_with_arity(
                "__getitem__",
                |args| {
                    crate::type_methods::require_tuple_receiver(args, "__getitem__", false)?;
                    crate::type_methods::arity_slot(args, 1)?;
                    crate::baseobjspace::getitem_slot(args[0], args[1])
                },
                2,
            ),
        )
    };
    // `tupleobject.py:181 descr_add` returns NotImplemented for a non-tuple
    // operand (so the `+` operator's reflected dispatch runs and a generic
    // "unsupported operand type(s)" TypeError is raised); `*` requires an
    // integer count.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__add__",
            make_builtin_function_with_arity(
                "__add__",
                |args| {
                    crate::type_methods::require_tuple_receiver(args, "__add__", false)?;
                    crate::type_methods::arity_slot(args, 1)?;
                    if unsafe { pyre_object::is_tuple(args[1]) } {
                        unsafe { crate::objspace::descroperation::tuple_concat(args[0], args[1]) }
                    } else {
                        Ok(pyre_object::w_not_implemented())
                    }
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__mul__",
            make_builtin_function_with_arity("__mul__", tuple_descr_mul, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__rmul__",
            make_builtin_function_with_arity("__rmul__", tuple_descr_rmul, 2),
        )
    };
    for (name, func) in [
        ("__eq__", tuple_dunder_eq as DunderFn),
        ("__ne__", tuple_dunder_ne),
        ("__lt__", tuple_dunder_lt),
        ("__le__", tuple_dunder_le),
        ("__gt__", tuple_dunder_gt),
        ("__ge__", tuple_dunder_ge),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
    // tupleobject.py descr_getnewargs — `((self-copy),)`
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__getnewargs__",
            make_builtin_function_with_arity(
                "__getnewargs__",
                |args| {
                    let tuple =
                        crate::type_methods::require_tuple_receiver(args, "__getnewargs__", true)?;
                    let items = unsafe { pyre_object::w_tuple_items_copy_as_vec(tuple) };
                    Ok(pyre_object::w_tuple_new(vec![pyre_object::w_tuple_new(
                        items,
                    )]))
                },
                1,
            ),
        )
    };
}

/// `tupleobject.c` `tuple * n` / `n * tuple`.  A non-integer count
/// raises the `__index__` TypeError.
fn tuple_descr_mul(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    tuple_descr_mul_impl(args, "__mul__")
}

fn tuple_descr_rmul(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    tuple_descr_mul_impl(args, "__rmul__")
}

fn tuple_descr_mul_impl(args: &[PyObjectRef], name: &str) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_tuple_receiver(args, name, false)?;
    crate::type_methods::arity_slot(args, 1)?;
    // tupleobject descr_mul routes the count through getindex_w, so a custom
    // __index__ repeats the tuple (and an out-of-range one overflows).
    // A non-__index__ operand yields NotImplemented, letting the `*` operator
    // try a reflected `__rmul__` and otherwise emit "can't multiply sequence by
    // non-int", instead of this method's own slot error.
    let Some(w_count) = list_repeat_index(args[1])? else {
        return Ok(pyre_object::w_not_implemented());
    };
    crate::objspace::descroperation::mul(args[0], w_count)
}

// ── Int/Float/Bool TypeDef (minimal) ─────────────────────────────────

// ── Type TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/typeobject.py TypeDef("type", ...)

/// types.UnionType — PyPy: _pypy_generic_alias.py UnionType
fn slice_receiver(args: &[PyObjectRef], name: &str) -> Result<PyObjectRef, crate::PyError> {
    let self_ = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if self_.is_null() {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' of 'slice' object needs an argument"
        )));
    }
    if !unsafe { pyre_object::sliceobject::is_slice(self_) } {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' requires a 'slice' object but received a '{}'",
            type_name_of(self_),
        )));
    }
    Ok(self_)
}

/// sliceobject.py:148 `W_SliceObject.descr_indices`.
fn slice_eval_index_big(value: PyObjectRef) -> Result<malachite_bigint::BigInt, crate::PyError> {
    match crate::baseobjspace::space_index(value) {
        Ok(indexed) => Ok(unsafe { pyre_object::range_obj_to_bigint(indexed) }),
        Err(error) if error.kind == crate::PyErrorKind::TypeError => {
            Err(crate::PyError::type_error(
                "slice indices must be integers or None or have an __index__ method",
            ))
        }
        Err(error) => Err(error),
    }
}

fn slice_method_indices(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = slice_receiver(args, "indices")?;
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "slice.indices() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let roots = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(self_);
    pyre_object::gc_roots::pin_root(args[1]);
    // sliceobject.py app-level `indices`: unlike the machine-word
    // `indices3()` used by concrete sequence operations, this rarely-used
    // public method deliberately keeps start/stop/step/length unbounded.
    let indexed_length = crate::baseobjspace::space_index(unsafe {
        pyre_object::gc_roots::shadow_stack_get(roots + 1)
    })?;
    let length = unsafe { pyre_object::range_obj_to_bigint(indexed_length) };
    let zero = malachite_bigint::BigInt::from(0);
    let one = malachite_bigint::BigInt::from(1);
    if length < zero {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "length should not be negative".to_string(),
        ));
    }
    let w_step = unsafe {
        pyre_object::sliceobject::w_slice_get_step(pyre_object::gc_roots::shadow_stack_get(roots))
    };
    let step = if unsafe { pyre_object::is_none(w_step) } {
        one.clone()
    } else {
        let step = slice_eval_index_big(w_step)?;
        if step == zero {
            return Err(crate::PyError::value_error("slice step cannot be zero"));
        }
        step
    };
    let negative_step = step < zero;
    let lower = if negative_step {
        -one.clone()
    } else {
        zero.clone()
    };
    let upper = if negative_step {
        &length - &one
    } else {
        length.clone()
    };

    let w_start = unsafe {
        pyre_object::sliceobject::w_slice_get_start(pyre_object::gc_roots::shadow_stack_get(roots))
    };
    let start = if unsafe { pyre_object::is_none(w_start) } {
        if negative_step {
            upper.clone()
        } else {
            lower.clone()
        }
    } else {
        let mut start = slice_eval_index_big(w_start)?;
        if start < zero {
            start += &length;
            if start < lower {
                start = lower.clone();
            }
        } else if start > upper {
            start = upper.clone();
        }
        start
    };

    let w_stop = unsafe {
        pyre_object::sliceobject::w_slice_get_stop(pyre_object::gc_roots::shadow_stack_get(roots))
    };
    let stop = if unsafe { pyre_object::is_none(w_stop) } {
        if negative_step { lower } else { upper }
    } else {
        let mut stop = slice_eval_index_big(w_stop)?;
        if stop < zero {
            stop += &length;
            if stop < lower {
                stop = lower;
            }
        } else if stop > upper {
            stop = upper;
        }
        stop
    };

    let w_start = pyre_object::range_bigint_to_obj(start);
    pyre_object::gc_roots::pin_root(w_start);
    let w_stop = pyre_object::range_bigint_to_obj(stop);
    pyre_object::gc_roots::pin_root(w_stop);
    let w_step = pyre_object::range_bigint_to_obj(step);
    pyre_object::gc_roots::pin_root(w_step);
    Ok(w_tuple_new(vec![
        unsafe { pyre_object::gc_roots::shadow_stack_get(roots + 2) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(roots + 3) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(roots + 4) },
    ]))
}

/// sliceobject.py `W_SliceObject.descr__new__` — `slice([start,] stop[, step])`.
fn slice_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let slice_type = gettypefor(&pyre_object::sliceobject::SLICE_TYPE).unwrap_or(PY_NULL);
    check_user_subclass(slice_type, cls)?;
    let (params, kwargs) = crate::builtins::split_builtin_kwargs(args.get(1..).unwrap_or(&[]));
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
    let self_ = slice_receiver(args, "__repr__")?;
    Ok(w_str_new(&unsafe { crate::display::py_repr(self_)? }))
}

/// sliceobject.py `descr_eq` / `descr_ne` — compare the three components.
/// `slice is slice` is always equal even with non-comparable params.
fn slice_components_eq(a: PyObjectRef, b: PyObjectRef) -> Result<bool, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(a);
    let a_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(b);
    let b_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    Ok(
        slice_component_eq(a_slot, b_slot, pyre_object::sliceobject::w_slice_get_start)?
            && slice_component_eq(a_slot, b_slot, pyre_object::sliceobject::w_slice_get_stop)?
            && slice_component_eq(a_slot, b_slot, pyre_object::sliceobject::w_slice_get_step)?,
    )
}

fn slice_component_eq(
    a_slot: usize,
    b_slot: usize,
    field: unsafe fn(PyObjectRef) -> PyObjectRef,
) -> Result<bool, crate::PyError> {
    let (left, right) = unsafe {
        (
            field(pyre_object::gc_roots::shadow_stack_get(a_slot)),
            field(pyre_object::gc_roots::shadow_stack_get(b_slot)),
        )
    };
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(left);
    let left_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(right);
    let right_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    crate::baseobjspace::eq_w(
        unsafe { pyre_object::gc_roots::shadow_stack_get(left_slot) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(right_slot) },
    )
}

fn slice_components_tuple(s: PyObjectRef) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(s);
    let s_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let start = unsafe {
        pyre_object::sliceobject::w_slice_get_start(pyre_object::gc_roots::shadow_stack_get(s_slot))
    };
    pyre_object::gc_roots::pin_root(start);
    let start_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let stop = unsafe {
        pyre_object::sliceobject::w_slice_get_stop(pyre_object::gc_roots::shadow_stack_get(s_slot))
    };
    pyre_object::gc_roots::pin_root(stop);
    let stop_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let step = unsafe {
        pyre_object::sliceobject::w_slice_get_step(pyre_object::gc_roots::shadow_stack_get(s_slot))
    };
    pyre_object::gc_roots::pin_root(step);
    let step_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    w_tuple_new(vec![
        unsafe { pyre_object::gc_roots::shadow_stack_get(start_slot) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(stop_slot) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(step_slot) },
    ])
}

fn slice_descr_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (a, b) = (slice_receiver(args, "__eq__")?, args[1]);
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
    let (a, b) = (slice_receiver(args, "__ne__")?, args[1]);
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
    slice_descr_richcompare(args, crate::baseobjspace::CompareOp::Lt)
}

fn slice_descr_le(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    slice_descr_richcompare(args, crate::baseobjspace::CompareOp::Le)
}

fn slice_descr_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    slice_descr_richcompare(args, crate::baseobjspace::CompareOp::Gt)
}

fn slice_descr_ge(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    slice_descr_richcompare(args, crate::baseobjspace::CompareOp::Ge)
}

/// CPython 3.14 `slice_richcompare`: compare the packed
/// `(start, stop, step)` tuples with the original operation. PyPy 3.11 has
/// the equivalent tuple walk only for `<`; the 3.14 version oracle exposes
/// all four ordering slots.
fn slice_descr_richcompare(
    args: &[PyObjectRef],
    op: crate::baseobjspace::CompareOp,
) -> Result<PyObjectRef, crate::PyError> {
    let name = match op {
        crate::baseobjspace::CompareOp::Lt => "__lt__",
        crate::baseobjspace::CompareOp::Le => "__le__",
        crate::baseobjspace::CompareOp::Gt => "__gt__",
        crate::baseobjspace::CompareOp::Ge => "__ge__",
        _ => unreachable!(),
    };
    let (a, b) = (slice_receiver(args, name)?, args[1]);
    if !unsafe { pyre_object::sliceobject::is_slice(b) } {
        return Ok(pyre_object::w_not_implemented());
    }
    if a == b {
        return Ok(pyre_object::w_bool_from(matches!(
            op,
            crate::baseobjspace::CompareOp::Le | crate::baseobjspace::CompareOp::Ge
        )));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(a);
    let a_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(b);
    let b_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let left = slice_components_tuple(unsafe { pyre_object::gc_roots::shadow_stack_get(a_slot) });
    pyre_object::gc_roots::pin_root(left);
    let left_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let right = slice_components_tuple(unsafe { pyre_object::gc_roots::shadow_stack_get(b_slot) });
    pyre_object::gc_roots::pin_root(right);
    let right_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    crate::baseobjspace::compare(
        unsafe { pyre_object::gc_roots::shadow_stack_get(left_slot) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(right_slot) },
        op,
    )
}

/// CPython 3.14 `slice_hash`, copied from the tuplehash-style three-lane
/// mixer in `Objects/sliceobject.c`.
fn slice_descr_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_ = slice_receiver(args, "__hash__")?;
    Ok(w_int_new(crate::builtins::try_hash_value(self_)?))
}

/// sliceobject.py `descr__reduce__` — `(type(self), (start, stop, step))`.
fn slice_descr_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let s = slice_receiver(args, "__reduce__")?;
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(s);
    let s_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let ty = r#type(s).unwrap_or(pyre_object::PY_NULL);
    pyre_object::gc_roots::pin_root(ty);
    let ty_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let components =
        slice_components_tuple(unsafe { pyre_object::gc_roots::shadow_stack_get(s_slot) });
    pyre_object::gc_roots::pin_root(components);
    let components_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    Ok(w_tuple_new(vec![
        unsafe { pyre_object::gc_roots::shadow_stack_get(ty_slot) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(components_slot) },
    ]))
}

fn init_slice_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new(
                "slice(stop)\nslice(start, stop[, step])\n\nCreate a slice object. This is used for extended slicing (e.g. a[0:10:2]).",
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(slice_descr_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity("__repr__", slice_descr_repr, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__eq__",
            make_builtin_function_with_arity("__eq__", slice_descr_eq, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ne__",
            make_builtin_function_with_arity("__ne__", slice_descr_ne, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__lt__",
            make_builtin_function_with_arity("__lt__", slice_descr_lt, 2),
        )
    };
    for (name, func) in [
        ("__le__", slice_descr_le as DunderFn),
        ("__gt__", slice_descr_gt),
        ("__ge__", slice_descr_ge),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
    // PyPy 3.11 has `__hash__ = None`; CPython 3.14 made slices hashable.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity("__hash__", slice_descr_hash, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reduce__",
            make_builtin_function_with_arity("__reduce__", slice_descr_reduce, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "indices",
            make_builtin_function_with_arity("indices", slice_method_indices, 2),
        )
    };
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

fn init_union_type(ns: PyObjectRef) {
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__args__",
            make_getset_descriptor(args_getter),
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__parameters__",
            make_getset_descriptor(params_getter),
        )
    };
    // UnionType.__getitem__ (`_pypy_generic_alias.py:312`) — substitute the
    // free parameters with `items`, then fold the results back into a union
    // with `|`.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__getitem__",
            make_builtin_function("__getitem__", union_getitem),
        )
    };
    // UnionType.__class_getitem__ — `typing.Union` is this type, so
    // `Union[int, str]` folds members into a union.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                union_class_getitem,
            )),
        )
    };
    // UnionType.__or__ — PyPy: UnionType.__or__ → _create_union
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // UnionType.__ror__
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // UnionType.__eq__ — `set(self.__args__) == set(other.__args__)`
    // (`_pypy_generic_alias.py:270`).
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
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
fn init_getset_descriptor_type(ns: PyObjectRef) {
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                            return Err(getset_descr_mismatch(w_self, w_obj, reqcls));
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
                    Err(e) if e.kind == crate::PyErrorKind::DescrMismatch => {
                        Err(getset_descr_mismatch(w_self, w_obj, reqcls))
                    }
                    Err(e) => Err(e),
                }
            }),
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                                return Err(getset_descr_mismatch(w_self, w_obj, reqcls));
                            }
                            return Err(e);
                        }
                    }
                    match crate::call::call_function_impl_result(fset, &[w_self, w_obj, w_value]) {
                        Ok(_) => Ok(pyre_object::w_none()),
                        Err(e) if e.kind == crate::PyErrorKind::DescrMismatch => {
                            Err(getset_descr_mismatch(w_self, w_obj, reqcls))
                        }
                        Err(e) => Err(e),
                    }
                },
                3,
            ),
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                        let name =
                            if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
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
                                return Err(getset_descr_mismatch(w_self, w_obj, reqcls));
                            }
                            return Err(e);
                        }
                    }
                    match crate::call::call_function_impl_result(fdel, &[w_self, w_obj]) {
                        Ok(_) => Ok(pyre_object::w_none()),
                        Err(e) if e.kind == crate::PyErrorKind::DescrMismatch => {
                            Err(getset_descr_mismatch(w_self, w_obj, reqcls))
                        }
                        Err(e) => Err(e),
                    }
                },
                2,
            ),
        )
    };
    // CPython 3.14 `PyGetSetDescr_Type.tp_repr` renders
    // `<attribute 'name' of 'Owner' objects>`. PyPy's GetSetProperty typedef
    // has no explicit repr; this is the selected 3.14 surface.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    let descr = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                    if descr.is_null()
                        || !unsafe { pyre_object::typedef::is_getset_property(descr) }
                    {
                        let received = if descr.is_null() {
                            "NoneType".to_string()
                        } else {
                            crate::typedef::r#type(descr)
                                .map(|tp| unsafe { pyre_object::w_type_get_name(tp) }.to_string())
                                .unwrap_or_else(|| "object".to_string())
                        };
                        return Err(crate::PyError::type_error(format!(
                            "descriptor '__repr__' requires a 'getset_descriptor' object but received a '{received}'"
                        )));
                    }
                    Ok(pyre_object::w_str_new(&unsafe {
                        getset_descriptor_repr(descr)
                    }))
                },
                1,
            ),
        )
    };
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
    if !crate::type_dict_has_storage(tp) {
        return;
    }
    // typedef.py:470 __name__
    crate::type_dict_store(
        tp,
        "__name__",
        copy_for_type(
            make_getset_descriptor_named(
                make_builtin_function_with_arity(
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
                ),
                "__name__",
            ),
            tp,
        ),
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
    crate::type_dict_store(
        tp,
        "__qualname__",
        copy_for_type(
            make_getset_descriptor_named(
                make_builtin_function_with_arity(
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
                            // PyPy's original only consults `reqcls`. During type
                            // materialisation `copy_for_type` deliberately leaves
                            // reqcls null and records the concrete owner in
                            // `w_objclass`; CPython 3.14 uses that owner for both
                            // __objclass__ and __qualname__, so prefer it here.
                            let owner = getset_descriptor_owner(descr);
                            let type_qualname = if owner.is_null() {
                                "?".to_string()
                            } else {
                                descriptor_owner_qualname(owner)
                            };
                            let name_obj = pyre_object::typedef::w_getset_get_name(descr);
                            let name = if !name_obj.is_null() && pyre_object::is_str(name_obj) {
                                pyre_object::w_str_get_value(name_obj).to_string()
                            } else {
                                "<generic property>".to_string()
                            };
                            let combined =
                                pyre_object::w_str_new(&format!("{type_qualname}.{name}"));
                            pyre_object::typedef::w_getset_set_qualname(descr, combined);
                            Ok(combined)
                        }
                    },
                    2,
                ),
                "__qualname__",
            ),
            tp,
        ),
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
    crate::type_dict_store(
        tp,
        "__objclass__",
        copy_for_type(
            make_getset_descriptor_named(
                make_builtin_function_with_arity(
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
                ),
                "__objclass__",
            ),
            tp,
        ),
    );
    // typedef.py:473 __doc__ = interp_attrproperty('doc', ...)
    crate::type_dict_store(
        tp,
        "__doc__",
        copy_for_type(
            make_getset_descriptor_named(
                make_builtin_function_with_arity(
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
                ),
                "__doc__",
            ),
            tp,
        ),
    );
}

#[inline]
unsafe fn getset_descriptor_owner(descr: PyObjectRef) -> PyObjectRef {
    let objclass = unsafe { pyre_object::typedef::w_getset_get_objclass(descr) };
    if !objclass.is_null() {
        objclass
    } else {
        unsafe { pyre_object::typedef::w_getset_get_reqcls(descr) }
    }
}

/// The owner component used by CPython 3.14 descriptor `__qualname__`.
/// Heap classes carry their compiler-stamped qualified name in their own
/// namespace; static types fall back to the bare final component of tp_name.
unsafe fn descriptor_owner_qualname(owner: PyObjectRef) -> String {
    unsafe { pyre_object::w_type_get_qualname(owner) }.to_string()
}

/// CPython 3.14 getset descriptor repr, shared with `display::py_repr` because
/// native GetSetProperty instances do not carry a heap-instance `w_class`.
pub(crate) unsafe fn getset_descriptor_repr(descr: PyObjectRef) -> String {
    let name_obj = unsafe { pyre_object::typedef::w_getset_get_name(descr) };
    let name = if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
        unsafe { pyre_object::w_str_get_value(name_obj) }
    } else {
        "<generic property>"
    };
    let owner = unsafe { getset_descriptor_owner(descr) };
    let owner_name = if owner.is_null() {
        "?"
    } else {
        unsafe { pyre_object::w_type_get_name(owner) }
    };
    format!("<attribute '{name}' of '{owner_name}' objects>")
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
    make_getset_property_full(
        fget,
        fset,
        fdel,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        None,
    )
}

/// `GetSetProperty(fget, fset, fdel)` with explicit `name` — see
/// `make_getset_descriptor_named` for the typedef.py:58 motivation.
pub(crate) fn make_getset_property_named(
    fget: pyre_object::PyObjectRef,
    fset: pyre_object::PyObjectRef,
    fdel: pyre_object::PyObjectRef,
    name: &str,
) -> pyre_object::PyObjectRef {
    make_getset_property_full(
        fget,
        fset,
        fdel,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        Some(name),
    )
}

/// `GetSetProperty(..., doc=..., name=...)` — the full descriptor payload
/// used by doc-bearing getsets such as `mapping`, `__dict__`, and `__weakref__`.
fn make_getset_property_named_doc(
    fget: pyre_object::PyObjectRef,
    fset: pyre_object::PyObjectRef,
    fdel: pyre_object::PyObjectRef,
    doc: &str,
    name: &str,
) -> pyre_object::PyObjectRef {
    make_getset_property_full(
        fget,
        fset,
        fdel,
        pyre_object::w_str_new(doc),
        pyre_object::PY_NULL,
        Some(name),
    )
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
    doc: pyre_object::PyObjectRef,
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
        doc,
        cls,
        false, // use_closure
        resolved_name,
    )
}

fn init_type_type(ns: PyObjectRef) {
    // type.__new__(metatype, name, bases, dict) — creates new type
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(crate::builtins::type_descr_new),
        )
    };
    // `type[int]` builds a GenericAlias, but `type` carries no
    // `__class_getitem__` in its dict — `descroperation.getitem` special-cases
    // `is_w(w_obj, w_type)` (`descroperation.py:362`).  The wiring lives in
    // `baseobjspace::getitem_type`, so `hasattr(type, "__class_getitem__")`
    // stays False to match.
    // type.__init__ — no-op for now
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__init__",
            make_builtin_function("__init__", |_| Ok(pyre_object::w_none())),
        )
    };
    // type.__call__(cls, *args) — typeobject.c type_call.  The implicit
    // instantiation path handles `Cls()` directly, but a custom metaclass
    // whose `__call__` delegates via `super().__call__(...)` needs this
    // entry to resolve to the default __new__/__init__ behaviour.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // type.__annotations__ / __dict__ / __mro__ / __name__ / __bases__ /
    // __base__
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
            crate::baseobjspace::type_get_annotations(cls)
        },
        2,
    );
    let annotations_setter = make_builtin_function_with_arity(
        "__annotations__",
        |args| crate::baseobjspace::type_set_annotations(args[1], args[2]),
        3,
    );
    let annotations_deleter = make_builtin_function_with_arity(
        "__annotations__",
        |args| crate::baseobjspace::type_del_annotations(args[1]),
        2,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__annotations__",
            make_getset_property(annotations_getter, annotations_setter, annotations_deleter),
        )
    };

    let mro_getter = make_builtin_function_with_arity(
        "__mro__",
        |args| {
            let cls = args[1];
            unsafe {
                let mro_ptr = pyre_object::w_type_get_mro(cls);
                if mro_ptr.is_null() {
                    return Ok(pyre_object::w_tuple_new(vec![]));
                }
                Ok(pyre_object::w_tuple_new((*mro_ptr).to_vec()))
            }
        },
        2,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__mro__",
            make_getset_descriptor(mro_getter),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__flags__",
            make_getset_descriptor(flags_getter),
        )
    };

    // `type.mro(cls)` — typeobject.c `mro_external` / `type.mro`: the method
    // form returns the MRO as a fresh list (the `__mro__` getset above
    // returns the tuple).  Bound as a regular method, so `cls` is at args[0].
    let mro_method = make_builtin_function("mro", |args| {
        let cls = args[0];
        // typeobject.py:1081-1084 `descr_mro` computes the default C3 MRO
        // afresh. In particular this is callable from `Meta.mro()` while the
        // nascent class has not installed its final MRO yet.
        Ok(pyre_object::w_list_new(unsafe {
            crate::baseobjspace::compute_default_mro(cls)
        }))
    });
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, "mro", mro_method) };

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
        let subs = unsafe { pyre_object::w_type_get_subclasses(cls, true) };
        Ok(pyre_object::w_list_new(subs))
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__subclasses__",
            subclasses_method,
        )
    };

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
                    crate::type_dict_store(cls, "__module__", value);
                }
            }
            Ok(pyre_object::w_none())
        },
        3,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__module__",
            make_getset_property_named(
                module_getter,
                module_setter,
                pyre_object::PY_NULL,
                "__module__",
            ),
        )
    };

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
                // returns a read-only live view over the type's canonical
                // regular dict object.
                let canonical = ns_ptr as PyObjectRef;
                Ok(pyre_object::w_dict_proxy_new(canonical))
            }
        },
        2,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__dict__",
            make_getset_descriptor(dict_getter),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__name__",
            make_getset_property_named(name_getter, name_setter, pyre_object::PY_NULL, "__name__"),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__bases__",
            make_getset_property_named(
                bases_getter,
                bases_setter,
                pyre_object::PY_NULL,
                "__bases__",
            ),
        )
    };

    // PyPy typeobject.py:1164-1166 descr__base.  `object` has no best base,
    // which is surfaced as None; for multiple inheritance this follows the
    // most-derived instance layout rather than blindly choosing bases[0].
    let base_getter = make_builtin_function_with_arity(
        "__base__",
        |args| unsafe {
            let base = pyre_object::typeobject::w_type_get_best_base(args[1]);
            if base.is_null() {
                Ok(pyre_object::w_none())
            } else {
                Ok(base)
            }
        },
        2,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__base__",
            make_getset_descriptor(base_getter),
        )
    };
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
                    pyre_object::type_name_of(w_base)
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
fn function_receiver(obj: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    if obj.is_null() || !unsafe { pyre_object::py_type_check(obj, &crate::function::FUNCTION_TYPE) }
    {
        let received = if obj.is_null() {
            "object"
        } else {
            crate::typedef::r#type(obj)
                .map(|tp| unsafe { pyre_object::w_type_get_name(tp) })
                .unwrap_or("object")
        };
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' for 'function' objects doesn't apply to a '{received}' object"
        )));
    }
    Ok(obj)
}

fn init_function_type_common(ns: PyObjectRef) {
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
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__doc__",
        )?;
        Ok(unsafe { crate::function::fget_func_doc(func) })
    });
    let doc_setter = make_builtin_function("__doc__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__doc__",
        )?;
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_doc(func, value)? };
        Ok(pyre_object::w_none())
    });
    let doc_deleter = make_builtin_function("__doc__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__doc__",
        )?;
        unsafe { crate::function::fdel_func_doc(func)? };
        Ok(pyre_object::w_none())
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            make_getset_property(doc_getter, doc_setter, doc_deleter),
        )
    };
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
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__annotations__",
        )?;
        unsafe { crate::function::function_get_annotations(func) }
    });
    let ann_setter = make_builtin_function("__annotations__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__annotations__",
        )?;
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_annotations(func, value)? };
        Ok(pyre_object::w_none())
    });
    let ann_deleter = make_builtin_function("__annotations__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__annotations__",
        )?;
        unsafe { crate::function::fdel_func_annotations(func)? };
        Ok(pyre_object::w_none())
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__annotations__",
            make_getset_property(ann_getter, ann_setter, ann_deleter),
        )
    };
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
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__name__",
        )?;
        Ok(unsafe { crate::function::fget_func_name(func) })
    });
    let name_setter = make_builtin_function("__name__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__name__",
        )?;
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_name(func, value)? };
        Ok(pyre_object::w_none())
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__name__",
            make_getset_property(name_getter, name_setter, pyre_object::PY_NULL),
        )
    };
    // `typedef.py:782 getset_func_qualname = GetSetProperty(
    //   Function.fget_func_qualname, Function.fset_func_qualname)`.
    // Both getter and setter wired so `f.__qualname__ = "C.m"`
    // reaches `fset_func_qualname`'s str validation
    // (function.py:476-485) instead of falling through to
    // `setdictvalue` and silently shadowing the slot.
    let qualname_getter = make_builtin_function("__qualname__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__qualname__",
        )?;
        let s = unsafe { crate::function::function_get_qualname(func) };
        Ok(pyre_object::w_str_new(&s))
    });
    let qualname_setter = make_builtin_function("__qualname__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__qualname__",
        )?;
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_qualname(func, value)? };
        Ok(pyre_object::w_none())
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__qualname__",
            make_getset_property(qualname_getter, qualname_setter, pyre_object::PY_NULL),
        )
    };
    // `typedef.py:768-770 getset___module__ = GetSetProperty(
    //   Function.fget___module__, fset___module__, fdel___module__)`.
    let module_getter = make_builtin_function("__module__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__module__",
        )?;
        Ok(unsafe { crate::function::fget___module__(func) })
    });
    let module_setter = make_builtin_function("__module__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__module__",
        )?;
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset___module__(func, value)? };
        Ok(pyre_object::w_none())
    });
    let module_deleter = make_builtin_function("__module__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__module__",
        )?;
        unsafe { crate::function::fdel___module__(func)? };
        Ok(pyre_object::w_none())
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__module__",
            make_getset_property(module_getter, module_setter, module_deleter),
        )
    };
    // `typedef.py:772-774 getset_func_defaults = GetSetProperty(
    //   Function.fget_func_defaults, fset_func_defaults, fdel_func_defaults)`.
    let defaults_getter = make_builtin_function("__defaults__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__defaults__",
        )?;
        Ok(unsafe { crate::function::fget_func_defaults(func) })
    });
    let defaults_setter = make_builtin_function("__defaults__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__defaults__",
        )?;
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_defaults(func, value)? };
        Ok(pyre_object::w_none())
    });
    let defaults_deleter = make_builtin_function("__defaults__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__defaults__",
        )?;
        unsafe { crate::function::fdel_func_defaults(func)? };
        Ok(pyre_object::w_none())
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__defaults__",
            make_getset_property(defaults_getter, defaults_setter, defaults_deleter),
        )
    };
    // `typedef.py:775-777 getset_func_kwdefaults = GetSetProperty(...)`.
    let kwdefaults_getter = make_builtin_function("__kwdefaults__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__kwdefaults__",
        )?;
        Ok(unsafe { crate::function::fget_func_kwdefaults(func) })
    });
    let kwdefaults_setter = make_builtin_function("__kwdefaults__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__kwdefaults__",
        )?;
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_kwdefaults(func, value)? };
        Ok(pyre_object::w_none())
    });
    let kwdefaults_deleter = make_builtin_function("__kwdefaults__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__kwdefaults__",
        )?;
        unsafe { crate::function::fdel_func_kwdefaults(func)? };
        Ok(pyre_object::w_none())
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__kwdefaults__",
            make_getset_property(kwdefaults_getter, kwdefaults_setter, kwdefaults_deleter),
        )
    };
    // `typedef.py:778-779 getset_func_code = GetSetProperty(
    //   Function.fget_func_code, fset_func_code)`.
    let code_getter = make_builtin_function("__code__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__code__",
        )?;
        let raw = unsafe { crate::function::fget_func_code(func) };
        Ok(raw as pyre_object::PyObjectRef)
    });
    let code_setter = make_builtin_function("__code__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__code__",
        )?;
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::fset_func_code(func, value)? };
        Ok(pyre_object::w_none())
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__code__",
            make_getset_property(code_getter, code_setter, pyre_object::PY_NULL),
        )
    };
    // `typedef.py:813 __closure__ = GetSetProperty(Function.fget_func_closure)`
    // — read-only.
    let closure_getter = make_builtin_function("__closure__", |args| {
        let func = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__closure__",
        )?;
        Ok(unsafe { crate::function::fget_func_closure(func) })
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__closure__",
            make_getset_descriptor(closure_getter),
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__globals__",
            make_getset_descriptor(globals_getter),
        )
    };
    // CPython 3.14 `func.__builtins__` is the construction-time
    // `func_builtins` field, not a fresh lookup in `func_globals`.  The
    // allocator resolves a Module to its dict and roots that exact object.
    let func_builtins_getter = make_builtin_function("__builtins__", |args| {
        let func = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        if func.is_null() {
            return Ok(pyre_object::w_none());
        }
        let w_builtin = unsafe { crate::function::function_get_builtins(func) };
        if w_builtin.is_null() {
            Ok(pyre_object::w_none())
        } else {
            Ok(w_builtin)
        }
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__builtins__",
            make_getset_descriptor(func_builtins_getter),
        )
    };
    // `pypy/interpreter/typedef.py:805 __objclass__ = getset_func_objclass`
    //
    // ```python
    // getset_func_objclass = GetSetProperty(Function.fget_func_objclass)
    // ```
    //
    // Read-only descriptor that surfaces `self.w_objclass` for
    // introspection helpers (`inspect.getfullargspec` etc.); raises
    // AttributeError when no class is bound (`function.py:498-501`).
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__text_signature__",
            make_getset_property(
                text_signature_getter,
                text_signature_setter,
                pyre_object::PY_NULL,
            ),
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // typedef.py:793-794 `Function.typedef.__call__`; copied verbatim into
    // `BuiltinFunction.typedef` by `**Function.typedef.rawdict`.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__call__",
            make_builtin_function("__call__", function_descr_call),
        )
    };
}

/// PyPy `Function.descr_function_call(self, __args__)`: forward the complete
/// Arguments object, including keyword names, to the wrapped function.
fn function_descr_call(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let function = function_receiver(
        positional.first().copied().unwrap_or(pyre_object::PY_NULL),
        "__call__",
    )?;
    function_descr_call_impl(positional, kwargs, function)
}

fn function_descr_call_impl(
    positional: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
    function: PyObjectRef,
) -> crate::PyResult {
    let call_args = positional.get(1..).unwrap_or(&[]);
    if !crate::builtins::has_real_kwargs(kwargs) {
        return crate::call::call_function_impl_result(function, call_args);
    }
    let keyword_args: Vec<(Wtf8Buf, PyObjectRef)> = unsafe {
        pyre_object::w_dict_str_entries(kwargs.unwrap())
            .into_iter()
            .filter(|(name, _)| name != "__pyre_kw__")
            .map(|(name, value)| (Wtf8Buf::from_string(name), value))
            .collect()
    };
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "function call has no current frame",
            ));
        }
        crate::call::call_with_kwargs(unsafe { &mut *frame }, function, call_args, &keyword_args)
    })
}

/// Python 3.14 direct `PyMemberDef` accessors. The tagged Member index keeps
/// native object fields on the descriptor itself rather than in a side table.
pub(crate) unsafe fn direct_member_get(member: PyObjectRef, obj: PyObjectRef) -> crate::PyResult {
    match unsafe { pyre_object::w_member_get_direct_kind(member) } {
        pyre_object::MEMBER_FUNCTION_CLOSURE => {
            Ok(unsafe { crate::function::fget_func_closure(obj) })
        }
        pyre_object::MEMBER_FUNCTION_DOC => Ok(unsafe { crate::function::fget_func_doc(obj) }),
        pyre_object::MEMBER_FUNCTION_GLOBALS => {
            let value = unsafe { crate::function::function_get_globals_obj(obj) };
            Ok(if value.is_null() {
                pyre_object::w_none()
            } else {
                value
            })
        }
        pyre_object::MEMBER_FUNCTION_MODULE => Ok(unsafe { crate::function::fget___module__(obj) }),
        pyre_object::MEMBER_FUNCTION_BUILTINS => {
            let builtins = unsafe { crate::function::function_get_builtins(obj) };
            Ok(if builtins.is_null() {
                pyre_object::w_none()
            } else {
                builtins
            })
        }
        pyre_object::MEMBER_MODULE_DICT => Ok(unsafe { pyre_object::w_module_get_w_dict(obj) }),
        _ => Err(crate::PyError::attribute_error(unsafe {
            pyre_object::w_member_get_name(member)
        })),
    }
}

pub(crate) unsafe fn direct_member_set(
    member: PyObjectRef,
    obj: PyObjectRef,
    value: PyObjectRef,
) -> crate::PyResult {
    match unsafe { pyre_object::w_member_get_direct_kind(member) } {
        pyre_object::MEMBER_FUNCTION_DOC => {
            unsafe { crate::function::fset_func_doc(obj, value)? };
            Ok(pyre_object::w_none())
        }
        pyre_object::MEMBER_FUNCTION_MODULE => {
            if unsafe { pyre_object::py_type_check(obj, &crate::function::BUILTIN_FUNCTION_TYPE) } {
                unsafe { crate::function::builtin_function_set_module_attr(obj, value) };
            } else {
                unsafe { crate::function::fset___module__(obj, value)? };
            }
            Ok(pyre_object::w_none())
        }
        _ => Err(crate::PyError::attribute_error("readonly attribute")),
    }
}

pub(crate) unsafe fn direct_member_delete(
    member: PyObjectRef,
    obj: PyObjectRef,
) -> crate::PyResult {
    match unsafe { pyre_object::w_member_get_direct_kind(member) } {
        pyre_object::MEMBER_FUNCTION_DOC => {
            unsafe { crate::function::fdel_func_doc(obj)? };
            Ok(pyre_object::w_none())
        }
        pyre_object::MEMBER_FUNCTION_MODULE => {
            if unsafe { pyre_object::py_type_check(obj, &crate::function::BUILTIN_FUNCTION_TYPE) } {
                unsafe {
                    crate::function::builtin_function_set_module_attr(obj, pyre_object::w_none())
                };
            } else {
                unsafe { crate::function::fdel___module__(obj)? };
            }
            Ok(pyre_object::w_none())
        }
        _ => Err(crate::PyError::attribute_error("readonly attribute")),
    }
}

fn init_function_type(ns: PyObjectRef) {
    init_function_type_common(ns);
    // CPython 3.14 `func_memberlist`: these five entries are direct
    // member_descriptor objects. PyPy's equivalent values live in
    // Function.typedef as GetSetProperty; the observable descriptor kind is
    // the 3.14 difference selected by this project.
    for (name, kind) in [
        ("__closure__", pyre_object::MEMBER_FUNCTION_CLOSURE),
        ("__doc__", pyre_object::MEMBER_FUNCTION_DOC),
        ("__globals__", pyre_object::MEMBER_FUNCTION_GLOBALS),
        ("__module__", pyre_object::MEMBER_FUNCTION_MODULE),
        ("__builtins__", pyre_object::MEMBER_FUNCTION_BUILTINS),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                pyre_object::w_member_new_direct(kind, name.to_owned(), pyre_object::PY_NULL),
            );
        }
    }
    // CPython 3.14 func_repr: `<function {qualname} at {address}>`.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    let function = function_receiver(
                        args.first().copied().unwrap_or(pyre_object::PY_NULL),
                        "__repr__",
                    )?;
                    let qualname = unsafe { crate::function::function_get_qualname(function) };
                    Ok(pyre_object::w_str_new(&format!(
                        "<function {qualname} at {function:p}>"
                    )))
                },
                1,
            ),
        )
    };
    // PyPy typedef.py:796 `getset_func_dict = GetSetProperty(
    // descr_get_dict, descr_set_dict, cls=Function)` — storage is the
    // typed `Function.w_func_dict` field from function.py:68.
    let dict_getter = make_builtin_function_with_arity(
        "__dict__",
        |args| {
            function_receiver(
                args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
                "__dict__",
            )?;
            descr_get_dict(args)
        },
        2,
    );
    let dict_setter = make_builtin_function_with_arity(
        "__dict__",
        |args| {
            function_receiver(
                args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
                "__dict__",
            )?;
            descr_set_dict(args)
        },
        3,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__dict__",
            make_getset_property(dict_getter, dict_setter, pyre_object::PY_NULL),
        )
    };
    // CPython 3.14 `function.__annotate__`: callable-or-None, not deletable.
    let annotate_getter = make_builtin_function("__annotate__", |args| {
        let function = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__annotate__",
        )?;
        if unsafe { crate::function::function_has_builtin_code(function) } {
            return Err(crate::PyError::attribute_error(
                "builtin function has no attribute '__annotate__'",
            ));
        }
        Ok(unsafe { crate::function::function_get_annotate(function) })
    });
    let annotate_setter = make_builtin_function("__annotate__", |args| {
        let function = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__annotate__",
        )?;
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::function_set_annotate(function, value)? };
        Ok(pyre_object::w_none())
    });
    let annotate_deleter = make_builtin_function("__annotate__", |args| {
        let function = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__annotate__",
        )?;
        unsafe { crate::function::function_set_annotate(function, pyre_object::PY_NULL)? };
        Ok(pyre_object::w_none())
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__annotate__",
            make_getset_property(annotate_getter, annotate_setter, annotate_deleter),
        )
    };
    // CPython 3.14 `function.__type_params__`: tuple-only and not deletable.
    let typeparams_getter = make_builtin_function("__type_params__", |args| {
        let function = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__type_params__",
        )?;
        Ok(unsafe { crate::function::function_get_typeparams(function) })
    });
    let typeparams_setter = make_builtin_function("__type_params__", |args| {
        let function = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__type_params__",
        )?;
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        unsafe { crate::function::function_set_typeparams(function, value)? };
        Ok(pyre_object::w_none())
    });
    let typeparams_deleter = make_builtin_function("__type_params__", |args| {
        let function = function_receiver(
            args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
            "__type_params__",
        )?;
        unsafe { crate::function::function_set_typeparams(function, pyre_object::PY_NULL)? };
        Ok(pyre_object::w_none())
    });
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__type_params__",
            make_getset_property(typeparams_getter, typeparams_setter, typeparams_deleter),
        )
    };
    // The shared rawdict mirrors PyPy and is still used by
    // BuiltinFunction.  Python 3.14's user `function` type omits these
    // PyPy-only introspection extensions.
    unsafe {
        pyre_object::dictmultiobject::w_dict_delitem_str_no_proxy(ns, "__objclass__");
        pyre_object::dictmultiobject::w_dict_delitem_str_no_proxy(ns, "__text_signature__");
        pyre_object::dictmultiobject::w_dict_delitem_str_no_proxy(ns, "__defaults_count__");
    }
    // `funcobject.c func_new` — `FunctionType(code, globals, name=None,
    // argdefs=None, closure=None, kwdefaults=None)`.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(crate::function::descr_function_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__get__",
            make_builtin_function("__get__", |args| {
                let w_function = function_receiver(
                    args.first().copied().unwrap_or(pyre_object::PY_NULL),
                    "__get__",
                )?;
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
                // Python 3.14 `func_descr_get`: omitting `type` is equivalent
                // to passing None, but `__get__(None, None)` is invalid.
                if obj_is_none && cls_is_none {
                    return Err(crate::PyError::type_error("__get__(None, None) is invalid"));
                }
                if obj_is_none {
                    Ok(w_function)
                } else {
                    // function.py:470 Method(space, w_function, w_obj, w_cls),
                    // with CPython's inferred owner when `type` is omitted.
                    let owner = if cls_is_none {
                        r#type(w_obj).unwrap_or(pyre_object::PY_NULL)
                    } else {
                        w_cls
                    };
                    Ok(pyre_object::w_method_new(w_function, w_obj, owner))
                }
            }),
        )
    };
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
fn builtin_function_receiver(obj: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    if obj.is_null()
        || !unsafe { pyre_object::py_type_check(obj, &crate::function::BUILTIN_FUNCTION_TYPE) }
    {
        let received = if obj.is_null() {
            "object"
        } else {
            crate::typedef::r#type(obj)
                .map(|tp| unsafe { pyre_object::w_type_get_name(tp) })
                .unwrap_or("object")
        };
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' for 'builtin_function_or_method' objects doesn't apply to a '{received}' object"
        )));
    }
    Ok(obj)
}

fn init_builtin_function_type(ns: PyObjectRef) {
    init_function_type_common(ns);

    // CPython 3.14 `PyCFunction_Type` does not expose the user-function
    // storage copied by PyPy's `**Function.typedef.rawdict`. Keep the PyPy
    // construction above, then apply the version-selected surface delta.
    for name in [
        "__annotations__",
        "__builtins__",
        "__closure__",
        "__code__",
        "__defaults__",
        "__defaults_count__",
        "__globals__",
        "__kwdefaults__",
        "__new__",
        "__objclass__",
    ] {
        unsafe { pyre_object::dictmultiobject::w_dict_delitem_str_no_proxy(ns, name) };
    }

    // methodobject.c `meth_members`: `__module__` is the one direct member;
    // it accepts arbitrary assignments and deletion stores None.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__module__",
            pyre_object::w_member_new_direct(
                pyre_object::MEMBER_FUNCTION_MODULE,
                "__module__".to_owned(),
                pyre_object::PY_NULL,
            ),
        )
    };

    // methodobject.c `meth_getsets`. All five are read-only in 3.14. PyPy's
    // `BuiltinFunction.w_moduleobj` supplies `__self__`; type `__new__`
    // carriers fall back to their separately stamped defining type.
    for (name, getter) in [
        (
            "__doc__",
            (|args: &[PyObjectRef]| Ok(unsafe { crate::function::fget_func_doc(args[1]) }))
                as fn(&[PyObjectRef]) -> crate::PyResult,
        ),
        ("__name__", |args: &[PyObjectRef]| {
            Ok(unsafe { crate::function::fget_func_name(args[1]) })
        }),
        ("__qualname__", |args: &[PyObjectRef]| {
            Ok(pyre_object::w_str_new(&unsafe {
                crate::function::function_get_qualname(args[1])
            }))
        }),
        ("__self__", |args: &[PyObjectRef]| {
            Ok(unsafe { crate::function::function_get_self_or_none(args[1]) })
        }),
        ("__text_signature__", |args: &[PyObjectRef]| unsafe {
            crate::function::fget_func_text_signature(args[1])
        }),
    ] {
        let get = make_builtin_function_with_arity(name, getter, 2);
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_getset_descriptor(get),
            )
        };
    }

    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__call__",
            make_builtin_function("__call__", |args| {
                let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
                let function = builtin_function_receiver(
                    positional.first().copied().unwrap_or(pyre_object::PY_NULL),
                    "__call__",
                )?;
                function_descr_call_impl(positional, kwargs, function)
            }),
        )
    };

    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    let func = builtin_function_receiver(
                        args.first().copied().unwrap_or(pyre_object::PY_NULL),
                        "__repr__",
                    )?;
                    let name = unsafe { crate::function_get_name(func) };
                    Ok(pyre_object::w_str_new(&format!(
                        "<built-in function {name}>"
                    )))
                },
                1,
            ),
        )
    };

    // function.py:807-808 `BuiltinFunction.descr__reduce__`.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                |args| unsafe {
                    let function = builtin_function_receiver(
                        args.first().copied().unwrap_or(pyre_object::PY_NULL),
                        "__reduce__",
                    )?;
                    crate::function::descr_builtin_function_reduce(function)
                },
                1,
            ),
        )
    };

    // CPython 3.14's method-wrapper slots are materialised directly on
    // `PyCFunction_Type`. Their semantics are the object identity defaults.
    for (name, method) in [
        (
            "__eq__",
            (|args: &[PyObjectRef]| {
                let function = builtin_function_receiver(
                    args.first().copied().unwrap_or(pyre_object::PY_NULL),
                    "__eq__",
                )?;
                Ok(pyre_object::w_bool_from(std::ptr::eq(function, args[1])))
            }) as fn(&[PyObjectRef]) -> crate::PyResult,
        ),
        ("__ne__", |args: &[PyObjectRef]| {
            let function = builtin_function_receiver(
                args.first().copied().unwrap_or(pyre_object::PY_NULL),
                "__ne__",
            )?;
            Ok(pyre_object::w_bool_from(!std::ptr::eq(function, args[1])))
        }),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, method, 2),
            )
        };
    }
    type BuiltinOrderFn = fn(&[PyObjectRef]) -> crate::PyResult;
    fn order(args: &[PyObjectRef], name: &str) -> crate::PyResult {
        builtin_function_receiver(args.first().copied().unwrap_or(pyre_object::PY_NULL), name)?;
        Ok(pyre_object::w_not_implemented())
    }
    fn lt(args: &[PyObjectRef]) -> crate::PyResult {
        order(args, "__lt__")
    }
    fn le(args: &[PyObjectRef]) -> crate::PyResult {
        order(args, "__le__")
    }
    fn gt(args: &[PyObjectRef]) -> crate::PyResult {
        order(args, "__gt__")
    }
    fn ge(args: &[PyObjectRef]) -> crate::PyResult {
        order(args, "__ge__")
    }
    for (name, function) in [
        ("__lt__", lt as BuiltinOrderFn),
        ("__le__", le),
        ("__gt__", gt),
        ("__ge__", ge),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, function, 2),
            )
        };
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| {
                    let function = builtin_function_receiver(
                        args.first().copied().unwrap_or(pyre_object::PY_NULL),
                        "__hash__",
                    )?;
                    Ok(pyre_object::w_int_new(function as i64))
                },
                1,
            ),
        )
    };
}

/// Stamp CPython 3.14's builtin-function getset/member descriptors after the
/// W_TypeObject is reachable. PyPy supplies `cls=BuiltinFunction` for the
/// inherited descriptors; CPython's `PyGetSetDef` / `PyMemberDef` entries all
/// carry `PyCFunction_Type` as their owner.
fn patch_builtin_function_descriptors() {
    let bf_type =
        gettypefor(&crate::BUILTIN_FUNCTION_TYPE as *const PyType).unwrap_or(pyre_object::PY_NULL);
    if bf_type.is_null() {
        return;
    }
    if !crate::type_dict_has_storage(bf_type) {
        return;
    }
    for name in [
        "__doc__",
        "__name__",
        "__qualname__",
        "__self__",
        "__text_signature__",
    ] {
        if let Some(descr) = crate::type_dict_lookup(bf_type, name) {
            if unsafe { pyre_object::typedef::is_getset_property(descr) } {
                unsafe { pyre_object::typedef::w_getset_set_reqcls(descr, bf_type) };
            }
        }
    }
    if let Some(descr) = crate::type_dict_lookup(bf_type, "__module__") {
        if unsafe { pyre_object::is_member(descr) && pyre_object::w_member_is_direct(descr) } {
            unsafe { pyre_object::w_member_set_cls(descr, bf_type) };
        }
    }
}

/// Stamp the owner of Python 3.14's direct function member descriptors after
/// the Function W_TypeObject is available. This is the Member counterpart of
/// the GetSetProperty reqcls patch immediately above.
fn patch_function_member_descriptors() {
    let function_type =
        gettypefor(&crate::FUNCTION_TYPE as *const PyType).unwrap_or(pyre_object::PY_NULL);
    if function_type.is_null() || !crate::type_dict_has_storage(function_type) {
        return;
    }
    for name in [
        "__closure__",
        "__doc__",
        "__globals__",
        "__module__",
        "__builtins__",
    ] {
        if let Some(descr) = crate::type_dict_lookup(function_type, name) {
            if unsafe { pyre_object::is_member(descr) && pyre_object::w_member_is_direct(descr) } {
                unsafe { pyre_object::w_member_set_cls(descr, function_type) };
            }
        }
    }
}

/// module.py Module.typedef binds the `__annotations__` methods to Module;
/// stamp that inferred `cls=Module` after the W_TypeObject is registered so
/// foreign receivers fail in GetSetProperty.typecheck before field access.
fn patch_module_descriptors() {
    let module_type =
        gettypefor(&pyre_object::MODULE_TYPE as *const PyType).unwrap_or(pyre_object::PY_NULL);
    if module_type.is_null() || !crate::type_dict_has_storage(module_type) {
        return;
    }
    for name in ["__annotations__", "__annotate__"] {
        if let Some(descr) = crate::type_dict_lookup(module_type, name) {
            if unsafe { pyre_object::typedef::is_getset_property(descr) } {
                unsafe { pyre_object::typedef::w_getset_set_reqcls(descr, module_type) };
            }
        }
    }
}

/// typedef.py:947-951 stamps `cls=Cell` on `cell_contents`. The descriptor
/// is constructed before the cell W_TypeObject exists, so fill its `reqcls`
/// field in the same post-registration pass used for BuiltinFunction and
/// frame descriptors.
fn patch_cell_descriptor() {
    let cell_type =
        gettypefor(&pyre_object::nestedscope::CELL_TYPE).unwrap_or(pyre_object::PY_NULL);
    if cell_type.is_null() || !crate::type_dict_has_storage(cell_type) {
        return;
    }
    if let Some(descr) = crate::type_dict_lookup(cell_type, "cell_contents") {
        if unsafe { pyre_object::typedef::is_getset_property(descr) } {
            unsafe { pyre_object::typedef::w_getset_set_reqcls(descr, cell_type) };
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
        let ns = unsafe { pyre_object::w_type_get_dict_ptr(w_type) } as PyObjectRef;
        if ns.is_null() {
            continue;
        }
        let descrs: Vec<PyObjectRef> = unsafe { pyre_object::w_dict_items(ns) }
            .into_iter()
            .filter_map(|(_, descr)| {
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
fn init_builtin_code_type(ns: PyObjectRef) {
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "co_name",
            make_getset_descriptor(co_name_getter),
        )
    };

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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "co_argcount",
            make_getset_descriptor(argcount_getter),
        )
    };
    let posonly_getter = make_builtin_function_with_arity(
        "co_posonlyargcount",
        |args| {
            Ok(pyre_object::w_int_new(
                code_sig(args).map_or(0, |s| s.num_posonlyargnames()) as i64,
            ))
        },
        2,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "co_posonlyargcount",
            make_getset_descriptor(posonly_getter),
        )
    };
    let kwonly_getter = make_builtin_function_with_arity(
        "co_kwonlyargcount",
        |args| {
            Ok(pyre_object::w_int_new(
                code_sig(args).map_or(0, |s| s.num_kwonlyargnames()) as i64,
            ))
        },
        2,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "co_kwonlyargcount",
            make_getset_descriptor(kwonly_getter),
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "co_varnames",
            make_getset_descriptor(varnames_getter),
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "co_flags",
            make_getset_descriptor(flags_getter),
        )
    };
}

fn init_method_type(ns: PyObjectRef) {
    // typedef.py:833-848 Method.typedef, completed with CPython 3.14's
    // ordering wrappers. Bound methods carry one wrapped callable and one
    // bound instance; every operation below reads those two typed fields.
    let doc_getter = make_builtin_function_with_arity(
        "__doc__",
        |args| {
            let method = crate::function::require_method(
                args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
                "__doc__",
            )?;
            let function = unsafe { pyre_object::w_method_get_func(method) };
            crate::baseobjspace::getattr_str(function, "__doc__")
        },
        2,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            make_getset_descriptor(doc_getter),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(|args| {
                if args.len() != 3 {
                    return Err(crate::PyError::type_error(format!(
                        "method expected 2 arguments, got {}",
                        args.len().saturating_sub(1),
                    )));
                }
                crate::function::descr_method__new__(args[0], args[1], args[2])
            }),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__call__",
            make_builtin_function("__call__", crate::function::descr_method_call),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__get__",
            make_builtin_function("__get__", |args| unsafe {
                let method = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                let cls = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
                crate::function::descr_method_get(method, obj, cls)
            }),
        )
    };
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
            let method = crate::function::require_method(
                args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
                "__func__",
            )?;
            let w_value = unsafe { pyre_object::w_method_get_func(method) };
            if w_value.is_null() {
                Ok(pyre_object::w_none())
            } else {
                Ok(w_value)
            }
        },
        2,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__func__",
            make_getset_descriptor(func_getter),
        )
    };
    let self_getter = make_builtin_function_with_arity(
        "__self__",
        |args| {
            let method = crate::function::require_method(
                args.get(1).copied().unwrap_or(pyre_object::PY_NULL),
                "__self__",
            )?;
            let w_value = unsafe { pyre_object::w_method_get_self(method) };
            if w_value.is_null() {
                Ok(pyre_object::w_none())
            } else {
                Ok(w_value)
            }
        },
        2,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__self__",
            make_getset_descriptor(self_getter),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__getattribute__",
            make_builtin_function_with_arity(
                "__getattribute__",
                |args| unsafe { crate::function::descr_method_getattribute(args[0], args[1]) },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__eq__",
            make_builtin_function_with_arity(
                "__eq__",
                |args| unsafe { crate::function::descr_method_eq(args[0], args[1]) },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ne__",
            make_builtin_function_with_arity(
                "__ne__",
                |args| unsafe { crate::function::descr_method_ne(args[0], args[1]) },
                2,
            ),
        )
    };
    type MethodOrderFn = fn(&[PyObjectRef]) -> crate::PyResult;
    fn order(args: &[PyObjectRef], name: &str) -> crate::PyResult {
        crate::function::require_method(
            args.first().copied().unwrap_or(pyre_object::PY_NULL),
            name,
        )?;
        Ok(pyre_object::special::w_not_implemented())
    }
    fn lt(args: &[PyObjectRef]) -> crate::PyResult {
        order(args, "__lt__")
    }
    fn le(args: &[PyObjectRef]) -> crate::PyResult {
        order(args, "__le__")
    }
    fn gt(args: &[PyObjectRef]) -> crate::PyResult {
        order(args, "__gt__")
    }
    fn ge(args: &[PyObjectRef]) -> crate::PyResult {
        order(args, "__ge__")
    }
    for (name, function) in [
        ("__lt__", lt as MethodOrderFn),
        ("__le__", le),
        ("__gt__", gt),
        ("__ge__", ge),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, function, 2),
            )
        };
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| unsafe {
                    Ok(pyre_object::w_int_new(crate::function::descr_method_hash(
                        args[0],
                    )?))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| unsafe { crate::function::descr_method_repr(args[0]) },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                |args| unsafe { crate::function::descr_method__reduce__(args[0]) },
                1,
            ),
        )
    };
}

fn code_descr_new(args: &[PyObjectRef]) -> crate::PyResult {
    unsafe { crate::pycode::code_new(args) }
}

fn code_descr_eq(args: &[PyObjectRef]) -> crate::PyResult {
    unsafe { crate::pycode::code_eq(args[0], args[1]) }
}

fn code_descr_ne(args: &[PyObjectRef]) -> crate::PyResult {
    unsafe { crate::pycode::code_ne(args[0], args[1]) }
}

fn code_descr_positions(args: &[PyObjectRef]) -> crate::PyResult {
    unsafe { crate::pycode::code_positions(args[0]) }
}

fn code_descr_lines(args: &[PyObjectRef]) -> crate::PyResult {
    unsafe { crate::pycode::code_lines(args[0]) }
}

fn code_descr_branches(args: &[PyObjectRef]) -> crate::PyResult {
    unsafe { crate::pycode::code_branches(args[0]) }
}

fn code_field_getter(args: &[PyObjectRef]) -> crate::PyResult {
    let descriptor = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let name = unsafe { pyre_object::typedef::w_getset_get_name(descriptor) };
    let Some(name) = (unsafe { pyre_object::w_str_get_value_opt(name) }) else {
        return Err(crate::PyError::runtime_error(
            "code field descriptor has no name",
        ));
    };
    unsafe {
        crate::pycode::code_get_field(args.get(1).copied().unwrap_or(pyre_object::PY_NULL), name)
    }
}

fn init_code_type(ns: PyObjectRef) {
    // PyPy typedef.py:695-725 `PyCode.typedef`, with the Python 3.14-only
    // slots (`__replace__`, `co_branches`, adaptive bytes and ordering
    // wrappers) added from `PyCode_Type`. Every field descriptor reads the
    // single compiler CodeObject stored by `PyCode`.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            pyre_object::w_str_new("Create a code object.  Not for the faint of heart."),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(code_descr_new),
        );
    }
    for (name, function) in [
        ("__eq__", code_descr_eq as crate::gateway::BuiltinCodeFn),
        ("__ne__", code_descr_ne as crate::gateway::BuiltinCodeFn),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, function, 2),
            );
        }
    }
    for name in ["__lt__", "__le__", "__gt__", "__ge__"] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(
                    name,
                    |_args| Ok(pyre_object::special::w_not_implemented()),
                    2,
                ),
            );
        }
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| unsafe { Ok(pyre_object::w_int_new(crate::pycode::code_hash(args[0])?)) },
                1,
            ),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| unsafe { crate::pycode::code_repr(args[0]) },
                1,
            ),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__sizeof__",
            make_builtin_function_with_arity(
                "__sizeof__",
                |args| unsafe { crate::pycode::code_sizeof(args[0]) },
                1,
            ),
        );
    }

    for name in ["replace", "__replace__"] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function(name, |args| unsafe { crate::pycode::code_replace(args) }),
            );
        }
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "_varname_from_oparg",
            make_builtin_function_with_arity(
                "_varname_from_oparg",
                |args| unsafe { crate::pycode::code_varname_from_oparg(args[0], args[1]) },
                2,
            ),
        );
    }
    for (name, function) in [
        (
            "co_positions",
            code_descr_positions as crate::gateway::BuiltinCodeFn,
        ),
        (
            "co_lines",
            code_descr_lines as crate::gateway::BuiltinCodeFn,
        ),
        (
            "co_branches",
            code_descr_branches as crate::gateway::BuiltinCodeFn,
        ),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, function, 1),
            );
        }
    }

    for name in [
        "_co_code_adaptive",
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_nlocals",
        "co_stacksize",
        "co_flags",
        "co_code",
        "co_consts",
        "co_names",
        "co_varnames",
        "co_freevars",
        "co_cellvars",
        "co_filename",
        "co_name",
        "co_qualname",
        "co_firstlineno",
        "co_linetable",
        "co_exceptiontable",
        "co_lnotab",
    ] {
        let getter = make_builtin_function_with_arity(name, code_field_getter, 2);
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_getset_descriptor_named(getter, name),
            );
        }
    }
}

/// typedef.py:533-540 Member.typedef
fn init_member_descriptor_type(ns: PyObjectRef) {
    // typedef.py:535 __get__ = interp2app(Member.descr_member_get)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                            pyre_object::type_name_of(obj),
                        )));
                    }
                }
                if unsafe { pyre_object::w_member_is_direct(descr) } {
                    return unsafe { direct_member_get(descr, obj) };
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
        )
    };
    // typedef.py:536 __set__ = interp2app(Member.descr_member_set)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                            pyre_object::type_name_of(obj),
                        )));
                    }
                }
                if unsafe { pyre_object::w_member_is_direct(descr) } {
                    return unsafe { direct_member_set(descr, obj, value) };
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
        )
    };
    // typedef.py:537 __delete__ = interp2app(Member.descr_member_del)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                            pyre_object::type_name_of(obj),
                        )));
                    }
                }
                if unsafe { pyre_object::w_member_is_direct(descr) } {
                    return unsafe { direct_member_delete(descr, obj) };
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
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__name__",
            make_getset_descriptor(name_getter),
        )
    };
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__objclass__",
            make_getset_descriptor(objclass_getter),
        )
    };
    // CPython 3.14 `PyMemberDescr_Type` metadata.  PyPy's Member typedef
    // stops at __name__/__objclass__; these four entries are the selected
    // 3.14 surface.
    let doc_getter =
        make_builtin_function_with_arity("__doc__", |_args| Ok(pyre_object::w_none()), 2);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            make_getset_descriptor_named(doc_getter, "__doc__"),
        )
    };

    let qualname_getter = make_builtin_function_with_arity(
        "__qualname__",
        |args| {
            let member = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if member.is_null() || !unsafe { pyre_object::typedef::is_member(member) } {
                return Ok(pyre_object::w_none());
            }
            let name = unsafe { pyre_object::w_member_get_name(member) };
            let owner = unsafe { pyre_object::w_member_get_cls(member) };
            let owner_qualname = if owner.is_null() {
                "?".to_string()
            } else {
                unsafe { pyre_object::w_type_get_qualname(owner) }.to_string()
            };
            Ok(pyre_object::w_str_new(&format!("{owner_qualname}.{name}")))
        },
        2,
    );
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__qualname__",
            make_getset_descriptor_named(qualname_getter, "__qualname__"),
        )
    };

    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    let member = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                    if member.is_null() || !unsafe { pyre_object::typedef::is_member(member) } {
                        return Err(crate::PyError::type_error(
                            "descriptor '__repr__' requires a 'member_descriptor' object",
                        ));
                    }
                    Ok(pyre_object::w_str_new(&unsafe {
                        member_descriptor_repr(member)
                    }))
                },
                1,
            ),
        )
    };

    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                |args| {
                    let member = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                    if member.is_null() || !unsafe { pyre_object::typedef::is_member(member) } {
                        return Err(crate::PyError::type_error(
                            "descriptor '__reduce__' requires a 'member_descriptor' object",
                        ));
                    }
                    let owner = unsafe { pyre_object::w_member_get_cls(member) };
                    let name =
                        pyre_object::w_str_new(unsafe { pyre_object::w_member_get_name(member) });
                    Ok(pyre_object::w_tuple_new(vec![
                        crate::baseobjspace::builtin_callable("getattr"),
                        pyre_object::w_tuple_new(vec![owner, name]),
                    ]))
                },
                1,
            ),
        )
    };
}

/// CPython 3.14 `member_get_qualname` / `member_repr` surface shared by the
/// registered `__repr__` method and `display::py_repr`'s native descriptor
/// dispatch.
pub(crate) unsafe fn member_descriptor_repr(member: PyObjectRef) -> String {
    let name = unsafe { pyre_object::w_member_get_name(member) };
    let owner = unsafe { pyre_object::w_member_get_cls(member) };
    let owner_name = if owner.is_null() {
        "?"
    } else {
        unsafe { pyre_object::w_type_get_name(owner) }
    };
    format!("<member '{name}' of '{owner_name}' objects>")
}

/// CPython 3.14 `cell_new`, matching PyPy `descr_new_cell` except that the
/// 3.14 positional-only argument surface rejects keywords.
fn cell_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "cell() takes no keyword arguments",
        ));
    }
    let cls = positional.first().copied().unwrap_or(PY_NULL);
    let cell_type = gettypefor(&pyre_object::nestedscope::CELL_TYPE).unwrap_or(PY_NULL);
    check_user_subclass(cell_type, cls)?;
    let contents = match positional.get(1..) {
        Some([]) | None => PY_NULL,
        Some([contents]) => *contents,
        Some(rest) => {
            return Err(crate::PyError::type_error(format!(
                "cell expected at most 1 argument, got {}",
                rest.len()
            )));
        }
    };
    Ok(pyre_object::w_cell_new(contents))
}

/// `nestedscope.py:9-19 make_cell_cmp` / CPython 3.14
/// `cell_compare_impl`: compare contents, with an empty cell ordered before
/// a populated cell. A foreign right operand returns NotImplemented.
fn cell_descr_compare(
    args: &[PyObjectRef],
    op: crate::baseobjspace::CompareOp,
) -> Result<PyObjectRef, crate::PyError> {
    let a = args.first().copied().unwrap_or(PY_NULL);
    let b = args.get(1).copied().unwrap_or(PY_NULL);
    if a.is_null() || !unsafe { pyre_object::is_cell(a) } {
        let dunder = match op {
            crate::baseobjspace::CompareOp::Eq => "__eq__",
            crate::baseobjspace::CompareOp::Ne => "__ne__",
            crate::baseobjspace::CompareOp::Lt => "__lt__",
            crate::baseobjspace::CompareOp::Le => "__le__",
            crate::baseobjspace::CompareOp::Gt => "__gt__",
            crate::baseobjspace::CompareOp::Ge => "__ge__",
        };
        return Err(crate::PyError::type_error(format!(
            "descriptor '{}' requires a 'cell' object",
            dunder
        )));
    }
    if b.is_null() || !unsafe { pyre_object::is_cell(b) } {
        return Ok(pyre_object::w_not_implemented());
    }
    let a_value = unsafe { pyre_object::w_cell_get(a) };
    let b_value = unsafe { pyre_object::w_cell_get(b) };
    if !a_value.is_null() && !b_value.is_null() {
        return crate::baseobjspace::compare(a_value, b_value, op);
    }
    // Cell._cmp_one_empty: empty/empty = 0, empty/value = -1,
    // value/empty = 1; compare that result with zero using the requested op.
    let ordering = match (a_value.is_null(), b_value.is_null()) {
        (true, true) => 0,
        (true, false) => -1,
        (false, true) => 1,
        (false, false) => unreachable!(),
    };
    Ok(pyre_object::w_bool_from(match op {
        crate::baseobjspace::CompareOp::Eq => ordering == 0,
        crate::baseobjspace::CompareOp::Ne => ordering != 0,
        crate::baseobjspace::CompareOp::Lt => ordering < 0,
        crate::baseobjspace::CompareOp::Le => ordering <= 0,
        crate::baseobjspace::CompareOp::Gt => ordering > 0,
        crate::baseobjspace::CompareOp::Ge => ordering >= 0,
    }))
}

fn cell_descr_eq(args: &[PyObjectRef]) -> crate::PyResult {
    cell_descr_compare(args, crate::baseobjspace::CompareOp::Eq)
}
fn cell_descr_ne(args: &[PyObjectRef]) -> crate::PyResult {
    cell_descr_compare(args, crate::baseobjspace::CompareOp::Ne)
}
fn cell_descr_lt(args: &[PyObjectRef]) -> crate::PyResult {
    cell_descr_compare(args, crate::baseobjspace::CompareOp::Lt)
}
fn cell_descr_gt(args: &[PyObjectRef]) -> crate::PyResult {
    cell_descr_compare(args, crate::baseobjspace::CompareOp::Gt)
}
fn cell_descr_le(args: &[PyObjectRef]) -> crate::PyResult {
    cell_descr_compare(args, crate::baseobjspace::CompareOp::Le)
}
fn cell_descr_ge(args: &[PyObjectRef]) -> crate::PyResult {
    cell_descr_compare(args, crate::baseobjspace::CompareOp::Ge)
}

/// `nestedscope.py:101-110 Cell.descr__repr__`, with CPython 3.14's
/// 80-character type-name cap from `Objects/cellobject.c:cell_repr`.
fn cell_descr_repr(args: &[PyObjectRef]) -> crate::PyResult {
    let cell = args.first().copied().unwrap_or(PY_NULL);
    if cell.is_null() || !unsafe { pyre_object::is_cell(cell) } {
        let received = crate::baseobjspace::object_functionstr_type_name(cell);
        return Err(crate::PyError::type_error(format!(
            "descriptor '__repr__' requires a 'cell' object but received a '{received}'"
        )));
    }
    let value = unsafe { pyre_object::w_cell_get(cell) };
    let text = if value.is_null() {
        format!("<cell at 0x{:x}: empty>", cell as usize)
    } else {
        let type_name = crate::typedef::r#type(value)
            .map(|tp| unsafe { pyre_object::w_type_get_name(tp) })
            .unwrap_or_else(|| unsafe { (*(*value).ob_type).name });
        let type_name: String = type_name.chars().take(80).collect();
        format!(
            "<cell at 0x{:x}: {type_name} object at 0x{:x}>",
            cell as usize, value as usize
        )
    };
    Ok(w_str_new(&text))
}

/// `nestedscope.py:934-952 Cell.typedef`, in source order. CPython 3.14 is
/// the version oracle where it differs: its public cell type deliberately
/// omits PyPy 3.11's `__reduce__` and `__setstate__` pickle hooks.
fn init_cell_type(ns: PyObjectRef) {
    let entries = [
        (
            "__doc__",
            w_str_new(
                "Create a new cell object.\n\n  contents\n    the contents of the cell. If not specified, the cell will be empty,\n    and \n further attempts to access its cell_contents attribute will\n    raise a ValueError.",
            ),
        ),
        ("__new__", make_new_descr(cell_descr_new)),
        (
            "__eq__",
            make_builtin_function_with_arity("__eq__", cell_descr_eq, 2),
        ),
        (
            "__ne__",
            make_builtin_function_with_arity("__ne__", cell_descr_ne, 2),
        ),
        (
            "__lt__",
            make_builtin_function_with_arity("__lt__", cell_descr_lt, 2),
        ),
        (
            "__gt__",
            make_builtin_function_with_arity("__gt__", cell_descr_gt, 2),
        ),
        (
            "__le__",
            make_builtin_function_with_arity("__le__", cell_descr_le, 2),
        ),
        (
            "__ge__",
            make_builtin_function_with_arity("__ge__", cell_descr_ge, 2),
        ),
        ("__hash__", w_none()),
        (
            "__repr__",
            make_builtin_function_with_arity("__repr__", cell_descr_repr, 1),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
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
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "cell_contents",
            make_getset_property_named(
                cell_contents_getter,
                cell_contents_setter,
                cell_contents_deleter,
                "cell_contents",
            ),
        )
    };
}

fn staticmethod_require(obj: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    if obj.is_null() || !unsafe { pyre_object::function::is_staticmethod(obj) } {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' requires a 'staticmethod' object"
        )));
    }
    Ok(obj)
}

/// function.py:695-698 `StaticMethod.descr_staticmethod__new__` / CPython
/// 3.14 `sm_new`: allocate first with a None callable; `__init__` installs
/// the user argument and copies presentation attributes.
fn staticmethod_descr_new(args: &[PyObjectRef]) -> crate::PyResult {
    let cls = args.first().copied().unwrap_or(PY_NULL);
    let staticmethod_type = gettypeobject(&pyre_object::function::STATICMETHOD_TYPE);
    check_user_subclass(staticmethod_type, cls)?;
    let sm = pyre_object::function::w_staticmethod_new(w_none());
    if !std::ptr::eq(cls, staticmethod_type) {
        unsafe { (*sm).w_class = cls };
    }
    Ok(sm)
}

/// function.py:700-703 `StaticMethod.descr_init`, adjusted to CPython 3.14:
/// `functools_wraps` copies the four presentation attributes while
/// `__annotations__` and `__annotate__` remain lazy proxy descriptors.
fn staticmethod_descr_init(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "staticmethod() takes no keyword arguments",
        ));
    }
    let sm = staticmethod_require(positional.first().copied().unwrap_or(PY_NULL), "__init__")?;
    let supplied = positional.len().saturating_sub(1);
    if supplied != 1 {
        return Err(crate::PyError::type_error(format!(
            "staticmethod expected 1 argument, got {supplied}"
        )));
    }
    let function = positional[1];
    unsafe { pyre_object::function::w_staticmethod_set_func(sm, function) };
    let w_dict = unsafe { pyre_object::function::w_staticmethod_getdict(sm) };
    for name in ["__module__", "__name__", "__qualname__", "__doc__"] {
        match crate::baseobjspace::getattr_str(function, name) {
            Ok(value) => {
                crate::baseobjspace::setitem(w_dict, w_str_new(name), value)?;
            }
            Err(err) if err.kind == crate::PyErrorKind::AttributeError => {}
            Err(err) => return Err(err),
        }
    }
    Ok(w_none())
}

/// function.py:691-693 `descr_staticmethod_get`.
fn staticmethod_descr_get(args: &[PyObjectRef]) -> crate::PyResult {
    let sm = staticmethod_require(args.first().copied().unwrap_or(PY_NULL), "__get__")?;
    let function = unsafe { pyre_object::function::w_staticmethod_get_func(sm) };
    Ok(if function.is_null() {
        w_none()
    } else {
        function
    })
}

/// function.py:712-713 `descr_call` / CPython 3.14 `sm_call`.
fn staticmethod_descr_call(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let sm = staticmethod_require(positional.first().copied().unwrap_or(PY_NULL), "__call__")?;
    let function = unsafe { pyre_object::function::w_staticmethod_get_func(sm) };
    let call_args = positional.get(1..).unwrap_or(&[]);
    if !crate::builtins::has_real_kwargs(kwargs) {
        return crate::call::call_function_impl_result(function, call_args);
    }
    let keyword_args: Vec<(Wtf8Buf, PyObjectRef)> = unsafe {
        pyre_object::w_dict_str_entries(kwargs.unwrap())
            .into_iter()
            .filter(|(name, _)| name != "__pyre_kw__")
            .map(|(name, value)| (Wtf8Buf::from_string(name), value))
            .collect()
    };
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "staticmethod call has no current frame",
            ));
        }
        crate::call::call_with_kwargs(unsafe { &mut *frame }, function, call_args, &keyword_args)
    })
}

fn staticmethod_func_attr(args: &[PyObjectRef]) -> crate::PyResult {
    let sm = staticmethod_require(args.get(1).copied().unwrap_or(PY_NULL), "__func__")?;
    let value = unsafe { pyre_object::function::w_staticmethod_get_func(sm) };
    Ok(if value.is_null() { w_none() } else { value })
}

fn staticmethod_isabstract(args: &[PyObjectRef]) -> crate::PyResult {
    let sm = staticmethod_require(
        args.get(1).copied().unwrap_or(PY_NULL),
        "__isabstractmethod__",
    )?;
    let function = unsafe { pyre_object::function::w_staticmethod_get_func(sm) };
    Ok(w_bool_from(crate::baseobjspace::isabstractmethod_w(
        function,
    )?))
}

fn staticmethod_wrapped_attr_get(obj: PyObjectRef, name: &str) -> crate::PyResult {
    let sm = staticmethod_require(obj, name)?;
    let w_dict = unsafe { pyre_object::function::w_staticmethod_getdict(sm) };
    if let Some(value) = crate::baseobjspace::finditem_str(w_dict, name)? {
        return Ok(value);
    }
    let function = unsafe { pyre_object::function::w_staticmethod_get_func(sm) };
    let value = crate::baseobjspace::getattr_str(function, name)?;
    crate::baseobjspace::setitem(w_dict, w_str_new(name), value)?;
    Ok(value)
}

fn staticmethod_annotations_get(args: &[PyObjectRef]) -> crate::PyResult {
    staticmethod_wrapped_attr_get(args.get(1).copied().unwrap_or(PY_NULL), "__annotations__")
}

fn staticmethod_annotate_get(args: &[PyObjectRef]) -> crate::PyResult {
    staticmethod_wrapped_attr_get(args.get(1).copied().unwrap_or(PY_NULL), "__annotate__")
}

fn staticmethod_wrapped_attr_set(args: &[PyObjectRef], name: &str) -> crate::PyResult {
    let sm = staticmethod_require(args.get(1).copied().unwrap_or(PY_NULL), name)?;
    let value = args.get(2).copied().unwrap_or(PY_NULL);
    let w_dict = unsafe { pyre_object::function::w_staticmethod_getdict(sm) };
    crate::baseobjspace::setitem(w_dict, w_str_new(name), value)?;
    Ok(w_none())
}

fn staticmethod_annotations_set(args: &[PyObjectRef]) -> crate::PyResult {
    staticmethod_wrapped_attr_set(args, "__annotations__")
}

fn staticmethod_annotate_set(args: &[PyObjectRef]) -> crate::PyResult {
    staticmethod_wrapped_attr_set(args, "__annotate__")
}

fn staticmethod_wrapped_attr_del(args: &[PyObjectRef], name: &str) -> crate::PyResult {
    let sm = staticmethod_require(args.get(1).copied().unwrap_or(PY_NULL), name)?;
    let w_dict = unsafe { pyre_object::function::w_staticmethod_getdict(sm) };
    if let Err(err) = crate::baseobjspace::delitem(w_dict, w_str_new(name)) {
        if err.kind == crate::PyErrorKind::KeyError {
            return Err(crate::PyError::attribute_error(format!(
                "'staticmethod' object has no attribute '{name}'"
            )));
        }
        return Err(err);
    }
    Ok(w_none())
}

fn staticmethod_annotations_del(args: &[PyObjectRef]) -> crate::PyResult {
    staticmethod_wrapped_attr_del(args, "__annotations__")
}

fn staticmethod_annotate_del(args: &[PyObjectRef]) -> crate::PyResult {
    staticmethod_wrapped_attr_del(args, "__annotate__")
}

fn staticmethod_dict_del(_args: &[PyObjectRef]) -> crate::PyResult {
    Err(crate::PyError::type_error("cannot delete __dict__"))
}

/// function.py:715-716 / CPython 3.14 `sm_repr`.
fn staticmethod_descr_repr(args: &[PyObjectRef]) -> crate::PyResult {
    let sm = staticmethod_require(args.first().copied().unwrap_or(PY_NULL), "__repr__")?;
    let function = unsafe { pyre_object::function::w_staticmethod_get_func(sm) };
    let repr = if function.is_null() {
        "<NULL>".to_string()
    } else {
        unsafe { crate::display::py_repr(function)? }
    };
    Ok(w_str_new(&format!("<staticmethod({repr})>")))
}

/// PyPy `typedef.py:852-877 StaticMethod.typedef`, with the CPython 3.14
/// surface taking precedence: PEP 649 proxy descriptors and generic alias
/// support are present, while PyPy 3.11's `__reduce_ex__` is absent.
fn init_staticmethod_type(ns: PyObjectRef) {
    let dict_getter = make_builtin_function_with_arity("__dict__", descr_get_dict, 2);
    let dict_setter = make_builtin_function_with_arity("__dict__", descr_set_dict, 3);
    let dict_deleter = make_builtin_function_with_arity("__dict__", staticmethod_dict_del, 2);
    let annotations_getter =
        make_builtin_function_with_arity("__annotations__", staticmethod_annotations_get, 2);
    let annotations_setter =
        make_builtin_function_with_arity("__annotations__", staticmethod_annotations_set, 3);
    let annotations_deleter =
        make_builtin_function_with_arity("__annotations__", staticmethod_annotations_del, 2);
    let annotate_getter =
        make_builtin_function_with_arity("__annotate__", staticmethod_annotate_get, 2);
    let annotate_setter =
        make_builtin_function_with_arity("__annotate__", staticmethod_annotate_set, 3);
    let annotate_deleter =
        make_builtin_function_with_arity("__annotate__", staticmethod_annotate_del, 2);
    let entries = [
        (
            "__doc__",
            w_str_new(
                "Convert a function to be a static method.\n\nA static method does not receive an implicit first argument.\nTo declare a static method, use this idiom:\n\n     class C:\n         @staticmethod\n         def f(arg1, arg2, argN):\n             ...\n\nIt can be called either on the class (e.g. C.f()) or on an instance\n(e.g. C().f()). Both the class and the instance are ignored, and\nneither is passed implicitly as the first argument to the method.\n\nStatic methods in Python are similar to those found in Java or C++.\nFor a more advanced concept, see the classmethod builtin.",
            ),
        ),
        (
            "__get__",
            make_builtin_function("__get__", staticmethod_descr_get),
        ),
        ("__new__", make_new_descr(staticmethod_descr_new)),
        (
            "__init__",
            make_builtin_function("__init__", staticmethod_descr_init),
        ),
        (
            "__call__",
            make_builtin_function("__call__", staticmethod_descr_call),
        ),
        (
            "__func__",
            make_getset_descriptor(make_builtin_function_with_arity(
                "__func__",
                staticmethod_func_attr,
                2,
            )),
        ),
        (
            "__wrapped__",
            make_getset_descriptor(make_builtin_function_with_arity(
                "__wrapped__",
                staticmethod_func_attr,
                2,
            )),
        ),
        (
            "__isabstractmethod__",
            make_getset_descriptor(make_builtin_function_with_arity(
                "__isabstractmethod__",
                staticmethod_isabstract,
                2,
            )),
        ),
        (
            "__dict__",
            make_getset_property_named(dict_getter, dict_setter, dict_deleter, "__dict__"),
        ),
        (
            "__annotations__",
            make_getset_property_named(
                annotations_getter,
                annotations_setter,
                annotations_deleter,
                "__annotations__",
            ),
        ),
        (
            "__annotate__",
            make_getset_property_named(
                annotate_getter,
                annotate_setter,
                annotate_deleter,
                "__annotate__",
            ),
        ),
        (
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                crate::_pypy_generic_alias::generic_alias_class_getitem,
            )),
        ),
        (
            "__repr__",
            make_builtin_function("__repr__", staticmethod_descr_repr),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn classmethod_require(obj: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    if obj.is_null() || !unsafe { pyre_object::function::is_classmethod(obj) } {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' requires a 'classmethod' object"
        )));
    }
    Ok(obj)
}

/// function.py:750-753 `ClassMethod.descr_classmethod__new__` / CPython
/// 3.14 `cm_new`: allocate the requested subtype with a temporary None
/// callable; `__init__` installs the actual callable.
fn classmethod_descr_new(args: &[PyObjectRef]) -> crate::PyResult {
    let cls = args.first().copied().unwrap_or(PY_NULL);
    let classmethod_type = gettypeobject(&pyre_object::function::CLASSMETHOD_TYPE);
    check_user_subclass(classmethod_type, cls)?;
    let cm = pyre_object::function::w_classmethod_new(w_none());
    if !std::ptr::eq(cls, classmethod_type) {
        unsafe { (*cm).w_class = cls };
    }
    Ok(cm)
}

/// function.py:755-758 `ClassMethod.descr_init`, adjusted to CPython 3.14's
/// `functools_wraps`: copy the four eager presentation attributes while the
/// two annotation attributes remain lazy proxy descriptors.
fn classmethod_descr_init(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "classmethod() takes no keyword arguments",
        ));
    }
    let cm = classmethod_require(positional.first().copied().unwrap_or(PY_NULL), "__init__")?;
    let supplied = positional.len().saturating_sub(1);
    if supplied != 1 {
        return Err(crate::PyError::type_error(format!(
            "classmethod expected 1 argument, got {supplied}"
        )));
    }
    let function = positional[1];
    unsafe { pyre_object::function::w_classmethod_set_func(cm, function) };
    let w_dict = unsafe { pyre_object::function::w_classmethod_getdict(cm) };
    for name in ["__module__", "__name__", "__qualname__", "__doc__"] {
        match crate::baseobjspace::getattr_str(function, name) {
            Ok(value) => {
                crate::baseobjspace::setitem(w_dict, w_str_new(name), value)?;
            }
            Err(err) if err.kind == crate::PyErrorKind::AttributeError => {}
            Err(err) => return Err(err),
        }
    }
    Ok(w_none())
}

/// function.py:738-748 `descr_classmethod_get`. Python 3.14's `cm_descr_get`
/// binds the stored callable directly to the selected class; it no longer
/// invokes a descriptor nested inside classmethod.
fn classmethod_descr_get(args: &[PyObjectRef]) -> crate::PyResult {
    let cm = classmethod_require(args.first().copied().unwrap_or(PY_NULL), "__get__")?;
    let w_obj = args.get(1).copied().unwrap_or(PY_NULL);
    let mut w_klass = args.get(2).copied().unwrap_or(PY_NULL);
    if w_klass.is_null() || unsafe { pyre_object::is_none(w_klass) } {
        if w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) } {
            return Err(crate::PyError::type_error("__get__(None, None) is invalid"));
        }
        w_klass = r#type(w_obj).unwrap_or(PY_NULL);
    }
    let function = unsafe { pyre_object::function::w_classmethod_get_func(cm) };
    Ok(pyre_object::w_method_new(function, w_klass, w_klass))
}

fn classmethod_func_attr(args: &[PyObjectRef]) -> crate::PyResult {
    let cm = classmethod_require(args.get(1).copied().unwrap_or(PY_NULL), "__func__")?;
    let value = unsafe { pyre_object::function::w_classmethod_get_func(cm) };
    Ok(if value.is_null() { w_none() } else { value })
}

fn classmethod_isabstract(args: &[PyObjectRef]) -> crate::PyResult {
    let cm = classmethod_require(
        args.get(1).copied().unwrap_or(PY_NULL),
        "__isabstractmethod__",
    )?;
    let function = unsafe { pyre_object::function::w_classmethod_get_func(cm) };
    Ok(w_bool_from(crate::baseobjspace::isabstractmethod_w(
        function,
    )?))
}

fn classmethod_wrapped_attr_get(obj: PyObjectRef, name: &str) -> crate::PyResult {
    let cm = classmethod_require(obj, name)?;
    let w_dict = unsafe { pyre_object::function::w_classmethod_getdict(cm) };
    if let Some(value) = crate::baseobjspace::finditem_str(w_dict, name)? {
        return Ok(value);
    }
    let function = unsafe { pyre_object::function::w_classmethod_get_func(cm) };
    let value = crate::baseobjspace::getattr_str(function, name)?;
    crate::baseobjspace::setitem(w_dict, w_str_new(name), value)?;
    Ok(value)
}

fn classmethod_annotations_get(args: &[PyObjectRef]) -> crate::PyResult {
    classmethod_wrapped_attr_get(args.get(1).copied().unwrap_or(PY_NULL), "__annotations__")
}

fn classmethod_annotate_get(args: &[PyObjectRef]) -> crate::PyResult {
    classmethod_wrapped_attr_get(args.get(1).copied().unwrap_or(PY_NULL), "__annotate__")
}

fn classmethod_wrapped_attr_set(args: &[PyObjectRef], name: &str) -> crate::PyResult {
    let cm = classmethod_require(args.get(1).copied().unwrap_or(PY_NULL), name)?;
    let value = args.get(2).copied().unwrap_or(PY_NULL);
    let w_dict = unsafe { pyre_object::function::w_classmethod_getdict(cm) };
    crate::baseobjspace::setitem(w_dict, w_str_new(name), value)?;
    Ok(w_none())
}

fn classmethod_annotations_set(args: &[PyObjectRef]) -> crate::PyResult {
    classmethod_wrapped_attr_set(args, "__annotations__")
}

fn classmethod_annotate_set(args: &[PyObjectRef]) -> crate::PyResult {
    classmethod_wrapped_attr_set(args, "__annotate__")
}

fn classmethod_wrapped_attr_del(args: &[PyObjectRef], name: &str) -> crate::PyResult {
    let cm = classmethod_require(args.get(1).copied().unwrap_or(PY_NULL), name)?;
    let w_dict = unsafe { pyre_object::function::w_classmethod_getdict(cm) };
    if let Err(err) = crate::baseobjspace::delitem(w_dict, w_str_new(name)) {
        if err.kind == crate::PyErrorKind::KeyError {
            return Err(crate::PyError::attribute_error(format!(
                "'classmethod' object has no attribute '{name}'"
            )));
        }
        return Err(err);
    }
    Ok(w_none())
}

fn classmethod_annotations_del(args: &[PyObjectRef]) -> crate::PyResult {
    classmethod_wrapped_attr_del(args, "__annotations__")
}

fn classmethod_annotate_del(args: &[PyObjectRef]) -> crate::PyResult {
    classmethod_wrapped_attr_del(args, "__annotate__")
}

fn classmethod_dict_del(_args: &[PyObjectRef]) -> crate::PyResult {
    Err(crate::PyError::type_error("cannot delete __dict__"))
}

/// function.py:767-768 `descr_repr` / CPython 3.14 `cm_repr`.
fn classmethod_descr_repr(args: &[PyObjectRef]) -> crate::PyResult {
    let cm = classmethod_require(args.first().copied().unwrap_or(PY_NULL), "__repr__")?;
    let function = unsafe { pyre_object::function::w_classmethod_get_func(cm) };
    let repr = if function.is_null() {
        "<NULL>".to_string()
    } else {
        unsafe { crate::display::py_repr(function)? }
    };
    Ok(w_str_new(&format!("<classmethod({repr})>")))
}

/// PyPy `typedef.py:878-908 ClassMethod.typedef`, with the Python 3.14
/// surface taking precedence: PEP 649 proxy descriptors and generic alias
/// support replace PyPy 3.11's eager annotations copy and `__reduce_ex__`.
fn init_classmethod_type(ns: PyObjectRef) {
    let dict_getter = make_builtin_function_with_arity("__dict__", descr_get_dict, 2);
    let dict_setter = make_builtin_function_with_arity("__dict__", descr_set_dict, 3);
    let dict_deleter = make_builtin_function_with_arity("__dict__", classmethod_dict_del, 2);
    let annotations_getter =
        make_builtin_function_with_arity("__annotations__", classmethod_annotations_get, 2);
    let annotations_setter =
        make_builtin_function_with_arity("__annotations__", classmethod_annotations_set, 3);
    let annotations_deleter =
        make_builtin_function_with_arity("__annotations__", classmethod_annotations_del, 2);
    let annotate_getter =
        make_builtin_function_with_arity("__annotate__", classmethod_annotate_get, 2);
    let annotate_setter =
        make_builtin_function_with_arity("__annotate__", classmethod_annotate_set, 3);
    let annotate_deleter =
        make_builtin_function_with_arity("__annotate__", classmethod_annotate_del, 2);
    let entries = [
        (
            "__doc__",
            w_str_new(
                "Convert a function to be a class method.\n\nA class method receives the class as implicit first argument,\njust like an instance method receives the instance.\nTo declare a class method, use this idiom:\n\n  class C:\n      @classmethod\n      def f(cls, arg1, arg2, argN):\n          ...\n\nIt can be called either on the class (e.g. C.f()) or on an instance\n(e.g. C().f()).  The instance is ignored except for its class.\nIf a class method is called for a derived class, the derived class\nobject is passed as the implied first argument.\n\nClass methods are different than C++ or Java static methods.\nIf you want those, see the staticmethod builtin.",
            ),
        ),
        (
            "__get__",
            make_builtin_function("__get__", classmethod_descr_get),
        ),
        ("__new__", make_new_descr(classmethod_descr_new)),
        (
            "__init__",
            make_builtin_function("__init__", classmethod_descr_init),
        ),
        (
            "__func__",
            make_getset_descriptor(make_builtin_function_with_arity(
                "__func__",
                classmethod_func_attr,
                2,
            )),
        ),
        (
            "__wrapped__",
            make_getset_descriptor(make_builtin_function_with_arity(
                "__wrapped__",
                classmethod_func_attr,
                2,
            )),
        ),
        (
            "__isabstractmethod__",
            make_getset_descriptor(make_builtin_function_with_arity(
                "__isabstractmethod__",
                classmethod_isabstract,
                2,
            )),
        ),
        (
            "__dict__",
            make_getset_property_named(dict_getter, dict_setter, dict_deleter, "__dict__"),
        ),
        (
            "__annotations__",
            make_getset_property_named(
                annotations_getter,
                annotations_setter,
                annotations_deleter,
                "__annotations__",
            ),
        ),
        (
            "__annotate__",
            make_getset_property_named(
                annotate_getter,
                annotate_setter,
                annotate_deleter,
                "__annotate__",
            ),
        ),
        (
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                crate::_pypy_generic_alias::generic_alias_class_getitem,
            )),
        ),
        (
            "__repr__",
            make_builtin_function("__repr__", classmethod_descr_repr),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

/// CPython 3.14 `PyFunction_Type.tp_doc`. The instance `__doc__` getset
/// occupies the function type dictionary's key, so exact type access is
/// served separately by `baseobjspace`, as for property.
pub(crate) const FUNCTION_DOC: &str = r#"Create a function object.

  code
    a code object
  globals
    the globals dictionary
  name
    a string that overrides the name from the code object
  argdefs
    a tuple that specifies the default argument values
  closure
    a tuple that supplies the bindings for free variables
  kwdefaults
    a dictionary that specifies the default keyword argument values"#;

/// CPython 3.14 `PyMethod_Type.tp_doc`.
pub(crate) const METHOD_DOC: &str = "Create a bound instance method object.";

/// CPython 3.14 `PyProperty_Type.tp_doc`.  The instance-level `__doc__`
/// descriptor occupies the same type-dict key; `baseobjspace` serves this
/// separate type doc for the exact builtin, matching PyPy's TypeDef rawdict
/// replacement and CPython's `tp_doc` + member-descriptor split.
pub(crate) const PROPERTY_DOC: &str = r#"Property attribute.

  fget
    function to be used for getting an attribute value
  fset
    function to be used for setting an attribute value
  fdel
    function to be used for del'ing an attribute
  doc
    docstring

Typical use is to define a managed attribute x:

class C(object):
    def getx(self): return self._x
    def setx(self, value): self._x = value
    def delx(self): del self._x
    x = property(getx, setx, delx, "I'm the 'x' property.")

Decorators make defining new properties or modifying existing ones easy:

class C(object):
    @property
    def x(self):
        "I am the 'x' property."
        return self._x
    @x.setter
    def x(self, value):
        self._x = value
    @x.deleter
    def x(self):
        del self._x"#;

fn property_require(obj: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    if obj.is_null() || !unsafe { pyre_object::descriptor::is_property(obj) } {
        let received = if obj.is_null() {
            "NULL".to_string()
        } else {
            crate::type_methods::arg_type_name(obj)
        };
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' for 'property' objects doesn't apply to a '{received}' object"
        )));
    }
    Ok(obj)
}

/// PyPy `generic_new_descr(W_Property)` / CPython 3.14 `PyType_GenericNew`:
/// allocate the requested subtype and leave argument processing to `__init__`.
fn property_descr_new(args: &[PyObjectRef]) -> crate::PyResult {
    let cls = args.first().copied().unwrap_or(PY_NULL);
    if cls.is_null() {
        return Err(crate::PyError::type_error(
            "property.__new__(): not enough arguments",
        ));
    }
    let property_type = gettypeobject(&pyre_object::descriptor::PROPERTY_TYPE);
    check_user_subclass(property_type, cls)?;
    let prop = pyre_object::w_property_new(PY_NULL, PY_NULL, PY_NULL);
    if !std::ptr::eq(cls, property_type) {
        unsafe { (*prop).w_class = cls };
    }
    Ok(prop)
}

/// PyPy `W_Property.init`, with CPython 3.14's `prop_name` reset and
/// subclass-doc placement taking precedence where the versions differ.
fn property_descr_init(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if positional.is_empty() {
        return Err(crate::PyError::type_error(
            "descriptor '__init__' of 'property' object needs an argument",
        ));
    }
    let prop = positional[0];
    if !unsafe { pyre_object::descriptor::is_property(prop) } {
        return Err(crate::PyError::type_error(format!(
            "descriptor '__init__' requires a 'property' object but received a '{}'",
            crate::type_methods::arg_type_name(prop),
        )));
    }
    let supplied = positional.len() - 1;
    if supplied > 4 {
        return Err(crate::PyError::type_error(format!(
            "property() takes at most 4 arguments ({supplied} given)"
        )));
    }
    crate::builtins::kwarg_reject_unknown(kwargs, &["fget", "fset", "fdel", "doc"], "property")?;
    let fget = crate::builtins::resolve_pos_or_kw(
        positional.get(1).copied(),
        kwargs,
        "fget",
        "property",
        1,
    )?
    .unwrap_or_else(w_none);
    let fset = crate::builtins::resolve_pos_or_kw(
        positional.get(2).copied(),
        kwargs,
        "fset",
        "property",
        2,
    )?
    .unwrap_or_else(w_none);
    let fdel = crate::builtins::resolve_pos_or_kw(
        positional.get(3).copied(),
        kwargs,
        "fdel",
        "property",
        3,
    )?
    .unwrap_or_else(w_none);
    let w_doc = crate::builtins::resolve_pos_or_kw(
        positional.get(4).copied(),
        kwargs,
        "doc",
        "property",
        4,
    )?
    .unwrap_or_else(w_none);

    unsafe { pyre_object::descriptor::w_property_reinit(prop, fget, fset, fdel) };

    // CPython 3.14 property_init_impl: explicit non-None doc wins; otherwise
    // inherit a non-None getter doc and remember that `_copy` must rederive it.
    let mut getter_doc = false;
    let prop_doc = if !unsafe { pyre_object::is_none(w_doc) } {
        Some(w_doc)
    } else if !unsafe { pyre_object::is_none(fget) } {
        match crate::baseobjspace::getattr_str(fget, "__doc__") {
            Ok(value) if !unsafe { pyre_object::is_none(value) } => {
                getter_doc = true;
                Some(value)
            }
            Ok(_) => None,
            Err(err) if err.kind == crate::PyErrorKind::AttributeError => None,
            Err(err) => return Err(err),
        }
    } else {
        None
    };

    let property_type = gettypeobject(&pyre_object::descriptor::PROPERTY_TYPE);
    let exact = r#type(prop).is_some_and(|tp| std::ptr::eq(tp, property_type));
    if exact {
        if let Some(doc) = prop_doc {
            unsafe {
                if getter_doc {
                    pyre_object::descriptor::w_property_set_getter_doc(prop, doc);
                } else {
                    pyre_object::descriptor::w_property_set_doc(prop, doc);
                }
            }
        }
    } else {
        // CPython 3.14 puts a subclass property's doc in its instance dict or
        // designated slot because the subclass class dict shadows the base
        // descriptor with its own `__doc__` entry.
        let visible_doc = prop_doc.unwrap_or_else(w_none);
        match crate::baseobjspace::setattr_str(prop, "__doc__", visible_doc) {
            Ok(_) => {}
            Err(err) if !getter_doc && err.kind == crate::PyErrorKind::AttributeError => {}
            Err(err) => return Err(err),
        }
        if getter_doc {
            unsafe { pyre_object::descriptor::w_property_mark_getter_doc(prop) };
        }
    }
    Ok(w_none())
}

fn property_fget(args: &[PyObjectRef]) -> crate::PyResult {
    let prop = property_require(args.get(1).copied().unwrap_or(PY_NULL), "fget")?;
    let value = unsafe { pyre_object::descriptor::w_property_get_fget(prop) };
    Ok(if value.is_null() { w_none() } else { value })
}

fn property_fset(args: &[PyObjectRef]) -> crate::PyResult {
    let prop = property_require(args.get(1).copied().unwrap_or(PY_NULL), "fset")?;
    let value = unsafe { pyre_object::descriptor::w_property_get_fset(prop) };
    Ok(if value.is_null() { w_none() } else { value })
}

fn property_fdel(args: &[PyObjectRef]) -> crate::PyResult {
    let prop = property_require(args.get(1).copied().unwrap_or(PY_NULL), "fdel")?;
    let value = unsafe { pyre_object::descriptor::w_property_get_fdel(prop) };
    Ok(if value.is_null() { w_none() } else { value })
}

fn property_doc_get(args: &[PyObjectRef]) -> crate::PyResult {
    let prop = property_require(args.get(1).copied().unwrap_or(PY_NULL), "__doc__")?;
    let value = unsafe { pyre_object::descriptor::w_property_get_doc(prop) };
    Ok(if value.is_null() { w_none() } else { value })
}

fn property_doc_set(args: &[PyObjectRef]) -> crate::PyResult {
    let prop = property_require(args.get(1).copied().unwrap_or(PY_NULL), "__doc__")?;
    let value = args.get(2).copied().unwrap_or(PY_NULL);
    unsafe { pyre_object::descriptor::w_property_set_doc(prop, value) };
    Ok(w_none())
}

fn property_doc_del(args: &[PyObjectRef]) -> crate::PyResult {
    let prop = property_require(args.get(1).copied().unwrap_or(PY_NULL), "__doc__")?;
    unsafe { pyre_object::descriptor::w_property_set_doc(prop, PY_NULL) };
    Ok(w_none())
}

fn property_name_get(args: &[PyObjectRef]) -> crate::PyResult {
    let prop = property_require(args.get(1).copied().unwrap_or(PY_NULL), "__name__")?;
    let stored = unsafe { pyre_object::descriptor::w_property_get_name(prop) };
    if !stored.is_null() {
        return Ok(stored);
    }
    let fget = unsafe { pyre_object::descriptor::w_property_get_fget(prop) };
    if !fget.is_null() && !unsafe { pyre_object::is_none(fget) } {
        match crate::baseobjspace::getattr_str(fget, "__name__") {
            Ok(name) => return Ok(name),
            Err(err) if err.kind == crate::PyErrorKind::AttributeError => {}
            Err(err) => return Err(err),
        }
    }
    Err(crate::PyError::attribute_error(
        "'property' object has no attribute '__name__'",
    ))
}

fn property_name_set(args: &[PyObjectRef]) -> crate::PyResult {
    let prop = property_require(args.get(1).copied().unwrap_or(PY_NULL), "__name__")?;
    let value = args.get(2).copied().unwrap_or(PY_NULL);
    unsafe { pyre_object::descriptor::w_property_set_name(prop, value) };
    Ok(w_none())
}

fn property_name_del(args: &[PyObjectRef]) -> crate::PyResult {
    let prop = property_require(args.get(1).copied().unwrap_or(PY_NULL), "__name__")?;
    unsafe { pyre_object::descriptor::w_property_set_name(prop, PY_NULL) };
    Ok(w_none())
}

fn property_isabstract(args: &[PyObjectRef]) -> crate::PyResult {
    let prop = property_require(
        args.get(1).copied().unwrap_or(PY_NULL),
        "__isabstractmethod__",
    )?;
    let is_abstract = |f: PyObjectRef| -> Result<bool, crate::PyError> {
        if f.is_null() || unsafe { pyre_object::is_none(f) } {
            Ok(false)
        } else {
            crate::baseobjspace::isabstractmethod_w(f)
        }
    };
    let result = is_abstract(unsafe { pyre_object::descriptor::w_property_get_fget(prop) })?
        || is_abstract(unsafe { pyre_object::descriptor::w_property_get_fset(prop) })?
        || is_abstract(unsafe { pyre_object::descriptor::w_property_get_fdel(prop) })?;
    Ok(pyre_object::w_bool_from(result))
}

/// PyPy `W_Property.typedef`, extended only where Python 3.14's public
/// surface differs (`__name__`, `__set_name__`, no PyPy-3.11 `__reduce__`).
fn init_property_type(ns: PyObjectRef) {
    let entries = [
        ("__new__", make_new_descr(property_descr_new)),
        (
            "__init__",
            make_builtin_function("__init__", property_descr_init),
        ),
        (
            "__get__",
            make_builtin_function("__get__", crate::baseobjspace::property_descr_get_impl),
        ),
        (
            "__set__",
            make_builtin_function("__set__", crate::baseobjspace::property_descr_set_impl),
        ),
        (
            "__delete__",
            make_builtin_function_with_arity(
                "__delete__",
                crate::baseobjspace::property_descr_delete_impl,
                2,
            ),
        ),
        (
            "fget",
            make_getset_descriptor(make_builtin_function_with_arity("fget", property_fget, 2)),
        ),
        (
            "fset",
            make_getset_descriptor(make_builtin_function_with_arity("fset", property_fset, 2)),
        ),
        (
            "fdel",
            make_getset_descriptor(make_builtin_function_with_arity("fdel", property_fdel, 2)),
        ),
        (
            "__doc__",
            make_getset_property_named(
                make_builtin_function_with_arity("__doc__", property_doc_get, 2),
                make_builtin_function_with_arity("__doc__", property_doc_set, 3),
                make_builtin_function_with_arity("__doc__", property_doc_del, 2),
                "__doc__",
            ),
        ),
        (
            "__name__",
            make_getset_property_named(
                make_builtin_function_with_arity("__name__", property_name_get, 2),
                make_builtin_function_with_arity("__name__", property_name_set, 3),
                make_builtin_function_with_arity("__name__", property_name_del, 2),
                "__name__",
            ),
        ),
        (
            "__isabstractmethod__",
            make_getset_descriptor(make_builtin_function_with_arity(
                "__isabstractmethod__",
                property_isabstract,
                2,
            )),
        ),
        (
            "getter",
            make_builtin_function_with_arity(
                "getter",
                crate::baseobjspace::property_getter_impl,
                2,
            ),
        ),
        (
            "setter",
            make_builtin_function_with_arity(
                "setter",
                crate::baseobjspace::property_setter_impl,
                2,
            ),
        ),
        (
            "deleter",
            make_builtin_function_with_arity(
                "deleter",
                crate::baseobjspace::property_deleter_impl,
                2,
            ),
        ),
        (
            "__set_name__",
            make_builtin_function_with_arity(
                "__set_name__",
                crate::baseobjspace::property_set_name_impl,
                3,
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

/// `self` as a plain int — `int.real` / `numerator` / `conjugate` /
/// `as_integer_ratio` and the integer conversion dunders return the
/// integer value, so a non-exact receiver is down-converted.
fn int_as_plain_int(args: &[PyObjectRef]) -> PyObjectRef {
    let obj = args.first().copied().unwrap_or(pyre_object::w_int_new(0));
    unsafe {
        if pyre_object::is_bool(obj) {
            return pyre_object::w_int_new(pyre_object::w_bool_get_value(obj) as i64);
        }
        if pyre_object::is_int(obj)
            && !(pyre_object::tagged_int::CAN_BE_TAGGED
                && pyre_object::tagged_int::is_tagged_int(obj))
            && (*obj).w_class != pyre_object::get_instantiate(&pyre_object::INT_TYPE)
        {
            return pyre_object::w_int_new(pyre_object::w_int_get_value(obj));
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
    // intobject.py:674 descr_pow — a non-int exponent defers to the other
    // operand's reflected slot.
    if !unsafe { pyre_object::pyobject::is_int_or_long(args[1]) } {
        return Ok(pyre_object::w_not_implemented());
    }
    if args.len() >= 3 && !unsafe { pyre_object::pyobject::is_none(args[2]) } {
        // Only an integer modulus routes through the modular-power path.
        // The object protocol has no ternary reflected slot, so an int base
        // with a non-int modulus yields NotImplemented and the caller raises
        // the three-operand type error.
        if !unsafe { pyre_object::pyobject::is_int_or_long(args[2]) } {
            return Ok(pyre_object::w_not_implemented());
        }
        // intobject.py:686 — self, exponent and modulus are all integers, so
        // compute the modular power here rather than re-entering the ternary
        // dispatch (which would recurse back into this slot).
        return match crate::objspace::descroperation::try_int_long_pow_with_modulo(
            args[0], args[1], args[2],
        )? {
            Some(result) => Ok(result),
            None => Ok(pyre_object::w_not_implemented()),
        };
    }
    crate::objspace::descroperation::pow_builtin(args[0], args[1])
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
/// floatobject.py:588 — a float `__pow__`/`__rpow__` rejects a ternary
/// modulus argument, which is meaningful only for integer power.
fn float_pow_reject_modulus(args: &[PyObjectRef]) -> Result<(), crate::PyError> {
    if args.len() >= 3 && !unsafe { pyre_object::pyobject::is_none(args[2]) } {
        return Err(crate::PyError::type_error(
            "pow() 3rd argument not allowed unless all arguments are integers",
        ));
    }
    Ok(())
}

/// `float.__pow__` / `__rpow__` — the ternary-power slot accepts an
/// optional modulus argument, so arity is validated as one-or-two.
fn float_dunder_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_pow(args)?;
    let b = args[1];
    if unsafe { pyre_object::pyobject::is_float(b) || pyre_object::pyobject::is_int_or_long(b) } {
        // The operand is coerced to a double first, so an over-range int
        // raises OverflowError before the ternary modulus is rejected.
        unsafe { crate::objspace::descroperation::reject_pow_operand_overflow(b)? };
        float_pow_reject_modulus(args)?;
        crate::objspace::descroperation::pow_builtin(args[0], b)
    } else {
        Ok(pyre_object::w_not_implemented())
    }
}
fn float_dunder_rpow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_pow(args)?;
    let b = args[1];
    if unsafe { pyre_object::pyobject::is_float(b) || pyre_object::pyobject::is_int_or_long(b) } {
        unsafe { crate::objspace::descroperation::reject_pow_operand_overflow(b)? };
        float_pow_reject_modulus(args)?;
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
/// complexobject.py:525 — a complex `__pow__`/`__rpow__` rejects a ternary
/// modulus argument with `ValueError: complex modulo`.
fn complex_pow_reject_modulus(args: &[PyObjectRef]) -> Result<(), crate::PyError> {
    if args.len() >= 3 && !unsafe { pyre_object::pyobject::is_none(args[2]) } {
        return Err(crate::PyError::value_error("complex modulo"));
    }
    Ok(())
}

/// `complex.__pow__` / `__rpow__` — the ternary-power slot accepts an
/// optional modulus argument, so arity is validated as one-or-two.
fn complex_dunder_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_pow(args)?;
    if complex_binop_operand(args[1]) {
        complex_pow_reject_modulus(args)?;
        crate::objspace::descroperation::pow_builtin(args[0], args[1])
    } else {
        Ok(pyre_object::w_not_implemented())
    }
}
fn complex_dunder_rpow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_pow(args)?;
    if complex_binop_operand(args[1]) {
        complex_pow_reject_modulus(args)?;
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
macro_rules! list_cmp_dunder {
    ($function:ident, $name:literal, $op:ident) => {
        fn $function(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::require_list_receiver(args, $name, false)?;
            crate::type_methods::arity_slot(args, 1)?;
            if cmp_guard_list(args[1]) {
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
list_cmp_dunder!(list_dunder_eq, "__eq__", Eq);
list_cmp_dunder!(list_dunder_ne, "__ne__", Ne);
list_cmp_dunder!(list_dunder_lt, "__lt__", Lt);
list_cmp_dunder!(list_dunder_le, "__le__", Le);
list_cmp_dunder!(list_dunder_gt, "__gt__", Gt);
list_cmp_dunder!(list_dunder_ge, "__ge__", Ge);
macro_rules! tuple_cmp_dunder {
    ($function:ident, $name:literal, $op:ident) => {
        fn $function(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::require_tuple_receiver(args, $name, false)?;
            crate::type_methods::arity_slot(args, 1)?;
            if cmp_guard_tuple(args[1]) {
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
tuple_cmp_dunder!(tuple_dunder_eq, "__eq__", Eq);
tuple_cmp_dunder!(tuple_dunder_ne, "__ne__", Ne);
tuple_cmp_dunder!(tuple_dunder_lt, "__lt__", Lt);
tuple_cmp_dunder!(tuple_dunder_le, "__le__", Le);
tuple_cmp_dunder!(tuple_dunder_gt, "__gt__", Gt);
tuple_cmp_dunder!(tuple_dunder_ge, "__ge__", Ge);
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

fn init_int_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new(
                "int([x]) -> integer\n\
                 int(x, base=10) -> integer\n\n\
                 Convert a number or string to an integer, or return 0 if no arguments\n\
                 are given.  If x is a number, return x.__int__().  For floating-point\n\
                 numbers, this truncates towards zero.\n\n\
                 If x is not a number or if base is given, then x must be a string,\n\
                 bytes, or bytearray instance representing an integer literal in the\n\
                 given base.  The literal can be preceded by '+' or '-' and be surrounded\n\
                 by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.\n\
                 Base 0 means to interpret the base from the string as an integer literal.",
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(int_descr_new),
        )
    };
    // intobject.py descr_repr. CPython 3.14 inherits object.__str__, whose
    // implementation delegates virtually to this repr slot.
    let int_to_text = |args: &[PyObjectRef]| {
        Ok(pyre_object::w_str_new(&unsafe {
            crate::builtins::int_to_decimal_string(args[0])?
        }))
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity("__repr__", int_to_text, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| Ok(w_int_new(crate::builtins::hash_value(args[0]))),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__sizeof__",
            make_builtin_function_with_arity(
                "__sizeof__",
                |args| {
                    let bits = unsafe { crate::builtins::obj_to_bigint(args[0]).bits() } as usize;
                    // CPython 3.14's compact PyLong layout: three pointer-sized
                    // header words and at least one 30-bit, four-byte digit.
                    let digits = std::cmp::max(1, (bits + 29) / 30);
                    Ok(w_int_new(
                        (3 * std::mem::size_of::<usize>() + digits * 4) as i64,
                    ))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "is_integer",
            make_builtin_function_with_arity(
                "is_integer",
                |_| Ok(pyre_object::w_bool_from(true)),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // int.to_bytes(length=1, byteorder='big', *, signed=False)
    // PyPy: longobject.py descr_to_bytes
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                let val = if !pos.is_empty()
                    && unsafe { pyre_object::pyobject::is_int_or_long(pos[0]) }
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
                            unsafe { pyre_object::type_name_of(o) }
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
        )
    };
    // int.from_bytes(bytes, byteorder='big', *, signed=False) — classmethod.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "from_bytes",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "from_bytes",
                int_from_bytes,
            )),
        )
    };
    // int.__index__ / __int__ / __trunc__ — exact ints preserve identity;
    // subclasses and bools are normalized by `int_as_plain_int`.
    for method in ["__index__", "__int__", "__trunc__"] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                method,
                make_builtin_function_with_arity(method, |args| Ok(int_as_plain_int(args)), 1),
            )
        };
    }
    // int.conjugate — identity (bool → int)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "conjugate",
            make_builtin_function_with_arity("conjugate", |args| Ok(int_as_plain_int(args)), 1),
        )
    };
    // int.as_integer_ratio — (self, 1)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // int.real / int.imag / int.numerator — properties
    // True.real → 1 (int, not bool), False.real → 0
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "real",
            pyre_object::w_property_new(
                make_builtin_function_with_arity("real", |args| Ok(int_as_plain_int(args)), 1),
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "imag",
            pyre_object::w_property_new(
                make_builtin_function_with_arity("imag", |_| Ok(pyre_object::w_int_new(0)), 1),
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "numerator",
            pyre_object::w_property_new(
                make_builtin_function_with_arity("numerator", |args| Ok(int_as_plain_int(args)), 1),
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
            ),
        )
    };
    let denom_getter =
        make_builtin_function_with_arity("denominator", |_| Ok(pyre_object::w_int_new(1)), 1);
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "denominator",
            make_getset_descriptor(denom_getter),
        )
    };
    // Unary / conversion slots exposed as callable dunders.  These have
    // no NotImplemented dispatch, so each delegates to the object-space
    // op, which fast-paths the concrete int (no re-dispatch through the
    // dunder).  Binary arithmetic dunders are registered separately.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__round__",
            make_builtin_function("__round__", crate::builtins::builtin_round),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__format__",
            make_builtin_function_with_arity(
                "__format__",
                crate::type_methods::builtin_value_format,
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__float__",
            make_builtin_function_with_arity("__float__", crate::builtins::builtin_float, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__abs__",
            make_builtin_function_with_arity("__abs__", crate::builtins::builtin_abs, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__neg__",
            make_builtin_function_with_arity(
                "__neg__",
                |args| crate::objspace::descroperation::neg(args[0]),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__pos__",
            make_builtin_function_with_arity(
                "__pos__",
                |args| crate::objspace::descroperation::pos(args[0]),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__invert__",
            make_builtin_function_with_arity("__invert__", int_descr_invert, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // `int.__floor__` / `int.__ceil__` return the int itself.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__floor__",
            make_builtin_function_with_arity("__floor__", |args| Ok(args[0]), 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ceil__",
            make_builtin_function_with_arity("__ceil__", |args| Ok(args[0]), 1),
        )
    };
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
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
    // `__pow__` takes an optional modulus, so it is variadic.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__pow__",
            make_builtin_function("__pow__", int_dunder_pow),
        )
    };
    for (name, func) in [
        ("__eq__", int_dunder_eq as DunderFn),
        ("__ne__", int_dunder_ne),
        ("__lt__", int_dunder_lt),
        ("__le__", int_dunder_le),
        ("__gt__", int_dunder_gt),
        ("__ge__", int_dunder_ge),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
    // intobject.py descr_getnewargs — `(wrapint(self.intval),)`: a fresh
    // plain int from the value, so an int subclass (e.g. bool) reduces to
    // the base int.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
}
/// Complex `repr` (`Xj` for a pure-`+0` real part, else `(re±imj)`),
/// delegated to `rustpython_literal::complex::to_string`.
pub(crate) fn complex_repr_string(re: f64, im: f64) -> String {
    rustpython_literal::complex::to_string(re, im)
}

fn init_complex_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new(
                "Create a complex number from a string or numbers.\n\n\
                 If a string is given, parse it as a complex number.\n\
                 If a single number is given, convert it to a complex number.\n\
                 If the 'real' or 'imag' arguments are given, create a complex number\n\
                 with the specified real and imaginary components.",
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(complex_descr_new),
        )
    };
    let repr = |args: &[PyObjectRef]| {
        let (re, im) = unsafe {
            (
                pyre_object::w_complex_get_real(args[0]),
                pyre_object::w_complex_get_imag(args[0]),
            )
        };
        Ok(pyre_object::w_str_new(&complex_repr_string(re, im)))
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity("__repr__", repr, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "from_number",
            pyre_object::function::w_classmethod_new(make_builtin_function_with_arity(
                "from_number",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "complex.from_number() missing required argument 'number' (pos 1)",
                        ));
                    }
                    let value = args[1];
                    if unsafe {
                        pyre_object::is_str(value)
                            || pyre_object::is_bytes(value)
                            || pyre_object::is_bytearray(value)
                    } {
                        return Err(crate::PyError::type_error(format!(
                            "must be real number, not {}",
                            crate::type_methods::arg_type_name(value)
                        )));
                    }
                    // Reuse complex.__new__'s exact-base identity and subclass
                    // allocation.  The constructor's numeric-only path runs
                    // __complex__, __float__, then __index__ without parsing
                    // text because those inputs were rejected above.
                    complex_descr_new(&[args[0], value])
                },
                2,
            )),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__format__",
            make_builtin_function_with_arity(
                "__format__",
                crate::type_methods::builtin_value_format,
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                        crate::objspace::descroperation::complex_hash(args[0], re, im),
                    ))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__abs__",
            make_builtin_function_with_arity(
                "__abs__",
                |args| crate::objspace::descroperation::complex_abs(args[0]),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__neg__",
            make_builtin_function_with_arity(
                "__neg__",
                |args| crate::objspace::descroperation::neg(args[0]),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__pos__",
            make_builtin_function_with_arity(
                "__pos__",
                |args| crate::objspace::descroperation::pos(args[0]),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                    // complexobject.py descr___getnewargs__: two base floats,
                    // preserving the sign bits of zero components.
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_float_new(re),
                        pyre_object::w_float_new(im),
                    ]))
                },
                1,
            ),
        )
    };
    // complex.real / complex.imag — read-only float components.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
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
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
    for (name, func) in [
        ("__eq__", complex_dunder_eq as DunderFn),
        ("__ne__", complex_dunder_ne),
        ("__lt__", complex_dunder_lt),
        ("__le__", complex_dunder_le),
        ("__gt__", complex_dunder_gt),
        ("__ge__", complex_dunder_ge),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
}

fn init_float_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new("Convert a string or number to a floating-point number, if possible."),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(float_descr_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    Ok(w_str_new(&crate::display::format_float_repr(unsafe {
                        pyre_object::w_float_get_value(args[0])
                    })))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| Ok(w_int_new(crate::builtins::hash_value(args[0]))),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "from_number",
            pyre_object::function::w_classmethod_new(make_builtin_function_with_arity(
                "from_number",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "float.from_number() missing required argument 'number' (pos 1)",
                        ));
                    }
                    let value = args[1];
                    // Python 3.14 float.from_number uses the numeric
                    // conversion protocol (__float__, then __index__) but,
                    // unlike float(), never accepts textual inputs.
                    if unsafe {
                        pyre_object::is_str(value)
                            || pyre_object::is_bytes(value)
                            || pyre_object::is_bytearray(value)
                            || pyre_object::is_complex(value)
                    } {
                        return Err(crate::PyError::type_error(format!(
                            "must be real number, not {}",
                            crate::type_methods::arg_type_name(value)
                        )));
                    }
                    // Reuse float.__new__'s exact base/subclass allocation:
                    // exact base floats retain identity, while a classmethod
                    // invoked on a float subclass returns that subclass.
                    float_descr_new(&[args[0], value])
                },
                2,
            )),
        )
    };
    // float.__getformat__(kind) → returns the format string for the
    // given kind. PyPy: floatobject.py W_FloatObject.descr__getformat__.
    // Both 'double' and 'float' are IEEE 754 little-endian on x86/ARM.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "fromhex",
            pyre_object::function::w_classmethod_new(make_builtin_function_with_arity(
                "fromhex",
                |args| {
                    // float.fromhex(s) — PyPy: floatobject.py descr_fromhex.
                    // Parse hexadecimal floating-point literals like '0x1.8p3'.
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "fromhex() requires a string argument",
                        ));
                    }
                    let s_arg = if unsafe { pyre_object::is_str(args[1]) } {
                        unsafe { pyre_object::w_str_get_value(args[1]).to_string() }
                    } else {
                        return Err(crate::PyError::type_error(
                            "fromhex() requires a string argument",
                        ));
                    };
                    // Delegate parsing to the shared hex-float reader, which rounds
                    // round-half-even over the full exponent range (subnormals down to
                    // 0x1p-1074), accepts the inf/nan spellings, handles surrounding
                    // ASCII whitespace itself, and flags overflow distinctly.
                    match rustpython_common::float_ops::from_hex(&s_arg) {
                        Ok(v) => {
                            let w_float = pyre_object::w_float_new(v);
                            // floatobject.py:419: return
                            // space.call_function(w_cls, w_float).  This runs a
                            // subclass's __new__ and __init__ rather than merely
                            // retagging the parsed base float.
                            crate::call::call_function_impl_result(args[0], &[w_float])
                        }
                        Err(e) => {
                            use rustpython_common::float_ops::HexFloatError;
                            Err(match e {
                                HexFloatError::Overflow => crate::PyError::overflow_error(
                                    "hexadecimal value too large to represent as a float",
                                ),
                                HexFloatError::TooLong => crate::PyError::value_error(
                                    "hexadecimal string too long to convert",
                                ),
                                HexFloatError::Invalid => crate::PyError::value_error(
                                    "invalid hexadecimal floating-point string",
                                ),
                            })
                        }
                    }
                },
                2,
            )),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // float.conjugate — identity for a real number.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // float.real / float.imag — a float is its own real part; imag is 0.0.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "imag",
            pyre_object::w_property_new(
                make_builtin_function_with_arity("imag", |_| Ok(pyre_object::w_float_new(0.0)), 1),
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
            ),
        )
    };
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
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                method,
                make_builtin_function_with_arity(method, func, 1),
            )
        };
    }
    // Unary / conversion slots exposed as callable dunders (no
    // NotImplemented dispatch).  Binary arithmetic dunders are
    // registered separately.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__round__",
            make_builtin_function("__round__", crate::builtins::builtin_round),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__format__",
            make_builtin_function_with_arity(
                "__format__",
                crate::type_methods::builtin_value_format,
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__float__",
            make_builtin_function_with_arity("__float__", crate::builtins::builtin_float_dunder, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__abs__",
            make_builtin_function_with_arity("__abs__", crate::builtins::builtin_abs, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__neg__",
            make_builtin_function_with_arity(
                "__neg__",
                |args| crate::objspace::descroperation::neg(args[0]),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__pos__",
            make_builtin_function_with_arity(
                "__pos__",
                |args| crate::objspace::descroperation::pos(args[0]),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
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
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
    for (name, func) in [
        ("__eq__", float_dunder_eq as DunderFn),
        ("__ne__", float_dunder_ne),
        ("__lt__", float_dunder_lt),
        ("__le__", float_dunder_le),
        ("__gt__", float_dunder_gt),
        ("__ge__", float_dunder_ge),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
    // floatobject.py descr_getnewargs — `(self.descr_float(),)`: a fresh
    // plain float from the value.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
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
    let a = args[0];
    let b = args[1];
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

/// `W_IntObject.descr_invert` — the integer inversion slot.  `~x` reaches
/// bool's warning-bearing slot first, but `int.__invert__` resolves straight
/// to this one, so a bool receiver is inverted without the deprecation
/// warning.  The registered arity is only a dispatch hint, so the positional
/// count is enforced here.
fn int_descr_invert(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(&w_self) = args.first() else {
        return Err(crate::PyError::type_error(
            "int.__invert__() missing 1 required positional argument: 'self'",
        ));
    };
    if args.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "int.__invert__() takes 1 positional argument but {} were given",
            args.len(),
        )));
    }
    if unsafe { pyre_object::is_bool(w_self) } {
        return Ok(w_int_new(!unsafe {
            crate::objspace::descroperation::int_value(w_self)
        }));
    }
    crate::objspace::descroperation::invert(w_self)
}

/// CPython 3.14 `Objects/boolobject.c:bool_invert`.
///
/// The bundled PyPy source inherits `int.__invert__`; Python 3.14 instead
/// installs a bool-specific number slot solely to issue this deprecation
/// warning before delegating to the underlying integer inversion.
fn bool_descr_invert(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(&w_self) = args.first() else {
        return Err(crate::PyError::type_error(
            "descriptor '__invert__' of 'bool' object needs an argument",
        ));
    };
    if !unsafe { pyre_object::is_bool(w_self) } {
        return Err(crate::PyError::type_error(format!(
            "descriptor '__invert__' requires a 'bool' object but received a '{}'",
            crate::baseobjspace::object_functionstr_type_name(w_self),
        )));
    }
    crate::type_methods::arity_slot(args, 0)?;
    crate::objspace::descroperation::invert(w_self)
}

fn init_bool_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new(
                "Returns True when the argument is true, False otherwise.\n\
                 The builtins True and False are the only two instances of the class bool.\n\
                 The class bool is a subclass of the class int, and cannot be subclassed.",
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(bool_descr_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function("__repr__", bool_repr),
        )
    };
    // CPython 3.14 gives bool an explicit deprecated `__invert__` wrapper;
    // the bundled PyPy source inherits the int descriptor instead.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__invert__",
            make_builtin_function("__invert__", bool_descr_invert),
        )
    };
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
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                and_name,
                make_builtin_function(and_name, f),
            )
        };
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                rand_name,
                make_builtin_function(rand_name, f),
            )
        };
    }
}

// ── Object TypeDef ───────────────────────────────────────────────────
// PyPy: pypy/objspace/std/objectobject.py TypeDef("object", ...)

// The `__new__` override check below uses raw slot identity
// (`same_inherited_slot`), not a `staticmethod`-unwrapping compare.  PyPy's
// `_same_static_method` (objectobject.py:113-116) unwraps a
// `staticmethod`-wrapped `__new__` to its function before the identity
// compare, so `__new__ = staticmethod(object.__new__)` counts as inherited;
// CPython 3.14 keeps the wrapper as a distinct `tp_new` slot, so the same
// assignment counts as an OVERRIDE and `Cls(1, 2)` raises
// "object.__new__() takes exactly one argument (the type to instantiate)".
// pyre targets CPython, so raw identity is the correct comparison here.

/// `object.__new__(cls)` — allocate a bare instance of cls.
///
/// PyPy: objectobject.py descr__new__
/// objectobject.py:17-21 _abstract_method_error (CPython 3.12+ message form).
fn abstract_instantiation_error(cls: PyObjectRef) -> crate::PyError {
    let type_name = unsafe { pyre_object::w_type_get_name(cls) };
    let mut methods: Vec<String> = Vec::new();
    if let Ok(w_abstracts) = crate::baseobjspace::getattr_str(cls, "__abstractmethods__") {
        // The setter marks any truthy value abstract without enforcing a set,
        // so guard the unchecked casts: only iterate a genuine set of strings.
        if unsafe { pyre_object::is_set_or_frozenset(w_abstracts) } {
            for item in unsafe { pyre_object::w_set_items(w_abstracts) } {
                if unsafe { pyre_object::is_str(item) } {
                    if let Ok(s) = unsafe { pyre_object::w_str_get_wtf8(item) }.as_str() {
                        methods.push(s.to_string());
                    }
                }
            }
        }
    }
    methods.sort();
    let plural = if methods.len() == 1 { "" } else { "s" };
    let joined = methods
        .iter()
        .map(|m| format!("'{m}'"))
        .collect::<Vec<_>>()
        .join(", ");
    crate::PyError::type_error(format!(
        "Can't instantiate abstract class {type_name} without an implementation \
         for abstract method{plural} {joined}"
    ))
}

/// Identity comparison of two MRO lookups (`space.is_w`), used to decide
/// whether a type inherits `object`'s `__new__`/`__init__` unchanged.  An
/// inherited slot resolves to the very object stored in `object`'s dict, so
/// pointer identity suffices; an override resolves to a distinct object.
fn same_inherited_slot(a: Option<PyObjectRef>, b: Option<PyObjectRef>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x == y,
        (None, None) => true,
        _ => false,
    }
}

fn object_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_object = w_object();
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let Some(&w_type_arg) = positional.first() else {
        return Err(crate::PyError::type_error(
            "object.__new__(): not enough arguments".to_string(),
        ));
    };
    let cls = w_type_arg;
    // Bootstrap: before `object` is installed the slot lookups below cannot
    // run, and no excess-args call reaches here that early.
    if w_object.is_null() {
        return Ok(if unsafe { is_type(cls) } {
            w_instance_new(cls)
        } else {
            w_instance_new(PY_NULL)
        });
    }
    // `_precheck_for_new` (typeobject.py:1001-1004): a non-type first
    // argument is a `TypeError`, not a silent bare instance.  `%T` names the
    // Python type of `cls` (tag-safe via `r#type`), not its raw C layout.
    if !unsafe { is_type(cls) } {
        let name = crate::typedef::r#type(cls)
            .map(|t| unsafe { pyre_object::w_type_get_name(t) })
            .unwrap_or("object");
        return Err(crate::PyError::type_error(format!(
            "object.__new__(X): X is not a type object ({name})"
        )));
    }
    // objectobject.py descr__new__ — surplus arguments are accepted only
    // when __new__ or __init__ is overridden; the bare object() takes
    // none.  A type that overrides __new__ but forwards excess args to
    // object.__new__ hits the first error.
    if positional.len() > 1 || crate::builtins::has_real_kwargs(kwargs) {
        let tp_new = unsafe { crate::baseobjspace::lookup_in_type(cls, "__new__") };
        let obj_new = unsafe { crate::baseobjspace::lookup_in_type(w_object, "__new__") };
        if !same_inherited_slot(tp_new, obj_new) {
            return Err(crate::PyError::type_error(
                "object.__new__() takes exactly one argument (the type to instantiate)",
            ));
        }
        let tp_init = unsafe { crate::baseobjspace::lookup_in_type(cls, "__init__") };
        let obj_init = unsafe { crate::baseobjspace::lookup_in_type(w_object, "__init__") };
        if same_inherited_slot(tp_init, obj_init) {
            let name = unsafe { pyre_object::w_type_get_name(cls) };
            return Err(crate::PyError::type_error(format!(
                "{name}() takes no arguments"
            )));
        }
    }
    // objectobject.py:131 descr__new__ — abstract classes refuse instantiation,
    // checked after the excess-args gate and before allocating.
    if unsafe { pyre_object::w_type_is_abstract(cls) } {
        return Err(abstract_instantiation_error(cls));
    }
    Ok(w_instance_new(cls))
}

/// `object.__init__(self)` — no-op base __init__.  Surplus arguments are
/// accepted only when __init__ or __new__ is overridden (objectobject.py
/// descr__init__); otherwise the bare object initializer takes none.
fn object_descr_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_object = w_object();
    if w_object.is_null() {
        return Ok(w_none());
    }
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if !(positional.len() > 1 || crate::builtins::has_real_kwargs(kwargs)) {
        return Ok(w_none());
    }
    let Some(&w_obj) = positional.first() else {
        return Ok(w_none());
    };
    if let Some(w_type) = crate::typedef::r#type(w_obj) {
        let tp_init = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__init__") };
        let obj_init = unsafe { crate::baseobjspace::lookup_in_type(w_object, "__init__") };
        if !same_inherited_slot(tp_init, obj_init) {
            return Err(crate::PyError::type_error(
                "object.__init__() takes exactly one argument (the instance to initialize)",
            ));
        }
        let tp_new = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__new__") };
        let obj_new = unsafe { crate::baseobjspace::lookup_in_type(w_object, "__new__") };
        if same_inherited_slot(tp_new, obj_new) {
            let name = unsafe { pyre_object::w_type_get_name(w_type) };
            return Err(crate::PyError::type_error(format!(
                "{name}.__init__() takes exactly one argument (the instance to initialize)"
            )));
        }
    }
    Ok(w_none())
}

/// `object.__sizeof__` — CPython 3.14's generic object size is the fixed
/// object header plus one pointer-sized word for every declared slot.  The
/// instance dict and weakref storage are accounted for separately by
/// `sys.getsizeof`, so they are deliberately absent here.
fn object_descr_sizeof(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_slot(args, 0)?;
    let mut size = std::mem::size_of::<pyre_object::PyObject>();
    if let Some(w_type) = crate::typedef::r#type(args[0]) {
        if unsafe { pyre_object::is_type(w_type) } {
            let layout = unsafe { pyre_object::w_type_get_layout_ptr(w_type) };
            if !layout.is_null() {
                size += unsafe { (*layout).nslots as usize }
                    * std::mem::size_of::<pyre_object::PyObjectRef>();
            }
        }
    }
    Ok(w_int_new(size as i64))
}

fn init_object_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new(
                "The base class of the class hierarchy.\n\n\
                 When called, it accepts no arguments and returns a new featureless\n\
                 instance that has no instance attributes and cannot be given any.\n",
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(object_descr_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__init__",
            make_builtin_function("__init__", object_descr_init),
        )
    };
    // PyPy: objectobject.py — default comparison/hash/repr for all objects
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__eq__",
            make_builtin_function_with_arity(
                "__eq__",
                |args| {
                    crate::type_methods::arity_slot(args, 1)?;
                    if std::ptr::eq(args[0], args[1]) {
                        Ok(pyre_object::w_bool_from(true))
                    } else {
                        // objectobject.py descr__eq__: give the reflected
                        // comparison a chance instead of deciding False here.
                        Ok(pyre_object::w_not_implemented())
                    }
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                    // objectobject.py descr__ne__: look up and call the live
                    // receiver's __eq__ descriptor, then invert that one
                    // result.  Running the full comparison dispatcher here
                    // would apply reflection and the identity fallback too
                    // early.
                    let eq_descr = unsafe {
                        crate::baseobjspace::lookup(args[0], "__eq__")
                            .expect("every object type inherits object.__eq__")
                    };
                    let w_type =
                        crate::typedef::r#type(args[0]).expect("every Python object has a type");
                    let eq = unsafe {
                        crate::baseobjspace::get_and_call_function(
                            eq_descr,
                            args[0],
                            w_type,
                            &[args[1]],
                        )?
                    };
                    // A `NotImplemented` from `__eq__` must pass through so the
                    // caller can try the reflected comparison.
                    if unsafe { pyre_object::is_not_implemented(eq) } {
                        return Ok(eq);
                    }
                    Ok(pyre_object::w_bool_from(!crate::baseobjspace::is_true(eq)?))
                },
                2,
            ),
        )
    };
    // objectobject.py:321 / typedef :451-458 — all four ordering methods
    // return NotImplemented and leave reflected comparison / TypeError to the
    // object space.
    for name in ["__lt__", "__le__", "__gt__", "__ge__"] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(
                    name,
                    |args| {
                        crate::type_methods::arity_slot(args, 1)?;
                        Ok(pyre_object::w_not_implemented())
                    },
                    2,
                ),
            )
        };
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| {
                    // The fixed arity above is only a fast-dispatch hint; the
                    // direct path still delivers whatever the caller passed.
                    if args.len() != 1 {
                        let message = if args.is_empty() {
                            "object.__hash__() missing 1 required positional argument: 'obj'"
                                .to_string()
                        } else {
                            format!(
                                "object.__hash__() takes 1 positional argument but {} were given",
                                args.len(),
                            )
                        };
                        return Err(crate::PyError::type_error(message));
                    }
                    Ok(default_identity_hash(pyre_object::PY_NULL, args[0]))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            // PyPy: objectobject.py descr___repr__ — base __repr__ for all objects
            make_builtin_function_with_arity(
                "__repr__",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "descriptor '__repr__' of 'object' object needs an argument",
                        ));
                    }
                    crate::type_methods::arity_no_args(args, "object.__repr__")?;
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__str__",
            make_builtin_function_with_arity(
                "__str__",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "descriptor '__str__' of 'object' object needs an argument",
                        ));
                    }
                    crate::type_methods::arity_no_args(args, "object.__str__")?;
                    // Delegate to __repr__ to avoid infinite recursion
                    // PyPy: objectobject.py descr___str__ → space.repr(w_self)
                    Ok(pyre_object::w_str_new(&unsafe { crate::py_repr(args[0])? }))
                },
                1,
            ),
        )
    };
    // PyPy: objectobject.py descr___format__
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__format__",
            make_builtin_function_with_arity(
                "__format__",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "unbound method object.__format__() needs an argument",
                        ));
                    }
                    crate::type_methods::arity_exact(args, "object.__format__", 1)?;
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
                    Ok(pyre_object::w_str_new(&unsafe { crate::py_str(args[0])? }))
                },
                2,
            ),
        )
    };
    // objectobject.py descr__reduce__ / descr__reduce_ex__ / descr__getstate__
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "unbound method object.__reduce__() needs an argument",
                        ));
                    }
                    crate::type_methods::arity_no_args(args, "object.__reduce__")?;
                    crate::reduce_protocol::descr_reduce(args[0])
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reduce_ex__",
            make_builtin_function_with_arity(
                "__reduce_ex__",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "unbound method object.__reduce_ex__() needs an argument",
                        ));
                    }
                    crate::type_methods::arity_exact(args, "object.__reduce_ex__", 1)?;
                    let proto = crate::builtins::space_index_w(args[1])?;
                    crate::reduce_protocol::descr_reduce_ex(args[0], proto)
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__getstate__",
            make_builtin_function_with_arity(
                "__getstate__",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "unbound method object.__getstate__() needs an argument",
                        ));
                    }
                    crate::type_methods::arity_no_args(args, "object.__getstate__")?;
                    crate::reduce_protocol::object_getstate_default(args[0])
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__dir__",
            make_builtin_function_with_arity(
                "__dir__",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "unbound method object.__dir__() needs an argument",
                        ));
                    }
                    crate::type_methods::arity_no_args(args, "object.__dir__")?;
                    crate::builtins::object_dir_default(args[0])
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__sizeof__",
            make_builtin_function_with_arity(
                "__sizeof__",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "unbound method object.__sizeof__() needs an argument",
                        ));
                    }
                    crate::type_methods::arity_no_args(args, "object.__sizeof__")?;
                    object_descr_sizeof(args)
                },
                1,
            ),
        )
    };
    // typeobject.py descr___init_subclass__ — the default accepts no
    // keywords; class-definition keywords reaching it via the builtin
    // kwargs ABI are an error, not silently dropped.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__subclasshook__",
            make_builtin_function("__subclasshook__", |_| Ok(pyre_object::w_not_implemented())),
        )
    };
    // PyPy: objectobject.py descr___setattr__
    // object.__setattr__(self, name, value) → setattr dispatch
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // PyPy: objectobject.py descr___delattr__
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // PyPy: objectobject.py descr___getattribute__
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                            crate::baseobjspace::object_getattribute_surrogate(
                                args[0], args[1], name,
                            )
                        },
                    }
                },
                2,
            ),
        )
    };
}

fn bytearray_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let value = bytearray_descr_new_impl(args)?;
    if let Some(sub) = subclass_to_tag(cls, &pyre_object::bytearrayobject::BYTEARRAY_TYPE)? {
        let data = unsafe { pyre_object::bytesobject::bytes_like_data(value).to_vec() };
        let fresh = pyre_object::bytearrayobject::w_bytearray_subclass_from_bytes(&data, sub);
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
fn init_bytes_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new(
                "bytes(iterable_of_ints) -> bytes\n\
                 bytes(string, encoding[, errors]) -> bytes\n\
                 bytes(bytes_or_buffer) -> immutable copy of bytes_or_buffer\n\
                 bytes(int) -> bytes object of size given by the parameter initialized with null bytes\n\
                 bytes() -> empty bytes object\n\n\
                 Construct an immutable array of bytes from:\n\
                   - an iterable yielding integers in range(256)\n\
                   - a text string encoded using the specified encoding\n\
                   - any object implementing the buffer API.\n\
                   - an integer",
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| Ok(w_int_new(crate::builtins::hash_value(args[0]))),
                1,
            ),
        )
    };
    // Python 3.14 exposes the PEP 688 buffer slot as `bytes.__buffer__`.
    // The returned memoryview retains the immutable bytes backing and reports
    // `readonly=True`; flags are advisory for this always-readable exporter.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__buffer__",
            make_builtin_function_with_arity(
                "__buffer__",
                |args| crate::builtins::w_memoryview_new(args[0]),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(bytes_descr_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__bytes__",
            make_builtin_function_with_arity("__bytes__", bytes_method_bytes, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "decode",
            make_builtin_function("decode", bytes_method_decode),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity("__repr__", bytes_method_repr, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__str__",
            make_builtin_function_with_arity("__str__", bytes_method_repr, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "hex",
            make_builtin_function("hex", bytes_method_hex),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "find",
            make_builtin_function("find", bytes_method_find),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rfind",
            make_builtin_function("rfind", bytes_method_rfind),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "index",
            make_builtin_function("index", bytes_method_index),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rindex",
            make_builtin_function("rindex", bytes_method_rindex),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "count",
            make_builtin_function("count", bytes_method_count),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "startswith",
            make_builtin_function("startswith", bytes_method_startswith),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "endswith",
            make_builtin_function("endswith", bytes_method_endswith),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "upper",
            make_builtin_function("upper", bytes_method_upper),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "lower",
            make_builtin_function("lower", bytes_method_lower),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "strip",
            make_builtin_function("strip", bytes_method_strip),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "lstrip",
            make_builtin_function("lstrip", bytes_method_lstrip),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rstrip",
            make_builtin_function("rstrip", bytes_method_rstrip),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "replace",
            make_builtin_function("replace", bytes_method_replace),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "split",
            make_builtin_function("split", bytes_method_split),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rsplit",
            make_builtin_function("rsplit", bytes_method_rsplit),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "join",
            make_builtin_function("join", bytes_method_join),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "partition",
            make_builtin_function("partition", bytes_method_partition),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rpartition",
            make_builtin_function("rpartition", bytes_method_rpartition),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "translate",
            make_builtin_function("translate", bytes_method_translate),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isdigit",
            make_builtin_function("isdigit", bytes_method_isdigit),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isalpha",
            make_builtin_function("isalpha", bytes_method_isalpha),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isalnum",
            make_builtin_function("isalnum", bytes_method_isalnum),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isspace",
            make_builtin_function("isspace", bytes_method_isspace),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isascii",
            make_builtin_function("isascii", bytes_method_isascii),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isupper",
            make_builtin_function("isupper", bytes_method_isupper),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "islower",
            make_builtin_function("islower", bytes_method_islower),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "istitle",
            make_builtin_function("istitle", bytes_method_istitle),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "title",
            make_builtin_function("title", bytes_method_title),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "capitalize",
            make_builtin_function("capitalize", bytes_method_capitalize),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "swapcase",
            make_builtin_function("swapcase", bytes_method_swapcase),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "removeprefix",
            make_builtin_function("removeprefix", bytes_method_removeprefix),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "removesuffix",
            make_builtin_function("removesuffix", bytes_method_removesuffix),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "ljust",
            make_builtin_function("ljust", bytes_method_ljust),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rjust",
            make_builtin_function("rjust", bytes_method_rjust),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "center",
            make_builtin_function("center", bytes_method_center),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "zfill",
            make_builtin_function("zfill", bytes_method_zfill),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "splitlines",
            make_builtin_function("splitlines", bytes_method_splitlines),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "expandtabs",
            make_builtin_function("expandtabs", bytes_method_expandtabs),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "maketrans",
            make_maketrans_descr(bytes_maketrans),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "fromhex",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "fromhex",
                bytes_fromhex,
            )),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__add__",
            make_builtin_function_with_arity(
                "__add__",
                |args| {
                    crate::type_methods::arity_slot(args, 1)?;
                    // `descr_add` returns NotImplemented for a non-buffer operand
                    // so the `+` operator raises the generic operator TypeError.
                    unsafe {
                        match buffer_as_bytes_like(args[1])? {
                            Some(_) => {
                                crate::objspace::descroperation::bytes_concat(args[0], args[1])
                            }
                            None => Ok(pyre_object::w_not_implemented()),
                        }
                    }
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__mul__",
            make_builtin_function_with_arity("__mul__", |args| bytes_descr_repeat(args), 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__rmul__",
            make_builtin_function_with_arity("__rmul__", |args| bytes_descr_repeat(args), 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__mod__",
            make_builtin_function_with_arity(
                "__mod__",
                |args| {
                    crate::type_methods::arity_slot(args, 1)?;
                    unsafe {
                        crate::objspace::std::formatting::bytes_format_percent(args[0], args[1])
                    }
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    for (name, func) in [
        ("__eq__", bytes_dunder_eq as DunderFn),
        ("__ne__", bytes_dunder_ne),
        ("__lt__", bytes_dunder_lt),
        ("__le__", bytes_dunder_le),
        ("__gt__", bytes_dunder_gt),
        ("__ge__", bytes_dunder_ge),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
    // bytesobject.py descr_getnewargs — a fresh plain bytes from the value,
    // so a bytes subclass reduces to bytes.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
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
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    if let Some(data) = crate::module::_ctypes::cdata::cdata_bytes(obj) {
        return Ok(Some(pyre_object::bytesobject::w_bytes_from_bytes(data)));
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
    // A tagged int immediate is an exact builtin int; skip the ob_type deref.
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return "int".to_string();
    }
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
            unsafe { pyre_object::type_name_of(pos[0]) }
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
            unsafe { pyre_object::type_name_of(pos[0]) },
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
            unsafe { pyre_object::type_name_of(pos[0]) },
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
                unsafe { pyre_object::type_name_of(a) }
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
            let tname = unsafe { pyre_object::type_name_of(b) };
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
    // No sep / default grouping — produces "ffff" for [0xff, 0xff].
    // The sep + bytes_per_sep kwargs are deferred until first observed
    // need; CPython callers without args hit the hot path.
    let sep_arg = pos
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "sep"));
    if sep_arg.is_none() {
        // Nothing below can run Python code, so the payload is read here.
        let data = unsafe { pyre_object::bytesobject::bytes_like_data(pos[0]) };
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
    // CPython 3.14 `_Py_strhex_impl` deliberately uses `PyObject_Length`
    // before inspecting the separator payload.  A bytes/str subclass may
    // therefore run arbitrary `__len__` code here (gh-143195).  Once it
    // reports one, the first payload unit is used; an empty payload supplies
    // the terminating NUL, matching `_PyUnicode_AsUTF8AndSize`/`PyBytes_AS_STRING`.
    let sep_length_error =
        || crate::PyError::new(crate::PyErrorKind::ValueError, "sep must be length 1.");
    let sep_ascii_error =
        || crate::PyError::new(crate::PyErrorKind::ValueError, "sep must be ASCII.");
    if crate::baseobjspace::len_w(sep_obj)? != 1 {
        return Err(sep_length_error());
    }
    let sep_char: char = if unsafe { pyre_object::is_str(sep_obj) } {
        let s = unsafe { pyre_object::w_str_get_value(sep_obj) };
        if !s.is_ascii() {
            return Err(sep_ascii_error());
        }
        s.chars().next().unwrap_or('\0')
    } else if unsafe { pyre_object::is_bytes(sep_obj) } {
        let sep_bytes = unsafe { pyre_object::bytesobject::bytes_like_data(sep_obj) };
        if !sep_bytes.is_ascii() {
            return Err(sep_ascii_error());
        }
        sep_bytes.first().copied().unwrap_or(0) as char
    } else {
        return Err(crate::PyError::type_error("sep must be str or bytes."));
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
    // Read the payload only now: `bytes_per_sep` coercion above can run
    // `__index__`, which may clear or resize a bytearray receiver and
    // leave any slice taken earlier describing a stale length.
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(pos[0]) };
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
pub(crate) fn decode_error_encode_only_handler() -> crate::PyError {
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
) -> Result<(usize, Option<Vec<u8>>), crate::PyError> {
    match err_mode {
        "strict" => {
            utf8_strict_handler(data, start, end, reason)?;
            unreachable!()
        }
        "ignore" => Ok((end, None)),
        "replace" => {
            out.push_char('\u{FFFD}');
            Ok((end, None))
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
            Ok((start + consumed, None))
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
            Ok((start + 3, None))
        }
        "backslashreplace" => {
            for &b in &data[start..end] {
                out.push_str(&format!("\\x{:02x}", b));
            }
            Ok((end, None))
        }
        "xmlcharrefreplace" | "namereplace" => Err(decode_error_encode_only_handler()),
        _ => crate::type_methods::call_registered_decode_error_handler(
            err_mode, "utf-8", data, start, end, reason, out,
        ),
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
    // A custom error handler may replace exc.object; decoding then resumes
    // from the new bytes (`s`), re-evaluating `size` each iteration.  The
    // common path keeps the borrowed slice (no allocation).
    let mut s: std::borrow::Cow<[u8]> = std::borrow::Cow::Borrowed(data);
    let mut size = s.len();
    let mut result = Wtf8Buf::new();
    let mut pos = 0;
    let final_ = true; // pyre always decodes complete buffers

    // Run a utf-8 error handler and rebind `s`/`size` when it returns
    // replacement bytes; then advance `pos` to the resume position.
    macro_rules! run_err {
        ($start:expr, $end:expr, $reason:expr) => {{
            let (np, nb) = utf8_error_handler(err_mode, &s, $start, $end, $reason, &mut result)?;
            if let Some(b) = nb {
                s = std::borrow::Cow::Owned(b);
                size = s.len();
            }
            pos = np;
        }};
    }

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
                run_err!(pos, pos + 1, "unexpected end of data");
                continue;
            }
            let ordch2 = s[pos + 1];
            if n == 3 {
                // unicodehelper.py:417-434
                if invalid_byte_2_of_3(ordch1, ordch2) {
                    run_err!(pos, pos + 1, "invalid continuation byte");
                    continue;
                }
                if !final_ {
                    break;
                }
                run_err!(pos, pos + 2, "unexpected end of data");
                continue;
            } else if n == 4 {
                // unicodehelper.py:435-459
                if invalid_byte_2_of_4(ordch1, ordch2) {
                    run_err!(pos, pos + 1, "invalid continuation byte");
                    continue;
                }
                if charsleft == 2 && invalid_cont_byte(s[pos + 2]) {
                    run_err!(pos, pos + 2, "invalid continuation byte");
                    continue;
                }
                if !final_ {
                    break;
                }
                run_err!(pos, pos + charsleft + 1, "unexpected end of data");
                continue;
            }
            unreachable!("n must be 3 or 4 when charsleft > 0");
        }

        // unicodehelper.py:462 n == 0 → invalid start byte
        if n == 0 {
            run_err!(pos, pos + 1, "invalid start byte");
            continue;
        }

        if n == 2 {
            // unicodehelper.py:471-482
            let ordch2 = s[pos + 1];
            if invalid_cont_byte(ordch2) {
                run_err!(pos, pos + 1, "invalid continuation byte");
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
                run_err!(pos, pos + 1, "invalid continuation byte");
                continue;
            }
            if invalid_cont_byte(ordch3) {
                run_err!(pos, pos + 2, "invalid continuation byte");
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
                run_err!(pos, pos + 1, "invalid continuation byte");
                continue;
            }
            if invalid_cont_byte(ordch3) {
                run_err!(pos, pos + 2, "invalid continuation byte");
                continue;
            }
            if invalid_cont_byte(ordch4) {
                run_err!(pos, pos + 3, "invalid continuation byte");
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
            let tn = unsafe { pyre_object::type_name_of(enc) };
            return Err(crate::PyError::type_error(format!(
                "decode() argument 'encoding' must be str, not {tn}",
            )));
        }
    }
    if let Some(err) = w_errors {
        if !unsafe { pyre_object::is_str(err) } && !unsafe { pyre_object::is_none(err) } {
            let tn = unsafe { pyre_object::type_name_of(err) };
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
    if crate::importing::dev_mode_flag()
        && matches!(
            enc_lower.as_str(),
            "utf-8"
                | "utf8"
                | "u8"
                | "ascii"
                | "us-ascii"
                | "646"
                | "latin-1"
                | "latin1"
                | "iso-8859-1"
                | "8859"
                | "raw-unicode-escape"
                | "utf-16"
                | "utf-16-le"
                | "utf-16-be"
                | "utf-32"
                | "utf-32-le"
                | "utf-32-be"
        )
    {
        crate::module::_codecs::validate_error_handler(errors)?;
    }
    let s = match enc_lower.as_str() {
        "utf-8" | "utf8" | "u8" => decode_utf8_with_errors(data, err_mode)?,
        "ascii" | "us-ascii" | "646" => {
            let mut out = Wtf8Buf::new();
            // A custom error handler may replace exc.object; decoding then
            // resumes from the new bytes (`abuf`).
            let mut abuf: std::borrow::Cow<[u8]> = std::borrow::Cow::Borrowed(data);
            let mut i = 0;
            while i < abuf.len() {
                let b = abuf[i];
                if b >= 0x80 {
                    match err_mode {
                        "strict" => {
                            return Err(unicode_decode_error(
                                "ascii",
                                &abuf,
                                i,
                                i + 1,
                                "ordinal not in range(128)",
                            ));
                        }
                        "ignore" => {
                            i += 1;
                            continue;
                        }
                        "replace" => {
                            out.push_char('\u{FFFD}');
                            i += 1;
                            continue;
                        }
                        // surrogateescape escapes the non-ASCII byte as a lone
                        // surrogate 0xdc00+b (interp_codecs.py:536-555).
                        "surrogateescape" => {
                            out.push(CodePoint::from_u32(0xDC00 + b as u32).unwrap());
                            i += 1;
                            continue;
                        }
                        // surrogatepass only decodes three-byte UTF-8 surrogate
                        // sequences; a single non-ASCII byte is not one, so it
                        // re-raises (interp_codecs.py:476-510).
                        "surrogatepass" => {
                            return Err(unicode_decode_error(
                                "ascii",
                                &abuf,
                                i,
                                i + 1,
                                "ordinal not in range(128)",
                            ));
                        }
                        "backslashreplace" => {
                            out.push_str(&format!("\\x{:02x}", b));
                            i += 1;
                            continue;
                        }
                        "xmlcharrefreplace" | "namereplace" => {
                            return Err(decode_error_encode_only_handler());
                        }
                        _ => {
                            let (np, nb) =
                                crate::type_methods::call_registered_decode_error_handler(
                                    err_mode,
                                    "ascii",
                                    &abuf,
                                    i,
                                    i + 1,
                                    "ordinal not in range(128)",
                                    &mut out,
                                )?;
                            if let Some(nb) = nb {
                                abuf = std::borrow::Cow::Owned(nb);
                            }
                            i = np;
                            continue;
                        }
                    }
                }
                out.push_char(b as char);
                i += 1;
            }
            out
        }
        "latin-1" | "latin1" | "iso-8859-1" | "8859" => {
            Wtf8Buf::from_string(data.iter().map(|&b| b as char).collect::<String>())
        }
        "raw-unicode-escape" => crate::type_methods::decode_raw_unicode_escape(data, err_mode)?,
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
        let fresh = pyre_object::bytesobject::w_bytes_subclass_from_bytes(&data, sub);
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
    crate::type_methods::arity_exact(args, "bytearray.append", 1)?;
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
    crate::type_methods::arity_no_args(args, "bytearray.reverse")?;
    unsafe { pyre_object::bytearrayobject::w_bytearray_vec_mut(args[0]).reverse() };
    Ok(pyre_object::w_none())
}

/// `bytearrayobject.py:descr_clear` — empty the bytearray in place.
fn bytearray_method_clear(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "clear")?;
    crate::type_methods::arity_no_args(args, "bytearray.clear")?;
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
    crate::type_methods::arity_no_args(args, "bytearray.copy")?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(data))
}

/// `bytearrayobject.py descr_releasebuffer` — the Python 3.12
/// `__release_buffer__` protocol entry for a released bytearray export.
fn bytearray_method_release_buffer(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unsafe { pyre_object::bytearrayobject::w_bytearray_exports_decref(args[0]) };
    Ok(pyre_object::w_none())
}

/// `bytearrayobject.py:247 descr_init` — materialize the replacement first,
/// then replace the receiver's resizable storage in one step.
fn bytearray_descr_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "__init__")?;
    let fresh = bytearray_descr_new_impl(args)?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(fresh).to_vec() };
    unsafe {
        crate::builtins::bytearray_check_exports(args[0])?;
        *pyre_object::bytearrayobject::w_bytearray_vec_mut(args[0]) = data;
    }
    Ok(w_none())
}

fn bytearray_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_receiver(args, "__repr__")?;
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(args[0]) };
    let class_name = crate::typedef::r#type(args[0])
        .map(|tp| unsafe { pyre_object::w_type_get_name(tp) })
        .unwrap_or("bytearray");
    Ok(w_str_new(&crate::display::bytearray_repr_string(
        data, class_name,
    )))
}

fn bytearray_reduce_impl(
    obj: PyObjectRef,
    protocol: Option<i64>,
) -> Result<PyObjectRef, crate::PyError> {
    let data = unsafe { pyre_object::bytesobject::bytes_like_data(obj) };
    let args = if data.is_empty() {
        w_tuple_new(vec![])
    } else if protocol.is_some_and(|p| p >= 3) {
        w_tuple_new(vec![pyre_object::bytesobject::w_bytes_from_bytes(data)])
    } else {
        // bytearrayobject.py:221-233 — legacy protocols carry a latin-1
        // unicode string plus the explicit codec name.
        let latin1: String = data.iter().map(|&b| char::from(b)).collect();
        w_tuple_new(vec![w_str_new(&latin1), w_str_new("latin-1")])
    };
    let cls = crate::typedef::r#type(obj)
        .unwrap_or_else(|| gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE));
    let state = crate::reduce_protocol::object_getstate_default(obj)?;
    Ok(w_tuple_new(vec![cls, args, state]))
}

fn bytearray_descr_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_slot(args, 0)?;
    bytearray_reduce_impl(args[0], None)
}

fn bytearray_descr_reduce_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_slot(args, 1)?;
    let protocol = crate::baseobjspace::int_w(args[1])?;
    bytearray_reduce_impl(args[0], Some(protocol))
}

fn bytearray_descr_alloc(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_slot(args, 0)?;
    let capacity = unsafe { pyre_object::bytearrayobject::w_bytearray_capacity(args[0]) };
    // PyPy's resizable list includes its trailing NUL. CPython 3.14 exposes
    // the same convention: empty has alloc 0, otherwise payload capacity + 1.
    Ok(w_int_new(if capacity == 0 {
        0
    } else {
        (capacity + 1) as i64
    }))
}

fn bytearray_descr_sizeof(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_slot(args, 0)?;
    let alloc = unsafe { pyre_object::bytearrayobject::w_bytearray_capacity(args[0]) };
    let alloc = if alloc == 0 { 0 } else { alloc + 1 };
    // Header + data/export fields + the separately allocated Vec descriptor,
    // followed by its reserved byte payload.
    let fixed =
        pyre_object::bytearrayobject::W_BYTEARRAY_OBJECT_SIZE + std::mem::size_of::<Vec<u8>>();
    Ok(w_int_new((fixed + alloc) as i64))
}

fn bytearray_descr_resize(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::arity_slot(args, 1)?;
    let size = crate::builtins::space_index_w(args[1])?;
    if size < 0 {
        return Err(crate::PyError::value_error(format!(
            "Can only resize to positive sizes, got {size}"
        )));
    }
    unsafe {
        crate::builtins::bytearray_check_exports(args[0])?;
        pyre_object::bytearrayobject::w_bytearray_vec_mut(args[0]).resize(size as usize, 0);
    }
    Ok(w_none())
}

/// PyPy: bytearrayobject.py W_BytearrayObject.typedef
fn init_bytearray_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new(
                "bytearray(iterable_of_ints) -> bytearray\n\
                 bytearray(string, encoding[, errors]) -> bytearray\n\
                 bytearray(bytes_or_buffer) -> mutable copy of bytes_or_buffer\n\
                 bytearray(int) -> bytes array of size given by the parameter initialized with null bytes\n\
                 bytearray() -> empty bytes array\n\n\
                 Construct a mutable bytearray object from:\n\
                   - an iterable yielding integers in range(256)\n\
                   - a text string encoded using the specified encoding\n\
                   - a bytes or a buffer object\n\
                   - any object implementing the buffer API.\n\
                   - an integer",
            ),
        )
    };
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, "__hash__", w_none()) };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(bytearray_descr_new),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__init__",
            make_builtin_function("__init__", bytearray_descr_init),
        )
    };
    for (name, function, arity) in [
        ("__repr__", bytearray_descr_repr as DunderFn, 1),
        ("__str__", bytearray_descr_repr, 1),
        ("__reduce__", bytearray_descr_reduce, 1),
        ("__reduce_ex__", bytearray_descr_reduce_ex, 2),
        ("__alloc__", bytearray_descr_alloc, 1),
        ("__sizeof__", bytearray_descr_sizeof, 1),
        ("resize", bytearray_descr_resize, 2),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, function, arity),
            )
        };
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__buffer__",
            make_builtin_function_with_arity(
                "__buffer__",
                |args| crate::builtins::w_memoryview_new(args[0]),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__mod__",
            make_builtin_function_with_arity(
                "__mod__",
                |args| unsafe {
                    crate::objspace::std::formatting::bytes_format_percent(args[0], args[1])
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__rmod__",
            make_builtin_function_with_arity(
                "__rmod__",
                |args| {
                    if unsafe { pyre_object::is_bytearray(args[1]) } {
                        unsafe {
                            crate::objspace::std::formatting::bytes_format_percent(args[1], args[0])
                        }
                    } else {
                        Ok(w_not_implemented())
                    }
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__release_buffer__",
            make_builtin_function_with_arity(
                "__release_buffer__",
                bytearray_method_release_buffer,
                2,
            ),
        )
    };
    // `bytearrayobject.py W_BytearrayObject.descr_decode` shares the
    // bytes decode machinery — `bytes_method_decode` already pulls the
    // payload via `bytes_like_data`, which handles both kinds.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "decode",
            make_builtin_function("decode", bytes_method_decode),
        )
    };
    // The scalar-returning read-only methods (int / bool results) read
    // their payload via `bytes_like_data`, which handles both bytes and
    // bytearray, so they share the bytes implementations verbatim.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "find",
            make_builtin_function("find", bytes_method_find),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rfind",
            make_builtin_function("rfind", bytes_method_rfind),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "index",
            make_builtin_function("index", bytes_method_index),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rindex",
            make_builtin_function("rindex", bytes_method_rindex),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "count",
            make_builtin_function("count", bytes_method_count),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "startswith",
            make_builtin_function("startswith", bytes_method_startswith),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "endswith",
            make_builtin_function("endswith", bytes_method_endswith),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isdigit",
            make_builtin_function("isdigit", bytes_method_isdigit),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isalpha",
            make_builtin_function("isalpha", bytes_method_isalpha),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isalnum",
            make_builtin_function("isalnum", bytes_method_isalnum),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isspace",
            make_builtin_function("isspace", bytes_method_isspace),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isascii",
            make_builtin_function("isascii", bytes_method_isascii),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isupper",
            make_builtin_function("isupper", bytes_method_isupper),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "islower",
            make_builtin_function("islower", bytes_method_islower),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "istitle",
            make_builtin_function("istitle", bytes_method_istitle),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                        // `descr_add` returns NotImplemented for a non-buffer
                        // operand so the `+` operator raises the generic TypeError.
                        let Some(src) = buffer_as_bytes_like(b)? else {
                            return Ok(pyre_object::w_not_implemented());
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                        // `descr_inplace_add` requires a bytes-like operand; a
                        // non-buffer one raises rather than silently leaving the
                        // bytearray unchanged.
                        let Some(src) = buffer_as_bytes_like(other)? else {
                            return Err(crate::PyError::new(
                                crate::PyErrorKind::TypeError,
                                format!(
                                    "a bytes-like object is required, not '{}'",
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__imul__",
            make_builtin_function_with_arity("__imul__", bytearray_method_imul, 2),
        )
    };
    // The transform methods read via `bytes_like_data` and build their
    // result with `new_bytes_like`, which yields a bytearray for a
    // bytearray receiver, so they share the bytes implementations.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "translate",
            make_builtin_function("translate", bytes_method_translate),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "upper",
            make_builtin_function("upper", bytes_method_upper),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "lower",
            make_builtin_function("lower", bytes_method_lower),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "strip",
            make_builtin_function("strip", bytes_method_strip),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "lstrip",
            make_builtin_function("lstrip", bytes_method_lstrip),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rstrip",
            make_builtin_function("rstrip", bytes_method_rstrip),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "replace",
            make_builtin_function("replace", bytes_method_replace),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "split",
            make_builtin_function("split", bytes_method_split),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rsplit",
            make_builtin_function("rsplit", bytes_method_rsplit),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "splitlines",
            make_builtin_function("splitlines", bytes_method_splitlines),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "join",
            make_builtin_function("join", bytes_method_join),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "partition",
            make_builtin_function("partition", bytes_method_partition),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rpartition",
            make_builtin_function("rpartition", bytes_method_rpartition),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "title",
            make_builtin_function("title", bytes_method_title),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "capitalize",
            make_builtin_function("capitalize", bytes_method_capitalize),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "swapcase",
            make_builtin_function("swapcase", bytes_method_swapcase),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "removeprefix",
            make_builtin_function("removeprefix", bytes_method_removeprefix),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "removesuffix",
            make_builtin_function("removesuffix", bytes_method_removesuffix),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "ljust",
            make_builtin_function("ljust", bytes_method_ljust),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "rjust",
            make_builtin_function("rjust", bytes_method_rjust),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "center",
            make_builtin_function("center", bytes_method_center),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "zfill",
            make_builtin_function("zfill", bytes_method_zfill),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "expandtabs",
            make_builtin_function("expandtabs", bytes_method_expandtabs),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "hex",
            make_builtin_function("hex", bytes_method_hex),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "maketrans",
            make_maketrans_descr(bytes_maketrans),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "fromhex",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "fromhex",
                bytearray_fromhex,
            )),
        )
    };
    // In-place mutators specific to the mutable bytearray.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "append",
            make_builtin_function("append", bytearray_method_append),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "extend",
            make_builtin_function("extend", bytearray_method_extend),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "insert",
            make_builtin_function("insert", bytearray_method_insert),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "remove",
            make_builtin_function("remove", bytearray_method_remove),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "pop",
            make_builtin_function("pop", bytearray_method_pop),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "reverse",
            make_builtin_function("reverse", bytearray_method_reverse),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "clear",
            make_builtin_function("clear", bytearray_method_clear),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "copy",
            make_builtin_function("copy", bytearray_method_copy),
        )
    };
    // Subscript slots exposed as callable dunders.  Each binds the direct
    // slot body so a subclass override's `super().__getitem__` reaches the
    // inherited builtin subscript instead of re-entering override dispatch.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // `bytes_descr_repeat` builds its result via `bytes_repeat`, which yields a
    // bytearray for a bytearray receiver, so the repeat dunders are shared.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__mul__",
            make_builtin_function_with_arity("__mul__", |args| bytes_descr_repeat(args), 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__rmul__",
            make_builtin_function_with_arity("__rmul__", |args| bytes_descr_repeat(args), 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__imul__",
            make_builtin_function_with_arity(
                "__imul__",
                |args| {
                    // descr_inplace_mul: the count goes through `__index__`; a
                    // non-index operand becomes NotImplemented.
                    crate::type_methods::arity_slot(args, 1)?;
                    let Some(w_count) = list_repeat_index(args[1])? else {
                        return Ok(pyre_object::w_not_implemented());
                    };
                    unsafe {
                        crate::objspace::descroperation::bytearray_inplace_repeat(args[0], w_count)?
                    };
                    Ok(args[0])
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    for (name, func) in [
        ("__eq__", bytearray_dunder_eq as DunderFn),
        ("__ne__", bytearray_dunder_ne),
        ("__lt__", bytearray_dunder_lt),
        ("__le__", bytearray_dunder_le),
        ("__gt__", bytearray_dunder_gt),
        ("__ge__", bytearray_dunder_ge),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, func, 2),
            )
        };
    }
}

// ── set / frozenset TypeDef ──────────────────────────────────────────
// PyPy: pypy/objspace/std/setobject.py W_BaseSetObject.typedef
// pyre splits the shared methods through `init_setlike_common` so the
// frozenset typedef can omit the in-place mutators.

/// `setobject.py _convert_set_to_frozenset` — a set is unhashable, but
/// it stands in for the frozenset holding the same elements when it is used
/// to look one up. Returns `None` for anything that is not a set, leaving the
/// caller to re-raise. Upstream hands the strategy and storage to the new
/// frozenset rather than copying; the elements hashed when they entered the
/// set, so sharing them cannot raise.
fn convert_set_to_frozenset(w_obj: PyObjectRef) -> Option<PyObjectRef> {
    unsafe {
        if !pyre_object::is_set(w_obj) {
            return None;
        }
        let w_frozenset = pyre_object::w_frozenset_new();
        pyre_object::w_set_copy_storage_from(w_frozenset, w_obj);
        Some(w_frozenset)
    }
}

/// `setobject.py EmptySetStrategy.has_key` hashes the key ("make sure
/// the key is hashable, issue 3824"), so membership hashes even against an
/// empty set, unlike removal.
///
/// `setobject.py W_BaseSetObject.descr_contains`.
pub(crate) fn set_descr_contains(
    w_set: PyObjectRef,
    w_other: PyObjectRef,
) -> Result<bool, crate::PyError> {
    match crate::type_methods::set_contains_checked(w_set, w_other) {
        Ok(found) => Ok(found),
        Err(e) => {
            if e.kind == crate::PyErrorKind::TypeError {
                if let Some(w_f) = convert_set_to_frozenset(w_other) {
                    return crate::type_methods::set_contains_checked(w_set, w_f);
                }
            }
            Err(e)
        }
    }
}

/// `setobject.py EmptySetStrategy.remove` returns False without
/// hashing, so an empty set removes nothing and never raises. Every other
/// strategy hashes; pyre carries no strategies, so the length stands in for
/// the strategy dispatch.
fn set_remove(w_set: PyObjectRef, w_item: PyObjectRef) -> Result<bool, crate::PyError> {
    if unsafe { pyre_object::w_set_len(w_set) } == 0 {
        return Ok(false);
    }
    crate::type_methods::set_discard_checked(w_set, w_item)
}

/// Discard an element from a set, with automatic conversion to frozenset if
/// the argument is a set. Returns true if successfully removed.
///
/// `setobject.py W_BaseSetObject._discard_from_set`. Upstream's trailing
/// `switch_to_empty_strategy` has no counterpart here.
fn set_discard_from_set(w_set: PyObjectRef, w_item: PyObjectRef) -> Result<bool, crate::PyError> {
    match set_remove(w_set, w_item) {
        Ok(deleted) => Ok(deleted),
        Err(e) => {
            if e.kind != crate::PyErrorKind::TypeError {
                return Err(e);
            }
            match convert_set_to_frozenset(w_item) {
                None => Err(e),
                Some(w_f) => set_remove(w_set, w_f),
            }
        }
    }
}

fn setlike_descr_len(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(pyre_object::w_int_new(
        unsafe { pyre_object::w_set_len(args[0]) } as i64,
    ))
}

fn setlike_descr_iter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(pyre_object::w_set_iter_new(args[0]))
}

fn setlike_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unsafe { Ok(pyre_object::w_str_new(&crate::display::py_repr(args[0])?)) }
}

fn setlike_descr_sizeof(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let size = std::mem::size_of::<pyre_object::setobject::W_SetObject>()
        + unsafe { pyre_object::w_set_capacity(args[0]) }
            * std::mem::size_of::<pyre_object::dictmultiobject::ObjectKey>();
    Ok(pyre_object::w_int_new(size as i64))
}

fn setlike_descr_contains_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(pyre_object::w_bool_from(set_descr_contains(
        args[0], args[1],
    )?))
}

fn setlike_descr_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::reduce_protocol::set_reduce(args[0])
}

fn setlike_descr_isdisjoint(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::is_set_or_frozenset(args[1]) } {
        return Ok(pyre_object::w_bool_from(set_is_disjoint_from(
            args[0], args[1],
        )?));
    }
    for item in crate::builtins::collect_iterable(args[1])? {
        if crate::type_methods::set_contains_checked(args[0], item)? {
            return Ok(pyre_object::w_bool_from(false));
        }
    }
    Ok(pyre_object::w_bool_from(true))
}

fn setlike_descr_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name = if unsafe { pyre_object::is_frozenset(args[0]) } {
        "frozenset.copy"
    } else {
        "set.copy"
    };
    crate::type_methods::arity_no_args(args, name)?;
    if unsafe { pyre_object::is_exact_type(args[0], &pyre_object::setobject::FROZENSET_TYPE) } {
        return Ok(args[0]);
    }
    Ok(set_copy_real(args[0]))
}

macro_rules! setlike_wrapper_gateways {
    ($set_fn:ident, $frozenset_fn:ident, $name:literal, $implementation:ident) => {
        fn $set_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::require_set_receiver(args, $name, false)?;
            $implementation(args)
        }

        fn $frozenset_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::require_frozenset_receiver(args, $name, false)?;
            $implementation(args)
        }
    };
}

macro_rules! setlike_method_gateways {
    ($set_fn:ident, $frozenset_fn:ident, $name:literal, $implementation:ident) => {
        fn $set_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::require_set_receiver(args, $name, true)?;
            $implementation(args)
        }

        fn $frozenset_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            crate::type_methods::require_frozenset_receiver(args, $name, true)?;
            $implementation(args)
        }
    };
}

setlike_wrapper_gateways!(
    set_gateway_len,
    frozenset_gateway_len,
    "__len__",
    setlike_descr_len
);
setlike_wrapper_gateways!(
    set_gateway_iter,
    frozenset_gateway_iter,
    "__iter__",
    setlike_descr_iter
);
setlike_wrapper_gateways!(
    set_gateway_repr,
    frozenset_gateway_repr,
    "__repr__",
    setlike_descr_repr
);
setlike_wrapper_gateways!(set_gateway_or, frozenset_gateway_or, "__or__", set_op_or);
setlike_wrapper_gateways!(
    set_gateway_and,
    frozenset_gateway_and,
    "__and__",
    set_op_and
);
setlike_wrapper_gateways!(
    set_gateway_sub,
    frozenset_gateway_sub,
    "__sub__",
    set_op_sub
);
setlike_wrapper_gateways!(
    set_gateway_xor,
    frozenset_gateway_xor,
    "__xor__",
    set_op_xor
);
setlike_wrapper_gateways!(
    set_gateway_rsub,
    frozenset_gateway_rsub,
    "__rsub__",
    set_op_rsub
);
setlike_wrapper_gateways!(
    set_gateway_rand,
    frozenset_gateway_rand,
    "__rand__",
    set_op_and
);
setlike_wrapper_gateways!(set_gateway_ror, frozenset_gateway_ror, "__ror__", set_op_or);
setlike_wrapper_gateways!(
    set_gateway_rxor,
    frozenset_gateway_rxor,
    "__rxor__",
    set_op_xor
);
setlike_wrapper_gateways!(set_gateway_eq, frozenset_gateway_eq, "__eq__", set_descr_eq);
setlike_wrapper_gateways!(set_gateway_ne, frozenset_gateway_ne, "__ne__", set_descr_ne);
setlike_wrapper_gateways!(set_gateway_le, frozenset_gateway_le, "__le__", set_descr_le);
setlike_wrapper_gateways!(set_gateway_ge, frozenset_gateway_ge, "__ge__", set_descr_ge);
setlike_wrapper_gateways!(set_gateway_lt, frozenset_gateway_lt, "__lt__", set_descr_lt);
setlike_wrapper_gateways!(set_gateway_gt, frozenset_gateway_gt, "__gt__", set_descr_gt);
setlike_method_gateways!(
    set_gateway_sizeof,
    frozenset_gateway_sizeof,
    "__sizeof__",
    setlike_descr_sizeof
);
setlike_method_gateways!(
    set_gateway_contains,
    frozenset_gateway_contains,
    "__contains__",
    setlike_descr_contains_impl
);
setlike_method_gateways!(
    set_gateway_reduce,
    frozenset_gateway_reduce,
    "__reduce__",
    setlike_descr_reduce
);
setlike_method_gateways!(
    set_gateway_union,
    frozenset_gateway_union,
    "union",
    set_method_union
);
setlike_method_gateways!(
    set_gateway_intersection,
    frozenset_gateway_intersection,
    "intersection",
    set_method_intersection
);
setlike_method_gateways!(
    set_gateway_difference,
    frozenset_gateway_difference,
    "difference",
    set_method_difference
);
setlike_method_gateways!(
    set_gateway_symmetric_difference,
    frozenset_gateway_symmetric_difference,
    "symmetric_difference",
    set_method_symmetric_difference
);
setlike_method_gateways!(
    set_gateway_issubset,
    frozenset_gateway_issubset,
    "issubset",
    set_method_le
);
setlike_method_gateways!(
    set_gateway_issuperset,
    frozenset_gateway_issuperset,
    "issuperset",
    set_method_ge
);
setlike_method_gateways!(
    set_gateway_isdisjoint,
    frozenset_gateway_isdisjoint,
    "isdisjoint",
    setlike_descr_isdisjoint
);
setlike_method_gateways!(
    set_gateway_copy,
    frozenset_gateway_copy,
    "copy",
    setlike_descr_copy
);

fn setlike_gateway(
    frozen: bool,
    set_gateway: crate::gateway::BuiltinCodeFn,
    frozenset_gateway: crate::gateway::BuiltinCodeFn,
) -> crate::gateway::BuiltinCodeFn {
    if frozen {
        frozenset_gateway
    } else {
        set_gateway
    }
}

fn init_setlike_common(ns: PyObjectRef, frozen: bool) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__sizeof__",
            make_builtin_function_with_arity(
                "__sizeof__",
                setlike_gateway(frozen, set_gateway_sizeof, frozenset_gateway_sizeof),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__contains__",
            make_builtin_function_with_arity(
                "__contains__",
                setlike_gateway(frozen, set_gateway_contains, frozenset_gateway_contains),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__len__",
            make_builtin_function_with_arity(
                "__len__",
                setlike_gateway(frozen, set_gateway_len, frozenset_gateway_len),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__iter__",
            make_builtin_function_with_arity(
                "__iter__",
                setlike_gateway(frozen, set_gateway_iter, frozenset_gateway_iter),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity(
                "__repr__",
                setlike_gateway(frozen, set_gateway_repr, frozenset_gateway_repr),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                setlike_gateway(frozen, set_gateway_reduce, frozenset_gateway_reduce),
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__or__",
            make_builtin_function_with_arity(
                "__or__",
                setlike_gateway(frozen, set_gateway_or, frozenset_gateway_or),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__and__",
            make_builtin_function_with_arity(
                "__and__",
                setlike_gateway(frozen, set_gateway_and, frozenset_gateway_and),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__sub__",
            make_builtin_function_with_arity(
                "__sub__",
                setlike_gateway(frozen, set_gateway_sub, frozenset_gateway_sub),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__xor__",
            make_builtin_function_with_arity(
                "__xor__",
                setlike_gateway(frozen, set_gateway_xor, frozenset_gateway_xor),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__rsub__",
            make_builtin_function_with_arity(
                "__rsub__",
                setlike_gateway(frozen, set_gateway_rsub, frozenset_gateway_rsub),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__rand__",
            make_builtin_function_with_arity(
                "__rand__",
                setlike_gateway(frozen, set_gateway_rand, frozenset_gateway_rand),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ror__",
            make_builtin_function_with_arity(
                "__ror__",
                setlike_gateway(frozen, set_gateway_ror, frozenset_gateway_ror),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__rxor__",
            make_builtin_function_with_arity(
                "__rxor__",
                setlike_gateway(frozen, set_gateway_rxor, frozenset_gateway_rxor),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__eq__",
            make_builtin_function_with_arity(
                "__eq__",
                setlike_gateway(frozen, set_gateway_eq, frozenset_gateway_eq),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ne__",
            make_builtin_function_with_arity(
                "__ne__",
                setlike_gateway(frozen, set_gateway_ne, frozenset_gateway_ne),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__le__",
            make_builtin_function_with_arity(
                "__le__",
                setlike_gateway(frozen, set_gateway_le, frozenset_gateway_le),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ge__",
            make_builtin_function_with_arity(
                "__ge__",
                setlike_gateway(frozen, set_gateway_ge, frozenset_gateway_ge),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__lt__",
            make_builtin_function_with_arity(
                "__lt__",
                setlike_gateway(frozen, set_gateway_lt, frozenset_gateway_lt),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__gt__",
            make_builtin_function_with_arity(
                "__gt__",
                setlike_gateway(frozen, set_gateway_gt, frozenset_gateway_gt),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "union",
            make_builtin_function(
                "union",
                setlike_gateway(frozen, set_gateway_union, frozenset_gateway_union),
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "intersection",
            make_builtin_function(
                "intersection",
                setlike_gateway(
                    frozen,
                    set_gateway_intersection,
                    frozenset_gateway_intersection,
                ),
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "difference",
            make_builtin_function(
                "difference",
                setlike_gateway(frozen, set_gateway_difference, frozenset_gateway_difference),
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "symmetric_difference",
            make_builtin_function_with_arity(
                "symmetric_difference",
                setlike_gateway(
                    frozen,
                    set_gateway_symmetric_difference,
                    frozenset_gateway_symmetric_difference,
                ),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "issubset",
            make_builtin_function_with_arity(
                "issubset",
                setlike_gateway(frozen, set_gateway_issubset, frozenset_gateway_issubset),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "issuperset",
            make_builtin_function_with_arity(
                "issuperset",
                setlike_gateway(frozen, set_gateway_issuperset, frozenset_gateway_issuperset),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isdisjoint",
            make_builtin_function_with_arity(
                "isdisjoint",
                setlike_gateway(frozen, set_gateway_isdisjoint, frozenset_gateway_isdisjoint),
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "copy",
            // `setobject.py descr_copy` — a shallow copy, taking the
            // storage over rather than hashing the elements again.
            make_builtin_function_with_arity(
                "copy",
                setlike_gateway(frozen, set_gateway_copy, frozenset_gateway_copy),
                1,
            ),
        )
    };
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
fn set_op_rsub(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if set_op_requires_set(args) {
        return Ok(pyre_object::w_not_implemented());
    }
    set_method_difference(&[args[1], args[0]])
}
fn set_op_xor(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if set_op_requires_set(args) {
        return Ok(pyre_object::w_not_implemented());
    }
    set_method_symmetric_difference(args)
}

/// `setobject.py W_BaseSetObject.descr_union` — the result starts as a copy
/// of self and every operand is merged in: a set operand through its storage
/// (`:366`), any other iterable element by element (`:368-369`).
pub(crate) fn set_method_union(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_set_new());
    }
    let result = set_copy_real(args[0]);
    for other in &args[1..] {
        if unsafe { pyre_object::is_set_or_frozenset(*other) } {
            unsafe { pyre_object::w_set_update_from_set(result, *other) }
                .map_err(|_| crate::baseobjspace::take_pending_hash_error())?;
        } else {
            let other_items = crate::builtins::collect_iterable(*other)?;
            crate::builtins::builtin_set_add_items(result, &other_items)?;
        }
    }
    Ok(result)
}

/// A clone of the set, keeping the digest each element was stored under.
///
/// `setobject.py W_BaseSetObject.copy_real` — "returns a clone of the
/// set; frozensets storages are also copied". The clone keeps self's class, so
/// `frozenset.union` stays a frozenset.
fn set_copy_real(w_set: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    unsafe {
        let result = if pyre_object::is_frozenset(w_set) {
            pyre_object::w_frozenset_new()
        } else {
            pyre_object::w_set_new()
        };
        pyre_object::w_set_copy_storage_from(result, w_set);
        result
    }
}

/// Build a set holding `w_iterable`'s elements.
///
/// `setobject.py W_SetObject._newobj` / `setobject.py
/// W_FrozensetObject._newobj` — both take ownership of the iterable's
/// elements; only the resulting class differs, and the callers here read the
/// elements back rather than the object, so one set-shaped result serves both.
///
/// Both build the set and hand the iterable to its `__init__`, so the set
/// operand branch is the one `set_init_from_iterable` already carries.
fn set_newobj(
    w_iterable: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let w_set = pyre_object::w_set_new();
    set_init_from_iterable(w_set, w_iterable)?;
    Ok(w_set)
}

/// The operand as a set: itself when it already is one, otherwise a set built
/// from it.
///
/// `setobject.py descr_difference_update`, `:499-503
/// descr_symmetric_difference_update` and `:312-316 descr_intersection` all
/// open with this branch — a set operand is handed to the storage-level
/// operation as it stands, and only a non-set is walked and hashed into one.
fn set_operand_as_set(
    w_other: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::is_set_or_frozenset(w_other) } {
        return Ok(w_other);
    }
    set_newobj(w_other)
}

/// Keep only the elements the two sides share.
///
/// `setobject.py AbstractUnwrappedSetStrategy.intersect_update` swaps the
/// operands when self is the longer, and `setobject.py _intersect_base`
/// swaps again on the way through, so either way the shorter side is walked
/// and it is that side's objects the result holds. Equal elements can be
/// distinct objects, so which side is walked is observable; a tie walks self.
///
/// `setobject.py _intersect_unwrapped` walks the storage as
/// `(key, keyhash)` pairs and probes the other side with `contains_with_hash`,
/// so the elements are neither re-hashed nor handed to a user `__hash__` again;
/// only `eq_w` runs, from the bucket probes.
fn set_intersect_update(
    w_set: pyre_object::PyObjectRef,
    w_other: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    unsafe {
        let (keep, probe) = if pyre_object::w_set_len(w_set) > pyre_object::w_set_len(w_other) {
            (w_other, w_set)
        } else {
            (w_set, w_other)
        };
        let result = pyre_object::w_set_new();
        // The three sets are old-gen allocations and keep their addresses across
        // a collection, but their elements are young and move, so each key is
        // re-read from the table the collector rewrites rather than carried
        // across the `eq_w` a bucket probe can run.
        let mut i = 0;
        while let Some(key) = pyre_object::w_set_key_at(keep, i) {
            if pyre_object::w_set_contains_key_checked(probe, key)
                .map_err(|_| crate::baseobjspace::take_pending_hash_error())?
            {
                let Some(key) = pyre_object::w_set_key_at(keep, i) else {
                    break;
                };
                pyre_object::w_set_insert_key_checked(result, key)
                    .map_err(|_| crate::baseobjspace::take_pending_hash_error())?;
            }
            i += 1;
        }
        Ok(result)
    }
}

/// `setobject.py W_BaseSetObject.descr_intersection` — the shortest
/// operand seeds the result and the rest are intersected into it. Length is
/// measured on the operands as given, before any is turned into a set, so an
/// operand whose length cannot be taken (a generator) never seeds and a list
/// with duplicates is measured longer than the set it becomes. That measure
/// disagreeing with the set lengths is what leaves work for the swap in
/// `set_intersect_update`.
pub(crate) fn set_method_intersection(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_set_new());
    }
    let mut others_w: Vec<pyre_object::PyObjectRef> = args.to_vec();

    // find smallest set in others_w to reduce comparisons
    let mut startindex = 0usize;
    let mut startlength: i64 = -1;
    for i in 0..others_w.len() {
        let length = match crate::baseobjspace::len_w(others_w[i]) {
            Ok(length) => length,
            Err(e)
                if e.kind == crate::PyErrorKind::TypeError
                    || e.kind == crate::PyErrorKind::AttributeError =>
            {
                continue;
            }
            Err(e) => return Err(e),
        };
        if startlength == -1 || length < startlength {
            startindex = i;
            startlength = length;
        }
    }
    others_w.swap(0, startindex);

    // `setobject.py` — the seed and every operand become sets, and a
    // set operand is intersected as it stands rather than rebuilt.
    let mut result = set_newobj(others_w[0])?;
    for &w_other in &others_w[1..] {
        let w_other_as_set = if unsafe { pyre_object::is_set_or_frozenset(w_other) } {
            w_other
        } else {
            set_newobj(w_other)?
        };
        result = set_intersect_update(result, w_other_as_set)?;
    }
    unsafe {
        if pyre_object::is_frozenset(args[0]) {
            let w_frozenset = pyre_object::w_frozenset_new();
            pyre_object::w_set_copy_storage_from(w_frozenset, result);
            return Ok(w_frozenset);
        }
    }
    Ok(result)
}

/// `setobject.py W_BaseSetObject.descr_difference` — a copy of self with
/// `descr_difference_update` run over it.
pub(crate) fn set_method_difference(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_set_new());
    }
    let result = set_copy_real(args[0]);
    let mut update_args: Vec<pyre_object::PyObjectRef> = vec![result];
    update_args.extend_from_slice(&args[1..]);
    set_method_difference_update(&update_args)?;
    Ok(result)
}

pub(crate) fn set_method_symmetric_difference(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        if args.is_empty() {
            return Ok(pyre_object::w_set_new());
        }
        return Ok(args[0]);
    }
    // `setobject.py symmetric_difference` wraps the computed storage
    // in a set of self's class.
    let w_other_as_set = set_operand_as_set(args[1])?;
    let w_new = set_symmetric_difference_storage(args[0], w_other_as_set)?;
    unsafe {
        if pyre_object::is_frozenset(args[0]) {
            let w_frozenset = pyre_object::w_frozenset_new();
            pyre_object::w_set_copy_storage_from(w_frozenset, w_new);
            return Ok(w_frozenset);
        }
    }
    Ok(w_new)
}

/// Set rich comparisons accept only another set/frozenset.  The named
/// `issubset`/`issuperset` methods below deliberately have a different shape:
/// they materialize any iterable.  PyPy keeps these as distinct descr_* entry
/// points (`setobject.py`), so do not route the operator slots through
/// `set_method_le`/`set_method_ge`.
fn set_descr_eq(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 || !unsafe { pyre_object::is_set_or_frozenset(args[1]) } {
        return Ok(pyre_object::w_not_implemented());
    }
    set_method_eq(args)
}

fn set_descr_ne(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let w_eq = set_descr_eq(args)?;
    if unsafe { pyre_object::is_not_implemented(w_eq) } {
        return Ok(w_eq);
    }
    Ok(pyre_object::w_bool_from(!unsafe {
        pyre_object::w_bool_get_value(w_eq)
    }))
}

fn set_descr_le(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 || !unsafe { pyre_object::is_set_or_frozenset(args[1]) } {
        return Ok(pyre_object::w_not_implemented());
    }
    unsafe {
        if pyre_object::w_set_len(args[0]) > pyre_object::w_set_len(args[1]) {
            return Ok(pyre_object::w_bool_from(false));
        }
    }
    Ok(pyre_object::w_bool_from(set_is_subset_of(
        args[0], args[1],
    )?))
}

fn set_descr_lt(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 || !unsafe { pyre_object::is_set_or_frozenset(args[1]) } {
        return Ok(pyre_object::w_not_implemented());
    }
    unsafe {
        if pyre_object::w_set_len(args[0]) >= pyre_object::w_set_len(args[1]) {
            return Ok(pyre_object::w_bool_from(false));
        }
    }
    Ok(pyre_object::w_bool_from(set_is_subset_of(
        args[0], args[1],
    )?))
}

fn set_descr_ge(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 || !unsafe { pyre_object::is_set_or_frozenset(args[1]) } {
        return Ok(pyre_object::w_not_implemented());
    }
    unsafe {
        if pyre_object::w_set_len(args[0]) < pyre_object::w_set_len(args[1]) {
            return Ok(pyre_object::w_bool_from(false));
        }
    }
    Ok(pyre_object::w_bool_from(set_is_subset_of(
        args[1], args[0],
    )?))
}

fn set_descr_gt(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 || !unsafe { pyre_object::is_set_or_frozenset(args[1]) } {
        return Ok(pyre_object::w_not_implemented());
    }
    unsafe {
        if pyre_object::w_set_len(args[0]) <= pyre_object::w_set_len(args[1]) {
            return Ok(pyre_object::w_bool_from(false));
        }
    }
    Ok(pyre_object::w_bool_from(set_is_subset_of(
        args[1], args[0],
    )?))
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
    }
    Ok(pyre_object::w_bool_from(set_is_subset_of(
        args[0], args[1],
    )?))
}

/// Whether the two sets share no element. Both must be sets.
///
/// `setobject.py _isdisjoint_unwrapped` walks one storage as
/// `(key, keyhash)` pairs and probes the other with `contains_with_hash`;
/// `:1214-1215 isdisjoint` walks the shorter side.
fn set_is_disjoint_from(
    w_set: pyre_object::PyObjectRef,
    w_other: pyre_object::PyObjectRef,
) -> Result<bool, crate::PyError> {
    unsafe {
        let (walk, probe) = if pyre_object::w_set_len(w_set) > pyre_object::w_set_len(w_other) {
            (w_other, w_set)
        } else {
            (w_set, w_other)
        };
        let mut i = 0;
        while let Some(key) = pyre_object::w_set_key_at(walk, i) {
            if pyre_object::w_set_contains_key_checked(probe, key)
                .map_err(|_| crate::baseobjspace::take_pending_hash_error())?
            {
                return Ok(false);
            }
            i += 1;
        }
        Ok(true)
    }
}

/// Whether every element of `w_set` is in `w_other`. Both must be sets.
///
/// `setobject.py _issubset_unwrapped` walks self's storage as
/// `(key, keyhash)` pairs and probes with `contains_with_hash`, so a comparison
/// re-hashes nothing.
pub(crate) fn set_is_subset_of(
    w_set: pyre_object::PyObjectRef,
    w_other: pyre_object::PyObjectRef,
) -> Result<bool, crate::PyError> {
    unsafe {
        let mut i = 0;
        while let Some(key) = pyre_object::w_set_key_at(w_set, i) {
            if !pyre_object::w_set_contains_key_checked(w_other, key)
                .map_err(|_| crate::baseobjspace::take_pending_hash_error())?
            {
                return Ok(false);
            }
            i += 1;
        }
        Ok(true)
    }
}

fn set_method_le(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_bool_from(true));
    }
    let w_other_as_set = set_operand_as_set(args[1])?;
    Ok(pyre_object::w_bool_from(set_is_subset_of(
        args[0],
        w_other_as_set,
    )?))
}

fn set_method_ge(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(pyre_object::w_bool_from(true));
    }
    // `setobject.py descr_issuperset` — the operand becomes a set and
    // the subset test runs the other way round.
    let w_other_as_set = set_operand_as_set(args[1])?;
    Ok(pyre_object::w_bool_from(set_is_subset_of(
        w_other_as_set,
        args[0],
    )?))
}

// `setobject.py` W_BaseSetObject mutating helpers — shared by the
// `*_update` methods (which accept any iterable) and the in-place operator
// slots (which pre-filter their operand to a set/frozenset).
fn set_method_update(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_none());
    }
    // `setobject.py _descr_update` — a set operand's storage merges in
    // as it stands; only another iterable is walked and hashed element by
    // element.
    for other in &args[1..] {
        if unsafe { pyre_object::is_set_or_frozenset(*other) } {
            unsafe { pyre_object::w_set_update_from_set(args[0], *other) }
                .map_err(|_| crate::baseobjspace::take_pending_hash_error())?;
        } else {
            let other_items = crate::builtins::collect_iterable(*other)?;
            crate::builtins::builtin_set_add_items(args[0], &other_items)?;
        }
    }
    Ok(pyre_object::w_none())
}

/// `setobject.py W_BaseSetObject.descr_difference_update` — a non-set
/// operand is turned into a set first, so it is hashed and deduped before
/// anything is removed and a later unhashable element leaves self untouched.
/// A set operand is used as it stands (`:392-393`).
fn set_method_difference_update(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_none());
    }
    for other in &args[1..] {
        let w_other_as_set = set_operand_as_set(*other)?;
        unsafe { pyre_object::w_set_difference_update_from_set(args[0], w_other_as_set) }
            .map_err(|_| crate::baseobjspace::take_pending_hash_error())?;
    }
    Ok(pyre_object::w_none())
}

/// `setobject.py W_SetObject.descr_intersection_update` — the result of
/// `descr_intersection` replaces self's storage wholesale, so self keeps the
/// surviving objects of whichever operand seeded that result rather than its
/// own.
fn set_method_intersection_update(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_none());
    }
    // Every other operand becomes a set, hashing its elements, so an
    // unhashable one raises even when self is empty and there is nothing
    // left to compare against.
    let result = set_method_intersection(args)?;
    // `setobject.py` — self takes the result's storage over as it is;
    // the elements hashed on their way into it.
    unsafe { pyre_object::w_set_copy_storage_from(args[0], result) };
    Ok(pyre_object::w_none())
}

/// `setobject.py W_SetObject.descr_symmetric_difference_update` — a
/// non-set operand is turned into a set first, so it is hashed and deduped
/// before anything is toggled: a later unhashable element leaves self
/// untouched, and a duplicate toggles once rather than twice.
fn set_method_symmetric_difference_update(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    if args.is_empty() || args.len() < 2 {
        return Ok(pyre_object::w_none());
    }
    let w_other_as_set = set_operand_as_set(args[1])?;
    let w_new = set_symmetric_difference_storage(args[0], w_other_as_set)?;
    // `setobject.py` — the computed storage replaces self's.
    unsafe { pyre_object::w_set_copy_storage_from(args[0], w_new) };
    Ok(pyre_object::w_none())
}

/// The elements on exactly one of the two sides, as a set.
///
/// `setobject.py _symmetric_difference_unwrapped` — each side is
/// walked as `(key, keyhash)` pairs and probed against the other with
/// `contains_with_hash`, and what survives is placed under the digest it
/// already carries. Membership is decided by the table, so an element only
/// counts as present when it lands in the same bucket and compares equal; a
/// bare `eq_w` scan over the elements would instead call two objects the same
/// element on `__eq__` alone, and place them where their hashes never meet.
fn set_symmetric_difference_storage(
    w_set: pyre_object::PyObjectRef,
    w_other: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    unsafe {
        let w_new = pyre_object::w_set_new();
        for (walk, probe) in [(w_other, w_set), (w_set, w_other)] {
            let mut i = 0;
            while let Some(key) = pyre_object::w_set_key_at(walk, i) {
                if !pyre_object::w_set_contains_key_checked(probe, key)
                    .map_err(|_| crate::baseobjspace::take_pending_hash_error())?
                {
                    // The probe's `eq_w` can move the element, so the key is
                    // re-read from the table the collector rewrites.
                    let Some(key) = pyre_object::w_set_key_at(walk, i) else {
                        break;
                    };
                    pyre_object::w_set_insert_key_checked(w_new, key)
                        .map_err(|_| crate::baseobjspace::take_pending_hash_error())?;
                }
                i += 1;
            }
        }
        Ok(w_new)
    }
}

// `setobject.py` W_SetObject.descr_inplace_sub / _and / _or / _xor — a
// non-set/-frozenset operand yields NotImplemented; otherwise mutate self
// through the matching update helper and return self.
fn set_op_inplace_sub(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    crate::type_methods::require_set_receiver(args, "__isub__", false)?;
    if set_op_requires_set(args) {
        return Ok(pyre_object::w_not_implemented());
    }
    set_method_difference_update(args)?;
    Ok(args[0])
}
fn set_op_inplace_and(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    crate::type_methods::require_set_receiver(args, "__iand__", false)?;
    if set_op_requires_set(args) {
        return Ok(pyre_object::w_not_implemented());
    }
    set_method_intersection_update(args)?;
    Ok(args[0])
}
fn set_op_inplace_or(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    crate::type_methods::require_set_receiver(args, "__ior__", false)?;
    if set_op_requires_set(args) {
        return Ok(pyre_object::w_not_implemented());
    }
    set_method_update(args)?;
    Ok(args[0])
}
fn set_op_inplace_xor(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    crate::type_methods::require_set_receiver(args, "__ixor__", false)?;
    if set_op_requires_set(args) {
        return Ok(pyre_object::w_not_implemented());
    }
    set_method_symmetric_difference_update(args)?;
    Ok(args[0])
}

fn init_set_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::w_dict_setitem_str(
            ns,
            "__doc__",
            pyre_object::w_str_new("Build an unordered collection of unique elements."),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(set_descr_new),
        )
    };
    // setobject.py __class_getitem__ = gateway.interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                crate::_pypy_generic_alias::generic_alias_class_getitem,
            )),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__init__",
            make_builtin_function("__init__", set_descr_init),
        )
    };
    init_setlike_common(ns, false);
    // setobject.py `__hash__ = None` — keep the slot visible to
    // introspection as well as the unhashable fast path in builtin_hash.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            pyre_object::w_none(),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "add",
            make_builtin_function_with_arity(
                "add",
                |args| {
                    crate::type_methods::require_set_receiver(args, "add", true)?;
                    crate::type_methods::arity_exact(args, "set.add", 1)?;
                    // `try_hash_value` may run a user `__hash__` that
                    // allocates and triggers a moving minor collection;
                    // root `self` and the element across it, then reload.
                    // Its digest keys the store, so the element is hashed
                    // once.
                    unsafe {
                        let _roots = pyre_object::gc_roots::push_roots();
                        let sp = pyre_object::gc_roots::shadow_stack_len();
                        pyre_object::gc_roots::pin_root(args[0]);
                        pyre_object::gc_roots::pin_root(args[1]);
                        let hash = crate::builtins::try_hash_value(args[1]).map_err(|err| {
                            crate::baseobjspace::wrap_set_element_hash_error(
                                pyre_object::gc_roots::shadow_stack_get(sp + 1),
                                err,
                            )
                        })?;
                        let set = pyre_object::gc_roots::shadow_stack_get(sp);
                        let item = pyre_object::gc_roots::shadow_stack_get(sp + 1);
                        pyre_object::w_set_add_hashed_checked(set, item, hash)
                            .map_err(|_| crate::baseobjspace::take_pending_hash_error())?;
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "discard",
            make_builtin_function_with_arity(
                "discard",
                |args| {
                    crate::type_methods::require_set_receiver(args, "discard", true)?;
                    crate::type_methods::arity_exact(args, "set.discard", 1)?;
                    set_discard_from_set(args[0], args[1])?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "remove",
            make_builtin_function_with_arity(
                "remove",
                |args| {
                    crate::type_methods::require_set_receiver(args, "remove", true)?;
                    crate::type_methods::arity_exact(args, "set.remove", 1)?;
                    if !set_discard_from_set(args[0], args[1])? {
                        return Err(crate::PyError::key_error_with_key(args[1]));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "pop",
            make_builtin_function_with_arity(
                "pop",
                |args| {
                    crate::type_methods::require_set_receiver(args, "pop", true)?;
                    crate::type_methods::arity_no_args(args, "set.pop")?;
                    if let Some(item) = unsafe { pyre_object::w_set_popitem(args[0]) } {
                        return Ok(item);
                    }
                    Err(crate::PyError::new(
                        crate::PyErrorKind::KeyError,
                        "pop from an empty set",
                    ))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "clear",
            make_builtin_function_with_arity(
                "clear",
                |args| {
                    crate::type_methods::require_set_receiver(args, "clear", true)?;
                    crate::type_methods::arity_no_args(args, "set.clear")?;
                    unsafe { pyre_object::w_set_clear(args[0]) };
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "update",
            make_builtin_function("update", |args| {
                crate::type_methods::require_set_receiver(args, "update", true)?;
                set_method_update(args)
            }),
        )
    };
    // `setobject.py W_BaseSetObject.descr_difference_update` /
    // `:1217 descr_intersection_update` / `:1244
    // descr_symmetric_difference_update` — in-place set ops that mirror the
    // non-update variants but mutate `self` instead of returning a fresh set.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "difference_update",
            make_builtin_function("difference_update", |args| {
                crate::type_methods::require_set_receiver(args, "difference_update", true)?;
                set_method_difference_update(args)
            }),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "intersection_update",
            make_builtin_function("intersection_update", |args| {
                crate::type_methods::require_set_receiver(args, "intersection_update", true)?;
                set_method_intersection_update(args)
            }),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "symmetric_difference_update",
            make_builtin_function("symmetric_difference_update", |args| {
                crate::type_methods::require_set_receiver(
                    args,
                    "symmetric_difference_update",
                    true,
                )?;
                set_method_symmetric_difference_update(args)
            }),
        )
    };
    // `setobject.py` __isub__/__iand__/__ior__/__ixor__ — mutable-set-only
    // in-place operator slots.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__isub__",
            make_builtin_function_with_arity("__isub__", set_op_inplace_sub, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__iand__",
            make_builtin_function_with_arity("__iand__", set_op_inplace_and, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ior__",
            make_builtin_function_with_arity("__ior__", set_op_inplace_or, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__ixor__",
            make_builtin_function_with_arity("__ixor__", set_op_inplace_xor, 2),
        )
    };
}

fn init_frozenset_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::w_dict_setitem_str(
            ns,
            "__doc__",
            pyre_object::w_str_new("Build an immutable unordered collection of unique elements."),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            make_new_descr(frozenset_descr_new),
        )
    };
    // setobject.py __class_getitem__ = gateway.interp2app(
    //     generic_alias_class_getitem, as_classmethod=True)
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                crate::_pypy_generic_alias::generic_alias_class_getitem,
            )),
        )
    };
    init_setlike_common(ns, true);
    // setobject.py descr_hash.  The Result-bearing hash helper walks the
    // elements and propagates an element hash error; the storage itself is not
    // rebuilt and no set element is re-inserted.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__hash__",
            make_builtin_function_with_arity(
                "__hash__",
                |args| {
                    crate::type_methods::require_frozenset_receiver(args, "__hash__", false)?;
                    Ok(pyre_object::w_int_new(crate::builtins::try_hash_value(
                        args[0],
                    )?))
                },
                1,
            ),
        )
    };
}

fn set_iter_self(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_set_iterator_receiver(args, "__iter__", false)
}

fn set_iter_next(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_set_iterator_receiver(args, "__next__", false)?;
    crate::baseobjspace::next(args[0])
}

fn set_iter_length_hint(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_set_iterator_receiver(args, "__length_hint__", true)?;
    unsafe {
        let w_set = pyre_object::w_set_iter_get_set(args[0]);
        let startlen = pyre_object::w_set_iter_get_startlen(args[0]);
        if w_set.is_null() || startlen == usize::MAX || pyre_object::w_set_len(w_set) != startlen {
            return Ok(pyre_object::w_int_new(0));
        }
        let index = pyre_object::w_set_iter_get_index(args[0]);
        Ok(pyre_object::w_int_new(startlen.saturating_sub(index) as i64))
    }
}

/// setobject.py `W_SetIterObject.descr_reduce`: materialize only
/// the clone's remaining entries, then return `(iter, (list,))`.
fn set_iter_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::type_methods::require_set_iterator_receiver(args, "__reduce__", true)?;
    unsafe {
        let w_set = pyre_object::w_set_iter_get_set(args[0]);
        let startlen = pyre_object::w_set_iter_get_startlen(args[0]);
        if w_set.is_null() {
            let state = pyre_object::w_tuple_new(vec![pyre_object::w_list_new(vec![])]);
            return Ok(pyre_object::w_tuple_new(vec![
                crate::baseobjspace::builtin_callable("iter"),
                state,
            ]));
        }
        if startlen == usize::MAX || pyre_object::w_set_len(w_set) != startlen {
            return Err(crate::PyError::new(
                crate::PyErrorKind::RuntimeError,
                "Set changed size during iteration",
            ));
        }
        let index = pyre_object::w_set_iter_get_index(args[0]);
        let mut remaining = Vec::with_capacity(startlen.saturating_sub(index));
        for i in index..startlen {
            if let Some(key) = pyre_object::w_set_key_at(w_set, i) {
                remaining.push(key.obj);
            }
        }
        let state = pyre_object::w_tuple_new(vec![pyre_object::w_list_new(remaining)]);
        Ok(pyre_object::w_tuple_new(vec![
            crate::baseobjspace::builtin_callable("iter"),
            state,
        ]))
    }
}

fn init_set_iterator_type(ns: PyObjectRef) {
    unsafe { pyre_object::w_dict_setitem_str(ns, "__doc__", pyre_object::w_none()) };
    let entries = [
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", set_iter_self, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", set_iter_next, 1),
        ),
        (
            "__length_hint__",
            make_builtin_function_with_arity("__length_hint__", set_iter_length_hint, 1),
        ),
        (
            "__reduce__",
            make_builtin_function_with_arity("__reduce__", set_iter_reduce, 1),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn generator_frame(obj: PyObjectRef) -> *mut crate::pyframe::PyFrame {
    unsafe { pyre_object::generator::w_generator_get_frame(obj) as *mut crate::pyframe::PyFrame }
}

fn generator_descr_repr(args: &[PyObjectRef]) -> crate::PyResult {
    let name = generator_name_value(args[0], true)?;
    Ok(w_str_new(&format!(
        "<generator object {} at {:p}>",
        unsafe { pyre_object::w_str_get_value(name) },
        args[0]
    )))
}

fn coroutine_descr_repr(args: &[PyObjectRef]) -> crate::PyResult {
    let name = generator_name_value(args[0], true)?;
    Ok(w_str_new(&format!(
        "<coroutine object {} at {:p}>",
        unsafe { pyre_object::w_str_get_value(name) },
        args[0]
    )))
}

fn generator_name_value(obj: PyObjectRef, qualname: bool) -> crate::PyResult {
    let override_value = unsafe {
        if qualname {
            pyre_object::generator::w_generator_get_qualname(obj)
        } else {
            pyre_object::generator::w_generator_get_name(obj)
        }
    };
    if !override_value.is_null() {
        return Ok(override_value);
    }
    let frame = generator_frame(obj);
    if frame.is_null() {
        return Ok(w_str_new("<generator>"));
    }
    let code = unsafe { (*frame).code() };
    Ok(w_str_new(if qualname {
        &code.qualname
    } else {
        &code.obj_name
    }))
}

fn generator_getter_for(args: &[PyObjectRef], field: usize, coroutine: bool) -> crate::PyResult {
    let obj = args.get(1).copied().unwrap_or(PY_NULL);
    let matches = unsafe {
        if coroutine {
            pyre_object::generator::is_coroutine(obj)
        } else {
            pyre_object::generator::is_generator(obj)
        }
    };
    if !matches {
        return Err(crate::PyError::type_error(format!(
            "descriptor is for '{}'",
            if coroutine { "coroutine" } else { "generator" }
        )));
    }
    let frame = generator_frame(obj);
    Ok(match field {
        0 => w_bool_from(unsafe { pyre_object::generator::w_generator_is_running(obj) }),
        1 => w_bool_from(unsafe {
            pyre_object::generator::w_generator_is_started(obj)
                && !pyre_object::generator::w_generator_is_running(obj)
                && !pyre_object::generator::w_generator_is_exhausted(obj)
        }),
        2 => {
            if frame.is_null() || unsafe { pyre_object::generator::w_generator_is_exhausted(obj) } {
                w_none()
            } else {
                frame as PyObjectRef
            }
        }
        3 => {
            if frame.is_null() {
                w_none()
            } else {
                unsafe { (*frame).pycode as PyObjectRef }
            }
        }
        4 => {
            if frame.is_null() {
                w_none()
            } else {
                let delegated = unsafe { (*frame).w_yielding_from };
                if delegated.is_null() {
                    w_none()
                } else {
                    delegated
                }
            }
        }
        5 => return generator_name_value(obj, false),
        _ => return generator_name_value(obj, true),
    })
}

fn generator_getter(args: &[PyObjectRef], field: usize) -> crate::PyResult {
    generator_getter_for(args, field, false)
}

fn coroutine_getter(args: &[PyObjectRef], field: usize) -> crate::PyResult {
    generator_getter_for(args, field, true)
}

fn generator_get_running(args: &[PyObjectRef]) -> crate::PyResult {
    generator_getter(args, 0)
}
fn generator_get_suspended(args: &[PyObjectRef]) -> crate::PyResult {
    generator_getter(args, 1)
}
fn generator_get_frame(args: &[PyObjectRef]) -> crate::PyResult {
    generator_getter(args, 2)
}
fn generator_get_code(args: &[PyObjectRef]) -> crate::PyResult {
    generator_getter(args, 3)
}
fn generator_get_yieldfrom(args: &[PyObjectRef]) -> crate::PyResult {
    generator_getter(args, 4)
}
fn generator_get_name(args: &[PyObjectRef]) -> crate::PyResult {
    generator_getter(args, 5)
}
fn generator_get_qualname(args: &[PyObjectRef]) -> crate::PyResult {
    generator_getter(args, 6)
}
fn coroutine_get_running(args: &[PyObjectRef]) -> crate::PyResult {
    coroutine_getter(args, 0)
}
fn coroutine_get_suspended(args: &[PyObjectRef]) -> crate::PyResult {
    coroutine_getter(args, 1)
}
fn coroutine_get_frame(args: &[PyObjectRef]) -> crate::PyResult {
    coroutine_getter(args, 2)
}
fn coroutine_get_code(args: &[PyObjectRef]) -> crate::PyResult {
    coroutine_getter(args, 3)
}
fn coroutine_get_await(args: &[PyObjectRef]) -> crate::PyResult {
    coroutine_getter(args, 4)
}
fn coroutine_get_origin(args: &[PyObjectRef]) -> crate::PyResult {
    let obj = args.get(1).copied().unwrap_or(PY_NULL);
    if !unsafe { pyre_object::generator::is_coroutine(obj) } {
        return Err(crate::PyError::type_error("descriptor is for 'coroutine'"));
    }
    Ok(unsafe { pyre_object::generator::w_coroutine_get_origin(obj) })
}
fn coroutine_get_name(args: &[PyObjectRef]) -> crate::PyResult {
    coroutine_getter(args, 5)
}
fn coroutine_get_qualname(args: &[PyObjectRef]) -> crate::PyResult {
    coroutine_getter(args, 6)
}

fn generator_set_name_common(
    args: &[PyObjectRef],
    qualname: bool,
    coroutine: bool,
) -> crate::PyResult {
    let obj = args[1];
    let value = args[2];
    let matches = unsafe {
        if coroutine {
            pyre_object::generator::is_coroutine(obj)
        } else {
            pyre_object::generator::is_generator(obj)
        }
    };
    if !matches {
        return Err(crate::PyError::type_error(format!(
            "descriptor is for '{}'",
            if coroutine { "coroutine" } else { "generator" }
        )));
    }
    if !unsafe { pyre_object::is_str(value) } {
        return Err(crate::PyError::type_error(format!(
            "__{}__ must be set to a string object",
            if qualname { "qualname" } else { "name" }
        )));
    }
    unsafe {
        if qualname {
            pyre_object::generator::w_generator_set_qualname(obj, value);
        } else {
            pyre_object::generator::w_generator_set_name(obj, value);
        }
    }
    Ok(w_none())
}

fn generator_set_name(args: &[PyObjectRef]) -> crate::PyResult {
    generator_set_name_common(args, false, false)
}
fn generator_set_qualname(args: &[PyObjectRef]) -> crate::PyResult {
    generator_set_name_common(args, true, false)
}
fn coroutine_set_name(args: &[PyObjectRef]) -> crate::PyResult {
    generator_set_name_common(args, false, true)
}
fn coroutine_set_qualname(args: &[PyObjectRef]) -> crate::PyResult {
    generator_set_name_common(args, true, true)
}

fn generator_descr_sizeof(_args: &[PyObjectRef]) -> crate::PyResult {
    Ok(w_int_new(
        (pyre_object::generator::W_GENERATOR_OBJECT_SIZE
            + std::mem::size_of::<crate::pyframe::PyFrame>()) as i64,
    ))
}

/// PyPy `generator.py GeneratorIterator.typedef`, augmented only by the
/// concrete slots Python 3.14 exposes on `types.GeneratorType`.
fn init_generator_type(ns: PyObjectRef) {
    unsafe { pyre_object::w_dict_setitem_str(ns, "__doc__", w_none()) };
    for (name, function, arity) in [
        ("__repr__", generator_descr_repr as DunderFn, 1),
        ("__next__", crate::baseobjspace::generator_next_method, 1),
        ("send", crate::baseobjspace::generator_send_method, 2),
        ("close", crate::baseobjspace::generator_close_method, 1),
        ("__iter__", crate::baseobjspace::iter_self_method, 1),
        ("__del__", crate::baseobjspace::generator_close_method, 1),
        ("__sizeof__", generator_descr_sizeof, 1),
    ] {
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, function, arity),
            )
        };
    }
    unsafe {
        pyre_object::w_dict_setitem_str(
            ns,
            "throw",
            make_builtin_function("throw", crate::baseobjspace::generator_throw_method),
        );
        pyre_object::w_dict_setitem_str(
            ns,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                crate::_pypy_generic_alias::generic_alias_class_getitem,
            )),
        );
    }
    for (name, getter) in [
        ("gi_running", generator_get_running as DunderFn),
        ("gi_suspended", generator_get_suspended as DunderFn),
        ("gi_frame", generator_get_frame as DunderFn),
        ("gi_code", generator_get_code as DunderFn),
        ("gi_yieldfrom", generator_get_yieldfrom as DunderFn),
    ] {
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_getset_descriptor_named(
                    make_builtin_function_with_arity(name, getter, 2),
                    name,
                ),
            )
        };
    }
    for (name, getter, setter) in [
        (
            "__name__",
            generator_get_name as DunderFn,
            generator_set_name as DunderFn,
        ),
        (
            "__qualname__",
            generator_get_qualname as DunderFn,
            generator_set_qualname as DunderFn,
        ),
    ] {
        let get = make_builtin_function_with_arity(name, getter, 2);
        let set = make_builtin_function_with_arity(name, setter, 3);
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_getset_property_full(get, set, PY_NULL, PY_NULL, PY_NULL, Some(name)),
            )
        };
    }
}

/// PyPy `generator.py Coroutine.typedef`.
fn init_coroutine_type(ns: PyObjectRef) {
    unsafe { pyre_object::w_dict_setitem_str(ns, "__doc__", w_none()) };
    for (name, function, arity) in [
        ("__repr__", coroutine_descr_repr as DunderFn, 1),
        ("send", crate::baseobjspace::generator_send_method, 2),
        ("close", crate::baseobjspace::generator_close_method, 1),
        ("__await__", crate::baseobjspace::coroutine_await_method, 1),
        ("__del__", crate::baseobjspace::generator_close_method, 1),
        ("__sizeof__", generator_descr_sizeof, 1),
    ] {
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, function, arity),
            )
        };
    }
    unsafe {
        pyre_object::w_dict_setitem_str(
            ns,
            "throw",
            make_builtin_function("throw", crate::baseobjspace::generator_throw_method),
        );
    }
    for (name, getter) in [
        ("cr_running", coroutine_get_running as DunderFn),
        ("cr_suspended", coroutine_get_suspended as DunderFn),
        ("cr_frame", coroutine_get_frame as DunderFn),
        ("cr_code", coroutine_get_code as DunderFn),
        ("cr_await", coroutine_get_await as DunderFn),
        ("cr_origin", coroutine_get_origin as DunderFn),
    ] {
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_getset_descriptor_named(
                    make_builtin_function_with_arity(name, getter, 2),
                    name,
                ),
            )
        };
    }
    for (name, getter, setter) in [
        (
            "__name__",
            coroutine_get_name as DunderFn,
            coroutine_set_name as DunderFn,
        ),
        (
            "__qualname__",
            coroutine_get_qualname as DunderFn,
            coroutine_set_qualname as DunderFn,
        ),
    ] {
        let get = make_builtin_function_with_arity(name, getter, 2);
        let set = make_builtin_function_with_arity(name, setter, 3);
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_getset_property_full(get, set, PY_NULL, PY_NULL, PY_NULL, Some(name)),
            )
        };
    }
}

/// PyPy `generator.py CoroutineWrapper.typedef`.
fn init_coroutine_wrapper_type(ns: PyObjectRef) {
    unsafe { pyre_object::w_dict_setitem_str(ns, "__doc__", w_none()) };
    for (name, function, arity) in [
        (
            "__iter__",
            crate::baseobjspace::iter_self_method as DunderFn,
            1,
        ),
        (
            "__next__",
            crate::baseobjspace::coroutine_wrapper_next_method,
            1,
        ),
        (
            "send",
            crate::baseobjspace::coroutine_wrapper_send_method,
            2,
        ),
        (
            "close",
            crate::baseobjspace::coroutine_wrapper_close_method,
            1,
        ),
    ] {
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, function, arity),
            )
        };
    }
    unsafe {
        pyre_object::w_dict_setitem_str(
            ns,
            "throw",
            make_builtin_function("throw", crate::baseobjspace::coroutine_wrapper_throw_method),
        );
    }
}

/// PyPy `iterobject.py W_AbstractSeqIterObject.typedef`.
fn init_sequence_iterator_type(ns: PyObjectRef) {
    // PyPy carries the `iter()` builtin documentation on the abstract typedef;
    // Python 3.14's concrete `iterator` type exposes `__doc__ is None`.
    unsafe { pyre_object::w_dict_setitem_str(ns, "__doc__", pyre_object::w_none()) };
    let entries = [
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::iter_self_method, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", crate::baseobjspace::iter_next_method, 1),
        ),
        (
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                crate::baseobjspace::seq_iter_reduce_method,
                1,
            ),
        ),
        (
            "__length_hint__",
            make_builtin_function_with_arity(
                "__length_hint__",
                crate::baseobjspace::seq_iter_length_hint_method,
                1,
            ),
        ),
        (
            "__setstate__",
            make_builtin_function_with_arity(
                "__setstate__",
                crate::baseobjspace::seq_iter_setstate_method,
                2,
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

/// Python 3.14 `PyCallIter_Type` (`callable_iterator`) surface. PyPy 3.11's
/// `_CallableIterator` is app-level and has only the iteration methods; 3.14
/// additionally exposes the native pickle reduction hook.
fn init_callable_iterator_type(ns: PyObjectRef) {
    for (name, function) in [
        (
            "__iter__",
            crate::baseobjspace::iter_self_method as fn(&[PyObjectRef]) -> crate::PyResult,
        ),
        ("__next__", crate::baseobjspace::iter_next_method),
        (
            "__reduce__",
            crate::baseobjspace::callable_iter_reduce_method,
        ),
    ] {
        unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, function, 1),
            )
        };
    }
}

fn dict_iterator_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
    owner: &str,
    expected: &'static PyType,
    allow_subclass: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method {owner}.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of '{owner}' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    // `dict` method descriptors bind to any dict subclass (an `OrderedDict`
    // passed to `dict.__reversed__`); the concrete iterator typedefs are not
    // subclassable, so they keep the exact-type check on the hot path.
    let matches = if allow_subclass {
        unsafe { crate::baseobjspace::isinstance_w(receiver, gettypeobject(expected)) }
    } else {
        unsafe { pyre_object::py_type_check(receiver, expected) }
    };
    if !matches {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!(
                "descriptor '{name}' for '{owner}' objects doesn't apply to a '{received}' object"
            )
        } else {
            format!("descriptor '{name}' requires a '{owner}' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// PyPy's six concrete `W_BaseDictMultiIterObject` typedefs share their
/// implementations, while each interp2app gateway still enforces its own
/// concrete receiver class.
macro_rules! define_dict_iterator_type {
    ($init:ident, $self_fn:ident, $next_fn:ident, $len_fn:ident, $reduce_fn:ident, $owner:literal, $expected:path) => {
        fn $self_fn(args: &[PyObjectRef]) -> crate::PyResult {
            dict_iterator_receiver(args, "__iter__", false, $owner, &$expected, false)
        }

        fn $next_fn(args: &[PyObjectRef]) -> crate::PyResult {
            dict_iterator_receiver(args, "__next__", false, $owner, &$expected, false)?;
            crate::baseobjspace::next(args[0])
        }

        fn $len_fn(args: &[PyObjectRef]) -> crate::PyResult {
            dict_iterator_receiver(args, "__length_hint__", true, $owner, &$expected, false)?;
            crate::baseobjspace::dict_view_iter_length_hint_method(args)
        }

        fn $reduce_fn(args: &[PyObjectRef]) -> crate::PyResult {
            dict_iterator_receiver(args, "__reduce__", true, $owner, &$expected, false)?;
            crate::baseobjspace::dict_view_iter_reduce_method(args)
        }

        fn $init(ns: PyObjectRef) {
            unsafe { pyre_object::w_dict_setitem_str(ns, "__doc__", pyre_object::w_none()) };
            let entries = [
                (
                    "__iter__",
                    make_builtin_function_with_arity("__iter__", $self_fn, 1),
                ),
                (
                    "__next__",
                    make_builtin_function_with_arity("__next__", $next_fn, 1),
                ),
                (
                    "__length_hint__",
                    make_builtin_function_with_arity("__length_hint__", $len_fn, 1),
                ),
                (
                    "__reduce__",
                    make_builtin_function_with_arity("__reduce__", $reduce_fn, 1),
                ),
            ];
            for (name, value) in entries {
                unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
            }
        }
    };
}

define_dict_iterator_type!(
    init_dict_key_iterator_type,
    dict_key_iter_self,
    dict_key_iter_next,
    dict_key_iter_len,
    dict_key_iter_reduce,
    "dict_keyiterator",
    pyre_object::dictmultiobject::DICT_KEYITERATOR_TYPE
);
define_dict_iterator_type!(
    init_dict_value_iterator_type,
    dict_value_iter_self,
    dict_value_iter_next,
    dict_value_iter_len,
    dict_value_iter_reduce,
    "dict_valueiterator",
    pyre_object::dictmultiobject::DICT_VALUEITERATOR_TYPE
);
define_dict_iterator_type!(
    init_dict_item_iterator_type,
    dict_item_iter_self,
    dict_item_iter_next,
    dict_item_iter_len,
    dict_item_iter_reduce,
    "dict_itemiterator",
    pyre_object::dictmultiobject::DICT_ITEMITERATOR_TYPE
);
define_dict_iterator_type!(
    init_dict_reverse_key_iterator_type,
    dict_reverse_key_iter_self,
    dict_reverse_key_iter_next,
    dict_reverse_key_iter_len,
    dict_reverse_key_iter_reduce,
    "dict_reversekeyiterator",
    pyre_object::dictmultiobject::DICT_REVERSEKEYITERATOR_TYPE
);
define_dict_iterator_type!(
    init_dict_reverse_value_iterator_type,
    dict_reverse_value_iter_self,
    dict_reverse_value_iter_next,
    dict_reverse_value_iter_len,
    dict_reverse_value_iter_reduce,
    "dict_reversevalueiterator",
    pyre_object::dictmultiobject::DICT_REVERSEVALUEITERATOR_TYPE
);
define_dict_iterator_type!(
    init_dict_reverse_item_iterator_type,
    dict_reverse_item_iter_self,
    dict_reverse_item_iter_next,
    dict_reverse_item_iter_len,
    dict_reverse_item_iter_reduce,
    "dict_reverseitemiterator",
    pyre_object::dictmultiobject::DICT_REVERSEITEMITERATOR_TYPE
);

fn range_iterator_self(args: &[PyObjectRef]) -> crate::PyResult {
    crate::type_methods::require_range_iterator_receiver(args, "__iter__", false, false)
}

fn range_iterator_next(args: &[PyObjectRef]) -> crate::PyResult {
    crate::type_methods::require_range_iterator_receiver(args, "__next__", false, false)?;
    crate::baseobjspace::next(args[0])
}

fn range_iterator_length_hint(args: &[PyObjectRef]) -> crate::PyResult {
    crate::type_methods::require_range_iterator_receiver(args, "__length_hint__", true, false)?;
    crate::baseobjspace::range_iter_length_hint_method(args)
}

fn range_iterator_reduce(args: &[PyObjectRef]) -> crate::PyResult {
    crate::type_methods::require_range_iterator_receiver(args, "__reduce__", true, false)?;
    crate::baseobjspace::range_iter_reduce_method(args)
}

fn range_iterator_setstate(args: &[PyObjectRef]) -> crate::PyResult {
    crate::type_methods::require_range_iterator_receiver(args, "__setstate__", true, false)?;
    crate::baseobjspace::range_iter_setstate_method(args)
}

fn long_range_iterator_self(args: &[PyObjectRef]) -> crate::PyResult {
    crate::type_methods::require_range_iterator_receiver(args, "__iter__", false, true)
}

fn long_range_iterator_next(args: &[PyObjectRef]) -> crate::PyResult {
    crate::type_methods::require_range_iterator_receiver(args, "__next__", false, true)?;
    crate::baseobjspace::next(args[0])
}

fn long_range_iterator_length_hint(args: &[PyObjectRef]) -> crate::PyResult {
    crate::type_methods::require_range_iterator_receiver(args, "__length_hint__", true, true)?;
    crate::baseobjspace::long_range_iter_length_hint_method(args)
}

fn long_range_iterator_reduce(args: &[PyObjectRef]) -> crate::PyResult {
    crate::type_methods::require_range_iterator_receiver(args, "__reduce__", true, true)?;
    crate::baseobjspace::long_range_iter_reduce_method(args)
}

fn long_range_iterator_setstate(args: &[PyObjectRef]) -> crate::PyResult {
    crate::type_methods::require_range_iterator_receiver(args, "__setstate__", true, true)?;
    crate::baseobjspace::range_iter_setstate_method(args)
}

/// PyPy `functional.py W_AbstractRangeIterator.typedef`.
fn init_range_iterator_type(ns: PyObjectRef) {
    unsafe { pyre_object::w_dict_setitem_str(ns, "__doc__", pyre_object::w_none()) };
    let entries = [
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", range_iterator_self, 1),
        ),
        (
            "__length_hint__",
            make_builtin_function_with_arity("__length_hint__", range_iterator_length_hint, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", range_iterator_next, 1),
        ),
        (
            "__reduce__",
            make_builtin_function_with_arity("__reduce__", range_iterator_reduce, 1),
        ),
        (
            "__setstate__",
            make_builtin_function_with_arity("__setstate__", range_iterator_setstate, 2),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

/// Python 3.14 exposes the arbitrary-precision implementation as the distinct
/// `longrange_iterator` type, while retaining the same five protocol entries.
fn init_long_range_iterator_type(ns: PyObjectRef) {
    unsafe { pyre_object::w_dict_setitem_str(ns, "__doc__", pyre_object::w_none()) };
    let entries = [
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", long_range_iterator_self, 1),
        ),
        (
            "__length_hint__",
            make_builtin_function_with_arity("__length_hint__", long_range_iterator_length_hint, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", long_range_iterator_next, 1),
        ),
        (
            "__reduce__",
            make_builtin_function_with_arity("__reduce__", long_range_iterator_reduce, 1),
        ),
        (
            "__setstate__",
            make_builtin_function_with_arity("__setstate__", long_range_iterator_setstate, 2),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn init_list_iterator_type(ns: PyObjectRef) {
    unsafe { pyre_object::w_dict_setitem_str(ns, "__doc__", pyre_object::w_none()) };
    let entries = [
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::iter_self_method, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", crate::baseobjspace::iter_next_method, 1),
        ),
        (
            "__length_hint__",
            make_builtin_function_with_arity(
                "__length_hint__",
                crate::baseobjspace::list_iter_length_hint_method,
                1,
            ),
        ),
        (
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                crate::baseobjspace::list_iter_reduce_method,
                1,
            ),
        ),
        (
            "__setstate__",
            make_builtin_function_with_arity(
                "__setstate__",
                crate::baseobjspace::list_iter_setstate_method,
                2,
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn init_list_reverse_iterator_type(ns: PyObjectRef) {
    unsafe { pyre_object::w_dict_setitem_str(ns, "__doc__", pyre_object::w_none()) };
    let entries = [
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::iter_self_method, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", crate::baseobjspace::iter_next_method, 1),
        ),
        (
            "__length_hint__",
            make_builtin_function_with_arity(
                "__length_hint__",
                crate::baseobjspace::list_reverse_iter_length_hint_method,
                1,
            ),
        ),
        (
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                crate::baseobjspace::list_reverse_iter_reduce_method,
                1,
            ),
        ),
        (
            "__setstate__",
            make_builtin_function_with_arity(
                "__setstate__",
                crate::baseobjspace::list_reverse_iter_setstate_method,
                2,
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn init_tuple_iterator_type(ns: PyObjectRef) {
    unsafe { pyre_object::w_dict_setitem_str(ns, "__doc__", pyre_object::w_none()) };
    let entries = [
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::iter_self_method, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", crate::baseobjspace::iter_next_method, 1),
        ),
        (
            "__length_hint__",
            make_builtin_function_with_arity(
                "__length_hint__",
                crate::baseobjspace::tuple_iter_length_hint_method,
                1,
            ),
        ),
        (
            "__reduce__",
            make_builtin_function_with_arity(
                "__reduce__",
                crate::baseobjspace::tuple_iter_reduce_method,
                1,
            ),
        ),
        (
            "__setstate__",
            make_builtin_function_with_arity(
                "__setstate__",
                crate::baseobjspace::tuple_iter_setstate_method,
                2,
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

// ── itertools.count / itertools.repeat TypeDefs ─────────────────────
// PyPy: pypy/module/itertools/interp_itertools.py W_Count.typedef and
// W_Repeat.typedef.  Python 3.14 deliberately omits the old PyPy
// `__reduce__` entries, so these two concrete types are not picklable.

fn itertools_constructor_scope(
    args: &[PyObjectRef],
    fn_name: &str,
    names: Vec<&'static str>,
    defaults: &[PyObjectRef],
) -> Result<(PyObjectRef, Vec<PyObjectRef>), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let cls = positional.first().copied().unwrap_or(PY_NULL);
    let positional = positional.get(1..).unwrap_or(&[]);
    let mut keyword_names_w = Vec::new();
    let mut keywords_w = Vec::new();
    if let Some(dict) = kwargs {
        for (key, value) in unsafe { pyre_object::w_dict_str_entries_wtf8(dict) } {
            if key.as_str() == Ok("__pyre_kw__") {
                continue;
            }
            let w_name = pyre_object::w_str_from_wtf8(key);
            pyre_object::gc_roots::pin_root(w_name);
            keyword_names_w.push(w_name);
            keywords_w.push(value);
        }
    }
    let signature = crate::gateway::Signature::new(names, None, None, 0, 0);
    let arguments = crate::argument::Arguments::with_kw(positional, &keyword_names_w, &keywords_w);
    let mut scope_w = vec![PY_NULL; signature.scope_length()];
    arguments.parse_into_scope(
        PY_NULL,
        &mut scope_w,
        fn_name,
        &signature,
        Some(defaults),
        PY_NULL,
    )?;
    Ok((cls, scope_w))
}

fn itertools_alloc_for_class(
    cls: PyObjectRef,
    exact_type: PyObjectRef,
    obj: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    // typedef.py:511 `allocate_instance` first checks that the requested
    // subtype shares the builtin's layout, then installs that class on the
    // freshly allocated interpreter object.
    check_user_subclass(exact_type, cls)?;
    if !std::ptr::eq(cls, exact_type) {
        unsafe { (*obj).w_class = cls };
    }
    Ok(obj)
}

fn count_check_number(obj: PyObjectRef) -> Result<(), crate::PyError> {
    // interp_itertools.py `check_number`, with CPython 3.14's public error
    // wording (`a number is required`) in place of PyPy 3.11's older text.
    let has_int = unsafe { crate::baseobjspace::lookup(obj, "__int__") }.is_some();
    let has_float = unsafe { crate::baseobjspace::lookup(obj, "__float__") }.is_some();
    let complex_type = gettypefor(&pyre_object::COMPLEX_TYPE).unwrap_or(PY_NULL);
    if !has_int
        && !has_float
        && (complex_type.is_null()
            || !unsafe { crate::baseobjspace::isinstance_w(obj, complex_type) })
    {
        return Err(crate::PyError::type_error("a number is required"));
    }
    Ok(())
}

fn count_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // W_Count___new__(space, w_subtype, w_start=0, w_step=1).
    let _roots = pyre_object::gc_roots::push_roots();
    let w_zero = w_int_new(0);
    pyre_object::gc_roots::pin_root(w_zero);
    let w_one = w_int_new(1);
    pyre_object::gc_roots::pin_root(w_one);
    let (cls, scope) =
        itertools_constructor_scope(args, "count", vec!["start", "step"], &[w_zero, w_one])?;
    let w_start = scope[0];
    let w_step = scope[1];
    count_check_number(w_start)?;
    count_check_number(w_step)?;
    let exact = gettypefor(&pyre_object::interp_itertools::COUNT_TYPE).unwrap_or(PY_NULL);
    let obj = pyre_object::interp_itertools::w_count_new(w_start, w_step);
    itertools_alloc_for_class(cls, exact, obj)
}

fn count_single_argument(w_step: PyObjectRef) -> Result<bool, crate::PyError> {
    // W_Count.single_argument: isinstance(step, int) and step == 1.
    let int_type = gettypefor(&pyre_object::INT_TYPE).unwrap_or(PY_NULL);
    Ok(!int_type.is_null()
        && unsafe { crate::baseobjspace::isinstance_w(w_step, int_type) }
        && crate::baseobjspace::eq_w(w_step, w_int_new(1))?)
}

fn count_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args.first().copied().unwrap_or(PY_NULL);
    let cls = r#type(obj).unwrap_or(PY_NULL);
    let full_name = unsafe { pyre_object::w_type_get_name(cls) };
    let cls_name = full_name.rsplit('.').next().unwrap_or(full_name);
    let w_c = unsafe { pyre_object::interp_itertools::w_count_get_c(obj) };
    let w_step = unsafe { pyre_object::interp_itertools::w_count_get_step(obj) };
    let c = unsafe { crate::display::py_repr(w_c)? };
    let text = if count_single_argument(w_step)? {
        format!("{cls_name}({c})")
    } else {
        let step = unsafe { crate::display::py_repr(w_step)? };
        format!("{cls_name}({c}, {step})")
    };
    Ok(w_str_new(&text))
}

fn init_count_type(ns: PyObjectRef) {
    // Source order follows W_Count.typedef.  `__reduce__` is the one explicit
    // Python-3.14 semantic delta (the concrete type no longer exposes it).
    let entries = [
        ("__new__", make_new_descr(count_descr_new)),
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::iter_self_method, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", crate::baseobjspace::iter_next_method, 1),
        ),
        (
            "__repr__",
            make_builtin_function_with_arity("__repr__", count_descr_repr, 1),
        ),
        (
            "__doc__",
            w_str_new(
                "Return a count object whose .__next__() method returns consecutive values.\n\nEquivalent to:\n    def count(firstval=0, step=1):\n        x = firstval\n        while 1:\n            yield x\n            x += step",
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn repeat_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // W_Repeat___new__(space, w_subtype, w_object, w_times=None).  A null
    // default represents a missing `times`; an explicit Python None still
    // passes through `space.index_w` and raises TypeError, as in 3.14.
    let (cls, scope) =
        itertools_constructor_scope(args, "repeat", vec!["object", "times"], &[PY_NULL])?;
    let w_obj = scope[0];
    let w_times = scope[1];
    let times = if w_times.is_null() {
        None
    } else {
        Some(crate::builtins::space_index_w(w_times)?)
    };
    let exact = gettypefor(&pyre_object::interp_itertools::REPEAT_TYPE).unwrap_or(PY_NULL);
    let obj = pyre_object::interp_itertools::w_repeat_new(w_obj, times);
    itertools_alloc_for_class(cls, exact, obj)
}

fn repeat_descr_length_hint(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args.first().copied().unwrap_or(PY_NULL);
    if unsafe { pyre_object::interp_itertools::w_repeat_get_counting(obj) } {
        Ok(w_int_new(unsafe {
            pyre_object::interp_itertools::w_repeat_get_count(obj)
        }))
    } else {
        // PyPy 3.11 returned NotImplemented; Python 3.14 raises directly.
        Err(crate::PyError::type_error("len() of unsized object"))
    }
}

fn repeat_descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args.first().copied().unwrap_or(PY_NULL);
    let cls = r#type(obj).unwrap_or(PY_NULL);
    let full_name = unsafe { pyre_object::w_type_get_name(cls) };
    let cls_name = full_name.rsplit('.').next().unwrap_or(full_name);
    let w_obj = unsafe { pyre_object::interp_itertools::w_repeat_get_obj(obj) };
    let objrepr = unsafe { crate::display::py_repr(w_obj)? };
    let text = if unsafe { pyre_object::interp_itertools::w_repeat_get_counting(obj) } {
        let count = unsafe { pyre_object::interp_itertools::w_repeat_get_count(obj) };
        format!("{cls_name}({objrepr}, {count})")
    } else {
        format!("{cls_name}({objrepr})")
    };
    Ok(w_str_new(&text))
}

fn init_repeat_type(ns: PyObjectRef) {
    let entries = [
        ("__new__", make_new_descr(repeat_descr_new)),
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::iter_self_method, 1),
        ),
        (
            "__length_hint__",
            make_builtin_function_with_arity("__length_hint__", repeat_descr_length_hint, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", crate::baseobjspace::iter_next_method, 1),
        ),
        (
            "__repr__",
            make_builtin_function_with_arity("__repr__", repeat_descr_repr, 1),
        ),
        (
            "__doc__",
            w_str_new(
                "repeat(object [,times]) -> create an iterator which returns the object\nfor the specified number of times.  If not specified, returns the object\nendlessly.",
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

// ── itertools predicate iterator TypeDefs ────────────────────────────
// PyPy W_TakeWhile / W_DropWhile / W_FilterFalse.  Python 3.14 keeps the
// constructor/iterator shape but no longer exposes PyPy 3.11's pickle state
// methods on these concrete types.

fn itertools_twoarg_new(
    args: &[PyObjectRef],
    exact_type: PyObjectRef,
    name: &str,
) -> Result<(PyObjectRef, PyObjectRef, PyObjectRef), crate::PyError> {
    // interp_itertools.py W_Twoarg__new__, kept in source order.
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let cls = positional.first().copied().unwrap_or(PY_NULL);
    let args_w = positional.get(1..).unwrap_or(&[]);
    let init_matches = std::ptr::eq(cls, exact_type)
        || unsafe {
            match (
                crate::baseobjspace::lookup_in_type(cls, "__init__"),
                crate::baseobjspace::lookup_in_type(exact_type, "__init__"),
            ) {
                (Some(sub), Some(base)) => std::ptr::eq(sub, base),
                (None, None) => true,
                _ => false,
            }
        };
    if init_matches && crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes no keyword arguments"
        )));
    }
    if args_w.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "{name} expected 2 arguments, got {}",
            args_w.len()
        )));
    }
    Ok((cls, args_w[0], args_w[1]))
}

fn takewhile_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exact = gettypefor(&pyre_object::interp_itertools::TAKEWHILE_TYPE).unwrap_or(PY_NULL);
    let (cls, predicate, iterable) = itertools_twoarg_new(args, exact, "takewhile")?;
    let iterator = crate::baseobjspace::iter(iterable)?;
    let obj = pyre_object::interp_itertools::w_takewhile_new(predicate, iterator);
    itertools_alloc_for_class(cls, exact, obj)
}

fn dropwhile_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exact = gettypefor(&pyre_object::interp_itertools::DROPWHILE_TYPE).unwrap_or(PY_NULL);
    let (cls, predicate, iterable) = itertools_twoarg_new(args, exact, "dropwhile")?;
    let iterator = crate::baseobjspace::iter(iterable)?;
    let obj = pyre_object::interp_itertools::w_dropwhile_new(predicate, iterator);
    itertools_alloc_for_class(cls, exact, obj)
}

fn filterfalse_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exact = gettypefor(&pyre_object::interp_itertools::FILTERFALSE_TYPE).unwrap_or(PY_NULL);
    let (cls, predicate, iterable) = itertools_twoarg_new(args, exact, "filterfalse")?;
    let predicate = if unsafe { pyre_object::is_none(predicate) } {
        PY_NULL
    } else {
        predicate
    };
    let iterator = crate::baseobjspace::iter(iterable)?;
    let obj = pyre_object::interp_itertools::w_filterfalse_new(predicate, iterator);
    itertools_alloc_for_class(cls, exact, obj)
}

fn compress_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // PyPy W_Compress__new__ allocates the subtype and applies space.iter to
    // both inputs.  Python 3.14's Argument Clinic surface additionally
    // accepts the `data` and `selectors` keyword names.
    let exact = gettypefor(&pyre_object::interp_itertools::COMPRESS_TYPE).unwrap_or(PY_NULL);
    let (cls, scope_w) =
        itertools_constructor_scope(args, "compress", vec!["data", "selectors"], &[])?;
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(scope_w[0]);
    let data_arg_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(scope_w[1]);
    let selectors_arg_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_data = crate::baseobjspace::iter(unsafe {
        pyre_object::gc_roots::shadow_stack_get(data_arg_slot)
    })?;
    pyre_object::gc_roots::pin_root(w_data);
    let data_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_selectors = crate::baseobjspace::iter(unsafe {
        pyre_object::gc_roots::shadow_stack_get(selectors_arg_slot)
    })?;
    let obj = pyre_object::interp_itertools::w_compress_new(
        unsafe { pyre_object::gc_roots::shadow_stack_get(data_slot) },
        w_selectors,
    );
    itertools_alloc_for_class(
        unsafe { pyre_object::gc_roots::shadow_stack_get(cls_slot) },
        exact,
        obj,
    )
}

fn compress_iter_self(args: &[PyObjectRef]) -> crate::PyResult {
    singleton_receiver(
        args,
        "itertools.compress",
        "__iter__",
        pyre_object::interp_itertools::is_compress,
    )
}

fn compress_iter_next(args: &[PyObjectRef]) -> crate::PyResult {
    let obj = singleton_receiver(
        args,
        "itertools.compress",
        "__next__",
        pyre_object::interp_itertools::is_compress,
    )?;
    crate::baseobjspace::next(obj)
}

fn starmap_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // interp_itertools.py W_StarMap___new__, kept in source order.
    let exact = gettypefor(&pyre_object::interp_itertools::STARMAP_TYPE).unwrap_or(PY_NULL);
    let (cls, w_fun, w_iterable) = itertools_twoarg_new(args, exact, "starmap")?;
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(w_fun);
    let fun_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(w_iterable);
    let iterable_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let iterator = crate::baseobjspace::iter(unsafe {
        pyre_object::gc_roots::shadow_stack_get(iterable_slot)
    })?;
    let obj = pyre_object::interp_itertools::w_starmap_new(
        unsafe { pyre_object::gc_roots::shadow_stack_get(fun_slot) },
        iterator,
    );
    itertools_alloc_for_class(
        unsafe { pyre_object::gc_roots::shadow_stack_get(cls_slot) },
        exact,
        obj,
    )
}

fn starmap_iter_self(args: &[PyObjectRef]) -> crate::PyResult {
    singleton_receiver(
        args,
        "itertools.starmap",
        "__iter__",
        pyre_object::interp_itertools::is_starmap,
    )
}

fn starmap_iter_next(args: &[PyObjectRef]) -> crate::PyResult {
    let obj = singleton_receiver(
        args,
        "itertools.starmap",
        "__next__",
        pyre_object::interp_itertools::is_starmap,
    )?;
    crate::baseobjspace::next(obj)
}

fn accumulate_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // W_Accumulate__new__(space, w_subtype, w_iterable, w_func=None,
    //                     __kwonly__=None, w_initial=None).
    let exact = gettypefor(&pyre_object::interp_itertools::ACCUMULATE_TYPE).unwrap_or(PY_NULL);
    let w_none = pyre_object::w_none();
    let (cls, scope_w) = itertools_constructor_scope(
        args,
        "accumulate",
        vec!["iterable", "func", "initial"],
        &[w_none, w_none],
    )?;
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(scope_w[0]);
    let iterable_arg_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(scope_w[1]);
    let func_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(scope_w[2]);
    let initial_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_iterable = crate::baseobjspace::iter(unsafe {
        pyre_object::gc_roots::shadow_stack_get(iterable_arg_slot)
    })?;
    let w_func = unsafe { pyre_object::gc_roots::shadow_stack_get(func_slot) };
    let w_func = if unsafe { pyre_object::is_none(w_func) } {
        PY_NULL
    } else {
        w_func
    };
    let obj = pyre_object::interp_itertools::w_accumulate_new(w_iterable, w_func, unsafe {
        pyre_object::gc_roots::shadow_stack_get(initial_slot)
    });
    itertools_alloc_for_class(
        unsafe { pyre_object::gc_roots::shadow_stack_get(cls_slot) },
        exact,
        obj,
    )
}

fn zip_longest_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // W_ZipLongest___new__: keep all positional sources as live iterators and
    // accept only the fillvalue keyword.
    let exact = gettypefor(&pyre_object::interp_itertools::ZIP_LONGEST_TYPE).unwrap_or(PY_NULL);
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let cls = positional.first().copied().unwrap_or(PY_NULL);
    let sources = positional.get(1..).unwrap_or(&[]);
    crate::builtins::kwarg_reject_unknown(kwargs, &["fillvalue"], "zip_longest")?;
    let w_fillvalue =
        crate::builtins::kwarg_get(kwargs, "fillvalue").unwrap_or_else(pyre_object::w_none);

    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(w_fillvalue);
    let fill_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let sources_base = pyre_object::gc_roots::shadow_stack_len();
    for &source in sources {
        pyre_object::gc_roots::pin_root(source);
    }
    let iterators_base = pyre_object::gc_roots::shadow_stack_len();
    for index in 0..sources.len() {
        let w_iterable = crate::baseobjspace::iter(unsafe {
            pyre_object::gc_roots::shadow_stack_get(sources_base + index)
        })?;
        pyre_object::gc_roots::pin_root(w_iterable);
    }
    let iterators = (0..sources.len())
        .map(|index| unsafe { pyre_object::gc_roots::shadow_stack_get(iterators_base + index) })
        .collect();
    let w_iterators = pyre_object::w_list_new(iterators);
    let obj = pyre_object::interp_itertools::w_zip_longest_new(
        w_iterators,
        unsafe { pyre_object::gc_roots::shadow_stack_get(fill_slot) },
        sources.len() as i64,
    );
    itertools_alloc_for_class(
        unsafe { pyre_object::gc_roots::shadow_stack_get(cls_slot) },
        exact,
        obj,
    )
}

fn init_takewhile_type(ns: PyObjectRef) {
    // W_TakeWhile.typedef, in source order (minus the 3.14-removed pickle
    // entries between __next__ and __doc__).
    let entries = [
        ("__new__", make_new_descr(takewhile_descr_new)),
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::iter_self_method, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", crate::baseobjspace::iter_next_method, 1),
        ),
        (
            "__doc__",
            w_str_new(
                "Return successive entries from an iterable as long as the predicate evaluates to true for each entry.",
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn init_dropwhile_type(ns: PyObjectRef) {
    // W_DropWhile.typedef, in source order.
    let entries = [
        ("__new__", make_new_descr(dropwhile_descr_new)),
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::iter_self_method, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", crate::baseobjspace::iter_next_method, 1),
        ),
        (
            "__doc__",
            w_str_new(
                "Drop items from the iterable while predicate(item) is true.\n\nAfterwards, return every element until the iterable is exhausted.",
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn init_filterfalse_type(ns: PyObjectRef) {
    // W_FilterFalse.typedef, in source order.
    let entries = [
        ("__new__", make_new_descr(filterfalse_descr_new)),
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::iter_self_method, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", crate::baseobjspace::iter_next_method, 1),
        ),
        (
            "__doc__",
            w_str_new(
                "Return those items of iterable for which function(item) is false.\n\nIf function is None, return the items that are false.",
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn init_compress_type(ns: PyObjectRef) {
    // interp_itertools.py W_Compress.typedef, in source order.  Python 3.14
    // uses the shorter public docstring below.
    let entries = [
        ("__new__", make_new_descr(compress_descr_new)),
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", compress_iter_self, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", compress_iter_next, 1),
        ),
        (
            "__doc__",
            w_str_new(
                "Return data elements corresponding to true selector elements.\n\nForms a shorter iterator from selected data elements using the selectors to\nchoose the data elements.",
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn init_starmap_type(ns: PyObjectRef) {
    // interp_itertools.py W_StarMap.typedef, with Python 3.14's public doc.
    let entries = [
        ("__new__", make_new_descr(starmap_descr_new)),
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", starmap_iter_self, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", starmap_iter_next, 1),
        ),
        (
            "__doc__",
            w_str_new(
                "Return an iterator whose values are returned from the function evaluated with an argument tuple taken from the given sequence.",
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn init_accumulate_type(ns: PyObjectRef) {
    // interp_itertools.py W_Accumulate.typedef.  Pickle state methods remain
    // to be ported; the iterator and constructor slots preserve the live
    // PyPy state machine instead of materializing the input.
    let entries = [
        ("__new__", make_new_descr(accumulate_descr_new)),
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::iter_self_method, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", crate::baseobjspace::iter_next_method, 1),
        ),
        (
            "__doc__",
            w_str_new("Return series of accumulated sums (or other binary function results)."),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
}

fn init_zip_longest_type(ns: PyObjectRef) {
    let entries = [
        ("__new__", make_new_descr(zip_longest_descr_new)),
        (
            "__iter__",
            make_builtin_function_with_arity("__iter__", crate::baseobjspace::iter_self_method, 1),
        ),
        (
            "__next__",
            make_builtin_function_with_arity("__next__", crate::baseobjspace::iter_next_method, 1),
        ),
        (
            "__doc__",
            w_str_new(
                "Return a zip_longest object whose next method returns a tuple from each iterable.",
            ),
        ),
    ];
    for (name, value) in entries {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, value) };
    }
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
        make_getset_property_named_doc(
            fget,
            fset,
            fdel,
            "dictionary for instance variables",
            "__dict__",
        ) as usize
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
        make_getset_property_named_doc(
            fget,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
            "list of weak references to the object",
            "__weakref__",
        ) as usize
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

/// CPython 3.14 `getset_get` / `getset_set` receiver mismatch wording.
/// PyPy routes the same `DescrMismatch` through `descr_call_mismatch`; only
/// the version-selected public TypeError text differs here.
fn getset_descr_mismatch(
    descr: pyre_object::PyObjectRef,
    obj: pyre_object::PyObjectRef,
    reqcls: pyre_object::PyObjectRef,
) -> crate::PyError {
    let name_obj = read_descr_name(descr);
    let name = if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
        unsafe { pyre_object::w_str_get_value(name_obj) }
    } else {
        "<generic property>"
    };
    let owner = if reqcls.is_null() {
        "?"
    } else {
        unsafe { pyre_object::w_type_get_name(reqcls) }
    };
    crate::PyError::type_error(format!(
        "descriptor '{name}' for '{owner}' objects doesn't apply to a '{}' object",
        type_name_of(obj)
    ))
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
        let tp_name = unsafe { pyre_object::type_name_of(w_obj) };
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
    /// Concurrent `init_typeobjects` callers must not observe the
    /// post-registration patch passes mid-write: the libtest harness
    /// calls it from many test threads, and an unsynchronized second
    /// caller used to probe `object`'s type dict while the first was
    /// still inserting `__class__` into it (IndexMap read/write race).
    /// Only the first init in the process has the race window, so this
    /// mainly guards the fix's `Once` barrier when scheduled early.
    #[test]
    fn init_typeobjects_races_no_corruption() {
        let threads: Vec<_> = (0..8)
            .map(|_| std::thread::spawn(crate::typedef::init_typeobjects))
            .collect();
        for t in threads {
            t.join()
                .expect("concurrent init_typeobjects must not panic");
        }
        // Also init on this thread: installs the thread-local hash hook the
        // dict probe below needs, and exercises the already-initialized path.
        crate::typedef::init_typeobjects();
        let object_type = crate::typedef::w_object();
        assert!(crate::type_dict_contains(object_type, "__class__"));
    }

    #[test]
    fn test_ellipsis_has_registered_typeobject() {
        crate::typedef::init_typeobjects();
        let w_type = crate::typedef::r#type(pyre_object::special::w_ellipsis())
            .expect("Ellipsis should resolve to a W_TypeObject");
        unsafe {
            assert_eq!(pyre_object::w_type_get_name(w_type), "ellipsis");
            assert!(!pyre_object::w_type_get_acceptable_as_base_class(w_type));
        }
        check_cell_typedef_python314_surface();
        check_cell_comparison_repr_and_hash();
    }

    fn check_cell_typedef_python314_surface() {
        crate::typedef::init_typeobjects();
        let w_type = crate::typedef::gettypefor(&pyre_object::nestedscope::CELL_TYPE)
            .expect("cell should resolve to a W_TypeObject");
        let expected = [
            "__doc__",
            "__eq__",
            "__ge__",
            "__gt__",
            "__hash__",
            "__le__",
            "__lt__",
            "__ne__",
            "__new__",
            "__repr__",
            "cell_contents",
        ];
        for name in expected {
            assert!(crate::type_dict_lookup(w_type, name).is_some(), "{name}");
        }
        assert!(crate::type_dict_lookup(w_type, "__reduce__").is_none());
        assert!(crate::type_dict_lookup(w_type, "__setstate__").is_none());
        unsafe {
            assert!(!pyre_object::w_type_get_acceptable_as_base_class(w_type));
        }
        let contents = crate::type_dict_lookup(w_type, "cell_contents").unwrap();
        assert!(std::ptr::eq(
            unsafe { pyre_object::typedef::w_getset_get_reqcls(contents) },
            w_type
        ));
    }

    fn check_cell_comparison_repr_and_hash() {
        crate::typedef::init_typeobjects();
        let empty = pyre_object::w_cell_new(pyre_object::PY_NULL);
        let one = pyre_object::w_cell_new(pyre_object::w_int_new(1));
        let one_again = pyre_object::w_cell_new(pyre_object::w_int_new(1));
        let two = pyre_object::w_cell_new(pyre_object::w_int_new(2));

        assert!(
            crate::baseobjspace::is_true(super::cell_descr_lt(&[empty, one]).unwrap()).unwrap()
        );
        assert!(
            crate::baseobjspace::is_true(super::cell_descr_eq(&[one, one_again]).unwrap()).unwrap()
        );
        assert!(crate::baseobjspace::is_true(super::cell_descr_gt(&[two, one]).unwrap()).unwrap());
        let foreign = super::cell_descr_eq(&[one, pyre_object::w_int_new(1)]).unwrap();
        assert!(unsafe { pyre_object::is_not_implemented(foreign) });

        let repr = super::cell_descr_repr(&[one]).unwrap();
        let repr = unsafe { pyre_object::w_str_get_value(repr) };
        assert!(repr.starts_with("<cell at 0x"));
        assert!(repr.contains(": int object at 0x"));
        let err = crate::builtins::try_hash_value(one).unwrap_err();
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert_eq!(err.message, "unhashable type: 'cell'");
    }
}
