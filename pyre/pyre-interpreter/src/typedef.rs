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

use crate::{PyNamespace, make_builtin_function, namespace_store};

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

pub fn descr_get_dict(_space: PyObjectRef, _obj: PyObjectRef) -> PyObjectRef {
    let _ = _space;
    _obj
}

pub fn descr_set_dict(_space: PyObjectRef, _obj: PyObjectRef, _w_dict: PyObjectRef) {
    let _ = (_space, _obj, _w_dict);
}

pub fn descr_del_dict(_space: PyObjectRef, _obj: PyObjectRef) {
    let _ = (_space, _obj);
}

pub fn descr_get_weakref(_space: PyObjectRef, _obj: PyObjectRef) -> PyObjectRef {
    let _ = (_space, _obj);
    PY_NULL
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
static TYPEOBJECT_CACHE: OnceLock<HashMap<usize, usize>> = OnceLock::new();

/// Get the cached W_TypeObject for a builtin runtime type.
///
/// PyPy: `space.gettypefor(cls)` / `space.gettypeobject(typedef)`
pub fn gettypefor(tp: *const PyType) -> Option<PyObjectRef> {
    TYPEOBJECT_CACHE
        .get()
        .and_then(|reg| reg.get(&(tp as usize)).copied())
        .map(|v| v as PyObjectRef)
}

/// Get the W_TypeObject for any PyObjectRef.
///
/// PyPy: `space.type(w_obj)` → `w_obj.getclass(space)`
/// - Instance of user class → w_instance_get_type
/// - Builtin object with __class__ override → ATTR_TABLE["__class__"]
/// - Builtin object → type registry lookup
/// - Type object → metaclass or type(type)
pub fn r#type(obj: PyObjectRef) -> Option<PyObjectRef> {
    if obj.is_null() {
        return None;
    }
    unsafe {
        if is_instance(obj) {
            // Check for __class__ override in ATTR_TABLE first.
            // Python allows obj.__class__ = OtherClass for compatible layouts.
            // PyPy: w_obj.getclass(space) respects __class__ descriptor.
            let class_override = crate::baseobjspace::ATTR_TABLE.with(|table| {
                table
                    .borrow()
                    .get(&(obj as usize))
                    .and_then(|d| d.get("__class__").copied())
            });
            if let Some(cls) = class_override {
                return Some(cls);
            }
            return Some(w_instance_get_type(obj));
        }
        if is_type(obj) {
            // Check for custom metaclass in ATTR_TABLE
            let mc = crate::baseobjspace::ATTR_TABLE.with(|table| {
                table
                    .borrow()
                    .get(&(obj as usize))
                    .and_then(|d| d.get("__metaclass__").copied())
            });
            if let Some(metaclass) = mc {
                return Some(metaclass);
            }
            // Default: type of type is type
            return gettypefor(&pyre_object::pyobject::TYPE_TYPE);
        }
        // Check for __class__ override in ATTR_TABLE (e.g. int subclass
        // created via int.__new__(MyIntSubclass, value) sets
        // ATTR_TABLE[obj]["__class__"] = MyIntSubclass).
        // This matches PyPy's w_obj.getclass(space) which returns
        // the real Python class, including __class__ overrides.
        let class_override = crate::baseobjspace::ATTR_TABLE.with(|table| {
            table
                .borrow()
                .get(&(obj as usize))
                .and_then(|d| d.get("__class__").copied())
        });
        if let Some(cls) = class_override {
            return Some(cls);
        }
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
    if TYPEOBJECT_CACHE.get().is_some() {
        return;
    }

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
    reg.insert(&TYPE_TYPE as *const PyType as usize, type_type as usize);
    let _ = W_TYPE_TYPEOBJECT.set(type_type as usize);

    // int — PyPy: intobject.py W_IntObject.typedef, bases=(object,)
    let int_type = new_typeobject_with_base("int", init_int_type, object_type);
    reg.insert(&INT_TYPE as *const PyType as usize, int_type as usize);

    // float — PyPy: floatobject.py, bases=(object,)
    reg.insert(
        &FLOAT_TYPE as *const PyType as usize,
        new_typeobject_with_base("float", init_float_type, object_type) as usize,
    );

    // bool — PyPy: boolobject.py, bases=(int,)
    reg.insert(
        &BOOL_TYPE as *const PyType as usize,
        new_typeobject_with_base("bool", init_bool_type, int_type) as usize,
    );

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

    // dict — PyPy: dictobject.py, bases=(object,)
    reg.insert(
        &DICT_TYPE as *const PyType as usize,
        new_typeobject_with_base("dict", init_dict_type, object_type) as usize,
    );

    // function — PyPy: funcobject.py
    // Functions are descriptors: function.__get__ returns a bound method.
    reg.insert(
        &crate::FUNCTION_TYPE as *const PyType as usize,
        new_typeobject_with_base("function", init_function_type, object_type) as usize,
    );

    // builtin-code — PyPy: BuiltinCode.typedef = TypeDef('builtin-code', ...)
    reg.insert(
        &crate::BUILTIN_CODE_TYPE as *const PyType as usize,
        new_typeobject_with_base("builtin-code", init_builtin_code_type, object_type) as usize,
    );

    reg.insert(
        &pyre_object::methodobject::METHOD_TYPE as *const PyType as usize,
        new_typeobject_with_base("method", init_method_type, object_type) as usize,
    );

    reg.insert(
        &crate::pycode::CODE_TYPE as *const PyType as usize,
        new_typeobject_with_base("code", init_code_type, object_type) as usize,
    );

    // staticmethod — PyPy: function.py StaticMethod, bases=(object,)
    reg.insert(
        &pyre_object::propertyobject::STATICMETHOD_TYPE as *const PyType as usize,
        new_typeobject_with_base("staticmethod", init_staticmethod_type, object_type) as usize,
    );

    // classmethod — PyPy: function.py ClassMethod, bases=(object,)
    reg.insert(
        &pyre_object::propertyobject::CLASSMETHOD_TYPE as *const PyType as usize,
        new_typeobject_with_base("classmethod", init_classmethod_type, object_type) as usize,
    );

    // NoneType — bases=(object,)
    reg.insert(
        &NONE_TYPE as *const PyType as usize,
        new_typeobject_with_base("NoneType", |_| {}, object_type) as usize,
    );

    // types.UnionType — PyPy: _pypy_generic_alias.py UnionType, bases=(object,)
    reg.insert(
        &pyre_object::UNION_TYPE as *const PyType as usize,
        new_typeobject_with_base("types.UnionType", init_union_type, object_type) as usize,
    );

    // slice — PyPy: sliceobject.py, bases=(object,)
    reg.insert(
        &pyre_object::sliceobject::SLICE_TYPE as *const PyType as usize,
        new_typeobject_with_base("slice", |_| {}, object_type) as usize,
    );

    // bytearray — PyPy: bytearrayobject.py, bases=(object,)
    reg.insert(
        &pyre_object::bytearrayobject::BYTEARRAY_TYPE as *const PyType as usize,
        new_typeobject_with_base("bytearray", init_bytearray_type, object_type) as usize,
    );

    let _ = TYPEOBJECT_CACHE.set(reg);
}

/// The global `object` type object, accessible from builtins.
static W_OBJECT_TYPEOBJECT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
/// The global `type` type object.
static W_TYPE_TYPEOBJECT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Get the wrapped `type` typeobject.
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
pub fn w_object() -> PyObjectRef {
    W_OBJECT_TYPEOBJECT
        .get()
        .map(|v| *v as PyObjectRef)
        .unwrap_or(PY_NULL)
}

/// Create the root `object` type. MRO = [object].
fn new_root_typeobject(name: &str, init: fn(&mut PyNamespace)) -> PyObjectRef {
    let mut ns = Box::new(PyNamespace::new());
    ns.fix_ptr();
    init(&mut ns);
    let ns_ptr = Box::into_raw(ns);
    let type_obj = w_type_new(name, PY_NULL, ns_ptr as *mut u8);
    unsafe { w_type_set_mro(type_obj, vec![type_obj]) };
    type_obj
}

/// Create a builtin type with a single base. MRO = [self] + base.mro().
/// PyPy: typeobject.py — compute_default_mro for single-inheritance
fn new_typeobject_with_base(
    name: &str,
    init: fn(&mut PyNamespace),
    base: PyObjectRef,
) -> PyObjectRef {
    let mut ns = Box::new(PyNamespace::new());
    ns.fix_ptr();
    init(&mut ns);
    let ns_ptr = Box::into_raw(ns);
    let bases = w_tuple_new(vec![base]);
    let type_obj = w_type_new(name, bases, ns_ptr as *mut u8);
    // MRO = [self] + base_mro
    let base_mro = unsafe { w_type_get_mro(base) };
    let mut mro = vec![type_obj];
    if !base_mro.is_null() {
        mro.extend_from_slice(unsafe { &*base_mro });
    } else {
        mro.push(base);
    }
    unsafe { w_type_set_mro(type_obj, mro) };
    type_obj
}

/// Create a named builtin type inheriting from `object`.
///
/// Used by extension modules (e.g. _sre) to define their own types.
pub fn make_builtin_type(name: &str, init: fn(&mut PyNamespace)) -> PyObjectRef {
    new_typeobject_with_base(name, init, w_object())
}

/// Generate a `__new__` wrapper that skips `cls` (first arg) and delegates
/// to the builtin constructor. PyPy: each type's descr__new__ strips cls
/// and calls the type-specific allocator.
macro_rules! descr_new_wrapper {
    ($fn_name:ident, $ctor:path) => {
        fn $fn_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            // args[0] = cls (ignored for builtin types)
            $ctor(&args[1..])
        }
    };
}

/// int.__new__(cls, *args) — PyPy: intobject.py descr__new__
///
/// If cls is the builtin int type, returns a plain W_IntObject.
/// If cls is a subclass of int, returns a W_InstanceObject with the
/// int value stored internally (for int subclasses like IntFlag).
fn int_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = if args.is_empty() {
        std::ptr::null_mut() as PyObjectRef
    } else {
        args[0]
    };
    let value = crate::builtins::builtin_int(&args[1..])?;
    // If cls is int itself (or null), return a plain int.
    // Compare against both the static &INT_TYPE and the W_TypeObject for int.
    if cls.is_null() || !unsafe { pyre_object::is_type(cls) } {
        return Ok(value);
    }
    let int_typeobj = gettypefor(&pyre_object::INT_TYPE);
    if int_typeobj.map_or(false, |t| std::ptr::eq(cls, t)) {
        return Ok(value);
    }
    // cls is a subclass of int — create a W_IntObject with cls's type.
    // Register the type as an int subclass so is_int() recognizes it.
    // cls is a subclass of int. Create a unique W_IntObject (bypassing
    // the small-int cache so each instance has its own identity for
    // ATTR_TABLE). Tag with __class__ = cls so type()/isinstance()
    // see the subclass while preserving W_IntObject layout for arithmetic.
    let int_val = unsafe { pyre_object::w_int_get_value(value) };
    let obj = pyre_object::w_int_new_unique(int_val);
    let _ = crate::baseobjspace::setattr(obj, "__class__", cls);
    Ok(obj)
}

descr_new_wrapper!(float_descr_new, crate::builtins::builtin_float);
descr_new_wrapper!(str_descr_new, crate::builtins::builtin_str);

/// dict.__new__(cls, *args) — if cls is a dict subclass, create an instance
/// with a backing dict for storage. PyPy: dictobject.py descr__new__
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

    // cls is a dict subclass — create instance with backing dict
    // PyPy: allocate W_DictObject with custom type
    let instance = pyre_object::w_instance_new(cls);
    let backing = pyre_object::w_dict_new();
    let _ = crate::baseobjspace::setattr(instance, "__dict_data__", backing);

    // Initialize from args if provided
    if args.len() > 1 {
        // dict(iterable) or dict(**kwargs)
        let src = args[1];
        unsafe {
            if pyre_object::is_dict(src) {
                let d = &*(src as *const pyre_object::dictobject::W_DictObject);
                for &(k, v) in &*d.entries {
                    pyre_object::w_dict_store(backing, k, v);
                }
            }
        }
    }
    Ok(instance)
}
descr_new_wrapper!(bool_descr_new, crate::builtins::builtin_bool);
descr_new_wrapper!(list_descr_new, crate::builtins::builtin_list_ctor);
descr_new_wrapper!(tuple_descr_new, crate::builtins::builtin_tuple);
// dict_new handled by dict_descr_new above (supports dict subclasses)

// ── List TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/listobject.py TypeDef("list", ...)

fn init_list_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", list_descr_new),
    );
    namespace_store(
        ns,
        "append",
        make_builtin_function("append", crate::type_methods::list_method_append),
    );
    namespace_store(
        ns,
        "extend",
        make_builtin_function("extend", crate::type_methods::list_method_extend),
    );
    namespace_store(
        ns,
        "copy",
        make_builtin_function("copy", crate::type_methods::list_method_copy),
    );
    namespace_store(
        ns,
        "insert",
        make_builtin_function("insert", crate::type_methods::list_method_insert),
    );
    namespace_store(
        ns,
        "pop",
        make_builtin_function("pop", crate::type_methods::list_method_pop),
    );
    namespace_store(
        ns,
        "clear",
        make_builtin_function("clear", crate::type_methods::list_method_clear),
    );
    namespace_store(
        ns,
        "reverse",
        make_builtin_function("reverse", crate::type_methods::list_method_reverse),
    );
    namespace_store(
        ns,
        "sort",
        make_builtin_function("sort", crate::type_methods::list_method_sort),
    );
    namespace_store(
        ns,
        "index",
        make_builtin_function("index", crate::type_methods::list_method_index),
    );
    namespace_store(
        ns,
        "count",
        make_builtin_function("count", crate::type_methods::list_method_count),
    );
    namespace_store(
        ns,
        "remove",
        make_builtin_function("remove", crate::type_methods::list_method_remove),
    );
}

// ── Str TypeDef ──────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/unicodeobject.py TypeDef("str", ...)

fn init_str_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", str_descr_new),
    );
    namespace_store(
        ns,
        "join",
        make_builtin_function("join", crate::type_methods::str_method_join),
    );
    namespace_store(
        ns,
        "split",
        make_builtin_function("split", crate::type_methods::str_method_split),
    );
    namespace_store(
        ns,
        "strip",
        make_builtin_function("strip", crate::type_methods::str_method_strip),
    );
    namespace_store(
        ns,
        "lstrip",
        make_builtin_function("lstrip", crate::type_methods::str_method_lstrip),
    );
    namespace_store(
        ns,
        "rstrip",
        make_builtin_function("rstrip", crate::type_methods::str_method_rstrip),
    );
    namespace_store(
        ns,
        "startswith",
        make_builtin_function("startswith", crate::type_methods::str_method_startswith),
    );
    namespace_store(
        ns,
        "endswith",
        make_builtin_function("endswith", crate::type_methods::str_method_endswith),
    );
    namespace_store(
        ns,
        "replace",
        make_builtin_function("replace", crate::type_methods::str_method_replace),
    );
    namespace_store(
        ns,
        "find",
        make_builtin_function("find", crate::type_methods::str_method_find),
    );
    namespace_store(
        ns,
        "rfind",
        make_builtin_function("rfind", crate::type_methods::str_method_rfind),
    );
    namespace_store(
        ns,
        "upper",
        make_builtin_function("upper", crate::type_methods::str_method_upper),
    );
    namespace_store(
        ns,
        "lower",
        make_builtin_function("lower", crate::type_methods::str_method_lower),
    );
    namespace_store(
        ns,
        "format",
        make_builtin_function("format", crate::type_methods::str_method_format),
    );
    namespace_store(
        ns,
        "encode",
        make_builtin_function("encode", crate::type_methods::str_method_encode),
    );
    namespace_store(
        ns,
        "isdigit",
        make_builtin_function("isdigit", crate::type_methods::str_method_isdigit),
    );
    namespace_store(
        ns,
        "isalpha",
        make_builtin_function("isalpha", crate::type_methods::str_method_isalpha),
    );
    namespace_store(
        ns,
        "isidentifier",
        make_builtin_function("isidentifier", crate::type_methods::str_method_isidentifier),
    );
    namespace_store(
        ns,
        "zfill",
        make_builtin_function("zfill", crate::type_methods::str_method_zfill),
    );
    namespace_store(
        ns,
        "count",
        make_builtin_function("count", crate::type_methods::str_method_count),
    );
    namespace_store(
        ns,
        "index",
        make_builtin_function("index", crate::type_methods::str_method_index),
    );
    namespace_store(
        ns,
        "title",
        make_builtin_function("title", crate::type_methods::str_method_title),
    );
    namespace_store(
        ns,
        "capitalize",
        make_builtin_function("capitalize", crate::type_methods::str_method_capitalize),
    );
    namespace_store(
        ns,
        "swapcase",
        make_builtin_function("swapcase", crate::type_methods::str_method_swapcase),
    );
    namespace_store(
        ns,
        "center",
        make_builtin_function("center", crate::type_methods::str_method_center),
    );
    namespace_store(
        ns,
        "ljust",
        make_builtin_function("ljust", crate::type_methods::str_method_ljust),
    );
    namespace_store(
        ns,
        "rjust",
        make_builtin_function("rjust", crate::type_methods::str_method_rjust),
    );
    namespace_store(
        ns,
        "isspace",
        make_builtin_function("isspace", crate::type_methods::str_method_isspace),
    );
    namespace_store(
        ns,
        "isupper",
        make_builtin_function("isupper", crate::type_methods::str_method_isupper),
    );
    namespace_store(
        ns,
        "islower",
        make_builtin_function("islower", crate::type_methods::str_method_islower),
    );
    namespace_store(
        ns,
        "isalnum",
        make_builtin_function("isalnum", crate::type_methods::str_method_isalnum),
    );
    namespace_store(
        ns,
        "isascii",
        make_builtin_function("isascii", crate::type_methods::str_method_isascii),
    );
    namespace_store(
        ns,
        "partition",
        make_builtin_function("partition", crate::type_methods::str_method_partition),
    );
    namespace_store(
        ns,
        "rpartition",
        make_builtin_function("rpartition", crate::type_methods::str_method_rpartition),
    );
    namespace_store(
        ns,
        "splitlines",
        make_builtin_function("splitlines", crate::type_methods::str_method_splitlines),
    );
    namespace_store(
        ns,
        "removeprefix",
        make_builtin_function("removeprefix", crate::type_methods::str_method_removeprefix),
    );
    namespace_store(
        ns,
        "removesuffix",
        make_builtin_function("removesuffix", crate::type_methods::str_method_removesuffix),
    );
    namespace_store(
        ns,
        "expandtabs",
        make_builtin_function("expandtabs", crate::type_methods::str_method_expandtabs),
    );
    namespace_store(
        ns,
        "translate",
        make_builtin_function("translate", crate::type_methods::str_method_translate),
    );
    // str dunder methods
    namespace_store(
        ns,
        "__contains__",
        make_builtin_function("__contains__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            Ok(pyre_object::w_bool_from(
                crate::baseobjspace::contains(args[0], args[1]).unwrap_or(false),
            ))
        }),
    );
    namespace_store(
        ns,
        "__len__",
        make_builtin_function("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            crate::baseobjspace::len(args[0])
        }),
    );
    namespace_store(
        ns,
        "__getitem__",
        make_builtin_function("__getitem__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__getitem__"));
            }
            crate::baseobjspace::getitem(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "__iter__",
        make_builtin_function("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            crate::baseobjspace::iter(args[0])
        }),
    );
    namespace_store(
        ns,
        "__add__",
        make_builtin_function("__add__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__add__"));
            }
            crate::baseobjspace::add(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "__mul__",
        make_builtin_function("__mul__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__mul__"));
            }
            crate::baseobjspace::mul(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "__mod__",
        make_builtin_function("__mod__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__mod__"));
            }
            crate::baseobjspace::mod_(args[0], args[1])
        }),
    );
    // maketrans — PyPy: unicodeobject.py descr_maketrans
    namespace_store(
        ns,
        "maketrans",
        make_builtin_function("maketrans", |args| {
            // maketrans(x[, y[, z]]) → translation dict
            let d = pyre_object::w_dict_new();
            if args.len() >= 3 {
                // maketrans(x, y, z) — z is chars to delete (map to None)
                let x = unsafe { pyre_object::w_str_get_value(args[0]) };
                let y = unsafe { pyre_object::w_str_get_value(args[1]) };
                let z = unsafe { pyre_object::w_str_get_value(args[2]) };
                for (xc, yc) in x.chars().zip(y.chars()) {
                    unsafe {
                        pyre_object::w_dict_store(
                            d,
                            pyre_object::w_int_new(xc as i64),
                            pyre_object::w_int_new(yc as i64),
                        );
                    }
                }
                for zc in z.chars() {
                    unsafe {
                        pyre_object::w_dict_store(
                            d,
                            pyre_object::w_int_new(zc as i64),
                            pyre_object::w_none(),
                        );
                    }
                }
            } else if args.len() >= 2 {
                let x = unsafe { pyre_object::w_str_get_value(args[0]) };
                let y = unsafe { pyre_object::w_str_get_value(args[1]) };
                for (xc, yc) in x.chars().zip(y.chars()) {
                    unsafe {
                        pyre_object::w_dict_store(
                            d,
                            pyre_object::w_int_new(xc as i64),
                            pyre_object::w_int_new(yc as i64),
                        );
                    }
                }
            } else if args.len() == 1 && unsafe { pyre_object::is_dict(args[0]) } {
                // 1-arg dict form: maketrans({ord_or_char: replacement, ...})
                let src = args[0];
                unsafe {
                    let entries = &*((src as *const pyre_object::dictobject::W_DictObject)
                        .as_ref()
                        .unwrap()
                        .entries);
                    for &(k, v) in entries {
                        let ord_key = if pyre_object::is_int(k) {
                            k
                        } else if pyre_object::is_str(k) {
                            let s = pyre_object::w_str_get_value(k);
                            let ch = s.chars().next().unwrap_or('\0');
                            pyre_object::w_int_new(ch as i64)
                        } else {
                            k
                        };
                        pyre_object::w_dict_store(d, ord_key, v);
                    }
                }
            }
            Ok(d)
        }),
    );
}

// ── Dict TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/dictobject.py TypeDef("dict", ...)

fn init_dict_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", dict_descr_new),
    );
    namespace_store(
        ns,
        "get",
        make_builtin_function("get", crate::type_methods::dict_method_get),
    );
    namespace_store(
        ns,
        "keys",
        make_builtin_function("keys", crate::type_methods::dict_method_keys),
    );
    namespace_store(
        ns,
        "values",
        make_builtin_function("values", crate::type_methods::dict_method_values),
    );
    namespace_store(
        ns,
        "items",
        make_builtin_function("items", crate::type_methods::dict_method_items),
    );
    namespace_store(
        ns,
        "update",
        make_builtin_function("update", crate::type_methods::dict_method_update),
    );
    namespace_store(
        ns,
        "pop",
        make_builtin_function("pop", crate::type_methods::dict_method_pop),
    );
    namespace_store(
        ns,
        "setdefault",
        make_builtin_function("setdefault", crate::type_methods::dict_method_setdefault),
    );
    namespace_store(
        ns,
        "__setitem__",
        make_builtin_function("__setitem__", |args| {
            if args.len() < 3 {
                return Err(crate::PyError::type_error("__setitem__ requires 3 args"));
            }
            // For plain dict: direct store. For dict subclass instance: use backing dict.
            unsafe {
                if pyre_object::is_dict(args[0]) {
                    pyre_object::w_dict_store(args[0], args[1], args[2]);
                } else if pyre_object::is_instance(args[0]) {
                    // dict subclass — store in __dict_data__ backing dict
                    if let Ok(backing) = crate::baseobjspace::getattr(args[0], "__dict_data__") {
                        if pyre_object::is_dict(backing) {
                            pyre_object::w_dict_store(backing, args[1], args[2]);
                        }
                    }
                }
            }
            Ok(pyre_object::w_none())
        }),
    );
    namespace_store(
        ns,
        "__getitem__",
        make_builtin_function("__getitem__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__getitem__ requires 2 args"));
            }
            unsafe {
                if pyre_object::is_dict(args[0]) {
                    return crate::baseobjspace::getitem(args[0], args[1]);
                }
                if pyre_object::is_instance(args[0]) {
                    if let Ok(backing) = crate::baseobjspace::getattr(args[0], "__dict_data__") {
                        if pyre_object::is_dict(backing) {
                            return crate::baseobjspace::getitem(backing, args[1]);
                        }
                    }
                }
            }
            crate::baseobjspace::getitem(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "__contains__",
        make_builtin_function("__contains__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            let dict = crate::type_methods::resolve_dict_backing(args[0]);
            if !dict.is_null() {
                // Dict or dict subclass — look up in backing dict
                return Ok(pyre_object::w_bool_from(
                    unsafe { pyre_object::w_dict_lookup(dict, args[1]) }.is_some(),
                ));
            }
            Ok(pyre_object::w_bool_from(
                crate::baseobjspace::contains(args[0], args[1]).unwrap_or(false),
            ))
        }),
    );
    namespace_store(
        ns,
        "__len__",
        make_builtin_function("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            let dict = crate::type_methods::resolve_dict_backing(args[0]);
            if !dict.is_null() {
                return Ok(pyre_object::w_int_new(
                    unsafe { pyre_object::w_dict_len(dict) } as i64,
                ));
            }
            crate::baseobjspace::len(args[0])
        }),
    );
    namespace_store(
        ns,
        "__iter__",
        make_builtin_function("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let dict = crate::type_methods::resolve_dict_backing(args[0]);
            if !dict.is_null() {
                // Iterate over dict keys
                return crate::baseobjspace::iter(dict);
            }
            crate::baseobjspace::iter(args[0])
        }),
    );
    namespace_store(
        ns,
        "__delitem__",
        make_builtin_function("__delitem__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__delitem__ requires 2 args"));
            }
            crate::baseobjspace::delitem(args[0], args[1])?;
            Ok(pyre_object::w_none())
        }),
    );
    namespace_store(
        ns,
        "__eq__",
        make_builtin_function("__eq__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Eq)
        }),
    );
    namespace_store(
        ns,
        "__or__",
        make_builtin_function("__or__", |args| {
            // dict | dict → merge
            if args.len() < 2 {
                return Ok(args[0]);
            }
            Ok(args[0]) // stub
        }),
    );
    namespace_store(
        ns,
        "copy",
        make_builtin_function("copy", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_dict_new());
            }
            // Shallow copy — resolve backing dict for subclass instances
            let src = crate::type_methods::resolve_dict_backing(args[0]);
            if src.is_null() {
                return Ok(pyre_object::w_dict_new());
            }
            let dst = pyre_object::w_dict_new();
            unsafe {
                let d = &*(src as *const pyre_object::dictobject::W_DictObject);
                for &(k, v) in &*d.entries {
                    pyre_object::w_dict_store(dst, k, v);
                }
            }
            Ok(dst)
        }),
    );
    namespace_store(
        ns,
        "clear",
        make_builtin_function("clear", |_args| Ok(pyre_object::w_none())),
    );
    // dict.fromkeys(iterable, value=None) — classmethod
    namespace_store(
        ns,
        "fromkeys",
        make_builtin_function("fromkeys", |args| {
            // Called as dict.fromkeys(iter, val): args = [iter, val] (no cls binding)
            let (iterable, value) = if args.len() >= 2 {
                (args[0], args[1])
            } else if args.len() == 1 {
                (args[0], pyre_object::w_none())
            } else {
                return Ok(pyre_object::w_dict_new());
            };
            let d = pyre_object::w_dict_new();
            let items = crate::builtins::collect_iterable(iterable)?;
            for key in items {
                unsafe { pyre_object::w_dict_store(d, key, value) };
            }
            Ok(d)
        }),
    );
}

// ── Tuple TypeDef ────────────────────────────────────────────────────

fn init_tuple_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", tuple_descr_new),
    );
    namespace_store(
        ns,
        "index",
        make_builtin_function("index", crate::type_methods::tuple_method_index),
    );
    namespace_store(
        ns,
        "count",
        make_builtin_function("count", crate::type_methods::tuple_method_count),
    );
    namespace_store(
        ns,
        "__contains__",
        make_builtin_function("__contains__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            Ok(pyre_object::w_bool_from(
                crate::baseobjspace::contains(args[0], args[1]).unwrap_or(false),
            ))
        }),
    );
    namespace_store(
        ns,
        "__len__",
        make_builtin_function("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            Ok(pyre_object::w_int_new(
                unsafe { pyre_object::w_tuple_len(args[0]) } as i64,
            ))
        }),
    );
    namespace_store(
        ns,
        "__iter__",
        make_builtin_function("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            crate::baseobjspace::iter(args[0])
        }),
    );
}

// ── Int/Float/Bool TypeDef (minimal) ─────────────────────────────────

// ── Type TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/typeobject.py TypeDef("type", ...)

/// types.UnionType — PyPy: _pypy_generic_alias.py UnionType
fn init_union_type(ns: &mut PyNamespace) {
    // UnionType.__args__ — returns the tuple of union member types
    namespace_store(
        ns,
        "__args__",
        make_builtin_function("__args__", |args| {
            let self_ = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if unsafe { pyre_object::is_union(self_) } {
                Ok(unsafe { pyre_object::w_union_get_args(self_) })
            } else {
                Ok(pyre_object::PY_NULL)
            }
        }),
    );
    // UnionType.__or__ — PyPy: UnionType.__or__ → _create_union
    namespace_store(
        ns,
        "__or__",
        make_builtin_function("__or__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__or__ requires 2 arguments"));
            }
            Ok(pyre_object::w_union_new(args[0], args[1]))
        }),
    );
    // UnionType.__ror__
    namespace_store(
        ns,
        "__ror__",
        make_builtin_function("__ror__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__ror__ requires 2 arguments"));
            }
            Ok(pyre_object::w_union_new(args[1], args[0]))
        }),
    );
}

thread_local! {
    static GETSET_DESCRIPTOR_TYPE: std::cell::OnceCell<pyre_object::PyObjectRef>
        = const { std::cell::OnceCell::new() };
}

fn getset_descriptor_type() -> pyre_object::PyObjectRef {
    GETSET_DESCRIPTOR_TYPE.with(|cell| {
        *cell.get_or_init(|| make_builtin_type("getset_descriptor", init_getset_descriptor_type))
    })
}

fn init_getset_descriptor_type(ns: &mut PyNamespace) {
    // descr.__get__(self, instance, owner=None) — call wrapped getter
    // with instance when it is non-null. Returns the descriptor itself when
    // accessed directly on the class.
    namespace_store(
        ns,
        "__get__",
        make_builtin_function("__get__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let descr_self = args[0];
            let instance = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            if instance.is_null() || unsafe { pyre_object::is_none(instance) } {
                return Ok(descr_self);
            }
            let wrapped = crate::baseobjspace::getattr(descr_self, "__wrapped__")?;
            Ok(crate::call::call_function_impl_raw(wrapped, &[instance]))
        }),
    );
}

/// Wrap a getter so that `descr.__get__(obj)` invokes the getter with obj.
/// Mimics CPython's `getset_descriptor` for type-level accessors like
/// `type.__dict__['__annotations__']`.
fn make_getset_descriptor(getter: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    use pyre_object::instanceobject::w_instance_new;
    let obj = w_instance_new(getset_descriptor_type());
    let _ = crate::baseobjspace::setattr(obj, "__wrapped__", getter);
    obj
}

fn init_type_type(ns: &mut PyNamespace) {
    // type.__new__(metatype, name, bases, dict) — creates new type
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", crate::builtins::type_descr_new),
    );
    // type.__init__ — no-op for now
    namespace_store(
        ns,
        "__init__",
        make_builtin_function("__init__", |_| Ok(pyre_object::w_none())),
    );
    // type.__annotations__ / __dict__ / __mro__ / __name__ / __bases__
    // are exposed as getset descriptors so
    // `type.__dict__['<name>'].__get__(cls)` invokes the underlying getter
    // and returns the real value (matching CPython's getset_descriptor).
    //
    // PyPy: pypy/objspace/std/typeobject.py get_annotations / descr_getdict
    // / descr_getmro / descr_getname / descr_getbases.
    let annotations_getter = make_builtin_function("__annotations__", |args| {
        if args.is_empty() {
            return Ok(pyre_object::w_dict_new());
        }
        let cls = args[0];
        // Pyre stores class annotations as an ATTR_TABLE entry on the
        // type object under the "__annotations__" key.
        let stored = crate::baseobjspace::ATTR_TABLE.with(|table| {
            let table = table.borrow();
            if let Some(attrs) = table.get(&(cls as usize)) {
                for (name, value) in attrs {
                    if name == "__annotations__" {
                        return Some(*value);
                    }
                }
            }
            None
        });
        Ok(stored.unwrap_or_else(pyre_object::w_dict_new))
    });
    namespace_store(
        ns,
        "__annotations__",
        make_getset_descriptor(annotations_getter),
    );

    let mro_getter = make_builtin_function("__mro__", |args| {
        if args.is_empty() {
            return Ok(pyre_object::w_tuple_new(vec![]));
        }
        let cls = args[0];
        unsafe {
            let mro_ptr = pyre_object::w_type_get_mro(cls);
            if mro_ptr.is_null() {
                return Ok(pyre_object::w_tuple_new(vec![]));
            }
            Ok(pyre_object::w_tuple_new((*mro_ptr).clone()))
        }
    });
    namespace_store(ns, "__mro__", make_getset_descriptor(mro_getter));

    let dict_getter = make_builtin_function("__dict__", |args| {
        if args.is_empty() {
            return Ok(pyre_object::w_dict_new());
        }
        let cls = args[0];
        unsafe {
            let ns_ptr = pyre_object::typeobject::w_type_get_dict_ptr(cls);
            if ns_ptr.is_null() {
                return Ok(pyre_object::w_dict_new());
            }
            let dict = pyre_object::w_dict_new_with_namespace(ns_ptr);
            let ns = &*(ns_ptr as *const PyNamespace);
            for (name, &value) in ns.entries() {
                pyre_object::w_dict_store(dict, pyre_object::w_str_new(name), value);
            }
            Ok(dict)
        }
    });
    namespace_store(ns, "__dict__", make_getset_descriptor(dict_getter));

    let name_getter = make_builtin_function("__name__", |args| {
        if args.is_empty() {
            return Ok(pyre_object::w_str_new(""));
        }
        unsafe {
            let name = pyre_object::w_type_get_name(args[0]);
            Ok(pyre_object::w_str_new(name))
        }
    });
    namespace_store(ns, "__name__", make_getset_descriptor(name_getter));

    let bases_getter = make_builtin_function("__bases__", |args| {
        if args.is_empty() {
            return Ok(pyre_object::w_tuple_new(vec![]));
        }
        unsafe {
            let bases = pyre_object::w_type_get_bases(args[0]);
            if bases.is_null() {
                return Ok(pyre_object::w_tuple_new(vec![]));
            }
            Ok(bases)
        }
    });
    namespace_store(ns, "__bases__", make_getset_descriptor(bases_getter));
}

/// function/builtin_function_or_method — PyPy: function.py Function typedef
/// descr_function_get (function.py:462): always returns a Method.
fn init_function_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__get__",
        make_builtin_function("__get__", |args| {
            let w_function = args.first().copied().unwrap_or(pyre_object::w_none());
            let w_obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let w_cls = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
            // function.py:466-468 descr_function_get
            let asking_for_bound = unsafe {
                (w_cls.is_null() || pyre_object::is_none(w_cls))
                    || (!w_obj.is_null() && !pyre_object::is_none(w_obj))
                    || std::ptr::eq(w_cls, gettypeobject(&pyre_object::NONE_TYPE))
            };
            if asking_for_bound {
                // function.py:470  Method(space, w_function, w_obj, w_cls)
                Ok(pyre_object::w_method_new(w_function, w_obj, w_cls))
            } else {
                // function.py:472  Method(space, w_function, None, w_cls)
                Ok(pyre_object::w_method_new(
                    w_function,
                    pyre_object::PY_NULL,
                    w_cls,
                ))
            }
        }),
    );
}

/// BuiltinCode.typedef (typedef.py) — code object attributes for builtins.
///
/// PyPy exposes co_name, co_varnames, co_argcount, co_flags, co_consts.
/// No __get__ — BuiltinCode is a code object, not a descriptor.
fn init_builtin_code_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "co_name",
        make_builtin_function("co_name", |args| {
            let code = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            if code.is_null() {
                return Ok(pyre_object::w_none());
            }
            let name = unsafe { crate::builtin_code_name(code) };
            Ok(pyre_object::w_str_new(name))
        }),
    );
}

fn init_method_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__func__",
        make_builtin_function("__func__", |args| {
            Ok(args
                .first()
                .map(|&method| unsafe { pyre_object::w_method_get_func(method) })
                .unwrap_or(pyre_object::w_none()))
        }),
    );
    namespace_store(
        ns,
        "__self__",
        make_builtin_function("__self__", |args| {
            Ok(args
                .first()
                .map(|&method| unsafe { pyre_object::w_method_get_self(method) })
                .unwrap_or(pyre_object::w_none()))
        }),
    );
}

fn init_code_type(_ns: &mut PyNamespace) {}

/// `staticmethod.__new__(cls, func)` — PyPy: function.py StaticMethod.descr__new__
fn init_staticmethod_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", |args| {
            // staticmethod(func) — args[0] is cls (staticmethod type), args[1] is func
            let func = if args.len() > 1 {
                args[1]
            } else {
                pyre_object::w_none()
            };
            Ok(pyre_object::propertyobject::w_staticmethod_new(func))
        }),
    );
}

/// `classmethod.__new__(cls, func)` — PyPy: function.py ClassMethod.descr__new__
fn init_classmethod_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", |args| {
            let func = if args.len() > 1 {
                args[1]
            } else {
                pyre_object::w_none()
            };
            Ok(pyre_object::propertyobject::w_classmethod_new(func))
        }),
    );
}

fn init_int_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", int_descr_new),
    );
    namespace_store(
        ns,
        "bit_length",
        make_builtin_function("bit_length", |args| {
            let val = if !args.is_empty() && unsafe { pyre_object::is_int(args[0]) } {
                unsafe { pyre_object::w_int_get_value(args[0]) }
            } else {
                0
            };
            let bits = if val == 0 {
                0
            } else {
                64 - val.unsigned_abs().leading_zeros()
            };
            Ok(pyre_object::w_int_new(bits as i64))
        }),
    );
    namespace_store(
        ns,
        "bit_count",
        make_builtin_function("bit_count", |args| {
            let val = if !args.is_empty() && unsafe { pyre_object::is_int(args[0]) } {
                unsafe { pyre_object::w_int_get_value(args[0]) }
            } else {
                0
            };
            Ok(pyre_object::w_int_new(
                val.unsigned_abs().count_ones() as i64
            ))
        }),
    );
    // int.to_bytes(length=1, byteorder='big', *, signed=False)
    // PyPy: longobject.py descr_to_bytes
    namespace_store(
        ns,
        "to_bytes",
        make_builtin_function("to_bytes", |args| {
            let val = if !args.is_empty() && unsafe { pyre_object::is_int(args[0]) } {
                unsafe { pyre_object::w_int_get_value(args[0]) }
            } else {
                0
            };
            let length = if args.len() >= 2 && unsafe { pyre_object::is_int(args[1]) } {
                unsafe { pyre_object::w_int_get_value(args[1]) as usize }
            } else {
                1
            };
            let little_endian = if args.len() >= 3 && unsafe { pyre_object::is_str(args[2]) } {
                unsafe { pyre_object::w_str_get_value(args[2]) == "little" }
            } else {
                false
            };
            let mut bytes = vec![0u8; length];
            let uval = val as u64;
            for i in 0..length {
                let shift = if little_endian { i } else { length - 1 - i } * 8;
                bytes[i] = ((uval >> shift) & 0xff) as u8;
            }
            Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(&bytes))
        }),
    );
    // int.from_bytes(bytes, byteorder='big', *, signed=False) — classmethod in CPython
    namespace_store(
        ns,
        "from_bytes",
        make_builtin_function("from_bytes", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            // First arg can be the cls or the data depending on call site.
            let data_arg = if args.len() >= 2 && !unsafe { pyre_object::is_type(args[0]) } {
                args[0]
            } else if args.len() >= 2 {
                args[1]
            } else {
                args[0]
            };
            let byteorder_arg = if args.len() >= 3 {
                args[2]
            } else if args.len() >= 2 && unsafe { pyre_object::is_str(args[1]) } {
                args[1]
            } else {
                pyre_object::w_str_new("big")
            };
            let bytes: Vec<u8> = unsafe {
                if pyre_object::bytearrayobject::is_bytearray(data_arg) {
                    pyre_object::bytearrayobject::w_bytearray_data(data_arg).to_vec()
                } else if pyre_object::is_str(data_arg) {
                    pyre_object::w_str_get_value(data_arg).as_bytes().to_vec()
                } else {
                    vec![]
                }
            };
            let little_endian = unsafe {
                pyre_object::is_str(byteorder_arg)
                    && pyre_object::w_str_get_value(byteorder_arg) == "little"
            };
            let mut val: u64 = 0;
            if little_endian {
                for (i, &b) in bytes.iter().enumerate() {
                    val |= (b as u64) << (i * 8);
                }
            } else {
                for &b in &bytes {
                    val = (val << 8) | b as u64;
                }
            }
            Ok(pyre_object::w_int_new(val as i64))
        }),
    );
    // int.__index__ / __int__ / __trunc__ — identity
    for method in ["__index__", "__int__", "__trunc__"] {
        namespace_store(
            ns,
            method,
            make_builtin_function(method, |args| {
                Ok(args.first().copied().unwrap_or(pyre_object::w_int_new(0)))
            }),
        );
    }
    // int.conjugate — identity
    namespace_store(
        ns,
        "conjugate",
        make_builtin_function("conjugate", |args| {
            Ok(args.first().copied().unwrap_or(pyre_object::w_int_new(0)))
        }),
    );
    // int.real / int.imag — return value or 0
    namespace_store(
        ns,
        "real",
        make_builtin_function("real", |args| {
            Ok(args.first().copied().unwrap_or(pyre_object::w_int_new(0)))
        }),
    );
    namespace_store(
        ns,
        "imag",
        make_builtin_function("imag", |_| Ok(pyre_object::w_int_new(0))),
    );
    namespace_store(
        ns,
        "numerator",
        make_builtin_function("numerator", |args| {
            Ok(args.first().copied().unwrap_or(pyre_object::w_int_new(0)))
        }),
    );
    namespace_store(
        ns,
        "denominator",
        make_builtin_function("denominator", |_| Ok(pyre_object::w_int_new(1))),
    );
}
fn init_float_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", float_descr_new),
    );
}
fn init_bool_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", bool_descr_new),
    );
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

fn init_object_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__new__",
        make_builtin_function("__new__", object_descr_new),
    );
    namespace_store(
        ns,
        "__init__",
        make_builtin_function("__init__", object_descr_init),
    );
    // PyPy: objectobject.py — default comparison/hash/repr for all objects
    namespace_store(
        ns,
        "__eq__",
        make_builtin_function("__eq__", |args| {
            Ok(pyre_object::w_bool_from(
                args.len() >= 2 && std::ptr::eq(args[0], args[1]),
            ))
        }),
    );
    namespace_store(
        ns,
        "__ne__",
        make_builtin_function("__ne__", |args| {
            Ok(pyre_object::w_bool_from(
                args.len() >= 2 && !std::ptr::eq(args[0], args[1]),
            ))
        }),
    );
    namespace_store(
        ns,
        "__hash__",
        make_builtin_function("__hash__", |args| {
            Ok(pyre_object::w_int_new(if args.is_empty() {
                0
            } else {
                args[0] as i64
            }))
        }),
    );
    namespace_store(
        ns,
        "__repr__",
        // PyPy: objectobject.py descr___repr__ — base __repr__ for all objects
        make_builtin_function("__repr__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new("<object>"));
            }
            let obj = args[0];
            unsafe {
                if pyre_object::is_instance(obj) {
                    let w_type = pyre_object::w_instance_get_type(obj);
                    let name = pyre_object::w_type_get_name(w_type);
                    return Ok(pyre_object::w_str_new(&format!(
                        "<{name} object at {obj:?}>"
                    )));
                }
            }
            // For non-instances, delegate to display
            Ok(pyre_object::w_str_new(&format!("<object at {:?}>", obj)))
        }),
    );
    namespace_store(
        ns,
        "__str__",
        make_builtin_function("__str__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new("<object>"));
            }
            // Delegate to __repr__ to avoid infinite recursion
            // PyPy: objectobject.py descr___str__ → space.repr(w_self)
            Ok(pyre_object::w_str_new(&unsafe { crate::py_repr(args[0]) }))
        }),
    );
    // PyPy: objectobject.py descr___format__
    namespace_store(
        ns,
        "__format__",
        make_builtin_function("__format__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new(""));
            }
            Ok(pyre_object::w_str_new(&unsafe { crate::py_str(args[0]) }))
        }),
    );
    // PyPy: objectobject.py descr___reduce_ex__
    namespace_store(
        ns,
        "__reduce_ex__",
        make_builtin_function("__reduce_ex__", |_| Ok(pyre_object::w_none())),
    );
    namespace_store(
        ns,
        "__init_subclass__",
        make_builtin_function("__init_subclass__", |_| Ok(pyre_object::w_none())),
    );
    namespace_store(
        ns,
        "__subclasshook__",
        make_builtin_function("__subclasshook__", |_| Ok(pyre_object::w_not_implemented())),
    );
    // PyPy: objectobject.py descr___setattr__
    // object.__setattr__(self, name, value) → setattr dispatch
    namespace_store(
        ns,
        "__setattr__",
        make_builtin_function("__setattr__", |args| {
            if args.len() < 3 {
                return Err(crate::PyError::type_error(
                    "__setattr__ requires 3 arguments",
                ));
            }
            let name = unsafe { pyre_object::w_str_get_value(args[1]) };
            crate::baseobjspace::setattr(args[0], name, args[2])
        }),
    );
    // PyPy: objectobject.py descr___delattr__
    namespace_store(
        ns,
        "__delattr__",
        make_builtin_function("__delattr__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(
                    "__delattr__ requires 2 arguments",
                ));
            }
            let name = unsafe { pyre_object::w_str_get_value(args[1]) };
            crate::baseobjspace::delattr(args[0], name)
        }),
    );
    // PyPy: objectobject.py descr___getattribute__
    namespace_store(
        ns,
        "__getattribute__",
        make_builtin_function("__getattribute__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(
                    "__getattribute__ requires 2 arguments",
                ));
            }
            let name = unsafe { pyre_object::w_str_get_value(args[1]) };
            crate::baseobjspace::getattr(args[0], name)
        }),
    );
}

/// PyPy: bytearrayobject.py W_BytearrayObject.typedef
fn init_bytearray_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "find",
        make_builtin_function("find", |args| {
            assert!(args.len() >= 2, "find() takes at least 1 argument");
            let ba = args[0];
            let value = args[1];
            let start = if args.len() > 2 {
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as usize
            } else {
                0
            };
            unsafe {
                let v = pyre_object::w_int_get_value(value) as u8;
                Ok(pyre_object::w_int_new(
                    pyre_object::bytearrayobject::w_bytearray_find(ba, v, start),
                ))
            }
        }),
    );
    namespace_store(
        ns,
        "__add__",
        make_builtin_function("__add__", |args| {
            assert!(args.len() >= 2, "__add__ requires 2 arguments");
            let a = args[0];
            let b = args[1];
            unsafe {
                let a_data = pyre_object::bytearrayobject::w_bytearray_data(a);
                let b_data = if pyre_object::bytearrayobject::is_bytearray(b) {
                    pyre_object::bytearrayobject::w_bytearray_data(b).to_vec()
                } else {
                    vec![]
                };
                let mut result = a_data.to_vec();
                result.extend_from_slice(&b_data);
                Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(
                    &result,
                ))
            }
        }),
    );
    namespace_store(
        ns,
        "__iadd__",
        make_builtin_function("__iadd__", |args| {
            assert!(args.len() >= 2);
            let ba = args[0];
            let other = args[1];
            unsafe {
                if pyre_object::bytearrayobject::is_bytearray(other) {
                    let data = pyre_object::bytearrayobject::w_bytearray_data(other).to_vec();
                    pyre_object::bytearrayobject::w_bytearray_extend(ba, &data);
                }
            }
            Ok(ba)
        }),
    );
    namespace_store(
        ns,
        "translate",
        make_builtin_function("translate", |args| {
            assert!(args.len() >= 2);
            let ba = args[0];
            let table = args[1];
            unsafe {
                let data = pyre_object::bytearrayobject::w_bytearray_data(ba);
                // Translation table may be bytes/bytearray, or str (because
                // pyre treats `b'...'` literals as str). Accept both.
                let table_bytes_owned;
                let table_data: &[u8] = if pyre_object::bytearrayobject::is_bytearray(table) {
                    pyre_object::bytearrayobject::w_bytearray_data(table)
                } else if pyre_object::is_str(table) {
                    table_bytes_owned = pyre_object::w_str_get_value(table).as_bytes();
                    table_bytes_owned
                } else {
                    return Ok(ba);
                };
                let mut result = Vec::with_capacity(data.len());
                for &b in data {
                    if (b as usize) < table_data.len() {
                        result.push(table_data[b as usize]);
                    } else {
                        result.push(b);
                    }
                }
                Ok(pyre_object::bytearrayobject::w_bytearray_from_bytes(
                    &result,
                ))
            }
        }),
    );
}
