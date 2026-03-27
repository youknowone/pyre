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

use crate::{PyNamespace, builtin_code_new, namespace_store};

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
/// PyPy: `space.type(w_obj)`
/// - Instance of user class → w_instance_get_type
/// - Builtin object → type registry lookup
/// - Type object → type of type (not implemented, returns None)
pub fn r#type(obj: PyObjectRef) -> Option<PyObjectRef> {
    if obj.is_null() {
        return None;
    }
    unsafe {
        if is_instance(obj) {
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

    // builtin_function_or_method
    reg.insert(
        &crate::BUILTIN_CODE_TYPE as *const PyType as usize,
        new_typeobject_with_base(
            "builtin_function_or_method",
            init_builtin_function_type,
            object_type,
        ) as usize,
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

descr_new_wrapper!(int_descr_new, crate::builtins::builtin_int);
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
    namespace_store(ns, "__new__", builtin_code_new("__new__", list_descr_new));
    namespace_store(
        ns,
        "append",
        builtin_code_new("append", crate::type_methods::list_method_append),
    );
    namespace_store(
        ns,
        "extend",
        builtin_code_new("extend", crate::type_methods::list_method_extend),
    );
    namespace_store(
        ns,
        "copy",
        builtin_code_new("copy", crate::type_methods::list_method_copy),
    );
    namespace_store(
        ns,
        "insert",
        builtin_code_new("insert", crate::type_methods::list_method_insert),
    );
    namespace_store(
        ns,
        "pop",
        builtin_code_new("pop", crate::type_methods::list_method_pop),
    );
    namespace_store(
        ns,
        "clear",
        builtin_code_new("clear", crate::type_methods::list_method_clear),
    );
    namespace_store(
        ns,
        "reverse",
        builtin_code_new("reverse", crate::type_methods::list_method_reverse),
    );
    namespace_store(
        ns,
        "sort",
        builtin_code_new("sort", crate::type_methods::list_method_sort),
    );
    namespace_store(
        ns,
        "index",
        builtin_code_new("index", crate::type_methods::list_method_index),
    );
    namespace_store(
        ns,
        "count",
        builtin_code_new("count", crate::type_methods::list_method_count),
    );
    namespace_store(
        ns,
        "remove",
        builtin_code_new("remove", crate::type_methods::list_method_remove),
    );
}

// ── Str TypeDef ──────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/unicodeobject.py TypeDef("str", ...)

fn init_str_type(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", builtin_code_new("__new__", str_descr_new));
    namespace_store(
        ns,
        "join",
        builtin_code_new("join", crate::type_methods::str_method_join),
    );
    namespace_store(
        ns,
        "split",
        builtin_code_new("split", crate::type_methods::str_method_split),
    );
    namespace_store(
        ns,
        "strip",
        builtin_code_new("strip", crate::type_methods::str_method_strip),
    );
    namespace_store(
        ns,
        "lstrip",
        builtin_code_new("lstrip", crate::type_methods::str_method_lstrip),
    );
    namespace_store(
        ns,
        "rstrip",
        builtin_code_new("rstrip", crate::type_methods::str_method_rstrip),
    );
    namespace_store(
        ns,
        "startswith",
        builtin_code_new("startswith", crate::type_methods::str_method_startswith),
    );
    namespace_store(
        ns,
        "endswith",
        builtin_code_new("endswith", crate::type_methods::str_method_endswith),
    );
    namespace_store(
        ns,
        "replace",
        builtin_code_new("replace", crate::type_methods::str_method_replace),
    );
    namespace_store(
        ns,
        "find",
        builtin_code_new("find", crate::type_methods::str_method_find),
    );
    namespace_store(
        ns,
        "rfind",
        builtin_code_new("rfind", crate::type_methods::str_method_rfind),
    );
    namespace_store(
        ns,
        "upper",
        builtin_code_new("upper", crate::type_methods::str_method_upper),
    );
    namespace_store(
        ns,
        "lower",
        builtin_code_new("lower", crate::type_methods::str_method_lower),
    );
    namespace_store(
        ns,
        "format",
        builtin_code_new("format", crate::type_methods::str_method_format),
    );
    namespace_store(
        ns,
        "encode",
        builtin_code_new("encode", crate::type_methods::str_method_encode),
    );
    namespace_store(
        ns,
        "isdigit",
        builtin_code_new("isdigit", crate::type_methods::str_method_isdigit),
    );
    namespace_store(
        ns,
        "isalpha",
        builtin_code_new("isalpha", crate::type_methods::str_method_isalpha),
    );
    namespace_store(
        ns,
        "zfill",
        builtin_code_new("zfill", crate::type_methods::str_method_zfill),
    );
    namespace_store(
        ns,
        "count",
        builtin_code_new("count", crate::type_methods::str_method_count),
    );
    namespace_store(
        ns,
        "index",
        builtin_code_new("index", crate::type_methods::str_method_index),
    );
    namespace_store(
        ns,
        "title",
        builtin_code_new("title", crate::type_methods::str_method_title),
    );
    namespace_store(
        ns,
        "capitalize",
        builtin_code_new("capitalize", crate::type_methods::str_method_capitalize),
    );
    namespace_store(
        ns,
        "swapcase",
        builtin_code_new("swapcase", crate::type_methods::str_method_swapcase),
    );
    namespace_store(
        ns,
        "center",
        builtin_code_new("center", crate::type_methods::str_method_center),
    );
    namespace_store(
        ns,
        "ljust",
        builtin_code_new("ljust", crate::type_methods::str_method_ljust),
    );
    namespace_store(
        ns,
        "rjust",
        builtin_code_new("rjust", crate::type_methods::str_method_rjust),
    );
    namespace_store(
        ns,
        "isspace",
        builtin_code_new("isspace", crate::type_methods::str_method_isspace),
    );
    namespace_store(
        ns,
        "isupper",
        builtin_code_new("isupper", crate::type_methods::str_method_isupper),
    );
    namespace_store(
        ns,
        "islower",
        builtin_code_new("islower", crate::type_methods::str_method_islower),
    );
    namespace_store(
        ns,
        "isalnum",
        builtin_code_new("isalnum", crate::type_methods::str_method_isalnum),
    );
    namespace_store(
        ns,
        "isascii",
        builtin_code_new("isascii", crate::type_methods::str_method_isascii),
    );
    namespace_store(
        ns,
        "partition",
        builtin_code_new("partition", crate::type_methods::str_method_partition),
    );
    namespace_store(
        ns,
        "rpartition",
        builtin_code_new("rpartition", crate::type_methods::str_method_rpartition),
    );
    namespace_store(
        ns,
        "splitlines",
        builtin_code_new("splitlines", crate::type_methods::str_method_splitlines),
    );
    namespace_store(
        ns,
        "removeprefix",
        builtin_code_new("removeprefix", crate::type_methods::str_method_removeprefix),
    );
    namespace_store(
        ns,
        "removesuffix",
        builtin_code_new("removesuffix", crate::type_methods::str_method_removesuffix),
    );
    namespace_store(
        ns,
        "expandtabs",
        builtin_code_new("expandtabs", crate::type_methods::str_method_expandtabs),
    );
    // str dunder methods
    namespace_store(
        ns,
        "__contains__",
        builtin_code_new("__contains__", |args| {
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
        builtin_code_new("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            crate::baseobjspace::len(args[0])
        }),
    );
    namespace_store(
        ns,
        "__getitem__",
        builtin_code_new("__getitem__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__getitem__"));
            }
            crate::baseobjspace::getitem(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "__iter__",
        builtin_code_new("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            crate::baseobjspace::iter(args[0])
        }),
    );
    namespace_store(
        ns,
        "__add__",
        builtin_code_new("__add__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__add__"));
            }
            crate::baseobjspace::add(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "__mul__",
        builtin_code_new("__mul__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__mul__"));
            }
            crate::baseobjspace::mul(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "__mod__",
        builtin_code_new("__mod__", |args| {
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
        builtin_code_new("maketrans", |args| {
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
            }
            Ok(d)
        }),
    );
}

// ── Dict TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/dictobject.py TypeDef("dict", ...)

fn init_dict_type(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", builtin_code_new("__new__", dict_descr_new));
    namespace_store(
        ns,
        "get",
        builtin_code_new("get", crate::type_methods::dict_method_get),
    );
    namespace_store(
        ns,
        "keys",
        builtin_code_new("keys", crate::type_methods::dict_method_keys),
    );
    namespace_store(
        ns,
        "values",
        builtin_code_new("values", crate::type_methods::dict_method_values),
    );
    namespace_store(
        ns,
        "items",
        builtin_code_new("items", crate::type_methods::dict_method_items),
    );
    namespace_store(
        ns,
        "update",
        builtin_code_new("update", crate::type_methods::dict_method_update),
    );
    namespace_store(
        ns,
        "pop",
        builtin_code_new("pop", crate::type_methods::dict_method_pop),
    );
    namespace_store(
        ns,
        "setdefault",
        builtin_code_new("setdefault", crate::type_methods::dict_method_setdefault),
    );
    namespace_store(
        ns,
        "__setitem__",
        builtin_code_new("__setitem__", |args| {
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
        builtin_code_new("__getitem__", |args| {
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
        builtin_code_new("__contains__", |args| {
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
        builtin_code_new("__len__", |args| {
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
        builtin_code_new("__iter__", |args| {
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
        builtin_code_new("__delitem__", |args| {
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
        builtin_code_new("__eq__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Eq)
        }),
    );
    namespace_store(
        ns,
        "__or__",
        builtin_code_new("__or__", |args| {
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
        builtin_code_new("copy", |args| {
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
        builtin_code_new("clear", |_args| Ok(pyre_object::w_none())),
    );
    // dict.fromkeys(iterable, value=None) — classmethod
    namespace_store(
        ns,
        "fromkeys",
        builtin_code_new("fromkeys", |args| {
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
    namespace_store(ns, "__new__", builtin_code_new("__new__", tuple_descr_new));
    namespace_store(
        ns,
        "index",
        builtin_code_new("index", crate::type_methods::tuple_method_index),
    );
    namespace_store(
        ns,
        "count",
        builtin_code_new("count", crate::type_methods::tuple_method_count),
    );
    namespace_store(
        ns,
        "__contains__",
        builtin_code_new("__contains__", |args| {
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
        builtin_code_new("__len__", |args| {
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
        builtin_code_new("__iter__", |args| {
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

fn init_type_type(ns: &mut PyNamespace) {
    // type.__new__(metatype, name, bases, dict) — creates new type
    namespace_store(
        ns,
        "__new__",
        builtin_code_new("__new__", crate::builtins::type_descr_new),
    );
    // type.__init__ — no-op for now
    namespace_store(
        ns,
        "__init__",
        builtin_code_new("__init__", |_| Ok(pyre_object::w_none())),
    );
}

/// function/builtin_function_or_method — PyPy: funcobject.py Function typedef
/// Functions are descriptors: __get__ returns the function itself (for class access)
/// or a bound method (for instance access).
fn init_function_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__get__",
        builtin_code_new("__get__", |args| {
            let func = args.first().copied().unwrap_or(pyre_object::w_none());
            let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
            let objtype = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
            if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
                Ok(func)
            } else {
                Ok(pyre_object::w_method_new(func, obj, objtype))
            }
        }),
    );
}

fn init_builtin_function_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__get__",
        builtin_code_new("__get__", |args| {
            Ok(args.first().copied().unwrap_or(pyre_object::w_none()))
        }),
    );
}

fn init_method_type(ns: &mut PyNamespace) {
    namespace_store(
        ns,
        "__func__",
        builtin_code_new("__func__", |args| {
            Ok(args
                .first()
                .map(|&method| unsafe { pyre_object::w_method_get_func(method) })
                .unwrap_or(pyre_object::w_none()))
        }),
    );
    namespace_store(
        ns,
        "__self__",
        builtin_code_new("__self__", |args| {
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
        builtin_code_new("__new__", |args| {
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
        builtin_code_new("__new__", |args| {
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
    namespace_store(ns, "__new__", builtin_code_new("__new__", int_descr_new));
}
fn init_float_type(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", builtin_code_new("__new__", float_descr_new));
}
fn init_bool_type(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", builtin_code_new("__new__", bool_descr_new));
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
    namespace_store(ns, "__new__", builtin_code_new("__new__", object_descr_new));
    namespace_store(
        ns,
        "__init__",
        builtin_code_new("__init__", object_descr_init),
    );
    // PyPy: objectobject.py — default comparison/hash/repr for all objects
    namespace_store(
        ns,
        "__eq__",
        builtin_code_new("__eq__", |args| {
            Ok(pyre_object::w_bool_from(
                args.len() >= 2 && std::ptr::eq(args[0], args[1]),
            ))
        }),
    );
    namespace_store(
        ns,
        "__ne__",
        builtin_code_new("__ne__", |args| {
            Ok(pyre_object::w_bool_from(
                args.len() >= 2 && !std::ptr::eq(args[0], args[1]),
            ))
        }),
    );
    namespace_store(
        ns,
        "__hash__",
        builtin_code_new("__hash__", |args| {
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
        builtin_code_new("__repr__", |args| {
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
        builtin_code_new("__str__", |args| {
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
        builtin_code_new("__format__", |args| {
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
        builtin_code_new("__reduce_ex__", |_| Ok(pyre_object::w_none())),
    );
    namespace_store(
        ns,
        "__init_subclass__",
        builtin_code_new("__init_subclass__", |_| Ok(pyre_object::w_none())),
    );
    namespace_store(
        ns,
        "__subclasshook__",
        builtin_code_new("__subclasshook__", |_| Ok(pyre_object::w_not_implemented())),
    );
}
