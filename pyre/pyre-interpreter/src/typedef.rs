//! TypeDef — builtin type descriptor registry.
//!
//! PyPy equivalent: pypy/interpreter/typedef.py
//!
//! Each builtin type (list, str, dict, tuple, int, float, bool, etc.)
//! gets a W_TypeObject with methods pre-installed in its namespace.
//! `py_getattr` looks up the type object from the registry and searches
//! its namespace via MRO, exactly like user-defined classes.
//!
//! This eliminates the `builtin_type_method` match-based dispatch and
//! unifies attribute lookup for all object types.

use std::collections::HashMap;
use std::sync::OnceLock;

use pyre_object::pyobject::*;
use pyre_object::*;

use crate::{PyNamespace, namespace_store, w_builtin_func_new};

/// Global type registry: maps static PyType pointer → W_TypeObject (as usize).
///
/// PyPy equivalent: space.gettypeobject(cls.typedef) → cached W_TypeObject
/// Stored as usize to satisfy Send+Sync requirements of OnceLock.
static TYPE_REGISTRY: OnceLock<HashMap<usize, usize>> = OnceLock::new();

/// Look up the W_TypeObject for a builtin type.
///
/// PyPy: `space.type(w_obj)` → W_TypeObject
pub fn get_type_object(tp: *const PyType) -> Option<PyObjectRef> {
    TYPE_REGISTRY
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
pub fn type_of(obj: PyObjectRef) -> Option<PyObjectRef> {
    if obj.is_null() {
        return None;
    }
    unsafe {
        if is_instance(obj) {
            return Some(w_instance_get_type(obj));
        }
        if is_type(obj) {
            return None; // type of type — metaclass, not yet
        }
        let tp = (*obj).ob_type;
        get_type_object(tp)
    }
}

/// Initialize the type registry with all builtin types.
///
/// PyPy: each W_XxxObject.typedef = TypeDef("xxx", ...) is set at
/// module load time. In pyre, we do it once at startup.
///
/// Must be called before any py_getattr on builtin objects.
pub fn install_builtin_typedefs() {
    if TYPE_REGISTRY.get().is_some() {
        return;
    }

    let mut reg: HashMap<usize, usize> = HashMap::new();

    // 'object' first — PyPy: objectobject.py W_ObjectObject.typedef
    // MRO = [object]. All other types inherit from object.
    let object_type = make_type_root("object", init_object_typedef);
    reg.insert(
        &INSTANCE_TYPE as *const PyType as usize,
        object_type as usize,
    );
    let _ = OBJECT_TYPE_OBJ.set(object_type as usize);

    // type — PyPy: typeobject.py, bases=(object,)
    // type.__new__(metatype, name, bases, dict) creates new types
    let type_type = make_type_with_base("type", init_type_typedef, object_type);
    reg.insert(&TYPE_TYPE as *const PyType as usize, type_type as usize);
    let _ = TYPE_TYPE_OBJ.set(type_type as usize);

    // int — PyPy: intobject.py W_IntObject.typedef, bases=(object,)
    let int_type = make_type_with_base("int", init_int_typedef, object_type);
    reg.insert(&INT_TYPE as *const PyType as usize, int_type as usize);

    // float — PyPy: floatobject.py, bases=(object,)
    reg.insert(
        &FLOAT_TYPE as *const PyType as usize,
        make_type_with_base("float", init_float_typedef, object_type) as usize,
    );

    // bool — PyPy: boolobject.py, bases=(int,)
    reg.insert(
        &BOOL_TYPE as *const PyType as usize,
        make_type_with_base("bool", init_bool_typedef, int_type) as usize,
    );

    // str — PyPy: unicodeobject.py, bases=(object,)
    reg.insert(
        &STR_TYPE as *const PyType as usize,
        make_type_with_base("str", init_str_typedef, object_type) as usize,
    );

    // list — PyPy: listobject.py, bases=(object,)
    reg.insert(
        &LIST_TYPE as *const PyType as usize,
        make_type_with_base("list", init_list_typedef, object_type) as usize,
    );

    // tuple — PyPy: tupleobject.py, bases=(object,)
    reg.insert(
        &TUPLE_TYPE as *const PyType as usize,
        make_type_with_base("tuple", init_tuple_typedef, object_type) as usize,
    );

    // dict — PyPy: dictobject.py, bases=(object,)
    reg.insert(
        &DICT_TYPE as *const PyType as usize,
        make_type_with_base("dict", init_dict_typedef, object_type) as usize,
    );

    // NoneType — bases=(object,)
    reg.insert(
        &NONE_TYPE as *const PyType as usize,
        make_type_with_base("NoneType", |_| {}, object_type) as usize,
    );

    let _ = TYPE_REGISTRY.set(reg);
}

/// The global `object` type object, accessible from builtins.
static OBJECT_TYPE_OBJ: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
/// The global `type` type object.
static TYPE_TYPE_OBJ: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Retrieve a W_TypeObject for a builtin type by its PyType pointer.
/// Used to register `int`, `str`, etc. as types in builtins (not functions).
/// Get the `type` W_TypeObject for use as a builtin.
pub fn get_type_type() -> PyObjectRef {
    TYPE_TYPE_OBJ
        .get()
        .map(|v| *v as PyObjectRef)
        .unwrap_or(PY_NULL)
}

pub fn get_builtin_type(tp: &PyType) -> PyObjectRef {
    get_type_object(tp as *const PyType).unwrap_or(PY_NULL)
}

/// Get the `object` W_TypeObject for use as a builtin.
pub fn get_object_type() -> PyObjectRef {
    OBJECT_TYPE_OBJ
        .get()
        .map(|v| *v as PyObjectRef)
        .unwrap_or(PY_NULL)
}

/// Create the root `object` type. MRO = [object].
fn make_type_root(name: &str, init: fn(&mut PyNamespace)) -> PyObjectRef {
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
fn make_type_with_base(name: &str, init: fn(&mut PyNamespace), base: PyObjectRef) -> PyObjectRef {
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
macro_rules! type_new_wrapper {
    ($fn_name:ident, $ctor:path) => {
        fn $fn_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            // args[0] = cls (ignored for builtin types)
            $ctor(&args[1..])
        }
    };
}

type_new_wrapper!(int_new, crate::builtins::builtin_int_pub);
type_new_wrapper!(float_new, crate::builtins::builtin_float_pub);
type_new_wrapper!(str_new, crate::builtins::builtin_str_pub);
type_new_wrapper!(bool_new, crate::builtins::builtin_bool_pub);
type_new_wrapper!(list_new, crate::builtins::builtin_list_ctor_pub);
type_new_wrapper!(tuple_new, crate::builtins::builtin_tuple_pub);
type_new_wrapper!(dict_new, crate::builtins::builtin_dict_ctor_pub);

// ── List TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/listobject.py TypeDef("list", ...)

fn init_list_typedef(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", w_builtin_func_new("__new__", list_new));
    namespace_store(
        ns,
        "append",
        w_builtin_func_new("append", crate::type_methods::list_method_append),
    );
    namespace_store(
        ns,
        "extend",
        w_builtin_func_new("extend", crate::type_methods::list_method_extend),
    );
    namespace_store(
        ns,
        "copy",
        w_builtin_func_new("copy", crate::type_methods::list_method_copy),
    );
    namespace_store(
        ns,
        "insert",
        w_builtin_func_new("insert", crate::type_methods::list_method_insert),
    );
    namespace_store(
        ns,
        "pop",
        w_builtin_func_new("pop", crate::type_methods::list_method_pop),
    );
    namespace_store(
        ns,
        "clear",
        w_builtin_func_new("clear", crate::type_methods::list_method_clear),
    );
    namespace_store(
        ns,
        "reverse",
        w_builtin_func_new("reverse", crate::type_methods::list_method_reverse),
    );
    namespace_store(
        ns,
        "sort",
        w_builtin_func_new("sort", crate::type_methods::list_method_sort),
    );
    namespace_store(
        ns,
        "index",
        w_builtin_func_new("index", crate::type_methods::list_method_index),
    );
    namespace_store(
        ns,
        "count",
        w_builtin_func_new("count", crate::type_methods::list_method_count),
    );
    namespace_store(
        ns,
        "remove",
        w_builtin_func_new("remove", crate::type_methods::list_method_remove),
    );
}

// ── Str TypeDef ──────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/unicodeobject.py TypeDef("str", ...)

fn init_str_typedef(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", w_builtin_func_new("__new__", str_new));
    namespace_store(
        ns,
        "join",
        w_builtin_func_new("join", crate::type_methods::str_method_join),
    );
    namespace_store(
        ns,
        "split",
        w_builtin_func_new("split", crate::type_methods::str_method_split),
    );
    namespace_store(
        ns,
        "strip",
        w_builtin_func_new("strip", crate::type_methods::str_method_strip),
    );
    namespace_store(
        ns,
        "lstrip",
        w_builtin_func_new("lstrip", crate::type_methods::str_method_lstrip),
    );
    namespace_store(
        ns,
        "rstrip",
        w_builtin_func_new("rstrip", crate::type_methods::str_method_rstrip),
    );
    namespace_store(
        ns,
        "startswith",
        w_builtin_func_new("startswith", crate::type_methods::str_method_startswith),
    );
    namespace_store(
        ns,
        "endswith",
        w_builtin_func_new("endswith", crate::type_methods::str_method_endswith),
    );
    namespace_store(
        ns,
        "replace",
        w_builtin_func_new("replace", crate::type_methods::str_method_replace),
    );
    namespace_store(
        ns,
        "find",
        w_builtin_func_new("find", crate::type_methods::str_method_find),
    );
    namespace_store(
        ns,
        "rfind",
        w_builtin_func_new("rfind", crate::type_methods::str_method_rfind),
    );
    namespace_store(
        ns,
        "upper",
        w_builtin_func_new("upper", crate::type_methods::str_method_upper),
    );
    namespace_store(
        ns,
        "lower",
        w_builtin_func_new("lower", crate::type_methods::str_method_lower),
    );
    namespace_store(
        ns,
        "format",
        w_builtin_func_new("format", crate::type_methods::str_method_format),
    );
    namespace_store(
        ns,
        "encode",
        w_builtin_func_new("encode", crate::type_methods::str_method_encode),
    );
    namespace_store(
        ns,
        "isdigit",
        w_builtin_func_new("isdigit", crate::type_methods::str_method_isdigit),
    );
    namespace_store(
        ns,
        "isalpha",
        w_builtin_func_new("isalpha", crate::type_methods::str_method_isalpha),
    );
    namespace_store(
        ns,
        "zfill",
        w_builtin_func_new("zfill", crate::type_methods::str_method_zfill),
    );
    namespace_store(
        ns,
        "count",
        w_builtin_func_new("count", crate::type_methods::str_method_count),
    );
    namespace_store(
        ns,
        "index",
        w_builtin_func_new("index", crate::type_methods::str_method_index),
    );
    namespace_store(
        ns,
        "title",
        w_builtin_func_new("title", crate::type_methods::str_method_title),
    );
    namespace_store(
        ns,
        "capitalize",
        w_builtin_func_new("capitalize", crate::type_methods::str_method_capitalize),
    );
    namespace_store(
        ns,
        "swapcase",
        w_builtin_func_new("swapcase", crate::type_methods::str_method_swapcase),
    );
    namespace_store(
        ns,
        "center",
        w_builtin_func_new("center", crate::type_methods::str_method_center),
    );
    namespace_store(
        ns,
        "ljust",
        w_builtin_func_new("ljust", crate::type_methods::str_method_ljust),
    );
    namespace_store(
        ns,
        "rjust",
        w_builtin_func_new("rjust", crate::type_methods::str_method_rjust),
    );
    namespace_store(
        ns,
        "isspace",
        w_builtin_func_new("isspace", crate::type_methods::str_method_isspace),
    );
    namespace_store(
        ns,
        "isupper",
        w_builtin_func_new("isupper", crate::type_methods::str_method_isupper),
    );
    namespace_store(
        ns,
        "islower",
        w_builtin_func_new("islower", crate::type_methods::str_method_islower),
    );
    namespace_store(
        ns,
        "isalnum",
        w_builtin_func_new("isalnum", crate::type_methods::str_method_isalnum),
    );
    namespace_store(
        ns,
        "isascii",
        w_builtin_func_new("isascii", crate::type_methods::str_method_isascii),
    );
    namespace_store(
        ns,
        "partition",
        w_builtin_func_new("partition", crate::type_methods::str_method_partition),
    );
    namespace_store(
        ns,
        "rpartition",
        w_builtin_func_new("rpartition", crate::type_methods::str_method_rpartition),
    );
    namespace_store(
        ns,
        "splitlines",
        w_builtin_func_new("splitlines", crate::type_methods::str_method_splitlines),
    );
    namespace_store(
        ns,
        "removeprefix",
        w_builtin_func_new("removeprefix", crate::type_methods::str_method_removeprefix),
    );
    namespace_store(
        ns,
        "removesuffix",
        w_builtin_func_new("removesuffix", crate::type_methods::str_method_removesuffix),
    );
    namespace_store(
        ns,
        "expandtabs",
        w_builtin_func_new("expandtabs", crate::type_methods::str_method_expandtabs),
    );
    // str dunder methods
    namespace_store(
        ns,
        "__contains__",
        w_builtin_func_new("__contains__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            Ok(pyre_object::w_bool_from(
                crate::space::py_contains(args[0], args[1]).unwrap_or(false),
            ))
        }),
    );
    namespace_store(
        ns,
        "__len__",
        w_builtin_func_new("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            crate::space::py_len(args[0])
        }),
    );
    namespace_store(
        ns,
        "__getitem__",
        w_builtin_func_new("__getitem__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__getitem__"));
            }
            crate::space::py_getitem(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "__iter__",
        w_builtin_func_new("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            crate::space::py_iter(args[0])
        }),
    );
    namespace_store(
        ns,
        "__add__",
        w_builtin_func_new("__add__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__add__"));
            }
            crate::space::py_add(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "__mul__",
        w_builtin_func_new("__mul__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__mul__"));
            }
            crate::space::py_mul(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "__mod__",
        w_builtin_func_new("__mod__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__mod__"));
            }
            crate::space::py_mod(args[0], args[1])
        }),
    );
    // maketrans — PyPy: unicodeobject.py descr_maketrans
    namespace_store(
        ns,
        "maketrans",
        w_builtin_func_new("maketrans", |args| {
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

fn init_dict_typedef(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", w_builtin_func_new("__new__", dict_new));
    namespace_store(
        ns,
        "get",
        w_builtin_func_new("get", crate::type_methods::dict_method_get),
    );
    namespace_store(
        ns,
        "keys",
        w_builtin_func_new("keys", crate::type_methods::dict_method_keys),
    );
    namespace_store(
        ns,
        "values",
        w_builtin_func_new("values", crate::type_methods::dict_method_values),
    );
    namespace_store(
        ns,
        "items",
        w_builtin_func_new("items", crate::type_methods::dict_method_items),
    );
    namespace_store(
        ns,
        "update",
        w_builtin_func_new("update", crate::type_methods::dict_method_update),
    );
    namespace_store(
        ns,
        "pop",
        w_builtin_func_new("pop", crate::type_methods::dict_method_pop),
    );
    namespace_store(
        ns,
        "setdefault",
        w_builtin_func_new("setdefault", crate::type_methods::dict_method_setdefault),
    );
    namespace_store(
        ns,
        "__setitem__",
        w_builtin_func_new("__setitem__", |args| {
            if args.len() < 3 {
                return Err(crate::PyError::type_error("__setitem__ requires 3 args"));
            }
            crate::space::py_setitem(args[0], args[1], args[2])
        }),
    );
    namespace_store(
        ns,
        "__getitem__",
        w_builtin_func_new("__getitem__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__getitem__ requires 2 args"));
            }
            crate::space::py_getitem(args[0], args[1])
        }),
    );
    namespace_store(
        ns,
        "__contains__",
        w_builtin_func_new("__contains__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            Ok(pyre_object::w_bool_from(
                crate::space::py_contains(args[0], args[1]).unwrap_or(false),
            ))
        }),
    );
    namespace_store(
        ns,
        "__len__",
        w_builtin_func_new("__len__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_int_new(0));
            }
            crate::space::py_len(args[0])
        }),
    );
    namespace_store(
        ns,
        "__iter__",
        w_builtin_func_new("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            crate::space::py_iter(args[0])
        }),
    );
    namespace_store(
        ns,
        "__delitem__",
        w_builtin_func_new("__delitem__", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("__delitem__ requires 2 args"));
            }
            crate::space::py_delitem(args[0], args[1])?;
            Ok(pyre_object::w_none())
        }),
    );
    namespace_store(
        ns,
        "__eq__",
        w_builtin_func_new("__eq__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            crate::space::py_compare(args[0], args[1], crate::space::CompareOp::Eq)
        }),
    );
    namespace_store(
        ns,
        "__or__",
        w_builtin_func_new("__or__", |args| {
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
        w_builtin_func_new("copy", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_dict_new());
            }
            // Shallow copy
            let src = args[0];
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
        w_builtin_func_new("clear", |_args| Ok(pyre_object::w_none())),
    );
    // dict.fromkeys(iterable, value=None) — classmethod
    namespace_store(
        ns,
        "fromkeys",
        w_builtin_func_new("fromkeys", |args| {
            // Called as dict.fromkeys(iter, val): args = [iter, val] (no cls binding)
            let (iterable, value) = if args.len() >= 2 {
                (args[0], args[1])
            } else if args.len() == 1 {
                (args[0], pyre_object::w_none())
            } else {
                return Ok(pyre_object::w_dict_new());
            };
            let d = pyre_object::w_dict_new();
            let items = crate::builtins::collect_iterable_pub(iterable)?;
            for key in items {
                unsafe { pyre_object::w_dict_store(d, key, value) };
            }
            Ok(d)
        }),
    );
}

// ── Tuple TypeDef ────────────────────────────────────────────────────

fn init_tuple_typedef(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", w_builtin_func_new("__new__", tuple_new));
    namespace_store(
        ns,
        "index",
        w_builtin_func_new("index", crate::type_methods::tuple_method_index),
    );
    namespace_store(
        ns,
        "count",
        w_builtin_func_new("count", crate::type_methods::tuple_method_count),
    );
    namespace_store(
        ns,
        "__contains__",
        w_builtin_func_new("__contains__", |args| {
            if args.len() < 2 {
                return Ok(pyre_object::w_bool_from(false));
            }
            Ok(pyre_object::w_bool_from(
                crate::space::py_contains(args[0], args[1]).unwrap_or(false),
            ))
        }),
    );
    namespace_store(
        ns,
        "__len__",
        w_builtin_func_new("__len__", |args| {
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
        w_builtin_func_new("__iter__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            crate::space::py_iter(args[0])
        }),
    );
}

// ── Int/Float/Bool TypeDef (minimal) ─────────────────────────────────

// ── Type TypeDef ─────────────────────────────────────────────────────
// PyPy: pypy/objspace/std/typeobject.py TypeDef("type", ...)

fn init_type_typedef(ns: &mut PyNamespace) {
    // type.__new__(metatype, name, bases, dict) — creates new type
    namespace_store(
        ns,
        "__new__",
        w_builtin_func_new("__new__", crate::builtins::builtin_type_new_pub),
    );
    // type.__init__ — no-op for now
    namespace_store(
        ns,
        "__init__",
        w_builtin_func_new("__init__", |_| Ok(pyre_object::w_none())),
    );
}

fn init_int_typedef(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", w_builtin_func_new("__new__", int_new));
}
fn init_float_typedef(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", w_builtin_func_new("__new__", float_new));
}
fn init_bool_typedef(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", w_builtin_func_new("__new__", bool_new));
}

// ── Object TypeDef ───────────────────────────────────────────────────
// PyPy: pypy/objspace/std/objectobject.py TypeDef("object", ...)

/// `object.__new__(cls)` — allocate a bare instance of cls.
///
/// PyPy: objectobject.py descr__new__
fn object_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(
        !args.is_empty(),
        "object.__new__() requires a type argument"
    );
    let cls = crate::space::unwrap_cell(args[0]);
    // cls should be a W_TypeObject — create instance of it
    if unsafe { is_type(cls) } {
        return Ok(w_instance_new(cls));
    }
    // Fallback: create bare instance with no type
    Ok(w_instance_new(PY_NULL))
}

/// `object.__init__(self)` — no-op base __init__.
fn object_init(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_none())
}

fn init_object_typedef(ns: &mut PyNamespace) {
    namespace_store(ns, "__new__", w_builtin_func_new("__new__", object_new));
    namespace_store(ns, "__init__", w_builtin_func_new("__init__", object_init));
    // PyPy: objectobject.py — default comparison/hash/repr for all objects
    namespace_store(
        ns,
        "__eq__",
        w_builtin_func_new("__eq__", |args| {
            Ok(pyre_object::w_bool_from(
                args.len() >= 2 && std::ptr::eq(args[0], args[1]),
            ))
        }),
    );
    namespace_store(
        ns,
        "__ne__",
        w_builtin_func_new("__ne__", |args| {
            Ok(pyre_object::w_bool_from(
                args.len() >= 2 && !std::ptr::eq(args[0], args[1]),
            ))
        }),
    );
    namespace_store(
        ns,
        "__hash__",
        w_builtin_func_new("__hash__", |args| {
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
        w_builtin_func_new("__repr__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new("<object>"));
            }
            Ok(pyre_object::w_str_new(&unsafe { crate::py_repr(args[0]) }))
        }),
    );
    namespace_store(
        ns,
        "__str__",
        w_builtin_func_new("__str__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_str_new("<object>"));
            }
            Ok(pyre_object::w_str_new(&unsafe { crate::py_str(args[0]) }))
        }),
    );
    namespace_store(
        ns,
        "__init_subclass__",
        w_builtin_func_new("__init_subclass__", |_| Ok(pyre_object::w_none())),
    );
    namespace_store(
        ns,
        "__subclasshook__",
        w_builtin_func_new("__subclasshook__", |_| Ok(pyre_object::w_not_implemented())),
    );
}
