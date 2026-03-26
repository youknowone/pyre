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
    let mut reg: HashMap<usize, usize> = HashMap::new();

    reg.insert(
        &LIST_TYPE as *const PyType as usize,
        make_type("list", &LIST_TYPE, init_list_typedef) as usize,
    );
    reg.insert(
        &STR_TYPE as *const PyType as usize,
        make_type("str", &STR_TYPE, init_str_typedef) as usize,
    );
    reg.insert(
        &DICT_TYPE as *const PyType as usize,
        make_type("dict", &DICT_TYPE, init_dict_typedef) as usize,
    );
    reg.insert(
        &TUPLE_TYPE as *const PyType as usize,
        make_type("tuple", &TUPLE_TYPE, init_tuple_typedef) as usize,
    );
    reg.insert(
        &INT_TYPE as *const PyType as usize,
        make_type("int", &INT_TYPE, init_int_typedef) as usize,
    );
    reg.insert(
        &FLOAT_TYPE as *const PyType as usize,
        make_type("float", &FLOAT_TYPE, init_float_typedef) as usize,
    );
    reg.insert(
        &BOOL_TYPE as *const PyType as usize,
        make_type("bool", &BOOL_TYPE, init_bool_typedef) as usize,
    );
    reg.insert(
        &NONE_TYPE as *const PyType as usize,
        make_type("NoneType", &NONE_TYPE, |_| {}) as usize,
    );

    // 'object' type — PyPy: objectobject.py W_ObjectObject.typedef
    // Registered under INSTANCE_TYPE so that py_getattr on instances
    // can also find __new__ via the type registry.
    let object_type = make_type("object", &INSTANCE_TYPE, init_object_typedef);
    reg.insert(
        &INSTANCE_TYPE as *const PyType as usize,
        object_type as usize,
    );
    // Store the object type for builtins to reference
    let _ = OBJECT_TYPE_OBJ.set(object_type as usize);

    let _ = TYPE_REGISTRY.set(reg);
}

/// The global `object` type object, accessible from builtins.
static OBJECT_TYPE_OBJ: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Retrieve a W_TypeObject for a builtin type by its PyType pointer.
/// Used to register `int`, `str`, etc. as types in builtins (not functions).
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

fn make_type(name: &str, _tp: &PyType, init: fn(&mut PyNamespace)) -> PyObjectRef {
    let mut ns = Box::new(PyNamespace::new());
    ns.fix_ptr();
    init(&mut ns);
    let ns_ptr = Box::into_raw(ns);
    let type_obj = w_type_new(name, PY_NULL, ns_ptr as *mut u8);
    // Cache MRO (just self, no bases for builtin types)
    unsafe { w_type_set_mro(type_obj, vec![type_obj]) };
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
}

// ── Int/Float/Bool TypeDef (minimal) ─────────────────────────────────

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
}
