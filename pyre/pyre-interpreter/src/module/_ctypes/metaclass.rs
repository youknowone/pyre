//! ctypes metaclasses, `CField` descriptors, and the `Structure`/`Union` bases.
//!
//! `PyCSimpleType` / `PyCStructType` / `PyCUnionType` / `PyCArrayType` /
//! `PyCPointerType` are the metaclasses of `_SimpleCData` / `Structure` /
//! `Union` / `Array` / `_Pointer`.  Their `__new__` builds the class and
//! computes its [`super::stginfo::StgInfoData`] (simple validation, the
//! struct/union layout in [`process_fields`], or the array/pointer element
//! metadata).  `CField` is the per-field data descriptor installed into a class
//! dict; its `__get__`/`__set__` read and write the instance buffer at the
//! field offset, aliasing nested aggregates as sub-views
//! (`super::cdata::make_subview`).
//!
//! `ctype * n` builds a cached `Array` subtype (`array_type_from_ctype`);
//! `POINTER(T)` builds a cached `_Pointer` subtype, memoised on `T`'s
//! `StgInfo.pointer_type` and read back through the `__pointer_type__` getset.

use super::cdata;
use super::stginfo::{self, StgInfoData};
use super::type_ns_store;
use pyre_object::PyObjectRef;
use rustpython_host_env::ctypes as host_ctypes;
use std::sync::OnceLock;

type PyResult = Result<PyObjectRef, crate::PyError>;

// ── cached type objects ────────────────────────────────────────────────

macro_rules! cached_type {
    ($cell:ident, $f:ident, $build:expr) => {
        static $cell: OnceLock<usize> = OnceLock::new();
        pub(super) fn $f() -> PyObjectRef {
            *$cell.get_or_init(|| $build() as usize) as PyObjectRef
        }
    };
}

cached_type!(PYCSIMPLETYPE, pycsimpletype_type, || {
    crate::typedef::make_builtin_type_with_base(
        "PyCSimpleType",
        |ns| {
            install_new(ns, csimpletype_new);
            install_init(ns, csimpletype_init);
            install_shared_meta(ns);
        },
        crate::typedef::w_type(),
    )
});

cached_type!(PYCSTRUCTTYPE, pycstructtype_type, || {
    crate::typedef::make_builtin_type_with_base(
        "PyCStructType",
        |ns| {
            install_new(ns, cstructtype_new);
            install_init(ns, cstructtype_init);
            install_shared_meta(ns);
            install_fields_getset(ns);
        },
        crate::typedef::w_type(),
    )
});

cached_type!(PYCUNIONTYPE, pycuniontype_type, || {
    // The union metaclass's Python-visible name is `UnionType` (matching
    // `_ctypes.UnionType`), though its Rust identifier is PyCUnionType.
    crate::typedef::make_builtin_type_with_base(
        "UnionType",
        |ns| {
            install_new(ns, cuniontype_new);
            install_init(ns, cuniontype_init);
            install_shared_meta(ns);
            install_fields_getset(ns);
        },
        crate::typedef::w_type(),
    )
});

cached_type!(STRUCTURE, structure_type, || {
    let tp = crate::typedef::make_builtin_type_with_base(
        "Structure",
        init_aggregate_base,
        cdata::cdata_type(),
    );
    finish_aggregate_base(tp, pycstructtype_type(), "struct");
    tp
});

cached_type!(UNION, union_type, || {
    let tp = crate::typedef::make_builtin_type_with_base(
        "Union",
        init_aggregate_base,
        cdata::cdata_type(),
    );
    finish_aggregate_base(tp, pycuniontype_type(), "union");
    tp
});

cached_type!(CFIELD, cfield_type, || {
    let tp = crate::typedef::make_builtin_type("CField", |ns| {
        type_ns_store(
            ns,
            "__new__",
            crate::make_builtin_function("__new__", cfield_new_internal),
        );
        type_ns_store(
            ns,
            "__get__",
            crate::make_builtin_function("__get__", cfield_get),
        );
        type_ns_store(
            ns,
            "__set__",
            crate::make_builtin_function("__set__", cfield_set),
        );
        type_ns_store(
            ns,
            "__delete__",
            crate::make_builtin_function("__delete__", |_args| {
                Err(crate::PyError::type_error("can't delete attribute"))
            }),
        );
        type_ns_store(
            ns,
            "__repr__",
            crate::make_builtin_function("__repr__", cfield_repr),
        );
        type_ns_store(
            ns,
            "__setattr__",
            crate::make_builtin_function("__setattr__", |_args| {
                Err(crate::PyError::attribute_error(
                    "CField attributes are read-only",
                ))
            }),
        );
    });
    unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
    tp
});

cached_type!(PYCARRAYTYPE, pycarraytype_type, || {
    crate::typedef::make_builtin_type_with_base(
        "PyCArrayType",
        |ns| {
            install_new(ns, carraytype_new);
            install_init(ns, carraytype_init);
            install_shared_meta(ns);
        },
        crate::typedef::w_type(),
    )
});

cached_type!(PYCPOINTERTYPE, pycpointertype_type, || {
    crate::typedef::make_builtin_type_with_base(
        "PyCPointerType",
        |ns| {
            install_new(ns, cpointertype_new);
            install_init(ns, cpointertype_init);
            install_shared_meta(ns);
        },
        crate::typedef::w_type(),
    )
});

cached_type!(ARRAY, array_type, || {
    let tp =
        crate::typedef::make_builtin_type_with_base("Array", init_array_base, cdata::cdata_type());
    finish_element_base(tp, pycarraytype_type());
    tp
});

cached_type!(POINTER_BASE, pointer_base_type, || {
    let tp = crate::typedef::make_builtin_type_with_base(
        "_Pointer",
        init_pointer_base,
        cdata::cdata_type(),
    );
    finish_element_base(tp, pycpointertype_type());
    tp
});

fn install_new(ns: PyObjectRef, f: crate::gateway::BuiltinCodeFn) {
    type_ns_store(ns, "__new__", crate::make_builtin_function("__new__", f));
}

fn install_init(ns: PyObjectRef, f: crate::gateway::BuiltinCodeFn) {
    type_ns_store(ns, "__init__", crate::make_builtin_function("__init__", f));
}

/// `__mul__`, `__pointer_type__`, and `from_param` — shared by all metaclasses.
fn install_shared_meta(ns: PyObjectRef) {
    type_ns_store(
        ns,
        "from_address",
        crate::make_builtin_function("from_address", cdata::cdata_from_address),
    );
    type_ns_store(
        ns,
        "in_dll",
        crate::make_builtin_function("in_dll", cdata::cdata_in_dll),
    );
    type_ns_store(
        ns,
        "__mul__",
        crate::make_builtin_function("__mul__", meta_mul),
    );
    type_ns_store(
        ns,
        "__pointer_type__",
        crate::typedef::make_getset_property_named(
            crate::make_builtin_function_with_arity("__pointer_type__", pointer_type_get, 2),
            crate::make_builtin_function_with_arity("__pointer_type__", pointer_type_set, 3),
            pyre_object::PY_NULL,
            "__pointer_type__",
        ),
    );
    type_ns_store(
        ns,
        "from_param",
        crate::make_builtin_function("from_param", meta_from_param),
    );
}

fn install_fields_getset(ns: PyObjectRef) {
    type_ns_store(
        ns,
        "_fields_",
        crate::typedef::make_getset_property_named(
            crate::make_builtin_function_with_arity("_fields_", fields_get, 2),
            crate::make_builtin_function_with_arity("_fields_", fields_set, 3),
            pyre_object::PY_NULL,
            "_fields_",
        ),
    );
}

fn init_aggregate_base(ns: PyObjectRef) {
    type_ns_store(
        ns,
        "__new__",
        crate::make_builtin_function("__new__", structure_new),
    );
    type_ns_store(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", structure_init),
    );
    type_ns_store(
        ns,
        "__setattr__",
        crate::make_builtin_function("__setattr__", structure_setattr),
    );
}

fn structure_setattr(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 3 || !unsafe { pyre_object::is_str(args[1]) } {
        return Err(crate::PyError::type_error(
            "attribute name must be a string",
        ));
    }
    let obj = args[0];
    let name = unsafe { pyre_object::w_str_get_value(args[1]) };
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let slots_empty = crate::type_dict_lookup(cls, "__slots__").is_some_and(|slots| {
        (unsafe { pyre_object::is_tuple(slots) } && unsafe { pyre_object::w_tuple_len(slots) } == 0)
            || (unsafe { pyre_object::is_list(slots) }
                && unsafe { pyre_object::w_list_len(slots) } == 0)
    });
    if slots_empty && unsafe { crate::baseobjspace::lookup_in_type(cls, name) }.is_none() {
        return Err(crate::PyError::attribute_error(format!(
            "'{}' object has no attribute '{}'",
            type_name(cls),
            name,
        )));
    }
    crate::baseobjspace::object_setattr(obj, name, args[2])
}

fn finish_aggregate_base(tp: PyObjectRef, metaclass: PyObjectRef, paramfunc: &'static str) {
    unsafe {
        pyre_object::typeobject::w_type_set_hasdict(tp, true);
        (*tp).w_class = metaclass;
    }
    // Bare Structure/Union are abstract; the first real subtype gets the
    // default zero-sized storage state from its metaclass __init__.
    let _ = paramfunc;
}

/// Finish the `Array` / `_Pointer` base: hasdict, acceptable-as-base, stamp its
/// metaclass.  Unlike the aggregate bases these get **no** default `StgInfo`, so
/// a bare `Array()` / `_Pointer()` is abstract until a subtype supplies
/// `_type_` (`POINTER(T)`) or `_type_`+`_length_` (`T * n`).
fn finish_element_base(tp: PyObjectRef, metaclass: PyObjectRef) {
    unsafe {
        pyre_object::typeobject::w_type_set_hasdict(tp, true);
        pyre_object::typeobject::w_type_set_acceptable_as_base_class(tp, true);
        (*tp).w_class = metaclass;
    }
}

fn init_array_base(ns: PyObjectRef) {
    install_new(ns, array_new);
    type_ns_store(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", array_init),
    );
    type_ns_store(
        ns,
        "__len__",
        crate::make_builtin_function("__len__", array_len),
    );
    type_ns_store(
        ns,
        "__getitem__",
        crate::make_builtin_function("__getitem__", array_getitem),
    );
    type_ns_store(
        ns,
        "__setitem__",
        crate::make_builtin_function("__setitem__", array_setitem),
    );
}

fn init_pointer_base(ns: PyObjectRef) {
    install_new(ns, pointer_new);
    type_ns_store(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", pointer_init),
    );
    type_ns_store(
        ns,
        "__getitem__",
        crate::make_builtin_function("__getitem__", pointer_getitem),
    );
    type_ns_store(
        ns,
        "__setitem__",
        crate::make_builtin_function("__setitem__", pointer_setitem),
    );
    type_ns_store(
        ns,
        "__bool__",
        crate::make_builtin_function("__bool__", pointer_bool),
    );
    type_ns_store(
        ns,
        "contents",
        crate::typedef::make_getset_property_named(
            crate::make_builtin_function_with_arity("contents", contents_get, 2),
            crate::make_builtin_function_with_arity("contents", contents_set, 3),
            pyre_object::PY_NULL,
            "contents",
        ),
    );
    type_ns_store(
        ns,
        "set_type",
        pyre_object::function::w_classmethod_new(crate::make_builtin_function(
            "set_type",
            pointer_set_type,
        )),
    );
}

fn pointer_set_type(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 2 || !unsafe { pyre_object::is_type(args[0]) && pyre_object::is_type(args[1]) }
    {
        return Err(crate::PyError::type_error(
            "set_type() requires a ctypes type",
        ));
    }
    let pointer_cls = args[0];
    let proto = args[1];
    let mut data = StgInfoData::new(
        host_ctypes::pointer_size(),
        host_ctypes::simple_type_align("P").unwrap_or(host_ctypes::pointer_size()),
        "pointer",
    );
    data.element_size = stginfo::field_size_of(proto).unwrap_or(0);
    data.length = 1;
    data.proto = Some(proto);
    data.flags |= stginfo::TYPEFLAG_ISPOINTER;
    stginfo::stginfo_set(pointer_cls, stginfo::stginfo_new(data));
    set_type_attr(pointer_cls, "_type_", proto);
    if let Some(info) = stginfo::stginfo_of(proto) {
        stginfo::stginfo_set_pointer_type(info, pointer_cls);
    }
    Ok(pyre_object::w_none())
}

// ── metaclass `__new__` ────────────────────────────────────────────────

fn csimpletype_new(args: &[PyObjectRef]) -> PyResult {
    crate::builtins::type_descr_new(args)
}

fn cstructtype_new(args: &[PyObjectRef]) -> PyResult {
    crate::builtins::type_descr_new(args)
}

fn cuniontype_new(args: &[PyObjectRef]) -> PyResult {
    crate::builtins::type_descr_new(args)
}

const META_INITIALIZED_KEY: &str = "_ctypes_stginfo_initialized";

fn metaclass_init(args: &[PyObjectRef], initialize: fn(PyObjectRef) -> PyResult) -> PyResult {
    if args.len() < 4 {
        return Err(crate::PyError::type_error(
            "metaclass __init__ expects name, bases and namespace",
        ));
    }
    let cls = args
        .first()
        .copied()
        .filter(|&obj| unsafe { pyre_object::is_type(obj) })
        .ok_or_else(|| crate::PyError::type_error("metaclass __init__ requires a type"))?;
    if own_dict_get(cls, META_INITIALIZED_KEY).is_some() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "already initialized",
        ));
    }
    initialize(cls)?;
    set_type_attr(cls, META_INITIALIZED_KEY, pyre_object::w_bool_from(true));
    Ok(pyre_object::w_none())
}

fn csimpletype_init(args: &[PyObjectRef]) -> PyResult {
    metaclass_init(args, simple_init_stginfo)
}

fn cstructtype_init(args: &[PyObjectRef]) -> PyResult {
    metaclass_init(args, |cls| struct_union_init_stginfo(cls, false))
}

fn cuniontype_init(args: &[PyObjectRef]) -> PyResult {
    metaclass_init(args, |cls| struct_union_init_stginfo(cls, true))
}

/// `PyCSimpleType` layout: validate `_type_` and build a simple `StgInfo`.
fn simple_init_stginfo(cls: PyObjectRef) -> PyResult {
    let tc = match cdata::type_code_of(cls) {
        Some(tc) => tc,
        // No `_type_` at all: `_SimpleCData` itself and abstract intermediates.
        None => return Ok(pyre_object::w_none()),
    };
    if tc.chars().count() != 1 || !host_ctypes::simple_type_chars().contains(tc.as_str()) {
        return Err(cdata::invalid_type_code_error());
    }
    let size = host_ctypes::simple_type_size(&tc).ok_or_else(cdata::invalid_type_code_error)?;
    let align = host_ctypes::simple_type_align(&tc).ok_or_else(cdata::invalid_type_code_error)?;
    let mut data = StgInfoData::new(size, align, "simple");
    data.format = Some(tc.clone());
    if host_ctypes::simple_type_is_pointer(&tc) {
        data.flags |= stginfo::TYPEFLAG_ISPOINTER;
    }
    stginfo::stginfo_set(cls, stginfo::stginfo_new(data));

    let endian_capable = matches!(
        tc.as_str(),
        "b" | "B" | "h" | "H" | "i" | "I" | "l" | "L" | "q" | "Q" | "f" | "d" | "g"
    );
    if endian_capable && size == 1 {
        set_type_attr(cls, "__ctype_le__", cls);
        set_type_attr(cls, "__ctype_be__", cls);
    } else if endian_capable && own_dict_get(cls, "_ctypes_native_peer").is_none() {
        let ns = pyre_object::w_dict_new();
        unsafe {
            pyre_object::w_dict_setitem_str(ns, "_type_", pyre_object::w_str_new(&tc));
            pyre_object::w_dict_setitem_str(ns, "_swappedbytes_", pyre_object::w_bool_from(true));
            pyre_object::w_dict_setitem_str(ns, "_ctypes_native_peer", cls);
        }
        let bases = unsafe { pyre_object::typeobject::w_type_get_bases(cls) };
        let name = unsafe { pyre_object::typeobject::w_type_get_name(cls) };
        let swapped = crate::call::type_call_instantiate(
            pycsimpletype_type(),
            &[pyre_object::w_str_new(name), bases, ns],
        )?;
        if cfg!(target_endian = "little") {
            set_type_attr(cls, "__ctype_le__", cls);
            set_type_attr(cls, "__ctype_be__", swapped);
            set_type_attr(swapped, "__ctype_le__", cls);
            set_type_attr(swapped, "__ctype_be__", swapped);
        } else {
            set_type_attr(cls, "__ctype_le__", swapped);
            set_type_attr(cls, "__ctype_be__", cls);
            set_type_attr(swapped, "__ctype_le__", swapped);
            set_type_attr(swapped, "__ctype_be__", cls);
        }
    }
    Ok(pyre_object::w_none())
}

// ── struct/union layout ────────────────────────────────────────────────

/// MRO items of `cls`; empty while the type is still being initialised and
/// carries no mro yet.
fn mro_items<'a>(cls: PyObjectRef) -> &'a [PyObjectRef] {
    let mro = unsafe { pyre_object::typeobject::w_type_get_mro(cls) };
    if mro.is_null() {
        return &[];
    }
    unsafe { (*mro).as_slice() }
}

/// First base (in MRO order after `cls`) that carries a `StgInfo`.
fn first_base_stginfo(cls: PyObjectRef) -> Option<PyObjectRef> {
    mro_items(cls)
        .iter()
        .skip(1)
        .find_map(|&t| stginfo::stginfo_of(t))
}

fn usize_attr(cls: PyObjectRef, name: &str, default: usize) -> usize {
    match unsafe { crate::baseobjspace::lookup_in_type(cls, name) } {
        Some(o) if unsafe { pyre_object::is_int(o) } => {
            (unsafe { pyre_object::w_int_get_value(o) }).max(0) as usize
        }
        _ => default,
    }
}

fn align_attr(cls: PyObjectRef) -> Result<usize, crate::PyError> {
    match unsafe { crate::baseobjspace::lookup_in_type(cls, "_align_") } {
        Some(o) if unsafe { pyre_object::is_int(o) } => {
            let value = unsafe { pyre_object::w_int_get_value(o) };
            if value < 0 {
                Err(crate::PyError::value_error(
                    "_align_ must be a non-negative integer",
                ))
            } else {
                Ok((value as usize).max(1))
            }
        }
        Some(_) => Err(crate::PyError::value_error(
            "_align_ must be a non-negative integer",
        )),
        None => Ok(1),
    }
}

fn anonymous_names(cls: PyObjectRef) -> Result<Vec<String>, crate::PyError> {
    let Some(value) = crate::type_dict_lookup(cls, "_anonymous_") else {
        return Ok(Vec::new());
    };
    let items = seq_items(value)
        .ok_or_else(|| crate::PyError::type_error("_anonymous_ must be a sequence"))?;
    items
        .into_iter()
        .map(|item| {
            if unsafe { pyre_object::is_str(item) } {
                Ok(unsafe { pyre_object::w_str_get_value(item) }.to_string())
            } else {
                Err(crate::PyError::type_error(
                    "_anonymous_ items must be strings",
                ))
            }
        })
        .collect()
}

/// Install the flattened descriptors contributed by an anonymous aggregate.
/// Anonymous carrier fields are skipped while their contents are recursively
/// promoted, exactly as `PyCStructUnionType_update_stginfo` does.
fn promote_anonymous_fields(
    cls: PyObjectRef,
    proto: PyObjectRef,
    base_offset: usize,
) -> Result<(), crate::PyError> {
    let anonymous = anonymous_names(proto)?;
    let fields = unsafe { crate::baseobjspace::lookup_in_type(proto, "_fields_") }
        .ok_or_else(|| crate::PyError::attribute_error("anonymous field has no _fields_"))?;
    for entry in field_entries(fields)? {
        let name = entry.name;
        let child_proto = entry.ty;
        let child =
            unsafe { crate::baseobjspace::lookup_in_type(proto, &name) }.ok_or_else(|| {
                crate::PyError::attribute_error(format!("type has no attribute '{name}'"))
            })?;
        let offset = base_offset + cf_usize(child, "offset");
        if anonymous.iter().any(|anon| anon == &name) {
            promote_anonymous_fields(cls, child_proto, offset)?;
        } else {
            let promoted = cfield_new(
                &name,
                child_proto,
                offset,
                cf_usize(child, "byte_size"),
                cf_usize(child, "index"),
            );
            let d = crate::baseobjspace::getdict(promoted);
            unsafe {
                for key in ["size", "bit_size", "bit_offset", "is_bitfield"] {
                    if let Some(value) =
                        pyre_object::w_dict_getitem_str(crate::baseobjspace::getdict(child), key)
                    {
                        pyre_object::w_dict_setitem_str(d, key, value);
                    }
                }
            }
            set_type_attr(cls, &name, promoted);
        }
    }
    Ok(())
}

struct FieldEntry {
    name: String,
    ty: PyObjectRef,
    bits: Option<usize>,
}

/// Parse `_fields_` into the same `(name, ctype[, bits])` records consumed by
/// CPython/PyPy's aggregate layout builder.
fn field_entries(fields: PyObjectRef) -> Result<Vec<FieldEntry>, crate::PyError> {
    let items = seq_items(fields)
        .ok_or_else(|| crate::PyError::type_error("_fields_ must be a sequence of 2-tuples"))?;
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        if !unsafe { pyre_object::is_tuple(it) } {
            return Err(crate::PyError::type_error(
                "_fields_ entries must be tuples",
            ));
        }
        let n = unsafe { pyre_object::w_tuple_len(it) };
        if n < 2 {
            return Err(crate::PyError::type_error(
                "_fields_ entries must be (name, type) pairs",
            ));
        }
        if n > 3 {
            return Err(crate::PyError::type_error(
                "_fields_ entries must be (name, type) or (name, type, bits)",
            ));
        }
        let name = unsafe { pyre_object::w_tuple_getitem(it, 0) }.unwrap_or(pyre_object::PY_NULL);
        let ty = unsafe { pyre_object::w_tuple_getitem(it, 1) }.unwrap_or(pyre_object::PY_NULL);
        if name.is_null() || !unsafe { pyre_object::is_str(name) } {
            return Err(crate::PyError::type_error("field name must be a string"));
        }
        if ty.is_null() || !unsafe { pyre_object::is_type(ty) } {
            return Err(crate::PyError::type_error(
                "field type must be a ctypes type",
            ));
        }
        let name = unsafe { pyre_object::w_str_get_value(name) }.to_string();
        let bits = if n == 3 {
            let value =
                unsafe { pyre_object::w_tuple_getitem(it, 2) }.unwrap_or(pyre_object::PY_NULL);
            if value.is_null()
                || !unsafe { pyre_object::is_int(value) || pyre_object::is_long(value) }
            {
                return Err(crate::PyError::type_error("bit width must be an integer"));
            }
            let tc = cdata::type_code_of(ty).unwrap_or_default();
            if !matches!(
                tc.as_str(),
                "b" | "B" | "h" | "H" | "i" | "I" | "l" | "L" | "q" | "Q" | "?"
            ) {
                return Err(crate::PyError::type_error(format!(
                    "bit fields not allowed for type {}",
                    type_name(ty),
                )));
            }
            let size = stginfo::field_size_of(ty).unwrap_or(0);
            let width = crate::baseobjspace::int_w(value).unwrap_or(-1);
            if width <= 0 || width as usize > size.saturating_mul(8) {
                return Err(crate::PyError::value_error(format!(
                    "number of bits invalid for bit field '{name}'",
                )));
            }
            Some(width as usize)
        } else {
            None
        };
        out.push(FieldEntry { name, ty, bits });
    }
    Ok(out)
}

fn seq_items(obj: PyObjectRef) -> Option<Vec<PyObjectRef>> {
    if unsafe { pyre_object::is_tuple(obj) } {
        let n = unsafe { pyre_object::w_tuple_len(obj) } as i64;
        Some(
            (0..n)
                .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(obj, i) })
                .collect(),
        )
    } else if unsafe { pyre_object::is_list(obj) } {
        let n = unsafe { pyre_object::w_list_len(obj) } as i64;
        Some(
            (0..n)
                .filter_map(|i| unsafe { pyre_object::w_list_getitem(obj, i) })
                .collect(),
        )
    } else {
        None
    }
}

/// Mark a field type's `StgInfo` FINAL (creating a minimal one if absent), so
/// it cannot later gain `_fields_`.
fn mark_type_final(ty: PyObjectRef, size: usize, align: usize) {
    match stginfo::stginfo_of(ty) {
        Some(info) => stginfo::stginfo_mark_final(info),
        None => {
            let mut data = StgInfoData::new(size, align, "simple");
            data.flags |= stginfo::DICTFLAG_FINAL;
            stginfo::stginfo_set(ty, stginfo::stginfo_new(data));
        }
    }
}

/// Compute the layout for a struct (`is_union=false`) or union, installing the
/// `CField` descriptors and the class `StgInfo`.  Port of `process_fields`.
fn struct_union_init_stginfo(cls: PyObjectRef, is_union: bool) -> PyResult {
    // `_fields_` directly in the new class dict → process it; else clone the
    // first base's StgInfo (or a default).
    let own_fields =
        crate::type_dict_lookup(cls, "_fields_").filter(|&f| !unsafe { pyre_object::is_none(f) });
    match own_fields {
        Some(fields) => process_fields(cls, fields, is_union),
        None => {
            let paramfunc = if is_union { "union" } else { "struct" };
            match first_base_stginfo(cls) {
                Some(base_info) => {
                    let mut data = StgInfoData::new(
                        stginfo::stginfo_size(base_info),
                        stginfo::stginfo_align(base_info),
                        paramfunc,
                    );
                    data.length = stginfo::stginfo_length(base_info);
                    // Cleared FINAL / pointer_type on the clone; mark base FINAL.
                    stginfo::stginfo_set(cls, stginfo::stginfo_new(data));
                    stginfo::stginfo_mark_final(base_info);
                }
                None => stginfo::stginfo_set(
                    cls,
                    stginfo::stginfo_new(StgInfoData::new(0, 1, paramfunc)),
                ),
            }
            Ok(pyre_object::w_none())
        }
    }
}

fn process_fields(cls: PyObjectRef, fields: PyObjectRef, is_union: bool) -> PyResult {
    // `_ctypes.cfield.process_fields` replaces the layout of an incomplete
    // aggregate in place.  Keep the pointer type already cached on the old
    // StgInfo: POINTER(Complete) must remain the type made while Complete was
    // incomplete (including its frozen PEP 3118 "&B" format).
    let cached_pointer_type = stginfo::stginfo_of(cls).and_then(stginfo::stginfo_pointer_type);
    let entries = field_entries(fields)?;
    let anonymous = anonymous_names(cls)?;

    for name in &anonymous {
        let Some(entry) = entries.iter().find(|entry| &entry.name == name) else {
            return Err(crate::PyError::attribute_error(format!(
                "'{name}' is specified in _anonymous_ but not in _fields_"
            )));
        };
        if !matches!(proto_kind(entry.ty).as_str(), "struct" | "union") {
            return Err(crate::PyError::type_error(
                "anonymous field must be a structure or union",
            ));
        }
    }

    let is_swapped =
        unsafe { crate::baseobjspace::lookup_in_type(cls, "_swappedbytes_") }.is_some();
    let pack = usize_attr(cls, "_pack_", 0);
    let forced = align_attr(cls)?;
    if pack > 0
        && !cfg!(windows)
        && unsafe { crate::baseobjspace::lookup_in_type(cls, "_layout_") }.is_none()
    {
        crate::warn::warn_deprecation(
            "_pack_ without explicit _layout_ uses deprecated MSVC layout",
        );
    }

    // Reject before mutating the class: a type already frozen (used as a field
    // elsewhere) or one that would embed itself by value cannot define fields.
    // Checking up front keeps a rejected `_fields_` from leaving half-installed
    // descriptors and a final flag behind.
    if stginfo::stginfo_of(cls).is_some_and(stginfo::stginfo_is_final)
        || entries.iter().any(|entry| entry.ty == cls)
    {
        return Err(crate::PyError::attribute_error(
            "Structure or union cannot contain itself",
        ));
    }

    let (base_size, base_length, mut max_align) = match first_base_stginfo(cls) {
        Some(bi) => (
            stginfo::stginfo_size(bi),
            stginfo::stginfo_length(bi),
            stginfo::stginfo_align(bi).max(forced),
        ),
        None => (0usize, 0usize, forced),
    };
    // A union inherits its base's footprint as a floor, so a derived union with
    // only smaller fields cannot shrink below the base; a struct starts laying
    // its own fields out after the inherited prefix.
    let mut offset = base_size;
    let mut union_max = base_size;
    let mut has_pointer = false;

    // Pass 1 — compute the full layout without touching the class.
    // Active allocation unit for consecutive bitfields.  MS layout (selected
    // implicitly by `_pack_`) only coalesces identical declared types; this is
    // the behavior exercised by CPython's `test_pack_layout_switch`.
    let mut active_bits: Option<(PyObjectRef, usize, usize, usize)> = None;
    let little_bits = is_swapped ^ cfg!(target_endian = "little");
    let mut pending: Vec<(usize, usize, usize, Option<(usize, usize)>)> =
        Vec::with_capacity(entries.len());
    for entry in &entries {
        let name = &entry.name;
        let ftype = entry.ty;
        if is_swapped
            && (matches!(proto_kind(ftype).as_str(), "pointer")
                || cdata::type_code_of(ftype)
                    .is_some_and(|code| matches!(code.as_str(), "u" | "P" | "z" | "Z" | "O")))
        {
            return Err(crate::PyError::type_error(format!(
                "This type does not support other endian: {name}",
            )));
        }
        let size = stginfo::field_size_of(ftype)
            .ok_or_else(|| crate::PyError::type_error(format!("field '{name}' has no size")))?;
        let align = stginfo::field_align_of(ftype).unwrap_or(1).max(1);
        let eff = if pack > 0 { pack.min(align) } else { align };
        max_align = max_align.max(eff);

        if let Some(fi) = stginfo::stginfo_of(ftype) {
            if stginfo::stginfo_flags(fi)
                & (stginfo::TYPEFLAG_ISPOINTER | stginfo::TYPEFLAG_HASPOINTER)
                != 0
            {
                has_pointer = true;
            }
        }

        if let Some(bits) = entry.bits {
            if is_union {
                let bit_offset = if little_bits { 0 } else { size * 8 - bits };
                pending.push((0, size, align, Some((bits, bit_offset))));
                union_max = union_max.max(size);
                continue;
            }
            let reuse = active_bits.filter(|(active_ty, active_size, _, used)| {
                *active_ty == ftype && *active_size == size && *used + bits <= size * 8
            });
            let (field_offset, used) = if let Some((_, _, at, used)) = reuse {
                (at, used)
            } else {
                if eff > 0 && offset % eff != 0 {
                    offset += eff - (offset % eff);
                }
                let at = offset;
                offset += size;
                (at, 0)
            };
            let bit_offset = if little_bits {
                used
            } else {
                size * 8 - used - bits
            };
            active_bits = Some((ftype, size, field_offset, used + bits));
            pending.push((field_offset, size, align, Some((bits, bit_offset))));
        } else {
            active_bits = None;
            if !is_union && eff > 0 && offset % eff != 0 {
                offset += eff - (offset % eff);
            }
            let field_offset = if is_union { 0 } else { offset };
            pending.push((field_offset, size, align, None));
            if is_union {
                union_max = union_max.max(size);
            } else {
                offset += size;
            }
        }
    }

    let total_align = max_align.max(forced);
    let raw = if is_union { union_max } else { offset };
    let aligned = if total_align > 0 {
        raw.div_ceil(total_align) * total_align
    } else {
        raw
    };

    // Pass 2 — commit: freeze field types, install `CField` descriptors, then
    // the class `StgInfo` and `_fields_`.
    for (index, entry) in entries.iter().enumerate() {
        let (field_offset, size, align, bitfield) = pending[index];
        mark_type_final(entry.ty, size, align);
        let cf = cfield_new(&entry.name, entry.ty, field_offset, size, index);
        let d = crate::baseobjspace::getdict(cf);
        if let Some((bits, bit_offset)) = bitfield {
            unsafe {
                pyre_object::w_dict_setitem_str(
                    d,
                    "size",
                    pyre_object::w_int_new(((bits << 16) | bit_offset) as i64),
                );
                pyre_object::w_dict_setitem_str(d, "bit_size", pyre_object::w_int_new(bits as i64));
                pyre_object::w_dict_setitem_str(
                    d,
                    "bit_offset",
                    pyre_object::w_int_new(bit_offset as i64),
                );
                pyre_object::w_dict_setitem_str(d, "is_bitfield", pyre_object::w_bool_from(true));
            }
        }
        if anonymous
            .iter()
            .any(|anonymous_name| anonymous_name == &entry.name)
        {
            unsafe {
                pyre_object::w_dict_setitem_str(d, "is_anonymous", pyre_object::w_bool_from(true));
            }
        }
        set_type_attr(cls, &entry.name, cf);
    }
    for name in &anonymous {
        let (index, entry) = entries
            .iter()
            .enumerate()
            .find(|(_, entry)| &entry.name == name)
            .expect("anonymous fields validated above");
        promote_anonymous_fields(cls, entry.ty, pending[index].0)?;
    }

    let mut flags = stginfo::DICTFLAG_FINAL;
    if has_pointer {
        flags |= stginfo::TYPEFLAG_HASPOINTER;
    }
    if is_union {
        flags |= stginfo::TYPEFLAG_HASUNION;
    }
    let mut data = StgInfoData::new(
        aligned,
        total_align,
        if is_union { "union" } else { "struct" },
    );
    // Field count includes the inherited prefix, not just the own fields.
    data.length = base_length + entries.len();
    data.flags = flags;
    data.big_endian = is_swapped ^ cfg!(target_endian = "big");
    let new_info = stginfo::stginfo_new(data);
    if let Some(pointer_type) = cached_pointer_type {
        stginfo::stginfo_set_pointer_type(new_info, pointer_type);
    }
    stginfo::stginfo_set(cls, new_info);

    // Store the raw `_fields_` so the metaclass getset can return it.
    set_type_attr(cls, "_fields_", fields);
    Ok(pyre_object::w_none())
}

// ── metaclass namespace methods ────────────────────────────────────────

/// `ctype * n` — build (and cache) the `n`-element array type of `ctype`.
fn meta_mul(args: &[PyObjectRef]) -> PyResult {
    let cls = args[0];
    let count = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    if count.is_null() || !unsafe { pyre_object::is_int(count) || pyre_object::is_long(count) } {
        return Err(crate::PyError::not_implemented(
            "array length must be an int",
        ));
    }
    if unsafe { pyre_object::is_long(count) } {
        let big = unsafe { pyre_object::longobject::w_long_get_value(count) };
        if big.sign() == malachite_bigint::Sign::Minus {
            return Err(crate::PyError::value_error(
                "array length must not be negative",
            ));
        }
        return Err(crate::PyError::overflow_error("array too large"));
    }
    let n = unsafe { pyre_object::w_int_get_value(count) };
    if n < 0 {
        return Err(crate::PyError::value_error(format!(
            "Array length must be >= 0, not {n}"
        )));
    }
    array_type_from_ctype(cls, n as usize)
}

fn meta_from_param(args: &[PyObjectRef]) -> PyResult {
    if args
        .first()
        .copied()
        .is_some_and(cdata::is_simplecdata_type)
    {
        return cdata::simple_from_param(args);
    }
    Ok(args.get(1).copied().unwrap_or_else(pyre_object::w_none))
}

fn pointer_type_get(args: &[PyObjectRef]) -> PyResult {
    let cls = args[1];
    if let Some(info) = stginfo::stginfo_of(cls) {
        if let Some(pt) = stginfo::stginfo_pointer_type(info) {
            return Ok(pt);
        }
    }
    Err(crate::PyError::attribute_error(
        "type has no attribute '__pointer_type__'",
    ))
}

fn pointer_type_set(args: &[PyObjectRef]) -> PyResult {
    let cls = args[1];
    let value = args[2];
    match stginfo::stginfo_of(cls) {
        Some(info) => {
            stginfo::stginfo_set_pointer_type(info, value);
            Ok(pyre_object::w_none())
        }
        None => Err(crate::PyError::attribute_error(
            "cannot set '__pointer_type__'",
        )),
    }
}

fn fields_get(args: &[PyObjectRef]) -> PyResult {
    let cls = args[1];
    unsafe { crate::baseobjspace::lookup_in_type(cls, "_fields_") }
        .filter(|&f| !f.is_null())
        .ok_or_else(|| crate::PyError::attribute_error("_fields_"))
}

fn fields_set(args: &[PyObjectRef]) -> PyResult {
    let cls = args[1];
    let value = args[2];
    let Some(info) = stginfo::stginfo_of(cls) else {
        return Err(crate::PyError::type_error(
            "ctypes state is not initialized",
        ));
    };
    if stginfo::stginfo_is_final(info) {
        return Err(crate::PyError::attribute_error("_fields_ is final"));
    }
    let is_union = stginfo::stginfo_paramfunc(info) == "union";
    process_fields(cls, value, is_union)
}

// ── CField descriptor ──────────────────────────────────────────────────

fn cfield_new(
    name: &str,
    proto: PyObjectRef,
    offset: usize,
    size: usize,
    index: usize,
) -> PyObjectRef {
    let inst = pyre_object::w_instance_new(cfield_type());
    let d = crate::baseobjspace::getdict(inst);
    unsafe {
        pyre_object::w_dict_setitem_str(d, "name", pyre_object::w_str_new(name));
        pyre_object::w_dict_setitem_str(d, "proto", proto);
        pyre_object::w_dict_setitem_str(d, "type", proto);
        pyre_object::w_dict_setitem_str(d, "offset", pyre_object::w_int_new(offset as i64));
        pyre_object::w_dict_setitem_str(d, "byte_offset", pyre_object::w_int_new(offset as i64));
        pyre_object::w_dict_setitem_str(d, "size", pyre_object::w_int_new(size as i64));
        pyre_object::w_dict_setitem_str(d, "byte_size", pyre_object::w_int_new(size as i64));
        let bit_size = malachite_bigint::BigInt::from(size) * malachite_bigint::BigInt::from(8u8);
        pyre_object::w_dict_setitem_str(
            d,
            "bit_size",
            pyre_object::longobject::w_long_new(bit_size),
        );
        pyre_object::w_dict_setitem_str(d, "bit_offset", pyre_object::w_int_new(0));
        pyre_object::w_dict_setitem_str(d, "is_bitfield", pyre_object::w_bool_from(false));
        pyre_object::w_dict_setitem_str(d, "is_anonymous", pyre_object::w_bool_from(false));
        pyre_object::w_dict_setitem_str(d, "index", pyre_object::w_int_new(index as i64));
    }
    inst
}

fn cf_obj(cfield: PyObjectRef, key: &str) -> PyObjectRef {
    unsafe { pyre_object::w_dict_getitem_str(crate::baseobjspace::getdict(cfield), key) }
        .unwrap_or(pyre_object::PY_NULL)
}

fn cf_usize(cfield: PyObjectRef, key: &str) -> usize {
    let o = cf_obj(cfield, key);
    if !o.is_null() && unsafe { pyre_object::is_int(o) } {
        (unsafe { pyre_object::w_int_get_value(o) }).max(0) as usize
    } else {
        0
    }
}

fn cf_bool(cfield: PyObjectRef, key: &str) -> bool {
    let value = cf_obj(cfield, key);
    !value.is_null() && crate::baseobjspace::is_true(value).unwrap_or(false)
}

fn bytes_to_native_uint(bytes: &[u8]) -> u64 {
    if cfg!(target_endian = "little") {
        bytes
            .iter()
            .take(8)
            .enumerate()
            .fold(0u64, |value, (i, byte)| value | ((*byte as u64) << (i * 8)))
    } else {
        bytes
            .iter()
            .take(8)
            .fold(0u64, |value, byte| (value << 8) | *byte as u64)
    }
}

fn native_uint_to_bytes(value: u64, size: usize) -> Vec<u8> {
    let bytes = if cfg!(target_endian = "little") {
        value.to_le_bytes()
    } else {
        value.to_be_bytes()
    };
    if cfg!(target_endian = "little") {
        bytes[..size.min(8)].to_vec()
    } else {
        bytes[8 - size.min(8)..].to_vec()
    }
}

/// The storage kind of a field's `proto` ("simple"/"struct"/"union"/…).
fn proto_kind(proto: PyObjectRef) -> String {
    if let Some(info) = stginfo::stginfo_of(proto) {
        let pf = stginfo::stginfo_paramfunc(info);
        if !pf.is_empty() {
            return pf;
        }
    }
    if cdata::type_code_of(proto).is_some() {
        return "simple".to_string();
    }
    String::new()
}

fn field_needs_swap(obj: PyObjectRef, proto: PyObjectRef, size: usize) -> bool {
    if size <= 1 {
        return false;
    }
    let oc = unsafe { pyre_object::w_instance_get_type(obj) };
    unsafe { crate::baseobjspace::lookup_in_type(oc, "_swappedbytes_") }.is_some()
        || unsafe { crate::baseobjspace::lookup_in_type(proto, "_swappedbytes_") }.is_some()
}

fn cfield_get(args: &[PyObjectRef]) -> PyResult {
    let cfield = args[0];
    let obj = args.get(1).copied().unwrap_or_else(pyre_object::w_none);
    // Accessed on the class (`Point.x`) → return the descriptor itself.
    if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
        return Ok(cfield);
    }
    if !cdata::is_cdata_instance(obj) {
        return Err(crate::PyError::type_error("not a ctypes instance"));
    }
    let proto = cf_obj(cfield, "proto");
    let offset = cf_usize(cfield, "offset");
    let size = cf_usize(cfield, "byte_size");
    let index = cf_usize(cfield, "index");

    match proto_kind(proto).as_str() {
        "simple" => {
            let tc = cdata::type_code_of(proto)
                .ok_or_else(|| crate::PyError::type_error("field has no '_type_'"))?;
            let all = cdata::cdata_bytes(obj)
                .ok_or_else(|| crate::PyError::type_error("instance has no buffer"))?;
            let start = offset.min(all.len());
            let end = (offset + size).min(all.len());
            let mut field_bytes = all[start..end].to_vec();
            if field_needs_swap(obj, proto, size) {
                field_bytes.reverse();
            }
            if cf_bool(cfield, "is_bitfield") {
                let bits = cf_usize(cfield, "bit_size");
                let shift = cf_usize(cfield, "bit_offset");
                let mask = if bits >= 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                };
                let raw = (bytes_to_native_uint(&field_bytes) >> shift) & mask;
                if matches!(tc.as_str(), "b" | "h" | "i" | "l" | "q") {
                    let signed = if bits < 64 && raw & (1u64 << (bits - 1)) != 0 {
                        (raw | !mask) as i64
                    } else {
                        raw as i64
                    };
                    return Ok(pyre_object::w_int_new(signed));
                }
                return Ok(pyre_object::longobject::w_long_new(
                    malachite_bigint::BigInt::from(raw),
                ));
            }
            Ok(cdata::decoded_to_pyobject(host_ctypes::decode_type_code(
                &tc,
                &field_bytes,
            )))
        }
        "array" => {
            let element = stginfo::stginfo_of(proto).and_then(stginfo::stginfo_proto);
            let all = cdata::cdata_bytes(obj)
                .ok_or_else(|| crate::PyError::type_error("instance has no buffer"))?;
            let start = offset.min(all.len());
            let end = (offset + size).min(all.len());
            let field = &all[start..end];
            match element.and_then(cdata::type_code_of).as_deref() {
                Some("c") => {
                    let n = field.iter().position(|&b| b == 0).unwrap_or(field.len());
                    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&field[..n]))
                }
                Some("u") => Ok(pyre_object::w_str_new(&host_ctypes::wstring_from_bytes(
                    field,
                ))),
                _ => Ok(cdata::make_indexed_subview(proto, obj, offset, size, index)),
            }
        }
        "struct" | "union" | "pointer" => {
            Ok(cdata::make_indexed_subview(proto, obj, offset, size, index))
        }
        _ => Err(crate::PyError::type_error("field type has no storage info")),
    }
}

fn cfield_set(args: &[PyObjectRef]) -> PyResult {
    let cfield = args[0];
    let obj = args[1];
    let value = args[2];
    if !cdata::is_cdata_instance(obj) {
        return Err(crate::PyError::type_error("not a ctypes instance"));
    }
    let proto = cf_obj(cfield, "proto");
    let offset = cf_usize(cfield, "offset");
    let size = cf_usize(cfield, "byte_size");
    let index = cf_usize(cfield, "index");

    match proto_kind(proto).as_str() {
        "simple" => {
            let tc = cdata::type_code_of(proto)
                .ok_or_else(|| crate::PyError::type_error("field has no '_type_'"))?;
            if cf_bool(cfield, "is_bitfield") {
                let bits = cf_usize(cfield, "bit_size");
                let shift = cf_usize(cfield, "bit_offset");
                let mask = if bits >= 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                };
                let integer = crate::baseobjspace::int_w(value)? as u64;
                let mut storage = cdata::cdata_bytes(obj)
                    .unwrap_or(&[])
                    .get(offset..offset.saturating_add(size))
                    .unwrap_or(&[])
                    .to_vec();
                storage.resize(size, 0);
                if field_needs_swap(obj, proto, size) {
                    storage.reverse();
                }
                let old = bytes_to_native_uint(&storage);
                let combined = (old & !(mask << shift)) | ((integer & mask) << shift);
                let mut bytes = native_uint_to_bytes(combined, size);
                if field_needs_swap(obj, proto, size) {
                    bytes.reverse();
                }
                cdata::cdata_write(obj, offset, &bytes);
                return Ok(pyre_object::w_none());
            }
            let mut bytes = cdata::encode_value_into(&tc, value, obj, &index.to_string())?;
            if field_needs_swap(obj, proto, size) {
                bytes.reverse();
            }
            cdata::cdata_write(obj, offset, &bytes);
            if cdata::is_cdata_instance(value) {
                cdata::keep_ref(obj, &index.to_string(), value);
            }
            Ok(pyre_object::w_none())
        }
        "pointer" => {
            if unsafe { pyre_object::is_none(value) } {
                cdata::cdata_write(obj, offset, &vec![0; size]);
                return Ok(pyre_object::w_none());
            }
            let direct = unsafe { crate::baseobjspace::isinstance_w(value, proto) };
            let expected = stginfo::stginfo_of(proto).and_then(stginfo::stginfo_proto);
            let array_decay = if cdata::is_cdata_instance(value) {
                let value_cls = unsafe { pyre_object::w_instance_get_type(value) };
                stginfo::stginfo_of(value_cls)
                    .filter(|&i| stginfo::stginfo_paramfunc(i) == "array")
                    .and_then(stginfo::stginfo_proto)
                    == expected
            } else {
                false
            };
            let bytes = if direct {
                cdata::cdata_bytes(value).unwrap_or(&[]).to_vec()
            } else if array_decay {
                let addr = cdata::cdata_addr(value)
                    .ok_or_else(|| crate::PyError::type_error("incompatible types"))?;
                host_ctypes::simple_storage_value_to_bytes_endian(
                    "P",
                    host_ctypes::SimpleStorageValue::Pointer(addr),
                    false,
                )
            } else {
                return Err(crate::PyError::type_error("incompatible types"));
            };
            cdata::cdata_write(obj, offset, &bytes[..size.min(bytes.len())]);
            let keep = if array_decay {
                value
            } else {
                cdata::objects_for_keep(value)
            };
            cdata::keep_ref(obj, &index.to_string(), keep);
            Ok(pyre_object::w_none())
        }
        "array" => {
            let element = stginfo::stginfo_of(proto).and_then(stginfo::stginfo_proto);
            match element.and_then(cdata::type_code_of).as_deref() {
                Some("c") => {
                    let source = crate::typedef::buffer_as_bytes_like(value)?
                        .ok_or_else(|| crate::PyError::type_error("bytes-like object expected"))?;
                    let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(source) };
                    if bytes.len() > size {
                        return Err(crate::PyError::value_error("bytes too long"));
                    }
                    // PyCField_set: a character-array field is assigned with
                    // string semantics, so bytes beyond the first NUL do not
                    // overwrite the existing tail of the field.
                    let written = bytes
                        .iter()
                        .position(|&byte| byte == 0)
                        .map_or(bytes.len(), |nul| nul + 1);
                    cdata::cdata_write(obj, offset, &bytes[..written]);
                    Ok(pyre_object::w_none())
                }
                Some("u") => {
                    if !unsafe { pyre_object::is_str(value) } {
                        return Err(crate::PyError::type_error("unicode string expected"));
                    }
                    let mut field = cdata::cdata_bytes(obj)
                        .unwrap_or(&[])
                        .get(offset..offset.saturating_add(size))
                        .unwrap_or(&[])
                        .to_vec();
                    field.resize(size, 0);
                    host_ctypes::write_wchar_array_value(&mut field, unsafe {
                        pyre_object::w_str_get_wtf8(value)
                    })
                    .map_err(|_| crate::PyError::value_error("string too long"))?;
                    cdata::cdata_write(obj, offset, &field);
                    Ok(pyre_object::w_none())
                }
                _ => {
                    let array_value = if unsafe { crate::baseobjspace::isinstance_w(value, proto) }
                    {
                        value
                    } else if unsafe { pyre_object::is_tuple(value) } {
                        let values = seq_items(value).unwrap_or_default();
                        crate::call::type_call_instantiate(proto, &values)
                            .map_err(|error| crate::PyError::runtime_error(error.message))?
                    } else {
                        return Err(crate::PyError::type_error("incompatible types"));
                    };
                    let source = cdata::cdata_bytes(array_value).unwrap_or(&[]).to_vec();
                    cdata::cdata_write(obj, offset, &source[..source.len().min(size)]);
                    cdata::keep_ref(
                        obj,
                        &index.to_string(),
                        cdata::objects_for_keep(array_value),
                    );
                    Ok(pyre_object::w_none())
                }
            }
        }
        // Structures and unions are copied by value.
        "struct" | "union" => {
            if !unsafe { crate::baseobjspace::isinstance_w(value, proto) } {
                return Err(crate::PyError::type_error("incompatible types"));
            }
            // Snapshot the source: `s.f = s.f` aliases the destination buffer,
            // and `cdata_write`'s `copy_from_slice` assumes non-overlap.
            let src = cdata::cdata_bytes(value).unwrap_or(&[]).to_vec();
            let n = size.min(src.len());
            cdata::cdata_write(obj, offset, &src[..n]);
            cdata::keep_ref(obj, &index.to_string(), cdata::objects_for_keep(value));
            Ok(pyre_object::w_none())
        }
        _ => Err(crate::PyError::type_error(
            "assignment to this field type is not supported in this slice",
        )),
    }
}

fn cfield_repr(args: &[PyObjectRef]) -> PyResult {
    let cfield = args[0];
    let proto = cf_obj(cfield, "proto");
    let tyname = if !proto.is_null() && unsafe { pyre_object::is_type(proto) } {
        unsafe { pyre_object::typeobject::w_type_get_name(proto) }.to_string()
    } else {
        "?".to_string()
    };
    let s = format!(
        "<Field type={}, ofs={}, size={}>",
        tyname,
        cf_usize(cfield, "offset"),
        cf_usize(cfield, "size"),
    );
    Ok(pyre_object::w_str_new(&s))
}

fn cfield_new_internal(args: &[PyObjectRef]) -> PyResult {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let get = |name: &str, position: usize| {
        crate::builtins::resolve_pos_or_kw(
            pos.get(position).copied(),
            kwargs,
            name,
            "CField",
            position,
        )
    };
    let internal = get("_internal_use", 6)?
        .is_some_and(|value| crate::baseobjspace::is_true(value).unwrap_or(false));
    if !internal {
        return Err(crate::PyError::type_error(
            "CField is not intended to be used directly",
        ));
    }
    let name_obj = get("name", 1)?.ok_or_else(|| crate::PyError::type_error("missing name"))?;
    let proto = get("type", 2)?.ok_or_else(|| crate::PyError::type_error("missing type"))?;
    let byte_size = get("byte_size", 3)?
        .map(crate::baseobjspace::int_w)
        .transpose()?
        .ok_or_else(|| crate::PyError::type_error("missing byte_size"))?;
    let byte_offset = get("byte_offset", 4)?
        .map(crate::baseobjspace::int_w)
        .transpose()?
        .unwrap_or(0);
    let index = get("index", 5)?
        .map(crate::baseobjspace::int_w)
        .transpose()?
        .unwrap_or(0);
    if !unsafe { pyre_object::is_str(name_obj) } || !unsafe { pyre_object::is_type(proto) } {
        return Err(crate::PyError::type_error("invalid CField arguments"));
    }
    let expected = stginfo::field_size_of(proto)
        .ok_or_else(|| crate::PyError::type_error("type has no size"))?;
    if byte_size < 0 || byte_size as usize != expected {
        return Err(crate::PyError::value_error(
            "byte_size does not match type size",
        ));
    }
    let field = cfield_new(
        unsafe { pyre_object::w_str_get_value(name_obj) },
        proto,
        byte_offset.max(0) as usize,
        byte_size as usize,
        index.max(0) as usize,
    );
    let bit_size = get("bit_size", 7)?
        .map(crate::baseobjspace::int_w)
        .transpose()?;
    let bit_offset = get("bit_offset", 8)?
        .map(crate::baseobjspace::int_w)
        .transpose()?
        .unwrap_or(0);
    if let Some(bits) = bit_size {
        if bits <= 0 || bit_offset < 0 || bit_offset + bits > byte_size * 8 {
            return Err(crate::PyError::value_error(format!(
                "bit field '{}' overflows its type ({} + {} > {})",
                unsafe { pyre_object::w_str_get_value(name_obj) },
                bit_offset,
                bits,
                byte_size * 8,
            )));
        }
        let d = crate::baseobjspace::getdict(field);
        unsafe {
            pyre_object::w_dict_setitem_str(d, "bit_size", pyre_object::w_int_new(bits));
            pyre_object::w_dict_setitem_str(d, "bit_offset", pyre_object::w_int_new(bit_offset));
            pyre_object::w_dict_setitem_str(d, "is_bitfield", pyre_object::w_bool_from(true));
        }
    }
    Ok(field)
}

// ── Structure / Union instances ────────────────────────────────────────

fn structure_new(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() || !unsafe { pyre_object::is_type(args[0]) } {
        return Err(crate::PyError::type_error(
            "Structure.__new__ requires a type",
        ));
    }
    let cls = args[0];
    if unsafe { crate::baseobjspace::lookup_in_type(cls, "_abstract_") }.is_some() {
        return Err(crate::PyError::type_error("abstract class"));
    }
    let info =
        stginfo::stginfo_of(cls).ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let size = stginfo::stginfo_size(info);
    stginfo::stginfo_mark_final(info);
    let obj = pyre_object::w_instance_new(cls);
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return Err(crate::PyError::type_error("ctypes instance has no dict"));
    }
    unsafe { pyre_object::w_dict_setitem_str(d, "_b_", pyre_object::w_bytearray_new(size)) };
    Ok(obj)
}

/// Field names in base-first order (`init_pos_args`).
fn field_names_base_first(cls: PyObjectRef) -> Vec<String> {
    let mut names = Vec::new();
    for &t in mro_items(cls).iter().rev() {
        if let Some(f) = crate::type_dict_lookup(t, "_fields_") {
            if let Ok(entries) = field_entries(f) {
                names.extend(entries.into_iter().map(|entry| entry.name));
            }
        }
    }
    names
}

fn structure_init(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error("__init__ requires self"));
    }
    let obj = args[0];
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let names = field_names_base_first(cls);

    if pos.len() > names.len() {
        return Err(crate::PyError::type_error("too many initializers"));
    }
    for (i, &val) in pos.iter().enumerate() {
        crate::baseobjspace::setattr_str(obj, &names[i], val)?;
    }

    if let Some(kw) = kwargs {
        for (key_obj, val) in unsafe { pyre_object::w_dict_items(kw) } {
            if !unsafe { pyre_object::is_str(key_obj) } {
                continue;
            }
            let key = unsafe { pyre_object::w_str_get_value(key_obj) }.to_string();
            if key == "__pyre_kw__" {
                continue;
            }
            // Duplicate positional + keyword assignment for the same field.
            if let Some(pos_idx) = names.iter().position(|n| *n == key) {
                if pos_idx < pos.len() {
                    return Err(crate::PyError::type_error(format!(
                        "duplicate values for field '{key}'"
                    )));
                }
            }
            crate::baseobjspace::setattr_str(obj, &key, val)?;
        }
    }
    Ok(pyre_object::w_none())
}

// ── shared metaclass helpers (arrays / pointers) ───────────────────────

/// Read a key from `cls`'s **own** dict (never the MRO).
fn own_dict_get(cls: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    crate::type_dict_lookup(cls, key)
}

/// Store a class attribute directly and invalidate the type cache for it.
fn set_type_attr(cls: PyObjectRef, key: &str, value: PyObjectRef) {
    if crate::type_dict_store(cls, key, value) {
        pyre_object::gc_hook::try_gc_write_barrier(cls as *mut u8);
        unsafe { crate::baseobjspace::mutated(cls, Some(key)) };
    }
}

fn type_name(cls: PyObjectRef) -> String {
    match unsafe { crate::baseobjspace::lookup_in_type(cls, "__name__") } {
        Some(o) if unsafe { pyre_object::is_str(o) } => {
            unsafe { pyre_object::w_str_get_value(o) }.to_string()
        }
        _ => "?".to_string(),
    }
}

// ── PyCArrayType + Array ───────────────────────────────────────────────

/// Reserved key under which an element type caches the array types built over
/// it (`_ctypes._array_type_cache`), an `int`-keyed (by length) dict.  Holding
/// the cache on the element type's own dict keeps the built array types
/// GC-traced through the rooted element type rather than an untraced side table.
const ARRAY_TYPE_CACHE_KEY: &str = "_array_type_cache";

fn carraytype_new(args: &[PyObjectRef]) -> PyResult {
    crate::builtins::type_descr_new(args)
}

fn carraytype_init(args: &[PyObjectRef]) -> PyResult {
    metaclass_init(args, array_init_stginfo)
}

/// `PyCArrayType` layout: resolve `_length_` + `_type_` and build the array
/// `StgInfo` (`size = element_size * length`, align = element align).
fn array_init_stginfo(cls: PyObjectRef) -> PyResult {
    let length = match own_dict_get(cls, "_length_") {
        Some(v) => {
            if !unsafe { pyre_object::is_int(v) || pyre_object::is_long(v) } {
                return Err(crate::PyError::type_error(
                    "The '_length_' attribute must be an integer",
                ));
            }
            let n = if unsafe { pyre_object::is_long(v) } {
                let big = unsafe { pyre_object::longobject::w_long_get_value(v) };
                if big.sign() == malachite_bigint::Sign::Minus {
                    return Err(crate::PyError::value_error(
                        "The '_length_' attribute must not be negative",
                    ));
                }
                return Err(crate::PyError::overflow_error("array too large"));
            } else {
                unsafe { pyre_object::w_int_get_value(v) }
            };
            if n < 0 {
                return Err(crate::PyError::value_error(
                    "The '_length_' attribute must not be negative",
                ));
            }
            n as usize
        }
        None => match first_base_stginfo(cls) {
            Some(bi) => stginfo::stginfo_length(bi),
            None => {
                return Err(crate::PyError::attribute_error(
                    "class must define a '_length_' attribute",
                ));
            }
        },
    };

    let elem = match own_dict_get(cls, "_type_") {
        Some(t) => {
            if !unsafe { pyre_object::is_type(t) } {
                return Err(crate::PyError::type_error("_type_ must be a type"));
            }
            t
        }
        None => match first_base_stginfo(cls).and_then(stginfo::stginfo_proto) {
            Some(p) => p,
            None => {
                return Err(crate::PyError::attribute_error(
                    "class must define a '_type_' attribute",
                ));
            }
        },
    };

    let elem_size = stginfo::field_size_of(elem)
        .ok_or_else(|| crate::PyError::type_error("_type_ must have storage info"))?;
    let elem_align = stginfo::field_align_of(elem).unwrap_or(1).max(1);
    if elem_size != 0 && (length > usize::MAX / elem_size || elem_size * length > i64::MAX as usize)
    {
        return Err(crate::PyError::overflow_error("array too large"));
    }

    let mut data = StgInfoData::new(elem_size * length, elem_align, "array");
    data.length = length;
    data.element_size = elem_size;
    data.proto = Some(elem);
    if let Some(ei) = stginfo::stginfo_of(elem) {
        if stginfo::stginfo_flags(ei) & (stginfo::TYPEFLAG_ISPOINTER | stginfo::TYPEFLAG_HASPOINTER)
            != 0
        {
            data.flags |= stginfo::TYPEFLAG_HASPOINTER;
        }
    }
    stginfo::stginfo_set(cls, stginfo::stginfo_new(data));

    set_type_attr(cls, "_type_", elem);
    set_type_attr(cls, "_length_", pyre_object::w_int_new(length as i64));

    // Character arrays expose the CPython/PyPy convenience descriptors.
    match cdata::type_code_of(elem).as_deref() {
        Some("c") => install_char_array_getsets(cls),
        Some("u") => install_wchar_array_getsets(cls),
        _ => {}
    }
    Ok(pyre_object::w_none())
}

/// `ctype * n` — cache lookup on `(ctype, n)`, else create `ctype_Array_n`.
fn array_type_from_ctype(elem: PyObjectRef, n: usize) -> PyResult {
    let cache = match crate::type_dict_lookup(elem, ARRAY_TYPE_CACHE_KEY) {
        Some(d) => d,
        None => {
            let d = pyre_object::w_dict_new();
            if crate::type_dict_store(elem, ARRAY_TYPE_CACHE_KEY, d) {
                pyre_object::gc_hook::try_gc_write_barrier(elem as *mut u8);
            }
            d
        }
    };
    if let Some(found) = unsafe { pyre_object::w_dict_getitem(cache, n as i64) } {
        return Ok(found);
    }
    let name = format!("{}_Array_{}", type_name(elem), n);
    let ns = pyre_object::w_dict_new();
    unsafe {
        pyre_object::w_dict_setitem_str(ns, "_type_", elem);
        pyre_object::w_dict_setitem_str(ns, "_length_", pyre_object::w_int_new(n as i64));
    }
    let bases = pyre_object::w_tuple_new(vec![array_type()]);
    let args = [pyre_object::w_str_new(&name), bases, ns];
    let new_cls = crate::call::type_call_instantiate(pycarraytype_type(), &args)?;
    unsafe { pyre_object::w_dict_setitem(cache, n as i64, new_cls) };
    Ok(new_cls)
}

/// Resolved array-instance metadata.
struct ArrayMeta {
    length: usize,
    element_size: usize,
    proto: PyObjectRef,
}

fn array_meta(obj: PyObjectRef) -> Result<ArrayMeta, crate::PyError> {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let info =
        stginfo::stginfo_of(cls).ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let proto =
        stginfo::stginfo_proto(info).ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    Ok(ArrayMeta {
        length: stginfo::stginfo_length(info),
        element_size: stginfo::stginfo_element_size(info),
        proto,
    })
}

fn array_new(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() || !unsafe { pyre_object::is_type(args[0]) } {
        return Err(crate::PyError::type_error("Array.__new__ requires a type"));
    }
    let cls = args[0];
    let info = stginfo::stginfo_of(cls)
        .filter(|&i| stginfo::stginfo_proto(i).is_some())
        .ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let size = stginfo::stginfo_size(info);
    let obj = pyre_object::w_instance_new(cls);
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return Err(crate::PyError::type_error("ctypes instance has no dict"));
    }
    unsafe { pyre_object::w_dict_setitem_str(d, "_b_", pyre_object::w_bytearray_new(size)) };
    Ok(obj)
}

fn array_init(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error("__init__ requires self"));
    }
    let obj = args[0];
    let (pos, _kw) = crate::builtins::split_builtin_kwargs(&args[1..]);
    let meta = array_meta(obj)?;
    if pos.len() > meta.length {
        return Err(crate::PyError::index_error("too many initializers"));
    }
    for (i, &val) in pos.iter().enumerate() {
        array_set_index(obj, &meta, i, val)?;
    }
    Ok(pyre_object::w_none())
}

fn array_len(args: &[PyObjectRef]) -> PyResult {
    let meta = array_meta(args[0])?;
    Ok(pyre_object::w_int_new(meta.length as i64))
}

fn normalize_index(mut i: i64, length: usize) -> Result<usize, crate::PyError> {
    if i < 0 {
        i += length as i64;
    }
    if i < 0 || i >= length as i64 {
        return Err(crate::PyError::index_error("invalid index"));
    }
    Ok(i as usize)
}

fn array_getitem(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let key = args[1];
    let meta = array_meta(obj)?;
    if unsafe { pyre_object::is_int(key) } {
        let idx = normalize_index(unsafe { pyre_object::w_int_get_value(key) }, meta.length)?;
        return array_get_index(obj, &meta, idx);
    }
    if unsafe { pyre_object::is_slice(key) } {
        return array_get_slice(obj, &meta, key);
    }
    Err(crate::PyError::type_error("indices must be integers"))
}

fn array_get_index(obj: PyObjectRef, meta: &ArrayMeta, idx: usize) -> PyResult {
    let offset = idx * meta.element_size;
    match proto_kind(meta.proto).as_str() {
        "simple" => {
            let tc = cdata::type_code_of(meta.proto)
                .ok_or_else(|| crate::PyError::type_error("element has no '_type_'"))?;
            let all = cdata::cdata_bytes(obj)
                .ok_or_else(|| crate::PyError::type_error("instance has no buffer"))?;
            let end = (offset + meta.element_size).min(all.len());
            let start = offset.min(end);
            Ok(cdata::decoded_to_pyobject(host_ctypes::decode_type_code(
                &tc,
                &all[start..end],
            )))
        }
        "struct" | "union" | "array" | "pointer" => Ok(cdata::make_indexed_subview(
            meta.proto,
            obj,
            offset,
            meta.element_size,
            idx,
        )),
        _ => Err(crate::PyError::type_error(
            "element type has no storage info",
        )),
    }
}

fn array_setitem(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let key = args[1];
    let value = args[2];
    let meta = array_meta(obj)?;
    if unsafe { pyre_object::is_int(key) } {
        let idx = normalize_index(unsafe { pyre_object::w_int_get_value(key) }, meta.length)?;
        return array_set_index(obj, &meta, idx, value);
    }
    if unsafe { pyre_object::is_slice(key) } {
        return array_set_slice(obj, &meta, key, value);
    }
    Err(crate::PyError::type_error("indices must be integers"))
}

fn array_set_index(obj: PyObjectRef, meta: &ArrayMeta, idx: usize, value: PyObjectRef) -> PyResult {
    let offset = idx * meta.element_size;
    match proto_kind(meta.proto).as_str() {
        "simple" => {
            let tc = cdata::type_code_of(meta.proto)
                .ok_or_else(|| crate::PyError::type_error("element has no '_type_'"))?;
            let bytes = cdata::encode_value_into(&tc, value, obj, &idx.to_string())?;
            cdata::cdata_write(obj, offset, &bytes);
            if cdata::is_cdata_instance(value) {
                cdata::keep_ref(obj, &idx.to_string(), value);
            }
            Ok(pyre_object::w_none())
        }
        "struct" | "union" | "array" | "pointer" => {
            if !unsafe { crate::baseobjspace::isinstance_w(value, meta.proto) } {
                return Err(crate::PyError::type_error("incompatible types"));
            }
            // Snapshot the source: `a[i] = a[i]` aliases the destination
            // buffer, and `cdata_write`'s `copy_from_slice` assumes non-overlap.
            let src = cdata::cdata_bytes(value).unwrap_or(&[]).to_vec();
            let n = meta.element_size.min(src.len());
            cdata::cdata_write(obj, offset, &src[..n]);
            cdata::keep_ref(obj, &idx.to_string(), cdata::objects_for_keep(value));
            Ok(pyre_object::w_none())
        }
        _ => Err(crate::PyError::type_error(
            "assignment to this element type is not supported in this slice",
        )),
    }
}

/// The concrete indices a slice selects over `length`, PySlice-adjusted.
fn slice_index_list(slice: PyObjectRef, length: usize) -> Result<Vec<usize>, crate::PyError> {
    let (start, stop, step) = crate::sliceobject::slice_unpack(
        unsafe { pyre_object::w_slice_get_start(slice) },
        unsafe { pyre_object::w_slice_get_stop(slice) },
        unsafe { pyre_object::w_slice_get_step(slice) },
    )?;
    let (start, _, step, slice_length) =
        crate::sliceobject::slice_adjust_indices(start, stop, step, length as i64);
    let mut out = Vec::with_capacity(slice_length as usize);
    let mut i = start;
    for _ in 0..slice_length {
        out.push(i as usize);
        i = i.saturating_add(step);
    }
    Ok(out)
}

fn array_get_slice(obj: PyObjectRef, meta: &ArrayMeta, slice: PyObjectRef) -> PyResult {
    let idxs = slice_index_list(slice, meta.length)?;
    // Character arrays slice to bytes/str; other elements slice to a list.
    let tc = cdata::type_code_of(meta.proto);
    if tc.as_deref() == Some("c") {
        let all = cdata::cdata_bytes(obj).unwrap_or(&[]);
        let bytes: Vec<u8> = idxs
            .iter()
            .map(|&i| all.get(i * meta.element_size).copied().unwrap_or(0))
            .collect();
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes));
    }
    if tc.as_deref() == Some("u") {
        let mut value = String::new();
        for i in idxs {
            let offset = i * meta.element_size;
            let all = cdata::cdata_bytes(obj).unwrap_or(&[]);
            let end = (offset + meta.element_size).min(all.len());
            if let host_ctypes::DecodedValue::String(s) =
                host_ctypes::decode_type_code("u", &all[offset.min(end)..end])
            {
                value.push_str(&s);
            }
        }
        return Ok(pyre_object::w_str_new(&value));
    }
    let mut items = Vec::with_capacity(idxs.len());
    for i in idxs {
        items.push(array_get_index(obj, meta, i)?);
    }
    Ok(pyre_object::w_list_new(items))
}

fn array_set_slice(
    obj: PyObjectRef,
    meta: &ArrayMeta,
    slice: PyObjectRef,
    value: PyObjectRef,
) -> PyResult {
    let idxs = slice_index_list(slice, meta.length)?;
    let tc = cdata::type_code_of(meta.proto);
    let items = if unsafe { pyre_object::is_bytes(value) }
        && matches!(tc.as_deref(), Some("c" | "b" | "B"))
    {
        unsafe { pyre_object::bytesobject::w_bytes_data(value) }
            .iter()
            .map(|&byte| {
                if tc.as_deref() == Some("c") {
                    pyre_object::bytesobject::w_bytes_from_bytes(&[byte])
                } else {
                    pyre_object::w_int_new(byte as i64)
                }
            })
            .collect()
    } else if unsafe { pyre_object::is_str(value) } && tc.as_deref() == Some("u") {
        unsafe { pyre_object::w_str_get_wtf8(value) }
            .code_points()
            .map(|point| {
                let mut text = rustpython_wtf8::Wtf8Buf::new();
                text.push(point);
                pyre_object::w_str_from_wtf8(text)
            })
            .collect()
    } else {
        seq_items(value).ok_or_else(|| crate::PyError::type_error("can only assign a sequence"))?
    };
    if items.len() != idxs.len() {
        return Err(crate::PyError::value_error(
            "Can only assign sequence of same size",
        ));
    }
    for (i, v) in idxs.into_iter().zip(items) {
        array_set_index(obj, meta, i, v)?;
    }
    Ok(pyre_object::w_none())
}

// ── c_char array `.value` / `.raw` ─────────────────────────────────────

fn install_char_array_getsets(cls: PyObjectRef) {
    set_type_attr(
        cls,
        "value",
        crate::typedef::make_getset_property_named(
            crate::make_builtin_function_with_arity("value", char_array_get_value, 2),
            crate::make_builtin_function_with_arity("value", char_array_set_value, 3),
            crate::make_builtin_function_with_arity(
                "value",
                |_args| Err(crate::PyError::type_error("can't delete attribute")),
                2,
            ),
            "value",
        ),
    );
    set_type_attr(
        cls,
        "raw",
        crate::typedef::make_getset_property_named(
            crate::make_builtin_function_with_arity("raw", char_array_get_raw, 2),
            crate::make_builtin_function_with_arity("raw", char_array_set_raw, 3),
            pyre_object::PY_NULL,
            "raw",
        ),
    );
}

fn char_array_get_value(args: &[PyObjectRef]) -> PyResult {
    let obj = args[1];
    let buf = cdata::cdata_bytes(obj).unwrap_or(&[]);
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(
        host_ctypes::char_array_field_value(buf),
    ))
}

fn char_array_set_value(args: &[PyObjectRef]) -> PyResult {
    let obj = args[1];
    let value = args[2];
    if !unsafe { pyre_object::is_bytes(value) } {
        return Err(crate::PyError::type_error("bytes expected"));
    }
    let src = unsafe { pyre_object::bytesobject::w_bytes_data(value) };
    let size = cdata::cdata_len(obj).unwrap_or(0);
    if src.len() > size {
        return Err(crate::PyError::value_error("byte string too long"));
    }
    // Copy `src`, then a NUL terminator when the buffer has room (no tail zero).
    let mut buf = src.to_vec();
    if src.len() < size {
        buf.push(0);
    }
    cdata::cdata_write(obj, 0, &buf);
    Ok(pyre_object::w_none())
}

fn char_array_get_raw(args: &[PyObjectRef]) -> PyResult {
    let obj = args[1];
    let buf = cdata::cdata_bytes(obj).unwrap_or(&[]);
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(buf))
}

fn char_array_set_raw(args: &[PyObjectRef]) -> PyResult {
    let obj = args[1];
    let value = args[2];
    let source = crate::typedef::buffer_as_bytes_like(value)?
        .ok_or_else(|| crate::PyError::type_error("bytes-like object expected"))?;
    let src = unsafe { pyre_object::bytesobject::bytes_like_data(source) };
    let size = cdata::cdata_len(obj).unwrap_or(0);
    if src.len() > size {
        return Err(crate::PyError::value_error("byte string too long"));
    }
    cdata::cdata_write(obj, 0, src);
    Ok(pyre_object::w_none())
}

fn install_wchar_array_getsets(cls: PyObjectRef) {
    set_type_attr(
        cls,
        "value",
        crate::typedef::make_getset_property_named(
            crate::make_builtin_function_with_arity("value", wchar_array_get_value, 2),
            crate::make_builtin_function_with_arity("value", wchar_array_set_value, 3),
            pyre_object::PY_NULL,
            "value",
        ),
    );
}

fn wchar_array_get_value(args: &[PyObjectRef]) -> PyResult {
    Ok(pyre_object::w_str_new(&host_ctypes::wstring_from_bytes(
        cdata::cdata_bytes(args[1]).unwrap_or(&[]),
    )))
}

fn wchar_array_set_value(args: &[PyObjectRef]) -> PyResult {
    let obj = args[1];
    let value = args[2];
    if !unsafe { pyre_object::is_str(value) } {
        return Err(crate::PyError::type_error("unicode string expected"));
    }
    let text = unsafe { pyre_object::w_str_get_wtf8(value) };
    let size = cdata::cdata_len(obj).unwrap_or(0);
    // PyCArray_set_value overwrites the characters and one terminator only;
    // bytes after that terminator retain their previous values.
    let mut bytes = cdata::cdata_bytes(obj).unwrap_or(&[]).to_vec();
    bytes.resize(size, 0);
    host_ctypes::write_wchar_array_value(&mut bytes, text)
        .map_err(|_| crate::PyError::value_error("string too long"))?;
    cdata::cdata_write(obj, 0, &bytes);
    Ok(pyre_object::w_none())
}

// ── PyCPointerType + _Pointer ──────────────────────────────────────────

fn cpointertype_new(args: &[PyObjectRef]) -> PyResult {
    crate::builtins::type_descr_new(args)
}

fn cpointertype_init(args: &[PyObjectRef]) -> PyResult {
    metaclass_init(args, pointer_init_stginfo)
}

/// `PyCPointerType` layout: pointer-sized `StgInfo` with `ISPOINTER`, and
/// memoise the pointer type on the pointed-to type (`POINTER` identity).
fn pointer_init_stginfo(cls: PyObjectRef) -> PyResult {
    let proto = unsafe { crate::baseobjspace::lookup_in_type(cls, "_type_") }
        .filter(|&t| !t.is_null() && unsafe { pyre_object::is_type(t) });
    if let Some(p) = proto {
        if stginfo::field_size_of(p).is_none() {
            return Err(crate::PyError::type_error("_type_ must have storage info"));
        }
    }
    let psize = host_ctypes::pointer_size();
    let mut data = StgInfoData::new(psize, psize, "pointer");
    data.length = 1;
    data.flags |= stginfo::TYPEFLAG_ISPOINTER;
    data.proto = proto;
    data.format = Some(match proto {
        Some(p) if stginfo::field_size_of(p).unwrap_or(0) > 0 => {
            let shape = {
                let mut dims = Vec::new();
                let mut current = p;
                while let Some(info) = stginfo::stginfo_of(current) {
                    if stginfo::stginfo_paramfunc(info) != "array" {
                        break;
                    }
                    dims.push(stginfo::stginfo_length(info));
                    let Some(next) = stginfo::stginfo_proto(info) else {
                        break;
                    };
                    current = next;
                }
                dims
            };
            let prefix = if shape.is_empty() {
                String::new()
            } else {
                format!(
                    "({})",
                    shape
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                )
            };
            format!("&{prefix}{}", cdata::ctype_pep3118_format(p, None))
        }
        _ => "&B".to_string(),
    });
    stginfo::stginfo_set(cls, stginfo::stginfo_new(data));

    if let Some(p) = proto {
        let pinfo = match stginfo::stginfo_of(p) {
            Some(i) => i,
            None => {
                let size = stginfo::field_size_of(p).unwrap_or(0);
                let align = stginfo::field_align_of(p).unwrap_or(1);
                let info = stginfo::stginfo_new(StgInfoData::new(size, align, "simple"));
                stginfo::stginfo_set(p, info);
                info
            }
        };
        stginfo::stginfo_set_pointer_type(pinfo, cls);
    }
    Ok(pyre_object::w_none())
}

fn pointer_new(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() || !unsafe { pyre_object::is_type(args[0]) } {
        return Err(crate::PyError::type_error(
            "_Pointer.__new__ requires a type",
        ));
    }
    let cls = args[0];
    if stginfo::stginfo_of(cls)
        .and_then(stginfo::stginfo_proto)
        .is_none()
    {
        return Err(crate::PyError::type_error(
            "Cannot create instance: has no _type_",
        ));
    }
    let obj = pyre_object::w_instance_new(cls);
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return Err(crate::PyError::type_error("ctypes instance has no dict"));
    }
    let psize = host_ctypes::pointer_size();
    unsafe { pyre_object::w_dict_setitem_str(d, "_b_", pyre_object::w_bytearray_new(psize)) };
    Ok(obj)
}

fn pointer_init(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() {
        return Err(crate::PyError::type_error("__init__ requires self"));
    }
    let obj = args[0];
    let (pos, _kw) = crate::builtins::split_builtin_kwargs(&args[1..]);
    if let Some(&val) = pos.first() {
        if !unsafe { pyre_object::is_none(val) } {
            pointer_set_contents(obj, val)?;
        }
    }
    Ok(pyre_object::w_none())
}

/// Store `value`'s buffer address in the pointer and keep `value` alive.
fn pointer_set_contents(obj: PyObjectRef, value: PyObjectRef) -> PyResult {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let proto = stginfo::stginfo_of(cls)
        .and_then(stginfo::stginfo_proto)
        .ok_or_else(|| crate::PyError::type_error("Cannot create instance: has no _type_"))?;
    if !cdata::is_cdata_instance(value)
        || !unsafe { crate::baseobjspace::isinstance_w(value, proto) }
    {
        return Err(crate::PyError::type_error(format!(
            "expected {} instead of {}",
            type_name(proto),
            type_name(unsafe { pyre_object::w_instance_get_type(value) })
        )));
    }
    let addr = cdata::cdata_addr(value)
        .ok_or_else(|| crate::PyError::type_error("target has no buffer"))?;
    let bytes = host_ctypes::simple_storage_value_to_bytes_endian(
        "P",
        host_ctypes::SimpleStorageValue::Pointer(addr),
        false,
    );
    cdata::cdata_write(obj, 0, &bytes);
    cdata::keep_ref(obj, "1", value);
    Ok(pyre_object::w_none())
}

fn contents_get(args: &[PyObjectRef]) -> PyResult {
    let obj = args[1];
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let proto = stginfo::stginfo_of(cls)
        .and_then(stginfo::stginfo_proto)
        .ok_or_else(|| crate::PyError::type_error("has no _type_"))?;
    let ptr = host_ctypes::read_pointer_from_buffer(cdata::cdata_bytes(obj).unwrap_or(&[]));
    if ptr == 0 {
        return Err(crate::PyError::value_error("NULL pointer access"));
    }
    let size = stginfo::field_size_of(proto).unwrap_or_else(host_ctypes::pointer_size);
    Ok(cdata::make_at_address(proto, ptr, size, obj))
}

fn contents_set(args: &[PyObjectRef]) -> PyResult {
    pointer_set_contents(args[1], args[2])
}

/// `(proto, element_size, ptr_value)` for a pointer instance.
fn pointer_meta(obj: PyObjectRef) -> Result<(PyObjectRef, usize, usize), crate::PyError> {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let proto = stginfo::stginfo_of(cls)
        .and_then(stginfo::stginfo_proto)
        .ok_or_else(|| crate::PyError::type_error("has no _type_"))?;
    let element_size = stginfo::field_size_of(proto).unwrap_or_else(host_ctypes::pointer_size);
    let ptr = host_ctypes::read_pointer_from_buffer(cdata::cdata_bytes(obj).unwrap_or(&[]));
    Ok((proto, element_size, ptr))
}

fn pointer_getitem(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let key = args[1];
    if unsafe { pyre_object::is_slice(key) } {
        let start_obj = unsafe { pyre_object::sliceobject::w_slice_get_start(key) };
        let stop_obj = unsafe { pyre_object::sliceobject::w_slice_get_stop(key) };
        let step_obj = unsafe { pyre_object::sliceobject::w_slice_get_step(key) };
        if unsafe { pyre_object::is_none(stop_obj) } {
            return Err(crate::PyError::value_error("slice stop is required"));
        }
        let step = if unsafe { pyre_object::is_none(step_obj) } {
            1
        } else {
            crate::sliceobject::eval_slice_index(step_obj)?
        };
        if step == 0 {
            return Err(crate::PyError::value_error("slice step cannot be zero"));
        }
        let start = if unsafe { pyre_object::is_none(start_obj) } {
            if step < 0 { -1 } else { 0 }
        } else {
            crate::sliceobject::eval_slice_index(start_obj)?
        };
        let stop = crate::sliceobject::eval_slice_index(stop_obj)?;
        let mut values = Vec::new();
        let mut index = start;
        while if step > 0 { index < stop } else { index > stop } {
            values.push(pointer_get_index(obj, index as isize)?);
            index = index.saturating_add(step);
        }
        return Ok(pyre_object::w_list_new(values));
    }
    if !unsafe { pyre_object::is_int(key) } {
        return Err(crate::PyError::type_error(
            "Pointer indices must be integer",
        ));
    }
    pointer_get_index(obj, unsafe { pyre_object::w_int_get_value(key) } as isize)
}

fn pointer_get_index(obj: PyObjectRef, index: isize) -> PyResult {
    let (proto, element_size, ptr) = pointer_meta(obj)?;
    if ptr == 0 {
        return Err(crate::PyError::value_error("NULL pointer access"));
    }
    let addr = host_ctypes::pointer_item_address(ptr, index, element_size);
    match proto_kind(proto).as_str() {
        "simple" => {
            let tc = cdata::type_code_of(proto)
                .ok_or_else(|| crate::PyError::type_error("element has no '_type_'"))?;
            let bytes = unsafe { host_ctypes::borrow_memory(addr as *const u8, element_size) };
            Ok(cdata::decoded_to_pyobject(host_ctypes::decode_type_code(
                &tc, bytes,
            )))
        }
        _ => Ok(cdata::make_at_address(proto, addr, element_size, obj)),
    }
}

fn pointer_setitem(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let key = args[1];
    let value = args[2];
    if !unsafe { pyre_object::is_int(key) } {
        return Err(crate::PyError::type_error(
            "Pointer indices must be integer",
        ));
    }
    let (proto, element_size, ptr) = pointer_meta(obj)?;
    if ptr == 0 {
        return Err(crate::PyError::value_error("NULL pointer access"));
    }
    let index = unsafe { pyre_object::w_int_get_value(key) } as isize;
    let addr = host_ctypes::pointer_item_address(ptr, index, element_size);
    match proto_kind(proto).as_str() {
        "simple" => {
            let tc = cdata::type_code_of(proto)
                .ok_or_else(|| crate::PyError::type_error("element has no '_type_'"))?;
            let bytes = cdata::encode_value_into(&tc, value, obj, &index.to_string())?;
            unsafe { host_ctypes::copy_bytes_to_address(addr, &bytes, element_size) };
            if cdata::is_cdata_instance(value) {
                cdata::keep_ref(obj, &index.to_string(), value);
            }
            Ok(pyre_object::w_none())
        }
        _ => {
            if !unsafe { crate::baseobjspace::isinstance_w(value, proto) } {
                return Err(crate::PyError::type_error("incompatible types"));
            }
            let src = cdata::cdata_bytes(value).unwrap_or(&[]);
            unsafe { host_ctypes::copy_bytes_to_address(addr, src, element_size) };
            Ok(pyre_object::w_none())
        }
    }
}

fn pointer_bool(args: &[PyObjectRef]) -> PyResult {
    let ptr = host_ctypes::read_pointer_from_buffer(cdata::cdata_bytes(args[0]).unwrap_or(&[]));
    Ok(pyre_object::w_bool_from(ptr != 0))
}
