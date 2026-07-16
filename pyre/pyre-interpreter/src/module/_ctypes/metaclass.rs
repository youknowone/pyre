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
use std::cell::RefCell;

type PyResult = Result<PyObjectRef, crate::PyError>;

// ── cached type objects ────────────────────────────────────────────────

macro_rules! cached_type {
    ($cell:ident, $f:ident, $build:expr) => {
        thread_local! {
            static $cell: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
        }
        pub(super) fn $f() -> PyObjectRef {
            $cell.with(|c| *c.get_or_init($build))
        }
    };
}

cached_type!(PYCSIMPLETYPE, pycsimpletype_type, || {
    crate::typedef::make_builtin_type_with_base(
        "PyCSimpleType",
        |ns| {
            install_new(ns, csimpletype_new);
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
        crate::typedef::w_object(),
    );
    finish_aggregate_base(tp, pycstructtype_type(), "struct");
    tp
});

cached_type!(UNION, union_type, || {
    let tp = crate::typedef::make_builtin_type_with_base(
        "Union",
        init_aggregate_base,
        crate::typedef::w_object(),
    );
    finish_aggregate_base(tp, pycuniontype_type(), "union");
    tp
});

cached_type!(CFIELD, cfield_type, || {
    let tp = crate::typedef::make_builtin_type("CField", |ns| {
        type_ns_store(
            ns,
            "__new__",
            crate::make_builtin_function("__new__", cfield_new_disabled),
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
            "__repr__",
            crate::make_builtin_function("__repr__", cfield_repr),
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
            install_shared_meta(ns);
        },
        crate::typedef::w_type(),
    )
});

cached_type!(ARRAY, array_type, || {
    let tp = crate::typedef::make_builtin_type_with_base(
        "Array",
        init_array_base,
        crate::typedef::w_object(),
    );
    finish_element_base(tp, pycarraytype_type());
    tp
});

cached_type!(POINTER_BASE, pointer_base_type, || {
    let tp = crate::typedef::make_builtin_type_with_base(
        "_Pointer",
        init_pointer_base,
        crate::typedef::w_object(),
    );
    finish_element_base(tp, pycpointertype_type());
    tp
});

fn install_new(ns: PyObjectRef, f: crate::gateway::BuiltinCodeFn) {
    type_ns_store(ns, "__new__", crate::make_builtin_function("__new__", f));
}

/// `__mul__`, `__pointer_type__`, and `from_param` — shared by all metaclasses.
fn install_shared_meta(ns: PyObjectRef) {
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
}

fn finish_aggregate_base(tp: PyObjectRef, metaclass: PyObjectRef, paramfunc: &'static str) {
    unsafe {
        pyre_object::typeobject::w_type_set_hasdict(tp, true);
        (*tp).w_class = metaclass;
    }
    // A default 0-size StgInfo so `clone-from-base` works and a bare
    // `Structure()`/`Union()` is not treated as abstract.
    stginfo::stginfo_set(tp, stginfo::stginfo_new(StgInfoData::new(0, 1, paramfunc)));
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
}

// ── metaclass `__new__` ────────────────────────────────────────────────

fn csimpletype_new(args: &[PyObjectRef]) -> PyResult {
    let cls = crate::builtins::type_descr_new(args)?;
    simple_init_stginfo(cls)?;
    Ok(cls)
}

fn cstructtype_new(args: &[PyObjectRef]) -> PyResult {
    let cls = crate::builtins::type_descr_new(args)?;
    struct_union_init_stginfo(cls, false)?;
    Ok(cls)
}

fn cuniontype_new(args: &[PyObjectRef]) -> PyResult {
    let cls = crate::builtins::type_descr_new(args)?;
    struct_union_init_stginfo(cls, true)?;
    Ok(cls)
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

/// Parse `_fields_` into `(name, ctype)` pairs (2-tuples only).
fn field_entries(fields: PyObjectRef) -> Result<Vec<(String, PyObjectRef)>, crate::PyError> {
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
        if n >= 3 {
            return Err(crate::PyError::type_error(
                "bit fields are not supported in this slice",
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
        out.push((
            unsafe { pyre_object::w_str_get_value(name) }.to_string(),
            ty,
        ));
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
    let entries = field_entries(fields)?;

    let is_swapped =
        unsafe { crate::baseobjspace::lookup_in_type(cls, "_swappedbytes_") }.is_some();
    let pack = usize_attr(cls, "_pack_", 0);
    let forced = usize_attr(cls, "_align_", 1).max(1);

    let (mut offset, mut max_align) = match first_base_stginfo(cls) {
        Some(bi) => (
            stginfo::stginfo_size(bi),
            stginfo::stginfo_align(bi).max(forced),
        ),
        None => (0usize, forced),
    };
    let mut union_max = 0usize;
    let mut has_pointer = false;

    for (index, (name, ftype)) in entries.iter().enumerate() {
        let size = stginfo::field_size_of(*ftype)
            .ok_or_else(|| crate::PyError::type_error(format!("field '{name}' has no size")))?;
        let align = stginfo::field_align_of(*ftype).unwrap_or(1).max(1);
        let eff = if pack > 0 { pack.min(align) } else { align };

        if !is_union && eff > 0 && offset % eff != 0 {
            offset += eff - (offset % eff);
        }
        max_align = max_align.max(eff);

        if let Some(fi) = stginfo::stginfo_of(*ftype) {
            if stginfo::stginfo_flags(fi)
                & (stginfo::TYPEFLAG_ISPOINTER | stginfo::TYPEFLAG_HASPOINTER)
                != 0
            {
                has_pointer = true;
            }
        }
        mark_type_final(*ftype, size, align);

        let field_offset = if is_union { 0 } else { offset };
        let cf = cfield_new(name, *ftype, field_offset, size, index);
        set_type_attr(cls, name, cf);

        if is_union {
            union_max = union_max.max(size);
        } else {
            offset += size;
        }
    }

    let total_align = max_align.max(forced);
    let raw = if is_union { union_max } else { offset };
    let aligned = if total_align > 0 {
        raw.div_ceil(total_align) * total_align
    } else {
        raw
    };

    if let Some(ci) = stginfo::stginfo_of(cls) {
        if stginfo::stginfo_is_final(ci) {
            return Err(crate::PyError::attribute_error(
                "Structure or union cannot contain itself",
            ));
        }
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
    data.length = entries.len();
    data.flags = flags;
    data.big_endian = is_swapped ^ cfg!(target_endian = "big");
    stginfo::stginfo_set(cls, stginfo::stginfo_new(data));

    // Store the raw `_fields_` so the metaclass getset can return it.
    set_type_attr(cls, "_fields_", fields);
    Ok(pyre_object::w_none())
}

// ── metaclass namespace methods ────────────────────────────────────────

/// `ctype * n` — build (and cache) the `n`-element array type of `ctype`.
fn meta_mul(args: &[PyObjectRef]) -> PyResult {
    let cls = args[0];
    let count = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    if count.is_null() || !unsafe { pyre_object::is_int(count) } {
        return Err(crate::PyError::not_implemented(
            "array length must be an int",
        ));
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
    if let Some(info) = stginfo::stginfo_of(cls) {
        if stginfo::stginfo_is_final(info) {
            return Err(crate::PyError::attribute_error("_fields_ is final"));
        }
    }
    let is_union = stginfo::stginfo_of(cls)
        .map(|i| stginfo::stginfo_paramfunc(i) == "union")
        .unwrap_or(false);
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
        pyre_object::w_dict_setitem_str(d, "offset", pyre_object::w_int_new(offset as i64));
        pyre_object::w_dict_setitem_str(d, "size", pyre_object::w_int_new(size as i64));
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
    let proto = cf_obj(cfield, "proto");
    let offset = cf_usize(cfield, "offset");
    let size = cf_usize(cfield, "size");

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
            Ok(cdata::decoded_to_pyobject(host_ctypes::decode_type_code(
                &tc,
                &field_bytes,
            )))
        }
        "struct" | "union" | "array" | "pointer" => {
            Ok(cdata::make_subview(proto, obj, offset, size))
        }
        _ => Err(crate::PyError::type_error("field type has no storage info")),
    }
}

fn cfield_set(args: &[PyObjectRef]) -> PyResult {
    let cfield = args[0];
    let obj = args[1];
    let value = args[2];
    let proto = cf_obj(cfield, "proto");
    let offset = cf_usize(cfield, "offset");
    let size = cf_usize(cfield, "size");
    let index = cf_usize(cfield, "index");

    match proto_kind(proto).as_str() {
        "simple" => {
            let tc = cdata::type_code_of(proto)
                .ok_or_else(|| crate::PyError::type_error("field has no '_type_'"))?;
            let mut bytes = cdata::encode_value(&tc, value)?;
            if field_needs_swap(obj, proto, size) {
                bytes.reverse();
            }
            cdata::cdata_write(obj, offset, &bytes);
            if cdata::is_cdata_instance(value) || unsafe { pyre_object::is_bytes(value) } {
                cdata::keep_ref(obj, &index.to_string(), value);
            }
            Ok(pyre_object::w_none())
        }
        "struct" | "union" => {
            if !unsafe { crate::baseobjspace::isinstance_w(value, proto) } {
                return Err(crate::PyError::type_error("incompatible types"));
            }
            let src = cdata::cdata_bytes(value).unwrap_or(&[]);
            let n = size.min(src.len());
            cdata::cdata_write(obj, offset, &src[..n]);
            cdata::keep_ref(obj, &index.to_string(), value);
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
    let tyname = match unsafe { crate::baseobjspace::lookup_in_type(proto, "__name__") } {
        Some(o) if unsafe { pyre_object::is_str(o) } => {
            unsafe { pyre_object::w_str_get_value(o) }.to_string()
        }
        _ => "?".to_string(),
    };
    let s = format!(
        "<Field type={}, ofs={}, size={}>",
        tyname,
        cf_usize(cfield, "offset"),
        cf_usize(cfield, "size"),
    );
    Ok(pyre_object::w_str_new(&s))
}

fn cfield_new_disabled(_args: &[PyObjectRef]) -> PyResult {
    Err(crate::PyError::type_error(
        "CField is not intended to be used directly",
    ))
}

// ── Structure / Union instances ────────────────────────────────────────

fn structure_new(args: &[PyObjectRef]) -> PyResult {
    if args.is_empty() || !unsafe { pyre_object::is_type(args[0]) } {
        return Err(crate::PyError::type_error(
            "Structure.__new__ requires a type",
        ));
    }
    let cls = args[0];
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
                names.extend(entries.into_iter().map(|(n, _)| n));
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

thread_local! {
    /// `(element_type, length) → array_type`, the pyre form of the
    /// `_ctypes._array_type_cache` dict (keyed by type identity + length).
    static ARRAY_TYPE_CACHE: RefCell<Vec<(PyObjectRef, usize, PyObjectRef)>> =
        const { RefCell::new(Vec::new()) };
}

fn carraytype_new(args: &[PyObjectRef]) -> PyResult {
    let cls = crate::builtins::type_descr_new(args)?;
    array_init_stginfo(cls)?;
    Ok(cls)
}

/// `PyCArrayType` layout: resolve `_length_` + `_type_` and build the array
/// `StgInfo` (`size = element_size * length`, align = element align).
fn array_init_stginfo(cls: PyObjectRef) -> PyResult {
    let length = match own_dict_get(cls, "_length_") {
        Some(v) => {
            if !unsafe { pyre_object::is_int(v) } {
                return Err(crate::PyError::type_error(
                    "The '_length_' attribute must be an integer",
                ));
            }
            let n = unsafe { pyre_object::w_int_get_value(v) };
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
    if elem_size != 0 && length > usize::MAX / elem_size {
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

    // `c` element arrays gain `value`/`raw`; `u` (wchar) is deferred.
    if cdata::type_code_of(elem).as_deref() == Some("c") {
        install_char_array_getsets(cls);
    }
    Ok(pyre_object::w_none())
}

/// `ctype * n` — cache lookup on `(ctype, n)`, else create `ctype_Array_n`.
fn array_type_from_ctype(elem: PyObjectRef, n: usize) -> PyResult {
    if let Some(found) = ARRAY_TYPE_CACHE.with(|c| {
        c.borrow()
            .iter()
            .find(|(t, k, _)| *t == elem && *k == n)
            .map(|(_, _, ty)| *ty)
    }) {
        return Ok(found);
    }
    let name = format!("{}_Array_{}", type_name(elem), n);
    let ns = pyre_object::w_dict_new();
    unsafe {
        pyre_object::w_dict_setitem_str(ns, "_type_", elem);
        pyre_object::w_dict_setitem_str(ns, "_length_", pyre_object::w_int_new(n as i64));
    }
    let bases = pyre_object::w_tuple_new(vec![array_type()]);
    let args = [
        pycarraytype_type(),
        pyre_object::w_str_new(&name),
        bases,
        ns,
    ];
    let new_cls = carraytype_new(&args)?;
    ARRAY_TYPE_CACHE.with(|c| c.borrow_mut().push((elem, n, new_cls)));
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
        "struct" | "union" | "array" | "pointer" => Ok(cdata::make_subview(
            meta.proto,
            obj,
            offset,
            meta.element_size,
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
            let bytes = cdata::encode_value(&tc, value)?;
            cdata::cdata_write(obj, offset, &bytes);
            if cdata::is_cdata_instance(value) || unsafe { pyre_object::is_bytes(value) } {
                cdata::keep_ref(obj, &idx.to_string(), value);
            }
            Ok(pyre_object::w_none())
        }
        "struct" | "union" | "array" | "pointer" => {
            if !unsafe { crate::baseobjspace::isinstance_w(value, meta.proto) } {
                return Err(crate::PyError::type_error("incompatible types"));
            }
            let src = cdata::cdata_bytes(value).unwrap_or(&[]);
            let n = meta.element_size.min(src.len());
            cdata::cdata_write(obj, offset, &src[..n]);
            cdata::keep_ref(obj, &idx.to_string(), value);
            Ok(pyre_object::w_none())
        }
        _ => Err(crate::PyError::type_error(
            "assignment to this element type is not supported in this slice",
        )),
    }
}

/// The concrete indices a slice selects over `length`, PySlice-adjusted.
fn slice_index_list(slice: PyObjectRef, length: usize) -> Result<Vec<usize>, crate::PyError> {
    let len = length as i64;
    let as_int = |o: PyObjectRef| -> Option<i64> {
        if o.is_null() || unsafe { pyre_object::is_none(o) } {
            None
        } else if unsafe { pyre_object::is_int(o) } {
            Some(unsafe { pyre_object::w_int_get_value(o) })
        } else {
            None
        }
    };
    let step = as_int(unsafe { pyre_object::w_slice_get_step(slice) }).unwrap_or(1);
    if step == 0 {
        return Err(crate::PyError::value_error("slice step cannot be zero"));
    }
    let (lower, upper) = if step > 0 { (0, len) } else { (-1, len - 1) };
    let clamp = |v: i64| -> i64 {
        let v = if v < 0 { v + len } else { v };
        v.clamp(lower, upper)
    };
    let start = match as_int(unsafe { pyre_object::w_slice_get_start(slice) }) {
        Some(v) => clamp(v),
        None => {
            if step > 0 {
                0
            } else {
                len - 1
            }
        }
    };
    let stop = match as_int(unsafe { pyre_object::w_slice_get_stop(slice) }) {
        Some(v) => clamp(v),
        None => {
            if step > 0 {
                len
            } else {
                -1
            }
        }
    };
    let mut out = Vec::new();
    let mut i = start;
    if step > 0 {
        while i < stop {
            out.push(i as usize);
            i += step;
        }
    } else {
        while i > stop {
            out.push(i as usize);
            i += step;
        }
    }
    Ok(out)
}

fn array_get_slice(obj: PyObjectRef, meta: &ArrayMeta, slice: PyObjectRef) -> PyResult {
    let idxs = slice_index_list(slice, meta.length)?;
    // `c` element arrays slice to `bytes`; other elements slice to a list.
    if cdata::type_code_of(meta.proto).as_deref() == Some("c") {
        let all = cdata::cdata_bytes(obj).unwrap_or(&[]);
        let bytes: Vec<u8> = idxs
            .iter()
            .map(|&i| all.get(i * meta.element_size).copied().unwrap_or(0))
            .collect();
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes));
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
    let items =
        seq_items(value).ok_or_else(|| crate::PyError::type_error("can only assign a sequence"))?;
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
            pyre_object::PY_NULL,
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
    if !unsafe { pyre_object::is_bytes(value) } {
        return Err(crate::PyError::type_error("bytes-like object expected"));
    }
    let src = unsafe { pyre_object::bytesobject::w_bytes_data(value) };
    let size = cdata::cdata_len(obj).unwrap_or(0);
    if src.len() > size {
        return Err(crate::PyError::value_error("byte string too long"));
    }
    cdata::cdata_write(obj, 0, src);
    Ok(pyre_object::w_none())
}

// ── PyCPointerType + _Pointer ──────────────────────────────────────────

fn cpointertype_new(args: &[PyObjectRef]) -> PyResult {
    let cls = crate::builtins::type_descr_new(args)?;
    pointer_init_stginfo(cls)?;
    Ok(cls)
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
    Ok(cdata::make_at_address(proto, ptr, size))
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
            let bytes = unsafe { host_ctypes::borrow_memory(addr as *const u8, element_size) };
            Ok(cdata::decoded_to_pyobject(host_ctypes::decode_type_code(
                &tc, bytes,
            )))
        }
        _ => Ok(cdata::make_at_address(proto, addr, element_size)),
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
            let bytes = cdata::encode_value(&tc, value)?;
            unsafe { host_ctypes::copy_bytes_to_address(addr, &bytes, element_size) };
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
