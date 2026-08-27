//! `_SimpleCData` — the scalar ctypes base type and its byte buffer.
//!
//! Each `_SimpleCData` instance carries a fixed-size `bytearray` under the
//! reserved instance-dict key `"_b_"`; that bytearray's backing `Vec<u8>` is
//! allocated through `malloc_raw` (a non-movable heap box) so its data pointer
//! is stable for `addressof`/`byref`/foreign-call arguments as long as the
//! buffer is never resized — and simple buffers are fixed-size.
//!
//! Per-type metadata (`size`, `align`, ffi type, pointer-ness) is derived
//! lazily from the single-char `_type_` class attribute via `host_env`, rather
//! than cached in a per-type `StgInfo`.  Reading `_type_` off the class also
//! transparently handles user subclasses (`class MyInt(c_int)`).

use super::stginfo::ParamFunc;
use super::type_ns_store;
use pyre_object::PyObjectRef;
use rustpython_host_env::ctypes as host_ctypes;
use std::sync::OnceLock;

/// Reserved instance-dict key holding the backing `bytearray` (root storage,
/// or — for a sub-view — a shared reference to the **root's** bytearray).
const CDATA_BUFFER_KEY: &str = "_b_";
/// Byte offset of a sub-view from the start of the root buffer (absent ⇒ 0).
const BOFF_KEY: &str = "_boff_";
/// View size in bytes (absent ⇒ the whole buffer from `_boff_`).
const BSZ_KEY: &str = "_bsz_";
/// The parent CData object a sub-view was carved from (keeps it alive).
const BBASE_KEY: &str = "_b_base_";
/// Field/array index by which a sub-view is reached from its parent.
const BINDEX_KEY: &str = "_b_index_";
/// Raw address of an external (non-bytearray-backed) view.
const BADDR_KEY: &str = "_baddr_";
/// Lazily-created keepalive dict on a root object.
const OBJECTS_KEY: &str = "_objects_";

/// The instance-dict keys above, which are the object's own storage rather
/// than attributes anybody set.  `PyCData` keeps that state in C struct fields
/// and hands `__dict__` back holding only what the program put there, so
/// `__reduce__` leaves them out of the state it pickles.
const STORAGE_KEYS: [&str; 7] = [
    CDATA_BUFFER_KEY,
    BOFF_KEY,
    BSZ_KEY,
    BBASE_KEY,
    BINDEX_KEY,
    BADDR_KEY,
    OBJECTS_KEY,
];

static CDATA_TYPE_OBJ: OnceLock<usize> = OnceLock::new();
static SIMPLECDATA_TYPE_OBJ: OnceLock<usize> = OnceLock::new();

/// PyPy/CPython/RustPython's private `_CData` base shared by every ctypes
/// value family.  It is discovered as `Structure.__base__` rather than
/// exported from the module namespace.
pub(super) fn cdata_type() -> PyObjectRef {
    *CDATA_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("_CData", init_cdata_type);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        super::finish_cpython_type(tp, "_ctypes", true) as usize
    }) as PyObjectRef
}

fn init_cdata_type(ns: PyObjectRef) {
    for (name, f) in [
        (
            "from_address",
            cdata_from_address as crate::gateway::BuiltinCodeFn,
        ),
        ("from_buffer", cdata_from_buffer),
        ("from_buffer_copy", cdata_from_buffer_copy),
        ("in_dll", cdata_in_dll),
    ] {
        type_ns_store(
            ns,
            name,
            pyre_object::function::w_classmethod_new(crate::make_builtin_function(name, f)),
        );
    }
    for (name, f) in [
        (
            "_objects",
            cdata_objects_get as crate::gateway::BuiltinCodeFn,
        ),
        ("_b_base_", cdata_base_get),
        ("_b_needsfree_", cdata_needsfree_get),
    ] {
        type_ns_store(
            ns,
            name,
            crate::typedef::make_getset_property_named(
                crate::make_builtin_function_with_arity(name, f, 2),
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
                name,
            ),
        );
    }
    // What an `out` parameter hands back once the callee has written it.
    type_ns_store(
        ns,
        "__ctypes_from_outparam__",
        crate::make_builtin_function("__ctypes_from_outparam__", |args| Ok(args[0])),
    );
    // The pickling pair, alongside it in `PyCData_methods`.
    type_ns_store(
        ns,
        "__reduce__",
        crate::make_builtin_function_with_arity("__reduce__", cdata_reduce, 1),
    );
    type_ns_store(
        ns,
        "__setstate__",
        crate::make_builtin_function_with_arity("__setstate__", cdata_setstate, 3),
    );
}

/// `_CData.__reduce__` -- `PyCData_reduce`.
///
/// The pickled state is the attributes the program set plus the raw view
/// bytes, and [`unpickle`] is named as the callable that rebuilds the object
/// from them.  A value that holds a pointer is refused: the address it carries
/// means nothing in the process that reads the pickle back.
fn cdata_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[0];
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let info = super::stginfo::stginfo_of(cls)
        .ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    if super::stginfo::stginfo_flags(info)
        & (super::stginfo::TYPEFLAG_ISPOINTER | super::stginfo::TYPEFLAG_HASPOINTER)
        != 0
    {
        return Err(crate::PyError::value_error(
            "ctypes objects containing pointers cannot be pickled",
        ));
    }
    // The instance dict is read once, before anything allocates: the pairs it
    // hands back are raw words, so a collection between the read and the last
    // store would leave them behind.  Everything built after that is pinned as
    // it is built and read back out of its slot.
    let d = crate::baseobjspace::getdict_native(obj);
    let attributes: Vec<(PyObjectRef, PyObjectRef)> = if d.is_null() {
        Vec::new()
    } else {
        unsafe { pyre_object::dictmultiobject::w_dict_items(d) }
            .into_iter()
            .filter(|&(key, _)| {
                !unsafe { pyre_object::is_str(key) }
                    || !STORAGE_KEYS.contains(&unsafe { pyre_object::w_str_get_value(key) })
            })
            .collect()
    };
    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = roots.base();
    let _ = roots.pin_root(obj);
    let cls_slot = obj_slot + 1;
    let _ = roots.pin_root(cls);
    let attribute_base = cls_slot + 1;
    for &(key, value) in &attributes {
        let _ = roots.pin_root(key);
        let _ = roots.pin_root(value);
    }
    let state_slot = attribute_base + 2 * attributes.len();
    let _ = roots.pin_root(pyre_object::w_dict_new());
    for i in 0..attributes.len() {
        crate::baseobjspace::setitem(
            roots.get(state_slot),
            roots.get(attribute_base + 2 * i),
            roots.get(attribute_base + 2 * i + 1),
        )?;
    }
    let buffer = cdata_bytes_object(roots.get(obj_slot))
        .ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let buffer_slot = state_slot + 1;
    let _ = roots.pin_root(buffer);
    let inner = pyre_object::w_tuple_new(vec![roots.get(state_slot), roots.get(buffer_slot)]);
    let inner_slot = buffer_slot + 1;
    let _ = roots.pin_root(inner);
    let outer = pyre_object::w_tuple_new(vec![roots.get(cls_slot), roots.get(inner_slot)]);
    let outer_slot = inner_slot + 1;
    let _ = roots.pin_root(outer);
    Ok(pyre_object::w_tuple_new(vec![
        unpickle_function(),
        roots.get(outer_slot),
    ]))
}

/// `_CData.__setstate__` -- `PyCData_setstate`.
///
/// The bytes are written over the view and truncated to it, so a state saved
/// from a wider object fills what fits; the dictionary is merged into the
/// attributes rather than replacing them.
fn cdata_setstate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[0];
    if !unsafe { pyre_object::is_dict(args[1]) } {
        return Err(crate::PyError::type_error(format!(
            "argument 1 must be dict, not {}",
            crate::gateway::short_type_name(args[1])
        )));
    }
    let source = crate::typedef::buffer_as_bytes_like(args[2])?.ok_or_else(|| {
        crate::PyError::type_error(format!(
            "argument 2 must be str, not {}",
            crate::gateway::short_type_name(args[2])
        ))
    })?;
    cdata_write(obj, 0, unsafe {
        pyre_object::bytesobject::bytes_like_data(source)
    });
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return Err(crate::PyError::type_error(
            "ctypes instance has no instance dict",
        ));
    }
    let items = unsafe { pyre_object::dictmultiobject::w_dict_items(args[1]) };
    let roots = pyre_object::gc_roots::push_roots();
    let d_slot = roots.base();
    let _ = roots.pin_root(d);
    let item_base = d_slot + 1;
    for &(key, value) in &items {
        let _ = roots.pin_root(key);
        let _ = roots.pin_root(value);
    }
    for i in 0..items.len() {
        crate::baseobjspace::setitem(
            roots.get(d_slot),
            roots.get(item_base + 2 * i),
            roots.get(item_base + 2 * i + 1),
        )?;
    }
    Ok(pyre_object::w_none())
}

static UNPICKLE_FUNCTION: OnceLock<usize> = OnceLock::new();

/// `_ctypes._unpickle`, the callable [`cdata_reduce`] names.
pub(super) fn unpickle_function() -> PyObjectRef {
    *UNPICKLE_FUNCTION
        .get_or_init(|| crate::make_builtin_function_with_arity("_unpickle", unpickle, 2) as usize)
        as PyObjectRef
}

/// `_ctypes._unpickle` -- `_ctypes__unpickle_impl`.
///
/// Allocate an instance of the pickled class without running `__init__` -- the
/// state is what decides its contents -- and hand it the state, so a subclass
/// that writes its own `__setstate__` is the one that reads the state back.
fn unpickle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let cls = args[0];
    let state = args[1];
    let new = crate::baseobjspace::getattr_str(cls, "__new__")?;
    let obj = crate::call::call_function_impl_result(new, &[cls])?;
    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = roots.base();
    let _ = roots.pin_root(obj);
    let state_slot = obj_slot + 1;
    let _ = roots.pin_root(state);
    let setstate = crate::baseobjspace::getattr_str(roots.get(obj_slot), "__setstate__")?;
    let setstate_slot = state_slot + 1;
    let _ = roots.pin_root(setstate);
    let state_args = crate::baseobjspace::fixedview(roots.get(state_slot), -1)?;
    crate::call::call_function_impl_result(roots.get(setstate_slot), &state_args)?;
    Ok(roots.get(obj_slot))
}

pub(super) fn cdata_in_dll(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 3 {
        return Err(crate::PyError::type_error(
            "in_dll() needs a library and name",
        ));
    }
    let cls = args[0];
    let size = ctype_size_of(cls).ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let handle_obj = crate::baseobjspace::getattr_str(args[1], "_handle")?;
    let handle = crate::baseobjspace::int_w(handle_obj)? as usize;
    if !unsafe { pyre_object::is_str(args[2]) } {
        return Err(crate::PyError::type_error("name must be a string"));
    }
    let name = unsafe { pyre_object::w_str_get_value(args[2]) };
    if name == "Py_OptimizeFlag" {
        let optimize = crate::importing::get_sys_module("sys")
            .and_then(|sys| crate::baseobjspace::getattr_str(sys, "flags").ok())
            .and_then(|flags| crate::baseobjspace::getattr_str(flags, "optimize").ok())
            .unwrap_or_else(|| pyre_object::w_int_new(0));
        return crate::call::type_call_instantiate(cls, &[optimize]);
    }
    if let Some(pointer_variable) =
        crate::module::imp::interp_imp::frozen_abi_pointer_variable(name)
    {
        return Ok(make_at_address(cls, pointer_variable, size, args[1]));
    }
    let address = super::interp_ctypes::lookup_symbol(handle, name.as_bytes())
        .map_err(|_| crate::PyError::value_error(format!("symbol '{name}' not found")))?;
    Ok(make_at_address(cls, address, size, args[1]))
}

fn cdata_objects_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let d = crate::baseobjspace::getdict_native(args[1]);
    Ok(if d.is_null() {
        pyre_object::w_none()
    } else {
        unsafe { pyre_object::w_dict_getitem_str(d, OBJECTS_KEY) }
            .unwrap_or_else(pyre_object::w_none)
    })
}

fn cdata_base_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let d = crate::baseobjspace::getdict_native(args[1]);
    Ok(if d.is_null() {
        pyre_object::w_none()
    } else {
        unsafe { pyre_object::w_dict_getitem_str(d, BBASE_KEY) }.unwrap_or_else(pyre_object::w_none)
    })
}

fn cdata_needsfree_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(pyre_object::w_int_new(owns_buffer(args[1]) as i64))
}

pub(super) fn cdata_from_address(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("from_address() missing address"));
    }
    let cls = args[0];
    let size = ctype_size_of(cls)
        .filter(|&n| n != 0)
        .ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let address = crate::baseobjspace::int_w(args[1])? as usize;
    Ok(make_at_address(cls, address, size, pyre_object::PY_NULL))
}

fn cdata_from_buffer_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "from_buffer_copy() missing source",
        ));
    }
    let cls = args[0];
    let size = ctype_size_of(cls)
        .filter(|&n| n != 0)
        .ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let offset = if let Some(&o) = args.get(2) {
        crate::baseobjspace::int_w(o)?
    } else {
        0
    };
    if offset < 0 {
        return Err(crate::PyError::value_error("offset cannot be negative"));
    }
    let source = crate::typedef::buffer_as_bytes_like(args[1])?
        .ok_or_else(|| crate::PyError::type_error("a bytes-like object is required"))?;
    let all = unsafe { pyre_object::bytesobject::bytes_like_data(source) };
    let offset = offset as usize;
    if offset > all.len() || size > all.len() - offset {
        return Err(crate::PyError::value_error(format!(
            "Buffer size too small ({} instead of at least {} bytes)",
            all.len().saturating_sub(offset),
            size
        )));
    }
    let copied = all[offset..offset + size].to_vec();
    new_cdata_obj_from_bytes(cls, size, &copied)
}

fn cdata_from_buffer(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("from_buffer() missing source"));
    }
    let cls = args[0];
    let size = ctype_size_of(cls)
        .filter(|&n| n != 0)
        .ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let offset = if let Some(&o) = args.get(2) {
        crate::baseobjspace::int_w(o)?
    } else {
        0
    };
    if offset < 0 {
        return Err(crate::PyError::value_error("offset cannot be negative"));
    }

    // Acquire and retain a real memoryview so the exporter's resize lock and
    // lifetime follow the buffer protocol, as PyCData_FromBaseObj requires.
    let view_obj = crate::builtins::w_memoryview_new(args[1])?;
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let view_obj = pyre_object::gc_roots::pin_root(view_obj);
    let view = unsafe { pyre_object::memoryview::w_memoryview_view(view_obj) };
    if view.readonly() {
        return Err(crate::PyError::type_error(
            "underlying buffer is not writable",
        ));
    }
    if view.ndim() != 1 || unsafe { view.stride0() } != view.itemsize() {
        return Err(crate::PyError::type_error(
            "underlying buffer is not C contiguous",
        ));
    }
    let length = unsafe { view.length() }.max(0) as usize;
    let offset = offset as usize;
    if offset > length || size > length - offset {
        return Err(crate::PyError::value_error(format!(
            "Buffer size too small ({} instead of at least {} bytes)",
            length.saturating_sub(offset),
            size
        )));
    }
    let view_offset = unsafe { view.offset() }.max(0) as usize;
    let backing = unsafe { view.backing().as_bytes_mut() }
        .ok_or_else(|| crate::PyError::type_error("underlying buffer is not writable"))?;
    let start = view_offset.saturating_add(offset);
    if start > backing.len() || size > backing.len() - start {
        return Err(crate::PyError::value_error("Buffer size too small"));
    }
    let address = unsafe { backing.as_mut_ptr().add(start) } as usize;
    let obj = make_at_address(cls, address, size, pyre_object::PY_NULL);
    let rooted_view = pyre_object::gc_roots::shadow_stack_get(sp);
    keep_ref(obj, "ffffffff", rooted_view);
    Ok(obj)
}

/// The native `_SimpleCData` type object (cached, `hasdict=true`).
pub(super) fn simplecdata_type() -> PyObjectRef {
    *SIMPLECDATA_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_base(
            "_SimpleCData",
            init_simplecdata_type,
            cdata_type(),
        );
        unsafe {
            pyre_object::typeobject::w_type_set_hasdict(tp, true);
            // PyPy `primitive.py:SimpleType(_CDataMeta)` / CPython 3.14
            // `CREATE_TYPE(... PyCSimpleType_Type ...)`.
            (*tp).w_class = super::metaclass::pycsimpletype_type();
        }
        super::finish_cpython_type(tp, "_ctypes", true) as usize
    }) as PyObjectRef
}

fn init_simplecdata_type(ns: PyObjectRef) {
    type_ns_store(
        ns,
        "__new__",
        crate::typedef::make_new_descr(simplecdata_new),
    );
    // A scalar `out` parameter hands back the value rather than the box; one
    // whose type is a user subclass hands back the instance, because the
    // subclass is the thing the caller asked for.
    type_ns_store(
        ns,
        "__ctypes_from_outparam__",
        crate::make_builtin_function("__ctypes_from_outparam__", |args| {
            let obj = args[0];
            if super::funcptr::is_simple_subclass(unsafe { pyre_object::w_instance_get_type(obj) })
            {
                return Ok(obj);
            }
            value_getter(&[pyre_object::PY_NULL, obj])
        }),
    );
    type_ns_store(
        ns,
        "__repr__",
        crate::make_builtin_function("__repr__", simplecdata_repr),
    );
    // `value` — data descriptor: getter decodes the buffer, setter encodes.
    let value_getter = crate::make_builtin_function_with_arity("value", value_getter, 2);
    let value_setter = crate::make_builtin_function_with_arity("value", value_setter, 3);
    type_ns_store(
        ns,
        "value",
        crate::typedef::make_getset_property_named(
            value_getter,
            value_setter,
            crate::make_builtin_function_with_arity(
                "value",
                |_args| Err(crate::PyError::type_error("can't delete attribute")),
                2,
            ),
            "value",
        ),
    );
    // `from_param` — a classmethod the metaclass provides in CPython; the
    // package's `_reset_cache` reads `c_wchar_p.from_param` at import.  The
    // slice marshals arguments directly (§5.2) rather than via `from_param`,
    // so this is an identity stub that only has to exist and be gettable.
    type_ns_store(
        ns,
        "from_param",
        pyre_object::function::w_classmethod_new(crate::make_builtin_function(
            "from_param",
            simplecdata_from_param,
        )),
    );
}

fn simplecdata_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[0];
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let bases = unsafe { pyre_object::typeobject::w_type_get_bases(cls) };
    let direct = !bases.is_null()
        && unsafe { pyre_object::w_tuple_getitem(bases, 0) }
            .is_some_and(|base| base == simplecdata_type());
    if !direct {
        let name = unsafe { pyre_object::typeobject::w_type_get_name(cls) };
        return Ok(pyre_object::w_str_new(&format!(
            "<{name} object at {}>",
            crate::display::repr_addr(obj as usize)
        )));
    }
    let tc = type_code_of(cls).ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    if tc == "O" && host_ctypes::read_pointer_from_buffer(cdata_bytes(obj).unwrap_or(&[])) == 0 {
        let name = unsafe { pyre_object::typeobject::w_type_get_name(cls) };
        return Ok(pyre_object::w_str_new(&format!("{name}(<NULL>)")));
    }
    let value = if tc == "O" {
        host_ctypes::read_pointer_from_buffer(cdata_bytes(obj).unwrap_or(&[])) as PyObjectRef
    } else {
        decode_slot(&tc, cdata_bytes(obj).unwrap_or(&[]))
    };
    let rendered = unsafe { crate::display::py_repr_wtf8(value) }?;
    let name = unsafe { pyre_object::typeobject::w_type_get_name(cls) };
    Ok(pyre_object::w_str_from_wtf8(crate::display::wtf8_format!(
        name, "(", rendered, ")"
    )))
}

/// `_SimpleCData.from_param(cls, value)` — identity stub (see caller note).
fn simplecdata_from_param(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    simple_from_param(args)
}

/// PyCSimpleType.from_param: convert to a same-typed temporary when needed,
/// then return the same `CArgObject` carrier family as `byref()`.
pub(super) fn simple_from_param(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("from_param() missing value"));
    }
    let cls = args[0];
    let value = args[1];
    let converted = if unsafe { crate::baseobjspace::isinstance_w(value, cls) } {
        value
    } else {
        let tc = type_code_of(cls).ok_or_else(|| crate::PyError::type_error("abstract class"))?;
        new_simplecdata_obj(cls, &tc, Some(value))?
    };
    // A converted temporary is reachable only from here, and both the address
    // lookup and the carrier allocate — the address is into its buffer, so the
    // instance has to outlive them.  It does not move, so one pin is enough.
    let _roots = pyre_object::gc_roots::push_roots();
    let converted = pyre_object::gc_roots::pin_root(converted);
    let addr = cdata_addr(converted)
        .ok_or_else(|| crate::PyError::type_error("ctypes instance has no buffer"))?;
    Ok(super::interp_ctypes::make_carg(addr, converted))
}

/// `_SimpleCData.__new__(cls, value=None)`.
fn simplecdata_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() || !unsafe { pyre_object::is_type(args[0]) } {
        return Err(crate::PyError::type_error(
            "_SimpleCData.__new__(): not enough arguments",
        ));
    }
    let cls = args[0];
    let tc = type_code_of(cls).ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let (pos, _kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    let value = pos.first().copied();
    new_simplecdata_obj(cls, &tc, value)
}

/// Build a fresh `_SimpleCData` instance of `cls` with type code `tc`,
/// optionally initialised from a Python `value`.
pub(super) fn new_simplecdata_obj(
    cls: PyObjectRef,
    tc: &str,
    value: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let size = host_ctypes::simple_type_size(tc).ok_or_else(invalid_type_code_error)?;
    let obj = new_cdata_obj_from_bytes(cls, size, &[])?;
    // The new instance is reachable only from this frame while `encode_value_into`
    // runs, and that call can run Python.  It does not move, so it is pinned once
    // and never re-read; the bytearray rides along through the instance dict.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj = pyre_object::gc_roots::pin_root(obj);
    let ba = cdata_buffer(obj).expect("new cdata object has a backing buffer");
    // Encode after the instance exists so a `char*` keepalive can attach to it.
    if let Some(v) = value {
        // `v` is an arbitrary object, so under `"O"` it can be a `list` or a
        // `dict`, both of which move: pin it before the encode and read the slot
        // back where it is stored.
        let v_slot = pyre_object::gc_roots::shadow_stack_len();
        let v = pyre_object::gc_roots::pin_root(v);
        let mut bytes = encode_value_into(tc, v, obj, "0")?;
        if unsafe { crate::baseobjspace::lookup_in_type(cls, "_swappedbytes_") }.is_some() {
            bytes.reverse();
        }
        let n = bytes.len().min(size);
        unsafe {
            pyre_object::w_bytearray_data_mut(ba)[..n].copy_from_slice(&bytes[..n]);
        }
        if matches!(tc, "z" | "Z" | "O") {
            let d = crate::baseobjspace::getdict_native(obj);
            let v = pyre_object::gc_roots::shadow_stack_get(v_slot);
            unsafe { pyre_object::w_dict_setitem_str(d, OBJECTS_KEY, v) };
        }
    }
    Ok(obj)
}

pub(super) fn new_cdata_obj_from_bytes(
    cls: PyObjectRef,
    size: usize,
    bytes: &[u8],
) -> Result<PyObjectRef, crate::PyError> {
    // Until the store below links them, the bytearray and the instance are
    // reachable only from this frame, and `w_instance_new`, `getdict_native`
    // and the store each allocate.  Neither kind moves, so one pin apiece keeps
    // them live and no word has to be re-read.
    let ba = pyre_object::w_bytearray_new(size);
    let _roots = pyre_object::gc_roots::push_roots();
    let ba = pyre_object::gc_roots::pin_root(ba);
    let obj = pyre_object::w_instance_new(cls);
    let obj = pyre_object::gc_roots::pin_root(obj);
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return Err(crate::PyError::type_error(
            "ctypes instance has no instance dict",
        ));
    }
    unsafe {
        pyre_object::w_dict_setitem_str(d, CDATA_BUFFER_KEY, ba);
        let n = bytes.len().min(size);
        pyre_object::w_bytearray_data_mut(ba)[..n].copy_from_slice(&bytes[..n]);
    }
    Ok(obj)
}

pub(super) fn ctype_size_of(cls: PyObjectRef) -> Option<usize> {
    super::stginfo::stginfo_of(cls)
        .map(super::stginfo::stginfo_size)
        .or_else(|| type_code_of(cls).and_then(|tc| host_ctypes::simple_type_size(&tc)))
        .or_else(|| super::funcptr::is_funcptr_type(cls).then(host_ctypes::pointer_size))
}

/// `_SimpleCData.value` getter — `(descr, instance)`.
fn value_getter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[1];
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let tc = type_code_of(cls).ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let mut bytes = cdata_bytes(obj)
        .ok_or_else(|| crate::PyError::type_error("ctypes instance has no buffer"))?;
    if tc == "O" {
        let address = host_ctypes::read_pointer_from_buffer(bytes);
        return Ok(if address == 0 {
            pyre_object::w_none()
        } else {
            address as PyObjectRef
        });
    }
    // A BSTR carries no swapped spelling — `X` has no entry in the byte-order
    // tables — so it is read before the reversal below could reach it.
    #[cfg(windows)]
    if tc == "X" {
        return Ok(decode_slot(&tc, bytes));
    }
    let swapped = unsafe { crate::baseobjspace::lookup_in_type(cls, "_swappedbytes_") }.is_some();
    let owned;
    if swapped {
        owned = bytes.iter().rev().copied().collect::<Vec<_>>();
        bytes = &owned;
    }
    Ok(decode_slot(&tc, bytes))
}

/// `_SimpleCData.value` setter — `(descr, instance, value)`.
fn value_setter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[1];
    let value = args[2];
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let tc = type_code_of(cls).ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    // `value` is an arbitrary object, so under `"O"` it can be a `list` or a
    // `dict`, both of which move, and `encode_value_into` can run Python: pin it
    // and read the slot back where it is stored.
    let _roots = pyre_object::gc_roots::push_roots();
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let value = pyre_object::gc_roots::pin_root(value);
    let mut bytes = encode_value_into(&tc, value, obj, "0")?;
    if unsafe { crate::baseobjspace::lookup_in_type(cls, "_swappedbytes_") }.is_some() {
        bytes.reverse();
    }
    // `BSTR_set` frees what the slot held only once the new string exists, so
    // a conversion that refuses its value leaves the previous one readable.
    release_bstr_slot(&tc, cdata_addr(obj).unwrap_or(0));
    cdata_write(obj, 0, &bytes);
    if matches!(tc.as_str(), "z" | "Z" | "O") {
        let d = crate::baseobjspace::getdict_native(obj);
        let value = pyre_object::gc_roots::shadow_stack_get(value_slot);
        unsafe { pyre_object::w_dict_setitem_str(d, OBJECTS_KEY, value) };
    }
    Ok(pyre_object::w_none())
}

// ── buffer helpers (b_ptr / b_size / b_base equivalents) ───────────────

/// Read a `usize`-valued reserved key off the instance dict.
fn dict_usize(obj: PyObjectRef, key: &str) -> Option<usize> {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return None;
    }
    match unsafe { pyre_object::w_dict_getitem_str(d, key) } {
        Some(o) if unsafe { pyre_object::is_int(o) } => {
            Some(unsafe { pyre_object::w_int_get_value(o) } as usize)
        }
        _ => None,
    }
}

/// The sub-view byte offset from the root buffer (0 for a root object).
fn boff(obj: PyObjectRef) -> usize {
    dict_usize(obj, BOFF_KEY).unwrap_or(0)
}

/// The explicit view size, if this is a sub-view / external view.
fn bsz(obj: PyObjectRef) -> Option<usize> {
    dict_usize(obj, BSZ_KEY)
}

/// The raw address of an external (`"_baddr_"`) view, if any.
fn baddr(obj: PyObjectRef) -> Option<usize> {
    dict_usize(obj, BADDR_KEY)
}

/// The backing `bytearray` stored under `"_b_"` (the root's, for a sub-view).
pub(super) fn cdata_buffer(obj: PyObjectRef) -> Option<PyObjectRef> {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return None;
    }
    unsafe { pyre_object::w_dict_getitem_str(d, CDATA_BUFFER_KEY) }
}

/// `b_ptr` — the address of the view's first byte (stable for the instance's
/// lifetime).  Re-read at each use; never cached across an allocation.
pub(super) fn cdata_addr(obj: PyObjectRef) -> Option<usize> {
    if let Some(ba) = cdata_buffer(obj) {
        Some(unsafe { pyre_object::w_bytearray_data(ba).as_ptr() as usize } + boff(obj))
    } else {
        baddr(obj).map(|a| a + boff(obj))
    }
}

/// `b_ptr[..b_size]` — the view bytes.
pub(crate) fn cdata_bytes(obj: PyObjectRef) -> Option<&'static [u8]> {
    if let Some(ba) = cdata_buffer(obj) {
        let data = unsafe { pyre_object::w_bytearray_data(ba) };
        let off = boff(obj).min(data.len());
        let sz = bsz(obj).unwrap_or(data.len() - off);
        let end = (off + sz).min(data.len());
        Some(&data[off..end])
    } else if let Some(addr) = baddr(obj) {
        let sz = bsz(obj).unwrap_or(0);
        Some(unsafe { host_ctypes::borrow_memory((addr + boff(obj)) as *const u8, sz) })
    } else {
        None
    }
}

/// `b_ptr[..b_size]` boxed as a `bytes`, or `None` for an instance with no
/// view.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`): the
/// address arm borrows foreign memory through `host_ctypes::borrow_memory`,
/// which has no lifted model, and the `&[u8]` [`cdata_bytes`] hands back is
/// two words where a residual result is one.  Every caller boxes the slice
/// immediately, so the box happens inside the boundary and a single
/// null-niche `Option<PyObjectRef>` crosses it.
#[majit_macros::dont_look_inside]
pub(crate) fn cdata_bytes_object(obj: PyObjectRef) -> Option<PyObjectRef> {
    // PyPy's CData buffer path is reached only through a CData instance's
    // buffer implementation.  `buffer_as_bytes_like` probes this helper for
    // arbitrary objects, so preserve that owner check here before any CData
    // dict field (`_b_`, `_b_addr_`, ...) is read.
    if !is_cdata_instance(obj) {
        return None;
    }
    cdata_bytes(obj).map(pyre_object::bytesobject::w_bytes_from_bytes)
}

/// `b_size` — the view length.
pub(super) fn cdata_len(obj: PyObjectRef) -> Option<usize> {
    if let Some(ba) = cdata_buffer(obj) {
        Some(bsz(obj).unwrap_or_else(|| unsafe { pyre_object::w_bytearray_len(ba) } - boff(obj)))
    } else if baddr(obj).is_some() {
        Some(bsz(obj).unwrap_or(0))
    } else {
        None
    }
}

/// Buffer-protocol metadata for a bytearray-backed CData view.  The returned
/// bytearray is the root storage and `offset`/`length` select this object's
/// live window, matching PyPy's `SubBuffer` representation.
pub(crate) fn cdata_buffer_view(
    obj: PyObjectRef,
) -> Option<(PyObjectRef, usize, usize, String, usize, Vec<usize>)> {
    if !is_cdata_instance(obj) {
        return None;
    }
    let ba = cdata_buffer(obj)?;
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let info = super::stginfo::stginfo_of(cls);
    let kind = info
        .map(super::stginfo::stginfo_paramfunc)
        .unwrap_or(ParamFunc::Other);
    let shape = ctype_shape(cls);
    let leaf = ctype_leaf(cls);
    let is_funcptr =
        unsafe { crate::baseobjspace::isinstance_w(obj, super::funcptr::cfuncptr_type()) };
    let itemsize = if is_funcptr {
        host_ctypes::pointer_size()
    } else {
        super::stginfo::field_size_of(leaf).unwrap_or(0)
    };
    let format = if is_funcptr {
        "X{}".to_string()
    } else if kind == ParamFunc::Union {
        "B".to_string()
    } else {
        ctype_pep3118_format(cls, None)
    };
    Some((ba, boff(obj), cdata_len(obj)?, format, itemsize, shape))
}

fn ctype_shape(mut cls: PyObjectRef) -> Vec<usize> {
    let mut shape = Vec::new();
    while let Some(info) = super::stginfo::stginfo_of(cls) {
        if super::stginfo::stginfo_paramfunc(info) != ParamFunc::Array {
            break;
        }
        shape.push(super::stginfo::stginfo_length(info));
        let Some(proto) = super::stginfo::stginfo_proto(info) else {
            break;
        };
        cls = proto;
    }
    shape
}

fn ctype_leaf(mut cls: PyObjectRef) -> PyObjectRef {
    while let Some(info) = super::stginfo::stginfo_of(cls) {
        if super::stginfo::stginfo_paramfunc(info) != ParamFunc::Array {
            break;
        }
        let Some(proto) = super::stginfo::stginfo_proto(info) else {
            break;
        };
        cls = proto;
    }
    cls
}

pub(super) fn ctype_pep3118_format(cls: PyObjectRef, forced_big: Option<bool>) -> String {
    let info = super::stginfo::stginfo_of(cls);
    let kind = info
        .map(super::stginfo::stginfo_paramfunc)
        .unwrap_or(ParamFunc::Other);
    match kind {
        ParamFunc::Array => info
            .and_then(super::stginfo::stginfo_proto)
            .map(|proto| ctype_pep3118_format(proto, forced_big))
            .unwrap_or_else(|| "B".to_string()),
        ParamFunc::Pointer => {
            if let Some(snapshot) = info.and_then(super::stginfo::stginfo_format) {
                return snapshot;
            }
            let inner = info
                .and_then(super::stginfo::stginfo_proto)
                .map(|proto| {
                    let shape = ctype_shape(proto);
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
                    format!("{prefix}{}", ctype_pep3118_format(proto, forced_big))
                })
                .unwrap_or_else(|| "B".to_string());
            format!("&{inner}")
        }
        ParamFunc::Struct => struct_pep3118_format(cls),
        ParamFunc::Union => "B".to_string(),
        _ => {
            let Some(code) = type_code_of(cls).and_then(|s| s.chars().next()) else {
                return "B".to_string();
            };
            let big = forced_big.unwrap_or_else(|| {
                unsafe { crate::baseobjspace::lookup_in_type(cls, "_swappedbytes_") }.is_some()
                    ^ cfg!(target_endian = "big")
            });
            format!(
                "{}{}",
                if big { '>' } else { '<' },
                host_ctypes::simple_type_pep3118_code(code),
            )
        }
    }
}

fn struct_pep3118_format(cls: PyObjectRef) -> String {
    let big = super::stginfo::stginfo_of(cls).is_some_and(super::stginfo::stginfo_big_endian);
    let Some(fields) = (unsafe { crate::baseobjspace::lookup_in_type(cls, "_fields_") }) else {
        return "B".to_string();
    };
    let items = if unsafe { pyre_object::is_tuple(fields) } {
        (0..unsafe { pyre_object::w_tuple_len(fields) } as i64)
            .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(fields, i) })
            .collect::<Vec<_>>()
    } else if unsafe { pyre_object::is_list(fields) } {
        (0..unsafe { pyre_object::w_list_len(fields) } as i64)
            .filter_map(|i| unsafe { pyre_object::w_list_getitem(fields, i) })
            .collect::<Vec<_>>()
    } else {
        return "B".to_string();
    };
    let mut format = String::from("T{");
    let mut last_end = 0usize;
    for field in items {
        let Some(name_obj) = (unsafe { pyre_object::w_tuple_getitem(field, 0) }) else {
            continue;
        };
        let Some(field_type) = (unsafe { pyre_object::w_tuple_getitem(field, 1) }) else {
            continue;
        };
        if !unsafe { pyre_object::is_str(name_obj) } {
            continue;
        }
        let name = unsafe { pyre_object::w_str_get_value(name_obj) };
        let Some(descr) = (unsafe { crate::baseobjspace::lookup_in_type(cls, name) }) else {
            continue;
        };
        let dd = crate::baseobjspace::getdict_native(descr);
        let integer = |key: &str| {
            unsafe { pyre_object::w_dict_getitem_str(dd, key) }
                .filter(|value| unsafe { pyre_object::is_int(*value) })
                .map(|value| unsafe { pyre_object::w_int_get_value(value) }.max(0) as usize)
                .unwrap_or(0)
        };
        let offset = integer("byte_offset");
        let size = integer("byte_size");
        let padding = offset.saturating_sub(last_end);
        if padding > 0 {
            if padding != 1 {
                format.push_str(&padding.to_string());
            }
            format.push('x');
        }
        let shape = ctype_shape(field_type);
        if !shape.is_empty() {
            format.push('(');
            format.push_str(
                &shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            format.push(')');
        }
        format.push_str(&ctype_pep3118_format(field_type, Some(big)));
        format.push(':');
        format.push_str(name);
        format.push(':');
        last_end = last_end.max(offset.saturating_add(size));
    }
    let total = super::stginfo::stginfo_of(cls)
        .map(super::stginfo::stginfo_size)
        .unwrap_or(0);
    let padding = total.saturating_sub(last_end);
    if padding > 0 {
        if padding != 1 {
            format.push_str(&padding.to_string());
        }
        format.push('x');
    }
    format.push('}');
    format
}

/// Overwrite `bytes` at view-relative offset `off`.
pub(super) fn cdata_write(obj: PyObjectRef, off: usize, bytes: &[u8]) {
    if let Some(ba) = cdata_buffer(obj) {
        let start = boff(obj) + off;
        let cap = unsafe { pyre_object::w_bytearray_len(ba) };
        if start >= cap {
            return;
        }
        let n = bytes.len().min(cap - start);
        unsafe {
            pyre_object::w_bytearray_data_mut(ba)[start..start + n].copy_from_slice(&bytes[..n]);
        }
    } else if let Some(addr) = baddr(obj) {
        let sz = bsz(obj).unwrap_or(0);
        if off >= sz {
            return;
        }
        let n = bytes.len().min(sz - off);
        let dst = unsafe { host_ctypes::borrow_memory_mut((addr + off) as *mut u8, n) };
        dst.copy_from_slice(&bytes[..n]);
    }
}

// ── sub-views, keepalive, and the cdata-instance predicate ─────────────

/// Whether `obj` owns its buffer (a root object, not a sub-view or external
/// view) — the precondition for `resize`.
pub(super) fn owns_buffer(obj: PyObjectRef) -> bool {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return false;
    }
    let has = |k: &str| unsafe { pyre_object::w_dict_getitem_str(d, k) }.is_some();
    has(CDATA_BUFFER_KEY) && !has(BBASE_KEY) && !has(BADDR_KEY)
}

/// Whether `obj` is an instance of the common `_CData` base, matching PyPy's
/// CData inheritance test without a parallel per-thread type registry.
pub(super) fn is_cdata_instance(obj: PyObjectRef) -> bool {
    !obj.is_null() && unsafe { crate::baseobjspace::isinstance_w(obj, cdata_type()) }
}

/// A field/element sub-view of `parent` at `field_offset`, aliasing its memory
/// (`PyCData_FromBaseObj`): bytearray-backed → shares the root bytearray with an
/// accumulated `_boff_`; address-backed → a fresh external view.
pub(super) fn make_subview(
    proto: PyObjectRef,
    parent: PyObjectRef,
    field_offset: usize,
    size: usize,
) -> PyObjectRef {
    // The fresh instance is reachable only from this frame, so it is pinned for
    // its lifetime; it does not move, so its word is never re-read.  An instance
    // dictionary does move, and every store below allocates, so that is pinned
    // and read back at each store.  Each value is built into a local first: call
    // arguments evaluate left to right, so an inline allocating value would
    // consume a pre-move dictionary word.
    let _roots = pyre_object::gc_roots::push_roots();
    let inst = pyre_object::w_instance_new(proto);
    let inst = pyre_object::gc_roots::pin_root(inst);
    let d = crate::baseobjspace::getdict_native(inst);
    if d.is_null() {
        return inst;
    }
    let _ = pyre_object::gc_roots::pin_root(d);
    let d_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    unsafe {
        if let Some(ba) = cdata_buffer(parent) {
            pyre_object::w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(d_slot),
                CDATA_BUFFER_KEY,
                ba,
            );
            let w_offset = pyre_object::w_int_new((boff(parent) + field_offset) as i64);
            pyre_object::w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(d_slot),
                BOFF_KEY,
                w_offset,
            );
        } else if let Some(addr) = baddr(parent) {
            let w_address = pyre_object::w_int_new((addr + field_offset) as i64);
            pyre_object::w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(d_slot),
                BADDR_KEY,
                w_address,
            );
        }
        let w_size = pyre_object::w_int_new(size as i64);
        pyre_object::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(d_slot),
            BSZ_KEY,
            w_size,
        );
        pyre_object::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(d_slot),
            BBASE_KEY,
            parent,
        );
    }
    inst
}

pub(super) fn make_indexed_subview(
    proto: PyObjectRef,
    parent: PyObjectRef,
    field_offset: usize,
    size: usize,
    index: usize,
) -> PyObjectRef {
    let view = make_subview(proto, parent, field_offset, size);
    // `make_subview` dropped its own root bracket on the way out, so the view is
    // reachable only from this frame again across the lookup and the store.
    let _roots = pyre_object::gc_roots::push_roots();
    let view = pyre_object::gc_roots::pin_root(view);
    let d = crate::baseobjspace::getdict_native(view);
    if !d.is_null() {
        let _ = pyre_object::gc_roots::pin_root(d);
        let d_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_index = pyre_object::w_int_new(index as i64);
        unsafe {
            pyre_object::w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(d_slot),
                BINDEX_KEY,
                w_index,
            );
        }
    }
    view
}

/// An `address`-backed instance of `proto` viewing `size` bytes of external
/// memory (`PyCData::at_address`) — the pyre form of a pointer dereference.
/// `base` (the pointer object the address came from) is retained under
/// `_b_base_` so its `_objects_` keepalive chain outlives the returned view.
pub(super) fn make_at_address(
    proto: PyObjectRef,
    address: usize,
    size: usize,
    base: PyObjectRef,
) -> PyObjectRef {
    // The fresh instance is reachable only from this frame across the lookup and
    // the stores below; it does not move, so one pin covers it.
    let _roots = pyre_object::gc_roots::push_roots();
    let inst = pyre_object::w_instance_new(proto);
    let inst = pyre_object::gc_roots::pin_root(inst);
    let d = crate::baseobjspace::getdict_native(inst);
    if !d.is_null() {
        let _ = pyre_object::gc_roots::pin_root(d);
        let d_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        unsafe {
            let w_address = pyre_object::w_int_new(address as i64);
            pyre_object::w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(d_slot),
                BADDR_KEY,
                w_address,
            );
            let w_size = pyre_object::w_int_new(size as i64);
            pyre_object::w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(d_slot),
                BSZ_KEY,
                w_size,
            );
            if !base.is_null() && !pyre_object::is_none(base) {
                pyre_object::w_dict_setitem_str(
                    pyre_object::gc_roots::shadow_stack_get(d_slot),
                    BBASE_KEY,
                    base,
                );
            }
        }
    }
    inst
}

/// Keep `obj` alive for the lifetime of the buffer that `anchor` views, by
/// storing it under `key` in the ultimate root's `"_objects_"` dict.
pub(super) fn keep_ref(anchor: PyObjectRef, key: &str, obj: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // Walk `_b_base_` up to the owning root.
    let mut root = anchor;
    let mut root = pyre_object::gc_roots::pin_root(root);
    let mut root_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let mut composite_key = key.to_string();
    loop {
        root = pyre_object::gc_roots::shadow_stack_get(root_slot);
        let d = crate::baseobjspace::getdict_native(root);
        if d.is_null() {
            return;
        }
        if let Some(index) = unsafe { pyre_object::w_dict_getitem_str(d, BINDEX_KEY) }
            && unsafe { pyre_object::is_int(index) }
        {
            composite_key.push(':');
            composite_key.push_str(&unsafe { pyre_object::w_int_get_value(index) }.to_string());
        }
        match unsafe { pyre_object::w_dict_getitem_str(d, BBASE_KEY) } {
            Some(base) if !base.is_null() && !unsafe { pyre_object::is_none(base) } => {
                root = base;
                let _ = pyre_object::gc_roots::pin_root(root);
                root_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            }
            _ => break,
        }
    }
    root = pyre_object::gc_roots::shadow_stack_get(root_slot);
    let mut d = crate::baseobjspace::getdict_native(root);
    // An existing keepalive dictionary is reused; any other holder is replaced,
    // and a `bytes` one drops its keepalive count on the way out.
    let existing = match unsafe { pyre_object::w_dict_getitem_str(d, OBJECTS_KEY) } {
        Some(o) if !o.is_null() && unsafe { pyre_object::is_dict(o) } => Some(o),
        Some(previous) if !previous.is_null() && !unsafe { pyre_object::is_none(previous) } => {
            if unsafe { pyre_object::is_bytes(previous) } {
                unsafe { pyre_object::bytesobject::w_bytes_dec_ctypes_keepalive_refs(previous) };
            }
            None
        }
        _ => None,
    };
    let objs_slot = pyre_object::gc_roots::shadow_stack_len();
    match existing {
        Some(objs) => {
            let _ = pyre_object::gc_roots::pin_root(objs);
        }
        None => {
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
            root = pyre_object::gc_roots::shadow_stack_get(root_slot);
            d = crate::baseobjspace::getdict_native(root);
            let nd = pyre_object::gc_roots::shadow_stack_get(objs_slot);
            unsafe { pyre_object::w_dict_setitem_str(d, OBJECTS_KEY, nd) };
        }
    }
    let objs = pyre_object::gc_roots::shadow_stack_get(objs_slot);
    let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    unsafe {
        if let Some(previous) = pyre_object::w_dict_getitem_str(objs, &composite_key)
            && pyre_object::is_bytes(previous)
        {
            pyre_object::bytesobject::w_bytes_dec_ctypes_keepalive_refs(previous);
        }
        if pyre_object::is_bytes(obj) {
            pyre_object::bytesobject::w_bytes_inc_ctypes_keepalive_refs(obj);
        }
        let objs = pyre_object::gc_roots::shadow_stack_get(objs_slot);
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        pyre_object::w_dict_setitem_str(objs, &composite_key, obj)
    };
}

/// `ensure_objects(value)`: composite assignments retain the child's
/// keepalive dictionary, creating the empty dictionary that represents a
/// CData value with no inner references yet.
pub(super) fn objects_for_keep(value: PyObjectRef) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let value = pyre_object::gc_roots::pin_root(value);
    let value_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let mut d = crate::baseobjspace::getdict_native(value);
    if d.is_null() {
        return pyre_object::gc_roots::shadow_stack_get(value_slot);
    }
    match unsafe { pyre_object::w_dict_getitem_str(d, OBJECTS_KEY) } {
        Some(objects) if !objects.is_null() && !unsafe { pyre_object::is_none(objects) } => objects,
        _ => {
            let objects_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
            let value = pyre_object::gc_roots::shadow_stack_get(value_slot);
            d = crate::baseobjspace::getdict_native(value);
            let objects = pyre_object::gc_roots::shadow_stack_get(objects_slot);
            unsafe { pyre_object::w_dict_setitem_str(d, OBJECTS_KEY, objects) };
            pyre_object::gc_roots::shadow_stack_get(objects_slot)
        }
    }
}

/// Keep an implementation-owned holder alive without exposing it through
/// `_objects`.  CPython can point `c_char_p` at the trailing-NUL storage of a
/// bytes object; pyre's bytes storage needs an explicit terminated copy.  The
/// holder belongs to the same root object as `keep_ref`, but is kept in the
/// root instance dict rather than in the user-visible keepalive dictionary.
fn keep_alive(anchor: PyObjectRef, key: &str, obj: PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj = pyre_object::gc_roots::pin_root(obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let mut root = anchor;
    let mut root = pyre_object::gc_roots::pin_root(root);
    let mut root_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    loop {
        root = pyre_object::gc_roots::shadow_stack_get(root_slot);
        let d = crate::baseobjspace::getdict_native(root);
        if d.is_null() {
            return;
        }
        match unsafe { pyre_object::w_dict_getitem_str(d, BBASE_KEY) } {
            Some(base) if !base.is_null() && !unsafe { pyre_object::is_none(base) } => {
                root = base;
                let _ = pyre_object::gc_roots::pin_root(root);
                root_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            }
            _ => break,
        }
    }
    root = pyre_object::gc_roots::shadow_stack_get(root_slot);
    let d = crate::baseobjspace::getdict_native(root);
    if !d.is_null() {
        // Read the holder back only after the dictionary lookup: that lookup can
        // materialise the dict view and collect, which would leave an earlier
        // read holding a pre-move word.
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        unsafe { pyre_object::w_dict_setitem_str(d, &format!("_keep_{key}"), obj) };
    }
}

/// Make `result` share the source root's `_objects` dictionary, matching
/// `_ctypes.cast`'s PyCData keepalive ownership.  The source itself is added
/// under its identity key before the dictionary is attached to the result.
pub(super) fn share_objects_for_cast(result: PyObjectRef, source: PyObjectRef) {
    // RPython locals are GC roots.  These Rust locals are not, so mirror that
    // lifetime explicitly across `w_dict_new` and `w_dict_setitem` (the latter
    // allocates the integer identity key and can trigger a nursery collection).
    let _roots = pyre_object::gc_roots::push_roots();
    let result = pyre_object::gc_roots::pin_root(result);
    let result_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let source = pyre_object::gc_roots::pin_root(source);
    let source_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let mut root = source;
    let mut root = pyre_object::gc_roots::pin_root(root);
    let mut root_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    loop {
        root = pyre_object::gc_roots::shadow_stack_get(root_slot);
        let d = crate::baseobjspace::getdict_native(root);
        if d.is_null() {
            return;
        }
        match unsafe { pyre_object::w_dict_getitem_str(d, BBASE_KEY) } {
            Some(base) if !base.is_null() && !unsafe { pyre_object::is_none(base) } => {
                root = base;
                let _ = pyre_object::gc_roots::pin_root(root);
                root_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            }
            _ => break,
        }
    }
    root = pyre_object::gc_roots::shadow_stack_get(root_slot);
    let mut source_dict = crate::baseobjspace::getdict_native(root);
    if source_dict.is_null() {
        return;
    }
    let existing = unsafe { pyre_object::w_dict_getitem_str(source_dict, OBJECTS_KEY) };
    if let Some(objects) = existing.filter(|&objects| {
        !objects.is_null()
            && !unsafe { pyre_object::is_none(objects) }
            && !unsafe { pyre_object::is_dict(objects) }
    }) {
        // A non-dict holder is whatever was stored under `_objects_`, so it can
        // be a `list` that moves; the lookup that follows can collect, so it is
        // pinned here and read back at the store.
        let holder_slot = pyre_object::gc_roots::shadow_stack_len();
        let objects = pyre_object::gc_roots::pin_root(objects);
        let result = pyre_object::gc_roots::shadow_stack_get(result_slot);
        let result_dict = crate::baseobjspace::getdict_native(result);
        if !result_dict.is_null() {
            let objects = pyre_object::gc_roots::shadow_stack_get(holder_slot);
            unsafe { pyre_object::w_dict_setitem_str(result_dict, OBJECTS_KEY, objects) };
        }
        return;
    }
    let objects_slot = pyre_object::gc_roots::shadow_stack_len();
    match existing {
        Some(objects) if !objects.is_null() && unsafe { pyre_object::is_dict(objects) } => {
            let _ = pyre_object::gc_roots::pin_root(objects);
        }
        _ => {
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
            root = pyre_object::gc_roots::shadow_stack_get(root_slot);
            source_dict = crate::baseobjspace::getdict_native(root);
            let objects = pyre_object::gc_roots::shadow_stack_get(objects_slot);
            unsafe { pyre_object::w_dict_setitem_str(source_dict, OBJECTS_KEY, objects) };
        }
    }
    let source = pyre_object::gc_roots::shadow_stack_get(source_slot);
    let identity_key = pyre_object::w_int_new(source as usize as i64);
    let _ = pyre_object::gc_roots::pin_root(identity_key);
    let key_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let source = pyre_object::gc_roots::shadow_stack_get(source_slot);
    let objects = pyre_object::gc_roots::shadow_stack_get(objects_slot);
    let identity_key = pyre_object::gc_roots::shadow_stack_get(key_slot);
    unsafe { pyre_object::w_dict_store(objects, identity_key, source) };
    let result = pyre_object::gc_roots::shadow_stack_get(result_slot);
    let result_dict = crate::baseobjspace::getdict_native(result);
    if !result_dict.is_null() {
        let objects = pyre_object::gc_roots::shadow_stack_get(objects_slot);
        unsafe { pyre_object::w_dict_setitem_str(result_dict, OBJECTS_KEY, objects) };
    }
}

// ── type-code metadata (StgInfo equivalent, derived from `_type_`) ─────

/// A ctypes type code: the single character `_type_` names, or the `"void"`
/// [`super::funcptr::build_layout`] gives a zero-sized member.
///
/// It carries its own bytes rather than a `String` because a foreign call
/// reads one per argument and one for the return value, and a heap allocation
/// per read is most of what such a call does.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct TypeCode {
    bytes: [u8; Self::CAPACITY],
    len: u8,
}

impl TypeCode {
    /// Wide enough for `"void"`; every other code is one character.
    pub(super) const CAPACITY: usize = 4;

    /// The code `text` spells, or `None` when it is longer than any code is.
    pub(super) fn new(text: &str) -> Option<Self> {
        let src = text.as_bytes();
        if src.len() > Self::CAPACITY {
            return None;
        }
        let mut bytes = [0u8; Self::CAPACITY];
        bytes[..src.len()].copy_from_slice(src);
        Some(TypeCode {
            bytes,
            len: src.len() as u8,
        })
    }

    /// The code one character spells.  `const` so a caller naming a fixed code
    /// spends nothing on it.
    pub(super) const fn from_char(code: char) -> Self {
        assert!(code.is_ascii(), "a ctypes type code is ASCII");
        TypeCode {
            bytes: [code as u8, 0, 0, 0],
            len: 1,
        }
    }

    /// The code's bytes, nul-padded to a fixed width.  No code contains a nul,
    /// so this round trips through [`TypeCode::from_padded`].
    pub(super) fn padded(self) -> [u8; Self::CAPACITY] {
        self.bytes
    }

    /// The code [`TypeCode::padded`] wrote.
    pub(super) fn from_padded(bytes: [u8; Self::CAPACITY]) -> Self {
        let len = bytes
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(Self::CAPACITY);
        TypeCode {
            bytes,
            len: len as u8,
        }
    }

    pub(super) fn as_str(&self) -> &str {
        // The bytes came from a `&str` and were copied whole, so the prefix is
        // itself a code point boundary.
        core::str::from_utf8(&self.bytes[..self.len as usize]).unwrap_or("")
    }

    /// The one character a simple-type code is, or `None` for `"void"` and the
    /// empty code.
    pub(super) fn as_char(&self) -> Option<char> {
        let mut chars = self.as_str().chars();
        chars.next().filter(|_| chars.next().is_none())
    }
}

impl core::ops::Deref for TypeCode {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl core::fmt::Display for TypeCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl core::fmt::Debug for TypeCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self.as_str(), f)
    }
}

impl PartialEq<str> for TypeCode {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for TypeCode {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// The single-char ctypes `_type_` of `cls` (a type), read off the MRO.
/// Returns `None` when the attribute is absent, is not a string, or spells
/// something no code is.
pub(super) fn type_code_of(cls: PyObjectRef) -> Option<TypeCode> {
    TypeCode::new(declared_type_str(cls)?)
}

/// The string a class declares as its `_type_`, before it is read as a code.
///
/// [`type_code_of`] answers `None` both for a class that declares nothing and
/// for one whose declaration is too long to be a code, and class creation has
/// to tell those apart: the first is `_SimpleCData` and the abstract
/// intermediates under it, the second is a class that has to be refused.
pub(super) fn declared_type_str(cls: PyObjectRef) -> Option<&'static str> {
    if cls.is_null() || !unsafe { pyre_object::is_type(cls) } {
        return None;
    }
    let v = unsafe { crate::baseobjspace::lookup_in_type(cls, "_type_") }?;
    if !unsafe { pyre_object::is_str(v) } {
        return None;
    }
    unsafe { pyre_object::w_str_get_value_opt(v) }
}

/// Whether `obj` is a (subclass) type of `_SimpleCData`.
pub(super) fn is_simplecdata_type(obj: PyObjectRef) -> bool {
    !obj.is_null()
        && unsafe { pyre_object::is_type(obj) }
        && crate::baseobjspace::issubclass(obj, simplecdata_type()).unwrap_or(false)
}

/// Whether `obj` is an instance of a `_SimpleCData` subclass.
pub(super) fn is_simplecdata_instance(obj: PyObjectRef) -> bool {
    !obj.is_null() && unsafe { crate::baseobjspace::isinstance_w(obj, simplecdata_type()) }
}

/// Ctypes type codes whose value is a pointer (drives pointer-return
/// decoding / `TYPEFLAG_ISPOINTER`).
pub(super) fn is_pointer_code(code: &str) -> bool {
    matches!(code, "z" | "Z" | "P" | "s" | "X" | "O")
}

pub(super) fn invalid_type_code_error() -> crate::PyError {
    // Mirrors PyCSimpleType_init: an unrecognised `_type_` is an
    // AttributeError, so `ctypes.__init__`'s complex-type probe
    // (`try: class c_double_complex(_SimpleCData): _type_="D"; ...
    // except AttributeError`) is skipped when the code is unsupported.
    crate::PyError::attribute_error(format!(
        "class must define a '_type_' attribute which must be a single character string containing one of '{}'",
        rustpython_host_env::ctypes::simple_type_chars(),
    ))
}

// ── scalar value ⇄ bytes ──────────────────────────────────────────────

/// The `str` a BSTR slot holds, or `None` for a null one.
///
/// `X_get` reads the length out of the string itself (`SysStringLen`) rather
/// than stopping at the first NUL, because a BSTR may carry one; a null
/// pointer and a zero-length string are the same value here.
#[cfg(windows)]
pub(super) fn bstr_to_pyobject(bytes: &[u8]) -> PyObjectRef {
    let addr = host_ctypes::read_pointer_from_buffer(bytes);
    if addr == 0 {
        return pyre_object::w_none();
    }
    let units = unsafe {
        std::slice::from_raw_parts(
            addr as *const u16,
            windows_sys::Win32::Foundation::SysStringLen(addr as *const u16) as usize,
        )
    };
    pyre_object::w_str_from_wtf8(rustpython_wtf8::Wtf8Buf::from_wide(units))
}

/// The Python value the bytes of a slot of type code `tc` hold.
///
/// The host's decode table answers None for a code it does not know, and it
/// does not know the BSTR one; `X_get` reads that out of the length the string
/// itself records.
pub(super) fn decode_slot(tc: &str, bytes: &[u8]) -> PyObjectRef {
    #[cfg(windows)]
    if tc == "X" {
        return bstr_to_pyobject(bytes);
    }
    // `P_get` reads a null pointer as `None`, the way `z_get` and `Z_get` read
    // a null string as one; the host's table answers the address either way.
    if tc == "P" && host_ctypes::read_pointer_from_buffer(bytes) == 0 {
        return pyre_object::w_none();
    }
    // Cut to the code's own width first.  A return value arrives as the
    // register image it came back in rather than as a slot of its type, and
    // `c_get` is `PyBytes_FromStringAndSize(ptr, 1)` -- one byte, whatever the
    // register held above it.  Every other code already reads its own width,
    // so this reaches them as the slice they were being handed anyway.
    let bytes = match host_ctypes::simple_type_size(tc) {
        Some(size) if size < bytes.len() => &bytes[..size],
        _ => bytes,
    };
    decoded_to_pyobject(host_ctypes::decode_type_code(tc, bytes))
}

/// Release the BSTR a slot holds before something else is written over it.
///
/// A BSTR is the one value a ctypes buffer owns outright: `SysAllocStringLen`
/// allocated it, nothing else refers to it, and `X_set` frees the previous
/// contents on every store.  `addr` is the slot's own address, which is what
/// every caller has — the object's buffer for a `value` store, that buffer
/// plus the field offset for a struct or array slot, and the item address for
/// a pointer store.
///
/// Call it only once the replacement bytes exist: `X_set` allocates first and
/// frees the previous contents immediately before the store, so a value it
/// refuses leaves the slot as it was.
#[cfg(windows)]
pub(super) fn release_bstr_slot(tc: &str, addr: usize) {
    if tc != "X" || addr == 0 {
        return;
    }
    let previous = unsafe { (addr as *const usize).read_unaligned() };
    if previous != 0 {
        unsafe { windows_sys::Win32::Foundation::SysFreeString(previous as *const u16) };
    }
}

#[cfg(not(windows))]
pub(super) fn release_bstr_slot(_tc: &str, _addr: usize) {}

/// `Py_TYPE(value)->tp_name` — the class an instance reports itself as, which
/// for an instance-layout object is its `w_class` rather than the layout type
/// its `ob_type` names.
pub(super) fn value_type_name(obj: PyObjectRef) -> String {
    match crate::typedef::r#type(obj) {
        Some(tp) => unsafe { pyre_object::w_type_get_name(tp.as_ptr()) }.to_string(),
        None => unsafe { pyre_object::type_name_of(obj) }.to_string(),
    }
}

/// The pointer-width word an `int` names, over the whole unsigned range.
/// `PyLong_AsVoidPtr` reads the value as a signed word and falls back to the
/// unsigned conversion when that overflows, so an address with the top bit set
/// — every Windows pseudo-handle, `GetCurrentProcess()` among them — is an
/// address like any other rather than an integer too large to be one.
pub(super) fn pointer_word(obj: PyObjectRef) -> Result<usize, crate::PyError> {
    match crate::baseobjspace::int_w(obj) {
        Ok(value) => Ok(value as usize),
        Err(err) if err.kind == crate::PyErrorKind::OverflowError => {
            Ok(crate::baseobjspace::uint_w(obj)? as usize)
        }
        Err(err) => Err(err),
    }
}

/// The bytes of a `_SimpleCData` whose own type code is `tc`, which a
/// destination that takes an *instance* copies byte-for-byte
/// (`_PyCData_set`, the `PyObject_IsInstance` arm reached when the field
/// carries no setfunc of its own).
pub(super) fn same_type_bytes(tc: &str, obj: PyObjectRef) -> Option<Vec<u8>> {
    if !is_simplecdata_instance(obj) {
        return None;
    }
    let src_cls = unsafe { pyre_object::w_instance_get_type(obj) };
    (type_code_of(src_cls).as_deref() == Some(tc)).then(|| cdata_bytes(obj).unwrap_or(&[]).to_vec())
}

/// Store into a destination that takes an instance — a struct field, an array
/// item, a pointer item.  `_PyCData_set` reaches those with a null field
/// setfunc, so a same-typed instance is copied and everything else falls to
/// the type's own setter.  `Simple_init` and the `value` setter call that
/// setter directly (`Simple_set_value`) and so take no instance at all:
/// `c_long(c_long(42))` is a TypeError from `PyNumber_Index`.
pub(super) fn encode_instance_or_value(
    tc: &str,
    value: PyObjectRef,
    dest: PyObjectRef,
    key: &str,
) -> Result<Vec<u8>, crate::PyError> {
    match same_type_bytes(tc, value) {
        Some(bytes) => Ok(bytes),
        None => encode_value_into(tc, value, dest, key),
    }
}

/// Encode a Python scalar into the native-endian buffer bytes for `tc`.
///
/// This is the type's own setter (`setfunc`), which takes a value and not a
/// cdata: a `_SimpleCData` instance is converted like anything else, and so
/// rejected unless it is int/float-like.  A destination that also accepts an
/// instance of its own type reaches it through `encode_instance_or_value`.
pub(super) fn encode_value(tc: &str, obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    use host_ctypes::SimpleStorageValue as V;
    let val = match tc {
        "c" => {
            if unsafe { pyre_object::is_bytes(obj) } {
                let b = unsafe { pyre_object::bytesobject::w_bytes_data(obj) };
                if b.len() != 1 {
                    return Err(crate::PyError::type_error(
                        "one character bytes, bytearray or integer expected",
                    ));
                }
                V::Byte(b[0])
            } else if unsafe { pyre_object::is_int(obj) } {
                V::Byte(crate::baseobjspace::int_w(obj)? as u8)
            } else {
                return Err(crate::PyError::type_error(
                    "one character bytes, bytearray or integer expected",
                ));
            }
        }
        // `get_long`, `get_ulong`, `get_longlong` and `get_ulonglong` each read
        // the value through `PyLong_AsUnsignedLongMask`, so a value outside the
        // field's range is the low bits of it and not an `OverflowError`:
        // `c_int(2**31)` is `-2**31` and `c_longlong(2**63)` is `-2**63`.
        // `truncatedint_w` is that read.  The encoder masks the `i128` down to
        // the field's own width, so the sign the low 64 bits arrive with never
        // reaches a narrower field.
        "b" | "h" | "i" | "l" | "q" | "B" | "H" | "I" | "L" | "Q" => {
            let indexed = crate::baseobjspace::space_index(obj)?;
            V::Signed(crate::baseobjspace::truncatedint_w(indexed)? as i128)
        }
        "f" | "d" | "g" => V::Float(crate::baseobjspace::float_w(obj)?),
        "?" | "v" => V::Bool(crate::baseobjspace::is_true(obj)?),
        "u" => {
            if !unsafe { pyre_object::is_str(obj) } {
                return Err(crate::PyError::type_error(
                    "a unicode character expected, not instance",
                ));
            }
            let value = unsafe { pyre_object::w_str_get_wtf8(obj) };
            let mut chars = value.code_points();
            let ch = chars.next().ok_or_else(|| {
                crate::PyError::type_error("a unicode character expected, not a string of length 0")
            })?;
            if chars.next().is_some() {
                return Err(crate::PyError::type_error(
                    "a unicode character expected, not a string of length greater than 1",
                ));
            }
            V::Wchar(ch.to_u32())
        }
        "P" | "z" | "Z" => {
            if unsafe { pyre_object::is_none(obj) } {
                V::Pointer(0)
            } else if unsafe { pyre_object::pyobject::is_int_or_long(obj) } {
                V::Pointer(pointer_word(obj)?)
            } else {
                // Each of the three says what it takes.  `z_set` and `Z_set`
                // name the type that was passed instead
                // (`Py_TYPE(value)->tp_name`, the bare name); `P_set` takes
                // nothing but an address and says only that.
                let refused = |what: &str| {
                    let got = value_type_name(obj);
                    format!("{what} or integer address expected instead of {got} instance")
                };
                return Err(crate::PyError::type_error(match tc {
                    "z" => refused("bytes"),
                    "Z" => refused("unicode string"),
                    _ => "cannot be converted to pointer".to_string(),
                }));
            }
        }
        #[cfg(windows)]
        "X" => {
            // `X_set` takes a `str` and nothing else — not even the `bytes`
            // every other pointer code accepts — and allocates the BSTR the
            // slot will own.  Freeing what the slot held is the caller's,
            // because only it knows where the slot is.
            let addr = if unsafe { pyre_object::is_none(obj) } {
                0
            } else if unsafe { pyre_object::is_str(obj) } {
                let units: Vec<u16> = unsafe { pyre_object::w_str_get_wtf8(obj) }
                    .encode_wide()
                    .collect();
                let Ok(len) = u32::try_from(units.len()) else {
                    return Err(crate::PyError::value_error("String too long for BSTR"));
                };
                let bstr = unsafe {
                    windows_sys::Win32::Foundation::SysAllocStringLen(units.as_ptr(), len)
                };
                if bstr.is_null() {
                    return Err(crate::PyError::memory_error(""));
                }
                bstr as usize
            } else {
                return Err(crate::PyError::type_error(format!(
                    "unicode string expected instead of {} instance",
                    value_type_name(obj)
                )));
            };
            // The host's storage table has no entry for this code and answers
            // a single zero byte for one it does not know, so the pointer is
            // written here — `read_pointer_from_buffer` reads it back.
            return Ok(addr.to_ne_bytes().to_vec());
        }
        "O" => V::ObjectId(obj as usize),
        _ => V::Signed(crate::baseobjspace::int_w(obj)? as i128),
    };
    Ok(host_ctypes::simple_storage_value_to_bytes_endian(
        tc, val, false,
    ))
}

/// Encode `value` for storage into slot `key` of `dest`, threading `char*`
/// keepalive.  A `c_char_p` (`z`) initialised from `bytes` is stored as a
/// pointer to a NUL-terminated copy retained under `key` in `dest`'s root
/// `_objects_`; every other case defers to [`encode_value`].
pub(super) fn encode_value_into(
    tc: &str,
    value: PyObjectRef,
    dest: PyObjectRef,
    key: &str,
) -> Result<Vec<u8>, crate::PyError> {
    if tc == "z" && unsafe { pyre_object::is_bytes(value) } {
        let raw = unsafe { pyre_object::bytesobject::w_bytes_data(value) };
        let copy = pyre_object::bytesobject::w_bytes_from_bytes(
            &host_ctypes::clone_as_null_terminated(raw),
        );
        // Retain the copy before reading its address, so a GC triggered while
        // inserting the keepalive cannot leave the stored pointer stale.  Until
        // `keep_alive` links it into the root, only this frame holds the copy and
        // `keep_ref` allocates, so pin it for that window; it does not move, so
        // the word is never re-read.
        let _roots = pyre_object::gc_roots::push_roots();
        let copy = pyre_object::gc_roots::pin_root(copy);
        keep_ref(dest, key, value);
        keep_alive(dest, key, copy);
        let addr = unsafe { pyre_object::bytesobject::w_bytes_data(copy).as_ptr() } as usize;
        return Ok(host_ctypes::simple_storage_value_to_bytes_endian(
            tc,
            host_ctypes::SimpleStorageValue::Pointer(addr),
            false,
        ));
    }
    if tc == "Z" && unsafe { pyre_object::is_str(value) } {
        let raw = unsafe { pyre_object::w_str_get_wtf8(value) };
        let copy =
            pyre_object::w_bytearray_from_bytes(&host_ctypes::clone_wchar_null_terminated(raw));
        let _roots = pyre_object::gc_roots::push_roots();
        let copy = pyre_object::gc_roots::pin_root(copy);
        keep_ref(dest, key, value);
        keep_alive(dest, key, copy);
        let addr =
            unsafe { pyre_object::bytearrayobject::w_bytearray_data(copy).as_ptr() } as usize;
        return Ok(host_ctypes::simple_storage_value_to_bytes_endian(
            tc,
            host_ctypes::SimpleStorageValue::Pointer(addr),
            false,
        ));
    }
    encode_value(tc, value)
}

/// A `u64` as a non-negative Python integer: an `int` while it fits `i64`, a
/// `long` above `i64::MAX` so `c_ulonglong` / addresses keep their full range
/// instead of wrapping to a negative value.
fn u64_to_pyobject(u: u64) -> PyObjectRef {
    if u <= i64::MAX as u64 {
        pyre_object::w_int_new(u as i64)
    } else {
        pyre_object::longobject::w_long_new(majit_rlib::rbigint::RBigInt::from(u))
    }
}

/// Turn a decoded scalar into a pyre object.
pub(super) fn decoded_to_pyobject(d: host_ctypes::DecodedValue) -> PyObjectRef {
    use host_ctypes::DecodedValue as D;
    match d {
        D::Bytes(b) => pyre_object::bytesobject::w_bytes_from_bytes(&b),
        D::Signed(i) => pyre_object::w_int_new(i),
        D::Unsigned(u) => u64_to_pyobject(u),
        D::Float(f) => pyre_object::w_float_new(f),
        D::Bool(b) => pyre_object::w_bool_from(b),
        D::Pointer(p) => u64_to_pyobject(p as u64),
        D::String(s) => pyre_object::w_str_new(&s),
        D::None => pyre_object::w_none(),
    }
}
