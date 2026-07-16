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

use super::type_ns_store;
use pyre_object::PyObjectRef;
use rustpython_host_env::ctypes as host_ctypes;
use std::cell::RefCell;

/// Reserved instance-dict key holding the backing `bytearray` (root storage,
/// or — for a sub-view — a shared reference to the **root's** bytearray).
const CDATA_BUFFER_KEY: &str = "_b_";
/// Byte offset of a sub-view from the start of the root buffer (absent ⇒ 0).
const BOFF_KEY: &str = "_boff_";
/// View size in bytes (absent ⇒ the whole buffer from `_boff_`).
const BSZ_KEY: &str = "_bsz_";
/// The parent CData object a sub-view was carved from (keeps it alive).
const BBASE_KEY: &str = "_b_base_";
/// Raw address of an external (non-bytearray-backed) view.
const BADDR_KEY: &str = "_baddr_";
/// Lazily-created keepalive dict on a root object.
const OBJECTS_KEY: &str = "_objects_";

thread_local! {
    static SIMPLECDATA_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> =
        const { std::cell::OnceCell::new() };
}

/// The native `_SimpleCData` type object (cached, `hasdict=true`).
pub(super) fn simplecdata_type() -> PyObjectRef {
    SIMPLECDATA_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("_SimpleCData", init_simplecdata_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

fn init_simplecdata_type(ns: PyObjectRef) {
    type_ns_store(
        ns,
        "__new__",
        crate::make_builtin_function("__new__", simplecdata_new),
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
            pyre_object::PY_NULL,
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

/// `_SimpleCData.from_param(cls, value)` — identity stub (see caller note).
fn simplecdata_from_param(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // args[0] is the bound `cls`; the argument to convert is args[1].
    Ok(args.get(1).copied().unwrap_or_else(pyre_object::w_none))
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
    let size = host_ctypes::simple_type_size(tc).ok_or_else(|| invalid_type_code_error())?;
    let ba = pyre_object::w_bytearray_new(size);
    if let Some(v) = value {
        let bytes = encode_value(tc, v)?;
        let n = bytes.len().min(size);
        unsafe {
            pyre_object::w_bytearray_data_mut(ba)[..n].copy_from_slice(&bytes[..n]);
        }
    }
    let obj = pyre_object::w_instance_new(cls);
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return Err(crate::PyError::type_error(
            "ctypes instance has no instance dict",
        ));
    }
    unsafe { pyre_object::w_dict_setitem_str(d, CDATA_BUFFER_KEY, ba) };
    Ok(obj)
}

/// `_SimpleCData.value` getter — `(descr, instance)`.
fn value_getter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[1];
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let tc = type_code_of(cls).ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let bytes = cdata_bytes(obj)
        .ok_or_else(|| crate::PyError::type_error("ctypes instance has no buffer"))?;
    Ok(decoded_to_pyobject(host_ctypes::decode_type_code(
        &tc, bytes,
    )))
}

/// `_SimpleCData.value` setter — `(descr, instance, value)`.
fn value_setter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[1];
    let value = args[2];
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let tc = type_code_of(cls).ok_or_else(|| crate::PyError::type_error("abstract class"))?;
    let bytes = encode_value(&tc, value)?;
    cdata_write(obj, 0, &bytes);
    Ok(pyre_object::w_none())
}

// ── buffer helpers (b_ptr / b_size / b_base equivalents) ───────────────

/// Read a `usize`-valued reserved key off the instance dict.
fn dict_usize(obj: PyObjectRef, key: &str) -> Option<usize> {
    let d = crate::baseobjspace::getdict(obj);
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
    let d = crate::baseobjspace::getdict(obj);
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
pub(super) fn cdata_bytes(obj: PyObjectRef) -> Option<&'static [u8]> {
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

thread_local! {
    /// The ctypes base types whose instances are CData (registered at module
    /// init): `_SimpleCData`, `Structure`, `Union`, `Array`, `_Pointer`.
    static CDATA_BASES: RefCell<Vec<PyObjectRef>> = const { RefCell::new(Vec::new()) };
}

/// Register `tp` as a CData base type (widens [`is_cdata_instance`]).
pub(super) fn register_cdata_base(tp: PyObjectRef) {
    CDATA_BASES.with(|b| {
        let mut v = b.borrow_mut();
        if !v.iter().any(|&t| t == tp) {
            v.push(tp);
        }
    });
}

/// Whether `obj` owns its buffer (a root object, not a sub-view or external
/// view) — the precondition for `resize`.
pub(super) fn owns_buffer(obj: PyObjectRef) -> bool {
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return false;
    }
    let has = |k: &str| unsafe { pyre_object::w_dict_getitem_str(d, k) }.is_some();
    has(CDATA_BUFFER_KEY) && !has(BBASE_KEY) && !has(BADDR_KEY)
}

/// Whether `obj` is an instance of any registered CData base type.
pub(super) fn is_cdata_instance(obj: PyObjectRef) -> bool {
    !obj.is_null()
        && CDATA_BASES.with(|b| {
            b.borrow()
                .iter()
                .any(|&base| unsafe { crate::baseobjspace::isinstance_w(obj, base) })
        })
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
    let inst = pyre_object::w_instance_new(proto);
    let d = crate::baseobjspace::getdict(inst);
    if d.is_null() {
        return inst;
    }
    unsafe {
        if let Some(ba) = cdata_buffer(parent) {
            pyre_object::w_dict_setitem_str(d, CDATA_BUFFER_KEY, ba);
            pyre_object::w_dict_setitem_str(
                d,
                BOFF_KEY,
                pyre_object::w_int_new((boff(parent) + field_offset) as i64),
            );
        } else if let Some(addr) = baddr(parent) {
            pyre_object::w_dict_setitem_str(
                d,
                BADDR_KEY,
                pyre_object::w_int_new((addr + field_offset) as i64),
            );
        }
        pyre_object::w_dict_setitem_str(d, BSZ_KEY, pyre_object::w_int_new(size as i64));
        pyre_object::w_dict_setitem_str(d, BBASE_KEY, parent);
    }
    inst
}

/// An `address`-backed instance of `proto` viewing `size` bytes of external
/// memory (`PyCData::at_address`) — the pyre form of a pointer dereference.
/// The memory is owned elsewhere; the caller keeps it alive.
pub(super) fn make_at_address(proto: PyObjectRef, address: usize, size: usize) -> PyObjectRef {
    let inst = pyre_object::w_instance_new(proto);
    let d = crate::baseobjspace::getdict(inst);
    if !d.is_null() {
        unsafe {
            pyre_object::w_dict_setitem_str(d, BADDR_KEY, pyre_object::w_int_new(address as i64));
            pyre_object::w_dict_setitem_str(d, BSZ_KEY, pyre_object::w_int_new(size as i64));
        }
    }
    inst
}

/// Keep `obj` alive for the lifetime of the buffer that `anchor` views, by
/// storing it under `key` in the ultimate root's `"_objects_"` dict.
pub(super) fn keep_ref(anchor: PyObjectRef, key: &str, obj: PyObjectRef) {
    // Walk `_b_base_` up to the owning root.
    let mut root = anchor;
    loop {
        let d = crate::baseobjspace::getdict(root);
        if d.is_null() {
            return;
        }
        match unsafe { pyre_object::w_dict_getitem_str(d, BBASE_KEY) } {
            Some(base) if !base.is_null() && !unsafe { pyre_object::is_none(base) } => root = base,
            _ => break,
        }
    }
    let d = crate::baseobjspace::getdict(root);
    let objs = match unsafe { pyre_object::w_dict_getitem_str(d, OBJECTS_KEY) } {
        Some(o) if !o.is_null() => o,
        _ => {
            let nd = pyre_object::w_dict_new();
            unsafe { pyre_object::w_dict_setitem_str(d, OBJECTS_KEY, nd) };
            nd
        }
    };
    unsafe { pyre_object::w_dict_setitem_str(objs, key, obj) };
}

// ── type-code metadata (StgInfo equivalent, derived from `_type_`) ─────

/// The single-char ctypes `_type_` of `cls` (a type), read off the MRO.
/// Returns `None` when the attribute is absent or not a string.
pub(super) fn type_code_of(cls: PyObjectRef) -> Option<String> {
    if cls.is_null() || !unsafe { pyre_object::is_type(cls) } {
        return None;
    }
    let v = unsafe { crate::baseobjspace::lookup_in_type(cls, "_type_") }?;
    if !unsafe { pyre_object::is_str(v) } {
        return None;
    }
    Some(unsafe { pyre_object::w_str_get_value(v) }.to_string())
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
    crate::PyError::attribute_error(
        "class must define a '_type_' attribute which must be \
         a single character string containing one of the valid ctypes type codes",
    )
}

// ── scalar value ⇄ bytes ──────────────────────────────────────────────

/// Encode a Python scalar into the native-endian buffer bytes for `tc`.
///
/// A same-typed `_SimpleCData` instance is accepted and copied byte-for-byte.
pub(super) fn encode_value(tc: &str, obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    use host_ctypes::SimpleStorageValue as V;
    if is_simplecdata_instance(obj) {
        return Ok(cdata_bytes(obj).unwrap_or(&[]).to_vec());
    }
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
        "b" | "B" | "h" | "H" | "i" | "I" | "l" | "L" | "q" | "Q" => {
            V::Signed(crate::baseobjspace::int_w(obj)? as i128)
        }
        "f" | "d" | "g" => V::Float(crate::baseobjspace::float_w(obj)?),
        "?" => V::Bool(crate::baseobjspace::is_true(obj)?),
        "u" => V::Wchar(crate::baseobjspace::int_w(obj)? as u32),
        "P" | "z" | "Z" => {
            if unsafe { pyre_object::is_none(obj) } {
                V::Pointer(0)
            } else if unsafe { pyre_object::is_int(obj) } {
                V::Pointer(crate::baseobjspace::int_w(obj)? as usize)
            } else {
                return Err(crate::PyError::type_error("cannot be converted to pointer"));
            }
        }
        "O" => V::ObjectId(crate::baseobjspace::int_w(obj)? as usize),
        _ => V::Signed(crate::baseobjspace::int_w(obj)? as i128),
    };
    Ok(host_ctypes::simple_storage_value_to_bytes_endian(
        tc, val, false,
    ))
}

/// Turn a decoded scalar into a pyre object.
pub(super) fn decoded_to_pyobject(d: host_ctypes::DecodedValue) -> PyObjectRef {
    use host_ctypes::DecodedValue as D;
    match d {
        D::Bytes(b) => pyre_object::bytesobject::w_bytes_from_bytes(&b),
        D::Signed(i) => pyre_object::w_int_new(i),
        // No unsigned-i64 int constructor exists; values above i64::MAX
        // wrap.  The slice's scalar returns stay within i64 range.
        D::Unsigned(u) => pyre_object::w_int_new(u as i64),
        D::Float(f) => pyre_object::w_float_new(f),
        D::Bool(b) => pyre_object::w_bool_from(b),
        D::Pointer(p) => pyre_object::w_int_new(p as i64),
        D::String(s) => pyre_object::w_str_new(&s),
        D::None => pyre_object::w_none(),
    }
}
