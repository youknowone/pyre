//! `marshal` — PyPy `pypy/module/marshal/` object glue over the shared
//! RustPython compiler-core marshal format.
//!
//! PyPy's `interp_marshal.py` owns the Python object dispatch while
//! `marshal_impl.py` defines the wire typecodes.  Pyre follows that split:
//! this module maps `W_Root` objects to the shared wire implementation, and
//! compiler-core serializes/deserializes the authoritative `CodeObject`.

use num_complex::Complex64;
use pyre_object::*;
use rustpython_compiler_core::bytecode::{BasicBag, CodeObject, ConstantBag, ConstantData};
use rustpython_compiler_core::marshal::{self as wire, DumpableValue, Write};
use rustpython_wtf8::Wtf8;

use crate::{PyError, PyResult};

const MAX_DEPTH: usize = wire::MAX_MARSHAL_STACK_DEPTH;

fn marshal_error(error: wire::MarshalError) -> PyError {
    match error {
        wire::MarshalError::Eof => eof_error("marshal data too short"),
        _ => PyError::value_error("bad marshal data"),
    }
}

fn eof_error(message: &str) -> PyError {
    PyError::new(crate::PyErrorKind::EOFError, message)
}

fn call_method(obj: PyObjectRef, name: &str, args: &[PyObjectRef]) -> PyResult {
    let result = crate::baseobjspace::call_method(obj, name, args);
    if result.is_null() {
        Err(crate::call::take_call_error()
            .unwrap_or_else(|| PyError::runtime_error("method call failed")))
    } else {
        Ok(result)
    }
}

fn bytes_like(obj: PyObjectRef, function: &str) -> Result<Vec<u8>, PyError> {
    if unsafe { bytesobject::is_bytes_like(obj) } {
        return Ok(unsafe { bytesobject::bytes_like_data(obj) }.to_vec());
    }
    // Any readable buffer is accepted (`interp_marshal` unwraps via
    // `space.readbuf_w`): `SourcelessFileLoader.get_code` hands `loads` a
    // sliced memoryview of the pyc payload.
    if let Some(src) = crate::typedef::buffer_as_bytes_like(obj)? {
        return Ok(unsafe { bytesobject::bytes_like_data(src) }.to_vec());
    }
    Err(PyError::type_error(format!(
        "{function}() argument must be a bytes-like object"
    )))
}

/// PyPy `marshal_impl.py:104-120 marshal` buffer fallback.
///
/// Heap types bypass all builtin marshallers, then any object satisfying a
/// simple contiguous buffer request is written as `TYPE_STRING`.  Keep the
/// acquired bytes in a Rust `Vec`: unlike materialising a Python `bytes`, this
/// cannot move objects while `write_object` holds raw traversal locals.
fn marshal_buffer_bytes(obj: PyObjectRef) -> Result<Option<Vec<u8>>, PyError> {
    if let Some(target) = crate::module::__pypy__::interp_buffer::forwarded_exporter(obj) {
        return marshal_buffer_bytes(target?);
    }
    unsafe {
        if bytesobject::is_bytes_like(obj) {
            return Ok(Some(bytesobject::bytes_like_data(obj).to_vec()));
        }
        if interp_array::is_array(obj) {
            return Ok(Some(interp_array::w_array_bytes(obj).to_vec()));
        }
        if memoryview::is_w_memoryview(obj) {
            crate::builtins::memoryview_check_released(obj)?;
            crate::typedef::require_contiguous_buffer(obj)?;
            return Ok(Some(crate::builtins::memoryview_gather_bytes(obj)));
        }
    }
    Ok(None)
}

/// Transient equivalent of PyPy's `Marshaller.all_refs` dict.  A VecMap is
/// sufficient because marshal streams normally contain few shared objects;
/// each entry is a shadow-stack slot so a collection updates object identity.
struct WriterRefs {
    entries: Vec<WriterRefEntry>,
}

#[derive(Clone, Copy)]
struct WriterRefEntry {
    slot: usize,
    incomplete: bool,
}

impl WriterRefs {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn find(&self, obj: PyObjectRef) -> Option<(u32, bool)> {
        let obj =
            pyre_object::gc_hook::try_gc_current_object_address(obj as *mut u8) as PyObjectRef;
        self.entries
            .iter()
            .position(|entry| {
                std::ptr::eq(pyre_object::gc_roots::shadow_stack_get(entry.slot), obj)
            })
            .and_then(|index| {
                u32::try_from(index)
                    .ok()
                    .map(|index| (index, self.entries[index as usize].incomplete))
            })
    }

    fn reserve(&mut self, obj: PyObjectRef, incomplete: bool) -> Result<u32, PyError> {
        if self.entries.len() >= i32::MAX as usize {
            return Err(PyError::value_error("too many objects to marshal"));
        }
        let index = self.entries.len() as u32;
        let slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(obj);
        self.entries.push(WriterRefEntry { slot, incomplete });
        Ok(index)
    }

    fn complete(&mut self, index: u32) {
        let entry = &mut self.entries[index as usize];
        debug_assert!(entry.incomplete);
        entry.incomplete = false;
    }
}

fn is_singleton(obj: PyObjectRef) -> bool {
    (unsafe { is_none(obj) || is_bool(obj) || is_ellipsis(obj) })
        || crate::builtins::lookup_exc_class("StopIteration") == Some(obj)
}

fn write_object(
    out: &mut Vec<u8>,
    obj: PyObjectRef,
    refs: &mut Option<WriterRefs>,
    version: i32,
    depth: usize,
) -> Result<(), PyError> {
    // `obj` is held as a raw local across the whole traversal, which is sound
    // only because writing allocates nothing the GC manages: the output is a
    // Rust `Vec<u8>`, and the one conversion here (`obj_to_bigint`) reads the
    // int or clones a Rust `BigInt`.  A minor collection therefore cannot run
    // between the branch test and the read, so no reload from the shadow slot
    // is needed.  Anything added here that allocates breaks that.
    if depth == 0 {
        return Err(PyError::value_error("object too deeply nested to marshal"));
    }

    if !is_singleton(obj)
        && let Some(table) = refs.as_ref()
        && let Some((index, incomplete)) = table.find(obj)
    {
        // CPython 3.14 `w_ref`: code and slice entries retain the high-bit
        // incomplete marker until `w_complete`, because their immutable
        // representation cannot be reconstructed from a recursive reference.
        if incomplete {
            return Err(PyError::value_error(format!(
                "cannot marshal recursion {} objects",
                unsafe { pyre_object::type_name_of(obj) }
            )));
        }
        out.write_u8(b'r');
        out.write_u32(index);
        return Ok(());
    }

    let type_pos = out.len();
    let use_ref = refs.is_some() && !is_singleton(obj);
    // CPython 3.14 `w_ref` marks only code and slice objects incomplete and
    // clears that marker in `w_complete` after their contents are written.
    let requires_completion = unsafe { crate::pycode::is_code(obj) || is_slice(obj) };
    let ref_index = if use_ref {
        Some(refs.as_mut().unwrap().reserve(obj, requires_completion)?)
    } else {
        None
    };

    // PyPy `marshal`: instances of user heap types skip the builtin
    // marshaller table completely.  They may still reach the buffer fallback
    // below (notably bytes/bytearray subclasses).
    let is_heap_type = crate::typedef::r#type(obj)
        .is_some_and(|w_type| unsafe { typeobject::w_type_is_heaptype(w_type.as_ptr()) });

    unsafe {
        if !is_heap_type && is_none(obj) {
            out.write_u8(b'N');
        } else if !is_heap_type && crate::builtins::lookup_exc_class("StopIteration") == Some(obj) {
            out.write_u8(b'S');
        } else if !is_heap_type && is_bool(obj) {
            out.write_u8(if w_bool_get_value(obj) { b'T' } else { b'F' });
        } else if !is_heap_type && is_ellipsis(obj) {
            out.write_u8(b'.');
        } else if !is_heap_type && is_int_or_long(obj) {
            let value = crate::builtins::obj_to_bigint(obj);
            let value = crate::rbigint_to_compiler_bigint(&value);
            wire::serialize_value::<_, ConstantData>(out, DumpableValue::Integer(&value))
                .unwrap_or_else(|never| match never {});
        } else if !is_heap_type && is_float(obj) {
            out.write_u8(b'g');
            out.write_u64(w_float_get_value(obj).to_bits());
        } else if !is_heap_type && is_complex(obj) {
            out.write_u8(b'y');
            out.write_u64(w_complex_get_real(obj).to_bits());
            out.write_u64(w_complex_get_imag(obj).to_bits());
        } else if !is_heap_type && is_str(obj) {
            let value = w_str_get_wtf8(obj).as_bytes();
            out.write_u8(b'u');
            out.write_u32(
                value
                    .len()
                    .try_into()
                    .map_err(|_| PyError::value_error("object too large to marshal"))?,
            );
            out.write_slice(value);
        } else if let Some(value) = marshal_buffer_bytes(obj)? {
            out.write_u8(b's');
            out.write_u32(
                value
                    .len()
                    .try_into()
                    .map_err(|_| PyError::value_error("object too large to marshal"))?,
            );
            out.write_slice(&value);
        } else if !is_heap_type && is_tuple(obj) {
            let len = w_tuple_len(obj);
            out.write_u8(if version >= 4 && len < 256 {
                b')'
            } else {
                b'('
            });
            if version >= 4 && len < 256 {
                out.write_u8(len as u8);
            } else {
                out.write_u32(
                    len.try_into()
                        .map_err(|_| PyError::value_error("object too large to marshal"))?,
                );
            }
            for index in 0..len {
                let item = w_tuple_getitem(obj, index as i64)
                    .ok_or_else(|| PyError::value_error("unmarshallable object"))?;
                write_object(out, item, refs, version, depth - 1)?;
            }
        } else if !is_heap_type && is_list(obj) {
            let len = w_list_len(obj);
            out.write_u8(b'[');
            out.write_u32(
                len.try_into()
                    .map_err(|_| PyError::value_error("object too large to marshal"))?,
            );
            for index in 0..len {
                let item = w_list_getitem(obj, index as i64)
                    .ok_or_else(|| PyError::value_error("unmarshallable object"))?;
                write_object(out, item, refs, version, depth - 1)?;
            }
        } else if !is_heap_type && is_dict(obj) {
            out.write_u8(b'{');
            for (key, value) in dictmultiobject::w_dict_items(obj) {
                write_object(out, key, refs, version, depth - 1)?;
                write_object(out, value, refs, version, depth - 1)?;
            }
            out.write_u8(b'0');
        } else if !is_heap_type && setobject::is_set_or_frozenset(obj) {
            out.write_u8(if setobject::is_frozenset(obj) {
                b'>'
            } else {
                b'<'
            });
            let items = setobject::w_set_items(obj);
            out.write_u32(
                items
                    .len()
                    .try_into()
                    .map_err(|_| PyError::value_error("object too large to marshal"))?,
            );
            for item in items {
                write_object(out, item, refs, version, depth - 1)?;
            }
        } else if !is_heap_type && crate::pycode::is_code(obj) {
            let ptr = crate::pycode::w_code_get_ptr(obj) as *const crate::CodeObject;
            if ptr.is_null() {
                return Err(PyError::value_error("unmarshallable object"));
            }
            out.write_u8(b'c');
            wire::serialize_code(out, &*ptr);
        } else if !is_heap_type && is_slice(obj) && version >= 5 {
            out.write_u8(b':');
            write_object(
                out,
                sliceobject::w_slice_get_start(obj),
                refs,
                version,
                depth - 1,
            )?;
            write_object(
                out,
                sliceobject::w_slice_get_stop(obj),
                refs,
                version,
                depth - 1,
            )?;
            write_object(
                out,
                sliceobject::w_slice_get_step(obj),
                refs,
                version,
                depth - 1,
            )?;
        } else {
            return Err(PyError::value_error("unmarshallable object"));
        }
    }

    if use_ref {
        out[type_pos] |= wire::FLAG_REF;
    }
    if requires_completion && let (Some(table), Some(index)) = (refs.as_mut(), ref_index) {
        table.complete(index);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Rooted(usize);

impl Rooted {
    fn new(obj: PyObjectRef) -> Self {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(obj);
        Self(slot)
    }

    fn get(self) -> PyObjectRef {
        pyre_object::gc_roots::shadow_stack_get(self.0)
    }
}

/// PyPy `interp_marshal.py:70-101 FileReader`, with CPython 3.14's
/// `readinto()` validation when that method is available.
///
/// `wire::Read` requires the returned bytes to borrow from the reader, so each
/// exact read is copied into this reusable Rust buffer.  Python exceptions are
/// parked separately and restored by `load_impl` after the wire reader stops.
struct FileReader {
    file: Rooted,
    scratch: Vec<u8>,
    pending_error: Option<PyError>,
}

impl FileReader {
    fn new(file: PyObjectRef) -> Self {
        Self {
            file: Rooted::new(file),
            scratch: Vec::new(),
            pending_error: None,
        }
    }

    fn python_error<T>(&mut self, error: PyError) -> Result<T, wire::MarshalError> {
        self.pending_error = Some(error);
        Err(wire::MarshalError::BadType)
    }

    fn read_with_read(&mut self, n: u32) -> Result<Vec<u8>, wire::MarshalError> {
        let roots = pyre_object::gc_roots::push_roots();
        let file_slot = roots.base();
        roots.pin_root(self.file.get());
        let result = match call_method(roots.get(file_slot), "read", &[w_int_new(n as i64)]) {
            Ok(result) => result,
            Err(error) => return self.python_error(error),
        };
        let result_slot = file_slot + 1;
        roots.pin_root(result);
        let bytes = match bytes_like(roots.get(result_slot), "load") {
            Ok(bytes) => bytes,
            Err(error) => return self.python_error(error),
        };
        if bytes.len() < n as usize {
            return Err(wire::MarshalError::Eof);
        }
        if bytes.len() > n as usize {
            return self.python_error(PyError::value_error(format!(
                "read() returned too much data: {n} bytes requested, {} returned",
                bytes.len()
            )));
        }
        Ok(bytes)
    }
}

impl wire::Read for FileReader {
    fn read_slice(&mut self, n: u32) -> Result<&[u8], wire::MarshalError> {
        let roots = pyre_object::gc_roots::push_roots();
        let file_slot = roots.base();
        roots.pin_root(self.file.get());
        let buffer_slot = file_slot + 1;
        roots.pin_root(bytearrayobject::w_bytearray_new(n as usize));

        let bytes = match call_method(roots.get(file_slot), "readinto", &[roots.get(buffer_slot)]) {
            Ok(count) => {
                let count = match crate::baseobjspace::int_w(count) {
                    Ok(count) => count,
                    Err(error) => return self.python_error(error),
                };
                if count < 0 || count as u64 > n as u64 {
                    return self
                        .python_error(PyError::value_error("readinto() returned invalid length"));
                }
                if count as u32 != n {
                    return Err(wire::MarshalError::Eof);
                }
                unsafe { bytearrayobject::w_bytearray_data(roots.get(buffer_slot)).to_vec() }
            }
            Err(error) if error.kind == crate::PyErrorKind::AttributeError => {
                drop(roots);
                self.read_with_read(n)?
            }
            Err(error) => return self.python_error(error),
        };

        self.scratch = bytes;
        Ok(&self.scratch)
    }
}

#[derive(Clone, Copy)]
struct PyreMarshalBag;

impl wire::MarshalBag for PyreMarshalBag {
    type Value = Rooted;
    type ConstantBag = BasicBag;

    fn make_bool(&self, value: bool) -> Rooted {
        Rooted::new(w_bool_from(value))
    }

    fn make_none(&self) -> Rooted {
        Rooted::new(w_none())
    }

    fn make_ellipsis(&self) -> Rooted {
        Rooted::new(special::w_ellipsis())
    }

    fn make_float(&self, value: f64) -> Rooted {
        Rooted::new(w_float_new(value))
    }

    fn make_complex(&self, value: Complex64) -> Rooted {
        Rooted::new(w_complex_new(value.re, value.im))
    }

    fn make_str(&self, value: &Wtf8) -> Rooted {
        Rooted::new(w_str_from_wtf8(value.to_owned()))
    }

    fn make_bytes(&self, value: &[u8]) -> Rooted {
        Rooted::new(bytesobject::w_bytes_from_bytes(value))
    }

    fn make_int(&self, value: malachite_bigint::BigInt) -> Rooted {
        let value = crate::compiler_bigint_to_rbigint(&value);
        let obj = if longobject::jit_bigint_to_i64_fits(&value) != 0 {
            w_int_new(longobject::jit_bigint_to_i64_value(&value))
        } else {
            longobject::w_long_new(value)
        };
        Rooted::new(obj)
    }

    fn make_tuple(&self, elements: impl Iterator<Item = Rooted>) -> Rooted {
        Rooted::new(w_tuple_new(elements.map(Rooted::get).collect()))
    }

    fn make_code(&self, code: CodeObject<ConstantData>) -> Rooted {
        Rooted::new(crate::pycode::box_code_constant(&code))
    }

    fn make_stop_iter(&self) -> Result<Rooted, wire::MarshalError> {
        crate::builtins::lookup_exc_class("StopIteration")
            .map(Rooted::new)
            .ok_or(wire::MarshalError::BadType)
    }

    fn make_list(
        &self,
        elements: impl Iterator<Item = Rooted>,
    ) -> Result<Rooted, wire::MarshalError> {
        Ok(Rooted::new(w_list_new(elements.map(Rooted::get).collect())))
    }

    fn make_set(
        &self,
        elements: impl Iterator<Item = Rooted>,
    ) -> Result<Rooted, wire::MarshalError> {
        let set = Rooted::new(setobject::w_set_new());
        for item in elements {
            let hash = crate::baseobjspace::hash_w_strict(item.get())
                .map_err(|_| wire::MarshalError::BadType)?;
            unsafe { setobject::w_set_add_hashed_checked(set.get(), item.get(), hash) }
                .map_err(|_| wire::MarshalError::BadType)?;
        }
        Ok(set)
    }

    fn make_frozenset(
        &self,
        elements: impl Iterator<Item = Rooted>,
    ) -> Result<Rooted, wire::MarshalError> {
        let set = Rooted::new(setobject::w_frozenset_new());
        for item in elements {
            let hash = crate::baseobjspace::hash_w_strict(item.get())
                .map_err(|_| wire::MarshalError::BadType)?;
            unsafe { setobject::w_set_add_hashed_checked(set.get(), item.get(), hash) }
                .map_err(|_| wire::MarshalError::BadType)?;
        }
        Ok(set)
    }

    fn make_dict(
        &self,
        elements: impl Iterator<Item = (Rooted, Rooted)>,
    ) -> Result<Rooted, wire::MarshalError> {
        let dict = Rooted::new(w_dict_new());
        for (key, value) in elements {
            unsafe { w_dict_store_checked(dict.get(), key.get(), value.get()) }
                .map_err(|_| wire::MarshalError::BadType)?;
        }
        Ok(dict)
    }

    fn make_slice(
        &self,
        start: Rooted,
        stop: Rooted,
        step: Rooted,
    ) -> Result<Rooted, wire::MarshalError> {
        Ok(Rooted::new(w_slice_new(
            start.get(),
            stop.get(),
            step.get(),
        )))
    }

    fn constant_bag(self) -> BasicBag {
        BasicBag
    }

    fn constant_ref_from_value(&self, value: &Rooted) -> Option<ConstantData> {
        unsafe { crate::pycode::obj_to_constant_data(value.get()).ok() }
    }
}

/// Resolve the optional `version` slot: an omitted slot uses the current
/// format version; an explicit value goes through `int_w`, so a `None`
/// raises `TypeError` like any other non-integer.
fn resolve_version(version: Option<PyObjectRef>) -> Result<i32, PyError> {
    match version {
        Some(value) => Ok(crate::baseobjspace::int_w(value)? as i32),
        None => Ok(wire::FORMAT_VERSION as i32),
    }
}

/// Resolve the keyword-only `allow_code` slot by truth-testing; an omitted
/// slot defaults to true.
fn resolve_allow_code(allow_code: Option<PyObjectRef>) -> Result<bool, PyError> {
    match allow_code {
        Some(value) => crate::baseobjspace::is_true(value),
        None => Ok(true),
    }
}

/// Serialize one object to a marshal byte stream at `version`.  With
/// `allow_code` false, a nested code object is rejected before writing
/// (marshal.check_no_code).
fn marshal_to_bytes(
    value: PyObjectRef,
    version: i32,
    allow_code: bool,
) -> Result<Vec<u8>, PyError> {
    if !allow_code {
        reject_code(value)?;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(value);
    let mut out = Vec::new();
    let mut refs = (version >= 3).then(WriterRefs::new);
    write_object(&mut out, value, &mut refs, version, MAX_DEPTH)?;
    Ok(out)
}

/// Deserialize one object from a marshal byte stream.  With `allow_code`
/// false, a decoded code object is rejected (marshal.check_no_code).
fn unmarshal_bytes(data: &[u8], allow_code: bool) -> PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let mut reader: &[u8] = data;
    let result = wire::deserialize_value(&mut reader, PyreMarshalBag).map_err(marshal_error)?;
    let result = result.get();
    if !allow_code {
        reject_code(result)?;
    }
    Ok(result)
}

/// RustPython `marshal.check_no_code`, matching CPython's recursive
/// `allow_code=False` check.  The Vec is a transient identity set (PyPy's
/// traversal does not persist it); it also prevents container cycles from
/// recursing forever without introducing a side table.
fn contains_code(obj: PyObjectRef, seen: &mut Vec<PyObjectRef>) -> bool {
    if unsafe { crate::pycode::is_code(obj) } {
        return true;
    }
    if seen.iter().any(|&seen_obj| std::ptr::eq(seen_obj, obj)) {
        return false;
    }
    let is_container = unsafe {
        is_tuple(obj) || is_list(obj) || is_dict(obj) || setobject::is_set_or_frozenset(obj)
    };
    if !is_container {
        return false;
    }
    seen.push(obj);
    unsafe {
        if is_tuple(obj) {
            (0..w_tuple_len(obj)).any(|index| {
                w_tuple_getitem(obj, index as i64).is_some_and(|item| contains_code(item, seen))
            })
        } else if is_list(obj) {
            (0..w_list_len(obj)).any(|index| {
                w_list_getitem(obj, index as i64).is_some_and(|item| contains_code(item, seen))
            })
        } else if is_dict(obj) {
            dictmultiobject::w_dict_items(obj)
                .into_iter()
                .any(|(key, value)| contains_code(key, seen) || contains_code(value, seen))
        } else {
            setobject::w_set_items(obj)
                .into_iter()
                .any(|item| contains_code(item, seen))
        }
    }
}

fn reject_code(value: PyObjectRef) -> Result<(), PyError> {
    if contains_code(value, &mut Vec::new()) {
        Err(PyError::value_error(
            "unmarshalling code objects is disallowed",
        ))
    } else {
        Ok(())
    }
}

/// `PyMarshal_ReadObjectFromString` — deserialize one object out of a raw byte
/// buffer.  `_imp.get_frozen_object` unmarshals caller-supplied frozen data
/// through this and rewrites any failure into its own diagnostic.
pub(crate) fn loads_bytes(data: &[u8]) -> PyResult {
    unmarshal_bytes(data, true)
}

/// `PyMarshal_WriteObjectToString` — serialize one object into a raw byte
/// buffer at the current format version.  `_imp.find_frozen(withdata=True)`
/// hands these bytes back as the frozen data, so they are the same stream
/// `loads_bytes` reads.
pub(crate) fn dumps_bytes(value: PyObjectRef) -> Result<Vec<u8>, PyError> {
    marshal_to_bytes(value, wire::FORMAT_VERSION as i32, true)
}

crate::py_module! {
    "marshal",
    int_constants: {
        "version" => wire::FORMAT_VERSION as i64,
    },
    inline_functions: {
        // interp_marshal.py:33 `dumps(w_data, version=Py_MARSHAL_VERSION)` —
        // `value` / `version` positional-only, `allow_code` keyword-only
        // (the 3.13 accelerator signature `dumps(value, version, /, *,
        // allow_code=True)`).  `version` stays `Option` so an explicit
        // `None` reaches `int_w` and raises rather than defaulting.
        fn dumps(
            value: PyObjectRef,
            version: Option<PyObjectRef>,
            #[posonly]
            #[kwonly]
            allow_code: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            let version = resolve_version(version)?;
            let allow_code = resolve_allow_code(allow_code)?;
            let out = marshal_to_bytes(value, version, allow_code)?;
            Ok(bytesobject::w_bytes_from_bytes(&out))
        }
        // interp_marshal.py:49 `loads(w_str)` — `bytes` positional-only,
        // `allow_code` keyword-only (`loads(bytes, /, *, allow_code=True)`).
        fn loads(
            data: PyObjectRef,
            #[posonly]
            #[kwonly]
            allow_code: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            let allow_code = resolve_allow_code(allow_code)?;
            let data = bytes_like(data, "loads")?;
            unmarshal_bytes(&data, allow_code)
        }
        // interp_marshal.py:26 `dump(w_data, w_f, version=Py_MARSHAL_VERSION)`
        // — writes the stream `dumps` would return to `f.write`
        // (`dump(value, file, version, /, *, allow_code=True)`).
        fn dump(
            value: PyObjectRef,
            file: PyObjectRef,
            version: Option<PyObjectRef>,
            #[posonly]
            #[kwonly]
            allow_code: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            let version = resolve_version(version)?;
            let allow_code = resolve_allow_code(allow_code)?;
            let out = marshal_to_bytes(value, version, allow_code)?;
            let bytes = bytesobject::w_bytes_from_bytes(&out);
            call_method(file, "write", &[bytes])?;
            Ok(w_none())
        }
        // interp_marshal.py:40 `load(w_f)` reads one value from `f` and
        // rewinds `f` past exactly the bytes consumed
        // (`load(file, /, *, allow_code=True)`).
        fn load(
            file: PyObjectRef,
            #[posonly]
            #[kwonly]
            allow_code: Option<PyObjectRef>,
        ) -> Result<PyObjectRef, crate::PyError> {
            let allow_code = resolve_allow_code(allow_code)?;
            let _roots = pyre_object::gc_roots::push_roots();
            let mut reader = FileReader::new(file);
            let result = match wire::deserialize_value(&mut reader, PyreMarshalBag) {
                Ok(result) => result,
                Err(error) => {
                    if let Some(error) = reader.pending_error.take() {
                        return Err(error);
                    }
                    return Err(marshal_error(error));
                }
            };
            let result = result.get();
            if !allow_code {
                reject_code(result)?;
            }
            Ok(result)
        }
    },
}
