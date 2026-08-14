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

fn write_len(out: &mut Vec<u8>, len: usize) -> Result<(), PyError> {
    out.write_u32(
        len.try_into()
            .map_err(|_| PyError::value_error("object too large to marshal"))?,
    );
    Ok(())
}

fn write_bytes(out: &mut Vec<u8>, value: &[u8]) -> Result<(), PyError> {
    out.write_u8(b's');
    write_len(out, value.len())?;
    out.write_slice(value);
    Ok(())
}

fn write_marshal_str(out: &mut Vec<u8>, value: &str, version: i32) -> Result<(), PyError> {
    let bytes = value.as_bytes();
    if version >= 4 && bytes.len() < 256 && bytes.is_ascii() {
        out.write_u8(b'z');
        out.write_u8(bytes.len() as u8);
    } else {
        out.write_u8(b'u');
        write_len(out, bytes.len())?;
    }
    out.write_slice(bytes);
    Ok(())
}

fn write_name_tuple<N: AsRef<str>>(
    out: &mut Vec<u8>,
    names: &[N],
    version: i32,
) -> Result<(), PyError> {
    out.write_u8(b'(');
    write_len(out, names.len())?;
    for name in names {
        write_marshal_str(out, name.as_ref(), version)?;
    }
    Ok(())
}

/// CPython 3.14 `w_code` / PyPy `marshal_pycode`, in compiler-core's 3.14
/// field order.  The one deliberate boundary from `wire::serialize_code` is
/// co_consts: PyPy writes `x.co_consts_w`, so recurse through the wrapped
/// objects owned by this exact PyCode instead of the compiler backing enum.
unsafe fn write_code(
    out: &mut Vec<u8>,
    w_code: PyObjectRef,
    refs: &mut Option<WriterRefs>,
    version: i32,
    depth: usize,
) -> Result<(), PyError> {
    let code_root = Rooted::new(w_code);
    let code =
        unsafe { &*(crate::pycode::w_code_get_ptr(code_root.get()) as *const crate::CodeObject) };
    out.write_u32(code.arg_count);
    out.write_u32(code.posonlyarg_count);
    out.write_u32(code.kwonlyarg_count);
    out.write_u32(code.max_stackdepth);
    out.write_u32(code.flags.bits());

    write_bytes(out, &code.instructions.original_bytes())?;

    out.write_u8(b'(');
    write_len(out, code.constants.len())?;
    // Realizing one constant may collect and move the wrapper. Keep a shadow
    // slot and reload it just like PyPy keeps `x` live throughout w_code.
    let constant_count = code.constants.len();
    for index in 0..constant_count {
        let constant = crate::pycode::w_code_const(code_root.get(), index);
        if constant.is_null() {
            return Err(PyError::value_error("unmarshallable object"));
        }
        write_object(out, constant, refs, version, depth - 1)?;
    }

    // A wrapped constant may have allocated and moved `w_code`; recover the
    // backing payload from the forwarded wrapper before touching later fields.
    let code =
        unsafe { &*(crate::pycode::w_code_get_ptr(code_root.get()) as *const crate::CodeObject) };
    write_name_tuple(out, &code.names, version)?;

    let cell_only_names: Vec<&str> = code
        .cellvars
        .iter()
        .filter(|cell| !code.varnames.iter().any(|local| local == *cell))
        .map(|cell| cell.as_ref())
        .collect();
    out.write_u8(b'(');
    write_len(
        out,
        code.varnames.len() + cell_only_names.len() + code.freevars.len(),
    )?;
    for name in &code.varnames {
        write_marshal_str(out, name.as_ref(), version)?;
    }
    for name in cell_only_names {
        write_marshal_str(out, name, version)?;
    }
    for name in &code.freevars {
        write_marshal_str(out, name.as_ref(), version)?;
    }

    write_bytes(out, &code.localspluskinds)?;
    write_marshal_str(out, code.source_path.as_ref(), version)?;
    write_marshal_str(out, code.obj_name.as_ref(), version)?;
    write_marshal_str(out, code.qualname.as_ref(), version)?;
    out.write_u32(
        unsafe { (*(code_root.get() as *const crate::pycode::PyCode)).co_firstlineno_raw } as u32,
    );
    write_bytes(out, &code.linetable)?;
    write_bytes(out, &code.exceptiontable)?;
    Ok(())
}

fn write_object(
    out: &mut Vec<u8>,
    obj: PyObjectRef,
    refs: &mut Option<WriterRefs>,
    version: i32,
    depth: usize,
) -> Result<(), PyError> {
    if depth == 0 {
        return Err(PyError::value_error("object too deeply nested to marshal"));
    }
    // Recursive writes can realize code constants and run the collector. Keep
    // every current object rooted even for marshal versions without a ref
    // table, then reload containers before each child fetch.
    let obj_root = Rooted::new(obj);
    let obj = obj_root.get();

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
        let obj = obj_root.get();
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
            let interned = unicodeobject::is_interned_exact_str(obj);
            let ascii = value.iter().all(u8::is_ascii);
            if version >= 4 && ascii && value.len() < 256 {
                out.write_u8(if interned { b'Z' } else { b'z' });
                out.write_u8(value.len() as u8);
            } else {
                out.write_u8(if version >= 4 && ascii {
                    if interned { b'A' } else { b'a' }
                } else if version >= 3 && interned {
                    b't'
                } else {
                    b'u'
                });
                out.write_u32(
                    value
                        .len()
                        .try_into()
                        .map_err(|_| PyError::value_error("object too large to marshal"))?,
                );
            }
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
                let item = w_tuple_getitem(obj_root.get(), index as i64)
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
                let item = w_list_getitem(obj_root.get(), index as i64)
                    .ok_or_else(|| PyError::value_error("unmarshallable object"))?;
                write_object(out, item, refs, version, depth - 1)?;
            }
        } else if !is_heap_type && is_dict(obj) {
            out.write_u8(b'{');
            let len = dictmultiobject::w_dict_len(obj);
            for index in 0..len {
                let (key, value) = dictmultiobject::w_dict_nth_item(obj_root.get(), index)
                    .ok_or_else(|| PyError::value_error("unmarshallable object"))?;
                let key_root = Rooted::new(key);
                let value_root = Rooted::new(value);
                write_object(out, key_root.get(), refs, version, depth - 1)?;
                write_object(out, value_root.get(), refs, version, depth - 1)?;
            }
            out.write_u8(b'0');
        } else if !is_heap_type && setobject::is_set_or_frozenset(obj) {
            out.write_u8(if setobject::is_frozenset(obj) {
                b'>'
            } else {
                b'<'
            });
            let len = setobject::w_set_len(obj);
            write_len(out, len)?;
            for index in 0..len {
                let item = setobject::w_set_key_at(obj_root.get(), index)
                    .ok_or_else(|| PyError::value_error("unmarshallable object"))?
                    .obj;
                let item_root = Rooted::new(item);
                write_object(out, item_root.get(), refs, version, depth - 1)?;
            }
        } else if !is_heap_type && crate::pycode::is_code(obj) {
            let ptr = crate::pycode::w_code_get_ptr(obj_root.get()) as *const crate::CodeObject;
            if ptr.is_null() {
                return Err(PyError::value_error("unmarshallable object"));
            }
            out.write_u8(b'c');
            write_code(out, obj_root.get(), refs, version, depth)?;
        } else if !is_heap_type && is_slice(obj) && version >= 5 {
            out.write_u8(b':');
            let start = Rooted::new(sliceobject::w_slice_get_start(obj_root.get()));
            let stop = Rooted::new(sliceobject::w_slice_get_stop(obj_root.get()));
            let step = Rooted::new(sliceobject::w_slice_get_step(obj_root.get()));
            write_object(out, start.get(), refs, version, depth - 1)?;
            write_object(out, stop.get(), refs, version, depth - 1)?;
            write_object(out, step.get(), refs, version, depth - 1)?;
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
    has_readinto: bool,
    errors: ErrorSink,
}

impl FileReader {
    fn new(file: PyObjectRef, errors: ErrorSink) -> Result<Self, PyError> {
        let file = Rooted::new(file);
        // Probe exactly once. A missing attribute selects `read`; an exception
        // raised by the lookup itself remains observable.
        let has_readinto = crate::baseobjspace::findattr_result(file.get(), "readinto")?.is_some();
        Ok(Self {
            file,
            scratch: Vec::new(),
            has_readinto,
            errors,
        })
    }

    fn python_error<T>(&mut self, error: PyError) -> Result<T, wire::MarshalError> {
        self.errors.remember(error);
        Err(wire::MarshalError::BadType)
    }

    fn read_with_read(&mut self, n: usize) -> Result<(), wire::MarshalError> {
        let roots = pyre_object::gc_roots::push_roots();
        // Build and root the argument before reloading the file. Integer
        // construction may collect, so neither raw pointer crosses it.
        let count_slot = roots.base();
        roots.pin_root(w_int_new(n as i64));
        let result = match call_method(self.file.get(), "read", &[roots.get(count_slot)]) {
            Ok(result) => result,
            Err(error) => return self.python_error(error),
        };
        let result_slot = count_slot + 1;
        roots.pin_root(result);
        let bytes = match bytes_like(roots.get(result_slot), "load") {
            Ok(bytes) => bytes,
            Err(error) => return self.python_error(error),
        };
        if bytes.len() < n {
            return Err(wire::MarshalError::Eof);
        }
        if bytes.len() > n {
            return self.python_error(PyError::value_error(format!(
                "read() returned too much data: {n} bytes requested, {} returned",
                bytes.len()
            )));
        }
        self.scratch.extend_from_slice(&bytes);
        Ok(())
    }

    fn read_with_readinto(&mut self, n: usize) -> Result<(), wire::MarshalError> {
        let roots = pyre_object::gc_roots::push_roots();
        let buffer_slot = roots.base();
        roots.pin_root(bytearrayobject::w_bytearray_new(n));
        let count = match call_method(self.file.get(), "readinto", &[roots.get(buffer_slot)]) {
            Ok(count) => count,
            // AttributeError from inside `readinto` is an application error,
            // not evidence that the method was absent: absence was cached by
            // `new` before any reads began.
            Err(error) => return self.python_error(error),
        };
        let count = match crate::baseobjspace::int_w(count) {
            Ok(count) => count,
            Err(error) => return self.python_error(error),
        };
        if count < 0 || count as usize > n {
            return self.python_error(PyError::value_error("readinto() returned invalid length"));
        }
        if count as usize != n {
            return Err(wire::MarshalError::Eof);
        }
        self.scratch.extend_from_slice(unsafe {
            bytearrayobject::w_bytearray_data(roots.get(buffer_slot))
        });
        Ok(())
    }
}

impl wire::Read for FileReader {
    fn read_slice(&mut self, n: u32) -> Result<&[u8], wire::MarshalError> {
        // A hostile length prefix must not allocate `n` bytes before the file
        // proves they exist. Fill the reusable result incrementally with a
        // bounded Python buffer (or bounded `read` request) per iteration.
        const CHUNK: usize = 1024 * 1024;
        self.scratch.clear();
        let mut remaining = n as usize;
        while remaining != 0 {
            let chunk = remaining.min(CHUNK);
            if self.has_readinto {
                self.read_with_readinto(chunk)?;
            } else {
                self.read_with_read(chunk)?;
            }
            remaining -= chunk;
        }
        Ok(&self.scratch)
    }
}

#[derive(Clone, Copy)]
struct ErrorSink(*mut Option<PyError>);

impl ErrorSink {
    fn remember(self, error: PyError) {
        unsafe {
            if (*self.0).is_none() {
                *self.0 = Some(error);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct PyreMarshalBag {
    errors: ErrorSink,
}

impl PyreMarshalBag {
    fn new(pending_error: &mut Option<PyError>) -> Self {
        Self {
            errors: ErrorSink(pending_error),
        }
    }

    fn remember_python_error(&self, error: PyError) -> wire::MarshalError {
        // The pointer names the stack-owned slot of the synchronous loads/load
        // call.  `MarshalBag: Copy` copies only this borrow; no state escapes
        // the deserialize call and no TLS side channel is involved.
        self.errors.remember(error);
        wire::MarshalError::BadType
    }
}

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

    fn make_interned_str(&self, value: &Wtf8) -> Rooted {
        let value = Rooted::new(w_str_from_wtf8(value.to_owned()));
        Rooted::new(unsafe { unicodeobject::intern_exact_str(value.get()) })
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

    fn make_tuple_placeholder(&self, len: usize) -> Option<Rooted> {
        Some(Rooted::new(tupleobject::w_tuple_new_array_backed(vec![
            PY_NULL;
            len
        ])))
    }

    fn set_tuple_item(
        &self,
        tuple: &Rooted,
        index: usize,
        value: Rooted,
    ) -> Result<(), wire::MarshalError> {
        unsafe { tupleobject::w_tuple_setitem_initializing(tuple.get(), index, value.get()) }
            .then_some(())
            .ok_or(wire::MarshalError::BadType)
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

    fn make_list_placeholder(&self, len: usize) -> Option<Rooted> {
        Some(Rooted::new(listobject::w_list_new_object(vec![
            PY_NULL;
            len
        ])))
    }

    fn set_list_item(
        &self,
        list: &Rooted,
        index: usize,
        value: Rooted,
    ) -> Result<(), wire::MarshalError> {
        unsafe { listobject::w_list_setitem(list.get(), index as i64, value.get()) }
            .then_some(())
            .ok_or(wire::MarshalError::BadType)
    }

    fn make_set(
        &self,
        elements: impl Iterator<Item = Rooted>,
    ) -> Result<Rooted, wire::MarshalError> {
        let set = Rooted::new(setobject::w_set_new());
        for item in elements {
            let hash = crate::baseobjspace::hash_w_strict(item.get())
                .map_err(|error| self.remember_python_error(error))?;
            unsafe { setobject::w_set_add_hashed_checked(set.get(), item.get(), hash) }.map_err(
                |error| {
                    self.remember_python_error(crate::baseobjspace::map_set_update_error(error))
                },
            )?;
        }
        Ok(set)
    }

    fn make_set_placeholder(&self) -> Option<Rooted> {
        Some(Rooted::new(setobject::w_set_new()))
    }

    fn insert_set_item(&self, set: &Rooted, item: Rooted) -> Result<(), wire::MarshalError> {
        let hash = crate::baseobjspace::hash_w_strict(item.get())
            .map_err(|error| self.remember_python_error(error))?;
        unsafe { setobject::w_set_add_hashed_checked(set.get(), item.get(), hash) }.map_err(
            |error| self.remember_python_error(crate::baseobjspace::map_set_update_error(error)),
        )
    }

    fn make_frozenset(
        &self,
        elements: impl Iterator<Item = Rooted>,
    ) -> Result<Rooted, wire::MarshalError> {
        let set = Rooted::new(setobject::w_frozenset_new());
        for item in elements {
            let hash = crate::baseobjspace::hash_w_strict(item.get())
                .map_err(|error| self.remember_python_error(error))?;
            unsafe { setobject::w_set_add_hashed_checked(set.get(), item.get(), hash) }.map_err(
                |error| {
                    self.remember_python_error(crate::baseobjspace::map_set_update_error(error))
                },
            )?;
        }
        Ok(set)
    }

    fn make_dict(
        &self,
        elements: impl Iterator<Item = (Rooted, Rooted)>,
    ) -> Result<Rooted, wire::MarshalError> {
        let dict = Rooted::new(w_dict_new());
        for (key, value) in elements {
            unsafe { w_dict_store_checked(dict.get(), key.get(), value.get()) }.map_err(|_| {
                self.remember_python_error(crate::baseobjspace::take_pending_dict_key_error(
                    key.get(),
                ))
            })?;
        }
        Ok(dict)
    }

    fn make_dict_placeholder(&self) -> Option<Rooted> {
        Some(Rooted::new(w_dict_new()))
    }

    fn insert_dict_item(
        &self,
        dict: &Rooted,
        key: Rooted,
        value: Rooted,
    ) -> Result<(), wire::MarshalError> {
        unsafe { w_dict_store_checked(dict.get(), key.get(), value.get()) }.map_err(|_| {
            self.remember_python_error(crate::baseobjspace::take_pending_dict_key_error(key.get()))
        })
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

    // `deserialize_code_value_inner` reads a code object's fields as bag
    // values, so the three shapes `write_code` writes them in — bytes for
    // co_code/co_localspluskinds/co_linetable/co_exceptiontable, tuples for
    // co_consts/co_names/co_localsplusnames, and str for the three names —
    // have to be readable back out of the wrapped objects.  Without them the
    // reader has no way to reach the bytes and every code object it decodes
    // fails as `BadType`.
    fn bytes_from_value(&self, value: &Rooted) -> Option<Vec<u8>> {
        let obj = value.get();
        unsafe { bytesobject::is_bytes(obj) }
            .then(|| unsafe { bytesobject::w_bytes_data(obj) }.to_vec())
    }

    fn str_from_value(&self, value: &Rooted) -> Option<String> {
        let obj = value.get();
        unsafe { unicodeobject::is_str(obj) }.then(|| {
            unsafe { unicodeobject::w_str_get_wtf8(obj) }
                .to_string_lossy()
                .into_owned()
        })
    }

    fn tuple_elements_from_value(&self, value: &Rooted) -> Option<Vec<Rooted>> {
        let tuple = Rooted::new(value.get());
        if !unsafe { is_tuple(tuple.get()) } {
            return None;
        }
        // Each element is rooted as it is read: nothing here allocates, but
        // the caller holds these across the reads for the remaining fields.
        (0..unsafe { w_tuple_len(tuple.get()) })
            .map(|index| unsafe { w_tuple_getitem(tuple.get(), index as i64) }.map(Rooted::new))
            .collect()
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
    let mut pending_error = None;
    let result = match wire::deserialize_value(&mut reader, PyreMarshalBag::new(&mut pending_error))
    {
        Ok(result) => result,
        Err(error) => return Err(pending_error.unwrap_or_else(|| marshal_error(error))),
    };
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
            let mut pending_error = None;
            let bag = PyreMarshalBag::new(&mut pending_error);
            let mut reader = FileReader::new(file, bag.errors)?;
            let result = match wire::deserialize_value(&mut reader, bag) {
                Ok(result) => result,
                Err(error) => {
                    if let Some(error) = pending_error.take() {
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
