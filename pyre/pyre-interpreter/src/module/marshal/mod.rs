//! `marshal` — PyPy `pypy/module/marshal/` object glue over the shared
//! RustPython compiler-core marshal format.
//!
//! PyPy's `interp_marshal.py` owns the Python object dispatch while
//! `marshal_impl.py` defines the wire typecodes.  Pyre follows that split:
//! this module maps `W_Root` objects to the shared wire implementation, and
//! compiler-core serializes/deserializes the authoritative `CodeObject`.

use malachite_bigint::BigInt;
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
    let mut error = PyError::value_error(message);
    if let Some(cls) = crate::builtins::lookup_exc_class("EOFError") {
        let args = [cls, w_str_new(message)];
        if let Ok(exc) = crate::builtins::exc_exception_new(&args) {
            error.exc_object = exc;
        }
    }
    error
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
        Ok(unsafe { bytesobject::bytes_like_data(obj) }.to_vec())
    } else {
        Err(PyError::type_error(format!(
            "{function}() argument must be a bytes-like object"
        )))
    }
}

/// Transient equivalent of PyPy's `Marshaller.all_refs` dict.  A VecMap is
/// sufficient because marshal streams normally contain few shared objects;
/// each entry is a shadow-stack slot so a collection updates object identity.
struct WriterRefs {
    slots: Vec<usize>,
}

impl WriterRefs {
    fn new() -> Self {
        Self { slots: Vec::new() }
    }

    fn find(&self, obj: PyObjectRef) -> Option<u32> {
        let obj =
            pyre_object::gc_hook::try_gc_current_object_address(obj as *mut u8) as PyObjectRef;
        self.slots
            .iter()
            .position(|&slot| std::ptr::eq(pyre_object::gc_roots::shadow_stack_get(slot), obj))
            .and_then(|index| u32::try_from(index).ok())
    }

    fn reserve(&mut self, obj: PyObjectRef) -> Result<(), PyError> {
        if self.slots.len() >= i32::MAX as usize {
            return Err(PyError::value_error("too many objects to marshal"));
        }
        let slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(obj);
        self.slots.push(slot);
        Ok(())
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
    if depth == 0 {
        return Err(PyError::value_error("object too deeply nested to marshal"));
    }

    if !is_singleton(obj)
        && let Some(table) = refs.as_ref()
        && let Some(index) = table.find(obj)
    {
        out.write_u8(b'r');
        out.write_u32(index);
        return Ok(());
    }

    let type_pos = out.len();
    let use_ref = refs.is_some() && !is_singleton(obj);
    if use_ref {
        refs.as_mut().unwrap().reserve(obj)?;
    }

    unsafe {
        if is_none(obj) {
            out.write_u8(b'N');
        } else if crate::builtins::lookup_exc_class("StopIteration") == Some(obj) {
            out.write_u8(b'S');
        } else if is_bool(obj) {
            out.write_u8(if w_bool_get_value(obj) { b'T' } else { b'F' });
        } else if is_ellipsis(obj) {
            out.write_u8(b'.');
        } else if is_int_or_long(obj) {
            let value = crate::builtins::obj_to_bigint(obj);
            wire::serialize_value::<_, ConstantData>(out, DumpableValue::Integer(&value))
                .unwrap_or_else(|never| match never {});
        } else if is_float(obj) {
            out.write_u8(b'g');
            out.write_u64(w_float_get_value(obj).to_bits());
        } else if is_complex(obj) {
            out.write_u8(b'y');
            out.write_u64(w_complex_get_real(obj).to_bits());
            out.write_u64(w_complex_get_imag(obj).to_bits());
        } else if is_str(obj) {
            let value = w_str_get_wtf8(obj).as_bytes();
            out.write_u8(b'u');
            out.write_u32(
                value
                    .len()
                    .try_into()
                    .map_err(|_| PyError::value_error("object too large to marshal"))?,
            );
            out.write_slice(value);
        } else if bytesobject::is_bytes_like(obj) {
            let value = bytesobject::bytes_like_data(obj);
            out.write_u8(b's');
            out.write_u32(
                value
                    .len()
                    .try_into()
                    .map_err(|_| PyError::value_error("object too large to marshal"))?,
            );
            out.write_slice(value);
        } else if is_tuple(obj) {
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
        } else if is_list(obj) {
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
        } else if is_dict(obj) {
            out.write_u8(b'{');
            for (key, value) in dictmultiobject::w_dict_items(obj) {
                write_object(out, key, refs, version, depth - 1)?;
                write_object(out, value, refs, version, depth - 1)?;
            }
            out.write_u8(b'0');
        } else if setobject::is_set_or_frozenset(obj) {
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
        } else if crate::pycode::is_code(obj) {
            let ptr = crate::pycode::w_code_get_ptr(obj) as *const crate::CodeObject;
            if ptr.is_null() {
                return Err(PyError::value_error("unmarshallable object"));
            }
            out.write_u8(b'c');
            wire::serialize_code(out, &*ptr);
        } else if is_slice(obj) && version >= 5 {
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

    fn make_int(&self, value: BigInt) -> Rooted {
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

fn parse_version(positional: &[PyObjectRef], kwargs: Option<PyObjectRef>) -> Result<i32, PyError> {
    let value =
        crate::builtins::kwarg_get(kwargs, "version").or_else(|| positional.get(1).copied());
    match value {
        Some(value) => Ok(crate::baseobjspace::int_w(value)? as i32),
        None => Ok(wire::FORMAT_VERSION as i32),
    }
}

fn parse_allow_code(kwargs: Option<PyObjectRef>) -> Result<bool, PyError> {
    match crate::builtins::kwarg_get(kwargs, "allow_code") {
        Some(value) => crate::baseobjspace::is_true(value),
        None => Ok(true),
    }
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

fn dumps_impl(args: &[PyObjectRef]) -> PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["version", "allow_code"], "dumps")?;
    let Some(&value) = positional.first() else {
        return Err(PyError::type_error(
            "dumps() missing required argument 'value'",
        ));
    };
    if positional.len() > 2 {
        return Err(PyError::type_error("dumps() takes at most 2 arguments"));
    }
    let version = parse_version(positional, kwargs)?;
    let allow_code = parse_allow_code(kwargs)?;
    if !allow_code {
        reject_code(value)?;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(value);
    let mut out = Vec::new();
    let mut refs = (version >= 3).then(WriterRefs::new);
    write_object(&mut out, value, &mut refs, version, MAX_DEPTH)?;
    Ok(bytesobject::w_bytes_from_bytes(&out))
}

fn loads_impl(args: &[PyObjectRef]) -> PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["allow_code"], "loads")?;
    let Some(&data) = positional.first() else {
        return Err(PyError::type_error(
            "loads() missing required argument 'bytes'",
        ));
    };
    if positional.len() != 1 {
        return Err(PyError::type_error("loads() takes exactly 1 argument"));
    }
    let allow_code = parse_allow_code(kwargs)?;
    let data = bytes_like(data, "loads")?;
    let _roots = pyre_object::gc_roots::push_roots();
    let mut reader: &[u8] = &data;
    let result = wire::deserialize_value(&mut reader, PyreMarshalBag).map_err(marshal_error)?;
    let result = result.get();
    if !allow_code {
        reject_code(result)?;
    }
    Ok(result)
}

fn dump_impl(args: &[PyObjectRef]) -> PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["version", "allow_code"], "dump")?;
    if positional.len() < 2 || positional.len() > 3 {
        return Err(PyError::type_error("dump() expected 2 or 3 arguments"));
    }
    let mut dump_args = vec![positional[0]];
    if let Some(version) = positional.get(2) {
        dump_args.push(*version);
    }
    if let Some(kwargs) = kwargs {
        dump_args.push(kwargs);
    }
    let bytes = dumps_impl(&dump_args)?;
    call_method(positional[1], "write", &[bytes])?;
    Ok(w_none())
}

fn load_impl(args: &[PyObjectRef]) -> PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["allow_code"], "load")?;
    if positional.len() != 1 {
        return Err(PyError::type_error("load() takes exactly 1 argument"));
    }
    let file = positional[0];
    let before = crate::baseobjspace::int_w(call_method(file, "tell", &[])?)?;
    let bytes_obj = call_method(file, "read", &[])?;
    let data = bytes_like(bytes_obj, "load")?;
    let allow_code = parse_allow_code(kwargs)?;
    let _roots = pyre_object::gc_roots::push_roots();
    let mut reader = wire::Cursor {
        data: data.as_slice(),
        position: 0,
    };
    let result = wire::deserialize_value(&mut reader, PyreMarshalBag).map_err(marshal_error)?;
    let new_position = w_int_new(before.saturating_add(reader.position as i64));
    call_method(file, "seek", &[new_position])?;
    let result = result.get();
    if !allow_code {
        reject_code(result)?;
    }
    Ok(result)
}

crate::py_module! {
    "marshal",
    int_constants: {
        "version" => wire::FORMAT_VERSION as i64,
    },
    functions: {
        "dump" / * = dump_impl,
        "dumps" / * = dumps_impl,
        "load" / * = load_impl,
        "loads" / * = loads_impl,
    },
}
