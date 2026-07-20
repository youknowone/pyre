//! `array` module — the `array.array` type.
//!
//! PyPy: pypy/module/array/interp_array.py + reconstructor.py
//!
//! The element storage lives in `pyre_object::interp_array` (an off-GC
//! `*mut Vec<u8>` of native-order bytes).  This module supplies the
//! interpreter-level behaviour: range-checked packing of a Python object
//! into element bytes, the bound methods, the `__new__` constructor, and
//! the module-level `_array_reconstructor` unpickler.

use crate::baseobjspace::{float_w, int_w, uint_w};
use crate::objspace::descroperation::{CompareOp, compare};
use crate::{PyError, PyErrorKind, PyResult, make_builtin_function_with_arity, module_ns_store};
use pyre_object::interp_array as arr;
use pyre_object::{PY_NULL, PyObjectRef};
use rustpython_wtf8::{CodePoint, Wtf8Buf};

/// A fixed inline byte buffer for one packed element (≤ 8 bytes).
type Bytes = [u8; 8];

// ──────────────────────────────────────────────────────────────────────
// Packing: Python object → native-order element bytes (range-checked).
// ──────────────────────────────────────────────────────────────────────

/// Pack `w` into `out[..itemsize]` per `typecode`, returning the item size.
/// `interp_array.py W_Array.item_w` + the per-`TypeCode` overflow checks.
fn pack_into(typecode: u8, w: PyObjectRef, out: &mut Bytes) -> Result<usize, PyError> {
    fn signed_range(v: i64, lo: i64, hi: i64, name: &str) -> Result<(), PyError> {
        if v < lo {
            Err(PyError::overflow_error(format!(
                "{name} is less than minimum"
            )))
        } else if v > hi {
            Err(PyError::overflow_error(format!(
                "{name} is greater than maximum"
            )))
        } else {
            Ok(())
        }
    }
    let n = match typecode {
        b'b' => {
            let v = int_w(w)?;
            signed_range(v, i8::MIN as i64, i8::MAX as i64, "signed char")?;
            out[..1].copy_from_slice(&(v as i8).to_ne_bytes());
            1
        }
        b'B' => {
            let v = int_w(w)?;
            signed_range(v, 0, u8::MAX as i64, "unsigned byte integer")?;
            out[..1].copy_from_slice(&(v as u8).to_ne_bytes());
            1
        }
        b'h' => {
            let v = int_w(w)?;
            signed_range(v, i16::MIN as i64, i16::MAX as i64, "signed short")?;
            out[..2].copy_from_slice(&(v as i16).to_ne_bytes());
            2
        }
        b'H' => {
            let v = int_w(w)?;
            signed_range(v, 0, u16::MAX as i64, "unsigned short")?;
            out[..2].copy_from_slice(&(v as u16).to_ne_bytes());
            2
        }
        b'i' => {
            let v = int_w(w)?;
            signed_range(v, i32::MIN as i64, i32::MAX as i64, "signed int")?;
            out[..4].copy_from_slice(&(v as i32).to_ne_bytes());
            4
        }
        b'I' => {
            let v = int_w(w)?;
            signed_range(v, 0, u32::MAX as i64, "unsigned int")?;
            out[..4].copy_from_slice(&(v as u32).to_ne_bytes());
            4
        }
        b'l' | b'q' => {
            // C long / long long on 64-bit — full i64 range; `int_w` itself
            // raises OverflowError outside it.
            let v = int_w(w)?;
            out[..8].copy_from_slice(&v.to_ne_bytes());
            8
        }
        b'L' | b'Q' => {
            // Unsigned 64-bit — `uint_w` handles bignums and rejects negatives.
            let v = uint_w(w)?;
            out[..8].copy_from_slice(&v.to_ne_bytes());
            8
        }
        b'f' => {
            let v = float_w(w)? as f32;
            out[..4].copy_from_slice(&v.to_ne_bytes());
            4
        }
        b'd' => {
            let v = float_w(w)?;
            out[..8].copy_from_slice(&v.to_ne_bytes());
            8
        }
        b'u' => {
            let cp = unicode_char_w(w)?;
            out[..4].copy_from_slice(&cp.to_ne_bytes());
            4
        }
        _ => return Err(PyError::value_error("bad typecode")),
    };
    Ok(n)
}

/// Extract a single Unicode code point from a length-1 str (`'u'` items).
fn unicode_char_w(w: PyObjectRef) -> Result<u32, PyError> {
    if !unsafe { pyre_object::is_str(w) } {
        return Err(PyError::type_error(
            "array item must be a unicode character, not a different type",
        ));
    }
    let s = unsafe { pyre_object::unicodeobject::w_str_get_wtf8(w) };
    let mut points = s.code_points();
    match (points.next(), points.next()) {
        (Some(c), None) => Ok(c.to_u32()),
        _ => Err(PyError::type_error(
            "array item must be a unicode character, not str",
        )),
    }
}

// ──────────────────────────────────────────────────────────────────────
// Core mutation helpers.
// ──────────────────────────────────────────────────────────────────────

/// Append one packed value (`append` / single-element `extend`).
fn array_append(obj: PyObjectRef, w_value: PyObjectRef) -> Result<(), PyError> {
    array_check_resize(obj)?;
    let tc = unsafe { arr::w_array_typecode(obj) };
    let mut buf: Bytes = [0u8; 8];
    let n = pack_into(tc, w_value, &mut buf)?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.extend_from_slice(&buf[..n]);
    Ok(())
}

fn array_check_resize(obj: PyObjectRef) -> Result<(), PyError> {
    if unsafe { arr::w_array_exports(obj) } != 0 {
        Err(PyError::new(
            PyErrorKind::BufferError,
            "cannot resize an array that is exporting buffers",
        ))
    } else {
        Ok(())
    }
}

/// Extend from any iterable, packing each element (`descr_extend`).
fn array_extend_iterable(obj: PyObjectRef, w_iterable: PyObjectRef) -> Result<(), PyError> {
    array_check_resize(obj)?;
    // A fast path for same-typecode arrays: raw byte concat.
    if unsafe { arr::is_array(w_iterable) } {
        let dst_tc = unsafe { arr::w_array_typecode(obj) };
        let src_tc = unsafe { arr::w_array_typecode(w_iterable) };
        if dst_tc != src_tc {
            return Err(PyError::type_error(
                "can only extend with array of same kind",
            ));
        }
        let src_bytes = unsafe { arr::w_array_bytes(w_iterable) }.to_vec();
        let vec = unsafe { arr::w_array_vec_mut(obj) };
        vec.extend_from_slice(&src_bytes);
        return Ok(());
    }
    let w_iter = crate::baseobjspace::iter(w_iterable)?;
    loop {
        match crate::baseobjspace::next(w_iter) {
            Ok(w_item) => array_append(obj, w_item)?,
            Err(e) if e.kind == PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Append raw bytes (`frombytes`); length must be a multiple of itemsize.
fn array_frombytes(obj: PyObjectRef, bytes: &[u8]) -> Result<(), PyError> {
    array_check_resize(obj)?;
    let isz = unsafe { arr::w_array_itemsize(obj) };
    if bytes.len() % isz != 0 {
        return Err(PyError::value_error(
            "bytes length not a multiple of item size",
        ));
    }
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.extend_from_slice(bytes);
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// __new__
// ──────────────────────────────────────────────────────────────────────

/// `array.__new__(cls, typecode, [initializer])` — `interp_array.py w_array`.
fn array_descr_new(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 2 {
        return Err(PyError::type_error(
            "array() takes at least 1 argument (0 given)",
        ));
    }
    let cls = args[0];
    let w_typecode = args[1];
    // typecode must be a 1-character str.
    if !unsafe { pyre_object::is_str(w_typecode) } {
        return Err(PyError::type_error(
            "array() argument 1 must be a unicode character, not a different type",
        ));
    }
    let tc_str = unsafe { pyre_object::unicodeobject::w_str_get_value(w_typecode) };
    let tc_bytes = tc_str.as_bytes();
    if tc_bytes.len() != 1 {
        return Err(PyError::type_error(
            "array() argument 1 must be a unicode character, not str",
        ));
    }
    let typecode = tc_bytes[0];
    let itemsize = arr::typecode_itemsize(typecode).ok_or_else(|| {
        PyError::value_error("bad typecode (must be b, B, u, h, H, i, I, l, L, q, Q, f or d)")
    })?;
    let obj = arr::w_array_new(typecode, itemsize);
    // Subclass: retag the fresh array with the requested class.
    if !cls.is_null() && unsafe { pyre_object::is_type(cls) } {
        if let Some(canonical) = crate::typedef::gettypefor(&pyre_object::interp_array::ARRAY_TYPE)
        {
            if !std::ptr::eq(cls, canonical) {
                unsafe {
                    (*obj).w_class = cls;
                }
            }
        }
    }
    // Optional initializer.
    if args.len() >= 3 {
        let w_init = args[2];
        if unsafe { pyre_object::is_str(w_init) } {
            if typecode == b'u' {
                array_fromunicode(obj, w_init)?;
            } else {
                return Err(PyError::type_error(format!(
                    "cannot use a str to initialize an array with typecode '{}'",
                    typecode as char
                )));
            }
        } else if unsafe { pyre_object::bytesobject::is_bytes_like(w_init) } {
            let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(w_init) }.to_vec();
            array_frombytes(obj, &bytes)?;
        } else {
            array_extend_iterable(obj, w_init)?;
        }
    }
    Ok(obj)
}

/// `fromunicode` — append code points of a str to a `'u'` array.
fn array_fromunicode(obj: PyObjectRef, w_str: PyObjectRef) -> Result<(), PyError> {
    array_check_resize(obj)?;
    if unsafe { arr::w_array_typecode(obj) } != b'u' {
        return Err(PyError::value_error(
            "fromunicode() may only be called on unicode type arrays",
        ));
    }
    if !unsafe { pyre_object::is_str(w_str) } {
        return Err(PyError::type_error("fromunicode() argument must be str"));
    }
    let s = unsafe { pyre_object::unicodeobject::w_str_get_wtf8(w_str) };
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    for cp in s.code_points() {
        vec.extend_from_slice(&cp.to_u32().to_ne_bytes());
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Indexing.
// ──────────────────────────────────────────────────────────────────────

/// Normalize an integer index against `len`, raising IndexError out of range.
fn index_in_range(w_index: PyObjectRef, len: usize, what: &str) -> Result<usize, PyError> {
    let mut i = crate::builtins::getindex_w(w_index)?;
    if i < 0 {
        i += len as i64;
    }
    if i < 0 || i >= len as i64 {
        return Err(PyError::new(
            PyErrorKind::IndexError,
            format!("{what} index out of range"),
        ));
    }
    Ok(i as usize)
}

/// slice element count for `(start, stop, step)`.
fn slice_length(start: i64, stop: i64, step: i64) -> i64 {
    if step > 0 {
        if stop > start {
            (stop - start - 1) / step + 1
        } else {
            0
        }
    } else if start > stop {
        (start - stop - 1) / (-step) + 1
    } else {
        0
    }
}

fn array_getitem(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__getitem__")?;
    let obj = args[0];
    let key = args[1];
    let len = unsafe { arr::w_array_len(obj) };
    let isz = unsafe { arr::w_array_itemsize(obj) };
    let tc = unsafe { arr::w_array_typecode(obj) };
    if unsafe { pyre_object::sliceobject::is_slice(key) } {
        let (start, stop, step) = crate::sliceobject::indices3(
            unsafe { pyre_object::sliceobject::w_slice_get_start(key) },
            unsafe { pyre_object::sliceobject::w_slice_get_stop(key) },
            unsafe { pyre_object::sliceobject::w_slice_get_step(key) },
            len as i64,
        )?;
        let n = slice_length(start, stop, step);
        let src = unsafe { arr::w_array_bytes(obj) }.to_vec();
        let mut out: Vec<u8> = Vec::with_capacity(n as usize * isz);
        let mut i = start;
        for _ in 0..n {
            let off = i as usize * isz;
            out.extend_from_slice(&src[off..off + isz]);
            i += step;
        }
        return Ok(arr::w_array_from_bytes(tc, isz as u8, out));
    }
    let i = index_in_range(key, len, "array")?;
    Ok(unsafe { arr::w_array_unpack_item(obj, i) })
}

fn array_setitem(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 3, "array.__setitem__")?;
    let obj = args[0];
    let key = args[1];
    let w_value = args[2];
    let len = unsafe { arr::w_array_len(obj) };
    let isz = unsafe { arr::w_array_itemsize(obj) };
    let tc = unsafe { arr::w_array_typecode(obj) };
    if unsafe { pyre_object::sliceobject::is_slice(key) } {
        // Slice assignment accepts only an array of the same typecode.
        if !unsafe { arr::is_array(w_value) } || unsafe { arr::w_array_typecode(w_value) } != tc {
            return Err(PyError::type_error(
                "can only assign array (not \"other\") to array slice",
            ));
        }
        let (start, stop, step) = crate::sliceobject::indices3(
            unsafe { pyre_object::sliceobject::w_slice_get_start(key) },
            unsafe { pyre_object::sliceobject::w_slice_get_stop(key) },
            unsafe { pyre_object::sliceobject::w_slice_get_step(key) },
            len as i64,
        )?;
        let n = slice_length(start, stop, step);
        let src = unsafe { arr::w_array_bytes(w_value) }.to_vec();
        let src_len = src.len() / isz;
        if step == 1 {
            // Contiguous: may resize.
            let vec = unsafe { arr::w_array_vec_mut(obj) };
            let lo = start as usize * isz;
            let hi = stop.max(start) as usize * isz;
            vec.splice(lo..hi, src.iter().copied());
        } else {
            if src_len as i64 != n {
                return Err(PyError::value_error(format!(
                    "attempt to assign array of size {src_len} to extended slice of size {n}"
                )));
            }
            let vec = unsafe { arr::w_array_vec_mut(obj) };
            let mut i = start;
            for k in 0..n {
                let dst = i as usize * isz;
                let s = k as usize * isz;
                vec[dst..dst + isz].copy_from_slice(&src[s..s + isz]);
                i += step;
            }
        }
        return Ok(pyre_object::w_none());
    }
    let i = index_in_range(key, len, "array")?;
    let mut buf: Bytes = [0u8; 8];
    // pack_into may run user code (`__index__`/`__int__`/`__float__`) that
    // resizes the array mid-assignment (gh-142555); re-validate the slot
    // against the current length before writing.
    let n = pack_into(tc, w_value, &mut buf)?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    let end = i * isz + n;
    if end > vec.len() {
        return Err(PyError::new(
            PyErrorKind::IndexError,
            "array assignment index out of range".to_string(),
        ));
    }
    vec[i * isz..end].copy_from_slice(&buf[..n]);
    Ok(pyre_object::w_none())
}

fn array_delitem(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__delitem__")?;
    let obj = args[0];
    let key = args[1];
    let len = unsafe { arr::w_array_len(obj) };
    let isz = unsafe { arr::w_array_itemsize(obj) };
    if unsafe { pyre_object::sliceobject::is_slice(key) } {
        let (start, stop, step) = crate::sliceobject::indices3(
            unsafe { pyre_object::sliceobject::w_slice_get_start(key) },
            unsafe { pyre_object::sliceobject::w_slice_get_stop(key) },
            unsafe { pyre_object::sliceobject::w_slice_get_step(key) },
            len as i64,
        )?;
        let n = slice_length(start, stop, step);
        // Collect element indices to drop, then rebuild the buffer.
        let mut drop_set: Vec<usize> = Vec::with_capacity(n as usize);
        let mut i = start;
        for _ in 0..n {
            drop_set.push(i as usize);
            i += step;
        }
        drop_set.sort_unstable();
        let src = unsafe { arr::w_array_bytes(obj) }.to_vec();
        let mut out: Vec<u8> = Vec::with_capacity(src.len());
        let mut di = 0usize;
        for e in 0..len {
            if di < drop_set.len() && drop_set[di] == e {
                di += 1;
                continue;
            }
            out.extend_from_slice(&src[e * isz..e * isz + isz]);
        }
        let vec = unsafe { arr::w_array_vec_mut(obj) };
        *vec = out;
        return Ok(pyre_object::w_none());
    }
    let i = index_in_range(key, len, "array")?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.drain(i * isz..i * isz + isz);
    Ok(pyre_object::w_none())
}

// ──────────────────────────────────────────────────────────────────────
// Sequence / list methods.
// ──────────────────────────────────────────────────────────────────────

fn array_len(args: &[PyObjectRef]) -> PyResult {
    Ok(pyre_object::w_int_new(
        unsafe { arr::w_array_len(args[0]) } as i64
    ))
}

fn array_iter(args: &[PyObjectRef]) -> PyResult {
    let len = unsafe { arr::w_array_len(args[0]) };
    Ok(pyre_object::w_seq_iter_new(args[0], len))
}

/// Ensure a method received at least `min_total` positional slots
/// (`self` included); otherwise raise the "takes exactly N argument(s)"
/// TypeError rather than panicking on an out-of-range `args` index.
fn check_arity(args: &[PyObjectRef], min_total: usize, name: &str) -> Result<(), PyError> {
    if args.len() < min_total {
        let want = min_total - 1;
        let noun = if want == 1 { "argument" } else { "arguments" };
        return Err(PyError::type_error(format!(
            "{name}() takes exactly {want} {noun} ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(())
}

fn array_append_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.append")?;
    array_append(args[0], args[1])?;
    Ok(pyre_object::w_none())
}

fn array_extend_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.extend")?;
    array_extend_iterable(args[0], args[1])?;
    Ok(pyre_object::w_none())
}

fn array_insert_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 3, "array.insert")?;
    let obj = args[0];
    array_check_resize(obj)?;
    let len = unsafe { arr::w_array_len(obj) };
    let isz = unsafe { arr::w_array_itemsize(obj) };
    let tc = unsafe { arr::w_array_typecode(obj) };
    // Clamp index like list.insert.
    let mut i = crate::builtins::getindex_w(args[1])?;
    if i < 0 {
        i += len as i64;
        if i < 0 {
            i = 0;
        }
    }
    if i > len as i64 {
        i = len as i64;
    }
    let mut buf: Bytes = [0u8; 8];
    let n = pack_into(tc, args[2], &mut buf)?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    let at = i as usize * isz;
    vec.splice(at..at, buf[..n].iter().copied());
    Ok(pyre_object::w_none())
}

fn array_pop_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    array_check_resize(obj)?;
    let len = unsafe { arr::w_array_len(obj) };
    if len == 0 {
        return Err(PyError::new(
            PyErrorKind::IndexError,
            "pop from empty array".to_string(),
        ));
    }
    let mut i = if args.len() >= 2 {
        crate::builtins::getindex_w(args[1])?
    } else {
        -1
    };
    if i < 0 {
        i += len as i64;
    }
    if i < 0 || i >= len as i64 {
        return Err(PyError::new(
            PyErrorKind::IndexError,
            "pop index out of range".to_string(),
        ));
    }
    let isz = unsafe { arr::w_array_itemsize(obj) };
    let w_val = unsafe { arr::w_array_unpack_item(obj, i as usize) };
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.drain(i as usize * isz..i as usize * isz + isz);
    Ok(w_val)
}

fn array_remove_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.remove")?;
    let obj = args[0];
    array_check_resize(obj)?;
    let idx = array_find(obj, args[1])?;
    match idx {
        Some(i) => {
            let isz = unsafe { arr::w_array_itemsize(obj) };
            let vec = unsafe { arr::w_array_vec_mut(obj) };
            vec.drain(i * isz..i * isz + isz);
            Ok(pyre_object::w_none())
        }
        None => Err(PyError::value_error("array.remove(x): x not in array")),
    }
}

/// First index whose element equals `w_value`, via `==`.
fn array_find(obj: PyObjectRef, w_value: PyObjectRef) -> Result<Option<usize>, PyError> {
    let len = unsafe { arr::w_array_len(obj) };
    for i in 0..len {
        let w_item = unsafe { arr::w_array_unpack_item(obj, i) };
        if crate::baseobjspace::eq_w(w_item, w_value)? {
            return Ok(Some(i));
        }
    }
    Ok(None)
}

fn array_index_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.index")?;
    let obj = args[0];
    let w_value = args[1];
    let len = unsafe { arr::w_array_len(obj) } as i64;
    // Optional start/stop, unwrapped via __index__, clamped like descr_index.
    let mut start = if args.len() >= 3 {
        crate::builtins::getindex_w(args[2])?
    } else {
        0
    };
    let mut stop = if args.len() >= 4 {
        crate::builtins::getindex_w(args[3])?
    } else {
        len
    };
    if start < 0 {
        start += len;
        if start < 0 {
            start = 0;
        }
    }
    if stop < 0 {
        stop += len;
        if stop < 0 {
            stop = 0;
        }
    }
    if stop > len {
        stop = len;
    }
    let mut i = start;
    while i < stop {
        let w_item = unsafe { arr::w_array_unpack_item(obj, i as usize) };
        if crate::baseobjspace::eq_w(w_item, w_value)? {
            return Ok(pyre_object::w_int_new(i));
        }
        i += 1;
    }
    Err(PyError::value_error("array.index(x): x not in array"))
}

fn array_clear_method(args: &[PyObjectRef]) -> PyResult {
    // descr_clear — empty the buffer, preserving typecode/itemsize.
    array_check_resize(args[0])?;
    unsafe { arr::w_array_vec_mut(args[0]) }.clear();
    Ok(pyre_object::w_none())
}

fn array_release_buffer(args: &[PyObjectRef]) -> PyResult {
    unsafe { arr::w_array_exports_decref(args[0]) };
    Ok(pyre_object::w_none())
}

fn array_count_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.count")?;
    let obj = args[0];
    let len = unsafe { arr::w_array_len(obj) };
    let mut count = 0i64;
    for i in 0..len {
        let w_item = unsafe { arr::w_array_unpack_item(obj, i) };
        if crate::baseobjspace::eq_w(w_item, args[1])? {
            count += 1;
        }
    }
    Ok(pyre_object::w_int_new(count))
}

fn array_reverse_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let isz = unsafe { arr::w_array_itemsize(obj) };
    let len = unsafe { arr::w_array_len(obj) };
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    let mut lo = 0usize;
    let mut hi = len;
    while lo < hi {
        hi -= 1;
        for k in 0..isz {
            vec.swap(lo * isz + k, hi * isz + k);
        }
        lo += 1;
    }
    Ok(pyre_object::w_none())
}

// ──────────────────────────────────────────────────────────────────────
// Conversion methods.
// ──────────────────────────────────────────────────────────────────────

fn array_tolist_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let len = unsafe { arr::w_array_len(obj) };
    let mut items = Vec::with_capacity(len);
    for i in 0..len {
        items.push(unsafe { arr::w_array_unpack_item(obj, i) });
    }
    Ok(pyre_object::w_list_new(items))
}

fn array_fromlist_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.fromlist")?;
    array_extend_iterable(args[0], args[1])?;
    Ok(pyre_object::w_none())
}

fn array_tobytes_method(args: &[PyObjectRef]) -> PyResult {
    let bytes = unsafe { arr::w_array_bytes(args[0]) };
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(bytes))
}

fn array_frombytes_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.frombytes")?;
    if !unsafe { pyre_object::bytesobject::is_bytes_like(args[1]) } {
        return Err(PyError::type_error("a bytes-like object is required"));
    }
    let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(args[1]) }.to_vec();
    array_frombytes(args[0], &bytes)?;
    Ok(pyre_object::w_none())
}

fn array_tounicode_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    if unsafe { arr::w_array_typecode(obj) } != b'u' {
        return Err(PyError::value_error(
            "tounicode() may only be called on unicode type arrays",
        ));
    }
    let len = unsafe { arr::w_array_len(obj) };
    let bytes = unsafe { arr::w_array_bytes(obj) };
    let mut wb = Wtf8Buf::new();
    for i in 0..len {
        let cp = u32::from_ne_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        if let Some(point) = CodePoint::from_u32(cp) {
            wb.push(point);
        }
    }
    Ok(pyre_object::unicodeobject::w_str_from_wtf8(wb))
}

fn array_fromunicode_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.fromunicode")?;
    array_fromunicode(args[0], args[1])?;
    Ok(pyre_object::w_none())
}

// ──────────────────────────────────────────────────────────────────────
// Misc / dunder.
// ──────────────────────────────────────────────────────────────────────

fn array_contains_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__contains__")?;
    Ok(pyre_object::w_bool_from(
        array_find(args[0], args[1])?.is_some(),
    ))
}

fn array_buffer_info_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let addr = unsafe { arr::w_array_bytes(obj) }.as_ptr() as i64;
    let len = unsafe { arr::w_array_len(obj) } as i64;
    Ok(pyre_object::w_tuple_new(vec![
        pyre_object::w_int_new(addr),
        pyre_object::w_int_new(len),
    ]))
}

fn array_byteswap_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let isz = unsafe { arr::w_array_itemsize(obj) };
    if !matches!(isz, 1 | 2 | 4 | 8) {
        return Err(PyError::runtime_error(
            "don't know how to byteswap this array type",
        ));
    }
    let len = unsafe { arr::w_array_len(obj) };
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    for i in 0..len {
        vec[i * isz..i * isz + isz].reverse();
    }
    Ok(pyre_object::w_none())
}

fn array_copy_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let tc = unsafe { arr::w_array_typecode(obj) };
    let isz = unsafe { arr::w_array_itemsize(obj) } as u8;
    let bytes = unsafe { arr::w_array_bytes(obj) }.to_vec();
    Ok(arr::w_array_from_bytes(tc, isz, bytes))
}

/// `array.__repr__` formatting (`interp_array.py descr_repr`).  Shared with
/// `display::py_repr` so an array nested in a list / error / tuple formats
/// the same way.
pub fn array_repr_string(obj: PyObjectRef) -> Result<String, PyError> {
    let tc = unsafe { arr::w_array_typecode(obj) } as char;
    let len = unsafe { arr::w_array_len(obj) };
    if len == 0 {
        return Ok(format!("array('{tc}')"));
    }
    if tc == 'u' {
        let s = array_tounicode_method(&[obj])?;
        let inner_s = unsafe { crate::display::py_repr(s)? };
        return Ok(format!("array('u', {inner_s})"));
    }
    let mut parts = Vec::with_capacity(len);
    for i in 0..len {
        let w_item = unsafe { arr::w_array_unpack_item(obj, i) };
        parts.push(unsafe { crate::display::py_repr(w_item)? });
    }
    Ok(format!("array('{tc}', [{}])", parts.join(", ")))
}

fn array_repr_method(args: &[PyObjectRef]) -> PyResult {
    Ok(pyre_object::w_str_new(&array_repr_string(args[0])?))
}

// `interp_array.py compare_arrays`: compare each element with the requested
// operation.  In particular, an unordered pair such as NaN is neither less
// nor greater; treating every non-equal/non-less pair as greater is wrong.
fn array_richcompare(a: PyObjectRef, b: PyObjectRef, op: u8) -> PyResult {
    if !unsafe { arr::is_array(a) } || !unsafe { arr::is_array(b) } {
        return Ok(pyre_object::w_not_implemented());
    }
    let la = unsafe { arr::w_array_len(a) };
    let lb = unsafe { arr::w_array_len(b) };
    if op == 0 && la != lb {
        return Ok(pyre_object::w_bool_from(false));
    }
    if op == 1 && la != lb {
        return Ok(pyre_object::w_bool_from(true));
    }
    let n = la.min(lb);
    for i in 0..n {
        let ea = unsafe { arr::w_array_unpack_item(a, i) };
        let eb = unsafe { arr::w_array_unpack_item(b, i) };
        match op {
            0 => {
                if !crate::baseobjspace::is_true(compare(ea, eb, CompareOp::Eq)?)? {
                    return Ok(pyre_object::w_bool_from(false));
                }
            }
            1 => {
                if crate::baseobjspace::is_true(compare(ea, eb, CompareOp::Ne)?)? {
                    return Ok(pyre_object::w_bool_from(true));
                }
            }
            2 | 4 => {
                let cmp = if op == 2 {
                    CompareOp::Lt
                } else {
                    CompareOp::Gt
                };
                if crate::baseobjspace::is_true(compare(ea, eb, cmp)?)? {
                    return Ok(pyre_object::w_bool_from(true));
                }
                if !crate::baseobjspace::is_true(compare(ea, eb, CompareOp::Eq)?)? {
                    return Ok(pyre_object::w_bool_from(false));
                }
            }
            3 | 5 => {
                let cmp = if op == 3 {
                    CompareOp::Le
                } else {
                    CompareOp::Ge
                };
                if !crate::baseobjspace::is_true(compare(ea, eb, cmp)?)? {
                    return Ok(pyre_object::w_bool_from(false));
                }
                if !crate::baseobjspace::is_true(compare(ea, eb, CompareOp::Eq)?)? {
                    return Ok(pyre_object::w_bool_from(true));
                }
            }
            _ => unreachable!(),
        }
    }
    let result = match op {
        0 => true,
        1 => false,
        2 => la < lb,
        3 => la <= lb,
        4 => la > lb,
        5 => la >= lb,
        _ => unreachable!(),
    };
    Ok(pyre_object::w_bool_from(result))
}

fn array_eq_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__eq__")?;
    array_richcompare(args[0], args[1], 0)
}
fn array_ne_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__ne__")?;
    array_richcompare(args[0], args[1], 1)
}
fn array_lt_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__lt__")?;
    array_richcompare(args[0], args[1], 2)
}
fn array_le_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__le__")?;
    array_richcompare(args[0], args[1], 3)
}
fn array_gt_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__gt__")?;
    array_richcompare(args[0], args[1], 4)
}
fn array_ge_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__ge__")?;
    array_richcompare(args[0], args[1], 5)
}

// Arithmetic.
fn array_add_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__add__")?;
    let a = args[0];
    let b = args[1];
    if !unsafe { arr::is_array(b) } {
        return Err(PyError::type_error(
            "can only append array (not \"other\") to array",
        ));
    }
    let tc = unsafe { arr::w_array_typecode(a) };
    if unsafe { arr::w_array_typecode(b) } != tc {
        return Err(PyError::type_error(
            "bad argument type for built-in operation",
        ));
    }
    let isz = unsafe { arr::w_array_itemsize(a) } as u8;
    let mut out = unsafe { arr::w_array_bytes(a) }.to_vec();
    out.extend_from_slice(unsafe { arr::w_array_bytes(b) });
    Ok(arr::w_array_from_bytes(tc, isz, out))
}

fn array_iadd_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__iadd__")?;
    let a = args[0];
    array_check_resize(a)?;
    let b = args[1];
    if !unsafe { arr::is_array(b) }
        || unsafe { arr::w_array_typecode(b) } != unsafe { arr::w_array_typecode(a) }
    {
        return Err(PyError::type_error(
            "can only extend array with array of same kind",
        ));
    }
    let src = unsafe { arr::w_array_bytes(b) }.to_vec();
    let vec = unsafe { arr::w_array_vec_mut(a) };
    vec.extend_from_slice(&src);
    Ok(a)
}

fn array_repeat_bytes(obj: PyObjectRef, count: i64) -> PyResult {
    let tc = unsafe { arr::w_array_typecode(obj) };
    let isz = unsafe { arr::w_array_itemsize(obj) } as u8;
    let src = unsafe { arr::w_array_bytes(obj) };
    let n = count.max(0) as usize;
    // ovfcheck(oldlen * repeat) -> MemoryError on overflow (_mul_helper).
    let total = src
        .len()
        .checked_mul(n)
        .ok_or_else(|| PyError::memory_error(""))?;
    let mut out = Vec::with_capacity(total);
    for _ in 0..n {
        out.extend_from_slice(src);
    }
    Ok(arr::w_array_from_bytes(tc, isz, out))
}

fn array_mul_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__mul__")?;
    let count = crate::builtins::getindex_w(args[1])?;
    array_repeat_bytes(args[0], count)
}

fn array_imul_method(args: &[PyObjectRef]) -> PyResult {
    check_arity(args, 2, "array.__imul__")?;
    let obj = args[0];
    array_check_resize(obj)?;
    let count = crate::builtins::getindex_w(args[1])?.max(0) as usize;
    let src = unsafe { arr::w_array_bytes(obj) }.to_vec();
    if count == 0 {
        unsafe { arr::w_array_vec_mut(obj) }.clear();
        return Ok(obj);
    }
    // ovfcheck(oldlen * repeat) -> MemoryError on overflow.
    let extra = src
        .len()
        .checked_mul(count - 1)
        .ok_or_else(|| PyError::memory_error(""))?;
    let vec = unsafe { arr::w_array_vec_mut(obj) };
    vec.reserve(extra);
    for _ in 1..count {
        vec.extend_from_slice(&src);
    }
    Ok(obj)
}

// ──────────────────────────────────────────────────────────────────────
// Pickle: __reduce_ex__ + _array_reconstructor.
// ──────────────────────────────────────────────────────────────────────

/// `array.__reduce_ex__(protocol)` — list-based reduce
/// `(type(self), (typecode, tolist()), __dict__)` (`interp_array.py
/// descr_reduce_ex`; the list form is accepted for every protocol).
fn array_reduce_ex_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let w_type = crate::typedef::r#type(obj).unwrap_or(PY_NULL);
    let tc = unsafe { arr::w_array_typecode(obj) } as char;
    let w_typecode = pyre_object::w_str_new(&tc.to_string());
    let w_items = array_tolist_method(&[obj])?;
    let ctor_args = pyre_object::w_tuple_new(vec![w_typecode, w_items]);
    Ok(pyre_object::w_tuple_new(vec![
        w_type,
        ctor_args,
        pyre_object::w_none(),
    ]))
}

/// `_array_reconstructor(cls, typecode, mformat_code, items)` — pyre stores
/// native machine-format bytes, so this rebuilds via `frombytes`
/// (`reconstructor.py array_reconstructor`).  The slow per-format decode is
/// not needed because pyre never produces the bytes-based reduce form.
fn array_reconstructor(args: &[PyObjectRef]) -> PyResult {
    if args.len() < 4 {
        return Err(PyError::type_error(
            "_array_reconstructor() takes exactly 4 arguments",
        ));
    }
    let w_cls = args[0];
    if !unsafe { pyre_object::is_type(w_cls) } {
        return Err(PyError::type_error(
            "_array_reconstructor() argument 1 must be type, not other",
        ));
    }
    // mformat_code: int in [MACHINE_FORMAT_CODE_MIN, MACHINE_FORMAT_CODE_MAX].
    if !unsafe { pyre_object::is_int(args[2]) } {
        return Err(PyError::type_error(
            "an integer is required (got type other)",
        ));
    }
    let mformat = unsafe { pyre_object::w_int_get_value(args[2]) };
    if !(0..=21).contains(&mformat) {
        return Err(PyError::value_error(
            "third argument must be a valid machine format code.",
        ));
    }
    if !unsafe { pyre_object::bytesobject::is_bytes_like(args[3]) } {
        return Err(PyError::type_error(
            "fourth argument should be bytes, not other",
        ));
    }
    // pyre stores native machine-format bytes, so the native fast-path
    // (mformat == native) is a direct frombytes; array_descr_new validates
    // the typecode (ValueError) and retags any array subclass.
    let new_args = [w_cls, args[1]];
    let obj = array_descr_new(&new_args)?;
    let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(args[3]) }.to_vec();
    array_frombytes(obj, &bytes)?;
    Ok(obj)
}

// ──────────────────────────────────────────────────────────────────────
// Type / module registration.
// ──────────────────────────────────────────────────────────────────────

/// Register all `array.array` methods/getsets into the type namespace.
pub fn init_array_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            crate::typedef::make_new_descr(array_descr_new),
        )
    };
    let m = |ns: PyObjectRef, name: &'static str, f: fn(&[PyObjectRef]) -> PyResult, arity: u16| {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, f, arity),
            )
        };
    };
    m(ns, "__len__", array_len, 1);
    m(ns, "__iter__", array_iter, 1);
    m(ns, "__getitem__", array_getitem, 2);
    m(ns, "__setitem__", array_setitem, 3);
    m(ns, "__delitem__", array_delitem, 2);
    m(ns, "__contains__", array_contains_method, 2);
    m(ns, "__repr__", array_repr_method, 1);
    m(ns, "__str__", array_repr_method, 1);
    m(ns, "__eq__", array_eq_method, 2);
    m(ns, "__ne__", array_ne_method, 2);
    m(ns, "__lt__", array_lt_method, 2);
    m(ns, "__le__", array_le_method, 2);
    m(ns, "__gt__", array_gt_method, 2);
    m(ns, "__ge__", array_ge_method, 2);
    m(ns, "__add__", array_add_method, 2);
    m(ns, "__iadd__", array_iadd_method, 2);
    m(ns, "__mul__", array_mul_method, 2);
    m(ns, "__rmul__", array_mul_method, 2);
    m(ns, "__imul__", array_imul_method, 2);
    m(ns, "__reduce_ex__", array_reduce_ex_method, 2);
    m(ns, "append", array_append_method, 2);
    m(ns, "extend", array_extend_method, 2);
    m(ns, "insert", array_insert_method, 3);
    m(ns, "remove", array_remove_method, 2);
    // `index` accepts optional start/stop.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "index",
            crate::make_builtin_function("index", array_index_method),
        )
    };
    m(ns, "count", array_count_method, 2);
    m(ns, "clear", array_clear_method, 1);
    m(ns, "__release_buffer__", array_release_buffer, 2);
    m(ns, "reverse", array_reverse_method, 1);
    m(ns, "tolist", array_tolist_method, 1);
    m(ns, "fromlist", array_fromlist_method, 2);
    m(ns, "tobytes", array_tobytes_method, 1);
    m(ns, "frombytes", array_frombytes_method, 2);
    m(ns, "tounicode", array_tounicode_method, 1);
    m(ns, "fromunicode", array_fromunicode_method, 2);
    m(ns, "buffer_info", array_buffer_info_method, 1);
    m(ns, "byteswap", array_byteswap_method, 1);
    m(ns, "__copy__", array_copy_method, 1);
    m(ns, "__deepcopy__", array_copy_method, 2);
    // `pop` accepts an optional index.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "pop",
            crate::make_builtin_function("pop", array_pop_method),
        )
    };
    // typecode / itemsize read-only properties.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "typecode",
            pyre_object::w_property_new(
                make_builtin_function_with_arity(
                    "typecode",
                    |args| {
                        let tc = unsafe { arr::w_array_typecode(args[0]) } as char;
                        Ok(pyre_object::w_str_new(&tc.to_string()))
                    },
                    1,
                ),
                PY_NULL,
                PY_NULL,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "itemsize",
            pyre_object::w_property_new(
                make_builtin_function_with_arity(
                    "itemsize",
                    |args| {
                        Ok(pyre_object::w_int_new(
                            unsafe { arr::w_array_itemsize(args[0]) } as i64,
                        ))
                    },
                    1,
                ),
                PY_NULL,
                PY_NULL,
            ),
        )
    };
}

/// `array` module init — `moduledef.py interpleveldefs`.
pub fn init_array_module(ns: pyre_object::PyObjectRef) {
    let type_obj = crate::typedef::gettypeobject(&pyre_object::interp_array::ARRAY_TYPE);
    module_ns_store(ns, "array", type_obj);
    module_ns_store(ns, "ArrayType", type_obj);
    module_ns_store(ns, "typecodes", pyre_object::w_str_new(arr::TYPECODES));
    module_ns_store(
        ns,
        "_array_reconstructor",
        crate::make_builtin_function("_array_reconstructor", array_reconstructor),
    );
}

/// PyPy `pypy/module/array/moduledef.py:Module.startup`: array is a
/// virtual `MutableSequence`, registered only after the builtin module is in
/// `sys.modules` so importing `_collections_abc` cannot create a second array
/// module during a cycle.
pub fn startup_array_module(
    _module: pyre_object::PyObjectRef,
    execution_context: *const crate::PyExecutionContext,
) -> Result<(), crate::PyError> {
    let abc_module = crate::importing::importhook(
        "_collections_abc",
        pyre_object::PY_NULL,
        pyre_object::w_tuple_new(vec![pyre_object::w_str_new("MutableSequence")]),
        0,
        execution_context,
    )?;
    let mutable_sequence = crate::baseobjspace::getattr_str(abc_module, "MutableSequence")?;
    let register = crate::baseobjspace::getattr_str(mutable_sequence, "register")?;
    let array_type = crate::typedef::gettypeobject(&pyre_object::interp_array::ARRAY_TYPE);
    crate::call::call_function_impl_result(register, &[array_type])?;
    Ok(())
}
