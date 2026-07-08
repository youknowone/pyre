use malachite_bigint::BigInt;
use num_traits::ToPrimitive;

use crate::executioncontext::DictStorage;
use crate::{
    make_builtin_function, make_builtin_function_with_arity, make_module_builtin_function,
    make_module_builtin_function_with_arity,
};
use pyre_object::*;
use rustpython_wtf8::{CodePoint, Wtf8Buf};

/// `buffer_w` — select the byte-storage `Buffer` variant for a memoryview
/// backing by concrete kind, so a bytes / bytearray / array *subclass* backing
/// is tagged for its own fields (`bytes_like_data` exact-branches on the type
/// and would mis-read a subclass).  Construction lives here, not in
/// pyre-object's `Buffer`, because the subclass fallback needs `isinstance_w`.
unsafe fn memoryview_backing_buffer(backing: PyObjectRef) -> pyre_object::buffer::Buffer {
    use pyre_object::buffer::Buffer;
    unsafe {
        if pyre_object::interp_array::is_array(backing) {
            Buffer::Array { w_obj: backing }
        } else if pyre_object::bytearrayobject::is_bytearray(backing) {
            Buffer::Byte { w_obj: backing }
        } else if pyre_object::bytesobject::is_bytes(backing) {
            Buffer::String { w_obj: backing }
        } else if crate::baseobjspace::isinstance_w(
            backing,
            crate::typedef::gettypeobject(&pyre_object::interp_array::ARRAY_TYPE),
        ) {
            Buffer::Array { w_obj: backing }
        } else if crate::baseobjspace::isinstance_w(
            backing,
            crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE),
        ) {
            Buffer::Byte { w_obj: backing }
        } else {
            Buffer::String { w_obj: backing }
        }
    }
}

/// The full byte storage of a memoryview backing, read through its tagged
/// `Buffer` variant.
unsafe fn memoryview_backing_slice(backing: PyObjectRef) -> &'static [u8] {
    unsafe { memoryview_backing_buffer(backing).as_bytes() }
}

/// Mutable raw byte storage of a writable memoryview backing — `bytearray`
/// or `array.array`, the two mutable buffer exporters.  `None` for a
/// read-only exporter (`bytes`) or one without in-place byte storage, so a
/// write assignment reports "cannot modify read-only memory".
unsafe fn memoryview_backing_bytes_mut(backing: PyObjectRef) -> Option<&'static mut [u8]> {
    unsafe {
        let bytearray_ty =
            crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE);
        if pyre_object::bytearrayobject::is_bytearray(backing)
            || crate::baseobjspace::isinstance_w(backing, bytearray_ty)
        {
            return Some(pyre_object::bytearrayobject::w_bytearray_data_mut(backing));
        }
        let array_ty = crate::typedef::gettypeobject(&pyre_object::interp_array::ARRAY_TYPE);
        if pyre_object::interp_array::is_array(backing)
            || crate::baseobjspace::isinstance_w(backing, array_ty)
        {
            return Some(pyre_object::interp_array::w_array_vec_mut(backing).as_mut_slice());
        }
        None
    }
}

/// Read element `i` of a memoryview shape/strides tuple as an `i64`.
unsafe fn memoryview_dim_value(tuple: PyObjectRef, i: i64) -> i64 {
    unsafe {
        pyre_object::tupleobject::w_tuple_getitem(tuple, i)
            .map(|w| pyre_object::w_int_get_value(w))
            .unwrap_or(0)
    }
}

/// Allocate a `W_MemoryView` over a freshly built off-heap `BufferView`.
/// Selects the backing `Buffer` variant (needs `isinstance_w`, hence here and
/// not in pyre-object) and pins every ref across the allocation — building the
/// box and allocating the object can trigger a collection before the new
/// memoryview roots them.
#[allow(clippy::too_many_arguments)]
unsafe fn w_memoryview_alloc(
    w_obj: PyObjectRef,
    w_backing: PyObjectRef,
    w_format: PyObjectRef,
    w_shape: PyObjectRef,
    w_strides: PyObjectRef,
    itemsize: i64,
    ndim: i64,
    offset: i64,
    length: i64,
    readonly: bool,
    released: bool,
) -> PyObjectRef {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(w_obj);
        pyre_object::gc_roots::pin_root(w_backing);
        pyre_object::gc_roots::pin_root(w_format);
        pyre_object::gc_roots::pin_root(w_shape);
        pyre_object::gc_roots::pin_root(w_strides);
        // Allocate the managed header first; old-gen allocation may trigger a
        // moving collection, so read the relocated refs back from the shadow
        // stack before building the off-heap view (mirrors `W_TupleObject`
        // filling its items block from the `pop_roots`-relocated slots).
        let mv = pyre_object::memoryview::w_memoryview_alloc_header(released);
        let r_obj = pyre_object::gc_roots::shadow_stack_get(sp);
        let r_backing = pyre_object::gc_roots::shadow_stack_get(sp + 1);
        let r_format = pyre_object::gc_roots::shadow_stack_get(sp + 2);
        let r_shape = pyre_object::gc_roots::shadow_stack_get(sp + 3);
        let r_strides = pyre_object::gc_roots::shadow_stack_get(sp + 4);
        let view = pyre_object::bufferview::BufferView::Strided {
            backing: memoryview_backing_buffer(r_backing),
            w_obj: r_obj,
            w_format: r_format,
            w_shape: r_shape,
            w_strides: r_strides,
            itemsize,
            ndim,
            offset,
            length,
            readonly,
        };
        let view_ptr = pyre_object::memoryview::bufferview_alloc(view);
        pyre_object::memoryview::w_memoryview_set_view(mv, view_ptr);
        mv
    }
}

/// The LIVE logical bytes of a view, honouring `offset`/strides/shape so a
/// strided slice (`m[::2]`, `m[::-1]`) or an N-D view gathers the right
/// elements in C order (`buffer.py as_str`).  Reads the backing object's own
/// storage — no detached copy — so the view observes later mutation of a
/// bytearray / array source.
///
/// # Safety
/// `mv` must point to a valid `W_MemoryView` with a live backing.
pub(crate) unsafe fn memoryview_gather_bytes(mv: PyObjectRef) -> Vec<u8> {
    unsafe { pyre_object::memoryview::w_memoryview_view(mv).gather() }
}

/// Buffer-acquisition parameters `(format, itemsize, readonly, total_bytes)`
/// for a bytes / bytearray / array exporter (or a subclass of one), or
/// `None` when `obj` provides no buffer.
unsafe fn memoryview_buffer_params(obj: PyObjectRef) -> Option<(String, i64, bool, usize)> {
    unsafe {
        let array_ty = crate::typedef::gettypeobject(&pyre_object::interp_array::ARRAY_TYPE);
        if pyre_object::interp_array::is_array(obj)
            || crate::baseobjspace::isinstance_w(obj, array_ty)
        {
            let tc = pyre_object::interp_array::w_array_typecode(obj);
            let isz = pyre_object::interp_array::w_array_itemsize(obj);
            let fmt = String::from_utf8_lossy(&[tc]).into_owned();
            let nbytes = pyre_object::interp_array::w_array_bytes(obj).len();
            return Some((fmt, isz as i64, false, nbytes));
        }
        let bytearray_ty =
            crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE);
        if pyre_object::bytearrayobject::is_bytearray(obj)
            || crate::baseobjspace::isinstance_w(obj, bytearray_ty)
        {
            return Some((
                "B".to_owned(),
                1,
                false,
                pyre_object::bytearrayobject::w_bytearray_len(obj),
            ));
        }
        let bytes_ty = crate::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE);
        if pyre_object::bytesobject::is_bytes(obj)
            || crate::baseobjspace::isinstance_w(obj, bytes_ty)
        {
            return Some((
                "B".to_owned(),
                1,
                true,
                pyre_object::bytesobject::w_bytes_len(obj),
            ));
        }
        None
    }
}

/// `memoryview(obj)` — acquire a 1-D byte view over a buffer-providing
/// exporter.  Sharing another memoryview copies its view parameters (and
/// reports the original exporter as `.obj`); a non-buffer raises TypeError.
pub(crate) fn w_memoryview_new(w_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::memoryview::*;
    if let Some(target) = crate::module::__pypy__::interp_buffer::forwarded_exporter(w_obj) {
        return w_memoryview_new(target?);
    }
    unsafe {
        if is_w_memoryview(w_obj) {
            memoryview_check_released(w_obj)?;
            let backing = w_memoryview_backing(w_obj);
            return Ok(w_memoryview_alloc(
                w_memoryview_obj(w_obj),
                backing,
                w_memoryview_format(w_obj),
                w_memoryview_shape(w_obj),
                w_memoryview_strides(w_obj),
                w_memoryview_itemsize(w_obj),
                w_memoryview_ndim(w_obj),
                w_memoryview_offset(w_obj),
                w_memoryview_length(w_obj),
                w_memoryview_readonly(w_obj),
                false,
            ));
        }
        let (fmt, itemsize, readonly, byte_len) = match memoryview_buffer_params(w_obj) {
            Some(p) => p,
            None => {
                let tname = crate::typedef::r#type(w_obj)
                    .map(|t| pyre_object::w_type_get_name(t))
                    .unwrap_or("object");
                return Err(crate::PyError::type_error(&format!(
                    "memoryview: a bytes-like object is required, not '{tname}'"
                )));
            }
        };
        let count = if itemsize > 0 {
            byte_len as i64 / itemsize
        } else {
            0
        };
        let shape = pyre_object::w_tuple_new(vec![w_int_new(count)]);
        let strides = pyre_object::w_tuple_new(vec![w_int_new(itemsize)]);
        Ok(w_memoryview_alloc(
            w_obj,
            w_obj,
            w_str_new(&fmt),
            shape,
            strides,
            itemsize,
            1,
            0,
            byte_len as i64,
            readonly,
            false,
        ))
    }
}

/// `_check_released` — every accessing method rejects a released view with
/// `ValueError` before touching the (logically dropped) backing.
pub(crate) unsafe fn memoryview_check_released(mv: PyObjectRef) -> Result<(), crate::PyError> {
    if unsafe { pyre_object::memoryview::w_memoryview_released(mv) } {
        return Err(crate::PyError::value_error(
            "operation forbidden on released memoryview object",
        ));
    }
    Ok(())
}

/// Raw logical bytes of a memoryview, or `None` when `obj` is not one.
/// `bytes(memoryview)` / `bytearray(memoryview)` copy the view per the
/// buffer protocol rather than iterating element values.
pub(crate) unsafe fn memoryview_as_bytes(obj: PyObjectRef) -> Option<Vec<u8>> {
    unsafe { pyre_object::memoryview::is_w_memoryview(obj).then(|| memoryview_gather_bytes(obj)) }
}

/// Little-endian unsigned unpack of one `itemsize`-wide element at byte
/// offset `base` — the fallback for formats the shared decoder rejects.
fn memoryview_unpack(data: &[u8], itemsize: usize, base: usize) -> i64 {
    let mut val: i64 = 0;
    for j in 0..itemsize {
        val |= (data[base + j] as i64) << (8 * j);
    }
    val
}

/// The native element typecode of a buffer/struct format string, with an
/// optional leading byte-order modifier (`@=<>!`) stripped.  memoryview
/// formats are native single characters (`@x` or `x`); an empty string
/// falls back to unsigned bytes.
fn memoryview_format_code(fmt: &str) -> u8 {
    let b = fmt.as_bytes();
    match b.first() {
        Some(b'@' | b'=' | b'<' | b'>' | b'!') => b.get(1).copied().unwrap_or(b'B'),
        Some(&c) => c,
        None => b'B',
    }
}

/// Box one `itemsize`-wide element at byte offset `base` per the view's
/// format (`buffer.py value_from_bytes`).  Numeric typecodes route through
/// the shared array decoder (`unpack_value`); `c` yields a length-1 bytes,
/// `?` a bool, and any code the decoder rejects falls back to unsigned LE.
unsafe fn memoryview_unpack_element(
    fmt: &str,
    data: &[u8],
    base: usize,
    itemsize: usize,
) -> PyObjectRef {
    let buf = &data[base..base + itemsize];
    match memoryview_format_code(fmt) {
        b'c' => pyre_object::bytesobject::w_bytes_from_bytes(buf),
        b'?' => w_bool_from(buf.iter().any(|&x| x != 0)),
        tc => {
            let w = pyre_object::interp_array::unpack_value(tc, buf);
            if w == pyre_object::PY_NULL {
                w_int_new(memoryview_unpack(data, itemsize, base))
            } else {
                w
            }
        }
    }
}

/// Pack `w_val` into `itemsize` native-order bytes per `fmt`
/// (`buffer.py bytes_from_value`).  Both a wrong operand type and an
/// out-of-range value surface the `StructError` the packer raises as the
/// TypeError "memoryview: invalid type for format '%s'".
fn memoryview_pack_value(
    fmt: &str,
    itemsize: usize,
    w_val: PyObjectRef,
) -> Result<Vec<u8>, crate::PyError> {
    let bad_type =
        || crate::PyError::type_error(format!("memoryview: invalid type for format '{fmt}'"));
    let range = |v: i64, lo: i64, hi: i64| -> Result<(), crate::PyError> {
        if (lo..=hi).contains(&v) {
            Ok(())
        } else {
            Err(bad_type())
        }
    };
    // Integer formats coerce via `__index__` (`pack_single`/`PyNumber_Index`),
    // not an exact-int check; a value with no `__index__` is the format error.
    let as_index = || -> Result<PyObjectRef, crate::PyError> {
        if unsafe { pyre_object::is_int_or_long(w_val) } {
            Ok(w_val)
        } else {
            unsafe { crate::baseobjspace::space_index(w_val) }.map_err(|_| bad_type())
        }
    };
    let int_val = || -> Result<i64, crate::PyError> {
        crate::baseobjspace::int_w(as_index()?).map_err(|_| bad_type())
    };
    let bytes = match memoryview_format_code(fmt) {
        b'b' => {
            let v = int_val()?;
            range(v, i8::MIN as i64, i8::MAX as i64)?;
            (v as i8).to_ne_bytes().to_vec()
        }
        b'B' => {
            let v = int_val()?;
            range(v, 0, u8::MAX as i64)?;
            (v as u8).to_ne_bytes().to_vec()
        }
        b'h' => {
            let v = int_val()?;
            range(v, i16::MIN as i64, i16::MAX as i64)?;
            (v as i16).to_ne_bytes().to_vec()
        }
        b'H' => {
            let v = int_val()?;
            range(v, 0, u16::MAX as i64)?;
            (v as u16).to_ne_bytes().to_vec()
        }
        b'i' | b'l' if itemsize == 4 => {
            let v = int_val()?;
            range(v, i32::MIN as i64, i32::MAX as i64)?;
            (v as i32).to_ne_bytes().to_vec()
        }
        b'I' | b'L' if itemsize == 4 => {
            let v = int_val()?;
            range(v, 0, u32::MAX as i64)?;
            (v as u32).to_ne_bytes().to_vec()
        }
        b'l' | b'q' | b'n' => {
            let v = int_val()?;
            v.to_ne_bytes().to_vec()
        }
        b'L' | b'Q' | b'N' | b'P' => {
            let v = crate::baseobjspace::uint_w(as_index()?).map_err(|_| bad_type())?;
            v.to_ne_bytes().to_vec()
        }
        b'f' => {
            if !unsafe { pyre_object::is_int_or_long(w_val) || pyre_object::is_float(w_val) } {
                return Err(bad_type());
            }
            let v = crate::baseobjspace::float_w(w_val).map_err(|_| bad_type())? as f32;
            v.to_ne_bytes().to_vec()
        }
        b'd' => {
            if !unsafe { pyre_object::is_int_or_long(w_val) || pyre_object::is_float(w_val) } {
                return Err(bad_type());
            }
            let v = crate::baseobjspace::float_w(w_val).map_err(|_| bad_type())?;
            v.to_ne_bytes().to_vec()
        }
        b'?' => {
            vec![crate::baseobjspace::is_true(w_val)? as u8]
        }
        b'c' => {
            if unsafe { pyre_object::bytesobject::is_bytes(w_val) } {
                let d = unsafe { pyre_object::bytesobject::w_bytes_data(w_val) };
                if d.len() == 1 {
                    return Ok(d.to_vec());
                }
            }
            return Err(bad_type());
        }
        _ => return Err(bad_type()),
    };
    Ok(bytes)
}

/// Element-value list of a 1-D view (format-aware per `value_from_bytes`).
unsafe fn memoryview_values(mv: PyObjectRef) -> Vec<PyObjectRef> {
    unsafe {
        let itemsize = pyre_object::memoryview::w_memoryview_itemsize(mv) as usize;
        let fmt = pyre_object::w_str_get_value(pyre_object::memoryview::w_memoryview_format(mv));
        let data = memoryview_gather_bytes(mv);
        let mut items = Vec::new();
        let mut base = 0;
        while itemsize > 0 && base + itemsize <= data.len() {
            items.push(memoryview_unpack_element(fmt, &data, base, itemsize));
            base += itemsize;
        }
        items
    }
}

/// A live strided sub-view `m[start:stop:step]`, sharing the same backing.
/// Slicing operates on dimension 0 (`buffer.py BufferSlice`): the element
/// count is `shape[0]`, the step rides `strides[0]`, and dimensions `1..ndim`
/// are preserved, so slicing an N-D view keeps its dimensionality.  The byte
/// `offset += start*strides[0]` and `strides[0] *= step` make the result read
/// through to the original storage.
unsafe fn memoryview_slice_view(
    mv: PyObjectRef,
    index: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::memoryview::*;
    unsafe {
        let itemsize = w_memoryview_itemsize(mv);
        let ndim = w_memoryview_ndim(mv);
        let parent_shape = w_memoryview_shape(mv);
        let parent_strides = w_memoryview_strides(mv);
        let count = if ndim >= 1 {
            memoryview_dim_value(parent_shape, 0)
        } else {
            0
        };
        let stride_p = if ndim >= 1 {
            memoryview_dim_value(parent_strides, 0)
        } else {
            itemsize
        };
        let offset_p = w_memoryview_offset(mv);
        let (start, stop, step) = crate::baseobjspace::normalize_slice(index, count)?;
        let slicelength = if step > 0 {
            if stop > start {
                (stop - start + step - 1) / step
            } else {
                0
            }
        } else if start > stop {
            (start - stop - step - 1) / -step
        } else {
            0
        };
        let new_offset = offset_p + start * stride_p;
        let new_stride = stride_p * step;
        // Dimension 0 takes the slice's shape/stride; later dimensions ride
        // along unchanged.
        let mut shape_v = vec![slicelength];
        let mut strides_v = vec![new_stride];
        for d in 1..ndim {
            shape_v.push(memoryview_dim_value(parent_shape, d));
            strides_v.push(memoryview_dim_value(parent_strides, d));
        }
        let new_length = shape_v.iter().product::<i64>() * itemsize;
        let shape = pyre_object::w_tuple_new(shape_v.iter().map(|&x| w_int_new(x)).collect());
        let strides = pyre_object::w_tuple_new(strides_v.iter().map(|&x| w_int_new(x)).collect());
        Ok(w_memoryview_alloc(
            w_memoryview_obj(mv),
            w_memoryview_backing(mv),
            w_memoryview_format(mv),
            shape,
            strides,
            itemsize,
            ndim.max(1),
            new_offset,
            new_length,
            w_memoryview_readonly(mv),
            false,
        ))
    }
}

/// `is_byte_format` — `b`/`B`/`c`, the single-byte element formats that
/// `cast` may freely convert to or from.
fn memoryview_is_byte_format(fmt: &str) -> bool {
    matches!(memoryview_format_code(fmt), b'b' | b'B' | b'c')
}

/// `get_native_fmtchar` — the native byte width of a single-character
/// format (`x` or `@x`), or `None` for an unrecognised / non-native one.
fn memoryview_native_fmtchar(fmt: &str) -> Option<i64> {
    let b = fmt.as_bytes();
    let f = match b.first()? {
        b'@' if b.len() == 2 => b[1],
        _ if b.len() == 1 => b[0],
        _ => return None,
    };
    Some(match f {
        b'c' | b'b' | b'B' | b'?' => 1,
        b'h' | b'H' => 2,
        b'i' | b'I' | b'f' => 4,
        b'l' | b'L' | b'q' | b'Q' | b'n' | b'N' | b'd' | b'P' => 8,
        _ => return None,
    })
}

/// `_strides_from_shape` — C-contiguous strides for `shape`: the last
/// dimension steps by `itemsize`, each earlier one by the product of the
/// faster dimensions.
fn memoryview_strides_from_shape(shape: &[i64], itemsize: i64) -> Vec<i64> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut s = vec![0i64; ndim];
    s[ndim - 1] = itemsize;
    for i in (0..ndim - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// `get_offset` — the byte offset of `index` along dimension `dim`,
/// bounds-checked against `shape[dim]` (negative indices wrap).
unsafe fn memoryview_get_offset(
    mv: PyObjectRef,
    dim: i64,
    index: i64,
) -> Result<i64, crate::PyError> {
    use pyre_object::memoryview::*;
    unsafe {
        let read = |t: PyObjectRef, i: i64| {
            pyre_object::tupleobject::w_tuple_getitem(t, i)
                .map(|w| pyre_object::w_int_get_value(w))
                .unwrap_or(0)
        };
        let nitems = read(w_memoryview_shape(mv), dim);
        let mut idx = index;
        if idx < 0 {
            idx += nitems;
        }
        if idx < 0 || idx >= nitems {
            return Err(crate::PyError::index_error(format!(
                "index out of bounds on dimension {}",
                dim + 1
            )));
        }
        Ok(read(w_memoryview_strides(mv), dim) * idx)
    }
}

/// An index key — `getindex_w` accepts any object with `__index__`, not only
/// an exact int, so a scalar key or a multi-index tuple element counts as an
/// index when it is an int or exposes `__index__`.
unsafe fn memoryview_is_index(w: PyObjectRef) -> bool {
    unsafe { pyre_object::is_int(w) || crate::baseobjspace::lookup(w, "__index__").is_some() }
}

/// `_start_from_tuple` — the summed byte offset of a multi-index tuple
/// (one integer per dimension).
unsafe fn memoryview_start_from_tuple(
    mv: PyObjectRef,
    index: PyObjectRef,
) -> Result<i64, crate::PyError> {
    unsafe {
        let n = pyre_object::w_tuple_len(index) as i64;
        let mut start = 0;
        for dim in 0..n {
            let w = pyre_object::w_tuple_getitem(index, dim).unwrap_or(w_none());
            if !memoryview_is_index(w) {
                return Err(crate::PyError::type_error("memoryview: invalid slice key"));
            }
            start += memoryview_get_offset(mv, dim, getindex_w(w)?)?;
        }
        Ok(start)
    }
}

/// Classify a tuple key: all-integer is a multi-index element access,
/// all-slice (non-empty) is multi-dimensional slicing.
unsafe fn memoryview_tuple_kind(index: PyObjectRef) -> (bool, bool) {
    unsafe {
        let n = pyre_object::w_tuple_len(index);
        let mut all_index = true;
        let mut all_slice = n > 0;
        for i in 0..n {
            let w = pyre_object::w_tuple_getitem(index, i as i64).unwrap_or(w_none());
            if !memoryview_is_index(w) {
                all_index = false;
            }
            if !pyre_object::is_slice(w) {
                all_slice = false;
            }
        }
        (all_index, all_slice)
    }
}

/// `memoryview.__getitem__` — an integer index unpacks the element at its
/// strided byte address; a slice returns a live sub-view; a multi-index
/// tuple reads an element of an N-D view.
fn memoryview_getitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    let index = args.get(1).copied().unwrap_or(w_none());
    unsafe {
        use pyre_object::memoryview::*;
        memoryview_check_released(mv)?;
        let ndim = w_memoryview_ndim(mv);
        if pyre_object::is_slice(index) {
            return memoryview_slice_view(mv, index);
        }
        if memoryview_is_index(index) {
            if ndim == 0 {
                return Err(crate::PyError::type_error(
                    "invalid indexing of 0-dim memory",
                ));
            }
            if ndim != 1 {
                return Err(crate::PyError::not_implemented(
                    "multi-dimensional sub-views are not implemented",
                ));
            }
            let itemsize = w_memoryview_itemsize(mv);
            let length = w_memoryview_length(mv);
            let count = if itemsize > 0 { length / itemsize } else { 0 };
            let mut i = getindex_w(index)?;
            if i < 0 {
                i += count;
            }
            if i < 0 || i >= count {
                return Err(crate::PyError::index_error("index out of bounds"));
            }
            let base = (w_memoryview_offset(mv) + i * w_memoryview_stride0(mv)) as usize;
            let full = memoryview_backing_slice(w_memoryview_backing(mv));
            let fmt = pyre_object::w_str_get_value(w_memoryview_format(mv));
            return Ok(memoryview_unpack_element(
                fmt,
                full,
                base,
                itemsize as usize,
            ));
        }
        if pyre_object::is_tuple(index) {
            let (all_index, all_slice) = memoryview_tuple_kind(index);
            if all_index {
                let length = pyre_object::w_tuple_len(index) as i64;
                if length < ndim {
                    return Err(crate::PyError::not_implemented(
                        "sub-views are not implemented",
                    ));
                }
                if length > ndim {
                    return Err(crate::PyError::type_error(format!(
                        "cannot index {length}-dimension view with {ndim}-element tuple"
                    )));
                }
                let start = memoryview_start_from_tuple(mv, index)?;
                let itemsize = w_memoryview_itemsize(mv);
                let base = (w_memoryview_offset(mv) + start) as usize;
                let full = memoryview_backing_slice(w_memoryview_backing(mv));
                let fmt = pyre_object::w_str_get_value(w_memoryview_format(mv));
                return Ok(memoryview_unpack_element(
                    fmt,
                    full,
                    base,
                    itemsize as usize,
                ));
            }
            if all_slice {
                return Err(crate::PyError::not_implemented(
                    "multi-dimensional slicing is not implemented",
                ));
            }
            return Err(crate::PyError::type_error("memoryview: invalid slice key"));
        }
        Err(crate::PyError::type_error(
            "memoryview: invalid slice key, must be int or slice",
        ))
    }
}

/// `memoryview.__setitem__` — write through to a mutable bytearray-backed
/// view, packing the value per the view's format (`memoryobject.py
/// descr_setitem`).  An integer index writes one element; a slice writes a
/// same-length bytes-like / memoryview rvalue element-by-element.  Read-only
/// views raise TypeError.
fn memoryview_setitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    let index = args.get(1).copied().unwrap_or(w_none());
    let value = args.get(2).copied().unwrap_or(w_none());
    unsafe {
        use pyre_object::memoryview::*;
        memoryview_check_released(mv)?;
        if w_memoryview_readonly(mv) {
            return Err(crate::PyError::type_error("cannot modify read-only memory"));
        }
        let backing = w_memoryview_backing(mv);
        if memoryview_backing_bytes_mut(backing).is_none() {
            return Err(crate::PyError::type_error("cannot modify read-only memory"));
        }
        let itemsize = w_memoryview_itemsize(mv);
        let isz = itemsize.max(0) as usize;
        let fmt = pyre_object::w_str_get_value(w_memoryview_format(mv)).to_owned();
        let count = if itemsize > 0 {
            w_memoryview_length(mv) / itemsize
        } else {
            0
        };
        let stride0 = w_memoryview_stride0(mv);
        let offset = w_memoryview_offset(mv);
        // Slice assignment writes the rvalue's element bytes through to the
        // strided positions of the view (`_setitem_slice`).
        if pyre_object::is_slice(index) {
            if w_memoryview_ndim(mv) != 1 {
                return Err(crate::PyError::not_implemented(
                    "memoryview slice assignments are currently restricted to ndim = 1",
                ));
            }
            let (start, stop, step) = crate::baseobjspace::normalize_slice(index, count)?;
            let mut indices = Vec::new();
            let mut i = start;
            while (step > 0 && i < stop) || (step < 0 && i > stop) {
                indices.push(i);
                i += step;
            }
            let src: Vec<u8> = match crate::typedef::buffer_as_bytes_like(value)? {
                Some(b) => pyre_object::bytesobject::bytes_like_data(b).to_vec(),
                None => {
                    return Err(crate::PyError::type_error(
                        "memoryview: a bytes-like object is required",
                    ));
                }
            };
            if isz == 0 || src.len() != indices.len() * isz {
                return Err(crate::PyError::value_error(
                    "cannot modify size of memoryview object",
                ));
            }
            let full =
                memoryview_backing_bytes_mut(backing).expect("writable backing checked above");
            for (k, &idx) in indices.iter().enumerate() {
                let dst = (offset + idx * stride0) as usize;
                full[dst..dst + isz].copy_from_slice(&src[k * isz..k * isz + isz]);
            }
            return Ok(w_none());
        }
        // Multi-index tuple writes one element of an N-D view; an all-slice
        // tuple is multi-dimensional slice assignment (`_setitem_tuple_indexed`).
        if pyre_object::is_tuple(index) {
            let ndim = w_memoryview_ndim(mv);
            let (all_index, all_slice) = memoryview_tuple_kind(index);
            if all_slice {
                return Err(crate::PyError::not_implemented(
                    "multi-dimensional slicing is not implemented",
                ));
            }
            if !all_index {
                return Err(crate::PyError::type_error("memoryview: invalid slice key"));
            }
            let length = pyre_object::w_tuple_len(index) as i64;
            if length < ndim {
                return Err(crate::PyError::not_implemented(
                    "sub-views are not implemented",
                ));
            }
            if length > ndim {
                return Err(crate::PyError::type_error(format!(
                    "cannot index {length}-dimension view with {ndim}-element tuple"
                )));
            }
            let packed = memoryview_pack_value(&fmt, isz, value)?;
            // memory_ass_sub: pack the value, then re-check release before the
            // write — the value's `__index__`/`__float__` coercion may have
            // released the view (`bytes_from_value` → `_check_released` →
            // `setbytes`).
            memoryview_check_released(mv)?;
            let start = memoryview_start_from_tuple(mv, index)?;
            let addr = (offset + start) as usize;
            let full =
                memoryview_backing_bytes_mut(backing).expect("writable backing checked above");
            full[addr..addr + isz].copy_from_slice(&packed);
            return Ok(w_none());
        }
        if !memoryview_is_index(index) {
            return Err(crate::PyError::type_error(
                "memoryview: invalid slice key, must be int or slice",
            ));
        }
        let mut i = getindex_w(index)?;
        if i < 0 {
            i += count;
        }
        if i < 0 || i >= count {
            return Err(crate::PyError::index_error("index out of bounds"));
        }
        let packed = memoryview_pack_value(&fmt, isz, value)?;
        // Re-check release after value coercion (see tuple path above).
        memoryview_check_released(mv)?;
        let addr = (offset + i * stride0) as usize;
        let full = memoryview_backing_bytes_mut(backing).expect("writable backing checked above");
        full[addr..addr + isz].copy_from_slice(&packed);
        Ok(w_none())
    }
}

/// `memoryview.tobytes` — copy the live view (honouring stride) to `bytes`.
fn memoryview_tobytes(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(
            &memoryview_gather_bytes(mv),
        ))
    }
}

/// `memoryview.__iter__` — yield the unpacked elements in order.
fn memoryview_iter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        crate::baseobjspace::iter(w_list_new(memoryview_values(mv)))
    }
}

/// `memoryview.readonly` — true for a bytes / array (Stage-1) backing or a
/// view explicitly made read-only via `toreadonly`.
fn memoryview_readonly(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(w_bool_from(pyre_object::memoryview::w_memoryview_readonly(
            mv,
        )))
    }
}

/// `memoryview.nbytes` — `product(shape) * itemsize`, the accessible bytes.
fn memoryview_nbytes(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(w_int_new(pyre_object::memoryview::w_memoryview_length(mv)))
    }
}

/// `memoryview._pypy_raw_address` — the integer raw address of the view's
/// backing store, base data pointer plus the view's byte offset
/// (`descr_pypy_raw_address` → `get_raw_address`).  Every pyre backing
/// (bytes / bytearray / array) is a contiguous in-heap store with a real
/// data pointer, so the "no raw address" branch is structurally unreachable.
fn memoryview_raw_address(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        let backing = pyre_object::memoryview::w_memoryview_backing(mv);
        let base = memoryview_backing_slice(backing).as_ptr() as usize;
        let offset = pyre_object::memoryview::w_memoryview_offset(mv) as usize;
        Ok(w_int_new((base + offset) as i64))
    }
}

/// `memoryview.format` — the struct format string of an element.
fn memoryview_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(pyre_object::memoryview::w_memoryview_format(mv))
    }
}

/// `memoryview.ndim` — the number of dimensions.
fn memoryview_ndim(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(w_int_new(pyre_object::memoryview::w_memoryview_ndim(mv)))
    }
}

/// `memoryview.obj` — the original exporter the view was built from.
fn memoryview_obj(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(pyre_object::memoryview::w_memoryview_obj(mv))
    }
}

/// `memoryview.itemsize` — the byte width of one element.
fn memoryview_itemsize(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(w_int_new(pyre_object::memoryview::w_memoryview_itemsize(
            mv,
        )))
    }
}

/// `memoryview.shape` — `tuple[int]` of per-dimension element counts.
fn memoryview_shape(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(pyre_object::memoryview::w_memoryview_shape(mv))
    }
}

/// `memoryview.strides` — `tuple[int]` of per-dimension byte steps.
fn memoryview_strides(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(pyre_object::memoryview::w_memoryview_strides(mv))
    }
}

/// `memoryview.__len__` — the element count `product(shape)` (1-D: `shape[0]`).
fn memoryview_len(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        let dim = pyre_object::memoryview::w_memoryview_ndim(mv);
        if dim == 0 {
            return Ok(w_int_new(1));
        }
        match pyre_object::tupleobject::w_tuple_getitem(
            pyre_object::memoryview::w_memoryview_shape(mv),
            0,
        ) {
            Some(s) => Ok(w_int_new(pyre_object::w_int_get_value(s))),
            None => Ok(w_int_new(0)),
        }
    }
}

/// `_tolist_rec` — the nested element-value list of dimension `idim` of an
/// N-D view, reading the raw backing by true strides.  The innermost
/// dimension unpacks `shape[ndim-1]` elements stepping `pos` by
/// `strides[ndim-1]`; an outer dimension collects `shape[idim]` sublists,
/// advancing `start` by `strides[idim]`.
unsafe fn memoryview_tolist_rec(
    mv: PyObjectRef,
    fmt: &str,
    full: &[u8],
    isz: usize,
    ndim: i64,
    idim: i64,
    start: i64,
) -> PyObjectRef {
    use pyre_object::memoryview::*;
    unsafe {
        let dimshape = memoryview_dim_value(w_memoryview_shape(mv), idim);
        let dimstride = memoryview_dim_value(w_memoryview_strides(mv), idim);
        let mut items = Vec::with_capacity(dimshape.max(0) as usize);
        let mut pos = start;
        if idim == ndim - 1 {
            for _ in 0..dimshape {
                items.push(memoryview_unpack_element(fmt, full, pos as usize, isz));
                pos += dimstride;
            }
        } else {
            for _ in 0..dimshape {
                items.push(memoryview_tolist_rec(
                    mv,
                    fmt,
                    full,
                    isz,
                    ndim,
                    idim + 1,
                    pos,
                ));
                pos += dimstride;
            }
        }
        w_list_new(items)
    }
}

/// `memoryview.tolist` — the element-value list (format-aware); a 1-D view
/// is flat, an N-D view nests one list per dimension (`_tolist_rec`).
fn memoryview_tolist(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        let ndim = pyre_object::memoryview::w_memoryview_ndim(mv);
        if ndim == 0 {
            // `buffer.py w_tolist` raises for a 0-dim view.
            return Err(crate::PyError::not_implemented(""));
        }
        if ndim == 1 {
            return Ok(w_list_new(memoryview_values(mv)));
        }
        let isz = pyre_object::memoryview::w_memoryview_itemsize(mv) as usize;
        let fmt = pyre_object::w_str_get_value(pyre_object::memoryview::w_memoryview_format(mv));
        let full = memoryview_backing_slice(pyre_object::memoryview::w_memoryview_backing(mv));
        let start = pyre_object::memoryview::w_memoryview_offset(mv);
        Ok(memoryview_tolist_rec(mv, fmt, full, isz, ndim, 0, start))
    }
}

/// `memoryview.cast(format[, shape])` — reinterpret a C-contiguous view
/// under a new native format and optionally a new N-D shape, sharing the
/// same backing (`descr_cast` → `_cast_to_1D` / `_cast_to_ND`).
fn memoryview_cast(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["format", "shape"], "cast")?;
    let mv = positional.first().copied().unwrap_or(w_none());
    let fmt_obj = resolve_pos_or_kw(positional.get(1).copied(), kwargs, "format", "cast", 1)?
        .ok_or_else(|| {
            crate::PyError::type_error("cast() missing required argument 'format' (pos 1)")
        })?;
    let shape_obj = resolve_pos_or_kw(positional.get(2).copied(), kwargs, "shape", "cast", 2)?;
    unsafe {
        use pyre_object::memoryview::*;
        memoryview_check_released(mv)?;
        if !pyre_object::is_str(fmt_obj) {
            return Err(crate::PyError::type_error(
                "memoryview: format argument must be a string",
            ));
        }
        let fmt = pyre_object::w_str_get_value(fmt_obj).to_owned();
        let has_shape = shape_obj.is_some_and(|s| !pyre_object::is_none(s));
        let orig_ndim = w_memoryview_ndim(mv);
        // Casts are restricted to C-contiguous source views.
        if !memoryview_contiguity(mv).0 {
            return Err(crate::PyError::type_error(
                "memoryview: casts are restricted to C-contiguous views",
            ));
        }
        // A reshape, or reinterpreting a multi-dim view, rejects an empty
        // dimension (`_zero_in_shape`).
        if has_shape || orig_ndim != 1 {
            let shape_t = w_memoryview_shape(mv);
            let has_zero = (0..orig_ndim).any(|i| {
                pyre_object::tupleobject::w_tuple_getitem(shape_t, i)
                    .map(|w| pyre_object::w_int_get_value(w))
                    .unwrap_or(0)
                    == 0
            });
            if has_zero {
                return Err(crate::PyError::type_error(
                    "memoryview: cannot casts view with zeros in shape or strides",
                ));
            }
        }
        // Validate the destination shape's dimension count before computing
        // the new element layout.
        let mut dims: Vec<i64> = Vec::new();
        if has_shape {
            let shape_seq = shape_obj.unwrap();
            if !(pyre_object::is_list(shape_seq) || pyre_object::is_tuple(shape_seq)) {
                let tname = match crate::typedef::r#type(shape_seq) {
                    Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
                    None => (*(*shape_seq).ob_type).name.to_string(),
                };
                return Err(crate::PyError::type_error(format!(
                    "expected list or tuple got {tname}"
                )));
            }
            dims = crate::baseobjspace::unpackiterable(shape_seq, -1)?
                .into_iter()
                .map(crate::baseobjspace::int_w)
                .collect::<Result<_, _>>()?;
            let ndim = dims.len() as i64;
            if ndim > 64 {
                return Err(crate::PyError::value_error(format!(
                    "memoryview: number of dimensions must not exceed {ndim}"
                )));
            }
            if ndim > 1 && orig_ndim != 1 {
                return Err(crate::PyError::type_error(
                    "memoryview: cast must be 1D -> ND or ND -> 1D",
                ));
            }
        }
        // _cast_to_1D: a native single-character destination format.
        let Some(new_itemsize) = memoryview_native_fmtchar(&fmt) else {
            return Err(crate::PyError::value_error(
                "memoryview: destination format must be a native single \
                 character format prefixed with an optional '@'",
            ));
        };
        let orig_fmt = pyre_object::w_str_get_value(w_memoryview_format(mv));
        if (memoryview_native_fmtchar(orig_fmt).is_none() || !memoryview_is_byte_format(orig_fmt))
            && !memoryview_is_byte_format(&fmt)
        {
            return Err(crate::PyError::type_error(
                "memoryview: cannot cast between two non-byte formats",
            ));
        }
        let total = w_memoryview_length(mv);
        if new_itemsize <= 0 || total % new_itemsize != 0 {
            return Err(crate::PyError::type_error(
                "memoryview: length is not a multiple of itemsize",
            ));
        }
        let offset = w_memoryview_offset(mv);
        let backing = w_memoryview_backing(mv);
        let obj = w_memoryview_obj(mv);
        let readonly = w_memoryview_readonly(mv);
        let w_fmt = w_str_new(&fmt);
        if !has_shape {
            let count = total / new_itemsize;
            let shape = pyre_object::w_tuple_new(vec![w_int_new(count)]);
            let strides = pyre_object::w_tuple_new(vec![w_int_new(new_itemsize)]);
            return Ok(w_memoryview_alloc(
                obj,
                backing,
                w_fmt,
                shape,
                strides,
                new_itemsize,
                1,
                offset,
                total,
                readonly,
                false,
            ));
        }
        // _cast_to_ND: `length = itemsize; for d in shape: length *= d`, then
        // `length != view.getlength()` rejects.  A negative dimension makes the
        // product mismatch `total`; checked multiplication keeps an overflow
        // (which can never equal a real buffer size) a rejection rather than a
        // debug-build panic.
        let ndim = dims.len() as i64;
        let mut product = new_itemsize;
        for &d in &dims {
            match product.checked_mul(d) {
                Some(p) => product = p,
                None => {
                    return Err(crate::PyError::type_error(
                        "memoryview: product(shape) * itemsize != buffer size",
                    ));
                }
            }
        }
        if product != total {
            return Err(crate::PyError::type_error(
                "memoryview: product(shape) * itemsize != buffer size",
            ));
        }
        let strides_v = memoryview_strides_from_shape(&dims, new_itemsize);
        let shape = pyre_object::w_tuple_new(dims.iter().map(|&d| w_int_new(d)).collect());
        let strides = pyre_object::w_tuple_new(strides_v.iter().map(|&s| w_int_new(s)).collect());
        Ok(w_memoryview_alloc(
            obj,
            backing,
            w_fmt,
            shape,
            strides,
            new_itemsize,
            ndim,
            offset,
            total,
            readonly,
            false,
        ))
    }
}

/// `memoryview.toreadonly` — a live read-only view sharing the same backing.
fn memoryview_toreadonly(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        use pyre_object::memoryview::*;
        memoryview_check_released(mv)?;
        Ok(w_memoryview_alloc(
            w_memoryview_obj(mv),
            w_memoryview_backing(mv),
            w_memoryview_format(mv),
            w_memoryview_shape(mv),
            w_memoryview_strides(mv),
            w_memoryview_itemsize(mv),
            w_memoryview_ndim(mv),
            w_memoryview_offset(mv),
            w_memoryview_length(mv),
            true,
            false,
        ))
    }
}

/// `memoryview.__repr__` — `memory_repr`: `<memory at 0x...>` keyed on the
/// view's own address (`<released memory at 0x...>` once released), not the
/// default `<memoryview object at 0x...>`.
fn memoryview_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    let label = if unsafe { pyre_object::memoryview::w_memoryview_released(mv) } {
        "released memory"
    } else {
        "memory"
    };
    Ok(w_str_new(&format!("<{label} at {mv:?}>")))
}

/// `memoryview.release` — drop the view; subsequent access raises ValueError.
/// Idempotent (a second `release` on an already-released view is a no-op),
/// matching `descr_release`.
fn memoryview_release(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        if !pyre_object::memoryview::w_memoryview_released(mv) {
            pyre_object::memoryview::w_memoryview_set_released(mv);
        }
    }
    Ok(w_none())
}

/// `memoryview.__enter__` — check-released, then return the view itself.
fn memoryview_enter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe { memoryview_check_released(mv)? };
    Ok(mv)
}

/// `memoryview.__exit__` — release on context-manager exit (any exc args).
fn memoryview_exit(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    memoryview_release(&args[..1])
}

/// The raw logical bytes (`view.as_str`) of an operand that exports a
/// contiguous buffer, or `None` when it exports none (so `__eq__` returns
/// NotImplemented).  `descr__cmp` compares the two `as_str` byte strings,
/// mirroring `space.buffer_w(w_other, space.BUF_CONTIG_RO)`: a memoryview,
/// a bytes-like object, or a non-bytes contiguous exporter (`array.array`)
/// are all gathered to bytes and compared.
unsafe fn memoryview_operand_bytes(obj: PyObjectRef) -> Option<Vec<u8>> {
    unsafe {
        if pyre_object::memoryview::is_w_memoryview(obj) {
            // A released view has no buffer to gather (its box is dropped);
            // `descr__cmp` falls through to identity, so report no bytes.
            if pyre_object::memoryview::w_memoryview_released(obj) {
                return None;
            }
            return Some(memoryview_gather_bytes(obj));
        }
        if pyre_object::bytesobject::is_bytes_like(obj) {
            return Some(pyre_object::bytesobject::bytes_like_data(obj).to_vec());
        }
        if let Ok(Some(b)) = crate::typedef::buffer_as_bytes_like(obj) {
            return Some(pyre_object::bytesobject::bytes_like_data(b).to_vec());
        }
        None
    }
}

/// True when `mv` or a memoryview `other` operand is released — either side
/// then compares by identity instead of reading a buffer past `release()`.
unsafe fn memoryview_released_either(mv: PyObjectRef, other: PyObjectRef) -> bool {
    unsafe {
        pyre_object::memoryview::w_memoryview_released(mv)
            || (pyre_object::memoryview::is_w_memoryview(other)
                && pyre_object::memoryview::w_memoryview_released(other))
    }
}

/// `memoryview.__eq__` — `descr__cmp('eq')`: compares the two views'
/// raw byte strings (`as_str`); NotImplemented for any other operand.
fn memoryview_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    let other = args.get(1).copied().unwrap_or(w_none());
    unsafe {
        // A released view (on either side) compares by identity (`view is None`
        // branch); its backing must not be read after release.
        if memoryview_released_either(mv, other) {
            return Ok(w_bool_from(mv == other));
        }
        let a = memoryview_gather_bytes(mv);
        match memoryview_operand_bytes(other) {
            Some(b) => Ok(w_bool_from(a == b)),
            None => Ok(pyre_object::w_not_implemented()),
        }
    }
}

/// `memoryview.__ne__` — `descr__cmp('ne')`, the negation of `__eq__`.
fn memoryview_ne(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    let other = args.get(1).copied().unwrap_or(w_none());
    unsafe {
        if memoryview_released_either(mv, other) {
            return Ok(w_bool_from(mv != other));
        }
        let a = memoryview_gather_bytes(mv);
        match memoryview_operand_bytes(other) {
            Some(b) => Ok(w_bool_from(a != b)),
            None => Ok(pyre_object::w_not_implemented()),
        }
    }
}

/// `memoryview.hex` — the view's bytes as a hex string, reusing the
/// bytes `hex(sep, bytes_per_sep)` formatter on a gathered copy.
fn memoryview_hex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    let w_bytes = unsafe {
        memoryview_check_released(mv)?;
        pyre_object::bytesobject::w_bytes_from_bytes(&memoryview_gather_bytes(mv))
    };
    let mut fwd = Vec::with_capacity(args.len());
    fwd.push(w_bytes);
    fwd.extend_from_slice(&args[1..]);
    crate::typedef::bytes_method_hex(&fwd)
}

/// `memoryview.__hash__` — `descr_hash`: a writable view is unhashable;
/// a read-only view hashes its raw bytes (so `hash(mv) == hash(bytes)`).
fn memoryview_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        if !pyre_object::memoryview::w_memoryview_readonly(mv) {
            return Err(crate::PyError::value_error(
                "cannot hash writable memoryview object",
            ));
        }
        // `compute_hash(self.view.as_str())` — the same content digest the
        // bytes path uses, so `hash(memoryview(b)) == hash(b)`.
        Ok(w_int_new(hash_str_bytes(&memoryview_gather_bytes(mv))))
    }
}

/// `memoryview.__delitem__` — memoryview does not support item deletion.
/// `memory_ass_sub` checks released, then read-only, before the delete
/// rejection, so a released view reports the released error and a read-only
/// view reports "cannot modify read-only memory".
pub(crate) fn memoryview_delitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        if pyre_object::memoryview::w_memoryview_readonly(mv) {
            return Err(crate::PyError::type_error("cannot modify read-only memory"));
        }
    }
    Err(crate::PyError::type_error("cannot delete memory"))
}

/// `_IsCContiguous` — C order has the last (fastest) dimension's stride
/// equal to `itemsize`, growing by the dimension sizes toward the front.
fn memoryview_is_c_contiguous(shape: &[i64], strides: &[i64], itemsize: i64) -> bool {
    let ndim = shape.len();
    if ndim == 0 {
        return true;
    }
    if ndim == 1 {
        return shape[0] == 1 || strides[0] == itemsize;
    }
    let mut sd = itemsize;
    for i in (0..ndim).rev() {
        if shape[i] == 0 {
            return true;
        }
        if strides[i] != sd {
            return false;
        }
        sd *= shape[i];
    }
    true
}

/// `_IsFortranContiguous` — Fortran order has the first (fastest)
/// dimension's stride equal to `itemsize`, growing toward the back.
fn memoryview_is_f_contiguous(shape: &[i64], strides: &[i64], itemsize: i64) -> bool {
    let ndim = shape.len();
    if ndim == 0 {
        return true;
    }
    if ndim == 1 {
        return shape[0] == 1 || strides[0] == itemsize;
    }
    let mut sd = itemsize;
    for i in 0..ndim {
        if shape[i] == 0 {
            return true;
        }
        if strides[i] != sd {
            return false;
        }
        sd *= shape[i];
    }
    true
}

/// `(c_contiguous, f_contiguous)` for a view, from `_init_flags` /
/// `PyBuffer_isContiguous`.  A 0-dim (scalar) view is both.
pub(crate) unsafe fn memoryview_contiguity(mv: PyObjectRef) -> (bool, bool) {
    use pyre_object::memoryview::*;
    unsafe {
        let ndim = w_memoryview_ndim(mv);
        if ndim == 0 {
            return (true, true);
        }
        let itemsize = w_memoryview_itemsize(mv);
        let shape_t = w_memoryview_shape(mv);
        let strides_t = w_memoryview_strides(mv);
        let read = |t: PyObjectRef, i: i64| {
            pyre_object::tupleobject::w_tuple_getitem(t, i)
                .map(|w| pyre_object::w_int_get_value(w))
                .unwrap_or(0)
        };
        let shape: Vec<i64> = (0..ndim).map(|i| read(shape_t, i)).collect();
        let strides: Vec<i64> = (0..ndim).map(|i| read(strides_t, i)).collect();
        (
            memoryview_is_c_contiguous(&shape, &strides, itemsize),
            memoryview_is_f_contiguous(&shape, &strides, itemsize),
        )
    }
}

/// `memoryview.c_contiguous` — the buffer is C-contiguous.
fn memoryview_c_contiguous(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(w_bool_from(memoryview_contiguity(mv).0))
    }
}

/// `memoryview.f_contiguous` — the buffer is Fortran-contiguous.
fn memoryview_f_contiguous(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(w_bool_from(memoryview_contiguity(mv).1))
    }
}

/// `memoryview.contiguous` — the buffer is C- or Fortran-contiguous.
fn memoryview_contiguous(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        let (c, f) = memoryview_contiguity(mv);
        Ok(w_bool_from(c || f))
    }
}

/// `memoryview.suboffsets` — always the empty tuple (no PIL-style views).
fn memoryview_suboffsets(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(pyre_object::w_tuple_new(vec![]))
    }
}

/// `memoryview.__new__` — `memoryview(object)`; `args[0]` is the class, so
/// exactly one buffer object follows.  Zero or more than one positional
/// argument raises the gateway arity TypeError.
fn memoryview_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let scope = bind_builtin_kwargs(&args[1..], &["object"], &[true], "memoryview")?;
    w_memoryview_new(scope[0])
}

/// Install the `memoryview` type-dict methods and properties.  Wired into
/// `MEMORYVIEW_TYPE` from `typedef::init_typeobjects`; each method reads the
/// native `W_MemoryView` fields rather than per-instance attribute slots.
pub(crate) fn init_memoryview_type(ns: &mut DictStorage) {
    type MvFn = fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>;
    // `__new__` is a `BuiltinFunction`-typed staticmethod descriptor like
    // every other native type's `tp_new` (typedef::make_new_descr), so it
    // does not bind the class and pickle's `isinstance(new, type(int.__new__))`
    // check matches.
    crate::dict_storage_store(
        ns,
        "__new__",
        crate::typedef::make_new_descr(memoryview_descr_new),
    );
    for (name, f, arity) in [
        ("__getitem__", memoryview_getitem as MvFn, 2u16),
        ("__setitem__", memoryview_setitem, 3),
        ("__len__", memoryview_len, 1),
        ("__iter__", memoryview_iter, 1),
        ("__repr__", memoryview_repr, 1),
        ("__eq__", memoryview_eq, 2),
        ("__ne__", memoryview_ne, 2),
        ("tobytes", memoryview_tobytes, 1),
        ("tolist", memoryview_tolist, 1),
        ("toreadonly", memoryview_toreadonly, 1),
        ("release", memoryview_release, 1),
        ("__enter__", memoryview_enter, 1),
        ("__hash__", memoryview_hash, 1),
        ("_pypy_raw_address", memoryview_raw_address, 1),
    ] {
        crate::dict_storage_store(ns, name, make_builtin_function_with_arity(name, f, arity));
    }
    // `__exit__(self, *exc)`, `__release_buffer__(self, view)`,
    // `__delitem__(self, *args)`, `hex(self, sep=, bytes_per_sep=)`, and
    // `cast(format[, shape])` take variable / optional trailing arguments,
    // so they register as plain (non-arity-pinned) builtins.
    for (name, f) in [
        ("__exit__", memoryview_exit as MvFn),
        ("__release_buffer__", memoryview_release),
        ("__delitem__", memoryview_delitem),
        ("hex", memoryview_hex),
        ("cast", memoryview_cast),
    ] {
        crate::dict_storage_store(ns, name, make_builtin_function(name, f));
    }
    for (attr, getter) in [
        ("obj", memoryview_obj as MvFn),
        ("format", memoryview_format),
        ("itemsize", memoryview_itemsize),
        ("nbytes", memoryview_nbytes),
        ("readonly", memoryview_readonly),
        ("ndim", memoryview_ndim),
        ("shape", memoryview_shape),
        ("strides", memoryview_strides),
        ("suboffsets", memoryview_suboffsets),
        ("c_contiguous", memoryview_c_contiguous),
        ("f_contiguous", memoryview_f_contiguous),
        ("contiguous", memoryview_contiguous),
    ] {
        crate::dict_storage_store(
            ns,
            attr,
            pyre_object::w_property_new(
                make_builtin_function_with_arity(attr, getter, 1),
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
            ),
        );
    }
}

pub fn install_default_builtins(namespace: &mut DictStorage) {
    namespace.get_or_insert_with("print", || {
        make_module_builtin_function("print", builtin_print)
    });
    namespace.get_or_insert_with("range", || {
        crate::typedef::gettypeobject(&pyre_object::functional::RANGE_TYPE)
    });
    namespace.get_or_insert_with("len", || {
        // operation.py `len(space, w_obj)` — one positional-only `obj`, so
        // `len` no longer receives the `__pyre_kw__` marker dict and any
        // keyword is rejected with "takes no keyword arguments".
        crate::gateway::make_module_builtin_function_with_arity_and_sig(
            "len",
            builtin_len,
            1,
            crate::gateway::Signature::new(vec!["obj"], None, None, 0, 1),
        )
    });
    namespace.get_or_insert_with("abs", || {
        // operation.py `abs(space, w_val)` — one positional-only `val`, so
        // `abs` no longer receives the `__pyre_kw__` marker dict and any
        // keyword is rejected with "takes no keyword arguments".
        crate::gateway::make_module_builtin_function_with_arity_and_sig(
            "abs",
            builtin_abs,
            1,
            crate::gateway::Signature::new(vec!["val"], None, None, 0, 1),
        )
    });
    namespace.get_or_insert_with("min", || make_module_builtin_function("min", builtin_min));
    namespace.get_or_insert_with("max", || make_module_builtin_function("max", builtin_max));
    namespace.get_or_insert_with("type", || crate::typedef::w_type());
    namespace.get_or_insert_with("isinstance", || {
        make_module_builtin_function_with_arity("isinstance", builtin_isinstance, 2)
    });
    namespace.get_or_insert_with("str", || crate::typedef::gettypeobject(&STR_TYPE));
    namespace.get_or_insert_with("repr", || {
        make_module_builtin_function_with_arity("repr", builtin_repr, 1)
    });
    namespace.get_or_insert_with("ascii", || {
        make_module_builtin_function_with_arity("ascii", builtin_ascii, 1)
    });
    namespace.get_or_insert_with("int", || crate::typedef::gettypeobject(&INT_TYPE));
    namespace.get_or_insert_with("float", || crate::typedef::gettypeobject(&FLOAT_TYPE));
    namespace.get_or_insert_with("bool", || crate::typedef::gettypeobject(&BOOL_TYPE));
    namespace.get_or_insert_with("True", || w_bool_from(true));
    namespace.get_or_insert_with("False", || w_bool_from(false));
    namespace.get_or_insert_with("None", || w_none());
    namespace.get_or_insert_with("NotImplemented", || w_not_implemented());
    namespace.get_or_insert_with("hasattr", || {
        make_module_builtin_function_with_arity("hasattr", builtin_hasattr, 2)
    });
    namespace.get_or_insert_with("getattr", || {
        make_module_builtin_function("getattr", builtin_getattr)
    });
    namespace.get_or_insert_with("setattr", || {
        make_module_builtin_function_with_arity("setattr", builtin_setattr, 3)
    });
    namespace.get_or_insert_with("delattr", || {
        make_module_builtin_function_with_arity("delattr", builtin_delattr, 2)
    });
    namespace.get_or_insert_with("tuple", || crate::typedef::gettypeobject(&TUPLE_TYPE));
    namespace.get_or_insert_with("list", || crate::typedef::gettypeobject(&LIST_TYPE));
    namespace.get_or_insert_with("dict", || crate::typedef::gettypeobject(&DICT_TYPE));
    namespace.get_or_insert_with("object", || {
        // `object` is a W_TypeObject, not a builtin function.
        // PyPy: baseobjspace.py w_object = W_TypeObject("object", ...)
        crate::typedef::w_object()
    });
    namespace.get_or_insert_with("super", || {
        make_module_builtin_function("super", builtin_super)
    });
    namespace.get_or_insert_with("id", || {
        make_module_builtin_function_with_arity("id", builtin_id, 1)
    });
    namespace.get_or_insert_with("hash", || {
        make_module_builtin_function_with_arity("hash", builtin_hash, 1)
    });
    namespace.get_or_insert_with("ord", || {
        make_module_builtin_function_with_arity("ord", builtin_ord, 1)
    });
    namespace.get_or_insert_with("chr", || {
        make_module_builtin_function_with_arity("chr", builtin_chr, 1)
    });
    namespace.get_or_insert_with("map", || {
        crate::typedef::gettypeobject(&pyre_object::functional::MAP_TYPE)
    });
    namespace.get_or_insert_with("zip", || {
        crate::typedef::gettypeobject(&pyre_object::functional::ZIP_TYPE)
    });
    namespace.get_or_insert_with("enumerate", || {
        crate::typedef::gettypeobject(&pyre_object::functional::ENUMERATE_TYPE)
    });
    namespace.get_or_insert_with("reversed", || {
        crate::typedef::gettypeobject(&pyre_object::functional::REVERSED_TYPE)
    });
    namespace.get_or_insert_with("sorted", || {
        make_module_builtin_function("sorted", builtin_sorted)
    });
    namespace.get_or_insert_with("iter", || {
        make_module_builtin_function("iter", builtin_iter)
    });
    namespace.get_or_insert_with("next", || {
        make_module_builtin_function("next", builtin_next)
    });
    namespace.get_or_insert_with("callable", || {
        make_module_builtin_function_with_arity("callable", builtin_callable, 1)
    });
    namespace.get_or_insert_with("vars", || {
        make_module_builtin_function("vars", builtin_vars)
    });
    namespace.get_or_insert_with("dir", || make_module_builtin_function("dir", builtin_dir));
    namespace.get_or_insert_with("__build_class__", || {
        make_module_builtin_function("__build_class__", |args| {
            crate::call::real_build_class(args)
        })
    });
    // bytearrayobject.py W_BytearrayObject — register the real type
    // (callable as a constructor and usable in isinstance(x, bytearray)).
    namespace.get_or_insert_with("bytearray", || {
        crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE)
    });
    // bytesobject.py W_BytesObject — immutable bytes type.
    namespace.get_or_insert_with("bytes", || {
        crate::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE)
    });
    namespace.get_or_insert_with("slice", || {
        // The slice type object, for isinstance(x, slice) checks.
        crate::typedef::gettypefor(&pyre_object::sliceobject::SLICE_TYPE)
            .unwrap_or(pyre_object::PY_NULL)
    });
    namespace.get_or_insert_with("frozenset", || {
        crate::typedef::gettypeobject(&pyre_object::setobject::FROZENSET_TYPE)
    });
    namespace.get_or_insert_with("set", || {
        crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE)
    });
    namespace.get_or_insert_with("property", || {
        crate::typedef::gettypeobject(&pyre_object::descriptor::PROPERTY_TYPE)
    });
    namespace.get_or_insert_with("staticmethod", || {
        crate::typedef::gettypeobject(&pyre_object::function::STATICMETHOD_TYPE)
    });
    namespace.get_or_insert_with("classmethod", || {
        crate::typedef::gettypeobject(&pyre_object::function::CLASSMETHOD_TYPE)
    });
    namespace.get_or_insert_with("Ellipsis", || pyre_object::special::w_ellipsis());
    namespace.get_or_insert_with("__debug__", || w_bool_from(true));
    namespace.get_or_insert_with("memoryview", || {
        crate::typedef::gettypeobject(&pyre_object::memoryview::MEMORYVIEW_TYPE)
    });
    namespace.get_or_insert_with("globals", || {
        make_module_builtin_function_with_arity("globals", builtin_globals, 0)
    });
    namespace.get_or_insert_with("locals", || {
        make_module_builtin_function_with_arity("locals", builtin_locals, 0)
    });
    namespace.get_or_insert_with("exec", || {
        make_module_builtin_function("exec", builtin_exec)
    });
    namespace.get_or_insert_with("eval", || {
        make_module_builtin_function("eval", builtin_eval)
    });
    namespace.get_or_insert_with("compile", || {
        make_module_builtin_function("compile", builtin_compile)
    });
    namespace.get_or_insert_with("complex", || {
        crate::typedef::gettypeobject(&pyre_object::COMPLEX_TYPE)
    });
    namespace.get_or_insert_with("filter", || {
        crate::typedef::gettypeobject(&pyre_object::functional::FILTER_TYPE)
    });
    namespace.get_or_insert_with("input", || {
        make_module_builtin_function("input", |_| Ok(pyre_object::w_str_new("")))
    });
    namespace.get_or_insert_with("open", || {
        make_module_builtin_function("open", builtin_open)
    });
    // Exception hierarchy — exceptions are real types so they can be
    // subclassed (`class FrozenInstanceError(AttributeError): pass`).
    // Built in dependency order: each subclass refers to its already-built
    // parent. PyPy: each typedef.py W_<Exception>.typedef registers a real
    // W_TypeObject in space.builtin.
    let base_exc = make_exc_type_with_init(
        "BaseException",
        exc_base_exception_new,
        Some(exc_base_exception_init),
        crate::typedef::w_object(),
    );
    crate::dict_storage_store(namespace, "BaseException", base_exc);

    let exception = make_exc_type("Exception", exc_exception_new, base_exc);
    crate::dict_storage_store(namespace, "Exception", exception);

    let arithmetic = make_exc_type("ArithmeticError", exc_arithmetic_error_new, exception);
    crate::dict_storage_store(namespace, "ArithmeticError", arithmetic);
    crate::dict_storage_store(
        namespace,
        "ZeroDivisionError",
        make_exc_type("ZeroDivisionError", exc_zero_division_new, arithmetic),
    );
    crate::dict_storage_store(
        namespace,
        "OverflowError",
        make_exc_type("OverflowError", exc_overflow_error_new, arithmetic),
    );
    crate::dict_storage_store(
        namespace,
        "FloatingPointError",
        make_exc_type("FloatingPointError", exc_arithmetic_error_new, arithmetic),
    );

    let lookup_error = make_exc_type("LookupError", exc_lookup_error_new, exception);
    crate::dict_storage_store(namespace, "LookupError", lookup_error);
    crate::dict_storage_store(
        namespace,
        "IndexError",
        make_exc_type("IndexError", exc_index_error_new, lookup_error),
    );
    crate::dict_storage_store(
        namespace,
        "KeyError",
        make_exc_type("KeyError", exc_key_error_new, lookup_error),
    );

    crate::dict_storage_store(
        namespace,
        "AttributeError",
        make_exc_type_with_init(
            "AttributeError",
            exc_attribute_error_new,
            Some(exc_attribute_error_init),
            exception,
        ),
    );
    crate::dict_storage_store(
        namespace,
        "TypeError",
        make_exc_type("TypeError", exc_type_error_new, exception),
    );
    let value_error = make_exc_type("ValueError", exc_value_error_new, exception);
    crate::dict_storage_store(namespace, "ValueError", value_error);
    let name_error = make_exc_type_with_init(
        "NameError",
        exc_name_error_new,
        Some(exc_name_error_init),
        exception,
    );
    crate::dict_storage_store(namespace, "NameError", name_error);
    // `exceptions.c` — `UnboundLocalError(NameError)`.  pyre raises a plain
    // NameError for unbound locals, but the builtin name must exist for
    // `except UnboundLocalError` clauses in copyreg / pickle.
    crate::dict_storage_store(
        namespace,
        "UnboundLocalError",
        make_exc_type("UnboundLocalError", exc_name_error_new, name_error),
    );

    let runtime_error = make_exc_type("RuntimeError", exc_runtime_error_new, exception);
    crate::dict_storage_store(namespace, "RuntimeError", runtime_error);
    crate::dict_storage_store(
        namespace,
        "NotImplementedError",
        make_exc_type(
            "NotImplementedError",
            exc_not_implemented_error_new,
            runtime_error,
        ),
    );
    crate::dict_storage_store(
        namespace,
        "RecursionError",
        make_exc_type("RecursionError", exc_recursion_error_new, runtime_error),
    );

    crate::dict_storage_store(
        namespace,
        "StopIteration",
        make_exc_type("StopIteration", exc_stop_iteration_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "StopAsyncIteration",
        make_exc_type("StopAsyncIteration", exc_exception_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "GeneratorExit",
        make_exc_type("GeneratorExit", exc_generator_exit_new, base_exc),
    );
    crate::dict_storage_store(
        namespace,
        "SystemExit",
        make_exc_type("SystemExit", exc_system_exit_new, base_exc),
    );
    crate::dict_storage_store(
        namespace,
        "KeyboardInterrupt",
        make_exc_type("KeyboardInterrupt", exc_base_exception_new, base_exc),
    );

    let import_error = make_exc_type_with_init(
        "ImportError",
        exc_import_error_new,
        Some(exc_import_error_init),
        exception,
    );
    crate::dict_storage_store(namespace, "ImportError", import_error);
    crate::dict_storage_store(
        namespace,
        "ModuleNotFoundError",
        make_exc_type("ModuleNotFoundError", exc_import_error_new, import_error),
    );
    crate::dict_storage_store(
        namespace,
        "AssertionError",
        make_exc_type("AssertionError", exc_assertion_error_new, exception),
    );

    let os_error = make_exc_type_with_init(
        "OSError",
        exc_os_error_new,
        Some(exc_os_error_init),
        exception,
    );
    crate::dict_storage_store(namespace, "OSError", os_error);
    crate::dict_storage_store(namespace, "IOError", os_error);
    // `exceptions.c` — `EnvironmentError` is a deprecated alias of `OSError`.
    crate::dict_storage_store(namespace, "EnvironmentError", os_error);
    crate::dict_storage_store(
        namespace,
        "FileNotFoundError",
        make_exc_type("FileNotFoundError", exc_file_not_found_error_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "FileExistsError",
        make_exc_type("FileExistsError", exc_os_error_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "PermissionError",
        make_exc_type("PermissionError", exc_os_error_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "NotADirectoryError",
        make_exc_type("NotADirectoryError", exc_os_error_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "IsADirectoryError",
        make_exc_type("IsADirectoryError", exc_os_error_new, os_error),
    );

    let warning = make_exc_type("Warning", exc_exception_new, exception);
    crate::dict_storage_store(namespace, "Warning", warning);
    for warn_name in [
        "UserWarning",
        "DeprecationWarning",
        "PendingDeprecationWarning",
        "RuntimeWarning",
        "FutureWarning",
        "ImportWarning",
        "UnicodeWarning",
        "BytesWarning",
        "ResourceWarning",
        "SyntaxWarning",
        "EncodingWarning",
    ] {
        crate::dict_storage_store(
            namespace,
            warn_name,
            make_exc_type(warn_name, exc_exception_new, warning),
        );
    }

    let unicode_error = make_exc_type("UnicodeError", exc_unicode_error_new, value_error);
    crate::dict_storage_store(namespace, "UnicodeError", unicode_error);
    crate::dict_storage_store(
        namespace,
        "UnicodeDecodeError",
        make_exc_type_with_init(
            "UnicodeDecodeError",
            exc_unicode_decode_error_new,
            Some(exc_unicode_decode_error_init),
            unicode_error,
        ),
    );
    crate::dict_storage_store(
        namespace,
        "UnicodeEncodeError",
        make_exc_type_with_init(
            "UnicodeEncodeError",
            exc_unicode_encode_error_new,
            Some(exc_unicode_encode_error_init),
            unicode_error,
        ),
    );
    crate::dict_storage_store(
        namespace,
        "UnicodeTranslateError",
        make_exc_type_with_init(
            "UnicodeTranslateError",
            exc_unicode_translate_error_new,
            Some(exc_unicode_translate_error_init),
            unicode_error,
        ),
    );

    crate::dict_storage_store(
        namespace,
        "BufferError",
        make_exc_type("BufferError", exc_exception_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "MemoryError",
        make_exc_type("MemoryError", exc_memory_error_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "ReferenceError",
        make_exc_type("ReferenceError", exc_reference_error_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "SystemError",
        make_exc_type("SystemError", exc_system_error_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "EOFError",
        make_exc_type("EOFError", exc_exception_new, exception),
    );
    let syntax_error = make_exc_type("SyntaxError", exc_syntax_error_new, exception);
    crate::dict_storage_store(namespace, "SyntaxError", syntax_error);
    let indentation_error = make_exc_type("IndentationError", exc_syntax_error_new, syntax_error);
    crate::dict_storage_store(namespace, "IndentationError", indentation_error);
    crate::dict_storage_store(
        namespace,
        "TabError",
        make_exc_type("TabError", exc_syntax_error_new, indentation_error),
    );
    crate::dict_storage_store(
        namespace,
        "BlockingIOError",
        make_exc_type("BlockingIOError", exc_os_error_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "ChildProcessError",
        make_exc_type("ChildProcessError", exc_os_error_new, os_error),
    );
    let connection_error = make_exc_type("ConnectionError", exc_os_error_new, os_error);
    crate::dict_storage_store(namespace, "ConnectionError", connection_error);
    crate::dict_storage_store(
        namespace,
        "BrokenPipeError",
        make_exc_type("BrokenPipeError", exc_os_error_new, connection_error),
    );
    crate::dict_storage_store(
        namespace,
        "ConnectionAbortedError",
        make_exc_type("ConnectionAbortedError", exc_os_error_new, connection_error),
    );
    crate::dict_storage_store(
        namespace,
        "ConnectionRefusedError",
        make_exc_type("ConnectionRefusedError", exc_os_error_new, connection_error),
    );
    crate::dict_storage_store(
        namespace,
        "ConnectionResetError",
        make_exc_type("ConnectionResetError", exc_os_error_new, connection_error),
    );
    crate::dict_storage_store(
        namespace,
        "InterruptedError",
        make_exc_type("InterruptedError", exc_os_error_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "ProcessLookupError",
        make_exc_type("ProcessLookupError", exc_os_error_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "TimeoutError",
        make_exc_type("TimeoutError", exc_os_error_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "BaseExceptionGroup",
        make_exc_type("BaseExceptionGroup", exc_base_exception_new, base_exc),
    );
    crate::dict_storage_store(
        namespace,
        "ExceptionGroup",
        make_exc_type("ExceptionGroup", exc_exception_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "PythonFinalizationError",
        make_exc_type(
            "PythonFinalizationError",
            exc_runtime_error_new,
            runtime_error,
        ),
    );
    namespace.get_or_insert_with("any", || {
        make_module_builtin_function_with_arity("any", builtin_any, 1)
    });
    namespace.get_or_insert_with("all", || {
        make_module_builtin_function_with_arity("all", builtin_all, 1)
    });
    namespace.get_or_insert_with("sum", || make_module_builtin_function("sum", builtin_sum));
    namespace.get_or_insert_with("round", || {
        make_module_builtin_function("round", builtin_round)
    });
    namespace.get_or_insert_with("divmod", || {
        make_module_builtin_function("divmod", builtin_divmod)
    });
    namespace.get_or_insert_with("pow", || make_module_builtin_function("pow", builtin_pow));
    namespace.get_or_insert_with("hex", || make_module_builtin_function("hex", builtin_hex));
    namespace.get_or_insert_with("oct", || make_module_builtin_function("oct", builtin_oct));
    namespace.get_or_insert_with("bin", || make_module_builtin_function("bin", builtin_bin));
    namespace.get_or_insert_with("format", || {
        make_module_builtin_function("format", builtin_format)
    });
    namespace.get_or_insert_with("issubclass", || {
        make_module_builtin_function_with_arity("issubclass", builtin_issubclass, 2)
    });
    namespace.get_or_insert_with("__import__", || {
        make_module_builtin_function("__import__", builtin_import_stub)
    });

    // Descriptor types
    namespace.get_or_insert_with("property", || {
        crate::typedef::gettypeobject(&pyre_object::descriptor::PROPERTY_TYPE)
    });
    // staticmethod/classmethod registered as types for isinstance() support.
    // The type's __new__ creates the descriptor wrapper.
    namespace.get_or_insert_with("staticmethod", || {
        crate::typedef::gettypeobject(&pyre_object::function::STATICMETHOD_TYPE)
    });
    namespace.get_or_insert_with("classmethod", || {
        crate::typedef::gettypeobject(&pyre_object::function::CLASSMETHOD_TYPE)
    });
}

/// Create a fresh namespace seeded with the default builtins.
pub fn new_builtin_dict_storage() -> DictStorage {
    crate::typedef::init_typeobjects();
    let mut namespace = DictStorage::new();
    install_default_builtins(&mut namespace);
    namespace
}

/// `pypy/objspace/std/dictmultiobject.py:60-69
/// allocate_and_init_instance(module=True)` parity — allocate the
/// builtins module dict as a `W_ModuleDictObject` backed by
/// `ModuleDictStrategy` (`celldict.py:28`).  Seeds the same entries
/// `install_default_builtins` populates on a `DictStorage`, then
/// transfers them into the strategy storage; the temporary
/// `DictStorage` drops at function exit and the W_ModuleDictObject
/// owns the live builtins.
pub fn new_builtin_module_dict() -> pyre_object::PyObjectRef {
    crate::typedef::init_typeobjects();
    let mut seed = DictStorage::new();
    install_default_builtins(&mut seed);
    let w_dict = pyre_object::w_module_dict_new();
    // MixedModule parity: the interp-level builtins carry "builtins" as
    // `__module__`, so `pickle.save_global` resolves them by reference
    // without the `whichmodule` guess (every module namespace exposes the
    // builtins, which makes that guess unstable).
    let module_name = w_str_new("builtins");
    for (key, &value) in seed.entries() {
        if !value.is_null() {
            unsafe { crate::function::builtin_function_set_module(value, module_name) };
            unsafe { pyre_object::w_dict_setitem_str(w_dict, key, value) };
        }
    }
    w_dict
}

/// `print`'s `sep`/`end` type check, applied up front: `None` (or absent)
/// selects the default and a non-`str` is a TypeError ("sep must be None or
/// a string, not <type>").  A `str` value is returned for the caller to
/// render at write time with `Py_PRINT_RAW`, which goes through `str()`, so
/// a `str` subclass `__str__` override is honored — and a raising one
/// surfaces only after the preceding argument has already been written.
fn print_sep_check(
    val: Option<PyObjectRef>,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    let Some(v) = val else {
        return Ok(None);
    };
    if unsafe { pyre_object::is_none(v) } {
        return Ok(None);
    }
    if unsafe { pyre_object::is_str(v) } {
        return Ok(Some(v));
    }
    Err(crate::PyError::type_error(format!(
        "{name} must be None or a string, not {}",
        crate::type_methods::arg_type_name(v)
    )))
}

/// Render `str(obj)` for writing to the (utf-8, strict) stdout stream.  The
/// common all-UTF-8 result is returned directly; a lone surrogate is routed
/// through `encode_object`'s strict handler, raising `UnicodeEncodeError`
/// rather than panicking in `w_str_get_value`.
unsafe fn print_render(obj: PyObjectRef) -> Result<String, crate::PyError> {
    let w = unsafe { crate::py_str_wtf8(obj)? };
    if let Ok(s) = w.as_str() {
        return Ok(s.to_owned());
    }
    let s_obj = pyre_object::w_str_from_wtf8(w);
    let bytes = crate::type_methods::encode_object(s_obj, "utf-8", "strict")?;
    Ok(String::from_utf8(bytes).expect("strict utf-8 encode yields valid utf-8"))
}

/// A text stream's `encoding`/`errors` (defaults `utf-8`/`strict`), read so a
/// `str` write encodes through `encode_object` — routing a lone surrogate to
/// the error handler instead of panicking in `w_str_get_value`.
unsafe fn stream_encoding_errors(stream: PyObjectRef) -> (String, String) {
    let attr = |name: &str, default: &str| -> String {
        crate::baseobjspace::getattr_str(stream, name)
            .ok()
            .filter(|v| !v.is_null() && unsafe { pyre_object::is_str(*v) })
            .map(|v| unsafe { pyre_object::w_str_get_value(v) }.to_string())
            .unwrap_or_else(|| default.to_string())
    };
    (attr("encoding", "utf-8"), attr("errors", "strict"))
}

/// `print(*args)` — write space-separated str representations to stdout.
/// The sink `print()` uses when no explicit `file=` is given, resolved from
/// the live `sys.stdout` on each call — `bltinmodule.c builtin_print` /
/// `app_io.py print_` map `file is None` to `sys.stdout`, so a Python-level
/// `sys.stdout = ...` redirects `print()`.
enum DefaultPrintTarget {
    /// Unmodified default stdout (`sys.stdout is sys.__stdout__`): keep pyre's
    /// native `print_output` path (its strict-utf-8 `print_render` render and
    /// direct write), leaving default output and surrogate handling unchanged.
    Native,
    /// A rebound `sys.stdout`; write through its `write` / `flush` methods.
    Rebound(PyObjectRef),
    /// `sys.stdout` is `None`; emit nothing (builtin_print returns `None`).
    Silent,
}

/// Resolve `print()`'s default sink from the live `sys` module.
///
/// Only a user redirect (`sys.stdout` rebound to some object other than the
/// saved `sys.__stdout__`) is routed through Python `write` / `flush`; the
/// unmodified default keeps the native path. A missing `sys.stdout` attribute
/// raises `RuntimeError("lost sys.stdout")` as builtin_print does.
fn resolve_default_print_target() -> Result<DefaultPrintTarget, crate::PyError> {
    let Some(sys_mod) = crate::importing::get_sys_module("sys") else {
        // No `sys` yet (very early bootstrap) — native path.
        return Ok(DefaultPrintTarget::Native);
    };
    let stdout = match crate::baseobjspace::getattr_str(sys_mod, "stdout") {
        Ok(w) => w,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
            return Err(crate::PyError::runtime_error("lost sys.stdout"));
        }
        Err(e) => return Err(e),
    };
    if unsafe { pyre_object::is_none(stdout) } {
        return Ok(DefaultPrintTarget::Silent);
    }
    if let Ok(orig) = crate::baseobjspace::getattr_str(sys_mod, "__stdout__") {
        if std::ptr::eq(orig, stdout) {
            return Ok(DefaultPrintTarget::Native);
        }
    }
    Ok(DefaultPrintTarget::Rebound(stdout))
}

fn builtin_print(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // Check if last arg is a kwargs dict (from CALL_KW builtin dispatch).
    // Distinguished from regular dict args by __pyre_kw__ marker key.
    let is_kwargs = !args.is_empty()
        && unsafe {
            let last = *args.last().unwrap();
            is_dict(last)
                && pyre_object::w_dict_lookup(last, w_str_new("__pyre_kw__"))
                    .is_some_and(pyre_object::kw_marker::is_kw_marker_sentinel)
        };
    let (positional, end, sep, file, flush) = if is_kwargs {
        let kwargs = *args.last().unwrap();
        // app_io.py print_ — the app-level signature is
        // `(*args, sep, end, file, flush)`, so any other keyword is an
        // unexpected-keyword TypeError.
        for (k, _) in unsafe { pyre_object::w_dict_items(kwargs) } {
            let name = unsafe { pyre_object::w_str_get_wtf8(k) };
            match name.as_str() {
                Ok("__pyre_kw__") | Ok("sep") | Ok("end") | Ok("file") | Ok("flush") => {}
                _ => {
                    return Err(crate::PyError::type_error(format!(
                        "print() got an unexpected keyword argument '{name}'"
                    )));
                }
            }
        }
        let end_val = unsafe { pyre_object::w_dict_lookup(kwargs, w_str_new("end")) };
        let sep_val = unsafe { pyre_object::w_dict_lookup(kwargs, w_str_new("sep")) };
        // The type check is up front; the str() rendering happens at write
        // time so a raising `__str__` leaves the preceding output in place.
        let end_obj = print_sep_check(end_val, "end")?;
        let sep_obj = print_sep_check(sep_val, "sep")?;
        // `file=None` (or absent) uses the native stdout path; any other
        // object is written through its `write` / `flush` methods.
        let file_obj = match unsafe { pyre_object::w_dict_lookup(kwargs, w_str_new("file")) } {
            Some(f) if !unsafe { pyre_object::is_none(f) } => Some(f),
            _ => None,
        };
        let flush = match unsafe { pyre_object::w_dict_lookup(kwargs, w_str_new("flush")) } {
            Some(f) => crate::baseobjspace::is_true(f)?,
            None => false,
        };
        (&args[..args.len() - 1], end_obj, sep_obj, file_obj, flush)
    } else {
        (args, None, None, None, false)
    };

    // With no explicit `file` (absent or `file=None`), the sink is the live
    // `sys.stdout`, resolved per call so a Python-level rebinding redirects
    // `print()`.  The unmodified default keeps the native path (`file = None`).
    let file = match file {
        Some(f) => Some(f),
        None => match resolve_default_print_target()? {
            DefaultPrintTarget::Native => None,
            DefaultPrintTarget::Rebound(fp) => Some(fp),
            DefaultPrintTarget::Silent => return Ok(w_none()),
        },
    };

    // `bltinmodule.c print_impl` writes incrementally: `str(arg)`, then the
    // separator before each following arg, then `end`.  Each source is rendered
    // at emit time so a raising `__str__` leaves the bytes already emitted on
    // the stream.  With a `file`, `str(source)` is handed to `file.write` as a
    // str object untouched (`PyFile_WriteObject`), so a lone surrogate is the
    // sink's concern — a `StringIO` or custom writer accepts it.  The native
    // stdout path renders through the strict utf-8 error handler in
    // `print_render`.
    let emit = |source: PyObjectRef| -> Result<(), crate::PyError> {
        let Some(fp) = file else {
            let s = unsafe { print_render(source)? };
            crate::print_output(&s);
            return Ok(());
        };
        let s_obj = pyre_object::w_str_from_wtf8(unsafe { crate::py_str_wtf8(source)? });
        let r = crate::baseobjspace::call_method(fp, "write", &[s_obj]);
        if r.is_null() {
            return Err(crate::call::take_call_error()
                .unwrap_or_else(|| crate::PyError::runtime_error("print: file.write() failed")));
        }
        Ok(())
    };
    for (i, &obj) in positional.iter().enumerate() {
        if i > 0 {
            emit(sep.unwrap_or_else(|| w_str_new(" ")))?;
        }
        emit(obj)?;
    }
    emit(end.unwrap_or_else(|| w_str_new("\n")))?;
    if flush {
        match file {
            None => {
                crate::host_seam::flush_stdout();
            }
            Some(fp) => {
                let r = crate::baseobjspace::call_method(fp, "flush", &[]);
                if r.is_null() {
                    return Err(crate::call::take_call_error().unwrap_or_else(|| {
                        crate::PyError::runtime_error("print: file.flush() failed")
                    }));
                }
            }
        }
    }
    Ok(w_none())
}

/// Bind `builtins._` best-effort; a missing `builtins` module (early
/// bootstrap) is ignored.
fn set_builtins_underscore(value: PyObjectRef) {
    if let Some(b) = crate::importing::get_sys_module("builtins") {
        let _ = crate::baseobjspace::setattr_str(b, "_", value);
    }
}

/// `sys.displayhook(value)` — print `repr(value)` followed by a newline to
/// the live `sys.stdout` and bind `builtins._` to the value. A `None` value
/// prints nothing and leaves `_` unchanged (`sys_displayhook` in sysmodule.c).
pub(crate) fn sys_displayhook(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let value = args.first().copied().unwrap_or_else(w_none);
    if unsafe { pyre_object::is_none(value) } {
        return Ok(w_none());
    }
    // `_` is cleared before rendering so a failing repr does not leave a
    // stale binding, then set to the value once the write succeeds.
    set_builtins_underscore(w_none());
    let repr = pyre_object::w_str_from_wtf8(unsafe { crate::display::py_repr_wtf8(value)? });
    let newline = w_str_new("\n");
    match resolve_default_print_target()? {
        DefaultPrintTarget::Native => {
            let s = unsafe { print_render(repr)? };
            crate::print_output(&s);
            crate::print_output("\n");
        }
        DefaultPrintTarget::Rebound(fp) => {
            for part in [repr, newline] {
                let r = crate::baseobjspace::call_method(fp, "write", &[part]);
                if r.is_null() {
                    return Err(crate::call::take_call_error().unwrap_or_else(|| {
                        crate::PyError::runtime_error("displayhook: file.write() failed")
                    }));
                }
            }
        }
        DefaultPrintTarget::Silent => return Ok(w_none()),
    }
    set_builtins_underscore(value);
    Ok(w_none())
}

/// `space.index` re-wraps a result whose type is not exactly `int` (a
/// bool, or a strict int subclass) as a plain int (descroperation.py:622
/// `index`).  A range stores its bounds wrapped, so normalize each here —
/// otherwise `range(True).stop` would expose `True` instead of `1`.
///
/// # Safety
/// `obj` must be a valid object.
unsafe fn range_index_bound(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let w = crate::baseobjspace::space_index(obj)?;
    Ok(pyre_object::range_bigint_to_obj(
        pyre_object::range_obj_to_bigint(w),
    ))
}

/// `range(stop)` / `range(start, stop[, step])` — `functional.py
/// W_Range.descr_new`.  Each bound passes through `space.index`
/// (`__index__`) and is stored wrapped, so a range may span past a
/// machine word; `iter()` then produces a `rangeiterator` (machine-int,
/// JIT-specializable) or a `longrange_iterator` accordingly.
pub(crate) fn builtin_range(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let n = args.len();
    if n == 0 {
        return Err(crate::PyError::type_error(
            "range expected at least 1 argument, got 0",
        ));
    }
    if n > 3 {
        return Err(crate::PyError::type_error(format!(
            "range expected at most 3 arguments, got {n}"
        )));
    }
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let mut w_start = range_index_bound(args[0])?;
        pyre_object::gc_roots::pin_root(w_start);
        let w_stop;
        let w_step;
        if n == 1 {
            // Only `stop` given — `w_start, w_stop = 0, w_start`.
            w_stop = w_start;
            w_start = w_int_new(0);
            pyre_object::gc_roots::pin_root(w_start);
            w_step = w_int_new(1);
            pyre_object::gc_roots::pin_root(w_step);
        } else {
            w_stop = range_index_bound(args[1])?;
            pyre_object::gc_roots::pin_root(w_stop);
            if n == 3 {
                w_step = range_index_bound(args[2])?;
                pyre_object::gc_roots::pin_root(w_step);
                if pyre_object::range_obj_to_bigint(w_step) == BigInt::from(0) {
                    return Err(crate::PyError::value_error(
                        "step argument must not be zero",
                    ));
                }
            } else {
                w_step = w_int_new(1);
                pyre_object::gc_roots::pin_root(w_step);
            }
        }
        Ok(pyre_object::w_range_new(w_start, w_stop, w_step))
    }
}

/// True iff `callable` is the builtin `len` function object — a
/// builtin-code function whose code wraps [`builtin_len`].  The JIT
/// walker uses this to recognize a `len(x)` residual it can lower to the
/// container's inline length read.
pub fn is_builtin_len_function(callable: PyObjectRef) -> bool {
    unsafe {
        if callable.is_null() || !crate::is_function(callable) {
            return false;
        }
        let code = crate::function_get_code(callable) as PyObjectRef;
        if code.is_null() || !crate::gateway::is_builtin_code(code) {
            return false;
        }
        std::ptr::fn_addr_eq(
            crate::gateway::builtin_code_get(code),
            builtin_len as crate::gateway::BuiltinCodeFn,
        )
    }
}

/// `len(obj)` — return the length of an object.
/// `len(obj)` — PyPy: operation.py len → space.len_w
fn builtin_len(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // operation.py `len(space, w_obj)`.  The gateway Signature binds the keyword
    // form at the call site, so `args` is positional only and never carries the
    // `__pyre_kw__` marker: a single argument is `obj` (even a dict that holds
    // the marker key), so index it directly.  A wrong positional count carries
    // no single-dict value, so route it through the gateway for the faithful
    // `_match_signature` arity error.
    if let [obj] = args {
        return crate::baseobjspace::len(*obj);
    }
    let obj = parse_single_required(args, "obj", "len")?;
    crate::baseobjspace::len(obj)
}

/// `abs(x)` — return the absolute value of a number.
pub fn builtin_abs(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // operation.py `abs(space, w_val)`.  The gateway Signature binds the keyword
    // form at the call site, so a single positional argument is `val` (even a
    // dict holding the marker key); index it directly and route a wrong
    // positional count through the gateway for the `_match_signature` arity error.
    let obj = match args {
        [val] => *val,
        _ => parse_single_required(args, "val", "abs")?,
    };
    unsafe {
        if is_bool(obj) {
            return Ok(w_int_new(w_bool_get_value(obj) as i64));
        }
        if is_int(obj) {
            let v = w_int_get_value(obj);
            // i64::MIN.abs() overflows; promote to long
            return Ok(match v.checked_abs() {
                Some(r) => w_int_new(r),
                None => w_long_new(-BigInt::from(v)),
            });
        }
        if is_long(obj) {
            let val = w_long_get_value(obj).clone();
            return Ok(w_long_new(if val < BigInt::from(0) { -val } else { val }));
        }
        if is_float(obj) {
            return Ok(w_float_new(w_float_get_value(obj).abs()));
        }
        if pyre_object::is_complex(obj) {
            // abs(complex) → the float magnitude.
            let re = pyre_object::w_complex_get_real(obj);
            let im = pyre_object::w_complex_get_imag(obj);
            return Ok(w_float_new(re.hypot(im)));
        }
    }
    // Instance __abs__ — PyPy: baseobjspace.py abs
    unsafe {
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__abs__") {
                return crate::call::call_function_impl_result(method, &[obj]);
            }
        }
    }
    Err(crate::PyError::type_error(format!(
        "bad operand type for abs(): '{}'",
        unsafe { (*(*obj).ob_type).name }
    )))
}

/// Strip the trailing `__pyre_kw__` dict that `call_with_kwargs`
/// (`call.rs`) appends for builtin callees and return the positional
/// slice paired with a keyword lookup helper.
///
/// PRE-EXISTING-ADAPTATION (builtin kwargs ABI, consumer side). PyPy's
/// gateway gives each builtin a `Signature` (`gateway.py:740 BuiltinCode`,
/// `:804`) and resolves keywords by name through `args.parse_obj` →
/// `_match_signature` (`argument.py:173`) before the interp-level function
/// runs; the builtin never sees a marker dict. Pyre's flat `BuiltinCodeFn`
/// ABI lacks that Signature surface, so each kwarg-aware builtin reaches into
/// the `__pyre_kw__`-tagged trailing dict via this shared helper. The builtin
/// Signature/unwrap_spec gateway is not yet ported; once it routes builtin
/// kwargs through `Arguments::_match_signature` into named slots, this helper
/// and the `__pyre_kw__` marker can be removed.
pub(crate) fn split_builtin_kwargs(args: &[PyObjectRef]) -> (&[PyObjectRef], Option<PyObjectRef>) {
    if let Some(&last) = args.last() {
        // The marker dict stores an unforgeable sentinel under `__pyre_kw__`
        // (`call_with_kwargs`), so detection is by value identity.  A dict
        // passed positionally that merely contains a `__pyre_kw__` string key
        // (`float({'__pyre_kw__': True})`) carries a different value and is a
        // value, not the marker, so it must not be stripped.
        let is_marker = unsafe {
            is_dict(last)
                && pyre_object::w_dict_lookup(last, w_str_new("__pyre_kw__"))
                    .is_some_and(pyre_object::kw_marker::is_kw_marker_sentinel)
        };
        if is_marker {
            return (&args[..args.len() - 1], Some(last));
        }
    }
    (args, None)
}

/// True when the kwargs dict from [`split_builtin_kwargs`] carries a real
/// keyword (any entry other than the `__pyre_kw__` marker).  An empty
/// `**{}` therefore reports `false`.
pub(crate) fn has_real_kwargs(kwargs: Option<PyObjectRef>) -> bool {
    let Some(dict) = kwargs else {
        return false;
    };
    unsafe { pyre_object::w_dict_str_entries(dict) }
        .iter()
        .any(|(key, _)| key != "__pyre_kw__")
}

/// Number of real keyword arguments in the kwargs dict from
/// [`split_builtin_kwargs`] — every entry other than the `__pyre_kw__`
/// marker.  The clinic-style "takes at most N arguments (M given)" builtins
/// (`sum`, `round`, `pow`) count positionals plus this against their limit.
pub(crate) fn real_kwarg_count(kwargs: Option<PyObjectRef>) -> usize {
    let Some(dict) = kwargs else {
        return 0;
    };
    unsafe { pyre_object::w_dict_str_entries(dict) }
        .iter()
        .filter(|(key, _)| key != "__pyre_kw__")
        .count()
}

/// Look up a single keyword argument from the kwargs dict produced by
/// `split_builtin_kwargs`. Returns `None` when no kwargs dict is present
/// or the requested key is absent.
pub(crate) fn kwarg_get(kwargs: Option<PyObjectRef>, name: &str) -> Option<PyObjectRef> {
    let dict = kwargs?;
    unsafe { pyre_object::w_dict_lookup(dict, w_str_new(name)) }
}

/// Reject any keyword argument whose name is not in `allowed`.  Mirrors
/// PyPy's `unwrap_spec` strict-keyword behaviour — for example
/// `pypy/module/__builtin__/functional.py:198-201 min_max` raises
/// `TypeError("min() got unexpected keyword argument")` whenever an
/// unknown kwarg slips in (only `key` and `default` are accepted).
/// pyre's flat builtin ABI has to police this manually because
/// `split_builtin_kwargs` does not enforce a signature.
///
/// `fn_name` is the bare function name used in the error message
/// ("min", "zip_longest", ...).  The `__pyre_kw__` marker entry the
/// gateway appends is filtered out; it is an implementation detail of
/// the kwargs encoding, not a user-visible argument.
pub(crate) fn kwarg_reject_unknown(
    kwargs: Option<PyObjectRef>,
    allowed: &[&str],
    fn_name: &str,
) -> Result<(), crate::PyError> {
    let dict = match kwargs {
        Some(d) => d,
        None => return Ok(()),
    };
    let entries = unsafe { pyre_object::w_dict_str_entries_wtf8(dict) };
    for (key, _) in entries.iter() {
        if key.as_str() == Ok("__pyre_kw__") {
            continue;
        }
        if !allowed.iter().any(|name| key.as_str() == Ok(*name)) {
            return Err(crate::PyError::type_error(format!(
                "{fn_name}() got an unexpected keyword argument '{key}'"
            )));
        }
    }
    Ok(())
}

/// Bind a positional-or-keyword parameter that follows a positional-only
/// prefix (`eval`/`exec`/`compile` style).  Prefers the positional slot and
/// falls back to the matching keyword.  Raises the argument-clinic
/// "given by name and position" TypeError when the same parameter is supplied
/// both ways.  `pos_index` is the 1-based position used in that message.
pub(crate) fn bind_pos_or_kw(
    positional: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
    slot: usize,
    key: &str,
    fn_name: &str,
    pos_index: usize,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    match (positional.get(slot).copied(), kwarg_get(kwargs, key)) {
        (Some(_), Some(_)) => Err(crate::PyError::type_error(format!(
            "argument for {fn_name}() given by name ('{key}') and position ({pos_index})"
        ))),
        (Some(p), None) => Ok(Some(p)),
        (None, Some(k)) => Ok(Some(k)),
        (None, None) => Ok(None),
    }
}

/// `true` when the last argument is the `__pyre_kw__`-tagged dict the
/// CALL_KW builtin dispatch appends — i.e. the call carried keywords.
pub(crate) fn has_builtin_kwargs(args: &[PyObjectRef]) -> bool {
    matches!(args.last(), Some(&last) if unsafe {
        is_dict(last)
            && pyre_object::w_dict_lookup(last, w_str_new("__pyre_kw__"))
                .is_some_and(pyre_object::kw_marker::is_kw_marker_sentinel)
    })
}

/// Resolve a single positional-or-keyword builtin argument: prefer the
/// positional value, fall back to the keyword `name`.  Supplying both
/// raises the `argument.py:_match_keywords` TypeError
/// "argument for X() given by name ('name') and position (N)" (N is the
/// 1-based positional index of the slot).  An absent argument is `None`.
pub(crate) fn resolve_pos_or_kw(
    positional: Option<PyObjectRef>,
    kwargs: Option<PyObjectRef>,
    name: &str,
    fn_name: &str,
    position: usize,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    let keyword = kwarg_get(kwargs, name);
    match (positional, keyword) {
        (Some(_), Some(_)) => Err(crate::PyError::type_error(format!(
            "argument for {fn_name}() given by name ('{name}') and position ({position})"
        ))),
        (Some(v), None) | (None, Some(v)) => Ok(Some(v)),
        (None, None) => Ok(None),
    }
}

/// Bind positional + `__pyre_kw__` keyword arguments into a resolved
/// scope of length `names.len()`, mirroring the gateway's
/// `Arguments._match_signature` (`pypy/interpreter/argument.py`). Each
/// slot is filled by a positional, then by a keyword of the matching
/// name; an absent optional slot becomes `PY_NULL` (the generated
/// `#[pyre_function]` unwrap reads that as "argument omitted"). An absent
/// required slot, an unknown keyword, a keyword duplicating a positional,
/// or too many positionals raises `TypeError`.
///
/// This is the consumer-side counterpart that lets a builtin resolve
/// keywords by parameter name without a per-function `Signature`; the
/// `#[pyre_function]` wrapper supplies the name/required tables it knows
/// at expansion time.
pub(crate) fn bind_builtin_kwargs(
    args: &[PyObjectRef],
    names: &[&str],
    required: &[bool],
    fn_name: &str,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    if positional.len() > names.len() {
        return Err(crate::PyError::type_error(format!(
            "{fn_name}() takes at most {} positional argument{} ({} given)",
            names.len(),
            if names.len() == 1 { "" } else { "s" },
            positional.len(),
        )));
    }
    let mut scope: Vec<PyObjectRef> = vec![PY_NULL; names.len()];
    let mut filled: Vec<bool> = vec![false; names.len()];
    for (i, &v) in positional.iter().enumerate() {
        scope[i] = v;
        filled[i] = true;
    }
    if let Some(dict) = kwargs {
        let entries = unsafe { pyre_object::w_dict_str_entries_wtf8(dict) };
        for (key, val) in entries.iter() {
            if key.as_str() == Ok("__pyre_kw__") {
                continue;
            }
            match names.iter().position(|n| key.as_str() == Ok(*n)) {
                Some(idx) => {
                    if filled[idx] {
                        return Err(crate::PyError::type_error(format!(
                            "{fn_name}() got multiple values for argument '{key}'"
                        )));
                    }
                    scope[idx] = *val;
                    filled[idx] = true;
                }
                None => {
                    return Err(crate::PyError::type_error(format!(
                        "{fn_name}() got an unexpected keyword argument '{key}'"
                    )));
                }
            }
        }
    }
    for i in 0..names.len() {
        if !filled[i] && required[i] {
            return Err(crate::PyError::type_error(format!(
                "{fn_name}() missing required argument: '{}'",
                names[i]
            )));
        }
    }
    Ok(scope)
}

/// Resolve a builtin with a single required positional-or-keyword parameter
/// through the gateway `parse_into_scope`, so the argument binds by name and
/// the trailing `__pyre_kw__` marker dict never leaks as a value. Mirrors an
/// `interp2app` function with `Signature([name])` (e.g. `operation.py`
/// `abs(space, w_val)` / `len(space, w_obj)`): a missing argument, an unknown
/// keyword, or a surplus positional raises the matching `_match_signature`
/// TypeError.
fn parse_single_required(
    args: &[PyObjectRef],
    name: &'static str,
    fn_name: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    // Fast path for the hot fixed-arity call (`len(x)` / `abs(x)`): one
    // positional and no keywords binds directly, skipping the Signature /
    // Arguments / scope allocation the general keyword and arity-error path
    // needs.  Zero or surplus positionals, or any keyword, fall through so the
    // `_match_signature` TypeError is still raised.
    if kwargs.is_none() && positional.len() == 1 {
        return Ok(positional[0]);
    }
    let mut keyword_names_w: Vec<PyObjectRef> = Vec::new();
    let mut keywords_w: Vec<PyObjectRef> = Vec::new();
    if let Some(dict) = kwargs {
        for (key, val) in unsafe { pyre_object::w_dict_str_entries_wtf8(dict) } {
            if key.as_str() == Ok("__pyre_kw__") {
                continue;
            }
            keyword_names_w.push(pyre_object::w_str_from_wtf8(key));
            keywords_w.push(val);
        }
    }
    let signature = crate::gateway::Signature::new(vec![name], None, None, 0, 0);
    let arguments = crate::argument::Arguments::with_kw(positional, &keyword_names_w, &keywords_w);
    let mut scope_w = vec![PY_NULL; signature.scope_length()];
    arguments.parse_into_scope(PY_NULL, &mut scope_w, fn_name, &signature, None, PY_NULL)?;
    Ok(scope_w[0])
}

/// Reject `f(x, name=...)` when `name` already arrived positionally.
/// The flat builtin ABI leaves this validation to each kw-aware method.
pub(crate) fn kwarg_reject_duplicate(
    kwargs: Option<PyObjectRef>,
    fn_name: &str,
    name: &str,
    positional_present: bool,
) -> Result<(), crate::PyError> {
    if positional_present && kwarg_get(kwargs, name).is_some() {
        return Err(crate::PyError::type_error(format!(
            "{fn_name}() got multiple values for argument '{name}'"
        )));
    }
    Ok(())
}

/// `space.index_w(obj)` parity — `pypy/interpreter/baseobjspace.py
/// space.index_w` returns the int value of an object exposing
/// `__index__`.  Pyre handles the int / long / bool fast paths
/// directly and falls through to looking up `__index__` on the
/// object's type, mirroring PyPy's `lookup_in_type` pass before
/// raising `TypeError`.
pub(crate) fn space_index_w(obj: PyObjectRef) -> Result<i64, crate::PyError> {
    unsafe {
        if pyre_object::is_int(obj) {
            return Ok(pyre_object::w_int_get_value(obj));
        }
        if pyre_object::is_bool(obj) {
            return Ok(if pyre_object::w_bool_get_value(obj) {
                1
            } else {
                0
            });
        }
        if let Some(w_type) = crate::typedef::r#type(obj) {
            if let Some(index_fn) = crate::baseobjspace::lookup_in_type(w_type, "__index__") {
                let result = crate::call::call_function_impl_result(index_fn, &[obj])?;
                if pyre_object::is_int(result) {
                    return Ok(pyre_object::w_int_get_value(result));
                }
                if pyre_object::is_bool(result) {
                    return Ok(if pyre_object::w_bool_get_value(result) {
                        1
                    } else {
                        0
                    });
                }
            }
        }
    }
    let tp_name = unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => "object".to_string(),
        }
    };
    Err(crate::PyError::type_error(format!(
        "'{tp_name}' object cannot be interpreted as an integer"
    )))
}

/// Convert an int or long object to BigInt for comparison.
pub(crate) unsafe fn obj_to_bigint(obj: PyObjectRef) -> BigInt {
    unsafe {
        if is_int(obj) {
            BigInt::from(w_int_get_value(obj))
        } else {
            w_long_get_value(obj).clone()
        }
    }
}

/// `min(*args)` / `min(iterable)` — return the smallest value.
///
/// `pypy/module/__builtin__/functional.py:188-218 min_max`:
///   - reject any kwargs other than `key` / `default`
///   - reject `default=` paired with multiple positional args
///   - require ≥1 positional arg
fn builtin_min(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    min_max_dispatch(args, /* want_max= */ false, "min")
}

/// `max(a, b)` / `max(iterable)` — return the largest of two values or an iterable.
fn builtin_max(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    min_max_dispatch(args, /* want_max= */ true, "max")
}

fn min_max_dispatch(
    args: &[PyObjectRef],
    want_max: bool,
    fn_name: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    // functional.py:198-201 — only `key` and `default` are accepted.
    kwarg_reject_unknown(kwargs, &["key", "default"], fn_name)?;
    let key_fn = kwarg_get(kwargs, "key").filter(|k| unsafe { !pyre_object::is_none(*k) });
    let default = kwarg_get(kwargs, "default");
    // functional.py:216-218 — empty positional → TypeError, not panic.
    if positional.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "{fn_name}() expected at least one argument, got 0"
        )));
    }
    // functional.py:206-210 — `default=` is only meaningful for the
    // single-iterable form; combining it with multiple positional args
    // is a user error.
    if positional.len() > 1 && default.is_some() {
        return Err(crate::PyError::type_error(format!(
            "Cannot specify a default for {fn_name}() with multiple positional arguments"
        )));
    }
    let items: Vec<PyObjectRef> = if positional.len() == 1 {
        collect_iterable(positional[0])?
    } else {
        positional.to_vec()
    };
    if items.is_empty() {
        if let Some(d) = default {
            return Ok(d);
        }
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            format!("{fn_name}() iterable argument is empty"),
        ));
    }
    select_extremum(&items, key_fn, want_max)
}

/// Shared min/max body — `pypy/module/__builtin__/functional.py:115-148
/// min_max`.  Builds (key, item) pairs (identity when no `key=`),
/// keeps a running best by comparing keys via `space.gt`/`space.lt`
/// (the PyPy compare paths invoke `__gt__` / `__lt__` and propagate
/// errors), returns the corresponding item.  PyPy's stable-tie rule:
/// keep the first-seen extremum (`<` for min, `>` for max), matching
/// CPython 3.x semantics.
fn select_extremum(
    items: &[PyObjectRef],
    key_fn: Option<PyObjectRef>,
    want_max: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let key_of = |item: PyObjectRef| -> PyObjectRef {
        match key_fn {
            Some(kf) => crate::call_function(kf, &[item]),
            None => item,
        }
    };
    let cmp_op = if want_max {
        crate::baseobjspace::CompareOp::Gt
    } else {
        crate::baseobjspace::CompareOp::Lt
    };
    let mut best_item = items[0];
    let mut best_key = key_of(best_item);
    for &item in &items[1..] {
        let key = key_of(item);
        // `functional.py:139 if space.is_true(space.gt(key, best_key))`
        // — route through the generic comparison dispatch which
        // handles int/long/str/float/tuple natively and falls
        // through to user-defined `__gt__`/`__lt__` for other
        // types.  Errors (TypeError from incomparable types) are
        // propagated to the caller as PyPy does.
        let result = crate::baseobjspace::compare(key, best_key, cmp_op)?;
        if crate::baseobjspace::is_true(result)? {
            best_item = item;
            best_key = key;
        }
    }
    Ok(best_item)
}

/// `type(obj)` — return the type name as a string (simplified).
/// `type(obj)` — return the type of an object as a W_TypeObject.
///
/// PyPy: `space.type(w_obj)` → W_TypeObject
pub(crate) fn type_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // type.__new__(metatype, name, bases, dict)
    // May be called with extra self-binding from super():
    //   [self, metatype, name, bases, dict] — 5 args
    //   [metatype, name, bases, dict] — 4 args
    //   [metatype, obj] — 2 args (type(obj))
    // Find the (name, bases, dict) triple by scanning for the first str arg.
    // Also extract the metatype (first type arg before the name str).
    // The class-definition keywords arrive as a trailing `__pyre_kw__`
    // dict (the builtin kwargs ABI); strip it before the arity scan and
    // hand it to __init_subclass__ via `type_descr_new_with_metaclass`.
    let (pos, kwargs) = split_builtin_kwargs(args);
    let mut w_metaclass = pyre_object::PY_NULL;
    for i in 0..pos.len() {
        if unsafe { pyre_object::is_str(pos[i]) } && i + 2 < pos.len() {
            // Extract metatype from preceding args
            for j in 0..i {
                if unsafe { pyre_object::is_type(pos[j]) } {
                    w_metaclass = pos[j];
                }
            }
            return type_descr_new_with_metaclass(&pos[i..], w_metaclass, kwargs);
        }
    }
    if pos.len() == 1 && unsafe { pyre_object::is_type(pos[0]) } {
        return Err(crate::PyError::type_error("type() takes 1 or 3 arguments"));
    }
    if pos.len() == 1 {
        return type_descr_new_without_metaclass(pos, kwargs);
    }
    if pos.len() == 2 {
        return type_descr_new_without_metaclass(&pos[1..], kwargs);
    }
    Err(crate::PyError::type_error("type() takes 1 or 3 arguments"))
}
fn type_descr_new_without_metaclass(
    args: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    type_descr_new_with_metaclass(args, pyre_object::PY_NULL, kwargs)
}

/// typeobject.py:141 `_check_surrogate` — a type name may not contain a
/// lone surrogate.  Scan the code points through the surrogate-aware WTF-8
/// view (reading the name as `&str` would fail on the surrogate) and raise
/// `UnicodeEncodeError('utf8', name, pos, pos + 1, 'surrogates not allowed')`
/// at the first one, matching `check_utf8(name, allow_surrogates=False)`.
pub(crate) fn check_surrogate(w_name: PyObjectRef) -> Result<(), crate::PyError> {
    let wtf8 = unsafe { pyre_object::w_str_get_wtf8(w_name) };
    let mut pos = 0usize;
    for cp in wtf8.code_points() {
        let c = cp.to_u32();
        if c >= 0xd800 && c <= 0xdfff {
            return Err(crate::typedef::unicode_encode_error(
                "utf8",
                w_name,
                pos,
                pos + 1,
                "surrogates not allowed",
            ));
        }
        pos += 1;
    }
    Ok(())
}

/// `type_new_staticmethod` / `type_new_classmethod` (Objects/typeobject.c):
/// a class body's plain-function `__new__` becomes a `staticmethod` and
/// `__init_subclass__` / `__class_getitem__` become `classmethod`s, so that
/// `cls.__dict__['__new__'].__func__` resolves and the descriptors bind with
/// the right implicit first argument.
pub(crate) fn type_new_wrap_special_methods(ns: &mut crate::DictStorage) {
    if let Some(f) = ns.get("__new__").copied() {
        if unsafe { crate::function::is_function(f) }
            && !unsafe { pyre_object::function::is_staticmethod(f) }
        {
            crate::dict_storage_store(ns, "__new__", pyre_object::function::w_staticmethod_new(f));
        }
    }
    for name in ["__init_subclass__", "__class_getitem__"] {
        if let Some(f) = ns.get(name).copied() {
            if unsafe { crate::function::is_function(f) }
                && !unsafe { pyre_object::function::is_classmethod(f) }
            {
                crate::dict_storage_store(ns, name, pyre_object::function::w_classmethod_new(f));
            }
        }
    }
}

fn type_descr_new_with_metaclass(
    args: &[PyObjectRef],
    w_metaclass: PyObjectRef,
    kwargs: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 && args.len() != 3 {
        return Err(crate::PyError::type_error("type() takes 1 or 3 arguments"));
    }
    // type(name, bases, dict) — 3-arg form creates a new type
    // PyPy: typeobject.py type.__new__(metatype, name, bases, dict)
    if args.len() == 3 {
        let name_obj = args[0];
        let bases = args[1];
        let w_namespace_dict = args[2];
        // typeobject.py:953 `_check_surrogate(space, name)` — reject a lone
        // surrogate in the name before it is read as UTF-8 below.
        if unsafe { pyre_object::is_str(name_obj) } {
            check_surrogate(name_obj)?;
        }
        let name = unsafe { pyre_object::w_str_get_value(name_obj) };

        // CPython: calculate_metaclass — if bases have a custom metaclass,
        // delegate to that metaclass instead of using type.__new__ directly.
        if w_metaclass.is_null() && !bases.is_null() && unsafe { is_tuple(bases) } {
            let n = unsafe { w_tuple_len(bases) };
            for i in 0..n {
                if let Some(base) = unsafe { pyre_object::w_tuple_getitem(bases, i as i64) } {
                    if unsafe { pyre_object::is_type(base) } {
                        // baseobjspace.py:76 — metaclass from w_class
                        let w_metaclass = unsafe {
                            let w_class = (*base).w_class;
                            let w_type_type = crate::typedef::w_type();
                            if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                                Some(w_class)
                            } else {
                                None
                            }
                        };
                        if let Some(w_metaclass) = w_metaclass {
                            // Delegate: call metaclass(name, bases, dict, **kwds)
                            // Pass extra args from the original call
                            let mut metaclass_args = vec![name_obj, bases, w_namespace_dict];
                            if args.len() > 3 {
                                metaclass_args.extend_from_slice(&args[3..]);
                            }
                            return Ok(crate::call_function(w_metaclass, &metaclass_args));
                        }
                    }
                }
            }
        }

        // Convert dict to DictStorage.  `w_dict_items` dispatches
        // through `is_module_dict`, so the rare `__build_class__`
        // case where the namespace is a W_ModuleDictObject still
        // walks correctly.
        let mut class_ns = Box::new(crate::DictStorage::new());
        class_ns.fix_ptr();
        // type_new_classcell — capture the `__classcell__` cell and keep
        // both explicit class cells out of the new type's `__dict__`
        // (CPython consumes them here rather than storing them).
        let mut classcell = pyre_object::PY_NULL;
        // `type.__new__` accepts any `dict` subclass as the namespace
        // (the check is `PyDict_Check`, not `PyDict_CheckExact`); resolve
        // the dict backing so e.g. an `enum._EnumDict` class body is
        // walked instead of dropped.
        let w_ns_backing = unsafe { crate::type_methods::resolve_dict_backing(w_namespace_dict) };
        if !w_ns_backing.is_null() {
            for (k, v) in unsafe { pyre_object::w_dict_items(w_ns_backing) } {
                if unsafe { is_str(k) } {
                    let key = unsafe { pyre_object::w_str_get_wtf8(k) };
                    if key.as_str() == Ok("__classcell__") {
                        if !unsafe { pyre_object::is_cell(v) } {
                            let tp_name = match unsafe { crate::typedef::r#type(v) } {
                                Some(tp) => unsafe { pyre_object::w_type_get_name(tp) }.to_string(),
                                None => "object".to_string(),
                            };
                            return Err(crate::PyError::type_error(format!(
                                "__classcell__ must be a nonlocal cell, not {tp_name}"
                            )));
                        }
                        classcell = v;
                        continue;
                    }
                    if key.as_str() == Ok("__classdictcell__") {
                        continue;
                    }
                    crate::dict_storage_store_wtf8(&mut class_ns, key, v);
                }
            }
        }
        let ns_ptr = Box::into_raw(class_ns);
        unsafe { type_new_wrap_special_methods(&mut *ns_ptr) };

        // Default bases to (object,) if empty
        let w_effective_bases =
            if bases.is_null() || !unsafe { is_tuple(bases) } || unsafe { w_tuple_len(bases) } == 0
            {
                let w_object = crate::typedef::w_object();
                if !w_object.is_null() {
                    pyre_object::w_tuple_new(vec![w_object])
                } else {
                    bases
                }
            } else {
                bases
            };

        // CPython: calculate_metaclass — delegate to winner if different
        let default_meta = if w_metaclass.is_null() {
            crate::typedef::w_type()
        } else {
            w_metaclass
        };
        let w_winner = crate::call::calculate_metaclass(default_meta, w_effective_bases)
            .unwrap_or(default_meta);
        if !std::ptr::eq(w_winner, default_meta) {
            // Winner is a different metaclass — delegate to its __new__
            if let Some(w_metaclass_new) =
                unsafe { crate::baseobjspace::lookup_in_type(w_winner, "__new__") }
            {
                // `__new__` is stored as a staticmethod; unwrap before the
                // direct delegation call.
                let w_metaclass_new = unsafe {
                    if pyre_object::function::is_staticmethod(w_metaclass_new) {
                        pyre_object::function::w_staticmethod_get_func(w_metaclass_new)
                    } else {
                        w_metaclass_new
                    }
                };
                let mut new_args = vec![w_winner, name_obj, bases, w_namespace_dict];
                if args.len() > 3 {
                    new_args.extend_from_slice(&args[3..]);
                }
                drop(unsafe { Box::from_raw(ns_ptr) });
                return Ok(crate::call_function(w_metaclass_new, &new_args));
            }
        }
        let w_metaclass = w_winner;

        let w_type = pyre_object::w_type_new(name, w_effective_bases, ns_ptr as *mut u8);
        // typeobject.py:1143-1204 create_all_slots parity.
        unsafe {
            let ns = &*ns_ptr;
            crate::call::create_all_slots(w_type, ns, w_effective_bases)?;
        }
        // rclass.py:739-743 — set w_class (typeptr) at allocation time.
        // For type objects, w_class is the metaclass (type(C) → Meta).
        // baseobjspace.py:76 getclass() returns the metatype.
        unsafe {
            (*w_type).w_class = w_metaclass;
        }
        let mro = unsafe { crate::baseobjspace::compute_default_mro(w_type) };
        unsafe { pyre_object::w_type_set_mro(w_type, mro) };
        // typeobject.py:373-377 ready() — link self into each base's
        // `weak_subclasses` so `mutated()` and `__subclasses__()`
        // observe this class.
        unsafe { pyre_object::typeobject::w_type_ready(w_type) };

        // type_new_classcell — bind the captured `__classcell__` to the
        // new type so `__class__` / zero-arg `super()` in the methods
        // resolve; the key was already dropped from the namespace above.
        if !classcell.is_null() && unsafe { pyre_object::is_cell(classcell) } {
            unsafe { pyre_object::w_cell_set(classcell, w_type) };
        }

        // _set_names (typeobject.py:1006) — call `__set_name__(owner, name)`
        // on each descriptor in the type's FINAL `__dict__` (`w_type.dict_w`),
        // i.e. the filtered namespace with `__classcell__`/`__classdictcell__`
        // already removed, not the original backing.  Collect the entries
        // first so the storage borrow is released before the call, which may
        // re-enter the type's dict.
        let set_name_entries: Vec<(Wtf8Buf, PyObjectRef)> = unsafe {
            (*ns_ptr)
                .entries_wtf8()
                .map(|(k, v)| (k.to_owned(), *v))
                .collect()
        };
        for (key, v) in set_name_entries {
            let k = pyre_object::w_str_from_wtf8(key);
            unsafe { crate::baseobjspace::set_name(w_type, k, v) }?;
        }

        // type_new_init_subclass — fire __init_subclass__ with the
        // keywords that reached type.__new__ (the stripped `__pyre_kw__`
        // dict).  This is the single site for the metaclass path; the
        // default-metaclass `__build_class__` shortcut fires it itself
        // because it bypasses type.__new__.
        let init_subclass_kwargs: Vec<(PyObjectRef, PyObjectRef)> = match kwargs {
            Some(kw) => unsafe {
                pyre_object::w_dict_items(kw)
                    .into_iter()
                    .filter(|(k, _)| {
                        is_str(*k) && pyre_object::w_str_get_wtf8(*k).as_str() != Ok("__pyre_kw__")
                    })
                    .collect()
            },
            None => Vec::new(),
        };
        crate::call::call_init_subclass_on_bases(w_type, w_effective_bases, &init_subclass_kwargs)?;

        return Ok(w_type);
    }

    // type(obj) — 1-arg form returns the type
    // PyPy objspace.py:400: space.type(w_obj) → w_obj.getclass(space)
    // typedef::type() respects __class__ override for all object kinds.
    let obj = args[0];
    if let Some(tp) = crate::typedef::r#type(obj) {
        return Ok(tp);
    }
    if obj.is_null() {
        return Ok(crate::typedef::gettypeobject(
            &pyre_object::pyobject::NONE_TYPE,
        ));
    }
    let name = unsafe { (*(*obj).ob_type).name };
    Ok(box_str_constant(rustpython_wtf8::Wtf8::new(name)))
}

/// `isinstance(obj, cls)` — pypy/module/__builtin__/abstractinst.py
/// `app_isinstance` → `abstract_isinstance_w(allow_override=True)`.
fn builtin_isinstance(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "isinstance() takes exactly two arguments ({} given)",
            args.len()
        )));
    }
    Ok(w_bool_from(crate::baseobjspace::isinstance(
        args[0], args[1],
    )?))
}

/// isinstance(obj, cls) for JIT fast path.
///
/// Returns Some(bool) if the check can be resolved, None if cls format
/// is not supported for the fast path (e.g. tuple of types).
/// Uses the same MRO-based `issubtype_w` as the full dispatch.
pub fn call_isinstance(obj: PyObjectRef, cls: PyObjectRef) -> Option<bool> {
    unsafe {
        if is_type(cls) {
            return Some(crate::baseobjspace::isinstance_w(obj, cls));
        }
    }
    None
}

/// `issubclass(cls, classinfo)` — pypy/module/__builtin__/abstractinst.py
/// `app_issubclass` → `abstract_issubclass_w(allow_override=True)`.
fn builtin_issubclass(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "issubclass() takes exactly two arguments ({} given)",
            args.len()
        )));
    }
    Ok(w_bool_from(crate::baseobjspace::issubclass(
        args[0], args[1],
    )?))
}

// Descroperation helpers (lookup_type_special, should_try_reverse_first,
// try_dispatch_binary_special, try_dispatch_ternary_special,
// try_int_long_pow_with_modulo, binary_builtin_type_error,
// box_bigint_result, issubtype_w) live in `crate::baseobjspace` because
// they are space-level semantics shared between the builtin module,
// weakproxy wrappers, and any future opcode dispatch.

/// Exception type constructor — called as e.g. `ValueError("msg")`.
///
/// `pypy/module/exceptions/interp_exceptions.py:121-124
/// W_BaseException.descr_init` stores the constructor positional
/// arguments on `self.args_w` (an RPython list), then
/// `descr_str/descr_repr` (line 126-147) format from the same field.
/// Pyre wraps the args into a `W_ListObject` and stamps it into the
/// typed slot via `w_exception_set_args`, matching PyPy's
/// `self.args_w = args_w` shape; `w_exception_get_args` rebuilds a
/// fresh tuple per read so `e.args` mirrors
/// `space.newtuple(self.args_w)` semantics.  The message string keeps
/// driving `w_exception_get_message` for the lower-level error path.
macro_rules! exc_constructor {
    ($fn_name:ident, $kind:expr) => {
        fn $fn_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            // `interp_exceptions.py:121-124 W_BaseException.descr_init`:
            // `self.args_w = args_w`.  The string form of the exception
            // is derived from `args_w` on demand (`descr_str`), so the
            // constructor only captures the args — no eager message copy.
            let exc = pyre_object::interp_exceptions::w_exception_new_empty($kind);
            let args_list = pyre_object::w_list_new(args.to_vec());
            unsafe {
                pyre_object::interp_exceptions::w_exception_set_args(exc, args_list);
            }
            Ok(exc)
        }
    };
}

exc_constructor!(
    exc_base_exception,
    pyre_object::interp_exceptions::ExcKind::BaseException
);
exc_constructor!(
    exc_exception,
    pyre_object::interp_exceptions::ExcKind::Exception
);
exc_constructor!(
    exc_arithmetic_error,
    pyre_object::interp_exceptions::ExcKind::ArithmeticError
);
exc_constructor!(
    exc_zero_division,
    pyre_object::interp_exceptions::ExcKind::ZeroDivisionError
);
exc_constructor!(
    exc_type_error,
    pyre_object::interp_exceptions::ExcKind::TypeError
);
exc_constructor!(
    exc_value_error,
    pyre_object::interp_exceptions::ExcKind::ValueError
);
exc_constructor!(
    exc_key_error,
    pyre_object::interp_exceptions::ExcKind::KeyError
);
exc_constructor!(
    exc_index_error,
    pyre_object::interp_exceptions::ExcKind::IndexError
);
exc_constructor!(
    exc_attribute_error,
    pyre_object::interp_exceptions::ExcKind::AttributeError
);
exc_constructor!(
    exc_name_error,
    pyre_object::interp_exceptions::ExcKind::NameError
);
exc_constructor!(
    exc_runtime_error,
    pyre_object::interp_exceptions::ExcKind::RuntimeError
);
exc_constructor!(
    exc_stop_iteration,
    pyre_object::interp_exceptions::ExcKind::StopIteration
);
exc_constructor!(
    exc_overflow_error,
    pyre_object::interp_exceptions::ExcKind::OverflowError
);
exc_constructor!(
    exc_import_error,
    pyre_object::interp_exceptions::ExcKind::ImportError
);
exc_constructor!(
    exc_not_implemented_error,
    pyre_object::interp_exceptions::ExcKind::NotImplementedError
);
exc_constructor!(
    exc_assertion_error,
    pyre_object::interp_exceptions::ExcKind::AssertionError
);
exc_constructor!(
    exc_lookup_error,
    pyre_object::interp_exceptions::ExcKind::LookupError
);
exc_constructor!(
    exc_unicode_error,
    pyre_object::interp_exceptions::ExcKind::UnicodeError
);
exc_constructor!(
    exc_generator_exit,
    pyre_object::interp_exceptions::ExcKind::GeneratorExit
);
exc_constructor!(
    exc_system_exit,
    pyre_object::interp_exceptions::ExcKind::SystemExit
);
exc_constructor!(
    exc_recursion_error,
    pyre_object::interp_exceptions::ExcKind::RecursionError
);
exc_constructor!(
    exc_memory_error,
    pyre_object::interp_exceptions::ExcKind::MemoryError
);
exc_constructor!(
    exc_reference_error,
    pyre_object::interp_exceptions::ExcKind::ReferenceError
);
exc_constructor!(
    exc_system_error,
    pyre_object::interp_exceptions::ExcKind::SystemError
);
exc_constructor!(
    exc_syntax_error,
    pyre_object::interp_exceptions::ExcKind::SyntaxError
);

/// `interp_exceptions.py:121-124 W_BaseException.descr_init` — store the
/// constructor positional arguments on `self.args_w`.  Installed as
/// `BaseException.__init__` so the type-call `__new__` ⇒ `__init__`
/// protocol re-stamps `args` to the values forwarded by a subclass's
/// `super().__init__(*args)`, instead of leaving the full original
/// argument list captured by `__new__`.  `args[0]` is `self`.
///
/// `descr_init`'s `(self, args_w)` interp2app signature is positional-only,
/// so the argument matcher rejects any keyword with "takes no keyword
/// arguments".  pyre's flat builtin ABI has no signature to enforce that,
/// so the keyword dict is policed here directly; the type name comes from
/// `self`, matching `_PyArg_NoKeywords(Py_TYPE(self)->tp_name, kwds)`.
fn exc_base_exception_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__init__() missing 1 required positional argument: 'self'")
    })?;
    let (positional, kwargs) = split_builtin_kwargs(&args[1..]);
    if let Some(dict) = kwargs {
        let has_keyword = unsafe { pyre_object::w_dict_str_entries(dict) }
            .iter()
            .any(|(key, _)| key != "__pyre_kw__");
        if has_keyword {
            let type_name = unsafe {
                match crate::typedef::r#type(w_self) {
                    Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
                    None => "BaseException".to_string(),
                }
            };
            return Err(crate::PyError::type_error(format!(
                "{type_name}() takes no keyword arguments"
            )));
        }
    }
    let args_list = pyre_object::w_list_new(positional.to_vec());
    unsafe { pyre_object::interp_exceptions::w_exception_set_args(w_self, args_list) };
    Ok(pyre_object::w_none())
}

/// `interp_exceptions.py:551-652 W_OSError._parse_init_args` + `_init_error`.
/// A 2..=5 positional-argument call fills the `errno` / `strerror` /
/// `filename` / `filename2` slots; when a filename is present it is
/// dropped from `args_w` (`self.args_w = [w_errno, w_strerror]`, line
/// 652) for pickle / repr compatibility.  The winerror argument (idx 3,
/// Windows-only) and the `BlockingIOError.written` special-case are not
/// modelled.  `kind` is `OSError` for the base type and `FileNotFoundError`
/// for that dedicated kind; every other OSError subclass routes here as
/// `OSError` with its `w_class` retagged by `exc_new_wrapper!`.
fn os_error_build(
    kind: pyre_object::interp_exceptions::ExcKind,
    args: &[PyObjectRef],
) -> PyObjectRef {
    use pyre_object::interp_exceptions;
    let exc = if args.len() == 1 && unsafe { pyre_object::is_str(args[0]) } {
        let w = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
        interp_exceptions::w_exception_new_wtf8(kind, w)
    } else {
        let msg: String = if args.is_empty() {
            String::new()
        } else if args.len() == 1 {
            // exception construction is non-raising machinery, per the F7
            // display policy; a raising __str__/__repr__ on the args degrades
            // to the empty string rather than propagating.
            unsafe { crate::display::py_str(args[0]) }.unwrap_or_default()
        } else {
            let parts: Vec<String> = args
                .iter()
                .map(|&a| unsafe { crate::display::py_repr(a) }.unwrap_or_default())
                .collect();
            format!("({})", parts.join(", "))
        };
        interp_exceptions::w_exception_new(kind, &msg)
    };
    // Seed `args_w` so a deferred-init instance (`_use_init`, no `__new__`
    // slot fill) still reports the empty tuple until `__init__` runs.
    let args_list = pyre_object::w_list_new(args.to_vec());
    unsafe { interp_exceptions::w_exception_set_args(exc, args_list) };
    exc
}

/// `Py_IS_TYPE(self, BlockingIOError)` — the exact-type test `_init_error`
/// uses to read a numeric third argument as `characters_written` rather than
/// a filename.
fn exc_is_blocking_io_error(exc: PyObjectRef) -> bool {
    matches!(
        (crate::typedef::r#type(exc), lookup_exc_class("BlockingIOError")),
        (Some(c), Some(b)) if std::ptr::eq(c, b)
    )
}

/// `_parse_init_args` + `_init_error` slot assignment shared by `__new__`
/// (after the kind-aware allocation and `w_class` retag) and `__init__`
/// (re-stamping an already-allocated `self`).  Sets `args_w` and, for a 2..=5
/// argument call, the errno/strerror/filename/filename2 slots — dropping the
/// filename from `args_w` per `_init_error` (line 652).  Must run after the
/// `w_class` retag so the `BlockingIOError` numeric-filename special-case can
/// see the resolved class.
fn os_error_fill_slots(exc: PyObjectRef, args: &[PyObjectRef]) {
    use pyre_object::interp_exceptions;
    let args_list = pyre_object::w_list_new(args.to_vec());
    unsafe { interp_exceptions::w_exception_set_args(exc, args_list) };
    // `_parse_init_args`: only a 2..=5 argument call carries
    // errno/strerror (and optionally filename/filename2).
    let n = args.len();
    if (2..=5).contains(&n) {
        unsafe {
            interp_exceptions::w_exception_set_errno(exc, args[0]);
            interp_exceptions::w_exception_set_strerror(exc, args[1]);
            // idx 2 = filename, idx 3 = winerror (ignored off Windows),
            // idx 4 = filename2.
            let w_filename = args.get(2).copied().filter(|&f| !pyre_object::is_none(f));
            if let Some(fname) = w_filename {
                // `_init_error` line 636-643: for an exact `BlockingIOError`, a
                // numeric third argument is `characters_written`, not a
                // filename — it stays in `args_w` and the tuple is not trimmed.
                if exc_is_blocking_io_error(exc) && pyre_object::is_int(fname) {
                    return;
                }
                interp_exceptions::w_exception_set_filename(exc, fname);
                if let Some(f2) = args.get(4).copied().filter(|&f| !pyre_object::is_none(f)) {
                    interp_exceptions::w_exception_set_filename2(exc, f2);
                }
                // `_init_error`: filename is removed from the args tuple.
                let rebind = pyre_object::w_list_new(vec![args[0], args[1]]);
                interp_exceptions::w_exception_set_args(exc, rebind);
            }
        }
    }
}

/// `ESHUTDOWN` is a POSIX errno absent from the MSVC CRT, so the
/// `BrokenPipeError` mapping is gated on it being defined (`#ifdef ESHUTDOWN`).
#[cfg(unix)]
fn errno_is_eshutdown(e: i32) -> bool {
    e == libc::ESHUTDOWN
}
/// wasm32 has no libc errnos; match the darwin/BSD numeric value.
#[cfg(target_arch = "wasm32")]
fn errno_is_eshutdown(e: i32) -> bool {
    e == 58
}
#[cfg(all(not(unix), not(target_arch = "wasm32")))]
fn errno_is_eshutdown(_e: i32) -> bool {
    false
}

/// darwin/BSD numeric errnos for wasm32, which has no libc errno constants.
/// Kept in sync with the `errno` module's host_env-off fallback so the errno →
/// OSError-subclass remap selects the subclass a given `errno.X` value implies.
#[cfg(target_arch = "wasm32")]
mod wasm_errno {
    pub const EAGAIN: i32 = 35;
    pub const EWOULDBLOCK: i32 = 35;
    pub const EINPROGRESS: i32 = 36;
    pub const EALREADY: i32 = 37;
    pub const EPIPE: i32 = 32;
    pub const ECHILD: i32 = 10;
    pub const ECONNABORTED: i32 = 53;
    pub const ECONNREFUSED: i32 = 61;
    pub const ECONNRESET: i32 = 54;
    pub const EEXIST: i32 = 17;
    pub const ENOENT: i32 = 2;
    pub const EISDIR: i32 = 21;
    pub const ENOTDIR: i32 = 20;
    pub const EINTR: i32 = 4;
    pub const EACCES: i32 = 13;
    pub const EPERM: i32 = 1;
    pub const ESRCH: i32 = 3;
    pub const ETIMEDOUT: i32 = 60;
}

/// `interp_exceptions.py:1207-1227 ERRNO_MAP` — the OSError subclass the
/// exact `OSError` constructor selects for a recognised errno, by
/// registered class name.  Returns `None` for an unmapped errno.
fn os_error_errno_subclass(errno: i64) -> Option<&'static str> {
    // `ESHUTDOWN` is sourced through `errno_is_eshutdown` (MSVC CRT lacks it);
    // the rest come from `libc`, or the darwin/BSD `wasm_errno` table on wasm32.
    #[cfg(not(target_arch = "wasm32"))]
    use libc::{
        EACCES, EAGAIN, EALREADY, ECHILD, ECONNABORTED, ECONNREFUSED, ECONNRESET, EEXIST,
        EINPROGRESS, EINTR, EISDIR, ENOENT, ENOTDIR, EPERM, EPIPE, ESRCH, ETIMEDOUT, EWOULDBLOCK,
    };
    #[cfg(target_arch = "wasm32")]
    use wasm_errno::*;

    let Ok(e) = i32::try_from(errno) else {
        return None;
    };
    let name = if e == EAGAIN || e == EALREADY || e == EINPROGRESS || e == EWOULDBLOCK {
        "BlockingIOError"
    } else if e == EPIPE || errno_is_eshutdown(e) {
        "BrokenPipeError"
    } else if e == ECHILD {
        "ChildProcessError"
    } else if e == ECONNABORTED {
        "ConnectionAbortedError"
    } else if e == ECONNREFUSED {
        "ConnectionRefusedError"
    } else if e == ECONNRESET {
        "ConnectionResetError"
    } else if e == EEXIST {
        "FileExistsError"
    } else if e == ENOENT {
        "FileNotFoundError"
    } else if e == EISDIR {
        "IsADirectoryError"
    } else if e == ENOTDIR {
        "NotADirectoryError"
    } else if e == EINTR {
        "InterruptedError"
    } else if e == EACCES || e == EPERM {
        "PermissionError"
    } else if e == ESRCH {
        "ProcessLookupError"
    } else if e == ETIMEDOUT {
        "TimeoutError"
    } else {
        return None;
    };
    Some(name)
}

/// `_parse_init_args` yields an errno only for a 2..=5 argument call whose
/// first argument is an int; map that errno to its OSError subclass name.
fn os_error_errno_subclass_for(args: &[PyObjectRef]) -> Option<&'static str> {
    if !(2..=5).contains(&args.len()) || !unsafe { pyre_object::is_int(args[0]) } {
        return None;
    }
    os_error_errno_subclass(unsafe { pyre_object::w_int_get_value(args[0]) })
}

/// `W_OSError._use_init` (`interp_exceptions.py:531-549`): the slots are
/// already filled by `__new__`, so `descr_init` does extra work only when
/// the instance's type defines its own `__init__` while keeping
/// `OSError.__new__`.  Returns `False` for the exact `OSError` type, for
/// builtin subclasses, and for user subclasses that do not override
/// `__init__` — every other case routes `descr_init` to a no-op
/// (`descr_init` early-return at line 618-620).
///
/// The base `__init__` / `__new__` are read from the registered runtime
/// `OSError` class object (`lookup_exc_class`), not the static layout
/// `PyType`, which carries no namespace dict; comparing them by identity
/// against the instance type's MRO-resolved entries detects an override.
fn os_error_use_init(w_self: PyObjectRef) -> bool {
    let Some(w_type) = crate::typedef::r#type(w_self) else {
        return false;
    };
    os_error_type_use_init(w_type)
}

/// `_use_init` keyed on the type object directly (the `__new__` half has the
/// subtype, not yet an instance).
fn os_error_type_use_init(w_type: PyObjectRef) -> bool {
    let Some(w_os_error) = lookup_exc_class("OSError") else {
        return false;
    };
    // The exact OSError type is never `_use_init` (line 542-543).
    if std::ptr::eq(w_type, w_os_error) {
        return false;
    }
    let self_init = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__init__") };
    let base_init = unsafe { crate::baseobjspace::lookup_in_type(w_os_error, "__init__") };
    let overrides_init = match (self_init, base_init) {
        (Some(a), Some(b)) => !std::ptr::eq(a, b),
        (Some(_), None) => true,
        _ => false,
    };
    if !overrides_init {
        return false;
    }
    // `_use_init` also requires `__new__` to be the inherited `OSError.__new__`
    // (line 546-547): a subclass overriding `__new__` keeps `descr_init` a
    // no-op even when it also defines `__init__`.  Every builtin OSError type
    // carries its own `__new__` Function wrapping the shared native
    // constructor, so an identity comparison against `OSError.__new__` is
    // unreliable; instead detect a *Python-level* override — a user
    // `def __new__` resolves to a Function backed by a code object, while the
    // inherited family `__new__` is a builtin function backed by a `BuiltinCode`.
    let Some(self_new) = (unsafe { crate::baseobjspace::lookup_in_type(w_type, "__new__") }) else {
        return false;
    };
    unsafe {
        crate::function::is_function(self_new)
            && crate::gateway::is_builtin_code(
                crate::function::function_get_code(self_new) as PyObjectRef
            )
    }
}

/// `_PyArg_NoKeywords(type_name, kwds)` message — `OSError` and its family
/// take only positional arguments.
fn os_error_no_keywords_error(w_type: Option<PyObjectRef>) -> crate::PyError {
    let type_name = match w_type {
        Some(t) => unsafe { pyre_object::w_type_get_name(t) }.to_string(),
        None => "OSError".to_string(),
    };
    crate::PyError::type_error(format!("{type_name}() takes no keyword arguments"))
}

/// `interp_exceptions.py:551-652 W_OSError._parse_init_args` as the
/// `descr_init` half: re-stamp the errno/strerror/filename slots and the
/// trimmed `args_w` onto an already-allocated `self`.  Installed as
/// `OSError.__init__` so the inherited `W_BaseException.descr_init` (which
/// would overwrite `args_w` with the full, untrimmed argument list) does
/// not run for the OSError family.  `args[0]` is `self`.
fn exc_os_error_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__init__() missing 1 required positional argument: 'self'")
    })?;
    // `descr_init` early-return (line 618-620): when the type does not
    // override `OSError.__init__`, every slot is already filled by `__new__`
    // (`os_error_fill_slots`), so re-stamping here would corrupt the args/errno
    // that construction already set.
    if !os_error_use_init(w_self) {
        return Ok(pyre_object::w_none());
    }
    let (positional, kwargs) = split_builtin_kwargs(&args[1..]);
    // `descr_init` line 621-623: OSError takes no keyword arguments.
    if has_real_kwargs(kwargs) {
        return Err(os_error_no_keywords_error(crate::typedef::r#type(w_self)));
    }
    os_error_fill_slots(w_self, positional);
    Ok(pyre_object::w_none())
}

/// `interp_exceptions.py:233-237 BaseException.descr_reduce` —
/// `(cls, args[, dict])`: a 2-tuple normally, a 3-tuple when the instance
/// dict is non-empty.  Inherited by every builtin exception class through
/// the MRO, so a subclass pickles via its own class object.
fn base_exception_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__reduce__() missing 1 required positional argument: 'self'")
    })?;
    let cls = crate::typedef::r#type(w_self)
        .unwrap_or_else(|| crate::baseobjspace::exception_getclass(w_self));
    let w_args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(w_self) };
    let w_dict = unsafe { pyre_object::interp_exceptions::w_exception_peek_dict(w_self) };
    if !w_dict.is_null() && unsafe { pyre_object::w_dict_len(w_dict) } > 0 {
        Ok(pyre_object::w_tuple_new(vec![cls, w_args, w_dict]))
    } else {
        Ok(pyre_object::w_tuple_new(vec![cls, w_args]))
    }
}

/// `interp_exceptions.py:239-241 BaseException.descr_setstate` —
/// `self.getdict(space).update(state)`.
fn base_exception_setstate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__setstate__() missing 1 required positional argument: 'self'")
    })?;
    let w_state = *args.get(1).ok_or_else(|| {
        crate::PyError::type_error("__setstate__() missing 1 required positional argument: 'state'")
    })?;
    let w_olddict = unsafe { pyre_object::interp_exceptions::w_exception_getdict(w_self) };
    if crate::baseobjspace::call_method(w_olddict, "update", &[w_state]).is_null() {
        if let Some(e) = crate::call::take_call_error() {
            return Err(e);
        }
    }
    Ok(pyre_object::w_none())
}

/// `interp_exceptions.py:379-391 W_ImportError.descr_reduce` plus the
/// 3.14 `name_from` field: the reduce-state dict carries
/// `name`/`path`/`name_from` (each only when set), merged over any
/// instance-dict entries.
fn import_error_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::interp_exceptions;
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__reduce__() missing 1 required positional argument: 'self'")
    })?;
    let cls = crate::typedef::r#type(w_self)
        .unwrap_or_else(|| crate::baseobjspace::exception_getclass(w_self));
    let w_args = unsafe { interp_exceptions::w_exception_get_args(w_self) };
    let stored = unsafe { interp_exceptions::w_exception_peek_dict(w_self) };
    let w_dict = if !stored.is_null() && unsafe { pyre_object::w_dict_len(stored) } > 0 {
        let copy = crate::baseobjspace::call_method(stored, "copy", &[]);
        if copy.is_null() {
            if let Some(e) = crate::call::take_call_error() {
                return Err(e);
            }
        }
        copy
    } else {
        pyre_object::w_dict_new()
    };
    for (key, w_value) in [
        ("name", unsafe {
            interp_exceptions::w_exception_get_name(w_self)
        }),
        ("path", unsafe {
            interp_exceptions::w_exception_get_import_path(w_self)
        }),
        ("name_from", unsafe {
            interp_exceptions::w_exception_get_import_name_from(w_self)
        }),
    ] {
        if !w_value.is_null() && !unsafe { pyre_object::is_none(w_value) } {
            unsafe { pyre_object::w_dict_setitem_str(w_dict, key, w_value) };
        }
    }
    if unsafe { pyre_object::w_dict_len(w_dict) } > 0 {
        Ok(pyre_object::w_tuple_new(vec![cls, w_args, w_dict]))
    } else {
        Ok(pyre_object::w_tuple_new(vec![cls, w_args]))
    }
}

/// `interp_exceptions.py:393-397 W_ImportError.descr_setstate` plus
/// `name_from`: pop `name`/`path`/`name_from` into their slots, then update
/// the instance dict with whatever remains.
fn import_error_setstate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::interp_exceptions;
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__setstate__() missing 1 required positional argument: 'self'")
    })?;
    let w_state = *args.get(1).ok_or_else(|| {
        crate::PyError::type_error("__setstate__() missing 1 required positional argument: 'state'")
    })?;
    type ExcSetter = unsafe fn(PyObjectRef, PyObjectRef);
    for (key, set) in [
        ("name", interp_exceptions::w_exception_set_name as ExcSetter),
        ("path", interp_exceptions::w_exception_set_import_path),
        (
            "name_from",
            interp_exceptions::w_exception_set_import_name_from,
        ),
    ] {
        let popped = crate::baseobjspace::call_method(
            w_state,
            "pop",
            &[pyre_object::w_str_new(key), pyre_object::w_none()],
        );
        if popped.is_null() {
            if let Some(e) = crate::call::take_call_error() {
                return Err(e);
            }
        }
        unsafe { set(w_self, popped) };
    }
    let w_olddict = unsafe { interp_exceptions::w_exception_getdict(w_self) };
    if crate::baseobjspace::call_method(w_olddict, "update", &[w_state]).is_null() {
        if let Some(e) = crate::call::take_call_error() {
            return Err(e);
        }
    }
    Ok(pyre_object::w_none())
}

/// `interp_exceptions.py:655-665 W_OSError.descr_reduce` — re-append the
/// `filename`/`filename2` that `os_error_fill_slots` stripped from `args_w`
/// so the reconstruction call receives the full positional list.  OSError
/// has no own `__setstate__`; it inherits `BaseException.__setstate__`.
fn os_error_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::interp_exceptions;
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__reduce__() missing 1 required positional argument: 'self'")
    })?;
    let cls = crate::typedef::r#type(w_self)
        .unwrap_or_else(|| crate::baseobjspace::exception_getclass(w_self));
    let w_args = unsafe { interp_exceptions::w_exception_get_args(w_self) };
    let n = unsafe { pyre_object::w_tuple_len(w_args) };
    let mut items: Vec<PyObjectRef> = (0..n as i64)
        .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(w_args, i) })
        .collect();
    let w_filename = unsafe { interp_exceptions::w_exception_get_filename(w_self) };
    if !w_filename.is_null() && !unsafe { pyre_object::is_none(w_filename) } {
        items.push(w_filename);
        let w_filename2 = unsafe { interp_exceptions::w_exception_get_filename2(w_self) };
        if !w_filename2.is_null() && !unsafe { pyre_object::is_none(w_filename2) } {
            items.push(pyre_object::w_none());
            items.push(w_filename2);
        }
    }
    let w_full_args = pyre_object::w_tuple_new(items);
    let w_dict = unsafe { interp_exceptions::w_exception_peek_dict(w_self) };
    if !w_dict.is_null() && unsafe { pyre_object::w_dict_len(w_dict) } > 0 {
        Ok(pyre_object::w_tuple_new(vec![cls, w_full_args, w_dict]))
    } else {
        Ok(pyre_object::w_tuple_new(vec![cls, w_full_args]))
    }
}

/// `ImportError.__init__` — consume the `name` / `path` / `name_from`
/// keyword arguments into their typed slots and store the single
/// positional argument as `msg`, then pass the positional arguments to
/// `W_BaseException.descr_init` (`args_w`).  Every slot is re-stamped on
/// each call (kwarg value or `None`; `msg` the lone positional else
/// `None`) so a repeated `__init__` resets stale values.  Any other
/// keyword raises `ImportError() got an unexpected keyword argument`
/// (the name hard-codes `ImportError` even for `ModuleNotFoundError`).
/// Installed as `ImportError.__init__` and inherited by
/// `ModuleNotFoundError`.  `args[0]` is `self`.
fn exc_import_error_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::interp_exceptions;
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__init__() missing 1 required positional argument: 'self'")
    })?;
    let (positional, kwargs) = split_builtin_kwargs(&args[1..]);
    kwarg_reject_unknown(kwargs, &["name", "path", "name_from"], "ImportError")?;
    let w_name = kwarg_get(kwargs, "name").unwrap_or_else(pyre_object::w_none);
    let w_path = kwarg_get(kwargs, "path").unwrap_or_else(pyre_object::w_none);
    let w_name_from = kwarg_get(kwargs, "name_from").unwrap_or_else(pyre_object::w_none);
    let w_msg = if positional.len() == 1 {
        positional[0]
    } else {
        pyre_object::w_none()
    };
    unsafe {
        // Unconditional re-stamp so a repeated `__init__` resets stale slots.
        interp_exceptions::w_exception_set_name(w_self, w_name);
        interp_exceptions::w_exception_set_import_path(w_self, w_path);
        interp_exceptions::w_exception_set_import_name_from(w_self, w_name_from);
        interp_exceptions::w_exception_set_import_msg(w_self, w_msg);
        // Only the positional arguments reach `args_w`.
        let args_list = pyre_object::w_list_new(positional.to_vec());
        interp_exceptions::w_exception_set_args(w_self, args_list);
    }
    Ok(pyre_object::w_none())
}

/// `W_NameError.descr_init` (Python 3.10+) — consume the `name` keyword
/// into the shared name slot and pass the positional arguments to
/// `W_BaseException.descr_init`.  Any other keyword raises
/// `NameError() got an unexpected keyword argument`.  Installed as
/// `NameError.__init__`.  `args[0]` is `self`.
fn exc_name_error_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::interp_exceptions;
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__init__() missing 1 required positional argument: 'self'")
    })?;
    let (positional, kwargs) = split_builtin_kwargs(&args[1..]);
    kwarg_reject_unknown(kwargs, &["name"], "NameError")?;
    let w_name = kwarg_get(kwargs, "name").unwrap_or_else(pyre_object::w_none);
    unsafe {
        // `self.w_name = w_name` (WrappedDefault(None)) — unconditional
        // re-stamp so a repeated `__init__` resets a stale name.
        interp_exceptions::w_exception_set_name(w_self, w_name);
        let args_list = pyre_object::w_list_new(positional.to_vec());
        interp_exceptions::w_exception_set_args(w_self, args_list);
    }
    Ok(pyre_object::w_none())
}

/// `W_AttributeError.descr_init` (Python 3.10+) — consume the `name` and
/// `obj` keywords into their slots and pass the positional arguments to
/// `W_BaseException.descr_init`.  Any other keyword raises
/// `AttributeError() got an unexpected keyword argument`.  Installed as
/// `AttributeError.__init__`.  `args[0]` is `self`.
fn exc_attribute_error_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::interp_exceptions;
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__init__() missing 1 required positional argument: 'self'")
    })?;
    let (positional, kwargs) = split_builtin_kwargs(&args[1..]);
    kwarg_reject_unknown(kwargs, &["name", "obj"], "AttributeError")?;
    let w_name = kwarg_get(kwargs, "name").unwrap_or_else(pyre_object::w_none);
    let w_obj = kwarg_get(kwargs, "obj").unwrap_or_else(pyre_object::w_none);
    unsafe {
        // `self.w_name = w_name` / `self.w_obj = w_obj` (WrappedDefault(None))
        // — unconditional re-stamp so a repeated `__init__` resets stale slots.
        interp_exceptions::w_exception_set_name(w_self, w_name);
        interp_exceptions::w_exception_set_attr_obj(w_self, w_obj);
        let args_list = pyre_object::w_list_new(positional.to_vec());
        interp_exceptions::w_exception_set_args(w_self, args_list);
    }
    Ok(pyre_object::w_none())
}

fn exc_os_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(os_error_build(
        pyre_object::interp_exceptions::ExcKind::OSError,
        args,
    ))
}

fn exc_file_not_found_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(os_error_build(
        pyre_object::interp_exceptions::ExcKind::FileNotFoundError,
        args,
    ))
}

/// `interp_exceptions.py:583-614 W_OSError.descr_new` — the `__new__` shared
/// by `OSError` and every errno subclass.  For the exact `OSError` type it
/// rejects keyword arguments (line 591-593) and remaps a recognised errno to
/// the matching subclass (line 596-608), so `OSError(ENOENT, ...)`
/// constructs a `FileNotFoundError`.  A subclass call keeps its own class,
/// `w_class`-retagged like `exc_new_wrapper!`.  `ctor` builds the base object
/// with the called type's `ExcKind` (`OSError` for the base type and every
/// retagged subclass, `FileNotFoundError` for that dedicated kind).
fn os_error_family_new(
    args: &[PyObjectRef],
    ctor: impl Fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
) -> Result<PyObjectRef, crate::PyError> {
    let cls = args.first().copied();
    let rest: &[PyObjectRef] = if args.is_empty() { args } else { &args[1..] };
    let is_exact_os_error = matches!(
        (cls, lookup_exc_class("OSError")),
        (Some(c), Some(w_os)) if std::ptr::eq(c, w_os)
    );
    let use_init = matches!(cls, Some(c) if os_error_type_use_init(c));
    let (positional, kwargs) = split_builtin_kwargs(rest);
    // `descr_new` line 590-593: only when `_use_init` is false (exact OSError
    // and builtin subclasses) does `__new__` parse the args and reject
    // keywords; a user subclass overriding `__init__` while keeping the
    // inherited `__new__` defers both to `__init__`.
    if !use_init && has_real_kwargs(kwargs) {
        return Err(os_error_no_keywords_error(cls));
    }
    // When `_use_init`, `__new__` allocates without parsing the args — the
    // errno/strerror/filename slots and `args_w` stay unset for `__init__` to
    // fill (line 608-611).  Otherwise `__new__` parses them itself.
    let exc = ctor(if use_init { &[] } else { positional })?;
    // Only the exact OSError type remaps the errno to a subclass; resolve the
    // retag target (subclass on a recognised errno, else the called class).
    let w_target = if is_exact_os_error {
        os_error_errno_subclass_for(positional)
            .and_then(lookup_exc_class)
            .or(cls)
    } else {
        cls
    };
    if let Some(w_target) = w_target {
        unsafe {
            (*(exc as *mut pyre_object::PyObject)).w_class = w_target;
        }
    }
    // Fill the slots after the retag so `os_error_fill_slots` can see the
    // resolved class (the `BlockingIOError` numeric-filename special-case).
    if !use_init {
        os_error_fill_slots(exc, positional);
    }
    Ok(exc)
}

pub(crate) fn exc_os_error_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    os_error_family_new(args, exc_os_error)
}

pub(crate) fn exc_file_not_found_error_new(
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    os_error_family_new(args, exc_file_not_found_error)
}

/// `pypy/module/exceptions/interp_exceptions.py:274-284 _new`'s shape
/// applied to UnicodeTranslateError: allocate the W_BaseException
/// and store the raw constructor args verbatim into `args_w`.  PyPy's
/// `_new` runs no per-arg validation — type checks live in
/// `descr_init` (line 433-445) and only fire when `__init__` is
/// invoked by the type-call protocol after `__new__`.  Pyre's
/// type-call (call.rs:982-996) routes through that same `__new__` ⇒
/// `__init__` sequence, so `__new__` here can stay validation-free.
fn exc_unicode_translate_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = pyre_object::interp_exceptions::w_exception_new(
        pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError,
        "",
    );
    let args_list = pyre_object::w_list_new(args.to_vec());
    unsafe { pyre_object::interp_exceptions::w_exception_set_args(exc, args_list) };
    Ok(exc)
}

/// `pypy/module/exceptions/interp_exceptions.py:274-284 _new` shape
/// for UnicodeDecodeError — allocation + raw args_w only.  Encoding,
/// object, start/end/reason type checks happen in `descr_init` at
/// `:1041-1059`.
fn exc_unicode_decode_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = pyre_object::interp_exceptions::w_exception_new(
        pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError,
        "",
    );
    let args_list = pyre_object::w_list_new(args.to_vec());
    unsafe { pyre_object::interp_exceptions::w_exception_set_args(exc, args_list) };
    Ok(exc)
}

/// `pypy/module/exceptions/interp_exceptions.py:274-284 _new` shape
/// for UnicodeEncodeError — allocation + raw args_w only.
fn exc_unicode_encode_error(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let exc = pyre_object::interp_exceptions::w_exception_new(
        pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError,
        "",
    );
    let args_list = pyre_object::w_list_new(args.to_vec());
    unsafe { pyre_object::interp_exceptions::w_exception_set_args(exc, args_list) };
    Ok(exc)
}

/// `pypy/module/exceptions/interp_exceptions.py:433-445
/// W_UnicodeTranslateError.descr_init` —
///
/// ```python
/// def descr_init(self, space, w_object, w_start, w_end, w_reason):
///     space.utf8_w(w_object); space.int_w(w_start); space.int_w(w_end)
///     space.realtext_w(w_reason)
///     self.w_object = w_object; self.w_start = w_start
///     self.w_end = w_end; self.w_reason = w_reason
///     W_BaseException.descr_init(self, space,
///         [w_object, w_start, w_end, w_reason])
/// ```
///
/// Typechecks go through subclass-accepting `isinstance_*_w` helpers
/// to match PyPy's `space.utf8_w` / `space.int_w` / `space.realtext_w`
/// behavior — `class MyStr(str): pass` and `class MyInt(int): pass`
/// instances satisfy the check.  PyPy's `*_w` helpers raise
/// `TypeError` from the typechecks; pyre mirrors via
/// `PyError::type_error`.
fn exc_unicode_translate_error_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 5 {
        // first arg is `self`; PyPy reports argcount excluding `self`.
        return Err(crate::PyError::type_error(
            "function takes exactly 4 arguments",
        ));
    }
    let w_self = args[0];
    let w_object = args[1];
    let w_start = args[2];
    let w_end = args[3];
    let w_reason = args[4];
    unsafe {
        if !crate::baseobjspace::isinstance_str_w(w_object) {
            return Err(crate::PyError::type_error(
                "argument 1 must be str, not other",
            ));
        }
        if !crate::baseobjspace::isinstance_int_w(w_start) {
            return Err(crate::PyError::type_error("an integer is required"));
        }
        if !crate::baseobjspace::isinstance_int_w(w_end) {
            return Err(crate::PyError::type_error("an integer is required"));
        }
        if !crate::baseobjspace::isinstance_str_w(w_reason) {
            return Err(crate::PyError::type_error(
                "argument 4 must be str, not other",
            ));
        }
        pyre_object::interp_exceptions::w_exception_set_object(w_self, w_object);
        pyre_object::interp_exceptions::w_exception_set_start(w_self, w_start);
        pyre_object::interp_exceptions::w_exception_set_end(w_self, w_end);
        pyre_object::interp_exceptions::w_exception_set_reason(w_self, w_reason);
        // `W_BaseException.descr_init(self, space, [w_object, w_start,
        // w_end, w_reason])` → `self.args_w = args_w`.  The
        // `W_BaseException.args_w` slot already carries the same
        // tuple shape from `__new__`, so we re-stamp it from the
        // bound init args here for parity with PyPy line 444-445.
        let args_list = pyre_object::w_list_new(vec![w_object, w_start, w_end, w_reason]);
        pyre_object::interp_exceptions::w_exception_set_args(w_self, args_list);
    }
    Ok(pyre_object::w_none())
}

/// `pypy/module/exceptions/interp_exceptions.py:1041-1059
/// W_UnicodeDecodeError.descr_init` — `(w_encoding, w_object, w_start,
/// w_end, w_reason)`.  `w_object` may be `bytearray`; PyPy coerces it
/// via `space.newbytes(space.charbuf_w(w_object))` before storing.
/// Pyre accepts either `bytes` or `bytearray` and stores the coerced
/// `bytes` so reads of `e.object` round-trip as `bytes` per PyPy.
fn exc_unicode_decode_error_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 6 {
        return Err(crate::PyError::type_error(
            "function takes exactly 5 arguments",
        ));
    }
    let w_self = args[0];
    let w_encoding = args[1];
    let w_object_in = args[2];
    let w_start = args[3];
    let w_end = args[4];
    let w_reason = args[5];
    unsafe {
        if !crate::baseobjspace::isinstance_str_w(w_encoding) {
            return Err(crate::PyError::type_error(
                "argument 1 must be str, not other",
            ));
        }
        if !crate::baseobjspace::isinstance_bytes_like_w(w_object_in) {
            return Err(crate::PyError::type_error(
                "argument 2 must be bytes-like, not other",
            ));
        }
        if !crate::baseobjspace::isinstance_int_w(w_start) {
            return Err(crate::PyError::type_error("an integer is required"));
        }
        if !crate::baseobjspace::isinstance_int_w(w_end) {
            return Err(crate::PyError::type_error("an integer is required"));
        }
        if !crate::baseobjspace::isinstance_str_w(w_reason) {
            return Err(crate::PyError::type_error(
                "argument 5 must be str, not other",
            ));
        }
        // `interp_exceptions.py:1043-1046` — `space.charbuf_w` /
        // `space.newbytes` coerce buffer-protocol producers
        // (`bytearray`, exact `bytes`, and `bytes` subclasses) to a
        // canonical `bytes`.  Exact `bytes` already IS the canonical
        // shape; bytearray and `bytes` subclasses (`class
        // MyBytes(bytes): pass`) are funneled through
        // `w_bytes_from_bytes(...)` so `e.object` always holds a
        // canonical `bytes` regardless of the input shape.
        //
        // Codex P1 (PR #89 round 2): `bytes_like_data` dispatches via
        // exact-type pointer identity (`is_bytes` → `py_type_check`)
        // and silently reads the operand through the `W_BytearrayObject`
        // layout for any non-exact-bytes input — including `bytes`
        // subclasses, whose underlying struct IS `W_BytesObject`.
        // `isinstance_w(obj, bytes)` is subclass-aware, so once exact
        // `bytes` is filtered the remaining branches split cleanly:
        // bytes subclass → `w_bytes_data` (`W_BytesObject` layout);
        // bytearray (exact or subclass) → `w_bytearray_data`
        // (`W_BytearrayObject` layout).
        let w_object = if pyre_object::is_bytes(w_object_in) {
            w_object_in
        } else {
            let bytes_type = crate::typedef::gettypefor(&pyre_object::BYTES_TYPE);
            let inherits_bytes =
                bytes_type.is_some_and(|bt| crate::baseobjspace::isinstance_w(w_object_in, bt));
            let data = if inherits_bytes {
                pyre_object::bytesobject::w_bytes_data(w_object_in)
            } else {
                pyre_object::bytearrayobject::w_bytearray_data(w_object_in)
            };
            pyre_object::w_bytes_from_bytes(data)
        };
        pyre_object::interp_exceptions::w_exception_set_encoding(w_self, w_encoding);
        pyre_object::interp_exceptions::w_exception_set_object(w_self, w_object);
        pyre_object::interp_exceptions::w_exception_set_start(w_self, w_start);
        pyre_object::interp_exceptions::w_exception_set_end(w_self, w_end);
        pyre_object::interp_exceptions::w_exception_set_reason(w_self, w_reason);
        // `interp_exceptions.py:1058-1059` — the args list passed to
        // `W_BaseException.descr_init` is the un-coerced
        // `[w_encoding, w_object, w_start, w_end, w_reason]`, so PyPy
        // preserves the original `bytearray` in `e.args[1]` while
        // storing the coerced `bytes` in `e.object`.
        let args_list =
            pyre_object::w_list_new(vec![w_encoding, w_object_in, w_start, w_end, w_reason]);
        pyre_object::interp_exceptions::w_exception_set_args(w_self, args_list);
    }
    Ok(pyre_object::w_none())
}

/// `pypy/module/exceptions/interp_exceptions.py:1159-1173
/// W_UnicodeEncodeError.descr_init` — `(w_encoding, w_object, w_start,
/// w_end, w_reason)`.  Encoding errors require `w_object` to be a
/// `str` (`space.realutf8_w`).
fn exc_unicode_encode_error_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 6 {
        return Err(crate::PyError::type_error(
            "function takes exactly 5 arguments",
        ));
    }
    let w_self = args[0];
    let w_encoding = args[1];
    let w_object = args[2];
    let w_start = args[3];
    let w_end = args[4];
    let w_reason = args[5];
    unsafe {
        if !crate::baseobjspace::isinstance_str_w(w_encoding) {
            return Err(crate::PyError::type_error(
                "argument 1 must be str, not other",
            ));
        }
        if !crate::baseobjspace::isinstance_str_w(w_object) {
            return Err(crate::PyError::type_error(
                "argument 2 must be str, not other",
            ));
        }
        if !crate::baseobjspace::isinstance_int_w(w_start) {
            return Err(crate::PyError::type_error("an integer is required"));
        }
        if !crate::baseobjspace::isinstance_int_w(w_end) {
            return Err(crate::PyError::type_error("an integer is required"));
        }
        if !crate::baseobjspace::isinstance_str_w(w_reason) {
            return Err(crate::PyError::type_error(
                "argument 5 must be str, not other",
            ));
        }
        pyre_object::interp_exceptions::w_exception_set_encoding(w_self, w_encoding);
        pyre_object::interp_exceptions::w_exception_set_object(w_self, w_object);
        pyre_object::interp_exceptions::w_exception_set_start(w_self, w_start);
        pyre_object::interp_exceptions::w_exception_set_end(w_self, w_end);
        pyre_object::interp_exceptions::w_exception_set_reason(w_self, w_reason);
        let args_list =
            pyre_object::w_list_new(vec![w_encoding, w_object, w_start, w_end, w_reason]);
        pyre_object::interp_exceptions::w_exception_set_args(w_self, args_list);
    }
    Ok(pyre_object::w_none())
}

/// `cls.__new__` wrapper that strips `cls` and calls an exception constructor.
/// PyPy: each exception type's descr__new__ creates a W_<Kind>Object.
macro_rules! exc_new_wrapper {
    ($wrapper:ident, $ctor:ident) => {
        pub(crate) fn $wrapper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let cls = args.first().copied();
            let rest: &[PyObjectRef] = if args.is_empty() { args } else { &args[1..] };
            let exc = $ctor(rest)?;
            // Set the exception's w_class to the actual exception type (e.g. AssertionError)
            // so that `type(e) is AssertionError` holds and `except ExcType` via isinstance works.
            if let Some(cls) = cls {
                unsafe {
                    (*(exc as *mut pyre_object::PyObject)).w_class = cls;
                }
            }
            Ok(exc)
        }
    };
}

exc_new_wrapper!(exc_base_exception_new, exc_base_exception);
exc_new_wrapper!(exc_exception_new, exc_exception);
exc_new_wrapper!(exc_arithmetic_error_new, exc_arithmetic_error);
exc_new_wrapper!(exc_zero_division_new, exc_zero_division);
exc_new_wrapper!(exc_type_error_new, exc_type_error);
exc_new_wrapper!(exc_value_error_new, exc_value_error);
exc_new_wrapper!(exc_key_error_new, exc_key_error);
exc_new_wrapper!(exc_index_error_new, exc_index_error);
exc_new_wrapper!(exc_attribute_error_new, exc_attribute_error);
exc_new_wrapper!(exc_name_error_new, exc_name_error);
exc_new_wrapper!(exc_runtime_error_new, exc_runtime_error);
exc_new_wrapper!(exc_stop_iteration_new, exc_stop_iteration);
exc_new_wrapper!(exc_overflow_error_new, exc_overflow_error);
exc_new_wrapper!(exc_import_error_new, exc_import_error);
exc_new_wrapper!(exc_not_implemented_error_new, exc_not_implemented_error);
exc_new_wrapper!(exc_assertion_error_new, exc_assertion_error);
exc_new_wrapper!(exc_lookup_error_new, exc_lookup_error);
exc_new_wrapper!(exc_unicode_error_new, exc_unicode_error);
exc_new_wrapper!(exc_unicode_decode_error_new, exc_unicode_decode_error);
exc_new_wrapper!(exc_unicode_encode_error_new, exc_unicode_encode_error);
exc_new_wrapper!(exc_unicode_translate_error_new, exc_unicode_translate_error);
exc_new_wrapper!(exc_generator_exit_new, exc_generator_exit);
exc_new_wrapper!(exc_system_exit_new, exc_system_exit);
exc_new_wrapper!(exc_recursion_error_new, exc_recursion_error);
exc_new_wrapper!(exc_memory_error_new, exc_memory_error);
exc_new_wrapper!(exc_reference_error_new, exc_reference_error);
exc_new_wrapper!(exc_system_error_new, exc_system_error);
exc_new_wrapper!(exc_syntax_error_new, exc_syntax_error);

/// Build a builtin exception type with the given name, base, and __new__ wrapper.
pub(crate) fn make_exc_type(
    name: &'static str,
    new_fn: crate::gateway::BuiltinCodeFn,
    base: PyObjectRef,
) -> PyObjectRef {
    make_exc_type_with_init(name, new_fn, None, base)
}

/// Variant of `make_exc_type` that also installs a per-class `__init__`
/// descriptor.  Used for the three Unicode*Error subclasses whose PyPy
/// `descr_init` does typed slot stamping after `__new__`'s raw
/// `args_w` capture (`interp_exceptions.py:433-445`, `:1041-1059`,
/// `:1159-1173`).  Without this split, every direct
/// `UnicodeDecodeError.__new__(cls, *args)` call would inherit the
/// typechecking that PyPy keeps confined to `descr_init` — see
/// `_new` at `:274-284` (no per-arg validation).
fn make_exc_type_with_init(
    name: &'static str,
    new_fn: crate::gateway::BuiltinCodeFn,
    init_fn: Option<crate::gateway::BuiltinCodeFn>,
    base: PyObjectRef,
) -> PyObjectRef {
    let cls = crate::typedef::make_builtin_type_with_base(
        name,
        move |ns| {
            crate::dict_storage_store(ns, "__new__", make_builtin_function("__new__", new_fn));
            if let Some(init_fn) = init_fn {
                crate::dict_storage_store(
                    ns,
                    "__init__",
                    make_builtin_function("__init__", init_fn),
                );
            }
            // `pypy/module/exceptions/interp_exceptions.py:225-235`
            // `BaseException.with_traceback` — installed on every
            // builtin exception class so MRO lookup from a subclass
            // (`MyError.with_traceback`) hits the canonical method
            // even before user-level `class MyError(BaseException):`
            // metaclass walks BaseException's namespace.  PyPy adds
            // this to BaseException only; pyre's `make_exc_type`
            // wires it into every class because Pyre doesn't run
            // `BaseException.__init_subclass__` at builtin-bootstrap
            // time, so without per-class install `subclass.with_traceback`
            // raises AttributeError.
            if name == "BaseException" {
                crate::dict_storage_store(
                    ns,
                    "with_traceback",
                    make_builtin_function_with_arity(
                        "with_traceback",
                        |args| {
                            let w_self = *args.first().ok_or_else(|| {
                                crate::PyError::type_error(
                                    "with_traceback() missing 1 required positional argument: 'self'",
                                )
                            })?;
                            let w_tb = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                            if !w_self.is_null() && unsafe { pyre_object::is_exception(w_self) } {
                                // `interp_exceptions.py:213-219
                                // descr_settraceback` — only None or
                                // PyTraceback is accepted.
                                let value =
                                    if w_tb.is_null() || unsafe { pyre_object::is_none(w_tb) } {
                                        pyre_object::PY_NULL
                                    } else if unsafe { crate::pytraceback::is_pytraceback(w_tb) } {
                                        w_tb
                                    } else {
                                        return Err(crate::PyError::type_error(
                                            "__traceback__ must be a traceback or None",
                                        ));
                                    };
                                unsafe {
                                    pyre_object::interp_exceptions::w_exception_set_traceback(
                                        w_self, value,
                                    );
                                }
                            }
                            Ok(w_self)
                        },
                        2,
                    ),
                );
                // `interp_exceptions.py:236-247 BaseException.add_note`
                // (Python 3.11+ PEP 678).  Appends a string to
                // `self.__notes__`, allocating the list on first call.
                // The list lives in the exception's instance dict
                // (`W_BaseException.w_dict`), reached through the
                // setattr/getattr paths in baseobjspace.
                crate::dict_storage_store(
                    ns,
                    "add_note",
                    make_builtin_function_with_arity(
                        "add_note",
                        |args| {
                            let w_self = *args.first().ok_or_else(|| {
                                crate::PyError::type_error(
                                    "add_note() missing 1 required positional argument: 'self'",
                                )
                            })?;
                            let w_note = *args.get(1).ok_or_else(|| {
                                crate::PyError::type_error(
                                    "add_note() missing 1 required positional argument: 'note'",
                                )
                            })?;
                            // `interp_exceptions.py:238-239` — accept
                            // `str` and any `str` subclass
                            // (`isinstance_w(w_note, space.w_unicode)`).
                            if !unsafe { crate::baseobjspace::isinstance_str_w(w_note) } {
                                return Err(crate::PyError::type_error("note must be a string"));
                            }
                            // `interp_exceptions.py:240-254` — lazy
                            // list allocation on first call; if the
                            // attribute is already set but NOT a list,
                            // PyPy raises TypeError("Cannot add note:
                            // __notes__ is not a list") per `:254`.
                            let existing = crate::baseobjspace::getattr_str(w_self, "__notes__")
                                .ok()
                                .filter(|w| !w.is_null());
                            let notes = match existing {
                                Some(v) if unsafe { crate::baseobjspace::isinstance_list_w(v) } => {
                                    v
                                }
                                Some(_) => {
                                    return Err(crate::PyError::type_error(
                                        "Cannot add note: __notes__ is not a list",
                                    ));
                                }
                                None => {
                                    let fresh = pyre_object::w_list_new(Vec::new());
                                    crate::baseobjspace::setattr_str(w_self, "__notes__", fresh)?;
                                    fresh
                                }
                            };
                            unsafe { pyre_object::w_list_append(notes, w_note) };
                            Ok(pyre_object::w_none())
                        },
                        2,
                    ),
                );
                // `interp_exceptions.py:233-241` — `descr_reduce` /
                // `descr_setstate`, installed on `BaseException` so every
                // subclass inherits them through the MRO.
                crate::dict_storage_store(
                    ns,
                    "__reduce__",
                    make_builtin_function_with_arity("__reduce__", base_exception_reduce, 1),
                );
                crate::dict_storage_store(
                    ns,
                    "__setstate__",
                    make_builtin_function_with_arity("__setstate__", base_exception_setstate, 2),
                );
            }
            // `interp_exceptions.py:379-397` — ImportError overrides reduce
            // and setstate to carry the `name`/`path`/`name_from` slots.
            // ModuleNotFoundError (built via `make_exc_type`) inherits these
            // through the MRO.
            if name == "ImportError" {
                crate::dict_storage_store(
                    ns,
                    "__reduce__",
                    make_builtin_function_with_arity("__reduce__", import_error_reduce, 1),
                );
                crate::dict_storage_store(
                    ns,
                    "__setstate__",
                    make_builtin_function_with_arity("__setstate__", import_error_setstate, 2),
                );
            }
            // `interp_exceptions.py:655-665` — OSError overrides reduce to
            // re-append the filename(s); its subclasses inherit it.
            if name == "OSError" {
                crate::dict_storage_store(
                    ns,
                    "__reduce__",
                    make_builtin_function_with_arity("__reduce__", os_error_reduce, 1),
                );
            }
        },
        base,
    );
    // Record the class so typedef::r#type can map a raised exception
    // back to its specific builtin class (TypeError, ValueError, ...).
    register_exc_class(name, cls);
    cls
}

/// Build a builtin exception class with more than one base, e.g.
/// `class UnsupportedOperation(OSError, ValueError)`
/// (`Modules/_io/_iomodule.c`).  The MRO is the C3 linearization over
/// `bases`; the first base drives instance layout.  `with_traceback` /
/// `add_note` are inherited through the MRO from `BaseException`, so
/// only `__new__` is installed here.
pub(crate) fn make_exc_type_multi(
    name: &'static str,
    new_fn: crate::gateway::BuiltinCodeFn,
    bases: &[PyObjectRef],
) -> PyObjectRef {
    let cls = crate::typedef::make_builtin_type_with_bases(
        name,
        move |ns| {
            crate::dict_storage_store(ns, "__new__", make_builtin_function("__new__", new_fn));
        },
        bases,
    );
    register_exc_class(name, cls);
    cls
}

/// Thread-local registry from exception class name (as used by
/// `ExcKind → exc_kind_name`) to the W_TypeObject exposed in the builtins
/// namespace. Populated at init-builtins time via `make_exc_type`.
///
/// Also propagates into `pyre_object::interp_exceptions`'s kind-indexed
/// registry so `w_exception_new(kind, ...)` populates
/// `ob_header.w_class` with the registered class — every
/// builtin-raised exception then satisfies
/// `space.type(w_exc) == registered class` per `baseobjspace.py:1367
/// exception_getclass`.
fn register_exc_class(name: &'static str, cls: PyObjectRef) {
    EXC_CLASS_REGISTRY.with(|r| {
        r.borrow_mut().insert(name, cls);
    });
    if let Some(kind) = pyre_object::interp_exceptions::exc_kind_from_name(name) {
        pyre_object::interp_exceptions::register_exc_class_for_kind(kind, cls);
    }
}

/// Look up a builtin exception class by its `ExcKind` name. Returns
/// `None` if the registry hasn't been populated yet (e.g. before
/// install_default_builtins).
pub fn lookup_exc_class(name: &str) -> Option<PyObjectRef> {
    EXC_CLASS_REGISTRY.with(|r| r.borrow().get(name).copied())
}

/// Look up the reusable prebuilt instance for a builtin exception
/// class, addressed by `ExcKind` name.  Mirrors RPython's
/// `rpython/rtyper/exceptiondata.py:34-45 get_standard_ll_exc_instance`
/// — the JIT's `_ovf` direct-raise rewrite
/// (`rpython/jit/codewriter/flatten.py:165-170`) emits
/// `raise <Constant(ll_ovf)>` with the prebuilt instance pointer (NOT
/// the class pointer).  The instance lives forever; callers can stamp
/// its pointer into a JIT constant pool.
///
/// Returns `None` when `name` is not one of the recognised `ExcKind`
/// names (`exc_kind_from_name` returns `None`); standard exceptions
/// listed by `pyre_jit::jit::exceptiondata::STANDARD_EXCEPTIONS` all
/// map through.
pub fn lookup_exc_instance(name: &str) -> Option<PyObjectRef> {
    let kind = pyre_object::interp_exceptions::exc_kind_from_name(name)?;
    Some(pyre_object::interp_exceptions::standard_exc_instance(kind))
}

thread_local! {
    static EXC_CLASS_REGISTRY: std::cell::RefCell<std::collections::HashMap<&'static str, PyObjectRef>>
        = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// `__build_class__(body, name, *bases)` — class creation.
///
/// PyPy equivalent: pyopcode.py BUILD_CLASS
/// Direct call to call::real_build_class (no callback needed —
/// interpreter and runtime are in the same crate).
fn builtin_build_class(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::call::real_build_class(args)
}

/// Get a reference to the `__build_class__` builtin function.
pub fn get_build_class_func() -> PyObjectRef {
    make_builtin_function("__build_class__", builtin_build_class)
}

/// `str(obj)` → convert to string
pub(crate) fn builtin_str(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["object", "encoding", "errors"], "str")?;
    // `str(object='', encoding='utf-8', errors='strict')` — every parameter
    // is positional-or-keyword (unicodeobject.py:descr_new).  An absent
    // `object` yields the empty string; an encoding/errors of None counts as
    // "not given".
    let obj = match resolve_pos_or_kw(pos.first().copied(), kwargs, "object", "str", 1)? {
        Some(o) => o,
        None => return Ok(w_str_new("")),
    };
    let w_encoding = resolve_pos_or_kw(pos.get(1).copied(), kwargs, "encoding", "str", 2)?;
    let w_errors = resolve_pos_or_kw(pos.get(2).copied(), kwargs, "errors", "str", 3)?;
    // `_get_encoding_and_errors` — a *supplied* encoding/errors must be a
    // str; an explicit `None` is supplied (not "omitted") and so is
    // rejected.  Encoding is validated before errors.
    if let Some(w) = w_encoding {
        if !unsafe { is_str(w) } {
            let tn = unsafe { (*(*w).ob_type).name };
            return Err(crate::PyError::type_error(format!(
                "str() argument 'encoding' must be str, not {tn}"
            )));
        }
    }
    if let Some(w) = w_errors {
        if !unsafe { is_str(w) } {
            let tn = unsafe { (*(*w).ob_type).name };
            return Err(crate::PyError::type_error(format!(
                "str() argument 'errors' must be str, not {tn}"
            )));
        }
    }
    let has_encoding = w_encoding.is_some();
    let has_errors = w_errors.is_some();
    if has_encoding || has_errors {
        if unsafe { is_str(obj) } {
            return Err(crate::PyError::type_error("decoding str is not supported"));
        }
        if !unsafe { pyre_object::bytesobject::is_bytes_like(obj) } {
            let tn = unsafe { (*(*obj).ob_type).name };
            return Err(crate::PyError::type_error(format!(
                "decoding to str: need a bytes-like object, {tn} found"
            )));
        }
        let mut decode_args = vec![obj, w_encoding.unwrap_or_else(w_none)];
        if let Some(e) = w_errors {
            decode_args.push(e);
        }
        return crate::typedef::bytes_method_decode(&decode_args);
    }
    unsafe {
        if is_str(obj) {
            // A `str` subclass keeps `ob_type` at STR_TYPE but carries the
            // Python class in `w_class`; honor its `__str__` override before
            // returning the raw value.
            let tp = (*obj).ob_type;
            // WTF-8-preserving so a `__str__` override returning a lone
            // surrogate yields that str rather than panicking.
            if let Some(r) = crate::display::builtin_subclass_dunder_obj(obj, tp, "__str__")? {
                return Ok(pyre_object::w_str_from_wtf8(
                    pyre_object::w_str_get_wtf8(r).to_owned(),
                ));
            }
            // `str(s) is s` only for an exact `str`; a subclass with no
            // `__str__` override is copied to a fresh base `str`.
            if is_exact_type(obj, &STR_TYPE) {
                return Ok(obj);
            }
            return Ok(pyre_object::w_str_from_wtf8(
                pyre_object::w_str_get_wtf8(obj).to_owned(),
            ));
        }
    }
    let w = unsafe { crate::py_str_wtf8(obj)? };
    Ok(pyre_object::w_str_from_wtf8(w))
}

/// `repr(obj)` → string representation
fn builtin_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "repr() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    // WTF-8-preserving so a `__repr__` override returning a lone surrogate
    // yields that str rather than panicking in the `String` path.
    let w = unsafe { crate::py_repr_wtf8(args[0])? };
    Ok(pyre_object::w_str_from_wtf8(w))
}

/// `unicodeobject.c:unicode_repr` post-pass — take the repr of `obj`
/// and escape every non-ASCII code point as `\xXX` / `\uXXXX` /
/// `\UXXXXXXXX`.  Shared by the `ascii()` builtin and the `!a`
/// `str.format` conversion.
pub(crate) fn py_ascii(obj: PyObjectRef) -> Result<String, crate::PyError> {
    let s = unsafe { crate::py_repr(obj)? };
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        let cp = ch as u32;
        if cp < 0x80 {
            out.push(ch);
        } else if cp <= 0xFF {
            out.push_str(&format!("\\x{cp:02x}"));
        } else if cp <= 0xFFFF {
            out.push_str(&format!("\\u{cp:04x}"));
        } else {
            out.push_str(&format!("\\U{cp:08x}"));
        }
    }
    Ok(out)
}

/// `bltinmodule.c:builtin_ascii` — like `repr`, but escape every
/// non-ASCII code point in the repr as `\xXX` / `\uXXXX` / `\UXXXXXXXX`.
fn builtin_ascii(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "ascii() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Ok(w_str_new(&py_ascii(args[0])?))
}

/// `int(obj)` → convert to int
/// call_function with exception propagation.
/// PyPy's space.get_and_call_function returns normally or raises;
/// pyre's call_function stashes errors as PY_NULL. This helper
/// recovers stashed errors as Result.
pub(crate) fn call_and_check(
    method: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let result = crate::call_function(method, args);
    if result == pyre_object::PY_NULL {
        if let Some(err) = crate::call::take_call_error() {
            return Err(err);
        }
        return Err(crate::PyError::type_error("call returned NULL"));
    }
    Ok(result)
}

/// intobject.py:989-1050 _new_baseint
pub fn builtin_int(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `int(x=0, base=10)` — `x` is positional-only (a `base` keyword is the
    // only one accepted), `base` is positional-or-keyword at position 2
    // (intobject.py descr_new).
    let (pos, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["base"], "int")?;
    let w_base = resolve_pos_or_kw(pos.get(1).copied(), kwargs, "base", "int", 2)?;
    let obj = match pos.first().copied() {
        Some(o) => o,
        None => {
            // intobject.py:986 — a base without a value is a missing source.
            if w_base.is_some() {
                return Err(crate::PyError::type_error("int() missing string argument"));
            }
            return Ok(w_int_new(0));
        }
    };

    if w_base.is_none() {
        // intobject.py:991: space.is_w(space.type(w_value), space.w_int)
        let w_type = crate::typedef::r#type(obj);
        let w_int = crate::typedef::gettypefor(&INT_TYPE);
        if w_type.is_some() && w_type == w_int {
            return Ok(obj);
        }
        // intobject.py:994: space.lookup(w_value, '__int__')
        if let Some(method) = unsafe { crate::baseobjspace::lookup(obj, "__int__") } {
            // intobject.py:995: w_intvalue = space.int(w_value)
            let w_intvalue = call_and_check(method, &[obj])?;
            return ensure_baseint_result(w_intvalue, obj);
        }
        // intobject.py:997: space.lookup(w_value, '__trunc__')
        if let Some(method) = unsafe { crate::baseobjspace::lookup(obj, "__trunc__") } {
            // intobject.py:998-999: DeprecationWarning
            crate::warn::warn_deprecation("The delegation of int() to __trunc__ is deprecated.");
            // intobject.py:1001: w_obj = space.trunc(w_value)
            let w_obj = call_and_check(method, &[obj])?;
            // intobject.py:1002: if not space.isinstance_w(w_obj, space.w_int)
            if !unsafe { pyre_object::pyobject::is_int_or_long(w_obj) } {
                // intobject.py:1003-1004: try: w_obj = space.index(w_obj)
                if let Some(idx_method) = unsafe { crate::baseobjspace::lookup(w_obj, "__index__") }
                {
                    let w_indexed = call_and_check(idx_method, &[w_obj])?;
                    return ensure_baseint_result(w_indexed, obj);
                }
                // intobject.py:1008-1011
                return Err(crate::PyError::type_error(
                    "__trunc__ returned non-Integral (type '%T')",
                ));
            }
            return ensure_baseint_result(w_obj, obj);
        }
        // intobject.py:1015: space.lookup(w_value, '__index__')
        if let Some(method) = unsafe { crate::baseobjspace::lookup(obj, "__index__") } {
            // intobject.py:1016: w_obj = space.index(w_value)
            let w_obj = call_and_check(method, &[obj])?;
            // intobject.py:1017: if not space.is_w(space.type(w_obj), space.w_int)
            let w_obj_type = crate::typedef::r#type(w_obj);
            if w_obj_type != w_int {
                // intobject.py:1018: if space.isinstance_w(w_obj, space.w_int)
                if unsafe { pyre_object::pyobject::is_int_or_long(w_obj) } {
                    // intobject.py:1019: w_obj = space.int(w_obj)
                    return ensure_baseint_result(w_obj, obj);
                }
                // intobject.py:1020-1023
                return Err(crate::PyError::type_error(format!(
                    "int() argument must be a string, a bytes-like object or a real number, not '{}'",
                    crate::type_methods::arg_type_name(obj)
                )));
            }
            return ensure_baseint_result(w_obj, obj);
        }
        // intobject.py:1047 — unicode is normalized through
        // `unicode_to_decimal_w` so non-ASCII decimal digits parse.
        if unsafe { is_str(obj) } {
            let s = unsafe { w_str_get_value(obj) };
            return parse_int_from_str(&unicode_to_decimal(s), 10);
        }
        // intobject.py:1056-1070 — bytes / bytearray, then any object
        // exposing a readable buffer (`space.charbuf_w`).
        if let Some(src) = crate::typedef::buffer_as_bytes_like(obj)? {
            let data = unsafe { pyre_object::bytesobject::bytes_like_data(src) };
            let s = String::from_utf8_lossy(data);
            return parse_int_from_str(&s, 10);
        }
        return Err(crate::PyError::type_error(format!(
            "int() argument must be a string, a bytes-like object or a real number, not '{}'",
            crate::type_methods::arg_type_name(obj)
        )));
    }

    // intobject.py:1051-1072: w_base is not None — parse with base
    let base = getindex_w_for_base(w_base.unwrap())?;
    unsafe {
        // intobject.py:1079 — unicode normalized through `unicode_to_decimal_w`.
        if is_str(obj) {
            return parse_int_from_str(&unicode_to_decimal(w_str_get_value(obj)), base);
        }
        // With an explicit base only str / bytes / bytearray are accepted.
        if pyre_object::bytesobject::is_bytes_like(obj) {
            let data = pyre_object::bytesobject::bytes_like_data(obj);
            let s = String::from_utf8_lossy(data);
            return parse_int_from_str(&s, base);
        }
    }
    Err(crate::PyError::type_error(
        "int() can't convert non-string with explicit base",
    ))
}

/// intobject.py:1093-1107 _ensure_baseint
fn ensure_baseint_result(
    obj: PyObjectRef,
    _original: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        if is_int(obj) {
            // intobject.py:1096-1098: W_IntObject (or subclass) → wrapint
            return Ok(w_int_new(w_int_get_value(obj)));
        }
        if pyre_object::pyobject::is_long(obj) {
            // intobject.py:1100-1102: W_AbstractLongObject → newlong
            return Ok(pyre_object::longobject::w_long_new(
                pyre_object::longobject::w_long_get_value(obj).clone(),
            ));
        }
    }
    // intobject.py:1104-1107: shouldn't happen
    Err(crate::PyError::new(
        crate::PyErrorKind::RuntimeError,
        "internal error in int.__new__()".to_string(),
    ))
}

/// baseobjspace.py:1564-1596 space.getindex_w(w_base, None)
///
/// Calls __index__() on w_base and converts to i64.
/// On OverflowError (long that doesn't fit i64), returns 37 sentinel
/// (intobject.py:1057: causes ValueError in string_to_bigint).
fn getindex_w_for_base(w_base: PyObjectRef) -> Result<u32, crate::PyError> {
    let value = getindex_w(w_base)?;
    if value < 0 || value == 1 || value > 36 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            format!("int() base must be >= 2 and <= 36, or 0"),
        ));
    }
    Ok(value as u32)
}

/// baseobjspace.py:1564-1596 space.getindex_w(w_obj, None)
///
/// Return w_obj.__index__() as i64. On overflow, clamp to i64::MAX
/// (w_exception=None path).
pub(crate) fn getindex_w(w_obj: PyObjectRef) -> Result<i64, crate::PyError> {
    unsafe {
        if is_int(w_obj) {
            return Ok(w_int_get_value(w_obj));
        }
        if pyre_object::pyobject::is_long(w_obj) {
            // baseobjspace.py:1586-1591: try int_w, on overflow clamp
            let big = pyre_object::longobject::w_long_get_value(w_obj);
            return Ok(
                if pyre_object::longobject::jit_bigint_to_i64_fits(big) != 0 {
                    pyre_object::longobject::jit_bigint_to_i64_value(big)
                } else {
                    i64::MAX
                },
            );
        }
        // baseobjspace.py:1568: w_index = self.index(w_obj)
        if let Some(method) = crate::baseobjspace::lookup(w_obj, "__index__") {
            let w_index = call_and_check(method, &[w_obj])?;
            if is_int(w_index) {
                return Ok(w_int_get_value(w_index));
            }
            if pyre_object::pyobject::is_long(w_index) {
                let big = pyre_object::longobject::w_long_get_value(w_index);
                return Ok(
                    if pyre_object::longobject::jit_bigint_to_i64_fits(big) != 0 {
                        pyre_object::longobject::jit_bigint_to_i64_value(big)
                    } else {
                        i64::MAX
                    },
                );
            }
        }
    }
    Err(crate::PyError::type_error(format!(
        "int() second argument must be an integer, not '{}'",
        unsafe { (*(*w_obj).ob_type).name }
    )))
}

/// `unicodeobject.py unicode_to_decimal_w` — normalize a unicode string for
/// numeric parsing: a non-ASCII decimal digit (`Numeric_Type=Decimal`)
/// becomes its ASCII digit and non-ASCII whitespace becomes a space, so
/// `int("４２")` and `float("١٫٥")`-style inputs parse.  An all-ASCII string
/// is returned untouched.
fn unicode_to_decimal(s: &str) -> std::borrow::Cow<'_, str> {
    if s.is_ascii() {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c as u32 > 127 {
            if c.is_whitespace() {
                out.push(' ');
                continue;
            }
            if let Some(v) = rustpython_unicode::Ucd::new(true).decimal(CodePoint::from(c)) {
                out.push((b'0' + v as u8) as char);
                continue;
            }
        }
        out.push(c);
    }
    std::borrow::Cow::Owned(out)
}

/// Parse an integer from a string with the given base.
fn parse_int_from_str(s: &str, base: u32) -> Result<PyObjectRef, crate::PyError> {
    let s = s.trim();
    let (sign, rest) = if let Some(r) = s.strip_prefix('-') {
        (-1i64, r)
    } else if let Some(r) = s.strip_prefix('+') {
        (1i64, r)
    } else {
        (1i64, s)
    };
    let (radix, digits) = if base == 0 {
        if let Some(r) = rest.strip_prefix("0x").or(rest.strip_prefix("0X")) {
            (16u32, r)
        } else if let Some(r) = rest.strip_prefix("0b").or(rest.strip_prefix("0B")) {
            (2u32, r)
        } else if let Some(r) = rest.strip_prefix("0o").or(rest.strip_prefix("0O")) {
            (8u32, r)
        } else {
            (10u32, rest)
        }
    } else {
        let stripped = match base {
            16 => rest
                .strip_prefix("0x")
                .or(rest.strip_prefix("0X"))
                .unwrap_or(rest),
            2 => rest
                .strip_prefix("0b")
                .or(rest.strip_prefix("0B"))
                .unwrap_or(rest),
            8 => rest
                .strip_prefix("0o")
                .or(rest.strip_prefix("0O"))
                .unwrap_or(rest),
            _ => rest,
        };
        (base, stripped)
    };
    let cleaned: String = digits.chars().filter(|&c| c != '_').collect();
    if let Ok(v) = i64::from_str_radix(&cleaned, radix) {
        return Ok(w_int_new(sign * v));
    }
    // Values outside the machine-int range parse as arbitrary precision.
    if let Some(big) = BigInt::parse_bytes(cleaned.as_bytes(), radix) {
        let signed = if sign < 0 { -big } else { big };
        return Ok(w_long_new(signed));
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::ValueError,
        format!("invalid literal for int() with base {base}: '{s}'"),
    ))
}

/// Remove PEP 515 underscore digit separators, rejecting any underscore
/// that is not flanked by two ASCII digits — `_Py_string_to_number_with_
/// underscores`. Returns `None` for an invalid placement (leading,
/// trailing, doubled, or adjacent to `.`/`e`/sign).
fn strip_numeric_underscores(s: &str) -> Option<String> {
    if !s.contains('_') {
        return Some(s.to_string());
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(chars.len());
    for i in 0..chars.len() {
        let c = chars[i];
        if c == '_' {
            let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
            let next_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
            if prev_digit && next_digit {
                continue;
            }
            return None;
        }
        out.push(c);
    }
    Some(out)
}

/// `float.__float__(self)` — floatobject.py descr___float__: an exact float
/// is returned as-is; a strict subclass is down-converted to a fresh base
/// `float`.  Kept separate from the `float()` constructor so `float()` can look
/// `__float__` up on a subclass (honoring an override) without recursing.
pub(crate) fn builtin_float_dunder(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[0];
    unsafe {
        if is_exact_type(obj, &FLOAT_TYPE) {
            return Ok(obj);
        }
        if is_float(obj) {
            return Ok(floatobject::w_float_new(w_float_get_value(obj)));
        }
    }
    // interp2app(W_FloatObject.descr_float) requires a float `self`.
    Err(crate::PyError::type_error(format!(
        "descriptor '__float__' requires a 'float' object but received a '{}'",
        crate::type_methods::arg_type_name(obj)
    )))
}

/// `float(obj)` → convert to float
pub(crate) fn builtin_float(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(floatobject::w_float_new(0.0));
    }
    // Skip `cls` if called via `float.__new__(cls, value)`.
    let value_idx = if args.len() >= 2 && unsafe { pyre_object::is_type(args[0]) } {
        1
    } else {
        0
    };
    if value_idx >= args.len() {
        return Ok(floatobject::w_float_new(0.0));
    }
    let obj = args[value_idx];
    unsafe {
        if is_float(obj) {
            // `float(f) is f` for an exact `float`. A subclass falls through
            // to the `__float__` lookup below so an overridden `__float__` is
            // honored; a subclass that does not override it resolves to
            // `float.__float__`, which returns a fresh base `float`.
            if is_exact_type(obj, &FLOAT_TYPE) {
                return Ok(obj);
            }
        } else if is_int(obj) {
            return Ok(floatobject::w_float_new(w_int_get_value(obj) as f64));
        }
        if is_bool(obj) {
            return Ok(floatobject::w_float_new(if w_bool_get_value(obj) {
                1.0
            } else {
                0.0
            }));
        }
        if pyre_object::is_long(obj) {
            // A Python int is finite, so a non-finite conversion means the
            // magnitude exceeds f64 range.
            let v = pyre_object::jit_bigint_to_f64_or_nan(pyre_object::w_long_get_value(obj));
            if !v.is_finite() {
                return Err(crate::PyError::overflow_error(
                    "int too large to convert to float",
                ));
            }
            return Ok(floatobject::w_float_new(v));
        }
        if is_str(obj) {
            let raw = w_str_get_value(obj);
            // floatobject.py:242 — unicode is normalized through
            // `unicode_to_decimal_w` before `_string_to_float`, so non-ASCII
            // decimal digits parse.
            let s = unicode_to_decimal(raw);
            // `float_from_string` strips PEP 515 underscore separators
            // (between digits only) before parsing; the numeric conversion
            // uses the Python-literal float grammar.
            if let Some(cleaned) = strip_numeric_underscores(s.trim()) {
                if let Some(v) = rustpython_literal::float::parse_str(&cleaned) {
                    return Ok(floatobject::w_float_new(v));
                }
            }
            // `floatobject.py:descr_new` — message uses single-quoted str:
            // "could not convert string to float: '<s>'".
            return Err(crate::PyError::value_error(format!(
                "could not convert string to float: '{raw}'"
            )));
        }
    }
    // descroperation.py float — type-MRO __float__ then __index__
    if let Some(tp) = crate::typedef::r#type(obj) {
        if let Some(method) = unsafe { crate::baseobjspace::lookup_in_type(tp, "__float__") } {
            let result = crate::call::call_function_impl_result(method, &[obj])?;
            unsafe {
                if is_float(result) {
                    // floatobject.py:228-238 — an exact float is returned as-is;
                    // a strict subclass warns (deprecated) and is converted to a
                    // base float value.
                    if is_exact_type(result, &FLOAT_TYPE) {
                        return Ok(result);
                    }
                    let value_type = crate::type_methods::arg_type_name(obj);
                    let result_type = crate::type_methods::arg_type_name(result);
                    crate::warn::warn_deprecation(&format!(
                        "{value_type}.__float__ returned non-float (type {result_type}).  \
                         The ability to return an instance of a strict subclass of \
                         float is deprecated, and may be removed in a future version \
                         of Python."
                    ));
                    return Ok(floatobject::w_float_new(w_float_get_value(result)));
                }
            }
            // descroperation.py:891 — __float__ returned non-float (type '%T')
            let result_type = unsafe { (*(*result).ob_type).name };
            return Err(crate::PyError::type_error(format!(
                "__float__ returned non-float (type '{result_type}')",
            )));
        }
        if let Some(method) = unsafe { crate::baseobjspace::lookup_in_type(tp, "__index__") } {
            let r = crate::call::call_function_impl_result(method, &[obj])?;
            // descroperation.py:609 — exact int or bool (int subclass)
            unsafe {
                if is_int(r) || is_bool(r) {
                    return Ok(floatobject::w_float_new(w_int_get_value(r) as f64));
                }
            }
            let result_type = unsafe { (*(*r).ob_type).name };
            return Err(crate::PyError::type_error(format!(
                "__index__ returned non-int (type '{result_type}')",
            )));
        }
    }
    // floatobject.py:247-255 — a readable buffer (`charbuf_w`: bytes /
    // bytearray / array / memoryview) is decoded and parsed like a str; an
    // unparseable value reprs as `b'...'` in the error (space.repr / `%R`).
    if let Some(src) = crate::typedef::buffer_as_bytes_like(obj)? {
        let data = unsafe { pyre_object::bytesobject::bytes_like_data(src) };
        let decoded = String::from_utf8_lossy(data);
        if let Some(cleaned) = strip_numeric_underscores(decoded.trim()) {
            if let Ok(v) = cleaned.parse::<f64>() {
                return Ok(floatobject::w_float_new(v));
            }
        }
        let r = unsafe { crate::py_repr(obj)? };
        return Err(crate::PyError::value_error(format!(
            "could not convert string to float: {r}"
        )));
    }
    // The message uses the modern "real number" wording (3.14) rather than
    // the older "a number" phrasing.
    Err(crate::PyError::type_error(format!(
        "float() argument must be a string or a real number, not '{}'",
        crate::type_methods::arg_type_name(obj)
    )))
}

/// The attribute-name check mirroring `operation.py:41-45 checkattrname`
/// (accept `str` and any `str` subclass via `isinstance_w`), raising
/// `"attribute name must be string, not '<type>'"`. getattr/hasattr/
/// setattr/delattr all route through it, matching `operation.py` (and the
/// unified 3.12+ message).
fn checkattrname(w_name: PyObjectRef) -> Result<(), crate::PyError> {
    if !unsafe { crate::baseobjspace::isinstance_str_w(w_name) } {
        let name_type = unsafe { (*(*w_name).ob_type).name };
        return Err(crate::PyError::type_error(format!(
            "attribute name must be string, not '{name_type}'",
        )));
    }
    Ok(())
}

/// `operation.py:65-74 hasattr(obj, name)` → bool: `checkattrname`, then
/// (unlike Py2) only an `AttributeError` yields `False`; any other error
/// propagates.
fn builtin_hasattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "hasattr() takes exactly two arguments ({} given)",
            args.len()
        )));
    }
    let obj = args[0];
    checkattrname(args[1])?;
    match crate::baseobjspace::getattr(obj, args[1]) {
        Ok(_) => Ok(w_bool_from(true)),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => Ok(w_bool_from(false)),
        Err(e) => Err(e),
    }
}

/// `getattr(obj, name[, default])` → value — direct call
fn builtin_getattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "getattr() takes no keyword arguments",
        ));
    }
    // `getattr(object, name[, default])`: two or three app-level arguments.
    if args.len() < 2 {
        return Err(crate::PyError::type_error(format!(
            "getattr expected at least 2 arguments, got {}",
            args.len()
        )));
    }
    if args.len() > 3 {
        return Err(crate::PyError::type_error(format!(
            "getattr expected at most 3 arguments, got {}",
            args.len()
        )));
    }
    let obj = args[0];
    checkattrname(args[1])?;
    // operation.py:58-64: the default replaces the error ONLY when a default
    // was supplied AND the error is an AttributeError; other errors (and the
    // no-default case) propagate.
    match crate::baseobjspace::getattr(obj, args[1]) {
        Ok(val) => Ok(val),
        Err(e) => {
            if args.len() > 2 && e.kind == crate::PyErrorKind::AttributeError {
                Ok(args[2]) // default value
            } else {
                Err(e) // propagate
            }
        }
    }
}

/// `pypy/module/__builtin__/operation.py:191-196 setattr`:
///
/// ```python
/// def setattr(space, w_object, w_name, w_val):
///     w_name = checkattrname(space, w_name)
///     space.setattr(w_object, w_name, w_val)
///     return space.w_None
/// ```
///
/// The space-level `setattr` may raise (AttributeError on read-only
/// descriptors, TypeError on wrong-type values, etc.) and PyPy
/// propagates those errors — they are NOT swallowed here.
fn builtin_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 3 {
        return Err(crate::PyError::type_error(format!(
            "setattr() takes exactly three arguments ({} given)",
            args.len()
        )));
    }
    let obj = args[0];
    checkattrname(args[1])?;
    crate::baseobjspace::setattr(obj, args[1], args[2])?;
    Ok(w_none())
}

/// `delattr(obj, name)` — PyPy: baseobjspace.py delattr
fn builtin_delattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "delattr() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    let obj = args[0];
    checkattrname(args[1])?;
    crate::baseobjspace::delattr(obj, args[1])?;
    Ok(w_none())
}

pub(crate) fn builtin_tuple(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `tuple.__new__` is positional-only, so any keyword is a TypeError
    // (an empty `**{}` is not a keyword and is allowed).
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "tuple() takes no keyword arguments",
        ));
    }
    if args.is_empty() {
        return Ok(w_tuple_new(vec![]));
    }
    let obj = args[0];
    unsafe {
        // `tuple(t)` returns `t` unchanged only for an EXACT tuple
        // (`PyTuple_CheckExact`); a tuple subclass must be re-iterated
        // through its (possibly overridden) `__iter__`, and the raw
        // storage fast paths below likewise apply only to exact
        // tuple/list instances.
        if is_exact_tuple(obj) {
            return Ok(obj);
        }
        if is_exact_list(obj) {
            let n = w_list_len(obj);
            let items: Vec<_> = (0..n)
                .filter_map(|i| w_list_getitem(obj, i as i64))
                .collect();
            return Ok(w_tuple_new(items));
        }
    }
    Ok(w_tuple_new(collect_iterable(obj)?))
}

pub(crate) fn builtin_list_ctor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `list.__new__` is positional-only, so any keyword is a TypeError
    // (an empty `**{}` is not a keyword and is allowed).
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "list() takes no keyword arguments",
        ));
    }
    if args.is_empty() {
        return Ok(w_list_new(vec![]));
    }
    let obj = args[0];
    unsafe {
        // The raw-storage copy fast paths apply only to EXACT tuple/list
        // instances; a subclass may override `__iter__`, so it must go
        // through `collect_iterable` (`iter(obj)`).
        if is_exact_list(obj) {
            // Copy the list
            let n = w_list_len(obj);
            let items: Vec<_> = (0..n)
                .filter_map(|i| w_list_getitem(obj, i as i64))
                .collect();
            return Ok(w_list_new(items));
        }
        if is_exact_tuple(obj) {
            let n = w_tuple_len(obj);
            let items: Vec<_> = (0..n)
                .filter_map(|i| w_tuple_getitem(obj, i as i64))
                .collect();
            return Ok(w_list_new(items));
        }
    }
    // Consume iterator — PyPy: listobject.py W_ListObject(iterable)
    Ok(w_list_new(collect_iterable(obj)?))
}

pub fn collect_iterable(obj: PyObjectRef) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let it = crate::baseobjspace::iter(obj)?;
    let mut items = Vec::new();
    loop {
        match crate::baseobjspace::next(it) {
            Ok(v) => items.push(v),
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    Ok(items)
}

/// Create a `set` from a slice of elements.
///
/// PyPy: `setobject.py` W_SetObject.descr_init → `_initialize_set`.
/// Each `add` hashes the element through `space.hash_w`, so an unhashable
/// one (or a raising / `None` / non-int `__hash__`) raises. Build the set
/// element-by-element so the first such element surfaces in left-to-right
/// order before it is stored; `w_set_add` keeps the `dict_keys_equal`
/// dedup, and the hash itself is not used for storage.
pub fn builtin_set_from_items(items: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let set = pyre_object::w_set_new();
    for &item in items {
        try_hash_value(item)?;
        unsafe { pyre_object::w_set_add(set, item) };
    }
    Ok(set)
}

/// `dict()` — PyPy: dictmultiobject.py W_DictMultiObject.descr_init
pub(crate) fn builtin_dict_ctor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_dict_new());
    }
    let src = args[0];
    unsafe {
        if is_dict(src) {
            // PyPy: descr_init → shallow copy when first arg is a dict.
            // `dict(**kwargs)` reaches here with the `__pyre_kw__`-tagged
            // kwargs vehicle as the source; its marker entry carries the
            // unforgeable sentinel value, so drop that entry by identity — a
            // real `__pyre_kw__` string key in a copied user dict is kept.
            let dict = w_dict_new();
            for (k, v) in pyre_object::w_dict_items(src) {
                if pyre_object::kw_marker::is_kw_marker_sentinel(v) {
                    continue;
                }
                w_dict_store(dict, k, v);
            }
            return Ok(dict);
        }
    }
    // PyPy: dictmultiobject.py update1 -> update1_dict_dict (mapping) or update1_pairs (seq)
    //
    // Mapping protocol: if arg has `keys()`, iterate keys and use __getitem__.
    // This handles dict subclasses (e.g. enum.EnumDict) where is_dict() is false.
    if let Ok(keys_method) = crate::baseobjspace::getattr_str(src, "keys") {
        let dict = w_dict_new();
        let keys_obj = crate::call_function(keys_method, &[]);
        let keys = collect_iterable(keys_obj)?;
        for key in keys {
            let val = crate::baseobjspace::getitem(src, key)?;
            unsafe { w_dict_store(dict, key, val) };
        }
        return Ok(dict);
    }
    // Construct from iterable of (key, value) pairs — dictmultiobject.py
    // update1_pairs coerces each pair with `space.fixedview` (any 2-element
    // iterable), not just tuple/list.
    let dict = w_dict_new();
    let items = collect_iterable(src)?;
    for (i, pair) in items.into_iter().enumerate() {
        let elems = collect_iterable(pair)?;
        if elems.len() != 2 {
            return Err(crate::PyError::value_error(format!(
                "dictionary update sequence element #{i} has length {}; 2 is required",
                elems.len()
            )));
        }
        unsafe { w_dict_store(dict, elems[0], elems[1]) };
    }
    Ok(dict)
}

/// `super()` — PyPy: descriptor.py W_Super
/// `super(cls, obj)` — PyPy: pypy/module/__builtin__/descriptor.py W_Super
///
/// Returns a proxy that looks up methods in cls's MRO starting after cls.
/// `getattr` handles the super proxy via `is_super` check.
///
/// Zero-arg super() finds __class__ and self from the calling frame.
/// CPython: Objects/typeobject.c super_init
fn builtin_super(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "super() takes no keyword arguments",
        ));
    }
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "super() expected at most 2 arguments, got {}",
            args.len()
        )));
    }
    if args.len() >= 2 {
        let cls = args[0];
        let obj = args[1];
        return Ok(pyre_object::descriptor::w_super_new(cls, obj));
    }
    // Zero-arg super(): find __class__ cell and first arg from calling frame
    //
    // IMPORTANT: CURRENT_FRAME points to the frame that is currently
    // executing the `super()` CALL.  For zero-arg super the __class__
    // cell lives in the *caller* of super(), which IS the current frame
    // (super is a builtin, not a user function that gets its own frame).
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame_ptr = current.get();
        if frame_ptr.is_null() {
            return Err(crate::PyError::runtime_error("super(): no current frame"));
        }
        let frame = unsafe { &*frame_ptr };
        let code = frame.code();

        // Find __class__ in freevars (it's a cell variable from the enclosing class scope)
        let num_locals = code.varnames.len();
        let ncellvars = code.cellvars.len();
        let locals = frame.locals_w().as_slice();

        let mut w_class = pyre_object::PY_NULL;

        // Check freevars for __class__
        for (slot, name) in code.freevars.iter().enumerate() {
            if name == "__class__" {
                let idx = num_locals + ncellvars + slot;
                if idx < locals.len() {
                    let cell = locals[idx];
                    if !cell.is_null() {
                        if unsafe { pyre_object::is_cell(cell) } {
                            w_class = unsafe { pyre_object::w_cell_get(cell) };
                        } else {
                            w_class = cell;
                        }
                    }
                }
                break;
            }
        }

        // Also check cellvars for __class__
        if w_class.is_null() {
            for (slot, name) in code.cellvars.iter().enumerate() {
                if name == "__class__" {
                    let idx = if code.varnames.iter().any(|v| v == name) {
                        code.varnames.iter().position(|v| v == name).unwrap()
                    } else {
                        num_locals + slot
                    };
                    if idx < locals.len() {
                        let cell = locals[idx];
                        if !cell.is_null() {
                            if unsafe { pyre_object::is_cell(cell) } {
                                w_class = unsafe { pyre_object::w_cell_get(cell) };
                            } else {
                                w_class = cell;
                            }
                        }
                    }
                    break;
                }
            }
        }

        if w_class.is_null() {
            return Err(crate::PyError::runtime_error(
                "super(): __class__ cell not found",
            ));
        }

        // First argument is self/cls/mcs (locals[0])
        let w_self = if locals.is_empty() {
            pyre_object::PY_NULL
        } else {
            locals[0]
        };

        if w_self.is_null() {
            return Err(crate::PyError::runtime_error(
                "super(): no first argument found",
            ));
        }

        Ok(pyre_object::descriptor::w_super_new(w_class, w_self))
    })
}

/// `iter(obj)` / `iter(callable, sentinel)` — PyPy:
/// `module/__builtin__/operation.py` iter
fn builtin_iter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `iter` is a keyword-rejecting builtin; a `sentinel` is only ever
    // positional (`iter(callable, sentinel)`).  Strip the kwargs marker so a
    // keyword call raises instead of consuming the marker dict as an argument.
    let (positional, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "iter() takes no keyword arguments",
        ));
    }
    match positional.len() {
        0 => Err(crate::PyError::type_error(
            "iter() requires at least one argument",
        )),
        1 => crate::baseobjspace::iter(positional[0]),
        2 => {
            if !crate::baseobjspace::callable_w(positional[0]) {
                return Err(crate::PyError::type_error("iter(v, w): v must be callable"));
            }
            Ok(pyre_object::operation::w_callable_iterator_new(
                positional[0],
                positional[1],
            ))
        }
        n => Err(crate::PyError::type_error(format!(
            "iter expected at most 2 arguments, got {n}"
        ))),
    }
}

/// `next(iterator[, default])` — PyPy: baseobjspace.py next
fn builtin_next(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "next() takes no keyword arguments",
        ));
    }
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "next() requires at least one argument",
        ));
    }
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "next expected at most 2 arguments, got {}",
            args.len()
        )));
    }
    match crate::baseobjspace::next(args[0]) {
        Ok(v) => Ok(v),
        Err(e) if e.kind == crate::PyErrorKind::StopIteration && args.len() > 1 => {
            Ok(args[1]) // default value
        }
        Err(e) => Err(e),
    }
}

/// `callable(obj)` — PyPy: baseobjspace.py callable
fn builtin_callable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[0];
    // `PyCallable_Check` — true when `type(obj)` has `tp_call`.  The builtin
    // callable kinds (function / builtin function, bound method, static- and
    // classmethod, type) are dispatched through dedicated slots in `call.rs`
    // rather than a `__call__` dict entry, so each is recognised directly;
    // any other object is callable iff its type defines `__call__`.
    let is_callable = unsafe {
        crate::is_function(obj)
            || pyre_object::is_type(obj)
            || pyre_object::is_method(obj)
            || pyre_object::function::is_staticmethod(obj)
            || pyre_object::function::is_classmethod(obj)
            || crate::typedef::r#type(obj)
                .and_then(|t| crate::baseobjspace::lookup_in_type(t, "__call__"))
                .is_some()
    };
    Ok(w_bool_from(is_callable))
}

/// `compile(source, filename, mode, ...)` — PyPy: pyopcode.py builtin_compile
///
/// Compiles a Python string to a code object. Only `source`, `filename` and
/// `mode` are honoured; flags / dont_inherit / optimize are accepted but
/// ignored, matching the minimal stub PyPy uses for shim modules.
/// Map a compiler failure string to a Python `SyntaxError`, matching
/// CPython where `compile`/`exec`/`eval`/`ast.parse` raise `SyntaxError`
/// (not `ValueError`) for malformed source.  The `compile error: ` prefix
/// `compile_source` prepends is stripped so the message reads like
/// CPython's (`'yield' outside function`).
fn compile_err_to_syntax_error(e: String) -> crate::PyError {
    let msg = e.strip_prefix("compile error: ").unwrap_or(&e).to_string();
    crate::PyError::syntax_error(msg)
}

fn builtin_compile(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `compile(source, filename, mode, flags=0, dont_inherit=False,
    // optimize=-1, *, _feature_version=-1)`: the three required parameters are
    // positional-or-keyword.  pyre honours only source/filename/mode; the
    // remaining optional parameters are accepted (so a keyword call does not
    // error) but not yet applied.
    let (pos, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(
        kwargs,
        &[
            "source",
            "filename",
            "mode",
            "flags",
            "dont_inherit",
            "optimize",
            "_feature_version",
        ],
        "compile",
    )?;
    let source = bind_pos_or_kw(pos, kwargs, 0, "source", "compile", 1)?.ok_or_else(|| {
        crate::PyError::type_error("compile() missing required argument 'source' (pos 1)")
    })?;
    let filename_obj =
        bind_pos_or_kw(pos, kwargs, 1, "filename", "compile", 2)?.ok_or_else(|| {
            crate::PyError::type_error("compile() missing required argument 'filename' (pos 2)")
        })?;
    let mode_obj = bind_pos_or_kw(pos, kwargs, 2, "mode", "compile", 3)?.ok_or_else(|| {
        crate::PyError::type_error("compile() missing required argument 'mode' (pos 3)")
    })?;
    let source_str = unsafe {
        if pyre_object::is_str(source) {
            // The source is decoded to UTF-8 for the tokenizer; a lone
            // surrogate raises `UnicodeEncodeError` (strict) rather than
            // panicking in `w_str_get_value`.
            let bytes = crate::type_methods::encode_object(source, "utf-8", "strict")?;
            String::from_utf8(bytes).expect("strict utf-8 encode yields valid utf-8")
        } else if pyre_object::bytesobject::is_bytes_like(source) {
            String::from_utf8_lossy(pyre_object::bytesobject::bytes_like_data(source)).into_owned()
        } else {
            return Err(crate::PyError::type_error(
                "compile() arg 1 must be a string or bytes",
            ));
        }
    };
    let filename = unsafe {
        if pyre_object::is_str(filename_obj) {
            pyre_object::w_str_get_value(filename_obj).to_string()
        } else {
            "<string>".to_string()
        }
    };
    let mode = unsafe {
        if pyre_object::is_str(mode_obj) {
            pyre_object::w_str_get_value(mode_obj).to_string()
        } else {
            "exec".to_string()
        }
    };
    let mode = match mode.as_str() {
        "exec" => crate::compile::Mode::Exec,
        "eval" => crate::compile::Mode::Eval,
        "single" => crate::compile::Mode::Single,
        other => {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                format!("compile() mode must be 'exec', 'eval' or 'single', not {other:?}"),
            ));
        }
    };
    let code = crate::compile::compile_source_with_filename(&source_str, mode, &filename)
        .map_err(compile_err_to_syntax_error)?;
    let code_ptr = Box::into_raw(Box::new(code)) as *const ();
    Ok(crate::w_code_new(code_ptr))
}

/// `exec(source_or_code, globals=None, locals=None)` — PyPy:
/// pyopcode.py builtin_exec.
///
/// Compiles `source` if necessary, then runs the resulting code object in
/// the supplied namespaces.  When the namespaces are dicts, pyre converts
/// them into `DictStorage`s before invocation and copies the post-run
/// namespace contents back so that callers see the new bindings.
fn builtin_exec(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `exec(source, /, globals=None, locals=None, *, closure=None)`: source is
    // positional-only; globals/locals are positional-or-keyword.  `closure` is
    // accepted (keyword-only) but pyre's exec plumbing carries no closure, so a
    // None closure is a no-op and a non-None one is unsupported.
    let (pos, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["globals", "locals", "closure"], "exec")?;
    if pos.is_empty() {
        return Err(crate::PyError::type_error(
            "exec() takes at least 1 positional argument (0 given)",
        ));
    }
    let source = pos[0];
    let globals_arg =
        bind_pos_or_kw(pos, kwargs, 1, "globals", "exec", 2)?.unwrap_or(pyre_object::PY_NULL);
    let locals_arg =
        bind_pos_or_kw(pos, kwargs, 2, "locals", "exec", 3)?.unwrap_or(pyre_object::PY_NULL);
    if let Some(closure) = kwarg_get(kwargs, "closure") {
        if !unsafe { pyre_object::is_none(closure) } {
            return Err(crate::PyError::type_error(
                "exec() closure argument is not supported",
            ));
        }
    }
    exec_or_eval(source, globals_arg, locals_arg, false)
}

/// `eval(source_or_code, globals=None, locals=None)` — same plumbing as
/// exec but returns the value of the expression.
fn builtin_eval(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `eval(source, /, globals=None, locals=None)`: source is positional-only;
    // globals/locals are positional-or-keyword.
    let (pos, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["globals", "locals"], "eval")?;
    if pos.is_empty() {
        return Err(crate::PyError::type_error(
            "eval() takes at least 1 positional argument (0 given)",
        ));
    }
    let source = pos[0];
    let globals_arg =
        bind_pos_or_kw(pos, kwargs, 1, "globals", "eval", 2)?.unwrap_or(pyre_object::PY_NULL);
    let locals_arg =
        bind_pos_or_kw(pos, kwargs, 2, "locals", "eval", 3)?.unwrap_or(pyre_object::PY_NULL);
    exec_or_eval(source, globals_arg, locals_arg, true)
}

fn exec_or_eval(
    source: PyObjectRef,
    globals_arg: PyObjectRef,
    locals_arg: PyObjectRef,
    is_eval: bool,
) -> Result<PyObjectRef, crate::PyError> {
    // Resolve a runnable code object: accept a precompiled W_Code or
    // compile a str on the fly.
    let code_obj_ref = unsafe {
        // `compiling.py:103 source_as_str` — accepts a str or a bytes-like
        // source (decoded), or an already-compiled code object.
        let source_str = if pyre_object::is_str(source) {
            Some(pyre_object::w_str_get_value(source).to_string())
        } else if pyre_object::bytesobject::is_bytes_like(source) {
            Some(
                String::from_utf8_lossy(pyre_object::bytesobject::bytes_like_data(source))
                    .into_owned(),
            )
        } else {
            None
        };
        if let Some(s) = source_str {
            let mode = if is_eval {
                crate::compile::Mode::Eval
            } else {
                crate::compile::Mode::Exec
            };
            let code =
                crate::compile::compile_source(&s, mode).map_err(compile_err_to_syntax_error)?;
            let code_ptr = Box::into_raw(Box::new(code)) as *const ();
            crate::w_code_new(code_ptr)
        } else if !source.is_null() && crate::is_code(source) {
            source
        } else {
            return Err(crate::PyError::type_error(format!(
                "{}() arg 1 must be a string, bytes or code object",
                if is_eval { "eval" } else { "exec" }
            )));
        }
    };
    let raw_code = unsafe {
        crate::w_code_get_ptr(code_obj_ref as pyre_object::PyObjectRef) as *const crate::CodeObject
    };

    // pyopcode.py:738-759 exec_ — refuse a code object that needs a
    // closure with the exec-side TypeError.  PyPy exposes a keyword-only
    // `w_closure` parameter and builds an `outer_func` from it after
    // validating the tuple of cells.  Pyre's exec() builtin currently
    // accepts only source/globals/locals, so no closure cells can be
    // supplied yet.
    //
    // eval() takes no closure parameter even on CPython; the eval-side
    // error message comes from initialize_frame_scopes (pyframe.py:242-246
    // "directly executed code object may not contain free variables") via
    // createframe → ?, which is what compiling.py:99 eval_function reaches
    // through code.exec_code.
    if !is_eval {
        let needed_freevars = unsafe { (&*raw_code).freevars.len() };
        if needed_freevars > 0 {
            return Err(crate::PyError::type_error(format!(
                "code object requires a closure of exactly length {needed_freevars}"
            )));
        }
    }

    // pypy/interpreter/eval.py:28-33 Code.exec_code keeps w_globals and
    // w_locals as separate dict references — STORE_GLOBAL writes to
    // w_globals and STORE_NAME writes to w_locals.  Pyre mirrors this by
    // building a fresh DictStorage per role and syncing each back to the
    // caller's dict on exit.  When `locals is globals` (module-level exec
    // / dataclasses), both sides reuse the same storage so semantics
    // collapse to PyPy's `space.createframe(self, w_globals)` followed by
    // a same-dict setdictscope.
    fn is_none_or_null(w_obj: PyObjectRef) -> bool {
        w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) }
    }

    fn type_name_of(w_obj: PyObjectRef) -> String {
        unsafe {
            match crate::typedef::r#type(w_obj) {
                Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
                None => (*(*w_obj).ob_type).name.to_string(),
            }
        }
    }

    fn is_dict_w(w_obj: PyObjectRef) -> bool {
        unsafe {
            let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
            crate::baseobjspace::isinstance_w(w_obj, w_dict_type)
        }
    }

    /// Ensure the globals dict OBJECT carries a str-keyed `DictStorage`
    /// proxy for the bytecode globals fastpath (LOAD_GLOBAL /
    /// STORE_GLOBAL / DELETE_GLOBAL).
    ///
    /// `pypy/interpreter/pyopcode.py:771-776` runs the frame on the user
    /// dict directly; pyre's bytecode handlers still read a `*mut
    /// DictStorage`, so the dict needs a storage proxy attached.  Module
    /// dicts and dicts returned by `globals()` already carry one — reused
    /// as-is, so a second `exec(src, g)` allocates nothing (heap-stable).
    ///
    /// A fresh dict (`exec(src, {})`) gets one storage allocated,
    /// pre-populated from its str-keyed entries (LOAD_GLOBAL must see
    /// bindings the dict already held), back-mirrored (`set_mirror_target`,
    /// so STORE_GLOBAL / DELETE_GLOBAL through the storage propagate into
    /// the dict's entries), and attached as the proxy (so the dict's own
    /// `__setitem__` fans str-keyed writes back into the storage via
    /// `maybe_sync_dict_storage_store`).  The two halves stand in for
    /// PyPy's single `W_DictMultiObject`.
    ///
    /// The storage is owned by the dict for its lifetime — functions
    /// defined during the run capture the dict OBJECT as `__globals__` and
    /// recover this storage via `get_w_globals_storage()` — so it is intentionally
    /// leaked alongside the dict rather than freed.  Retires with the
    /// `dict_storage_proxy` bridge when the dict becomes the single store.
    fn ensure_globals_storage_proxy(w_globals: pyre_object::PyObjectRef) {
        if w_globals.is_null() {
            return;
        }
        // Dispatch storage on `resolve_dict_backing` (a dict subclass
        // resolves to itself; a dict-proxy / mapping to its inner dict),
        // mirroring PyPy's storage-vs-original split.
        let backing = unsafe { crate::type_methods::resolve_dict_backing(w_globals) };
        if backing.is_null() {
            return;
        }
        if !unsafe { pyre_object::w_dict_get_dict_storage_proxy(backing) }.is_null() {
            // module.__dict__ / globals() / a dict reused across exec —
            // already storage-backed, nothing to allocate.
            return;
        }
        let mut ns = Box::new(crate::DictStorage::new());
        unsafe {
            for (key, value) in pyre_object::w_dict_items(backing) {
                if !value.is_null() && pyre_object::is_str(key) {
                    let name = pyre_object::w_str_get_value(key).to_string();
                    crate::dict_storage_store(&mut ns, &name, value);
                }
            }
            ns.set_mirror_target(backing);
        }
        // The fresh immortal storage now holds refs copied out of a GC
        // dict (possibly young); rescan on the next minor collection.
        pyre_object::gc_roots::mark_prebuilt_roots_dirty();
        ns.fix_ptr();
        let storage_ptr: *mut crate::DictStorage = ns.as_mut() as *mut _;
        unsafe {
            pyre_object::w_dict_set_dict_storage_proxy(backing, storage_ptr as *mut u8);
        }
        // Owned by the dict for its lifetime; leaked rather than freed.
        let _ = Box::into_raw(ns);
    }

    fn ensure_eval_builtins(
        w_globals: pyre_object::PyObjectRef,
        exec_ctx: *const crate::PyExecutionContext,
    ) -> Result<(), crate::PyError> {
        // pypy/module/__builtin__/compiling.py:109-110 eval:
        //
        //   if not space.contains_w(w_globals, space.newtext("__builtins__")):
        //       space.setitem_str(w_globals, "__builtins__", space.builtin)
        //
        // This is intentionally NOT pyopcode.py:773's `setdefault`
        // call-method path; dict-subclass `setdefault` overrides do
        // not fire for eval() in PyPy.  Dispatch on the dict object so
        // the str-keyed write fans into the storage proxy.
        let w_builtin = if !exec_ctx.is_null() {
            unsafe { (*exec_ctx).get_builtin() }
        } else {
            pyre_object::PY_NULL
        };
        if w_builtin.is_null() || w_globals.is_null() {
            return Ok(());
        }
        let key = pyre_object::w_str_new("__builtins__");
        if !crate::baseobjspace::contains(w_globals, key)? {
            crate::baseobjspace::setitem(w_globals, key, w_builtin)?;
        }
        Ok(())
    }

    fn ensure_exec_builtins(
        w_globals: pyre_object::PyObjectRef,
        caller_frame: *const crate::PyFrame,
        exec_ctx: *const crate::PyExecutionContext,
    ) -> Result<(), crate::PyError> {
        // pypy/interpreter/pyopcode.py:773-774
        //   space.call_method(w_globals, 'setdefault',
        //                     '__builtins__', self.get_builtin())
        // — `self` is the caller frame, so `get_builtin()` returns the
        // builtin picked at caller-frame creation (`pyframe.py:115-116`),
        // not the EC's default.  When the caller frame's picked builtin
        // is unavailable (e.g. exec called from outside any frame), fall
        // through to the EC default.  The receiver is the ORIGINAL
        // `w_globals` object so a dict-subclass `setdefault` override
        // fires.
        let w_builtin = if !caller_frame.is_null() {
            unsafe { (*caller_frame).get_builtin() }
        } else if !exec_ctx.is_null() {
            unsafe { (*exec_ctx).get_builtin() }
        } else {
            pyre_object::PY_NULL
        };
        if w_builtin.is_null() || w_globals.is_null() {
            return Ok(());
        }
        let key = pyre_object::w_str_new("__builtins__");
        let setdefault = crate::baseobjspace::getattr_str(w_globals, "setdefault")?;
        crate::call_and_check(setdefault, &[key, w_builtin])?;
        Ok(())
    }

    // pypy/interpreter/pyopcode.py:2003-2013 ensure_ns —
    //   globals: not None ⇒ isinstance_w(w_dict) else TypeError
    //   locals : not None ⇒ space.lookup(__getitem__) is not None
    //                       else TypeError "must be a mapping or None"
    let funcname = if is_eval { "eval" } else { "exec" };
    if !is_none_or_null(globals_arg) && !is_dict_w(globals_arg) {
        return Err(crate::PyError::type_error(format!(
            "{funcname}() arg 2 must be a dict, not {}",
            type_name_of(globals_arg)
        )));
    }
    if !is_none_or_null(locals_arg)
        && unsafe { crate::baseobjspace::lookup(locals_arg, "__getitem__").is_none() }
    {
        return Err(crate::PyError::type_error(format!(
            "{funcname}() arg 3 must be a mapping or None, not {}",
            type_name_of(locals_arg)
        )));
    }

    let caller_frame = crate::eval::CURRENT_FRAME.with(|current| current.get());
    let exec_ctx = if caller_frame.is_null() {
        std::ptr::null::<crate::PyExecutionContext>()
    } else {
        unsafe { (*caller_frame).execution_context }
    };

    // pyopcode.py:2005-2009 ensure_ns — the globals object is the
    // user-supplied dict, else the caller frame's globals, else a fresh
    // empty dict (`exec(src)` outside any frame, PyPy `newdict('module')`).
    let w_globals = if !is_none_or_null(globals_arg) {
        globals_arg
    } else if !caller_frame.is_null() {
        unsafe { (*caller_frame).get_w_globals() }
    } else {
        pyre_object::w_dict_new()
    };
    // Seed the str-keyed storage proxy that the LOAD_GLOBAL / STORE_GLOBAL
    // bytecode fastpath reads, mirroring the dict's str entries.  A dict
    // already backing a frame (module.__dict__ / globals()) keeps its proxy.
    ensure_globals_storage_proxy(w_globals);
    // pyopcode.py:773-774 `space.call_method(w_globals, 'setdefault', ...)`
    // (exec) and compiling.py:109-110 `space.setitem_str(w_globals, ...)`
    // (eval) dispatch on the ORIGINAL `w_globals` object so a dict-subclass
    // `setdefault` / `__contains__` / `__setitem__` override fires; the
    // str-keyed write fans into the storage proxy seeded above.
    if is_eval {
        ensure_eval_builtins(w_globals, exec_ctx)?;
    } else {
        ensure_exec_builtins(w_globals, caller_frame, exec_ctx)?;
    }

    // pypy/interpreter/pyopcode.py:771-776 — `code.exec_code(space,
    // w_globals, w_locals, outer_func)` runs the frame on the
    // user-supplied dict directly.  Pyre routes locals through
    // `frame.setdictscope(w_locals)` so STORE_NAME / LOAD_NAME /
    // DELETE_NAME dispatch via `space.setitem` / `space.getitem` /
    // `space.delitem` on the live mapping (dict subclass `__getitem__`
    // overrides win, alias mutations are visible immediately, and
    // there is no entry/exit storage copy + drain pair).  Both exact
    // dicts and arbitrary `__getitem__`-bearing mappings now share
    // this path.
    //
    // pypy/interpreter/pyopcode.py:2015 ensure_ns — when the caller
    // omits both globals and locals, exec falls back to caller globals
    // (already wired above) AND caller `getdictscope()`.  When the
    // caller omits ONLY locals, locals collapse to globals (PyPy
    // `pyopcode.py:2010-2013`), which the existing same-storage shape
    // below covers via the `is_none_or_null(locals_arg)` skip.
    //
    // Resolve the implicit caller-locals only when globals_arg is also
    // None: that's the `exec(src)` shape where PyPy hands the caller's
    // live local mapping in via `frame.getdictscope()`.  When
    // globals_arg is supplied but locals_arg is None, PyPy collapses
    // locals=globals and pyre's existing same-dict path handles it.
    let mut implicit_caller_locals: pyre_object::PyObjectRef = std::ptr::null_mut();
    if is_none_or_null(globals_arg) && is_none_or_null(locals_arg) && !caller_frame.is_null() {
        // pyframe.py:540 getdictscope returns the caller's
        // w_locals (PyObjectRef) — same dict-or-mapping the
        // interpreter sees inside the calling function body.
        implicit_caller_locals = unsafe { (*caller_frame).getdictscope()? };
    }
    let mut locals_object_arg: pyre_object::PyObjectRef = std::ptr::null_mut();
    if !is_none_or_null(locals_arg) {
        let same_as_globals =
            !is_none_or_null(globals_arg) && std::ptr::eq(locals_arg, globals_arg);
        if !same_as_globals {
            // Dict and non-dict mapping arms share the
            // setdictscope path — for exact dict locals this
            // matches PyPy's `code.exec_code(space, w_globals,
            // w_locals)` chain (pyopcode.py:776) which feeds
            // `space.setitem(w_locals, name, value)` to STORE_NAME.
            // Pyre's earlier `is_dict_w` arm built a storage copy and
            // drained it back through a `Vec<String>` snapshot to
            // mirror DELETE_GLOBAL while preserving alias mutations;
            // routing through `setdictscope` retires the copy +
            // snapshot entirely.
            locals_object_arg = locals_arg;
        }
    }
    // eval.py:31-33 Code.exec_code → space.createframe(...) + frame.run().
    // For eval() with a code object that carries freevars, createframe
    // surfaces pyframe.py:242-246's TypeError "directly executed code
    // object may not contain free variables" directly — exec()'s
    // closure-mismatch TypeError was already raised above.
    let mut frame =
        match crate::createframe_obj(code_obj_ref as *const (), w_globals, exec_ctx, None) {
            Ok(frame) => frame,
            Err(err) => {
                let _ = raw_code;
                return Err(err);
            }
        };
    frame.fix_array_ptrs();
    // eval.py:32 frame.setdictscope(w_locals, ...) — only when locals
    // were separately supplied.  Without this call, initialize_frame_scopes'
    // module-code arm has already bound w_locals = w_globals, matching
    // PyPy's `exec(src, g)` (and `exec(src, g, l)` where `l is g`).
    if !locals_object_arg.is_null() {
        frame.setdictscope(locals_object_arg)?;
    } else if !implicit_caller_locals.is_null() {
        // pyopcode.py:2015 — `exec(src)` with no globals/locals uses
        // the caller's `getdictscope()` as locals.  Skip when the
        // resolved object is the caller's globals (module-level exec
        // collapses to locals=globals — same-dict shape kept by the
        // module-frame's initialize_frame_scopes binding).
        let caller_globals_obj = unsafe { (*caller_frame).get_w_globals() };
        let same_as_globals = !caller_globals_obj.is_null()
            && std::ptr::eq(implicit_caller_locals, caller_globals_obj);
        if !same_as_globals {
            frame.setdictscope(implicit_caller_locals)?;
        }
    }
    // run_with_jit rather than execute_frame so that
    // `eval(compile("(x for x in [])", ..., 'eval'))` of generator-flagged
    // code returns the wrapped generator object instead of executing the
    // body inline, and so the exec'd body reaches the JIT portal (a hot loop
    // inside exec()'d source warms into a trace like any function would).
    // STORE_GLOBAL / DELETE_GLOBAL writes during the run land on the
    // storage proxy and back-mirror to the dict object, so the user dict
    // and the frame's globals stay one and the same throughout the run
    // (pyopcode.py:771-776 parity — no entry/exit drain needed).
    let result = frame.run_with_jit();

    let _ = raw_code; // keep raw_code alive until after exec for safety.
    match result {
        Ok(v) if is_eval => Ok(v),
        Ok(_) => Ok(pyre_object::w_none()),
        Err(e) => Err(e),
    }
}

fn builtin_globals(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if !args.is_empty() {
        return Err(crate::PyError::type_error("globals() takes no arguments"));
    }
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "globals() requires an active frame",
            ));
        }
        // `pypy/module/__builtin__/interp_inspect.py:5 globals_w` →
        // `caller.get_w_globals_storage()` returns the dict directly without
        // wrapping.  PyPy keeps a single dict per module so subsequent
        // `globals()` / `frame.f_globals` / `f.__globals__` /
        // `module.__dict__` accesses on the same module share one
        // identity.  Pyre routes through the lazy cached
        // `get_w_globals` (Step 1 of the w_globals type
        // migration) which returns the canonical W_DictObject paired
        // with the frame's storage — same identity invariant via
        // `dict_storage_to_dict`'s mirror_target.  Returning a fresh
        // wrapper per call (as the previous shape did) silently
        // diverged on `globals() is module.__dict__`.
        let dict = unsafe { (*frame).get_w_globals() };
        if dict.is_null() {
            return Err(crate::PyError::runtime_error(
                "globals() requires an active frame",
            ));
        }
        Ok(dict)
    })
}

fn builtin_locals(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if !args.is_empty() {
        return Err(crate::PyError::type_error("locals() takes no arguments"));
    }
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "locals() requires an active frame",
            ));
        }
        // `interp_inspect.py:7-11 locals` returns
        // `ec.gettopframe_nohidden().getdictscope()` unconditionally.
        // `getdictscope` (`pyframe.py:525-530`) always runs `fast2locals()`
        // before returning `debugdata.w_locals`, so a second `locals()`
        // re-syncs the mapping with the current fast locals —
        // `x = 1; locals(); x = 2; locals()["x"]` reads `2`.  `fast2locals`
        // lazily allocates and caches the mapping on first call, so identity
        // holds (`locals() is locals()`, and `locals() is globals()` at
        // module scope where `debugdata.w_locals is w_globals`); for a
        // non-dict exec/eval mapping it returns that live object and writes
        // through its `__setitem__`.
        let frame_mut = unsafe { &mut *frame };
        frame_mut.getdictscope()
    })
}

fn builtin_vars(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "vars() takes no keyword arguments",
        ));
    }
    if args.is_empty() {
        return builtin_locals(args);
    }
    if args.len() != 1 {
        return Err(crate::PyError::type_error(
            "vars() takes at most 1 argument.",
        ));
    }
    let obj = args[0];
    let has_dict = unsafe {
        pyre_object::is_instance(obj)
            || pyre_object::is_type(obj)
            || crate::is_function(obj)
            || pyre_object::is_module(obj)
    };
    if !has_dict {
        return Err(crate::PyError::type_error(
            "vars() argument must have __dict__ attribute",
        ));
    }
    let dict = crate::baseobjspace::getattr_str(obj, "__dict__")
        .map_err(|_| crate::PyError::type_error("vars() argument must have __dict__ attribute"))?;
    if dict.is_null() || unsafe { pyre_object::is_none(dict) } {
        return Err(crate::PyError::type_error(
            "vars() argument must have __dict__ attribute",
        ));
    }
    Ok(dict)
}

/// util.py:62 `_classdir` — union `getattr(klass, '__dict__')`'s keys with,
/// recursively, `_classdir(base)` for each base in `getattr(klass,
/// '__bases__')`.  Both attributes are read through the attribute protocol so
/// a metaclass that customizes `__dict__`/`__bases__` access participates.
unsafe fn classdir_into(
    w_cls: PyObjectRef,
    names: &mut Vec<Wtf8Buf>,
) -> Result<(), crate::PyError> {
    unsafe { classdir_recurse(w_cls, names) }
}

unsafe fn classdir_recurse(
    w_cls: PyObjectRef,
    names: &mut Vec<Wtf8Buf>,
) -> Result<(), crate::PyError> {
    // getattr(klass, '__dict__', None): names.update(ns).  This is deliberately
    // iterable-driven, not dict-only: app-level PyPy accepts any iterable here.
    match crate::baseobjspace::getattr_str(w_cls, "__dict__") {
        Ok(w_ns) if !w_ns.is_null() && !unsafe { pyre_object::is_none(w_ns) } => {
            for k in collect_iterable(w_ns)? {
                if unsafe { pyre_object::is_str(k) } {
                    names.push(unsafe { pyre_object::w_str_get_wtf8(k) }.to_owned());
                }
            }
        }
        Ok(_) => {}
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
    }

    // getattr(klass, '__bases__', None): for base in bases.
    match crate::baseobjspace::getattr_str(w_cls, "__bases__") {
        Ok(bases) if !bases.is_null() && !unsafe { pyre_object::is_none(bases) } => {
            for base in collect_iterable(bases)? {
                unsafe { classdir_recurse(base, names) }?;
            }
        }
        Ok(_) => {}
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
    }
    Ok(())
}

/// `dir([obj])` — PyPy: pypy/module/__builtin__/app_inspect.py dir
///
/// Without argument: names in the current local scope (not supported).
/// With argument: sorted list of attribute names from obj.__dict__ plus
/// type MRO. Modules expose their namespace via w_module_get_namespace.
pub(crate) fn builtin_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "dir() takes no keyword arguments",
        ));
    }
    if args.is_empty() {
        // `bltinmodule.c builtin_dir` — with no argument, list the names in
        // the caller's local scope: `sorted(frame.f_locals)`.  Resolve the
        // locals mapping exactly as `locals()` does (module scope returns
        // the globals dict), then return its sorted keys.
        return crate::eval::CURRENT_FRAME.with(|current| {
            let frame = current.get();
            if frame.is_null() {
                return Ok(w_list_new(vec![]));
            }
            let frame_mut = unsafe { &mut *frame };
            let w_locals_dict = frame_mut.getdictscope()?;
            if w_locals_dict.is_null() {
                return Ok(w_list_new(vec![]));
            }
            let keys_iter = crate::baseobjspace::iter(w_locals_dict)?;
            let keys = collect_iterable(keys_iter)?;
            builtin_sorted(&[w_list_new(keys)])
        });
    }
    if args.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "dir expected at most 1 argument, got {}",
            args.len()
        )));
    }
    let obj = args[0];
    // app_inspect.py:57-62 — dir() is driven by the object's `__dir__`:
    // `lookup_special(obj, '__dir__')` then `sorted(result)`.  pyre's builtin
    // types do not register a default `__dir__` slot, so the manual
    // enumeration below stands in for the default object / type / module
    // `__dir__`; a `__dir__` found on the type here is a user override (or a
    // builtin such as traceback) and drives dir() directly.
    if let Some(w_type) = crate::typedef::r#type(obj) {
        if let Some(dir_meth) =
            unsafe { crate::baseobjspace::lookup_in_type_where(w_type, "__dir__") }
        {
            let result =
                unsafe { crate::baseobjspace::get_and_call_function(dir_meth, obj, w_type, &[]) }?;
            return builtin_sorted(&[result]);
        }
    }
    // `GenericAlias.__dir__` (`_pypy_generic_alias.py:85`) merges the alias's
    // own attribute names with `dir(__origin__)`.
    if unsafe { pyre_object::is_generic_alias(obj) } {
        return crate::_pypy_generic_alias::dir_list(obj);
    }
    let mut names: Vec<Wtf8Buf> = Vec::new();
    unsafe {
        if pyre_object::is_module(obj) {
            // Route through `w_module.w_dict` so dict-subclass-backed
            // Modules (`pypy/module/__builtin__/moduledef.py:102-103
            // Module(space, None, w_builtin)`) surface their entries
            // alongside storage-backed modules.  PyPy
            // `pypy/interpreter/module.py:77 Module.getdict()` returns
            // the dict directly regardless of subclass; pyre branches
            // on the underlying shape:
            //   - exact `W_DictObject` → `w_dict_str_entries_wtf8` returns
            //     the storage-proxy union view in one call, keeping
            //     lone-surrogate global names.
            //   - dict subclass instance → iterate keys via the
            //     standard `iter()` protocol so the subclass's
            //     `__iter__` override participates (PyPy's
            //     `space.iter(w_dict)` would do the same).
            let w_dict = pyre_object::w_module_get_w_dict(obj);
            if !w_dict.is_null() {
                // module.py:163 descr_module__dir__ — a `__dir__` stored in the
                // module's own dict drives dir() (called with no arguments);
                // otherwise the dict keys are listed.
                if let Some(mod_dir) = crate::baseobjspace::finditem_str(w_dict, "__dir__")? {
                    if !mod_dir.is_null() {
                        let result = crate::call::call_function_impl_result(mod_dir, &[])?;
                        return builtin_sorted(&[result]);
                    }
                }
                if pyre_object::is_dict(w_dict) {
                    for (name, _) in pyre_object::dictmultiobject::w_dict_str_entries_wtf8(w_dict) {
                        names.push(name);
                    }
                } else if let Ok(keys_iter) = crate::baseobjspace::iter(w_dict) {
                    if let Ok(keys) = crate::builtins::collect_iterable(keys_iter) {
                        for k in keys {
                            if pyre_object::is_str(k) {
                                names.push(pyre_object::w_str_get_wtf8(k).to_owned());
                            }
                        }
                    }
                }
            }
        } else if pyre_object::is_type(obj) {
            // util.py:62 `_classdir` (`type.__dir__`, typeobject.py:1234) —
            // the class's `__dict__` keys unioned with `_classdir` of each
            // base, recursively.
            classdir_into(obj, &mut names)?;
        } else if pyre_object::is_instance(obj) {
            // util.py:80 `_objectdir` (`object.__dir__`) — the instance
            // `__dict__` keys plus `_classdir(type(obj))`.  The instance dict
            // for hasdict objects is the live W_DictObject returned by
            // `w_obj.getdict(space)`; `__slots__` Member descriptor names live
            // in the class namespaces walked by `classdir_into`, so every slot
            // is listed regardless of whether it currently holds a value.
            let w_dict = crate::baseobjspace::getdict(obj);
            if !w_dict.is_null() {
                for (k, _) in pyre_object::w_dict_items(w_dict) {
                    if pyre_object::is_str(k) {
                        names.push(pyre_object::w_str_get_wtf8(k).to_owned());
                    }
                }
            }
            let w_type = pyre_object::w_instance_get_type(obj);
            if !w_type.is_null() && pyre_object::is_type(w_type) {
                classdir_into(w_type, &mut names)?;
            }
        } else {
            // Fallback `_objectdir` (util.py:80) for builtin W_Root types
            // (PyTraceback, dict, dict view, etc.) that have no instance dict:
            // `_classdir` of their type.  Excluded for module/instance/type
            // above because those have richer paths that combine instance and
            // class entries.
            if let Some(w_type) = crate::typedef::r#type(obj) {
                if pyre_object::is_type(w_type) {
                    classdir_into(w_type, &mut names)?;
                }
            }
        }
    }
    names.sort();
    names.dedup();
    let items: Vec<_> = names
        .into_iter()
        .map(pyre_object::w_str_from_wtf8)
        .collect();
    Ok(w_list_new(items))
}

/// `id(obj)` — PyPy: baseobjspace.py id → object identity as int
fn builtin_id(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "id() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    // `space.id` (baseobjspace.py:843-854): a plain `int` yields its
    // value-derived `immutable_unique_id`; every other object falls back
    // to `compute_unique_id` — its address.
    let obj = args[0];
    Ok(match crate::function::immutable_unique_id(obj) {
        Some(w_id) => w_id,
        None => w_int_new(obj as usize as i64),
    })
}

/// `hash(obj)` — PyPy: `descroperation.py:1006 hash`.
///
/// CPython / PyPy raise `TypeError: unhashable type: 'X'` when the
/// object's class lacks a non-None `__hash__` slot.  Built-in
/// mutable containers (dict, list, set, bytearray) explicitly set
/// `__hash__ = None` (`dictmultiobject.py:1431`, `listobject.py`,
/// `setobject.py`).  `try_hash_value` is the Result-bearing variant
/// used by both `hash()` and dict key gates: it rejects known
/// unhashables, recurses through tuple/frozenset contents, and
/// propagates user `__hash__` errors.
pub(crate) fn builtin_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "hash() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Ok(w_int_new(try_hash_value(args[0])?))
}

pub fn try_hash_value(obj: PyObjectRef) -> Result<i64, crate::PyError> {
    if obj.is_null() {
        return Err(crate::PyError::type_error("hash() argument is null"));
    }
    unsafe {
        let kind = if pyre_object::is_dict(obj) {
            Some("dict")
        } else if pyre_object::is_list(obj) {
            Some("list")
        } else if pyre_object::is_set(obj) {
            // `frozenset` is hashable per setobject.py:623-642 _hash_frozenset.
            Some("set")
        } else if pyre_object::is_bytearray(obj) {
            Some("bytearray")
        } else if pyre_object::dictmultiobject::is_dict_view(obj) {
            // `dictmultiobject.py:1619 _is_set_like` views inherit set's
            // unhashable semantics; values view also isn't hashable.
            Some("dict view")
        } else if pyre_object::sliceobject::is_slice(obj) {
            // sliceobject.py:205 `__hash__ = None`.
            Some("slice")
        } else {
            None
        };
        if let Some(name) = kind {
            return Err(crate::PyError::type_error(&format!(
                "unhashable type: '{}'",
                name
            )));
        }
        if pyre_object::memoryview::is_w_memoryview(obj) {
            // `descr_hash` — released views and writable views are
            // unhashable; a read-only view hashes its raw bytes.
            memoryview_check_released(obj)?;
            if !pyre_object::memoryview::w_memoryview_readonly(obj) {
                return Err(crate::PyError::value_error(
                    "cannot hash writable memoryview object",
                ));
            }
            return Ok(_hash_str(&memoryview_gather_bytes(obj)));
        }
        if is_tuple(obj) {
            let n = w_tuple_len(obj);
            let mut hashes = Vec::with_capacity(n);
            for i in 0..(n as i64) {
                if let Some(item) = w_tuple_getitem(obj, i) {
                    hashes.push(try_hash_value(item)?);
                }
            }
            return Ok(_hash_tuple_xx(&hashes));
        }
        if pyre_object::is_frozenset(obj) {
            let hashes: Result<Vec<i64>, crate::PyError> = pyre_object::w_set_items(obj)
                .into_iter()
                .map(try_hash_value)
                .collect();
            return Ok(_hash_frozenset(&hashes?));
        }
        if pyre_object::is_generic_alias(obj) {
            // GenericAlias.__hash__ (`_pypy_generic_alias.py:82`) —
            // `hash(self.__origin__) ^ hash(self.__args__)`.  Routed through
            // `try_hash_value` so an unhashable element in `__args__`
            // surfaces its TypeError instead of being swallowed.
            let origin = pyre_object::w_generic_alias_get_origin(obj);
            let args = pyre_object::w_generic_alias_get_args(obj);
            return Ok(try_hash_value(origin)? ^ try_hash_value(args)?);
        }
        if pyre_object::is_union(obj) {
            // UnionType.__hash__ (`_pypy_generic_alias.py:275`) —
            // `hash(frozenset(self.__args__))`, order-independent so it
            // agrees with `__eq__`'s set equality.
            let args = pyre_object::w_union_get_args(obj);
            let n = pyre_object::w_tuple_len(args);
            let mut members = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(item) = pyre_object::w_tuple_getitem(args, i as i64) {
                    members.push(item);
                }
            }
            return try_hash_value(pyre_object::w_frozenset_from_items(&members));
        }
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__hash__") {
                if pyre_object::is_none(method) {
                    return Err(unhashable_type_error(obj));
                }
                let r = call_and_check(method, &[obj])?;
                // descroperation.py:576-579 — normalize -1 to -2
                let h = if is_bool(r) {
                    pyre_object::w_bool_get_value(r) as i64
                } else if is_int(r) {
                    w_int_get_value(r)
                } else if is_long(r) {
                    _hash_long(pyre_object::w_long_get_value(r))
                } else {
                    return Err(crate::PyError::type_error(
                        "__hash__ method should return an integer",
                    ));
                };
                return Ok(if h == -1 { -2 } else { h });
            }
        }
    }
    Ok(hash_value(obj))
}

fn unhashable_type_error(obj: PyObjectRef) -> crate::PyError {
    let name = unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None if !obj.is_null() => (*(*obj).ob_type).name.to_string(),
            None => "NULL".to_string(),
        }
    };
    crate::PyError::type_error(format!("unhashable type: '{}'", name))
}

/// `pypy/objspace/std/intobject.py:36-37` — `HASH_BITS = 61` (64-bit
/// host); `HASH_MODULUS = 2**HASH_BITS - 1`.  The Mersenne-prime
/// modulus is what makes pyre's `hash(42) == hash(42.0) ==
/// hash(2**100 + 42)`-class invariants hold: every per-type hash
/// reduces its input modulo the same `HASH_MODULUS`, so equal
/// numeric values land on the same residue.
const HASH_BITS: u32 = 61;
const HASH_MODULUS: u64 = (1u64 << HASH_BITS) - 1;
/// `floatobject.py:29-30` HASH_NAN sentinel.
const HASH_NAN: i64 = 0;

/// Numeric hash of a machine-word integer: reduce `a` modulo the
/// Mersenne prime `HASH_MODULUS = 2**61 - 1` (the residue keeps `a`'s
/// sign) and bump a `-1` result to `-2`.  Delegated to
/// `rustpython_common::hash`; `mod_int` is `value % HASH_MODULUS` and
/// `fix_sentinel` is the `-1 -> -2` guard.  Shares the reduction with
/// `_hash_long`, so `hash(42) == hash(2**100 + 42)`-class invariants
/// hold.  `mod_int`/`fix_sentinel` are `const fn`, so this inlines to
/// the same arithmetic the hand-rolled port produced.
#[inline]
pub(crate) fn _hash_int(a: i64) -> i64 {
    use rustpython_common::hash;
    hash::fix_sentinel(hash::mod_int(a))
}

/// Numeric hash of an arbitrary-precision integer: `value` reduced
/// modulo the Mersenne prime `HASH_MODULUS = 2**61 - 1` (the residue
/// keeps the sign), with a `-1` result bumped to `-2`.  Delegated to
/// `rustpython_common::hash::hash_bigint`.  Because the modulus is a
/// Mersenne prime the residue is independent of the digit base, so
/// `_hash_int(v) == _hash_long(BigInt::from(v))` for any `v` that fits
/// a machine word.
#[inline]
pub(crate) fn _hash_long(v: &BigInt) -> i64 {
    rustpython_common::hash::hash_bigint(v)
}

/// Numeric hash of a `float`.  `rustpython_common::hash::hash_float`
/// reduces the mantissa/exponent modulo the Mersenne prime
/// `HASH_MODULUS = 2**61 - 1` (keeping the sign), so `hash(2.0) == hash(2)`
/// and the `±inf` sentinels are `±314159`; subnormals decompose exactly.
/// It returns `None` for NaN, and the sole caller (`hash_value`) reaches
/// here without a prior NaN check, so map that to `HASH_NAN`.
#[inline]
pub(crate) fn _hash_float(v: f64) -> i64 {
    rustpython_common::hash::hash_float(v).unwrap_or(HASH_NAN)
}

/// `tupleobject.py:358-401 descr_hash` — the xxHash sequence hash, delegated
/// to `rustpython_common::hash::hash_tuple`.  The caller has already computed
/// each element's hash into `items`, so the fold is infallible here.
///
/// The shared fold reproduces the accumulator loop and length mangle exactly.
/// It also corrects the `acc == (uint)-1` sentinel: `tupleobject.py:403`
/// computes `acc += (acc == -1) * (1546275796 + 1)`, so a wrapped `acc` of
/// `-1` becomes `1546275796` — the value CPython's `tuplehash` also returns.
/// The prior local port set the result to `1546275796 + 1` instead of adding
/// to `acc`, an off-by-one in that (2**-64-probability) case; delegation fixes
/// it.
#[inline]
fn _hash_tuple_xx(items: &[i64]) -> i64 {
    rustpython_common::hash::hash_tuple(items.iter().map(|&h| Ok::<i64, ()>(h)))
        .expect("element hashes are precomputed, so the fold cannot fail")
}

/// `pypy/objspace/std/unicodeobject.py:341-345 W_UnicodeObject.hash_w`
/// parity:
///
/// ```python
/// def hash_w(self):
///     x = compute_hash(self._utf8)
///     x -= (x == -1)
///     return x
/// ```
///
/// `compute_hash` is `rpython.rlib.objectmodel.compute_hash` —
/// on 64-bit hosts it delegates to `rpython.rlib.rsiphash.siphash24`
/// with a 16-byte secret key set via `rsiphash.choose_initial_seed`
/// (rpython/rlib/rsiphash.py:48).  The seed is read from
/// `PYTHONHASHSEED`, defaulting to a randomised value at process
/// start (CPython parity: `Random_Hash_Function_Seed_String`).
///
/// Pyre uses a fixed 16-byte key here so test runs are deterministic
/// (matching `PYTHONHASHSEED=0`).  Switching to a randomised seed
/// is straight-forward (`OnceLock<[u8; 16]>` seeded from
/// `getrandom` or the env var) once tests are robust to it.
/// Hash a string by its WTF-8 bytes — `unicodeobject.py descr_hash` hashes
/// `self._utf8`, so a lone-surrogate string hashes by its byte sequence
/// instead of panicking on the `&str` view.
fn _hash_str(bytes: &[u8]) -> i64 {
    use core::hash::Hasher;
    // Empty input hashes to 0 (`""` and `b""`), short-circuiting the
    // siphash digest.
    if bytes.is_empty() {
        return 0;
    }
    // `rpython/rlib/rsiphash.py:60-62 _build_key_from_seed` — when
    // `PYTHONHASHSEED=0` the key is the 16-byte all-zero buffer.
    // Pyre runs with the deterministic seed for reproducibility,
    // matching PyPy's `PYTHONHASHSEED=0` byte-for-byte.  Wiring a
    // user-overridable seed is straight-forward (`OnceLock<[u8; 16]>`
    // sampled from `getrandom` or the env var) once tests are
    // robust to it.
    //
    // Not delegated to `rustpython_common::hash::hash_str`: that path
    // needs a `HashSecret`, whose all-zero key is un-constructable at
    // the pinned rev (private `k0`/`k1`, only a seeded `new`), and it
    // hashes through the slice `Hash` impl (length prefix) plus a
    // `mod_int` reduction that this raw siphash24 digest omits.
    static SECRET: [u8; 16] = [0u8; 16];
    let mut hasher = siphasher::sip::SipHasher24::new_with_key(&SECRET);
    hasher.write(bytes);
    let raw = hasher.finish() as i64;
    raw - ((raw == -1) as i64)
}

/// `space.hash_w` digest for a `str` computed directly from its WTF-8 bytes
/// — the value [`hash_value`] returns for a str key — exposed for the
/// `dict_eq_hook::HASH_STR_HOOK` trampoline so str-keyed dict GET probes hash
/// without a `W_UnicodeObject`.
#[inline]
pub fn hash_str_bytes(bytes: &[u8]) -> i64 {
    _hash_str(bytes)
}

/// `setobject.py:623-642 W_FrozensetObject.descr_hash` — the order-independent
/// XOR-fold, delegated to `rustpython_common::hash::FrozenSetHash`.  The caller
/// has already computed each element's hash into `items`.
///
/// The shared accumulator is bit-identical: its `shuffle_bits`
/// `((h ^ 89869747) ^ (h << 16)) * 3644798167` equals the port's
/// `(h ^ (h << 16) ^ 89869747) * (1822399083 * 2 + 1)` (xor is associative;
/// `3644798167 == 1822399083 * 2 + 1`), and the seed `(len + 1) * 1927868237`,
/// the final dispersion, and the `-1 -> 590923713` sentinel all match.
#[inline]
fn _hash_frozenset(items: &[i64]) -> i64 {
    let mut acc = rustpython_common::hash::FrozenSetHash::new(items.len());
    for &item_hash in items {
        acc.add(item_hash);
    }
    acc.finish()
}

/// `pypy/objspace/std/objspace.py StdObjSpace.hash` parity — share one
/// implementation across builtin `hash()`, dict / set lookup, and
/// tuple/frozenset content hashing.  Dispatches to PyPy's per-type
/// hash helpers (`_hash_int`/`_hash_long`/`_hash_float`/
/// `_hash_tuple_xx`/`_hash_frozenset`), so:
///
/// - `hash(42) == hash(42.0) == hash(2**100 + 42)` (Mersenne mod)
/// - `hash((1, 2)) == hash((1, 2))` regardless of allocation identity
/// - `hash(frozenset(...))` is deterministic and order-independent
///
/// `unicodeobject.py W_UnicodeObject.descr_hash` routes through
/// RPython's `compute_hash(self._utf8)` which is siphash on 64-bit;
/// pyre keeps an FNV-style multiplicative mix here (functional but
/// not bit-identical to CPython/PyPy).  Convergence target: import
/// siphash24 from a workspace dep.
pub fn hash_value(obj: PyObjectRef) -> i64 {
    unsafe {
        // `is_int` is true for a bool (`BOOL_TYPE`), so test `is_bool` first.
        if is_bool(obj) {
            return if pyre_object::w_bool_get_value(obj) {
                1
            } else {
                0
            };
        }
        if is_int(obj) {
            return _hash_int(w_int_get_value(obj));
        }
        if is_long(obj) {
            return _hash_long(pyre_object::w_long_get_value(obj));
        }
        if is_float(obj) {
            return _hash_float(pyre_object::w_float_get_value(obj));
        }
        if pyre_object::is_complex(obj) {
            return crate::objspace::descroperation::complex_hash(
                pyre_object::w_complex_get_real(obj),
                pyre_object::w_complex_get_imag(obj),
            );
        }
        if is_str(obj) {
            return _hash_str(pyre_object::w_str_get_wtf8(obj).as_bytes());
        }
        // `bytesobject.py descr_hash` — `compute_hash(self._value)`, the same
        // byte-string digest str uses (bytearray is mutable / unhashable).
        if pyre_object::is_bytes(obj) {
            return _hash_str(pyre_object::bytesobject::w_bytes_data(obj));
        }
        // `memoryobject.py descr_hash` — `compute_hash(self.view.as_str())`;
        // a released or writable view is unhashable, so this infallible
        // hasher only digests a live read-only view and otherwise falls
        // through to the unhashable tail (the fallible `try_hash_value`
        // raises the proper ValueError).
        if pyre_object::memoryview::is_w_memoryview(obj) {
            if memoryview_check_released(obj).is_ok()
                && pyre_object::memoryview::w_memoryview_readonly(obj)
            {
                return _hash_str(&memoryview_gather_bytes(obj));
            }
        }
        if pyre_object::is_none(obj) {
            return 0;
        }
        if is_tuple(obj) {
            let n = w_tuple_len(obj);
            let mut hashes = Vec::with_capacity(n);
            for i in 0..(n as i64) {
                if let Some(item) = w_tuple_getitem(obj, i) {
                    hashes.push(hash_value(item));
                }
            }
            return _hash_tuple_xx(&hashes);
        }
        if pyre_object::is_frozenset(obj) {
            let hashes: Vec<i64> = pyre_object::w_set_items(obj)
                .into_iter()
                .map(hash_value)
                .collect();
            return _hash_frozenset(&hashes);
        }
        if pyre_object::is_w_range(obj) {
            // `descr_hash` — `hash((length, start|None, step|None))` so two
            // ranges denoting the same sequence hash equally.
            let w_len = pyre_object::w_range_length(obj);
            let (start, _stop, step) = pyre_object::w_range_fields(obj);
            let len_b = pyre_object::range_obj_to_bigint(w_len);
            let none = w_none();
            let (a, b) = if len_b == BigInt::from(0) {
                (none, none)
            } else if len_b == BigInt::from(1) {
                (start, none)
            } else {
                (start, step)
            };
            let tup = pyre_object::w_tuple_new(vec![w_len, a, b]);
            return hash_value(tup);
        }
        if pyre_object::is_generic_alias(obj) {
            // GenericAlias.__hash__ (`_pypy_generic_alias.py:82`) —
            // `hash(self.__origin__) ^ hash(self.__args__)`.  Resolved
            // here because `hash_w` does not consult a typedef `__hash__`
            // for builtin W_Roots.
            let origin = pyre_object::w_generic_alias_get_origin(obj);
            let args = pyre_object::w_generic_alias_get_args(obj);
            return hash_value(origin) ^ hash_value(args);
        }
        if pyre_object::is_union(obj) {
            // UnionType.__hash__ (`_pypy_generic_alias.py:275`) —
            // `hash(frozenset(self.__args__))`.  Resolved here too so the
            // infallible `hash_w`/`hash_value` path agrees with the
            // fallible `try_hash_value` one.
            let args = pyre_object::w_union_get_args(obj);
            let n = w_tuple_len(args);
            let mut members = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(args, i as i64) {
                    members.push(item);
                }
            }
            return hash_value(pyre_object::w_frozenset_from_items(&members));
        }
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__hash__") {
                let r = crate::call_function(method, &[obj]);
                if !r.is_null() && is_int(r) {
                    return w_int_get_value(r);
                }
            }
        }
        obj as i64
    }
}

/// `ord(c)` — PyPy: operation.py ord (dispatches to space.ord);
/// `unicodeobject.py:155-160` raises TypeError on multi-char strings.
fn builtin_ord(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(
            "ord() takes exactly one argument",
        ));
    }
    let obj = args[0];
    unsafe {
        if is_str(obj) {
            // Read the code point through the WTF-8 view so a lone-surrogate
            // single-character string yields its ordinal (0xD800-0xDFFF).
            let count = w_str_len(obj);
            if count != 1 {
                return Err(crate::PyError::type_error(format!(
                    "ord() expected a character, but string of length {count} found"
                )));
            }
            let cp = w_str_get_wtf8(obj).code_points().next().unwrap();
            return Ok(w_int_new(cp.to_u32() as i64));
        }
        // bytesobject.py:464 — bytes of length 1
        if pyre_object::bytesobject::is_bytes_like(obj) {
            let data = pyre_object::bytesobject::bytes_like_data(obj);
            if data.len() != 1 {
                return Err(crate::PyError::type_error(format!(
                    "ord() expected a character, but string of length {} found",
                    data.len()
                )));
            }
            return Ok(w_int_new(data[0] as i64));
        }
    }
    Err(crate::PyError::type_error(
        "ord() expected string of length 1, but other type found",
    ))
}

/// `chr(i)` — PyPy: operation.py chr
fn builtin_chr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "chr() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let obj = args[0];
    // operation.py:28 — space.int_w unwraps to int
    let val = if unsafe { is_int(obj) } {
        unsafe { w_int_get_value(obj) }
    } else {
        // int subclass instance — check __int_value__ via builtin_int
        match builtin_int(args) {
            Ok(v) if unsafe { is_int(v) } => unsafe { w_int_get_value(v) },
            _ => {
                return Err(crate::PyError::type_error(
                    "an integer is required (got type non-int)",
                ));
            }
        }
    };
    if val < 0 || val > 0x10ffff {
        // `pypy/module/__builtin__/operation.py:31-32 chr` — out-of-range
        // raises ValueError, message "chr() arg out of range".
        return Err(crate::PyError::value_error("chr() arg out of range"));
    }
    match char::from_u32(val as u32) {
        Some(c) => Ok(w_str_new(&c.to_string())),
        // Surrogate code points (0xD800-0xDFFF) are valid chr() arguments and
        // produce a lone-surrogate string; char::from_u32 rejects them, so
        // build the string through a WTF-8 code point instead.
        None => {
            let cp = CodePoint::from_u32(val as u32)
                .expect("val is in 0..=0x10ffff per the range check above");
            let mut one = Wtf8Buf::new();
            one.push(cp);
            Ok(w_str_from_wtf8(one))
        }
    }
}

/// `filter(function or None, iterable)` — `functional.py:980-995
/// W_Filter___new__`.  A lazy iterator: `function == None` keeps truthy
/// items, otherwise `function(item)` is the predicate.
pub(crate) fn builtin_filter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "filter expected 2 arguments, got {}",
            args.len()
        )));
    }
    let func = args[0];
    // `functional.py:921-924` — a None predicate is stored as PY_NULL.
    let w_predicate = if unsafe { pyre_object::is_none(func) } {
        pyre_object::PY_NULL
    } else {
        func
    };
    // `functional.py:925 self.w_iterable = space.iter(w_iterable)`.
    let w_iterable = crate::baseobjspace::iter(args[1])?;
    Ok(pyre_object::functional::w_filter_new(
        w_predicate,
        w_iterable,
    ))
}

/// `map(func, *iterables, strict=False)` — `functional.py:888-902
/// W_Map___new__` plus the CPython 3.14 `strict` keyword.  A lazy iterator:
/// each `next()` pulls one item per iterable and calls `func(*items)`.
pub(crate) fn builtin_map(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["strict"], "map")?;
    let strict = kwarg_get(kwargs, "strict")
        .map(|v| crate::baseobjspace::is_true(v))
        .transpose()?
        .unwrap_or(false);
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "map() must have at least two arguments.",
        ));
    }
    let func = args[0];
    // `functional.py:835-836 build_iterators_from_args` — `iter()` each input.
    let mut iters = Vec::with_capacity(args.len() - 1);
    for &arg in &args[1..] {
        iters.push(crate::baseobjspace::iter(arg)?);
    }
    let w_iterators = pyre_object::w_list_new(iters);
    Ok(pyre_object::functional::w_map_new(
        func,
        w_iterators,
        strict,
    ))
}

/// `zip(*iterables, strict=False)` — `functional.py:1101-1105 W_Zip___new__`.
/// A lazy iterator: each `next()` pulls one item per iterable into a tuple,
/// stopping at the shortest (an empty `zip()` stops immediately); `strict`
/// raises `ValueError` on a length mismatch.
pub(crate) fn builtin_zip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // Pyre's flat builtin ABI surfaces kwargs as a trailing dict; strip it
    // before the positional walk and look up `strict` from it.
    let (args, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["strict"], "zip")?;
    let strict = kwarg_get(kwargs, "strict")
        .map(|v| crate::baseobjspace::is_true(v))
        .transpose()?
        .unwrap_or(false);
    // `functional.py:835-836 build_iterators_from_args` — `iter()` each input.
    let mut iters = Vec::with_capacity(args.len());
    for &arg in args {
        iters.push(crate::baseobjspace::iter(arg)?);
    }
    let w_iterators = pyre_object::w_list_new(iters);
    Ok(pyre_object::functional::w_zip_new(w_iterators, strict))
}

/// `pypy/module/__builtin__/functional.py:253-272 W_Enumerate.descr_new`
/// parity:
///
/// ```python
/// def descr_new(space, w_subtype, w_iterable, w_start=None):
///     ...
///     if w_start is None:
///         start = 0
///     else:
///         start = space.index_w(w_start)
///     ...
/// ```
///
/// `space.index_w` accepts ANY object exposing `__index__`
/// (subclasses of int, NumPy ints, etc.) — not just exact int.  The
/// kwarg surface is also strict: anything other than `start=` is a
/// TypeError per the gateway's parsed signature.
// `pypy/module/__builtin__/functional.py:253-275 W_Enumerate.descr___new__`
// line-by-line port — constructs the lazy `W_Enumerate` iterator,
// resolving `start` via `space.index_w` (with overflow promotion to a
// bigint slot) and capturing either the source iterator or the
// source list directly when `start == 0 + isinstance(it, list)`.
pub(crate) fn builtin_enumerate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    if positional.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "enumerate() takes at most 2 arguments ({} given)",
            positional.len()
        )));
    }
    // `iterable` and `start` are the only accepted keywords; an unknown one is
    // reported with the vectorcall-style "invalid keyword argument" message.
    if let Some(dict) = kwargs {
        for (key, _) in unsafe { pyre_object::w_dict_str_entries_wtf8(dict) }.iter() {
            let k = key.as_str().unwrap_or("");
            if k == "__pyre_kw__" || k == "iterable" || k == "start" {
                continue;
            }
            return Err(crate::PyError::type_error(format!(
                "'{key}' is an invalid keyword argument for enumerate()"
            )));
        }
    }
    // `iterable` is positional-or-keyword; once filled positionally a stray
    // `iterable=` keyword is rejected as invalid rather than "multiple values".
    let source = if !positional.is_empty() {
        if kwarg_get(kwargs, "iterable").is_some() {
            return Err(crate::PyError::type_error(
                "'iterable' is an invalid keyword argument for enumerate()",
            ));
        }
        positional[0]
    } else if let Some(iterable) = kwarg_get(kwargs, "iterable") {
        iterable
    } else {
        return Err(crate::PyError::type_error(
            "enumerate() missing required argument 'iterable'",
        ));
    };
    let start_obj = if positional.len() > 1 {
        Some(positional[1])
    } else {
        kwarg_get(kwargs, "start")
    };
    // `functional.py:255-264 descr___new__` — `space.index(w_start)`
    // then `space.int_w(w_start)`; on OverflowError, drop into bigint
    // slot.  Pyre uses i64 directly and would overflow on bigint
    // start; TODO: W_Enumerate
    // can still promote during iteration once start fits in i64).
    let start = match start_obj {
        Some(o) if !unsafe { pyre_object::is_none(o) } => space_index_w(o)?,
        _ => 0,
    };
    // `functional.py:268-271` — `if start == 0 and type(w_iterable) is
    // W_ListObject: w_iter = w_iterable` (skip space.iter for the
    // common list-source case so __next__ can `getitem(index)`
    // directly).  Otherwise call `space.iter(w_iterable)`.
    let w_iter_or_list = if start == 0 && unsafe { pyre_object::is_list(source) } {
        source
    } else {
        crate::baseobjspace::iter(source)?
    };
    Ok(pyre_object::functional::w_enumerate_new(
        w_iter_or_list,
        start,
        pyre_object::PY_NULL, // i64 fast-path active per :225-227
    ))
}

/// `reversed()` — PyPy: functional.py W_ReversedIterator
pub(crate) fn builtin_reversed(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "reversed() takes no keyword arguments",
        ));
    }
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "reversed expected 1 argument, got {}",
            args.len()
        )));
    }
    let obj = args[0];
    unsafe {
        // EXACT builtin list / tuple → a lazy `W_ReversedIterator` over the
        // sequence (`functional.py:354-359 __init__`): `remaining = len - 1` and
        // `descr_next` walks `getitem(seq, remaining)` downward.  A subclass
        // shares the builtin `ob_type` but has its own `w_class`, so it falls
        // through to the `__reversed__` MRO lookup below — CPython honors a
        // subclass override (a non-overriding subclass inherits the builtin
        // `list.__reversed__`, which is the same lazy iterator).
        if pyre_object::is_exact_builtin_instance(obj) {
            if pyre_object::is_list(obj) {
                let n = pyre_object::w_list_len(obj) as i64;
                return Ok(pyre_object::functional::w_reversed_new(obj, n - 1));
            }
            if pyre_object::is_tuple(obj) {
                let n = pyre_object::w_tuple_len(obj) as i64;
                return Ok(pyre_object::functional::w_reversed_new(obj, n - 1));
            }
            // bytes / bytearray expose the sequence protocol at the C level but
            // not as `__getitem__` / `__len__` type slots, so they would miss
            // the `PySequence_Check` path below. `getitem` indexes them
            // (returning the int byte) for `W_ReversedIterator` to walk.
            if pyre_object::bytesobject::is_bytes(obj)
                || pyre_object::bytearrayobject::is_bytearray(obj)
            {
                let n = crate::baseobjspace::len_w(obj)?;
                return Ok(pyre_object::functional::w_reversed_new(obj, n - 1));
            }
        }
        // range: functional.py W_Range.descr_reversed — reflect
        // the span and hand back a fresh reverse-walking iterator. (range is
        // not subclassable, so no override can apply.)
        if pyre_object::is_w_range(obj) {
            return Ok(pyre_object::w_range_reversed(obj));
        }
        // range_iterator: a bare iterator (e.g. from `iter(range(n))`)
        // can also be reversed. Mirror `W_IntRangeIterator`'s live
        // `(current, remaining, step)` cursor by starting at the last
        // remaining item, keeping the same count, and negating the step.
        if pyre_object::is_range_iter(obj) {
            let (current, remaining, step) = pyre_object::w_range_iter_fields(obj);
            if remaining <= 0 {
                return Ok(pyre_object::w_range_iter_new(0, 0, 1));
            }
            let last = current + (remaining - 1) * step;
            return Ok(pyre_object::w_range_iter_new(last, remaining, -step));
        }
        // bytes / bytearray: yield the byte values in reverse.
        if pyre_object::bytesobject::is_bytes_like(obj) {
            let n = pyre_object::bytesobject::bytes_like_len(obj);
            let mut items = Vec::with_capacity(n);
            for i in (0..n).rev() {
                items.push(w_int_new(
                    pyre_object::bytesobject::bytes_like_getitem(obj, i) as i64,
                ));
            }
            return Ok(pyre_object::w_seq_iter_new(
                pyre_object::w_list_new(items),
                n,
            ));
        }
    }
    // `__reversed__` resolved through the type MRO (`functional.py:362-366`) —
    // honors a subclass override and the inherited builtin `list.__reversed__`,
    // and any user object defining `__reversed__`.
    if let Some(tp) = crate::typedef::r#type(obj) {
        if let Some(method) = unsafe { crate::baseobjspace::lookup_in_type(tp, "__reversed__") } {
            return Ok(crate::call_function(method, &[obj]));
        }
    }
    // functional.py:351 — without `__reversed__`, require the sequence
    // protocol. `PySequence_Check` first (an object with `__getitem__`): a
    // non-sequence is "not reversible", while a sequence missing `__len__`
    // raises the regular "has no len()" from `len`.
    if let Some(tp) = crate::typedef::r#type(obj) {
        let has_getitem =
            unsafe { crate::baseobjspace::lookup_in_type(tp, "__getitem__") }.is_some();
        if has_getitem {
            // `functional.py:354-359` — reverse lazily through `W_ReversedIterator`.
            let n = crate::baseobjspace::len_w(obj)?;
            return Ok(pyre_object::functional::w_reversed_new(obj, n - 1));
        }
    }
    Err(crate::PyError::type_error(format!(
        "'{}' object is not reversible",
        crate::baseobjspace::object_functionstr_type_name(obj)
    )))
}

/// `pypy/module/__builtin__/functional.py:328-340 builtin_sorted`
/// parity:
///
/// ```python
/// @unwrap_spec(reverse=bool)
/// def sorted(space, w_iterable, w_key=None, reverse=False):
///     w_lst = space.call_function(space.w_list, w_iterable)
///     space.call_method(w_lst, "sort", w_key, space.newbool(reverse))
///     return w_lst
/// ```
///
/// PyPy's `sort` then calls into `listobject.py W_ListObject.descr_sort`
/// which dispatches keys through `space.lt`.  Pyre mirrors:
///   - exactly one positional iterable (extras → TypeError),
///   - kwargs limited to `{key, reverse}` (others → TypeError),
///   - per-comparison errors (e.g. user `__lt__` raises) propagate
///     instead of silently falling back to "treat as not less".
pub(crate) fn builtin_sorted(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    if positional.is_empty() {
        return Err(crate::PyError::type_error(
            "sorted() requires at least one argument",
        ));
    }
    if positional.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "sorted() takes at most 1 positional argument ({} given)",
            positional.len()
        )));
    }
    kwarg_reject_unknown(kwargs, &["key", "reverse"], "sorted")?;
    let iterable = positional[0];
    let key_fn = kwarg_get(kwargs, "key").filter(|k| unsafe { !pyre_object::is_none(*k) });
    let reverse = kwarg_get(kwargs, "reverse")
        .map(|v| crate::baseobjspace::is_true(v))
        .transpose()?
        .unwrap_or(false);
    let mut items = collect_iterable(iterable)?;
    // `pypy/objspace/std/listobject.py W_ListObject.descr_sort` →
    // build (key, item) pairs, sort by key, optionally reverse.
    let keyed: Vec<(PyObjectRef, PyObjectRef)> = if let Some(kf) = key_fn {
        items
            .iter()
            .map(|&item| {
                let k = crate::call_function(kf, &[item]);
                (k, item)
            })
            .collect()
    } else {
        items.iter().map(|&item| (item, item)).collect()
    };
    let mut keyed = keyed;
    // `rpython/rlib/listsort.py listsort.lt` defers to
    // `space.lt(a, b)` and propagates exceptions; if the user's
    // `__lt__` raises, sort halts with that error.  Rust's
    // `sort_by` closure cannot return Result, so capture the first
    // error via a Cell and surface it after the sort completes.
    // `pypy/objspace/std/listobject.py descr_sort` reverses before and after a stable
    // ascending sort for `reverse=True`, so equal elements keep their
    // original relative order (a stable descending sort). A single
    // post-sort reverse would instead flip ties.
    if reverse {
        keyed.reverse();
    }
    let sort_error: std::cell::Cell<Option<crate::PyError>> = std::cell::Cell::new(None);
    let sort_lt = |ka: PyObjectRef, kb: PyObjectRef| -> bool {
        if sort_error
            .take()
            .map(|e| {
                sort_error.set(Some(e));
                true
            })
            .unwrap_or(false)
        {
            return false;
        }
        match crate::baseobjspace::compare(ka, kb, crate::baseobjspace::CompareOp::Lt) {
            Ok(r) => crate::baseobjspace::is_true(r).unwrap_or_else(|e| {
                sort_error.set(Some(e));
                false
            }),
            Err(e) => {
                sort_error.set(Some(e));
                false
            }
        }
    };
    keyed.sort_by(|(ka, _), (kb, _)| {
        let ab = sort_lt(*ka, *kb);
        if ab {
            return std::cmp::Ordering::Less;
        }
        let ba = sort_lt(*kb, *ka);
        if ba {
            return std::cmp::Ordering::Greater;
        }
        // Fast-path tail kept for the cases where `compare` returns
        // `False` for both directions (legacy unhashable / unorderable
        // pairs that pyre still has) — preserves prior behaviour.
        unsafe {
            if is_int(*ka) && is_int(*kb) {
                return w_int_get_value(*ka).cmp(&w_int_get_value(*kb));
            }
            if is_str(*ka) && is_str(*kb) {
                return w_str_get_value(*ka).cmp(w_str_get_value(*kb));
            }
            if is_float(*ka) && is_float(*kb) {
                return pyre_object::w_float_get_value(*ka)
                    .partial_cmp(&pyre_object::w_float_get_value(*kb))
                    .unwrap_or(std::cmp::Ordering::Equal);
            }
            std::cmp::Ordering::Equal
        }
    });
    if let Some(err) = sort_error.take() {
        return Err(err);
    }
    // Second half of the `reverse=True` double-reverse (see above).
    if reverse {
        keyed.reverse();
    }
    items = keyed.into_iter().map(|(_, v)| v).collect();
    Ok(w_list_new(items))
}

/// `any(iterable)` — PyPy: operation.py any
/// `any(iterable)` — PyPy: baseobjspace.py any_w
pub fn builtin_any_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_any(args)
}
fn builtin_any(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "any() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let items = collect_iterable(args[0])?;
    for item in items {
        if crate::baseobjspace::is_true(item)? {
            return Ok(w_bool_from(true));
        }
    }
    Ok(w_bool_from(false))
}

/// `all(iterable)` — PyPy: operation.py all
/// `all(iterable)` — PyPy: baseobjspace.py all_w
/// Shared file wrapper type — plain instance with hasdict so that
/// open() can attach `__file_data__` / `__file_pos__` / `__file_mode__`
/// as instance attributes, matching the PyPy FileIO/TextIOWrapper
/// duck-typing surface without a dedicated W_FileObject.
pub fn file_wrapper_type() -> PyObjectRef {
    thread_local! {
        static FILE_WRAPPER_TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    FILE_WRAPPER_TYPE.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("_io.TextIOWrapper", init_file_wrapper_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

/// PyPy: pypy/module/_io/interp_iobase.py W_IOBase.
fn init_file_wrapper_type(ns: &mut DictStorage) {
    crate::dict_storage_store(ns, "read", make_builtin_function("read", file_method_read));
    crate::dict_storage_store(
        ns,
        "readline",
        make_builtin_function_with_arity("readline", file_method_readline, 1),
    );
    crate::dict_storage_store(
        ns,
        "readlines",
        make_builtin_function_with_arity("readlines", file_method_readlines, 1),
    );
    crate::dict_storage_store(
        ns,
        "write",
        make_builtin_function_with_arity("write", file_method_write, 2),
    );
    crate::dict_storage_store(
        ns,
        "close",
        make_builtin_function_with_arity("close", file_method_close, 1),
    );
    crate::dict_storage_store(
        ns,
        "flush",
        make_builtin_function_with_arity("flush", file_method_flush, 1),
    );
    crate::dict_storage_store(
        ns,
        "__enter__",
        make_builtin_function_with_arity("__enter__", |args| Ok(args[0]), 1),
    );
    crate::dict_storage_store(
        ns,
        "__exit__",
        make_builtin_function("__exit__", |args| {
            // Call close on exit.
            file_method_close(&args[..1])?;
            Ok(w_none())
        }),
    );
    crate::dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function_with_arity("__iter__", |args| Ok(args[0]), 1),
    );
    crate::dict_storage_store(
        ns,
        "__next__",
        make_builtin_function_with_arity(
            "__next__",
            |args| {
                let line = file_method_readline(args)?;
                unsafe {
                    let s = pyre_object::w_str_get_value(line);
                    if s.is_empty() {
                        return Err(crate::PyError::stop_iteration());
                    }
                }
                Ok(line)
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "fileno",
        make_builtin_function_with_arity(
            "fileno",
            |args| match args.first().copied().and_then(file_get_fd) {
                Some(fd) => Ok(w_int_new(fd as i64)),
                None => Err(crate::PyError::os_error(
                    "fileno() on a file without a descriptor",
                )),
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "readable",
        make_builtin_function_with_arity(
            "readable",
            |args| {
                let mode = crate::baseobjspace::getattr_str(args[0], "__file_mode__")
                    .ok()
                    .map(|m| unsafe { pyre_object::w_str_get_value(m).to_string() })
                    .unwrap_or_default();
                Ok(w_bool_from(mode.contains('r') || mode.contains('+')))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "writable",
        make_builtin_function_with_arity(
            "writable",
            |args| {
                let mode = crate::baseobjspace::getattr_str(args[0], "__file_mode__")
                    .ok()
                    .map(|m| unsafe { pyre_object::w_str_get_value(m).to_string() })
                    .unwrap_or_default();
                Ok(w_bool_from(
                    mode.contains('w')
                        || mode.contains('a')
                        || mode.contains('x')
                        || mode.contains('+'),
                ))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "seekable",
        make_builtin_function_with_arity(
            "seekable",
            // An fd-backed object is seekable iff `lseek` succeeds: a real
            // file does, a pipe/socket fails with ESPIPE.  The in-memory
            // path wrapper is always seekable.
            |args| {
                if let Some(fd) = file_get_fd(args[0]) {
                    #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
                    {
                        #[cfg(not(feature = "sandbox"))]
                        return Ok(w_bool_from(
                            unsafe { libc::lseek(fd, 0, libc::SEEK_CUR) } >= 0,
                        ));
                        #[cfg(feature = "sandbox")]
                        return Ok(w_bool_from(
                            crate::host_seam::ops::lseek(fd, 0, libc::SEEK_CUR).is_ok(),
                        ));
                    }
                    #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
                    {
                        let _ = fd;
                        return Ok(w_bool_from(false));
                    }
                }
                Ok(w_bool_from(true))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "seek",
        make_builtin_function("seek", |args| {
            if let Some(fd) = file_get_fd(args[0]) {
                let offset = args
                    .get(1)
                    .map(|&o| unsafe { pyre_object::w_int_get_value(o) })
                    .unwrap_or(0);
                let whence = args
                    .get(2)
                    .map(|&o| unsafe { pyre_object::w_int_get_value(o) })
                    .unwrap_or(0) as i32;
                #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
                {
                    #[cfg(not(feature = "sandbox"))]
                    let pos = {
                        let pos = unsafe { libc::lseek(fd, offset as libc::off_t, whence) };
                        if pos < 0 {
                            return Err(fd_io_err(std::io::Error::last_os_error()));
                        }
                        pos
                    };
                    #[cfg(feature = "sandbox")]
                    let pos = crate::host_seam::ops::lseek(fd, offset, whence)
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(w_int_new(pos as i64));
                }
                #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
                {
                    let _ = (fd, offset, whence);
                    return Err(crate::PyError::not_implemented(
                        "fd seek requires host_env feature",
                    ));
                }
            }
            if args.len() >= 2 {
                let _ = crate::baseobjspace::setattr_str(args[0], "__file_pos__", args[1]);
            }
            Ok(w_none())
        }),
    );
    crate::dict_storage_store(
        ns,
        "tell",
        make_builtin_function_with_arity(
            "tell",
            |args| {
                if let Some(fd) = file_get_fd(args[0]) {
                    #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
                    {
                        #[cfg(not(feature = "sandbox"))]
                        let pos = {
                            let pos = unsafe { libc::lseek(fd, 0, libc::SEEK_CUR) };
                            if pos < 0 {
                                return Err(fd_io_err(std::io::Error::last_os_error()));
                            }
                            pos
                        };
                        #[cfg(feature = "sandbox")]
                        let pos = crate::host_seam::ops::lseek(fd, 0, libc::SEEK_CUR)
                            .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                        return Ok(w_int_new(pos as i64));
                    }
                    #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
                    {
                        let _ = fd;
                    }
                }
                if let Ok(pos) = crate::baseobjspace::getattr_str(args[0], "__file_pos__") {
                    Ok(pos)
                } else {
                    Ok(w_int_new(0))
                }
            },
            1,
        ),
    );
}

/// The path-backed file object's buffered contents as raw bytes.  Binary and
/// text streams both hold the exact file bytes here; text reads decode on the
/// way out (`fd_bytes_to_obj`), so non-UTF-8 content survives a round trip.
fn file_get_data(self_obj: PyObjectRef) -> Vec<u8> {
    crate::baseobjspace::getattr_str(self_obj, "__file_data__")
        .ok()
        .and_then(|d| unsafe {
            if pyre_object::bytesobject::is_bytes_like(d) {
                Some(pyre_object::bytesobject::bytes_like_data(d).to_vec())
            } else if pyre_object::is_str(d) {
                Some(pyre_object::w_str_get_wtf8(d).as_bytes().to_vec())
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn file_get_pos(self_obj: PyObjectRef) -> usize {
    crate::baseobjspace::getattr_str(self_obj, "__file_pos__")
        .ok()
        .and_then(|p| unsafe {
            if pyre_object::is_int(p) {
                Some(pyre_object::w_int_get_value(p) as usize)
            } else {
                None
            }
        })
        .unwrap_or(0)
}

fn file_set_pos(self_obj: PyObjectRef, pos: usize) {
    // Private storage slot on a fresh hasdict file wrapper (no custom
    // `__setattr__`, `__file_pos__` is not a descriptor), so the write is
    // the infallible instance-dict store `W_Root.setdictvalue`
    // (baseobjspace.py:51) that `setattr_str` would itself reach.
    crate::baseobjspace::setdictvalue(self_obj, "__file_pos__", w_int_new(pos as i64));
}

/// The raw file descriptor for an fd-backed file object (`open(fd, ...)`),
/// or `None` for an in-memory (path-backed) wrapper.
fn file_get_fd(self_obj: PyObjectRef) -> Option<i32> {
    crate::baseobjspace::getattr_str(self_obj, "__file_fd__")
        .ok()
        .and_then(|v| unsafe {
            if pyre_object::is_int(v) {
                Some(pyre_object::w_int_get_value(v) as i32)
            } else {
                None
            }
        })
}

fn file_is_binary(self_obj: PyObjectRef) -> bool {
    crate::baseobjspace::getattr_str(self_obj, "__file_binary__")
        .ok()
        .map(|v| unsafe { pyre_object::is_bool(v) && pyre_object::w_bool_get_value(v) })
        .unwrap_or(false)
}

/// Reduce a [`SeamError`] to a `std::io::Error` so the fd helpers keep their
/// `io::Result` signature (the caller's `fd_io_err` then maps it to `OSError`).
#[cfg(all(feature = "host_env", not(target_arch = "wasm32"), feature = "sandbox"))]
fn seam_to_io(e: crate::host_seam::SeamError) -> std::io::Error {
    match e {
        crate::host_seam::SeamError::Os(errno) => std::io::Error::from_raw_os_error(errno),
        _ => std::io::Error::other("sandbox error"),
    }
}

#[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
fn fd_read_into(fd: i32, buf: &mut [u8]) -> std::io::Result<usize> {
    #[cfg(feature = "sandbox")]
    {
        // The controller services one read per request; copy the reply (at most
        // `buf.len()` bytes) into the caller's buffer.
        let data = crate::host_seam::ops::read(fd, buf.len() as i64).map_err(seam_to_io)?;
        let n = data.len().min(buf.len());
        buf[..n].copy_from_slice(&data[..n]);
        return Ok(n);
    }
    #[cfg(not(feature = "sandbox"))]
    loop {
        // `count` is `size_t` on Unix but `c_uint` on Windows; `as _` casts
        // to whichever the platform's `libc::read` expects.
        let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len() as _) };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err(e);
        }
        return Ok(n as usize);
    }
}

/// Read up to `n` bytes (or until EOF when `n` is `None`) from `fd`.
#[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
fn fd_read(fd: i32, n: Option<usize>) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut buf = [0u8; 65536];
    loop {
        let want = match n {
            Some(limit) => {
                if out.len() >= limit {
                    break;
                }
                (limit - out.len()).min(buf.len())
            }
            None => buf.len(),
        };
        let got = fd_read_into(fd, &mut buf[..want])?;
        if got == 0 {
            break;
        }
        out.extend_from_slice(&buf[..got]);
    }
    Ok(out)
}

/// Wrap raw bytes from a file read into `bytes` (binary mode) or decode them
/// through the text stream's codec/error handler.
fn fd_bytes_to_obj(self_obj: PyObjectRef, data: Vec<u8>) -> Result<PyObjectRef, crate::PyError> {
    if file_is_binary(self_obj) {
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
    } else {
        let (encoding, errors) = unsafe { stream_encoding_errors(self_obj) };
        let w_bytes = pyre_object::bytesobject::w_bytes_from_bytes(&data);
        crate::typedef::bytes_method_decode(&[w_bytes, w_str_new(&encoding), w_str_new(&errors)])
    }
}

fn fd_io_err(e: std::io::Error) -> crate::PyError {
    crate::PyError::os_error_with_errno(e.raw_os_error().unwrap_or(5), e.to_string())
}

fn file_method_read(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("read() requires self"));
    }
    if let Some(fd) = file_get_fd(args[0]) {
        let n = args.get(1).and_then(|&o| unsafe {
            if pyre_object::is_int(o) {
                let v = pyre_object::w_int_get_value(o);
                if v < 0 { None } else { Some(v as usize) }
            } else {
                None
            }
        });
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        {
            let data = fd_read(fd, n).map_err(fd_io_err)?;
            return fd_bytes_to_obj(args[0], data);
        }
        #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
        {
            let _ = (fd, n);
            return Err(crate::PyError::not_implemented(
                "fd read requires host_env feature",
            ));
        }
    }
    let data = file_get_data(args[0]);
    let pos = file_get_pos(args[0]).min(data.len());
    let remaining = &data[pos..];
    let n = if args.len() >= 2 {
        let n_val = unsafe { pyre_object::w_int_get_value(args[1]) };
        if n_val < 0 {
            remaining.len()
        } else {
            (n_val as usize).min(remaining.len())
        }
    } else {
        remaining.len()
    };
    // Count by bytes; binary mode hands back `bytes`, text mode `str`.
    let end = n.min(remaining.len());
    let chunk = remaining[..end].to_vec();
    file_set_pos(args[0], pos + end);
    fd_bytes_to_obj(args[0], chunk)
}

fn file_method_readline(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("readline() requires self"));
    }
    // Optional size cap (`readline(size)`): stop after `size` bytes even
    // before a newline. A missing or negative size means no cap.
    let max = args.get(1).and_then(|&o| unsafe {
        if pyre_object::is_int(o) {
            let v = pyre_object::w_int_get_value(o);
            if v < 0 { None } else { Some(v as usize) }
        } else {
            None
        }
    });
    if let Some(fd) = file_get_fd(args[0]) {
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        {
            // Raw fds cannot un-read, so consume one byte at a time up to
            // the newline (or EOF) to avoid over-reading past the line.
            let mut out = Vec::new();
            let mut byte = [0u8; 1];
            loop {
                if max == Some(out.len()) {
                    break;
                }
                let got = fd_read_into(fd, &mut byte).map_err(fd_io_err)?;
                if got == 0 {
                    break;
                }
                out.push(byte[0]);
                if byte[0] == b'\n' {
                    break;
                }
            }
            return fd_bytes_to_obj(args[0], out);
        }
        #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
        {
            let _ = fd;
            return Err(crate::PyError::not_implemented(
                "fd readline requires host_env feature",
            ));
        }
    }
    let data = file_get_data(args[0]);
    let pos = file_get_pos(args[0]);
    if pos >= data.len() {
        return fd_bytes_to_obj(args[0], Vec::new());
    }
    let rest = &data[pos..];
    let mut end = rest
        .iter()
        .position(|&b| b == b'\n')
        .map(|i| i + 1)
        .unwrap_or(rest.len());
    if let Some(m) = max {
        end = end.min(m);
    }
    let line = rest[..end].to_vec();
    file_set_pos(args[0], pos + end);
    fd_bytes_to_obj(args[0], line)
}

fn file_method_readlines(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("readlines() requires self"));
    }
    let mut lines = Vec::new();
    loop {
        let line = file_method_readline(args)?;
        // readline returns `bytes` in binary mode and `str` otherwise; an
        // empty result of either kind marks EOF.
        let empty = unsafe {
            if pyre_object::bytesobject::is_bytes_like(line) {
                pyre_object::bytesobject::bytes_like_data(line).is_empty()
            } else {
                pyre_object::w_str_get_value(line).is_empty()
            }
        };
        if empty {
            break;
        }
        lines.push(line);
    }
    Ok(w_list_new(lines))
}

fn file_method_write(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("write() requires (self, data)"));
    }
    if let Some(fd) = file_get_fd(args[0]) {
        let bytes: Vec<u8> = unsafe {
            if pyre_object::is_str(args[1]) {
                // Encode through the stream's codec + error handler; a lone
                // surrogate is routed to the handler (`strict` →
                // UnicodeEncodeError) rather than panicking in
                // `w_str_get_value`.
                let (encoding, errors) = stream_encoding_errors(args[0]);
                crate::type_methods::encode_object(args[1], &encoding, &errors)?
            } else if pyre_object::bytesobject::is_bytes_like(args[1]) {
                pyre_object::bytesobject::bytes_like_data(args[1]).to_vec()
            } else {
                return Err(crate::PyError::type_error("write() expects str or bytes"));
            }
        };
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        {
            #[cfg(not(feature = "sandbox"))]
            let n = {
                // `count` is `size_t` on Unix but `c_uint` on Windows.
                let n = unsafe {
                    libc::write(fd, bytes.as_ptr() as *const libc::c_void, bytes.len() as _)
                };
                if n < 0 {
                    return Err(fd_io_err(std::io::Error::last_os_error()));
                }
                n as i64
            };
            #[cfg(feature = "sandbox")]
            let n = crate::host_seam::ops::write(fd, &bytes)
                .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
            return Ok(w_int_new(n));
        }
        #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
        {
            let _ = (fd, bytes);
            return Err(crate::PyError::not_implemented(
                "fd write requires host_env feature",
            ));
        }
    }
    // Append to __file_data__ and update on close.
    unsafe {
        let mut prev = file_get_data(args[0]);
        let (bytes, len) = if pyre_object::is_str(args[1]) {
            // Encode through the stream's codec + error handler so a lone
            // surrogate raises (`strict`) instead of panicking in
            // `w_str_get_value`. The reported count is characters written.
            let (encoding, errors) = stream_encoding_errors(args[0]);
            let bytes = crate::type_methods::encode_object(args[1], &encoding, &errors)?;
            (bytes, pyre_object::w_str_len(args[1]))
        } else if pyre_object::bytesobject::is_bytes_like(args[1]) {
            let data = pyre_object::bytesobject::bytes_like_data(args[1]).to_vec();
            let len = data.len();
            (data, len)
        } else {
            return Err(crate::PyError::type_error("write() expects str or bytes"));
        };
        prev.extend_from_slice(&bytes);
        let _ = crate::baseobjspace::setattr_str(
            args[0],
            "__file_data__",
            pyre_object::bytesobject::w_bytes_from_bytes(&prev),
        );
        let _ = crate::baseobjspace::setattr_str(args[0], "__file_dirty__", w_bool_from(true));
        let append = crate::baseobjspace::getattr_str(args[0], "__file_mode__")
            .ok()
            .map(|mode| pyre_object::w_str_get_value(mode).contains('a'))
            .unwrap_or(false);
        if !append {
            file_flush_dirty(args[0])?;
        }
        Ok(w_int_new(len as i64))
    }
}

fn file_method_close(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_none());
    }
    if let Some(fd) = file_get_fd(args[0]) {
        let already = crate::baseobjspace::getattr_str(args[0], "closed")
            .ok()
            .map(|v| unsafe { pyre_object::is_bool(v) && pyre_object::w_bool_get_value(v) })
            .unwrap_or(false);
        if !already {
            // Mark closed first so the fd is not reusable even if the underlying
            // close reports an error, then surface that error (matching
            // _io.FileIO.close).
            let _ = crate::baseobjspace::setattr_str(args[0], "closed", w_bool_from(true));
            #[cfg(all(
                feature = "host_env",
                not(target_arch = "wasm32"),
                not(feature = "sandbox")
            ))]
            // SAFETY: close(2) on the file object's own fd.
            if unsafe { libc::close(fd) } < 0 {
                let e = std::io::Error::last_os_error();
                return Err(crate::PyError::os_error_with_errno(
                    e.raw_os_error().unwrap_or(0),
                    format!("close: {e}"),
                ));
            }
            #[cfg(all(feature = "host_env", not(target_arch = "wasm32"), feature = "sandbox"))]
            crate::host_seam::ops::close(fd).map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
            #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
            let _ = fd;
        }
        return Ok(w_none());
    }
    // If the file was opened in a writable mode, flush the in-memory
    // buffer to disk.
    file_flush_dirty(args[0])?;
    Ok(w_none())
}

/// Write a writable file's dirty in-memory buffer out to disk, leaving the
/// object open. Shared by `close` and `flush`.
fn file_flush_dirty(obj: PyObjectRef) -> Result<(), crate::PyError> {
    // Under sandbox every file object is fd-backed (opens go through the seam)
    // and the controller enforces read-only, so a dirty writable buffer never
    // reaches here; keep the raw std::fs write out of the sandbox build.
    #[cfg(feature = "sandbox")]
    {
        let _ = obj;
        Ok(())
    }
    #[cfg(not(feature = "sandbox"))]
    {
        let dirty = crate::baseobjspace::getattr_str(obj, "__file_dirty__")
            .ok()
            .map(|v| unsafe { pyre_object::is_bool(v) && pyre_object::w_bool_get_value(v) })
            .unwrap_or(false);
        if !dirty {
            return Ok(());
        }
        if let (Ok(name), Ok(mode)) = (
            crate::baseobjspace::getattr_str(obj, "__file_name__"),
            crate::baseobjspace::getattr_str(obj, "__file_mode__"),
        ) {
            let name_s = unsafe { pyre_object::w_str_get_value(name).to_string() };
            let mode_s = unsafe { pyre_object::w_str_get_value(mode).to_string() };
            let data = file_get_data(obj);
            let append = mode_s.contains('a');
            let write_res = if append {
                std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(&name_s)
                    .and_then(|mut f| std::io::Write::write_all(&mut f, &data))
            } else {
                std::fs::write(&name_s, &data)
            };
            if let Err(e) = write_res {
                return Err(crate::PyError::os_error_with_errno(
                    e.raw_os_error().unwrap_or(5),
                    format!("{e}: '{name_s}'"),
                ));
            }
            crate::baseobjspace::setattr_str(obj, "__file_dirty__", w_bool_from(false))?;
        }
        Ok(())
    }
}

/// `flush()` — push any buffered writes to disk without closing. For
/// fd-backed objects writes go straight through, so this is a no-op.
fn file_method_flush(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_none());
    }
    if file_get_fd(args[0]).is_some() {
        return Ok(w_none());
    }
    file_flush_dirty(args[0])?;
    Ok(w_none())
}

/// builtins.open(file, mode='r', ...) — PyPy: io.open → FileIO + TextIOWrapper.
/// Minimal implementation that loads the entire file into memory and
/// returns a file wrapper instance.
/// POSIX `open(2)` flags for a text/binary mode string, used when an
/// `opener` is supplied (the opener receives `(file, flags)`).
#[cfg(unix)]
fn open_flags_for_mode(mode: &str) -> i32 {
    let write = mode.contains('w');
    let append = mode.contains('a');
    let exclusive = mode.contains('x');
    let updating = mode.contains('+');
    let mut flags = if updating {
        libc::O_RDWR
    } else if write || append || exclusive {
        libc::O_WRONLY
    } else {
        libc::O_RDONLY
    };
    if write {
        flags |= libc::O_CREAT | libc::O_TRUNC;
    }
    if append {
        flags |= libc::O_CREAT | libc::O_APPEND;
    }
    if exclusive {
        flags |= libc::O_CREAT | libc::O_EXCL;
    }
    flags
}
#[cfg(not(unix))]
fn open_flags_for_mode(_mode: &str) -> i32 {
    0
}

pub fn builtin_open(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("open() missing 'file' argument"));
    }
    let (open_pos, open_kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(
        open_kwargs,
        &[
            "file",
            "mode",
            "buffering",
            "encoding",
            "errors",
            "newline",
            "closefd",
            "opener",
        ],
        "open",
    )?;
    let path_obj = resolve_pos_or_kw(open_pos.first().copied(), open_kwargs, "file", "open", 1)?
        .ok_or_else(|| crate::PyError::type_error("open() missing 'file' argument"))?;
    let mode_obj = resolve_pos_or_kw(open_pos.get(1).copied(), open_kwargs, "mode", "open", 2)?;
    let encoding_obj =
        resolve_pos_or_kw(open_pos.get(3).copied(), open_kwargs, "encoding", "open", 4)?;
    let errors_obj = resolve_pos_or_kw(open_pos.get(4).copied(), open_kwargs, "errors", "open", 5)?;
    let str_or_none =
        |obj: Option<PyObjectRef>, name: &str| -> Result<Option<String>, crate::PyError> {
            match obj {
                Some(o) if unsafe { pyre_object::is_str(o) } => Ok(Some(unsafe {
                    pyre_object::w_str_get_value(o)
                        .to_ascii_lowercase()
                        .replace('_', "-")
                })),
                Some(o) if unsafe { pyre_object::is_none(o) } => Ok(None),
                Some(o) => Err(crate::PyError::type_error(format!(
                    "open() argument '{name}' must be str or None, not {}",
                    crate::type_methods::arg_type_name(o)
                ))),
                None => Ok(None),
            }
        };
    let encoding = str_or_none(encoding_obj, "encoding")?.unwrap_or_else(|| "utf-8".to_string());
    let errors = str_or_none(errors_obj, "errors")?.unwrap_or_else(|| "strict".to_string());
    let mode: String = match mode_obj {
        Some(m) if unsafe { pyre_object::is_str(m) } => unsafe {
            pyre_object::w_str_get_value(m).to_string()
        },
        Some(m) if unsafe { pyre_object::is_none(m) } => "r".to_string(),
        Some(m) => {
            return Err(crate::PyError::type_error(format!(
                "open() argument 'mode' must be str, not {}",
                crate::type_methods::arg_type_name(m)
            )));
        }
        None => "r".to_string(),
    };

    // Integer file descriptor → fd-backed file object, reading/writing
    // through the descriptor directly (io.open(fd, ...) — used by
    // subprocess pipe handling).
    if unsafe { pyre_object::is_int(path_obj) } {
        let fd = unsafe { pyre_object::w_int_get_value(path_obj) } as i32;
        let binary = mode.contains('b');
        let wrapper = pyre_object::w_instance_new(file_wrapper_type());
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_fd__", w_int_new(fd as i64));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_binary__", w_bool_from(binary));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_mode__", w_str_new(&mode));
        let _ = crate::baseobjspace::setattr_str(wrapper, "encoding", w_str_new(&encoding));
        let _ = crate::baseobjspace::setattr_str(wrapper, "errors", w_str_new(&errors));
        let _ = crate::baseobjspace::setattr_str(wrapper, "name", w_int_new(fd as i64));
        let _ = crate::baseobjspace::setattr_str(wrapper, "mode", w_str_new(&mode));
        let _ = crate::baseobjspace::setattr_str(wrapper, "closed", w_bool_from(false));
        return Ok(wrapper);
    }

    let path = unsafe {
        if pyre_object::is_str(path_obj) {
            pyre_object::w_str_get_value(path_obj).to_string()
        } else if pyre_object::bytesobject::is_bytes_like(path_obj) {
            let data = pyre_object::bytesobject::bytes_like_data(path_obj);
            String::from_utf8_lossy(data).into_owned()
        } else if let Some(fspath_fn) = crate::typedef::r#type(path_obj)
            .and_then(|pt| crate::baseobjspace::lookup_in_type(pt, "__fspath__"))
        {
            // `type(path).__fspath__(path)` — unbound descriptor + single arg.
            let result = crate::call::call_function_impl_result(fspath_fn, &[path_obj])?;
            if pyre_object::is_str(result) {
                pyre_object::w_str_get_value(result).to_string()
            } else if pyre_object::bytesobject::is_bytes_like(result) {
                let data = pyre_object::bytesobject::bytes_like_data(result);
                String::from_utf8_lossy(data).into_owned()
            } else {
                return Err(crate::PyError::type_error(
                    "open(): path should be str, bytes, os.PathLike",
                ));
            }
        } else {
            return Err(crate::PyError::type_error(
                "open(): path should be str, bytes, os.PathLike",
            ));
        }
    };
    let binary = mode.contains('b');
    let writing = mode.contains('w') || mode.contains('a') || mode.contains('x');
    let reading = mode.contains('r') || !writing;

    // `open(..., opener=callable)`: the opener supplies the file descriptor
    // (e.g. `tempfile.NamedTemporaryFile` creates the temp file and records
    // its name in the opener). Call it with `(file, flags)` and wrap the
    // returned fd directly.
    if let Some(opener) = kwarg_get(open_kwargs, "opener") {
        if !unsafe { pyre_object::is_none(opener) } {
            let flags = open_flags_for_mode(&mode);
            let fd_obj = crate::call::call_function_impl_result(
                opener,
                &[path_obj, w_int_new(flags as i64)],
            )?;
            let fd = unsafe { pyre_object::w_int_get_value(fd_obj) } as i32;
            let wrapper = pyre_object::w_instance_new(file_wrapper_type());
            let _ = crate::baseobjspace::setattr_str(wrapper, "__file_fd__", w_int_new(fd as i64));
            let _ =
                crate::baseobjspace::setattr_str(wrapper, "__file_binary__", w_bool_from(binary));
            let _ = crate::baseobjspace::setattr_str(wrapper, "__file_mode__", w_str_new(&mode));
            let _ = crate::baseobjspace::setattr_str(wrapper, "encoding", w_str_new(&encoding));
            let _ = crate::baseobjspace::setattr_str(wrapper, "errors", w_str_new(&errors));
            let _ = crate::baseobjspace::setattr_str(wrapper, "name", path_obj);
            let _ = crate::baseobjspace::setattr_str(wrapper, "mode", w_str_new(&mode));
            let _ = crate::baseobjspace::setattr_str(wrapper, "closed", w_bool_from(false));
            return Ok(wrapper);
        }
    }

    // The sandbox routes the whole open→read/write→close chain through the
    // controller: acquire a real fd via the trampoline and hand back an
    // fd-backed wrapper. The in-memory `host_env::fs::read` path below would
    // otherwise read the real filesystem, escaping the sandbox.
    #[cfg(feature = "sandbox")]
    {
        let _ = (reading, writing);
        let flags = open_flags_for_mode(&mode);
        let fd = crate::host_seam::ops::open(path.as_bytes(), flags, 0o666)
            .map_err(|e| crate::host_seam::seam_os_err(e, &path))?;
        let wrapper = pyre_object::w_instance_new(file_wrapper_type());
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_fd__", w_int_new(fd as i64));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_binary__", w_bool_from(binary));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_mode__", w_str_new(&mode));
        let _ = crate::baseobjspace::setattr_str(wrapper, "encoding", w_str_new(&encoding));
        let _ = crate::baseobjspace::setattr_str(wrapper, "errors", w_str_new(&errors));
        let _ = crate::baseobjspace::setattr_str(wrapper, "name", path_obj);
        let _ = crate::baseobjspace::setattr_str(wrapper, "mode", w_str_new(&mode));
        let _ = crate::baseobjspace::setattr_str(wrapper, "closed", w_bool_from(false));
        Ok(wrapper)
    }
    #[cfg(not(feature = "sandbox"))]
    {
        let data: Vec<u8> = if reading && !mode.contains('w') && !mode.contains('x') {
            #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
            {
                // Sandbox-intentional: with the host_env feature off the
                // interpreter must not reach `std::fs` directly.  Callers in
                // sandbox builds route file I/O through the VFS shim instead;
                // returning NotImplementedError keeps the open() builtin from
                // silently leaking real-FS reads here.
                let _ = (binary, &path);
                return Err(crate::PyError::not_implemented(
                    "open() for reading requires host_env feature",
                ));
            }
            #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
            let read_result = rustpython_host_env::fs::read(&path);
            #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
            match read_result {
                // Hold the exact file bytes; text-mode reads decode on the way
                // out (`fd_bytes_to_obj`), so non-UTF-8 content is preserved.
                Ok(bytes) => bytes,
                Err(_e) if writing => Vec::new(),
                Err(e) => {
                    return Err(crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(2),
                        format!("{e}: '{path}'"),
                    ));
                }
            }
        } else {
            Vec::new()
        };

        let wrapper = pyre_object::w_instance_new(file_wrapper_type());
        let _ = crate::baseobjspace::setattr_str(
            wrapper,
            "__file_data__",
            pyre_object::bytesobject::w_bytes_from_bytes(&data),
        );
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_pos__", w_int_new(0));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_name__", w_str_new(&path));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_mode__", w_str_new(&mode));
        // Carry binary-ness so read/readline wrap their chunks as `bytes` in
        // binary mode (`'rb'`), matching the fd-backed branch above.  Without
        // this a path-backed `open(p, 'rb').readline()` would hand back `str`,
        // breaking `tokenize.detect_encoding` (`first.startswith(BOM_UTF8)`).
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_binary__", w_bool_from(binary));
        let _ = crate::baseobjspace::setattr_str(wrapper, "encoding", w_str_new(&encoding));
        let _ = crate::baseobjspace::setattr_str(wrapper, "errors", w_str_new(&errors));
        let _ = crate::baseobjspace::setattr_str(wrapper, "name", w_str_new(&path));
        let _ = crate::baseobjspace::setattr_str(wrapper, "mode", w_str_new(&mode));
        let _ = crate::baseobjspace::setattr_str(wrapper, "closed", w_bool_from(false));
        Ok(wrapper)
    }
}

// ── _io.TextIOWrapper — thin text layer over a binary buffer ─────────
// PyPy: pypy/module/_io/interp_textio.py W_TextIOWrapper. A minimal
// decode/encode shim over the underlying (fd-backed) binary file object,
// enough for `subprocess` text mode (`text=True`/`encoding=`).

fn textio_buffer(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    crate::baseobjspace::getattr_str(self_obj, "__textio_buffer__")
}

/// Call a method on the wrapped buffer (`self` is bound by getattr).
fn textio_call_buffer(
    self_obj: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let buf = textio_buffer(self_obj)?;
    let r = crate::baseobjspace::call_method(buf, name, args);
    if r.is_null() {
        return Err(crate::call::take_call_error()
            .unwrap_or_else(|| crate::PyError::runtime_error(format!("{name} failed"))));
    }
    Ok(r)
}

/// Read the wrapper's stored `encoding` / `errors` (set at construction),
/// defaulting to UTF-8 / strict.
fn textio_enc_err(self_obj: PyObjectRef) -> (String, String) {
    let read = |name: &str, default: &str| {
        crate::baseobjspace::getattr_str(self_obj, name)
            .ok()
            .and_then(|o| unsafe {
                if pyre_object::is_str(o) {
                    Some(pyre_object::w_str_get_value(o).to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| default.to_string())
    };
    (read("encoding", "utf-8"), read("errors", "strict"))
}

/// Decode raw bytes through the wrapper's codec (honoring `encoding` /
/// `errors`), or pass a str through, then apply universal-newline
/// translation (`\r\n`/`\r` → `\n`).
fn textio_decode(obj: PyObjectRef, encoding: &str, errors: &str) -> Result<String, crate::PyError> {
    let s = unsafe {
        if pyre_object::bytesobject::is_bytes_like(obj) {
            let decoded = crate::typedef::bytes_method_decode(&[
                obj,
                w_str_new(encoding),
                w_str_new(errors),
            ])?;
            pyre_object::w_str_get_value(decoded).to_string()
        } else if pyre_object::is_str(obj) {
            pyre_object::w_str_get_value(obj).to_string()
        } else {
            String::new()
        }
    };
    Ok(s.replace("\r\n", "\n").replace('\r', "\n"))
}

/// `io.TextIOWrapper(buffer, encoding=None, errors=None, newline=None, ...)`.
/// Configure a `TextIOWrapper` instance from its constructor arguments.
/// `positional[0]` is the underlying buffer; encoding/errors arrive
/// positionally (1, 2) or as keywords.  The selector-path `_communicate`
/// reads `.encoding`/`.errors` directly to decode raw byte chunks, so they
/// are stamped as concrete strings.
fn textio_configure(
    self_obj: PyObjectRef,
    positional: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
) -> Result<(), crate::PyError> {
    let buffer = positional
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("TextIOWrapper() requires a buffer"))?;
    let str_arg = |obj: Option<PyObjectRef>| -> Option<String> {
        obj.and_then(|o| unsafe {
            if pyre_object::is_str(o) {
                Some(pyre_object::w_str_get_value(o).to_string())
            } else {
                None
            }
        })
    };
    let encoding = str_arg(positional.get(1).copied())
        .or_else(|| str_arg(crate::builtins::kwarg_get(kwargs, "encoding")))
        .unwrap_or_else(|| "utf-8".to_string());
    let errors = str_arg(positional.get(2).copied())
        .or_else(|| str_arg(crate::builtins::kwarg_get(kwargs, "errors")))
        .unwrap_or_else(|| "strict".to_string());
    crate::baseobjspace::setattr_str(self_obj, "__textio_buffer__", buffer)?;
    crate::baseobjspace::setattr_str(self_obj, "closed", w_bool_from(false))?;
    crate::baseobjspace::setattr_str(self_obj, "encoding", w_str_new(&encoding))?;
    crate::baseobjspace::setattr_str(self_obj, "errors", w_str_new(&errors))?;
    if let Ok(name) = crate::baseobjspace::getattr_str(buffer, "name") {
        crate::baseobjspace::setattr_str(self_obj, "name", name)?;
    }
    Ok(())
}

pub fn text_io_wrapper_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let wrapper = pyre_object::w_instance_new(text_io_wrapper_type());
    textio_configure(wrapper, positional, kwargs)?;
    Ok(wrapper)
}

/// `_io.TextIOWrapper.__init__(self, buffer, encoding=None, errors=None,
/// ...)` — configures `self` so the type is subclassable (a subclass's
/// `super().__init__(...)` reaches here).
fn textio_method_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("__init__ requires self"))?;
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    textio_configure(self_obj, positional, kwargs)?;
    Ok(w_none())
}

/// Forward an integer size argument to the underlying buffer; a missing or
/// non-int (e.g. `None`) size means "read everything".
fn textio_size_arg(args: &[PyObjectRef]) -> &[PyObjectRef] {
    match args.get(1) {
        Some(o) if unsafe { pyre_object::is_int(*o) } => std::slice::from_ref(&args[1]),
        _ => &[],
    }
}

fn textio_method_read(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("read() requires self"));
    }
    let raw = textio_call_buffer(args[0], "read", textio_size_arg(args))?;
    let (encoding, errors) = textio_enc_err(args[0]);
    Ok(w_str_new(&textio_decode(raw, &encoding, &errors)?))
}

fn textio_method_readline(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("readline() requires self"));
    }
    let raw = textio_call_buffer(args[0], "readline", textio_size_arg(args))?;
    let (encoding, errors) = textio_enc_err(args[0]);
    Ok(w_str_new(&textio_decode(raw, &encoding, &errors)?))
}

fn textio_method_write(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("write() requires (self, data)"));
    }
    if unsafe { !pyre_object::is_str(args[1]) } {
        return Err(crate::PyError::type_error("write() expects str"));
    }
    // Encode through the stream's codec + error handler so a lone surrogate is
    // routed to the handler (`strict` → UnicodeEncodeError) instead of
    // panicking in `w_str_get_value`.
    let (encoding, errors) = unsafe { stream_encoding_errors(args[0]) };
    let encoded = crate::type_methods::encode_object(args[1], &encoding, &errors)?;
    let nchars = unsafe { pyre_object::w_str_len(args[1]) };
    let bytes = pyre_object::bytesobject::w_bytes_from_bytes(&encoded);
    textio_call_buffer(args[0], "write", &[bytes])?;
    Ok(w_int_new(nchars as i64))
}

fn textio_method_close(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_none());
    }
    let _ = textio_call_buffer(args[0], "close", &[]);
    let _ = crate::baseobjspace::setattr_str(args[0], "closed", w_bool_from(true));
    Ok(w_none())
}

/// Shared `_io.TextIOWrapper` type for text-mode file objects.
pub fn text_io_wrapper_type() -> PyObjectRef {
    thread_local! {
        static TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    TYPE.with(|c| {
        *c.get_or_init(|| {
            let tp =
                crate::typedef::make_builtin_type("_io.TextIOWrapper", init_text_io_wrapper_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

fn init_text_io_wrapper_type(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "__init__",
        make_builtin_function("__init__", textio_method_init),
    );
    crate::dict_storage_store(
        ns,
        "read",
        make_builtin_function("read", textio_method_read),
    );
    crate::dict_storage_store(
        ns,
        "readline",
        make_builtin_function_with_arity("readline", textio_method_readline, 1),
    );
    crate::dict_storage_store(
        ns,
        "readlines",
        make_builtin_function_with_arity(
            "readlines",
            |args| {
                let mut lines = Vec::new();
                loop {
                    let line = textio_method_readline(args)?;
                    if unsafe { pyre_object::w_str_get_value(line).is_empty() } {
                        break;
                    }
                    lines.push(line);
                }
                Ok(w_list_new(lines))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "write",
        make_builtin_function_with_arity("write", textio_method_write, 2),
    );
    crate::dict_storage_store(
        ns,
        "close",
        make_builtin_function_with_arity("close", textio_method_close, 1),
    );
    crate::dict_storage_store(
        ns,
        "flush",
        make_builtin_function_with_arity("flush", |_| Ok(w_none()), 1),
    );
    crate::dict_storage_store(
        ns,
        "fileno",
        make_builtin_function_with_arity(
            "fileno",
            |args| textio_call_buffer(args[0], "fileno", &[]),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "detach",
        make_builtin_function_with_arity("detach", |args| textio_buffer(args[0]), 1),
    );
    crate::dict_storage_store(
        ns,
        "readable",
        make_builtin_function_with_arity(
            "readable",
            |args| textio_call_buffer(args[0], "readable", &[]),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "writable",
        make_builtin_function_with_arity(
            "writable",
            |args| textio_call_buffer(args[0], "writable", &[]),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "seekable",
        make_builtin_function_with_arity(
            "seekable",
            |args| textio_call_buffer(args[0], "seekable", &[]),
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__enter__",
        make_builtin_function_with_arity("__enter__", |args| Ok(args[0]), 1),
    );
    crate::dict_storage_store(
        ns,
        "__exit__",
        make_builtin_function("__exit__", |args| {
            let _ = textio_method_close(&args[..1]);
            Ok(w_none())
        }),
    );
    crate::dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function_with_arity("__iter__", |args| Ok(args[0]), 1),
    );
    crate::dict_storage_store(
        ns,
        "__next__",
        make_builtin_function_with_arity(
            "__next__",
            |args| {
                let line = textio_method_readline(args)?;
                if unsafe { pyre_object::w_str_get_value(line).is_empty() } {
                    return Err(crate::PyError::stop_iteration());
                }
                Ok(line)
            },
            1,
        ),
    );
}

pub fn builtin_all_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_all(args)
}
fn builtin_all(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "all() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let items = collect_iterable(args[0])?;
    for item in items {
        if !crate::baseobjspace::is_true(item)? {
            return Ok(w_bool_from(false));
        }
    }
    Ok(w_bool_from(true))
}

/// `sum(sequence, start=0)` — PyPy `__builtin__/app_functional.py sum`.
///
/// A plain left-fold through `space.add` (`_regular_sum`'s
/// `last = last + x`).  No Kahan/Neumaier compensation: float operands
/// accumulate with ordinary left-to-right IEEE rounding, exactly as PyPy
/// does (`sum([0.1, 0.2, 0.3])` is `0.6000000000000001`, not `0.6`).  A
/// `str`/`bytes`/`bytearray` `start` is rejected up front.
fn builtin_sum(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `sum(iterable, /, start=0)`: iterable is positional-only, start is
    // positional-or-keyword; at most two arguments total.
    let (pos, kwargs) = split_builtin_kwargs(args);
    let total = pos.len() + real_kwarg_count(kwargs);
    if total > 2 {
        return Err(crate::PyError::type_error(format!(
            "sum() takes at most 2 arguments ({total} given)"
        )));
    }
    if pos.is_empty() {
        return Err(crate::PyError::type_error(
            "sum() takes at least 1 positional argument (0 given)",
        ));
    }
    kwarg_reject_unknown(kwargs, &["start"], "sum")?;
    let iterable = pos[0];
    let start = bind_pos_or_kw(pos, kwargs, 1, "start", "sum", 2)?.unwrap_or_else(|| w_int_new(0));
    if unsafe { pyre_object::is_str(start) } {
        return Err(crate::PyError::type_error(
            "sum() can't sum strings [use ''.join(seq) instead]",
        ));
    }
    if unsafe { pyre_object::is_bytes(start) } {
        return Err(crate::PyError::type_error(
            "sum() can't sum bytes [use b''.join(seq) instead]",
        ));
    }
    if unsafe { pyre_object::is_bytearray(start) } {
        return Err(crate::PyError::type_error(
            "sum() can't sum bytearray [use b''.join(seq) instead]",
        ));
    }
    // `_regular_sum`: `last = last + x` over the generic iterator protocol
    // (so generators, ranges, sets, dict views, ... all work).  Very
    // intentionally `last + x`, not `+=` — preserving a mutable `start`
    // (e.g. a list) matches PyPy's app-level definition.
    let mut last = start;
    for item in crate::builtins::collect_iterable(iterable)? {
        last = crate::baseobjspace::add(last, item)?;
    }
    Ok(last)
}

/// `round(number, ndigits=None)` — PyPy: operation.py round
/// Round half to even (banker's rounding), matching Python 3 semantics.
fn round_half_even(v: f64) -> f64 {
    let rounded = v.round();
    // When exactly halfway, round to even.
    if (v - rounded).abs() == 0.5 {
        let truncated = v.trunc();
        if truncated % 2.0 == 0.0 {
            truncated
        } else {
            rounded
        }
    } else {
        rounded
    }
}

/// `round(float, ndigits)` to `ndigits` decimal places, correctly rounded
/// (round-half-to-even) on the true binary value — `floatobject.c
/// double_round`, which formats with `_Py_dg_dtoa` mode 3 then parses the
/// decimal string back. Scaling by `10**ndigits` and rounding loses
/// precision (`2.675 * 100.0` rounds up to `267.5`, so the naive path
/// yields `2.68` where the true value `2.67499…` rounds to `2.67`); the
/// decimal-string round-trip avoids that.
fn float_round_ndigits(v: f64, ndigits: i64) -> f64 {
    // double_round bounds: beyond `NDIGITS_MAX` the value is unchanged;
    // below `NDIGITS_MIN` it collapses to a zero with the sign of `v`.
    const NDIGITS_MAX: i64 = 323;
    const NDIGITS_MIN: i64 = -308;
    if ndigits > NDIGITS_MAX {
        return v;
    }
    if ndigits < NDIGITS_MIN {
        return 0.0 * v;
    }
    if ndigits >= 0 {
        format!("{:.*}", ndigits as usize, v)
            .parse::<f64>()
            .unwrap_or(v)
    } else {
        let factor = 10f64.powi((-ndigits) as i32);
        round_half_even(v / factor) * factor
    }
}

pub(crate) fn builtin_round(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `round(number, ndigits=None)`: both positional-or-keyword; at most two.
    let (pos, kwargs) = split_builtin_kwargs(args);
    let total = pos.len() + real_kwarg_count(kwargs);
    if total > 2 {
        return Err(crate::PyError::type_error(format!(
            "round() takes at most 2 arguments ({total} given)"
        )));
    }
    kwarg_reject_unknown(kwargs, &["number", "ndigits"], "round")?;
    let obj = bind_pos_or_kw(pos, kwargs, 0, "number", "round", 1)?.ok_or_else(|| {
        crate::PyError::type_error("round() missing required argument 'number' (pos 1)")
    })?;
    let ndigits_arg = bind_pos_or_kw(pos, kwargs, 1, "ndigits", "round", 2)?;
    let ndigits = ndigits_arg.as_ref();
    unsafe {
        if is_float(obj) {
            let v = floatobject::w_float_get_value(obj);
            return match ndigits {
                // `floatobject.py:966-967 _round_float`: nan/inf round to
                // themselves when an explicit ndigits is supplied.  `ndigits`
                // is taken through `space.getindex_w`, so any `__index__`
                // object works and a non-index one raises.
                Some(nd) if !pyre_object::is_none(*nd) => {
                    let n = space_index_w(*nd)?;
                    if !v.is_finite() {
                        Ok(floatobject::w_float_new(v))
                    } else {
                        Ok(floatobject::w_float_new(float_round_ndigits(v, n)))
                    }
                }
                // `floatobject.py:954-960 _round_float`: single-argument
                // round routes through newint_from_float, which raises
                // ValueError on NaN and OverflowError on ±inf.
                _ => crate::typedef::float_to_pyint(
                    round_half_even(v),
                    crate::typedef::FloatToIntMode::Trunc,
                ),
            };
        }
        if is_int(obj) || is_long(obj) {
            // `longobject.c:long_round` — single-arg round and any
            // ndigits >= 0 leave an int unchanged; ndigits < 0 rounds to
            // the nearest multiple of 10**(-ndigits), ties to even.
            let nd = match ndigits {
                Some(nd) if !pyre_object::is_none(*nd) => space_index_w(*nd)?,
                _ => return Ok(obj),
            };
            if nd >= 0 {
                return Ok(obj);
            }
            use num_integer::Integer;
            let a = obj_to_bigint(obj);
            let mut b = BigInt::from(1);
            let ten = BigInt::from(10);
            for _ in 0..(-nd) {
                b = &b * &ten;
            }
            // `_PyLong_DivmodNear`: q = round(a / b) ties-to-even,
            // result = q * b.  Floor division gives 0 <= r < b.
            let (q, r) = a.div_mod_floor(&b);
            let two_r = &r * BigInt::from(2);
            let q_even = (&q % BigInt::from(2)) == BigInt::from(0);
            let q = if two_r < b {
                q
            } else if two_r > b {
                q + 1
            } else if q_even {
                q
            } else {
                q + 1
            };
            let result = q * b;
            return if pyre_object::jit_bigint_to_i64_fits(&result) != 0 {
                Ok(w_int_new(pyre_object::jit_bigint_to_i64_value(&result)))
            } else {
                Ok(w_long_new(result))
            };
        }
    }
    // operation.py:97 — lookup __round__ on user objects.  An omitted or
    // explicit-None `ndigits` calls `__round__()` with no second argument.
    if let Some(tp) = crate::typedef::r#type(obj) {
        if let Some(method) = unsafe { crate::baseobjspace::lookup_in_type(tp, "__round__") } {
            let result = match ndigits {
                Some(nd) if !unsafe { pyre_object::is_none(*nd) } => {
                    crate::call::call_function_impl_result(method, &[obj, *nd])?
                }
                _ => crate::call::call_function_impl_result(method, &[obj])?,
            };
            return Ok(result);
        }
    }
    let type_name = match crate::typedef::r#type(obj) {
        Some(tp) => unsafe { pyre_object::w_type_get_name(tp).to_string() },
        None => unsafe { (*(*obj).ob_type).name.to_string() },
    };
    Err(crate::PyError::type_error(format!(
        "type {} doesn't define __round__ method",
        type_name
    )))
}

/// `divmod(a, b)` — pypy/interpreter/baseobjspace.py:2159 divmod row.
fn builtin_divmod(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "divmod() takes no keyword arguments",
        ));
    }
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "divmod() takes exactly two arguments ({} given)",
            args.len()
        )));
    }
    crate::baseobjspace::divmod(args[0], args[1])
}

/// `pow(base, exp[, mod])` — pypy/interpreter/baseobjspace.py:2160 pow row.
fn builtin_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `pow(base, exp, mod=None)`: all positional-or-keyword; at most three.
    let (pos, kwargs) = split_builtin_kwargs(args);
    let total = pos.len() + real_kwarg_count(kwargs);
    if total > 3 {
        return Err(crate::PyError::type_error(format!(
            "pow() takes at most 3 arguments ({total} given)"
        )));
    }
    kwarg_reject_unknown(kwargs, &["base", "exp", "mod"], "pow")?;
    let base = bind_pos_or_kw(pos, kwargs, 0, "base", "pow", 1)?.ok_or_else(|| {
        crate::PyError::type_error("pow() missing required argument 'base' (pos 1)")
    })?;
    let exp = bind_pos_or_kw(pos, kwargs, 1, "exp", "pow", 2)?.ok_or_else(|| {
        crate::PyError::type_error("pow() missing required argument 'exp' (pos 2)")
    })?;
    let modulus = bind_pos_or_kw(pos, kwargs, 2, "mod", "pow", 3)?;
    match modulus {
        Some(m) if !unsafe { pyre_object::is_none(m) } => crate::baseobjspace::pow3(base, exp, m),
        _ => crate::baseobjspace::pow(base, exp),
    }
}

/// Coerce `obj` to a `BigInt` through the index protocol (`_operator.index`),
/// so `hex`/`oct`/`bin` accept any `__index__` object and arbitrarily large
/// integers.
fn index_to_bigint(obj: PyObjectRef) -> Result<BigInt, crate::PyError> {
    unsafe {
        if is_int(obj) || is_long(obj) {
            return Ok(obj_to_bigint(obj));
        }
        if pyre_object::is_bool(obj) {
            return Ok(BigInt::from(pyre_object::w_bool_get_value(obj) as i64));
        }
        if let Some(tp) = crate::typedef::r#type(obj) {
            if let Some(index_fn) = crate::baseobjspace::lookup_in_type(tp, "__index__") {
                let result = crate::call::call_function_impl_result(index_fn, &[obj])?;
                if is_int(result) || is_long(result) {
                    return Ok(obj_to_bigint(result));
                }
                if pyre_object::is_bool(result) {
                    return Ok(BigInt::from(pyre_object::w_bool_get_value(result) as i64));
                }
            }
        }
    }
    let tp_name = unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => "object".to_string(),
        }
    };
    Err(crate::PyError::type_error(format!(
        "'{tp_name}' object cannot be interpreted as an integer"
    )))
}

/// Format a `BigInt` in `radix` with the given `0x`/`0o`/`0b` prefix, keeping
/// the sign ahead of the prefix (`-0xff`).
fn format_int_radix(value: &BigInt, radix: u32, prefix: &str) -> String {
    let digits = value.to_str_radix(radix);
    match digits.strip_prefix('-') {
        Some(magnitude) => format!("-{prefix}{magnitude}"),
        None => format!("{prefix}{digits}"),
    }
}

/// `hex(x)` — PyPy: operation.py hex
fn builtin_hex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "hex() takes no keyword arguments",
        ));
    }
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "hex() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let s = format_int_radix(&index_to_bigint(args[0])?, 16, "0x");
    Ok(w_str_new(&s))
}

/// `oct(x)` — PyPy: operation.py oct
fn builtin_oct(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "oct() takes no keyword arguments",
        ));
    }
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "oct() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let s = format_int_radix(&index_to_bigint(args[0])?, 8, "0o");
    Ok(w_str_new(&s))
}

/// `bin(x)` — PyPy: operation.py bin
fn builtin_bin(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "bin() takes no keyword arguments",
        ));
    }
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "bin() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    let s = format_int_radix(&index_to_bigint(args[0])?, 2, "0b");
    Ok(w_str_new(&s))
}

/// Parse a complex literal string into `(real, imag)`.
///
/// `complexobject.c complex_from_string_inner`: optional surrounding
/// parens, then `[real][+/-]imag[j]` with no internal whitespace.
fn parse_complex_str(raw: &str) -> Option<(f64, f64)> {
    let mut s = raw.trim();
    if s.starts_with('(') && s.ends_with(')') {
        s = s[1..s.len() - 1].trim();
    }
    if s.is_empty() || s.contains(char::is_whitespace) {
        return None;
    }
    let parse_part = |p: &str, is_imag: bool| -> Option<f64> {
        // A bare/sign-only imaginary coefficient means ±1.
        if is_imag {
            match p {
                "" | "+" => return Some(1.0),
                "-" => return Some(-1.0),
                _ => {}
            }
        }
        p.parse::<f64>().ok()
    };
    let bytes = s.as_bytes();
    // Boundary between real and imaginary parts: the last '+'/'-' that is
    // not the leading sign and not part of an exponent (`e+`/`e-`).
    let mut split = None;
    for i in 1..bytes.len() {
        let c = bytes[i];
        if (c == b'+' || c == b'-') && bytes[i - 1] != b'e' && bytes[i - 1] != b'E' {
            split = Some(i);
        }
    }
    let ends_j = matches!(bytes.last(), Some(b'j') | Some(b'J'));
    match split {
        Some(i) => {
            if !ends_j {
                return None;
            }
            let real = parse_part(&s[..i], false)?;
            let imag = parse_part(&s[i..s.len() - 1], true)?;
            Some((real, imag))
        }
        None => {
            if ends_j {
                let imag = parse_part(&s[..s.len() - 1], true)?;
                Some((0.0, imag))
            } else {
                let real = parse_part(s, false)?;
                Some((real, 0.0))
            }
        }
    }
}

/// Coerce a value to `(real, imag)` for `complex()` construction.
///
/// `int`/`bool`/`float` become a real-only pair; a `complex` keeps both
/// components; an instance is asked for `__complex__` then `__float__`.
fn complex_coerce(obj: PyObjectRef) -> Result<(f64, f64), crate::PyError> {
    use pyre_object::*;
    unsafe {
        if is_complex(obj) {
            return Ok((w_complex_get_real(obj), w_complex_get_imag(obj)));
        }
        if is_bool(obj) {
            return Ok((w_bool_get_value(obj) as i64 as f64, 0.0));
        }
        if is_int(obj) {
            return Ok((w_int_get_value(obj) as f64, 0.0));
        }
        if is_long(obj) {
            return Ok((crate::baseobjspace::float_w(obj)?, 0.0));
        }
        if is_float(obj) {
            return Ok((w_float_get_value(obj), 0.0));
        }
    }
    // `__complex__` then `__float__` (complexobject.c try_complex_special_method).
    unsafe {
        if is_instance(obj) {
            let t = w_instance_get_type(obj);
            if crate::baseobjspace::lookup_in_type(t, "__complex__").is_some() {
                let res = crate::baseobjspace::call_method(obj, "__complex__", &[]);
                if res.is_null() {
                    return Err(crate::call::take_call_error()
                        .unwrap_or_else(|| crate::PyError::type_error("__complex__ call failed")));
                }
                if is_complex(res) {
                    return Ok((w_complex_get_real(res), w_complex_get_imag(res)));
                }
                return Err(crate::PyError::type_error(
                    "__complex__ should return a complex object",
                ));
            }
        }
    }
    let f = crate::baseobjspace::float_w(obj)?;
    Ok((f, 0.0))
}

/// `complex(real=0, imag=0)` — complexobject.c complex_new.
pub(crate) fn builtin_complex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::*;
    // `complex(real=0, imag=0)` — both arguments are positional-or-keyword
    // (complexobject.py descr__new__ `w_real`/`w_imag`).
    let (pos, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["real", "imag"], "complex")?;
    let w_real = resolve_pos_or_kw(pos.first().copied(), kwargs, "real", "complex", 1)?;
    let w_imag = resolve_pos_or_kw(pos.get(1).copied(), kwargs, "imag", "complex", 2)?;

    // String form accepts only the real argument.
    if let Some(a) = w_real {
        if unsafe { is_str(a) } {
            if w_imag.is_some() {
                return Err(crate::PyError::type_error(
                    "complex() can't take second arg if first is a string",
                ));
            }
            let s = unsafe { w_str_get_value(a) };
            let (r, i) = parse_complex_str(s).ok_or_else(|| {
                crate::PyError::new(
                    crate::PyErrorKind::ValueError,
                    format!("complex() arg is a malformed string"),
                )
            })?;
            return Ok(w_complex_new(r, i));
        }
    }
    let (mut real, mut imag) = match w_real {
        Some(a) => complex_coerce(a)?,
        None => (0.0, 0.0),
    };
    if let Some(b) = w_imag {
        if unsafe { is_str(b) } {
            return Err(crate::PyError::type_error(
                "complex() second arg can't be a string",
            ));
        }
        // complexobject.py:370-377 preserves signed zeroes by checking the
        // numeric components, not whether either operand is a complex object.
        let (br, bi) = complex_coerce(b)?;
        if bi != 0.0 {
            real -= bi;
        }
        if imag != 0.0 {
            imag += br;
        } else {
            imag = br;
        }
    }
    Ok(w_complex_new(real, imag))
}

/// `format(value, format_spec='')` — operation.py format → space.format
fn builtin_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "format() takes no keyword arguments",
        ));
    }
    if args.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "format() takes at least one argument ({} given)",
            args.len()
        )));
    }
    if args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "format expected at most 2 arguments, got {}",
            args.len()
        )));
    }
    let value = args[0];
    // `builtin_format_impl`: the `format_spec` must be a `str` — validated
    // here, before dispatch, so `format(value, 34)` reports `format()
    // argument 2 must be str, not int` for any `value`.  Its stored value is
    // used directly (no `str` subclass `__str__`), then `format_value_dispatch`
    // applies it — dispatching a `__format__` override (including a builtin
    // subclass's) or, for a plain builtin, the shared spec parser; the same
    // path f-string `{v:spec}` and `"{:spec}".format(v)` use.
    let spec = if args.len() > 1 {
        crate::type_methods::read_format_spec(args[1], "format() argument 2")?
    } else {
        String::new()
    };
    let s = crate::type_methods::format_value_dispatch(value, &spec)?;
    Ok(pyre_object::w_str_from_wtf8(s))
}

/// `__import__(name, globals=None, locals=None, fromlist=(), level=0)`
/// — PyPy: `pypy/module/imp/importing.py:importhook`.
fn builtin_import_stub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `__import__(name, globals, locals, fromlist, level)` — every parameter
    // may be passed by keyword (`__import__("a.b", fromlist=["c"])`), so the
    // positional slots fall back to the matching kwarg.
    let (pos, kwargs) = split_builtin_kwargs(args);
    let arg = |idx: usize, key: &str| -> PyObjectRef {
        pos.get(idx)
            .copied()
            .or_else(|| kwarg_get(kwargs, key))
            .unwrap_or(pyre_object::PY_NULL)
    };
    let name_obj = arg(0, "name");
    if name_obj.is_null() {
        return Err(crate::PyError::type_error(
            "__import__() missing required argument 'name' (pos 1)",
        ));
    }
    if !unsafe { pyre_object::is_str(name_obj) } {
        return Err(crate::PyError::type_error("module name must be a string"));
    }
    let name = unsafe { pyre_object::w_str_get_value(name_obj) };
    let globals = arg(1, "globals");
    let fromlist = arg(3, "fromlist");
    let level_obj = arg(4, "level");
    // `@unwrap_spec(level=int)` — an omitted level defaults to 0; a supplied
    // non-integer raises through the index protocol rather than defaulting.
    let level = if level_obj.is_null() {
        0
    } else {
        space_index_w(level_obj)?
    };
    let exec_ctx = crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            std::ptr::null::<crate::PyExecutionContext>()
        } else {
            unsafe { (*frame).execution_context }
        }
    });
    crate::importing::importhook(name, globals, fromlist, level, exec_ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The integer hashes now delegate to `rustpython_common::hash`
    /// (`_hash_int` = `fix_sentinel(mod_int(a))`, `_hash_long` =
    /// `hash_bigint`).  This locks the two crate entry points that reach
    /// the same Mersenne reduction to one value across the machine-word
    /// and big-integer ranges.
    #[test]
    fn common_hash_matches_pyre_int_helpers() {
        use rustpython_common::hash;
        let ints: [i64; 13] = [
            0,
            1,
            -1,
            42,
            -42,
            255,
            i64::MAX,
            i64::MIN,
            HASH_MODULUS as i64,
            HASH_MODULUS as i64 + 1,
            (HASH_MODULUS as i64).wrapping_neg(),
            1 << 40,
            -(1 << 40),
        ];
        for &a in &ints {
            assert_eq!(
                hash::hash_bigint(&BigInt::from(a)),
                _hash_int(a),
                "hash_bigint vs _hash_int for {a}"
            );
        }
        // -1 is the reserved sentinel, remapped to -2.
        assert_eq!(_hash_int(-1), -2);
        let bigs = [
            BigInt::from(2).pow(100) + BigInt::from(42),
            -(BigInt::from(2).pow(100)),
            BigInt::from(2).pow(200) - BigInt::from(1),
            BigInt::from(u128::MAX),
            -BigInt::from(u128::MAX),
        ];
        for b in &bigs {
            assert_eq!(
                hash::hash_bigint(b),
                _hash_long(b),
                "hash_bigint vs _hash_long"
            );
        }
    }

    /// `_hash_float` delegates to `rustpython_common::hash::hash_float`,
    /// which reproduces the reference `hash(5e-324) == 16777216` for
    /// subnormals, the `hash(2.0) == hash(2)` integral invariant, and the
    /// `±inf`/NaN sentinels.
    #[test]
    fn float_hash_delegates_to_common() {
        use rustpython_common::hash;
        // Subnormal reference point (the value that motivated the fix).
        assert_eq!(_hash_float(f64::from_bits(1)), 16777216);
        // Integral, zero, and sentinel reference points.
        assert_eq!(_hash_float(2.0), 2);
        assert_eq!(_hash_float(-1.0), -2);
        assert_eq!(_hash_float(0.0), 0);
        assert_eq!(_hash_float(-0.0), 0);
        assert_eq!(_hash_float(f64::INFINITY), 314159);
        assert_eq!(_hash_float(f64::NEG_INFINITY), -314159);
        // NaN hashes to HASH_NAN here; common returns None for it.
        assert_eq!(_hash_float(f64::NAN), HASH_NAN);
        assert_eq!(hash::hash_float(f64::NAN), None);
        // Differential battery of tricky finite floats: `_hash_float` must
        // equal `Some(hash_float(f))` on every one.
        let cases = [
            0.0f64,
            -0.0,
            f64::from_bits(1), // smallest positive subnormal
            -f64::from_bits(1),
            5e-324,
            1e-310,                  // subnormal
            2.2250738585072014e-308, // smallest normal
            0.1,
            -0.1,
            0.5,
            1.0,
            -1.0,
            1.5,
            -1.5,
            2.0,
            3.14,
            123.456,
            9.999e15,
            1e16,
            1e20,
            1e100,
            1e308,
            f64::MAX,
            f64::MIN,
            -123456789.123456789,
            9.995,
            268435456.0, // 2**28
            1.7976931348623157e308,
        ];
        for &f in &cases {
            assert_eq!(
                Some(_hash_float(f)),
                hash::hash_float(f),
                "hash_float divergence for {f}"
            );
        }
    }

    /// Empty `str`/`bytes` hash to 0; non-empty inputs take the siphash
    /// digest.
    #[test]
    fn empty_str_and_bytes_hash_to_zero() {
        assert_eq!(_hash_str(b""), 0);
        assert_eq!(hash_str_bytes(b""), 0);
        assert_ne!(_hash_str(b"a"), 0);
    }

    /// Tuple and frozenset hashes delegate to `rustpython_common::hash`
    /// (`hash_tuple` / `FrozenSetHash`).  Both fold only element hashes and are
    /// seed-independent, so the values are fixed and match CPython 3.14.
    /// `_hash_tuple_xx` / `_hash_frozenset` take the already-computed element
    /// hashes; small ints hash to themselves and `hash(-1) == -2`.
    #[test]
    fn tuple_and_frozenset_hash_match_cpython() {
        let h12 = _hash_tuple_xx(&[1, 2]);
        let h34 = _hash_tuple_xx(&[3, 4]);
        assert_eq!(_hash_tuple_xx(&[]), 5740354900026072187); // ()
        assert_eq!(_hash_tuple_xx(&[1]), -6644214454873602895); // (1,)
        assert_eq!(h12, -3550055125485641917); // (1, 2)
        assert_eq!(_hash_tuple_xx(&[1, 2, 3]), 529344067295497451);
        assert_eq!(_hash_tuple_xx(&[-2, 0, 1]), 5003556802939907908); // (-1, 0, 1)
        assert_eq!(_hash_tuple_xx(&[h12, h34]), -1467267874458550984); // ((1,2), (3,4))
        assert_eq!(_hash_tuple_xx(&[549755813930, -5]), 6589866070287121549); // (2**100+42, -5)

        // Frozenset fold is order-independent (XOR), so element order is free.
        let fs1 = _hash_frozenset(&[1]);
        let fs2 = _hash_frozenset(&[2]);
        assert_eq!(_hash_frozenset(&[]), 133146708735736); // frozenset()
        assert_eq!(fs1, -558064481276695278); // frozenset({1})
        assert_eq!(_hash_frozenset(&[1, 2, 3]), -272375401224217160);
        assert_eq!(_hash_frozenset(&[-2, 0, 1]), 8868930259606097796); // {-1, 0, 1}
        assert_eq!(_hash_frozenset(&[fs1, fs2]), 304806268181062474); // {fs{1}, fs{2}}
    }

    #[test]
    fn test_hash_rejects_tuple_containing_unhashable_key() {
        let value = w_tuple_new(vec![w_list_new(vec![])]);
        let err = builtin_hash(&[value]).expect_err("tuple hash should reject list element");

        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
    }

    #[test]
    fn test_builtin_divmod_delegates_through_proxy() {
        crate::typedef::init_typeobjects();
        let proxy = crate::module::_weakref::interp__weakref::W_Proxy_new(w_int_new(5), PY_NULL);
        let result = builtin_divmod(&[proxy, w_int_new(3)]).unwrap();
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 0).unwrap()) },
            1
        );
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 1).unwrap()) },
            2
        );
    }

    #[test]
    fn test_builtin_complex_preserves_imag_arg_negative_zero_with_complex_real() {
        let result = builtin_complex(&[w_complex_new(1.0, 0.0), w_float_new(-0.0)]).unwrap();
        assert_eq!(
            unsafe { w_complex_get_real(result).to_bits() },
            1.0f64.to_bits()
        );
        assert_eq!(
            unsafe { w_complex_get_imag(result).to_bits() },
            (-0.0f64).to_bits()
        );
    }

    #[test]
    fn test_builtin_divmod_allows_lhs_dunder_before_dead_proxy_rhs() {
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("DivmodLhs", |ns| {
            crate::dict_storage_store(
                ns,
                "__divmod__",
                make_builtin_function("__divmod__", |_| {
                    Ok(w_tuple_new(vec![w_int_new(41), w_int_new(1)]))
                }),
            );
        });
        let lhs = pyre_object::objectobject::w_instance_new(user_type);
        let dead_proxy = crate::module::_weakref::interp__weakref::W_Proxy_new(w_none(), PY_NULL);
        let result = builtin_divmod(&[lhs, dead_proxy]).unwrap();
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 0).unwrap()) },
            41
        );
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 1).unwrap()) },
            1
        );
    }

    #[test]
    fn test_builtin_pow_three_arg_delegates_through_proxy() {
        crate::typedef::init_typeobjects();
        let proxy = crate::module::_weakref::interp__weakref::W_Proxy_new(w_int_new(5), PY_NULL);
        let result = builtin_pow(&[proxy, w_int_new(3), w_int_new(13)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 8);
    }

    #[test]
    fn test_builtin_pow_two_arg_delegates_through_proxy() {
        crate::typedef::init_typeobjects();
        let proxy = crate::module::_weakref::interp__weakref::W_Proxy_new(w_int_new(5), PY_NULL);
        let result = builtin_pow(&[proxy, w_int_new(3)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 125);
    }

    #[test]
    fn test_builtin_pow_three_arg_allows_lhs_dunder_before_dead_proxy_exp() {
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("PowLhs", |ns| {
            crate::dict_storage_store(
                ns,
                "__pow__",
                make_builtin_function("__pow__", |_| Ok(w_int_new(99))),
            );
        });
        let lhs = pyre_object::objectobject::w_instance_new(user_type);
        let dead_proxy = crate::module::_weakref::interp__weakref::W_Proxy_new(w_none(), PY_NULL);
        let result = builtin_pow(&[lhs, dead_proxy, w_int_new(7)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 99);
    }

    #[test]
    fn test_builtin_pow_three_arg_negative_exponent_modular_inverse() {
        crate::typedef::init_typeobjects();
        // pow(5, -1, 13) is the modular inverse of 5 mod 13: 5*8 == 40 == 1.
        let result = builtin_pow(&[w_int_new(5), w_int_new(-1), w_int_new(13)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 8);
        // pow(3, -3, 7) == pow(pow(3, -1, 7), 3, 7) == 5^3 % 7 == 6.
        let cubed = builtin_pow(&[w_int_new(3), w_int_new(-3), w_int_new(7)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(cubed) }, 6);
    }

    #[test]
    fn test_builtin_pow_three_arg_non_invertible_base() {
        crate::typedef::init_typeobjects();
        // 2 and 4 share a factor, so 2 has no inverse modulo 4.
        let err = builtin_pow(&[w_int_new(2), w_int_new(-1), w_int_new(4)]).unwrap_err();
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        assert_eq!(err.message, "base is not invertible for the given modulus");
    }
}
