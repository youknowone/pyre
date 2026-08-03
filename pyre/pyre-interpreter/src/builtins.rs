use num_traits::ToPrimitive;
use pyre_object::rbigint::RBigInt as BigInt;

use crate::{
    make_builtin_function, make_builtin_function_with_arity, make_module_builtin_function,
    make_module_builtin_function_with_arity, make_module_builtin_function_with_doc,
};
use pyre_object::*;
use ruff_text_size::Ranged;
use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};
#[cfg(unix)]
use std::os::unix::ffi::OsStringExt;

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

/// True when `obj` is a `bytearray` (exact or subclass — a subclass shares the
/// primitive layout).  The `_exports` lock only tracks bytearray sources: the
/// sole resizable exporter carrying the lock (`array.array` has none).
unsafe fn backing_is_bytearray(obj: PyObjectRef) -> bool {
    unsafe {
        pyre_object::bytearrayobject::is_bytearray(obj)
            || crate::baseobjspace::isinstance_w(
                obj,
                crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE),
            )
    }
}

/// Take one `_exports` count on whichever resizable exporter ultimately backs
/// `buffer`, so the storage cannot be reallocated while a view over it is
/// alive.  `Buffer::sub` never nests, so one parent peel is sufficient.  The
/// matching release runs through the backing's `__release_buffer__`.
pub(crate) unsafe fn backing_exports_incref(buffer: &pyre_object::buffer::Buffer) {
    use pyre_object::buffer::Buffer;
    let root = match buffer {
        Buffer::Sub { parent, .. } => parent.as_ref(),
        other => other,
    };
    unsafe {
        match root {
            Buffer::Byte { w_obj } => {
                pyre_object::bytearrayobject::w_bytearray_exports_incref(*w_obj)
            }
            Buffer::Array { w_obj } => pyre_object::interp_array::w_array_exports_incref(*w_obj),
            Buffer::External { w_obj, .. } => {
                // The only counted external exporter is `mmap` (ctypes'
                // `memoryview_at` stores `None` and keeps no count), so a
                // derived mmap view takes its own count here — otherwise
                // releasing the view it came from would let `close`/`resize`
                // unmap while this one still reads the mapping.
                let _ = w_obj;
                #[cfg(all(unix, not(feature = "sandbox")))]
                if crate::module::mmap::interp_mmap::is_mmap(*w_obj) {
                    crate::module::mmap::interp_mmap::mmap_exports_incref(*w_obj);
                }
            }
            _ => {}
        }
    }
}

/// Acquire one live buffer export directly from a Python exporter.
///
/// This is the object-level half of `space.acquire_py_buffer`: callers keep
/// the exporter itself in a traced field and this function accounts for the
/// matching `bf_releasebuffer` obligation.  Immutable `bytes` needs no
/// release and returns `false`; the mutable/runtime-owned exporters return
/// `true` and must later be paired with [`buffer_export_decref`].
///
/// PyPy keeps this state on the acquired `Py_buffer`/`Buffer` object.  Until
/// pyre's generic `Py_buffer` carrier is shared by every consumer, keeping the
/// boolean beside the owning object field preserves the same single-owner
/// shape without an address-keyed side table.
pub(crate) unsafe fn buffer_export_incref(obj: PyObjectRef) -> bool {
    unsafe {
        if pyre_object::bytearrayobject::is_bytearray(obj) {
            pyre_object::bytearrayobject::w_bytearray_exports_incref(obj);
            return true;
        }
        if pyre_object::interp_array::is_array(obj) {
            pyre_object::interp_array::w_array_exports_incref(obj);
            return true;
        }
        if pyre_object::memoryview::is_w_memoryview(obj) {
            pyre_object::memoryview::w_memoryview_exports_incref(obj);
            return true;
        }
        #[cfg(all(unix, not(feature = "sandbox")))]
        if crate::module::mmap::interp_mmap::is_mmap(obj) {
            crate::module::mmap::interp_mmap::mmap_exports_incref(obj);
            return true;
        }
    }
    false
}

/// Release one export acquired by [`buffer_export_incref`].
///
/// # Safety
/// `obj` must still be the exporter paired with a successful (`true`)
/// acquisition, and the pair must be released exactly once.
pub(crate) unsafe fn buffer_export_decref(obj: PyObjectRef) {
    unsafe {
        if pyre_object::bytearrayobject::is_bytearray(obj) {
            pyre_object::bytearrayobject::w_bytearray_exports_decref(obj);
        } else if pyre_object::interp_array::is_array(obj) {
            pyre_object::interp_array::w_array_exports_decref(obj);
        } else if pyre_object::memoryview::is_w_memoryview(obj) {
            pyre_object::memoryview::w_memoryview_exports_decref(obj);
        } else {
            #[cfg(all(unix, not(feature = "sandbox")))]
            crate::module::mmap::interp_mmap::mmap_exports_decref(obj);
        }
    }
}

/// `_check_exports` — reject a size-changing mutation of a bytearray while a
/// buffer export (a live memoryview) is outstanding.
pub(crate) unsafe fn bytearray_check_exports(obj: PyObjectRef) -> Result<(), crate::PyError> {
    if unsafe { pyre_object::bytearrayobject::w_bytearray_exports(obj) } > 0 {
        // A tracing collector does not reclaim an expression-temporary
        // `memoryview` at the end of the statement as CPython's refcounting
        // does.  Make non-moving major progress before rejecting the resize:
        // dead stable-allocated views run `memoryview_object_destructor` and
        // release their backing export, while a live view remains counted.
        // `collect_oldgen` is essential here: callers still hold `obj` in a
        // by-value interpreter local, so this check must not move it.
        pyre_object::gc_hook::try_gc_collect_oldgen();
    }
    if unsafe { pyre_object::bytearrayobject::w_bytearray_exports(obj) } > 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::BufferError,
            "Existing exports of data: object cannot be re-sized",
        ));
    }
    Ok(())
}

/// Wrap native per-dimension extents (a `shape` or `strides`) into a fresh
/// `tuple[int]` for the `descr` getters.  Each int is pinned as built, so a
/// later element's allocation cannot strand an earlier one before
/// `w_tuple_new` roots the whole set.
unsafe fn memoryview_wrap_dims(dims: &[i64]) -> PyObjectRef {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        for &d in dims {
            pyre_object::gc_roots::pin_root(w_int_new(d));
        }
        let items: Vec<PyObjectRef> = (0..dims.len())
            .map(|i| pyre_object::gc_roots::shadow_stack_get(sp + i))
            .collect();
        pyre_object::w_tuple_new(items)
    }
}

/// Allocate a `W_MemoryView` whose view DERIVES from an existing view — the
/// copy (`W_MemoryView.copy`) and zero-copy slice (`new_slice`) constructors.
/// PyPy hands the same immutable view object to the derived memoryview; pyre
/// clones it into the new owner's box via `derive`.
///
/// GC-safety: the source memoryview is pinned across the header allocation
/// (the sole collection point).  Its custom trace keeps every ref inside its
/// off-heap box alive and updated in place, so `derive` — which must not
/// allocate on the GC heap — runs on post-collection refs.
unsafe fn w_memoryview_new_derived(
    mv_src: PyObjectRef,
    derive: impl FnOnce(&pyre_object::bufferview::BufferView) -> pyre_object::bufferview::BufferView,
) -> PyObjectRef {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(mv_src);
        // A derived view stays alive independently of the view it came from,
        // so it takes its own `_exports` count on the backing rather than
        // relying on the source's: releasing the source must not leave this
        // view pointing at storage the exporter is then free to reallocate.
        let mv = pyre_object::memoryview::w_memoryview_alloc_header(false, true);
        let r_src = pyre_object::gc_roots::shadow_stack_get(sp);
        let view = derive(pyre_object::memoryview::w_memoryview_view(r_src));
        let view_ptr = pyre_object::memoryview::bufferview_alloc(view);
        pyre_object::memoryview::w_memoryview_set_view(mv, view_ptr);
        backing_exports_incref(pyre_object::memoryview::w_memoryview_view(mv).backing());
        mv
    }
}

/// `_cast_to_1D` (`memoryobject.py:635`) — a `View1D` reinterpreting the
/// source view's bytes under a new 1-D element format.  The source
/// memoryview and the fresh format object are pinned across the header
/// allocation (the sole collection point); the source view then clones over
/// as the boxed parent with post-collection refs.
unsafe fn w_memoryview_cast_1d(mv_src: PyObjectRef, fmt: &str, itemsize: i64) -> PyObjectRef {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(mv_src);
        pyre_object::gc_roots::pin_root(w_str_new(fmt));
        let mv = pyre_object::memoryview::w_memoryview_alloc_header(false, true);
        let r_src = pyre_object::gc_roots::shadow_stack_get(sp);
        let r_fmt = pyre_object::gc_roots::shadow_stack_get(sp + 1);
        let src_view = pyre_object::memoryview::w_memoryview_view(r_src);
        let view = pyre_object::bufferview::BufferView::View1D {
            parent: Box::new(src_view.clone()),
            w_obj: src_view.w_obj(),
            w_fmt: r_fmt,
            itemsize,
        };
        let view_ptr = pyre_object::memoryview::bufferview_alloc(view);
        pyre_object::memoryview::w_memoryview_set_view(mv, view_ptr);
        backing_exports_incref(pyre_object::memoryview::w_memoryview_view(mv).backing());
        mv
    }
}

/// `descr_cast` with a shape — `_cast_to_1D` then `_cast_to_ND`
/// (`memoryobject.py:599-603`): a `ViewND` reshaping a fresh `View1D` over
/// the source view.  The shape / strides tuples are built and pinned inside
/// the rooted region alongside the source memoryview and format object.
unsafe fn w_memoryview_cast_nd(
    mv_src: PyObjectRef,
    fmt: &str,
    itemsize: i64,
    shape: &[i64],
    strides: &[i64],
) -> PyObjectRef {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(mv_src);
        // Build each geometry object and pin it as produced: a later allocation
        // may relocate an earlier one, so the pinned shadow slot (re-read below)
        // is the source of truth, not the stale local.
        pyre_object::gc_roots::pin_root(w_str_new(fmt));
        pyre_object::gc_roots::pin_root(memoryview_wrap_dims(shape));
        pyre_object::gc_roots::pin_root(memoryview_wrap_dims(strides));
        let mv = pyre_object::memoryview::w_memoryview_alloc_header(false, true);
        let r_src = pyre_object::gc_roots::shadow_stack_get(sp);
        let r_fmt = pyre_object::gc_roots::shadow_stack_get(sp + 1);
        let r_shape = pyre_object::gc_roots::shadow_stack_get(sp + 2);
        let r_strides = pyre_object::gc_roots::shadow_stack_get(sp + 3);
        let src_view = pyre_object::memoryview::w_memoryview_view(r_src);
        let view = pyre_object::bufferview::BufferView::ViewND {
            parent: Box::new(pyre_object::bufferview::BufferView::View1D {
                parent: Box::new(src_view.clone()),
                w_obj: src_view.w_obj(),
                w_fmt: r_fmt,
                itemsize,
            }),
            w_obj: src_view.w_obj(),
            ndim: shape.len() as i64,
            w_shape: r_shape,
            w_strides: r_strides,
        };
        let view_ptr = pyre_object::memoryview::bufferview_alloc(view);
        pyre_object::memoryview::w_memoryview_set_view(mv, view_ptr);
        backing_exports_incref(pyre_object::memoryview::w_memoryview_view(mv).backing());
        mv
    }
}

/// PyPy `interp_buffer.newmemoryview`'s `FormatBufferViewND`: retain the
/// source export while replacing only the public format/itemsize and geometry.
/// Unlike `memoryview.cast`, the format may be a structured (non-native)
/// string, so this helper intentionally skips cast's format validation.
pub(crate) unsafe fn w_memoryview_new_formatted_nd(
    mv_src: PyObjectRef,
    fmt: &str,
    itemsize: i64,
    shape: &[i64],
    strides: &[i64],
) -> PyObjectRef {
    unsafe { w_memoryview_cast_nd(mv_src, fmt, itemsize, shape, strides) }
}

/// Build a `memoryview` over a plain contiguous 1-D exporter — a `SimpleView`
/// (`bytes` / `bytearray`, derived format `'B'`) or a `RawBufferView`
/// (`array.array`, explicit format).  The exporter's `Buffer` variant picks
/// which, and both derive shape / strides / ndim / offset, so no geometry
/// Python objects are constructed; only a
/// `Raw` view keeps a format object.  Pins the exporter (and, for `Raw`, the
/// format) across the header allocation (the sole collection point), then
/// re-reads them relocated before building the off-heap box.
unsafe fn w_memoryview_new_plain(
    w_obj: PyObjectRef,
    fmt: &str,
    itemsize: i64,
    length: i64,
) -> PyObjectRef {
    use pyre_object::bufferview::BufferView;
    unsafe {
        let array_ty = crate::typedef::gettypeobject(&pyre_object::interp_array::ARRAY_TYPE);
        let is_array = pyre_object::interp_array::is_array(w_obj)
            || crate::baseobjspace::isinstance_w(w_obj, array_ty);
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(w_obj);
        // A Raw view keeps its explicit format object; a Simple view derives 'B'.
        if is_array {
            pyre_object::gc_roots::pin_root(w_str_new(fmt));
        }
        // Root view over an exporter: it owns the backing's buffer export.
        let mv = pyre_object::memoryview::w_memoryview_alloc_header(false, true);
        let r_obj = pyre_object::gc_roots::shadow_stack_get(sp);
        // `buffer_w`: record the export on a bytearray backing so a
        // size-changing mutation is refused while this view is live.
        if backing_is_bytearray(r_obj) {
            pyre_object::bytearrayobject::w_bytearray_exports_incref(r_obj);
        } else if is_array {
            pyre_object::interp_array::w_array_exports_incref(r_obj);
        }
        let backing = memoryview_backing_buffer(r_obj);
        let view = if is_array {
            let r_fmt = pyre_object::gc_roots::shadow_stack_get(sp + 1);
            BufferView::Raw {
                backing,
                w_obj: r_obj,
                w_fmt: r_fmt,
                itemsize,
                length,
            }
        } else {
            BufferView::Simple {
                backing,
                w_obj: r_obj,
                length,
            }
        };
        let view_ptr = pyre_object::memoryview::bufferview_alloc(view);
        pyre_object::memoryview::w_memoryview_set_view(mv, view_ptr);
        mv
    }
}

/// Build the `W_MMap.readbuf_w`/`writebuf_w` view: one contiguous external
/// byte window whose owner remains the mmap object.
#[cfg(all(unix, not(feature = "sandbox")))]
unsafe fn w_memoryview_new_mmap(
    w_obj: PyObjectRef,
    address: usize,
    length: usize,
    readonly: bool,
) -> PyObjectRef {
    use pyre_object::buffer::Buffer;
    use pyre_object::bufferview::BufferView;
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(w_obj);
        let mv = pyre_object::memoryview::w_memoryview_alloc_header(false, true);
        let r_obj = pyre_object::gc_roots::shadow_stack_get(sp);
        let view = BufferView::Simple {
            backing: Buffer::External {
                w_obj: r_obj,
                address,
                size: length,
                readonly,
            },
            w_obj: r_obj,
            length: length as i64,
        };
        let view_ptr = pyre_object::memoryview::bufferview_alloc(view);
        pyre_object::memoryview::w_memoryview_set_view(mv, view_ptr);
        // Unlike the GC-owned exporters, the mapping is foreign memory that
        // `close`/`resize` hand straight back to the kernel, so the window
        // must keep it from being unmapped while this view can still read it.
        crate::module::mmap::interp_mmap::mmap_exports_incref(r_obj);
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
///
/// The gather walks the exporter's backing storage through pointer
/// arithmetic and grows a `Vec<u8>` (`buffer.py:117-127 _copy_base`, an
/// rstring `StringBuilder` append the tracer does not model element by
/// element); the sub-slice `&full[b..b+isz]` is a windowed copy, not a
/// value-model view.  Residualize the whole geometry/copy subtree behind
/// this single `.gather()` call surface (`@jit.dont_look_inside`).
#[majit_macros::dont_look_inside]
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

/// Is `descr` one of pyre's native `bf_getbuffer` wrappers?
///
/// CPython installs `slot_bf_getbuffer` only when the descriptor selected by
/// the concrete type's MRO is Python-defined.  Comparing the selected object
/// with each native base descriptor preserves that slot decision for mixed
/// inheritance: `class C(bytearray, A)` stays native, while
/// `class C(A, bytearray)` reaches `A.__buffer__`.
unsafe fn memoryview_is_native_buffer_descr(descr: PyObjectRef) -> bool {
    unsafe {
        for base in [
            crate::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE),
            crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE),
            crate::typedef::gettypeobject(&pyre_object::memoryview::MEMORYVIEW_TYPE),
            crate::typedef::gettypeobject(&pyre_object::interp_array::ARRAY_TYPE),
        ] {
            if crate::baseobjspace::lookup_in_type(base, "__buffer__")
                .is_some_and(|native| std::ptr::eq(native, descr))
            {
                return true;
            }
        }
        false
    }
}

/// The Python `__buffer__` descriptor selected for `obj`, or `None` when the
/// type uses a native buffer slot (or has no Python-visible descriptor).
unsafe fn memoryview_python_buffer_descr(obj: PyObjectRef) -> Option<PyObjectRef> {
    unsafe {
        let descr = crate::baseobjspace::lookup(obj, "__buffer__")?;
        (!memoryview_is_native_buffer_descr(descr)).then_some(descr)
    }
}

/// CPython 3.14 `slot_bf_getbuffer`: call Python `__buffer__(flags)`, acquire
/// the returned memoryview, allocate `_buffer_wrapper(mv, obj)`, and copy its
/// geometry while replacing only `Py_buffer.obj` with that wrapper.
unsafe fn w_memoryview_new_python_buffer(
    w_obj: PyObjectRef,
    w_descr: PyObjectRef,
    flags: i32,
) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        use pyre_object::memoryview::*;
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(w_obj);
        pyre_object::gc_roots::pin_root(w_descr);
        pyre_object::gc_roots::pin_root(w_int_new(flags as i64));

        let r_obj = pyre_object::gc_roots::shadow_stack_get(sp);
        let w_type = crate::typedef::r#type(r_obj).map_or(r_obj, |p| p.as_ptr());
        let w_ret = crate::baseobjspace::get_and_call_function(
            pyre_object::gc_roots::shadow_stack_get(sp + 1),
            r_obj,
            w_type,
            &[pyre_object::gc_roots::shadow_stack_get(sp + 2)],
        )?;
        if !is_w_memoryview(w_ret) {
            return Err(crate::PyError::type_error(
                "__buffer__ returned non-memoryview object",
            ));
        }
        pyre_object::gc_roots::pin_root(w_ret);
        let ret_slot = sp + 3;
        let r_ret = pyre_object::gc_roots::shadow_stack_get(ret_slot);
        memoryview_check_released(r_ret)?;
        if w_memoryview_restricted(r_ret) {
            return Err(crate::PyError::value_error(
                "cannot create a new view from a restricted memoryview",
            ));
        }
        if flags & 0x0001 != 0 && w_memoryview_readonly(r_ret) {
            return Err(crate::PyError::new(
                crate::PyErrorKind::BufferError,
                "Object is not writable.",
            ));
        }

        // PyObject_GetBuffer(ret, buffer, flags): the returned memoryview may
        // not be released while this copied Py_buffer is outstanding.
        w_memoryview_exports_incref(r_ret);
        let wrapper = w_buffer_wrapper_new(
            pyre_object::gc_roots::shadow_stack_get(ret_slot),
            pyre_object::gc_roots::shadow_stack_get(sp),
        );
        pyre_object::gc_roots::pin_root(wrapper);
        let wrapper_slot = sp + 4;
        let outer = w_memoryview_alloc_header(false, true);
        let r_ret = pyre_object::gc_roots::shadow_stack_get(ret_slot);
        let r_wrapper = pyre_object::gc_roots::shadow_stack_get(wrapper_slot);
        let copied = w_memoryview_view(r_ret).clone_with_obj(r_wrapper);
        w_memoryview_set_view(outer, bufferview_alloc(copied));
        Ok(outer)
    }
}

/// `memoryview(obj)` — acquire a 1-D byte view over a buffer-providing
/// exporter.  Sharing another memoryview copies its view parameters (and
/// reports the original exporter as `.obj`); a non-buffer raises TypeError.
pub(crate) fn w_memoryview_new(w_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    // `PyMemoryView_FromObject` requests the full read-only buffer interface.
    w_memoryview_new_with_flags(w_obj, 0x011c)
}

/// `memoryview._from_flags` / native `bf_getbuffer` entry with an explicit
/// CPython `PyBUF_*` mask.
pub(crate) fn w_memoryview_new_with_flags(
    w_obj: PyObjectRef,
    flags: i32,
) -> Result<PyObjectRef, crate::PyError> {
    w_memoryview_new_with_flags_impl(w_obj, flags, true)
}

/// Invoke a built-in exporter's native `bf_getbuffer` implementation.
///
/// Python `super().__buffer__(flags)` reaches the base slot directly; it must
/// not redispatch to the concrete subclass's Python `__buffer__`, which would
/// recurse back into the method that called `super()`.
pub(crate) fn w_memoryview_new_native_with_flags(
    w_obj: PyObjectRef,
    flags: i32,
) -> Result<PyObjectRef, crate::PyError> {
    w_memoryview_new_with_flags_impl(w_obj, flags, false)
}

fn w_memoryview_new_with_flags_impl(
    w_obj: PyObjectRef,
    flags: i32,
    allow_python_slot: bool,
) -> Result<PyObjectRef, crate::PyError> {
    // CPython abstract.c:PyBuffer_FillInfo — PyBUF_READ and PyBUF_WRITE are
    // request kinds for PyMemoryView_GetContiguous, not valid standalone
    // getbuffer flags.  (Their bits may still appear in valid composite
    // masks such as PyBUF_FULL_RO.)
    if flags == 0x0100 || flags == 0x0200 {
        return Err(crate::PyError::system_error(
            "bad argument to internal function",
        ));
    }
    use pyre_object::memoryview::*;
    if let Some(target) = crate::module::__pypy__::interp_buffer::forwarded_exporter(w_obj) {
        return w_memoryview_new_with_flags_impl(target?, flags, allow_python_slot);
    }
    unsafe {
        if allow_python_slot {
            if let Some(w_descr) = memoryview_python_buffer_descr(w_obj) {
                return w_memoryview_new_python_buffer(w_obj, w_descr, flags);
            }
        }
        if is_w_memoryview(w_obj) {
            memoryview_check_released(w_obj)?;
            if w_memoryview_restricted(w_obj) {
                return Err(crate::PyError::value_error(
                    "cannot create a new view from a restricted memoryview",
                ));
            }
            // `W_MemoryView.copy` shares the source's (immutable) view; the
            // clone preserves the variant, so copying a sliced / plain view
            // keeps its zero-copy window and derived geometry.
            return Ok(w_memoryview_new_derived(w_obj, |v| v.clone()));
        }
        #[cfg(all(unix, not(feature = "sandbox")))]
        if let Some(view) = crate::module::mmap::interp_mmap::mmap_buffer_view(w_obj) {
            let (address, length, readonly) = view?;
            return Ok(w_memoryview_new_mmap(w_obj, address, length, readonly));
        }
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        if let Some((backing_obj, offset, byte_len, fmt, itemsize, shape)) =
            crate::module::_ctypes::cdata::cdata_buffer_view(w_obj)
        {
            use pyre_object::buffer::Buffer;
            use pyre_object::bufferview::BufferView;
            let _roots = pyre_object::gc_roots::push_roots();
            let sp = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(w_obj);
            pyre_object::gc_roots::pin_root(backing_obj);
            pyre_object::gc_roots::pin_root(w_str_new(&fmt));
            let shape_i64 = shape.iter().map(|&dim| dim as i64).collect::<Vec<_>>();
            let mut strides = vec![0i64; shape.len()];
            let mut stride = itemsize as i64;
            for (i, &dim) in shape_i64.iter().enumerate().rev() {
                strides[i] = stride;
                stride = stride.saturating_mul(dim);
            }
            pyre_object::gc_roots::pin_root(memoryview_wrap_dims(&shape_i64));
            pyre_object::gc_roots::pin_root(memoryview_wrap_dims(&strides));
            let mv = w_memoryview_alloc_header(false, true);
            let r_obj = pyre_object::gc_roots::shadow_stack_get(sp);
            let r_backing = pyre_object::gc_roots::shadow_stack_get(sp + 1);
            let r_fmt = pyre_object::gc_roots::shadow_stack_get(sp + 2);
            let r_shape = pyre_object::gc_roots::shadow_stack_get(sp + 3);
            let r_strides = pyre_object::gc_roots::shadow_stack_get(sp + 4);
            pyre_object::bytearrayobject::w_bytearray_exports_incref(r_backing);
            let backing = Buffer::sub(Buffer::Byte { w_obj: r_backing }, offset, byte_len as i64);
            let raw = BufferView::Raw {
                backing,
                w_obj: r_obj,
                w_fmt: r_fmt,
                itemsize: itemsize as i64,
                length: byte_len as i64,
            };
            let view = BufferView::ViewND {
                parent: Box::new(raw),
                w_obj: r_obj,
                ndim: shape.len() as i64,
                w_shape: r_shape,
                w_strides: r_strides,
            };
            let view_ptr = pyre_object::memoryview::bufferview_alloc(view);
            pyre_object::memoryview::w_memoryview_set_view(mv, view_ptr);
            return Ok(mv);
        }
        let (fmt, itemsize, readonly, byte_len) = match memoryview_buffer_params(w_obj) {
            Some(p) => p,
            None => {
                let tname = crate::typedef::r#type(w_obj)
                    .map(|t| pyre_object::w_type_get_name(t.as_ptr()))
                    .unwrap_or("object");
                return Err(crate::PyError::type_error(&format!(
                    "memoryview: a bytes-like object is required, not '{tname}'"
                )));
            }
        };
        if flags & 0x0001 != 0 && readonly {
            return Err(crate::PyError::new(
                crate::PyErrorKind::BufferError,
                "Object is not writable.",
            ));
        }
        // A plain view derives its geometry: a bytes / bytearray backing builds
        // a `SimpleView`, an array.array a `RawBufferView` (readonly follows the
        // backing kind).
        Ok(w_memoryview_new_plain(
            w_obj,
            &fmt,
            itemsize,
            byte_len as i64,
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
        b'e' => {
            let bits = u16::from_ne_bytes(buf.try_into().unwrap());
            w_float_new(crate::module::r#struct::unpack_half(bits))
        }
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
            match unsafe { crate::baseobjspace::space_index(w_val) } {
                Ok(index) => Ok(index),
                Err(err) if err.kind == crate::PyErrorKind::TypeError => Err(bad_type()),
                Err(err) => Err(err),
            }
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
            // `PackFormatIterator.accept_float_arg`: use `space.float_w`,
            // including a user `__float__`; only TypeError becomes the
            // memoryview format error and other user exceptions propagate.
            let v = match crate::baseobjspace::float_w(w_val) {
                Ok(value) => value,
                Err(err) if err.kind == crate::PyErrorKind::TypeError => return Err(bad_type()),
                Err(err) => return Err(err),
            } as f32;
            v.to_ne_bytes().to_vec()
        }
        b'd' => {
            let v = match crate::baseobjspace::float_w(w_val) {
                Ok(value) => value,
                Err(err) if err.kind == crate::PyErrorKind::TypeError => return Err(bad_type()),
                Err(err) => return Err(err),
            };
            v.to_ne_bytes().to_vec()
        }
        b'e' => {
            let v = match crate::baseobjspace::float_w(w_val) {
                Ok(value) => value,
                Err(err) if err.kind == crate::PyErrorKind::TypeError => return Err(bad_type()),
                Err(err) => return Err(err),
            };
            crate::module::r#struct::pack_half(v)?
                .to_ne_bytes()
                .to_vec()
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
        let fmt = pyre_object::memoryview::w_memoryview_format_str(mv);
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

/// A live sub-view `m[start:stop:step]` sharing the same storage
/// (`descr_getitem` slice arm → `view.new_slice(start, step, slicelength)`).
/// A step==1 slice of a plain view stays a `Simple` / `Raw` view over a
/// `Buffer::Sub` window; a strided slice wraps the view in a
/// `BufferView::Slice`, whose dimension-0 shape / stride derive from the
/// parent's, so slicing an N-D view keeps its dimensionality.
unsafe fn memoryview_slice_view(
    mv: PyObjectRef,
    index: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::memoryview::*;
    unsafe {
        if w_memoryview_restricted(mv) {
            return Err(crate::PyError::value_error(
                "cannot create a new view from a restricted memoryview",
            ));
        }
        let ndim = w_memoryview_ndim(mv);
        let count = if ndim >= 1 {
            w_memoryview_native_shape(mv).first().copied().unwrap_or(0)
        } else {
            0
        };

        // CPython 3.14 `memory_subscript`: `mbuf_add_view(self->mbuf, view)`
        // registers the result *before* `init_slice` evaluates the slice
        // bounds.  A bound's `__index__` may release `mv`; the registered
        // result must keep the export alive and continue from its private
        // view snapshot (gh-92888).
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(mv);
        pyre_object::gc_roots::pin_root(index);
        let sliced = w_memoryview_alloc_header(false, true);
        let r_mv = pyre_object::gc_roots::shadow_stack_get(sp);
        let snapshot = w_memoryview_view(r_mv).clone();
        // The root `Buffer` tag was fixed by `buffer_w`; follow it directly
        // instead of repeating an objspace isinstance lookup in this hot/JIT
        // path.  `Buffer::sub` never nests, so one parent peel is sufficient.
        backing_exports_incref(snapshot.backing());
        w_memoryview_set_view(sliced, bufferview_alloc(snapshot));
        pyre_object::gc_roots::pin_root(sliced);

        let r_index = pyre_object::gc_roots::shadow_stack_get(sp + 1);
        let (start, stop, step) = crate::baseobjspace::normalize_slice(r_index, count)?;
        let slicelength = if (step < 0 && stop >= start) || (step > 0 && start >= stop) {
            0
        } else if step < 0 {
            (stop - start + 1) / step + 1
        } else {
            (stop - start - 1) / step + 1
        };
        let r_sliced = pyre_object::gc_roots::shadow_stack_get(sp + 2);
        let old_view =
            (*(r_sliced as *mut W_MemoryView)).view as *mut pyre_object::bufferview::BufferView;
        let view = (&*old_view).new_slice(start, step, slicelength);
        w_memoryview_set_view(r_sliced, bufferview_alloc(view));
        drop(Box::from_raw(old_view));
        Ok(r_sliced)
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
        b'h' | b'H' | b'e' => 2,
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
        let shape = w_memoryview_native_shape(mv);
        let strides = w_memoryview_native_strides(mv);
        let nitems = shape.get(dim as usize).copied().unwrap_or(0);
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
        Ok(strides.get(dim as usize).copied().unwrap_or(0) * idx)
    }
}

/// `_PyIndex_Check(v)` — whether the type exposes `nb_index` at all.
///
/// This is a *type* test and never a conversion, which is what lets a caller
/// substitute its own "not an index" message for a miss while anything
/// `__index__` itself raises still propagates unchanged.  Converting here
/// instead would both run a user slot twice and relabel its exception.
pub(crate) unsafe fn index_check(w: PyObjectRef) -> bool {
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
            if !index_check(w) {
                return Err(crate::PyError::type_error("memoryview: invalid slice key"));
            }
            let index = getindex_w(w)?;
            // memoryobject.py `_start_from_tuple`: `__index__` is arbitrary
            // Python code and may release this memoryview before the offset
            // reads its view geometry.
            memoryview_check_released(mv)?;
            start += memoryview_get_offset(mv, dim, index)?;
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
            if !index_check(w) {
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
        if index_check(index) {
            if ndim == 0 {
                return Err(crate::PyError::type_error(
                    "invalid indexing of 0-dim memory",
                ));
            }
            let itemsize = w_memoryview_itemsize(mv);
            let length = w_memoryview_length(mv);
            let mut i = getindex_w(index)?;
            // memoryobject.py `descr_getitem`: `_decode_index` may invoke a
            // user `__index__` which releases the view.
            memoryview_check_released(mv)?;
            if ndim != 1 {
                return Err(crate::PyError::not_implemented(
                    "multi-dimensional sub-views are not implemented",
                ));
            }
            let count = if itemsize > 0 { length / itemsize } else { 0 };
            if i < 0 {
                i += count;
            }
            if i < 0 || i >= count {
                return Err(crate::PyError::index_error("index out of bounds"));
            }
            let base = (w_memoryview_offset(mv) + i * w_memoryview_stride0(mv)) as usize;
            let full = w_memoryview_view(mv).backing().as_bytes();
            let fmt = w_memoryview_format_str(mv);
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
                let full = w_memoryview_view(mv).backing().as_bytes();
                let fmt = w_memoryview_format_str(mv);
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
        if w_memoryview_view(mv).backing().as_bytes_mut().is_none() {
            return Err(crate::PyError::type_error("cannot modify read-only memory"));
        }
        let itemsize = w_memoryview_itemsize(mv);
        let isz = itemsize.max(0) as usize;
        let fmt = w_memoryview_format_str(mv).to_owned();
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
            // `decode_index4` evaluates arbitrary slice-bound `__index__`
            // methods before the assignment touches the backing.
            memoryview_check_released(mv)?;
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
            let full = w_memoryview_view(mv)
                .backing()
                .as_bytes_mut()
                .expect("writable backing checked above");
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
            let full = w_memoryview_view(mv)
                .backing()
                .as_bytes_mut()
                .expect("writable backing checked above");
            full[addr..addr + isz].copy_from_slice(&packed);
            return Ok(w_none());
        }
        if !index_check(index) {
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
        let full = w_memoryview_view(mv)
            .backing()
            .as_bytes_mut()
            .expect("writable backing checked above");
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
        let view = pyre_object::memoryview::w_memoryview_view(mv);
        let base = view.backing().as_bytes().as_ptr() as usize;
        let offset = view.offset() as usize;
        Ok(w_int_new((base + offset) as i64))
    }
}

/// `memoryview.format` — the struct format string of an element.
fn memoryview_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(w_str_new(pyre_object::memoryview::w_memoryview_format_str(
            mv,
        )))
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
        Ok(memoryview_wrap_dims(
            &pyre_object::memoryview::w_memoryview_native_shape(mv),
        ))
    }
}

/// `memoryview.strides` — `tuple[int]` of per-dimension byte steps.
fn memoryview_strides(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        Ok(memoryview_wrap_dims(
            &pyre_object::memoryview::w_memoryview_native_strides(mv),
        ))
    }
}

/// `memoryview.__len__` — the element count `product(shape)` (1-D: `shape[0]`).
fn memoryview_len(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        memoryview_check_released(mv)?;
        let dim = pyre_object::memoryview::w_memoryview_ndim(mv);
        if dim == 0 {
            return Err(crate::PyError::type_error("0-dim memory has no length"));
        }
        match pyre_object::memoryview::w_memoryview_native_shape(mv).first() {
            Some(&s) => Ok(w_int_new(s)),
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
        let dimshape = w_memoryview_native_shape(mv)
            .get(idim as usize)
            .copied()
            .unwrap_or(0);
        let dimstride = w_memoryview_native_strides(mv)
            .get(idim as usize)
            .copied()
            .unwrap_or(0);
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
        let fmt = pyre_object::memoryview::w_memoryview_format_str(mv);
        let full = pyre_object::memoryview::w_memoryview_view(mv)
            .backing()
            .as_bytes();
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
        if w_memoryview_restricted(mv) {
            return Err(crate::PyError::value_error(
                "cannot create a new view from a restricted memoryview",
            ));
        }
        if !pyre_object::is_str(fmt_obj) {
            return Err(crate::PyError::type_error(
                "memoryview: format argument must be a string",
            ));
        }
        // Read through the raw buffer: only the handful of native single-
        // character codes are accepted, so a lone surrogate is rejected by the
        // format check below (with the replacement character `Wtf8`'s `Display`
        // substitutes) the same way any other unsupported format is.
        let fmt = unsafe { pyre_object::w_str_get_wtf8(fmt_obj) }.to_string();
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
            let shape_v = w_memoryview_native_shape(mv);
            let has_zero = shape_v.iter().take(orig_ndim as usize).any(|&x| x == 0);
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
                    Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
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
        let orig_fmt = w_memoryview_format_str(mv);
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
        if !has_shape {
            return Ok(w_memoryview_cast_1d(mv, &fmt, new_itemsize));
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
        Ok(w_memoryview_cast_nd(
            mv,
            &fmt,
            new_itemsize,
            &dims,
            &strides_v,
        ))
    }
}

/// `memoryview.toreadonly` — a live read-only view sharing the same backing.
fn memoryview_toreadonly(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        use pyre_object::memoryview::*;
        memoryview_check_released(mv)?;
        if w_memoryview_restricted(mv) {
            return Err(crate::PyError::value_error(
                "cannot create a new view from a restricted memoryview",
            ));
        }
        // `ReadonlyWrapper(self.view)` (memoryobject.py:256).
        Ok(w_memoryview_new_derived(mv, |v| {
            pyre_object::bufferview::BufferView::Readonly {
                view: Box::new(v.clone()),
                w_obj: v.w_obj(),
            }
        }))
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

/// Drop an mmap-backed view's export directly, bypassing any Python-callable
/// release.  Returns `true` when it handled an mmap backing.
#[cfg(all(unix, not(feature = "sandbox")))]
unsafe fn release_external_backing(backing: PyObjectRef) -> bool {
    if crate::module::mmap::interp_mmap::is_mmap(backing) {
        unsafe { crate::module::mmap::interp_mmap::mmap_exports_decref(backing) };
        return true;
    }
    false
}

#[cfg(not(all(unix, not(feature = "sandbox"))))]
unsafe fn release_external_backing(_backing: PyObjectRef) -> bool {
    false
}

/// Invoke the native `bf_releasebuffer` paired with a root memoryview's
/// acquisition.  Python-visible `__release_buffer__` is a slot wrapper in
/// CPython 3.14: it calls `memoryview.release()`, which in turn reaches this
/// native slot.  Calling the Python wrapper again here would recurse.
unsafe fn release_native_backing(backing: PyObjectRef) -> bool {
    unsafe {
        if pyre_object::bytearrayobject::is_bytearray(backing) {
            pyre_object::bytearrayobject::w_bytearray_exports_decref(backing);
            return true;
        }
        if pyre_object::interp_array::is_array(backing) {
            pyre_object::interp_array::w_array_exports_decref(backing);
            return true;
        }
        if pyre_object::bytesobject::is_bytes(backing) {
            // `bytes` has no `bf_releasebuffer`; its immutable export needs no
            // accounting and has no Python-visible release wrapper.
            return true;
        }
        release_external_backing(backing)
    }
}

/// Is `descr` a native `bf_releasebuffer` wrapper rather than a Python slot?
unsafe fn memoryview_is_native_release_descr(descr: PyObjectRef) -> bool {
    unsafe {
        for base in [
            crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE),
            crate::typedef::gettypeobject(&pyre_object::memoryview::MEMORYVIEW_TYPE),
            crate::typedef::gettypeobject(&pyre_object::interp_array::ARRAY_TYPE),
        ] {
            if crate::baseobjspace::lookup_in_type(base, "__release_buffer__")
                .is_some_and(|native| std::ptr::eq(native, descr))
            {
                return true;
            }
        }
        false
    }
}

/// Python `slot_bf_releasebuffer` selected by the concrete type's MRO.
unsafe fn memoryview_python_release_descr(obj: PyObjectRef) -> Option<PyObjectRef> {
    unsafe {
        let descr = crate::baseobjspace::lookup(obj, "__release_buffer__")?;
        (!memoryview_is_native_release_descr(descr)).then_some(descr)
    }
}

/// `releasebuffer_call_python`: the C slot returns `void`, so callback
/// exceptions are reported as unraisable and never replace an active error.
unsafe fn memoryview_call_python_release_unraisable(
    exporter: PyObjectRef,
    view: PyObjectRef,
    descr: PyObjectRef,
) {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(exporter);
        pyre_object::gc_roots::pin_root(view);
        pyre_object::gc_roots::pin_root(descr);
        let r_exporter = pyre_object::gc_roots::shadow_stack_get(sp);
        let w_type = crate::typedef::r#type(r_exporter).map_or(r_exporter, |p| p.as_ptr());
        if let Err(mut error) = crate::baseobjspace::get_and_call_function(
            pyre_object::gc_roots::shadow_stack_get(sp + 2),
            r_exporter,
            w_type,
            &[pyre_object::gc_roots::shadow_stack_get(sp + 1)],
        ) {
            let r_exporter = pyre_object::gc_roots::shadow_stack_get(sp);
            error.write_unraisable(
                w_none(),
                &format!(
                    "Exception ignored in __release_buffer__ of {}:",
                    crate::baseobjspace::object_functionstr_type_name(r_exporter)
                ),
                r_exporter,
            );
        }
    }
}

/// `PyMemoryView_FromBuffer(buffer)` in `releasebuffer_call_python`, followed
/// by `_Py_MEMORYVIEW_RESTRICTED`.  The snapshot owns no second native export:
/// it exists only for the duration of the release callback.
unsafe fn w_memoryview_new_restricted_snapshot(source: PyObjectRef) -> PyObjectRef {
    unsafe {
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(source);
        let restricted = pyre_object::memoryview::w_memoryview_alloc_header(false, false);
        let r_source = pyre_object::gc_roots::shadow_stack_get(sp);
        let snapshot = pyre_object::memoryview::w_memoryview_view(r_source).clone();
        pyre_object::memoryview::w_memoryview_set_view(
            restricted,
            pyre_object::memoryview::bufferview_alloc(snapshot),
        );
        pyre_object::memoryview::w_memoryview_set_restricted(restricted, true);
        restricted
    }
}

/// Run a Python release override for a native exporter with the temporary
/// restricted view used by CPython, then make that temporary unusable even
/// when user code retained it.
unsafe fn memoryview_release_native_python_slot(exporter: PyObjectRef, source: PyObjectRef) {
    unsafe {
        let Some(descr) = memoryview_python_release_descr(exporter) else {
            return;
        };
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(exporter);
        pyre_object::gc_roots::pin_root(source);
        pyre_object::gc_roots::pin_root(descr);
        let restricted =
            w_memoryview_new_restricted_snapshot(pyre_object::gc_roots::shadow_stack_get(sp + 1));
        pyre_object::gc_roots::pin_root(restricted);
        memoryview_call_python_release_unraisable(
            pyre_object::gc_roots::shadow_stack_get(sp),
            pyre_object::gc_roots::shadow_stack_get(sp + 3),
            pyre_object::gc_roots::shadow_stack_get(sp + 2),
        );
        let r_restricted = pyre_object::gc_roots::shadow_stack_get(sp + 3);
        if !pyre_object::memoryview::w_memoryview_released(r_restricted) {
            pyre_object::memoryview::w_memoryview_set_released(r_restricted);
        }
    }
}

/// CPython `bufferwrapper_releasebuf`.
///
/// The returned memoryview's `bf_releasebuffer` first ends the temporary
/// `PyObject_GetBuffer` export.  If it exposes another owner, call the
/// original Python exporter with that exact memoryview.  If it exposes the
/// exporter itself (the `super().__buffer__` native-subclass case), release
/// the native view now; CPython's refcounted `Py_CLEAR(bw->mv)` performs this
/// immediately when the wrapper held the last reference.
unsafe fn memoryview_release_buffer_wrapper(wrapper: PyObjectRef) {
    unsafe {
        use pyre_object::memoryview::*;
        let w_mv = w_buffer_wrapper_mv(wrapper);
        let w_obj = w_buffer_wrapper_obj(wrapper);
        if w_mv.is_null() || w_obj.is_null() {
            return;
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(wrapper);
        pyre_object::gc_roots::pin_root(w_mv);
        pyre_object::gc_roots::pin_root(w_obj);

        let r_mv = pyre_object::gc_roots::shadow_stack_get(sp + 1);
        w_memoryview_exports_decref(r_mv);
        let returned_owner = w_memoryview_obj(r_mv);
        let r_obj = pyre_object::gc_roots::shadow_stack_get(sp + 2);
        if !std::ptr::eq(returned_owner, r_obj) {
            if let Some(descr) = memoryview_python_release_descr(r_obj) {
                memoryview_call_python_release_unraisable(r_obj, r_mv, descr);
            }
        } else if !w_memoryview_released(r_mv) {
            memoryview_release_native_python_slot(r_obj, r_mv);
            let backing = w_memoryview_backing(r_mv);
            let _ = release_native_backing(backing);
            w_memoryview_set_released(r_mv);
        }
        w_buffer_wrapper_clear(pyre_object::gc_roots::shadow_stack_get(sp));
    }
}

/// `memoryview.release` — drop the view; subsequent access raises ValueError.
/// Idempotent (a second `release` on an already-released view is a no-op),
/// matching `descr_release`.
pub(crate) fn memoryview_release(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe {
        if !pyre_object::memoryview::w_memoryview_released(mv) {
            let exports = pyre_object::memoryview::w_memoryview_exports(mv);
            if exports > 0 {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::BufferError,
                    format!(
                        "memoryview has {exports} exported buffer{}",
                        if exports == 1 { "" } else { "s" }
                    ),
                ));
            }
            if exports < 0 {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::SystemError,
                    "memoryview: negative export count",
                ));
            }
            // `_release_underlying`.  A slice / copy (`owns_export == false`)
            // shares the export and must not release it.
            if pyre_object::memoryview::w_memoryview_owns_export(mv) {
                let owner = pyre_object::memoryview::w_memoryview_obj(mv);
                let backing = pyre_object::memoryview::w_memoryview_backing(mv);
                // Mark the view released before invoking the exporter hook so a
                // re-entrant release is a no-op, but keep the view box until
                // the hook returns: it is handed this memoryview and reads the
                // backing back off it to identify the export it is undoing.
                pyre_object::memoryview::w_memoryview_mark_released(mv);
                let released = if pyre_object::memoryview::is_w_buffer_wrapper(owner) {
                    memoryview_release_buffer_wrapper(owner);
                    Ok(())
                } else {
                    // `slot_bf_releasebuffer`: call a Python override with a
                    // restricted snapshot, then unconditionally invoke the
                    // first native base release slot.
                    memoryview_release_native_python_slot(owner, mv);
                    if release_native_backing(backing) {
                        Ok(())
                    } else {
                        match crate::baseobjspace::lookup(backing, "__release_buffer__") {
                            Some(release_fn) => {
                                crate::call::call_function_impl_result(release_fn, &[backing, mv])
                                    .map(|_| ())
                            }
                            None => Ok(()),
                        }
                    }
                };
                pyre_object::memoryview::w_memoryview_drop_view(mv);
                released?;
            } else {
                pyre_object::memoryview::w_memoryview_set_released(mv);
            }
        }
    }
    Ok(w_none())
}

/// CPython 3.14 `wrap_releasebuffer`: validate that `view` belongs to the
/// exporter, then release the memoryview itself.  The native
/// `bf_releasebuffer` reached by [`memoryview_release`] performs the single
/// matching export decrement.
pub(crate) fn buffer_exporter_release_view(
    exporter: PyObjectRef,
    view: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        if !pyre_object::memoryview::is_w_memoryview(view) {
            return Err(crate::PyError::type_error("expected a memoryview object"));
        }
        if pyre_object::memoryview::w_memoryview_released(view) {
            return Ok(w_none());
        }
        if !std::ptr::eq(
            pyre_object::memoryview::w_memoryview_backing(view),
            exporter,
        ) {
            return Err(crate::PyError::value_error(
                "memoryview's buffer is not this object",
            ));
        }
    }
    memoryview_release(&[view])
}

fn memoryview_release_buffer(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    buffer_exporter_release_view(args[0], args[1])
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

/// Gateway receiver check shared by the rich-comparison descriptors.  The
/// type slot normally guarantees this, but direct descriptor calls must raise
/// instead of interpreting an arbitrary object as `W_MemoryView`.
fn memoryview_receiver(args: &[PyObjectRef], name: &str) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    if mv.is_null() {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' of 'memoryview' object needs an argument"
        )));
    }
    if !unsafe { pyre_object::memoryview::is_w_memoryview(mv) } {
        let received = crate::typedef::r#type(mv)
            .map(|t| unsafe { pyre_object::w_type_get_name(t.as_ptr()) })
            .unwrap_or("object");
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' requires a 'memoryview' object but received a '{received}'"
        )));
    }
    Ok(mv)
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

/// Python 3.14 `memory_richcompare`: equality first requires equivalent
/// shapes, then compares unpacked element values.  Raw bytes are deliberately
/// insufficient (`int(1)` and `float(1.0)` compare equal despite different
/// encodings, while identical bytes can decode to unequal values).
fn memoryview_compare_eq(args: &[PyObjectRef], name: &str) -> Result<Option<bool>, crate::PyError> {
    let mv = memoryview_receiver(args, name)?;
    let other = args.get(1).copied().unwrap_or(w_none());
    unsafe {
        if memoryview_released_either(mv, other) {
            return Ok(Some(mv == other));
        }

        let _roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(mv);
        pyre_object::gc_roots::pin_root(other);
        let other = pyre_object::gc_roots::shadow_stack_get(base + 1);
        let (rhs, temporary) = if pyre_object::memoryview::is_w_memoryview(other) {
            (other, false)
        } else {
            match w_memoryview_new(other) {
                Ok(view) => {
                    pyre_object::gc_roots::pin_root(view);
                    (pyre_object::gc_roots::shadow_stack_get(base + 2), true)
                }
                // `PyObject_GetBuffer(..., PyBUF_FULL_RO)` failures are
                // cleared by CPython's `memory_richcompare`.
                Err(_) => return Ok(None),
            }
        };
        let lhs = pyre_object::gc_roots::shadow_stack_get(base);
        let comparison = (|| -> Result<bool, crate::PyError> {
            if pyre_object::memoryview::w_memoryview_ndim(lhs)
                != pyre_object::memoryview::w_memoryview_ndim(rhs)
                || pyre_object::memoryview::w_memoryview_native_shape(lhs)
                    != pyre_object::memoryview::w_memoryview_native_shape(rhs)
            {
                return Ok(false);
            }
            let lhs_data = memoryview_gather_bytes(lhs);
            let rhs_data = memoryview_gather_bytes(rhs);
            let lhs_size = pyre_object::memoryview::w_memoryview_itemsize(lhs) as usize;
            let rhs_size = pyre_object::memoryview::w_memoryview_itemsize(rhs) as usize;
            if lhs_size == 0 || rhs_size == 0 {
                return Ok(lhs_data.is_empty() && rhs_data.is_empty());
            }
            let count = lhs_data.len() / lhs_size;
            if count != rhs_data.len() / rhs_size {
                return Ok(false);
            }
            let lhs_fmt = pyre_object::memoryview::w_memoryview_format_str(lhs);
            let rhs_fmt = pyre_object::memoryview::w_memoryview_format_str(rhs);
            for index in 0..count {
                let _values = pyre_object::gc_roots::push_roots();
                let values = pyre_object::gc_roots::shadow_stack_len();
                pyre_object::gc_roots::pin_root(memoryview_unpack_element(
                    lhs_fmt,
                    &lhs_data,
                    index * lhs_size,
                    lhs_size,
                ));
                pyre_object::gc_roots::pin_root(memoryview_unpack_element(
                    rhs_fmt,
                    &rhs_data,
                    index * rhs_size,
                    rhs_size,
                ));
                if !crate::baseobjspace::eq_w(
                    pyre_object::gc_roots::shadow_stack_get(values),
                    pyre_object::gc_roots::shadow_stack_get(values + 1),
                )? {
                    return Ok(false);
                }
            }
            Ok(true)
        })();
        if temporary {
            let release = memoryview_release(&[rhs]);
            if comparison.is_ok() {
                release?;
            }
        }
        comparison.map(Some)
    }
}

fn memoryview_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(match memoryview_compare_eq(args, "__eq__")? {
        Some(equal) => w_bool_from(equal),
        None => pyre_object::w_not_implemented(),
    })
}

/// `memoryview.__ne__` — the negation of the same element-wise comparison.
fn memoryview_ne(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(match memoryview_compare_eq(args, "__ne__")? {
        Some(equal) => w_bool_from(!equal),
        None => pyre_object::w_not_implemented(),
    })
}

/// Python 3.14 exposes all ordering slots but `memory_richcompare` returns
/// `NotImplemented` for them after validating the descriptor receiver.
fn memoryview_order(args: &[PyObjectRef], name: &str) -> Result<PyObjectRef, crate::PyError> {
    memoryview_receiver(args, name)?;
    Ok(pyre_object::w_not_implemented())
}

fn memoryview_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    memoryview_order(args, "__lt__")
}
fn memoryview_le(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    memoryview_order(args, "__le__")
}
fn memoryview_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    memoryview_order(args, "__gt__")
}
fn memoryview_ge(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    memoryview_order(args, "__ge__")
}

/// Python 3.14 `memoryview._from_flags(object, flags)`.  A memoryview input
/// is cloned with its managed buffer and ignores flags.  For other exporters,
/// pyre currently provides the full read-only view; the observable writable
/// request bit is enforced here exactly like `PyBUF_WRITABLE`.
fn memoryview_from_flags(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let object = args.get(1).copied().unwrap_or(w_none());
    let flags = crate::baseobjspace::c_int_w(args.get(2).copied().unwrap_or(w_none()))?;
    w_memoryview_new_with_flags(object, flags)
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
    // CPython 3.14 `memoryview_hex_impl`: a contiguous view increments its
    // export count while separator coercion may run arbitrary Python code.
    // All pyre memoryview formats accepted here are gathered from the same
    // live view, so retain the identical release guard across the formatter.
    unsafe { pyre_object::memoryview::w_memoryview_exports_incref(mv) };
    let result = crate::typedef::bytes_method_hex(&fwd);
    unsafe { pyre_object::memoryview::w_memoryview_exports_decref(mv) };
    result
}

/// `descr_hash` (memoryobject.py:476) — a writable view is unhashable; a
/// read-only view hashes its raw bytes (so `hash(mv) == hash(bytes)`),
/// cached in `self._hash` with the `-1` sentinel (`_hash_bytes` never returns
/// `-1`) — the release / readonly checks run only on the first call, and a
/// view hashed before `release()` keeps hashing afterwards.
unsafe fn memoryview_hash_value(mv: PyObjectRef) -> Result<i64, crate::PyError> {
    unsafe {
        let mut hash = pyre_object::memoryview::w_memoryview_hash(mv);
        if hash == -1 {
            memoryview_check_released(mv)?;
            if !pyre_object::memoryview::w_memoryview_readonly(mv) {
                return Err(crate::PyError::value_error(
                    "cannot hash writable memoryview object",
                ));
            }
            // CPython 3.14 `memory_hash`: hash the exporter while holding a
            // temporary export.  The digest is deliberately discarded; the
            // call validates hashability and, crucially, prevents a
            // re-entrant `mv.release()` from freeing the bytes being hashed.
            let backing = pyre_object::memoryview::w_memoryview_obj(mv);
            pyre_object::memoryview::w_memoryview_exports_incref(mv);
            let backing_hash = crate::baseobjspace::hash_w_strict(backing);
            pyre_object::memoryview::w_memoryview_exports_decref(mv);
            backing_hash?;
            // `compute_hash(self.view.as_str())` — the same content digest the
            // bytes path uses, so `hash(memoryview(b)) == hash(b)`.
            hash = _hash_bytes(&memoryview_gather_bytes(mv));
            pyre_object::memoryview::w_memoryview_set_hash(mv, hash);
        }
        Ok(hash)
    }
}

/// `memoryview.__hash__` — see [`memoryview_hash_value`].
fn memoryview_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mv = args.first().copied().unwrap_or(w_none());
    unsafe { Ok(w_int_new(memoryview_hash_value(mv)?)) }
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
        let shape = w_memoryview_native_shape(mv);
        let strides = w_memoryview_native_strides(mv);
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

/// Python 3.14 `memoryview.count(value)` — iteration is intentional: element
/// decoding and equality therefore follow the view's live format exactly.
fn memoryview_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    unsafe { memoryview_check_released(args[0])? };
    let ndim = unsafe { pyre_object::memoryview::w_memoryview_ndim(args[0]) };
    if ndim > 1 {
        // CPython 3.14's memoryobject.c routes multidimensional count
        // through the unsupported sub-view path.  Keep the same explicit
        // boundary rather than silently counting only the first dimension.
        return Err(crate::PyError::not_implemented(
            "multi-dimensional sub-views are not implemented",
        ));
    }
    let iterator = crate::baseobjspace::iter(args[0])?;
    let mut count = 0i64;
    loop {
        match crate::baseobjspace::next(iterator) {
            Ok(item) => {
                if std::ptr::eq(item, args[1])
                    || crate::baseobjspace::is_true(crate::baseobjspace::compare(
                        item,
                        args[1],
                        crate::baseobjspace::CompareOp::Eq,
                    )?)?
                {
                    count += 1;
                }
            }
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    Ok(w_int_new(count))
}

/// Python 3.14 `memoryview.index(value, start=0, stop=sys.maxsize)`.
fn memoryview_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(crate::PyError::type_error(format!(
            "index expected at most 3 arguments, got {}",
            args.len().saturating_sub(1)
        )));
    }
    let mv = args[0];
    unsafe { memoryview_check_released(mv)? };
    let ndim = unsafe { pyre_object::memoryview::w_memoryview_ndim(mv) };
    if ndim == 0 {
        return Err(crate::PyError::type_error("invalid lookup on 0-dim memory"));
    }
    if ndim != 1 {
        return Err(crate::PyError::not_implemented(
            "multi-dimensional lookup is not implemented",
        ));
    }
    // CPython 3.14 `memoryview_index_impl`: lookup bounds are measured in
    // elements (`view->shape[0]`), not bytes (`view->len`).
    let n = unsafe {
        pyre_object::memoryview::w_memoryview_native_shape(mv)
            .first()
            .copied()
            .unwrap_or(0)
    };
    let mut start = if args.len() >= 3 {
        crate::baseobjspace::getindex_w(args[2])?
    } else {
        0
    };
    let mut stop = if args.len() >= 4 {
        crate::baseobjspace::getindex_w(args[3])?
    } else {
        i64::MAX
    };
    if start < 0 {
        start = start.saturating_add(n).max(0);
    }
    if stop < 0 {
        stop = stop.saturating_add(n).max(0);
    }
    stop = stop.min(n);
    start = start.min(stop);
    for index in start..stop {
        let item = memoryview_getitem(&[mv, w_int_new(index)])?;
        if std::ptr::eq(item, args[1])
            || crate::baseobjspace::is_true(crate::baseobjspace::compare(
                item,
                args[1],
                crate::baseobjspace::CompareOp::Eq,
            )?)?
        {
            return Ok(w_int_new(index));
        }
    }
    Err(crate::PyError::value_error(
        "memoryview.index(x): x not found",
    ))
}

/// Install the `memoryview` type-dict methods and properties.  Wired into
/// `MEMORYVIEW_TYPE` from `typedef::init_typeobjects`; each method reads the
/// native `W_MemoryView` fields rather than per-instance attribute slots.
pub(crate) fn init_memoryview_type(ns: PyObjectRef) {
    type MvFn = fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>;
    // `__new__` is a `BuiltinFunction`-typed staticmethod descriptor like
    // every other native type's `tp_new` (typedef::make_new_descr), so it
    // does not bind the class and pickle's `isinstance(new, type(int.__new__))`
    // check matches.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__doc__",
            w_str_new("Create a new memoryview object which references the given object."),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            crate::typedef::make_new_descr(memoryview_descr_new),
        )
    };
    for (name, f, arity) in [
        ("__getitem__", memoryview_getitem as MvFn, 2u16),
        ("__setitem__", memoryview_setitem, 3),
        ("__len__", memoryview_len, 1),
        ("__iter__", memoryview_iter, 1),
        ("__repr__", memoryview_repr, 1),
        ("__eq__", memoryview_eq, 2),
        ("__ne__", memoryview_ne, 2),
        ("__lt__", memoryview_lt, 2),
        ("__le__", memoryview_le, 2),
        ("__gt__", memoryview_gt, 2),
        ("__ge__", memoryview_ge, 2),
        ("count", memoryview_count, 2),
        ("tobytes", memoryview_tobytes, 1),
        ("tolist", memoryview_tolist, 1),
        ("toreadonly", memoryview_toreadonly, 1),
        ("release", memoryview_release, 1),
        ("__enter__", memoryview_enter, 1),
        ("__hash__", memoryview_hash, 1),
        ("_pypy_raw_address", memoryview_raw_address, 1),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function_with_arity(name, f, arity),
            )
        };
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__buffer__",
            make_builtin_function_with_arity(
                "__buffer__",
                |args| {
                    let flags = crate::baseobjspace::c_int_w(args[1])?;
                    w_memoryview_new_native_with_flags(args[0], flags)
                },
                2,
            ),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__class_getitem__",
            pyre_object::function::w_classmethod_new(make_builtin_function(
                "__class_getitem__",
                crate::_pypy_generic_alias::generic_alias_class_getitem,
            )),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "_from_flags",
            pyre_object::function::w_classmethod_new(make_builtin_function_with_arity(
                "_from_flags",
                memoryview_from_flags,
                3,
            )),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "index",
            make_builtin_function("index", memoryview_index),
        );
    }
    // `__exit__(self, *exc)`, `__release_buffer__(self, view)`,
    // `__delitem__(self, *args)`, `hex(self, sep=, bytes_per_sep=)`, and
    // `cast(format[, shape])` take variable / optional trailing arguments,
    // so they register as plain (non-arity-pinned) builtins.
    for (name, f) in [
        ("__exit__", memoryview_exit as MvFn),
        ("__release_buffer__", memoryview_release_buffer),
        ("__delitem__", memoryview_delitem),
        ("hex", memoryview_hex),
        ("cast", memoryview_cast),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                make_builtin_function(name, f),
            )
        };
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
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                attr,
                pyre_object::w_property_new(
                    make_builtin_function_with_arity(attr, getter, 1),
                    pyre_object::PY_NULL,
                    pyre_object::PY_NULL,
                ),
            )
        };
    }
}

pub fn install_default_builtins(ns: PyObjectRef) {
    crate::module_ns_get_or_insert_with(ns, "print", || {
        make_module_builtin_function("print", builtin_print)
    });
    crate::module_ns_get_or_insert_with(ns, "range", || {
        crate::typedef::gettypeobject(&pyre_object::functional::RANGE_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "len", || {
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
    crate::module_ns_get_or_insert_with(ns, "abs", || {
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
    crate::module_ns_get_or_insert_with(ns, "min", || {
        make_module_builtin_function_with_doc(
            "min",
            builtin_min,
            "min(iterable, *[, default=obj, key=func]) -> value\n\
             min(arg1, arg2, *args, *[, key=func]) -> value\n\
             \n\
             With a single iterable argument, return its smallest item. The\n\
             default keyword-only argument specifies an object to return if\n\
             the provided iterable is empty.\n\
             With two or more positional arguments, return the smallest argument.",
        )
    });
    crate::module_ns_get_or_insert_with(ns, "max", || {
        make_module_builtin_function_with_doc(
            "max",
            builtin_max,
            "max(iterable, *[, default=obj, key=func]) -> value\n\
             max(arg1, arg2, *args, *[, key=func]) -> value\n\
             \n\
             With a single iterable argument, return its biggest item. The\n\
             default keyword-only argument specifies an object to return if\n\
             the provided iterable is empty.\n\
             With two or more positional arguments, return the largest argument.",
        )
    });
    crate::module_ns_get_or_insert_with(ns, "type", || crate::typedef::w_type());
    crate::module_ns_get_or_insert_with(ns, "isinstance", || {
        make_module_builtin_function_with_arity("isinstance", __pyre_wrap_builtin_isinstance, 2)
    });
    crate::module_ns_get_or_insert_with(ns, "str", || crate::typedef::gettypeobject(&STR_TYPE));
    crate::module_ns_get_or_insert_with(ns, "repr", || {
        make_module_builtin_function_with_arity("repr", builtin_repr, 1)
    });
    crate::module_ns_get_or_insert_with(ns, "ascii", || {
        make_module_builtin_function_with_arity("ascii", builtin_ascii, 1)
    });
    crate::module_ns_get_or_insert_with(ns, "int", || crate::typedef::gettypeobject(&INT_TYPE));
    crate::module_ns_get_or_insert_with(ns, "float", || crate::typedef::gettypeobject(&FLOAT_TYPE));
    crate::module_ns_get_or_insert_with(ns, "bool", || crate::typedef::gettypeobject(&BOOL_TYPE));
    crate::module_ns_get_or_insert_with(ns, "True", || w_bool_from(true));
    crate::module_ns_get_or_insert_with(ns, "False", || w_bool_from(false));
    crate::module_ns_get_or_insert_with(ns, "None", || w_none());
    crate::module_ns_get_or_insert_with(ns, "NotImplemented", || w_not_implemented());
    crate::module_ns_get_or_insert_with(ns, "hasattr", || {
        make_module_builtin_function_with_arity("hasattr", builtin_hasattr, 2)
    });
    crate::module_ns_get_or_insert_with(ns, "getattr", || {
        make_module_builtin_function("getattr", builtin_getattr)
    });
    crate::module_ns_get_or_insert_with(ns, "setattr", || {
        make_module_builtin_function_with_arity("setattr", builtin_setattr, 3)
    });
    crate::module_ns_get_or_insert_with(ns, "delattr", || {
        make_module_builtin_function_with_arity("delattr", builtin_delattr, 2)
    });
    crate::module_ns_get_or_insert_with(ns, "tuple", || crate::typedef::gettypeobject(&TUPLE_TYPE));
    crate::module_ns_get_or_insert_with(ns, "list", || crate::typedef::gettypeobject(&LIST_TYPE));
    crate::module_ns_get_or_insert_with(ns, "dict", || crate::typedef::gettypeobject(&DICT_TYPE));
    crate::module_ns_get_or_insert_with(ns, "object", || {
        // `object` is a W_TypeObject, not a builtin function.
        // PyPy: baseobjspace.py w_object = W_TypeObject("object", ...)
        crate::typedef::w_object()
    });
    crate::module_ns_get_or_insert_with(ns, "super", || {
        crate::typedef::gettypeobject(&pyre_object::descriptor::SUPER_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "id", || {
        make_module_builtin_function_with_arity("id", builtin_id, 1)
    });
    crate::module_ns_get_or_insert_with(ns, "hash", || {
        make_module_builtin_function_with_arity("hash", builtin_hash, 1)
    });
    crate::module_ns_get_or_insert_with(ns, "ord", || {
        make_module_builtin_function_with_arity("ord", builtin_ord, 1)
    });
    crate::module_ns_get_or_insert_with(ns, "chr", || {
        make_module_builtin_function_with_arity("chr", builtin_chr, 1)
    });
    crate::module_ns_get_or_insert_with(ns, "map", || {
        crate::typedef::gettypeobject(&pyre_object::functional::MAP_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "zip", || {
        crate::typedef::gettypeobject(&pyre_object::functional::ZIP_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "enumerate", || {
        crate::typedef::gettypeobject(&pyre_object::functional::ENUMERATE_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "reversed", || {
        crate::typedef::gettypeobject(&pyre_object::functional::REVERSED_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "sorted", || {
        make_module_builtin_function("sorted", builtin_sorted)
    });
    crate::module_ns_get_or_insert_with(ns, "iter", || {
        make_module_builtin_function("iter", builtin_iter)
    });
    crate::module_ns_get_or_insert_with(ns, "next", || {
        make_module_builtin_function("next", builtin_next)
    });
    // aiter/anext resolve their app-level implementations lazily; the
    // builtins dict is filled before an execution context exists.  Arity is
    // enforced by the app-level `def aiter(obj)` / `def anext(iterator,
    // default=...)` signatures, so no interp-level arity is imposed here.
    crate::module_ns_get_or_insert_with(ns, "aiter", || {
        make_module_builtin_function("aiter", crate::async_operation::builtin_aiter)
    });
    crate::module_ns_get_or_insert_with(ns, "anext", || {
        make_module_builtin_function("anext", crate::async_operation::builtin_anext)
    });
    crate::module_ns_get_or_insert_with(ns, "breakpoint", || {
        make_module_builtin_function("breakpoint", builtin_breakpoint)
    });
    crate::module_ns_get_or_insert_with(ns, "callable", || {
        make_module_builtin_function_with_arity("callable", builtin_callable, 1)
    });
    crate::module_ns_get_or_insert_with(ns, "vars", || {
        make_module_builtin_function("vars", builtin_vars)
    });
    crate::module_ns_get_or_insert_with(ns, "dir", || {
        make_module_builtin_function("dir", builtin_dir)
    });
    crate::module_ns_get_or_insert_with(ns, "__build_class__", || {
        make_module_builtin_function("__build_class__", |args| {
            crate::call::real_build_class(args)
        })
    });
    // bytearrayobject.py W_BytearrayObject — register the real type
    // (callable as a constructor and usable in isinstance(x, bytearray)).
    crate::module_ns_get_or_insert_with(ns, "bytearray", || {
        crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE)
    });
    // bytesobject.py W_BytesObject — immutable bytes type.
    crate::module_ns_get_or_insert_with(ns, "bytes", || {
        crate::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "slice", || {
        // The slice type object, for isinstance(x, slice) checks.
        crate::typedef::gettypefor(&pyre_object::sliceobject::SLICE_TYPE)
            .map_or(pyre_object::PY_NULL, |p| p.as_ptr())
    });
    crate::module_ns_get_or_insert_with(ns, "frozenset", || {
        crate::typedef::gettypeobject(&pyre_object::setobject::FROZENSET_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "set", || {
        crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "property", || {
        crate::typedef::gettypeobject(&pyre_object::descriptor::PROPERTY_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "staticmethod", || {
        crate::typedef::gettypeobject(&pyre_object::function::STATICMETHOD_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "classmethod", || {
        crate::typedef::gettypeobject(&pyre_object::function::CLASSMETHOD_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "Ellipsis", || pyre_object::special::w_ellipsis());
    crate::module_ns_get_or_insert_with(ns, "__debug__", || w_bool_from(true));
    crate::module_ns_get_or_insert_with(ns, "memoryview", || {
        crate::typedef::gettypeobject(&pyre_object::memoryview::MEMORYVIEW_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "globals", || {
        make_module_builtin_function_with_arity("globals", builtin_globals, 0)
    });
    crate::module_ns_get_or_insert_with(ns, "locals", || {
        make_module_builtin_function_with_arity("locals", builtin_locals, 0)
    });
    crate::module_ns_get_or_insert_with(ns, "exec", || {
        make_module_builtin_function("exec", builtin_exec)
    });
    crate::module_ns_get_or_insert_with(ns, "eval", || {
        make_module_builtin_function("eval", builtin_eval)
    });
    crate::module_ns_get_or_insert_with(ns, "compile", || {
        make_module_builtin_function("compile", builtin_compile)
    });
    crate::module_ns_get_or_insert_with(ns, "complex", || {
        crate::typedef::gettypeobject(&pyre_object::COMPLEX_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "filter", || {
        crate::typedef::gettypeobject(&pyre_object::functional::FILTER_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "input", || {
        make_module_builtin_function("input", builtin_input)
    });
    crate::module_ns_get_or_insert_with(ns, "open", || {
        make_module_builtin_function("open", builtin_open)
    });
    // Exception hierarchy — exceptions are real types so they can be
    // subclassed (`class FrozenInstanceError(AttributeError): pass`).
    // Built in dependency order: each subclass refers to its already-built
    // parent. PyPy: each typedef.py W_<Exception>.typedef registers a real
    // W_TypeObject in space.builtin.
    let base_exc = make_exc_type_with_init(
        "BaseException",
        Some("Common base class for all exceptions"),
        exc_base_exception_new,
        Some(exc_base_exception_init),
        crate::typedef::w_object(),
    );
    crate::module_ns_store(ns, "BaseException", base_exc);

    let exception = make_exc_type_with_doc(
        "Exception",
        "Common base class for all non-exit exceptions.",
        exc_exception_new,
        base_exc,
    );
    crate::module_ns_store(ns, "Exception", exception);

    let arithmetic = make_exc_type_with_doc(
        "ArithmeticError",
        "Base class for arithmetic errors.",
        exc_arithmetic_error_new,
        exception,
    );
    crate::module_ns_store(ns, "ArithmeticError", arithmetic);
    crate::module_ns_store(
        ns,
        "ZeroDivisionError",
        make_exc_type_with_doc(
            "ZeroDivisionError",
            "Second argument to a division or modulo operation was zero.",
            exc_zero_division_new,
            arithmetic,
        ),
    );
    crate::module_ns_store(
        ns,
        "OverflowError",
        make_exc_type_with_doc(
            "OverflowError",
            "Result too large to be represented.",
            exc_overflow_error_new,
            arithmetic,
        ),
    );
    crate::module_ns_store(
        ns,
        "FloatingPointError",
        make_exc_type_with_doc(
            "FloatingPointError",
            "Floating-point operation failed.",
            exc_arithmetic_error_new,
            arithmetic,
        ),
    );

    let lookup_error = make_exc_type_with_doc(
        "LookupError",
        "Base class for lookup errors.",
        exc_lookup_error_new,
        exception,
    );
    crate::module_ns_store(ns, "LookupError", lookup_error);
    crate::module_ns_store(
        ns,
        "IndexError",
        make_exc_type_with_doc(
            "IndexError",
            "Sequence index out of range.",
            exc_index_error_new,
            lookup_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "KeyError",
        make_exc_type_with_doc(
            "KeyError",
            "Mapping key not found.",
            exc_key_error_new,
            lookup_error,
        ),
    );

    crate::module_ns_store(
        ns,
        "AttributeError",
        make_exc_type_with_init(
            "AttributeError",
            Some("Attribute not found."),
            exc_attribute_error_new,
            Some(exc_attribute_error_init),
            exception,
        ),
    );
    crate::module_ns_store(
        ns,
        "TypeError",
        make_exc_type_with_doc(
            "TypeError",
            "Inappropriate argument type.",
            exc_type_error_new,
            exception,
        ),
    );
    let value_error = make_exc_type_with_doc(
        "ValueError",
        "Inappropriate argument value (of correct type).",
        exc_value_error_new,
        exception,
    );
    crate::module_ns_store(ns, "ValueError", value_error);
    let name_error = make_exc_type_with_init(
        "NameError",
        Some("Name not found globally."),
        exc_name_error_new,
        Some(exc_name_error_init),
        exception,
    );
    crate::module_ns_store(ns, "NameError", name_error);
    // `exceptions.c` — `UnboundLocalError(NameError)`.
    crate::module_ns_store(
        ns,
        "UnboundLocalError",
        make_exc_type_with_doc(
            "UnboundLocalError",
            "Local name referenced but not bound to a value.",
            exc_name_error_new,
            name_error,
        ),
    );

    let runtime_error = make_exc_type_with_doc(
        "RuntimeError",
        "Unspecified run-time error.",
        exc_runtime_error_new,
        exception,
    );
    crate::module_ns_store(ns, "RuntimeError", runtime_error);
    crate::module_ns_store(
        ns,
        "NotImplementedError",
        make_exc_type_with_doc(
            "NotImplementedError",
            "Method or function hasn't been implemented yet.",
            exc_not_implemented_error_new,
            runtime_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "RecursionError",
        make_exc_type_with_doc(
            "RecursionError",
            "Recursion limit exceeded.",
            exc_recursion_error_new,
            runtime_error,
        ),
    );

    crate::module_ns_store(
        ns,
        "StopIteration",
        make_exc_type_with_init(
            "StopIteration",
            Some("Signal the end from iterator.__next__()."),
            exc_stop_iteration_new,
            Some(exc_stop_iteration_init),
            exception,
        ),
    );
    crate::module_ns_store(
        ns,
        "StopAsyncIteration",
        make_exc_type_with_doc(
            "StopAsyncIteration",
            "Signal the end from iterator.__anext__().",
            exc_stop_async_iteration_new,
            exception,
        ),
    );
    crate::module_ns_store(
        ns,
        "GeneratorExit",
        make_exc_type_with_doc(
            "GeneratorExit",
            "Request that a generator exit.",
            exc_generator_exit_new,
            base_exc,
        ),
    );
    crate::module_ns_store(
        ns,
        "SystemExit",
        make_exc_type_with_init(
            "SystemExit",
            Some("Request to exit from the interpreter."),
            exc_system_exit_new,
            Some(exc_system_exit_init),
            base_exc,
        ),
    );
    crate::module_ns_store(
        ns,
        "KeyboardInterrupt",
        make_exc_type_with_doc(
            "KeyboardInterrupt",
            "Program interrupted by user.",
            exc_base_exception_new,
            base_exc,
        ),
    );

    let import_error = make_exc_type_with_init(
        "ImportError",
        Some("Import can't find module, or can't find name in module."),
        exc_import_error_new,
        Some(exc_import_error_init),
        exception,
    );
    crate::module_ns_store(ns, "ImportError", import_error);
    crate::module_ns_store(
        ns,
        "ModuleNotFoundError",
        make_exc_type_with_doc(
            "ModuleNotFoundError",
            "Module not found.",
            exc_import_error_new,
            import_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "AssertionError",
        make_exc_type_with_doc(
            "AssertionError",
            "Assertion failed.",
            exc_assertion_error_new,
            exception,
        ),
    );

    let os_error = make_exc_type_with_init(
        "OSError",
        Some("Base class for I/O related errors."),
        exc_os_error_new,
        Some(exc_os_error_init),
        exception,
    );
    crate::module_ns_store(ns, "OSError", os_error);
    crate::module_ns_store(ns, "IOError", os_error);
    // `exceptions.c` — `EnvironmentError` is a deprecated alias of `OSError`.
    crate::module_ns_store(ns, "EnvironmentError", os_error);
    crate::module_ns_store(
        ns,
        "FileNotFoundError",
        make_exc_type_with_doc(
            "FileNotFoundError",
            "File not found.",
            exc_file_not_found_error_new,
            os_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "FileExistsError",
        make_exc_type_with_doc(
            "FileExistsError",
            "File already exists.",
            exc_os_error_new,
            os_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "PermissionError",
        make_exc_type_with_doc(
            "PermissionError",
            "Not enough permissions.",
            exc_os_error_new,
            os_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "NotADirectoryError",
        make_exc_type_with_doc(
            "NotADirectoryError",
            "Operation only works on directories.",
            exc_os_error_new,
            os_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "IsADirectoryError",
        make_exc_type_with_doc(
            "IsADirectoryError",
            "Operation doesn't work on directories.",
            exc_os_error_new,
            os_error,
        ),
    );

    let warning = make_exc_type_with_doc(
        "Warning",
        "Base class for warning categories.",
        exc_exception_new,
        exception,
    );
    crate::module_ns_store(ns, "Warning", warning);
    for (warn_name, doc) in [
        (
            "UserWarning",
            "Base class for warnings generated by user code.",
        ),
        (
            "DeprecationWarning",
            "Base class for warnings about deprecated features.",
        ),
        (
            "PendingDeprecationWarning",
            "Base class for warnings about features which will be deprecated\nin the future.",
        ),
        (
            "RuntimeWarning",
            "Base class for warnings about dubious runtime behavior.",
        ),
        (
            "FutureWarning",
            "Base class for warnings about constructs that will change semantically\nin the future.",
        ),
        (
            "ImportWarning",
            "Base class for warnings about probable mistakes in module imports",
        ),
        (
            "UnicodeWarning",
            "Base class for warnings about Unicode related problems, mostly\nrelated to conversion problems.",
        ),
        (
            "BytesWarning",
            "Base class for warnings about bytes and buffer related problems, mostly\nrelated to conversion from str or comparing to str.",
        ),
        (
            "ResourceWarning",
            "Base class for warnings about resource usage.",
        ),
        (
            "SyntaxWarning",
            "Base class for warnings about dubious syntax.",
        ),
        (
            "EncodingWarning",
            "Base class for warnings about encodings.",
        ),
    ] {
        crate::module_ns_store(
            ns,
            warn_name,
            make_exc_type_with_doc(warn_name, doc, exc_exception_new, warning),
        );
    }

    let unicode_error = make_exc_type_with_doc(
        "UnicodeError",
        "Unicode related error.",
        exc_unicode_error_new,
        value_error,
    );
    crate::module_ns_store(ns, "UnicodeError", unicode_error);
    crate::module_ns_store(
        ns,
        "UnicodeDecodeError",
        make_exc_type_with_init(
            "UnicodeDecodeError",
            Some("Unicode decoding error."),
            exc_unicode_decode_error_new,
            Some(exc_unicode_decode_error_init),
            unicode_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "UnicodeEncodeError",
        make_exc_type_with_init(
            "UnicodeEncodeError",
            Some("Unicode encoding error."),
            exc_unicode_encode_error_new,
            Some(exc_unicode_encode_error_init),
            unicode_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "UnicodeTranslateError",
        make_exc_type_with_init(
            "UnicodeTranslateError",
            Some("Unicode translation error."),
            exc_unicode_translate_error_new,
            Some(exc_unicode_translate_error_init),
            unicode_error,
        ),
    );

    crate::module_ns_store(
        ns,
        "BufferError",
        make_exc_type_with_doc("BufferError", "Buffer error.", exc_exception_new, exception),
    );
    crate::module_ns_store(
        ns,
        "MemoryError",
        make_exc_type_with_doc(
            "MemoryError",
            "Out of memory.",
            exc_memory_error_new,
            exception,
        ),
    );
    crate::module_ns_store(
        ns,
        "ReferenceError",
        make_exc_type_with_doc(
            "ReferenceError",
            "Weak ref proxy used after referent went away.",
            exc_reference_error_new,
            exception,
        ),
    );
    crate::module_ns_store(
        ns,
        "SystemError",
        make_exc_type_with_doc(
            "SystemError",
            "Internal error in the Python interpreter.\n\nPlease report this to the Python maintainer, along with the traceback,\nthe Python version, and the hardware/OS platform and version.",
            exc_system_error_new,
            exception,
        ),
    );
    crate::module_ns_store(
        ns,
        "EOFError",
        make_exc_type_with_doc(
            "EOFError",
            "Read beyond end of file.",
            exc_eof_error_new,
            exception,
        ),
    );
    let syntax_error = make_exc_type_with_init(
        "SyntaxError",
        Some("Invalid syntax."),
        exc_syntax_error_new,
        Some(exc_syntax_error_init),
        exception,
    );
    crate::module_ns_store(ns, "SyntaxError", syntax_error);
    // Python 3.14 `exceptions.c` — the private exception raised by
    // `compile(..., flags=PyCF_ALLOW_INCOMPLETE_INPUT)` for an unfinished
    // interactive input.  `codeop._maybe_compile` intentionally resolves the
    // underscore-prefixed name through builtins rather than importing it.
    crate::module_ns_store(
        ns,
        "_IncompleteInputError",
        make_exc_type_with_doc(
            "_IncompleteInputError",
            "incomplete input.",
            exc_syntax_error_new,
            syntax_error,
        ),
    );
    let indentation_error = make_exc_type_with_doc(
        "IndentationError",
        "Improper indentation.",
        exc_syntax_error_new,
        syntax_error,
    );
    crate::module_ns_store(ns, "IndentationError", indentation_error);
    crate::module_ns_store(
        ns,
        "TabError",
        make_exc_type_with_doc(
            "TabError",
            "Improper mixture of spaces and tabs.",
            exc_syntax_error_new,
            indentation_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "BlockingIOError",
        make_exc_type_with_doc(
            "BlockingIOError",
            "I/O operation would block.",
            exc_os_error_new,
            os_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "ChildProcessError",
        make_exc_type_with_doc(
            "ChildProcessError",
            "Child process error.",
            exc_os_error_new,
            os_error,
        ),
    );
    let connection_error = make_exc_type_with_doc(
        "ConnectionError",
        "Connection error.",
        exc_os_error_new,
        os_error,
    );
    crate::module_ns_store(ns, "ConnectionError", connection_error);
    crate::module_ns_store(
        ns,
        "BrokenPipeError",
        make_exc_type_with_doc(
            "BrokenPipeError",
            "Broken pipe.",
            exc_os_error_new,
            connection_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "ConnectionAbortedError",
        make_exc_type_with_doc(
            "ConnectionAbortedError",
            "Connection aborted.",
            exc_os_error_new,
            connection_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "ConnectionRefusedError",
        make_exc_type_with_doc(
            "ConnectionRefusedError",
            "Connection refused.",
            exc_os_error_new,
            connection_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "ConnectionResetError",
        make_exc_type_with_doc(
            "ConnectionResetError",
            "Connection reset.",
            exc_os_error_new,
            connection_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "InterruptedError",
        make_exc_type_with_doc(
            "InterruptedError",
            "Interrupted by signal.",
            exc_os_error_new,
            os_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "ProcessLookupError",
        make_exc_type_with_doc(
            "ProcessLookupError",
            "Process not found.",
            exc_os_error_new,
            os_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "TimeoutError",
        make_exc_type_with_doc(
            "TimeoutError",
            "Timeout expired.",
            exc_os_error_new,
            os_error,
        ),
    );
    let base_exception_group = make_exception_group_type(
        "BaseExceptionGroup",
        Some("A combination of multiple unrelated exceptions."),
        &[base_exc],
    );
    crate::module_ns_store(ns, "BaseExceptionGroup", base_exception_group);
    let exception_group =
        make_exception_group_type("ExceptionGroup", None, &[base_exception_group, exception]);
    crate::module_ns_store(ns, "ExceptionGroup", exception_group);
    crate::module_ns_store(
        ns,
        "PythonFinalizationError",
        make_exc_type_with_doc(
            "PythonFinalizationError",
            "Operation blocked during Python finalization.",
            exc_runtime_error_new,
            runtime_error,
        ),
    );
    crate::module_ns_get_or_insert_with(ns, "any", || {
        make_module_builtin_function_with_arity("any", builtin_any, 1)
    });
    crate::module_ns_get_or_insert_with(ns, "all", || {
        make_module_builtin_function_with_arity("all", builtin_all, 1)
    });
    crate::module_ns_get_or_insert_with(ns, "sum", || {
        make_module_builtin_function("sum", builtin_sum)
    });
    crate::module_ns_get_or_insert_with(ns, "round", || {
        make_module_builtin_function("round", builtin_round)
    });
    crate::module_ns_get_or_insert_with(ns, "divmod", || {
        make_module_builtin_function("divmod", builtin_divmod)
    });
    crate::module_ns_get_or_insert_with(ns, "pow", || {
        make_module_builtin_function("pow", builtin_pow)
    });
    crate::module_ns_get_or_insert_with(ns, "hex", || {
        make_module_builtin_function("hex", builtin_hex)
    });
    crate::module_ns_get_or_insert_with(ns, "oct", || {
        make_module_builtin_function("oct", builtin_oct)
    });
    crate::module_ns_get_or_insert_with(ns, "bin", || {
        make_module_builtin_function("bin", builtin_bin)
    });
    crate::module_ns_get_or_insert_with(ns, "format", || {
        make_module_builtin_function("format", builtin_format)
    });
    crate::module_ns_get_or_insert_with(ns, "issubclass", || {
        make_module_builtin_function_with_arity("issubclass", builtin_issubclass, 2)
    });
    crate::module_ns_get_or_insert_with(ns, "__import__", || {
        make_module_builtin_function("__import__", builtin_dunder_import)
    });

    // Descriptor types
    crate::module_ns_get_or_insert_with(ns, "property", || {
        crate::typedef::gettypeobject(&pyre_object::descriptor::PROPERTY_TYPE)
    });
    // staticmethod/classmethod registered as types for isinstance() support.
    // The type's __new__ creates the descriptor wrapper.
    crate::module_ns_get_or_insert_with(ns, "staticmethod", || {
        crate::typedef::gettypeobject(&pyre_object::function::STATICMETHOD_TYPE)
    });
    crate::module_ns_get_or_insert_with(ns, "classmethod", || {
        crate::typedef::gettypeobject(&pyre_object::function::CLASSMETHOD_TYPE)
    });
}

/// `pypy/objspace/std/dictmultiobject.py:60-69
/// allocate_and_init_instance(module=True)` parity — allocate the
/// builtins module dict as a `W_ModuleDictObject` backed by
/// `ModuleDictStrategy` (`celldict.py:28`). `install_default_builtins`
/// writes directly into the GC module dict. MixedModule parity stamps
/// interp-level builtins with `__module__ = "builtins"` so pickle can
/// save them by reference without an unstable `whichmodule` guess.
pub fn new_builtin_module_dict() -> pyre_object::PyObjectRef {
    crate::typedef::init_typeobjects();
    let w_dict = pyre_object::w_module_dict_new();
    let _roots = pyre_object::gc_roots::push_roots();
    let save_point = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_dict);
    let name_obj = pyre_object::w_str_new("builtins");
    pyre_object::gc_roots::pin_root(name_obj);
    install_default_builtins(w_dict);
    let keys: Vec<String> = unsafe { pyre_object::w_dict_str_entries(w_dict) }
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    for key in &keys {
        if let Some(value) = unsafe { pyre_object::w_dict_getitem_str(w_dict, key) } {
            unsafe {
                // MixedModule._load_lazily: a function directly in the builtins
                // module is a non-descriptor `BuiltinFunction` (no `__get__`).
                crate::function::demote_module_function_to_builtin(value);
                crate::function::builtin_function_set_module(
                    value,
                    pyre_object::gc_roots::shadow_stack_get(save_point + 1),
                );
            }
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

fn input_eof_error() -> crate::PyError {
    let mut error = crate::PyError::value_error("");
    if let Some(cls) = lookup_exc_class("EOFError") {
        let args = [cls];
        if let Ok(exc) = exc_exception_new(&args) {
            error.exc_object = exc;
        }
    }
    error
}

/// `pypy/module/__builtin__/app_io.py:24-66 input` — non-readline path.
///
/// The tty/readline hook remains owned by `sys.__raw_input__`; when it is not
/// installed (the normal Pyre configuration), this is the literal app-level
/// sequence: fetch the three live `sys` streams, flush stderr, write and flush
/// the prompt, call `stdin.readline()`, then strip one trailing newline.
fn builtin_input(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "input() takes no keyword arguments",
        ));
    }
    if pos.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "input expected at most 1 argument, got {}",
            pos.len()
        )));
    }

    let prompt = pos
        .first()
        .copied()
        .unwrap_or_else(|| pyre_object::w_str_new(""));
    let Some(sys) = crate::importing::get_sys_module("sys") else {
        return Err(crate::PyError::runtime_error("lost sys.stdin"));
    };

    let _roots = pyre_object::gc_roots::push_roots();
    let root = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(sys);
    pyre_object::gc_roots::pin_root(prompt);

    // `bltinmodule.c builtin_input_impl`: an absent — or `None` — standard
    // stream is a `RuntimeError`, not an attribute error from the stream call.
    let stream = |name: &str| -> Result<PyObjectRef, crate::PyError> {
        let sys = pyre_object::gc_roots::shadow_stack_get(root);
        let lost = || crate::PyError::runtime_error(format!("lost sys.{name}"));
        match crate::baseobjspace::getattr_str(sys, name) {
            Ok(value) if unsafe { pyre_object::is_none(value) } => Err(lost()),
            Ok(value) => Ok(value),
            Err(error) if error.kind == crate::PyErrorKind::AttributeError => Err(lost()),
            Err(error) => Err(error),
        }
    };

    let stdin = stream("stdin")?;
    pyre_object::gc_roots::pin_root(stdin);
    let stdin_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let stdout = stream("stdout")?;
    pyre_object::gc_roots::pin_root(stdout);
    let stdout_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let stderr = stream("stderr")?;
    pyre_object::gc_roots::pin_root(stderr);
    let stderr_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    // app_io.py:44 `stderr.flush()`.
    let stderr_flush = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(stderr_slot),
        "flush",
    )?;
    pyre_object::gc_roots::pin_root(stderr_flush);
    crate::call_and_check(
        pyre_object::gc_roots::shadow_stack_get(pyre_object::gc_roots::shadow_stack_len() - 1),
        &[],
    )?;

    // app_io.py `_write_prompt`: `str(prompt)`, `stdout.write`, then an
    // optional `stdout.flush`.
    let prompt_text = builtin_str(&[pyre_object::gc_roots::shadow_stack_get(root + 1)])?;
    pyre_object::gc_roots::pin_root(prompt_text);
    let prompt_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let stdout_write = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(stdout_slot),
        "write",
    )?;
    pyre_object::gc_roots::pin_root(stdout_write);
    let write_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    crate::call_and_check(
        pyre_object::gc_roots::shadow_stack_get(write_slot),
        &[pyre_object::gc_roots::shadow_stack_get(prompt_slot)],
    )?;
    match crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(stdout_slot),
        "flush",
    ) {
        Ok(flush) => {
            pyre_object::gc_roots::pin_root(flush);
            crate::call_and_check(
                pyre_object::gc_roots::shadow_stack_get(
                    pyre_object::gc_roots::shadow_stack_len() - 1,
                ),
                &[],
            )?;
        }
        Err(error) if error.kind == crate::PyErrorKind::AttributeError => {}
        Err(error) => return Err(error),
    }

    // app_io.py:56-65 `line = stdin.readline()` and strip one LF.
    let readline = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(stdin_slot),
        "readline",
    )?;
    pyre_object::gc_roots::pin_root(readline);
    let line = crate::call_and_check(
        pyre_object::gc_roots::shadow_stack_get(pyre_object::gc_roots::shadow_stack_len() - 1),
        &[],
    )?;
    if unsafe { !pyre_object::is_str(line) } {
        return Err(crate::PyError::type_error(
            "input(): stdin.readline() returned non-string",
        ));
    }
    let text = unsafe { pyre_object::w_str_get_wtf8(line) };
    if text.as_bytes().is_empty() {
        return Err(input_eof_error());
    }
    if text.as_bytes().last() == Some(&b'\n') {
        let body = rustpython_wtf8::Wtf8::from_bytes(&text.as_bytes()[..text.len() - 1])
            .expect("removing ASCII LF preserves WTF-8");
        Ok(pyre_object::w_str_from_wtf8_managed(body.to_wtf8_buf()))
    } else {
        Ok(line)
    }
}

fn builtin_print(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // Check if last arg is a kwargs dict (from CALL_KW builtin dispatch).
    // Distinguished from regular dict args by __pyre_kw__ marker key.
    let is_kwargs = !args.is_empty()
        && unsafe {
            let last = *args.last().unwrap();
            is_dict(last)
                && pyre_object::w_dict_getitem_str(last, "__pyre_kw__")
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
        let end_val = unsafe { pyre_object::w_dict_getitem_str(kwargs, "end") };
        let sep_val = unsafe { pyre_object::w_dict_getitem_str(kwargs, "sep") };
        // The type check is up front; the str() rendering happens at write
        // time so a raising `__str__` leaves the preceding output in place.
        let end_obj = print_sep_check(end_val, "end")?;
        let sep_obj = print_sep_check(sep_val, "sep")?;
        // `file=None` (or absent) uses the native stdout path; any other
        // object is written through its `write` / `flush` methods.
        let file_obj = match unsafe { pyre_object::w_dict_getitem_str(kwargs, "file") } {
            Some(f) if !unsafe { pyre_object::is_none(f) } => Some(f),
            _ => None,
        };
        let flush = match unsafe { pyre_object::w_dict_getitem_str(kwargs, "flush") } {
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
        let s_obj = pyre_object::w_str_from_wtf8_managed(unsafe { crate::py_str_wtf8(source)? });
        // `s_obj` is a fresh managed str reachable only through this Rust local.
        // `call_method`'s "write" attribute lookup can run a Python descriptor or
        // `__getattr__`, re-entering the eval loop where the `PYRE_GC_INTERP`
        // safepoint may sweep an unrooted old-gen object. Pin it across the call.
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(s_obj);
        let s_obj =
            pyre_object::gc_roots::shadow_stack_get(pyre_object::gc_roots::shadow_stack_len() - 1);
        let r = crate::baseobjspace::call_method(fp, "write", &[s_obj]);
        if r.is_null() {
            return Err(crate::call::take_call_error()
                .unwrap_or_else(|| crate::PyError::runtime_error("print: file.write() failed")));
        }
        Ok(())
    };
    // A default separator / terminator is a fixed ASCII literal; on the native
    // stdout path write it straight to the stream rather than building a
    // throwaway str object per gap/line.  The `file=` path still hands a str
    // to `file.write`.
    let emit_literal = |lit: &str| -> Result<(), crate::PyError> {
        let Some(fp) = file else {
            crate::print_output(lit);
            return Ok(());
        };
        // Pin the managed separator/terminator across the write, whose "write"
        // lookup may re-enter the eval loop and reach the safepoint. See `emit`.
        let s = pyre_object::w_str_new_managed(lit);
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(s);
        let s =
            pyre_object::gc_roots::shadow_stack_get(pyre_object::gc_roots::shadow_stack_len() - 1);
        let r = crate::baseobjspace::call_method(fp, "write", &[s]);
        if r.is_null() {
            return Err(crate::call::take_call_error()
                .unwrap_or_else(|| crate::PyError::runtime_error("print: file.write() failed")));
        }
        Ok(())
    };
    for (i, &obj) in positional.iter().enumerate() {
        if i > 0 {
            match sep {
                Some(s) => emit(s)?,
                None => emit_literal(" ")?,
            }
        }
        emit(obj)?;
    }
    match end {
        Some(e) => emit(e)?,
        None => emit_literal("\n")?,
    }
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

/// `sys.excepthook(exc_type, exc_value, exc_tb)` and its preserved
/// `__excepthook__` alias.  CPython routes both through `_PyErr_Display`;
/// using the shared structured renderer keeps exception chains and
/// tracebacks visible to tests that deliberately call the original hook.
pub(crate) fn sys_excepthook(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let value = args.get(1).copied().unwrap_or_else(w_none);
    let traceback = args.get(2).copied().unwrap_or_else(w_none);
    let mut rendered = Vec::new();
    crate::error::write_exception_from_parts(&mut rendered, value, traceback)
        .map_err(|_| crate::PyError::runtime_error("sys.excepthook: failed to write exception"))?;
    // Route through the live `sys.stderr`, not directly through the host
    // seam: tests and applications are allowed to replace `sys.stderr` and
    // `sys.__excepthook__` must honor that replacement.
    let stderr = crate::importing::get_sys_module("sys")
        .and_then(|sys| crate::baseobjspace::getattr_str(sys, "stderr").ok());
    if let Some(stderr) = stderr {
        // An application that set `sys.stderr = None` disabled the stream, and
        // `_PyErr_Display` returns without printing in that case.  Only a
        // missing `sys` or a failing lookup may fall through to the host seam;
        // treating the two alike leaks the render past the configured sink.
        if stderr.is_null() || unsafe { pyre_object::is_none(stderr) } {
            return Ok(w_none());
        }
        let text = String::from_utf8_lossy(&rendered).into_owned();
        let w_text = pyre_object::w_str_new(&text);
        let result = crate::baseobjspace::call_method(stderr, "write", &[w_text]);
        if result.is_null() {
            let _ = crate::call::take_call_error();
        } else {
            return Ok(w_none());
        }
    }
    crate::host_seam::emit_stderr(&rendered);
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
        crate::gateway::builtin_code_fn_eq(
            crate::gateway::builtin_code_get(code),
            builtin_len as crate::gateway::BuiltinCodeFn,
        )
    }
}

/// True iff `callable` is the canonical builtin `repr` function object.
/// The JIT walker uses the builtin-code identity to distinguish it from an
/// arbitrary replacement stored under the same global name.
pub fn is_builtin_repr_function(callable: PyObjectRef) -> bool {
    unsafe {
        if callable.is_null() || !crate::is_function(callable) {
            return false;
        }
        let code = crate::function_get_code(callable) as PyObjectRef;
        if code.is_null() || !crate::gateway::is_builtin_code(code) {
            return false;
        }
        crate::gateway::builtin_code_fn_eq(
            crate::gateway::builtin_code_get(code),
            builtin_repr as crate::gateway::BuiltinCodeFn,
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
        return Ok(w_int_new(crate::baseobjspace::len_w(*obj)?));
    }
    let obj = parse_single_required(args, "obj", "len")?;
    Ok(w_int_new(crate::baseobjspace::len_w(obj)?))
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
            let val = w_long_get_value(obj);
            // rbigint.py:1303-1308 returns `self` for nonnegative values;
            // longobject.py:272-277 then allocates a fresh W_LongObject around
            // that same rbigint. Preserve both wrapper identity and payload
            // sharing instead of translating `Clone` into another GC payload.
            if val.get_sign() != -1 {
                return Ok(pyre_object::longobject::w_long_from_raw(
                    pyre_object::longobject::w_long_get_raw_value(obj),
                ));
            }
            return Ok(w_long_new(val.neg()));
        }
        if is_float(obj) {
            return Ok(w_float_new(w_float_get_value(obj).abs()));
        }
        if pyre_object::is_complex(obj) {
            return crate::objspace::descroperation::complex_abs(obj);
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
        crate::baseobjspace::object_functionstr_type_name(obj)
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
    if !args.is_empty() {
        let last = args[args.len() - 1];
        // The marker dict stores an unforgeable sentinel under `__pyre_kw__`
        // (`call_with_kwargs`), so detection is by value identity.  A dict
        // passed positionally that merely contains a `__pyre_kw__` string key
        // (`float({'__pyre_kw__': True})`) carries a different value and is a
        // value, not the marker, so it must not be stripped.
        let is_marker = (unsafe { is_dict(last) }) && builtin_kwargs_marker_dict(last);
        if is_marker {
            return (&args[..args.len() - 1], Some(last));
        }
    }
    (args, None)
}

/// Cold dictionary-strategy half of the flat builtin-keyword ABI.
///
/// The caller first proves `last` is a dict.  Keeping the strategy dispatch
/// residual lets the trace-visible non-dict path reject the marker test
/// without entering `w_dict_getitem_str`'s dynamic strategy function.
#[majit_macros::dont_look_inside]
pub fn builtin_kwargs_marker_dict(last: PyObjectRef) -> bool {
    unsafe {
        pyre_object::w_dict_getitem_str(last, "__pyre_kw__")
            .is_some_and(pyre_object::kw_marker::is_kw_marker_sentinel)
    }
}

/// True when the kwargs dict from [`split_builtin_kwargs`] carries a real
/// keyword (any entry other than the `__pyre_kw__` marker).  An empty
/// `**{}` therefore reports `false`.
pub(crate) fn has_real_kwargs(kwargs: Option<PyObjectRef>) -> bool {
    real_kwarg_count(kwargs) > 0
}

/// Number of real keyword arguments in the kwargs dict from
/// [`split_builtin_kwargs`] — every entry other than the `__pyre_kw__`
/// marker.  The clinic-style "takes at most N arguments (M given)" builtins
/// (`sum`, `round`, `pow`) count positionals plus this against their limit.
///
/// Read through the surrogate-preserving iterator, the same one
/// [`call_forwarding_args`] rebuilds the keywords with: `w_dict_str_entries`
/// drops a `**{'\udc80': v}` key outright, which would make this report a
/// keyword-free call and let the keyword be silently discarded.
pub(crate) fn real_kwarg_count(kwargs: Option<PyObjectRef>) -> usize {
    let Some(dict) = kwargs else {
        return 0;
    };
    unsafe { pyre_object::w_dict_str_entries_wtf8(dict) }
        .iter()
        .filter(|(key, _)| key.as_str() != Ok("__pyre_kw__"))
        .count()
}

/// The real keyword `(name, value)` pairs in the kwargs dict from
/// [`split_builtin_kwargs`] — every entry other than the `__pyre_kw__`
/// marker.  A builtin that has to re-issue its own call as a keyword call
/// (`_thread._local` replays the constructor per thread, `os_local.py:57`)
/// hands these to `call::call_with_kwargs`.
pub(crate) fn builtin_kwarg_entries(kwargs: Option<PyObjectRef>) -> Vec<(Wtf8Buf, PyObjectRef)> {
    let Some(dict) = kwargs else {
        return Vec::new();
    };
    let marker = Wtf8Buf::from_string("__pyre_kw__".to_string());
    unsafe { pyre_object::w_dict_str_entries_wtf8(dict) }
        .into_iter()
        .filter(|(key, _)| *key != marker)
        .collect()
}

/// Look up a single keyword argument from the kwargs dict produced by
/// `split_builtin_kwargs`. Returns `None` when no kwargs dict is present
/// or the requested key is absent.
pub(crate) fn kwarg_get(kwargs: Option<PyObjectRef>, name: &str) -> Option<PyObjectRef> {
    let dict = kwargs?;
    unsafe { pyre_object::w_dict_getitem_str(dict, name) }
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
    if args.is_empty() {
        return false;
    }
    let last = args[args.len() - 1];
    (unsafe { is_dict(last) }) && builtin_kwargs_marker_dict(last)
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

/// `_PyArg_UnpackKeywords`' two argument-count checks for a callable
/// declaring `maxpos` positional-or-keyword slots followed by `kwonly`
/// keyword-only ones, of which `minargs` are required.
///
/// A call whose positionals and keywords together exceed the declared total
/// is reported against that total ("takes at most 2 arguments (3 given)"),
/// and only a call that stays within it but oversupplies the positional part
/// is reported against `maxpos`.  The order matters: `list.sort` declares no
/// positional slot and two keyword-only ones, so one stray positional is a
/// positional error while three arguments are a total-count error.
pub(crate) fn clinic_arity(
    fn_name: &str,
    npos: usize,
    nkw: usize,
    minargs: usize,
    maxpos: usize,
    kwonly: usize,
) -> Result<(), crate::PyError> {
    let maxargs = maxpos + kwonly;
    if npos + nkw > maxargs {
        return Err(crate::PyError::type_error(format!(
            "{fn_name}() takes {} {maxargs} {}argument{} ({} given)",
            if minargs < maxargs {
                "at most"
            } else {
                "exactly"
            },
            // bpo-31229: a call that passed only keywords names them, so
            // "takes exactly 1 argument (2 given)" cannot read as a claim
            // about positional arguments that were never supplied.
            if npos == 0 { "keyword " } else { "" },
            if maxargs == 1 { "" } else { "s" },
            npos + nkw,
        )));
    }
    if npos > maxpos {
        let limit = if maxpos == 0 {
            "no positional arguments".to_string()
        } else {
            format!(
                "{} {maxpos} positional argument{} ({npos} given)",
                if minargs < maxpos {
                    "at most"
                } else {
                    "exactly"
                },
                if maxpos == 1 { "" } else { "s" },
            )
        };
        return Err(crate::PyError::type_error(format!(
            "{fn_name}() takes {limit}"
        )));
    }
    Ok(())
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
    clinic_arity(
        fn_name,
        positional.len(),
        real_kwarg_count(kwargs),
        required.iter().filter(|r| **r).count(),
        names.len(),
        0,
    )?;
    let mut scope: Vec<PyObjectRef> = vec![PY_NULL; names.len()];
    let mut filled: Vec<bool> = vec![false; names.len()];
    let mut unknown: Option<String> = None;
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
                            "argument for {fn_name}() given by name ('{key}') and position ({})",
                            idx + 1,
                        )));
                    }
                    scope[idx] = *val;
                    filled[idx] = true;
                }
                // `_PyArg_UnpackKeywords` collects the unrecognized names and
                // only reports them once every declared slot has been filled,
                // so a call that misses a required argument is reported
                // against that argument even when it also passed a keyword
                // the function does not know.
                None => unknown = Some(key.to_string_lossy().into_owned()),
            }
        }
    }
    for i in 0..names.len() {
        if !filled[i] && required[i] {
            return Err(crate::PyError::type_error(format!(
                "{fn_name}() missing required argument '{}' (pos {})",
                names[i],
                i + 1,
            )));
        }
    }
    if let Some(key) = unknown {
        return Err(crate::PyError::type_error(format!(
            "{fn_name}() got an unexpected keyword argument '{key}'"
        )));
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
    // Read the machine-word value of a bool / int / long object, raising
    // OverflowError when a bigint does not fit. Returns `None` for anything
    // else so the caller can fall through to the `__index__` lookup / error.
    unsafe fn as_index_value(o: PyObjectRef) -> Option<Result<i64, crate::PyError>> {
        unsafe {
            if pyre_object::is_bool(o) {
                return Some(Ok(pyre_object::w_bool_get_value(o) as i64));
            }
            if pyre_object::is_int(o) {
                return Some(Ok(pyre_object::w_int_get_value(o)));
            }
            if pyre_object::is_long(o) {
                let big = pyre_object::longobject::w_long_get_value(o);
                return Some(
                    if pyre_object::longobject::jit_bigint_to_i64_fits(big) != 0 {
                        Ok(pyre_object::longobject::jit_bigint_to_i64_value(big))
                    } else {
                        Err(crate::PyError::overflow_error(
                            "int too large to convert to int",
                        ))
                    },
                );
            }
        }
        None
    }
    unsafe {
        if let Some(v) = as_index_value(obj) {
            return v;
        }
        if let Some(w_type) = crate::typedef::r#type(obj) {
            if let Some(index_fn) =
                crate::baseobjspace::lookup_in_type(w_type.as_ptr(), "__index__")
            {
                let result = crate::call::call_function_impl_result(index_fn, &[obj])?;
                if let Some(v) = as_index_value(result) {
                    return v;
                }
            }
        }
    }
    let tp_name = unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
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
        // PyPy's W_BoolObject subclasses W_IntObject and shares `intval`;
        // pyre uses a distinct layout, so preserve that upstream semantic arm
        // with the correct storage projection before testing `is_int`.
        if is_bool(obj) {
            BigInt::from(w_bool_get_value(obj) as i64)
        } else if is_int(obj) {
            BigInt::from(w_int_get_value(obj))
        } else {
            // PyPy's bigint_w/asbigint returns W_LongObject.num itself.
            w_long_get_value(obj).translated_alias()
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
    // `min_max` unpacks its positionals before it looks at the keywords, so a
    // keywords-only call is reported against the missing operand.
    if positional.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "{fn_name} expected at least 1 argument, got 0"
        )));
    }
    // functional.py:198-201 — only `key` and `default` are accepted.
    kwarg_reject_unknown(kwargs, &["key", "default"], fn_name)?;
    let key_fn = kwarg_get(kwargs, "key").filter(|k| unsafe { !pyre_object::is_none(*k) });
    let default = kwarg_get(kwargs, "default");
    // functional.py:206-210 — `default=` is only meaningful for the
    // single-iterable form; combining it with multiple positional args
    // is a user error.
    if positional.len() > 1 && default.is_some() {
        return Err(crate::PyError::type_error(format!(
            "Cannot specify a default for {fn_name}() with multiple positional arguments"
        )));
    }
    if positional.len() == 1 {
        min_max_sequence(positional[0], key_fn, default, want_max, fn_name)
    } else {
        min_max_multiple_args(positional, key_fn, want_max)
    }
}

/// `pypy/module/__builtin__/functional.py:125-158 min_max_sequence`.
fn min_max_sequence(
    sequence: PyObjectRef,
    key_fn: Option<PyObjectRef>,
    default: Option<PyObjectRef>,
    want_max: bool,
    fn_name: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let sequence_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(sequence);
    let it = crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(sequence_slot))?;
    let iterator_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(it);
    // Dict-view iterators are off-GC carriers, so retain their source dict
    // explicitly while the iterator is rooted by this native loop.
    if unsafe {
        pyre_object::dictmultiobject::is_dict_view_iterator(
            pyre_object::gc_roots::shadow_stack_get(iterator_slot),
        )
    } {
        let w_dict = unsafe {
            pyre_object::dictmultiobject::w_dict_view_iterator_get_dict(
                pyre_object::gc_roots::shadow_stack_get(iterator_slot),
            )
        };
        pyre_object::gc_roots::pin_root(w_dict);
    }
    let key_fn_slot = key_fn.map(|key| {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(key);
        slot
    });
    let has_default = default.is_some();
    let best_item_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(default.unwrap_or_else(pyre_object::w_none));
    let best_key_slot = if key_fn_slot.is_some() {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::w_none());
        slot
    } else {
        best_item_slot
    };
    let candidate_item_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(pyre_object::w_none());
    let candidate_key_slot = if key_fn_slot.is_some() {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::w_none());
        slot
    } else {
        candidate_item_slot
    };
    let cmp_op = if want_max {
        crate::baseobjspace::CompareOp::Gt
    } else {
        crate::baseobjspace::CompareOp::Lt
    };
    let mut has_item = false;
    loop {
        let it_now = pyre_object::gc_roots::shadow_stack_get(iterator_slot);
        let item = match crate::baseobjspace::next(it_now) {
            Ok(item) => item,
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        };
        pyre_object::gc_roots::shadow_stack_set(candidate_item_slot, item);
        if let Some(key_fn_slot) = key_fn_slot {
            let candidate_key = crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(key_fn_slot),
                &[pyre_object::gc_roots::shadow_stack_get(candidate_item_slot)],
            )?;
            pyre_object::gc_roots::shadow_stack_set(candidate_key_slot, candidate_key);
        }
        if !has_item {
            pyre_object::gc_roots::shadow_stack_set(
                best_item_slot,
                pyre_object::gc_roots::shadow_stack_get(candidate_item_slot),
            );
            if key_fn_slot.is_some() {
                pyre_object::gc_roots::shadow_stack_set(
                    best_key_slot,
                    pyre_object::gc_roots::shadow_stack_get(candidate_key_slot),
                );
            }
            has_item = true;
            continue;
        }
        if crate::baseobjspace::is_true(crate::baseobjspace::compare(
            pyre_object::gc_roots::shadow_stack_get(candidate_key_slot),
            pyre_object::gc_roots::shadow_stack_get(best_key_slot),
            cmp_op,
        )?)? {
            pyre_object::gc_roots::shadow_stack_set(
                best_item_slot,
                pyre_object::gc_roots::shadow_stack_get(candidate_item_slot),
            );
            if key_fn_slot.is_some() {
                pyre_object::gc_roots::shadow_stack_set(
                    best_key_slot,
                    pyre_object::gc_roots::shadow_stack_get(candidate_key_slot),
                );
            }
        }
    }
    if !has_item && !has_default {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            format!("{fn_name}() iterable argument is empty"),
        ));
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(best_item_slot))
}

/// `pypy/module/__builtin__/functional.py:163-184 min_max_multiple_args`.
fn min_max_multiple_args(
    positional: &[PyObjectRef],
    key_fn: Option<PyObjectRef>,
    want_max: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let item_base = pyre_object::gc_roots::shadow_stack_len();
    for &item in positional {
        pyre_object::gc_roots::pin_root(item);
    }
    let key_fn_slot = key_fn.map(|key| {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(key);
        slot
    });
    let cmp_op = if want_max {
        crate::baseobjspace::CompareOp::Gt
    } else {
        crate::baseobjspace::CompareOp::Lt
    };
    let best_item_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(pyre_object::gc_roots::shadow_stack_get(item_base));
    let best_key_slot = if let Some(key_fn_slot) = key_fn_slot {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::w_none());
        let first_key = crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(key_fn_slot),
            &[pyre_object::gc_roots::shadow_stack_get(best_item_slot)],
        )?;
        pyre_object::gc_roots::shadow_stack_set(slot, first_key);
        slot
    } else {
        best_item_slot
    };
    let candidate_item_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(pyre_object::gc_roots::shadow_stack_get(best_item_slot));
    let candidate_key_slot = if key_fn_slot.is_some() {
        let slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::gc_roots::shadow_stack_get(best_key_slot));
        slot
    } else {
        candidate_item_slot
    };
    for i in 1..positional.len() {
        pyre_object::gc_roots::shadow_stack_set(
            candidate_item_slot,
            pyre_object::gc_roots::shadow_stack_get(item_base + i),
        );
        if let Some(key_fn_slot) = key_fn_slot {
            let candidate_key = crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(key_fn_slot),
                &[pyre_object::gc_roots::shadow_stack_get(candidate_item_slot)],
            )?;
            pyre_object::gc_roots::shadow_stack_set(candidate_key_slot, candidate_key);
        }
        // `functional.py:139 if space.is_true(space.gt(key, best_key))`
        // — route through the generic comparison dispatch which
        // handles int/long/str/float/tuple natively and falls
        // through to user-defined `__gt__`/`__lt__` for other
        // types.  Errors (TypeError from incomparable types or an
        // exception from the key call, as in functional.py min_max)
        // are propagated to the caller.
        if crate::baseobjspace::is_true(crate::baseobjspace::compare(
            pyre_object::gc_roots::shadow_stack_get(candidate_key_slot),
            pyre_object::gc_roots::shadow_stack_get(best_key_slot),
            cmp_op,
        )?)? {
            pyre_object::gc_roots::shadow_stack_set(
                best_item_slot,
                pyre_object::gc_roots::shadow_stack_get(candidate_item_slot),
            );
            if key_fn_slot.is_some() {
                pyre_object::gc_roots::shadow_stack_set(
                    best_key_slot,
                    pyre_object::gc_roots::shadow_stack_get(candidate_key_slot),
                );
            }
        }
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(best_item_slot))
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
pub(crate) fn type_new_wrap_special_methods(ns: PyObjectRef) {
    let _ns_root = pyre_object::gc_roots::push_roots();
    let ns_root = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(ns);
    let ns = pyre_object::gc_roots::shadow_stack_get(ns_root);
    if let Some(f) = unsafe { pyre_object::w_dict_getitem_str(ns, "__new__") } {
        if unsafe { crate::function::is_function(f) }
            && !unsafe { pyre_object::function::is_staticmethod(f) }
        {
            let wrapped = pyre_object::function::w_staticmethod_new(f);
            let wrapped_root = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(wrapped);
            let ns = pyre_object::gc_roots::shadow_stack_get(ns_root);
            let wrapped = pyre_object::gc_roots::shadow_stack_get(wrapped_root);
            unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, "__new__", wrapped) };
        }
    }
    for name in ["__init_subclass__", "__class_getitem__"] {
        let ns = pyre_object::gc_roots::shadow_stack_get(ns_root);
        if let Some(f) = unsafe { pyre_object::w_dict_getitem_str(ns, name) } {
            if unsafe { crate::function::is_function(f) }
                && !unsafe { pyre_object::function::is_classmethod(f) }
            {
                let wrapped = pyre_object::function::w_classmethod_new(f);
                let wrapped_root = pyre_object::gc_roots::shadow_stack_len();
                pyre_object::gc_roots::pin_root(wrapped);
                let ns = pyre_object::gc_roots::shadow_stack_get(ns_root);
                let wrapped = pyre_object::gc_roots::shadow_stack_get(wrapped_root);
                unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, name, wrapped) };
            }
        }
    }
}

/// A class that supplies equality but no hash is explicitly unhashable.
pub(crate) fn type_new_set_hash_if_eq(ns: PyObjectRef) {
    let _ns_root = pyre_object::gc_roots::push_roots();
    let ns_root = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(ns);
    let ns = pyre_object::gc_roots::shadow_stack_get(ns_root);
    if unsafe { pyre_object::w_dict_getitem_str(ns, "__eq__") }.is_some()
        && unsafe { pyre_object::w_dict_getitem_str(ns, "__hash__") }.is_none()
    {
        let ns = pyre_object::gc_roots::shadow_stack_get(ns_root);
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, "__hash__", pyre_object::w_none()) };
    }
}

/// CPython `type_new_set_classdict`: every new class owns a `__doc__` entry,
/// using None when its namespace has no docstring. A declared `__doc__` slot
/// is the sole exception because the class variable would collide with its
/// member descriptor.
pub(crate) fn type_new_set_doc(ns: PyObjectRef) -> crate::PyResult {
    if unsafe { pyre_object::w_dict_getitem_str(ns, "__doc__") }.is_some() {
        return Ok(pyre_object::w_none());
    }
    // `create_all_slots` owns iteration of `__slots__`.  In particular, it
    // must consume a one-shot iterator exactly once.  Defer the default doc
    // entry whenever slots are present; `create_all_slots` installs it after
    // collecting the complete slot-name sequence if `__doc__` was not among
    // those names.
    if unsafe { pyre_object::w_dict_getitem_str(ns, "__slots__") }.is_none() {
        unsafe { pyre_object::w_dict_setitem_str_no_proxy(ns, "__doc__", pyre_object::w_none()) };
    }
    Ok(pyre_object::w_none())
}

/// PyPy `typeobject.py:223-235 W_TypeObject.__init__`: consume the compiler's
/// `__qualname__` entry before slot setup and retain it on the type object.
/// Keeping it in the namespace changes `cls.__dict__` and makes libraries
/// such as `typing.Protocol` mistake it for a user-declared member.
pub(crate) fn type_new_take_qualname(w_type: PyObjectRef, ns: PyObjectRef) -> crate::PyResult {
    let Some(value) = (unsafe { pyre_object::w_dict_getitem_str(ns, "__qualname__") }) else {
        return Ok(pyre_object::w_none());
    };
    if !unsafe { pyre_object::is_str(value) } {
        return Err(crate::PyError::type_error(format!(
            "type __qualname__ must be a str, not {}",
            crate::type_methods::arg_type_name(value),
        )));
    }
    check_surrogate(value)?;
    unsafe {
        pyre_object::w_type_set_qualname(w_type, value);
        pyre_object::w_dict_delitem_str_no_proxy(ns, "__qualname__");
    }
    Ok(pyre_object::w_none())
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
        // typeobject.py:_check_new_args — validate the three public
        // arguments before attempting to copy or normalise any of them.
        if !unsafe { crate::baseobjspace::isinstance_str_w(name_obj) } {
            return Err(crate::PyError::type_error(format!(
                "type() argument 1 must be str, not {}",
                unsafe { pyre_object::type_name_of(name_obj) }
            )));
        }
        if !unsafe { is_tuple(bases) } {
            return Err(crate::PyError::type_error(format!(
                "type() argument 2 must be tuple, not {}",
                unsafe { pyre_object::type_name_of(bases) }
            )));
        }
        let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
        if !unsafe { crate::baseobjspace::isinstance_w(w_namespace_dict, w_dict_type) } {
            return Err(crate::PyError::type_error(format!(
                "type() argument 3 must be dict, not {}",
                unsafe { pyre_object::type_name_of(w_namespace_dict) }
            )));
        }
        let w_ns_backing = unsafe { crate::type_methods::resolve_dict_backing(w_namespace_dict) };
        // typeobject.py:953 `_check_surrogate(space, name)` — reject a lone
        // surrogate in the name before it is read as UTF-8 below.
        check_surrogate(name_obj)?;
        for cp in unsafe { pyre_object::w_str_get_wtf8(name_obj) }.code_points() {
            if cp.to_u32() == 0 {
                return Err(crate::PyError::value_error(
                    "type name must not contain null characters",
                ));
            }
        }
        // typeobject.py `type.__new__` — `isinstance_w(w_bases, w_tuple)` and
        // `isinstance_w(w_dict, w_dict)`: bases must be a tuple and the
        // namespace a dict (subclasses of each are accepted, e.g. a dict
        // subclass namespace); neither is silently coerced.
        let w_tuple_type = crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE);
        if !unsafe { crate::baseobjspace::isinstance_w(bases, w_tuple_type) } {
            return Err(crate::PyError::type_error(format!(
                "type() argument 2 must be tuple, not {}",
                crate::baseobjspace::object_functionstr_type_name(bases)
            )));
        }
        let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
        if !unsafe { crate::baseobjspace::isinstance_w(w_namespace_dict, w_dict_type) } {
            return Err(crate::PyError::type_error(format!(
                "type() argument 3 must be dict, not {}",
                crate::baseobjspace::object_functionstr_type_name(w_namespace_dict)
            )));
        }
        let name = crate::baseobjspace::str_utf8_w(name_obj)?;

        // typeobject.py:938-942 `_create_new_type` — direct three-argument
        // `type()` never performs PEP 560 base rewriting.  A non-type base
        // advertising `__mro_entries__` gets the dedicated diagnostic before
        // metaclass calculation; class statements resolve it earlier in
        // `__build_class__` and therefore never reach this branch.
        let n_bases = unsafe { w_tuple_len(bases) };
        for index in 0..n_bases {
            let Some(base) = (unsafe { pyre_object::w_tuple_getitem(bases, index as i64) }) else {
                continue;
            };
            if !unsafe { pyre_object::is_type(base) }
                && unsafe { crate::baseobjspace::lookup(base, "__mro_entries__") }.is_some()
            {
                return Err(crate::PyError::type_error(
                    "type() doesn't support MRO entry resolution; use types.new_class()",
                ));
            }
        }

        // CPython: calculate_metaclass — if bases have a custom metaclass,
        // delegate to that metaclass instead of using type.__new__ directly.
        if w_metaclass.is_null() && !bases.is_null() && unsafe { is_tuple(bases) } {
            let n = unsafe { w_tuple_len(bases) };
            for i in 0..n {
                if let Some(base) = unsafe { pyre_object::w_tuple_getitem(bases, i as i64) } {
                    if unsafe { pyre_object::is_type(base) } {
                        // baseobjspace.py — metaclass from w_class
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
                            return crate::call::call_function_impl_result(
                                w_metaclass,
                                &metaclass_args,
                            );
                        }
                    }
                }
            }
        }

        // Copy the namespace into the class-namespace scratch.
        // `w_dict_items` dispatches through `is_module_dict`, so the rare
        // `__build_class__` case where the namespace is a W_ModuleDictObject
        // still walks correctly.
        let _class_ns_root = pyre_object::gc_roots::push_roots();
        let namespace_root = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(w_namespace_dict);
        let class_ns = pyre_object::w_dict_new();
        let class_ns_root = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(class_ns);
        // type_new_classcell — capture the `__classcell__` cell and keep
        // both explicit class cells out of the new type's `__dict__`
        // (CPython consumes them here rather than storing them).
        let mut classcell_root = None;
        let mut classdictcell_root = None;
        // `type.__new__` accepts any `dict` subclass as the namespace
        // (the check is `PyDict_Check`, not `PyDict_CheckExact`); resolve
        // the dict backing so e.g. an `enum._EnumDict` class body is
        // walked instead of dropped.
        let w_namespace_dict = pyre_object::gc_roots::shadow_stack_get(namespace_root);
        if !w_ns_backing.is_null() {
            let backing_root = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(w_ns_backing);
            let items = unsafe { pyre_object::w_dict_items(w_ns_backing) };
            let item_root = pyre_object::gc_roots::shadow_stack_len();
            for &(key, value) in &items {
                pyre_object::gc_roots::pin_root(key);
                pyre_object::gc_roots::pin_root(value);
            }
            let mut has_non_string_key = false;
            for index in 0..items.len() {
                let key = pyre_object::gc_roots::shadow_stack_get(item_root + index * 2);
                let value = pyre_object::gc_roots::shadow_stack_get(item_root + index * 2 + 1);
                let key_text = if unsafe { pyre_object::is_str(key) } {
                    Some(unsafe { pyre_object::w_str_get_wtf8(key) })
                } else {
                    has_non_string_key = true;
                    None
                };
                if key_text
                    .as_ref()
                    .is_some_and(|key| key.as_str() == Ok("__classcell__"))
                {
                    if !unsafe { pyre_object::is_cell(value) } {
                        let tp_name = match unsafe { crate::typedef::r#type(value) } {
                            Some(tp) => {
                                unsafe { pyre_object::w_type_get_name(tp.as_ptr()) }.to_string()
                            }
                            None => "object".to_string(),
                        };
                        return Err(crate::PyError::type_error(format!(
                            "__classcell__ must be a nonlocal cell, not {tp_name}"
                        )));
                    }
                    let root = pyre_object::gc_roots::shadow_stack_len();
                    pyre_object::gc_roots::pin_root(value);
                    classcell_root = Some(root);
                    continue;
                }
                if key_text
                    .as_ref()
                    .is_some_and(|key| key.as_str() == Ok("__classdictcell__"))
                {
                    if unsafe { pyre_object::is_cell(value) } {
                        let root = pyre_object::gc_roots::shadow_stack_len();
                        pyre_object::gc_roots::pin_root(value);
                        classdictcell_root = Some(root);
                    }
                    continue;
                }
                let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                unsafe { pyre_object::w_dict_store(class_ns, key, value) };
            }
            if has_non_string_key {
                crate::warn::warn_category(
                    &format!("type '{}' has a non-string key in its dictionary", name),
                    "RuntimeWarning",
                    1,
                )?;
            }
        }
        // typeobject.py:ensure_common_attributes / ensure_module_attr.  Direct
        // three-argument type() calls do not pass through __build_class__, so
        // fill __module__ from the live caller frame when the namespace did
        // not supply it.
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        if unsafe { pyre_object::w_dict_getitem_str(class_ns, "__module__") }.is_none() {
            let frame = crate::eval::CURRENT_FRAME.with(|current| current.get());
            if !frame.is_null() {
                let globals = unsafe { (*frame).get_w_globals() };
                if !globals.is_null() {
                    if let Some(module) = crate::baseobjspace::finditem_str(globals, "__name__")? {
                        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
                        unsafe {
                            pyre_object::w_dict_setitem_str_no_proxy(class_ns, "__module__", module)
                        };
                    }
                }
            }
        }
        // CPython 3.14 rejects lone surrogates in the initial type doc while
        // retaining the historical freedom to assign arbitrary objects later.
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        if let Some(doc) = unsafe { pyre_object::w_dict_getitem_str(class_ns, "__doc__") } {
            if unsafe { pyre_object::is_str(doc) } {
                check_surrogate(doc)?;
            }
        }
        // W_TypeObject.__init__: pop and validate __qualname__ before slot
        // construction.  Pyre keeps it in the canonical namespace because it
        // has no separate qualname field, but preserves the same validation.
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        if let Some(qualname) = unsafe { pyre_object::w_dict_getitem_str(class_ns, "__qualname__") }
        {
            if !unsafe { crate::baseobjspace::isinstance_str_w(qualname) } {
                return Err(crate::PyError::type_error(format!(
                    "type __qualname__ must be a str, not {}",
                    unsafe { pyre_object::type_name_of(qualname) }
                )));
            }
        }
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        type_new_set_doc(class_ns)?;
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        type_new_set_hash_if_eq(class_ns);
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        type_new_wrap_special_methods(class_ns);

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
        // A metaclass conflict among the bases (or an explicit metaclass that
        // is not a subclass of every base's metaclass) is a hard error, not a
        // silent fall-back to `default_meta`.
        let w_winner = crate::call::calculate_metaclass(default_meta, w_effective_bases)?;
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
                let w_namespace_dict = pyre_object::gc_roots::shadow_stack_get(namespace_root);
                let mut new_args = vec![w_winner, name_obj, bases, w_namespace_dict];
                if args.len() > 3 {
                    new_args.extend_from_slice(&args[3..]);
                }
                return crate::call::call_function_impl_result(w_metaclass_new, &new_args);
            }
        }
        let w_metaclass = w_winner;

        // This is type.__new__'s own construction path. A different winning
        // metaclass above received the original bases without a C3 pre-check.
        unsafe { crate::baseobjspace::validate_c3_mro(w_effective_bases, false)? };

        let _dict_root = pyre_object::gc_roots::push_roots();
        let dict_root = pyre_object::gc_roots::shadow_stack_len();
        let class_ns = pyre_object::gc_roots::shadow_stack_get(class_ns_root);
        // Clone the assembled namespace preserving its cached key hashes,
        // mirroring the final `PyDict_Copy` for the type's `__dict__`.  A
        // per-key reinsert would hash every non-string key again, running a
        // side-effecting `__hash__` an extra time (and dropping any error it
        // raised, since the store cannot propagate one).
        let dict_obj = unsafe { pyre_object::w_dict_copy(class_ns) };
        pyre_object::gc_roots::pin_root(dict_obj);
        let dict_obj = pyre_object::gc_roots::shadow_stack_get(dict_root);
        let w_type = pyre_object::w_type_new(name, w_effective_bases, dict_obj as *mut u8);
        type_new_take_qualname(w_type, dict_obj)?;
        // typeobject.py:1143-1204 create_all_slots parity.
        unsafe { crate::call::create_all_slots(w_type, w_effective_bases)? };
        // rclass.py:739-743 — set w_class (typeptr) at allocation time.
        // For type objects, w_class is the metaclass (type(C) → Meta).
        // baseobjspace.py getclass() returns the metatype.
        crate::typedef::tag_subclass_instance(w_type, w_metaclass);

        // type_new_classcell — bind the captured `__classcell__` to the
        // new type so `__class__` / zero-arg `super()` in the methods
        // resolve; the key was already dropped from the namespace above.
        // PyPy `_store_type_in_classcell` runs before W_TypeObject.__init__,
        // whose `ensure_common_attributes` invokes a custom metaclass mro().
        if let Some(classcell_root) = classcell_root {
            let classcell = pyre_object::gc_roots::shadow_stack_get(classcell_root);
            unsafe { pyre_object::w_cell_set(classcell, w_type) };
        }

        // typeobject.py:1595-1613 compute_mro — a custom metaclass `mro`
        // runs after the class cell is bound and before ready().
        unsafe { crate::baseobjspace::compute_and_set_mro(w_type)? };
        // typeobject.py:373-377 ready() — link self into each base's
        // `weak_subclasses` so `mutated()` and `__subclasses__()`
        // observe this class.
        unsafe { pyre_object::typeobject::w_type_ready(w_type) };

        // CPython type_new_set_classdict precedes type_new_set_names: lazy
        // annotation thunks invoked by __set_name__ must resolve class-local
        // names through the completed type dictionary.
        if let Some(classdictcell_root) = classdictcell_root {
            let classdictcell = pyre_object::gc_roots::shadow_stack_get(classdictcell_root);
            let type_dict = unsafe { pyre_object::w_type_get_dict_ptr(w_type) as PyObjectRef };
            if !type_dict.is_null() {
                unsafe { pyre_object::w_cell_set(classdictcell, type_dict) };
            }
        }

        // _set_names (typeobject.py:1006) — call `__set_name__(owner, name)`
        // on each descriptor in the type's FINAL `__dict__` (`w_type.dict_w`),
        // i.e. the filtered namespace with `__classcell__`/`__classdictcell__`
        // already removed, not the original backing.  Collect the entries
        // first so the storage borrow is released before the call, which may
        // re-enter the type's dict.
        let dict_obj = pyre_object::gc_roots::shadow_stack_get(dict_root);
        let set_name_entries = unsafe { pyre_object::w_dict_items(dict_obj) };
        for (key, v) in set_name_entries {
            if unsafe { pyre_object::is_str(key) } {
                unsafe { crate::baseobjspace::set_name(w_type, key, v) }?;
            }
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
        return Ok(tp.as_ptr());
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

/// Manual interp2app gateway for `isinstance`.
///
/// `#[pyre_methods]` emits this same wrapper/descriptor pair automatically.
/// Builtins installed by hand must publish the equivalent `BuiltinCode.func`
/// PBC member so source translation can discover and codewrite its graph.
pub fn __pyre_wrap_builtin_isinstance(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_isinstance(args)
}

#[cfg(not(target_arch = "wasm32"))]
#[linkme::distributed_slice(crate::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
#[allow(non_upper_case_globals)]
static __pyre_wrap_builtin_isinstance_target: crate::gateway::BuiltinWrapperDescriptor =
    crate::gateway::BuiltinWrapperDescriptor {
        path: concat!(module_path!(), "::", "__pyre_wrap_builtin_isinstance"),
        func: __pyre_wrap_builtin_isinstance,
    };

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

// Descroperation helpers (lookup_type_special, try_dispatch_binary_special,
// try_int_long_pow_with_modulo, binary_builtin_type_error,
// issubtype_w) live in `crate::baseobjspace` because
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
    exc_eof_error,
    pyre_object::interp_exceptions::ExcKind::EOFError
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
    exc_stop_async_iteration,
    pyre_object::interp_exceptions::ExcKind::StopAsyncIteration
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
                    Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
                    None => "BaseException".to_string(),
                }
            };
            return Err(crate::PyError::type_error(format!(
                "{type_name}() takes no keyword arguments"
            )));
        }
    }
    // `interp_exceptions.py:123-124 descr_init` and `:277-282
    // descr_new_base_exception` both store the `args_w` list their own
    // signature was bound to, so `type.__call__` reaches here holding a list
    // that already contains exactly these objects.  Keep it rather than build
    // a second one over the same elements: `args` is read out as a fresh
    // tuple and the storage is never handed out, so its identity is not
    // observable.
    if !exception_args_already(w_self, positional) {
        let args_list = pyre_object::w_list_new(positional.to_vec());
        unsafe { pyre_object::interp_exceptions::w_exception_set_args(w_self, args_list) };
    }
    Ok(pyre_object::w_none())
}

/// `interp_exceptions.py:490-498 W_StopIteration.descr_init` — initialize
/// `w_value` to the first positional argument (or None) independently of the
/// `args_w` list, then run `W_Exception.descr_init`.
fn exc_stop_iteration_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__init__() missing 1 required positional argument: 'self'")
    })?;
    let (positional, _) = split_builtin_kwargs(&args[1..]);
    let w_value = positional
        .first()
        .copied()
        .unwrap_or_else(pyre_object::w_none);
    let result = exc_base_exception_init(args)?;
    unsafe { pyre_object::interp_exceptions::w_exception_set_value(w_self, w_value) };
    Ok(result)
}

/// `_PyArg_NoKeywords(type_name, kwds)` message for an exception initializer
/// that a subclass inherits: the reported name is the receiver's own type, not
/// the class the initializer was installed on.
fn exc_no_keywords_error(w_self: PyObjectRef, fallback: &str) -> crate::PyError {
    let type_name = unsafe {
        match crate::typedef::r#type(w_self) {
            Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
            None => fallback.to_string(),
        }
    };
    crate::PyError::type_error(format!("{type_name}() takes no keyword arguments"))
}

/// `interp_exceptions.py:836-858 W_SyntaxError.descr_init` — validate the
/// optional details sequence before forwarding the original positional
/// arguments to `BaseException.__init__`.  The details tuple must contain
/// either four fields or all six location fields; a five-field form is
/// specifically rejected because `end_offset` is required with
/// `end_lineno`.  SyntaxError subclasses inherit this initializer.
fn exc_syntax_error_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__init__() missing 1 required positional argument: 'self'")
    })?;
    let (positional, kwargs) = split_builtin_kwargs(&args[1..]);
    if has_real_kwargs(kwargs) {
        return Err(exc_no_keywords_error(w_self, "SyntaxError"));
    }
    // `fixedview` runs the details object's own iteration protocol and can
    // reach a collection safepoint, but the positional arguments live in a
    // native slice the gateway copied off the shadow stack, which a collection
    // does not rewrite.  Publish the receiver and the arguments first, then
    // read each one back from its slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let mut flat = Vec::with_capacity(positional.len() + 1);
    flat.push(w_self);
    flat.extend_from_slice(positional);
    let base = pyre_object::gc_roots::pin_roots(&flat);
    if !positional.is_empty() {
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_syntax_msg(
                pyre_object::gc_roots::shadow_stack_get(base),
                pyre_object::gc_roots::shadow_stack_get(base + 1),
            );
        }
    }
    if positional.len() == 2 {
        let details =
            crate::baseobjspace::fixedview(pyre_object::gc_roots::shadow_stack_get(base + 2), -1)?;
        match details.len() {
            4 | 6 => {}
            5 => {
                return Err(crate::PyError::type_error(
                    "end_offset must be provided when end_lineno is provided",
                ));
            }
            n if n < 4 => {
                return Err(crate::PyError::type_error(format!(
                    "function takes at least 4 arguments ({n} given)"
                )));
            }
            n => {
                return Err(crate::PyError::type_error(format!(
                    "function takes at most 6 arguments ({n} given)"
                )));
            }
        }
        let w_self = pyre_object::gc_roots::shadow_stack_get(base);
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_syntax_filename(w_self, details[0]);
            pyre_object::interp_exceptions::w_exception_set_syntax_lineno(w_self, details[1]);
            pyre_object::interp_exceptions::w_exception_set_syntax_offset(w_self, details[2]);
            pyre_object::interp_exceptions::w_exception_set_syntax_text(w_self, details[3]);
            if details.len() == 6 {
                pyre_object::interp_exceptions::w_exception_set_syntax_end_lineno(
                    w_self, details[4],
                );
                pyre_object::interp_exceptions::w_exception_set_syntax_end_offset(
                    w_self, details[5],
                );
            } else {
                // CPython 3.14 clears both end positions when a repeated
                // `__init__` call supplies the four-field details form.
                pyre_object::interp_exceptions::w_exception_set_syntax_end_lineno(
                    w_self,
                    pyre_object::w_none(),
                );
                pyre_object::interp_exceptions::w_exception_set_syntax_end_offset(
                    w_self,
                    pyre_object::w_none(),
                );
            }
        }
    }
    let args_list = pyre_object::w_list_new(
        (0..positional.len())
            .map(|index| pyre_object::gc_roots::shadow_stack_get(base + 1 + index))
            .collect(),
    );
    unsafe {
        pyre_object::interp_exceptions::w_exception_set_args(
            pyre_object::gc_roots::shadow_stack_get(base),
            args_list,
        );
    }
    Ok(pyre_object::w_none())
}

/// Whether `w_self`'s `args_w` storage already holds exactly `positional`,
/// element for element by identity.
fn exception_args_already(w_self: PyObjectRef, positional: &[PyObjectRef]) -> bool {
    unsafe {
        let stored = pyre_object::interp_exceptions::w_exception_get_args_storage(w_self);
        if stored.is_null() || !pyre_object::is_list(stored) {
            return false;
        }
        if pyre_object::w_list_len(stored) != positional.len() {
            return false;
        }
        positional.iter().enumerate().all(|(i, &item)| {
            pyre_object::w_list_getitem(stored, i as i64)
                .is_some_and(|held| std::ptr::eq(held, item))
        })
    }
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
        (Some(c), Some(b)) if std::ptr::eq(c.as_ptr(), b)
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
pub(crate) fn os_error_errno_subclass(errno: i64) -> Option<&'static str> {
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

/// The POSIX errno for a `std::io::Error`, so the errno-specific OSError
/// subclass (`os_error_errno_subclass`) and the `.errno` slot stay consistent
/// across platforms.  On Windows `raw_os_error()` reports a Win32 error code
/// (e.g. `ERROR_ALREADY_EXISTS` 183), not a C errno, so the raw value never
/// matches the libc constants the subclass table keys on; translate the common
/// kinds to their POSIX errno through `ErrorKind` (which the standard library
/// normalises per platform), the way `winerror_to_errno` fills `OSError.errno`
/// alongside `.winerror`.  Unrecognised kinds keep the raw code, and an error
/// carrying no OS code at all falls back to `default` (the errno each call site
/// used before the translation existed).
pub(crate) fn io_error_posix_errno(e: &std::io::Error, default: i32) -> i32 {
    #[cfg(windows)]
    {
        use std::io::ErrorKind;
        let mapped = match e.kind() {
            ErrorKind::NotFound => Some(libc::ENOENT),
            ErrorKind::AlreadyExists => Some(libc::EEXIST),
            ErrorKind::PermissionDenied => Some(libc::EACCES),
            ErrorKind::ConnectionRefused => Some(libc::ECONNREFUSED),
            ErrorKind::ConnectionReset => Some(libc::ECONNRESET),
            ErrorKind::ConnectionAborted => Some(libc::ECONNABORTED),
            ErrorKind::BrokenPipe => Some(libc::EPIPE),
            ErrorKind::TimedOut => Some(libc::ETIMEDOUT),
            ErrorKind::Interrupted => Some(libc::EINTR),
            ErrorKind::WouldBlock => Some(libc::EWOULDBLOCK),
            _ => None,
        };
        if let Some(errno) = mapped {
            return errno;
        }
    }
    e.raw_os_error().unwrap_or(default)
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
    os_error_type_use_init(w_type.as_ptr())
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
        return Err(os_error_no_keywords_error(
            crate::typedef::r#type(w_self).map(|p| p.as_ptr()),
        ));
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
        .map(|p| p.as_ptr())
        .unwrap_or_else(|| crate::baseobjspace::exception_getclass(w_self));
    let w_args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(w_self) };
    let w_dict = unsafe { pyre_object::interp_exceptions::w_exception_peek_dict(w_self) };
    if !w_dict.is_null() && unsafe { pyre_object::w_dict_len(w_dict) } > 0 {
        Ok(pyre_object::w_tuple_new(vec![cls, w_args, w_dict]))
    } else {
        Ok(pyre_object::w_tuple_new(vec![cls, w_args]))
    }
}

/// `BaseException_setstate` / `ImportError_setstate` reject a non-dict state
/// before touching the instance dict; `None` is the "nothing to restore"
/// case, which both leave alone.
fn require_setstate_dict(state: PyObjectRef) -> Result<bool, crate::PyError> {
    if unsafe { pyre_object::is_none(state) } {
        return Ok(false);
    }
    if !unsafe { pyre_object::is_dict(state) } {
        return Err(crate::PyError::type_error("state is not a dictionary"));
    }
    Ok(true)
}

/// CPython 3.14 `BaseException.__setstate__` — restore every state entry
/// through `PyObject_SetAttr`.  PyPy's older `descr_setstate` updates the
/// instance dict directly, but 3.14 deliberately routes names such as
/// `args` through their descriptors for backward-compatible exception
/// pickles.
fn base_exception_setstate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__setstate__() missing 1 required positional argument: 'self'")
    })?;
    let w_state = *args.get(1).ok_or_else(|| {
        crate::PyError::type_error("__setstate__() missing 1 required positional argument: 'state'")
    })?;
    if !require_setstate_dict(w_state)? {
        return Ok(pyre_object::w_none());
    }

    // `setattr` may allocate and collect.  Root the receiver, state dict, and
    // the materialised key/value snapshot for the complete PyDict_Next loop.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[w_self, w_state]);
    let entries =
        unsafe { pyre_object::w_dict_items(pyre_object::gc_roots::shadow_stack_get(base + 1)) };
    let entry_base = pyre_object::gc_roots::shadow_stack_len();
    for &(key, value) in &entries {
        pyre_object::gc_roots::pin_roots(&[key, value]);
    }
    for index in 0..entries.len() {
        let key = pyre_object::gc_roots::shadow_stack_get(entry_base + index * 2);
        let value = pyre_object::gc_roots::shadow_stack_get(entry_base + index * 2 + 1);
        crate::baseobjspace::setattr(pyre_object::gc_roots::shadow_stack_get(base), key, value)?;
    }
    Ok(pyre_object::w_none())
}

/// CPython 3.14 `AttributeError_getstate` — copy the instance dict, then add
/// the `name` and `args` slots.  `obj` is intentionally omitted because the
/// object that triggered an attribute lookup is frequently unpicklable
/// (GH-103352).
fn attribute_error_getstate_value(w_self: PyObjectRef) -> PyObjectRef {
    use pyre_object::interp_exceptions;

    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[w_self]);
    let stored = unsafe {
        interp_exceptions::w_exception_peek_dict(pyre_object::gc_roots::shadow_stack_get(base))
    };
    let state = if stored.is_null() {
        pyre_object::w_dict_new()
    } else {
        unsafe { pyre_object::w_dict_copy(stored) }
    };
    pyre_object::gc_roots::pin_root(state);
    let state_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_name = unsafe {
        interp_exceptions::w_exception_get_name(pyre_object::gc_roots::shadow_stack_get(base))
    };
    if !w_name.is_null() {
        unsafe {
            pyre_object::w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(state_slot),
                "name",
                w_name,
            )
        };
    }
    let w_args = unsafe {
        interp_exceptions::w_exception_get_args(pyre_object::gc_roots::shadow_stack_get(base))
    };
    pyre_object::gc_roots::pin_root(w_args);
    let args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    unsafe {
        pyre_object::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(state_slot),
            "args",
            pyre_object::gc_roots::shadow_stack_get(args_slot),
        )
    };
    pyre_object::gc_roots::shadow_stack_get(state_slot)
}

fn attribute_error_getstate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__getstate__() missing 1 required positional argument: 'self'")
    })?;
    Ok(attribute_error_getstate_value(w_self))
}

/// CPython 3.14 `AttributeError_reduce` — always return the three-item form
/// so the typed `name` slot survives pickling while `obj` is discarded.
fn attribute_error_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__reduce__() missing 1 required positional argument: 'self'")
    })?;
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[w_self]);
    let cls = crate::typedef::r#type(pyre_object::gc_roots::shadow_stack_get(base))
        .map(|p| p.as_ptr())
        .unwrap_or_else(|| {
            crate::baseobjspace::exception_getclass(pyre_object::gc_roots::shadow_stack_get(base))
        });
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_args = unsafe {
        pyre_object::interp_exceptions::w_exception_get_args(
            pyre_object::gc_roots::shadow_stack_get(base),
        )
    };
    pyre_object::gc_roots::pin_root(w_args);
    let args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let state = attribute_error_getstate_value(pyre_object::gc_roots::shadow_stack_get(base));
    pyre_object::gc_roots::pin_root(state);
    let state_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    Ok(pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
        pyre_object::gc_roots::shadow_stack_get(args_slot),
        pyre_object::gc_roots::shadow_stack_get(state_slot),
    ]))
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
        .map(|p| p.as_ptr())
        .unwrap_or_else(|| crate::baseobjspace::exception_getclass(w_self));
    // `w_exception_get_args` builds a fresh tuple, and the `copy` below runs a
    // Python-level method, so both it and the receiver have to be published
    // before that call rather than sit in untraced Rust locals across it.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[w_self, cls, unsafe {
        interp_exceptions::w_exception_get_args(w_self)
    }]);
    // Read the instance dict back off the rooted receiver: the slot may be
    // null, which `pin_roots` must not be handed.
    let stored = unsafe {
        interp_exceptions::w_exception_peek_dict(pyre_object::gc_roots::shadow_stack_get(base))
    };
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
    pyre_object::gc_roots::pin_root(w_dict);
    let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    type ExcGetter = unsafe fn(PyObjectRef) -> PyObjectRef;
    for (key, get) in [
        ("name", interp_exceptions::w_exception_get_name as ExcGetter),
        ("path", interp_exceptions::w_exception_get_import_path),
        (
            "name_from",
            interp_exceptions::w_exception_get_import_name_from,
        ),
    ] {
        let w_value = unsafe { get(pyre_object::gc_roots::shadow_stack_get(base)) };
        if !w_value.is_null() && !unsafe { pyre_object::is_none(w_value) } {
            unsafe {
                pyre_object::w_dict_setitem_str(
                    pyre_object::gc_roots::shadow_stack_get(dict_slot),
                    key,
                    w_value,
                )
            };
        }
    }
    let cls = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let w_args = pyre_object::gc_roots::shadow_stack_get(base + 2);
    let w_dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
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
    if !require_setstate_dict(w_state)? {
        return Ok(pyre_object::w_none());
    }
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
        .map(|p| p.as_ptr())
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
    // CPython 3.14 retains the distinction between an omitted member (NULL)
    // and an explicitly supplied `None`; AttributeError_getstate serializes
    // the latter as `{"name": None}`.
    let w_name = kwarg_get(kwargs, "name").unwrap_or(std::ptr::null_mut());
    let w_obj = kwarg_get(kwargs, "obj").unwrap_or(std::ptr::null_mut());
    unsafe {
        // Unconditional re-stamp so a repeated `__init__` resets stale slots.
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
        crate::typedef::tag_subclass_instance(exc, w_target);
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
        // `bytes_like_data` dispatches via
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
            let inherits_bytes = bytes_type
                .is_some_and(|bt| crate::baseobjspace::isinstance_w(w_object_in, bt.as_ptr()));
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
                crate::typedef::tag_subclass_instance(exc, cls);
            }
            Ok(exc)
        }
    };
}

exc_new_wrapper!(exc_base_exception_new, exc_base_exception);
exc_new_wrapper!(exc_exception_new, exc_exception);
exc_new_wrapper!(exc_eof_error_new, exc_eof_error);
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
exc_new_wrapper!(exc_stop_async_iteration_new, exc_stop_async_iteration);
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

/// True when `method` is one of the `descr_str` / `descr_repr` builtins that
/// `make_exc_type_with_init` installs.  `py_str` / `py_repr` already run those
/// rules natively, so the native path must not dispatch back into them —
/// unlike, say, `BaseExceptionGroup.__str__`, which has no native arm and
/// whose builtin the native path does have to call.
pub(crate) unsafe fn is_native_exception_dunder(method: PyObjectRef) -> bool {
    // Every fixed-code carrier, not just `is_function`: `__str__` and
    // `__repr__` are slot names, so the namespace sweep publishes the three
    // builtins below as `wrapper_descriptor`, which `is_function` excludes.
    // Missing them here lets `exc_user_dunder_obj` call the native dunder back
    // and recurse until the stack overflows.
    if method.is_null() || !unsafe { crate::function::is_function_carrier(method) } {
        return false;
    }
    let code = crate::function::getcode(method) as PyObjectRef;
    if code.is_null() || !unsafe { crate::gateway::is_builtin_code(code) } {
        return false;
    }
    let f = unsafe { crate::gateway::builtin_code_get(code) };
    [
        base_exception_str_method as crate::gateway::BuiltinCodeFn,
        exception_str_method as crate::gateway::BuiltinCodeFn,
        exception_repr_method as crate::gateway::BuiltinCodeFn,
    ]
    .iter()
    .any(|&target| std::ptr::fn_addr_eq(f, target))
}

/// `interp_exceptions.py:993-998 W_SystemExit.descr_init` — a lone argument
/// becomes `code` verbatim, several become the args tuple, and none leaves
/// the `None` class default; `W_BaseException.descr_init` then stamps `args`.
/// It runs first here so its keyword rejection precedes the `code` write.
fn exc_system_exit_init(args: &[PyObjectRef]) -> crate::PyResult {
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__init__() missing 1 required positional argument: 'self'")
    })?;
    exc_base_exception_init(args)?;
    let (positional, _) = split_builtin_kwargs(&args[1..]);
    let code = match positional.len() {
        0 => return Ok(pyre_object::w_none()),
        1 => positional[0],
        _ => pyre_object::w_tuple_new(positional.to_vec()),
    };
    unsafe { pyre_object::interp_exceptions::w_exception_set_code(w_self, code) };
    Ok(pyre_object::w_none())
}

/// `interp_exceptions.py:126-133 W_BaseException.descr_str` — the base rule,
/// which reports the args alone.  It is what `BaseException.__str__(exc)`
/// runs even when `exc`'s own class registers a `descr_str` override.
fn base_exception_str_method(args: &[PyObjectRef]) -> crate::PyResult {
    let obj = args[0];
    Ok(pyre_object::w_str_new(&unsafe {
        crate::display::base_exception_str(obj)?
    }))
}

/// The `descr_str` of the class that registers one, falling back to
/// `W_BaseException.descr_str` for an arg shape it does not special-case
/// (`KeyError('a', 'b')`, an `OSError` with neither errno nor strerror).
fn exception_str_method(args: &[PyObjectRef]) -> crate::PyResult {
    let obj = args[0];
    let text = unsafe {
        match crate::display::exception_kind_str(obj)? {
            Some(s) => s,
            None => crate::display::base_exception_str(obj)?,
        }
    };
    Ok(pyre_object::w_str_new(&text))
}

/// `interp_exceptions.py:135-151 W_BaseException.descr_repr` — every builtin
/// exception class inherits this one, so it is registered on `BaseException`
/// alone and reads the receiver's own class name.
fn exception_repr_method(args: &[PyObjectRef]) -> crate::PyResult {
    let obj = args[0];
    Ok(pyre_object::w_str_new(&unsafe {
        crate::display::py_repr(obj)?
    }))
}

/// `interp_exceptions.py` typedef `GetSetProperty` entries, per class.
///
/// Each class declares only the attributes its own `TypeDef` adds; a
/// subclass reaches the rest through the MRO the way
/// `ModuleNotFoundError` reaches `ImportError.msg`.
fn exception_typedef_attrs(class_name: &str) -> &'static [&'static str] {
    match class_name {
        "BaseException" => &[
            "__dict__",
            "args",
            "__cause__",
            "__context__",
            "__traceback__",
        ],
        "SystemExit" => &[],
        "StopIteration" => &[],
        "OSError" => &["characters_written"],
        "ImportError" => &[],
        "NameError" => &[],
        "AttributeError" => &[],
        "SyntaxError" => &[],
        "UnicodeDecodeError" | "UnicodeEncodeError" | "UnicodeTranslateError" => &[],
        "BaseExceptionGroup" => &["exceptions", "message"],
        _ => &[],
    }
}

/// The attribute name a descriptor built by [`make_exception_getset`] carries.
fn exception_getset_name(w_descr: PyObjectRef) -> String {
    let w_name = unsafe { pyre_object::typedef::w_getset_get_name(w_descr) };
    if w_name.is_null() || !unsafe { pyre_object::is_str(w_name) } {
        return String::new();
    }
    unsafe { pyre_object::w_str_get_value(w_name) }.to_string()
}

/// A name the receiving exception kind does not declare — `OSError`'s
/// `characters_written` on anything but a `BlockingIOError`, say.  The
/// descriptor is inherited but reads back as absent.
fn exception_getset_absent(w_obj: PyObjectRef, name: &str) -> crate::PyError {
    crate::PyError::attribute_error_with_context(
        format!(
            "'{}' object has no attribute '{name}'",
            crate::baseobjspace::object_functionstr_type_name(w_obj)
        ),
        w_obj,
        name,
    )
}

fn exception_getset_fget(args: &[PyObjectRef]) -> crate::PyResult {
    let (w_descr, w_obj) = (args[0], args[1]);
    let name = exception_getset_name(w_descr);
    let found = crate::baseobjspace::exception_attr_get(w_obj, &name)?;
    if found.is_null() {
        return Err(exception_getset_absent(w_obj, &name));
    }
    Ok(found)
}

fn exception_getset_fset(args: &[PyObjectRef]) -> crate::PyResult {
    let (w_descr, w_obj, w_value) = (args[0], args[1], args[2]);
    let name = exception_getset_name(w_descr);
    let handled = crate::baseobjspace::exception_attr_set(w_obj, &name, w_value)?;
    if handled.is_null() {
        return Err(exception_getset_absent(w_obj, &name));
    }
    Ok(handled)
}

fn exception_getset_fdel(args: &[PyObjectRef]) -> crate::PyResult {
    let (w_descr, w_obj) = (args[0], args[1]);
    let name = exception_getset_name(w_descr);
    let handled = crate::baseobjspace::exception_attr_delete(w_obj, &name)?;
    if handled.is_null() {
        return Err(exception_getset_absent(w_obj, &name));
    }
    Ok(handled)
}

/// The three ends of every exception `GetSetProperty`.  One function object
/// backs all of them, so `exception_attr_slot_fold` recognises a descriptor
/// it may look through by comparing the `fget` it found against this one.
fn exception_getset_ends() -> (PyObjectRef, PyObjectRef, PyObjectRef) {
    static ENDS: std::sync::OnceLock<(usize, usize, usize)> = std::sync::OnceLock::new();
    let (fget, fset, fdel) = *ENDS.get_or_init(|| {
        (
            make_builtin_function_with_arity("__get__", exception_getset_fget, 2) as usize,
            make_builtin_function_with_arity("__set__", exception_getset_fset, 3) as usize,
            make_builtin_function_with_arity("__delete__", exception_getset_fdel, 2) as usize,
        )
    });
    (
        fget as PyObjectRef,
        fset as PyObjectRef,
        fdel as PyObjectRef,
    )
}

/// The shared `fget` every exception `GetSetProperty` carries.
pub(crate) fn exception_getset_fget_obj() -> PyObjectRef {
    exception_getset_ends().0
}

/// Install the class's `interp_exceptions.py` typedef getsets into `ns`.
fn install_exception_getsets(ns: PyObjectRef, class_name: &str) {
    let (fget, fset, fdel) = exception_getset_ends();
    for attr in exception_typedef_attrs(class_name) {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                attr,
                crate::typedef::make_getset_property_named(fget, fset, fdel, attr),
            )
        };
    }
}

/// Build a builtin exception type with the given name, base, and __new__ wrapper.
pub(crate) fn make_exc_type(
    name: &'static str,
    new_fn: crate::gateway::BuiltinCodeFn,
    base: PyObjectRef,
) -> PyObjectRef {
    make_exc_type_with_init(name, None, new_fn, None, base)
}

/// PyPy's `_new_exception` stores the supplied docstring directly in each
/// builtin exception's TypeDef namespace.
fn make_exc_type_with_doc(
    name: &'static str,
    doc: &'static str,
    new_fn: crate::gateway::BuiltinCodeFn,
    base: PyObjectRef,
) -> PyObjectRef {
    make_exc_type_with_init(name, Some(doc), new_fn, None, base)
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
    doc: Option<&'static str>,
    new_fn: crate::gateway::BuiltinCodeFn,
    init_fn: Option<crate::gateway::BuiltinCodeFn>,
    base: PyObjectRef,
) -> PyObjectRef {
    if let Some(cls) = lookup_exc_class(name) {
        return cls;
    }
    // Every exception class shares one instance layout, distinct from
    // `object`'s: `class E(Exception, ValueError)` is fine, `class E(Exception,
    // list)` is an instance lay-out conflict.  The subclasses reach the same
    // Layout object through the reuse rule (their parent layout already names
    // this typedef).
    let cls = crate::typedef::make_builtin_type_with_layout(
        name,
        move |ns| {
            unsafe {
                if let Some(doc) = doc {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__doc__",
                        pyre_object::w_str_new(doc),
                    );
                }
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__new__",
                    crate::typedef::make_new_descr(new_fn),
                )
            };
            if let Some(init_fn) = init_fn {
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__init__",
                        make_builtin_function("__init__", init_fn),
                    )
                };
            }
            // `interp_exceptions.py` declares each class's typed attributes
            // as `GetSetProperty` entries on its own `TypeDef`.
            install_exception_getsets(ns, name);
            if name == "SyntaxError" {
                for (member_name, kind, doc) in [
                    ("msg", pyre_object::MEMBER_SYNTAX_ERROR_MSG, "exception msg"),
                    (
                        "filename",
                        pyre_object::MEMBER_SYNTAX_ERROR_FILENAME,
                        "exception filename",
                    ),
                    (
                        "lineno",
                        pyre_object::MEMBER_SYNTAX_ERROR_LINENO,
                        "exception lineno",
                    ),
                    (
                        "offset",
                        pyre_object::MEMBER_SYNTAX_ERROR_OFFSET,
                        "exception offset",
                    ),
                    (
                        "text",
                        pyre_object::MEMBER_SYNTAX_ERROR_TEXT,
                        "exception text",
                    ),
                    (
                        "end_lineno",
                        pyre_object::MEMBER_SYNTAX_ERROR_END_LINENO,
                        "exception end lineno",
                    ),
                    (
                        "end_offset",
                        pyre_object::MEMBER_SYNTAX_ERROR_END_OFFSET,
                        "exception end offset",
                    ),
                    (
                        "print_file_and_line",
                        pyre_object::MEMBER_SYNTAX_ERROR_PRINT_FILE_AND_LINE,
                        "exception print_file_and_line",
                    ),
                    (
                        "_metadata",
                        pyre_object::MEMBER_SYNTAX_ERROR_METADATA,
                        "exception private metadata",
                    ),
                ] {
                    unsafe {
                        pyre_object::w_dict_setitem_str_no_proxy(
                            ns,
                            member_name,
                            pyre_object::w_member_new_direct_with_doc(
                                kind,
                                member_name.to_owned(),
                                doc.to_owned(),
                                pyre_object::PY_NULL,
                            ),
                        );
                    }
                }
            }
            if name == "StopIteration" {
                unsafe {
                    pyre_object::w_dict_setitem_str_no_proxy(
                        ns,
                        "value",
                        pyre_object::w_member_new_direct_with_doc(
                            pyre_object::MEMBER_STOP_ITERATION_VALUE,
                            "value".to_owned(),
                            "generator return value".to_owned(),
                            pyre_object::PY_NULL,
                        ),
                    );
                }
            }
            if name == "BaseException" {
                unsafe {
                    pyre_object::w_dict_setitem_str_no_proxy(
                        ns,
                        "__suppress_context__",
                        pyre_object::w_member_new_direct(
                            pyre_object::MEMBER_EXCEPTION_SUPPRESS_CONTEXT,
                            "__suppress_context__".to_owned(),
                            pyre_object::PY_NULL,
                        ),
                    );
                }
            }
            if name == "AttributeError" {
                for (member_name, kind, doc) in [
                    (
                        "name",
                        pyre_object::MEMBER_ATTRIBUTE_ERROR_NAME,
                        "attribute name",
                    ),
                    ("obj", pyre_object::MEMBER_ATTRIBUTE_ERROR_OBJ, "object"),
                ] {
                    unsafe {
                        pyre_object::w_dict_setitem_str_no_proxy(
                            ns,
                            member_name,
                            pyre_object::w_member_new_direct_with_doc(
                                kind,
                                member_name.to_owned(),
                                doc.to_owned(),
                                pyre_object::PY_NULL,
                            ),
                        );
                    }
                }
            }
            if name == "NameError" {
                unsafe {
                    pyre_object::w_dict_setitem_str_no_proxy(
                        ns,
                        "name",
                        pyre_object::w_member_new_direct_with_doc(
                            pyre_object::MEMBER_NAME_ERROR_NAME,
                            "name".to_owned(),
                            "name".to_owned(),
                            pyre_object::PY_NULL,
                        ),
                    );
                }
            }
            if name == "ImportError" {
                for (member_name, kind, doc) in [
                    (
                        "msg",
                        pyre_object::MEMBER_IMPORT_ERROR_MSG,
                        "exception message",
                    ),
                    ("name", pyre_object::MEMBER_IMPORT_ERROR_NAME, "module name"),
                    (
                        "name_from",
                        pyre_object::MEMBER_IMPORT_ERROR_NAME_FROM,
                        "name imported from module",
                    ),
                    ("path", pyre_object::MEMBER_IMPORT_ERROR_PATH, "module path"),
                ] {
                    unsafe {
                        pyre_object::w_dict_setitem_str_no_proxy(
                            ns,
                            member_name,
                            pyre_object::w_member_new_direct_with_doc(
                                kind,
                                member_name.to_owned(),
                                doc.to_owned(),
                                pyre_object::PY_NULL,
                            ),
                        );
                    }
                }
            }
            if name == "OSError" {
                for (member_name, kind, doc) in [
                    (
                        "errno",
                        pyre_object::MEMBER_OS_ERROR_ERRNO,
                        "POSIX exception code",
                    ),
                    (
                        "strerror",
                        pyre_object::MEMBER_OS_ERROR_STRERROR,
                        "exception strerror",
                    ),
                    (
                        "filename",
                        pyre_object::MEMBER_OS_ERROR_FILENAME,
                        "exception filename",
                    ),
                    (
                        "filename2",
                        pyre_object::MEMBER_OS_ERROR_FILENAME2,
                        "second exception filename",
                    ),
                ] {
                    unsafe {
                        pyre_object::w_dict_setitem_str_no_proxy(
                            ns,
                            member_name,
                            pyre_object::w_member_new_direct_with_doc(
                                kind,
                                member_name.to_owned(),
                                doc.to_owned(),
                                pyre_object::PY_NULL,
                            ),
                        );
                    }
                }
            }
            if name == "SystemExit" {
                unsafe {
                    pyre_object::w_dict_setitem_str_no_proxy(
                        ns,
                        "code",
                        pyre_object::w_member_new_direct_with_doc(
                            pyre_object::MEMBER_SYSTEM_EXIT_CODE,
                            "code".to_owned(),
                            "exception code".to_owned(),
                            pyre_object::PY_NULL,
                        ),
                    );
                }
            }
            if matches!(
                name,
                "UnicodeDecodeError" | "UnicodeEncodeError" | "UnicodeTranslateError"
            ) {
                for (member_name, kind, doc) in [
                    (
                        "encoding",
                        pyre_object::MEMBER_UNICODE_ERROR_ENCODING,
                        "exception encoding",
                    ),
                    (
                        "object",
                        pyre_object::MEMBER_UNICODE_ERROR_OBJECT,
                        "exception object",
                    ),
                    (
                        "start",
                        pyre_object::MEMBER_UNICODE_ERROR_START,
                        "exception start",
                    ),
                    (
                        "end",
                        pyre_object::MEMBER_UNICODE_ERROR_END,
                        "exception end",
                    ),
                    (
                        "reason",
                        pyre_object::MEMBER_UNICODE_ERROR_REASON,
                        "exception reason",
                    ),
                ] {
                    unsafe {
                        pyre_object::w_dict_setitem_str_no_proxy(
                            ns,
                            member_name,
                            pyre_object::w_member_new_direct_with_doc(
                                kind,
                                member_name.to_owned(),
                                doc.to_owned(),
                                pyre_object::PY_NULL,
                            ),
                        );
                    }
                }
            }
            // `interp_exceptions.py:291-292` registers `__str__` /
            // `__repr__` on `BaseException`'s typedef, and each of the
            // classes below registers a `descr_str` of its own on top.  A
            // class that inherits both stays out of this list, so
            // `LookupError.__str__` resolves up the MRO to
            // `BaseException`'s the way `KeyError.__str__` does not.
            if name == "BaseException" {
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__repr__",
                        make_builtin_function_with_arity("__repr__", exception_repr_method, 1),
                    );
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__str__",
                        make_builtin_function_with_arity("__str__", base_exception_str_method, 1),
                    );
                };
            } else if matches!(
                name,
                "KeyError"
                    | "OSError"
                    | "ImportError"
                    | "SyntaxError"
                    | "UnicodeDecodeError"
                    | "UnicodeEncodeError"
                    | "UnicodeTranslateError"
            ) {
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__str__",
                        make_builtin_function_with_arity("__str__", exception_str_method, 1),
                    )
                };
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
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                                if !w_self.is_null() && unsafe { pyre_object::is_exception(w_self) }
                                {
                                    // `interp_exceptions.py:213-219
                                    // descr_settraceback` — only None or
                                    // PyTraceback is accepted.
                                    let value = if w_tb.is_null()
                                        || unsafe { pyre_object::is_none(w_tb) }
                                    {
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
                    )
                };
                // `interp_exceptions.py:236-247 BaseException.add_note`
                // (Python 3.11+ PEP 678).  Appends a string to
                // `self.__notes__`, allocating the list on first call.
                // The list lives in the exception's instance dict
                // (`W_BaseException.w_dict`), reached through the
                // setattr/getattr paths in baseobjspace.
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                                // `interp_exceptions.py:257-260` — accept
                                // `str` and any `str` subclass
                                // (`isinstance_w(w_note, space.w_unicode)`).
                                // The rejection wording is the argument-clinic
                                // one (`_PyArg_BadArgument`), which names the
                                // method and renders `None` as `None` rather
                                // than as its type.
                                if !unsafe { crate::baseobjspace::isinstance_str_w(w_note) } {
                                    let got = if w_note == pyre_object::w_none() {
                                        "None".to_string()
                                    } else {
                                        crate::baseobjspace::object_functionstr_type_name(w_note)
                                    };
                                    return Err(crate::PyError::type_error(format!(
                                        "add_note() argument must be str, not {got}"
                                    )));
                                }
                                // `interp_exceptions.py:240-254` — lazy
                                // list allocation on first call; if the
                                // attribute is already set but NOT a list,
                                // PyPy raises TypeError("Cannot add note:
                                // __notes__ is not a list") per `:254`.
                                let existing =
                                    crate::baseobjspace::getattr_str(w_self, "__notes__")
                                        .ok()
                                        .filter(|w| !w.is_null());
                                let notes = match existing {
                                    Some(v)
                                        if unsafe { crate::baseobjspace::isinstance_list_w(v) } =>
                                    {
                                        v
                                    }
                                    Some(_) => {
                                        return Err(crate::PyError::type_error(
                                            "Cannot add note: __notes__ is not a list",
                                        ));
                                    }
                                    None => {
                                        let fresh = pyre_object::w_list_new(Vec::new());
                                        crate::baseobjspace::setattr_str(
                                            w_self,
                                            "__notes__",
                                            fresh,
                                        )?;
                                        fresh
                                    }
                                };
                                unsafe { pyre_object::w_list_append(notes, w_note) };
                                Ok(pyre_object::w_none())
                            },
                            2,
                        ),
                    )
                };
                // `interp_exceptions.py:233-241` — `descr_reduce` /
                // `descr_setstate`, installed on `BaseException` so every
                // subclass inherits them through the MRO.
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__reduce__",
                        make_builtin_function_with_arity("__reduce__", base_exception_reduce, 1),
                    )
                };
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__setstate__",
                        make_builtin_function_with_arity(
                            "__setstate__",
                            base_exception_setstate,
                            2,
                        ),
                    )
                };
            }
            // `interp_exceptions.py:379-397` — ImportError overrides reduce
            // and setstate to carry the `name`/`path`/`name_from` slots.
            // ModuleNotFoundError (built via `make_exc_type`) inherits these
            // through the MRO.
            if name == "ImportError" {
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__reduce__",
                        make_builtin_function_with_arity("__reduce__", import_error_reduce, 1),
                    )
                };
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__setstate__",
                        make_builtin_function_with_arity("__setstate__", import_error_setstate, 2),
                    )
                };
            }
            // CPython 3.14 gives AttributeError a producer-specific pickle
            // state: preserve `name` and `args`, but deliberately omit `obj`.
            if name == "AttributeError" {
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__getstate__",
                        make_builtin_function_with_arity(
                            "__getstate__",
                            attribute_error_getstate,
                            1,
                        ),
                    )
                };
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__reduce__",
                        make_builtin_function_with_arity("__reduce__", attribute_error_reduce, 1),
                    )
                };
            }
            // `interp_exceptions.py:655-665` — OSError overrides reduce to
            // re-append the filename(s); its subclasses inherit it.
            if name == "OSError" {
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                        ns,
                        "__reduce__",
                        make_builtin_function_with_arity("__reduce__", os_error_reduce, 1),
                    )
                };
            }
        },
        base,
        &pyre_object::interp_exceptions::EXCEPTION_TYPE as *const pyre_object::PyType,
    );
    if name == "SyntaxError" {
        for member_name in [
            "msg",
            "filename",
            "lineno",
            "offset",
            "text",
            "end_lineno",
            "end_offset",
            "print_file_and_line",
            "_metadata",
        ] {
            if let Some(member) = crate::type_dict_lookup(cls, member_name) {
                unsafe { pyre_object::w_member_set_cls(member, cls) };
            }
        }
    }
    if name == "StopIteration" {
        if let Some(member) = crate::type_dict_lookup(cls, "value") {
            unsafe { pyre_object::w_member_set_cls(member, cls) };
        }
    }
    if name == "BaseException" {
        if let Some(member) = crate::type_dict_lookup(cls, "__suppress_context__") {
            unsafe { pyre_object::w_member_set_cls(member, cls) };
        }
    }
    if name == "AttributeError" {
        for member_name in ["name", "obj"] {
            if let Some(member) = crate::type_dict_lookup(cls, member_name) {
                unsafe { pyre_object::w_member_set_cls(member, cls) };
            }
        }
    }
    if name == "NameError" {
        if let Some(member) = crate::type_dict_lookup(cls, "name") {
            unsafe { pyre_object::w_member_set_cls(member, cls) };
        }
    }
    if name == "ImportError" {
        for member_name in ["msg", "name", "name_from", "path"] {
            if let Some(member) = crate::type_dict_lookup(cls, member_name) {
                unsafe { pyre_object::w_member_set_cls(member, cls) };
            }
        }
    }
    if name == "OSError" {
        for member_name in ["errno", "strerror", "filename", "filename2"] {
            if let Some(member) = crate::type_dict_lookup(cls, member_name) {
                unsafe { pyre_object::w_member_set_cls(member, cls) };
            }
        }
    }
    if name == "SystemExit" {
        if let Some(member) = crate::type_dict_lookup(cls, "code") {
            unsafe { pyre_object::w_member_set_cls(member, cls) };
        }
    }
    if matches!(
        name,
        "UnicodeDecodeError" | "UnicodeEncodeError" | "UnicodeTranslateError"
    ) {
        for member_name in ["encoding", "object", "start", "end", "reason"] {
            if let Some(member) = crate::type_dict_lookup(cls, member_name) {
                unsafe { pyre_object::w_member_set_cls(member, cls) };
            }
        }
    }
    // Record the class so typedef::r#type can map a raised exception
    // back to its specific builtin class (TypeError, ValueError, ...).
    register_exc_class(name, cls)
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
    if let Some(cls) = lookup_exc_class(name) {
        return cls;
    }
    let cls = crate::typedef::make_builtin_type_with_bases(
        name,
        move |ns| {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__new__",
                    crate::typedef::make_new_descr(new_fn),
                )
            };
        },
        bases,
    );
    register_exc_class(name, cls)
}

const EG_MESSAGE_KEY: &str = "__pyre_exception_group_message";
const EG_EXCEPTIONS_KEY: &str = "__pyre_exception_group_exceptions";
const EG_EXCEPTIONS_REPR_KEY: &str = "__pyre_exception_group_exceptions_repr";

pub(crate) fn exception_group_fields(
    w_self: PyObjectRef,
) -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
    let w_dict = unsafe { pyre_object::interp_exceptions::w_exception_getdict(w_self) };
    let message = unsafe { pyre_object::w_dict_getitem_str(w_dict, EG_MESSAGE_KEY) }
        .ok_or_else(|| crate::PyError::attribute_error("exception group has no message"))?;
    let exceptions = unsafe { pyre_object::w_dict_getitem_str(w_dict, EG_EXCEPTIONS_KEY) }
        .ok_or_else(|| crate::PyError::attribute_error("exception group has no exceptions"))?;
    Ok((message, exceptions))
}

fn exception_group_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 3 {
        return Err(crate::PyError::type_error(format!(
            "BaseExceptionGroup.__new__() takes exactly 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    // `py_repr`, `fixedview` and the `isinstance`/`issubclass` checks below all
    // run Python, so the three operands cannot stay in untraced Rust locals
    // across them.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[args[0], args[1], args[2]]);
    let mut cls = pyre_object::gc_roots::shadow_stack_get(base);
    let message = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let w_exceptions = pyre_object::gc_roots::shadow_stack_get(base + 2);
    if !unsafe { crate::baseobjspace::isinstance_str_w(message) } {
        let type_name = crate::baseobjspace::object_functionstr_type_name(message);
        return Err(crate::PyError::type_error(format!(
            "argument 1 must be str, not {type_name}"
        )));
    }
    let is_list_or_tuple = unsafe {
        crate::baseobjspace::isinstance_list_w(w_exceptions) || pyre_object::is_tuple(w_exceptions)
    };
    if !is_list_or_tuple {
        let has_len = crate::baseobjspace::getattr_str(w_exceptions, "__len__").is_ok();
        let has_getitem = crate::baseobjspace::getattr_str(w_exceptions, "__getitem__").is_ok();
        if !has_len || !has_getitem {
            return Err(crate::PyError::type_error(
                "second argument (exceptions) must be a sequence",
            ));
        }
    }
    // CPython 3.14 Objects/exceptions.c BaseExceptionGroup_new (gh-141732):
    // preserve the initial representation of a non-list/tuple sequence before
    // converting it to the immutable `exceptions` tuple.  Besides keeping the
    // repr stable when the source sequence is later mutated, doing this here
    // deliberately propagates a raising or non-string custom `__repr__` from
    // the constructor.
    let repr_slot = if is_list_or_tuple {
        None
    } else {
        let rendered = pyre_object::w_str_new(&unsafe {
            crate::display::py_repr(pyre_object::gc_roots::shadow_stack_get(base + 2))?
        });
        pyre_object::gc_roots::pin_root(rendered);
        Some(pyre_object::gc_roots::shadow_stack_len() - 1)
    };
    let exceptions =
        crate::baseobjspace::fixedview(pyre_object::gc_roots::shadow_stack_get(base + 2), -1)?;
    // `fixedview` returns an untraced Vec and the membership checks below run
    // Python, so publish every element before walking them.
    let items = pyre_object::gc_roots::pin_roots(&exceptions);
    let exceptions: Vec<_> = (0..exceptions.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(items + i))
        .collect();
    if exceptions.is_empty() {
        return Err(crate::PyError::value_error(
            "second argument (exceptions) must be a non-empty sequence",
        ));
    }
    for (index, exc) in exceptions.iter().copied().enumerate() {
        if !unsafe { pyre_object::is_exception(exc) } {
            return Err(crate::PyError::value_error(format!(
                "Item {index} of second argument (exceptions) is not an exception"
            )));
        }
    }
    let base_group = lookup_exc_class("BaseExceptionGroup").unwrap();
    let exception_group = lookup_exc_class("ExceptionGroup").unwrap();
    let exception = lookup_exc_class("Exception").unwrap();
    let all_exceptions = exceptions
        .iter()
        .all(|exc| crate::baseobjspace::isinstance(*exc, exception).unwrap_or(false));
    if std::ptr::eq(cls, base_group) && all_exceptions {
        cls = exception_group;
    }
    if crate::baseobjspace::issubclass(cls, exception)? && !all_exceptions {
        let name = unsafe { pyre_object::w_type_get_name(cls) };
        let msg = if std::ptr::eq(cls, exception_group) {
            "Cannot nest BaseExceptions in an ExceptionGroup".to_string()
        } else {
            format!("Cannot nest BaseExceptions in '{name}'")
        };
        return Err(crate::PyError::type_error(msg));
    }

    let kind = if crate::baseobjspace::issubclass(cls, exception)? {
        pyre_object::interp_exceptions::ExcKind::Exception
    } else {
        pyre_object::interp_exceptions::ExcKind::BaseException
    };
    // `cls` may have been retargeted to `ExceptionGroup` above; publish
    // whichever class the instance is about to be tagged with.
    pyre_object::gc_roots::pin_root(cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // Each allocation below is a safepoint, so the nascent group and the tuple
    // it stores both go on the shadow stack before the next one runs.
    let exc = pyre_object::interp_exceptions::w_exception_new_empty(kind);
    pyre_object::gc_roots::pin_root(exc);
    let exc_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    crate::typedef::tag_subclass_instance(
        pyre_object::gc_roots::shadow_stack_get(exc_slot),
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
    );
    unsafe {
        let tuple = pyre_object::w_tuple_new(exceptions);
        pyre_object::gc_roots::pin_root(tuple);
        let tuple_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_dict = pyre_object::interp_exceptions::w_exception_getdict(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
        );
        pyre_object::gc_roots::pin_root(w_dict);
        let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        pyre_object::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            EG_MESSAGE_KEY,
            pyre_object::gc_roots::shadow_stack_get(base + 1),
        );
        pyre_object::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            EG_EXCEPTIONS_KEY,
            pyre_object::gc_roots::shadow_stack_get(tuple_slot),
        );
        if let Some(repr_slot) = repr_slot {
            pyre_object::w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
                EG_EXCEPTIONS_REPR_KEY,
                pyre_object::gc_roots::shadow_stack_get(repr_slot),
            );
        }
        let stored_args = pyre_object::w_list_new(vec![
            pyre_object::gc_roots::shadow_stack_get(base + 1),
            pyre_object::gc_roots::shadow_stack_get(base + 2),
        ]);
        pyre_object::interp_exceptions::w_exception_set_args(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
            stored_args,
        );
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(exc_slot))
}

/// `interp_group.py:26-29 W_BaseExceptionGroup.descr_init` — retain the first
/// positional value in the shared exception value slot, then delegate the
/// public `args` tuple update and keyword rejection to BaseException.
fn exception_group_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = *args.first().ok_or_else(|| {
        crate::PyError::type_error("__init__() missing 1 required positional argument: 'self'")
    })?;
    let (positional, _) = split_builtin_kwargs(&args[1..]);
    if let Some(value) = positional.first().copied() {
        unsafe { pyre_object::interp_exceptions::w_exception_set_value(w_self, value) };
    }
    exc_base_exception_init(args)
}

enum ExceptionGroupCondition {
    Class(PyObjectRef),
    Callable(PyObjectRef),
    Identity(Vec<usize>),
}

impl ExceptionGroupCondition {
    fn matches(&self, exc: PyObjectRef) -> Result<bool, crate::PyError> {
        match *self {
            // Match on the exception's type (`exception_match`), not
            // `isinstance`, so a matcher class with a custom metaclass
            // `__instancecheck__` cannot pull unrelated leaves into a subgroup.
            Self::Class(classinfo) => Ok(crate::eval::check_exc_match_against(exc, classinfo)),
            Self::Callable(callable) => {
                let result = crate::call::call_function_impl_result(callable, &[exc])?;
                crate::baseobjspace::is_true(result)
            }
            Self::Identity(ref addresses) => Ok(addresses.contains(&(exc as usize))),
        }
    }
}

fn exception_group_condition(
    w_condition: PyObjectRef,
) -> Result<ExceptionGroupCondition, crate::PyError> {
    let base_exc = lookup_exc_class("BaseException").unwrap();
    let valid_type = unsafe { pyre_object::is_type(w_condition) }
        && crate::baseobjspace::issubclass(w_condition, base_exc).unwrap_or(false);
    let valid_tuple = unsafe { pyre_object::is_tuple(w_condition) }
        && (0..unsafe { pyre_object::w_tuple_len(w_condition) }).all(|i| {
            let item = unsafe { pyre_object::w_tuple_getitem(w_condition, i as i64) }.unwrap();
            (unsafe { pyre_object::is_type(item) })
                && crate::baseobjspace::issubclass(item, base_exc).unwrap_or(false)
        });
    if valid_type || valid_tuple {
        return Ok(ExceptionGroupCondition::Class(w_condition));
    }
    if crate::baseobjspace::callable_w(w_condition) {
        return Ok(ExceptionGroupCondition::Callable(w_condition));
    }
    Err(crate::PyError::type_error(
        "expected a function, exception type or tuple of exception types",
    ))
}

fn exception_group_copy_attrs(
    source: PyObjectRef,
    target: PyObjectRef,
) -> Result<(), crate::PyError> {
    if let Ok(notes) = crate::baseobjspace::getattr_str(source, "__notes__") {
        if let Ok(items) = crate::baseobjspace::fixedview(notes, -1) {
            crate::baseobjspace::setattr_str(target, "__notes__", pyre_object::w_list_new(items))?;
        }
    }
    for name in ["__cause__", "__context__", "__traceback__"] {
        let value = crate::baseobjspace::getattr_str(source, name)?;
        crate::baseobjspace::setattr_str(target, name, value)?;
    }
    Ok(())
}

fn exception_group_derive_and_copy(
    w_self: PyObjectRef,
    exceptions: Vec<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    // _derive_and_copy_attrs: construct the sub-result through the overridable
    // `derive` method so a subclass can control reconstruction (e.g. thread
    // extra constructor args), then copy the metadata attrs onto it.
    let derive = crate::baseobjspace::getattr_str(w_self, "derive")?;
    let list = pyre_object::w_list_new(exceptions);
    let group = crate::call::call_function_impl_result(derive, &[list])?;
    let base_group = lookup_exc_class("BaseExceptionGroup").unwrap();
    if !crate::baseobjspace::isinstance(group, base_group)? {
        return Err(crate::PyError::type_error(
            "derive must return an instance of BaseExceptionGroup",
        ));
    }
    exception_group_copy_attrs(w_self, group)?;
    Ok(group)
}

fn exception_group_subgroup_inner(
    w_self: PyObjectRef,
    condition: &ExceptionGroupCondition,
) -> Result<PyObjectRef, crate::PyError> {
    if condition.matches(w_self)? {
        return Ok(w_self);
    }
    let (_, exceptions) = exception_group_fields(w_self)?;
    let base_group = lookup_exc_class("BaseExceptionGroup").unwrap();
    let mut selected = Vec::new();
    let mut modified = false;
    for exc in unsafe { pyre_object::w_tuple_items_copy_as_vec(exceptions) } {
        if crate::baseobjspace::isinstance(exc, base_group)? {
            let subgroup = exception_group_subgroup_inner(exc, condition)?;
            if !unsafe { pyre_object::is_none(subgroup) } {
                selected.push(subgroup);
            }
            if !std::ptr::eq(subgroup, exc) {
                modified = true;
            }
        } else if condition.matches(exc)? {
            selected.push(exc);
        } else {
            modified = true;
        }
    }
    if !modified {
        Ok(w_self)
    } else if selected.is_empty() {
        Ok(pyre_object::w_none())
    } else {
        exception_group_derive_and_copy(w_self, selected)
    }
}

fn exception_group_split_inner(
    w_self: PyObjectRef,
    condition: &ExceptionGroupCondition,
) -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
    if condition.matches(w_self)? {
        return Ok((w_self, pyre_object::w_none()));
    }
    let (_, exceptions) = exception_group_fields(w_self)?;
    let base_group = lookup_exc_class("BaseExceptionGroup").unwrap();
    let mut matching = Vec::new();
    let mut nonmatching = Vec::new();
    for exc in unsafe { pyre_object::w_tuple_items_copy_as_vec(exceptions) } {
        if crate::baseobjspace::isinstance(exc, base_group)? {
            let (yes, no) = exception_group_split_inner(exc, condition)?;
            if !unsafe { pyre_object::is_none(yes) } {
                matching.push(yes);
            }
            if !unsafe { pyre_object::is_none(no) } {
                nonmatching.push(no);
            }
        } else if condition.matches(exc)? {
            matching.push(exc);
        } else {
            nonmatching.push(exc);
        }
    }
    let yes = if matching.is_empty() {
        pyre_object::w_none()
    } else {
        exception_group_derive_and_copy(w_self, matching)?
    };
    let no = if nonmatching.is_empty() {
        pyre_object::w_none()
    } else {
        exception_group_derive_and_copy(w_self, nonmatching)?
    };
    Ok((yes, no))
}

pub(crate) fn exception_group_match(
    w_exc: PyObjectRef,
    w_type: PyObjectRef,
) -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
    let base_group = lookup_exc_class("BaseExceptionGroup").unwrap();
    if crate::eval::check_exc_match_against(w_exc, w_type) {
        if crate::baseobjspace::isinstance(w_exc, base_group)? {
            return Ok((w_exc, pyre_object::w_none()));
        }
        let message = unsafe { pyre_object::w_str_new("") };
        let exceptions = pyre_object::w_tuple_new(vec![w_exc]);
        let group = exception_group_new(&[base_group, message, exceptions])?;
        return Ok((group, pyre_object::w_none()));
    }
    if crate::baseobjspace::isinstance(w_exc, base_group)? {
        // Partial match: call the (overridable) `split` method and validate it
        // returns a 2-tuple of (match, rest).
        let split = crate::baseobjspace::getattr_str(w_exc, "split")?;
        let pair = crate::call::call_function_impl_result(split, &[w_type])?;
        if !unsafe { pyre_object::is_tuple(pair) } {
            let name = crate::baseobjspace::object_functionstr_type_name(pair);
            return Err(crate::PyError::type_error(format!(
                "split must return a tuple, not {name}"
            )));
        }
        let n = unsafe { pyre_object::w_tuple_len(pair) };
        if n < 2 {
            return Err(crate::PyError::type_error(format!(
                "split must return a 2-tuple, got tuple of size {n}"
            )));
        }
        // Tuples longer than 2 are accepted for backwards compatibility; only
        // the first two elements (match, rest) are used.
        let matching = unsafe { pyre_object::w_tuple_getitem(pair, 0) }.unwrap();
        let rest = unsafe { pyre_object::w_tuple_getitem(pair, 1) }.unwrap();
        return Ok((matching, rest));
    }
    Ok((pyre_object::w_none(), w_exc))
}

fn exception_group_notes(w_exc: PyObjectRef) -> Result<Option<PyObjectRef>, crate::PyError> {
    match crate::baseobjspace::getattr_str(w_exc, "__notes__") {
        Ok(notes) => Ok(Some(notes)),
        Err(err) if err.kind == crate::PyErrorKind::AttributeError => Ok(None),
        Err(err) => Err(err),
    }
}

/// Identity comparison of a `__traceback__`/`__cause__`/`__context__` slot,
/// treating an unset (raw NULL) slot and the `None` singleton as the same
/// Python-level value.  `_is_same_exception_metadata` reads these through the
/// attribute layer, which normalizes an unset slot to `None`; the raw slot
/// accessors do not, so a copy made via getattr/setattr (holding `None`) would
/// otherwise not compare equal to a source whose slot was never materialized.
fn exception_group_meta_ref_eq(w_left: PyObjectRef, w_right: PyObjectRef) -> bool {
    let left_none = w_left.is_null() || unsafe { pyre_object::is_none(w_left) };
    let right_none = w_right.is_null() || unsafe { pyre_object::is_none(w_right) };
    if left_none || right_none {
        left_none && right_none
    } else {
        std::ptr::eq(w_left, w_right)
    }
}

fn exception_group_same_metadata(
    w_left: PyObjectRef,
    w_right: PyObjectRef,
) -> Result<bool, crate::PyError> {
    let left_notes = exception_group_notes(w_left)?;
    let right_notes = exception_group_notes(w_right)?;
    if !match (left_notes, right_notes) {
        (Some(left), Some(right)) => std::ptr::eq(left, right),
        (None, None) => true,
        _ => false,
    } {
        return Ok(false);
    }
    Ok(unsafe {
        exception_group_meta_ref_eq(
            pyre_object::interp_exceptions::w_exception_get_traceback(w_left),
            pyre_object::interp_exceptions::w_exception_get_traceback(w_right),
        ) && exception_group_meta_ref_eq(
            pyre_object::interp_exceptions::w_exception_get_cause(w_left),
            pyre_object::interp_exceptions::w_exception_get_cause(w_right),
        ) && exception_group_meta_ref_eq(
            pyre_object::interp_exceptions::w_exception_get_context(w_left),
            pyre_object::interp_exceptions::w_exception_get_context(w_right),
        )
    })
}

fn exception_group_collect_leaf_addresses(
    w_exc: PyObjectRef,
    addresses: &mut Vec<usize>,
) -> Result<(), crate::PyError> {
    if unsafe { pyre_object::is_none(w_exc) } {
        return Ok(());
    }
    let base_group = lookup_exc_class("BaseExceptionGroup").unwrap();
    if crate::baseobjspace::isinstance(w_exc, base_group)? {
        let (_, exceptions) = exception_group_fields(w_exc)?;
        for child in unsafe { pyre_object::w_tuple_items_copy_as_vec(exceptions) } {
            exception_group_collect_leaf_addresses(child, addresses)?;
        }
    } else if unsafe { pyre_object::is_exception(w_exc) } {
        let address = w_exc as usize;
        if !addresses.contains(&address) {
            addresses.push(address);
        }
    } else {
        let name = crate::baseobjspace::object_functionstr_type_name(w_exc);
        return Err(crate::PyError::type_error(format!(
            "expected BaseException, got {name}"
        )));
    }
    Ok(())
}

fn exception_group_projection(
    w_group: PyObjectRef,
    keep: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let mut addresses = Vec::new();
    for w_exc in keep.iter().copied() {
        exception_group_collect_leaf_addresses(w_exc, &mut addresses)?;
    }
    let (matching, _) =
        exception_group_split_inner(w_group, &ExceptionGroupCondition::Identity(addresses))?;
    Ok(matching)
}

pub(crate) fn exception_group_prep_reraise_star(
    w_orig: PyObjectRef,
    w_exc_list: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let exceptions = crate::baseobjspace::fixedview(w_exc_list, -1)?;
    if exceptions.is_empty() {
        return Ok(pyre_object::w_none());
    }
    let base_group = lookup_exc_class("BaseExceptionGroup").unwrap();
    if !crate::baseobjspace::isinstance(w_orig, base_group)? {
        return Ok(exceptions[0]);
    }

    let mut raised = Vec::new();
    let mut reraised = Vec::new();
    for w_exc in exceptions {
        if !unsafe { pyre_object::is_none(w_exc) } {
            if exception_group_same_metadata(w_exc, w_orig)? {
                reraised.push(w_exc);
            } else {
                raised.push(w_exc);
            }
        }
    }
    let reraised_group = exception_group_projection(w_orig, &reraised)?;
    if raised.is_empty() {
        return Ok(reraised_group);
    }
    if !unsafe { pyre_object::is_none(reraised_group) } {
        raised.push(reraised_group);
    }
    if raised.len() == 1 {
        return Ok(raised[0]);
    }
    // Construct through BaseExceptionGroup so a merged result that carries a
    // bare BaseException (e.g. a reraised KeyboardInterrupt alongside a freshly
    // raised Exception) stays a BaseExceptionGroup; the constructor promotes to
    // ExceptionGroup only when every leaf is an Exception.
    let base_group = lookup_exc_class("BaseExceptionGroup").unwrap();
    let message = unsafe { pyre_object::w_str_new("") };
    let list = pyre_object::w_list_new(raised);
    exception_group_new(&[base_group, message, list])
}

fn exception_group_subgroup(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(
            "subgroup() takes exactly one argument",
        ));
    }
    let condition = exception_group_condition(args[1])?;
    exception_group_subgroup_inner(args[0], &condition)
}

fn exception_group_split(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(
            "split() takes exactly one argument",
        ));
    }
    let condition = exception_group_condition(args[1])?;
    let (yes, no) = exception_group_split_inner(args[0], &condition)?;
    Ok(pyre_object::w_tuple_new(vec![yes, no]))
}

fn exception_group_derive(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(
            "derive() takes exactly one argument",
        ));
    }
    // The default derive constructs a BaseExceptionGroup (promoted to
    // ExceptionGroup when every leaf is an Exception), NOT `type(self)`: a
    // subclass that adds constructor args must override `derive` to preserve
    // its type, otherwise split/subgroup fall back to the base class.
    let (message, _) = exception_group_fields(args[0])?;
    let base_group = lookup_exc_class("BaseExceptionGroup").unwrap();
    crate::call::call_function_impl_result(base_group, &[message, args[1]])
}

fn exception_group_str(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (message, exceptions) = exception_group_fields(args[0])?;
    let message = unsafe { pyre_object::w_str_get_wtf8(message) };
    let count = unsafe { pyre_object::w_tuple_len(exceptions) };
    let suffix = if count == 1 { "" } else { "s" };
    Ok(pyre_object::w_str_new(&format!(
        "{} ({count} sub-exception{suffix})",
        message.to_string_lossy()
    )))
}

fn exception_group_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let w_self = args[0];
    let (message, exceptions) = exception_group_fields(w_self)?;
    let cls = crate::typedef::r#type(w_self).unwrap();
    let name = unsafe { pyre_object::w_type_get_name(cls.as_ptr()) };
    let message_repr = unsafe { crate::display::py_repr(message)? };
    let w_dict = unsafe { pyre_object::interp_exceptions::w_exception_getdict(w_self) };
    let exceptions_repr = if let Some(saved) =
        unsafe { pyre_object::w_dict_getitem_str(w_dict, EG_EXCEPTIONS_REPR_KEY) }
    {
        unsafe { pyre_object::w_str_get_value(saved) }.to_string()
    } else {
        // CPython 3.14 BaseExceptionGroup_repr: render the immutable internal
        // tuple using the original list/tuple spelling.  `args` intentionally
        // still retains the caller's original sequence, but its contents must
        // not influence repr after construction.
        let w_args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(w_self) };
        let original_was_list = unsafe {
            pyre_object::is_tuple(w_args)
                && pyre_object::w_tuple_len(w_args) >= 2
                && pyre_object::is_list(
                    pyre_object::w_tuple_getitem(w_args, 1).unwrap_or(pyre_object::PY_NULL),
                )
        };
        if original_was_list {
            let items = unsafe { pyre_object::w_tuple_items_copy_as_vec(exceptions) };
            unsafe { crate::display::py_repr(pyre_object::w_list_new(items))? }
        } else {
            unsafe { crate::display::py_repr(exceptions)? }
        }
    };
    Ok(pyre_object::w_str_new(&format!(
        "{name}({message_repr}, {exceptions_repr})"
    )))
}

fn make_exception_group_type(
    name: &'static str,
    doc: Option<&'static str>,
    bases: &[PyObjectRef],
) -> PyObjectRef {
    if let Some(cls) = lookup_exc_class(name) {
        return cls;
    }
    let cls = crate::typedef::make_builtin_type_with_bases(
        name,
        move |ns| {
            if let Some(doc) = doc {
                unsafe {
                    pyre_object::w_dict_setitem_str_no_proxy(
                        ns,
                        "__doc__",
                        pyre_object::w_str_new(doc),
                    )
                };
            }
            if name == "ExceptionGroup" {
                // CPython 3.14 creates ExceptionGroup as the heap type over
                // BaseExceptionGroup + Exception.  Its own namespace therefore
                // carries `__module__ = "builtins"` and `__doc__ = None`;
                // BaseExceptionGroup is a static builtin and has no own module
                // entry.
                unsafe {
                    pyre_object::w_dict_setitem_str_no_proxy(
                        ns,
                        "__module__",
                        pyre_object::w_str_new("builtins"),
                    );
                    pyre_object::w_dict_setitem_str_no_proxy(ns, "__doc__", pyre_object::w_none());
                };
            }
            if name != "BaseExceptionGroup" {
                return;
            }
            for (method_name, function, arity) in [
                (
                    "subgroup",
                    exception_group_subgroup as crate::gateway::BuiltinCodeFn,
                    2,
                ),
                (
                    "split",
                    exception_group_split as crate::gateway::BuiltinCodeFn,
                    2,
                ),
                (
                    "derive",
                    exception_group_derive as crate::gateway::BuiltinCodeFn,
                    2,
                ),
                (
                    "__str__",
                    exception_group_str as crate::gateway::BuiltinCodeFn,
                    1,
                ),
                (
                    "__repr__",
                    exception_group_repr as crate::gateway::BuiltinCodeFn,
                    1,
                ),
            ] {
                unsafe {
                    pyre_object::w_dict_setitem_str_no_proxy(
                        ns,
                        method_name,
                        make_builtin_function_with_arity(method_name, function, arity),
                    )
                };
            }
            unsafe {
                pyre_object::w_dict_setitem_str_no_proxy(
                    ns,
                    "__new__",
                    crate::typedef::make_new_descr(exception_group_new),
                );
                pyre_object::w_dict_setitem_str_no_proxy(
                    ns,
                    "__init__",
                    make_builtin_function("__init__", exception_group_init),
                );
                pyre_object::w_dict_setitem_str_no_proxy(
                    ns,
                    "__class_getitem__",
                    pyre_object::function::w_classmethod_new(make_builtin_function(
                        "__class_getitem__",
                        crate::_pypy_generic_alias::generic_alias_class_getitem,
                    )),
                );
                for (member_name, kind, doc) in [
                    (
                        "message",
                        pyre_object::MEMBER_EXCEPTION_GROUP_MESSAGE,
                        "exception message",
                    ),
                    (
                        "exceptions",
                        pyre_object::MEMBER_EXCEPTION_GROUP_EXCEPTIONS,
                        "nested exceptions",
                    ),
                ] {
                    pyre_object::w_dict_setitem_str_no_proxy(
                        ns,
                        member_name,
                        pyre_object::w_member_new_direct_with_doc(
                            kind,
                            member_name.to_owned(),
                            doc.to_owned(),
                            pyre_object::PY_NULL,
                        ),
                    );
                }
            };
        },
        bases,
    );
    if name == "BaseExceptionGroup" {
        for member_name in ["message", "exceptions"] {
            if let Some(member) = crate::type_dict_lookup(cls, member_name) {
                unsafe { pyre_object::w_member_set_cls(member, cls) };
            }
        }
    } else if name == "ExceptionGroup" {
        // PyPy's app-level multiple-inheritance type receives the ordinary
        // heap-instance weakref slot.  pyre constructs the equivalent type
        // directly, so install the copied descriptor and flag explicitly.
        let descr = crate::typedef::copy_descriptor_for_type(crate::typedef::weakref_descr(), cls);
        crate::type_dict_store(cls, "__weakref__", descr);
        unsafe {
            pyre_object::w_type_set_weakrefable(cls, true);
            // CPython 3.14 declares ExceptionGroup as a heap type (unlike the
            // static BaseExceptionGroup); this controls mutability as well as
            // the public Py_TPFLAGS_HEAPTYPE/IMMUTABLETYPE bits.
            pyre_object::w_type_set_heaptype(cls, true);
        };
    }
    register_exc_class(name, cls)
}

/// Process-global registry from exception class name (as used by
/// `ExcKind → exc_kind_name`) to the W_TypeObject exposed in the builtins
/// namespace. Populated at init-builtins time via `make_exc_type`. Entries are
/// first-writer-wins so every execution context installs the same canonical
/// class hierarchy in its builtins dictionary.
///
/// Also propagates into `pyre_object::interp_exceptions`'s kind-indexed
/// registry so `w_exception_new(kind, ...)` populates
/// `ob_header.w_class` with the registered class — every
/// builtin-raised exception then satisfies
/// `space.type(w_exc) == registered class` per `baseobjspace.py
/// exception_getclass`.
fn register_exc_class(name: &'static str, cls: PyObjectRef) -> PyObjectRef {
    let registry =
        EXC_CLASS_REGISTRY.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut registry = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let canonical = *registry.entry(name).or_insert(cls as usize) as PyObjectRef;
    crate::typedef::stamp_exception_method_owners(canonical, name);
    if let Some(kind) = pyre_object::interp_exceptions::exc_kind_from_name(name) {
        let by_kind = pyre_object::interp_exceptions::register_exc_class_for_kind(kind, canonical);
        debug_assert_eq!(by_kind, canonical);
    }
    canonical
}

/// Look up a builtin exception class by its `ExcKind` name. Returns
/// `None` if the registry hasn't been populated yet (e.g. before
/// install_default_builtins).
///
/// `dont_look_inside` because the body reads `EXC_CLASS_REGISTRY`, a
/// runtime-populated `static` of a host type rather than a build-time constant,
/// so the front-end cannot lift it. Reaching it from a lifted body fails that
/// callee's lift and, transitively, every caller's — which is how
/// `warn::warn_category_w` lost its jitcode and made every interpreter-level
/// warning an opaque residual. Hiding the registry behind the opaque call keeps
/// the callers liftable, exactly as `w_dict_new` does for its host `IndexMap`,
/// and the scalar `Option<PyObjectRef>` result is the same residual boundary
/// shape as `baseobjspace::lookup_in_type_where_uncached`.
#[majit_macros::dont_look_inside]
pub fn lookup_exc_class(name: &str) -> Option<PyObjectRef> {
    let registry = EXC_CLASS_REGISTRY.get()?;
    let registry = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.get(name).copied().map(|cls| cls as PyObjectRef)
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

static EXC_CLASS_REGISTRY: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<&'static str, usize>>,
> = std::sync::OnceLock::new();

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
    let kw_count = kwargs
        .map(|dict| unsafe {
            pyre_object::w_dict_str_entries_wtf8(dict)
                .iter()
                .filter(|(key, _)| key.as_str() != Ok("__pyre_kw__"))
                .count()
        })
        .unwrap_or(0);
    let arg_count = pos.len() + kw_count;
    if pos.len() > 3 {
        return Err(crate::PyError::type_error(format!(
            "str expected at most 3 arguments, got {}",
            pos.len()
        )));
    }
    if arg_count > 3 {
        return Err(crate::PyError::type_error(format!(
            "str() takes at most 3 arguments ({} given)",
            arg_count
        )));
    }
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
            let tn = unsafe { pyre_object::type_name_of(w) };
            return Err(crate::PyError::type_error(format!(
                "str() argument 'encoding' must be str, not {tn}"
            )));
        }
    }
    if let Some(w) = w_errors {
        if !unsafe { is_str(w) } {
            let tn = unsafe { pyre_object::type_name_of(w) };
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
        let Some(src) = crate::typedef::buffer_as_bytes_like(obj)? else {
            let tn = unsafe { pyre_object::type_name_of(obj) };
            return Err(crate::PyError::type_error(format!(
                "decoding to str: need a bytes-like object, {tn} found"
            )));
        };
        let mut decode_args = vec![src, w_encoding.unwrap_or_else(w_none)];
        if let Some(e) = w_errors {
            decode_args.push(e);
        }
        return crate::typedef::bytes_method_decode(&decode_args);
    }
    // A tagged `int` immediate stringifies to its decimal value; format it
    // before `is_str` / `ob_type` touch it as a pointer.
    // Mirrors `py_str_wtf8` / `py_repr_obj`. Gated on `CAN_BE_TAGGED`.
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return Ok(pyre_object::w_str_new_managed(&format!(
            "{}",
            pyre_object::tagged_int::untag_int(obj)
        )));
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
                return Ok(r);
            }
            // `str(s) is s` only for an exact `str`; a subclass with no
            // `__str__` override is copied to a fresh base `str`.
            if is_exact_type(obj, &STR_TYPE) {
                return Ok(obj);
            }
            return Ok(pyre_object::w_str_from_wtf8_managed(
                pyre_object::w_str_get_wtf8(obj).to_owned(),
            ));
        }
    }
    unsafe {
        if !obj.is_null() && std::ptr::eq((*obj).ob_type, &INSTANCE_TYPE as *const PyType) {
            if let Some(r) = crate::display::try_call_dunder_obj_above_object(obj, "__str__")? {
                return Ok(r);
            }
            if let Some(r) = crate::display::try_call_dunder_obj(obj, "__repr__")? {
                return Ok(r);
            }
        }
        // `space.str` returns an app-level `__str__` result object directly;
        // a str subclass is valid and retains its Python class. The WTF-8
        // conversion below is for builtin formatting, where a fresh base str
        // is the appropriate result object.
        if !obj.is_null()
            && pyre_object::is_exception(obj)
            && let Some(r) = crate::display::exc_user_dunder_obj(obj, "__str__")?
        {
            return Ok(r);
        }
    }
    let w = unsafe { crate::py_str_wtf8(obj)? };
    Ok(pyre_object::w_str_from_wtf8_managed(w))
}

unsafe fn py_repr_obj(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
            return Ok(pyre_object::w_str_new_managed(&format!(
                "{}",
                pyre_object::tagged_int::untag_int(obj)
            )));
        }
        if !obj.is_null() {
            let tp = (*obj).ob_type;
            if let Some(r) = crate::display::builtin_subclass_dunder_obj(obj, tp, "__repr__")? {
                return Ok(r);
            }
            if let Some(r) = crate::display::type_metaclass_dunder_obj(obj, "__repr__")? {
                return Ok(r);
            }
            if std::ptr::eq(tp, &INSTANCE_TYPE as *const PyType) {
                if let Some(r) = crate::display::try_call_dunder_obj(obj, "__repr__")? {
                    return Ok(r);
                }
            }
        }
        Ok(pyre_object::w_str_from_wtf8_managed(
            crate::display::py_repr_wtf8(obj)?,
        ))
    }
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
    unsafe { py_repr_obj(args[0]) }
}

/// `unicodeobject.c:unicode_repr` post-pass — take the repr of `obj`
/// and escape every non-ASCII code point as `\xXX` / `\uXXXX` /
/// `\UXXXXXXXX`.  Shared by the `ascii()` builtin and the `!a`
/// `str.format` conversion.
fn ascii_escape_wtf8(s: &Wtf8) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.code_points() {
        let cp = ch.to_u32();
        if cp < 0x80 {
            out.push(char::from_u32(cp).unwrap());
        } else if cp <= 0xFF {
            out.push_str(&format!("\\x{cp:02x}"));
        } else if cp <= 0xFFFF {
            out.push_str(&format!("\\u{cp:04x}"));
        } else {
            out.push_str(&format!("\\U{cp:08x}"));
        }
    }
    out
}

pub(crate) fn py_ascii(obj: PyObjectRef) -> Result<String, crate::PyError> {
    let s = unsafe { crate::display::py_repr_wtf8(obj)? };
    Ok(ascii_escape_wtf8(&s))
}

fn py_ascii_obj(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let r = unsafe { py_repr_obj(obj)? };
    let r_wtf8 = unsafe { pyre_object::w_str_get_wtf8(r) };
    let out = ascii_escape_wtf8(r_wtf8);
    let changed = r_wtf8.as_str().map(|s| s != out).unwrap_or(true);
    if unsafe { is_exact_type(r, &STR_TYPE) } || changed {
        return Ok(w_str_new(&out));
    }
    let Some(tp) = (unsafe { crate::typedef::r#type(r) }) else {
        return Ok(w_str_new(&out));
    };
    let Some(new_fn) = (unsafe { crate::baseobjspace::lookup_in_type(tp.as_ptr(), "__new__") })
    else {
        return Ok(w_str_new(&out));
    };
    crate::builtins::call_and_check(new_fn, &[tp.as_ptr(), w_str_new(&out)])
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
    py_ascii_obj(args[0])
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
    // `descr_new(space, w_inttype, w_x, __posonly__, w_base=None)` has two
    // user-visible slots.  PyPy's gateway rejects surplus positionals while
    // binding that signature, before `_new_int` runs; pyre's flat builtin
    // ABI must perform the same gateway check explicitly.
    if pos.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "int expected at most 2 arguments, got {}",
            pos.len()
        )));
    }
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
        if unsafe { crate::baseobjspace::lookup(obj, "__int__") }.is_some() {
            // intobject.py:995: w_intvalue = space.int(w_value)
            let w_intvalue = crate::baseobjspace::space_int(obj)?;
            return ensure_baseint_result(w_intvalue, obj);
        }
        // Python 3.14 difference: the deprecated `__trunc__` delegation in
        // this PyPy source was removed from `int()` after Python 3.11.  Our
        // language target is 3.14, so proceed directly to `__index__`.
        // intobject.py:1015: space.lookup(w_value, '__index__')
        if unsafe { crate::baseobjspace::lookup(obj, "__index__") }.is_some() {
            // intobject.py:1016: w_obj = space.index(w_value)
            let w_obj = crate::baseobjspace::space_index(obj)?;
            return ensure_baseint_result(w_obj, obj);
        }
        // intobject.py:1047 — unicode is normalized through
        // `unicode_to_decimal_w` so non-ASCII decimal digits parse.
        if unsafe { is_str(obj) } {
            let normalized = unicode_to_decimal_w(obj).map_err(|_| invalid_int_literal(obj, 10))?;
            return parse_int_from_str(obj, &normalized, 10);
        }
        // intobject.py:1056-1070 — bytes / bytearray, then any object
        // exposing a readable buffer (`space.charbuf_w`).
        if let Some(src) = crate::typedef::buffer_as_bytes_like(obj)? {
            let data = unsafe { pyre_object::bytesobject::bytes_like_data(src) };
            let s = String::from_utf8_lossy(data);
            return parse_int_from_str(obj, &s, 10);
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
            let normalized =
                unicode_to_decimal_w(obj).map_err(|_| invalid_int_literal(obj, base))?;
            return parse_int_from_str(obj, &normalized, base);
        }
        // With an explicit base only str / bytes / bytearray are accepted.
        if pyre_object::bytesobject::is_bytes_like(obj) {
            let data = pyre_object::bytesobject::bytes_like_data(obj);
            let s = String::from_utf8_lossy(data);
            return parse_int_from_str(obj, &s, base);
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
            return Ok(pyre_object::longobject::w_long_from_raw(
                pyre_object::longobject::w_long_get_raw_value(obj),
            ));
        }
    }
    // intobject.py:1104-1107: shouldn't happen
    Err(crate::PyError::new(
        crate::PyErrorKind::RuntimeError,
        "internal error in int.__new__()".to_string(),
    ))
}

/// baseobjspace.py space.getindex_w(w_base, None)
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

/// baseobjspace.py space.getindex_w(w_obj, None)
///
/// Return w_obj.__index__() as i64. On overflow, clamp to i64::MAX
/// (w_exception=None path).
pub(crate) fn getindex_w(w_obj: PyObjectRef) -> Result<i64, crate::PyError> {
    unsafe {
        if is_int(w_obj) {
            return Ok(w_int_get_value(w_obj));
        }
        if pyre_object::pyobject::is_long(w_obj) {
            // baseobjspace.py: try int_w, on overflow clamp to
            // -sys.maxint-1 for a negative value, sys.maxint otherwise.
            let big = pyre_object::longobject::w_long_get_value(w_obj);
            return Ok(
                if pyre_object::longobject::jit_bigint_to_i64_fits(big) != 0 {
                    pyre_object::longobject::jit_bigint_to_i64_value(big)
                } else if pyre_object::longobject::jit_bigint_sign_i64(big) < 0 {
                    i64::MIN
                } else {
                    i64::MAX
                },
            );
        }
        // baseobjspace.py: w_index = self.index(w_obj)
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
                    } else if pyre_object::longobject::jit_bigint_sign_i64(big) < 0 {
                        i64::MIN
                    } else {
                        i64::MAX
                    },
                );
            }
        }
    }
    Err(crate::PyError::type_error(format!(
        "'{}' object cannot be interpreted as an integer",
        unsafe { (*(*w_obj).ob_type).name }
    )))
}

/// `unicodeobject.py unicode_to_decimal_w` — normalize a unicode string for
/// numeric parsing: a non-ASCII decimal digit (`Numeric_Type=Decimal`)
/// becomes its ASCII digit and non-ASCII whitespace becomes a space, so
/// `int("４２")` and `float("١٫٥")`-style inputs parse.  An all-ASCII string
/// is returned untouched.
fn unicode_to_decimal_w(
    w_unistr: PyObjectRef,
) -> Result<std::borrow::Cow<'static, str>, crate::PyError> {
    let utf8 = unsafe { w_str_get_wtf8(w_unistr) };
    if let Ok(s) = utf8.as_str()
        && s.is_ascii()
    {
        return Ok(std::borrow::Cow::Borrowed(s));
    }

    // unicodeobject.py `_unicode_to_decimal_w`: iterate the internal UTF-8
    // representation by code point, translate Unicode decimal digits and
    // whitespace, then encode each result as strict UTF-8.  WTF-8 exposes the
    // same surrogate-bearing representation as RPython's rutf8 iterator.
    let ucd = rustpython_unicode::Ucd::new(true);
    let mut out = String::with_capacity(utf8.len());
    for (pos, cp) in utf8.code_points().enumerate() {
        let mut c = cp;
        if c.to_u32() > 127 {
            let Some(ch) = c.to_char() else {
                return Err(crate::typedef::unicode_encode_error(
                    "utf-8",
                    w_unistr,
                    pos,
                    pos + 1,
                    "surrogates not allowed",
                ));
            };
            if ch.is_whitespace() {
                out.push(' ');
                continue;
            }
            if let Some(v) = ucd.decimal(c) {
                c = CodePoint::from((b'0' + v as u8) as char);
            }
        }
        let Some(ch) = c.to_char() else {
            return Err(crate::typedef::unicode_encode_error(
                "utf-8",
                w_unistr,
                pos,
                pos + 1,
                "surrogates not allowed",
            ));
        };
        out.push(ch);
    }
    Ok(std::borrow::Cow::Owned(out))
}

/// `wrap_parsestringerror(space, e, w_source)` — report the original Python
/// source object, not the whitespace-trimmed or Unicode-normalized buffer the
/// number parser consumes internally.
fn invalid_int_literal(w_source: PyObjectRef, base: u32) -> crate::PyError {
    let source_repr = unsafe { crate::display::py_repr(w_source) }
        .unwrap_or_else(|_| "<unprintable>".to_string());
    crate::PyError::value_error(format!(
        "invalid literal for int() with base {base}: {source_repr}"
    ))
}

/// Parse an integer from a string with the given base.
pub(crate) fn parse_int_from_str(
    w_source: PyObjectRef,
    s: &str,
    base: u32,
) -> Result<PyObjectRef, crate::PyError> {
    // rarithmetic.py:936-957 `string_to_int` first handles short, plain
    // decimal strings without constructing a NumberStringParser.
    const OVF_DIGITS: usize = 19; // len(str(sys.maxint)) on pyre's i64 target
    if base == 10 && !s.is_empty() && s.len() < OVF_DIGITS {
        let bytes = s.as_bytes();
        let (start, sign) = match bytes[0] {
            b'-' => (1, -1_i64),
            b'+' => (1, 1_i64),
            _ => (0, 1_i64),
        };
        if start != bytes.len() {
            let mut result = 0_i64;
            let mut all_decimal = true;
            for &byte in &bytes[start..] {
                if !byte.is_ascii_digit() {
                    all_decimal = false;
                    break;
                }
                result = result * 10 + (byte - b'0') as i64;
            }
            if all_decimal {
                return Ok(w_int_new(result * sign));
            }
        }
    }

    // intobject.py:955-982 `_string_to_int_or_long` constructs the one
    // NumberStringParser used by both the machine-int attempt and the
    // rbigint overflow retry.  Do not maintain a second parser here: it
    // diverges on prefix/underscore ordering and erases rbigint allocation
    // failures into an "invalid literal" error.
    let maxdigits = if base == 0 || base.is_power_of_two() {
        0
    } else {
        crate::module::sys::state::int_max_str_digits()
    };
    let mut parser =
        pyre_object::rbigint::NumberStringParser::new(s, base as i64, true, true, 0, None, 0, true)
            .map_err(|error| match error {
                pyre_object::rbigint::RBigIntError::Memory => crate::PyError::memory_error(""),
                _ => invalid_int_literal(w_source, base),
            })?;

    // NumberStringParser performs this check immediately after its structural
    // initialization.  Constructing it with a zero limit lets us retain
    // PyPy's `MaxDigitsError.digits` value in the application-level message.
    if maxdigits != 0 {
        let digit_count =
            parser.end - parser.start - s.bytes().filter(|&c| c == b'_').count() as i64;
        if digit_count > maxdigits as i64 {
            return Err(crate::PyError::value_error(format!(
                "Exceeds the limit ({maxdigits} digits) for integer string conversion: value has {digit_count} digits; use sys.set_int_max_str_digits() to increase the limit"
            )));
        }
    }

    // rarithmetic.py:962-977 accumulates a signed machine word and hands the
    // same rewound parser to rbigint only after `ovfcheck` fails.
    let parsed_base = parser.base;
    let parsed_sign = parser.sign;
    let mut machine_value = 0_i64;
    loop {
        let digit = parser.next_digit().map_err(|error| match error {
            pyre_object::rbigint::RBigIntError::Memory => crate::PyError::memory_error(""),
            _ => invalid_int_literal(w_source, base),
        })?;
        if digit < 0 {
            return Ok(w_int_new(machine_value));
        }
        let signed_digit = if parsed_sign < 0 { -digit } else { digit };
        let Some(value) = machine_value
            .checked_mul(parsed_base)
            .and_then(|value| value.checked_add(signed_digit))
        else {
            parser.rewind();
            break;
        };
        machine_value = value;
    }

    let value = BigInt::_from_numberstring_parser(&mut parser).map_err(|error| match error {
        pyre_object::rbigint::RBigIntError::Memory => crate::PyError::memory_error(""),
        pyre_object::rbigint::RBigIntError::MaxStrDigits => unreachable!("limit checked above"),
        _ => invalid_int_literal(w_source, base),
    })?;
    Ok(w_long_new(value))
}

/// The error every decimal `int` conversion raises once the result would
/// exceed `sys.get_int_max_str_digits()` digits.
pub(crate) fn int_max_str_digits_error(maxdigits: i32) -> crate::PyError {
    crate::PyError::value_error(format!(
        "Exceeds the limit ({maxdigits} digits) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit"
    ))
}

/// PyPy `W_AbstractLongObject.descr_str` / Python 3.14 integer-to-decimal
/// conversion guard. The bit-length lower bound rejects enormous values
/// before the quadratic decimal conversion; the resulting string supplies
/// the exact boundary check.
pub(crate) unsafe fn int_to_decimal_string(obj: PyObjectRef) -> Result<String, crate::PyError> {
    let owned;
    let value = if pyre_object::is_bool(obj) {
        owned = BigInt::from(pyre_object::w_bool_get_value(obj) as i64);
        &owned
    } else if pyre_object::is_int(obj) {
        owned = BigInt::from(pyre_object::w_int_get_value(obj));
        &owned
    } else {
        debug_assert!(pyre_object::is_long(obj));
        pyre_object::w_long_get_value(obj)
    };
    let maxdigits = crate::module::sys::state::int_max_str_digits();
    let too_long = int_max_str_digits_error;
    if maxdigits != 0 {
        let bits = value.bits();
        let decimal_digits_lower_bound = if bits == 0 {
            1
        } else {
            ((bits - 1).saturating_mul(30_103) / 100_000) + 1
        };
        if decimal_digits_lower_bound > maxdigits as u64 {
            return Err(too_long(maxdigits));
        }
    }
    // longobject.py:109 calls `self.asbigint().str(max_str_digits=...)`.
    // Going through Rust's Display/ToString adapter erases rbigint's
    // MaxIntError and MemoryError edges (and can turn either into a formatting
    // panic), so preserve the direct consumer contract.
    value.str(maxdigits as i64).map_err(|error| match error {
        pyre_object::rbigint::RBigIntError::MaxStrDigits => too_long(maxdigits),
        pyre_object::rbigint::RBigIntError::Memory => crate::PyError::memory_error(""),
        _ => unreachable!("rbigint.str returned an unrelated error"),
    })
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
    }
    // descroperation.py float — type-MRO __float__ then __index__
    if let Some(tp) = crate::typedef::r#type(obj) {
        if let Some(method) =
            unsafe { crate::baseobjspace::lookup_in_type(tp.as_ptr(), "__float__") }
        {
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
                    ))?;
                    return Ok(floatobject::w_float_new(w_float_get_value(result)));
                }
            }
            // descroperation.py:891 — __float__ returned non-float (type '%T')
            let result_type = unsafe { pyre_object::type_name_of(result) };
            return Err(crate::PyError::type_error(format!(
                "__float__ returned non-float (type '{result_type}')",
            )));
        }
        if let Some(method) =
            unsafe { crate::baseobjspace::lookup_in_type(tp.as_ptr(), "__index__") }
        {
            let r = crate::call::call_function_impl_result(method, &[obj])?;
            // descroperation.py:609 `space.index` returns an arbitrary-size
            // Python int.  Accept both the machine-word and W_LongObject
            // layouts, then reuse the integer float conversion so an
            // out-of-range bigint raises `OverflowError`.
            unsafe {
                if is_int(r) || is_bool(r) || pyre_object::is_long(r) {
                    return builtin_float(&[r]);
                }
            }
            let result_type = unsafe { (*(*r).ob_type).name };
            return Err(crate::PyError::type_error(format!(
                "__index__ returned non-int (type '{result_type}')",
            )));
        }
    }
    // floatobject.py:242 — only after the `__float__` / `__index__` lookup,
    // unicode (including subclasses without a numeric override) is normalized
    // through `unicode_to_decimal_w` before `_string_to_float`.
    if unsafe { is_str(obj) } {
        let s = unsafe { unicode_to_decimal_w(obj)? };
        // `float_from_string` strips PEP 515 underscore separators (between
        // digits only) before parsing the Python-literal float grammar.
        if let Some(cleaned) = strip_numeric_underscores(s.trim()) {
            if let Some(v) = rustpython_literal::float::parse_str(&cleaned) {
                return Ok(floatobject::w_float_new(v));
            }
        }
        // floatobject.py `_string_to_float`: `%R` uses the original source's
        // repr, preserving quotes and escaping controls/NUL/non-ASCII text.
        let source_repr = unsafe { crate::py_repr(obj)? };
        return Err(crate::PyError::value_error(format!(
            "could not convert string to float: {source_repr}"
        )));
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
        let name_type = unsafe { pyre_object::type_name_of(w_name) };
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
    // listobject.py:1049-1053 `_extend_from_iterable` asks for the source's
    // length hint before obtaining/consuming its iterator.  The hint is only a
    // preallocation aid, but RuntimeError and other non-TypeError failures
    // from `__len__` / `__length_hint__` are observable and must propagate.
    let _ = crate::baseobjspace::length_hint(obj, 0)?;
    // Consume iterator — PyPy: listobject.py W_ListObject(iterable)
    Ok(w_list_new(collect_iterable(obj)?))
}

/// The message-less `MemoryError` an unsatisfiable reservation raises.
pub fn reservation_failed() -> crate::PyError {
    crate::PyError::memory_error("")
}

/// An empty `Vec` with room for `count` elements, reserved fallibly.
///
/// A builtin that sizes its storage from a Python-supplied count —
/// `itertools.combinations([], sys.maxsize)`, `b'a'.rjust(sys.maxsize)` and
/// their neighbours — cannot use `Vec::with_capacity`: an oversized request
/// aborts the process instead of unwinding, so ordinary Python code
/// terminates the interpreter. PyPy answers those calls with `MemoryError`.
pub fn try_vec_with_capacity<T>(count: usize) -> Result<Vec<T>, crate::PyError> {
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| reservation_failed())?;
    Ok(out)
}

/// The `Wtf8Buf` counterpart of [`try_vec_with_capacity`], for the str
/// methods that size their buffer from a caller-supplied `width`.
pub fn try_wtf8_with_capacity(
    byte_count: usize,
) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    out.try_reserve_exact(byte_count)
        .map_err(|_| reservation_failed())?;
    Ok(out)
}

pub fn collect_iterable(obj: PyObjectRef) -> Result<Vec<PyObjectRef>, crate::PyError> {
    // `iter(obj)` itself can execute arbitrary allocating Python code.  Root
    // the input before that first collection point, matching the root that
    // RPython's shadowstack transform emits around `space.iter`.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(obj);
    let it = crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(obj_slot))?;
    collect_iterator(it)
}

/// `PySequence_Fast(seq, msg)` — materialise `seq`, replacing only the
/// `TypeError` raised while obtaining the iterator with `msg`.  A `TypeError`
/// the iterator raises later keeps its own message, and every other error
/// propagates unchanged.
pub(crate) fn sequence_fast(
    obj: PyObjectRef,
    msg: &str,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(obj);
    let it = match crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(obj_slot)) {
        Ok(it) => it,
        Err(e) if e.kind == crate::PyErrorKind::TypeError => {
            return Err(crate::PyError::type_error(msg));
        }
        Err(e) => return Err(e),
    };
    collect_iterator(it)
}

/// Consume an iterator that has already been obtained.  Kept separate from
/// [`collect_iterable`] for CPython `PySequence_Fast` parity: callers such as
/// dict sequence-pair conversion must distinguish an error from `iter(obj)`
/// from an error raised later by `next()`.
pub(crate) fn collect_iterator(it: PyObjectRef) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    // Each `next` runs arbitrary allocating code (a generator body, a JIT
    // callee that boxes a fresh int) which can trigger a moving minor
    // collection. A raw `Vec<PyObjectRef>` on the malloc heap is invisible to
    // the collector, so already-collected nursery elements would be stranded /
    // not forwarded. Pin the iterator and each yielded element onto the shadow
    // stack (a real GC root the collector walks and updates on relocation),
    // then read the forwarded slots back out — the manual equivalent of the
    // translator's shadowstack save/restore around a collecting call
    // (framework.py:853-856).
    let base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(it);
    // `pin_root` itself queries the collector, so read the iterator back from
    // its slot instead of reusing the caller's copy: a foreign collection that
    // ran inside the pin forwarded the object and rewrote the slot, not the
    // local.
    let it = pyre_object::gc_roots::shadow_stack_get(base);
    // Dict-view iterators are currently off-GC RPython-style carriers.  Their
    // registered raw-field walker covers frame/value-stack reachability, but
    // this helper keeps the iterator on the shadow stack directly; retain its
    // source dict explicitly for the duration of the native loop.
    if unsafe { pyre_object::dictmultiobject::is_dict_view_iterator(it) } {
        let w_dict = unsafe { pyre_object::dictmultiobject::w_dict_view_iterator_get_dict(it) };
        pyre_object::gc_roots::pin_root(w_dict);
    }
    let item_base = pyre_object::gc_roots::shadow_stack_len();
    let mut count = 0usize;
    loop {
        // The iterator may itself have moved during a prior `next`; reload it
        // from its (post-relocation) shadow-stack slot before each call.
        let it_now = pyre_object::gc_roots::shadow_stack_get(base);
        match crate::baseobjspace::next(it_now) {
            Ok(v) => {
                pyre_object::gc_roots::pin_root(v);
                count += 1;
            }
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    // Read the forwarded element slots back out. `item_base` follows the
    // iterator and any carrier-specific backing roots above.
    let items = (0..count)
        .map(|i| pyre_object::gc_roots::shadow_stack_get(item_base + i))
        .collect();
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
    builtin_set_add_items(set, items)?;
    Ok(set)
}

/// Add each of `items` to `set`, hashing it as it enters.
///
/// `setobject.py descr_union` adds a non-set operand's elements one by
/// one, each hashed on the way in — unlike a set operand, whose elements
/// already carry the digest they were stored under.
pub fn builtin_set_add_items(
    set: PyObjectRef,
    items: &[PyObjectRef],
) -> Result<(), crate::PyError> {
    builtin_set_add_items_impl(set, items, true)
}

/// Intersection materializes an operand through PyPy's internal `_newobj`
/// path.  Unlike the public set constructor/mutators, CPython 3.14 and PyPy
/// let the original `unhashable type` escape from this path unchanged.
pub(crate) fn builtin_set_add_items_intersection(
    set: PyObjectRef,
    items: &[PyObjectRef],
) -> Result<(), crate::PyError> {
    builtin_set_add_items_impl(set, items, false)
}

fn builtin_set_add_items_impl(
    set: PyObjectRef,
    items: &[PyObjectRef],
    wrap_hash_error: bool,
) -> Result<(), crate::PyError> {
    unsafe {
        // `try_hash_value` may run a user `__hash__` that allocates and
        // triggers a moving minor collection; `set` and every not-yet-added
        // item are rooted for the whole loop and reloaded after each hash,
        // matching `set_update_value`.
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(set);
        let item_base = sp + 1;
        for &item in items {
            pyre_object::gc_roots::pin_root(item);
        }
        let item_len = pyre_object::gc_roots::shadow_stack_len() - item_base;
        for i in 0..item_len {
            let item = pyre_object::gc_roots::shadow_stack_get(item_base + i);
            let hash = try_hash_value(item).map_err(|err| {
                if wrap_hash_error {
                    crate::baseobjspace::wrap_set_element_hash_error(
                        pyre_object::gc_roots::shadow_stack_get(item_base + i),
                        err,
                    )
                } else {
                    err
                }
            })?;
            let set = pyre_object::gc_roots::shadow_stack_get(sp);
            let item = pyre_object::gc_roots::shadow_stack_get(item_base + i);
            pyre_object::w_set_add_hashed_checked(set, item, hash)
                .map_err(crate::baseobjspace::map_set_update_error)?;
        }
        Ok(())
    }
}

/// `super()` — PyPy: descriptor.py W_Super
/// `super(cls, obj)` — PyPy: pypy/module/__builtin__/descriptor.py W_Super
///
/// Returns a proxy that looks up methods in cls's MRO starting after cls.
/// `getattr` handles the super proxy via `is_super` check.
///
/// Zero-arg super() finds __class__ and self from the calling frame.
/// CPython: Objects/typeobject.c super_init
pub(crate) fn builtin_super(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
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
    if args.len() == 1 {
        let cls = args[0];
        if !unsafe { pyre_object::is_type(cls) } {
            return Err(crate::PyError::type_error(format!(
                "super() argument 1 must be a type, not {}",
                crate::baseobjspace::object_functionstr_type_name(cls)
            )));
        }
        return Ok(pyre_object::descriptor::w_super_new(
            cls,
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        ));
    }
    if args.len() == 2 {
        let cls = args[0];
        let obj = args[1];
        if !unsafe { pyre_object::is_type(cls) } {
            return Err(crate::PyError::type_error(format!(
                "super() argument 1 must be a type, not {}",
                crate::baseobjspace::object_functionstr_type_name(cls)
            )));
        }
        // descriptor.py:28-30 — `None` for the second argument builds the
        // *unbound* super object (`super_init_impl`'s `if (obj == Py_None)
        // obj = NULL`), so `_super_check` never runs on it.
        if unsafe { pyre_object::is_none(obj) } {
            return Ok(pyre_object::descriptor::w_super_new(
                cls,
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
            ));
        }
        let obj_type = super_check(cls, obj)?;
        return Ok(pyre_object::descriptor::w_super_new(cls, obj_type, obj));
    }
    // descriptor.py `_super_from_frame`: zero-arg super() finds the first
    // argument and the `__class__` free variable in the live caller frame.
    // CURRENT_FRAME is that caller because a builtin call creates no Python
    // frame of its own.
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame_ptr = current.get();
        if frame_ptr.is_null() {
            return Err(crate::PyError::runtime_error("super(): no current frame"));
        }
        let frame = unsafe { &*frame_ptr };
        let code = frame.code();
        if code.arg_count == 0 {
            return Err(crate::PyError::runtime_error("super(): no arguments"));
        }

        // CPython 3.11+ localsplus lets an argument cellvar share slot zero;
        // this is PyPy's `_get_self_location` / `_getcell(self_cell)` branch
        // expressed in the unified slot layout.
        let self_slot = frame.locals_w()[0];
        let first_arg_is_cellvar = code
            .varnames
            .first()
            .is_some_and(|first| code.cellvars.iter().any(|cell| cell == first));
        let w_self = if first_arg_is_cellvar
            && !self_slot.is_null()
            && unsafe { pyre_object::is_cell(self_slot) }
        {
            unsafe { pyre_object::w_cell_get(self_slot) }
        } else {
            self_slot
        };
        if w_self.is_null() {
            return Err(crate::PyError::runtime_error("super(): arg[0] deleted"));
        }

        let class_freevar = code
            .freevars
            .iter()
            .position(|name| name == "__class__")
            .ok_or_else(|| crate::PyError::runtime_error("super(): __class__ cell not found"))?;
        let class_slot = code.varnames.len() + crate::pyframe::npure_cellvars(code) + class_freevar;
        let class_cell = frame
            .locals_w()
            .as_slice()
            .get(class_slot)
            .copied()
            .unwrap_or(PY_NULL);
        let w_class = if !class_cell.is_null() && unsafe { pyre_object::is_cell(class_cell) } {
            unsafe { pyre_object::w_cell_get(class_cell) }
        } else {
            class_cell
        };
        if w_class.is_null() {
            return Err(crate::PyError::runtime_error(
                "super(): empty __class__ cell",
            ));
        }

        let obj_type = super_check(w_class, w_self)?;
        Ok(pyre_object::descriptor::w_super_new(
            w_class, obj_type, w_self,
        ))
    })
}

/// `descriptor.py _super_check` — validate the explicit `(type, obj)` pair
/// and return the class whose MRO a bound super proxy walks.
pub(crate) fn super_check(
    start_type: PyObjectRef,
    obj_or_type: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        if pyre_object::is_type(obj_or_type)
            && crate::baseobjspace::issubtype_w(obj_or_type, start_type)
        {
            return Ok(obj_or_type);
        }
        if let Some(obj_type) = crate::typedef::r#type(obj_or_type) {
            if crate::baseobjspace::issubtype_w(obj_type.as_ptr(), start_type) {
                return Ok(obj_type.as_ptr());
            }
        }
        match crate::baseobjspace::getattr_str(obj_or_type, "__class__") {
            Ok(apparent_type) => {
                if pyre_object::is_type(apparent_type)
                    && crate::baseobjspace::issubtype_w(apparent_type, start_type)
                {
                    return Ok(apparent_type);
                }
            }
            // descriptor.py:139-143 — only AttributeError falls back to
            // type(obj) (the normal case already rejected it above); any
            // other exception from a `__class__` property propagates.
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
            Err(e) => return Err(e),
        }
    }
    // typeobject.c super_init: a type obj names its own class ("type str"),
    // otherwise the message names the instance's class ("instance of str").
    let obj_desc = if unsafe { pyre_object::is_type(obj_or_type) } {
        format!("type {}", unsafe {
            pyre_object::w_type_get_name(obj_or_type)
        })
    } else {
        let obj_type_name = crate::typedef::r#type(obj_or_type)
            .map(|t| unsafe { pyre_object::w_type_get_name(t.as_ptr()) })
            .unwrap_or("object");
        format!("instance of {obj_type_name}")
    };
    let start_type_name = unsafe { pyre_object::w_type_get_name(start_type) };
    Err(crate::PyError::type_error(format!(
        "super(type, obj): obj ({obj_desc}) is not an instance \
         or subtype of type ({start_type_name})."
    )))
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
            "iter expected at least 1 argument, got 0",
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
            "next expected at least 1 argument, got 0",
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
    // callable kinds (function / builtin function, bound method,
    // staticmethod, type) are dispatched through dedicated slots in `call.rs`
    // rather than a `__call__` dict entry, so each is recognised directly;
    // any other object is callable iff its type defines `__call__`.
    let is_callable = unsafe {
        crate::is_function(obj)
            || pyre_object::is_type(obj)
            || pyre_object::is_method(obj)
            || pyre_object::function::is_staticmethod(obj)
            || crate::typedef::r#type(obj)
                .and_then(|t| crate::baseobjspace::lookup_in_type(t.as_ptr(), "__call__"))
                .is_some()
    };
    Ok(w_bool_from(is_callable))
}

/// `compile(source, filename, mode, ...)` — compiling.py `compile`.
///
/// Compiles a Python string to a code object.  `flags` carries the
/// `__future__` compiler-feature bits (and the `PyCF_*` compilation
/// bits), `dont_inherit` controls whether the caller's `__future__`
/// flags are inherited, and `optimize` selects the -1/0/1/2 optimisation
/// level (assert / `__debug__` / docstring stripping).
/// Map a compiler failure to a Python `SyntaxError`, matching where
/// `compile`/`exec`/`eval`/`ast.parse` raise `SyntaxError` (not
/// `ValueError`) for malformed source.  The error's `python_location` /
/// `python_end_location` / `source_path` populate `e.lineno` / `e.offset`
/// / `e.end_lineno` / `e.end_offset` / `e.filename`.  Parser errors carry
/// the offending line sliced from `source`; compiler errors such as
/// ``return`` outside a function retain their location but expose `e.text`
/// as `None` for synthetic ``<string>`` input, matching CPython's
/// `PyErr_SyntaxLocationObject` path.  File compilation retains the decoded
/// line for error display.  A location-less codegen error (line 0) falls back
/// to a bare, message-only SyntaxError.
pub fn compile_err_to_syntax_error(
    e: crate::compile::CompileError,
    source: &str,
) -> crate::PyError {
    compile_err_to_syntax_error_maybe_incomplete(e, source, false)
}

/// RustPython `vm_new.rs:new_syntax_error_maybe_incomplete` — select the
/// private Python 3.14 `_IncompleteInputError` subclass for parser failures
/// which end at an unfinished interactive input when
/// `PyCF_ALLOW_INCOMPLETE_INPUT` is set.  PyPy's compiler has the same
/// parser-level distinction through its `compile_command` path; pyre consumes
/// RustPython's parser, so keep its exact error-shape classification here.
fn compile_err_to_syntax_error_maybe_incomplete(
    e: crate::compile::CompileError,
    source: &str,
    allow_incomplete: bool,
) -> crate::PyError {
    use rustpython_compiler::parser::{
        InterpolatedStringErrorType, LexicalErrorType, ParseErrorType,
    };

    // Parser locations are byte offsets, so index the source as bytes: a
    // non-ASCII character ahead of the quote would otherwise shift a
    // character-indexed lookup off the run being tested.  Quotes are ASCII,
    // so the byte comparison is exact.
    fn opens_triple_quote(source: &str, loc: usize) -> bool {
        let bytes = source.as_bytes();
        bytes.get(loc).is_some_and(|quote| {
            bytes.get(loc + 1) == Some(quote) && bytes.get(loc + 2) == Some(quote)
        })
    }

    let incomplete = if allow_incomplete {
        match &e {
            crate::compile::CompileError::Parse(error) => match &error.error {
                ParseErrorType::Lexical(LexicalErrorType::Eof) => true,
                _ if error.is_unclosed_bracket => true,
                ParseErrorType::Lexical(LexicalErrorType::FStringError(
                    InterpolatedStringErrorType::UnterminatedTripleQuotedString,
                )) => true,
                ParseErrorType::Lexical(LexicalErrorType::UnclosedStringError) => {
                    opens_triple_quote(source, error.raw_location.start().to_usize())
                }
                ParseErrorType::OtherError(message)
                    if message.starts_with("Expected an indented block after")
                        || message.starts_with("expected an indented block after") =>
                {
                    // Byte offsets again: the span covers the blank run that
                    // should have held the indented block.
                    let bytes = source.as_bytes();
                    let from = error
                        .raw_location
                        .start()
                        .to_usize()
                        .saturating_add(1)
                        .min(bytes.len());
                    let to = error
                        .raw_location
                        .end()
                        .to_usize()
                        .saturating_add(1)
                        .min(bytes.len());
                    from <= to && bytes[from..to].iter().all(u8::is_ascii_whitespace)
                }
                // `codeop._maybe_compile` retries a failed command with one
                // trailing newline.  The pinned RustPython parser reports the
                // first unterminated triple quote as `UnclosedStringError`,
                // but the newline retry as this `OtherError` shape.  CPython
                // keeps both attempts incomplete; preserve RustPython's
                // triple-quote test across that parser-shape variation.
                ParseErrorType::OtherError(message)
                    if message.starts_with("unterminated triple-quoted string literal") =>
                {
                    opens_triple_quote(source, error.raw_location.start().to_usize())
                }
                ParseErrorType::OtherError(_) => {
                    error.raw_location.end().to_usize() >= source.len() && !source.ends_with('\n')
                }
                _ => false,
            },
            _ => false,
        }
    } else {
        false
    };

    // `syntax_error_subclass` can supply a replacement message, so it
    // runs before the located error is built.
    let subclass = syntax_error_subclass(&e, source);
    let msg = if let Some((_, Some(replacement))) = subclass {
        replacement.to_string()
    } else if let crate::compile::CompileError::Parse(parse_err) = &e {
        // astcompiler/codegen.py:3048-3051 reports duplicate call/class
        // keywords after parsing, using CPython 3.14's lower-case wording.
        // Ruff detects the same condition in its parser, so normalize only
        // that structured error rather than rewriting arbitrary messages.
        match &parse_err.error {
            ParseErrorType::DuplicateKeywordArgumentError(name) => {
                format!("keyword argument repeated: {name}")
            }
            _ => e.to_string(),
        }
    } else {
        e.to_string()
    };
    let (lineno, offset) = e.python_location();
    let mut error = if lineno == 0 {
        crate::PyError::syntax_error(msg)
    } else {
        let (end_lineno, end_offset) = e
            .python_end_location()
            .or_else(|| codegen_statement_end(&e, source, lineno, offset))
            .unwrap_or((lineno, offset));
        let filename = e.source_path().to_string();
        // Parser errors own their source text directly, keeping the trailing
        // newline like `e.text`.  Later compiler failures omit it for the
        // synthetic ``<string>`` input, while file compilation keeps the
        // decoded source line for the top-level error printer.
        let text = if matches!(&e, crate::compile::CompileError::Parse(_)) || filename != "<string>"
        {
            source.split_inclusive('\n').nth(lineno - 1)
        } else {
            None
        };
        crate::PyError::syntax_error_located(
            msg,
            &filename,
            lineno as i64,
            offset as i64,
            end_lineno as i64,
            end_offset as i64,
            text,
        )
    };
    if incomplete {
        let w_error = error.to_exc_object();
        let w_type = lookup_exc_class("_IncompleteInputError")
            .expect("_IncompleteInputError must be installed before compile()");
        unsafe {
            (*(w_error as *mut pyre_object::PyObject)).w_class = w_type;
            return crate::PyError::from_exc_object(w_error);
        }
    }
    // An incomplete input has returned already; the indentation subclass
    // is the remaining retag.
    if let Some((name, _)) = subclass {
        error.retag_exception_class(name);
    }
    error
}

/// Which of `tokenizer.c tok_get_normal_mode`'s two indentation rejections a
/// line hits first.
enum IndentFault {
    /// `TabError`: the line's indentation measures differently with tabs
    /// expanded to 8 columns than with tabs counted as 1, so which block it
    /// belongs to depends on the tab width.
    TabsAndSpaces,
    /// `IndentationError`: a dedent that lands between two enclosing levels.
    UnmatchedDedent,
}

/// The first indentation the tokenizer would reject, and why.
///
/// `tokenizer.c tok_get_normal_mode` keeps two indent stacks — `indstack`
/// measured with tabs expanded to the next multiple of 8, `altindstack` with
/// each tab counted as 1 — and compares the incoming line against both. Equal,
/// deeper and shallower each have an `altcol` companion check, and it is the
/// disagreement between the two measures that is `TabError`; a dedent matching
/// no `indstack` entry is the plain indentation error.
///
/// A file-wide census cannot stand in for this. Tabs in one block and spaces in
/// another is legal as long as no single comparison disagrees, so counting both
/// characters anywhere in the source retags an unrelated failure.
///
/// Answers `None` for anything this cannot decide from lines alone — an
/// unterminated string, or a line still inside brackets when the source runs
/// out — because the tokenizer processes indentation only outside both, and
/// `TabError` should be claimed only on positive evidence.
fn first_indent_fault(source: &str) -> Option<IndentFault> {
    let mut indstack = vec![0usize];
    let mut altindstack = vec![0usize];
    let mut depth = 0usize;
    let mut in_triple: Option<u8> = None;
    for line in source.lines() {
        let bytes = line.as_bytes();
        // Only a logical line's first physical line carries indentation; a
        // continuation inside brackets or a triple-quoted string does not.
        let measures = (depth == 0 && in_triple.is_none()).then(|| {
            let mut col = 0usize;
            let mut altcol = 0usize;
            let mut i = 0;
            while i < bytes.len() {
                match bytes[i] {
                    b' ' => {
                        col += 1;
                        altcol += 1;
                    }
                    b'\t' => {
                        col = col / 8 * 8 + 8;
                        altcol += 1;
                    }
                    _ => break,
                }
                i += 1;
            }
            (col, altcol, i)
        });
        scan_line_nesting(bytes, &mut depth, &mut in_triple)?;
        let Some((col, altcol, indent_len)) = measures else {
            continue;
        };
        // `tok->blankline`: a blank or comment-only line is not indentation.
        match bytes.get(indent_len) {
            None | Some(b'#') | Some(b'\r') => continue,
            _ => {}
        }
        let (top, alttop) = (
            indstack[indstack.len() - 1],
            altindstack[altindstack.len() - 1],
        );
        if col == top {
            if altcol != alttop {
                return Some(IndentFault::TabsAndSpaces);
            }
        } else if col > top {
            if altcol <= alttop {
                return Some(IndentFault::TabsAndSpaces);
            }
            indstack.push(col);
            altindstack.push(altcol);
        } else {
            while indstack.len() > 1 && col < indstack[indstack.len() - 1] {
                indstack.pop();
                altindstack.pop();
            }
            if col != indstack[indstack.len() - 1] {
                return Some(IndentFault::UnmatchedDedent);
            }
            if altcol != altindstack[altindstack.len() - 1] {
                return Some(IndentFault::TabsAndSpaces);
            }
        }
    }
    None
}

/// Advance `depth` (bracket nesting) and `in_triple` (the open triple quote's
/// character) across one physical line, skipping what a quote or a `#` hides.
/// `None` when the line ends inside a single-quoted string that is not a
/// continuation, which means this scan has lost track.
fn scan_line_nesting(bytes: &[u8], depth: &mut usize, in_triple: &mut Option<u8>) -> Option<()> {
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if let Some(quote) = *in_triple {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == quote && bytes[i + 1..].starts_with(&[quote, quote]) {
                *in_triple = None;
                i += 3;
                continue;
            }
            i += 1;
            continue;
        }
        match c {
            b'#' => return Some(()),
            b'(' | b'[' | b'{' => *depth += 1,
            b')' | b']' | b'}' => *depth = depth.saturating_sub(1),
            b'"' | b'\'' => {
                if bytes[i + 1..].starts_with(&[c, c]) {
                    *in_triple = Some(c);
                    i += 3;
                    continue;
                }
                // A single-quoted string closes on this line or the source is
                // malformed in a way this scan cannot follow.
                let mut j = i + 1;
                loop {
                    match bytes.get(j) {
                        None => return None,
                        Some(b'\\') => j += 2,
                        Some(&b) if b == c => break,
                        Some(_) => j += 1,
                    }
                }
                i = j + 1;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    Some(())
}

/// The `SyntaxError` subclass a compile failure belongs to.  3.14 raises
/// `IndentationError` for every indentation-shaped tokenizer failure,
/// `TabError` when that failure comes from mixing tabs and spaces, and plain
/// `SyntaxError` for the rest.
fn syntax_error_subclass(
    e: &crate::compile::CompileError,
    source: &str,
) -> Option<(&'static str, Option<&'static str>)> {
    use rustpython_compiler::parser::{LexicalErrorType, ParseErrorType};
    let crate::compile::CompileError::Parse(parse_err) = e else {
        return None;
    };
    // Every indentation-shaped failure is one of the two the tokenizer
    // distinguishes, and only the source says which — the parser reports a
    // single column, having already collapsed the two measures.  `TabError`
    // carries the tokenizer's own message rather than the parser's, which
    // describes the shape it saw instead of the tab/space clash behind it.
    let indentation = |plain: Option<&'static str>| match first_indent_fault(source) {
        Some(IndentFault::TabsAndSpaces) => (
            "TabError",
            Some("inconsistent use of tabs and spaces in indentation"),
        ),
        _ => ("IndentationError", plain),
    };
    match &parse_err.error {
        // The parser spells this one "Unexpected indentation"; the tokenizer
        // raises `unexpected indent`.  The dedent message below already reads
        // as the tokenizer writes it, so it keeps the parser's.
        ParseErrorType::UnexpectedIndentation => Some(indentation(Some("unexpected indent"))),
        ParseErrorType::Lexical(LexicalErrorType::IndentationError) => Some(indentation(None)),
        // `pegen`'s "expected an indented block after <clause> on line N",
        // which the compiler reconstructs as a plain message.
        ParseErrorType::OtherError(msg) if msg.starts_with("expected an indented block") => {
            Some(("IndentationError", None))
        }
        _ => None,
    }
}

/// CPython attaches the complete top-level ``return`` statement range to the
/// compiler-stage "outside function" error.  RustPython's codegen error keeps
/// only its start location, although the parsed statement still has the exact
/// byte range.  Recover that range before materialising `SyntaxError`.
fn codegen_statement_end(
    error: &crate::compile::CompileError,
    source: &str,
    lineno: usize,
    offset: usize,
) -> Option<(usize, usize)> {
    let crate::compile::CompileError::Codegen(codegen) = error else {
        return None;
    };
    if !matches!(
        codegen.error,
        crate::compile::codegen::error::CodegenErrorType::InvalidReturn
    ) {
        return None;
    }
    let line_start = source
        .split_inclusive('\n')
        .take(lineno.saturating_sub(1))
        .map(str::len)
        .sum::<usize>();
    let start = line_start.checked_add(offset.saturating_sub(1))?;
    let parsed = crate::compile::parser::parse_module(source).ok()?;
    let end = parsed
        .syntax()
        .body
        .iter()
        .find(|stmt| stmt.range().start().to_usize() == start)?
        .range()
        .end()
        .to_usize();
    let end_line_start = source[..end].rfind('\n').map_or(0, |index| index + 1);
    let end_lineno = source[..end].bytes().filter(|byte| *byte == b'\n').count() + 1;
    Some((end_lineno, end - end_line_start + 1))
}

/// `pypy/interpreter/astcompiler/consts.py` compilation flag bits.
const PYCF_ONLY_AST: i64 = 0x0400;
const PYCF_DONT_IMPLY_DEDENT: i64 = 0x0200;
const PYCF_SOURCE_IS_UTF8: i64 = 0x0100;
const PYCF_IGNORE_COOKIE: i64 = 0x0800;
const PYCF_TYPE_COMMENTS: i64 = 0x4000_0000;
const PYCF_ALLOW_TOP_LEVEL_AWAIT: i64 = 0x2000;
const PYCF_ALLOW_INCOMPLETE_INPUT: i64 = 0x4000;
const PYCF_OPTIMIZED_AST: i64 = 0x8000 | PYCF_ONLY_AST;
const PYCF_ACCEPT_NULL_BYTES: i64 = 0x1000_0000;
/// `future.py` `allowed_flags` — the union of the `__future__`
/// `compiler_flag` bits (`CO_FUTURE_DIVISION` … `CO_FUTURE_ANNOTATIONS`),
/// i.e. the flags `getcodeflags` masks out of a caller's `co_flags`.
const COMPILER_FLAGS: i64 = 0x01FE_0000;

/// `pyopcode.py:source_as_str` — turn text or one readable `PyBUF_SIMPLE`
/// export into compiler input.  The export remains acquired while its bytes
/// are copied and is released before decoding can raise.
fn source_as_str(
    source: PyObjectRef,
    funcname: &str,
    what: &str,
    filename: &str,
    flags: &mut i64,
) -> Result<String, crate::PyError> {
    unsafe {
        if pyre_object::is_str(source) {
            // A text source is encoded strictly and coding-cookie detection is
            // disabled, matching PyPy's unicode branch.
            let bytes = crate::type_methods::encode_object(source, "utf-8", "strict")?;
            *flags |= PYCF_IGNORE_COOKIE;
            return Ok(String::from_utf8(bytes).expect("strict utf-8 encode yields valid utf-8"));
        }
        if pyre_object::bytesobject::is_bytes(source) {
            return crate::compile::decode_source_bytes(
                pyre_object::bytesobject::bytes_like_data(source),
                filename,
                *flags & PYCF_IGNORE_COOKIE != 0,
            );
        }
    }

    let buffer = match crate::baseobjspace::simple_buffer_bytes(source) {
        Ok(Some(buffer)) => buffer,
        Ok(None) => {
            return Err(crate::PyError::type_error(format!(
                "{funcname}() arg 1 must be a {what} object"
            )));
        }
        Err(error) if error.kind == crate::PyErrorKind::TypeError => {
            return Err(crate::PyError::type_error(format!(
                "{funcname}() arg 1 must be a {what} object"
            )));
        }
        Err(error) => return Err(error),
    };
    let bytes = buffer.as_bytes().to_vec();
    buffer.release();
    crate::compile::decode_source_bytes(&bytes, filename, *flags & PYCF_IGNORE_COOKIE != 0)
}

fn builtin_compile(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `compile(source, filename, mode, flags=0, dont_inherit=False,
    // optimize=-1, *, _feature_version=-1)`: the three required parameters and
    // flags/dont_inherit/optimize are positional-or-keyword; _feature_version
    // is keyword-only.  PyCF_ONLY_AST follows PyPy's compile_to_ast boundary.
    let (pos, kwargs) = split_builtin_kwargs(args);
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
    // Every declared slot binds before the unrecognized keywords are
    // reported, so a call missing a required argument names that argument.
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
    let filename = if unsafe { pyre_object::is_str(filename_obj) } {
        crate::baseobjspace::str_utf8_w(filename_obj)?.to_string()
    } else {
        "<string>".to_string()
    };
    let mode = if unsafe { pyre_object::is_str(mode_obj) } {
        crate::baseobjspace::str_utf8_w(mode_obj)?.to_string()
    } else {
        "exec".to_string()
    };
    // flags / dont_inherit / optimize are positional-or-keyword ints
    // (unwrap_spec flags=int, dont_inherit=int, optimize=int).
    let mut flags = match bind_pos_or_kw(pos, kwargs, 3, "flags", "compile", 4)? {
        Some(v) => crate::baseobjspace::gateway_int_w(v)?,
        None => 0,
    };
    let dont_inherit = match bind_pos_or_kw(pos, kwargs, 4, "dont_inherit", "compile", 5)? {
        Some(v) => crate::baseobjspace::gateway_int_w(v)?,
        None => 0,
    };
    let mut optimize = match bind_pos_or_kw(pos, kwargs, 5, "optimize", "compile", 6)? {
        Some(v) => crate::baseobjspace::gateway_int_w(v)?,
        None => -1,
    };

    // Any bit outside the recognised compilation-flag set is rejected.
    let recognized = COMPILER_FLAGS
        | PYCF_ONLY_AST
        | PYCF_DONT_IMPLY_DEDENT
        | PYCF_SOURCE_IS_UTF8
        | PYCF_ACCEPT_NULL_BYTES
        | PYCF_TYPE_COMMENTS
        | PYCF_ALLOW_TOP_LEVEL_AWAIT
        | PYCF_ALLOW_INCOMPLETE_INPUT
        | PYCF_OPTIMIZED_AST
        | PYCF_IGNORE_COOKIE;
    if flags & !recognized != 0 {
        return Err(crate::PyError::value_error("compile(): unrecognised flags"));
    }

    // dont_inherit=0 folds in the caller's __future__ flags
    // (getcodeflags: `co_flags & compiler_flags`).
    if dont_inherit == 0 {
        let caller_frame = crate::eval::CURRENT_FRAME.with(|current| current.get());
        if !caller_frame.is_null() {
            let ec = unsafe { (*caller_frame).execution_context };
            if !ec.is_null() {
                let top = unsafe { (*ec).gettopframe_nohidden() };
                if !top.is_null() {
                    let caller_flags = unsafe { (*top).getcode().flags.bits() } as i64;
                    flags |= caller_flags & COMPILER_FLAGS;
                }
            }
        }
    }

    let mode = match mode.as_str() {
        "exec" => crate::compile::Mode::Exec,
        "eval" => crate::compile::Mode::Eval,
        "single" => crate::compile::Mode::Single,
        _ => {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "compile() mode must be 'exec', 'eval' or 'single'",
            ));
        }
    };

    if !(-1..=2).contains(&optimize) {
        return Err(crate::PyError::value_error(
            "compile(): invalid optimize value",
        ));
    }
    if optimize == -1 {
        // pycompiler.py `compile_ast`: -1 resolves to sys.flags.optimize.
        optimize = i64::from(crate::importing::optimize_flag());
    }

    // PyPy checks the public AST boundary before source_as_str so AST
    // subclasses are compiled as trees rather than probed for a buffer.
    let ast_module = crate::importing::importhook(
        "_ast",
        PY_NULL,
        PY_NULL,
        0,
        crate::call::take_last_exec_ctx(),
    )?;
    let ast_type = crate::baseobjspace::getattr_str(ast_module, "AST")?;
    let source_is_ast = unsafe { crate::baseobjspace::isinstance_w(source, ast_type) };
    let source_str = if source_is_ast {
        None
    } else {
        Some(source_as_str(
            source,
            "compile",
            "string, bytes or AST",
            &filename,
            &mut flags,
        )?)
    };

    // Assemble CompileOpts: the __future__ feature bits, the two PyCF_*
    // bits the codegen honours, and the optimisation level.
    let opts = crate::compile::CompileOpts {
        optimize: optimize as u8,
        allow_top_level_await: flags & PYCF_ALLOW_TOP_LEVEL_AWAIT != 0,
        dont_imply_dedent: flags & PYCF_DONT_IMPLY_DEDENT != 0,
        future_features: crate::CodeFlags::from_bits_truncate((flags & COMPILER_FLAGS) as u32),
        ..Default::default()
    };
    if flags & PYCF_ONLY_AST != 0 {
        // CPython 3.14 bltinmodule.c:847 / pythonrun.c:1524:
        // plain ONLY_AST runs syntax-only preprocessing; OPTIMIZED_AST
        // includes ONLY_AST and enables constant folding.
        let syntax_check_only = flags & PYCF_OPTIMIZED_AST == PYCF_ONLY_AST;
        return if source_is_ast {
            // An already materialised AST under plain `PyCF_ONLY_AST` is
            // handed straight back, so `compile(tree, ..., PyCF_ONLY_AST) is
            // tree`.  Re-converting it would hand out a copy instead.
            if syntax_check_only {
                Ok(source)
            } else {
                crate::module::_ast::convert::preprocess_object_to_object(
                    source,
                    "",
                    mode,
                    opts,
                    syntax_check_only,
                )
            }
        } else {
            crate::module::_ast::convert::parse_to_object_with_opts(
                source_str
                    .as_deref()
                    .expect("non-AST source has decoded text"),
                mode,
                opts,
                syntax_check_only,
            )
        };
    }
    if source_str.is_none() {
        let code = crate::module::_ast::convert::compile_object(source, &filename, mode, opts)?;
        let code_ptr = Box::into_raw(Box::new(code)) as *const ();
        return Ok(crate::w_code_new(code_ptr));
    }
    let source = source_str.as_deref().expect("non-AST source");
    let code =
        crate::compile::compile_source_with_opts(source, mode, &filename, opts).map_err(|e| {
            compile_err_to_syntax_error_maybe_incomplete(
                e,
                source,
                flags & PYCF_ALLOW_INCOMPLETE_INPUT != 0,
            )
        })?;
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
pub(crate) fn builtin_exec(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `exec(source, /, globals=None, locals=None, *, closure=None)`: source is
    // positional-only; globals/locals are positional-or-keyword; `closure` is
    // keyword-only.  `closure` supplies the cell objects that bind a code
    // object's free variables (bltinmodule.c builtin_exec_impl); a None
    // closure normalises to "absent" (PY_NULL).
    let (pos, kwargs) = split_builtin_kwargs(args);
    // The positional-only `source` is bound before the unrecognized keywords
    // are reported, so a keywords-only call is reported against `source`.
    if pos.is_empty() {
        return Err(crate::PyError::type_error(
            "exec() takes at least 1 positional argument (0 given)",
        ));
    }
    kwarg_reject_unknown(kwargs, &["globals", "locals", "closure"], "exec")?;
    let source = pos[0];
    let globals_arg =
        bind_pos_or_kw(pos, kwargs, 1, "globals", "exec", 2)?.unwrap_or(pyre_object::PY_NULL);
    let locals_arg =
        bind_pos_or_kw(pos, kwargs, 2, "locals", "exec", 3)?.unwrap_or(pyre_object::PY_NULL);
    // `if closure is None: closure = NULL` — treat None as unset.
    let closure = match kwarg_get(kwargs, "closure") {
        Some(c) if !unsafe { pyre_object::is_none(c) } => c,
        _ => pyre_object::PY_NULL,
    };
    exec_or_eval(source, globals_arg, locals_arg, false, closure)
}

/// `eval(source_or_code, globals=None, locals=None)` — same plumbing as
/// exec but returns the value of the expression.
fn builtin_eval(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `eval(source, /, globals=None, locals=None)`: source is positional-only;
    // globals/locals are positional-or-keyword.
    let (pos, kwargs) = split_builtin_kwargs(args);
    // The positional-only `source` is bound before the unrecognized keywords
    // are reported, so a keywords-only call is reported against `source`.
    if pos.is_empty() {
        return Err(crate::PyError::type_error(
            "eval() takes at least 1 positional argument (0 given)",
        ));
    }
    kwarg_reject_unknown(kwargs, &["globals", "locals"], "eval")?;
    let source = pos[0];
    let globals_arg =
        bind_pos_or_kw(pos, kwargs, 1, "globals", "eval", 2)?.unwrap_or(pyre_object::PY_NULL);
    let locals_arg =
        bind_pos_or_kw(pos, kwargs, 2, "locals", "eval", 3)?.unwrap_or(pyre_object::PY_NULL);
    // eval() has no closure parameter — pass "absent".
    exec_or_eval(source, globals_arg, locals_arg, true, pyre_object::PY_NULL)
}

fn exec_or_eval(
    source: PyObjectRef,
    globals_arg: PyObjectRef,
    locals_arg: PyObjectRef,
    is_eval: bool,
    closure: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    // Resolve a runnable code object: accept a precompiled W_Code or
    // compile a str on the fly.  `source_is_code` records whether the
    // original argument was already a code object (vs a compiled str /
    // bytes) — the closure validation below branches on it.
    let (code_obj_ref, source_is_code) = unsafe {
        if !source.is_null() && crate::is_code(source) {
            (source, true)
        } else {
            let mut flags = PYCF_SOURCE_IS_UTF8;
            let mut source = source_as_str(
                source,
                if is_eval { "eval" } else { "exec" },
                "string, bytes or code",
                "<string>",
                &mut flags,
            )?;
            if is_eval {
                // compiling.py:eval compiles `source.lstrip(' \t')`.
                source = source.trim_start_matches([' ', '\t']).to_owned();
            }
            let mode = if is_eval {
                crate::compile::Mode::Eval
            } else {
                crate::compile::Mode::Exec
            };
            let code = crate::compile::compile_source(&source, mode)
                .map_err(|e| compile_err_to_syntax_error(e, &source))?;
            let code_ptr = Box::into_raw(Box::new(code)) as *const ();
            (crate::w_code_new(code_ptr), false)
        }
    };
    let raw_code = unsafe {
        crate::w_code_get_ptr(code_obj_ref as pyre_object::PyObjectRef) as *const crate::CodeObject
    };

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
        // A tagged int immediate is an exact builtin int; skip the ob_type deref.
        if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(w_obj) {
            return "int".to_string();
        }
        unsafe {
            match crate::typedef::r#type(w_obj) {
                Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
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

    /// `PyEval_GetBuiltins()` — the value `exec`/`eval` plant under
    /// `__builtins__` in a namespace that lacks it.  PyPy stores the picked
    /// `Module` (`compiling.py:110 space.builtin`); 3.14 stores that module's
    /// *dict*, which is what `isinstance(__builtins__, dict)` inside
    /// `exec(src, {})` observes.
    fn planted_builtins(w_builtin: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
        if !w_builtin.is_null() && unsafe { pyre_object::is_module(w_builtin) } {
            return unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        }
        w_builtin
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
            planted_builtins(unsafe { (*exec_ctx).get_builtin() })
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
            planted_builtins(unsafe { (*caller_frame).get_builtin() })
        } else if !exec_ctx.is_null() {
            planted_builtins(unsafe { (*exec_ctx).get_builtin() })
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
    if !is_none_or_null(globals_arg) && !is_dict_w(globals_arg) {
        // `builtin_eval_impl` splits on `PyMapping_Check`, so a mapping that
        // is merely not a dict is told how to pass it as the locals instead;
        // `builtin_exec_impl` names the type it got.
        let message = if is_eval {
            if unsafe { crate::baseobjspace::lookup(globals_arg, "__getitem__").is_some() } {
                "globals must be a real dict; try eval(expr, {}, mapping)".to_string()
            } else {
                "globals must be a dict".to_string()
            }
        } else {
            format!(
                "exec() globals must be a dict, not {}",
                type_name_of(globals_arg)
            )
        };
        return Err(crate::PyError::type_error(message));
    }
    if !is_none_or_null(locals_arg)
        && unsafe { crate::baseobjspace::lookup(locals_arg, "__getitem__").is_none() }
    {
        let message = if is_eval {
            "locals must be a mapping".to_string()
        } else {
            format!(
                "locals must be a mapping or None, not {}",
                type_name_of(locals_arg)
            )
        };
        return Err(crate::PyError::type_error(message));
    }

    // bltinmodule.c builtin_exec_impl — validate the closure against the
    // resolved code object (after the globals/locals type checks), then build
    // an `outer_func` carrying the cells so `initialize_frame_scopes` binds
    // them as the code's free variables (the same path a nested function call
    // takes via `function.__closure__`).  `closure` is already normalised so
    // PY_NULL means "absent".
    //
    // eval() takes no closure parameter; the eval-side error for a code
    // object that carries free variables comes from initialize_frame_scopes
    // (pyframe.py:242-246 "directly executed code object may not contain free
    // variables") when createframe runs below with no outer_func.
    // `inject_closure` records that a validated closure must be bound into the
    // frame; the `outer_func` carrier is built just before createframe so it
    // needs no GC rooting across the namespace-setup allocations below.
    let mut inject_closure = false;
    if !is_eval {
        if source_is_code {
            let num_free = unsafe { (&*raw_code).freevars.len() };
            if num_free == 0 {
                // A code object without free variables accepts no closure.
                if !closure.is_null() {
                    return Err(crate::PyError::type_error(
                        "cannot use a closure with this code object",
                    ));
                }
            } else {
                // The closure must be an exact tuple of `num_free` cells.
                let closure_ok = !closure.is_null()
                    && unsafe { pyre_object::is_tuple(closure) }
                    && unsafe { pyre_object::w_tuple_len(closure) } == num_free
                    && (0..num_free).all(|i| unsafe {
                        pyre_object::w_tuple_getitem(closure, i as i64)
                            .is_some_and(|cell| pyre_object::is_cell(cell))
                    });
                if !closure_ok {
                    return Err(crate::PyError::type_error(format!(
                        "code object requires a closure of exactly length {num_free}"
                    )));
                }
                inject_closure = true;
            }
        } else if !closure.is_null() {
            // A str/bytes source is compiled fresh (no free variables), so a
            // closure is meaningless here.
            return Err(crate::PyError::type_error(
                "closure can only be used when source is a code object",
            ));
        }
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
    // pyopcode.py:773-774 `space.call_method(w_globals, 'setdefault', ...)`
    // (exec) and compiling.py:109-110 `space.setitem_str(w_globals, ...)`
    // (eval) dispatch on the ORIGINAL `w_globals` object so a dict-subclass
    // `setdefault` / `__contains__` / `__setitem__` override fires.
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
        // The caller's locals as `locals()` reports them: its real namespace
        // for a module or class frame, an independent snapshot for an
        // optimized one.  The snapshot is what keeps `exec("y = 1")` inside a
        // function from adding `y` to that function's locals.
        implicit_caller_locals = unsafe { (*caller_frame).frame_locals_snapshot()? };
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
    // function.py Function.__init__ — build the closure carrier for
    // `exec(code, ..., closure=...)`.  A `Function` whose `__closure__` is the
    // validated cell tuple; createframe reads it back through
    // `function_get_closure` and injects each cell into the frame's freevar
    // slots (the same path a nested function call takes).  Its globals are
    // never read (initialize_frame_scopes only consults the closure), so a
    // PY_NULL globals carrier suffices.  Built here — after the namespace
    // setup allocations — and pinned so the intervening createframe allocation
    // cannot reclaim it before its cells are copied into the frame.
    let _closure_root = pyre_object::gc_roots::push_roots();
    let outer_func = if inject_closure {
        let name = unsafe { (&*raw_code).obj_name.clone() };
        let f = crate::function::function_new_with_closure(
            code_obj_ref as *const (),
            name,
            pyre_object::PY_NULL,
            closure,
        );
        pyre_object::gc_roots::pin_root(f);
        Some(f)
    } else {
        None
    };
    // eval.py:31-33 Code.exec_code → space.createframe(...) + frame.run().
    // For eval() with a code object that carries freevars, `outer_func` is
    // None so createframe surfaces pyframe.py:242-246's TypeError "directly
    // executed code object may not contain free variables" — exec()'s
    // closure-mismatch TypeError was already raised above.
    let mut frame =
        match crate::createframe_obj(code_obj_ref as *const (), w_globals, exec_ctx, outer_func) {
            Ok(frame) => frame,
            Err(err) => {
                let _ = raw_code;
                return Err(err);
            }
        };
    // Python 3.14 always selects the builtin namespace supplied through an
    // explicit exec/eval globals dict.  PyPy expresses the same selection as
    // `space.builtin.pick_builtin(w_globals)` when
    // `objspace.honor__builtins__` is enabled.  Keep Pyre's global PyPy
    // configuration unchanged, but apply that exact selection to the frame
    // constructed by this exec/eval path.
    //
    // `moduledef.py:95-100` has a common-case identity exit when the globals
    // entry is already `space.builtin`.  Preserve that shape as an
    // allocation-free exact-dict probe: imported modules overwhelmingly carry
    // this entry, and must not enter the general mapping dispatch again.
    let current_globals = frame.get_w_globals();
    let keeps_existing_builtin = unsafe {
        pyre_object::is_dict(current_globals)
            && pyre_object::w_dict_getitem_str(current_globals, "__builtins__")
                .is_some_and(|w_builtin| std::ptr::eq(w_builtin, frame.w_builtin))
    };
    if !keeps_existing_builtin {
        // `pick_builtin_obj_checked` can invoke a dict-subclass `__getitem__`
        // and therefore allocate or collect.  The new frame is not current
        // yet, so expose it as a temporary GC root while selecting and storing
        // its builtin Module.
        let _frame_roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(frame.as_mut_ptr() as pyre_object::PyObjectRef);
        let current_globals = frame.get_w_globals();
        frame.w_builtin = crate::baseobjspace::pick_builtin_obj_checked(current_globals, exec_ctx)?;
    }
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
        // identity. Pyre returns the frame's globals dict object directly.
        // Returning a fresh wrapper per call would silently
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

/// The frame whose locals `locals()` / `vars()` / `dir()` report on
/// (`interp_inspect.py:7-11 locals` — `ec.gettopframe_nohidden()`).
///
/// Going through `gettopframe_nohidden` is what makes the frame's fastlocals
/// readable: it runs `force_frame` on every frame it walks
/// (`executioncontext.rs:409-421`), and `fast2locals` reads
/// `locals_cells_stack_w` directly, so an unforced virtualizable hands back an
/// array of nulls — which `fast2locals` renders as an EMPTY mapping rather
/// than a stale one.  Reading `CURRENT_FRAME` instead skips that force, and
/// under the JIT the caller then sees no locals at all.
fn topframe_for_locals() -> *mut crate::PyFrame {
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if ec.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*ec).gettopframe_nohidden() }
}

fn builtin_locals(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if !args.is_empty() {
        return Err(crate::PyError::type_error("locals() takes no arguments"));
    }
    // `frame_locals_snapshot` (`_PyEval_GetFrameLocals`) hands an optimized
    // frame an independent copy, so a snapshot neither tracks later stores nor
    // writes back — `x = 1; d = locals(); x = 2; d["x"]` still reads `1`, and
    // `locals() is locals()` is false.  Module and class frames keep returning
    // their real namespace, so `locals() is globals()` still holds at module
    // scope and a non-dict exec/eval mapping is still the live object written
    // through its `__setitem__`.
    let frame = topframe_for_locals();
    if frame.is_null() {
        return Err(crate::PyError::runtime_error(
            "locals() requires an active frame",
        ));
    }
    let frame_mut = unsafe { &mut *frame };
    frame_mut.frame_locals_snapshot()
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
        return Err(crate::PyError::type_error(format!(
            "vars expected at most 1 argument, got {}",
            args.len()
        )));
    }
    // app_inspect.py:21-24 — the attribute lookup itself decides whether the
    // argument qualifies, so anything carrying a `__dict__` descriptor answers,
    // including exceptions.  Only AttributeError becomes the TypeError.
    let dict = match crate::baseobjspace::getattr_str(args[0], "__dict__") {
        Ok(dict) => dict,
        Err(error) if error.kind == crate::PyErrorKind::AttributeError => {
            return Err(crate::PyError::type_error(
                "vars() argument must have __dict__ attribute",
            ));
        }
        Err(error) => return Err(error),
    };
    if dict.is_null() {
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

/// util.py:80 `_objectdir` / objectobject.py:324 `descr__dir__`.
///
/// Return the generic object's own dict keys together with the recursive
/// class namespace.  `dir(obj)` sorts the result after invoking this special
/// method; `object.__dir__(obj)` itself only promises a list.
pub(crate) fn object_dir_default(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let mut names: Vec<Wtf8Buf> = Vec::new();
    unsafe {
        let mut w_dict = crate::baseobjspace::getdict(obj)?;
        // `method.__dir__` forwards the underlying function's instance
        // dictionary as well as the method type's attributes.  The bound
        // method wrapper itself has no writable dict field.
        if w_dict.is_null() && pyre_object::function::is_method(obj) {
            let func = pyre_object::function::w_method_get_func(obj);
            if !func.is_null() {
                w_dict = crate::baseobjspace::getdict(func)?;
            }
        }
        if !w_dict.is_null() && pyre_object::is_dict(w_dict) {
            for (key, _) in pyre_object::w_dict_items(w_dict) {
                if pyre_object::is_str(key) {
                    names.push(pyre_object::w_str_get_wtf8(key).to_owned());
                }
            }
        }
        if let Some(w_type) = crate::typedef::r#type(obj) {
            if pyre_object::is_type(w_type.as_ptr()) {
                classdir_into(w_type.as_ptr(), &mut names)?;
            }
        }
    }
    names.sort();
    names.dedup();
    Ok(w_list_new(
        names
            .into_iter()
            .map(pyre_object::w_str_from_wtf8)
            .collect(),
    ))
}

/// util.py:62 `_classdir` / typeobject.py:1234 `descr__dir__`.
///
/// Unlike [`object_dir_default`], a type receiver contributes its own class
/// namespace and recursively the namespaces of its bases; its metaclass is
/// not part of the result.  Keep this as the direct `type.__dir__` target so
/// installing that descriptor does not make `dir()` redispatch recursively.
pub(crate) fn type_dir_default(w_type: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let mut names: Vec<Wtf8Buf> = Vec::new();
    unsafe { classdir_into(w_type, &mut names) }?;
    names.sort();
    names.dedup();
    Ok(w_list_new(
        names
            .into_iter()
            .map(pyre_object::w_str_from_wtf8)
            .collect(),
    ))
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
        // frame exactly as `locals()` does (module scope returns the globals
        // dict), then return the mapping's sorted keys.  `getdictscope` rather
        // than `frame_locals_snapshot`: only the key set is read, and the two
        // agree on it, so the snapshot copy would be a pure allocation.
        let frame = topframe_for_locals();
        if frame.is_null() {
            return Ok(w_list_new(vec![]));
        }
        let frame_mut = unsafe { &mut *frame };
        let w_locals_dict = frame_mut.getdictscope()?;
        if w_locals_dict.is_null() {
            return Ok(w_list_new(vec![]));
        }
        // `_dir_locals` reads the key set through `PyMapping_Keys(locals)`,
        // so a locals mapping that overrides `keys()` (`eval(src, {}, A())`
        // with `A(dict)`) is honoured; only an exact dict takes the
        // `PyDict_Keys` fast path that iterates the storage directly.
        let w_keys = if unsafe {
            pyre_object::pyobject::is_exact_type(w_locals_dict, &pyre_object::pyobject::DICT_TYPE)
        } {
            w_locals_dict
        } else {
            let keys_meth = crate::baseobjspace::getattr_str(w_locals_dict, "keys")?;
            crate::call_and_check(keys_meth, &[])?
        };
        let keys_iter = crate::baseobjspace::iter(w_keys)?;
        let keys = collect_iterable(keys_iter)?;
        return builtin_sorted(&[w_list_new(keys)]);
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
        if let Some((owner, dir_meth)) =
            unsafe { crate::baseobjspace::lookup_where_pair(w_type.as_ptr(), "__dir__") }
        {
            // The default object.__dir__ is implemented by the manual generic
            // paths below.  Keep descending when it is merely inherited so
            // module.__dir__, type.__dir__, and their surrogate-preserving
            // namespace walkers retain their specialized behavior.  A real
            // override on any other owner is invoked directly, exactly as
            // app_inspect.py's lookup_special requires.
            if !std::ptr::eq(owner, crate::typedef::w_object()) {
                let result = unsafe {
                    crate::baseobjspace::get_and_call_function(dir_meth, obj, w_type.as_ptr(), &[])
                }?;
                return builtin_sorted(&[result]);
            }
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
            // util.py:80 `_objectdir` (`object.__dir__`) — use ordinary
            // `getattr(obj, '__dict__'/'__class__', None)`, not raw layout
            // fields.  A class may shadow either name with a slot/descriptor;
            // in particular an uninitialised `__class__` slot suppresses the
            // recursive class namespace while the live `__dict__` remains.
            let w_dict = match crate::baseobjspace::getattr_str(obj, "__dict__") {
                Ok(w_dict) if pyre_object::is_dict(w_dict) => Some(w_dict),
                Ok(_) => None,
                Err(e) if e.kind == crate::PyErrorKind::AttributeError => None,
                Err(e) => return Err(e),
            };
            if let Some(w_dict) = w_dict {
                for (k, _) in pyre_object::w_dict_items(w_dict) {
                    if pyre_object::is_str(k) {
                        names.push(pyre_object::w_str_get_wtf8(k).to_owned());
                    }
                }
            }
            let w_class = match crate::baseobjspace::getattr_str(obj, "__class__") {
                Ok(w_class) => Some(w_class),
                Err(e) if e.kind == crate::PyErrorKind::AttributeError => None,
                Err(e) => return Err(e),
            };
            if let Some(w_class) = w_class {
                if pyre_object::is_type(w_class) {
                    classdir_into(w_class, &mut names)?;
                }
            }
        } else {
            // Fallback `_objectdir` (util.py:80) for builtin W_Root types
            // (Function, StaticMethod, PyTraceback, dict views, etc.).  Some
            // W_Root subclasses own a typed dictionary field even though
            // `is_instance()` is false, so use W_Root.getdict + _classdir
            // rather than assuming this whole branch has no instance dict.
            return object_dir_default(obj);
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
    // `space.id` (baseobjspace.py): a plain `int` yields its
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
    // The scalar builtins are not subclassable in place: an *exact* instance
    // can only ever reach the `__hash__` these arms of `hash_value` already
    // compute, so the slot lookup and the call protocol below would just
    // route back here.  This is the digest every interpreter-level dict probe
    // needs, and it is on the path of every module / `sys.modules` /
    // namespace lookup — `UnicodeDictStrategy`'s upstream storage hashes its
    // unwrapped key the same way, without an app-level call.
    unsafe {
        for tp in [
            &pyre_object::STR_TYPE,
            &pyre_object::INT_TYPE,
            &pyre_object::BOOL_TYPE,
            &pyre_object::LONG_TYPE,
            &pyre_object::FLOAT_TYPE,
            &pyre_object::bytesobject::BYTES_TYPE,
        ] {
            if is_exact_type(obj, tp) {
                return Ok(hash_value(obj));
            }
        }
    }
    unsafe {
        let kind = if pyre_object::is_dict(obj) {
            Some("dict")
        } else if pyre_object::is_list(obj) {
            Some("list")
        } else if pyre_object::is_set(obj) {
            // `frozenset` is hashable per setobject.py _hash_frozenset.
            Some("set")
        } else if pyre_object::is_bytearray(obj) {
            Some("bytearray")
        } else if pyre_object::dictmultiobject::is_dict_view_keys(obj) {
            // `dictmultiobject.py:1626 _is_set_like` — only the keys and items
            // views are set-like: they define `__eq__` and so are unhashable.
            // The values view keeps `object.__hash__`.
            Some("dict_keys")
        } else if pyre_object::dictmultiobject::is_dict_view_items(obj) {
            Some("dict_items")
        } else {
            None
        };
        if let Some(name) = kind {
            // The concrete builtin advertises `__hash__ = None`, but a user
            // subclass may replace that slot.  PyPy's normal special-method
            // lookup reaches the subclass entry before the inherited typedef
            // value (`descroperation.py:556-565`), so only reject the object
            // after giving a non-None override the call.
            if let Some(w_type) = crate::typedef::r#type(obj) {
                if let Some(method) =
                    crate::baseobjspace::lookup_in_type(w_type.as_ptr(), "__hash__")
                {
                    if !pyre_object::is_none(method) {
                        return hash_call_normalize(method, obj, w_type.as_ptr());
                    }
                }
            }
            return Err(crate::PyError::type_error(&format!(
                "unhashable type: '{}'",
                name
            )));
        }
        if pyre_object::sliceobject::is_slice(obj) {
            return slice_hash_value(obj);
        }
        if pyre_object::memoryview::is_w_memoryview(obj) {
            // `descr_hash` — released and writable views are unhashable; a
            // read-only view hashes its raw bytes, cached in `_hash`.
            return memoryview_hash_value(obj);
        }
        // A tuple/frozenset *subclass* can set `__hash__ = None` to become
        // unhashable, or override `__hash__` with its own method; the
        // structural fast paths below hash by contents and would ignore both.
        // For a non-exact receiver consult the `__hash__` slot: `None` raises,
        // a genuine override dispatches, and the inherited structural hash
        // falls through to the fast path.
        let is_tuple_recv = is_tuple(obj);
        let is_frozenset_recv = pyre_object::is_frozenset(obj);
        if (is_tuple_recv && !is_exact_type(obj, &TUPLE_TYPE))
            || (is_frozenset_recv && !is_exact_type(obj, &pyre_object::setobject::FROZENSET_TYPE))
        {
            if let Some(w_type) = crate::typedef::r#type(obj) {
                if let Some(method) =
                    crate::baseobjspace::lookup_in_type(w_type.as_ptr(), "__hash__")
                {
                    if pyre_object::is_none(method) {
                        return Err(unhashable_type_error(obj));
                    }
                    let base = if is_tuple_recv {
                        crate::typedef::gettypefor(&TUPLE_TYPE)
                    } else {
                        crate::typedef::gettypefor(&pyre_object::setobject::FROZENSET_TYPE)
                    };
                    let base_hash = base
                        .and_then(|b| crate::baseobjspace::lookup_in_type(b.as_ptr(), "__hash__"));
                    if Some(method) != base_hash {
                        return hash_call_normalize(method, obj, w_type.as_ptr());
                    }
                }
            }
        }
        if is_tuple(obj) {
            // CPython 3.14 `tuple_hash`: a successful aggregate hash is
            // retained on the tuple.  Check before the recursive stack guard
            // so repeated dict probes do not touch the elements at all.
            if let Some(hash) = pyre_object::w_tuple_cached_hash(obj) {
                return Ok(hash);
            }
            // The element walk is the one recursive step here, and the scalar
            // fast path above no longer reaches the call protocol's own
            // guard, so a nest deep enough to exhaust the C stack has to be
            // caught on the way down.
            crate::stack_check::stack_check()?;
            if let Some(hash) = pyre_object::w_tuple_cached_hash(obj) {
                return Ok(hash);
            }
            let n = w_tuple_len(obj);
            let mut hashes = Vec::with_capacity(n);
            for i in 0..(n as i64) {
                if let Some(item) = w_tuple_getitem(obj, i) {
                    hashes.push(try_hash_value(item)?);
                }
            }
            let hash = _hash_tuple_xx(&hashes);
            pyre_object::w_tuple_set_cached_hash(obj, hash);
            return Ok(hash);
        }
        if pyre_object::is_frozenset(obj) {
            return Ok(frozenset_hash_from_storage(obj));
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
            let mut hashes = Vec::with_capacity(n);
            for i in 0..n {
                if let Some(item) = pyre_object::w_tuple_getitem(args, i as i64) {
                    // Preserve frozenset construction's fallible element-hash
                    // phase.  Building a low-level frozenset first would use
                    // its stored infallible digest path and hide an
                    // unhashable union member.
                    hashes.push(try_hash_value(item)?);
                }
            }
            return Ok(_hash_frozenset(&hashes));
        }
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__hash__") {
                if pyre_object::is_none(method) {
                    return Err(unhashable_type_error(obj));
                }
                return hash_call_normalize(method, obj, w_type);
            }
        }
        // A type may declare itself unhashable via `__hash__ = None`
        // (a typed-payload `#[pyre_methods(unhashable)]` layout such as
        // `deque`); the `is_instance` arm above covers user classes, this
        // covers the builtin/typed-payload layouts before the identity-hash
        // fallback.
        if let Some(w_type) = crate::typedef::r#type(obj) {
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type.as_ptr(), "__hash__") {
                if pyre_object::is_none(method) {
                    return Err(unhashable_type_error(obj));
                }
                // A subclass may override `__hash__` (e.g. `class D(deque)`);
                // honor the override.  The inherited default `object.__hash__`
                // (identity) is left to the `hash_value` fallback below, which
                // computes the correct per-type builtin hash for the base
                // payload rather than a pointer identity.
                let default_hash =
                    crate::baseobjspace::lookup_in_type(crate::typedef::w_object(), "__hash__");
                if default_hash != Some(method) {
                    return hash_call_normalize(method, obj, w_type.as_ptr());
                }
            }
        }
    }
    Ok(hash_value(obj))
}

/// CPython 3.14 `slice_hash`, the tuplehash-style three-lane mixer added
/// after the PyPy 3.11 source version. Component hash errors propagate.
fn slice_hash_value(obj: PyObjectRef) -> Result<i64, crate::PyError> {
    const PRIME1: u64 = 11_400_714_785_074_694_791;
    const PRIME2: u64 = 14_029_467_366_897_019_727;
    const PRIME5: u64 = 2_870_177_450_012_600_261;
    let parts = unsafe {
        [
            pyre_object::sliceobject::w_slice_get_start(obj),
            pyre_object::sliceobject::w_slice_get_stop(obj),
            pyre_object::sliceobject::w_slice_get_step(obj),
        ]
    };
    let mut acc = PRIME5;
    for part in parts {
        let lane = try_hash_value(part)? as u64;
        acc = acc.wrapping_add(lane.wrapping_mul(PRIME2));
        acc = acc.rotate_left(31);
        acc = acc.wrapping_mul(PRIME1);
    }
    if acc == u64::MAX {
        acc = 1_546_275_796;
    }
    Ok(acc as i64)
}

/// Call a resolved `__hash__` method and normalize its result:
/// `descroperation.py:576-579` — accept `bool` / `int` / `long`, reject other
/// return types, and map a `-1` result to `-2`.
fn hash_call_normalize(
    method: PyObjectRef,
    obj: PyObjectRef,
    w_type: PyObjectRef,
) -> Result<i64, crate::PyError> {
    let r = unsafe { crate::baseobjspace::get_and_call_function(method, obj, w_type, &[]) }?;
    let h = unsafe {
        if is_bool(r) {
            pyre_object::w_bool_get_value(r) as i64
        } else if is_int(r) {
            w_int_get_value(r)
        } else if is_long(r) {
            _hash_long(pyre_object::w_long_get_value(r))
        } else {
            return Err(crate::PyError::type_error(
                "__hash__ method should return an integer",
            ));
        }
    };
    Ok(if h == -1 { -2 } else { h })
}

fn unhashable_type_error(obj: PyObjectRef) -> crate::PyError {
    let name = unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
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
const HASH_BITS: i64 = 61;
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

/// pypy/objspace/std/longobject.py `_hash_long`.
///
/// Reduce the 63-bit rbigint digits modulo the Mersenne prime
/// `2**61 - 1`, then apply the sign and the `-1 -> -2` sentinel rule.
#[inline]
pub(crate) fn _hash_long(v: &BigInt) -> i64 {
    const HASH_SHIFT: i64 = pyre_object::rbigint::SHIFT % HASH_BITS;
    let mut i = v.numdigits();
    let mut x = 0_u64;
    while i > 0 {
        i -= 1;
        x = ((x << HASH_SHIFT) & HASH_MODULUS) + (x >> (HASH_BITS - HASH_SHIFT));
        x += v.udigit(i);
        if pyre_object::rbigint::SHIFT > HASH_BITS {
            x = (x & HASH_MODULUS) + (x >> HASH_BITS);
        }
        if x >= HASH_MODULUS {
            x -= HASH_MODULUS;
        }
    }
    let hash = (x as i64) * v.get_sign();
    hash - (hash == -1) as i64
}

/// Numeric hash of a `float`.  `rustpython_common::hash::hash_float`
/// reduces the mantissa/exponent modulo the Mersenne prime
/// `HASH_MODULUS = 2**61 - 1` (keeping the sign), so `hash(2.0) == hash(2)`
/// and the `±inf` sentinels are `±314159`; subnormals decompose exactly.
/// It returns `None` for NaN.  Object-aware callers replace that case with
/// `default_identity_hash`; `_hash_float` retains PyPy's internal sentinel
/// for call sites that have no wrapped object.
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
    _hash_tuple_xx_iter(items.iter().copied())
}

/// `_hash_tuple_xx` over element digests produced on the fly.
fn _hash_tuple_xx_iter(items: impl Iterator<Item = i64>) -> i64 {
    rustpython_common::hash::hash_tuple(items.map(Ok::<i64, ()>))
        .expect("element hashes are precomputed, so the fold cannot fail")
}

/// CPython/PyPy's deterministic seed expansion
/// (`Python/bootstrap_hash.c` / `rsiphash.py:lcg_urandom`).
fn hash_secret_from_seed(mut seed: u32) -> [u8; 16] {
    let mut secret = [0u8; 16];
    for byte in &mut secret {
        seed = seed.wrapping_mul(214013).wrapping_add(2531011);
        *byte = ((seed >> 16) & 0xff) as u8;
    }
    secret
}

/// Process-global CPython 3.14 hash secret.  RPython's `rsiphash.seed` is
/// likewise a single process-global owner initialized from `PYTHONHASHSEED`;
/// it is not interpreter-thread-local state.
static HASH_SECRET: std::sync::OnceLock<[u8; 16]> = std::sync::OnceLock::new();

/// How a `PYTHONHASHSEED` value that names no seed is reported.
pub const HASH_SEED_ERROR: &str =
    "PYTHONHASHSEED must be \"random\" or an integer in range [0; 4294967295]";

/// Fix the process-global hash secret from `PYTHONHASHSEED`, leaving it unset
/// for `"random"` or an absent value so the first digest samples a random one.
///
/// The variable is read through the host seam, so under sandbox the value comes
/// from the controller's environment rather than the child's cleared one.  A
/// value that names no seed yields [`HASH_SEED_ERROR`] instead of terminating:
/// the launcher owns the process lifetime, so reporting it is its call.
///
/// A digest taken before this runs pins a random secret and the variable is
/// ignored from then on, so the launcher calls it before anything hashes.
pub fn init_hash_secret_from_env() -> Result<(), &'static str> {
    let value = crate::host_seam::getenv(b"PYTHONHASHSEED")
        .ok()
        .flatten()
        .unwrap_or_default();
    if value.is_empty() || value.as_slice() == b"random" {
        return Ok(());
    }
    let seed = std::str::from_utf8(&value)
        .ok()
        .and_then(|text| text.parse::<u32>().ok())
        .ok_or(HASH_SEED_ERROR)?;
    // `_build_key_from_seed` gives seed 0 the all-zero key rather than running
    // the expansion over it.
    let secret = if seed == 0 {
        [0; 16]
    } else {
        hash_secret_from_seed(seed)
    };
    let _ = HASH_SECRET.set(secret);
    Ok(())
}

fn hash_secret() -> &'static [u8; 16] {
    HASH_SECRET.get_or_init(|| {
        let mut secret = [0u8; 16];
        getrandom::fill(&mut secret)
            .expect("failed to get random numbers to initialize Python hash secret");
        secret
    })
}

/// CPython 3.14 `_Py_HashBytes`: empty input hashes to zero; every other byte
/// sequence uses SipHash-1-3 with the process-global 128-bit secret.
fn _hash_bytes_with_key(bytes: &[u8], secret: &[u8; 16]) -> i64 {
    if bytes.is_empty() {
        return 0;
    }
    use core::hash::Hasher;
    let mut hasher = siphasher::sip::SipHasher13::new_with_key(secret);
    hasher.write(bytes);
    let raw = hasher.finish() as i64;
    raw - ((raw == -1) as i64)
}

#[inline]
fn _hash_bytes(bytes: &[u8]) -> i64 {
    _hash_bytes_with_key(bytes, hash_secret())
}

/// CPython 3.14 `unicode_hash`: hash the PEP 393 canonical code-unit storage,
/// selecting one-, two-, or four-byte native-endian units from the maximum
/// code point.  Pyre stores WTF-8, so materialize only this hash input.
fn _hash_unicode_with_key(wtf8: &rustpython_wtf8::Wtf8, secret: &[u8; 16]) -> i64 {
    // An all-ASCII string's PEP 393 canonical code-unit storage is its
    // one-byte-per-char form, byte-identical to the WTF-8 bytes, so hash those
    // directly instead of materializing a separate code-unit buffer. This is
    // the common case (identifiers, dict keys) and skips two Vec allocations.
    let storage = wtf8.as_bytes();
    if storage.is_ascii() {
        return _hash_bytes_with_key(storage, secret);
    }
    let codepoints: Vec<u32> = wtf8.code_points().map(|cp| cp.to_u32()).collect();
    let maxchar = codepoints.iter().copied().max().unwrap_or(0);
    let mut bytes = Vec::with_capacity(
        codepoints.len()
            * if maxchar <= 0xff {
                1
            } else if maxchar <= 0xffff {
                2
            } else {
                4
            },
    );
    if maxchar <= 0xff {
        bytes.extend(codepoints.into_iter().map(|cp| cp as u8));
    } else if maxchar <= 0xffff {
        for cp in codepoints {
            bytes.extend_from_slice(&(cp as u16).to_ne_bytes());
        }
    } else {
        for cp in codepoints {
            bytes.extend_from_slice(&cp.to_ne_bytes());
        }
    }
    _hash_bytes_with_key(&bytes, secret)
}

#[inline]
fn _hash_unicode(wtf8: &rustpython_wtf8::Wtf8) -> i64 {
    _hash_unicode_with_key(wtf8, hash_secret())
}

/// `space.hash_w` digest for a `str` computed directly from its WTF-8 bytes
/// — the value [`hash_value`] returns for a str key — exposed for the
/// `dict_eq_hook::HASH_STR_HOOK` trampoline so str-keyed dict GET probes hash
/// without a `W_UnicodeObject`.
#[inline]
pub fn hash_str_bytes(bytes: &[u8]) -> i64 {
    // ASCII fast path (see `_hash_unicode_with_key`): the storage bytes are the
    // canonical hash input, so hash them without validating/constructing a Wtf8.
    if bytes.is_ascii() {
        return _hash_bytes(bytes);
    }
    let wtf8 = rustpython_wtf8::Wtf8::from_bytes(bytes).expect("str storage is valid WTF-8");
    _hash_unicode(wtf8)
}

/// `setobject.py W_FrozensetObject.descr_hash` — the order-independent
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

/// CPython 3.14 consumes the digests cached in the set table and caches the
/// aggregate on the frozenset. This is observably newer than PyPy 3.11's
/// first `descr_hash` walk: an element's `__hash__` runs only on insertion,
/// never again for `hash(frozenset)`.
fn frozenset_hash_from_storage(obj: PyObjectRef) -> i64 {
    unsafe {
        if let Some(hash) = pyre_object::w_frozenset_cached_hash(obj) {
            return hash;
        }
        let hashes = pyre_object::w_set_stored_hashes(obj);
        let hash = _hash_frozenset(&hashes);
        pyre_object::w_frozenset_set_cached_hash(obj, hash);
        hash
    }
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
/// CPython 3.14's `_Py_HashBytes` uses SipHash-1-3 and returns zero for empty
/// input. Strings hash their PEP 393 canonical code-unit representation rather
/// than pyre's internal WTF-8 bytes, keeping all seeded values bit-identical.
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
            let value = pyre_object::w_float_get_value(obj);
            return if value.is_nan() {
                crate::typedef::default_identity_hash_value(obj)
            } else {
                _hash_float(value)
            };
        }
        if pyre_object::is_complex(obj) {
            return crate::objspace::descroperation::complex_hash(
                obj,
                pyre_object::w_complex_get_real(obj),
                pyre_object::w_complex_get_imag(obj),
            );
        }
        if is_str(obj) {
            // CPython `unicode_hash` caches the PEP 393 content digest.
            let cached = pyre_object::w_str_get_hash(obj);
            if cached != 0 {
                return cached;
            }
            let hash = _hash_unicode(pyre_object::w_str_get_wtf8(obj));
            pyre_object::w_str_set_hash(obj, hash);
            return hash;
        }
        // CPython `bytes_hash` — the same `_Py_HashBytes` primitive, over the
        // bytes payload itself (bytearray is mutable / unhashable).
        if pyre_object::is_bytes(obj) {
            return _hash_bytes(pyre_object::bytesobject::w_bytes_data(obj));
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
                return _hash_bytes(&memoryview_gather_bytes(obj));
            }
        }
        if pyre_object::is_none(obj) {
            // CPython 3.12+ `Objects/object.c:none_hash`; introduced by
            // gh-99541 so singleton hashes are reproducible across processes.
            // PyPy inherits identity hashing here, so this is an intentional
            // CPython 3.14 surface delta rather than a PyPy semantic port.
            return 0xFCA8_6420;
        }
        if is_tuple(obj) {
            // CPython 3.14 `tuple_hash`: cache the successful aggregate after
            // the first element walk. This supersedes PyPy's otherwise
            // structurally equivalent `_descr_hash_unroll` for the requested
            // 3.14 observable semantics.
            if let Some(hash) = pyre_object::w_tuple_cached_hash(obj) {
                return hash;
            }
            let n = w_tuple_len(obj) as i64;
            let hash =
                _hash_tuple_xx_iter((0..n).filter_map(|i| w_tuple_getitem(obj, i).map(hash_value)));
            pyre_object::w_tuple_set_cached_hash(obj, hash);
            return hash;
        }
        if pyre_object::is_frozenset(obj) {
            return frozenset_hash_from_storage(obj);
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
        crate::typedef::default_identity_hash_value(obj)
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
        // bytesobject.py:473 names bytes "bytes"; bytearrayobject.py:213
        // names bytearray "string".
        if pyre_object::bytesobject::is_bytes_like(obj) {
            let data = pyre_object::bytesobject::bytes_like_data(obj);
            if data.len() != 1 {
                let noun = if pyre_object::bytesobject::is_bytes(obj) {
                    "bytes"
                } else {
                    "string"
                };
                return Err(crate::PyError::type_error(format!(
                    "ord() expected a character, but {noun} of length {} found",
                    data.len()
                )));
            }
            return Ok(w_int_new(data[0] as i64));
        }
    }
    Err(crate::PyError::type_error(format!(
        "ord() expected string of length 1, but {} found",
        crate::baseobjspace::object_functionstr_type_name(obj)
    )))
}

/// `chr(i)` — PyPy: operation.py chr
fn builtin_chr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "chr() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    // Python 3.14 `builtin_chr` uses the index protocol: `__int__` alone is
    // insufficient, floats are rejected, and an arbitrarily large int reaches
    // the range check rather than overflowing a host machine word.  This is
    // the target-version difference from PyPy's `@unwrap_spec(code=int)`.
    let val = index_to_bigint(args[0])?;
    if val < BigInt::from(0) || val > BigInt::from(0x10ffff_u32) {
        return Err(crate::PyError::value_error(
            "chr() arg not in range(0x110000)",
        ));
    }
    let val = val
        .to_u32()
        .expect("chr range check guarantees a value fitting u32");
    let cp = match char::from_u32(val) {
        Some(c) => CodePoint::from(c),
        // Surrogate code points (0xD800-0xDFFF) are valid chr() arguments and
        // produce a lone-surrogate string; char::from_u32 rejects them, so
        // build the string through a WTF-8 code point instead.
        None => CodePoint::from_u32(val).expect("val is in 0..=0x10ffff per the range check above"),
    };
    let mut one = Wtf8Buf::new();
    one.push(cp);
    Ok(pyre_object::w_str_from_wtf8_managed(one))
}

/// `filter(function or None, iterable)` — `functional.py:980-995
/// W_Filter___new__`.  A lazy iterator: `function == None` keeps truthy
/// items, otherwise `function(item)` is the predicate.
pub(crate) fn builtin_filter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "filter() takes no keyword arguments",
        ));
    }
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "filter expected 2 arguments, got {}",
            args.len()
        )));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(args[0]);
    let predicate_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(args[1]);
    let iterable_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let func = unsafe { pyre_object::gc_roots::shadow_stack_get(predicate_slot) };
    // `functional.py:921-924` — a None predicate is stored as PY_NULL.
    let w_predicate = if unsafe { pyre_object::is_none(func) } {
        pyre_object::PY_NULL
    } else {
        func
    };
    // `functional.py:925 self.w_iterable = space.iter(w_iterable)`.
    let w_iterable = crate::baseobjspace::iter(unsafe {
        pyre_object::gc_roots::shadow_stack_get(iterable_slot)
    })?;
    pyre_object::gc_roots::pin_root(w_iterable);
    let iterator_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    Ok(pyre_object::functional::w_filter_new(
        if w_predicate.is_null() {
            pyre_object::PY_NULL
        } else {
            unsafe { pyre_object::gc_roots::shadow_stack_get(predicate_slot) }
        },
        unsafe { pyre_object::gc_roots::shadow_stack_get(iterator_slot) },
    ))
}

/// `map(func, *iterables, strict=False)` — `functional.py:888-902
/// W_Map___new__` plus the CPython 3.14 `strict` keyword.  A lazy iterator:
/// each `next()` pulls one item per iterable and calls `func(*items)`.
pub(crate) fn builtin_map(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["strict"], "map")?;
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "map() must have at least two arguments.",
        ));
    }

    // PyPy's `args_w` and `build_iterators_from_args` keep every argument live
    // while `space.iter` and the Python 3.14 `strict` truth conversion execute.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::shadow_stack_len();
    for &arg in args {
        pyre_object::gc_roots::pin_root(arg);
    }
    let w_strict = kwarg_get(kwargs, "strict");
    let strict_slot = w_strict.map(|value| {
        pyre_object::gc_roots::pin_root(value);
        pyre_object::gc_roots::shadow_stack_len() - 1
    });
    let strict = strict_slot
        .map(|slot| {
            crate::baseobjspace::is_true(unsafe { pyre_object::gc_roots::shadow_stack_get(slot) })
        })
        .transpose()?
        .unwrap_or(false);

    // `functional.py:835-836 build_iterators_from_args` — `iter()` each input.
    let mut iter_slots = Vec::with_capacity(args.len() - 1);
    for index in 1..args.len() {
        let w_iter = crate::baseobjspace::iter(unsafe {
            pyre_object::gc_roots::shadow_stack_get(args_base + index)
        })?;
        pyre_object::gc_roots::pin_root(w_iter);
        iter_slots.push(pyre_object::gc_roots::shadow_stack_len() - 1);
    }
    let w_iterators = pyre_object::w_list_new(
        iter_slots
            .into_iter()
            .map(|slot| unsafe { pyre_object::gc_roots::shadow_stack_get(slot) })
            .collect(),
    );
    pyre_object::gc_roots::pin_root(w_iterators);
    let iterators_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    Ok(pyre_object::functional::w_map_new(
        unsafe { pyre_object::gc_roots::shadow_stack_get(args_base) },
        unsafe { pyre_object::gc_roots::shadow_stack_get(iterators_slot) },
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
    // `zip_new` parses its keywords with `PyArg_ParseTupleAndKeywords(empty,
    // kwds, "|$p:zip", ...)`, so the keyword *count* is checked against the
    // single `strict` slot before any name is looked at.
    clinic_arity("zip", 0, real_kwarg_count(kwargs), 0, 0, 1)?;
    kwarg_reject_unknown(kwargs, &["strict"], "zip")?;
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::shadow_stack_len();
    for &arg in args {
        pyre_object::gc_roots::pin_root(arg);
    }
    let strict_slot = kwarg_get(kwargs, "strict").map(|value| {
        pyre_object::gc_roots::pin_root(value);
        pyre_object::gc_roots::shadow_stack_len() - 1
    });
    let strict = strict_slot
        .map(|slot| {
            crate::baseobjspace::is_true(unsafe { pyre_object::gc_roots::shadow_stack_get(slot) })
        })
        .transpose()?
        .unwrap_or(false);
    // `functional.py:835-836 build_iterators_from_args` — `iter()` each input.
    let mut iter_slots = Vec::with_capacity(args.len());
    for index in 0..args.len() {
        let w_iter = crate::baseobjspace::iter(unsafe {
            pyre_object::gc_roots::shadow_stack_get(args_base + index)
        })?;
        pyre_object::gc_roots::pin_root(w_iter);
        iter_slots.push(pyre_object::gc_roots::shadow_stack_len() - 1);
    }
    let w_iterators = pyre_object::w_list_new(
        iter_slots
            .into_iter()
            .map(|slot| unsafe { pyre_object::gc_roots::shadow_stack_get(slot) })
            .collect(),
    );
    pyre_object::gc_roots::pin_root(w_iterators);
    let iterators_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    Ok(pyre_object::functional::w_zip_new(
        unsafe { pyre_object::gc_roots::shadow_stack_get(iterators_slot) },
        strict,
    ))
}

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
    // The constructor can execute user `__index__` and allocate. Keep both
    // arguments in the shadow stack exactly as the translated RPython locals
    // remain GC-visible across `space.index` / `space.iter`.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(source);
    let source_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let raw_start = start_obj.unwrap_or(pyre_object::PY_NULL);
    pyre_object::gc_roots::pin_root(raw_start);
    let raw_start_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    // `functional.py:255-264 descr___new__` — `space.index(w_start)` then
    // `space.int_w(w_start)`; ONLY OverflowError activates the bigint slot.
    // This preserves an arbitrary-precision start object verbatim, while a
    // machine-sized int/long/bool stays in the unboxed `index` fast path.
    let (start, w_start) = if raw_start.is_null() {
        (0, pyre_object::PY_NULL)
    } else {
        let indexed = crate::baseobjspace::space_index(unsafe {
            pyre_object::gc_roots::shadow_stack_get(raw_start_slot)
        })?;
        pyre_object::gc_roots::pin_root(indexed);
        let indexed_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        match crate::baseobjspace::int_w(indexed) {
            Ok(value) => (value, pyre_object::PY_NULL),
            Err(e) if e.kind == crate::PyErrorKind::OverflowError => (-1, unsafe {
                pyre_object::gc_roots::shadow_stack_get(indexed_slot)
            }),
            Err(e) => return Err(e),
        }
    };
    pyre_object::gc_roots::pin_root(w_start);
    let w_start_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // `functional.py:268-271` — `if start == 0 and type(w_iterable) is
    // W_ListObject: w_iter = w_iterable` (skip space.iter for the
    // common list-source case so __next__ can `getitem(index)`
    // directly).  Otherwise call `space.iter(w_iterable)`.
    let source = unsafe { pyre_object::gc_roots::shadow_stack_get(source_slot) };
    let w_iter_or_list =
        if start == 0 && w_start.is_null() && unsafe { pyre_object::is_list(source) } {
            source
        } else {
            crate::baseobjspace::iter(source)?
        };
    Ok(pyre_object::functional::w_enumerate_new(
        w_iter_or_list,
        start,
        unsafe { pyre_object::gc_roots::shadow_stack_get(w_start_slot) },
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
    // `descr___new__2` can execute `__reversed__`, `__len__`, or allocate an
    // iterator. The translated RPython argument remains a GC root throughout;
    // keep the Rust carrier in the shadow stack and reload at each call site.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(args[0]);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let obj = unsafe { pyre_object::gc_roots::shadow_stack_get(obj_slot) };
    unsafe {
        // EXACT builtin list → `iterobject.py W_ReverseSeqIterObject` (the
        // Python 3.14-visible `list_reverseiterator`); tuple and the other
        // sequence fallbacks use `functional.py W_ReversedIterator`.
        // A subclass
        // shares the builtin `ob_type` but has its own `w_class`, so it falls
        // through to the `__reversed__` MRO lookup below — CPython honors a
        // subclass override (a non-overriding subclass inherits the builtin
        // `list.__reversed__`, which is the same lazy iterator).
        if pyre_object::is_exact_builtin_instance(obj) {
            if pyre_object::is_list(obj) {
                let n = pyre_object::w_list_len(obj) as i64;
                return Ok(pyre_object::w_list_reverse_iter_new(
                    pyre_object::gc_roots::shadow_stack_get(obj_slot),
                    n - 1,
                ));
            }
            if pyre_object::is_tuple(obj) {
                let n = pyre_object::w_tuple_len(obj) as i64;
                return Ok(pyre_object::functional::w_reversed_new(
                    pyre_object::gc_roots::shadow_stack_get(obj_slot),
                    n - 1,
                ));
            }
            // bytes / bytearray expose the sequence protocol at the C level but
            // not as `__getitem__` / `__len__` type slots, so they would miss
            // the `PySequence_Check` path below. `getitem` indexes them
            // (returning the int byte) for `W_ReversedIterator` to walk.
            if pyre_object::bytesobject::is_bytes(obj)
                || pyre_object::bytearrayobject::is_bytearray(obj)
            {
                let n = crate::baseobjspace::len_w(obj)?;
                return Ok(pyre_object::functional::w_reversed_new(
                    pyre_object::gc_roots::shadow_stack_get(obj_slot),
                    n - 1,
                ));
            }
        }
        // range: functional.py W_Range.descr_reversed — reflect
        // the span and hand back a fresh reverse-walking iterator. (range is
        // not subclassable, so no override can apply.)
        if pyre_object::is_w_range(obj) {
            return Ok(pyre_object::w_range_reversed(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
            ));
        }
    }
    // `__reversed__` resolved through the type MRO (`functional.py:362-366`) —
    // honors a subclass override and the inherited builtin `list.__reversed__`,
    // and any user object defining `__reversed__`.
    let obj = unsafe { pyre_object::gc_roots::shadow_stack_get(obj_slot) };
    if let Some(tp) = crate::typedef::r#type(obj) {
        if let Some(method) =
            unsafe { crate::baseobjspace::lookup_in_type(tp.as_ptr(), "__reversed__") }
        {
            // Python 3.14 `slot1` semantics: explicitly assigning None disables
            // the protocol and does not fall back to `__len__`/`__getitem__`.
            if unsafe { pyre_object::is_none(method) } {
                return Err(crate::PyError::type_error(format!(
                    "'{}' object is not reversible",
                    crate::baseobjspace::object_functionstr_type_name(obj)
                )));
            }
            pyre_object::gc_roots::pin_root(method);
            let method_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            // Propagate a raised error from `__reversed__` instead of handing
            // back a null result.
            return unsafe {
                crate::baseobjspace::get_and_call_function(
                    pyre_object::gc_roots::shadow_stack_get(method_slot),
                    pyre_object::gc_roots::shadow_stack_get(obj_slot),
                    tp.as_ptr(),
                    &[],
                )
            };
        }
    }
    // functional.py:351 — without `__reversed__`, require the sequence
    // protocol. `PySequence_Check` first (an object with `__getitem__`): a
    // non-sequence is "not reversible", while a sequence missing `__len__`
    // raises the regular "has no len()" from `len`.
    if let Some(tp) = crate::typedef::r#type(obj) {
        let has_getitem =
            unsafe { crate::baseobjspace::lookup_in_type(tp.as_ptr(), "__getitem__") }.is_some();
        if has_getitem {
            // `functional.py:354-359` — reverse lazily through `W_ReversedIterator`.
            let n = crate::baseobjspace::len_w(obj)?;
            return Ok(pyre_object::functional::w_reversed_new(
                unsafe { pyre_object::gc_roots::shadow_stack_get(obj_slot) },
                n - 1,
            ));
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
    // `builtin_sorted` unpacks its one positional itself and leaves every
    // keyword to the `list.sort` it delegates to, so an unknown keyword is
    // reported by `sort`, not by `sorted`.
    if positional.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "sorted expected 1 argument, got {}",
            positional.len()
        )));
    }
    kwarg_reject_unknown(kwargs, &["key", "reverse"], "sort")?;
    let iterable = positional[0];
    let key_fn = kwarg_get(kwargs, "key").filter(|k| unsafe { !pyre_object::is_none(*k) });
    let reverse = kwarg_get(kwargs, "reverse")
        .map(|v| crate::baseobjspace::is_true(v))
        .transpose()?
        .unwrap_or(false);
    // `app_functional.py:7` `sorted` is `list(iterable)` followed by
    // `sorted_lst.sort(key=key, reverse=reverse)`, so the built list picks its
    // storage strategy first and the sort runs through the same body
    // `list.sort` uses.
    let w_list = w_list_new(collect_iterable(iterable)?);
    let _roots = pyre_object::gc_roots::push_roots();
    let list_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_list);
    sort_list_in_place(list_slot, key_fn, reverse)?;
    Ok(pyre_object::gc_roots::shadow_stack_get(list_slot))
}

/// `listobject.py:809 descr_sort` — the shared body of `list.sort` and
/// `sorted`.  `list_slot` is a shadow-stack slot holding the list, because a
/// key call or a comparison dunder can collect and move it.
pub(crate) fn sort_list_in_place(
    list_slot: usize,
    key_fn: Option<PyObjectRef>,
    reverse: bool,
) -> Result<(), crate::PyError> {
    // `descr_sort` sorts through the strategy's own `sort` whenever the list
    // is not object-strategy and no key was given: the storage is already
    // unwrapped, so there is no boxing, no comparison dunder, and no need to
    // empty the receiver first — no user code can run to observe it.
    if key_fn.is_none() {
        let list = pyre_object::gc_roots::shadow_stack_get(list_slot);
        unsafe {
            if let Some((items, len)) = pyre_object::listobject::w_list_int_items_raw(list) {
                sort_scalars(std::slice::from_raw_parts_mut(items, len), reverse)?;
                return Ok(());
            }
            if let Some((items, len)) = pyre_object::listobject::w_list_float_items_raw(list) {
                sort_scalars(std::slice::from_raw_parts_mut(items, len), reverse)?;
                return Ok(());
            }
        }
    }
    unsafe {
        // Hold the detached values in shadow-stack slots, not a bare Rust Vec,
        // while key and comparison calls can collect.  The receiver is empty
        // for the whole operation, so user code cannot alter this sorting
        // slice through the visible list.
        let list = pyre_object::gc_roots::shadow_stack_get(list_slot);
        let saved = pyre_object::listobject::w_list_items_copy_as_vec(list);
        let _roots = pyre_object::gc_roots::push_roots();
        let item_base = pyre_object::gc_roots::shadow_stack_len();
        for item in saved {
            pyre_object::gc_roots::pin_root(item);
        }
        let saved_len = pyre_object::gc_roots::shadow_stack_len() - item_base;
        pyre_object::listobject::w_list_clear(list);

        let (order, sorted) = sort_rooted_items(item_base, saved_len, key_fn, reverse);
        let list = pyre_object::gc_roots::shadow_stack_get(list_slot);
        // Whether the user mucked with the list during the sort: any mutation
        // switches the emptied receiver away from the Empty strategy, and a
        // list never switches back, so a net-zero append+pop is caught too.
        let mucked = !pyre_object::listobject::w_list_is_empty_strategy(list);

        // `descr_sort`'s `finally` puts the items back unconditionally,
        // discarding whatever the user stored into the receiver meanwhile.
        let restored = order
            .into_iter()
            .map(|index| pyre_object::gc_roots::shadow_stack_get(item_base + index))
            .collect();
        pyre_object::listobject::w_list_init_items(list, restored);
        sorted?;
        if mucked {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "list modified during sort",
            ));
        }
    }
    Ok(())
}

/// `IntegerListStrategy.sort` (`listobject.py:1963`) / `FloatListStrategy.sort`
/// (`:2067`) on the unwrapped storage.
///
/// `descr_sort`'s reverse handling — reverse, stable ascending sort, reverse
/// again — is kept here rather than the strategies' single trailing
/// `l.reverse()`: for `float` the difference is observable, since `-0.0` and
/// `0.0` compare equal but are distinct values whose relative order a stable
/// sort must preserve.
fn sort_scalars<T: Copy>(items: &mut [T], reverse: bool) -> Result<(), crate::PyError>
where
    ScalarLt: crate::listsort::SortLt<T>,
{
    if reverse {
        items.reverse();
    }
    crate::listsort::sort_with(items, &mut ScalarLt)?;
    if reverse {
        items.reverse();
    }
    Ok(())
}

/// `listobject.py:2429 IntSort.lt` / `:2434 FloatSort.lt` — a direct scalar
/// comparison, never a dunder.
struct ScalarLt;

impl crate::listsort::SortLt<i64> for ScalarLt {
    fn lt(&mut self, a: i64, b: i64) -> Result<bool, crate::PyError> {
        Ok(a < b)
    }
}

impl crate::listsort::SortLt<f64> for ScalarLt {
    fn lt(&mut self, a: f64, b: f64) -> Result<bool, crate::PyError> {
        Ok(a < b)
    }
}

/// Sort rooted item slots and return the resulting permutation.  All object
/// references that survive a Python call live in the shadow stack; the sort
/// itself only moves integer indices.
///
/// The permutation comes back even when a key call or a comparison raises,
/// because `descr_sort` puts `sorter.list` back from a `finally` — an
/// interrupted sort leaves the list holding its elements in whatever order the
/// merge had reached, not the order it started in.
pub(crate) fn sort_rooted_items(
    item_base: usize,
    item_len: usize,
    key_fn: Option<PyObjectRef>,
    reverse: bool,
) -> (Vec<usize>, Result<(), crate::PyError>) {
    let mut order: Vec<usize> = (0..item_len).collect();
    let _key_roots = pyre_object::gc_roots::push_roots();
    let key_base = pyre_object::gc_roots::shadow_stack_len();
    let key_fn_slot = key_fn.map(|key| {
        pyre_object::gc_roots::pin_root(key);
        key_base
    });
    let key_base = key_base + usize::from(key_fn_slot.is_some());
    if let Some(key_fn_slot) = key_fn_slot {
        for index in 0..item_len {
            // `_compute_keys_for_sorting` (listobject.py:894) runs before the
            // `reverse` flip, so a raising key leaves the input order.
            match crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(key_fn_slot),
                &[pyre_object::gc_roots::shadow_stack_get(item_base + index)],
            ) {
                Ok(key) => pyre_object::gc_roots::pin_root(key),
                Err(err) => return (order, Err(err)),
            }
        }
    }

    // `descr_sort` reverses before and after a stable ascending sort for
    // `reverse=True`, so equal elements keep their original relative order (a
    // stable descending sort).  A single post-sort reverse would instead flip
    // ties.
    if reverse {
        order.reverse();
    }
    let base = if key_fn_slot.is_some() {
        key_base
    } else {
        item_base
    };
    let mut compare = sort_compare_for(base, item_len, key_fn_slot.is_some());
    let result = crate::listsort::sort_with(&mut order, &mut compare);
    // Second half of the double-reverse (see above), which runs on the raising
    // path too — an interrupted `reverse=True` sort must not leave the list in
    // the opposite orientation from the one it was handed.  `descr_sort` puts
    // this reverse inside its `try` and so skips it; `list_sort_impl` does not,
    // and that is the behaviour to match.
    if reverse {
        order.reverse();
    }
    (order, result)
}

/// The comparison primitive `listsort.py`'s TimSort drives, holding the
/// shadow-stack base of the values being compared (the keys under `key=`,
/// otherwise the items themselves).
///
/// `descr_sort` (`listobject.py:809`) resolves the sorter class from the
/// list's storage strategy, so an integer-strategy list sorts through
/// `IntSort` (`listobject.py:2429`) and never dispatches a comparison dunder
/// at all; `FloatSort` (`:2434`) is the same for floats, and `SimpleSort`
/// (`:2423`) / `CustomKeySort` (`:2446`) are the generic `space.lt` path.
enum SortCompare {
    Int(usize),
    Float(usize),
    Str(usize),
    Generic(usize),
}

impl crate::listsort::SortLt<usize> for SortCompare {
    fn lt(&mut self, a: usize, b: usize) -> Result<bool, crate::PyError> {
        let (Self::Int(base) | Self::Float(base) | Self::Str(base) | Self::Generic(base)) = *self;
        let left = pyre_object::gc_roots::shadow_stack_get(base + a);
        let right = pyre_object::gc_roots::shadow_stack_get(base + b);
        // Each arm is what `compare_slot` reaches for that exact pair, taken
        // directly: `int_lt`'s `int_value`, `float_lt`'s unwrapped `<`, and the
        // WTF-8 byte order the str arm compares on.
        unsafe {
            match self {
                Self::Int(_) => Ok(crate::objspace::descroperation::int_value(left)
                    < crate::objspace::descroperation::int_value(right)),
                Self::Float(_) => Ok(
                    pyre_object::w_float_get_value(left) < pyre_object::w_float_get_value(right)
                ),
                Self::Str(_) => {
                    Ok(w_str_get_wtf8(left).as_bytes() < w_str_get_wtf8(right).as_bytes())
                }
                Self::Generic(_) => {
                    let result = crate::baseobjspace::compare(
                        left,
                        right,
                        crate::baseobjspace::CompareOp::Lt,
                    )?;
                    crate::baseobjspace::is_true(result)
                }
            }
        }
    }
}

/// Pick the sorter the way `descr_sort` picks it from the list's strategy.
///
/// pyre's list stores plain object references with no strategy to read, so the
/// same decision is one pass over the values.  What makes the specialization
/// equivalent is that a subclass carrying its own `__lt__` must fail the test
/// and take the generic path, exactly as it would keep an object-strategy list
/// upstream — so the test has to be `is_exact_type` (pyobject.rs:179), which
/// compares the instance's `w_class` against the builtin's type object and so
/// rejects a subclass, which retags `w_class` to its own.  `is_int` / `is_str`
/// / `is_float` are NOT usable here: they are `py_type_check`, an `ob_type`
/// layout test a subclass instance also passes because it shares the builtin
/// vtable.  A `key=` sort is `CustomKeySort`, generic upstream as well.
fn sort_compare_for(base: usize, len: usize, keyed: bool) -> SortCompare {
    if keyed {
        return SortCompare::Generic(base);
    }
    let (mut all_int, mut all_float, mut all_str) = (true, true, true);
    for index in 0..len {
        let item = pyre_object::gc_roots::shadow_stack_get(base + index);
        unsafe {
            // `bool` is admitted alongside `int` because it cannot be
            // subclassed, so it can never carry an overriding `__lt__`, and
            // `int_value` reads it the same way.
            all_int &= pyre_object::is_exact_type(item, &pyre_object::INT_TYPE)
                || pyre_object::is_exact_type(item, &pyre_object::BOOL_TYPE);
            all_float &= pyre_object::is_exact_type(item, &pyre_object::FLOAT_TYPE);
            all_str &= pyre_object::is_exact_type(item, &pyre_object::STR_TYPE);
        }
        if !(all_int || all_float || all_str) {
            return SortCompare::Generic(base);
        }
    }
    if all_int {
        SortCompare::Int(base)
    } else if all_float {
        SortCompare::Float(base)
    } else if all_str {
        SortCompare::Str(base)
    } else {
        SortCompare::Generic(base)
    }
}

/// `any(iterable)` — PyPy: operation.py any
/// `any(iterable)` — PyPy: baseobjspace.py any_w
pub fn builtin_any_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_any(args)
}
fn builtin_any(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "any() takes no keyword arguments",
        ));
    }
    if positional.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "any() takes exactly one argument ({} given)",
            positional.len()
        )));
    }
    // `iter` and every `next` run arbitrary Python. Pin the iterable, the
    // iterator, and a dict view's backing dict for the whole walk, and reload
    // the iterator from its slot before each call: a moving minor collection
    // relocates it behind a plain Rust local. The shape `collect_iterator` uses.
    let _roots = pyre_object::gc_roots::push_roots();
    let iterable_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(positional[0]);
    let it = crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(iterable_slot))?;
    let iterator_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(it);
    // The pin queries the collector, so the slot — not this local — is what
    // holds the forwarded iterator once it returns.
    let it = pyre_object::gc_roots::shadow_stack_get(iterator_slot);
    if unsafe { pyre_object::dictmultiobject::is_dict_view_iterator(it) } {
        let w_dict = unsafe { pyre_object::dictmultiobject::w_dict_view_iterator_get_dict(it) };
        pyre_object::gc_roots::pin_root(w_dict);
    }
    loop {
        let it_now = pyre_object::gc_roots::shadow_stack_get(iterator_slot);
        match crate::baseobjspace::next(it_now) {
            Ok(item) if crate::baseobjspace::is_true(item)? => return Ok(w_bool_from(true)),
            Ok(_) => {}
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => return Ok(w_bool_from(false)),
            Err(e) => return Err(e),
        }
    }
}

/// `all(iterable)` — PyPy: operation.py all
/// `all(iterable)` — PyPy: baseobjspace.py all_w
/// Shared file wrapper type — plain instance with hasdict so that
/// open() can attach `__file_data__` / `__file_pos__` / `__file_mode__`
/// as instance attributes, matching the PyPy FileIO/TextIOWrapper
/// duck-typing surface without a dedicated W_FileObject.
pub fn file_wrapper_type() -> PyObjectRef {
    // A Python type is process-global, exactly as PyPy's W_TypeObject is.
    // Storing it in TLS would manufacture one incompatible TextIOWrapper type
    // per host thread and break cross-thread isinstance/type identity.
    static FILE_WRAPPER_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *FILE_WRAPPER_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("_io.TextIOWrapper", init_file_wrapper_type);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
}

/// PyPy: pypy/module/_io/interp_iobase.py W_IOBase.
pub(crate) fn init_file_wrapper_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "read",
            make_builtin_function("read", file_method_read),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "readline",
            make_builtin_function("readline", file_method_readline),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "readlines",
            make_builtin_function("readlines", file_method_readlines),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "write",
            make_builtin_function_with_arity("write", file_method_write, 2),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "writelines",
            make_builtin_function_with_arity(
                "writelines",
                crate::module::_io::iobase_writelines,
                2,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "close",
            make_builtin_function_with_arity("close", file_method_close, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "flush",
            make_builtin_function_with_arity("flush", file_method_flush, 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__enter__",
            make_builtin_function_with_arity("__enter__", |args| Ok(args[0]), 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__exit__",
            make_builtin_function("__exit__", |args| {
                // Call close on exit.
                file_method_close(&args[..1])?;
                Ok(w_none())
            }),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__iter__",
            make_builtin_function_with_arity("__iter__", |args| Ok(args[0]), 1),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "fileno",
            make_builtin_function_with_arity(
                "fileno",
                |args| {
                    let self_obj = args[0];
                    file_check_closed(self_obj)?;
                    match file_get_fd(self_obj) {
                        Some(fd) => Ok(w_int_new(fd as i64)),
                        None => Err(crate::PyError::os_error(
                            "fileno() on a file without a descriptor",
                        )),
                    }
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "readable",
            make_builtin_function_with_arity(
                "readable",
                |args| {
                    file_check_closed(args[0])?;
                    let mode = crate::baseobjspace::getattr_str(args[0], "__file_mode__")
                        .ok()
                        .map(|m| unsafe { pyre_object::w_str_get_value(m).to_string() })
                        .unwrap_or_default();
                    Ok(w_bool_from(mode.contains('r') || mode.contains('+')))
                },
                1,
            ),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "writable",
            make_builtin_function_with_arity(
                "writable",
                |args| {
                    file_check_closed(args[0])?;
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "seekable",
            make_builtin_function_with_arity(
                "seekable",
                // An fd-backed object is seekable iff `lseek` succeeds: a real
                // file does, a pipe/socket fails with ESPIPE.  The in-memory
                // path wrapper is always seekable.
                |args| {
                    file_check_closed(args[0])?;
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "seek",
            make_builtin_function("seek", |args| {
                if args.len() < 2 || args.len() > 3 {
                    return Err(crate::PyError::type_error(format!(
                        "seek() takes from 1 to 2 arguments ({} given)",
                        args.len().saturating_sub(1)
                    )));
                }
                file_check_closed(args[0])?;
                let offset = args
                    .get(1)
                    .map(|&o| unsafe { pyre_object::w_int_get_value(o) })
                    .unwrap_or(0);
                let whence = args
                    .get(2)
                    .map(|&o| unsafe { pyre_object::w_int_get_value(o) })
                    .unwrap_or(0) as i32;
                // CPython 3.14 TextIOWrapper only permits the opaque-cookie
                // forms for current/end-relative seeks; FileIO remains free
                // to use ordinary non-zero offsets.
                if !file_is_binary(args[0]) && offset != 0 {
                    // POSIX/Python's public SEEK_CUR and SEEK_END values are
                    // 1 and 2.  Keep this semantic validation target-neutral;
                    // wasm libc intentionally does not expose lseek constants.
                    if whence == 1 {
                        return Err(crate::module::_io::unsupported(
                            "can't do nonzero cur-relative seeks",
                        ));
                    }
                    if whence == 2 {
                        return Err(crate::module::_io::unsupported(
                            "can't do nonzero end-relative seeks",
                        ));
                    }
                }
                if let Some(fd) = file_get_fd(args[0]) {
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
                let base = match whence {
                    0 => 0,
                    1 => file_get_pos(args[0]) as i64,
                    2 => file_get_data(args[0]).len() as i64,
                    _ => return Err(crate::PyError::value_error("invalid whence")),
                };
                let pos = base
                    .checked_add(offset)
                    .filter(|pos| *pos >= 0)
                    .ok_or_else(|| crate::PyError::value_error("negative seek position"))?;
                file_set_pos(args[0], pos as usize);
                Ok(w_int_new(pos))
            }),
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "tell",
            make_builtin_function_with_arity(
                "tell",
                |args| {
                    file_check_closed(args[0])?;
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
        )
    };
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "isatty",
            make_builtin_function_with_arity("isatty", file_method_isatty, 1),
        )
    };
}

/// PyPy `W_FileIO.typedef` — raw-stream methods override the shared
/// TextIOWrapper-shaped helpers installed by [`init_file_wrapper_type`].
/// Keeping this as a separate initializer mirrors FileIO's distinct typedef
/// instead of teaching the text wrapper about raw-only operations.
pub(crate) fn init_fileio_type(ns: PyObjectRef) {
    for (name, function) in [
        ("read", make_builtin_function("read", fileio_method_read)),
        (
            "readinto",
            make_builtin_function_with_arity("readinto", fileio_method_readinto, 2),
        ),
        (
            "readall",
            make_builtin_function_with_arity("readall", fileio_method_readall, 1),
        ),
        (
            "write",
            make_builtin_function_with_arity("write", fileio_method_write, 2),
        ),
        (
            "truncate",
            make_builtin_function("truncate", fileio_method_truncate),
        ),
    ] {
        unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, function) };
    }
    for (name, getter, doc) in [
        (
            "closed",
            fileio_get_closed as crate::gateway::BuiltinCodeFn,
            Some("True if the file is closed"),
        ),
        (
            "closefd",
            fileio_get_closefd as crate::gateway::BuiltinCodeFn,
            Some("True if the file descriptor will be closed"),
        ),
        (
            "mode",
            fileio_get_mode as crate::gateway::BuiltinCodeFn,
            Some("String giving the file mode"),
        ),
        (
            "_blksize",
            fileio_get_blksize as crate::gateway::BuiltinCodeFn,
            None,
        ),
    ] {
        let getter = make_builtin_function_with_arity(name, getter, 2);
        let descriptor = match doc {
            Some(doc) => crate::typedef::make_getset_property_named_doc(
                getter,
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
                doc,
                name,
            ),
            None => crate::typedef::make_getset_descriptor_named(getter, name),
        };
        unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, descriptor) };
    }
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__repr__",
            make_builtin_function_with_arity("__repr__", fileio_method_repr, 1),
        )
    };
}

fn fileio_get_slot(args: &[PyObjectRef], storage: &str) -> Result<PyObjectRef, crate::PyError> {
    let self_obj = args
        .get(1)
        .copied()
        .ok_or_else(|| crate::PyError::type_error("descriptor requires an instance"))?;
    crate::baseobjspace::getattr_str(self_obj, storage)
}

fn fileio_get_closed(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    fileio_get_slot(args, "__file_closed__")
}

fn fileio_get_closefd(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    fileio_get_slot(args, "__file_closefd__")
}

fn fileio_get_mode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    fileio_get_slot(args, "__file_public_mode__")
}

fn fileio_get_blksize(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    fileio_get_slot(args, "__file_blksize__")
}

/// PyPy `W_FileIO.repr_w`.
fn fileio_method_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("__repr__ requires self"))?;
    let type_name = unsafe {
        crate::typedef::r#type(self_obj)
            .map(|tp| pyre_object::w_type_get_name(tp.as_ptr()).to_string())
            .unwrap_or_else(|| "FileIO".to_string())
    };
    // CPython 3.14's subclass repr names the concrete subclass; the builtin
    // itself retains its public `_io.FileIO` spelling.
    let repr_type = if type_name == "FileIO" {
        "_io.FileIO"
    } else {
        type_name.as_str()
    };
    if file_is_closed(self_obj) {
        return Ok(w_str_new(&format!("<{repr_type} [closed]>")));
    }
    let closefd = if file_closefd(self_obj) {
        "True"
    } else {
        "False"
    };
    let mode = crate::baseobjspace::getattr_str(self_obj, "__file_public_mode__")
        .ok()
        .and_then(|value| unsafe {
            pyre_object::is_str(value).then(|| pyre_object::w_str_get_value(value).to_string())
        })
        .unwrap_or_default();
    let body = if let Ok(name) = crate::baseobjspace::getattr_str(self_obj, "name") {
        format!("name={} mode='{mode}' closefd={closefd}", unsafe {
            crate::display::py_repr(name)?
        })
    } else {
        format!(
            "fd={} mode='{mode}' closefd={closefd}",
            file_get_fd(self_obj).unwrap_or(-1)
        )
    };
    Ok(w_str_new(&format!("<{repr_type} {body}>")))
}

/// `_io.FileIO.__init__` — PyPy `W_FileIO.descr_init`.
///
/// The existing file helpers keep their state in reserved instance slots; a
/// FileIO instance uses the same slots but always exposes a binary raw stream.
pub(crate) fn fileio_init(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["file", "mode", "closefd", "opener"], "FileIO")?;
    let self_obj = pos
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("FileIO.__init__() missing self"))?;
    let file = bind_pos_or_kw(pos, kwargs, 1, "file", "FileIO", 1)?
        .ok_or_else(|| crate::PyError::type_error("FileIO() missing required argument 'file'"))?;
    let mode_obj =
        bind_pos_or_kw(pos, kwargs, 2, "mode", "FileIO", 2)?.unwrap_or_else(|| w_str_new("r"));
    if !unsafe { pyre_object::is_str(mode_obj) } {
        return Err(crate::PyError::type_error(
            "FileIO() argument 'mode' must be str",
        ));
    }
    let mode = crate::baseobjspace::str_utf8_w(mode_obj)?;
    // PyPy `decode_mode`: exactly one r/w/x/a flag, at most one '+', and
    // optional 'b'.  Keep the individual booleans because `_mode()` derives
    // the public canonical mode from capabilities, not from input spelling
    // (`wb+` is reported as `rb+`).
    let mut primary = None;
    let mut updating = false;
    for ch in mode.chars() {
        match ch {
            'r' | 'w' | 'a' | 'x' => {
                if primary.is_some() {
                    return Err(crate::PyError::value_error(
                        "Must have exactly one of read/write/create/append mode",
                    ));
                }
                primary = Some(ch);
            }
            '+' => {
                if updating {
                    return Err(crate::PyError::value_error(
                        "Must have exactly one of read/write/create/append mode",
                    ));
                }
                updating = true;
            }
            'b' => {}
            _ => {
                return Err(crate::PyError::value_error(format!("invalid mode: {mode}")));
            }
        }
    }
    let primary = primary.ok_or_else(|| {
        crate::PyError::value_error("Must have exactly one of read/write/create/append mode")
    })?;
    let raw_mode = format!("{primary}b{}", if updating { "+" } else { "" });
    let binary_mode = match (primary, updating) {
        ('x', false) => "xb",
        ('x', true) => "xb+",
        ('a', false) => "ab",
        ('a', true) => "ab+",
        ('r', false) => "rb",
        ('r', true) | ('w', true) => "rb+",
        ('w', false) => "wb",
        _ => unreachable!(),
    };

    let closefd_obj = bind_pos_or_kw(pos, kwargs, 3, "closefd", "FileIO", 3)?
        .unwrap_or_else(|| w_bool_from(true));
    let closefd = crate::baseobjspace::is_true(closefd_obj)?;
    if unsafe { pyre_object::is_bool(file) } {
        crate::warn::warn_category("bool is used as a file descriptor", "RuntimeWarning", 2)?;
    }
    if !unsafe { pyre_object::is_int(file) } && !closefd {
        return Err(crate::PyError::value_error(
            "Cannot use closefd=False with file name",
        ));
    }
    let opener = bind_pos_or_kw(pos, kwargs, 4, "opener", "FileIO", 4)?.unwrap_or_else(w_none);
    let opened = open_raw_file(&[
        file,
        w_str_new(&raw_mode),
        w_int_new(-1),
        w_none(),
        w_none(),
        w_none(),
        closefd_obj,
        opener,
    ])?;
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(self_obj);
    pyre_object::gc_roots::pin_root(opened);
    for name in [
        "__file_data__",
        "__file_pos__",
        "__file_name__",
        "__file_mode__",
        "__file_binary__",
        "__file_fd__",
        "__file_dirty__",
        "name",
        "encoding",
        "errors",
    ] {
        if let Ok(value) = crate::baseobjspace::getattr_str(opened, name) {
            crate::baseobjspace::setattr_str(self_obj, name, value)?;
        }
    }
    // PyPy keeps these on W_FileIO fields and exposes them through typedef
    // descriptors.  Our generic instance layout stores the corresponding
    // fields under private mapdict names so descriptor writes cannot be
    // shadowed by user attributes.
    if !crate::baseobjspace::setdictvalue(self_obj, "__file_public_mode__", w_str_new(binary_mode))?
        || !crate::baseobjspace::setdictvalue(self_obj, "__file_closefd__", w_bool_from(closefd))?
        || !crate::baseobjspace::setdictvalue(self_obj, "__file_closed__", w_bool_from(false))?
    {
        return Err(crate::PyError::runtime_error(
            "FileIO instance has no state dictionary",
        ));
    }
    #[cfg(not(unix))]
    let blksize = 8192;
    #[cfg(unix)]
    let blksize = file_get_fd(self_obj)
        .and_then(|fd| crate::host_seam::ops::fstat(fd).ok())
        .map(|st| if st.blksize > 1 { st.blksize } else { 8192 })
        .unwrap_or(8192);
    if !crate::baseobjspace::setdictvalue(self_obj, "__file_blksize__", w_int_new(blksize as i64))?
    {
        return Err(crate::PyError::runtime_error(
            "FileIO instance has no state dictionary",
        ));
    }
    // PyPy `W_FileIO.descr_init`: append streams are positioned at EOF
    // immediately, rather than waiting for their first O_APPEND write.
    if primary == 'a' {
        if let Some(fd) = file_get_fd(self_obj) {
            #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
            {
                #[cfg(not(feature = "sandbox"))]
                if unsafe { libc::lseek(fd, 0, libc::SEEK_END) } < 0 {
                    return Err(fd_io_err(std::io::Error::last_os_error()));
                }
                #[cfg(feature = "sandbox")]
                crate::host_seam::ops::lseek(fd, 0, libc::SEEK_END)
                    .map_err(|error| crate::host_seam::seam_os_err(error, ""))?;
            }
        } else {
            file_set_pos(self_obj, file_get_data(self_obj).len());
        }
    }
    Ok(w_none())
}

fn file_is_closed(self_obj: PyObjectRef) -> bool {
    for name in ["__file_closed__", "closed"] {
        if let Ok(value) = crate::baseobjspace::getattr_str(self_obj, name) {
            return unsafe { pyre_object::is_bool(value) && pyre_object::w_bool_get_value(value) };
        }
    }
    false
}

fn file_set_closed(self_obj: PyObjectRef, closed: bool) -> Result<(), crate::PyError> {
    if crate::baseobjspace::getattr_str(self_obj, "__file_closed__").is_ok() {
        if crate::baseobjspace::setdictvalue(self_obj, "__file_closed__", w_bool_from(closed))? {
            return Ok(());
        }
    }
    crate::baseobjspace::setattr_str(self_obj, "closed", w_bool_from(closed)).map(|_| ())
}

fn file_closefd(self_obj: PyObjectRef) -> bool {
    for name in ["__file_closefd__", "closefd"] {
        if let Ok(value) = crate::baseobjspace::getattr_str(self_obj, name) {
            return unsafe { !pyre_object::is_bool(value) || pyre_object::w_bool_get_value(value) };
        }
    }
    true
}

fn file_check_closed(self_obj: PyObjectRef) -> Result<(), crate::PyError> {
    if file_is_closed(self_obj) {
        Err(crate::PyError::value_error("I/O operation on closed file"))
    } else {
        Ok(())
    }
}

fn file_mode_string(self_obj: PyObjectRef) -> String {
    crate::baseobjspace::getattr_str(self_obj, "__file_mode__")
        .ok()
        .and_then(|mode| unsafe {
            pyre_object::is_str(mode).then(|| pyre_object::w_str_get_value(mode).to_string())
        })
        .unwrap_or_default()
}

/// PyPy `W_FileIO._check_readable` / `_check_writable`.
fn file_check_readable(self_obj: PyObjectRef) -> Result<(), crate::PyError> {
    let mode = file_mode_string(self_obj);
    if mode.contains('r') || mode.contains('+') {
        Ok(())
    } else {
        Err(crate::module::_io::unsupported("File not open for reading"))
    }
}

fn file_check_writable(self_obj: PyObjectRef) -> Result<(), crate::PyError> {
    let mode = file_mode_string(self_obj);
    if mode.contains('w') || mode.contains('a') || mode.contains('x') || mode.contains('+') {
        Ok(())
    } else {
        Err(crate::module::_io::unsupported("File not open for writing"))
    }
}

/// One `space.acquire_writebuf` export held for a FileIO `readinto` call.
/// PyPy's `with view:` keeps this lock until after `output_slice`/`c_read`.
pub(crate) struct WritableBuffer {
    _roots: pyre_object::gc_roots::RootScope,
    owner_slot: usize,
    held: bool,
    address: *mut u8,
    length: usize,
}

impl WritableBuffer {
    pub(crate) unsafe fn acquire(obj: PyObjectRef) -> Result<Self, crate::PyError> {
        // `space.acquire_writebuf` owns a traced exporter for the complete
        // `with view:` extent.  Root both the requested object and the
        // concrete storage owner so Python called while the buffer is live
        // cannot leave Drop with a pre-GC address.
        let roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let (data, owner) =
            unsafe { fileio_writebuf(pyre_object::gc_roots::shadow_stack_get(obj_slot))? };
        pyre_object::gc_roots::pin_root(owner);
        let owner_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let owner = pyre_object::gc_roots::shadow_stack_get(owner_slot);
        let held = unsafe { buffer_export_incref(owner) };
        Ok(Self {
            _roots: roots,
            owner_slot,
            held,
            address: data.as_mut_ptr(),
            length: data.len(),
        })
    }

    pub(crate) unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.address, self.length) }
    }
}

impl Drop for WritableBuffer {
    fn drop(&mut self) {
        if self.held {
            let owner = pyre_object::gc_roots::shadow_stack_get(self.owner_slot);
            unsafe { buffer_export_decref(owner) };
        }
    }
}

/// `space.acquire_writebuf` for FileIO.readinto.  These are pyre's native
/// writable exporters; a memoryview contributes its exact contiguous window.
unsafe fn fileio_writebuf(
    obj: PyObjectRef,
) -> Result<(&'static mut [u8], PyObjectRef), crate::PyError> {
    unsafe {
        if pyre_object::bytearrayobject::is_bytearray(obj) {
            return Ok((pyre_object::bytearrayobject::w_bytearray_data_mut(obj), obj));
        }
        if pyre_object::interp_array::is_array(obj) {
            return Ok((
                pyre_object::interp_array::w_array_vec_mut(obj).as_mut_slice(),
                obj,
            ));
        }
        #[cfg(all(unix, not(feature = "sandbox")))]
        if let Some(view) = crate::module::mmap::interp_mmap::mmap_buffer_view(obj) {
            let (address, length, readonly) = view?;
            if !readonly {
                return Ok((
                    std::slice::from_raw_parts_mut(address as *mut u8, length),
                    obj,
                ));
            }
        }
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        if let Some((backing, offset, length, _format, _itemsize, _shape)) =
            crate::module::_ctypes::cdata::cdata_buffer_view(obj)
        {
            let full = pyre_object::bytearrayobject::w_bytearray_data_mut(backing);
            if offset <= full.len() && length <= full.len() - offset {
                return Ok((&mut full[offset..offset + length], backing));
            }
        }
        if pyre_object::memoryview::is_w_memoryview(obj) {
            memoryview_check_released(obj)?;
            if pyre_object::memoryview::w_memoryview_readonly(obj) || !memoryview_contiguity(obj).0
            {
                return Err(crate::PyError::type_error(
                    "readinto() argument must be read-write bytes-like object",
                ));
            }
            let view = pyre_object::memoryview::w_memoryview_view(obj);
            let Some(full) = view.backing().as_bytes_mut() else {
                return Err(crate::PyError::type_error(
                    "readinto() argument must be read-write bytes-like object",
                ));
            };
            let offset = view.offset() as usize;
            let length = pyre_object::memoryview::w_memoryview_length(obj) as usize;
            if offset > full.len() || length > full.len() - offset {
                return Err(crate::PyError::value_error(
                    "memoryview buffer is no longer valid",
                ));
            }
            return Ok((&mut full[offset..offset + length], obj));
        }
        Err(crate::PyError::type_error(
            "readinto() argument must be read-write bytes-like object",
        ))
    }
}

/// PyPy `W_FileIO.read_w` / `readall_w` / `readinto_w` / `write_w`.
fn fileio_method_read(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("read() requires self"))?;
    file_check_closed(self_obj)?;
    file_check_readable(self_obj)?;
    file_method_read(args)
}

fn fileio_method_readall(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error("readall() takes no arguments"));
    }
    fileio_method_read(args)
}

fn fileio_method_readinto(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "readinto() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let self_obj = args[0];
    let buffer_obj = args[1];
    file_check_closed(self_obj)?;
    file_check_readable(self_obj)?;
    let mut buffer = unsafe { WritableBuffer::acquire(buffer_obj) }?;
    let target = unsafe { buffer.as_mut_slice() };

    if let Some(fd) = file_get_fd(self_obj) {
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        {
            match fd_read_into(fd, target) {
                Ok(n) => return Ok(w_int_new(n as i64)),
                Err(err)
                    if err.raw_os_error().is_some_and(|errno| {
                        errno == libc::EAGAIN || errno == libc::EWOULDBLOCK
                    }) =>
                {
                    return Ok(w_none());
                }
                Err(err) => return Err(fd_io_err(err)),
            }
        }
        #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
        {
            let _ = (fd, target);
            return Err(crate::PyError::not_implemented(
                "fd readinto requires host_env feature",
            ));
        }
    }

    let data = file_get_data(self_obj);
    let pos = file_get_pos(self_obj).min(data.len());
    let count = target.len().min(data.len() - pos);
    target[..count].copy_from_slice(&data[pos..pos + count]);
    file_set_pos(self_obj, pos + count);
    Ok(w_int_new(count as i64))
}

fn fileio_method_write(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "write() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    file_check_closed(args[0])?;
    file_check_writable(args[0])?;
    // `space.bufferstr_w`, unlike TextIOWrapper.write, rejects str.
    unsafe { file_write_buffer_bytes(args[1]) }?;
    file_method_write(args)
}

fn fileio_method_truncate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() || args.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "truncate() takes at most one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let self_obj = args[0];
    file_check_closed(self_obj)?;
    file_check_writable(self_obj)?;
    let size_obj = match args.get(1).copied() {
        Some(value) if !unsafe { pyre_object::is_none(value) } => value,
        _ => {
            let value = crate::baseobjspace::call_method(self_obj, "tell", &[]);
            if value.is_null() {
                return Err(crate::call::take_call_error()
                    .unwrap_or_else(|| crate::PyError::runtime_error("tell failed")));
            }
            value
        }
    };
    let index = crate::baseobjspace::space_index(size_obj)?;
    let size = crate::baseobjspace::int_w(index)?;
    if size < 0 {
        #[cfg(unix)]
        let invalid_argument = libc::EINVAL;
        #[cfg(not(unix))]
        let invalid_argument = 22;
        return Err(crate::PyError::os_error_with_errno(
            invalid_argument,
            "Invalid argument",
        ));
    }

    if let Some(fd) = file_get_fd(self_obj) {
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            if unsafe { libc::ftruncate(fd, size as libc::off_t) } < 0 {
                return Err(fd_io_err(std::io::Error::last_os_error()));
            }
            return Ok(index);
        }
        #[cfg(any(not(unix), feature = "sandbox"))]
        {
            let _ = (fd, size);
            return Err(crate::PyError::not_implemented(
                "fd truncate is unavailable on this target",
            ));
        }
    }

    let mut data = file_get_data(self_obj);
    data.resize(size as usize, 0);
    crate::baseobjspace::setattr_str(
        self_obj,
        "__file_data__",
        pyre_object::bytesobject::w_bytes_from_bytes(&data),
    )?;
    crate::baseobjspace::setattr_str(self_obj, "__file_dirty__", w_bool_from(true))?;
    Ok(index)
}

fn file_method_isatty(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let self_obj = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("isatty() requires self"))?;
    file_check_closed(self_obj)?;
    let Some(fd) = file_get_fd(self_obj) else {
        return Ok(w_bool_from(false));
    };
    #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
    {
        return Ok(w_bool_from(unsafe { libc::isatty(fd) } != 0));
    }
    #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
    {
        let _ = fd;
        Ok(w_bool_from(false))
    }
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
    // (baseobjspace.py) that `setattr_str` would itself reach.
    crate::baseobjspace::setdictvalue_native(self_obj, "__file_pos__", w_int_new(pos as i64));
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
        let (n, errno) = crate::module::thread::call_external_function(|| unsafe {
            libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len() as _)
        });
        if n < 0 {
            if errno == libc::EINTR {
                continue;
            }
            return Err(std::io::Error::from_raw_os_error(errno));
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
    crate::PyError::os_error_with_errno(io_error_posix_errno(&e, 5), e.to_string())
}

fn file_method_read(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("read() requires self"));
    }
    file_check_closed(args[0])?;
    file_check_readable(args[0])?;
    if let Some(fd) = file_get_fd(args[0]) {
        let n = match args.get(1).copied() {
            None => None,
            Some(value) if unsafe { pyre_object::is_none(value) } => None,
            Some(value) => {
                let value = crate::baseobjspace::int_w(crate::baseobjspace::space_index(value)?)?;
                (value >= 0).then_some(value as usize)
            }
        };
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
    file_check_closed(args[0])?;
    file_check_readable(args[0])?;
    // Optional size cap (`readline(size)`): stop after `size` bytes even
    // before a newline. A missing or negative size means no cap.
    let max = match args.get(1).copied() {
        None => None,
        Some(value) if unsafe { pyre_object::is_none(value) } => None,
        Some(value) => {
            let value = crate::baseobjspace::int_w(crate::baseobjspace::space_index(value)?)?;
            (value >= 0).then_some(value as usize)
        }
    };
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

/// `_io/interp_fileio.py:write_w` — `space.getarg_w('s*', w_data).as_str()`.
/// The `s*` converter accepts readable buffer exporters, not only bytes.
pub(crate) unsafe fn file_write_buffer_bytes(obj: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    unsafe {
        if pyre_object::bytesobject::is_bytes_like(obj) {
            return Ok(pyre_object::bytesobject::bytes_like_data(obj).to_vec());
        }
        if pyre_object::memoryview::is_w_memoryview(obj) {
            memoryview_check_released(obj)?;
            if !memoryview_contiguity(obj).0 {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::BufferError,
                    "memoryview: underlying buffer is not C-contiguous",
                ));
            }
            return Ok(memoryview_gather_bytes(obj));
        }
        if pyre_object::interp_array::is_array(obj) {
            return Ok(pyre_object::interp_array::w_array_bytes(obj).to_vec());
        }
        Err(crate::PyError::type_error("write() expects str or bytes"))
    }
}

fn file_method_write(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("write() requires (self, data)"));
    }
    file_check_closed(args[0])?;
    file_check_writable(args[0])?;
    if let Some(fd) = file_get_fd(args[0]) {
        let bytes: Vec<u8> = unsafe {
            if pyre_object::is_str(args[1]) {
                // Encode through the stream's codec + error handler; a lone
                // surrogate is routed to the handler (`strict` →
                // UnicodeEncodeError) rather than panicking in
                // `w_str_get_value`.
                let (encoding, errors) = stream_encoding_errors(args[0]);
                crate::type_methods::encode_object(args[1], &encoding, &errors)?
            } else {
                file_write_buffer_bytes(args[1])?
            }
        };
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        {
            #[cfg(not(feature = "sandbox"))]
            let n = {
                // `count` is `size_t` on Unix but `c_uint` on Windows.
                let (n, errno) = crate::module::thread::call_external_function(|| unsafe {
                    libc::write(fd, bytes.as_ptr() as *const libc::c_void, bytes.len() as _)
                });
                if n < 0 {
                    return Err(fd_io_err(std::io::Error::from_raw_os_error(errno)));
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
        } else {
            let data = file_write_buffer_bytes(args[1])?;
            let len = data.len();
            (data, len)
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
        let already = file_is_closed(args[0]);
        if !already {
            // Mark closed first so the fd is not reusable even if the underlying
            // close reports an error, then surface that error (matching
            // _io.FileIO.close).
            file_set_closed(args[0], true)?;
            let closefd = file_closefd(args[0]);
            if !closefd {
                return Ok(w_none());
            }
            #[cfg(all(
                feature = "host_env",
                not(target_arch = "wasm32"),
                not(feature = "sandbox")
            ))]
            // SAFETY: close(2) on the file object's own fd.
            if unsafe { libc::close(fd) } < 0 {
                let e = std::io::Error::last_os_error();
                return Err(crate::PyError::os_error_with_errno(
                    io_error_posix_errno(&e, 0),
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
    file_set_closed(args[0], true)?;
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
                    io_error_posix_errno(&e, 5),
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
    file_check_closed(args[0])?;
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

/// PyPy `W_FileIO.descr_init`: immediately fstat every supplied/opened fd and
/// reject directory descriptors before publishing the FileIO object.
#[cfg(unix)]
fn fileio_validate_fd(
    fd: i32,
    w_name: PyObjectRef,
) -> Result<crate::host_seam::StatBuf, crate::PyError> {
    let st = crate::host_seam::ops::fstat(fd)
        .map_err(|error| crate::host_seam::seam_os_err(error, ""))?;
    #[cfg(unix)]
    if st.mode & libc::S_IFMT as u32 == libc::S_IFDIR as u32 {
        return Err(crate::PyError::os_error_syscall(libc::EISDIR, w_name));
    }
    Ok(st)
}

#[cfg(unix)]
fn fileio_close_owned_fd(fd: i32) {
    let _ = crate::host_seam::ops::close(fd);
}

/// `open()` — PyPy `pypy/module/_io/interp_io.py:open`.
///
/// Keep the upstream construction order literal: validate the mode, create a
/// `FileIO`, choose exactly one buffered class, and only then add the text
/// wrapper.  In particular, binary unbuffered I/O returns the raw `FileIO`.
pub fn builtin_open(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    // Every declared slot binds before the unrecognized keywords are
    // reported, so a call missing `file` names `file`.
    let file = bind_pos_or_kw(positional, kwargs, 0, "file", "open", 1)?.ok_or_else(|| {
        crate::PyError::type_error("open() missing required argument 'file' (pos 1)")
    })?;
    kwarg_reject_unknown(
        kwargs,
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
    let w_mode =
        bind_pos_or_kw(positional, kwargs, 1, "mode", "open", 2)?.unwrap_or_else(|| w_str_new("r"));
    if unsafe { !pyre_object::is_str(w_mode) } {
        return Err(crate::PyError::type_error(format!(
            "open() argument 'mode' must be str, not {}",
            crate::type_methods::arg_type_name(w_mode)
        )));
    }
    let mode = crate::baseobjspace::str_utf8_w(w_mode)?.to_string();
    let w_buffering = bind_pos_or_kw(positional, kwargs, 2, "buffering", "open", 3)?
        .unwrap_or_else(|| w_int_new(-1));
    // A buffering value outside the machine-int range is an OverflowError, not a
    // silent fallback to default buffering, so index through `space_index_w`
    // rather than the negative-preserving sentinel converter.
    let mut buffering = crate::builtins::space_index_w(w_buffering)?;
    let w_encoding =
        bind_pos_or_kw(positional, kwargs, 3, "encoding", "open", 4)?.unwrap_or_else(w_none);
    let w_errors =
        bind_pos_or_kw(positional, kwargs, 4, "errors", "open", 5)?.unwrap_or_else(w_none);
    let w_newline =
        bind_pos_or_kw(positional, kwargs, 5, "newline", "open", 6)?.unwrap_or_else(w_none);
    let w_closefd = bind_pos_or_kw(positional, kwargs, 6, "closefd", "open", 7)?
        .unwrap_or_else(|| w_bool_from(true));
    let w_opener =
        bind_pos_or_kw(positional, kwargs, 7, "opener", "open", 8)?.unwrap_or_else(w_none);

    for (name, value) in [
        ("encoding", w_encoding),
        ("errors", w_errors),
        ("newline", w_newline),
    ] {
        if unsafe { !pyre_object::is_none(value) && !pyre_object::is_str(value) } {
            return Err(crate::PyError::type_error(format!(
                "open() argument '{name}' must be str or None, not {}",
                crate::type_methods::arg_type_name(value)
            )));
        }
    }

    let mut reading = false;
    let mut writing = false;
    let mut appending = false;
    let mut exclusive = false;
    let mut updating = false;
    let mut text = false;
    let mut binary = false;
    let mut seen = String::new();
    for flag in mode.chars() {
        if seen.contains(flag) {
            return Err(crate::PyError::value_error(format!(
                "invalid mode: '{mode}'"
            )));
        }
        seen.push(flag);
        match flag {
            'r' => reading = true,
            'w' => writing = true,
            'a' => appending = true,
            'x' => exclusive = true,
            '+' => updating = true,
            't' => text = true,
            'b' => binary = true,
            _ => {
                return Err(crate::PyError::value_error(format!(
                    "invalid mode: '{mode}'"
                )));
            }
        }
    }
    if text && binary {
        return Err(crate::PyError::value_error(
            "can't have text and binary mode at once",
        ));
    }
    if [reading, writing, appending, exclusive]
        .into_iter()
        .filter(|flag| *flag)
        .count()
        != 1
    {
        return Err(crate::PyError::value_error(
            "must have exactly one of create/read/write/append mode",
        ));
    }
    if binary && unsafe { !pyre_object::is_none(w_encoding) } {
        return Err(crate::PyError::value_error(
            "binary mode doesn't take an encoding argument",
        ));
    }
    if binary && unsafe { !pyre_object::is_none(w_errors) } {
        return Err(crate::PyError::value_error(
            "binary mode doesn't take an errors argument",
        ));
    }
    if binary && unsafe { !pyre_object::is_none(w_newline) } {
        return Err(crate::PyError::value_error(
            "binary mode doesn't take a newline argument",
        ));
    }

    let primary = if reading {
        'r'
    } else if writing {
        'w'
    } else if appending {
        'a'
    } else {
        'x'
    };
    let raw_mode = format!("{primary}{}", if updating { "+" } else { "" });
    let raw = crate::call::call_function_impl_result(
        crate::module::_io::fileio_type(),
        &[file, w_str_new(&raw_mode), w_closefd, w_opener],
    )?;
    let roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(raw);
    let raw_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    // `_open` closes the outermost constructed layer when a later layer's
    // constructor raises, so a failed `open()` never leaks the descriptor.
    let mut close_target = pyre_object::gc_roots::shadow_stack_get(raw_slot);
    let outcome = (|| -> Result<PyObjectRef, crate::PyError> {
        let isatty = crate::baseobjspace::call_method(
            pyre_object::gc_roots::shadow_stack_get(raw_slot),
            "isatty",
            &[],
        );
        if isatty.is_null() {
            return Err(crate::call::take_call_error()
                .unwrap_or_else(|| crate::PyError::runtime_error("isatty failed")));
        }
        let line_buffering =
            buffering == 1 || (buffering < 0 && crate::baseobjspace::is_true(isatty)?);
        if line_buffering {
            buffering = -1;
        }
        if buffering < 0 {
            buffering = crate::baseobjspace::getattr_str(
                pyre_object::gc_roots::shadow_stack_get(raw_slot),
                "_blksize",
            )
            .ok()
            .and_then(|value| crate::baseobjspace::int_w(value).ok())
            .filter(|size| *size > 1)
            .unwrap_or(crate::module::_io::DEFAULT_BUFFER_SIZE);
        }
        if buffering == 0 {
            if !binary {
                return Err(crate::PyError::value_error(
                    "can't have unbuffered text I/O",
                ));
            }
            return Ok(pyre_object::gc_roots::shadow_stack_get(raw_slot));
        }

        let buffer_type = if updating {
            crate::module::_io::buffered_random_type()
        } else if writing || appending || exclusive {
            crate::module::_io::buffered_writer_type()
        } else {
            crate::module::_io::buffered_reader_type()
        };
        let buffer = crate::call::call_function_impl_result(
            buffer_type,
            &[
                pyre_object::gc_roots::shadow_stack_get(raw_slot),
                w_int_new(buffering),
            ],
        )?;
        pyre_object::gc_roots::pin_root(buffer);
        let buffer_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        close_target = pyre_object::gc_roots::shadow_stack_get(buffer_slot);
        if binary {
            return Ok(pyre_object::gc_roots::shadow_stack_get(buffer_slot));
        }

        let wrapper = crate::call::call_function_impl_result(
            text_io_wrapper_type(),
            &[
                pyre_object::gc_roots::shadow_stack_get(buffer_slot),
                w_encoding,
                w_errors,
                w_newline,
                w_bool_from(line_buffering),
            ],
        )?;
        crate::baseobjspace::setattr_str(wrapper, "mode", w_str_new(&mode))?;
        Ok(wrapper)
    })();
    let result = match outcome {
        Ok(object) => Ok(object),
        Err(error) => {
            // The original error takes precedence; a `close` failure here is
            // discarded (its call-error slot is cleared so it cannot leak).
            if crate::baseobjspace::call_method(close_target, "close", &[]).is_null() {
                let _ = crate::call::take_call_error();
            }
            Err(error)
        }
    };
    drop(roots);
    result
}

/// Low-level storage opener used by `W_FileIO.descr_init`.
///
/// This is pyre's `_open_fd`/raw-storage boundary.  It deliberately does not
/// assemble buffered or text layers; public `open()` below does that after it
/// has constructed the real `_io.FileIO` object.
fn open_raw_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "open() missing required argument 'file' (pos 1)",
        ));
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
        .ok_or_else(|| {
            crate::PyError::type_error("open() missing required argument 'file' (pos 1)")
        })?;
    let mode_obj = resolve_pos_or_kw(open_pos.get(1).copied(), open_kwargs, "mode", "open", 2)?;
    let encoding_obj =
        resolve_pos_or_kw(open_pos.get(3).copied(), open_kwargs, "encoding", "open", 4)?;
    let errors_obj = resolve_pos_or_kw(open_pos.get(4).copied(), open_kwargs, "errors", "open", 5)?;
    let closefd_obj =
        resolve_pos_or_kw(open_pos.get(6).copied(), open_kwargs, "closefd", "open", 7)?;
    let opener_obj = resolve_pos_or_kw(open_pos.get(7).copied(), open_kwargs, "opener", "open", 8)?;
    let str_or_none =
        |obj: Option<PyObjectRef>, name: &str| -> Result<Option<String>, crate::PyError> {
            match obj {
                Some(o) if unsafe { pyre_object::is_str(o) } => Ok(Some(
                    crate::baseobjspace::str_utf8_w(o)?
                        .to_ascii_lowercase()
                        .replace('_', "-"),
                )),
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
        Some(m) if unsafe { pyre_object::is_str(m) } => {
            crate::baseobjspace::str_utf8_w(m)?.to_string()
        }
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
        let fd_value = unsafe { pyre_object::w_int_get_value(path_obj) };
        let fd = i32::try_from(fd_value).map_err(|_| {
            crate::PyError::overflow_error("Python int too large to convert to C int")
        })?;
        if fd < 0 {
            return Err(crate::PyError::value_error("negative file descriptor"));
        }
        #[cfg(unix)]
        fileio_validate_fd(fd, path_obj)?;
        let closefd = match closefd_obj {
            Some(value) => crate::baseobjspace::is_true(value)?,
            None => true,
        };
        let binary = mode.contains('b');
        let wrapper = pyre_object::w_instance_new(file_wrapper_type());
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_fd__", w_int_new(fd as i64));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_binary__", w_bool_from(binary));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_mode__", w_str_new(&mode));
        let _ = crate::baseobjspace::setattr_str(wrapper, "encoding", w_str_new(&encoding));
        let _ = crate::baseobjspace::setattr_str(wrapper, "errors", w_str_new(&errors));
        let _ = crate::baseobjspace::setattr_str(wrapper, "name", w_int_new(fd as i64));
        let _ = crate::baseobjspace::setattr_str(wrapper, "mode", w_str_new(&mode));
        let _ = crate::baseobjspace::setattr_str(wrapper, "closefd", w_bool_from(closefd));
        let _ = crate::baseobjspace::setattr_str(wrapper, "closed", w_bool_from(false));
        return Ok(wrapper);
    }

    if closefd_obj.map(crate::baseobjspace::is_true).transpose()? == Some(false) {
        return Err(crate::PyError::value_error(
            "Cannot use closefd=False with file name",
        ));
    }

    // The OS path is a byte string.  A `str` path may carry surrogateescape
    // code points, which spell bytes that are not valid UTF-8; folding those
    // to U+FFFD would open a different file, so the encoded bytes are kept
    // and handed to the seam verbatim.
    let path_bytes = unsafe {
        if pyre_object::is_str(path_obj) {
            crate::gateway::fsencode_bytes_w(path_obj)?
        } else if pyre_object::bytesobject::is_bytes_like(path_obj) {
            pyre_object::bytesobject::bytes_like_data(path_obj).to_vec()
        } else if let Some(fspath_fn) = crate::typedef::r#type(path_obj)
            .and_then(|pt| crate::baseobjspace::lookup_in_type(pt.as_ptr(), "__fspath__"))
        {
            // `type(path).__fspath__(path)` — unbound descriptor + single arg.
            let result = crate::call::call_function_impl_result(fspath_fn, &[path_obj])?;
            if pyre_object::is_str(result) {
                crate::gateway::fsencode_bytes_w(result)?
            } else if pyre_object::bytesobject::is_bytes_like(result) {
                pyre_object::bytesobject::bytes_like_data(result).to_vec()
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
    // `host_env::fs` takes a `Path`; only the seam consumes the raw bytes.
    #[cfg(unix)]
    let path = std::ffi::OsString::from_vec(path_bytes.clone());
    #[cfg(not(unix))]
    let path = String::from_utf8_lossy(&path_bytes).into_owned();
    let binary = mode.contains('b');
    let writing = mode.contains('w') || mode.contains('a') || mode.contains('x');
    let reading = mode.contains('r') || !writing;

    // `open(..., opener=callable)`: the opener supplies the file descriptor
    // (e.g. `tempfile.NamedTemporaryFile` creates the temp file and records
    // its name in the opener). Call it with `(file, flags)` and wrap the
    // returned fd directly.
    if let Some(opener) = opener_obj {
        if !unsafe { pyre_object::is_none(opener) } {
            let flags = open_flags_for_mode(&mode);
            let fd_obj = crate::call::call_function_impl_result(
                opener,
                &[path_obj, w_int_new(flags as i64)],
            )?;
            if !unsafe { pyre_object::is_int(fd_obj) } {
                return Err(crate::PyError::type_error("expected integer from opener"));
            }
            let fd_value = unsafe { pyre_object::w_int_get_value(fd_obj) };
            let fd = i32::try_from(fd_value).map_err(|_| {
                crate::PyError::overflow_error("Python int too large to convert to C int")
            })?;
            if fd < 0 {
                return Err(crate::PyError::value_error(format!("opener returned {fd}")));
            }
            #[cfg(unix)]
            if let Err(error) = fileio_validate_fd(fd, path_obj) {
                fileio_close_owned_fd(fd);
                return Err(error);
            }
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
        let path_display = String::from_utf8_lossy(&path_bytes);
        let fd = crate::host_seam::ops::open(&path_bytes, flags, 0o666)
            .map_err(|e| crate::host_seam::seam_os_err(e, &path_display))?;
        if let Err(error) = fileio_validate_fd(fd, path_obj) {
            fileio_close_owned_fd(fd);
            return Err(error);
        }
        let wrapper = pyre_object::w_instance_new(file_wrapper_type());
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_fd__", w_int_new(fd as i64));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_binary__", w_bool_from(binary));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_mode__", w_str_new(&mode));
        let _ = crate::baseobjspace::setattr_str(wrapper, "encoding", w_str_new(&encoding));
        let _ = crate::baseobjspace::setattr_str(wrapper, "errors", w_str_new(&errors));
        let _ = crate::baseobjspace::setattr_str(wrapper, "name", path_obj);
        let _ = crate::baseobjspace::setattr_str(wrapper, "mode", w_str_new(&mode));
        let _ = crate::baseobjspace::setattr_str(wrapper, "closefd", w_bool_from(true));
        let _ = crate::baseobjspace::setattr_str(wrapper, "closed", w_bool_from(false));
        Ok(wrapper)
    }
    #[cfg(all(not(feature = "sandbox"), unix))]
    {
        // PyPy: interp_io._open constructs W_FileIO, whose descr_init opens a
        // pathname immediately through _open_fd.  In particular, `w` creates
        // and truncates before open() returns, `x` reports EEXIST here, and
        // `a` owns an O_APPEND descriptor.  Keeping pathname-backed streams in
        // an in-memory side buffer postpones all three observable effects until
        // close(), which is not the W_FileIO storage shape.
        let _ = (reading, writing);
        let flags = open_flags_for_mode(&mode);
        let path_display = String::from_utf8_lossy(&path_bytes);
        let fd = crate::host_seam::ops::open(&path_bytes, flags, 0o666)
            .map_err(|e| crate::host_seam::seam_os_err(e, &path_display))?;
        if let Err(error) = fileio_validate_fd(fd, path_obj) {
            fileio_close_owned_fd(fd);
            return Err(error);
        }
        let wrapper = pyre_object::w_instance_new(file_wrapper_type());
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_fd__", w_int_new(fd as i64));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_binary__", w_bool_from(binary));
        let _ = crate::baseobjspace::setattr_str(wrapper, "__file_mode__", w_str_new(&mode));
        let _ = crate::baseobjspace::setattr_str(wrapper, "encoding", w_str_new(&encoding));
        let _ = crate::baseobjspace::setattr_str(wrapper, "errors", w_str_new(&errors));
        let _ = crate::baseobjspace::setattr_str(wrapper, "name", path_obj);
        let _ = crate::baseobjspace::setattr_str(wrapper, "mode", w_str_new(&mode));
        let _ = crate::baseobjspace::setattr_str(wrapper, "closefd", w_bool_from(true));
        let _ = crate::baseobjspace::setattr_str(wrapper, "closed", w_bool_from(false));
        Ok(wrapper)
    }
    #[cfg(all(not(feature = "sandbox"), not(unix)))]
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
                    // Report the original path object as `.filename`, keeping a
                    // bytes path a bytes object (`interp_fileio.py` wraps the raw
                    // `w_name`, not the decoded buffer).
                    return Err(crate::PyError::os_error_syscall(
                        io_error_posix_errno(&e, 2),
                        path_obj,
                    ));
                }
            }
        } else {
            Vec::new()
        };

        // `W_FileIO.descr_init` opens a pathname immediately: `w`/`w+` create
        // and truncate, `x` reports EEXIST, `a` creates without truncating.
        // The in-memory buffer above otherwise postpones the on-disk file to
        // the first flush, so `open(p, "w").close()` (no write) would never
        // create the file and `x` would not enforce exclusivity.  Perform the
        // eager open-time create here to match the fd-backed path.
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        if writing {
            let create_res = if mode.contains('x') {
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .map(|_| ())
            } else if mode.contains('a') {
                std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(&path)
                    .map(|_| ())
            } else {
                std::fs::write(&path, b"")
            };
            if let Err(e) = create_res {
                let errno = match e.kind() {
                    std::io::ErrorKind::AlreadyExists => libc::EEXIST,
                    std::io::ErrorKind::NotFound => libc::ENOENT,
                    std::io::ErrorKind::PermissionDenied => libc::EACCES,
                    _ => io_error_posix_errno(&e, libc::EACCES),
                };
                return Err(crate::PyError::os_error_syscall(errno, path_obj));
            }
        }

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

/// Compatibility entry point used by the builtin `open()` pipeline.
pub fn text_io_wrapper_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::call::call_function_impl_result(text_io_wrapper_type(), args)
}

/// Process-global `_io.TextIOWrapper` type owned by the `_io` module.
pub fn text_io_wrapper_type() -> PyObjectRef {
    crate::module::_io::text_io_wrapper_type()
}
pub fn builtin_all_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_all(args)
}
fn builtin_all(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "all() takes no keyword arguments",
        ));
    }
    if positional.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "all() takes exactly one argument ({} given)",
            positional.len()
        )));
    }
    // Rooted exactly as `builtin_any`: `iter` and every `next` run arbitrary
    // Python, so the iterator is reloaded from its slot before each call.
    let _roots = pyre_object::gc_roots::push_roots();
    let iterable_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(positional[0]);
    let it = crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(iterable_slot))?;
    let iterator_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(it);
    // The pin queries the collector, so the slot — not this local — is what
    // holds the forwarded iterator once it returns.
    let it = pyre_object::gc_roots::shadow_stack_get(iterator_slot);
    if unsafe { pyre_object::dictmultiobject::is_dict_view_iterator(it) } {
        let w_dict = unsafe { pyre_object::dictmultiobject::w_dict_view_iterator_get_dict(it) };
        pyre_object::gc_roots::pin_root(w_dict);
    }
    loop {
        let it_now = pyre_object::gc_roots::shadow_stack_get(iterator_slot);
        match crate::baseobjspace::next(it_now) {
            Ok(item) if !crate::baseobjspace::is_true(item)? => return Ok(w_bool_from(false)),
            Ok(_) => {}
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => return Ok(w_bool_from(true)),
            Err(e) => return Err(e),
        }
    }
}

/// `sum(sequence, start=0)` — PyPy `__builtin__/app_functional.py sum`.
///
/// A left-fold through `space.add` (`_regular_sum`'s `last = last + x`) while
/// the running total is an exact int, then the improved Kahan-Babuška
/// (Neumaier) compensated float accumulator `builtin_sum_impl` uses — so
/// `sum([0.1] * 10)` is exactly `1.0` rather than the naive partial sum
/// `functional.py:_sum` produces.  A `str`/`bytes`/`bytearray` `start` is
/// rejected up front.
#[derive(Clone, Copy)]
struct CompensatedSum {
    hi: f64,
    lo: f64,
}

/// CPython 3.14 `cs_from_double`.
#[inline]
fn compensated_sum_from_double(value: f64) -> CompensatedSum {
    CompensatedSum { hi: value, lo: 0.0 }
}

/// CPython 3.14 `cs_add` (improved Kahan-Babuška / Neumaier summation).
#[inline]
fn compensated_sum_add(mut total: CompensatedSum, value: f64) -> CompensatedSum {
    let next = total.hi + value;
    total.lo += if total.hi.abs() >= value.abs() {
        (total.hi - next) + value
    } else {
        (value - next) + total.hi
    };
    CompensatedSum {
        hi: next,
        lo: total.lo,
    }
}

/// CPython 3.14 `cs_to_double`.
#[inline]
fn compensated_sum_to_double(total: CompensatedSum) -> f64 {
    // Skipping an exact zero preserves a negative high part: `-0.0 + +0.0`
    // would otherwise erase its sign.  A non-finite residual is skipped so
    // `inf - inf` cannot turn an infinite running total into NaN.
    if total.lo != 0.0 && total.lo.is_finite() {
        total.hi + total.lo
    } else {
        total.hi
    }
}

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
    let items = crate::builtins::collect_iterable(iterable)?;
    let mut last = start;
    let mut i = 0;
    // `builtin_sum_impl` runs an exact-int accumulator until the running
    // total turns into an exact float, then a float accumulator that carries
    // the improved Kahan-Babuška (Neumaier) compensation term — so
    // `sum([0.1] * 10)` is exactly `1.0` rather than the naive partial sum
    // `functional.py:_sum` produces.  The int phase is the generic `last + x`
    // loop, which already keeps exact ints exact and promotes to a float on
    // the first float item, so only the float phase needs its own arithmetic.
    while i < items.len()
        && unsafe { is_exact_int_operand(last) }
        && !unsafe { is_exact_float_operand(last) }
    {
        last = crate::baseobjspace::add(last, items[i])?;
        i += 1;
    }
    if unsafe { is_exact_float_operand(last) } {
        let mut real_sum =
            compensated_sum_from_double(unsafe { pyre_object::w_float_get_value(last) });
        while i < items.len() {
            let item = items[i];
            // A float *subclass* leaves the fast path — its `__add__` may be
            // overridden — while `bool` and an `int` subclass stay in it,
            // matching the `PyFloat_CheckExact` / `PyLong_Check` asymmetry of
            // the C loop.  An int too wide for a machine word leaves it too,
            // the way `PyLong_AsLongAndOverflow` signals overflow.
            let x = unsafe {
                if pyre_object::is_float(item) && pyre_object::is_exact_builtin_instance(item) {
                    pyre_object::w_float_get_value(item)
                } else if pyre_object::pyobject::is_int(item) {
                    pyre_object::w_int_get_value(item) as f64
                } else {
                    break;
                }
            };
            real_sum = compensated_sum_add(real_sum, x);
            i += 1;
        }
        last = pyre_object::w_float_new(compensated_sum_to_double(real_sum));
    }
    // CPython 3.14's complex fast path carries independent compensated real
    // and imaginary accumulators.  Exact complex items, all ints, and all
    // floats stay on the path; an arbitrary numeric object falls back to the
    // generic left-fold below.
    if unsafe { is_exact_complex_operand(last) } {
        let mut real_sum =
            compensated_sum_from_double(unsafe { pyre_object::w_complex_get_real(last) });
        let mut imag_sum =
            compensated_sum_from_double(unsafe { pyre_object::w_complex_get_imag(last) });
        while i < items.len() {
            let item = items[i];
            unsafe {
                if is_exact_complex_operand(item) {
                    real_sum = compensated_sum_add(real_sum, pyre_object::w_complex_get_real(item));
                    imag_sum = compensated_sum_add(imag_sum, pyre_object::w_complex_get_imag(item));
                } else if pyre_object::pyobject::is_int(item) {
                    real_sum =
                        compensated_sum_add(real_sum, pyre_object::w_int_get_value(item) as f64);
                } else if pyre_object::is_float(item) {
                    real_sum = compensated_sum_add(real_sum, pyre_object::w_float_get_value(item));
                } else {
                    break;
                }
            }
            i += 1;
        }
        last = pyre_object::w_complex_new(
            compensated_sum_to_double(real_sum),
            compensated_sum_to_double(imag_sum),
        );
    }
    for &item in &items[i..] {
        last = crate::baseobjspace::add(last, item)?;
    }
    Ok(last)
}

/// `PyLong_CheckExact` — an exact `int`, excluding `bool` and any subclass.
unsafe fn is_exact_int_operand(obj: PyObjectRef) -> bool {
    unsafe {
        pyre_object::pyobject::is_int_or_long(obj)
            && !pyre_object::pyobject::is_bool(obj)
            && pyre_object::is_exact_builtin_instance(obj)
    }
}

/// `PyFloat_CheckExact` — an exact `float`, excluding any subclass.
unsafe fn is_exact_float_operand(obj: PyObjectRef) -> bool {
    unsafe { pyre_object::is_float(obj) && pyre_object::is_exact_builtin_instance(obj) }
}

/// `PyComplex_CheckExact` — an exact `complex`, excluding any subclass.
unsafe fn is_exact_complex_operand(obj: PyObjectRef) -> bool {
    unsafe { pyre_object::is_complex(obj) && pyre_object::is_exact_builtin_instance(obj) }
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
    let obj = bind_pos_or_kw(pos, kwargs, 0, "number", "round", 1)?.ok_or_else(|| {
        crate::PyError::type_error("round() missing required argument 'number' (pos 1)")
    })?;
    kwarg_reject_unknown(kwargs, &["number", "ndigits"], "round")?;
    let ndigits_arg = bind_pos_or_kw(pos, kwargs, 1, "ndigits", "round", 2)?;
    let ndigits = ndigits_arg.as_ref();
    unsafe {
        if is_float(obj) {
            let v = floatobject::w_float_get_value(obj);
            return match ndigits {
                // `floatobject.py:966-967 _round_float`: nan/inf round to
                // themselves when an explicit ndigits is supplied.  `ndigits`
                // is taken through `space.getindex_w(w, None)`, so any
                // `__index__` object works, an out-of-word value clamps, and a
                // non-index one raises.
                Some(nd) if !pyre_object::is_none(*nd) => {
                    let n = crate::baseobjspace::getindex_w(*nd)?;
                    if !v.is_finite() {
                        Ok(floatobject::w_float_new(v))
                    } else {
                        let rounded = float_round_ndigits(v, n);
                        if rounded.is_infinite() {
                            // CPython 3.14 `double_round`: PyPy performs the
                            // same post-rounding infinity check, but retains
                            // its older error text.
                            Err(crate::PyError::overflow_error(
                                "rounded value too large to represent",
                            ))
                        } else {
                            Ok(floatobject::w_float_new(rounded))
                        }
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
            // intobject.py:144-175 `descr_round`: keep ndigits as an rbigint
            // obtained through `space.index`, then compute
            // `10 ** -ndigits` and divmod-near.  Converting ndigits to an
            // index-sized Rust word or formatting the operand merely to count
            // decimal digits both diverge from that consumer shape.
            let nd = match ndigits {
                Some(nd) if !pyre_object::is_none(*nd) => index_to_bigint(*nd)?,
                _ => return Ok(obj),
            };
            if nd.get_sign() >= 0 {
                return Ok(obj);
            }
            let owned_a;
            let a = if is_long(obj) {
                w_long_get_value(obj)
            } else {
                owned_a = BigInt::from(w_int_get_value(obj));
                &owned_a
            };
            let exponent = nd.neg();
            let b = BigInt::from(10)
                .pow(&exponent, None)
                .map_err(|_| crate::PyError::memory_error("exponent too large"))?;
            // `_PyLong_DivmodNear`: q = round(a / b) ties-to-even,
            // result = q * b.  Floor division gives 0 <= r < b.
            let (q, r) = a
                .divmod(&b)
                .expect("round's positive power-of-ten divisor is nonzero");
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
        if let Some(method) =
            unsafe { crate::baseobjspace::lookup_in_type(tp.as_ptr(), "__round__") }
        {
            let result = unsafe {
                match ndigits {
                    Some(nd) if !pyre_object::is_none(*nd) => {
                        crate::baseobjspace::get_and_call_function(
                            method,
                            obj,
                            tp.as_ptr(),
                            &[*nd],
                        )?
                    }
                    _ => crate::baseobjspace::get_and_call_function(method, obj, tp.as_ptr(), &[])?,
                }
            };
            return Ok(result);
        }
    }
    let type_name = match crate::typedef::r#type(obj) {
        Some(tp) => unsafe { pyre_object::w_type_get_name(tp.as_ptr()).to_string() },
        None => unsafe { (*(*obj).ob_type).name.to_string() },
    };
    Err(crate::PyError::type_error(format!(
        "type {} doesn't define __round__ method",
        type_name
    )))
}

/// True iff `callable` is the canonical builtin `divmod` function object.
/// The JIT walker uses the builtin-code identity to recognize a `divmod(a, b)`
/// residual it can lower to the inline pair shape, and to distinguish it from
/// an arbitrary replacement stored under the same global name.
pub fn is_builtin_divmod_function(callable: PyObjectRef) -> bool {
    unsafe {
        if callable.is_null() || !crate::is_function(callable) {
            return false;
        }
        let code = crate::function_get_code(callable) as PyObjectRef;
        if code.is_null() || !crate::gateway::is_builtin_code(code) {
            return false;
        }
        crate::gateway::builtin_code_fn_eq(
            crate::gateway::builtin_code_get(code),
            builtin_divmod as crate::gateway::BuiltinCodeFn,
        )
    }
}

/// `divmod(a, b)` — pypy/interpreter/baseobjspace.py divmod row.
fn builtin_divmod(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (args, kwargs) = split_builtin_kwargs(args);
    if has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "divmod() takes no keyword arguments",
        ));
    }
    if args.len() != 2 {
        // `_PyArg_CheckPositional` names the function bare and parenthesis-free
        // once it declares two or more arguments.
        return Err(crate::PyError::type_error(format!(
            "divmod expected 2 arguments, got {}",
            args.len()
        )));
    }
    crate::baseobjspace::divmod(args[0], args[1])
}

/// `pow(base, exp[, mod])` — pypy/interpreter/baseobjspace.py pow row.
fn builtin_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `pow(base, exp, mod=None)`: all positional-or-keyword; at most three.
    let (pos, kwargs) = split_builtin_kwargs(args);
    let total = pos.len() + real_kwarg_count(kwargs);
    if total > 3 {
        return Err(crate::PyError::type_error(format!(
            "pow() takes at most 3 arguments ({total} given)"
        )));
    }
    let base = bind_pos_or_kw(pos, kwargs, 0, "base", "pow", 1)?.ok_or_else(|| {
        crate::PyError::type_error("pow() missing required argument 'base' (pos 1)")
    })?;
    let exp = bind_pos_or_kw(pos, kwargs, 1, "exp", "pow", 2)?.ok_or_else(|| {
        crate::PyError::type_error("pow() missing required argument 'exp' (pos 2)")
    })?;
    kwarg_reject_unknown(kwargs, &["base", "exp", "mod"], "pow")?;
    let modulus = bind_pos_or_kw(pos, kwargs, 2, "mod", "pow", 3)?;
    match modulus {
        Some(m) if !unsafe { pyre_object::is_none(m) } => crate::baseobjspace::pow3(base, exp, m),
        _ => crate::baseobjspace::pow(base, exp),
    }
}

/// Coerce `obj` to a `BigInt` through the index protocol (`space.index`), so
/// `hex`/`oct`/`bin` accept any `__index__` object and arbitrarily large
/// integers. A missing `__index__` raises "'X' object cannot be interpreted as
/// an integer"; an `__index__` returning a non-int raises "__index__ returned
/// non-int (type X)".
fn index_to_bigint(obj: PyObjectRef) -> Result<BigInt, crate::PyError> {
    let w_index = crate::baseobjspace::space_index(obj)?;
    unsafe {
        if pyre_object::is_bool(w_index) {
            return Ok(BigInt::from(pyre_object::w_bool_get_value(w_index) as i64));
        }
        Ok(obj_to_bigint(w_index))
    }
}

/// Format a `BigInt` in `radix` with the given `0x`/`0o`/`0b` prefix, keeping
/// the sign ahead of the prefix (`-0xff`).
fn format_int_radix(value: &BigInt, radix: u32, prefix: &str) -> Result<String, crate::PyError> {
    let digits = match radix {
        2 => "01",
        8 => pyre_object::rbigint::BASE8,
        16 => pyre_object::rbigint::BASE16,
        _ => unreachable!("hex/oct/bin use only power-of-two radices"),
    };
    value
        .format(digits, prefix, "", 0)
        .map_err(|error| match error {
            pyre_object::rbigint::RBigIntError::Memory => crate::PyError::memory_error(""),
            _ => unreachable!("validated radix formatting returned an unrelated error"),
        })
}

/// `space.index(obj).asbigint().format(...)` without cloning a W_LongObject's
/// immutable payload. PyPy's `W_LongObject.asbigint()` returns `self.num`;
/// only compact int/bool results need a temporary rbigint materialisation.
fn format_index_radix(
    obj: PyObjectRef,
    radix: u32,
    prefix: &str,
) -> Result<String, crate::PyError> {
    let w_index = crate::baseobjspace::space_index(obj)?;
    unsafe {
        let owned;
        let value = if pyre_object::is_bool(w_index) {
            owned = BigInt::from(pyre_object::w_bool_get_value(w_index) as i64);
            &owned
        } else if pyre_object::is_int(w_index) {
            owned = BigInt::from(pyre_object::w_int_get_value(w_index));
            &owned
        } else {
            debug_assert!(pyre_object::is_long(w_index));
            pyre_object::w_long_get_value(w_index)
        };
        format_int_radix(value, radix, prefix)
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
    let s = format_index_radix(args[0], 16, "0x")?;
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
    let s = format_index_radix(args[0], 8, "0o")?;
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
    let s = format_index_radix(args[0], 2, "0b")?;
    Ok(w_str_new(&s))
}

/// Parse a complex literal string into `(real, imag)`, delegated to
/// `rustpython_literal::complex::parse_str`.
fn parse_complex_str(raw: &str) -> Option<(f64, f64)> {
    rustpython_literal::complex::parse_str(raw)
}

/// Coerce a value to `(real, imag)` for `complex()` construction.
///
/// `int`/`bool`/`float` become a real-only pair; a `complex` keeps both
/// components; an instance is asked for `__complex__` then `__float__`.
pub(crate) fn complex_coerce(obj: PyObjectRef) -> Result<(f64, f64), crate::PyError> {
    use pyre_object::*;
    unsafe {
        if is_exact_type(obj, &COMPLEX_TYPE) {
            return Ok((w_complex_get_real(obj), w_complex_get_imag(obj)));
        }
    }
    // CPython 3.14 try_complex_special_method runs before the complex-subclass
    // fallback, so a subclass override is honored instead of reading its raw
    // lanes.  The inherited complex.__complex__ returns an exact base value.
    if let Some(w_type) = crate::typedef::r#type(obj) {
        if let Some(w_complex) =
            unsafe { crate::baseobjspace::lookup_in_type(w_type.as_ptr(), "__complex__") }
        {
            let res = unsafe {
                crate::baseobjspace::get_and_call_function(w_complex, obj, w_type.as_ptr(), &[])?
            };
            unsafe {
                if is_complex(res) {
                    if !is_exact_type(res, &COMPLEX_TYPE) {
                        crate::warn::warn_deprecation(&format!(
                            "__complex__ returned non-complex (type {}). The ability to return an instance of a strict subclass of complex is deprecated, and may be removed in a future version of Python.",
                            crate::type_methods::arg_type_name(res)
                        ))?;
                    }
                    return Ok((w_complex_get_real(res), w_complex_get_imag(res)));
                }
            }
            return Err(crate::PyError::type_error(format!(
                "__complex__ returned non-complex (type {})",
                crate::type_methods::arg_type_name(res)
            )));
        }
    }
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
    // `complexobject.py unpackcomplex`: after `__complex__`, conversion uses
    // the real-number protocol.  Reuse float's `__float__` then `__index__`
    // ladder so an index-only object is accepted without admitting strings.
    if unsafe { is_str(obj) || pyre_object::is_bytes(obj) || pyre_object::is_bytearray(obj) } {
        return Err(crate::PyError::type_error(format!(
            "must be real number, not {}",
            crate::type_methods::arg_type_name(obj)
        )));
    }
    let w_float = builtin_float(&[obj])?;
    Ok((unsafe { w_float_get_value(w_float) }, 0.0))
}

/// `complex(real=0, imag=0)` — complexobject.c complex_new.
unsafe fn complex_constructor_has_real_protocol(obj: PyObjectRef) -> bool {
    is_bool(obj)
        || is_int(obj)
        || is_long(obj)
        || is_float(obj)
        || crate::baseobjspace::lookup(obj, "__float__").is_some()
        || crate::baseobjspace::lookup(obj, "__index__").is_some()
}

fn complex_constructor_argument_error(name: &str, obj: PyObjectRef) -> crate::PyError {
    crate::PyError::type_error(format!(
        "complex() argument '{name}' must be a real number, not {}",
        crate::type_methods::arg_type_name(obj)
    ))
}

pub(crate) fn builtin_complex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::*;
    // `complex(real=0, imag=0)` — both arguments are positional-or-keyword
    // (complexobject.py descr__new__ `w_real`/`w_imag`).
    let (pos, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["real", "imag"], "complex")?;
    let w_real = resolve_pos_or_kw(pos.first().copied(), kwargs, "real", "complex", 1)?;
    let w_imag = resolve_pos_or_kw(pos.get(1).copied(), kwargs, "imag", "complex", 2)?;
    let simple_single_positional = pos.len() == 1 && !has_real_kwargs(kwargs);

    // `complex.__new__`: an exact complex passed as the sole argument is
    // returned unchanged.  A strict subclass continues through coercion and
    // is copied into a base complex.
    if simple_single_positional && w_imag.is_none() {
        if let Some(a) = w_real {
            if unsafe { is_exact_type(a, &COMPLEX_TYPE) } {
                return Ok(a);
            }
        }
    }

    // String form accepts only the real argument.
    if let Some(a) = w_real {
        if unsafe { is_str(a) } && simple_single_positional {
            // complexobject.py:342 applies `unicode_to_decimal_w` before
            // underscore removal and parsing, including strict surrogate
            // rejection.
            let s = unicode_to_decimal_w(a)?;
            let (r, i) = parse_complex_str(&s).ok_or_else(|| {
                crate::PyError::new(
                    crate::PyErrorKind::ValueError,
                    format!("complex() arg is a malformed string"),
                )
            })?;
            return Ok(w_complex_new(r, i));
        }
    }
    let (mut real, mut imag) = match w_real {
        Some(a) => {
            let has_real_protocol = unsafe { complex_constructor_has_real_protocol(a) };
            let has_complex_protocol =
                unsafe { is_complex(a) || crate::baseobjspace::lookup(a, "__complex__").is_some() };
            if !has_real_protocol && !has_complex_protocol {
                if simple_single_positional {
                    return Err(crate::PyError::type_error(format!(
                        "complex() argument must be a string or a number, not {}",
                        crate::type_methods::arg_type_name(a)
                    )));
                }
                return Err(complex_constructor_argument_error("real", a));
            }
            let value = complex_coerce(a)?;
            // CPython 3.14 complex_new_impl: using a complex-valued real
            // argument in the general (keyword/two-argument) constructor is
            // deprecated unless the original object also has a real-number
            // conversion protocol.  The single-positional conversion path is
            // intentionally exempt.
            if !simple_single_positional && has_complex_protocol && !has_real_protocol {
                crate::warn::warn_deprecation(&format!(
                    "complex() argument 'real' must be a real number, not {}",
                    crate::type_methods::arg_type_name(a)
                ))?;
            }
            value
        }
        None => (0.0, 0.0),
    };
    if let Some(b) = w_imag {
        let (br, bi) = if unsafe { is_complex(b) } {
            crate::warn::warn_deprecation(&format!(
                "complex() argument 'imag' must be a real number, not {}",
                crate::type_methods::arg_type_name(b)
            ))?;
            unsafe { (w_complex_get_real(b), w_complex_get_imag(b)) }
        } else {
            if !unsafe { complex_constructor_has_real_protocol(b) } {
                return Err(complex_constructor_argument_error("imag", b));
            }
            let converted = builtin_float(&[b])?;
            (unsafe { w_float_get_value(converted) }, 0.0)
        };
        // complex(x, y) == x + y*j even if y is already complex; preserve the
        // signs of zero lanes by subtracting/adding only when the source lane
        // is nonzero, otherwise taking the operand's real part directly.
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
        return Err(crate::PyError::type_error(
            "format expected at least 1 argument, got 0",
        ));
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
        rustpython_wtf8::Wtf8Buf::new()
    };
    crate::type_methods::format_value_dispatch_w(value, &spec)
}

/// `__import__(name, globals=None, locals=None, fromlist=(), level=0)`
/// Call `callable` with a builtin's own argument slice, re-splitting the
/// trailing `__pyre_kw__` marker back into real keyword arguments.  This is
/// what a builtin that forwards `*args, **kwargs` onward needs: passing the
/// slice through unchanged would hand the marker dict over as a positional.
pub(crate) fn call_forwarding_args(
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    if !has_real_kwargs(kwargs) {
        return crate::call::call_function_impl_result(callable, positional);
    }
    // The keyword ABI keeps names byte-ish, so the surrogate-preserving reader
    // is the one to use here: `w_dict_str_entries` drops a `**{'\udc80': v}`
    // key outright rather than forwarding it.
    let keyword_args: Vec<(rustpython_wtf8::Wtf8Buf, PyObjectRef)> = unsafe {
        pyre_object::w_dict_str_entries_wtf8(kwargs.unwrap())
            .into_iter()
            .filter(|(name, _)| name.as_str() != Ok("__pyre_kw__"))
            .collect()
    };
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error("call has no current frame"));
        }
        crate::call::call_with_kwargs(unsafe { &mut *frame }, callable, positional, &keyword_args)
    })
}

/// `app_breakpoint.py breakpoint` — forward to `sys.breakpointhook`, which
/// must accept whatever arguments are passed.  By default that drops into pdb.
fn builtin_breakpoint(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(sys) = crate::importing::get_sys_module("sys") else {
        return Err(crate::PyError::runtime_error("lost sys.breakpointhook"));
    };
    let Ok(hook) = crate::baseobjspace::getattr_str(sys, "breakpointhook") else {
        return Err(crate::PyError::runtime_error("lost sys.breakpointhook"));
    };
    call_forwarding_args(hook, args)
}

/// — PyPy: `_frozen_importlib/interp_import.py:interp___import__`.
fn builtin_dunder_import(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `__import__(name, globals, locals, fromlist, level)` — PyPy's gateway
    // binds the five named slots before the import runs.  Use the shared
    // flat-ABI equivalent so duplicate positional/keyword values, unknown
    // keywords, and surplus positionals raise at the same boundary.
    let scope = bind_builtin_kwargs(
        args,
        &["name", "globals", "locals", "fromlist", "level"],
        &[true, false, false, false, false],
        "__import__",
    )?;
    let name_obj = scope[0];
    if !unsafe { pyre_object::is_str(name_obj) } {
        return Err(crate::PyError::type_error("module name must be a string"));
    }
    // Keep the wrapped name live across `level.__index__`: the precise
    // collector can relocate a young string while the index protocol runs.
    let _name_roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(name_obj);
    let globals = scope[1];
    let locals = scope[2];
    let fromlist = scope[3];
    let level_obj = scope[4];
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
    // The native importer keys every lookup by `&str`, so a name that has no
    // such spelling goes straight to the app-level bootstrap.  Re-read the
    // name through its root: `space_index_w` above may have moved it.
    let name_obj = pyre_object::gc_roots::shadow_stack_get(name_slot);
    let Some(name) = (unsafe { pyre_object::w_str_get_value_opt(name_obj) }) else {
        return crate::importing::dunder_import_name_obj(
            name_obj, globals, locals, fromlist, level,
        );
    };
    crate::importing::dunder_import(name, globals, locals, fromlist, level, exec_ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_abs_reuses_nonnegative_rbigint_payload() {
        let value = BigInt::one().lshift(80).unwrap();
        let input = w_long_new(value);
        let result = builtin_abs(&[input]).unwrap();

        assert_ne!(result, input, "int.__abs__ returns a fresh long wrapper");
        unsafe {
            assert!(
                std::ptr::eq(w_long_get_value(result), w_long_get_value(input)),
                "nonnegative rbigint.abs returns its original payload"
            );
        }
    }

    #[test]
    fn long_abs_negates_negative_rbigint_payload() {
        let value = BigInt::one().lshift(80).unwrap().neg();
        let input = w_long_new(value);
        let result = builtin_abs(&[input]).unwrap();

        unsafe {
            assert!(!std::ptr::eq(
                w_long_get_value(result),
                w_long_get_value(input)
            ));
            assert_eq!(w_long_get_value(result).get_sign(), 1);
        }
    }

    #[test]
    fn ensure_baseint_long_rewraps_without_cloning_payload() {
        let input = w_long_new(BigInt::one().lshift(80).unwrap());
        let result = ensure_baseint_result(input, input).unwrap();

        assert_ne!(result, input);
        unsafe {
            assert!(std::ptr::eq(
                w_long_get_value(result),
                w_long_get_value(input)
            ));
        }
    }

    /// RPython rbigint and the small-int path must implement the same
    /// Mersenne reduction across the machine-word range.
    #[test]
    fn common_hash_matches_pyre_int_helpers() {
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
                _hash_long(&BigInt::from(a)),
                _hash_int(a),
                "rbigint.hash vs _hash_int for {a}"
            );
        }
        // -1 is the reserved sentinel, remapped to -2.
        assert_eq!(_hash_int(-1), -2);
        let bigs = [
            BigInt::from(2).int_pow(100, None).unwrap() + BigInt::from(42),
            -BigInt::from(2).int_pow(100, None).unwrap(),
            BigInt::from(2).int_pow(200, None).unwrap() - BigInt::from(1),
            BigInt::from(u128::MAX),
            -BigInt::from(u128::MAX),
        ];
        for b in &bigs {
            assert_eq!(
                _hash_long(b),
                rustpython_common::hash::hash_bigint(&crate::rbigint_to_compiler_bigint(b)),
                "rbigint.hash vs independent compiler-bigint hash"
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

    /// CPython 3.14 uses SipHash-1-3, treats empty input specially as zero, and
    /// expands numeric `PYTHONHASHSEED` values with its deterministic LCG.
    #[test]
    fn string_and_bytes_hash_match_cpython_seed_vectors() {
        let zero = [0; 16];
        assert_eq!(_hash_bytes_with_key(b"", &zero), 0);
        assert_eq!(_hash_bytes_with_key(b"abc", &zero), -4594863902769663758);

        let seed_42 = hash_secret_from_seed(42);
        assert_eq!(_hash_bytes_with_key(b"abc", &seed_42), 3869580338025362921);
        assert_eq!(
            _hash_bytes_with_key(b"abcdefghijk", &seed_42),
            7764564197781545852
        );

        let non_ascii =
            rustpython_wtf8::Wtf8Buf::from_string("\u{e4}\u{fa}\u{2211}\u{2107}".to_owned());
        assert_eq!(
            _hash_unicode_with_key(&non_ascii, &zero),
            -2810468059467891395
        );
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
    fn tuple_hash_is_cached_for_general_and_specialised_layouts() {
        crate::typedef::init_typeobjects();
        for value in [
            w_tuple_new(vec![w_int_new(1), w_int_new(2)]),
            w_tuple_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]),
        ] {
            unsafe {
                assert_eq!(pyre_object::w_tuple_cached_hash(value), None);
            }
            let first = try_hash_value(value).unwrap();
            unsafe {
                assert_eq!(pyre_object::w_tuple_cached_hash(value), Some(first));
            }
            assert_eq!(try_hash_value(value).unwrap(), first);
        }
    }

    #[test]
    fn test_hash_rejects_tuple_containing_unhashable_key() {
        crate::typedef::init_typeobjects();
        let value = w_tuple_new(vec![w_list_new(vec![])]);
        let err = builtin_hash(&[value]).expect_err("tuple hash should reject list element");

        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
    }

    #[test]
    fn int_string_digit_limit_precedes_digit_buffer_allocation() {
        crate::typedef::init_typeobjects();
        let text = "7".repeat(crate::module::sys::state::DEFAULT_MAX_STR_DIGITS as usize + 1);
        let source = w_str_new(&text);
        let err = parse_int_from_str(source, &text, 10).unwrap_err();
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        assert_eq!(
            err.message,
            "Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits; use sys.set_int_max_str_digits() to increase the limit"
        );
    }

    #[test]
    fn int_string_leading_underscore_error_precedes_digit_limit() {
        crate::typedef::init_typeobjects();
        let text = format!(
            "_{}",
            "7".repeat(crate::module::sys::state::DEFAULT_MAX_STR_DIGITS as usize + 1)
        );
        let source = w_str_new(&text);
        let err = parse_int_from_str(source, &text, 10).unwrap_err();
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        assert!(
            err.message
                .starts_with("invalid literal for int() with base 10:")
        );
    }

    #[test]
    fn int_string_rejects_whitespace_after_sign() {
        crate::typedef::init_typeobjects();
        for text in ["- 1", "+ 1", " + 1 "] {
            let source = w_str_new(text);
            let err = parse_int_from_str(source, text, 10).unwrap_err();
            assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        }
    }

    #[test]
    fn int_string_uses_rbigint_number_parser_for_prefixes_and_overflow() {
        crate::typedef::init_typeobjects();
        for (text, base, expected) in [
            ("0x_10", 0, BigInt::from(16)),
            ("0x_10", 16, BigInt::from(16)),
            ("00_0", 0, BigInt::from(0)),
            ("9223372036854775808", 10, BigInt::from(i64::MAX).int_add(1)),
            (
                "-9223372036854775809",
                10,
                BigInt::from(i64::MIN).int_sub(1),
            ),
        ] {
            let result = parse_int_from_str(w_str_new(text), text, base).unwrap();
            if let Ok(expected_int) = expected.toint() {
                assert!(unsafe { is_int(result) });
                assert_eq!(unsafe { w_int_get_value(result) }, expected_int);
            } else {
                assert!(unsafe { is_long(result) });
                assert!(unsafe { w_long_get_value(result).eq(&expected) });
            }
        }

        for text in ["012", "0x__1", "1__0", "1_"] {
            let error = parse_int_from_str(w_str_new(text), text, 0).unwrap_err();
            assert_eq!(error.kind, crate::PyErrorKind::ValueError);
        }
    }

    #[test]
    fn integer_round_keeps_rbigint_ndigits_and_ties_to_even() {
        crate::typedef::init_typeobjects();
        for (value, expected) in [(2500, 2000), (3500, 4000), (-2500, -2000), (-3500, -4000)] {
            let rounded = builtin_round(&[w_int_new(value), w_long_new(BigInt::from(-3))]).unwrap();
            assert_eq!(unsafe { w_int_get_value(rounded) }, expected);
        }

        let value = BigInt::fromdecimalstr("1234567890123456789012345");
        let rounded = builtin_round(&[w_long_new(value.clone()), w_int_new(-5)]).unwrap();
        let divisor = BigInt::from(10).int_pow(5, None).unwrap();
        let expected = value
            .divmod(&divisor)
            .expect("positive divisor")
            .0
            .mul(&divisor);
        assert!(unsafe { w_long_get_value(rounded).eq(&expected) });

        let original = w_long_new(value);
        assert_eq!(
            builtin_round(&[original, w_long_new(BigInt::one().lshift(70).unwrap())]).unwrap(),
            original
        );
    }

    #[test]
    fn int_string_rejects_trailing_nondigit() {
        crate::typedef::init_typeobjects();
        for text in ["1234L", "12x", "0x12g"] {
            let source = w_str_new(text);
            let base = if text.starts_with("0x") { 0 } else { 10 };
            let err = parse_int_from_str(source, text, base).unwrap_err();
            assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        }
    }
    /// PyPy `interp_io._open` constructs `W_FileIO`, and
    /// `W_FileIO.descr_init` calls `_open_fd` before returning.  A writable
    /// pathname must therefore be created/truncated while the stream is still
    /// open, rather than when our wrapper is later flushed or closed.
    #[cfg(unix)]
    #[test]
    fn open_write_path_opens_and_truncates_immediately() {
        crate::typedef::init_typeobjects();
        static NEXT_PATH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let serial = NEXT_PATH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("pyre-open-write-{}-{serial}-æ", std::process::id()));
        std::fs::write(&path, b"old contents").unwrap();

        let path_text = path.to_str().unwrap();
        let file = builtin_open(&[w_str_new(path_text), w_str_new("w")]).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
        assert!(!unsafe {
            pyre_object::w_bool_get_value(crate::baseobjspace::getattr_str(file, "closed").unwrap())
        });

        let closed = crate::baseobjspace::call_method(file, "close", &[]);
        assert!(!closed.is_null());

        let raw = builtin_open(&[w_str_new(path_text), w_str_new("rb"), w_int_new(0)]).unwrap();
        let raw_type = crate::typedef::r#type(raw).unwrap().as_ptr();
        assert_eq!(unsafe { pyre_object::w_type_get_name(raw_type) }, "FileIO");
        let closed = crate::baseobjspace::call_method(raw, "close", &[]);
        assert!(!closed.is_null());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_builtin_divmod_delegates_through_proxy() {
        let _g = crate::module::_weakref::interp__weakref::lock_proxy_tests();
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
        crate::typedef::init_typeobjects();
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
        let _g = crate::module::_weakref::interp__weakref::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("DivmodLhs", |ns| {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__divmod__",
                    make_builtin_function("__divmod__", |_| {
                        Ok(w_tuple_new(vec![w_int_new(41), w_int_new(1)]))
                    }),
                )
            };
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
        let _g = crate::module::_weakref::interp__weakref::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let proxy = crate::module::_weakref::interp__weakref::W_Proxy_new(w_int_new(5), PY_NULL);
        let result = builtin_pow(&[proxy, w_int_new(3), w_int_new(13)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 8);
    }

    #[test]
    fn test_builtin_pow_two_arg_delegates_through_proxy() {
        let _g = crate::module::_weakref::interp__weakref::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let proxy = crate::module::_weakref::interp__weakref::W_Proxy_new(w_int_new(5), PY_NULL);
        let result = builtin_pow(&[proxy, w_int_new(3)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 125);
    }

    #[test]
    fn test_builtin_pow_three_arg_allows_lhs_dunder_before_dead_proxy_exp() {
        let _g = crate::module::_weakref::interp__weakref::lock_proxy_tests();
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("PowLhs", |ns| {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__pow__",
                    make_builtin_function("__pow__", |_| Ok(w_int_new(99))),
                )
            };
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

        let negative_modulus = builtin_pow(&[
            w_long_new(BigInt::from(5)),
            w_long_new(BigInt::from(-1)),
            w_long_new(BigInt::from(-13)),
        ])
        .unwrap();
        assert!(unsafe { is_long(negative_modulus) });
        assert!(unsafe { w_long_get_value(negative_modulus).int_eq(-5) });
    }

    #[test]
    fn test_builtin_pow_three_arg_borrows_long_operands() {
        crate::typedef::init_typeobjects();
        let base = BigInt::one().lshift(100).unwrap().int_add(3);
        let exponent = BigInt::from(5);
        let modulus = BigInt::one().lshift(70).unwrap().int_add(33);
        let expected = base.pow(&exponent, Some(&modulus)).unwrap();
        let result =
            builtin_pow(&[w_long_new(base), w_long_new(exponent), w_long_new(modulus)]).unwrap();
        assert!(unsafe { is_long(result) });
        assert!(unsafe { w_long_get_value(result).eq(&expected) });

        let zero_exp = builtin_pow(&[
            w_long_new(BigInt::from(2)),
            w_long_new(BigInt::zero()),
            w_long_new(BigInt::from(-13)),
        ])
        .unwrap();
        assert!(unsafe { w_long_get_value(zero_exp).int_eq(-12) });
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
