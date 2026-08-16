//! The buffer protocol -- PyPy `cpyext/buffer.py` and `cpyext/memoryobject.py`.
//!
//! Two directions meet here.  A C-defined type's `bf_getbuffer` becomes a
//! `memoryview` over the exported memory ([`buffer_view`]), the exporter's
//! `bf_releasebuffer` running when that view is released ([`release_view`]).
//! In the other direction `PyObject_GetBuffer` hands C a `Py_buffer`; for an
//! interpreter object that is a **snapshot**, because pyre's collector moves
//! object storage and the address a `Py_buffer` carries has to stay put for as
//! long as C holds it.  A snapshot cannot receive writes, so a `PyBUF_WRITABLE`
//! request for an interpreter object is refused rather than silently dropped.

use super::object::{argument, result};
use super::pyobject::{self, CPyObject};
use super::typeobject::{CPyTypeObject, pending_or, slot_of};
use parking_lot::Mutex;
use pyre_object::PyObjectRef;
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::OnceLock;

/// `Py_buffer`.
#[repr(C)]
pub struct CPyBuffer {
    pub buf: *mut c_void,
    pub obj: *mut CPyObject,
    pub len: isize,
    pub itemsize: isize,
    pub readonly: c_int,
    pub ndim: c_int,
    pub format: *mut c_char,
    pub shape: *mut isize,
    pub strides: *mut isize,
    pub suboffsets: *mut isize,
    pub internal: *mut c_void,
}

const PY_BUF_WRITABLE: c_int = 0x0001;
const PY_BUF_FORMAT: c_int = 0x0004;
const PY_BUF_ND: c_int = 0x0008;
const PY_BUF_STRIDES: c_int = 0x0010 | PY_BUF_ND;
const PY_BUF_WRITE: c_int = 0x200;

/// Unsigned bytes — the format `PyBuffer_FillInfo` reports and the fallback for
/// an exporter that leaves `format` NULL.
static DEFAULT_FORMAT: &[u8] = b"B\0";

// ── the tables a bare `PyBufferProcs` lookup goes through ───────────────

fn getbuffer_of(tp: *mut CPyTypeObject) -> *const c_void {
    unsafe {
        let table = (*tp).tp_as_buffer;
        if table.is_null() {
            std::ptr::null()
        } else {
            (*table).bf_getbuffer
        }
    }
}

fn releasebuffer_of(tp: *mut CPyTypeObject) -> *const c_void {
    unsafe {
        let table = (*tp).tp_as_buffer;
        if table.is_null() {
            std::ptr::null()
        } else {
            (*table).bf_releasebuffer
        }
    }
}

// ── the exports this layer owns ─────────────────────────────────────────

/// The `Py_buffer` behind each live `memoryview` this layer built, keyed by the
/// `BufferView` that memoryview carries.
///
/// A view has to hand the exporter back the exact structure it filled in --
/// `internal` is the exporter's own state and nothing else can reconstruct it,
/// and two live views over one address each have their own.  The key is the
/// `BufferView` allocation rather than the memoryview because the collector
/// moves the object and not that block, and rather than the exported address
/// because several exports can name the same one.
fn exports() -> &'static Mutex<HashMap<usize, usize>> {
    static EXPORTS: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    EXPORTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The snapshots [`PyObject_GetBuffer`] took of interpreter objects, keyed by
/// the address handed to C, with the byte length `PyBuffer_Release` frees.
fn snapshots() -> &'static Mutex<HashMap<usize, usize>> {
    static SNAPSHOTS: OnceLock<Mutex<HashMap<usize, usize>>> = OnceLock::new();
    SNAPSHOTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn record_export(carrier: usize, view: *mut CPyBuffer) {
    exports().lock().insert(carrier, view as usize);
}

fn take_export(carrier: usize) -> Option<*mut CPyBuffer> {
    exports()
        .lock()
        .remove(&carrier)
        .map(|view| view as *mut CPyBuffer)
}

// ── reading the geometry a `Py_buffer` carries ──────────────────────────

fn format_of(format: *const c_char) -> String {
    if format.is_null() {
        return "B".to_string();
    }
    unsafe { CStr::from_ptr(format) }
        .to_string_lossy()
        .into_owned()
}

/// One dimension vector, or the derived default when the exporter answered a
/// request that does not ask for it -- `init_shape_strides`.
fn dims(pointer: *const isize, ndim: i64, derived: &[i64]) -> Vec<i64> {
    if pointer.is_null() || ndim <= 0 {
        return derived.to_vec();
    }
    (0..ndim as usize)
        .map(|index| unsafe { *pointer.add(index) } as i64)
        .collect()
}

/// `_IsCContiguous`.
fn c_contiguous(shape: &[i64], strides: &[i64], itemsize: i64) -> bool {
    let mut expected = itemsize;
    for (&extent, &stride) in shape.iter().zip(strides).rev() {
        if extent > 1 && stride != expected {
            return false;
        }
        expected *= extent;
    }
    true
}

/// `_IsFortranContiguous`.
fn fortran_contiguous(shape: &[i64], strides: &[i64], itemsize: i64) -> bool {
    let mut expected = itemsize;
    for (&extent, &stride) in shape.iter().zip(strides) {
        if extent > 1 && stride != expected {
            return false;
        }
        expected *= extent;
    }
    true
}

// ── `bf_getbuffer` as a `memoryview` ────────────────────────────────────

/// Does `w_obj`'s type export a buffer through a C `bf_getbuffer`?
pub fn exports_buffer(w_obj: PyObjectRef) -> bool {
    !slot_of(w_obj, getbuffer_of).is_null()
}

/// The view a C exporter's `bf_getbuffer` produces, or `None` when `w_obj`'s
/// type declares none.
pub fn buffer_view(
    w_obj: PyObjectRef,
    flags: c_int,
) -> Option<Result<PyObjectRef, crate::PyError>> {
    let slot = slot_of(w_obj, getbuffer_of);
    if slot.is_null() {
        return None;
    }
    Some(acquire(w_obj, slot, flags))
}

fn acquire(
    w_obj: PyObjectRef,
    slot: *const c_void,
    flags: c_int,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::buffer::Buffer;
    use pyre_object::bufferview::BufferView;

    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_obj);

    let view = Box::into_raw(Box::new(unsafe { std::mem::zeroed::<CPyBuffer>() }));
    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(obj_slot));
    let status = unsafe {
        let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyBuffer, c_int) -> c_int =
            std::mem::transmute(slot);
        call(receiver, view, flags)
    };
    unsafe { pyobject::decref(receiver) };
    if status < 0 {
        drop(unsafe { Box::from_raw(view) });
        return Err(pending_or(
            "a cpyext buffer export failed without setting an exception",
        ));
    }

    let (address, length, itemsize, readonly, ndim) = unsafe {
        (
            (*view).buf as usize,
            (*view).len as i64,
            if (*view).itemsize > 0 {
                (*view).itemsize as i64
            } else {
                1
            },
            (*view).readonly != 0,
            (*view).ndim as i64,
        )
    };
    let format = format_of(unsafe { (*view).format });
    let elements = if itemsize > 0 { length / itemsize } else { 0 };
    let shape = dims(unsafe { (*view).shape }, ndim, &[elements]);
    let strides = dims(unsafe { (*view).strides }, ndim, &[itemsize]);
    let refuse = |message: &str| {
        release_c_view(view);
        Err(crate::PyError::new(
            crate::PyErrorKind::BufferError,
            message.to_string(),
        ))
    };
    if unsafe { !(*view).suboffsets.is_null() } {
        return refuse("cpyext: a buffer export with suboffsets is not supported");
    }
    if ndim > 1 && !c_contiguous(&shape, &strides, itemsize) {
        return refuse("cpyext: a multi-dimensional buffer export must be C-contiguous");
    }
    if ndim == 1 && strides.first() != Some(&itemsize) {
        return refuse("cpyext: a strided buffer export is not supported");
    }

    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(pyre_object::w_str_new(&format));
    if ndim != 1 {
        roots.pin_root(wrap_dims(&shape));
        roots.pin_root(wrap_dims(&strides));
    }
    let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    let mv = pyre_object::memoryview::w_memoryview_alloc_header(false, true);
    let r_obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let flat = BufferView::Raw {
        backing: Buffer::External {
            w_obj: r_obj,
            address,
            size: length as usize,
            readonly,
        },
        w_obj: r_obj,
        w_fmt: reload(0),
        itemsize,
        length,
    };
    let built = if ndim == 1 {
        flat
    } else {
        BufferView::ViewND {
            parent: Box::new(flat),
            w_obj: r_obj,
            ndim,
            w_shape: reload(1),
            w_strides: reload(2),
        }
    };
    let carrier = pyre_object::memoryview::bufferview_alloc(built);
    unsafe { pyre_object::memoryview::w_memoryview_set_view(mv, carrier) };
    record_export(carrier as usize, view);
    Ok(mv)
}

/// A dimension vector as the tuple a `ViewND` rides on.
fn wrap_dims(dims: &[i64]) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    for &extent in dims {
        roots.pin_root(pyre_object::w_int_new(extent));
    }
    let items = (0..dims.len())
        .map(|index| pyre_object::gc_roots::shadow_stack_get(base + index))
        .collect();
    pyre_object::w_tuple_new(items)
}

/// Run `bf_releasebuffer` for a structure this layer allocated, drop the
/// reference it holds, and free it.
fn release_c_view(view: *mut CPyBuffer) {
    let raw = unsafe { (*view).obj };
    if !raw.is_null() {
        let w_obj = unsafe { pyobject::from_ref(raw) };
        let release = slot_of(w_obj, releasebuffer_of);
        if !release.is_null() {
            unsafe {
                let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyBuffer) =
                    std::mem::transmute(release);
                call(raw, view);
            }
        }
        unsafe { pyobject::decref(raw) };
    }
    drop(unsafe { Box::from_raw(view) });
}

/// End the export a released `memoryview` held, if a C exporter produced it.
///
/// # Safety
/// `mv` must be a live memoryview and `backing` the object its view names.
pub unsafe fn release_view(mv: PyObjectRef, backing: PyObjectRef) -> bool {
    use pyre_object::buffer::Buffer;
    if slot_of(backing, getbuffer_of).is_null() {
        return false;
    }
    let carrier = unsafe { pyre_object::memoryview::w_memoryview_view(mv) };
    let Buffer::External { .. } = *carrier.backing() else {
        return false;
    };
    if let Some(view) = take_export(carrier as *const _ as usize) {
        release_c_view(view);
    }
    true
}

// ── `PyObject_GetBuffer` and its snapshot ───────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CheckBuffer(object: *mut CPyObject) -> c_int {
    let w_obj = unsafe { pyobject::from_ref(object) };
    if w_obj.is_null() {
        return 0;
    }
    if !slot_of(w_obj, getbuffer_of).is_null() {
        return 1;
    }
    unsafe { crate::baseobjspace::lookup(w_obj, "__buffer__") }.is_some() as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetBuffer(
    object: *mut CPyObject,
    view: *mut CPyBuffer,
    flags: c_int,
) -> c_int {
    if view.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let Some(w_obj) = argument(object) else {
        return -1;
    };
    let slot = slot_of(w_obj, getbuffer_of);
    if !slot.is_null() {
        return unsafe {
            let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyBuffer, c_int) -> c_int =
                std::mem::transmute(slot);
            call(object, view, flags)
        };
    }
    if flags & PY_BUF_WRITABLE != 0 {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::BufferError,
            "an interpreter object exports a read-only snapshot to C",
        ));
        return -1;
    }
    let acquired = crate::baseobjspace::full_ro_buffer_bytes(w_obj);
    let bytes = match super::pyerrors::trap(acquired) {
        Some(Some(acquired)) => {
            let bytes = acquired.as_bytes().to_vec();
            acquired.release();
            bytes
        }
        Some(None) => {
            super::pyerrors::set_pending_error(crate::PyError::type_error(
                "a bytes-like object is required",
            ));
            return -1;
        }
        None => return -1,
    };
    let length = bytes.len();
    let block = Box::into_raw(bytes.into_boxed_slice()) as *mut u8;
    snapshots().lock().insert(block as usize, length);
    unsafe {
        fill_info(
            view,
            object,
            block as *mut c_void,
            length as isize,
            1,
            flags,
        )
    }
}

/// `PyBuffer_FillInfo`'s body, shared with the snapshot path.
///
/// # Safety
/// `view` must point at a writable `Py_buffer`.
unsafe fn fill_info(
    view: *mut CPyBuffer,
    object: *mut CPyObject,
    buf: *mut c_void,
    length: isize,
    readonly: c_int,
    flags: c_int,
) -> c_int {
    if flags & PY_BUF_WRITABLE != 0 && readonly != 0 {
        super::pyerrors::set_pending_error(crate::PyError::value_error("Object is not writable"));
        return -1;
    }
    unsafe {
        (*view).buf = buf;
        (*view).obj = pyobject::make_ref(pyobject::from_ref(object));
        (*view).len = length;
        (*view).itemsize = 1;
        (*view).readonly = readonly;
        (*view).ndim = 1;
        (*view).format = if flags & PY_BUF_FORMAT == PY_BUF_FORMAT {
            DEFAULT_FORMAT.as_ptr() as *mut c_char
        } else {
            std::ptr::null_mut()
        };
        // The one-dimensional geometry is the structure's own `len` and
        // `itemsize`, so no separate storage has to outlive the call.
        (*view).shape = if flags & PY_BUF_ND == PY_BUF_ND {
            &raw mut (*view).len
        } else {
            std::ptr::null_mut()
        };
        (*view).strides = if flags & PY_BUF_STRIDES == PY_BUF_STRIDES {
            &raw mut (*view).itemsize
        } else {
            std::ptr::null_mut()
        };
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_FillInfo(
    view: *mut CPyBuffer,
    object: *mut CPyObject,
    buf: *mut c_void,
    length: isize,
    readonly: c_int,
    flags: c_int,
) -> c_int {
    if view.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    unsafe { fill_info(view, object, buf, length, readonly, flags) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_Release(view: *mut CPyBuffer) {
    if view.is_null() {
        return;
    }
    let raw = unsafe { (*view).obj };
    if raw.is_null() {
        return;
    }
    let block = unsafe { (*view).buf } as usize;
    let snapshot = snapshots().lock().remove(&block);
    match snapshot {
        Some(length) => drop(unsafe {
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(block as *mut u8, length))
        }),
        None => {
            let w_obj = unsafe { pyobject::from_ref(raw) };
            let release = slot_of(w_obj, releasebuffer_of);
            if !release.is_null() {
                unsafe {
                    let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyBuffer) =
                        std::mem::transmute(release);
                    call(raw, view);
                }
            }
        }
    }
    unsafe {
        (*view).obj = std::ptr::null_mut();
        (*view).buf = std::ptr::null_mut();
        pyobject::decref(raw);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_IsContiguous(view: *const CPyBuffer, order: c_char) -> c_int {
    if view.is_null() {
        return 0;
    }
    if unsafe { !(*view).suboffsets.is_null() } {
        return 0;
    }
    let ndim = unsafe { (*view).ndim } as i64;
    let itemsize = unsafe { (*view).itemsize } as i64;
    let length = unsafe { (*view).len } as i64;
    let elements = if itemsize > 0 { length / itemsize } else { 0 };
    let shape = dims(unsafe { (*view).shape }, ndim, &[elements]);
    let strides = dims(unsafe { (*view).strides }, ndim, &[itemsize]);
    let contiguous = match order as u8 {
        b'C' => c_contiguous(&shape, &strides, itemsize),
        b'F' => fortran_contiguous(&shape, &strides, itemsize),
        b'A' => {
            c_contiguous(&shape, &strides, itemsize)
                || fortran_contiguous(&shape, &strides, itemsize)
        }
        _ => false,
    };
    contiguous as c_int
}

/// The number of bytes one item of `format` occupies -- `_struct.calcsize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_SizeFromFormat(format: *const c_char) -> isize {
    if format.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let format = format_of(format);
    let sized = super::import_::import_module("_struct").and_then(|module| {
        let roots = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(module);
        roots.pin_root(pyre_object::w_str_new(&format));
        let reload = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
        let calcsize = crate::baseobjspace::getattr_str(reload(0), "calcsize")?;
        let size = crate::call::call_function_impl_result(calcsize, &[reload(1)])?;
        crate::baseobjspace::int_w(size)
    });
    match super::pyerrors::trap(sized) {
        Some(size) => size as isize,
        None => -1,
    }
}

// ── the strided copies ──────────────────────────────────────────────────

/// The byte offset of `indices` within `view`.
fn offset_of(strides: &[i64], indices: &[i64]) -> isize {
    strides
        .iter()
        .zip(indices)
        .map(|(stride, index)| stride * index)
        .sum::<i64>() as isize
}

/// Step `indices` one element in C order (`order == b'C'`) or Fortran order,
/// answering `false` once the walk is over.
fn advance(indices: &mut [i64], shape: &[i64], order: u8) -> bool {
    let axes: Vec<usize> = if order == b'F' {
        (0..indices.len()).collect()
    } else {
        (0..indices.len()).rev().collect()
    };
    for axis in axes {
        indices[axis] += 1;
        if indices[axis] < shape[axis] {
            return true;
        }
        indices[axis] = 0;
    }
    false
}

/// The `(shape, strides, itemsize, total)` a filled `Py_buffer` describes.
///
/// # Safety
/// `view` must point at a filled `Py_buffer` with no suboffsets.
unsafe fn geometry(view: *const CPyBuffer) -> (Vec<i64>, Vec<i64>, i64, i64) {
    let ndim = unsafe { (*view).ndim } as i64;
    let itemsize = unsafe { (*view).itemsize } as i64;
    let length = unsafe { (*view).len } as i64;
    let elements = if itemsize > 0 { length / itemsize } else { 0 };
    let shape = dims(unsafe { (*view).shape }, ndim, &[elements]);
    let strides = dims(unsafe { (*view).strides }, ndim, &[itemsize]);
    (shape, strides, itemsize, length)
}

/// Walk every element of `view` in `order`, handing each element's address and
/// its position in the contiguous block to `visit`.
///
/// # Safety
/// `view` must point at a filled `Py_buffer` with no suboffsets.
unsafe fn walk(view: *const CPyBuffer, order: u8, mut visit: impl FnMut(*mut u8, usize, usize)) {
    let (shape, strides, itemsize, _) = unsafe { geometry(view) };
    if shape.iter().any(|&extent| extent <= 0) {
        return;
    }
    let base = unsafe { (*view).buf } as *mut u8;
    let mut indices = vec![0i64; shape.len()];
    let mut position = 0usize;
    loop {
        let element = unsafe { base.offset(offset_of(&strides, &indices)) };
        visit(element, position, itemsize as usize);
        position += itemsize as usize;
        if !advance(&mut indices, &shape, order) {
            break;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_GetPointer(
    view: *const CPyBuffer,
    indices: *const isize,
) -> *mut c_void {
    let (_, strides, _, _) = unsafe { geometry(view) };
    let mut pointer = unsafe { (*view).buf } as *mut u8;
    for (axis, stride) in strides.iter().enumerate() {
        let index = unsafe { *indices.add(axis) } as i64;
        pointer = unsafe { pointer.offset((stride * index) as isize) };
        let suboffsets = unsafe { (*view).suboffsets };
        if !suboffsets.is_null() {
            let suboffset = unsafe { *suboffsets.add(axis) };
            if suboffset >= 0 {
                pointer = unsafe { (*(pointer as *mut *mut u8)).offset(suboffset) };
            }
        }
    }
    pointer as *mut c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_ToContiguous(
    buf: *mut c_void,
    view: *const CPyBuffer,
    length: isize,
    order: c_char,
) -> c_int {
    if buf.is_null() || view.is_null() || unsafe { (*view).len } != length {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let target = buf as *mut u8;
    unsafe {
        walk(view, order as u8, |element, position, itemsize| {
            std::ptr::copy_nonoverlapping(element, target.add(position), itemsize)
        })
    };
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBuffer_FromContiguous(
    view: *const CPyBuffer,
    buf: *const c_void,
    length: isize,
    order: c_char,
) -> c_int {
    if buf.is_null() || view.is_null() || unsafe { (*view).len } != length {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    if unsafe { (*view).readonly } != 0 {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::BufferError,
            "Object is not writable.",
        ));
        return -1;
    }
    let source = buf as *const u8;
    unsafe {
        walk(view, order as u8, |element, position, itemsize| {
            std::ptr::copy_nonoverlapping(source.add(position), element, itemsize)
        })
    };
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CopyData(
    destination: *mut CPyObject,
    source: *mut CPyObject,
) -> c_int {
    super::object::realize_all([destination, source]);
    let mut target = unsafe { std::mem::zeroed::<CPyBuffer>() };
    let mut origin = unsafe { std::mem::zeroed::<CPyBuffer>() };
    if unsafe { PyObject_GetBuffer(destination, &raw mut target, PY_BUF_WRITABLE | PY_BUF_ND) } < 0
    {
        return -1;
    }
    if unsafe { PyObject_GetBuffer(source, &raw mut origin, PY_BUF_ND) } < 0 {
        unsafe { PyBuffer_Release(&raw mut target) };
        return -1;
    }
    let status = if target.len < origin.len {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::BufferError,
            "destination is too small to receive data from source",
        ));
        -1
    } else {
        let bytes = vec![0u8; origin.len as usize];
        let block = bytes.into_boxed_slice();
        let block = Box::into_raw(block) as *mut u8;
        unsafe {
            walk(&raw const origin, b'C', |element, position, itemsize| {
                std::ptr::copy_nonoverlapping(element, block.add(position), itemsize)
            });
            walk(&raw const target, b'C', |element, position, itemsize| {
                std::ptr::copy_nonoverlapping(block.add(position), element, itemsize)
            });
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                block,
                origin.len as usize,
            )));
        }
        0
    };
    unsafe {
        PyBuffer_Release(&raw mut origin);
        PyBuffer_Release(&raw mut target);
    }
    status
}

// ── the legacy character-buffer entry points ────────────────────────────

/// These spellings hand out an address and have no release call at all, so the
/// export is closed here as `buffer.py:179-195` closes it -- an exporter that
/// counts its exports, or that allocated `internal`, otherwise never gets the
/// balancing call, and its memory outlives the export either way.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_AsCharBuffer(
    object: *mut CPyObject,
    buffer: *mut *const c_char,
    size: *mut isize,
) -> c_int {
    let Some(w_obj) = argument(object) else {
        return -1;
    };
    if !slot_of(w_obj, getbuffer_of).is_null() {
        let mut view = unsafe { std::mem::zeroed::<CPyBuffer>() };
        if unsafe { PyObject_GetBuffer(object, &raw mut view, 0) } < 0 {
            return -1;
        }
        unsafe {
            *buffer = view.buf as *const c_char;
            *size = view.len;
            PyBuffer_Release(&raw mut view);
        }
        return 0;
    }
    // An interpreter object has no exporter to hand an address back to, so the
    // bytes C reads are this layer's own copy.  With no release call to free it
    // by, the copy is the mirror's cached one, which dies with the mirror -- a
    // per-call snapshot would have no owner at all.
    let acquired = crate::baseobjspace::full_ro_buffer_bytes(w_obj);
    let bytes = match super::pyerrors::trap(acquired) {
        Some(Some(acquired)) => {
            let bytes = acquired.as_bytes().to_vec();
            acquired.release();
            bytes
        }
        Some(None) => {
            super::pyerrors::set_pending_error(crate::PyError::type_error(
                "a bytes-like object is required",
            ));
            return -1;
        }
        None => return -1,
    };
    let (pointer, length) = unsafe { pyobject::cached_bytes(object, || bytes) };
    unsafe {
        *buffer = pointer;
        *size = length as isize;
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_AsReadBuffer(
    object: *mut CPyObject,
    buffer: *mut *const c_void,
    size: *mut isize,
) -> c_int {
    let mut text: *const c_char = std::ptr::null();
    let status = unsafe { PyObject_AsCharBuffer(object, &raw mut text, size) };
    if status == 0 {
        unsafe { *buffer = text as *const c_void };
    }
    status
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_AsWriteBuffer(
    object: *mut CPyObject,
    buffer: *mut *mut c_void,
    size: *mut isize,
) -> c_int {
    let mut view = unsafe { std::mem::zeroed::<CPyBuffer>() };
    if unsafe { PyObject_GetBuffer(object, &raw mut view, PY_BUF_WRITABLE) } < 0 {
        return -1;
    }
    unsafe {
        *buffer = view.buf;
        *size = view.len;
        PyBuffer_Release(&raw mut view);
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CheckReadBuffer(object: *mut CPyObject) -> c_int {
    unsafe { PyObject_CheckBuffer(object) }
}

// ── memoryview ──────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMemoryView_Check(object: *mut CPyObject) -> c_int {
    let w_obj = unsafe { pyobject::from_ref(object) };
    if w_obj.is_null() {
        return 0;
    }
    unsafe { pyre_object::memoryview::is_w_memoryview(w_obj) as c_int }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMemoryView_FromObject(object: *mut CPyObject) -> *mut CPyObject {
    let Some(w_obj) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(crate::builtins::w_memoryview_new(w_obj))
}

/// A view over memory nothing in the interpreter owns.
///
/// The window keeps no reference: the caller guarantees the memory outlives
/// every view of it, which is what this entry point's contract already says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMemoryView_FromMemory(
    memory: *mut c_char,
    size: isize,
    flags: c_int,
) -> *mut CPyObject {
    if memory.is_null() || size < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    result(Ok(unowned_view(
        memory as usize,
        size as i64,
        1,
        "B",
        flags != PY_BUF_WRITE,
    )))
}

/// `PyMemoryView_FromBuffer` — the caller hands its filled `Py_buffer` over and
/// must not release it itself.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMemoryView_FromBuffer(view: *const CPyBuffer) -> *mut CPyObject {
    if view.is_null() || unsafe { (*view).buf.is_null() } {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let (_, strides, itemsize, length) = unsafe { geometry(view) };
    if strides.first() != Some(&itemsize) || unsafe { (*view).ndim } > 1 {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::BufferError,
            "cpyext: only a contiguous one-dimensional Py_buffer becomes a memoryview",
        ));
        return std::ptr::null_mut();
    }
    let format = format_of(unsafe { (*view).format });
    let readonly = unsafe { (*view).readonly } != 0;
    result(Ok(unowned_view(
        unsafe { (*view).buf } as usize,
        length,
        itemsize,
        &format,
        readonly,
    )))
}

fn unowned_view(
    address: usize,
    length: i64,
    itemsize: i64,
    format: &str,
    readonly: bool,
) -> PyObjectRef {
    use pyre_object::buffer::Buffer;
    use pyre_object::bufferview::BufferView;
    let roots = pyre_object::gc_roots::push_roots();
    let fmt_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(pyre_object::w_str_new(format));
    let mv = pyre_object::memoryview::w_memoryview_alloc_header(false, false);
    let w_obj = pyre_object::w_none();
    let built = BufferView::Raw {
        backing: Buffer::External {
            w_obj,
            address,
            size: length as usize,
            readonly,
        },
        w_obj,
        w_fmt: pyre_object::gc_roots::shadow_stack_get(fmt_slot),
        itemsize,
        length,
    };
    unsafe {
        pyre_object::memoryview::w_memoryview_set_view(
            mv,
            pyre_object::memoryview::bufferview_alloc(built),
        )
    };
    mv
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyObject_CheckBuffer as *const ());
    std::hint::black_box(PyObject_GetBuffer as *const ());
    std::hint::black_box(PyBuffer_Release as *const ());
    std::hint::black_box(PyBuffer_FillInfo as *const ());
    std::hint::black_box(PyBuffer_IsContiguous as *const ());
    std::hint::black_box(PyBuffer_SizeFromFormat as *const ());
    std::hint::black_box(PyBuffer_GetPointer as *const ());
    std::hint::black_box(PyBuffer_ToContiguous as *const ());
    std::hint::black_box(PyBuffer_FromContiguous as *const ());
    std::hint::black_box(PyObject_CopyData as *const ());
    std::hint::black_box(PyObject_AsCharBuffer as *const ());
    std::hint::black_box(PyObject_AsReadBuffer as *const ());
    std::hint::black_box(PyObject_AsWriteBuffer as *const ());
    std::hint::black_box(PyObject_CheckReadBuffer as *const ());
    std::hint::black_box(PyMemoryView_Check as *const ());
    std::hint::black_box(PyMemoryView_FromObject as *const ());
    std::hint::black_box(PyMemoryView_FromMemory as *const ());
    std::hint::black_box(PyMemoryView_FromBuffer as *const ());
}
