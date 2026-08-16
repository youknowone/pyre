//! `bytes` -- PyPy `cpyext/bytesobject.py`.

use super::object::argument;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::collections::HashSet;
use std::ffi::{CStr, c_char, c_int};
use std::hash::BuildHasherDefault;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_FromString(text: *const c_char) -> *mut CPyObject {
    if text.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
    pyobject::make_ref(pyre_object::bytesobject::w_bytes_from_bytes(bytes))
}

/// Mirrors [`PyBytes_FromStringAndSize`] handed out with no `bytes` behind
/// them, whose buffer the caller is still writing.
///
/// The buffer itself is the mirror's [`pyobject::cached_bytes`] entry — the
/// side table that stands in for the `ob_sval` field upstream's mirror carries
/// — so `PyBytes_AS_STRING` hands out one address before and after the `bytes`
/// exists.  This set only records which mirrors have not been read as a value
/// yet.
type PendingSet = HashSet<usize, BuildHasherDefault<std::hash::DefaultHasher>>;
static PENDING: super::ForkMutex<PendingSet> =
    super::ForkMutex::new(HashSet::with_hasher(BuildHasherDefault::new()));

pub(super) unsafe fn after_fork_child() {
    unsafe { PENDING.reinit_after_fork() };
}

/// Drop what a dying mirror recorded here.
pub(super) fn forget_pending(mirror: usize) {
    PENDING.lock().remove(&mirror);
}

/// Whether `raw` is a mirror [`PyBytes_FromStringAndSize`] handed out and
/// nothing has read as a value yet — so C may still be writing it.
fn is_pending(raw: *mut CPyObject) -> bool {
    !raw.is_null() && PENDING.lock().contains(&(raw as usize))
}

/// The buffer such a mirror hands out and its length, or `None` for a mirror
/// that already has its `bytes`.
///
/// `bytesobject.py:145-151 _PyBytes_AsString` answers the accessors from the
/// buffer rather than the object, so reading one does not decide the contents
/// early.  The producer below is unreachable: a pending mirror is entered in
/// the cache when it is allocated.
fn pending_buffer(raw: *mut CPyObject) -> Option<(*mut c_char, usize)> {
    if !is_pending(raw) {
        return None;
    }
    let (pointer, length) = unsafe { pyobject::cached_bytes(raw, Vec::new) };
    Some((pointer as *mut c_char, length))
}

/// A NULL `text` asks for an uninitialized buffer of `size` bytes the caller
/// then fills through `PyBytes_AS_STRING` — `bytesobject.py:64 new_empty_str`.
///
/// There is no interpreter object yet: what the `bytes` will hold is not
/// decided until C stops writing.  The mirror is handed out unlinked, and
/// [`realize_pending`] builds the `bytes` at the one point where it crosses
/// back into the interpreter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_FromStringAndSize(
    text: *const c_char,
    size: isize,
) -> *mut CPyObject {
    if size < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    if !text.is_null() {
        let bytes = unsafe { std::slice::from_raw_parts(text as *const u8, size as usize) };
        return pyobject::make_ref(pyre_object::bytesobject::w_bytes_from_bytes(bytes));
    }
    let w_bytes_type = crate::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE);
    if w_bytes_type.is_null() {
        return unsafe { super::pyerrors::PyErr_NoMemory() };
    }
    let mut data: Vec<u8> = Vec::new();
    if data.try_reserve_exact(size as usize).is_err() {
        return unsafe { super::pyerrors::PyErr_NoMemory() };
    }
    data.resize(size as usize, 0);
    let ob_type = pyobject::borrow_mirror(w_bytes_type) as *mut super::typeobject::CPyTypeObject;
    let raw = pyobject::allocate_raw(size_of::<CPyObject>(), true) as *mut CPyObject;
    if raw.is_null() {
        return unsafe { super::pyerrors::PyErr_NoMemory() };
    }
    unsafe {
        // One reference, the caller's; no link share, because there is nothing
        // linked yet.
        (*raw).ob_refcnt = 1;
        (*raw).ob_pyre_link = pyre_object::PY_NULL;
        (*raw).ob_type = ob_type;
    }
    PENDING.lock().insert(raw as usize);
    // The terminator `cached_bytes` appends is the NUL upstream's `ob_sval`
    // carries past `ob_size`.
    unsafe { pyobject::cached_bytes(raw, || data) };
    raw
}

/// Give a mirror [`PyBytes_FromStringAndSize`] handed out the `bytes` its
/// buffer now holds, and link the two — `bytesobject.py:100 bytes_realize`.
///
/// Reached from [`super::pyobject::realize`], so the `bytes` is built the first
/// time the mirror is read as a value.  What C wrote up to that point is what
/// it holds; a later write reaches the buffer alone, as upstream's "the
/// `ob_sval` must not be modified after this call" says.
pub(super) fn realize_pending(raw: *mut CPyObject) {
    if raw.is_null() {
        return;
    }
    {
        let mut pending = PENDING.lock();
        if !pending.remove(&(raw as usize)) {
            return;
        }
    }
    let (pointer, length) = unsafe { pyobject::cached_bytes(raw, Vec::new) };
    // Copied out, and both locks released, before the allocation below: it is a
    // collection point, and the deallocator takes them.
    let data = unsafe { std::slice::from_raw_parts(pointer as *const u8, length) }.to_vec();
    let roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(pyre_object::bytesobject::w_bytes_from_bytes(&data));
    let refcnt = unsafe { (*raw).ob_refcnt };
    pyobject::link_allocated(
        pyre_object::gc_roots::shadow_stack_get(slot),
        raw,
        pyobject::REFCNT_FROM_PYRE + refcnt,
    );
}

fn bytes_argument(object: *mut CPyObject, function: &str) -> Option<pyre_object::PyObjectRef> {
    let value = argument(object)?;
    if !unsafe { pyre_object::bytesobject::is_bytes(value) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "{function}(): bytes expected"
        )));
        return None;
    }
    Some(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_AsString(object: *mut CPyObject) -> *mut c_char {
    if let Some((pointer, _)) = pending_buffer(object) {
        return pointer;
    }
    let Some(value) = bytes_argument(object, "PyBytes_AsString") else {
        return std::ptr::null_mut();
    };
    let (pointer, _) = unsafe {
        pyobject::cached_bytes(object, || {
            pyre_object::bytesobject::w_bytes_data(value).to_vec()
        })
    };
    pointer as *mut c_char
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_AsStringAndSize(
    object: *mut CPyObject,
    buffer: *mut *mut c_char,
    size: *mut isize,
) -> c_int {
    let (pointer, length) = match pending_buffer(object) {
        Some(pending) => pending,
        None => {
            let Some(value) = bytes_argument(object, "PyBytes_AsStringAndSize") else {
                return -1;
            };
            let (pointer, length) = unsafe {
                pyobject::cached_bytes(object, || {
                    pyre_object::bytesobject::w_bytes_data(value).to_vec()
                })
            };
            (pointer as *mut c_char, length)
        }
    };
    if buffer.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    unsafe {
        *buffer = pointer;
        if !size.is_null() {
            *size = length as isize;
        } else if std::slice::from_raw_parts(pointer as *const u8, length).contains(&0) {
            super::pyerrors::set_pending_error(crate::PyError::value_error("embedded null byte"));
            return -1;
        }
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_Size(object: *mut CPyObject) -> isize {
    if let Some((_, length)) = pending_buffer(object) {
        return length as isize;
    }
    let Some(value) = bytes_argument(object, "PyBytes_Size") else {
        return -1;
    };
    unsafe { pyre_object::bytesobject::w_bytes_len(value) as isize }
}

/// `bytesobject.py:261-268 PyBytes_FromObject` — the `bytes` an object's buffer
/// or its elements make, without the `bytes(n)` count spelling and without a
/// codec.
///
/// A `__bytes__` override is *not* consulted here.  That is
/// `invoke_bytes_method`, and the only entry point that runs it is
/// [`super::object::PyObject_Bytes`] (`object.py:199-201`).
pub(super) fn bytes_of(object: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { pyre_object::bytesobject::is_bytes(object) } {
        // An exact `bytes` is its own answer; a subclass instance is copied
        // down to one.
        if unsafe {
            pyre_object::pyobject::is_exact_type(object, &pyre_object::bytesobject::BYTES_TYPE)
        } {
            return Ok(object);
        }
        let data = unsafe { pyre_object::bytesobject::w_bytes_data(object) }.to_vec();
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data));
    }
    if unsafe { pyre_object::is_str(object) } {
        return Err(crate::PyError::type_error(
            "cannot convert 'str' object to bytes",
        ));
    }
    // `_convert_from_buffer_or_iterable`: a read-only buffer if the object
    // exports one, otherwise its elements as bytes.
    if let Some(buffer) = crate::baseobjspace::full_ro_buffer_bytes(object)? {
        let data = buffer.as_bytes().to_vec();
        buffer.release();
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let iterator = match crate::baseobjspace::iter(object) {
        Ok(iterator) => iterator,
        Err(error) => {
            if unsafe { crate::baseobjspace::lookup(object, "__iter__") }.is_none() {
                return Err(crate::PyError::type_error(format!(
                    "cannot convert '{}' object to bytes",
                    crate::type_methods::arg_type_name(object)
                )));
            }
            return Err(error);
        }
    };
    let iterator_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(iterator);
    let mut data = Vec::new();
    loop {
        let item =
            crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(iterator_slot));
        match item {
            Ok(item) => data.push(unsafe { crate::baseobjspace::byte_w(item, "bytes") }?),
            Err(error) if error.kind == crate::PyErrorKind::StopIteration => break,
            Err(error) => return Err(error),
        }
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&data))
}

/// `PyBytes_FromObject(object)` — [`bytes_of`] as an entry point.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_FromObject(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    super::object::result(bytes_of(object))
}

/// `PyBytes_AS_STRING(object)` — the unchecked spelling of
/// [`PyBytes_AsString`], which takes `void *` so a caller may hand it a
/// `PyBytesObject *` without a cast.
///
/// # Safety
/// `object` must be a live `bytes` mirror.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_AS_STRING(object: *mut std::ffi::c_void) -> *mut c_char {
    unsafe { PyBytes_AsString(object as *mut CPyObject) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_Check(object: *mut CPyObject) -> c_int {
    // A pending mirror is a `bytes` by construction, and answering from its
    // type rather than from an object it does not have yet is what lets a
    // caller ask mid-fill -- `bytesobject.py:145-146`.
    if is_pending(object) {
        return 1;
    }
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::bytesobject::is_bytes(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyBytes_CheckExact(object: *mut CPyObject) -> c_int {
    if is_pending(object) {
        return 1;
    }
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && super::object::is_exactly(object, &pyre_object::bytesobject::BYTES_TYPE))
        as c_int
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyBytes_FromString as *const ());
    std::hint::black_box(PyBytes_FromStringAndSize as *const ());
    std::hint::black_box(PyBytes_AsString as *const ());
    std::hint::black_box(PyBytes_AsStringAndSize as *const ());
    std::hint::black_box(PyBytes_Size as *const ());
    std::hint::black_box(PyBytes_FromObject as *const ());
    std::hint::black_box(PyBytes_AS_STRING as *const ());
    std::hint::black_box(PyBytes_Check as *const ());
    std::hint::black_box(PyBytes_CheckExact as *const ());
}
