//! `str` -- PyPy `cpyext/unicodeobject.py`.
//!
//! `PyUnicode_AsUTF8` has to hand out a stable, NUL-terminated address.  The
//! interpreter's own storage is WTF-8, unterminated and movable, so the bytes
//! are copied into the mirror's cache — the counterpart of the `c_utf8` field
//! upstream fills on the `PyUnicodeObject` mirror.  The address is therefore
//! valid for exactly as long as the caller's reference to the object.

use super::object::argument;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use rustpython_wtf8::{CodePoint, Wtf8Buf};
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};

/// One code point, as `Include/unicodeobject.h:94` declares it.
#[allow(non_camel_case_types)]
pub type Py_UCS4 = u32;
use std::hash::BuildHasherDefault;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_FromString(text: *const c_char) -> *mut CPyObject {
    if text.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
    from_utf8_bytes(bytes)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_FromStringAndSize(
    text: *const c_char,
    size: isize,
) -> *mut CPyObject {
    if text.is_null() || size < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let bytes = unsafe { std::slice::from_raw_parts(text as *const u8, size as usize) };
    from_utf8_bytes(bytes)
}

fn from_utf8_bytes(bytes: &[u8]) -> *mut CPyObject {
    match std::str::from_utf8(bytes) {
        Ok(text) => pyobject::make_ref(pyre_object::w_str_new(text)),
        Err(error) => {
            super::pyerrors::set_pending_error(crate::PyError::new(
                crate::PyErrorKind::UnicodeDecodeError,
                format!(
                    "'utf-8' codec can't decode byte at position {}",
                    error.valid_up_to()
                ),
            ));
            std::ptr::null_mut()
        }
    }
}

fn text_argument(object: *mut CPyObject, function: &str) -> Option<pyre_object::PyObjectRef> {
    let value = argument(object)?;
    if !unsafe { pyre_object::unicodeobject::is_str(value) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "{function}(): str expected"
        )));
        return None;
    }
    Some(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_AsUTF8(object: *mut CPyObject) -> *const c_char {
    unsafe { PyUnicode_AsUTF8AndSize(object, std::ptr::null_mut()) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_AsUTF8AndSize(
    object: *mut CPyObject,
    size: *mut isize,
) -> *const c_char {
    let Some(value) = text_argument(object, "PyUnicode_AsUTF8") else {
        return std::ptr::null();
    };
    // The interpreter buffer is WTF-8, so a lone surrogate has to be refused
    // here rather than handed to C as invalid UTF-8.
    let Some(encoded) = super::pyerrors::trap(crate::baseobjspace::str_utf8_w(value)) else {
        if !size.is_null() {
            unsafe { *size = -1 };
        }
        return std::ptr::null();
    };
    let (pointer, length) =
        unsafe { pyobject::cached_bytes(object, || encoded.as_bytes().to_vec()) };
    if !size.is_null() {
        unsafe { *size = length as isize };
    }
    pointer
}

/// The number of code points, which is what `len()` reports.
///
/// Answered from the canonical block, so a string still being filled through
/// [`PyUnicode_New`] reports the width it was created with.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_GetLength(object: *mut CPyObject) -> isize {
    match canonical(object) {
        Some((_, length, _, _)) => length as isize,
        None => {
            let _ = text_argument(object, "PyUnicode_GetLength");
            -1
        }
    }
}

/// Both spellings of the type test answer for a string still being filled
/// without reading it as a value: [`PyUnicode_New`] hands back an exact `str`,
/// and converting it would build it out of what C has written so far.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_Check(object: *mut CPyObject) -> c_int {
    if is_pending(object) {
        return 1;
    }
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::unicodeobject::is_str(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_CheckExact(object: *mut CPyObject) -> c_int {
    if is_pending(object) {
        return 1;
    }
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && super::object::is_exactly(object, &pyre_object::STR_TYPE)) as c_int
}

// ── the canonical representation (PEP 393) ──────────────────────────────

/// The code points of a `str`, in the fixed-width form C reads and writes.
///
/// A mirror carries no storage, so this is a side-table entry keyed by the
/// mirror address — the shape [`pyobject::cached_bytes`] already has for the
/// UTF-8 view.  `data` holds `kind * (length + 1)` bytes: the code points, then
/// the NUL a compact string carries.
struct Block {
    kind: c_int,
    length: usize,
    ascii: bool,
    data: Box<[u8]>,
    /// A block [`PyUnicode_New`] handed out and C has not finished writing.
    /// No interpreter object mirrors it yet, so there is nothing to read the
    /// contents back from and nothing for `from_ref` to answer with.
    pending: bool,
}

type BlockTable = HashMap<usize, Block, BuildHasherDefault<std::hash::DefaultHasher>>;
static BLOCKS: super::ForkMutex<BlockTable> =
    super::ForkMutex::new(HashMap::with_hasher(BuildHasherDefault::new()));

pub(super) unsafe fn after_fork_child() {
    unsafe { BLOCKS.reinit_after_fork() };
}

/// Drop what a dying mirror's canonical form occupied.
pub(super) fn forget_block(mirror: usize) {
    BLOCKS.lock().remove(&mirror);
}

/// Whether `raw` is a string [`PyUnicode_New`] handed out and nothing has read
/// as a value yet — so C may still be writing it.
fn is_pending(raw: *mut CPyObject) -> bool {
    !raw.is_null()
        && BLOCKS
            .lock()
            .get(&(raw as usize))
            .is_some_and(|block| block.pending)
}

/// The widest code point `kind` can hold, and the kind a code point needs.
fn kind_for(maxchar: u32) -> c_int {
    match maxchar {
        0..=0xff => 1,
        0x100..=0xffff => 2,
        _ => 4,
    }
}

fn write_unit(data: &mut [u8], kind: c_int, index: usize, value: u32) {
    let at = index * kind as usize;
    match kind {
        1 => data[at] = value as u8,
        2 => data[at..at + 2].copy_from_slice(&(value as u16).to_ne_bytes()),
        _ => data[at..at + 4].copy_from_slice(&value.to_ne_bytes()),
    }
}

fn read_unit(data: &[u8], kind: c_int, index: usize) -> u32 {
    let at = index * kind as usize;
    match kind {
        1 => data[at] as u32,
        2 => u16::from_ne_bytes([data[at], data[at + 1]]) as u32,
        _ => u32::from_ne_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]),
    }
}

/// The canonical form of an existing `str`.
fn encode(w_obj: PyObjectRef) -> Block {
    let points: Vec<u32> = unsafe { pyre_object::w_str_get_wtf8(w_obj) }
        .code_points()
        .map(|point| point.to_u32())
        .collect();
    let maxchar = points.iter().copied().max().unwrap_or(0);
    let kind = kind_for(maxchar);
    let mut data = vec![0u8; kind as usize * (points.len() + 1)].into_boxed_slice();
    for (index, &point) in points.iter().enumerate() {
        write_unit(&mut data, kind, index, point);
    }
    Block {
        kind,
        length: points.len(),
        ascii: maxchar < 0x80,
        data,
        pending: false,
    }
}

/// `(kind, length, ascii, data)` for a mirror, built from its `str` on first
/// demand.  `None` for a mirror that is neither pending nor a `str`.
fn canonical(raw: *mut CPyObject) -> Option<(c_int, usize, bool, *mut u8)> {
    if raw.is_null() {
        return None;
    }
    // The `str` is read outside the lock: `is_str` and the code point walk are
    // interpreter operations, and this lock is held by the deallocator too.
    let existing = BLOCKS.lock().contains_key(&(raw as usize));
    if !existing {
        let w_obj = unsafe { pyobject::from_ref(raw) };
        if w_obj.is_null() || !unsafe { pyre_object::unicodeobject::is_str(w_obj) } {
            return None;
        }
        let block = encode(w_obj);
        BLOCKS.lock().entry(raw as usize).or_insert(block);
    }
    let mut table = BLOCKS.lock();
    let block = table.get_mut(&(raw as usize))?;
    // The box owns its bytes, so the address survives the map rehashing.
    Some((
        block.kind,
        block.length,
        block.ascii,
        block.data.as_mut_ptr(),
    ))
}

/// `PyUnicode_New(size, maxchar)` — an uninitialized string of `size` code
/// points wide enough for `maxchar`, for the caller to fill through
/// [`PyUnicode_DATA`].
///
/// There is no interpreter object yet: what its contents will be is not decided
/// until C stops writing.  The mirror is handed out unlinked, and
/// [`realize_pending`] builds the `str` at the one point where the result
/// crosses back into the interpreter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_New(size: isize, maxchar: Py_UCS4) -> *mut CPyObject {
    if size < 0 || maxchar > 0x10ffff {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let kind = kind_for(maxchar);
    let Some(bytes) = (size as usize + 1).checked_mul(kind as usize) else {
        return unsafe { super::pyerrors::PyErr_NoMemory() };
    };
    let mut data: Vec<u8> = Vec::new();
    if data.try_reserve_exact(bytes).is_err() {
        return unsafe { super::pyerrors::PyErr_NoMemory() };
    }
    data.resize(bytes, 0);
    let w_str_type = crate::typedef::gettypeobject(&pyre_object::STR_TYPE);
    if w_str_type.is_null() {
        return unsafe { super::pyerrors::PyErr_NoMemory() };
    }
    let ob_type = pyobject::borrow_mirror(w_str_type) as *mut super::typeobject::CPyTypeObject;
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
    BLOCKS.lock().insert(
        raw as usize,
        Block {
            kind,
            length: size as usize,
            ascii: maxchar < 0x80,
            data: data.into_boxed_slice(),
            pending: true,
        },
    );
    raw
}

/// Give a mirror [`PyUnicode_New`] handed out the `str` its contents now
/// describe, and link the two — `unicodeobject.py:118 unicode_realize`.
///
/// Reached from [`super::pyobject::realize`], so the string is built the first
/// time the mirror is read as a value.  What C wrote up to that point is what
/// the string holds; a later write reaches the block alone, as upstream's
/// "the buffer must not be modified after this call" says.
pub(super) fn realize_pending(raw: *mut CPyObject) {
    if raw.is_null() {
        return;
    }
    let text = {
        let mut table = BLOCKS.lock();
        match table.get_mut(&(raw as usize)) {
            Some(block) if block.pending => {
                block.pending = false;
                let mut text = Wtf8Buf::with_capacity(block.length);
                for index in 0..block.length {
                    let point = read_unit(&block.data, block.kind, index);
                    text.push(
                        CodePoint::from_u32(point).unwrap_or(CodePoint::from_char('\u{fffd}')),
                    );
                }
                text
            }
            _ => return,
        }
    };
    // Outside the lock: the allocation below is a collection point, and the
    // deallocator takes this lock.
    let roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(pyre_object::w_str_from_wtf8_managed(text));
    let refcnt = unsafe { (*raw).ob_refcnt };
    pyobject::link_allocated(
        pyre_object::gc_roots::shadow_stack_get(slot),
        raw,
        pyobject::REFCNT_FROM_PYRE + refcnt,
    );
}

/// `PyUnicode_KIND` — 1, 2 or 4 bytes per code point.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_KIND(object: *mut CPyObject) -> c_int {
    match canonical(object) {
        Some((kind, _, _, _)) => kind,
        None => {
            unsafe { super::pyerrors::PyErr_BadInternalCall() };
            0
        }
    }
}

/// `PyUnicode_DATA` — the code points, `PyUnicode_KIND` bytes each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_DATA(object: *mut CPyObject) -> *mut c_void {
    match canonical(object) {
        Some((_, _, _, data)) => data as *mut c_void,
        None => {
            unsafe { super::pyerrors::PyErr_BadInternalCall() };
            std::ptr::null_mut()
        }
    }
}

/// `PyUnicode_IS_ASCII` — whether every code point is below 128, which is what
/// decides the narrower of the two one-byte forms.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_IS_ASCII(object: *mut CPyObject) -> c_uint {
    match canonical(object) {
        Some((_, _, ascii, _)) => ascii as c_uint,
        None => 0,
    }
}

/// `PyUnicode_MAX_CHAR_VALUE` — the widest code point the representation can
/// hold, not the widest it does hold.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_MAX_CHAR_VALUE(object: *mut CPyObject) -> c_uint {
    match canonical(object) {
        Some((_, _, true, _)) => 0x7f,
        Some((1, _, _, _)) => 0xff,
        Some((2, _, _, _)) => 0xffff,
        Some(_) => 0x10ffff,
        None => 0,
    }
}

/// `PyUnicode_ReadChar(object, index)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_ReadChar(object: *mut CPyObject, index: isize) -> Py_UCS4 {
    let Some((kind, length, _, data)) = canonical(object) else {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return Py_UCS4::MAX;
    };
    if index < 0 || index as usize >= length {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "string index out of range",
        ));
        return Py_UCS4::MAX;
    }
    let data = unsafe { std::slice::from_raw_parts(data, kind as usize * (length + 1)) };
    read_unit(data, kind, index as usize)
}

/// `PyUnicode_WriteChar(object, index, value)` — only meaningful while the
/// string is still being filled, which is the state `PyUnicode_New` leaves it
/// in.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_WriteChar(
    object: *mut CPyObject,
    index: isize,
    value: Py_UCS4,
) -> c_int {
    let Some((kind, length, _, data)) = canonical(object) else {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    };
    if index < 0 || index as usize >= length {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "string index out of range",
        ));
        return -1;
    }
    let data = unsafe { std::slice::from_raw_parts_mut(data, kind as usize * (length + 1)) };
    write_unit(data, kind, index as usize, value);
    0
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyUnicode_FromString as *const ());
    std::hint::black_box(PyUnicode_FromStringAndSize as *const ());
    std::hint::black_box(PyUnicode_AsUTF8 as *const ());
    std::hint::black_box(PyUnicode_AsUTF8AndSize as *const ());
    std::hint::black_box(PyUnicode_GetLength as *const ());
    std::hint::black_box(PyUnicode_Check as *const ());
    std::hint::black_box(PyUnicode_CheckExact as *const ());
    std::hint::black_box(PyUnicode_New as *const ());
    std::hint::black_box(PyUnicode_KIND as *const ());
    std::hint::black_box(PyUnicode_DATA as *const ());
    std::hint::black_box(PyUnicode_IS_ASCII as *const ());
    std::hint::black_box(PyUnicode_MAX_CHAR_VALUE as *const ());
    std::hint::black_box(PyUnicode_ReadChar as *const ());
    std::hint::black_box(PyUnicode_WriteChar as *const ());
}
