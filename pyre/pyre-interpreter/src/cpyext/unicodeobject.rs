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
            let start = error.valid_up_to();
            // The five arguments a decode error carries, so the instance reads
            // back the way the `utf-8` decoder's own does rather than as a
            // message with an empty `str()`.
            let (end, reason) = match error.error_len() {
                None => (bytes.len(), "unexpected end of data"),
                Some(length) => (
                    start + length,
                    match matches!(bytes[start], 0x00..=0x7f | 0xc2..=0xf4) {
                        true => "invalid continuation byte",
                        false => "invalid start byte",
                    },
                ),
            };
            super::pyerrors::set_pending_error(crate::typedef::unicode_decode_error(
                "utf-8", bytes, start, end, reason,
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

// ── the operations an extension reaches str through ─────────────────────

/// A `str` argument, or `None` with `message` recorded as a `TypeError`.
/// Accepts a subclass, which is what `PyUnicode_Check` answers for.
fn str_argument(
    object: *mut CPyObject,
    message: impl FnOnce(&str) -> String,
) -> Option<PyObjectRef> {
    let value = argument(object)?;
    if !unsafe { crate::baseobjspace::isinstance_str_w(value) } {
        let name = crate::type_methods::arg_type_name(value);
        super::pyerrors::set_pending_error(crate::PyError::type_error(message(&name)));
        return None;
    }
    Some(value)
}

/// `unicodeobject.c:1040 ensure_unicode` — the message an operand that is not
/// a `str` is refused with where the entry point has none of its own.
fn must_be_str(name: &str) -> String {
    format!("must be str, not {name}")
}

/// `unicodeobject.py:961 PyUnicode_FromOrdinal` — `chr(ordinal)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_FromOrdinal(ordinal: c_int) -> *mut CPyObject {
    if !(0..=0x10ffff).contains(&(ordinal as i64)) {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "chr() arg not in range(0x110000)".to_string(),
        ));
        return std::ptr::null_mut();
    }
    pyobject::make_ref(pyre_object::w_str_from_codepoint(ordinal as u32))
}

/// The handler a codec entry point was given, `strict` for a NULL one.
fn error_handler(errors: *const c_char) -> String {
    if errors.is_null() {
        return "strict".to_owned();
    }
    unsafe { CStr::from_ptr(errors) }
        .to_string_lossy()
        .into_owned()
}

/// The body every `PyUnicode_Decode*` shares.
///
/// The error handler is the interpreter's own, reached by decoding through
/// `bytes.decode` rather than by naming the handlers this understands.
fn decode_through(
    string: *const c_char,
    length: isize,
    encoding: &str,
    errors: *const c_char,
) -> *mut CPyObject {
    if string.is_null() || length < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let bytes = unsafe { std::slice::from_raw_parts(string as *const u8, length as usize) };
    let handler = error_handler(errors);
    let w_bytes = pyre_object::bytesobject::w_bytes_from_bytes(bytes);
    super::object::result(super::object::call_method(
        w_bytes,
        "decode",
        &[
            pyre_object::w_str_new(encoding),
            pyre_object::w_str_new(&handler),
        ],
    ))
}

/// The body every `PyUnicode_As*String` shares: `str.encode`, refusing
/// anything that is not a `str` the way `PyErr_BadArgument` does.
fn encode_through(object: *mut CPyObject, encoding: &str, errors: *const c_char) -> *mut CPyObject {
    let Some(value) = argument(object) else {
        return std::ptr::null_mut();
    };
    if !unsafe { pyre_object::unicodeobject::is_str(value) } {
        unsafe { super::pyerrors::PyErr_BadArgument() };
        return std::ptr::null_mut();
    }
    let handler = error_handler(errors);
    super::object::result(super::object::call_method(
        value,
        "encode",
        &[
            pyre_object::w_str_new(encoding),
            pyre_object::w_str_new(&handler),
        ],
    ))
}

/// The encoding an entry point was given, `utf-8` for a NULL one.
fn encoding_name(encoding: *const c_char) -> String {
    if encoding.is_null() {
        return "utf-8".to_owned();
    }
    unsafe { CStr::from_ptr(encoding) }
        .to_string_lossy()
        .into_owned()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_DecodeUTF8(
    string: *const c_char,
    length: isize,
    errors: *const c_char,
) -> *mut CPyObject {
    decode_through(string, length, "utf-8", errors)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_DecodeASCII(
    string: *const c_char,
    length: isize,
    errors: *const c_char,
) -> *mut CPyObject {
    decode_through(string, length, "ascii", errors)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_DecodeLatin1(
    string: *const c_char,
    length: isize,
    errors: *const c_char,
) -> *mut CPyObject {
    decode_through(string, length, "latin-1", errors)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_Decode(
    string: *const c_char,
    length: isize,
    encoding: *const c_char,
    errors: *const c_char,
) -> *mut CPyObject {
    decode_through(string, length, &encoding_name(encoding), errors)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_AsUTF8String(object: *mut CPyObject) -> *mut CPyObject {
    encode_through(object, "utf-8", std::ptr::null())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_AsASCIIString(object: *mut CPyObject) -> *mut CPyObject {
    encode_through(object, "ascii", std::ptr::null())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_AsLatin1String(object: *mut CPyObject) -> *mut CPyObject {
    encode_through(object, "latin-1", std::ptr::null())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_AsEncodedString(
    object: *mut CPyObject,
    encoding: *const c_char,
    errors: *const c_char,
) -> *mut CPyObject {
    encode_through(object, &encoding_name(encoding), errors)
}

// ── the `wchar_t` forms ─────────────────────────────────────────────────

/// `wchar_t`, whose width is the host's: four bytes holding one code point
/// where the C library says so, two holding one UTF-16 unit on Windows.
#[allow(non_camel_case_types)]
#[cfg(windows)]
pub type wchar_t = u16;

#[allow(non_camel_case_types)]
#[cfg(not(windows))]
pub type wchar_t = i32;

/// The string as the host's wide units.
///
/// The two arms are the two things a `wchar_t` holds. Where it is four bytes
/// the interpreter's own code points go across one for one, an unpaired
/// surrogate included; where it is two the string has to be spelled as UTF-16
/// first, which is what makes the count differ from `len()` for anything
/// outside the basic plane.
fn wide_units(value: PyObjectRef) -> Vec<wchar_t> {
    let wtf8 = unsafe { pyre_object::w_str_get_wtf8(value) };
    #[cfg(windows)]
    {
        wtf8.encode_wide().collect()
    }
    #[cfg(not(windows))]
    {
        wtf8.code_points()
            .map(|cp| cp.to_u32() as wchar_t)
            .collect()
    }
}

/// [`wide_units`]' inverse.
fn wide_text(units: &[wchar_t]) -> Option<rustpython_wtf8::Wtf8Buf> {
    #[cfg(windows)]
    {
        Some(rustpython_wtf8::Wtf8Buf::from_wide(units))
    }
    #[cfg(not(windows))]
    {
        let mut text = rustpython_wtf8::Wtf8Buf::new();
        for &unit in units {
            text.push(CodePoint::from_u32(unit as u32)?);
        }
        Some(text)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_FromWideChar(
    wide: *const wchar_t,
    size: isize,
) -> *mut CPyObject {
    if wide.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let size = match size < 0 {
        true => {
            let mut length = 0isize;
            while unsafe { *wide.offset(length) } != 0 {
                length += 1;
            }
            length
        }
        false => size,
    };
    let units = unsafe { std::slice::from_raw_parts(wide, size as usize) };
    // A unit no code point answers to is a mistake the caller made; upstream
    // reads it unchecked, which is a bad character rather than a report.
    let Some(text) = wide_text(units) else {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "wide character not in range(0x110000)".to_owned(),
        ));
        return std::ptr::null_mut();
    };
    pyobject::make_ref(pyre_object::w_str_from_wtf8_managed(text))
}

/// The count [`PyUnicode_AsWideChar`] and [`PyUnicode_AsWideCharString`] both
/// answer with, or `None` with the failure already recorded.
fn wide_argument(object: *mut CPyObject) -> Option<PyObjectRef> {
    let value = argument(object)?;
    if !unsafe { pyre_object::unicodeobject::is_str(value) } {
        unsafe { super::pyerrors::PyErr_BadArgument() };
        return None;
    }
    Some(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_AsWideChar(
    object: *mut CPyObject,
    wide: *mut wchar_t,
    size: isize,
) -> isize {
    let Some(value) = wide_argument(object) else {
        return -1;
    };
    let units = wide_units(value);
    // With nowhere to write, the answer is what a buffer would have to hold,
    // the trailing NUL included.
    if wide.is_null() {
        return units.len() as isize + 1;
    }
    // A buffer with room to spare takes the NUL as well, and answers the
    // length without it; one without room is filled and answers what it took.
    let (copied, answer) = match size > units.len() as isize {
        true => (units.len() + 1, units.len() as isize),
        false => (size.max(0) as usize, size),
    };
    for (index, &unit) in units.iter().chain(&[0]).take(copied).enumerate() {
        unsafe { *wide.add(index) = unit };
    }
    answer
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_AsWideCharString(
    object: *mut CPyObject,
    size: *mut isize,
) -> *mut wchar_t {
    let Some(value) = wide_argument(object) else {
        return std::ptr::null_mut();
    };
    let units = wide_units(value);
    // Freed by the caller with `PyMem_Free`, so it has to come from the
    // allocator that answers to it.
    let bytes = std::mem::size_of::<wchar_t>() * (units.len() + 1);
    let block = unsafe { super::pymem::PyMem_Malloc(bytes) } as *mut wchar_t;
    if block.is_null() {
        unsafe { super::pyerrors::PyErr_NoMemory() };
        return std::ptr::null_mut();
    }
    for (index, &unit) in units.iter().chain(&[0]).enumerate() {
        unsafe { *block.add(index) = unit };
    }
    if !size.is_null() {
        unsafe { *size = units.len() as isize };
        return block;
    }
    // Without a length beside it the block is read to its first NUL, so one
    // inside the string would hand out a shorter string than there is.
    if units.contains(&0) {
        unsafe { super::pymem::PyMem_Free(block as *mut c_void) };
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "embedded null character".to_owned(),
        ));
        return std::ptr::null_mut();
    }
    block
}

// ── the filesystem encoding ─────────────────────────────────────────────

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_DecodeFSDefault(string: *const c_char) -> *mut CPyObject {
    if string.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let bytes = unsafe { CStr::from_ptr(string) }.to_bytes();
    pyobject::make_ref(crate::gateway::fsdecode_filename_bytes(bytes))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_DecodeFSDefaultAndSize(
    string: *const c_char,
    size: isize,
) -> *mut CPyObject {
    if string.is_null() || size < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let bytes = unsafe { std::slice::from_raw_parts(string as *const u8, size as usize) };
    pyobject::make_ref(crate::gateway::fsdecode_filename_bytes(bytes))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_EncodeFSDefault(object: *mut CPyObject) -> *mut CPyObject {
    let Some(value) = wide_argument(object) else {
        return std::ptr::null_mut();
    };
    let Some(bytes) = super::pyerrors::trap(crate::gateway::fsencode(value)) else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
}

/// The name every failure the locale codec reports is spelled with.
const LOCALE_CODEC: &str = "locale";

/// The locale codec's two handlers, `true` for surrogateescape.
///
/// The conversion refuses anything else before it runs rather than reaching
/// the codec registry, so the check is here rather than in a codec lookup.
fn locale_handler(errors: *const c_char) -> Option<bool> {
    match error_handler(errors).as_str() {
        "strict" => Some(false),
        "surrogateescape" => Some(true),
        _ => {
            super::pyerrors::set_pending_error(crate::PyError::value_error(
                "unsupported error handler".to_owned(),
            ));
            None
        }
    }
}

/// `PyUnicode_DecodeLocaleAndSize` — the bytes as the current locale spells
/// them, which is UTF-8 on the platforms this builds for.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_DecodeLocaleAndSize(
    string: *const c_char,
    length: isize,
    errors: *const c_char,
) -> *mut CPyObject {
    if string.is_null() || length < 0 {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let count = length as usize;
    let bytes = unsafe { std::slice::from_raw_parts(string as *const u8, count) };
    // The block is read as a NUL-terminated string, so the terminator has to
    // sit exactly where the length says and nowhere before it.
    if unsafe { *string.add(count) } != 0 || bytes.contains(&0) {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "embedded null byte".to_owned(),
        ));
        return std::ptr::null_mut();
    }
    let Some(escaping) = locale_handler(errors) else {
        return std::ptr::null_mut();
    };
    let error = match std::str::from_utf8(bytes) {
        Ok(text) => return pyobject::make_ref(pyre_object::w_str_new(text)),
        Err(error) => error,
    };
    if escaping {
        return pyobject::make_ref(crate::gateway::fsdecode_filename_bytes(bytes));
    }
    // The position is counted in the units the conversion answers with, which
    // are code points rather than bytes.
    let position = unsafe { std::str::from_utf8_unchecked(&bytes[..error.valid_up_to()]) }
        .chars()
        .count();
    super::pyerrors::set_pending_error(crate::typedef::unicode_decode_error(
        LOCALE_CODEC,
        bytes,
        position,
        position + 1,
        "decoding error",
    ));
    std::ptr::null_mut()
}

/// The NUL-terminated spelling of the same decode.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_DecodeLocale(
    string: *const c_char,
    errors: *const c_char,
) -> *mut CPyObject {
    if string.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let length = unsafe { CStr::from_ptr(string) }.to_bytes().len() as isize;
    unsafe { PyUnicode_DecodeLocaleAndSize(string, length, errors) }
}

/// `PyUnicode_EncodeLocale` — the string as the current locale spells it.
///
/// A surrogate in the escape range is the byte it stands for, and every other
/// one has no spelling at all, escaping or not.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_EncodeLocale(
    object: *mut CPyObject,
    errors: *const c_char,
) -> *mut CPyObject {
    let Some(value) = wide_argument(object) else {
        return std::ptr::null_mut();
    };
    let points: Vec<u32> = unsafe { pyre_object::w_str_get_wtf8(value) }
        .code_points()
        .map(|point| point.to_u32())
        .collect();
    // The block the conversion hands back is read to its first NUL, so a
    // string holding one has nowhere to be put.
    if points.contains(&0) {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "embedded null character".to_owned(),
        ));
        return std::ptr::null_mut();
    }
    let Some(escaping) = locale_handler(errors) else {
        return std::ptr::null_mut();
    };
    let mut bytes: Vec<u8> = Vec::with_capacity(points.len());
    for (position, point) in points.into_iter().enumerate() {
        if let Some(character) = char::from_u32(point) {
            let mut buffer = [0u8; 4];
            bytes.extend_from_slice(character.encode_utf8(&mut buffer).as_bytes());
            continue;
        }
        if escaping && (0xdc80..=0xdcff).contains(&point) {
            bytes.push((point - 0xdc00) as u8);
            continue;
        }
        super::pyerrors::set_pending_error(crate::typedef::unicode_encode_error(
            LOCALE_CODEC,
            value,
            position,
            position + 1,
            "encoding error",
        ));
        return std::ptr::null_mut();
    }
    pyobject::make_ref(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
}

/// `PyUnicode_FromKindAndData(kind, buffer, size)` — a string from an array of
/// code points of one of the three widths.
///
/// The units are code points rather than an encoding, so a 2-byte unit in the
/// surrogate range is one, which is what makes the result a lone surrogate
/// rather than a decode error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_FromKindAndData(
    kind: c_int,
    buffer: *const std::ffi::c_void,
    size: isize,
) -> *mut CPyObject {
    if size < 0 {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "size must be positive".to_owned(),
        ));
        return std::ptr::null_mut();
    }
    let count = size as usize;
    let points: Vec<u32> = match kind {
        1 => unsafe { std::slice::from_raw_parts(buffer as *const u8, count) }
            .iter()
            .map(|&unit| unit as u32)
            .collect(),
        2 => unsafe { std::slice::from_raw_parts(buffer as *const u16, count) }
            .iter()
            .map(|&unit| unit as u32)
            .collect(),
        4 => unsafe { std::slice::from_raw_parts(buffer as *const u32, count) }.to_vec(),
        _ => {
            super::pyerrors::set_pending_error(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                "invalid kind".to_owned(),
            ));
            return std::ptr::null_mut();
        }
    };
    let mut text = Wtf8Buf::with_capacity(count);
    for point in points {
        // A surrogate is a code point a `str` holds, so the unit is read as
        // one rather than as half of an encoding.
        let Some(point) = CodePoint::from_u32(point) else {
            super::pyerrors::set_pending_error(crate::PyError::value_error(format!(
                "character U+{point:x} is not in range [U+0000; U+10ffff]"
            )));
            return std::ptr::null_mut();
        };
        text.push(point);
    }
    pyobject::make_ref(pyre_object::w_str_from_wtf8_managed(text))
}

/// `Py_CLEANUP_SUPPORTED`, which both converters answer with so that a parse
/// that goes on to fail can hand the reference back.
const CLEANUP_SUPPORTED: c_int = 2;

/// The release call both converters take: `arg` NULL means the parse is
/// undoing what an earlier conversion wrote.
fn release_converted(target: *mut c_void) -> c_int {
    let slot = target as *mut *mut CPyObject;
    let held = unsafe { *slot };
    if !held.is_null() {
        unsafe { pyobject::decref(held) };
    }
    unsafe { *slot = std::ptr::null_mut() };
    1
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_FSConverter(
    argument: *mut CPyObject,
    target: *mut c_void,
) -> c_int {
    if argument.is_null() {
        return release_converted(target);
    }
    let Some(value) = super::object::argument(argument) else {
        return 0;
    };
    let Some(path) = super::pyerrors::trap(crate::module::posix::interp_posix::fspath(value))
    else {
        return 0;
    };
    let bytes = if unsafe { pyre_object::bytesobject::is_bytes(path) } {
        unsafe { pyre_object::bytesobject::w_bytes_data(path) }.to_vec()
    } else {
        match super::pyerrors::trap(crate::gateway::fsencode(path)) {
            Some(bytes) => bytes,
            None => return 0,
        }
    };
    // A name the syscall would read to its first NUL is not the name that was
    // passed, so it is refused rather than silently shortened.
    if bytes.contains(&0) {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "embedded null byte".to_owned(),
        ));
        return 0;
    }
    let encoded = pyre_object::bytesobject::w_bytes_from_bytes(&bytes);
    unsafe { *(target as *mut *mut CPyObject) = pyobject::make_ref(encoded) };
    CLEANUP_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_FSDecoder(
    argument: *mut CPyObject,
    target: *mut c_void,
) -> c_int {
    if argument.is_null() {
        return release_converted(target);
    }
    let Some(value) = super::object::argument(argument) else {
        return 0;
    };
    let Some(path) = super::pyerrors::trap(crate::module::posix::interp_posix::fspath(value))
    else {
        return 0;
    };
    let text = if unsafe { pyre_object::unicodeobject::is_str(path) } {
        path
    } else {
        let bytes = unsafe { pyre_object::bytesobject::w_bytes_data(path) }.to_vec();
        crate::gateway::fsdecode_filename_bytes(&bytes)
    };
    if unsafe { pyre_object::w_str_get_wtf8(text) }
        .as_bytes()
        .contains(&0)
    {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "embedded null character".to_owned(),
        ));
        return 0;
    }
    unsafe { *(target as *mut *mut CPyObject) = pyobject::make_ref(text) };
    CLEANUP_SUPPORTED
}

/// `unicodeobject.py:716 PyUnicode_FromObject` — an exact `str`, so a subclass
/// instance is copied and anything else is refused.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_FromObject(object: *mut CPyObject) -> *mut CPyObject {
    let Some(value) = str_argument(object, |name| {
        format!("Can't convert '{name}' object to str implicitly")
    }) else {
        return std::ptr::null_mut();
    };
    if super::object::is_exactly(value, &pyre_object::STR_TYPE) {
        return pyobject::make_ref(value);
    }
    let copy = pyre_object::w_str_from_wtf8_managed(
        unsafe { pyre_object::w_str_get_wtf8(value) }.to_owned(),
    );
    pyobject::make_ref(copy)
}

/// `unicodeobject.py:937 PyUnicode_InternFromString`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_InternFromString(text: *const c_char) -> *mut CPyObject {
    let created = unsafe { PyUnicode_FromString(text) };
    if created.is_null() {
        return created;
    }
    let mut pointer = created;
    unsafe { PyUnicode_InternInPlace(&raw mut pointer) };
    pointer
}

/// `unicodeobject.py:921 PyUnicode_InternInPlace` — reference-count-neutral:
/// the caller owns what `*pointer` names after the call exactly as it did
/// before.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_InternInPlace(pointer: *mut *mut CPyObject) {
    if pointer.is_null() {
        return;
    }
    let held = unsafe { *pointer };
    let Some(value) = argument(held) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return;
    };
    if !super::object::is_exactly(value, &pyre_object::STR_TYPE) {
        return;
    }
    let interned = unsafe { pyre_object::intern_exact_str(value) };
    if interned == value {
        return;
    }
    unsafe {
        *pointer = pyobject::make_ref(interned);
        pyobject::decref(held);
    }
}

/// `unicodeobject.py:1229 PyUnicode_Concat` — `left + right`.
///
/// Both operands are checked here rather than left to `+`, whose own message
/// names the operator instead of the two types.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_Concat(
    left: *mut CPyObject,
    right: *mut CPyObject,
) -> *mut CPyObject {
    let Some(left) = str_argument(left, must_be_str) else {
        return std::ptr::null_mut();
    };
    let Some(right) = str_argument(right, |name| {
        format!("can only concatenate str (not \"{name}\") to str")
    }) else {
        return std::ptr::null_mut();
    };
    super::object::result(crate::baseobjspace::add(left, right))
}

/// `PyUnicode_Append(&left, right)` — `*left` becomes the concatenation, and
/// NULL if it cannot be built.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_Append(left: *mut *mut CPyObject, right: *mut CPyObject) {
    if left.is_null() {
        return;
    }
    let held = unsafe { *left };
    if held.is_null() {
        return;
    }
    let joined = unsafe { PyUnicode_Concat(held, right) };
    unsafe {
        pyobject::decref(held);
        *left = joined;
    }
}

/// `PyUnicode_AppendAndDel(&left, right)` — [`PyUnicode_Append`] and the
/// caller's reference to `right` given up.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_AppendAndDel(left: *mut *mut CPyObject, right: *mut CPyObject) {
    unsafe {
        PyUnicode_Append(left, right);
        if !right.is_null() {
            pyobject::decref(right);
        }
    }
}

/// `unicodeobject.py:1404 PyUnicode_Substring` — `str[start:end]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_Substring(
    object: *mut CPyObject,
    start: isize,
    end: isize,
) -> *mut CPyObject {
    let Some(value) = str_argument(object, must_be_str) else {
        return std::ptr::null_mut();
    };
    if start < 0 || end < 0 {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "string index out of range",
        ));
        return std::ptr::null_mut();
    }
    let length = unsafe { pyre_object::w_str_len(value) } as isize;
    let start = start.min(length);
    let end = end.clamp(start, length);
    let cut = unsafe {
        pyre_object::w_str_slice_codepoints(value, start as i64, 1, (end - start) as i64)
    };
    pyobject::make_ref(cut)
}

/// `unicodeobject.py:1312 PyUnicode_Join` — `separator.join(sequence)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_Join(
    separator: *mut CPyObject,
    sequence: *mut CPyObject,
) -> *mut CPyObject {
    let Some([separator, sequence]) = super::object::arguments([separator, sequence]) else {
        return std::ptr::null_mut();
    };
    super::object::result(super::object::call_method(separator, "join", &[sequence]))
}

/// `unicodeobject.py:1515 PyUnicode_FindChar` — the index of `ch`, -1 when it
/// is not there and -2 on a bad argument.
///
/// `direction` is +1 to search forward and -1 to search back.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_FindChar(
    object: *mut CPyObject,
    ch: Py_UCS4,
    start: isize,
    end: isize,
    direction: c_int,
) -> isize {
    let Some(value) = str_argument(object, must_be_str) else {
        return -2;
    };
    let length = unsafe { pyre_object::w_str_len(value) } as isize;
    let start = start.clamp(0, length);
    let end = end.clamp(start, length);
    let matches = |index: isize| {
        unsafe { pyre_object::w_str_codepoint_at(value, index as usize) }
            .is_some_and(|point| point.to_u32() == ch)
    };
    let found = if direction >= 0 {
        (start..end).find(|&index| matches(index))
    } else {
        (start..end).rev().find(|&index| matches(index))
    };
    found.unwrap_or(-1)
}

/// `unicodeobject.py:1369 PyUnicode_Contains` — `element in container`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_Contains(
    container: *mut CPyObject,
    element: *mut CPyObject,
) -> c_int {
    let Some(element) = str_argument(element, |name| {
        format!("'in <string>' requires string as left operand, not {name}")
    }) else {
        return -1;
    };
    let Some(container) = str_argument(container, must_be_str) else {
        return -1;
    };
    match super::pyerrors::trap(crate::baseobjspace::contains(container, element)) {
        Some(found) => found as c_int,
        None => -1,
    }
}

/// `unicodeobject.py:1219 PyUnicode_Compare` — -1, 0 or 1, and -1 with an error
/// recorded for an argument that is not a `str`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_Compare(left: *mut CPyObject, right: *mut CPyObject) -> c_int {
    let Some([left, right]) = super::object::arguments([left, right]) else {
        return -1;
    };
    if !unsafe {
        crate::baseobjspace::isinstance_str_w(left) && crate::baseobjspace::isinstance_str_w(right)
    } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(format!(
            "Can't compare {} and {}",
            crate::type_methods::arg_type_name(left),
            crate::type_methods::arg_type_name(right)
        )));
        return -1;
    }
    let a = unsafe { pyre_object::w_str_get_wtf8(left) };
    let b = unsafe { pyre_object::w_str_get_wtf8(right) };
    match a.cmp(b) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// The code points of `text` against the bytes of `right`, which is read as
/// ISO-8859-1 so a byte above 127 still names a code point.
fn compare_with_bytes(text: PyObjectRef, right: &[u8]) -> std::cmp::Ordering {
    let mut bytes = right.iter();
    for point in unsafe { pyre_object::w_str_get_wtf8(text) }.code_points() {
        match bytes.next() {
            None => return std::cmp::Ordering::Greater,
            Some(&byte) => match point.to_u32().cmp(&(byte as u32)) {
                std::cmp::Ordering::Equal => {}
                other => return other,
            },
        }
    }
    match bytes.next() {
        None => std::cmp::Ordering::Equal,
        Some(_) => std::cmp::Ordering::Less,
    }
}

/// `unicodeobject.py:1272 PyUnicode_CompareWithASCIIString`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_CompareWithASCIIString(
    left: *mut CPyObject,
    right: *const c_char,
) -> c_int {
    let left = unsafe { pyobject::from_ref(left) };
    if left.is_null() || right.is_null() {
        return -1;
    }
    let bytes = unsafe { CStr::from_ptr(right) }.to_bytes();
    match compare_with_bytes(left, bytes) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// `PyUnicode_RichCompare(left, right, op)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_RichCompare(
    left: *mut CPyObject,
    right: *mut CPyObject,
    op: c_int,
) -> *mut CPyObject {
    unsafe { super::object::PyObject_RichCompare(left, right, op) }
}

/// The bytes `text` encodes to, compared against `right`.
fn equal_to_bytes(text: *mut CPyObject, right: *const c_char, length: Option<isize>) -> c_int {
    let text = unsafe { pyobject::from_ref(text) };
    if text.is_null() || right.is_null() || !unsafe { crate::baseobjspace::isinstance_str_w(text) }
    {
        return 0;
    }
    let bytes = match length {
        Some(length) if length >= 0 => unsafe {
            std::slice::from_raw_parts(right as *const u8, length as usize)
        },
        _ => unsafe { CStr::from_ptr(right) }.to_bytes(),
    };
    // A lone surrogate has no UTF-8 spelling, so it equals no byte string.
    match crate::baseobjspace::str_utf8_w(text) {
        Ok(encoded) => (encoded.as_bytes() == bytes) as c_int,
        Err(_) => {
            unsafe { super::pyerrors::PyErr_Clear() };
            0
        }
    }
}

/// `PyUnicode_EqualToUTF8(str, text)` — no exception is ever recorded.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_EqualToUTF8(
    object: *mut CPyObject,
    text: *const c_char,
) -> c_int {
    equal_to_bytes(object, text, None)
}

/// `PyUnicode_EqualToUTF8AndSize(str, text, size)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_EqualToUTF8AndSize(
    object: *mut CPyObject,
    text: *const c_char,
    size: isize,
) -> c_int {
    equal_to_bytes(object, text, Some(size))
}

/// `PyUnicode_Equal(left, right)` — 1, 0, or -1 with a `TypeError` recorded for
/// an argument that is not a `str`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnicode_Equal(left: *mut CPyObject, right: *mut CPyObject) -> c_int {
    let Some(left) = str_argument(left, |name| {
        format!("first argument must be str, not {name}")
    }) else {
        return -1;
    };
    let Some(right) = str_argument(right, |name| {
        format!("second argument must be str, not {name}")
    }) else {
        return -1;
    };
    let a = unsafe { pyre_object::w_str_get_wtf8(left) };
    let b = unsafe { pyre_object::w_str_get_wtf8(right) };
    (a == b) as c_int
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
    std::hint::black_box(PyUnicode_FromOrdinal as *const ());
    std::hint::black_box(PyUnicode_DecodeUTF8 as *const ());
    std::hint::black_box(PyUnicode_FromObject as *const ());
    std::hint::black_box(PyUnicode_InternFromString as *const ());
    std::hint::black_box(PyUnicode_InternInPlace as *const ());
    std::hint::black_box(PyUnicode_Concat as *const ());
    std::hint::black_box(PyUnicode_Append as *const ());
    std::hint::black_box(PyUnicode_AppendAndDel as *const ());
    std::hint::black_box(PyUnicode_Substring as *const ());
    std::hint::black_box(PyUnicode_Join as *const ());
    std::hint::black_box(PyUnicode_FindChar as *const ());
    std::hint::black_box(PyUnicode_Contains as *const ());
    std::hint::black_box(PyUnicode_Compare as *const ());
    std::hint::black_box(PyUnicode_CompareWithASCIIString as *const ());
    std::hint::black_box(PyUnicode_RichCompare as *const ());
    std::hint::black_box(PyUnicode_EqualToUTF8 as *const ());
    std::hint::black_box(PyUnicode_EqualToUTF8AndSize as *const ());
    std::hint::black_box(PyUnicode_Equal as *const ());
}
