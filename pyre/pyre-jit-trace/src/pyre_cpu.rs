//! `Cpu` trait impl for pyre's runtime string layout.
//!
//! `llmodel.py:557 gc_ll_descr.str_descr / unicode_descr` parity — the
//! typed `ArrayDescr` that backend init caches and that the speculative
//! protect / length read / per-character read all route through.
//! `model.py:209+` for the read-family. PyPy stores
//! `gc_ll_descr.str_descr` once at backend init; pyre exposes it via
//! the `Cpu` trait so `protect_speculative_string`, `bh_strlen` and
//! `bh_strgetitem` all reach the same descr.
//!
//! Python 3 unifies `str` and `unicode` into one `W_UnicodeObject`
//! (UTF-8), but the RPython-level STR / UNICODE split is preserved:
//! `str_descr()` returns `PyreStrDescr` (len_descr → byte_len) and
//! `unicode_descr()` returns `PyreUnicodeDescr` (len_descr → codepoint len).
//!
//! `W_UnicodeObject` (pyre-object) stores char data behind a
//! `*mut Wtf8Buf` pointer at `UNICODE_VALUE_OFFSET`; the default
//! `bh_getarrayitem_gc_i(base + index)` path would read wrong memory,
//! so `bh_strgetitem` is overridden to follow the indirection.

use std::sync::{Arc, OnceLock};

use majit_ir::box_ref::BoxRef;
use majit_ir::{ArrayDescr, Descr, FieldDescr, GcRef, Type};
use majit_metainterp::cpu::{Cpu, DefaultCpu};
use pyre_object::unicodeobject::{
    UNICODE_BYTE_LEN_OFFSET, UNICODE_LEN_OFFSET, UNICODE_VALUE_OFFSET, W_UNICODE_GC_TYPE_ID,
    W_UNICODE_OBJECT_SIZE,
};
use rustpython_wtf8::Wtf8Buf;

/// FieldDescr for `W_UnicodeObject.byte_len` — UTF-8 byte count.
/// RPython STR is `Array(Char)` byte string (`rstr.py:1226`);
/// `llmodel.py:667 bh_strlen` reads byte count.
#[derive(Debug)]
struct PyreStrByteLenFieldDescr;

impl Descr for PyreStrByteLenFieldDescr {}

impl FieldDescr for PyreStrByteLenFieldDescr {
    fn offset(&self) -> usize {
        UNICODE_BYTE_LEN_OFFSET
    }
    fn field_size(&self) -> usize {
        // `W_UnicodeObject.byte_len` is a `usize`: 8 bytes on 64-bit, 4 on
        // wasm32. A hardcoded 8 reads the adjacent field into the high
        // half on a 32-bit target.
        std::mem::size_of::<usize>()
    }
    fn field_type(&self) -> Type {
        Type::Int
    }
    fn is_field_signed(&self) -> bool {
        true
    }
    fn field_name(&self) -> &'static str {
        "W_UnicodeObject.byte_len"
    }
}

/// FieldDescr for `W_UnicodeObject.len` — codepoint count.
/// RPython UNICODE uses codepoint-indexed arrays;
/// `bh_unicodelen` reads codepoint count.
#[derive(Debug)]
struct PyreUnicodeLenFieldDescr;

impl Descr for PyreUnicodeLenFieldDescr {}

impl FieldDescr for PyreUnicodeLenFieldDescr {
    fn offset(&self) -> usize {
        UNICODE_LEN_OFFSET
    }
    fn field_size(&self) -> usize {
        // `W_UnicodeObject.len` is a `usize`: 8 bytes on 64-bit, 4 on wasm32.
        // A hardcoded 8 reads the adjacent field into the high half on a
        // 32-bit target.
        std::mem::size_of::<usize>()
    }
    fn field_type(&self) -> Type {
        Type::Int
    }
    fn is_field_signed(&self) -> bool {
        true
    }
    fn field_name(&self) -> &'static str {
        "W_UnicodeObject.len"
    }
}

/// ArrayDescr for STR (byte string per `rstr.py:1226 Array(Char)`).
/// `len_descr` → `byte_len` field.
#[derive(Debug)]
struct PyreStrDescr;

/// ArrayDescr for UNICODE (codepoint string).
/// `len_descr` → `len` (codepoint count) field.
#[derive(Debug)]
struct PyreUnicodeDescr;

const PYRE_STR_BYTE_LEN_DESCR: PyreStrByteLenFieldDescr = PyreStrByteLenFieldDescr;
const PYRE_UNICODE_LEN_DESCR: PyreUnicodeLenFieldDescr = PyreUnicodeLenFieldDescr;
const PYRE_STR_DESCR: PyreStrDescr = PyreStrDescr;
const PYRE_UNICODE_DESCR: PyreUnicodeDescr = PyreUnicodeDescr;

impl Descr for PyreStrDescr {}

impl ArrayDescr for PyreStrDescr {
    fn base_size(&self) -> usize {
        W_UNICODE_OBJECT_SIZE
    }
    fn item_size(&self) -> usize {
        1
    }
    fn type_id(&self) -> u32 {
        W_UNICODE_GC_TYPE_ID as u32
    }
    fn item_type(&self) -> Type {
        Type::Int
    }
    fn is_item_signed(&self) -> bool {
        false
    }
    fn len_descr(&self) -> Option<&dyn FieldDescr> {
        Some(&PYRE_STR_BYTE_LEN_DESCR)
    }
}

impl Descr for PyreUnicodeDescr {}

impl ArrayDescr for PyreUnicodeDescr {
    fn base_size(&self) -> usize {
        W_UNICODE_OBJECT_SIZE
    }
    fn item_size(&self) -> usize {
        4
    }
    fn type_id(&self) -> u32 {
        W_UNICODE_GC_TYPE_ID as u32
    }
    fn item_type(&self) -> Type {
        Type::Int
    }
    fn is_item_signed(&self) -> bool {
        false
    }
    fn len_descr(&self) -> Option<&dyn FieldDescr> {
        Some(&PYRE_UNICODE_LEN_DESCR)
    }
}

/// `Cpu` impl for pyre's runtime.  Delegates to `DefaultCpu` for the
/// methods `DefaultCpu` overrides (`cls_of_box` / `cls_of_gcref` /
/// `bh_getfield_gc_{i,r,f}`) and exposes pyre-specific descrs for the
/// str / unicode family.  `bh_strgetitem` / `bh_unicodegetitem` follow
/// the `W_UnicodeObject.value: *mut String` indirection that PyPy's STR
/// layout does not need (PyPy stores chars in-line after the header).
pub struct PyreCpu(DefaultCpu);

impl PyreCpu {
    pub fn new() -> Self {
        Self(DefaultCpu)
    }
}

impl Default for PyreCpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu for PyreCpu {
    fn cls_of_box(&self, box_: &BoxRef) -> i64 {
        self.0.cls_of_box(box_)
    }
    fn cls_of_gcref(&self, gcref: GcRef) -> i64 {
        self.0.cls_of_gcref(gcref)
    }
    fn bh_getfield_gc_i(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> i64 {
        self.0.bh_getfield_gc_i(struct_ptr, fd)
    }
    fn bh_getfield_gc_r(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> GcRef {
        self.0.bh_getfield_gc_r(struct_ptr, fd)
    }
    fn bh_getfield_gc_f(&self, struct_ptr: usize, fd: &dyn FieldDescr) -> f64 {
        self.0.bh_getfield_gc_f(struct_ptr, fd)
    }

    fn str_descr(&self) -> Option<&dyn ArrayDescr> {
        Some(&PYRE_STR_DESCR)
    }
    fn unicode_descr(&self) -> Option<&dyn ArrayDescr> {
        Some(&PYRE_UNICODE_DESCR)
    }

    fn bh_strlen(&self, string: GcRef) -> Option<i64> {
        // RPython STR is `Array(Char)` byte string (`rstr.py:1226-1228`);
        // `llmodel.py:667 bh_strlen` returns the byte count.
        // `str_descr().len_descr()` reads `W_UnicodeObject.byte_len` for the
        // compiled path; this override follows the `*mut Wtf8Buf` indirection
        // directly for the blackhole interpreter path.
        if string.is_null() {
            return None;
        }
        let value_addr = string.0 + UNICODE_VALUE_OFFSET;
        let value_ptr = unsafe { *(value_addr as *const *const Wtf8Buf) };
        if value_ptr.is_null() {
            return None;
        }
        let s = unsafe { &*value_ptr };
        Some(s.len() as i64)
    }

    fn bh_strgetitem(&self, string: GcRef, index: i64) -> Option<i64> {
        // RPython STR is `Array(Char)` byte string (`rstr.py:1226-1228`);
        // STRGETITEM returns `ord(char)` = byte value.
        // `intbounds.rs:3109` narrows the result to `[0, 255]`
        // (`vstring.py:393-400 IntBound.make_ge(0).make_lt(256)`).
        // `W_UnicodeObject.value: *mut Wtf8Buf` at `UNICODE_VALUE_OFFSET` —
        // follow the indirection and read the WTF-8 byte at `index`.
        // PyPy's STR stores chars in-line at `base + item_size * index`;
        // pyre diverges structurally so this override replaces the
        // default `bh_getarrayitem_gc_i` routing.
        if string.is_null() {
            return None;
        }
        let value_addr = string.0 + UNICODE_VALUE_OFFSET;
        let value_ptr = unsafe { *(value_addr as *const *const Wtf8Buf) };
        if value_ptr.is_null() {
            return None;
        }
        let s = unsafe { &*value_ptr };
        let bytes = s.as_bytes();
        let i = index as usize;
        if i >= bytes.len() {
            return None;
        }
        Some(bytes[i] as i64)
    }

    fn bh_unicodegetitem(&self, unicode: GcRef, index: i64) -> Option<i64> {
        // RPython UNICODE is codepoint-indexed; UNICODEGETITEM returns
        // the codepoint value.  Pyre's `W_UnicodeObject` stores WTF-8, so
        // walk codepoints via `code_points().nth(index)`; `to_u32`
        // yields the ordinal (including lone surrogates D800-DFFF).
        if unicode.is_null() {
            return None;
        }
        let value_addr = unicode.0 + UNICODE_VALUE_OFFSET;
        let value_ptr = unsafe { *(value_addr as *const *const Wtf8Buf) };
        if value_ptr.is_null() {
            return None;
        }
        let s = unsafe { &*value_ptr };
        let i = index as usize;
        s.code_points().nth(i).map(|c| c.to_u32() as i64)
    }
}

/// Shared `Arc<dyn Cpu>` for pyre.  Initialised once per process and
/// installed on `MetaInterp<PyreMeta>` via `set_cpu` at the
/// `trace_bytecode` entry point.
pub fn shared() -> Arc<dyn Cpu> {
    static CELL: OnceLock<Arc<dyn Cpu>> = OnceLock::new();
    CELL.get_or_init(|| Arc::new(PyreCpu::new()) as Arc<dyn Cpu>)
        .clone()
}
