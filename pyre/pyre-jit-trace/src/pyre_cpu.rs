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

use majit_ir::operand::Operand;
use majit_ir::{ArrayDescr, Descr, FieldDescr, GcRef, Type};
use majit_metainterp::cpu::{Cpu, DefaultCpu};
use pyre_object::rutf8::Utf8IndexStorage;
use pyre_object::unicodeobject::{
    UNICODE_BYTE_LEN_OFFSET, UNICODE_INDEX_STORAGE_OFFSET, UNICODE_LEN_OFFSET,
    UNICODE_VALUE_OFFSET, W_UNICODE_GC_TYPE_ID, W_UNICODE_OBJECT_SIZE,
};
use rustpython_wtf8::Wtf8Buf;

/// FieldDescr for `W_UnicodeObject.byte_len` — UTF-8 byte count.
/// RPython STR is `Array(Char)` byte string (`rstr.py`);
/// `llmodel.py bh_strlen` reads byte count.
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

/// ArrayDescr for STR (byte string per `rstr.py Array(Char)`).
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

/// The offset a blackhole item read takes, refused unless it is one: `as
/// usize` wraps a negative operand and truncates one wider than the target's
/// `usize`, and either turns the bounds test that follows into a read of the
/// wrong element.
fn item_index(index: i64) -> Option<usize> {
    usize::try_from(index).ok()
}

impl Cpu for PyreCpu {
    fn cls_of_box(&self, box_: &Operand) -> i64 {
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
        // RPython STR is `Array(Char)` byte string (`rstr.py`);
        // `llmodel.py bh_strlen` returns the byte count.
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
        // RPython STR is `Array(Char)` byte string (`rstr.py`);
        // STRGETITEM returns `ord(char)` = byte value.
        // `intbounds.rs`'s `propagate_postprocess` narrows the result to `[0, 255]`
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
        let i = item_index(index)?;
        if i >= bytes.len() {
            return None;
        }
        Some(bytes[i] as i64)
    }

    fn bh_unicodegetitem(&self, unicode: GcRef, index: i64) -> Option<i64> {
        // RPython UNICODE is codepoint-indexed; UNICODEGETITEM returns the
        // codepoint value, `to_u32` (including lone surrogates D800-DFFF).
        // Pyre's `W_UnicodeObject` stores WTF-8, where a codepoint index is a
        // byte offset only for an ASCII payload, so this resolves it the two
        // ways `w_str_codepoint_at` does.
        //
        // Neither arm builds the index table: a blackhole runs inside a deopt,
        // so it reads a table that is already there and otherwise walks. The
        // walk is the cost the table exists to remove, and it is the one
        // upstream's array read never pays.
        if unicode.is_null() {
            return None;
        }
        let value_addr = unicode.0 + UNICODE_VALUE_OFFSET;
        let value_ptr = unsafe { *(value_addr as *const *const Wtf8Buf) };
        if value_ptr.is_null() {
            return None;
        }
        let s = unsafe { &*value_ptr };
        let i = item_index(index)?;
        let len = unsafe { *((unicode.0 + UNICODE_LEN_OFFSET) as *const usize) };
        if i >= len {
            return None;
        }
        let byte_len = unsafe { *((unicode.0 + UNICODE_BYTE_LEN_OFFSET) as *const usize) };
        // `w_str_is_ascii` — one byte per codepoint, so the index is the offset.
        if len == byte_len {
            return Some(s.as_bytes()[i] as i64);
        }
        let storage = unsafe {
            *((unicode.0 + UNICODE_INDEX_STORAGE_OFFSET) as *const *const Utf8IndexStorage)
        };
        if storage.is_null() {
            return s.code_points().nth(i).map(|c| c.to_u32() as i64);
        }
        Some(pyre_object::rutf8::codepoint_at_index(s, unsafe { &*storage }, i).to_u32() as i64)
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

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_wtf8::CodePoint;

    /// Every arm of `bh_unicodegetitem` must answer what a codepoint walk
    /// answers. The arms are chosen by two facts about the operand — whether
    /// it is ASCII, and whether its index table has been built — so each case
    /// below puts a string in one of those states and compares the whole
    /// index range against the walk.
    fn agrees_with_walk(obj: pyre_object::PyObjectRef) {
        let cpu = PyreCpu::new();
        let gc = GcRef(obj as usize);
        let walk: Vec<i64> = unsafe { pyre_object::w_str_get_wtf8(obj) }
            .code_points()
            .map(|c| c.to_u32() as i64)
            .collect();
        for (i, expected) in walk.iter().enumerate() {
            assert_eq!(
                cpu.bh_unicodegetitem(gc, i as i64),
                Some(*expected),
                "index {i}"
            );
        }
        assert_eq!(
            cpu.bh_unicodegetitem(gc, walk.len() as i64),
            None,
            "one past the end"
        );
        assert_eq!(cpu.bh_unicodegetitem(gc, -1), None, "negative index");
    }

    #[test]
    fn bh_unicodegetitem_ascii_reads_the_byte() {
        let obj = pyre_object::w_str_new("hello");
        assert!(unsafe { pyre_object::unicodeobject::w_str_is_ascii(obj) });
        agrees_with_walk(obj);
    }

    #[test]
    fn bh_unicodegetitem_wide_without_a_table_walks() {
        let obj = pyre_object::w_str_new("héllo wörld ☃");
        assert!(!unsafe { pyre_object::unicodeobject::w_str_is_ascii(obj) });
        agrees_with_walk(obj);
    }

    #[test]
    fn bh_unicodegetitem_wide_with_a_table_reads_the_table() {
        // Long enough to span more than one 64-codepoint group, so the read
        // exercises `baseindex` selection rather than only the first entry.
        let obj = pyre_object::w_str_new(&"ábç".repeat(60));
        // Force the lazy build the blackhole arm refuses to do itself.
        assert!(unsafe { pyre_object::w_str_codepoint_at(obj, 100) }.is_some());
        agrees_with_walk(obj);
    }

    #[test]
    fn bh_unicodegetitem_yields_a_lone_surrogate() {
        let mut buf = rustpython_wtf8::Wtf8Buf::new();
        buf.push(CodePoint::from_char('a'));
        buf.push(CodePoint::from_u32(0xD800).unwrap());
        buf.push(CodePoint::from_char('b'));
        let obj = pyre_object::unicodeobject::w_str_from_wtf8(buf);
        let cpu = PyreCpu::new();
        assert_eq!(cpu.bh_unicodegetitem(GcRef(obj as usize), 1), Some(0xD800));
        agrees_with_walk(obj);
    }
}
