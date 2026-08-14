//! `array_fields = { Struct::field => ElementType }` — the element descr's
//! sign flag.
//!
//! `descr.py:240-254 get_type_flag(FIELDTYPE)` reads signedness off the
//! declared element type, and a load honours it: a signed descr sign-extends
//! the sub-word it read.  So an element declared `u8` whose byte is `0x80`
//! must reach the trace as `128` — the value the concrete Rust read produces —
//! not as `-128`.
//!
//! Nothing downstream can recover from getting this wrong.  The sign extension
//! happens inside the traced load and again in the backend, so the wrong value
//! simply *is* the value; there is no later point that could tell it from a
//! genuine negative element.

use majit_macros::jit_inline;
use majit_metainterp::jitcode::{CanonicalBhDescr, RuntimeBhDescr};

/// Unsigned elements.
#[repr(C)]
struct ByteStack {
    data: *mut u8,
    sp: usize,
}

/// Signed elements, so the unsigned assertion has a counterpart: without it a
/// change that hard-coded *unsigned* would pass just as the hard-coded
/// *signed* one did.
#[repr(C)]
struct WordStack {
    data: *mut i64,
    sp: usize,
}

#[jit_inline(
    ref_params = { stack: ref(ByteStack) },
    array_fields = { ByteStack::data => u8 },
)]
fn byte_elem_read(stack: usize) -> i64 {
    let sp = stack.sp;
    let value = stack.data[sp];
    value as i64
}

#[jit_inline(
    ref_params = { stack: ref(WordStack) },
    array_fields = { WordStack::data => i64 },
)]
fn word_elem_read(stack: usize) -> i64 {
    let sp = stack.sp;
    let value = stack.data[sp];
    value
}

/// Every array element descr the jitcode interns, as
/// `(itemsize, is_item_signed)`.
fn array_element_descrs(jc: &majit_metainterp::JitCode) -> Vec<(usize, bool)> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while let Some(entry) = jc.descr_at(index) {
        if let RuntimeBhDescr::Descr(descr) = entry
            && let CanonicalBhDescr::Array {
                itemsize,
                is_item_signed,
                ..
            } = &**descr
        {
            out.push((*itemsize, *is_item_signed));
        }
        index += 1;
    }
    out
}

#[test]
fn an_unsigned_element_type_interns_an_unsigned_element_descr() {
    let mut asm = majit_metainterp::Assembler::new();
    let jitcode = __majit_inline_jitcode_byte_elem_read_with_asm(&mut asm);
    let descrs = array_element_descrs(&jitcode);
    let byte_signs: Vec<bool> = descrs
        .iter()
        .filter(|(size, _)| *size == 1)
        .map(|(_, signed)| *signed)
        .collect();
    assert!(
        !byte_signs.is_empty(),
        "the fixture must intern a 1-byte element descr; interned {descrs:?}"
    );
    assert!(
        byte_signs.iter().all(|signed| !signed),
        "`u8` elements must intern an UNSIGNED descr, or a 0x80 byte loads as \
         -128 where the interpreter reads 128; interned {descrs:?}"
    );
}

#[test]
fn a_signed_element_type_still_interns_a_signed_element_descr() {
    let mut asm = majit_metainterp::Assembler::new();
    let jitcode = __majit_inline_jitcode_word_elem_read_with_asm(&mut asm);
    let descrs = array_element_descrs(&jitcode);
    let word_signs: Vec<bool> = descrs
        .iter()
        .filter(|(size, _)| *size == std::mem::size_of::<i64>())
        .map(|(_, signed)| *signed)
        .collect();
    assert!(
        !word_signs.is_empty(),
        "the fixture must intern an 8-byte element descr; interned {descrs:?}"
    );
    assert!(
        word_signs.iter().all(|signed| *signed),
        "`i64` elements must stay SIGNED; interned {descrs:?}"
    );
}
