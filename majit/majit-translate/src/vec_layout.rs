//! Buffer-pointer and length offsets of `alloc::vec::Vec<T>`.
//!
//! Charon's type decl for `alloc::vec::Vec` is `Opaque` with `layout: []`
//! (`rhai.ullbc`, `lang_item: Vec`), so neither the type-decl walker nor the
//! codewriter's struct-field table has a `RawVec` / `RawVecInner` row to read.
//! The three words are measured once from the `Vec` this crate is compiled
//! against — the same compiler that lays out the interpreter's fields.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VecLayout {
    pub ptr_offset: usize,
    pub len_offset: usize,
    pub cap_offset: usize,
}

pub fn probe() -> VecLayout {
    static CELL: OnceLock<VecLayout> = OnceLock::new();
    *CELL.get_or_init(measure)
}

fn measure() -> VecLayout {
    const WORD: usize = std::mem::size_of::<usize>();
    const {
        assert!(std::mem::size_of::<Vec<u8>>() == 3 * WORD);
        assert!(std::mem::size_of::<Vec<i64>>() == 3 * WORD);
        assert!(std::mem::align_of::<Vec<u8>>() == WORD);
        assert!(std::mem::align_of::<Vec<i64>>() == WORD);
    }
    let mut sample = Vec::<u8>::with_capacity(4);
    sample.push(1);
    assert_ne!(
        sample.len(),
        sample.capacity(),
        "len and cap must occupy different words"
    );
    let words =
        unsafe { std::slice::from_raw_parts((&sample as *const Vec<u8>).cast::<usize>(), 3) };
    let ptr = sample.as_ptr() as usize;
    let len = sample.len();
    let cap = sample.capacity();
    let mut ptr_offset = None;
    let mut len_offset = None;
    let mut cap_offset = None;
    for (index, word) in words.iter().copied().enumerate() {
        let offset = index * WORD;
        if word == ptr {
            ptr_offset = Some(offset);
        } else if word == len {
            len_offset = Some(offset);
        } else if word == cap {
            cap_offset = Some(offset);
        }
    }
    VecLayout {
        ptr_offset: ptr_offset.expect("Vec buffer-pointer word"),
        len_offset: len_offset.expect("Vec length word"),
        cap_offset: cap_offset.expect("Vec capacity word"),
    }
}

/// A field-layout spelling that is an inline `Vec<T>`, not `Box<Vec<T>>`
/// and not `&Vec<T>`.
pub fn field_layout_is_inline_vec(type_str: &str) -> bool {
    let leaf = type_str.rsplit("::").next().unwrap_or(type_str);
    leaf.starts_with("Vec<")
}
