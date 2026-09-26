//! Buffer-pointer and length offsets of `alloc::vec::Vec<T>`.
//!
//! Charon's type decl for `alloc::vec::Vec` is `Opaque` with no field
//! layout, so the component order is taken from the three-word value this
//! crate is compiled against (pointer, length, capacity — `Global` is a
//! ZST). Offsets are that word index times [`crate::layout::target_word_size`],
//! not the host `usize` the measurement ran on.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VecLayout {
    pub ptr_offset: usize,
    pub len_offset: usize,
    pub cap_offset: usize,
}

pub fn probe() -> VecLayout {
    static CELL: OnceLock<VecLayout> = OnceLock::new();
    *CELL.get_or_init(|| layout_for_word(crate::layout::target_word_size()))
}

/// `Vec` component offsets for a target whose pointer is `word` bytes.
///
/// The word index of each component is fixed by the host measurement
/// (`measure_indices`); only the byte stride changes with the target.
pub fn layout_for_word(word: usize) -> VecLayout {
    let (ptr_index, len_index, cap_index) = component_indices();
    VecLayout {
        ptr_offset: ptr_index * word,
        len_offset: len_index * word,
        cap_offset: cap_index * word,
    }
}

fn component_indices() -> (usize, usize, usize) {
    static CELL: OnceLock<(usize, usize, usize)> = OnceLock::new();
    *CELL.get_or_init(measure_indices)
}

fn measure_indices() -> (usize, usize, usize) {
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
    let mut ptr_index = None;
    let mut len_index = None;
    let mut cap_index = None;
    for (index, word) in words.iter().copied().enumerate() {
        if word == ptr {
            ptr_index = Some(index);
        } else if word == len {
            len_index = Some(index);
        } else if word == cap {
            cap_index = Some(index);
        }
    }
    let ptr_index = ptr_index.expect("Vec buffer-pointer word");
    let len_index = len_index.expect("Vec length word");
    let cap_index = cap_index.expect("Vec capacity word");
    assert!(
        ptr_index != len_index && ptr_index != cap_index && len_index != cap_index,
        "Vec components must occupy three distinct words"
    );
    (ptr_index, len_index, cap_index)
}

/// A field-layout spelling that is an inline `Vec<T>`, not `Box<Vec<T>>`
/// and not `&Vec<T>`.
pub fn field_layout_is_inline_vec(type_str: &str) -> bool {
    let leaf = type_str.rsplit("::").next().unwrap_or(type_str);
    leaf.starts_with("Vec<")
}

#[cfg(test)]
mod tests {
    use super::layout_for_word;

    #[test]
    fn component_offsets_follow_the_target_word() {
        let wide = layout_for_word(8);
        assert_eq!(
            (wide.ptr_offset, wide.len_offset),
            (8, 16),
            "a 64-bit target places the buffer and length at 8 and 16"
        );
        let wasm = layout_for_word(4);
        assert_eq!(
            (wasm.ptr_offset, wasm.len_offset),
            (4, 8),
            "a wasm32 target places the buffer and length at 4 and 8"
        );
    }
}
