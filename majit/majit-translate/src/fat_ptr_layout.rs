//! Data-pointer and length offsets of a fat pointer (`Box<[T]>`, `Box<str>`,
//! `Box<dyn Trait>`).
//!
//! The two words are the data address and the metadata. Charon does not
//! record that split, and a `Vec<T>`'s `(capacity, pointer, length)` order
//! is a different value. The word index is measured on the host the way
//! [`crate::vec_layout`] measures `Vec`; the byte offset is that index
//! times [`crate::layout::target_word_size`].

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FatPtrLayout {
    pub data_offset: usize,
    pub len_offset: usize,
}

pub fn probe() -> FatPtrLayout {
    static CELL: OnceLock<FatPtrLayout> = OnceLock::new();
    *CELL.get_or_init(|| layout_for_word(crate::layout::target_word_size()))
}

/// Fat-pointer component offsets for a target whose pointer is `word` bytes.
pub fn layout_for_word(word: usize) -> FatPtrLayout {
    let (data_index, len_index) = component_indices();
    FatPtrLayout {
        data_offset: data_index * word,
        len_offset: len_index * word,
    }
}

/// `&dyn Trait` / `&mut dyn Trait` / `Box<dyn Trait>` occupy two words:
/// the data pointer and the vtable (metadata) pointer.
pub fn spelling_is_dyn_fat_ptr(s: &str) -> bool {
    let s = s.trim();
    let inner = s
        .strip_prefix("&mut ")
        .or_else(|| s.strip_prefix('&'))
        .or_else(|| {
            s.strip_prefix("Box<")
                .and_then(|rest| rest.strip_suffix('>'))
        })
        .unwrap_or(s)
        .trim();
    inner.starts_with("dyn ")
}

fn component_indices() -> (usize, usize) {
    static CELL: OnceLock<(usize, usize)> = OnceLock::new();
    *CELL.get_or_init(measure_indices)
}

fn measure_indices() -> (usize, usize) {
    const WORD: usize = std::mem::size_of::<usize>();
    const {
        assert!(std::mem::size_of::<Box<[u8]>>() == 2 * WORD);
        assert!(std::mem::align_of::<Box<[u8]>>() == WORD);
    }
    let sample: Box<[u8]> = vec![1u8, 2, 3, 4, 5].into_boxed_slice();
    assert_ne!(sample.len(), sample.as_ptr() as usize);
    let words =
        unsafe { std::slice::from_raw_parts((&sample as *const Box<[u8]>).cast::<usize>(), 2) };
    let data = sample.as_ptr() as usize;
    let len = sample.len();
    let mut data_index = None;
    let mut len_index = None;
    for (index, word) in words.iter().copied().enumerate() {
        if word == data {
            data_index = Some(index);
        } else if word == len {
            len_index = Some(index);
        }
    }
    let data_index = data_index.expect("fat-pointer data word");
    let len_index = len_index.expect("fat-pointer length word");
    assert_ne!(data_index, len_index);
    (data_index, len_index)
}

#[cfg(test)]
mod tests {
    use super::layout_for_word;

    #[test]
    fn dyn_trait_spellings_are_fat() {
        assert!(super::spelling_is_dyn_fat_ptr("&dyn Storage"));
        assert!(super::spelling_is_dyn_fat_ptr("&mut dyn Storage"));
        assert!(super::spelling_is_dyn_fat_ptr("Box<dyn Storage>"));
        assert!(!super::spelling_is_dyn_fat_ptr("&Holder"));
        assert!(!super::spelling_is_dyn_fat_ptr("Box<Holder>"));
    }

    #[test]
    fn component_offsets_follow_the_target_word() {
        let wide = layout_for_word(8);
        assert_eq!(
            (wide.data_offset, wide.len_offset),
            (0, 8),
            "a 64-bit target places the data pointer then the length"
        );
        let wasm = layout_for_word(4);
        assert_eq!(
            (wasm.data_offset, wasm.len_offset),
            (0, 4),
            "a wasm32 target places the data pointer then the length"
        );
    }
}
