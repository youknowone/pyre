//! Host Rust layout of a raw struct or enum: field offsets and the tag
//! rustc actually stores, including a niche.

use majit_charon_reader::ullbc::{TagEncoding, TagLayout};

/// Byte layout of one concrete type, as Charon recorded it for the
/// extraction target. Independent of the explicit sum shell the front
/// uses when a payload would overwrite a tag.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HostLayout {
    pub size: u64,
    pub align: u64,
    /// `variant_field_offsets[variant][field]` is the byte offset of that
    /// field. A struct has one variant.
    pub variant_field_offsets: Vec<Vec<u64>>,
    pub tag: Option<TagLayout>,
}

pub fn host_layout_from_type(layout: &majit_charon_reader::ullbc::TypeLayout) -> HostLayout {
    HostLayout {
        size: layout.size.unwrap_or(0),
        align: layout.align.unwrap_or(1),
        variant_field_offsets: layout
            .variant_layouts
            .iter()
            .map(|variant| variant.field_offsets.clone())
            .collect(),
        tag: layout.tag(),
    }
}

fn mask(bits: u32, raw: u128) -> u128 {
    if bits >= 128 {
        raw
    } else {
        raw & ((1u128 << bits) - 1)
    }
}

/// Variant index of `raw_tag_value` under rustc's niche rule.
///
/// `relative = raw.wrapping_sub(niche_start)` in the tag width.
/// `relative <= end - start` selects `start + relative`; every other
/// value is the untagged variant. A direct tag matches `tags[index]`.
pub fn host_variant_index(tag: &TagLayout, raw_tag_value: u128) -> usize {
    let raw = mask(tag.bits, raw_tag_value);
    match &tag.encoding {
        TagEncoding::Direct { tags } => tags
            .iter()
            .position(|known| mask(tag.bits, *known) == raw)
            .unwrap_or(0),
        TagEncoding::Niche {
            untagged_variant,
            niche_variants,
            niche_start,
        } => {
            let relative = mask(tag.bits, raw.wrapping_sub(*niche_start));
            let span = (niche_variants.end() - niche_variants.start()) as u128;
            if relative <= span {
                niche_variants.start() + relative as usize
            } else {
                *untagged_variant
            }
        }
    }
}

/// Tag word a niche or direct variant stores. `None` for the untagged
/// niche variant, which has no single tag value.
pub fn host_tag_value_for_variant(tag: &TagLayout, variant: usize) -> Option<u128> {
    match &tag.encoding {
        TagEncoding::Direct { tags } => tags.get(variant).map(|word| mask(tag.bits, *word)),
        TagEncoding::Niche {
            untagged_variant,
            niche_variants,
            niche_start,
        } => {
            if variant == *untagged_variant || !niche_variants.contains(&variant) {
                None
            } else {
                let step = (variant - niche_variants.start()) as u128;
                Some(mask(tag.bits, niche_start.wrapping_add(step)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::NonNull;

    fn tag_word(bytes: &[u8], offset: usize, bits: u32) -> u128 {
        let width = (bits / 8) as usize;
        let mut buf = [0u8; 16];
        buf[..width].copy_from_slice(&bytes[offset..offset + width]);
        u128::from_le_bytes(buf)
    }

    fn bytes_of<T>(value: &T) -> Vec<u8> {
        let mut out = vec![0u8; std::mem::size_of::<T>()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                value as *const T as *const u8,
                out.as_mut_ptr(),
                out.len(),
            );
        }
        out
    }

    #[test]
    fn niche_bool_and_pointer_and_direct_tags_round_trip() {
        #[derive(Clone, Copy)]
        #[allow(dead_code)]
        enum BoolNiche {
            T(bool),
            U,
            V,
        }
        let bool_tag = TagLayout {
            offset: 0,
            signed: false,
            bits: 8,
            encoding: TagEncoding::Niche {
                untagged_variant: 0,
                niche_variants: 1..=2,
                niche_start: 2,
            },
        };
        let samples = [
            (BoolNiche::T(false), 0usize),
            (BoolNiche::T(true), 0),
            (BoolNiche::U, 1),
            (BoolNiche::V, 2),
        ];
        for (value, variant) in samples {
            let word = tag_word(&bytes_of(&value), 0, 8);
            assert_eq!(host_variant_index(&bool_tag, word), variant);
            if let Some(encoded) = host_tag_value_for_variant(&bool_tag, variant) {
                assert_eq!(host_variant_index(&bool_tag, encoded), variant);
            }
        }
        assert!(host_tag_value_for_variant(&bool_tag, 0).is_none());

        let none_tag = TagLayout {
            offset: 0,
            signed: false,
            bits: 64,
            encoding: TagEncoding::Niche {
                untagged_variant: 1,
                niche_variants: 0..=0,
                niche_start: 0,
            },
        };
        let none = Option::<NonNull<u8>>::None;
        let some = NonNull::new(0x10 as *mut u8).map(Some).unwrap();
        assert_eq!(
            host_variant_index(&none_tag, tag_word(&bytes_of(&none), 0, 64)),
            0
        );
        assert_eq!(
            host_variant_index(&none_tag, tag_word(&bytes_of(&some), 0, 64)),
            1
        );
        assert_eq!(host_tag_value_for_variant(&none_tag, 0), Some(0));

        #[derive(Clone, Copy)]
        enum Direct {
            A,
            B,
            C,
        }
        let direct = TagLayout {
            offset: 0,
            signed: false,
            bits: 8,
            encoding: TagEncoding::Direct {
                tags: vec![0, 1, 2],
            },
        };
        for (value, variant) in [(Direct::A, 0usize), (Direct::B, 1), (Direct::C, 2)] {
            let word = tag_word(&bytes_of(&value), 0, 8);
            assert_eq!(word, variant as u128);
            assert_eq!(host_variant_index(&direct, word), variant);
            assert_eq!(host_tag_value_for_variant(&direct, variant), Some(word));
        }

        #[derive(Clone, Copy)]
        #[allow(dead_code)]
        enum RefNiche {
            A(&'static u8),
            B,
            C,
        }
        let static_byte: &'static u8 = &7;
        let a = bytes_of(&RefNiche::A(static_byte));
        let b = bytes_of(&RefNiche::B);
        let c = bytes_of(&RefNiche::C);
        let b_word = tag_word(&b, 0, 64);
        let c_word = tag_word(&c, 0, 64);
        let a_word = tag_word(&a, 0, 64);
        let (niche_start, first, second) = if b_word < c_word {
            (b_word, 1usize, 2usize)
        } else {
            (c_word, 2, 1)
        };
        let step = c_word.abs_diff(b_word);
        assert_eq!(step, 1, "ref niche tags {b_word:#x} {c_word:#x}");
        let ref_tag = TagLayout {
            offset: 0,
            signed: false,
            bits: 64,
            encoding: TagEncoding::Niche {
                untagged_variant: 0,
                niche_variants: 1..=2,
                niche_start,
            },
        };
        assert_eq!(host_variant_index(&ref_tag, a_word), 0);
        assert_eq!(host_variant_index(&ref_tag, b_word), first);
        assert_eq!(host_variant_index(&ref_tag, c_word), second);
        assert_eq!(host_tag_value_for_variant(&ref_tag, first), Some(b_word));
        assert_eq!(host_tag_value_for_variant(&ref_tag, second), Some(c_word));
    }
}
