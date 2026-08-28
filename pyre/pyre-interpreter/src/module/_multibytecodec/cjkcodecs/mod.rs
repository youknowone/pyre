//! Rust ports of PyPy's `pypy/module/_multibytecodec/src/cjkcodecs` units.

mod cn;
mod hk;
mod kr;
mod mappings_cn;
mod mappings_hk;
mod mappings_kr;
mod mappings_tw;
mod tw;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Codec {
    EucKr,
    Cp949,
    Johab,
    Big5,
    Cp950,
    Big5Hkscs,
    Gb2312,
    Gbk,
    Gb18030,
    Hz,
}

impl Codec {
    pub(super) fn from_name(name: &str) -> Option<Self> {
        match name {
            "euc_kr" => Some(Self::EucKr),
            "cp949" => Some(Self::Cp949),
            "johab" => Some(Self::Johab),
            "big5" => Some(Self::Big5),
            "cp950" => Some(Self::Cp950),
            "big5hkscs" => Some(Self::Big5Hkscs),
            "gb2312" => Some(Self::Gb2312),
            "gbk" => Some(Self::Gbk),
            "gb18030" => Some(Self::Gb18030),
            "hz" => Some(Self::Hz),
            _ => None,
        }
    }
}

pub(super) enum DecodeOne {
    Char(u32, usize),
    Pair(u32, u32, usize),
    Skip(usize),
    Incomplete,
    Illegal(usize),
}

pub(super) enum EncodeOne {
    Bytes([u8; 8], usize, usize),
    Incomplete,
    Illegal(usize),
}

pub(super) fn decode_one(codec: Codec, input: &[u8], state: &mut [u8; 8]) -> DecodeOne {
    debug_assert!(!input.is_empty());
    match codec {
        Codec::EucKr => kr::decode_euc_kr(input),
        Codec::Cp949 => kr::decode_cp949(input),
        Codec::Johab => kr::decode_johab(input),
        Codec::Big5 => tw::decode_big5(input),
        Codec::Cp950 => tw::decode_cp950(input),
        Codec::Big5Hkscs => hk::decode_big5hkscs(input),
        Codec::Gb2312 => cn::decode_gb2312(input),
        Codec::Gbk => cn::decode_gbk(input),
        Codec::Gb18030 => cn::decode_gb18030(input),
        Codec::Hz => cn::decode_hz(input, state),
    }
}

pub(super) fn encode_one(
    codec: Codec,
    input: &[u32],
    final_input: bool,
    state: &mut [u8; 8],
) -> EncodeOne {
    debug_assert!(!input.is_empty());
    match codec {
        Codec::EucKr => kr::encode_euc_kr(input[0]),
        Codec::Cp949 => kr::encode_cp949(input[0]),
        Codec::Johab => kr::encode_johab(input[0]),
        Codec::Big5 => tw::encode_big5(input[0]),
        Codec::Cp950 => tw::encode_cp950(input[0]),
        Codec::Big5Hkscs => hk::encode_big5hkscs(input, final_input),
        Codec::Gb2312 => cn::encode_gb2312(input[0]),
        Codec::Gbk => cn::encode_gbk(input[0]),
        Codec::Gb18030 => cn::encode_gb18030(input[0]),
        Codec::Hz => cn::encode_hz(input[0], state),
    }
}

pub(super) fn encode_reset(codec: Codec, state: &mut [u8; 8]) -> Option<([u8; 8], usize)> {
    match codec {
        Codec::Hz => cn::reset_hz(state),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_one_and_two_byte_candidate_is_total() {
        let codecs = [
            Codec::EucKr,
            Codec::Cp949,
            Codec::Johab,
            Codec::Big5,
            Codec::Cp950,
            Codec::Big5Hkscs,
            Codec::Gb2312,
            Codec::Gbk,
            Codec::Gb18030,
            Codec::Hz,
        ];
        for codec in codecs {
            for first in 0..=u8::MAX {
                let _ = decode_one(codec, &[first], &mut [0; 8]);
                for second in 0..=u8::MAX {
                    let _ = decode_one(codec, &[first, second], &mut [0; 8]);
                }
            }
        }
    }
}
