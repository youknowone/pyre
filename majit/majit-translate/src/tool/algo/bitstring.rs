//! Re-export of `rpython/tool/algo/bitstring.py` helpers.
//!
//! The concrete implementation lives in `majit-ir` because both the
//! translator and metainterp/backend crates consume EffectInfo bitstrings.
//! Keep this module so the source path mirrors PyPy's `rpython.tool.algo`.

pub use majit_ir::bitstring::{bitcheck, make_bitstring, num_bits};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_bitstring_matches_rpython_tests() {
        assert_eq!(make_bitstring(&[]), b"");
        assert_eq!(make_bitstring(&[0]), b"\x01");
        assert_eq!(make_bitstring(&[7]), b"\x80");
        assert_eq!(make_bitstring(&[8]), b"\x00\x01");
        assert_eq!(make_bitstring(&[2, 4, 20]), b"\x14\x00\x10");
    }

    #[test]
    fn bitcheck_matches_rpython_tests() {
        assert!(bitcheck(b"\x01", 0));
        assert!(!bitcheck(b"\x01", 1));
        assert!(!bitcheck(b"\x01", 10));
        let set_bits: Vec<u32> = (0..32).filter(|n| bitcheck(b"\x14\x00\x10", *n)).collect();
        assert_eq!(set_bits, vec![2, 4, 20]);
    }

    #[test]
    fn num_bits_matches_rpython_tests() {
        assert_eq!(num_bits(b""), 0);
        assert_eq!(num_bits(b"a"), 8);
        assert_eq!(num_bits(b"bcd"), 24);
    }
}
