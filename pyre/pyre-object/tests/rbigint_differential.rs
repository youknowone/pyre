//! Deterministic, arbitrary-size differential checks for the RPython rbigint
//! port. Malachite is a dev-only oracle; the pyre runtime remains entirely on
//! `pyre_object::rbigint::RBigInt`.

use malachite_bigint::{BigInt as Oracle, Sign as OracleSign};
use pyre_object::rbigint::{RBigInt, RBigIntSign};

fn next_u64(state: &mut u64) -> u64 {
    // Fixed xorshift64* stream: reproducible on every target and independent
    // of the `rand` crate's version/algorithm.
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn case(state: &mut u64, max_bytes: usize) -> (RBigInt, Oracle) {
    let len = (next_u64(state) as usize % max_bytes) + 1;
    let mut bytes = vec![0_u8; len];
    for chunk in bytes.chunks_mut(8) {
        let random = next_u64(state).to_le_bytes();
        chunk.copy_from_slice(&random[..chunk.len()]);
    }
    // Include both signs and canonical zero in the same stream.
    if next_u64(state) & 15 == 0 {
        bytes.fill(0);
    }
    let negative = next_u64(state) & 1 != 0 && bytes.iter().any(|&byte| byte != 0);
    let magnitude = RBigInt::frombytes(&bytes, "little", false).unwrap();
    let value = if negative { magnitude.neg() } else { magnitude };
    let oracle = Oracle::from_bytes_le(
        if negative {
            OracleSign::Minus
        } else if bytes.iter().all(|&byte| byte == 0) {
            OracleSign::NoSign
        } else {
            OracleSign::Plus
        },
        &bytes,
    );
    (value, oracle)
}

fn to_oracle(value: &RBigInt) -> Oracle {
    let sign = match value.sign() {
        RBigIntSign::Minus => OracleSign::Minus,
        RBigIntSign::NoSign => OracleSign::NoSign,
        RBigIntSign::Plus => OracleSign::Plus,
    };
    if sign == OracleSign::NoSign {
        return Oracle::from(0);
    }
    let magnitude = value.abs();
    let byte_len = (magnitude.bit_length().unwrap() + 7) / 8;
    let bytes = magnitude.tobytes(byte_len, "little", false).unwrap();
    Oracle::from_bytes_le(sign, &bytes)
}

fn oracle_floor_divmod(a: &Oracle, b: &Oracle) -> (Oracle, Oracle) {
    let mut q = a / b;
    let mut r = a % b;
    if r != Oracle::from(0) && ((a.sign() == OracleSign::Minus) != (b.sign() == OracleSign::Minus))
    {
        q -= Oracle::from(1);
        r += b;
    }
    (q, r)
}

fn oracle_abs(value: &Oracle) -> Oracle {
    if value.sign() == OracleSign::Minus {
        -value
    } else {
        value.clone()
    }
}

fn oracle_gcd(a: &Oracle, b: &Oracle) -> Oracle {
    let mut a = oracle_abs(a);
    let mut b = oracle_abs(b);
    while b != Oracle::from(0) {
        let r = &a % &b;
        a = b;
        b = r;
    }
    a
}

#[test]
fn deterministic_arbitrary_size_arithmetic_matches_independent_oracle() {
    let mut state = 0x7262_6967_696e_7421_u64;
    for iteration in 0..320 {
        let max_bytes = if iteration < 300 { 256 } else { 2048 };
        let (a, oa) = case(&mut state, max_bytes);
        let (b, ob) = case(&mut state, max_bytes);

        assert_eq!(to_oracle(&a), oa, "input a, iteration {iteration}");
        assert_eq!(to_oracle(&b), ob, "input b, iteration {iteration}");
        assert_eq!(
            to_oracle(&a.add(&b)),
            &oa + &ob,
            "add, iteration {iteration}"
        );
        assert_eq!(
            to_oracle(&a.sub(&b)),
            &oa - &ob,
            "sub, iteration {iteration}"
        );
        assert_eq!(
            to_oracle(&a.mul(&b)),
            &oa * &ob,
            "mul, iteration {iteration}"
        );

        assert_eq!(
            to_oracle(&a.and_(&b)),
            &oa & &ob,
            "and, iteration {iteration}"
        );
        assert_eq!(
            to_oracle(&a.or_(&b)),
            &oa | &ob,
            "or, iteration {iteration}"
        );
        assert_eq!(
            to_oracle(&a.xor(&b)),
            &oa ^ &ob,
            "xor, iteration {iteration}"
        );
        assert_eq!(
            to_oracle(&a.invert()),
            !&oa,
            "invert, iteration {iteration}"
        );

        let shift = (next_u64(&mut state) % 768) as i64;
        assert_eq!(
            to_oracle(&a.lshift(shift).unwrap()),
            &oa << shift,
            "lshift, iteration {iteration}"
        );
        assert_eq!(
            to_oracle(&a.rshift(shift, false).unwrap()),
            &oa >> shift,
            "rshift, iteration {iteration}"
        );

        assert_eq!(
            a.bit_length().unwrap() as u64,
            oa.bits(),
            "bit_length, iteration {iteration}"
        );
        let oracle_count: i64 = oa
            .magnitude()
            .to_bytes_le()
            .iter()
            .map(|byte| byte.count_ones() as i64)
            .sum();
        assert_eq!(
            a.bit_count().unwrap(),
            oracle_count,
            "bit_count, iteration {iteration}"
        );

        assert_eq!(
            a.str(0).unwrap(),
            oa.to_string(),
            "decimal format, iteration {iteration}"
        );
        let parsed = RBigInt::fromstr(&oa.to_string(), 10, false).unwrap();
        assert_eq!(
            to_oracle(&parsed),
            oa,
            "decimal parse, iteration {iteration}"
        );
        let radix = (next_u64(&mut state) % 35 + 2) as i64;
        let digits = &"0123456789abcdefghijklmnopqrstuvwxyz"[..radix as usize];
        assert_eq!(
            a.format(digits, "", "", 0).unwrap(),
            oa.to_str_radix(radix as u32),
            "radix format, iteration {iteration}, radix {radix}"
        );
        let radix_parsed = RBigInt::fromstr(&oa.to_str_radix(radix as u32), radix, false).unwrap();
        assert_eq!(
            to_oracle(&radix_parsed),
            oa,
            "radix parse, iteration {iteration}, radix {radix}"
        );

        let signed_le = oa.to_signed_bytes_le();
        let signed_be = oa.to_signed_bytes_be();
        assert_eq!(
            to_oracle(&RBigInt::frombytes(&signed_le, "little", true).unwrap()),
            oa,
            "signed little-endian frombytes, iteration {iteration}"
        );
        assert_eq!(
            to_oracle(&RBigInt::frombytes(&signed_be, "big", true).unwrap()),
            oa,
            "signed big-endian frombytes, iteration {iteration}"
        );
        assert_eq!(
            a.tobytes(signed_le.len() as i64, "little", true).unwrap(),
            signed_le,
            "signed little-endian tobytes, iteration {iteration}"
        );
        assert_eq!(
            a.tobytes(signed_be.len() as i64, "big", true).unwrap(),
            signed_be,
            "signed big-endian tobytes, iteration {iteration}"
        );

        assert_eq!(
            to_oracle(&a.gcd(&b).unwrap()),
            oracle_gcd(&oa, &ob),
            "gcd, iteration {iteration}"
        );
        assert_eq!(
            to_oracle(&a.abs().isqrt().unwrap()),
            oracle_abs(&oa).sqrt(),
            "isqrt, iteration {iteration}"
        );

        if ob != Oracle::from(0) {
            let (q, r) = a.divmod(&b).unwrap();
            let (oq, or) = oracle_floor_divmod(&oa, &ob);
            assert_eq!(to_oracle(&q), oq, "floordiv, iteration {iteration}");
            assert_eq!(to_oracle(&r), or, "mod, iteration {iteration}");
        }

        let exponent = (next_u64(&mut state) % 13) as i64;
        assert_eq!(
            to_oracle(&a.int_pow(exponent, None).unwrap()),
            oa.pow(exponent as u32),
            "pow, iteration {iteration}"
        );
        let modulus = oracle_abs(&ob) + Oracle::from(1);
        let rmodulus = RBigInt::fromstr(&modulus.to_string(), 10, false).unwrap();
        let rexponent = RBigInt::fromint(exponent);
        assert_eq!(
            to_oracle(&a.pow(&rexponent, Some(&rmodulus)).unwrap()),
            oa.modpow(&Oracle::from(exponent), &modulus),
            "modular pow, iteration {iteration}"
        );
    }
}
