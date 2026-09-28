/// Integer division by constant using magic number multiplication.
///
/// Translated from rpython/jit/metainterp/optimizeopt/intdiv.py.
///
/// Replaces signed integer division by a constant with a sequence of
/// UINT_MUL_HIGH + shift operations, avoiding the expensive `idiv` instruction.
use majit_ir::operand::Operand;
use majit_ir::{Const, Op, OpCode, OpRc};

/// Compute magic numbers for division by constant `m`.
///
/// Returns `(k, i)` where `k` is the multiplier and `i` is the shift amount.
/// The relationship is: `k = 2^(64+i) / m + 1`.
///
/// Preconditions: `m >= 3`, `m` is not a power of two.
pub fn magic_numbers(m: i64) -> (u64, u32) {
    debug_assert!(m >= 3);
    debug_assert!(m & (m - 1) != 0, "m must not be a power of two");
    let m_u = m as u64;

    // Find i such that 2^i < m < 2^(i+1)
    let mut i: u32 = 1;
    while (1u64 << (i + 1)) < m_u {
        i += 1;
    }

    // Compute quotient = 2^(64+i) // m using bit-by-bit long division.
    // We cannot represent 2^(64+i) directly, so we use the fact that
    // UINT_MUL_HIGH(t, m) gives us the high 64 bits of t*m.
    let high_word_dividend = 1u64 << i;
    let mut quotient: u64 = 0;
    for bit in (0..64).rev() {
        let t = quotient + (1u64 << bit);
        // Check: is t * m < 2^(64+i)?
        // Equivalently: UINT_MUL_HIGH(t, m) < high_word_dividend
        let (_, high) = full_mul_u64(t, m_u);
        if high < high_word_dividend {
            quotient = t;
        }
    }

    // k = 2^(64+i) // m + 1
    let k = quotient + 1;

    debug_assert!(k != 0);
    // k > 2^63 because m < 2^(i+1) implies 2^(64+i) // m >= 2^63
    debug_assert!(k > (1u64 << 63));

    (k, i)
}

/// Full 128-bit multiplication of two u64 values.
/// Returns (low, high) halves.
#[inline]
fn full_mul_u64(a: u64, b: u64) -> (u64, u64) {
    let result = (a as u128) * (b as u128);
    (result as u64, (result >> 64) as u64)
}

fn const_int(value: i64) -> Operand {
    Operand::const_(Const::Int(value))
}

/// intdiv.py `division_operations`: the operations computing `n // m`,
/// not yet sent, the result last.
///
/// Algorithm:
/// ```text
///   t = n >> 63            (sign bits: 0 or -1)
///   nt = n ^ t             (conditional negate: n if n >= 0, ~n if n < 0)
///   mul = UINT_MUL_HIGH(nt, k)
///   sh = UINT_RSHIFT(mul, i)
///   result = sh ^ t        (negate back if needed)
/// ```
///
/// When `known_nonneg` is true, skips sign correction (saves 3 ops):
/// ```text
///   mul = UINT_MUL_HIGH(n, k)
///   result = UINT_RSHIFT(mul, i)
/// ```
pub fn division_operations(n_box: &Operand, m: i64, known_nonneg: bool) -> Vec<OpRc> {
    let (kk, ii) = magic_numbers(m);

    let sign = (!known_nonneg).then(|| {
        let t_box = OpRc::new(Op::new(
            OpCode::IntRshift,
            &[n_box.clone(), const_int(i64::from(i64::BITS - 1))],
        ));
        let nt_box = OpRc::new(Op::new(
            OpCode::IntXor,
            &[n_box.clone(), Operand::from_bound_op(&t_box)],
        ));
        (t_box, nt_box)
    });
    let nt = match &sign {
        Some((_, nt_box)) => Operand::from_bound_op(nt_box),
        None => n_box.clone(),
    };
    let mul_box = OpRc::new(Op::new(OpCode::UintMulHigh, &[nt, const_int(kk as i64)]));
    let sh_box = OpRc::new(Op::new(
        OpCode::UintRshift,
        &[Operand::from_bound_op(&mul_box), const_int(i64::from(ii))],
    ));
    match sign {
        Some((t_box, nt_box)) => {
            let final_box = OpRc::new(Op::new(
                OpCode::IntXor,
                &[
                    Operand::from_bound_op(&sh_box),
                    Operand::from_bound_op(&t_box),
                ],
            ));
            vec![t_box, nt_box, mul_box, sh_box, final_box]
        }
        None => vec![mul_box, sh_box],
    }
}

/// intdiv.py `modulo_operations`: the operations computing
/// `n - (n // m) * m`, not yet sent, the result last.
pub fn modulo_operations(n_box: &Operand, m: i64, known_nonneg: bool) -> Vec<OpRc> {
    let mut operations = division_operations(n_box, m, known_nonneg);
    let quotient = Operand::from_bound_op(operations.last().expect("a division emits operations"));
    let mul_box = OpRc::new(Op::new(OpCode::IntMul, &[quotient, const_int(m)]));
    let diff_box = OpRc::new(Op::new(
        OpCode::IntSub,
        &[n_box.clone(), Operand::from_bound_op(&mul_box)],
    ));
    operations.push(mul_box);
    operations.push(diff_box);
    operations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> Operand {
        Operand::from_bound_op(&OpRc::new(Op::new(OpCode::SameAsI, &[])))
    }

    fn opcodes(operations: &[OpRc]) -> Vec<OpCode> {
        operations.iter().map(|op| op.opcode).collect()
    }

    // ── magic_numbers tests ──

    #[test]
    fn test_magic_numbers_3() {
        let (k, i) = magic_numbers(3);
        assert_eq!(i, 1);
        // k = 2^(64+1) // 3 + 1
        // 2^65 // 3 = 12297829382473034410
        // k = 12297829382473034411
        assert_eq!(k, 0xAAAAAAAAAAAAAAABu64);
    }

    #[test]
    fn test_magic_numbers_7() {
        let (k, i) = magic_numbers(7);
        assert_eq!(i, 2);
        // k should be > 2^63
        assert!(k > (1u64 << 63));
    }

    #[test]
    fn test_magic_numbers_10() {
        let (k, i) = magic_numbers(10);
        assert_eq!(i, 3);
        assert!(k > (1u64 << 63));
    }

    #[test]
    fn test_magic_numbers_various() {
        for m in [3, 5, 6, 7, 9, 10, 11, 12, 13, 100, 1000, 127, 255] {
            let (k, i) = magic_numbers(m);
            assert!(k > (1u64 << 63), "k too small for m={m}");
            assert!(i < 64, "i too large for m={m}");
        }
    }

    /// Verify the magic number multiplication gives correct division results.
    #[test]
    fn test_magic_numbers_correctness() {
        for m in [3i64, 5, 7, 10, 13, 100, 127, 1000] {
            let (k, i) = magic_numbers(m);
            // Test with positive dividends
            for n in [0i64, 1, 2, m - 1, m, m + 1, 100, 999, 10000, i64::MAX / 2] {
                let expected = n / m; // positive n: floor == trunc
                let actual = apply_magic_div(n, k, i);
                assert_eq!(
                    actual, expected,
                    "division failed: {n} / {m} = expected {expected}, got {actual}"
                );
            }
            // Test with negative dividends.
            // The algorithm produces floor division (towards -inf),
            // matching Python's // operator.
            for n in [-1i64, -2, -(m - 1), -m, -(m + 1), -100, -999, -10000] {
                let expected = floor_div(n, m);
                let actual = apply_magic_div(n, k, i);
                assert_eq!(
                    actual, expected,
                    "division failed: {n} // {m} = expected {expected}, got {actual}"
                );
            }
        }
    }

    /// Apply the magic number division algorithm manually.
    fn apply_magic_div(n: i64, k: u64, i: u32) -> i64 {
        let t = n >> 63; // 0 or -1
        let nt = n ^ t; // if negative: ~n; if positive: n
        let (_, high) = full_mul_u64(nt as u64, k);
        let sh = high >> i;
        (sh as i64) ^ t
    }

    /// Floor division (towards negative infinity), matching Python's // operator.
    fn floor_div(a: i64, b: i64) -> i64 {
        let d = a / b;
        let r = a % b;
        if (r != 0) && ((r ^ b) < 0) { d - 1 } else { d }
    }

    // ── division_operations tests ──

    #[test]
    fn test_division_ops_emits_correct_sequence() {
        let n = input();
        let operations = division_operations(&n, 7, false);
        assert_eq!(
            opcodes(&operations),
            [
                OpCode::IntRshift,
                OpCode::IntXor,
                OpCode::UintMulHigh,
                OpCode::UintRshift,
                OpCode::IntXor,
            ]
        );
        // The sign word feeds the negate and the final correction.
        let t = Operand::from_bound_op(&operations[0]);
        assert!(operations[1].arg(1).same_box(&t));
        assert!(operations[4].arg(1).same_box(&t));
        assert!(operations[0].arg(0).same_box(&n));
    }

    #[test]
    fn test_division_ops_known_nonneg() {
        let n = input();
        let operations = division_operations(&n, 7, true);
        assert_eq!(
            opcodes(&operations),
            [OpCode::UintMulHigh, OpCode::UintRshift]
        );
        assert!(operations[0].arg(0).same_box(&n));
    }

    // ── modulo_operations tests ──

    #[test]
    fn test_modulo_ops_emits_correct_sequence() {
        let n = input();
        let operations = modulo_operations(&n, 7, false);
        assert_eq!(operations.len(), 7);
        assert_eq!(operations[5].opcode, OpCode::IntMul);
        assert_eq!(operations[6].opcode, OpCode::IntSub);
        assert!(
            operations[5]
                .arg(0)
                .same_box(&Operand::from_bound_op(&operations[4]))
        );
        assert!(operations[6].arg(0).same_box(&n));
    }

    #[test]
    fn test_modulo_ops_known_nonneg() {
        let n = input();
        let operations = modulo_operations(&n, 7, true);
        assert_eq!(
            opcodes(&operations),
            [
                OpCode::UintMulHigh,
                OpCode::UintRshift,
                OpCode::IntMul,
                OpCode::IntSub,
            ]
        );
    }

    // ── Edge cases ──

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_magic_numbers_panics_for_power_of_two() {
        magic_numbers(4);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_magic_numbers_panics_for_two() {
        magic_numbers(2);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn test_magic_numbers_panics_for_one() {
        magic_numbers(1);
    }

    #[test]
    fn test_magic_numbers_large_divisor() {
        // Large odd divisor
        let m = (1i64 << 50) + 1;
        let (k, i) = magic_numbers(m);
        assert!(k > (1u64 << 63));
        assert!(i < 64);
    }
}
