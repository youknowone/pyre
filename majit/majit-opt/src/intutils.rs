/// Integer bound tracking for the optimizer.
///
/// Translated from rpython/jit/metainterp/optimizeopt/intutils.py.
/// Tracks both classical interval bounds (lower, upper) and known-bits
/// information (tvalue, tmask) using a tristate number representation.
///
/// A tristate number represents partial knowledge about an integer:
/// - tvalue: the known bits (where tmask is 0)
/// - tmask: 1 where the bit is unknown, 0 where it's known
///
/// This is a placeholder with the core structure; the full 1712-line
/// translation will be completed in Stream B1.

/// An abstract bound on an integer value.
///
/// Combines interval analysis [lower, upper] with known-bits analysis (tvalue, tmask).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntBound {
    /// Lower bound (inclusive). i64::MIN means unbounded below.
    pub lower: i64,
    /// Upper bound (inclusive). i64::MAX means unbounded above.
    pub upper: i64,
    /// Known bits: where tmask bit is 0, tvalue bit is the known value.
    pub tvalue: u64,
    /// Unknown bit mask: 1 means the bit is unknown.
    pub tmask: u64,
}

impl IntBound {
    /// Completely unbounded.
    pub fn unbounded() -> Self {
        IntBound {
            lower: i64::MIN,
            upper: i64::MAX,
            tvalue: 0,
            tmask: u64::MAX,
        }
    }

    /// Exact constant.
    pub fn from_constant(value: i64) -> Self {
        IntBound {
            lower: value,
            upper: value,
            tvalue: value as u64,
            tmask: 0,
        }
    }

    /// Bounded in [lower, upper].
    pub fn bounded(lower: i64, upper: i64) -> Self {
        debug_assert!(lower <= upper);
        let mut bound = IntBound {
            lower,
            upper,
            tvalue: 0,
            tmask: u64::MAX,
        };
        bound.update_knownbits_from_bounds();
        bound
    }

    /// Known to be non-negative.
    pub fn nonnegative() -> Self {
        Self::bounded(0, i64::MAX)
    }

    /// From known bits.
    pub fn from_knownbits(tvalue: u64, tmask: u64) -> Self {
        debug_assert!(tvalue & tmask == 0, "tvalue must be 0 where tmask is 1");
        let mut bound = IntBound {
            lower: i64::MIN,
            upper: i64::MAX,
            tvalue,
            tmask,
        };
        bound.update_bounds_from_knownbits();
        bound
    }

    /// Whether the value is exactly known.
    pub fn is_constant(&self) -> bool {
        self.tmask == 0
    }

    /// Get the known constant value (panics if not constant).
    pub fn get_constant(&self) -> i64 {
        debug_assert!(self.is_constant());
        self.lower
    }

    /// Whether the value is known to be non-negative.
    pub fn known_nonnegative(&self) -> bool {
        self.lower >= 0
    }

    /// Whether the value is known to be less than `other`.
    pub fn known_lt(&self, other: &IntBound) -> bool {
        self.upper < other.lower
    }

    /// Whether the value is known to be less than or equal to `other`.
    pub fn known_le(&self, other: &IntBound) -> bool {
        self.upper <= other.lower
    }

    /// Whether the value is known to be greater than `other`.
    pub fn known_gt(&self, other: &IntBound) -> bool {
        self.lower > other.upper
    }

    /// Whether the value is known to be greater than or equal to `other`.
    pub fn known_ge(&self, other: &IntBound) -> bool {
        self.lower >= other.upper
    }

    /// Intersect this bound with another (tighten).
    pub fn intersect(&mut self, other: &IntBound) -> bool {
        let mut changed = false;
        if other.lower > self.lower {
            self.lower = other.lower;
            changed = true;
        }
        if other.upper < self.upper {
            self.upper = other.upper;
            changed = true;
        }
        // Intersect known bits: if a bit is known in either, it must match.
        let both_known = !self.tmask & !other.tmask;
        // Check for contradiction
        if (self.tvalue ^ other.tvalue) & both_known != 0 {
            // Contradiction — this bound becomes empty.
            // In practice this means the guard is always false.
            self.lower = 1;
            self.upper = 0;
            return true;
        }
        let new_tmask = self.tmask & other.tmask;
        let new_tvalue = (self.tvalue & !self.tmask) | (other.tvalue & !other.tmask) & !new_tmask;
        if new_tmask != self.tmask {
            self.tmask = new_tmask;
            self.tvalue = new_tvalue & !new_tmask;
            changed = true;
        }
        if changed {
            self.update_bounds_from_knownbits();
        }
        changed
    }

    // ── Transfer functions (add, sub, mul, and, or, xor, shift) ──
    // Stub implementations — full versions from intutils.py will be completed in B1.

    /// Bound after addition.
    pub fn add_bound(&self, other: &IntBound) -> IntBound {
        let lower = self.lower.checked_add(other.lower).unwrap_or(i64::MIN);
        let upper = self.upper.checked_add(other.upper).unwrap_or(i64::MAX);
        let (tvalue, tmask) = self.tnum_add(other);
        let mut result = IntBound { lower, upper, tvalue, tmask };
        result.update_bounds_from_knownbits();
        result.update_knownbits_from_bounds();
        result
    }

    /// Bound after subtraction.
    pub fn sub_bound(&self, other: &IntBound) -> IntBound {
        let lower = self.lower.checked_sub(other.upper).unwrap_or(i64::MIN);
        let upper = self.upper.checked_sub(other.lower).unwrap_or(i64::MAX);
        IntBound::bounded(lower, upper)
    }

    /// Bound after multiplication.
    pub fn mul_bound(&self, other: &IntBound) -> IntBound {
        let candidates = [
            self.lower.checked_mul(other.lower),
            self.lower.checked_mul(other.upper),
            self.upper.checked_mul(other.lower),
            self.upper.checked_mul(other.upper),
        ];
        let mut lower = i64::MAX;
        let mut upper = i64::MIN;
        let mut overflow = false;
        for c in &candidates {
            match c {
                Some(v) => {
                    lower = lower.min(*v);
                    upper = upper.max(*v);
                }
                None => overflow = true,
            }
        }
        if overflow {
            IntBound::unbounded()
        } else {
            IntBound::bounded(lower, upper)
        }
    }

    /// Bound after bitwise AND.
    pub fn and_bound(&self, other: &IntBound) -> IntBound {
        let tvalue = self.tvalue & other.tvalue;
        let tmask = (self.tmask | self.tvalue) & (other.tmask | other.tvalue) & !tvalue;
        let mut result = IntBound::from_knownbits(tvalue & !tmask, tmask);
        // If both non-negative, the result is bounded by the minimum upper.
        if self.known_nonnegative() && other.known_nonnegative() {
            let min_upper = self.upper.min(other.upper);
            if result.upper > min_upper {
                result.upper = min_upper;
            }
        }
        result
    }

    /// Bound after bitwise OR.
    pub fn or_bound(&self, other: &IntBound) -> IntBound {
        let tvalue = self.tvalue | other.tvalue;
        let tmask = self.tmask | other.tmask;
        let tmask = tmask & !tvalue;
        IntBound::from_knownbits(tvalue & !tmask, tmask)
    }

    /// Bound after bitwise XOR.
    pub fn xor_bound(&self, other: &IntBound) -> IntBound {
        let tvalue = self.tvalue ^ other.tvalue;
        let tmask = self.tmask | other.tmask;
        IntBound::from_knownbits(tvalue & !tmask, tmask)
    }

    /// Bound after left shift by a constant.
    pub fn lshift_bound(&self, shift: u32) -> IntBound {
        if shift >= 64 {
            return IntBound::from_constant(0);
        }
        let tvalue = self.tvalue << shift;
        let tmask = self.tmask << shift;
        let lower = self.lower.checked_shl(shift).unwrap_or(i64::MIN);
        let upper = self.upper.checked_shl(shift).unwrap_or(i64::MAX);
        IntBound {
            lower,
            upper,
            tvalue: tvalue & !tmask,
            tmask,
        }
    }

    /// Bound after arithmetic right shift by a constant.
    pub fn rshift_bound(&self, shift: u32) -> IntBound {
        if shift >= 64 {
            if self.lower >= 0 { return IntBound::from_constant(0); }
            if self.upper < 0 { return IntBound::from_constant(-1); }
            return IntBound::bounded(-1, 0);
        }
        let lower = self.lower >> shift;
        let upper = self.upper >> shift;
        IntBound::bounded(lower, upper)
    }

    // ── Internal helpers ──

    /// Tristate number addition.
    fn tnum_add(&self, other: &IntBound) -> (u64, u64) {
        let sum = self.tvalue.wrapping_add(other.tvalue);
        let carry_in = sum ^ self.tvalue ^ other.tvalue;
        let carry_mask = carry_in | self.tmask | other.tmask;
        // Propagate carry uncertainty
        let carry_out = carry_mask.wrapping_add(carry_mask) | carry_in;
        let tmask = carry_out ^ carry_in | self.tmask | other.tmask;
        let tvalue = sum & !tmask;
        (tvalue, tmask)
    }

    /// Update interval bounds from known bits.
    fn update_bounds_from_knownbits(&mut self) {
        if self.tmask == u64::MAX {
            return; // No known bits
        }
        // Lower bound: set unknown bits to 0 (smallest possible)
        let min_val = self.tvalue as i64;
        // Upper bound: set unknown bits to 1 (largest possible)
        let max_val = (self.tvalue | self.tmask) as i64;

        // Only tighten, never loosen
        if min_val > self.lower {
            self.lower = min_val;
        }
        if max_val < self.upper {
            self.upper = max_val;
        }
    }

    /// Update known bits from interval bounds.
    fn update_knownbits_from_bounds(&mut self) {
        if self.lower == self.upper {
            // Exact constant
            self.tvalue = self.lower as u64;
            self.tmask = 0;
            return;
        }
        // If both bounds share the same high bits, those bits are known.
        let diff = (self.lower as u64) ^ (self.upper as u64);
        if diff == 0 {
            return;
        }
        // Find the highest differing bit
        let highest_diff_bit = 63 - diff.leading_zeros();
        // All bits above the highest differing bit are the same
        let known_mask = !((1u64 << (highest_diff_bit + 1)) - 1);
        let new_tvalue = (self.lower as u64) & known_mask;
        let new_tmask = !known_mask | self.tmask;
        self.tvalue = (self.tvalue & !known_mask) | (new_tvalue & known_mask);
        self.tmask &= new_tmask;
    }
}

impl Default for IntBound {
    fn default() -> Self {
        Self::unbounded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant() {
        let b = IntBound::from_constant(42);
        assert!(b.is_constant());
        assert_eq!(b.get_constant(), 42);
        assert_eq!(b.lower, 42);
        assert_eq!(b.upper, 42);
        assert_eq!(b.tvalue, 42);
        assert_eq!(b.tmask, 0);
    }

    #[test]
    fn test_unbounded() {
        let b = IntBound::unbounded();
        assert!(!b.is_constant());
        assert_eq!(b.lower, i64::MIN);
        assert_eq!(b.upper, i64::MAX);
    }

    #[test]
    fn test_bounded() {
        let b = IntBound::bounded(0, 100);
        assert!(b.known_nonnegative());
        assert!(!b.is_constant());
    }

    #[test]
    fn test_add_constants() {
        let a = IntBound::from_constant(10);
        let b = IntBound::from_constant(20);
        let result = a.add_bound(&b);
        assert!(result.is_constant());
        assert_eq!(result.get_constant(), 30);
    }

    #[test]
    fn test_add_bounded() {
        let a = IntBound::bounded(0, 10);
        let b = IntBound::bounded(0, 20);
        let result = a.add_bound(&b);
        assert_eq!(result.lower, 0);
        assert_eq!(result.upper, 30);
    }

    #[test]
    fn test_sub_bounded() {
        let a = IntBound::bounded(10, 20);
        let b = IntBound::bounded(0, 5);
        let result = a.sub_bound(&b);
        assert_eq!(result.lower, 5);
        assert_eq!(result.upper, 20);
    }

    #[test]
    fn test_and_knownbits() {
        let a = IntBound::from_constant(0xFF);
        let b = IntBound::unbounded();
        let result = a.and_bound(&b);
        // Result must be in [0, 0xFF] since a is all-ones in low byte
        assert!(result.upper <= 0xFF || !result.known_nonnegative());
    }

    #[test]
    fn test_known_comparisons() {
        let a = IntBound::bounded(0, 5);
        let b = IntBound::bounded(10, 20);
        assert!(a.known_lt(&b));
        assert!(a.known_le(&b));
        assert!(!a.known_gt(&b));
        assert!(b.known_gt(&a));
    }

    #[test]
    fn test_intersect() {
        let mut a = IntBound::bounded(0, 100);
        let b = IntBound::bounded(50, 200);
        let changed = a.intersect(&b);
        assert!(changed);
        assert_eq!(a.lower, 50);
        assert_eq!(a.upper, 100);
    }
}
