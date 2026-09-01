//! The matcher, without the JIT.
//!
//! `shift` is part 1's `Regex.shift` — "An Efficient and Elegant Regular
//! Expression Matcher in Python" — over integers instead of booleans, and with
//! `&`/`|` where the Python has `and`/`or`, the substitution "A JIT for Regular
//! Expression Matching" makes so the body carries no short-circuit branch.
//!
//! One `shift` per input character propagates the marks left to right, and each
//! node stores its own new mark before returning it.  That is the whole
//! algorithm: `O(m * n)`, no backtracking, no per-character allocation.

use crate::regex::{
    KIND_ALTERNATIVE, KIND_CHAR, KIND_EPSILON, KIND_REPETITION, KIND_SEQUENCE, NodeRec,
};

#[inline(always)]
fn kind(n: *mut NodeRec) -> u8 {
    unsafe { (*n).kind }
}
#[inline(always)]
fn chr(n: *mut NodeRec) -> i64 {
    unsafe { (*n).ch as i64 }
}
#[inline(always)]
fn empty(n: *mut NodeRec) -> i64 {
    unsafe { i64::from((*n).empty) }
}
#[inline(always)]
fn marked(n: *mut NodeRec) -> i64 {
    unsafe { i64::from((*n).marked) }
}
#[inline(always)]
fn set_marked(n: *mut NodeRec, m: i64) {
    unsafe { (*n).marked = m != 0 }
}
#[inline(always)]
fn left(n: *mut NodeRec) -> *mut NodeRec {
    unsafe { (*n).left }
}
#[inline(always)]
fn right(n: *mut NodeRec) -> *mut NodeRec {
    unsafe { (*n).right }
}

/// Push `mark` into `n` for input character `c`, and answer the mark that comes
/// out on the right.  The mark left inside `n` is the one this call computed.
pub fn shift(n: *mut NodeRec, c: i64, mark: i64) -> i64 {
    let m = match kind(n) {
        KIND_CHAR => mark & ((chr(n) == c) as i64),
        KIND_EPSILON => 0,
        KIND_ALTERNATIVE => shift(left(n), c, mark) | shift(right(n), c, mark),
        KIND_REPETITION => shift(left(n), c, mark | marked(n)),
        KIND_SEQUENCE => {
            // The left mark from the *previous* character is what enters the
            // right side, so read it before `shift` overwrites it.
            let old_marked_left = marked(left(n));
            let marked_left = shift(left(n), c, mark);
            let marked_right = shift(right(n), c, old_marked_left | (mark & empty(left(n))));
            (marked_left & empty(right(n))) | marked_right
        }
        _ => 0,
    };
    set_marked(n, m);
    m
}

/// Clear every mark, so the graph can match another string.
pub fn reset(n: *mut NodeRec) {
    set_marked(n, 0);
    if !left(n).is_null() {
        reset(left(n));
    }
    if !right(n).is_null() {
        reset(right(n));
    }
}

/// Part 2's `match`: shift one mark in from the left for `s[0]`, then shift the
/// marks already inside the graph along for every remaining character.  An
/// empty string matches exactly the regexes that accept it.
pub fn matches(root: *mut NodeRec, s: &[u8]) -> bool {
    if s.is_empty() {
        return empty(root) != 0;
    }
    let mut result = shift(root, s[0] as i64, 1);
    let mut i = 1usize;
    while i < s.len() {
        result = shift(root, s[i] as i64, 0);
        i += 1;
    }
    reset(root);
    result != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regex::{
        Node, alt, bench_regex, bench_regex_left, ch, lower, nonmatching, rep, seq, vectors,
    };

    fn run(re: &Node, s: &str) -> bool {
        matches(lower(re), s.as_bytes())
    }

    #[test]
    fn test_vectors() {
        for (re, s, want) in vectors() {
            assert_eq!(run(&re, s), want, "input {s:?}");
        }
    }

    /// `((abc)*|(abcd))(d|e)` on `"abcde"`: the right alternative takes `abcd`
    /// and `e` closes it.  The left one cannot, which is what makes `"abcdf"`
    /// the discriminating negative.
    #[test]
    fn test_alternation_across_a_shared_prefix() {
        let re = || {
            seq(
                alt(
                    rep(seq(seq(ch(b'a'), ch(b'b')), ch(b'c'))),
                    seq(seq(seq(ch(b'a'), ch(b'b')), ch(b'c')), ch(b'd')),
                ),
                alt(ch(b'd'), ch(b'e')),
            )
        };
        assert!(run(&re(), "abcde"));
        assert!(!run(&re(), "abcdf"));
    }

    /// The graph is reusable: `matches` clears the marks it left behind.
    #[test]
    fn test_reset_between_matches() {
        let root = lower(&bench_regex(2));
        for _ in 0..3 {
            assert!(matches(root, b"abba"));
            assert!(!matches(root, b"aba"));
        }
    }

    /// A plain random `a`/`b` string almost surely contains two `a`s exactly
    /// `n + 1` apart and therefore matches; `nonmatching` is what removes them.
    #[test]
    fn test_nonmatching_really_does_not_match() {
        let root = lower(&bench_regex(20));
        for seed in [1u64, 42, 9999] {
            assert!(!matches(root, &nonmatching(4096, 20, seed)));
        }
    }

    #[test]
    fn test_random_input_would_match_without_the_fixup() {
        let root = lower(&bench_regex(20));
        let mut s = nonmatching(4096, 20, 42);
        // Reinstate one cleared pair: 'a' at i and i + 21 is the whole regex.
        s[100] = b'a';
        s[121] = b'a';
        assert!(matches(root, &s));
    }

    /// Association does not change the language, only the depth: 1560 random
    /// `a`/`b` strings per `n`, at the depth-8 `n = 20` and at an `n` small
    /// enough that a length-25 string reaches the second `a`.
    #[test]
    fn test_balanced_and_left_associated_agree() {
        for n in [2usize, 20] {
            let balanced = lower(&bench_regex(n));
            let left_assoc = lower(&bench_regex_left(n));
            let mut seed: u64 = 12345;
            for len in 0..26usize {
                for _ in 0..60 {
                    let mut s = Vec::with_capacity(len);
                    for _ in 0..len {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        s.push(if (seed >> 33) & 1 == 0 { b'a' } else { b'b' });
                    }
                    assert_eq!(
                        matches(balanced, &s),
                        matches(left_assoc, &s),
                        "input {s:?}"
                    );
                }
            }
        }
    }
}
