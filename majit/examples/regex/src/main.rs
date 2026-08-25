//! Marked-regex matching, from "An Efficient and Elegant Regular Expression
//! Matcher in Python" and "A JIT for Regular Expression Matching".
//!
//! A regex is a tree; each node holds one mutable `marked` bit, and one
//! `shift(c, mark)` per input character propagates the marks left to right.
//! Two things a reader should not have to guess: **the marks live in the
//! nodes**, not in the matcher, and `empty` — does this subexpression accept
//! the empty string? — is **computed once, at lowering time**, so the matcher
//! only reads it.
//!
//! `regex` builds and lowers the tree; `interp` is the plain matcher over the
//! lowered graph.
pub mod interp;
pub mod regex;

use regex::{bench_regex, bench_regex_left, count, depth, lower, nonmatching, vectors};

fn main() {
    let mut bad = 0;
    let total = vectors().len();
    for (re, s, want) in vectors() {
        let got = interp::matches(lower(&re), s.as_bytes());
        if got != want {
            bad += 1;
            println!("  FAIL {s:?}: got {got}, want {want}");
        }
    }
    println!("vectors: {}/{total} pass", total - bad);

    let balanced = lower(&bench_regex(20));
    let left_assoc = lower(&bench_regex_left(20));
    println!(
        "(a|b)*a(a|b){{20}}a(a|b)*  balanced: {} nodes, depth {}",
        count(balanced),
        depth(balanced)
    );
    println!(
        "(a|b)*a(a|b){{20}}a(a|b)*  left    : {} nodes, depth {}",
        count(left_assoc),
        depth(left_assoc)
    );

    // The benchmark input: no early exit, so every character is looked at.
    let s = nonmatching(1 << 16, 20, 42);
    assert!(
        !interp::matches(balanced, &s),
        "the benchmark input is supposed NOT to match"
    );
    println!("nonmatching({} chars): no match, as intended", s.len());

    if bad != 0 {
        std::process::exit(1);
    }
}
