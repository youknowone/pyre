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
pub mod jit_interp;
pub mod regex;

use regex::{bench_regex, bench_regex_left, count, depth, lower, nonmatching, vectors};
use std::time::Instant;

/// Characters scanned per second.
fn rate(chars: usize, secs: f64) -> f64 {
    chars as f64 / secs
}

fn commas(x: f64) -> String {
    let n = x.round() as u64;
    let d = n.to_string();
    let mut out = String::new();
    for (i, c) in d.chars().enumerate() {
        if i > 0 && (d.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

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
    let s = nonmatching(1 << 20, 20, 42);
    assert!(
        !interp::matches(balanced, &s),
        "the benchmark input is supposed NOT to match"
    );
    println!("nonmatching({} chars): no match, as intended\n", s.len());

    // Fresh trees per row: `matches` clears the marks it leaves, but two rows
    // sharing one tree would also share whatever the other left behind if that
    // ever stopped being true.
    let plain_root = lower(&bench_regex(20));
    let t0 = Instant::now();
    let plain_hit = interp::matches(plain_root, &s);
    let plain = rate(s.len(), t0.elapsed().as_secs_f64());

    let jit_root = lower(&bench_regex(20));
    // Warm up on a short prefix, so the timed run measures compiled code and
    // not the trace it had to record first.
    let _ = jit_interp::matches(jit_root, &s[..4096], 3);
    let t0 = Instant::now();
    let jit_hit = jit_interp::matches(jit_root, &s, 3);
    let jit = rate(s.len(), t0.elapsed().as_secs_f64());
    assert_eq!(plain_hit, jit_hit, "the two matchers disagreed");

    println!("(a|b)*a(a|b){{20}}a(a|b)*  over {} chars", s.len());
    println!(
        "  Rust interp over NodeRec, no JIT : {:>12} chars/s",
        commas(plain)
    );
    println!(
        "  majit JIT, regex promoted        : {:>12} chars/s",
        commas(jit)
    );
    println!(
        "  speedup                          : {:>11.1}x",
        jit / plain
    );

    // Context, measured on this machine on 2026-08-25 with the same regex and
    // the same kind of input — not from this run.
    println!();
    println!("for context, same regex and input kind, measured separately:");
    println!("  CPython 3.14.2, pure Python      :      145,454 chars/s");
    println!("  PyPy 7.3.20, general JIT, warm   :      335,325 chars/s");
    println!(
        "  CPython `re`                     :  5.4M-13.3M chars/s  \
         NOT comparable: `re` may bail out of a non-match early, while the \
         marked matcher always scans the whole string"
    );
    println!();
    println!("the 2010 post's own figures, on 2010 hardware, for context only:");
    println!("  pure Python                      :       12,200 chars/s");
    println!("  RPython translated to C          :      720,000 chars/s");
    println!("  RPython + JIT, regex green       :   16,500,000 chars/s");
    println!(
        "  its own JIT-over-no-JIT ratio    :         22.9x  \
         the quantity the speedup above is comparable to: both divide a JIT \
         that specialized on the regex by the same matcher compiled ahead of \
         time. The absolute rows are not comparable to anything here"
    );

    if bad != 0 {
        std::process::exit(1);
    }
}
