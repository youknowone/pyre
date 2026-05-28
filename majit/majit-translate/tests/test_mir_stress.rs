//! Stress test the MIR-driven lowering driver against the full
//! extracted `pyre-interpreter.ullbc` snapshot (issue #97 Step 3).
//!
//! Skipped by default: the snapshot is 133 MB and not checked into git
//! (regenerable via `scripts/extract-llbc.sh`). Set
//! `PYRE_MIR_STRESS_LLBC=path/to/file.ullbc` to enable, or use the
//! default path the extractor writes to.

use majit_charon_reader::Llbc;
use majit_translate::front::mir::{LowerError, lower_fun_decl};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn stress_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("PYRE_MIR_STRESS_LLBC") {
        return Some(PathBuf::from(p));
    }
    let default = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build/llbc/pyre-interpreter.ullbc"
    ));
    if default.exists() {
        Some(default)
    } else {
        None
    }
}

#[test]
fn mir_lowering_tally_pyre_interpreter() {
    let Some(path) = stress_path() else {
        eprintln!(
            "skip: set PYRE_MIR_STRESS_LLBC or run scripts/extract-llbc.sh to make \
             build/llbc/pyre-interpreter.ullbc available"
        );
        return;
    };
    let llbc = Llbc::load(&path).expect("load stress llbc");

    let mut ok = 0usize;
    let mut total = 0usize;
    let mut unsupported_bucket: BTreeMap<String, usize> = BTreeMap::new();
    let mut schema_bucket: BTreeMap<String, usize> = BTreeMap::new();

    for fd in llbc.iter_local_fns() {
        if fd.unstructured().is_none() {
            // Opaque body / extraction error / structured-only —
            // outside Step 3's scope.
            continue;
        }
        total += 1;
        match lower_fun_decl(&llbc, fd) {
            Ok(_) => ok += 1,
            Err(LowerError::Unsupported(msg)) => {
                let bucket = bucket_message(&msg);
                *unsupported_bucket.entry(bucket).or_default() += 1;
            }
            Err(LowerError::Schema(msg)) => {
                let bucket = bucket_message(&msg);
                *schema_bucket.entry(bucket).or_default() += 1;
            }
            Err(LowerError::FunctionNotFound(_)) => unreachable!("only iter_local_fns"),
        }
    }

    eprintln!("\n=== MIR lowering stress tally ===");
    eprintln!("path: {}", path.display());
    eprintln!("ok    {ok} / {total}");
    eprintln!("\nUnsupported buckets (top 30):");
    let mut bucks: Vec<_> = unsupported_bucket.iter().collect();
    bucks.sort_by(|a, b| b.1.cmp(a.1));
    for (msg, n) in bucks.iter().take(30) {
        eprintln!("  {n:>5}  {msg}");
    }
    if !schema_bucket.is_empty() {
        eprintln!("\nSchema buckets:");
        let mut bs: Vec<_> = schema_bucket.iter().collect();
        bs.sort_by(|a, b| b.1.cmp(a.1));
        for (msg, n) in bs.iter().take(20) {
            eprintln!("  {n:>5}  {msg}");
        }
    }
}

/// Collapse a fail-loud message down to a stable bucket key. Replaces
/// `bb<digits>` with `bb*`, strips inline JSON tails after `: {`, and
/// caps the bucket at 120 chars so noisy payloads do not blow up the
/// histogram.
fn bucket_message(msg: &str) -> String {
    // 1. Replace `bb<digits>` with `bb*`.
    let mut s = String::with_capacity(msg.len());
    let mut chars = msg.chars().peekable();
    while let Some(c) = chars.next() {
        if c == 'b' && chars.peek() == Some(&'b') {
            // peek past the second 'b' and require at least one digit
            let mut tmp = chars.clone();
            tmp.next();
            if matches!(tmp.peek(), Some(d) if d.is_ascii_digit()) {
                chars.next();
                while let Some(&n) = chars.peek() {
                    if n.is_ascii_digit() {
                        chars.next();
                    } else {
                        break;
                    }
                }
                s.push_str("bb*");
                continue;
            }
        }
        s.push(c);
    }
    // 2. Trim inline JSON `{...}` tails to keep the bucket stable.
    if let Some(idx) = s.find(": {") {
        s.truncate(idx);
    }
    // 3. Cap length so a runaway payload doesn't dominate the histogram.
    if s.len() > 120 {
        s.truncate(120);
    }
    s
}
