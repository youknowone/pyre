//! Stress test the MIR-driven lowering driver against the full
//! extracted `pyre-interpreter.ullbc` snapshot (issue #97 Step 3).
//!
//! Skipped by default: the snapshot is 133 MB and not checked into git
//! (regenerable via `scripts/extract-llbc.sh`). Set
//! `PYRE_MIR_STRESS_LLBC=path/to/file.ullbc` to enable, or use the
//! default path the extractor writes to.

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{BasicBlock, Rvalue, StmtKind, TermKind, Unstructured};
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

/// Classification of where a Call/Assert/Drop `on_unwind` target leads,
/// after following any `Goto` chain to its eventual terminator.
#[derive(Default)]
struct UnwindTally {
    /// Total Call/Assert/Drop terminators inspected.
    total_call_terms: usize,
    /// Per terminator-kind ("Call"/"Assert"/"Drop") count.
    by_term_kind: BTreeMap<&'static str, usize>,
    /// Eventual-terminator taxonomy after following the Goto chain.
    eventual: BTreeMap<&'static str, usize>,
    /// How many `on_unwind` targets' goto-chains execute *any*
    /// non-trivial statement before reaching UnwindResume/Abort.
    real_work: usize,
    /// How many of those non-trivial chains carry a `Drop` terminator
    /// somewhere in the chain (destructor cleanup, the most common
    /// "looks like work" case).
    real_work_drop_in_chain: usize,
    /// fn::bb examples of real-work unwind chains (capped).
    examples: Vec<String>,
    /// Goto-chain length histogram (0 = on_unwind target is itself the
    /// terminating block, 1 = one Goto hop, …).
    chain_len_hist: BTreeMap<usize, usize>,
}

/// Does this statement do non-trivial (catch-like) work? Storage
/// markers, place mentions, and inline overflow-`Assert`s are trivial
/// bookkeeping; an `Assign` whose rvalue is anything other than a plain
/// `Use` (move/copy/const) or a `Ref`/`RawPtr` is treated as real work,
/// and any unknown statement (e.g. `SetDiscriminant`, `Deinit`) is real
/// work too.
fn stmt_is_real_work(stmt: &majit_charon_reader::Statement) -> bool {
    match stmt.stmt_kind() {
        Ok(StmtKind::StorageLive(_))
        | Ok(StmtKind::StorageDead(_))
        | Ok(StmtKind::PlaceMention(_))
        | Ok(StmtKind::Assert(_)) => false,
        Ok(StmtKind::Assign(_, rv)) => !matches!(
            rv,
            Rvalue::Use(_) | Rvalue::Ref { .. } | Rvalue::RawPtr { .. }
        ),
        // Unknown statement kind (SetDiscriminant, Deinit, …) — treat
        // as real work so we never under-count.
        Ok(StmtKind::Unknown) => true,
        // Failed to project — be conservative, count as real work so it
        // shows up as an example to inspect.
        Err(_) => true,
    }
}

/// Follow the `on_unwind` goto-chain from `start_bb`, classifying the
/// eventual terminator and whether any block on the chain does real
/// work. Returns `(eventual_kind, did_real_work, drop_in_chain,
/// chain_len)`.
fn classify_unwind_chain(
    blocks: &[BasicBlock],
    start_bb: usize,
) -> (&'static str, bool, bool, usize) {
    let mut cur = start_bb;
    let mut did_real_work = false;
    let mut drop_in_chain = false;
    let mut hops = 0usize;
    // Bound the walk so a malformed/cyclic chain can't hang the test.
    for _ in 0..64 {
        let Some(block) = blocks.get(cur) else {
            return ("oob-block", did_real_work, drop_in_chain, hops);
        };
        // Any non-trivial statement in this block is catch-like work.
        if block.statements.iter().any(stmt_is_real_work) {
            did_real_work = true;
        }
        match block.term() {
            Ok(TermKind::UnwindResume) => {
                return ("UnwindResume", did_real_work, drop_in_chain, hops);
            }
            Ok(TermKind::Abort(_)) => {
                return ("Abort", did_real_work, drop_in_chain, hops);
            }
            Ok(TermKind::Return) => {
                // An on_unwind path that *returns* would be genuine
                // cleanup-then-return; flag it as work.
                return ("Return", true, drop_in_chain, hops);
            }
            Ok(TermKind::Goto { target }) => {
                cur = target as usize;
                hops += 1;
                continue;
            }
            Ok(TermKind::Drop { target, .. }) => {
                // A Drop on the unwind path is a destructor call — the
                // chain continues to its own `target` (the next cleanup
                // step). This is "work" only in the destructor sense.
                drop_in_chain = true;
                cur = target as usize;
                hops += 1;
                continue;
            }
            Ok(TermKind::Call { .. }) => {
                // A real Call on the unwind path = catch-like work
                // (e.g. a cleanup routine that itself calls a fn).
                return ("Call", true, drop_in_chain, hops);
            }
            Ok(TermKind::Assert { target, .. }) => {
                // Inline assert on the unwind path; keep following.
                cur = target as usize;
                hops += 1;
                continue;
            }
            Ok(TermKind::Switch { .. }) => {
                // A branch on the unwind path is genuine control flow
                // (cleanup that inspects state) — flag as work.
                return ("Switch", true, drop_in_chain, hops);
            }
            Ok(TermKind::Unknown) | Err(_) => {
                return ("other/unknown-term", did_real_work, drop_in_chain, hops);
            }
        }
    }
    ("chain-too-long", true, drop_in_chain, hops)
}

#[test]
#[ignore = "requires the 205MB pyre-interpreter.ullbc snapshot; \
            set PYRE_MIR_STRESS_LLBC"]
fn mir_on_unwind_target_taxonomy() {
    let Some(path) = stress_path() else {
        eprintln!(
            "skip: set PYRE_MIR_STRESS_LLBC or run scripts/extract-llbc.sh to make \
             build/llbc/pyre-interpreter.ullbc available"
        );
        return;
    };
    let llbc = Llbc::load(&path).expect("load stress llbc");

    let mut tally = UnwindTally::default();

    for fd in llbc.iter_local_fns() {
        let Some(body): Option<Unstructured> = fd.unstructured() else {
            continue;
        };
        let blocks = &body.body;
        for (bb_idx, block) in blocks.iter().enumerate() {
            let (term_kind_label, on_unwind): (&'static str, u64) = match block.term() {
                Ok(TermKind::Call { on_unwind, .. }) => ("Call", on_unwind),
                Ok(TermKind::Assert { on_unwind, .. }) => ("Assert", on_unwind),
                Ok(TermKind::Drop { on_unwind, .. }) => ("Drop", on_unwind),
                _ => continue,
            };
            tally.total_call_terms += 1;
            *tally.by_term_kind.entry(term_kind_label).or_default() += 1;

            let (eventual, did_work, drop_in_chain, chain_len) =
                classify_unwind_chain(blocks, on_unwind as usize);
            *tally.eventual.entry(eventual).or_default() += 1;
            *tally.chain_len_hist.entry(chain_len).or_default() += 1;
            if did_work {
                tally.real_work += 1;
                if drop_in_chain {
                    tally.real_work_drop_in_chain += 1;
                }
                if tally.examples.len() < 40 {
                    tally.examples.push(format!(
                        "{}::bb{bb_idx} [{term_kind_label}] on_unwind=bb{on_unwind} \
                         -> {eventual} (chain_len={chain_len}, drop_in_chain={drop_in_chain})",
                        fd.item_meta.name_path()
                    ));
                }
            }
        }
    }

    eprintln!("\n=== on_unwind target taxonomy (whole interpreter) ===");
    eprintln!("path: {}", path.display());
    eprintln!("total Call/Assert/Drop terminators inspected: {}", tally.total_call_terms);
    eprintln!("\nby terminator kind:");
    for (k, n) in &tally.by_term_kind {
        eprintln!("  {n:>8}  {k}");
    }
    eprintln!("\neventual terminator of on_unwind goto-chain:");
    let mut ev: Vec<_> = tally.eventual.iter().collect();
    ev.sort_by(|a, b| b.1.cmp(a.1));
    for (k, n) in ev {
        eprintln!("  {n:>8}  {k}");
    }
    eprintln!("\ngoto-chain length histogram (hops to terminator):");
    for (len, n) in &tally.chain_len_hist {
        eprintln!("  len={len:>2}  {n:>8}");
    }
    eprintln!("\nany_handler_does_real_work: {}", tally.real_work > 0);
    eprintln!("  chains doing real (non-trivial) work: {}", tally.real_work);
    eprintln!(
        "    of which carry a Drop (destructor) in the chain: {}",
        tally.real_work_drop_in_chain
    );
    eprintln!(
        "    non-Drop real-work chains (genuine catch suspects): {}",
        tally.real_work - tally.real_work_drop_in_chain
    );
    if !tally.examples.is_empty() {
        eprintln!("\nreal-work examples (capped at 40):");
        for ex in &tally.examples {
            eprintln!("  {ex}");
        }
    }

    // This test is observational: it never fails, it only prints the
    // taxonomy. The decisive number to read is
    // `non-Drop real-work chains` — if that is 0, every on_unwind path
    // is a bare panic-propagation (UnwindResume/Abort) or pure
    // destructor cleanup, and dropping it loses no try/except.
    assert!(
        tally.total_call_terms > 0,
        "expected at least one Call/Assert/Drop terminator in the snapshot"
    );
    // Reviewer #63: the decisive invariant — no on_unwind path does
    // catch-like work (a Call/Switch/Return or a non-trivial statement)
    // other than pure destructor drop-glue. If this ever trips, the
    // corpus grew a Rust catch/cleanup that the front-graph driver would
    // silently drop, and the "drop on_unwind" adaptation must be revisited.
    let non_drop_real_work = tally.real_work - tally.real_work_drop_in_chain;
    assert_eq!(
        non_drop_real_work, 0,
        "found {non_drop_real_work} on_unwind chain(s) doing non-destructor \
         catch-like work; dropping on_unwind would lose semantics — see \
         examples above"
    );
}

#[test]
#[ignore = "requires the 205MB pyre-interpreter.ullbc snapshot; \
            set PYRE_MIR_STRESS_LLBC"]
fn coverage_gate_accepts_the_real_snapshot() {
    // The fail-loud coverage gate in `build_semantic_program_from_llbc`
    // must return `Ok` over the real snapshot: every current lowering
    // skip is the tracked uninitialised-local gap, so none is classified
    // a regression. If a future change makes a body fail to lower with an
    // unrecognised error, this builder returns `Err` and the build
    // (and this test) fails loudly instead of silently dropping the fn.
    let Some(path) = stress_path() else {
        eprintln!("skip: set PYRE_MIR_STRESS_LLBC");
        return;
    };
    let llbc = Llbc::load(&path).expect("load stress llbc");
    majit_translate::front::mir::build_semantic_program_from_llbc(&llbc).expect(
        "coverage gate must accept the real snapshot — all skips are the \
         tracked uninitialised-local gap",
    );
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
