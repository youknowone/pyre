//! Stress test the MIR-driven lowering driver against the full
//! extracted `pyre-interpreter.ullbc` snapshot.
//!
//! Skipped by default: the snapshot is 133 MB and not checked into git
//! (regenerable via `scripts/extract-llbc.py`). Set
//! `PYRE_MIR_STRESS_LLBC=path/to/file.ullbc` to enable, or use the
//! default path the extractor writes to.

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    BasicBlock, PlaceKind, Rvalue, StmtKind, SwitchTargets, TermKind, Unstructured,
};
// `CallPayload`'s `dest` is the binding site for a Call-terminator local,
// which the lowering driver writes into `local_var[dest_local]` — so the
// classifier must treat Call-`dest` blocks as Assign-equivalent.
use majit_translate::front::mir::{LowerError, lower_fun_decl};
use std::collections::{BTreeMap, BTreeSet};
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

/// Successors of a basic block, used by the RPO / back-edge classifier
/// `classify_uninitialised_local_rpo_vs_loop_carried`.  Mirrors the
/// `lower_terminator` edges: the normal target *and* `on_unwind` for
/// `Call`/`Assert`/`Drop`; both arms of an `If`; every `SwitchInt` arm
/// plus its default.
fn block_succs(blk: &BasicBlock) -> Vec<usize> {
    let ts: Vec<u64> = match blk.term() {
        Ok(TermKind::Goto { target }) => vec![target],
        Ok(TermKind::Call {
            target, on_unwind, ..
        })
        | Ok(TermKind::Assert {
            target, on_unwind, ..
        })
        | Ok(TermKind::Drop {
            target, on_unwind, ..
        }) => vec![target, on_unwind],
        Ok(TermKind::Switch { targets, .. }) => match targets {
            SwitchTargets::If(a, b) => vec![a, b],
            SwitchTargets::SwitchInt(_, arms, default) => {
                let mut v: Vec<u64> = arms.iter().map(|(_, bb)| *bb).collect();
                v.push(default);
                v
            }
        },
        _ => vec![],
    };
    ts.into_iter().map(|t| t as usize).collect()
}

/// Peel a `PlaceKind` down to the outermost MIR `Local(N)` it ultimately
/// addresses. `Local(N)` -> `Some(N)`; `Projection(inner, _)` recurses
/// into `inner` (so `*local`, `local.field`, `local[i]` all report `N`);
/// `Global`/`Unknown` -> `None`.
fn place_base_local(kind: &PlaceKind) -> Option<u64> {
    match kind {
        PlaceKind::Local(i) => Some(*i),
        PlaceKind::Projection(inner, _) => place_base_local(&inner.kind),
        _ => None,
    }
}

/// Compute reverse-postorder of the CFG rooted at block 0, *and* the set
/// of back-edges (child appears on the DFS recursion stack when the edge
/// is traversed). Returns `(rpo_index, back_edges)` where
/// `rpo_index[b]` is `b`'s position in reverse-postorder (lower = earlier)
/// or `usize::MAX` if `b` is unreachable from block 0.
fn rpo_and_back_edges(blocks: &[BasicBlock]) -> (Vec<usize>, BTreeSet<(usize, usize)>) {
    let n = blocks.len();
    // Iterative DFS that records postorder and back-edges. `state`:
    // 0 = white (unvisited), 1 = grey (on stack), 2 = black (done).
    let mut state = vec![0u8; n];
    let mut postorder: Vec<usize> = Vec::with_capacity(n);
    let mut back_edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    // Stack of (node, next-successor-index-to-process).
    let mut stack: Vec<(usize, usize)> = Vec::new();
    if n == 0 {
        return (vec![], back_edges);
    }
    state[0] = 1;
    stack.push((0, 0));
    while let Some(&(node, idx)) = stack.last() {
        let succs = block_succs(&blocks[node]);
        if idx < succs.len() {
            stack.last_mut().unwrap().1 += 1;
            let s = succs[idx];
            if s >= n {
                continue; // out-of-range successor, ignore
            }
            match state[s] {
                0 => {
                    state[s] = 1;
                    stack.push((s, 0));
                }
                1 => {
                    // Edge to a node currently on the recursion stack:
                    // a genuine back-edge.
                    back_edges.insert((node, s));
                }
                _ => {}
            }
        } else {
            state[node] = 2;
            postorder.push(node);
            stack.pop();
        }
    }
    // reverse-postorder index: reverse of postorder.
    let mut rpo_index = vec![usize::MAX; n];
    for (rank, &b) in postorder.iter().rev().enumerate() {
        rpo_index[b] = rank;
    }
    (rpo_index, back_edges)
}

/// Can `from` reach `to` using only forward edges (i.e. never traversing
/// a member of `back_edges`)? Plain BFS over the back-edge-pruned CFG.
fn reaches_without_backedge(
    blocks: &[BasicBlock],
    from: usize,
    to: usize,
    back_edges: &BTreeSet<(usize, usize)>,
) -> bool {
    if from == to {
        return true;
    }
    let n = blocks.len();
    let mut seen = vec![false; n];
    let mut queue = std::collections::VecDeque::new();
    seen[from] = true;
    queue.push_back(from);
    while let Some(b) = queue.pop_front() {
        for s in block_succs(&blocks[b]) {
            if s >= n || back_edges.contains(&(b, s)) {
                continue;
            }
            if s == to {
                return true;
            }
            if !seen[s] {
                seen[s] = true;
                queue.push_back(s);
            }
        }
    }
    false
}

/// Parse `read_bb` and `failing_local` out of the fail-loud message
/// `bb{R}: read of MIR local {N} before any Assign — …`.
fn parse_read_bb_and_local(msg: &str) -> Option<(usize, u64)> {
    // "bb{R}: read of MIR local {N} before any Assign"
    let rest = msg.strip_prefix("bb")?;
    let (r_str, after) = rest.split_once(':')?;
    let r: usize = r_str.trim().parse().ok()?;
    let marker = "read of MIR local ";
    let idx = after.find(marker)? + marker.len();
    let tail = &after[idx..];
    let end = tail.find(|c: char| !c.is_ascii_digit())?;
    let n: u64 = tail[..end].parse().ok()?;
    Some((r, n))
}

/// Diagnostic: for each uninitialised-local
/// lowering failure, classify whether processing blocks in
/// reverse-postorder would bind the failing local before its read
/// (forward-ref, fixable by traversal order alone, no phi) or whether the
/// only `Assign` to the local reaches the reading block exclusively via a
/// back-edge (genuine loop-carried value, needs a phi / block-inputarg).
#[test]
#[ignore = "diagnostic; set PYRE_MIR_STRESS_LLBC"]
fn classify_uninitialised_local_rpo_vs_loop_carried() {
    let Some(path) = stress_path() else {
        eprintln!("skip: set PYRE_MIR_STRESS_LLBC");
        return;
    };
    let llbc = Llbc::load(&path).expect("load stress llbc");

    let mut forward_ref = 0usize;
    let mut loop_carried = 0usize;
    let mut other = 0usize;

    for fd in llbc.iter_local_fns() {
        if fd.is_global_initializer.is_some() {
            continue;
        }
        let Some(body): Option<Unstructured> = fd.unstructured() else {
            continue;
        };
        let Err(LowerError::Unsupported(msg)) = lower_fun_decl(&llbc, fd) else {
            continue;
        };
        if !msg.contains("uninitialised local") {
            continue;
        }
        let name = fd.item_meta.name_path();
        let Some((read_bb, local_n)) = parse_read_bb_and_local(&msg) else {
            eprintln!("PARSE-FAIL {name} | {msg}");
            other += 1;
            continue;
        };

        let blocks = &body.body;

        // (1) Every block that BINDS local N, i.e. seeds `local_var[N]`.
        // The driver seeds the slot in two places:
        //   * `lower_assign` on a *direct* `PlaceKind::Local(N)` Assign
        //     statement (mir.rs:700), and
        //   * the Call-terminator destination `CallPayload.dest` when it
        //     is a direct `Local(N)` (mir.rs:1624).
        // A *projection* write (`(*N).field = …`) presupposes N already
        // bound and never seeds the slot — we record those separately for
        // the detail string only.
        let mut direct_assign_blocks: Vec<usize> = Vec::new();
        let mut proj_assign_blocks: Vec<usize> = Vec::new();
        for (bb_idx, blk) in blocks.iter().enumerate() {
            let mut seeds = false;
            let mut proj = false;
            for st in &blk.statements {
                if let Ok(StmtKind::Assign(place, _)) = st.stmt_kind() {
                    match &place.kind {
                        PlaceKind::Local(i) if *i == local_n => seeds = true,
                        PlaceKind::Projection(..)
                            if place_base_local(&place.kind) == Some(local_n) =>
                        {
                            proj = true
                        }
                        _ => {}
                    }
                }
            }
            // Call-terminator destination — the dominant binding site for
            // these failures (the local is the result of a fn call).
            if let Ok(TermKind::Call { call, .. }) = blk.term() {
                match &call.dest.kind {
                    PlaceKind::Local(i) if *i == local_n => seeds = true,
                    PlaceKind::Projection(..)
                        if place_base_local(&call.dest.kind) == Some(local_n) =>
                    {
                        proj = true
                    }
                    _ => {}
                }
            }
            if seeds {
                direct_assign_blocks.push(bb_idx);
            }
            if proj {
                proj_assign_blocks.push(bb_idx);
            }
        }
        direct_assign_blocks.dedup();
        proj_assign_blocks.dedup();

        // (2)+(3) CFG: reverse-postorder + back-edge set from block 0.
        let (rpo_index, back_edges) = rpo_and_back_edges(blocks);

        // Classify against the *binding* (direct) assign blocks.
        let read_rpo = rpo_index.get(read_bb).copied().unwrap_or(usize::MAX);

        // Forward-ref signal A: at least one direct-assign block precedes
        // the read block in reverse-postorder.
        let mut rpo_precedes = false;
        // Forward-ref signal B (stronger): at least one direct-assign
        // block reaches the read block over forward edges only (never
        // crossing a back-edge), i.e. on a non-loop path.
        let mut forward_reaches = false;
        for &ab in &direct_assign_blocks {
            let ab_rpo = rpo_index.get(ab).copied().unwrap_or(usize::MAX);
            if ab_rpo != usize::MAX && read_rpo != usize::MAX && ab_rpo < read_rpo {
                rpo_precedes = true;
            }
            if reaches_without_backedge(blocks, ab, read_bb, &back_edges) {
                forward_reaches = true;
            }
        }

        let category;
        let detail;
        if direct_assign_blocks.is_empty() {
            // No direct binding site at all (only projection writes, or
            // none). RPO ordering cannot help; this is not a plain
            // forward-ref. Treat as loop-carried/other and explain.
            if proj_assign_blocks.is_empty() {
                category = "unknown";
                detail = format!(
                    "local {local_n} read at bb{read_bb} has NO Assign anywhere \
                     (direct or projection); cannot classify"
                );
                other += 1;
            } else {
                category = "loop-carried";
                detail = format!(
                    "local {local_n} read at bb{read_bb} has only projection-base \
                     writes at {proj_assign_blocks:?} (no direct Local({local_n}) \
                     Assign to seed local_var); RPO ordering cannot bind it"
                );
                loop_carried += 1;
            }
        } else if rpo_precedes || forward_reaches {
            category = "forward-ref";
            forward_ref += 1;
            detail = format!(
                "direct Assign(Local({local_n})) at blocks {direct_assign_blocks:?}; \
                 read at bb{read_bb}; RPO read-rank={read_rpo}; \
                 rpo_precedes={rpo_precedes} forward_reaches={forward_reaches} \
                 (an assign-block precedes/forward-reaches the read => RPO traversal \
                 alone binds it, no phi){}",
                if proj_assign_blocks.is_empty() {
                    String::new()
                } else {
                    format!("; proj-writes at {proj_assign_blocks:?}")
                }
            );
        } else {
            category = "loop-carried";
            loop_carried += 1;
            detail = format!(
                "direct Assign(Local({local_n})) at blocks {direct_assign_blocks:?}; \
                 read at bb{read_bb}; RPO read-rank={read_rpo}; back-edges={:?}; \
                 every assign-block reaches the read ONLY through a back-edge \
                 (none precede it in RPO, none forward-reach it) => genuine \
                 loop-carried value needing a phi/block-inputarg{}",
                back_edges,
                if proj_assign_blocks.is_empty() {
                    String::new()
                } else {
                    format!("; proj-writes at {proj_assign_blocks:?}")
                }
            );
        }

        eprintln!("CLASSIFY [{category}] {name} | {detail}");
    }

    eprintln!("\n=== uninitialised-local classification tally ===");
    eprintln!("forward_ref  (RPO-fixable, no phi): {forward_ref}");
    eprintln!("loop_carried (needs phi/inputarg):  {loop_carried}");
    eprintln!("other/unknown:                       {other}");
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
            "skip: set PYRE_MIR_STRESS_LLBC or run scripts/extract-llbc.py to make \
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
    eprintln!(
        "total Call/Assert/Drop terminators inspected: {}",
        tally.total_call_terms
    );
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
    eprintln!(
        "  chains doing real (non-trivial) work: {}",
        tally.real_work
    );
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
    // The decisive invariant — no on_unwind path does
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

/// Diagnostic: emit an alpha-equivalence
/// structural signature for every function that lowers OK, so the
/// reverse-postorder lowering order can be proven to alter ONLY the
/// binding of the 15 forward-ref failures and nothing else.
///
/// The signature is invariant under Variable *renaming*: every operand,
/// op-result, inputarg, exit-arg and exitswitch Variable is replaced by
/// its def-site *position* (`b{block}o{op}` / `b{block}i{arg}`), which is
/// independent of allocation order. Op kinds are tagged by discriminant
/// — op params, op order/count and block structure are all
/// reverse-postorder-invariant (each block lowers its own fixed MIR
/// statements regardless of *when* it is processed), so the only thing
/// that can move under a traversal-order change is *which Variable a read
/// resolves to*, and that shows up as a changed def-site label. If
/// reverse-postorder rebinds any read the signature differs; if it
/// changes nothing the signature is byte-identical.
///
/// Prints `SIG\t{idx}\t{name}\t{hash}` per fn (`idx` = iteration order so
/// pre/post outputs align line-by-line even across name collisions); a
/// function that fails to lower prints hash `FAIL`. Capture before and
/// after the change, then `diff`: the expected delta is exactly the 15
/// forward-ref fns flipping `FAIL`->hash, with zero hash->hash changes.
#[test]
#[ignore = "diagnostic; set PYRE_MIR_STRESS_LLBC"]
fn dump_lowering_signatures() {
    use majit_translate::model::{ExitSwitch, FunctionGraph, LinkArg};
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};

    let Some(path) = stress_path() else {
        eprintln!("skip: set PYRE_MIR_STRESS_LLBC");
        return;
    };
    let llbc = Llbc::load(&path).expect("load stress llbc");

    fn signature(graph: &FunctionGraph) -> String {
        // def-site label for every Variable defined in the graph: an
        // inputarg slot or an op-result, keyed by position (RPO-stable).
        let mut site: HashMap<u64, String> = HashMap::new();
        for (b, blk) in graph.blocks.iter().enumerate() {
            for (k, v) in blk.inputargs.iter().enumerate() {
                site.insert(v.id(), format!("b{b}i{k}"));
            }
            for (j, op) in blk.operations.iter().enumerate() {
                if let Some(r) = &op.result {
                    site.insert(r.id(), format!("b{b}o{j}"));
                }
            }
        }
        // Synthetic Variables that are neither an inputarg nor an op
        // result — e.g. the etype/evalue a `set_raise` passes into the
        // exceptblock — carry an allocation-order-dependent raw id.
        // Canonicalise them by first-encounter order (which follows the
        // RPO-stable block/op iteration, NOT the allocation order) so the
        // signature stays alpha-invariant; a raw id would otherwise make
        // every panic/unwind-bearing fn spuriously differ pre/post.
        let mut unknown: HashMap<u64, String> = HashMap::new();
        let mut label = |id: u64| -> String {
            if let Some(l) = site.get(&id) {
                return l.clone();
            }
            let next = unknown.len();
            unknown
                .entry(id)
                .or_insert_with(|| format!("U{next}"))
                .clone()
        };
        let mut s = String::new();
        for (b, blk) in graph.blocks.iter().enumerate() {
            s.push_str(&format!("B{b}["));
            for v in &blk.inputargs {
                s.push_str(&format!("in:{};", label(v.id())));
            }
            for op in &blk.operations {
                // discriminant tag (param-free; RPO never changes params).
                let disc = format!("{:?}", std::mem::discriminant(&op.kind));
                s.push_str(&format!("op<{disc}>("));
                for operand in majit_translate::inline::op_variable_refs(&op.kind) {
                    s.push_str(&label(operand.id()));
                    s.push(',');
                }
                s.push(')');
            }
            match &blk.exitswitch {
                Some(ExitSwitch::Value(v)) => s.push_str(&format!("sw:{};", label(v.id()))),
                Some(ExitSwitch::LastException) => s.push_str("swLE;"),
                // `optimize_goto_if_not` fuses a compare into a `Fused`
                // switch; the flat MIR driver never produces one (it is a
                // jtransform-stage rewrite), so this arm only keeps the
                // match exhaustive.
                Some(ExitSwitch::Fused { opname, args }) => {
                    s.push_str(&format!("swF:{opname}("));
                    for a in args {
                        s.push_str(&format!("{},", label(a.id())));
                    }
                    s.push_str(");");
                }
                None => {}
            }
            for link in &blk.exits {
                // The remaining Variable-carrying Link fields,
                // `last_exception` / `last_exc_value`, are the only other
                // operands a rebind could touch.  The flat MIR driver
                // never populates them (rtyper-stage fields), so the
                // signature omits them — assert that invariant so the
                // omission is provably safe rather than an oversight.
                // `exitcase` / `llexitcase` are deliberately omitted too,
                // but carry no Variable (pure MIR-switch constants,
                // RPO-invariant), so they need no guard.
                assert!(
                    link.last_exception.is_none() && link.last_exc_value.is_none(),
                    "MIR-built Link carries last_exception/last_exc_value; \
                     signature() must label them"
                );
                s.push_str(&format!("->{:?}(", link.target));
                for a in &link.args {
                    match a {
                        LinkArg::Value(v) => s.push_str(&format!("{},", label(v.id()))),
                        LinkArg::Const(_) => s.push_str("K,"),
                    }
                }
                s.push(')');
            }
            s.push(']');
        }
        s
    }

    for (idx, fd) in llbc.iter_local_fns().enumerate() {
        if fd.is_global_initializer.is_some() {
            continue;
        }
        let Some(_body): Option<Unstructured> = fd.unstructured() else {
            continue;
        };
        let name = fd.item_meta.name_path();
        let hash = match lower_fun_decl(&llbc, fd) {
            Ok(graph) => {
                let sig = signature(&graph);
                let mut h = std::collections::hash_map::DefaultHasher::new();
                sig.hash(&mut h);
                format!("{:016x}", h.finish())
            }
            Err(_) => "FAIL".to_string(),
        };
        println!("SIG\t{idx}\t{name}\t{hash}");
    }
}
