//! The JIT portal: `shift` traced with the regex tree held constant.
//!
//! "A JIT for Regular Expression Matching" makes the regex the JitDriver's
//! green and the input position a red. The tracer then sees the tree as
//! constant, inlines the whole `shift` recursion for one character, and folds
//! every structural read away — what is left is the marks and the character
//! comparisons, which is subset construction performed by the tracer.
//!
//! Two things this file has to say in majit's vocabulary:
//!
//! * `#[jit_interp]` drives a bytecode portal, so the loop over the input is
//!   spelled as a three-instruction program whose back edge closes at `pc = 0`.
//!   `pc` and `program` are the greens and both are constant there, so the
//!   whole input is one green key and one trace.
//! * The regex root is a red `ref(NodeRec)` promoted once per iteration. The
//!   promote is what makes the root a `ConstPtr`; the `_immutable_fields_`
//!   declaration on `NodeRec` is what makes every read below it fold, so the
//!   walk collapses into the marks alone.
//!
//! The result, measured on `(a|b)*a(a|b){20}a(a|b)*` — 93 nodes — and
//! identical on dynasm and cranelift. The peeled loop body, which is what runs
//! per input character:
//!
//! ```text
//! 194 ops: 0 getfield_gc_r, 24 getfield_gc_i, 93 setfield_gc,
//!          2 int_eq, 0 guard_value, 1 getarrayitem_gc_i
//! ```
//!
//! Not one pointer read: the 92 edges of the tree walk are gone. 93 stores is
//! one mark per node, so every node is in the trace. The promote and the
//! input buffer's base pointer are loop invariant and sit in the preamble. And
//! `2 int_eq` for 46 `Char` nodes is the subset construction itself — a node
//! whose incoming mark the optimizer proved constant zero has nothing to
//! compare, so its comparison is not in the loop at all.
//!
//! Association is free, as it should be: the depth-26 left-associated tree
//! produces the same 194-op body as the depth-8 balanced one.

use crate::regex::NodeRec;
use majit_metainterp::JitDriver;

/// Shift one mark through the tree for character `c`.
///
/// `interp::shift` with the same body: the kinds are spelled as literals
/// because a value-position comparison against a `const` path is not part of
/// the lowerable vocabulary, and the arms are ordered so the two recursive
/// shapes sit together.
#[majit_macros::jit_inline(
    ref_params = { n: ref(NodeRec) },
    ref_fields = {
        NodeRec::left => NodeRec,
        NodeRec::right => NodeRec,
    },
    int_fields = {
        NodeRec::kind => u8,
        NodeRec::ch => u8,
        NodeRec::empty => u8,
        NodeRec::marked => u8,
    },
)]
fn shift(n: usize, c: i64, mark: i64) -> i64 {
    let k = n.kind as i64;
    let m = if k == 0i64 {
        // Char
        let ch = n.ch as i64;
        mark & ((ch == c) as i64)
    } else if k == 2i64 {
        // Alternative
        shift(n.left, c, mark) | shift(n.right, c, mark)
    } else if k == 3i64 {
        // Sequence. The left mark from the PREVIOUS character is what enters
        // the right side, so read it before `shift` overwrites it.
        let l = n.left;
        let r = n.right;
        let old_left = l.marked as i64;
        let marked_left = shift(l, c, mark);
        let marked_right = shift(r, c, old_left | (mark & l.empty as i64));
        (marked_left & r.empty as i64) | marked_right
    } else if k == 4i64 {
        // Repetition
        shift(n.left, c, mark | n.marked as i64)
    } else {
        // Epsilon, and anything else: no mark ever comes out.
        0i64
    };
    n.marked = m as u8;
    m
}

// ── the portal ─────────────────────────────────────────────────────────────

pub type Bytecode = [u8];

/// Shift the next character in, and advance.
const OP_SHIFT: u8 = 0;
/// Close the back edge while the input has characters left.
const OP_LOOP: u8 = 1;
const OP_HALT: u8 = 2;

/// The whole "program". `pc` and `program` are the greens, and the back edge
/// returns to `pc = 0`, so every character shares one green key — the regex,
/// which the loop promotes, is what the trace actually specializes on.
pub const PROGRAM: [u8; 3] = [OP_SHIFT, OP_LOOP, OP_HALT];

/// The input string, as a headerless buffer plus its length.
///
/// It is reached through a `ref` state field and read with `array_fields`, NOT
/// carried as an `[int; virt]` state array: a virtualizable array's elements
/// ride every guard's `vable_array` resume section, so the snapshot would be
/// O(input length) — past 32767 characters that overflows the tagged short a
/// resume entry is written as (`resumecode.py Writer.append_int`).
#[repr(C)]
struct Input {
    data: *mut u8,
    len: i64,
}

struct MatchState {
    /// The lowered regex. Red, then promoted: the promote is the one guard the
    /// specialization costs.
    root: usize,
    /// The input buffer.
    inp: usize,
    pos: i64,
    len: i64,
    result: i64,
}

pub static COMPILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static LAST_OPS_AFTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static LAST_BODY: std::sync::Mutex<Vec<majit_ir::OpCode>> = std::sync::Mutex::new(Vec::new());

#[majit_macros::jit_interp(
    state = MatchState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        root: ref(NodeRec),
        inp: ref(Input),
        pos: int,
        len: int,
        result: int,
    },
    array_fields = { Input::data => u8 },
    ref_fields = {
        NodeRec::left => NodeRec,
        NodeRec::right => NodeRec,
    },
    int_fields = {
        NodeRec::kind => u8,
        NodeRec::ch => u8,
        NodeRec::empty => u8,
        NodeRec::marked => u8,
    },
    calls = { shift => inline_int },
)]
#[allow(unused_assignments, unused_variables)]
fn mainloop(
    program: &Bytecode,
    threshold: u32,
    root: usize,
    inp: usize,
    len: i64,
    first: i64,
) -> i64 {
    let mut driver: JitDriver<MatchState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, ops_after, opcodes| {
        COMPILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        LAST_OPS_AFTER.store(ops_after, std::sync::atomic::Ordering::Relaxed);
        *LAST_BODY.lock().unwrap() = opcodes.to_vec();
    });
    let mut pc: usize = 0;
    let mut state = MatchState {
        root,
        inp,
        pos: 1i64,
        len,
        result: first,
    };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_SHIFT => {
                let root = state.root;
                majit_metainterp::jit::promote(root);
                let idx = state.pos as usize;
                let c = state.inp.data[idx] as i64;
                state.result = shift(root, c, 0i64);
                state.pos = state.pos + 1i64;
            }
            OP_LOOP => {
                if state.pos < state.len {
                    can_enter_jit!(driver, 0usize, &mut state, program, || {});
                    pc = 0;
                    continue;
                }
            }
            OP_HALT => break,
            _ => break,
        }
    }
    state.result
}

/// `interp::matches`, with the per-character loop handed to the JIT.
///
/// The first character is shifted in outside the loop — it is the one that
/// carries a mark in from the left — so the loop body is uniform and the trace
/// has no first-iteration special case.
pub fn matches(root: *mut NodeRec, s: &[u8], threshold: u32) -> bool {
    if s.is_empty() {
        return unsafe { (*root).empty } != 0;
    }
    let first = crate::interp::shift(root, s[0] as i64, 1);
    let result = if s.len() == 1 {
        first
    } else {
        let mut input = Input {
            data: s.as_ptr() as *mut u8,
            len: s.len() as i64,
        };
        mainloop(
            &PROGRAM,
            threshold,
            root as usize,
            &mut input as *mut Input as usize,
            input.len,
            first,
        )
    };
    crate::interp::reset(root);
    result != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regex::{bench_regex, count, lower, nonmatching, vectors};
    use majit_ir::OpCode;
    use std::sync::atomic::Ordering;

    /// `COMPILES` / `LAST_BODY` are process-wide and libtest runs these in
    /// parallel, so EVERY test that enters the JIT takes this — not only the
    /// ones that read the counters. A test that enters without it does not read
    /// the numbers, but it does move them.
    static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn matches_locked(root: *mut NodeRec, s: &[u8], threshold: u32) -> bool {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        matches(root, s, threshold)
    }

    /// One measured run: `(matched, loops compiled, last compiled body)`.
    ///
    /// Takes [`PROBE_LOCK`] itself, so callers must not also hold it — the
    /// plain mutex would deadlock on one thread.
    fn measure(root: *mut NodeRec, s: &[u8]) -> (bool, usize, Vec<OpCode>) {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        COMPILES.store(0, Ordering::Relaxed);
        LAST_BODY.lock().unwrap().clear();
        let matched = matches(root, s, 3);
        let compiles = COMPILES.load(Ordering::Relaxed);
        let body = LAST_BODY.lock().unwrap().clone();
        (matched, compiles, body)
    }

    /// The peeled loop body — everything from the LAST `Label` on.
    ///
    /// A compiled loop is preamble plus peeled body, and only the peeled body
    /// runs per character. Grading the whole thing double-counts, and it also
    /// charges the body for the reads the preamble hoisted out of it.
    fn loop_body(body: &[OpCode]) -> &[OpCode] {
        match body.iter().rposition(|op| *op == OpCode::Label) {
            Some(i) => &body[i..],
            None => body,
        }
    }

    /// The op census a compiled body is graded on.
    fn census(body: &[OpCode]) -> String {
        let n = |op: OpCode| body.iter().filter(|o| **o == op).count();
        format!(
            "{} ops: {} getfield_gc_r, {} getfield_gc_i, {} setfield_gc, {} int_eq, {} guard_value",
            body.len(),
            n(OpCode::GetfieldGcR),
            n(OpCode::GetfieldGcI),
            n(OpCode::SetfieldGc),
            n(OpCode::IntEq),
            n(OpCode::GuardValue),
        )
    }

    fn degraded() -> Vec<(&'static str, &'static str)> {
        majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "MatchState")
            .map(|a| (a.arm, a.reason))
            .collect()
    }

    #[test]
    fn the_jit_matcher_agrees_with_the_interpreter() {
        for (re, s, want) in vectors() {
            let root = lower(&re);
            assert_eq!(matches_locked(root, s.as_bytes(), 3), want, "input {s:?}");
        }
    }

    /// Cold and warm must agree on an input long enough to cross the
    /// threshold: a compiled loop that disagrees is the failure this catches.
    #[test]
    fn warm_agrees_with_cold_on_a_long_input() {
        let s = nonmatching(4096, 20, 42);
        let root = lower(&bench_regex(20));
        let cold = matches_locked(root, &s, u32::MAX);
        let warm = matches_locked(root, &s, 3);
        assert!(!cold, "the benchmark input is supposed NOT to match");
        assert_eq!(warm, cold);
    }

    /// The non-matching input is the benchmark's, and it is also the input on
    /// which most of the tree carries no mark — so a trace specialized on it
    /// would still pass the test above. This one matches.
    #[test]
    fn warm_agrees_with_cold_on_a_matching_input() {
        let mut s = nonmatching(4096, 20, 42);
        // Two `a`s exactly 21 apart is the whole regex.
        s[100] = b'a';
        s[121] = b'a';
        let root = lower(&bench_regex(20));
        let cold = matches_locked(root, &s, u32::MAX);
        let warm = matches_locked(root, &s, 3);
        assert!(cold, "the reinstated pair is supposed to match");
        assert_eq!(warm, cold);
    }

    /// Association changes the tree depth and nothing else, and the depth is
    /// what the tracer's inline ceiling counts. 26 levels against the balanced
    /// 8 — if the ceiling refuses the deep one, this is where it shows, and the
    /// answers must still agree either way.
    #[test]
    fn the_left_associated_tree_gives_the_same_answers() {
        use crate::regex::bench_regex_left;
        let s = nonmatching(2048, 20, 7);
        let left = lower(&bench_regex_left(20));
        let balanced = lower(&bench_regex(20));

        let (deep_matched, deep_compiles, deep_body) = measure(left, &s);
        let (flat_matched, _, _) = measure(balanced, &s);
        assert_eq!(deep_matched, flat_matched, "association changed the answer");
        println!(
            "[regex-jit] depth 26: {deep_compiles} compiled, per character: {}",
            census(loop_body(&deep_body)),
        );
        assert!(
            deep_compiles > 0,
            "the depth-26 tree compiled nothing: the tracer's inline ceiling \
             refused the walk that the depth-8 tree fits inside; degraded={:?}",
            degraded(),
        );

        let mut hit = s.clone();
        hit[500] = b'a';
        hit[521] = b'a';
        assert!(matches_locked(left, &hit, 3));
    }

    /// The experiment. One trace covers the whole input, and the 93-node tree
    /// walk inside it costs no pointer read at all: every `left` / `right` is
    /// declared immutable and its base is constant, so the walk is folded away
    /// and what survives is the marks.
    ///
    /// `GetfieldGcR` is the subject and zero is the claim. The denominator is
    /// printed beside it — a body of a handful of ops would also have no
    /// pointer reads, and would mean the walk never entered the trace.
    #[test]
    fn the_tree_walk_folds_to_marks_and_comparisons() {
        let s = nonmatching(4096, 20, 42);
        let root = lower(&bench_regex(20));
        let nodes = count(root);

        let (matched, compiles, full) = measure(root, &s);
        assert!(!matched, "the benchmark input is supposed NOT to match");
        assert!(
            compiles > 0,
            "nothing compiled, so every character ran through the \
             interpreter; degraded={:?}",
            degraded(),
        );
        let body = loop_body(&full);
        println!(
            "[regex-jit] {nodes} nodes, {compiles} compiled, per character: {}",
            census(body),
        );

        let n = |op: OpCode| body.iter().filter(|o| **o == op).count();
        let stores = n(OpCode::SetfieldGc);
        let ptr_reads = n(OpCode::GetfieldGcR);

        assert!(
            stores >= nodes,
            "the walk stored {stores} marks for a {nodes}-node tree, so it did \
             not reach every node: the recursion was cut short rather than \
             traced through; body={body:?}"
        );
        assert_eq!(
            ptr_reads, 0,
            "the walk reloaded a `left`/`right` pointer {ptr_reads} times. \
             Those reads are what `_immutable_fields_` plus a promoted root are \
             supposed to fold; body={body:?}"
        );
        assert_eq!(
            n(OpCode::GetarrayitemGcI),
            1,
            "one character is read per pass and the buffer base is loop \
             invariant, so exactly one array read belongs here; body={body:?}"
        );
        assert_eq!(
            n(OpCode::GuardValue),
            0,
            "the root promote is loop invariant and belongs in the preamble; a \
             per-pass `guard_value` means it was not hoisted; body={body:?}"
        );
    }
}
