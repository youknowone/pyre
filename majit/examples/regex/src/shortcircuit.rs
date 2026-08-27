//! The post's matcher as its own source writes it: a `shift` that branches.
//!
//! This is the module RPython's JIT produces the same trace for — see the
//! cross-check below. `jit_interp` next door is the *adapted* portal the
//! post's short-circuit remark tells you to write.
//!
//! "A JIT for Regular Expression Matching" says, without showing the numbers:
//!
//! > the JIT-version of the code needs to be adapted to use the
//! > non-short-circuiting operators `&` and `|` ... If you don't change the
//! > "and" and "or" you get a lot of assembler code generated, and it's not
//! > particularly fast
//!
//! `interp.rs` and `jit_interp.rs` both take that advice, and this crate had
//! documented it without ever measuring it. This module is the unadapted side
//! of the A/B, and therefore the post's own source: `jit_interp`'s portal
//! copied whole — same state fields, same greens, same three-instruction
//! program, same promote, same `_immutable_fields_` declarations, same inline
//! policy — with `shift`'s four arms written the way part 1's Python writes
//! them, `and`/`or` that stop early. The state struct's NAME is the one other
//! difference, and only so a degraded arm stays attributable; the field list is
//! identical. So the two portals differ in exactly the thing the remark is
//! about, which is what makes the comparison below an A/B rather than two
//! programs.
//!
//! # This is the module RPython's own JIT agrees with
//!
//! `rpython_original/` carries the post's matcher written in RPython — the
//! class hierarchy, `_immutable_fields_`, and a JitDriver whose green is the
//! regex — and `runner.py` puts it through `LLJitMixin`, the harness RPython's
//! own JIT tests use: the real metainterpreter, the real optimizer, the LLGraph
//! backend, in process. The loop that comes out is the trace RPython would
//! compile. Both sides scan the same 4096 bytes, pinned by an FNV-1a digest
//! asserted on both (`NONMATCHING_4096_FNV1A`).
//!
//! Peeled body, 93 nodes:
//!
//! ```text
//!                    RPython   this module   jit_interp
//!   getfield_gc_r          0             0            0
//!   getfield_gc_i         24            24           24
//!   setfield_gc           93            93           93
//!   int_eq                 2             2            2
//!   guard_true             6             6            1
//!   guard_false           21            20            0
//! ```
//!
//! Every structural count is exact: no pointer read survives on either side,
//! one mark store per node, and two comparisons for 46 `Char` nodes — the
//! subset construction, performed by both tracers. The loop tails are the same
//! ops in the same order (`int_add`, `setfield_gc`, `int_lt`, `guard_true`,
//! `jump`).
//!
//! **And the guards say this module, not `jit_interp`, is the post's own
//! spelling.** The post's source is Python `and`/`or`, which really do
//! short-circuit, so RPython's tracer emits branches — 27 guards. This module
//! reproduces them with explicit `if`s and lands on 26. `jit_interp` masks with
//! `&`/`|` and has 1. So `jit_interp` is the *adapted* version the post's
//! remark tells you to write, and this module is what the post's code does
//! before you take that advice.
//!
//! # Where the 23 remaining ops come from
//!
//! 153 ops (RPython) against 176 (here). The difference is the port's integer
//! typing, and it accounts for every one of the 23.
//!
//! RPython carries the marks as `Bool`. A `Bool` local *is* a branch
//! condition, so `flatten.py` emits a plain `goto_if_not` and
//! `pyjitpl.py opimpl_goto_if_not` hands that box straight to
//! `generate_guard` — no truth test. A `Bool` field is likewise stored as it
//! stands, with no mask.
//!
//! Our `NodeRec.marked` is a `u8` and `shift` carries marks as `i64`, so both
//! conversions become real operations:
//!
//! * `if mark != 0` is an `IntIsTrue` — **24** of them, exactly one per guard
//!   whose condition is not already a comparison (26 guards, less the loop
//!   exit's `IntLt` and one `SetfieldGc`-fed guard); and
//! * `n.marked = m as u8` is an `IntAnd(m, 255)` — **2** survive, on the two
//!   live `Char` marks.
//!
//! 24 + 2, less RPython's one extra `guard_false` and its two zero-cost
//! `debug_merge_point`s, is 23.
//!
//! This is not a place majit falls short of RPython. `rewrite.py
//! optimize_INT_IS_TRUE` folds the test only when the argument's bounds are
//! `is_bool()`, and a one-byte unsigned field read is [0, 255] — upstream's
//! own `FieldDescr.get_integer_min` / `get_integer_max` answer the same for
//! `lltype.Bool`, since it is `FLAG_UNSIGNED` at size 1. The fold arm is
//! ported and fires elsewhere; it has nothing to fire on here. The two
//! `IntAnd` are the same story from the other end: `autogenintrules.py`'s
//! `and_x_c_in_range` would remove `int_and(x, 255)` for an `x` bounded by
//! [0, 1], but the `IntEq` producing that `x` is a postponed pure op forced
//! out *after* its consumer was optimized, so `make_bool` had not run on it
//! yet. Upstream never meets the pattern, because it never masks a `Bool`.
//!
//! `REGEX_LISTING=1` on the census test prints both peeled bodies op by op,
//! and `runner.py --listing` prints RPython's, for diffing.
//!
//! # The measurements
//!
//! Measured on this machine, dynasm, on the post's own
//! `(a|b)*a(a|b){20}a(a|b)*` — 93 nodes, balanced — over a non-matching random
//! `a`/`b` string. Two separate claims, because they have separate evidence.
//!
//! # 1. The post's claim, about code shape. It holds.
//!
//! The peeled loop body, which is what runs per input character:
//!
//! ```text
//!            ops guard_true guard_false gf_gc_r gf_gc_i setfield_gc int_eq guard_value
//! masking    194     1           0         0      24        93        2        0
//! branching  176     6          20         0      24        93        2        0
//! ```
//!
//! One guard against twenty-six. The masking body's single guard is the loop
//! exit — the `pos < len` test — and the branching body carries that one plus
//! twenty-five more, one per `and`/`or` whose operand the optimizer could not
//! prove constant. That is exactly the mechanism the post names: a branch the
//! tracer records becomes a guard, and masking has no branch to record.
//!
//! The surprise is the op count. The branching body is SMALLER, not bigger —
//! 176 against 194 — because a guard replaces the `int_and` / `int_or` it
//! stands in for, and because the dead side of a branch takes its ops out of
//! the trace with it. The specialization is untouched too: both bodies fold
//! the whole 93-node tree walk to zero `getfield_gc_r`, both store one mark
//! per node, and both reduce 46 `Char` nodes to 2 `int_eq`. Both compile
//! exactly one loop, and `degraded_dispatch_arms()` is empty for both, so the
//! branching body lowers completely — this is NOT the "refuses to lower"
//! outcome.
//!
//! So the post's "a lot of assembler code generated" is not about the loop
//! body: that gets smaller. It is about what the twenty-six guards go on to
//! cost, which is section 2.
//!
//! Both backends agree op for op on every number in that table, and on the
//! runtime counts below: dynasm and cranelift were measured separately and
//! differ only in the throughput rows.
//!
//! # 2. What majit's runtime then does with those guards. This is the cost.
//!
//! Over a 4096-character non-matching input, the two portals record:
//!
//! ```text
//! branching  guard failures 4080   bridges compiled 10   traces aborted 0
//! masking    guard failures    1   bridges compiled  0   traces aborted 0
//! ```
//!
//! The masking portal is the control and it is the number to read first: over
//! the same 4096 characters it leaves its compiled loop **once**. The
//! branching portal leaves it 4080 times — one guard failure per input
//! character — and it stays one per character even though ten bridges get
//! built. That is the shape of the finding: bridging works here, and it does
//! not catch up.
//!
//! The ten bridges are real machine code and they are entered. Before them
//! every failure in the run reported `tid=1` — the one compiled loop. Now the
//! failures spread across trace ids 1 through 11, the loop and its ten
//! bridges, which is what a bridge failing its OWN guard looks like: the loop
//! covers one path through the mark pattern, a bridge covers the next, that
//! bridge exits at a guard of its own, and the next bridge grows off THAT.
//! `must_compile` fires 10 times and all 10 reach `compile_trace`; none abort.
//!
//! Ten is not a ceiling the bridge tree ran into; it is a rate, and eight
//! times the input is what tells those apart:
//!
//! ```text
//!  4,096 chars:  4,080 guard failures, 10 bridges
//! 32,768 chars: 32,676 guard failures, 84 bridges
//! ```
//!
//! 8.0x the failures, 8.4x the bridges, and the failures stay at 0.997 per
//! character at both lengths. `trace_eagerness` is 200 — `PARAMETERS` carries
//! `rlib/jit.py`'s own default — so one guard must fail 200 times before it is
//! worth a bridge, which caps a pass of `n` characters at about `n / 200` of
//! them no matter how many distinct mark patterns the tree has. Bridging here
//! is not converging on the pattern and stopping; it is trailing it at a fixed
//! rate that the deopt outruns.
//!
//! So the post's "you get a lot of assembler code generated" IS what happens
//! here, once the bridges can be built at all — and it does not pay for
//! itself: the interpreter round trip per character stays, at every length
//! measured.
//!
//! ## What it took to get there, and what it says about majit
//!
//! Until this module was written, NO bridge was built here at all: the same
//! run recorded `guard failures 4090   bridges compiled 0   traces aborted 19`,
//! every one of the 19 aborting on the log line after its tracing session
//! opened, with zero bytecode-body ops walked —
//!
//! ```text
//! [bridgeB] multi-frame resume (7 frames) — only the root frame is seedable,
//!           giving up on the bridge
//! ```
//!
//! `shift` is a recursive `#[jit_inline]` helper, so a guard that fails inside
//! it resumes into a STACK of frames — 4 to 7 of them, the recursion depth at
//! the failing guard — and `start_bridge_tracing` refused whenever a
//! state-field driver's resume data spanned more than one, because
//! `setup_bridge_sym` seeds the root frame only. That refusal would have
//! applied to any `#[jit_interp]` machine whose hot loop calls an inlined
//! recursive helper; the masking portal was not spared it either, it simply
//! has one guard, which does not fail.
//!
//! It is fixed, in `majit-metainterp`, by doing what `resume.py
//! rebuild_from_resumedata` does: push one `MIFrame` per encoded resume
//! section, outermost first, each `setup_resume_at_op(pc)` at its own position
//! with its own liveness-decoded registers, and resume in the innermost one.
//! Two things the resume stream does not carry had to be recovered from the
//! bytecode the way `pyjitpl.py make_result_of_lastop` recovers them: which
//! callee each suspended caller is waiting on, and which of its registers
//! takes the result. `JitCode::inline_call_ending_at` is that read.
//!
//! `MAJIT_BRIDGE_DEBUG=1` now names each rebuilt stack instead of the give-up.
//! One 4096-character run rebuilds 10 of them — one 7-frame, three 6, three 5
//! and three 4 —
//!
//! ```text
//! [bridgeB] rebuild 7 frame(s): [(0, 101, 8), (1, 269, 7), (1, 269, 7),
//!           (1, 269, 7), (1, 269, 7), (1, 214, 2), (1, 131, 4)]
//! ```
//!
//! — jitcode 0 is the mainloop and jitcode 1 is `shift`, so the stack reads
//! directly as "the mainloop called `shift`, which called itself five more
//! times". The distinct `pc`s in the `shift` frames are the distinct call
//! sites: 269 is the recursive descent, 214 and 131 the shallower ones.
//!
//! # Throughput
//!
//! Median of 5 over 1048576 characters, one untimed warm-up first, all three
//! rows measured in one process by one test. Three runs, one before the
//! multi-frame bridge fix and two after it, each with the 1-minute load
//! average it ran at — which is the only way these rows can be read:
//!
//! ```text
//!                        pre-fix        post-fix A      post-fix B
//!   1-min load         7.7 -> 8.2      3.4 -> 50       34 -> 83
//!   masking          46,889,491       17,327,682      17,382,474
//!   no JIT at all     8,789,264        3,840,742       3,807,803
//!   branching            22,034           10,154           9,407
//!   masking / branching    2128x            1706x           1848x
//!   no JIT  / branching     399x             378x            405x
//! ```
//!
//! **All three absolute rows fell by about the same factor** — masking 2.7x,
//! no-JIT 2.3x, branching 2.2x — which is what a loaded machine does to every
//! row at once and is not what a code change does to one of them. The ratios,
//! which are taken inside a single process and so survive that, did not move:
//! the verdict row reads 378x and 405x after the fix against 399x before it.
//!
//! So the fix does not show up here, and the reason is the bridge arithmetic
//! above: `trace_eagerness` is 200, a 1M-character pass can grow at most a few
//! thousand bridges, and the mark pattern has far more states than that. The
//! branching portal stays about 400x slower than running the same algorithm
//! with no JIT under it at all.
//!
//! The in-process ratio the suite runs on every invocation, at 4096
//! characters, is the one quantity that did move: 66x, 68x and 89x on dynasm
//! before the fix; four runs after it, at 1-minute load 33 to 36, read 32x,
//! 35x, 50x and 51x. The band moved and does not overlap, which is the
//! direction the ten bridges predict; the spread inside it is the load, and
//! neither band is a number to quote to two digits.
//!
//! The third row is what turns "slower" into a verdict, and the fix does not
//! overturn it. The branching portal does not merely give back the JIT's win
//! over the plain matcher: it is far slower than running the same algorithm
//! with no JIT under it at all, because a character that misses every bridge
//! still pays an interpreter round trip the plain matcher never pays. "Not
//! particularly fast" is, here, orders of magnitude.
//!
//! Each column is three rows of one process, which is what makes the ratios
//! worth quoting: the machine was never idle, and a ratio taken inside one run
//! survives that where an absolute row does not. Earlier runs of the same
//! source at higher load read 1716x, 1579x and 2336x — the branching row is
//! the load-sensitive one and moves by up to 1.7x, while the finding does not.
//!
//! The 4096-character ratio is much smaller than the 1M one either way. It
//! grows with input length because `matches` builds a fresh `JitDriver` per
//! call, so at 4096 characters the masking row is still paying for a recording
//! it only amortizes over a long input, while the branching row has far less
//! to amortize it against — ten bridges that each serve one path, against a
//! compiled loop that serves every character.
//!
//! # The spelling the post's advice suggests literally
//!
//! Swapping `&`/`|` for `&&`/`||` and leaving the shape alone is NOT the A/B
//! this module runs, for two reasons, both measured on this machine in the
//! same session as the tables above.
//!
//! First, in a lowered body `&&` and `||` are not short-circuit operators at
//! all. `opcode_for_binop` maps `BinOp::And` / `BinOp::Or` to `IntAnd` /
//! `IntOr`, so they lower to the same two ops `&` and `|` do, and
//! `lower_bool_if` collapses `if cond { 1 } else { 0 }` back to `cond` with no
//! branch emitted. Measured: with ONLY the `Char` arm rewritten that way, the
//! peeled body carries one guard — the loop exit — exactly as the masking
//! column does. A branch has to be written as a branch whose arms are not the
//! literals `1` and `0`, which is why the arms below hand back the operand,
//! the way Python's `and` / `or` actually evaluate.
//!
//! Second, writing that spelling here is what found a majit defect, now
//! fixed. `(mark != 0) as i64 | (n.marked as i64 != 0) as i64` in the
//! `Repetition` arm made the optimizer die after the trace recorded, on both
//! backends, in `OptContext::getnullness` under `optimize_int_is_true`. The
//! `int_or` of two `int_is_true` results asked about an operand that forwarded
//! to an op whose `Rc` had been dropped: `_forwarded` held a `Weak`, so
//! `get_box_replacement` stopped one hop early and handed back an operand that
//! was still forwarding, which `getnullness` asserts cannot happen. The slot
//! now owns its target the way RPython's `_forwarded` does
//! (`majit-ir/src/forwarding.rs`). Measured after the fix: that spelling in
//! the `Repetition` arm alone, in the `Sequence` arm alone, and in all four
//! arms at once each compiles and agrees with the interpreter on both
//! backends. `majit-ir`'s
//! `operand::tests::a_forwarded_target_survives_every_other_reference_being_dropped`
//! is the gate.

use crate::regex::{KIND_ALTERNATIVE, KIND_CHAR, KIND_REPETITION, KIND_SEQUENCE, NodeRec};
use majit_metainterp::JitDriver;

/// `jit_interp::shift`, arm for arm, with every `&`/`|` replaced by the
/// short-circuiting spelling part 1's Python uses.
///
/// The `and`/`or` are transcribed as Python evaluates them, not as booleans:
/// `x and y` is `if x { y } else { x }` and `x or y` is `if x { x } else { y }`,
/// which is why the else arms hand back the operand rather than a literal.
/// That matters twice over — it is what makes the field read on the far side of
/// the operator conditional (`mark and c == self.c` never touches `self.c` when
/// the mark is clear), and it is what keeps the macro from folding the whole
/// `if` away.
///
/// The recursive `shift` calls stay where the Python puts them: both children
/// of an `Alternative` and both sides of a `Sequence` are shifted before the
/// combinator runs, because each call stores a mark and skipping one would
/// change the answer, not just the trace.
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
    let k = n.kind;
    let m = match k {
        // `mark and c == self.c`
        KIND_CHAR => {
            if mark != 0 {
                let ch = n.ch as i64;
                (ch == c) as i64
            } else {
                mark
            }
        }
        // `marked_left or marked_right`
        KIND_ALTERNATIVE => {
            let marked_left = shift(n.left, c, mark);
            let marked_right = shift(n.right, c, mark);
            if marked_left != 0 {
                marked_left
            } else {
                marked_right
            }
        }
        // `self.re._shift(c, mark or self.marked)`
        KIND_REPETITION => {
            let inner_mark = if mark != 0 { mark } else { n.marked as i64 };
            shift(n.left, c, inner_mark)
        }
        KIND_SEQUENCE => {
            // The left mark from the PREVIOUS character is what enters the
            // right side, so read it before `shift` overwrites it.
            let l = n.left;
            let r = n.right;
            let old_left = l.marked as i64;
            let marked_left = shift(l, c, mark);
            // `old_marked_left or (mark and self.left.empty)`
            let into_right = if old_left != 0 {
                old_left
            } else if mark != 0 {
                l.empty as i64
            } else {
                mark
            };
            let marked_right = shift(r, c, into_right);
            // `marked_left and self.right.empty or marked_right`
            let left_closes = if marked_left != 0 {
                r.empty as i64
            } else {
                marked_left
            };
            if left_closes != 0 {
                left_closes
            } else {
                marked_right
            }
        }
        // Epsilon, and anything else: no mark ever comes out.
        _ => 0i64,
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
#[repr(C)]
struct Input {
    data: *mut u8,
    len: i64,
}

/// `jit_interp::MatchState`'s fields, under a different type name.
///
/// The name is the one deliberate deviation from "identical everything else":
/// `degraded_dispatch_arms()` reports an arm under its state type, and two
/// portals sharing the name would make a degraded arm unattributable — and
/// `jit_interp`'s own test filters that list on `interp == "MatchState"`.
struct ShortCircuitState {
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
/// Bridges compiled. A guard that keeps failing grows a bridge, and bridges are
/// the machine code the post's "a lot of assembler code generated" is about —
/// the loop counter cannot see them, because a guard that declines to bridge
/// still deopts through the blackhole and leaves the loop count at 1.
pub static BRIDGES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Traces abandoned. `COMPILES > 0` does not rule this out: a bridge can abort
/// while the loop stands.
pub static ABORTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Guard failures, counted only while [`GUARD_FAILURE_PROBE`] is set.
pub static GUARD_FAILURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Whether [`mainloop`] installs the guard-failure callback at all.
///
/// The other three callbacks fire on compile and abort events — a handful of
/// times per run, and never on the path a character takes. This one fires on
/// every deopt, so leaving it installed would put an instrument inside the very
/// quantity the timing row measures. It is therefore installed per call, and
/// the timed rows run with it off — `jit_interp`'s portal gates its own the
/// same way, which is what keeps the two portals' timed rows comparable.
pub static GUARD_FAILURE_PROBE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// The last run's aborts split by `Counters.ABORT_*` reason, indexed as
/// `majit_metainterp::jitprof::ABORT_COUNTER_KINDS`.
///
/// Read off the driver once, after the loop, because that is the only place
/// the driver is in scope; `set_on_trace_abort` reports `permanent` and not the
/// reason. Written per `matches` call, never per character.
pub static ABORT_REASONS: [std::sync::atomic::AtomicUsize; 6] = {
    const Z: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    [Z; 6]
};
static LAST_BODY: std::sync::Mutex<Vec<majit_ir::OpCode>> = std::sync::Mutex::new(Vec::new());

/// The peeled loop body of the last loop this portal compiled — everything
/// from the last `Label` on, which is what runs per input character.
///
/// A compiled loop is preamble plus peeled body. Grading the whole thing
/// double-counts, and it also charges the body for the reads the preamble
/// hoisted out of it.
pub fn last_peeled_body() -> Vec<majit_ir::OpCode> {
    let body = LAST_BODY.lock().unwrap_or_else(|e| e.into_inner());
    match body.iter().rposition(|op| *op == majit_ir::OpCode::Label) {
        Some(i) => body[i..].to_vec(),
        None => body.clone(),
    }
}

#[majit_macros::jit_interp(
    state = ShortCircuitState,
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
    use std::sync::atomic::Ordering::Relaxed;
    let mut driver: JitDriver<ShortCircuitState> = JitDriver::new(threshold);
    driver.set_on_compile_loop(|_gk, _before, _ops_after, opcodes| {
        COMPILES.fetch_add(1, Relaxed);
        *LAST_BODY.lock().unwrap() = opcodes.to_vec();
    });
    driver.set_on_compile_bridge(|_gk, _fail_index, _num_ops| {
        BRIDGES.fetch_add(1, Relaxed);
    });
    driver.set_on_trace_abort(|_gk, _permanent| {
        ABORTS.fetch_add(1, Relaxed);
    });
    if GUARD_FAILURE_PROBE.load(Relaxed) {
        driver.set_on_guard_failure(|_gk, _trace_id, _fail_index| {
            GUARD_FAILURES.fetch_add(1, Relaxed);
        });
    }
    let mut pc: usize = 0;
    let mut state = ShortCircuitState {
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
    for (i, slot) in ABORT_REASONS.iter().enumerate() {
        slot.store(driver.abort_diag(i) as usize, Relaxed);
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
    use std::time::Instant;

    /// The counters and `LAST_BODY` — this module's and `jit_interp`'s — are
    /// process-wide, and libtest runs these in parallel. EVERY test that enters
    /// either portal takes this, not only the ones that read the numbers: a
    /// test that enters without it does not read the counters, but it moves
    /// them.
    static PROBE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn matches_locked(root: *mut NodeRec, s: &[u8], threshold: u32) -> bool {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        matches(root, s, threshold)
    }

    /// What one measured run of either portal answers.
    struct Run {
        matched: bool,
        compiles: usize,
        /// `None` for the masking portal, which installs no bridge callback.
        bridges: Option<usize>,
        aborts: Option<usize>,
        guard_failures: Option<usize>,
        body: Vec<OpCode>,
    }

    fn peeled(full: Vec<OpCode>) -> Vec<OpCode> {
        match full.iter().rposition(|op| *op == OpCode::Label) {
            Some(i) => full[i..].to_vec(),
            None => full,
        }
    }

    /// One measured run of the short-circuit portal, guard-failure probe on.
    fn measure(root: *mut NodeRec, s: &[u8]) -> Run {
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        COMPILES.store(0, Ordering::Relaxed);
        BRIDGES.store(0, Ordering::Relaxed);
        ABORTS.store(0, Ordering::Relaxed);
        GUARD_FAILURES.store(0, Ordering::Relaxed);
        GUARD_FAILURE_PROBE.store(true, Ordering::Relaxed);
        LAST_BODY.lock().unwrap().clear();
        let matched = matches(root, s, 3);
        GUARD_FAILURE_PROBE.store(false, Ordering::Relaxed);
        Run {
            matched,
            compiles: COMPILES.load(Ordering::Relaxed),
            bridges: Some(BRIDGES.load(Ordering::Relaxed)),
            aborts: Some(ABORTS.load(Ordering::Relaxed)),
            guard_failures: Some(GUARD_FAILURES.load(Ordering::Relaxed)),
            body: last_peeled_body(),
        }
    }

    /// The same, for the masking portal next door.
    ///
    /// It carries the same four counters, so this side is a control that
    /// answers rather than one that reports "not measured": whatever the
    /// branching portal's numbers mean, they mean it against these.
    fn measure_masking(root: *mut NodeRec, s: &[u8]) -> Run {
        use crate::jit_interp;
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        jit_interp::COMPILES.store(0, Ordering::Relaxed);
        jit_interp::BRIDGES.store(0, Ordering::Relaxed);
        jit_interp::ABORTS.store(0, Ordering::Relaxed);
        jit_interp::GUARD_FAILURES.store(0, Ordering::Relaxed);
        jit_interp::GUARD_FAILURE_PROBE.store(true, Ordering::Relaxed);
        jit_interp::LAST_BODY.lock().unwrap().clear();
        let matched = jit_interp::matches(root, s, 3);
        jit_interp::GUARD_FAILURE_PROBE.store(false, Ordering::Relaxed);
        Run {
            matched,
            compiles: jit_interp::COMPILES.load(Ordering::Relaxed),
            bridges: Some(jit_interp::BRIDGES.load(Ordering::Relaxed)),
            aborts: Some(jit_interp::ABORTS.load(Ordering::Relaxed)),
            guard_failures: Some(jit_interp::GUARD_FAILURES.load(Ordering::Relaxed)),
            body: peeled(jit_interp::LAST_BODY.lock().unwrap().clone()),
        }
    }

    fn n_of(body: &[OpCode], op: OpCode) -> usize {
        body.iter().filter(|o| **o == op).count()
    }

    fn guards(body: &[OpCode]) -> usize {
        n_of(body, OpCode::GuardTrue) + n_of(body, OpCode::GuardFalse)
    }

    fn census(body: &[OpCode]) -> String {
        format!(
            "{} ops: {} guard_true, {} guard_false, {} getfield_gc_r, \
             {} getfield_gc_i, {} setfield_gc, {} int_eq, {} guard_value",
            body.len(),
            n_of(body, OpCode::GuardTrue),
            n_of(body, OpCode::GuardFalse),
            n_of(body, OpCode::GetfieldGcR),
            n_of(body, OpCode::GetfieldGcI),
            n_of(body, OpCode::SetfieldGc),
            n_of(body, OpCode::IntEq),
            n_of(body, OpCode::GuardValue),
        )
    }

    /// The last run's aborts, `Counters.ABORT_*` label to count, zeroes
    /// dropped.
    fn abort_reasons() -> Vec<(&'static str, usize)> {
        majit_metainterp::jitprof::ABORT_COUNTER_KINDS
            .iter()
            .enumerate()
            .map(|(i, (_, label))| (*label, ABORT_REASONS[i].load(Ordering::Relaxed)))
            .filter(|(_, n)| *n != 0)
            .collect()
    }

    fn degraded() -> Vec<(&'static str, &'static str)> {
        majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "ShortCircuitState")
            .map(|a| (a.arm, a.reason))
            .collect()
    }

    #[test]
    fn the_short_circuit_matcher_agrees_with_the_interpreter() {
        for (re, s, want) in vectors() {
            let root = lower(&re);
            assert_eq!(matches_locked(root, s.as_bytes(), 3), want, "input {s:?}");
        }
    }

    /// Cold and warm must agree on an input long enough to cross the threshold
    /// — on one that does not match, and on one that does. A branch the tracer
    /// recorded one way and a guard that fails to re-check it shows up here and
    /// nowhere else, and the branching body has twenty-six places to get that
    /// wrong where the masking body has one.
    #[test]
    fn warm_agrees_with_cold_and_with_the_interpreter_on_a_long_input() {
        let s = nonmatching(4096, 20, 42);
        let root = lower(&bench_regex(20));
        let cold = matches_locked(root, &s, u32::MAX);
        let warm = matches_locked(root, &s, 3);
        assert!(!cold, "the benchmark input is supposed NOT to match");
        assert_eq!(warm, cold);
        assert_eq!(warm, crate::interp::matches(root, &s));

        let mut hit = s.clone();
        // Two `a`s exactly 21 apart is the whole regex.
        hit[100] = b'a';
        hit[121] = b'a';
        let cold = matches_locked(root, &hit, u32::MAX);
        let warm = matches_locked(root, &hit, 3);
        assert!(cold, "the reinstated pair is supposed to match");
        assert_eq!(warm, cold);
        assert_eq!(warm, crate::interp::matches(root, &hit));
    }

    /// The experiment's first half: what the two bodies look like.
    ///
    /// The numbers pinned here are the ones the module doc's table reports, and
    /// they are pinned as ranges wide enough to survive an optimizer change
    /// that keeps the finding and narrow enough to fail if the finding goes
    /// away. The finding is NOT "the branching body is bigger" — it is smaller,
    /// which is the surprise — it is that the branching body pays in guards.
    #[test]
    fn the_branching_body_trades_ops_for_guards() {
        let s = nonmatching(4096, 20, 42);
        let nodes = count(lower(&bench_regex(20)));

        let sc = measure(lower(&bench_regex(20)), &s);
        let mk = measure_masking(lower(&bench_regex(20)), &s);
        assert_eq!(sc.matched, mk.matched, "the two portals disagreed");
        assert!(!sc.matched, "the benchmark input is supposed NOT to match");

        println!(
            "[shortcircuit] {nodes} nodes, branching: {} loop(s), {:?} bridge(s), \
             {:?} abort(s), {:?} guard failure(s), per character: {}",
            sc.compiles,
            sc.bridges,
            sc.aborts,
            sc.guard_failures,
            census(&sc.body),
        );
        // `REGEX_LISTING=1` prints both peeled bodies op by op, which is what
        // the RPython cross-check reads: `rpython_original/runner.py
        // --listing` prints the same thing for the trace RPython's own
        // optimizer produces, and the two lists are meant to be diffed. A
        // census alone cannot tell "same ops in a different order" from "the
        // same trace".
        if std::env::var("REGEX_LISTING").is_ok() {
            for (label, body) in [("branching", &sc.body), ("masking", &mk.body)] {
                for (i, op) in body.iter().enumerate() {
                    println!("[listing:{label}] {i:4}  {op:?}");
                }
            }
        }
        println!(
            "[shortcircuit] {nodes} nodes, masking  : {} loop(s), {:?} bridge(s), \
             {:?} abort(s), {:?} guard failure(s), per character: {}",
            mk.compiles,
            mk.bridges,
            mk.aborts,
            mk.guard_failures,
            census(&mk.body),
        );

        // Both portals still reach the JIT, and the branching one lowers: a
        // refusal would make every comparison below a comparison of nothing.
        assert!(
            sc.compiles > 0,
            "the branching portal compiled nothing, so every character ran \
             interpreted; degraded={:?}",
            degraded(),
        );
        assert!(mk.compiles > 0, "the masking portal compiled nothing");
        // The control, and the single number this whole comparison rests on:
        // the masking portal leaves its compiled loop ONE time over the same
        // 4096 characters, where the branching portal leaves it 4080. Without
        // this the branching row is just a slow number; with it, the two rows
        // differ by whether the compiled code is what runs.
        let mk_failures = mk.guard_failures.expect("the probe was installed");
        assert!(
            mk_failures * 100 < s.len(),
            "the masking portal deopted {mk_failures} times over {} characters, \
             so it is not the stays-in-compiled-code control the branching \
             portal is being measured against",
            s.len(),
        );
        assert_eq!(
            mk.bridges,
            Some(0),
            "the masking portal grew a bridge, so its guards fail too and it \
             is no longer the control",
        );
        assert_eq!(
            degraded(),
            Vec::new(),
            "the branching `shift` degraded a dispatch arm; with the arm \
             interpreted the body below is not the branching body at all",
        );
        // Not asserted to be zero, because it is not: the branching portal
        // aborts, and the abort is half the mechanism. See the test below.

        // Every node is still in the trace, and the tree walk still folds: the
        // operator moved the guards, not the specialization.
        assert!(
            n_of(&sc.body, OpCode::SetfieldGc) >= nodes,
            "the branching walk stored {} marks for a {nodes}-node tree",
            n_of(&sc.body, OpCode::SetfieldGc),
        );
        assert_eq!(
            n_of(&sc.body, OpCode::GetfieldGcR),
            0,
            "the branching walk reloaded a `left`/`right` pointer",
        );

        // The finding. The masking body carries exactly one guard — the loop
        // exit — and the branching body carries that one plus one per `and`/
        // `or` the optimizer could not prove constant.
        assert_eq!(
            guards(&mk.body),
            1,
            "the masking body is supposed to carry only the loop-exit guard",
        );
        assert!(
            guards(&sc.body) >= 20,
            "the branching body carried {} guards; the whole claim is that \
             short-circuit operators become guards, and under 20 it did not \
             happen",
            guards(&sc.body),
        );
        // And it is SMALLER, not bigger: the dead side of each branch takes
        // its ops with it.
        assert!(
            sc.body.len() < mk.body.len(),
            "the branching body ({} ops) was not smaller than the masking one \
             ({} ops); the doc table says it is",
            sc.body.len(),
            mk.body.len(),
        );
    }

    /// The mechanism behind the timing row, and the half that is free to
    /// assert: the branching body deopts once per input character.
    ///
    /// 26 guards in the peeled body and 4080 failures over 4096 characters is
    /// not "some guards sometimes fail" — it is compiled code leaving through a
    /// guard on essentially every pass. Ten of those failures now grow a
    /// bridge, and the rest still take a blackhole deopt, because a bridge
    /// covers one more path through the mark pattern and the pattern has far
    /// more paths than one 4096-character pass can bridge. That is a
    /// per-character cost measured in microseconds against a compiled body
    /// measured in nanoseconds, and it is why the timing row below is not a
    /// percentage.
    #[test]
    fn the_branching_body_deopts_once_per_character() {
        let s = nonmatching(4096, 20, 42);
        let sc = measure(lower(&bench_regex(20)), &s);
        let failures = sc.guard_failures.expect("the probe was installed");
        println!(
            "[shortcircuit] {} chars: {failures} guard failures, {:?} bridges, \
             {:?} aborts {:?}, {} loop(s)",
            s.len(),
            sc.bridges,
            sc.aborts,
            abort_reasons(),
            sc.compiles,
        );
        assert!(
            failures * 2 > s.len(),
            "only {failures} guard failures over {} characters, so the \
             compiled loop was actually running; the timing row's explanation \
             does not hold",
            s.len(),
        );
        // Bridges ARE built now — 10 of them in this run — and the deopt
        // survives them anyway. Both halves are asserted, because each is a
        // separate way for the finding to move: a run with no bridges at all
        // is the multi-frame resume regressing back to a give-up, and a run
        // whose failures collapse is the bridge tree finally covering the mark
        // pattern, which would make the timing row wrong rather than stale.
        // Why ten and not more is measured next door, in
        // `the_bridge_count_is_set_by_trace_eagerness_not_by_a_ceiling`.
        assert!(
            sc.bridges.unwrap_or(0) > 0,
            "no bridge was compiled. Every guard here fails inside the
             recursive `shift` helper, so its resume data spans 4 to 7 frames;
             a state-field driver that cannot rebuild them all declines the
             bridge, which is where this portal used to sit",
        );
        assert_eq!(
            sc.aborts,
            Some(0),
            "a bridge attempt was abandoned; `MAJIT_BRIDGE_DEBUG=1` names the \
             decline on the `[bridgeB] DECLINE` line",
        );
    }

    /// This portal reserves no identity-only register, which is why its
    /// bridges exist at all.
    ///
    /// A state that reserves identity slots cannot be bridged across inline
    /// frames: the snapshot trim blanks that range on every non-root frame and
    /// nothing re-derives it, so `bridge_from_guard_resume_position` refuses
    /// and the blackhole serves the guard. This portal fails its guards 4 to 7
    /// frames deep, so if it ever started reserving slots every bridge count
    /// next door would silently drop to zero and the finding would read as a
    /// regression in the fix rather than in the declaration. Assert the
    /// premise instead of inferring it from the counts.
    #[test]
    fn this_portal_reserves_no_identity_slots_so_the_decline_is_inert() {
        assert_eq!(
            <ShortCircuitState as majit_metainterp::JitState>::reserved_int_identity_range(),
            None,
            "the branching portal now reserves identity slots, so every \
             multi-frame bridge is refused and the bridge counts in this \
             module are measuring a different thing",
        );
    }

    /// How many bridges a pass grows, and what sets that number.
    ///
    /// The finding next door is that ten bridges do not move the deopt rate.
    /// The obvious reading — "the bridge tree saturates" — is wrong, and this
    /// is the measurement that says so: the count is not a ceiling, it is a
    /// rate. `trace_eagerness` is 200 (`rlib/jit.py`'s own default, carried in
    /// `PARAMETERS`), so ONE guard has to fail 200 times before it is worth a
    /// bridge, and a pass of `n` characters can therefore grow at most about
    /// `n / 200` of them however many distinct mark patterns the tree has.
    ///
    /// Eight times the input is what separates a rate from a ceiling, so that
    /// is what this runs. A saturating tree would answer the same count twice.
    #[test]
    fn the_bridge_count_is_set_by_trace_eagerness_not_by_a_ceiling() {
        let short = nonmatching(4096, 20, 42);
        let long = nonmatching(4096 * 8, 20, 42);
        let a = measure(lower(&bench_regex(20)), &short);
        let b = measure(lower(&bench_regex(20)), &long);
        let (ab, bb) = (a.bridges.unwrap_or(0), b.bridges.unwrap_or(0));
        println!(
            "[shortcircuit] bridges: {} chars -> {ab}, {} chars -> {bb}; \
             guard failures {:?} -> {:?}",
            short.len(),
            long.len(),
            a.guard_failures,
            b.guard_failures,
        );
        assert!(
            bb > ab,
            "8x the input grew no more bridges ({ab} -> {bb}), so the count is \
             a ceiling after all and the module doc's explanation is wrong",
        );
    }

    /// The experiment's second half: what the two bodies cost.
    ///
    /// Short input and a loose floor, because this one runs in the normal
    /// suite. The doc table's figure — median of 5 over 1048576 characters —
    /// comes from [`the_branching_body_is_timed_at_the_full_length`], which is
    /// `#[ignore]`d because at the branching portal's rate a single pass over
    /// 1M characters takes over a minute.
    ///
    /// The ratio here is much smaller than the doc table's, and the reason is
    /// worth stating rather than smoothing over: `matches` builds a fresh
    /// `JitDriver` per call, so every call re-records and re-compiles, and at
    /// 4096 characters that one-off recording is most of the masking row's
    /// wall clock. The branching row has far less to amortize it against — ten
    /// bridges that each serve one path, against a compiled loop that serves
    /// every character. Longer input therefore moves the two rows in opposite
    /// directions, and the ratio grows with length — which is itself the shape
    /// of the finding.
    #[test]
    fn the_branching_body_is_far_slower_per_character() {
        let s = nonmatching(1 << 12, 20, 42);
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let sc = time_rate(lower(&bench_regex(20)), &s, 3, |r, s| matches(r, s, 3));
        let mk = time_rate(lower(&bench_regex(20)), &s, 3, |r, s| {
            crate::jit_interp::matches(r, s, 3)
        });
        let ratio = median(&mk) / median(&sc);
        println!(
            "[shortcircuit] {} chars, branching: {:.0} chars/s, masking: \
             {:.0} chars/s, masking / branching = {ratio:.0}x",
            s.len(),
            median(&sc),
            median(&mk),
        );
        // Measured at this length: 66x, 68x and 89x on dynasm BEFORE the
        // multi-frame bridge fix, 19x on cranelift; 32x, 35x, 50x and 51x on
        // dynasm after it, at 1-minute load 33 to 36. The backends differ here
        // and not in the op census because at 4096 characters this row is
        // dominated by the ONE recording each `matches` call pays for, and
        // cranelift compiles slower than dynasm. The floor stays 10x: low
        // enough to clear cranelift under contention and to survive further
        // bridging, and still far above the ~1x a lost finding would read.
        assert!(
            ratio > 10.0,
            "masking was only {ratio:.1}x the branching rate at {} characters. \
             The module doc reports 32-51x here on dynasm since the \
             multi-frame bridge fix, 19x on cranelift before it and 1716x at \
             1M, and under 10x that table is wrong rather than stale",
            s.len(),
        );
    }

    /// The doc table's timing row: median of [`TIMED_RUNS`] over
    /// [`TIMED_CHARS`] characters, one untimed warm-up first so the timed runs
    /// measure compiled code and not the trace it had to record.
    ///
    /// Median rather than mean because benchmark noise is one-sided: a
    /// preemption, an interrupt or a thermal step can only make a run slower,
    /// so a mean reports a run that never happened and a rerun does not
    /// reproduce it.
    ///
    /// `#[ignore]`d: at 10k chars/s the branching portal needs about 100
    /// seconds per pass, six of them counting the warm-up, and the two fast
    /// rows add a second. Measured end to end at 547s and 694s, on a machine
    /// at 1-minute load 3 and 34 — which is most of the difference between
    /// those two, and is why the doc table stamps the load on every column.
    /// `cargo test -p regex --release --no-default-features --features dynasm \
    /// -- --ignored --nocapture the_branching_body_is_timed_at_the_full_length`
    #[test]
    #[ignore = "~10 minutes: the branching portal scans 1M characters in ~100s"]
    fn the_branching_body_is_timed_at_the_full_length() {
        let s = nonmatching(TIMED_CHARS, 20, 42);
        let _guard = PROBE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let sc = time_rate(lower(&bench_regex(20)), &s, TIMED_RUNS, |r, s| {
            matches(r, s, 3)
        });
        let mk = time_rate(lower(&bench_regex(20)), &s, TIMED_RUNS, |r, s| {
            crate::jit_interp::matches(r, s, 3)
        });
        // The third row is the one that turns "slower" into a verdict: the same
        // algorithm with no JIT under it at all.
        let plain = time_rate(lower(&bench_regex(20)), &s, TIMED_RUNS, |r, s| {
            crate::interp::matches(r, s)
        });
        println!(
            "[shortcircuit] {TIMED_CHARS} chars, branching: {:.0} chars/s \
             (min {:.0}, max {:.0})",
            median(&sc),
            sc[0],
            sc[sc.len() - 1],
        );
        println!(
            "[shortcircuit] {TIMED_CHARS} chars, masking  : {:.0} chars/s \
             (min {:.0}, max {:.0})",
            median(&mk),
            mk[0],
            mk[mk.len() - 1],
        );
        println!(
            "[shortcircuit] {TIMED_CHARS} chars, no JIT    : {:.0} chars/s \
             (min {:.0}, max {:.0})",
            median(&plain),
            plain[0],
            plain[plain.len() - 1],
        );
        println!(
            "[shortcircuit] masking / branching = {:.0}x, \
             no-JIT / branching = {:.0}x",
            median(&mk) / median(&sc),
            median(&plain) / median(&sc),
        );
    }

    const TIMED_CHARS: usize = 1 << 20;
    const TIMED_RUNS: usize = 5;

    /// chars/s per timed run, ascending. The untimed first pass takes the page
    /// faults and the cold branch predictors, which are not what a row is about.
    fn time_rate(
        root: *mut NodeRec,
        s: &[u8],
        runs: usize,
        run: impl Fn(*mut NodeRec, &[u8]) -> bool,
    ) -> Vec<f64> {
        assert!(
            !run(root, s),
            "the benchmark input is supposed NOT to match"
        );
        let mut rates = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t0 = Instant::now();
            let hit = run(root, s);
            let secs = t0.elapsed().as_secs_f64();
            assert!(!hit, "the benchmark input is supposed NOT to match");
            rates.push(s.len() as f64 / secs);
        }
        rates.sort_by(f64::total_cmp);
        rates
    }

    fn median(rates: &[f64]) -> f64 {
        rates[rates.len() / 2]
    }
}
