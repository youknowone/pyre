//! Regression guard for polymorphic per-pc liveness.
//!
//! Background: B.2 ports `rpython/jit/codewriter/liveness.py:33-79
//! _compute_liveness_must_continue` (backward dataflow over `op_metadata`),
//! B.3 registers each per-marker `(live_i, live_r, live_f)` triple via
//! `Assembler::_register_liveness_offset`, and B.4 patches the BC_LIVE
//! 2-byte slot per marker via `JitCodeBuilder::finalize_liveness`.
//!
//! Before B.3-B.4 every BC_LIVE marker in macro-lowered JitCodes pointed
//! at the canonical "everything-alive" entry at offset 0 — over-approximating
//! `frame.fail_args` during blackhole resume.  This test pins the post-B.3-B.4
//! invariants by driving a synthetic `#[jit_interp]` consumer whose arms have
//! intentionally-different op chains before each guard, so the lowerer's
//! per-marker analysis must produce distinct live sets per arm.
//!
//! Hard regression invariants enforced:
//! 1. `__prebuild_jitcode_liveness_*` registers more than just the canonical
//!    entry into the shared `Assembler` (proving the macro pipeline is
//!    actually running per-marker analysis, not collapsing every arm to the
//!    canonical set).
//! 2. `__dispatch_jitcode_*` builder calls AFTER `install_canonical_liveness`
//!    must not grow `asm.all_liveness()` — every per-marker triple (both
//!    dispatch-body and per-arm sub-builder) must already be registered by
//!    prebuild.  Mirrors the runtime assertion at `codegen_trace.rs:178-185`.
//! 3. Every lowerable arm's sub-JitCode embedded under the dispatch
//!    JitCode (`BC_INLINE_CALL` target, jitcode_lower/dispatch.rs:1860-1872)
//!    must emit at least one per-pc BC_LIVE marker past the leading
//!    canonical, and the union of those per-pc offsets across all lowerable
//!    arms must contain at least two distinct values (the polymorphism
//!    check).  An arm that fell to `__sub_builder.abort()` at
//!    `jitcode_lower/dispatch.rs:1874` would produce a sub-JitCode with
//!    zero BC_LIVE markers — the per-arm assertion catches that regression
//!    even when the remaining arms still emit ≥2 distinct offsets between
//!    them.

use majit_metainterp::jitcode::insns::BC_LIVE;
use majit_metainterp::{Assembler, JitCode, JitDriver, JitState as _};

struct Polymorphic4State {
    a: i64,
    b: i64,
    c: i64,
    d: i64,
}

const OP_GUARD_A: u8 = 1;
const OP_SUM_AB: u8 = 2;
const OP_SUM_ABC: u8 = 3;
const OP_SUM_ABCD: u8 = 4;
const OP_END: u8 = 0;

pub type Bytecode = [u8];

// `BytecodeExt::get_op` is consumed by the macro-emitted `__trace_*` fn
// (codegen_trace.rs:61 `program.get_op(pc)`), but the integration tests
// below drive only `__dispatch_jitcode_*` / `__prebuild_*`, so the compiler
// flags the trait as dead code in this binary's reachability graph.
#[allow(dead_code)]
trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}

impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

// Each arm drives a guard with a deliberately-different number of
// `load_state_field` + `record_binop_i` ops upstream, so the backward
// walker (`compute_per_marker_liveness`) sees distinct register
// allocations live at each arm's BC_LIVE marker.
//
// arm_GUARD_A:  R0 = a;                 marker; goto_if_not(R0)            → live={R0}
// arm_SUM_AB:   R0 = a; R1 = b; R2 = R0+R1; marker; goto_if_not(R2)        → live={R2}
// arm_SUM_ABC:  R0..R4 chain;            marker; goto_if_not(R4)            → live={R4}
// arm_SUM_ABCD: R0..R6 chain;            marker; goto_if_not(R6)            → live={R6}
//
// Encoded as `live_i` byte payloads, those four sets live at distinct
// offsets in `asm.all_liveness` (each register-index lands in a different
// bitmap byte position).
#[majit_macros::jit_interp(
    state = Polymorphic4State,
    env = Bytecode,
    state_fields = {
        a: int,
        b: int,
        c: int,
        d: int,
    },
    greens = [],
)]
#[allow(unused_assignments, unused_variables)]
fn polymorphic_mainloop(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<Polymorphic4State> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = Polymorphic4State {
        a: 1,
        b: 1,
        c: 1,
        d: 1,
    };

    {
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_GUARD_A => {
                if state.a != 0 {
                    state.a += 1;
                }
            }
            OP_SUM_AB => {
                if state.a + state.b != 0 {
                    state.a += 1;
                }
            }
            OP_SUM_ABC => {
                if state.a + state.b + state.c != 0 {
                    state.a += 1;
                }
            }
            OP_SUM_ABCD => {
                if state.a + state.b + state.c + state.d != 0 {
                    state.a += 1;
                }
            }
            _ => break,
        }
    }
    state.a
}

/// Walk a JitCode body and collect every BC_LIVE marker's 2-byte offset.
fn collect_bc_live_offsets(jitcode: &JitCode) -> Vec<u16> {
    let code = &jitcode.code;
    let mut offsets = Vec::new();
    let mut i = 0;
    while i < code.len() {
        if code[i] == BC_LIVE {
            // BC_LIVE is followed by a 2-byte little-endian offset into
            // `Assembler::all_liveness` (assembler.py:248 `encode_offset`).
            assert!(
                i + 2 < code.len(),
                "BC_LIVE at end of code without 2-byte offset payload"
            );
            offsets.push(u16::from_le_bytes([code[i + 1], code[i + 2]]));
            i += 3;
        } else {
            i += 1;
        }
    }
    offsets
}

/// Mirrors production ordering: stage canonical liveness, prebuild per-marker
/// entries, then bind the canonical entry after the IR walk.
fn install_canonical_for_test(asm: &mut Assembler, canonical: &[u8]) {
    asm.set_canonical_liveness_triple(canonical.to_vec(), Vec::new(), Vec::new());
    __prebuild_jitcode_liveness_polymorphic_mainloop(asm);
    let _ = asm.ensure_canonical_liveness_offset();
}

#[test]
fn prebuild_registers_more_than_canonical_entry() {
    // Drive the macro-emitted prebuild against a fresh Assembler. With four
    // arms whose register layouts at the BC_LIVE marker differ, the prebuild
    // must register at least one per-marker triple distinct from the
    // canonical `[0,1,2,3]` entry — otherwise the per-pc walker collapsed
    // every arm onto the canonical set (B.2 walker regression).
    let mut asm = Assembler::new();
    let canonical: Vec<u8> = (0..4u8).collect();
    // Stage the canonical triple lazily — production's
    // `install_canonical_liveness` does not register it up front; the
    // prebuild's per-marker triples land first.
    asm.set_canonical_liveness_triple(canonical.clone(), Vec::new(), Vec::new());
    let pre_prebuild_len = asm.all_liveness().len();

    __prebuild_jitcode_liveness_polymorphic_mainloop(&mut asm);

    let post_prebuild_len = asm.all_liveness().len();
    assert!(
        post_prebuild_len > pre_prebuild_len,
        "prebuild must register at least one per-marker triple \
         (pre_prebuild_len={pre_prebuild_len}, post_prebuild_len={post_prebuild_len})"
    );
}

#[test]
fn factory_does_not_grow_asm_after_prebuild() {
    // Mirror the runtime assertion at `codegen_trace.rs:178-185`: every
    // per-marker triple emitted by the dispatch JitCode builder (and its
    // per-arm sub-builders embedded via `BC_INLINE_CALL`) must already be
    // in `asm.all_liveness` (prebuild hit), so `finalize_liveness` only
    // dedups against existing offsets.  A regression in the prebuild walker
    // (e.g., missing emit-site coverage) would surface as growth here.
    let mut asm = Assembler::new();
    let canonical: Vec<u8> = (0..4u8).collect();
    install_canonical_for_test(&mut asm, &canonical);

    let post_install_len = asm.all_liveness().len();

    let _dispatch = __dispatch_jitcode_polymorphic_mainloop(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture");

    assert_eq!(
        asm.all_liveness().len(),
        post_install_len,
        "dispatch JitCode build must not grow all_liveness past prebuild snapshot"
    );
}

#[test]
fn distinct_arms_emit_distinct_bc_live_offsets() {
    // The polymorphism core has two layers:
    //
    // (1) Per-arm survival: every lowerable arm's sub-JitCode embedded
    //     under the dispatch JitCode (`BC_INLINE_CALL` target,
    //     jitcode_lower/dispatch.rs:1860-1872) must emit at least one
    //     per-pc BC_LIVE marker past the leading canonical.  If the
    //     dispatch arm lowerer fell to `__sub_builder.abort()` at
    //     `jitcode_lower/dispatch.rs:1874` for one of the four
    //     `SUM`/`GUARD_A` arms, the resulting sub-JitCode body is a
    //     single `BC_ABORT` byte with zero BC_LIVE markers — the
    //     per-arm assertion below catches that regression.  The
    //     Polymorphic4State fixture has four lowerable arms, so at
    //     least four sub-JitCodes must satisfy this guarantee.
    //
    // (2) Polymorphism: the union of all per-pc offsets across the
    //     lowerable sub-JitCodes must contain at least two distinct
    //     values.  A regression to "every arm uses the canonical
    //     triple" would collapse this set to size 1.
    //
    // The dispatch JitCode body itself emits only the canonical
    // `live_placeholder()` markers (jit_merge_point pre/post + the
    // trailing -live- after each inline_call_*), so the per-pc
    // distinctness signal lives in the embedded sub-JitCodes — one
    // per arm — rather than the parent dispatch body.
    let mut asm = Assembler::new();
    let canonical: Vec<u8> = (0..4u8).collect();
    install_canonical_for_test(&mut asm, &canonical);

    let dispatch = __dispatch_jitcode_polymorphic_mainloop(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture");

    // Identify lowerable arm sub-JitCodes by the leading
    // `__sub_builder.live_placeholder()` (jitcode_lower/dispatch.rs:1863):
    // a lowerable sub-JitCode body starts with BC_LIVE at offset 0.
    // Halt/Nop arms produce an empty body (no BC_LIVE); the
    // abort-fallback path at :1874 produces a single BC_ABORT byte
    // (also no BC_LIVE) — both shapes collapse to "no BC_LIVE markers"
    // and are filtered out here.  An abort-fallback regression on a
    // *lowerable* arm therefore reduces the count of qualifying
    // sub-JitCodes, which the lowerable-count assertion below catches.
    let mut per_arm_offsets: Vec<Vec<u16>> = Vec::new();
    for d in dispatch.exec.descrs.iter() {
        if let majit_metainterp::jitcode::RuntimeBhDescr::JitCode(jc) = d {
            let offs = collect_bc_live_offsets(jc);
            if offs.is_empty() {
                // Halt/Nop arm (empty body) or abort-fallback (`BC_ABORT`
                // only).  Skip — not a lowerable arm sub-JitCode.
                continue;
            }
            // The first offset is the canonical marker; the rest are per-pc
            // liveness markers emitted inside the arm body.
            let per_pc: Vec<u16> = offs.into_iter().skip(1).collect();
            assert!(
                !per_pc.is_empty(),
                "lowerable arm sub-JitCode must emit at least one per-pc \
                 BC_LIVE marker past the leading canonical; got none — \
                 the dispatch arm lowerer at \
                 jitcode_lower/dispatch.rs:1874 may have dropped a real \
                 per-pc -live- entry, or the body emitted only the \
                 leading `live_placeholder()`"
            );
            per_arm_offsets.push(per_pc);
        }
    }

    // Polymorphic4State fixture: four lowerable arms (OP_GUARD_A /
    // OP_SUM_AB / OP_SUM_ABC / OP_SUM_ABCD).  Fewer than four
    // qualifying sub-JitCodes means at least one lowerable arm fell to
    // the abort-fallback at jitcode_lower/dispatch.rs:1874.
    assert!(
        per_arm_offsets.len() >= 4,
        "expected at least 4 lowerable arm sub-JitCodes (one per \
         OP_GUARD_A/OP_SUM_AB/OP_SUM_ABC/OP_SUM_ABCD); got {} — the \
         dispatch arm lowerer dropped one or more lowerable bodies to \
         `__sub_builder.abort()` at jitcode_lower/dispatch.rs:1874",
        per_arm_offsets.len()
    );

    // Polymorphism: the union of all per-pc marker offsets across the
    // lowerable sub-JitCodes must contain at least two distinct values.
    let all_offsets: std::collections::BTreeSet<u16> = per_arm_offsets
        .iter()
        .flat_map(|offs| offs.iter().copied())
        .collect();
    assert!(
        all_offsets.len() >= 2,
        "polymorphic per-pc liveness regressed: every per-pc BC_LIVE marker \
         across embedded sub-JitCodes points at the same offset — saw \
         {all_offsets:?} across {} arms",
        per_arm_offsets.len()
    );
}

#[test]
fn install_canonical_liveness_registers_dispatch_jitcode_singleton() {
    let mut driver: JitDriver<Polymorphic4State> = JitDriver::new(100);
    let state = Polymorphic4State {
        a: 1,
        b: 1,
        c: 1,
        d: 1,
    };
    let program = [OP_GUARD_A, OP_END];
    state
        .build_meta(0, &program)
        .install_canonical_liveness(&mut driver);
    let stored = driver.dispatch_jitcode();
    assert!(
        stored.is_some(),
        "install_canonical_liveness must register dispatch JitCode singleton"
    );
    let jc = stored.unwrap();
    assert!(
        !jc.code.is_empty(),
        "registered dispatch JitCode body must be non-empty"
    );
    let live_offsets = collect_bc_live_offsets(jc);
    assert!(
        !live_offsets.is_empty(),
        "registered dispatch JitCode must contain at least one BC_LIVE marker"
    );
}

/// Regression pin: `BC_INLINE_CALL` targets in the dispatch
/// JitCode descr table must be `RuntimeBhDescr::JitCode` (frame-chain
/// interpreter path), never fnaddr handlers (which target native call
/// wrappers). Production runtime enforcement at
/// `pyjitpl/dispatch.rs:1690-1692` panics if `descrs[sub_idx].as_jitcode()`
/// is `None`; this build-time test pins the lowerer's emit invariant so
/// accidental migration to a fnaddr emit path surfaces at compile + test
/// time, not at first dispatch. RPython parity: `blackhole.py:150-157`
/// argcode `j` resolves via `self.descrs[idx]` asserted to be a `JitCode`.
#[test]
fn dispatch_inline_call_descrs_have_jitcode_entries() {
    let mut asm = Assembler::new();
    let canonical: Vec<u8> = (0..4u8).collect();
    install_canonical_for_test(&mut asm, &canonical);

    let dispatch = __dispatch_jitcode_polymorphic_mainloop(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture");

    let mut jitcode_count = 0;
    for d in dispatch.exec.descrs.iter() {
        if let majit_metainterp::jitcode::RuntimeBhDescr::JitCode(jc) = d {
            jitcode_count += 1;
            // Each sub-jitcode body is at least 1 byte. Cross-scope arms
            // fall back to a single abort byte (`jitcode_lower.rs:5170-5202`),
            // which is intentional — but a zero-byte body would mean the
            // build pipeline silently dropped the sub-jitcode entirely.
            assert!(
                !jc.code.is_empty(),
                "BC_INLINE_CALL target sub-jitcode body must be non-empty"
            );
        }
    }
    assert!(
        jitcode_count >= 1,
        "dispatch must emit at least one BC_INLINE_CALL JitCode descr; \
         a fnaddr-only descr table would mean the lowerer wired native \
         call targets instead of frame-chain sub-jitcodes, regressing \
         the dispatch.rs:1690-1692 contract"
    );
}
