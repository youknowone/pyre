//! A dispatch arm may call a result-returning inline helper for its effect.
//!
//! `inline_int` / `inline_ref` / `inline_float` helpers were lowered in value
//! position only. In statement position — the result discarded — they reached
//! the trailing `_ => return None` of `lower_config_call_stmt`, and a statement
//! that does not lower takes its whole arm with it: the arm becomes an abort
//! stub and every trace that reaches the opcode aborts, forever.
//!
//! Two things made that hard to see. `explicit_call_emits_post_live` already
//! answers for these three kinds, so the accounting said the policy was handled
//! while the lowering did not handle it. And the workaround is invisible in the
//! result — binding the value to a `let _x` the arm never reads makes the same
//! source lower, so a machine that hit this looks like one that never did.
//!
//! `#[jit_inline]` fails the build on the same input
//! (`jit_interp_inline_helper_typed_return.rs` pins that half). `#[jit_interp]`
//! does not: it degrades the arm and keeps going, which is why this file
//! grades the registry rather than the compiler.

use majit_macros::jit_inline;
use majit_metainterp::{Assembler, JitDriver};

#[repr(C)]
struct PopNode {
    value: i64,
    next: *mut PopNode,
}

#[repr(C)]
struct PopStack {
    head: *mut PopNode,
    size: i64,
}

/// The subject helper: it RETURNS the popped value and MUTATES the stack, so a
/// caller that wants only the mutation has no reason to bind the result.
#[jit_inline(
    ref_params = {
        stack: ref(PopStack),
    },
    ref_fields = {
        PopStack::head => PopNode,
        PopNode::next => PopNode,
    },
)]
fn pop_stack(stack: usize) -> i64 {
    let head = stack.head;
    let value = head.value;
    let next = head.next;
    stack.head = next;
    stack.size = stack.size - 1i64;
    value
}

struct PopState {
    total: i64,
    sel: usize,
}

pub type Bytecode = [u8];

/// The subject: the helper's result is discarded.
const OP_DISCARD_POP: u8 = 1;
/// The control: the same helper, same arguments, result bound. This is the
/// spelling that already lowered, and it is what the workaround turns the
/// subject into — so it must stay green for the subject's failure to mean
/// anything.
const OP_BIND_POP: u8 = 2;
/// The denominator. An empty degraded registry means either "no arm degraded"
/// or "no arm was looked at", and only the first is a pass. This arm is
/// unlowerable on purpose (a `break` enclosed in an `if`), so its presence in
/// the registry proves the registry was written to at all.
const OP_ENCLOSED_BREAK: u8 = 3;
const OP_BACK: u8 = 4;

#[majit_macros::jit_interp(
    state = PopState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { total: int, sel: ref(PopStack) },
    calls = { pop_stack => inline_int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_pop(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<PopState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = PopState { total: 0, sel: 0 };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_DISCARD_POP => {
                pop_stack(state.sel);
            }
            OP_BIND_POP => {
                let popped = pop_stack(state.sel);
                state.total = state.total + popped;
            }
            OP_ENCLOSED_BREAK => {
                if state.total == 0 {
                    break;
                }
            }
            OP_BACK => {
                if state.total != 0 {
                    pc = 0;
                    continue;
                }
            }
            _ => break,
        }
    }
    state.total
}

/// The dispatch portal, plus the helper's own jitcode body built off the SAME
/// assembler. Sub-JitCodes carry no name, so the helper's byte body is what
/// identifies its splices in the portal's descr list; building it from a second
/// assembler would give it different liveness offsets and match nothing.
fn install() -> (majit_metainterp::JitCode, Vec<u8>) {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_dispatch_pop(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch_jc =
        __dispatch_jitcode_dispatch_pop(&mut asm, 0i64).expect("dispatch lower must succeed");
    let helper_code = __majit_inline_jitcode_pop_stack_with_asm(&mut asm)
        .code
        .clone();
    (dispatch_jc, helper_code)
}

fn degraded_arms() -> Vec<String> {
    let _ = install();
    majit_metainterp::degraded_dispatch_arms()
        .into_iter()
        .filter(|e| e.interp == "PopState")
        .map(|e| e.arm.to_string())
        .collect()
}

/// The subject. Read the denominator first: without the control arm in the
/// registry, "the subject is absent" is what a registry nobody wrote to also
/// looks like.
#[test]
fn a_discarded_inline_int_call_does_not_degrade_its_arm() {
    let degraded = degraded_arms();

    assert!(
        degraded.iter().any(|arm| arm == "OP_ENCLOSED_BREAK"),
        "denominator: the deliberately-unlowerable arm must be recorded, or an \
         empty registry is indistinguishable from an unread one; degraded={degraded:?}"
    );
    assert!(
        !degraded.iter().any(|arm| arm == "OP_DISCARD_POP"),
        "an `inline_int` helper called for its effect degraded its arm to an \
         abort stub. Every trace reaching that opcode aborts and the storage \
         mutation never enters the trace; degraded={degraded:?}"
    );
    assert!(
        !degraded.iter().any(|arm| arm == "OP_BIND_POP"),
        "control: binding the result has always lowered. If this fails the \
         helper itself stopped lowering and the subject above is measuring \
         something else; degraded={degraded:?}"
    );
}

/// …and the arm must carry the splice, not merely avoid degrading. An arm that
/// lowered while silently dropping the call would pass the test above.
///
/// The oracle is the portal's sub-JitCode list — one `add_sub_jitcode` per
/// splice site — not a byte scan of the bodies. A jitcode body is a bytecode
/// stream whose operands are bytes too, so `code.contains(&BC_INLINE_CALL)`
/// answers about any operand that happens to equal that opcode: counting
/// `BC_INLINE_CALL` bytes here returns 2 even with the discarding arm degraded
/// to `[BC_ABORT]`, which is exactly the reading this test exists to reject.
#[test]
fn the_discarded_call_is_spliced_into_a_jitcode_body() {
    let (dispatch_jc, helper_code) = install();
    let sub_bodies: Vec<&[u8]> = dispatch_jc
        .exec
        .descrs
        .iter()
        .filter_map(|descr| descr.as_jitcode())
        .map(|sub| sub.code.as_slice())
        .collect();
    let helper_splices = sub_bodies
        .iter()
        .filter(|body| **body == helper_code.as_slice())
        .count();
    assert_eq!(
        helper_splices, 2,
        "exactly two arms call `pop_stack` — the discarding one and the \
         binding one — so the portal must register the helper twice. One \
         registration means the discarding arm dropped its call; \
         helper={helper_code:?} sub-bodies={sub_bodies:?}"
    );
}

/// The concrete path must perform the same mutation the trace now carries.
#[test]
fn the_discarded_call_still_pops_on_the_concrete_path() {
    let mut second = PopNode {
        value: 22,
        next: std::ptr::null_mut(),
    };
    let mut first = PopNode {
        value: 11,
        next: &mut second,
    };
    let mut stack = PopStack {
        head: &mut first,
        size: 2,
    };

    pop_stack(&mut stack as *mut PopStack as usize);

    assert_eq!(
        stack.size, 1,
        "the discarded pop must still shrink the stack"
    );
    assert_eq!(
        unsafe { (*stack.head).value },
        22,
        "the discarded pop must still advance head past the popped node"
    );
}
