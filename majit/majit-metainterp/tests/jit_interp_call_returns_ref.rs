//! `call_returns` — the declaration that types a ref-returning call's result.
//!
//! A call whose result is a ref produces a ref binding, and `result.field`
//! after it needs a struct to resolve the field against.
//! `call_returns = { func => Struct }` is that declaration, and
//! `lower_ref_binding_getfield` bails without it — so the field read does not
//! lower, and a statement that does not lower takes its whole dispatch arm
//! down with it.
//!
//! The concrete path reads the same declaration through `RefFieldRewriter`, so
//! a degraded arm still computes the right answer. Only the arm census can say
//! it degraded, which is why this file grades that and not the return value.
//!
//! Two arms, one difference: `OP_PLAIN` reaches the callee through
//! `residual_ref` and `OP_WRAPPED` through `residual_ref_wrapped`. The
//! non-wrapped arm returns early from `lower_call_value` to attach the
//! declared type; every wrapped policy falls through to the shared tail, which
//! is where the type has to be attached for all of them.
//!
//! No fixture in this corpus used `call_returns`, or any ref call policy,
//! before this one.

use majit_metainterp::jitcode::insns::BC_GETFIELD_GC_I;
use majit_metainterp::{Assembler, JitDriver};

/// Two int fields and a self-edge: `value` is what the arms read back through
/// the returned ref, `next` is what the callees walk.
#[repr(C)]
struct Cell {
    value: i64,
    next: *mut Cell,
}

/// The control's callee. `residual_ref` bakes the function address directly,
/// so this one needs no policy attribute.
fn plain_next(node: usize) -> *mut Cell {
    // SAFETY: the fixture hands over a `Cell` it owns for the whole run.
    unsafe { (*(node as *mut Cell)).next }
}

/// The subject's callee. `#[dont_look_inside]` emits the
/// `__majit_call_policy_*` accessor and the `extern "C"` call-target wrappers
/// that every `*_wrapped` policy resolves; a raw-pointer return is what makes
/// it the ref member of that family.
#[majit_macros::dont_look_inside]
fn wrapped_next(node: usize) -> *mut Cell {
    // SAFETY: as above.
    unsafe { (*(node as *mut Cell)).next }
}

struct CellState {
    total: i64,
    head: usize,
}

pub type Bytecode = [u8];

const OP_PLAIN: u8 = 1;
const OP_WRAPPED: u8 = 2;

#[majit_macros::jit_interp(
    state = CellState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { total: int, head: ref(Cell) },
    ref_fields = { Cell::next => Cell },
    int_fields = { Cell::value => i64 },
    calls = {
        plain_next => residual_ref,
        wrapped_next => residual_ref_wrapped,
    },
    call_returns = {
        plain_next => Cell,
        wrapped_next => Cell,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_cells(program: &Bytecode, threshold: u32, head: usize) -> i64 {
    let mut driver: JitDriver<CellState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = CellState { total: 0, head };
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
            // The control: the arm that has always read `call_returns`.
            OP_PLAIN => {
                let n = plain_next(state.head);
                state.total = state.total + n.value;
            }
            // The subject: the same shape through the wrapped policy.
            OP_WRAPPED => {
                let n = wrapped_next(state.head);
                state.total = state.total + n.value;
            }
            _ => break,
        }
    }
    state.total
}

fn build_dispatch_jitcode() -> majit_metainterp::JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_dispatch_cells(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_cells(&mut asm, 0i64).expect("dispatch lower must succeed")
}

/// Both arms lower. The control says the fixture is aimed right; the subject
/// is the one that used to degrade.
#[test]
fn a_declared_ref_return_types_its_result_through_a_wrapped_policy() {
    let _ = build_dispatch_jitcode();

    let census = majit_metainterp::dispatch_arm_census();
    let counted = census
        .iter()
        .find(|e| e.interp == "CellState")
        .unwrap_or_else(|| panic!("the portal must record its arm count; census={census:?}"));
    assert_eq!(
        counted.arms, 2,
        "OP_PLAIN and OP_WRAPPED are counted; the `_` default is not; \
         census={census:?}"
    );

    let degraded: Vec<&'static str> = majit_metainterp::degraded_dispatch_arms()
        .into_iter()
        .filter(|e| e.interp == "CellState")
        .map(|e| e.arm)
        .collect();
    assert!(
        !degraded.contains(&"OP_PLAIN"),
        "the control degraded, so the fixture is measuring its own mistake and \
         the subject below says nothing; degraded={degraded:?}"
    );

    majit_metainterp::assert_no_degraded_dispatch_arms("CellState");
}

/// Every jitcode body the portal produced: the dispatch JitCode plus each
/// per-arm sub-JitCode.
fn all_jitcode_bodies(dispatch_jc: &majit_metainterp::JitCode) -> Vec<Vec<u8>> {
    let mut bodies = vec![dispatch_jc.code.clone()];
    bodies.extend(
        dispatch_jc
            .exec
            .descrs
            .iter()
            .filter_map(|descr| descr.as_jitcode())
            .map(|sub| sub.code.clone()),
    );
    bodies
}

/// What an un-degraded arm was supposed to emit. The gate above says the arm
/// lowered; this says what it lowered TO, so a lowering that dropped the field
/// read and still reported a whole arm would not pass.
#[test]
fn each_arm_reads_the_field_through_a_getfield_gc() {
    let dispatch_jc = build_dispatch_jitcode();
    let reads: usize = all_jitcode_bodies(&dispatch_jc)
        .iter()
        .map(|body| body.iter().filter(|op| **op == BC_GETFIELD_GC_I).count())
        .sum();
    assert_eq!(
        reads, 2,
        "one `n.value` per arm; 1 is the wrapped arm degraded to a stub",
    );
}

/// The answer. Both arms follow `head.next` and add the `value` they find
/// there, so the sum names which field was read: 14 is `next.value` twice, and
/// 2 would be `head.value` — the same shape misread one edge short.
#[test]
fn both_policies_read_the_field_behind_the_returned_ref() {
    let mut tail = Cell {
        value: 7,
        next: std::ptr::null_mut(),
    };
    let mut head = Cell {
        value: 1,
        next: &mut tail as *mut Cell,
    };
    let head_addr = &mut head as *mut Cell as usize;

    let program: &Bytecode = &[OP_PLAIN, OP_WRAPPED];
    assert_eq!(
        dispatch_cells(program, u32::MAX, head_addr),
        14,
        "each arm adds `head.next.value`",
    );
}
