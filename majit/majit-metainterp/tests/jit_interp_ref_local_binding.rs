//! Binding a `ref(T)` state field to a local before reading a field off it.
//!
//! `state.sel.value` lowers as one expression, because that path resolves the
//! pointee `T` out of the `ref(T)` declaration itself. Split across two
//! statements the ref becomes an ordinary local binding, and both walks then
//! have to find the same declaration through that binding. Neither did:
//! `RefFieldRewriter` never recorded the local, so the concrete arm did not
//! compile (`usize` has no fields), and `lower_ref_binding_getfield` found no
//! struct type on the binding, so the arm degraded to a stub.
//!
//! The two arms below differ only in that split, and once the concrete side
//! compiles they agree on the answer — which is why the arm census is what
//! grades the lowering here.

use majit_metainterp::jitcode::insns::BC_GETFIELD_GC_I;
use majit_metainterp::{Assembler, JitDriver};

#[repr(C)]
struct Node {
    value: i64,
    next: *mut Node,
}

struct NodeState {
    total: i64,
    sel: usize,
}

pub type Bytecode = [u8];

/// The control: one expression, `state.sel.value`.
const OP_DIRECT: u8 = 1;
/// The subject: the same read with the ref bound first.
const OP_VIA_LOCAL: u8 = 2;

#[majit_macros::jit_interp(
    state = NodeState,
    env = Bytecode,
    greens = [pc, program],
    state_fields = { total: int, sel: ref(Node) },
    ref_fields = { Node::next => Node },
    int_fields = { Node::value => i64 },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_nodes(program: &Bytecode, threshold: u32, sel: usize) -> i64 {
    let mut driver: JitDriver<NodeState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = NodeState { total: 0, sel };
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
            OP_DIRECT => {
                state.total = state.total + state.sel.value;
            }
            OP_VIA_LOCAL => {
                let n = state.sel;
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
    __prebuild_jitcode_liveness_dispatch_nodes(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_dispatch_nodes(&mut asm, 0i64).expect("dispatch lower must succeed")
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

#[test]
fn a_ref_state_field_bound_to_a_local_keeps_its_declared_pointee() {
    let _ = build_dispatch_jitcode();

    let census = majit_metainterp::dispatch_arm_census();
    let counted = census
        .iter()
        .find(|e| e.interp == "NodeState")
        .unwrap_or_else(|| panic!("the portal must record its arm count; census={census:?}"));
    assert_eq!(
        counted.arms, 2,
        "OP_DIRECT and OP_VIA_LOCAL are counted; the `_` default is not; \
         census={census:?}"
    );

    let degraded: Vec<&'static str> = majit_metainterp::degraded_dispatch_arms()
        .into_iter()
        .filter(|e| e.interp == "NodeState")
        .map(|e| e.arm)
        .collect();
    assert!(
        !degraded.contains(&"OP_DIRECT"),
        "the control degraded, so the fixture is measuring its own mistake and \
         the subject says nothing; degraded={degraded:?}"
    );

    majit_metainterp::assert_no_degraded_dispatch_arms("NodeState");
}

/// What an un-degraded arm was supposed to emit. The gate above says the arm
/// lowered; this says what it lowered TO.
#[test]
fn each_arm_reads_the_field_through_a_getfield_gc() {
    let dispatch_jc = build_dispatch_jitcode();
    let reads: usize = all_jitcode_bodies(&dispatch_jc)
        .iter()
        .map(|body| body.iter().filter(|op| **op == BC_GETFIELD_GC_I).count())
        .sum();
    assert_eq!(
        reads, 2,
        "one `value` read per arm; 1 is the local-binding arm degraded to a stub",
    );
}

/// The answer both spellings owe. It is the same either way — which is the
/// point: the degradation above never showed up here.
#[test]
fn both_spellings_read_the_same_field() {
    let node = Node {
        value: 5,
        next: std::ptr::null_mut(),
    };
    let sel = &node as *const Node as usize;

    let program: &Bytecode = &[OP_DIRECT, OP_VIA_LOCAL];
    assert_eq!(
        dispatch_nodes(program, u32::MAX, sel),
        10,
        "each arm adds `sel.value`",
    );
}
