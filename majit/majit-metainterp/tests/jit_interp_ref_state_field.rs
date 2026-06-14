//! Genuine `ref(T)` state-field lowering + value-typing tests.
//!
//! A `ref(T)` state scalar is a genuine `InputArgRef` carried in the ref
//! register bank.  A dispatch arm that reads and writes such a field must
//! lower to `load_state_field_ref` / `store_state_field_ref`, and the
//! interp's `live_value_types` must tag the appended ref slot `Type::Ref`
//! so guard-failure resume routes it to the ref bank.

use majit_metainterp::jitcode::insns::{BC_LOAD_STATE_FIELD_REF, BC_STORE_STATE_FIELD_REF};
use majit_metainterp::{Assembler, JitDriver};

/// The heap class the ref field points at — only a type tag for `ref(Stack)`;
/// the field's storage is a `usize` carrier.
#[allow(dead_code)]
struct Stack;

struct RefTestState {
    a: i64,
    sel: usize,
}

const OP_NOP: u8 = 0;
const OP_INC_A: u8 = 1;
const OP_TOUCH_SEL: u8 = 2;

pub type Bytecode = [u8];

#[allow(dead_code)]
trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}
impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

#[majit_macros::jit_interp(
    state = RefTestState,
    env = Bytecode,
    state_fields = { a: int, sel: ref(Stack) },
)]
#[allow(unused_assignments, unused_variables)]
fn ref_minimal(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<RefTestState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = RefTestState { a: 0, sel: 0 };
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
            OP_NOP => {}
            OP_INC_A => state.a += 1,
            // ref read (RHS) feeding a ref write (LHS): lowers to
            // load_state_field_ref + store_state_field_ref.
            OP_TOUCH_SEL => state.sel = state.sel,
            _ => break,
        }
    }
    state.a
}

/// Every jitcode body produced for the dispatch portal: the dispatch
/// JitCode itself plus each per-arm sub-JitCode.
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

fn build_ref_dispatch_jitcode(asm: &mut Assembler) -> majit_metainterp::JitCode {
    asm.set_canonical_liveness_triple(vec![0], vec![0], vec![]);
    __prebuild_jitcode_liveness_ref_minimal(asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_ref_minimal(asm, 0i64).expect("dispatch lower must succeed for ref fixture")
}

#[test]
fn ref_state_field_lowers_load_and_store_state_field_ref() {
    let mut asm = Assembler::new();
    let dispatch_jc = build_ref_dispatch_jitcode(&mut asm);
    let bodies = all_jitcode_bodies(&dispatch_jc);
    assert!(
        bodies
            .iter()
            .any(|body| body.iter().any(|&b| b == BC_LOAD_STATE_FIELD_REF)),
        "ref read must lower to load_state_field_ref; bodies: {bodies:?}"
    );
    assert!(
        bodies
            .iter()
            .any(|body| body.iter().any(|&b| b == BC_STORE_STATE_FIELD_REF)),
        "ref write must lower to store_state_field_ref; bodies: {bodies:?}"
    );
}

#[test]
fn ref_state_field_live_value_types_tags_ref_slot() {
    use majit_metainterp::JitState as _;
    let program: &Bytecode = &[OP_NOP];
    let state = RefTestState { a: 7, sel: 0 };
    let meta = state.build_meta(0, program);
    let live = state.extract_live(&meta);
    let types = state.live_value_types(&meta);
    assert_eq!(
        live.len(),
        types.len(),
        "live_value_types must be 1:1 with extract_live"
    );
    // Layout: [a (Int), sel (Ref)] — the int scalar then the appended ref.
    assert_eq!(types.len(), 2);
    assert_eq!(types[0], majit_ir::Type::Int);
    assert_eq!(
        *types.last().unwrap(),
        majit_ir::Type::Ref,
        "the appended ref scalar slot must be tagged Type::Ref"
    );
}
