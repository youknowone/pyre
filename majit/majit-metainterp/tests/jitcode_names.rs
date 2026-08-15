//! Every macro-built JitCode carries the name of what it was built from.
//!
//! `jitcode.py:14-15 def __init__(self, name, ...): self.name = name` — upstream
//! has no unnamed JitCode, and the name is what `dump()` and every diagnostic
//! that prints one use to say *which* code they are talking about.
//!
//! `JitCodeBuilder` defaults the field to the empty string, and the macro emit
//! sites never called `set_name`, so the whole macro-built population reported
//! `""`. The cost was not cosmetic: `BC_ABORT` has two emitter families and one
//! opcode, so the abort log had to discriminate them by frame shape
//! (`depth>1 body=1` = a degraded arm stub) — a heuristic standing in for a
//! name that was simply absent. The bytecode encoder's ceiling audit prints one
//! line per finished builder and identified none of them.
//!
//! The names here are diagnostic strings, not identity: nothing keys a cache or
//! a comparison on them. What the tests pin is that each JitCode names its own
//! source, because a shared or empty name is exactly as useless as no name.

use majit_macros::jit_inline;
use majit_metainterp::{Assembler, JitCode, JitDriver};

#[repr(C)]
struct Cell {
    value: i64,
}

#[jit_inline(ref_params = { cell: ref(Cell) })]
fn bump_named_cell(cell: usize) -> i64 {
    let value = cell.value;
    value + 1
}

#[test]
fn an_inline_helper_names_its_source_function() {
    let mut asm = Assembler::new();
    let jitcode = __majit_inline_jitcode_bump_named_cell_with_asm(&mut asm);
    assert_eq!(
        jitcode.name(),
        "bump_named_cell",
        "an inline helper's JitCode names the `#[jit_inline]` function it was \
         lowered from",
    );
}

struct NamedState {
    a: i64,
}

const OP_NOP: u8 = 0;
const OP_INC_A: u8 = 1;

pub type Bytecode = [u8];

#[majit_macros::jit_interp(
    state = NamedState,
    env = Bytecode,
    state_fields = { a: int },
    greens = [],
)]
#[allow(unused_assignments, unused_variables)]
fn named_dispatch(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<NamedState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = NamedState { a: 0 };
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
            _ => break,
        }
    }
    state.a
}

fn build_named_dispatch() -> JitCode {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
    __prebuild_jitcode_liveness_named_dispatch(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    __dispatch_jitcode_named_dispatch(&mut asm, 0i64).expect("dispatch lower must succeed")
}

#[test]
fn the_dispatch_jitcode_names_the_machine_function() {
    assert_eq!(
        build_named_dispatch().name(),
        "named_dispatch",
        "the root JitCode of a `#[jit_interp]` machine names the dispatch \
         function itself",
    );
}

/// The arm sub-JitCodes, in the order the dispatch registered them.
fn arm_names(dispatch: &JitCode) -> Vec<String> {
    dispatch
        .exec
        .descrs
        .iter()
        .filter_map(|descr| descr.as_jitcode())
        .map(|sub| sub.name().to_string())
        .collect()
}

#[test]
fn each_arm_subjitcode_names_the_arm_it_came_from() {
    let dispatch = build_named_dispatch();
    let names = arm_names(&dispatch);
    assert_eq!(
        names.len(),
        2,
        "one sub-JitCode per non-default arm; got {names:?}",
    );

    // The interp prefix is what makes a name readable in a log that carries
    // more than one machine's traces — `OP_NOP` alone does not say whose.
    for name in &names {
        assert!(
            name.starts_with("NamedState::"),
            "an arm sub-JitCode names the state type it dispatches on; got {name:?}",
        );
    }

    // The arm's own spelling, and the reason the whole exercise is worth
    // anything: two arms of one machine must not answer with the same string.
    assert!(
        names.iter().any(|n| n.contains("OP_NOP")),
        "the nop arm names its own pattern; got {names:?}",
    );
    assert!(
        names.iter().any(|n| n.contains("OP_INC_A")),
        "the lowered arm names its own pattern; got {names:?}",
    );
    assert_ne!(
        names[0], names[1],
        "two arms sharing a name identify nothing, the same as sharing the \
         empty one; got {names:?}",
    );
}
