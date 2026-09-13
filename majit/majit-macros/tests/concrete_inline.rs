//! The concrete half of a helper must not require JIT metadata.

#[repr(C)]
struct Node {
    value: i64,
}

struct State {
    node: usize,
}
struct Program;

// These names deliberately shadow the crates. Any ungated generated JIT
// item would fail to resolve its types in this module.
mod concrete {
    use super::{Node, Program, State};
    mod majit_metainterp {}
    mod majit_ir {}
    mod majit_meta {}

    macro_rules! jit_merge_point {
        ($($args:tt)*) => {};
    }
    macro_rules! can_enter_jit {
        ($($args:tt)*) => {};
    }

    #[majit_macros::jit_interp(
        trace_cfg = (any()),
        state = State,
        env = Program,
        state_fields = { node: ref(Node) },
        int_fields = { Node::value => i64 },
    )]
    pub fn run(state: &mut State, _program: &Program) -> i64 {
        let mut pc = 0;
        while pc < 1 {
            jit_merge_point!(missing_driver, missing_program, pc; state);
            state.node.value = state.node.value + 1;
            pc += 1;
            can_enter_jit!(missing_driver, pc, state);
        }
        state.node.value
    }

    #[majit_macros::jit_inline(
        trace_cfg = (any()),
        ref_params = { node: ref(Node) },
        int_fields = { Node::value => i64 },
    )]
    pub fn increment(node: usize) -> i64 {
        node.value = node.value + 1;
        node.value
    }
}

#[test]
fn concrete_ref_reads_and_writes_survive_without_metadata() {
    let mut node = Node { value: 41 };
    assert_eq!(concrete::increment(&mut node as *mut Node as usize), 42);
    assert_eq!(node.value, 42);
    let mut state = State {
        node: &mut node as *mut Node as usize,
    };
    assert_eq!(concrete::run(&mut state, &Program), 43);
    assert_eq!(node.value, 43);
}
