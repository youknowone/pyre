/// The Trace data structure — a completed sequence of IR operations.
///
/// A Trace is the output of the TraceRecorder and the input to the
/// optimizer and backend. It represents a linear sequence of operations
/// that forms a loop (ending with JUMP) or an exit (ending with FINISH).
///
/// Reference: rpython/jit/metainterp/history.py TreeLoop
use majit_ir::{InputArg, Op, OpCode, OpRef};

/// A completed trace ready for optimization and compilation.
#[derive(Clone, Debug)]
pub struct Trace {
    /// Input arguments to the trace (loop header variables).
    pub inputargs: Vec<InputArg>,
    /// The recorded operations, in execution order.
    pub ops: Vec<Op>,
}

impl Trace {
    /// Create a new trace from input arguments and operations.
    pub fn new(inputargs: Vec<InputArg>, ops: Vec<Op>) -> Self {
        Trace { inputargs, ops }
    }

    /// Number of operations in the trace.
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Number of input arguments.
    pub fn num_inputargs(&self) -> usize {
        self.inputargs.len()
    }

    /// Whether this trace ends with a JUMP (i.e., is a loop).
    pub fn is_loop(&self) -> bool {
        self.ops.last().is_some_and(|op| op.opcode == OpCode::Jump)
    }

    /// Whether this trace ends with FINISH.
    pub fn is_finished(&self) -> bool {
        self.ops
            .last()
            .is_some_and(|op| op.opcode == OpCode::Finish)
    }

    /// Get a reference to the operation at the given OpRef index.
    pub fn get_op(&self, opref: OpRef) -> Option<&Op> {
        self.ops.get(opref.0 as usize)
    }

    /// Iterate over all operations.
    pub fn iter_ops(&self) -> impl Iterator<Item = &Op> {
        self.ops.iter()
    }

    /// Iterate over all guard operations.
    pub fn iter_guards(&self) -> impl Iterator<Item = &Op> {
        self.ops.iter().filter(|op| op.opcode.is_guard())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::Type;

    #[test]
    fn test_empty_trace() {
        let trace = Trace::new(vec![], vec![]);
        assert_eq!(trace.num_ops(), 0);
        assert_eq!(trace.num_inputargs(), 0);
        assert!(!trace.is_loop());
        assert!(!trace.is_finished());
    }

    #[test]
    fn test_trace_with_jump() {
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
            Op::new(OpCode::Jump, &[OpRef(1)]),
        ];
        let trace = Trace::new(inputargs, ops);
        assert!(trace.is_loop());
        assert!(!trace.is_finished());
        assert_eq!(trace.num_ops(), 2);
        assert_eq!(trace.num_inputargs(), 1);
    }

    #[test]
    fn test_trace_with_finish() {
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
            Op::new(OpCode::Finish, &[OpRef(1)]),
        ];
        let trace = Trace::new(inputargs, ops);
        assert!(!trace.is_loop());
        assert!(trace.is_finished());
    }

    #[test]
    fn test_inputarg_types() {
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
        ];
        let trace = Trace::new(inputargs, vec![]);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Ref);
        assert_eq!(trace.inputargs[2].tp, Type::Float);
    }

    // ══════════════════════════════════════════════════════════════════
    // History / TreeLoop parity tests
    // Ported from rpython/jit/metainterp/test/test_history.py
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_trace_structure_inputargs_and_ops() {
        // TreeLoop has inputargs and operations as primary fields.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::IntSub, &[OpRef(2), OpRef(0)]),
            Op::new(OpCode::Jump, &[OpRef(3), OpRef(1)]),
        ];
        let trace = Trace::new(inputargs, ops);

        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.num_ops(), 3);
        assert!(trace.is_loop());
    }

    #[test]
    fn test_trace_guards_can_have_fail_args() {
        // Guards in a trace carry fail_args.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(0)]);
        guard.fail_args = Some(smallvec::smallvec![OpRef(0), OpRef(1)]);

        let ops = vec![
            guard,
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(2), OpRef(1)]),
        ];
        let trace = Trace::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);
        let fa = guards[0].fail_args.as_ref().unwrap();
        assert_eq!(fa.len(), 2);
        assert_eq!(fa[0], OpRef(0));
        assert_eq!(fa[1], OpRef(1));
    }

    #[test]
    fn test_trace_get_op() {
        // get_op retrieves ops by index.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
            Op::new(OpCode::Jump, &[OpRef(1)]),
        ];
        let trace = Trace::new(inputargs, ops);

        let op0 = trace.get_op(OpRef(0)).unwrap();
        assert_eq!(op0.opcode, OpCode::IntAdd);

        let op1 = trace.get_op(OpRef(1)).unwrap();
        assert_eq!(op1.opcode, OpCode::Jump);

        assert!(trace.get_op(OpRef(99)).is_none());
    }

    #[test]
    fn test_trace_iter_guards_filters_correctly() {
        // iter_guards returns only guard ops.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef(2)]),
            Op::new(OpCode::IntSub, &[OpRef(2), OpRef(0)]),
            Op::new(OpCode::GuardFalse, &[OpRef(3)]),
            Op::new(OpCode::Jump, &[OpRef(3), OpRef(1)]),
        ];
        let trace = Trace::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 2);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
        assert_eq!(guards[1].opcode, OpCode::GuardFalse);
    }

    #[test]
    fn test_trace_not_loop_not_finished() {
        // A trace without Jump or Finish is neither loop nor finished.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)])];
        let trace = Trace::new(inputargs, ops);
        assert!(!trace.is_loop());
        assert!(!trace.is_finished());
    }

    #[test]
    fn test_trace_loop_vs_finish_exclusive() {
        // A trace cannot be both a loop and finished.
        let inputargs = vec![InputArg::new_int(0)];

        let loop_trace = Trace::new(
            inputargs.clone(),
            vec![
                Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
                Op::new(OpCode::Jump, &[OpRef(1)]),
            ],
        );
        assert!(loop_trace.is_loop());
        assert!(!loop_trace.is_finished());

        let finish_trace = Trace::new(
            inputargs,
            vec![
                Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(0)]),
                Op::new(OpCode::Finish, &[OpRef(1)]),
            ],
        );
        assert!(!finish_trace.is_loop());
        assert!(finish_trace.is_finished());
    }

    #[test]
    fn test_trace_mixed_type_inputargs() {
        // Traces support mixed-type input arguments (int, ref, float).
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_ref(1),
            InputArg::new_float(2),
        ];
        let ops = vec![Op::new(OpCode::Jump, &[OpRef(0), OpRef(1), OpRef(2)])];
        let trace = Trace::new(inputargs, ops);

        assert_eq!(trace.num_inputargs(), 3);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Ref);
        assert_eq!(trace.inputargs[2].tp, Type::Float);
        assert!(trace.is_loop());
    }

    #[test]
    fn test_trace_multiple_guards_with_different_fail_args() {
        // Multiple guards can have different fail_args.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];

        let mut g0 = Op::new(OpCode::GuardTrue, &[OpRef(0)]);
        g0.fail_args = Some(smallvec::smallvec![OpRef(0)]);

        let mut g1 = Op::new(OpCode::GuardFalse, &[OpRef(1)]);
        g1.fail_args = Some(smallvec::smallvec![OpRef(0), OpRef(1)]);

        let ops = vec![
            g0,
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            g1,
            Op::new(OpCode::Jump, &[OpRef(2), OpRef(1)]),
        ];
        let trace = Trace::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 2);

        assert_eq!(guards[0].fail_args.as_ref().unwrap().len(), 1);
        assert_eq!(guards[1].fail_args.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_trace_guard_without_fail_args() {
        // Guards without explicitly set fail_args have None.
        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        let trace = Trace::new(inputargs, ops);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);
        assert!(guards[0].fail_args.is_none());
    }

    #[test]
    fn test_trace_ops_have_correct_opcodes() {
        // iter_ops preserves op order and opcodes.
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::IntMul, &[OpRef(2), OpRef(0)]),
            Op::new(OpCode::IntSub, &[OpRef(3), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(4), OpRef(1)]),
        ];
        let trace = Trace::new(inputargs, ops);

        let opcodes: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(
            opcodes,
            vec![OpCode::IntAdd, OpCode::IntMul, OpCode::IntSub, OpCode::Jump]
        );
    }
}
