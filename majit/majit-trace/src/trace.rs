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
}
