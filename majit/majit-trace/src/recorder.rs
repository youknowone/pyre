/// Trace recorder — records IR operations during interpreter execution.
///
/// The recorder is the bridge between the interpreter and the JIT.
/// When tracing is active, every interpreter operation is fed to the
/// recorder, which builds a linear sequence of IR operations (a trace).
///
/// Reference: rpython/jit/metainterp/pyjitpl.py MetaInterp.record()
use majit_ir::{DescrRef, InputArg, Op, OpCode, OpRef, Type};

use crate::trace::Trace;

/// Maximum number of operations before the trace is considered too long.
const TRACE_LIMIT: usize = 6000;

/// The trace recorder: accumulates operations during tracing.
pub struct TraceRecorder {
    /// Recorded operations.
    ops: Vec<Op>,
    /// Input arguments to the trace (live variables at the loop header).
    inputargs: Vec<InputArg>,
    /// Next OpRef index to assign.
    op_count: u32,
    /// Whether the recorder has been finalized (closed or finished).
    finalized: bool,
    /// Whether the trace was aborted.
    aborted: bool,
}

impl TraceRecorder {
    /// Create a new, empty trace recorder.
    pub fn new() -> Self {
        TraceRecorder {
            ops: Vec::with_capacity(256),
            inputargs: Vec::new(),
            op_count: 0,
            finalized: false,
            aborted: false,
        }
    }

    /// Register an input argument of the given type.
    /// Returns an OpRef that can be used as an argument to subsequent operations.
    /// Input arguments are numbered starting from 0; the OpRef index matches
    /// the input argument index.
    pub fn record_input_arg(&mut self, tp: Type) -> OpRef {
        assert!(!self.finalized, "recorder already finalized");
        assert!(
            self.ops.is_empty(),
            "input args must be registered before any operations"
        );
        let index = self.inputargs.len() as u32;
        self.inputargs.push(InputArg::from_type(tp, index));
        let opref = OpRef(self.op_count);
        self.op_count += 1;
        opref
    }

    /// Record a regular (non-guard) operation.
    /// Returns the OpRef for this operation's result.
    pub fn record_op(&mut self, opcode: OpCode, args: &[OpRef]) -> OpRef {
        assert!(!self.finalized, "recorder already finalized");
        assert!(!opcode.is_guard(), "use record_guard for guard operations");
        let opref = OpRef(self.op_count);
        let mut op = Op::new(opcode, args);
        op.pos = opref;
        self.ops.push(op);
        self.op_count += 1;
        opref
    }

    /// Record an operation with a descriptor (e.g., field access, call).
    /// Returns the OpRef for this operation's result.
    pub fn record_op_with_descr(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        assert!(!self.finalized, "recorder already finalized");
        assert!(!opcode.is_guard(), "use record_guard for guard operations");
        let opref = OpRef(self.op_count);
        let mut op = Op::with_descr(opcode, args, descr);
        op.pos = opref;
        self.ops.push(op);
        self.op_count += 1;
        opref
    }

    /// Record a guard operation.
    /// Guards carry a FailDescr that describes what happens when the guard fails.
    /// Returns the OpRef for this guard.
    pub fn record_guard(&mut self, opcode: OpCode, args: &[OpRef], descr: DescrRef) -> OpRef {
        assert!(!self.finalized, "recorder already finalized");
        assert!(opcode.is_guard(), "opcode {:?} is not a guard", opcode);
        let opref = OpRef(self.op_count);
        let mut op = Op::with_descr(opcode, args, descr);
        op.pos = opref;
        self.ops.push(op);
        self.op_count += 1;
        opref
    }

    /// Close the loop: add a JUMP operation back to the start.
    /// `jump_args` are the values of the input arguments at the end of the loop.
    pub fn close_loop(&mut self, jump_args: &[OpRef]) {
        assert!(!self.finalized, "recorder already finalized");
        assert_eq!(
            jump_args.len(),
            self.inputargs.len(),
            "JUMP must have the same number of args as input args"
        );
        let opref = OpRef(self.op_count);
        let mut op = Op::new(OpCode::Jump, jump_args);
        op.pos = opref;
        self.ops.push(op);
        self.op_count += 1;
        self.finalized = true;
    }

    /// Finish the trace (non-looping): add a FINISH operation.
    /// `finish_args` are the values returned from the trace.
    pub fn finish(&mut self, finish_args: &[OpRef], descr: DescrRef) {
        assert!(!self.finalized, "recorder already finalized");
        let opref = OpRef(self.op_count);
        let mut op = Op::with_descr(OpCode::Finish, finish_args, descr);
        op.pos = opref;
        self.ops.push(op);
        self.op_count += 1;
        self.finalized = true;
    }

    /// Return the completed trace.
    /// The recorder is consumed; no further operations can be recorded.
    pub fn get_trace(self) -> Trace {
        assert!(
            self.finalized,
            "trace must be finalized with close_loop() or finish() before calling get_trace()"
        );
        assert!(!self.aborted, "cannot get trace from aborted recorder");
        Trace::new(self.inputargs, self.ops)
    }

    /// Abort the current trace, discarding all recorded operations.
    pub fn abort(mut self) {
        self.aborted = true;
        self.ops.clear();
        self.inputargs.clear();
    }

    /// Number of operations recorded so far (not counting input args).
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Number of input arguments registered.
    pub fn num_inputargs(&self) -> usize {
        self.inputargs.len()
    }

    /// Whether the trace has exceeded the maximum allowed length.
    pub fn is_too_long(&self) -> bool {
        self.ops.len() >= TRACE_LIMIT
    }

    /// Whether the recorder has been finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }
}

impl Default for TraceRecorder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{DescrRef, FailDescr, Type};
    use std::sync::Arc;

    /// A minimal FailDescr implementation for testing.
    #[derive(Debug)]
    struct TestFailDescr {
        index: u32,
    }

    impl majit_ir::Descr for TestFailDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
            Some(self)
        }
    }

    impl FailDescr for TestFailDescr {
        fn fail_index(&self) -> u32 {
            self.index
        }
        fn fail_arg_types(&self) -> &[Type] {
            &[]
        }
    }

    fn make_fail_descr(index: u32) -> DescrRef {
        Arc::new(TestFailDescr { index })
    }

    #[test]
    fn test_record_simple_loop() {
        // Trace: i0 -> i1 = int_add(i0, i0) -> jump(i1)
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert_eq!(i0, OpRef(0));

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(i1, OpRef(1));

        rec.close_loop(&[i1]);

        let trace = rec.get_trace();
        assert!(trace.is_loop());
        assert_eq!(trace.num_inputargs(), 1);
        assert_eq!(trace.num_ops(), 2); // IntAdd + Jump
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::Jump);
        assert_eq!(trace.ops[1].args[0], i1);
    }

    #[test]
    fn test_record_with_guard() {
        // i0 -> guard_true(i0) -> i1 = int_add(i0, i0) -> jump(i1)
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(0);
        let _g = rec.record_guard(OpCode::GuardTrue, &[i0], descr);

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.close_loop(&[i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_ops(), 3); // GuardTrue + IntAdd + Jump
        assert!(trace.ops[0].opcode.is_guard());
        assert!(trace.ops[0].descr.is_some());
    }

    #[test]
    fn test_record_finish() {
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(99);
        rec.finish(&[i1], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());
        assert!(!trace.is_loop());
    }

    #[test]
    fn test_record_multiple_inputargs() {
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let r0 = rec.record_input_arg(Type::Ref);
        let f0 = rec.record_input_arg(Type::Float);
        assert_eq!(i0, OpRef(0));
        assert_eq!(r0, OpRef(1));
        assert_eq!(f0, OpRef(2));
        assert_eq!(rec.num_inputargs(), 3);

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.close_loop(&[i1, r0, f0]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 3);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Ref);
        assert_eq!(trace.inputargs[2].tp, Type::Float);
    }

    #[test]
    fn test_opref_assignment() {
        // OpRef indices are monotonically increasing, with input args taking
        // the first slots.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        assert_eq!(i0, OpRef(0));
        assert_eq!(i1, OpRef(1));

        let i2 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        assert_eq!(i2, OpRef(2));

        let descr = make_fail_descr(0);
        let g0 = rec.record_guard(OpCode::GuardTrue, &[i2], descr);
        assert_eq!(g0, OpRef(3));

        let i3 = rec.record_op(OpCode::IntSub, &[i2, i0]);
        assert_eq!(i3, OpRef(4));

        rec.close_loop(&[i3, i1]);
        let trace = rec.get_trace();

        // The op's .pos should match
        assert_eq!(trace.ops[0].pos, OpRef(2)); // IntAdd
        assert_eq!(trace.ops[1].pos, OpRef(3)); // GuardTrue
        assert_eq!(trace.ops[2].pos, OpRef(4)); // IntSub
        assert_eq!(trace.ops[3].pos, OpRef(5)); // Jump
    }

    #[test]
    fn test_abort() {
        let mut rec = TraceRecorder::new();
        rec.record_input_arg(Type::Int);
        rec.record_op(OpCode::IntAdd, &[OpRef(0), OpRef(0)]);
        assert_eq!(rec.num_ops(), 1);
        rec.abort(); // should not panic
    }

    #[test]
    #[should_panic(expected = "recorder already finalized")]
    fn test_record_after_finalize() {
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.close_loop(&[i0]);
        // This should panic
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
    }

    #[test]
    #[should_panic(expected = "JUMP must have the same number of args")]
    fn test_close_loop_arg_mismatch() {
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let _i1 = rec.record_input_arg(Type::Int);
        // Should panic: 2 input args but only 1 jump arg
        rec.close_loop(&[i0]);
    }

    #[test]
    #[should_panic(expected = "use record_guard")]
    fn test_record_op_with_guard_opcode() {
        let mut rec = TraceRecorder::new();
        rec.record_input_arg(Type::Int);
        // Should panic: guard opcodes must use record_guard
        rec.record_op(OpCode::GuardTrue, &[OpRef(0)]);
    }

    #[test]
    fn test_is_too_long() {
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert!(!rec.is_too_long());

        // Record operations up to the limit
        let mut last = i0;
        for _ in 0..6000 {
            last = rec.record_op(OpCode::IntAdd, &[last, i0]);
        }
        assert!(rec.is_too_long());
    }

    #[test]
    fn test_record_op_with_descr() {
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let descr = make_fail_descr(42);
        let i1 = rec.record_op_with_descr(OpCode::CallI, &[i0], descr);
        assert_eq!(i1, OpRef(1));

        // Verify the op has a descriptor
        assert!(rec.ops[0].descr.is_some());
    }

    #[test]
    fn test_complex_trace() {
        // Build a more realistic trace:
        //   i0, i1 = inputargs
        //   i2 = int_lt(i0, i1)
        //   guard_true(i2)
        //   i3 = int_add(i0, 1)  -- 1 is a const, but we model as OpRef
        //   jump(i3, i1)
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let i2 = rec.record_op(OpCode::IntLt, &[i0, i1]);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[i2], descr);

        let i3 = rec.record_op(OpCode::IntAdd, &[i0, i1]);

        rec.close_loop(&[i3, i1]);

        let trace = rec.get_trace();
        assert!(trace.is_loop());
        assert_eq!(trace.num_ops(), 4);

        // Verify all guard ops can be iterated
        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
    }
}
