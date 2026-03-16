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

    /// Create a trace recorder pre-configured for retracing from a guard.
    ///
    /// The recorder starts with `num_inputs` int-typed input args,
    /// matching the guard's fail_args.
    pub fn with_num_inputs(num_inputs: usize) -> Self {
        let mut recorder = Self::new();
        for _ in 0..num_inputs {
            recorder.record_input_arg(Type::Int);
        }
        recorder
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

    /// Record a guard with explicit fail_args — values stored in the dead
    /// frame on guard failure. Mirrors rpython setfailargs().
    pub fn record_guard_with_fail_args(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
        fail_args: &[OpRef],
    ) -> OpRef {
        assert!(!self.finalized, "recorder already finalized");
        assert!(opcode.is_guard(), "opcode {:?} is not a guard", opcode);
        let opref = OpRef(self.op_count);
        let mut op = Op::with_descr(opcode, args, descr);
        op.pos = opref;
        op.fail_args = Some(smallvec::SmallVec::from_slice(fail_args));
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

    // ── Trace limit parity tests (RPython: test_tracelimit.py) ──

    #[test]
    fn test_is_too_long_boundary() {
        // Verify exact boundary: TRACE_LIMIT-1 ops is not too long,
        // TRACE_LIMIT ops is too long.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);

        let mut last = i0;
        for _ in 0..(TRACE_LIMIT - 1) {
            last = rec.record_op(OpCode::IntAdd, &[last, i0]);
        }
        assert!(
            !rec.is_too_long(),
            "TRACE_LIMIT - 1 ops should not be too long"
        );

        // One more op pushes it to the limit.
        let _over = rec.record_op(OpCode::IntAdd, &[last, i0]);
        assert!(
            rec.is_too_long(),
            "TRACE_LIMIT ops should be too long"
        );
    }

    #[test]
    fn test_trace_abort_on_too_long() {
        // Simulate what the meta-interpreter does when a trace is too long:
        // check is_too_long() then call abort().
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);

        let mut last = i0;
        for _ in 0..TRACE_LIMIT {
            last = rec.record_op(OpCode::IntAdd, &[last, i0]);
        }
        assert!(rec.is_too_long());

        // Abort discards the trace.
        rec.abort();
        // After abort, the recorder is consumed; can't call get_trace().
    }

    #[test]
    fn test_trace_too_long_still_records() {
        // Even after exceeding the limit, the recorder still accepts ops.
        // The meta-interpreter is responsible for checking is_too_long()
        // and aborting. This mirrors RPython's non-exception trace limit.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);

        let mut last = i0;
        for _ in 0..(TRACE_LIMIT + 100) {
            last = rec.record_op(OpCode::IntAdd, &[last, i0]);
        }
        assert!(rec.is_too_long());
        assert_eq!(rec.num_ops(), TRACE_LIMIT + 100);

        // Can still finalize if we want to (the limit is advisory).
        rec.close_loop(&[last]);
        let trace = rec.get_trace();
        // The trace contains all ops, including those beyond the limit.
        assert!(trace.num_ops() > TRACE_LIMIT);
    }

    #[test]
    fn test_fresh_recorder_not_too_long() {
        // A freshly created recorder should never be too long.
        let rec = TraceRecorder::new();
        assert!(!rec.is_too_long());
        assert_eq!(rec.num_ops(), 0);
    }

    #[test]
    fn test_retrace_recorder_not_too_long() {
        // A recorder created for retracing (with_num_inputs) starts with
        // no operations, so it should not be too long.
        let rec = TraceRecorder::with_num_inputs(5);
        assert!(!rec.is_too_long());
        assert_eq!(rec.num_ops(), 0);
        assert_eq!(rec.num_inputargs(), 5);
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

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
    }

    // ══════════════════════════════════════════════════════════════════
    // Opencoder parity tests
    // Ported from rpython/jit/metainterp/test/test_opencoder.py
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_simple_iterator() {
        // Parity: test_simple_iterator
        // Record two INT_ADD ops and verify trace structure matches.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add0 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let add1 = rec.record_op(OpCode::IntAdd, &[add0, i0]);

        rec.close_loop(&[add1, i1]);
        let trace = rec.get_trace();

        // Verify the trace has the correct number of ops (2 + Jump).
        assert_eq!(trace.num_ops(), 3);
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[2].opcode, OpCode::Jump);

        // First add uses input args i0, i1.
        assert_eq!(trace.ops[0].args[0], i0);
        assert_eq!(trace.ops[0].args[1], i1);

        // Second add references the result of first add and i0.
        assert_eq!(trace.ops[1].args[0], add0);
        assert_eq!(trace.ops[1].args[1], i0);
    }

    #[test]
    fn test_inputargs_preserved() {
        // Parity: Trace([i0, i1], ...) preserves input args.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        assert_eq!(rec.num_inputargs(), 2);

        rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let add = rec.record_op(OpCode::IntAdd, &[OpRef(2), i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Int);
    }

    #[test]
    fn test_op_references_chain() {
        // Parity: ops that reference previous ops form correct chains.
        // i0 -> add = int_add(i0, i0) -> sub = int_sub(add, i0) -> jump(sub)
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);

        rec.close_loop(&[sub]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[0].pos, OpRef(1)); // after 1 inputarg
        assert_eq!(trace.ops[1].opcode, OpCode::IntSub);
        assert_eq!(trace.ops[1].args[0], add); // references the add result
        assert_eq!(trace.ops[1].args[1], i0);  // references the input arg
    }

    #[test]
    fn test_guard_with_fail_args() {
        // Parity: guards can carry fail_args describing live values at guard.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let descr = make_fail_descr(0);
        let guard = rec.record_guard_with_fail_args(
            OpCode::GuardTrue,
            &[add],
            descr,
            &[i0, i1, add],
        );

        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        rec.close_loop(&[sub, i1]);

        let trace = rec.get_trace();
        // Find the guard op.
        let guard_op = &trace.ops[1]; // after IntAdd
        assert_eq!(guard_op.pos, guard);
        assert!(guard_op.opcode.is_guard());
        let fail_args = guard_op.fail_args.as_ref().unwrap();
        assert_eq!(fail_args.len(), 3);
        assert_eq!(fail_args[0], i0);
        assert_eq!(fail_args[1], i1);
        assert_eq!(fail_args[2], add);
    }

    #[test]
    fn test_multiple_guards() {
        // Parity: multiple guards in one trace.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let descr0 = make_fail_descr(0);
        rec.record_guard_with_fail_args(
            OpCode::GuardTrue,
            &[i0],
            descr0,
            &[i0, i1],
        );

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);

        let descr1 = make_fail_descr(1);
        rec.record_guard_with_fail_args(
            OpCode::GuardFalse,
            &[add],
            descr1,
            &[i0, add],
        );

        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        rec.close_loop(&[sub, i1]);

        let trace = rec.get_trace();
        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 2);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
        assert_eq!(guards[1].opcode, OpCode::GuardFalse);

        // First guard's fail_args
        let fa0 = guards[0].fail_args.as_ref().unwrap();
        assert_eq!(fa0.len(), 2);
        assert_eq!(fa0[0], i0);
        assert_eq!(fa0[1], i1);

        // Second guard's fail_args
        let fa1 = guards[1].fail_args.as_ref().unwrap();
        assert_eq!(fa1.len(), 2);
        assert_eq!(fa1[0], i0);
        assert_eq!(fa1[1], add);
    }

    #[test]
    fn test_close_loop_jump_targets_inputargs() {
        // Parity: close_loop produces a JUMP whose args correspond to inputargs.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        let jump = trace.ops.last().unwrap();
        assert_eq!(jump.opcode, OpCode::Jump);
        assert_eq!(jump.args.len(), trace.num_inputargs());
        assert_eq!(jump.args[0], add);
        assert_eq!(jump.args[1], i1);
    }

    #[test]
    fn test_finish_produces_finish_op() {
        // Parity: finish() produces a FINISH op with the given args.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(42);
        rec.finish(&[add], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());
        assert!(!trace.is_loop());

        let finish_op = trace.ops.last().unwrap();
        assert_eq!(finish_op.opcode, OpCode::Finish);
        assert_eq!(finish_op.args.len(), 1);
        assert_eq!(finish_op.args[0], add);
        assert!(finish_op.descr.is_some());
    }

    #[test]
    fn test_trace_length_tracking() {
        // Parity: num_ops() tracks the count accurately.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert_eq!(rec.num_ops(), 0);

        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(rec.num_ops(), 1);

        rec.record_op(OpCode::IntSub, &[OpRef(1), i0]);
        assert_eq!(rec.num_ops(), 2);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[OpRef(2)], descr);
        assert_eq!(rec.num_ops(), 3);

        rec.close_loop(&[OpRef(2)]);
        // After close_loop, Jump is added.
        assert_eq!(rec.num_ops(), 4);
    }

    #[test]
    fn test_with_num_inputs_creates_inputargs() {
        // Parity: TraceRecorder::with_num_inputs pre-creates input args.
        let rec = TraceRecorder::with_num_inputs(3);
        assert_eq!(rec.num_inputargs(), 3);
        assert_eq!(rec.num_ops(), 0);
    }

    #[test]
    fn test_with_num_inputs_oprefs() {
        // Input args from with_num_inputs get OpRef(0), OpRef(1), ...
        let mut rec = TraceRecorder::with_num_inputs(3);

        // The input args consumed OpRef(0..2), so next op gets OpRef(3).
        let add = rec.record_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
        assert_eq!(add, OpRef(3));

        rec.close_loop(&[OpRef(0), OpRef(1), OpRef(2)]);
        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 3);
        assert_eq!(trace.ops[0].pos, OpRef(3));
    }

    #[test]
    fn test_guard_descr_preserved() {
        // Parity: guard descriptors are preserved through get_trace().
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(77);
        rec.record_guard(OpCode::GuardNoException, &[], descr);

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();
        let guard = &trace.ops[0];
        assert!(guard.descr.is_some());
        let d = guard.descr.as_ref().unwrap();
        assert_eq!(d.index(), 77);
    }

    #[test]
    fn test_op_with_descr_preserved() {
        // Parity: op descriptors (e.g., for calls) are preserved.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);

        let call_descr = make_fail_descr(55);
        let result = rec.record_op_with_descr(OpCode::CallI, &[i0], call_descr);

        rec.close_loop(&[result]);
        let trace = rec.get_trace();
        let call_op = &trace.ops[0];
        assert_eq!(call_op.opcode, OpCode::CallI);
        assert!(call_op.descr.is_some());
        assert_eq!(call_op.descr.as_ref().unwrap().index(), 55);
    }

    #[test]
    fn test_empty_fail_args() {
        // Parity: guard with empty fail_args is valid.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(0);
        rec.record_guard_with_fail_args(
            OpCode::GuardTrue,
            &[i0],
            descr,
            &[],
        );

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();
        let guard = &trace.ops[0];
        let fail_args = guard.fail_args.as_ref().unwrap();
        assert!(fail_args.is_empty());
    }

    #[test]
    #[should_panic(expected = "opcode")]
    fn test_record_guard_with_non_guard_opcode() {
        // Parity: record_guard rejects non-guard opcodes.
        let mut rec = TraceRecorder::new();
        rec.record_input_arg(Type::Int);
        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::IntAdd, &[OpRef(0)], descr);
    }

    #[test]
    #[should_panic(expected = "input args must be registered before any operations")]
    fn test_inputarg_after_ops() {
        // Parity: input args must come before any operations.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        // This should panic.
        rec.record_input_arg(Type::Int);
    }

    // ══════════════════════════════════════════════════════════════════
    // History / TreeLoop parity tests
    // Ported from rpython/jit/metainterp/test/test_history.py
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_trace_has_inputargs_ops_structure() {
        // Parity: TreeLoop has inputargs and operations.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);

        rec.close_loop(&[sub, i1]);
        let trace = rec.get_trace();

        // inputargs
        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Int);

        // ops: IntAdd, IntSub, Jump
        assert_eq!(trace.num_ops(), 3);
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::IntSub);
        assert_eq!(trace.ops[2].opcode, OpCode::Jump);
    }

    #[test]
    fn test_trace_guards_have_fail_args() {
        // Parity: guards in a trace carry fail_args.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let cmp = rec.record_op(OpCode::IntLt, &[i0, i1]);
        let descr = make_fail_descr(0);
        rec.record_guard_with_fail_args(
            OpCode::GuardTrue,
            &[cmp],
            descr,
            &[i0, i1],
        );

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);

        let fail_args = guards[0].fail_args.as_ref().unwrap();
        assert_eq!(fail_args.len(), 2);
        assert_eq!(fail_args[0], i0);
        assert_eq!(fail_args[1], i1);
    }

    #[test]
    fn test_trace_iter_ops() {
        // Parity: can iterate over all ops.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.record_op(OpCode::IntSub, &[OpRef(1), i0]);
        rec.close_loop(&[OpRef(2)]);

        let trace = rec.get_trace();
        let opcodes: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(opcodes, vec![OpCode::IntAdd, OpCode::IntSub, OpCode::Jump]);
    }

    #[test]
    fn test_trace_mixed_types() {
        // Parity: traces can have mixed-type inputargs (int, ref, float).
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let r0 = rec.record_input_arg(Type::Ref);
        let f0 = rec.record_input_arg(Type::Float);

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.close_loop(&[i1, r0, f0]);

        let trace = rec.get_trace();
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Ref);
        assert_eq!(trace.inputargs[2].tp, Type::Float);
        assert!(trace.is_loop());
    }

    #[test]
    fn test_trace_pos_matches_opref() {
        // Parity: each op's .pos matches the OpRef returned by record_op.
        let mut rec = TraceRecorder::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let ref0 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let ref1 = rec.record_op(OpCode::IntMul, &[ref0, i1]);
        let ref2 = rec.record_op(OpCode::IntSub, &[ref1, ref0]);

        rec.close_loop(&[ref2, i1]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].pos, ref0);
        assert_eq!(trace.ops[1].pos, ref1);
        assert_eq!(trace.ops[2].pos, ref2);
    }
}
