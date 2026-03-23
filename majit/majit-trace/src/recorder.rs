/// Trace recorder — records IR operations during interpreter execution.
///
/// The recorder is the bridge between the interpreter and the JIT.
/// When tracing is active, every interpreter operation is fed to the
/// recorder, which builds a linear sequence of IR operations (a trace).
///
/// Reference: rpython/jit/metainterp/pyjitpl.py MetaInterp.record()
use majit_ir::{DescrRef, InputArg, Op, OpCode, OpRef, Type};

use crate::history::TreeLoop;

/// Default maximum number of operations before the trace is considered too long.
/// Mirrors the configurable warmstate `trace_limit` parameter.
pub const DEFAULT_TRACE_LIMIT: usize = 6000;

/// opencoder.py: cut_point() — snapshot of trace recorder position.
///
/// In RPython this is a 5-tuple `(_pos, _count, _index, snapshot_len, array_len)`.
/// majit's structured IR ops make byte-level tracking and snapshot buffers
/// unnecessary, so position reduces to op_count + ops.len().
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TracePosition {
    /// Next OpRef index at this position (Trace.op_count).
    pub op_count: u32,
    /// Number of ops in the ops vec at this position.
    pub ops_len: usize,
}

/// The trace recorder: accumulates operations during tracing.
pub struct Trace {
    /// Recorded operations.
    ops: Vec<Op>,
    /// Input arguments to the trace (live variables at the loop header).
    inputargs: Vec<InputArg>,
    /// Next OpRef index to assign.
    op_count: u32,
    /// Maximum number of operations allowed before this trace is too long.
    trace_limit: usize,
    /// Whether the recorder has been finalized (closed or finished).
    finalized: bool,
    /// Whether the trace was aborted.
    aborted: bool,
}

impl Trace {
    /// Create a new, empty trace recorder.
    pub fn new() -> Self {
        Self::with_limit(DEFAULT_TRACE_LIMIT)
    }

    /// Create a new, empty trace recorder with an explicit trace limit.
    pub fn with_limit(trace_limit: usize) -> Self {
        Trace {
            ops: Vec::with_capacity(256),
            inputargs: Vec::new(),
            op_count: 0,
            trace_limit,
            finalized: false,
            aborted: false,
        }
    }

    /// Create a trace recorder pre-configured for retracing from a guard.
    ///
    /// The recorder starts with `num_inputs` int-typed input args,
    /// matching the guard's fail_args.
    pub fn with_num_inputs(num_inputs: usize) -> Self {
        Self::with_num_inputs_and_limit(num_inputs, DEFAULT_TRACE_LIMIT)
    }

    /// Create a trace recorder pre-configured for retracing from a guard
    /// with an explicit trace limit.
    pub fn with_num_inputs_and_limit(num_inputs: usize, trace_limit: usize) -> Self {
        Self::with_input_types_and_limit(&vec![Type::Int; num_inputs], trace_limit)
    }

    /// Create a trace recorder pre-configured for retracing from a guard
    /// with explicit input arg types.
    pub fn with_input_types(input_types: &[Type]) -> Self {
        Self::with_input_types_and_limit(input_types, DEFAULT_TRACE_LIMIT)
    }

    /// Create a trace recorder pre-configured for retracing from a guard
    /// with explicit input arg types and an explicit trace limit.
    pub fn with_input_types_and_limit(input_types: &[Type], trace_limit: usize) -> Self {
        let mut recorder = Self::with_limit(trace_limit);
        for tp in input_types {
            recorder.record_input_arg(*tp);
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
        // RPython parity: Jump args may differ from InputArgs count when
        // virtualizable arrays change depth. The optimizer (OptUnroll preamble
        // peeling) bridges the gap by creating a Label with the extended count.
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

    /// Apply OpRef replacements to all recorded ops.
    /// Used by inline callee returns to replace placeholder OpRefs
    /// with actual result OpRefs throughout the trace.
    pub fn apply_replacements(&mut self, replacements: &std::collections::HashMap<OpRef, OpRef>) {
        for op in &mut self.ops {
            for arg in &mut op.args {
                if let Some(&new) = replacements.get(arg) {
                    *arg = new;
                }
            }
            if let Some(ref mut fa) = op.fail_args {
                for arg in fa.iter_mut() {
                    if let Some(&new) = replacements.get(arg) {
                        *arg = new;
                    }
                }
            }
        }
    }

    /// Return the completed trace.
    /// The recorder is consumed; no further operations can be recorded.
    pub fn get_trace(self) -> TreeLoop {
        assert!(
            self.finalized,
            "trace must be finalized with close_loop() or finish() before calling get_trace()"
        );
        assert!(!self.aborted, "cannot get trace from aborted recorder");
        TreeLoop::new(self.inputargs, self.ops)
    }

    /// Abort the current trace, discarding all recorded operations.
    pub fn abort(mut self) {
        self.aborted = true;
        self.ops.clear();
        self.inputargs.clear();
    }

    /// opencoder.py: cut_point() — snapshot the current recorder position.
    ///
    /// Used by compile_trace to save position before recording a JUMP,
    /// and by reached_loop_header to record merge points.
    pub fn get_position(&self) -> TracePosition {
        TracePosition {
            op_count: self.op_count,
            ops_len: self.ops.len(),
        }
    }

    /// opencoder.py: cut_at() — restore the recorder to a previously saved position.
    ///
    /// Discards all operations recorded after `pos`. Used to undo a
    /// tentative JUMP after compile_trace succeeds or fails.
    pub fn cut(&mut self, pos: TracePosition) {
        assert!(!self.finalized, "cannot cut a finalized trace");
        self.ops.truncate(pos.ops_len);
        self.op_count = pos.op_count;
    }

    /// Reset finalized flag so cut() can be called after close_loop().
    ///
    /// compile_trace records a tentative JUMP (close_loop), snapshots ops,
    /// then needs to undo the JUMP via cut(). close_loop sets finalized=true,
    /// so this must be called before cut() in that sequence.
    pub fn unfinalize(&mut self) {
        self.finalized = false;
    }

    /// Number of operations recorded so far (not counting input args).
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Number of input arguments registered.
    pub fn num_inputargs(&self) -> usize {
        self.inputargs.len()
    }

    /// Input argument types in loop-header order.
    pub fn inputarg_types(&self) -> Vec<Type> {
        self.inputargs.iter().map(|arg| arg.tp).collect()
    }

    /// Whether the trace has exceeded the maximum allowed length.
    pub fn is_too_long(&self) -> bool {
        self.ops.len() >= self.trace_limit
    }

    /// Whether the recorder has been finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Number of guards recorded so far.
    pub fn num_guards(&self) -> usize {
        self.ops.iter().filter(|op| op.opcode.is_guard()).count()
    }

    /// Access the recorded operations.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Get the last recorded operation, if any.
    pub fn last_op(&self) -> Option<&Op> {
        self.ops.last()
    }

    /// Get an operation by its OpRef position.
    pub fn get_op_by_pos(&self, pos: OpRef) -> Option<&Op> {
        self.ops.iter().find(|op| op.pos == pos)
    }

    /// Remaining capacity before the trace is too long.
    pub fn remaining_capacity(&self) -> usize {
        self.trace_limit.saturating_sub(self.ops.len())
    }

    /// Replace an argument in the last recorded operation.
    /// Used by the tracer to fix up arguments after recording.
    pub fn replace_last_arg(&mut self, arg_index: usize, new_ref: OpRef) {
        if let Some(last) = self.ops.last_mut() {
            if arg_index < last.args.len() {
                last.args[arg_index] = new_ref;
            }
        }
    }
}

impl Default for Trace {
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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Int);
        rec.record_op(OpCode::IntAdd, &[OpRef(0), OpRef(0)]);
        assert_eq!(rec.num_ops(), 1);
        rec.abort(); // should not panic
    }

    #[test]
    #[should_panic(expected = "recorder already finalized")]
    fn test_record_after_finalize() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.close_loop(&[i0]);
        // This should panic
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
    }

    #[test]
    #[should_panic(expected = "use record_guard")]
    fn test_record_op_with_guard_opcode() {
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Int);
        // Should panic: guard opcodes must use record_guard
        rec.record_op(OpCode::GuardTrue, &[OpRef(0)]);
    }

    #[test]
    fn test_is_too_long() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert!(!rec.is_too_long());

        // Record operations up to the limit
        let mut last = i0;
        for _ in 0..DEFAULT_TRACE_LIMIT {
            last = rec.record_op(OpCode::IntAdd, &[last, i0]);
        }
        assert!(rec.is_too_long());
    }

    // ── Trace limit parity tests (RPython: test_tracelimit.py) ──

    #[test]
    fn test_is_too_long_boundary() {
        // Verify exact boundary: DEFAULT_TRACE_LIMIT-1 ops is not too long,
        // DEFAULT_TRACE_LIMIT ops is too long.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let mut last = i0;
        for _ in 0..(DEFAULT_TRACE_LIMIT - 1) {
            last = rec.record_op(OpCode::IntAdd, &[last, i0]);
        }
        assert!(
            !rec.is_too_long(),
            "DEFAULT_TRACE_LIMIT - 1 ops should not be too long"
        );

        // One more op pushes it to the limit.
        let _over = rec.record_op(OpCode::IntAdd, &[last, i0]);
        assert!(
            rec.is_too_long(),
            "DEFAULT_TRACE_LIMIT ops should be too long"
        );
    }

    #[test]
    fn test_trace_abort_on_too_long() {
        // Simulate what the meta-interpreter does when a trace is too long:
        // check is_too_long() then call abort().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let mut last = i0;
        for _ in 0..DEFAULT_TRACE_LIMIT {
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
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let mut last = i0;
        for _ in 0..(DEFAULT_TRACE_LIMIT + 100) {
            last = rec.record_op(OpCode::IntAdd, &[last, i0]);
        }
        assert!(rec.is_too_long());
        assert_eq!(rec.num_ops(), DEFAULT_TRACE_LIMIT + 100);

        // Can still finalize if we want to (the limit is advisory).
        rec.close_loop(&[last]);
        let trace = rec.get_trace();
        // The trace contains all ops, including those beyond the limit.
        assert!(trace.num_ops() > DEFAULT_TRACE_LIMIT);
    }

    #[test]
    fn test_fresh_recorder_not_too_long() {
        // A freshly created recorder should never be too long.
        let rec = Trace::new();
        assert!(!rec.is_too_long());
        assert_eq!(rec.num_ops(), 0);
    }

    #[test]
    fn test_retrace_recorder_not_too_long() {
        // A recorder created for retracing (with_num_inputs) starts with
        // no operations, so it should not be too long.
        let rec = Trace::with_num_inputs(5);
        assert!(!rec.is_too_long());
        assert_eq!(rec.num_ops(), 0);
        assert_eq!(rec.num_inputargs(), 5);
    }

    #[test]
    fn test_record_op_with_descr() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let descr = make_fail_descr(42);
        let i1 = rec.record_op_with_descr(OpCode::CallI, &[i0], descr);
        assert_eq!(i1, OpRef(1));

        // Verify the op has a descriptor
        assert!(rec.ops[0].descr.is_some());
    }

    #[test]
    fn test_complex_trace() {
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);

        rec.close_loop(&[sub]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[0].pos, OpRef(1)); // after 1 inputarg
        assert_eq!(trace.ops[1].opcode, OpCode::IntSub);
        assert_eq!(trace.ops[1].args[0], add); // references the add result
        assert_eq!(trace.ops[1].args[1], i0); // references the input arg
    }

    #[test]
    fn test_guard_with_fail_args() {
        // Parity: guards can carry fail_args describing live values at guard.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let descr = make_fail_descr(0);
        let guard =
            rec.record_guard_with_fail_args(OpCode::GuardTrue, &[add], descr, &[i0, i1, add]);

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
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let descr0 = make_fail_descr(0);
        rec.record_guard_with_fail_args(OpCode::GuardTrue, &[i0], descr0, &[i0, i1]);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);

        let descr1 = make_fail_descr(1);
        rec.record_guard_with_fail_args(OpCode::GuardFalse, &[add], descr1, &[i0, add]);

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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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
        // Parity: Trace::with_num_inputs pre-creates input args.
        let rec = Trace::with_num_inputs(3);
        assert_eq!(rec.num_inputargs(), 3);
        assert_eq!(rec.num_ops(), 0);
    }

    #[test]
    fn test_with_num_inputs_oprefs() {
        // Input args from with_num_inputs get OpRef(0), OpRef(1), ...
        let mut rec = Trace::with_num_inputs(3);

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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(0);
        rec.record_guard_with_fail_args(OpCode::GuardTrue, &[i0], descr, &[]);

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
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Int);
        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::IntAdd, &[OpRef(0)], descr);
    }

    #[test]
    #[should_panic(expected = "input args must be registered before any operations")]
    fn test_inputarg_after_ops() {
        // Parity: input args must come before any operations.
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let cmp = rec.record_op(OpCode::IntLt, &[i0, i1]);
        let descr = make_fail_descr(0);
        rec.record_guard_with_fail_args(OpCode::GuardTrue, &[cmp], descr, &[i0, i1]);

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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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
        let mut rec = Trace::new();
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

    // ══════════════════════════════════════════════════════════════════
    // Opencoder breadth tests — deeper parity with test_opencoder.py
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_recorder_const_int_via_constant_oprefs() {
        // Constants in majit use OpRef indices >= 10_000.
        // Recording ops that reference constants should preserve them.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        // Simulate constant references: OpRef(10_000) is a const
        let const_ref = OpRef(10_000);
        let add = rec.record_op(OpCode::IntAdd, &[i0, const_ref]);

        rec.close_loop(&[add]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].args[1], const_ref);
        // Constant OpRef is preserved through recording
        assert!(trace.ops[0].args[1].0 >= 10_000);
    }

    #[test]
    fn test_recorder_const_deduplication_by_opref() {
        // Two ops referencing the same constant OpRef share the same value.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let const_ref = OpRef(10_001);
        let add1 = rec.record_op(OpCode::IntAdd, &[i0, const_ref]);
        let add2 = rec.record_op(OpCode::IntAdd, &[add1, const_ref]);

        rec.close_loop(&[add2]);
        let trace = rec.get_trace();

        // Both ops reference the same constant
        assert_eq!(trace.ops[0].args[1], trace.ops[1].args[1]);
        assert_eq!(trace.ops[0].args[1], const_ref);
    }

    #[test]
    fn test_recorder_descriptors_preserved_on_ops() {
        // Descriptors on non-guard ops should survive through get_trace().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let descr1 = make_fail_descr(10);
        let descr2 = make_fail_descr(20);

        let call1 = rec.record_op_with_descr(OpCode::CallI, &[i0], descr1);
        let call2 = rec.record_op_with_descr(OpCode::CallI, &[i1], descr2);

        rec.close_loop(&[call1, call2]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].descr.as_ref().unwrap().index(), 10);
        assert_eq!(trace.ops[1].descr.as_ref().unwrap().index(), 20);
    }

    #[test]
    fn test_recorder_100_plus_ops_stress() {
        // Recording 200 ops should work without issue.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let mut prev = i0;
        for _ in 0..200 {
            prev = rec.record_op(OpCode::IntAdd, &[prev, i0]);
        }
        assert_eq!(rec.num_ops(), 200);
        assert!(!rec.is_too_long());

        rec.close_loop(&[prev]);
        let trace = rec.get_trace();
        // 200 IntAdd + 1 Jump
        assert_eq!(trace.num_ops(), 201);
        assert!(trace.is_loop());

        // Verify chain: each op references the previous op's result.
        // i0 = OpRef(0), first IntAdd = OpRef(1), second = OpRef(2), ...
        for (i, op) in trace.ops[..200].iter().enumerate() {
            assert_eq!(op.opcode, OpCode::IntAdd);
            if i > 0 {
                // The previous IntAdd produced OpRef(i as u32) (offset by inputarg)
                assert_eq!(op.args[0], OpRef(i as u32));
            }
        }
    }

    #[test]
    fn test_recorder_mixed_type_input_args() {
        // Mixed-type inputs (Int, Float, Ref) produce correct types.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let f0 = rec.record_input_arg(Type::Float);
        let r0 = rec.record_input_arg(Type::Ref);
        let i1 = rec.record_input_arg(Type::Int);

        assert_eq!(i0, OpRef(0));
        assert_eq!(f0, OpRef(1));
        assert_eq!(r0, OpRef(2));
        assert_eq!(i1, OpRef(3));

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, f0, r0, i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Float);
        assert_eq!(trace.inputargs[2].tp, Type::Ref);
        assert_eq!(trace.inputargs[3].tp, Type::Int);
    }

    #[test]
    fn test_recorder_abort_clears_state() {
        // After abort(), the recorder's ops and inputargs are cleared.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.record_op(OpCode::IntSub, &[OpRef(1), i0]);
        assert_eq!(rec.num_ops(), 2);
        assert_eq!(rec.num_inputargs(), 1);

        rec.abort();
        // After abort, the recorder is consumed; create a fresh one to verify
        // independent state.
        let fresh = Trace::new();
        assert_eq!(fresh.num_ops(), 0);
        assert_eq!(fresh.num_inputargs(), 0);
    }

    #[test]
    #[should_panic(expected = "cannot get trace from aborted recorder")]
    fn test_recorder_abort_prevents_get_trace() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        // Manually mark as finalized+aborted to trigger panic
        // Actually, abort() consumes self, so we need a different approach:
        // abort() sets aborted=true and clears ops, but also consumes self.
        // So we can't call get_trace() after abort(). Let's verify the panic
        // path by constructing a finalized-then-aborted scenario.
        // Since abort() consumes self, this test verifies that the assertion
        // text is correct by using a different flow:
        let mut rec2 = Trace::new();
        let i0 = rec2.record_input_arg(Type::Int);
        rec2.close_loop(&[i0]);
        // Manually break the invariant for testing the assertion message
        let mut rec3 = Trace {
            ops: vec![],
            inputargs: vec![InputArg::from_type(Type::Int, 0)],
            op_count: 1,
            finalized: true,
            aborted: true,
            trace_limit: DEFAULT_TRACE_LIMIT,
        };
        rec3.get_trace(); // should panic
    }

    #[test]
    fn test_recorder_guard_with_many_fail_args() {
        // Guard with 10+ fail_args.
        let mut rec = Trace::new();
        let mut inputs = Vec::new();
        for _ in 0..12 {
            inputs.push(rec.record_input_arg(Type::Int));
        }

        // Record some ops
        let add = rec.record_op(OpCode::IntAdd, &[inputs[0], inputs[1]]);

        // Guard with all 12 inputs + 1 computed value as fail_args (13 total)
        let mut fail_args: Vec<OpRef> = inputs.clone();
        fail_args.push(add);

        let descr = make_fail_descr(0);
        let guard = rec.record_guard_with_fail_args(OpCode::GuardTrue, &[add], descr, &fail_args);

        rec.close_loop(&inputs);
        let trace = rec.get_trace();

        // Find the guard
        let guard_op = trace.iter_guards().next().unwrap();
        assert_eq!(guard_op.pos, guard);
        let fa = guard_op.fail_args.as_ref().unwrap();
        assert_eq!(fa.len(), 13);

        // Verify all fail_args match what we specified
        for (i, &expected) in fail_args.iter().enumerate() {
            assert_eq!(fa[i], expected);
        }
    }

    #[test]
    fn test_recorder_guard_descr_index_survives() {
        // Each guard's descriptor index must survive through get_trace().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let d0 = make_fail_descr(100);
        let d1 = make_fail_descr(200);
        let d2 = make_fail_descr(300);

        rec.record_guard(OpCode::GuardTrue, &[i0], d0);
        rec.record_guard(OpCode::GuardFalse, &[i0], d1);
        rec.record_guard(OpCode::GuardNoException, &[], d2);

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 3);
        assert_eq!(guards[0].descr.as_ref().unwrap().index(), 100);
        assert_eq!(guards[1].descr.as_ref().unwrap().index(), 200);
        assert_eq!(guards[2].descr.as_ref().unwrap().index(), 300);
    }

    #[test]
    fn test_recorder_ops_interleaved_with_guards() {
        // Interleaved ops and guards: verify ordering is correct.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let d0 = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[add], d0);
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        let d1 = make_fail_descr(1);
        rec.record_guard(OpCode::GuardFalse, &[sub], d1);
        let mul = rec.record_op(OpCode::IntMul, &[sub, i1]);

        rec.close_loop(&[mul, i1]);
        let trace = rec.get_trace();

        let expected = vec![
            OpCode::IntAdd,
            OpCode::GuardTrue,
            OpCode::IntSub,
            OpCode::GuardFalse,
            OpCode::IntMul,
            OpCode::Jump,
        ];
        let actual: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_recorder_finish_with_descr() {
        // finish() records a FINISH op with the given descriptor.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(999);
        rec.finish(&[add], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());

        let finish = trace.ops.last().unwrap();
        assert_eq!(finish.opcode, OpCode::Finish);
        assert_eq!(finish.descr.as_ref().unwrap().index(), 999);
    }

    #[test]
    fn test_recorder_no_ops_just_inputargs_and_jump() {
        // Minimal trace: input args directly jump back.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        rec.close_loop(&[i0, i1]);
        let trace = rec.get_trace();

        assert_eq!(trace.num_ops(), 1); // Only Jump
        assert_eq!(trace.ops[0].opcode, OpCode::Jump);
        assert_eq!(trace.ops[0].args[0], i0);
        assert_eq!(trace.ops[0].args[1], i1);
    }

    // ══════════════════════════════════════════════════════════════════
    // Trace limit deeper coverage (RPython: test_tracelimit.py parity)
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_trace_limit_independent_per_recorder() {
        // Each Trace has its own independent op count.
        // Two recorders can be at different fill levels simultaneously.
        let mut rec_a = Trace::new();
        let a0 = rec_a.record_input_arg(Type::Int);

        let mut rec_b = Trace::new();
        let b0 = rec_b.record_input_arg(Type::Int);

        // Fill rec_a to 100 ops
        let mut last_a = a0;
        for _ in 0..100 {
            last_a = rec_a.record_op(OpCode::IntAdd, &[last_a, a0]);
        }
        assert_eq!(rec_a.num_ops(), 100);
        assert!(!rec_a.is_too_long());

        // rec_b should still be empty
        assert_eq!(rec_b.num_ops(), 0);
        assert!(!rec_b.is_too_long());

        // Fill rec_b past the limit
        let mut last_b = b0;
        for _ in 0..DEFAULT_TRACE_LIMIT {
            last_b = rec_b.record_op(OpCode::IntAdd, &[last_b, b0]);
        }
        assert!(rec_b.is_too_long());
        assert!(!rec_a.is_too_long(), "rec_a should be unaffected");

        // Both can still finalize
        rec_a.close_loop(&[last_a]);
        let _ = rec_a.get_trace();

        rec_b.close_loop(&[last_b]);
        let _ = rec_b.get_trace();
    }

    #[test]
    fn test_trace_limit_guards_count_toward_limit() {
        // Guards count toward the trace limit just like regular ops.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let mut last = i0;
        for idx in 0..(DEFAULT_TRACE_LIMIT / 2) {
            last = rec.record_op(OpCode::IntAdd, &[last, i0]);
            let descr = make_fail_descr(idx as u32);
            rec.record_guard(OpCode::GuardTrue, &[last], descr);
        }

        // Each iteration adds 2 ops (IntAdd + GuardTrue), total = DEFAULT_TRACE_LIMIT
        assert!(
            rec.is_too_long(),
            "guards should count toward trace limit: num_ops={}",
            rec.num_ops()
        );
    }

    #[test]
    fn test_retrace_recorder_independent_limit() {
        // A retrace recorder (from guard failure) starts fresh and has
        // its own independent trace limit, simulating RPython's behavior
        // where a bridge trace has its own length budget.
        let mut rec = Trace::with_num_inputs(3);
        assert!(!rec.is_too_long());
        assert_eq!(rec.num_ops(), 0);

        // The retrace recorder can record up to DEFAULT_TRACE_LIMIT ops
        let mut last = OpRef(0);
        for _ in 0..(DEFAULT_TRACE_LIMIT - 1) {
            last = rec.record_op(OpCode::IntAdd, &[last, OpRef(1)]);
        }
        assert!(
            !rec.is_too_long(),
            "retrace with DEFAULT_TRACE_LIMIT-1 ops should not be too long"
        );

        let _ = rec.record_op(OpCode::IntAdd, &[last, OpRef(1)]);
        assert!(
            rec.is_too_long(),
            "retrace with DEFAULT_TRACE_LIMIT ops should be too long"
        );
    }

    #[test]
    fn test_trace_limit_abort_preserves_inputargs_count() {
        // After abort, the recorder is consumed. The caller should be
        // able to create a new recorder with the same number of inputs.
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Int);
        rec.record_input_arg(Type::Float);
        let input_count = rec.num_inputargs();
        assert_eq!(input_count, 2);

        let i0 = OpRef(0);
        let mut last = i0;
        for _ in 0..DEFAULT_TRACE_LIMIT {
            last = rec.record_op(OpCode::IntAdd, &[last, i0]);
        }
        assert!(rec.is_too_long());

        // Abort and create fresh recorder with same inputs
        rec.abort();
        let new_rec = Trace::with_num_inputs(input_count);
        assert_eq!(new_rec.num_inputargs(), input_count);
        assert!(!new_rec.is_too_long());
        assert_eq!(new_rec.num_ops(), 0);
    }

    #[test]
    fn test_get_position_and_cut() {
        let mut rec = Trace::with_num_inputs(2);
        let pos0 = rec.get_position();
        assert_eq!(pos0.ops_len, 0);
        assert_eq!(pos0.op_count, 2); // 2 inputargs

        let _a = rec.record_op(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
        let pos1 = rec.get_position();
        assert_eq!(pos1.ops_len, 1);
        assert_eq!(pos1.op_count, 3);

        let _b = rec.record_op(OpCode::IntSub, &[OpRef(0), OpRef(1)]);
        let _c = rec.record_op(OpCode::IntMul, &[OpRef(0), OpRef(1)]);
        assert_eq!(rec.num_ops(), 3);

        // Cut back to pos1 — should discard IntSub and IntMul
        rec.cut(pos1);
        assert_eq!(rec.num_ops(), 1);
        assert_eq!(rec.get_position(), pos1);

        // Can record more ops after cut
        let d = rec.record_op(OpCode::IntNeg, &[OpRef(0)]);
        assert_eq!(d, OpRef(3)); // continues from pos1.op_count
        assert_eq!(rec.num_ops(), 2);

        // Cut back to pos0 — should discard everything
        rec.cut(pos0);
        assert_eq!(rec.num_ops(), 0);
    }
}
