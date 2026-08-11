//! Helpers for downstream tests that need to construct backend input traces.

use crate::operand::Operand;
use crate::{DescrRef, InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, Type};

pub struct RecordedTrace {
    pub inputargs: Vec<InputArgRc>,
    pub ops: Vec<OpRc>,
}

pub struct Trace {
    inputargs: Vec<InputArgRc>,
    ops: Vec<OpRc>,
    next_position: u32,
}

impl Trace {
    pub fn new() -> Self {
        Self {
            inputargs: Vec::new(),
            ops: Vec::new(),
            next_position: 0,
        }
    }

    pub fn record_input_arg(&mut self, tp: Type) -> OpRef {
        assert!(self.ops.is_empty(), "input args must precede operations");
        assert_ne!(tp, Type::Void, "input args cannot be void");

        let position = self.next_position;
        self.inputargs.push(InputArg::from_type_rc(tp, position));
        self.next_position += 1;
        OpRef::input_arg_typed(position, tp)
    }

    pub fn record_op(&mut self, opcode: OpCode, args: &[OpRef]) -> OpRef {
        assert!(!opcode.is_guard(), "use record_guard for guards");
        self.push_op(opcode, args, None, None)
    }

    pub fn record_op_with_descr(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        assert!(!opcode.is_guard(), "use record_guard for guards");
        self.push_op(opcode, args, Some(descr), None)
    }

    pub fn record_guard(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
    ) -> OpRef {
        assert!(opcode.is_guard(), "opcode is not a guard");
        self.push_op(opcode, args, descr, None)
    }

    pub fn record_guard_with_fail_args(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
        fail_args: &[OpRef],
    ) -> OpRef {
        assert!(opcode.is_guard(), "opcode is not a guard");
        self.push_op(opcode, args, descr, Some(fail_args))
    }

    pub fn close_loop(&mut self, args: &[OpRef]) {
        self.push_op(OpCode::Jump, args, None, None);
    }

    pub fn finish(&mut self, args: &[OpRef], descr: DescrRef) {
        self.push_op(OpCode::Finish, args, Some(descr), None);
    }

    pub fn get_trace(self) -> RecordedTrace {
        RecordedTrace {
            inputargs: self.inputargs,
            ops: self.ops,
        }
    }

    fn push_op(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
        fail_args: Option<&[OpRef]>,
    ) -> OpRef {
        let position = self.next_position;
        let opref = OpRef::op_typed(position, opcode.result_type());
        let args = self.bind_operands(args);
        let op = match descr {
            Some(descr) => Op::with_descr(opcode, &args, descr),
            None => Op::new(opcode, &args),
        };
        op.pos.set(opref);
        if let Some(fail_args) = fail_args {
            op.setfailargs(self.bind_operands(fail_args).into_iter().collect());
        }
        self.ops.push(OpRc::new(op));
        self.next_position += 1;
        opref
    }

    fn bind_operands(&self, refs: &[OpRef]) -> Vec<Operand> {
        refs.iter().map(|&opref| self.bind_operand(opref)).collect()
    }

    fn bind_operand(&self, opref: OpRef) -> Operand {
        if opref.is_none() || opref.is_constant() {
            return Operand::from_opref(opref);
        }
        if matches!(
            opref,
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
        ) {
            let inputarg = self
                .inputargs
                .get(opref.raw() as usize)
                .expect("input operand must name a recorded input argument");
            return Operand::from_bound_inputarg(inputarg);
        }

        let op_index = (opref.raw() as usize)
            .checked_sub(self.inputargs.len())
            .expect("operation operand must follow the input arguments");
        let op = self
            .ops
            .get(op_index)
            .expect("operation operand must name a recorded operation");
        assert_eq!(op.pos.get(), opref, "operation operand type must match");
        Operand::from_bound_op(op)
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}
