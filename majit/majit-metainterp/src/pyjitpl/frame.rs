/// MIFrame and MIFrameStack — execution frames for jitcode interpretation.
///
/// RPython pyjitpl.py: class MIFrame holds jitcode reference, PC,
/// and three register arrays (int/ref/float). MIFrameStack manages
/// the call stack of nested inline calls.
use majit_ir::OpRef;

use crate::jitcode::{JitArgKind, JitCode, read_u8, read_u16};

/// A single execution frame for jitcode bytecode.
///
/// RPython pyjitpl.py: class MIFrame
pub struct MIFrame<'a> {
    pub jitcode: &'a JitCode,
    pub pc: usize,
    pub code_cursor: usize,
    pub int_regs: Vec<Option<OpRef>>,
    pub int_values: Vec<Option<i64>>,
    pub ref_regs: Vec<Option<OpRef>>,
    pub ref_values: Vec<Option<i64>>,
    pub float_regs: Vec<Option<OpRef>>,
    pub float_values: Vec<Option<i64>>,
    pub inline_frame: bool,
    pub return_i: Option<(usize, usize)>,
    pub return_r: Option<(usize, usize)>,
    pub return_f: Option<(usize, usize)>,
    /// pyjitpl.py `MIFrame.pushed_box` — the box that the previous
    /// `_opimpl_any_push` instruction parked on the frame.  Reset to
    /// `None` by `cleanup_registers` so a recycled frame does not keep
    /// the parked box alive.
    pub pushed_box: Option<OpRef>,
}

impl<'a> MIFrame<'a> {
    pub fn new(jitcode: &'a JitCode, pc: usize) -> Self {
        Self {
            jitcode,
            pc,
            code_cursor: 0,
            int_regs: vec![None; jitcode.c_num_regs_i as usize],
            int_values: vec![None; jitcode.c_num_regs_i as usize],
            ref_regs: vec![None; jitcode.c_num_regs_r as usize],
            ref_values: vec![None; jitcode.c_num_regs_r as usize],
            float_regs: vec![None; jitcode.c_num_regs_f as usize],
            float_values: vec![None; jitcode.c_num_regs_f as usize],
            inline_frame: false,
            return_i: None,
            return_r: None,
            return_f: None,
            pushed_box: None,
        }
    }

    pub fn next_u8(&mut self) -> u8 {
        read_u8(&self.jitcode.code, &mut self.code_cursor)
    }

    pub fn next_u16(&mut self) -> u16 {
        read_u16(&self.jitcode.code, &mut self.code_cursor)
    }

    pub fn finished(&self) -> bool {
        self.code_cursor >= self.jitcode.code.len()
    }

    /// pyjitpl.py:121-127 `MIFrame.cleanup_registers()`.
    ///
    /// ```python
    /// def cleanup_registers(self):
    ///     for i in range(self.jitcode.num_regs_r()):
    ///         self.registers_r[i] = None
    ///     self.pushed_box = None
    /// ```
    ///
    /// Iterates `0..num_regs_r()` (RPython skips the constants area
    /// that lives past `num_regs_r`); pyre's `ref_regs` is sized
    /// exactly to `num_regs_r` so the loop scans the same slots.
    /// `ref_values` is cleared in lockstep — it is the pyre-only
    /// concrete-value mirror that lives next to each box.
    pub fn cleanup_registers(&mut self) {
        let num_regs_r = self.jitcode.num_regs_r() as usize;
        for i in 0..num_regs_r {
            self.ref_regs[i] = None;
            self.ref_values[i] = None;
        }
        self.pushed_box = None;
    }

    /// pyjitpl.py:1878-1879 `MIFrame.setup_resume_at_op(pc)`.
    pub fn setup_resume_at_op(&mut self, pc: usize) {
        self.pc = pc;
    }

    /// pyjitpl.py:258-275 `MIFrame.make_result_of_lastop(resultbox)`.
    ///
    /// Stores the result of the last opimpl into the typed register at
    /// `target_index`. RPython reads `target_index = ord(self.bytecode[self.pc-1])`
    /// from the bytecode; pyre's call BC encodes `dst` explicitly so
    /// callers pass it directly.
    pub fn make_result_of_lastop(
        &mut self,
        kind: JitArgKind,
        target_index: usize,
        opref: OpRef,
        concrete: i64,
    ) {
        match kind {
            JitArgKind::Int => {
                self.int_regs[target_index] = Some(opref);
                self.int_values[target_index] = Some(concrete);
            }
            JitArgKind::Ref => {
                self.ref_regs[target_index] = Some(opref);
                self.ref_values[target_index] = Some(concrete);
            }
            JitArgKind::Float => {
                self.float_regs[target_index] = Some(opref);
                self.float_values[target_index] = Some(concrete);
            }
        }
    }

    /// pyjitpl.py:1862-1876 `MIFrame.setup_call(argboxes)`.
    ///
    /// Resets `pc` to 0 and copies each argbox into the first slot of
    /// its typed register bank in declaration order. RPython's
    /// `setup_call` consults `box.type`; pyre's `OpRef` does not carry
    /// type info, so the caller passes a typed `(kind, value, concrete)`
    /// tuple per arg.
    pub fn setup_call(&mut self, argboxes: &[(JitArgKind, OpRef, i64)]) {
        self.pc = 0;
        let mut count_i = 0;
        let mut count_r = 0;
        let mut count_f = 0;
        for (kind, value, concrete) in argboxes {
            match kind {
                JitArgKind::Int => {
                    self.int_regs[count_i] = Some(*value);
                    self.int_values[count_i] = Some(*concrete);
                    count_i += 1;
                }
                JitArgKind::Ref => {
                    self.ref_regs[count_r] = Some(*value);
                    self.ref_values[count_r] = Some(*concrete);
                    count_r += 1;
                }
                JitArgKind::Float => {
                    self.float_regs[count_f] = Some(*value);
                    self.float_values[count_f] = Some(*concrete);
                    count_f += 1;
                }
            }
        }
    }
}

/// RPython pyjitpl.py: MetaInterp.framestack
pub struct MIFrameStack<'a> {
    pub(crate) frames: Vec<MIFrame<'a>>,
}

impl<'a> MIFrameStack<'a> {
    pub fn new(root: MIFrame<'a>) -> Self {
        Self { frames: vec![root] }
    }

    pub fn current_mut(&mut self) -> &mut MIFrame<'a> {
        self.frames.last_mut().expect("empty JitCode frame stack")
    }

    pub fn push(&mut self, frame: MIFrame<'a>) {
        self.frames.push(frame);
    }

    pub fn pop(&mut self) -> Option<MIFrame<'a>> {
        self.frames.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitcode::JitCodeBuilder;
    use majit_ir::OpRef;

    fn make_jitcode_with_regs(num_i: u16, num_r: u16, num_f: u16) -> JitCode {
        let mut builder = JitCodeBuilder::new();
        for i in 0..num_i {
            builder.load_const_i_value(i, 0);
        }
        for i in 0..num_r {
            builder.load_const_r_value(i, 0);
        }
        for i in 0..num_f {
            builder.load_const_f_value(i, 0);
        }
        builder.finish()
    }

    #[test]
    fn setup_call_distributes_argboxes_by_kind_in_declaration_order() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(&jitcode, 5);
        frame.pc = 99;
        frame.setup_call(&[
            (JitArgKind::Int, OpRef(10), 100),
            (JitArgKind::Ref, OpRef(20), 200),
            (JitArgKind::Int, OpRef(11), 101),
            (JitArgKind::Float, OpRef(30), 300),
            (JitArgKind::Ref, OpRef(21), 201),
        ]);

        assert_eq!(frame.pc, 0);
        assert_eq!(frame.int_regs[0], Some(OpRef(10)));
        assert_eq!(frame.int_values[0], Some(100));
        assert_eq!(frame.int_regs[1], Some(OpRef(11)));
        assert_eq!(frame.int_values[1], Some(101));
        assert_eq!(frame.ref_regs[0], Some(OpRef(20)));
        assert_eq!(frame.ref_values[0], Some(200));
        assert_eq!(frame.ref_regs[1], Some(OpRef(21)));
        assert_eq!(frame.ref_values[1], Some(201));
        assert_eq!(frame.float_regs[0], Some(OpRef(30)));
        assert_eq!(frame.float_values[0], Some(300));
    }

    #[test]
    fn cleanup_registers_clears_ref_slots_and_pushed_box() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(&jitcode, 0);
        frame.int_regs[0] = Some(OpRef(1));
        frame.int_values[0] = Some(11);
        frame.ref_regs[0] = Some(OpRef(2));
        frame.ref_values[0] = Some(22);
        frame.float_regs[0] = Some(OpRef(3));
        frame.float_values[0] = Some(33);
        frame.pushed_box = Some(OpRef(99));

        frame.cleanup_registers();

        // pyjitpl.py:121-127: int and float slots are untouched.
        assert_eq!(frame.int_regs[0], Some(OpRef(1)));
        assert_eq!(frame.int_values[0], Some(11));
        assert_eq!(frame.float_regs[0], Some(OpRef(3)));
        assert_eq!(frame.float_values[0], Some(33));
        // pyjitpl.py:124-126: ref slots [0, num_regs_r()) are cleared.
        assert!(frame.ref_regs.iter().all(|r| r.is_none()));
        assert!(frame.ref_values.iter().all(|v| v.is_none()));
        // pyjitpl.py:127: pushed_box is reset to None.
        assert_eq!(frame.pushed_box, None);
    }

    #[test]
    fn make_result_of_lastop_stores_into_typed_slot() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(&jitcode, 0);

        frame.make_result_of_lastop(JitArgKind::Int, 1, OpRef(7), 77);
        assert_eq!(frame.int_regs[1], Some(OpRef(7)));
        assert_eq!(frame.int_values[1], Some(77));

        frame.make_result_of_lastop(JitArgKind::Ref, 0, OpRef(8), 88);
        assert_eq!(frame.ref_regs[0], Some(OpRef(8)));
        assert_eq!(frame.ref_values[0], Some(88));

        frame.make_result_of_lastop(JitArgKind::Float, 0, OpRef(9), 99);
        assert_eq!(frame.float_regs[0], Some(OpRef(9)));
        assert_eq!(frame.float_values[0], Some(99));
    }

    #[test]
    fn setup_resume_at_op_assigns_pc() {
        let jitcode = make_jitcode_with_regs(0, 0, 0);
        let mut frame = MIFrame::new(&jitcode, 0);
        frame.setup_resume_at_op(123);
        assert_eq!(frame.pc, 123);
    }

    #[test]
    fn setup_call_with_empty_argboxes_only_resets_pc() {
        let jitcode = make_jitcode_with_regs(1, 1, 1);
        let mut frame = MIFrame::new(&jitcode, 5);
        frame.pc = 42;
        frame.setup_call(&[]);
        assert_eq!(frame.pc, 0);
        assert!(frame.int_regs.iter().all(|r| r.is_none()));
        assert!(frame.ref_regs.iter().all(|r| r.is_none()));
        assert!(frame.float_regs.iter().all(|r| r.is_none()));
    }
}
