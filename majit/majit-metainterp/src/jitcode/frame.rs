/// MIFrame and MIFrameStack — execution frames for jitcode interpretation.
///
/// RPython pyjitpl.py: class MIFrame holds jitcode reference, PC,
/// and three register arrays (int/ref/float). MIFrameStack manages
/// the call stack of nested inline calls.
use majit_ir::OpRef;

use super::JitCode;

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
        }
    }

    pub fn next_u8(&mut self) -> u8 {
        super::read_u8(&self.jitcode.code, &mut self.code_cursor)
    }

    pub fn next_u16(&mut self) -> u16 {
        super::read_u16(&self.jitcode.code, &mut self.code_cursor)
    }

    pub fn finished(&self) -> bool {
        self.code_cursor >= self.jitcode.code.len()
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
