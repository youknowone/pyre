use majit_ir::OpRef;

/// Symbolic stack for stack-based interpreters during tracing.
///
/// Tracks the OpRef for each stack position. Push/pop mirror the
/// interpreter's stack operations but operate on symbolic values.
pub struct SymbolicStack {
    oprefs: Vec<OpRef>,
}

impl SymbolicStack {
    pub fn new() -> Self {
        SymbolicStack { oprefs: Vec::new() }
    }

    /// Initialize from InputArg OpRefs.
    /// `start` is the first OpRef index, `len` is the number of slots.
    pub fn from_input_args(start: usize, len: usize) -> Self {
        let oprefs = (start..start + len).map(|i| OpRef(i as u32)).collect();
        SymbolicStack { oprefs }
    }

    pub fn push(&mut self, opref: OpRef) {
        self.oprefs.push(opref);
    }

    pub fn pop(&mut self) -> Option<OpRef> {
        self.oprefs.pop()
    }

    pub fn peek(&self) -> Option<OpRef> {
        self.oprefs.last().copied()
    }

    /// Peek at a given depth from the bottom (index 0 = bottom).
    pub fn peek_at(&self, index: usize) -> OpRef {
        self.oprefs[index]
    }

    pub fn swap(&mut self) {
        let len = self.oprefs.len();
        assert!(len >= 2, "swap requires at least 2 elements");
        self.oprefs.swap(len - 1, len - 2);
    }

    pub fn dup(&mut self) {
        let top = *self.oprefs.last().expect("dup on empty stack");
        self.oprefs.push(top);
    }

    pub fn len(&self) -> usize {
        self.oprefs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.oprefs.is_empty()
    }

    /// Collect all OpRefs bottom-to-top (for jump_args).
    pub fn to_jump_args(&self) -> Vec<OpRef> {
        self.oprefs.clone()
    }
}

impl Default for SymbolicStack {
    fn default() -> Self {
        Self::new()
    }
}
