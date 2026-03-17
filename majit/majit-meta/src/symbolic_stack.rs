use majit_ir::{OpRef, Type};

/// Symbolic stack for stack-based interpreters during tracing.
///
/// Tracks the OpRef and Type for each stack position. Push/pop mirror the
/// interpreter's stack operations but operate on symbolic values.
///
/// For backward compatibility with integer-only interpreters, `push(opref)`
/// defaults to `Type::Int`. Mixed-type interpreters should use `push_typed`.
pub struct SymbolicStack {
    entries: Vec<(OpRef, Type)>,
}

impl SymbolicStack {
    pub fn new() -> Self {
        SymbolicStack {
            entries: Vec::new(),
        }
    }

    /// Initialize from InputArg OpRefs (all typed as Int).
    /// `start` is the first OpRef index, `len` is the number of slots.
    pub fn from_input_args(start: usize, len: usize) -> Self {
        let entries = (start..start + len)
            .map(|i| (OpRef(i as u32), Type::Int))
            .collect();
        SymbolicStack { entries }
    }

    /// Initialize from InputArg OpRefs with explicit types.
    pub fn from_input_args_typed(start: usize, types: &[Type]) -> Self {
        let entries = types
            .iter()
            .enumerate()
            .map(|(i, &tp)| (OpRef((start + i) as u32), tp))
            .collect();
        SymbolicStack { entries }
    }

    /// Push a value with default type `Type::Int` (backward compatible).
    pub fn push(&mut self, opref: OpRef) {
        self.entries.push((opref, Type::Int));
    }

    /// Push a value with explicit type.
    pub fn push_typed(&mut self, opref: OpRef, tp: Type) {
        self.entries.push((opref, tp));
    }

    /// Pop a value, returning only the OpRef (backward compatible).
    pub fn pop(&mut self) -> Option<OpRef> {
        self.entries.pop().map(|(opref, _)| opref)
    }

    /// Pop a value with its type.
    pub fn pop_typed(&mut self) -> Option<(OpRef, Type)> {
        self.entries.pop()
    }

    /// Peek at the top, returning only the OpRef (backward compatible).
    pub fn peek(&self) -> Option<OpRef> {
        self.entries.last().map(|(opref, _)| *opref)
    }

    /// Peek at the top with its type.
    pub fn peek_typed(&self) -> Option<(OpRef, Type)> {
        self.entries.last().copied()
    }

    /// Peek at a given depth from the bottom (index 0 = bottom).
    pub fn peek_at(&self, index: usize) -> OpRef {
        self.entries[index].0
    }

    /// Peek at a given depth from the bottom with type info.
    pub fn peek_at_typed(&self, index: usize) -> (OpRef, Type) {
        self.entries[index]
    }

    /// Set the OpRef at a given position from the bottom (index 0 = bottom).
    /// Preserves the existing type tag.
    pub fn set_at(&mut self, index: usize, opref: OpRef) {
        self.entries[index].0 = opref;
    }

    pub fn swap(&mut self) {
        let len = self.entries.len();
        assert!(len >= 2, "swap requires at least 2 elements");
        self.entries.swap(len - 1, len - 2);
    }

    pub fn dup(&mut self) {
        let top = *self.entries.last().expect("dup on empty stack");
        self.entries.push(top);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Collect all OpRefs bottom-to-top (for jump_args, backward compatible).
    pub fn to_jump_args(&self) -> Vec<OpRef> {
        self.entries.iter().map(|(opref, _)| *opref).collect()
    }

    /// Collect all (OpRef, Type) pairs bottom-to-top.
    pub fn to_typed_jump_args(&self) -> Vec<(OpRef, Type)> {
        self.entries.clone()
    }
}

impl Default for SymbolicStack {
    fn default() -> Self {
        Self::new()
    }
}
