//! Narrow semantic graph scaffold for the future graph-based translator.
//!
//! This is intentionally much smaller than a full Rust compiler IR.  It exists
//! to model only the semantics needed by majit's translation/codewriter layer.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BasicBlockId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValueId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueType {
    Int,
    Ref,
    Float,
    Void,
    State,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpKind {
    Input {
        name: String,
        ty: ValueType,
    },
    ConstInt(i64),
    FieldRead {
        base: ValueId,
        field: String,
        ty: ValueType,
    },
    FieldWrite {
        base: ValueId,
        field: String,
        value: ValueId,
        ty: ValueType,
    },
    ArrayRead {
        base: ValueId,
        index: ValueId,
        item_ty: ValueType,
    },
    ArrayWrite {
        base: ValueId,
        index: ValueId,
        value: ValueId,
        item_ty: ValueType,
    },
    Call {
        target: String,
        args: Vec<ValueId>,
        result_ty: ValueType,
    },
    GuardTrue {
        cond: ValueId,
    },
    GuardFalse {
        cond: ValueId,
    },
    Unknown {
        summary: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Op {
    pub result: Option<ValueId>,
    pub kind: OpKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Terminator {
    Goto(BasicBlockId),
    Branch {
        cond: ValueId,
        if_true: BasicBlockId,
        if_false: BasicBlockId,
    },
    Return(Option<ValueId>),
    Abort {
        reason: String,
    },
    Unreachable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BasicBlock {
    pub id: BasicBlockId,
    pub ops: Vec<Op>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MajitGraph {
    pub name: String,
    pub entry: BasicBlockId,
    pub blocks: Vec<BasicBlock>,
    #[serde(default)]
    pub notes: Vec<String>,
    next_value: usize,
}

impl MajitGraph {
    pub fn new(name: impl Into<String>) -> Self {
        let entry = BasicBlockId(0);
        Self {
            name: name.into(),
            entry,
            blocks: vec![BasicBlock {
                id: entry,
                ops: Vec::new(),
                terminator: Terminator::Unreachable,
            }],
            notes: Vec::new(),
            next_value: 0,
        }
    }

    pub fn create_block(&mut self) -> BasicBlockId {
        let id = BasicBlockId(self.blocks.len());
        self.blocks.push(BasicBlock {
            id,
            ops: Vec::new(),
            terminator: Terminator::Unreachable,
        });
        id
    }

    pub fn alloc_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        id
    }

    pub fn push_op(
        &mut self,
        block: BasicBlockId,
        kind: OpKind,
        has_result: bool,
    ) -> Option<ValueId> {
        let result = has_result.then(|| self.alloc_value());
        self.blocks[block.0].ops.push(Op { result, kind });
        result
    }

    pub fn set_terminator(&mut self, block: BasicBlockId, terminator: Terminator) {
        self.blocks[block.0].terminator = terminator;
    }

    pub fn block(&self, block: BasicBlockId) -> &BasicBlock {
        &self.blocks[block.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_allocates_values_and_blocks() {
        let mut graph = MajitGraph::new("demo");
        let entry = graph.entry;
        let cond = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let next = graph.create_block();
        graph.set_terminator(
            entry,
            Terminator::Branch {
                cond,
                if_true: next,
                if_false: next,
            },
        );
        assert_eq!(graph.blocks.len(), 2);
        assert_eq!(graph.block(entry).ops.len(), 1);
    }
}
