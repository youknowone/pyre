//! JitCode assembler — scaffold for converting SSARepr to JitCode bytecode.
//!
//! RPython equivalent: `rpython/jit/codewriter/assembler.py` class `Assembler`.
//!
//! **Status: scaffold only.** Currently produces metadata-only JitCode (register
//! counts, name). Full bytecode encoding (`write_insn`, `fix_labels`, liveness
//! encoding) is not yet implemented — RPython assembler.py does ~300 lines of
//! instruction encoding that this module does not replicate.

use std::collections::HashMap;

use crate::graph::ValueId;
use crate::passes::flatten::{FlatOp, FlattenedFunction, Label, RegKind};
use crate::regalloc::RegAllocResult;

/// Assembled JitCode — the output of the assembler.
///
/// RPython: `jitcode.py::JitCode` — contains bytecode, constants, and
/// register counts for the meta-interpreter to execute.
#[derive(Debug, Clone)]
pub struct JitCode {
    /// RPython: JitCode.name
    pub name: String,
    /// RPython: JitCode.code — bytecode string
    pub code: Vec<u8>,
    /// RPython: JitCode.constants_i — integer constant pool
    pub constants_i: Vec<i64>,
    /// RPython: JitCode.constants_r — reference constant pool
    pub constants_r: Vec<u64>,
    /// RPython: JitCode.constants_f — float constant pool
    pub constants_f: Vec<f64>,
    /// RPython: num_regs_i, num_regs_r, num_regs_f
    pub num_regs_i: usize,
    pub num_regs_r: usize,
    pub num_regs_f: usize,
    /// Total flat ops (for statistics)
    pub num_ops: usize,
}

/// Assembler — converts SSARepr to JitCode.
///
/// RPython: `assembler.py::Assembler`.
///
/// The assembler maintains state across multiple JitCode assemblies
/// (shared descriptor table, liveness encoding, etc.)
pub struct Assembler {
    /// RPython: Assembler.insns — map {opcode_key: opcode_number}
    insns: HashMap<String, u8>,
    /// RPython: Assembler.descrs — list of descriptors
    descrs: Vec<String>,
    /// RPython: Assembler._count_jitcodes
    count_jitcodes: usize,
}

impl Assembler {
    /// RPython: `Assembler.__init__()`.
    pub fn new() -> Self {
        Self {
            insns: HashMap::new(),
            descrs: Vec::new(),
            count_jitcodes: 0,
        }
    }

    /// Scaffold: produces metadata-only JitCode (no bytecode).
    ///
    /// RPython equivalent: `Assembler.assemble()` which does full
    /// instruction encoding, label fixup, and liveness encoding.
    /// This scaffold returns register counts only — call sites
    /// that need real bytecode should not use this yet.
    pub fn assemble(
        &mut self,
        flattened: &FlattenedFunction,
        regallocs: &HashMap<RegKind, RegAllocResult>,
    ) -> JitCode {
        let num_regs_i = regallocs.get(&RegKind::Int).map_or(0, |r| r.num_regs);
        let num_regs_r = regallocs.get(&RegKind::Ref).map_or(0, |r| r.num_regs);
        let num_regs_f = regallocs.get(&RegKind::Float).map_or(0, |r| r.num_regs);

        // For now, produce a minimal JitCode with just metadata.
        // Full bytecode encoding (RPython assembler.py write_insn/fix_labels)
        // will be added when the meta-interpreter needs it.
        let jitcode = JitCode {
            name: flattened.name.clone(),
            code: Vec::new(),
            constants_i: Vec::new(),
            constants_r: Vec::new(),
            constants_f: Vec::new(),
            num_regs_i,
            num_regs_r,
            num_regs_f,
            num_ops: flattened.ops.len(),
        };

        self.count_jitcodes += 1;
        jitcode
    }

    /// RPython: `Assembler.finished()` — finalize all JitCodes.
    pub fn finished(&self) {
        // Future: finalize shared liveness data, descriptor table, etc.
    }

    /// Number of JitCodes assembled so far.
    pub fn count_jitcodes(&self) -> usize {
        self.count_jitcodes
    }
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regalloc;

    #[test]
    fn assemble_basic() {
        let flat = FlattenedFunction {
            name: "test".into(),
            ops: vec![],
            num_values: 0,
            num_blocks: 1,
            value_kinds: HashMap::new(),
        };

        let regallocs = regalloc::perform_all_register_allocations(&flat);
        let mut asm = Assembler::new();
        let jitcode = asm.assemble(&flat, &regallocs);

        assert_eq!(jitcode.name, "test");
        assert_eq!(jitcode.num_regs_i, 0);
        assert_eq!(jitcode.num_regs_r, 0);
        assert_eq!(jitcode.num_regs_f, 0);
        assert_eq!(asm.count_jitcodes(), 1);
    }

    #[test]
    fn assemble_with_registers() {
        use crate::graph::{Op, OpKind, ValueType};
        let flat = FlattenedFunction {
            name: "add".into(),
            ops: vec![
                FlatOp::Op(Op {
                    result: Some(ValueId(0)),
                    kind: OpKind::Input {
                        name: "a".into(),
                        ty: ValueType::Int,
                    },
                }),
                FlatOp::Op(Op {
                    result: Some(ValueId(1)),
                    kind: OpKind::BinOp {
                        op: "add".into(),
                        lhs: ValueId(0),
                        rhs: ValueId(0),
                        result_ty: ValueType::Int,
                    },
                }),
                FlatOp::Op(Op {
                    result: Some(ValueId(2)),
                    kind: OpKind::Input {
                        name: "r".into(),
                        ty: ValueType::Ref,
                    },
                }),
            ],
            num_values: 3,
            num_blocks: 1,
            value_kinds: {
                let mut m = HashMap::new();
                m.insert(ValueId(0), RegKind::Int);
                m.insert(ValueId(1), RegKind::Int);
                m.insert(ValueId(2), RegKind::Ref);
                m
            },
        };

        let regallocs = regalloc::perform_all_register_allocations(&flat);
        let mut asm = Assembler::new();
        let jitcode = asm.assemble(&flat, &regallocs);

        // v0 and v1 interfere (v1 uses v0), so they need different regs
        assert_eq!(jitcode.num_regs_i, 2);
        assert_eq!(jitcode.num_regs_r, 1);
        assert_eq!(jitcode.num_regs_f, 0);
    }
}
