//! JitCode assembler — scaffold for converting SSARepr to JitCode bytecode.
//!
//! RPython equivalent: `rpython/jit/codewriter/assembler.py` class `Assembler`.
//!
//! **Status: scaffold only.** Currently produces metadata-only JitCode (register
//! counts, name). Full bytecode encoding (`write_insn`, `fix_labels`, liveness
//! encoding) is not yet implemented — RPython assembler.py does ~300 lines of
//! instruction encoding that this module does not replicate.

use std::collections::HashMap;

use crate::model::ValueId;
use crate::passes::flatten::{FlatOp, Label, RegKind, SSARepr};
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
    /// RPython: Assembler.all_liveness — shared liveness table.
    /// Encoded as bytes: [count_i, count_r, count_f, reg_indices...].
    /// Deduplicated across all JitCodes via all_liveness_positions.
    all_liveness: Vec<u8>,
    /// RPython: Assembler.all_liveness_positions — dedup cache.
    /// Maps (live_i set, live_r set, live_f set) → offset in all_liveness.
    all_liveness_positions: HashMap<(Vec<u8>, Vec<u8>, Vec<u8>), usize>,
}

impl Assembler {
    /// RPython: `Assembler.__init__()`.
    pub fn new() -> Self {
        Self {
            insns: HashMap::new(),
            descrs: Vec::new(),
            count_jitcodes: 0,
            all_liveness: Vec::new(),
            all_liveness_positions: HashMap::new(),
        }
    }

    /// RPython: `Assembler.assemble(ssarepr, jitcode, num_regs)`.
    ///
    /// Takes the SSARepr (flattened instruction sequence) and register
    /// allocation results, and produces a JitCode with encoded bytecode,
    /// constant pools, and register counts.
    ///
    /// RPython assembler.py:34-54.
    /// RPython: `Assembler.assemble(ssarepr, jitcode, num_regs)`.
    ///
    /// RPython codewriter.py:53-56:
    ///   ssarepr = flatten_graph(graph, regallocs)
    ///   compute_liveness(ssarepr)          ← step 3b
    ///   self.assembler.assemble(ssarepr)   ← step 4
    pub fn assemble(
        &mut self,
        ssarepr: &mut SSARepr,
        regallocs: &HashMap<RegKind, RegAllocResult>,
    ) -> JitCode {
        // RPython codewriter.py:56: compute_liveness(ssarepr)
        // Must run BEFORE assembly so -live- markers carry the full
        // set of alive registers.
        crate::liveness::compute_liveness(ssarepr);

        let num_regs_i = regallocs.get(&RegKind::Int).map_or(0, |r| r.num_regs);
        let num_regs_r = regallocs.get(&RegKind::Ref).map_or(0, |r| r.num_regs);
        let num_regs_f = regallocs.get(&RegKind::Float).map_or(0, |r| r.num_regs);

        // RPython assembler.py:56-70: self.setup(ssarepr.name)
        let mut state = AssemblyState {
            code: Vec::new(),
            constants_i: Vec::new(),
            num_regs_i,
            label_positions: HashMap::new(),
            tlabel_fixups: Vec::new(),
            startpoints: std::collections::HashSet::new(),
        };

        // RPython assembler.py:42-44: for insn in ssarepr.insns: write_insn(insn)
        for op in &ssarepr.ops {
            self.write_insn(op, regallocs, &mut state);
        }

        // RPython assembler.py:45,250-258: self.fix_labels()
        for (label, fixup_pos) in &state.tlabel_fixups {
            let target = state.label_positions.get(label).copied().unwrap_or(0);
            let target_u16 = target as u16;
            if fixup_pos + 1 < state.code.len() {
                state.code[*fixup_pos] = (target_u16 & 0xFF) as u8;
                state.code[*fixup_pos + 1] = (target_u16 >> 8) as u8;
            }
        }

        let jitcode = JitCode {
            name: ssarepr.name.clone(),
            code: state.code,
            constants_i: state.constants_i,
            constants_r: Vec::new(),
            constants_f: Vec::new(),
            num_regs_i,
            num_regs_r,
            num_regs_f,
            num_ops: ssarepr.ops.len(),
        };

        self.count_jitcodes += 1;
        jitcode
    }

    /// RPython: `Assembler.write_insn(insn)` — assembler.py:140-223.
    ///
    /// Encodes a single FlatOp into the bytecode stream. Each instruction
    /// is encoded as: opcode_byte + argument_bytes. The opcode byte is
    /// looked up from `self.insns` using a key of the form
    /// `opname/argcodes` (RPython assembler.py:220).
    fn write_insn(
        &mut self,
        op: &FlatOp,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        state: &mut AssemblyState,
    ) {
        match op {
            // RPython assembler.py:143-144: Label → record bytecode position
            FlatOp::Label(label) => {
                state.label_positions.insert(*label, state.code.len());
            }

            // RPython assembler.py:146-158: -live- → encode liveness
            // Separates live registers by kind (int/ref/float) and encodes
            // as offset into shared liveness table.
            FlatOp::Live { live_values } => {
                let key = state.code.len();
                state.startpoints.insert(key);
                // Separate live values by kind
                let mut live_i = Vec::new();
                let mut live_r = Vec::new();
                let mut live_f = Vec::new();
                for &v in live_values {
                    match self.lookup_kind(v, regallocs) {
                        Some(RegKind::Int) => live_i.push(self.lookup_reg(v, regallocs)),
                        Some(RegKind::Ref) => live_r.push(self.lookup_reg(v, regallocs)),
                        Some(RegKind::Float) => live_f.push(self.lookup_reg(v, regallocs)),
                        None => {}
                    }
                }
                let opnum = self.get_opnum("live/");
                state.code.push(opnum);
                // RPython assembler.py:234-248: _encode_liveness
                // Deduplicate liveness data in shared table. Bytecode
                // gets a 2-byte offset into all_liveness.
                live_i.sort();
                live_r.sort();
                live_f.sort();
                let liveness_key = (live_i.clone(), live_r.clone(), live_f.clone());
                let offset = if let Some(&pos) = self.all_liveness_positions.get(&liveness_key) {
                    pos
                } else {
                    let pos = self.all_liveness.len();
                    self.all_liveness_positions.insert(liveness_key, pos);
                    // RPython assembler.py:241: 3 count bytes
                    self.all_liveness.push(live_i.len() as u8);
                    self.all_liveness.push(live_r.len() as u8);
                    self.all_liveness.push(live_f.len() as u8);
                    // RPython assembler.py:243-247: encode_liveness per kind
                    // liveness.py:147-166: bitset encoding — each byte is an
                    // 8-bit bitmap of register indices (bit N = reg N is live).
                    for live in [&live_i, &live_r, &live_f] {
                        let encoded = encode_liveness(live);
                        self.all_liveness.extend_from_slice(&encoded);
                    }
                    pos
                };
                // RPython liveness.py:127-131: encode_offset — 2-byte LE
                state.code.push((offset & 0xFF) as u8);
                state.code.push((offset >> 8) as u8);
            }

            // RPython assembler.py:141-142: '---' → skip
            FlatOp::Unreachable => {}

            // RPython assembler.py:159-223: regular operation
            FlatOp::Op(inner_op) => {
                self.encode_op(inner_op, regallocs, state);
            }

            // RPython flatten.py: 'goto' + TLabel
            FlatOp::Jump(label) => {
                let opnum = self.get_opnum("goto/L");
                state.code.push(opnum);
                state.tlabel_fixups.push((*label, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            // RPython flatten.py:247-267: goto_if_not(cond, TLabel(false_path))
            // Only goto_if_not exists — no goto_if_true in RPython.
            FlatOp::GotoIfNot { cond, target } => {
                let (reg, _) = self.lookup_reg_with_kind(*cond, regallocs);
                let opnum = self.get_opnum("goto_if_not/iL");
                state.code.push(opnum);
                state.code.push(reg);
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            // RPython flatten.py:319-333: {kind}_copy
            FlatOp::Move { dst, src } => {
                let (src_reg, src_kind) = self.lookup_reg_with_kind(*src, regallocs);
                let (dst_reg, _) = self.lookup_reg_with_kind(*dst, regallocs);
                let kind_name = match src_kind {
                    'r' => "ref",
                    'f' => "float",
                    _ => "int",
                };
                let key = format!("{kind_name}_copy/{src_kind}{src_kind}");
                let opnum = self.get_opnum(&key);
                state.code.push(opnum);
                state.code.push(src_reg);
                state.code.push(dst_reg);
            }
        }
    }

    /// RPython assembler.py:159-223: encode one SpaceOperation.
    ///
    /// The encoding for each instruction is:
    /// [opcode_byte][arg1_byte][arg2_byte]...[->][result_byte]
    ///
    /// Where args are:
    /// - Register: 1 byte (index), argcode = kind char ('i','r','f')
    /// - Constant: 1 byte (pool index), argcode = kind char
    /// - TLabel: 2 bytes (u16 LE offset), argcode = 'L'
    /// - ListOfKind: 1 byte (len) + items, argcode = uppercase kind
    /// - Descr: 2 bytes (u16 LE index), argcode = 'd'
    fn encode_op(
        &mut self,
        op: &crate::model::SpaceOperation,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        state: &mut AssemblyState,
    ) {
        use crate::model::OpKind;

        let startposition = state.code.len();
        state.code.push(0); // placeholder for opcode byte
        state.startpoints.insert(startposition);

        let mut argcodes = String::new();

        match &op.kind {
            // RPython: inline_call → [jitcode, I[...], R[...], F[...]]
            OpKind::InlineCall {
                jitcode_index,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            } => {
                // jitcode index as descr-like 2-byte value
                state.code.push((*jitcode_index & 0xFF) as u8);
                state.code.push((*jitcode_index >> 8) as u8);
                argcodes.push('d');
                // ListOfKind 'int'
                self.emit_list_of_kind(args_i, RegKind::Int, regallocs, state);
                argcodes.push('I');
                // ListOfKind 'ref'
                self.emit_list_of_kind(args_r, RegKind::Ref, regallocs, state);
                argcodes.push('R');
                // ListOfKind 'float'
                self.emit_list_of_kind(args_f, RegKind::Float, regallocs, state);
                argcodes.push('F');
                // Result
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                // RPython jtransform.py:423,434: inline_call_{kinds}_{reskind}
                let opname = format!(
                    "inline_call_{}_{}",
                    self.kinds_suffix(args_i, args_r, args_f),
                    result_kind
                );
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython: recursive_call → [jd_index, G_I, G_R, G_F, R_I, R_R, R_F]
            OpKind::RecursiveCall {
                jd_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
                result_kind,
            } => {
                // jd_index as const
                let idx = self.emit_const_i(*jd_index as i64, state);
                state.code.push(idx);
                argcodes.push('c');
                // green lists
                self.emit_list_of_kind(greens_i, RegKind::Int, regallocs, state);
                argcodes.push('I');
                self.emit_list_of_kind(greens_r, RegKind::Ref, regallocs, state);
                argcodes.push('R');
                self.emit_list_of_kind(greens_f, RegKind::Float, regallocs, state);
                argcodes.push('F');
                // red lists
                self.emit_list_of_kind(reds_i, RegKind::Int, regallocs, state);
                argcodes.push('I');
                self.emit_list_of_kind(reds_r, RegKind::Ref, regallocs, state);
                argcodes.push('R');
                self.emit_list_of_kind(reds_f, RegKind::Float, regallocs, state);
                argcodes.push('F');
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("recursive_call_{result_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython: residual_call/call_may_force/call_elidable
            // → [funcptr/descr, I[...], R[...], F[...]]
            // Currently these use flat args (not kind-separated), so we
            // encode the descriptor as a 2-byte index and args as registers.
            OpKind::CallResidual {
                descriptor,
                args,
                result_ty,
            }
            | OpKind::CallMayForce {
                descriptor,
                args,
                result_ty,
            }
            | OpKind::CallElidable {
                descriptor,
                args,
                result_ty,
            } => {
                let base = match &op.kind {
                    OpKind::CallMayForce { .. } => "call_may_force",
                    OpKind::CallElidable { .. } => "call_elidable",
                    _ => "residual_call",
                };
                // Descriptor as 2-byte index (RPython assembler.py:197-207)
                let descr_idx = self.descrs.len();
                self.descrs.push(format!("{}", descriptor.target));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Args as registers (flat — kind separation is in the
                // opname suffix, not the encoding)
                for &v in args {
                    let (reg, kc) = self.lookup_reg_with_kind(v, regallocs);
                    state.code.push(reg);
                    argcodes.push(kc);
                }
                // Result
                let result_kind = value_type_to_kind(result_ty);
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("{base}_{result_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython: ConstInt is NOT a separate op. Constants appear as
            // arguments to other instructions. We encode it as a standalone
            // const_int op that puts the value into a register.
            OpKind::ConstInt(val) => {
                let idx = self.emit_const_i(*val, state);
                state.code.push(idx);
                argcodes.push('c');
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("const_int/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Default: encode operand registers + result register
            other => {
                for v in crate::inline::op_value_refs(other) {
                    let (reg, kind_char) = self.lookup_reg_with_kind(v, regallocs);
                    state.code.push(reg);
                    argcodes.push(kind_char);
                }
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kind_char) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kind_char);
                    state.code.push(reg);
                }
                // RPython: jtransform already produces typed opnames like
                // getfield_vable_i. op_kind_to_opname approximates this.
                let opname = op_kind_to_opname(other);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
        }
    }

    /// Emit a ListOfKind: [u8 count][reg0][reg1]...
    /// RPython assembler.py:181-196.
    fn emit_list_of_kind(
        &self,
        args: &[ValueId],
        _kind: RegKind,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        state: &mut AssemblyState,
    ) {
        state.code.push(args.len().min(255) as u8);
        for &v in args {
            let reg = self.lookup_reg(v, regallocs);
            state.code.push(reg);
        }
    }

    /// RPython rewrite_call: determine the 'kinds' suffix (ir/r/irf).
    /// assembler.py:424-426.
    fn kinds_suffix(
        &self,
        args_i: &[ValueId],
        args_r: &[ValueId],
        args_f: &[ValueId],
    ) -> &'static str {
        if !args_f.is_empty() {
            "irf"
        } else if !args_i.is_empty() {
            "ir"
        } else {
            "r"
        }
    }

    /// RPython: opcode key → opcode number.
    /// RPython assembler.py:220-222: key = opname + '/' + argcodes
    fn get_opnum(&mut self, key: &str) -> u8 {
        let next = self.insns.len() as u8;
        *self.insns.entry(key.to_string()).or_insert(next)
    }

    /// Look up the register index (as u8) for a ValueId.
    /// RPython: registers are single-byte indices.
    fn lookup_reg(&self, v: ValueId, regallocs: &HashMap<RegKind, RegAllocResult>) -> u8 {
        for ra in regallocs.values() {
            if let Some(&color) = ra.coloring.get(&v) {
                return color as u8;
            }
        }
        0
    }

    /// Look up register index AND kind character for a ValueId.
    /// Returns (register_index, kind_char) where kind_char ∈ {'i','r','f'}.
    fn lookup_reg_with_kind(
        &self,
        v: ValueId,
        regallocs: &HashMap<RegKind, RegAllocResult>,
    ) -> (u8, char) {
        for (kind, ra) in regallocs {
            if let Some(&color) = ra.coloring.get(&v) {
                let kind_char = match kind {
                    RegKind::Int => 'i',
                    RegKind::Ref => 'r',
                    RegKind::Float => 'f',
                };
                return (color as u8, kind_char);
            }
        }
        (0, 'i')
    }

    /// Look up just the kind for a ValueId.
    fn lookup_kind(
        &self,
        v: ValueId,
        regallocs: &HashMap<RegKind, RegAllocResult>,
    ) -> Option<RegKind> {
        for (kind, ra) in regallocs {
            if ra.coloring.contains_key(&v) {
                return Some(*kind);
            }
        }
        None
    }

    /// RPython assembler.py:80-138: emit_const for integer constants.
    /// Adds to constant pool and returns the index byte.
    fn emit_const_i(&mut self, value: i64, state: &mut AssemblyState) -> u8 {
        // Check if already in pool
        for (i, &existing) in state.constants_i.iter().enumerate() {
            if existing == value {
                return (state.num_regs_i + i) as u8;
            }
        }
        // Add to pool: index = num_regs + pool_position
        state.constants_i.push(value);
        (state.num_regs_i + state.constants_i.len() - 1) as u8
    }
}

/// Per-assembly state (RPython: Assembler.setup() fields).
struct AssemblyState {
    code: Vec<u8>,
    constants_i: Vec<i64>,
    num_regs_i: usize,
    label_positions: HashMap<Label, usize>,
    tlabel_fixups: Vec<(Label, usize)>,
    startpoints: std::collections::HashSet<usize>,
}

/// RPython: getkind(v.concretetype)[0] → 'i', 'r', 'f', 'v'.
fn value_type_to_kind(ty: &crate::model::ValueType) -> char {
    use crate::model::ValueType;
    match ty {
        ValueType::Int => 'i',
        ValueType::Ref => 'r',
        ValueType::Float => 'f',
        ValueType::Void | ValueType::State | ValueType::Unknown => 'v',
    }
}

/// RPython liveness.py:147-166: encode_liveness.
///
/// Encodes a sorted set of register indices as a bitset. Each byte
/// represents 8 consecutive register indices: bit N = register (offset+N)
/// is live. The output is a compact bitmap.
fn encode_liveness(live: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    let mut offset: u16 = 0;
    let mut byte: u8 = 0;
    let mut i = 0;
    while i < live.len() {
        let x = live[i] as u16;
        let rel = x - offset;
        if rel >= 8 {
            result.push(byte);
            byte = 0;
            offset += 8;
            continue;
        }
        byte |= 1 << rel;
        i += 1;
    }
    if byte != 0 {
        result.push(byte);
    }
    result
}

/// Convert OpKind to an opname string for the assembler's instruction table.
/// RPython: the opname comes from SpaceOperation.opname.
fn op_kind_to_opname(kind: &crate::model::OpKind) -> String {
    use crate::model::OpKind;
    match kind {
        OpKind::Input { .. } => "input".into(),
        OpKind::ConstInt(_) => "const_int".into(),
        OpKind::FieldRead { .. } => "getfield_gc".into(),
        OpKind::FieldWrite { .. } => "setfield_gc".into(),
        OpKind::ArrayRead { .. } => "getarrayitem_gc".into(),
        OpKind::ArrayWrite { .. } => "setarrayitem_gc".into(),
        OpKind::Call { .. } => "direct_call".into(),
        OpKind::GuardTrue { .. } => "guard_true".into(),
        OpKind::GuardFalse { .. } => "guard_false".into(),
        OpKind::GuardValue { kind_char, .. } => format!("{kind_char}_guard_value"),
        OpKind::VableFieldRead { .. } => "getfield_vable".into(),
        OpKind::VableFieldWrite { .. } => "setfield_vable".into(),
        OpKind::VableArrayRead { .. } => "getarrayitem_vable".into(),
        OpKind::VableArrayWrite { .. } => "setarrayitem_vable".into(),
        OpKind::BinOp { op, .. } => format!("int_{op}"),
        OpKind::UnaryOp { op, .. } => format!("int_{op}"),
        OpKind::VableForce => "hint_force_virtualizable".into(),
        OpKind::Live => "live".into(),
        OpKind::CallElidable { .. } => "call_elidable".into(),
        OpKind::CallResidual { .. } => "residual_call".into(),
        OpKind::CallMayForce { .. } => "call_may_force".into(),
        OpKind::InlineCall { .. } => "inline_call".into(),
        OpKind::RecursiveCall { .. } => "recursive_call".into(),
        OpKind::Unknown { .. } => "unknown".into(),
    }
}

impl Assembler {
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
        let mut flat = SSARepr {
            name: "test".into(),
            ops: vec![],
            num_values: 0,
            num_blocks: 1,
            value_kinds: HashMap::new(),
        };

        // Empty regalloc results
        let mut regallocs = HashMap::new();
        regallocs.insert(
            RegKind::Int,
            regalloc::RegAllocResult {
                coloring: HashMap::new(),
                num_regs: 0,
            },
        );
        regallocs.insert(
            RegKind::Ref,
            regalloc::RegAllocResult {
                coloring: HashMap::new(),
                num_regs: 0,
            },
        );
        regallocs.insert(
            RegKind::Float,
            regalloc::RegAllocResult {
                coloring: HashMap::new(),
                num_regs: 0,
            },
        );
        let mut asm = Assembler::new();
        let jitcode = asm.assemble(&mut flat, &regallocs);

        assert_eq!(jitcode.name, "test");
        assert_eq!(jitcode.num_regs_i, 0);
        assert_eq!(jitcode.num_regs_r, 0);
        assert_eq!(jitcode.num_regs_f, 0);
        assert_eq!(asm.count_jitcodes(), 1);
    }

    #[test]
    fn assemble_with_registers() {
        use crate::model::{FunctionGraph, OpKind, Terminator, ValueType};
        // Build graph for regalloc (regalloc operates on graph, not SSARepr)
        let mut graph = FunctionGraph::new("add");
        let entry = graph.entry;
        let v0 = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "a".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let v1 = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: v0,
                    rhs: v0,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let v2 = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "r".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.set_terminator(entry, Terminator::Return(Some(v1)));

        let mut value_kinds = HashMap::new();
        value_kinds.insert(v0, RegKind::Int);
        value_kinds.insert(v1, RegKind::Int);
        value_kinds.insert(v2, RegKind::Ref);

        let regallocs = regalloc::perform_all_register_allocations(&graph, &value_kinds);
        let mut flat = SSARepr {
            name: "add".into(),
            ops: vec![],
            num_values: 3,
            num_blocks: 1,
            value_kinds,
        };
        let mut asm = Assembler::new();
        let jitcode = asm.assemble(&mut flat, &regallocs);

        // v0 dies when v1 is defined → they share a register → 1 int reg
        assert_eq!(jitcode.num_regs_i, 1);
        assert_eq!(jitcode.num_regs_r, 1);
        assert_eq!(jitcode.num_regs_f, 0);
    }
}
