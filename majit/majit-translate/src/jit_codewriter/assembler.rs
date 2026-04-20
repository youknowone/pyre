//! Assembler — converts flattened SSARepr into a `JitCode`.
//!
//! RPython equivalent: `rpython/jit/codewriter/assembler.py` class
//! `Assembler`. The `JitCode` struct itself lives in `crate::jitcode`
//! (RPython parity: `rpython/jit/codewriter/jitcode.py`).
//!
//! **Status: partial port.** `write_insn`, `fix_labels`, the
//! `all_liveness` shared table, and the `IndirectCallTargets` sidecar
//! merge for `residual_call` are implemented in the pyre-relevant
//! subset.  `SwitchDictDescr` and full RPython parity for the
//! remaining descriptor kinds are still pending.

use std::collections::HashMap;

use crate::flatten::{FlatOp, Label, RegKind, SSARepr};
use crate::jitcode::{BhCallDescr, JitCodeBody};
use crate::model::ValueId;
use crate::regalloc::RegAllocResult;

/// Assembler — converts SSARepr to JitCode.
///
/// RPython: `assembler.py::Assembler`.
///
/// The assembler maintains state across multiple JitCode assemblies
/// (shared descriptor table, liveness encoding, etc.)
pub struct Assembler {
    /// RPython: Assembler.insns — map {opcode_key: opcode_number}
    insns: HashMap<String, u8>,
    /// RPython: Assembler.descrs — list of descriptors.
    /// RPython stores AbstractDescr objects; pyre uses BhDescr enum.
    descrs: Vec<crate::jitcode::BhDescr>,
    /// RPython: `Assembler.indirectcalltargets` — merged `IndirectCallTargets`
    /// sidecars from every `residual_call` emitted during assembly
    /// (`assembler.py:208-209`).  RPython stores `JitCode` objects; we
    /// store their jitcode indices because codewriter owns the
    /// jitcode-index allocator.
    /// RPython `assembler.py:209` `self.indirectcalltargets.update(x.lst)`:
    /// a `set` of JitCode objects (Python identity dedup). pyre uses
    /// `JitCodeHandle` as the identity-keyed wrapper around
    /// `Arc<JitCode>` so the same shells handed out by
    /// `CallControl::get_jitcode` survive into the metainterp side
    /// without copying.
    pub indirectcalltargets: std::collections::HashSet<crate::jitcode::JitCodeHandle>,
    /// RPython: Assembler.list_of_addr2name — (addr, name) pairs for debugging.
    /// In majit: (target_path, name) string pairs since we don't have raw addresses.
    pub list_of_addr2name: Vec<(String, String)>,
    /// RPython: Assembler._count_jitcodes
    count_jitcodes: usize,
    /// RPython: Assembler._seen_raw_objects — dedup set for see_raw_object.
    seen_raw_objects: std::collections::HashSet<String>,
    /// RPython: Assembler.all_liveness — shared liveness table.
    /// Encoded as bytes: [count_i, count_r, count_f, reg_indices...].
    /// Deduplicated across all JitCodes via all_liveness_positions.
    all_liveness: Vec<u8>,
    /// RPython: Assembler.all_liveness_length (assembler.py:30).
    pub all_liveness_length: usize,
    /// RPython: Assembler.all_liveness_positions — dedup cache.
    /// Maps (live_i set, live_r set, live_f set) → offset in all_liveness.
    all_liveness_positions: HashMap<(Vec<u8>, Vec<u8>, Vec<u8>), usize>,
    /// RPython: Assembler.num_liveness_ops (assembler.py:32).
    pub num_liveness_ops: usize,
}

impl Assembler {
    /// RPython: `Assembler.__init__()` (assembler.py:21-32).
    pub fn new() -> Self {
        Self {
            insns: HashMap::new(),
            descrs: Vec::new(),
            indirectcalltargets: std::collections::HashSet::new(),
            list_of_addr2name: Vec::new(),
            count_jitcodes: 0,
            seen_raw_objects: std::collections::HashSet::new(),
            all_liveness: Vec::new(),
            all_liveness_length: 0,
            all_liveness_positions: HashMap::new(),
            num_liveness_ops: 0,
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
    ) -> JitCodeBody {
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
            alllabels: std::collections::HashSet::new(),
            resulttypes: HashMap::new(),
        };

        // RPython assembler.py:41-44:
        //     ssarepr._insns_pos = []
        //     for insn in ssarepr.insns:
        //         ssarepr._insns_pos.append(len(self.code))
        //         self.write_insn(insn)
        let mut insns_pos = Vec::with_capacity(ssarepr.insns.len());
        // Borrow split: clone the insn vec so we can mutate ssarepr
        // (insns_pos write) without aliasing the borrow used by the
        // write_insn loop.
        let ops = ssarepr.insns.clone();
        for op in &ops {
            insns_pos.push(state.code.len());
            self.write_insn(op, regallocs, &mut state);
        }
        ssarepr.insns_pos = Some(insns_pos);

        // RPython assembler.py:45,250-258: self.fix_labels()
        for (label, fixup_pos) in &state.tlabel_fixups {
            let target = state.label_positions.get(label).copied().unwrap_or(0);
            let target_u16 = target as u16;
            if fixup_pos + 1 < state.code.len() {
                state.code[*fixup_pos] = (target_u16 & 0xFF) as u8;
                state.code[*fixup_pos + 1] = (target_u16 >> 8) as u8;
            }
        }

        // RPython assembler.py:271-281: jitcode.setup(code, ...)
        // Build the body that the codewriter will commit into the
        // pre-allocated `Arc<JitCode>` shell via `set_body`.
        // RPython jitcode.py:36 `assert num_regs_i < 256 and ...`. The
        // assembler limits register pressure via the same invariant.
        assert!(
            num_regs_i < 256 && num_regs_r < 256 && num_regs_f < 256,
            "too many registers (i={num_regs_i} r={num_regs_r} f={num_regs_f})"
        );
        // RPython assembler.py:49 `jitcode._ssarepr = ssarepr`
        let body = JitCodeBody {
            calldescr: BhCallDescr::default(),
            code: state.code,
            constants_i: state.constants_i,
            constants_r: Vec::new(),
            constants_f: Vec::new(),
            c_num_regs_i: num_regs_i as u8,
            c_num_regs_r: num_regs_r as u8,
            c_num_regs_f: num_regs_f as u8,
            startpoints: state.startpoints,
            alllabels: state.alllabels,
            resulttypes: state.resulttypes,
            _ssarepr: Some(ssarepr.clone()),
        };

        self.count_jitcodes += 1;
        body
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
                self.num_liveness_ops += 1;
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
                    self.all_liveness_length = self.all_liveness.len();
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
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                // RPython assembler.py:175-179: TLabel → record position + 2 placeholder bytes
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*label, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            FlatOp::CatchException { target } => {
                let opnum = self.get_opnum("catch_exception/L");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            FlatOp::GotoIfExceptionMismatch { llexitcase, target } => {
                // RPython `flatten.py:228-231`:
                //   emitline('goto_if_exception_mismatch',
                //            Constant(link.llexitcase,
                //                     lltype.typeOf(link.llexitcase)),
                //            TLabel(link))
                // Flatten already narrowed per `lltype.typeOf` to `i64`
                // (`lltype.Signed`); the opname suffix encodes the
                // matching kind.
                let opnum = self.get_opnum("goto_if_exception_mismatch/iL");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                let encoded_llexitcase = self.emit_const_i(*llexitcase, state);
                state.code.push(encoded_llexitcase);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            // RPython flatten.py:247-267: goto_if_not(cond, TLabel(false_path))
            // Only goto_if_not exists — no goto_if_true in RPython.
            FlatOp::GotoIfNot { cond, target } => {
                let (reg, _) = self.lookup_reg_with_kind(*cond, regallocs);
                let opnum = self.get_opnum("goto_if_not/iL");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
                // RPython assembler.py:175-179: TLabel operand position
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            // RPython flatten.py:333 `self.emitline('%s_copy' % kind,
            // v, "->", w)` — argcodes `i>i` (typed src, result marker,
            // typed dst). The `>` bears no byte in the stream; it only
            // flags the result position in the key so the blackhole
            // wire `int_copy/i>i` (blackhole.rs:5670) finds the handler
            // and `Assembler.resulttypes[pc]` is populated correctly.
            FlatOp::Move { dst, src } => {
                let (src_reg, src_kind) = self.lookup_reg_with_kind(*src, regallocs);
                let (dst_reg, _) = self.lookup_reg_with_kind(*dst, regallocs);
                let kind_name = match src_kind {
                    'r' => "ref",
                    'f' => "float",
                    _ => "int",
                };
                let key = format!("{kind_name}_copy/{src_kind}>{src_kind}");
                let opnum = self.get_opnum(&key);
                state.code.push(opnum);
                state.code.push(src_reg);
                state.code.push(dst_reg);
                // RPython `assembler.py:210-212`: when argcodes contain
                // `>`, record the reskind at the current pc. Mirrors
                // `encode_op`'s handling of any op with a result slot.
                state.resulttypes.insert(state.code.len(), src_kind);
            }

            // RPython `flatten.py:329` `self.emitline('%s_push' % kind, v)`.
            // Argcodes: one typed register, no result marker. Blackhole
            // wires under `{kind}_push/{kind}` (see blackhole.rs:5727-5729).
            FlatOp::Push(src) => {
                let (src_reg, src_kind) = self.lookup_reg_with_kind(*src, regallocs);
                let kind_name = match src_kind {
                    'r' => "ref",
                    'f' => "float",
                    _ => "int",
                };
                let key = format!("{kind_name}_push/{src_kind}");
                let opnum = self.get_opnum(&key);
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(src_reg);
                // No `>` in argcodes → no resulttypes entry (upstream
                // `assembler.py:210-212` guarded on `'>' in argcodes`).
            }

            // RPython `flatten.py:331` `self.emitline('%s_pop' % kind, "->", w)`.
            // Argcodes: `>{kind}` — result marker then one typed register.
            // Blackhole wires under `{kind}_pop/>{kind}` (see
            // blackhole.rs:5730-5732).
            FlatOp::Pop(dst) => {
                let (dst_reg, dst_kind) = self.lookup_reg_with_kind(*dst, regallocs);
                let kind_name = match dst_kind {
                    'r' => "ref",
                    'f' => "float",
                    _ => "int",
                };
                let key = format!("{kind_name}_pop/>{dst_kind}");
                let opnum = self.get_opnum(&key);
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(dst_reg);
                // `>` present in argcodes → record reskind per upstream
                // `assembler.py:210-212`.
                state.resulttypes.insert(state.code.len(), dst_kind);
            }

            FlatOp::LastException { dst } => {
                let (reg, kind) = self.lookup_reg_with_kind(*dst, regallocs);
                debug_assert_eq!(kind, 'i');
                let opnum = self.get_opnum("last_exception/>i");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
                state.resulttypes.insert(state.code.len(), 'i');
            }

            FlatOp::LastExcValue { dst } => {
                let (reg, kind) = self.lookup_reg_with_kind(*dst, regallocs);
                debug_assert_eq!(kind, 'r');
                let opnum = self.get_opnum("last_exc_value/>r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
                state.resulttypes.insert(state.code.len(), 'r');
            }

            FlatOp::Reraise => {
                let opnum = self.get_opnum("reraise/");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
            }

            // RPython `flatten.py:131-138` `make_return`.  Blackhole
            // handlers: `blackhole.py:841-863 bhimpl_{int,ref,float,void}_return`.
            FlatOp::IntReturn(v) => {
                let (reg, kind) = self.lookup_reg_with_kind(*v, regallocs);
                debug_assert_eq!(kind, 'i');
                let opnum = self.get_opnum("int_return/i");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
            FlatOp::RefReturn(v) => {
                let (reg, kind) = self.lookup_reg_with_kind(*v, regallocs);
                debug_assert_eq!(kind, 'r');
                let opnum = self.get_opnum("ref_return/r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
            FlatOp::FloatReturn(v) => {
                let (reg, kind) = self.lookup_reg_with_kind(*v, regallocs);
                debug_assert_eq!(kind, 'f');
                let opnum = self.get_opnum("float_return/f");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
            FlatOp::VoidReturn => {
                let opnum = self.get_opnum("void_return/");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
            }
            // RPython `flatten.py:139-143` `make_return` 2-inputarg case.
            // Blackhole: `blackhole.py:1000 bhimpl_raise(excvalue)`.
            FlatOp::Raise(v) => {
                let (reg, kind) = self.lookup_reg_with_kind(*v, regallocs);
                debug_assert_eq!(kind, 'r');
                let opnum = self.get_opnum("raise/r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
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
                // RPython assembler.py:197-207: jitcode → descrs[index]
                // The JitCode object IS the descriptor for inline_call.
                let descr_idx = self.descrs.len();
                let calldescr = crate::jitcode::BhCallDescr {
                    arg_classes: self.kinds_suffix(args_i, args_r, args_f).to_string(),
                    result_type: *result_kind,
                };
                self.descrs.push(crate::jitcode::BhDescr::JitCode {
                    jitcode_index: *jitcode_index,
                    fnaddr: 0, // resolved at runtime from all_jitcodes[jitcode_index]
                    calldescr,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython jtransform.py:422-431: rewrite_call
                // Only emit the kind sublists that are in 'kinds'.
                let kinds = self.kinds_suffix(args_i, args_r, args_f);
                if kinds.contains('i') {
                    self.emit_list_of_kind(args_i, RegKind::Int, regallocs, state);
                    argcodes.push('I');
                }
                if kinds.contains('r') {
                    self.emit_list_of_kind(args_r, RegKind::Ref, regallocs, state);
                    argcodes.push('R');
                }
                if kinds.contains('f') {
                    self.emit_list_of_kind(args_f, RegKind::Float, regallocs, state);
                    argcodes.push('F');
                }
                // Result
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                // RPython jtransform.py:434: inline_call_{kinds}_{reskind}
                let key = format!("inline_call_{kinds}_{result_kind}/{argcodes}");
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
            // → [funcptr, calldescr, I[...], R[...], F[...]]
            // RPython jtransform.py:414-435: rewrite_call splits args
            // by kind via make_three_lists.
            OpKind::CallResidual {
                funcptr,
                descriptor: _,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            }
            | OpKind::CallMayForce {
                funcptr,
                descriptor: _,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            }
            | OpKind::CallElidable {
                funcptr,
                descriptor: _,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            } => {
                // RPython `assembler.py:208-209`: the sidecar
                // `IndirectCallTargets(lst)` on a `residual_call`
                // merges into the global `Assembler.indirectcalltargets`
                // set so the metainterp can later look up jitcodes by
                // funcptr address during runtime dispatch.  Only
                // `OpKind::CallResidual` carries the sidecar today.
                if let OpKind::CallResidual {
                    indirect_targets: Some(t),
                    ..
                } = &op.kind
                {
                    self.indirectcalltargets.extend(t.lst.iter().cloned());
                }
                let base = match &op.kind {
                    OpKind::CallMayForce { .. } => "call_may_force",
                    OpKind::CallElidable { .. } => "call_elidable",
                    _ => "residual_call",
                };
                // RPython `jtransform.py:422-431` `rewrite_call` emits args
                // by kind (I, R, F) first, then the calldescr, producing
                // keys like `residual_call_ir_r/iIRd>r`. jtransform now
                // materializes direct-call funcptrs as `ConstInt` values,
                // so every post-jtransform call op reaches the assembler
                // as `CallFuncPtr::Value(...)` and encodes the orthodox
                // leading `i` operand.
                match funcptr {
                    crate::model::CallFuncPtr::Value(vid) => {
                        let (reg, kc) = self.lookup_reg_with_kind(*vid, regallocs);
                        state.code.push(reg);
                        argcodes.push(kc);
                    }
                    crate::model::CallFuncPtr::Target(target) => {
                        panic!("call op reached assembler without materialized funcptr: {target}");
                    }
                }
                // Reserve the descr slot up front — the index is allocated
                // here so the `arg_classes` suffix below can reference it,
                // but the two bytes are written AFTER the I/R/F lists to
                // match `jtransform.py:422-431` ordering: `iIRFd` /
                // `iIRd` / `iRd`.
                let descr_idx = self.descrs.len();
                let calldescr = crate::jitcode::BhCallDescr {
                    arg_classes: self.kinds_suffix(args_i, args_r, args_f).to_string(),
                    result_type: *result_kind,
                };
                self.descrs
                    .push(crate::jitcode::BhDescr::Call { calldescr });
                // RPython jtransform.py:422-431: kind-separated sublists
                let kinds = self.kinds_suffix(args_i, args_r, args_f);
                if kinds.contains('i') {
                    self.emit_list_of_kind(args_i, RegKind::Int, regallocs, state);
                    argcodes.push('I');
                }
                if kinds.contains('r') {
                    self.emit_list_of_kind(args_r, RegKind::Ref, regallocs, state);
                    argcodes.push('R');
                }
                if kinds.contains('f') {
                    self.emit_list_of_kind(args_f, RegKind::Float, regallocs, state);
                    argcodes.push('F');
                }
                // RPython assembler.py:197-207: descriptor as 2-byte index,
                // emitted last per jtransform.py:422-431 ordering so the
                // blackhole key suffix is `...d>k`.
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Result
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                // RPython jtransform.py:434: {base}_{kinds}_{reskind}
                let key = format!("{base}_{kinds}_{result_kind}/{argcodes}");
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

            // Field/array operations: encode registers + descriptor.
            // RPython assembler.py:197-207: AbstractDescr → 2-byte index.
            // Field operations: register + descriptor.
            // RPython assembler.py:197-207: AbstractDescr → 2-byte index.
            OpKind::FieldRead { base, field, .. } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx = self.descrs.len();
                self.descrs.push(crate::jitcode::BhDescr::Field {
                    offset: 0,
                    name: field.name.clone(),
                    owner: field.owner_root.clone().unwrap_or_default(),
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // RPython `rpython/jit/codewriter/jtransform.py:546` emits
            // `int_guard_value(op.args[0])` where `op.args[0]` is already a
            // `Ptr(FuncType)` integer after rtype.  Rust `&dyn Trait` is a
            // fat pointer so the rtyper-equivalent layer
            // (`translator/rtyper/rclass.rs::class_get_method_ptr`) emits
            // `OpKind::VtableMethodPtr(receiver)` with the
            // `(trait_root, method_name)` pair; the assembler encodes it
            // as `vtable_method_ptr/rd>i` carrying a `BhDescr::VtableMethod`.
            // The result register is the integer function address that
            // `int_guard_value` and the subsequent `residual_call_*`
            // consume — backend lowering of the actual vtable slot read
            // is deferred (separate epic).
            OpKind::VtableMethodPtr {
                receiver,
                trait_root,
                method_name,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*receiver, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx = self.descrs.len();
                self.descrs.push(crate::jitcode::BhDescr::VtableMethod {
                    trait_root: trait_root.clone(),
                    method_name: method_name.clone(),
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // RPython `rpython/jit/codewriter/jtransform.py:901-903` emits
            // `record_quasiimmut_field(v_inst, fielddescr, mutatefielddescr)`
            // — a register followed by two descrs.  The blackhole counterpart
            // `bhimpl_record_quasiimmut_field(struct, fielddescr,
            // mutatefielddescr)` (`rpython/jit/metainterp/blackhole.py:1537-1539`)
            // expects argcodes `rdd`.
            OpKind::RecordQuasiImmutField {
                base,
                field,
                mutate_field,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                for fd in [field, mutate_field] {
                    let descr_idx = self.descrs.len();
                    self.descrs.push(crate::jitcode::BhDescr::Field {
                        offset: 0,
                        name: fd.name.clone(),
                        owner: fd.owner_root.clone().unwrap_or_default(),
                    });
                    state.code.push((descr_idx & 0xFF) as u8);
                    state.code.push((descr_idx >> 8) as u8);
                    argcodes.push('d');
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::FieldWrite {
                base, value, field, ..
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind(*value, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx = self.descrs.len();
                self.descrs.push(crate::jitcode::BhDescr::Field {
                    offset: 0,
                    name: field.name.clone(),
                    owner: field.owner_root.clone().unwrap_or_default(),
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ArrayRead {
                base,
                index,
                item_ty,
                array_type_id,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind(*index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (itemsize, is_item_signed) = arraydescrof(item_ty, array_type_id.as_deref());
                let descr_idx = self.descrs.len();
                self.descrs.push(crate::jitcode::BhDescr::Array {
                    itemsize,
                    is_array_of_pointers: matches!(item_ty, crate::model::ValueType::Ref),
                    is_array_of_structs: false,
                    is_item_signed,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ArrayWrite {
                base,
                index,
                value,
                item_ty,
                array_type_id,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind(*index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind(*value, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (itemsize, is_item_signed) = arraydescrof(item_ty, array_type_id.as_deref());
                let descr_idx = self.descrs.len();
                self.descrs.push(crate::jitcode::BhDescr::Array {
                    itemsize,
                    is_array_of_pointers: matches!(item_ty, crate::model::ValueType::Ref),
                    is_array_of_structs: false,
                    is_item_signed,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // Vable field/array: encode field_index as descriptor.
            // These don't have a base register (vable is implicit).
            OpKind::VableFieldRead { field_index, .. } => {
                // RPython: vable field → VableField descriptor (index, not byte offset).
                let descr_idx = self.descrs.len();
                self.descrs.push(crate::jitcode::BhDescr::VableField {
                    index: *field_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableFieldWrite {
                field_index, value, ..
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*value, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx = self.descrs.len();
                self.descrs.push(crate::jitcode::BhDescr::VableField {
                    index: *field_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableArrayRead {
                array_index,
                elem_index,
                item_ty,
                array_itemsize,
                array_is_signed,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*elem_index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // RPython: two descriptors — fielddescr (vable array field) + arraydescr.
                let descr_idx = self.descrs.len();
                self.descrs.push(crate::jitcode::BhDescr::VableArray {
                    index: *array_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Second descriptor: arraydescr from VirtualizableInfo.array_descrs.
                let descr_idx2 = self.descrs.len();
                self.descrs.push(crate::jitcode::BhDescr::Array {
                    itemsize: *array_itemsize,
                    is_array_of_pointers: matches!(item_ty, crate::model::ValueType::Ref),
                    is_array_of_structs: false,
                    is_item_signed: *array_is_signed,
                });
                state.code.push((descr_idx2 & 0xFF) as u8);
                state.code.push((descr_idx2 >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableArrayWrite {
                array_index,
                elem_index,
                value,
                item_ty,
                array_itemsize,
                array_is_signed,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind(*elem_index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind(*value, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // RPython: two descriptors — fielddescr (vable array field) + arraydescr.
                let descr_idx = self.descrs.len();
                self.descrs.push(crate::jitcode::BhDescr::VableArray {
                    index: *array_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Second descriptor: arraydescr from VirtualizableInfo.array_descrs.
                let descr_idx2 = self.descrs.len();
                self.descrs.push(crate::jitcode::BhDescr::Array {
                    itemsize: *array_itemsize,
                    is_array_of_pointers: matches!(item_ty, crate::model::ValueType::Ref),
                    is_array_of_structs: false,
                    is_item_signed: *array_is_signed,
                });
                state.code.push((descr_idx2 & 0xFF) as u8);
                state.code.push((descr_idx2 >> 8) as u8);
                argcodes.push('d');
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Default: encode operand registers + result register (no descriptor)
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
                let opname = op_kind_to_opname(other);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
        }

        // RPython assembler.py:217-219: record result type position.
        // If argcodes contains '>', the last char is the result kind,
        // and we record the current code length as the result position.
        if argcodes.contains('>') {
            if let Some(reskind) = argcodes.chars().last() {
                state.resulttypes.insert(state.code.len(), reskind);
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
        _args_r: &[ValueId],
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
    /// RPython assembler.py:176: positions in bytecode where TLabel operands
    /// are written. Used by JitCode.follow_jump() for verification.
    alllabels: std::collections::HashSet<usize>,
    /// RPython assembler.py:217-219: map from bytecode offset (after `->`)
    /// to result kind character. Recorded when encoding result registers.
    resulttypes: HashMap<usize, char>,
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

fn value_type_to_itemsize(ty: &crate::model::ValueType) -> usize {
    use crate::model::ValueType;
    match ty {
        ValueType::Int => 8,
        ValueType::Ref => 8,
        ValueType::Float => 8,
        _ => 8,
    }
}

/// jtransform.py:773,802 cpu.arraydescrof(ARRAY) equivalent.
///
/// Determines (itemsize, is_item_signed) from the array element type.
/// When `array_type_id` is available (e.g. `Vec<i32>` → element `i32`),
/// the result is exact. Fallback uses descr.py:241-254 get_type_flag()
/// semantics: Int → FLAG_SIGNED, Float/Ref → FLAG_UNSIGNED/FLAG_FLOAT.
fn arraydescrof(ty: &crate::model::ValueType, array_type_id: Option<&str>) -> (usize, bool) {
    // Primary path: extract element type from the array type identity
    // (our equivalent of `ARRAY.OF` in RPython).
    if let Some(elem) = array_type_id.and_then(extract_element_type_from_str) {
        return match elem.as_str() {
            "i8" => (1, true),
            "i16" => (2, true),
            "i32" => (4, true),
            "i64" | "isize" => (8, true),
            "u8" | "bool" => (1, false),
            "u16" => (2, false),
            "u32" => (4, false),
            "u64" | "usize" => (8, false),
            "f32" => (4, false),
            "f64" => (8, false),
            _ => (value_type_to_itemsize(ty), false),
        };
    }
    // Fallback: descr.py:241-254 get_type_flag(ARRAY.OF).
    // Int (lltype.Signed) → FLAG_SIGNED, Float → FLAG_FLOAT,
    // Ref (gc pointer) → FLAG_POINTER. Only FLAG_SIGNED → is_item_signed=true.
    match ty {
        crate::model::ValueType::Int => (8, true),
        crate::model::ValueType::Float => (8, false),
        crate::model::ValueType::Ref => (8, false),
        _ => (8, false),
    }
}

fn extract_element_type_from_str(type_str: &str) -> Option<String> {
    let s = type_str.trim();
    if let (Some(start), Some(end)) = (s.find('<'), s.rfind('>')) {
        if start < end {
            return Some(s[start + 1..end].trim().to_string());
        }
    }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        let elem = if let Some(semi) = inner.find(';') {
            inner[..semi].trim()
        } else {
            inner.trim()
        };
        if !elem.is_empty() {
            return Some(elem.to_string());
        }
    }
    None
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
/// Convert OpKind to a typed opname matching RPython's jtransform output.
///
/// RPython jtransform produces fully-qualified names like `getfield_vable_i`,
/// `setfield_gc_r`, `int_add`. The kind suffix comes from the result type
/// or value type of the operation.
fn op_kind_to_opname(kind: &crate::model::OpKind) -> String {
    use crate::model::OpKind;
    match kind {
        OpKind::Input { ty, .. } => format!("input_{}", value_type_to_kind(ty)),
        OpKind::ConstInt(_) => "const_int".into(),
        // RPython: getfield_gc_i, getfield_gc_r, getfield_gc_f and `_pure`
        // variants from jtransform.py rewrite_op_getfield().
        OpKind::FieldRead { ty, pure, .. } => {
            let mut opname = format!("getfield_gc_{}", value_type_to_kind(ty));
            if *pure {
                opname.push_str("_pure");
            }
            opname
        }
        OpKind::FieldWrite { ty, .. } => format!("setfield_gc_{}", value_type_to_kind(ty)),
        // RPython: getarrayitem_gc_i etc.
        OpKind::ArrayRead { item_ty, .. } => {
            format!("getarrayitem_gc_{}", value_type_to_kind(item_ty))
        }
        OpKind::ArrayWrite { item_ty, .. } => {
            format!("setarrayitem_gc_{}", value_type_to_kind(item_ty))
        }
        // RPython: getinteriorfield_gc_i etc.
        OpKind::InteriorFieldRead { item_ty, .. } => {
            format!("getinteriorfield_gc_{}", value_type_to_kind(item_ty))
        }
        OpKind::InteriorFieldWrite { item_ty, .. } => {
            format!("setinteriorfield_gc_{}", value_type_to_kind(item_ty))
        }
        OpKind::Call { result_ty, .. } => {
            format!("direct_call_{}", value_type_to_kind(result_ty))
        }
        OpKind::GuardTrue { .. } => "guard_true".into(),
        OpKind::GuardFalse { .. } => "guard_false".into(),
        OpKind::GuardValue { kind_char, .. } => format!("{kind_char}_guard_value"),
        // RPython: getfield_vable_i, getfield_vable_r, getfield_vable_f
        OpKind::VableFieldRead { ty, .. } => {
            format!("getfield_vable_{}", value_type_to_kind(ty))
        }
        OpKind::VableFieldWrite { ty, .. } => {
            format!("setfield_vable_{}", value_type_to_kind(ty))
        }
        // RPython: getarrayitem_vable_i etc.
        OpKind::VableArrayRead { item_ty, .. } => {
            format!("getarrayitem_vable_{}", value_type_to_kind(item_ty))
        }
        OpKind::VableArrayWrite { item_ty, .. } => {
            format!("setarrayitem_vable_{}", value_type_to_kind(item_ty))
        }
        // RPython `blackhole.py:500` canonical opnames for bitwise ints are
        // `int_and` / `int_or` / `int_xor`. pyre's front-end (`front/ast.rs`
        // `binary_op_name`) uses Rust's `syn::BinOp` trait names
        // (`bitand`/`bitor`/`bitxor`) for source faithfulness, so rename them
        // here at the emission boundary instead of duplicating wire entries
        // in the blackhole dispatch table.
        OpKind::BinOp { op, .. } => match op.as_str() {
            "bitand" => "int_and".into(),
            "bitor" => "int_or".into(),
            "bitxor" => "int_xor".into(),
            _ => format!("int_{op}"),
        },
        // RPython `blackhole.py:488-498`: bitwise NOT on i64 is `int_invert`.
        // pyre's front-end uses Rust's `syn::UnOp::Not` spelling `not` for
        // both logical-not and bitwise-not (they share the `!` token at the
        // AST level); canonicalize to `int_invert` at the emission boundary.
        OpKind::UnaryOp { op, .. } => match op.as_str() {
            "not" => "int_invert".into(),
            _ => format!("int_{op}"),
        },
        OpKind::VableForce => "hint_force_virtualizable".into(),
        // jtransform.py:1731-1743 — jit.* builtin ops
        OpKind::JitDebug { .. } => "jit_debug".into(),
        OpKind::AssertGreen { kind_char, .. } => format!("{kind_char}_assert_green"),
        OpKind::CurrentTraceLength => "current_trace_length".into(),
        OpKind::IsConstant { kind_char, .. } => format!("{kind_char}_isconstant"),
        OpKind::IsVirtual { kind_char, .. } => format!("{kind_char}_isvirtual"),
        OpKind::RecordKnownResult { result_kind, .. } => {
            format!("record_known_result_{result_kind}")
        }
        // jtransform.py:1665-1688 — conditional_call ops
        OpKind::ConditionalCall { .. } => "conditional_call".into(),
        OpKind::ConditionalCallValue { result_kind, .. } => {
            format!("conditional_call_value_{result_kind}")
        }
        OpKind::Live => "live".into(),
        // Call variants are handled by encode_op directly, not here.
        OpKind::CallElidable { .. } => "call_elidable".into(),
        OpKind::CallResidual { .. } => "residual_call".into(),
        OpKind::CallMayForce { .. } => "call_may_force".into(),
        OpKind::InlineCall { .. } => "inline_call".into(),
        OpKind::RecursiveCall { .. } => "recursive_call".into(),
        // RPython: no dedicated opname — the vtable entry becomes the `funcptr`
        // Variable that `int_guard_value` + `residual_call_*` consume.
        OpKind::VtableMethodPtr { .. } => "vtable_method_ptr".into(),
        OpKind::IndirectCall { .. } => "indirect_call".into(),
        // jtransform.py:901-903 — `record_quasiimmut_field(v_inst, descr, descr1)`.
        OpKind::RecordQuasiImmutField { .. } => "record_quasiimmut_field".into(),
        OpKind::Unknown { .. } => "unknown".into(),
    }
}

// Re-export CallInfoCollection from majit-ir (effectinfo.py::CallInfoCollection).
// majit-ir already has the RPython-parity version with OopSpecIndex keys.
pub use majit_ir::CallInfoCollection;

impl Assembler {
    /// RPython: `Assembler.see_raw_object(value)` (assembler.py:283-298).
    ///
    /// Registers a function/vtable name for debugging.
    /// RPython stores `(addr, name)` pairs; majit stores `(path, name)`.
    pub fn see_raw_object(&mut self, path: &str, name: &str) {
        if self.seen_raw_objects.insert(path.to_string()) {
            self.list_of_addr2name
                .push((path.to_string(), name.to_string()));
        }
    }

    /// RPython: `Assembler.finished(callinfocollection)` (assembler.py:300-305).
    ///
    /// ```python
    /// def finished(self, callinfocollection):
    ///     for func in callinfocollection.all_function_addresses_as_int():
    ///         func = int2adr(func)
    ///         self.see_raw_object(func.ptr)
    /// ```
    ///
    /// RPython's `see_raw_object` extracts `func.ptr._obj._name` to build
    /// `list_of_addr2name`. In majit, names are registered at `add()` time
    /// via `register_func_name()`.
    /// RPython: Assembler.insns — the opcode table. Needed by
    /// BlackholeInterpBuilder::setup_insns() to build the dispatch table.
    pub fn insns(&self) -> &HashMap<String, u8> {
        &self.insns
    }

    /// RPython: Assembler.descrs — the descriptor table. Needed by
    /// BlackholeInterpBuilder::setup_descrs() for 'd'/'j' argcode resolution.
    pub fn descrs(&self) -> &[crate::jitcode::BhDescr] {
        &self.descrs
    }

    pub fn finished(&mut self, callinfocollection: &CallInfoCollection) {
        for func_addr in callinfocollection.all_function_addresses_as_int() {
            // RPython: see_raw_object(func.ptr)
            // → name = value._obj._name (for FuncType)
            // → self.list_of_addr2name.append((addr, name))
            let name = callinfocollection.func_name(func_addr).unwrap_or("?");
            let addr_key = format!("{func_addr:#x}");
            self.see_raw_object(&addr_key, name);
        }
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
            insns: vec![],
            num_values: 0,
            num_blocks: 1,
            value_kinds: HashMap::new(),
            insns_pos: None,
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
        let body = asm.assemble(&mut flat, &regallocs);

        assert_eq!(flat.name, "test");
        assert_eq!(body.c_num_regs_i as usize, 0);
        assert_eq!(body.c_num_regs_r as usize, 0);
        assert_eq!(body.c_num_regs_f as usize, 0);
        assert_eq!(asm.count_jitcodes(), 1);
    }

    #[test]
    fn assemble_with_registers() {
        use crate::model::{FunctionGraph, OpKind, Terminator, ValueType};
        // Build graph for regalloc (regalloc operates on graph, not SSARepr)
        let mut graph = FunctionGraph::new("add");
        let entry = graph.startblock;
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
            insns: vec![],
            num_values: 3,
            num_blocks: 1,
            value_kinds,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        // v0 dies when v1 is defined → they share a register → 1 int reg
        assert_eq!(body.c_num_regs_i as usize, 1);
        assert_eq!(body.c_num_regs_r as usize, 1);
        assert_eq!(body.c_num_regs_f as usize, 0);
    }

    #[test]
    fn assemble_direct_residual_call_encodes_leading_funcptr_operand() {
        use crate::call::CallControl;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FunctionGraph, OpKind, Terminator, ValueType};
        use crate::translate_legacy::annotator::annrpython::annotate;
        use crate::translate_legacy::rtyper::rtyper::resolve_types;

        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("caller");
        let arg = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "arg".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: crate::model::CallTarget::function_path(["custom_reader"]),
                args: vec![arg],
                result_ty: ValueType::Ref,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

        let annotations = annotate(&graph);
        let type_state = resolve_types(&graph, &annotations);
        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_type_state(&type_state);
        let rewritten = transformer.transform(&graph);
        let rewritten_types = resolve_types(&rewritten.graph, &annotations);
        let value_kinds =
            crate::translate_legacy::rtyper::rtyper::build_value_kinds(&rewritten_types);
        let regallocs = regalloc::perform_all_register_allocations(&rewritten.graph, &value_kinds);
        let mut flat =
            crate::flatten::flatten_with_types(&rewritten.graph, &rewritten_types, &regallocs);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        // RPython jtransform.py:422-431 canonical order:
        // `residual_call_r_r/iRd>r` (funcptr, R-list, descr, >result).
        assert!(
            asm.insns.contains_key("residual_call_r_r/iRd>r"),
            "expected funcptr-first residual_call key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    /// Boundary test (plan Rev 2 Phase E
    /// `no_legacy_funcptr_from_vtable_after_new_pipeline`): after the
    /// rtyper-equivalent `lower_indirect_calls` pass + jtransform +
    /// assemble, no `funcptr_from_vtable/*` opcode key may survive.
    /// The RPython-orthodox replacement is `vtable_method_ptr/rd>i`
    /// emitted by `OpKind::VtableMethodPtr`
    /// (`translator/rtyper/rclass.rs::class_get_method_ptr` ↔
    /// `rpython/rtyper/rclass.py:371-377` + `rpython/rtyper/rpbc.py:1203-1205`).
    /// `OpKind::FuncptrFromVtable` has been deleted from the enum
    /// (Phase D compile-time guarantee); this test adds a runtime
    /// guarantee at the assembled-opname level.
    #[test]
    fn no_legacy_funcptr_from_vtable_after_new_pipeline() {
        use crate::call::CallControl;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{CallTarget, FunctionGraph, OpKind, Terminator, ValueType};
        use crate::translate_legacy::annotator::annrpython::annotate;
        use crate::translate_legacy::rtyper::rtyper::resolve_types;
        use crate::translator::rtyper::rpbc::lower_indirect_calls;

        fn build_run_impl(name: &str) -> FunctionGraph {
            let mut g = FunctionGraph::new(name);
            g.push_op(
                g.startblock,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
            g.set_terminator(g.startblock, Terminator::Return(None));
            g
        }

        let mut cc = CallControl::new();
        cc.register_trait_method("run", Some("Handler"), "A", build_run_impl("A::run"));
        cc.register_trait_method("run", Some("Handler"), "B", build_run_impl("B::run"));
        cc.find_all_graphs_for_tests();

        let mut graph = FunctionGraph::new("outer");
        let receiver = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "handler".into(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.push_op(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::indirect("Handler", "run"),
                args: vec![receiver],
                result_ty: ValueType::Void,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));

        let annotations = annotate(&graph);
        let mut type_state = resolve_types(&graph, &annotations);
        lower_indirect_calls(&mut graph, &mut type_state, &cc);

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config)
            .with_callcontrol(&mut cc)
            .with_type_state(&type_state);
        let rewritten = transformer.transform(&graph);
        let rewritten_types = resolve_types(&rewritten.graph, &annotations);
        let value_kinds =
            crate::translate_legacy::rtyper::rtyper::build_value_kinds(&rewritten_types);
        let regallocs = regalloc::perform_all_register_allocations(&rewritten.graph, &value_kinds);
        let mut flat =
            crate::flatten::flatten_with_types(&rewritten.graph, &rewritten_types, &regallocs);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        let legacy_keys: Vec<&String> = asm
            .insns
            .keys()
            .filter(|k| k.starts_with("funcptr_from_vtable"))
            .collect();
        assert!(
            legacy_keys.is_empty(),
            "legacy funcptr_from_vtable opname survived after the new \
             lower_indirect_calls + jtransform + assemble pipeline: {:?}",
            legacy_keys,
        );

        let has_vtable_method_ptr = asm.insns.keys().any(|k| k.starts_with("vtable_method_ptr"));
        assert!(
            has_vtable_method_ptr,
            "expected vtable_method_ptr opname from OpKind::VtableMethodPtr, \
             insns keys: {:?}",
            asm.insns.keys().collect::<Vec<_>>(),
        );
    }

    /// `OpKind::RecordQuasiImmutField` must lower to a single opcode
    /// keyed `record_quasiimmut_field/rdd`, with the field+mutate
    /// FieldDescriptor pair pushed as two `BhDescr::Field` entries — see
    /// `rpython/jit/codewriter/jtransform.py:901-903` and
    /// `rpython/jit/metainterp/blackhole.py:1537-1539`.
    #[test]
    fn assembles_record_quasiimmut_field_with_two_descrs() {
        use crate::call::CallControl;
        use crate::flatten::flatten as flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{
            FieldDescriptor, FunctionGraph, ImmutableRank, OpKind, Terminator, ValueType,
        };

        let mut cc = CallControl::new();
        cc.immutable_fields_by_struct.insert(
            "Cell".to_string(),
            vec![("value".to_string(), ImmutableRank::QuasiImmutable)],
        );

        let mut graph = FunctionGraph::new("read_cell");
        let base = graph
            .push_op(
                graph.startblock,
                OpKind::Input {
                    name: "cell".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let result = graph
            .push_op(
                graph.startblock,
                OpKind::FieldRead {
                    base,
                    field: FieldDescriptor::new("value", Some("Cell".to_string())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_terminator(graph.startblock, Terminator::Return(Some(result)));

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let rewritten = transformer.transform(&graph).graph;

        let mut value_kinds = HashMap::new();
        value_kinds.insert(base, RegKind::Ref);
        value_kinds.insert(result, RegKind::Int);
        let regallocs = regalloc::perform_all_register_allocations(&rewritten, &value_kinds);
        let mut flat = flatten_graph(&rewritten, &regallocs);
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        let key_count = asm
            .insns
            .keys()
            .filter(|k| k.starts_with("record_quasiimmut_field/"))
            .count();
        assert_eq!(
            key_count,
            1,
            "expected exactly one record_quasiimmut_field/* key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("record_quasiimmut_field/rdd"),
            "expected key record_quasiimmut_field/rdd, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("getfield_gc_i_pure/rd>i"),
            "expected pure getfield opcode, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        // Two BhDescr::Field entries — for `value` and `mutate_value`.
        let field_descr_names: Vec<&str> = asm
            .descrs
            .iter()
            .filter_map(|d| match d {
                crate::jitcode::BhDescr::Field { name, owner, .. } if owner == "Cell" => {
                    Some(name.as_str())
                }
                _ => None,
            })
            .collect();
        assert!(
            field_descr_names.contains(&"value") && field_descr_names.contains(&"mutate_value"),
            "expected Field descrs for `value` + `mutate_value`, got {:?}",
            field_descr_names
        );
    }
}
