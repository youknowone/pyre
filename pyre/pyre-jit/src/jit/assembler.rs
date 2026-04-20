//! Line-by-line port of `rpython/jit/codewriter/assembler.py`.
//!
//! This module mirrors the high-level `Assembler` flow from RPython:
//! `setup()` → `write_insn()` → `fix_labels()` → `check_result()` →
//! `make_jitcode()`. pyre still targets `majit_metainterp::jitcode::JitCode`
//! and its fixed `BC_*` builder API, so regular-op dispatch lowers the
//! textual `SSARepr` opnames into `JitCodeBuilder` calls instead of growing
//! `self.insns` into a runtime opcode table.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use majit_metainterp::jitcode::{JitCallArg, JitCode, JitCodeBuilder};
use majit_translate::jitcode::BhDescr;

use super::flatten::{DescrOperand, Insn, Kind, ListOfKind, Operand, Register, SSARepr, TLabel};

/// `assembler.py:65` `count_regs = dict.fromkeys(KINDS, 0)` override.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumRegs {
    pub int: u16,
    pub ref_: u16,
    pub float: u16,
}

/// `assembler.py:19-32` `class Assembler(object)`.
///
/// Writer-side state, per-instance, matching `Assembler.__init__`
/// (rpython/jit/codewriter/assembler.py:19-32) line-by-line. A fresh
/// `Assembler::new()` produces independent `all_liveness` /
/// `all_liveness_positions` / `num_liveness_ops` /
/// `indirectcalltargets`; upstream does the same — each `CodeWriter`
/// owns one `Assembler` and carries its own dedup state.
///
/// The reader side (`blackhole` / `trace_opcode`, both in the
/// lower `pyre_jit_trace` crate) still needs to read `all_liveness`
/// at runtime. pyre-jit-trace cannot depend on pyre-jit, so we
/// publish the latest `all_liveness` snapshot to
/// `pyre_jit_trace::assembler::ASSEMBLER_STATE` after every write
/// — PRE-EXISTING-ADAPTATION for the crate layering, not a state
/// relocation.
#[derive(Debug, Default)]
pub struct Assembler {
    /// `assembler.py:29` `self.all_liveness = []`.
    all_liveness: Vec<u8>,
    /// `assembler.py:30` `self.all_liveness_length = 0`.
    all_liveness_length: usize,
    /// `assembler.py:31` `self.all_liveness_positions = {}`.
    all_liveness_positions: HashMap<(Vec<u8>, Vec<u8>, Vec<u8>), u16>,
    /// `assembler.py:32` `self.num_liveness_ops = 0`.
    num_liveness_ops: usize,
    /// `assembler.py:24` `self.indirectcalltargets = set()    # set of JitCodes`.
    /// Accumulated across every `assemble()` call; `pyjitpl.py:2262`
    /// `self.setup_indirectcalltargets(asm.indirectcalltargets)` pipes
    /// this set to `MetaInterpStaticData.setup_indirectcalltargets`.
    pub indirectcalltargets: HashSet<ArcByPtr>,
}

/// Identity-keyed wrapper around `Arc<JitCode>` so `HashSet<ArcByPtr>`
/// matches RPython's `set of JitCodes` (Python set dedupes by object
/// identity, not value equality).  Used as the element type of
/// `Assembler::indirectcalltargets`.
#[derive(Debug, Clone)]
pub struct ArcByPtr(pub Arc<JitCode>);

impl ArcByPtr {
    pub fn new(jc: Arc<JitCode>) -> Self {
        Self(jc)
    }

    pub fn into_inner(self) -> Arc<JitCode> {
        self.0
    }
}

impl PartialEq for ArcByPtr {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ArcByPtr {}

impl std::hash::Hash for ArcByPtr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.0) as *const () as usize).hash(state);
    }
}

/// Builder-local state created by `setup()` for one `assemble()` call.
///
/// RPython keeps most of these on `self`; pyre keeps them in a per-call
/// struct because the runtime `JitCodeBuilder` is consumed by `finish()`.
struct AssemblyState {
    builder: JitCodeBuilder,
    /// `assembler.py:59` `self.label_positions = {}`.
    label_positions: HashMap<String, usize>,
    /// Builder adapter for `Label/TLabel` name → builder label id.
    /// RPython stores bytecode positions directly in `label_positions`; this
    /// extra vector exists only because `JitCodeBuilder` patches jumps by
    /// symbolic label id rather than by rewriting raw bytes in `fix_labels()`.
    builder_labels: Vec<(String, u16)>,
    /// SSARepr-side switch descrs that must be attached after all labels have
    /// final positions (`assembler.py:258-263`).
    switch_descrs: Vec<(usize, Vec<(i64, TLabel)>)>,
    /// Runtime descr table for this jitcode (`assembler.py:26` `self.descrs`).
    descrs: Vec<BhDescr>,
    /// `assembler.py:26` `self._descr_dict = {}` — identity-keyed dedup so
    /// re-using the same SSARepr descr across multiple ops yields a stable
    /// `descrs` index. `DescrOperand` has no inherent identity in Rust;
    /// the key is the Vec-slot pointer captured when the `Operand` was
    /// first interned — semantically equivalent to Python's `id()` lookup.
    descr_dict: HashMap<DescrKey, usize>,
}

/// Identity key for `DescrOperand` dedup. Matches
/// `assembler.py:197-199` `if x not in self._descr_dict` which uses
/// Python object identity.
///
/// `Bh` descrs are deduped by pointer-equality against the `DescrOperand`
/// borrow (the SSARepr owns the operand, so pointers are stable during
/// assemble()). `SwitchDict` descrs must be deduped by the same rule —
/// two distinct SwitchDictDescr operands produce two `descrs` entries
/// even if their `labels` happen to match (matches RPython's identity
/// semantics).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DescrKey(*const DescrOperand);

impl Assembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// `assembler.py:29` accessor.
    pub fn all_liveness(&self) -> &[u8] {
        &self.all_liveness
    }

    /// `assembler.py:30` accessor.
    pub fn all_liveness_length(&self) -> usize {
        self.all_liveness_length
    }

    /// `assembler.py:32` accessor.
    pub fn num_liveness_ops(&self) -> usize {
        self.num_liveness_ops
    }

    /// Return the accumulated indirect-call target set as a deduped
    /// `Vec<Arc<JitCode>>`.  Matches the shape expected by
    /// `MetaInterpStaticData::setup_indirectcalltargets` which pipes
    /// `asm.indirectcalltargets` into the staticdata per
    /// `pyjitpl.py:2262`.
    pub fn indirectcalltargets_vec(&self) -> Vec<Arc<JitCode>> {
        self.indirectcalltargets
            .iter()
            .map(|a| a.0.clone())
            .collect()
    }

    /// `assembler.py:300-305` `finished(self, callinfocollection)`.
    ///
    /// ```python
    /// def finished(self, callinfocollection):
    ///     # Helper called at the end of assembling.  Registers the extra
    ///     # functions shown in _callinfo_for_oopspec.
    ///     for func in callinfocollection.all_function_addresses_as_int():
    ///         func = int2adr(func)
    ///         self.see_raw_object(func.ptr)
    /// ```
    ///
    /// Called from `codewriter.py:85` after the `enum_pending_graphs`
    /// drain loop completes; registers raw helper-function addresses
    /// into `list_of_addr2name` for `MetaInterpStaticData`'s
    /// debug-symbol map.
    ///
    /// PRE-EXISTING-ADAPTATION: pyre has no `oopspec` system, so the
    /// callinfocollection is empty (see
    /// [`super::call::CallInfoCollection`]) and the iteration is a
    /// no-op. The slot is preserved so `make_jitcodes` calls through
    /// at exactly the same point as `codewriter.py:85`.
    pub fn finished(&mut self, callinfocollection: &super::call::CallInfoCollection) {
        let _ = callinfocollection;
    }

    /// `assembler.py:34-54` `assemble(self, ssarepr, jitcode=None, num_regs=None)`.
    pub fn assemble(
        &mut self,
        ssarepr: &mut SSARepr,
        mut builder: JitCodeBuilder,
        num_regs: Option<NumRegs>,
    ) -> JitCode {
        if let Some(nr) = num_regs {
            builder.ensure_i_regs(nr.int);
            builder.ensure_r_regs(nr.ref_);
            builder.ensure_f_regs(nr.float);
        }
        builder.set_name(ssarepr.name.clone());

        let mut state = AssemblyState {
            builder,
            label_positions: HashMap::new(),
            builder_labels: Vec::new(),
            switch_descrs: Vec::new(),
            descrs: Vec::new(),
            descr_dict: HashMap::new(),
        };

        ssarepr.insns_pos = Some(Vec::with_capacity(ssarepr.insns.len()));
        for insn in &ssarepr.insns {
            ssarepr
                .insns_pos
                .as_mut()
                .expect("insns_pos initialized")
                .push(state.builder.current_pos());
            self.write_insn(&mut state, insn);
        }

        self.fix_labels(&mut state);

        let mut jitcode = state.builder.finish();
        jitcode.descrs = state.descrs;
        // `assembler.py:54`: run `check_result()` unconditionally after
        // `fix_labels()` and before `make_jitcode`. Strict RPython parity
        // — an oversized jitcode is a translator-level programming error,
        // so this asserts and aborts rather than degrading to a runtime
        // fallback. Any pyre-only safety net belongs outside this unit.
        self.check_result(&jitcode);
        jitcode
    }

    /// `assembler.py:140-223` `write_insn(insn)`.
    fn write_insn(&mut self, state: &mut AssemblyState, insn: &Insn) {
        match insn {
            Insn::Unreachable => {}
            Insn::Label(label) => {
                let label_id = builder_label(state, &label.name);
                state
                    .label_positions
                    .insert(label.name.clone(), state.builder.current_pos());
                state.builder.mark_label(label_id);
            }
            Insn::Live(args) => {
                // `assembler.py:149` increments `self.num_liveness_ops`.
                // pyre's counter lives on `ASSEMBLER_STATE`; `intern_liveness`
                // below takes care of the `+= 1`, so there is no local
                // pre-increment here (matching upstream: the bump happens
                // inside `_encode_liveness`).
                let live_i = get_liveness_info(args, Kind::Int);
                let live_r = get_liveness_info(args, Kind::Ref);
                let live_f = get_liveness_info(args, Kind::Float);
                let patch_offset = state.builder.live_placeholder();
                let offset = self.encode_liveness_info(&live_i, &live_r, &live_f);
                state.builder.patch_live_offset(patch_offset, offset);
            }
            Insn::Op {
                opname,
                args,
                result,
            } => {
                // `assembler.py:208-209`:
                //   elif isinstance(x, IndirectCallTargets):
                //       self.indirectcalltargets.update(x.lst)
                //
                // RPython's `write_insn` iterates every operand and dispatches
                // by Python type.  pyre's `dispatch_op` is keyed on `opname`
                // (PRE-EXISTING-ADAPTATION — see below), so operand iteration
                // is split: `IndirectCallTargets` is collected here before the
                // opcode-specific lowering runs.  The operand is purely
                // metadata for the call — it is not written into the jitcode
                // byte stream (RPython emits nothing for this branch either).
                for arg in args {
                    if let Operand::IndirectCallTargets(targets) = arg {
                        for jc in &targets.lst {
                            self.indirectcalltargets.insert(ArcByPtr::new(jc.clone()));
                        }
                    }
                }
                // `assembler.py:197-206` registers descrs inline during op
                // encoding, guarded by `self._descr_dict`. The previous
                // pre-scan violated parity by interning every
                // `Operand::Descr` unconditionally (even on ops that don't
                // consume a descr in the emitted bytecode) and without
                // dedup. Dispatch arms that actually need a descr index
                // now call `record_descr_operand(state, descr)` themselves.
                dispatch_op(state, opname, args, result.as_ref());
            }
        }
    }

    /// `assembler.py:250-263` `fix_labels()`.
    fn fix_labels(&mut self, state: &mut AssemblyState) {
        for (descr_index, labels) in &state.switch_descrs {
            let mut dict = HashMap::new();
            for (key, label) in labels {
                let target = *state
                    .label_positions
                    .get(&label.name)
                    .unwrap_or_else(|| panic!("missing switch target label {:?}", label.name));
                dict.insert(*key, target);
            }
            state.descrs[*descr_index] = BhDescr::Switch { dict };
        }
    }

    /// `assembler.py:265-269` `check_result()`.
    ///
    /// RPython enforces a 256-entry cap per kind because the assembled
    /// bytecode stream treats register-or-constant operands as one-byte
    /// indices. The surrounding pyre pipeline still relies on the same
    /// invariant for liveness and jit-merge-point operand encoding, so we
    /// keep the orthodox translator-time assertion here instead of
    /// widening the limit to match internal `u16` builder fields.
    fn check_result(&self, jitcode: &JitCode) {
        assert!(
            (jitcode.num_regs_i() as usize) + jitcode.constants_i.len() <= 256,
            "too many int registers/constants"
        );
        assert!(
            (jitcode.num_regs_r() as usize) + jitcode.constants_r.len() <= 256,
            "too many ref registers/constants"
        );
        assert!(
            (jitcode.num_regs_f() as usize) + jitcode.constants_f.len() <= 256,
            "too many float registers/constants"
        );
    }

    /// `assembler.py:234-248` `_encode_liveness(...)`.
    ///
    /// Line-by-line port of RPython's `_encode_liveness`:
    ///   - Dedup key is `(live_i, live_r, live_f)` — pyre uses sorted-byte
    ///     slices where RPython uses `frozenset`. Equivalent modulo the
    ///     caller's responsibility to present sorted/dedup'd input, which
    ///     is done by `get_liveness_info` before this call.
    ///   - On miss: append the three u8 length bytes + the encoded
    ///     bitsets, and advance `all_liveness_length`.
    ///   - `num_liveness_ops` bumps once per `-live-` write regardless of
    ///     dedup (`assembler.py:149` does it in `write_insn`, before this
    ///     helper is reached). Matches upstream.
    ///
    /// After the per-instance update we publish the latest buffer to
    /// the `pyre_jit_trace::assembler::ASSEMBLER_STATE` thread-local so
    /// the blackhole reader (a lower-crate consumer that cannot depend
    /// on `pyre_jit`) sees the same bytes. PRE-EXISTING-ADAPTATION for
    /// the crate layering, not a relocation of the RPython state.
    fn encode_liveness_info(&mut self, live_i: &[u8], live_r: &[u8], live_f: &[u8]) -> u16 {
        use majit_translate::liveness::encode_liveness;

        self.num_liveness_ops += 1;
        let key = (live_i.to_vec(), live_r.to_vec(), live_f.to_vec());
        let pos = if let Some(&cached) = self.all_liveness_positions.get(&key) {
            cached
        } else {
            assert!(
                self.all_liveness_length <= u16::MAX as usize,
                "all_liveness offset overflow"
            );
            assert!(
                live_i.len() <= u8::MAX as usize
                    && live_r.len() <= u8::MAX as usize
                    && live_f.len() <= u8::MAX as usize,
                "too many live registers in one slot"
            );
            let pos = self.all_liveness_length as u16;
            self.all_liveness_positions.insert(key, pos);
            self.all_liveness.push(live_i.len() as u8);
            self.all_liveness.push(live_r.len() as u8);
            self.all_liveness.push(live_f.len() as u8);
            self.all_liveness_length += 3;
            for live in [live_i, live_r, live_f] {
                let encoded = encode_liveness(live);
                self.all_liveness.extend_from_slice(&encoded);
                self.all_liveness_length += encoded.len();
            }
            pos
        };
        pyre_jit_trace::assembler::publish_liveness(
            &self.all_liveness,
            self.all_liveness_length,
            self.num_liveness_ops,
        );
        pos
    }
}

/// Test-only convenience wrapper — creates a throwaway `Assembler` per
/// call. **Do not wire this into production paths.** RPython's
/// `CodeWriter` holds a single `Assembler` whose `all_liveness`,
/// `all_liveness_positions`, and `num_liveness_ops` accumulate across
/// every JitCode compiled in the session (`codewriter.py:19-22` /
/// `pyjitpl.py:2264`). A fresh `Assembler` per call splits the
/// global liveness table and breaks that invariant.
///
/// Production callers must construct one `Assembler` up front and call
/// `assembler.assemble(...)` repeatedly on it.
#[cfg(test)]
fn assemble(ssarepr: &mut SSARepr, builder: JitCodeBuilder, num_regs: Option<NumRegs>) -> JitCode {
    let mut assembler = Assembler::new();
    assembler.assemble(ssarepr, builder, num_regs)
}

fn builder_label(state: &mut AssemblyState, name: &str) -> u16 {
    if let Some((_, label)) = state
        .builder_labels
        .iter()
        .find(|(existing, _)| existing == name)
    {
        return *label;
    }
    let label = state.builder.new_label();
    state.builder_labels.push((name.to_owned(), label));
    label
}

fn get_liveness_info(args: &[Operand], kind: Kind) -> Vec<u8> {
    let mut lives = Vec::new();
    for arg in args {
        if let Operand::Register(Register {
            kind: reg_kind,
            index,
        }) = arg
        {
            if *reg_kind == kind {
                lives.push(u8::try_from(*index).expect("liveness register index exceeds u8"));
            }
        }
    }
    lives.sort_unstable();
    lives.dedup();
    lives
}

fn dispatch_op(
    state: &mut AssemblyState,
    opname: &str,
    args: &[Operand],
    result: Option<&Register>,
) {
    match opname {
        "goto" | "jump" => {
            let label = expect_tlabel(&args[0]);
            let label_id = builder_label(state, &label.name);
            state.builder.jump(label_id);
        }
        "goto_if_not" | "branch_reg_zero" => {
            let cond = expect_reg(&args[0], Kind::Int);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.branch_reg_zero(cond, label_id);
        }
        "goto_if_not_int_eq" => {
            let lhs = expect_reg(&args[0], Kind::Int);
            let rhs = expect_reg(&args[1], Kind::Int);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_eq(lhs, rhs, label_id);
        }
        "goto_if_not_int_ne" => {
            let lhs = expect_reg(&args[0], Kind::Int);
            let rhs = expect_reg(&args[1], Kind::Int);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_ne(lhs, rhs, label_id);
        }
        "goto_if_not_int_is_zero" => {
            let cond = expect_reg(&args[0], Kind::Int);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_is_zero(cond, label_id);
        }
        "catch_exception" => {
            let label = expect_tlabel(&args[0]);
            let label_id = builder_label(state, &label.name);
            state.builder.catch_exception(label_id);
        }
        "jit_merge_point" => {
            let greens_i = expect_list_regs(&args[0], Kind::Int);
            let greens_r = expect_list_regs(&args[1], Kind::Ref);
            let reds_r = expect_list_regs(&args[2], Kind::Ref);
            state.builder.jit_merge_point(&greens_i, &greens_r, &reds_r);
        }
        "loop_header" => {
            // RPython jtransform.py:1714-1718 handle_jit_marker__loop_header
            // emits SpaceOperation('loop_header', [c_index], None). Arg is
            // the jitdriver index as a Signed constant (pyre has a single
            // jitdriver, so jdindex is always 0).
            let jdindex = match &args[0] {
                Operand::ConstInt(v) => *v,
                other => panic!("loop_header expects ConstInt jdindex, got {:?}", other),
            };
            let jdindex: u8 = u8::try_from(jdindex)
                .unwrap_or_else(|_| panic!("loop_header jdindex {jdindex} out of u8 range"));
            state.builder.loop_header(jdindex);
        }
        "ref_return" => {
            let src = expect_result_or_first_reg(args, result, Kind::Ref);
            state.builder.ref_return(src);
        }
        "raise" => {
            let src = expect_reg(&args[0], Kind::Ref);
            state.builder.emit_raise(src);
        }
        "reraise" => state.builder.emit_reraise(),
        "last_exc_value" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            state.builder.last_exc_value(dst);
        }
        "abort" => state.builder.abort(),
        "abort_permanent" => state.builder.abort_permanent(),
        // `flatten.py:333` `self.emitline('%s_copy' % kind, v, "->", w)`.
        // `v` is either a `Register` or a `Constant`. Register source
        // lowers to `move_{i,r,f}`; Constant source lowers to
        // `load_const_{i,r,f}_value`, matching the two argcode variants
        // emitted by `assembler.py:162-174` (`i` for Register, `c` for
        // Constant). The `move_*` aliases are the pyre-only pre-parity
        // names that the B6 Phase 3b migration replaces with
        // `{kind}_copy`.
        "move_i" | "int_copy" => match source_operand(args, result) {
            MoveSource::Reg(src) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Int);
                let src = expect_reg(src, Kind::Int);
                state.builder.move_i(dst, src);
            }
            MoveSource::ConstInt(value) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Int);
                state.builder.load_const_i_value(dst, value);
            }
            other => panic!("int_copy expects Register or ConstInt, got {:?}", other),
        },
        "move_r" | "ref_copy" => match source_operand(args, result) {
            MoveSource::Reg(src) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Ref);
                let src = expect_reg(src, Kind::Ref);
                state.builder.move_r(dst, src);
            }
            MoveSource::ConstRef(value) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Ref);
                state.builder.load_const_r_value(dst, value);
            }
            other => panic!("ref_copy expects Register or ConstRef, got {:?}", other),
        },
        "move_f" | "float_copy" => match source_operand(args, result) {
            MoveSource::Reg(src) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Float);
                let src = expect_reg(src, Kind::Float);
                state.builder.move_f(dst, src);
            }
            MoveSource::ConstFloat(value) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Float);
                state.builder.load_const_f_value(dst, value);
            }
            other => panic!("float_copy expects Register or ConstFloat, got {:?}", other),
        },
        "load_const_i" => {
            let (dst, value) = expect_load_const_i(args, result);
            state.builder.load_const_i_value(dst, value);
        }
        "load_const_r" => {
            let (dst, value) = expect_load_const_r(args, result);
            state.builder.load_const_r_value(dst, value);
        }
        "load_const_f" => {
            let (dst, value) = expect_load_const_f(args, result);
            state.builder.load_const_f_value(dst, value);
        }
        "load_state_field" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state
                .builder
                .load_state_field(expect_small_u16(&args[0]), dst);
        }
        "store_state_field" => {
            state
                .builder
                .store_state_field(expect_small_u16(&args[0]), expect_reg(&args[1], Kind::Int));
        }
        "load_state_array" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state.builder.load_state_array(
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
                dst,
            );
        }
        "store_state_array" => {
            state.builder.store_state_array(
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
                expect_reg(&args[2], Kind::Int),
            );
        }
        "load_state_varray" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state.builder.load_state_varray(
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
                dst,
            );
        }
        "store_state_varray" => {
            state.builder.store_state_varray(
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
                expect_reg(&args[2], Kind::Int),
            );
        }
        // RPython opname parity (jtransform.py:765-927, blackhole.py:1374-1493):
        // SpaceOperation names use `*_vable_*` infix. pyre's JitCodeBuilder
        // methods retain `vable_*` prefix as a PRE-EXISTING-ADAPTATION so
        // the rename is scoped to the Insn::Op key.
        // `rpython/jit/codewriter/jtransform.py:844,923` emits field
        // accessors with the FULL kind name (`getfield_vable_int`,
        // `setfield_vable_ref`, etc.). The short-form aliases
        // (`getfield_vable_i`, …) are pre-existing pyre-side dispatch
        // arms kept here for backwards compatibility with any caller
        // that has not yet migrated to the RPython-parity names.
        "getfield_vable_int" | "getfield_vable_i" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state
                .builder
                .vable_getfield_int(dst, expect_small_u16(&args[0]));
        }
        "getfield_vable_ref" | "getfield_vable_r" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            state
                .builder
                .vable_getfield_ref(dst, expect_small_u16(&args[0]));
        }
        "getfield_vable_float" | "getfield_vable_f" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Float);
            state
                .builder
                .vable_getfield_float(dst, expect_small_u16(&args[0]));
        }
        "setfield_vable_int" | "setfield_vable_i" => state
            .builder
            .vable_setfield_int(expect_small_u16(&args[0]), expect_reg(&args[1], Kind::Int)),
        "setfield_vable_ref" | "setfield_vable_r" => state
            .builder
            .vable_setfield_ref(expect_small_u16(&args[0]), expect_reg(&args[1], Kind::Ref)),
        "setfield_vable_float" | "setfield_vable_f" => state.builder.vable_setfield_float(
            expect_small_u16(&args[0]),
            expect_reg(&args[1], Kind::Float),
        ),
        "getarrayitem_vable_i" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state.builder.vable_getarrayitem_int(
                dst,
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "getarrayitem_vable_r" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            state.builder.vable_getarrayitem_ref(
                dst,
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "getarrayitem_vable_f" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Float);
            state.builder.vable_getarrayitem_float(
                dst,
                expect_small_u16(&args[0]),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "setarrayitem_vable_i" => state.builder.vable_setarrayitem_int(
            expect_small_u16(&args[0]),
            expect_reg(&args[1], Kind::Int),
            expect_reg(&args[2], Kind::Int),
        ),
        "setarrayitem_vable_r" => state.builder.vable_setarrayitem_ref(
            expect_small_u16(&args[0]),
            expect_reg(&args[1], Kind::Int),
            expect_reg(&args[2], Kind::Ref),
        ),
        "setarrayitem_vable_f" => state.builder.vable_setarrayitem_float(
            expect_small_u16(&args[0]),
            expect_reg(&args[1], Kind::Int),
            expect_reg(&args[2], Kind::Float),
        ),
        "arraylen_vable" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state
                .builder
                .vable_arraylen(dst, expect_small_u16(&args[0]));
        }
        "hint_force_virtualizable" => state.builder.vable_force(),
        "record_binop_i" => {
            let dst = expect_result_reg(result, Kind::Int, "record_binop_i needs result");
            state.builder.record_binop_i(
                dst,
                expect_opcode(&args[0]),
                expect_reg(&args[1], Kind::Int),
                expect_reg(&args[2], Kind::Int),
            );
        }
        // `rpython/jit/metainterp/resoperation.py` / `jtransform.py`
        // emit per-OpCode opnames for integer comparisons and
        // arithmetic. pyre routes these through the same
        // `record_binop_i` builder (no IR-level difference because
        // pyre's tracer emits the specific `OpCode` at trace time),
        // but the SSARepr carries the RPython-parity opname so the
        // Phase 3c switchover can reproduce the byte-level dispatch
        // without a pyre-only `record_binop_i` entry.
        "int_eq" => {
            let dst = expect_result_reg(result, Kind::Int, "int_eq needs result");
            state.builder.record_binop_i(
                dst,
                majit_ir::OpCode::IntEq,
                expect_reg(&args[0], Kind::Int),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "record_binop_f" => {
            let dst = expect_result_reg(result, Kind::Float, "record_binop_f needs result");
            state.builder.record_binop_f(
                dst,
                expect_opcode(&args[0]),
                expect_reg(&args[1], Kind::Float),
                expect_reg(&args[2], Kind::Float),
            );
        }
        "record_unary_i" => {
            let dst = expect_result_reg(result, Kind::Int, "record_unary_i needs result");
            state.builder.record_unary_i(
                dst,
                expect_opcode(&args[0]),
                expect_reg(&args[1], Kind::Int),
            );
        }
        "record_unary_f" => {
            let dst = expect_result_reg(result, Kind::Float, "record_unary_f needs result");
            state.builder.record_unary_f(
                dst,
                expect_opcode(&args[0]),
                expect_reg(&args[1], Kind::Float),
            );
        }
        "call_void" | "residual_call_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .residual_call_void_typed_args(fn_idx, &call_args);
        }
        "call_may_force_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_may_force_void_typed_args(fn_idx, &call_args);
        }
        "call_release_gil_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_release_gil_void_typed_args(fn_idx, &call_args);
        }
        "call_loopinvariant_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_loopinvariant_void_typed_args(fn_idx, &call_args);
        }
        "call_assembler_void" => {
            let target_idx = expect_small_u16(&args[0]);
            let call_args = expect_call_args(&args[1..]);
            state
                .builder
                .call_assembler_void_typed_args(target_idx, &call_args);
        }
        "call_int"
        | "call_pure_int"
        | "call_may_force_int"
        | "call_release_gil_int"
        | "call_loopinvariant_int"
        | "call_assembler_int" => {
            let fn_idx = expect_small_u16(&args[0]);
            let dst = expect_result_reg(result, Kind::Int, "int call needs result");
            let call_args = expect_call_args(&args[1..]);
            match opname {
                "call_int" => state.builder.call_int_typed(fn_idx, &call_args, dst),
                "call_pure_int" => state.builder.call_pure_int_typed(fn_idx, &call_args, dst),
                "call_may_force_int" => state
                    .builder
                    .call_may_force_int_typed(fn_idx, &call_args, dst),
                "call_release_gil_int" => state
                    .builder
                    .call_release_gil_int_typed(fn_idx, &call_args, dst),
                "call_loopinvariant_int" => state
                    .builder
                    .call_loopinvariant_int_typed(fn_idx, &call_args, dst),
                _ => state
                    .builder
                    .call_assembler_int_typed(fn_idx, &call_args, dst),
            }
        }
        "call_ref"
        | "call_pure_ref"
        | "call_may_force_ref"
        | "call_release_gil_ref"
        | "call_loopinvariant_ref"
        | "call_assembler_ref" => {
            let fn_idx = expect_small_u16(&args[0]);
            let dst = expect_result_reg(result, Kind::Ref, "ref call needs result");
            let call_args = expect_call_args(&args[1..]);
            match opname {
                "call_ref" => state.builder.call_ref_typed(fn_idx, &call_args, dst),
                "call_pure_ref" => state.builder.call_pure_ref_typed(fn_idx, &call_args, dst),
                "call_may_force_ref" => state
                    .builder
                    .call_may_force_ref_typed(fn_idx, &call_args, dst),
                "call_release_gil_ref" => state
                    .builder
                    .call_release_gil_ref_typed(fn_idx, &call_args, dst),
                "call_loopinvariant_ref" => state
                    .builder
                    .call_loopinvariant_ref_typed(fn_idx, &call_args, dst),
                _ => state
                    .builder
                    .call_assembler_ref_typed(fn_idx, &call_args, dst),
            }
        }
        "call_float"
        | "call_pure_float"
        | "call_may_force_float"
        | "call_release_gil_float"
        | "call_loopinvariant_float"
        | "call_assembler_float" => {
            let fn_idx = expect_small_u16(&args[0]);
            let dst = expect_result_reg(result, Kind::Float, "float call needs result");
            let call_args = expect_call_args(&args[1..]);
            match opname {
                "call_float" => state.builder.call_float_typed(fn_idx, &call_args, dst),
                "call_pure_float" => state.builder.call_pure_float_typed(fn_idx, &call_args, dst),
                "call_may_force_float" => state
                    .builder
                    .call_may_force_float_typed(fn_idx, &call_args, dst),
                "call_release_gil_float" => state
                    .builder
                    .call_release_gil_float_typed(fn_idx, &call_args, dst),
                "call_loopinvariant_float" => state
                    .builder
                    .call_loopinvariant_float_typed(fn_idx, &call_args, dst),
                _ => state
                    .builder
                    .call_assembler_float_typed(fn_idx, &call_args, dst),
            }
        }
        "conditional_call_void" => {
            let fn_idx = expect_small_u16(&args[0]);
            let cond_reg = expect_reg(&args[1], Kind::Int);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_void_typed_args(fn_idx, cond_reg, &call_args);
        }
        "conditional_call_value_int" => {
            let fn_idx = expect_small_u16(&args[0]);
            let value_reg = expect_reg(&args[1], Kind::Int);
            let dst =
                expect_result_reg(result, Kind::Int, "conditional_call_value_int needs result");
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_value_int_typed_args(fn_idx, value_reg, &call_args, dst);
        }
        "conditional_call_value_ref" => {
            let fn_idx = expect_small_u16(&args[0]);
            let value_reg = expect_reg(&args[1], Kind::Int);
            let dst =
                expect_result_reg(result, Kind::Ref, "conditional_call_value_ref needs result");
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_value_ref_typed_args(fn_idx, value_reg, &call_args, dst);
        }
        "record_known_result_int" => {
            let fn_idx = expect_small_u16(&args[0]);
            let result_reg = expect_reg(&args[1], Kind::Int);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .record_known_result_int_typed_args(fn_idx, result_reg, &call_args);
        }
        "record_known_result_ref" => {
            let fn_idx = expect_small_u16(&args[0]);
            let result_reg = expect_reg(&args[1], Kind::Ref);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .record_known_result_ref_typed_args(fn_idx, result_reg, &call_args);
        }
        // `rpython/jit/codewriter/jtransform.py:414-435 rewrite_call` shape.
        // 12 opname combinations: residual_call_{r,ir,irf}_{i,r,f,v}. Only
        // the subset the walker currently emits is routed; unused arms
        // panic with a pointer to emit_residual_call.
        opname if opname.starts_with("residual_call_") => {
            dispatch_residual_call(state, opname, args, result);
        }
        other => panic!(
            "assemble(): unimplemented opname {:?} — add a builder mapping in jit/assembler.rs",
            other
        ),
    }
}

/// Phase 3c consumer for the `residual_call_{kinds}_{reskind}` shape
/// emitted by `codewriter::emit_residual_call_shape`.
///
/// SSARepr arg layout (as produced by `emit_residual_call_shape`):
/// ```text
/// [Const(fn_idx),
///  ListOfKind(int,   [...])?,   # present iff 'i' in kinds
///  ListOfKind(ref,   [...])?,   # present iff 'r' in kinds
///  ListOfKind(float, [...])?,   # present iff 'f' in kinds
///  Descr(CallDescrStub{flavor, arg_kinds})]
/// ```
///
/// pyre's `JitCodeBuilder::call_*_typed` takes a single flat
/// `&[JitCallArg]` in C-function parameter order. `arg_kinds` carries
/// that order; this helper reassembles by pulling one entry at a time
/// from the appropriate per-kind sublist in upstream's `bh_call_*`
/// pool layout.
fn dispatch_residual_call(
    state: &mut AssemblyState,
    opname: &str,
    args: &[Operand],
    result: Option<&Register>,
) {
    // opname suffix = `{kinds}_{reskind}`, kinds ∈ {r, ir, irf}.
    let tail = &opname["residual_call_".len()..];
    let (kinds, reskind_ch) = tail
        .rsplit_once('_')
        .unwrap_or_else(|| panic!("malformed residual_call opname: {opname:?}"));

    let fn_idx = expect_small_u16(&args[0]);

    // Walk `args[1..]`: per-kind ListOfKind sublists come first in
    // (int, ref, float) order per the `kinds` string, then the final
    // Descr operand carrying `CallDescrStub`.
    let mut cursor = 1usize;
    let mut int_list: &[Operand] = &[];
    let mut ref_list: &[Operand] = &[];
    let mut float_list: &[Operand] = &[];
    if kinds.contains('i') {
        match &args[cursor] {
            Operand::ListOfKind(ListOfKind {
                kind: Kind::Int,
                content,
            }) => int_list = content,
            other => {
                panic!("residual_call: expected ListOfKind(int) at index {cursor}, got {other:?}")
            }
        }
        cursor += 1;
    }
    if kinds.contains('r') {
        match &args[cursor] {
            Operand::ListOfKind(ListOfKind {
                kind: Kind::Ref,
                content,
            }) => ref_list = content,
            other => {
                panic!("residual_call: expected ListOfKind(ref) at index {cursor}, got {other:?}")
            }
        }
        cursor += 1;
    }
    if kinds.contains('f') {
        match &args[cursor] {
            Operand::ListOfKind(ListOfKind {
                kind: Kind::Float,
                content,
            }) => float_list = content,
            other => {
                panic!("residual_call: expected ListOfKind(float) at index {cursor}, got {other:?}")
            }
        }
        cursor += 1;
    }
    // CallDescrStub slot.
    let stub = match &args[cursor] {
        Operand::Descr(rc) => match &**rc {
            DescrOperand::CallDescrStub(stub) => stub,
            other => panic!(
                "residual_call: expected CallDescrStub descr at index {cursor}, got {other:?}"
            ),
        },
        other => panic!("residual_call: expected Descr operand at index {cursor}, got {other:?}"),
    };

    // Reassemble flat `&[JitCallArg]` in C-function parameter order.
    let mut call_args: Vec<JitCallArg> = Vec::with_capacity(stub.arg_kinds.len());
    let mut i_cursor = 0usize;
    let mut r_cursor = 0usize;
    let mut f_cursor = 0usize;
    for &ak in &stub.arg_kinds {
        match ak {
            Kind::Int => {
                let reg = expect_reg(&int_list[i_cursor], Kind::Int);
                call_args.push(JitCallArg::int(reg));
                i_cursor += 1;
            }
            Kind::Ref => {
                let reg = expect_reg(&ref_list[r_cursor], Kind::Ref);
                call_args.push(JitCallArg::reference(reg));
                r_cursor += 1;
            }
            Kind::Float => {
                let reg = expect_reg(&float_list[f_cursor], Kind::Float);
                call_args.push(JitCallArg::float(reg));
                f_cursor += 1;
            }
        }
    }

    use super::flatten::{CallFlavor, ResKind};
    let reskind = match reskind_ch {
        "i" => ResKind::Int,
        "r" => ResKind::Ref,
        "f" => ResKind::Float,
        "v" => ResKind::Void,
        other => panic!("residual_call: unknown reskind {other:?}"),
    };

    // Pick the same builder method the optimizeopt layer resolves from
    // `EffectInfo.extraeffect`. Reskind drives the return-value kind;
    // flavor drives the effect branch.
    match (stub.flavor, reskind) {
        (CallFlavor::Plain, ResKind::Int) => {
            let dst = expect_result_reg(result, Kind::Int, "residual_call int needs result");
            state.builder.call_int_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::Plain, ResKind::Ref) => {
            let dst = expect_result_reg(result, Kind::Ref, "residual_call ref needs result");
            state.builder.call_ref_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::Plain, ResKind::Void) => {
            state
                .builder
                .residual_call_void_typed_args(fn_idx, &call_args);
        }
        (CallFlavor::MayForce, ResKind::Int) => {
            let dst = expect_result_reg(
                result,
                Kind::Int,
                "residual_call may_force int needs result",
            );
            state
                .builder
                .call_may_force_int_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::MayForce, ResKind::Ref) => {
            let dst = expect_result_reg(
                result,
                Kind::Ref,
                "residual_call may_force ref needs result",
            );
            state
                .builder
                .call_may_force_ref_typed(fn_idx, &call_args, dst);
        }
        (CallFlavor::MayForce, ResKind::Void) => {
            state
                .builder
                .call_may_force_void_typed_args(fn_idx, &call_args);
        }
        // Remaining (flavor, reskind) combinations are reachable once more
        // bytecode handlers use them. Add arms as needed; the panic keeps
        // the pyre-jit walker in sync with this dispatch table.
        (flavor, reskind) => panic!(
            "dispatch_residual_call: unsupported (flavor, reskind) = ({:?}, {:?})",
            flavor, reskind
        ),
    }
}

/// `assembler.py:197-206` inline descr registration. Called from
/// dispatch arms that actually consume a descr operand in the emitted
/// bytecode.
///
/// Deduplicates on `Operand::Descr` pointer identity — matches
/// RPython's `if x not in self._descr_dict` which hashes by object id.
/// Reusing the same SSARepr descr operand twice yields a stable
/// `descrs` index both times.
fn record_descr_operand(state: &mut AssemblyState, descr: &DescrOperand) -> usize {
    let key = DescrKey(descr as *const DescrOperand);
    if let Some(&idx) = state.descr_dict.get(&key) {
        return idx;
    }
    let index = state.descrs.len();
    state.descr_dict.insert(key, index);
    match descr {
        DescrOperand::Bh(bh) => state.descrs.push(bh.clone()),
        DescrOperand::SwitchDict(switch) => {
            state.descrs.push(BhDescr::Switch {
                dict: HashMap::new(),
            });
            state.switch_descrs.push((index, switch.labels.clone()));
        }
        DescrOperand::CallDescrStub(_) => {
            // `CallDescrStub` stands in for the codewriter-layer
            // `calldescr` at the SSARepr level but does not correspond to
            // a runtime `BhDescr`. The `residual_call_*` dispatch arms
            // consume the stub directly (`dispatch_op` below) to pick the
            // builder method and reassemble per-arg kinds, so the argcode
            // `d` emission path must never see it.
            panic!(
                "record_descr_operand: CallDescrStub must be consumed by dispatch_op, not encoded as 'd'"
            );
        }
    }
    index
}

/// Helper for dispatch arms: decode an `Operand::Descr`, register it
/// inline, and return the 16-bit descr index. Used by ops that actually
/// emit a `d` argcode slot (`assembler.py:205-207`).
#[allow(dead_code)]
fn emit_descr(state: &mut AssemblyState, op: &Operand) -> u16 {
    match op {
        Operand::Descr(descr) => u16::try_from(record_descr_operand(state, descr))
            .expect("too many descrs (index > u16::MAX)"),
        _ => panic!("expected Descr operand, got {:?}", op),
    }
}

fn expect_result_or_first_reg(args: &[Operand], result: Option<&Register>, kind: Kind) -> u16 {
    match result {
        Some(reg) if reg.kind == kind => reg.index,
        Some(reg) => panic!("expected result register of kind {:?}, got {:?}", kind, reg),
        None => expect_reg(&args[0], kind),
    }
}

fn expect_result_reg(result: Option<&Register>, kind: Kind, msg: &str) -> u16 {
    match result {
        Some(reg) if reg.kind == kind => reg.index,
        Some(reg) => panic!("expected result register of kind {:?}, got {:?}", kind, reg),
        None => panic!("{}", msg),
    }
}

fn expect_reg(op: &Operand, expected: Kind) -> u16 {
    match op {
        Operand::Register(Register { kind, index }) if *kind == expected => *index,
        _ => panic!("expected Register({:?}, _), got {:?}", expected, op),
    }
}

fn expect_tlabel(op: &Operand) -> &TLabel {
    match op {
        Operand::TLabel(label) => label,
        _ => panic!("expected TLabel, got {:?}", op),
    }
}

fn expect_small_u16(op: &Operand) -> u16 {
    match op {
        Operand::ConstInt(value) => u16::try_from(*value).expect("expected u16-sized ConstInt"),
        _ => panic!("expected ConstInt(u16), got {:?}", op),
    }
}

/// `record_binop_*` / `record_unary_*` arg decoder.
///
/// The recorded op is passed as `Operand::OpCode(majit_ir::OpCode)`
/// matching `JitCodeBuilder::record_*`'s signature. RPython's
/// `codewriter/assembler.py` does not narrow this to a fixed enum; the
/// opcode is an `AbstractDescr` equivalent and the record path in the
/// metainterp consumes whatever is passed, so any valid `OpCode` must
/// round-trip through here without a hand-maintained allowlist.
fn expect_opcode(op: &Operand) -> majit_ir::OpCode {
    match op {
        Operand::OpCode(code) => *code,
        _ => panic!("expected OpCode operand, got {:?}", op),
    }
}

fn expect_list_regs(op: &Operand, expected: Kind) -> Vec<u8> {
    match op {
        Operand::ListOfKind(ListOfKind { kind, content }) if *kind == expected => content
            .iter()
            .map(|item| {
                u8::try_from(expect_reg(item, expected)).expect("register index exceeds u8")
            })
            .collect(),
        _ => panic!("expected ListOfKind({:?}), got {:?}", expected, op),
    }
}

fn expect_call_args(args: &[Operand]) -> Vec<JitCallArg> {
    args.iter().map(expect_call_arg).collect()
}

fn expect_call_arg(op: &Operand) -> JitCallArg {
    match op {
        Operand::Register(Register { kind, index }) => match kind {
            Kind::Int => JitCallArg::int(*index),
            Kind::Ref => JitCallArg::reference(*index),
            Kind::Float => JitCallArg::float(*index),
        },
        _ => panic!("expected typed call register, got {:?}", op),
    }
}

fn expect_move(args: &[Operand], result: Option<&Register>, kind: Kind) -> (u16, u16) {
    if let Some(dst) = result {
        return (dst.index, expect_reg(&args[0], kind));
    }
    (expect_reg(&args[0], kind), expect_reg(&args[1], kind))
}

/// `flatten.py:333` source operand for `{kind}_copy`. The source may
/// be either a Register (argcode `'i'`/`'r'`/`'f'`) or a Constant
/// (argcode `'c'` in short form). `MoveSource` captures the shape so
/// the dispatch arm can lower to the right `JitCodeBuilder` method.
#[derive(Debug)]
enum MoveSource<'a> {
    Reg(&'a Operand),
    ConstInt(i64),
    ConstRef(i64),
    ConstFloat(i64),
}

fn source_operand<'a>(args: &'a [Operand], result: Option<&Register>) -> MoveSource<'a> {
    let src_operand = if result.is_some() { &args[0] } else { &args[1] };
    match src_operand {
        Operand::Register(_) => MoveSource::Reg(src_operand),
        Operand::ConstInt(v) => MoveSource::ConstInt(*v),
        Operand::ConstRef(v) => MoveSource::ConstRef(*v),
        Operand::ConstFloat(v) => MoveSource::ConstFloat(*v),
        _ => panic!(
            "source_operand: unexpected source operand {:?}",
            src_operand
        ),
    }
}

fn expect_load_const_i(args: &[Operand], result: Option<&Register>) -> (u16, i64) {
    if let Some(dst) = result {
        let Operand::ConstInt(value) = &args[0] else {
            panic!("load_const_i expects ConstInt, got {:?}", args[0]);
        };
        return (dst.index, *value);
    }
    let Operand::ConstInt(value) = &args[1] else {
        panic!("load_const_i expects ConstInt, got {:?}", args[1]);
    };
    (expect_reg(&args[0], Kind::Int), *value)
}

fn expect_load_const_r(args: &[Operand], result: Option<&Register>) -> (u16, i64) {
    if let Some(dst) = result {
        let Operand::ConstRef(value) = &args[0] else {
            panic!("load_const_r expects ConstRef, got {:?}", args[0]);
        };
        return (dst.index, *value);
    }
    let Operand::ConstRef(value) = &args[1] else {
        panic!("load_const_r expects ConstRef, got {:?}", args[1]);
    };
    (expect_reg(&args[0], Kind::Ref), *value)
}

fn expect_load_const_f(args: &[Operand], result: Option<&Register>) -> (u16, i64) {
    if let Some(dst) = result {
        let Operand::ConstFloat(value) = &args[0] else {
            panic!("load_const_f expects ConstFloat, got {:?}", args[0]);
        };
        return (dst.index, *value);
    }
    let Operand::ConstFloat(value) = &args[1] else {
        panic!("load_const_f expects ConstFloat, got {:?}", args[1]);
    };
    (expect_reg(&args[0], Kind::Float), *value)
}

#[cfg(test)]
mod tests {
    use super::super::flatten::{DescrOperand, Label, SwitchDictDescr};
    use super::*;

    fn r(kind: Kind, index: u16) -> Register {
        Register::new(kind, index)
    }

    #[test]
    fn assemble_empty_ssarepr_produces_valid_jitcode() {
        let mut ssarepr = SSARepr::new("empty");
        let jitcode = assemble(&mut ssarepr, JitCodeBuilder::default(), None);
        assert_eq!(jitcode.name, "empty");
        assert!(jitcode.code.is_empty(), "empty SSARepr -> empty code");
        assert_eq!(ssarepr.insns_pos, Some(Vec::new()));
    }

    #[test]
    fn assemble_records_insn_positions_and_liveness() {
        let mut ssarepr = SSARepr::new("live");
        ssarepr
            .insns
            .push(Insn::Live(vec![Operand::Register(r(Kind::Ref, 0))]));
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));

        let mut assembler = Assembler::new();
        let before_ops = assembler.num_liveness_ops();
        let jitcode = assembler.assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        // `num_liveness_ops` and `all_liveness` live on the thread-local
        // `ASSEMBLER_STATE` (the single RPython-`Assembler` analog); other
        // tests running earlier on the same thread may already have
        // bumped the counter, so assert on the delta and the non-empty
        // buffer rather than a raw == 1.
        assert_eq!(assembler.num_liveness_ops() - before_ops, 1);
        assert!(!assembler.all_liveness().is_empty());
        assert_eq!(ssarepr.insns_pos.as_ref().map(Vec::len), Some(2));
        assert!(!jitcode.code.is_empty());
    }

    #[test]
    fn assemble_patches_jumps_through_labels() {
        let mut ssarepr = SSARepr::new("jump");
        ssarepr
            .insns
            .push(Insn::op("goto", vec![Operand::TLabel(TLabel::new("L1"))]));
        ssarepr.insns.push(Insn::Label(Label::new("L1")));
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        assert_eq!(jitcode.follow_jump(3), 3);
    }

    /// `assembler.py:197-206` parity: a `Descr` operand attached to an op
    /// that does not consume a descr in its emitted bytecode MUST NOT be
    /// registered in `jitcode.descrs`. Registration happens inline during
    /// per-op encoding — no pre-scan.
    #[test]
    fn assemble_does_not_register_unconsumed_descr() {
        let mut switch = SwitchDictDescr::new();
        switch.labels.push((4, TLabel::new("L2")));

        let mut ssarepr = SSARepr::new("switch");
        ssarepr.insns.push(Insn::Label(Label::new("L2")));
        // `abort_permanent` does not consume a descr in its bytecode (it
        // emits a single BC_ABORT_PERMANENT byte), so the attached
        // Descr operand is ignored by the assembler.
        ssarepr.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::descr(DescrOperand::SwitchDict(switch))],
        ));

        let jitcode = assemble(&mut ssarepr, JitCodeBuilder::default(), None);

        assert!(
            jitcode.descrs.is_empty(),
            "descrs attached to non-consuming ops must not be registered; \
             assembler.py:197-206 registers inline at 'd' argcode emission only, \
             got descrs={:?}",
            jitcode.descrs
        );
    }

    // TODO: once a descr-consuming op (e.g. `switch`, `getfield_gc_d`) is
    // ported into `dispatch_op`, add a positive test that
    // `record_descr_operand` is invoked during its encoding and that
    // `SwitchDictDescr._labels` → `BhDescr::Switch.dict` round-trips via
    // `fix_labels()`.

    /// `assembler.py:208-209` parity:
    /// ```python
    /// elif isinstance(x, IndirectCallTargets):
    ///     self.indirectcalltargets.update(x.lst)
    /// ```
    /// An `Operand::IndirectCallTargets(...)` attached to any op must be
    /// folded into `self.indirectcalltargets` with identity-dedup (RPython
    /// uses a Python `set` which dedupes by object identity, not value).
    fn make_test_jitcode(fnaddr: usize) -> Arc<JitCode> {
        let mut jc = JitCodeBuilder::default().finish();
        jc.fnaddr = fnaddr as i64;
        Arc::new(jc)
    }

    #[test]
    fn assemble_collects_indirectcalltargets_from_ops() {
        use super::super::flatten::IndirectCallTargets;

        let jc_a = make_test_jitcode(0x1000);
        let jc_b = make_test_jitcode(0x2000);

        let mut ssarepr = SSARepr::new("indirect");
        // `abort_permanent` is a zero-descr op in pyre's dispatch table, so
        // it tolerates the attached `IndirectCallTargets` operand without
        // needing a live dispatch arm.  The parity-relevant behaviour —
        // `write_insn` scanning the arg list and updating
        // `self.indirectcalltargets` — does not depend on the op kind.
        ssarepr.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::IndirectCallTargets(IndirectCallTargets {
                lst: vec![jc_a.clone(), jc_b.clone(), jc_a.clone()],
            })],
        ));

        let mut assembler = Assembler::new();
        assembler.assemble(&mut ssarepr, JitCodeBuilder::default(), None);

        // Identity-dedup: jc_a appears twice in the list, but the set keeps
        // a single entry (Python `set.update` on JitCode identity).
        assert_eq!(assembler.indirectcalltargets.len(), 2);
        let vec = assembler.indirectcalltargets_vec();
        assert_eq!(vec.len(), 2);
        assert!(vec.iter().any(|a| Arc::ptr_eq(a, &jc_a)));
        assert!(vec.iter().any(|a| Arc::ptr_eq(a, &jc_b)));
    }

    #[test]
    fn assemble_accumulates_indirectcalltargets_across_ops() {
        use super::super::flatten::IndirectCallTargets;

        let jc_a = make_test_jitcode(0x1000);
        let jc_b = make_test_jitcode(0x2000);
        let jc_c = make_test_jitcode(0x3000);

        let mut assembler = Assembler::new();

        // First assemble() registers jc_a and jc_b.
        let mut r1 = SSARepr::new("first");
        r1.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::IndirectCallTargets(IndirectCallTargets {
                lst: vec![jc_a.clone(), jc_b.clone()],
            })],
        ));
        assembler.assemble(&mut r1, JitCodeBuilder::default(), None);

        // Second assemble() registers jc_b (already present) and jc_c.
        let mut r2 = SSARepr::new("second");
        r2.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::IndirectCallTargets(IndirectCallTargets {
                lst: vec![jc_b.clone(), jc_c.clone()],
            })],
        ));
        assembler.assemble(&mut r2, JitCodeBuilder::default(), None);

        // `assembler.py:24` persists `indirectcalltargets` across every
        // `assemble()` call — the CodeWriter holds a single Assembler and
        // the set accumulates for the whole build (codewriter.py:19-22).
        assert_eq!(assembler.indirectcalltargets.len(), 3);
    }
}
