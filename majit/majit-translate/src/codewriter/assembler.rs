//! Assembler — converts flattened SSARepr into a `JitCode`.
//!
//! RPython equivalent: `rpython/jit/codewriter/assembler.py` class
//! `Assembler`. The `JitCode` struct itself lives in `crate::jitcode`
//! (RPython parity: `rpython/jit/codewriter/jitcode.py`).
//!
//! **Status: partial port.** `write_insn`, `fix_labels`, the
//! `all_liveness` shared table, and the `IndirectCallTargets` sidecar
//! merge for `residual_call` are implemented in the pyre-relevant
//! subset. Descriptor operands are deduplicated through the RPython
//! `_descr_dict` shape before bytecode emission.

use std::collections::HashMap;

use vecset::VecSet;

use crate::call::CallControl;
use crate::flatten::{FlatOp, IntOvfOp, Label, RegKind, SSARepr};

/// `flatten.py:30` `Register.kind[0]` — single-char prefix for opname keys.
fn kind_char_of(kind: RegKind) -> char {
    match kind {
        RegKind::Int => 'i',
        RegKind::Ref => 'r',
        RegKind::Float => 'f',
    }
}

/// Companion long form (`'int'`/`'ref'`/`'float'`) used in
/// `int_copy`/`ref_copy`/`float_copy` opname keys.
fn kind_long_name(kind: RegKind) -> &'static str {
    match kind {
        RegKind::Int => "int",
        RegKind::Ref => "ref",
        RegKind::Float => "float",
    }
}

/// `assembler.py:312` `USE_C_FORM` — the set of jitcode opnames whose
/// small ConstInt operands may take the inline `c` short-const argcode
/// (`assembler.py:163` `allow_short = (insn[0] in USE_C_FORM)`).  Listed
/// verbatim, kept identical to the runtime assembler's `use_c_form`
/// (`pyre-jit/src/jit/assembler.rs`).
///
/// In pyre's build-time flow model most operands carry no inline
/// Constant — the rtyper materialises constants through dedicated
/// `int_copy`/`ref_copy` ops and binops read registers — so only the
/// opnames that still hold an inline Constant build-time actually reach
/// the short branch: `int_copy` (`OpKind::ConstInt`/`ConstBool` and a
/// const `FlatOp::Move` source), `int_return` (a const `FlatOp::IntReturn`
/// source), `setfield_gc_i` (`OpKind::FieldWrite` with an inline int/bool
/// `LinkArg::Const`), and `jit_merge_point` (handled inline in its own
/// arm).  The remaining members are honoured by the runtime assembler,
/// which lowers array/string ops with literal small operands.
fn use_c_form(opname: &str) -> bool {
    matches!(
        opname,
        "copystrcontent"
            | "getarrayitem_gc_pure_i"
            | "getarrayitem_gc_pure_r"
            | "getarrayitem_gc_i"
            | "getarrayitem_gc_r"
            | "goto_if_not_int_eq"
            | "goto_if_not_int_ge"
            | "goto_if_not_int_gt"
            | "goto_if_not_int_le"
            | "goto_if_not_int_lt"
            | "goto_if_not_int_ne"
            | "int_add"
            | "int_and"
            | "int_copy"
            | "int_eq"
            | "int_ge"
            | "int_gt"
            | "int_le"
            | "int_lt"
            | "int_ne"
            | "int_return"
            | "int_sub"
            | "jit_merge_point"
            | "new_array"
            | "new_array_clear"
            | "newstr"
            | "setarrayitem_gc_i"
            | "setarrayitem_gc_r"
            | "setfield_gc_i"
            | "strgetitem"
            | "strsetitem"
            | "foobar"
            | "baz"
    )
}

// `reg_byte` and the `CURRENT_GRAPH` thread-local are absent because
// every FlatOp variant carries a [`crate::flatten::Register`] (or
// [`crate::flatten::RegOrConst`]) operand directly.  The assembler
// reads `r.kind` / `r.index` straight off the operand — no per-call
// kind-search, no fallback — exactly mirroring RPython's
// `Register(kind, index)` invariant from `flatten.py:28-33`.
use crate::flowspace::model::ConstValue;
use crate::jitcode::{BhCallDescr, JitCodeBody, StrConstDescriptor};
use crate::regalloc::RegAllocResult;

/// Assembler — converts SSARepr to JitCode.
///
/// RPython: `assembler.py::Assembler`.
///
/// The assembler maintains state across multiple JitCode assemblies
/// (shared descriptor table, liveness encoding, etc.)
pub struct Assembler {
    /// RPython: Assembler.insns — map {opcode_key: opcode_number}
    insns: majit_ir::VecMap<String, u8>,
    /// Next candidate for the translator-only `setdefault` fallback
    /// (`assembler.py:220`). RPython grows `self.insns` densely from
    /// zero; pyre keeps canonical / extension `BC_*` bytes reserved for
    /// build/runtime stability, so this cursor scans upward from zero
    /// and skips only those reserved bytes plus already-assigned
    /// translator-only bytes.
    dynamic_byte_cursor: u16,
    /// RPython: Assembler.descrs — list of descriptors. Inline-call
    /// descriptors keep the callee JitCode object until the final
    /// snapshot, where `jitcode.index` is guaranteed to be assigned.
    descrs: Vec<AssemblerDescr>,
    /// RPython: Assembler._descr_dict — descriptor to descrs[] index.
    /// Upstream `assembler.py:26` + `:197-203` keeps a Python dict to
    /// deduplicate AbstractDescr objects before emitting the two-byte 'd'
    /// operand; the no-HashMap house rule replaces the dict with a
    /// VecMap linear-scan lookup.
    descr_dict: majit_ir::VecMap<AssemblerDescrKey, usize>,
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
    all_liveness_positions: majit_ir::VecMap<(VecSet<u8>, VecSet<u8>, VecSet<u8>), usize>,
    /// RPython: Assembler.num_liveness_ops (assembler.py:32).
    pub num_liveness_ops: usize,
    /// State-field JIT canonical "all-live" liveness triple, set once at
    /// `__JitMeta::install_canonical_liveness` time (RPython
    /// `assembler.py:218-231 get_liveness_info` flat-state adaptation).
    /// `JitCodeBuilder::live_placeholder` defers patching of the leading
    /// `BC_LIVE` slot at the start of every per-opcode JitCode until
    /// `finalize_liveness` runs, at which point this triple is registered
    /// via `_register_liveness_offset` (the result is cached in
    /// `canonical_liveness_offset`).
    ///
    /// RPython `assembler.assemble` itself has no concept of a canonical
    /// entry — it only emits `-live-` markers as it walks the IR.  The
    /// canonical entry exists in pyre because per-opcode JitCodes need a
    /// leading `BC_LIVE` to satisfy `code[orgpc - SIZE_LIVE_OP] == op_live`
    /// at JitCode entry; lazy registration via `live_placeholder` keeps
    /// the `all_liveness` order encounter-driven (matching RPython's
    /// IR-walk order) instead of pre-seeding canonical at offset 0.
    canonical_liveness_triple: Option<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    /// Cached offset returned by the first `_register_liveness_offset`
    /// call against `canonical_liveness_triple`.
    canonical_liveness_offset: Option<usize>,
    /// Name of the graph currently being assembled, threaded through so
    /// diagnostic panics (e.g. missing regalloc coloring) can cite the
    /// exact function.  RPython tracks this via `self.jitcode.name`
    /// captured at `assembler.py:56 self.setup(ssarepr.name)`.
    current_graph_name: Option<String>,
    /// Pretty-printed FlatOp currently being encoded, only used by
    /// the `MAJIT_COVERAGE_PANIC=1` diagnostic so the missing-coloring
    /// panic can cite the offending op.
    current_flatop_debug: Option<String>,
}

impl Assembler {
    /// RPython: `Assembler.__init__()` (assembler.py:21-32).
    pub fn new() -> Self {
        Self {
            insns: majit_ir::VecMap::new(),
            dynamic_byte_cursor: 0,
            descrs: Vec::new(),
            descr_dict: majit_ir::VecMap::new(),
            indirectcalltargets: std::collections::HashSet::new(),
            list_of_addr2name: Vec::new(),
            count_jitcodes: 0,
            seen_raw_objects: std::collections::HashSet::new(),
            all_liveness: Vec::new(),
            all_liveness_length: 0,
            all_liveness_positions: majit_ir::VecMap::new(),
            num_liveness_ops: 0,
            canonical_liveness_triple: None,
            canonical_liveness_offset: None,
            current_graph_name: None,
            current_flatop_debug: None,
        }
    }

    /// Stage the state-field JIT canonical "all-live" triple for lazy
    /// registration by `ensure_canonical_liveness_offset`.  Called once
    /// per `__JitMeta::install_canonical_liveness` invocation, before any
    /// per-pc JitCode is built.
    pub fn set_canonical_liveness_triple(
        &mut self,
        live_i: Vec<u8>,
        live_r: Vec<u8>,
        live_f: Vec<u8>,
    ) {
        self.canonical_liveness_triple = Some((live_i, live_r, live_f));
    }

    /// Lazily register the canonical triple via
    /// `_register_liveness_offset` (deduplicating against
    /// `all_liveness_positions`) and cache the resulting offset.  Subsequent
    /// calls return the cached offset.  Panics if the triple has not been
    /// staged via `set_canonical_liveness_triple`.
    pub fn ensure_canonical_liveness_offset(&mut self) -> usize {
        if let Some(off) = self.canonical_liveness_offset {
            return off;
        }
        let (li, lr, lf) = self
            .canonical_liveness_triple
            .clone()
            .expect("canonical_liveness_triple not staged before ensure_canonical_liveness_offset");
        let off = self._register_liveness_offset(&li, &lr, &lf);
        self.canonical_liveness_offset = Some(off);
        off
    }

    /// RPython: `Assembler.assemble` descriptor operand path
    /// (`assembler.py:197-207`).
    ///
    /// A descriptor is inserted into `descrs` only once and every later bytecode
    /// operand reuses the same two-byte index from `_descr_dict`.
    fn emit_descr(&mut self, descr: AssemblerDescr) -> usize {
        let key = AssemblerDescrKey::from_descr(&descr);
        if let Some(index) = self.descr_dict.get(&key) {
            return *index;
        }
        let index = self.descrs.len();
        assert!(index <= 0xFFFF, "too many AbstractDescrs!");
        self.descrs.push(descr);
        self.descr_dict.insert(key, index);
        index
    }

    fn emit_ready_descr(&mut self, descr: crate::jitcode::BhDescr) -> usize {
        self.emit_descr(AssemblerDescr::Ready(descr))
    }

    fn emit_pending_jitcode_descr(&mut self, jitcode: crate::jitcode::JitCodeHandle) -> usize {
        self.emit_descr(AssemblerDescr::PendingJitCode { jitcode })
    }

    fn emit_pending_switch_descr(&mut self, cases: Vec<(i64, Label)>) -> usize {
        let index = self.descrs.len();
        assert!(index <= 0xFFFF, "too many AbstractDescrs!");
        // RPython creates a fresh SwitchDictDescr per switch site; do
        // not route through `_descr_dict`, because labels are local to
        // the currently assembled JitCode.
        self.descrs.push(AssemblerDescr::PendingSwitch { cases });
        index
    }

    /// RPython: `Assembler.assemble(ssarepr, jitcode, num_regs)`.
    ///
    /// Takes the SSARepr (flattened instruction sequence) and register
    /// allocation results, and produces a JitCode with encoded bytecode,
    /// constant pools, and register counts.
    ///
    /// RPython assembler.py:34-54.
    ///
    /// RPython codewriter.py:53-56:
    ///   ssarepr = flatten_graph(graph, regallocs)
    ///   compute_liveness(ssarepr)          ← step 3b
    ///   self.assembler.assemble(ssarepr)   ← step 4
    ///
    /// Like upstream's `Assembler.assemble`, this consumes only the
    /// `ssarepr` (plus the `regallocs` coloring) — never the graph.
    /// The kind source per operand is `Variable.concretetype`, read via
    /// the static `FunctionGraph::concretetype_of(&v)` exactly like
    /// `flatten.py:382 getcolor` reads `getkind(v.concretetype)`.
    pub fn assemble(
        &mut self,
        ssarepr: &mut SSARepr,
        regallocs: &HashMap<RegKind, RegAllocResult>,
    ) -> JitCodeBody {
        self.assemble_with_callcontrol(ssarepr, regallocs, None)
    }

    /// `assemble` overload with an attached [`CallControl`] — the
    /// production codewriter path threads the callcontrol so descriptor
    /// emission can reach the rtyper-built `CallDescriptor` cache.
    pub fn assemble_with_callcontrol(
        &mut self,
        ssarepr: &mut SSARepr,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        callcontrol: Option<&CallControl>,
    ) -> JitCodeBody {
        // RPython codewriter.py:56: compute_liveness(ssarepr)
        // Must run BEFORE assembly so -live- markers carry the full
        // set of alive registers.  The alive set uses
        // [`crate::flatten::Register`]-based identity, so liveness
        // also takes the regalloc result for the `Variable → Register`
        // bridge on `FlatOp::Op` operands.
        //
        // PyPy parity: liveness ALWAYS runs after flatten — there is
        // no graph-less path because every `ssarepr` arriving here is
        // produced by `flatten_graph(graph, …)` upstream, so the kind
        // source projection is always available.
        crate::liveness::compute_liveness(ssarepr, regallocs);
        self.current_graph_name = Some(ssarepr.name.clone());

        // Pyre-only diagnostic: under `MAJIT_COVERAGE_AUDIT=1` enumerate
        // every Variable referenced in `ssarepr.insns` that has no
        // regalloc coloring in any class.  Complements the
        // `MAJIT_COVERAGE_PANIC=1` path (which panics at the first gap
        // hit during `write_insn`) by surfacing the full per-graph gap
        // catalogue in one build.  Upstream RPython has no analogue —
        // the invariant is guaranteed by rtyper's `concretetype`
        // annotation so the lookup cannot miss.
        if std::env::var("MAJIT_COVERAGE_AUDIT").is_ok() {
            self.run_coverage_audit(ssarepr, regallocs);
        }

        let num_regs_i = regallocs.get(&RegKind::Int).map_or(0, |r| r.num_regs);
        let num_regs_r = regallocs.get(&RegKind::Ref).map_or(0, |r| r.num_regs);
        let num_regs_f = regallocs.get(&RegKind::Float).map_or(0, |r| r.num_regs);

        // RPython assembler.py:56-70: self.setup(ssarepr.name)
        let mut state = AssemblyState {
            code: Vec::new(),
            constants_i: Vec::new(),
            constants_r: Vec::new(),
            constants_f: Vec::new(),
            str_consts: Vec::new(),
            num_regs_i,
            num_regs_r,
            num_regs_f,
            label_positions: HashMap::new(),
            tlabel_fixups: Vec::new(),
            startpoints: majit_ir::vec_set::VecSet::new(),
            alllabels: majit_ir::vec_set::VecSet::new(),
            resulttypes: majit_ir::VecMap::new(),
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
        let debug_enabled = std::env::var("MAJIT_COVERAGE_PANIC").is_ok();
        for op in &ops {
            insns_pos.push(state.code.len());
            if debug_enabled {
                self.current_flatop_debug = Some(format!("{op:?}"));
            }
            self.write_insn(op, regallocs, &mut state, callcontrol);
        }
        self.current_flatop_debug = None;
        ssarepr.insns_pos = Some(insns_pos);

        // RPython assembler.py:45,250-258: self.fix_labels()
        // Upstream `target = self.label_positions[name]` raises KeyError
        // when the label is missing — never writes a silent 0 target.
        for (label, fixup_pos) in &state.tlabel_fixups {
            // RPython `assembler.py:254 target = self.label_positions[insn[1].name]`
            // — direct dict access, raises KeyError when the TLabel
            // references a label that was never defined. Mirror with
            // a fail-loud panic instead of silently writing 0.
            let target = *state
                .label_positions
                .get(label)
                .unwrap_or_else(|| panic!("undefined TLabel {label:?} at fixup {fixup_pos}"));
            let target_u16 = target as u16;
            // RPython `assembler.py:255 assert 0 <= target <= 0xFFFF`.
            assert!(target <= 0xFFFF, "label target {target} exceeds u16 range");
            // RPython `assembler.py:252-253 assert self.code[pos] == "temp 1"`
            // — the fixup must point to two reserved placeholder
            // bytes still in range.
            assert!(
                fixup_pos + 1 < state.code.len(),
                "tlabel fixup position {fixup_pos} past end of code (len={})",
                state.code.len(),
            );
            state.code[*fixup_pos] = (target_u16 & 0xFF) as u8;
            state.code[*fixup_pos + 1] = (target_u16 >> 8) as u8;
        }
        for descr in &mut self.descrs {
            let AssemblerDescr::PendingSwitch { cases } = descr else {
                continue;
            };
            let dict = cases
                .iter()
                .map(|(key, label)| {
                    // RPython `assembler.py:261 target =
                    // self.label_positions[switchlabel.name]` — KeyError
                    // for a missing switch case label. Same fail-loud
                    // policy as the TLabel fixup loop above.
                    let target = *state.label_positions.get(label).unwrap_or_else(|| {
                        panic!("undefined SwitchDictDescr label {label:?} for key {key}")
                    });
                    (*key, target)
                })
                .collect();
            *descr = AssemblerDescr::Ready(crate::jitcode::BhDescr::Switch { dict });
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
            constants_r: state.constants_r,
            constants_f: state.constants_f,
            str_consts: state.str_consts,
            c_num_regs_i: num_regs_i as u16,
            c_num_regs_r: num_regs_r as u16,
            c_num_regs_f: num_regs_f as u16,
            // self.startpoints, alllabels=self.alllabels,
            // resulttypes=self.resulttypes, ...)` — assembled jitcodes
            // always carry the recorded set, never `None`. Wrap in
            // `Some(...)` so the upstream None sentinel is reserved
            // for hand-built helper jitcodes that bypass the builder
            // (jitcode.py:24 defaults).
            startpoints: Some(state.startpoints),
            alllabels: Some(state.alllabels),
            resulttypes: Some(state.resulttypes),
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
        callcontrol: Option<&CallControl>,
    ) {
        match op {
            // RPython assembler.py:143-144: Label → record bytecode position
            FlatOp::Label(label) => {
                state.label_positions.insert(*label, state.code.len());
            }

            // RPython assembler.py:146-158 `Register('-live-', ...)`
            // case in `write_insn`: emit the `live/` opcode followed
            // by the 2-byte offset returned from `_encode_liveness`.
            FlatOp::Live { live_values } => {
                self.num_liveness_ops += 1;
                let key = state.code.len();
                state.startpoints.insert(key);
                // assembler.py:151-156 `live_i, live_r, live_f` —
                // partition live registers by kind.  Each [`Register`]
                // already carries `(kind, color)` from the flatten
                // pass, so no regalloc lookup is needed here.
                let mut live_i = Vec::new();
                let mut live_r = Vec::new();
                let mut live_f = Vec::new();
                for r in live_values {
                    match r.kind {
                        RegKind::Int => live_i.push(r.index as u8),
                        RegKind::Ref => live_r.push(r.index as u8),
                        RegKind::Float => live_f.push(r.index as u8),
                    }
                }
                // assembler.py:236 `key = (frozenset(live_i), …)` —
                // `_encode_liveness` collects the inputs into `VecSet`
                // for canonical ordering and dedup, so the caller emits
                // raw push-order without an extra sort.
                // assembler.py:148 `self.code.append(chr(self.insns['live/']))`
                let opnum = self.get_opnum("live/");
                state.code.push(opnum);
                // assembler.py:158 `self._encode_liveness(live_i, live_r, live_f)`
                // — appends the 2-byte offset into `state.code` after
                // registering or reusing the canonical entry.
                self._encode_liveness(&live_i, &live_r, &live_f, &mut state.code);
            }

            // RPython assembler.py:141-142: '---' → skip
            FlatOp::EndOfBlock => {}

            // RPython `flatten.py:292` `emitline("unreachable")` →
            // single-byte opcode for `bhimpl_unreachable`
            // (`blackhole.py:962-964`). Mirrors the
            // `assembler.py:140-159` general opcode path: a fresh
            // `startposition = len(self.code)` is recorded before the
            // opcode byte goes in so the `_check_no_branch_to_inside_an_op`
            // pass at `assembler.py:283` sees `unreachable/` as a valid
            // start address even though execution never reaches it.
            FlatOp::Unreachable => {
                let opnum = self.get_opnum("unreachable/");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
            }

            // RPython assembler.py:159-223: regular operation
            FlatOp::Op(inner_op) => {
                self.encode_op(inner_op, regallocs, state, callcontrol);
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
                let opnum = self.get_opnum("goto_if_exception_mismatch/iL");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                let encoded_llexitcase = self.emit_llexitcase(llexitcase, state);
                state.code.push(encoded_llexitcase);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            // RPython flatten.py:247-267: goto_if_not(cond, TLabel(false_path))
            // Only goto_if_not exists — no goto_if_true in RPython.
            FlatOp::GotoIfNot { cond, target } => {
                // RPython parity: `cond.kind == RegKind::Int` because
                // `block.exitswitch.concretetype == lltype.Bool` is the
                // build-time gate at `flatten.py:248`.  Pyre routes
                // every Variable-cond exitswitch installation through
                // `FunctionGraph::set_branch` (model.rs), which appends
                // a `bool` UnaryOp (flowcontext.py:756
                // `Variable.bool().eval(self)`) so the rtyper lowers
                // the cond to a Bool register before flatten emits.
                // Fail loud if a non-Int slips through — mirrors
                // `FlatOp::Switch`'s assert below.
                assert_eq!(
                    cond.kind,
                    RegKind::Int,
                    "FlatOp::GotoIfNot.cond must be RegKind::Int \
                     (flatten.py:248 `block.exitswitch.concretetype == lltype.Bool`); \
                     got {:?}",
                    cond.kind,
                );
                let opnum = self.get_opnum("goto_if_not/iL");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(cond.index as u8);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            // RPython `flatten.py:247` fused guard: the jitcode key is
            // `goto_if_not_<op>/<argcodes>`, where the argcodes are the
            // per-operand register kinds (`i`/`r`/`f`) followed by the
            // `L` label code (e.g. `goto_if_not_int_lt/iiL`,
            // `goto_if_not_int_is_zero/iL`).  A Bool-producing compare's
            // operands carry the matching concretetype, so the kinds
            // resolve to the keys registered in `insns.rs:780-796`.
            FlatOp::GotoIfNotOp {
                opname,
                args,
                target,
            } => {
                let mut argcodes = String::with_capacity(args.len() + 1);
                for arg in args {
                    argcodes.push(match arg.kind {
                        RegKind::Int => 'i',
                        RegKind::Ref => 'r',
                        RegKind::Float => 'f',
                    });
                }
                argcodes.push('L');
                let opnum = self.get_opnum(&format!("goto_if_not_{opname}/{argcodes}"));
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                for arg in args {
                    state.code.push(arg.index as u8);
                }
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
            }

            FlatOp::Switch { value, targets } => {
                // `flatten.py:275-276` — `kind = getkind(block.exitswitch.
                // concretetype); assert kind == 'int'`.  The production
                // path goes through `flatten.rs::insert_exits` (which
                // already asserts `kind == 'i'` before constructing
                // `FlatOp::Switch`), so a non-Int `value.kind` here can
                // only mean a test fixture built `FlatOp::Switch`
                // directly with the wrong kind.  Fail loud — the
                // `switch/id` opcode reads the int register file, so a
                // Ref / Float index byte would silently address the
                // wrong slot at runtime.
                assert_eq!(
                    value.kind,
                    RegKind::Int,
                    "FlatOp::Switch.value must be RegKind::Int \
                     (flatten.py:275-276 `assert kind == 'int'`); got {:?}",
                    value.kind,
                );
                let descr_idx = self.emit_pending_switch_descr(targets.clone());
                let opnum = self.get_opnum("switch/id");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(value.index as u8);
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
            }

            FlatOp::IntBinOpJumpIfOvf {
                op,
                target,
                lhs,
                rhs,
                dst,
            } => {
                // `flatten.py:195-204` only synthesises
                // `int_*_jump_if_ovf` for `add_ovf` / `sub_ovf` /
                // `mul_ovf` opnames whose all three operands are
                // already `Int` by lltype construction.  Origin/main
                // matched this with a `debug_assert_eq!(kind, 'i')`
                // for lhs/rhs/dst — keep the fail-fast guard on the
                // Register payload so test fixtures that hand-build a
                // miskinded `FlatOp::IntBinOpJumpIfOvf` surface here
                // instead of garbling the bytecode at decode time.
                debug_assert_eq!(
                    lhs.kind,
                    RegKind::Int,
                    "IntBinOpJumpIfOvf.lhs must be RegKind::Int; got {:?}",
                    lhs.kind,
                );
                debug_assert_eq!(
                    rhs.kind,
                    RegKind::Int,
                    "IntBinOpJumpIfOvf.rhs must be RegKind::Int; got {:?}",
                    rhs.kind,
                );
                debug_assert_eq!(
                    dst.kind,
                    RegKind::Int,
                    "IntBinOpJumpIfOvf.dst must be RegKind::Int; got {:?}",
                    dst.kind,
                );
                let opname = match op {
                    IntOvfOp::Add => "int_add_jump_if_ovf/Lii>i",
                    IntOvfOp::Sub => "int_sub_jump_if_ovf/Lii>i",
                    IntOvfOp::Mul => "int_mul_jump_if_ovf/Lii>i",
                };
                let opnum = self.get_opnum(opname);
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.alllabels.insert(state.code.len());
                state.tlabel_fixups.push((*target, state.code.len()));
                state.code.push(0);
                state.code.push(0);
                state.code.push(lhs.index as u8);
                state.code.push(rhs.index as u8);
                state.code.push(dst.index as u8);
                state.resulttypes.insert(state.code.len(), 'i');
            }

            // RPython flatten.py:333 `self.emitline('%s_copy' % kind,
            // v, "->", w)` — argcodes `i>i` (typed src, result marker,
            // typed dst). The `>` bears no byte in the stream; it only
            // flags the result position in the key so the blackhole
            // wire `int_copy/i>i` (blackhole.rs:5670) finds the handler
            // and `Assembler.resulttypes[pc]` is populated correctly.
            //
            // Upstream's source operand (`v`) can be either a `Register`
            // or a `Constant` (`getcolor` returns the Constant as-is at
            // flatten.py:382-384); in both cases the `assembler.py:164-174`
            // single-byte encoder shares the argcode kind letter and
            // disambiguates register vs constant at decode time via
            // `byte >= count_regs[kind]`.
            FlatOp::Move { dst, src } => {
                // `assembler.py:188-196` — every emitted instruction
                // (including `*_copy`) records its byte offset in
                // `startpoints`, so the label-fixup pass can land a
                // `Label(block)` on a block-opening copy without
                // misdescribing the bytecode boundary.  Previously
                // missed here because `FlatOp::Move` bypasses
                // `encode_op` and the per-arm emitters had each gotten
                // their own `startpoints.insert` call except this one.
                state.startpoints.insert(state.code.len());
                let kind_char = kind_char_of(dst.kind);
                let kind_name = kind_long_name(dst.kind);
                // `assembler.py:163` `allow_short = ('{kind}_copy' in
                // USE_C_FORM)` — only `int_copy` qualifies, so an int Move
                // with a small const source emits `int_copy/c>i`.
                let allow_short = use_c_form(&format!("{kind_name}_copy"));
                let (src_reg, src_argcode) =
                    self.encode_regorconst_source(src, dst.kind, allow_short, state, callcontrol);
                let key = format!("{kind_name}_copy/{src_argcode}>{kind_char}");
                let opnum = self.get_opnum(&key);
                state.code.push(opnum);
                state.code.push(src_reg);
                state.code.push(dst.index as u8);
                state.resulttypes.insert(state.code.len(), kind_char);
            }

            // RPython `flatten.py:329` `self.emitline('%s_push' % kind, v)`.
            FlatOp::Push(src) => {
                let kind_char = kind_char_of(src.kind);
                let kind_name = kind_long_name(src.kind);
                let key = format!("{kind_name}_push/{kind_char}");
                let opnum = self.get_opnum(&key);
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(src.index as u8);
            }

            // RPython `flatten.py:331` `self.emitline('%s_pop' % kind, "->", w)`.
            FlatOp::Pop(dst) => {
                let kind_char = kind_char_of(dst.kind);
                let kind_name = kind_long_name(dst.kind);
                let key = format!("{kind_name}_pop/>{kind_char}");
                let opnum = self.get_opnum(&key);
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(dst.index as u8);
                state.resulttypes.insert(state.code.len(), kind_char);
            }

            FlatOp::LastException { dst } => {
                // Parity expectation: `dst.kind == RegKind::Int`
                // (the exception class identity).  See GotoIfNot
                // notes above for the upstream-gap caveat.
                let opnum = self.get_opnum("last_exception/>i");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(dst.index as u8);
                state.resulttypes.insert(state.code.len(), 'i');
            }

            FlatOp::LastExcValue { dst } => {
                // Parity expectation: `dst.kind == RegKind::Ref`
                // (the exception instance pointer).
                let opnum = self.get_opnum("last_exc_value/>r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(dst.index as u8);
                state.resulttypes.insert(state.code.len(), 'r');
            }

            FlatOp::Reraise => {
                let opnum = self.get_opnum("reraise/");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
            }

            // RPython `flatten.py:131-138` `make_return`.  Blackhole
            // handlers: `blackhole.py:841-863 bhimpl_{int,ref,float,void}_return`.
            // `emit_const_*` returns a byte ≥ `num_regs_{kind}` so the
            // single-byte argcode `i`/`r`/`f` suffices for both register
            // and constant sources (upstream `assembler.py:164-174`).
            FlatOp::IntReturn(v) => {
                // `int_return` is in USE_C_FORM (`assembler.py:312`), so a
                // small const-int source emits `int_return/c`.
                let (reg, src_argcode) = self.encode_regorconst_source(
                    v,
                    RegKind::Int,
                    use_c_form("int_return"),
                    state,
                    callcontrol,
                );
                let opnum = self.get_opnum(&format!("int_return/{src_argcode}"));
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
            FlatOp::RefReturn(v) => {
                // `ref_return` is not in USE_C_FORM; ref always pools to `r`.
                let (reg, _) =
                    self.encode_regorconst_source(v, RegKind::Ref, false, state, callcontrol);
                let opnum = self.get_opnum("ref_return/r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
            FlatOp::FloatReturn(v) => {
                // `float_return` is not in USE_C_FORM; float always pools to `f`.
                let (reg, _) =
                    self.encode_regorconst_source(v, RegKind::Float, false, state, callcontrol);
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
            // RPython `flatten.py:139-143` `make_return` 2-inputarg case
            // plus the `flatten.py:166-173` overflow reraise.  Both paths
            // funnel through `raise/r` — `RegOrConst::Reg` is the raised
            // exception value's Register, `RegOrConst::Const` is the
            // standard OverflowError instance.  Blackhole:
            // `blackhole.py:1000 bhimpl_raise(excvalue)`.
            FlatOp::Raise(v) => {
                // `raise` is not in USE_C_FORM; the excvalue is a Ref → `r`.
                let (reg, _) =
                    self.encode_regorconst_source(v, RegKind::Ref, false, state, callcontrol);
                let opnum = self.get_opnum("raise/r");
                state.startpoints.insert(state.code.len());
                state.code.push(opnum);
                state.code.push(reg);
            }
        }
    }

    /// RPython `assembler.py:234-248` `_encode_liveness(live_i, live_r,
    /// live_f)` — register a `(live_i, live_r, live_f)` triple in the
    /// shared `all_liveness` table (deduplicating against
    /// `all_liveness_positions`) and append the 2-byte offset of the
    /// canonical entry into `code`.
    ///
    /// Mirrors RPython `assembler.py:235`
    /// `key = (frozenset(live_i), frozenset(live_r), frozenset(live_f))`:
    /// the cache key is set-valued, so callers may pass arbitrary-order
    /// or duplicated slices.  Each kind's effective payload is the
    /// sorted, deduplicated set, exactly as `liveness.py:148` `live =
    /// sorted(live)` produces during inner encoding.
    ///
    /// On a cache miss we append three header bytes
    /// (`len(live_i)`, `len(live_r)`, `len(live_f)`) followed by
    /// `encode_liveness` of each kind, exactly mirroring upstream
    /// `assembler.py:241-247` byte order.  The returned offset is
    /// finally written via `liveness::encode_offset` (parity with
    /// `liveness.py:127-131`).
    pub fn _encode_liveness(
        &mut self,
        live_i: &[u8],
        live_r: &[u8],
        live_f: &[u8],
        code: &mut Vec<u8>,
    ) {
        let pos = self._register_liveness_offset(live_i, live_r, live_f);
        // assembler.py:248 `encode_offset(pos, self.code)`.
        crate::codewriter::liveness::encode_offset(pos, code);
    }

    /// Registration-only sibling of [`_encode_liveness`]: deduplicate the
    /// `(live_i, live_r, live_f)` triple into the shared `all_liveness`
    /// table and return the entry's offset, without writing the 2-byte
    /// `encode_offset` bytes anywhere.
    ///
    /// The `live/<offset>` 2-byte slot in a JitCode is patched by the
    /// caller via `JitCodeBuilder::patch_live_offset` once the offset is
    /// known.  Used by the deferred-patch path in
    /// `JitCodeBuilder::finalize_liveness` where
    /// the lowerer collects per-marker triples first, then registers and
    /// patches them in a single post-emission pass.
    pub fn _register_liveness_offset(
        &mut self,
        live_i: &[u8],
        live_r: &[u8],
        live_f: &[u8],
    ) -> usize {
        // frozenset(live_f))`.  `VecSet` is a Vec-backed sorted set, so
        // collecting the input into one yields the same canonical form
        // `frozenset` would have produced.
        let key = (
            live_i.iter().copied().collect::<VecSet<u8>>(),
            live_r.iter().copied().collect::<VecSet<u8>>(),
            live_f.iter().copied().collect::<VecSet<u8>>(),
        );
        if let Some(&cached) = self.all_liveness_positions.get(&key) {
            return cached;
        }
        let pos = self.all_liveness.len();
        // assembler.py:241 `chr(len(live_i)) + chr(len(live_r)) + chr(len(live_f))`.
        // RPython `chr(N)` raises `ValueError` for N >= 256; Rust `as u8`
        // silently wraps. Strict assert mirrors the RPython failure mode
        // (`assembler.py:265` constants+regs <= 256 bound) so a regression
        // that emits a 256+-element bank surfaces here instead of being
        // mis-encoded into a low byte the decoder later misreads.
        let len_i = key.0.len();
        let len_r = key.1.len();
        let len_f = key.2.len();
        assert!(
            len_i < 256,
            "live_i length {len_i} exceeds u8; assembler.py:241 chr() would ValueError"
        );
        assert!(
            len_r < 256,
            "live_r length {len_r} exceeds u8; assembler.py:241 chr() would ValueError"
        );
        assert!(
            len_f < 256,
            "live_f length {len_f} exceeds u8; assembler.py:241 chr() would ValueError"
        );
        self.all_liveness.push(len_i as u8);
        self.all_liveness.push(len_r as u8);
        self.all_liveness.push(len_f as u8);
        // assembler.py:243-247 `for live in live_i, live_r, live_f:
        // liveness = encode_liveness(live); …`
        for live in [key.0.as_slice(), key.1.as_slice(), key.2.as_slice()] {
            let encoded = crate::codewriter::liveness::encode_liveness(live);
            self.all_liveness.extend_from_slice(&encoded);
        }
        self.all_liveness_length = self.all_liveness.len();
        self.all_liveness_positions.insert(key, pos);
        pos
    }

    /// Encode a [`RegOrConst`] operand into the byte stream and return
    /// its `(byte, argcode)`.
    ///
    /// `RegOrConst::Reg` carries `(kind, color)` directly — no regalloc
    /// lookup needed.  Constants emit through `emit_const` with
    /// `expected_kind` (variant-fixed) selecting the constant pool,
    /// mirroring `assembler.py:164-174` where the single-byte argcode kind
    /// letter chooses between register and constant via
    /// `byte >= count_regs[kind]`.  `allow_short` is `(opname in
    /// USE_C_FORM)` (`assembler.py:163`); when set, a small integer-kind
    /// Constant takes the inline `'c'` short form (`assembler.py:99-107`)
    /// instead of a pooled `'i'` byte.  Register sources and ref/float
    /// Constants always take their kind argcode.
    fn encode_regorconst_source(
        &mut self,
        arg: &crate::flatten::RegOrConst,
        expected_kind: RegKind,
        allow_short: bool,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> (u8, char) {
        match arg {
            // RPython `assembler.py:164-174`: the single-byte argcode
            // is keyed on `Register.kind`, so the source operand's
            // kind MUST match the expected kind for the surrounding
            // op (e.g. `int_copy/i>i`'s source must be Int).  PyPy
            // satisfies this by construction via
            // `flatten.py:333` (the source `Register` was created by
            // `getcolor(v)` against the matching `regallocs[w.kind]`
            // entry); pyre mirrors that with a strict assert so an
            // upstream kind-provenance gap surfaces here rather than
            // emitting a register byte that misrepresents its bank.
            crate::flatten::RegOrConst::Reg(r) => {
                assert_eq!(
                    r.kind, expected_kind,
                    "encode_regorconst_source: Register kind {:?} does not match \
                     variant-expected kind {expected_kind:?} (PyPy \
                     `assembler.py:164-174` requires the single-byte argcode kind \
                     and the operand kind to coincide) — graph {:?}, flatop {:?} \
                     (set MAJIT_COVERAGE_PANIC=1 to populate the flatop context)",
                    r.kind, self.current_graph_name, self.current_flatop_debug,
                );
                (r.index as u8, kind_char_of(expected_kind))
            }
            crate::flatten::RegOrConst::Const(c) => {
                // RPython `assembler.py:168` reads `getkind(x.concretetype)`
                // for the Constant operand.  When the Constant carries
                // a `concretetype` it MUST agree with the surrounding op's
                // `expected_kind` (the byte-stream argcode is keyed on
                // that kind, same constraint as the Register branch
                // above).  When concretetype is absent fall back to the
                // op's expected kind — that mirrors upstream's behavior
                // for synthesized constants whose kind only the caller
                // knows.
                let const_kind = crate::flatten::constant_kind(c);
                let kind_char = kind_char_of(expected_kind);
                if c.concretetype.is_some() {
                    assert_eq!(
                        const_kind, kind_char,
                        "encode_regorconst_source: Constant.concretetype kind {const_kind:?} \
                         does not match variant-expected kind {kind_char:?} (PyPy \
                         `assembler.py:168` requires `getkind(x.concretetype)` to coincide \
                         with the surrounding op's kind)",
                    );
                }
                // Only the int bank has a short-const form (argcode `c`);
                // ref/float Constants always take their pooled `r`/`f` byte.
                if kind_char == 'i' {
                    self.emit_const_i_from_const_allow_short(
                        &c.value,
                        allow_short,
                        state,
                        callcontrol,
                    )
                } else {
                    (
                        self.emit_const(&c.value, kind_char, state, callcontrol),
                        kind_char,
                    )
                }
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
        callcontrol: Option<&CallControl>,
    ) {
        use crate::model::OpKind;

        let startposition = state.code.len();
        state.code.push(0); // placeholder for opcode byte
        state.startpoints.insert(startposition);

        let mut argcodes = String::new();

        match &op.kind {
            // RPython flatten.py keeps inputargs on Block.inputargs and does
            // not serialize them as bytecode operations.
            OpKind::Input { .. } => {
                panic!("OpKind::Input must be eliminated before assembly");
            }
            // RPython: inline_call → [jitcode, I[...], R[...], F[...]]
            OpKind::InlineCall {
                jitcode,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            } => {
                // RPython assembler.py:197-207: jitcode → descrs[index]
                // The JitCode object IS the descriptor for inline_call.
                let descr_idx = self.emit_pending_jitcode_descr(jitcode.clone());
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython jtransform.py:422-431: rewrite_call
                // Only emit the kind sublists that are in 'kinds'.
                let kinds = self.kinds_suffix(args_i, args_r, args_f, *result_kind);
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
                // Result — see residual_call note below: derive the
                // key-level `reskind` from regalloc so `_r_i` / `_r_r`
                // match the actual `>X` argcode suffix.
                let result_key_kind = self.emit_call_result_arg(
                    op.result.as_ref(),
                    *result_kind,
                    regallocs,
                    state,
                    &mut argcodes,
                );
                // RPython jtransform.py:434: inline_call_{kinds}_{reskind}
                let key = format!("inline_call_{kinds}_{result_key_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython: recursive_call → [jd_index, G_I, G_R, G_F, R_I, R_R, R_F]
            //
            // `bhimpl_recursive_call_{i,r,f,v}` declares jd_index as
            // `@arguments("self", "i", ...)` (blackhole.py:1101-1132) so
            // the canonical argcode is `i` (register read). `emit_const_i`
            // returns a register-index into the int constant pool; the
            // dispatch side `bh.registers_i[code[p]]` reads the jd_index
            // back out. RPython does not include `recursive_call` in
            // `USE_C_FORM` (assembler.py:312), so the `c` short-const
            // form is not permitted here.
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
                let idx = self.emit_const_i(*jd_index as i64, state);
                state.code.push(idx);
                argcodes.push('i');
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
                let result_key_kind = self.emit_call_result_arg(
                    op.result.as_ref(),
                    *result_kind,
                    regallocs,
                    state,
                    &mut argcodes,
                );
                let key = format!("recursive_call_{result_key_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython: residual_call/call_may_force/call_elidable
            // → [funcptr, calldescr, I[...], R[...], F[...]]
            // RPython jtransform.py:414-435: rewrite_call splits args
            // by kind via make_three_lists.
            OpKind::CallResidual {
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            }
            | OpKind::CallMayForce {
                funcptr,
                descriptor,
                args_i,
                args_r,
                args_f,
                result_kind,
                ..
            }
            | OpKind::CallElidable {
                funcptr,
                descriptor,
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
                    crate::model::CallFuncPtr::Value(var) => {
                        let (reg, kc) = self.lookup_reg_with_kind_var(var, regallocs);
                        state.code.push(reg);
                        argcodes.push(kc);
                    }
                    crate::model::CallFuncPtr::Target(target) => {
                        panic!("call op reached assembler without materialized funcptr: {target}");
                    }
                }
                // RPython `assembler.py:197-203`: resolve the descriptor
                // through `_descr_dict` before writing the two bytes. The
                // bytes are still written AFTER the I/R/F lists to match
                // `jtransform.py:422-431` ordering: `iIRFd` / `iIRd` / `iRd`.
                let calldescr = descriptor.to_bh_calldescr();
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::Call { calldescr });
                // RPython jtransform.py:422-431: kind-separated sublists
                let kinds = self.kinds_suffix(args_i, args_r, args_f, *result_kind);
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
                // RPython `residual_call_r_r` / `residual_call_r_i` /
                // `residual_call_r_v` are *different* bhimpls
                // (`blackhole.py:1225-1231`): the `_r` / `_i` / `_v`
                // suffix encodes the actual result kind. When pyre's
                // rtyper (`translator::rtyper::legacy_resolve::resolve_types`)
                // upgrades a call result's concrete type to `Signed`
                // (e.g. via `is_int_arith` backward constraint), the
                // regalloc-assigned register class diverges from the
                // op struct's original `result_kind`. Derive the key
                // name suffix from the regalloc-determined class so
                // `base_{kinds}_{reskind}` stays consistent with the
                // argcode `>X` suffix. If no result, fall back to `v`.
                let result_key_kind = self.emit_call_result_arg(
                    op.result.as_ref(),
                    *result_kind,
                    regallocs,
                    state,
                    &mut argcodes,
                );
                // RPython jtransform.py:434: {base}_{kinds}_{reskind}
                let key = format!("{base}_{kinds}_{result_key_kind}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython `rpython/jit/codewriter/assembler.py:164-174`: ConstInt
            // is NOT a separate op — Constants appear as arguments to other
            // instructions via `emit_const`. Pyre's model forces constants
            // through a standalone materialization op since operands are
            // always `Variable`; lowering that limitation is deferred
            // (requires op-level constant operands). Until then, materialise
            // through `int_copy` and reuse the canonical `bhimpl_int_copy`
            // handler: `int_copy` is in USE_C_FORM (`assembler.py:312`), so a
            // small value (`-128..=127`) takes the inline `c` byte
            // (`int_copy/c>i`) and a larger one a pool-region `i` slot
            // (`int_copy/i>i`).
            OpKind::ConstInt(val) => {
                // assembler.py:163 `allow_short = ('int_copy' in USE_C_FORM)` →
                // a small constant takes the inline `c` byte (`int_copy/c>i`).
                let (idx, src_argcode) =
                    self.emit_const_i_allow_short(*val, use_c_form("int_copy"), state);
                state.code.push(idx);
                argcodes.push(src_argcode);
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("int_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // RPython folds `lltype.Bool` to kind `'int'` at codewriter
            // (`flatten.py:getkind`), so `ConstBool` materialises through
            // the same `int_copy/i>i` path as `ConstInt`. The bool value
            // collapses to 0/1 in the int constant pool and the
            // canonical `bhimpl_int_copy` handler runs in the blackhole.
            OpKind::ConstBool(val) => {
                // Bool folds to kind `int`; `int_copy` is in USE_C_FORM, so the
                // 0/1 value always fits the inline `c` byte (`int_copy/c>i`).
                let (idx, src_argcode) =
                    self.emit_const_i_allow_short(*val as i64, use_c_form("int_copy"), state);
                state.code.push(idx);
                argcodes.push(src_argcode);
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("int_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Ref-constant materialization mirrors `ConstInt`: the
            // `HostObject`'s `identity_id()` goes through
            // `emit_const_r` (the shared `'r'` constant pool) and the
            // pool-region register index is moved into the SSA
            // destination via `ref_copy`.  Emitted only for
            // unit-variant `SyntheticTransparentCtor` results that the
            // pre-jtransform fold (`translator/rtyper/unit_variant_fold.rs`)
            // rewrote from `OpKind::Call { args: [] }` (PyPy
            // `rtyper/rpbc.py::SingleFrozenPBCRepr` parity).
            OpKind::ConstRef(obj) => {
                let const_value = crate::flowspace::model::ConstValue::HostObject(obj.clone());
                let idx = self.emit_const_r(&const_value, state);
                state.code.push(idx);
                argcodes.push('r');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("ref_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ConstRefNull => {
                let const_value = crate::flowspace::model::ConstValue::LLAddress(
                    crate::translator::rtyper::lltypesystem::lltype::_address::Null,
                );
                let idx = self.emit_const_r(&const_value, state);
                state.code.push(idx);
                argcodes.push('r');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("ref_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ConstRefAddr(addr) => {
                let idx = self.emit_const_r_bits(*addr, state);
                state.code.push(idx);
                argcodes.push('r');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("ref_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Float-constant materialization mirrors `ConstInt`: the
            // bit pattern goes through `emit_const_f` (the same pool
            // every `'f'` constant uses) and the resulting pool-region
            // register index is moved into the SSA destination via
            // `float_copy`.
            OpKind::ConstFloat(bits) => {
                let const_value = crate::flowspace::model::ConstValue::Float(*bits);
                let idx = self.emit_const_f(&const_value, state);
                state.code.push(idx);
                argcodes.push('f');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let key = format!("float_copy/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Field/array operations: encode registers + descriptor.
            // RPython assembler.py:197-207: AbstractDescr → 2-byte index.
            // Field operations: register + descriptor.
            // RPython assembler.py:197-207: AbstractDescr → 2-byte index.
            OpKind::FieldRead {
                base,
                field,
                ty,
                pure,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx = self.emit_ready_descr(fielddescrof(field, ty, callcontrol));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_getfield_gc_{i,r,f}` canonical keys key
                // off the RESULT register's kind (`@arguments("cpu", "r",
                // "d", returns="X")`), not the declared field type —
                // declared field `ty` can be pyre-only Void/State/Unknown
                // while the SSA result register is always i/r/f after
                // regalloc. Using the result kind keeps the opname
                // aligned with the `>X` argcode the runtime dispatches on.
                let result_kind = if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                    kc
                } else {
                    'v'
                };
                let mut opname = format!("getfield_gc_{result_kind}");
                if *pure {
                    opname.push_str("_pure");
                }
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // Boxing GC allocation (`fuse_boxing_alloc`).  Mirrors the runtime
            // tracer oracle (`box_trace.rs trace_box_float`): a `new_with_vtable`
            // carrying ONLY a size descriptor and a fresh ref-kind result, no
            // register operands.  `bh_size_spec_from_callcontrol` resolves the
            // struct size / gc type-id from `owner` via the runtime `gc_cache`
            // Arc (`path_hash(owner)` keys `_cache_size`).  The `vtable` (type
            // pointer) is NOT in that Arc — `_cache_size` carries struct size
            // only — so it travels on the op (captured by `fuse_boxing_alloc`
            // from the dropped `ob_header.ob_type` store) and is what the
            // runtime stamps into the new object's `ob_type` / `w_class`
            // (`state.rs materialize_virtual_object`, `runner.rs
            // bh_new_with_vtable`: both write the type word only when vtable
            // != 0).  A zero vtable would silently leave `ob_type` null, so it
            // fails loud here.
            OpKind::NewWithVtable { owner, vtable } => {
                let spec = bh_size_spec_from_callcontrol(
                    callcontrol.expect("new_with_vtable assembly requires a CallControl"),
                    owner,
                )
                .unwrap_or_else(|| {
                    panic!("new_with_vtable: no struct layout registered for owner {owner:?}")
                });
                if *vtable == 0 {
                    panic!(
                        "new_with_vtable: boxing owner {owner:?} has no resolved type pointer \
                         (fuse_boxing_alloc could not capture the ob_type address); refusing to \
                         emit a null-ob_type allocation"
                    );
                }
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::Size {
                    size: spec.size,
                    type_id: spec.type_id,
                    vtable: *vtable as usize,
                    // `STRUCT._name` identity is left empty for the transient
                    // `bh_new_with_vtable` size descr; the gc_cache hit keys on
                    // `type_id` (`path_hash(owner)`), not this field.
                    owner: String::new(),
                    all_fielddescrs: spec.all_fielddescrs,
                    // Round-trip the GC-header flag off the resolved struct
                    // layout: a `new_with_vtable` boxes a header-carrying
                    // object, so its size descr must keep `GUARD_GC_TYPE`
                    // gating (a header-less raw struct would clear it).
                    is_gc_managed: spec.is_gc_managed,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // The allocation result is ALWAYS a fresh GC ref ('r').  #314:
                // assert the regalloc result kind structurally (mirror the
                // getarrayitem invariant), so an Int-banked result fails loud at
                // emit time instead of silently miscompiling.
                let result = op
                    .result
                    .as_ref()
                    .expect("new_with_vtable must produce a result register");
                argcodes.push('>');
                let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                assert_eq!(
                    kc, 'r',
                    "new_with_vtable result must be ref-kind ('r'), got '{kc}' for owner {owner:?}"
                );
                argcodes.push(kc);
                state.code.push(reg);
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
            // is not yet implemented.
            OpKind::VtableMethodPtr {
                receiver,
                trait_root,
                method_name,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(receiver, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VtableMethod {
                    trait_root: trait_root.clone(),
                    method_name: method_name.clone(),
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
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
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                for fd in [field, mutate_field] {
                    let descr_idx = self.emit_ready_descr(fielddescrof(
                        fd,
                        &crate::model::ValueType::Unknown,
                        callcontrol,
                    ));
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
                base,
                value,
                field,
                ty,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // RPython `bhimpl_setfield_gc_{i,r,f}` canonical keys key
                // off the VALUE's kind (`@arguments("cpu", "r", "X",
                // "d")`), not the declared field type — declared field
                // `ty` can be pyre-only Void/State/Unknown.  jtransform's
                // rewrite_op_setfield derives the value kind from
                // `getkind(v_newvalue.concretetype)`; mirror that via
                // `constant_kind` (concretetype, falling back to the Value
                // variant).  A Variable value is a register of its SSA
                // kind.  An int Constant takes the short `c` byte
                // (`setfield_gc_i/rcd`) when small or a pool `i` slot
                // (`/rid`) when wide (`assembler.py:99-107`, `setfield_gc`
                // ∈ USE_C_FORM at `assembler.py:312`); a ref/float Constant
                // takes its pooled `r`/`f` byte (`assembler.py:168`).
                let value_kind = match value {
                    crate::model::LinkArg::Value(var) => {
                        let (reg, kc) = self.lookup_reg_with_kind_var(var, regallocs);
                        state.code.push(reg);
                        argcodes.push(kc);
                        kc
                    }
                    crate::model::LinkArg::Const(c) => {
                        let kind = crate::flatten::constant_kind(c);
                        if kind == 'i' {
                            let (byte, argcode) = self.emit_const_i_from_const_allow_short(
                                &c.value,
                                use_c_form("setfield_gc_i"),
                                state,
                                callcontrol,
                            );
                            state.code.push(byte);
                            argcodes.push(argcode);
                        } else {
                            let byte = self.emit_const(&c.value, kind, state, callcontrol);
                            state.code.push(byte);
                            argcodes.push(kind);
                        }
                        kind
                    }
                };
                let descr_idx = self.emit_ready_descr(fielddescrof(field, ty, callcontrol));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                let opname = format!("setfield_gc_{value_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::ArrayRead {
                base,
                index,
                item_ty,
                array_type_id,
                nolength,
                pure,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(index, regallocs);
                // Array indices are typed Signed, so the index register is
                // always the int bank.  `bhimpl_getarrayitem_gc_*` only
                // dispatches the `i`-index argcode; a ref/float index is a
                // kind-flow bug upstream of the codewriter — fail here rather
                // than emit an unconsumable ref-index (`rrd`/`ird`) shape.
                assert_eq!(
                    kc, 'i',
                    "getarrayitem_gc array index must be int-kind, got {kc:?}",
                );
                state.code.push(reg);
                argcodes.push(kc);
                // descr.py:359-362 + ARRAY_INSIDE._hints.get('nolength',
                // False): the producer (model::OpKind::ArrayRead) carries
                // the layout bit. `nolength=true` → no length header
                // (`lendescr=None`); `nolength=false` → length word at
                // offset 0 (`lendescr=Some(0)`).
                let len_offset = if *nolength { None } else { Some(0) };
                let descr_idx = self.emit_ready_descr(arraydescrof(
                    item_ty,
                    array_type_id,
                    len_offset,
                    callcontrol,
                ));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_getarrayitem_gc_{i,r,f}` keys off the
                // result register's kind — same rationale as getfield_gc_*.
                let result_kind = if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                    kc
                } else {
                    'v'
                };
                // `getarrayitem_gc_{i,r,f}_pure` for a foldable/immutable
                // element load (`ll_getitem_foldable_nonneg`, rlist.py:724);
                // `blackhole.py:1339-1341` aliases `bhimpl_getarrayitem_gc_*_pure
                // = bhimpl_getarrayitem_gc_*`, so the `rid>X` argcodes are
                // identical to the non-pure form — only the opname differs.
                let opname = if *pure {
                    format!("getarrayitem_gc_{result_kind}_pure")
                } else {
                    format!("getarrayitem_gc_{result_kind}")
                };
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
                nolength,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(index, regallocs);
                // Array indices are typed Signed, so the index register is
                // always the int bank.  `bhimpl_setarrayitem_gc_*` only
                // dispatches the `i`-index argcode; a ref/float index is a
                // kind-flow bug upstream of the codewriter — fail here rather
                // than emit an unconsumable ref-index (`rrrd`) shape.
                assert_eq!(
                    kc, 'i',
                    "setarrayitem_gc array index must be int-kind, got {kc:?}",
                );
                state.code.push(reg);
                argcodes.push(kc);
                // RPython `bhimpl_setarrayitem_gc_{i,r,f}` keys off the
                // VALUE's kind; jtransform.py:803 derives it from
                // `getkind(op.args[2].concretetype)` and passes the operand
                // (Variable or Constant) verbatim.  Mirror the FieldWrite
                // c-form: a Variable is a register; an int Constant takes
                // the short `c` byte (`setarrayitem_gc_i/ricd`) when small
                // (`setarrayitem_gc_i` ∈ USE_C_FORM, assembler.py:339) or a
                // pool `i` slot otherwise; a ref/float Constant takes its
                // pooled byte.
                let value_kind = match value {
                    crate::model::LinkArg::Value(var) => {
                        let (reg, kc) = self.lookup_reg_with_kind_var(var, regallocs);
                        state.code.push(reg);
                        argcodes.push(kc);
                        kc
                    }
                    crate::model::LinkArg::Const(c) => {
                        let kind = crate::flatten::constant_kind(c);
                        if kind == 'i' {
                            let (byte, argcode) = self.emit_const_i_from_const_allow_short(
                                &c.value,
                                use_c_form("setarrayitem_gc_i"),
                                state,
                                callcontrol,
                            );
                            state.code.push(byte);
                            argcodes.push(argcode);
                        } else {
                            let byte = self.emit_const(&c.value, kind, state, callcontrol);
                            state.code.push(byte);
                            argcodes.push(kind);
                        }
                        kind
                    }
                };
                // pyre source-level array operations are emitted from
                // `Vec<T>` / GcArray-backed layouts that always carry a
                // length header at offset 0 (rust-source / codewriter
                // descr.py:359-362 + ARRAY_INSIDE._hints.get('nolength',
                // False): the producer carries the layout bit via
                // `OpKind::ArrayWrite::nolength`; same encoding rule as
                // ArrayRead above.
                let len_offset = if *nolength { None } else { Some(0) };
                let descr_idx = self.emit_ready_descr(arraydescrof(
                    item_ty,
                    array_type_id,
                    len_offset,
                    callcontrol,
                ));
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_setarrayitem_gc_{i,r,f}` keys off the
                // value register's kind — same rationale as setfield_gc_*.
                let opname = format!("setarrayitem_gc_{value_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            // Vable field/array: encode the base register followed by the
            // field_index descriptor, matching blackhole.py @arguments("r", "d").
            OpKind::VableFieldRead {
                base, field_index, ..
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // RPython: vable field → VableField descriptor (index, not byte offset).
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableField {
                    index: *field_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_getfield_vable_{i,r,f}` canonical keys
                // (blackhole.py:1446-1458) match on the RESULT register
                // kind. See FieldRead above for the Void/State/Unknown
                // rationale — the pyre-only declared ty can be Void
                // while the SSA result register is always i/r/f.
                let result_kind = if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                    kc
                } else {
                    'v'
                };
                let opname = format!("getfield_vable_{result_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableFieldWrite {
                base,
                field_index,
                value,
                ..
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // `setfield_vable_i` is not in USE_C_FORM
                // (`assembler.py:312-345`): a constant value never takes
                // the short `c` byte, always a pool slot.  The value kind
                // follows `getkind(v_newvalue.concretetype)` via
                // `constant_kind` — int → pool `i` (`setfield_vable_i/rid`),
                // ref/float → pool `r`/`f` (`assembler.py:168`).  A register
                // value keeps its own SSA kind.
                let value_kind = match value {
                    crate::model::LinkArg::Value(var) => {
                        let (reg, kc) = self.lookup_reg_with_kind_var(var, regallocs);
                        state.code.push(reg);
                        argcodes.push(kc);
                        kc
                    }
                    crate::model::LinkArg::Const(c) => {
                        let kind = crate::flatten::constant_kind(c);
                        let byte = self.emit_const(&c.value, kind, state, callcontrol);
                        state.code.push(byte);
                        argcodes.push(kind);
                        kind
                    }
                };
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableField {
                    index: *field_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // RPython `bhimpl_setfield_vable_{i,r,f}` canonical keys
                // (blackhole.py:1485-1495) match on the VALUE register's
                // kind. Same rationale as setfield_gc_*.
                let opname = format!("setfield_vable_{value_kind}");
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableArrayRead {
                base,
                array_index,
                elem_index,
                item_ty,
                array_itemsize,
                array_is_signed,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(elem_index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // RPython: two descriptors — fielddescr (vable array field) + arraydescr.
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableArray {
                    index: *array_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Second descriptor: arraydescr from VirtualizableInfo.array_descrs.
                let descr_idx2 = self.emit_ready_descr(vable_arraydescrof(
                    item_ty,
                    *array_itemsize,
                    *array_is_signed,
                ));
                state.code.push((descr_idx2 & 0xFF) as u8);
                state.code.push((descr_idx2 >> 8) as u8);
                argcodes.push('d');
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kc) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kc);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableArrayWrite {
                base,
                array_index,
                elem_index,
                value,
                item_ty,
                array_itemsize,
                array_is_signed,
            } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(elem_index, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                let (reg, kc) = self.lookup_reg_with_kind_var(value, regallocs);
                state.code.push(reg);
                argcodes.push(kc);
                // RPython: two descriptors — fielddescr (vable array field) + arraydescr.
                let descr_idx = self.emit_ready_descr(crate::jitcode::BhDescr::VableArray {
                    index: *array_index,
                });
                state.code.push((descr_idx & 0xFF) as u8);
                state.code.push((descr_idx >> 8) as u8);
                argcodes.push('d');
                // Second descriptor: arraydescr from VirtualizableInfo.array_descrs.
                let descr_idx2 = self.emit_ready_descr(vable_arraydescrof(
                    item_ty,
                    *array_itemsize,
                    *array_is_signed,
                ));
                state.code.push((descr_idx2 & 0xFF) as u8);
                state.code.push((descr_idx2 >> 8) as u8);
                argcodes.push('d');
                let opname = op_kind_to_opname(&op.kind);
                let key = format!("{opname}/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }
            OpKind::VableForce { base } => {
                let (reg, kc) = self.lookup_reg_with_kind_var(base, regallocs);
                assert_eq!(kc, 'r', "hint_force_virtualizable expects a Ref base");
                state.code.push(reg);
                argcodes.push(kc);
                let opnum = self.get_opnum("hint_force_virtualizable/r");
                state.code[startposition] = opnum;
            }

            // RPython jtransform.py:1714-1718 handle_jit_marker__loop_header
            // emits `SpaceOperation('loop_header', [c_index], None)`; upstream
            // assembler.py encodes that Constant via `emit_const(allow_short=
            // False)` which registers it in `constants_i` and emits a single
            // byte register index (argcodes `i`). The bhimpl signature
            // (`blackhole.py:1062 @arguments("i")`) looks the byte up in
            // `registers_i`. The canonical runtime key is `loop_header/i`
            // (`majit-metainterp/src/jitcode/mod.rs:293`); `emit_const_i`
            // returns `num_regs_i + pool_idx` which the runtime resolves
            // back to the constant via `registers_i[byte]`. Emitting via
            // the generic fallback would push zero operand bytes (because
            // `op_variable_refs(LoopHeader)` is empty), misaligning the
            // dispatch cursor.
            OpKind::LoopHeader { jitdriver_index } => {
                let reg_byte = self.emit_const_i(*jitdriver_index as i64, state);
                state.code.push(reg_byte);
                argcodes.push('i');
                let opnum = self.get_opnum("loop_header/i");
                state.code[startposition] = opnum;
            }

            // RPython jtransform.py:1690-1712 handle_jit_marker__jit_merge_point
            // emits `SpaceOperation('jit_merge_point',
            //   [Constant(jdindex), greens_i, greens_r, greens_f,
            //    reds_i, reds_r, reds_f], None)`. Upstream bhimpl signature
            // (`blackhole.py:1066 @arguments("self","i","I","R","F",
            // "I","R","F")`) reads jdindex + six typed register lists, each
            // encoded as `[len:u8][reg:u8 * N]` (assembler.py:181-196 ListOfKind).
            // pyre's runtime (`blackhole.rs:2012-2029`) consumes exactly this
            // six-list shape. The canonical runtime key is
            // `jit_merge_point/cIRFIRF` for signed-byte jitdriver indices or
            // `jit_merge_point/iIRFIRF` for constant-pool jitdriver indices.
            // The generic fallback would flatten SSA register bytes without the
            // length prefix and without the jdindex byte, corrupting the
            // stream.
            OpKind::JitMergePoint {
                jitdriver_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
            } => {
                let jdindex_value = *jitdriver_index as i64;
                let jdindex_argcode = if (-128..=127).contains(&jdindex_value) {
                    state.code.push(jdindex_value as i8 as u8);
                    'c'
                } else {
                    let jdindex_byte = self.emit_const_i(jdindex_value, state);
                    state.code.push(jdindex_byte);
                    'i'
                };
                self.emit_list_of_kind(greens_i, RegKind::Int, regallocs, state);
                self.emit_list_of_kind(greens_r, RegKind::Ref, regallocs, state);
                self.emit_list_of_kind(greens_f, RegKind::Float, regallocs, state);
                self.emit_list_of_kind(reds_i, RegKind::Int, regallocs, state);
                self.emit_list_of_kind(reds_r, RegKind::Ref, regallocs, state);
                self.emit_list_of_kind(reds_f, RegKind::Float, regallocs, state);
                argcodes.push(jdindex_argcode);
                argcodes.push('I');
                argcodes.push('R');
                argcodes.push('F');
                argcodes.push('I');
                argcodes.push('R');
                argcodes.push('F');
                let key = format!("jit_merge_point/{argcodes}");
                let opnum = self.get_opnum(&key);
                state.code[startposition] = opnum;
            }

            // Default: encode operand registers + result register (no descriptor)
            other => {
                let mut operand_kinds = String::new();
                for v in crate::inline::op_variable_refs(other) {
                    let (reg, kind_char) = self.lookup_reg_with_kind_var(&v, regallocs);
                    state.code.push(reg);
                    argcodes.push(kind_char);
                    operand_kinds.push(kind_char);
                }
                if let Some(result) = op.result.as_ref() {
                    argcodes.push('>');
                    let (reg, kind_char) = self.lookup_reg_with_kind_var(result, regallocs);
                    argcodes.push(kind_char);
                    state.code.push(reg);
                }
                let opname = op_kind_to_opname_with_kinds(other, &operand_kinds);
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
    ///
    /// RPython `assembler.py:181-196`: every item in the
    /// `ListOfKind(kind, [...])` shares the list's `kind` per
    /// construction (`flatten.py:35-51 ListOfKind` carries `kind` as
    /// an attribute and the constructors only accept matching
    /// Registers).  Pyre asserts the same invariant strictly: each
    /// item resolves through `lookup_coloring_var` and its kind must
    /// equal the list's `kind`.  A mismatch surfaces as a hard panic
    /// — the upstream contract has no escape hatch.
    fn emit_list_of_kind(
        &self,
        args: &[crate::flowspace::model::Variable],
        kind: RegKind,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        state: &mut AssemblyState,
    ) {
        // RPython `assembler.py` writes the count as a single byte and
        // does not silently truncate — clipping to 255 while still
        // emitting every item would desync the decoder by N - 255 bytes.
        // Fail loud here so the producer's contract violation surfaces
        // at the codewriter rather than as garbled bytecode downstream.
        assert!(
            args.len() < 256,
            "emit_list_of_kind: {} entries exceed the u8 count byte \
             (kind {kind:?}); RPython parity requires the count to fit \
             in a single byte",
            args.len(),
        );
        state.code.push(args.len() as u8);
        for v in args {
            let (reg, item_kind) = self.lookup_coloring_var(v, regallocs);
            assert_eq!(
                item_kind, kind,
                "emit_list_of_kind: item {v:?} has kind {item_kind:?} but the \
                 surrounding `ListOfKind` declares {kind:?} (PyPy `flatten.py:35-51` \
                 keeps every item's kind aligned with the list's `kind` attribute)",
            );
            state.code.push(reg);
        }
    }

    /// RPython `jtransform.py:424-426 rewrite_call`:
    /// ```text
    /// if lst_f or reskind == 'f': kinds = 'irf'
    /// elif lst_i or force_ir: kinds = 'ir'
    /// else: kinds = 'r'
    /// ```
    /// Result float forces `irf` even if no float args (`test_jtransform.py:356`
    /// `if RESTYPE == lltype.Float: with_f = True`). Without this rule a
    /// `&self -> f64` shape (empty `args_i`/`args_f`, single `args_r`,
    /// `result_kind='f'`) would map to a pyre-only `_r_f` handler that
    /// has no RPython `bhimpl_*_r_f` counterpart (`blackhole.py:1224,1278`
    /// only has `_r_{i,r,v}` / `_ir_*` / `_irf_*`).
    fn kinds_suffix<T, U, V>(
        &self,
        args_i: &[T],
        _args_r: &[U],
        args_f: &[V],
        result_kind: char,
    ) -> &'static str {
        if !args_f.is_empty() || result_kind == 'f' {
            "irf"
        } else if !args_i.is_empty() {
            "ir"
        } else {
            "r"
        }
    }

    /// RPython `flatten.py` emits the trailing `-> result` only for
    /// non-Void result variables. `jtransform.py:433` still computes
    /// `reskind = getkind(op.result.concretetype)[0]`, so a call op may
    /// carry a Void result placeholder while the bytecode opcode must be
    /// `*_v` with no `>` argcode.
    fn emit_call_result_arg(
        &self,
        result: Option<&crate::flowspace::model::Variable>,
        declared_result_kind: char,
        regallocs: &HashMap<RegKind, RegAllocResult>,
        state: &mut AssemblyState,
        argcodes: &mut String,
    ) -> char {
        if declared_result_kind == 'v' {
            return 'v';
        }
        let Some(result) = result else {
            return declared_result_kind;
        };
        argcodes.push('>');
        let (reg, kind) = self.lookup_coloring_var(result, regallocs);
        let kind_char = match kind {
            RegKind::Int => 'i',
            RegKind::Ref => 'r',
            RegKind::Float => 'f',
        };
        argcodes.push(kind_char);
        state.code.push(reg);
        kind_char
    }

    /// RPython: opcode key → opcode number.
    /// RPython `assembler.py:220-222`:
    /// ```text
    /// key = opname + '/' + ''.join(argcodes)
    /// num = self.insns.setdefault(key, len(self.insns))
    /// ```
    ///
    /// RPython parity: `assembler.py:220
    /// self.insns.setdefault(key, len(self.insns))`.  Each opname/
    /// argcodes key gets a stable opcode byte recorded into
    /// `self.insns`; subsequent emissions of the same key reuse the
    /// recorded byte.
    ///
    /// Pyre serialises `opcode_insns.bin` at build time and the runtime
    /// decoder reads those bytes verbatim, so canonical/extension keys
    /// pin a reserved `BC_*` (`crate::insns::wellknown_bh_insns` /
    /// `pyre_extension_insns`, merged through
    /// [`crate::insns::insn_byte_opt`]) — this preserves byte stability
    /// across builds for keys that the runtime walker dispatches.
    /// Translator-only keys (transient codewriter helpers, test
    /// fixtures) follow the upstream `setdefault` shape as closely as
    /// pyre's fixed-byte adaptation allows: scan upward from zero and
    /// allocate the lowest byte that is neither reserved by a
    /// canonical/extension key nor already used by another
    /// translator-only key.  Their byte landing in `self.insns` flows
    /// verbatim into the serialized pipeline.insns blob the runtime
    /// decoder reads.
    ///
    /// TODO: byte-stability vs. dynamic-range
    /// trade-off).  Upstream `assembler.py:221 setdefault(key,
    /// len(self.insns))` allocates densely from 0 — every emitted key
    /// consumes one of the full 256 byte slots, no reservation.  Pyre
    /// pins canonical/extension keys at fixed `BC_*` so build-time
    /// `pipeline.insns` and runtime `wellknown_bh_insns()` can decode
    /// the same byte to the same opname; the cost is that
    /// translator-only keys must avoid reserved bytes.  Earlier pyre
    /// builds allocated only above `canonical_byte_high_water()`, which
    /// made every gap below the high-water unusable.  The scanner below
    /// preserves fixed canonical bytes while recovering those gaps,
    /// leaving only actually reserved bytes unavailable.  The panic
    /// surfaces exhaustion at the offending registration site instead
    /// of silently wrapping.
    fn get_opnum(&mut self, key: &str) -> u8 {
        if let Some(&existing) = self.insns.get(&key.to_string()) {
            return existing;
        }
        if let Some(num) = crate::insns::insn_byte_opt(key) {
            debug_assert!(
                crate::insns::is_reserved_opcode_byte(num),
                "insn_byte_opt({key:?}) returned {num} which is not reserved — \
                 wellknown/extension tables out of sync with is_reserved_opcode_byte",
            );
            self.insns.insert(key.to_string(), num);
            return num;
        }
        let num = self.next_dynamic_opnum(key);
        self.insns.insert(key.to_string(), num);
        num
    }

    fn next_dynamic_opnum(&mut self, key: &str) -> u8 {
        let mut candidate = self.dynamic_byte_cursor;
        while candidate <= u8::MAX as u16 {
            let byte = candidate as u8;
            let is_available = !crate::insns::is_reserved_opcode_byte(byte)
                && !self.insns.values().any(|&used| used == byte);
            if is_available {
                self.dynamic_byte_cursor = candidate + 1;
                return byte;
            }
            candidate += 1;
        }
        panic!(
            "Assembler::get_opnum: opcode byte exhausted while assigning \
             translator-only key {key:?}; all non-reserved u8 opcode bytes \
             are already assigned"
        );
    }

    /// Resolve `(register_index, kind)` for a `&Variable`.
    ///
    /// **RPython invariant** (`flatten.py:382 getcolor`): every
    /// Variable has exactly one `(kind, color)` via
    /// `getkind(v.concretetype)` + `regallocs[kind]`.  This helper
    /// reads the declared kind directly from `Variable.concretetype`
    /// (`rtyper.py:258 v.concretetype = ...`) and looks up the color
    /// strictly in `regallocs[kind].coloring[var]` (orthodox per
    /// `tool/algo/regalloc.py:31 coloring: dict[Variable, int]`) —
    /// a hard panic on miss.
    ///
    /// When the kind cannot be derived (test fixtures whose Variables
    /// lack a populated `concretetype`), the helper falls back to a
    /// [`KINDS`]-ordered scan with a multi-class panic that still
    /// preserves "exactly one class per value" semantics.  RPython
    /// has no equivalent fallback because every assembler call comes
    /// from the typed flatten output; the fallback is documented
    /// divergence pending the migration of `encode_op` slot lookups
    /// to the strict `(v, expected_kind)` form.
    fn lookup_coloring_var(
        &self,
        var: &crate::flowspace::model::Variable,
        regallocs: &HashMap<RegKind, RegAllocResult>,
    ) -> (u8, RegKind) {
        // Strict path: `regallocs[kind]` supplies the color when
        // `Variable.concretetype` declares a non-Void/Unknown kind
        // (`flatten.py:386-387`).  A miss panics so an upstream
        // type ↔ regalloc mismatch surfaces immediately.
        use crate::model::ConcreteType;
        let declared = crate::model::FunctionGraph::concretetype_of(var);
        let kind = match declared {
            ConcreteType::Signed => Some(RegKind::Int),
            ConcreteType::GcRef => Some(RegKind::Ref),
            ConcreteType::Float => Some(RegKind::Float),
            ConcreteType::Void | ConcreteType::Unknown => None,
        };
        if let Some(kind) = kind {
            let ra = regallocs.get(&kind).unwrap_or_else(|| {
                panic!(
                    "lookup_coloring: type-state declared kind {kind:?} for {var:?} \
                     but regallocs map is missing the entry (graph={:?}, op={:?})",
                    self.current_graph_name, self.current_flatop_debug,
                )
            });
            let color = ra.coloring.get(var).copied().unwrap_or_else(|| {
                let other_classes: Vec<_> = [RegKind::Int, RegKind::Ref, RegKind::Float]
                    .iter()
                    .filter(|k| **k != kind)
                    .filter(|k| {
                        regallocs
                            .get(*k)
                            .is_some_and(|ra| ra.coloring.contains_key(var))
                    })
                    .copied()
                    .collect();
                panic!(
                    "lookup_coloring: type-state declared kind {kind:?} for {var:?} \
                     but regallocs[{kind:?}] has no coloring (other classes with a \
                     coloring: {other_classes:?}; graph={:?}, op={:?})",
                    self.current_graph_name, self.current_flatop_debug,
                )
            });
            return (color as u8, kind);
        }
        // No declared kind (Void / Unknown, or test fixture whose
        // Variable lacks a populated `concretetype`).  Fall back to
        // a KINDS-ordered scan with a single-class assertion.
        let mut found: Option<(u8, RegKind)> = None;
        for kind in [RegKind::Int, RegKind::Ref, RegKind::Float] {
            if let Some(ra) = regallocs.get(&kind) {
                if let Some(&color) = ra.coloring.get(var) {
                    if let Some((_, prev)) = found {
                        panic!(
                            "lookup_coloring: value {var:?} colored in multiple regalloc \
                             classes ({prev:?} and {kind:?}) — RPython `getkind` must \
                             give exactly one (graph={:?}, op={:?})",
                            self.current_graph_name, self.current_flatop_debug,
                        );
                    }
                    found = Some((color as u8, kind));
                }
            }
        }
        if let Some(result) = found {
            return result;
        }
        let class_coverage: Vec<_> = [RegKind::Int, RegKind::Ref, RegKind::Float]
            .iter()
            .filter_map(|k| regallocs.get(k).map(|ra| (*k, ra)))
            .map(|(k, ra)| (k, ra.coloring.len()))
            .collect();
        panic!(
            "lookup_coloring: value {var:?} has no coloring in any regalloc class \
             (graph={:?}, op={:?}, regalloc_coverage={:?})",
            self.current_graph_name, self.current_flatop_debug, class_coverage,
        );
    }

    /// Look up the register index (u8) AND kind character for a
    /// Variable — orthodox per `tool/algo/regalloc.py:31`
    /// `coloring: dict[Variable, int]`.  Returns
    /// (register_index, kind_char) where kind_char ∈ {'i','r','f'}.
    fn lookup_reg_with_kind_var(
        &self,
        var: &crate::flowspace::model::Variable,
        regallocs: &HashMap<RegKind, RegAllocResult>,
    ) -> (u8, char) {
        let (color, kind) = self.lookup_coloring_var(var, regallocs);
        let kind_char = match kind {
            RegKind::Int => 'i',
            RegKind::Ref => 'r',
            RegKind::Float => 'f',
        };
        (color, kind_char)
    }

    /// Eagerly walk every `FlatOp` in `ssarepr.insns` and report every
    /// `Variable` that lacks a regalloc coloring in any class.
    ///
    /// Pyre-only diagnostic — RPython's `assembler.py` never needs
    /// this because `rtyper` guarantees that every `Variable`'s
    /// `concretetype` produces exactly one `(kind, color)` via
    /// `getkind()` + `regalloc.py`.  Pyre's annotator / rtyper still
    /// has known coverage gaps (tracked as task #71 / #74), and the
    /// `lookup_reg_with_kind` fallback silently emits a `(0, 'r')`
    /// default at write time — which masks how many distinct gaps
    /// exist per graph.  `MAJIT_COVERAGE_PANIC=1` aborts at the first
    /// gap, losing the rest; this walker enumerates them all up
    /// front so the full gap catalogue surfaces in a single build.
    ///
    /// Output goes through `cargo:warning=` so the build script
    /// runner (`build.rs`) surfaces each line to the user.
    fn run_coverage_audit(&self, ssarepr: &SSARepr, regallocs: &HashMap<RegKind, RegAllocResult>) {
        // For each Variable, track: has a def site (result of some op),
        // count of direct operand uses, count of Live markers mentioning
        // it.  Live-only gaps (no def, no operand use) point at backward
        // liveness leakage; uses-without-def at missing rtyper coverage;
        // def-without-coverage at regalloc class mismatch.
        #[derive(Default)]
        struct ValueSites {
            has_def: bool,
            use_count: usize,
            live_count: usize,
            first_use_tag: Option<&'static str>,
        }

        fn opkind_tag(kind: &crate::model::OpKind) -> &'static str {
            use crate::model::OpKind;
            match kind {
                OpKind::Input { .. } => "Input",
                OpKind::ConstInt(_) => "ConstInt",
                OpKind::ConstBool(_) => "ConstBool",
                OpKind::ConstSymbolic { .. } => "ConstSymbolic",
                OpKind::ConstFloat(_) => "ConstFloat",
                OpKind::ConstRef(_) => "ConstRef",
                OpKind::ConstRefNull => "ConstRefNull",
                OpKind::ConstRefAddr(_) => "ConstRefAddr",
                OpKind::FieldRead { .. } => "FieldRead",
                OpKind::FieldWrite { .. } => "FieldWrite",
                OpKind::ArrayRead { .. } => "ArrayRead",
                OpKind::ArrayWrite { .. } => "ArrayWrite",
                OpKind::InteriorFieldRead { .. } => "InteriorFieldRead",
                OpKind::InteriorFieldWrite { .. } => "InteriorFieldWrite",
                OpKind::Call { .. } => "Call",
                OpKind::GuardTrue { .. } => "GuardTrue",
                OpKind::GuardFalse { .. } => "GuardFalse",
                OpKind::GuardValue { .. } => "GuardValue",
                OpKind::VtableMethodPtr { .. } => "VtableMethodPtr",
                OpKind::IndirectCall { .. } => "IndirectCall",
                OpKind::VableFieldRead { .. } => "VableFieldRead",
                OpKind::VableFieldWrite { .. } => "VableFieldWrite",
                OpKind::VableArrayRead { .. } => "VableArrayRead",
                OpKind::VableArrayWrite { .. } => "VableArrayWrite",
                OpKind::BinOp { .. } => "BinOp",
                OpKind::UnaryOp { .. } => "UnaryOp",
                OpKind::VableForce { .. } => "VableForce",
                OpKind::Hint { .. } => "Hint",
                OpKind::CallElidable { .. } => "CallElidable",
                OpKind::CallResidual { .. } => "CallResidual",
                OpKind::CallMayForce { .. } => "CallMayForce",
                OpKind::InlineCall { .. } => "InlineCall",
                OpKind::RecursiveCall { .. } => "RecursiveCall",
                OpKind::JitDebug { .. } => "JitDebug",
                OpKind::AssertGreen { .. } => "AssertGreen",
                OpKind::CurrentTraceLength => "CurrentTraceLength",
                OpKind::IsConstant { .. } => "IsConstant",
                OpKind::IsVirtual { .. } => "IsVirtual",
                OpKind::IsInstance { .. } => "IsInstance",
                OpKind::ConditionalCall { .. } => "ConditionalCall",
                OpKind::ConditionalCallValue { .. } => "ConditionalCallValue",
                OpKind::RecordKnownResult { .. } => "RecordKnownResult",
                OpKind::RecordQuasiImmutField { .. } => "RecordQuasiImmutField",
                OpKind::Live => "Live",
                OpKind::JitMergePoint { .. } => "JitMergePoint",
                OpKind::LoopHeader { .. } => "LoopHeader",
                OpKind::Abort { .. } => "Abort",
                OpKind::NewTuple { .. } => "NewTuple",
                OpKind::NewWithVtable { .. } => "NewWithVtable",
                OpKind::LoweredBlackholeOp { .. } => "LoweredBlackholeOp",
                OpKind::LoadStatic { .. } => "LoadStatic",
            }
        }
        let mut sites: std::collections::HashMap<crate::flowspace::model::Variable, ValueSites> =
            std::collections::HashMap::new();

        for op in &ssarepr.insns {
            match op {
                FlatOp::Op(inner) => {
                    let tag = opkind_tag(&inner.kind);
                    if let Some(r) = inner.result.as_ref() {
                        sites.entry(r.clone()).or_default().has_def = true;
                    }
                    // `op_variable_refs` walks the orthodox Variable
                    // operands (RPython `SpaceOperation.args:
                    // Vec<Hlvalue>`) directly out of OpKind.
                    for v in crate::inline::op_variable_refs(&inner.kind) {
                        let s = sites.entry(v).or_default();
                        s.use_count += 1;
                        s.first_use_tag.get_or_insert(tag);
                    }
                }
                FlatOp::GotoIfNot { .. }
                | FlatOp::GotoIfNotOp { .. }
                | FlatOp::Switch { .. }
                | FlatOp::IntBinOpJumpIfOvf { .. } => {
                    // Guard ops carry [`Register`] operands
                    // (post-regalloc identity); not tracked by the
                    // pre-regalloc Variable audit.
                }
                FlatOp::Move { .. } | FlatOp::Push(_) | FlatOp::Pop(_) => {
                    // Move/Push/Pop carry [`Register`]
                    // (post-regalloc identity) rather than pre-regalloc
                    // Variables, so the audit (which is keyed on
                    // pre-regalloc Variables) can no longer attribute
                    // uses/defs to a specific source Variable here.  The coverage
                    // gap-finder still reads SpaceOperation arguments
                    // via the surrounding match arms; cycle-break and
                    // copy register operands are by construction
                    // covered, so dropping them from the audit is
                    // safe.
                }
                FlatOp::LastException { .. } | FlatOp::LastExcValue { .. } => {
                    // Register operand carries (kind, color);
                    // not tracked by the pre-regalloc Variable audit.
                }
                FlatOp::IntReturn(_)
                | FlatOp::RefReturn(_)
                | FlatOp::FloatReturn(_)
                | FlatOp::Raise(_) => {
                    // Operand is RegOrConst (Register or Constant);
                    // not tracked by the pre-regalloc Variable audit.
                }
                FlatOp::Live { .. } => {
                    // Live carries [`Register`]s; not tracked by the
                    // pre-regalloc Variable audit.
                }
                FlatOp::Label(_)
                | FlatOp::Jump(_)
                | FlatOp::VoidReturn
                | FlatOp::EndOfBlock
                | FlatOp::Unreachable
                | FlatOp::CatchException { .. }
                | FlatOp::GotoIfExceptionMismatch { .. }
                | FlatOp::Reraise => {}
            }
        }

        let mut gaps: Vec<(&crate::flowspace::model::Variable, &ValueSites)> = Vec::new();
        for (var, s) in &sites {
            // The Variable-keyed `coloring` map is the orthodox
            // source per `tool/algo/regalloc.py:31`; if none of the
            // three classes registered this Variable, it surfaces as
            // a coverage gap.
            let covered = [RegKind::Int, RegKind::Ref, RegKind::Float]
                .iter()
                .any(|k| {
                    regallocs
                        .get(k)
                        .is_some_and(|ra| ra.coloring.contains_key(var))
                });
            if !covered {
                gaps.push((var, s));
            }
        }
        gaps.sort_by_key(|(var, _)| var.id());

        if gaps.is_empty() {
            return;
        }

        let class_coverage: Vec<(RegKind, usize)> = [RegKind::Int, RegKind::Ref, RegKind::Float]
            .iter()
            .filter_map(|k| regallocs.get(k).map(|ra| (*k, ra.coloring.len())))
            .collect();
        println!(
            "cargo:warning=[MAJIT_COVERAGE_AUDIT] graph={:?} gaps={} regalloc_coverage={:?}",
            ssarepr.name,
            gaps.len(),
            class_coverage,
        );
        for (v, s) in &gaps {
            let first_use = s.first_use_tag.unwrap_or("<no use>");
            println!(
                "cargo:warning=  - {v:?} def={} uses={} live={} first_use={}",
                s.has_def, s.use_count, s.live_count, first_use,
            );
        }
    }

    /// RPython assembler.py:80-138: emit_const for integer constants.
    /// Adds to constant pool and returns the index byte.
    fn emit_const(
        &mut self,
        value: &ConstValue,
        kind: char,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8 {
        match kind {
            'i' => self.emit_const_i_from_const(value, state, callcontrol),
            'r' => self.emit_const_r(value, state),
            'f' => self.emit_const_f(value, state),
            other => panic!("unknown constant kind {other:?} for {value:?}"),
        }
    }

    fn emit_const_i_from_const(
        &mut self,
        value: &ConstValue,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> u8 {
        let value = Self::resolve_const_i_value(value, callcontrol);
        self.emit_const_i(value, state)
    }

    /// Resolve an integer-kind [`ConstValue`] to its concrete `i64`
    /// (`assembler.py:168` value extraction).  `llmemory` address offsets
    /// are symbolic; resolve to the concrete byte size at code emission.
    /// Struct field offsets / sizes come from the `CallControl`'s struct
    /// layouts (it implements `OffsetLayout`); the layout-free offsets
    /// resolve even without a `callcontrol`.
    fn resolve_const_i_value(value: &ConstValue, callcontrol: Option<&CallControl>) -> i64 {
        match value {
            ConstValue::Int(n) => *n,
            ConstValue::Bool(b) => *b as i64,
            ConstValue::SpecTag(tag) => *tag as i64,
            // Symbolic inheritance-id marker: emit the resolved id. When
            // only the eager `value` is present it is the concrete id;
            // `cdef_id` reserves the key for a numbering resolution.
            ConstValue::InheritanceId { value, .. } => *value,
            ConstValue::AddressOffset(offset) => {
                let resolved = match callcontrol {
                    Some(cc) => offset.byte_size(cc),
                    None => offset.byte_size(&NoStructLayout),
                };
                resolved.unwrap_or_else(|err| panic!("emit_const_i: {err}"))
            }
            other => panic!("integer-kind constant not supported by emit_const_i: {other:?}"),
        }
    }

    /// `assembler.py:99-107` short-const for an integer-kind Constant
    /// operand: resolve to `i64`, then route through
    /// [`Self::emit_const_i_allow_short`].  Returns `(byte, argcode)` —
    /// `'c'` when the value took the inline short form, else `'i'`.
    fn emit_const_i_from_const_allow_short(
        &mut self,
        value: &ConstValue,
        allow_short: bool,
        state: &mut AssemblyState,
        callcontrol: Option<&CallControl>,
    ) -> (u8, char) {
        let value = Self::resolve_const_i_value(value, callcontrol);
        self.emit_const_i_allow_short(value, allow_short, state)
    }

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

    /// `assembler.py:99-107` — the `allow_short` branch of `emit_const`.
    /// When `allow_short` (the surrounding opname is in [`use_c_form`])
    /// and `value` fits in signed-i8 range, emit it inline as one byte
    /// (`assembler.py:106` `self.code.append(chr(value & 0xFF))`) and
    /// return argcode `'c'`; otherwise fall back to the constant pool and
    /// return argcode `'i'`.  Returns `(byte, argcode)` — the caller
    /// pushes `byte` and appends `argcode`.
    fn emit_const_i_allow_short(
        &mut self,
        value: i64,
        allow_short: bool,
        state: &mut AssemblyState,
    ) -> (u8, char) {
        if allow_short && (-128..=127).contains(&value) {
            (value as i8 as u8, 'c')
        } else {
            (self.emit_const_i(value, state), 'i')
        }
    }

    fn emit_llexitcase(&mut self, value: &ConstValue, state: &mut AssemblyState) -> u8 {
        match value {
            ConstValue::Int(value) => self.emit_const_i(*value, state),
            ConstValue::HostObject(obj) => self.emit_const_i(obj.identity_id() as i64, state),
            other => {
                panic!("goto_if_exception_mismatch: unsupported llexitcase constant {other:?}")
            }
        }
    }

    fn emit_const_r(&mut self, value: &ConstValue, state: &mut AssemblyState) -> u8 {
        // A prebuilt `Ptr(STR)` constant cannot be baked as a runtime
        // address: the translator runs in a separate build-script process,
        // so the `_ptr` target is process-local garbage at runtime.  Record
        // the string's bytes + precomputed hash for runtime materialization
        // and pool a non-canonical sentinel that the jitcode-load pass
        // overwrites with the live immortal STR GcStruct address.
        if let ConstValue::LLPtr(p) = value {
            if let Some((bytes, hash)) =
                crate::translator::rtyper::lltypesystem::rstr::prebuilt_str_bytes_and_hash(p)
            {
                return self.emit_str_const_r(bytes, hash, state);
            }
        }
        let bits = match value {
            ConstValue::HostObject(obj) => obj.identity_id() as i64,
            ConstValue::LLAddress(
                crate::translator::rtyper::lltypesystem::lltype::_address::Null,
            ) => 0,
            other => panic!("raise/r constant pool does not support {other:?}"),
        };
        self.emit_const_r_bits(bits, state)
    }

    /// Record a prebuilt-string constant for runtime materialization and
    /// pool its sentinel, returning the `r`-bank register index.  Identical
    /// literals (by bytes) share one descriptor and one sentinel, so
    /// [`Self::emit_const_r_bits`]' dedup-by-bits collapses them to a single
    /// pool entry — the analog of upstream `emit_const`'s `value_key` dedup
    /// (`assembler.py:116`).
    fn emit_str_const_r(
        &mut self,
        bytes: Vec<u8>,
        precomputed_hash: i64,
        state: &mut AssemblyState,
    ) -> u8 {
        if let Some(ordinal) = state.str_consts.iter().position(|d| d.bytes == bytes) {
            return self.emit_const_r_bits(str_const_sentinel(ordinal), state);
        }
        let ordinal = state.str_consts.len();
        let constants_r_index = state.constants_r.len();
        let reg = self.emit_const_r_bits(str_const_sentinel(ordinal), state);
        debug_assert_eq!(
            state.constants_r.len(),
            constants_r_index + 1,
            "a fresh str-const sentinel must push a new constants_r slot"
        );
        state.str_consts.push(StrConstDescriptor {
            constants_r_index,
            bytes,
            precomputed_hash,
        });
        reg
    }

    fn emit_const_r_bits(&mut self, bits: i64, state: &mut AssemblyState) -> u8 {
        if let Some(index) = state
            .constants_r
            .iter()
            .position(|&existing| existing == bits)
        {
            return (state.num_regs_r + index) as u8;
        }
        state.constants_r.push(bits);
        (state.num_regs_r + state.constants_r.len() - 1) as u8
    }

    fn emit_const_f(&mut self, value: &ConstValue, state: &mut AssemblyState) -> u8 {
        let bits = match value {
            ConstValue::Float(bits) => *bits as i64,
            other => panic!("float constant pool does not support {other:?}"),
        };
        if let Some(index) = state
            .constants_f
            .iter()
            .position(|&existing| existing == bits)
        {
            return (state.num_regs_f + index) as u8;
        }
        state.constants_f.push(bits);
        (state.num_regs_f + state.constants_f.len() - 1) as u8
    }
}

/// Non-canonical tag marking a deferred prebuilt-string slot in
/// `constants_r`.  x86-64 user addresses occupy `0..2^48`, so this high-word
/// pattern can never alias a real GCREF / host-static address; the low 48
/// bits carry the [`StrConstDescriptor`] ordinal.  The runtime load pass
/// overwrites every such slot with a live immortal STR address before the
/// jitcode is used, so the sentinel is never dereferenced (a non-canonical
/// deref would fault, surfacing any missed patch immediately).
pub const STR_CONST_SENTINEL_BASE: i64 = 0x7E57_0000_0000_0000u64 as i64;

/// `STR_CONST_SENTINEL_BASE | ordinal` — see [`STR_CONST_SENTINEL_BASE`].
fn str_const_sentinel(ordinal: usize) -> i64 {
    debug_assert!(
        (ordinal as u64) < (1u64 << 48),
        "too many prebuilt-string constants in one jitcode"
    );
    STR_CONST_SENTINEL_BASE | ordinal as i64
}

/// Per-assembly state (RPython: Assembler.setup() fields).
struct AssemblyState {
    code: Vec<u8>,
    constants_i: Vec<i64>,
    constants_r: Vec<i64>,
    constants_f: Vec<i64>,
    /// Prebuilt-string constants recorded while assembling, committed to
    /// [`JitCodeBody::str_consts`].  Parallel to `constants_r`: each entry
    /// owns the `constants_r` slot named by its `constants_r_index`.
    str_consts: Vec<StrConstDescriptor>,
    num_regs_i: usize,
    num_regs_r: usize,
    num_regs_f: usize,
    label_positions: HashMap<Label, usize>,
    tlabel_fixups: Vec<(Label, usize)>,
    startpoints: majit_ir::vec_set::VecSet<usize>,
    /// RPython assembler.py:176: positions in bytecode where TLabel operands
    /// are written. Used by JitCode.follow_jump() for verification.
    alllabels: majit_ir::vec_set::VecSet<usize>,
    /// RPython assembler.py:217-219: map from bytecode offset (after `->`)
    /// to result kind character. Recorded when encoding result registers.
    resulttypes: majit_ir::VecMap<usize, char>,
}

/// RPython: getkind(v.concretetype)[0] → 'i', 'r', 'f', 'v'.
fn value_type_to_kind(ty: &crate::model::ValueType) -> char {
    use crate::model::ValueType;
    match ty {
        // RPython `getkind(Bool/Unsigned)` returns `'int'` (`lloperation.
        // py:108`); BoolRepr's lowleveltype is `Bool` and IntegerRepr
        // shares register class with Signed/Unsigned — all `'i'` for
        // the codewriter.
        ValueType::Int | ValueType::Unsigned | ValueType::Bool => 'i',
        ValueType::Ref(_) => 'r',
        ValueType::Float => 'f',
        ValueType::Void | ValueType::State | ValueType::Unknown => 'v',
    }
}

/// `i`/`r`/`f`/`v` → `int`/`ref`/`float`/`void` for opname formation.
/// Mirrors RPython `bhimpl_<kind>_*` naming where the prefix is the full
/// kind word — `bhimpl_int_guard_value`, `bhimpl_ref_isvirtual`, etc. —
/// not the single-character argcode used inside the `/argcodes` suffix.
fn kind_char_to_name(c: char) -> &'static str {
    match c {
        'i' => "int",
        'r' => "ref",
        'f' => "float",
        _ => panic!(
            "kind_char_to_name: invalid kind char {c:?} — only 'i'/'r'/'f' \
             are valid for typed opname prefixes (void is not an operand kind)"
        ),
    }
}

fn value_type_to_itemsize(ty: &crate::model::ValueType) -> usize {
    use crate::model::ValueType;
    match ty {
        ValueType::Int => 8,
        ValueType::Ref(_) => 8,
        ValueType::Float => 8,
        _ => 8,
    }
}

fn value_type_to_ir_type_for_descr(ty: &crate::model::ValueType) -> majit_ir::value::Type {
    match ty {
        // `getkind(BOOL_TYPE)` returns `'int'` (`lloperation.py:108`);
        // `getkind(Unsigned) == 'int'` per `lltype.py` — descriptor IR
        // type tracks the register class so Bool/Unsigned alias to Int
        // rather than falling into the wildcard Ref branch.
        crate::model::ValueType::Int
        | crate::model::ValueType::Bool
        | crate::model::ValueType::Unsigned => majit_ir::value::Type::Int,
        crate::model::ValueType::Float => majit_ir::value::Type::Float,
        crate::model::ValueType::Void => majit_ir::value::Type::Void,
        _ => majit_ir::value::Type::Ref,
    }
}

fn type_flag_from_str(
    type_str: &str,
) -> (majit_ir::descr::ArrayFlag, majit_ir::value::Type, usize) {
    use majit_ir::descr::ArrayFlag;
    match type_str {
        s if s.starts_with('&')
            || s.starts_with("Box<")
            || s.starts_with("Arc<")
            || s.starts_with("Rc<")
            || s.starts_with("Vec<")
            || s.starts_with("Option<")
            || s == "String" =>
        {
            (ArrayFlag::Pointer, majit_ir::value::Type::Ref, 8)
        }
        "f64" => (ArrayFlag::Float, majit_ir::value::Type::Float, 8),
        "f32" => (ArrayFlag::Float, majit_ir::value::Type::Float, 4),
        "i64" | "isize" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 8),
        "i32" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 4),
        "i16" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 2),
        "i8" => (ArrayFlag::Signed, majit_ir::value::Type::Int, 1),
        "u64" | "usize" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 8),
        "u32" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 4),
        "u16" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 2),
        "u8" | "bool" => (ArrayFlag::Unsigned, majit_ir::value::Type::Int, 1),
        "()" => (ArrayFlag::Void, majit_ir::value::Type::Void, 0),
        _ => (ArrayFlag::Pointer, majit_ir::value::Type::Ref, 8),
    }
}

fn fallback_field_layout(
    ty: &crate::model::ValueType,
) -> (
    usize,
    majit_ir::value::Type,
    majit_ir::descr::ArrayFlag,
    bool,
) {
    let field_type = value_type_to_ir_type_for_descr(ty);
    let field_size = value_type_to_itemsize(ty);
    let field_flag = majit_ir::descr::ArrayFlag::from_field_type(field_type);
    let is_signed = field_flag == majit_ir::descr::ArrayFlag::Signed;
    (field_size, field_type, field_flag, is_signed)
}

fn bh_field_name(owner: &str, field_name: &str) -> String {
    if owner.is_empty() || field_name.contains('.') {
        field_name.to_string()
    } else {
        format!("{owner}.{field_name}")
    }
}

fn bh_field_spec_from_parts(
    index: u32,
    owner: &str,
    field_name: &str,
    offset: usize,
    field_size: usize,
    field_type: majit_ir::value::Type,
    field_flag: majit_ir::descr::ArrayFlag,
    is_immutable: bool,
    is_quasi_immutable: bool,
    index_in_parent: usize,
) -> crate::jitcode::BhFieldSpec {
    crate::jitcode::BhFieldSpec {
        index,
        name: bh_field_name(owner, field_name),
        offset,
        field_size,
        field_type,
        field_flag,
        is_field_signed: field_flag == majit_ir::descr::ArrayFlag::Signed,
        is_immutable,
        is_quasi_immutable,
        index_in_parent,
    }
}

/// Empty layout source used when no `CallControl` is threaded into
/// constant emission. The layout-free symbolic offsets (primitive
/// `ItemOffset`, `CompositeOffset`, and the standard array tokens) still
/// resolve; struct field offsets / sizes report as missing.
struct NoStructLayout;
impl crate::translator::rtyper::lltypesystem::llmemory::OffsetLayout for NoStructLayout {
    fn field_offset(&self, _struct_name: &str, _fldname: &str) -> Option<i64> {
        None
    }
    fn struct_size(&self, _struct_name: &str) -> Option<i64> {
        None
    }
}

fn bh_size_spec_from_callcontrol(
    cc: &CallControl,
    owner: &str,
) -> Option<crate::jitcode::BhSizeSpec> {
    if owner.is_empty() {
        return None;
    }
    let size = cc
        .struct_layout_for(owner)
        .map(|layout| layout.size)
        .or_else(|| heuristic_struct_size_for_bh(cc, owner))?;
    Some(crate::jitcode::BhSizeSpec {
        size,
        // Analyzer-side structs keep the guard (unchanged); the raw
        // header-less gate is driven by the runtime
        // `register_struct_layout` path.
        is_gc_managed: true,
        // `descr.py:105-127 get_size_descr` keys `_cache_size[STRUCT]` on
        // the lltype STRUCT object identity.  Pyre's analogue is
        // `path_hash(owner)` per `majit_ir::descr::path_hash` doc
        // (`majit-ir/src/descr.rs:120-141`): the analyzer side hashes
        // `field.owner_root`, the runtime macro hashes
        // `concat!(module_path!(), "::", stringify!(Struct))`.  The
        // analyzer hashes `owner` to the SAME u64 so analyzer-side
        // `BhSizeSpec` and runtime-side `__majit_type_id` produce the
        // same `LLType::Struct(u64)` cache key in `gc_cache._cache_size`.
        // MUST NOT truncate to u32 — `path_hash` has 64-bit range and
        // `as u32` collisions approach certainty around 2^16 distinct
        // structs (birthday paradox), whereas PyPy's `id(STRUCT)` never
        // aliases.  The rare hash-to-zero case (1 in 2^64) is handled
        // by `simple_descr_group_from_bh_size`'s no-identity branch.
        type_id: majit_ir::descr::path_hash(owner),
        vtable: 0,
        all_fielddescrs: bh_all_field_specs_for_struct(cc, owner),
    })
}

fn bh_all_field_specs_for_struct(
    cc: &CallControl,
    owner: &str,
) -> Vec<crate::jitcode::BhFieldSpec> {
    let mut specs = Vec::new();
    bh_all_field_specs_for_struct_into(cc, owner, &mut specs);
    specs
}

/// RPython `heaptracker.all_fielddescrs(STRUCT, res=...)` parity port:
/// recursively walks `STRUCT._names`, skipping `Void` / `typeptr` /
/// `c__pad`, and recursing into nested-struct fields so their leaf
/// fielddescrs land in the same flat `res` list with `index_in_parent`
/// matching `heaptracker.get_fielddescr_index_in()` (`heaptracker.py:51`).
///
/// Pyre keeps both a structured `struct_layouts` cache and a textual
/// `struct_field_entries` registry; the layout path doesn't carry the
/// nested-struct type name, so we cross-reference the entries registry
/// to recover the inner owner string before recursing.
fn bh_all_field_specs_for_struct_into(
    cc: &CallControl,
    owner: &str,
    specs: &mut Vec<crate::jitcode::BhFieldSpec>,
) {
    if let Some(layout) = cc.struct_layout_for(owner) {
        // The textual entries registry carries the inner-struct type name
        // for nested fields; match by field name to recover the owner
        // string when recursing.  Cloned out of `cc` so the immutable
        // borrow does not collide with the recursive call below.
        let entries: Vec<(String, String)> = cc
            .struct_field_entries(owner)
            .map(|fs| fs.to_vec())
            .unwrap_or_default();
        for fl in &layout.fields {
            if fl.field_type == majit_ir::value::Type::Void
                || fl.name == "typeptr"
                || fl.name.starts_with("c__pad")
            {
                continue;
            }
            if fl.flag == majit_ir::descr::ArrayFlag::Struct {
                // `heaptracker.py:68-69 isinstance(FIELD, lltype.Struct):
                //  all_fielddescrs(gccache, FIELD, only_gc, res, get_field_descr)`.
                if let Some(inner_owner) = entries
                    .iter()
                    .find(|(name, _)| name == &fl.name)
                    .map(|(_, ty)| ty.as_str())
                {
                    bh_all_field_specs_for_struct_into(cc, inner_owner, specs);
                }
                continue;
            }
            let index_in_parent = specs.len();
            specs.push(bh_field_spec_from_parts(
                index_in_parent as u32,
                owner,
                &fl.name,
                fl.offset,
                fl.size,
                fl.field_type,
                fl.flag,
                fl.is_immutable(),
                fl.is_quasi_immutable(),
                index_in_parent,
            ));
        }
        return;
    }

    let Some(fields) = cc.struct_field_entries(owner).map(|fs| fs.to_vec()) else {
        return;
    };
    let mut offset = 0usize;
    for (field_name, field_type_str) in &fields {
        let (field_flag, field_type, field_size) = if cc.is_known_struct(field_type_str) {
            (
                majit_ir::descr::ArrayFlag::Struct,
                majit_ir::value::Type::Ref,
                cc.struct_layout_for(field_type_str)
                    .map(|layout| layout.size)
                    .unwrap_or(std::mem::size_of::<usize>()),
            )
        } else {
            type_flag_from_str(field_type_str)
        };
        if field_type == majit_ir::value::Type::Void || field_size == 0 {
            continue;
        }
        let align = field_size.min(std::mem::size_of::<usize>());
        offset = (offset + align - 1) & !(align - 1);
        let is_skipped_field = field_name == "typeptr" || field_name.starts_with("c__pad");
        if !is_skipped_field {
            if field_flag == majit_ir::descr::ArrayFlag::Struct {
                // `heaptracker.py:68-69` recursive flatten for nested
                // structs.  `field_type_str` is the inner owner name in
                // this textual path.
                bh_all_field_specs_for_struct_into(cc, field_type_str, specs);
            } else {
                let index_in_parent = specs.len();
                let rank = cc.field_immutability(Some(owner), field_name);
                specs.push(bh_field_spec_from_parts(
                    index_in_parent as u32,
                    owner,
                    field_name,
                    offset,
                    field_size,
                    field_type,
                    field_flag,
                    rank.is_some(),
                    rank.map(|r| r.is_quasi_immutable()).unwrap_or(false),
                    index_in_parent,
                ));
            }
        }
        offset += field_size;
    }
}

fn heuristic_struct_size_for_bh(cc: &CallControl, owner: &str) -> Option<usize> {
    let fields = cc.struct_field_entries(owner)?;
    let mut offset = 0usize;
    let mut max_align = 0usize;
    for (_, field_type_str) in fields {
        let field_size = if cc.is_known_struct(field_type_str) {
            cc.struct_layout_for(field_type_str)
                .map(|layout| layout.size)
                .unwrap_or(std::mem::size_of::<usize>())
        } else {
            let (_, field_type, size) = type_flag_from_str(field_type_str);
            if field_type == majit_ir::value::Type::Void || size == 0 {
                continue;
            }
            size
        };
        // A zero-sized field (e.g. a known struct whose layout size is 0)
        // occupies no space and imposes no alignment; skip it so the
        // `align - 1` masking below never underflows on `align == 0`. The
        // final `max_align.max(1)` already floors overall alignment at 1.
        if field_size == 0 {
            continue;
        }
        let align = field_size.min(std::mem::size_of::<usize>());
        max_align = max_align.max(align);
        offset = (offset + align - 1) & !(align - 1);
        offset += field_size;
    }
    if offset == 0 {
        return Some(0);
    }
    let align = max_align.max(1);
    Some((offset + align - 1) & !(align - 1))
}

fn fielddescrof(
    field: &crate::model::FieldDescriptor,
    ty: &crate::model::ValueType,
    callcontrol: Option<&CallControl>,
) -> crate::jitcode::BhDescr {
    let (mut offset, mut field_size, mut field_type, mut field_flag, mut is_field_signed) = {
        let (field_size, field_type, field_flag, signed) = fallback_field_layout(ty);
        (0, field_size, field_type, field_flag, signed)
    };
    let mut is_immutable = false;
    let mut is_quasi_immutable = false;
    let mut index_in_parent = 0usize;
    let mut parent = None;

    if let (Some(cc), Some(owner)) = (callcontrol, field.owner_root.as_deref()) {
        parent = bh_size_spec_from_callcontrol(cc, owner);
        if let Some(parent_spec) = parent.as_ref() {
            let full_name = bh_field_name(owner, &field.name);
            if let Some(spec) = parent_spec
                .all_fielddescrs
                .iter()
                .find(|spec| spec.name == full_name)
            {
                offset = spec.offset;
                field_size = spec.field_size;
                field_type = spec.field_type;
                field_flag = spec.field_flag;
                is_field_signed = spec.is_field_signed;
                is_immutable = spec.is_immutable;
                is_quasi_immutable = spec.is_quasi_immutable;
                index_in_parent = spec.index_in_parent;
            }
        }
        if let Some(layout_field) = cc
            .struct_layout_for(owner)
            .and_then(|layout| layout.fields.iter().find(|fl| fl.name == field.name))
        {
            offset = layout_field.offset;
            field_size = layout_field.size;
            field_type = layout_field.field_type;
            field_flag = layout_field.flag;
            is_field_signed = field_flag == majit_ir::descr::ArrayFlag::Signed;
            is_immutable = layout_field.is_immutable();
            is_quasi_immutable = layout_field.is_quasi_immutable();
        } else if let Some((
            computed_offset,
            computed_size,
            computed_type,
            computed_flag,
            computed_signed,
        )) = heuristic_field_layout(cc, owner, &field.name)
        {
            offset = computed_offset;
            field_size = computed_size;
            field_type = computed_type;
            field_flag = computed_flag;
            is_field_signed = computed_signed;
        }

        if let Some(rank) = cc.field_immutability(Some(owner), &field.name) {
            is_immutable = rank.is_immutable();
            is_quasi_immutable = rank.is_quasi_immutable();
        }
    }

    crate::jitcode::BhDescr::Field {
        offset,
        field_size,
        field_type,
        field_flag,
        is_field_signed,
        is_immutable,
        is_quasi_immutable,
        index_in_parent,
        parent,
        name: field.name.clone(),
        owner: field.owner_root.clone().unwrap_or_default(),
    }
}

fn heuristic_field_layout(
    cc: &CallControl,
    owner: &str,
    field_name: &str,
) -> Option<(
    usize,
    usize,
    majit_ir::value::Type,
    majit_ir::descr::ArrayFlag,
    bool,
)> {
    let fields = cc.struct_field_entries(owner)?;
    let mut offset = 0usize;
    for (name, type_str) in fields {
        let (flag, field_type, mut field_size) = if cc.is_known_struct(type_str) {
            (
                majit_ir::descr::ArrayFlag::Struct,
                majit_ir::value::Type::Ref,
                cc.struct_layout_for(type_str)
                    .map(|layout| layout.size)
                    .unwrap_or(std::mem::size_of::<usize>()),
            )
        } else {
            type_flag_from_str(type_str)
        };
        if field_type == majit_ir::value::Type::Void || field_size == 0 {
            continue;
        }
        let align = field_size.min(std::mem::size_of::<usize>());
        offset = (offset + align - 1) & !(align - 1);
        if name == field_name {
            return Some((
                offset,
                field_size,
                field_type,
                flag,
                flag == majit_ir::descr::ArrayFlag::Signed,
            ));
        }
        field_size = field_size.max(1);
        offset += field_size;
    }
    None
}

fn bh_field_flag_from_descr(fd: &dyn majit_ir::descr::FieldDescr) -> majit_ir::descr::ArrayFlag {
    if fd.is_pointer_field() {
        majit_ir::descr::ArrayFlag::Pointer
    } else if fd.is_float_field() {
        majit_ir::descr::ArrayFlag::Float
    } else if fd.field_type() == majit_ir::value::Type::Void {
        majit_ir::descr::ArrayFlag::Void
    } else if fd.is_field_signed() {
        majit_ir::descr::ArrayFlag::Signed
    } else {
        majit_ir::descr::ArrayFlag::Unsigned
    }
}

fn bh_field_spec_from_descr(fd: &dyn majit_ir::descr::FieldDescr) -> crate::jitcode::BhFieldSpec {
    let field_flag = bh_field_flag_from_descr(fd);
    crate::jitcode::BhFieldSpec {
        index: fd.index(),
        name: fd.field_name().to_string(),
        offset: fd.offset(),
        field_size: fd.field_size(),
        field_type: fd.field_type(),
        field_flag,
        is_field_signed: fd.is_field_signed(),
        is_immutable: fd.is_immutable(),
        is_quasi_immutable: fd.is_quasi_immutable(),
        index_in_parent: fd.index_in_parent(),
    }
}

fn bh_size_spec_from_descr(sd: &dyn majit_ir::descr::SizeDescr) -> crate::jitcode::BhSizeSpec {
    crate::jitcode::BhSizeSpec {
        size: sd.size(),
        // Descr-back-to-spec inverse path: pyre's analyzer-side
        // `bh_size_spec_from_callcontrol` stamps
        // `type_id = path_hash(owner)` (u64) so the
        // `simple_descr_group_from_bh_size` round-trip resolves
        // `LLType::Struct(path_hash)` in `gc_cache._cache_size`.  The
        // `SizeDescr.cache_key()` accessor returns that same u64 (set
        // by `get_size_descr` cache-miss-mint).  Previously this used
        // `sd.type_id() as u64` — the dense GC tid widened to u64,
        // which lands on a DIFFERENT cache slot than the analyzer's
        // path_hash key, polluting cross-path identity.
        type_id: sd.cache_key(),
        vtable: sd.vtable(),
        // Round-trip the GC-header flag off the descr so a raw native
        // struct stays raw through the inverse path (it must not regain
        // a spurious `GUARD_GC_TYPE`).
        is_gc_managed: sd.is_gc_managed(),
        all_fielddescrs: sd
            .all_fielddescrs()
            .iter()
            .map(|fd| bh_field_spec_from_descr(fd.as_ref()))
            .collect(),
    }
}

pub(crate) fn bh_interior_field_specs_from_array_descr(
    array_descr: &dyn majit_ir::descr::ArrayDescr,
) -> Vec<crate::jitcode::BhInteriorFieldSpec> {
    array_descr
        .get_all_interiorfielddescrs()
        .unwrap_or(&[])
        .iter()
        .filter_map(|descr| {
            let interior = descr.as_interior_field_descr()?;
            let field = bh_field_spec_from_descr(interior.field_descr());
            let owner = interior
                .field_descr()
                .get_parent_descr()
                .and_then(|parent| parent.as_size_descr().map(bh_size_spec_from_descr))
                .unwrap_or_else(|| crate::jitcode::BhSizeSpec {
                    size: array_descr.item_size(),
                    type_id: 0,
                    vtable: 0,
                    is_gc_managed: true,
                    all_fielddescrs: vec![field.clone()],
                });
            Some(crate::jitcode::BhInteriorFieldSpec {
                index: descr.index(),
                field,
                owner,
            })
        })
        .collect()
}

/// jtransform.py:773,802 cpu.arraydescrof(ARRAY) equivalent.
///
/// Determines the full ArrayDescr shape from the array element type.
/// When `array_type_id` is available (e.g. `Vec<i32>` → element `i32`),
/// the result is exact. Fallback uses descr.py:241-254 get_type_flag()
/// semantics: Int → FLAG_SIGNED, Float/Ref → FLAG_UNSIGNED/FLAG_FLOAT.
///
/// When `callcontrol` is present, this routes through
/// `CallControl::arraydescrof_for_type` and carries the EffectInfo
/// `ei_index` across the BhDescr boundary.  The `callcontrol == None`
/// fallback is descriptor-shape-only: it must not be used for EffectInfo raw
/// sets because there is no codewriter-side array namespace to publish.
fn arraydescrof(
    ty: &crate::model::ValueType,
    array_type_id: &Option<String>,
    len_offset: Option<usize>,
    callcontrol: Option<&CallControl>,
) -> crate::jitcode::BhDescr {
    let ir_type = value_type_to_ir_type_for_descr(ty);
    if let Some(cc) = callcontrol {
        // Route through `arraydescrof_for_type` so the bytecode emit
        // path shares the same `(item_ty, array_type_id, len_offset) → ei_index`
        // table as `writeanalyze` in `call.rs`; otherwise
        // every emit-time descr lands at `ei_index = 0` and aliases
        // distinct ARRAY identities at `force_from_effectinfo`
        // (`heap.py:540-560`, `heap.rs:839 array_effect_index`).
        let descr = cc.arraydescrof_for_type(ty, array_type_id, ir_type, len_offset);
        let array_descr = descr
            .as_array_descr()
            .expect("CallControl::arraydescrof must return an ArrayDescr");
        return crate::jitcode::BhDescr::Array {
            base_size: array_descr.base_size(),
            itemsize: array_descr.item_size(),
            len_offset: array_descr.len_descr().map(|fd| fd.offset()),
            // `descr.py:348-378` cache identity — `ArrayDescr.cache_key()`
            // returns the u64 `path_hash(array_type_id)` slot the analyzer
            // stamped at `gc_cache.get_array_descr` cache-miss-mint.
            // Round-trips through `_cache_array[LLType::Array(cache_key)]`
            // on the runtime side.
            type_id: array_descr.cache_key(),
            item_type: array_descr.item_type(),
            is_array_of_pointers: array_descr.is_array_of_pointers(),
            is_array_of_structs: array_descr.is_array_of_structs(),
            is_item_signed: array_descr.is_item_signed(),
            // descr.py:465 compute_bitstrings — carry the SimpleArrayDescr's
            // ei_index across the BhDescr boundary so make_descr_from_bh
            // republishes it on the runtime SimpleArrayDescr.
            ei_index: descr.get_ei_index(),
            // descr.py:348-360 cache identity — carry the codewriter
            // `array_type_id` across the BhDescr boundary so the
            // runtime `ArrayDescrKey` keeps two distinct ARRAY lltypes
            // on distinct slots even when their structural tuples
            // coincide (`type_id == 0` default, same item layout).
            array_type_id: array_type_id.clone(),
            interior_fields: bh_interior_field_specs_from_array_descr(array_descr),
        };
    }

    // Primary path: extract element type from the array type identity
    // (our equivalent of `ARRAY.OF` in RPython).
    let (flag, item_type, itemsize) = if let Some(elem) = array_type_id
        .as_deref()
        .and_then(extract_element_type_from_str)
    {
        type_flag_from_str(elem.as_str())
    } else {
        match ty {
            crate::model::ValueType::Int => (
                majit_ir::descr::ArrayFlag::Signed,
                majit_ir::value::Type::Int,
                8,
            ),
            crate::model::ValueType::Float => (
                majit_ir::descr::ArrayFlag::Float,
                majit_ir::value::Type::Float,
                8,
            ),
            crate::model::ValueType::Ref(_) => (
                majit_ir::descr::ArrayFlag::Pointer,
                majit_ir::value::Type::Ref,
                8,
            ),
            _ => (
                majit_ir::descr::ArrayFlag::Unsigned,
                majit_ir::value::Type::Int,
                8,
            ),
        }
    };
    // descr.py:354/359-362 + symbolic.get_array_token — basesize follows
    // the lltype's nolength flag: nolength → items at offset 0;
    // length-prefixed → items past the header at len_offset + WORD.
    let base_size = match len_offset {
        None => 0,
        Some(off) => off + std::mem::size_of::<usize>(),
    };
    crate::jitcode::BhDescr::Array {
        base_size,
        itemsize,
        len_offset,
        type_id: 0,
        item_type,
        is_array_of_pointers: flag == majit_ir::descr::ArrayFlag::Pointer,
        is_array_of_structs: flag == majit_ir::descr::ArrayFlag::Struct,
        is_item_signed: flag == majit_ir::descr::ArrayFlag::Signed,
        // No CallControl-side `array_index` for the fallback path.  This
        // descriptor is shape-only for codewriter-less emission helpers; any
        // path that needs EffectInfo heap invalidation must pass CallControl
        // so `arraydescrof_for_type` can publish the real `ei_index`.
        ei_index: u32::MAX,
        // Codewriter-less fallback still carries the ARRAY identity
        // string so the runtime registry keeps distinct lltypes
        // distinct.
        array_type_id: array_type_id.clone(),
        interior_fields: Vec::new(),
    }
}

fn vable_arraydescrof(
    ty: &crate::model::ValueType,
    itemsize: usize,
    is_item_signed: bool,
) -> crate::jitcode::BhDescr {
    let item_type = value_type_to_ir_type_for_descr(ty);
    crate::jitcode::BhDescr::Array {
        base_size: std::mem::size_of::<usize>(),
        itemsize,
        len_offset: Some(0),
        type_id: 0,
        item_type,
        is_array_of_pointers: matches!(item_type, majit_ir::value::Type::Ref),
        is_array_of_structs: false,
        is_item_signed,
        // vable array slots are codewriter-known per-vinfo descrs, not
        // EffectInfo-keyed; ei_index stays unset.
        ei_index: u32::MAX,
        // vable array slots have no source-level array_type_id;
        // distinct vable indices are already disambiguated by the
        // parent `VableArray { index }` variant carried alongside.
        array_type_id: None,
        interior_fields: Vec::new(),
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

/// Convert OpKind to an opname string for the assembler's instruction table.
/// RPython: the opname comes from SpaceOperation.opname.
/// Convert OpKind to a typed opname matching RPython's jtransform output.
///
/// RPython jtransform produces fully-qualified names like `getfield_vable_i`,
/// `setfield_gc_r`, `int_add`. The kind suffix comes from the result type
/// or value type of the operation.
/// Variant of [`op_kind_to_opname`] that routes operand-kind-sensitive
/// names through the proper RPython opname.  Specifically:
///
/// `OpKind::UnaryOp { op: "bool", .. }` is the truthify operator
/// pyre's frontend emits from the `&&`/`||`/`!` desugar (the bool
/// switch discriminator).  Upstream RPython lowers `bool` per the
/// operand's repr at rtyper time:
///
/// - `IntegerRepr.rtype_bool` → `genop("int_is_true", ...)`
///   (`rint.py:200-205`)
/// - `PtrRepr.rtype_bool` → `genop("ptr_nonzero", ...)`
///   (`rmodel.py::PtrRepr.rtype_bool`)
/// - `FloatRepr.rtype_bool` → `genop("float_ne", ..., 0.0)`
///   (`rfloat.py:191`); the `/f>i` shape comes from the float
///   compare, but pyre's truthify uses `float_ne` against zero
///   directly so the assembler key is `float_ne/ff>i` after the
///   constant pool emits the 0.0.  No `bool` op survives at the
///   `f` operand kind here today, so the float arm is a defensive
///   placeholder.
///
/// The unconditional `int_<op>` prefix in [`op_kind_to_opname`]
/// would name these `int_bool/i>i` and `int_bool/r>i`, which has no
/// blackhole handler (RPython has no `int_bool` opname) and trips
/// the strict-coverage `default_bh_builder_unwired_set_matches_task_85_snapshot`
/// guard.  Routing on the operand kind here keeps the legacy/codewriter
/// path producing handler-backed opnames without requiring a full
/// rtyper port for every prebuilt graph.
fn op_kind_to_opname_with_kinds(kind: &crate::model::OpKind, operand_kinds: &str) -> String {
    use crate::model::OpKind;
    if let OpKind::UnaryOp { op, .. } = kind
        && op == "bool"
    {
        return match operand_kinds {
            "i" => "int_is_true".into(),
            "r" => "ptr_nonzero".into(),
            // RPython `jtransform.py:1627 rewrite_op_float_is_true`
            // collapses both `bool/f` and `float_is_true/f` to
            // `float_ne(x, 0.0)` upstream of the assembler — pyre's
            // jtransform mirror at `jtransform.rs:917-984` covers
            // both surfaces, so an `f` operand reaching here means
            // the rewrite was skipped.  Fail loud rather than emit
            // a `float_is_true` opname the backend does not register
            // (`rpython/jit/codewriter/jtransform.py:1627` is
            // unconditional, so pyre matches that invariant here).
            "f" => unreachable!(
                "OpKind::UnaryOp {{ op: \"bool\", .. }} over an `f` operand must be \
                 rewritten to float_ne in jtransform — see jtransform.rs:917"
            ),
            _ => format!("int_{op}"),
        };
    }
    // RPython parity (`jtransform.py:1243-1255`): equality / inequality
    // on Ref operands lowers to `ptr_eq` / `ptr_ne`, not the integer
    // form.  Float arithmetic carries the full `float_*` opname through
    // `op_kind_to_opname`'s BinOp arm already; here we only need to
    // route the `eq` / `ne` labels emitted by the MIR front-end when
    // they meet two Ref operands.  Mixed `ri` / `ir` Ref-Int shapes
    // remain a kind-flow gap (see `default_bh_builder_unwired_set_*`
    // snapshot) — surfaced via the canonical fallthrough below.
    if let OpKind::BinOp { op, .. } = kind {
        match (op.as_str(), operand_kinds) {
            ("eq", "rr") => return "ptr_eq".into(),
            ("ne", "rr") => return "ptr_ne".into(),
            _ => {}
        }
    }
    op_kind_to_opname(kind)
}

fn op_kind_to_opname(kind: &crate::model::OpKind) -> String {
    use crate::model::OpKind;
    match kind {
        OpKind::Input { ty, .. } => format!("input_{}", value_type_to_kind(ty)),
        // RPython: ConstInt is NOT a standalone op; see encode_op comment.
        // Pyre materialises constants as an int_copy from pool-region reg.
        OpKind::ConstInt(_) => "int_copy".into(),
        // RPython folds `lltype.Bool` into kind `'int'`
        // (`flatten.py:getkind`), so the bool constant materialises
        // through the same `int_copy` path as `ConstInt`.
        OpKind::ConstBool(_) => "int_copy".into(),
        // The `_we_are_jitted` symbolic is folded to `ConstBool(true)` by
        // `jtransform` before assembly, so it never reaches opname
        // emission.
        OpKind::ConstSymbolic { .. } => {
            unreachable!("OpKind::ConstSymbolic must be folded by jtransform before assembly")
        }
        // Mirrors `ConstInt` — the constant is materialised through the
        // shared `constants_f` pool, then a `float_copy` op moves it into
        // the SSA destination register.
        OpKind::ConstFloat(_) => "float_copy".into(),
        // Ref-bank analog of `ConstInt`. The singleton HostObject's
        // `identity_id()` enters the shared `constants_r` pool via
        // `emit_const_r`, then a `ref_copy/r>r` op moves it into the
        // SSA destination register.
        OpKind::ConstRef(_) | OpKind::ConstRefNull | OpKind::ConstRefAddr(_) => "ref_copy".into(),
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
        OpKind::NewWithVtable { .. } => "new_with_vtable".into(),
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
        OpKind::GuardValue { kind_char, .. } => {
            // `rpython/jit/codewriter/jtransform.py:611` emits one of
            // `int_guard_value` / `ref_guard_value` / `float_guard_value`
            // depending on the `getkind()` of the guarded arg.
            //
            // RPython also emits `str_guard_value` for `promote_string`
            // (jit.py:631) / `promote_unicode` (jit.py:647), but pyre
            // panics in those rewrite arms (pyre-object lacks an
            // `rstr.STR` / `rstr.UNICODE` GC layout) so `kind_char` is
            // always one of `'i'` / `'r'` / `'f'` here.
            format!("{}_guard_value", kind_char_to_name(*kind_char))
        }
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
        // `int_and` / `int_or` / `int_xor`. When an `OpKind::BinOp.op`
        // arrives spelled with Rust's `syn::BinOp` trait names
        // (`bitand`/`bitor`/`bitxor`) for source faithfulness, rename them
        // here at the emission boundary instead of duplicating wire entries
        // in the blackhole dispatch table.
        OpKind::BinOp { op, .. } => match op.as_str() {
            "bitand" => "int_and".into(),
            "bitor" => "int_or".into(),
            "bitxor" => "int_xor".into(),
            // RPython `jtransform.py:1243-1255` produces these opnames as-is —
            // do not prefix with `int_`.
            "ptr_eq" | "ptr_ne" => op.clone(),
            // jtransform-rewritten float operands carry the full RPython
            // opname (`float_add` / `float_lt` / etc.) — preserve as-is.
            s if s.starts_with("float_") => op.clone(),
            _ => format!("int_{op}"),
        },
        // RPython `blackhole.py:488-498`: bitwise NOT on i64 is `int_invert`.
        // pyre's front-end uses Rust's `syn::UnOp::Not` spelling `not` for
        // both logical-not and bitwise-not (they share the `!` token at the
        // AST level); canonicalize to `int_invert` at the emission boundary.
        //
        // `bool` (truthify) lacks per-operand-kind dispatch here — see
        // [`op_kind_to_opname_with_kinds`] which routes `bool` to
        // `int_is_true` / `ptr_nonzero` / `float_ne` based on the
        // operand's actual register-class.
        OpKind::UnaryOp { op, .. } => match op.as_str() {
            "not" => "int_invert".into(),
            // Already-canonical opnames produced by the rtyper / cast
            // family / jtransform rewrites — preserve as-is so the
            // unconditional `int_` prefix below does not double up
            // (`int_is_true` would otherwise become `int_int_is_true`,
            // which has no blackhole handler).  RPython's
            // `rint.py:rtype_int__Bool` emits `int_is_true` directly,
            // and `ptr_nonzero` / `same_as` / cast_* are similarly
            // already-canonical.
            "int_is_true" | "ptr_nonzero" | "ptr_iszero" | "same_as" => op.clone(),
            s if s.starts_with("int_") || s.starts_with("uint_") => op.clone(),
            s if s.starts_with("float_") || s.starts_with("cast_") => op.clone(),
            _ => format!("int_{op}"),
        },
        OpKind::VableForce { .. } => "hint_force_virtualizable".into(),
        // `Hint` is consumed by `jtransform::rewrite_op_hint` (lowered to
        // `same_as` / `VableForce` / `*_guard_value`) before assembly, so it
        // never reaches opname emission.
        OpKind::Hint { .. } => {
            unreachable!("OpKind::Hint must be rewritten by jtransform before assembly")
        }
        // jtransform.py:1731-1743 — jit.* builtin ops
        OpKind::JitDebug { .. } => "jit_debug".into(),
        OpKind::AssertGreen { kind_char, .. } => {
            format!("{}_assert_green", kind_char_to_name(*kind_char))
        }
        OpKind::CurrentTraceLength => "current_trace_length".into(),
        OpKind::IsConstant { kind_char, .. } => {
            format!("{}_isconstant", kind_char_to_name(*kind_char))
        }
        OpKind::IsVirtual { kind_char, .. } => {
            format!("{}_isvirtual", kind_char_to_name(*kind_char))
        }
        // The rtyper at rtyper.rs:2035 dispatches the flowspace
        // `"isinstance"` opname through `InstanceRepr::rtype_isinstance`,
        // which lowers it into a direct_call to the per-class
        // `ll_isinstance_const_*` or unspecialised `ll_isinstance`
        // helper.  By the time the codewriter assembler runs every
        // `OpKind::IsInstance` should already have been replaced with
        // that direct_call, so reaching this arm signals an rtyper
        // gap.  Fail loudly rather than emit an `"isinstance"` opname
        // with no corresponding entry in `insns.rs::wellknown_bh_insns`.
        OpKind::IsInstance { .. } => panic!(
            "OpKind::IsInstance reached the JitCode assembler; the rtyper \
             must lower isinstance to ll_isinstance* direct_call before \
             codewriter emission (rtyper.rs:2035 / \
             InstanceRepr::rtype_isinstance)"
        ),
        OpKind::RecordKnownResult { result_kind, .. } => {
            format!("record_known_result_{result_kind}")
        }
        // jtransform.py:1665-1688 — conditional_call ops
        OpKind::ConditionalCall { .. } => "conditional_call".into(),
        OpKind::ConditionalCallValue { result_kind, .. } => {
            format!("conditional_call_value_{result_kind}")
        }
        OpKind::Live => "live".into(),
        // jtransform.py:1707,1718 — jit_merge_point / loop_header markers.
        OpKind::JitMergePoint { .. } => "jit_merge_point".into(),
        OpKind::LoopHeader { .. } => "loop_header".into(),
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
        OpKind::Abort { .. } => "abort".into(),
        OpKind::NewTuple { .. } => "newtuple".into(),
        // The opname-dispatch spine lowers the rtyper helper graphs to
        // register-shaped blackhole insns, carrying the resolved opname
        // (`strlen`/`strgetitem`/`strsetitem`/`newstr`/…) verbatim.  The
        // blackhole interpreter resolves the handler by this name, and
        // `Assembler::get_opnum` assigns its byte dynamically when the
        // opname has no fixed `insns.rs` slot.
        OpKind::LoweredBlackholeOp { opname, .. } => opname.clone(),
        // RPython's codewriter never sees a crate-static carrier:
        // LOAD_GLOBAL is resolved to a Constant before JitCode assembly.
        // The flowspace adapter must either fold `LoadStatic` to a
        // constant `same_as` that rtyper/removenoops eliminates, or reject
        // the unresolved static before this layer.  Emitting `same_as/*`
        // here would create an opcode with no blackhole handler.
        OpKind::LoadStatic { segments, .. } => {
            panic!("unresolved LoadStatic reached JitCode assembly: {segments:?}")
        }
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
    pub fn insns(&self) -> &majit_ir::VecMap<String, u8> {
        &self.insns
    }

    /// Register an `(opname/argcodes, opnum)` pair into `self.insns`.
    ///
    /// RPython `assembler.py:222 self.insns[key] = opnum` records every
    /// opcode the assembler emits during `assemble()`.  Pyre's
    /// state-field-JIT macro path skips `assemble()` entirely (the
    /// `JitCodeBuilder` emits BC_* directly), so the canonical entries
    /// — `live/`, `catch_exception/L`, `*_return/*` — are populated
    /// here at install time so `MetaInterpStaticData::setup_insns`
    /// (`pyjitpl.py:2227-2243`) can do the dynamic
    /// `insns.get(name)` lookup instead of a parallel hardcoded
    /// `BC_*` seeding block.
    pub fn register_insn(&mut self, name: &str, opnum: u8) {
        self.insns.insert(name.to_string(), opnum);
    }

    /// RPython `assembler.py:29 self.all_liveness = []` — the shared
    /// liveness byte stream populated by `_encode_liveness`.  Returned
    /// as a contiguous `&[u8]` view so consumers (notably
    /// `MetaInterpStaticData::finish_setup` per `pyjitpl.py:2264`) can
    /// take a snapshot without depending on the dedup cache or
    /// position table.
    pub fn all_liveness(&self) -> &[u8] {
        &self.all_liveness
    }

    /// Snapshot the descriptor table after all jitcodes have been fully
    /// assembled. Pending inline-call descriptors are lowered here to the
    /// final `(jitcode_index, fnaddr, calldescr)` form that runtime
    /// consumers expect.
    pub fn snapshot_descrs(&self) -> Vec<crate::jitcode::BhDescr> {
        self.descrs
            .iter()
            .map(|descr| match descr {
                AssemblerDescr::Ready(descr) => descr.clone(),
                AssemblerDescr::PendingJitCode { jitcode } => crate::jitcode::BhDescr::JitCode {
                    jitcode_index: jitcode.index(),
                    fnaddr: jitcode.fnaddr,
                    calldescr: jitcode.calldescr().clone(),
                },
                AssemblerDescr::PendingSwitch { .. } => {
                    panic!("snapshot_descrs called before switch descriptors were resolved")
                }
            })
            .collect()
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

/// `effectinfo.py:152-164` `EffectInfo._cache` cache key parity.
///
/// PyPy keys the EI factory cache on the raw `frozenset[Descr]`
/// readonly/write sets, NOT on the `bitstring_*` fields.  The
/// bitstrings are setup-time derived state (`compute_bitstrings`
/// at `effectinfo.py:528`), so the same logical EI must hit the
/// same cache slot before AND after compaction.  Pyre's lift
/// projects the `Vec<DescrRef>` raw sets to `Arc::as_ptr` ptr-id
/// `Vec<usize>` for `Hash`/`Eq` — direct lift of PyPy's
/// `frozenset[id(descr)]` cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EffectInfoKey {
    extraeffect: majit_ir::descr::ExtraEffect,
    oopspecindex: majit_ir::descr::OopSpecIndex,
    readonly_descrs_fields: Option<Vec<usize>>,
    write_descrs_fields: Option<Vec<usize>>,
    readonly_descrs_arrays: Option<Vec<usize>>,
    write_descrs_arrays: Option<Vec<usize>>,
    readonly_descrs_interiorfields: Option<Vec<usize>>,
    write_descrs_interiorfields: Option<Vec<usize>>,
    can_invalidate: bool,
    can_collect: bool,
    call_release_gil_target: (u64, i32),
}

impl EffectInfoKey {
    fn from_effect_info(effect: &majit_ir::descr::EffectInfo) -> Self {
        Self {
            extraeffect: effect.extraeffect,
            oopspecindex: effect.oopspecindex,
            // `effectinfo.py:152-164` cache key: raw `_*_descrs_*`
            // sets (frozenset[Descr] lift), projected to
            // `Arc::as_ptr` ptr-ids.  NOT the lazily-published
            // `bitstring_*` fields.
            readonly_descrs_fields: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._readonly_descrs_fields,
            ),
            write_descrs_fields: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._write_descrs_fields,
            ),
            readonly_descrs_arrays: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._readonly_descrs_arrays,
            ),
            write_descrs_arrays: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._write_descrs_arrays,
            ),
            readonly_descrs_interiorfields: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._readonly_descrs_interiorfields,
            ),
            write_descrs_interiorfields: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._write_descrs_interiorfields,
            ),
            can_invalidate: effect.can_invalidate,
            can_collect: effect.can_collect,
            call_release_gil_target: effect.call_release_gil_target,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AssemblerDescrKey {
    Field {
        offset: usize,
        field_size: usize,
        field_type: majit_ir::value::Type,
        field_flag: majit_ir::descr::ArrayFlag,
        is_field_signed: bool,
        is_immutable: bool,
        is_quasi_immutable: bool,
        index_in_parent: usize,
        parent: Option<crate::jitcode::BhSizeSpec>,
        name: String,
        owner: String,
    },
    Array {
        base_size: usize,
        itemsize: usize,
        len_offset: Option<usize>,
        /// u64 cache-key surrogate matching `BhDescr::Array.type_id`.
        type_id: u64,
        item_type: majit_ir::value::Type,
        is_array_of_pointers: bool,
        is_array_of_structs: bool,
        is_item_signed: bool,
        // `ei_index` deliberately omitted from the identity tuple —
        // upstream `gccache._cache_array[ARRAY_OR_STRUCT]`
        // (`descr.py:348-360`) keys on the lltype itself, and
        // `compute_bitstrings` (`effectinfo.py:465`) later assigns the
        // index slot as a derived attribute that multiple descrs are
        // free to share.
        //
        // `array_type_id` joins the identity tuple as the codewriter
        // lltype-identity proxy so two ARRAYs that disagree only on
        // the Rust type string (e.g. `Vec<Foo>` vs `Vec<Bar>` with
        // both at `type_id == 0`) keep distinct slots in the
        // assembler's `_descr_dict`, mirroring upstream's per-lltype
        // cache identity.
        array_type_id: Option<String>,
        interior_fields: Vec<crate::jitcode::BhInteriorFieldSpec>,
    },
    Size {
        size: usize,
        /// u64 cache-key surrogate matching `BhDescr::Size.type_id`.
        type_id: u64,
        vtable: usize,
        owner: String,
        all_fielddescrs: Vec<crate::jitcode::BhFieldSpec>,
    },
    Call {
        arg_classes: String,
        result_type: char,
        result_signed: bool,
        result_size: usize,
        result_erased: crate::jitcode::CallResultErasedKey,
        effect: EffectInfoKey,
    },
    /// RPython uses the JitCode object itself as an AbstractDescr for
    /// inline_call. The Rust key is therefore the identity-keyed handle, not
    /// the callsite-local `BhCallDescr`.
    JitCode(crate::jitcode::JitCodeHandle),
    SnapshotJitCode {
        jitcode_index: usize,
        fnaddr: i64,
        arg_classes: String,
        result_type: char,
        result_signed: bool,
        result_size: usize,
        result_erased: crate::jitcode::CallResultErasedKey,
        effect: EffectInfoKey,
    },
    Switch(Vec<(i64, usize)>),
    VableField {
        index: usize,
    },
    VableArray {
        index: usize,
    },
    VtableMethod {
        trait_root: String,
        method_name: String,
    },
    /// `descr.py:388 InteriorFieldDescr(arraydescr, fielddescr)` identity
    /// is the composition of the array and field descriptor keys.
    InteriorField {
        array: Box<AssemblerDescrKey>,
        field: Box<AssemblerDescrKey>,
    },
}

impl AssemblerDescrKey {
    fn from_descr(descr: &AssemblerDescr) -> Self {
        match descr {
            AssemblerDescr::Ready(descr) => Self::from_ready(descr),
            AssemblerDescr::PendingJitCode { jitcode } => Self::JitCode(jitcode.clone()),
            AssemblerDescr::PendingSwitch { .. } => {
                unreachable!("switch descriptors bypass `_descr_dict`")
            }
        }
    }

    fn from_ready(descr: &crate::jitcode::BhDescr) -> Self {
        match descr {
            crate::jitcode::BhDescr::Field {
                offset,
                field_size,
                field_type,
                field_flag,
                is_field_signed,
                is_immutable,
                is_quasi_immutable,
                index_in_parent,
                parent,
                name,
                owner,
            } => Self::Field {
                offset: *offset,
                field_size: *field_size,
                field_type: *field_type,
                field_flag: *field_flag,
                is_field_signed: *is_field_signed,
                is_immutable: *is_immutable,
                is_quasi_immutable: *is_quasi_immutable,
                index_in_parent: *index_in_parent,
                parent: parent.clone(),
                name: name.clone(),
                owner: owner.clone(),
            },
            crate::jitcode::BhDescr::Array {
                base_size,
                itemsize,
                len_offset,
                type_id,
                item_type,
                is_array_of_pointers,
                is_array_of_structs,
                is_item_signed,
                // `ei_index` intentionally not part of the identity
                // tuple — see `AssemblerDescrKey::Array` comment.
                ei_index: _,
                array_type_id,
                interior_fields,
            } => Self::Array {
                base_size: *base_size,
                itemsize: *itemsize,
                len_offset: *len_offset,
                type_id: *type_id,
                item_type: *item_type,
                is_array_of_pointers: *is_array_of_pointers,
                is_array_of_structs: *is_array_of_structs,
                is_item_signed: *is_item_signed,
                array_type_id: array_type_id.clone(),
                interior_fields: interior_fields.clone(),
            },
            crate::jitcode::BhDescr::Size {
                size,
                type_id,
                vtable,
                owner,
                all_fielddescrs,
                // Functionally determined by `type_id` (a struct is GC or
                // raw, not both), so not part of the dedup-key identity.
                is_gc_managed: _,
            } => Self::Size {
                size: *size,
                type_id: *type_id,
                vtable: *vtable,
                owner: owner.clone(),
                all_fielddescrs: all_fielddescrs.clone(),
            },
            crate::jitcode::BhDescr::Call { calldescr } => Self::Call {
                arg_classes: calldescr.arg_classes.clone(),
                result_type: calldescr.result_type,
                result_signed: calldescr.result_signed,
                result_size: calldescr.result_size,
                result_erased: calldescr.result_erased,
                effect: EffectInfoKey::from_effect_info(&calldescr.extra_info),
            },
            crate::jitcode::BhDescr::JitCode {
                jitcode_index,
                fnaddr,
                calldescr,
            } => Self::SnapshotJitCode {
                jitcode_index: *jitcode_index,
                fnaddr: *fnaddr,
                arg_classes: calldescr.arg_classes.clone(),
                result_type: calldescr.result_type,
                result_signed: calldescr.result_signed,
                result_size: calldescr.result_size,
                result_erased: calldescr.result_erased,
                effect: EffectInfoKey::from_effect_info(&calldescr.extra_info),
            },
            crate::jitcode::BhDescr::Switch { dict } => {
                let mut items: Vec<_> = dict.iter().map(|(key, value)| (*key, *value)).collect();
                items.sort_unstable_by_key(|(key, _)| *key);
                Self::Switch(items)
            }
            crate::jitcode::BhDescr::VableField { index } => Self::VableField { index: *index },
            crate::jitcode::BhDescr::VableArray { index } => Self::VableArray { index: *index },
            crate::jitcode::BhDescr::VtableMethod {
                trait_root,
                method_name,
            } => Self::VtableMethod {
                trait_root: trait_root.clone(),
                method_name: method_name.clone(),
            },
            crate::jitcode::BhDescr::InteriorField { array, field } => Self::InteriorField {
                array: Box::new(Self::from_ready(array)),
                field: Box::new(Self::from_ready(field)),
            },
        }
    }
}

#[derive(Debug, Clone)]
enum AssemblerDescr {
    Ready(crate::jitcode::BhDescr),
    PendingJitCode {
        jitcode: crate::jitcode::JitCodeHandle,
    },
    PendingSwitch {
        cases: Vec<(i64, Label)>,
    },
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{ConstValue, HostObject};
    use crate::regalloc;

    fn empty_regallocs() -> HashMap<RegKind, regalloc::RegAllocResult> {
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
        regallocs
    }

    fn empty_state() -> AssemblyState {
        AssemblyState {
            code: Vec::new(),
            constants_i: Vec::new(),
            constants_r: Vec::new(),
            constants_f: Vec::new(),
            str_consts: Vec::new(),
            num_regs_i: 4,
            num_regs_r: 0,
            num_regs_f: 0,
            label_positions: HashMap::new(),
            tlabel_fixups: Vec::new(),
            startpoints: majit_ir::vec_set::VecSet::new(),
            alllabels: majit_ir::vec_set::VecSet::new(),
            resulttypes: majit_ir::VecMap::new(),
        }
    }

    #[test]
    fn use_c_form_matches_assembler_py_membership() {
        // `assembler.py:312` USE_C_FORM members reachable build-time …
        assert!(use_c_form("int_copy"));
        assert!(use_c_form("int_return"));
        assert!(use_c_form("jit_merge_point"));
        // … and non-members that must keep the pooled `i`/`r`/`f` argcode.
        assert!(!use_c_form("ref_copy"));
        assert!(!use_c_form("float_copy"));
        assert!(!use_c_form("getfield_gc_i"));
    }

    #[test]
    fn emit_const_i_allow_short_uses_inline_byte_for_small_ints() {
        // `assembler.py:99-107` — with `allow_short`, a signed-i8 value is
        // emitted inline as one byte (argcode `c`) and never pooled.
        let mut asm = Assembler::new();
        let mut state = empty_state();
        assert_eq!(
            asm.emit_const_i_allow_short(5, true, &mut state),
            (5u8, 'c')
        );
        // signed-i8 boundaries: 127 and -128 both stay inline.
        assert_eq!(
            asm.emit_const_i_allow_short(127, true, &mut state),
            (127u8, 'c')
        );
        assert_eq!(
            asm.emit_const_i_allow_short(-128, true, &mut state),
            (128u8, 'c') // -128 as i8 as u8
        );
        assert!(
            state.constants_i.is_empty(),
            "the short form must not touch the int constant pool"
        );

        // Out of signed-i8 range → constant pool, argcode `i`. First pool
        // slot is `num_regs_i + 0`.
        let (byte, argcode) = asm.emit_const_i_allow_short(128, true, &mut state);
        assert_eq!(argcode, 'i');
        assert_eq!(byte, state.num_regs_i as u8);
        assert_eq!(state.constants_i, vec![128]);

        // `allow_short = false` (opname not in USE_C_FORM) pools even small
        // ints — second pool slot is `num_regs_i + 1`.
        let (byte, argcode) = asm.emit_const_i_allow_short(5, false, &mut state);
        assert_eq!(argcode, 'i');
        assert_eq!(byte, (state.num_regs_i + 1) as u8);
        assert_eq!(state.constants_i, vec![128, 5]);
    }

    #[test]
    fn encode_regorconst_source_threads_short_argcode() {
        use crate::flatten::{RegOrConst, Register};
        use crate::flowspace::model::{ConstValue, Constant};

        let mut asm = Assembler::new();
        let mut state = empty_state();
        // A small int Constant in a USE_C_FORM op (`allow_short = true`,
        // e.g. `int_return`/`int_copy`) takes the inline `c` argcode.
        let arg = RegOrConst::Const(Constant::new(ConstValue::Int(42)));
        assert_eq!(
            asm.encode_regorconst_source(&arg, RegKind::Int, true, &mut state, None),
            (42u8, 'c')
        );
        assert!(state.constants_i.is_empty());
        // The same Constant without `allow_short` pools to `i`.
        assert_eq!(
            asm.encode_regorconst_source(&arg, RegKind::Int, false, &mut state, None),
            (state.num_regs_i as u8, 'i')
        );
        assert_eq!(state.constants_i, vec![42]);
        // A register source always takes its kind argcode, never `c`.
        let reg = RegOrConst::Reg(Register::new(RegKind::Int, 3));
        assert_eq!(
            asm.encode_regorconst_source(&reg, RegKind::Int, true, &mut state, None),
            (3u8, 'i')
        );
    }

    #[test]
    fn get_opnum_setdefault_allocates_dynamic_bytes_for_unregistered_keys() {
        // RPython parity: `assembler.py:220 self.insns.setdefault(key,
        // len(self.insns))` allocates a fresh byte when the key has
        // not been seen before.  Pyre skips reserved canonical /
        // extension bytes so those mappings keep their compile-time-
        // stable bytes for the runtime walker, but translator-only
        // keys still get assigned a unique byte instead of panicking.
        let mut asm = Assembler::new();
        let expected_first = (0..=u8::MAX)
            .find(|byte| !crate::insns::is_reserved_opcode_byte(*byte))
            .expect("expected at least one non-reserved opcode byte");
        let first = asm.get_opnum("translator_only_unknown_key/254");
        assert_eq!(
            first, expected_first,
            "first unregistered key should land on the lowest \
             non-reserved byte"
        );
        assert!(!crate::insns::is_reserved_opcode_byte(first));
        let expected_second = ((first as u16 + 1)..=u8::MAX as u16)
            .map(|byte| byte as u8)
            .find(|byte| !crate::insns::is_reserved_opcode_byte(*byte))
            .expect("expected a second non-reserved opcode byte");
        let second = asm.get_opnum("translator_only_other_key/254");
        assert_eq!(second, expected_second);
        assert!(!crate::insns::is_reserved_opcode_byte(second));
        // Re-querying the same key returns the cached byte, matching
        // `setdefault`'s dict semantics.
        let first_again = asm.get_opnum("translator_only_unknown_key/254");
        assert_eq!(first_again, first);
        // Canonical keys keep their reserved bytes regardless of the
        // dynamic counter.
        let live = asm.get_opnum("live/");
        assert_eq!(
            live,
            crate::insns::insn_byte("live/"),
            "canonical keys must keep their reserved BC_* bytes",
        );

        // Typed opname variants that are registered as canonical bytes
        // must resolve to their reserved BC_* values through get_opnum.
        for (key, canonical) in [
            ("int_guard_value/i", "int_guard_value/i"),
            ("ref_guard_value/r", "ref_guard_value/r"),
            ("float_guard_value/f", "float_guard_value/f"),
        ] {
            let byte = asm.get_opnum(key);
            let expected = crate::insns::insn_byte(canonical);
            assert_eq!(
                byte, expected,
                "{key} must map to its canonical BC_* byte {expected}",
            );
            assert!(
                crate::insns::is_reserved_opcode_byte(byte),
                "{key} byte {byte} must be reserved",
            );
        }

        // Typed opnames not in the canonical table get dynamic bytes.
        for key in [
            "int_assert_green/i",
            "ref_assert_green/r",
            "float_assert_green/f",
            "int_isconstant/i",
            "ref_isconstant/r",
            "float_isconstant/f",
            "ref_isvirtual/r",
        ] {
            assert!(
                crate::insns::insn_byte_opt(key).is_none(),
                "{key} should not be in the canonical table",
            );
            let byte = asm.get_opnum(key);
            assert!(
                !crate::insns::is_reserved_opcode_byte(byte),
                "{key} should get a non-reserved dynamic byte, got {byte}",
            );
        }
    }

    #[test]
    fn emit_descr_reuses_rpython_descr_dict_index() {
        let mut asm = Assembler::new();

        let first = asm.emit_ready_descr(crate::jitcode::BhDescr::Field {
            offset: 0,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 0,
            parent: None,
            name: "value".into(),
            owner: "Cell".into(),
        });
        let repeated = asm.emit_ready_descr(crate::jitcode::BhDescr::Field {
            offset: 0,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 0,
            parent: None,
            name: "value".into(),
            owner: "Cell".into(),
        });
        let other = asm.emit_ready_descr(crate::jitcode::BhDescr::Field {
            offset: 0,
            field_size: 8,
            field_type: majit_ir::value::Type::Ref,
            field_flag: majit_ir::descr::ArrayFlag::Pointer,
            is_field_signed: false,
            is_immutable: false,
            is_quasi_immutable: false,
            index_in_parent: 1,
            parent: None,
            name: "mutate_value".into(),
            owner: "Cell".into(),
        });

        assert_eq!(first, repeated);
        assert_ne!(first, other);
        assert_eq!(asm.snapshot_descrs().len(), 2);
    }

    #[test]
    fn emit_call_descr_key_uses_full_rpython_calldescr_shape() {
        let mut asm = Assembler::new();
        let effect = majit_ir::descr::EffectInfo::MOST_GENERAL;

        let signed_int =
            crate::jitcode::BhCallDescr::from_arg_classes("i".to_string(), 'i', effect.clone());
        let signed_int_repeat =
            crate::jitcode::BhCallDescr::from_arg_classes("i".to_string(), 'i', effect.clone());
        let single_float =
            crate::jitcode::BhCallDescr::from_arg_classes("i".to_string(), 'S', effect.clone());
        let unsigned_int = crate::jitcode::BhCallDescr {
            arg_classes: "i".to_string(),
            result_type: 'i',
            result_signed: false,
            result_size: 8,
            result_erased: crate::jitcode::CallResultErasedKey::Unsigned,
            extra_info: effect.clone(),
        };
        let raw_address = crate::jitcode::BhCallDescr {
            arg_classes: "i".to_string(),
            result_type: 'i',
            result_signed: false,
            result_size: 8,
            result_erased: crate::jitcode::CallResultErasedKey::Address,
            extra_info: effect,
        };

        assert_eq!(single_float.result_type, 'S');
        assert_eq!(single_float.result_size, 4);
        assert!(!single_float.result_signed);
        assert_eq!(
            single_float.result_erased,
            crate::jitcode::CallResultErasedKey::SingleFloat,
        );

        let signed_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: signed_int,
        });
        let signed_repeat_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: signed_int_repeat,
        });
        let single_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: single_float,
        });
        let unsigned_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: unsigned_int,
        });
        let address_idx = asm.emit_ready_descr(crate::jitcode::BhDescr::Call {
            calldescr: raw_address,
        });

        assert_eq!(signed_idx, signed_repeat_idx);
        assert_ne!(signed_idx, single_idx);
        assert_ne!(signed_idx, unsigned_idx);
        assert_ne!(unsigned_idx, address_idx);
        assert_eq!(asm.snapshot_descrs().len(), 4);
    }

    #[test]
    fn jit_merge_point_and_loop_header_opnames() {
        // jtransform.py:1707 `op1 = SpaceOperation('jit_merge_point', args, None)`
        let merge = crate::model::OpKind::JitMergePoint {
            jitdriver_index: 0,
            greens_i: vec![],
            greens_r: vec![],
            greens_f: vec![],
            reds_i: vec![],
            reds_r: vec![],
            reds_f: vec![],
        };
        assert_eq!(op_kind_to_opname(&merge), "jit_merge_point");
        // jtransform.py:1718 `SpaceOperation('loop_header', [c_index], None)`
        let header = crate::model::OpKind::LoopHeader { jitdriver_index: 0 };
        assert_eq!(op_kind_to_opname(&header), "loop_header");
    }

    #[test]
    fn assemble_basic() {
        let mut flat = SSARepr {
            name: "test".into(),
            insns: vec![],
            num_blocks: 1,
            insns_pos: None,
        };

        let regallocs = empty_regallocs();
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        assert_eq!(flat.name, "test");
        assert_eq!(body.c_num_regs_i as usize, 0);
        assert_eq!(body.c_num_regs_r as usize, 0);
        assert_eq!(body.c_num_regs_f as usize, 0);
        assert_eq!(asm.count_jitcodes(), 1);
    }

    #[test]
    fn assemble_fused_goto_if_not_routes_to_reserved_opcode() {
        // gh #37: a `FlatOp::GotoIfNotOp` assembles through the
        // registered `goto_if_not_<op>/<argcodes>` jitcode key — the
        // per-arg register kinds form the argcodes, and the key must
        // resolve to the reserved opcode the metainterp blackhole
        // dispatches, NOT a freshly-minted dynamic byte (which would
        // have no consumer).
        use crate::flatten::Register;
        let regallocs = empty_regallocs();

        // Binary compare: `int_lt` with two int operands → `iiL`.
        let mut flat = SSARepr {
            name: "fused".into(),
            insns: vec![
                FlatOp::GotoIfNotOp {
                    opname: "int_lt".into(),
                    args: vec![
                        Register::new(RegKind::Int, 0),
                        Register::new(RegKind::Int, 1),
                    ],
                    target: Label(0),
                },
                FlatOp::Label(Label(0)),
            ],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        assert_eq!(
            asm.insns.get("goto_if_not_int_lt/iiL").copied(),
            Some(crate::insns::BC_GOTO_IF_NOT_INT_LT),
            "binary int compare must route to the reserved fused opcode",
        );

        // Unary test: `int_is_zero` with one int operand → `iL`.
        let mut flat = SSARepr {
            name: "fused_unary".into(),
            insns: vec![
                FlatOp::GotoIfNotOp {
                    opname: "int_is_zero".into(),
                    args: vec![Register::new(RegKind::Int, 0)],
                    target: Label(0),
                },
                FlatOp::Label(Label(0)),
            ],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        assert_eq!(
            asm.insns.get("goto_if_not_int_is_zero/iL").copied(),
            Some(crate::insns::BC_GOTO_IF_NOT_INT_IS_ZERO),
            "unary int test must route to the reserved fused opcode",
        );
    }

    #[test]
    fn assemble_ref_return_with_host_object_constant() {
        let module = HostObject::new_module("hello");
        let mut flat = SSARepr {
            name: "return_host_object".into(),
            insns: vec![FlatOp::RefReturn(crate::flatten::RegOrConst::Const(
                crate::flowspace::model::Constant::new(ConstValue::HostObject(module.clone())),
            ))],
            num_blocks: 1,
            insns_pos: None,
        };

        let regallocs = empty_regallocs();
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        assert_eq!(body.constants_r, vec![module.identity_id() as i64]);
        assert!(asm.insns.contains_key("ref_return/r"));
    }

    #[test]
    fn assemble_ref_return_with_prebuilt_string_constant() {
        use crate::translator::rtyper::lltypesystem::rstr::{
            const_str_cache_llstr, ll_strhash_value,
        };
        // A `RefReturn` of a prebuilt `Ptr(STR)` constant drives a real
        // `ConstValue::LLPtr(STR)` through the whole assembler emit path
        // (encode_regorconst_source -> emit_const('r') -> emit_const_r ->
        // emit_str_const_r) and commits the recorded descriptor into the
        // produced `JitCodeBody.str_consts`.  This is the build-side half of
        // the deferred-materialization loop the runtime `materialize_str_consts`
        // pass closes (the runtime half is covered in pyre-jit-trace).
        let mut flat = SSARepr {
            name: "return_prebuilt_string".into(),
            insns: vec![FlatOp::RefReturn(crate::flatten::RegOrConst::Const(
                crate::flowspace::model::Constant::new(ConstValue::LLPtr(Box::new(
                    const_str_cache_llstr(b"hi").expect("cache llstr"),
                ))),
            ))],
            num_blocks: 1,
            insns_pos: None,
        };

        let regallocs = empty_regallocs();
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        // The committed body carries the deferred descriptor, and the
        // `constants_r` slot it names holds the non-canonical sentinel the
        // runtime load pass overwrites with the live STR address.
        assert_eq!(body.str_consts.len(), 1);
        assert_eq!(body.str_consts[0].bytes, b"hi".to_vec());
        assert_eq!(body.str_consts[0].precomputed_hash, ll_strhash_value(b"hi"));
        let idx = body.str_consts[0].constants_r_index;
        assert_eq!(body.constants_r[idx], str_const_sentinel(0));
        assert!(asm.insns.contains_key("ref_return/r"));
    }

    #[test]
    fn emit_const_r_records_prebuilt_string_descriptor_and_dedups() {
        use crate::translator::rtyper::lltypesystem::rstr::{
            const_str_cache_llstr, ll_strhash_value,
        };
        let str_const =
            |s: &[u8]| ConstValue::LLPtr(Box::new(const_str_cache_llstr(s).expect("cache llstr")));
        let mut asm = Assembler::new();
        let mut state = empty_state();

        // A prebuilt `Ptr(STR)` no longer panics: it records a descriptor
        // and pools a sentinel slot for runtime materialization.
        let reg_a = asm.emit_const_r(&str_const(b"hello"), &mut state);
        assert_eq!(state.str_consts.len(), 1);
        assert_eq!(state.str_consts[0].bytes, b"hello".to_vec());
        assert_eq!(
            state.str_consts[0].precomputed_hash,
            ll_strhash_value(b"hello")
        );
        assert_eq!(state.str_consts[0].constants_r_index, 0);
        assert_eq!(state.constants_r, vec![str_const_sentinel(0)]);

        // The same literal dedups to the same descriptor + pool slot.
        let reg_a2 = asm.emit_const_r(&str_const(b"hello"), &mut state);
        assert_eq!(reg_a2, reg_a);
        assert_eq!(state.str_consts.len(), 1);
        assert_eq!(state.constants_r.len(), 1);

        // A distinct literal gets a distinct descriptor + sentinel slot.
        let reg_b = asm.emit_const_r(&str_const(b"world"), &mut state);
        assert_ne!(reg_b, reg_a);
        assert_eq!(state.str_consts.len(), 2);
        assert_eq!(state.str_consts[1].bytes, b"world".to_vec());
        assert_eq!(state.str_consts[1].constants_r_index, 1);
        assert_eq!(
            state.constants_r,
            vec![str_const_sentinel(0), str_const_sentinel(1)]
        );
    }

    #[test]
    fn assemble_with_registers() {
        use crate::model::{FunctionGraph, OpKind, ValueType};
        // Build graph for regalloc (regalloc operates on graph, not SSARepr)
        let mut graph = FunctionGraph::new("add");
        let entry = graph.startblock;
        let v0_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "a".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let v1_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: v0_var.clone(),
                    rhs: v0_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let v2_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "r".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v1_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&v0_var, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v1_var, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&v2_var, crate::model::ConcreteType::GcRef);

        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = SSARepr {
            name: "add".into(),
            insns: vec![],
            num_blocks: 1,
            insns_pos: None,
        };
        let mut asm = Assembler::new();
        let body = asm.assemble(&mut flat, &regallocs);

        // v0 dies when v1 is defined → they share a register → 1 int reg
        assert_eq!(body.c_num_regs_i as usize, 1);
        assert_eq!(body.c_num_regs_r as usize, 1);
        assert_eq!(body.c_num_regs_f as usize, 0);
    }

    /// `OpKind::RecordQuasiImmutField` must lower to a single opcode
    /// keyed `record_quasiimmut_field/rdd`, with the field+mutate
    /// FieldDescriptor pair pushed as two `BhDescr::Field` entries — see
    /// `rpython/jit/codewriter/jtransform.py:901-903` and
    /// `rpython/jit/metainterp/blackhole.py:1537-1539`.
    #[test]
    fn assembles_record_quasiimmut_field_with_two_descrs() {
        use crate::call::CallControl;
        use crate::flatten::flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, ImmutableRank, OpKind, ValueType};

        let mut cc = CallControl::new();
        cc.immutable_fields_by_struct.insert(
            "Cell".to_string(),
            vec![("value".to_string(), ImmutableRank::QuasiImmutable)],
        );

        let mut graph = FunctionGraph::new("read_cell");
        let base_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "cell".to_string(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("value", Some("Cell".to_string())),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(result_var.clone()));

        let config = GraphTransformConfig::default();
        let mut transformer = Transformer::new(&config).with_callcontrol(&mut cc);
        let mut rewritten = transformer.transform(&graph).graph;

        FunctionGraph::set_concretetype_of_inline(&base_var, crate::model::ConcreteType::GcRef);
        FunctionGraph::set_concretetype_of_inline(&result_var, crate::model::ConcreteType::Signed);
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);
        // `assemble` derives every operand kind from the variable
        // concretetype + the `regallocs` coloring (lookup_coloring_var,
        // flatten.py:386-387); pass the same canonical-exceptblock-
        // augmented graph and regallocs that the flatten pass consumed.
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
        let descrs = asm.snapshot_descrs();
        let field_descr_names: Vec<&str> = descrs
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

    #[test]
    fn assemble_typed_writes_use_canonical_non_v_opnames() {
        use crate::flatten::flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("typed_writes");
        let base_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "obj".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let index_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "i".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let value_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "v".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            graph.startblock,
            OpKind::FieldWrite {
                base: base_var.clone(),
                field: FieldDescriptor::new("x", Some("Point".into())),
                value: crate::model::LinkArg::Value(value_var.clone()),
                ty: ValueType::Unknown,
            },
            false,
        );
        graph.push_op_var(
            graph.startblock,
            OpKind::ArrayWrite {
                base: base_var.clone(),
                index: index_var.clone(),
                value: crate::model::LinkArg::Value(value_var.clone()),
                item_ty: ValueType::Unknown,
                array_type_id: None,
                nolength: false,
            },
            false,
        );
        graph.set_return(graph.startblock, None);

        // Publish kinds to graph cells before jtransform.  Variable
        // Rc-shares the concretetype cell across clones, so the
        // cloned rewritten graph picks up the same kinds.
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        FunctionGraph::set_concretetype_of_inline(
            &index_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &value_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let mut rewritten = Transformer::new(&config).transform(&graph).graph;
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        assert!(
            asm.insns.contains_key("setfield_gc_i/rid"),
            "expected canonical setfield_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("setarrayitem_gc_i/riid"),
            "expected canonical setarrayitem_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setfield_gc_v/rid"),
            "unexpected setfield_gc_v key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setfield_gc_v/iid"),
            "unexpected setfield_gc_v/iid key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setfield_gc_v/ird"),
            "unexpected setfield_gc_v/ird key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setarrayitem_gc_v/riid"),
            "unexpected setarrayitem_gc_v key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("setarrayitem_gc_v/iiid"),
            "unexpected setarrayitem_gc_v/iiid key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_setfield_const_int_uses_c_form() {
        use crate::flatten::flatten_graph;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, ValueType};

        // A `setfield_gc_i` whose value is an inline integer Constant takes
        // the short `c` byte (`setfield_gc_i/rcd`) when it fits a signed
        // byte, else a pool `i` slot (`setfield_gc_i/rid`).  `setfield_gc`
        // ∈ USE_C_FORM, `assembler.py:99-107,312`.
        let build = |value: i64| -> Assembler {
            let mut graph = FunctionGraph::new("const_setfield");
            let base_var = graph
                .push_op_var(
                    graph.startblock,
                    OpKind::Input {
                        name: "obj".into(),
                        ty: ValueType::Ref(None),
                        class_root: None,
                    },
                    true,
                )
                .unwrap();
            graph.push_op_var(
                graph.startblock,
                OpKind::FieldWrite {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    value: LinkArg::Const(Constant::new(ConstValue::Int(value))),
                    ty: ValueType::Unknown,
                },
                false,
            );
            graph.set_return(graph.startblock, None);
            FunctionGraph::set_concretetype_of_inline(
                &base_var,
                crate::codewriter::type_state::ConcreteType::GcRef,
            );
            let config = GraphTransformConfig::default();
            let mut rewritten = Transformer::new(&config).transform(&graph).graph;
            regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
            let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
            let mut flat = flatten_graph(&rewritten, &mut regallocs);
            let mut asm = Assembler::new();
            let _ = asm.assemble(&mut flat, &regallocs);
            asm
        };

        let small = build(7);
        assert!(
            small.insns.contains_key("setfield_gc_i/rcd"),
            "small const setfield should take the c-form, got {:?}",
            small.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !small.insns.contains_key("setfield_gc_i/rid"),
            "small const setfield should not materialise an i-register, got {:?}",
            small.insns.keys().collect::<Vec<_>>()
        );

        let wide = build(100_000);
        assert!(
            wide.insns.contains_key("setfield_gc_i/rid"),
            "wide const setfield should take the pool i-form, got {:?}",
            wide.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !wide.insns.contains_key("setfield_gc_i/rcd"),
            "wide const setfield should not take the c-form, got {:?}",
            wide.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_setfield_vable_const_int_uses_pool_form() {
        use crate::flatten::flatten_graph;
        use crate::flowspace::model::{ConstValue, Constant};
        use crate::jtransform::{GraphTransformConfig, Transformer, VirtualizableFieldDescriptor};
        use crate::model::{FieldDescriptor, FunctionGraph, LinkArg, OpKind, ValueType};

        // A `setfield_gc_i` whose field is virtualizable rewrites to
        // `setfield_vable_i` (`jtransform.py:921-927`).  `setfield_vable_i`
        // is NOT in USE_C_FORM (`assembler.py:312-345`), so an inline
        // integer Constant value always takes the pool `i` slot
        // (`setfield_vable_i/rid`) — never the short `c` byte — even when
        // it fits a signed byte.
        let build = |value: i64| -> Assembler {
            let mut graph = FunctionGraph::new("const_setfield_vable");
            let base_var = graph
                .push_op_var(
                    graph.startblock,
                    OpKind::Input {
                        name: "frame".into(),
                        ty: ValueType::Ref(None),
                        class_root: None,
                    },
                    true,
                )
                .unwrap();
            graph.push_op_var(
                graph.startblock,
                OpKind::FieldWrite {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("depth", Some("Frame".into())),
                    value: LinkArg::Const(Constant::new(ConstValue::Int(value))),
                    ty: ValueType::Int,
                },
                false,
            );
            graph.set_return(graph.startblock, None);
            FunctionGraph::set_concretetype_of_inline(
                &base_var,
                crate::codewriter::type_state::ConcreteType::GcRef,
            );
            let config = GraphTransformConfig {
                vable_fields: vec![VirtualizableFieldDescriptor::new(
                    "depth",
                    Some("Frame".into()),
                    0,
                )],
                ..Default::default()
            };
            let mut rewritten = Transformer::new(&config).transform(&graph).graph;
            regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
            let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
            let mut flat = flatten_graph(&rewritten, &mut regallocs);
            let mut asm = Assembler::new();
            let _ = asm.assemble(&mut flat, &regallocs);
            asm
        };

        for value in [7, 100_000] {
            let asm = build(value);
            assert!(
                asm.insns.contains_key("setfield_vable_i/rid"),
                "vable const setfield (value={value}) should take the pool i-form, got {:?}",
                asm.insns.keys().collect::<Vec<_>>()
            );
            assert!(
                !asm.insns.contains_key("setfield_vable_i/rcd"),
                "vable const setfield (value={value}) must never take the c-form, got {:?}",
                asm.insns.keys().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn assemble_typed_reads_use_canonical_non_v_opnames() {
        use crate::flatten::flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FieldDescriptor, FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("typed_reads");
        let base_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "obj".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let index_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "i".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let field_result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::FieldRead {
                    base: base_var.clone(),
                    field: FieldDescriptor::new("x", Some("Point".into())),
                    ty: ValueType::Unknown,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let array_result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayRead {
                    base: base_var.clone(),
                    index: index_var.clone(),
                    item_ty: ValueType::Unknown,
                    array_type_id: None,
                    nolength: false,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(array_result_var.clone()));

        // Publish kinds to graph cells before jtransform.  Variable
        // Rc-shares the concretetype cell across clones, so the
        // cloned rewritten graph picks up the same kinds.
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        FunctionGraph::set_concretetype_of_inline(
            &index_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &field_result_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &array_result_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let mut rewritten = Transformer::new(&config).transform(&graph).graph;
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        assert!(
            asm.insns.contains_key("getfield_gc_i/rd>i"),
            "expected canonical getfield_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            asm.insns.contains_key("getarrayitem_gc_i/rid>i"),
            "expected canonical getarrayitem_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getfield_gc_v/rd>i"),
            "unexpected getfield_gc_v/rd>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getfield_gc_v/id>i"),
            "unexpected getfield_gc_v/id>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_v/rid>i"),
            "unexpected getarrayitem_gc_v/rid>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_v/iid>i"),
            "unexpected getarrayitem_gc_v/iid>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_v/ird>i"),
            "unexpected getarrayitem_gc_v/ird>i key: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    /// A `pure: true` ArrayRead assembles to the foldable
    /// `getarrayitem_gc_i_pure` opcode (rlist.py:724
    /// `ll_getitem_foldable_nonneg`); a `pure: false` ArrayRead keeps the
    /// non-pure `getarrayitem_gc_i`.  The `rid>i` argcodes are byte-identical
    /// either way (`blackhole.py:1339-1341` aliases the pure handler) — only
    /// the opname/byte differs.
    fn assemble_arrayread_pure(pure: bool) -> Assembler {
        use crate::flatten::flatten_graph;
        use crate::jtransform::{GraphTransformConfig, Transformer};
        use crate::model::{FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("arrayread_pure");
        let base_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "block".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let index_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "i".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let array_result_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::ArrayRead {
                    base: base_var.clone(),
                    index: index_var.clone(),
                    item_ty: ValueType::Int,
                    array_type_id: None,
                    nolength: false,
                    pure,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(array_result_var.clone()));
        // `assemble` derives every operand kind from the variable
        // concretetype + regallocs coloring — color the base (GcRef array
        // pointer) and index (Signed) inputs as well as the result, else
        // `lookup_coloring` finds no class for them.
        FunctionGraph::set_concretetype_of_inline(
            &base_var,
            crate::codewriter::type_state::ConcreteType::GcRef,
        );
        FunctionGraph::set_concretetype_of_inline(
            &index_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );
        FunctionGraph::set_concretetype_of_inline(
            &array_result_var,
            crate::codewriter::type_state::ConcreteType::Signed,
        );

        let config = GraphTransformConfig::default();
        let mut rewritten = Transformer::new(&config).transform(&graph).graph;
        regalloc::augment_canonical_exceptblock_on_graph(&mut rewritten);
        let mut regallocs = regalloc::perform_all_register_allocations(&rewritten);
        let mut flat = flatten_graph(&rewritten, &mut regallocs);

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
        asm
    }

    #[test]
    fn assemble_pure_arrayread_uses_pure_opcode() {
        let asm = assemble_arrayread_pure(true);
        assert!(
            asm.insns.contains_key("getarrayitem_gc_i_pure/rid>i"),
            "expected foldable getarrayitem_gc_i_pure key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_i/rid>i"),
            "unexpected non-pure getarrayitem_gc_i key for pure ArrayRead: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_non_pure_arrayread_uses_non_pure_opcode() {
        let asm = assemble_arrayread_pure(false);
        assert!(
            asm.insns.contains_key("getarrayitem_gc_i/rid>i"),
            "expected non-pure getarrayitem_gc_i key, got {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
        assert!(
            !asm.insns.contains_key("getarrayitem_gc_i_pure/rid>i"),
            "unexpected pure getarrayitem_gc_i_pure key for non-pure ArrayRead: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_skips_input_opnames_after_flatten() {
        use crate::flatten::flatten_graph;
        use crate::model::{FunctionGraph, OpKind, ValueType};

        let mut graph = FunctionGraph::new("input_free_bytecode");
        let lhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "lhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let rhs_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::Input {
                    name: "rhs".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let sum_var = graph
            .push_op_var(
                graph.startblock,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: lhs_var.clone(),
                    rhs: rhs_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(graph.startblock, Some(sum_var.clone()));

        FunctionGraph::set_concretetype_of_inline(&lhs_var, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&rhs_var, crate::model::ConcreteType::Signed);
        FunctionGraph::set_concretetype_of_inline(&sum_var, crate::model::ConcreteType::Signed);

        regalloc::augment_canonical_exceptblock_on_graph(&mut graph);
        let mut regallocs = regalloc::perform_all_register_allocations(&graph);
        let mut flat = flatten_graph(&graph, &mut regallocs);
        assert!(
            !flat.insns.iter().any(|op| matches!(
                op,
                FlatOp::Op(crate::model::SpaceOperation {
                    kind: OpKind::Input { .. },
                    ..
                })
            )),
            "flatten unexpectedly left input ops: {:?}",
            flat.insns
        );

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);

        assert!(
            !asm.insns.keys().any(|key| key.starts_with("input_")),
            "unexpected input opcode keys: {:?}",
            asm.insns.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    #[should_panic(expected = "OpKind::Input must be eliminated before assembly")]
    fn assemble_rejects_input_ops() {
        let mut graph = crate::model::FunctionGraph::new("bad_input");
        let vid_var = graph.alloc_value_var();
        let mut flat = SSARepr {
            name: "bad_input".into(),
            insns: vec![FlatOp::Op(crate::model::SpaceOperation {
                result: Some(vid_var),
                kind: crate::model::OpKind::Input {
                    name: "x".into(),
                    ty: crate::model::ValueType::Int,
                    class_root: None,
                },
            })],
            num_blocks: 1,
            insns_pos: None,
        };
        // Empty regallocs — the test panics before any coloring lookup
        // runs (the assembler rejects OpKind::Input outright).
        let regallocs = HashMap::new();

        let mut asm = Assembler::new();
        let _ = asm.assemble(&mut flat, &regallocs);
    }

    /// `rpython/jit/codewriter/jtransform.py:611` —
    /// `<kind>_guard_value` family opname mapping:
    ///   * `'i'` → `int_guard_value`
    ///   * `'r'` → `ref_guard_value`
    ///   * `'f'` → `float_guard_value`
    ///
    /// Pyre does not exercise the `str_guard_value` mapping
    /// (`jtransform.py:631 promote_string` / `:647 promote_unicode`)
    /// because the `PromoteString` / `PromoteUnicode` rewrite arms
    /// panic before emitting — pyre-object has no `rstr.STR` /
    /// `rstr.UNICODE` GC layout (`rpython/rtyper/lltypesystem/
    /// rstr.py:1226-1246`).
    #[test]
    fn op_kind_to_opname_routes_guard_value_kind_chars() {
        use crate::flowspace::model::Variable;
        use crate::model::OpKind;
        let value_var = Variable::new();
        let opnames = ['i', 'r', 'f'].map(|kc| {
            op_kind_to_opname(&OpKind::GuardValue {
                value: value_var.clone(),
                kind_char: kc,
            })
        });
        assert_eq!(opnames[0], "int_guard_value");
        assert_eq!(opnames[1], "ref_guard_value");
        assert_eq!(opnames[2], "float_guard_value");
    }
}
