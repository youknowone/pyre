/// Python bytecode → JitCode compiler.
///
/// RPython: rpython/jit/codewriter/codewriter.py
///
/// RPython's CodeWriter transforms flow graphs → JitCode via a 4-step pipeline:
///   1. jtransform  — rewrite operations
///   2. regalloc    — assign registers
///   3. flatten     — CFG → linear SSARepr
///   4. assemble    — SSARepr → JitCode bytecode
///
/// For pyre, Python bytecodes are already linearized and register-allocated
/// (fast locals = registers, value stack = runtime stack). Steps 1-3 collapse
/// into a single bytecode-to-bytecode translation.
///
/// The Assembler (majit JitCodeBuilder) is RPython's assembler.py equivalent.
use std::cell::{OnceCell, RefCell, UnsafeCell};
use std::collections::HashMap;

use majit_ir::OpCode;
use majit_metainterp::jitcode::{JitCode, JitCodeBuilder, LivenessInfo};

use super::assembler::Assembler;
use super::ssa_emitter::SSAReprEmitter;
use pyre_interpreter::bytecode::{CodeObject, Instruction, OpArgState};
use pyre_interpreter::runtime_ops::{binary_op_tag, compare_op_tag};

use super::flatten::{Insn, Kind, ListOfKind, Operand, Register, SSARepr, TLabel};

// ---------------------------------------------------------------------------
// RPython: codewriter/flatten.py KINDS = ['int', 'ref', 'float']
// ---------------------------------------------------------------------------

/// Python `var_num` → flat index into the `locals_cells_stack_w`
/// virtualizable array.
///
/// PyFrame lays out locals, cells, and the value stack in a single
/// vector; `var_num` from `LOAD_FAST`/`STORE_FAST` is already a direct
/// offset into that vector (no indirection).
///
/// jtransform.py:1877 do_fixed_list_getitem / :1898 do_fixed_list_setitem
/// rewrite portal local accesses into `getarrayitem_vable_r` /
/// `setarrayitem_vable_r` against the virtualizable array.
#[inline]
fn local_to_vable_slot(var_num: usize) -> usize {
    var_num
}

/// Pyre-only metadata attached to a Python CodeObject's compiled JitCode.
///
/// RPython does not need these fields because its bytecode PCs are already
/// JitCode PCs. Pyre translates CPython bytecode to JitCode lazily, so the
/// translation maps live here instead of polluting the canonical JitCode.
pub struct PyJitCodeMetadata {
    /// py_pc → jitcode byte offset. Named for RPython's `frame.pc →
    /// jitcode position` flow; `set_majit_jitcode` clones this into
    /// `state::JitCode.py_to_jit_pc` so the `pyjitpl` side can look up
    /// without touching the pyre-jit codewriter crate.
    pub pc_map: Vec<usize>,
    /// Value-stack depth at each Python PC, in slots above stack_base.
    pub depth_at_py_pc: Vec<u16>,
    /// Register allocated for the portal's frame red argument.
    pub portal_frame_reg: u16,
    /// Register allocated for the portal's execution-context red argument.
    pub portal_ec_reg: u16,
    /// Absolute start index of the operand stack in PyFrame.locals_cells_stack_w.
    pub stack_base: usize,
    /// Pyre-local decoded liveness view used by `resume_in_blackhole`'s
    /// per-section register fill. The canonical packed bytes live on
    /// `MetaInterpStaticData.liveness_info` and are read via inline
    /// `-live-` offsets embedded in `JitCode.code`.
    pub liveness: Vec<LivenessInfo>,
}

/// Compiled JitCode plus pyre-only metadata.
pub struct PyJitCode {
    pub jitcode: std::sync::Arc<JitCode>,
    pub metadata: PyJitCodeMetadata,
    /// True if the jitcode contains BC_ABORT opcodes (unsupported bytecodes).
    /// Precomputed at compile time to avoid repeated bytecode scanning.
    pub has_abort: bool,
    /// Python PC of the jit_merge_point opcode (trace entry header).
    pub merge_point_pc: Option<usize>,
}

#[derive(Clone, Copy)]
struct ExceptionCatchSite {
    landing_label: u16,
    handler_py_pc: usize,
    stack_depth: u16,
    push_lasti: bool,
    lasti_py_pc: usize,
}

/// rpython/jit/codewriter/liveness.py:139 "we encode the bitsets of which of
/// the 256 registers are live" — PyPy's compact liveness encoding treats a
/// register index as a single byte and assumes the precondition holds. PyPy
/// raises `AssertionError` (i.e. crashes the translator) if a JitCode ever
/// emits a register index ≥ 256, so the orthodox behavior is "panic".
///
/// **Source-equivalence divergence:** the Rust port returns `None` instead
/// of panicking, and the caller marks the JitCode with `has_abort = true`
/// so the driver falls back to the interpreter at runtime
/// (`pyre/pyre-jit/src/call_jit.rs:1819`). This is a deliberate safety net
/// — pyre's regalloc can blow past 256 ref registers when stdlib functions
/// load many locals + a deep value stack, and the orthodox panic would
/// kill the entire process. The proper fix is to make pyre's regalloc
/// produce ≤ 256 registers (e.g. by reusing dead slots), at which point
/// this fallback can be replaced with a hard panic to match PyPy.
///
/// Until then, the divergence is documented here and in
/// `pyre_unittest_landings_2026_04_09.md` so that any future
/// source-equivalence audit can find it without re-discovering the gap.
fn liveness_regs_to_u8_sorted(regs: &[u16]) -> Option<Vec<u8>> {
    let mut bytes = Vec::with_capacity(regs.len());
    for &reg in regs {
        let reg = u8::try_from(reg).ok()?;
        bytes.push(reg);
    }
    bytes.sort_unstable();
    bytes.dedup();
    Some(bytes)
}

impl PyJitCode {
    /// Check if this jitcode has BC_ABORT opcodes.
    pub fn has_abort_opcode(&self) -> bool {
        self.has_abort
    }

    /// Empty `PyJitCode` slot inserted by `CallControl::get_jitcode`
    /// (call.py:168 `jitcode = JitCode(graph.name, fnaddr, calldescr, ...)`).
    ///
    /// In RPython the `JitCode` constructor returns a fresh object whose
    /// `code` / `descrs` / `liveness` arrays are all empty until
    /// `assembler.assemble(...)` populates them later in
    /// `make_jitcodes`'s drain loop (codewriter.py:80).  The skeleton
    /// gives the dict an entry with a stable identity so re-entrant
    /// `get_jitcode` calls (or pyre's `merge_point_pc` refinement
    /// shortcut) can find an existing key without recompiling.
    ///
    /// Until the drain replaces the slot, the only field with meaningful
    /// content is `merge_point_pc` (the refinement hint passed in by
    /// `get_jitcode`).
    pub fn skeleton(merge_point_pc: Option<usize>) -> Self {
        Self {
            jitcode: std::sync::Arc::new(JitCode::default()),
            metadata: PyJitCodeMetadata {
                pc_map: Vec::new(),
                depth_at_py_pc: Vec::new(),
                portal_frame_reg: 0,
                portal_ec_reg: 0,
                stack_base: 0,
                liveness: Vec::new(),
            },
            has_abort: false,
            merge_point_pc,
        }
    }
}

// ---------------------------------------------------------------------------
// RPython: codewriter/codewriter.py — class CodeWriter
// ---------------------------------------------------------------------------

/// Compiles Python CodeObjects into JitCode for blackhole execution.
///
/// RPython: `rpython/jit/codewriter/codewriter.py::CodeWriter`.
/// `codewriter.py:20-23` stores `self.assembler = Assembler()` and
/// `self.callcontrol = CallControl(cpu, jitdrivers_sd)` once on the
/// CodeWriter and reuses them across every `transform_graph_to_jitcode`
/// call so `all_liveness` / `num_liveness_ops` and the `jitcodes` dict
/// accumulate over the whole translator session.
///
/// pyre mirrors that ownership via a per-thread singleton: the process
/// holds a single `CodeWriter` instance (one per thread) reachable via
/// [`CodeWriter::instance`], matching `warmspot.py:245`
/// `codewriter = CodeWriter(cpu, [jd])`. The owned `Assembler` lives on
/// a `RefCell<Assembler>` field so `transform_graph_to_jitcode` can
/// still mutate it under the immutable-by-default singleton borrow.
pub struct CodeWriter {
    /// `codewriter.py:22` `self.assembler = Assembler()`.
    ///
    /// Single Assembler instance shared across every `transform_graph_to_jitcode`
    /// call on this CodeWriter. `all_liveness` / `all_liveness_positions` /
    /// `num_liveness_ops` accumulate here just like the upstream object.
    assembler: RefCell<Assembler>,
    /// RPython: `self.callcontrol = CallControl(cpu, jitdrivers_sd)`
    /// (codewriter.py:23). Owned in a `UnsafeCell` so `&CodeWriter` can
    /// mint `&mut CallControl` through [`Self::callcontrol`] — matches
    /// the legacy `JITCODE_CACHE` interior-mutability contract.
    callcontrol: UnsafeCell<super::call::CallControl>,
}

impl CodeWriter {
    /// RPython: `CodeWriter.__init__(cpu, jitdrivers_sd)` (codewriter.py:20-23).
    ///
    /// Phase A: the cpu helpers are fixed module-level functions in
    /// `crate::call_jit`; Phase D.2 wired `callcontrol` as a field so
    /// `writer.callcontrol()` matches `self.callcontrol` in upstream.
    pub fn new() -> Self {
        // codewriter.py:21-23 `self.cpu = cpu; self.assembler = Assembler();
        //   self.callcontrol = CallControl(cpu, jitdrivers_sd)`.
        // pyre owns the single `Cpu` on `CallControl`; `CodeWriter::cpu()`
        // returns a borrow back out so the upstream attribute access
        // pattern (`self.cpu`) still works.
        let cpu = super::cpu::Cpu::new();
        Self {
            assembler: RefCell::new(Assembler::new()),
            callcontrol: UnsafeCell::new(super::call::CallControl::new(cpu, Vec::new())),
        }
    }

    /// `codewriter.py:21` `self.cpu = cpu`.
    ///
    /// Convenience accessor — pyre owns the single `Cpu` on
    /// `CallControl` (call.py:27 `self.cpu = cpu`); upstream both
    /// attributes point at the same object.
    pub fn cpu(&self) -> &super::cpu::Cpu {
        &self.callcontrol().cpu
    }

    /// RPython: `CodeWriter.setup_vrefinfo(self, vrefinfo)`
    /// (codewriter.py:91-94).
    ///
    /// ```python
    /// def setup_vrefinfo(self, vrefinfo):
    ///     # must be called at most once
    ///     assert self.callcontrol.virtualref_info is None
    ///     self.callcontrol.virtualref_info = vrefinfo
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: pyre has no `virtualref` machinery
    /// (no `@jit.virtual_ref`, no `vref_info` lookup); the slot is
    /// preserved so future warmspot wiring can call through with the
    /// same name.
    pub fn setup_vrefinfo(&self, vrefinfo: ()) {
        // codewriter.py:93 `assert self.callcontrol.virtualref_info is None`.
        assert!(self.callcontrol().virtualref_info.is_none());
        // codewriter.py:94 `self.callcontrol.virtualref_info = vrefinfo`.
        self.callcontrol().virtualref_info = Some(vrefinfo);
    }

    /// RPython: `CodeWriter.setup_jitdriver(self, jitdriver_sd)`
    /// (codewriter.py:96-99).
    ///
    /// ```python
    /// def setup_jitdriver(self, jitdriver_sd):
    ///     # Must be called once per jitdriver.
    ///     self.callcontrol.jitdrivers_sd.append(jitdriver_sd)
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: pyre has no `JitDriverStaticData` —
    /// the portal is implicit (any CodeObject containing
    /// `JUMP_BACKWARD`). The slot accepts `()` so future warmspot
    /// wiring can call through line-by-line.
    pub fn setup_jitdriver(&self, jitdriver_sd: ()) {
        // codewriter.py:99 `self.callcontrol.jitdrivers_sd.append(jitdriver_sd)`.
        self.callcontrol().jitdrivers_sd.push(jitdriver_sd);
    }

    /// RPython: `self.callcontrol` (codewriter.py:23).
    ///
    /// Returns a mutable reference to the owned `CallControl`. Safe under
    /// the same invariant as the legacy `JITCODE_CACHE` thread_local: the
    /// caller must not re-enter `callcontrol()` while the returned borrow
    /// is live.
    #[allow(clippy::mut_from_ref)]
    pub fn callcontrol(&self) -> &mut super::call::CallControl {
        // SAFETY: `CodeWriter` is only accessed via `instance()` which
        // returns a thread-local reference; all callers execute on the
        // owning thread.
        unsafe { &mut *self.callcontrol.get() }
    }

    /// Access the process-wide single `CodeWriter` — analog of the
    /// single `codewriter` owned by `warmspot.py:245-281` for the
    /// lifetime of the JIT.
    ///
    /// Implemented as a per-thread singleton: pyre's JIT currently runs
    /// one interpreter per thread and function pointers in `Self` are
    /// `Sync`, so a thread-local provides the RPython "one CodeWriter
    /// per warmspot" invariant without a global lock.
    pub fn instance() -> &'static CodeWriter {
        thread_local! {
            static INSTANCE: CodeWriter = CodeWriter::new();
        }
        INSTANCE.with(|cw| unsafe { &*(cw as *const CodeWriter) })
    }

    /// Transform a Python CodeObject into a JitCode.
    ///
    /// RPython: CodeWriter.transform_graph_to_jitcode(graph, jitcode, verbose, index)
    ///
    /// Python bytecodes serve as the "graph". Since they are already linear
    /// and register-allocated, jtransform/regalloc/flatten are identity
    /// transforms. We go directly to assembly.
    pub fn transform_graph_to_jitcode(
        &self,
        code: &CodeObject,
        w_code: *const (),
        merge_point_pc: Option<usize>,
    ) -> PyJitCode {
        let nlocals = code.varnames.len();
        // pyframe.py:111 `self.valuestackdepth = code.co_nlocals + ncellvars
        // + nfreevars`. pyre's `PyFrame.locals_cells_stack_w` uses the same
        // layout: [locals(nlocals), cells(ncells), stack(grows upward)].
        // `stack_base_absolute` is the array index where the operand stack
        // begins — the baseline for `frame.valuestackdepth`.
        let ncells = pyre_interpreter::pyframe::ncells(code);
        let stack_base_absolute = nlocals + ncells;
        // regalloc.py parity: all values (locals + stack temporaries) live in
        // typed register files. The value stack is mapped to ref registers
        // starting at `stack_base`.
        let max_stackdepth = code.max_stackdepth.max(1) as usize;
        let stack_base = nlocals as u16;
        let obj_tmp0 = (nlocals + max_stackdepth) as u16;
        let obj_tmp1 = (nlocals + max_stackdepth + 1) as u16;
        let arg_regs_start = (nlocals + max_stackdepth + 2) as u16;
        let null_ref_reg = (nlocals + max_stackdepth + 10) as u16;
        // interp_jit.py:64 parity: dedicated ref registers for portal reds.
        // reds = ['frame', 'ec'] → two registers beyond null_ref_reg.
        let portal_frame_reg = null_ref_reg + 1;
        let portal_ec_reg = null_ref_reg + 2;

        let int_tmp0 = 0u16;
        let int_tmp1 = 1u16;
        let op_code_reg = 2u16;
        // jtransform.py: virtualizable field indices for getfield_vable_*
        // interp_jit.py:25-31 / virtualizable_spec.rs parity:
        //   0=last_instr, 1=code, 2=vsd, 3=debugdata, 4=lastblock, 5=namespace
        const VABLE_CODE_FIELD_IDX: u16 = 1;
        const VABLE_VALUESTACKDEPTH_FIELD_IDX: u16 = 2;
        const VABLE_NAMESPACE_FIELD_IDX: u16 = 5;

        // regalloc.py: compile-time stack depth counter — tracks which
        // stack register (stack_base + depth) is the current TOS.
        let mut current_depth: u16 = 0;

        // RPython: self.assembler = Assembler() + JitCode(graph.name, ...)
        // (rpython/jit/codewriter/jitcode.py:14-15 takes name as the first
        // __init__ argument; majit's JitCodeBuilder::set_name mirrors that).
        let mut assembler = SSAReprEmitter::new();
        assembler.set_name(code.obj_name.to_string());
        // B6 Phase 3b scaffolding: grow an `SSARepr` alongside the direct
        // `JitCodeBuilder` calls. Currently only a handful of handlers
        // (`ref_return` below) dual-emit an `Insn::Op`; the remaining
        // bytecode handlers still route through the builder only. When
        // every handler has been converted, `ssarepr` becomes the
        // authoritative input to `jit::assembler::Assembler::assemble`
        // (Phase 3c switchover) and the direct builder calls disappear.
        // See `pyre/pyre-jit/src/jit/B6_CODEWRITER_PIPELINE_PLAN.md`.
        let mut ssarepr = SSARepr::new(code.obj_name.to_string());

        // RPython regalloc.py: keep kind-separated register files.
        assembler.ensure_r_regs(portal_ec_reg + 1);
        assembler.ensure_i_regs(op_code_reg + 1);

        // Register helper function pointers
        // RPython: CallControl manages fn addresses; assembler.finished()
        // writes them into callinfocollection.
        let cpu = self.cpu();
        let call_fn_idx = assembler.add_fn_ptr(cpu.call_fn as *const ());
        let load_global_fn_idx = assembler.add_fn_ptr(cpu.load_global_fn as *const ());
        let compare_fn_idx = assembler.add_fn_ptr(cpu.compare_fn as *const ());
        let binary_op_fn_idx = assembler.add_fn_ptr(cpu.binary_op_fn as *const ());
        let box_int_fn_idx = assembler.add_fn_ptr(cpu.box_int_fn as *const ());
        let truth_fn_idx = assembler.add_fn_ptr(cpu.truth_fn as *const ());
        let load_const_fn_idx = assembler.add_fn_ptr(cpu.load_const_fn as *const ());
        let store_subscr_fn_idx = assembler.add_fn_ptr(cpu.store_subscr_fn as *const ());
        let build_list_fn_idx = assembler.add_fn_ptr(cpu.build_list_fn as *const ());
        // Per-arity call helpers (appended AFTER existing fn_ptrs to preserve indices).
        let call_fn_0_idx = assembler.add_fn_ptr(cpu.call_fn_0 as *const ());
        let call_fn_2_idx = assembler.add_fn_ptr(cpu.call_fn_2 as *const ());
        let call_fn_3_idx = assembler.add_fn_ptr(cpu.call_fn_3 as *const ());
        let call_fn_4_idx = assembler.add_fn_ptr(cpu.call_fn_4 as *const ());
        let call_fn_5_idx = assembler.add_fn_ptr(cpu.call_fn_5 as *const ());
        let call_fn_6_idx = assembler.add_fn_ptr(cpu.call_fn_6 as *const ());
        let call_fn_7_idx = assembler.add_fn_ptr(cpu.call_fn_7 as *const ());
        let call_fn_8_idx = assembler.add_fn_ptr(cpu.call_fn_8 as *const ());

        // RPython flatten.py: pre-create labels for each block.
        // Python bytecodes are linear, so each instruction index gets a label.
        let num_instrs = code.instructions.len();
        let mut labels: Vec<u16> = Vec::with_capacity(num_instrs);
        for _ in 0..num_instrs {
            labels.push(assembler.new_label());
        }

        let exception_entries =
            pyre_interpreter::bytecode::decode_exception_table(&code.exceptiontable);
        let mut catch_for_pc: Vec<Option<u16>> = vec![None; num_instrs];
        let mut catch_sites: Vec<ExceptionCatchSite> = Vec::new();
        for py_pc in 0..num_instrs {
            let Some(entry) = exception_entries
                .iter()
                .find(|entry| py_pc >= entry.start as usize && py_pc < entry.end as usize)
            else {
                continue;
            };
            let handler_py_pc = entry.target as usize;
            if handler_py_pc >= num_instrs {
                continue;
            }
            let landing_label = assembler.new_label();
            catch_for_pc[py_pc] = Some(landing_label);
            catch_sites.push(ExceptionCatchSite {
                landing_label,
                handler_py_pc,
                stack_depth: entry.depth,
                push_lasti: entry.push_lasti,
                lasti_py_pc: py_pc,
            });
        }

        // interp_jit.py:118 `pypyjitdriver.can_enter_jit(...)` is called in
        // `jump_absolute` (`jumpto < next_instr` branch), i.e. at each
        // Python backward jump.  jtransform.py:1714-1723
        // `handle_jit_marker__can_enter_jit = handle_jit_marker__loop_header`
        // lowers each one to a `loop_header` jitcode op.  Pyre has no
        // `jump_absolute` Python wrapper — the equivalent is to pre-scan
        // `JumpBackward` opcodes and record their targets; each target PC
        // becomes a `loop_header` site.
        let loop_header_pcs = find_loop_header_pcs(code);

        // Exception table: handler target PC → handler stack depth.
        // Python sets the stack depth to handler.depth (+1 if push_lasti)
        // at exception handler entry. Used to correct current_depth at
        // non-linear control flow points.
        let handler_depth_at: std::collections::HashMap<usize, u16> = {
            exception_entries
                .iter()
                .map(|e| {
                    let extra = if e.push_lasti { 1u16 } else { 0 };
                    (e.target as usize, e.depth as u16 + extra + 1)
                })
                .collect()
        };

        // pc_map: Python PC → JitCode byte offset
        let mut pc_map = vec![0usize; num_instrs];

        // RPython: flatten_graph() walks blocks and emits instruction tuples.
        // RPython: assembler.assemble(ssarepr, jitcode, num_regs) emits bytecodes.
        // For pyre, we combine both steps: walk Python bytecodes and emit
        // JitCode bytecodes directly.
        let mut arg_state = OpArgState::default();
        // liveness.py parity: record stack depth at each Python PC for
        // precise liveness generation. Stack registers stack_base..stack_base+depth
        // are live at each PC.
        let mut depth_at_pc: Vec<u16> = vec![0; num_instrs];
        let mut live_patches: Vec<(usize, usize)> = Vec::with_capacity(num_instrs);

        // jitcode.py:18: jitdriver_sd is not None for portals.
        // call.py:148: jd.mainjitcode.jitdriver_sd = jd
        // interp_jit.py:78: dispatch() has jit_merge_point — is the portal.
        // pyre: every eval_loop_jit call acts as a portal (each Python
        // function has its own dispatch loop), so all jitcodes with loop
        // headers are portals.
        let is_portal = !loop_header_pcs.is_empty();
        // RPython parity: every backward jump goes through dispatch() →
        // jit_merge_point(). The blackhole's bhimpl_jit_merge_point raises
        // ContinueRunningNormally at the bottommost level. Ideally all
        // loop headers should emit BC_JIT_MERGE_POINT, but the
        // CRN→interpreter→JIT-reentry cycle crashes in JIT compiled code
        // because blackhole-modified frame locals can contain values
        // incompatible with the compiled trace's assumptions. Until the
        // full RPython CRN→portal_ptr restart is implemented, only the
        // first loop header is a merge point.
        // RPython jtransform.py:1690: jit_merge_point only in the portal
        // graph. merge_point_pc is the trace entry PC (from bound_reached).
        // Other loop headers use loop_header (no-op in the blackhole).
        let merge_point_pc = merge_point_pc.or_else(|| loop_header_pcs.iter().copied().min());

        // pyframe.py:379-417 pushvalue/popvalue_maybe_none parity:
        // Each push/pop writes self.valuestackdepth = depth ± 1.
        // jtransform.py:923-928 lowers this to setfield_vable_i.
        // This macro emits the equivalent BC_SETFIELD_VABLE_I after
        // every current_depth mutation so the frame's valuestackdepth
        // stays in sync at every guard/call point — matching RPython's
        // per-push/per-pop semantics.
        macro_rules! emit_vsd {
            ($asm:expr, $depth:expr) => {
                if is_portal {
                    emit_load_const_i!(
                        ssarepr,
                        $asm,
                        int_tmp0,
                        (stack_base_absolute + $depth as usize) as i64
                    );
                    emit_vable_setfield_int!(
                        ssarepr,
                        $asm,
                        VABLE_VALUESTACKDEPTH_FIELD_IDX,
                        int_tmp0
                    );
                }
            };
        }

        // PRE-EXISTING-ADAPTATION: the `BC_ABORT_PERMANENT` runtime
        // bytecode does not appear in `rpython/jit/codewriter/` or
        // `rpython/jit/metainterp/`. RPython refuses to build jitcode for
        // bytecodes it cannot translate (the translator surfaces the
        // failure at build time); pyre must always produce runnable
        // jitcode because bytecode translation is lazy at runtime. We
        // therefore keep the runtime-side adaptation (assembler emits
        // `BC_ABORT_PERMANENT` so the blackhole interpreter falls back to
        // CPython evaluation) but never surface the pyre-only opname into
        // the RPython-parity SSARepr layer — `flatten.py:106` uses plain
        // `Label` for loop headers and `assembler.py:159` does not encode
        // unsupported bytecodes as named opnames.

        // B6 Phase 3b dual emission for `last_exc_value`. RPython parity:
        // `flatten.py:347` `self.emitline("last_exc_value", "->",
        // self.getcolor(w))` — `assembler.py:220` turns it into
        // `last_exc_value/>r`. pyre emits this once per catch site to
        // load the thread-local exception into the handler's input
        // register.
        macro_rules! emit_last_exc_value {
            ($ssarepr:expr, $asm:expr, $dst:expr) => {{
                let dst = $dst;
                $ssarepr.insns.push(Insn::op_with_result(
                    "last_exc_value",
                    Vec::new(),
                    Register::new(Kind::Ref, dst),
                ));
                $asm.last_exc_value(dst);
            }};
        }

        // PRE-EXISTING-ADAPTATION: the `BC_JUMP_TARGET` runtime opcode
        // does not appear in `rpython/jit/codewriter/`. RPython marks
        // loop-header block entries with a plain `Insn::Label` and lets
        // the blackhole's dispatch loop recognise them via the label
        // position; pyre emits a dedicated `BC_JUMP_TARGET` opcode so the
        // runtime inner-loop can cheaply identify back-edge targets
        // without consulting a label table. The runtime-side adaptation
        // stays (assembler dispatch at `assembler.rs:367-372`) but the
        // pyre-only opname is not surfaced into the RPython-parity
        // SSARepr layer — `flatten.py:106` uses plain `Label` for loop
        // headers.

        // B6 Phase 3b dual emission for `int_copy` / `ref_copy` /
        // `float_copy` with a Constant source. RPython parity:
        // `flatten.py:333` `self.emitline('%s_copy' % kind, v, "->", w)`
        // — `v` is resolved via `getcolor(v)` which returns either a
        // `Register` or an unchanged `Constant` (see `flatten.py:382-384`).
        // The `assembler.py:140-222` dispatch handles both: the Register
        // source emits an `int_copy/ii` entry, and the Constant source
        // emits an `int_copy/ci` entry (argcode `'c'` for a compact
        // Constant). pyre's legacy `load_const_{i,r,f}_value` emits the
        // same runtime bytes under pyre-only `load_const_*` opnames; the
        // SSARepr now carries the RPython-parity `*_copy` name with a
        // ConstInt/ConstRef/ConstFloat source operand.
        // `int_eq` is NOT dual-emitted by itself.
        //
        // pyre's only emitter of `OpCode::IntEq` via `record_binop_i` is
        // the `PopJumpIfTrue` branch-folding chain below — which
        // `rpython/jit/codewriter/jtransform.py:1212` rewrites as
        // `int_is_zero(x)` and `flatten.py:247` further specialises into
        // `goto_if_not_int_is_zero`. Emitting a standalone `int_eq` at
        // the PopJumpIfTrue site would replicate pyre's builder-ABI
        // sequence into the SSARepr and lock in a pyre-only shape.
        // Keep the handler on the direct builder path until the
        // assembler learns the RPython specialisation.

        // Call family intentionally has NO dual-emit.
        //
        // `rpython/jit/codewriter/jtransform.py:414-435` `rewrite_call()`
        // emits `residual_call_{kinds}_{reskind}` with
        // `[fnptr_constant, ListOfKind(int)?, ListOfKind(ref),
        //   ListOfKind(float)?, calldescr]`. pyre's runtime ABI uses
        // a caller-order Register list plus a u16 helper-table index and
        // encodes `may_force` in the target bytecode — none of which
        // fit into RPython's SSA tuple shape. Reviewer guidance:
        //   "codewriter는 원본 RPython tuple 을 만들고, pyre 적응은
        //    assembler 가 해야 한다."
        // Rather than baking the pyre shape into the SSA (which would
        // ossify a pyre-only SSA vocabulary), we keep the call handlers
        // on the direct builder path until assembler.rs grows exact
        // `residual_call_*` dispatch that can reconstruct the pyre
        // caller-order list. See `B6_CODEWRITER_PIPELINE_PLAN.md`.

        macro_rules! emit_load_const_i {
            ($ssarepr:expr, $asm:expr, $dst:expr, $value:expr $(,)?) => {{
                let dst = $dst;
                let value: i64 = $value;
                $ssarepr.insns.push(Insn::op_with_result(
                    "int_copy",
                    vec![Operand::ConstInt(value)],
                    Register::new(Kind::Int, dst),
                ));
                $asm.load_const_i_value(dst, value);
            }};
        }

        // B6 Phase 3b dual emission for `ref_return`. Every site that used
        // to invoke `assembler.ref_return(src)` now also appends an
        // `Insn::Op { opname: "ref_return", args: [Register(Ref, src)] }`
        // to the SSARepr so the future `Assembler::assemble` path
        // (`assembler.rs::dispatch_op:374`) can reproduce the same byte at
        // the Phase 3c switchover. The direct builder call stays until the
        // switchover runs so the emitted JitCode remains bit-identical.
        // RPython parity: `rpython/jit/codewriter/jtransform.py` emits
        // `op_ref_return(v)` via `rewrite_op_jit_return` for the portal
        // return path; `assembler.py:221` turns that into the `ref_return/r`
        // bytecode key.
        macro_rules! emit_ref_return {
            ($ssarepr:expr, $asm:expr, $src:expr) => {{
                let src = $src;
                $ssarepr
                    .insns
                    .push(Insn::op("ref_return", vec![Operand::reg(Kind::Ref, src)]));
                $asm.ref_return(src);
            }};
        }

        // B6 Phase 3b dual emission for `goto`. RPython parity:
        // `flatten.py:161` `self.emitline('goto', TLabel(link.target))` —
        // `assembler.py:220` turns the op into `goto/L`. Pyre labels are
        // integer indices into `labels[]`, one per Python PC; the
        // `TLabel` carries the synthetic name `pc{target_py_pc}` so the
        // Phase 3c dispatch (`assembler.rs::dispatch_op:345`) can resolve
        // it against `builder_label`.
        macro_rules! emit_goto {
            ($ssarepr:expr, $asm:expr, $labels:expr, $target_py_pc:expr) => {{
                let target_py_pc = $target_py_pc;
                $ssarepr.insns.push(Insn::op(
                    "goto",
                    vec![Operand::TLabel(TLabel::new(format!("pc{}", target_py_pc)))],
                ));
                $asm.jump($labels[target_py_pc]);
            }};
        }

        // B6 Phase 3b dual emission for `raise`. RPython parity:
        // `flatten.py` emits `self.emitline("raise", self.getcolor(args[1]))`
        // inside the exception-link handler; `assembler.py:220` turns it
        // into `raise/r`. pyre's single `emit_raise(exc_reg)` call site
        // (RAISE_VARARGS with argc >= 1) corresponds to the same edge.
        macro_rules! emit_raise {
            ($ssarepr:expr, $asm:expr, $src:expr) => {{
                let src = $src;
                $ssarepr
                    .insns
                    .push(Insn::op("raise", vec![Operand::reg(Kind::Ref, src)]));
                $asm.emit_raise(src);
            }};
        }

        // B6 Phase 3b dual emission for `reraise`. RPython parity:
        // `flatten.py` emits the zero-arg `self.emitline("reraise")` for
        // the re-raise edge; `assembler.py:220` turns it into
        // `reraise/`. pyre emits this for RAISE_VARARGS with argc == 0.
        macro_rules! emit_reraise {
            ($ssarepr:expr, $asm:expr) => {{
                $ssarepr.insns.push(Insn::op("reraise", Vec::new()));
                $asm.emit_reraise();
            }};
        }

        // B6 Phase 3b dual emission for `catch_exception`. RPython parity:
        // `flatten.py` emits `self.emitline('catch_exception',
        // TLabel(block.exits[0]))` when a block has an exception edge;
        // `assembler.py:220` turns it into `catch_exception/L`. pyre
        // emits this after each Python PC that has an exception handler.
        // The catch landing block lives after the main loop
        // (`mark_label(site.landing_label)`), so the `TLabel` carries
        // `catch_landing_{landing_label}` — distinct from the
        // `pc{py_pc}` naming used for PC-indexed labels.
        macro_rules! emit_catch_exception {
            ($ssarepr:expr, $asm:expr, $catch_label:expr) => {{
                let catch_label = $catch_label;
                $ssarepr.insns.push(Insn::op(
                    "catch_exception",
                    vec![Operand::TLabel(TLabel::new(format!(
                        "catch_landing_{}",
                        catch_label
                    )))],
                ));
                $asm.catch_exception(catch_label);
            }};
        }

        // B6 Phase 3b dual emission for block `Label`. RPython parity:
        // `flatten.py:180` `self.emitline(Label(block))` marks block
        // entry; `assembler.py:157-158` records the label position in
        // `self.label_positions`. pyre marks a label at every Python PC
        // (`mark_label(labels[py_pc])`) and at each catch landing
        // block's entry. The two naming schemes (`pc{py_pc}` vs
        // `catch_landing_{u16}`) match the TLabel schemes used by
        // `emit_goto!` and `emit_catch_exception!`.
        macro_rules! emit_mark_label_pc {
            ($ssarepr:expr, $asm:expr, $labels:expr, $py_pc:expr) => {{
                let py_pc = $py_pc;
                $ssarepr
                    .insns
                    .push(Insn::Label(super::flatten::Label::new(format!(
                        "pc{}",
                        py_pc
                    ))));
                $asm.mark_label($labels[py_pc]);
            }};
        }
        macro_rules! emit_mark_label_catch_landing {
            ($ssarepr:expr, $asm:expr, $landing_label:expr) => {{
                let landing_label = $landing_label;
                $ssarepr
                    .insns
                    .push(Insn::Label(super::flatten::Label::new(format!(
                        "catch_landing_{}",
                        landing_label
                    ))));
                $asm.mark_label(landing_label);
            }};
        }

        // B6 Phase 3b dual emission for `-live-`. RPython parity:
        // `flatten.py` inserts `self.emitline('-live-')` at every block
        // entry and at every point with a live resume set; `assembler.py:146-158`
        // allocates an `all_liveness` slot and encodes the live-register
        // triple. pyre patches the live-offset after the assembler
        // computes liveness for the Python PC, so the SSARepr Insn::Live
        // is empty until Phase 4's compute_liveness replaces it.
        macro_rules! emit_live_placeholder {
            ($ssarepr:expr, $asm:expr) => {{
                $ssarepr.insns.push(Insn::Live(Vec::new()));
                $asm.live_placeholder()
            }};
        }

        // `goto_if_not` is NOT dual-emitted either.
        //
        // `rpython/jit/codewriter/jtransform.py:196` and
        // `rpython/jit/codewriter/flatten.py:247` emit the boolean
        // exitswitch specialisation `goto_if_not_<opname>` (e.g.
        // `goto_if_not_int_is_zero`), not a generic `goto_if_not` over
        // an intermediate boolean register. pyre's bytecode walker
        // currently produces the generic `branch_reg_zero` pattern —
        // the same shape the deferred `PopJumpIfTrue/False` SSA emit
        // would target. Emitting generic `goto_if_not` into the SSARepr
        // would again commit to pyre's non-parity shape. Keep the
        // handlers on the direct `branch_reg_zero` builder call.

        // B6 Phase 3b dual emission for the `*_vable_*` field/array
        // accessors. RPython parity: `jtransform.py:844`
        // `SpaceOperation('getfield_vable_%s' % kind, ...)` and
        // `jtransform.py:923` `SpaceOperation('setfield_vable_%s' % kind,
        // ...)` use the FULL kind name (`int` / `ref` / `float`);
        // `jtransform.py` `SpaceOperation('getarrayitem_vable_%s' %
        // kind[0], ...)` / `SpaceOperation('setarrayitem_vable_%s' %
        // kind[0], ...)` use the SHORT form (`i` / `r` / `f`).
        //
        // The runtime dispatch arms in `assembler.rs:486-524` were
        // already written against short-form field names; the SSA
        // emission here uses the RPython-parity full-form names and
        // relies on the Phase 3c dispatch aliases (added alongside)
        // to route the full-form opnames to the same builder methods.
        macro_rules! emit_vable_getfield_ref {
            ($ssarepr:expr, $asm:expr, $dst:expr, $field_idx:expr) => {{
                let dst = $dst;
                let field_idx = $field_idx;
                $ssarepr.insns.push(Insn::op_with_result(
                    "getfield_vable_ref",
                    vec![Operand::ConstInt(field_idx as i64)],
                    Register::new(Kind::Ref, dst),
                ));
                $asm.vable_getfield_ref(dst, field_idx);
            }};
        }
        macro_rules! emit_vable_setfield_int {
            ($ssarepr:expr, $asm:expr, $field_idx:expr, $src:expr) => {{
                let field_idx = $field_idx;
                let src = $src;
                $ssarepr.insns.push(Insn::op(
                    "setfield_vable_int",
                    vec![
                        Operand::ConstInt(field_idx as i64),
                        Operand::reg(Kind::Int, src),
                    ],
                ));
                $asm.vable_setfield_int(field_idx, src);
            }};
        }
        macro_rules! emit_vable_getarrayitem_ref {
            ($ssarepr:expr, $asm:expr, $dst:expr, $field_idx:expr, $index:expr) => {{
                let dst = $dst;
                let field_idx = $field_idx;
                let index = $index;
                $ssarepr.insns.push(Insn::op_with_result(
                    "getarrayitem_vable_r",
                    vec![
                        Operand::ConstInt(field_idx as i64),
                        Operand::reg(Kind::Int, index),
                    ],
                    Register::new(Kind::Ref, dst),
                ));
                $asm.vable_getarrayitem_ref(dst, field_idx, index);
            }};
        }
        macro_rules! emit_vable_setarrayitem_ref {
            ($ssarepr:expr, $asm:expr, $field_idx:expr, $index:expr, $src:expr) => {{
                let field_idx = $field_idx;
                let index = $index;
                let src = $src;
                $ssarepr.insns.push(Insn::op(
                    "setarrayitem_vable_r",
                    vec![
                        Operand::ConstInt(field_idx as i64),
                        Operand::reg(Kind::Int, index),
                        Operand::reg(Kind::Ref, src),
                    ],
                ));
                $asm.vable_setarrayitem_ref(field_idx, index, src);
            }};
        }

        // B6 Phase 3b dual emission for `ref_copy`. RPython parity:
        // `flatten.py:334` `self.emitline('%s_copy' % kind, v, "->", w)`
        // emits the register-to-register move as `ref_copy` when
        // `kind == 'ref'`; `assembler.py:220` turns it into `ref_copy/r>r`.
        // pyre's `move_r(dst, src)` is the single ref-move primitive.
        // The SSARepr arg list follows the RPython `(src, '->', dst)`
        // shape via `op_with_result`.
        macro_rules! emit_move_r {
            ($ssarepr:expr, $asm:expr, $dst:expr, $src:expr) => {{
                let dst = $dst;
                let src = $src;
                $ssarepr.insns.push(Insn::op_with_result(
                    "ref_copy",
                    vec![Operand::reg(Kind::Ref, src)],
                    Register::new(Kind::Ref, dst),
                ));
                $asm.move_r(dst, src);
            }};
        }

        // B6 Phase 3b dual emission for `jit_merge_point`. RPython parity:
        // `jtransform.py:rewrite_op_jit_merge_point` emits
        // `SpaceOperation('jit_merge_point', args, None)` with
        // ListOfKind-tagged greens/reds; `assembler.py:220` turns it into
        // `jit_merge_point/IRR` (int-list, ref-list, ref-list). pyre
        // emits this at the portal loop header.
        macro_rules! emit_jit_merge_point {
            ($ssarepr:expr, $asm:expr, $greens_i:expr, $greens_r:expr, $reds_r:expr) => {{
                let greens_i: &[u8] = $greens_i;
                let greens_r: &[u8] = $greens_r;
                let reds_r: &[u8] = $reds_r;
                let to_list = |kind: Kind, regs: &[u8]| {
                    ListOfKind::new(
                        kind,
                        regs.iter()
                            .map(|&r| Operand::Register(Register::new(kind, r as u16)))
                            .collect(),
                    )
                };
                $ssarepr.insns.push(Insn::op(
                    "jit_merge_point",
                    vec![
                        Operand::ListOfKind(to_list(Kind::Int, greens_i)),
                        Operand::ListOfKind(to_list(Kind::Ref, greens_r)),
                        Operand::ListOfKind(to_list(Kind::Ref, reds_r)),
                    ],
                ));
                $asm.jit_merge_point(greens_i, greens_r, reds_r);
            }};
        }

        for py_pc in 0..num_instrs {
            // Exception handler entry: Python resets stack depth to the
            // handler's specified depth. Correct current_depth to match.
            if let Some(&handler_depth) = handler_depth_at.get(&py_pc) {
                current_depth = handler_depth;
            }
            // RPython flatten.py: Label(block) at block entry
            emit_mark_label_pc!(ssarepr, assembler, labels, py_pc);
            pc_map[py_pc] = assembler.current_pos();
            let live_patch = emit_live_placeholder!(ssarepr, assembler);
            live_patches.push((py_pc, live_patch));
            depth_at_pc[py_pc] = current_depth;

            if loop_header_pcs.contains(&py_pc) {
                if merge_point_pc == Some(py_pc) {
                    // interp_jit.py:64 portal contract:
                    //   greens = ['next_instr', 'is_being_profiled', 'pycode']
                    //   reds = ['frame', 'ec']
                    assembler.emit_portal_jit_merge_point(
                        py_pc,
                        w_code as i64,
                        portal_frame_reg,
                        portal_ec_reg,
                    );
                } else {
                    // pyre has a single jitdriver (PyPyJitDriver), index 0.
                    assembler.loop_header(0);
                }
            }

            let code_unit = code.instructions[py_pc];
            let (instruction, op_arg) = arg_state.get(code_unit);

            // pyframe.py:379-417 pushvalue/popvalue_maybe_none parity:
            // RPython's push/pop each write `self.valuestackdepth = depth +/- 1`.
            // On the JIT, these map to per-push `setfield_vable_i`. pyre's
            // codewriter stores stack values in typed registers rather than
            // the `locals_cells_stack_w` array, so we cannot emit a vable
            // setitem for each push. As the coarsest RPython-compatible
            // approximation we flush `valuestackdepth` once at opcode entry,
            // reflecting the pre-opcode stack depth — which is what the
            // interpreter (eval.rs:92 `target_depth = frame.nlocals() +
            // frame.ncells() + entry.depth`) uses when an exception handler
            // unwinds the frame.
            //
            // RPython interp_jit.py keeps `next_instr` as a green portal
            // argument and updates `last_instr` in the interpreter loop; it
            // does not lower a per-bytecode virtualizable write here. pyre's
            // portal entry / guard-resume paths already restore
            // `frame.next_instr`, and the interpreter updates `last_instr`
            // once execution returns there. Emitting `py_pc + 1` here only
            // grows the int constant pool linearly with function size and
            // trips assembler.py's 256-entry cap.
            // pyframe.py:379-417: valuestackdepth is written per-push/per-pop
            // via setfield_vable_i (jtransform.py:923-928), NOT once at opcode
            // entry. The per-push/per-pop emit_vsd! calls below mirror that.
            // (The old single-entry flush is removed.)

            // RPython jtransform.py: rewrite_operation() dispatches per opname.
            // Each match arm is the pyre equivalent of rewrite_op_*.
            match instruction {
                Instruction::Resume { .. }
                | Instruction::Nop
                | Instruction::Cache
                | Instruction::NotTaken
                | Instruction::ExtendedArg => {
                    // RPython: no-op operations produce no jitcode output
                }

                // jtransform.py:1877 do_fixed_list_getitem vable case:
                // portal locals are virtualizable array items — emit
                // vable_getarrayitem_ref so the optimizer folds the read
                // against virtualizable_boxes and the blackhole pulls the
                // live frame value into stack_base+current_depth on
                // resume. Non-portal frames keep move_r (no virtualizable
                // in scope).
                Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    if is_portal {
                        emit_load_const_i!(
                            ssarepr,
                            assembler,
                            int_tmp0,
                            local_to_vable_slot(reg as usize) as i64
                        );
                        emit_vable_getarrayitem_ref!(
                            ssarepr,
                            assembler,
                            stack_base + current_depth,
                            0_u16,
                            int_tmp0
                        );
                    } else {
                        emit_move_r!(ssarepr, assembler, stack_base + current_depth, reg);
                    }
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // jtransform.py:1898 do_fixed_list_setitem vable case:
                // Shadow write: keep move_r AND write to vable array.
                // Pure vable write (RPython parity) blocked on the same
                // virtualizable_boxes propagation gap as LoadFast above.
                Instruction::StoreFast { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, reg, stack_base + current_depth);
                    // Shadow write to vable array for consume_vable_info.
                    if is_portal {
                        emit_load_const_i!(
                            ssarepr,
                            assembler,
                            int_tmp0,
                            local_to_vable_slot(reg as usize) as i64
                        );
                        emit_vable_setarrayitem_ref!(ssarepr, assembler, 0_u16, int_tmp0, reg);
                    }
                }

                Instruction::LoadSmallInt { i } => {
                    let val = i.get(op_arg) as u32 as i64;
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, val);
                    assembler.call_ref_typed(
                        box_int_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                        obj_tmp0,
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                Instruction::LoadConst { consti } => {
                    let idx = consti.get(op_arg).as_usize();
                    // jtransform.py: getfield_vable_r for pycode (field 1)
                    emit_vable_getfield_ref!(ssarepr, assembler, obj_tmp0, VABLE_CODE_FIELD_IDX);
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, idx as i64);
                    assembler.call_ref_typed(
                        load_const_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        obj_tmp0,
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // Superinstruction: two consecutive LoadFast / LoadFastBorrow.
                // Plain LoadFast (above) is safely lifted to vable_getarrayitem_ref;
                // the paired superinstruction is kept on move_r until the
                // snapshot-captured-Ref-with-Int-value issue seen on
                // nbody/fannkuch (memory: superinstruction Phase 5 crash)
                // is diagnosed and fixed.
                Instruction::LoadFastBorrowLoadFastBorrow { var_nums }
                | Instruction::LoadFastLoadFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let reg_a = u32::from(pair.idx_1()) as u16;
                    let reg_b = u32::from(pair.idx_2()) as u16;
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, reg_a);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, reg_b);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // Super-instruction STORE_FAST; LOAD_FAST: pop TOS into
                // idx_1 (store), then push idx_2 (load). Net depth 0.
                // Both halves use register-bank moves (not vable read/write)
                // pending A-3 Layer 2 (blackhole frame writeback); the
                // codewriter side still has a real implementation so the
                // trace doesn't abort on this super-instruction.
                Instruction::StoreFastLoadFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let store_reg = u32::from(pair.idx_1()) as u16;
                    let load_reg = u32::from(pair.idx_2()) as u16;
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, store_reg, stack_base + current_depth);
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, load_reg);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // STORE_SUBSCR: stack [value, obj, key] → obj[key] = value
                Instruction::StoreSubscr => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp1, stack_base + current_depth); // key
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth); // obj
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(
                        ssarepr,
                        assembler,
                        arg_regs_start,
                        stack_base + current_depth
                    ); // value
                    assembler.call_may_force_void_typed_args(
                        store_subscr_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::reference(arg_regs_start),
                        ],
                    );
                }

                Instruction::PopTop => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    // regalloc.py: discard = just decrement depth, no bytecode
                }

                Instruction::PushNull => {
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, null_ref_reg);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // jtransform.py: rewrite_op_int_add etc.
                Instruction::BinaryOp { op } => {
                    let op_val = binary_op_tag(op.get(op_arg))
                        .expect("unsupported binary op tag in jitcode lowering")
                        as u32;
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp1, stack_base + current_depth); // rhs
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth); // lhs
                    emit_load_const_i!(ssarepr, assembler, op_code_reg, op_val as i64);
                    assembler.call_may_force_ref_typed(
                        binary_op_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::int(op_code_reg),
                        ],
                        obj_tmp0,
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // jtransform.py: rewrite_op_int_lt, optimize_goto_if_not
                Instruction::CompareOp { opname } => {
                    let op_val = compare_op_tag(opname.get(op_arg)) as u32;
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp1, stack_base + current_depth); // rhs
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth); // lhs
                    emit_load_const_i!(ssarepr, assembler, op_code_reg, op_val as i64);
                    assembler.call_may_force_ref_typed(
                        compare_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::int(op_code_reg),
                        ],
                        obj_tmp0,
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // jtransform.py: optimize_goto_if_not → goto_if_not
                // PopJumpIfFalse / PopJumpIfTrue intentionally do NOT push
                // SSARepr entries for the `truth_fn + int_eq 0 + goto_if_not`
                // chain. `rpython/jit/codewriter/jtransform.py:1212`
                // rewrites `int_eq(x, 0)` to `int_is_zero(x)` (resp.
                // `int_is_true`), and `jtransform.py:196` +
                // `flatten.py:247` emit the boolean-exitswitch specialisation
                // `goto_if_not_int_is_zero` instead of a generic
                // `goto_if_not`. pyre currently has neither
                // `int_is_zero/int_is_true` nor `goto_if_not_<opname>` in
                // its builder vocabulary, so emitting the pyre-shaped
                // sequence into the SSARepr would lock in a pyre-only shape.
                // The direct builder calls below stay; when the builder
                // grows exact RPython counterparts, this block becomes
                // eligible for dual-emit again.
                Instruction::PopJumpIfFalse { delta } => {
                    let target_py_pc = jump_target_forward(
                        code,
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth);
                    assembler.call_int_typed(
                        truth_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0)],
                        int_tmp0,
                    );
                    if target_py_pc < num_instrs {
                        assembler.branch_reg_zero(int_tmp0, labels[target_py_pc]);
                    }
                }

                Instruction::PopJumpIfTrue { delta } => {
                    let target_py_pc = jump_target_forward(
                        code,
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth);
                    assembler.call_int_typed(
                        truth_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0)],
                        int_tmp0,
                    );
                    assembler.load_const_i_value(int_tmp1, 0);
                    assembler.record_binop_i(int_tmp0, majit_ir::OpCode::IntEq, int_tmp0, int_tmp1);
                    if target_py_pc < num_instrs {
                        assembler.branch_reg_zero(int_tmp0, labels[target_py_pc]);
                    }
                }

                // RPython flatten.py: goto Label
                Instruction::JumpForward { delta } => {
                    let target_py_pc = jump_target_forward(
                        code,
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    if target_py_pc < num_instrs {
                        emit_goto!(ssarepr, assembler, labels, target_py_pc);
                    }
                }

                instr @ Instruction::JumpBackward { .. } => {
                    if let Some(target_py_pc) = backward_jump_target(code, py_pc, instr, op_arg) {
                        if target_py_pc < num_instrs {
                            assembler.jump(labels[target_py_pc]);
                        }
                    }
                }

                // flatten.py: int_return / ref_return
                Instruction::ReturnValue => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth);
                    emit_ref_return!(ssarepr, assembler, obj_tmp0);
                }

                // RPython jtransform.py: rewrite_op_direct_call (residual)
                Instruction::LoadGlobal { namei } => {
                    let raw_namei = namei.get(op_arg) as usize as i64;
                    // jtransform.py: getfield_vable_r for w_globals (field 3)
                    // and pycode (field 1) — namespace for lookup, code for names.
                    emit_vable_getfield_ref!(
                        ssarepr,
                        assembler,
                        obj_tmp0,
                        VABLE_NAMESPACE_FIELD_IDX
                    );
                    emit_vable_getfield_ref!(ssarepr, assembler, obj_tmp1, VABLE_CODE_FIELD_IDX);
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, raw_namei);
                    assembler.call_may_force_ref_typed(
                        load_global_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        obj_tmp0,
                    );
                    // LOAD_GLOBAL with (namei >> 1) & 1: push NULL first
                    if raw_namei & 1 != 0 {
                        emit_move_r!(ssarepr, assembler, stack_base + current_depth, null_ref_reg);
                        current_depth += 1;
                        emit_vsd!(assembler, current_depth);
                    }
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // RPython jtransform.py: rewrite_op_direct_call →
                // call_may_force / residual_call
                //
                // RPython blackhole.py: bhimpl_recursive_call_i calls
                // portal_runner directly, bypassing JIT entry.
                // Here we pop args and callable from the stack into
                // registers, then call the helper with explicit args.
                //
                // Stack layout before CALL(argc):
                //   [NULL, callable, arg0, arg1, ..., arg(argc-1)]
                // We pop in reverse: args first, then callable, then NULL.
                Instruction::Call { argc } => {
                    let nargs = argc.get(op_arg) as usize;
                    for i in (0..nargs).rev() {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                        emit_move_r!(
                            ssarepr,
                            assembler,
                            arg_regs_start + i as u16,
                            stack_base + current_depth
                        );
                    }
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp1, stack_base + current_depth); // callable
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth); // NULL (discard)

                    // RPython: bhimpl_recursive_call_i(jdindex, greens, reds)
                    // call_fn(callable, arg0, ...) → result
                    // Parent frame accessed via BH_VABLE_PTR thread-local.
                    let mut call_args =
                        vec![majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1)];
                    for i in 0..nargs {
                        call_args.push(majit_metainterp::jitcode::JitCallArg::reference(
                            arg_regs_start + i as u16,
                        ));
                    }
                    // Select the correct arity-specific call helper.
                    // RPython blackhole.py: call_int_function transmutes
                    // to the correct arity. Each nargs needs a matching
                    // extern "C" fn with that many i64 parameters.
                    // nargs > 8 → abort_permanent (no matching helper).
                    if nargs > 8 {
                        assembler.abort_permanent();
                    } else {
                        let fn_idx = match nargs {
                            0 => call_fn_0_idx,
                            1 => call_fn_idx,
                            2 => call_fn_2_idx,
                            3 => call_fn_3_idx,
                            4 => call_fn_4_idx,
                            5 => call_fn_5_idx,
                            6 => call_fn_6_idx,
                            7 => call_fn_7_idx,
                            _ => call_fn_8_idx,
                        };
                        assembler.call_may_force_ref_typed(fn_idx, &call_args, obj_tmp0);
                    }
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // Python 3.13: ToBool converts TOS to bool before branch.
                // No-op in JitCode: the value is already truthy/falsy and
                // the following PopJumpIfFalse guards on it.
                Instruction::ToBool => {}

                // RPython bhimpl_int_neg: -obj via binary_op(0, obj, NB_SUBTRACT)
                Instruction::UnaryNegative => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth);
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, 0);
                    assembler.call_may_force_ref_typed(
                        box_int_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                        obj_tmp1,
                    );
                    emit_load_const_i!(
                        ssarepr,
                        assembler,
                        int_tmp0,
                        binary_op_tag(pyre_interpreter::bytecode::BinaryOperator::Subtract)
                            .expect("subtract must have a jit binary-op tag"),
                    );
                    assembler.call_may_force_ref_typed(
                        binary_op_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        obj_tmp0,
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // JumpBackwardNoInterrupt reuses `backward_jump_target`:
                // the encoding differs from JumpBackward (no skip_caches
                // on the next-PC base) but the helper routes each variant
                // to its correct arithmetic so pre-scan and emit stay in
                // lockstep.  interp_jit.py:103 + jtransform.py:1714.
                instr @ Instruction::JumpBackwardNoInterrupt { .. } => {
                    if let Some(target_py_pc) = backward_jump_target(code, py_pc, instr, op_arg) {
                        if target_py_pc < num_instrs {
                            assembler.jump(labels[target_py_pc]);
                        }
                    }
                }

                // RPython bhimpl_newlist: build list from N items on stack.
                Instruction::BuildList { count } => {
                    let argc = count.get(op_arg) as usize;
                    for i in (0..argc.min(2)).rev() {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                        emit_move_r!(
                            ssarepr,
                            assembler,
                            arg_regs_start + i as u16,
                            stack_base + current_depth
                        );
                    }
                    // Discard extra items beyond 2 (helper supports 0-2).
                    for _ in 2..argc {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                    }
                    // build_list_fn(argc, item0, item1) → list
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, argc as i64);
                    let item0 = if argc >= 1 {
                        majit_metainterp::jitcode::JitCallArg::reference(arg_regs_start)
                    } else {
                        majit_metainterp::jitcode::JitCallArg::int(int_tmp0) // dummy
                    };
                    let item1 = if argc >= 2 {
                        majit_metainterp::jitcode::JitCallArg::reference(arg_regs_start + 1)
                    } else {
                        majit_metainterp::jitcode::JitCallArg::int(int_tmp0) // dummy
                    };
                    assembler.call_may_force_ref_typed(
                        build_list_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                            item0,
                            item1,
                        ],
                        obj_tmp0,
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // Exception handling: residual calls to frame helpers.
                // RPython blackhole.py handles exceptions via dedicated
                // bhimpl_* functions. In pyre, we delegate to the frame's
                // exception machinery via call_fn.
                Instruction::RaiseVarargs { argc } => {
                    let n = argc.get(op_arg) as i64;
                    if n >= 1 {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                        emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth);
                        emit_raise!(ssarepr, assembler, obj_tmp0);
                    } else {
                        // reraise: re-raise exception_last_value
                        emit_reraise!(ssarepr, assembler);
                    }
                }

                Instruction::PushExcInfo => {
                    // flatten.py: dup = ref_copy TOS → TOS+1
                    emit_move_r!(
                        ssarepr,
                        assembler,
                        stack_base + current_depth,
                        stack_base + current_depth - 1
                    );
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                Instruction::CheckExcMatch => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp1, stack_base + current_depth); // match type
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth); // exception
                    // isinstance check via compare_fn(exc, type, ISINSTANCE_OP)
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, 10); // isinstance op
                    assembler.call_may_force_ref_typed(
                        compare_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        obj_tmp0,
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                Instruction::PopExcept => {
                    // Clear current exception — no-op in blackhole
                    // (exception state is in TLS, cleared by interpreter).
                }

                Instruction::Reraise { .. } => {
                    // Exception path: abort_permanent.
                    assembler.abort_permanent();
                }

                Instruction::Copy { i } => {
                    let d = i.get(op_arg) as usize;
                    if d == 1 {
                        assembler
                            .move_r(stack_base + current_depth, stack_base + current_depth - 1);
                        current_depth += 1;
                        emit_vsd!(assembler, current_depth);
                    } else {
                        // COPY(d>1): exception handler pattern only.
                        // Use abort_permanent (BC_ABORT_PERMANENT=14) so it
                        // doesn't trigger the has_abort(BC_ABORT=13) check.
                        assembler.abort_permanent();
                    }
                }

                Instruction::LoadName { .. }
                | Instruction::StoreName { .. }
                | Instruction::MakeFunction { .. } => {
                    // Module-level only: abort_permanent (won't block blackhole).
                    assembler.abort_permanent();
                }

                // CPython 3.13 superinstruction: STORE_FAST_STORE_FAST.
                // jtransform.py:1898 — each local write → setarrayitem_vable_r.
                Instruction::StoreFastStoreFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let reg_a = u32::from(pair.idx_1()) as u16;
                    let reg_b = u32::from(pair.idx_2()) as u16;
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, reg_a, stack_base + current_depth);
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, reg_b, stack_base + current_depth);
                    // Shadow write both stores to vable array.
                    if is_portal {
                        emit_load_const_i!(
                            ssarepr,
                            assembler,
                            int_tmp0,
                            local_to_vable_slot(reg_a as usize) as i64,
                        );
                        emit_vable_setarrayitem_ref!(ssarepr, assembler, 0_u16, int_tmp0, reg_a);
                        emit_load_const_i!(
                            ssarepr,
                            assembler,
                            int_tmp0,
                            local_to_vable_slot(reg_b as usize) as i64,
                        );
                        emit_vable_setarrayitem_ref!(ssarepr, assembler, 0_u16, int_tmp0, reg_b);
                    }
                }

                // CPython 3.13 UNPACK_SEQUENCE: pop 1 (seq), push `count`.
                // Emit abort_permanent (no getitem helper yet) but
                // adjust current_depth so subsequent instructions don't
                // underflow.
                Instruction::UnpackSequence { count } => {
                    let n = count.get(op_arg) as u16;
                    assembler.abort_permanent();
                    // Stack effect: pop 1 + push n = net (n - 1)
                    if current_depth > 0 {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                    }
                    current_depth += n;
                }

                // CPython 3.13 iterator protocol — emit abort_permanent
                // with correct depth tracking so subsequent instructions
                // don't underflow.
                Instruction::GetIter => {
                    // pop iterable, push iterator: net 0
                    assembler.abort_permanent();
                }

                Instruction::ForIter { .. } => {
                    // push next item: net +1
                    assembler.abort_permanent();
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                Instruction::EndFor => {
                    // pop iterator + last value: net -2
                    assembler.abort_permanent();
                    current_depth = current_depth.saturating_sub(2);
                    // No emit_vsd: after abort_permanent, depth is
                    // simulation-only for subsequent compile-time tracking.
                }

                Instruction::PopIter => {
                    // pop iterator: net -1
                    current_depth = current_depth.saturating_sub(1);
                    emit_vsd!(assembler, current_depth);
                }

                // Unsupported instruction: abort_permanent.
                _other => {
                    assembler.abort_permanent();
                }
            }
            if let Some(catch_label) = catch_for_pc[py_pc] {
                emit_catch_exception!(ssarepr, assembler, catch_label);
            }
        }

        // RPython flatten.py parity: every code path ends with an explicit
        // return/raise/goto/unreachable. No end-of-code sentinel needed —
        // falling off the end is unreachable if all bytecodes are covered.

        // pyre-only PyJitCode.has_abort: a "this jitcode cannot be
        // blackhole-dispatched, pipe straight to the interpreter" flag.
        // RPython has no such flag (rpython/jit/codewriter/jitcode.py:14
        // — no abort tracking on JitCode). Upstream's `Assembler.abort()`
        // (assembler.py:177-181, bhimpl_abort) emits BC_ABORT so the
        // blackhole raises SwitchToBlackhole(ABORT_ESCAPE) at runtime;
        // `abort_permanent()` is a different pyre-only bytecode we emit
        // for genuinely unsupported Python opcodes, and its execution
        // path already raises/aborts correctly from the blackhole. We
        // keep has_abort narrowly scoped to `abort()` emissions (matches
        // the JitCodeBuilder flag shape) so the flag's meaning doesn't
        // drift into "assembler overflow" or "abort_permanent present"
        // — both of which the assembler/blackhole already handle without
        // a front-end gate.
        let mut has_abort = assembler.has_abort_flag();

        // liveness.py parity: generate LivenessInfo at each bytecode PC.
        // RPython: compute_liveness() runs backward dataflow on SSARepr,
        // marking only actually-live registers. Both get_list_of_active_boxes
        // (pyjitpl.py:177) and consume_one_section (resume.py:1381) use the
        // same all_liveness data. In pyre, LiveVars::compute provides the
        // equivalent backward analysis on Python bytecodes.
        let liveness: Vec<LivenessInfo> = {
            let live_vars = pyre_jit_trace::state::liveness_for(code as *const _);
            let mut liveness = Vec::with_capacity(num_instrs);
            for py_pc in 0..num_instrs {
                let jit_pc = pc_map[py_pc];
                // RPython flatten_graph() emits blocks only for reachable
                // control-flow targets. Pyre's codewriter iterates every
                // py_pc linearly for layout (so `live_patches` has one
                // entry per pc), but must not generate liveness for
                // bytecodes the dataflow analysis never reached — those
                // pcs carry the `usize::MAX` sentinel from
                // liveness.rs:150 and would truncate into a 65535 stack
                // depth, blowing past the 256-register limit and
                // poisoning the entire jitcode with `liveness_overflow`.
                // Emit an empty LivenessInfo for unreachable pcs to keep
                // the `liveness.zip(live_patches)` alignment intact.
                let (live_r, live_i, live_f) = if live_vars.is_reachable(py_pc) {
                    let depth = depth_at_pc[py_pc];
                    // liveness.py: only live locals are included.
                    let mut live_r: Vec<u16> = (0..nlocals)
                        .filter(|&idx| live_vars.is_local_live(py_pc, idx))
                        .map(|idx| idx as u16)
                        .collect();
                    // Stack slots: live if index < depth at this PC.
                    for d in 0..depth {
                        live_r.push(stack_base + d);
                    }
                    (live_r, vec![], vec![])
                } else {
                    (vec![], vec![], vec![])
                };
                liveness.push(LivenessInfo {
                    pc: jit_pc as u16,
                    live_i_regs: live_i,
                    live_r_regs: live_r,
                    live_f_regs: live_f,
                });
            }
            liveness
        };

        // liveness.py:19-23 parity: fill each SSARepr Insn::Live with the
        // live register triple computed above. Assembler::assemble's
        // encode_liveness_info (routed through
        // pyre_jit_trace::state::intern_liveness) interns the bytes and
        // emits the BC_LIVE opcode plus 2-byte global offset.
        //
        // Overflow detection matches the earlier direct-builder loop:
        // if liveness_regs_to_u8_sorted rejects a register >= 256 or
        // intern_liveness runs out of u16 offset space, mark the jitcode
        // unexecutable (has_abort = true) so the caller falls back to
        // the interpreter. See liveness_regs_to_u8_sorted's doc comment.
        {
            let mut liveness_overflow = false;
            for (entry, &(_, patch_offset)) in liveness.iter().zip(live_patches.iter()) {
                let live_i_bytes = liveness_regs_to_u8_sorted(&entry.live_i_regs);
                let live_r_bytes = liveness_regs_to_u8_sorted(&entry.live_r_regs);
                let live_f_bytes = liveness_regs_to_u8_sorted(&entry.live_f_regs);
                if live_i_bytes.is_none() || live_r_bytes.is_none() || live_f_bytes.is_none() {
                    liveness_overflow = true;
                    break;
                }
                assembler.fill_live_args(
                    patch_offset,
                    &entry.live_i_regs,
                    &entry.live_r_regs,
                    &entry.live_f_regs,
                );
            }
            if liveness_overflow {
                // Propagate to the builder so Assembler::assemble can
                // skip its check_result assertion — the JitCode produced
                // here will never reach the blackhole anyway.
                assembler.set_abort_flag(true);
                has_abort = true;
            }
        }

        for site in catch_sites {
            emit_mark_label_catch_landing!(ssarepr, assembler, site.landing_label);
            let mut exc_slot = stack_base + site.stack_depth;
            if site.push_lasti {
                emit_load_const_i!(ssarepr, assembler, int_tmp0, site.lasti_py_pc as i64);
                assembler.call_ref_typed(
                    box_int_fn_idx,
                    &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                    obj_tmp0,
                );
                emit_move_r!(ssarepr, assembler, exc_slot, obj_tmp0);
                exc_slot += 1;
            }
            emit_last_exc_value!(ssarepr, assembler, exc_slot);
            emit_goto!(ssarepr, assembler, labels, site.handler_py_pc);
        }

        // pc_map[py_pc] currently holds SSARepr insn indices (returned by
        // SSAReprEmitter::current_pos()). Translate them to JitCode byte
        // offsets via ssarepr.insns_pos, populated during
        // Assembler::assemble (assembler.py:41-44). Runtime readers
        // (get_live_vars_info, resume dispatch) expect byte offsets.
        //
        // `codewriter.py:67` `self.assembler.assemble(ssarepr, jitcode, num_regs)`
        // parity: borrow the CodeWriter's single Assembler so
        // `all_liveness` / `num_liveness_ops` continue to accumulate
        // across every jitcode compiled on this thread. The throwaway
        // `SSAReprEmitter::finish_and_translate_positions()` path is
        // retained only for unit tests.
        let (mut jitcode, pc_map_bytes) = {
            let mut asm = self.assembler.borrow_mut();
            assembler.finish_with_positions(&mut *asm, &pc_map)
        };
        let pc_map = pc_map_bytes;

        // call.py:148 grab_initial_jitcodes: jd.mainjitcode.jitdriver_sd = jd
        // pyre uses index 0 (single jitdriver) for portal jitcodes.
        jitcode.jitdriver_sd = if is_portal { Some(0) } else { None };
        // call.py:174-187 get_jitcode_calldescr: calldescr from function type.
        // pyre portal: bh_portal_runner(frame_ptr: ref) -> ref.
        // All Python functions share the same calling convention.
        jitcode.calldescr = majit_codewriter::jitcode::BhCallDescr {
            arg_classes: "r".to_string(),
            result_type: 'r',
        };
        // Per-code stack base in `locals_cells_stack_w`. RPython's JitCode
        // does not carry PyFrame layout data; keep it in PyJitCodeMetadata
        // and attach it to BlackholeInterpreter setup when pyre needs it.
        let frame_stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);

        // call.py:167-169 jitcode.fnaddr = getfunctionptr(graph).
        // pyre: all Python functions go through the single portal runner.
        jitcode.fnaddr = crate::call_jit::bh_portal_runner as *const () as usize as i64;

        let metadata = PyJitCodeMetadata {
            pc_map,
            depth_at_py_pc: depth_at_pc,
            portal_frame_reg,
            portal_ec_reg,
            stack_base: frame_stack_base,
            liveness,
        };

        PyJitCode {
            jitcode: std::sync::Arc::new(jitcode),
            metadata,
            has_abort,
            merge_point_pc,
        }
    }

    /// RPython: `CodeWriter.make_jitcodes(verbose)` (codewriter.py:74-89).
    ///
    /// ```python
    /// def make_jitcodes(self, verbose=False):
    ///     log.info("making JitCodes...")
    ///     self.callcontrol.grab_initial_jitcodes()
    ///     count = 0
    ///     all_jitcodes = []
    ///     for graph, jitcode in self.callcontrol.enum_pending_graphs():
    ///         self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
    ///         all_jitcodes.append(jitcode)
    ///         count += 1
    ///         if not count % 500:
    ///             log.info("Produced %d jitcodes" % count)
    ///     self.assembler.finished(self.callcontrol.callinfocollection)
    ///     log.info("There are %d JitCode instances." % count)
    ///     log.info("There are %d -live- ops. Size of liveness is %s bytes" % (
    ///         self.assembler.num_liveness_ops, self.assembler.all_liveness_length))
    ///     return all_jitcodes
    /// ```
    ///
    /// The pyre drain loop adds one extra step that RPython's drain does
    /// not have: each freshly-compiled jitcode is published to
    /// `pyre_jit_trace::set_majit_jitcode` so the lower
    /// `pyre_jit_trace` crate (which cannot depend on pyre-jit) sees
    /// the latest `JitCode` / `pc_map` / `liveness` slots. RPython's
    /// equivalent is `MetaInterpStaticData::finish_setup` at
    /// `pyjitpl.py:2255-2269`, which the assembler's `finished()` call
    /// at `codewriter.py:85` is the upstream sibling of.
    ///
    /// The third tuple field — `merge_point_pc` — is a pyre-only
    /// refinement hook; see
    /// [`super::call::CallControl::grab_initial_jitcodes`].
    pub fn make_jitcodes(
        &self,
        initial: &[(&CodeObject, *const (), Option<usize>)],
    ) -> Vec<*const PyJitCode> {
        // codewriter.py:75 `log.info("making JitCodes...")` — pyre has no
        // codewriter.py log channel, intentionally elided.

        // codewriter.py:76 `self.callcontrol.grab_initial_jitcodes()`.
        self.callcontrol().grab_initial_jitcodes(initial);
        // codewriter.py:77 `count = 0`.
        let mut count: usize = 0;
        // codewriter.py:78 `all_jitcodes = []`.
        let mut all_jitcodes: Vec<*const PyJitCode> = Vec::new();
        // codewriter.py:79 `for graph, jitcode in
        //                     self.callcontrol.enum_pending_graphs():`.
        loop {
            let popped = self.callcontrol().enum_pending_graphs();
            let Some((code_ptr, w_code, merge_point_pc)) = popped else {
                break;
            };
            // codewriter.py:80 `self.transform_graph_to_jitcode(graph,
            //                     jitcode, verbose, len(all_jitcodes))`.
            //
            // PRE-EXISTING-ADAPTATION: RPython mutates the empty JitCode
            // skeleton in place (`assembler.assemble(ssarepr, jitcode,
            // num_regs)` at codewriter.py:67), whereas pyre's
            // `transform_graph_to_jitcode` returns a fresh `PyJitCode`
            // because the underlying `JitCodeBuilder` produces an
            // owned `JitCode`. We replace the skeleton entry in
            // `jitcodes` with the populated one to keep upstream's
            // "jitcode in dict has its body filled after drain"
            // invariant.
            let pyjitcode =
                self.transform_graph_to_jitcode(unsafe { &*code_ptr }, w_code, merge_point_pc);
            // pyre-only: pipe the freshly populated entry to the lower
            // `pyre_jit_trace` crate so the runtime side picks it up.
            // RPython's equivalent is the
            // `MetaInterpStaticData::finish_setup` pull at
            // `pyjitpl.py:2255-2269`.
            if !pyjitcode.has_abort {
                pyre_jit_trace::set_majit_jitcode(
                    w_code,
                    pyjitcode.jitcode.clone(),
                    pyjitcode.metadata.pc_map.clone(),
                    pyjitcode.metadata.liveness.clone(),
                );
            }
            let key = code_ptr as usize;
            let boxed = Box::new(pyjitcode);
            let raw_ptr = &*boxed as *const PyJitCode;
            self.callcontrol().jitcodes.insert(key, boxed);
            // codewriter.py:81 `all_jitcodes.append(jitcode)`.
            all_jitcodes.push(raw_ptr);
            // codewriter.py:82 `count += 1`.
            count += 1;
            // codewriter.py:83-84 `if not count % 500: log.info(...)` —
            // log channel intentionally elided.
        }
        // codewriter.py:85 `self.assembler.finished(self.callcontrol.callinfocollection)`.
        self.assembler
            .borrow_mut()
            .finished(&self.callcontrol().callinfocollection);
        // codewriter.py:86-88 final log lines — elided.
        let _ = count;
        // codewriter.py:89 `return all_jitcodes`.
        all_jitcodes
    }
}

// ---------------------------------------------------------------------------
// Jump target calculation (RPython: flatten.py link following)
// ---------------------------------------------------------------------------

/// Forward jump target: skip_caches(next_instr) + delta.
/// Must match pyre-interpreter/pyopcode.rs:jump_target_forward.
fn jump_target_forward(
    code: &CodeObject,
    num_instrs: usize,
    next_instr: usize,
    delta: usize,
) -> usize {
    let target = skip_caches(code, next_instr) + delta;
    target.min(num_instrs)
}

/// Single-source-of-truth backward-jump target calculation used by both
/// the loop-header pre-scan (pypy/module/pypyjit/interp_jit.py:103) and
/// the emitter (jtransform.py:1714 `handle_jit_marker__loop_header`).
///
/// Returns the target PC for `JumpBackward` (with `skip_caches` on the
/// next-PC base) and `JumpBackwardNoInterrupt` (direct `py_pc + 1 - delta`
/// arithmetic to match the interpreter's dispatch in pyopcode.rs).
/// Returns `None` for any non-backward-jump opcode.
fn backward_jump_target(
    code: &CodeObject,
    py_pc: usize,
    instr: Instruction,
    op_arg: pyre_interpreter::OpArg,
) -> Option<usize> {
    match instr {
        Instruction::JumpBackward { delta } => {
            Some(skip_caches(code, py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
        }
        Instruction::JumpBackwardNoInterrupt { delta } => {
            Some((py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
        }
        _ => None,
    }
}

/// Match pyre-interpreter/pyopcode.rs:skip_caches.
fn skip_caches(code: &CodeObject, mut pos: usize) -> usize {
    let mut state = OpArgState::default();
    while pos < code.instructions.len() {
        let (instruction, _) = state.get(code.instructions[pos]);
        if matches!(instruction, Instruction::Cache) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

// ---------------------------------------------------------------------------
// JitCode cache — RPython: `CallControl.get_jitcode` (call.py:155-172).
// The cache + `unfinished_graphs` queue live on `super::call::CallControl`;
// `CallControl::get_jitcode` is the canonical entry point.
// ---------------------------------------------------------------------------

/// RPython parity: `codewriter.make_jitcodes()` (codewriter.py:74-89)
/// is the single entry point that populates `MetaInterpStaticData`'s
/// JitCode set before any tracing happens.
///
/// pyre-adaptation: Python's dynamic dispatch prevents static callee
/// discovery, so pyre calls `make_jitcodes(&[portal_code])` at each JIT
/// entry rather than once at warmspot init. The `merge_point_pc` tuple
/// field threads through `CallControl::grab_initial_jitcodes` so that
/// the first trace's MERGE_POINT PC refines the cached JitCode (see
/// `super::call::CallControl::grab_initial_jitcodes`).
pub fn ensure_jitcode_for(
    code: &pyre_interpreter::CodeObject,
    w_code: *const (),
    merge_point_pc: Option<usize>,
) {
    CodeWriter::instance().make_jitcodes(&[(code, w_code, merge_point_pc)]);
}

/// Scan `code` for JUMP_BACKWARD targets — the PCs where
/// `transform_graph_to_jitcode` would emit `BC_JUMP_TARGET` and where
/// `jit_merge_point` is evaluated.
///
/// RPython parity: corresponds to `jtransform.py:1714-1718`
/// `handle_jit_marker__loop_header`, which walks the flow graph looking
/// for `jit_marker('loop_header', ...)` operations. pyre's "graph" is
/// raw Python bytecode, so the equivalent scan looks for
/// `JUMP_BACKWARD` instructions and resolves their target PCs.
///
/// Shared between `transform_graph_to_jitcode` (which uses the set to
/// decide emit locations) and `is_portal` (which tests emptiness to
/// classify the CodeObject without triggering a compile).
pub fn find_loop_header_pcs(
    code: &pyre_interpreter::CodeObject,
) -> std::collections::HashSet<usize> {
    let num_instrs = code.instructions.len();
    let mut loop_header_pcs: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut scan_state = OpArgState::default();
    for scan_pc in 0..num_instrs {
        let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
        if let Some(target) = backward_jump_target(code, scan_pc, scan_instr, scan_arg) {
            if target < num_instrs {
                loop_header_pcs.insert(target);
            }
        }
    }
    loop_header_pcs
}

/// jitcode.py:18: `jitcode.jitdriver_sd is not None`.
///
/// pyre checks the pre-compile signal — "does `code` contain any
/// `JUMP_BACKWARD` instructions" — which is the exact condition that
/// `transform_graph_to_jitcode` uses to set `JitCode.jitdriver_sd`
/// (codewriter.rs). The check is pure (no cache mutation, no compile)
/// so callers on the hot JIT-entry path can use it to classify a
/// CodeObject before deciding whether to trace.
pub fn is_portal(code: &pyre_interpreter::CodeObject) -> bool {
    !find_loop_header_pcs(code).is_empty()
}

#[cfg(test)]
mod tests {
    use super::liveness_regs_to_u8_sorted;

    #[test]
    fn liveness_regs_to_u8_sorted_sorts_and_dedups() {
        assert_eq!(
            liveness_regs_to_u8_sorted(&[7, 1, 7, 3]),
            Some(vec![1, 3, 7])
        );
    }

    /// rpython/jit/codewriter/liveness.py:139 enforces a 256-register cap.
    /// Register indices `>= 256` cannot be encoded and the helper must
    /// surface an overflow sentinel so the outer codewriter can fall back
    /// to `BC_ABORT` instead of panicking at runtime.
    #[test]
    fn liveness_regs_to_u8_sorted_rejects_out_of_range_registers() {
        assert_eq!(liveness_regs_to_u8_sorted(&[255, 256]), None);
    }

    /// Boundary: exactly 255 is still valid (u8::MAX).
    #[test]
    fn liveness_regs_to_u8_sorted_accepts_u8_max() {
        assert_eq!(liveness_regs_to_u8_sorted(&[255]), Some(vec![255]));
    }
}
