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
    /// py_pc → jitcode byte offset.
    pub pc_map: Vec<usize>,
    /// py_pc → jitcode byte offset, named for consumers that mirror RPython's
    /// frame.pc → jitcode position flow.
    pub py_to_jit_pc: Vec<usize>,
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
}

// ---------------------------------------------------------------------------
// RPython: codewriter/codewriter.py — class CodeWriter
// ---------------------------------------------------------------------------

/// Compiles Python CodeObjects into JitCode for blackhole execution.
///
/// RPython: CodeWriter class in codewriter/codewriter.py.
/// `codewriter.py:20-23` stores `self.assembler = Assembler()` once on
/// the CodeWriter and reuses it across every `transform_graph_to_jitcode`
/// call so `all_liveness` / `num_liveness_ops` accumulate over the
/// whole translator session. pyre mirrors that ownership via the
/// `RefCell<Assembler>` field below: the thread-local
/// `with_codewriter(...)` accessor hands out a single long-lived
/// instance, so every compiled jitcode in a thread's lifetime shares
/// the same `Assembler` and its per-kind counters actually accumulate.
pub struct CodeWriter {
    /// Blackhole helper function pointers, registered once.
    ///
    /// RPython: CallControl manages these via callinfocollection.
    /// In pyre, we store the concrete function pointers that the
    /// BlackholeInterpreter calls through JitCode.fn_ptrs.
    /// bhimpl_residual_call: per-arity call helpers.
    /// Parent frame via BH_VABLE_PTR thread-local.
    /// call_fn_0(callable) ... call_fn_8(callable, a0..a7).
    pub call_fn: extern "C" fn(i64, i64) -> i64,
    pub call_fn_0: extern "C" fn(i64) -> i64,
    pub call_fn_2: extern "C" fn(i64, i64, i64) -> i64,
    pub call_fn_3: extern "C" fn(i64, i64, i64, i64) -> i64,
    pub call_fn_4: extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub call_fn_5: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_6: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_7: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    pub call_fn_8: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64,
    /// jtransform.py: namespace+code from getfield_vable_r.
    pub load_global_fn: extern "C" fn(i64, i64, i64) -> i64,
    pub compare_fn: extern "C" fn(i64, i64, i64) -> i64,
    pub binary_op_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// Box a raw integer into a PyObject (w_int_new).
    pub box_int_fn: extern "C" fn(i64) -> i64,
    /// Truthiness check: PyObjectRef → raw 0 or 1.
    pub truth_fn: extern "C" fn(i64) -> i64,
    /// Load constant from frame's code object.
    pub load_const_fn: extern "C" fn(i64, i64) -> i64,
    /// Store subscript: obj[key] = value.
    pub store_subscr_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// Build list: (argc, item0, item1) → new list.
    pub build_list_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// `codewriter.py:22` `self.assembler = Assembler()`.
    ///
    /// Single Assembler instance shared across every `transform_graph_to_jitcode`
    /// call on this CodeWriter. `all_liveness` / `all_liveness_positions` /
    /// `num_liveness_ops` accumulate here just like the upstream object.
    assembler: RefCell<Assembler>,
}

impl CodeWriter {
    pub fn new(
        call_fn: extern "C" fn(i64, i64) -> i64,
        load_global_fn: extern "C" fn(i64, i64, i64) -> i64,
        compare_fn: extern "C" fn(i64, i64, i64) -> i64,
        binary_op_fn: extern "C" fn(i64, i64, i64) -> i64,
        box_int_fn: extern "C" fn(i64) -> i64,
        truth_fn: extern "C" fn(i64) -> i64,
        load_const_fn: extern "C" fn(i64, i64) -> i64,
        store_subscr_fn: extern "C" fn(i64, i64, i64) -> i64,
        build_list_fn: extern "C" fn(i64, i64, i64) -> i64,
    ) -> Self {
        Self {
            call_fn,
            call_fn_0: crate::call_jit::bh_call_fn_0,
            call_fn_2: crate::call_jit::bh_call_fn_2,
            call_fn_3: crate::call_jit::bh_call_fn_3,
            call_fn_4: crate::call_jit::bh_call_fn_4,
            call_fn_5: crate::call_jit::bh_call_fn_5,
            call_fn_6: crate::call_jit::bh_call_fn_6,
            call_fn_7: crate::call_jit::bh_call_fn_7,
            call_fn_8: crate::call_jit::bh_call_fn_8,
            load_global_fn,
            compare_fn,
            binary_op_fn,
            box_int_fn,
            truth_fn,
            load_const_fn,
            store_subscr_fn,
            build_list_fn,
            assembler: RefCell::new(Assembler::new()),
        }
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

        // RPython regalloc.py: keep kind-separated register files.
        assembler.ensure_r_regs(portal_ec_reg + 1);
        assembler.ensure_i_regs(op_code_reg + 1);

        // Register helper function pointers
        // RPython: CallControl manages fn addresses; assembler.finished()
        // writes them into callinfocollection.
        let call_fn_idx = assembler.add_fn_ptr(self.call_fn as *const ());
        let load_global_fn_idx = assembler.add_fn_ptr(self.load_global_fn as *const ());
        let compare_fn_idx = assembler.add_fn_ptr(self.compare_fn as *const ());
        let binary_op_fn_idx = assembler.add_fn_ptr(self.binary_op_fn as *const ());
        let box_int_fn_idx = assembler.add_fn_ptr(self.box_int_fn as *const ());
        let truth_fn_idx = assembler.add_fn_ptr(self.truth_fn as *const ());
        let load_const_fn_idx = assembler.add_fn_ptr(self.load_const_fn as *const ());
        let store_subscr_fn_idx = assembler.add_fn_ptr(self.store_subscr_fn as *const ());
        let build_list_fn_idx = assembler.add_fn_ptr(self.build_list_fn as *const ());
        // Per-arity call helpers (appended AFTER existing fn_ptrs to preserve indices).
        let call_fn_0_idx = assembler.add_fn_ptr(self.call_fn_0 as *const ());
        let call_fn_2_idx = assembler.add_fn_ptr(self.call_fn_2 as *const ());
        let call_fn_3_idx = assembler.add_fn_ptr(self.call_fn_3 as *const ());
        let call_fn_4_idx = assembler.add_fn_ptr(self.call_fn_4 as *const ());
        let call_fn_5_idx = assembler.add_fn_ptr(self.call_fn_5 as *const ());
        let call_fn_6_idx = assembler.add_fn_ptr(self.call_fn_6 as *const ());
        let call_fn_7_idx = assembler.add_fn_ptr(self.call_fn_7 as *const ());
        let call_fn_8_idx = assembler.add_fn_ptr(self.call_fn_8 as *const ());

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
        let mut loop_header_pcs: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        {
            let mut scan_state = OpArgState::default();
            for scan_pc in 0..num_instrs {
                let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
                if let Instruction::JumpBackward { delta } = scan_instr {
                    let target = skip_caches(code, scan_pc + 1)
                        .saturating_sub(delta.get(scan_arg).as_usize());
                    if target < num_instrs {
                        loop_header_pcs.insert(target);
                    }
                }
            }
        }
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
                    $asm.load_const_i_value(
                        int_tmp0,
                        (stack_base_absolute + $depth as usize) as i64,
                    );
                    $asm.vable_setfield_int(VABLE_VALUESTACKDEPTH_FIELD_IDX, int_tmp0);
                }
            };
        }

        for py_pc in 0..num_instrs {
            // Exception handler entry: Python resets stack depth to the
            // handler's specified depth. Correct current_depth to match.
            if let Some(&handler_depth) = handler_depth_at.get(&py_pc) {
                current_depth = handler_depth;
            }
            // RPython flatten.py: Label(block) at block entry
            assembler.mark_label(labels[py_pc]);
            pc_map[py_pc] = assembler.current_pos();
            let live_patch = assembler.live_placeholder();
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
                        assembler
                            .load_const_i_value(int_tmp0, local_to_vable_slot(reg as usize) as i64);
                        assembler.vable_getarrayitem_ref(stack_base + current_depth, 0, int_tmp0);
                    } else {
                        assembler.move_r(stack_base + current_depth, reg);
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
                    assembler.move_r(reg, stack_base + current_depth);
                    // Shadow write to vable array for consume_vable_info.
                    if is_portal {
                        assembler
                            .load_const_i_value(int_tmp0, local_to_vable_slot(reg as usize) as i64);
                        assembler.vable_setarrayitem_ref(0, int_tmp0, reg);
                    }
                }

                Instruction::LoadSmallInt { i } => {
                    let val = i.get(op_arg) as u32 as i64;
                    assembler.load_const_i_value(int_tmp0, val);
                    assembler.call_ref_typed(
                        box_int_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                        obj_tmp0,
                    );
                    assembler.move_r(stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                Instruction::LoadConst { consti } => {
                    let idx = consti.get(op_arg).as_usize();
                    // jtransform.py: getfield_vable_r for pycode (field 1)
                    assembler.vable_getfield_ref(obj_tmp0, VABLE_CODE_FIELD_IDX);
                    assembler.load_const_i_value(int_tmp0, idx as i64);
                    assembler.call_ref_typed(
                        load_const_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        obj_tmp0,
                    );
                    assembler.move_r(stack_base + current_depth, obj_tmp0);
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
                    assembler.move_r(stack_base + current_depth, reg_a);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(stack_base + current_depth, reg_b);
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
                    assembler.move_r(store_reg, stack_base + current_depth);
                    assembler.move_r(stack_base + current_depth, load_reg);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // STORE_SUBSCR: stack [value, obj, key] → obj[key] = value
                Instruction::StoreSubscr => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(obj_tmp1, stack_base + current_depth); // key
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(obj_tmp0, stack_base + current_depth); // obj
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(arg_regs_start, stack_base + current_depth); // value
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
                    assembler.move_r(stack_base + current_depth, null_ref_reg);
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
                    assembler.move_r(obj_tmp1, stack_base + current_depth); // rhs
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(obj_tmp0, stack_base + current_depth); // lhs
                    assembler.load_const_i_value(op_code_reg, op_val as i64);
                    assembler.call_may_force_ref_typed(
                        binary_op_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::int(op_code_reg),
                        ],
                        obj_tmp0,
                    );
                    assembler.move_r(stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // jtransform.py: rewrite_op_int_lt, optimize_goto_if_not
                Instruction::CompareOp { opname } => {
                    let op_val = compare_op_tag(opname.get(op_arg)) as u32;
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(obj_tmp1, stack_base + current_depth); // rhs
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(obj_tmp0, stack_base + current_depth); // lhs
                    assembler.load_const_i_value(op_code_reg, op_val as i64);
                    assembler.call_may_force_ref_typed(
                        compare_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::int(op_code_reg),
                        ],
                        obj_tmp0,
                    );
                    assembler.move_r(stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // jtransform.py: optimize_goto_if_not → goto_if_not
                Instruction::PopJumpIfFalse { delta } => {
                    let target_py_pc = jump_target_forward(
                        code,
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(obj_tmp0, stack_base + current_depth);
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
                    assembler.move_r(obj_tmp0, stack_base + current_depth);
                    assembler.call_int_typed(
                        truth_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0)],
                        int_tmp0,
                    );
                    assembler.load_const_i_value(int_tmp1, 0);
                    assembler.record_binop_i(int_tmp0, OpCode::IntEq, int_tmp0, int_tmp1);
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
                        assembler.jump(labels[target_py_pc]);
                    }
                }

                Instruction::JumpBackward { delta } => {
                    let target_py_pc =
                        skip_caches(code, py_pc + 1).saturating_sub(delta.get(op_arg).as_usize());
                    if target_py_pc < num_instrs {
                        assembler.jump(labels[target_py_pc]);
                    }
                }

                // flatten.py: int_return / ref_return
                Instruction::ReturnValue => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(obj_tmp0, stack_base + current_depth);
                    assembler.ref_return(obj_tmp0);
                }

                // RPython jtransform.py: rewrite_op_direct_call (residual)
                Instruction::LoadGlobal { namei } => {
                    let raw_namei = namei.get(op_arg) as usize as i64;
                    // jtransform.py: getfield_vable_r for w_globals (field 3)
                    // and pycode (field 1) — namespace for lookup, code for names.
                    assembler.vable_getfield_ref(obj_tmp0, VABLE_NAMESPACE_FIELD_IDX);
                    assembler.vable_getfield_ref(obj_tmp1, VABLE_CODE_FIELD_IDX);
                    assembler.load_const_i_value(int_tmp0, raw_namei);
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
                        assembler.move_r(stack_base + current_depth, null_ref_reg);
                        current_depth += 1;
                        emit_vsd!(assembler, current_depth);
                    }
                    assembler.move_r(stack_base + current_depth, obj_tmp0);
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
                        assembler.move_r(arg_regs_start + i as u16, stack_base + current_depth);
                    }
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(obj_tmp1, stack_base + current_depth); // callable
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
                    assembler.move_r(stack_base + current_depth, obj_tmp0);
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
                    assembler.move_r(obj_tmp0, stack_base + current_depth);
                    assembler.load_const_i_value(int_tmp0, 0);
                    assembler.call_may_force_ref_typed(
                        box_int_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                        obj_tmp1,
                    );
                    assembler.load_const_i_value(
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
                    assembler.move_r(stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // Match the interpreter's direct PC arithmetic for
                // JumpBackwardNoInterrupt. Unlike JumpBackward, the
                // bytecode target is encoded from the post-dispatch PC
                // without an extra cache skip.
                Instruction::JumpBackwardNoInterrupt { delta } => {
                    let target_py_pc = (py_pc + 1).saturating_sub(delta.get(op_arg).as_usize());
                    if target_py_pc < num_instrs {
                        assembler.jump(labels[target_py_pc]);
                    }
                }

                // RPython bhimpl_newlist: build list from N items on stack.
                Instruction::BuildList { count } => {
                    let argc = count.get(op_arg) as usize;
                    for i in (0..argc.min(2)).rev() {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                        assembler.move_r(arg_regs_start + i as u16, stack_base + current_depth);
                    }
                    // Discard extra items beyond 2 (helper supports 0-2).
                    for _ in 2..argc {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                    }
                    // build_list_fn(argc, item0, item1) → list
                    assembler.load_const_i_value(int_tmp0, argc as i64);
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
                    assembler.move_r(stack_base + current_depth, obj_tmp0);
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
                        assembler.move_r(obj_tmp0, stack_base + current_depth);
                        assembler.emit_raise(obj_tmp0);
                    } else {
                        // reraise: re-raise exception_last_value
                        assembler.emit_reraise();
                    }
                }

                Instruction::PushExcInfo => {
                    // flatten.py: dup = ref_copy TOS → TOS+1
                    assembler.move_r(stack_base + current_depth, stack_base + current_depth - 1);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                Instruction::CheckExcMatch => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(obj_tmp1, stack_base + current_depth); // match type
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(obj_tmp0, stack_base + current_depth); // exception
                    // isinstance check via compare_fn(exc, type, ISINSTANCE_OP)
                    assembler.load_const_i_value(int_tmp0, 10); // isinstance op
                    assembler.call_may_force_ref_typed(
                        compare_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        obj_tmp0,
                    );
                    assembler.move_r(stack_base + current_depth, obj_tmp0);
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
                    assembler.move_r(reg_a, stack_base + current_depth);
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    assembler.move_r(reg_b, stack_base + current_depth);
                    // Shadow write both stores to vable array.
                    if is_portal {
                        assembler.load_const_i_value(
                            int_tmp0,
                            local_to_vable_slot(reg_a as usize) as i64,
                        );
                        assembler.vable_setarrayitem_ref(0, int_tmp0, reg_a);
                        assembler.load_const_i_value(
                            int_tmp0,
                            local_to_vable_slot(reg_b as usize) as i64,
                        );
                        assembler.vable_setarrayitem_ref(0, int_tmp0, reg_b);
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
                assembler.catch_exception(catch_label);
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
            assembler.mark_label(site.landing_label);
            let mut exc_slot = stack_base + site.stack_depth;
            if site.push_lasti {
                assembler.load_const_i_value(int_tmp0, site.lasti_py_pc as i64);
                assembler.call_ref_typed(
                    box_int_fn_idx,
                    &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                    obj_tmp0,
                );
                assembler.move_r(exc_slot, obj_tmp0);
                exc_slot += 1;
            }
            assembler.last_exc_value(exc_slot);
            assembler.jump(labels[site.handler_py_pc]);
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
            pc_map: pc_map.clone(),
            py_to_jit_pc: pc_map.clone(),
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

    /// RPython: CodeWriter.make_jitcodes() — compile all reachable graphs.
    ///
    /// For pyre, JitCodes are compiled lazily per-CodeObject (not AOT).
    /// This method compiles a single CodeObject and caches the result.
    pub fn make_jitcode(
        &self,
        code: &CodeObject,
        w_code: *const (),
        merge_point_pc: Option<usize>,
    ) -> PyJitCode {
        self.transform_graph_to_jitcode(code, w_code, merge_point_pc)
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
// JitCode cache — lazy compilation per CodeObject
// RPython: CallControl.get_jitcode(graph, called_from) with dedup
// ---------------------------------------------------------------------------

thread_local! {
    static JITCODE_CACHE: UnsafeCell<HashMap<usize, PyJitCode>> =
        UnsafeCell::new(HashMap::new());

    /// `codewriter.py:20-23` single long-lived `CodeWriter` per thread.
    ///
    /// Matches RPython's "CodeWriter holds the Assembler" ownership model.
    /// All four `pyre/pyre-jit` call sites now route through
    /// `with_codewriter(...)` so every jitcode compiled on a thread's
    /// lifetime shares the same `Assembler` instance, letting its
    /// `all_liveness` / `num_liveness_ops` accumulate across calls just
    /// like upstream.
    static CODEWRITER: OnceCell<CodeWriter> = const { OnceCell::new() };
}

/// `codewriter.py:20-23` accessor. Initialises the thread-local
/// `CodeWriter` on first use with the standard `bh_*` helper pointers
/// (every current caller uses the same bundle). Additional call sites
/// should route through here rather than calling `CodeWriter::new(...)`
/// inline — a fresh CodeWriter-per-call splits the Assembler's
/// accumulator and diverges from RPython parity.
pub fn with_codewriter<R>(f: impl FnOnce(&CodeWriter) -> R) -> R {
    CODEWRITER.with(|cell| {
        let writer = cell.get_or_init(|| {
            CodeWriter::new(
                crate::call_jit::bh_call_fn,
                crate::call_jit::bh_load_global_fn,
                crate::call_jit::bh_compare_fn,
                crate::call_jit::bh_binary_op_fn,
                crate::call_jit::bh_box_int_fn,
                crate::call_jit::bh_truth_fn,
                crate::call_jit::bh_load_const_fn,
                crate::call_jit::bh_store_subscr_fn,
                crate::call_jit::bh_build_list_fn,
            )
        });
        f(writer)
    })
}

/// Get or compile JitCode for a CodeObject.
///
/// jitcode.py:18: jitdriver_sd is not None for portal jitcodes.
/// call.py:148 grab_initial_jitcodes: sets jitdriver_sd on the portal.
/// Cached per CodeObject pointer — jitdriver_sd is fixed per jitcode.
pub fn get_jitcode(
    code: &CodeObject,
    w_code: *const (),
    writer: &CodeWriter,
    merge_point_pc: Option<usize>,
) -> &'static PyJitCode {
    let key = code as *const CodeObject as usize;
    JITCODE_CACHE.with(|cell| {
        let cache = unsafe { &mut *cell.get() };
        let needs_rebuild = if let Some(existing) = cache.get(&key) {
            // Rebuild if merge_point_pc changed (first trace provides the
            // correct PC that was only an estimate at initial creation).
            merge_point_pc.is_some() && existing.merge_point_pc != merge_point_pc
        } else {
            true
        };
        if needs_rebuild {
            let pyjitcode = writer.make_jitcode(code, w_code, merge_point_pc);
            cache.insert(key, pyjitcode);
            // RPython parity: clone JitCode into state::JitCode (which is
            // Box'd in MetaInterpStaticData.jitcodes — stable address).
            // Skip when has_abort (liveness overflow cleared the data);
            // get_list_of_active_boxes falls back to LiveVars analysis.
            let entry = cache.get(&key).unwrap();
            if !entry.has_abort {
                pyre_jit_trace::set_majit_jitcode(
                    w_code,
                    entry.jitcode.clone(),
                    entry.metadata.py_to_jit_pc.clone(),
                    entry.metadata.liveness.clone(),
                );
            }
        }
        let entry = cache.get(&key).unwrap();
        // SAFETY: thread-local cache lives for thread lifetime
        unsafe { &*(entry as *const PyJitCode) }
    })
}

/// RPython parity: codewriter.make_jitcodes() runs before tracing.
/// In pyre, JitCode compilation is lazy. This function ensures the
/// JitCode (with liveness info) exists for a CodeObject so that
/// get_list_of_active_boxes can use it during tracing.
pub fn ensure_jitcode_for(
    code: &pyre_interpreter::CodeObject,
    w_code: *const (),
    merge_point_pc: Option<usize>,
) {
    with_codewriter(|writer| {
        let _ = get_jitcode(code, w_code, writer, merge_point_pc);
    });
}

/// jitcode.py:18: `jitcode.jitdriver_sd is not None`.
pub fn is_portal(code: &pyre_interpreter::CodeObject, w_code: *const ()) -> bool {
    ensure_jitcode_for(code, w_code, None);
    let key = code as *const pyre_interpreter::CodeObject as usize;
    JITCODE_CACHE.with(|cell| {
        let cache = unsafe { &*cell.get() };
        cache
            .get(&key)
            .map(|pjc| pjc.jitcode.jitdriver_sd.is_some())
            .unwrap_or(false)
    })
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
