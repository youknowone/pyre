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
use std::cell::UnsafeCell;
use std::collections::HashMap;

use majit_ir::OpCode;
use majit_metainterp::jitcode::{JitCode, JitCodeBuilder, LivenessInfo};
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
    pub jitcode: JitCode,
    pub metadata: PyJitCodeMetadata,
    /// True if the jitcode contains BC_ABORT opcodes (unsupported bytecodes).
    /// Precomputed at compile time to avoid repeated bytecode scanning.
    pub has_abort: bool,
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
/// RPython's CodeWriter holds an Assembler (shared across all JitCodes)
/// and a CallControl. For pyre, each CodeObject gets its own JitCodeBuilder
/// since Python functions are compiled lazily.
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
        }
    }

    /// Transform a Python CodeObject into a JitCode.
    ///
    /// RPython: CodeWriter.transform_graph_to_jitcode(graph, jitcode, verbose, index)
    ///
    /// Python bytecodes serve as the "graph". Since they are already linear
    /// and register-allocated, jtransform/regalloc/flatten are identity
    /// transforms. We go directly to assembly.
    pub fn transform_graph_to_jitcode(&self, code: &CodeObject, w_code: *const ()) -> PyJitCode {
        let nlocals = code.varnames.len();
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
        //   0=next_instr, 1=code, 2=vsd, 3=debugdata, 4=lastblock, 5=namespace
        const VABLE_CODE_FIELD_IDX: u16 = 1;
        const VABLE_NAMESPACE_FIELD_IDX: u16 = 5;

        // regalloc.py: compile-time stack depth counter — tracks which
        // stack register (stack_base + depth) is the current TOS.
        let mut current_depth: u16 = 0;

        // RPython: self.assembler = Assembler() + JitCode(graph.name, ...)
        // (rpython/jit/codewriter/jitcode.py:14-15 takes name as the first
        // __init__ argument; majit's JitCodeBuilder::set_name mirrors that).
        let mut assembler = JitCodeBuilder::default();
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

        // RPython jtransform.py:1714-1718 handle_jit_marker__loop_header:
        // Pre-scan JUMP_BACKWARD targets to identify loop headers.
        // RPython emits jit_merge_point + loop_header opcodes at these
        // positions; pyre emits BC_JUMP_TARGET as equivalent marker.
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
        let merge_point_pc = loop_header_pcs.iter().copied().min();
        let mut emitted_merge_point = false;

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
                if merge_point_pc == Some(py_pc) && !emitted_merge_point {
                    // interp_jit.py:64 portal contract:
                    //   greens = ['next_instr', 'is_being_profiled', 'pycode']
                    //   reds = ['frame', 'ec']
                    //
                    // jtransform.py:1690 make_three_lists parity:
                    //   gi = [next_instr, is_being_profiled]
                    //   gr = [pycode]
                    //   rr = [frame, ec]
                    //
                    // Green consts → constant pool → register bank slots.
                    let next_instr_const_idx = assembler.add_const_i(py_pc as i64);
                    let is_being_profiled_const_idx = assembler.add_const_i(0); // always 0
                    // Register index = num_regs_i + const_pool_idx
                    let num_regs_i = assembler.num_regs_i();
                    let gi_next_instr_reg = (num_regs_i + next_instr_const_idx) as u8;
                    let gi_is_profiled_reg = (num_regs_i + is_being_profiled_const_idx) as u8;

                    // pycode → ref constant pool → register bank slot.
                    // interp_jit.py:85: pycode is the W_CodeObject (PyObjectRef),
                    // not the inner CodeObject struct.
                    let pycode_const_idx = assembler.add_const_r(w_code as i64);
                    let num_regs_r = assembler.num_regs_r();
                    let gr_pycode_reg = (num_regs_r + pycode_const_idx) as u8;

                    // Red refs: dedicated portal registers (frame, ec).
                    let frame_reg = portal_frame_reg as u8;
                    let ec_reg = portal_ec_reg as u8;

                    assembler.jit_merge_point(
                        &[gi_next_instr_reg, gi_is_profiled_reg],
                        &[gr_pycode_reg],
                        &[frame_reg, ec_reg],
                    );
                    emitted_merge_point = true;
                } else {
                    assembler.jump_target();
                }
            }

            let code_unit = code.instructions[py_pc];
            let (instruction, op_arg) = arg_state.get(code_unit);

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
                // portal locals are virtualizable array items.
                // Phase 5 (vable read) blocked: emitting vable_getarrayitem_ref
                // alone causes nested_loop WRONG output; shadow read
                // would need virtualizable_boxes propagation across JUMP
                // back-edge (unroll.rs export_state/import_state parity).
                // Keep move_r until the blackhole resume path reads from
                // the frame directly.
                Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    assembler.move_r(stack_base + current_depth, reg);
                    current_depth += 1;
                }

                // jtransform.py:1898 do_fixed_list_setitem vable case:
                // Shadow write: keep move_r AND write to vable array.
                // Pure vable write (RPython parity) blocked on the same
                // virtualizable_boxes propagation gap as LoadFast above.
                Instruction::StoreFast { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    current_depth -= 1;
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
                }

                // Superinstruction: two consecutive LoadFast / LoadFastBorrow.
                // Phase 5 blocked — see LoadFast comment above.
                Instruction::LoadFastBorrowLoadFastBorrow { var_nums }
                | Instruction::LoadFastLoadFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let reg_a = u32::from(pair.idx_1()) as u16;
                    let reg_b = u32::from(pair.idx_2()) as u16;
                    assembler.move_r(stack_base + current_depth, reg_a);
                    current_depth += 1;
                    assembler.move_r(stack_base + current_depth, reg_b);
                    current_depth += 1;
                }

                // STORE_SUBSCR: stack [value, obj, key] → obj[key] = value
                Instruction::StoreSubscr => {
                    current_depth -= 1;
                    assembler.move_r(obj_tmp1, stack_base + current_depth); // key
                    current_depth -= 1;
                    assembler.move_r(obj_tmp0, stack_base + current_depth); // obj
                    current_depth -= 1;
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
                    // regalloc.py: discard = just decrement depth, no bytecode
                }

                Instruction::PushNull => {
                    assembler.move_r(stack_base + current_depth, null_ref_reg);
                    current_depth += 1;
                }

                // jtransform.py: rewrite_op_int_add etc.
                Instruction::BinaryOp { op } => {
                    let op_val = binary_op_tag(op.get(op_arg))
                        .expect("unsupported binary op tag in jitcode lowering")
                        as u32;
                    current_depth -= 1;
                    assembler.move_r(obj_tmp1, stack_base + current_depth); // rhs
                    current_depth -= 1;
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
                }

                // jtransform.py: rewrite_op_int_lt, optimize_goto_if_not
                Instruction::CompareOp { opname } => {
                    let op_val = compare_op_tag(opname.get(op_arg)) as u32;
                    current_depth -= 1;
                    assembler.move_r(obj_tmp1, stack_base + current_depth); // rhs
                    current_depth -= 1;
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
                    }
                    assembler.move_r(stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
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
                        assembler.move_r(arg_regs_start + i as u16, stack_base + current_depth);
                    }
                    current_depth -= 1;
                    assembler.move_r(obj_tmp1, stack_base + current_depth); // callable
                    current_depth -= 1; // NULL (discard)

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
                }

                // Python 3.13: ToBool converts TOS to bool before branch.
                // No-op in JitCode: the value is already truthy/falsy and
                // the following PopJumpIfFalse guards on it.
                Instruction::ToBool => {}

                // RPython bhimpl_int_neg: -obj via binary_op(0, obj, NB_SUBTRACT)
                Instruction::UnaryNegative => {
                    current_depth -= 1;
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
                        assembler.move_r(arg_regs_start + i as u16, stack_base + current_depth);
                    }
                    // Discard extra items beyond 2 (helper supports 0-2).
                    for _ in 2..argc {
                        current_depth -= 1;
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
                }

                // Exception handling: residual calls to frame helpers.
                // RPython blackhole.py handles exceptions via dedicated
                // bhimpl_* functions. In pyre, we delegate to the frame's
                // exception machinery via call_fn.
                Instruction::RaiseVarargs { argc } => {
                    let n = argc.get(op_arg) as i64;
                    if n >= 1 {
                        current_depth -= 1;
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
                }

                Instruction::CheckExcMatch => {
                    current_depth -= 1;
                    assembler.move_r(obj_tmp1, stack_base + current_depth); // match type
                    current_depth -= 1;
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
                    assembler.move_r(reg_a, stack_base + current_depth);
                    current_depth -= 1;
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
                }

                Instruction::EndFor => {
                    // pop iterator + last value: net -2
                    assembler.abort_permanent();
                    current_depth = current_depth.saturating_sub(2);
                }

                Instruction::PopIter => {
                    // pop iterator: net -1
                    current_depth = current_depth.saturating_sub(1);
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

        // RPython parity: JitCodeBuilder tracks has_abort via abort() calls
        // (BC_ABORT=13). abort_permanent() (BC_ABORT_PERMANENT=14) does not
        // set has_abort. This codewriter only uses abort_permanent() for
        // unsupported bytecodes, so has_abort comes from the assembler.
        // The flag lives on the builder, not on the finished JitCode —
        // matching RPython jitcode.py which has no has_abort field.
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
                liveness.push(LivenessInfo {
                    pc: jit_pc as u16,
                    live_i_regs: vec![],
                    live_r_regs: live_r,
                    live_f_regs: vec![],
                });
            }
            liveness
        };

        // liveness.py:147-166 encode_liveness parity: pack each
        // LivenessInfo into the all_liveness byte string with the
        // `[len_i][len_r][len_f] | bitset_i | bitset_r | bitset_f`
        // layout (liveness.py:139-145). The packed fragment is appended
        // to the single global `metainterp_sd.liveness_info` string
        // (pyjitpl.py:2264 parity) and each inline `live/` marker is
        // patched with a 2-byte GLOBAL offset.
        {
            use majit_codewriter::liveness::encode_liveness;

            let mut liveness_fragment: Vec<u8> = Vec::new();
            let mut liveness_positions: HashMap<(Vec<u8>, Vec<u8>, Vec<u8>), u16> = HashMap::new();
            let mut local_offsets: Vec<(usize, u16)> = Vec::with_capacity(live_patches.len());
            let mut liveness_overflow = false;
            for (entry, &(_, patch_offset)) in liveness.iter().zip(live_patches.iter()) {
                let live_i_bytes = liveness_regs_to_u8_sorted(&entry.live_i_regs);
                let live_r_bytes = liveness_regs_to_u8_sorted(&entry.live_r_regs);
                let live_f_bytes = liveness_regs_to_u8_sorted(&entry.live_f_regs);
                let (live_i_bytes, live_r_bytes, live_f_bytes) =
                    match (live_i_bytes, live_r_bytes, live_f_bytes) {
                        (Some(i), Some(r), Some(f)) => (i, r, f),
                        _ => {
                            liveness_overflow = true;
                            break;
                        }
                    };
                let key = (
                    live_i_bytes.clone(),
                    live_r_bytes.clone(),
                    live_f_bytes.clone(),
                );
                let offset = if let Some(&offset) = liveness_positions.get(&key) {
                    offset
                } else {
                    if liveness_fragment.len() > u16::MAX as usize {
                        liveness_overflow = true;
                        break;
                    }
                    let offset = liveness_fragment.len() as u16;
                    liveness_positions.insert(key, offset);
                    // liveness.py:144 header: three lengths.
                    liveness_fragment.push(live_i_bytes.len() as u8);
                    liveness_fragment.push(live_r_bytes.len() as u8);
                    liveness_fragment.push(live_f_bytes.len() as u8);
                    // liveness.py:145 body: three packed bitsets.
                    liveness_fragment.extend(encode_liveness(&live_i_bytes));
                    liveness_fragment.extend(encode_liveness(&live_r_bytes));
                    liveness_fragment.extend(encode_liveness(&live_f_bytes));
                    offset
                };
                local_offsets.push((patch_offset, offset));
            }
            if liveness_overflow {
                // Mark the jitcode as unexecutable; overflow means the
                // encoder could not fit into the u16 offset space. See
                // `liveness_regs_to_u8_sorted` comment for the TODO.
                has_abort = true;
            } else {
                // pyjitpl.py:2264 append to the global string, then
                // patch every inline `-live-` with the resulting global
                // offset. Fails iff the accumulated total would overflow
                // the 2-byte inline offset field.
                match pyre_jit_trace::state::append_liveness(&liveness_fragment) {
                    Some(base) => {
                        for (patch_offset, local_offset) in local_offsets {
                            let global = base.saturating_add(local_offset);
                            assembler.patch_live_offset(patch_offset, global);
                        }
                    }
                    None => has_abort = true,
                }
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

        let mut jitcode = assembler.finish();

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
            jitcode,
            metadata,
            has_abort,
        }
    }

    /// RPython: CodeWriter.make_jitcodes() — compile all reachable graphs.
    ///
    /// For pyre, JitCodes are compiled lazily per-CodeObject (not AOT).
    /// This method compiles a single CodeObject and caches the result.
    pub fn make_jitcode(&self, code: &CodeObject, w_code: *const ()) -> PyJitCode {
        self.transform_graph_to_jitcode(code, w_code)
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

/// Backward jump target: next_instr - delta.
fn jump_target_backward(next_instr: usize, delta: usize) -> usize {
    next_instr.saturating_sub(delta)
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
) -> &'static PyJitCode {
    let key = code as *const CodeObject as usize;
    JITCODE_CACHE.with(|cell| {
        let cache = unsafe { &mut *cell.get() };
        if !cache.contains_key(&key) {
            let pyjitcode = writer.make_jitcode(code, w_code);
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
pub fn ensure_jitcode_for(code: &pyre_interpreter::CodeObject, w_code: *const ()) {
    let writer = CodeWriter::new(
        crate::call_jit::bh_call_fn,
        crate::call_jit::bh_load_global_fn,
        crate::call_jit::bh_compare_fn,
        crate::call_jit::bh_binary_op_fn,
        crate::call_jit::bh_box_int_fn,
        crate::call_jit::bh_truth_fn,
        crate::call_jit::bh_load_const_fn,
        crate::call_jit::bh_store_subscr_fn,
        crate::call_jit::bh_build_list_fn,
    );
    let _ = get_jitcode(code, w_code, &writer);
}

/// jitcode.py:18: `jitcode.jitdriver_sd is not None`.
pub fn is_portal(code: &pyre_interpreter::CodeObject, w_code: *const ()) -> bool {
    ensure_jitcode_for(code, w_code);
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
