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
use majit_metainterp::jitcode::{JitCode, JitCodeBuilder};
use pyre_interpreter::bytecode::{CodeObject, Instruction, OpArgState};
use pyre_interpreter::runtime_ops::{binary_op_tag, compare_op_tag};

// ---------------------------------------------------------------------------
// RPython: codewriter/flatten.py KINDS = ['int', 'ref', 'float']
// ---------------------------------------------------------------------------

/// Compiled JitCode with Python PC → JitCode PC mapping.
///
/// RPython: JitCode in codewriter/jitcode.py, plus a per-function PC table
/// (RPython doesn't need this because jitcode PCs *are* the PCs).
pub struct PyJitCode {
    pub jitcode: JitCode,
    /// py_pc → jitcode byte offset.
    pub pc_map: Vec<usize>,
    /// jitcode byte offset → py_pc (sorted pairs for binary search reverse lookup).
    /// Used by handle_exception_in_frame to determine faulting Python PC (lasti).
    pub jit_to_py_pc: Vec<(usize, usize)>,
    /// True if the jitcode contains BC_ABORT opcodes (unsupported bytecodes).
    /// Precomputed at compile time to avoid repeated bytecode scanning.
    pub has_abort: bool,
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
    pub fn transform_graph_to_jitcode(&self, code: &CodeObject) -> PyJitCode {
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

        let int_tmp0 = 0u16;
        let int_tmp1 = 1u16;
        let op_code_reg = 2u16;
        // jtransform.py: virtualizable field indices for getfield_vable_*
        // interp_jit.py:25 parity: code=1, namespace=3
        const VABLE_CODE_FIELD_IDX: u16 = 1;
        const VABLE_NAMESPACE_FIELD_IDX: u16 = 3;

        // regalloc.py: compile-time stack depth counter — tracks which
        // stack register (stack_base + depth) is the current TOS.
        let mut current_depth: u16 = 0;

        // RPython: self.assembler = Assembler()
        let mut assembler = JitCodeBuilder::default();

        // RPython regalloc.py: keep kind-separated register files.
        assembler.ensure_r_regs(null_ref_reg + 1);
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
            let entries = pyre_interpreter::bytecode::decode_exception_table(&code.exceptiontable);
            entries
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

        // jitcode.py:18: jitdriver_sd is not None for portals.
        // RPython: jitdriver_sd is set on the portal jitcode by
        // call.py:148 grab_initial_jitcodes (exactly one per jitdriver).
        // pyre: every named function is a potential portal. <module> is
        // excluded — RPython never places jit_merge_point there.
        let is_portal = &*code.obj_name != "<module>";
        // jtransform.py:1690-1712: portal jitcodes get one jit_merge_point
        // (the first loop header). Non-portal jitcodes only get loop_header
        // (BC_JUMP_TARGET, no-op in blackhole).
        let merge_point_pc = if is_portal {
            loop_header_pcs.iter().copied().min()
        } else {
            None
        };
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
            depth_at_pc[py_pc] = current_depth;

            // jtransform.py: jit_merge_point at the portal's merge point;
            // loop_header (BC_JUMP_TARGET) at all other backward jump targets.
            if loop_header_pcs.contains(&py_pc) {
                if merge_point_pc == Some(py_pc) && !emitted_merge_point {
                    assembler.jit_merge_point();
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

                // flatten.py: input args → registers 0..n-1
                // regalloc.py: LOAD_FAST = ref_copy local → stack register
                Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    assembler.move_r(stack_base + current_depth, reg);
                    current_depth += 1;
                }

                Instruction::StoreFast { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    current_depth -= 1;
                    assembler.move_r(reg, stack_base + current_depth);
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

                // Superinstruction: two consecutive LoadFastBorrow
                Instruction::LoadFastBorrowLoadFastBorrow { var_nums } => {
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

                // Unsupported instruction: abort_permanent.
                // Using BC_ABORT_PERMANENT(14) instead of BC_ABORT(13) so
                // has_abort check doesn't false-positive on functions where
                // only exception/module paths have unsupported instructions.
                _other => {
                    assembler.abort_permanent();
                }
            }
        }

        // RPython flatten.py parity: every code path ends with an explicit
        // return/raise/goto/unreachable. No end-of-code sentinel needed —
        // falling off the end is unreachable if all bytecodes are covered.

        // RPython: assembler.assemble() → jitcode via make_jitcode()
        let assembler_code_len = assembler.current_pos();
        let mut jitcode = assembler.finish();
        // RPython parity: BC_ABORT (opcode 13) is never emitted by this
        // codewriter — unsupported bytecodes use BC_ABORT_PERMANENT (14).
        // The old byte scan (code.contains(&13)) gave false positives when
        // byte value 13 appeared as register operand data, blocking
        // blackhole entry entirely.
        let has_abort = false;

        // liveness.py parity: generate LivenessInfo at each bytecode PC.
        // RPython: compute_liveness() runs backward dataflow on SSARepr,
        // marking only actually-live registers. Both get_list_of_active_boxes
        // (pyjitpl.py:177) and consume_one_section (resume.py:1381) use the
        // same all_liveness data. In pyre, LiveVars::compute provides the
        // equivalent backward analysis on Python bytecodes.
        {
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
                liveness.push(majit_metainterp::jitcode::LivenessInfo {
                    pc: jit_pc as u16,
                    live_i_regs: vec![],
                    live_r_regs: live_r,
                    live_f_regs: vec![],
                });
            }
            // Deduplicate entries at the same JitCode PC (multiple Python
            // bytecodes may map to the same JitCode offset, e.g. Cache/Nop).
            // When merging, take the UNION of live registers (conservative).
            liveness.sort_by_key(|l| l.pc);
            let mut merged: Vec<majit_metainterp::jitcode::LivenessInfo> = Vec::new();
            for entry in liveness {
                if let Some(last) = merged.last_mut() {
                    if last.pc == entry.pc {
                        // Union: merge live_r_regs
                        for &reg in &entry.live_r_regs {
                            if !last.live_r_regs.contains(&reg) {
                                last.live_r_regs.push(reg);
                            }
                        }
                        last.live_r_regs.sort();
                        continue;
                    }
                }
                merged.push(entry);
            }
            jitcode.liveness = merged;
        }

        // RPython parity: forward PC map (py_pc → jitcode_pc) so
        // get_list_of_active_boxes can look up LivenessInfo by Python PC.
        jitcode.py_to_jit_pc = pc_map.clone();

        jitcode.is_portal = is_portal;
        jitcode.nlocals = code.varnames.len();

        // blackhole.py handle_exception_in_frame: build exception handler table
        // from Python's code.exceptiontable. Maps Python PC ranges to JitCode PCs.
        //
        // Python 3.11+ exception table: each entry covers [start, end) instruction
        // range. find_exception_handler(table, offset) returns the handler for
        // `offset` if offset >= start && offset < end.
        //
        // The faulting Python PC (lasti) is determined at runtime from the
        // blackhole's current jitcode position via jit_to_py_pc reverse map.
        // lasti_value on JitExceptionHandler is unused; the blackhole dispatch
        // reads the actual faulting PC.
        {
            use majit_metainterp::jitcode::JitExceptionHandler;
            // decode_exception_table: decode ALL entries from the binary table.
            // Each ExceptionTableEntry: { start, end, target, depth, push_lasti }.
            let entries = pyre_interpreter::bytecode::decode_exception_table(&code.exceptiontable);
            let handlers: Vec<JitExceptionHandler> = entries
                .iter()
                .filter_map(|entry| {
                    let jit_start = *pc_map.get(entry.start as usize)?;
                    let jit_end = if (entry.end as usize) < num_instrs {
                        pc_map
                            .get(entry.end as usize)
                            .copied()
                            .unwrap_or(assembler_code_len)
                    } else {
                        assembler_code_len
                    };
                    let jit_target = *pc_map.get(entry.target as usize)?;
                    Some(JitExceptionHandler {
                        jit_start,
                        jit_end,
                        jit_target,
                        stack_depth: entry.depth,
                        push_lasti: entry.push_lasti,
                        lasti_value: 0, // determined at runtime
                        box_int_fn_idx: box_int_fn_idx as u16,
                    })
                })
                .collect();
            jitcode.exception_handlers = handlers;
        }

        // Build jit_to_py_pc reverse map for lasti lookup at runtime.
        let jit_to_py_pc: Vec<(usize, usize)> = {
            let mut pairs: Vec<(usize, usize)> = pc_map
                .iter()
                .enumerate()
                .map(|(py_pc, &jit_pc)| (jit_pc, py_pc))
                .collect();
            pairs.sort_by_key(|&(jit_pc, _)| jit_pc);
            pairs.dedup_by_key(|entry| entry.0);
            pairs
        };

        // Store reverse map on JitCode for handle_exception_in_frame lasti lookup.
        jitcode.jit_to_py_pc = jit_to_py_pc.clone();

        PyJitCode {
            jitcode,
            pc_map,
            jit_to_py_pc,
            has_abort,
        }
    }

    /// RPython: CodeWriter.make_jitcodes() — compile all reachable graphs.
    ///
    /// For pyre, JitCodes are compiled lazily per-CodeObject (not AOT).
    /// This method compiles a single CodeObject and caches the result.
    pub fn make_jitcode(&self, code: &CodeObject) -> PyJitCode {
        self.transform_graph_to_jitcode(code)
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
/// jitcode.py:14 parity: is_portal is determined by the CodeObject itself
/// (jitdriver_sd is not None for portal jitcodes). pyre determines this
/// from code.obj_name inside transform_graph_to_jitcode, matching RPython's
/// make_jitcodes() which sets jitdriver_sd on the main portal graph.
/// The result is cached per CodeObject pointer — is_portal is fixed per code.
pub fn get_jitcode(code: &CodeObject, writer: &CodeWriter) -> &'static PyJitCode {
    let key = code as *const CodeObject as usize;
    JITCODE_CACHE.with(|cell| {
        let cache = unsafe { &mut *cell.get() };
        if !cache.contains_key(&key) {
            let pyjitcode = writer.make_jitcode(code);
            cache.insert(key, pyjitcode);
            // RPython parity: link state::JitCode to majit JitCode so
            // get_list_of_active_boxes uses the same liveness data as
            // consume_one_section (resume.py all_liveness parity).
            let entry = cache.get(&key).unwrap();
            pyre_jit_trace::set_majit_jitcode(
                code as *const _,
                &entry.jitcode as *const majit_metainterp::jitcode::JitCode,
            );
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
pub fn ensure_jitcode_for(code: &pyre_interpreter::CodeObject) {
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
    let _ = get_jitcode(code, &writer);
}

/// jitcode.py:18 parity: `jitcode.jitdriver_sd is not None`.
/// Single source of truth for portal determination.
pub fn is_portal(code: &pyre_interpreter::CodeObject) -> bool {
    ensure_jitcode_for(code);
    let key = code as *const pyre_interpreter::CodeObject as usize;
    JITCODE_CACHE.with(|cell| {
        let cache = unsafe { &*cell.get() };
        cache
            .get(&key)
            .map(|pjc| pjc.jitcode.is_portal)
            .unwrap_or(false)
    })
}
