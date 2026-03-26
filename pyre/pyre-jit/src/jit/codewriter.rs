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
use pyre_bytecode::bytecode::{CodeObject, Instruction, OpArgState};

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
    pub call_fn: extern "C" fn(i64, i64, i64) -> i64,
    pub load_global_fn: extern "C" fn(i64, i64) -> i64,
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
        call_fn: extern "C" fn(i64, i64, i64) -> i64,
        load_global_fn: extern "C" fn(i64, i64) -> i64,
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
        // RPython parity: Python bytecode locals / stack hold PyObject refs.
        // Keep object values in ref registers, and reserve a separate int
        // scratch bank for opcode immediates, truth flags, and the frame ptr.
        let obj_tmp0 = nlocals as u16;
        let obj_tmp1 = (nlocals + 1) as u16;
        let arg_regs_start = (nlocals + 2) as u16; // up to CALL 4
        let null_ref_reg = (nlocals + 6) as u16; // permanently zero / null

        let int_tmp0 = 0u16;
        let int_tmp1 = 1u16;
        let op_code_reg = 2u16;
        let frame_reg = 3u16;

        // RPython: self.assembler = Assembler()
        let mut assembler = JitCodeBuilder::default();

        // RPython regalloc.py: keep kind-separated register files.
        assembler.ensure_r_regs(null_ref_reg + 1);
        assembler.ensure_i_regs(frame_reg + 1);

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

        // RPython flatten.py: pre-create labels for each block.
        // Python bytecodes are linear, so each instruction index gets a label.
        let num_instrs = code.instructions.len();
        let mut labels: Vec<u16> = Vec::with_capacity(num_instrs);
        for _ in 0..num_instrs {
            labels.push(assembler.new_label());
        }

        // pc_map: Python PC → JitCode byte offset
        let mut pc_map = vec![0usize; num_instrs];

        // RPython: flatten_graph() walks blocks and emits instruction tuples.
        // RPython: assembler.assemble(ssarepr, jitcode, num_regs) emits bytecodes.
        // For pyre, we combine both steps: walk Python bytecodes and emit
        // JitCode bytecodes directly.
        let mut arg_state = OpArgState::default();

        for py_pc in 0..num_instrs {
            // RPython flatten.py: Label(block) at block entry
            assembler.mark_label(labels[py_pc]);
            pc_map[py_pc] = assembler.current_pos();

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

                // RPython flatten.py: input args → registers 0..n-1
                Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    assembler.push_r(reg);
                }

                Instruction::StoreFast { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    assembler.pop_r(reg);
                }

                Instruction::LoadSmallInt { i } => {
                    // RPython parity: Python stack carries boxed object refs.
                    // Box the raw integer via w_int_new and keep the result in
                    // the ref register file.
                    let val = i.get(op_arg) as u32 as i64;
                    assembler.load_const_i_value(int_tmp0, val);
                    assembler.call_ref_typed(
                        box_int_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                        obj_tmp0,
                    );
                    assembler.push_r(obj_tmp0);
                }

                Instruction::LoadConst { consti } => {
                    // RPython assembler.py: emit_const() stores constants in
                    // the jitcode constant pool. For pyre, load from the
                    // frame's code object at runtime via a helper call.
                    let idx = consti.get(op_arg).as_usize();
                    assembler.load_const_i_value(int_tmp0, idx as i64);
                    assembler.call_ref_typed(
                        load_const_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::int(frame_reg),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        obj_tmp0,
                    );
                    assembler.push_r(obj_tmp0);
                }

                // Superinstruction: two consecutive LoadFastBorrow
                Instruction::LoadFastBorrowLoadFastBorrow { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let reg_a = u32::from(pair.idx_1()) as u16;
                    let reg_b = u32::from(pair.idx_2()) as u16;
                    assembler.push_r(reg_a);
                    assembler.push_r(reg_b);
                }

                // STORE_SUBSCR: stack [value, obj, key] → obj[key] = value
                Instruction::StoreSubscr => {
                    assembler.pop_r(obj_tmp1); // key
                    assembler.pop_r(obj_tmp0); // obj
                    // value is still on stack; pop to arg_regs_start
                    assembler.pop_r(arg_regs_start);
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
                    assembler.pop_discard();
                }

                Instruction::PushNull => {
                    assembler.push_r(null_ref_reg);
                }

                // RPython jtransform.py: rewrite_op_int_add etc.
                Instruction::BinaryOp { op } => {
                    let op_val = op.get(op_arg) as u32;
                    assembler.pop_r(obj_tmp1); // rhs
                    assembler.pop_r(obj_tmp0); // lhs
                    // RPython: residual_call for generic binary dispatch,
                    // returning a boxed PyObject ref.
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
                    assembler.push_r(obj_tmp0);
                }

                // RPython jtransform.py: rewrite_op_int_lt, optimize_goto_if_not
                Instruction::CompareOp { opname } => {
                    let op_val = opname.get(op_arg) as u32;
                    assembler.pop_r(obj_tmp1); // rhs
                    assembler.pop_r(obj_tmp0); // lhs
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
                    assembler.push_r(obj_tmp0);
                }

                // RPython jtransform.py: optimize_goto_if_not → goto_if_not
                // Stack top is a Python object (True/False). We need to
                // convert to raw 0/1 for branch_reg_zero.
                Instruction::PopJumpIfFalse { delta } => {
                    let target_py_pc = jump_target_forward(
                        code,
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    assembler.pop_r(obj_tmp0);
                    // truth_fn: PyObjectRef → 0/1 (truthiness check)
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
                    // Invert: branch_reg_zero branches if zero, but we want
                    // branch if nonzero. Emit: tmp0 = (tmp0 == 0), then
                    // branch_reg_zero.
                    assembler.pop_r(obj_tmp0);
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

                // RPython flatten.py: int_return / ref_return
                Instruction::ReturnValue => {
                    // RPython bhimpl_ref_return: pop return value and
                    // emit BC_REF_RETURN so the blackhole returns cleanly
                    // (not BC_ABORT which marks a failed/unsupported path).
                    assembler.pop_r(obj_tmp0);
                    assembler.ref_return(obj_tmp0);
                }

                // RPython jtransform.py: rewrite_op_direct_call (residual)
                Instruction::LoadGlobal { namei } => {
                    let raw_namei = namei.get(op_arg) as usize as i64;
                    assembler.load_const_i_value(int_tmp0, raw_namei);
                    assembler.call_may_force_ref_typed(
                        load_global_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::int(frame_reg),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        obj_tmp0,
                    );
                    // LOAD_GLOBAL with (namei >> 1) & 1: push NULL first
                    if raw_namei & 1 != 0 {
                        assembler.push_r(null_ref_reg);
                    }
                    assembler.push_r(obj_tmp0);
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
                    // Pop args in reverse into scratch registers
                    // For now, support up to 3 args (covers fib_recursive CALL 1)
                    for i in (0..nargs).rev() {
                        assembler.pop_r(arg_regs_start + i as u16);
                    }
                    assembler.pop_r(obj_tmp1); // callable
                    assembler.pop_discard(); // NULL

                    // call_fn(callable, arg0, frame_ptr) → result
                    // RPython: bhimpl_recursive_call_i(jdindex, greens, reds)
                    let mut call_args =
                        vec![majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1)];
                    for i in 0..nargs {
                        call_args.push(majit_metainterp::jitcode::JitCallArg::reference(
                            arg_regs_start + i as u16,
                        ));
                    }
                    call_args.push(majit_metainterp::jitcode::JitCallArg::int(frame_reg));
                    assembler.call_may_force_ref_typed(call_fn_idx, &call_args, obj_tmp0);
                    assembler.push_r(obj_tmp0);
                }

                // Python 3.13: ToBool converts TOS to bool before branch.
                // No-op in JitCode: the value is already truthy/falsy and
                // the following PopJumpIfFalse guards on it.
                Instruction::ToBool => {}

                // RPython bhimpl_int_neg: -obj via binary_op(0, obj, NB_SUBTRACT)
                Instruction::UnaryNegative => {
                    assembler.pop_r(obj_tmp0);
                    assembler.load_const_i_value(int_tmp0, 0);
                    assembler.call_may_force_ref_typed(
                        box_int_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                        obj_tmp1,
                    );
                    assembler.load_const_i_value(int_tmp0, 11); // NB_SUBTRACT
                    assembler.call_may_force_ref_typed(
                        binary_op_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        obj_tmp0,
                    );
                    assembler.push_r(obj_tmp0);
                }

                // RPython: same as JumpBackward (no interrupt check in JIT)
                Instruction::JumpBackwardNoInterrupt { delta } => {
                    let target_py_pc =
                        skip_caches(code, py_pc + 1).saturating_sub(delta.get(op_arg).as_usize());
                    if target_py_pc < num_instrs {
                        assembler.jump(labels[target_py_pc]);
                    }
                }

                // RPython bhimpl_newlist: build list from N items on stack.
                Instruction::BuildList { count } => {
                    let argc = count.get(op_arg) as usize;
                    // Pop items into registers (reverse order like CALL).
                    for i in (0..argc.min(2)).rev() {
                        assembler.pop_r(arg_regs_start + i as u16);
                    }
                    // Discard extra items beyond 2 (helper supports 0-2).
                    for _ in 2..argc {
                        assembler.pop_discard();
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
                    assembler.push_r(obj_tmp0);
                }

                // Exception handling: residual calls to frame helpers.
                // RPython blackhole.py handles exceptions via dedicated
                // bhimpl_* functions. In pyre, we delegate to the frame's
                // exception machinery via call_fn.
                Instruction::RaiseVarargs { argc } => {
                    let n = argc.get(op_arg) as i64;
                    if n >= 1 {
                        assembler.pop_r(obj_tmp0); // exception value
                    }
                    // Signal abort — exception raised, blackhole will
                    // catch via LeaveFrame and check exception state.
                    assembler.abort();
                }

                Instruction::PushExcInfo => {
                    // TOS is the exception. Push exc_info (type, value, tb).
                    // For blackhole: peek TOS, push (type, value) pair.
                    // Simplified: duplicate TOS (exception value stays).
                    assembler.dup_stack();
                }

                Instruction::CheckExcMatch => {
                    // TOS = exception type to match, TOS1 = raised exception.
                    // Pop type, compare, push boolean result.
                    assembler.pop_r(obj_tmp1); // match type
                    assembler.pop_r(obj_tmp0); // exception
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
                    assembler.push_r(obj_tmp0);
                }

                Instruction::PopExcept => {
                    // Clear current exception — no-op in blackhole
                    // (exception state is in TLS, cleared by interpreter).
                }

                Instruction::Reraise { .. } => {
                    // Re-raise current exception.
                    assembler.abort();
                }

                Instruction::Copy { i } => {
                    let d = i.get(op_arg) as usize;
                    if d == 1 {
                        assembler.dup_stack();
                    } else {
                        // COPY d: duplicate the d-th item from TOS.
                        // copy_from_bottom(stack_depth - d) copies the item
                        // at position (stack_depth - d) to TOS.
                        // Not easily expressible with current JitCode ops;
                        // use pop to temp, push back, push temp.
                        // For now, emit no-op and let interpreter handle.
                        assembler.abort();
                    }
                }

                Instruction::LoadName { .. }
                | Instruction::StoreName { .. }
                | Instruction::MakeFunction { .. } => {
                    // Module-level instructions: never appear in traced
                    // function bodies (traces start at loop backedges
                    // inside functions). Safe to abort.
                    assembler.abort();
                }

                // Truly unsupported: abort to interpreter fallback.
                other => {
                    if majit_metainterp::majit_log_enabled() {
                        eprintln!("[codewriter] unsupported instruction: {:?}", other);
                    }
                    assembler.abort();
                }
            }
        }

        // RPython flatten.py parity: every code path ends with an explicit
        // return/raise/goto/unreachable. No end-of-code sentinel needed —
        // falling off the end is unreachable if all bytecodes are covered.

        // RPython: assembler.assemble() → jitcode via make_jitcode()
        let jitcode = assembler.finish();
        let has_abort = jitcode.code.contains(&13 /* BC_ABORT */);
        PyJitCode {
            jitcode,
            pc_map,
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
/// Must match pyre-interpreter/opcode_step.rs:jump_target_forward.
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

/// Match pyre-interpreter/opcode_step.rs:skip_caches.
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

/// Compute JitCode PC for a Python loop_header_pc.
/// The interpreter's loop_header_pc includes Cache skip, but the
/// codewriter's label targets don't. Search pc_map backwards from
/// loop_header_pc to find the matching JitCode offset.
pub fn jitcode_pc_for_loop_header(pc_map: &[usize], loop_header_pc: usize) -> Option<usize> {
    // pc_map[loop_header_pc] is the JitCode PC for that code unit.
    // But the codewriter may have placed the label at a slightly earlier
    // code unit (before Cache instructions). Try exact match first,
    // then search backwards.
    for offset in 0..4 {
        let py_pc = loop_header_pc.checked_sub(offset)?;
        let jitcode_pc = *pc_map.get(py_pc)?;
        if jitcode_pc > 0 || py_pc == 0 {
            return Some(jitcode_pc);
        }
    }
    pc_map.get(loop_header_pc).copied()
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
/// RPython: CallControl.get_jitcode() deduplicates per graph.
/// pyre: deduplicate per CodeObject pointer (thread-local cache).
pub fn get_or_compile_jitcode(code: &CodeObject, writer: &CodeWriter) -> &'static PyJitCode {
    let key = code as *const CodeObject as usize;
    JITCODE_CACHE.with(|cell| {
        let cache = unsafe { &mut *cell.get() };
        if !cache.contains_key(&key) {
            let pyjitcode = writer.make_jitcode(code);
            cache.insert(key, pyjitcode);
        }
        let entry = cache.get(&key).unwrap();
        // SAFETY: thread-local cache lives for thread lifetime
        unsafe { &*(entry as *const PyJitCode) }
    })
}
