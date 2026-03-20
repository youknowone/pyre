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
use majit_meta::jitcode::{JitCode, JitCodeBuilder};
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
    /// Used by resume_in_blackhole to locate the resume point after guard
    /// failure.
    pub pc_map: Vec<usize>,
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
}

impl CodeWriter {
    pub fn new(
        call_fn: extern "C" fn(i64, i64, i64) -> i64,
        load_global_fn: extern "C" fn(i64, i64) -> i64,
        compare_fn: extern "C" fn(i64, i64, i64) -> i64,
        binary_op_fn: extern "C" fn(i64, i64, i64) -> i64,
    ) -> Self {
        Self {
            call_fn,
            load_global_fn,
            compare_fn,
            binary_op_fn,
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
        // Scratch registers after locals:
        //   tmp0 = nlocals, tmp1 = nlocals+1, op_code_reg = nlocals+2,
        //   frame_reg = nlocals+3
        let tmp0 = nlocals as u16;
        let tmp1 = (nlocals + 1) as u16;
        let op_code_reg = (nlocals + 2) as u16;
        let frame_reg = (nlocals + 3) as u16;

        // RPython: self.assembler = Assembler()
        let mut assembler = JitCodeBuilder::default();

        // Register helper function pointers
        // RPython: CallControl manages fn addresses; assembler.finished()
        // writes them into callinfocollection.
        let call_fn_idx = assembler.add_fn_ptr(self.call_fn as *const ());
        let load_global_fn_idx = assembler.add_fn_ptr(self.load_global_fn as *const ());
        let compare_fn_idx = assembler.add_fn_ptr(self.compare_fn as *const ());
        let binary_op_fn_idx = assembler.add_fn_ptr(self.binary_op_fn as *const ());

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
                Instruction::LoadFast { var_num }
                | Instruction::LoadFastBorrow { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    assembler.push_i(reg);
                }

                Instruction::StoreFast { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    assembler.pop_i(reg);
                }

                Instruction::LoadSmallInt { i } => {
                    let val = i.get(op_arg) as u32 as i64;
                    assembler.load_const_i_value(tmp0, val);
                    assembler.push_i(tmp0);
                }

                Instruction::LoadConst { consti } => {
                    // RPython assembler.py: emit_const() deduplicates constants.
                    // For now, push 0 placeholder; blackhole caller populates
                    // from frame.
                    let _idx = consti.get(op_arg);
                    assembler.load_const_i_value(tmp0, 0);
                    assembler.push_i(tmp0);
                }

                Instruction::PopTop => {
                    assembler.pop_discard();
                }

                Instruction::PushNull => {
                    assembler.load_const_i_value(tmp0, 0);
                    assembler.push_i(tmp0);
                }

                // RPython jtransform.py: rewrite_op_int_add etc.
                Instruction::BinaryOp { op } => {
                    let op_val = op.get(op_arg) as u32;
                    assembler.pop_i(tmp1); // rhs
                    assembler.pop_i(tmp0); // lhs
                    // RPython: residual_call for generic binary dispatch
                    assembler.load_const_i_value(op_code_reg, op_val as i64);
                    assembler.call_may_force_int(
                        binary_op_fn_idx,
                        &[tmp0, tmp1, op_code_reg],
                        tmp0,
                    );
                    assembler.push_i(tmp0);
                }

                // RPython jtransform.py: rewrite_op_int_lt, optimize_goto_if_not
                Instruction::CompareOp { opname } => {
                    let op_val = opname.get(op_arg) as u32;
                    assembler.pop_i(tmp1); // rhs
                    assembler.pop_i(tmp0); // lhs
                    assembler.load_const_i_value(op_code_reg, op_val as i64);
                    assembler.call_may_force_int(
                        compare_fn_idx,
                        &[tmp0, tmp1, op_code_reg],
                        tmp0,
                    );
                    assembler.push_i(tmp0);
                }

                // RPython jtransform.py: optimize_goto_if_not → goto_if_not
                Instruction::PopJumpIfFalse { delta } => {
                    let target_py_pc = jump_target_forward(
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    assembler.pop_i(tmp0);
                    if target_py_pc < num_instrs {
                        assembler.branch_reg_zero(tmp0, labels[target_py_pc]);
                    }
                }

                Instruction::PopJumpIfTrue { delta } => {
                    let target_py_pc = jump_target_forward(
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    // Invert: branch_reg_zero branches if zero, but we want
                    // branch if nonzero. Emit: tmp0 = (tmp0 == 0), then
                    // branch_reg_zero.
                    assembler.pop_i(tmp0);
                    assembler.load_const_i_value(tmp1, 0);
                    assembler.record_binop_i(tmp0, OpCode::IntEq, tmp0, tmp1);
                    if target_py_pc < num_instrs {
                        assembler.branch_reg_zero(tmp0, labels[target_py_pc]);
                    }
                }

                // RPython flatten.py: goto Label
                Instruction::JumpForward { delta } => {
                    let target_py_pc = jump_target_forward(
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    if target_py_pc < num_instrs {
                        assembler.jump(labels[target_py_pc]);
                    }
                }

                Instruction::JumpBackward { delta } => {
                    let target_py_pc = jump_target_backward(
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    if target_py_pc < num_instrs {
                        assembler.jump(labels[target_py_pc]);
                    }
                }

                // RPython flatten.py: int_return / ref_return
                Instruction::ReturnValue => {
                    // Pop return value into register 0.
                    // The blackhole caller reads registers_i[0] after run().
                    assembler.pop_i(tmp0);
                    assembler.move_i(0, tmp0);
                    assembler.abort();
                }

                // RPython jtransform.py: rewrite_op_direct_call (residual)
                Instruction::LoadGlobal { namei } => {
                    let raw_namei = namei.get(op_arg) as usize as i64;
                    assembler.load_const_i_value(tmp0, raw_namei);
                    assembler.call_may_force_int(
                        load_global_fn_idx,
                        &[frame_reg, tmp0],
                        tmp0,
                    );
                    // LOAD_GLOBAL with (namei >> 1) & 1: push NULL first
                    if raw_namei & 1 != 0 {
                        assembler.load_const_i_value(tmp1, 0);
                        assembler.push_i(tmp1);
                    }
                    assembler.push_i(tmp0);
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
                    let arg_regs_start = (nlocals + 4) as u16;
                    for i in (0..nargs).rev() {
                        assembler.pop_i(arg_regs_start + i as u16);
                    }
                    assembler.pop_i(tmp1); // callable
                    assembler.pop_discard(); // NULL

                    // call_fn(callable, arg0, frame_ptr) → result
                    // RPython: bhimpl_recursive_call_i(jdindex, greens, reds)
                    let mut call_args: Vec<u16> = vec![tmp1];
                    for i in 0..nargs {
                        call_args.push(arg_regs_start + i as u16);
                    }
                    call_args.push(frame_reg);
                    assembler.call_may_force_int(
                        call_fn_idx,
                        &call_args,
                        tmp0,
                    );
                    assembler.push_i(tmp0);
                }

                // Unsupported: abort to interpreter fallback
                _ => {
                    assembler.abort();
                }
            }
        }

        // End of code: implicit abort
        assembler.abort();

        // RPython: assembler.assemble() → jitcode via make_jitcode()
        PyJitCode {
            jitcode: assembler.finish(),
            pc_map,
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

/// Forward jump target: next_instr + delta.
fn jump_target_forward(num_instrs: usize, next_instr: usize, delta: usize) -> usize {
    let target = next_instr + delta;
    target.min(num_instrs)
}

/// Backward jump target: next_instr - delta.
fn jump_target_backward(next_instr: usize, delta: usize) -> usize {
    next_instr.saturating_sub(delta)
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
pub fn get_or_compile_jitcode(
    code: &CodeObject,
    writer: &CodeWriter,
) -> &'static PyJitCode {
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
