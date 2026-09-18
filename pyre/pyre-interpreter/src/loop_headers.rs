//! Bytecode loop-header analysis and the portal-nameable predicates.
//!
//! Upstream fixes loop headers while the codewriter builds the graph's
//! JitCode (`rpython/jit/codewriter/jtransform.py handle_jit_marker__loop_header`); here the
//! scan is interpreter-side bytecode analysis so the accessor is nameable
//! by the fnaddr table.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rustpython_compiler_core::bytecode::{Instruction, OpArg, OpArgState};

use crate::CodeObject;
use crate::pyopcode::{jump_target_forward, skip_caches};

/// Single-source-of-truth backward-jump target calculation used by both
/// the loop-header pre-scan (`pypy/module/pypyjit/interp_jit.py jump_absolute`) and
/// the emitter (`jtransform.py` `handle_jit_marker__loop_header`).
///
/// Returns the target PC for `JumpBackward` (with `skip_caches` on the
/// next-PC base) and `JumpBackwardNoInterrupt` (direct `py_pc + 1 - delta`
/// arithmetic to match the interpreter's dispatch in pyopcode.rs).
/// Returns `None` for any non-backward-jump opcode.
pub fn backward_jump_target(
    code: &CodeObject,
    py_pc: usize,
    instr: Instruction,
    op_arg: OpArg,
) -> Option<usize> {
    match instr {
        Instruction::JumpBackward { delta } => Some(
            skip_caches(&code.instructions, py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()),
        ),
        Instruction::JumpBackwardNoInterrupt { delta } => {
            Some((py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
        }
        _ => None,
    }
}

/// Control-flow successors of every code-unit index: fall-through (except
/// after an unconditional transfer), forward and backward jump targets, and
/// the exception edge from each protected pc to its handler landing.  Built in
/// one sequential decode pass so an `EXTENDED_ARG` prefix folds into the
/// following opcode's argument, matching the edge model of
/// `find_branch_target_pcs`.
pub fn code_successors(code: &CodeObject) -> Vec<Vec<usize>> {
    let num_instrs = code.instructions.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); num_instrs];
    let mut scan_state = OpArgState::default();
    for pc in 0..num_instrs {
        let (instr, op_arg) = scan_state.get(code.instructions[pc]);
        if let Some(target) = backward_jump_target(code, pc, instr, op_arg) {
            if target < num_instrs {
                succ[pc].push(target);
            }
        }
        let forward_delta = match instr {
            Instruction::PopJumpIfFalse { delta }
            | Instruction::PopJumpIfTrue { delta }
            | Instruction::PopJumpIfNone { delta }
            | Instruction::PopJumpIfNotNone { delta }
            | Instruction::JumpForward { delta }
            | Instruction::ForIter { delta } => Some(delta.get(op_arg).as_usize()),
            _ => None,
        };
        if let Some(delta) = forward_delta {
            let target = jump_target_forward(&code.instructions, pc + 1, delta);
            if target < num_instrs {
                succ[pc].push(target);
            }
        }
        let terminates = matches!(
            instr,
            Instruction::JumpForward { .. }
                | Instruction::JumpBackward { .. }
                | Instruction::JumpBackwardNoInterrupt { .. }
                | Instruction::ReturnValue
                | Instruction::RaiseVarargs { .. }
                | Instruction::Reraise { .. }
        );
        if !terminates {
            let fallthrough = pc + 1;
            if fallthrough < num_instrs {
                succ[pc].push(fallthrough);
            }
        }
    }
    for entry in crate::pycode::decode_exceptiontable(&code.exceptiontable) {
        let start = entry.start as usize / 2;
        let end = (entry.end as usize / 2).min(num_instrs);
        let handler = entry.target as usize / 2;
        if handler < num_instrs {
            for pc in start..end {
                succ[pc].push(handler);
            }
        }
    }
    succ
}

/// True when `target_pc` dominates `source_pc`: every control-flow path from
/// the code entry (pc 0) to `source_pc` passes through `target_pc`.  Answered
/// by a forward reachability from the entry that never enters `target_pc` — if
/// `source_pc` stays unreachable, `target_pc` dominates it.
pub fn target_dominates(succ: &[Vec<usize>], target_pc: usize, source_pc: usize) -> bool {
    let num_instrs = succ.len();
    if target_pc == source_pc || target_pc == 0 {
        return true;
    }
    if source_pc >= num_instrs {
        return false;
    }
    let mut seen = vec![false; num_instrs];
    let mut stack = vec![0usize];
    seen[0] = true;
    while let Some(pc) = stack.pop() {
        for &next in &succ[pc] {
            if next == target_pc || seen[next] {
                continue;
            }
            seen[next] = true;
            stack.push(next);
        }
    }
    !seen[source_pc]
}

/// Scan `code` for JUMP_BACKWARD targets — the PCs where
/// `transform_graph_to_jitcode` would emit `BC_JUMP_TARGET` and where
/// `jit_merge_point` is evaluated.
///
/// Corresponds to `jtransform.py handle_jit_marker__loop_header`,
/// which walks the flow graph looking for `jit_marker('loop_header', ...)`
/// operations. pyre's "graph" is raw Python bytecode, so the equivalent scan
/// looks for `JUMP_BACKWARD` instructions and resolves their target PCs.
pub fn find_loop_header_pcs(code: &CodeObject) -> HashSet<usize> {
    let num_instrs = code.instructions.len();
    let succ = code_successors(code);
    let mut loop_header_pcs: HashSet<usize> = HashSet::new();
    let mut scan_state = OpArgState::default();
    for scan_pc in 0..num_instrs {
        let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
        if let Some(target) = backward_jump_target(code, scan_pc, scan_instr, scan_arg) {
            if target < num_instrs && target_dominates(&succ, target, scan_pc) {
                loop_header_pcs.insert(target);
            }
        }
    }
    loop_header_pcs
}

thread_local! {
    static LOOP_HEADER_PCS: RefCell<HashMap<usize, Arc<HashSet<usize>>>> =
        RefCell::new(HashMap::new());
    static LOOP_REGION_JIT_SAFE: RefCell<HashMap<(usize, usize), bool>> =
        RefCell::new(HashMap::new());
    static FUNCTION_ENTRY_TRACE_JIT_SAFE: RefCell<HashMap<usize, bool>> =
        RefCell::new(HashMap::new());
}

fn cached_loop_header_pcs(code: &CodeObject) -> Arc<HashSet<usize>> {
    let key = code as *const CodeObject as usize;
    LOOP_HEADER_PCS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(hit) = cache.get(&key) {
            return Arc::clone(hit);
        }
        let computed = Arc::new(find_loop_header_pcs(code));
        cache.insert(key, Arc::clone(&computed));
        computed
    })
}

/// Two-word predicate for the portal arm: is `pc` a loop header of `code`?
///
/// Collapses `CodeWriter::instance()` → `callcontrol()` →
/// `get_loop_header_pcs()` → `Arc::deref` → set membership into one residual
/// the fnaddr table can name. The set and its per-code-address cache stay
/// behind this call.
#[majit_macros::dont_look_inside]
pub fn code_pc_is_loop_header(code: *const CodeObject, pc: usize) -> bool {
    if code.is_null() {
        return false;
    }
    let code = unsafe { &*code };
    cached_loop_header_pcs(code).contains(&pc)
}

fn for_iter_body_op_is_jit_safe(instr: Instruction) -> bool {
    use Instruction as I;
    matches!(
        instr,
        I::LoadFast { .. }
            | I::LoadFastBorrow { .. }
            | I::LoadFastLoadFast { .. }
            | I::LoadFastBorrowLoadFastBorrow { .. }
            | I::LoadFastCheck { .. }
            | I::LoadFastAndClear { .. }
            | I::StoreFast { .. }
            | I::StoreFastLoadFast { .. }
            | I::StoreFastStoreFast { .. }
            | I::LoadConst { .. }
            | I::LoadSmallInt { .. }
            | I::LoadCommonConstant { .. }
            | I::BinaryOp { .. }
            | I::CompareOp { .. }
            | I::IsOp { .. }
            | I::UnaryNegative
            | I::UnaryNot
            | I::UnaryInvert
            | I::ToBool
            | I::Copy { .. }
            | I::Swap { .. }
            | I::PopTop
            | I::PushNull
            | I::Nop
            | I::NotTaken
            | I::PopJumpIfFalse { .. }
            | I::PopJumpIfTrue { .. }
            | I::PopJumpIfNone { .. }
            | I::PopJumpIfNotNone { .. }
            | I::JumpForward { .. }
            | I::JumpBackward { .. }
            | I::JumpBackwardNoInterrupt { .. }
            | I::ReturnValue
            | I::GetIter
            | I::ForIter { .. }
            | I::EndFor
            | I::UnpackSequence { .. }
            | I::UnpackEx { .. }
            | I::BuildTuple { .. }
            | I::BuildSlice { .. }
            | I::BinarySlice
            | I::ContainsOp { .. }
            | I::FormatSimple
            | I::FormatWithSpec
            | I::ConvertValue { .. }
            | I::BuildString { .. }
            | I::GetLen
            | I::PopIter
            | I::DeleteFast { .. }
            | I::LoadDeref { .. }
            | I::LoadAttr { .. }
            | I::LoadSuperAttr { .. }
            | I::Call { .. }
            | I::CallKw { .. }
            | I::CallFunctionEx
            | I::LoadGlobal { .. }
            | I::ImportName { .. }
            | I::ImportFrom { .. }
            | I::Resume { .. }
            | I::BuildList { .. }
            | I::BuildSet { .. }
            | I::BuildMap { .. }
            | I::MakeFunction { .. }
            | I::SetFunctionAttribute { .. }
            | I::RaiseVarargs { .. }
            | I::PushExcInfo
            | I::CheckExcMatch
            | I::PopExcept
            | I::Reraise { .. }
            | I::ExtendedArg
            | I::Cache
    )
}

/// True iff every `FOR_ITER` loop body in `code` is admissible.
pub fn function_entry_trace_is_jit_safe(code: &CodeObject) -> bool {
    use Instruction as I;
    let instructions = &code.instructions;
    let mut arg_state = OpArgState::default();
    for (pc, unit) in instructions.iter().copied().enumerate() {
        let (instr, _) = arg_state.get(unit);
        if matches!(instr, I::ForIter { .. }) && !for_iter_body_is_jit_safe_at(code, pc) {
            return false;
        }
    }
    true
}

/// Natural loop region of `loop_header_pc` as a set of pc ranges.
pub fn loop_region_ranges(
    code: &CodeObject,
    loop_header_pc: usize,
) -> Vec<std::ops::RangeInclusive<usize>> {
    let num_instrs = code.instructions.len();
    if loop_header_pc >= num_instrs {
        return Vec::new();
    }

    let mut backedge_sources: Vec<usize> = Vec::new();
    let mut arg_state = OpArgState::default();
    for (pc, unit) in code.instructions.iter().copied().enumerate() {
        let (instr, op_arg) = arg_state.get(unit);
        if backward_jump_target(code, pc, instr, op_arg) == Some(loop_header_pc) {
            backedge_sources.push(pc);
        }
    }
    if backedge_sources.is_empty() {
        return Vec::new();
    }

    let succ = code_successors(code);
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); num_instrs];
    for (pc, nexts) in succ.iter().enumerate() {
        for &next in nexts {
            pred[next].push(pc);
        }
    }

    let mut in_region = vec![false; num_instrs];
    in_region[loop_header_pc] = true;
    let mut stack: Vec<usize> = Vec::new();
    for source in backedge_sources {
        if source < num_instrs && !in_region[source] {
            in_region[source] = true;
            stack.push(source);
        }
    }
    while let Some(pc) = stack.pop() {
        for &prev in &pred[pc] {
            if !in_region[prev] {
                in_region[prev] = true;
                stack.push(prev);
            }
        }
    }

    let mut ranges: Vec<std::ops::RangeInclusive<usize>> = Vec::new();
    let mut run_start: Option<usize> = None;
    for pc in 0..num_instrs {
        match (in_region[pc], run_start) {
            (true, None) => run_start = Some(pc),
            (false, Some(start)) => {
                ranges.push(start..=pc - 1);
                run_start = None;
            }
            _ => {}
        }
    }
    if let Some(start) = run_start {
        ranges.push(start..=num_instrs - 1);
    }
    ranges
}

/// Apply the FOR_ITER safety gate only to loops inside the natural loop region.
pub fn loop_region_for_iter_bodies_all_jit_safe(code: &CodeObject, loop_header_pc: usize) -> bool {
    use Instruction as I;
    let ranges = loop_region_ranges(code, loop_header_pc);
    if ranges.is_empty() {
        return true;
    }
    let mut scan_state = OpArgState::default();
    for (pc, unit) in code.instructions.iter().copied().enumerate() {
        let (instr, _) = scan_state.get(unit);
        if !ranges.iter().any(|range| range.contains(&pc)) {
            continue;
        }
        if matches!(instr, I::ForIter { .. }) && !for_iter_body_is_jit_safe_at(code, pc) {
            return false;
        }
    }
    true
}

/// Whether `PYRE_FOR_ITER_GATE_DIAG` is set.
pub fn for_iter_gate_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FOR_ITER_GATE_DIAG").is_some())
}

pub fn for_iter_body_is_jit_safe_at(code: &CodeObject, pc: usize) -> bool {
    use Instruction as I;
    let instructions = &code.instructions;
    let Some((I::ForIter { delta }, op_arg)) = crate::decode_instruction_at(code, pc) else {
        return true;
    };
    let exit = jump_target_forward(instructions, pc + 1, delta.get(op_arg).as_usize());
    let mut body_state = OpArgState::default();
    let mut body_pc = pc + 1;
    while body_pc < exit && body_pc < instructions.len() {
        let (body_instr, body_arg) = body_state.get(instructions[body_pc]);
        if let I::ForIter { delta } = body_instr {
            body_pc =
                jump_target_forward(instructions, body_pc + 1, delta.get(body_arg).as_usize());
            body_state = OpArgState::default();
            continue;
        }
        let supported_call_intrinsic_1 = matches!(
            body_instr,
            I::CallIntrinsic1 { func }
                if matches!(
                    func.get(body_arg),
                    crate::bytecode::IntrinsicFunction1::UnaryPositive
                        | crate::bytecode::IntrinsicFunction1::ListToTuple
                )
        );
        let permitted = for_iter_body_op_is_jit_safe(body_instr)
            || supported_call_intrinsic_1
            || matches!(
                body_instr,
                I::StoreSubscr
                    | I::StoreAttr { .. }
                    | I::StoreName { .. }
                    | I::StoreGlobal { .. }
                    | I::StoreDeref { .. }
                    | I::DeleteSubscr
                    | I::DeleteAttr { .. }
                    | I::LoadName { .. }
                    | I::ListExtend { .. }
                    | I::SetAdd { .. }
                    | I::MapAdd { .. }
                    | I::SetUpdate { .. }
                    | I::DictUpdate { .. }
                    | I::DictMerge { .. }
                    | I::LoadBuildClass
            )
            || matches!(body_instr, I::ListAppend { .. });
        if !permitted {
            if for_iter_gate_diag_enabled() {
                eprintln!(
                    "[for-iter-gate-opcode] code={} source={} for_iter_pc={pc} body_pc={body_pc} opcode={body_instr:?}",
                    code.qualname, code.source_path
                );
            }
            return false;
        }
        body_pc += 1;
    }
    true
}

/// Cached `loop_region_for_iter_bodies_all_jit_safe` — two word args, word result.
#[majit_macros::dont_look_inside]
pub fn cached_loop_region_for_iter_bodies_all_jit_safe(
    code: *const CodeObject,
    loop_header_pc: usize,
) -> bool {
    if code.is_null() {
        return true;
    }
    let code = unsafe { &*code };
    let key = (code as *const _ as usize, loop_header_pc);
    LOOP_REGION_JIT_SAFE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(&safe) = cache.get(&key) {
            return safe;
        }
        let safe = loop_region_for_iter_bodies_all_jit_safe(code, loop_header_pc);
        cache.insert(key, safe);
        safe
    })
}

/// Cached `function_entry_trace_is_jit_safe` — one word arg, word result.
#[majit_macros::dont_look_inside]
pub fn cached_function_entry_trace_is_jit_safe(code: *const CodeObject) -> bool {
    if code.is_null() {
        return true;
    }
    let code = unsafe { &*code };
    let key = code as *const _ as usize;
    FUNCTION_ENTRY_TRACE_JIT_SAFE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(&safe) = cache.get(&key) {
            return safe;
        }
        let safe = function_entry_trace_is_jit_safe(code);
        cache.insert(key, safe);
        safe
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_exec;
    use rustpython_compiler_core::bytecode::ConstantData;

    fn first_function_code(source: &str) -> CodeObject {
        let module = compile_exec(source).expect("test source should compile");
        module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("expected nested function code object")
    }

    fn dominating_backedge_targets(code: &CodeObject) -> Vec<usize> {
        let succ = code_successors(code);
        let mut targets = Vec::new();
        let mut scan_state = OpArgState::default();
        for pc in 0..code.instructions.len() {
            let (instr, op_arg) = scan_state.get(code.instructions[pc]);
            if let Some(target) = backward_jump_target(code, pc, instr, op_arg) {
                if target < code.instructions.len() && target_dominates(&succ, target, pc) {
                    targets.push(target);
                }
            }
        }
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    #[test]
    fn nested_loops_have_two_header_pcs() {
        let code = first_function_code(
            "def f(n):\n    s = 0\n    for i in range(n):\n        for j in range(n):\n            s += 1\n    return s\n",
        );
        let headers = find_loop_header_pcs(&code);
        let expected = dominating_backedge_targets(&code);
        assert_eq!(headers.len(), 2);
        assert_eq!(expected.len(), 2);
        for &pc in &expected {
            assert!(headers.contains(&pc), "missing header {pc}");
            assert!(code_pc_is_loop_header(&code as *const _, pc));
        }
        for pc in 0..code.instructions.len() {
            assert_eq!(
                code_pc_is_loop_header(&code as *const _, pc),
                headers.contains(&pc)
            );
        }
    }

    #[test]
    fn while_with_continue_has_one_header_pc() {
        let code = first_function_code(
            "def f(n):\n    s = 0\n    i = 0\n    while i < n:\n        i += 1\n        if i % 2 == 0:\n            continue\n        s += i\n    return s\n",
        );
        let headers = find_loop_header_pcs(&code);
        let expected = dominating_backedge_targets(&code);
        assert_eq!(headers.len(), 1);
        assert_eq!(expected.len(), 1);
        let header = expected[0];
        assert!(headers.contains(&header));
        assert!(code_pc_is_loop_header(&code as *const _, header));
        let mut continue_targets = Vec::new();
        let mut scan_state = OpArgState::default();
        for pc in 0..code.instructions.len() {
            let (instr, op_arg) = scan_state.get(code.instructions[pc]);
            if let Some(target) = backward_jump_target(&code, pc, instr, op_arg) {
                continue_targets.push(target);
            }
        }
        assert!(
            continue_targets.iter().any(|&t| t == header),
            "continue must land on the while header"
        );
        for &target in &continue_targets {
            assert_eq!(
                code_pc_is_loop_header(&code as *const _, target),
                target == header
            );
        }
    }

    #[test]
    fn for_loop_with_try_except_keeps_its_header() {
        let code = first_function_code(
            "def run(n):\n    acc = 0\n    for i in range(n):\n        try:\n            raise ValueError\n        except ValueError:\n            acc += 1\n    return acc\n",
        );
        let headers = find_loop_header_pcs(&code);
        let expected = dominating_backedge_targets(&code);
        assert_eq!(expected.len(), 1);
        assert_eq!(headers.len(), 1);
        assert!(headers.contains(&expected[0]));
        assert!(code_pc_is_loop_header(&code as *const _, expected[0]));

        let succ = code_successors(&code);
        let mut handler_only = Vec::new();
        let mut scan_state = OpArgState::default();
        for pc in 0..code.instructions.len() {
            let (instr, op_arg) = scan_state.get(code.instructions[pc]);
            if let Some(target) = backward_jump_target(&code, pc, instr, op_arg) {
                if !target_dominates(&succ, target, pc) {
                    handler_only.push(target);
                }
            }
        }
        for target in handler_only {
            assert!(
                !code_pc_is_loop_header(&code as *const _, target),
                "handler-rejoin target {target} must not be a loop header"
            );
        }
    }
}
