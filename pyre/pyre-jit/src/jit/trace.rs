//! Public trace entrypoint for `pyre`'s JIT portal.
//!
//! RPython MetaInterp._interpret() parity: trace_bytecode loops over
//! all bytecodes in the loop body, recording IR while tracking concrete
//! values internally via PyreSym's concrete Box arrays. No external
//! frame execution is needed — each opcode handler computes concrete
//! results alongside symbolic IR recording.

use majit_meta::{TraceAction, TraceCtx};
use pyre_bytecode::CodeObject;
use pyre_bytecode::bytecode::Instruction;
use pyre_runtime::decode_instruction_at;

use crate::jit::state::{MIFrame, PyreSym};

fn semantic_fallthrough_pc(code: &CodeObject, pc: usize) -> usize {
    let mut next_pc = pc.saturating_add(1);
    loop {
        match decode_instruction_at(code, next_pc) {
            Some((
                Instruction::ExtendedArg
                | Instruction::Resume { .. }
                | Instruction::Nop
                | Instruction::Cache
                | Instruction::NotTaken,
                _,
            )) => {
                next_pc += 1;
            }
            _ => return next_pc,
        }
    }
}

/// Trace an entire loop body starting at `start_pc`.
///
/// RPython MetaInterp._interpret() parity: loops over bytecodes,
/// recording IR via MIFrame. Concrete values are tracked
/// internally in PyreSym's concrete_locals/concrete_stack arrays —
/// no external frame execution is needed. Stops when a back-edge
/// (CloseLoop) is reached or on error/abort.
pub fn trace_bytecode(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    code: &CodeObject,
    start_pc: usize,
    concrete_frame: usize,
) -> TraceAction {
    // RPython MetaInterp mode: PyreMetaFrame handles concrete dispatch
    // while MIFrame handles symbolic IR recording.
    // Both share PyreSym's concrete_locals/concrete_stack.
    let mut pc = start_pc;

    loop {
        if pc >= code.instructions.len() {
            return TraceAction::Abort;
        }

        // ── Symbolic + concrete trace step ──
        // Each opcode handler records IR AND updates concrete Box values
        // in PyreSym (via pending_concrete_push / concrete_locals).
        let mut frame_state =
            MIFrame::from_sym(ctx, sym, concrete_frame, semantic_fallthrough_pc(code, pc));
        frame_state.promote_valuestackdepth(concrete_frame);

        let action = frame_state.trace_code_step(code, pc);

        match action {
            TraceAction::Continue => {
                // Get next PC from symbolic state
                let next_pc = sym.pending_next_instr.take().unwrap_or(pc + 1);
                pc = next_pc;
            }

            TraceAction::CloseLoop | TraceAction::CloseLoopWithArgs { .. } => {
                // Retarget trace green key to loop header
                if let TraceAction::CloseLoopWithArgs {
                    loop_header_pc: Some(target_pc),
                    ..
                } = action
                {
                    let key = crate::eval::make_green_key(code as *const CodeObject, target_pc);
                    ctx.set_green_key(key);
                } else if matches!(action, TraceAction::CloseLoop) {
                    let key = crate::eval::make_green_key(code as *const CodeObject, pc);
                    ctx.set_green_key(key);
                }
                return action;
            }

            other => return other,
        }
    }
}

/// Test PyreMetaInterp on a code object — for validation only.
/// Returns concrete result if the interpret loop completes.
#[allow(dead_code)]
pub fn test_metainterp_trace(
    code: &CodeObject,
    namespace: *mut pyre_runtime::PyNamespace,
    args: &[super::state::ConcreteValue],
) -> Option<super::state::ConcreteValue> {
    use super::metainterp::{PyreMetaInterp, StepAction};

    let mut metainterp = PyreMetaInterp::new(code as *const CodeObject, namespace);
    let frontend_args: Vec<super::state::FrontendOp> = args
        .iter()
        .map(|c| super::state::FrontendOp::new(majit_ir::OpRef::NONE, *c))
        .collect();
    metainterp.perform_call(code as *const CodeObject, frontend_args, None);

    // Dummy TraceCtx — we only test concrete execution here
    // TODO: proper TraceCtx integration
    None // Cannot create TraceCtx without recorder
}

#[cfg(test)]
mod tests {
    use super::semantic_fallthrough_pc;
    use pyre_bytecode::bytecode::Instruction;
    use pyre_bytecode::compile_exec;
    use pyre_runtime::decode_instruction_at;

    #[test]
    fn test_semantic_fallthrough_pc_skips_branch_trivia() {
        let mut source = String::from("def f(x, y):\n    if x < y:\n");
        for i in 0..400 {
            source.push_str(&format!("        z{i} = {i}\n"));
        }
        source.push_str("    return 0\n");
        source.push_str("f(1, 2)\n");

        let module = compile_exec(&source).expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_bytecode::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let branch_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::PopJumpIfFalse { .. }, _))
                )
            })
            .expect("test bytecode should contain POP_JUMP_IF_FALSE");

        let fallthrough_pc = semantic_fallthrough_pc(&code, branch_pc);
        let fallthrough_instruction = decode_instruction_at(&code, fallthrough_pc)
            .map(|(instruction, _)| instruction)
            .expect("semantic fallthrough should decode");

        assert!(
            !matches!(
                fallthrough_instruction,
                Instruction::ExtendedArg
                    | Instruction::Resume { .. }
                    | Instruction::Nop
                    | Instruction::Cache
                    | Instruction::NotTaken
            ),
            "semantic fallthrough must skip bytecode trivia"
        );
    }
}
