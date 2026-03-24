//! Public trace entrypoint for `pyre`'s JIT portal.
//!
//! RPython MetaInterp._interpret() parity: trace_bytecode creates a
//! PyreMetaInterp and delegates to interpret(). The interpret loop
//! calls MIFrame::trace_code_step() for each bytecode, combining
//! concrete execution and symbolic IR recording.

use majit_meta::{TraceAction, TraceCtx};
use pyre_bytecode::CodeObject;

use crate::jit::metainterp::{MetaInterpFrame, PyreMetaInterp};
use crate::jit::state::PyreSym;

/// Trace an entire loop body starting at `start_pc`.
///
/// RPython MetaInterp._interpret() parity: creates a PyreMetaInterp
/// with a single frame and delegates to interpret(). The interpret
/// loop calls MIFrame::trace_code_step() for each bytecode, combining
/// concrete execution and symbolic IR recording.
pub fn trace_bytecode(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    code: &CodeObject,
    start_pc: usize,
    concrete_frame: usize,
) -> TraceAction {
    // Build a MetaInterp with a single root frame.
    // The root frame borrows PyreSym from the caller via raw pointer.
    // PyreMetaInterp.interpret() drives the dispatch loop.
    let frame = MetaInterpFrame {
        sym: sym as *mut PyreSym,
        jitcode: code as *const CodeObject,
        pc: start_pc,
        greenkey: None,
        concrete_frame,
    };

    let mut metainterp = PyreMetaInterp::new(code as *const CodeObject, std::ptr::null_mut());
    metainterp.framestack.push(frame);

    let action = metainterp.interpret(ctx);

    // Retarget green key based on back-edge target.
    // Done here (not inside interpret) because start_pc is needed
    // for CloseLoop — metainterp doesn't track the trace entry PC.
    match &action {
        TraceAction::CloseLoopWithArgs {
            loop_header_pc: Some(target_pc),
            ..
        } => {
            let key = crate::eval::make_green_key(code as *const CodeObject, *target_pc);
            ctx.set_green_key(key);
        }
        TraceAction::CloseLoop => {
            let key = crate::eval::make_green_key(code as *const CodeObject, start_pc);
            ctx.set_green_key(key);
        }
        _ => {}
    }

    action
}

#[cfg(test)]
mod tests {
    use crate::jit::metainterp::semantic_fallthrough_pc;
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
