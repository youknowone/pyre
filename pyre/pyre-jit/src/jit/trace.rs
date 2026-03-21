//! Public trace entrypoint for `pyre`'s JIT portal.
//!
//! This stays as the stable entry surface used by the interpreter loop while
//! trace recording executes directly through `TraceFrameState`.

use majit_meta::{TraceAction, TraceCtx};
use pyre_bytecode::CodeObject;
use pyre_bytecode::bytecode::Instruction;
use pyre_runtime::decode_instruction_at;

use crate::jit::state::{PyreSym, TraceFrameState};

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

pub fn trace_bytecode(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    code: &CodeObject,
    pc: usize,
    concrete_frame: usize,
) -> TraceAction {
    // RPython pyjitpl/blackhole pass the semantic fallthrough pc to
    // goto_if_not guards, not the raw bytecode successor. pyre bytecode keeps
    // EXTENDED_ARG/NOT_TAKEN/CACHE as separate units, so skip them here.
    let mut frame_state =
        TraceFrameState::from_sym(ctx, sym, concrete_frame, semantic_fallthrough_pc(code, pc));

    // PyPy interp_jit.py:89 — promote(valuestackdepth).
    // hint(self.valuestackdepth, promote=True) forces valuestackdepth
    // to be a compile-time constant, enabling constant-folding of
    // stack offset calculations.
    frame_state.promote_valuestackdepth(concrete_frame);

    let action = frame_state.trace_code_step(code, pc);

    // RPython pyjitpl.py reached_loop_header(): when a back-edge closes
    // the loop, the compiled trace must be registered under the back-edge's
    // green key, not the function-entry key.  Without this, func-entry
    // tracing stores the loop under PC=0, causing infinite re-entry.
    if let TraceAction::CloseLoopWithArgs { ref jump_args, .. } = action {
        let target_pc = jump_args
            .get(1)
            .and_then(|&opref| ctx.const_value(opref))
            .map(|pc| pc as usize)
            .unwrap_or(pc);
        let back_edge_key = crate::eval::make_green_key(code as *const CodeObject, target_pc);
        ctx.set_green_key(back_edge_key);
    } else if matches!(action, TraceAction::CloseLoop) {
        let back_edge_key = crate::eval::make_green_key(code as *const CodeObject, pc);
        ctx.set_green_key(back_edge_key);
    }

    action
}

#[cfg(test)]
mod tests {
    use super::semantic_fallthrough_pc;
    use pyre_bytecode::compile_exec;
    use pyre_bytecode::bytecode::Instruction;
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
