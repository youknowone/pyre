//! Public trace entrypoint for `pyre`'s JIT portal.
//!
//! RPython MetaInterp._interpret() parity: trace_bytecode creates a
//! PyreMetaInterp and delegates to interpret(). The interpret loop
//! calls MIFrame::trace_code_step() for each bytecode, combining
//! concrete execution and symbolic IR recording.

use majit_metainterp::{MetaInterp, TraceAction};
use pyre_interpreter::CodeObject;

use crate::metainterp::{MetaInterpFrame, PyreMetaInterp};
use crate::state::{PyreMeta, PyreSym};

/// Trace an entire loop body starting at `start_pc`.
///
/// RPython MetaInterp._interpret() parity: creates a PyreMetaInterp
/// with a single frame and delegates to interpret(). The interpret
/// loop calls MIFrame::trace_code_step() for each bytecode, combining
/// concrete execution and symbolic IR recording.
pub fn trace_bytecode(
    meta: &mut MetaInterp<PyreMeta>,
    sym: &mut PyreSym,
    _code: &CodeObject,
    start_pc: usize,
    mut concrete_frame: Box<pyre_interpreter::pyframe::PyFrame>,
) -> (TraceAction, Box<pyre_interpreter::pyframe::PyFrame>) {
    // `llmodel.py:557` parity — install pyre's `Cpu` impl so the
    // optimizer's `protect_speculative_string` / `bh_strlen` /
    // `bh_strgetitem` family routes through `W_StrObject`-shaped
    // `str_descr` / `unicode_descr` (`pyre_cpu` module).
    meta.set_cpu(crate::pyre_cpu::shared());

    let ctx = meta
        .trace_ctx()
        .expect("trace_bytecode invariant: meta.tracing must be Some during merge_point closure");
    // A multi-frame bridge carrier overrides the trace-start
    // pc with the OUTERMOST (`frames[0]`) resume pc. The passed `start_pc` is
    // the INNERMOST frame's pc (`decode_and_restore_guard_failure` returns
    // `jit_state.next_instr()`), which belongs to the deepest reconstructed
    // callee — NOT the root. The callees are reconstructed + pushed below
    // (innermost last) so `interpret()` resumes at the deepest frame; the root
    // resumes at `root_pc` once they return (`rebuild_from_resumedata`
    // resume.py:1049-1056). Snapshot the root frame's globals/EC now, before
    // `concrete_frame` is moved into the root `MetaInterpFrame`.
    let carrier = crate::state::take_bridge_inline_carrier();
    let (start_pc, root_globals, root_w_globals_obj, root_ec) = if let Some(ref c) = carrier {
        (
            c.root_pc,
            concrete_frame.get_w_globals(),
            concrete_frame.w_globals_obj,
            concrete_frame.execution_context,
        )
    } else {
        (
            start_pc,
            std::ptr::null_mut(),
            pyre_object::PY_NULL,
            std::ptr::null(),
        )
    };
    // RPython MetaInterp._interpret() parity: root frame owns a concrete
    // PyFrame snapshot. MetaInterp drives both symbolic tracing AND
    // concrete execution — the interpreter does not run during tracing.
    concrete_frame.set_last_instr_from_next_instr(start_pc);
    let w_code = concrete_frame.pycode;
    let cf_addr = &*concrete_frame as *const pyre_interpreter::pyframe::PyFrame as usize;
    // pyjitpl.py:65 MIFrame.__init__: sym fields populated once at frame
    // construction. Callee (inline) frames are set up by perform_call
    // (trace_opcode.rs:3323-3424) and don't call init_symbolic; this path
    // handles the root frame push.
    sym.init_symbolic(ctx, cf_addr);
    let frame = MetaInterpFrame {
        sym: sym as *mut PyreSym,
        owned_sym: None,
        jitcode: w_code,
        pc: start_pc,
        greenkey: None,
        concrete_frame: cf_addr,
        owned_concrete_frame: Some(concrete_frame),
        parent_frames: Vec::new(),
        drop_frame_opref: None,
        caller_result_stack_idx: None,
        caller_result_type: None,
        arg_state: pyre_interpreter::bytecode::OpArgState::default(),
        call_site_pc: None,
    };

    let mut metainterp = PyreMetaInterp::new(w_code, std::ptr::null_mut());
    metainterp.framestack.push(frame);

    // pyjitpl.py:2971-2973: register the initial merge point so
    // reached_loop_header recognizes the trace start backedge and closes
    // the loop instead of unrolling it as a first-visit inner loop.
    let start_key = crate::driver::make_green_key(w_code, start_pc);
    {
        // resoperation.py:719/727/739 InputArg{Int,Ref,Float}: each input
        // arg's typed Box is intrinsic to its `box.type`. Mint typed
        // OpRefs from `inputarg_types()` so the merge-point's args carry
        // RPython Box identity (variant-aware Eq) rather than collapsing
        // through an Untyped position (history.py:182 `box.type`).
        let input_types = ctx.inputarg_types();
        let input_args: Vec<majit_metainterp::GreenBox> = input_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| {
                majit_metainterp::GreenBox::new(majit_ir::OpRef::input_arg_typed(i as u32, tp), tp)
            })
            .collect();
        ctx.add_merge_point(start_key, input_args, start_pc);
    }

    // Assemble + push each reconstructed inline callee onto the
    // root, OUTERMOST-FIRST, so the framestack matches the inline depth the
    // guard fired at (`[root@root_pc, frames[1]@pc, .., frames[N]@pc]`).
    // `interpret()` resumes at the innermost (last-pushed) frame, runs it to
    // RETURN, writes its result into its caller's stack, and unwinds up to the
    // root — `rebuild_from_resumedata` resume.py:1049-1056 then `_interpret`.
    if let Some(carrier) = carrier {
        for recipe in &carrier.recipes {
            // Snapshot the immediate parent (current framestack top) before the
            // mutable push. The caller result slot uses the same formula the
            // forward call site applies (trace_opcode/metainterp `perform_call`):
            // `valuestackdepth - nlocals - 1` overwrites the consumed callable
            // slot the resume snapshot still carries.
            let (parent_sym, parent_cf_addr, parent_pc, parent_parents, result_idx) = {
                let parent = metainterp
                    .framestack
                    .last()
                    .expect("trace_bytecode: root frame pushed before carrier drain");
                let psym = unsafe { &*parent.sym };
                let result_idx = psym
                    .valuestackdepth
                    .saturating_sub(psym.nlocals)
                    .checked_sub(1);
                (
                    parent.sym,
                    parent.concrete_frame,
                    parent.pc,
                    parent.parent_frames.clone(),
                    result_idx,
                )
            };
            // opencoder.py:819-834: this callee's parent chain = immediate
            // parent (just snapshotted) followed by the parent's own ancestors.
            let mut parent_frames = vec![crate::state::ResumeFrameState {
                sym: parent_sym,
                concrete_frame_addr: parent_cf_addr,
                resume_pc: parent_pc,
                pending_result_stack_idx: None,
                pending_result_type: None,
            }];
            parent_frames.extend(parent_parents);
            let pending = crate::state::assemble_bridge_inline_pending(
                ctx,
                recipe,
                root_globals,
                root_w_globals_obj,
                root_ec,
                parent_frames,
            );
            metainterp.push_inline_frame(ctx, pending, result_idx);
            // push_inline_frame hardcodes MetaInterpFrame.pc = 0; retarget to
            // the reconstructed resume pc. The concrete frame's last_instr was
            // already set in assemble_bridge_inline_pending.
            let top = metainterp
                .framestack
                .last_mut()
                .expect("trace_bytecode: pushed inline frame missing");
            top.pc = recipe.pc;
        }
    }

    let action = metainterp.interpret(ctx);

    // Recover the root frame's owned_concrete_frame for writeback.
    let executed_frame = metainterp
        .framestack
        .pop()
        .and_then(|f| f.owned_concrete_frame);

    // pyjitpl.py:3160: greenkey = original_boxes[:num_green_args]
    // original_boxes comes from the merge point where the loop closes
    // (pyjitpl.py:2995), which may differ from start_pc when
    // cut_trace_from retargets to an inner loop.
    match &action {
        TraceAction::CloseLoopWithArgs {
            loop_header_pc: Some(target_pc),
            ..
        } if *target_pc != start_pc => {
            let target_key = crate::driver::make_green_key(w_code, *target_pc);
            ctx.set_green_key(target_key, (w_code as usize, *target_pc));
            ctx.header_pc = *target_pc;
            ctx.cut_inner_green_key = Some(target_key);
        }
        TraceAction::CloseLoop | TraceAction::CloseLoopWithArgs { .. } => {
            let key = crate::driver::make_green_key(w_code, start_pc);
            ctx.set_green_key(key, (w_code as usize, start_pc));
            ctx.header_pc = start_pc;
        }
        _ => {}
    }

    // On abort, root frame may still be on the stack.
    let root_frame = if let Some(frame) = executed_frame {
        frame
    } else {
        metainterp
            .framestack
            .pop()
            .and_then(|frame| frame.owned_concrete_frame)
            .expect("trace_bytecode must return the root concrete frame")
    };
    (action, root_frame)
}

#[cfg(test)]
mod tests {
    use crate::metainterp::semantic_fallthrough_pc;
    use pyre_interpreter::bytecode::Instruction;
    use pyre_interpreter::compile_exec;
    use pyre_interpreter::decode_instruction_at;

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
                pyre_interpreter::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
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
