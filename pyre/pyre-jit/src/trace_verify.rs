//! Verification: auto-generated trace functions produce correct IR.

#[cfg(test)]
mod tests {
    use majit_ir::{OpCode, OpRef};
    use majit_meta::TraceCtx;
    use crate::jit::descr::{make_field_descr, make_size_descr};

    fn ob_type_descr() -> majit_ir::DescrRef {
        make_field_descr(0, 8, majit_ir::Type::Int, false)
    }
    fn intval_descr() -> majit_ir::DescrRef {
        make_field_descr(8, 8, majit_ir::Type::Int, true)
    }
    fn w_int_size_descr() -> majit_ir::DescrRef {
        make_size_descr(16)
    }
    const FAKE_INT_TYPE: i64 = 0x1234_5678;

    /// Get ops from TraceCtx, excluding the dummy Finish we add for finalization.
    fn get_ops(ctx: TraceCtx) -> Vec<OpCode> {
        let mut recorder = ctx.into_recorder();
        let dummy_descr = make_size_descr(0);
        recorder.finish(&[], dummy_descr);
        let trace = recorder.get_trace();
        trace
            .ops
            .iter()
            .map(|o| o.opcode)
            .filter(|op| *op != OpCode::Finish)
            .collect()
    }

    #[test]
    fn test_unbox_int_ops() {
        let mut ctx = TraceCtx::for_test(1);
        let obj = OpRef(0);
        let _intval = crate::trace_unbox_int(
            &mut ctx,
            obj,
            FAKE_INT_TYPE,
            ob_type_descr(),
            intval_descr(),
            &[obj],
        );
        let ops = get_ops(ctx);
        assert_eq!(
            ops,
            vec![
                OpCode::GetfieldRawI,
                OpCode::GuardClass,
                OpCode::GetfieldRawI,
            ]
        );
        eprintln!("✓ trace_unbox_int: {:?}", ops);
    }

    #[test]
    fn test_box_int_ops() {
        let mut ctx = TraceCtx::for_test(1);
        let _obj = crate::trace_box_int(
            &mut ctx,
            OpRef(0),
            w_int_size_descr(),
            ob_type_descr(),
            intval_descr(),
            FAKE_INT_TYPE,
        );
        let ops = get_ops(ctx);
        assert_eq!(
            ops,
            vec![OpCode::New, OpCode::SetfieldRaw, OpCode::SetfieldRaw,]
        );
        eprintln!("✓ trace_box_int: {:?}", ops);
    }

    #[test]
    fn test_int_binop_ovf_ops() {
        let mut ctx = TraceCtx::for_test(2);
        let _result = crate::trace_int_binop_ovf(
            &mut ctx,
            OpRef(0),
            OpRef(1),
            OpCode::IntAddOvf,
            FAKE_INT_TYPE,
            ob_type_descr(),
            intval_descr(),
            w_int_size_descr(),
            &[OpRef(0), OpRef(1)],
        );
        let ops = get_ops(ctx);
        assert_eq!(
            ops,
            vec![
                OpCode::GetfieldRawI,
                OpCode::GuardClass,
                OpCode::GetfieldRawI, // unbox a
                OpCode::GetfieldRawI,
                OpCode::GuardClass,
                OpCode::GetfieldRawI, // unbox b
                OpCode::IntAddOvf,
                OpCode::GuardNoOverflow, // add + guard
                OpCode::New,
                OpCode::SetfieldRaw,
                OpCode::SetfieldRaw, // box
            ]
        );
        eprintln!("✓ trace_int_binop_ovf: {} ops", ops.len());
    }
}
