//! Float state-field lowering shape tests.

pub type Bytecode = [u8];

fn all_jitcode_bodies(dispatch_jc: &majit_metainterp::JitCode) -> Vec<Vec<u8>> {
    let mut bodies = vec![dispatch_jc.code.clone()];
    bodies.extend(
        dispatch_jc
            .exec
            .descrs
            .iter()
            .filter_map(|descr| descr.as_jitcode())
            .map(|sub| sub.code.clone()),
    );
    bodies
}

mod scalar {
    use super::{Bytecode, all_jitcode_bodies};
    use majit_metainterp::jitcode::insns::{BC_LOAD_STATE_FIELD_FLOAT, BC_STORE_STATE_FIELD_FLOAT};
    use majit_metainterp::{Assembler, JitDriver};

    struct FloatScalarState {
        a: i64,
        f: f64,
    }

    const OP_NOP: u8 = 0;
    const OP_TOUCH_F: u8 = 1;

    #[majit_macros::jit_interp(
        state = FloatScalarState,
        env = Bytecode,
        state_fields = { a: int, f: float },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn float_scalar_minimal(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<FloatScalarState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = FloatScalarState { a: 0, f: 0.0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_TOUCH_F => state.f = state.f + 1.5,
                _ => break,
            }
        }
        state.a
    }

    #[test]
    fn float_scalar_lowers_load_store_and_live_f() {
        use majit_metainterp::JitState as _;

        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![1], vec![], vec![0]);
        __prebuild_jitcode_liveness_float_scalar_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_float_scalar_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for float scalar fixture");
        let bodies = all_jitcode_bodies(&dispatch_jc);
        assert!(
            bodies
                .iter()
                .any(|body| body.iter().any(|&b| b == BC_LOAD_STATE_FIELD_FLOAT)),
            "float read must lower to load_state_field_float; bodies: {bodies:?}"
        );
        assert!(
            bodies
                .iter()
                .any(|body| body.iter().any(|&b| b == BC_STORE_STATE_FIELD_FLOAT)),
            "float write must lower to store_state_field_float; bodies: {bodies:?}"
        );

        let state = FloatScalarState { a: 7, f: 3.5 };
        let meta = state.build_meta(0, &[OP_NOP]);
        let live = state.extract_live(&meta);
        let types = state.live_value_types(&meta);
        let (_live_i, _live_r, live_f) = meta.canonical_liveness_slots();
        assert_eq!(live, vec![7, 3.5f64.to_bits() as i64]);
        assert_eq!(types, vec![majit_ir::Type::Int, majit_ir::Type::Float]);
        assert_eq!(live_f, vec![0]);
    }
}

mod virt_array {
    use super::{Bytecode, all_jitcode_bodies};
    use majit_metainterp::jitcode::insns::{BC_GETARRAYITEM_VABLE_F, BC_SETARRAYITEM_VABLE_F};
    use majit_metainterp::{Assembler, JitDriver};

    struct FloatArrayState {
        regs: Vec<f64>,
    }

    const OP_NOP: u8 = 0;
    const OP_TOUCH_REGS: u8 = 2;

    #[majit_macros::jit_interp(
        state = FloatArrayState,
        env = Bytecode,
        state_fields = { regs: [float; virt] },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn float_array_minimal(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<FloatArrayState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = FloatArrayState { regs: vec![0.0; 2] };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_TOUCH_REGS => state.regs[0] = state.regs[0] + 1.25,
                _ => break,
            }
        }
        0
    }

    #[test]
    fn float_virt_array_uses_float_vinfo_and_vable_ops() {
        let info = <FloatArrayState as majit_metainterp::JitState>::__build_virtualizable_info()
            .expect("float virt array should build vinfo");
        assert_eq!(info.array_fields.len(), 1);
        assert_eq!(info.array_fields[0].item_type, majit_ir::Type::Float);

        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![1, 2], vec![1], vec![]);
        __prebuild_jitcode_liveness_float_array_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_float_array_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for float virt array fixture");
        let bodies = all_jitcode_bodies(&dispatch_jc);
        assert!(
            bodies
                .iter()
                .any(|body| body.iter().any(|&b| b == BC_GETARRAYITEM_VABLE_F)),
            "float virt array read must lower to getarrayitem_vable_f; bodies: {bodies:?}"
        );
        assert!(
            bodies
                .iter()
                .any(|body| body.iter().any(|&b| b == BC_SETARRAYITEM_VABLE_F)),
            "float virt array write must lower to setarrayitem_vable_f; bodies: {bodies:?}"
        );
    }
}
