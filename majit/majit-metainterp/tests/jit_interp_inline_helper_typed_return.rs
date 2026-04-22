use majit_macros::{dont_look_inside, jit_inline};

fn assert_single_return_opcode(jitcode: &majit_metainterp::JitCode, key: &str) {
    let opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
        .get(key)
        .unwrap_or_else(|| panic!("missing wellknown opcode for {key}"));
    assert_eq!(
        jitcode.code.len(),
        3,
        "helper should emit one return opcode"
    );
    assert_eq!(jitcode.code[0], opcode, "helper should end with {key}");
}

// ── Ref-returning inline helpers ────────────────────────────────────

#[jit_inline]
fn inline_ref_identity(ptr: usize) -> usize {
    ptr
}

// ── Float-returning inline helpers ──────────────────────────────────

#[jit_inline]
fn inline_float_identity(value: f64) -> f64 {
    value
}

#[jit_inline]
fn inline_mixed_int_identity(_ptr: usize, value: i64, _scale: f64) -> i64 {
    value
}

#[dont_look_inside]
fn wrapped_ref_identity(ptr: *const i64) -> *const i64 {
    ptr
}

#[dont_look_inside]
fn wrapped_float_identity(value: f64) -> f64 {
    value
}

#[dont_look_inside]
fn wrapped_int_identity(value: i64) -> i64 {
    value
}

// ── Tests ───────────────────────────────────────────────────────────

#[test]
fn jit_inline_ref_identity_generates_valid_jitcode() {
    let (jitcode, return_reg, return_kind) = __majit_inline_jitcode_inline_ref_identity();
    // return_kind == 1 means Ref
    assert_eq!(return_kind, 1u8, "return kind should be Ref (1)");
    assert_eq!(
        return_reg, 0,
        "return register should be the parameter register"
    );
    // RPython jitcode.py:37-39 c_num_regs_i/r/f
    assert_eq!(jitcode.c_num_regs_i, 0, "no int registers needed");
    assert!(jitcode.c_num_regs_r >= 1, "at least 1 ref register needed");
    assert_eq!(jitcode.c_num_regs_f, 0, "no float registers needed");
    assert_single_return_opcode(&jitcode, "ref_return/r");
}

#[test]
fn jit_inline_float_identity_generates_valid_jitcode() {
    let (jitcode, return_reg, return_kind) = __majit_inline_jitcode_inline_float_identity();
    // return_kind == 2 means Float
    assert_eq!(return_kind, 2u8, "return kind should be Float (2)");
    assert_eq!(
        return_reg, 0,
        "return register should be the parameter register"
    );
    assert_eq!(jitcode.c_num_regs_i, 0, "no int registers needed");
    assert_eq!(jitcode.c_num_regs_r, 0, "no ref registers needed");
    assert!(
        jitcode.c_num_regs_f >= 1,
        "at least 1 float register needed"
    );
    assert_single_return_opcode(&jitcode, "float_return/f");
}

#[test]
fn jit_inline_ref_identity_keeps_interpreter_behavior() {
    assert_eq!(inline_ref_identity(42), 42);
    assert_eq!(inline_ref_identity(0), 0);
}

#[test]
fn jit_inline_float_identity_keeps_interpreter_behavior() {
    assert_eq!(inline_float_identity(3.14), 3.14);
    assert_eq!(
        inline_float_identity(-0.0f64).to_bits(),
        (-0.0f64).to_bits()
    );
}

#[test]
fn jit_inline_mixed_identity_generates_dense_per_kind_jitcode() {
    let (jitcode, return_reg, return_kind) = __majit_inline_jitcode_inline_mixed_int_identity();
    assert_eq!(return_kind, 0u8, "return kind should be Int (0)");
    assert_eq!(
        return_reg, 0,
        "mixed helper int parameter should live in dense int reg 0"
    );
    assert_eq!(jitcode.c_num_regs_i, 1, "one int register needed");
    assert_eq!(jitcode.c_num_regs_r, 1, "one ref register needed");
    assert_eq!(jitcode.c_num_regs_f, 1, "one float register needed");
    assert_single_return_opcode(&jitcode, "int_return/i");
}

#[test]
fn jit_inline_inferred_policy_only_advertises_int_return_helpers() {
    let (ref_policy, ref_builder, _, _) = __majit_call_policy_inline_ref_identity();
    assert_eq!(
        ref_policy, 0u8,
        "ref-return inline helper should not claim inferred inline_int parity"
    );
    assert!(
        ref_builder.is_null(),
        "ref-return inline helper should not expose inferred inline builder"
    );

    let (float_policy, float_builder, _, _) = __majit_call_policy_inline_float_identity();
    assert_eq!(
        float_policy, 0u8,
        "float-return inline helper should not claim inferred inline_int parity"
    );
    assert!(
        float_builder.is_null(),
        "float-return inline helper should not expose inferred inline builder"
    );

    let (int_policy, int_builder, _, _) = __majit_call_policy_inline_mixed_int_identity();
    assert_eq!(
        int_policy, 4u8,
        "int-return inline helper should keep inferred inline policy"
    );
    assert!(
        !int_builder.is_null(),
        "int-return inline helper should expose inferred inline builder"
    );
}

#[test]
fn wrapped_non_int_helpers_keep_targets_but_do_not_advertise_inferred_value_policy() {
    let (ref_policy, ref_inline_builder, ref_trace_target, ref_concrete_target) =
        __majit_call_policy_wrapped_ref_identity();
    assert_eq!(
        ref_policy, 0u8,
        "ref-return wrapped helper should not advertise inferred value-call policy"
    );
    assert!(
        ref_inline_builder.is_null(),
        "wrapped helper should not use inline builder slot"
    );
    assert!(
        !ref_trace_target.is_null() && !ref_concrete_target.is_null(),
        "explicit wrapped ref policy still needs trace/concrete targets"
    );

    let (float_policy, float_inline_builder, float_trace_target, float_concrete_target) =
        __majit_call_policy_wrapped_float_identity();
    assert_eq!(
        float_policy, 0u8,
        "float-return wrapped helper should not advertise inferred value-call policy"
    );
    assert!(
        float_inline_builder.is_null(),
        "wrapped helper should not use inline builder slot"
    );
    assert!(
        !float_trace_target.is_null() && !float_concrete_target.is_null(),
        "explicit wrapped float policy still needs trace/concrete targets"
    );

    let (int_policy, int_inline_builder, int_trace_target, int_concrete_target) =
        __majit_call_policy_wrapped_int_identity();
    assert_eq!(
        int_policy, 2u8,
        "int-return wrapped helper should keep inferred residual-int policy"
    );
    assert!(
        int_inline_builder.is_null(),
        "wrapped helper should not use inline builder slot"
    );
    assert!(
        !int_trace_target.is_null() && !int_concrete_target.is_null(),
        "int-return wrapped helper should still expose call targets"
    );
}

// ── JitCode runtime test: Ref inline call ───────────────────────────

#[test]
fn jit_inline_ref_identity_works_through_jitcode_builder() {
    use majit_metainterp::JitCodeBuilder;

    let (sub_jitcode, sub_return_reg, sub_return_kind) =
        __majit_inline_jitcode_inline_ref_identity();
    assert_eq!(sub_return_kind, 1u8, "ref helper should report Ref kind");

    let mut builder = JitCodeBuilder::new();
    // Simulate: caller has a ref value in ref register 0
    builder.load_const_r_value(0, 0xDEAD);
    let sub_idx = builder.add_sub_jitcode(sub_jitcode);
    builder.inline_call_r_r(sub_idx, &[(0, 0)], Some((sub_return_reg, 1)));
    let jitcode = builder.finish();

    // Verify the JitCode was built without panics and has correct structure
    assert!(
        jitcode.c_num_regs_r >= 2,
        "caller needs at least 2 ref registers"
    );
    assert_eq!(
        jitcode.exec.descrs.len(),
        1,
        "one sub-jitcode for inline call"
    );
}

// ── JitCode runtime test: Float inline call ─────────────────────────

#[test]
fn jit_inline_float_identity_works_through_jitcode_builder() {
    use majit_metainterp::JitCodeBuilder;

    let (sub_jitcode, sub_return_reg, sub_return_kind) =
        __majit_inline_jitcode_inline_float_identity();
    assert_eq!(
        sub_return_kind, 2u8,
        "float helper should report Float kind"
    );

    let mut builder = JitCodeBuilder::new();
    // Simulate: caller has a float value in float register 0
    builder.load_const_f_value(0, f64::to_bits(3.14) as i64);
    let sub_idx = builder.add_sub_jitcode(sub_jitcode);
    builder.inline_call_irf_f(sub_idx, &[], &[], &[(0, 0)], Some((sub_return_reg, 1)));
    let jitcode = builder.finish();

    assert!(
        jitcode.c_num_regs_f >= 2,
        "caller needs at least 2 float registers"
    );
    assert_eq!(
        jitcode.exec.descrs.len(),
        1,
        "one sub-jitcode for inline call"
    );
}

#[test]
fn jit_inline_mixed_identity_uses_dense_kind_banks_at_runtime() {
    use majit_metainterp::JitCodeBuilder;
    use majit_metainterp::blackhole::BlackholeInterpreter;

    let (sub_jitcode, sub_return_reg, sub_return_kind) =
        __majit_inline_jitcode_inline_mixed_int_identity();
    assert_eq!(sub_return_kind, 0u8, "mixed helper should report Int kind");
    assert_eq!(
        sub_return_reg, 0,
        "mixed helper int parameter should live in dense int reg 0"
    );

    let mut builder = JitCodeBuilder::new();
    builder.load_const_r_value(0, 0xDEAD);
    builder.load_const_i_value(0, 21);
    builder.load_const_f_value(0, f64::to_bits(3.5) as i64);
    let sub_idx = builder.add_sub_jitcode(sub_jitcode);
    builder.inline_call_irf_i(
        sub_idx,
        &[(0, 0)],
        &[(0, 0)],
        &[(0, 0)],
        Some((sub_return_reg, 1)),
    );
    let jitcode = builder.finish();

    let mut bh = BlackholeInterpreter::new();
    bh.setposition(std::sync::Arc::new(jitcode), 0);
    let _ = bh.run();

    assert_eq!(bh.registers_i[1], 21);
}
