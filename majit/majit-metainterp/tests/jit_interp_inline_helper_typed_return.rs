use majit_macros::jit_inline;

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
    // Body is empty (identity returns parameter directly)
    assert!(
        jitcode.code.is_empty(),
        "identity helper should have empty bytecode"
    );
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
    assert!(
        jitcode.code.is_empty(),
        "identity helper should have empty bytecode"
    );
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

// ── JitCode runtime test: Ref inline call ───────────────────────────

#[test]
fn jit_inline_ref_identity_works_through_jitcode_builder() {
    use majit_metainterp::JitCodeBuilder;

    let (sub_jitcode, sub_return_reg, sub_return_kind) =
        __majit_inline_jitcode_inline_ref_identity();

    let mut builder = JitCodeBuilder::new();
    // Simulate: caller has a ref value in ref register 0
    builder.load_const_r_value(0, 0xDEAD);
    let sub_idx = builder.add_sub_jitcode(sub_jitcode);
    // Use typed inline call: Ref arg from caller ref reg 0 to callee ref reg 0
    builder.inline_call_with_typed_args(
        sub_idx,
        &[(majit_metainterp::JitArgKind::Ref, 0, 0)],
        Some((sub_return_reg, 1)), // callee return reg -> caller ref reg 1
        sub_return_kind,
    );
    let jitcode = builder.finish();

    // Verify the JitCode was built without panics and has correct structure
    assert!(
        jitcode.c_num_regs_r >= 2,
        "caller needs at least 2 ref registers"
    );
    assert_eq!(
        jitcode.sub_jitcodes.len(),
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

    let mut builder = JitCodeBuilder::new();
    // Simulate: caller has a float value in float register 0
    builder.load_const_f_value(0, f64::to_bits(3.14) as i64);
    let sub_idx = builder.add_sub_jitcode(sub_jitcode);
    builder.inline_call_with_typed_args(
        sub_idx,
        &[(majit_metainterp::JitArgKind::Float, 0, 0)],
        Some((sub_return_reg, 1)), // callee return reg -> caller float reg 1
        sub_return_kind,
    );
    let jitcode = builder.finish();

    assert!(
        jitcode.c_num_regs_f >= 2,
        "caller needs at least 2 float registers"
    );
    assert_eq!(
        jitcode.sub_jitcodes.len(),
        1,
        "one sub-jitcode for inline call"
    );
}
