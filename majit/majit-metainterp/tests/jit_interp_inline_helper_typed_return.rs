use majit_macros::{dont_look_inside, dont_look_inside_cannot_raise, jit_inline};
use majit_metainterp::jitcode::JitCodeRuntimeExt;

fn assert_single_return_opcode(jitcode: &majit_metainterp::JitCode, key: &str) {
    let opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
        .get(key)
        .unwrap_or_else(|| panic!("missing wellknown opcode for {key}"));
    assert_eq!(
        jitcode.code.len(),
        2,
        "helper should emit one return opcode (1 byte) + 1-byte register operand"
    );
    assert_eq!(jitcode.code[0], opcode, "helper should end with {key}");
}

fn assert_trailing_return(
    jitcode: &majit_metainterp::JitCode,
    key: &str,
    kind: majit_metainterp::JitArgKind,
    reg: u16,
) {
    assert_single_return_opcode(jitcode, key);
    assert_eq!(
        jitcode.trailing_return_info(),
        Some((kind, reg)),
        "helper should expose trailing return info from jitcode bytecode"
    );
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

#[repr(C)]
struct InlineTypedStack {
    head: *mut InlineTypedNode,
    size: usize,
}

#[repr(C)]
struct InlineTypedNode {
    value: i64,
    next: *mut InlineTypedNode,
}

#[jit_inline(
    ref_params = {
        stack: ref(InlineTypedStack),
    },
    ref_fields = {
        InlineTypedStack::head => InlineTypedNode,
        InlineTypedNode::next => InlineTypedNode,
    },
)]
fn inline_typed_stack_pop(stack: usize) -> i64 {
    let head = stack.head;
    let value = head.value;
    let next = head.next;
    stack.head = next;
    stack.size = stack.size - 1usize;
    value
}

#[jit_inline(calls = { inline_typed_stack_pop => inline_int })]
fn inline_typed_stack_pop_caller(stack: usize) -> i64 {
    inline_typed_stack_pop(stack)
}

#[jit_inline(
    ref_params = {
        stack: ref(InlineTypedStack),
    },
    ref_fields = {
        InlineTypedStack::head => InlineTypedNode,
        InlineTypedNode::next => InlineTypedNode,
    },
)]
fn inline_typed_stack_swap(stack: usize) {
    let node1 = stack.head;
    let node2 = node1.next;
    let v1 = node1.value;
    let v2 = node2.value;
    node1.value = v2;
    node2.value = v1;
}

#[jit_inline(calls = { inline_typed_stack_swap => inline_void })]
fn inline_typed_stack_swap_caller(stack: usize) {
    inline_typed_stack_swap(stack);
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

// Ports of the same identity helpers under the explicit cannot-raise
// opt-in.  The function bodies are pure pass-throughs that PyPy
// `getcalldescr` would mark `EF_CANNOT_RAISE` (`call.py:303`); pyre's
// analyzer (`majit-translate/src/codewriter/call.rs:3250
// effectinfo_from_writeanalyze`) computes the equivalent in the
// codewriter pipeline but is not yet plumbed to runtime trace
// recording, so users opt in via
// `#[dont_look_inside_cannot_raise]` until the wire-up lands.

#[dont_look_inside_cannot_raise]
fn wrapped_ref_identity_cr(ptr: *const i64) -> *const i64 {
    ptr
}

#[dont_look_inside_cannot_raise]
fn wrapped_float_identity_cr(value: f64) -> f64 {
    value
}

#[dont_look_inside_cannot_raise]
fn wrapped_int_identity_cr(value: i64) -> i64 {
    value
}

// ── Tests ───────────────────────────────────────────────────────────

#[test]
fn jit_inline_ref_identity_generates_valid_jitcode() {
    // Inline helper jitcodes register their
    // per-marker liveness triples into an `Assembler` passed in by the
    // caller (the production path uses the driver-shared one).  Tests
    // that only inspect structural properties pass a freshly-allocated
    // `Assembler`; the resulting BC_LIVE 2-byte slots are scoped to that
    // local table — ref/float helpers below have no guards/markers, so
    // there's nothing for the test path to decode beyond the trailing
    // return.
    let mut asm = majit_metainterp::Assembler::new();
    let jitcode = __majit_inline_jitcode_inline_ref_identity_with_asm(&mut asm);
    // RPython jitcode.py:37-39 c_num_regs_i/r/f
    assert_eq!(jitcode.c_num_regs_i, 0, "no int registers needed");
    assert!(jitcode.c_num_regs_r >= 1, "at least 1 ref register needed");
    assert_eq!(jitcode.c_num_regs_f, 0, "no float registers needed");
    assert_trailing_return(
        &jitcode,
        "ref_return/r",
        majit_metainterp::JitArgKind::Ref,
        0,
    );
}

#[test]
fn jit_inline_float_identity_generates_valid_jitcode() {
    let mut asm = majit_metainterp::Assembler::new();
    let jitcode = __majit_inline_jitcode_inline_float_identity_with_asm(&mut asm);
    assert_eq!(jitcode.c_num_regs_i, 0, "no int registers needed");
    assert_eq!(jitcode.c_num_regs_r, 0, "no ref registers needed");
    assert!(
        jitcode.c_num_regs_f >= 1,
        "at least 1 float register needed"
    );
    assert_trailing_return(
        &jitcode,
        "float_return/f",
        majit_metainterp::JitArgKind::Float,
        0,
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

#[test]
fn jit_inline_mixed_identity_generates_dense_per_kind_jitcode() {
    let mut asm = majit_metainterp::Assembler::new();
    let jitcode = __majit_inline_jitcode_inline_mixed_int_identity_with_asm(&mut asm);
    assert_eq!(jitcode.c_num_regs_i, 1, "one int register needed");
    assert_eq!(jitcode.c_num_regs_r, 1, "one ref register needed");
    assert_eq!(jitcode.c_num_regs_f, 1, "one float register needed");
    assert_trailing_return(
        &jitcode,
        "int_return/i",
        majit_metainterp::JitArgKind::Int,
        0,
    );
}

#[test]
fn jit_inline_ref_param_field_access_lowers_to_native_field_ops() {
    let insns = majit_metainterp::jitcode::wellknown_bh_insns();
    let getfield_i = *insns.get("getfield_gc_i/rd>i").unwrap();
    let getfield_r = *insns.get("getfield_gc_r/rd>r").unwrap();
    let setfield_i = *insns.get("setfield_gc_i/rid").unwrap();
    let setfield_r = *insns.get("setfield_gc_r/rrd").unwrap();
    let inline_call = majit_metainterp::jitcode::insns::BC_INLINE_CALL;
    let residual_call_r_i = majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_R_I;
    let residual_call_ir_i = majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_IR_I;
    let residual_call_irf_i = majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_IRF_I;

    let mut asm = majit_metainterp::Assembler::new();
    let helper = __majit_inline_jitcode_inline_typed_stack_pop_with_asm(&mut asm);
    assert!(
        helper.code.contains(&getfield_r),
        "typed ref-param helper must read ref fields with getfield_gc_r; code={:?}",
        helper.code
    );
    assert!(
        helper.code.contains(&getfield_i),
        "typed ref-param helper must read int fields with getfield_gc_i; code={:?}",
        helper.code
    );
    assert!(
        helper.code.contains(&setfield_r),
        "typed ref-param helper must write ref fields with setfield_gc_r; code={:?}",
        helper.code
    );
    assert!(
        helper.code.contains(&setfield_i),
        "typed ref-param helper must write int fields with setfield_gc_i; code={:?}",
        helper.code
    );
    assert!(
        !helper.code.contains(&residual_call_r_i)
            && !helper.code.contains(&residual_call_ir_i)
            && !helper.code.contains(&residual_call_irf_i),
        "typed field helper must not fall back to residual int calls; code={:?}",
        helper.code
    );

    let caller = __majit_inline_jitcode_inline_typed_stack_pop_caller_with_asm(&mut asm);
    assert!(
        caller.code.contains(&inline_call),
        "caller helper should splice the typed helper through inline_call; code={:?}",
        caller.code
    );
    assert!(
        !caller.code.contains(&residual_call_r_i)
            && !caller.code.contains(&residual_call_ir_i)
            && !caller.code.contains(&residual_call_irf_i),
        "caller helper must not use a residual call for inline_typed_stack_pop; code={:?}",
        caller.code
    );

    let mut second = InlineTypedNode {
        value: 22,
        next: std::ptr::null_mut(),
    };
    let mut first = InlineTypedNode {
        value: 11,
        next: &mut second,
    };
    let mut stack = InlineTypedStack {
        head: &mut first,
        size: 2,
    };

    let mut top = majit_metainterp::JitCodeBuilder::new();
    let sub_idx = top.add_sub_jitcode(caller);
    top.inline_call_r_i(sub_idx, &[(0, 0)], Some(0));
    let top = top.finish();

    let mut bh_insns: indexmap::IndexMap<String, u8> =
        majit_metainterp::jitcode::wellknown_bh_insns()
            .iter()
            .map(|(key, value)| ((*key).to_string(), *value))
            .collect();
    bh_insns.extend(
        majit_metainterp::jitcode::pyre_extension_insns()
            .iter()
            .map(|(key, value)| ((*key).to_string(), *value)),
    );
    let mut bh_builder = majit_metainterp::blackhole::build_inline_call_only_bh_builder();
    bh_builder.setup_insns(&bh_insns);
    bh_builder.setup_cached_control_opcodes(
        majit_metainterp::jitcode::insns::BC_LIVE as i32,
        majit_metainterp::jitcode::insns::BC_CATCH_EXCEPTION as i32,
        majit_metainterp::jitcode::insns::BC_RVMPROF_CODE as i32,
    );
    majit_metainterp::blackhole::wire_bhimpl_handlers(&mut bh_builder);
    let mut bh = bh_builder.acquire_interp();
    bh.setposition(std::sync::Arc::new(top), 0);
    bh.registers_r[0] = (&mut stack as *mut InlineTypedStack) as i64;
    let _ = bh.run();

    assert_eq!(
        bh.registers_i[0], 11,
        "inline_call should return head.value"
    );
    assert!(
        std::ptr::eq(stack.head as *const InlineTypedNode, &second),
        "inline_call should write stack.head to the next node"
    );
    assert_eq!(stack.size, 1, "inline_call should decrement stack.size");
}

#[test]
fn jit_inline_void_ref_param_field_swap_lowers_to_native_field_ops() {
    let insns = majit_metainterp::jitcode::wellknown_bh_insns();
    let getfield_i = *insns.get("getfield_gc_i/rd>i").unwrap();
    let getfield_r = *insns.get("getfield_gc_r/rd>r").unwrap();
    let setfield_i = *insns.get("setfield_gc_i/rid").unwrap();
    let inline_call = majit_metainterp::jitcode::insns::BC_INLINE_CALL;
    let void_return = majit_metainterp::jitcode::insns::BC_VOID_RETURN;
    let residual_calls = [
        majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_R_V,
        majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_IR_V,
        majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
        majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_R_I,
        majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_IR_I,
        majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_IRF_I,
        majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_R_R,
        majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_IR_R,
        majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_IRF_R,
        majit_metainterp::jitcode::insns::BC_RESIDUAL_CALL_IRF_F,
    ];

    let mut asm = majit_metainterp::Assembler::new();
    let helper = __majit_inline_jitcode_inline_typed_stack_swap_with_asm(&mut asm);
    assert_eq!(
        helper.code.last().copied(),
        Some(void_return),
        "void helper should end in BC_VOID_RETURN; code={:?}",
        helper.code
    );
    assert_eq!(
        helper.trailing_return_info(),
        None,
        "BC_VOID_RETURN helpers have no typed trailing return info"
    );
    assert!(
        helper.code.contains(&getfield_r),
        "typed void helper must read ref fields with getfield_gc_r; code={:?}",
        helper.code
    );
    assert!(
        helper.code.contains(&getfield_i),
        "typed void helper must read int fields with getfield_gc_i; code={:?}",
        helper.code
    );
    assert!(
        helper.code.contains(&setfield_i),
        "typed void helper must write int fields with setfield_gc_i; code={:?}",
        helper.code
    );
    assert!(
        residual_calls
            .iter()
            .all(|opcode| !helper.code.contains(opcode)),
        "typed void helper must not fall back to residual calls; code={:?}",
        helper.code
    );

    let caller = __majit_inline_jitcode_inline_typed_stack_swap_caller_with_asm(&mut asm);
    assert!(
        caller.code.contains(&inline_call),
        "void caller helper should splice the typed helper through inline_call; code={:?}",
        caller.code
    );
    assert!(
        residual_calls
            .iter()
            .all(|opcode| !caller.code.contains(opcode)),
        "void caller helper must not use a residual call for inline_typed_stack_swap; code={:?}",
        caller.code
    );

    let mut second = InlineTypedNode {
        value: 22,
        next: std::ptr::null_mut(),
    };
    let mut first = InlineTypedNode {
        value: 11,
        next: &mut second,
    };
    let mut stack = InlineTypedStack {
        head: &mut first,
        size: 2,
    };

    let mut top = majit_metainterp::JitCodeBuilder::new();
    let sub_idx = top.add_sub_jitcode(caller);
    top.inline_call_r_v(sub_idx, &[(0, 0)], None);
    let top = top.finish();

    let mut bh_insns: indexmap::IndexMap<String, u8> =
        majit_metainterp::jitcode::wellknown_bh_insns()
            .iter()
            .map(|(key, value)| ((*key).to_string(), *value))
            .collect();
    bh_insns.extend(
        majit_metainterp::jitcode::pyre_extension_insns()
            .iter()
            .map(|(key, value)| ((*key).to_string(), *value)),
    );
    let mut bh_builder = majit_metainterp::blackhole::build_inline_call_only_bh_builder();
    bh_builder.setup_insns(&bh_insns);
    bh_builder.setup_cached_control_opcodes(
        majit_metainterp::jitcode::insns::BC_LIVE as i32,
        majit_metainterp::jitcode::insns::BC_CATCH_EXCEPTION as i32,
        majit_metainterp::jitcode::insns::BC_RVMPROF_CODE as i32,
    );
    majit_metainterp::blackhole::wire_bhimpl_handlers(&mut bh_builder);
    let mut bh = bh_builder.acquire_interp();
    bh.setposition(std::sync::Arc::new(top), 0);
    bh.registers_r[0] = (&mut stack as *mut InlineTypedStack) as i64;
    let _ = bh.run();

    assert_eq!(
        unsafe { (*stack.head).value },
        22,
        "inline void helper should swap head"
    );
    assert_eq!(
        unsafe { (*(*stack.head).next).value },
        11,
        "inline void helper should swap second node"
    );
    assert_eq!(stack.size, 2, "swap should leave stack size unchanged");
}

#[test]
fn jit_inline_inferred_policy_only_advertises_int_return_helpers() {
    let (ref_policy, ref_builder, _, _, _, _) = __majit_call_policy_inline_ref_identity();
    assert_eq!(
        ref_policy, 0u8,
        "ref-return inline helper should not claim inferred inline_int parity"
    );
    assert!(
        ref_builder.is_null(),
        "ref-return inline helper should not expose inferred inline builder"
    );

    let (float_policy, float_builder, _, _, _, _) = __majit_call_policy_inline_float_identity();
    assert_eq!(
        float_policy, 0u8,
        "float-return inline helper should not claim inferred inline_int parity"
    );
    assert!(
        float_builder.is_null(),
        "float-return inline helper should not expose inferred inline builder"
    );

    let (int_policy, int_builder, _, _, _, _) = __majit_call_policy_inline_mixed_int_identity();
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
fn wrapped_helpers_advertise_supported_inferred_policy_bytes() {
    let (ref_policy, ref_inline_builder, ref_trace_target, ref_concrete_target, _, _) =
        __majit_call_policy_wrapped_ref_identity();
    assert_eq!(
        ref_policy, 25u8,
        "ref-return wrapped helper should advertise inferred residual-ref policy"
    );
    assert!(
        ref_inline_builder.is_null(),
        "wrapped helper should not use inline builder slot"
    );
    assert!(
        !ref_trace_target.is_null() && !ref_concrete_target.is_null(),
        "explicit wrapped ref policy still needs trace/concrete targets"
    );

    let (float_policy, float_inline_builder, float_trace_target, float_concrete_target, _, _) =
        __majit_call_policy_wrapped_float_identity();
    assert_eq!(
        float_policy, 0u8,
        "float-return wrapped helper should stay unsupported via inferred path"
    );
    assert!(
        float_inline_builder.is_null(),
        "wrapped helper should not use inline builder slot"
    );
    assert!(
        !float_trace_target.is_null() && !float_concrete_target.is_null(),
        "explicit wrapped float policy still needs trace/concrete targets"
    );

    let (int_policy, int_inline_builder, int_trace_target, int_concrete_target, _, _) =
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

#[test]
fn dont_look_inside_cannot_raise_emits_dedicated_policy_bytes() {
    // Item 4-5 fix: `#[dont_look_inside_cannot_raise]` opt-in maps to
    // distinct policy bytes per result kind so the inferred slot lookup
    // (`jitcode_lower.rs:1711` via `byte 28u8|29u8|30u8 -> CannotRaise`)
    // can produce `cannot_raise_effect_info()` calldescrs and skip the
    // trailing `-live-` marker that the audit cited as parity-divergent
    // for `dont_look_inside` ref helpers.
    let (ref_policy, _, ref_trace_target, ref_concrete_target, _, _) =
        __majit_call_policy_wrapped_ref_identity_cr();
    assert_eq!(
        ref_policy, 30u8,
        "ref-return cannot-raise wrapped helper should emit byte 30u8"
    );
    assert!(
        !ref_trace_target.is_null() && !ref_concrete_target.is_null(),
        "explicit wrapped ref policy still needs trace/concrete targets"
    );

    let (float_policy, _, float_trace_target, float_concrete_target, _, _) =
        __majit_call_policy_wrapped_float_identity_cr();
    assert_eq!(
        float_policy, 0u8,
        "float-return cannot-raise wrapped helper stays unsupported via inferred path \
         (mirrors `wrapped_float_identity` 0u8 — separate explicit policy required)"
    );
    assert!(
        !float_trace_target.is_null() && !float_concrete_target.is_null(),
        "explicit wrapped float policy still needs trace/concrete targets"
    );

    let (int_policy, _, int_trace_target, int_concrete_target, _, _) =
        __majit_call_policy_wrapped_int_identity_cr();
    assert_eq!(
        int_policy, 29u8,
        "int-return cannot-raise wrapped helper should emit byte 29u8"
    );
    assert!(
        !int_trace_target.is_null() && !int_concrete_target.is_null(),
        "int-return cannot-raise wrapped helper should still expose call targets"
    );
}

// ── JitCode runtime test: Ref inline call ───────────────────────────

#[test]
fn jit_inline_ref_identity_works_through_jitcode_builder() {
    use majit_metainterp::JitCodeBuilder;

    let mut asm = majit_metainterp::Assembler::new();
    let sub_jitcode = __majit_inline_jitcode_inline_ref_identity_with_asm(&mut asm);
    let (sub_return_kind, _sub_return_reg) = sub_jitcode
        .trailing_return_info()
        .expect("ref helper should end in ref_return");
    assert_eq!(
        sub_return_kind,
        majit_metainterp::JitArgKind::Ref,
        "ref helper should report Ref kind"
    );

    let mut builder = JitCodeBuilder::new();
    // Simulate: caller has a ref value in ref register 0
    builder.load_const_r_value(0, 0xDEAD);
    let sub_idx = builder.add_sub_jitcode(sub_jitcode);
    builder.inline_call_r_r(sub_idx, &[(0, 0)], Some(1));
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

    let mut asm = majit_metainterp::Assembler::new();
    let sub_jitcode = __majit_inline_jitcode_inline_float_identity_with_asm(&mut asm);
    let (sub_return_kind, _sub_return_reg) = sub_jitcode
        .trailing_return_info()
        .expect("float helper should end in float_return");
    assert_eq!(
        sub_return_kind,
        majit_metainterp::JitArgKind::Float,
        "float helper should report Float kind"
    );

    let mut builder = JitCodeBuilder::new();
    // Simulate: caller has a float value in float register 0
    builder.load_const_f_value(0, f64::to_bits(3.14) as i64);
    let sub_idx = builder.add_sub_jitcode(sub_jitcode);
    builder.inline_call_irf_f(sub_idx, &[], &[], &[(0, 0)], Some(1));
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
    use majit_metainterp::blackhole::build_inline_call_only_bh_builder;

    let mut asm = majit_metainterp::Assembler::new();
    let sub_jitcode = __majit_inline_jitcode_inline_mixed_int_identity_with_asm(&mut asm);
    let (sub_return_kind, sub_return_reg) = sub_jitcode
        .trailing_return_info()
        .expect("mixed helper should end in int_return");
    assert_eq!(
        sub_return_kind,
        majit_metainterp::JitArgKind::Int,
        "mixed helper should report Int kind"
    );
    assert_eq!(
        sub_return_reg, 0,
        "mixed helper int parameter should live in dense int reg 0"
    );

    let mut jc_builder = JitCodeBuilder::new();
    jc_builder.load_const_r_value(0, 0xDEAD);
    jc_builder.load_const_i_value(0, 21);
    jc_builder.load_const_f_value(0, f64::to_bits(3.5) as i64);
    let sub_idx = jc_builder.add_sub_jitcode(sub_jitcode);
    jc_builder.inline_call_irf_i(sub_idx, &[(0, 0)], &[(0, 0)], &[(0, 0)], Some(1));
    let jitcode = jc_builder.finish();

    // Route through `handler_inline_call_pyre_nested`
    // (the production builder shape) instead of the legacy
    // `dispatch_one::BC_INLINE_CALL` fallback.
    let mut bh_builder = build_inline_call_only_bh_builder();
    let mut bh = bh_builder.acquire_interp();
    bh.setposition(std::sync::Arc::new(jitcode), 0);
    let _ = bh.run();

    assert_eq!(bh.registers_i[1], 21);
}
