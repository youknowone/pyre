//! Execution checks for the x86 emitters ported onto the upstream sequences:
//! `genop_int_and`, `genop_int_force_ge_zero`, `genop_float_neg`,
//! `genop_float_abs`, `_cmp_guard_gc_type`, `genop_guard_guard_is_object`,
//! `genop_guard_guard_subclass`.

#![cfg(target_arch = "x86_64")]

use majit_backend::{Backend, JitCellToken};
use majit_backend_dynasm::runner::DynasmBackend;
use majit_ir::forwarding::bound_operand_from_opref as rb;
use majit_ir::operand::Operand;
use majit_ir::{GcRef, InputArg, Op, OpCode, OpRc, OpRef, Type, Value};

fn compile_and_run_int(opcode: OpCode, extra: Option<i64>, input: i64, token_id: u64) -> i64 {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(token_id);
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();
    let mut args = vec![rb(i0)];
    if let Some(imm) = extra {
        args.push(rb(OpRef::const_int(imm)));
    }
    let body = Op::new(opcode, &args);
    body.pos().set(OpRef::int_op(1));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![Type::Int]);
    finish.setfailargs(vec![rb(OpRef::int_op(1))].into());
    let ops = vec![OpRc::new(body), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile {opcode:?}: {err:?}"));
    let frame = backend.execute_token(&token, &[Value::Int(input)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    backend.get_int_value(&frame, 0)
}

fn compile_float(opcode: OpCode, token_id: u64) -> (DynasmBackend, JitCellToken) {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(token_id);
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let i0 = inputargs[0].opref();
    let body = Op::new(opcode, &[rb(i0)]);
    body.pos().set(OpRef::float_op(1));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![Type::Float]);
    finish.setfailargs(vec![rb(OpRef::float_op(1))].into());
    let ops = vec![OpRc::new(body), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile {opcode:?}: {err:?}"));
    (backend, token)
}

fn float_bits(backend: &mut DynasmBackend, token: &JitCellToken, input: f64) -> u64 {
    let frame = backend.execute_token(token, &[Value::Float(input)]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    backend.get_float_value(&frame, 0).to_bits()
}

struct GcWorld {
    backend: DynasmBackend,
    child_tid: i64,
    root_vtable: usize,
    child: GcRef,
    raw: GcRef,
    other: GcRef,
}

fn gc_world(token_id: u64) -> (GcWorld, JitCellToken) {
    let mut gc = majit_gc::collector::MiniMarkGC::new();
    let root_tid = gc.register_type(majit_gc::TypeInfo::object(16));
    let child_tid = gc.register_type(majit_gc::TypeInfo::object_subclass(16, root_tid));
    let other_tid = gc.register_type(majit_gc::TypeInfo::object(16));
    let raw_tid = gc.register_type(majit_gc::TypeInfo::simple(16));
    let root_vtable: usize = 0x1240_5000 + token_id as usize;
    let other_vtable: usize = 0x1240_7000 + token_id as usize;
    let child_vtable: usize = 0x1240_6000 + token_id as usize;
    majit_gc::GcAllocator::register_vtable_for_type(&mut gc, root_vtable, root_tid);
    majit_gc::GcAllocator::register_vtable_for_type(&mut gc, other_vtable, other_tid);
    majit_gc::GcAllocator::register_vtable_for_type(&mut gc, child_vtable, child_tid);
    let child = gc.alloc_with_type(child_tid, 16);
    let raw = gc.alloc_with_type(raw_tid, 16);
    let other = gc.alloc_with_type(other_tid, 16);

    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
    majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
    backend.set_gc_allocator(Box::new(gc));
    let token = JitCellToken::new(token_id);
    (
        GcWorld {
            backend,
            child_tid: i64::from(child_tid),
            root_vtable,
            child,
            raw,
            other,
        },
        token,
    )
}

fn guard_finished(world: &mut GcWorld, token: &JitCellToken, ops: Vec<Op>, obj: GcRef) -> bool {
    let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
    let ops_rc: Vec<OpRc> = ops.into_iter().map(OpRc::new).collect();
    world
        .backend
        .compile_loop(&inputargs, &ops_rc, token)
        .unwrap_or_else(|err| panic!("compile guard: {err:?}"));
    let frame = world.backend.execute_token(token, &[Value::Ref(obj)]);
    world.backend.get_latest_descr(&frame).is_finish()
}

fn void_guard(opcode: OpCode, args: &[Operand], pos: u32) -> Op {
    let op = Op::new(opcode, args);
    op.pos().set(OpRef::void_op(pos));
    op.set_fail_arg_types(vec![]);
    op.setfailargs(vec![].into());
    op
}

fn finish(pos: u32) -> Op {
    let op = Op::new(OpCode::Finish, &[]);
    op.pos().set(OpRef::void_op(pos));
    op.set_fail_arg_types(vec![]);
    op.setfailargs(vec![].into());
    op
}

#[test]
fn int_and_low32_mask_zero_extends() {
    // genop_int_and: (1<<32)-1 is MOV32, which keeps the low 32 bits.
    let mask = (1i64 << 32) - 1;
    assert_eq!(
        compile_and_run_int(OpCode::IntAnd, Some(mask), -1, 81),
        mask
    );
    assert_eq!(
        compile_and_run_int(OpCode::IntAnd, Some(mask), 0x1_0000_0005, 82),
        5
    );
    // A negative immediate that does fit stays AND, so -1 is unchanged.
    assert_eq!(compile_and_run_int(OpCode::IntAnd, Some(-1), -1, 83), -1);
}

#[test]
fn int_force_ge_zero_clamps_negatives() {
    // genop_int_force_ge_zero: negatives become 0, non-negatives pass through.
    let cases = [(-7, 0), (0, 0), (9, 9), (i64::MIN, 0), (i64::MAX, i64::MAX)];
    let mut token_id = 90u64;
    for (input, expected) in cases {
        let got = compile_and_run_int(OpCode::IntForceGeZero, None, input, token_id);
        token_id += 1;
        assert_eq!(got, expected, "int_force_ge_zero({input})");
    }
}

#[test]
fn float_neg_and_abs_keep_zero_and_nan_payload() {
    let pos_nan = f64::from_bits(0x7ff8_0000_0000_0001);
    let neg_nan = f64::from_bits(0xfff8_0000_0000_0001);
    let (mut neg_backend, neg_token) = compile_float(OpCode::FloatNeg, 100);
    let neg_cases = [
        (0.0, (-0.0f64).to_bits()),
        (-0.0, 0.0f64.to_bits()),
        (1.5, (-1.5f64).to_bits()),
        (-2.25, 2.25f64.to_bits()),
        (pos_nan, neg_nan.to_bits()),
        (neg_nan, pos_nan.to_bits()),
    ];
    for (input, expected) in neg_cases {
        assert_eq!(
            float_bits(&mut neg_backend, &neg_token, input),
            expected,
            "float_neg({input:?})"
        );
    }

    let (mut abs_backend, abs_token) = compile_float(OpCode::FloatAbs, 101);
    let abs_cases = [
        (0.0, 0.0f64.to_bits()),
        (-0.0, 0.0f64.to_bits()),
        (-2.25, 2.25f64.to_bits()),
        (2.25, 2.25f64.to_bits()),
        (pos_nan, pos_nan.to_bits()),
        (neg_nan, pos_nan.to_bits()),
    ];
    for (input, expected) in abs_cases {
        assert_eq!(
            float_bits(&mut abs_backend, &abs_token, input),
            expected,
            "float_abs({input:?})"
        );
    }
}

#[test]
fn guard_gc_type_matches_header_typeid() {
    let (mut world, token) = gc_world(110);
    let i0 = OpRef::input_arg_ref(0);
    let expect = OpRef::const_int(world.child_tid);
    let ops = vec![
        void_guard(OpCode::GuardGcType, &[rb(i0), rb(expect)], 1),
        finish(2),
    ];
    let child = world.child;
    assert!(
        guard_finished(&mut world, &token, ops, child),
        "GUARD_GC_TYPE should pass for the object's typeid"
    );

    let (mut world, token) = gc_world(111);
    let i0 = OpRef::input_arg_ref(0);
    let expect = OpRef::const_int(world.child_tid);
    let ops = vec![
        void_guard(OpCode::GuardGcType, &[rb(i0), rb(expect)], 1),
        finish(2),
    ];
    let other = world.other;
    assert!(
        !guard_finished(&mut world, &token, ops, other),
        "GUARD_GC_TYPE should side-exit on a different typeid"
    );
}

#[test]
fn guard_is_object_tests_infobits_flag() {
    let (mut world, token) = gc_world(120);
    let i0 = OpRef::input_arg_ref(0);
    let ops = vec![void_guard(OpCode::GuardIsObject, &[rb(i0)], 1), finish(2)];
    let child = world.child;
    assert!(
        guard_finished(&mut world, &token, ops, child),
        "GUARD_IS_OBJECT should pass for an object type"
    );

    let (mut world, token) = gc_world(121);
    let i0 = OpRef::input_arg_ref(0);
    let ops = vec![void_guard(OpCode::GuardIsObject, &[rb(i0)], 1), finish(2)];
    let raw = world.raw;
    assert!(
        !guard_finished(&mut world, &token, ops, raw),
        "GUARD_IS_OBJECT should side-exit for a non-object type"
    );
}

#[test]
fn guard_subclass_typeid_range() {
    let (mut world, token) = gc_world(130);
    let i0 = OpRef::input_arg_ref(0);
    let classptr = OpRef::const_int(world.root_vtable as i64);
    let ops = vec![
        void_guard(OpCode::GuardSubclass, &[rb(i0), rb(classptr)], 1),
        finish(2),
    ];
    let child = world.child;
    assert!(
        guard_finished(&mut world, &token, ops, child),
        "GUARD_SUBCLASS should pass for a subclass"
    );

    let (mut world, token) = gc_world(131);
    let i0 = OpRef::input_arg_ref(0);
    let classptr = OpRef::const_int(world.root_vtable as i64);
    let ops = vec![
        void_guard(OpCode::GuardSubclass, &[rb(i0), rb(classptr)], 1),
        finish(2),
    ];
    let other = world.other;
    assert!(
        !guard_finished(&mut world, &token, ops, other),
        "GUARD_SUBCLASS should side-exit for an unrelated object"
    );
}
