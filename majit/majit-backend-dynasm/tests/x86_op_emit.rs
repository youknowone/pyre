//! Execution checks for the x86 emitters ported onto the upstream sequences:
//! `genop_int_and`, `genop_int_force_ge_zero`, `genop_float_neg`,
//! `genop_float_abs`, `_cmp_guard_gc_type`, `genop_guard_guard_is_object`,
//! `genop_guard_guard_subclass`, `_binaryop`, `_cmpop_float`,
//! `_store_force_index`, `store_force_descr`, `genop_guard_guard_no_exception`.

#![cfg(target_arch = "x86_64")]

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use majit_backend::{Backend, JitCellToken, make_resume_guard_descr_typed};
use majit_backend_dynasm::runner::DynasmBackend;
use majit_ir::forwarding::bound_operand_from_opref as rb;
use majit_ir::operand::Operand;
use majit_ir::{
    CallDescr, Descr, DescrRef, EffectInfo, ExtraEffect, FailDescr, GcRef, InputArg, OopSpecIndex,
    Op, OpCode, OpRc, OpRef, Type, Value,
};

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

fn next_token_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0x00D4_0001);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn fresh_backend() -> DynasmBackend {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    backend
}

fn assert_float_bits(got: f64, expected: f64, what: &str) {
    if expected.is_nan() {
        assert!(got.is_nan(), "{what}: expected NaN, got {got}");
    } else {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "{what}: got {got}, expected {expected}"
        );
    }
}

fn host_float_binop(opcode: OpCode, lhs: f64, rhs: f64) -> f64 {
    match opcode {
        OpCode::FloatAdd => lhs + rhs,
        OpCode::FloatSub => lhs - rhs,
        OpCode::FloatMul => lhs * rhs,
        OpCode::FloatTrueDiv => lhs / rhs,
        _ => unreachable!("{opcode:?} is not a float binop"),
    }
}

/// `_binaryop` / `_consider_float_op`: one side is a constant (`ConstFloatLoc`
/// or, when it is the accumulator, the register `force_result_in_reg` filled).
fn compile_float_binop(
    opcode: OpCode,
    const_on_left: bool,
    konst: f64,
) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let i0 = inputargs[0].opref();
    let c = OpRef::const_float(konst);
    let args = if const_on_left {
        [rb(c), rb(i0)]
    } else {
        [rb(i0), rb(c)]
    };
    let body = Op::new(opcode, &args);
    body.pos().set(OpRef::float_op(1));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![Type::Float]);
    finish.setfailargs(vec![rb(OpRef::float_op(1))].into());
    let ops = vec![OpRc::new(body), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| {
            panic!("compile {opcode:?} const_on_left={const_on_left} konst={konst}: {err:?}")
        });
    (backend, token)
}

fn run_float_binop(backend: &mut DynasmBackend, token: &JitCellToken, input: f64) -> f64 {
    let frame = backend.execute_token(token, &[Value::Float(input)]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "float binop input={input}"
    );
    backend.get_float_value(&frame, 0)
}

#[test]
fn float_binop_reads_const_operand_and_nan() {
    // `_binaryop("ADDSD"|"SUBSD"|"MULSD"|"DIVSD")` keeps the non-accumulator
    // operand in place, including a constant pool load and a NaN input.
    let finite = [
        (OpCode::FloatAdd, 0.5, [1.5, 0.0, -0.5]),
        (OpCode::FloatSub, 2.0, [5.0, 2.0, 0.0]),
        (OpCode::FloatMul, 2.0, [3.0, -1.5, 0.0]),
        (OpCode::FloatTrueDiv, 2.0, [6.0, -4.0, 2.0]),
    ];
    for (opcode, konst, inputs) in finite {
        for const_on_left in [false, true] {
            let (mut backend, token) = compile_float_binop(opcode, const_on_left, konst);
            for input in inputs {
                let (lhs, rhs) = if const_on_left {
                    (konst, input)
                } else {
                    (input, konst)
                };
                let got = run_float_binop(&mut backend, &token, input);
                assert_float_bits(
                    got,
                    host_float_binop(opcode, lhs, rhs),
                    &format!("{opcode:?} const_on_left={const_on_left} {lhs} . {rhs}"),
                );
            }
            let nan_input = f64::from_bits(0x7ff8_0000_0000_0001);
            let (lhs, rhs) = if const_on_left {
                (konst, nan_input)
            } else {
                (nan_input, konst)
            };
            let got = run_float_binop(&mut backend, &token, nan_input);
            assert_float_bits(
                got,
                host_float_binop(opcode, lhs, rhs),
                &format!("{opcode:?} NaN input const_on_left={const_on_left}"),
            );
        }
    }
    let nan_const = f64::from_bits(0x7ff8_0000_0000_0001);
    for opcode in [
        OpCode::FloatAdd,
        OpCode::FloatSub,
        OpCode::FloatMul,
        OpCode::FloatTrueDiv,
    ] {
        for const_on_left in [false, true] {
            let (mut backend, token) = compile_float_binop(opcode, const_on_left, nan_const);
            let input = 1.5;
            let (lhs, rhs) = if const_on_left {
                (nan_const, input)
            } else {
                (input, nan_const)
            };
            let got = run_float_binop(&mut backend, &token, input);
            assert_float_bits(
                got,
                host_float_binop(opcode, lhs, rhs),
                &format!("{opcode:?} NaN const const_on_left={const_on_left}"),
            );
        }
    }
}

fn host_float_cmp(opcode: OpCode, lhs: f64, rhs: f64) -> i64 {
    let pass = match opcode {
        OpCode::FloatLt => lhs < rhs,
        OpCode::FloatLe => lhs <= rhs,
        OpCode::FloatEq => lhs == rhs,
        OpCode::FloatNe => lhs != rhs,
        OpCode::FloatGt => lhs > rhs,
        OpCode::FloatGe => lhs >= rhs,
        _ => unreachable!("{opcode:?} is not a float compare"),
    };
    i64::from(pass)
}

/// `_cmpop_float` / `_consider_float_cmp`: the constant stays a memory
/// operand and the other side is forced into a register.
fn compile_float_cmp(
    opcode: OpCode,
    const_on_left: bool,
    konst: f64,
) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let i0 = inputargs[0].opref();
    let c = OpRef::const_float(konst);
    let args = if const_on_left {
        [rb(c), rb(i0)]
    } else {
        [rb(i0), rb(c)]
    };
    let body = Op::new(opcode, &args);
    body.pos().set(OpRef::int_op(1));
    let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![Type::Int]);
    finish.setfailargs(vec![rb(OpRef::int_op(1))].into());
    let ops = vec![OpRc::new(body), OpRc::new(finish)];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| {
            panic!("compile {opcode:?} const_on_left={const_on_left} konst={konst}: {err:?}")
        });
    (backend, token)
}

fn run_float_cmp(backend: &mut DynasmBackend, token: &JitCellToken, input: f64) -> i64 {
    let frame = backend.execute_token(token, &[Value::Float(input)]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "float cmp input={input}"
    );
    backend.get_int_value(&frame, 0)
}

#[test]
fn float_cmp_reads_const_operand_and_nan() {
    // `_cmpop_float`: unordered compares are false for LT/LE/EQ/GT/GE and
    // true for NE. `_if_parity_clear_zero_and_carry` is what makes B/E/BE
    // reject NaN. Both operand orders exercise `_consider_float_cmp`.
    let opcodes = [
        OpCode::FloatLt,
        OpCode::FloatLe,
        OpCode::FloatEq,
        OpCode::FloatNe,
        OpCode::FloatGt,
        OpCode::FloatGe,
    ];
    let inputs = [-1.0, 0.0, 1.0, 2.0, f64::NAN, -0.0];
    for opcode in opcodes {
        for const_on_left in [false, true] {
            for konst in [1.0, f64::NAN] {
                let (mut backend, token) = compile_float_cmp(opcode, const_on_left, konst);
                for input in inputs {
                    let (lhs, rhs) = if const_on_left {
                        (konst, input)
                    } else {
                        (input, konst)
                    };
                    let got = run_float_cmp(&mut backend, &token, input);
                    assert_eq!(
                        got,
                        host_float_cmp(opcode, lhs, rhs),
                        "{opcode:?} const_on_left={const_on_left} {lhs} ? {rhs}"
                    );
                }
            }
        }
    }
}

/// Test-only bridge into a `CALL_MAY_FORCE` helper. The helper is `extern "C"`
/// and has no Rust argument for the backend that owns the force token.
thread_local! {
    static FORCE_BACKEND: Cell<*const DynasmBackend> = const { Cell::new(std::ptr::null()) };
    static FORCE_SEEN: Cell<Option<i64>> = const { Cell::new(None) };
}

struct ClearForceBackend;

impl Drop for ClearForceBackend {
    fn drop(&mut self) {
        FORCE_BACKEND.with(|cell| cell.set(std::ptr::null()));
    }
}

extern "C" fn force_and_record(force_token: i64) {
    let ptr = FORCE_BACKEND.with(Cell::get);
    if ptr.is_null() {
        FORCE_SEEN.with(|cell| cell.set(Some(i64::MIN)));
        return;
    }
    let backend = unsafe { &*ptr };
    let recorded = catch_unwind(AssertUnwindSafe(|| {
        match backend.force(GcRef(force_token as usize)) {
            Some(frame) => backend.get_int_value(&frame, 0),
            None => -2,
        }
    }));
    let value = match recorded {
        Ok(value) => value,
        Err(_) => -3,
    };
    FORCE_SEEN.with(|cell| cell.set(Some(value)));
}

extern "C" fn force_nop(_force_token: i64) {}

#[derive(Debug)]
struct MayForceCallDescr {
    arg_types: Vec<Type>,
}

impl Descr for MayForceCallDescr {
    fn index(&self) -> u32 {
        u32::MAX
    }

    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
}

impl CallDescr for MayForceCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }

    fn result_type(&self) -> Type {
        Type::Void
    }

    fn result_size(&self) -> usize {
        0
    }

    fn get_extra_info(&self) -> &EffectInfo {
        static INFO: EffectInfo = EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
        &INFO
    }
}

/// `_store_force_index` then `genop_guard_guard_not_forced`.
fn compile_may_force(helper: extern "C" fn(i64)) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let force_token = Op::new(OpCode::ForceToken, &[]);
    force_token.pos().set(OpRef::ref_op(1));

    let call = Op::new(
        OpCode::CallMayForceN,
        &[
            rb(OpRef::const_int(helper as usize as i64)),
            rb(OpRef::ref_op(1)),
        ],
    );
    call.pos().set(OpRef::void_op(2));
    call.setdescr(Arc::new(MayForceCallDescr {
        arg_types: vec![Type::Ref],
    }) as DescrRef);

    let guard = Op::new(OpCode::GuardNotForced, &[]);
    guard.pos().set(OpRef::void_op(3));
    guard.setdescr(make_resume_guard_descr_typed(vec![Type::Int]));
    guard.set_fail_arg_types(vec![Type::Int]);
    guard.setfailargs(vec![rb(i0)].into());

    let finish = Op::new(OpCode::Finish, &[rb(i0)]);
    finish.pos().set(OpRef::void_op(4));
    finish.set_fail_arg_types(vec![Type::Int]);
    finish.setfailargs(vec![rb(i0)].into());

    let ops: Vec<OpRc> = vec![force_token, call, guard, finish]
        .into_iter()
        .map(OpRc::new)
        .collect();
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile CALL_MAY_FORCE: {err:?}"));
    (backend, token)
}

fn run_may_force(helper: extern "C" fn(i64)) -> (bool, i64, Option<i64>) {
    FORCE_SEEN.with(|cell| cell.set(None));
    let (backend, token) = compile_may_force(helper);
    FORCE_BACKEND.with(|cell| cell.set(&backend as *const DynasmBackend));
    let _clear = ClearForceBackend;
    let frame = backend.execute_token(&token, &[Value::Int(42)]);
    let finished = backend.get_latest_descr(&frame).is_finish();
    let after = backend.get_int_value(&frame, 0);
    let during = FORCE_SEEN.with(Cell::get);
    (finished, after, during)
}

#[test]
fn may_force_guard_not_forced_reads_failarg() {
    // `_store_force_index` publishes `jf_force_descr` and leaves `jf_descr`
    // alone. A helper that does not force falls through `GUARD_NOT_FORCED`
    // (`CMP [jf_descr], 0`). A helper that forces copies that descr into
    // `jf_descr`, so the guard fails and the live fail arg is readable both
    // during `force` and on the side exit.
    let (finished, after, during) = run_may_force(force_nop);
    assert!(during.is_none(), "nop helper must not force");
    assert!(finished, "GUARD_NOT_FORCED must pass when nothing forces");
    assert_eq!(after, 42);

    let (finished, after, during) = run_may_force(force_and_record);
    assert_eq!(during, Some(42), "force() must observe the live fail arg");
    assert!(!finished, "GUARD_NOT_FORCED must fail after force()");
    assert_eq!(after, 42, "side exit must keep the live fail arg");
}

#[test]
fn guard_not_forced_2_constant_failarg_is_hole() {
    // `store_force_descr` → `store_info_on_descr`: an absent fail arg is
    // None → 0xFFFF, and `get_int_value` of that hole is 0. Failargs must
    // not contain Const (`compute_vars_longevity`). The live fail arg is
    // still the spilled input, readable after `force`.
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let force_token = Op::new(OpCode::ForceToken, &[]);
    force_token.pos().set(OpRef::ref_op(1));

    let guard = Op::new(OpCode::GuardNotForced2, &[]);
    guard.pos().set(OpRef::void_op(2));
    guard.setdescr(make_resume_guard_descr_typed(vec![Type::Int, Type::Int]));
    guard.set_fail_arg_types(vec![Type::Int, Type::Int]);
    guard.setfailargs(vec![rb(i0), Operand::none()].into());

    let finish = Op::new(OpCode::Finish, &[rb(OpRef::ref_op(1))]);
    finish.pos().set(OpRef::void_op(3));
    finish.set_fail_arg_types(vec![Type::Ref]);
    finish.setfailargs(vec![rb(OpRef::ref_op(1))].into());

    let ops: Vec<OpRc> = vec![force_token, guard, finish]
        .into_iter()
        .map(OpRc::new)
        .collect();
    backend
        .compile_loop(&inputargs, &ops, &token)
        .expect("compile GUARD_NOT_FORCED_2");

    let returned = backend.execute_token(&token, &[Value::Int(42)]);
    assert!(backend.get_latest_descr(&returned).is_finish());
    let raw_token = backend.get_ref_value(&returned, 0);
    assert!(backend.is_force_token_armed(raw_token));
    let forced = backend.force(raw_token).expect("armed force token");
    let descr = backend.get_latest_descr(&forced);
    assert!(!descr.is_finish());
    assert_eq!(descr.rd_locs().get(1), Some(&0xFFFF));
    assert_eq!(backend.get_int_value(&forced, 0), 42);
    assert_eq!(backend.get_int_value(&forced, 1), 0);
}

static COND_CALL_EXC_LOCK: Mutex<()> = Mutex::new(());
static EXC_OBJ: [i64; 2] = [0x5151_0D11, 0];

struct ClearExc;

impl Drop for ClearExc {
    fn drop(&mut self) {
        majit_backend_dynasm::jit_exc_clear();
    }
}

extern "C" fn cond_call_raise() {
    majit_backend_dynasm::jit_exc_raise(EXC_OBJ.as_ptr() as i64);
}

extern "C" fn cond_call_quiet() {}

/// `genop_guard_guard_no_exception`: `COND_CALL` + `GUARD_NO_EXCEPTION`.
fn compile_cond_call_no_exception(
    predicate: i64,
    func: extern "C" fn(),
) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let call = Op::new(
        OpCode::CondCallN,
        &[
            rb(OpRef::const_int(predicate)),
            rb(OpRef::const_int(func as usize as i64)),
        ],
    );
    call.pos().set(OpRef::void_op(0));
    let guard = Op::new(OpCode::GuardNoException, &[]);
    guard.pos().set(OpRef::void_op(1));
    guard.set_fail_arg_types(vec![]);
    guard.setfailargs(vec![].into());
    let finish = Op::new(OpCode::Finish, &[]);
    finish.pos().set(OpRef::void_op(2));
    finish.set_fail_arg_types(vec![]);
    finish.setfailargs(vec![].into());
    let ops = vec![OpRc::new(call), OpRc::new(guard), OpRc::new(finish)];
    backend
        .compile_loop(&[], &ops, &token)
        .unwrap_or_else(|err| panic!("compile COND_CALL + GUARD_NO_EXCEPTION: {err:?}"));
    (backend, token)
}

#[test]
fn cond_call_guard_no_exception_checks_the_call_path() {
    // `genop_guard_guard_no_exception` / `generate_guard_no_exception`: the
    // fast path of a false `COND_CALL` emits no exception check, so a
    // pre-raised exception is still pending at `FINISH`. The call path runs
    // `CMP [pos_exception], 0` and, on failure, `grab_exc_value` reads the
    // value the recovery stub stashed.
    let _lock = COND_CALL_EXC_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let _clear = ClearExc;
    majit_backend_dynasm::jit_exc_clear();

    let (backend, token) = compile_cond_call_no_exception(0, cond_call_raise);
    majit_backend_dynasm::jit_exc_raise(EXC_OBJ.as_ptr() as i64);
    let frame = backend.execute_token(&token, &[]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "predicate 0 must skip the call and the exception check"
    );
    assert!(
        majit_backend_dynasm::jit_exc_is_pending(),
        "the don't-call edge must leave the pre-raised exception pending"
    );
    majit_backend_dynasm::jit_exc_clear();

    let (backend, token) = compile_cond_call_no_exception(1, cond_call_raise);
    let frame = backend.execute_token(&token, &[]);
    assert!(
        !backend.get_latest_descr(&frame).is_finish(),
        "a raising call must fail GUARD_NO_EXCEPTION"
    );
    assert_eq!(
        backend.grab_exc_value(&frame),
        GcRef(EXC_OBJ.as_ptr() as usize)
    );
    assert!(!majit_backend_dynasm::jit_exc_is_pending());
    majit_backend_dynasm::jit_exc_clear();

    let (backend, token) = compile_cond_call_no_exception(1, cond_call_quiet);
    let frame = backend.execute_token(&token, &[]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "a quiet call must pass GUARD_NO_EXCEPTION"
    );
    assert!(!majit_backend_dynasm::jit_exc_is_pending());
}
