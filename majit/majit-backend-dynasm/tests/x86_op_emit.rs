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
    CallDescr, Descr, DescrRef, EffectInfo, ExtraEffect, GcRef, InputArg, LoopTokenDescr,
    OopSpecIndex, Op, OpCode, OpRc, OpRef, Type, Value,
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

// Test-only bridge into a `CALL_MAY_FORCE` helper. The helper is `extern "C"`
// and has no Rust argument for the backend that owns the force token.
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

fn no_collect_effect() -> EffectInfo {
    let mut effect = EffectInfo::new(ExtraEffect::CannotRaise, OopSpecIndex::None);
    effect.can_collect = false;
    effect
}

fn call_descr(arg_types: Vec<Type>, result: Type, signed: bool, size: usize) -> DescrRef {
    majit_ir::descr::make_call_descr_full(0, arg_types, result, signed, size, no_collect_effect())
}

fn finish_of(result: OpRef, ty: Type, pos: u32) -> Op {
    let op = Op::new(OpCode::Finish, &[rb(result)]);
    op.pos().set(OpRef::void_op(pos));
    op.set_fail_arg_types(vec![ty]);
    op.setfailargs(vec![rb(result)].into());
    op
}

fn compile_call(
    opcode: OpCode,
    func: i64,
    real_args: &[Operand],
    arg_types: Vec<Type>,
    result: OpRef,
    result_type: Type,
    signed: bool,
    size: usize,
) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let mut args = vec![rb(OpRef::const_int(func))];
    args.extend_from_slice(real_args);
    let call = Op::new(opcode, &args);
    call.pos().set(result);
    call.setdescr(call_descr(arg_types, result_type, signed, size));
    let ops = vec![
        OpRc::new(call),
        OpRc::new(finish_of(result, result_type, 2)),
    ];
    backend
        .compile_loop(&[], &ops, &token)
        .unwrap_or_else(|err| panic!("compile {opcode:?}: {err:?}"));
    (backend, token)
}

extern "C" fn ret_i8() -> i8 {
    -42
}
extern "C" fn ret_u8() -> u8 {
    0xFE
}
extern "C" fn ret_i16() -> i16 {
    -300
}
extern "C" fn ret_u16() -> u16 {
    0xFFFE
}
extern "C" fn ret_i32() -> i32 {
    -2
}
extern "C" fn ret_u32() -> u32 {
    0xFFFF_FFFE
}

#[test]
fn call_narrow_results_are_sign_or_zero_extended() {
    // `CallBuilderX86.load_result` / `load_from_mem`: MOVSX/MOVZX/MOV32
    // on eax. A 32-bit return leaves the high half of rax clear, so a
    // missing extension is visible for every width below a word.
    let runs: Vec<(i64, bool, usize, i64)> = vec![
        (ret_i8 as usize as i64, true, 1, -42),
        (ret_u8 as usize as i64, false, 1, 0xFE),
        (ret_i16 as usize as i64, true, 2, -300),
        (ret_u16 as usize as i64, false, 2, 0xFFFE),
        (ret_i32 as usize as i64, true, 4, -2),
        (ret_u32 as usize as i64, false, 4, 0xFFFF_FFFE),
    ];
    for (func, signed, size, expected) in runs {
        let (backend, token) = compile_call(
            OpCode::CallI,
            func,
            &[],
            vec![],
            OpRef::int_op(1),
            Type::Int,
            signed,
            size,
        );
        let frame = backend.execute_token(&token, &[]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(
            backend.get_int_value(&frame, 0),
            expected,
            "size={size} signed={signed}"
        );
    }
}

static REF_BYTE: u8 = 0x5A;

extern "C" fn ref_static() -> *mut u8 {
    &REF_BYTE as *const u8 as *mut u8
}

#[test]
fn call_ref_result_is_the_pointer_word() {
    // A pointer-word ref result stays in eax.
    let (backend, token) = compile_call(
        OpCode::CallR,
        ref_static as usize as i64,
        &[],
        vec![],
        OpRef::ref_op(1),
        Type::Ref,
        false,
        8,
    );
    let frame = backend.execute_token(&token, &[]);
    assert_eq!(
        backend.get_ref_value(&frame, 0),
        GcRef(&REF_BYTE as *const u8 as usize)
    );
}

extern "C" fn neg_float(x: f64) -> f64 {
    -x
}

#[test]
fn call_float_result_is_read_from_xmm0() {
    // `load_result` does not spill xmm0. FINISH reads the register
    // `after_call` bound.
    let (backend, token) = compile_call(
        OpCode::CallF,
        neg_float as usize as i64,
        &[rb(OpRef::const_float(1.5))],
        vec![Type::Float],
        OpRef::float_op(1),
        Type::Float,
        false,
        8,
    );
    let frame = backend.execute_token(&token, &[]);
    assert!(backend.get_latest_descr(&frame).is_finish());
    assert_float_bits(backend.get_float_value(&frame, 0), -1.5, "call float");
}

#[derive(Debug)]
struct AssemblerLoopDescr {
    token: Arc<JitCellToken>,
}

impl Descr for AssemblerLoopDescr {
    fn as_loop_token_descr(&self) -> Option<&dyn LoopTokenDescr> {
        Some(self)
    }
}

impl LoopTokenDescr for AssemblerLoopDescr {
    fn loop_token_number(&self) -> u64 {
        self.token.number
    }

    fn token_handle_any(&self) -> Option<&dyn std::any::Any> {
        Some(&self.token)
    }
}

fn compile_identity_loop(backend: &mut DynasmBackend, ty: Type) -> Arc<JitCellToken> {
    let token = Arc::new(JitCellToken::new(next_token_id()));
    let inputargs = vec![InputArg::from_type_rc(ty, 0)];
    let i0 = inputargs[0].opref();
    let ops = vec![OpRc::new(finish_of(i0, ty, 1))];
    backend
        .compile_loop(&inputargs, &ops, &token)
        .unwrap_or_else(|err| panic!("compile identity {ty:?}: {err:?}"));
    assert_ne!(token.ll_function_addr(), 0, "callee entry was not baked");
    token
}

/// `handle_call_assembler` reads `jitframe_info` and allocates the callee
/// frame itself. The layout is process-wide and set-once; the tid has to
/// be the one this thread's collector registered.
fn backend_with_call_assembler_layout() -> DynasmBackend {
    use majit_backend_dynasm::jitframe::{
        FIRST_ITEM_OFFSET, JF_DESCR_OFS, JF_FORCE_DESCR_OFS, JF_FORWARD_OFS, JF_FRAME_INFO_OFS,
        JF_FRAME_OFS, JF_GUARD_EXC_OFS, JF_SAVEDATA_OFS, JITFRAME_FIXED_SIZE, LENGTHOFS, SIGN_SIZE,
    };
    let mut gc = majit_gc::collector::MiniMarkGC::new();
    let jitframe_tid = gc.register_type(majit_backend_dynasm::jitframe::jitframe_type_info());
    majit_gc::GcAllocator::set_jitframe_type_id(&mut gc, jitframe_tid);
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    backend.set_gc_allocator(Box::new(gc));
    majit_backend_dynasm::register_jitframe_layout(majit_backend_dynasm::JitFrameLayoutInfo {
        jitframe_descrs: Some(majit_gc::rewrite::JitFrameDescrs {
            jitframe_tid,
            jitframe_fixed_size: JITFRAME_FIXED_SIZE,
            jf_frame_info_ofs: JF_FRAME_INFO_OFS,
            jf_descr_ofs: JF_DESCR_OFS,
            jf_force_descr_ofs: JF_FORCE_DESCR_OFS,
            jf_savedata_ofs: JF_SAVEDATA_OFS,
            jf_guard_exc_ofs: JF_GUARD_EXC_OFS,
            jf_forward_ofs: JF_FORWARD_OFS,
            jf_frame_ofs: JF_FRAME_OFS,
            jf_frame_baseitemofs: FIRST_ITEM_OFFSET,
            jf_frame_lengthofs: JF_FRAME_OFS + LENGTHOFS,
            sign_size: SIGN_SIZE,
            jf_frame_itemsize: SIGN_SIZE,
        }),
    });
    backend
}

#[test]
fn call_assembler_returns_int_and_float_from_the_dead_frame() {
    // `call_assembler` / `_call_assembler_load_result`: the fast path
    // loads value index 0. Same backend so the done-descr compare hits.
    // `handle_call_assembler` turns the value argument into the callee
    // frame; the callee is registered on this thread first.
    let mut backend = backend_with_call_assembler_layout();
    let int_callee = compile_identity_loop(&mut backend, Type::Int);
    let float_callee = compile_identity_loop(&mut backend, Type::Float);

    let int_inputs = vec![InputArg::from_type_rc(Type::Int, 0)];
    let i0 = int_inputs[0].opref();
    let int_token = JitCellToken::new(next_token_id());
    let int_call = Op::new(OpCode::CallAssemblerI, &[rb(i0)]);
    int_call.pos().set(OpRef::int_op(1));
    int_call.setdescr(Arc::new(AssemblerLoopDescr {
        token: Arc::clone(&int_callee),
    }) as DescrRef);
    let int_ops = vec![
        OpRc::new(int_call),
        OpRc::new(finish_of(OpRef::int_op(1), Type::Int, 2)),
    ];
    backend
        .compile_loop(&int_inputs, &int_ops, &int_token)
        .unwrap_or_else(|err| panic!("compile CALL_ASSEMBLER_I: {err:?}"));
    let int_frame = backend.execute_token(&int_token, &[Value::Int(42)]);
    assert!(backend.get_latest_descr(&int_frame).is_finish());
    assert_eq!(backend.get_int_value(&int_frame, 0), 42);

    let float_inputs = vec![InputArg::from_type_rc(Type::Float, 0)];
    let f0 = float_inputs[0].opref();
    let float_token = JitCellToken::new(next_token_id());
    let float_call = Op::new(OpCode::CallAssemblerF, &[rb(f0)]);
    float_call.pos().set(OpRef::float_op(1));
    float_call.setdescr(Arc::new(AssemblerLoopDescr {
        token: Arc::clone(&float_callee),
    }) as DescrRef);
    let float_ops = vec![
        OpRc::new(float_call),
        OpRc::new(finish_of(OpRef::float_op(1), Type::Float, 2)),
    ];
    backend
        .compile_loop(&float_inputs, &float_ops, &float_token)
        .unwrap_or_else(|err| panic!("compile CALL_ASSEMBLER_F: {err:?}"));
    let float_frame = backend.execute_token(&float_token, &[Value::Float(-2.5)]);
    assert!(backend.get_latest_descr(&float_frame).is_finish());
    assert_float_bits(
        backend.get_float_value(&float_frame, 0),
        -2.5,
        "call_assembler float",
    );
    let _keep = (int_callee, float_callee);
}

fn errno_ptr() -> *mut i32 {
    #[cfg(target_os = "windows")]
    unsafe {
        unsafe extern "C" {
            fn _errno() -> *mut i32;
        }
        _errno()
    }
    #[cfg(not(target_os = "windows"))]
    unsafe {
        unsafe extern "C" {
            fn __errno_location() -> *mut i32;
        }
        __errno_location()
    }
}

extern "C" fn swap_errno(new_value: i64) -> i64 {
    let slot = errno_ptr();
    let old = unsafe { *slot } as i64;
    unsafe {
        *slot = new_value as i32;
    }
    old
}

fn errno_save_flags(base: i64) -> i64 {
    // `RFFI_READSAVED_LASTERROR`: Win64 spills the used argument
    // registers around `SetLastError`. Errno itself is unchanged.
    #[cfg(target_os = "windows")]
    {
        base | 16
    }
    #[cfg(not(target_os = "windows"))]
    {
        base
    }
}

fn run_errno_swap(flags: i64, new_value: i64) -> i64 {
    let (backend, token) = compile_release_gil(flags, new_value);
    let frame = backend.execute_token(&token, &[]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "save_err={flags:#x}"
    );
    backend.get_int_value(&frame, 0)
}

fn compile_release_gil(flags: i64, new_value: i64) -> (DynasmBackend, JitCellToken) {
    let mut backend = fresh_backend();
    let token = JitCellToken::new(next_token_id());
    let call = Op::new(
        OpCode::CallReleaseGilI,
        &[
            rb(OpRef::const_int(flags)),
            rb(OpRef::const_int(swap_errno as usize as i64)),
            rb(OpRef::const_int(new_value)),
        ],
    );
    call.pos().set(OpRef::int_op(1));
    call.setdescr(call_descr(vec![Type::Int], Type::Int, true, 8));
    let ops = vec![
        OpRc::new(call),
        OpRc::new(finish_of(OpRef::int_op(1), Type::Int, 2)),
    ];
    backend
        .compile_loop(&[], &ops, &token)
        .unwrap_or_else(|err| panic!("compile CALL_RELEASE_GIL: {err:?}"));
    (backend, token)
}

#[test]
fn call_release_gil_round_trips_errno() {
    // `write_real_errno` / `read_real_errno`: zero, restore the saved
    // copy, then read that copy back. The thread-local container
    // survives across the three calls.
    unsafe {
        *errno_ptr() = 11;
    }
    assert_eq!(run_errno_swap(errno_save_flags(5), 77), 0);
    unsafe {
        *errno_ptr() = 5;
    }
    assert_eq!(run_errno_swap(errno_save_flags(3), 9), 77);
    unsafe {
        *errno_ptr() = 3;
    }
    assert_eq!(run_errno_swap(errno_save_flags(2), 0), 9);
}
