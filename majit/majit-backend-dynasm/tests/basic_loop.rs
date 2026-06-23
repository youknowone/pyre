/// Basic test: compile and execute a simple int_add loop via dynasm backend.
///
/// Trace: i0 = input
///   label(i0)
///   i1 = int_add(i0, 1)
///   i2 = int_lt(i1, 10)
///   guard_true(i2)  [fail_args: i1]
///   jump(i1)        → label
///   finish(i1)      [on guard failure]
use std::rc::Rc;
use std::sync::{LazyLock, Mutex};

use majit_backend::{Backend, JitCellToken};
use majit_ir::{
    GcRef, InputArg, Op, OpCode, OpRef, Type, Value, make_array_descr, make_loop_target_descr,
};

use majit_backend_dynasm::runner::DynasmBackend;

static EXCEPTION_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

use majit_ir::box_ref::bound_operand_from_opref as rb;

#[test]
fn test_just_finish() {
    // Simplest possible trace: just FINISH with no args
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let mut token = JitCellToken::new(1);

    let inputargs = vec![];

    let finish_op = Op::new(OpCode::Finish, &[]);
    finish_op.pos.set(OpRef::void_op(0));
    finish_op.set_fail_arg_types(vec![]);
    finish_op.setfailargs(vec![].into());

    let ops = vec![finish_op];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();

    let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[]);
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());
}

#[test]
fn test_simple_int_add() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let mut token = JitCellToken::new(1);

    // Simple trace: i1 = int_add(i0, CONST_1)
    // finish(i1)  [fail_arg_types: [Int], fail_args: [i1]]
    // history.py:227 ConstInt.value inline.
    let const_1 = OpRef::const_int(1);

    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let add_op = Op::new(OpCode::IntAdd, &[rb(i0), rb(const_1)]);
    add_op.pos.set(OpRef::int_op(1)); // result is OpRef::int_op(1)

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish_op.pos.set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![Type::Int]);
    finish_op.setfailargs(vec![rb(OpRef::int_op(1)).to_boxref()].into());

    let ops = vec![add_op, finish_op];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();

    // Compile
    let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    // Execute with input i0 = 42
    let args = vec![Value::Int(42)];
    let frame = backend.execute_token(&token, &args);

    // Check result
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());

    let result_val = backend.get_int_value(&frame, 0);
    assert_eq!(result_val, 43, "42 + 1 should be 43");
}

#[test]
fn test_finish_infers_int_type_when_explicit_types_are_empty() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let mut token = JitCellToken::new(11);

    let const_1 = OpRef::const_int(1);

    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let add_op = Op::new(OpCode::IntAdd, &[rb(i0), rb(const_1)]);
    add_op.pos.set(OpRef::int_op(1));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish_op.pos.set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![]);
    finish_op.setfailargs(vec![rb(OpRef::int_op(1)).to_boxref()].into());

    let ops = vec![add_op, finish_op];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();

    let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[Value::Int(42)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());
    assert_eq!(descr.fail_arg_types(), &[Type::Int]);
    assert_eq!(backend.get_int_value(&frame, 0), 43);
}

#[test]
fn test_float_add() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let mut token = JitCellToken::new(1);

    let i0 = OpRef::input_arg_float(0); // input: f64
    // history.py:268 ConstFloat.value inline.
    let const_half = OpRef::const_float(0.5);

    let inputargs = vec![InputArg::from_type(Type::Float, 0)];

    let add_op = Op::new(OpCode::FloatAdd, &[rb(i0), rb(const_half)]);
    add_op.pos.set(OpRef::float_op(1));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
    finish_op.pos.set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![Type::Float]);
    finish_op.setfailargs(vec![rb(OpRef::float_op(1)).to_boxref()].into());

    let ops = vec![add_op, finish_op];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();

    let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    // Execute with input = 1.5
    let args = vec![Value::Float(1.5)];
    let frame = backend.execute_token(&token, &args);

    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());

    let result_val = backend.get_float_value(&frame, 0);
    assert!(
        (result_val - 2.0).abs() < 1e-10,
        "1.5 + 0.5 should be 2.0, got {}",
        result_val
    );
}

#[test]
fn test_setarrayitem_raw_float_roundtrip() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let mut token = JitCellToken::new(23);

    let const_index = OpRef::const_int(3);

    let array_descr = make_array_descr(0, 8, Type::Float);

    let inputargs = vec![
        InputArg::from_type(Type::Ref, 0),
        InputArg::from_type(Type::Float, 1),
    ];
    let base = inputargs[0].opref();
    let value = inputargs[1].opref();

    let set_op = Op::new(
        OpCode::SetarrayitemRaw,
        &[rb(base), rb(const_index), rb(value)],
    );
    set_op.pos.set(OpRef::void_op(2));
    set_op.setdescr(array_descr.clone());

    let get_op = Op::new(OpCode::GetarrayitemRawF, &[rb(base), rb(const_index)]);
    get_op.pos.set(OpRef::float_op(3));
    get_op.setdescr(array_descr);

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::float_op(3))]);
    finish_op.pos.set(OpRef::void_op(4));
    finish_op.set_fail_arg_types(vec![Type::Float]);
    finish_op.setfailargs(vec![rb(OpRef::float_op(3)).to_boxref()].into());

    let ops = vec![set_op, get_op, finish_op];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();
    let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let mut items = vec![0.0f64; 8];
    let base_ref = GcRef(items.as_mut_ptr() as usize);
    let frame = backend.execute_token(&token, &[Value::Ref(base_ref), Value::Float(1.25)]);

    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());
    assert_eq!(items[3], 1.25);
    assert_eq!(backend.get_float_value(&frame, 0), 1.25);
}

#[test]
fn test_setarrayitem_raw_float_roundtrip_with_variable_index() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let mut token = JitCellToken::new(24);

    let array_descr = make_array_descr(0, 8, Type::Float);

    let inputargs = vec![
        InputArg::from_type(Type::Ref, 0),
        InputArg::from_type(Type::Int, 1),
        InputArg::from_type(Type::Float, 2),
    ];
    let base = inputargs[0].opref();
    let index = inputargs[1].opref();
    let value = inputargs[2].opref();

    let set_op = Op::new(OpCode::SetarrayitemRaw, &[rb(base), rb(index), rb(value)]);
    set_op.pos.set(OpRef::void_op(3));
    set_op.setdescr(array_descr.clone());

    let get_op = Op::new(OpCode::GetarrayitemRawF, &[rb(base), rb(index)]);
    get_op.pos.set(OpRef::float_op(4));
    get_op.setdescr(array_descr);

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::float_op(4))]);
    finish_op.pos.set(OpRef::void_op(5));
    finish_op.set_fail_arg_types(vec![Type::Float]);
    finish_op.setfailargs(vec![rb(OpRef::float_op(4)).to_boxref()].into());

    let ops = vec![set_op, get_op, finish_op];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();
    let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let mut items = vec![0.0f64; 8];
    let base_ref = GcRef(items.as_mut_ptr() as usize);
    let frame = backend.execute_token(
        &token,
        &[Value::Ref(base_ref), Value::Int(5), Value::Float(-2.5)],
    );

    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish());
    assert_eq!(items[5], -2.5);
    assert_eq!(backend.get_float_value(&frame, 0), -2.5);
}

#[test]
fn test_guard_and_loop() {
    // Trace: loop that adds 1 until >= 5, then guard fails
    // i0 = input
    // label(i0)
    // i1 = int_add(i0, CONST_1)
    // i2 = int_lt(i1, CONST_5)
    // guard_true(i2)   [fail_args: i1]
    // jump(i1)
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let mut token = JitCellToken::new(1);

    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let loop_descr = make_loop_target_descr(token.number, false);

    let label_op = Op::new(OpCode::Label, &[rb(OpRef::input_arg_int(0))]);
    label_op.pos.set(OpRef::void_op(100));
    label_op.setdescr(loop_descr.clone());

    let add_op = Op::new(
        OpCode::IntAdd,
        &[rb(OpRef::input_arg_int(0)), rb(OpRef::const_int(1))],
    );
    add_op.pos.set(OpRef::int_op(1));

    let lt_op = Op::new(
        OpCode::IntLt,
        &[rb(OpRef::int_op(1)), rb(OpRef::const_int(5))],
    );
    lt_op.pos.set(OpRef::int_op(2));

    let guard_op = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(2))]);
    guard_op.pos.set(OpRef::void_op(3));
    guard_op.set_fail_arg_types(vec![Type::Int]);
    guard_op.setfailargs(vec![rb(OpRef::int_op(1)).to_boxref()].into());

    let jump_op = Op::new(OpCode::Jump, &[rb(OpRef::int_op(1))]);
    jump_op.pos.set(OpRef::void_op(4));
    jump_op.setdescr(loop_descr);

    let ops = vec![label_op, add_op, lt_op, guard_op, jump_op];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();

    let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    // Start with 0: should loop 0→1→2→3→4→5, guard fails at 5
    let args = vec![Value::Int(0)];
    let frame = backend.execute_token(&token, &args);

    let descr = backend.get_latest_descr(&frame);
    assert!(!descr.is_finish(), "should be guard failure, not finish");

    // fail_args = [OpRef::int_op(1)], so fail_arg index 0 = the IntAdd result.
    let result_val = backend.get_int_value(&frame, 0);
    assert_eq!(result_val, 5, "loop should stop at 5, fail_arg[0]");
}

#[test]
fn test_float_loop_carried_across_jump() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let mut token = JitCellToken::new(1);

    let inputargs = vec![
        InputArg::from_type(Type::Float, 0),
        InputArg::from_type(Type::Int, 1),
    ];
    let loop_descr = make_loop_target_descr(token.number, false);

    let label_op = Op::new(
        OpCode::Label,
        &[rb(OpRef::input_arg_float(0)), rb(OpRef::input_arg_int(1))],
    );
    label_op.pos.set(OpRef::void_op(100));
    label_op.setdescr(loop_descr.clone());

    let lt_op = Op::new(
        OpCode::IntLt,
        &[rb(OpRef::input_arg_int(1)), rb(OpRef::const_int(5))],
    );
    lt_op.pos.set(OpRef::int_op(2));

    let guard_op = Op::new(OpCode::GuardTrue, &[rb(OpRef::int_op(2))]);
    guard_op.pos.set(OpRef::void_op(3));
    guard_op.set_fail_arg_types(vec![Type::Float, Type::Int]);
    guard_op.setfailargs(
        vec![
            rb(OpRef::input_arg_float(0)).to_boxref(),
            rb(OpRef::input_arg_int(1)).to_boxref(),
        ]
        .into(),
    );

    let cast_op = Op::new(OpCode::CastIntToFloat, &[rb(OpRef::input_arg_int(1))]);
    cast_op.pos.set(OpRef::float_op(4));

    let mul_op = Op::new(
        OpCode::FloatMul,
        &[rb(OpRef::float_op(4)), rb(OpRef::const_float(0.5))],
    );
    mul_op.pos.set(OpRef::float_op(5));

    let add_op = Op::new(
        OpCode::FloatAdd,
        &[rb(OpRef::input_arg_float(0)), rb(OpRef::float_op(5))],
    );
    add_op.pos.set(OpRef::float_op(6));

    let inc_op = Op::new(
        OpCode::IntAdd,
        &[rb(OpRef::input_arg_int(1)), rb(OpRef::const_int(1))],
    );
    inc_op.pos.set(OpRef::int_op(7));

    let jump_op = Op::new(
        OpCode::Jump,
        &[rb(OpRef::float_op(6)), rb(OpRef::int_op(7))],
    );
    jump_op.pos.set(OpRef::void_op(8));
    jump_op.setdescr(loop_descr);

    let ops = vec![
        label_op, lt_op, guard_op, cast_op, mul_op, add_op, inc_op, jump_op,
    ];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();

    let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let args = vec![Value::Float(0.0), Value::Int(0)];
    let frame = backend.execute_token(&token, &args);

    let descr = backend.get_latest_descr(&frame);
    assert!(!descr.is_finish(), "should exit via guard failure");

    let sum = backend.get_float_value(&frame, 0);
    let index = backend.get_int_value(&frame, 1);
    assert!(
        (sum - 5.0).abs() < 1e-10,
        "expected carried float sum to be 5.0, got {}",
        sum
    );
    assert_eq!(index, 5, "expected guard failure at i=5");
}

#[test]
fn test_gc_typeinfo_guards_use_dynasm_emit() {
    let mut gc = majit_gc::collector::MiniMarkGC::new();
    let root_tid = gc.register_type(majit_gc::TypeInfo::object(16));
    let child_tid = gc.register_type(majit_gc::TypeInfo::object_subclass(16, root_tid));
    let root_vtable: usize = 0x1234_5000;
    let child_vtable: usize = 0x1234_6000;
    majit_gc::GcAllocator::register_vtable_for_type(&mut gc, root_vtable, root_tid);
    majit_gc::GcAllocator::register_vtable_for_type(&mut gc, child_vtable, child_tid);
    let child_obj = gc.alloc_with_type(child_tid, 16);

    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    backend.set_gc_allocator(Box::new(gc));
    let mut token = JitCellToken::new(41);

    let const_child_tid = OpRef::const_int(child_tid as i64);
    let const_root_vtable = OpRef::const_int(root_vtable as i64);

    let inputargs = vec![InputArg::from_type(Type::Ref, 0)];
    let i0 = inputargs[0].opref();

    let guard_gc_type = Op::new(OpCode::GuardGcType, &[rb(i0), rb(const_child_tid)]);
    guard_gc_type.pos.set(OpRef::void_op(1));
    guard_gc_type.set_fail_arg_types(vec![]);
    guard_gc_type.setfailargs(vec![].into());

    let guard_is_object = Op::new(OpCode::GuardIsObject, &[rb(i0)]);
    guard_is_object.pos.set(OpRef::void_op(2));
    guard_is_object.set_fail_arg_types(vec![]);
    guard_is_object.setfailargs(vec![].into());

    let guard_subclass = Op::new(OpCode::GuardSubclass, &[rb(i0), rb(const_root_vtable)]);
    guard_subclass.pos.set(OpRef::void_op(3));
    guard_subclass.set_fail_arg_types(vec![]);
    guard_subclass.setfailargs(vec![].into());

    let finish_op = Op::new(OpCode::Finish, &[]);
    finish_op.pos.set(OpRef::void_op(4));
    finish_op.set_fail_arg_types(vec![]);
    finish_op.setfailargs(vec![].into());

    let ops = vec![guard_gc_type, guard_is_object, guard_subclass, finish_op];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();
    let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[Value::Ref(child_obj)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish(), "all GC type-info guards should pass");
}

#[test]
fn test_gc_typeinfo_guards_side_exit_on_mismatch() {
    {
        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let root_tid = gc.register_type(majit_gc::TypeInfo::object(16));
        let child_tid = gc.register_type(majit_gc::TypeInfo::object_subclass(16, root_tid));
        let child_vtable: usize = 0x1235_6000;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, child_vtable, child_tid);
        let root_obj = gc.alloc_with_type(root_tid, 16);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        backend.set_gc_allocator(Box::new(gc));
        let mut token = JitCellToken::new(45);

        let const_child_tid = OpRef::const_int(child_tid as i64);

        let inputargs = vec![InputArg::from_type(Type::Ref, 0)];
        let i0 = inputargs[0].opref();
        let guard_gc_type = Op::new(OpCode::GuardGcType, &[rb(i0), rb(const_child_tid)]);
        guard_gc_type.pos.set(OpRef::void_op(1));
        guard_gc_type.set_fail_arg_types(vec![Type::Ref]);
        guard_gc_type.setfailargs(vec![rb(i0).to_boxref()].into());
        let finish_op = Op::new(OpCode::Finish, &[]);
        finish_op.pos.set(OpRef::void_op(2));
        finish_op.set_fail_arg_types(vec![]);
        finish_op.setfailargs(vec![].into());

        let ops_rc: Vec<Rc<Op>> = vec![guard_gc_type, finish_op]
            .into_iter()
            .map(Rc::new)
            .collect();
        let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
        assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
        let frame = backend.execute_token(&token, &[Value::Ref(root_obj)]);
        assert!(
            !backend.get_latest_descr(&frame).is_finish(),
            "GUARD_GC_TYPE should side-exit on typeid mismatch"
        );
        assert_eq!(backend.get_ref_value(&frame, 0), root_obj);
    }

    {
        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let raw_tid = gc.register_type(majit_gc::TypeInfo::simple(16));
        let raw_obj = gc.alloc_with_type(raw_tid, 16);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        backend.set_gc_allocator(Box::new(gc));
        let mut token = JitCellToken::new(46);

        let inputargs = vec![InputArg::from_type(Type::Ref, 0)];
        let i0 = inputargs[0].opref();
        let guard_is_object = Op::new(OpCode::GuardIsObject, &[rb(i0)]);
        guard_is_object.pos.set(OpRef::void_op(1));
        guard_is_object.set_fail_arg_types(vec![Type::Ref]);
        guard_is_object.setfailargs(vec![rb(i0).to_boxref()].into());
        let finish_op = Op::new(OpCode::Finish, &[]);
        finish_op.pos.set(OpRef::void_op(2));
        finish_op.set_fail_arg_types(vec![]);
        finish_op.setfailargs(vec![].into());

        let ops_rc: Vec<Rc<Op>> = vec![guard_is_object, finish_op]
            .into_iter()
            .map(Rc::new)
            .collect();
        let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
        assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
        let frame = backend.execute_token(&token, &[Value::Ref(raw_obj)]);
        assert!(
            !backend.get_latest_descr(&frame).is_finish(),
            "GUARD_IS_OBJECT should side-exit for non-rclass object types"
        );
        assert_eq!(backend.get_ref_value(&frame, 0), raw_obj);
    }

    {
        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let root_a_tid = gc.register_type(majit_gc::TypeInfo::object(16));
        let root_b_tid = gc.register_type(majit_gc::TypeInfo::object(16));
        let root_a_vtable: usize = 0x1236_5000;
        let root_b_vtable: usize = 0x1236_7000;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, root_a_vtable, root_a_tid);
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, root_b_vtable, root_b_tid);
        let root_b_obj = gc.alloc_with_type(root_b_tid, 16);

        let mut backend = DynasmBackend::new();
        backend.attach_default_test_descrs();
        backend.set_gc_allocator(Box::new(gc));
        let mut token = JitCellToken::new(47);

        let const_root_a_vtable = OpRef::const_int(root_a_vtable as i64);

        let inputargs = vec![InputArg::from_type(Type::Ref, 0)];
        let i0 = inputargs[0].opref();
        let guard_subclass = Op::new(OpCode::GuardSubclass, &[rb(i0), rb(const_root_a_vtable)]);
        guard_subclass.pos.set(OpRef::void_op(1));
        guard_subclass.set_fail_arg_types(vec![Type::Ref]);
        guard_subclass.setfailargs(vec![rb(i0).to_boxref()].into());
        let finish_op = Op::new(OpCode::Finish, &[]);
        finish_op.pos.set(OpRef::void_op(2));
        finish_op.set_fail_arg_types(vec![]);
        finish_op.setfailargs(vec![].into());

        let ops_rc: Vec<Rc<Op>> = vec![guard_subclass, finish_op]
            .into_iter()
            .map(Rc::new)
            .collect();
        let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
        assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
        let frame = backend.execute_token(&token, &[Value::Ref(root_b_obj)]);
        assert!(
            !backend.get_latest_descr(&frame).is_finish(),
            "GUARD_SUBCLASS should side-exit outside the expected subclass range"
        );
        assert_eq!(backend.get_ref_value(&frame, 0), root_b_obj);
    }
}

#[test]
fn test_exception_guards_use_dynasm_emit() {
    let _guard = EXCEPTION_TEST_LOCK.lock().unwrap();
    majit_backend_dynasm::jit_exc_clear();

    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let mut token = JitCellToken::new(42);

    let expected_class = 0x5151_0000_i64;
    let const_expected_class = OpRef::const_int(expected_class);

    let inputargs = vec![];

    let guard_exception = Op::new(OpCode::GuardException, &[rb(const_expected_class)]);
    guard_exception.pos.set(OpRef::ref_op(0));
    guard_exception.set_fail_arg_types(vec![]);
    guard_exception.setfailargs(vec![].into());

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::ref_op(0))]);
    finish_op.pos.set(OpRef::void_op(1));
    finish_op.set_fail_arg_types(vec![Type::Ref]);
    finish_op.setfailargs(vec![rb(OpRef::ref_op(0)).to_boxref()].into());

    let ops = vec![guard_exception, finish_op];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();
    let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let mut exc_obj = vec![expected_class as usize, 0usize];
    let exc_ref = GcRef(exc_obj.as_mut_ptr() as usize);
    majit_backend_dynasm::jit_exc_raise(exc_ref.0 as i64);

    let frame = backend.execute_token(&token, &[]);
    let descr = backend.get_latest_descr(&frame);
    assert!(descr.is_finish(), "matching GUARD_EXCEPTION should pass");
    assert_eq!(backend.get_ref_value(&frame, 0), exc_ref);
    assert!(!majit_backend_dynasm::jit_exc_is_pending());

    majit_backend_dynasm::jit_exc_clear();
}

#[test]
fn test_guard_no_exception_and_always_fails_emit_side_exits() {
    let _guard = EXCEPTION_TEST_LOCK.lock().unwrap();
    majit_backend_dynasm::jit_exc_clear();

    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let mut token = JitCellToken::new(43);

    let inputargs = vec![];
    let guard_no_exception = Op::new(OpCode::GuardNoException, &[]);
    guard_no_exception.pos.set(OpRef::void_op(0));
    guard_no_exception.set_fail_arg_types(vec![]);
    guard_no_exception.setfailargs(vec![].into());

    let finish_op = Op::new(OpCode::Finish, &[]);
    finish_op.pos.set(OpRef::void_op(1));
    finish_op.set_fail_arg_types(vec![]);
    finish_op.setfailargs(vec![].into());

    let ops = vec![guard_no_exception, finish_op];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();
    let result = backend.compile_loop(&inputargs, &ops_rc, &mut token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    let frame = backend.execute_token(&token, &[]);
    assert!(
        backend.get_latest_descr(&frame).is_finish(),
        "GUARD_NO_EXCEPTION should pass with clear exception state"
    );

    let mut exc_obj = vec![0x6161_0000_usize, 0usize];
    majit_backend_dynasm::jit_exc_raise(exc_obj.as_mut_ptr() as i64);
    let frame = backend.execute_token(&token, &[]);
    assert!(
        !backend.get_latest_descr(&frame).is_finish(),
        "GUARD_NO_EXCEPTION should side-exit with pending exception"
    );
    majit_backend_dynasm::jit_exc_clear();

    let mut always_backend = DynasmBackend::new();
    always_backend.attach_default_test_descrs();
    let mut always_token = JitCellToken::new(44);
    let guard_always_fails = Op::new(OpCode::GuardAlwaysFails, &[]);
    guard_always_fails.pos.set(OpRef::void_op(0));
    guard_always_fails.set_fail_arg_types(vec![]);
    guard_always_fails.setfailargs(vec![].into());
    let finish_op = Op::new(OpCode::Finish, &[]);
    finish_op.pos.set(OpRef::void_op(1));
    finish_op.set_fail_arg_types(vec![]);
    finish_op.setfailargs(vec![].into());
    let ops = vec![guard_always_fails, finish_op];
    let ops_rc: Vec<Rc<Op>> = ops.into_iter().map(Rc::new).collect();
    let result = always_backend.compile_loop(&[], &ops_rc, &mut always_token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
    let frame = always_backend.execute_token(&always_token, &[]);
    assert!(
        !always_backend.get_latest_descr(&frame).is_finish(),
        "GUARD_ALWAYS_FAILS should side-exit unconditionally"
    );
}
