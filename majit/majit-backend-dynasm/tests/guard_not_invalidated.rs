//! `GUARD_NOT_INVALIDATED` is the one guard that tests nothing at run time.
//!
//! `x86/assembler.py genop_guard_guard_not_invalidated` records the guard's
//! position and emits no test, and `x86/runner.py invalidate_loop` /
//! `aarch64/runner.py invalidate_loop` later write a branch to the guard's recovery stub over
//! the recorded position.  So the contract these tests pin is behavioural, not
//! structural: the same entry point runs to completion before
//! `invalidate_loop`, and takes the guard after it, with nothing in the trace
//! changed in between.
//!
//! `aarch64/runner.py invalidate_loop`'s docstring is the second half of it: "afterwards, if
//! one such guard fails often enough, it has a bridge attached to it; it is
//! possible then to re-call invalidate_loop() on the same looptoken, which must
//! invalidate all newer GUARD_NOT_INVALIDATED, but not the old one that already
//! has a bridge attached to it".  The list is emptied by the walk, so a second
//! call has nothing to write.

use std::rc::Rc;

use majit_backend::{Backend, JitCellToken, make_resume_guard_descr_typed};
use majit_ir::{InputArg, Op, OpCode, OpRef, Type, Value};

use majit_backend_dynasm::runner::DynasmBackend;
use majit_ir::forwarding::bound_operand_from_opref as rb;

/// `guard_not_invalidated() [i0]` / `i1 = int_add(i0, 1)` / `finish(i1)`.
///
/// The guard carries `i0` as its one fail argument and the finish carries `i1`,
/// so the two exits are distinguishable by value alone: 42 means the guard was
/// taken, 43 means it was not.
fn compile_guarded_add(backend: &mut DynasmBackend, token: &JitCellToken) -> majit_ir::DescrRef {
    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let guard_descr = make_resume_guard_descr_typed(vec![Type::Int]);
    let guard_op = Op::new(OpCode::GuardNotInvalidated, &[]);
    guard_op.pos.set(OpRef::void_op(0));
    guard_op.set_fail_arg_types(vec![Type::Int]);
    guard_op.setfailargs(vec![rb(i0)].into());
    guard_op.setdescr(guard_descr.clone());

    let add_op = Op::new(OpCode::IntAdd, &[rb(i0), rb(OpRef::const_int(1))]);
    add_op.pos.set(OpRef::int_op(1));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish_op.pos.set(OpRef::void_op(2));
    finish_op.set_fail_arg_types(vec![Type::Int]);
    finish_op.setfailargs(vec![rb(OpRef::int_op(1))].into());

    let ops_rc: Vec<Rc<Op>> = vec![Rc::new(guard_op), Rc::new(add_op), Rc::new(finish_op)];
    let result = backend.compile_loop(&inputargs, &ops_rc, token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());
    guard_descr
}

#[test]
fn the_guard_is_inert_until_invalidate_loop_writes_the_branch() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(1);
    compile_guarded_add(&mut backend, &token);

    // Before: the guard site holds the placeholder the emitter left, so control
    // falls through it into the trace body.
    let frame = backend.execute_token(&token, &[Value::Int(42)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(
        descr.is_finish(),
        "an un-invalidated GUARD_NOT_INVALIDATED must not be reachable as an exit"
    );
    assert_eq!(backend.get_int_value(&frame, 0), 43);

    // `quasiimmut.py QuasiImmut.invalidate`: `looptoken.invalidated = True;
    // cpu.invalidate_loop(looptoken)`.
    backend.invalidate_loop(&token);

    // After: the very same entry point takes the guard.  Nothing about the
    // trace changed — only the bytes at the recorded position.
    let frame = backend.execute_token(&token, &[Value::Int(42)]);
    let descr = backend.get_latest_descr(&frame);
    assert!(
        !descr.is_finish(),
        "an invalidated GUARD_NOT_INVALIDATED must exit through its recovery stub"
    );
    assert_eq!(
        backend.get_int_value(&frame, 0),
        42,
        "the deadframe must hold the guard's fail argument, not the finish's"
    );
}

/// `runner.py invalidate_loop`'s trailing
/// `looptoken.compiled_loop_token.invalidate_positions = []` —
/// the walk consumes the list, so calling it again writes nothing.  The already
/// written branch stays written.
#[test]
fn a_second_invalidation_has_nothing_left_to_write() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(2);
    compile_guarded_add(&mut backend, &token);

    backend.invalidate_loop(&token);
    backend.invalidate_loop(&token);

    let frame = backend.execute_token(&token, &[Value::Int(7)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(backend.get_int_value(&frame, 0), 7);
}

/// A trace with two of them.  Both positions are recorded and both are written,
/// and the first one reached is the one that exits — the guards are ordinary
/// trace positions, not a single per-trace switch.
#[test]
fn every_recorded_position_in_a_trace_is_written() {
    let mut backend = DynasmBackend::new();
    backend.attach_default_test_descrs();
    let token = JitCellToken::new(3);

    let inputargs = vec![InputArg::from_type(Type::Int, 0)];
    let i0 = inputargs[0].opref();

    let first = Op::new(OpCode::GuardNotInvalidated, &[]);
    first.pos.set(OpRef::void_op(0));
    first.set_fail_arg_types(vec![Type::Int]);
    first.setfailargs(vec![rb(i0)].into());
    first.setdescr(make_resume_guard_descr_typed(vec![Type::Int]));

    let add_op = Op::new(OpCode::IntAdd, &[rb(i0), rb(OpRef::const_int(1))]);
    add_op.pos.set(OpRef::int_op(1));

    let second = Op::new(OpCode::GuardNotInvalidated, &[]);
    second.pos.set(OpRef::void_op(2));
    second.set_fail_arg_types(vec![Type::Int]);
    second.setfailargs(vec![rb(OpRef::int_op(1))].into());
    second.setdescr(make_resume_guard_descr_typed(vec![Type::Int]));

    let finish_op = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
    finish_op.pos.set(OpRef::void_op(3));
    finish_op.set_fail_arg_types(vec![Type::Int]);
    finish_op.setfailargs(vec![rb(OpRef::int_op(1))].into());

    let ops_rc: Vec<Rc<Op>> = vec![
        Rc::new(first),
        Rc::new(add_op),
        Rc::new(second),
        Rc::new(finish_op),
    ];
    let result = backend.compile_loop(&inputargs, &ops_rc, &token);
    assert!(result.is_ok(), "compile_loop failed: {:?}", result.err());

    assert!(
        backend
            .get_latest_descr(&backend.execute_token(&token, &[Value::Int(42)]))
            .is_finish()
    );

    backend.invalidate_loop(&token);

    let frame = backend.execute_token(&token, &[Value::Int(42)]);
    assert!(!backend.get_latest_descr(&frame).is_finish());
    assert_eq!(
        backend.get_int_value(&frame, 0),
        42,
        "the first guard is the one reached, so its fail argument is the one saved"
    );
}
