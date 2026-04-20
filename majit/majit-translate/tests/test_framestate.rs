//! Line-by-line port of `rpython/flowspace/test/test_framestate.py`
//! (102 LOC, 10 tests).
//!
//! Deviation from upstream, per CLAUDE.md parity rule #1:
//!
//! * Upstream compiles a live Python function (`func_simple`) and
//!   builds a `FlowContext` over its `__code__`, then pokes
//!   `ctx.setstate` / `ctx.getstate` to shuttle `FrameState` values in
//!   and out. The Rust port avoids the rustpython-compiler round-trip
//!   (that integration lives on F3.9) by constructing a `FrameState`
//!   directly with the same shape the upstream helper would have
//!   produced: `[Variable("x"), Constant(None)]` matches
//!   `func_simple`'s `co_nlocals == 2` / `formalargcount == 1` state
//!   after upstream's `locals_w[-1] = Constant(None)` hack.
//!
//! Every test is named identically to its upstream counterpart.

use majit_translate::flowspace::framestate::FrameState;
use majit_translate::flowspace::model::{ConstValue, Constant, Hlvalue, Variable};

/// Upstream `get_context(func_simple)` returns a FlowContext whose
/// locals_w matches the `func_simple`-like shape. The Rust helper
/// constructs the equivalent FrameState directly.
///
/// Shape: `[Variable("x"), Constant(None)]`. The `Variable("x")`
/// represents the formal arg; the `Constant(None)` represents upstream's
/// `ctx.locals_w[-1] = Constant(None)` hack.
fn make_initial_state() -> (FrameState, Hlvalue) {
    let v_x = Hlvalue::Variable(Variable::named("x"));
    let state = FrameState::new(
        vec![
            Some(v_x.clone()),
            Some(Hlvalue::Constant(Constant::new(ConstValue::None))),
        ],
        Vec::new(),
        None,
        Vec::new(),
        0,
    );
    (state, v_x)
}

#[test]
fn test_eq_framestate() {
    let (fs1, _) = make_initial_state();
    let (fs2, _) = make_initial_state();
    assert!(fs1.matches(&fs2));
}

#[test]
fn test_neq_hacked_framestate() {
    // upstream: `ctx.locals_w[-1] = Variable()` generalises the last
    // slot to an unknown variable → mismatch against the Constant-slot
    // baseline.
    let (fs1, v_x) = make_initial_state();
    let fs2 = FrameState::new(
        vec![Some(v_x), Some(Hlvalue::Variable(Variable::new()))],
        Vec::new(),
        None,
        Vec::new(),
        0,
    );
    assert!(!fs1.matches(&fs2));
}

#[test]
fn test_union_on_equal_framestates() {
    let (fs1, _) = make_initial_state();
    let (fs2, _) = make_initial_state();
    let fs3 = fs1.union(&fs2).expect("equal states must union");
    assert!(fs3.matches(&fs1));
}

#[test]
fn test_union_on_hacked_framestates() {
    let (fs1, v_x) = make_initial_state();
    let fs2 = FrameState::new(
        vec![Some(v_x), Some(Hlvalue::Variable(Variable::new()))],
        Vec::new(),
        None,
        Vec::new(),
        0,
    );
    // fs2 is more general (Variable in slot 1 beats Constant).
    let u12 = fs1.union(&fs2).expect("variable/constant must union");
    assert!(u12.matches(&fs2));
    let u21 = fs2.union(&fs1).expect("symmetric union must succeed");
    assert!(u21.matches(&fs2));
}

#[test]
fn test_restore_frame() {
    // upstream: `ctx.setstate(fs1)` restores locals_w to fs1's view.
    // In the Rust port the equivalent is cloning fs1's locals_w onto a
    // fresh mutable buffer and reading the resulting state back.
    let (fs1, _) = make_initial_state();
    let fs1_copy = FrameState::new(
        fs1.locals_w.clone(),
        fs1.stack.clone(),
        fs1.last_exception.clone(),
        fs1.blocklist.clone(),
        fs1.next_offset,
    );
    assert!(fs1.matches(&fs1_copy));
}

#[test]
fn test_copy() {
    let (fs1, _) = make_initial_state();
    let fs2 = fs1.copy();
    assert!(fs1.matches(&fs2));
}

#[test]
fn test_getvariables() {
    // upstream: `len(fs1.getvariables()) == 1` — only Variable slots
    // count. Constant(None) in slot 1 is not a variable.
    let (fs1, _) = make_initial_state();
    let vars = fs1.getvariables();
    assert_eq!(vars.len(), 1);
}

#[test]
fn test_getoutputargs() {
    // upstream: fs1 has `[Variable(x), Constant(None)]`,
    // fs2 has `[Variable(x), Variable(fresh)]`. The outputargs list
    // reads fs1's source at each fs2 Variable slot and preserves
    // Constant slots.
    let (fs1, v_x) = make_initial_state();
    let fs2 = FrameState::new(
        vec![Some(v_x.clone()), Some(Hlvalue::Variable(Variable::new()))],
        Vec::new(),
        None,
        Vec::new(),
        0,
    );
    let outputargs = fs1.getoutputargs(&fs2);
    // upstream asserts `outputargs == [ctx.locals_w[0],
    // Constant(None)]`. Our mergeable is locals_w(2) + exc_sentinels(2)
    // = 4 slots; outputargs is indexed by fs2's mergeable.
    // Slot 0 is a Variable target → use fs1's source (Variable(x)).
    // Slot 1 is a Variable target → use fs1's source (Constant(None)).
    // Slots 2 and 3 (exc sentinels) match constants → preserved.
    assert_eq!(outputargs[0], Some(v_x));
    assert_eq!(
        outputargs[1],
        Some(Hlvalue::Constant(Constant::new(ConstValue::None)))
    );
}

#[test]
fn test_union_different_constants() {
    // upstream: slot 1 differs across fs1 / fs2 as Constant(None) vs
    // Constant(42). Union should generalise to a fresh Variable.
    let (fs1, v_x) = make_initial_state();
    let fs2 = FrameState::new(
        vec![
            Some(v_x),
            Some(Hlvalue::Constant(Constant::new(ConstValue::Int(42)))),
        ],
        Vec::new(),
        None,
        Vec::new(),
        0,
    );
    let fs3 = fs1.union(&fs2).expect("different constants must union");
    assert!(matches!(&fs3.locals_w[1], Some(Hlvalue::Variable(_))));
}

#[test]
fn test_union_spectag() {
    // upstream: `Constant(SpecTag())` identities are unique across
    // calls, so `fs1.union(fs2) is None` (UnionError).
    //
    // Rust port: `ConstValue::SpecTag(u64)` stamps a process-unique id
    // per call; two `ConstValue::SpecTag(fresh)` constants differ.
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(1);

    fn fresh_spectag() -> Hlvalue {
        Hlvalue::Constant(Constant::new(ConstValue::SpecTag(
            SEQ.fetch_add(1, Ordering::Relaxed),
        )))
    }

    let (_, v_x) = make_initial_state();
    let fs1 = FrameState::new(
        vec![Some(v_x.clone()), Some(fresh_spectag())],
        Vec::new(),
        None,
        Vec::new(),
        0,
    );
    let fs2 = FrameState::new(
        vec![Some(v_x), Some(fresh_spectag())],
        Vec::new(),
        None,
        Vec::new(),
        0,
    );
    assert!(fs1.union(&fs2).is_none());
}
