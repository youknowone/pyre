//! Charon fixture corpus: representative shapes from issue #97.
//!
//! 1. `straight_line_add` — straight-line interpreter-shaped function.
//! 2. `branch_loop_sum`   — branch + loop, like opcode dispatch fragments.
//! 3. `strategy_dispatch` — enum-as-strategy (dict-strategy stand-in).
//! 4. `desugar_mix`       — `?`, `match`, and iterator desugaring together.

#![allow(dead_code)]

pub type PyResult<T> = Result<T, &'static str>;

// 1. Straight-line
#[inline(never)]
pub fn straight_line_add(a: i64, b: i64, c: i64) -> i64 {
    let s = a + b;
    let t = s * 2;
    t + c
}

// 2. Branch and loop
#[inline(never)]
pub fn branch_loop_sum(slice: &[i64], threshold: i64) -> i64 {
    let mut acc: i64 = 0;
    for &v in slice {
        if v > threshold {
            acc += v;
        } else {
            acc -= v;
        }
    }
    acc
}

// 2b. Iterator element kinds.  `next()`'s payload carries one reference for
// a slice iterator (`core::slice::iter::Iter` yields `Option<&T>`) and none
// for a by-value one (`core::array::iter::IntoIter` yields `Option<T>`), so
// the two spell the same `Option<&i64>` payload for different reasons: here
// the element is `&i64` both times, and only the first has a reference the
// iterator added.  A frontend that peels unconditionally, or never, types one
// of the two into the wrong register bank.
#[inline(never)]
pub fn slice_of_refs_sum(slice: &[&i64]) -> i64 {
    let mut acc: i64 = 0;
    for r in slice {
        acc += **r;
    }
    acc
}

#[inline(never)]
pub fn array_of_refs_sum(refs: [&i64; 3]) -> i64 {
    let mut acc: i64 = 0;
    for r in refs {
        acc += *r;
    }
    acc
}

// 3. Strategy dispatch (dict-strategy stand-in)
pub enum Strategy {
    Empty,
    IntKeyed { len: usize },
    StrKeyed { len: usize, capacity: usize },
}

#[inline(never)]
pub fn strategy_len(s: &Strategy) -> usize {
    match s {
        Strategy::Empty => 0,
        Strategy::IntKeyed { len } => *len,
        Strategy::StrKeyed { len, capacity: _ } => *len,
    }
}

// 4. Desugaring mix: `?`, `match`, and iteration
pub enum Token {
    Add(i64),
    Sub(i64),
    Halt,
}

fn parse_one(raw: i64) -> PyResult<Token> {
    match raw {
        i64::MIN => Ok(Token::Halt),
        0 => Err("halt-zero forbidden"),
        v if v > 0 => Ok(Token::Add(v)),
        v => Ok(Token::Sub(-v)),
    }
}

#[inline(never)]
pub fn desugar_mix(input: &[i64]) -> PyResult<i64> {
    let mut acc: i64 = 0;
    for &raw in input.iter() {
        let tok = parse_one(raw)?;
        match tok {
            Token::Add(v) => acc += v,
            Token::Sub(v) => acc -= v,
            Token::Halt => break,
        }
    }
    Ok(acc)
}

// 5. Tuple round-trip: construct a tuple and read both fields
//
// Exercises `Rvalue::Aggregate` for a *non-Adt* (tuple) value paired
// with `Field` projection reads of that same local. The lowering must
// emit a `__pos_<idx>` `FieldRead` symmetric to the construction-side
// `FieldWrite` chain rather than collapsing every `.N` to the base.

#[inline(never)]
pub fn tuple_roundtrip(a: i64, b: i64) -> i64 {
    let pair = (a + b, a - b);
    pair.0 * pair.1
}

// 6. Closures
// `bool_then_closure` is the exact `core::bool::<Impl>::then` census shape:
// an opaque combinator taking a `FnOnce` closure that captures a value from
// the enclosing scope. Charon extracts the closure's `call_once` body as a
// transparent inherent method of the closure type.

#[inline(never)]
pub fn bool_then_closure(c: bool, x: i64) -> Option<i64> {
    c.then(|| x + 1)
}

// `then_some` is the eager sibling of `then`: it takes an already-evaluated
// value rather than a closure, so the diamond's `then` arm wraps it in `Some`
// directly (no `call_once`). Same Opaque-core-combinator residual shape.
#[inline(never)]
pub fn bool_then_some(c: bool, x: i64) -> Option<i64> {
    c.then_some(x + 1)
}

// 7. Option question mark
// Exercises `Try::branch` on `Option`: `Some(v)` continues with `v`, while
// `None` returns `None` normally from the enclosing Option-returning function.

#[inline(never)]
fn option_source(keep: bool, value: i64) -> Option<i64> {
    if keep { Some(value) } else { None }
}

#[inline(never)]
pub fn option_question_mark(keep: bool, value: i64, addend: i64) -> Option<i64> {
    let v = option_source(keep, value)?;
    Some(v + addend)
}

// A host-registered callback table.

/// The callback a host installs at run time. A bare `fn` pointer, so the set
/// of addresses that can reach a call through it is not recoverable from this
/// artifact — the shape used by host-settable callback hooks.
pub type HostCallback = fn(i64) -> i64;

pub struct HostRegistry {
    pub slot: HostCallback,
    pub maybe_slot: Option<HostCallback>,
}

/// Call through the registered callback. `front::mir` lowers this to
/// `OpKind::IndirectCall { graphs: None }` — `indirect_call` with an
/// unknown PBC family, which `guess_call_kind` answers `residual` for
/// (`call.py:105`/`137`, `jtransform.py:410-412`). The `__dyn_call`
/// placeholder it used to reach is an unregistered synthetic path with no
/// continuation.
#[inline(never)]
pub fn host_registry_dispatch(reg: &HostRegistry, x: i64) -> i64 {
    (reg.slot)(x)
}

/// The one-hop `Option<fn-ptr>` spelling of the same shape.
#[inline(never)]
pub fn host_registry_dispatch_optional(reg: &HostRegistry, x: i64) -> i64 {
    match reg.maybe_slot {
        Some(f) => f(x),
        None => 0,
    }
}
