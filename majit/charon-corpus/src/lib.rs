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
//
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
//
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

// 8. Header-first class universe (the `W_Root` shape)
//
// The four lowering premises a class-based, `Arc`-free value universe rests
// on. Each is a shape the frontend either recognises or silently degrades:
// a degraded one produces no error, just a worse graph, so each gets an
// assertion in `majit-translate/tests/test_mir_frontend.rs`.
//
//   1. `(*w).ob_type` off a `*mut CelObject` — a `FieldRead` preceded by a
//      `__pyre_cast_instance/<Root>` narrow, not a classdef-less read.
//   2. `lltype::malloc_typed(Leaf { ob_header: .., payload })` — one
//      by-value argument, header written before the call, so
//      `fuse_boxing_alloc` mints `NewWithVtable` with a real vtable.
//   3. `if ta == &CLS { concrete(..) }` — the arm body a `FunctionPath`
//      call, i.e. inlinable, rather than an indirect one.
//   4. an `_immutable_fields_<Struct>` marker const, so the payload read
//      can fold to a pure getfield.
//
// Port of `rclass.py:162-165` OBJECT = GcStruct('object', ('typeptr', ..),
// hints={'immutable':True,'shouldntbenull':True,'typeptr':True}), spelled
// as `pyre-object/src/pyobject.rs` PyObject.

#[repr(C)]
pub struct CelClass {
    pub name: &'static str,
    pub kind: u8,
}

#[repr(C)]
pub struct CelObject {
    pub ob_type: *const CelClass,
    pub w_class: *const CelClass,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_IntObject {
    pub ob_header: CelObject,
    pub intval: i64,
}

pub static CEL_INT_CLASS: CelClass = CelClass {
    name: "int",
    kind: 1,
};
pub static CEL_DOUBLE_CLASS: CelClass = CelClass {
    name: "double",
    kind: 2,
};

/// The allocation entry point. `is_malloc_typed` keys on the trailing path
/// segments `lltype::malloc_typed`, and `fuse_boxing_alloc` requires the
/// single by-value argument — an alloc-then-init spelling matches nothing
/// and degrades in silence.
pub mod lltype {
    #[inline(never)]
    pub fn malloc_typed<T>(value: T) -> *mut T {
        Box::into_raw(Box::new(value))
    }
}

/// Minimal stand-in for the production class-instantiation lookup. Keeping
/// the production path suffix makes the boxing fixture exercise the same
/// `get_instantiate(&T)` recognition without depending on `pyre-object`.
pub mod pyre_object {
    pub mod pyobject {
        use crate::CelClass;

        #[inline(never)]
        pub fn get_instantiate(tp: &CelClass) -> *const CelClass {
            tp
        }
    }
}

/// Premise 1: the header read.
#[inline(never)]
pub fn cel_w_type(w: *mut CelObject) -> *const CelClass {
    unsafe { (*w).ob_type }
}

/// Premise 2: the boxing cluster — header store, payload store, then one
/// by-value `malloc_typed`.
#[inline(never)]
pub fn cel_new_int(x: i64) -> *mut W_IntObject {
    lltype::malloc_typed(W_IntObject {
        ob_header: CelObject {
            ob_type: &CEL_INT_CLASS,
            w_class: pyre_object::pyobject::get_instantiate(&CEL_INT_CLASS),
        },
        intval: x,
    })
}

#[inline(never)]
fn w_int_add(a: *mut W_IntObject, b: *mut W_IntObject) -> i64 {
    unsafe { (*a).intval + (*b).intval }
}

/// Premise 3: the narrowing chain. `descroperation.py:706-712`
/// (`type(w_obj1) is type(w_obj2)`, then the per-class shortcut)
/// transliterated — the shape that lowers each arm to a direct call.
#[inline(never)]
pub fn cel_add(a: *mut CelObject, b: *mut CelObject) -> i64 {
    let ta = unsafe { (*a).ob_type };
    let tb = unsafe { (*b).ob_type };
    if ta == tb && ta == (&CEL_INT_CLASS as *const CelClass) {
        return w_int_add(a as *mut W_IntObject, b as *mut W_IntObject);
    }
    0
}

/// Premise 4: the immutability marker `harvest_immutable_fields_from_llbcs`
/// reads (`front/llbc_hints.rs:148` `_immutable_fields_` prefix). Written by
/// hand rather than through `#[jit_immutable_fields]` so the corpus keeps
/// its zero-dependency manifest.
#[allow(non_upper_case_globals)]
pub const _immutable_fields_W_IntObject: &str = "intval";

// A host-registered callback table.

/// The callback a host installs at run time. A bare `fn` pointer, so the set
/// of addresses that can reach a call through it is not recoverable from this
/// artifact — the shape `cel`'s overload table and pyre's settable hooks both
/// have.
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
