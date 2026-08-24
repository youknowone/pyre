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

// 8. Header-first object model
//
//   1. `(*w).ob_type` off a `*mut ObjectHeader` — a `FieldRead` preceded by
//      a `__cast_instance_intrinsic/<Root>` narrow, not a classdef-less read.
//   2. `lltype::malloc_typed(Leaf { ob_header: .., payload })` — one
//      by-value argument, header written before the call, so
//      `fuse_boxing_alloc` mints `NewWithVtable` with a real vtable.
//   3. `if ta == &CLS { concrete(..) }` — the arm body a `FunctionPath`
//      call, i.e. inlinable, rather than an indirect one.
//   4. an `_immutable_fields_<Struct>` marker const, so the payload read
//      can fold to a pure getfield.
//
// `TypeOnlyHeader` matches RPython's root `OBJECT`, whose only data field is
// `typeptr`. `ObjectHeader` additionally represents an object model with a
// per-instance class word. Both allocation shapes must fuse.

#[repr(C)]
pub struct ClassObject {
    pub name: &'static str,
    pub kind: u8,
}

#[repr(C)]
pub struct ObjectHeader {
    pub ob_type: *const ClassObject,
    pub w_class: *const ClassObject,
}

/// The one-word header: `ob_type` and nothing else.
///
/// `fuse_boxing_alloc`'s substitution asks whether the per-instance class
/// word agrees with `ob_type`; where the header declares no such word the
/// question has no subject, and `model.rs`'s `header_declares_no_class_word`
/// arm admits the cluster on the layout alone. That arm is unreachable from
/// any fixture whose header declares the field, so it gets its own.
#[repr(C)]
pub struct TypeOnlyHeader {
    pub ob_type: *const ClassObject,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_IntObject {
    pub ob_header: ObjectHeader,
    pub intval: i64,
}

#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_TypeOnlyIntObject {
    pub ob_header: TypeOnlyHeader,
    pub intval: i64,
}

pub static INT_CLASS: ClassObject = ClassObject {
    name: "int",
    kind: 1,
};
pub static DOUBLE_CLASS: ClassObject = ClassObject {
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

/// Minimal stand-in for the class-instantiation lookup whose result the
/// class word is stored from.
///
/// `model.rs`'s `get_instantiate_arg_addr` recognizes this helper by its
/// single-argument runtime contract.
pub mod runtime_object {
    pub mod object_model {
        use crate::ClassObject;

        #[inline(never)]
        pub fn get_instantiate(tp: &ClassObject) -> *const ClassObject {
            tp
        }
    }
}

/// Premise 1: the header read.
#[inline(never)]
pub fn w_object_type(w: *mut ObjectHeader) -> *const ClassObject {
    unsafe { (*w).ob_type }
}

/// Premise 2: the boxing cluster — header store, payload store, then one
/// by-value `malloc_typed`.
#[inline(never)]
pub fn w_new_int(x: i64) -> *mut W_IntObject {
    lltype::malloc_typed(W_IntObject {
        ob_header: ObjectHeader {
            ob_type: &INT_CLASS,
            w_class: runtime_object::object_model::get_instantiate(&INT_CLASS),
        },
        intval: x,
    })
}

/// Premise 2, second header shape: the same cluster over a header that
/// declares no class word, so the fuse has to admit it on the layout alone.
/// The `get_instantiate` call has no subject here and is absent.
#[inline(never)]
pub fn w_new_type_only_int(x: i64) -> *mut W_TypeOnlyIntObject {
    lltype::malloc_typed(W_TypeOnlyIntObject {
        ob_header: TypeOnlyHeader {
            ob_type: &INT_CLASS,
        },
        intval: x,
    })
}

#[inline(never)]
fn w_int_add(a: *mut W_IntObject, b: *mut W_IntObject) -> i64 {
    unsafe { (*a).intval + (*b).intval }
}

/// Premise 3: the narrowing chain. `descroperation.py` `binop_impl`
/// (`type(w_obj1) is type(w_obj2)`, then the per-class shortcut)
/// transliterated — the shape that lowers each arm to a direct call.
#[inline(never)]
pub fn w_number_add(a: *mut ObjectHeader, b: *mut ObjectHeader) -> i64 {
    let ta = unsafe { (*a).ob_type };
    let tb = unsafe { (*b).ob_type };
    if ta == tb && ta == (&INT_CLASS as *const ClassObject) {
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

// 9. An array whose element is a by-value aggregate.
//
// `v[i]` on a `Vec<T>` with a scalar subscript resolves to
// `<Vec<T> as Index<usize>>::index`, which `front::mir`'s
// `is_vec_index_call` (`vec_index_regular_leaf`) intercepts: the call site
// lowers eagerly to an `ArrayRead` and records an `IndexElemAlias` for the
// paired write and for the projections off the element, leaving no residual
// call. That arm gates on the *index* type and never on the *element* type —
// `item_ty` is whatever `tyref_deref_value_type(call.dest.ty)` answers — so
// the element bank it produces for a multi-word by-value ADT is decided by
// `tyref_to_value_type`'s fallback rather than by a deliberate arm.
//
// `SlotValue` carries an integer variant, a raw-pointer variant and a
// two-field variant. None of the three leaves a niche free, so the enum is a
// genuine tag-plus-payload aggregate several words wide and not a wrapper the
// front end can collapse to its inner bank (`tyref_transparent_inner_value_type`,
// `tyref_is_fieldless_enum_free`).
//
// That question is now answered, in the negative: the fallback bank is
// `ValueType::Ref(None)`, the `Vec` leg ships no `array_type_id`, and
// `arraydescrof_concrete` therefore hands the read a one-word `item_size`.
// The index spelling is UNSOUND over this element type — see
// `an_aggregate_element_array_read_is_given_a_one_word_item_size` in
// `majit-translate/tests/test_mir_frontend.rs` for the full chain. These four
// functions stay as they are because they are what witnesses it.
pub enum SlotValue {
    Int(i64),
    Object(*const ObjectHeader),
    Pair { lhs: i64, rhs: i64 },
}

/// The treatment: read one element by scalar index and match it. The
/// discriminant read and the per-variant payload reads all land on the
/// `ArrayRead` result, so this is the shape that says whether the alias
/// projections carry an aggregate element. They do not — this is the spelling
/// whose descr strides by one word, not a spelling to copy.
#[inline(never)]
pub fn aggregate_slot_index(v: &Vec<SlotValue>, i: usize) -> i64 {
    match &v[i] {
        SlotValue::Int(x) => *x,
        SlotValue::Object(h) => unsafe { (*(**h).ob_type).kind as i64 },
        SlotValue::Pair { lhs, rhs } => *lhs + *rhs,
    }
}

/// The control: the same body over `<[T]>::get`, reached by deref coercion
/// from the `Vec`. Its `Option<&SlotValue>` destination is what
/// `recognize_slice_get_site` accepts or declines, so the call either becomes
/// `front::slice_get`'s bounds-checked diamond or stays residual — either way
/// it is a different lowering from the index arm. If it is not, the pair
/// discriminates nothing and neither function grades the index arm. Given the
/// index arm's descr, the residual outcome is also the *safe* one: real Rust
/// keeps computing the element address at the real stride.
#[inline(never)]
pub fn aggregate_slot_get(v: &Vec<SlotValue>, i: usize) -> i64 {
    match v.get(i) {
        Some(SlotValue::Int(x)) => *x,
        Some(SlotValue::Object(h)) => unsafe { (*(**h).ob_type).kind as i64 },
        Some(SlotValue::Pair { lhs, rhs }) => *lhs + *rhs,
        None => 0,
    }
}

/// The same two spellings over an element bank the index arm is already
/// known to serve, so a difference between the two pairs is attributable to
/// the element kind and not to the fixture.
#[inline(never)]
pub fn scalar_slot_index(v: &Vec<i64>, i: usize) -> i64 {
    v[i]
}

#[inline(never)]
pub fn scalar_slot_get(v: &Vec<i64>, i: usize) -> i64 {
    match v.get(i) {
        Some(x) => *x,
        None => 0,
    }
}
