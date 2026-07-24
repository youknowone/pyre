//! W_LongObject -- arbitrary-precision integer backed by RPython `rbigint`.
//!
//! Used when i64 overflow is detected in `W_IntObject` arithmetic.
//! The JIT may specialize bigint operations by reading immutable `value`
//! payloads, calling pure raw-payload helpers, and boxing the resulting payload
//! with the same `W_LongObject` layout.

use crate::rbigint::{RBigInt as BigInt, RBigIntSign};

use crate::pyobject::*;

/// Arbitrary-precision integer object.
///
/// Layout: `[ob_type: *const PyType | value: *mut RBigInt]`
/// The `value` pointer references an immutable `rbigint` payload, usually
/// GC-managed and occasionally leaked via `malloc_raw` before GC init.
#[repr(C)]
pub struct W_LongObject {
    pub ob_header: PyObject,
    pub value: *mut BigInt,
}

// Safety: RBigInt is Send+Sync and W_LongObject only stores a raw pointer
// that is effectively owned.
unsafe impl Send for W_LongObject {}
unsafe impl Sync for W_LongObject {}

/// Field offset of `value` within `W_LongObject`, for potential JIT field access.
pub const LONG_VALUE_OFFSET: usize = std::mem::offset_of!(W_LongObject, value);

/// GC type id assigned to `W_LongObject` at JitDriver init time.
pub const W_LONG_GC_TYPE_ID: u32 = 35;

/// Fixed payload size (`framework.py:811`).
pub const W_LONG_OBJECT_SIZE: usize = std::mem::size_of::<W_LongObject>();

impl crate::lltype::GcType for W_LongObject {
    fn type_id() -> u32 {
        W_LONG_GC_TYPE_ID
    }
    const SIZE: usize = W_LONG_OBJECT_SIZE;
}

/// Payload size of a raw `rbigint` GC object.
pub const BIGINT_PAYLOAD_SIZE: usize = crate::rbigint::RBIGINT_PAYLOAD_SIZE;

/// Translated rbigint GC-reference result ABI.
///
/// Native LLBC extraction must retain the pointer result so the codewriter
/// models it as a GcRef. wasm32 direct residual calls use the JIT's uniform
/// i64 word family, so encode the same pointer in an i64 on that target.
#[cfg(not(target_arch = "wasm32"))]
pub type JitBigIntResult = *mut BigInt;
#[cfg(target_arch = "wasm32")]
pub type JitBigIntResult = i64;

#[cfg(not(target_arch = "wasm32"))]
#[inline]
pub fn encode_jit_bigint_result(value: *mut BigInt) -> JitBigIntResult {
    value
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn encode_jit_bigint_result(value: *mut BigInt) -> JitBigIntResult {
    value as usize as i64
}

/// GC type id for the raw `rbigint` payload, published at JitDriver init by
/// `set_bigint_gc_type_id`. `0` until then, in which case the alloc helpers
/// fall back to leaked raw allocations in bare tests / pre-init bootstrap.

/// Record the GC type id registered for the `rbigint` payload (called once from
/// `pyre-jit::eval` after `gc.register_type`).
pub fn set_bigint_gc_type_id(id: u32) {
    crate::rbigint::set_rbigint_gc_type_id(id);
}

/// Reads the runtime-assigned rbigint type id (set once at init by
/// [`set_bigint_gc_type_id`]); the value is not a build-time constant, so the
/// JIT residualises the read instead of tracing into it (`@dont_look_inside`).
#[majit_macros::dont_look_inside]
pub fn bigint_gc_type_id() -> u32 {
    crate::rbigint::rbigint_gc_type_id()
}

/// Allocate `value` as a GC-managed `rbigint` in the nursery (no-collect host
/// path). The caller must immediately store the result into a traced owner:
/// `w_long_new` uses a non-collecting old-gen wrapper allocation plus a
/// creation barrier, while JIT `*_raw` helpers flow straight into the boxing
/// `NewWithVtable`. Falls back to leaked `malloc_raw` when no GC hook is
/// installed (bare unit tests, where the result is never traced).
#[inline]
pub fn alloc_bigint_nursery(value: BigInt) -> *mut BigInt {
    crate::rbigint::alloc_rbigint_nursery(value)
}

/// Allocate `value` as a GC-managed `rbigint` through the collecting nursery.
/// a minor collection fires when the nursery is full (reclaiming dead bigints)
/// instead of spilling to old-gen unbounded. Only for the elidable bigint
/// payload helpers (`jit_bigint_*`), which the walker emits as a residual
/// `CallR` whose gcmap roots the trace's live set, and which read both operand
/// payloads into a local sum before allocating — so nothing unrooted is held
/// across the embedded minor cycle. Falls back to the no-collect path when no
/// collecting hook is installed (other backends), then to `malloc_raw`.
/// The result's digits are an ordinary nursery `GcArray(Signed)`, so their bytes
/// already participate in nursery pressure; no foreign/external-memory charge
/// is needed.
#[inline]
pub fn alloc_bigint_nursery_collecting(value: BigInt) -> *mut BigInt {
    crate::rbigint::alloc_rbigint_nursery_collecting(value)
}

/// Allocate `value` as a GC-managed `rbigint` at a stable (old-gen, non-moving)
/// address for callers that must retain a raw payload pointer across a
/// potentially collecting operation. Falls back to the leaked `malloc_raw`
/// pre-init.
#[inline]
pub fn alloc_bigint_stable(value: BigInt) -> *mut BigInt {
    crate::rbigint::alloc_rbigint_stable(value)
}

/// Wrap an already heap-allocated `*mut BigInt` in a fresh W_LongObject
/// without copying the payload — the wrapper just stores `value`, it does not
/// take exclusive ownership. Pure-call CSE of the elidable `rbigint` helpers
/// can fold two ops to the same `*mut BigInt`, so one payload may back more
/// than one wrapper; that is sound because payloads are immutable after
/// initialization and every wrapper/trace op treats the payload as a GC ref.
pub fn w_long_from_raw(value: *mut BigInt) -> PyObjectRef {
    // W_LongObject shares the `int` type with W_IntObject — the two only
    // differ in their storage layout, not their Python-level identity
    // (PyPy does the same via W_AbstractIntObject's typedef). Wire
    // `w_class` to INT_TYPE.instantiate so `type(x) is int` and
    // `isinstance(x, int)` both hold for long integers.
    let header = PyObject {
        ob_type: &LONG_TYPE as *const PyType,
        w_class: get_instantiate(&INT_TYPE),
    };
    // The wrapper must be GC-managed whenever its `value` payload is: a
    // `BigInt` routed through the GC (`bigint_gc_type_id() != 0`, the
    // `alloc_bigint_*` condition) is reclaimed by collections, so an immortal
    // `malloc_typed` wrapper — which the collector never traces — would leave
    // an untracked edge to a reclaimed RBigInt payload/digit array. Tie the
    // wrapper's GC path to the same predicate as the payload, not to
    // `gc_interp::enabled()` alone (unlike int/float, whose payload is inline).
    if crate::gc_interp::enabled() || bigint_gc_type_id() != 0 {
        // `alloc_oldgen_typed` is a direct, non-collecting old-gen allocation.
        // The young immutable payload therefore stays at the same address
        // until the wrapper is initialized and remembered below.
        let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_LONG_GC_TYPE_ID, W_LONG_OBJECT_SIZE);
        if !raw.is_null() {
            // Advance the dispatch-loop safepoint counter, as w_int_new /
            // w_float_new do for their stable allocs — otherwise a long-dominated
            // interpreter workload never reaches the safepoint threshold and the
            // dead old-gen long wrappers + their bigint payloads accumulate.
            crate::gc_interp::note_alloc();
            unsafe {
                std::ptr::write(
                    raw as *mut W_LongObject,
                    W_LongObject {
                        ob_header: header,
                        value,
                    },
                );
            }
            // Creation write barrier: the old-gen wrapper may reference a young
            // bigint, so remember it for the next minor collection's tracer.
            crate::gc_hook::try_gc_write_barrier(raw);
            return raw as PyObjectRef;
        }
    }
    crate::lltype::malloc_typed(W_LongObject {
        ob_header: header,
        value,
    }) as PyObjectRef
}

/// Allocate a new W_LongObject on the heap from a `BigInt` value. The bigint
/// payload is a fresh nursery object, matching RPython's ordinary GC
/// allocation. The wrapper itself is born in non-moving old-gen and its
/// creation barrier records the old→young `LONG_VALUE_OFFSET` edge.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py:139`): the body
/// delegates through `alloc_bigint_nursery -> *mut BigInt`, so tracing into it
/// leaks the raw `BigInt` pointee into the return model and unifies against the
/// `w_int_new` fast path as `PyObject ∪ BigInt`. Residualising the whole box
/// tail models it by signature — a plain `PyObjectRef` GCREF with no
/// discriminant to erase — the `w_str_new`/`box_bigint_constant` twin.
#[majit_macros::dont_look_inside]
pub fn w_long_new(value: BigInt) -> PyObjectRef {
    w_long_from_raw(alloc_bigint_nursery(value))
}

/// Create a W_LongObject from an i64 value.
pub fn w_long_from_i64(v: i64) -> PyObjectRef {
    w_long_new(BigInt::from(v))
}

/// Box a bigint constant into a heap Python int object.
pub fn box_bigint_constant(value: &BigInt) -> PyObjectRef {
    w_long_new(value.clone())
}

/// `rbigint.fromint()` (`rpython/rlib/rbigint.py:225`,
/// `@jit.elidable`) with the translated one-GC-reference ABI.
///
/// Rust returns `RBigInt` by value, but RPython returns a GC reference.  The
/// MIR front retargets machine-word constructor calls here so generated JIT
/// code receives a rooted `*mut RBigInt`, encoded in the JIT's uniform i64
/// word ABI on wasm32 rather than returned as a native wasm pointer.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_from_i64(value: i64) -> JitBigIntResult {
    encode_jit_bigint_result(alloc_bigint_nursery_collecting(BigInt::fromint(value)))
}

/// Unsigned machine-word companion of [`jit_bigint_from_i64`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_from_u64(value: u64) -> JitBigIntResult {
    encode_jit_bigint_result(alloc_bigint_nursery_collecting(BigInt::from_u128(
        value as u128,
    )))
}

macro_rules! bigint_comparison_residual {
    ($name:ident, $method:ident) => {
        #[doc = "Bare-RBigInt comparison residual using the translated GC-reference ABI."]
        #[majit_macros::elidable_cannot_raise]
        pub extern "C" fn $name(a: i64, b: i64) -> i64 {
            let (a, b) = (a as *const BigInt, b as *const BigInt);
            unsafe { BigInt::$method(&*a, &*b) as i64 }
        }
    };
}

bigint_comparison_residual!(jit_bigint_eq, eq);
bigint_comparison_residual!(jit_bigint_ne, ne);
bigint_comparison_residual!(jit_bigint_lt, lt);
bigint_comparison_residual!(jit_bigint_le, le);
bigint_comparison_residual!(jit_bigint_gt, gt);
bigint_comparison_residual!(jit_bigint_ge, ge);

macro_rules! bigint_scalar_residual {
    ($name:ident, $body:expr) => {
        #[doc = "Bare-RBigInt scalar residual using the translated GC-reference ABI."]
        #[majit_macros::elidable_cannot_raise]
        pub extern "C" fn $name(value: i64) -> i64 {
            let value = value as *const BigInt;
            let value = unsafe { &*value };
            ($body)(value) as i64
        }
    };
}

bigint_scalar_residual!(jit_bigint_bits, |value: &BigInt| value.bits());
bigint_scalar_residual!(jit_bigint_is_zero, |value: &BigInt| value.is_zero());
bigint_scalar_residual!(jit_bigint_is_one, |value: &BigInt| value.is_one());
bigint_scalar_residual!(jit_bigint_tobool, |value: &BigInt| value.tobool());
bigint_scalar_residual!(jit_bigint_hash, |value: &BigInt| value.hash());

/// `W_LongObject._fits_int()` — longobject.py:141 / rbigint.fits_int.
/// True if the value fits in a machine-word integer (i64 on 64-bit).
/// Used by `is_plain_int1` to accept long objects that are in the int range.
#[inline]
pub unsafe fn w_long_fits_int(obj: PyObjectRef) -> bool {
    unsafe {
        let big = w_long_get_value(obj);
        jit_bigint_to_i64_fits(big) != 0
    }
}

/// True when the W_LongObject's BigInt is zero. Divisor guard for the
/// can-raise floordiv/mod fast path (a zero divisor makes the payload helper
/// publish ZeroDivisionError, which the trait path defers to the generic
/// residual rather than triggering during tracing).
///
/// # Safety
/// `obj` must point to a valid `W_LongObject`.
#[inline]
pub unsafe fn w_long_is_zero(obj: PyObjectRef) -> bool {
    unsafe { w_long_get_value(obj).sign() == RBigIntSign::NoSign }
}

/// Extract a reference to the BigInt value from a known W_LongObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_LongObject`.
#[inline]
pub unsafe fn w_long_get_value(obj: PyObjectRef) -> &'static BigInt {
    unsafe {
        let long_obj = obj as *const W_LongObject;
        &*(*long_obj).value
    }
}

/// `rbigint.fits_int()` (`rpython/rlib/rbigint.py:490`) — JIT-callable
/// wrapper. Returns 1 when the W_LongObject's BigInt fits in i64,
/// 0 otherwise. Used as the runtime fits_int guard before
/// `jit_w_long_toint`.
///
/// Unlike `rbigint.toint()`, upstream `fits_int()` is not marked
/// `@jit.elidable`, so keep this call cannot-raise but non-elidable.
pub extern "C" fn jit_w_long_fits_int(obj: i64) -> i64 {
    let obj = obj as PyObjectRef;
    unsafe { w_long_fits_int(obj) as i64 }
}

/// `rbigint.fits_int()` on a bare `*mut BigInt` — the demote guard for the
/// inline-NEW boxing of a `jit_w_long_*_raw` result. Returns 1 when the bigint
/// fits i64 (i.e. should demote to `W_IntObject`), 0 otherwise. The walker/trait
/// emit `GuardFalse(fits)` after the raw op so a result that does fit deopts to
/// the interpreter (which performs the demote); the common bigint case (does
/// not fit) passes the guard and falls through to `NewWithVtable(W_LONG)`.
/// Non-elidable, cannot-raise (mirrors [`jit_w_long_fits_int`]).
///
/// # Safety note: `extern "C"` over an `i64`-encoded `*mut BigInt`, matching the
/// raw-helper ABI. The pointer is a live GC bigint produced by a preceding
/// raw op in the same trace.
pub extern "C" fn jit_bigint_fits_int(num: i64) -> i64 {
    let num = num as *const BigInt;
    unsafe { jit_bigint_to_i64_fits(&*num) }
}

/// `rbigint.fits_int()` (`rpython/rlib/rbigint.py:490`) on a borrowed
/// BigInt payload. Scalar half of the `BigInt::to_i64()` split used by
/// the two-phase rtyper so it never has to model an `Option<i64>` ABI.
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_i64_fits(num: &BigInt) -> i64 {
    i64::try_from(num).is_ok() as i64
}

/// `rbigint.toint()` (`rpython/rlib/rbigint.py:465`, `@jit.elidable`) on a
/// borrowed BigInt payload. Callers must first check
/// [`jit_bigint_to_i64_fits`]; overflow means that guard was violated.
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_i64_value(num: &BigInt) -> i64 {
    i64::try_from(num).unwrap_or_else(|_| {
        panic!("jit_bigint_to_i64_value: BigInt out of i64 range - fits guard violated")
    })
}

/// `rbigint.toint()` payload for the runtime-fallible narrowing lowering:
/// the value is read only on the fits (Ok) path, so out-of-range returns 0
/// instead of panicking — the walker's discriminant switch discards it on
/// the Err path. Companion of [`jit_bigint_to_i64_fits`].
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_i64_value_or_zero(num: &BigInt) -> i64 {
    i64::try_from(num).unwrap_or(0)
}

/// `BigInt::to_u64().is_some()` on a borrowed BigInt payload. Scalar half of
/// the `BigInt::to_u64()` split so the two-phase rtyper never has to model an
/// `Option<u64>` ABI. Companion of [`jit_bigint_to_u64_value`].
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_u64_fits(num: &BigInt) -> i64 {
    num.to_u64().is_some() as i64
}

/// `BigInt::to_u64()` on a borrowed BigInt payload. Callers must first check
/// [`jit_bigint_to_u64_fits`]; a `None` here means that guard was violated.
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_u64_value(num: &BigInt) -> u64 {
    num.to_u64().unwrap_or_else(|| {
        panic!("jit_bigint_to_u64_value: BigInt exceeds u64 range - fits guard violated")
    })
}

/// `rbigint.sign` / sign-digit use (`rpython/rlib/rbigint.py`) on a borrowed
/// RBigInt payload. Returns the scalar signum (-1, 0, +1) so the two-phase
/// rtyper does not need a synthetic Rust enum ABI at this boundary.
#[majit_macros::elidable_cannot_raise]
pub fn jit_bigint_sign_i64(num: &BigInt) -> i64 {
    match num.sign() {
        RBigIntSign::Minus => -1,
        RBigIntSign::NoSign => 0,
        RBigIntSign::Plus => 1,
    }
}

/// `rbigint.tofloat()` (`rpython/rlib/rbigint.py:503`) on a borrowed BigInt
/// payload, with the caller's existing overflow sentinel folded into the
/// scalar return.
#[majit_macros::elidable_cannot_raise]
pub fn jit_bigint_to_f64_or_inf(num: &BigInt) -> f64 {
    num.to_f64().unwrap_or(f64::INFINITY)
}

/// `rbigint.tofloat()` (`rpython/rlib/rbigint.py:503`) on a borrowed BigInt
/// payload, preserving callers that intentionally collapse overflow to NaN.
#[majit_macros::elidable_cannot_raise]
pub fn jit_bigint_to_f64_or_nan(num: &BigInt) -> f64 {
    num.to_f64().unwrap_or(f64::NAN)
}

/// `W_LongObject.toint()` (`pypy/objspace/std/longobject.py:138`) →
/// `rbigint.toint()` (`rpython/rlib/rbigint.py:465`, `@jit.elidable`).
/// Extract an i64 from a W_LongObject. RPython `toint` raises
/// `OverflowError` when the BigInt does not fit; the elidable
/// trace-time site emits a `fits_int` GUARD_TRUE first
/// (`pypy/objspace/std/listobject.py:2390 is_plain_int1` parity), so
/// the OverflowError path is unreachable in production. Pyre encodes
/// that unreachability as a panic. There is no `_int_w_unsafe` upstream —
/// this is the elidable `toint` after a `fits_int` guard.
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_toint(obj: i64) -> i64 {
    let obj = obj as PyObjectRef;
    unsafe {
        let big = w_long_get_value(obj);
        i64::try_from(big).unwrap_or_else(|_| {
            panic!("jit_w_long_toint: BigInt out of i64 range — fits_int guard violated")
        })
    }
}

/// `rbigint.add` (`rpython/rlib/rbigint.py:269`, `@jit.elidable`) — the
/// payload half of `W_LongObject._add` (`pypy/objspace/std/longobject.py:331`).
/// Both operands are guaranteed `W_LongObject` by a preceding
/// `GuardClass(LONG_TYPE)` on each, so the BigInt payloads are read
/// directly. Returns a freshly heap-allocated `*mut BigInt` (as i64) — the
/// arithmetic only, with no Python-object wrapper. `add` allocates a new
/// bigint, so its only failure mode is MemoryError: `EF_ELIDABLE_OR_MEMORYERROR`
/// (`call.py:294`, `cr == "mem"`). The value is still a pure function of the
/// operand payloads, so the optimizer may fold/CSE it; a trailing
/// `GuardNoException` covers the allocation. The result is an internal bigint
/// never exposed to Python `is`, so sharing one payload for two equal-input
/// adds is unobservable.
/// Wrapper-level (`W_LongObject` operands) variants used for record-time
/// concrete evaluation and the trait path, which run OUTSIDE a JIT safepoint and
/// hold the operand wrappers natively — so they allocate via the NO-COLLECT
/// `alloc_bigint_nursery` (a collection here would move the tracer's operands).
/// The walker-emitted runtime call uses the collecting payload variants below.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_add_raw(a: i64, b: i64) -> i64 {
    let (a, b) = (a as PyObjectRef, b as PyObjectRef);
    unsafe { alloc_bigint_nursery(w_long_get_value(a) + w_long_get_value(b)) as i64 }
}

/// `rbigint.sub` over `W_LongObject` operands (no-collect). See [`jit_w_long_add_raw`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_sub_raw(a: i64, b: i64) -> i64 {
    let (a, b) = (a as PyObjectRef, b as PyObjectRef);
    unsafe { alloc_bigint_nursery(w_long_get_value(a) - w_long_get_value(b)) as i64 }
}

/// `rbigint.mul` over `W_LongObject` operands (no-collect). See [`jit_w_long_add_raw`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_mul_raw(a: i64, b: i64) -> i64 {
    let (a, b) = (a as PyObjectRef, b as PyObjectRef);
    unsafe { alloc_bigint_nursery(w_long_get_value(a) * w_long_get_value(b)) as i64 }
}

/// `rbigint.and_` over `W_LongObject` operands (no-collect). See [`jit_w_long_add_raw`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_and_raw(a: i64, b: i64) -> i64 {
    let (a, b) = (a as PyObjectRef, b as PyObjectRef);
    unsafe { alloc_bigint_nursery(w_long_get_value(a) & w_long_get_value(b)) as i64 }
}

/// `rbigint.or_` over `W_LongObject` operands (no-collect). See [`jit_w_long_add_raw`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_or_raw(a: i64, b: i64) -> i64 {
    let (a, b) = (a as PyObjectRef, b as PyObjectRef);
    unsafe { alloc_bigint_nursery(w_long_get_value(a) | w_long_get_value(b)) as i64 }
}

/// `rbigint.xor_` over `W_LongObject` operands (no-collect). See [`jit_w_long_add_raw`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_w_long_xor_raw(a: i64, b: i64) -> i64 {
    let (a, b) = (a as PyObjectRef, b as PyObjectRef);
    unsafe { alloc_bigint_nursery(w_long_get_value(a) ^ w_long_get_value(b)) as i64 }
}

/// `rbigint.add`/`sub`/`mul`/`and_`/`or_`/`xor_` (`rpython/rlib/rbigint.py`,
/// each `@jit.elidable`) on bare `*const BigInt` payloads — the elidable
/// arithmetic the walker emits after reading each operand's immutable `value`
/// via an immutable `GetfieldGc`. Taking the payloads (not the `W_LongObject` wrappers)
/// keeps the call's inputs the immutable bigints, so the optimizer forwards the
/// field read and never reorders this elidable call ahead of the boxing
/// `setfield_gc` that initializes the fresh result wrapper. Allocates the result
/// via the COLLECTING nursery (the call is a gcmap-rooted residual `CallR`
/// holding no unrooted pointer across the alloc), so dead bigints are reclaimed
/// by minor collections instead of accumulating in old-gen. Returns a freshly
/// heap-allocated `*mut BigInt` (as i64). Allocates → `EF_ELIDABLE_OR_MEMORYERROR`.
///
/// # Safety note: `extern "C"` over `i64`-encoded `*const BigInt`. The pointers
/// are live GC bigints (the operands' value fields) for the duration of the call.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_add(a: i64, b: i64) -> i64 {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    // Two-limb fast path: when both operands and the sum fit i128 the result
    // is exact without the general limb machinery.
    unsafe {
        if let (Ok(x), Ok(y)) = (i128::try_from(&*a), i128::try_from(&*b)) {
            if let Some(z) = x.checked_add(y) {
                return alloc_bigint_nursery_collecting(BigInt::from(z)) as i64;
            }
        }
        alloc_bigint_nursery_collecting(&*a + &*b) as i64
    }
}

/// `rbigint.add_int_int_bigint_result` (`rpython/rlib/rbigint.py:717`,
/// `@jit.elidable`) — exact bigint sum of two machine ints. Allocates the
/// result via the COLLECTING nursery, matching [`jit_bigint_add`], and returns
/// a freshly heap-allocated `*mut BigInt` payload (as i64).
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_add_int_int(a: i64, b: i64) -> i64 {
    // Exact in i128 for any i64 pair; skips the general bigint add machinery.
    alloc_bigint_nursery_collecting(BigInt::from(a as i128 + b as i128)) as i64
}

/// `rbigint.sub` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_sub(a: i64, b: i64) -> i64 {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    // Two-limb fast path mirroring `jit_bigint_add`.
    unsafe {
        if let (Ok(x), Ok(y)) = (i128::try_from(&*a), i128::try_from(&*b)) {
            if let Some(z) = x.checked_sub(y) {
                return alloc_bigint_nursery_collecting(BigInt::from(z)) as i64;
            }
        }
        alloc_bigint_nursery_collecting(&*a - &*b) as i64
    }
}

/// `rbigint.sub_int_int_bigint_result` (`rpython/rlib/rbigint.py:788`,
/// `@jit.elidable`) — exact bigint difference of two machine ints. See
/// [`jit_bigint_add_int_int`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_sub_int_int(a: i64, b: i64) -> i64 {
    // Exact in i128 for any i64 pair; skips the general bigint sub machinery.
    alloc_bigint_nursery_collecting(BigInt::from(a as i128 - b as i128)) as i64
}

/// `rbigint.mul` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_mul(a: i64, b: i64) -> i64 {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe { alloc_bigint_nursery_collecting(&*a * &*b) as i64 }
}

/// `rbigint.mul_int_int_bigint_result` (`rpython/rlib/rbigint.py:873`,
/// `@jit.elidable`) — exact bigint product of two machine ints. See
/// [`jit_bigint_add_int_int`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_mul_int_int(a: i64, b: i64) -> i64 {
    // A 64x64 product is exact in i128; skips the general bigint mul machinery.
    alloc_bigint_nursery_collecting(BigInt::from(a as i128 * b as i128)) as i64
}

/// `rbigint.and_` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_and(a: i64, b: i64) -> i64 {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe { alloc_bigint_nursery_collecting(&*a & &*b) as i64 }
}

/// `rbigint.or_` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_or(a: i64, b: i64) -> i64 {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe { alloc_bigint_nursery_collecting(&*a | &*b) as i64 }
}

/// `rbigint.xor_` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_xor(a: i64, b: i64) -> i64 {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe { alloc_bigint_nursery_collecting(&*a ^ &*b) as i64 }
}

/// `rbigint` comparison payload for `W_LongObject` — returns the sign of
/// `a <=> b` as `-1` / `0` / `1`. RPython exposes the comparison as six methods
/// (`lt`/`le`/`eq`/`ne`/`gt`/`ge`, the latter built as `other.lt(self)`
/// wrappers, `rbigint.py:573/664`); Rust's total `Ord::cmp` collapses them into
/// one three-way result, and the caller recovers each relation with a plain
/// `int_<cmp>(sign, 0)` (e.g. `a < b` ⟺ `sign < 0`, `a == b` ⟺ `sign == 0`).
/// A comparison neither allocates nor raises, so this is
/// `EF_ELIDABLE_CANNOT_RAISE` and the fast path records `CallPure*` with NO
/// trailing guard.
#[majit_macros::elidable_cannot_raise]
pub extern "C" fn jit_w_long_cmp(a: i64, b: i64) -> i64 {
    use core::cmp::Ordering;
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe {
        match w_long_get_value(a).cmp(w_long_get_value(b)) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_create_and_read() {
        let obj = w_long_new(BigInt::from(42));
        unsafe {
            assert!(is_long(obj));
            assert!(!is_int(obj));
            assert_eq!(*w_long_get_value(obj), BigInt::from(42));
        }
    }

    #[test]
    fn test_long_from_i64() {
        let obj = w_long_from_i64(i64::MAX);
        unsafe {
            assert!(is_long(obj));
            assert_eq!(*w_long_get_value(obj), BigInt::from(i64::MAX));
        }
    }

    #[test]
    fn test_long_large_value() {
        let big = BigInt::from(i64::MAX) + BigInt::from(1);
        let obj = w_long_new(big.clone());
        unsafe {
            assert!(is_long(obj));
            assert_eq!(*w_long_get_value(obj), big);
        }
    }

    #[test]
    fn test_long_field_offset() {
        assert_eq!(LONG_VALUE_OFFSET, 16);
    }

    #[test]
    fn test_long_type_name_is_int() {
        // Python users see "int" for both W_IntObject and W_LongObject
        assert_eq!(LONG_TYPE.name, "int");
    }

    #[test]
    fn test_jit_w_long_fits_int_in_range() {
        let obj = w_long_from_i64(123);
        assert_eq!(jit_w_long_fits_int(obj as i64), 1);
        let obj = w_long_from_i64(i64::MAX);
        assert_eq!(jit_w_long_fits_int(obj as i64), 1);
        let obj = w_long_from_i64(i64::MIN);
        assert_eq!(jit_w_long_fits_int(obj as i64), 1);
    }

    #[test]
    fn test_jit_w_long_fits_int_out_of_range() {
        let big = BigInt::from(i64::MAX) + BigInt::from(1);
        let obj = w_long_new(big);
        assert_eq!(jit_w_long_fits_int(obj as i64), 0);
        let big = BigInt::from(i64::MIN) - BigInt::from(1);
        let obj = w_long_new(big);
        assert_eq!(jit_w_long_fits_int(obj as i64), 0);
    }

    #[test]
    fn test_jit_w_long_toint_extracts_i64() {
        let obj = w_long_from_i64(42);
        assert_eq!(jit_w_long_toint(obj as i64), 42);
        let obj = w_long_from_i64(i64::MAX);
        assert_eq!(jit_w_long_toint(obj as i64), i64::MAX);
        let obj = w_long_from_i64(i64::MIN);
        assert_eq!(jit_w_long_toint(obj as i64), i64::MIN);
    }

    #[test]
    fn test_jit_w_long_add_raw_payload() {
        // The elidable half returns a bare `*mut BigInt` carrying the sum,
        // with no Python-object wrapper.
        let a = w_long_new(BigInt::from(i64::MAX));
        let b = w_long_new(BigInt::from(i64::MAX));
        let raw = jit_w_long_add_raw(a as i64, b as i64) as *mut BigInt;
        unsafe {
            assert_eq!(*raw, BigInt::from(i64::MAX) * 2);
        }
    }

    #[test]
    fn test_jit_w_long_binop_raw_payloads() {
        // sub/mul/and/or/xor raw helpers mirror jit_w_long_add_raw: bare
        // `*mut BigInt` carrying the arithmetic result, no Python wrapper.
        let x = BigInt::from(i64::MAX) + BigInt::from(7);
        let y = BigInt::from(i64::MAX) - BigInt::from(3);
        let a = w_long_new(x.clone());
        let b = w_long_new(y.clone());
        unsafe {
            let sub = jit_w_long_sub_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*sub, &x - &y);
            let mul = jit_w_long_mul_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*mul, &x * &y);
            let and = jit_w_long_and_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*and, &x & &y);
            let or = jit_w_long_or_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*or, &x | &y);
            let xor = jit_w_long_xor_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*xor, &x ^ &y);
        }
    }

    #[test]
    fn test_jit_w_long_add_raw_keeps_payload_when_fits() {
        // The raw helper never demotes: a sum that fits i64 still yields a
        // `*mut BigInt` payload (the boxing NEW wraps it as a W_LongObject,
        // matching `newlong` which does not demote).
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_long_new(BigInt::from(-1) - BigInt::from(i64::MAX));
        let raw = jit_w_long_add_raw(a as i64, b as i64) as *mut BigInt;
        unsafe {
            assert_eq!(*raw, BigInt::from(0));
        }
    }

    #[test]
    fn test_bare_bigint_constructor_comparison_and_scalar_residuals() {
        let a = jit_bigint_from_i64(-42);
        let b = jit_bigint_from_u64(42);
        unsafe {
            assert_eq!(&*a, &BigInt::from(-42));
            assert_eq!(&*b, &BigInt::from(42));
        }
        assert_eq!(jit_bigint_eq(a as i64, b as i64), 0);
        assert_eq!(jit_bigint_lt(a as i64, b as i64), 1);
        assert_eq!(jit_bigint_ge(b as i64, a as i64), 1);
        assert_eq!(jit_bigint_sign_i64(unsafe { &*a }), -1);
        assert_eq!(jit_bigint_is_zero(a as i64), 0);
        assert_eq!(jit_bigint_tobool(a as i64), 1);
        assert_eq!(jit_bigint_bits(b as i64), 6);
        assert_eq!(jit_bigint_hash(b as i64), 42);
    }
}
