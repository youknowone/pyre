//! W_LongObject -- arbitrary-precision integer backed by `BigInt`.
//!
//! Used when i64 overflow is detected in `W_IntObject` arithmetic.
//! The JIT may specialize bigint operations by reading immutable `value`
//! payloads, calling pure raw-payload helpers, and boxing the resulting payload
//! with the same `W_LongObject` layout.

use malachite_bigint::BigInt;

use crate::pyobject::*;

/// Arbitrary-precision integer object.
///
/// Layout: `[ob_type: *const PyType | value: *mut BigInt]`
/// The `value` pointer references an immutable `BigInt` payload, usually
/// GC-managed and occasionally leaked via `malloc_raw` before GC init.
#[repr(C)]
pub struct W_LongObject {
    pub ob_header: PyObject,
    pub value: *mut BigInt,
}

// Safety: BigInt is Send+Sync and W_LongObject only stores a raw pointer
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

/// Payload size of a raw `BigInt` GC object (the malachite struct only; its
/// limb `Vec` lives in malachite's own heap and is freed by [`bigint_destructor`]).
pub const BIGINT_PAYLOAD_SIZE: usize = std::mem::size_of::<BigInt>();

/// GC type id for the raw `BigInt` payload, published at JitDriver init by
/// `set_bigint_gc_type_id`. `0` until then, in which case the alloc helpers
/// fall back to the leaked `malloc_raw` (bare unit tests / pre-init bootstrap).
///
/// Unlike the `W_*` object ids this is set at runtime rather than a fixed
/// const: the `BigInt` payload is never embedded in a JIT descr (it is only
/// ever allocated by the host `try_gc_alloc` path, never `NewWithVtable`'d in a
/// trace), so it needs no compile-time-stable value.
static BIGINT_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the `BigInt` payload (called once from
/// `pyre-jit::eval` after `gc.register_type`).
pub fn set_bigint_gc_type_id(id: u32) {
    BIGINT_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Reads the runtime-assigned `BIGINT_GC_TYPE_ID` atomic (set once at init by
/// [`set_bigint_gc_type_id`]); the value is not a build-time constant, so the
/// JIT residualises the read instead of tracing into it (`@dont_look_inside`).
#[majit_macros::dont_look_inside]
pub fn bigint_gc_type_id() -> u32 {
    BIGINT_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Lightweight GC destructor for the `BigInt` payload: run its drop glue so
/// malachite's limb `Vec` is freed when the collector reclaims a dead bigint.
/// Registered on [`BIGINT_GC_TYPE_ID`] via `TypeInfo::with_destructor`; the
/// copying nursery would otherwise abandon the payload without dropping it,
/// leaking the limbs (the `malloc_raw` leak this whole path replaces).
///
/// # Safety
/// `addr` must point at a live, initialized `BigInt` payload the GC is
/// reclaiming. The collector calls it exactly once, on the final dead copy
/// (a survived/forwarded object is re-listed, never destructed).
pub unsafe fn bigint_destructor(addr: usize) {
    unsafe { std::ptr::drop_in_place(addr as *mut BigInt) }
}

/// Allocate `value` as a GC-managed `BigInt` in the nursery (no-collect host
/// path). For the JIT `*_raw` arithmetic helpers, whose result flows straight
/// into the boxing `NewWithVtable` in the same trace — that collecting NEW
/// gcmap-roots the returned pointer, so a young payload is forwarded/promoted
/// rather than dangling. Falls back to the leaked `malloc_raw` when no GC hook
/// is installed (bare unit tests, where the result is never traced).
#[inline]
pub fn alloc_bigint_nursery(value: BigInt) -> *mut BigInt {
    let tid = bigint_gc_type_id();
    if tid != 0 {
        if let Some(raw) =
            crate::gc_hook::try_gc_alloc(tid, BIGINT_PAYLOAD_SIZE).filter(|p| !p.is_null())
        {
            let external = bigint_external_bytes(&value);
            unsafe {
                std::ptr::write(raw as *mut BigInt, value);
            }
            crate::gc_hook::try_gc_charge_oldgen_external(raw as usize, external);
            return raw as *mut BigInt;
        }
    }
    crate::lltype::malloc_raw(value)
}

/// The external (off-heap, GC-invisible) byte footprint of a `BigInt`'s limb
/// `Vec` — `ceil(bits/64)` limbs of 8 bytes each. `bits()` is constant-time
/// (reads only the top limb, no allocation). Values that fit one machine word
/// (`bits <= 64`) are stored inline with no `Vec`, so they carry 0 external bytes.
#[inline]
fn bigint_external_bytes(value: &BigInt) -> usize {
    let bits = value.bits();
    if bits <= 64 {
        0
    } else {
        ((bits + 63) / 64) as usize * 8
    }
}

/// External (off-heap) byte footprint of a GC `BigInt` payload at `addr` — the
/// payload base is the `BigInt` itself (see [`bigint_destructor`]). Registered
/// as the type's `external_size` so the collector folds a promoted bigint's
/// limb `Vec` into the major-collection threshold.
///
/// # Safety
/// `addr` must point at a live, initialized `BigInt` payload (true for a
/// promoted or surviving old-gen bigint the collector is accounting for).
pub unsafe fn bigint_external_size(addr: usize) -> usize {
    bigint_external_bytes(unsafe { &*(addr as *const BigInt) })
}

/// Allocate `value` as a GC-managed `BigInt` through the *collecting* nursery —
/// a minor collection fires when the nursery is full (reclaiming dead bigints)
/// instead of spilling to old-gen unbounded. Only for the elidable bigint
/// payload helpers (`jit_bigint_*`), which the walker emits as a residual
/// `CallR` whose gcmap roots the trace's live set, and which read both operand
/// payloads into a local sum before allocating — so nothing unrooted is held
/// across the embedded minor cycle. Falls back to the no-collect path when no
/// collecting hook is installed (other backends), then to `malloc_raw`.
///
/// Before allocating, the result's limb-`Vec` bytes are charged as off-heap
/// memory pressure so the nursery's minor cadence reflects the bignum's true
/// footprint, not just the 48-byte struct the bump pointer tracks (otherwise the
/// last nursery generation of large bigints accumulates uncollected). The charge
/// may itself force a minor; that is safe here because it runs before the fresh
/// `value` is written into the nursery (it still lives only on the Rust stack and
/// holds no nursery GC pointer) while the operand bigints are already boxed and
/// gcmap-rooted at the residual call.
#[inline]
pub fn alloc_bigint_nursery_collecting(value: BigInt) -> *mut BigInt {
    let tid = bigint_gc_type_id();
    if tid != 0 {
        let external = bigint_external_bytes(&value);
        crate::gc_hook::try_gc_charge_memory_pressure(external);
        if let Some(raw) = crate::gc_hook::try_gc_alloc_collecting(tid, BIGINT_PAYLOAD_SIZE)
            .filter(|p| !p.is_null())
        {
            unsafe {
                std::ptr::write(raw as *mut BigInt, value);
            }
            crate::gc_hook::try_gc_charge_oldgen_external(raw as usize, external);
            return raw as *mut BigInt;
        }
    }
    alloc_bigint_nursery(value)
}

/// Allocate `value` as a GC-managed `BigInt` at a stable (old-gen, non-moving)
/// address, for host/interpreter callers (`w_long_new`) that hold the pointer
/// on the Rust stack without rooting it. Mirrors `w_float_new`'s
/// `try_gc_alloc_stable`. Falls back to the leaked `malloc_raw` pre-init.
#[inline]
pub fn alloc_bigint_stable(value: BigInt) -> *mut BigInt {
    let tid = bigint_gc_type_id();
    if tid != 0 {
        if let Some(raw) =
            crate::gc_hook::try_gc_alloc_stable(tid, BIGINT_PAYLOAD_SIZE).filter(|p| !p.is_null())
        {
            // Charge the limb-`Vec` bytes against the old-gen external total so a
            // directly-old-gen bignum's footprint enters the major threshold now,
            // not only at the next major's recompute. No minor is forced, so this
            // is safe on the unrooted host/interpreter path (a memory-pressure
            // charge here could force an unsafe moving minor).
            let external = bigint_external_bytes(&value);
            unsafe {
                std::ptr::write(raw as *mut BigInt, value);
            }
            crate::gc_hook::try_gc_charge_oldgen_external(raw as usize, external);
            return raw as *mut BigInt;
        }
    }
    crate::lltype::malloc_raw(value)
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
    // `malloc_typed` wrapper — which the collector never traces — would let a
    // major collection run the payload's destructor (freeing the limb `Vec`)
    // while the wrapper still points at it. Tie the wrapper's GC path to the
    // same predicate as the payload, not to `gc_interp::enabled()` alone
    // (unlike int/float, whose payload is inline and carries no destructor).
    if crate::gc_interp::enabled() || bigint_gc_type_id() != 0 {
        // Pin the (possibly young) `value` payload across the wrapper malloc:
        // a minor collection inside `try_gc_alloc_stable` can move a young
        // bigint, so re-read its address afterwards. The proxy/dictproxy
        // `gct_fv_gc_malloc` bracket pattern, but for a raw `*mut BigInt`
        // rooted as a GcRef slot rather than a PyObjectRef.
        let mut slot = value as *mut u8;
        let pinned = unsafe { crate::gc_hook::try_gc_add_root(&mut slot as *mut *mut u8) };
        let raw = crate::gc_hook::try_gc_alloc_stable(W_LONG_GC_TYPE_ID, W_LONG_OBJECT_SIZE)
            .filter(|p| !p.is_null());
        let value = slot as *mut BigInt;
        if pinned {
            crate::gc_hook::try_gc_remove_root(&mut slot as *mut *mut u8);
        }
        if let Some(raw) = raw {
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
/// payload is GC-managed at a stable address (held on the Rust stack by host
/// callers without rooting), and the wrapper traces it via the registered
/// `LONG_VALUE_OFFSET` gc-pointer.
pub fn w_long_new(value: BigInt) -> PyObjectRef {
    w_long_from_raw(alloc_bigint_stable(value))
}

/// Create a W_LongObject from an i64 value.
pub fn w_long_from_i64(v: i64) -> PyObjectRef {
    w_long_new(BigInt::from(v))
}

/// Box a bigint constant into a heap Python int object.
pub fn box_bigint_constant(value: &BigInt) -> PyObjectRef {
    w_long_new(value.clone())
}

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
    use malachite_bigint::Sign;
    unsafe { w_long_get_value(obj).sign() == Sign::NoSign }
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

/// `BigInt::to_u64().is_some()` on a borrowed BigInt payload. Scalar half of
/// the `BigInt::to_u64()` split so the two-phase rtyper never has to model an
/// `Option<u64>` ABI. Companion of [`jit_bigint_to_u64_value`].
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_u64_fits(num: &BigInt) -> i64 {
    use num_traits::ToPrimitive;
    num.to_u64().is_some() as i64
}

/// `BigInt::to_u64()` on a borrowed BigInt payload. Callers must first check
/// [`jit_bigint_to_u64_fits`]; a `None` here means that guard was violated.
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_u64_value(num: &BigInt) -> u64 {
    use num_traits::ToPrimitive;
    num.to_u64().unwrap_or_else(|| {
        panic!("jit_bigint_to_u64_value: BigInt exceeds u64 range - fits guard violated")
    })
}

/// `rbigint.sign` / sign-digit use (`rpython/rlib/rbigint.py`) on a borrowed
/// BigInt payload. Returns the scalar signum (-1, 0, +1) so the two-phase
/// rtyper never has to model malachite's `Sign` enum ABI.
#[majit_macros::dont_look_inside]
pub fn jit_bigint_sign_i64(num: &BigInt) -> i64 {
    match num.sign() {
        malachite_bigint::Sign::Minus => -1,
        malachite_bigint::Sign::NoSign => 0,
        malachite_bigint::Sign::Plus => 1,
    }
}

/// `rbigint.tofloat()` (`rpython/rlib/rbigint.py:503`) on a borrowed BigInt
/// payload, with the caller's existing overflow sentinel folded into the
/// scalar return.
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_f64_or_inf(num: &BigInt) -> f64 {
    use num_traits::ToPrimitive;
    num.to_f64().unwrap_or(f64::INFINITY)
}

/// `rbigint.tofloat()` (`rpython/rlib/rbigint.py:503`) on a borrowed BigInt
/// payload, preserving callers that intentionally collapse overflow to NaN.
#[majit_macros::dont_look_inside]
pub fn jit_bigint_to_f64_or_nan(num: &BigInt) -> f64 {
    use num_traits::ToPrimitive;
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
/// via `GetfieldGcPure`. Taking the payloads (not the `W_LongObject` wrappers)
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
    unsafe { alloc_bigint_nursery_collecting(&*a + &*b) as i64 }
}

/// `rbigint.sub` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_sub(a: i64, b: i64) -> i64 {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe { alloc_bigint_nursery_collecting(&*a - &*b) as i64 }
}

/// `rbigint.mul` on bare payloads (collecting). See [`jit_bigint_add`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_mul(a: i64, b: i64) -> i64 {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe { alloc_bigint_nursery_collecting(&*a * &*b) as i64 }
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

/// `bigint_result` — wrap the bigint produced by [`jit_w_long_add_raw`] in a
/// Python int, demoting to `W_IntObject` when it fits in i64, otherwise
/// reusing the `*mut BigInt` payload in a fresh `W_LongObject`. This is the
/// `W_LongObject(...)` wrapper allocation that upstream keeps a residual `NEW`
/// outside the elidable `rbigint.add` (the int fast path boxes the same way,
/// via the `dont_look_inside` `jit_w_int_new`). Marked `dont_look_inside`, not
/// elidable, so the wrapper object is never pure-CSE'd and each add yields a
/// distinct boxed result, matching `W_LongObject(op(...))`.
///
/// The i64-range demotion to `W_IntObject` is pyre's two-class `int`
/// representation (small-int fast object + bigint object); PyPy's default
/// `newlong` (`longobject.py:495`, `withsmalllong=False`) keeps a
/// `W_LongObject`. Both denote the same `int` value — this is a representation
/// choice spanning every int path, not specific to this helper.
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_bigint_result_box(num: i64) -> i64 {
    let num = num as *mut BigInt;
    unsafe {
        if jit_bigint_to_i64_fits(&*num) != 0 {
            crate::intobject::w_int_new(jit_bigint_to_i64_value(&*num)) as usize as i64
        } else {
            w_long_from_raw(num) as usize as i64
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
    fn test_jit_bigint_result_box_keeps_long_out_of_range() {
        // Sum out of i64 range boxes as W_LongObject, reusing the payload.
        let a = w_long_new(BigInt::from(i64::MAX));
        let b = w_long_new(BigInt::from(i64::MAX));
        let raw = jit_w_long_add_raw(a as i64, b as i64);
        let r = jit_bigint_result_box(raw) as PyObjectRef;
        unsafe {
            assert!(is_long(r));
            assert_eq!(*w_long_get_value(r), BigInt::from(i64::MAX) * 2);
        }
    }

    #[test]
    fn test_jit_bigint_result_box_demotes_to_int_when_fits() {
        // `bigint_result` parity: a sum that fits in i64 demotes to W_IntObject
        // (so a later GuardClass(LONG_TYPE) on the result correctly side-exits).
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_long_new(BigInt::from(-1) - BigInt::from(i64::MAX));
        let raw = jit_w_long_add_raw(a as i64, b as i64);
        let r = jit_bigint_result_box(raw) as PyObjectRef;
        unsafe {
            assert!(is_int(r));
            assert!(!is_long(r));
            assert_eq!(crate::intobject::w_int_get_value(r), 0);
        }
    }
}
