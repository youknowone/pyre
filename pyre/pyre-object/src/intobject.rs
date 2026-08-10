//! W_IntObject — Python `int` type backed by i64.
//!
//! Uses a fixed i64 representation. BigInt support (like PyPy's
//! `W_LongObject`) will be added later.

use std::sync::LazyLock;

use crate::pyobject::*;

/// Python integer object.
///
/// Layout: `[ob_header: PyObject { ob_type, w_class } | intval: i64]`
/// The JIT reads `intval` via `GetfieldGcI` at `INT_INTVAL_OFFSET`.
#[repr(C)]
pub struct W_IntObject {
    pub ob_header: PyObject,
    pub intval: i64,
}

/// The translated user-subclass layout selected by `typedef.py:174-227`.
/// `W_IntObject` remains the base payload; `MapdictStorageMixin` contributes
/// its fields only to the generated user class.
#[repr(C)]
pub struct W_IntObjectUser {
    pub base: W_IntObject,
    pub map: *const u8,
    pub storage: *mut crate::object_array::ItemsBlock,
}

/// Field offset of `intval` within `W_IntObject`, for JIT field access.
pub const INT_INTVAL_OFFSET: usize = std::mem::offset_of!(W_IntObject, intval);

/// GC type id assigned to `W_IntObject` at JitDriver init time. Held
/// as a constant here (rather than runtime-queried) so the allocation
/// hook can reach it without a back-channel. `pyre-jit/src/eval.rs`
/// asserts the same id is returned by `gc.register_type(...)` so any
/// mismatch panics on startup instead of silently misclassifying the
/// type at collection time.
pub const W_INT_GC_TYPE_ID: u32 = 1;
pub static W_INT_USER_GC_TYPE_ID: crate::lltype::TypeIdCell = crate::lltype::TypeIdCell::auto();
pub const W_INT_USER_OBJECT_SIZE: usize = std::mem::size_of::<W_IntObjectUser>();

// ── Prebuilt-int cache ───────────────────────────────────────────────
//
// `pypy/config/pypyoption.py:206-213` parity:
//   withprebuiltint  default False — prebuild commonly used int objects
//   prebuiltintfrom  default -5    — lowest integer which is prebuilt
//   prebuiltintto    default 100   — highest integer (stop-exclusive)
//
// `pypy/objspace/std/intobject.py:873-897` `setup_prebuilt` / `wrapint`
// consult these three constants. Pyre keeps them as translation-time
// constants since pyre has no live `space.config` surface — the exact
// PyPy default (`WITHPREBUILTINT=false`) is used to keep `wrapint`'s
// allocation behaviour line-by-line with PyPy default.

pub const WITHPREBUILTINT: bool = false;
pub const PREBUILTINTFROM: i64 = -5;
pub const PREBUILTINTTO: i64 = 100;
pub const W_INT_OBJECT_SIZE: usize = std::mem::size_of::<W_IntObject>();

impl crate::lltype::GcType for W_IntObject {
    fn type_id() -> u32 {
        W_INT_GC_TYPE_ID
    }
    const SIZE: usize = W_INT_OBJECT_SIZE;
}

impl crate::lltype::GcType for W_IntObjectUser {
    fn type_id() -> u32 {
        W_INT_USER_GC_TYPE_ID.get()
    }
    const SIZE: usize = W_INT_USER_OBJECT_SIZE;
}

/// `intobject.py:873-880 setup_prebuilt`. Empty when
/// `WITHPREBUILTINT=false`; populated `[PREBUILTINTFROM, PREBUILTINTTO)`
/// (stop-exclusive) when `true`.
struct PrebuiltInts(Vec<W_IntObject>);

// `intobject.py:873-880 setup_prebuilt` publishes exact, immutable ints for
// process lifetime. Wrapping the contiguous allocation as Sync preserves that
// shared upstream owner without making mutable subclass instances globally
// shareable.
unsafe impl Send for PrebuiltInts {}
unsafe impl Sync for PrebuiltInts {}

static SMALL_INTS: LazyLock<PrebuiltInts> = LazyLock::new(|| {
    if !WITHPREBUILTINT {
        return PrebuiltInts(Vec::new());
    }
    PrebuiltInts(
        (PREBUILTINTFROM..PREBUILTINTTO)
            .map(|v| W_IntObject {
                ob_header: PyObject {
                    ob_type: &INT_TYPE as *const PyType,
                    w_class: std::ptr::null_mut(),
                },
                intval: v,
            })
            .collect(),
    )
});

/// `pypy/objspace/std/intobject.py:903-921 wrapint` parity.
///
/// `withprebuiltint=False` (PyPy default) → always allocate fresh,
/// matching upstream `return W_IntObject(x)`. With the flag enabled
/// and `value` inside `[PREBUILTINTFROM, PREBUILTINTTO)` the cache
/// returns the pre-allocated entry; outside the range we allocate
/// (`instantiate(W_IntObject)` upstream).
///
/// Traced, not residualised, and `#[inline]` — `wrapint` carries no
/// `@dont_look_inside` and its own comment reads "this whole function is
/// getting inlined into every caller so keeping the branching to a minimum
/// is a good idea" (intobject.py:908-910). The allocation upstream is
/// `instantiate(W_IntObject)` followed by `w_res.intval = x`
/// (intobject.py:913-920): alloc-then-init, which the rtyper lowers to the
/// `new_with_vtable` + payload `setfield_gc` pair the optimizer can
/// virtualize where the box does not escape. The stack-built
/// `malloc_typed(W_IntObject { .. })` spelling below is the Rust form of the
/// same thing: `fuse_boxing_alloc` (`majit-translate` `model.rs`) rewrites
/// that cluster into exactly that pair, and it does fire here — its walk
/// resolves the header pointers across the block boundary each call ends.
///
/// `bench/synth/list_pop_append` reads the same either way, and its negative
/// control (boundary removed with the fusion reverted) failed to reproduce
/// the regression that motivated the boundary, so that bench does not
/// discriminate this mechanism in either direction. What decides it is the
/// shape: a residual call can never be virtualized and `new_with_vtable`
/// can, and the boundary's own stated blocker — the fusion not resolving a
/// vtable here — is gone.
///
/// The allocation path goes through [`crate::lltype::malloc_typed`]
/// which carries `W_INT_GC_TYPE_ID` +
/// `W_INT_OBJECT_SIZE` via the [`crate::lltype::GcType`] impl above —
/// the Rust analog of `gct_fv_gc_malloc`'s compile-time `c_type_id`
/// / `c_size` (`rpython/memory/gctransform/framework.py:807-811`).
/// `malloc_typed` prepends a `GcHeader` (`alloc_with_gc_header`) but
/// allocates outside the collector's heap, so the box carries a readable
/// type id while staying off the sweep set; future GC integration
/// replaces only that body, this constructor stays unchanged.
#[inline]
pub fn w_int_new(value: i64) -> PyObjectRef {
    // `ll_int_box`: a value that fits the immediate range is returned as a
    // tagged pointer `(value << 1) | 1` with no allocation — the whole point
    // of the representation. Gated on `CAN_BE_TAGGED` (default false), so the
    // heap-box path below is the only live one until enablement. Subclass
    // instances never reach here (they use `w_int_new_unique`), so a tagged
    // result is always an exact builtin `int`.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::fits_tagged(value) {
        return crate::tagged_int::tag_int(value);
    }
    if WITHPREBUILTINT && (PREBUILTINTFROM..PREBUILTINTTO).contains(&value) {
        let idx = (value - PREBUILTINTFROM) as usize;
        return (&SMALL_INTS.0[idx] as *const W_IntObject).cast_mut() as PyObjectRef;
    }
    if crate::gc_interp::enabled() {
        let boxed = w_int_gc_alloc(value);
        if !boxed.is_null() {
            return boxed;
        }
    }
    crate::lltype::malloc_typed(W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        intval: value,
    }) as PyObjectRef
}

/// The collector-heap arm of [`w_int_new`]: the `gct_fv_gc_malloc` bracket
/// (`rpython/memory/gctransform/framework.py`) in its alloc-then-init form —
/// take the block, then write the header and payload into it, rather than
/// copying a stack-built struct over it. Returns null when no collector owns
/// the heap, which is the caller's signal to take the `malloc_typed` arm.
///
/// Residualised (`rlib/jit.py:139 @dont_look_inside`) rather than traced,
/// which [`w_int_new`] is not — a deviation this arm alone carries, because
/// neither spelling of it reaches the trace intact. Writing the fields into
/// the collector's block, as below, lands the header's own `ob_type` slot at
/// offset 0, which the wasm backend does not lower faithfully. Building the
/// struct on the stack and copying it in instead is not the
/// `malloc_typed(%agg)` route `fuse_boxing_alloc` rewrites, so its
/// `SyntheticTransparentCtor` for the `PyObject` header survives lowering as
/// a call to a `symbolic_fnaddr` hash, which a descending sub-jitcode walk
/// cannot record and declines on. The boundary keeps both shapes out of the
/// trace, and costs nothing where the arm is unreachable:
/// `gc_interp::enabled()` is false on the native backends, whose
/// `malloc_typed` arm is fused into a `NewWithVtable`. Drop it once the wasm
/// backend lowers an offset-0 field store faithfully.
///
/// Spelled `*mut PyObject` rather than `PyObjectRef` so the attribute emits
/// its `extern "C"` call trampoline: the macro recognises raw pointers
/// syntactically and declines to emit one for an aliased return type.
#[majit_macros::dont_look_inside]
pub fn w_int_gc_alloc(value: i64) -> *mut PyObject {
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_INT_GC_TYPE_ID, W_INT_OBJECT_SIZE);
    if raw.is_null() {
        return crate::PY_NULL;
    }
    unsafe {
        let p = raw as *mut W_IntObject;
        (*p).ob_header.ob_type = &INT_TYPE as *const PyType;
        (*p).ob_header.w_class = get_instantiate(&INT_TYPE);
        (*p).intval = value;
    }
    raw as PyObjectRef
}

/// Create a W_IntObject bypassing the small-int cache.
///
/// Used for int subclass instances that need unique object identity
/// (so per-object attributes don't collide). This must NEVER return a
/// tagged immediate: tagged ints are value-identical (`1 is 1`), which is
/// exactly the identity a subclass instance must not have. It always
/// allocates a distinct heap box, independent of `CAN_BE_TAGGED`.
pub fn w_int_new_unique(value: i64) -> PyObjectRef {
    let obj = W_IntObject {
        ob_header: PyObject {
            ob_type: &INT_TYPE as *const PyType,
            w_class: get_instantiate(&INT_TYPE),
        },
        intval: value,
    };
    // `intobject.py:_as_subint` creates a subtype through
    // `space.allocate_instance(W_IntObject, w_inttype)`, so the fresh box is
    // an ordinary collector-owned instance (and can enter the finalizer
    // queue when the subtype defines `__del__`).  Unlike exact ephemeral int
    // boxes, a user instance must not use the non-moving old-gen fast path:
    // that path is intentionally long-lived and would prevent the subtype's
    // registered finalizer from observing that the instance became dead.
    crate::lltype::malloc_typed_managed(obj) as PyObjectRef
}

/// Allocate a `W_IntObject` for an `int` subclass instance, on the managed
/// heap so it can be reclaimed.
///
/// [`w_int_new_unique`] is shared with the JIT's tagged-int unboxing paths
/// and keeps their `malloc_typed` allocation; a subclass instance cannot
/// use it, because `register_finalizer` drops anything outside the managed
/// heap, so a boxed instance would never die and its `__del__` would never
/// run. Allocation is unconditional rather than gated on
/// [`crate::gc_interp::enabled`], matching the other subclass allocators
/// ([`crate::bytesobject::w_bytes_subclass_from_bytes`],
/// `w_str_subclass_from_wtf8`).
pub fn w_int_subclass_new(value: i64) -> PyObjectRef {
    let obj = W_IntObjectUser {
        base: W_IntObject {
            ob_header: PyObject {
                ob_type: &INT_TYPE as *const PyType,
                w_class: get_instantiate(&INT_TYPE),
            },
            intval: value,
        },
        map: std::ptr::null(),
        storage: std::ptr::null_mut(),
    };
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(
        W_INT_USER_GC_TYPE_ID.get(),
        std::mem::size_of::<W_IntObjectUser>(),
    );
    if raw.is_null() {
        crate::lltype::malloc_typed(obj) as PyObjectRef
    } else {
        unsafe {
            std::ptr::write(raw as *mut W_IntObjectUser, obj);
            raw as PyObjectRef
        }
    }
}

/// Return the address of INT_TYPE for JIT type-id validation.
#[inline]
pub fn w_int_type_id() -> usize {
    &INT_TYPE as *const PyType as usize
}

/// Extract the i64 value from a known W_IntObject pointer.
///
/// # Safety
/// `obj` must point to a valid `W_IntObject`.
#[inline]
pub unsafe fn w_int_get_value(obj: PyObjectRef) -> i64 {
    // `ll_unboxed_to_int`: a tagged immediate carries its value in the
    // pointer bits (`>> 1`). The `crate::tagged_int::CAN_BE_TAGGED`
    // static (default `false`) short-circuits the check away, so the
    // heap-box read below is the only live path until enablement.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return crate::tagged_int::untag_int(obj);
    }
    unsafe { (*(obj as *const W_IntObject)).intval }
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_w_int_new(value: i64) -> i64 {
    w_int_new(value) as i64
}

/// True iff `value` falls inside the prebuilt-int cache range AND
/// the cache is enabled. Mirrors PyPy's `wrapint` in-range branch
/// (`intobject.py:891-895`).
#[inline]
pub fn w_int_small_cached(value: i64) -> bool {
    WITHPREBUILTINT && (PREBUILTINTFROM..PREBUILTINTTO).contains(&value)
}

#[inline]
pub fn w_int_small_cache_base_ptr() -> PyObjectRef {
    SMALL_INTS.0.as_ptr().cast_mut() as PyObjectRef
}

/// `intobject.py:516 _bit_count` parity — population count of an i64.
///
/// `@jit.elidable` (`rlib/jit.py:13`): deterministic, no allocation,
/// no raise → `EF_ELIDABLE_CANNOT_RAISE` (`call.py:299`).
///
/// Line-by-line port of the upstream RPython loop; `i64::MIN` is
/// special-cased because `-i64::MIN` overflows in two's complement,
/// matching `val == -sys.maxint - 1` in the upstream guard.
#[majit_macros::elidable_cannot_raise]
pub fn int_bit_count(val: i64) -> i64 {
    if val == i64::MIN {
        return 1;
    }
    let mut val = val;
    if val < 0 {
        val = -val;
    }
    let mut count: i64 = 0;
    while val != 0 {
        count += val & 1;
        val >>= 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn exact_builtin_layouts_do_not_include_mapdict_storage() {
        use crate::boolobject::W_BoolObject;
        use crate::tupleobject::W_TupleObject;
        use crate::unicodeobject::{UNICODE_VALUE_OFFSET, W_UnicodeObject};

        assert_eq!(std::mem::size_of::<W_IntObject>(), 24);
        assert_eq!(std::mem::size_of::<W_BoolObject>(), 24);
        assert_eq!(std::mem::size_of::<W_UnicodeObject>(), 64);
        assert_eq!(std::mem::size_of::<W_TupleObject>(), 40);
        assert_eq!(INT_INTVAL_OFFSET, 16);
        assert_eq!(UNICODE_VALUE_OFFSET, 16);
        assert_eq!(std::mem::offset_of!(W_TupleObject, hash), 16);
        assert_eq!(std::mem::offset_of!(W_TupleObject, wrappeditems), 24);
        assert_eq!(std::mem::offset_of!(W_TupleObject, w_dict), 32);
    }

    #[test]
    fn test_int_create_and_read() {
        let obj = w_int_new(42);
        unsafe {
            assert!(is_int(obj));
            assert!(!is_bool(obj));
            assert_eq!(w_int_get_value(obj), 42);
        }
    }

    #[test]
    fn test_int_negative() {
        let obj = w_int_new(-7);
        unsafe {
            assert_eq!(w_int_get_value(obj), -7);
        }
    }

    #[test]
    fn test_int_field_offset() {
        assert_eq!(INT_INTVAL_OFFSET, 16);
    }

    /// `intobject.py:884 wrapint` parity: with `withprebuiltint=False`
    /// (PyPy default, mirrored by `WITHPREBUILTINT=false`) every call
    /// allocates a fresh `W_IntObject`, so identity is per-call.
    #[test]
    fn test_w_int_new_identity_matches_config() {
        let a = w_int_new(42);
        let b = w_int_new(42);
        if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::fits_tagged(42) {
            assert_eq!(a, b, "tagged immediates are value-identical (1 is 1)");
        } else if WITHPREBUILTINT && (PREBUILTINTFROM..PREBUILTINTTO).contains(&42) {
            assert_eq!(a, b, "in-cache values share the prebuilt pointer");
        } else {
            assert_ne!(a, b, "no cache: each wrapint allocates fresh");
        }
        unsafe {
            assert_eq!(w_int_get_value(a), 42);
            assert_eq!(w_int_get_value(b), 42);
        }
    }

    /// `intobject.py:891-895 wrapint` in-cache branch — only meaningful
    /// when `WITHPREBUILTINT=true`. Skipped otherwise (no prebuilt
    /// pool means there is no boundary to test).
    #[test]
    fn test_prebuilt_cache_boundary() {
        if !WITHPREBUILTINT {
            return;
        }
        let low = w_int_new(PREBUILTINTFROM);
        let high = w_int_new(PREBUILTINTTO - 1);
        unsafe {
            assert_eq!(w_int_get_value(low), PREBUILTINTFROM);
            assert_eq!(w_int_get_value(high), PREBUILTINTTO - 1);
        }
        assert_eq!(low, w_int_new(PREBUILTINTFROM));
        assert_eq!(high, w_int_new(PREBUILTINTTO - 1));
    }

    #[test]
    fn test_outside_cache_allocates_fresh() {
        let a = w_int_new(1000);
        let b = w_int_new(1000);
        if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::fits_tagged(1000) {
            assert_eq!(a, b, "tagged immediates are value-identical");
        } else {
            assert_ne!(a, b, "out-of-cache ints allocate fresh per wrapint");
        }
        unsafe {
            assert_eq!(w_int_get_value(a), 1000);
            assert_eq!(w_int_get_value(b), 1000);
        }
    }

    #[test]
    fn test_w_int_gc_type_id_matches_descr() {
        // Guard against drift between the constant colocated with
        // `W_IntObject` and the id that `pyre-jit/src/eval.rs`
        // asserts at JitDriver init. See `descr.rs` re-export.
        assert_eq!(W_INT_GC_TYPE_ID, 1);
    }

    /// `intobject.py:516 _bit_count` parity — verifies the popcount
    /// against `i64::count_ones`, including the `i64::MIN` sentinel
    /// which the upstream short-circuits to `1`.
    #[test]
    fn test_int_bit_count_matches_intrinsic_popcount() {
        for &v in &[
            0i64,
            1,
            -1,
            42,
            -42,
            i64::MAX,
            i64::MAX - 1,
            // i64::MIN is the upstream `-sys.maxint - 1` guard: the
            // bit_count of i64::MIN in two's complement is 1.
            i64::MIN,
        ] {
            let abs = if v == i64::MIN { i64::MAX } else { v.abs() };
            let expected = if v == i64::MIN {
                1
            } else {
                abs.count_ones() as i64
            };
            assert_eq!(
                int_bit_count(v),
                expected,
                "int_bit_count({v}) must match the upstream popcount",
            );
        }
    }

    /// `#[elidable_cannot_raise]` macro on a production helper must
    /// emit `INT_ELIDABLE_CANNOT_RAISE = 19` (`call_policy_byte.rs:96`)
    /// and a non-null trace_target / concrete_target trampoline pair.
    #[test]
    fn test_int_bit_count_advertises_int_elidable_cannot_raise_byte() {
        let (policy, _, trace_target, concrete_target, _, _) = __majit_call_policy_int_bit_count();
        assert_eq!(policy, 19u8);
        assert!(!trace_target.is_null());
        assert!(!concrete_target.is_null());
    }
}
